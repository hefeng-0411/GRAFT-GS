"""Run fail-fast environment, dataset, and reference validation on the server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from validate_environment import audit_environment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_DATASET = Path(
    "/mnt/sda2/hef/Base/dataset"
)
EXPECTED_MESHFLEET_SCHEMA = "meshfleet-trellis-object-v2"
EXPECTED_DISCOVERY_REQUIRED_MODALITIES = ("renders", "latents", "mesh_normalized")
ALLOWED_REFERENCE_SKIP_REASONS = (
    "launch with torchrun",
    "set GRAFT_GS_REAL_IMAGE_DIR on the server",
)


def _run(command: list[str], environment: dict[str, str] | None = None) -> dict[str, object]:
    start = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    return {
        "command": command,
        "returncode": process.returncode,
        "seconds": time.perf_counter() - start,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def _accelerator_contract_errors(details: dict[str, object]) -> list[str]:
    """Validate the native A800 reference precision/runtime boundary."""

    errors: list[str] = []
    if details.get("cuda_available") is not True:
        errors.append("PyTorch CUDA is unavailable")
    if details.get("torch_cuda") != "11.8":
        errors.append("PyTorch was not built against the pinned CUDA 11.8 runtime")
    if details.get("bf16_supported") is not True:
        errors.append("the visible accelerator does not report native BF16 support")
    devices = details.get("devices")
    if not isinstance(devices, list) or not devices:
        errors.append("no CUDA device is visible to the reference process")
    elif any(
        not isinstance(device, dict) or "A800" not in str(device.get("name", "")).upper()
        for device in devices
    ):
        errors.append("every visible reference device must be an NVIDIA A800")
    return errors


def _probe_accelerator() -> dict[str, object]:
    program = """
import json
import torch

devices = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        })
details = {
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    "visible_device_count": torch.cuda.device_count(),
    "devices": devices,
}
print(json.dumps(details, sort_keys=True))
"""
    process = _run([sys.executable, "-c", program])
    details: dict[str, object] = {}
    parse_error = None
    if process["returncode"] == 0:
        try:
            details = json.loads(str(process["stdout"]))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            parse_error = str(error)
    errors = _accelerator_contract_errors(details)
    if process["returncode"] != 0:
        errors.insert(0, "accelerator probe subprocess failed")
    if parse_error is not None:
        errors.insert(0, f"accelerator probe emitted invalid JSON: {parse_error}")
    return {"valid": not errors, "details": details, "errors": errors, "process": process}


def _dataset_root(argument: Path | None) -> Path | None:
    if argument is not None:
        return argument.resolve()
    environment = os.environ.get("GRAFT_GS_MESHFLEET_ROOT")
    if environment:
        return Path(environment).resolve()
    return DEFAULT_SERVER_DATASET if DEFAULT_SERVER_DATASET.is_dir() else None


def _skip_reasons(stderr: str) -> list[str]:
    return re.findall(r"\.\.\. skipped ['\"](.+?)['\"]", stderr)


def _write_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf8")


def _load_object_id_catalog(path: Path) -> tuple[str, ...]:
    identifiers: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        value = raw_line.split("#", 1)[0].strip()
        if not value:
            continue
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"invalid object ID at {path}:{line_number}")
        if value in seen:
            raise ValueError(f"duplicate object ID at {path}:{line_number}: {value}")
        seen.add(value)
        identifiers.append(value)
    if not identifiers:
        raise ValueError(f"object ID catalog is empty: {path}")
    return tuple(identifiers)


def _object_id_digest(identifiers: tuple[str, ...]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(identifiers)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _inspect_manifest_contract(
    manifest: Path,
    dataset_root: Path,
    expected_object_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Audit manifest identity without importing model or dataset dependencies."""

    errors: list[str] = []
    summary_path = manifest.with_suffix(manifest.suffix + ".summary.json")
    if not manifest.is_file():
        errors.append("manifest is missing")
    if not summary_path.is_file():
        errors.append("manifest summary is missing")
    summary: dict[str, object] = {}
    if errors:
        return {"valid": False, "errors": errors, "summary": summary}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "valid": False,
            "errors": [f"manifest summary is unreadable: {error}"],
            "summary": {},
        }
    if not isinstance(summary, dict):
        return {
            "valid": False,
            "errors": ["manifest summary is not a JSON object"],
            "summary": {},
        }
    try:
        recorded_root = Path(str(summary["dataset_root"])).resolve()
    except (KeyError, TypeError, ValueError):
        recorded_root = None
    if recorded_root != dataset_root:
        errors.append("manifest summary belongs to a different dataset root")
    if summary.get("schema") != EXPECTED_MESHFLEET_SCHEMA:
        errors.append("manifest schema does not match the loader contract")
    discovery_policy = summary.get("discovery_policy")
    if not isinstance(discovery_policy, dict):
        errors.append("manifest summary has no modality discovery policy")
    else:
        if discovery_policy.get("layout") != "modality-centric":
            errors.append("manifest was not built from the modality-centric layout")
        if tuple(discovery_policy.get("required_modalities", ())) != (
            EXPECTED_DISCOVERY_REQUIRED_MODALITIES
        ):
            errors.append("manifest required-modality intersection differs")

    manifest_count = 0
    manifest_ids: list[str] = []
    try:
        with manifest.open("r", encoding="utf8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                manifest_count += 1
                item = json.loads(line)
                if not isinstance(item, dict):
                    errors.append(f"manifest line {line_number} is not a JSON object")
                    continue
                object_id = item.get("object_id")
                if isinstance(object_id, str) and re.fullmatch(
                    r"[0-9a-f]{64}", object_id
                ) is not None:
                    manifest_ids.append(object_id)
                else:
                    errors.append(f"manifest line {line_number} has an invalid object ID")
                if item.get("split") not in {"train", "test"}:
                    errors.append(f"manifest line {line_number} has an invalid split")
                discovery = item.get("discovery")
                if not isinstance(discovery, dict):
                    errors.append(
                        f"manifest line {line_number} has no modality discovery contract"
                    )
                else:
                    available = discovery.get("available_modalities", ())
                    missing = [
                        name
                        for name in EXPECTED_DISCOVERY_REQUIRED_MODALITIES
                        if name not in available
                    ]
                    if missing:
                        errors.append(
                            f"manifest line {line_number} violates required-modality intersection"
                        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"manifest is unreadable: {error}")
    if summary.get("record_count") != manifest_count:
        errors.append("manifest record count does not match its summary")
    split_counts = summary.get("split_counts")
    if not isinstance(split_counts, dict) or sum(
        int(value) for value in split_counts.values()
    ) != manifest_count:
        errors.append("manifest split counts do not match its records")
    if manifest_count == 0:
        errors.append("dynamic dataset discovery produced an empty manifest")
    duplicated_ids = sorted(
        object_id for object_id in set(manifest_ids)
        if manifest_ids.count(object_id) > 1
    )
    if duplicated_ids:
        errors.append("manifest contains duplicate object IDs across train/test")
    if summary.get("discovered_object_ids_sha256") != _object_id_digest(
        tuple(manifest_ids)
    ):
        errors.append("manifest discovered-object digest differs")
    if expected_object_ids is not None:
        expected = set(expected_object_ids)
        actual = set(manifest_ids)
        catalog = summary.get("object_id_catalog", {})
        expected_missing = sorted(expected - actual)
        if not isinstance(catalog, dict) or catalog.get("enabled") is not True:
            errors.append("manifest summary has no active object ID catalog")
        else:
            if catalog.get("count") != len(expected_object_ids):
                errors.append("manifest object ID catalog count differs")
            if catalog.get("sha256") != _object_id_digest(expected_object_ids):
                errors.append("manifest object ID catalog digest differs")
            if catalog.get("discovered_count") != len(actual):
                errors.append("manifest discovered ID count differs")
            if catalog.get("missing_ids") != expected_missing:
                errors.append("manifest missing-ID inventory differs")
        if actual - expected:
            errors.append("manifest contains IDs outside the configured catalog")
    return {
        "valid": not errors,
        "errors": errors,
        "summary": summary,
        "record_count": manifest_count,
        "discovered_object_count": len(set(manifest_ids)),
    }


def _manifest_requires_rebuild(force: bool, audit: dict[str, object]) -> bool:
    """Make the reuse/rebuild decision explicit and independently testable."""

    return force or audit.get("valid") is not True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/validation.json"))
    parser.add_argument("--requirements", type=Path, default=ROOT / "requirements.txt")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--object-id-file", type=Path)
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="re-audit the full remote dataset even when the manifest already exists",
    )
    args = parser.parse_args()
    object_ids = (
        _load_object_id_catalog(args.object_id_file.resolve())
        if args.object_id_file is not None
        else None
    )
    overall_start = time.perf_counter()
    record: dict[str, object] = {
        "command": [sys.executable, *sys.argv],
        "environment": audit_environment(args.requirements),
    }
    output_path = args.output.resolve()
    environment_record = record["environment"]
    if not isinstance(environment_record, dict) or not environment_record["valid"]:
        record["returncode"] = 2
        record["seconds"] = time.perf_counter() - overall_start
        record["remediation"] = (
            f"{sys.executable} -m pip install --no-deps --upgrade -r "
            f"{Path(args.requirements).resolve()}"
        )
        _write_record(output_path, record)
        print(json.dumps(record, indent=2, sort_keys=True))
        raise SystemExit(2)

    pip_check = _run([sys.executable, "-m", "pip", "check"])
    record["pip_check"] = pip_check
    if pip_check["returncode"] != 0:
        record["returncode"] = 2
        record["seconds"] = time.perf_counter() - overall_start
        _write_record(output_path, record)
        sys.stdout.write(str(pip_check["stdout"]))
        sys.stderr.write(str(pip_check["stderr"]))
        raise SystemExit(2)

    accelerator = _probe_accelerator()
    record["accelerator"] = accelerator
    if not accelerator["valid"]:
        record["returncode"] = 2
        record["seconds"] = time.perf_counter() - overall_start
        _write_record(output_path, record)
        print(json.dumps(record, indent=2, sort_keys=True))
        raise SystemExit(2)

    dataset_root = _dataset_root(args.dataset_root)
    if dataset_root is None or not dataset_root.is_dir():
        record["dataset"] = {
            "valid": False,
            "root": str(dataset_root) if dataset_root is not None else None,
            "reason": "server MeshFleet root is unavailable",
        }
        record["returncode"] = 2
        record["seconds"] = time.perf_counter() - overall_start
        _write_record(output_path, record)
        print(json.dumps(record, indent=2, sort_keys=True))
        raise SystemExit(2)
    manifest = (
        args.manifest.resolve()
        if args.manifest is not None
        else Path(
            os.environ.get(
                "GRAFT_GS_MESHFLEET_MANIFEST",
                str(output_path.parent / "meshfleet_server.jsonl"),
            )
        ).resolve()
    )
    manifest_build = None
    manifest_audit = _inspect_manifest_contract(manifest, dataset_root, object_ids)
    if _manifest_requires_rebuild(args.rebuild_manifest, manifest_audit):
        manifest_command = [
            sys.executable,
            str(ROOT / "scripts" / "build_meshfleet_manifest.py"),
            str(dataset_root),
            str(manifest),
        ]
        if args.object_id_file is not None:
            manifest_command.extend(
                ("--object-id-file", str(args.object_id_file.resolve()))
            )
        manifest_build = _run(manifest_command)
        if manifest_build["returncode"] != 0:
            record["manifest_build"] = manifest_build
            record["returncode"] = int(manifest_build["returncode"])
            record["seconds"] = time.perf_counter() - overall_start
            _write_record(output_path, record)
            sys.stdout.write(str(manifest_build["stdout"]))
            sys.stderr.write(str(manifest_build["stderr"]))
            raise SystemExit(int(manifest_build["returncode"]))
        manifest_audit = _inspect_manifest_contract(manifest, dataset_root, object_ids)
    if not manifest_audit["valid"]:
        record["dataset"] = {
            "valid": False,
            "root": str(dataset_root),
            "manifest": str(manifest),
            "errors": manifest_audit["errors"],
        }
        record["returncode"] = 2
        record["seconds"] = time.perf_counter() - overall_start
        _write_record(output_path, record)
        print(json.dumps(record, indent=2, sort_keys=True))
        raise SystemExit(2)
    summary = manifest_audit["summary"]
    record["dataset"] = {
        "valid": True,
        "root": str(dataset_root),
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "summary": summary,
        "discovered_object_count": manifest_audit["discovered_object_count"],
        "object_id_catalog": (
            {
                "path": str(args.object_id_file.resolve()),
                "count": len(object_ids),
                "sha256": _object_id_digest(object_ids),
            }
            if object_ids is not None
            else None
        ),
        "build": manifest_build,
    }

    test_environment = os.environ.copy()
    test_environment["GRAFT_GS_MESHFLEET_ROOT"] = str(dataset_root)
    test_environment["GRAFT_GS_MESHFLEET_MANIFEST"] = str(manifest)
    test_command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-v",
    ]
    tests = _run(test_command, test_environment)
    skips = _skip_reasons(str(tests["stderr"]))
    unexpected_skips = [
        reason
        for reason in skips
        if not any(allowed in reason for allowed in ALLOWED_REFERENCE_SKIP_REASONS)
    ]
    tests["skip_reasons"] = skips
    tests["unexpected_skip_reasons"] = unexpected_skips
    record["tests"] = tests
    returncode = int(tests["returncode"])
    if returncode == 0 and unexpected_skips:
        returncode = 3
    record["returncode"] = returncode
    record["seconds"] = time.perf_counter() - overall_start
    _write_record(output_path, record)
    sys.stdout.write(str(tests["stdout"]))
    sys.stderr.write(str(tests["stderr"]))
    if unexpected_skips:
        sys.stderr.write(
            "\nUnexpected reference-suite skips: "
            + json.dumps(unexpected_skips, indent=2)
            + "\n"
        )
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()

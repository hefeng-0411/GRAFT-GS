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
    "/mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97"
)
EXPECTED_MESHFLEET_SCHEMA = "meshfleet-trellis-object-v2"
CANONICAL_OBJECT_ID = "17a53839ae5da04c75ea21335d4bdc8ddc26b45f7bb9d0e18f5afaa397e43a17"
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


def _inspect_manifest_contract(manifest: Path, dataset_root: Path) -> dict[str, object]:
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

    manifest_count = 0
    canonical_splits: list[object] = []
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
                if item.get("object_id") == CANONICAL_OBJECT_ID:
                    canonical_splits.append(item.get("split"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"manifest is unreadable: {error}")
    if summary.get("record_count") != manifest_count:
        errors.append("manifest record count does not match its summary")
    if len(canonical_splits) != 1:
        errors.append(
            f"canonical object occurs {len(canonical_splits)} times instead of exactly once"
        )
    return {
        "valid": not errors,
        "errors": errors,
        "summary": summary,
        "record_count": manifest_count,
        "canonical_split": canonical_splits[0] if len(canonical_splits) == 1 else None,
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
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="re-audit the full remote dataset even when the manifest already exists",
    )
    args = parser.parse_args()
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
    for split in ("train", "test"):
        if not (dataset_root / split).is_dir():
            raise FileNotFoundError(f"remote dataset is missing {split!r}: {dataset_root / split}")

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
    manifest_audit = _inspect_manifest_contract(manifest, dataset_root)
    if _manifest_requires_rebuild(args.rebuild_manifest, manifest_audit):
        manifest_build = _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_meshfleet_manifest.py"),
                str(dataset_root),
                str(manifest),
            ]
        )
        if manifest_build["returncode"] != 0:
            record["manifest_build"] = manifest_build
            record["returncode"] = int(manifest_build["returncode"])
            record["seconds"] = time.perf_counter() - overall_start
            _write_record(output_path, record)
            sys.stdout.write(str(manifest_build["stdout"]))
            sys.stderr.write(str(manifest_build["stderr"]))
            raise SystemExit(int(manifest_build["returncode"]))
        manifest_audit = _inspect_manifest_contract(manifest, dataset_root)
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
        "canonical_split": manifest_audit["canonical_split"],
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

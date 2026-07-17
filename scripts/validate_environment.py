"""Audit the active Python environment against the exact pinned requirements."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
from typing import Iterable, Mapping


def canonical_distribution_name(value: str) -> str:
    """PEP-503-compatible distribution key without importing packaging."""

    return re.sub(r"[-_.]+", "-", value).lower()


def parse_pinned_requirements(path: str | Path) -> dict[str, dict[str, str]]:
    """Parse a requirements file whose active entries must all use exact ``==`` pins."""

    path = Path(path)
    result: dict[str, dict[str, str]] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r", "--requirement", "-c", "--constraint")):
            raise ValueError(
                f"nested requirement files are not permitted in the exact server contract: line {line_number}"
            )
        if "==" not in line or any(
            token in line for token in ("!=", "~=", ">=", "<=", " @ ", ";")
        ):
            raise ValueError(
                f"requirement line {line_number} is not one unconditional exact pin: {raw_line!r}"
            )
        name, version = (part.strip() for part in line.split("==", 1))
        if not name or not version or "[" in name:
            raise ValueError(f"unsupported exact requirement at line {line_number}: {raw_line!r}")
        key = canonical_distribution_name(name)
        previous = result.get(key)
        if previous is not None and previous["version"] != version:
            raise ValueError(f"conflicting pins for {name!r}")
        result[key] = {"name": name, "version": version}
    if not result:
        raise ValueError("requirements contract is empty")
    return result


def installed_distribution_versions(
    distributions: Iterable[importlib.metadata.Distribution] | None = None,
) -> dict[str, list[str]]:
    versions: dict[str, list[str]] = {}
    source = importlib.metadata.distributions() if distributions is None else distributions
    for distribution in source:
        name = distribution.metadata.get("Name")
        if not name:
            continue
        key = canonical_distribution_name(name)
        versions.setdefault(key, []).append(str(distribution.version))
    return {key: sorted(set(value)) for key, value in versions.items()}


def compare_environment(
    required: Mapping[str, Mapping[str, str]],
    installed: Mapping[str, list[str]],
) -> dict[str, object]:
    missing: list[dict[str, str]] = []
    mismatched: list[dict[str, object]] = []
    matched = 0
    for key in sorted(required):
        specification = required[key]
        versions = installed.get(key)
        if not versions:
            missing.append(dict(specification))
        elif versions != [specification["version"]]:
            mismatched.append(
                {
                    "name": specification["name"],
                    "required": specification["version"],
                    "installed": versions,
                }
            )
        else:
            matched += 1
    return {
        "valid": not missing and not mismatched,
        "required_count": len(required),
        "matched_count": matched,
        "missing": missing,
        "mismatched": mismatched,
    }


def audit_environment(requirements: str | Path) -> dict[str, object]:
    requirements = Path(requirements).resolve()
    required = parse_pinned_requirements(requirements)
    comparison = compare_environment(required, installed_distribution_versions())
    return {
        **comparison,
        "requirements": str(requirements),
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--requirements",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "requirements.txt",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = audit_environment(args.requirements)
    rendered = json.dumps(record, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf8")
    print(rendered)
    raise SystemExit(0 if record["valid"] else 2)


if __name__ == "__main__":
    main()

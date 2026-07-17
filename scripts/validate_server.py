"""Run the complete reference validation suite and emit a machine-readable log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/validation.json"))
    args = parser.parse_args()
    start = time.perf_counter()
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    process = subprocess.run(command, text=True, capture_output=True)
    record = {
        "command": command,
        "returncode": process.returncode,
        "seconds": time.perf_counter() - start,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf8")
    sys.stdout.write(process.stdout)
    sys.stderr.write(process.stderr)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()


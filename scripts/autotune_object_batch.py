"""Select a safe, high-throughput object batch on an explicit GPU set.

Every candidate runs in a fresh torchrun process group. An allocation failure,
mutated optimizer, allocator cache, or stochastic state from a probe therefore
cannot leak into the launched training job.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
OOM_MARKERS = (
    "torch.outofmemoryerror",
    "cuda out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "failed to allocate cuda",
    "cuda malloc failed",
)


def _validate_candidates(values: Sequence[int]) -> tuple[int, ...]:
    candidates = tuple(int(value) for value in values)
    if not candidates or any(value < 1 for value in candidates):
        raise ValueError("batch candidates must be positive")
    if tuple(sorted(set(candidates))) != candidates:
        raise ValueError("batch candidates must be strictly increasing")
    return candidates


def _gpu_environment(gpus: str | None) -> tuple[dict[str, str], str]:
    selected = gpus if gpus is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if selected is None or not selected.strip():
        raise ValueError(
            "--gpus or an existing CUDA_VISIBLE_DEVICES value must explicitly "
            "name the available devices"
        )
    selected = selected.strip()
    if re.fullmatch(r"[A-Za-z0-9:._,/\\-]+", selected) is None:
        raise ValueError("GPU identifiers contain unsupported characters")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = selected
    environment["PYTHONUNBUFFERED"] = "1"
    return environment, selected


def _cuda_inventory(
    python: str,
    environment: dict[str, str],
) -> list[dict[str, object]]:
    source = (
        "import json,torch\n"
        "rows=[]\n"
        "for i in range(torch.cuda.device_count()):\n"
        " free,total=torch.cuda.mem_get_info(i)\n"
        " rows.append({'logical_device':i,'name':torch.cuda.get_device_name(i),"
        "'free_bytes':int(free),'total_bytes':int(total),"
        "'free_fraction':float(free/total)})\n"
        "print(json.dumps(rows,sort_keys=True))\n"
    )
    completed = subprocess.run(
        [python, "-c", source],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=ROOT,
    )
    inventory = json.loads(completed.stdout)
    if not isinstance(inventory, list) or not inventory:
        raise RuntimeError("the explicit GPU set exposes no usable CUDA device")
    return inventory


def _run_and_tee(
    command: Sequence[str],
    log_path: Path,
    environment: dict[str, str],
) -> tuple[int, bool]:
    with log_path.open("w", encoding="utf8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        oom = False
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            normalized = line.lower()
            oom = oom or any(marker in normalized for marker in OOM_MARKERS)
        return process.wait(), oom


def _torchrun_command(
    python: str,
    world_size: int,
    train_arguments: Sequence[str],
) -> list[str]:
    return [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
        str(ROOT / "scripts" / "train_a800.py"),
        *train_arguments,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus")
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--python",
        default=os.environ.get("GRAFT_GS_PYTHON", sys.executable),
    )
    parser.add_argument("--maximum-allocated-fraction", type=float, default=0.85)
    parser.add_argument("--maximum-reserved-fraction", type=float, default=0.88)
    parser.add_argument("--minimum-driver-free-fraction", type=float, default=0.08)
    parser.add_argument(
        "--minimum-initial-driver-free-fraction",
        type=float,
        default=0.90,
    )
    parser.add_argument("--throughput-fraction", type=float, default=0.97)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="launch the full training command with the selected batch",
    )
    parser.add_argument(
        "train_arguments",
        nargs=argparse.REMAINDER,
        help="arguments for scripts/train_a800.py, following --",
    )
    args = parser.parse_args()
    candidates = _validate_candidates(args.candidates)
    for name in (
        "maximum_allocated_fraction",
        "maximum_reserved_fraction",
        "minimum_driver_free_fraction",
        "minimum_initial_driver_free_fraction",
        "throughput_fraction",
    ):
        value = float(getattr(args, name))
        if not 0 < value < 1:
            raise ValueError(f"{name.replace('_', '-')} must lie in (0,1)")
    train_arguments = list(args.train_arguments)
    if train_arguments and train_arguments[0] == "--":
        train_arguments.pop(0)
    if not train_arguments:
        raise ValueError("train_a800.py arguments must follow --")
    forbidden = {"--object-batch-size", "--batch-probe"}
    if forbidden.intersection(train_arguments):
        raise ValueError(
            "the autotuner owns --object-batch-size and --batch-probe"
        )
    if not Path(args.python).is_file():
        raise FileNotFoundError(args.python)
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise FileExistsError("autotune output directory must be fresh")
    args.output.mkdir(parents=True, exist_ok=True)

    environment, selected_gpus = _gpu_environment(args.gpus)
    subprocess.run(
        [
            args.python,
            str(ROOT / "scripts" / "validate_environment.py"),
            "--requirements",
            str(ROOT / "requirements.txt"),
            "--output",
            str(args.output / "environment.json"),
        ],
        check=True,
        cwd=ROOT,
        env=environment,
    )
    inventory = _cuda_inventory(args.python, environment)
    occupied = [
        row
        for row in inventory
        if float(row["free_fraction"])
        < args.minimum_initial_driver_free_fraction
    ]
    inventory_path = args.output / "initial_cuda_memory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "cuda_visible_devices": selected_gpus,
                "devices": inventory,
                "minimum_initial_driver_free_fraction": (
                    args.minimum_initial_driver_free_fraction
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf8",
    )
    if occupied:
        raise RuntimeError(
            f"an explicitly selected GPU is already occupied; inspect {inventory_path}"
        )

    world_size = len(inventory)
    runs: list[dict[str, object]] = []
    safe: list[dict[str, object]] = []
    for batch_size in candidates:
        run_directory = args.output / f"batch-{batch_size:03d}"
        run_directory.mkdir()
        probe_path = run_directory / "probe.json"
        command = _torchrun_command(
            args.python,
            world_size,
            [
                *train_arguments,
                "--object-batch-size",
                str(batch_size),
                "--batch-probe",
                str(probe_path),
                "--output",
                str(run_directory / "training"),
            ],
        )
        started = time.perf_counter()
        return_code, oom = _run_and_tee(
            command,
            run_directory / "run.log",
            environment,
        )
        probe = (
            json.loads(probe_path.read_text(encoding="utf8"))
            if return_code == 0 and probe_path.is_file()
            else None
        )
        reasons: list[str] = []
        if return_code != 0:
            reasons.append(f"probe exited with status {return_code}")
        if oom:
            reasons.append("CUDA allocation failure")
        if isinstance(probe, dict):
            if (
                int(probe["minimum_realized_object_batch_size"])
                != batch_size
                or int(probe["maximum_realized_object_batch_size"])
                != batch_size
            ):
                reasons.append("candidate object batch was not fully realized")
            if (
                float(probe["maximum_peak_allocated_fraction"])
                > args.maximum_allocated_fraction
            ):
                reasons.append("allocated-memory headroom failed")
            if (
                float(probe["maximum_peak_reserved_fraction"])
                > args.maximum_reserved_fraction
            ):
                reasons.append("reserved-memory headroom failed")
            if (
                float(probe["minimum_ending_driver_free_fraction"])
                < args.minimum_driver_free_fraction
            ):
                reasons.append("driver-free-memory headroom failed")
        else:
            reasons.append("probe report is unavailable")
        record = {
            "object_batch_size": batch_size,
            "return_code": return_code,
            "oom": oom,
            "seconds": time.perf_counter() - started,
            "probe": probe,
            "admissible": not reasons,
            "reasons": reasons,
            "command": command,
        }
        runs.append(record)
        if not reasons:
            safe.append(record)

    if not safe:
        summary_path = args.output / "selection.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema": "graft-gs-object-batch-selection-v1",
                    "selected_object_batch_size": None,
                    "runs": runs,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf8",
        )
        raise RuntimeError(f"no batch candidate passed; inspect {summary_path}")

    fastest = max(
        float(record["probe"]["aggregate_objects_per_second"])
        for record in safe
    )
    near_fastest = [
        record
        for record in safe
        if float(record["probe"]["aggregate_objects_per_second"])
        >= args.throughput_fraction * fastest
    ]
    selected = max(
        near_fastest, key=lambda record: int(record["object_batch_size"])
    )
    selected_batch = int(selected["object_batch_size"])
    launch_command = _torchrun_command(
        args.python,
        world_size,
        [
            *train_arguments,
            "--object-batch-size",
            str(selected_batch),
        ],
    )
    selection_path = args.output / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema": "graft-gs-object-batch-selection-v1",
                "cuda_visible_devices": selected_gpus,
                "world_size": world_size,
                "selected_object_batch_size": selected_batch,
                "throughput_fraction": args.throughput_fraction,
                "launch_command": launch_command,
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf8",
    )
    print(
        json.dumps(
            {
                "selection": str(selection_path),
                "selected_object_batch_size": selected_batch,
                "world_size": world_size,
                "aggregate_objects_per_second": selected["probe"][
                    "aggregate_objects_per_second"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.launch:
        raise SystemExit(
            subprocess.run(
                launch_command,
                cwd=ROOT,
                env=environment,
                check=False,
            ).returncode
        )


if __name__ == "__main__":
    main()

"""Measure and select a safe multi-object MeshFleet evaluation batch."""

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


def _run(
    command: Sequence[str],
    environment: dict[str, str],
    log_path: Path,
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


def _torchrun(
    python: str,
    world_size: int,
    arguments: Sequence[str],
) -> list[str]:
    return [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
        str(ROOT / "scripts" / "evaluate_meshfleet.py"),
        *arguments,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus")
    parser.add_argument("--candidates", type=int, nargs="+", default=(1, 2, 4, 8))
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
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("evaluation_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    candidates = tuple(int(value) for value in args.candidates)
    if (
        not candidates
        or any(value < 1 for value in candidates)
        or tuple(sorted(set(candidates))) != candidates
    ):
        raise ValueError("batch candidates must be positive and strictly increasing")
    values = (
        args.maximum_allocated_fraction,
        args.maximum_reserved_fraction,
        args.minimum_driver_free_fraction,
        args.minimum_initial_driver_free_fraction,
        args.throughput_fraction,
    )
    if any(not 0 < value < 1 for value in values):
        raise ValueError("memory and throughput fractions must lie in (0,1)")
    evaluation_arguments = list(args.evaluation_arguments)
    if evaluation_arguments and evaluation_arguments[0] == "--":
        evaluation_arguments.pop(0)
    if len(evaluation_arguments) < 4:
        raise ValueError(
            "evaluation arguments must include DATASET MANIFEST CHECKPOINT OUTPUT"
        )
    forbidden = {
        "--object-batch-size",
        "--maximum-batches-per-rank",
    }
    if forbidden.intersection(evaluation_arguments):
        raise ValueError("the autotuner owns evaluation batch/probe arguments")
    selected_gpus = args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES")
    if selected_gpus is None or not selected_gpus.strip():
        raise ValueError("--gpus or CUDA_VISIBLE_DEVICES is required")
    selected_gpus = selected_gpus.strip()
    if re.fullmatch(r"[A-Za-z0-9:._,/\\-]+", selected_gpus) is None:
        raise ValueError("GPU identifiers contain unsupported characters")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = selected_gpus
    environment["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        [
            args.python,
            "-c",
            (
                "import json,torch\n"
                "rows=[]\n"
                "for i in range(torch.cuda.device_count()):\n"
                " free,total=torch.cuda.mem_get_info(i)\n"
                " rows.append({'logical_device':i,"
                "'name':torch.cuda.get_device_name(i),"
                "'free_bytes':int(free),'total_bytes':int(total),"
                "'free_fraction':float(free/total)})\n"
                "print(json.dumps(rows,sort_keys=True))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=ROOT,
    )
    inventory = json.loads(completed.stdout)
    world_size = len(inventory)
    if world_size < 1:
        raise RuntimeError("the explicit GPU set exposes no CUDA device")
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise FileExistsError("autotune output directory must be fresh")
    args.output.mkdir(parents=True, exist_ok=True)
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
    occupied = [
        row
        for row in inventory
        if float(row["free_fraction"])
        < args.minimum_initial_driver_free_fraction
    ]
    if occupied:
        raise RuntimeError(
            f"an explicitly selected GPU is already occupied; inspect {inventory_path}"
        )

    final_output = evaluation_arguments[3]
    runs: list[dict[str, object]] = []
    admissible: list[dict[str, object]] = []
    for batch_size in candidates:
        run_directory = args.output / f"batch-{batch_size:03d}"
        evaluation_output = run_directory / "evaluation"
        run_directory.mkdir()
        probe_arguments = [
            *evaluation_arguments[:3],
            str(evaluation_output),
            *evaluation_arguments[4:],
            "--object-batch-size",
            str(batch_size),
            "--maximum-batches-per-rank",
            "1",
        ]
        command = _torchrun(args.python, world_size, probe_arguments)
        started = time.perf_counter()
        return_code, oom = _run(
            command, environment, run_directory / "run.log"
        )
        metrics_path = evaluation_output / "metrics.jsonl"
        metrics = []
        if return_code == 0 and metrics_path.is_file():
            metrics = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf8").splitlines()
                if line.strip()
            ]
        ok = [value for value in metrics if value.get("status") == "ok"]
        reasons: list[str] = []
        if return_code != 0:
            reasons.append(f"probe exited with status {return_code}")
        if oom:
            reasons.append("CUDA allocation failure")
        if not ok:
            reasons.append("probe emitted no successful objects")
        if ok and any(
            int(value["object_batch_size"]) != batch_size for value in ok
        ):
            reasons.append("candidate object batch was not fully realized")
        if ok and max(float(value["peak_allocated_fraction"]) for value in ok) > args.maximum_allocated_fraction:
            reasons.append("allocated-memory headroom failed")
        if ok and max(float(value["peak_reserved_fraction"]) for value in ok) > args.maximum_reserved_fraction:
            reasons.append("reserved-memory headroom failed")
        if ok and min(float(value["ending_driver_free_fraction"]) for value in ok) < args.minimum_driver_free_fraction:
            reasons.append("driver-free-memory headroom failed")
        throughput = (
            len(ok)
            / max(float(value["batch_seconds"]) for value in ok)
            if ok
            else 0.0
        )
        record = {
            "object_batch_size": batch_size,
            "return_code": return_code,
            "oom": oom,
            "seconds": time.perf_counter() - started,
            "objects_per_second": throughput,
            "admissible": not reasons,
            "reasons": reasons,
            "metrics": ok,
            "command": command,
        }
        runs.append(record)
        if not reasons:
            admissible.append(record)
    if not admissible:
        selected_batch = None
    else:
        fastest = max(float(value["objects_per_second"]) for value in admissible)
        near = [
            value
            for value in admissible
            if float(value["objects_per_second"])
            >= args.throughput_fraction * fastest
        ]
        selected_batch = max(
            int(value["object_batch_size"]) for value in near
        )
    launch_arguments = [
        *evaluation_arguments[:3],
        final_output,
        *evaluation_arguments[4:],
        "--object-batch-size",
        str(selected_batch),
    ]
    launch_command = (
        _torchrun(args.python, world_size, launch_arguments)
        if selected_batch is not None
        else None
    )
    selection_path = args.output / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema": "graft-gs-evaluation-batch-selection-v1",
                "cuda_visible_devices": selected_gpus,
                "world_size": world_size,
                "selected_object_batch_size": selected_batch,
                "launch_command": launch_command,
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf8",
    )
    if launch_command is None:
        raise RuntimeError(f"no evaluation batch passed; inspect {selection_path}")
    print(
        json.dumps(
            {
                "selection": str(selection_path),
                "selected_object_batch_size": selected_batch,
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

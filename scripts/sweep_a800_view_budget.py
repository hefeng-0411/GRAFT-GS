"""Run a fail-closed A800 same-object view-budget sweep.

The shell protocol used to continue through every larger candidate after a
CUDA OOM and could mix stale metric files from incompatible renderer policies.
This driver creates a fresh directory per candidate, streams exact child logs,
stops the monotone memory sweep after an OOM, and invokes the scientific
selector even when all completed candidates are rejected.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


OOM_MARKERS = (
    "torch.outofmemoryerror",
    "cuda out of memory",
    "cuda error: out of memory",
    "cuda error: 2[cudamalloc",
    "cuda error: 2 [cudamalloc",
)


def validate_candidates(values: Sequence[int]) -> tuple[int, ...]:
    candidates = tuple(int(value) for value in values)
    if not candidates or any(value < 2 for value in candidates):
        raise ValueError("view candidates must contain positive values >= 2")
    if tuple(sorted(set(candidates))) != candidates:
        raise ValueError("view candidates must be strictly increasing and unique")
    return candidates


def log_reports_oom(text: str) -> bool:
    normalized = text.lower()
    return any(marker in normalized for marker in OOM_MARKERS)


def visible_process_count(python: str) -> int:
    configured = os.environ.get("GRAFT_GS_NPROC_PER_NODE")
    completed = subprocess.run(
        [
            python,
            "-c",
            "import torch; print(torch.cuda.device_count())",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    visible = int(completed.stdout.strip())
    if visible < 1:
        raise RuntimeError("no CUDA device is visible to the pinned interpreter")
    if configured is not None and int(configured) != visible:
        raise RuntimeError(
            "GRAFT_GS_NPROC_PER_NODE does not match the CUDA_VISIBLE_DEVICES "
            f"projection: configured={configured}, visible={visible}"
        )
    return visible


def build_overfit_command(
    *,
    python: str,
    nproc_per_node: int,
    dataset_root: Path,
    manifest: Path,
    object_id: str,
    config: Path,
    vggt_checkpoint: str,
    trellis_checkpoint: str,
    views_per_rank: int,
    evaluation_views: int,
    steps: int,
    minimum_relative_improvement: float,
    output: Path,
) -> list[str]:
    return [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={nproc_per_node}",
        "scripts/overfit_meshfleet_object.py",
        str(dataset_root),
        str(manifest),
        "--split",
        "train",
        "--object-id",
        object_id,
        "--config",
        str(config),
        "--vggt-checkpoint",
        vggt_checkpoint,
        "--trellis-checkpoint",
        trellis_checkpoint,
        "--views-per-rank",
        str(views_per_rank),
        "--evaluation-views",
        str(evaluation_views),
        "--steps",
        str(steps),
        "--minimum-relative-improvement",
        str(minimum_relative_improvement),
        "--output",
        str(output),
    ]


def run_and_tee(command: Sequence[str], log_path: Path) -> tuple[int, bool]:
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf8") as log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        oom = False
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            if log_reports_oom(line):
                oom = True
        return process.wait(), oom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/graft_gs_a800_native.yaml"))
    parser.add_argument("--vggt-checkpoint", required=True)
    parser.add_argument("--trellis-checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--views-per-rank",
        type=int,
        nargs="+",
        default=(16, 24, 32, 48, 64),
    )
    parser.add_argument("--evaluation-views", type=int, default=24)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--minimum-relative-improvement", type=float, default=-1.0)
    parser.add_argument("--python", default=os.environ.get("GRAFT_GS_PYTHON", sys.executable))
    parser.add_argument("--maximum-reserved-fraction", type=float, default=0.85)
    parser.add_argument("--maximum-storage-underflow-fraction", type=float, default=0.05)
    parser.add_argument("--maximum-zero-marginal-fraction", type=float, default=0.05)
    parser.add_argument("--throughput-fraction", type=float, default=0.97)
    parser.add_argument(
        "--continue-after-oom",
        action="store_true",
        help="Diagnostic override; larger candidates are normally skipped after OOM",
    )
    args = parser.parse_args()

    candidates = validate_candidates(args.views_per_rank)
    if args.evaluation_views < 1 or args.steps < 1:
        raise ValueError("evaluation views and steps must be positive")
    for path in (args.dataset_root, args.manifest, args.config):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        if not args.output.is_dir() or any(args.output.iterdir()):
            raise FileExistsError(
                f"sweep output must be fresh to prevent stale report mixing: {args.output}"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    nproc_per_node = visible_process_count(args.python)

    runs: list[dict[str, object]] = []
    metrics_paths: list[Path] = []
    for views_per_rank in candidates:
        run_directory = args.output / f"vpr-{views_per_rank}"
        run_directory.mkdir()
        command = build_overfit_command(
            python=args.python,
            nproc_per_node=nproc_per_node,
            dataset_root=args.dataset_root,
            manifest=args.manifest,
            object_id=args.object_id,
            config=args.config,
            vggt_checkpoint=args.vggt_checkpoint,
            trellis_checkpoint=args.trellis_checkpoint,
            views_per_rank=views_per_rank,
            evaluation_views=args.evaluation_views,
            steps=args.steps,
            minimum_relative_improvement=args.minimum_relative_improvement,
            output=run_directory,
        )
        started = time.perf_counter()
        return_code, oom = run_and_tee(command, run_directory / "run.log")
        metrics = run_directory / "overfit_metrics.json"
        completed = return_code == 0 and metrics.is_file()
        runs.append(
            {
                "views_per_rank": views_per_rank,
                "return_code": return_code,
                "oom": oom,
                "completed_with_metrics": completed,
                "seconds": time.perf_counter() - started,
                "run_directory": str(run_directory),
                "command": command,
            }
        )
        if completed:
            metrics_paths.append(metrics)
        if oom and not args.continue_after_oom:
            break

    summary_path = args.output / "sweep_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema": "graft-gs-a800-sweep-v1",
                "nproc_per_node": nproc_per_node,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "candidates": list(candidates),
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf8",
    )
    if not metrics_paths:
        raise RuntimeError(
            f"no candidate completed with metrics; inspect {summary_path}"
        )

    selection_path = args.output / "selection.json"
    selector_command = [
        args.python,
        "scripts/select_a800_view_budget.py",
        *(str(path) for path in metrics_paths),
        "--maximum-reserved-fraction",
        str(args.maximum_reserved_fraction),
        "--maximum-storage-underflow-fraction",
        str(args.maximum_storage_underflow_fraction),
        "--maximum-zero-marginal-fraction",
        str(args.maximum_zero_marginal_fraction),
        "--throughput-fraction",
        str(args.throughput_fraction),
        "--output",
        str(selection_path),
    ]
    completed = subprocess.run(selector_command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "no completed candidate passed the scientific selector; "
            f"inspect {selection_path} and {summary_path}"
        )


if __name__ == "__main__":
    main()

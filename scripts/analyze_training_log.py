"""Summarize GRAFT-GS worker/tuner failures from a combined text log."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Optional


PREFIXES = {
    "progress": "GRAFT_GS_PROGRESS ",
    "control": "GRAFT_GS_AUTOTUNE_PROBE_CONTROL ",
    "training_control": "GRAFT_GS_TRAINING_SUPERVISOR_CONTROL ",
    "progress_timeout": "GRAFT_GS_AUTOTUNE_PROGRESS_TIMEOUT ",
    "training_progress_timeout": "GRAFT_GS_TRAINING_PROGRESS_TIMEOUT ",
    "ddp_initialized": "GRAFT_GS_DDP_INITIALIZED ",
    "ddp_stage_stall": "GRAFT_GS_DDP_STAGE_STALL ",
    "ddp_local_ready": "GRAFT_GS_DDP_LOCAL_READY ",
    "nonfinite": "GRAFT_GS_NONFINITE ",
}


def _payload(line: str, prefix: str) -> Optional[dict[str, object]]:
    marker = line.find(prefix)
    if marker < 0:
        return None
    try:
        value = json.loads(line[marker + len(prefix) :].strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def analyze(path: Path) -> dict[str, object]:
    counts: Counter[str] = Counter()
    ranks: set[int] = set()
    last_progress: dict[int, dict[str, object]] = {}
    controls: list[dict[str, object]] = []
    progress_timeouts: list[dict[str, object]] = []
    local_ready: list[dict[str, object]] = []
    stage_stalls: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf8", errors="replace").splitlines():
        normalized = line.lower()
        if "sampling: 100%" in normalized or "sampling: 12/12" in normalized:
            counts["completed_sampling_displays"] += 1
        if any(
            marker in normalized
            for marker in (
                "torch.outofmemoryerror",
                "cuda out of memory",
                "cuda error: out of memory",
            )
        ):
            counts["cuda_oom_lines"] += 1
        if "watchdog caught collective operation timeout" in normalized:
            counts["nccl_watchdog_timeout_lines"] += 1
        if "distbackenderror" in normalized or "nccl error" in normalized:
            counts["collective_failure_lines"] += 1
        for name, prefix in PREFIXES.items():
            value = _payload(line, prefix)
            if value is None:
                continue
            counts[name] += 1
            rank_value = value.get("rank")
            if isinstance(rank_value, int):
                ranks.add(rank_value)
            if name == "progress" and isinstance(rank_value, int):
                previous = last_progress.get(rank_value)
                if previous is None or int(value.get("sequence", 0)) > int(
                    previous.get("sequence", 0)
                ):
                    last_progress[rank_value] = value
            elif name in {"control", "training_control"}:
                controls.append(value)
            elif name in {"progress_timeout", "training_progress_timeout"}:
                progress_timeouts.append(value)
            elif name == "ddp_local_ready":
                local_ready.append(value)
            elif name == "ddp_stage_stall":
                stage_stalls.append(value)

    if counts["cuda_oom_lines"]:
        failure_class = "capacity.cuda_oom"
    elif counts["nonfinite"]:
        failure_class = "numerics.nonfinite"
    elif counts["nccl_watchdog_timeout_lines"] or counts["collective_failure_lines"]:
        failure_class = "distributed.collective_failure"
    elif progress_timeouts:
        failure_class = "liveness.no_semantic_progress"
    elif any("probe_timeout" in str(value.get("reason", "")) for value in controls):
        failure_class = "supervisor.legacy_fixed_wall_timeout"
    elif controls:
        failure_class = "supervisor.process_group_termination"
    else:
        failure_class = None

    synchronized_finite_backward_entry = bool(local_ready) and all(
        value.get("local_nonfinite") is False for value in local_ready
    )
    return {
        "schema": "graft-gs-training-log-analysis-v1",
        "path": str(path.resolve()),
        "failure_class": failure_class,
        "counts": dict(sorted(counts.items())),
        "observed_ranks": sorted(ranks),
        "synchronized_finite_backward_entry": synchronized_finite_backward_entry,
        "last_progress_by_rank": {
            str(rank): {
                name: value.get(name)
                for name in (
                    "stage",
                    "event",
                    "sequence",
                    "semantic_sequence",
                    "global_step",
                    "microstep",
                    "object_ids",
                    "cuda_allocated_bytes",
                    "cuda_reserved_bytes",
                )
                if name in value
            }
            for rank, value in sorted(last_progress.items())
        },
        "last_control": controls[-1] if controls else None,
        "last_progress_timeout": progress_timeouts[-1] if progress_timeouts else None,
        "local_ready": local_ready[-16:],
        "stage_stalls": stage_stalls[-16:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.log.is_file():
        raise FileNotFoundError(args.log)
    report = analyze(args.log)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

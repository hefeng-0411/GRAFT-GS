"""Lightweight NCCL/progress smoke test with no model or dataset loading."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist

from graft_gs.engine import bind_local_cuda_device
from graft_gs.observability import ProgressConfig, ProgressReporter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = bind_local_cuda_device(require_cuda=True)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=120),
    )
    reporter = ProgressReporter(
        ProgressConfig(
            heartbeat_interval_seconds=5.0,
            include_cuda_memory=True,
            profiler_ranges=False,
        ),
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )
    reporter.set_context(validation="progress_ddp")
    assignment = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "logical_device": device.index,
        "device_name": torch.cuda.get_device_name(device),
    }
    assignments: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(assignments, assignment)
    ownership = [
        (str(value["hostname"]), int(value["local_rank"]))
        for value in assignments
        if isinstance(value, dict)
    ]
    if len(ownership) != world_size or len(set(ownership)) != world_size:
        raise RuntimeError(f"rank/device ownership is not one-to-one: {assignments}")

    expected = world_size * (world_size + 1) / 2
    for iteration in range(args.iterations):
        reporter.set_context(validation="progress_ddp", iteration=iteration)
        with reporter.stage("validation.local_cuda"):
            value = torch.full(
                (1024,),
                float(rank + 1),
                dtype=torch.float32,
                device=device,
            )
            value = value.square().sqrt()
        with reporter.stage("collective.validation_all_reduce"):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if not bool(torch.all(value == expected)):
            raise RuntimeError("NCCL all-reduce returned an incorrect value")
        reporter.event(
            "validation.iteration",
            "complete",
            expected_sum=expected,
        )

    result = {
        "schema": "graft-gs-progress-ddp-validation-v1",
        "world_size": world_size,
        "iterations": args.iterations,
        "assignments": assignments,
        "collective": "nccl_all_reduce",
        "valid": True,
    }
    if rank == 0:
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf8")
        print("GRAFT_GS_PROGRESS_DDP_VALIDATION " + json.dumps(result, sort_keys=True))
    dist.barrier()
    reporter.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Visible-rank A800 environment and same-object DDP reference validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import unittest

import torch
import torch.distributed as dist

from validate_environment import audit_environment
from validate_server import _accelerator_contract_errors


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf8")


def _local_device_record(rank: int, local_rank: int) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(local_rank)
    return {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "name": properties.name,
        "capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "current_device": torch.cuda.current_device(),
    }


def _distributed_contract_errors(
    world_size: int,
    visible_device_count: int,
    visible_device_mask: str | None,
    devices: list[dict[str, object]],
    environment_valid: bool,
) -> list[str]:
    errors: list[str] = []
    if visible_device_mask is None or not visible_device_mask.strip():
        errors.append("CUDA_VISIBLE_DEVICES must explicitly identify the assigned GPU subset")
    if world_size < 1:
        errors.append(f"A800 validation requires a positive WORLD_SIZE, received {world_size}")
    if visible_device_count != world_size:
        errors.append(
            "torchrun world size must equal the CUDA_VISIBLE_DEVICES-resolved "
            f"device count; world_size={world_size}, visible={visible_device_count}"
        )
    if not environment_valid:
        errors.append("one or more ranks do not match the exact requirements contract")
    if len(devices) != world_size:
        errors.append("device record count does not match the process-group world size")
        return errors
    if sorted(int(device["rank"]) for device in devices) != list(range(world_size)):
        errors.append("global ranks are not a complete unique range")
    rank_keys = [
        (str(device["hostname"]), int(device["local_rank"])) for device in devices
    ]
    if len(set(rank_keys)) != world_size:
        errors.append("multiple ranks resolve to the same host/local CUDA device")
    if any("A800" not in str(device["name"]).upper() for device in devices):
        errors.append("every distributed rank must execute on an NVIDIA A800")
    if any(int(device["current_device"]) != int(device["local_rank"]) for device in devices):
        errors.append("at least one rank did not bind its declared local CUDA device")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--requirements", type=Path, default=ROOT / "requirements.txt"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/validation/ddp_environment.json"
    )
    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("visible-rank A800 validation requires CUDA")
    visible_device_count = torch.cuda.device_count()
    visible_device_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    environment = audit_environment(args.requirements)
    rank_device = torch.device("cuda", local_rank)
    local_environment_valid = torch.tensor(
        [int(environment["valid"])], dtype=torch.int64, device=rank_device
    )
    dist.all_reduce(local_environment_valid, op=dist.ReduceOp.MIN)
    local_device = _local_device_record(rank, local_rank)
    gathered_devices: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_devices, local_device)
    devices = [device for device in gathered_devices if isinstance(device, dict)]
    accelerator_details = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "devices": devices,
    }
    errors = _accelerator_contract_errors(accelerator_details)
    errors.extend(
        _distributed_contract_errors(
            world_size,
            visible_device_count,
            visible_device_mask,
            devices,
            bool(local_environment_valid.item()),
        )
    )
    preflight = {
        "valid": not errors,
        "world_size": world_size,
        "visible_device_count": visible_device_count,
        "cuda_visible_devices": visible_device_mask,
        "multi_rank": world_size > 1,
        "environment": environment,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "devices": devices,
        "errors": errors,
    }
    if rank == 0:
        _write(args.output.resolve(), {"preflight": preflight, "tests": None})

    validity = torch.tensor([int(not errors)], dtype=torch.int64, device=rank_device)
    dist.all_reduce(validity, op=dist.ReduceOp.MIN)
    if not bool(validity.item()):
        dist.barrier()
        dist.destroy_process_group()
        raise SystemExit(2)

    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite(
        (
            loader.discover(
                str(ROOT / "tests"), pattern="test_distributed_evidence.py"
            ),
            loader.loadTestsFromNames(
                (
                    "tests.test_atlas_mapping.PersistentAtlasTest."
                    "test_pca_frame_repeated_spectrum_has_finite_zero_gauge_gradient",
                    "tests.test_atlas_mapping.PersistentAtlasTest."
                    "test_pca_frame_distinct_spectrum_retains_finite_gradient",
                )
            ),
        )
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    success = torch.tensor(
        [int(result.wasSuccessful())], dtype=torch.int64, device=rank_device
    )
    dist.all_reduce(success, op=dist.ReduceOp.MIN)
    final_success = bool(success.item())
    if rank == 0:
        _write(
            args.output.resolve(),
            {
                "preflight": preflight,
                "tests": {
                    "successful_on_every_rank": final_success,
                    "tests_run_per_rank": result.testsRun,
                    "failures_on_rank_zero": len(result.failures),
                    "errors_on_rank_zero": len(result.errors),
                    "skipped_on_rank_zero": len(result.skipped),
                },
            },
        )
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0 if final_success else 1)


if __name__ == "__main__":
    main()

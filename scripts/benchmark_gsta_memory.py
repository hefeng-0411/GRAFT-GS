"""Measure GSTA recomputation retention and numerical parity on CPU or CUDA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch
from torch import Tensor, nn

from graft_gs.equivariant.gsta import (
    GSTAConfig,
    GaugeCovariantSparseTransportAttention,
    IrrepTensor,
    MultiplicityLinear,
)


@dataclass
class _SyntheticAtlas:
    active_indices: Tensor
    edge_index: Tensor
    chart_centers: Tensor
    chart_frames: Tensor
    cell_sides: Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.active_indices.numel())


def _atlas(vertices: int, device: torch.device) -> _SyntheticAtlas:
    active = torch.arange(vertices, dtype=torch.int64, device=device)
    neighbor = torch.remainder(active + 1, vertices)
    edge = torch.cat(
        (
            torch.stack((active, neighbor)),
            torch.stack((neighbor, active)),
        ),
        dim=1,
    )
    angle = torch.arange(vertices, dtype=torch.float32, device=device)
    angle = angle * (2.0 * torch.pi / float(vertices))
    centers = torch.stack(
        (torch.cos(angle), torch.sin(angle), 0.05 * torch.sin(3.0 * angle)),
        dim=-1,
    )
    frames = torch.eye(3, dtype=torch.float32, device=device).expand(
        vertices, -1, -1
    )
    sides = torch.full((vertices,), 0.1, dtype=torch.float32, device=device)
    return _SyntheticAtlas(active, edge, centers, frames, sides)


def _encoder(
    layers: int,
    checkpointing: bool,
    device: torch.device,
) -> nn.ModuleList:
    torch.manual_seed(711)
    config = GSTAConfig(activation_checkpointing=checkpointing)
    encoder = nn.ModuleList(
        GaugeCovariantSparseTransportAttention(config) for _ in range(layers)
    )
    for layer in encoder:
        for child in layer.modules():
            if isinstance(child, MultiplicityLinear):
                nn.utils.parametrizations.spectral_norm(
                    child,
                    name="weight",
                    n_power_iterations=1,
                )
    return encoder.to(device=device).train()


def _inputs(
    vertices: int,
    device: torch.device,
) -> IrrepTensor:
    generator = torch.Generator(device="cpu").manual_seed(913)
    return IrrepTensor(
        torch.randn(vertices, 60, generator=generator).to(device).requires_grad_(),
        torch.randn(vertices, 16, 3, generator=generator)
        .to(device)
        .requires_grad_(),
        torch.randn(vertices, 4, 5, generator=generator)
        .to(device)
        .requires_grad_(),
    )


def _execute(
    *,
    vertices: int,
    layers: int,
    checkpointing: bool,
    device: torch.device,
) -> tuple[dict[str, float | int | bool], Tensor, tuple[Tensor, ...], dict[str, Tensor]]:
    encoder = _encoder(layers, checkpointing, device)
    atlas = _atlas(vertices, device)
    fields = _inputs(vertices, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    saved_storages: dict[int, int] = {}
    saved_tensor_events = 0

    def pack_saved_tensor(tensor: Tensor) -> Tensor:
        nonlocal saved_tensor_events
        saved_tensor_events += 1
        storage = tensor.untyped_storage()
        saved_storages.setdefault(storage.data_ptr(), storage.nbytes())
        return tensor

    def unpack_saved_tensor(tensor: Tensor) -> Tensor:
        return tensor

    started = time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(
        pack_saved_tensor,
        unpack_saved_tensor,
    ):
        output = fields
        for layer in encoder:
            output = layer(atlas, output)
        packed = output.pack()
        probe = torch.linspace(
            -0.25,
            0.75,
            packed.shape[-1],
            dtype=packed.dtype,
            device=device,
        )
        loss = torch.mean(packed * probe)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    result = {
        "activation_checkpointing": checkpointing,
        "vertices": vertices,
        "directed_edges_with_self": 3 * vertices,
        "layers": layers,
        "seconds": seconds,
        "saved_tensor_storage_bytes": sum(saved_storages.values()),
        "saved_tensor_storage_count": len(saved_storages),
        "saved_tensor_events": saved_tensor_events,
        "loss": float(loss.detach().cpu()),
    }
    if device.type == "cuda":
        result.update(
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            ending_allocated_bytes=int(torch.cuda.memory_allocated(device)),
        )
    field_gradients = (
        fields.scalar.grad.detach().cpu(),
        fields.vector.grad.detach().cpu(),
        fields.tensor.grad.detach().cpu(),
    )
    parameter_gradients = {
        name: parameter.grad.detach().cpu()
        for name, parameter in encoder.named_parameters()
        if parameter.grad is not None
    }
    packed_cpu = packed.detach().cpu()
    del loss, packed, output, fields, atlas, encoder
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return result, packed_cpu, field_gradients, parameter_gradients


def _relative_l2(left: Tensor, right: Tensor) -> float:
    numerator = torch.linalg.vector_norm((left - right).to(torch.float64))
    denominator = torch.linalg.vector_norm(left.to(torch.float64)).clamp_min(1.0e-30)
    return float((numerator / denominator).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertices", type=int, default=8192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help=(
            "CUDA reports allocator peaks and requires exactly one explicitly "
            "selected GPU; CPU still audits autograd-retained storage"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maximum-output-relative-l2", type=float, default=2.0e-6)
    parser.add_argument("--maximum-gradient-relative-l2", type=float, default=5.0e-6)
    args = parser.parse_args()
    if args.vertices < 2 or args.layers < 1:
        raise ValueError("benchmark vertices/layers are outside their domains")
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "CUDA benchmark requires exactly one explicitly selected device"
            )
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    baseline, baseline_output, baseline_field_gradients, baseline_parameters = _execute(
        vertices=args.vertices,
        layers=args.layers,
        checkpointing=False,
        device=device,
    )
    recomputed, recomputed_output, recomputed_field_gradients, recomputed_parameters = _execute(
        vertices=args.vertices,
        layers=args.layers,
        checkpointing=True,
        device=device,
    )
    output_relative_l2 = _relative_l2(baseline_output, recomputed_output)
    field_gradient_relative_l2 = max(
        _relative_l2(left, right)
        for left, right in zip(
            baseline_field_gradients,
            recomputed_field_gradients,
        )
    )
    if set(baseline_parameters) != set(recomputed_parameters):
        raise RuntimeError("checkpointed encoder changed parameter-gradient topology")
    parameter_gradient_relative_l2 = max(
        _relative_l2(baseline_parameters[name], recomputed_parameters[name])
        for name in baseline_parameters
    )
    memory_reduced = int(recomputed["saved_tensor_storage_bytes"]) < int(
        baseline["saved_tensor_storage_bytes"]
    )
    if device.type == "cuda":
        memory_reduced = memory_reduced and int(
            recomputed["peak_allocated_bytes"]
        ) < int(baseline["peak_allocated_bytes"])
    valid = (
        output_relative_l2 <= args.maximum_output_relative_l2
        and field_gradient_relative_l2 <= args.maximum_gradient_relative_l2
        and parameter_gradient_relative_l2 <= args.maximum_gradient_relative_l2
        and memory_reduced
    )
    payload = {
        "schema": "graft-gs-gsta-memory-benchmark-v1",
        "device_type": device.type,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "compute_capability": (
            list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None
        ),
        "baseline": baseline,
        "recomputed": recomputed,
        "saved_tensor_storage_reduction_bytes": int(
            baseline["saved_tensor_storage_bytes"]
        )
        - int(recomputed["saved_tensor_storage_bytes"]),
        "saved_tensor_storage_reduction_fraction": 1.0
        - float(recomputed["saved_tensor_storage_bytes"])
        / float(baseline["saved_tensor_storage_bytes"]),
        "output_relative_l2": output_relative_l2,
        "maximum_field_gradient_relative_l2": field_gradient_relative_l2,
        "maximum_parameter_gradient_relative_l2": (
            parameter_gradient_relative_l2
        ),
        "valid": valid,
    }
    if device.type == "cuda":
        payload.update(
            peak_allocated_reduction_bytes=int(
                baseline["peak_allocated_bytes"]
            )
            - int(recomputed["peak_allocated_bytes"]),
            peak_allocated_reduction_fraction=1.0
            - float(recomputed["peak_allocated_bytes"])
            / float(baseline["peak_allocated_bytes"]),
        )
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf8")
    print(encoded, end="")
    if not valid:
        raise RuntimeError("GSTA recomputation memory/parity acceptance failed")


if __name__ == "__main__":
    main()

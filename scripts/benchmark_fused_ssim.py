"""Audit fused SSIM value/adjoint parity and CUDA allocation at target shape."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import time

import torch
from torch import Tensor

from graft_gs.engine.losses import (
    _recomputed_dense_loss,
    multiscale_perceptual_loss,
    robust_rgb,
)
from graft_gs.kernels.fused_ssim import (
    _reference_ssim_loss,
    fused_ssim_loss,
    triton_ssim_available,
)


def _relative_l2(left: Tensor, right: Tensor) -> float:
    numerator = torch.linalg.vector_norm((left - right).to(torch.float64))
    denominator = torch.linalg.vector_norm(right.to(torch.float64)).clamp_min(1.0e-30)
    return float((numerator / denominator).cpu())


def _measure(
    function: Callable[[Tensor, Tensor, Tensor], Tensor],
    predicted: Tensor,
    target: Tensor,
    mask: Tensor,
) -> tuple[dict[str, float | int], Tensor]:
    argument = predicted.detach().clone().requires_grad_()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(argument.device)
    baseline = torch.cuda.memory_allocated(argument.device)
    torch.cuda.synchronize(argument.device)
    started = time.perf_counter()
    loss = function(argument, target, mask)
    forward_allocated = torch.cuda.memory_allocated(argument.device)
    loss.backward()
    torch.cuda.synchronize(argument.device)
    seconds = time.perf_counter() - started
    result = {
        "loss": float(loss.detach().cpu()),
        "seconds": seconds,
        "baseline_allocated_bytes": baseline,
        "forward_retained_increment_bytes": forward_allocated - baseline,
        "peak_increment_bytes": (
            torch.cuda.max_memory_allocated(argument.device) - baseline
        ),
    }
    return result, argument.grad.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maximum-loss-error", type=float, default=2.0e-7)
    parser.add_argument("--maximum-gradient-relative-l2", type=float, default=5.0e-6)
    args = parser.parse_args()
    if min(args.batch, args.views, args.height, args.width) < 1:
        raise ValueError("all tensor dimensions must be positive")
    if not triton_ssim_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "benchmark requires Triton and exactly one CUDA_VISIBLE_DEVICES GPU"
        )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (args.batch, args.views, 3, args.height, args.width)
    predicted = torch.rand(shape, generator=generator, device=device)
    target = torch.rand(shape, generator=generator, device=device)
    mask = torch.rand(
        args.batch,
        args.views,
        1,
        args.height,
        args.width,
        generator=generator,
        device=device,
    )

    # Compile before timing; production likewise pays this once per shape.
    warmup = fused_ssim_loss(predicted.requires_grad_(), target, mask)
    warmup.backward()
    predicted = predicted.detach()
    reference, reference_gradient = _measure(
        _reference_ssim_loss, predicted, target, mask
    )
    fused, fused_gradient = _measure(fused_ssim_loss, predicted, target, mask)
    loss_error = abs(float(reference["loss"]) - float(fused["loss"]))
    gradient_relative_l2 = _relative_l2(fused_gradient, reference_gradient)

    def eager_dense(left: Tensor, right: Tensor, weight: Tensor) -> Tensor:
        return (
            robust_rgb(left, right, weight)
            + fused_ssim_loss(left, right, weight)
            + multiscale_perceptual_loss(left, right, weight)
        )

    def recomputed_dense(left: Tensor, right: Tensor, weight: Tensor) -> Tensor:
        return (
            _recomputed_dense_loss(robust_rgb, left, right, weight)
            + fused_ssim_loss(left, right, weight)
            + _recomputed_dense_loss(
                multiscale_perceptual_loss, left, right, weight
            )
        )

    eager_dense_result, eager_dense_gradient = _measure(
        eager_dense, predicted, target, mask
    )
    recomputed_dense_result, recomputed_dense_gradient = _measure(
        recomputed_dense, predicted, target, mask
    )
    dense_loss_error = abs(
        float(eager_dense_result["loss"])
        - float(recomputed_dense_result["loss"])
    )
    dense_gradient_relative_l2 = _relative_l2(
        recomputed_dense_gradient, eager_dense_gradient
    )
    report = {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "shape": list(shape),
        "reference": reference,
        "fused": fused,
        "loss_absolute_error": loss_error,
        "gradient_relative_l2": gradient_relative_l2,
        "peak_reduction_bytes": int(reference["peak_increment_bytes"])
        - int(fused["peak_increment_bytes"]),
        "forward_retention_reduction_bytes": int(
            reference["forward_retained_increment_bytes"]
        )
        - int(fused["forward_retained_increment_bytes"]),
        "dense_loss_recomputation": {
            "eager": eager_dense_result,
            "recomputed": recomputed_dense_result,
            "loss_absolute_error": dense_loss_error,
            "gradient_relative_l2": dense_gradient_relative_l2,
            "forward_retention_reduction_bytes": int(
                eager_dense_result["forward_retained_increment_bytes"]
            )
            - int(recomputed_dense_result["forward_retained_increment_bytes"]),
        },
    }
    if loss_error > args.maximum_loss_error:
        raise RuntimeError("fused SSIM loss differs from the eager oracle")
    if gradient_relative_l2 > args.maximum_gradient_relative_l2:
        raise RuntimeError("fused SSIM adjoint differs from the eager oracle")
    if report["peak_reduction_bytes"] <= 0:
        raise RuntimeError("fused SSIM did not reduce peak CUDA allocation")
    if dense_loss_error != 0.0 or dense_gradient_relative_l2 != 0.0:
        raise RuntimeError("dense-loss recomputation changed the value or adjoint")
    if (
        report["dense_loss_recomputation"][
            "forward_retention_reduction_bytes"
        ]
        <= 0
    ):
        raise RuntimeError("dense-loss recomputation did not reduce retention")
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

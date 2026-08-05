"""Memory-constant fused SSIM reduction for CUDA training.

The CUDA path is written in Triton and compiled to architecture-specific PTX.
Unlike the eager formulation, it never materializes full-resolution mean,
variance, covariance, similarity, or clamped-loss fields.  Backward recomputes
the local statistics and gathers the nine output-pixel contributions affecting
each input pixel, avoiding both image-sized saved tensors and atomic gradient
accumulation.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

try:  # Triton ships with supported CUDA PyTorch distributions.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by CPU-only installations.
    triton = None
    tl = None


_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def triton_ssim_available() -> bool:
    """Return whether the fused CUDA implementation can be dispatched."""

    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _ssim_forward_partials(
        left,
        right,
        mask,
        partials,
        total_elements: tl.constexpr,
        views: tl.constexpr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        ls0: tl.constexpr,
        ls1: tl.constexpr,
        ls2: tl.constexpr,
        ls3: tl.constexpr,
        ls4: tl.constexpr,
        rs0: tl.constexpr,
        rs1: tl.constexpr,
        rs2: tl.constexpr,
        rs3: tl.constexpr,
        rs4: tl.constexpr,
        ms0: tl.constexpr,
        ms1: tl.constexpr,
        ms3: tl.constexpr,
        ms4: tl.constexpr,
        masked: tl.constexpr,
        block_size: tl.constexpr,
    ):
        element = tl.program_id(0) * block_size + tl.arange(0, block_size)
        live = element < total_elements
        pixel_x = element % width
        quotient = element // width
        pixel_y = quotient % height
        quotient = quotient // height
        channel = quotient % channels
        view_batch = quotient // channels
        view = view_batch % views
        batch = view_batch // views

        sum_left = tl.zeros((block_size,), tl.float32)
        sum_right = tl.zeros((block_size,), tl.float32)
        sum_left_square = tl.zeros((block_size,), tl.float32)
        sum_right_square = tl.zeros((block_size,), tl.float32)
        sum_product = tl.zeros((block_size,), tl.float32)
        for delta_y in range(-1, 2):
            local_y = pixel_y + delta_y
            for delta_x in range(-1, 2):
                local_x = pixel_x + delta_x
                inside = (
                    live
                    & (local_y >= 0)
                    & (local_y < height)
                    & (local_x >= 0)
                    & (local_x < width)
                )
                left_offset = (
                    batch * ls0
                    + view * ls1
                    + channel * ls2
                    + local_y * ls3
                    + local_x * ls4
                )
                right_offset = (
                    batch * rs0
                    + view * rs1
                    + channel * rs2
                    + local_y * rs3
                    + local_x * rs4
                )
                x = tl.load(left + left_offset, mask=inside, other=0.0).to(tl.float32)
                y = tl.load(right + right_offset, mask=inside, other=0.0).to(tl.float32)
                sum_left += x
                sum_right += y
                sum_left_square += x * x
                sum_right_square += y * y
                sum_product += x * y

        inverse_window = 1.0 / 9.0
        mean_left = sum_left * inverse_window
        mean_right = sum_right * inverse_window
        variance_left = sum_left_square * inverse_window - mean_left * mean_left
        variance_right = sum_right_square * inverse_window - mean_right * mean_right
        covariance = sum_product * inverse_window - mean_left * mean_right
        luminance = 2.0 * mean_left * mean_right + 0.0001
        structure = 2.0 * covariance + 0.0009
        mean_energy = mean_left * mean_left + mean_right * mean_right + 0.0001
        variance_energy = variance_left + variance_right + 0.0009
        denominator = tl.maximum(mean_energy * variance_energy, 1.0e-12)
        similarity = luminance * structure / denominator
        similarity = tl.maximum(-1.0, tl.minimum(1.0, similarity))

        if masked:
            mask_offset = (
                batch * ms0
                + view * ms1
                + pixel_y * ms3
                + pixel_x * ms4
            )
            weight = tl.load(mask + mask_offset, mask=live, other=0.0).to(tl.float32)
        else:
            weight = tl.full((block_size,), 1.0, tl.float32)
        numerator = tl.where(live, 0.5 * (1.0 - similarity) * weight, 0.0)
        # Count each spatial weight once, not once per RGB channel.
        weight_value = tl.where(live & (channel == 0), weight, 0.0)
        output = tl.program_id(0) * 2
        tl.store(partials + output, tl.sum(numerator, axis=0))
        tl.store(partials + output + 1, tl.sum(weight_value, axis=0))


    @triton.jit
    def _reduce_pair_partials(
        source,
        destination,
        pair_count,
        block_size: tl.constexpr,
    ):
        pair = tl.program_id(0) * block_size + tl.arange(0, block_size)
        live = pair < pair_count
        numerator = tl.load(source + pair * 2, mask=live, other=0.0)
        denominator = tl.load(source + pair * 2 + 1, mask=live, other=0.0)
        output = tl.program_id(0) * 2
        tl.store(destination + output, tl.sum(numerator, axis=0))
        tl.store(destination + output + 1, tl.sum(denominator, axis=0))


    @triton.jit
    def _ssim_backward_gather(
        left,
        right,
        mask,
        reduced,
        upstream,
        gradient,
        total_elements: tl.constexpr,
        views: tl.constexpr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        ls0: tl.constexpr,
        ls1: tl.constexpr,
        ls2: tl.constexpr,
        ls3: tl.constexpr,
        ls4: tl.constexpr,
        rs0: tl.constexpr,
        rs1: tl.constexpr,
        rs2: tl.constexpr,
        rs3: tl.constexpr,
        rs4: tl.constexpr,
        ms0: tl.constexpr,
        ms1: tl.constexpr,
        ms3: tl.constexpr,
        ms4: tl.constexpr,
        differentiate_right: tl.constexpr,
        masked: tl.constexpr,
        block_size: tl.constexpr,
    ):
        element = tl.program_id(0) * block_size + tl.arange(0, block_size)
        live = element < total_elements
        input_x = element % width
        quotient = element // width
        input_y = quotient % height
        quotient = quotient // height
        channel = quotient % channels
        view_batch = quotient // channels
        view = view_batch % views
        batch = view_batch // views
        left_center_offset = (
            batch * ls0
            + view * ls1
            + channel * ls2
            + input_y * ls3
            + input_x * ls4
        )
        right_center_offset = (
            batch * rs0
            + view * rs1
            + channel * rs2
            + input_y * rs3
            + input_x * rs4
        )
        center_left = tl.load(
            left + left_center_offset, mask=live, other=0.0
        ).to(tl.float32)
        center_right = tl.load(
            right + right_center_offset, mask=live, other=0.0
        ).to(tl.float32)
        weight_sum = tl.maximum(tl.load(reduced + 1), 1.0)
        upstream_value = tl.load(upstream).to(tl.float32)
        normalization = -0.5 * upstream_value / (channels * weight_sum)
        accumulated = tl.zeros((block_size,), tl.float32)

        # An input pixel affects exactly the output centers in its 3x3
        # neighborhood.  Recompute each center's moments and gather its
        # derivative, so every gradient address has one deterministic writer.
        for output_delta_y in range(-1, 2):
            output_y = input_y + output_delta_y
            for output_delta_x in range(-1, 2):
                output_x = input_x + output_delta_x
                output_live = (
                    live
                    & (output_y >= 0)
                    & (output_y < height)
                    & (output_x >= 0)
                    & (output_x < width)
                )
                sum_left = tl.zeros((block_size,), tl.float32)
                sum_right = tl.zeros((block_size,), tl.float32)
                sum_left_square = tl.zeros((block_size,), tl.float32)
                sum_right_square = tl.zeros((block_size,), tl.float32)
                sum_product = tl.zeros((block_size,), tl.float32)
                for window_delta_y in range(-1, 2):
                    local_y = output_y + window_delta_y
                    for window_delta_x in range(-1, 2):
                        local_x = output_x + window_delta_x
                        inside = (
                            output_live
                            & (local_y >= 0)
                            & (local_y < height)
                            & (local_x >= 0)
                            & (local_x < width)
                        )
                        left_offset = (
                            batch * ls0
                            + view * ls1
                            + channel * ls2
                            + local_y * ls3
                            + local_x * ls4
                        )
                        right_offset = (
                            batch * rs0
                            + view * rs1
                            + channel * rs2
                            + local_y * rs3
                            + local_x * rs4
                        )
                        x = tl.load(
                            left + left_offset, mask=inside, other=0.0
                        ).to(tl.float32)
                        y = tl.load(
                            right + right_offset, mask=inside, other=0.0
                        ).to(tl.float32)
                        sum_left += x
                        sum_right += y
                        sum_left_square += x * x
                        sum_right_square += y * y
                        sum_product += x * y

                inverse_window = 1.0 / 9.0
                mean_left = sum_left * inverse_window
                mean_right = sum_right * inverse_window
                variance_left = (
                    sum_left_square * inverse_window - mean_left * mean_left
                )
                variance_right = (
                    sum_right_square * inverse_window - mean_right * mean_right
                )
                covariance = sum_product * inverse_window - mean_left * mean_right
                luminance = 2.0 * mean_left * mean_right + 0.0001
                structure = 2.0 * covariance + 0.0009
                mean_energy = (
                    mean_left * mean_left + mean_right * mean_right + 0.0001
                )
                variance_energy = variance_left + variance_right + 0.0009
                raw_denominator = mean_energy * variance_energy
                denominator = tl.maximum(raw_denominator, 1.0e-12)
                numerator = luminance * structure
                similarity = numerator / denominator

                if differentiate_right:
                    derivative_luminance = 2.0 * mean_left * inverse_window
                    derivative_structure = (
                        2.0 * (center_left - mean_left) * inverse_window
                    )
                    derivative_mean_energy = 2.0 * mean_right * inverse_window
                    derivative_variance_energy = (
                        2.0 * (center_right - mean_right) * inverse_window
                    )
                else:
                    derivative_luminance = 2.0 * mean_right * inverse_window
                    derivative_structure = (
                        2.0 * (center_right - mean_right) * inverse_window
                    )
                    derivative_mean_energy = 2.0 * mean_left * inverse_window
                    derivative_variance_energy = (
                        2.0 * (center_left - mean_left) * inverse_window
                    )
                derivative_numerator = (
                    derivative_luminance * structure
                    + luminance * derivative_structure
                )
                derivative_denominator = (
                    derivative_mean_energy * variance_energy
                    + mean_energy * derivative_variance_energy
                )
                derivative_similarity = tl.where(
                    raw_denominator >= 1.0e-12,
                    (
                        derivative_numerator * denominator
                        - numerator * derivative_denominator
                    )
                    / (denominator * denominator),
                    derivative_numerator / denominator,
                )
                clamp_active = (similarity >= -1.0) & (similarity <= 1.0)
                if masked:
                    mask_offset = (
                        batch * ms0
                        + view * ms1
                        + output_y * ms3
                        + output_x * ms4
                    )
                    output_weight = tl.load(
                        mask + mask_offset, mask=output_live, other=0.0
                    ).to(tl.float32)
                else:
                    output_weight = tl.full((block_size,), 1.0, tl.float32)
                accumulated += tl.where(
                    output_live & clamp_active,
                    normalization * output_weight * derivative_similarity,
                    0.0,
                )
        tl.store(gradient + element, accumulated, mask=live)


def _can_use_triton(left: Tensor, right: Tensor, mask: Optional[Tensor]) -> bool:
    return bool(
        triton is not None
        and left.is_cuda
        and right.is_cuda
        and left.device == right.device
        and left.dtype == right.dtype
        and left.dtype in _SUPPORTED_DTYPES
        and (mask is None or (mask.is_cuda and mask.device == left.device))
        and (
            mask is None
            or mask.dtype == left.dtype
            or mask.dtype == torch.bool
        )
        and (mask is None or not mask.requires_grad)
    )


def _validate_inputs(left: Tensor, right: Tensor, mask: Optional[Tensor]) -> None:
    if left.shape != right.shape or left.ndim != 5:
        raise ValueError("SSIM inputs must share shape [B,K,3,H,W]")
    if left.shape[2] != 3:
        raise ValueError("SSIM requires exactly three color channels")
    if any(dimension < 1 for dimension in left.shape):
        raise ValueError("SSIM dimensions must all be positive")
    if left.device != right.device or left.dtype != right.dtype:
        raise ValueError("SSIM inputs must share device and dtype")
    if mask is not None:
        expected = left.shape[:2] + (1,) + left.shape[-2:]
        if mask.shape != expected:
            raise ValueError(f"SSIM mask must have shape {expected}")
        if mask.device != left.device:
            raise ValueError("SSIM mask must share the image device")


def _reference_ssim_loss(
    left: Tensor,
    right: Tensor,
    mask: Optional[Tensor],
) -> Tensor:
    """Original eager definition used off CUDA and as the audit oracle."""

    batch, views, channels, height, width = left.shape
    left_flat = left.reshape(batch * views, channels, height, width)
    right_flat = right.reshape_as(left_flat)
    mean_left = torch.nn.functional.avg_pool2d(left_flat, 3, stride=1, padding=1)
    mean_right = torch.nn.functional.avg_pool2d(right_flat, 3, stride=1, padding=1)
    variance_left = (
        torch.nn.functional.avg_pool2d(left_flat.square(), 3, 1, 1)
        - mean_left.square()
    )
    variance_right = (
        torch.nn.functional.avg_pool2d(right_flat.square(), 3, 1, 1)
        - mean_right.square()
    )
    covariance = (
        torch.nn.functional.avg_pool2d(left_flat * right_flat, 3, 1, 1)
        - mean_left * mean_right
    )
    similarity = (
        (2.0 * mean_left * mean_right + 0.01**2)
        * (2.0 * covariance + 0.03**2)
        / (
            (mean_left.square() + mean_right.square() + 0.01**2)
            * (variance_left + variance_right + 0.03**2)
        ).clamp_min(1.0e-12)
    )
    loss = 0.5 * (1.0 - similarity.clamp(-1.0, 1.0)).mean(
        dim=1, keepdim=True
    )
    if mask is None:
        return loss.mean()
    weight = mask.reshape(batch * views, 1, height, width)
    return torch.sum(loss * weight) / weight.sum().clamp_min(1.0)


class _FusedSSIM(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: object,
        left: Tensor,
        right: Tensor,
        mask_storage: Tensor,
        masked: bool,
    ) -> Tensor:
        batch, views, channels, height, width = left.shape
        total = left.numel()
        block_size = 256
        pair_count = triton.cdiv(total, block_size)
        partials = torch.empty((pair_count, 2), device=left.device, dtype=torch.float32)
        left_stride = left.stride()
        right_stride = right.stride()
        mask_stride = mask_storage.stride() if masked else (0, 0, 0, 0, 0)
        with torch.cuda.device(left.device):
            _ssim_forward_partials[(pair_count,)](
                left,
                right,
                mask_storage,
                partials,
                total,
                views,
                channels,
                height,
                width,
                *left_stride,
                *right_stride,
                mask_stride[0],
                mask_stride[1],
                mask_stride[3],
                mask_stride[4],
                masked=masked,
                block_size=block_size,
                num_warps=8,
            )
            reduction_block = 256
            while pair_count > 1:
                reduced_count = triton.cdiv(pair_count, reduction_block)
                reduced = torch.empty(
                    (reduced_count, 2), device=left.device, dtype=torch.float32
                )
                _reduce_pair_partials[(reduced_count,)](
                    partials,
                    reduced,
                    pair_count,
                    block_size=reduction_block,
                    num_warps=8,
                )
                partials = reduced
                pair_count = reduced_count
        result = partials[0, 0] / (channels * partials[0, 1].clamp_min(1.0))
        ctx.save_for_backward(left, right, mask_storage, partials)
        ctx.masked = masked
        return result

    @staticmethod
    def backward(ctx: object, upstream: Tensor) -> tuple[Optional[Tensor], ...]:  # type: ignore[override]
        left, right, mask_storage, reduced = ctx.saved_tensors
        batch, views, channels, height, width = left.shape
        total = left.numel()
        block_size = 128
        grid = (triton.cdiv(total, block_size),)
        left_stride = left.stride()
        right_stride = right.stride()
        mask_stride = mask_storage.stride() if ctx.masked else (0, 0, 0, 0, 0)

        def calculate(differentiate_right: bool, template: Tensor) -> Tensor:
            gradient = torch.empty_like(template, memory_format=torch.contiguous_format)
            with torch.cuda.device(left.device):
                _ssim_backward_gather[grid](
                    left,
                    right,
                    mask_storage,
                    reduced,
                    upstream,
                    gradient,
                    total,
                    views,
                    channels,
                    height,
                    width,
                    *left_stride,
                    *right_stride,
                    mask_stride[0],
                    mask_stride[1],
                    mask_stride[3],
                    mask_stride[4],
                    differentiate_right=differentiate_right,
                    masked=ctx.masked,
                    block_size=block_size,
                    num_warps=4,
                )
            return gradient

        left_gradient = calculate(False, left) if ctx.needs_input_grad[0] else None
        right_gradient = calculate(True, right) if ctx.needs_input_grad[1] else None
        return left_gradient, right_gradient, None, None


def fused_ssim_loss(
    left: Tensor,
    right: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Compute the exact 3x3 SSIM objective with a memory-constant CUDA adjoint.

    CUDA FP16/BF16/FP32 inputs use FP32 register accumulation in the Triton/PTX
    kernel.  The deployed renderer/loss boundary is FP32 and is audited against
    the eager FP32 oracle. CPU tensors, float64 audit tensors, and
    differentiable masks use the eager reference expression.
    """

    _validate_inputs(left, right, mask)
    if left.is_cuda and left.dtype in _SUPPORTED_DTYPES and triton is None:
        raise RuntimeError(
            "CUDA SSIM requires Triton; refusing the image-sized eager path "
            "because it violates the bounded-memory training contract"
        )
    if not _can_use_triton(left, right, mask):
        return _reference_ssim_loss(left, right, mask)
    mask_storage = mask if mask is not None else left.new_empty((0,))
    return _FusedSSIM.apply(left, right, mask_storage, mask is not None)


__all__ = ["fused_ssim_loss", "triton_ssim_available"]

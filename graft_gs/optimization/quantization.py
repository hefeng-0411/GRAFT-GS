"""Matched-forward/backward block quantization restricted to tensor operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import Tensor, nn
import torch.nn.utils.parametrize as parametrize

from ..equivariant.gsta import MultiplicityLinear
from ..manifold.barrier import BarrierProjector
from ..manifold.geometry import ManifoldState


@dataclass(frozen=True)
class QuantizationConfig:
    bits: int = 8
    block_size: int = 16
    stochastic_rounding: bool = True
    quantize_dense_attention_maps: bool = True
    adversarial_log_scale_radius: float = 0.0

    def __post_init__(self) -> None:
        if self.bits not in (4, 8):
            raise ValueError("reference QAT supports signed INT4 or INT8")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if self.adversarial_log_scale_radius < 0:
            raise ValueError("adversarial log-scale radius must be non-negative")


@dataclass(frozen=True)
class QuantizationTopologyCertificate:
    """Measured terms in the conditional one-step topology certificate."""

    query_error: Tensor
    attention_score_error_bound: Tensor
    vector_field_perturbation_bound: Tensor
    step_displacement_bound: Tensor
    topology_boundary_margin: Tensor
    certified: Tensor

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "query_error": float(self.query_error.detach().cpu()),
            "attention_score_error_bound": float(
                self.attention_score_error_bound.detach().cpu()
            ),
            "vector_field_perturbation_bound": float(
                self.vector_field_perturbation_bound.detach().cpu()
            ),
            "step_displacement_bound": float(
                self.step_displacement_bound.detach().cpu()
            ),
            "topology_boundary_margin": float(
                self.topology_boundary_margin.detach().cpu()
            ),
            "certified": bool(self.certified.detach().cpu()),
        }


class BlockwiseFakeQuantizer(nn.Module):
    """Symmetric block quantizer with a straight-through matched backward path."""

    def __init__(self, config: QuantizationConfig = QuantizationConfig()) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "adversarial_log_scale",
            torch.zeros((), requires_grad=config.adversarial_log_scale_radius > 0),
            persistent=True,
        )

    def forward(self, value: Tensor) -> Tensor:
        shape = value.shape
        flat = value.reshape(-1)
        padding = (-flat.numel()) % self.config.block_size
        if padding:
            flat = torch.nn.functional.pad(flat, (0, padding))
        block = flat.reshape(-1, self.config.block_size)
        maximum = block.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(value.dtype).tiny)
        qmax = float((1 << (self.config.bits - 1)) - 1)
        base_scale = maximum / qmax
        bounded_log_scale = self.adversarial_log_scale.clamp(
            -self.config.adversarial_log_scale_radius,
            self.config.adversarial_log_scale_radius,
        )
        adversarial_multiplier = torch.exp(bounded_log_scale)
        scale = base_scale * adversarial_multiplier
        normalized = block / scale.detach()
        if self.training and self.config.stochastic_rounding:
            lower = torch.floor(normalized)
            probability = normalized - lower
            quantized_integer = lower + (torch.rand_like(probability) < probability).to(probability)
        else:
            quantized_integer = torch.round(normalized)
        quantized_integer = quantized_integer.clamp(-qmax, qmax)
        # The integer decision and base calibration are discrete for the
        # reference STE. The weight receives the identity gradient, while the
        # bounded scale adversary receives the exact dequantization derivative.
        quantized_scale_path = (
            quantized_integer.detach()
            * base_scale.detach()
            * adversarial_multiplier
        ).reshape(-1)[: value.numel()].reshape(shape)
        straight_through = value + (quantized_scale_path - value).detach()
        return straight_through + quantized_scale_path - quantized_scale_path.detach()

    @torch.no_grad()
    def reset_adversary(self) -> None:
        self.adversarial_log_scale.zero_()

    @torch.no_grad()
    def set_worst_case_from_gradient(self, gradient: Optional[Tensor]) -> None:
        radius = self.config.adversarial_log_scale_radius
        if radius == 0 or gradient is None:
            self.adversarial_log_scale.zero_()
            return
        self.adversarial_log_scale.copy_(radius * torch.sign(gradient.detach()))


class WeightQuantizationParametrization(nn.Module):
    def __init__(self, config: QuantizationConfig) -> None:
        super().__init__()
        self.quantizer = BlockwiseFakeQuantizer(config)

    def forward(self, weight: Tensor) -> Tensor:
        return self.quantizer(weight)


def apply_equivariant_qat(module: nn.Module, config: QuantizationConfig = QuantizationConfig()) -> list[str]:
    """Parametrize tensor-operation weights; manifold/geometric tensors are untouched."""

    registered = []
    for name, child in list(module.named_modules()):
        eligible = isinstance(child, MultiplicityLinear) or (
            config.quantize_dense_attention_maps and isinstance(child, nn.Linear)
        )
        if eligible and hasattr(child, "weight"):
            already_quantized = False
            if parametrize.is_parametrized(child, "weight"):
                already_quantized = any(
                    isinstance(item, WeightQuantizationParametrization)
                    for item in child.parametrizations.weight
                )
            if not already_quantized:
                parametrize.register_parametrization(child, "weight", WeightQuantizationParametrization(config))
                registered.append(name)
    return registered


def quantization_scale_adversaries(module: nn.Module) -> list[BlockwiseFakeQuantizer]:
    """Return every active bounded scale adversary in deterministic module order."""

    return [
        child
        for child in module.modules()
        if isinstance(child, BlockwiseFakeQuantizer)
        and child.config.adversarial_log_scale_radius > 0
    ]


def attention_score_error_bound(query_error: float | Tensor, temperature: float | Tensor) -> Tensor:
    error = torch.as_tensor(query_error)
    temperature_tensor = torch.as_tensor(temperature, dtype=error.dtype, device=error.device)
    if bool(torch.any(error < 0)) or bool(torch.any(temperature_tensor <= 0)):
        raise ValueError("query error must be non-negative and temperature positive")
    return (2.0 * error + error.square()) / temperature_tensor


def topology_step_is_certified(
    query_error: float | Tensor,
    temperature: float | Tensor,
    vector_field_lipschitz: float | Tensor,
    step_size: float | Tensor,
    topology_margin: float | Tensor,
) -> Tensor:
    bound = attention_score_error_bound(query_error, temperature)
    step = torch.as_tensor(step_size, dtype=bound.dtype, device=bound.device)
    lipschitz = torch.as_tensor(
        vector_field_lipschitz, dtype=bound.dtype, device=bound.device
    )
    margin = torch.as_tensor(topology_margin, dtype=bound.dtype, device=bound.device)
    if bool(torch.any(step <= 0)) or bool(torch.any(lipschitz < 0)):
        raise ValueError("step size must be positive and Lipschitz bound non-negative")
    return step * lipschitz * bound < margin


def certify_topology_quantization_step(
    projector: BarrierProjector,
    state: ManifoldState,
    query_error: float | Tensor,
    temperature: float | Tensor,
    vector_field_lipschitz: float | Tensor,
    step_size: float | Tensor,
) -> QuantizationTopologyCertificate:
    """Evaluate every measured term of the specification's boxed inequality.

    ``vector_field_lipschitz`` remains an explicit externally measured upper
    bound; it is never inferred from spectral normalization alone.  In
    contrast, the topology margin is computed from the selected complex,
    collision family, current state, and evidence Riemannian metric.
    """

    # The topology margin is high-precision geometric state and defines the
    # numerical domain of the certificate. Constructing Python scalar inputs
    # first would silently select float32 and round the measured margin.
    margin = projector.topology_boundary_margin(state)
    query_error_tensor = torch.as_tensor(
        query_error, dtype=margin.dtype, device=margin.device
    )
    temperature_tensor = torch.as_tensor(
        temperature, dtype=margin.dtype, device=margin.device
    )
    score_bound = attention_score_error_bound(
        query_error_tensor, temperature_tensor
    )
    lipschitz = torch.as_tensor(
        vector_field_lipschitz,
        dtype=score_bound.dtype,
        device=score_bound.device,
    )
    step = torch.as_tensor(step_size, dtype=score_bound.dtype, device=score_bound.device)
    if bool(torch.any(lipschitz < 0)) or bool(torch.any(step <= 0)):
        raise ValueError("Lipschitz bound must be non-negative and step size positive")
    field_bound = lipschitz * score_bound
    displacement_bound = step * field_bound
    return QuantizationTopologyCertificate(
        query_error=query_error_tensor,
        attention_score_error_bound=score_bound,
        vector_field_perturbation_bound=field_bound,
        step_displacement_bound=displacement_bound,
        topology_boundary_margin=margin,
        certified=displacement_bound < margin,
    )


__all__ = [
    "BlockwiseFakeQuantizer",
    "QuantizationConfig",
    "QuantizationTopologyCertificate",
    "apply_equivariant_qat",
    "attention_score_error_bound",
    "certify_topology_quantization_step",
    "quantization_scale_adversaries",
    "topology_step_is_certified",
]

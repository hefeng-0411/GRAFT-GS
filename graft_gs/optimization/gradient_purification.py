"""Robust Hilbert-space purification for multiview parameter gradients.

The implementation follows Section 5.11 of the GRAFT-GS specification without
ever flattening the full trainable parameter space.  All geometric-median,
cone, and subspace operations are expressed through the small Gram matrix of
the retained view gradients.  This is exact up to floating-point arithmetic;
the only approximation is the configured finite Weiszfeld iteration count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn


Gradient = tuple[Optional[Tensor], ...]


@dataclass(frozen=True)
class GradientPurificationConfig:
    maximum_views: int = 8
    consensus_cosine: float = 0.2
    consensus_relative_singular_value: float = 0.05
    artifact_relative_singular_value: float = 0.1
    weiszfeld_iterations: int = 12
    weiszfeld_smoothing: float = 1.0e-8
    fisher_decay: float = 0.95
    fisher_damping: float = 1.0e-6
    fisher_radius: float = 1.0
    minimum_reliability: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.maximum_views < 2:
            raise ValueError("gradient purification requires at least two views")
        if not 0.0 < self.consensus_cosine < 1.0:
            raise ValueError("consensus_cosine must lie strictly inside (0,1)")
        if not 0.0 < self.consensus_relative_singular_value <= 1.0:
            raise ValueError("consensus singular-value threshold must lie in (0,1]")
        if not 0.0 < self.artifact_relative_singular_value <= 1.0:
            raise ValueError("artifact singular-value threshold must lie in (0,1]")
        if self.weiszfeld_iterations < 1 or self.weiszfeld_smoothing <= 0:
            raise ValueError("Weiszfeld controls must be positive")
        if not 0.0 <= self.fisher_decay < 1.0:
            raise ValueError("fisher_decay must lie in [0,1)")
        if self.fisher_damping <= 0 or self.fisher_radius <= 0:
            raise ValueError("Fisher damping and radius must be positive")
        if self.minimum_reliability <= 0:
            raise ValueError("minimum_reliability must be positive")


@dataclass(frozen=True)
class GradientPurificationDiagnostics:
    retained_views: int
    consensus_rank: int
    artifact_rank: int
    cone_acceptance_fraction: Tensor
    median_residual: Tensor
    fisher_norm: Tensor
    fisher_scale: Tensor


def _first_tensor(gradient: Gradient) -> Tensor:
    for value in gradient:
        if value is not None:
            return value
    raise ValueError("a gradient sample cannot be identically unused")


def _accumulation_dtype(value: Tensor) -> torch.dtype:
    return torch.float64 if value.dtype == torch.float64 else torch.float32


def gradient_inner(left: Gradient, right: Gradient) -> Tensor:
    """Parameter-space Euclidean inner product with stable accumulation."""

    reference = _first_tensor(left if any(item is not None for item in left) else right)
    total = torch.zeros(
        (), device=reference.device, dtype=_accumulation_dtype(reference)
    )
    for lhs, rhs in zip(left, right):
        if lhs is None or rhs is None:
            continue
        total = total + torch.sum(
            lhs.to(total.dtype) * rhs.to(total.dtype)
        )
    return total


def gradient_linear_combination(
    gradients: Sequence[Gradient], coefficients: Tensor
) -> Gradient:
    if len(gradients) != int(coefficients.numel()):
        raise ValueError("one coefficient is required per gradient")
    if not gradients:
        raise ValueError("at least one gradient is required")
    result: list[Optional[Tensor]] = []
    for parameter_index in range(len(gradients[0])):
        value: Optional[Tensor] = None
        for sample_index, gradient in enumerate(gradients):
            component = gradient[parameter_index]
            if component is None:
                continue
            term = component * coefficients[sample_index].to(component.dtype)
            value = term if value is None else value + term
        result.append(value)
    return tuple(result)


def _subtract(left: Gradient, right: Gradient) -> Gradient:
    reference = _first_tensor(left if any(item is not None for item in left) else right)
    coefficients = torch.tensor(
        [1.0, -1.0], device=reference.device, dtype=_accumulation_dtype(reference)
    )
    return gradient_linear_combination((left, right), coefficients)


def _gram(gradients: Sequence[Gradient], weights: Optional[Tensor] = None) -> Tensor:
    count = len(gradients)
    reference = _first_tensor(gradients[0])
    gram = torch.empty(
        count,
        count,
        device=reference.device,
        dtype=_accumulation_dtype(reference),
    )
    for row in range(count):
        for column in range(row, count):
            value = gradient_inner(gradients[row], gradients[column])
            gram[row, column] = value
            gram[column, row] = value
    if weights is not None:
        root = torch.sqrt(weights.to(gram.dtype).clamp_min(0.0))
        gram = root[:, None] * gram * root[None, :]
    return 0.5 * (gram + gram.mT)


def weighted_geometric_median(
    gradients: Sequence[Gradient],
    reliability: Tensor,
    iterations: int,
    smoothing: float,
) -> tuple[Gradient, Tensor]:
    """Smoothed Weiszfeld solve in the implicit parameter Hilbert space."""

    weight = reliability / reliability.sum().clamp_min(smoothing)
    estimate = gradient_linear_combination(gradients, weight)
    residual = weight.new_tensor(float("inf"))
    for _ in range(iterations):
        distance = torch.stack(
            [
                torch.sqrt(
                    gradient_inner(_subtract(sample, estimate), _subtract(sample, estimate))
                    + smoothing**2
                )
                for sample in gradients
            ]
        )
        reweight = reliability.to(distance.dtype) / distance.clamp_min(smoothing)
        reweight = reweight / reweight.sum().clamp_min(smoothing)
        updated = gradient_linear_combination(gradients, reweight)
        delta = _subtract(updated, estimate)
        residual = torch.sqrt(gradient_inner(delta, delta).clamp_min(0.0))
        estimate = updated
    return estimate, residual


def project_to_consensus_cone(
    gradient: Gradient,
    axis: Gradient,
    minimum_cosine: float,
    epsilon: float,
) -> tuple[Gradient, Tensor]:
    """Exact Euclidean projection onto a circular cone around ``axis``."""

    axis_norm = torch.sqrt(gradient_inner(axis, axis).clamp_min(0.0))
    gradient_norm = torch.sqrt(gradient_inner(gradient, gradient).clamp_min(0.0))
    if bool(axis_norm <= epsilon) or bool(gradient_norm <= epsilon):
        return gradient, gradient_norm.new_tensor(True)
    unit_axis = gradient_linear_combination((axis,), axis_norm.reciprocal()[None])
    axial = gradient_inner(gradient, unit_axis)
    orthogonal = _subtract(
        gradient,
        gradient_linear_combination((unit_axis,), axial[None]),
    )
    radial = torch.sqrt(gradient_inner(orthogonal, orthogonal).clamp_min(0.0))
    cosine = axial / gradient_norm.clamp_min(epsilon)
    if bool(cosine >= minimum_cosine):
        return gradient, cosine.new_tensor(True)
    tangent = torch.sqrt(
        cosine.new_tensor(1.0 - minimum_cosine**2)
    ) / cosine.new_tensor(minimum_cosine).clamp_min(epsilon)
    boundary_axial = (axial + tangent * radial) / (1.0 + tangent.square())
    if bool(boundary_axial <= 0):
        zero = gradient_linear_combination((gradient,), axial.new_zeros(1))
        return zero, cosine.new_tensor(False)
    boundary_radial = tangent * boundary_axial
    if bool(radial <= epsilon):
        projected = gradient_linear_combination((unit_axis,), boundary_axial[None])
    else:
        projected = gradient_linear_combination(
            (unit_axis, orthogonal),
            torch.stack((boundary_axial, boundary_radial / radial)),
        )
    return projected, cosine.new_tensor(False)


def principal_subspace_projection(
    vector: Gradient,
    samples: Sequence[Gradient],
    reliability: Tensor,
    relative_singular_value: float,
    epsilon: float,
) -> tuple[Gradient, int]:
    """Project through the left singular vectors using only an M by M Gram matrix."""

    if not samples:
        return vector, 0
    gram = _gram(samples, reliability)
    eigenvalue, eigenvector = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalue, descending=True)
    eigenvalue = eigenvalue[order].clamp_min(0.0)
    eigenvector = eigenvector[:, order]
    if bool(eigenvalue[0] <= epsilon):
        zero = gradient_linear_combination((vector,), eigenvalue.new_zeros(1))
        return zero, 0
    keep = eigenvalue > eigenvalue[0] * relative_singular_value**2
    rank = int(keep.sum().item())
    root_weight = torch.sqrt(reliability.to(gram.dtype).clamp_min(0.0))
    basis: list[Gradient] = []
    for component in range(rank):
        coefficients = (
            root_weight * eigenvector[:, component]
            / torch.sqrt(eigenvalue[component].clamp_min(epsilon))
        )
        basis.append(gradient_linear_combination(samples, coefficients))
    coordinates = torch.stack([gradient_inner(item, vector) for item in basis])
    return gradient_linear_combination(basis, coordinates), rank


class HilbertGradientPurifier:
    """Stateful Fisher-normalized multiview gradient purifier.

    The Fisher state is a diagonal empirical second moment.  Pending moments
    are committed only at optimizer boundaries so gradient accumulation uses a
    single preconditioner.  ``commit_fisher`` accepts an all-reduce callback,
    making the state identical on every DDP rank before a checkpoint is saved.
    """

    def __init__(
        self,
        parameters: Sequence[nn.Parameter],
        config: GradientPurificationConfig = GradientPurificationConfig(),
    ) -> None:
        self.parameters = tuple(parameters)
        self.config = config
        self.fisher = [
            torch.zeros_like(
                parameter,
                dtype=(torch.float64 if parameter.dtype == torch.float64 else torch.float32),
                memory_format=torch.preserve_format,
            )
            for parameter in self.parameters
        ]
        self.pending_fisher = [torch.zeros_like(value) for value in self.fisher]
        self.pending_count = 0
        self.committed_steps = 0

    def purify(
        self,
        view_gradients: Sequence[Gradient],
        reliability: Tensor,
        artifact_gradients: Optional[Sequence[Gradient]] = None,
    ) -> tuple[Gradient, GradientPurificationDiagnostics]:
        if len(view_gradients) < 2:
            raise ValueError("gradient purification needs at least two retained views")
        if reliability.shape != (len(view_gradients),):
            raise ValueError("reliability must have one scalar per view gradient")
        reliability = reliability.detach().to(
            device=_first_tensor(view_gradients[0]).device,
            dtype=_accumulation_dtype(_first_tensor(view_gradients[0])),
        )
        reliability = reliability.clamp_min(self.config.minimum_reliability)
        median, median_residual = weighted_geometric_median(
            view_gradients,
            reliability,
            self.config.weiszfeld_iterations,
            self.config.weiszfeld_smoothing,
        )
        projected = []
        accepted = []
        for gradient in view_gradients:
            value, inside = project_to_consensus_cone(
                gradient,
                median,
                self.config.consensus_cosine,
                self.config.weiszfeld_smoothing,
            )
            projected.append(value)
            accepted.append(inside.to(reliability.dtype))
        weighted_mean = gradient_linear_combination(
            projected,
            reliability / reliability.sum().clamp_min(self.config.weiszfeld_smoothing),
        )
        consensus, consensus_rank = principal_subspace_projection(
            weighted_mean,
            projected,
            reliability,
            self.config.consensus_relative_singular_value,
            self.config.weiszfeld_smoothing,
        )
        artifact_rank = 0
        clean = consensus
        if artifact_gradients:
            artifact_weight = reliability.new_ones(len(artifact_gradients))
            artifact_component, artifact_rank = principal_subspace_projection(
                consensus,
                artifact_gradients,
                artifact_weight,
                self.config.artifact_relative_singular_value,
                self.config.weiszfeld_smoothing,
            )
            clean = _subtract(consensus, artifact_component)

        normalized_reliability = reliability / reliability.sum().clamp_min(
            self.config.weiszfeld_smoothing
        )
        moment: list[Tensor] = []
        for parameter_index, parameter in enumerate(self.parameters):
            value = torch.zeros_like(parameter)
            for view_index, gradient in enumerate(projected):
                component = gradient[parameter_index]
                if component is not None:
                    value.add_(
                        normalized_reliability[view_index].to(value.dtype)
                        * component.detach().to(value.dtype).square()
                    )
            moment.append(value)
            self.pending_fisher[parameter_index].add_(value)
        self.pending_count += 1

        candidate_fisher = [
            self.config.fisher_decay * old
            + (1.0 - self.config.fisher_decay) * current
            for old, current in zip(self.fisher, moment)
        ]
        fisher_norm_square = reliability.new_zeros(())
        for component, fisher in zip(clean, candidate_fisher):
            if component is None:
                continue
            fisher_norm_square = fisher_norm_square + torch.sum(
                component.to(fisher_norm_square.dtype).square()
                / (fisher.to(fisher_norm_square.dtype) + self.config.fisher_damping)
            )
        fisher_norm = torch.sqrt(fisher_norm_square.clamp_min(0.0))
        fisher_scale = torch.clamp(
            fisher_norm.new_tensor(self.config.fisher_radius)
            / fisher_norm.clamp_min(self.config.weiszfeld_smoothing),
            max=1.0,
        )
        final = gradient_linear_combination((clean,), fisher_scale[None])
        diagnostics = GradientPurificationDiagnostics(
            retained_views=len(view_gradients),
            consensus_rank=consensus_rank,
            artifact_rank=artifact_rank,
            cone_acceptance_fraction=torch.stack(accepted).mean(),
            median_residual=median_residual,
            fisher_norm=fisher_norm,
            fisher_scale=fisher_scale,
        )
        return final, diagnostics

    @torch.no_grad()
    def commit_fisher(
        self,
        all_reduce_mean: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        if self.pending_count == 0:
            return
        for index, pending in enumerate(self.pending_fisher):
            moment = pending / float(self.pending_count)
            if all_reduce_mean is not None:
                moment = all_reduce_mean(moment)
            self.fisher[index].mul_(self.config.fisher_decay).add_(
                moment, alpha=1.0 - self.config.fisher_decay
            )
            pending.zero_()
        self.pending_count = 0
        self.committed_steps += 1

    def state_dict(self) -> dict[str, object]:
        if self.pending_count:
            raise RuntimeError("purifier state is checkpointable only at an optimizer boundary")
        return {
            "config": self.config.__dict__.copy(),
            "fisher": [value.detach().cpu() for value in self.fisher],
            "committed_steps": self.committed_steps,
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("config") != self.config.__dict__:
            raise ValueError("gradient-purifier checkpoint configuration differs")
        fisher = state.get("fisher")
        if not isinstance(fisher, list) or len(fisher) != len(self.fisher):
            raise ValueError("gradient-purifier Fisher state has incompatible length")
        for target, source in zip(self.fisher, fisher):
            source_tensor = torch.as_tensor(source)
            if source_tensor.shape != target.shape:
                raise ValueError("gradient-purifier Fisher tensor shape differs")
            target.copy_(source_tensor.to(device=target.device, dtype=target.dtype))
        self.committed_steps = int(state.get("committed_steps", 0))


__all__ = [
    "Gradient",
    "GradientPurificationConfig",
    "GradientPurificationDiagnostics",
    "HilbertGradientPurifier",
    "gradient_inner",
    "gradient_linear_combination",
    "principal_subspace_projection",
    "project_to_consensus_cone",
    "weighted_geometric_median",
]

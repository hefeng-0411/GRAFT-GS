"""Deterministic supervision operators derived from audited surface geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..manifold.barrier import BarrierConfig, BarrierProjector, FeasibilityReport
from ..manifold.geometry import ManifoldState, ManifoldTangent


@dataclass(frozen=True)
class SurfaceTargetConfig:
    screening_weight: float = 0.25
    conjugate_gradient_iterations: int = 64
    conjugate_gradient_tolerance: float = 1.0e-8
    nearest_neighbor_chunk_size: int = 2048

    def __post_init__(self) -> None:
        if self.screening_weight < 0:
            raise ValueError("screening_weight must be non-negative")
        if self.conjugate_gradient_iterations < 1 or self.nearest_neighbor_chunk_size < 1:
            raise ValueError("iteration and chunk counts must be positive")
        if self.conjugate_gradient_tolerance <= 0:
            raise ValueError("conjugate-gradient tolerance must be positive")


def nearest_surface_points(query: Tensor, surface: Tensor, chunk_size: int = 2048) -> Tensor:
    """Exact chunked nearest samples on an explicitly voxelized surface."""

    if query.ndim != 2 or surface.ndim != 2 or query.shape[1] != 3 or surface.shape[1] != 3:
        raise ValueError("query and surface must have shapes [V,3] and [N,3]")
    if surface.shape[0] == 0:
        raise ValueError("surface target cannot be empty")
    selected: list[Tensor] = []
    for start in range(0, query.shape[0], chunk_size):
        distance = torch.cdist(query[start : start + chunk_size], surface)
        selected.append(surface[distance.argmin(dim=1)])
    return torch.cat(selected, dim=0)


def _graph_laplacian(value: Tensor, edges: Tensor) -> Tensor:
    output = torch.zeros_like(value)
    if edges.numel() == 0:
        return output
    source, target = edges.unbind(-1)
    delta = value[source] - value[target]
    output.index_add_(0, source, delta)
    output.index_add_(0, target, -delta)
    return output


def screened_surface_projection(
    reference: Tensor,
    nearest: Tensor,
    edges: Tensor,
    config: SurfaceTargetConfig = SurfaceTargetConfig(),
) -> Tensor:
    r"""Solve the screened deformation ``(I + lambda L)x=q+lambda Lp``.

    This is the unique minimizer of a surface attraction plus a topology-fixed
    edge-vector preservation energy.  Matrix-free conjugate gradients avoid a
    dense vertex Laplacian.
    """

    if reference.shape != nearest.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference and nearest must both have shape [V,3]")
    weight = config.screening_weight
    if weight == 0 or edges.numel() == 0:
        return nearest
    right = nearest + weight * _graph_laplacian(reference, edges)

    def operator(value: Tensor) -> Tensor:
        return value + weight * _graph_laplacian(value, edges)

    estimate = reference.clone()
    residual = right - operator(estimate)
    direction = residual.clone()
    residual_norm = torch.sum(residual * residual)
    initial_norm = residual_norm.clamp_min(torch.finfo(reference.dtype).tiny)
    for _ in range(config.conjugate_gradient_iterations):
        applied = operator(direction)
        step = residual_norm / torch.sum(direction * applied).clamp_min(
            torch.finfo(reference.dtype).tiny
        )
        estimate = estimate + step * direction
        next_residual = residual - step * applied
        next_norm = torch.sum(next_residual * next_residual)
        if float((next_norm / initial_norm).detach()) <= config.conjugate_gradient_tolerance**2:
            residual = next_residual
            break
        direction = next_residual + (
            next_norm / residual_norm.clamp_min(torch.finfo(reference.dtype).tiny)
        ) * direction
        residual, residual_norm = next_residual, next_norm
    return estimate


def _detached_state(state: ManifoldState) -> ManifoldState:
    return ManifoldState(
        position=state.position.detach(),
        rotation=state.rotation.detach(),
        covariance=state.covariance.detach(),
        opacity_logit=state.opacity_logit.detach(),
        appearance=state.appearance.detach(),
        latent=state.latent.detach(),
        evidence_metric=state.evidence_metric.detach(),
        complex=state.complex,
    )


def derive_feasible_surface_target(
    initial: ManifoldState,
    surface: Tensor,
    barrier_config: BarrierConfig,
    config: SurfaceTargetConfig = SurfaceTargetConfig(),
) -> tuple[ManifoldState, FeasibilityReport]:
    """Construct a detached, hard-feasible phase-C target on one stratum."""

    state = _detached_state(initial)
    surface = surface.to(device=state.position.device, dtype=state.position.dtype).detach()
    with torch.no_grad():
        nearest = nearest_surface_points(
            state.position,
            surface,
            chunk_size=config.nearest_neighbor_chunk_size,
        )
        screened = screened_surface_projection(
            state.position,
            nearest,
            state.complex.edges,
            config,
        )
        displacement = screened - state.position
    proposal = ManifoldTangent(
        position=displacement,
        rotation_body=torch.zeros_like(state.position),
        covariance=torch.zeros_like(state.covariance),
        opacity_logit=torch.zeros_like(state.opacity_logit),
        appearance=torch.zeros_like(state.appearance),
        latent=torch.zeros_like(state.latent),
    )
    with torch.no_grad():
        projector = BarrierProjector(state, barrier_config)
        projected, _ = projector.project(state, proposal)
        target, report = projector.retract_with_backtracking(state, projected, requested_step=1.0)
    if not report.feasible:
        raise RuntimeError("derived surface target failed its feasibility certificate")
    return _detached_state(target), report


__all__ = [
    "SurfaceTargetConfig",
    "derive_feasible_surface_target",
    "nearest_surface_points",
    "screened_surface_projection",
]

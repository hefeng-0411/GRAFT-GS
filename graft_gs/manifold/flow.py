"""Conditional Riemannian flow matching and topology-fixed Heun integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..equivariant.gsta import GSTAConfig, GaugeCovariantSparseTransportAttention, IrrepTensor, MultiplicityLinear, l2_to_matrix
from ..geometry.atlas import PersistentOctreeAtlas
from .geometry import ManifoldState, ManifoldTangent, geodesic_interpolate, product_metric_squared, retract


@dataclass(frozen=True)
class FlowConfig:
    layers: int = 8
    steps: int = 8
    appearance_channels: int = 48
    time_frequencies: int = 8
    residual_scale: float = 0.1
    spectral_bound: float = 1.0

    def __post_init__(self) -> None:
        if self.layers < 1 or self.steps < 1 or self.time_frequencies < 1:
            raise ValueError("flow depth, integration steps, and time frequencies must be positive")
        if self.appearance_channels < 1:
            raise ValueError("appearance channels must be positive")
        if self.residual_scale <= 0 or self.spectral_bound <= 0:
            raise ValueError("flow residual and spectral bounds must be positive")


class RiemannianVectorField(nn.Module):
    """Connection-attention vector field with manifold-valued output heads."""

    def __init__(self, config: FlowConfig = FlowConfig(), attention: GSTAConfig = GSTAConfig()) -> None:
        super().__init__()
        self.config = config
        self.attention_config = attention
        self.layers = nn.ModuleList([GaugeCovariantSparseTransportAttention(attention) for _ in range(config.layers)])
        for layer in self.layers:
            for child in layer.modules():
                if isinstance(child, MultiplicityLinear):
                    nn.utils.parametrizations.spectral_norm(child, name="weight", n_power_iterations=1)
                    child.set_operator_scale(config.spectral_bound)
        self.time_projection = nn.Linear(2 * config.time_frequencies, attention.scalar_channels, bias=False)
        self.position_head = nn.Parameter(torch.zeros(attention.vector_channels))
        self.rotation_head = nn.Parameter(torch.zeros(attention.vector_channels))
        self.covariance_scalar_head = nn.Parameter(torch.zeros(attention.scalar_channels, 3))
        self.covariance_tensor_head = nn.Parameter(torch.zeros(attention.tensor_channels))
        self.opacity_head = nn.Linear(attention.scalar_channels, 1)
        self.appearance_head = nn.Linear(attention.scalar_channels, config.appearance_channels)
        self.latent_head = nn.Linear(attention.scalar_channels, 128)
        self._zero_residual_heads()

    def _zero_residual_heads(self) -> None:
        for module in (self.opacity_head, self.appearance_head, self.latent_head):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def _time_embedding(self, time: Tensor, dtype: torch.dtype, device: torch.device) -> Tensor:
        frequency = (2.0 ** torch.arange(self.config.time_frequencies, dtype=dtype, device=device)) * torch.pi
        phase = time.reshape(-1, 1) * frequency
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)

    def forward(self, atlas: PersistentOctreeAtlas, state: ManifoldState, time: Tensor | float) -> ManifoldTangent:
        fields = IrrepTensor.from_packed(
            state.latent,
            self.attention_config.scalar_channels,
            self.attention_config.vector_channels,
            self.attention_config.tensor_channels,
        )
        time_tensor = torch.as_tensor(time, dtype=state.position.dtype, device=state.position.device).reshape(1)
        fields.scalar = fields.scalar + self.time_projection(self._time_embedding(time_tensor, state.position.dtype, state.position.device)).expand_as(fields.scalar)
        for layer in self.layers:
            fields = layer(
                atlas,
                fields,
                node_index=state.complex.atlas_node_index,
                local_edge_index=state.complex.edges.transpose(0, 1),
                centers=state.position,
                frames=state.rotation,
            )
        position_local = torch.einsum("c,vcj->vj", self.position_head, fields.vector)
        rotation_body = torch.einsum("c,vcj->vj", self.rotation_head, fields.vector)
        position_world = torch.einsum("vij,vj->vi", state.rotation, position_local)
        diagonal_local = torch.einsum("vc,cd->vd", fields.scalar, self.covariance_scalar_head)
        tensor_local = torch.einsum("c,vcij->vij", self.covariance_tensor_head, l2_to_matrix(fields.tensor))
        covariance_local = torch.diag_embed(diagonal_local) + tensor_local
        covariance_world = state.rotation @ covariance_local @ state.rotation.transpose(-1, -2)
        scalar = fields.scalar
        scale = self.config.residual_scale
        return ManifoldTangent(
            position=scale * position_world,
            rotation_body=scale * rotation_body,
            covariance=scale * covariance_world,
            opacity_logit=scale * self.opacity_head(scalar),
            appearance=scale * self.appearance_head(scalar),
            latent=scale * self.latent_head(scalar),
        )


class RiemannianFlowMatcher(nn.Module):
    def __init__(self, vector_field: RiemannianVectorField) -> None:
        super().__init__()
        self.vector_field = vector_field

    def forward(self, atlas: PersistentOctreeAtlas, start: ManifoldState, target: ManifoldState, time: Tensor) -> Tensor:
        interpolated, conditional_velocity = geodesic_interpolate(start, target, time)
        predicted = self.vector_field(atlas, interpolated, time)
        error = ManifoldTangent(
            position=predicted.position - conditional_velocity.position,
            rotation_body=predicted.rotation_body - conditional_velocity.rotation_body,
            covariance=predicted.covariance - conditional_velocity.covariance,
            opacity_logit=predicted.opacity_logit - conditional_velocity.opacity_logit,
            appearance=predicted.appearance - conditional_velocity.appearance,
            latent=predicted.latent - conditional_velocity.latent,
        )
        return product_metric_squared(interpolated, error) / interpolated.position.shape[0]


class SafeHeunIntegrator:
    """Explicit Heun solver with manifold retractions and feasibility hooks."""

    def __init__(self, steps: int = 8) -> None:
        if steps < 1:
            raise ValueError("steps must be positive")
        self.steps = steps

    def integrate(
        self,
        field: RiemannianVectorField,
        atlas: PersistentOctreeAtlas,
        initial: ManifoldState,
        projector: Optional[object] = None,
    ) -> Tuple[ManifoldState, list[object]]:
        state = initial
        reports: list[object] = []
        dt = 1.0 / self.steps
        for step_index in range(self.steps):
            time = step_index / self.steps
            first = field(atlas, state, time)
            if projector is not None:
                first, report = projector.project(state, first)
                reports.append(report)
                predictor, predictor_report = projector.retract_with_backtracking(
                    state,
                    first,
                    dt,
                )
                reports.append(predictor_report)
            else:
                predictor = retract(state, first, dt)
            second = field(atlas, predictor, time + dt)
            average = first.add(second).scaled(0.5)
            if projector is not None:
                average, report = projector.project(state, average)
                state, accepted_report = projector.retract_with_backtracking(state, average, dt)
                reports.extend((report, accepted_report))
            else:
                state = retract(state, average, dt)
        return state, reports


__all__ = ["FlowConfig", "RiemannianFlowMatcher", "RiemannianVectorField", "SafeHeunIntegrator"]

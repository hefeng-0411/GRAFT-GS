"""Connection-aware gauge-covariant sparse transport attention.

Irreducible fields are stored in local chart coordinates.  A local gauge change
``R_i -> R_i S_i`` acts as

``v_i -> S_i^T v_i`` and ``T_i -> S_i^T T_i S_i``.

The connection ``P_ji = R_i^T R_j`` transports fields from chart ``j`` to
chart ``i``.  The implementation represents l=2 fields as symmetric-traceless
3x3 tensors internally, which is exactly equivalent to a real five-dimensional
Wigner representation and makes covariance tests direct.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..geometry.atlas import PersistentOctreeAtlas


@dataclass(frozen=True)
class GSTAConfig:
    scalar_channels: int = 60
    vector_channels: int = 16
    tensor_channels: int = 4
    heads: int = 4
    radial_basis: int = 8
    radial_cutoff_factor: float = 3.0
    attention_temperature_scalar: float = 1.0
    attention_temperature_vector: float = 1.0
    attention_temperature_tensor: float = 1.0
    ot_bias_weight: float = 0.25
    uncertainty_bias_weight: float = 0.25
    residual_step: float = 0.1
    epsilon: float = 1.0e-8
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.heads < 1:
            raise ValueError("attention heads must be positive")
        for channels in (self.scalar_channels, self.vector_channels, self.tensor_channels):
            if channels < 1:
                raise ValueError("every irrep multiplicity must be positive")
            if channels % self.heads:
                raise ValueError("every irrep multiplicity must be divisible by heads")
        if self.radial_basis < 1:
            raise ValueError("radial basis size must be positive")
        if min(
            self.attention_temperature_scalar,
            self.attention_temperature_vector,
            self.attention_temperature_tensor,
            self.radial_cutoff_factor,
            self.epsilon,
        ) <= 0:
            raise ValueError("attention temperatures and numerical scales must be positive")
        if self.residual_step < 0:
            raise ValueError("attention residual step must be non-negative")
        if self.ot_bias_weight < 0 or self.uncertainty_bias_weight < 0:
            raise ValueError("transport and uncertainty penalties must be non-negative")
        if not isinstance(self.activation_checkpointing, bool):
            raise TypeError("GSTA activation_checkpointing must be Boolean")


@dataclass
class IrrepTensor:
    """Multiplicity-major local irreducible fields."""

    scalar: Tensor  # [V,C0]
    vector: Tensor  # [V,C1,3]
    tensor: Tensor  # [V,C2,5]

    @classmethod
    def from_packed(cls, packed: Tensor, scalar_channels: int = 60, vector_channels: int = 16, tensor_channels: int = 4) -> "IrrepTensor":
        expected = scalar_channels + 3 * vector_channels + 5 * tensor_channels
        if packed.ndim != 2 or packed.shape[-1] != expected:
            raise ValueError(f"packed irreps must have shape [V,{expected}]")
        scalar = packed[:, :scalar_channels]
        offset = scalar_channels
        vector = packed[:, offset : offset + 3 * vector_channels].reshape(-1, vector_channels, 3)
        offset += 3 * vector_channels
        tensor = packed[:, offset:].reshape(-1, tensor_channels, 5)
        return cls(scalar, vector, tensor)

    def pack(self) -> Tensor:
        return torch.cat((self.scalar, self.vector.flatten(1), self.tensor.flatten(1)), dim=-1)


def l2_to_matrix(value: Tensor) -> Tensor:
    """Map orthonormal real l=2 coordinates to symmetric-traceless matrices."""

    xy, yz, zz, xz, xx_minus_yy = value.unbind(-1)
    root3 = sqrt(3.0)
    a = -zz / 3.0 + xx_minus_yy / root3
    b = -zz / 3.0 - xx_minus_yy / root3
    c = 2.0 * zz / 3.0
    return torch.stack(
        (
            torch.stack((a, xy / root3, xz / root3), dim=-1),
            torch.stack((xy / root3, b, yz / root3), dim=-1),
            torch.stack((xz / root3, yz / root3, c), dim=-1),
        ),
        dim=-2,
    )


def matrix_to_l2(matrix: Tensor) -> Tensor:
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0
    matrix = matrix - trace[..., None, None] * torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    root3 = sqrt(3.0)
    return torch.stack(
        (
            root3 * matrix[..., 0, 1],
            root3 * matrix[..., 1, 2],
            1.5 * matrix[..., 2, 2],
            root3 * matrix[..., 0, 2],
            0.5 * root3 * (matrix[..., 0, 0] - matrix[..., 1, 1]),
        ),
        dim=-1,
    )


def direction_l2(direction: Tensor) -> Tensor:
    eye = torch.eye(3, dtype=direction.dtype, device=direction.device)
    tensor = direction[..., :, None] * direction[..., None, :] - eye / 3.0
    return matrix_to_l2(tensor)


def symmetric_traceless_outer(left: Tensor, right: Tensor) -> Tensor:
    product = 0.5 * (left[..., :, None] * right[..., None, :] + right[..., :, None] * left[..., None, :])
    trace = torch.sum(left * right, dim=-1) / 3.0
    eye = torch.eye(3, dtype=left.dtype, device=left.device)
    return product - trace[..., None, None] * eye


def segment_sum(values: Tensor, index: Tensor, size: int) -> Tensor:
    output = values.new_zeros((size, *values.shape[1:]))
    if values.numel():
        output.index_add_(0, index, values)
    return output


def segment_softmax(logits: Tensor, index: Tensor, size: int) -> Tensor:
    """Softmax independently for every source and head."""

    maximum = logits.new_full((size, logits.shape[-1]), -torch.inf)
    expanded = index[:, None].expand_as(logits)
    maximum.scatter_reduce_(0, expanded, logits, reduce="amax", include_self=True)
    exponential = torch.exp(logits - maximum[index])
    denominator = segment_sum(exponential, index, size)
    return exponential / denominator[index].clamp_min(torch.finfo(logits.dtype).tiny)


def normalized_inner_product(left: Tensor, right: Tensor, epsilon: float) -> Tensor:
    """Return the exact cosine numerator/denominator without normalized copies.

    ``F.normalize(left) * F.normalize(right)`` materializes two tensors with
    the complete edge/channel cardinality before their reduction.  The
    algebraically identical quotient below lets einsum fuse the dot-product
    reduction and retains only per-edge norms.  It deliberately does not use
    ``out=`` or mutate an autograd input.
    """

    if left.shape != right.shape or left.ndim < 1:
        raise ValueError("normalized inner-product operands must share a shape")
    if epsilon <= 0:
        raise ValueError("normalized inner-product epsilon must be positive")
    numerator = torch.einsum("...d,...d->...", left, right)
    left_norm = torch.linalg.vector_norm(left, dim=-1).clamp_min(epsilon)
    right_norm = torch.linalg.vector_norm(right, dim=-1).clamp_min(epsilon)
    return numerator / (left_norm * right_norm)


class MultiplicityLinear(nn.Module):
    """Linear map on multiplicities only; magnetic components are untouched."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.weight = nn.Parameter(torch.eye(channels))
        # Plain scalar policy, intentionally not checkpoint state: the value is
        # reconstructed from the phase/model configuration while the spectral
        # parametrization owns the trainable weight representation.
        self.operator_scale = 1.0

    def set_operator_scale(self, value: float) -> None:
        if value <= 0:
            raise ValueError("multiplicity operator scale must be positive")
        self.operator_scale = float(value)

    def forward(self, value: Tensor) -> Tensor:
        return self.forward_with_weight(value, self.weight)

    def forward_with_weight(self, value: Tensor, weight: Tensor) -> Tensor:
        """Apply one already-resolved effective multiplicity weight.

        The pipeline parametrizes ``weight`` with spectral normalization.  A
        checkpoint recomputation must reuse the effective weight from the
        original forward, because the parametrization's power-iteration
        buffers can advance again for later objects before backward starts.
        """

        if weight.shape != (self.channels, self.channels):
            raise ValueError("effective multiplicity weight has an invalid shape")
        return self.operator_scale * torch.einsum("oi,vi...->vo...", weight, value)


class CompactRadialBasis(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.count = count

    def forward(self, normalized_radius: Tensor) -> Tensor:
        centers = torch.linspace(0.0, 1.0, self.count, dtype=normalized_radius.dtype, device=normalized_radius.device)
        width = 1.0 / max(1, self.count - 1)
        x = (normalized_radius[:, None] - centers).abs() / width
        # Compact C2 Wendland basis, exactly zero outside one knot radius.
        one_minus = (1.0 - x).clamp_min(0.0)
        value = one_minus.pow(4) * (4.0 * x + 1.0)
        return value / value.sum(-1, keepdim=True).clamp_min(torch.finfo(value.dtype).eps)


def active_adjacency(atlas: PersistentOctreeAtlas) -> Tuple[Tensor, Tensor]:
    """Map persistent global node edges to contiguous active-chart indices."""

    active = atlas.active_indices
    global_to_local = torch.full((atlas.num_nodes,), -1, dtype=torch.int64, device=active.device)
    global_to_local[active] = torch.arange(active.numel(), device=active.device)
    source = global_to_local[atlas.edge_index[0]]
    target = global_to_local[atlas.edge_index[1]]
    valid = (source >= 0) & (target >= 0)
    source, target = source[valid], target[valid]
    # Self edges guarantee a well-defined update for isolated components.
    self_index = torch.arange(active.numel(), device=active.device)
    edge = torch.cat((torch.stack((source, target)), torch.stack((self_index, self_index))), dim=1)
    linear = torch.unique(edge[0] * active.numel() + edge[1], sorted=True)
    edge = torch.stack((torch.div(linear, active.numel(), rounding_mode="floor"), linear.remainder(active.numel())))
    return edge, active


class GaugeCovariantSparseTransportAttention(nn.Module):
    """Sparse connection attention with parity-valid l<=2 tensor products."""

    def __init__(self, config: GSTAConfig = GSTAConfig()) -> None:
        super().__init__()
        self.config = config
        self.q0, self.k0, self.v0 = (MultiplicityLinear(config.scalar_channels) for _ in range(3))
        self.q1, self.k1, self.v1 = (MultiplicityLinear(config.vector_channels) for _ in range(3))
        self.q2, self.k2, self.v2 = (MultiplicityLinear(config.tensor_channels) for _ in range(3))
        self.radial_basis = CompactRadialBasis(config.radial_basis)
        # Each path coefficient is a learned expansion in a compact radial basis.
        self.score_radial = nn.Parameter(torch.zeros(config.heads, config.radial_basis))
        self.path_00 = nn.Parameter(torch.zeros(config.scalar_channels, config.radial_basis))
        self.path_11_to_0 = nn.Parameter(torch.zeros(config.scalar_channels, config.radial_basis))
        self.path_22_to_0 = nn.Parameter(torch.zeros(config.scalar_channels, config.radial_basis))
        self.path_01_to_1 = nn.Parameter(torch.zeros(config.vector_channels, config.radial_basis))
        self.path_11 = nn.Parameter(torch.zeros(config.vector_channels, config.radial_basis))
        self.path_21_to_1 = nn.Parameter(torch.zeros(config.vector_channels, config.radial_basis))
        self.path_02_to_2 = nn.Parameter(torch.zeros(config.tensor_channels, config.radial_basis))
        self.path_11_to_2 = nn.Parameter(torch.zeros(config.tensor_channels, config.radial_basis))
        self.path_22 = nn.Parameter(torch.zeros(config.tensor_channels, config.radial_basis))
        self.gate0 = nn.Linear(config.scalar_channels, config.scalar_channels)
        self.gate1 = nn.Linear(config.scalar_channels, config.vector_channels)
        self.gate2 = nn.Linear(config.scalar_channels, config.tensor_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # Identity same-irrep paths start the layer as a stable local smoother;
            # cross-order tensor products are learned from zero.
            self.path_00[:, 0] = 1.0
            self.path_11[:, 0] = 1.0
            self.path_22[:, 0] = 1.0
            nn.init.zeros_(self.gate0.weight)
            nn.init.zeros_(self.gate1.weight)
            nn.init.zeros_(self.gate2.weight)
            nn.init.constant_(self.gate0.bias, -2.0)
            nn.init.constant_(self.gate1.bias, -2.0)
            nn.init.constant_(self.gate2.bias, -2.0)

    @staticmethod
    def _radial_coeff(coefficients: Tensor, basis: Tensor) -> Tensor:
        return basis @ coefficients.transpose(0, 1)

    def forward(
        self,
        atlas: PersistentOctreeAtlas,
        fields: IrrepTensor | Tensor,
        edge_ot_cost: Optional[Tensor] = None,
        edge_uncertainty: Optional[Tensor] = None,
        node_index: Optional[Tensor] = None,
        local_edge_index: Optional[Tensor] = None,
        centers: Optional[Tensor] = None,
        frames: Optional[Tensor] = None,
    ) -> IrrepTensor:
        """Apply sparse gauge attention with optional exact recomputation.

        The discrete graph is prepared once outside the checkpoint.  All
        differentiable geometric tensors and the effective spectrally
        normalized weights are explicit checkpoint inputs, so backward
        reconstructs the same numerical function even when this layer has
        processed later objects before recomputation begins.
        """

        cfg = self.config
        if isinstance(fields, Tensor):
            fields = IrrepTensor.from_packed(
                fields,
                cfg.scalar_channels,
                cfg.vector_channels,
                cfg.tensor_channels,
            )
        if node_index is None:
            edge, active = active_adjacency(atlas)
        else:
            active = node_index
            if local_edge_index is None:
                raise ValueError("local_edge_index is required with an explicit node subset")
            if local_edge_index.shape[0] != 2:
                local_edge_index = local_edge_index.transpose(0, 1)
            forward_edge = local_edge_index
            reverse_edge = local_edge_index.flip(0)
            self_index = torch.arange(active.numel(), device=active.device)
            edge = torch.cat((forward_edge, reverse_edge, torch.stack((self_index, self_index))), dim=1)
            linear = torch.unique(edge[0] * active.numel() + edge[1], sorted=True)
            edge = torch.stack((torch.div(linear, active.numel(), rounding_mode="floor"), linear.remainder(active.numel())))
        v = active.numel()
        if fields.scalar.shape != (v, cfg.scalar_channels):
            raise ValueError("scalar field does not match active atlas")
        centers = atlas.chart_centers[active] if centers is None else centers
        frames = atlas.chart_frames[active] if frames is None else frames
        if centers.shape != (v, 3) or frames.shape != (v, 3, 3):
            raise ValueError("dynamic centers/frames must match the selected chart subset")
        cell_sides = atlas.cell_sides[active]
        multiplicity_modules = (
            self.q0,
            self.k0,
            self.v0,
            self.q1,
            self.k1,
            self.v1,
            self.q2,
            self.k2,
            self.v2,
        )
        # Resolve each spectral parametrization once in the original forward.
        # These small effective matrices remain ordinary differentiable inputs
        # to the checkpoint; recomputation never advances a power iteration.
        multiplicity_weights = tuple(
            module.weight for module in multiplicity_modules
        )
        arguments = (
            fields.scalar,
            fields.vector,
            fields.tensor,
            edge,
            centers,
            frames,
            cell_sides,
            edge_ot_cost,
            edge_uncertainty,
            *multiplicity_weights,
        )
        if (
            cfg.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            output = checkpoint(
                self._forward_prepared,
                *arguments,
                use_reentrant=False,
                # GSTA contains no stochastic operator. Avoiding RNG snapshots
                # removes CPU/CUDA state copies while preserving exact values.
                preserve_rng_state=False,
            )
        else:
            output = self._forward_prepared(*arguments)
        return IrrepTensor(*output)

    def _forward_prepared(
        self,
        scalar: Tensor,
        vector: Tensor,
        tensor: Tensor,
        edge: Tensor,
        centers: Tensor,
        frames: Tensor,
        cell_sides: Tensor,
        edge_ot_cost: Optional[Tensor],
        edge_uncertainty: Optional[Tensor],
        q0_weight: Tensor,
        k0_weight: Tensor,
        v0_weight: Tensor,
        q1_weight: Tensor,
        k1_weight: Tensor,
        v1_weight: Tensor,
        q2_weight: Tensor,
        k2_weight: Tensor,
        v2_weight: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Differentiable prepared-graph kernel used by forward/recompute."""

        cfg = self.config
        fields = IrrepTensor(scalar, vector, tensor)
        source, target = edge
        v = scalar.shape[0]
        displacement_world = centers[target] - centers[source]
        displacement = torch.einsum("eji,ej->ei", frames[source], displacement_world)
        connection = frames[source].transpose(-1, -2) @ frames[target]
        cutoff = cfg.radial_cutoff_factor * torch.maximum(
            cell_sides[source], cell_sides[target]
        )
        # Self edges are required by the sparse attention contract and have
        # exactly zero displacement.  Use a cutoff-relative smooth norm so the
        # direction has the finite zero subgradient at that gauge-degenerate
        # point; raw vector_norm/clamp leaves an undefined norm derivative.
        direction_floor = cutoff * torch.finfo(displacement.dtype).eps**0.5
        safe_distance = torch.sqrt(
            displacement.square().sum(-1) + direction_floor.square()
        )
        distance = safe_distance - direction_floor
        direction = displacement / safe_distance[:, None]
        radial = self.radial_basis((distance / cutoff.clamp_min(cfg.epsilon)).clamp(0.0, 1.0))

        q0 = self.q0.forward_with_weight(fields.scalar, q0_weight)
        k0 = self.k0.forward_with_weight(fields.scalar, k0_weight)
        value0 = self.v0.forward_with_weight(fields.scalar, v0_weight)
        q1 = self.q1.forward_with_weight(fields.vector, q1_weight)
        k1 = self.k1.forward_with_weight(fields.vector, k1_weight)
        value1 = self.v1.forward_with_weight(fields.vector, v1_weight)
        q2 = l2_to_matrix(
            self.q2.forward_with_weight(fields.tensor, q2_weight)
        )
        k2 = l2_to_matrix(
            self.k2.forward_with_weight(fields.tensor, k2_weight)
        )
        value2 = l2_to_matrix(
            self.v2.forward_with_weight(fields.tensor, v2_weight)
        )
        # One indexed scalar value tensor is shared by every tensor-product
        # path.  Repeating this gather kept multiple edge-by-channel
        # activations alive for backward; on the logged batch-8 failure each
        # duplicate was roughly the requested 68 MiB allocation.
        target_value0 = value0[target]
        transported_k1 = torch.einsum("eij,ecj->eci", connection, k1[target])
        transported_v1 = torch.einsum("eij,ecj->eci", connection, value1[target])
        transported_k2 = connection[:, None] @ k2[target] @ connection[:, None].transpose(-1, -2)
        transported_v2 = connection[:, None] @ value2[target] @ connection[:, None].transpose(-1, -2)

        h = cfg.heads
        c0, c1, c2 = cfg.scalar_channels // h, cfg.vector_channels // h, cfg.tensor_channels // h
        score0 = normalized_inner_product(
            q0.reshape(v, h, c0)[source],
            k0.reshape(v, h, c0)[target],
            cfg.epsilon,
        ) / cfg.attention_temperature_scalar
        q1_head = q1.reshape(v, h, c1, 3)
        k1_head = transported_k1.reshape(-1, h, c1, 3)
        score1 = normalized_inner_product(
            q1_head[source].flatten(-2),
            k1_head.flatten(-2),
            cfg.epsilon,
        ) / cfg.attention_temperature_vector
        q2_head = q2.reshape(v, h, c2, 3, 3)
        k2_head = transported_k2.reshape(-1, h, c2, 3, 3)
        score2 = normalized_inner_product(
            q2_head[source].flatten(-3),
            k2_head.flatten(-3),
            cfg.epsilon,
        ) / cfg.attention_temperature_tensor
        logits = score0 + score1 + score2 + radial @ self.score_radial.transpose(0, 1)
        if edge_ot_cost is not None:
            if edge_ot_cost.shape != (edge.shape[1],):
                raise ValueError("edge_ot_cost must match active adjacency including self edges")
            logits = logits - cfg.ot_bias_weight * edge_ot_cost[:, None]
        if edge_uncertainty is not None:
            if edge_uncertainty.shape != (edge.shape[1],):
                raise ValueError(
                    "edge_uncertainty must match active adjacency including self edges"
                )
            logits = logits - cfg.uncertainty_bias_weight * edge_uncertainty[:, None]
        attention = segment_softmax(logits, source, v)

        y2 = l2_to_matrix(direction_l2(direction))
        coeff00 = self._radial_coeff(self.path_00, radial)
        scalar_edge = coeff00 * target_value0
        vector_dot = torch.sum(transported_v1 * direction[:, None], dim=-1)
        tensor_contract = torch.einsum("ecij,eij->ec", transported_v2, y2)
        scalar_edge = scalar_edge + self._radial_coeff(
            self.path_11_to_0, radial
        ) * vector_dot.mean(1, keepdim=True)
        scalar_edge = scalar_edge + self._radial_coeff(
            self.path_22_to_0, radial
        ) * tensor_contract.mean(1, keepdim=True)

        vector_edge = (
            self._radial_coeff(self.path_11, radial)[:, :, None]
            * transported_v1
        )
        scalar_target_mean = target_value0.mean(-1, keepdim=True)
        vector_edge = (
            vector_edge
            + self._radial_coeff(self.path_01_to_1, radial)[:, :, None]
            * scalar_target_mean[:, :, None]
            * direction[:, None]
        )
        tensor_direction = torch.einsum("ecij,ej->eci", transported_v2, direction)
        vector_edge = vector_edge + self._radial_coeff(
            self.path_21_to_1, radial
        )[:, :, None] * tensor_direction.mean(1, keepdim=True)

        tensor_edge = (
            self._radial_coeff(self.path_22, radial)[:, :, None, None]
            * transported_v2
        )
        tensor_edge = (
            tensor_edge
            + self._radial_coeff(self.path_02_to_2, radial)[:, :, None, None]
            * scalar_target_mean[:, :, None, None]
            * y2[:, None]
        )
        vector_mean = transported_v1.mean(1)
        mixed_tensor = symmetric_traceless_outer(direction, vector_mean)
        tensor_edge = tensor_edge + self._radial_coeff(
            self.path_11_to_2, radial
        )[:, :, None, None] * mixed_tensor[:, None]

        # Broadcast heads over their multiplicities during multiplication.
        # ``repeat_interleave`` materialized three additional dense edge
        # tensors that carried no information and inflated both forward memory
        # traffic and the retained autograd graph.
        edge_count = attention.shape[0]
        message0 = segment_sum(
            (attention[:, :, None] * scalar_edge.reshape(edge_count, h, c0)).reshape(
                edge_count, h * c0
            ),
            source,
            v,
        )
        message1 = segment_sum(
            (
                attention[:, :, None, None]
                * vector_edge.reshape(edge_count, h, c1, 3)
            ).reshape(edge_count, h * c1, 3),
            source,
            v,
        )
        message2_matrix = segment_sum(
            (
                attention[:, :, None, None, None]
                * tensor_edge.reshape(edge_count, h, c2, 3, 3)
            ).reshape(edge_count, h * c2, 3, 3),
            source,
            v,
        )
        message2 = matrix_to_l2(message2_matrix)
        invariant = fields.scalar
        gate0 = torch.sigmoid(self.gate0(invariant))
        gate1 = torch.sigmoid(self.gate1(invariant))
        gate2 = torch.sigmoid(self.gate2(invariant))
        # One common smooth RMS floor keeps all three irreps finite at an
        # exactly cancelled message.  The previous scalar vector_norm/clamp
        # retained the undefined derivative of ||0||.
        norm0 = torch.sqrt(
            torch.sum(message0.square(), dim=-1, keepdim=True) + cfg.epsilon
        )
        norm1 = torch.sqrt(torch.sum(message1.square(), dim=(-1, -2), keepdim=True) + cfg.epsilon)
        norm2 = torch.sqrt(torch.sum(message2.square(), dim=(-1, -2), keepdim=True) + cfg.epsilon)
        output0 = fields.scalar + cfg.residual_step * gate0 * message0 / norm0
        output1 = fields.vector + cfg.residual_step * gate1[:, :, None] * message1 / norm1
        output2 = fields.tensor + cfg.residual_step * gate2[:, :, None] * message2 / norm2
        return output0, output1, output2


__all__ = [
    "GSTAConfig",
    "GaugeCovariantSparseTransportAttention",
    "IrrepTensor",
    "active_adjacency",
    "direction_l2",
    "l2_to_matrix",
    "matrix_to_l2",
    "normalized_inner_product",
]

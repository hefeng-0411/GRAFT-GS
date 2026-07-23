"""Manifold Mapping Operator with sparse implicit unbalanced Sinkhorn.

The operator implements

    M_OT = W_O o T_epsilon o U_VGGT

as three explicit, inspectable stages:

1. camera-convention-safe unprojection into geometric evidence particles;
2. sparse unbalanced entropic optimal transport on a radius graph;
3. gauge-covariant chart writing into 0e/1o/2e irreducible moments.

The Sinkhorn fixed point is differentiated with the implicit function theorem.
Backward therefore solves the transposed fixed-point system and does not retain
the forward iteration tape.  Geometry, covariance inverses, potentials, and
the implicit solve are expected to run in FP32 or FP64 on the target server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log, sqrt
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..geometry.atlas import PersistentOctreeAtlas


@dataclass
class EvidenceParticles:
    """A geometric measure sampled from one multiview scene.

    Camera tables are retained once per scene rather than repeated per
    particle.  ``view_index`` indexes these tables and all cameras use the
    OpenCV camera-from-world convention.  Keeping this provenance is required
    for exact image-plane disagreement tests during adaptive atlas refinement.
    """

    positions: Tensor
    rays: Tensor
    features: Tensor
    covariance: Tensor
    confidence: Tensor
    mass: Tensor
    view_index: Tensor
    pixel_uv: Tensor
    extrinsics_world_to_camera: Tensor
    intrinsics: Tensor
    depth_variance: Tensor
    colors: Optional[Tensor] = None

    def validate(self, covariance_floor: float = 0.0) -> None:
        m = self.positions.shape[0]
        expected = {
            "positions": (m, 3),
            "rays": (m, 3),
            "covariance": (m, 3, 3),
            "confidence": (m,),
            "mass": (m,),
            "view_index": (m,),
            "pixel_uv": (m, 2),
            "depth_variance": (m,),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(getattr(self, name).shape)}")
        if self.features.ndim != 2 or self.features.shape[0] != m:
            raise ValueError("features must have shape [M, D]")
        view_count = self.extrinsics_world_to_camera.shape[0]
        if tuple(self.extrinsics_world_to_camera.shape) != (view_count, 3, 4):
            raise ValueError("extrinsics_world_to_camera must have shape [K,3,4]")
        if tuple(self.intrinsics.shape) != (view_count, 3, 3):
            raise ValueError("intrinsics must have shape [K,3,3]")
        if view_count == 0:
            raise ValueError("evidence must retain at least one camera")
        if self.view_index.dtype != torch.int64:
            raise ValueError("view_index must use int64 indices")
        if self.view_index.numel() and (
            int(self.view_index.amin()) < 0 or int(self.view_index.amax()) >= view_count
        ):
            raise ValueError("view_index contains an index outside the retained camera table")
        if not bool(torch.all(torch.isfinite(self.extrinsics_world_to_camera))):
            raise ValueError("camera extrinsics must be finite")
        if not bool(torch.all(torch.isfinite(self.intrinsics))):
            raise ValueError("camera intrinsics must be finite")
        if bool(torch.any(self.intrinsics[:, 0, 0] <= 0)) or bool(
            torch.any(self.intrinsics[:, 1, 1] <= 0)
        ):
            raise ValueError("camera focal lengths must be positive")
        if self.colors is not None and tuple(self.colors.shape) != (m, 3):
            raise ValueError("colors must have shape [M, 3]")
        finite_fields = {
            "positions": self.positions,
            "rays": self.rays,
            "features": self.features,
            "covariance": self.covariance,
            "confidence": self.confidence,
            "mass": self.mass,
            "pixel_uv": self.pixel_uv,
            "extrinsics_world_to_camera": self.extrinsics_world_to_camera,
            "intrinsics": self.intrinsics,
            "depth_variance": self.depth_variance,
        }
        if self.colors is not None:
            finite_fields["colors"] = self.colors
        nonfinite = [
            name
            for name, value in finite_fields.items()
            if not bool(torch.all(torch.isfinite(value)))
        ]
        if nonfinite:
            raise ValueError(
                f"evidence particle fields contain non-finite values: {nonfinite}"
            )
        if torch.any(self.confidence < 0) or torch.any(self.confidence > 1):
            raise ValueError("confidence must lie in [0, 1]")
        if torch.any(self.mass < 0):
            raise ValueError("particle mass must be non-negative")
        if not bool(torch.any(self.mass > 0)):
            raise ValueError("evidence measure must contain positive particle mass")
        ray_error = (torch.linalg.vector_norm(self.rays, dim=-1) - 1.0).abs().max()
        if float(ray_error) > 1.0e-4:
            raise ValueError("rays must be unit vectors")
        covariance = 0.5 * (self.covariance + self.covariance.transpose(-1, -2))
        if float(torch.linalg.eigvalsh(covariance).amin()) <= covariance_floor:
            raise ValueError("all evidence covariance matrices must be SPD")

    def to(self, *args: object, **kwargs: object) -> "EvidenceParticles":
        values: Dict[str, object] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, Tensor):
                converted = value.to(*args, **kwargs)
                # A scene-wide precision conversion must never turn sparse
                # provenance indices into floating-point values.
                if not value.dtype.is_floating_point and converted.dtype != value.dtype:
                    converted = converted.to(dtype=value.dtype)
                values[name] = converted
            else:
                values[name] = value
        return EvidenceParticles(**values)


@dataclass(frozen=True)
class ImplicitSinkhornConfig:
    epsilon: float = 0.03
    tau_source: float = 0.5
    tau_target: float = 0.5
    max_iterations: int = 300
    tolerance: float = 1.0e-7
    backward_max_iterations: int = 300
    backward_tolerance: float = 1.0e-8
    backward_damping: float = 0.8
    mass_floor: float = 1.0e-12
    convergence_check_interval: int = 8
    solve_in_float64: bool = True

    def __post_init__(self) -> None:
        if self.epsilon <= 0 or self.tau_source <= 0 or self.tau_target <= 0:
            raise ValueError("epsilon and both marginal relaxation parameters must be positive")
        if not 0 < self.backward_damping <= 1:
            raise ValueError("backward_damping must lie in (0, 1]")
        if self.max_iterations < 1 or self.backward_max_iterations < 1:
            raise ValueError("forward and backward Sinkhorn iterations must be positive")
        if self.tolerance <= 0 or self.backward_tolerance <= 0 or self.mass_floor <= 0:
            raise ValueError("Sinkhorn tolerances and mass floor must be positive")
        if self.convergence_check_interval < 1:
            raise ValueError("convergence_check_interval must be positive")


@dataclass(frozen=True)
class ManifoldMappingConfig:
    sinkhorn: ImplicitSinkhornConfig = field(default_factory=ImplicitSinkhornConfig)
    support_radius_factor: float = 2.25
    atlas_chunk_size: int = 2048
    evidence_chunk_size: int = 8192
    ensure_source_support: bool = True
    ensure_target_support: bool = True
    metric_epsilon: float = 1.0e-5
    metric_normal_weight: float = 1.0
    feature_cost_dim: int = 64
    radial_basis_count: int = 4
    radial_support_factor: float = 2.0
    retention_shrinkage: float = 0.05

    def __post_init__(self) -> None:
        if self.support_radius_factor <= 0 or self.radial_support_factor <= 0:
            raise ValueError("transport support factors must be positive")
        if self.atlas_chunk_size < 1 or self.evidence_chunk_size < 1:
            raise ValueError("transport chunk sizes must be positive")
        if self.metric_epsilon <= 0 or self.metric_normal_weight < 0:
            raise ValueError("metric regularization must be positive/non-negative")
        if self.retention_shrinkage <= 0:
            raise ValueError("retention_shrinkage must be positive")


@dataclass
class SparseTransportGraph:
    """COO bipartite support; source is active-chart local index."""

    edge_index: Tensor
    atlas_node_index: Tensor
    source_count: int
    target_count: int
    support_radius: Tensor

    @property
    def source(self) -> Tensor:
        return self.edge_index[0]

    @property
    def target(self) -> Tensor:
        return self.edge_index[1]

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass
class SinkhornDiagnostics:
    iterations: int
    fixed_point_residual: float
    source_transport_mass: Tensor
    target_transport_mass: Tensor
    objective: Tensor
    converged: bool = True
    effective_tolerance: float = 0.0
    internal_minimum_log_plan: float = 0.0
    storage_underflow_edges: int = 0
    storage_zero_source_rows: int = 0
    storage_zero_target_columns: int = 0
    internal_solve_dtype: str = ""


@dataclass
class IrrepMoments:
    """Chart-local ``48(0e)+16(1o)+4(2e)+12(0e)`` representation."""

    scalar_0e: Tensor  # [V, 48]
    vector_1o: Tensor  # [V, 16, 3]
    tensor_2e: Tensor  # [V, 4, 5]
    auxiliary_0e: Tensor  # [V, 12]

    def pack(self) -> Tensor:
        return torch.cat(
            (
                self.scalar_0e,
                self.auxiliary_0e,
                self.vector_1o.flatten(1),
                self.tensor_2e.flatten(1),
            ),
            dim=-1,
        )


@dataclass
class MappingResult:
    evidence: EvidenceParticles
    graph: SparseTransportGraph
    cost: Tensor
    plan: Tensor
    diagnostics: SinkhornDiagnostics
    transported_centers: Tensor
    transported_mass: Tensor
    observation_reliability: Tensor
    transported_color: Optional[Tensor]
    riemannian_metric: Tensor
    irreps: IrrepMoments
    latent: Tensor


def sparse_view_reprojection_variance(
    atlas: PersistentOctreeAtlas,
    mapping: MappingResult,
    minimum_projected_radius: float = 1.0,
) -> Tensor:
    r"""Compute plan-conditional image disagreement for every active chart.

    For an atlas row ``i`` and retained camera ``k``, the sparse UOT plan
    defines the view-conditional observed pixel barycenter

    ``u_ik = sum_{j in k} pi_ij u_j / sum_{j in k} pi_ij``.

    The transported chart center is projected through the *same calibrated
    camera*, and its residual is divided by the projected half-diagonal of the
    octree cell (with a one-pixel numerical floor).  The returned statistic is
    the conditional population variance of these 2-vectors across supported
    views.  It is therefore dimensionless, resolution-aware, sparse in the OT
    support, and differentiable with respect to the plan, center, and cameras.
    A chart supported by fewer than two views has zero cross-view variance;
    its geometric and uncertainty split criteria remain active separately.
    """

    if minimum_projected_radius <= 0:
        raise ValueError("minimum_projected_radius must be positive")
    active = mapping.graph.atlas_node_index
    if not torch.equal(active, atlas.active_indices):
        raise ValueError("mapping rows must follow active atlas order")
    source, target = mapping.graph.source, mapping.graph.target
    if source.numel() == 0:
        return mapping.plan.new_zeros(mapping.graph.source_count)
    evidence = mapping.evidence
    view = evidence.view_index[target]
    view_count = evidence.extrinsics_world_to_camera.shape[0]
    pair_id = source * view_count + view
    unique_pair, inverse = torch.unique(pair_id, sorted=True, return_inverse=True)
    group_count = unique_pair.shape[0]
    group_mass = mapping.plan.new_zeros(group_count)
    group_mass.index_add_(0, inverse, mapping.plan)
    pixel_numerator = mapping.plan.new_zeros((group_count, 2))
    pixel_numerator.index_add_(
        0,
        inverse,
        mapping.plan[:, None] * evidence.pixel_uv[target],
    )
    epsilon = torch.finfo(mapping.plan.dtype).eps
    observed_pixel = pixel_numerator / group_mass.clamp_min(epsilon)[:, None]

    group_source = torch.div(unique_pair, view_count, rounding_mode="floor")
    group_view = torch.remainder(unique_pair, view_count)
    world_point = mapping.transported_centers[group_source]
    extrinsic = evidence.extrinsics_world_to_camera[group_view]
    camera_point = (
        torch.einsum("gij,gj->gi", extrinsic[:, :3, :3], world_point)
        + extrinsic[:, :3, 3]
    )
    depth = camera_point[:, 2]
    safe_depth = depth.clamp_min(max(float(epsilon), 1.0e-8))
    homogeneous_pixel = torch.einsum(
        "gij,gj->gi", evidence.intrinsics[group_view], camera_point
    )
    projected_pixel = homogeneous_pixel[:, :2] / safe_depth[:, None]

    focal = torch.sqrt(
        (
            evidence.intrinsics[group_view, 0, 0]
            * evidence.intrinsics[group_view, 1, 1]
        ).clamp_min(epsilon)
    )
    # A cube side projects to a conservative surface-chart half diagonal.
    projected_radius = (
        (2.0**-0.5)
        * atlas.cell_sides[active][group_source]
        * focal
        / safe_depth.abs().clamp_min(max(float(epsilon), 1.0e-8))
    ).clamp_min(minimum_projected_radius)
    normalized_residual = (
        projected_pixel - observed_pixel
    ) / projected_radius[:, None]

    mean_numerator = mapping.plan.new_zeros((mapping.graph.source_count, 2))
    mean_numerator.index_add_(
        0, group_source, group_mass[:, None] * normalized_residual
    )
    mean = mean_numerator / mapping.transported_mass.clamp_min(epsilon)[:, None]
    second_numerator = mapping.plan.new_zeros(mapping.graph.source_count)
    second_numerator.index_add_(
        0,
        group_source,
        group_mass * normalized_residual.square().sum(-1),
    )
    second = second_numerator / mapping.transported_mass.clamp_min(epsilon)
    return (second - mean.square().sum(-1)).clamp_min(0.0)


def _inverse_softplus(value: float) -> float:
    return log(torch.expm1(torch.tensor(value, dtype=torch.float64)).item())


def _segment_sum(values: Tensor, index: Tensor, size: int) -> Tensor:
    output = values.new_zeros((size, *values.shape[1:]))
    if values.numel() > 0:
        output.index_add_(0, index, values)
    return output


def _segment_logsumexp(values: Tensor, index: Tensor, size: int) -> Tensor:
    """Differentiable COO segment logsumexp with explicit empty-row semantics."""

    maximum = values.new_full((size,), -torch.inf)
    maximum.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
    safe_maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    exponential = torch.exp(values - safe_maximum[index])
    total = _segment_sum(exponential, index, size)
    result = safe_maximum + torch.log(total.clamp_min(torch.finfo(values.dtype).tiny))
    return torch.where(total > 0, result, torch.full_like(result, -torch.inf))


def _generalized_kl(p: Tensor, q: Tensor, floor: float) -> Tensor:
    p_safe = p.clamp_min(floor)
    q_safe = q.clamp_min(floor)
    return torch.sum(p_safe * (torch.log(p_safe) - torch.log(q_safe)) - p_safe + q_safe)


def _sinkhorn_fixed_point(
    cost: Tensor,
    log_source_mass: Tensor,
    log_target_mass: Tensor,
    source: Tensor,
    target: Tensor,
    epsilon: float,
    rho_source: float,
    rho_target: float,
    max_iterations: int,
    tolerance: float,
    convergence_check_interval: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, int, float, float]:
    """Solve generalized Sinkhorn scaling on an arbitrary sparse support."""

    n, m = log_source_mass.numel(), log_target_mass.numel()
    log_kernel = log_source_mass[source] + log_target_mass[target] - cost / epsilon
    log_u = torch.zeros_like(log_source_mass)
    log_v = torch.zeros_like(log_target_mass)
    residual = float("inf")
    effective_tolerance = float("inf")
    iterations = max_iterations
    for iteration in range(max_iterations):
        row_lse = _segment_logsumexp(log_kernel + log_v[target], source, n)
        new_log_u = rho_source * (log_source_mass - row_lse)
        col_lse = _segment_logsumexp(log_kernel + new_log_u[source], target, m)
        new_log_v = rho_target * (log_target_mass - col_lse)
        should_check = (
            (iteration + 1) % convergence_check_interval == 0
            or iteration + 1 == max_iterations
        )
        if should_check:
            # Certify the actual coupled fixed-point equations, not merely a
            # small change between iterates (which damping could manufacture).
            check_row_lse = _segment_logsumexp(
                log_kernel + new_log_v[target], source, n
            )
            fixed_log_u = rho_source * (log_source_mass - check_row_lse)
            check_col_lse = _segment_logsumexp(
                log_kernel + new_log_u[source], target, m
            )
            fixed_log_v = rho_target * (log_target_mass - check_col_lse)
            residual_tensor = torch.maximum(
                (new_log_u - fixed_log_u).abs().amax(),
                (new_log_v - fixed_log_v).abs().amax(),
            )
            potential_scale = torch.maximum(
                new_log_u.abs().amax(), new_log_v.abs().amax()
            )
            threshold_tensor = tolerance * (1.0 + potential_scale)
        log_u, log_v = new_log_u, new_log_v
        if should_check:
            residual = float(residual_tensor.detach().cpu())
            effective_tolerance = float(threshold_tensor.detach().cpu())
            if not isfinite(residual):
                break
            if residual <= effective_tolerance:
                iterations = iteration + 1
                break
    log_plan = log_kernel + log_u[source] + log_v[target]
    plan = torch.exp(log_plan)
    return plan, log_plan, log_u, log_v, iterations, residual, effective_tolerance


class _ImplicitUnbalancedSinkhorn(torch.autograd.Function):
    """Custom autograd for the sparse unbalanced Sinkhorn fixed point."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        cost: Tensor,
        log_source_mass: Tensor,
        log_target_mass: Tensor,
        source: Tensor,
        target: Tensor,
        epsilon: float,
        rho_source: float,
        rho_target: float,
        max_iterations: int,
        tolerance: float,
        backward_max_iterations: int,
        backward_tolerance: float,
        backward_damping: float,
        convergence_check_interval: int,
        solve_in_float64: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        input_dtype = cost.dtype
        solve_dtype = (
            torch.float64
            if solve_in_float64 and input_dtype == torch.float32
            else input_dtype
        )
        (
            plan,
            log_plan,
            log_u,
            log_v,
            iterations,
            residual,
            effective_tolerance,
        ) = _sinkhorn_fixed_point(
            cost.to(dtype=solve_dtype),
            log_source_mass.to(dtype=solve_dtype),
            log_target_mass.to(dtype=solve_dtype),
            source,
            target,
            epsilon,
            rho_source,
            rho_target,
            max_iterations,
            tolerance,
            convergence_check_interval,
        )
        if not bool(torch.all(torch.isfinite(log_plan))):
            raise FloatingPointError(
                "sparse unbalanced Sinkhorn produced non-finite log transport mass"
            )
        if residual > effective_tolerance:
            raise RuntimeError(
                "sparse unbalanced Sinkhorn did not converge: "
                f"iterations={iterations}, residual={residual:.6e}, "
                f"effective_tolerance={effective_tolerance:.6e}"
            )
        # The UOT marginal penalties are soft: a geometrically incompatible
        # row/column may legitimately retain exponentially tiny mass.  Form the
        # conditional probabilities directly in log space so the implicit
        # Jacobian remains valid even when the absolute mass is below FP64.
        row_log_mass = _segment_logsumexp(
            log_plan, source, log_source_mass.numel()
        )
        col_log_mass = _segment_logsumexp(
            log_plan, target, log_target_mass.numel()
        )
        row_probability = torch.exp(log_plan - row_log_mass[source])
        col_probability = torch.exp(log_plan - col_log_mass[target])
        if (
            not bool(torch.all(torch.isfinite(row_probability)))
            or not bool(torch.all(torch.isfinite(col_probability)))
        ):
            raise FloatingPointError(
                "log-domain Sinkhorn conditional probabilities are non-finite"
            )
        storage_plan = plan.to(dtype=input_dtype)
        storage_row = _segment_sum(
            storage_plan, source, log_source_mass.numel()
        )
        storage_col = _segment_sum(
            storage_plan, target, log_target_mass.numel()
        )
        storage_underflow_edges = int(torch.count_nonzero(storage_plan == 0).item())
        storage_zero_rows = int(torch.count_nonzero(storage_row == 0).item())
        storage_zero_columns = int(torch.count_nonzero(storage_col == 0).item())
        minimum_log_plan = float(log_plan.amin().detach().cpu())
        if not bool(torch.any(storage_plan > 0)):
            raise FloatingPointError(
                "all sparse UOT mass lies below the geometric storage dtype; "
                f"minimum_log_plan={minimum_log_plan:.6e}, dtype={input_dtype}"
            )
        ctx.save_for_backward(
            plan, row_probability, col_probability, source, target
        )
        ctx.input_dtype = input_dtype
        ctx.source_count = log_source_mass.numel()
        ctx.target_count = log_target_mass.numel()
        ctx.epsilon = epsilon
        ctx.rho_source = rho_source
        ctx.rho_target = rho_target
        ctx.backward_max_iterations = backward_max_iterations
        ctx.backward_tolerance = backward_tolerance
        ctx.backward_damping = backward_damping
        ctx.convergence_check_interval = convergence_check_interval
        # FP64 preserves large COO edge counts exactly beyond FP32's 24-bit
        # integer mantissa and retains the log-domain dynamic-range diagnostic.
        status = torch.tensor(
            (
                float(iterations),
                residual,
                effective_tolerance,
                minimum_log_plan,
                float(storage_underflow_edges),
                float(storage_zero_rows),
                float(storage_zero_columns),
                64.0 if solve_dtype == torch.float64 else 32.0,
            ),
            dtype=torch.float64,
            device=cost.device,
        )
        storage_log_u = log_u.to(dtype=input_dtype)
        storage_log_v = log_v.to(dtype=input_dtype)
        ctx.mark_non_differentiable(storage_log_u, storage_log_v, status)
        return (
            storage_plan,
            storage_log_u,
            storage_log_v,
            status,
        )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_plan: Optional[Tensor],
        grad_log_u: Optional[Tensor],
        grad_log_v: Optional[Tensor],
        grad_status: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], ...]:
        del grad_log_u, grad_log_v, grad_status
        if grad_plan is None:
            return (None,) * 15
        plan, row_probability, col_probability, source, target = ctx.saved_tensors
        grad_plan = grad_plan.to(dtype=plan.dtype)
        weighted_gradient = grad_plan * plan
        source_count = ctx.source_count
        target_count = ctx.target_count
        q_source = _segment_sum(weighted_gradient, source, source_count)
        q_target = _segment_sum(weighted_gradient, target, target_count)

        # Solve J^T lambda = dL/d(log_u,log_v).  Since both relaxation
        # exponents are strictly below one, this block Gauss-Seidel map is a
        # contraction.  Damping remains explicit for near-balanced regimes.
        lambda_source = torch.zeros_like(q_source)
        lambda_target = torch.zeros_like(q_target)
        damping = ctx.backward_damping
        backward_iterations = ctx.backward_max_iterations
        backward_residual = float("inf")
        backward_threshold = float("inf")
        for iteration in range(ctx.backward_max_iterations):
            candidate_source = q_source - ctx.rho_target * _segment_sum(
                col_probability * lambda_target[target], source, source_count
            )
            candidate_source = (1.0 - damping) * lambda_source + damping * candidate_source
            candidate_target = q_target - ctx.rho_source * _segment_sum(
                row_probability * candidate_source[source], target, target_count
            )
            candidate_target = (1.0 - damping) * lambda_target + damping * candidate_target
            lambda_source, lambda_target = candidate_source, candidate_target
            should_check = (
                (iteration + 1) % ctx.convergence_check_interval == 0
                or iteration + 1 == ctx.backward_max_iterations
            )
            if should_check:
                equation_source = lambda_source + ctx.rho_target * _segment_sum(
                    col_probability * lambda_target[target], source, source_count
                ) - q_source
                equation_target = lambda_target + ctx.rho_source * _segment_sum(
                    row_probability * lambda_source[source], target, target_count
                ) - q_target
                residual = torch.maximum(
                    equation_source.abs().amax(), equation_target.abs().amax()
                )
                rhs_scale = torch.maximum(q_source.abs().amax(), q_target.abs().amax())
                threshold = ctx.backward_tolerance * (1.0 + rhs_scale)
                backward_residual = float(residual.detach().cpu())
                backward_threshold = float(threshold.detach().cpu())
                if not isfinite(backward_residual):
                    break
                if backward_residual <= backward_threshold:
                    backward_iterations = iteration + 1
                    break
        if backward_residual > backward_threshold:
            raise RuntimeError(
                "implicit Sinkhorn adjoint did not converge: "
                f"iterations={backward_iterations}, residual={backward_residual:.6e}, "
                f"effective_tolerance={backward_threshold:.6e}"
            )

        epsilon = ctx.epsilon
        grad_cost = -weighted_gradient / epsilon
        grad_cost = grad_cost + (
            ctx.rho_source * lambda_source[source] * row_probability
            + ctx.rho_target * lambda_target[target] * col_probability
        ) / epsilon
        direct_source = _segment_sum(weighted_gradient, source, source_count)
        direct_target = _segment_sum(weighted_gradient, target, target_count)
        grad_log_source = direct_source - ctx.rho_target * _segment_sum(
            col_probability * lambda_target[target], source, source_count
        )
        grad_log_target = direct_target - ctx.rho_source * _segment_sum(
            row_probability * lambda_source[source], target, target_count
        )
        gradients_finite = (
            torch.all(torch.isfinite(grad_cost))
            & torch.all(torch.isfinite(grad_log_source))
            & torch.all(torch.isfinite(grad_log_target))
        )
        if not bool(gradients_finite):
            raise FloatingPointError("implicit Sinkhorn adjoint produced non-finite gradients")
        return (
            grad_cost.to(dtype=ctx.input_dtype),
            grad_log_source.to(dtype=ctx.input_dtype),
            grad_log_target.to(dtype=ctx.input_dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class ImplicitUnbalancedSinkhorn(nn.Module):
    """Sparse UOT solver for ``<Gamma,C> + eps KL + tau_a KL + tau_b KL``."""

    def __init__(self, config: ImplicitSinkhornConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        cost: Tensor,
        source_mass: Tensor,
        target_mass: Tensor,
        edge_index: Tensor,
    ) -> Tuple[Tensor, SinkhornDiagnostics]:
        if edge_index.ndim != 2 or tuple(edge_index.shape[:1]) != (2,):
            raise ValueError("edge_index must have shape [2,E]")
        source, target = edge_index
        if cost.ndim != 1 or cost.shape[0] != source.shape[0]:
            raise ValueError("cost must have one scalar per sparse edge")
        if source.numel() == 0:
            raise ValueError("Sinkhorn support cannot be empty")
        if edge_index.dtype != torch.int64:
            raise ValueError("Sinkhorn sparse indices must use int64")
        if cost.dtype not in {torch.float32, torch.float64}:
            raise ValueError("Sinkhorn reference solve requires float32 or float64 cost")
        if source_mass.dtype != cost.dtype or target_mass.dtype != cost.dtype:
            raise ValueError("Sinkhorn cost and marginal masses must share a floating dtype")
        if (
            source.device != cost.device
            or target.device != cost.device
            or source_mass.device != cost.device
            or target_mass.device != cost.device
        ):
            raise ValueError("Sinkhorn costs, masses, and sparse support must share a device")
        if source_mass.ndim != 1 or target_mass.ndim != 1:
            raise ValueError("Sinkhorn marginal masses must be one-dimensional")
        inputs_finite = (
            torch.all(torch.isfinite(cost))
            & torch.all(torch.isfinite(source_mass))
            & torch.all(torch.isfinite(target_mass))
        )
        if not bool(inputs_finite):
            raise ValueError("Sinkhorn cost and marginal masses must be finite")
        if bool(torch.any(cost < 0)):
            raise ValueError("GRAFT-GS transport costs must be non-negative")
        if bool(torch.any(source_mass < 0)) or bool(torch.any(target_mass < 0)):
            raise ValueError("Sinkhorn marginal masses must be non-negative")
        if not bool(torch.any(source_mass > 0)) or not bool(torch.any(target_mass > 0)):
            raise ValueError("each Sinkhorn marginal measure must contain positive mass")
        if int(source.amin()) != 0 or int(source.amax()) != source_mass.numel() - 1:
            raise ValueError("source support indices must cover exactly [0,N)")
        if int(target.amin()) != 0 or int(target.amax()) != target_mass.numel() - 1:
            raise ValueError("target support indices must cover exactly [0,M)")
        if torch.unique(source).numel() != source_mass.numel():
            raise ValueError("every source node must have at least one transport edge")
        if torch.unique(target).numel() != target_mass.numel():
            raise ValueError("every target particle must have at least one transport edge")
        cfg = self.config
        log_a = torch.log(source_mass.clamp_min(cfg.mass_floor))
        log_b = torch.log(target_mass.clamp_min(cfg.mass_floor))
        rho_a = cfg.tau_source / (cfg.tau_source + cfg.epsilon)
        rho_b = cfg.tau_target / (cfg.tau_target + cfg.epsilon)
        plan, _, _, status = _ImplicitUnbalancedSinkhorn.apply(
            cost,
            log_a,
            log_b,
            source,
            target,
            cfg.epsilon,
            rho_a,
            rho_b,
            cfg.max_iterations,
            cfg.tolerance,
            cfg.backward_max_iterations,
            cfg.backward_tolerance,
            cfg.backward_damping,
            cfg.convergence_check_interval,
            cfg.solve_in_float64,
        )
        row = _segment_sum(plan, source, source_mass.numel())
        col = _segment_sum(plan, target, target_mass.numel())
        reference = source_mass[source] * target_mass[target]
        objective = torch.sum(plan * cost)
        objective = objective + cfg.epsilon * _generalized_kl(plan, reference, cfg.mass_floor)
        objective = objective + cfg.tau_source * _generalized_kl(row, source_mass, cfg.mass_floor)
        objective = objective + cfg.tau_target * _generalized_kl(col, target_mass, cfg.mass_floor)
        diagnostics = SinkhornDiagnostics(
            iterations=int(status[0].item()),
            fixed_point_residual=float(status[1].item()),
            source_transport_mass=row,
            target_transport_mass=col,
            objective=objective,
            converged=True,
            effective_tolerance=float(status[2].item()),
            internal_minimum_log_plan=float(status[3].item()),
            storage_underflow_edges=int(status[4].item()),
            storage_zero_source_rows=int(status[5].item()),
            storage_zero_target_columns=int(status[6].item()),
            internal_solve_dtype=f"float{int(status[7].item())}",
        )
        return plan, diagnostics


class ConfidenceCovarianceCalibrator(nn.Module):
    """Calibrate VGGT confidence and construct anisotropic ray covariance."""

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self.confidence_bias = nn.Parameter(torch.tensor(0.0))
        self.log_tangent_scale = nn.Parameter(torch.tensor(log(0.75)))
        self.log_normal_scale = nn.Parameter(torch.tensor(log(0.02)))
        self.log_sigma_floor = nn.Parameter(torch.tensor(log(1.0e-5)))

    def forward(
        self,
        raw_confidence: Tensor,
        depth: Tensor,
        focal_mean: Tensor,
        pixel_footprint: Tensor,
        world_rays: Tensor,
        reprojection_residual: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        parameters = {
            "log_temperature": self.log_temperature,
            "confidence_bias": self.confidence_bias,
            "log_tangent_scale": self.log_tangent_scale,
            "log_normal_scale": self.log_normal_scale,
            "log_sigma_floor": self.log_sigma_floor,
        }
        nonfinite = [
            name
            for name, value in parameters.items()
            if not bool(torch.all(torch.isfinite(value)))
        ]
        if nonfinite:
            raise FloatingPointError(
                f"confidence calibrator parameters are non-finite: {nonfinite}"
            )
        eps = torch.finfo(depth.dtype).eps
        temperature = torch.exp(
            self.log_temperature.clamp(log(0.05), log(20.0))
        )
        tangent_scale = torch.exp(
            self.log_tangent_scale.clamp(log(1.0e-4), log(10.0))
        ).to(depth)
        normal_scale = torch.exp(
            self.log_normal_scale.clamp(log(1.0e-5), log(2.0))
        ).to(depth)
        sigma_floor = torch.exp(
            self.log_sigma_floor.clamp(log(1.0e-8), log(1.0e-1))
        ).to(depth)
        # VGGT heads use expp1 confidence, so log(confidence) is a better
        # unconstrained calibration coordinate than treating it as a probability.
        logits = temperature * torch.log(raw_confidence.clamp_min(eps)) + self.confidence_bias
        if reprojection_residual is not None:
            logits = logits - reprojection_residual.square()
        confidence = torch.sigmoid(logits)
        tangent_sigma = tangent_scale * depth * pixel_footprint / focal_mean
        tangent_sigma = tangent_sigma.clamp_min(sigma_floor)
        normal_sigma = normal_scale * depth / torch.sqrt(confidence.clamp_min(eps))
        normal_sigma = normal_sigma.clamp_min(sigma_floor)

        helper = torch.zeros_like(world_rays)
        helper[..., 2] = 1
        parallel = world_rays[..., 2].abs() > 0.9
        alternate = torch.zeros_like(world_rays)
        alternate[..., 1] = 1
        helper = torch.where(parallel[..., None], alternate, helper)
        tangent_1 = F.normalize(torch.linalg.cross(helper, world_rays, dim=-1), dim=-1, eps=eps)
        tangent_2 = torch.linalg.cross(world_rays, tangent_1, dim=-1)
        basis = torch.stack((tangent_1, tangent_2, world_rays), dim=-1)
        variance = torch.stack((tangent_sigma.square(), tangent_sigma.square(), normal_sigma.square()), dim=-1)
        covariance = basis @ torch.diag_embed(variance) @ basis.transpose(-1, -2)
        return confidence, covariance, normal_sigma.square()


class GeometricEvidenceBuilder(nn.Module):
    """Unproject multiscale VGGT patch evidence using OpenCV camera-from-world poses."""

    def __init__(self, calibrator: Optional[ConfidenceCovarianceCalibrator] = None) -> None:
        super().__init__()
        self.calibrator = calibrator or ConfidenceCovarianceCalibrator()

    def forward(
        self,
        images: Tensor,
        depth: Tensor,
        raw_confidence: Tensor,
        extrinsics_world_to_camera: Tensor,
        intrinsics: Tensor,
        patch_features: Tensor,
        valid_mask: Optional[Tensor] = None,
        reprojection_residual: Optional[Tensor] = None,
    ) -> List[EvidenceParticles]:
        """Return one variable-length evidence measure per batch element.

        Shapes are ``images[B,K,3,H,W]``, ``depth[B,K,H,W,1]`` (or without
        the final singleton), cameras ``[B,K,3,4]``, and patch features either
        ``[B,K,Hp,Wp,D]`` or flattened square ``[B,K,P,D]``.
        """

        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B,K,3,H,W]")
        b, k, _, h, w = images.shape
        if depth.ndim == 5:
            depth = depth.squeeze(-1)
        if raw_confidence.ndim == 5:
            raw_confidence = raw_confidence.squeeze(-1)
        if patch_features.ndim == 4:
            p = patch_features.shape[2]
            hp = int(round(sqrt(p)))
            if hp * hp != p:
                raise ValueError("flattened patch features must have a square patch grid")
            wp = hp
            patch_features = patch_features.reshape(b, k, hp, wp, -1)
        elif patch_features.ndim == 5:
            hp, wp = patch_features.shape[2:4]
        else:
            raise ValueError("patch_features must be [B,K,P,D] or [B,K,Hp,Wp,D]")
        dtype, device = depth.dtype, depth.device
        depth_patch = F.interpolate(depth.reshape(b * k, 1, h, w), size=(hp, wp), mode="bilinear", align_corners=False)
        confidence_patch = F.interpolate(raw_confidence.reshape(b * k, 1, h, w), size=(hp, wp), mode="bilinear", align_corners=False)
        color_patch = F.interpolate(images.reshape(b * k, 3, h, w), size=(hp, wp), mode="area")
        if valid_mask is not None:
            mask_patch = F.interpolate(valid_mask.reshape(b * k, 1, h, w).float(), size=(hp, wp), mode="nearest") > 0.5
        else:
            mask_patch = torch.ones_like(depth_patch, dtype=torch.bool)
        if reprojection_residual is not None:
            residual_patch = F.interpolate(
                reprojection_residual.reshape(b * k, 1, h, w), size=(hp, wp), mode="bilinear", align_corners=False
            )
        else:
            residual_patch = None

        u = (torch.arange(wp, dtype=dtype, device=device) + 0.5) * (w / wp) - 0.5
        v = (torch.arange(hp, dtype=dtype, device=device) + 0.5) * (h / hp) - 0.5
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        homogeneous = torch.stack((uu, vv, torch.ones_like(uu)), dim=-1).reshape(1, 1, hp * wp, 3)
        depth_flat = depth_patch.reshape(b, k, hp * wp)
        raw_flat = confidence_patch.reshape(b, k, hp * wp)
        intrinsics_inv = torch.linalg.inv(intrinsics)
        camera_direction = torch.einsum("bkij,bknj->bkni", intrinsics_inv, homogeneous.expand(b, k, -1, -1))
        camera_points = camera_direction * depth_flat[..., None]
        rotation_w2c = extrinsics_world_to_camera[..., :3, :3]
        translation_w2c = extrinsics_world_to_camera[..., :3, 3]
        rotation_c2w = rotation_w2c.transpose(-1, -2)
        camera_center = -torch.einsum("bkij,bkj->bki", rotation_c2w, translation_w2c)
        world_points = torch.einsum("bkij,bknj->bkni", rotation_c2w, camera_points) + camera_center[:, :, None]
        world_rays = F.normalize(
            torch.einsum("bkij,bknj->bkni", rotation_c2w, camera_direction), dim=-1
        )
        focal_mean = torch.sqrt(intrinsics[..., 0, 0] * intrinsics[..., 1, 1])[:, :, None].expand_as(depth_flat)
        pixel_footprint = depth_flat.new_full(depth_flat.shape, sqrt((h / hp) * (w / wp)))
        residual_flat = residual_patch.reshape(b, k, hp * wp) if residual_patch is not None else None
        calibrated, covariance, depth_variance = self.calibrator(
            raw_flat,
            depth_flat,
            focal_mean,
            pixel_footprint,
            world_rays,
            residual_flat,
        )
        represented_pixel_area = (h / hp) * (w / wp)
        fx_fy = (intrinsics[..., 0, 0] * intrinsics[..., 1, 1])[:, :, None]
        mass = calibrated * depth_flat.square() / fx_fy * represented_pixel_area
        mask = mask_patch.reshape(b, k, hp * wp) & torch.isfinite(depth_flat) & (depth_flat > 0)
        uv = torch.stack((uu, vv), dim=-1).reshape(1, 1, hp * wp, 2).expand(b, k, -1, -1)
        view = torch.arange(k, device=device).reshape(1, k, 1).expand(b, k, hp * wp)
        feature = patch_features.reshape(b, k, hp * wp, -1)
        color = color_patch.permute(0, 2, 3, 1).reshape(b, k, hp * wp, 3)

        result: List[EvidenceParticles] = []
        for batch_index in range(b):
            keep = mask[batch_index].reshape(-1)
            particles = EvidenceParticles(
                positions=world_points[batch_index].reshape(-1, 3)[keep],
                rays=world_rays[batch_index].reshape(-1, 3)[keep],
                features=feature[batch_index].reshape(-1, feature.shape[-1])[keep],
                covariance=covariance[batch_index].reshape(-1, 3, 3)[keep],
                confidence=calibrated[batch_index].reshape(-1)[keep],
                mass=mass[batch_index].reshape(-1)[keep],
                view_index=view[batch_index].reshape(-1)[keep],
                pixel_uv=uv[batch_index].reshape(-1, 2)[keep],
                extrinsics_world_to_camera=extrinsics_world_to_camera[batch_index],
                intrinsics=intrinsics[batch_index],
                depth_variance=depth_variance[batch_index].reshape(-1)[keep],
                colors=color[batch_index].reshape(-1, 3)[keep],
            )
            particles.validate()
            result.append(particles)
        return result


@torch.no_grad()
def build_sparse_transport_graph(
    atlas: PersistentOctreeAtlas,
    evidence: EvidenceParticles,
    config: ManifoldMappingConfig,
) -> SparseTransportGraph:
    """Construct the discrete radius-truncated support with coverage fallbacks.

    Radius membership and nearest-neighbor identities are discrete. Recording
    a ``cdist`` autograd tape cannot differentiate those indices and retains a
    large useless workspace. The selected-edge cost is evaluated afterwards
    from the original tensors and carries the complete conditional gradient.
    """

    active = atlas.active_indices
    centers = atlas.chart_centers[active]
    radii = config.support_radius_factor * atlas.cell_sides[active]
    particles = evidence.positions
    n, m = centers.shape[0], particles.shape[0]
    if n == 0 or m == 0:
        raise ValueError("atlas and evidence must both be non-empty")
    edge_source: List[Tensor] = []
    edge_target: List[Tensor] = []
    nearest_target_distance = centers.new_full((n,), torch.inf)
    nearest_target = torch.zeros(n, dtype=torch.int64, device=centers.device)
    nearest_source_distance = centers.new_full((m,), torch.inf)
    nearest_source = torch.zeros(m, dtype=torch.int64, device=centers.device)

    for source_start in range(0, n, config.atlas_chunk_size):
        source_end = min(source_start + config.atlas_chunk_size, n)
        source_centers = centers[source_start:source_end]
        source_radii = radii[source_start:source_end]
        for target_start in range(0, m, config.evidence_chunk_size):
            target_end = min(target_start + config.evidence_chunk_size, m)
            distance = torch.cdist(source_centers, particles[target_start:target_end])
            local_source, local_target = torch.nonzero(distance < source_radii[:, None], as_tuple=True)
            if local_source.numel():
                edge_source.append(local_source + source_start)
                edge_target.append(local_target + target_start)
            values, indices = distance.min(dim=1)
            improve = values < nearest_target_distance[source_start:source_end]
            nearest_target_distance[source_start:source_end] = torch.where(
                improve, values, nearest_target_distance[source_start:source_end]
            )
            nearest_target[source_start:source_end] = torch.where(
                improve, indices + target_start, nearest_target[source_start:source_end]
            )
            values, indices = distance.min(dim=0)
            improve = values < nearest_source_distance[target_start:target_end]
            nearest_source_distance[target_start:target_end] = torch.where(
                improve, values, nearest_source_distance[target_start:target_end]
            )
            nearest_source[target_start:target_end] = torch.where(
                improve, indices + source_start, nearest_source[target_start:target_end]
            )

    if config.ensure_source_support:
        edge_source.append(torch.arange(n, device=centers.device))
        edge_target.append(nearest_target)
    if config.ensure_target_support:
        edge_source.append(nearest_source)
        edge_target.append(torch.arange(m, device=centers.device))
    source = torch.cat(edge_source)
    target = torch.cat(edge_target)
    linear = source * m + target
    linear = torch.unique(linear, sorted=True)
    source, target = torch.div(linear, m, rounding_mode="floor"), linear.remainder(m)
    edge_index = torch.stack((source, target))
    return SparseTransportGraph(
        edge_index=edge_index,
        atlas_node_index=active,
        source_count=n,
        target_count=m,
        support_radius=radii,
    )


def _dct_projection(rows: int, columns: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """Deterministic approximately orthogonal projection; no random geometry path."""

    row = torch.arange(rows, dtype=dtype)[:, None]
    col = torch.arange(columns, dtype=dtype)[None]
    basis = torch.cos(torch.pi / columns * (col + 0.5) * row)
    basis[0] *= 1.0 / sqrt(2.0)
    return basis * sqrt(2.0 / columns)


class TransportCost(nn.Module):
    """Learnable positive coefficients for the specification's geometric cost."""

    def __init__(self, feature_dim: int, projected_dim: int = 64) -> None:
        super().__init__()
        self.raw_lambda_x = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.raw_lambda_ray = nn.Parameter(torch.tensor(_inverse_softplus(0.25)))
        self.raw_lambda_depth = nn.Parameter(torch.tensor(_inverse_softplus(0.5)))
        self.raw_lambda_feature = nn.Parameter(torch.tensor(_inverse_softplus(0.1)))
        self.raw_lambda_visibility = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.feature_projection = nn.Linear(feature_dim, projected_dim, bias=False)
        with torch.no_grad():
            self.feature_projection.weight.copy_(_dct_projection(projected_dim, feature_dim))

    @staticmethod
    def _positive(value: Tensor) -> Tensor:
        return F.softplus(value)

    def forward(
        self,
        atlas: PersistentOctreeAtlas,
        evidence: EvidenceParticles,
        graph: SparseTransportGraph,
        atlas_features: Optional[Tensor] = None,
        visibility_barrier: Optional[Tensor] = None,
    ) -> Tensor:
        source, target = graph.source, graph.target
        nodes = graph.atlas_node_index[source]
        delta = atlas.chart_centers[nodes] - evidence.positions[target]
        covariance = 0.5 * (
            evidence.covariance[target]
            + evidence.covariance[target].transpose(-1, -2)
        )
        # Cholesky whitening evaluates delta^T Sigma^-1 delta without forming
        # an explicit inverse and remains non-negative by construction.
        covariance_factor = torch.linalg.cholesky(covariance)
        whitened = torch.linalg.solve_triangular(
            covariance_factor, delta.unsqueeze(-1), upper=False
        ).squeeze(-1)
        mahalanobis = whitened.square().sum(-1)
        ray = evidence.rays[target]
        axial = torch.sum(ray * delta, dim=-1)
        perpendicular = delta - axial[:, None] * ray
        ray_distance = perpendicular.square().sum(-1)
        depth_distance = axial.square() / evidence.depth_variance[target].clamp_min(torch.finfo(delta.dtype).eps)
        feature_distance = torch.zeros_like(mahalanobis)
        if atlas_features is not None:
            if atlas_features.shape[0] != graph.source_count:
                raise ValueError("atlas_features must have one row per active source chart")
            image_feature = F.normalize(self.feature_projection(evidence.features[target]), dim=-1)
            chart_feature = F.normalize(atlas_features[source], dim=-1)
            cosine = torch.sum(image_feature * chart_feature, dim=-1).clamp(-1.0, 1.0)
            feature_distance = 1.0 - cosine
        if visibility_barrier is None:
            # The particle is an observed first surface along its camera ray.
            # A chart substantially *behind* it should not consume that target
            # mass; hidden charts instead retain source mass through unbalanced
            # OT. The calibrated depth variance makes this a probabilistic,
            # dimensionless one-sided barrier rather than a fixed distance.
            signed_depth_sigma = axial / torch.sqrt(
                evidence.depth_variance[target].clamp_min(
                    torch.finfo(delta.dtype).eps
                )
            )
            visibility = F.softplus(signed_depth_sigma - 2.0).square()
        else:
            if visibility_barrier.shape != mahalanobis.shape:
                raise ValueError("visibility_barrier must have one value per transport edge")
            if not bool(torch.all(torch.isfinite(visibility_barrier))) or bool(
                torch.any(visibility_barrier < 0)
            ):
                raise ValueError("visibility_barrier must be finite and non-negative")
            visibility = visibility_barrier
        return (
            self._positive(self.raw_lambda_x) * mahalanobis
            + self._positive(self.raw_lambda_ray) * ray_distance
            + self._positive(self.raw_lambda_depth) * depth_distance
            + self._positive(self.raw_lambda_feature) * feature_distance
            + self._positive(self.raw_lambda_visibility) * visibility
        )


def _real_l2_basis(direction: Tensor) -> Tensor:
    """Five real, even l=2 components in a fixed orthogonal convention."""

    x, y, z = direction.unbind(-1)
    root3 = sqrt(3.0)
    return torch.stack(
        (
            root3 * x * y,
            root3 * y * z,
            0.5 * (2.0 * z.square() - x.square() - y.square()),
            root3 * x * z,
            0.5 * root3 * (x.square() - y.square()),
        ),
        dim=-1,
    )


class GaugeCovariantChartWriter(nn.Module):
    """Write transported features as chart-local irreducible geometric moments."""

    def __init__(self, feature_dim: int, radial_basis_count: int = 4) -> None:
        super().__init__()
        self.radial_basis_count = radial_basis_count
        self.scalar_projection = nn.Linear(feature_dim, 48, bias=False)
        self.vector_projection = nn.Linear(feature_dim, 16, bias=False)
        self.tensor_projection = nn.Linear(feature_dim, 4, bias=False)
        with torch.no_grad():
            self.scalar_projection.weight.copy_(_dct_projection(48, feature_dim))
            self.vector_projection.weight.copy_(_dct_projection(16, feature_dim))
            self.tensor_projection.weight.copy_(_dct_projection(4, feature_dim))

    def _radial_basis(self, radius: Tensor, support: Tensor) -> Tensor:
        normalized = radius / support.clamp_min(torch.finfo(radius.dtype).eps)
        centers = torch.linspace(0.0, 1.0, self.radial_basis_count, dtype=radius.dtype, device=radius.device)
        width = 1.0 / max(1, self.radial_basis_count - 1)
        value = (1.0 - (normalized[:, None] - centers[None]).abs() / width).clamp_min(0).pow(3)
        return value / value.sum(-1, keepdim=True).clamp_min(torch.finfo(radius.dtype).eps)

    def forward(
        self,
        atlas: PersistentOctreeAtlas,
        evidence: EvidenceParticles,
        graph: SparseTransportGraph,
        plan: Tensor,
        metric_epsilon: float,
        metric_normal_weight: float,
        radial_support_factor: float,
        source_mass: Tensor,
        retention_shrinkage: float,
    ) -> Tuple[Tensor, Tensor, Tensor, IrrepMoments, Tensor]:
        source, target = graph.source, graph.target
        nodes = graph.atlas_node_index[source]
        count = graph.source_count
        transported_mass = _segment_sum(plan, source, count)
        denominator = transported_mass.clamp_min(torch.finfo(plan.dtype).eps)
        conditional_centers = _segment_sum(
            plan[:, None] * evidence.positions[target], source, count
        ) / denominator[:, None]
        retained_ratio = transported_mass / source_mass.clamp_min(torch.finfo(plan.dtype).eps)
        reliability = retained_ratio / (retained_ratio + retention_shrinkage)
        chart_center = atlas.chart_centers[graph.atlas_node_index]
        transported_centers = chart_center + reliability[:, None] * (
            conditional_centers - chart_center
        )

        covariance = 0.5 * (
            evidence.covariance[target]
            + evidence.covariance[target].transpose(-1, -2)
        )
        precision = torch.cholesky_inverse(torch.linalg.cholesky(covariance))
        conditional_metric = _segment_sum(
            plan[:, None, None] * precision, source, count
        ) / denominator[:, None, None]
        normal = atlas.chart_frames[graph.atlas_node_index, :, 2]
        eye = torch.eye(3, dtype=plan.dtype, device=plan.device)
        radius = atlas.chart_radii[graph.atlas_node_index].clamp_min(
            torch.finfo(plan.dtype).eps
        )
        baseline_metric = eye / radius[:, None, None].square()
        metric = metric_epsilon * eye + (
            reliability[:, None, None] * conditional_metric
            + (1.0 - reliability)[:, None, None] * baseline_metric
            + metric_normal_weight * normal[:, :, None] * normal[:, None, :]
        )

        world_delta = evidence.positions[target] - atlas.chart_centers[nodes]
        local_delta = torch.einsum("eji,ej->ei", atlas.chart_frames[nodes], world_delta)
        radius = torch.linalg.vector_norm(local_delta, dim=-1)
        direction = local_delta / radius[:, None].clamp_min(torch.finfo(radius.dtype).eps)
        support = radial_support_factor * atlas.cell_sides[nodes]
        radial = self._radial_basis(radius, support)

        scalar_coeff = self.scalar_projection(evidence.features[target])
        scalar_channel_basis = torch.arange(48, device=plan.device).remainder(self.radial_basis_count)
        scalar_coeff = scalar_coeff * radial[:, scalar_channel_basis]
        scalar = _segment_sum(plan[:, None] * scalar_coeff, source, count) / denominator[:, None]
        vector_coeff = self.vector_projection(evidence.features[target])
        vector_channel_basis = torch.arange(16, device=plan.device).remainder(self.radial_basis_count)
        vector_coeff = vector_coeff * radial[:, vector_channel_basis]
        vector = _segment_sum(
            plan[:, None, None] * vector_coeff[:, :, None] * direction[:, None, :], source, count
        ) / denominator[:, None, None]
        tensor_coeff = self.tensor_projection(evidence.features[target])
        tensor_channel_basis = torch.arange(4, device=plan.device).remainder(self.radial_basis_count)
        tensor_coeff = tensor_coeff * radial[:, tensor_channel_basis]
        l2 = _real_l2_basis(direction)
        tensor = _segment_sum(
            plan[:, None, None] * tensor_coeff[:, :, None] * l2[:, None, :], source, count
        ) / denominator[:, None, None]

        # The final 12 invariant channels explicitly carry transported
        # uncertainty/measure statistics rather than duplicating appearance.
        covariance_local = torch.einsum(
            "eji,ejk,ekl->eil", atlas.chart_frames[nodes], covariance, atlas.chart_frames[nodes]
        )
        uncertainty_edge = torch.stack(
            (
                evidence.confidence[target],
                evidence.mass[target],
                evidence.depth_variance[target],
                radius,
                local_delta[:, 0],
                local_delta[:, 1],
                local_delta[:, 2],
                covariance_local[:, 0, 0],
                covariance_local[:, 1, 1],
                covariance_local[:, 2, 2],
                covariance_local[:, 0, 1],
                torch.linalg.slogdet(covariance).logabsdet,
            ),
            dim=-1,
        )
        auxiliary = _segment_sum(plan[:, None] * uncertainty_edge, source, count) / denominator[:, None]
        scalar = reliability[:, None] * scalar
        vector = reliability[:, None, None] * vector
        tensor = reliability[:, None, None] * tensor
        auxiliary = reliability[:, None] * auxiliary
        irreps = IrrepMoments(
            scalar_0e=scalar,
            vector_1o=vector,
            tensor_2e=tensor,
            auxiliary_0e=auxiliary,
        )
        latent = irreps.pack()
        if latent.shape[-1] != 128:
            raise RuntimeError("internal irrep layout must total 128 channels")
        return transported_centers, transported_mass, metric, irreps, reliability


class ManifoldMappingOperator(nn.Module):
    """Complete sparse uncertainty-aware OT lift from particles to an atlas."""

    def __init__(self, feature_dim: int, config: Optional[ManifoldMappingConfig] = None) -> None:
        super().__init__()
        self.config = config or ManifoldMappingConfig()
        self.cost_model = TransportCost(feature_dim, self.config.feature_cost_dim)
        self.sinkhorn = ImplicitUnbalancedSinkhorn(self.config.sinkhorn)
        self.chart_writer = GaugeCovariantChartWriter(feature_dim, self.config.radial_basis_count)

    def forward(
        self,
        atlas: PersistentOctreeAtlas,
        evidence: EvidenceParticles,
        atlas_features: Optional[Tensor] = None,
        visibility_barrier: Optional[Tensor] = None,
    ) -> MappingResult:
        evidence.validate()
        graph = build_sparse_transport_graph(atlas, evidence, self.config)
        cost = self.cost_model(atlas, evidence, graph, atlas_features, visibility_barrier)
        # Atlas prior mass is explicit chart area.  Unobserved charts may retain
        # mass because the transport is unbalanced instead of being forced to
        # consume erroneous or background image evidence.
        source_mass = torch.pi * atlas.chart_radii[graph.atlas_node_index].square()
        target_mass = evidence.mass
        plan, diagnostics = self.sinkhorn(cost, source_mass, target_mass, graph.edge_index)
        centers, mass, metric, irreps, reliability = self.chart_writer(
            atlas,
            evidence,
            graph,
            plan,
            self.config.metric_epsilon,
            self.config.metric_normal_weight,
            self.config.radial_support_factor,
            source_mass,
            self.config.retention_shrinkage,
        )
        transported_color = None
        if evidence.colors is not None:
            color_sum = _segment_sum(
                plan[:, None] * evidence.colors[graph.target], graph.source, graph.source_count
            )
            conditional_color = color_sum / mass.clamp_min(torch.finfo(plan.dtype).eps)[:, None]
            transported_color = 0.5 + reliability[:, None] * (conditional_color - 0.5)
        return MappingResult(
            evidence=evidence,
            graph=graph,
            cost=cost,
            plan=plan,
            diagnostics=diagnostics,
            transported_centers=centers,
            transported_mass=mass,
            observation_reliability=reliability,
            transported_color=transported_color,
            riemannian_metric=metric,
            irreps=irreps,
            latent=irreps.pack(),
        )


__all__ = [
    "ConfidenceCovarianceCalibrator",
    "EvidenceParticles",
    "GaugeCovariantChartWriter",
    "GeometricEvidenceBuilder",
    "ImplicitSinkhornConfig",
    "ImplicitUnbalancedSinkhorn",
    "IrrepMoments",
    "ManifoldMappingConfig",
    "ManifoldMappingOperator",
    "MappingResult",
    "SinkhornDiagnostics",
    "SparseTransportGraph",
    "TransportCost",
    "build_sparse_transport_graph",
    "sparse_view_reprojection_variance",
]

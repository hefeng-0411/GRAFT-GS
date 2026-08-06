"""Compiled CUDA geometry primitives with linear-memory tensor contracts.

Discrete neighborhood identities are deliberately computed outside autograd.
All continuous edge costs are evaluated afterwards from the original tensors,
so no gradient is lost by omitting a distance-search tape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor, nn


def _validate_point_pair(query: Tensor, reference: Tensor) -> None:
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError("point tensors must have shape [N,D] and [M,D]")
    if query.shape[-1] != reference.shape[-1]:
        raise ValueError("point tensors must have the same coordinate dimension")
    if query.shape[0] == 0 or reference.shape[0] == 0:
        raise ValueError("geometric reductions require two non-empty point sets")
    if query.device != reference.device or query.dtype != reference.dtype:
        raise ValueError("point tensors must share device and dtype")
    if not query.dtype.is_floating_point:
        raise TypeError("point tensors must be floating point")


def _scene_radius(query: Tensor, reference: Tensor) -> Tensor:
    minimum = torch.minimum(query.amin(dim=0), reference.amin(dim=0))
    maximum = torch.maximum(query.amax(dim=0), reference.amax(dim=0))
    diagonal = torch.linalg.vector_norm(maximum - minimum)
    floor = torch.finfo(query.dtype).eps**0.5
    return (diagonal * (1.0 + 8.0 * floor) + floor).reshape(1)


@torch.no_grad()
def fixed_radius_neighbors(
    query: Tensor,
    reference: Tensor,
    radii: Tensor,
    *,
    max_neighbors: int,
    ensure_query_support: bool = True,
    ensure_reference_support: bool = True,
) -> Tensor:
    """Return a bounded-degree COO radius graph using FRNN on CUDA.

    The result has shape ``[2,E]`` with query indices in row zero. The CUDA
    path stores at most ``N*K + N + M`` candidates. Saturated radius rows fail
    loudly instead of silently changing the transport support.
    """

    _validate_point_pair(query, reference)
    if radii.shape != (query.shape[0],):
        raise ValueError("radii must contain one value per query point")
    if radii.device != query.device or radii.dtype != query.dtype:
        raise ValueError("radii must share the point device and dtype")
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be positive")
    if not bool(torch.all(torch.isfinite(radii))) or bool(torch.any(radii <= 0)):
        raise ValueError("neighbor radii must be finite and positive")

    if query.device.type != "cuda":
        raise RuntimeError("production neighborhood construction requires CUDA FRNN")
    if query.dtype != torch.float32:
        raise TypeError("FRNN grid hashing requires float32 point tensors")
    source, target = _frnn_radius_edges(
        query.contiguous(),
        reference.contiguous(),
        radii.contiguous(),
        max_neighbors=max_neighbors,
    )
    if ensure_query_support:
        covered = torch.zeros(
            query.shape[0], dtype=torch.bool, device=query.device
        )
        covered.scatter_(0, source, True)
        missing = torch.nonzero(~covered, as_tuple=False).reshape(-1)
        if missing.numel() > 0:
            nearest = _frnn_nearest_indices(query, reference)
            source = torch.cat((source, missing)).contiguous()
            target = torch.cat((target, nearest[missing])).contiguous()
    if ensure_reference_support:
        covered = torch.zeros(
            reference.shape[0], dtype=torch.bool, device=query.device
        )
        covered.scatter_(0, target, True)
        missing = torch.nonzero(~covered, as_tuple=False).reshape(-1)
        if missing.numel() > 0:
            nearest = _frnn_nearest_indices(reference, query)
            source = torch.cat((source, nearest[missing])).contiguous()
            target = torch.cat((target, missing)).contiguous()
    return torch.stack((source, target), dim=0).contiguous()


def _frnn_radius_edges(
    query: Tensor,
    reference: Tensor,
    radii: Tensor,
    *,
    max_neighbors: int,
) -> tuple[Tensor, Tensor]:
    try:
        import frnn
    except ImportError as error:
        raise RuntimeError(
            "CUDA neighborhood construction requires the local FRNN extension"
        ) from error

    neighbor_count = min(max_neighbors, reference.shape[0])
    if query.dtype != torch.float32 or reference.dtype != torch.float32:
        raise TypeError("FRNN CUDA kernels require float32 coordinates")
    work_query = query.detach().contiguous().unsqueeze(0).contiguous()
    work_reference = reference.detach().contiguous().unsqueeze(0).contiguous()
    search_radius = (
        radii.detach().amax().reshape(1).contiguous()
    )
    squared_distance, index, _, _ = frnn.frnn_grid_points(
        work_query,
        work_reference,
        K=neighbor_count,
        r=search_radius,
        return_nn=False,
        return_sorted=True,
    )
    squared_distance = squared_distance[0]
    index = index[0]
    radius_squared = radii.contiguous().square()[:, None]
    valid = (index >= 0) & (squared_distance < radius_squared)
    if neighbor_count < reference.shape[0] and bool(torch.any(valid[:, -1])):
        saturated = int(torch.count_nonzero(valid[:, -1]).item())
        raise RuntimeError(
            "FRNN radius support saturated max_neighbors; increase "
            f"frnn_max_neighbors (saturated_rows={saturated}, K={neighbor_count})"
        )
    source = (
        torch.arange(query.shape[0], device=query.device)[:, None]
        .expand_as(index)[valid]
    )
    return source.contiguous(), index[valid].contiguous()


def _frnn_nearest_indices(query: Tensor, reference: Tensor) -> Tensor:
    try:
        import frnn
    except ImportError as error:
        raise RuntimeError("CUDA nearest-neighbor queries require FRNN") from error

    if query.dtype != torch.float32 or reference.dtype != torch.float32:
        raise TypeError("FRNN CUDA kernels require float32 coordinates")
    work_query = query.detach().contiguous().unsqueeze(0).contiguous()
    work_reference = reference.detach().contiguous().unsqueeze(0).contiguous()
    radius = _scene_radius(work_query[0], work_reference[0]).to(
        device=query.device, dtype=torch.float32
    )
    _, index, _, _ = frnn.frnn_grid_points(
        work_query,
        work_reference,
        K=1,
        r=radius.contiguous(),
        return_nn=False,
        return_sorted=True,
    )
    index = index[0, :, 0]
    if bool(torch.any(index < 0)):
        raise RuntimeError(
            "FRNN failed to return a nearest neighbor inside scene bounds"
        )
    return index.contiguous()


@torch.no_grad()
def nearest_neighbor_indices(query: Tensor, reference: Tensor) -> Tensor:
    """Return exact discrete nearest-reference indices without a dense matrix."""

    _validate_point_pair(query, reference)
    if query.device.type != "cuda":
        raise RuntimeError("production nearest-neighbor queries require CUDA FRNN")
    return _frnn_nearest_indices(query.contiguous(), reference.contiguous())


def keops_squared_distance_minima(
    left: Tensor,
    right: Tensor,
) -> tuple[Tensor, Tensor]:
    """Exact bidirectional squared-distance minima via symbolic KeOps reductions."""

    _validate_point_pair(left, right)
    try:
        from pykeops.torch import LazyTensor
    except ImportError as error:
        raise RuntimeError(
            "symbolic all-pairs reductions require the local pykeops extension"
        ) from error

    compute_dtype = (
        torch.float32
        if left.dtype in (torch.float16, torch.bfloat16)
        else left.dtype
    )
    x_i = LazyTensor(left.to(dtype=compute_dtype).contiguous()[:, None, :])
    y_j = LazyTensor(right.to(dtype=compute_dtype).contiguous()[None, :, :])
    squared = ((x_i - y_j) ** 2).sum(-1)
    left_minimum = squared.min(dim=1).reshape(-1)
    right_minimum = squared.min(dim=0).reshape(-1)
    return (
        left_minimum.to(dtype=left.dtype),
        right_minimum.to(dtype=left.dtype),
    )


def keops_gaussian_reduction(
    query: Tensor,
    reference: Tensor,
    values: Tensor,
    bandwidth: Tensor,
) -> Tensor:
    """Apply a Gaussian kernel to features without materializing ``[N,M]``."""

    _validate_point_pair(query, reference)
    if values.ndim != 2 or values.shape[0] != reference.shape[0]:
        raise ValueError("Gaussian-reduction values must have shape [M,C]")
    if values.device != query.device or values.dtype != query.dtype:
        raise ValueError("Gaussian-reduction values must share point dtype/device")
    if bandwidth.numel() != 1 or not bool(bandwidth.detach() > 0):
        raise ValueError("Gaussian bandwidth must be a positive scalar")
    try:
        from pykeops.torch import LazyTensor
    except ImportError as error:
        raise RuntimeError("Gaussian reductions require pykeops") from error

    x_i = LazyTensor(query.contiguous()[:, None, :])
    y_j = LazyTensor(reference.contiguous()[None, :, :])
    value_j = LazyTensor(values.contiguous()[None, :, :])
    bandwidth_parameter = LazyTensor(
        bandwidth.contiguous().reshape(1, 1, 1)
    )
    squared = ((x_i - y_j) ** 2).sum(-1)
    kernel = (-0.5 * squared / bandwidth_parameter.square()).exp()
    return (kernel * value_j).sum(dim=1)


def keops_compact_partition(
    query: Tensor,
    centers: Tensor,
    support: Tensor,
    values: Tensor,
) -> Tensor:
    """Compact bump partition-of-unity reduction with nearest fallback."""

    _validate_point_pair(query, centers)
    if support.shape != (centers.shape[0],):
        raise ValueError("partition support must have one radius per center")
    if values.ndim != 2 or values.shape[0] != centers.shape[0]:
        raise ValueError("partition values must have shape [centers,C]")
    if support.device != query.device or support.dtype != query.dtype:
        raise ValueError("partition support must share point dtype/device")
    if values.device != query.device or values.dtype != query.dtype:
        raise ValueError("partition values must share point dtype/device")
    try:
        from pykeops.torch import LazyTensor
    except ImportError as error:
        raise RuntimeError("compact partition reductions require pykeops") from error

    x_i = LazyTensor(query.contiguous()[:, None, :])
    c_j = LazyTensor(centers.contiguous()[None, :, :])
    radius_j = LazyTensor(support.contiguous()[None, :, None])
    value_j = LazyTensor(values.contiguous()[None, :, :])
    normalized_squared = ((x_i - c_j) ** 2).sum(-1) / radius_j.square()
    interior_margin = (1.0 - normalized_squared).relu()
    bump = (-1.0 / (interior_margin + 1.0e-12)).exp()
    total = bump.sum(dim=1).reshape(-1, 1)
    weighted = (bump * value_j).sum(dim=1)
    nearest = values[nearest_neighbor_indices(query, centers)]
    return torch.where(
        total > torch.finfo(total.dtype).tiny,
        weighted / total.clamp_min(torch.finfo(total.dtype).tiny),
        nearest,
    )


def geomloss_sinkhorn_divergence(
    left: Tensor,
    right: Tensor,
    *,
    left_mass: Optional[Tensor] = None,
    right_mass: Optional[Tensor] = None,
    blur: float = 0.01,
    reach: Optional[float] = None,
    scaling: float = 0.8,
) -> Tensor:
    """Debiased entropic OT using GeomLoss' KeOps ``online`` backend."""

    _validate_point_pair(left, right)
    if blur <= 0 or not 0 < scaling < 1:
        raise ValueError(
            "GeomLoss blur must be positive and scaling must lie in (0,1)"
        )
    if reach is not None and reach <= 0:
        raise ValueError("unbalanced reach must be positive")
    try:
        from geomloss import SamplesLoss
    except ImportError as error:
        raise RuntimeError(
            "Sinkhorn divergence requires the local geomloss and pykeops extensions"
        ) from error

    if left_mass is None:
        left_mass = left.new_full((left.shape[0],), 1.0 / left.shape[0])
    if right_mass is None:
        right_mass = right.new_full((right.shape[0],), 1.0 / right.shape[0])
    if left_mass.shape != (left.shape[0],) or right_mass.shape != (
        right.shape[0],
    ):
        raise ValueError("OT masses must contain one scalar per support point")
    loss = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=blur,
        reach=reach,
        scaling=scaling,
        debias=True,
        backend="online",
    )
    return loss(
        left_mass.contiguous(),
        left.contiguous(),
        right_mass.contiguous(),
        right.contiguous(),
    )


@dataclass(frozen=True)
class KeOpsFUGWConfig:
    """Fused unbalanced Gromov-Wasserstein block-coordinate policy."""

    alpha: float = 0.15
    max_iterations: int = 6
    tolerance: float = 1.0e-5
    normalize_geometry: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("FUGW alpha must lie in [0,1]")
        if self.max_iterations < 1:
            raise ValueError("FUGW requires at least one BCD iteration")
        if self.tolerance <= 0.0:
            raise ValueError("FUGW tolerance must be positive")


@dataclass
class KeOpsFUGWDiagnostics:
    iterations: int
    residual: float
    converged: bool
    structural_cost: Tensor
    linearized_cost: Tensor
    feature_objective: Tensor
    structural_objective: Tensor
    objective: Tensor
    transport_iterations: int
    transport_residual: float
    transport_effective_tolerance: float
    internal_minimum_log_plan: float
    storage_underflow_edges: int
    storage_zero_source_rows: int
    storage_zero_target_columns: int
    internal_solve_dtype: str
    storage_underflow_mass_fraction: float
    storage_zero_source_mass_fraction: float
    storage_zero_target_mass_fraction: float
    storage_relative_l1_error: float


def _validate_sparse_fugw_inputs(
    feature_cost: Tensor,
    source_geometry: Tensor,
    target_geometry: Tensor,
    source_mass: Tensor,
    target_mass: Tensor,
    edge_index: Tensor,
) -> None:
    _validate_point_pair(source_geometry, target_geometry)
    if source_geometry.device.type != "cuda":
        raise RuntimeError("KeOps FUGW is a CUDA production operator")
    if feature_cost.ndim != 1:
        raise ValueError("FUGW feature cost must have shape [E]")
    if edge_index.shape != (2, feature_cost.numel()):
        raise ValueError("FUGW edge_index must have shape [2,E]")
    if edge_index.dtype != torch.int64:
        raise TypeError("FUGW edge indices must use int64")
    if source_mass.shape != (source_geometry.shape[0],):
        raise ValueError("FUGW source mass must have shape [N]")
    if target_mass.shape != (target_geometry.shape[0],):
        raise ValueError("FUGW target mass must have shape [M]")
    tensors = (feature_cost, source_mass, target_mass)
    if any(value.device != source_geometry.device for value in tensors):
        raise ValueError("FUGW tensors must share one CUDA device")
    if any(value.dtype != source_geometry.dtype for value in tensors):
        raise ValueError("FUGW geometry, costs, and masses must share a dtype")
    if edge_index.device != source_geometry.device:
        raise ValueError("FUGW support must reside on the geometry CUDA device")
    if not bool(torch.all(torch.isfinite(feature_cost))):
        raise ValueError("FUGW feature costs must be finite")
    if bool(torch.any(feature_cost < 0.0)):
        raise ValueError("FUGW feature costs must be non-negative")
    if not bool(torch.all(torch.isfinite(source_mass))) or not bool(
        torch.all(torch.isfinite(target_mass))
    ):
        raise ValueError("FUGW masses must be finite")
    if bool(torch.any(source_mass < 0.0)) or bool(torch.any(target_mass < 0.0)):
        raise ValueError("FUGW masses must be non-negative")
    if not bool(torch.any(source_mass > 0.0)) or not bool(
        torch.any(target_mass > 0.0)
    ):
        raise ValueError("both FUGW measures must carry positive mass")


def _normalized_measure_geometry(points: Tensor, mass: Tensor) -> Tensor:
    probability = mass / mass.sum().clamp_min(torch.finfo(mass.dtype).tiny)
    center = torch.sum(probability[:, None] * points, dim=0)
    centered = points - center
    scale = torch.sqrt(
        torch.sum(probability * centered.square().sum(dim=-1))
    ).clamp_min(torch.finfo(points.dtype).eps**0.5)
    return (centered / scale).contiguous()


def keops_fugw_tensor_product(
    source_geometry: Tensor,
    target_geometry: Tensor,
    edge_index: Tensor,
    coupling: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Evaluate the exact sparse-support fourth-order GW contraction.

    For transport edge ``e=(i,j)``, the returned structural vector is

    ``sum_f (C1[i,k_f] - C2[j,l_f])^2 coupling[f]``.

    The second return value is exactly ``(C1 coupling C2^T)[i,j]`` restricted
    to the FRNN support. KeOps streams the symbolic ``E x E`` reduction and
    never instantiates ``C1``, ``C2``, or a fourth-order loss tensor.
    """

    _validate_point_pair(source_geometry, target_geometry)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("FUGW support must have shape [2,E]")
    if coupling.shape != (edge_index.shape[1],):
        raise ValueError("FUGW coupling must contain one value per edge")
    if edge_index.dtype != torch.int64:
        raise TypeError("FUGW support indices must use int64")
    if coupling.device != source_geometry.device or coupling.dtype != source_geometry.dtype:
        raise ValueError("FUGW coupling must share geometry dtype and device")
    if source_geometry.device.type != "cuda":
        raise RuntimeError("KeOps FUGW tensor products require CUDA")
    try:
        from pykeops.torch import LazyTensor
    except ImportError as error:
        raise RuntimeError("FUGW tensor products require local pykeops") from error

    support = edge_index.contiguous()
    source = support[0].contiguous()
    target = support[1].contiguous()
    source_edge = source_geometry.contiguous().index_select(0, source).contiguous()
    target_edge = target_geometry.contiguous().index_select(0, target).contiguous()
    mass_edge = coupling.contiguous().reshape(1, -1, 1).contiguous()

    source_i = LazyTensor(source_edge[:, None, :].contiguous())
    source_f = LazyTensor(source_edge[None, :, :].contiguous())
    target_i = LazyTensor(target_edge[:, None, :].contiguous())
    target_f = LazyTensor(target_edge[None, :, :].contiguous())
    mass_f = LazyTensor(mass_edge)
    c1 = ((source_i - source_f) ** 2).sum(-1)
    c2 = ((target_i - target_f) ** 2).sum(-1)

    c1_squared = ((c1**2) * mass_f).sum(dim=1).reshape(-1)
    c2_squared = ((c2**2) * mass_f).sum(dim=1).reshape(-1)
    cross = (c1 * c2 * mass_f).sum(dim=1).reshape(-1)
    structural = (c1_squared + c2_squared - 2.0 * cross).clamp_min(0.0)
    return structural.contiguous(), cross.contiguous()


def _run_keops_fugw_bcd(
    feature_cost: Tensor,
    source_geometry: Tensor,
    target_geometry: Tensor,
    source_mass: Tensor,
    target_mass: Tensor,
    edge_index: Tensor,
    transport_solver: nn.Module,
    *,
    alpha: float,
    max_iterations: int,
    tolerance: float,
    fixed_iterations: bool,
) -> tuple[Tensor, Tensor, int, Tensor, object]:
    plan, last_transport_diagnostics = transport_solver(
        feature_cost.contiguous(),
        source_mass.contiguous(),
        target_mass.contiguous(),
        edge_index.contiguous(),
    )
    auxiliary = plan
    residual = feature_cost.new_tensor(float("inf"))
    completed = max_iterations
    for iteration in range(max_iterations):
        structural, _ = keops_fugw_tensor_product(
            source_geometry, target_geometry, edge_index, auxiliary
        )
        plan_cost = (
            (1.0 - alpha) * feature_cost + alpha * structural
        ).contiguous()
        next_plan, _ = transport_solver(
            plan_cost,
            source_mass.contiguous(),
            target_mass.contiguous(),
            edge_index.contiguous(),
        )
        reverse_structural, _ = keops_fugw_tensor_product(
            source_geometry, target_geometry, edge_index, next_plan
        )
        auxiliary_cost = (
            (1.0 - alpha) * feature_cost + alpha * reverse_structural
        ).contiguous()
        next_auxiliary, last_transport_diagnostics = transport_solver(
            auxiliary_cost,
            source_mass.contiguous(),
            target_mass.contiguous(),
            edge_index.contiguous(),
        )
        plan_scale = plan.abs().amax().clamp_min(1.0)
        auxiliary_scale = auxiliary.abs().amax().clamp_min(1.0)
        residual = torch.maximum(
            (next_plan - plan).abs().amax() / plan_scale,
            (next_auxiliary - auxiliary).abs().amax() / auxiliary_scale,
        )
        plan = next_plan
        auxiliary = next_auxiliary
        if not fixed_iterations and bool(residual.detach() <= tolerance):
            completed = iteration + 1
            break
    return (
        (0.5 * (plan + auxiliary)).contiguous(),
        auxiliary.contiguous(),
        completed,
        residual,
        last_transport_diagnostics,
    )


class _KeOpsFUGWBCD(torch.autograd.Function):
    """Recomputed custom adjoint for the finite FUGW BCD fixed-point map."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        feature_cost: Tensor,
        source_geometry: Tensor,
        target_geometry: Tensor,
        source_mass: Tensor,
        target_mass: Tensor,
        edge_index: Tensor,
        transport_solver: nn.Module,
        alpha: float,
        max_iterations: int,
        tolerance: float,
    ) -> tuple[Tensor, Tensor]:
        plan, _, completed, residual, transport_diagnostics = _run_keops_fugw_bcd(
            feature_cost,
            source_geometry,
            target_geometry,
            source_mass,
            target_mass,
            edge_index,
            transport_solver,
            alpha=alpha,
            max_iterations=max_iterations,
            tolerance=tolerance,
            fixed_iterations=False,
        )
        ctx.save_for_backward(
            feature_cost,
            source_geometry,
            target_geometry,
            source_mass,
            target_mass,
            edge_index,
        )
        ctx.transport_solver = transport_solver
        ctx.alpha = alpha
        ctx.completed = completed
        ctx.tolerance = tolerance
        status = torch.tensor(
            (
                float(completed),
                float(residual.detach().cpu()),
                float(transport_diagnostics.iterations),
                float(transport_diagnostics.fixed_point_residual),
                float(transport_diagnostics.effective_tolerance),
                float(transport_diagnostics.internal_minimum_log_plan),
                float(transport_diagnostics.storage_underflow_edges),
                float(transport_diagnostics.storage_zero_source_rows),
                float(transport_diagnostics.storage_zero_target_columns),
                64.0 if transport_diagnostics.internal_solve_dtype == "float64" else 32.0,
                float(transport_diagnostics.storage_underflow_mass_fraction),
                float(transport_diagnostics.storage_zero_source_mass_fraction),
                float(transport_diagnostics.storage_zero_target_mass_fraction),
                float(transport_diagnostics.storage_relative_l1_error),
            ),
            dtype=torch.float64,
            device=feature_cost.device,
        ).contiguous()
        ctx.mark_non_differentiable(status)
        return plan, status

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_plan: Optional[Tensor],
        grad_status: Optional[Tensor],
    ) -> tuple[Optional[Tensor], ...]:
        del grad_status
        if grad_plan is None:
            return (None,) * 10
        saved = ctx.saved_tensors
        differentiable = []
        active_positions = []
        for position, (value, required) in enumerate(
            zip(saved[:5], ctx.needs_input_grad[:5])
        ):
            replay_value = value.detach().requires_grad_(required)
            differentiable.append(replay_value)
            if required:
                active_positions.append(position)
        edge_index = saved[5]
        active_inputs = [differentiable[index] for index in active_positions]
        gradients_by_position: dict[int, Optional[Tensor]] = {}
        if active_inputs:
            with torch.enable_grad():
                replayed, _, _, _, _ = _run_keops_fugw_bcd(
                    differentiable[0],
                    differentiable[1],
                    differentiable[2],
                    differentiable[3],
                    differentiable[4],
                    edge_index,
                    ctx.transport_solver,
                    alpha=ctx.alpha,
                    max_iterations=ctx.completed,
                    tolerance=ctx.tolerance,
                    fixed_iterations=True,
                )
            active_gradients = torch.autograd.grad(
                replayed,
                active_inputs,
                grad_plan.contiguous(),
                allow_unused=True,
                create_graph=torch.is_grad_enabled(),
            )
            gradients_by_position = dict(zip(active_positions, active_gradients))
        return (
            gradients_by_position.get(0),
            gradients_by_position.get(1),
            gradients_by_position.get(2),
            gradients_by_position.get(3),
            gradients_by_position.get(4),
            None,
            None,
            None,
            None,
            None,
        )


class KeOpsFUGWSolver(nn.Module):
    """Linear-storage FUGW BCD on a bounded FRNN transport support."""

    def __init__(self, config: KeOpsFUGWConfig = KeOpsFUGWConfig()) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        feature_cost: Tensor,
        source_geometry: Tensor,
        target_geometry: Tensor,
        source_mass: Tensor,
        target_mass: Tensor,
        edge_index: Tensor,
        transport_solver: nn.Module,
    ) -> tuple[Tensor, KeOpsFUGWDiagnostics]:
        _validate_sparse_fugw_inputs(
            feature_cost,
            source_geometry,
            target_geometry,
            source_mass,
            target_mass,
            edge_index,
        )
        feature_cost = feature_cost.contiguous()
        source_geometry = source_geometry.contiguous()
        target_geometry = target_geometry.contiguous()
        source_mass = source_mass.contiguous()
        target_mass = target_mass.contiguous()
        edge_index = edge_index.contiguous()
        if self.config.normalize_geometry:
            source_geometry = _normalized_measure_geometry(
                source_geometry, source_mass
            )
            target_geometry = _normalized_measure_geometry(
                target_geometry, target_mass
            )
        plan, status = _KeOpsFUGWBCD.apply(
            feature_cost,
            source_geometry,
            target_geometry,
            source_mass,
            target_mass,
            edge_index,
            transport_solver,
            self.config.alpha,
            self.config.max_iterations,
            self.config.tolerance,
        )
        structural_cost, _ = keops_fugw_tensor_product(
            source_geometry, target_geometry, edge_index, plan
        )
        linearized_cost = (
            (1.0 - self.config.alpha) * feature_cost
            + self.config.alpha * structural_cost
        ).contiguous()
        feature_objective = (1.0 - self.config.alpha) * torch.sum(
            feature_cost * plan
        )
        structural_objective = self.config.alpha * torch.sum(
            plan * structural_cost
        )
        objective = feature_objective + structural_objective
        residual = float(status[1].detach().cpu())
        diagnostics = KeOpsFUGWDiagnostics(
            iterations=int(status[0].item()),
            residual=residual,
            converged=residual <= self.config.tolerance,
            structural_cost=structural_cost,
            linearized_cost=linearized_cost,
            feature_objective=feature_objective,
            structural_objective=structural_objective,
            objective=objective,
            transport_iterations=int(status[2].item()),
            transport_residual=float(status[3].item()),
            transport_effective_tolerance=float(status[4].item()),
            internal_minimum_log_plan=float(status[5].item()),
            storage_underflow_edges=int(status[6].item()),
            storage_zero_source_rows=int(status[7].item()),
            storage_zero_target_columns=int(status[8].item()),
            internal_solve_dtype=f"float{int(status[9].item())}",
            storage_underflow_mass_fraction=float(status[10].item()),
            storage_zero_source_mass_fraction=float(status[11].item()),
            storage_zero_target_mass_fraction=float(status[12].item()),
            storage_relative_l1_error=float(status[13].item()),
        )
        if not diagnostics.converged:
            raise RuntimeError(
                "KeOps FUGW BCD did not converge: "
                f"iterations={diagnostics.iterations}, "
                f"residual={diagnostics.residual:.6e}, "
                f"tolerance={self.config.tolerance:.6e}"
            )
        return plan.contiguous(), diagnostics


@dataclass
class VolumetricRenderResult:
    color: Tensor
    opacity: Tensor
    depth: Tensor
    weights: Tensor
    transmittance: Tensor
    alpha: Tensor


def nerfacc_volume_render(
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Tensor,
    density: Tensor,
    color: Tensor,
    *,
    ray_count: int,
) -> VolumetricRenderResult:
    """Fused packed transmittance and accumulation through ``nerfacc``."""

    if ray_count < 1:
        raise ValueError("ray_count must be positive")
    if not (
        t_starts.shape == t_ends.shape == ray_indices.shape == density.shape
    ):
        raise ValueError(
            "packed intervals, ray indices, and density must share shape"
        )
    if color.shape != (density.numel(), 3):
        raise ValueError("packed colors must have shape [samples,3]")
    if ray_indices.dtype != torch.int64:
        raise TypeError("ray_indices must use int64")
    try:
        from nerfacc import accumulate_along_rays, render_weight_from_density
    except ImportError as error:
        raise RuntimeError(
            "volumetric integration requires the local nerfacc extension"
        ) from error

    weights, transmittance, alpha = render_weight_from_density(
        t_starts.contiguous(),
        t_ends.contiguous(),
        density.contiguous(),
        ray_indices=ray_indices.contiguous(),
        n_rays=ray_count,
    )
    midpoint = (0.5 * (t_starts + t_ends))[:, None]
    rendered_color = accumulate_along_rays(
        weights, color.contiguous(), ray_indices, ray_count
    )
    opacity = accumulate_along_rays(weights, None, ray_indices, ray_count)
    depth = accumulate_along_rays(weights, midpoint, ray_indices, ray_count)
    depth = depth / opacity.clamp_min(torch.finfo(depth.dtype).eps)
    return VolumetricRenderResult(
        rendered_color, opacity, depth, weights, transmittance, alpha
    )


class NerfaccOccupancyGrid(nn.Module):
    """Thin module wrapper around ``nerfacc.OccGridEstimator``."""

    def __init__(
        self,
        roi_aabb: Tensor,
        resolution: int = 128,
        levels: int = 1,
    ) -> None:
        super().__init__()
        if roi_aabb.shape != (6,):
            raise ValueError("occupancy-grid AABB must have shape [6]")
        try:
            from nerfacc import OccGridEstimator
        except ImportError as error:
            raise RuntimeError(
                "occupancy-grid acceleration requires nerfacc"
            ) from error
        self.estimator = OccGridEstimator(
            roi_aabb.contiguous(), resolution=resolution, levels=levels
        )

    @torch.no_grad()
    def update(
        self,
        step: int,
        occupancy_fn: Callable[[Tensor], Tensor],
        *,
        threshold: float = 1.0e-2,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
        interval: int = 16,
    ) -> None:
        self.estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occupancy_fn,
            occ_thre=threshold,
            ema_decay=ema_decay,
            warmup_steps=warmup_steps,
            n=interval,
        )

    @torch.no_grad()
    def sample(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        *,
        sigma_fn: Optional[Callable[..., Tensor]] = None,
        near: float = 0.0,
        far: float = 1.0e10,
        step_size: float = 1.0e-3,
        early_stop_epsilon: float = 1.0e-4,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.estimator.sampling(
            rays_o.contiguous(),
            rays_d.contiguous(),
            sigma_fn=sigma_fn,
            near_plane=near,
            far_plane=far,
            render_step_size=step_size,
            early_stop_eps=early_stop_epsilon,
        )


__all__ = [
    "KeOpsFUGWConfig",
    "KeOpsFUGWDiagnostics",
    "KeOpsFUGWSolver",
    "NerfaccOccupancyGrid",
    "VolumetricRenderResult",
    "fixed_radius_neighbors",
    "geomloss_sinkhorn_divergence",
    "keops_compact_partition",
    "keops_fugw_tensor_product",
    "keops_gaussian_reduction",
    "keops_squared_distance_minima",
    "nearest_neighbor_indices",
    "nerfacc_volume_render",
]

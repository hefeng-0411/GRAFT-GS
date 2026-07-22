"""Control-barrier projection and nonlinear feasibility certification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch import Tensor

from .geometry import ManifoldState, ManifoldTangent, retract


def _point_segment_distance_squared(point: Tensor, start: Tensor, end: Tensor) -> Tensor:
    direction = end - start
    parameter = torch.sum((point - start) * direction, dim=-1) / direction.square().sum(-1).clamp_min(1.0e-16)
    parameter = parameter.clamp(0.0, 1.0)
    closest = start + parameter[:, None] * direction
    return (point - closest).square().sum(-1)


def _segment_distance_squared(p0: Tensor, p1: Tensor, q0: Tensor, q1: Tensor) -> Tensor:
    """Exact piecewise squared distance between batched closed segments."""

    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a = u.square().sum(-1)
    b = torch.sum(u * v, dim=-1)
    c = v.square().sum(-1)
    d = torch.sum(u * w, dim=-1)
    e = torch.sum(v * w, dim=-1)
    denominator = a * c - b.square()
    s = (b * e - c * d) / denominator.clamp_min(1.0e-16)
    t = (a * e - b * d) / denominator.clamp_min(1.0e-16)
    interior_valid = (denominator > 1.0e-16) & (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
    interior = (w + s[:, None] * u - t[:, None] * v).square().sum(-1)
    endpoint = torch.stack(
        (
            _point_segment_distance_squared(p0, q0, q1),
            _point_segment_distance_squared(p1, q0, q1),
            _point_segment_distance_squared(q0, p0, p1),
            _point_segment_distance_squared(q1, p0, p1),
        ),
        dim=-1,
    ).amin(-1)
    return torch.where(interior_valid, torch.minimum(interior, endpoint), endpoint)


def _point_triangle_interior_distance_squared(point: Tensor, triangle: Tensor) -> Tensor:
    """Plane distance when projection is inside; infinity otherwise."""

    a, b, c = triangle.unbind(1)
    edge_0, edge_1 = b - a, c - a
    normal = torch.linalg.cross(edge_0, edge_1, dim=-1)
    normal_squared = normal.square().sum(-1).clamp_min(1.0e-16)
    signed_numerator = torch.sum((point - a) * normal, dim=-1)
    projection = point - (signed_numerator / normal_squared)[:, None] * normal
    relative = projection - a
    d00 = edge_0.square().sum(-1)
    d01 = torch.sum(edge_0 * edge_1, dim=-1)
    d11 = edge_1.square().sum(-1)
    d20 = torch.sum(relative * edge_0, dim=-1)
    d21 = torch.sum(relative * edge_1, dim=-1)
    denominator = (d00 * d11 - d01.square()).clamp_min(1.0e-16)
    barycentric_b = (d11 * d20 - d01 * d21) / denominator
    barycentric_c = (d00 * d21 - d01 * d20) / denominator
    inside = (barycentric_b >= 0.0) & (barycentric_c >= 0.0) & (barycentric_b + barycentric_c <= 1.0)
    distance = signed_numerator.square() / normal_squared
    return torch.where(inside, distance, torch.full_like(distance, torch.inf))


def triangle_distance_squared(left: Tensor, right: Tensor) -> Tensor:
    """Exact triangle distance from all vertex-face and edge-edge cases."""

    candidates = []
    for corner in range(3):
        candidates.append(_point_triangle_interior_distance_squared(left[:, corner], right))
        candidates.append(_point_triangle_interior_distance_squared(right[:, corner], left))
    edges = ((0, 1), (1, 2), (2, 0))
    for i, j in edges:
        for k, l in edges:
            candidates.append(_segment_distance_squared(left[:, i], left[:, j], right[:, k], right[:, l]))
    return torch.stack(candidates, dim=-1).amin(-1)


@dataclass(frozen=True)
class BarrierConfig:
    minimum_face_area: float = 1.0e-8
    minimum_orientation_cosine: float = 0.1
    minimum_separation: float = 1.0e-4
    minimum_covariance_eigenvalue: float = 1.0e-8
    maximum_covariance_eigenvalue: float = 1.0
    activation_margin: float = 1.0e-3
    decay_rate: float = 5.0
    dual_iterations: int = 96
    dual_tolerance: float = 1.0e-7
    dual_regularization: float = 1.0e-8
    dual_check_interval: int = 8
    maximum_backtracks: int = 14
    backtrack_factor: float = 0.5
    maximum_position_speed: float = 5.0e-2
    restoration_iterations: int = 12
    restoration_relative_margin: float = 5.0e-2

    def __post_init__(self) -> None:
        if min(
            self.minimum_face_area,
            self.minimum_separation,
            self.minimum_covariance_eigenvalue,
            self.activation_margin,
            self.decay_rate,
            self.dual_regularization,
            self.maximum_position_speed,
            self.restoration_relative_margin,
        ) <= 0:
            raise ValueError("barrier scales, margins, regularization, and speed bound must be positive")
        if not -1.0 < self.minimum_orientation_cosine < 1.0:
            raise ValueError("minimum_orientation_cosine must lie in (-1,1)")
        if self.maximum_covariance_eigenvalue <= self.minimum_covariance_eigenvalue:
            raise ValueError("maximum covariance eigenvalue must exceed the minimum")
        if not 0.0 < self.backtrack_factor < 1.0:
            raise ValueError("backtrack_factor must lie in (0,1)")
        if (
            self.dual_iterations < 1
            or self.dual_check_interval < 1
            or self.maximum_backtracks < 0
        ):
            raise ValueError("barrier solver iterations must be positive and backtracks non-negative")
        if self.restoration_iterations < 1:
            raise ValueError("restoration_iterations must be positive")


@dataclass
class FeasibilityReport:
    feasible: bool
    minimum_area_margin: float
    minimum_orientation_margin: float
    minimum_separation_margin: float
    minimum_covariance_margin: float
    maximum_covariance_margin: float
    projected_constraints: int = 0
    dual_residual: float = 0.0
    minimum_linearized_margin: float = float("inf")
    accepted_step: float = 0.0
    restoration_iterations: int = 0
    restoration_maximum_displacement: float = 0.0


class BarrierProjector:
    """Metric projection onto linearized CBF constraints plus exact line search.

    Collision broad phase is exact over the unit-time flow provided every
    vertex speed is bounded by ``maximum_position_speed``: a pair initially
    farther than ``d_min + 2 v_max`` cannot enter the forbidden set.  This
    converts the all-pairs certificate into a sparse, fixed constraint family.
    """

    def __init__(self, reference: ManifoldState, config: BarrierConfig = BarrierConfig()) -> None:
        self.config = config
        self.faces = reference.complex.faces
        face_position = reference.position[self.faces]
        cross = torch.linalg.cross(face_position[:, 1] - face_position[:, 0], face_position[:, 2] - face_position[:, 0])
        self.reference_face_normal = F.normalize(cross, dim=-1)
        diagnostic_face_position = reference.position.detach().to(dtype=torch.float64)[
            self.faces
        ]
        diagnostic_cross = torch.linalg.cross(
            diagnostic_face_position[:, 1] - diagnostic_face_position[:, 0],
            diagnostic_face_position[:, 2] - diagnostic_face_position[:, 0],
            dim=-1,
        )
        self.reference_face_normal_diagnostics = F.normalize(
            diagnostic_cross,
            dim=-1,
        )
        adjacent = {tuple(sorted(edge)) for edge in reference.complex.edges.tolist()}
        search_radius = config.minimum_separation + 2.0 * config.maximum_position_speed
        position_cpu = reference.position.detach().to(device="cpu", dtype=torch.float64).numpy()
        broad_phase = cKDTree(position_cpu).query_pairs(search_radius, output_type="ndarray")
        broad_phase = np.asarray(broad_phase, dtype=np.int64).reshape(-1, 2)
        pairs: List[Tuple[int, int]] = [
            (int(i), int(j)) for i, j in broad_phase if (int(i), int(j)) not in adjacent
        ]
        self.nonlocal_pairs = torch.tensor(pairs, dtype=torch.int64, device=reference.position.device).reshape(-1, 2)
        self.nonlocal_face_pairs = self._build_face_broad_phase(reference)

    def _build_face_broad_phase(self, reference: ManifoldState) -> Tensor:
        faces_cpu = reference.complex.faces.detach().cpu().numpy()
        if faces_cpu.shape[0] == 0:
            return torch.empty(0, 2, dtype=torch.int64, device=reference.position.device)
        vertex = reference.position.detach().to(device="cpu", dtype=torch.float64).numpy()
        triangle = vertex[faces_cpu]
        center = triangle.mean(axis=1)
        radius = np.linalg.norm(triangle - center[:, None], axis=-1).max(axis=1)
        padding = self.config.minimum_separation + 2.0 * self.config.maximum_position_speed
        tree = cKDTree(center)
        maximum_radius = float(radius.max())
        pairs: List[Tuple[int, int]] = []
        for i, neighbors in enumerate(tree.query_ball_point(center, radius + maximum_radius + padding)):
            left_vertices = set(int(value) for value in faces_cpu[i])
            for j in neighbors:
                if j <= i or left_vertices.intersection(int(value) for value in faces_cpu[j]):
                    continue
                if np.linalg.norm(center[i] - center[j]) <= radius[i] + radius[j] + padding:
                    pairs.append((i, int(j)))
        return torch.tensor(pairs, dtype=torch.int64, device=reference.position.device).reshape(-1, 2)

    def _limit_position_speed(self, tangent: ManifoldTangent) -> ManifoldTangent:
        # A *single* positive scale is essential after the coupled CBF solve.
        # Per-vertex clipping changes the direction of the stacked velocity and
        # can turn a satisfied constraint a.v + gamma*h >= 0 into a violation.
        # For s in (0,1] and h > 0, global rescaling preserves feasibility:
        #   a.(s v) + gamma*h
        #     = s (a.v + gamma*h) + (1-s) gamma*h > 0.
        # It also enforces the vertex-wise path-length assumption used by the
        # fixed collision broad phase because the largest speed is bounded.
        maximum_speed = torch.linalg.vector_norm(tangent.position, dim=-1).amax()
        factor = (
            self.config.maximum_position_speed
            / maximum_speed.clamp_min(1.0e-12)
        ).clamp_max(1.0)
        return ManifoldTangent(
            position=tangent.position * factor,
            rotation_body=tangent.rotation_body,
            covariance=tangent.covariance,
            opacity_logit=tangent.opacity_logit,
            appearance=tangent.appearance,
            latent=tangent.latent,
        )

    def _face_quantities(
        self,
        position: Tensor,
        reference_normal: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        vertices = position[self.faces]
        cross = torch.linalg.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        double_area = torch.linalg.vector_norm(cross, dim=-1)
        normal = cross / double_area[:, None].clamp_min(torch.finfo(position.dtype).eps)
        area_margin = 0.5 * double_area - self.config.minimum_face_area
        if reference_normal is None:
            reference_normal = self.reference_face_normal
        orientation_margin = torch.sum(normal * reference_normal, dim=-1) - self.config.minimum_orientation_cosine
        return area_margin, orientation_margin

    def _separation_margin(self, position: Tensor) -> Tensor:
        if self.nonlocal_pairs.numel() == 0:
            return position.new_empty(0)
        delta = position[self.nonlocal_pairs[:, 0]] - position[self.nonlocal_pairs[:, 1]]
        return delta.square().sum(-1) - self.config.minimum_separation**2

    def _triangle_separation_margin(self, position: Tensor) -> Tensor:
        if self.nonlocal_face_pairs.numel() == 0:
            return position.new_empty(0)
        left = position[self.faces[self.nonlocal_face_pairs[:, 0]]]
        right = position[self.faces[self.nonlocal_face_pairs[:, 1]]]
        return triangle_distance_squared(left, right) - self.config.minimum_separation**2

    def position_constraints(self, position: Tensor, diagnostics: bool = False) -> Tensor:
        area, orientation = self._face_quantities(
            position,
            self.reference_face_normal_diagnostics if diagnostics else None,
        )
        separation = self._separation_margin(position)
        triangle_separation = self._triangle_separation_margin(position)
        return torch.cat((area, orientation, separation, triangle_separation))

    def _restoration_targets_and_scales(self, position: Tensor) -> Tuple[Tensor, Tensor]:
        """Return dimensionally normalized strict-interior targets.

        Topology proposal precedes the continuous topology-preserving flow, but
        transported chart centers can lie just outside the selected stratum's
        collision-free embedding set.  Restoration therefore solves for a
        positive buffer in the native units of each constraint family.  Scaling
        the inequalities changes neither their feasible set nor the primal
        minimum-displacement problem; it only conditions the dual system.
        """

        face_count = int(self.faces.shape[0])
        pair_count = int(self.nonlocal_pairs.shape[0])
        face_pair_count = int(self.nonlocal_face_pairs.shape[0])
        relative = self.config.restoration_relative_margin
        area_scale = position.new_tensor(self.config.minimum_face_area)
        orientation_scale = position.new_tensor(
            1.0 - self.config.minimum_orientation_cosine
        )
        separation_scale = position.new_tensor(self.config.minimum_separation**2)
        separation_target = position.new_tensor(
            ((1.0 + relative) * self.config.minimum_separation) ** 2
            - self.config.minimum_separation**2
        )
        target = torch.cat(
            (
                area_scale.expand(face_count) * relative,
                orientation_scale.expand(face_count) * relative,
                separation_target.expand(pair_count),
                separation_target.expand(face_pair_count),
            )
        )
        scale = torch.cat(
            (
                area_scale.expand(face_count),
                orientation_scale.expand(face_count),
                separation_scale.expand(pair_count),
                separation_scale.expand(face_pair_count),
            )
        )
        return target, scale

    @staticmethod
    def _replace_position(state: ManifoldState, position: Tensor) -> ManifoldState:
        return ManifoldState(
            position=position,
            rotation=state.rotation,
            covariance=state.covariance,
            opacity_logit=state.opacity_logit,
            appearance=state.appearance,
            latent=state.latent,
            evidence_metric=state.evidence_metric,
            complex=state.complex,
        )

    def restore_feasible_embedding(
        self,
        state: ManifoldState,
    ) -> Tuple[ManifoldState, FeasibilityReport]:
        r"""Restore a proposed stratum by metric-minimal hard-constraint steps.

        Each iteration solves the linearized convex problem

        ``min_delta 1/2 sum_v delta_v^T G_v delta_v``

        subject to ``h_j(p) + D h_j(p)[delta] >= target_j`` for the
        currently active area, orientation, vertex-separation, and
        triangle-separation constraints.  A deterministic merit line search
        accepts only decreasing normalized violation, and the returned state
        must pass the existing detached FP64 *strict* feasibility report.

        This is an initialization operation performed after discrete topology
        proposal and before topology-preserving flow.  It is conditionally
        differentiable for fixed active sets/closest features; it is not an
        isotopy claim from an infeasible input.  Total displacement is bounded
        by ``maximum_position_speed``, preserving the fixed broad-phase search
        certificate constructed around the transported input.
        """

        initial_report = self.report(state)
        if initial_report.feasible:
            return state, initial_report
        if (
            initial_report.minimum_covariance_margin <= 0.0
            or initial_report.maximum_covariance_margin <= 0.0
        ):
            raise RuntimeError(
                "embedding restoration cannot repair covariance spectral-box violations"
            )

        original_dtype = state.position.dtype
        base_position = state.position.to(dtype=torch.float64)
        position = base_position
        metric = state.evidence_metric.to(dtype=torch.float64)
        target, scale = self._restoration_targets_and_scales(position)
        last_report = initial_report
        completed_iterations = 0
        last_dual_residual = float("inf")

        for iteration in range(1, self.config.restoration_iterations + 1):
            completed_iterations = iteration
            constraint = self.position_constraints(position, diagnostics=True)
            normalized = (constraint - target) / scale
            active = normalized < 0.0
            if not bool(torch.any(active)):
                candidate_state = self._replace_position(
                    state, position.to(dtype=original_dtype)
                )
                last_report = self.report(candidate_state)
                if last_report.feasible:
                    break

            active = active.detach()
            jacobian = torch.autograd.functional.jacobian(
                lambda value: (
                    (self.position_constraints(value, diagnostics=True) - target)
                    / scale
                )[active],
                position,
                create_graph=torch.is_grad_enabled(),
                vectorize=True,
            )
            linear = normalized[active]
            weighted = torch.linalg.solve(
                metric.unsqueeze(0), jacobian.unsqueeze(-1)
            ).squeeze(-1)
            gram = torch.einsum("iva,jva->ij", jacobian, weighted)
            gram = gram + self.config.dual_regularization * torch.eye(
                gram.shape[0], dtype=gram.dtype, device=gram.device
            )
            if not bool(torch.all(torch.isfinite(gram))) or not bool(
                torch.all(torch.isfinite(linear))
            ):
                raise RuntimeError("embedding-restoration QP contains non-finite coefficients")
            spectral_bound = torch.linalg.matrix_norm(
                gram, ord=float("inf")
            ).clamp_min(self.config.dual_regularization)
            dual_step = 1.0 / spectral_bound
            dual = torch.zeros_like(linear)
            for dual_iteration in range(self.config.dual_iterations):
                candidate_dual = torch.relu(
                    dual - dual_step * (gram @ dual + linear)
                )
                dual = candidate_dual
                should_check = (
                    (dual_iteration + 1) % self.config.dual_check_interval == 0
                    or dual_iteration + 1 == self.config.dual_iterations
                )
                if should_check:
                    fixed_dual = torch.relu(
                        dual - dual_step * (gram @ dual + linear)
                    )
                    residual = torch.max(torch.abs(fixed_dual - dual))
                    threshold = self.config.dual_tolerance * (
                        1.0 + dual.abs().amax()
                    )
                    residual_value, threshold_value = torch.stack(
                        (residual, threshold)
                    ).detach().cpu().tolist()
                    last_dual_residual = float(residual_value)
                    if last_dual_residual <= float(threshold_value):
                        break
            correction = torch.einsum("jva,j->va", weighted, dual)
            if not bool(torch.all(torch.isfinite(correction))):
                raise RuntimeError("embedding-restoration QP produced a non-finite step")

            current_merit = torch.relu(-normalized).amax()
            accepted = False
            step = 1.0
            for _ in range(self.config.maximum_backtracks + 1):
                candidate_position = position + step * correction
                displacement = candidate_position - base_position
                maximum_displacement = torch.linalg.vector_norm(
                    displacement, dim=-1
                ).amax()
                displacement_scale = (
                    position.new_tensor(self.config.maximum_position_speed)
                    / maximum_displacement.clamp_min(torch.finfo(position.dtype).eps)
                ).clamp_max(1.0)
                candidate_position = base_position + displacement_scale * displacement
                candidate_constraint = self.position_constraints(
                    candidate_position, diagnostics=True
                )
                candidate_merit = torch.relu(
                    (target - candidate_constraint) / scale
                ).amax()
                candidate_state = self._replace_position(
                    state, candidate_position.to(dtype=original_dtype)
                )
                candidate_report = self.report(candidate_state)
                if candidate_report.feasible or bool(
                    candidate_merit < current_merit * (1.0 - 1.0e-6)
                ):
                    position = candidate_position
                    last_report = candidate_report
                    accepted = True
                    break
                step *= self.config.backtrack_factor
            if not accepted:
                raise RuntimeError(
                    "embedding-restoration QP could not decrease normalized hard-constraint violation"
                )
            if last_report.feasible:
                break

        restored = self._replace_position(state, position.to(dtype=original_dtype))
        final_report = self.report(restored)
        final_report.restoration_iterations = completed_iterations
        final_report.dual_residual = last_dual_residual
        final_report.restoration_maximum_displacement = float(
            torch.linalg.vector_norm(
                restored.position.detach().to(dtype=torch.float64)
                - state.position.detach().to(dtype=torch.float64),
                dim=-1,
            )
            .amax()
            .cpu()
        )
        if not final_report.feasible:
            raise RuntimeError(
                "embedding restoration exhausted its iterations without reaching the strict feasible set: "
                f"{final_report}"
            )
        return restored, final_report

    def topology_boundary_margin(
        self,
        state: ManifoldState,
        epsilon: float = 1.0e-12,
    ) -> Tensor:
        r"""Compute ``min_j h_j/(||grad_g h_j||_g + epsilon)``.

        This is the geometric distance-to-boundary term used by the
        quantization certificate.  The reference implementation evaluates
        exact piecewise constraint gradients one scalar at a time, avoiding a
        dense ``J x V x 3`` Jacobian allocation.  The metric dual norm is
        ``sqrt(dh G^{-1} dh)``.  Constraint switching (closest triangle
        feature, broad-phase family) remains a discrete conditional boundary.
        """

        if epsilon <= 0:
            raise ValueError("topology-margin epsilon must be positive")
        with torch.enable_grad():
            position = state.position.detach().to(dtype=torch.float64).requires_grad_(True)
            constraint = self.position_constraints(position, diagnostics=True)
            if constraint.numel() == 0:
                return position.new_tensor(torch.inf)
            metric_inverse = torch.linalg.inv(
                state.evidence_metric.detach().to(dtype=torch.float64)
            )
            margins: List[Tensor] = []
            for index in range(constraint.numel()):
                gradient = torch.autograd.grad(
                    constraint[index],
                    position,
                    retain_graph=index + 1 < constraint.numel(),
                    create_graph=False,
                )[0]
                dual_norm = torch.sqrt(
                    torch.einsum(
                        "va,vab,vb->",
                        gradient,
                        metric_inverse,
                        gradient,
                    ).clamp_min(0.0)
                )
                margins.append(
                    constraint[index].detach() / (dual_norm.detach() + epsilon)
                )
            return torch.stack(margins).amin()

    def report(self, state: ManifoldState, accepted_step: float = 0.0) -> FeasibilityReport:
        # Acceptance/certification is detached and deliberately recomputed in
        # FP64. The differentiable CBF projection remains in the state dtype.
        position = state.position.detach().to(dtype=torch.float64)
        covariance = state.covariance.detach().to(dtype=torch.float64)
        area, orientation = self._face_quantities(
            position,
            self.reference_face_normal_diagnostics,
        )
        separation = self._separation_margin(position)
        triangle_separation = self._triangle_separation_margin(position)
        eigenvalues = torch.linalg.eigvalsh(
            0.5 * (covariance + covariance.transpose(-1, -2))
        )
        covariance_min = eigenvalues[:, 0] - self.config.minimum_covariance_eigenvalue
        covariance_max = self.config.maximum_covariance_eigenvalue - eigenvalues[:, -1]
        infinity = position.new_tensor(torch.inf)
        area_minimum = area.amin() if area.numel() else infinity
        orientation_minimum = orientation.amin() if orientation.numel() else infinity
        all_separation = torch.cat((separation, triangle_separation))
        separation_minimum = all_separation.amin() if all_separation.numel() else infinity
        diagnostic_minima = torch.stack(
            (
                area_minimum,
                orientation_minimum,
                separation_minimum,
                covariance_min.amin(),
                covariance_max.amin(),
            )
        ).cpu().tolist()
        (
            area_min,
            orientation_min,
            separation_min,
            covariance_minimum,
            covariance_maximum,
        ) = (float(value) for value in diagnostic_minima)
        feasible = min(area_min, orientation_min, separation_min, covariance_minimum, covariance_maximum) > 0.0
        return FeasibilityReport(
            feasible=feasible,
            minimum_area_margin=area_min,
            minimum_orientation_margin=orientation_min,
            minimum_separation_margin=separation_min,
            minimum_covariance_margin=covariance_minimum,
            maximum_covariance_margin=covariance_maximum,
            accepted_step=accepted_step,
        )

    def project(self, state: ManifoldState, tangent: ManifoldTangent) -> Tuple[ManifoldTangent, FeasibilityReport]:
        """Solve ``min ||v-v_raw||_G`` subject to linearized CBF inequalities."""

        tangent = self._limit_position_speed(tangent)
        constraints = self.position_constraints(state.position)
        if constraints.numel() == 0:
            return tangent, self.report(state)
        _, directional = torch.autograd.functional.jvp(
            self.position_constraints,
            state.position,
            tangent.position,
            create_graph=torch.is_grad_enabled(),
        )
        rhs_margin = directional + self.config.decay_rate * constraints
        active = (constraints < self.config.activation_margin) | (rhs_margin < 0)
        if not torch.any(active):
            return tangent, self.report(state)
        a = torch.autograd.functional.jacobian(
            lambda value: self.position_constraints(value)[active],
            state.position,
            create_graph=torch.is_grad_enabled(),
            vectorize=True,
        )
        h = constraints[active]
        metric_inverse = torch.linalg.inv(state.evidence_metric)
        weighted = torch.einsum("vab,jvb->jva", metric_inverse, a)
        gram = torch.einsum("iva,jva->ij", a, weighted)
        gram = gram + self.config.dual_regularization * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        linear = torch.einsum("jva,va->j", a, tangent.position) + self.config.decay_rate * h
        spectral_bound = torch.linalg.matrix_norm(gram, ord=float("inf")).clamp_min(self.config.dual_regularization)
        step = 1.0 / spectral_bound
        dual = torch.zeros_like(linear)
        residual_value = float("inf")
        for iteration in range(self.config.dual_iterations):
            candidate = torch.relu(dual - step * (gram @ dual + linear))
            dual = candidate
            should_check = (
                (iteration + 1) % self.config.dual_check_interval == 0
                or iteration + 1 == self.config.dual_iterations
            )
            if should_check:
                fixed_dual = torch.relu(dual - step * (gram @ dual + linear))
                residual = torch.max(torch.abs(fixed_dual - dual))
                threshold = self.config.dual_tolerance * (1.0 + dual.abs().amax())
                residual_value, threshold_value = (
                    float(value)
                    for value in torch.stack((residual, threshold))
                    .detach()
                    .cpu()
                    .tolist()
                )
                if residual_value <= threshold_value:
                    break
        correction = torch.einsum("jva,j->va", weighted, dual)
        safe_position = tangent.position + correction
        projected = ManifoldTangent(
            position=safe_position,
            rotation_body=tangent.rotation_body,
            covariance=tangent.covariance,
            opacity_logit=tangent.opacity_logit,
            appearance=tangent.appearance,
            latent=tangent.latent,
        )
        # Global positive rescaling preserves every a.v + gamma*h >= 0
        # inequality because all accepted states have h > 0.
        projected = self._limit_position_speed(projected)
        linearized_margin = (
            torch.einsum("jva,va->j", a, projected.position)
            + self.config.decay_rate * h
        )
        minimum_linearized_margin = float(
            linearized_margin.min().detach().cpu()
        )
        if minimum_linearized_margin < -10.0 * self.config.dual_tolerance:
            raise RuntimeError(
                "control-barrier QP did not satisfy its active linearized "
                f"constraints: minimum margin={minimum_linearized_margin:.3e}"
            )
        report = self.report(state)
        report.projected_constraints = int(active.sum().item())
        report.dual_residual = residual_value
        report.minimum_linearized_margin = minimum_linearized_margin
        return projected, report

    def retract_with_backtracking(
        self,
        state: ManifoldState,
        tangent: ManifoldTangent,
        requested_step: float,
    ) -> Tuple[ManifoldState, FeasibilityReport]:
        """Accept only a nonlinear manifold step with strictly positive margins."""

        step = requested_step
        for _ in range(self.config.maximum_backtracks + 1):
            candidate = retract(state, tangent, step)
            report = self.report(candidate, accepted_step=step)
            if report.feasible:
                return candidate, report
            step *= self.config.backtrack_factor
        raise RuntimeError(
            "barrier-certified integration failed: no positive-margin step was found; "
            "the initial state, collision candidate set, or vector-field scale violates the theorem assumptions"
        )


__all__ = ["BarrierConfig", "BarrierProjector", "FeasibilityReport", "triangle_distance_squared"]

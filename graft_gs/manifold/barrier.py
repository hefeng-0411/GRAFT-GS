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
    maximum_backtracks: int = 14
    backtrack_factor: float = 0.5
    maximum_position_speed: float = 5.0e-2

    def __post_init__(self) -> None:
        if min(
            self.minimum_face_area,
            self.minimum_separation,
            self.minimum_covariance_eigenvalue,
            self.activation_margin,
            self.decay_rate,
            self.dual_regularization,
            self.maximum_position_speed,
        ) <= 0:
            raise ValueError("barrier scales, margins, regularization, and speed bound must be positive")
        if not -1.0 < self.minimum_orientation_cosine < 1.0:
            raise ValueError("minimum_orientation_cosine must lie in (-1,1)")
        if self.maximum_covariance_eigenvalue <= self.minimum_covariance_eigenvalue:
            raise ValueError("maximum covariance eigenvalue must exceed the minimum")
        if not 0.0 < self.backtrack_factor < 1.0:
            raise ValueError("backtrack_factor must lie in (0,1)")


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

    def _face_quantities(self, position: Tensor) -> Tuple[Tensor, Tensor]:
        vertices = position[self.faces]
        cross = torch.linalg.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        double_area = torch.linalg.vector_norm(cross, dim=-1)
        normal = cross / double_area[:, None].clamp_min(torch.finfo(position.dtype).eps)
        area_margin = 0.5 * double_area - self.config.minimum_face_area
        orientation_margin = torch.sum(normal * self.reference_face_normal, dim=-1) - self.config.minimum_orientation_cosine
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

    def position_constraints(self, position: Tensor) -> Tensor:
        area, orientation = self._face_quantities(position)
        separation = self._separation_margin(position)
        triangle_separation = self._triangle_separation_margin(position)
        return torch.cat((area, orientation, separation, triangle_separation))

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
            position = state.position.detach().clone().requires_grad_(True)
            constraint = self.position_constraints(position)
            if constraint.numel() == 0:
                return state.position.new_tensor(torch.inf)
            metric_inverse = torch.linalg.inv(state.evidence_metric.detach())
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
        area, orientation = self._face_quantities(state.position)
        separation = self._separation_margin(state.position)
        triangle_separation = self._triangle_separation_margin(state.position)
        eigenvalues = torch.linalg.eigvalsh(0.5 * (state.covariance + state.covariance.transpose(-1, -2)))
        covariance_min = eigenvalues[:, 0] - self.config.minimum_covariance_eigenvalue
        covariance_max = self.config.maximum_covariance_eigenvalue - eigenvalues[:, -1]
        area_min = float(area.min().detach().cpu()) if area.numel() else float("inf")
        orientation_min = float(orientation.min().detach().cpu()) if orientation.numel() else float("inf")
        all_separation = torch.cat((separation, triangle_separation))
        separation_min = float(all_separation.min().detach().cpu()) if all_separation.numel() else float("inf")
        covariance_minimum = float(covariance_min.min().detach().cpu())
        covariance_maximum = float(covariance_max.min().detach().cpu())
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
        for _ in range(self.config.dual_iterations):
            candidate = torch.relu(dual - step * (gram @ dual + linear))
            residual = torch.max(torch.abs(candidate - dual))
            dual = candidate
            residual_value = float(residual.detach().cpu())
            if residual_value < self.config.dual_tolerance:
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

"""Numerically stable operations on the GRAFT-GS product manifold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from ..topology.strata import SimplicialComplex


def hat(vector: Tensor) -> Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )


def vee(matrix: Tensor) -> Tensor:
    return torch.stack((matrix[..., 2, 1] - matrix[..., 1, 2], matrix[..., 0, 2] - matrix[..., 2, 0], matrix[..., 1, 0] - matrix[..., 0, 1]), dim=-1) * 0.5


def so3_exp(omega: Tensor, epsilon: float = 1.0e-8) -> Tensor:
    theta = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
    generator = hat(omega)
    theta2 = theta.square()
    a = torch.where(theta > epsilon, torch.sin(theta) / theta, 1.0 - theta2 / 6.0 + theta2.square() / 120.0)
    b = torch.where(theta > epsilon, (1.0 - torch.cos(theta)) / theta2, 0.5 - theta2 / 24.0 + theta2.square() / 720.0)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    return eye + a[..., None] * generator + b[..., None] * (generator @ generator)


def so3_log(rotation: Tensor, epsilon: float = 1.0e-7) -> Tensor:
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    sine = torch.sin(theta)
    scale = torch.where(theta.abs() < epsilon, 0.5 + theta.square() / 12.0, theta / (2.0 * sine.clamp_min(epsilon)))
    regular = scale[..., None] * torch.stack(
        (rotation[..., 2, 1] - rotation[..., 1, 2], rotation[..., 0, 2] - rotation[..., 2, 0], rotation[..., 1, 0] - rotation[..., 0, 1]), dim=-1
    )
    # Near pi, the skew part vanishes. Recover an axis from the symmetric part
    # and align its sign with the largest available skew component.
    near_pi = cosine < -1.0 + 1.0e-5
    symmetric_axis_matrix = 0.5 * (rotation + torch.eye(3, dtype=rotation.dtype, device=rotation.device))
    _, axis_vectors = torch.linalg.eigh(0.5 * (symmetric_axis_matrix + symmetric_axis_matrix.transpose(-1, -2)))
    axis = axis_vectors[..., :, -1]
    sign_reference = torch.stack(
        (rotation[..., 2, 1] - rotation[..., 1, 2], rotation[..., 0, 2] - rotation[..., 2, 0], rotation[..., 1, 0] - rotation[..., 0, 1]), dim=-1
    )
    largest = sign_reference.abs().argmax(-1)
    selected_sign = torch.gather(sign_reference, -1, largest[..., None]).sign()
    axis = axis * torch.where(selected_sign == 0, torch.ones_like(selected_sign), selected_sign)
    axis = torch.nn.functional.normalize(axis, dim=-1, eps=epsilon)
    return torch.where(near_pi[..., None], theta[..., None] * axis, regular)


def _symmetric_eigh(matrix: Tensor, floor: float = 1.0e-8) -> Tuple[Tensor, Tensor]:
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    values, vectors = torch.linalg.eigh(matrix)
    return values.clamp_min(floor), vectors


def spd_power(matrix: Tensor, power: float | Tensor, floor: float = 1.0e-8) -> Tensor:
    values, vectors = _symmetric_eigh(matrix, floor)
    powered = values.pow(power)
    return (vectors * powered[..., None, :]) @ vectors.transpose(-1, -2)


def spd_log(matrix: Tensor, floor: float = 1.0e-8) -> Tensor:
    values, vectors = _symmetric_eigh(matrix, floor)
    return (vectors * torch.log(values)[..., None, :]) @ vectors.transpose(-1, -2)


def symmetric_exp(matrix: Tensor) -> Tensor:
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    values, vectors = torch.linalg.eigh(matrix)
    return (vectors * torch.exp(values)[..., None, :]) @ vectors.transpose(-1, -2)


def spd_geodesic(start: Tensor, end: Tensor, time: Tensor | float) -> Tensor:
    start_sqrt = spd_power(start, 0.5)
    start_inverse_sqrt = spd_power(start, -0.5)
    relative = start_inverse_sqrt @ end @ start_inverse_sqrt
    return start_sqrt @ spd_power(relative, time) @ start_sqrt


def spd_geodesic_velocity(start: Tensor, end: Tensor, time: Tensor | float) -> Tensor:
    start_sqrt = spd_power(start, 0.5)
    start_inverse_sqrt = spd_power(start, -0.5)
    relative = start_inverse_sqrt @ end @ start_inverse_sqrt
    logarithm = spd_log(relative)
    relative_t = spd_power(relative, time)
    return start_sqrt @ (relative_t @ logarithm) @ start_sqrt


def spd_retract(base: Tensor, tangent: Tensor, step: float | Tensor) -> Tensor:
    inverse_sqrt = spd_power(base, -0.5)
    sqrt_base = spd_power(base, 0.5)
    local = inverse_sqrt @ tangent @ inverse_sqrt
    return sqrt_base @ symmetric_exp(step * local) @ sqrt_base


def spd_parallel_transport(start: Tensor, end: Tensor, tangent: Tensor) -> Tensor:
    r"""Affine-invariant parallel transport along the SPD geodesic.

    For ``S=P^{-1/2} Q P^{-1/2}``, the congruence
    ``E=P^{1/2} S^{1/2} P^{-1/2}`` satisfies ``E P E^T=Q``. Therefore
    ``U -> E U E^T`` is an isometry between the two affine-invariant tangent
    metrics. The result is explicitly symmetrized against roundoff.
    """

    if start.shape != end.shape or start.shape != tangent.shape:
        raise ValueError("SPD parallel transport tensors must share [...,3,3]")
    start_root = spd_power(start, 0.5)
    start_inverse_root = spd_power(start, -0.5)
    relative = start_inverse_root @ end @ start_inverse_root
    congruence = start_root @ spd_power(relative, 0.5) @ start_inverse_root
    transported = congruence @ tangent @ congruence.transpose(-1, -2)
    return 0.5 * (transported + transported.transpose(-1, -2))


@dataclass
class ManifoldState:
    position: Tensor
    rotation: Tensor
    covariance: Tensor
    opacity_logit: Tensor
    appearance: Tensor
    latent: Tensor
    evidence_metric: Tensor
    complex: SimplicialComplex

    def to(self, *args: object, **kwargs: object) -> "ManifoldState":
        values = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, Tensor):
                values[name] = value.to(*args, **kwargs)
            elif isinstance(value, SimplicialComplex):
                values[name] = SimplicialComplex(
                    value.atlas_node_index.to(*args, **kwargs),
                    value.edges.to(*args, **kwargs),
                    value.faces.to(*args, **kwargs),
                )
            else:
                values[name] = value
        return ManifoldState(**values)

    def validate(self, tolerance: float = 1.0e-5) -> None:
        v = self.position.shape[0]
        contracts = {
            "position": (v, 3),
            "rotation": (v, 3, 3),
            "covariance": (v, 3, 3),
            "opacity_logit": (v, 1),
            "evidence_metric": (v, 3, 3),
        }
        for name, shape in contracts.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.appearance.shape[0] != v or self.latent.shape != (v, 128):
            raise ValueError("appearance/latent leading dimensions must match the complex")
        eye = torch.eye(3, dtype=self.rotation.dtype, device=self.rotation.device)
        orthogonality = torch.linalg.matrix_norm(self.rotation.transpose(-1, -2) @ self.rotation - eye, dim=(-2, -1)).max()
        if float(orthogonality) > tolerance or float(torch.linalg.det(self.rotation).min()) < 1.0 - tolerance:
            raise ValueError("rotation state is not in SO(3)")
        if float(torch.linalg.eigvalsh(self.covariance).min()) <= 0 or float(torch.linalg.eigvalsh(self.evidence_metric).min()) <= 0:
            raise ValueError("covariance and evidence metric must be SPD")
        if self.complex.num_vertices != v:
            raise ValueError("state and topology complex vertex counts disagree")
        if not self.complex.manifold_incidence_valid():
            raise ValueError("manifold state requires edge incidence in {1,2}")
        if not self.complex.orientation_consistent():
            raise ValueError("manifold state requires a consistently oriented complex")


@dataclass
class ManifoldTangent:
    position: Tensor
    rotation_body: Tensor
    covariance: Tensor
    opacity_logit: Tensor
    appearance: Tensor
    latent: Tensor

    def scaled(self, value: float | Tensor) -> "ManifoldTangent":
        return ManifoldTangent(*(getattr(self, name) * value for name in self.__dataclass_fields__))

    def add(self, other: "ManifoldTangent") -> "ManifoldTangent":
        return ManifoldTangent(*(getattr(self, name) + getattr(other, name) for name in self.__dataclass_fields__))


def retract(state: ManifoldState, tangent: ManifoldTangent, step: float | Tensor) -> ManifoldState:
    rotation = state.rotation @ so3_exp(step * tangent.rotation_body)
    covariance = spd_retract(state.covariance, tangent.covariance, step)
    return ManifoldState(
        position=state.position + step * tangent.position,
        rotation=rotation,
        covariance=covariance,
        opacity_logit=state.opacity_logit + step * tangent.opacity_logit,
        appearance=state.appearance + step * tangent.appearance,
        latent=state.latent + step * tangent.latent,
        evidence_metric=state.evidence_metric,
        complex=state.complex,
    )


def geodesic_interpolate(start: ManifoldState, end: ManifoldState, time: Tensor | float) -> Tuple[ManifoldState, ManifoldTangent]:
    if (
        start.complex.num_vertices != end.complex.num_vertices
        or not torch.equal(
            start.complex.atlas_node_index,
            end.complex.atlas_node_index,
        )
        or not torch.equal(start.complex.edges, end.complex.edges)
        or not torch.equal(start.complex.faces, end.complex.faces)
    ):
        raise ValueError("geodesic interpolation requires one fixed topology stratum")
    relative_rotation = start.rotation.transpose(-1, -2) @ end.rotation
    rotation_log = so3_log(relative_rotation)
    rotation = start.rotation @ so3_exp(time * rotation_log)
    covariance = spd_geodesic(start.covariance, end.covariance, time)
    state = ManifoldState(
        position=(1.0 - time) * start.position + time * end.position,
        rotation=rotation,
        covariance=covariance,
        opacity_logit=(1.0 - time) * start.opacity_logit + time * end.opacity_logit,
        appearance=(1.0 - time) * start.appearance + time * end.appearance,
        latent=(1.0 - time) * start.latent + time * end.latent,
        evidence_metric=(1.0 - time) * start.evidence_metric + time * end.evidence_metric,
        complex=start.complex,
    )
    tangent = ManifoldTangent(
        position=end.position - start.position,
        rotation_body=rotation_log,
        covariance=spd_geodesic_velocity(start.covariance, end.covariance, time),
        opacity_logit=end.opacity_logit - start.opacity_logit,
        appearance=end.appearance - start.appearance,
        latent=end.latent - start.latent,
    )
    return state, tangent


def product_metric_squared(
    state: ManifoldState,
    tangent: ManifoldTangent,
    position_weight: float = 1.0,
    rotation_weight: float = 1.0,
    covariance_weight: float = 1.0,
    opacity_weight: float = 1.0,
    appearance_weight: float = 1.0,
    latent_weight: float = 0.1,
) -> Tensor:
    position = torch.einsum("vi,vij,vj->v", tangent.position, state.evidence_metric, tangent.position).sum()
    rotation = tangent.rotation_body.square().sum()
    inverse = torch.linalg.inv(state.covariance)
    covariance = torch.einsum("vij,vjk,vkl,vli->v", inverse, tangent.covariance, inverse, tangent.covariance).sum()
    return (
        position_weight * position
        + rotation_weight * rotation
        + covariance_weight * covariance
        + opacity_weight * tangent.opacity_logit.square().sum()
        + appearance_weight * tangent.appearance.square().sum()
        + latent_weight * tangent.latent.square().sum()
    )


__all__ = [
    "ManifoldState",
    "ManifoldTangent",
    "geodesic_interpolate",
    "hat",
    "product_metric_squared",
    "retract",
    "so3_exp",
    "so3_log",
    "spd_geodesic",
    "spd_log",
    "spd_parallel_transport",
    "spd_retract",
]

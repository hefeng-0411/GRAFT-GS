"""Differential-geometric Gaussian readout and unified PLY/GLB serialization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import ceil, pi, sqrt
from pathlib import Path
import struct
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..geometry.atlas import PersistentOctreeAtlas
from ..manifold.geometry import ManifoldState
from ..mapping.manifold_mapping import MappingResult


@dataclass(frozen=True)
class AnalyticalReadoutConfig:
    target_surface_area_per_gaussian: float = 2.5e-4
    minimum_samples_per_chart: int = 1
    maximum_samples_per_chart: int = 64
    surface_area_quadrature_samples: int = 64
    tangent_scale_factor: float = 0.85
    normal_scale_factor: float = 0.25
    minimum_scale: float = 1.0e-5
    maximum_normal_scale: float = 5.0e-2
    color_ridge: float = 1.0e-3
    color_prior_weight: float = 1.0e-2
    observation_bandwidth_factor: float = 0.75
    uncertainty_normal_weight: float = 0.25
    opacity_epsilon: float = 1.0e-8
    metric_epsilon: float = 1.0e-10
    metric_relative_eigengap: float = 1.0e-4
    opacity_tile_size: int = 16
    maximum_tile_opacity: float = 0.995

    def __post_init__(self) -> None:
        if self.target_surface_area_per_gaussian <= 0:
            raise ValueError("target surface area per Gaussian must be positive")
        if not 1 <= self.minimum_samples_per_chart <= self.maximum_samples_per_chart:
            raise ValueError("Gaussian sample-count bounds are inconsistent")
        if self.surface_area_quadrature_samples < 4:
            raise ValueError("surface-area quadrature requires at least four samples")
        if self.opacity_tile_size < 1 or not 0.0 < self.maximum_tile_opacity < 1.0:
            raise ValueError("opacity tile policy is outside its numerical domain")
        if min(
            self.tangent_scale_factor,
            self.normal_scale_factor,
            self.minimum_scale,
            self.maximum_normal_scale,
            self.color_ridge,
            self.opacity_epsilon,
            self.metric_epsilon,
        ) <= 0:
            raise ValueError("analytical readout scales and regularizers must be positive")
        if not 0.0 < self.metric_relative_eigengap < 1.0:
            raise ValueError("metric_relative_eigengap must lie in (0,1)")


@dataclass
class GaussianAsset:
    means: Tensor  # [G,3]
    covariance: Tensor  # [G,3,3]
    rotation: Tensor  # [G,3,3]
    scales: Tensor  # [G,3]
    sh_coefficients: Tensor  # [G,16,3]
    opacity: Tensor  # [G,1]
    chart_index: Tensor  # [G], local selected-complex chart
    chart_coordinates: Tensor  # [G,2]
    represented_area: Tensor  # [G]

    def validate(self, surface_tolerance: float = 1.0e-6) -> None:
        g = self.means.shape[0]
        if self.covariance.shape != (g, 3, 3) or self.rotation.shape != (g, 3, 3):
            raise ValueError("Gaussian covariance/rotation contracts are invalid")
        if self.scales.shape != (g, 3) or self.sh_coefficients.shape != (g, 16, 3):
            raise ValueError("Gaussian scale/SH contracts are invalid")
        if torch.any(torch.linalg.eigvalsh(self.covariance) <= 0):
            raise ValueError("Gaussian covariance must be SPD")
        if torch.any(self.opacity <= 0) or torch.any(self.opacity >= 1):
            raise ValueError("opacity must lie strictly inside (0,1)")
        eye = torch.eye(3, dtype=self.rotation.dtype, device=self.rotation.device)
        if float(torch.linalg.matrix_norm(self.rotation.transpose(-1, -2) @ self.rotation - eye, dim=(-2, -1)).max()) > 1.0e-5:
            raise ValueError("Gaussian rotations must be orthogonal")
        reconstructed = self.rotation @ torch.diag_embed(self.scales.square()) @ self.rotation.transpose(-1, -2)
        torch.testing.assert_close(reconstructed, self.covariance, atol=surface_tolerance, rtol=surface_tolerance)


@dataclass
class MeshAsset:
    vertices: Tensor
    faces: Tensor
    normals: Tensor
    colors: Tensor


def real_sh_basis_degree3(direction: Tensor) -> Tensor:
    """Orthonormal real spherical harmonics ordered by increasing l,m."""

    x, y, z = direction.unbind(-1)
    return torch.stack(
        (
            torch.full_like(x, 0.28209479177387814),
            -0.4886025119029199 * y,
            0.4886025119029199 * z,
            -0.4886025119029199 * x,
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (3.0 * z.square() - 1.0),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (x.square() - y.square()),
            -0.5900435899266435 * y * (3.0 * x.square() - y.square()),
            2.890611442640554 * x * y * z,
            -0.4570457994644658 * y * (5.0 * z.square() - 1.0),
            0.3731763325901154 * z * (5.0 * z.square() - 3.0),
            -0.4570457994644658 * x * (5.0 * z.square() - 1.0),
            1.445305721320277 * z * (x.square() - y.square()),
            -0.5900435899266435 * x * (x.square() - 3.0 * y.square()),
        ),
        dim=-1,
    )


def _disk_samples(count: int, radius: Tensor) -> Tensor:
    """Deterministic equal-area golden-angle disk samples."""

    index = torch.arange(count, dtype=radius.dtype, device=radius.device)
    radial = torch.sqrt((index + 0.5) / count) * radius
    angle = index * (pi * (3.0 - sqrt(5.0)))
    return torch.stack((radial * torch.cos(angle), radial * torch.sin(angle)), dim=-1)


def _chart_evaluate(center: Tensor, rotation: Tensor, curvature: Tensor, xi: Tensor) -> Tuple[Tensor, Tensor]:
    height = 0.5 * torch.einsum("ni,ij,nj->n", xi, curvature, xi)
    local = torch.cat((xi, height[:, None]), dim=-1)
    means = center + local @ rotation.transpose(0, 1)
    slope = xi @ curvature.transpose(0, 1)
    local_jacobian = torch.stack(
        (
            torch.stack((torch.ones_like(slope[:, 0]), torch.zeros_like(slope[:, 0]), slope[:, 0]), dim=-1),
            torch.stack((torch.zeros_like(slope[:, 1]), torch.ones_like(slope[:, 1]), slope[:, 1]), dim=-1),
        ),
        dim=-1,
    )
    jacobian = rotation @ local_jacobian
    return means, jacobian


def _stratified_metric_eigh(
    metric: Tensor,
    epsilon: float,
    relative_eigengap: float,
) -> tuple[Tensor, Tensor]:
    """Eigendecompose 2D metrics with a finite isotropic-stratum derivative.

    Principal tangent directions are gauge-valued when both metric eigenvalues
    coincide.  Separated spectra use the exact differentiable eigensystem.  At
    an unresolved gap, the exact forward eigenvalues/vectors are retained, the
    vector gauge has zero derivative, and both eigenvalues receive the common
    trace derivative.  This preserves isotropic scale learning without the
    undefined ``1 / (lambda_1-lambda_0)`` eigenvector term.
    """

    if metric.shape[-2:] != (2, 2):
        raise ValueError("chart metric must have shape [...,2,2]")
    if epsilon <= 0.0 or not 0.0 < relative_eigengap < 1.0:
        raise ValueError("invalid chart-metric eigengap policy")
    leading_shape = metric.shape[:-2]
    symmetric = 0.5 * (metric + metric.transpose(-1, -2))
    flat = symmetric.reshape(-1, 2, 2)
    with torch.no_grad():
        diagnostic_value, diagnostic_vector = torch.linalg.eigh(flat)
        scale = diagnostic_value.abs().amax(dim=-1).clamp_min(epsilon)
        stable = (
            diagnostic_value[:, 1] - diagnostic_value[:, 0]
            > epsilon + relative_eigengap * scale
        )
    mean = 0.5 * flat.diagonal(dim1=-2, dim2=-1).sum(-1)
    value = diagnostic_value + (mean - mean.detach())[:, None]
    vector = diagnostic_vector + 0.0 * flat.sum(dim=(-2, -1))[:, None, None]
    stable_index = torch.nonzero(stable, as_tuple=False).flatten()
    if stable_index.numel():
        stable_value, stable_vector = torch.linalg.eigh(flat[stable_index])
        value = value.index_copy(0, stable_index, stable_value)
        vector = vector.index_copy(0, stable_index, stable_vector)
    return (
        value.reshape(*leading_shape, 2),
        vector.reshape(*leading_shape, 2, 2),
    )


class AnalyticalSurfaceReadout(nn.Module):
    """Construct every Gaussian and mesh attribute from one selected atlas."""

    def __init__(self, config: AnalyticalReadoutConfig = AnalyticalReadoutConfig()) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        atlas: PersistentOctreeAtlas,
        state: ManifoldState,
        mapping: MappingResult,
    ) -> Tuple[GaussianAsset, MeshAsset]:
        node_to_mapping = {int(node): i for i, node in enumerate(mapping.graph.atlas_node_index.tolist())}
        gaussian_means = []
        gaussian_covariance = []
        gaussian_rotation = []
        gaussian_scales = []
        gaussian_sh = []
        gaussian_opacity = []
        gaussian_chart = []
        gaussian_xi = []
        gaussian_area = []
        cfg = self.config
        for local_chart, global_node in enumerate(state.complex.atlas_node_index.tolist()):
            mapping_index = node_to_mapping[global_node]
            radius = atlas.chart_radii[global_node]
            curvature = atlas.curvature[global_node]
            domain_area = pi * radius.square()
            quadrature_xi = _disk_samples(
                cfg.surface_area_quadrature_samples,
                radius,
            )
            _, quadrature_jacobian = _chart_evaluate(
                state.position[local_chart],
                state.rotation[local_chart],
                curvature,
                quadrature_xi,
            )
            quadrature_first_form = (
                quadrature_jacobian.transpose(-1, -2) @ quadrature_jacobian
            )
            surface_area = domain_area * torch.sqrt(
                torch.linalg.det(quadrature_first_form).clamp_min(
                    cfg.opacity_epsilon
                )
            ).mean()
            count = int(torch.ceil(surface_area / cfg.target_surface_area_per_gaussian).clamp(cfg.minimum_samples_per_chart, cfg.maximum_samples_per_chart).item())
            xi = _disk_samples(count, radius)
            means, jacobian = _chart_evaluate(state.position[local_chart], state.rotation[local_chart], curvature, xi)
            first_form = jacobian.transpose(-1, -2) @ jacobian
            eigenvalue, eigenvector = _stratified_metric_eigh(
                first_form,
                cfg.metric_epsilon,
                cfg.metric_relative_eigengap,
            )
            eigenvalue = eigenvalue.clamp_min(cfg.metric_epsilon)
            tangent = jacobian @ eigenvector @ torch.diag_embed(eigenvalue.rsqrt())
            tangent_1, tangent_2 = tangent[..., 0], tangent[..., 1]
            normal = F.normalize(torch.linalg.cross(tangent_1, tangent_2, dim=-1), dim=-1)
            rotation = torch.stack((tangent_1, tangent_2, normal), dim=-1)
            delta_q = torch.sqrt(domain_area / count)
            scale_1 = cfg.tangent_scale_factor * delta_q * torch.sqrt(eigenvalue[:, 0])
            scale_2 = cfg.tangent_scale_factor * delta_q * torch.sqrt(eigenvalue[:, 1])
            # Frobenius curvature is a smooth conservative upper bound on the
            # spectral curvature. The spectral norm has a set-valued gradient
            # at a flat (zero) chart, exactly where initialization begins.
            kappa = torch.sqrt(
                curvature.square().sum() + cfg.metric_epsilon**2
            )
            normal_scale = cfg.normal_scale_factor * torch.minimum(scale_1, scale_2) / (
                1.0 + kappa * torch.minimum(scale_1, scale_2)
            )
            continuous_metric = atlas.partition_of_unity_metric(
                means,
                mapping.riemannian_metric,
                node_index=mapping.graph.atlas_node_index,
            )
            evidence_uncertainty = torch.linalg.inv(continuous_metric)
            uncertainty = 0.5 * (
                state.covariance[local_chart][None] + evidence_uncertainty
            )
            normal_variance = torch.einsum(
                "ni,nij,nj->n", normal, uncertainty, normal
            )
            total_variance = uncertainty.diagonal(
                dim1=-2, dim2=-1
            ).sum(-1).clamp_min(cfg.opacity_epsilon)
            relative_normal_uncertainty = (normal_variance / total_variance).clamp(0.0, 1.0)
            uncertainty_thickness = 0.25 * delta_q * torch.sqrt(relative_normal_uncertainty)
            normal_scale = torch.sqrt(
                normal_scale.square() + cfg.uncertainty_normal_weight * uncertainty_thickness.square()
            )
            normal_scale = normal_scale.clamp(cfg.minimum_scale, cfg.maximum_normal_scale)
            scales = torch.stack((scale_1.clamp_min(cfg.minimum_scale), scale_2.clamp_min(cfg.minimum_scale), normal_scale), dim=-1)
            # The tangent covariance is basis-free: the principal-axis formula
            # T diag((a sqrt(lambda))^2) T^T simplifies exactly to a^2 J J^T.
            # Constructing it this way keeps the renderer gradient smooth at an
            # isotropic tangent metric while rotation/scales retain the same
            # exact forward covariance for PLY serialization.
            tangent_factor = cfg.tangent_scale_factor * delta_q
            tangent_covariance = tangent_factor.square() * (
                jacobian @ jacobian.transpose(-1, -2)
            )
            covariance = tangent_covariance + normal_scale.square()[:, None, None] * (
                normal[:, :, None] * normal[:, None, :]
            )
            covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
            area = torch.sqrt(torch.linalg.det(first_form)).clamp_min(cfg.opacity_epsilon) * domain_area / count
            base_alpha = torch.sigmoid(state.opacity_logit[local_chart, 0])
            base_optical_depth = -torch.log1p(-base_alpha.clamp_max(1.0 - cfg.opacity_epsilon))
            # State opacity is optical depth per chart-domain area.  Dividing by
            # the chart Jacobian preserves transported mass on the surface.
            optical_depth = base_optical_depth * (domain_area / count) / area
            opacity = -torch.expm1(-optical_depth.clamp_min(cfg.opacity_epsilon))
            opacity = opacity.clamp(cfg.opacity_epsilon, 1.0 - cfg.opacity_epsilon)
            sh = self._solve_appearance(local_chart, mapping_index, means, state, mapping, atlas)
            gaussian_means.append(means)
            gaussian_covariance.append(covariance)
            gaussian_rotation.append(rotation)
            gaussian_scales.append(scales)
            gaussian_sh.append(sh)
            gaussian_opacity.append(opacity[:, None])
            gaussian_chart.append(torch.full((count,), local_chart, dtype=torch.int64, device=means.device))
            gaussian_xi.append(xi)
            gaussian_area.append(area)
        gaussians = GaussianAsset(
            means=torch.cat(gaussian_means),
            covariance=torch.cat(gaussian_covariance),
            rotation=torch.cat(gaussian_rotation),
            scales=torch.cat(gaussian_scales),
            sh_coefficients=torch.cat(gaussian_sh),
            opacity=torch.cat(gaussian_opacity),
            chart_index=torch.cat(gaussian_chart),
            chart_coordinates=torch.cat(gaussian_xi),
            represented_area=torch.cat(gaussian_area),
        )
        gaussians.validate()
        mesh = self._mesh_from_state(state)
        return gaussians, mesh

    def _solve_appearance(
        self,
        local_chart: int,
        mapping_index: int,
        means: Tensor,
        state: ManifoldState,
        mapping: MappingResult,
        atlas: PersistentOctreeAtlas,
    ) -> Tensor:
        cfg = self.config
        edge_mask = mapping.graph.source == mapping_index
        target = mapping.graph.target[edge_mask]
        plan = mapping.plan[edge_mask]
        count = means.shape[0]
        prior_color = torch.sigmoid(state.appearance[local_chart, :3])
        if target.numel() == 0 or mapping.evidence.colors is None:
            coefficients = means.new_zeros((count, 16, 3))
            coefficients[:, 0] = (prior_color - 0.5) / 0.28209479177387814
            return coefficients
        evidence_position = mapping.evidence.positions[target]
        view_direction = -mapping.evidence.rays[target]
        basis = real_sh_basis_degree3(view_direction)
        color = mapping.evidence.colors[target] - 0.5
        bandwidth = cfg.observation_bandwidth_factor * atlas.chart_radii[state.complex.atlas_node_index[local_chart]]
        distance = torch.cdist(means, evidence_position)
        weight = plan[None] * torch.exp(-0.5 * distance.square() / bandwidth.square().clamp_min(1.0e-12))
        weighted_basis = basis[None] * weight[:, :, None]
        gram = torch.einsum("gni,gnj->gij", weighted_basis, basis[None].expand(count, -1, -1))
        rhs = torch.einsum("gni,nc->gic", weighted_basis, color)
        eye = torch.eye(16, dtype=means.dtype, device=means.device)
        gram = gram + cfg.color_ridge * eye
        rhs[:, 0] = rhs[:, 0] + cfg.color_prior_weight * (prior_color - 0.5) / 0.28209479177387814
        gram[:, 0, 0] = gram[:, 0, 0] + cfg.color_prior_weight
        cholesky = torch.linalg.cholesky(gram)
        return torch.cholesky_solve(rhs, cholesky)

    @staticmethod
    def _mesh_from_state(state: ManifoldState) -> MeshAsset:
        vertices = state.position
        faces = state.complex.faces
        face_cross = torch.linalg.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        normals = torch.zeros_like(vertices)
        for corner in range(3):
            normals.index_add_(0, faces[:, corner], face_cross)
        normals = F.normalize(normals, dim=-1)
        colors = torch.sigmoid(state.appearance[:, :3])
        return MeshAsset(vertices, faces, normals, colors)


def _rotation_to_quaternion(rotation: Tensor) -> Tensor:
    """Stable matrix-to-quaternion in scalar-first order."""

    r = rotation
    qw = torch.sqrt(torch.clamp(1.0 + r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2], min=0.0)) / 2.0
    qx = torch.copysign(torch.sqrt(torch.clamp(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2], min=0.0)) / 2.0, r[:, 2, 1] - r[:, 1, 2])
    qy = torch.copysign(torch.sqrt(torch.clamp(1.0 - r[:, 0, 0] + r[:, 1, 1] - r[:, 2, 2], min=0.0)) / 2.0, r[:, 0, 2] - r[:, 2, 0])
    qz = torch.copysign(torch.sqrt(torch.clamp(1.0 - r[:, 0, 0] - r[:, 1, 1] + r[:, 2, 2], min=0.0)) / 2.0, r[:, 1, 0] - r[:, 0, 1])
    return F.normalize(torch.stack((qw, qx, qy, qz), dim=-1), dim=-1)


def write_gaussian_ply(path: str | Path, gaussian: GaussianAsset) -> None:
    """Write deterministic binary little-endian 3DGS PLY attributes."""

    gaussian.validate()
    path = Path(path)
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(45)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    means = gaussian.means.detach().float().cpu().numpy()
    normals = gaussian.rotation[:, :, 2].detach().float().cpu().numpy()
    sh = gaussian.sh_coefficients.detach().float().cpu()
    dc = sh[:, 0].numpy()
    rest = sh[:, 1:].transpose(1, 2).reshape(-1, 45).numpy()
    opacity_logit = torch.logit(gaussian.opacity.clamp(1.0e-7, 1 - 1.0e-7)).detach().float().cpu().numpy()
    log_scale = torch.log(gaussian.scales).detach().float().cpu().numpy()
    quaternion = _rotation_to_quaternion(gaussian.rotation).detach().float().cpu().numpy()
    values = np.concatenate((means, normals, dc, rest, opacity_logit, log_scale, quaternion), axis=1).astype("<f4", copy=False)
    dtype = np.dtype([(name, "<f4") for name in names])
    structured = np.empty(values.shape[0], dtype=dtype)
    for index, name in enumerate(names):
        structured[name] = values[:, index]
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {values.shape[0]}\n"
    header += "".join(f"property float {name}\n" for name in names)
    header += "end_header\n"
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        file.write(structured.tobytes(order="C"))


def _aligned_append(buffer: bytearray, payload: bytes, alignment: int = 4) -> Tuple[int, int]:
    while len(buffer) % alignment:
        buffer.append(0)
    offset = len(buffer)
    buffer.extend(payload)
    return offset, len(payload)


def write_mesh_glb(path: str | Path, mesh: MeshAsset) -> None:
    """Write a deterministic glTF 2.0 binary mesh without an isosurface decoder."""

    path = Path(path)
    position = np.ascontiguousarray(mesh.vertices.detach().float().cpu().numpy().astype("<f4"))
    normal = np.ascontiguousarray(mesh.normals.detach().float().cpu().numpy().astype("<f4"))
    color = np.ascontiguousarray(mesh.colors.detach().float().cpu().numpy().astype("<f4"))
    indices = np.ascontiguousarray(mesh.faces.detach().to(torch.int64).cpu().numpy().astype("<u4").reshape(-1))
    binary = bytearray()
    views = []
    for payload, target in ((position.tobytes(), 34962), (normal.tobytes(), 34962), (color.tobytes(), 34962), (indices.tobytes(), 34963)):
        offset, length = _aligned_append(binary, payload)
        views.append({"buffer": 0, "byteLength": length, "byteOffset": offset, "target": target})
    accessors = [
        {
            "bufferView": 0,
            "byteOffset": 0,
            "componentType": 5126,
            "count": int(position.shape[0]),
            "type": "VEC3",
            "min": position.min(0).tolist(),
            "max": position.max(0).tolist(),
        },
        {"bufferView": 1, "byteOffset": 0, "componentType": 5126, "count": int(normal.shape[0]), "type": "VEC3"},
        {"bufferView": 2, "byteOffset": 0, "componentType": 5126, "count": int(color.shape[0]), "type": "VEC3"},
        {
            "bufferView": 3,
            "byteOffset": 0,
            "componentType": 5125,
            "count": int(indices.size),
            "type": "SCALAR",
            "min": [int(indices.min())],
            "max": [int(indices.max())],
        },
    ]
    document = {
        "accessors": accessors,
        "asset": {"generator": "GRAFT-GS", "version": "2.0"},
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
        "materials": [
            {
                "name": "GRAFT-GS Atlas PBR",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                },
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "COLOR_0": 2,
                            "NORMAL": 1,
                            "POSITION": 0,
                        },
                        "indices": 3,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "nodes": [{"mesh": 0, "name": "GRAFT-GS Atlas"}],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
    }
    json_chunk = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf8")
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    binary.extend(b"\x00" * ((4 - len(binary) % 4) % 4))
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    with path.open("wb") as file:
        file.write(struct.pack("<4sII", b"glTF", 2, total))
        file.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        file.write(json_chunk)
        file.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
        file.write(binary)


__all__ = [
    "AnalyticalReadoutConfig",
    "AnalyticalSurfaceReadout",
    "GaussianAsset",
    "MeshAsset",
    "real_sh_basis_degree3",
    "write_gaussian_ply",
    "write_mesh_glb",
]

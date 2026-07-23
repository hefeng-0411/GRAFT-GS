"""Derived depth/normal supervision from raw source-mesh geometry.

Rasterization treats faces as a triangle soup. It therefore remains valid for
visibility, camera-z depth, and supplied vertex normals even when connectivity
is non-manifold. No topological statement is derived from this operator.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass
class TriangleSoup:
    vertices: Tensor
    faces: Tensor
    normals: Tensor
    normal_provenance: str


@dataclass
class MeshDerivedTargets:
    depth: Tensor
    normal: Tensor
    visibility: Tensor
    normal_validity: Tensor
    normal_provenance: str


def load_triangle_soup(path: str | Path, device: torch.device) -> TriangleSoup:
    """Load finite triangle geometry while making no manifold assumption."""

    try:
        from plyfile import PlyData
    except ImportError as error:
        raise ImportError("mesh-derived supervision requires the declared plyfile dependency") from error
    ply = PlyData.read(str(path))
    vertex_element = ply["vertex"]
    vertices_np = np.column_stack(
        (vertex_element["x"], vertex_element["y"], vertex_element["z"])
    ).astype(np.float32, copy=False)
    face_values = ply["face"].data["vertex_indices"]
    if any(len(face) != 3 for face in face_values):
        raise ValueError("mesh-derived reference rasterizer requires triangular faces")
    faces_np = np.stack(face_values).astype(np.int32, copy=False)
    if faces_np.size and (faces_np.min() < 0 or faces_np.max() >= vertices_np.shape[0]):
        raise ValueError("source mesh contains out-of-range face indices")
    vertex_names = set(ply["vertex"].data.dtype.names or ())
    if {"nx", "ny", "nz"}.issubset(vertex_names):
        normals_np = np.column_stack(
            (ply["vertex"]["nx"], ply["vertex"]["ny"], ply["vertex"]["nz"])
        ).astype(np.float32, copy=False)
        provenance = "source_vertex_normals"
    else:
        normals_np = np.zeros_like(vertices_np)
        triangle = vertices_np[faces_np]
        face_normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
        for corner in range(3):
            np.add.at(normals_np, faces_np[:, corner], face_normal)
        provenance = "derived_area_weighted_face_normals"
    if not np.isfinite(vertices_np).all() or not np.isfinite(normals_np).all():
        raise ValueError("source mesh contains non-finite geometry or normals")
    vertices = torch.from_numpy(np.array(vertices_np, copy=True)).to(device=device)
    faces = torch.from_numpy(np.array(faces_np, copy=True)).to(device=device, dtype=torch.int32)
    normals = torch.from_numpy(np.array(normals_np, copy=True)).to(device=device)
    normal_length = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
    normals = torch.where(
        normal_length > 1.0e-12,
        normals / normal_length.clamp_min(1.0e-12),
        torch.zeros_like(normals),
    )
    return TriangleSoup(vertices, faces, normals, provenance)


class MeshGroundTruthRasterizer:
    """Exact-camera CUDA rasterization with bounded geometry and view memory.

    Mesh targets are immutable supervision, not a trainable renderer.  Views are
    therefore rasterized in deterministic contiguous chunks.  This preserves
    the exact per-view camera/triangle result while preventing nvdiffrast's
    transient geometry and binning buffers from scaling with the full
    same-object DDP view budget.
    """

    def __init__(
        self,
        device: torch.device,
        near: float = 0.01,
        far: float = 100.0,
        cache_size: int = 2,
        view_chunk_size: int = 2,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("mesh-derived rasterization is an A800 CUDA supervision path")
        if not 0 < near < far or cache_size < 1 or view_chunk_size < 1:
            raise ValueError("invalid near/far planes, cache size, or view chunk size")
        try:
            import nvdiffrast.torch as dr
        except ImportError as error:
            raise RuntimeError(
                "mesh-derived depth/normal supervision requires nvdiffrast on the A800 server"
            ) from error
        self.dr = dr
        self.context = dr.RasterizeCudaContext(device=device)
        self.device = device
        self.near = near
        self.far = far
        self.cache_size = cache_size
        self.view_chunk_size = view_chunk_size
        self.cache: OrderedDict[str, TriangleSoup] = OrderedDict()

    def _mesh(self, path: str | Path) -> TriangleSoup:
        key = str(Path(path).resolve())
        if key in self.cache:
            mesh = self.cache.pop(key)
            self.cache[key] = mesh
            return mesh
        mesh = load_triangle_soup(key, self.device)
        self.cache[key] = mesh
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return mesh

    @torch.no_grad()
    def __call__(
        self,
        path: str | Path,
        extrinsics_world_to_camera: Tensor,
        intrinsics_pixels: Tensor,
        height: int,
        width: int,
    ) -> MeshDerivedTargets:
        mesh = self._mesh(path)
        extrinsics = extrinsics_world_to_camera.to(
            device=self.device, dtype=mesh.vertices.dtype
        )
        intrinsics = intrinsics_pixels.to(device=self.device, dtype=mesh.vertices.dtype)
        if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (3, 4):
            raise ValueError("mesh cameras must have shape [K,3,4]")
        if intrinsics.shape != (extrinsics.shape[0], 3, 3):
            raise ValueError("mesh intrinsics must have shape [K,3,3]")
        k = extrinsics.shape[0]
        if k < 1:
            raise ValueError("mesh supervision requires at least one camera")
        homogeneous = torch.cat(
            (mesh.vertices, torch.ones_like(mesh.vertices[:, :1])), dim=-1
        )
        depths: list[Tensor] = []
        normals: list[Tensor] = []
        visibilities: list[Tensor] = []
        normal_validities: list[Tensor] = []
        for start in range(0, k, self.view_chunk_size):
            stop = min(start + self.view_chunk_size, k)
            chunk_extrinsics = extrinsics[start:stop]
            chunk_intrinsics = intrinsics[start:stop]
            chunk_size = stop - start
            full_extrinsic = (
                torch.eye(4, device=self.device, dtype=mesh.vertices.dtype)
                .expand(chunk_size, 4, 4)
                .clone()
            )
            full_extrinsic[:, :3] = chunk_extrinsics
            normalized = chunk_intrinsics.clone()
            normalized[:, 0] /= float(width)
            normalized[:, 1] /= float(height)
            projection = torch.zeros(
                chunk_size, 4, 4, dtype=mesh.vertices.dtype, device=self.device
            )
            projection[:, 0, 0] = 2.0 * normalized[:, 0, 0]
            projection[:, 1, 1] = 2.0 * normalized[:, 1, 1]
            projection[:, 0, 2] = 2.0 * normalized[:, 0, 2] - 1.0
            projection[:, 1, 2] = -2.0 * normalized[:, 1, 2] + 1.0
            projection[:, 2, 2] = self.far / (self.far - self.near)
            projection[:, 2, 3] = self.near * self.far / (self.near - self.far)
            projection[:, 3, 2] = 1.0

            chunk_homogeneous = homogeneous.expand(chunk_size, -1, -1)
            camera_vertex = chunk_homogeneous @ full_extrinsic.transpose(-1, -2)
            clip_vertex = chunk_homogeneous @ (
                projection @ full_extrinsic
            ).transpose(-1, -2)
            raster, _ = self.dr.rasterize(
                self.context,
                clip_vertex,
                mesh.faces,
                (height, width),
            )
            visibility = raster[..., 3:4] > 0
            depth = self.dr.interpolate(
                camera_vertex[..., 2:3].contiguous(), raster, mesh.faces
            )[0]
            camera_normal = torch.einsum(
                "kij,vj->kvi",
                chunk_extrinsics[:, :3, :3],
                mesh.normals,
            )
            normal = self.dr.interpolate(
                camera_normal.contiguous(), raster, mesh.faces
            )[0]
            normal_length = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
            normal_validity = visibility & (normal_length > 1.0e-8)
            normal = torch.where(
                normal_validity,
                normal / normal_length.clamp_min(1.0e-8),
                torch.zeros_like(normal),
            )
            depth = torch.where(visibility, depth, torch.zeros_like(depth))
            depths.append(depth.permute(0, 3, 1, 2))
            normals.append(normal.permute(0, 3, 1, 2))
            visibilities.append(visibility.permute(0, 3, 1, 2))
            normal_validities.append(normal_validity.permute(0, 3, 1, 2))

        return MeshDerivedTargets(
            depth=torch.cat(depths, dim=0),
            normal=torch.cat(normals, dim=0),
            visibility=torch.cat(visibilities, dim=0),
            normal_validity=torch.cat(normal_validities, dim=0),
            normal_provenance=mesh.normal_provenance,
        )


__all__ = [
    "MeshDerivedTargets",
    "MeshGroundTruthRasterizer",
    "TriangleSoup",
    "load_triangle_soup",
]

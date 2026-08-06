"""Batched CUDA Gaussian rasterization through the local gsplat extension."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .assets import GaussianAsset


@dataclass
class CameraBatch:
    extrinsics_world_to_camera: Tensor  # [K,3,4], OpenCV world-to-camera
    intrinsics: Tensor  # [K,3,3], OpenCV pinhole calibration
    height: int
    width: int

    def __post_init__(self) -> None:
        extrinsics = self.extrinsics_world_to_camera
        intrinsics = self.intrinsics
        if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (3, 4):
            raise ValueError("camera extrinsics must have shape [K,3,4]")
        if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError("camera intrinsics must have shape [K,3,3]")
        if extrinsics.shape[0] != intrinsics.shape[0] or intrinsics.shape[0] < 1:
            raise ValueError("camera batches must have the same positive view count")
        if self.height < 1 or self.width < 1:
            raise ValueError("render dimensions must be positive")
        if not extrinsics.dtype.is_floating_point or not intrinsics.dtype.is_floating_point:
            raise TypeError("camera tensors must be floating point")
        if extrinsics.device != intrinsics.device or extrinsics.dtype != intrinsics.dtype:
            raise ValueError("camera tensors must share device and dtype")
        with torch.no_grad():
            if not bool(torch.all(torch.isfinite(extrinsics))) or not bool(
                torch.all(torch.isfinite(intrinsics))
            ):
                raise ValueError("camera tensors contain non-finite values")
            if bool(torch.any(intrinsics[:, (0, 1), (0, 1)] <= 0.0)):
                raise ValueError("OpenCV focal lengths must be positive")
            expected_last_row = intrinsics.new_tensor((0.0, 0.0, 1.0)).expand(
                intrinsics.shape[0], -1
            )
            if not torch.allclose(
                intrinsics[:, 2], expected_last_row, atol=1.0e-6, rtol=1.0e-6
            ):
                raise ValueError("intrinsics must use the OpenCV homogeneous last row")
            rotation = extrinsics[:, :3, :3]
            identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
            orthogonality = torch.linalg.matrix_norm(
                rotation.transpose(-1, -2) @ rotation - identity,
                ord="fro",
                dim=(-2, -1),
            )
            determinant_error = (torch.linalg.det(rotation) - 1.0).abs()
            if bool(torch.any(orthogonality > 5.0e-3)) or bool(
                torch.any(determinant_error > 5.0e-3)
            ):
                raise ValueError("world-to-camera rotations must lie on SO(3)")


@dataclass
class RenderResult:
    color: Tensor  # [K,3,H,W]
    alpha: Tensor  # [K,1,H,W]
    depth: Tensor  # [K,1,H,W]
    normal: Tensor  # [K,3,H,W], camera-coordinate atlas normals


@dataclass(frozen=True)
class RasterizationContract:
    """Numerical settings mapped directly to gsplat's CUDA rasterizer ABI."""

    kernel_size: float = 0.1
    sigma_extent: float = 3.0
    alpha_ceiling: float = 0.99
    alpha_threshold: float = 1.0 / 255.0
    transmittance_threshold: float = 1.0e-4
    determinant_epsilon: float = 1.0e-6
    visibility_near: float = 0.2
    tile_size: int = 16

    def __post_init__(self) -> None:
        if self.kernel_size < 0.0:
            raise ValueError("gsplat eps2d must be non-negative")
        if self.visibility_near < 0.0:
            raise ValueError("gsplat near plane must be non-negative")
        if self.tile_size not in (8, 16, 32):
            raise ValueError("gsplat tile_size must be 8, 16, or 32")


def _background_color(background: Tensor | float, reference: Tensor) -> Tensor:
    value = torch.as_tensor(
        background, dtype=reference.dtype, device=reference.device
    )
    if value.ndim == 0:
        value = value.expand(3)
    if value.shape != (3,):
        raise ValueError("render background must be a scalar or RGB vector")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError("render background contains non-finite values")
    return value.contiguous()


def _cuda_float32_arena(name: str, value: Tensor) -> Tensor:
    if value.device.type != "cuda":
        raise ValueError(f"{name} must reside on CUDA")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use float32 for the gsplat ABI")
    if not value.is_contiguous():
        raise ValueError(
            f"{name} must already be contiguous; implicit arena copies are forbidden"
        )
    return value.contiguous()


class GsplatRenderer(nn.Module):
    """One-call, packed, tile-sorted gsplat forward/backward rasterization."""

    def __init__(
        self,
        near: float = 0.01,
        far: float = 100.0,
        contract: RasterizationContract = RasterizationContract(),
        checkpoint_views: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < near < far:
            raise ValueError("projection planes must satisfy 0 < near < far")
        if not isinstance(checkpoint_views, bool):
            raise TypeError("checkpoint_views must be Boolean")
        self.near = near
        self.far = far
        self.contract = contract
        self.checkpoint_views = checkpoint_views

    def forward(
        self,
        gaussian: GaussianAsset,
        cameras: CameraBatch,
        background: Tensor | float = 0.0,
    ) -> RenderResult:
        try:
            from gsplat.cuda._math import _rotmat_to_quat
            from gsplat.rendering import rasterization
        except ImportError as error:
            raise RuntimeError(
                "production rendering requires the local gsplat CUDA extension"
            ) from error

        gaussian.validate()
        if gaussian.means.device != cameras.intrinsics.device:
            raise ValueError("Gaussian assets and cameras must share a CUDA device")
        means = _cuda_float32_arena("gaussian.means", gaussian.means)
        rotations = _cuda_float32_arena("gaussian.rotation", gaussian.rotation)
        scales = _cuda_float32_arena("gaussian.scales", gaussian.scales)
        spherical_harmonics = _cuda_float32_arena(
            "gaussian.sh_coefficients", gaussian.sh_coefficients
        )
        opacity_matrix = _cuda_float32_arena(
            "gaussian.opacity", gaussian.opacity
        )
        extrinsics = _cuda_float32_arena(
            "camera.extrinsics_world_to_camera",
            cameras.extrinsics_world_to_camera,
        )
        intrinsics = _cuda_float32_arena("camera.intrinsics", cameras.intrinsics)

        quaternions = _rotmat_to_quat(rotations.contiguous()).contiguous()
        if quaternions.shape != (means.shape[0], 4):
            raise RuntimeError("gsplat quaternion conversion returned an invalid shape")
        opacities = opacity_matrix[:, 0].contiguous()
        view_count = extrinsics.shape[0]
        viewmats = torch.eye(
            4, dtype=torch.float32, device=means.device
        ).repeat(view_count, 1, 1)
        viewmats[:, :3, :] = extrinsics.contiguous()
        viewmats = viewmats.contiguous()
        backgrounds = (
            _background_color(background, means)
            .reshape(1, 3)
            .expand(view_count, 3)
            .contiguous()
        )
        world_normals = rotations[:, :, 2].contiguous()

        with torch.autocast(device_type="cuda", enabled=False):
            rendered, alpha, metadata = rasterization(
                means=means.contiguous(),
                quats=quaternions.contiguous(),
                scales=scales.contiguous(),
                opacities=opacities.contiguous(),
                colors=spherical_harmonics.contiguous(),
                viewmats=viewmats.contiguous(),
                Ks=intrinsics.contiguous(),
                width=cameras.width,
                height=cameras.height,
                near_plane=max(self.near, self.contract.visibility_near),
                far_plane=self.far,
                radius_clip=0.0,
                eps2d=self.contract.kernel_size,
                sh_degree=3,
                packed=True,
                tile_size=self.contract.tile_size,
                backgrounds=backgrounds.contiguous(),
                render_mode="RGB+ED",
                sparse_grad=False,
                absgrad=False,
                rasterize_mode="antialiased",
                channel_chunk=32,
                distributed=False,
                camera_model="pinhole",
                segmented=False,
                global_z_order=True,
                extra_signals=world_normals.contiguous(),
            )

        expected_render_shape = (
            view_count,
            cameras.height,
            cameras.width,
            4,
        )
        expected_alpha_shape = expected_render_shape[:-1] + (1,)
        if rendered.shape != expected_render_shape or alpha.shape != expected_alpha_shape:
            raise RuntimeError(
                "gsplat returned an invalid RGB+ED raster shape: "
                f"rendered={tuple(rendered.shape)}, alpha={tuple(alpha.shape)}"
            )
        accumulated_world_normal = metadata.get("render_extra_signals")
        if accumulated_world_normal is None or accumulated_world_normal.shape != (
            view_count,
            cameras.height,
            cameras.width,
            3,
        ):
            raise RuntimeError("gsplat did not return the fused atlas-normal signal")

        color = rendered[..., :3].permute(0, 3, 1, 2).contiguous()
        depth = rendered[..., 3:4].permute(0, 3, 1, 2).contiguous()
        alpha_nchw = alpha.permute(0, 3, 1, 2).contiguous()
        camera_normal = torch.einsum(
            "khwc,kdc->khwd",
            accumulated_world_normal.contiguous(),
            extrinsics[:, :3, :3].contiguous(),
        )
        camera_normal = torch.nn.functional.normalize(
            camera_normal, dim=-1, eps=torch.finfo(torch.float32).eps
        )
        camera_normal = torch.where(
            alpha > 0.0,
            camera_normal,
            torch.zeros_like(camera_normal),
        )
        normal = camera_normal.permute(0, 3, 1, 2).contiguous()
        return RenderResult(
            color=color,
            alpha=alpha_nchw,
            depth=depth,
            normal=normal,
        )


class GsplatGaussianRenderer(GsplatRenderer):
    """Compatibility name for existing serialized configurations."""


class CudaGaussianRenderer(GsplatRenderer):
    """Compatibility name; execution remains the gsplat CUDA path."""


__all__ = [
    "CameraBatch",
    "CudaGaussianRenderer",
    "GsplatGaussianRenderer",
    "GsplatRenderer",
    "RasterizationContract",
    "RenderResult",
]

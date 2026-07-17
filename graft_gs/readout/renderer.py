"""High-precision differentiable reference Gaussian rasterizer.

This path prioritizes numerical transparency and gradient verification.  It is
not intended to replace a numerically matched tile CUDA kernel on the A800
server.  Sorting and raster bounds are discrete visibility decisions, while all
Gaussian contributions inside retained bounds remain differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor, nn

from .assets import GaussianAsset, real_sh_basis_degree3


@dataclass
class CameraBatch:
    extrinsics_world_to_camera: Tensor  # [K,3,4]
    intrinsics: Tensor  # [K,3,3]
    height: int
    width: int


@dataclass
class RenderResult:
    color: Tensor  # [K,3,H,W]
    alpha: Tensor  # [K,1,H,W]
    depth: Tensor  # [K,1,H,W]
    normal: Tensor  # [K,3,H,W], analytically oriented atlas normals


class ReferenceGaussianRenderer(nn.Module):
    def __init__(self, sigma_extent: float = 3.5, covariance_floor_pixels: float = 0.25, alpha_ceiling: float = 0.999) -> None:
        super().__init__()
        self.sigma_extent = sigma_extent
        self.covariance_floor_pixels = covariance_floor_pixels
        self.alpha_ceiling = alpha_ceiling

    def forward(self, gaussian: GaussianAsset, cameras: CameraBatch, background: Tensor | float = 0.0) -> RenderResult:
        gaussian.validate()
        outputs: List[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        for camera_index in range(cameras.extrinsics_world_to_camera.shape[0]):
            outputs.append(
                self._render_one(
                    gaussian,
                    cameras.extrinsics_world_to_camera[camera_index],
                    cameras.intrinsics[camera_index],
                    cameras.height,
                    cameras.width,
                    background,
                )
            )
        color, alpha, depth, normal = zip(*outputs)
        return RenderResult(
            torch.stack(color),
            torch.stack(alpha),
            torch.stack(depth),
            torch.stack(normal),
        )

    def _render_one(
        self,
        gaussian: GaussianAsset,
        extrinsic: Tensor,
        intrinsic: Tensor,
        height: int,
        width: int,
        background: Tensor | float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dtype, device = gaussian.means.dtype, gaussian.means.device
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        camera_points = gaussian.means @ rotation.transpose(0, 1) + translation
        z = camera_points[:, 2]
        x, y = camera_points[:, 0], camera_points[:, 1]
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        pixel = torch.stack((fx * x / z + cx, fy * y / z + cy), dim=-1)
        projection_jacobian_camera = torch.stack(
            (
                torch.stack((fx / z, torch.zeros_like(z), -fx * x / z.square()), dim=-1),
                torch.stack((torch.zeros_like(z), fy / z, -fy * y / z.square()), dim=-1),
            ),
            dim=-2,
        )
        projection_jacobian = projection_jacobian_camera @ rotation
        covariance_2d = projection_jacobian @ gaussian.covariance @ projection_jacobian.transpose(-1, -2)
        covariance_2d = covariance_2d + self.covariance_floor_pixels * torch.eye(2, dtype=dtype, device=device)
        camera_center = -(rotation.transpose(0, 1) @ translation)
        normal_camera = gaussian.rotation[:, :, 2] @ rotation.transpose(0, 1)
        view_direction = torch.nn.functional.normalize(camera_center[None] - gaussian.means, dim=-1)
        basis = real_sh_basis_degree3(view_direction)
        color_gaussian = torch.clamp(0.5 + torch.einsum("gi,gic->gc", basis, gaussian.sh_coefficients), 0.0, 1.0)
        order = torch.argsort(z)
        background_tensor = torch.as_tensor(background, dtype=dtype, device=device)
        if background_tensor.ndim == 0:
            background_tensor = background_tensor.expand(3)
        accumulated = torch.zeros(3, height, width, dtype=dtype, device=device)
        alpha_accumulated = torch.zeros(height, width, dtype=dtype, device=device)
        depth_accumulated = torch.zeros(height, width, dtype=dtype, device=device)
        normal_accumulated = torch.zeros(3, height, width, dtype=dtype, device=device)
        transmittance = torch.ones(height, width, dtype=dtype, device=device)
        for index_tensor in order:
            index = int(index_tensor)
            if float(z[index].detach()) <= 0:
                continue
            eigenvalue = torch.linalg.eigvalsh(covariance_2d[index]).clamp_min(0)
            radius = self.sigma_extent * torch.sqrt(eigenvalue[-1])
            u_min = max(0, int(torch.floor(pixel[index, 0] - radius).detach()))
            u_max = min(width, int(torch.ceil(pixel[index, 0] + radius).detach()) + 1)
            v_min = max(0, int(torch.floor(pixel[index, 1] - radius).detach()))
            v_max = min(height, int(torch.ceil(pixel[index, 1] + radius).detach()) + 1)
            if u_min >= u_max or v_min >= v_max:
                continue
            u = torch.arange(u_min, u_max, dtype=dtype, device=device)
            v = torch.arange(v_min, v_max, dtype=dtype, device=device)
            vv, uu = torch.meshgrid(v, u, indexing="ij")
            delta = torch.stack((uu - pixel[index, 0], vv - pixel[index, 1]), dim=-1)
            precision = torch.linalg.inv(covariance_2d[index])
            exponent = -0.5 * torch.einsum("...i,ij,...j->...", delta, precision, delta)
            footprint = torch.exp(exponent)
            alpha = (gaussian.opacity[index, 0] * footprint).clamp(0.0, self.alpha_ceiling)
            local_transmittance = transmittance[v_min:v_max, u_min:u_max]
            contribution = local_transmittance * alpha
            accumulated[:, v_min:v_max, u_min:u_max] = accumulated[:, v_min:v_max, u_min:u_max] + contribution[None] * color_gaussian[index, :, None, None]
            depth_accumulated[v_min:v_max, u_min:u_max] = depth_accumulated[v_min:v_max, u_min:u_max] + contribution * z[index]
            normal_accumulated[:, v_min:v_max, u_min:u_max] = (
                normal_accumulated[:, v_min:v_max, u_min:u_max]
                + contribution[None] * normal_camera[index, :, None, None]
            )
            alpha_accumulated[v_min:v_max, u_min:u_max] = alpha_accumulated[v_min:v_max, u_min:u_max] + contribution
            transmittance = transmittance.clone()
            transmittance[v_min:v_max, u_min:u_max] = local_transmittance * (1.0 - alpha)
        accumulated = accumulated + transmittance[None] * background_tensor[:, None, None]
        depth = depth_accumulated / alpha_accumulated.clamp_min(torch.finfo(dtype).eps)
        normal = torch.nn.functional.normalize(normal_accumulated, dim=0, eps=torch.finfo(dtype).eps)
        normal = torch.where(alpha_accumulated[None] > 0, normal, torch.zeros_like(normal))
        return accumulated, alpha_accumulated[None], depth[None], normal


def _rotation_to_quaternion(rotation: Tensor) -> Tensor:
    r = rotation
    qw = torch.sqrt(torch.clamp(1.0 + r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2], min=0.0)) / 2.0
    qx = torch.copysign(torch.sqrt(torch.clamp(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2], min=0.0)) / 2.0, r[:, 2, 1] - r[:, 1, 2])
    qy = torch.copysign(torch.sqrt(torch.clamp(1.0 - r[:, 0, 0] + r[:, 1, 1] - r[:, 2, 2], min=0.0)) / 2.0, r[:, 0, 2] - r[:, 2, 0])
    qz = torch.copysign(torch.sqrt(torch.clamp(1.0 - r[:, 0, 0] - r[:, 1, 1] + r[:, 2, 2], min=0.0)) / 2.0, r[:, 1, 0] - r[:, 0, 1])
    return torch.nn.functional.normalize(torch.stack((qw, qx, qy, qz), dim=-1), dim=-1)


class CudaGaussianRenderer(nn.Module):
    """Differentiable CUDA rasterizer adapter used after reference equivalence."""

    def __init__(self, near: float = 0.01, far: float = 100.0) -> None:
        super().__init__()
        self.near = near
        self.far = far

    @staticmethod
    def _projection(intrinsic: Tensor, height: int, width: int, near: float, far: float) -> Tensor:
        projection = torch.zeros(4, 4, dtype=intrinsic.dtype, device=intrinsic.device)
        projection[0, 0] = 2.0 * intrinsic[0, 0] / width
        projection[1, 1] = 2.0 * intrinsic[1, 1] / height
        projection[0, 2] = 2.0 * intrinsic[0, 2] / width - 1.0
        projection[1, 2] = -2.0 * intrinsic[1, 2] / height + 1.0
        projection[2, 2] = far / (far - near)
        projection[2, 3] = near * far / (near - far)
        projection[3, 2] = 1.0
        return projection

    def forward(self, gaussian: GaussianAsset, cameras: CameraBatch, background: Tensor | float = 0.0) -> RenderResult:
        try:
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        except ImportError as error:
            raise RuntimeError(
                "CUDA rendering requires the server-built diff_gaussian_rasterization extension; "
                "run numerical equivalence tests against ReferenceGaussianRenderer before use"
            ) from error
        color_images, alpha_images, depth_images, normal_images = [], [], [], []
        quaternion = _rotation_to_quaternion(gaussian.rotation)
        for view in range(cameras.extrinsics_world_to_camera.shape[0]):
            extrinsic = cameras.extrinsics_world_to_camera[view]
            intrinsic = cameras.intrinsics[view]
            view_matrix = torch.eye(4, dtype=extrinsic.dtype, device=extrinsic.device)
            view_matrix[:3] = extrinsic
            projection = self._projection(intrinsic, cameras.height, cameras.width, self.near, self.far)
            camera_center = torch.linalg.inv(view_matrix)[:3, 3]
            view_direction = torch.nn.functional.normalize(camera_center[None] - gaussian.means, dim=-1)
            basis = real_sh_basis_degree3(view_direction)
            color = torch.clamp(0.5 + torch.einsum("gi,gic->gc", basis, gaussian.sh_coefficients), 0.0, 1.0)
            bg = torch.as_tensor(background, dtype=gaussian.means.dtype, device=gaussian.means.device)
            if bg.ndim == 0:
                bg = bg.expand(3)
            settings_kwargs = dict(
                image_height=cameras.height,
                image_width=cameras.width,
                tanfovx=float(cameras.width / (2.0 * intrinsic[0, 0])),
                tanfovy=float(cameras.height / (2.0 * intrinsic[1, 1])),
                bg=bg,
                scale_modifier=1.0,
                viewmatrix=view_matrix.transpose(0, 1).contiguous(),
                projmatrix=(projection @ view_matrix).transpose(0, 1).contiguous(),
                sh_degree=0,
                campos=camera_center,
                prefiltered=False,
                debug=False,
            )
            try:
                settings = GaussianRasterizationSettings(**settings_kwargs)
            except TypeError:
                settings_kwargs.update(
                    kernel_size=0.1,
                    subpixel_offset=torch.zeros(cameras.height, cameras.width, 2, device=gaussian.means.device),
                )
                settings = GaussianRasterizationSettings(**settings_kwargs)
            rasterizer = GaussianRasterizer(raster_settings=settings)
            screen = torch.zeros_like(gaussian.means, requires_grad=True)

            def rasterize(precomputed_color: Tensor) -> Tensor:
                rendered, _ = rasterizer(
                    means3D=gaussian.means,
                    means2D=screen,
                    shs=None,
                    colors_precomp=precomputed_color,
                    opacities=gaussian.opacity,
                    scales=gaussian.scales,
                    rotations=quaternion,
                    cov3D_precomp=None,
                )
                return rendered

            rendered_color = rasterize(color)
            alpha_rgb = rasterize(torch.ones_like(color))
            alpha = alpha_rgb.mean(0, keepdim=True).clamp(0.0, 1.0)
            camera_depth = (gaussian.means @ extrinsic[:3, :3].transpose(0, 1) + extrinsic[:3, 3])[:, 2]
            depth_rgb = rasterize(camera_depth[:, None].expand(-1, 3))
            depth = depth_rgb.mean(0, keepdim=True) / alpha.clamp_min(1.0e-8)
            normal_camera = gaussian.rotation[:, :, 2] @ extrinsic[:3, :3].transpose(0, 1)
            normal_rgb = rasterize(normal_camera)
            normal = normal_rgb / alpha.clamp_min(1.0e-8)
            normal = torch.nn.functional.normalize(normal, dim=0, eps=1.0e-8)
            normal = torch.where(alpha > 0, normal, torch.zeros_like(normal))
            color_images.append(rendered_color)
            alpha_images.append(alpha)
            depth_images.append(depth)
            normal_images.append(normal)
        return RenderResult(
            torch.stack(color_images),
            torch.stack(alpha_images),
            torch.stack(depth_images),
            torch.stack(normal_images),
        )


__all__ = ["CameraBatch", "CudaGaussianRenderer", "ReferenceGaussianRenderer", "RenderResult"]

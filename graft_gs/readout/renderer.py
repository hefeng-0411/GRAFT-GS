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

    def __post_init__(self) -> None:
        if self.extrinsics_world_to_camera.ndim != 3 or self.extrinsics_world_to_camera.shape[-2:] != (3, 4):
            raise ValueError("camera extrinsics must have shape [K,3,4]")
        if self.intrinsics.ndim != 3 or self.intrinsics.shape[-2:] != (3, 3):
            raise ValueError("camera intrinsics must have shape [K,3,3]")
        if self.extrinsics_world_to_camera.shape[0] != self.intrinsics.shape[0] or self.intrinsics.shape[0] < 1:
            raise ValueError("camera batches must contain the same positive view count")
        if self.height < 1 or self.width < 1:
            raise ValueError("render dimensions must be positive")
        if not self.extrinsics_world_to_camera.dtype.is_floating_point or not self.intrinsics.dtype.is_floating_point:
            raise TypeError("camera tensors must be floating point")
        if self.extrinsics_world_to_camera.device != self.intrinsics.device:
            raise ValueError("camera extrinsics and intrinsics must share a device")
        if self.extrinsics_world_to_camera.dtype != self.intrinsics.dtype:
            raise ValueError("camera extrinsics and intrinsics must share a dtype")
        with torch.no_grad():
            if not bool(torch.all(torch.isfinite(self.extrinsics_world_to_camera))) or not bool(
                torch.all(torch.isfinite(self.intrinsics))
            ):
                raise ValueError("camera tensors contain non-finite values")
            if bool(torch.any(self.intrinsics[:, (0, 1), (0, 1)] <= 0)):
                raise ValueError("OpenCV focal lengths must be positive")
            expected_last_row = self.intrinsics.new_tensor([0.0, 0.0, 1.0]).expand(
                self.intrinsics.shape[0], -1
            )
            if not torch.allclose(
                self.intrinsics[:, 2], expected_last_row, atol=1.0e-6, rtol=1.0e-6
            ):
                raise ValueError("intrinsics must use the OpenCV homogeneous last row")
            rotation = self.extrinsics_world_to_camera[:, :3, :3]
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
    normal: Tensor  # [K,3,H,W], analytically oriented atlas normals


@dataclass(frozen=True)
class RasterizationContract:
    r"""Numerical contract shared with TRELLIS mip-splatting.

    ``kernel_size`` is the variance of the isotropic screen-space mip filter,
    in squared pixels.  The remaining thresholds are the explicit constants
    in TRELLIS' ``diff_gaussian_rasterization`` CUDA forward kernel.  Keeping
    them here prevents the high-precision reference path from silently
    optimizing a different image formation model than the A800 kernel.
    """

    kernel_size: float = 0.1
    sigma_extent: float = 3.0
    alpha_ceiling: float = 0.99
    alpha_threshold: float = 1.0 / 255.0
    transmittance_threshold: float = 1.0e-4
    determinant_epsilon: float = 1.0e-6
    visibility_near: float = 0.2
    tile_size: int = 16

    def __post_init__(self) -> None:
        if self.kernel_size < 0:
            raise ValueError("mip kernel variance must be non-negative")
        if self.sigma_extent <= 0:
            raise ValueError("Gaussian extent must be positive")
        if not 0 < self.alpha_ceiling < 1:
            raise ValueError("alpha ceiling must lie in (0,1)")
        if not 0 <= self.alpha_threshold < self.alpha_ceiling:
            raise ValueError("alpha threshold is outside its numerical domain")
        if not 0 < self.transmittance_threshold < 1:
            raise ValueError("transmittance threshold must lie in (0,1)")
        if self.determinant_epsilon <= 0 or self.visibility_near < 0:
            raise ValueError("determinant epsilon and near visibility must be valid")
        if self.tile_size < 1:
            raise ValueError("tile size must be positive")


def _mip_filter_covariance(covariance_2d: Tensor, contract: RasterizationContract) -> tuple[Tensor, Tensor]:
    """Apply the TRELLIS mip filter and its measure-preserving peak factor."""

    raw_determinant = torch.linalg.det(covariance_2d)
    identity = torch.eye(2, dtype=covariance_2d.dtype, device=covariance_2d.device)
    filtered = covariance_2d + contract.kernel_size * identity
    filtered_determinant = torch.linalg.det(filtered)
    epsilon = covariance_2d.new_tensor(contract.determinant_epsilon)
    determinant_0 = raw_determinant.clamp_min(epsilon)
    determinant_1 = filtered_determinant.clamp_min(epsilon)
    peak_scale = torch.sqrt(
        determinant_0 / (determinant_1 + epsilon) + epsilon
    )
    valid = (raw_determinant > epsilon) & (filtered_determinant > epsilon)
    return filtered, torch.where(valid, peak_scale, torch.zeros_like(peak_scale))


def _background_color(background: Tensor | float, reference: Tensor) -> Tensor:
    value = torch.as_tensor(background, dtype=reference.dtype, device=reference.device)
    if value.ndim == 0:
        value = value.expand(3)
    if value.shape != (3,):
        raise ValueError("render background must be a scalar or RGB vector")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError("render background contains non-finite values")
    return value


class ReferenceGaussianRenderer(nn.Module):
    def __init__(self, contract: RasterizationContract = RasterizationContract()) -> None:
        super().__init__()
        self.contract = contract

    def forward(self, gaussian: GaussianAsset, cameras: CameraBatch, background: Tensor | float = 0.0) -> RenderResult:
        gaussian.validate()
        if gaussian.means.device != cameras.intrinsics.device:
            raise ValueError("Gaussian assets and cameras must share a device")
        if gaussian.means.dtype != cameras.intrinsics.dtype:
            raise ValueError("reference rendering requires matching Gaussian/camera dtypes")
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
        tangent_x = width / (2.0 * fx)
        tangent_y = height / (2.0 * fy)
        covariance_x = (x / z).clamp(-1.3 * tangent_x, 1.3 * tangent_x) * z
        covariance_y = (y / z).clamp(-1.3 * tangent_y, 1.3 * tangent_y) * z
        projection_jacobian_camera = torch.stack(
            (
                torch.stack(
                    (fx / z, torch.zeros_like(z), -fx * covariance_x / z.square()),
                    dim=-1,
                ),
                torch.stack(
                    (torch.zeros_like(z), fy / z, -fy * covariance_y / z.square()),
                    dim=-1,
                ),
            ),
            dim=-2,
        )
        projection_jacobian = projection_jacobian_camera @ rotation
        covariance_2d_raw = projection_jacobian @ gaussian.covariance @ projection_jacobian.transpose(-1, -2)
        covariance_2d, mip_peak_scale = _mip_filter_covariance(
            covariance_2d_raw,
            self.contract,
        )
        camera_center = -(rotation.transpose(0, 1) @ translation)
        normal_camera = gaussian.rotation[:, :, 2] @ rotation.transpose(0, 1)
        view_direction = torch.nn.functional.normalize(camera_center[None] - gaussian.means, dim=-1)
        basis = real_sh_basis_degree3(view_direction)
        color_gaussian = torch.clamp(0.5 + torch.einsum("gi,gic->gc", basis, gaussian.sh_coefficients), 0.0, 1.0)
        # CUB radix sorting in the TRELLIS kernel is stable for equal depth
        # keys, so preserve analytical Gaussian order on coplanar charts.
        order = torch.argsort(z, stable=True)
        background_tensor = _background_color(background, gaussian.means)
        accumulated = torch.zeros(3, height, width, dtype=dtype, device=device)
        alpha_accumulated = torch.zeros(height, width, dtype=dtype, device=device)
        depth_accumulated = torch.zeros(height, width, dtype=dtype, device=device)
        normal_accumulated = torch.zeros(3, height, width, dtype=dtype, device=device)
        transmittance = torch.ones(height, width, dtype=dtype, device=device)
        active_pixel = torch.ones(height, width, dtype=torch.bool, device=device)
        for index_tensor in order:
            index = int(index_tensor)
            if float(z[index].detach()) <= self.contract.visibility_near:
                continue
            covariance = covariance_2d[index]
            determinant = torch.linalg.det(covariance)
            midpoint = 0.5 * torch.trace(covariance)
            # TRELLIS intentionally floors the discriminant at 0.1 before
            # choosing the 3-sigma tile rectangle.
            lambda_max = midpoint + torch.sqrt(
                torch.clamp(midpoint.square() - determinant, min=0.1)
            )
            radius = torch.ceil(
                self.contract.sigma_extent * torch.sqrt(lambda_max.clamp_min(0.0))
            )
            tile = self.contract.tile_size
            tile_width = (width + tile - 1) // tile
            tile_height = (height + tile - 1) // tile
            tile_minimum_u = max(
                0,
                int(torch.trunc((pixel[index, 0] - radius) / tile).detach()),
            )
            tile_maximum_u = min(
                tile_width,
                int(torch.trunc((pixel[index, 0] + radius + tile - 1) / tile).detach()),
            )
            tile_minimum_v = max(
                0,
                int(torch.trunc((pixel[index, 1] - radius) / tile).detach()),
            )
            tile_maximum_v = min(
                tile_height,
                int(torch.trunc((pixel[index, 1] + radius + tile - 1) / tile).detach()),
            )
            u_min = tile_minimum_u * tile
            u_max = min(width, tile_maximum_u * tile)
            v_min = tile_minimum_v * tile
            v_max = min(height, tile_maximum_v * tile)
            if u_min >= u_max or v_min >= v_max:
                continue
            u = torch.arange(u_min, u_max, dtype=dtype, device=device)
            v = torch.arange(v_min, v_max, dtype=dtype, device=device)
            vv, uu = torch.meshgrid(v, u, indexing="ij")
            delta = torch.stack((uu - pixel[index, 0], vv - pixel[index, 1]), dim=-1)
            precision = torch.linalg.inv(covariance_2d[index])
            exponent = -0.5 * torch.einsum("...i,ij,...j->...", delta, precision, delta)
            footprint = torch.exp(exponent)
            alpha = (
                gaussian.opacity[index, 0] * mip_peak_scale[index] * footprint
            ).clamp(max=self.contract.alpha_ceiling)
            alpha = torch.where(
                alpha >= self.contract.alpha_threshold,
                alpha,
                torch.zeros_like(alpha),
            )
            local_transmittance = transmittance[v_min:v_max, u_min:u_max]
            local_active = active_pixel[v_min:v_max, u_min:u_max]
            accepted = (
                local_active
                & (local_transmittance * (1.0 - alpha) >= self.contract.transmittance_threshold)
            )
            alpha = torch.where(accepted, alpha, torch.zeros_like(alpha))
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
            active_pixel = active_pixel.clone()
            active_pixel[v_min:v_max, u_min:u_max] = local_active & accepted
        accumulated = accumulated + transmittance[None] * background_tensor[:, None, None]
        depth = depth_accumulated / alpha_accumulated.clamp_min(torch.finfo(dtype).eps)
        normal = torch.nn.functional.normalize(normal_accumulated, dim=0, eps=torch.finfo(dtype).eps)
        normal = torch.where(alpha_accumulated[None] > 0, normal, torch.zeros_like(normal))
        return accumulated, alpha_accumulated[None], depth[None], normal


class CudaGaussianRenderer(nn.Module):
    """Differentiable TRELLIS mip-splatting adapter for native FP32 geometry."""

    def __init__(
        self,
        near: float = 0.01,
        far: float = 100.0,
        contract: RasterizationContract = RasterizationContract(),
    ) -> None:
        super().__init__()
        if not 0 < near < far:
            raise ValueError("projection near/far planes must satisfy 0 < near < far")
        self.near = near
        self.far = far
        self.contract = contract
        compiled = RasterizationContract(kernel_size=contract.kernel_size)
        for name in (
            "sigma_extent",
            "alpha_ceiling",
            "alpha_threshold",
            "transmittance_threshold",
            "determinant_epsilon",
            "visibility_near",
            "tile_size",
        ):
            if getattr(contract, name) != getattr(compiled, name):
                raise ValueError(
                    f"TRELLIS CUDA compiles RasterizationContract.{name}; it cannot be overridden"
                )

    @staticmethod
    def _projection(intrinsic: Tensor, height: int, width: int, near: float, far: float) -> Tensor:
        projection = torch.zeros(4, 4, dtype=intrinsic.dtype, device=intrinsic.device)
        projection[0, 0] = 2.0 * intrinsic[0, 0] / width
        projection[1, 1] = 2.0 * intrinsic[1, 1] / height
        # TRELLIS maps NDC with ((ndc + 1) * size - 1) / 2 and evaluates
        # integer pixel centers.  The +1 terms therefore preserve the OpenCV
        # projection u=fx*x/z+cx, v=fy*y/z+cy exactly, including off-axis K.
        projection[0, 2] = (2.0 * intrinsic[0, 2] + 1.0) / width - 1.0
        projection[1, 2] = (2.0 * intrinsic[1, 2] + 1.0) / height - 1.0
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
        if gaussian.means.device.type != "cuda":
            raise ValueError("CUDA rasterization requires a CUDA-resident Gaussian asset")
        if gaussian.means.device != cameras.intrinsics.device:
            raise ValueError("Gaussian assets and cameras must share a CUDA device")
        settings_fields = tuple(getattr(GaussianRasterizationSettings, "_fields", ()))
        required_fields = {"kernel_size", "subpixel_offset"}
        if not required_fields.issubset(settings_fields):
            raise RuntimeError(
                "the loaded diff_gaussian_rasterization extension is not the TRELLIS "
                "mip-splatting ABI (kernel_size/subpixel_offset are required)"
            )
        color_images, alpha_images, depth_images, normal_images = [], [], [], []
        gaussian.validate()
        # The CUDA extension is a native-FP32 kernel.  Explicit casts keep it
        # outside BF16 autocast while retaining gradients to analytical state.
        means = gaussian.means.to(dtype=torch.float32).contiguous()
        opacity = gaussian.opacity.to(dtype=torch.float32).contiguous()
        covariance = gaussian.covariance.to(dtype=torch.float32)
        covariance_packed = torch.stack(
            (
                covariance[:, 0, 0],
                covariance[:, 0, 1],
                covariance[:, 0, 2],
                covariance[:, 1, 1],
                covariance[:, 1, 2],
                covariance[:, 2, 2],
            ),
            dim=-1,
        ).contiguous()
        for view in range(cameras.extrinsics_world_to_camera.shape[0]):
            extrinsic = cameras.extrinsics_world_to_camera[view].to(dtype=torch.float32)
            intrinsic = cameras.intrinsics[view].to(dtype=torch.float32)
            view_matrix = torch.eye(4, dtype=extrinsic.dtype, device=extrinsic.device)
            view_matrix[:3] = extrinsic
            projection = self._projection(intrinsic, cameras.height, cameras.width, self.near, self.far)
            camera_center = -(extrinsic[:3, :3].transpose(0, 1) @ extrinsic[:3, 3])
            view_direction = torch.nn.functional.normalize(camera_center[None] - means, dim=-1)
            basis = real_sh_basis_degree3(view_direction)
            color = torch.clamp(
                0.5
                + torch.einsum(
                    "gi,gic->gc",
                    basis,
                    gaussian.sh_coefficients.to(dtype=torch.float32),
                ),
                0.0,
                1.0,
            ).contiguous()
            bg = _background_color(background, means)
            common_settings = dict(
                image_height=cameras.height,
                image_width=cameras.width,
                tanfovx=float(cameras.width / (2.0 * intrinsic[0, 0])),
                tanfovy=float(cameras.height / (2.0 * intrinsic[1, 1])),
                kernel_size=self.contract.kernel_size,
                subpixel_offset=torch.zeros(
                    cameras.height,
                    cameras.width,
                    2,
                    dtype=torch.float32,
                    device=means.device,
                ),
                scale_modifier=1.0,
                viewmatrix=view_matrix.transpose(0, 1).contiguous(),
                projmatrix=(projection @ view_matrix).transpose(0, 1).contiguous(),
                sh_degree=0,
                campos=camera_center,
                prefiltered=False,
                debug=False,
            )
            color_rasterizer = GaussianRasterizer(
                raster_settings=GaussianRasterizationSettings(bg=bg, **common_settings)
            )
            auxiliary_rasterizer = GaussianRasterizer(
                raster_settings=GaussianRasterizationSettings(
                    bg=torch.zeros(3, dtype=torch.float32, device=means.device),
                    **common_settings,
                )
            )
            screen = torch.zeros_like(means, requires_grad=True)

            def rasterize(rasterizer: nn.Module, precomputed_color: Tensor) -> Tensor:
                rendered, _ = rasterizer(
                    means3D=means,
                    means2D=screen,
                    shs=None,
                    colors_precomp=precomputed_color,
                    opacities=opacity,
                    scales=None,
                    rotations=None,
                    cov3D_precomp=covariance_packed,
                )
                return rendered

            rendered_color = rasterize(color_rasterizer, color)
            camera_depth = (means @ extrinsic[:3, :3].transpose(0, 1) + extrinsic[:3, 3])[:, 2]
            alpha_depth_features = torch.stack(
                (torch.ones_like(camera_depth), camera_depth, torch.zeros_like(camera_depth)),
                dim=-1,
            )
            alpha_depth = rasterize(auxiliary_rasterizer, alpha_depth_features)
            alpha = alpha_depth[0:1].clamp(0.0, 1.0)
            depth = alpha_depth[1:2] / alpha.clamp_min(1.0e-8)
            normal_camera = (
                gaussian.rotation[:, :, 2].to(dtype=torch.float32)
                @ extrinsic[:3, :3].transpose(0, 1)
            ).contiguous()
            normal_rgb = rasterize(auxiliary_rasterizer, normal_camera)
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


__all__ = [
    "CameraBatch",
    "CudaGaussianRenderer",
    "RasterizationContract",
    "ReferenceGaussianRenderer",
    "RenderResult",
]

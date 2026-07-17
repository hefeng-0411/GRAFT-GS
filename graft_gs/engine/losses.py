"""Traceable multilevel objectives for staged GRAFT-GS training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import log, pi
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from ..equivariant.gsta import IrrepTensor, l2_to_matrix, matrix_to_l2
from ..integration.pipeline import GraftGS, GraftGSOutput, SceneOutput
from ..manifold.barrier import triangle_distance_squared
from ..manifold.geometry import (
    geodesic_interpolate,
    ManifoldState,
    ManifoldTangent,
    product_metric_squared,
    retract,
    so3_log,
    spd_parallel_transport,
)
from .supervision import derive_feasible_surface_target


@dataclass(frozen=True)
class LossWeights:
    transport: float = 1.0
    surface: float = 5.0
    surface_uncertainty_nll: float = 0.1
    confidence_brier: float = 1.0
    render: float = 10.0
    ssim: float = 2.0
    perceptual: float = 1.0
    mask: float = 1.0
    mesh_depth: float = 1.0
    mesh_normal: float = 0.2
    vggt_depth_reprojection: float = 0.2
    vggt_track_cycle: float = 0.1
    vggt_depth_normal: float = 0.1
    camera_center: float = 1.0
    camera_rotation: float = 0.5
    camera_intrinsics: float = 0.25
    immersion: float = 0.1
    atlas_c0: float = 0.5
    atlas_c1: float = 0.2
    atlas_curvature: float = 0.05
    atlas_multilevel: float = 0.1
    topology_prior: float = 0.1
    topology_supervision: float = 1.0
    dino_relational: float = 0.05
    trellis_latent_relational: float = 0.05
    sheet: float = 0.05
    opacity: float = 0.01
    tile_opacity: float = 0.1
    metric_spd: float = 0.01
    feasibility: float = 1.0
    flow: float = 1.0
    distill_state: float = 1.0
    distill_transport: float = 0.1
    distill_topology: float = 0.1
    distill_render: float = 1.0
    distill_field: float = 0.5
    distill_activation: float = 0.25
    distill_jacobian: float = 0.1


@dataclass(frozen=True)
class ViewConditionedObjectives:
    """Per-view objectives and reliability for robust Phase-F gradients.

    ``objective[b,k]`` is a locally normalized rendering/observation loss, not
    a fabricated decomposition of the global scalar loss.  Phase F removes the
    original global view terms and replaces their gradient by the robust
    consensus of these local objectives.  ``artifact_delta`` changes only the
    color/background target while holding geometry and visibility fixed.
    """

    objective: Tensor
    artifact_delta: Tensor
    reliability: Tensor


class LearnedPerceptualPyramid(nn.Module):
    """Frozen, hash-pinned VGG16 feature metric with no network download path."""

    feature_layers = (3, 8, 15, 22)

    def __init__(self, features: nn.Sequential, checkpoint_sha256: str) -> None:
        super().__init__()
        self.features = features.eval()
        self.checkpoint_sha256 = checkpoint_sha256
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, expected_sha256: Optional[str] = None
    ) -> "LearnedPerceptualPyramid":
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("learned perceptual checkpoint SHA-256 differs")
        try:
            from torchvision.models import vgg16
        except ImportError as error:
            raise ImportError("learned perceptual loss requires declared torchvision") from error
        model = vgg16(weights=None)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, Mapping):
            raise TypeError("VGG16 perceptual checkpoint must contain a state dictionary")
        state = {}
        for name, value in payload.items():
            key = str(name)
            for prefix in ("module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            state[key] = value
        incompatible = model.load_state_dict(state, strict=False)
        missing_features = [
            key for key in incompatible.missing_keys if key.startswith("features.")
        ]
        unexpected_features = [
            key for key in incompatible.unexpected_keys if key.startswith("features.")
        ]
        if missing_features or unexpected_features:
            raise ValueError(
                "perceptual checkpoint is not a compatible torchvision VGG16 feature state"
            )
        return cls(model.features, digest)

    def forward(
        self, predicted: Tensor, target: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        if predicted.shape != target.shape or predicted.ndim != 5:
            raise ValueError("learned perceptual inputs must share [B,K,3,H,W]")
        batch, views, channels, height, width = predicted.shape
        left = predicted.reshape(batch * views, channels, height, width)
        right = target.reshape(batch * views, channels, height, width)
        left = (left - self.image_mean) / self.image_std
        right = (right - self.image_mean) / self.image_std
        weight = (
            mask.reshape(batch * views, 1, height, width)
            if mask is not None
            else None
        )
        losses = []
        for index, layer in enumerate(self.features):
            left = layer(left)
            with torch.no_grad():
                right = layer(right)
            if index not in self.feature_layers:
                continue
            residual = torch.nn.functional.smooth_l1_loss(
                left, right.detach(), reduction="none"
            ).mean(dim=1, keepdim=True)
            if weight is None:
                losses.append(residual.mean())
            else:
                resized = torch.nn.functional.interpolate(
                    weight, size=residual.shape[-2:], mode="area"
                )
                losses.append(
                    torch.sum(residual * resized) / resized.sum().clamp_min(1.0)
                )
            if index >= self.feature_layers[-1]:
                break
        if len(losses) != len(self.feature_layers):
            raise RuntimeError("VGG16 feature pyramid terminated before required layers")
        return torch.stack(losses).mean()


def _view_supervision_masks(
    batch: Mapping[str, object], predicted: Tensor
) -> tuple[Tensor, Optional[Tensor], Tensor]:
    batch_size, view_count, _, height, width = predicted.shape
    availability_value = batch.get("valid_mask")
    if availability_value is None:
        availability = torch.ones(
            batch_size, view_count, dtype=torch.bool, device=predicted.device
        )
        spatial_validity = None
    else:
        validity = torch.as_tensor(
            availability_value, device=predicted.device, dtype=torch.bool
        )
        if validity.shape == (batch_size, view_count):
            availability = validity
            spatial_validity = None
        else:
            if validity.ndim == 4:
                validity = validity[:, :, None]
            expected = (batch_size, view_count, 1, height, width)
            if validity.shape != expected:
                raise ValueError(
                    "valid_mask must have shape [B,K] or [B,K,1,H,W]"
                )
            spatial_validity = validity
            availability = torch.any(validity, dim=(-3, -2, -1))
    alpha_value = batch.get("alpha")
    alpha: Optional[Tensor]
    if alpha_value is None:
        alpha = None
    else:
        alpha = torch.as_tensor(
            alpha_value, device=predicted.device, dtype=predicted.dtype
        )
        if alpha.ndim == 4:
            alpha = alpha[:, :, None]
        expected = (batch_size, view_count, 1, height, width)
        if alpha.shape != expected:
            raise ValueError(f"foreground alpha must have shape {expected}")
        alpha = alpha.clamp(0.0, 1.0)
    available_pixels = availability[:, :, None, None, None].to(predicted.dtype)
    supervision_mask = available_pixels.expand(
        batch_size, view_count, 1, height, width
    )
    if spatial_validity is not None:
        supervision_mask = supervision_mask * spatial_validity.to(predicted.dtype)
    if alpha is not None:
        supervision_mask = supervision_mask * alpha
    return availability, alpha, supervision_mask


def _canonical_vggt_depth(output: GraftGSOutput) -> Tensor:
    depth = output.vggt.depth
    if depth.ndim == 5:
        if depth.shape[-1] != 1:
            raise ValueError("VGGT depth trailing channel must be one")
        return depth[..., 0]
    if depth.ndim == 4:
        return depth
    raise ValueError("VGGT depth must have shape [B,K,H,W,(1)]")


def _calibrated_vggt_confidence(output: GraftGSOutput) -> Tensor:
    confidence = output.vggt.depth_confidence
    if confidence.ndim == 5:
        confidence = confidence[..., 0]
    if confidence.ndim != 4:
        raise ValueError("VGGT depth confidence must have shape [B,K,H,W,(1)]")
    confidence = confidence.clamp_min(0.0)
    return confidence / (1.0 + confidence)


def multiview_reprojection_cycle_loss(
    output: GraftGSOutput,
    valid_mask: Optional[Tensor] = None,
    sampling_stride: int = 16,
    visibility_log_depth_scale: float = 0.1,
) -> Tensor:
    """Differentiable depth/camera track cycle over deterministic sparse pixels.

    Source depth is unprojected to world, projected into a second view, and the
    bilinearly sampled target depth is unprojected back to world and source.
    This is derived VGGT self-supervision, not a fabricated external track head.
    """

    if sampling_stride < 1 or visibility_log_depth_scale <= 0:
        raise ValueError("track-cycle stride/visibility scale must be positive")
    depth = _canonical_vggt_depth(output)
    confidence = _calibrated_vggt_confidence(output)
    extrinsic = output.vggt.extrinsics_world_to_camera
    intrinsic = output.vggt.intrinsics
    batch_size, view_count, height, width = depth.shape
    if valid_mask is None:
        available = torch.ones(
            batch_size, view_count, dtype=torch.bool, device=depth.device
        )
        spatial_validity = None
    else:
        validity = torch.as_tensor(valid_mask, device=depth.device, dtype=torch.bool)
        if validity.shape == (batch_size, view_count):
            available = validity
            spatial_validity = None
        else:
            if validity.ndim == 4:
                validity = validity[:, :, None]
            if validity.shape != (batch_size, view_count, 1, height, width):
                raise ValueError(
                    "track valid_mask must have shape [B,K] or [B,K,1,H,W]"
                )
            available = torch.any(validity, dim=(-3, -2, -1))
            spatial_validity = validity
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(0, height, sampling_stride, device=depth.device, dtype=depth.dtype),
        torch.arange(0, width, sampling_stride, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    losses = []
    weights = []
    for batch_index in range(batch_size):
        for source in range(view_count):
            if not bool(available[batch_index, source]):
                continue
            source_depth = depth[
                batch_index, source, ::sampling_stride, ::sampling_stride
            ]
            source_confidence = confidence[
                batch_index, source, ::sampling_stride, ::sampling_stride
            ]
            source_validity = (
                spatial_validity[
                    batch_index,
                    source,
                    0,
                    ::sampling_stride,
                    ::sampling_stride,
                ]
                if spatial_validity is not None
                else torch.ones_like(source_depth, dtype=torch.bool)
            )
            source_intrinsic = intrinsic[batch_index, source]
            source_extrinsic = extrinsic[batch_index, source]
            source_camera = torch.stack(
                (
                    (pixel_x - source_intrinsic[0, 2])
                    / source_intrinsic[0, 0]
                    * source_depth,
                    (pixel_y - source_intrinsic[1, 2])
                    / source_intrinsic[1, 1]
                    * source_depth,
                    source_depth,
                ),
                dim=-1,
            )
            source_world = torch.einsum(
                "ij,hwj->hwi",
                source_extrinsic[:3, :3].transpose(0, 1),
                source_camera - source_extrinsic[:3, 3],
            )
            for target in range(view_count):
                if source == target or not bool(available[batch_index, target]):
                    continue
                target_extrinsic = extrinsic[batch_index, target]
                target_intrinsic = intrinsic[batch_index, target]
                target_camera = torch.einsum(
                    "ij,hwj->hwi", target_extrinsic[:3, :3], source_world
                ) + target_extrinsic[:3, 3]
                target_z = target_camera[..., 2]
                target_x = (
                    target_intrinsic[0, 0] * target_camera[..., 0] / target_z.clamp_min(1.0e-8)
                    + target_intrinsic[0, 2]
                )
                target_y = (
                    target_intrinsic[1, 1] * target_camera[..., 1] / target_z.clamp_min(1.0e-8)
                    + target_intrinsic[1, 2]
                )
                grid = torch.stack(
                    (
                        2.0 * (target_x + 0.5) / width - 1.0,
                        2.0 * (target_y + 0.5) / height - 1.0,
                    ),
                    dim=-1,
                )[None]
                sampled_depth = torch.nn.functional.grid_sample(
                    depth[batch_index, target][None, None],
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )[0, 0]
                sampled_confidence = torch.nn.functional.grid_sample(
                    confidence[batch_index, target][None, None],
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )[0, 0]
                sampled_validity = (
                    torch.nn.functional.grid_sample(
                        spatial_validity[batch_index, target].to(depth.dtype)[None],
                        grid,
                        mode="nearest",
                        padding_mode="zeros",
                        align_corners=False,
                    )[0, 0]
                    > 0.5
                    if spatial_validity is not None
                    else torch.ones_like(sampled_depth, dtype=torch.bool)
                )
                inside = (
                    (target_z > 0)
                    & (target_x >= 0)
                    & (target_x <= width - 1)
                    & (target_y >= 0)
                    & (target_y <= height - 1)
                    & (source_depth > 0)
                    & (sampled_depth > 0)
                    & torch.isfinite(source_depth)
                    & torch.isfinite(sampled_depth)
                    & source_validity
                    & sampled_validity
                )
                if not bool(torch.any(inside)):
                    continue
                sampled_camera = torch.stack(
                    (
                        (target_x - target_intrinsic[0, 2])
                        / target_intrinsic[0, 0]
                        * sampled_depth,
                        (target_y - target_intrinsic[1, 2])
                        / target_intrinsic[1, 1]
                        * sampled_depth,
                        sampled_depth,
                    ),
                    dim=-1,
                )
                sampled_world = torch.einsum(
                    "ij,hwj->hwi",
                    target_extrinsic[:3, :3].transpose(0, 1),
                    sampled_camera - target_extrinsic[:3, 3],
                )
                source_cycle_camera = torch.einsum(
                    "ij,hwj->hwi", source_extrinsic[:3, :3], sampled_world
                ) + source_extrinsic[:3, 3]
                cycle_x = (
                    source_intrinsic[0, 0]
                    * source_cycle_camera[..., 0]
                    / source_cycle_camera[..., 2].clamp_min(1.0e-8)
                    + source_intrinsic[0, 2]
                )
                cycle_y = (
                    source_intrinsic[1, 1]
                    * source_cycle_camera[..., 1]
                    / source_cycle_camera[..., 2].clamp_min(1.0e-8)
                    + source_intrinsic[1, 2]
                )
                log_depth_disagreement = torch.abs(
                    torch.log(sampled_depth.clamp_min(1.0e-8))
                    - torch.log(target_z.clamp_min(1.0e-8))
                )
                visibility = 1.0 / (
                    1.0 + (log_depth_disagreement / visibility_log_depth_scale).square()
                )
                weight = (
                    source_confidence
                    * sampled_confidence
                    * visibility
                    * inside.to(depth.dtype)
                )
                world_error = torch.linalg.vector_norm(
                    sampled_world - source_world, dim=-1
                ) / source_depth.clamp_min(1.0e-4)
                pixel_error = torch.sqrt(
                    (cycle_x - pixel_x).square() + (cycle_y - pixel_y).square()
                ) / float(max(height, width))
                losses.append(torch.sum((world_error + pixel_error) * weight))
                weights.append(weight.sum())
    if not losses:
        return depth.new_zeros(())
    return torch.stack(losses).sum() / torch.stack(weights).sum().clamp_min(1.0)


def vggt_depth_normal_field(output: GraftGSOutput) -> tuple[Tensor, Tensor, Tensor]:
    """Return confidence-weighted world normals derived from VGGT depth."""

    depth = _canonical_vggt_depth(output)
    confidence = _calibrated_vggt_confidence(output)
    batch_size, view_count, height, width = depth.shape
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    intrinsic = output.vggt.intrinsics
    extrinsic = output.vggt.extrinsics_world_to_camera
    camera = torch.stack(
        (
            (pixel_x[None, None] - intrinsic[..., 0, 2, None, None])
            / intrinsic[..., 0, 0, None, None]
            * depth,
            (pixel_y[None, None] - intrinsic[..., 1, 2, None, None])
            / intrinsic[..., 1, 1, None, None]
            * depth,
            depth,
        ),
        dim=-1,
    )
    world = torch.einsum(
        "bkij,bkhwj->bkhwi",
        extrinsic[..., :3, :3].transpose(-1, -2),
        camera - extrinsic[..., :3, 3][:, :, None, None, :],
    )
    tangent_x = world[:, :, :-1, 1:] - world[:, :, :-1, :-1]
    tangent_y = world[:, :, 1:, :-1] - world[:, :, :-1, :-1]
    normal = torch.linalg.cross(tangent_x, tangent_y, dim=-1)
    normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    valid = (
        torch.isfinite(normal_norm[..., 0])
        & (normal_norm[..., 0] > 1.0e-10)
        & (depth[:, :, :-1, :-1] > 0)
        & (depth[:, :, :-1, 1:] > 0)
        & (depth[:, :, 1:, :-1] > 0)
    )
    normal = normal / normal_norm.clamp_min(1.0e-10)
    normal = torch.nn.functional.pad(
        normal.permute(0, 1, 4, 2, 3).reshape(batch_size * view_count, 3, height - 1, width - 1),
        (0, 1, 0, 1),
        mode="replicate",
    ).reshape(batch_size, view_count, 3, height, width)
    valid = torch.nn.functional.pad(valid, (0, 1, 0, 1), value=False)[:, :, None]
    confidence = confidence[:, :, None]
    return normal, valid, confidence


def robust_rgb(
    predicted: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    epsilon: float = 1.0e-3,
) -> Tensor:
    residual = torch.sqrt(
        (predicted - target).square().sum(dim=2, keepdim=True) + epsilon**2
    )
    if mask is None:
        return residual.mean()
    if mask.shape != residual.shape:
        raise ValueError("RGB mask must have shape [B,K,1,H,W]")
    return torch.sum(residual * mask) / mask.sum().clamp_min(1.0)


def structural_similarity_loss(
    predicted: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Three-channel 3x3 SSIM dissimilarity on batched multiview images."""

    if predicted.shape != target.shape or predicted.ndim != 5:
        raise ValueError("SSIM inputs must share shape [B,K,3,H,W]")
    batch, views, channels, height, width = predicted.shape
    left = predicted.reshape(batch * views, channels, height, width)
    right = target.reshape_as(left)
    mean_left = torch.nn.functional.avg_pool2d(left, 3, stride=1, padding=1)
    mean_right = torch.nn.functional.avg_pool2d(right, 3, stride=1, padding=1)
    variance_left = torch.nn.functional.avg_pool2d(left.square(), 3, 1, 1) - mean_left.square()
    variance_right = torch.nn.functional.avg_pool2d(right.square(), 3, 1, 1) - mean_right.square()
    covariance = torch.nn.functional.avg_pool2d(left * right, 3, 1, 1) - mean_left * mean_right
    c1, c2 = 0.01**2, 0.03**2
    similarity = (
        (2.0 * mean_left * mean_right + c1)
        * (2.0 * covariance + c2)
        / (
            (mean_left.square() + mean_right.square() + c1)
            * (variance_left + variance_right + c2)
        ).clamp_min(1.0e-12)
    )
    loss = 0.5 * (1.0 - similarity.clamp(-1.0, 1.0)).mean(dim=1, keepdim=True)
    if mask is None:
        return loss.mean()
    weight = mask.reshape(batch * views, 1, height, width)
    return torch.sum(loss * weight) / weight.sum().clamp_min(1.0)


def multiscale_perceptual_loss(
    predicted: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    levels: int = 3,
) -> Tensor:
    """Fixed multiscale color/gradient feature metric.

    This deliberately avoids claiming a learned LPIPS checkpoint.  It supplies
    deterministic coarse structure and edge supervision at three scales and is
    valid in the high-precision reference path without another external model.
    """

    if predicted.shape != target.shape or predicted.ndim != 5:
        raise ValueError("perceptual inputs must share shape [B,K,3,H,W]")
    batch, views, channels, height, width = predicted.shape
    left = predicted.reshape(batch * views, channels, height, width)
    right = target.reshape_as(left)
    weight = (
        mask.reshape(batch * views, 1, height, width)
        if mask is not None
        else None
    )
    losses = []
    for _ in range(levels):
        color_error = (left - right).abs().mean(dim=1, keepdim=True)
        gradient_x = (
            (left[..., :, 1:] - left[..., :, :-1])
            - (right[..., :, 1:] - right[..., :, :-1])
        ).abs().mean(dim=1, keepdim=True)
        gradient_y = (
            (left[..., 1:, :] - left[..., :-1, :])
            - (right[..., 1:, :] - right[..., :-1, :])
        ).abs().mean(dim=1, keepdim=True)
        if weight is None:
            level_loss = color_error.mean() + 0.5 * (
                gradient_x.mean() + gradient_y.mean()
            )
        else:
            level_loss = (
                torch.sum(color_error * weight) / weight.sum().clamp_min(1.0)
                + 0.5
                * torch.sum(gradient_x * weight[..., :, 1:])
                / weight[..., :, 1:].sum().clamp_min(1.0)
                + 0.5
                * torch.sum(gradient_y * weight[..., 1:, :])
                / weight[..., 1:, :].sum().clamp_min(1.0)
            )
        losses.append(level_loss)
        if min(left.shape[-2:]) < 4:
            break
        left = torch.nn.functional.avg_pool2d(left, 2, stride=2)
        right = torch.nn.functional.avg_pool2d(right, 2, stride=2)
        if weight is not None:
            weight = torch.nn.functional.avg_pool2d(weight, 2, stride=2)
    return torch.stack(losses).mean()


def conservative_tile_opacity_bound(
    scene: SceneOutput,
    tile_size: int,
    sigma_extent: float = 3.5,
    covariance_floor_pixels: float = 0.25,
) -> Tensor:
    r"""Conservative opacity upper bound for every rendered image tile.

    A Gaussian footprint is at most one, so its peak optical depth
    ``tau_g=-log(1-alpha_g)`` bounds its contribution at every covered pixel.
    Summing peak depths for all 3-sigma boxes overlapping a tile gives
    ``alpha_tile <= 1-exp(-sum_g tau_g)``. Box membership is discrete; the
    retained optical-depth values remain differentiable.
    """

    if scene.gaussians is None or scene.render_cameras is None:
        raise ValueError("tile opacity bounds require Gaussians and render cameras")
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    if covariance_floor_pixels < 0:
        raise ValueError("pixel covariance floor must be non-negative")
    gaussian = scene.gaussians
    cameras = scene.render_cameras
    tile_height = (cameras.height + tile_size - 1) // tile_size
    tile_width = (cameras.width + tile_size - 1) // tile_size
    optical_depth = -torch.log1p(-gaussian.opacity[:, 0].clamp_max(1.0 - 1.0e-8))
    bounds = []
    for view in range(cameras.extrinsics_world_to_camera.shape[0]):
        extrinsic = cameras.extrinsics_world_to_camera[view]
        intrinsic = cameras.intrinsics[view]
        rotation = extrinsic[:3, :3]
        camera_point = gaussian.means @ rotation.transpose(0, 1) + extrinsic[:3, 3]
        x, y, z = camera_point.unbind(-1)
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        pixel_x = fx * x / z + intrinsic[0, 2]
        pixel_y = fy * y / z + intrinsic[1, 2]
        projection_camera = torch.stack(
            (
                torch.stack((fx / z, torch.zeros_like(z), -fx * x / z.square()), dim=-1),
                torch.stack((torch.zeros_like(z), fy / z, -fy * y / z.square()), dim=-1),
            ),
            dim=-2,
        )
        projection = projection_camera @ rotation
        covariance_2d = projection @ gaussian.covariance @ projection.transpose(-1, -2)
        covariance_2d = covariance_2d + covariance_floor_pixels * torch.eye(
            2,
            dtype=covariance_2d.dtype,
            device=covariance_2d.device,
        )
        radius = sigma_extent * torch.sqrt(
            torch.linalg.eigvalsh(covariance_2d).clamp_min(0.0)[:, -1]
        )
        tile_indices: list[Tensor] = []
        gaussian_indices: list[Tensor] = []
        for index in range(gaussian.means.shape[0]):
            visible = bool(torch.isfinite(z[index]).detach()) and float(z[index].detach()) > 0
            if not visible:
                continue
            minimum_x = max(
                0,
                int(torch.floor((pixel_x[index] - radius[index]) / tile_size).detach()),
            )
            maximum_x = min(
                tile_width - 1,
                int(torch.floor((pixel_x[index] + radius[index]) / tile_size).detach()),
            )
            minimum_y = max(
                0,
                int(torch.floor((pixel_y[index] - radius[index]) / tile_size).detach()),
            )
            maximum_y = min(
                tile_height - 1,
                int(torch.floor((pixel_y[index] + radius[index]) / tile_size).detach()),
            )
            if minimum_x > maximum_x or minimum_y > maximum_y:
                continue
            y_index, x_index = torch.meshgrid(
                torch.arange(minimum_y, maximum_y + 1, device=z.device),
                torch.arange(minimum_x, maximum_x + 1, device=z.device),
                indexing="ij",
            )
            flat_tile = (y_index * tile_width + x_index).reshape(-1)
            tile_indices.append(flat_tile)
            gaussian_indices.append(torch.full_like(flat_tile, index))
        tile_depth = optical_depth.new_zeros(tile_height * tile_width)
        if tile_indices:
            tile_index = torch.cat(tile_indices)
            gaussian_index = torch.cat(gaussian_indices)
            tile_depth = tile_depth.index_add(
                0,
                tile_index,
                optical_depth[gaussian_index],
            )
        tile_depth = tile_depth.reshape(tile_height, tile_width)
        bounds.append(-torch.expm1(-tile_depth))
    return torch.stack(bounds)


def view_conditioned_objectives(
    model: GraftGS,
    output: GraftGSOutput,
    batch: Mapping[str, object],
    weights: LossWeights,
    learned_perceptual: Optional[LearnedPerceptualPyramid] = None,
) -> ViewConditionedObjectives:
    """Build locally normalized multiview objectives for gradient purification.

    Global atlas, transport, topology, and barrier terms deliberately do not
    enter this function.  The artifact direction is the change in the
    appearance/mask gradient under a deterministic luminance perturbation and
    a one-pixel soft segmentation-boundary perturbation.  Geometry, cameras,
    depth, normals, and the rendered prediction are held fixed.
    """

    if not output.scenes or any(scene.render is None for scene in output.scenes):
        raise ValueError("view-conditioned objectives require rendered scenes")
    predicted = torch.stack([scene.render.color for scene in output.scenes])
    target = output.vggt.images
    if predicted.shape != target.shape:
        raise ValueError("rendered and target images must share [B,K,3,H,W]")
    availability, alpha, supervision_mask = _view_supervision_masks(batch, predicted)
    batch_size, view_count = predicted.shape[:2]
    predicted_alpha = torch.stack([scene.render.alpha for scene in output.scenes])
    predicted_depth = torch.stack([scene.render.depth for scene in output.scenes])
    predicted_normal = torch.stack([scene.render.normal for scene in output.scenes])

    vggt_depth = output.vggt.depth
    if vggt_depth.ndim == 5:
        vggt_depth = vggt_depth.permute(0, 1, 4, 2, 3)
    elif vggt_depth.ndim == 4:
        vggt_depth = vggt_depth[:, :, None]
    else:
        raise ValueError("VGGT depth must have shape [B,K,H,W,(1)]")
    raw_confidence = output.vggt.depth_confidence
    if raw_confidence.ndim == 5:
        raw_confidence = raw_confidence.squeeze(-1)
    if raw_confidence.shape != predicted.shape[:2] + predicted.shape[-2:]:
        raise ValueError("VGGT depth confidence must have shape [B,K,H,W]")
    confidence = raw_confidence.clamp_min(0.0) / (
        1.0 + raw_confidence.clamp_min(0.0)
    )
    vggt_normal, vggt_normal_valid, vggt_normal_confidence = (
        vggt_depth_normal_field(output)
    )

    mesh_depth_value = batch.get("mesh_depth_target")
    mesh_visibility_value = batch.get("mesh_visibility_mask")
    mesh_normal_value = batch.get("mesh_normal_target")
    mesh_normal_validity_value = batch.get("mesh_normal_validity")
    mesh_depth = (
        torch.as_tensor(mesh_depth_value, device=predicted.device, dtype=predicted.dtype)
        if mesh_depth_value is not None
        else None
    )
    mesh_visibility = (
        torch.as_tensor(mesh_visibility_value, device=predicted.device, dtype=torch.bool)
        if mesh_visibility_value is not None
        else None
    )
    mesh_normal = (
        torch.as_tensor(mesh_normal_value, device=predicted.device, dtype=predicted.dtype)
        if mesh_normal_value is not None
        else None
    )
    mesh_normal_validity = (
        torch.as_tensor(mesh_normal_validity_value, device=predicted.device, dtype=torch.bool)
        if mesh_normal_validity_value is not None
        else None
    )
    if (mesh_depth is None) != (mesh_visibility is None):
        raise ValueError("mesh depth objective requires both target and validity")
    if (mesh_normal is None) != (mesh_normal_validity is None):
        raise ValueError("mesh normal objective requires both target and validity")

    tile_bounds = [
        conservative_tile_opacity_bound(
            scene, tile_size=model.config.readout.opacity_tile_size
        )
        for scene in output.scenes
    ]
    objective_rows = []
    artifact_rows = []
    reliability_rows = []
    luminance_weights = predicted.new_tensor([0.2126, 0.7152, 0.0722]).view(
        1, 1, 3, 1, 1
    )
    for batch_index in range(batch_size):
        objective_views = []
        artifact_views = []
        reliability_views = []
        for view_index in range(view_count):
            prediction = predicted[batch_index : batch_index + 1, view_index : view_index + 1]
            reference = target[batch_index : batch_index + 1, view_index : view_index + 1]
            mask = supervision_mask[
                batch_index : batch_index + 1, view_index : view_index + 1
            ]
            available = availability[batch_index, view_index]
            appearance = (
                weights.render * robust_rgb(prediction, reference, mask)
                + weights.ssim * structural_similarity_loss(prediction, reference, mask)
                + weights.perceptual
                * (
                    learned_perceptual(prediction, reference, mask)
                    if learned_perceptual is not None
                    else multiscale_perceptual_loss(prediction, reference, mask)
                )
            )
            value = appearance

            gray = torch.sum(reference * luminance_weights, dim=2, keepdim=True)
            augmented_reference = 0.7 * gray.expand_as(reference) + 0.3 * (1.0 - reference)
            artifact_alpha = None
            artifact_mask = mask
            if alpha is not None:
                alpha_view = alpha[
                    batch_index : batch_index + 1, view_index : view_index + 1
                ]
                artifact_alpha = torch.nn.functional.avg_pool2d(
                    alpha_view.reshape(1, 1, *alpha_view.shape[-2:]),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ).reshape_as(alpha_view)
                artifact_mask = (
                    artifact_alpha
                    * available.to(predicted.dtype).reshape(1, 1, 1, 1, 1)
                )
            augmented_appearance = (
                weights.render
                * robust_rgb(prediction, augmented_reference, artifact_mask)
                + weights.ssim
                * structural_similarity_loss(
                    prediction, augmented_reference, artifact_mask
                )
                + weights.perceptual
                * (
                    learned_perceptual(
                        prediction, augmented_reference, artifact_mask
                    )
                    if learned_perceptual is not None
                    else multiscale_perceptual_loss(
                        prediction, augmented_reference, artifact_mask
                    )
                )
            )
            artifact_value = augmented_appearance - appearance

            if alpha is not None:
                alpha_view = alpha[
                    batch_index : batch_index + 1, view_index : view_index + 1
                ]
                prediction_alpha = predicted_alpha[
                    batch_index : batch_index + 1, view_index : view_index + 1
                ].clamp(1.0e-6, 1.0 - 1.0e-6)
                mask_loss = torch.nn.functional.binary_cross_entropy(
                    prediction_alpha, alpha_view
                )
                value = value + weights.mask * mask_loss
                if artifact_alpha is not None:
                    artifact_value = artifact_value + weights.mask * (
                        torch.nn.functional.binary_cross_entropy(
                            prediction_alpha, artifact_alpha
                        )
                        - mask_loss
                    )

            depth_prediction = predicted_depth[
                batch_index, view_index
            ]
            depth_target = vggt_depth[batch_index, view_index]
            valid_depth = (
                torch.isfinite(depth_prediction)
                & torch.isfinite(depth_target)
                & (depth_prediction > 0)
                & (depth_target > 0)
                & (mask[0, 0] > 0)
            )
            if bool(torch.any(valid_depth)):
                depth_residual = torch.nn.functional.smooth_l1_loss(
                    torch.log(depth_prediction.clamp_min(1.0e-8)),
                    torch.log(depth_target.clamp_min(1.0e-8)),
                    reduction="none",
                )
                depth_weight = confidence[batch_index, view_index][None] * valid_depth
                value = value + weights.vggt_depth_reprojection * (
                    torch.sum(depth_residual * depth_weight)
                    / depth_weight.sum().clamp_min(1.0)
                )

            local_rendered_normal = torch.nn.functional.normalize(
                predicted_normal[batch_index, view_index], dim=0, eps=1.0e-8
            )
            local_normal_cosine = torch.sum(
                local_rendered_normal
                * vggt_normal[batch_index, view_index].detach(),
                dim=0,
                keepdim=True,
            ).abs()
            local_normal_mask = (
                vggt_normal_valid[batch_index, view_index]
                & (mask[0, 0] > 0)
            )
            local_normal_weight = (
                vggt_normal_confidence[batch_index, view_index].detach()
                * local_normal_mask.to(predicted.dtype)
            )
            value = value + weights.vggt_depth_normal * (
                torch.sum(
                    (1.0 - local_normal_cosine.clamp(0.0, 1.0))
                    * local_normal_weight
                )
                / local_normal_weight.sum().clamp_min(1.0)
            )

            tile_penalty = torch.relu(
                tile_bounds[batch_index][view_index]
                - model.config.readout.maximum_tile_opacity
            ).square().mean()
            value = value + weights.tile_opacity * tile_penalty

            if mesh_depth is not None and mesh_visibility is not None:
                valid_mesh_depth = (
                    mesh_visibility[batch_index, view_index]
                    & torch.isfinite(mesh_depth[batch_index, view_index])
                    & (mesh_depth[batch_index, view_index] > 0)
                    & available
                )
                if bool(torch.any(valid_mesh_depth)):
                    value = value + weights.mesh_depth * torch.nn.functional.smooth_l1_loss(
                        torch.log(depth_prediction.clamp_min(1.0e-8))[valid_mesh_depth],
                        torch.log(mesh_depth[batch_index, view_index].clamp_min(1.0e-8))[valid_mesh_depth],
                    )

            if mesh_normal is not None and mesh_normal_validity is not None:
                normal_prediction = torch.nn.functional.normalize(
                    predicted_normal[batch_index, view_index], dim=0, eps=1.0e-8
                )
                normal_target = torch.nn.functional.normalize(
                    mesh_normal[batch_index, view_index], dim=0, eps=1.0e-8
                )
                cosine = torch.sum(normal_prediction * normal_target, dim=0, keepdim=True).abs()
                valid_normal = (
                    mesh_normal_validity[batch_index, view_index]
                    & torch.isfinite(cosine)
                    & available
                )
                if bool(torch.any(valid_normal)):
                    value = value + weights.mesh_normal * (
                        1.0 - cosine.clamp(0.0, 1.0)
                    )[valid_normal].mean()

            confidence_mask = mask[0, 0, 0]
            reliability = (
                torch.sum(confidence[batch_index, view_index] * confidence_mask)
                / confidence_mask.sum().clamp_min(1.0)
            )
            reliability = reliability * available.to(reliability.dtype)
            objective_views.append(value * available.to(value.dtype))
            artifact_views.append(artifact_value * available.to(value.dtype))
            reliability_views.append(reliability.detach())
        objective_rows.append(torch.stack(objective_views))
        artifact_rows.append(torch.stack(artifact_views))
        reliability_rows.append(torch.stack(reliability_views))
    return ViewConditionedObjectives(
        objective=torch.stack(objective_rows),
        artifact_delta=torch.stack(artifact_rows),
        reliability=torch.stack(reliability_rows),
    )


def symmetric_surface_chamfer(predicted: Tensor, target: Tensor, chunk_size: int = 2048) -> Tensor:
    """Memory-bounded symmetric squared Chamfer distance between surface measures."""

    if predicted.ndim != 2 or target.ndim != 2 or predicted.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("surface point sets must have shapes [G,3] and [N,3]")
    if predicted.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("surface Chamfer distance is undefined for an empty point set")
    target_minimum = target.new_full((target.shape[0],), torch.inf)
    predicted_minimum: list[Tensor] = []
    for start in range(0, predicted.shape[0], chunk_size):
        distance = torch.cdist(predicted[start : start + chunk_size], target).square()
        predicted_minimum.append(distance.amin(dim=1))
        target_minimum = torch.minimum(target_minimum, distance.amin(dim=0))
    return torch.cat(predicted_minimum).mean() + target_minimum.mean()


def nearest_surface_targets(query: Tensor, target: Tensor, chunk_size: int = 2048) -> Tensor:
    """Piecewise-differentiable exact nearest samples without a dense global matrix."""

    if query.ndim != 2 or target.ndim != 2 or query.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("surface point sets must have shapes [M,3] and [N,3]")
    if query.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("nearest surface assignment is undefined for an empty point set")
    nearest: list[Tensor] = []
    for start in range(0, query.shape[0], chunk_size):
        distance = torch.cdist(query[start : start + chunk_size], target)
        nearest.append(target[distance.argmin(dim=1)])
    return torch.cat(nearest, dim=0)


def evidence_surface_calibration(
    positions: Tensor,
    covariance: Tensor,
    confidence: Tensor,
    surface: Tensor,
    cell_size: float,
) -> tuple[Tensor, Tensor]:
    r"""Quantization-aware Gaussian NLL and confidence Brier score.

    A surface voxel represents an unknown point inside a cell.  Uniform cell
    quantization contributes ``h^2/12 I`` to learned evidence covariance,
    preventing the likelihood from rewarding a covariance collapse onto cell
    centers.  The confidence target is a smooth inlier probability at the
    cell's half-diagonal scale.
    """

    if cell_size <= 0:
        raise ValueError("surface cell size must be positive")
    nearest = nearest_surface_targets(positions, surface)
    residual = positions - nearest
    quantization_variance = positions.new_tensor(cell_size**2 / 12.0)
    eye = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
    observation_covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    observation_covariance = observation_covariance + quantization_variance * eye
    cholesky = torch.linalg.cholesky(observation_covariance)
    whitened = torch.cholesky_solve(residual[..., None], cholesky)[..., 0]
    mahalanobis = torch.sum(residual * whitened, dim=-1)
    log_determinant = 2.0 * torch.log(
        cholesky.diagonal(dim1=-2, dim2=-1)
    ).sum(-1)
    nll = 0.5 * (mahalanobis + log_determinant + 3.0 * log(2.0 * pi))
    half_diagonal = positions.new_tensor(0.5 * (3.0**0.5) * cell_size)
    inlier_probability = torch.exp(
        -0.5 * residual.square().sum(-1) / half_diagonal.square()
    ).detach()
    brier = (confidence - inlier_probability).square()
    return nll.mean(), brier.mean()


def atlas_overlap_consistency(scene: SceneOutput) -> tuple[Tensor, Tensor, Tensor]:
    r"""Gauge-invariant ``C0/C1`` and curvature agreement on chart overlaps.

    Each undirected atlas edge is evaluated at a deterministic midpoint in the
    source chart.  The same world point is expressed in the target tangent
    coordinates, defining the local transition approximation
    ``psi_ij = P_2 R_j^T (phi_i-c_j)``.  Consequently
    ``D psi_ij = P_2 R_j^T J_i`` and the C1 residual is
    ``J_i - J_j D psi_ij``.  Curvature is compared after lifting each local
    Hessian to a symmetric world-frame tangent tensor, so arbitrary in-plane
    gauge rotations and global SE(3) transformations do not change the loss.
    """

    atlas = scene.atlas
    if atlas.edge_index.numel() == 0:
        zero = scene.mapping.latent.new_zeros(())
        return zero, zero, zero
    source, target = atlas.edge_index
    undirected = source < target
    source, target = source[undirected], target[undirected]
    if source.numel() == 0:
        zero = scene.mapping.latent.new_zeros(())
        return zero, zero, zero
    frame_source = atlas.chart_frames[source]
    frame_target = atlas.chart_frames[target]
    center_delta = atlas.chart_centers[target] - atlas.chart_centers[source]
    delta_source = torch.einsum("eji,ej->ei", frame_source, center_delta)
    xi = 0.5 * delta_source[:, :2]
    world_source = atlas.evaluate_chart(source, xi)
    delta_target = torch.einsum(
        "eji,ej->ei",
        frame_target,
        world_source - atlas.chart_centers[target],
    )
    eta = delta_target[:, :2]
    world_target = atlas.evaluate_chart(target, eta)
    scale = 0.5 * (atlas.chart_radii[source] + atlas.chart_radii[target])
    c0 = ((world_source - world_target) / scale[:, None].clamp_min(1.0e-8)).square().sum(-1)

    jacobian_source = atlas.chart_jacobian(source, xi)
    jacobian_target = atlas.chart_jacobian(target, eta)
    transition_jacobian = (
        frame_target.transpose(-1, -2) @ jacobian_source
    )[:, :2, :]
    reconstructed_jacobian = jacobian_target @ transition_jacobian
    c1 = (
        (jacobian_source - reconstructed_jacobian).square().sum(dim=(-2, -1))
        / jacobian_source.square().sum(dim=(-2, -1)).clamp_min(1.0e-8)
    )

    tangent_source = frame_source[:, :, :2]
    tangent_target = frame_target[:, :, :2]
    world_curvature_source = (
        tangent_source @ atlas.curvature[source] @ tangent_source.transpose(-1, -2)
    )
    world_curvature_target = (
        tangent_target @ atlas.curvature[target] @ tangent_target.transpose(-1, -2)
    )
    curvature_scale = scale.square().clamp_min(1.0e-8)
    curvature = (
        (world_curvature_source - world_curvature_target)
        .square()
        .sum(dim=(-2, -1))
        * curvature_scale
    )
    return c0.mean(), c1.mean(), curvature.mean()


def atlas_multilevel_consistency(scene: SceneOutput) -> Tensor:
    """Persistent parent/child agreement in SE(3)-invariant quantities."""

    atlas = scene.atlas
    children = torch.nonzero(atlas.parent >= 0, as_tuple=False).flatten()
    if children.numel() == 0:
        return scene.mapping.latent.new_zeros(())
    losses = []
    for parent in torch.unique(atlas.parent[children]).tolist():
        child = children[atlas.parent[children] == parent]
        weight = atlas.evidence_mass[child] + atlas.prior_mass[child]
        if not bool(torch.any(weight > 0)):
            weight = torch.ones_like(weight)
        weight = weight / weight.sum().clamp_min(1.0e-12)
        child_center = torch.sum(
            weight[:, None] * atlas.chart_centers[child], dim=0
        )
        center_loss = (
            (atlas.chart_centers[parent] - child_center).square().sum()
            / atlas.cell_sides[parent].square().clamp_min(1.0e-12)
        )
        parent_normal = atlas.chart_frames[parent, :, 2]
        child_normal = atlas.chart_frames[child, :, 2]
        sign = torch.where(
            torch.sum(child_normal * parent_normal[None], dim=-1) >= 0,
            torch.ones_like(weight),
            -torch.ones_like(weight),
        ).detach()
        mean_normal = torch.nn.functional.normalize(
            torch.sum(weight[:, None] * sign[:, None] * child_normal, dim=0),
            dim=0,
            eps=1.0e-8,
        )
        normal_loss = 1.0 - torch.sum(parent_normal * mean_normal).abs().clamp_max(1.0)
        parent_tangent = atlas.chart_frames[parent, :, :2]
        parent_curvature = (
            parent_tangent
            @ atlas.curvature[parent]
            @ parent_tangent.transpose(-1, -2)
        )
        child_tangent = atlas.chart_frames[child, :, :2]
        child_curvature = (
            child_tangent
            @ atlas.curvature[child]
            @ child_tangent.transpose(-1, -2)
        )
        mean_curvature = torch.sum(
            weight[:, None, None] * child_curvature, dim=0
        )
        curvature_loss = (
            (parent_curvature - mean_curvature).square().sum()
            * atlas.cell_sides[parent].square()
        )
        losses.append(center_loss + normal_loss + curvature_loss)
    return torch.stack(losses).mean()


def provenance_weighted_topology_supervision(
    output: GraftGSOutput,
    batch: Mapping[str, object],
) -> Tensor:
    """Expected Betti mismatch under an explicitly admissible label mask.

    The raw source mesh is never consulted here. A target can enter only after
    the dataset contract marks it as validated/repaired, assigns nonzero
    confidence, records provenance, and supplies a non-null Betti tuple.
    Persistence and stratum targets require separate explicit fields and are
    intentionally unavailable in the audited sample.
    """

    zero = output.vggt.images.new_zeros(())
    mask_value = batch.get("topology_betti_supervision_mask")
    if mask_value is None:
        # Backward-compatible generic mask is accepted only when no explicit
        # target-specific mask exists.
        mask_value = batch.get("topology_supervision_mask")
    if mask_value is None:
        return zero
    mask = torch.as_tensor(mask_value, device=zero.device, dtype=torch.bool).reshape(-1)
    if mask.numel() != len(output.scenes):
        raise ValueError("topology_betti_supervision_mask must have one value per scene")
    target_value = batch.get("topology_target_betti_z2")
    if not bool(torch.any(mask)):
        if target_value is not None:
            raise ValueError("an inadmissible topology label must not carry a Betti target")
        return zero
    if target_value is None:
        raise ValueError("admissible topology supervision requires topology_target_betti_z2")
    target = torch.as_tensor(target_value, device=zero.device, dtype=torch.int64)
    if target.ndim == 1:
        target = target[None]
    if tuple(target.shape) != (len(output.scenes), 3):
        raise ValueError("topology_target_betti_z2 must have shape [B,3]")
    confidence_value = batch.get("topology_supervision_confidence")
    if confidence_value is None:
        raise ValueError("admissible topology supervision requires an explicit confidence")
    confidence = torch.as_tensor(
        confidence_value,
        device=zero.device,
        dtype=output.vggt.depth.dtype,
    ).reshape(-1)
    if confidence.shape != mask.shape or bool(torch.any((confidence[mask] <= 0) | (confidence[mask] > 1))):
        raise ValueError("admissible topology confidence must have shape [B] and lie in (0,1]")
    provenance = batch.get("topology_label_provenance")
    provenance_values = [provenance] if isinstance(provenance, str) else list(provenance or ())
    if len(provenance_values) != len(output.scenes):
        raise ValueError("topology label provenance must have one entry per scene")
    allowed = {"validated_raw_source_connectivity", "validated_repaired_topology"}
    for index in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist():
        if provenance_values[index] not in allowed:
            raise ValueError(
                f"hard topology label provenance {provenance_values[index]!r} is not admissible"
            )
    scene_loss = []
    for index, scene in enumerate(output.scenes):
        if not bool(mask[index]):
            scene_loss.append(zero)
            continue
        mismatch = output.vggt.depth.new_tensor(
            [
                sum(abs(int(candidate.betti[dimension]) - int(target[index, dimension])) for dimension in range(3))
                for candidate in scene.topology.candidates
            ]
        )
        scene_loss.append(confidence[index] * torch.sum(scene.topology.probability * mismatch))
    active_count = mask.to(output.vggt.depth.dtype).sum().clamp_min(1.0)
    return torch.stack(scene_loss).sum() / active_count


def surface_pseudo_relational_distillation(
    scene: SceneOutput,
    batch: Mapping[str, object],
) -> tuple[Tensor, Tensor]:
    """Match edgewise representation geometry, never teacher channel bases.

    DINOv2 patch tokens and TRELLIS structured latents live in unrelated
    learned coordinate systems. Direct regression or concatenation is not
    mathematically meaningful. We assign their verified canonical surface
    cells to persistent charts, average within each chart, and match cosine
    similarity on atlas overlap edges against the gauge-invariant scalar field.
    """

    zero = scene.mapping.latent.new_zeros(())
    if (
        batch.get("trellis_patchtokens") is None
        and batch.get("trellis_latent_features") is None
    ):
        return zero, zero
    surface_value = batch.get("surface_voxel_centers")
    if surface_value is None:
        return zero, zero
    surface = torch.as_tensor(
        surface_value,
        device=scene.mapping.latent.device,
        dtype=scene.mapping.latent.dtype,
    )
    if surface.ndim == 3 and surface.shape[0] == 1:
        surface = surface[0]
    if surface.ndim != 2 or surface.shape[1] != 3:
        raise ValueError("pseudo relational surface must have shape [N,3]")
    assignment = scene.atlas.assign_points(surface)
    global_to_local = torch.full(
        (scene.atlas.num_nodes,),
        -1,
        dtype=torch.int64,
        device=surface.device,
    )
    global_to_local[scene.mapping.graph.atlas_node_index] = torch.arange(
        scene.mapping.graph.source_count, device=surface.device
    )
    chart = global_to_local[assignment]
    valid_surface = chart >= 0
    chart = chart[valid_surface]
    edge_source = global_to_local[scene.atlas.edge_index[0]]
    edge_target = global_to_local[scene.atlas.edge_index[1]]
    valid_edge = (edge_source >= 0) & (edge_target >= 0)
    edge_source, edge_target = edge_source[valid_edge], edge_target[valid_edge]
    if edge_source.numel() == 0:
        return zero, zero
    scalar = scene.mapping.latent[:, :60]
    scalar = torch.nn.functional.normalize(scalar, dim=-1, eps=1.0e-8)
    predicted_similarity = torch.sum(
        scalar[edge_source] * scalar[edge_target], dim=-1
    )

    def relational_loss(
        feature_name: str,
        coordinate_name: str,
        mask_name: str,
        confidence_name: str,
        provenance_name: str,
        allowed_provenance: str,
    ) -> Tensor:
        feature_value = batch.get(feature_name)
        mask_value = batch.get(mask_name)
        if mask_value is not None:
            mask_tensor = torch.as_tensor(
                mask_value, device=surface.device, dtype=torch.bool
            ).reshape(-1)
            if mask_tensor.numel() != 1:
                raise ValueError(f"{mask_name} must contain one object-level value")
            mask = bool(mask_tensor[0])
        else:
            mask = feature_value is not None
        if feature_value is None:
            if mask:
                raise ValueError(f"{mask_name} is true but {feature_name} is unavailable")
            return zero
        if not mask:
            raise ValueError(f"{feature_name} is present while {mask_name} is false")
        provenance = batch.get(provenance_name)
        if provenance != allowed_provenance:
            raise ValueError(f"{feature_name} has missing or inadmissible pseudo-label provenance")
        confidence_value = batch.get(confidence_name)
        if confidence_value is None:
            raise ValueError(f"{feature_name} requires explicit pseudo-label confidence")
        confidence = torch.as_tensor(
            confidence_value,
            device=surface.device,
            dtype=scene.mapping.latent.dtype,
        ).reshape(-1)
        if confidence.numel() != 1 or not bool(0 <= confidence[0] <= 1):
            raise ValueError("pseudo-label confidence must be one scalar in [0,1]")
        if float(confidence[0]) == 0.0:
            return zero
        feature = torch.as_tensor(
            feature_value,
            device=surface.device,
            dtype=scene.mapping.latent.dtype,
        )
        if feature.ndim == 3 and feature.shape[0] == 1:
            feature = feature[0]
        if feature.ndim != 2 or feature.shape[0] != surface.shape[0]:
            raise ValueError(f"{feature_name} must have shape [N,D] aligned to surface cells")
        pseudo_coordinate = batch.get(coordinate_name)
        surface_coordinate = batch.get("surface_voxel_indices")
        if pseudo_coordinate is None or surface_coordinate is None:
            raise ValueError(f"{feature_name} requires explicit sparse-coordinate alignment")
        pseudo_coordinate = torch.as_tensor(pseudo_coordinate, device=surface.device)
        surface_coordinate = torch.as_tensor(surface_coordinate, device=surface.device)
        if pseudo_coordinate.ndim == 3 and pseudo_coordinate.shape[0] == 1:
            pseudo_coordinate = pseudo_coordinate[0]
        if surface_coordinate.ndim == 3 and surface_coordinate.shape[0] == 1:
            surface_coordinate = surface_coordinate[0]
        if not torch.equal(pseudo_coordinate.to(torch.int64), surface_coordinate.to(torch.int64)):
            raise ValueError(f"{feature_name} sparse coordinates differ from surface cells")
        feature = feature[valid_surface].detach()
        aggregate = feature.new_zeros((scene.mapping.graph.source_count, feature.shape[1]))
        aggregate.index_add_(0, chart, feature)
        count = feature.new_zeros((scene.mapping.graph.source_count,))
        count.index_add_(0, chart, torch.ones_like(chart, dtype=feature.dtype))
        aggregate = aggregate / count.clamp_min(1.0)[:, None]
        teacher = torch.nn.functional.normalize(aggregate, dim=-1, eps=1.0e-8)
        supervised_edge = (count[edge_source] > 0) & (count[edge_target] > 0)
        if not bool(torch.any(supervised_edge)):
            return zero
        teacher_similarity = torch.sum(
            teacher[edge_source] * teacher[edge_target], dim=-1
        )
        residual = torch.nn.functional.smooth_l1_loss(
            predicted_similarity[supervised_edge],
            teacher_similarity[supervised_edge],
            reduction="mean",
        )
        return confidence[0] * residual

    dino = relational_loss(
        "trellis_patchtokens",
        "trellis_feature_indices",
        "dino_pseudo_supervision_mask",
        "dino_pseudo_confidence",
        "dino_pseudo_provenance",
        "pretrained_dinov2_surface_feature",
    )
    trellis = relational_loss(
        "trellis_latent_features",
        "trellis_latent_coords",
        "trellis_latent_pseudo_supervision_mask",
        "trellis_latent_pseudo_confidence",
        "trellis_latent_pseudo_provenance",
        "pretrained_trellis_structured_latent_encoder",
    )
    return dino, trellis


def differentiable_feasibility_loss(
    scene: SceneOutput,
    barrier_config: object,
    relative_hardening_margin: float = 0.0,
    temperature: float = 0.1,
) -> Tensor:
    """Soft objective over dimensionless exact-certificate slack ratios.

    Area, squared separation, orientation cosine, and covariance eigenvalues
    have different physical units.  They must not be concatenated under one
    dimensional temperature.  Each value is normalized by the corresponding
    hard threshold; zero is the certified boundary and Phase F requests a
    positive relative safety margin.
    """

    if relative_hardening_margin < 0 or temperature <= 0:
        raise ValueError("hardening margin must be non-negative and temperature positive")

    state = scene.final_state
    faces = state.complex.faces
    current_vertices = state.position[faces]
    initial_vertices = scene.initial_state.position[faces]
    current_cross = torch.linalg.cross(
        current_vertices[:, 1] - current_vertices[:, 0],
        current_vertices[:, 2] - current_vertices[:, 0],
    )
    initial_cross = torch.linalg.cross(
        initial_vertices[:, 1] - initial_vertices[:, 0],
        initial_vertices[:, 2] - initial_vertices[:, 0],
    )
    area = 0.5 * torch.linalg.vector_norm(current_cross, dim=-1)
    area_margin = (
        area / barrier_config.minimum_face_area - 1.0
    )
    orientation_cosine = torch.sum(
        torch.nn.functional.normalize(current_cross, dim=-1)
        * torch.nn.functional.normalize(initial_cross, dim=-1),
        dim=-1,
    )
    orientation_margin = (
        orientation_cosine - barrier_config.minimum_orientation_cosine
    ) / (1.0 - barrier_config.minimum_orientation_cosine)
    covariance_eigenvalue = torch.linalg.eigvalsh(state.covariance)
    covariance_lower_margin = (
        covariance_eigenvalue[:, 0]
        / barrier_config.minimum_covariance_eigenvalue
        - 1.0
    )
    covariance_upper_margin = (
        1.0
        - covariance_eigenvalue[:, -1]
        / barrier_config.maximum_covariance_eigenvalue
    )
    separation_margin = state.position.new_empty(0)
    if scene.collision_pairs is not None and scene.collision_pairs.numel():
        pair = scene.collision_pairs
        delta = state.position[pair[:, 0]] - state.position[pair[:, 1]]
        separation_margin = (
            delta.square().sum(-1) / barrier_config.minimum_separation**2 - 1.0
        )
    triangle_margin = state.position.new_empty(0)
    if (
        scene.collision_face_pairs is not None
        and scene.collision_face_pairs.numel()
    ):
        pair = scene.collision_face_pairs
        left = state.position[faces[pair[:, 0]]]
        right = state.position[faces[pair[:, 1]]]
        triangle_margin = (
            triangle_distance_squared(left, right)
            / barrier_config.minimum_separation**2
            - 1.0
        )
    all_margin = torch.cat(
        (
            area_margin,
            orientation_margin,
            separation_margin,
            triangle_margin,
            covariance_lower_margin,
            covariance_upper_margin,
        )
    )
    return torch.nn.functional.softplus(
        (relative_hardening_margin - all_margin) / temperature
    ).mean()


class GraftGSLoss(nn.Module):
    def __init__(
        self,
        weights: LossWeights = LossWeights(),
        learned_perceptual: Optional[LearnedPerceptualPyramid] = None,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.learned_perceptual = learned_perceptual

    def forward(
        self,
        model: GraftGS,
        output: GraftGSOutput,
        batch: Mapping[str, object],
        phase: str,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        device = output.vggt.images.device
        if phase == "A":
            return self._evidence_calibration_loss(output, batch)
        if phase == "C":
            return self._flow_pretraining_loss(model, output, batch)
        if any(scene.gaussians is None or scene.mesh is None for scene in output.scenes):
            raise ValueError(
                f"phase {phase} requires analytical assets, got execution stage "
                f"{output.execution_stage!r}"
            )
        terms: dict[str, Tensor] = {}
        terms["transport"] = torch.stack([scene.mapping.diagnostics.objective for scene in output.scenes]).mean()
        zero = torch.zeros((), device=device)
        if output.camera_alignment is not None:
            terms["camera_center"] = output.camera_alignment.center_rmse.mean()
            terms["camera_rotation"] = output.camera_alignment.rotation_geodesic.mean()
            terms["camera_intrinsics"] = output.camera_alignment.intrinsic_log_focal_error.mean()
        else:
            terms["camera_center"] = zero
            terms["camera_rotation"] = zero
            terms["camera_intrinsics"] = zero
        terms["vggt_track_cycle"] = multiview_reprojection_cycle_loss(
            output,
            valid_mask=(
                torch.as_tensor(batch["valid_mask"], device=device)
                if batch.get("valid_mask") is not None
                else None
            ),
        )
        surface_target = batch.get("surface_voxel_centers")
        if surface_target is not None:
            surface = torch.as_tensor(surface_target, device=device, dtype=output.vggt.depth.dtype)
            if surface.ndim == 2:
                surface = surface[None]
            if surface.ndim != 3 or surface.shape[0] != len(output.scenes):
                raise ValueError("surface_voxel_centers must have shape [N,3] or [B,N,3]")
            terms["surface"] = torch.stack(
                [
                    symmetric_surface_chamfer(scene.gaussians.means, surface[index])
                    for index, scene in enumerate(output.scenes)
                ]
            ).mean()
            cell_size_value = batch.get("surface_cell_size", 1.0 / 64.0)
            if isinstance(cell_size_value, Tensor):
                unique_cell_size = torch.as_tensor(cell_size_value).reshape(-1)
                if unique_cell_size.numel() != 1:
                    raise ValueError("one object batch must provide one surface_cell_size")
                cell_size = float(unique_cell_size[0])
            else:
                cell_size = float(cell_size_value)
            calibration = [
                evidence_surface_calibration(
                    scene.evidence.positions,
                    scene.evidence.covariance,
                    scene.evidence.confidence,
                    surface[index],
                    cell_size,
                )
                for index, scene in enumerate(output.scenes)
            ]
            terms["surface_uncertainty_nll"] = torch.stack(
                [value[0] for value in calibration]
            ).mean()
            terms["confidence_brier"] = torch.stack(
                [value[1] for value in calibration]
            ).mean()
        else:
            terms["surface"] = zero
            terms["surface_uncertainty_nll"] = zero
            terms["confidence_brier"] = zero
        immersion_losses = []
        atlas_c0_losses = []
        atlas_c1_losses = []
        atlas_curvature_losses = []
        atlas_multilevel_losses = []
        topology_prior_losses = []
        sheet_losses = []
        opacity_losses = []
        spd_losses = []
        feasibility_losses = []
        barrier_config = model.config.barrier
        hardening_margin = float(batch.get("feasibility_relative_margin", 0.0))
        feasibility_temperature = float(
            batch.get("feasibility_relative_temperature", 0.1)
        )
        for scene in output.scenes:
            margin = scene.atlas.chart_immersion_margin()
            immersion_losses.append(-torch.log(margin.clamp_min(1.0e-8)).mean())
            c0, c1, curvature = atlas_overlap_consistency(scene)
            atlas_c0_losses.append(c0)
            atlas_c1_losses.append(c1)
            atlas_curvature_losses.append(curvature)
            atlas_multilevel_losses.append(atlas_multilevel_consistency(scene))
            energy = torch.stack([candidate.total_energy for candidate in scene.topology.candidates])
            topology_prior_losses.append(torch.sum(scene.topology.probability * energy))
            scales = scene.gaussians.scales
            sheet_losses.append((scales[:, 2] / torch.minimum(scales[:, 0], scales[:, 1]).clamp_min(1.0e-8)).square().mean())
            alpha = scene.gaussians.opacity
            opacity_losses.append(-(torch.log(alpha) + torch.log1p(-alpha)).mean())
            metric_eigenvalue = torch.linalg.eigvalsh(scene.mapping.riemannian_metric)
            spd_losses.append(torch.relu(1.0e-6 - metric_eigenvalue).square().mean())
            feasibility_losses.append(
                differentiable_feasibility_loss(
                    scene,
                    barrier_config,
                    relative_hardening_margin=hardening_margin,
                    temperature=feasibility_temperature,
                )
            )
        terms["immersion"] = torch.stack(immersion_losses).mean()
        terms["atlas_c0"] = torch.stack(atlas_c0_losses).mean()
        terms["atlas_c1"] = torch.stack(atlas_c1_losses).mean()
        terms["atlas_curvature"] = torch.stack(atlas_curvature_losses).mean()
        terms["atlas_multilevel"] = torch.stack(atlas_multilevel_losses).mean()
        terms["topology_prior"] = torch.stack(topology_prior_losses).mean()
        terms["topology_supervision"] = provenance_weighted_topology_supervision(
            output,
            batch,
        )
        if len(output.scenes) != 1 and (
            batch.get("trellis_patchtokens") is not None
            or batch.get("trellis_latent_features") is not None
        ):
            raise ValueError("variable-size pseudo-label distillation requires one object per rank")
        if len(output.scenes) == 1:
            dino_relational, trellis_relational = surface_pseudo_relational_distillation(
                output.scenes[0], batch
            )
        else:
            dino_relational, trellis_relational = zero, zero
        terms["dino_relational"] = dino_relational
        terms["trellis_latent_relational"] = trellis_relational
        terms["sheet"] = torch.stack(sheet_losses).mean()
        terms["opacity"] = torch.stack(opacity_losses).mean()
        terms["metric_spd"] = torch.stack(spd_losses).mean()
        terms["feasibility"] = torch.stack(feasibility_losses).mean()
        if all(scene.render is not None for scene in output.scenes):
            predicted = torch.stack([scene.render.color for scene in output.scenes])
            target = output.vggt.images
            view_availability, target_alpha, target_mask = _view_supervision_masks(
                batch, predicted
            )
            terms["render"] = robust_rgb(predicted, target, target_mask)
            terms["ssim"] = structural_similarity_loss(
                predicted,
                target,
                target_mask,
            )
            terms["perceptual"] = multiscale_perceptual_loss(
                predicted, target, target_mask
            ) if self.learned_perceptual is None else self.learned_perceptual(
                predicted, target, target_mask
            )
            tile_opacity_losses = []
            for scene in output.scenes:
                bound = conservative_tile_opacity_bound(
                    scene,
                    tile_size=model.config.readout.opacity_tile_size,
                )
                tile_opacity_losses.append(
                    torch.relu(
                        bound - model.config.readout.maximum_tile_opacity
                    ).square().mean()
                )
            terms["tile_opacity"] = torch.stack(tile_opacity_losses).mean()
            vggt_normal, vggt_normal_valid, vggt_normal_confidence = (
                vggt_depth_normal_field(output)
            )
            rendered_normal = torch.nn.functional.normalize(
                torch.stack([scene.render.normal for scene in output.scenes]),
                dim=2,
                eps=1.0e-8,
            )
            normal_cosine = torch.sum(
                rendered_normal * vggt_normal.detach(), dim=2, keepdim=True
            ).abs()
            vggt_normal_mask = vggt_normal_valid & (target_mask > 0)
            vggt_normal_weight = (
                vggt_normal_confidence.detach()
                * vggt_normal_mask.to(predicted.dtype)
            )
            terms["vggt_depth_normal"] = torch.sum(
                (1.0 - normal_cosine.clamp(0.0, 1.0)) * vggt_normal_weight
            ) / vggt_normal_weight.sum().clamp_min(1.0)
            predicted_depth = torch.stack(
                [scene.render.depth for scene in output.scenes]
            )
            vggt_depth = output.vggt.depth
            if vggt_depth.ndim == 5:
                vggt_depth = vggt_depth.permute(0, 1, 4, 2, 3)
            elif vggt_depth.ndim == 4:
                vggt_depth = vggt_depth[:, :, None]
            else:
                raise ValueError("VGGT depth must have shape [B,K,H,W,(1)]")
            raw_vggt_confidence = output.vggt.depth_confidence
            if raw_vggt_confidence.ndim == 5:
                raw_vggt_confidence = raw_vggt_confidence.squeeze(-1)
            raw_vggt_confidence = raw_vggt_confidence[:, :, None].clamp_min(0.0)
            vggt_confidence = raw_vggt_confidence / (
                1.0 + raw_vggt_confidence
            )
            if predicted_depth.shape != vggt_depth.shape:
                raise ValueError("VGGT and rendered depth tensors use incompatible grids")
            valid_reprojection = (
                torch.isfinite(predicted_depth)
                & torch.isfinite(vggt_depth)
                & (predicted_depth > 0)
                & (vggt_depth > 0)
            )
            if target_mask is not None:
                valid_reprojection &= target_mask > 0
            if bool(torch.any(valid_reprojection)):
                depth_residual = torch.nn.functional.smooth_l1_loss(
                    torch.log(predicted_depth.clamp_min(1.0e-8)),
                    torch.log(vggt_depth.clamp_min(1.0e-8)),
                    reduction="none",
                )
                confidence_weight = vggt_confidence.expand_as(
                    depth_residual
                )
                terms["vggt_depth_reprojection"] = torch.sum(
                    depth_residual
                    * confidence_weight
                    * valid_reprojection.to(depth_residual.dtype)
                ) / torch.sum(
                    confidence_weight
                    * valid_reprojection.to(depth_residual.dtype)
                ).clamp_min(1.0)
            else:
                terms["vggt_depth_reprojection"] = zero
            if target_alpha is not None:
                predicted_mask = torch.stack([scene.render.alpha for scene in output.scenes])
                if target_alpha.shape != predicted_mask.shape:
                    raise ValueError(
                        f"valid_mask must match rendered alpha {tuple(predicted_mask.shape)}, "
                        f"got {tuple(target_alpha.shape)}"
                    )
                mask_residual = torch.nn.functional.binary_cross_entropy(
                    predicted_mask.clamp(1.0e-6, 1.0 - 1.0e-6),
                    target_alpha,
                    reduction="none",
                )
                available_pixel = view_availability[:, :, None, None, None].to(
                    mask_residual.dtype
                )
                terms["mask"] = torch.sum(mask_residual * available_pixel) / (
                    available_pixel.sum() * mask_residual.shape[-2] * mask_residual.shape[-1]
                ).clamp_min(1.0)
            else:
                terms["mask"] = torch.zeros((), device=device)
            mesh_depth_target = batch.get("mesh_depth_target")
            mesh_visibility = batch.get("mesh_visibility_mask")
            if mesh_depth_target is not None or mesh_visibility is not None:
                if mesh_depth_target is None or mesh_visibility is None:
                    raise ValueError("mesh depth supervision requires both target and visibility mask")
                predicted_depth = torch.stack([scene.render.depth for scene in output.scenes])
                target_depth = torch.as_tensor(
                    mesh_depth_target, device=device, dtype=predicted_depth.dtype
                )
                depth_mask = torch.as_tensor(mesh_visibility, device=device, dtype=torch.bool)
                if target_depth.shape != predicted_depth.shape or depth_mask.shape != predicted_depth.shape:
                    raise ValueError("mesh depth target/mask must match rendered [B,K,1,H,W]")
                valid_depth = depth_mask & torch.isfinite(target_depth) & (target_depth > 0)
                if bool(torch.any(valid_depth)):
                    epsilon = torch.finfo(predicted_depth.dtype).eps
                    terms["mesh_depth"] = torch.nn.functional.smooth_l1_loss(
                        torch.log(predicted_depth.clamp_min(epsilon))[valid_depth],
                        torch.log(target_depth.clamp_min(epsilon))[valid_depth],
                    )
                else:
                    terms["mesh_depth"] = zero
            else:
                terms["mesh_depth"] = zero
            mesh_normal_target = batch.get("mesh_normal_target")
            mesh_normal_validity = batch.get("mesh_normal_validity")
            if mesh_normal_target is not None or mesh_normal_validity is not None:
                if mesh_normal_target is None or mesh_normal_validity is None:
                    raise ValueError("mesh normal supervision requires both target and validity mask")
                predicted_normal = torch.stack([scene.render.normal for scene in output.scenes])
                target_normal = torch.as_tensor(
                    mesh_normal_target, device=device, dtype=predicted_normal.dtype
                )
                normal_mask = torch.as_tensor(
                    mesh_normal_validity, device=device, dtype=torch.bool
                )
                if target_normal.shape != predicted_normal.shape:
                    raise ValueError("mesh normal target must match rendered [B,K,3,H,W]")
                if normal_mask.shape != predicted_normal.shape[:2] + (1,) + predicted_normal.shape[-2:]:
                    raise ValueError("mesh normal validity must have shape [B,K,1,H,W]")
                predicted_normal = torch.nn.functional.normalize(predicted_normal, dim=2, eps=1.0e-8)
                target_normal = torch.nn.functional.normalize(target_normal, dim=2, eps=1.0e-8)
                cosine = torch.sum(predicted_normal * target_normal, dim=2, keepdim=True).abs()
                valid_normal = normal_mask & torch.isfinite(cosine)
                terms["mesh_normal"] = (
                    (1.0 - cosine.clamp(0.0, 1.0))[valid_normal].mean()
                    if bool(torch.any(valid_normal))
                    else zero
                )
            else:
                terms["mesh_normal"] = zero
        else:
            terms["render"] = zero
            terms["ssim"] = zero
            terms["perceptual"] = zero
            terms["tile_opacity"] = zero
            terms["vggt_depth_reprojection"] = zero
            terms["vggt_depth_normal"] = zero
            terms["mask"] = zero
            terms["mesh_depth"] = zero
            terms["mesh_normal"] = zero
        terms["flow"] = zero
        w = self.weights
        enabled = {
            "A": (
                "transport",
                "surface",
                "surface_uncertainty_nll",
                "confidence_brier",
                "immersion",
                "metric_spd",
            ),
            "B": (
                "transport",
                "surface",
                "surface_uncertainty_nll",
                "confidence_brier",
                "render",
                "ssim",
                "perceptual",
                "tile_opacity",
                "vggt_depth_reprojection",
                "vggt_track_cycle",
                "vggt_depth_normal",
                "mask",
                "mesh_depth",
                "mesh_normal",
                "immersion",
                "atlas_c0",
                "atlas_c1",
                "atlas_curvature",
                "atlas_multilevel",
                "topology_prior",
                "topology_supervision",
                "dino_relational",
                "trellis_latent_relational",
                "sheet",
                "opacity",
            ),
            "D": tuple(terms.keys()),
            "E": tuple(terms.keys()),
            "F": tuple(terms.keys()),
        }[phase]
        weight = {
            "transport": w.transport,
            "surface": w.surface,
            "surface_uncertainty_nll": w.surface_uncertainty_nll,
            "confidence_brier": w.confidence_brier,
            "render": w.render,
            "ssim": w.ssim,
            "perceptual": w.perceptual,
            "mask": w.mask,
            "mesh_depth": w.mesh_depth,
            "mesh_normal": w.mesh_normal,
            "vggt_depth_reprojection": w.vggt_depth_reprojection,
            "vggt_track_cycle": w.vggt_track_cycle,
            "vggt_depth_normal": w.vggt_depth_normal,
            "camera_center": w.camera_center,
            "camera_rotation": w.camera_rotation,
            "camera_intrinsics": w.camera_intrinsics,
            "immersion": w.immersion,
            "atlas_c0": w.atlas_c0,
            "atlas_c1": w.atlas_c1,
            "atlas_curvature": w.atlas_curvature,
            "atlas_multilevel": w.atlas_multilevel,
            "topology_prior": w.topology_prior,
            "topology_supervision": w.topology_supervision,
            "dino_relational": w.dino_relational,
            "trellis_latent_relational": w.trellis_latent_relational,
            "sheet": w.sheet,
            "opacity": w.opacity,
            "tile_opacity": w.tile_opacity,
            "metric_spd": w.metric_spd,
            "feasibility": w.feasibility,
            "flow": w.flow,
        }
        total = sum(weight[name] * terms[name] for name in enabled)
        return total, terms

    def _flow_pretraining_loss(
        self,
        model: GraftGS,
        output: GraftGSOutput,
        batch: Mapping[str, object],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if output.execution_stage != "flow_pretraining":
            raise ValueError("Phase C must use the flow_pretraining execution stage")
        if any(scene.gaussians is not None or scene.mesh is not None for scene in output.scenes):
            raise ValueError("Phase C must stop before analytical asset construction")
        device = output.vggt.images.device
        target_states = batch.get("target_states")
        target_confidence_value = batch.get("teacher_bundle_confidence")
        target_mask_value = batch.get("teacher_bundle_supervision_mask")
        if target_states is None:
            surface_target = batch.get("surface_voxel_centers")
            if surface_target is None:
                raise ValueError(
                    "Phase C requires target_states or direct surface_voxel_centers"
                )
            surface = torch.as_tensor(
                surface_target,
                device=device,
                dtype=output.vggt.depth.dtype,
            )
            if surface.ndim == 2:
                surface = surface[None]
            if surface.ndim != 3 or surface.shape[0] != len(output.scenes):
                raise ValueError("Phase C surface targets must have shape [B,N,3]")
            target_states = [
                derive_feasible_surface_target(
                    scene.initial_state,
                    surface[index],
                    model.config.barrier,
                )[0]
                for index, scene in enumerate(output.scenes)
            ]
            target_confidence = torch.ones(
                len(output.scenes), device=device, dtype=output.vggt.depth.dtype
            )
        else:
            provenance = batch.get("target_state_provenance")
            if provenance == "explicit_serialized_manifold_target":
                target_confidence = torch.as_tensor(
                    batch.get("target_state_confidence", 1.0),
                    device=device,
                    dtype=output.vggt.depth.dtype,
                ).reshape(-1)
            elif provenance == "teacher_refined_fixed_stratum":
                if target_confidence_value is None or target_mask_value is None:
                    raise ValueError(
                        "teacher Phase-C states require confidence and activation mask"
                    )
                target_confidence = torch.as_tensor(
                    target_confidence_value,
                    device=device,
                    dtype=output.vggt.depth.dtype,
                ).reshape(-1)
                target_mask = torch.as_tensor(
                    target_mask_value, device=device, dtype=torch.bool
                ).reshape(-1)
                if target_mask.numel() != len(output.scenes) or not bool(
                    torch.all(target_mask)
                ):
                    raise ValueError("Phase-C teacher activation masks are inconsistent")
            else:
                raise ValueError("Phase-C target state has unavailable/unsupported provenance")
            if target_confidence.numel() not in {1, len(output.scenes)}:
                raise ValueError("Phase-C target confidence has incompatible shape")
            if target_confidence.numel() == 1:
                target_confidence = target_confidence.expand(len(output.scenes))
            if torch.any((target_confidence <= 0) | (target_confidence > 1)):
                raise ValueError("Phase-C target confidence must lie in (0,1]")
        if len(target_states) != len(output.scenes):
            raise ValueError("Phase C requires one manifold target per scene")
        target_states = minibatch_ot_flow_coupling(output.scenes, target_states)
        flow_losses = []
        for scene, target in zip(output.scenes, target_states):
            if not isinstance(target, ManifoldState):
                raise TypeError("Phase C target_states must contain ManifoldState objects")
            time = torch.rand(
                (), device=device, dtype=scene.initial_state.position.dtype
            )
            flow_losses.append(model_module_flow_loss(model, scene, target, time))
        terms = {
            "flow": torch.sum(torch.stack(flow_losses) * target_confidence)
            / target_confidence.sum().clamp_min(1.0e-8),
            "feasibility": torch.stack(
                [
                    differentiable_feasibility_loss(scene, model.config.barrier)
                    for scene in output.scenes
                ]
            ).mean(),
        }
        total = (
            self.weights.flow * terms["flow"]
            + self.weights.feasibility * terms["feasibility"]
        )
        return total, terms

    def _evidence_calibration_loss(
        self,
        output: GraftGSOutput,
        batch: Mapping[str, object],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if output.scenes:
            raise ValueError("Phase A must use the evidence_calibration execution stage")
        if output.evidence_particles is None:
            raise ValueError("Phase A output has no geometric evidence particles")
        surface_value = batch.get("surface_voxel_centers")
        if surface_value is None:
            raise ValueError("Phase A requires direct surface_voxel_centers supervision")
        surface = torch.as_tensor(
            surface_value,
            device=output.vggt.images.device,
            dtype=output.vggt.depth.dtype,
        )
        if surface.ndim == 2:
            surface = surface[None]
        if surface.shape[0] != len(output.evidence_particles):
            raise ValueError("Phase A surface batch and evidence batch disagree")
        cell_size_value = batch.get("surface_cell_size")
        if cell_size_value is None:
            raise ValueError("Phase A requires explicit surface_cell_size quantization metadata")
        cell_size_tensor = torch.as_tensor(cell_size_value).reshape(-1)
        if cell_size_tensor.numel() not in {1, len(output.evidence_particles)}:
            raise ValueError("surface_cell_size must be scalar or one value per object")
        nll_values = []
        brier_values = []
        for index, evidence in enumerate(output.evidence_particles):
            cell_size = float(
                cell_size_tensor[0 if cell_size_tensor.numel() == 1 else index]
            )
            nll, brier = evidence_surface_calibration(
                evidence.positions,
                evidence.covariance,
                evidence.confidence,
                surface[index],
                cell_size,
            )
            nll_values.append(nll)
            brier_values.append(brier)
        terms = {
            "surface_uncertainty_nll": torch.stack(nll_values).mean(),
            "confidence_brier": torch.stack(brier_values).mean(),
        }
        total = (
            self.weights.surface_uncertainty_nll * terms["surface_uncertainty_nll"]
            + self.weights.confidence_brier * terms["confidence_brier"]
        )
        return total, terms


def model_module_flow_loss(model: GraftGS, scene: object, target: ManifoldState, time: Tensor) -> Tensor:
    interpolated, velocity = geodesic_interpolate(scene.initial_state, target, time)
    predicted = model.vector_field(scene.atlas, interpolated, time)
    error = ManifoldTangent(
        predicted.position - velocity.position,
        predicted.rotation_body - velocity.rotation_body,
        predicted.covariance - velocity.covariance,
        predicted.opacity_logit - velocity.opacity_logit,
        predicted.appearance - velocity.appearance,
        predicted.latent - velocity.latent,
    )
    return product_metric_squared(interpolated, error) / interpolated.position.shape[0]


def minibatch_ot_flow_coupling(
    scenes: Sequence[SceneOutput],
    targets: Sequence[ManifoldState],
) -> list[ManifoldState]:
    """Exact discrete OT coupling inside compatible topology strata."""

    if len(scenes) != len(targets):
        raise ValueError("flow coupling requires one target per scene")

    def signature(state: ManifoldState) -> tuple[object, ...]:
        return (
            tuple(state.complex.atlas_node_index.tolist()),
            tuple(map(tuple, state.complex.edges.tolist())),
            tuple(map(tuple, state.complex.faces.tolist())),
        )

    groups: dict[tuple[object, ...], list[int]] = {}
    for index, (scene, target) in enumerate(zip(scenes, targets)):
        left_signature = signature(scene.initial_state)
        if left_signature != signature(target):
            raise ValueError("a Phase-C target does not share its scene topology stratum")
        groups.setdefault(left_signature, []).append(index)
    coupled = list(targets)
    from scipy.optimize import linear_sum_assignment

    with torch.no_grad():
        for indices in groups.values():
            if len(indices) < 2:
                continue
            rows = []
            for source_index in indices:
                start = scenes[source_index].initial_state
                costs = []
                for target_index in indices:
                    _, velocity = geodesic_interpolate(
                        start,
                        targets[target_index],
                        start.position.new_zeros(()),
                    )
                    costs.append(
                        product_metric_squared(start, velocity)
                        / start.position.shape[0]
                    )
                rows.append(torch.stack(costs))
            cost = torch.stack(rows)
            row, column = linear_sum_assignment(cost.detach().cpu().numpy())
            for local_row, local_column in zip(row.tolist(), column.tolist()):
                coupled[indices[local_row]] = targets[indices[local_column]]
    return coupled


def generalized_transport_kl(student_plan: Tensor, teacher_plan: Tensor) -> Tensor:
    """Generalized KL valid for finite positive measures of unequal mass."""

    if student_plan.shape != teacher_plan.shape:
        raise ValueError("student and teacher transport plans must have equal shape")
    teacher = teacher_plan.detach().clamp_min(1.0e-12)
    student = student_plan.clamp_min(1.0e-12)
    return torch.sum(
        teacher * (torch.log(teacher) - torch.log(student)) - teacher + student
    ) / teacher.sum().clamp_min(1.0e-12)


def transport_packed_irrep(
    value: Tensor,
    source_rotation: Tensor,
    target_rotation: Tensor,
) -> Tensor:
    """Transport packed local irreps between two chart gauges."""

    fields = IrrepTensor.from_packed(value)
    connection = target_rotation.transpose(-1, -2) @ source_rotation
    vector = torch.einsum("vij,vcj->vci", connection, fields.vector)
    matrix = l2_to_matrix(fields.tensor)
    matrix = torch.einsum(
        "vij,vcjk,vkl->vcil",
        connection,
        matrix,
        connection.transpose(-1, -2),
    )
    return IrrepTensor(fields.scalar, vector, matrix_to_l2(matrix)).pack()


def gauge_covariant_activation_distillation(
    student_scene: SceneOutput,
    teacher_scene: SceneOutput,
) -> Tensor:
    """Compare encoder irreps in world tensors, independent of local gauges."""

    student_activation = student_scene.encoder_activations
    teacher_activation = teacher_scene.encoder_activations
    if student_activation is None or teacher_activation is None:
        raise ValueError("Phase-E activation distillation requires captured encoder fields")
    if len(student_activation) != len(teacher_activation):
        raise ValueError("teacher and student captured different encoder depths")
    student_nodes = student_scene.mapping.graph.atlas_node_index
    teacher_nodes = teacher_scene.mapping.graph.atlas_node_index
    if not torch.equal(student_nodes, teacher_nodes):
        raise ValueError("activation distillation requires identical persistent atlas rows")
    student_frame = student_scene.atlas.chart_frames[student_nodes]
    teacher_frame = teacher_scene.atlas.chart_frames[teacher_nodes]
    losses = []
    for left, right in zip(student_activation, teacher_activation):
        if (
            left.scalar.shape != right.scalar.shape
            or left.vector.shape != right.vector.shape
            or left.tensor.shape != right.tensor.shape
        ):
            raise ValueError("teacher/student irrep activation contracts differ")
        scalar = torch.nn.functional.smooth_l1_loss(
            left.scalar, right.scalar.detach()
        )
        left_vector = torch.einsum("vij,vcj->vci", student_frame, left.vector)
        right_vector = torch.einsum("vij,vcj->vci", teacher_frame, right.vector).detach()
        vector = torch.nn.functional.smooth_l1_loss(left_vector, right_vector)
        left_tensor = torch.einsum(
            "vij,vcjk,vkl->vcil",
            student_frame,
            l2_to_matrix(left.tensor),
            student_frame.transpose(-1, -2),
        )
        right_tensor = torch.einsum(
            "vij,vcjk,vkl->vcil",
            teacher_frame,
            l2_to_matrix(right.tensor),
            teacher_frame.transpose(-1, -2),
        ).detach()
        tensor = torch.nn.functional.smooth_l1_loss(left_tensor, right_tensor)
        losses.append(scalar + vector + tensor)
    return torch.stack(losses).mean()


def _deterministic_probe_like(value: Tensor, phase: float) -> Tensor:
    index = torch.arange(value.numel(), device=value.device, dtype=torch.float64)
    probe = torch.sin(index * 1.6180339887498948 + phase)
    return probe.to(value.dtype).reshape_as(value)


def deterministic_manifold_probe(state: ManifoldState) -> ManifoldTangent:
    """Unit product-metric tangent used for reproducible Hutchinson JVPs."""

    covariance = _deterministic_probe_like(state.covariance, 2.1)
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    probe = ManifoldTangent(
        _deterministic_probe_like(state.position, 0.1),
        _deterministic_probe_like(state.rotation[..., 0], 1.1),
        covariance,
        _deterministic_probe_like(state.opacity_logit, 3.1),
        _deterministic_probe_like(state.appearance, 4.1),
        _deterministic_probe_like(state.latent, 5.1),
    )
    norm = torch.sqrt(product_metric_squared(state, probe).clamp_min(1.0e-12))
    return probe.scaled(norm.reciprocal())


def _vector_field_directional_jvp(
    model: GraftGS,
    scene: SceneOutput,
    state: ManifoldState,
    probe: ManifoldTangent,
    time: Tensor,
    create_graph: bool,
) -> ManifoldTangent:
    zero = state.position.new_zeros((), requires_grad=True)

    def evaluate(step: Tensor) -> tuple[Tensor, ...]:
        velocity = model.vector_field(scene.atlas, retract(state, probe, step), time)
        return (
            velocity.position,
            velocity.rotation_body,
            velocity.covariance,
            velocity.opacity_logit,
            velocity.appearance,
            velocity.latent,
        )

    _, directional = torch.autograd.functional.jvp(
        evaluate,
        zero,
        zero.new_ones(()),
        create_graph=create_graph,
        strict=False,
    )
    return ManifoldTangent(*directional)


def vector_field_jacobian_distillation(
    student_model: GraftGS,
    teacher_model: GraftGS,
    student_scene: SceneOutput,
    teacher_scene: SceneOutput,
    time: Tensor,
) -> Tensor:
    """Match one deterministic product-manifold Jacobian-vector product."""

    student_state = student_scene.final_state
    teacher_state = teacher_scene.final_state
    student_probe = deterministic_manifold_probe(student_state)
    student_to_teacher = (
        teacher_state.rotation.transpose(-1, -2) @ student_state.rotation
    )
    teacher_probe = ManifoldTangent(
        student_probe.position,
        torch.einsum(
            "vij,vj->vi", student_to_teacher, student_probe.rotation_body
        ),
        spd_parallel_transport(
            student_state.covariance,
            teacher_state.covariance,
            student_probe.covariance,
        ),
        student_probe.opacity_logit,
        student_probe.appearance,
        transport_packed_irrep(
            student_probe.latent,
            student_state.rotation,
            teacher_state.rotation,
        ),
    )
    student_jvp = _vector_field_directional_jvp(
        student_model,
        student_scene,
        student_state,
        student_probe,
        time,
        create_graph=True,
    )
    teacher_jvp = _vector_field_directional_jvp(
        teacher_model,
        teacher_scene,
        teacher_state,
        teacher_probe,
        time,
        create_graph=False,
    )
    teacher_to_student = (
        student_state.rotation.transpose(-1, -2) @ teacher_state.rotation
    )
    teacher_jvp_student = ManifoldTangent(
        teacher_jvp.position.detach(),
        torch.einsum(
            "vij,vj->vi",
            teacher_to_student,
            teacher_jvp.rotation_body.detach(),
        ),
        spd_parallel_transport(
            teacher_state.covariance,
            student_state.covariance,
            teacher_jvp.covariance.detach(),
        ),
        teacher_jvp.opacity_logit.detach(),
        teacher_jvp.appearance.detach(),
        transport_packed_irrep(
            teacher_jvp.latent.detach(),
            teacher_state.rotation,
            student_state.rotation,
        ),
    )
    error = ManifoldTangent(
        student_jvp.position - teacher_jvp_student.position,
        student_jvp.rotation_body - teacher_jvp_student.rotation_body,
        student_jvp.covariance - teacher_jvp_student.covariance,
        student_jvp.opacity_logit - teacher_jvp_student.opacity_logit,
        student_jvp.appearance - teacher_jvp_student.appearance,
        student_jvp.latent - teacher_jvp_student.latent,
    )
    return product_metric_squared(student_state, error) / student_state.position.shape[0]


def distillation_loss(
    student: GraftGSOutput,
    teacher: GraftGSOutput,
    weights: LossWeights = LossWeights(),
    teacher_confidence: float = 1.0,
    teacher_topology_confidence: float = 0.5,
    student_model: Optional[GraftGS] = None,
    teacher_model: Optional[GraftGS] = None,
) -> Tensor:
    if not 0.0 <= teacher_topology_confidence <= 1.0:
        raise ValueError("teacher_topology_confidence must lie in [0,1]")
    if not 0.0 <= teacher_confidence <= 1.0:
        raise ValueError("teacher_confidence must lie in [0,1]")
    losses = []
    for left, right in zip(student.scenes, teacher.scenes):
        aligned_stratum = (
            left.final_state.position.shape == right.final_state.position.shape
            and torch.equal(left.final_state.complex.faces, right.final_state.complex.faces)
            and torch.equal(
                left.final_state.complex.atlas_node_index,
                right.final_state.complex.atlas_node_index,
            )
        )
        state = left.mapping.plan.new_zeros(())
        if aligned_stratum:
            state_error = ManifoldTangent(
                left.final_state.position - right.final_state.position.detach(),
                so3_log(
                    left.final_state.rotation.transpose(-1, -2)
                    @ right.final_state.rotation.detach()
                ),
                left.final_state.covariance - right.final_state.covariance.detach(),
                left.final_state.opacity_logit - right.final_state.opacity_logit.detach(),
                left.final_state.appearance - right.final_state.appearance.detach(),
                left.final_state.latent - right.final_state.latent.detach(),
            )
            state = product_metric_squared(left.final_state, state_error) / left.final_state.position.shape[0]
        transport = left.mapping.plan.new_zeros(())
        if torch.equal(left.mapping.graph.edge_index, right.mapping.graph.edge_index):
            transport = generalized_transport_kl(
                left.mapping.plan,
                right.mapping.plan,
            )
        teacher_ids = [candidate.identifier for candidate in right.topology.candidates]
        student_ids = [candidate.identifier for candidate in left.topology.candidates]
        if teacher_ids == student_ids:
            teacher_probability = right.topology.probability.detach().clamp_min(1.0e-12)
            topology = -torch.sum(teacher_probability * torch.log(left.topology.probability.clamp_min(1.0e-12)))
        else:
            identifier = right.topology.selected.identifier
            if identifier in student_ids:
                topology = -torch.log(
                    left.topology.probability[student_ids.index(identifier)].clamp_min(1.0e-12)
                )
            else:
                teacher_betti = right.topology.selected.betti
                mismatch = left.topology.probability.new_tensor(
                    [
                        sum(
                            abs(candidate.betti[dimension] - teacher_betti[dimension])
                            for dimension in range(3)
                        )
                        for candidate in left.topology.candidates
                    ]
                )
                topology = torch.sum(left.topology.probability * mismatch)
        render = left.mapping.plan.new_zeros(())
        if left.render is not None and right.render is not None:
            teacher_color = right.render.color.detach()
            teacher_alpha = right.render.alpha.detach()
            render = torch.nn.functional.smooth_l1_loss(
                left.render.color,
                teacher_color,
            ) + torch.nn.functional.smooth_l1_loss(
                left.render.alpha,
                teacher_alpha,
            )
            valid_depth = (
                (teacher_alpha > 1.0e-3)
                & torch.isfinite(right.render.depth)
                & torch.isfinite(left.render.depth)
            )
            if bool(torch.any(valid_depth)):
                render = render + torch.nn.functional.smooth_l1_loss(
                    left.render.depth[valid_depth],
                    right.render.depth.detach()[valid_depth],
                )
            teacher_normal = torch.nn.functional.normalize(
                right.render.normal.detach(), dim=1, eps=1.0e-8
            )
            student_normal = torch.nn.functional.normalize(
                left.render.normal, dim=1, eps=1.0e-8
            )
            normal_cosine = torch.sum(
                teacher_normal * student_normal, dim=1, keepdim=True
            ).abs()
            valid_normal = teacher_alpha > 1.0e-3
            if bool(torch.any(valid_normal)):
                render = render + (1.0 - normal_cosine[valid_normal]).mean()
        field = left.mapping.plan.new_zeros(())
        activation = left.mapping.plan.new_zeros(())
        jacobian = left.mapping.plan.new_zeros(())
        if aligned_stratum and student_model is not None and teacher_model is not None:
            activation = gauge_covariant_activation_distillation(left, right)
            time = left.final_state.position.new_tensor(0.5)
            student_velocity = student_model.vector_field(
                left.atlas,
                left.final_state,
                time,
            )
            with torch.no_grad():
                teacher_velocity = teacher_model.vector_field(
                    right.atlas,
                    right.final_state,
                    time,
                )
            teacher_to_student = (
                left.final_state.rotation.transpose(-1, -2)
                @ right.final_state.rotation.detach()
            )
            teacher_rotation_body = torch.einsum(
                "vij,vj->vi",
                teacher_to_student,
                teacher_velocity.rotation_body.detach(),
            )
            teacher_covariance = spd_parallel_transport(
                right.final_state.covariance,
                left.final_state.covariance,
                teacher_velocity.covariance.detach(),
            )
            teacher_latent = transport_packed_irrep(
                teacher_velocity.latent.detach(),
                right.final_state.rotation,
                left.final_state.rotation,
            )
            field_error = ManifoldTangent(
                student_velocity.position - teacher_velocity.position.detach(),
                student_velocity.rotation_body - teacher_rotation_body,
                student_velocity.covariance - teacher_covariance,
                student_velocity.opacity_logit
                - teacher_velocity.opacity_logit.detach(),
                student_velocity.appearance - teacher_velocity.appearance.detach(),
                student_velocity.latent - teacher_latent,
            )
            field = product_metric_squared(
                left.final_state,
                field_error,
            ) / left.final_state.position.shape[0]
            jacobian = vector_field_jacobian_distillation(
                student_model,
                teacher_model,
                left,
                right,
                time,
            )
        losses.append(
            teacher_confidence
            * (
                weights.distill_state * state
                + weights.distill_transport * transport
                + teacher_topology_confidence
                * weights.distill_topology
                * topology
                + weights.distill_render * render
                + weights.distill_field * field
                + weights.distill_activation * activation
                + weights.distill_jacobian * jacobian
            )
        )
    return torch.stack(losses).mean()


__all__ = [
    "GraftGSLoss",
    "LossWeights",
    "LearnedPerceptualPyramid",
    "ViewConditionedObjectives",
    "atlas_overlap_consistency",
    "atlas_multilevel_consistency",
    "conservative_tile_opacity_bound",
    "distillation_loss",
    "robust_rgb",
    "symmetric_surface_chamfer",
    "evidence_surface_calibration",
    "generalized_transport_kl",
    "gauge_covariant_activation_distillation",
    "multiscale_perceptual_loss",
    "multiview_reprojection_cycle_loss",
    "minibatch_ot_flow_coupling",
    "nearest_surface_targets",
    "provenance_weighted_topology_supervision",
    "structural_similarity_loss",
    "view_conditioned_objectives",
    "vggt_depth_normal_field",
    "vector_field_jacobian_distillation",
]

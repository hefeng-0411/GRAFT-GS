"""Traceable adapter around the released VGGT implementation.

The baseline package is imported dynamically and never modified.  Released
cached aggregator taps concatenate frame/global features and are 2048-wide;
GRAFT-GS reduces them by an orthogonally initialized multiplicity projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torch.nn.utils.parametrize as parametrize


@dataclass
class VGGTGeometryOutput:
    images: Tensor
    patch_features: Tensor
    extrinsics_world_to_camera: Tensor
    intrinsics: Tensor
    depth: Tensor
    depth_confidence: Tensor
    world_points: Tensor
    world_points_confidence: Tensor


@dataclass
class CameraAlignmentDiagnostics:
    """Differentiable Sim(3) gauge alignment into a supervised world frame."""

    scale: Tensor
    rotation_predicted_to_target: Tensor
    translation_predicted_to_target: Tensor
    center_rmse: Tensor
    rotation_geodesic: Tensor
    intrinsic_log_focal_error: Tensor


def _dct_projection(rows: int, columns: int) -> Tensor:
    row = torch.arange(rows, dtype=torch.float32)[:, None]
    column = torch.arange(columns, dtype=torch.float32)[None]
    basis = torch.cos(torch.pi / columns * (column + 0.5) * row)
    basis[0] *= 1.0 / sqrt(2.0)
    return basis * sqrt(2.0 / columns)


class VGGTAdapter(nn.Module):
    def __init__(self, model: nn.Module, feature_dim: int = 1024, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.model = model
        self.feature_projection = nn.Linear(2048, feature_dim, bias=False)
        self.tap_logits = nn.Parameter(torch.zeros(4))
        with torch.no_grad():
            self.feature_projection.weight.copy_(_dct_projection(feature_dim, 2048))
        if freeze_backbone:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        self._lora_modules: list[nn.Module] = []

    def install_late_lora(self, last_blocks: int = 4, rank: int = 8, alpha: float = 8.0) -> int:
        """Install low-rank updates in released VGGT late attention/FFN maps."""

        if self._lora_modules:
            return len(self._lora_modules)
        blocks = list(self.model.aggregator.frame_blocks[-last_blocks:]) + list(
            self.model.aggregator.global_blocks[-last_blocks:]
        )
        for block in blocks:
            for child in block.modules():
                if isinstance(child, nn.Linear) and not parametrize.is_parametrized(child, "weight"):
                    parametrization = _LowRankWeight(child.in_features, child.out_features, rank, alpha)
                    parametrize.register_parametrization(child, "weight", parametrization)
                    self._lora_modules.append(parametrization)
        return len(self._lora_modules)

    def lora_parameters(self):
        for module in self._lora_modules:
            yield from module.parameters()

    @classmethod
    def from_pretrained(cls, checkpoint: str = "facebook/VGGT-1B", feature_dim: int = 1024, freeze_backbone: bool = True) -> "VGGTAdapter":
        from vggt.models.vggt import VGGT

        return cls(VGGT.from_pretrained(checkpoint), feature_dim, freeze_backbone)

    def forward(self, images: Tensor) -> VGGTGeometryOutput:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5:
            raise ValueError("images must have shape [B,K,3,H,W]")
        use_bf16 = images.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            taps, patch_start = self.model.aggregator(images)
        cached = [value for value in taps if value is not None]
        if len(cached) != 4:
            raise RuntimeError(f"released VGGT contract requires four multiscale taps, found {len(cached)}")
        patch_taps = torch.stack([F.layer_norm(value[:, :, patch_start:].float(), value.shape[-1:]) for value in cached], dim=0)
        tap_weight = torch.softmax(self.tap_logits, dim=0).reshape(-1, 1, 1, 1, 1)
        patch_features = self.feature_projection(torch.sum(tap_weight * patch_taps, dim=0))
        with torch.autocast(device_type=images.device.type, enabled=False):
            pose_encoding = self.model.camera_head(taps)[-1].float()
            depth, depth_confidence = self.model.depth_head(taps, images=images, patch_start_idx=patch_start)
            world_points, world_points_confidence = self.model.point_head(taps, images=images, patch_start_idx=patch_start)
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri

            extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_encoding, images.shape[-2:])
        return VGGTGeometryOutput(
            images=images,
            patch_features=patch_features,
            extrinsics_world_to_camera=extrinsics,
            intrinsics=intrinsics,
            depth=depth,
            depth_confidence=depth_confidence,
            world_points=world_points,
            world_points_confidence=world_points_confidence,
        )


def align_vggt_to_supervised_cameras(
    output: VGGTGeometryOutput,
    target_extrinsics_world_to_camera: Tensor,
    target_intrinsics: Tensor,
    orientation_weight: float = 1.0,
) -> tuple[VGGTGeometryOutput, CameraAlignmentDiagnostics]:
    """Remove VGGT's global similarity gauge without hiding relative pose error.

    A Kabsch solve jointly aligns centered camera centers and camera-frame axes.
    The positive scale is then solved analytically from the centers.  Only a
    *single* scene-level Sim(3) is removed; per-camera residuals remain visible
    to the supervised objective.  Geometry, depths, and extrinsics are all
    transformed together, preserving their projective consistency.
    """

    predicted = output.extrinsics_world_to_camera
    target = target_extrinsics_world_to_camera.to(device=predicted.device, dtype=predicted.dtype)
    target_intrinsics = target_intrinsics.to(device=output.intrinsics.device, dtype=output.intrinsics.dtype)
    if predicted.shape != target.shape or predicted.ndim != 4 or predicted.shape[-2:] != (3, 4):
        raise ValueError("predicted and supervised extrinsics must both have shape [B,K,3,4]")
    if target_intrinsics.shape != output.intrinsics.shape:
        raise ValueError("supervised intrinsics must match the VGGT [B,K,3,3] camera batch")
    if predicted.shape[1] < 2:
        raise ValueError("Sim(3) camera gauge alignment requires at least two views")
    if orientation_weight < 0:
        raise ValueError("orientation_weight must be non-negative")

    predicted_r_c2w = predicted[..., :3, :3].transpose(-1, -2)
    target_r_c2w = target[..., :3, :3].transpose(-1, -2)
    predicted_center = -torch.einsum(
        "bkij,bkj->bki", predicted_r_c2w, predicted[..., :3, 3]
    )
    target_center = -torch.einsum("bkij,bkj->bki", target_r_c2w, target[..., :3, 3])
    predicted_mean = predicted_center.mean(dim=1, keepdim=True)
    target_mean = target_center.mean(dim=1, keepdim=True)
    predicted_centered = predicted_center - predicted_mean
    target_centered = target_center - target_mean
    predicted_spread = torch.sqrt(predicted_centered.square().sum((1, 2)) / predicted.shape[1])
    target_spread = torch.sqrt(target_centered.square().sum((1, 2)) / predicted.shape[1])
    epsilon = torch.finfo(predicted.dtype).eps
    if bool(torch.any(predicted_spread.detach() <= 32.0 * epsilon)) or bool(
        torch.any(target_spread.detach() <= 32.0 * epsilon)
    ):
        raise ValueError("camera centers are degenerate for similarity-scale estimation")

    normalized_predicted = predicted_centered / predicted_spread[:, None, None]
    normalized_target = target_centered / target_spread[:, None, None]
    cross_covariance = normalized_predicted.transpose(1, 2) @ normalized_target
    orientation_covariance = torch.einsum(
        "bkic,bkjc->bij", predicted_r_c2w, target_r_c2w
    ) / float(predicted.shape[1])
    cross_covariance = cross_covariance + orientation_weight * orientation_covariance
    u, _, vh = torch.linalg.svd(cross_covariance)
    candidate = vh.transpose(-1, -2) @ u.transpose(-1, -2)
    sign = torch.where(
        torch.linalg.det(candidate) < 0,
        candidate.new_tensor(-1.0),
        candidate.new_tensor(1.0),
    )
    correction = torch.diag_embed(torch.stack((torch.ones_like(sign), torch.ones_like(sign), sign), dim=-1))
    rotation = vh.transpose(-1, -2) @ correction @ u.transpose(-1, -2)
    rotated_centered = torch.einsum("bij,bkj->bki", rotation, predicted_centered)
    scale = (
        (target_centered * rotated_centered).sum((1, 2))
        / predicted_centered.square().sum((1, 2)).clamp_min(epsilon)
    ).clamp_min(32.0 * epsilon)
    translation = target_mean[:, 0] - scale[:, None] * torch.einsum(
        "bij,bj->bi", rotation, predicted_mean[:, 0]
    )

    aligned_center = scale[:, None, None] * torch.einsum(
        "bij,bkj->bki", rotation, predicted_center
    ) + translation[:, None]
    aligned_r_c2w = rotation[:, None] @ predicted_r_c2w
    aligned_r_w2c = aligned_r_c2w.transpose(-1, -2)
    aligned_translation = -torch.einsum("bkij,bkj->bki", aligned_r_w2c, aligned_center)
    aligned_extrinsics = torch.cat((aligned_r_w2c, aligned_translation[..., None]), dim=-1)

    world_points = output.world_points
    if world_points.shape[-1] != 3 or world_points.shape[:2] != predicted.shape[:2]:
        raise ValueError("VGGT world points must have shape [B,K,...,3]")
    flat_world = world_points.reshape(world_points.shape[0], -1, 3)
    aligned_world = scale[:, None, None] * torch.einsum("bij,bnj->bni", rotation, flat_world)
    aligned_world = (aligned_world + translation[:, None]).reshape_as(world_points)
    depth_scale_shape = (scale.shape[0],) + (1,) * (output.depth.ndim - 1)
    aligned_depth = output.depth * scale.reshape(depth_scale_shape)

    center_residual = aligned_center - target_center
    relative_rotation = target_r_c2w.transpose(-1, -2) @ aligned_r_c2w
    cosine = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    skew = torch.stack(
        (
            relative_rotation[..., 2, 1] - relative_rotation[..., 1, 2],
            relative_rotation[..., 0, 2] - relative_rotation[..., 2, 0],
            relative_rotation[..., 1, 0] - relative_rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    rotation_error = torch.atan2(sine, cosine)
    predicted_focal = output.intrinsics[..., (0, 1), (0, 1)].clamp_min(epsilon)
    target_focal = target_intrinsics[..., (0, 1), (0, 1)].clamp_min(epsilon)
    focal_residual = torch.log(predicted_focal / target_focal)

    def stable_rms(value: Tensor, dimensions: tuple[int, ...]) -> Tensor:
        smoothing = 32.0 * epsilon
        return torch.sqrt(value.square().mean(dim=dimensions) + smoothing**2) - smoothing

    diagnostics = CameraAlignmentDiagnostics(
        scale=scale,
        rotation_predicted_to_target=rotation,
        translation_predicted_to_target=translation,
        center_rmse=stable_rms(center_residual, (1, 2)),
        rotation_geodesic=rotation_error.mean(dim=1),
        intrinsic_log_focal_error=stable_rms(focal_residual, (1, 2)),
    )
    aligned = VGGTGeometryOutput(
        images=output.images,
        patch_features=output.patch_features,
        extrinsics_world_to_camera=aligned_extrinsics,
        intrinsics=output.intrinsics,
        depth=aligned_depth,
        depth_confidence=output.depth_confidence,
        world_points=aligned_world,
        world_points_confidence=output.world_points_confidence,
    )
    return aligned, diagnostics


__all__ = [
    "CameraAlignmentDiagnostics",
    "VGGTAdapter",
    "VGGTGeometryOutput",
    "align_vggt_to_supervised_cameras",
]


class _LowRankWeight(nn.Module):
    def __init__(self, input_features: int, output_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1 or rank > min(input_features, output_features):
            raise ValueError("LoRA rank is invalid for this projection")
        self.a = nn.Parameter(torch.empty(rank, input_features))
        self.b = nn.Parameter(torch.zeros(output_features, rank))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a, a=sqrt(5.0))

    def forward(self, weight: Tensor) -> Tensor:
        return weight + self.scale * (self.b @ self.a)

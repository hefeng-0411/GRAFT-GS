"""Offline topology-fixed robust teacher bundle refinement.

This module implements the teacher-only optimization in GRAFT-GS Section 5.10.
It refines cameras and one certified product-manifold atlas state, then invokes
the same analytical readout used by student inference.  It never optimizes a
free Gaussian cloud or changes the selected topology stratum.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional

import torch
from torch import Tensor, nn

from ..integration.pipeline import GraftGS, SceneOutput
from ..manifold.barrier import BarrierProjector, FeasibilityReport
from ..manifold.geometry import ManifoldState, ManifoldTangent, product_metric_squared, retract, so3_exp
from ..readout.assets import GaussianAsset, MeshAsset
from ..readout.renderer import CameraBatch, RenderResult
from ..topology.strata import SimplicialComplex
from .losses import differentiable_feasibility_loss


@dataclass(frozen=True)
class TeacherBundleConfig:
    iterations: int = 300
    learning_rate: float = 2.0e-3
    cauchy_scale: float = 0.03
    camera_rotation_radius: float = 0.03
    camera_translation_radius: float = 0.03
    log_focal_radius: float = 0.05
    principal_point_radius_pixels: float = 4.0
    camera_prior_weight: float = 0.1
    state_prior_weight: float = 0.01
    alpha_weight: float = 0.5
    feasibility_weight: float = 1.0
    feasibility_relative_margin: float = 0.05
    confidence_reprojection_sigma: float = 0.05
    confidence_topology_temperature: float = 0.5
    confidence_cycle_temperature: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            self.iterations,
            self.learning_rate,
            self.cauchy_scale,
            self.camera_rotation_radius,
            self.camera_translation_radius,
            self.log_focal_radius,
            self.principal_point_radius_pixels,
            self.confidence_reprojection_sigma,
            self.confidence_topology_temperature,
            self.confidence_cycle_temperature,
        )
        if any(float(value) <= 0 for value in positive):
            raise ValueError("teacher refinement scales and iteration count must be positive")
        if min(
            self.camera_prior_weight,
            self.state_prior_weight,
            self.alpha_weight,
            self.feasibility_weight,
            self.feasibility_relative_margin,
        ) < 0:
            raise ValueError("teacher refinement weights/margin must be non-negative")


@dataclass(frozen=True)
class TeacherBundleResult:
    state: ManifoldState
    cameras: CameraBatch
    gaussians: GaussianAsset
    mesh: MeshAsset
    render: RenderResult
    feasibility: FeasibilityReport
    teacher_confidence: Tensor
    reprojection_rmse: Tensor
    topology_entropy: Tensor
    cycle_residual: Tensor
    loss_history: tuple[float, ...]


@dataclass(frozen=True)
class LoadedTeacherBundle:
    state: ManifoldState
    confidence: float
    topology_provenance: str
    metadata: Mapping[str, object]


class TopologyFixedTeacherBundleRefiner(nn.Module):
    """Robustly refine a production scene without crossing its topology stratum."""

    def __init__(
        self,
        model: GraftGS,
        scene: SceneOutput,
        cameras: CameraBatch,
        config: TeacherBundleConfig = TeacherBundleConfig(),
    ) -> None:
        super().__init__()
        if scene.gaussians is None or scene.mesh is None:
            raise ValueError("teacher refinement requires a full production scene")
        if scene.render_cameras is None:
            raise ValueError("teacher refinement requires audited render cameras")
        self.model = model
        self.scene = scene
        self.base_state = scene.final_state
        self.base_cameras = cameras
        self.config = config
        vertex_count = self.base_state.position.shape[0]
        view_count = cameras.extrinsics_world_to_camera.shape[0]
        dtype = self.base_state.position.dtype
        device = self.base_state.position.device
        self.position = nn.Parameter(torch.zeros(vertex_count, 3, dtype=dtype, device=device))
        self.rotation_body = nn.Parameter(torch.zeros(vertex_count, 3, dtype=dtype, device=device))
        self.covariance = nn.Parameter(torch.zeros(vertex_count, 3, 3, dtype=dtype, device=device))
        self.opacity_logit = nn.Parameter(torch.zeros_like(self.base_state.opacity_logit))
        self.appearance = nn.Parameter(torch.zeros_like(self.base_state.appearance))
        self.camera_rotation = nn.Parameter(torch.zeros(view_count, 3, dtype=dtype, device=device))
        self.camera_translation = nn.Parameter(torch.zeros(view_count, 3, dtype=dtype, device=device))
        self.log_focal = nn.Parameter(torch.zeros(view_count, 2, dtype=dtype, device=device))
        self.principal_point = nn.Parameter(torch.zeros(view_count, 2, dtype=dtype, device=device))
        self.projector = BarrierProjector(self.base_state, model.config.barrier)

    def state_tangent(self) -> ManifoldTangent:
        covariance = 0.5 * (self.covariance + self.covariance.transpose(-1, -2))
        return ManifoldTangent(
            self.position,
            self.rotation_body,
            covariance,
            self.opacity_logit,
            self.appearance,
            torch.zeros_like(self.base_state.latent),
        )

    def refinement_parameters(self) -> tuple[nn.Parameter, ...]:
        """Exclude the frozen GRAFT-GS teacher weights from bundle fitting."""

        return (
            self.position,
            self.rotation_body,
            self.covariance,
            self.opacity_logit,
            self.appearance,
            self.camera_rotation,
            self.camera_translation,
            self.log_focal,
            self.principal_point,
        )

    def refined_state(self) -> tuple[ManifoldState, FeasibilityReport]:
        state, report = self.projector.retract_with_backtracking(
            self.base_state, self.state_tangent(), 1.0
        )
        if not report.feasible:
            raise RuntimeError("teacher state could not be retracted into the fixed feasible stratum")
        return state, report

    def refined_cameras(self) -> CameraBatch:
        base_extrinsic = self.base_cameras.extrinsics_world_to_camera
        delta_rotation = so3_exp(self.camera_rotation)
        rotation = delta_rotation @ base_extrinsic[:, :3, :3]
        translation = torch.einsum(
            "vij,vj->vi", delta_rotation, base_extrinsic[:, :3, 3]
        ) + self.camera_translation
        extrinsic = torch.cat((rotation, translation[:, :, None]), dim=-1)
        intrinsic = self.base_cameras.intrinsics.clone()
        intrinsic[:, 0, 0] = intrinsic[:, 0, 0] * torch.exp(self.log_focal[:, 0])
        intrinsic[:, 1, 1] = intrinsic[:, 1, 1] * torch.exp(self.log_focal[:, 1])
        intrinsic[:, 0, 2] = intrinsic[:, 0, 2] + self.principal_point[:, 0]
        intrinsic[:, 1, 2] = intrinsic[:, 1, 2] + self.principal_point[:, 1]
        return CameraBatch(extrinsic, intrinsic, self.base_cameras.height, self.base_cameras.width)

    @torch.no_grad()
    def project_parameters(self) -> None:
        _, report = self.projector.retract_with_backtracking(
            self.base_state, self.state_tangent(), 1.0
        )
        if report.accepted_step < 1.0:
            for parameter in (
                self.position,
                self.rotation_body,
                self.covariance,
                self.opacity_logit,
                self.appearance,
            ):
                parameter.mul_(report.accepted_step)
        for parameter, radius in (
            (self.camera_rotation, self.config.camera_rotation_radius),
            (self.camera_translation, self.config.camera_translation_radius),
        ):
            norm = torch.linalg.vector_norm(parameter, dim=-1, keepdim=True)
            parameter.mul_(torch.clamp(radius / norm.clamp_min(1.0e-12), max=1.0))
        self.log_focal.clamp_(-self.config.log_focal_radius, self.config.log_focal_radius)
        self.principal_point.clamp_(
            -self.config.principal_point_radius_pixels,
            self.config.principal_point_radius_pixels,
        )

    def _objective(
        self,
        target_images: Tensor,
        target_alpha: Optional[Tensor],
    ) -> tuple[Tensor, ManifoldState, CameraBatch, GaussianAsset, MeshAsset, RenderResult, FeasibilityReport, Tensor]:
        state, report = self.refined_state()
        cameras = self.refined_cameras()
        gaussians, mesh = self.model.readout(
            self.scene.atlas, state, self.scene.mapping
        )
        render = self.model.renderer(gaussians, cameras)
        if render.color.shape != target_images.shape:
            raise ValueError("teacher target images must have shape [K,3,H,W]")
        if target_alpha is None:
            mask = torch.ones_like(render.alpha)
        else:
            mask = target_alpha.to(device=render.alpha.device, dtype=render.alpha.dtype)
            if mask.shape != render.alpha.shape:
                raise ValueError("teacher target alpha must have shape [K,1,H,W]")
        squared_color = (render.color - target_images).square().sum(dim=1, keepdim=True)
        robust_reprojection = torch.sum(
            torch.log1p(squared_color / self.config.cauchy_scale**2) * mask
        ) / mask.sum().clamp_min(1.0)
        alpha_loss = (
            torch.nn.functional.binary_cross_entropy(
                render.alpha.clamp(1.0e-6, 1.0 - 1.0e-6), mask
            )
            if target_alpha is not None
            else render.alpha.new_zeros(())
        )
        camera_prior = (
            self.camera_rotation.square().mean()
            / self.config.camera_rotation_radius**2
            + self.camera_translation.square().mean()
            / self.config.camera_translation_radius**2
            + self.log_focal.square().mean() / self.config.log_focal_radius**2
            + self.principal_point.square().mean()
            / self.config.principal_point_radius_pixels**2
        )
        state_prior = product_metric_squared(
            self.base_state, self.state_tangent()
        ) / self.base_state.position.shape[0]
        refined_scene = replace(self.scene, final_state=state)
        feasibility = differentiable_feasibility_loss(
            refined_scene,
            self.model.config.barrier,
            relative_hardening_margin=self.config.feasibility_relative_margin,
        )
        total = (
            robust_reprojection
            + self.config.alpha_weight * alpha_loss
            + self.config.camera_prior_weight * camera_prior
            + self.config.state_prior_weight * state_prior
            + self.config.feasibility_weight * feasibility
        )
        rmse = torch.sqrt(
            torch.sum(squared_color * mask) / (3.0 * mask.sum().clamp_min(1.0))
        )
        return total, state, cameras, gaussians, mesh, render, report, rmse

    def refine(
        self,
        target_images: Tensor,
        target_alpha: Optional[Tensor] = None,
        cycle_residual: float | Tensor = 0.0,
    ) -> TeacherBundleResult:
        target_images = target_images.to(
            device=self.base_state.position.device, dtype=self.base_state.position.dtype
        )
        if target_alpha is not None:
            target_alpha = target_alpha.to(
                device=target_images.device, dtype=target_images.dtype
            )
        optimizer = torch.optim.Adam(
            self.refinement_parameters(), lr=self.config.learning_rate
        )
        history = []
        for _ in range(self.config.iterations):
            optimizer.zero_grad(set_to_none=True)
            total, *_ = self._objective(target_images, target_alpha)
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("teacher refinement produced a non-finite objective")
            total.backward()
            optimizer.step()
            self.project_parameters()
            history.append(float(total.detach().cpu()))
        with torch.no_grad():
            _, state, cameras, gaussians, mesh, render, report, rmse = self._objective(
                target_images, target_alpha
            )
            probability = self.scene.topology.probability.clamp_min(1.0e-12)
            topology_entropy = -torch.sum(probability * torch.log(probability))
            cycle = torch.as_tensor(
                cycle_residual, device=rmse.device, dtype=rmse.dtype
            )
            confidence = (
                torch.exp(
                    -rmse.square()
                    / (2.0 * self.config.confidence_reprojection_sigma**2)
                )
                * torch.exp(
                    -topology_entropy / self.config.confidence_topology_temperature
                )
                * torch.exp(
                    -cycle / self.config.confidence_cycle_temperature
                )
            ).clamp(0.0, 1.0)
        return TeacherBundleResult(
            state,
            cameras,
            gaussians,
            mesh,
            render,
            report,
            confidence,
            rmse,
            topology_entropy,
            cycle,
            tuple(history),
        )


def load_teacher_bundle(
    path: str | Path,
    *,
    expected_object_id: Optional[str] = None,
    expected_manifest_sha256: Optional[str] = None,
    minimum_confidence: float = 0.0,
) -> LoadedTeacherBundle:
    """Load a typed teacher target without promoting it to direct ground truth."""

    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum teacher confidence must lie in [0,1]")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != "graft_gs_teacher_bundle_v1":
        raise ValueError("unsupported teacher bundle schema")
    if expected_object_id is not None and payload.get("object_id") != expected_object_id:
        raise ValueError("teacher bundle object identity differs from the dataset record")
    if (
        expected_manifest_sha256 is not None
        and payload.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError("teacher bundle was generated from a different dataset manifest")
    provenance = str(payload.get("topology_provenance", ""))
    if provenance != "teacher_refined_fixed_stratum":
        raise ValueError("teacher bundle has unsupported topology provenance")
    confidence = float(torch.as_tensor(payload.get("teacher_confidence", -1.0)))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("teacher bundle confidence is outside [0,1]")
    if confidence < minimum_confidence:
        raise ValueError(
            f"teacher bundle confidence {confidence:.6f} is below {minimum_confidence:.6f}"
        )
    value = payload.get("state")
    if not isinstance(value, Mapping):
        raise ValueError("teacher bundle has no manifold state")
    complex_ = SimplicialComplex(
        torch.as_tensor(value["atlas_node_index"], dtype=torch.int64),
        torch.as_tensor(value["edges"], dtype=torch.int64),
        torch.as_tensor(value["faces"], dtype=torch.int64),
    )
    state = ManifoldState(
        torch.as_tensor(value["position"]),
        torch.as_tensor(value["rotation"]),
        torch.as_tensor(value["covariance"]),
        torch.as_tensor(value["opacity_logit"]),
        torch.as_tensor(value["appearance"]),
        torch.as_tensor(value["latent"]),
        torch.as_tensor(value["evidence_metric"]),
        complex_,
    )
    state.validate()
    return LoadedTeacherBundle(state, confidence, provenance, payload)


__all__ = [
    "TeacherBundleConfig",
    "TeacherBundleResult",
    "LoadedTeacherBundle",
    "TopologyFixedTeacherBundleRefiner",
    "load_teacher_bundle",
]

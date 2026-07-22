"""Complete static-3D GRAFT-GS reference data flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.profiler import record_function

from ..equivariant.gsta import (
    GSTAConfig,
    GaugeCovariantSparseTransportAttention,
    IrrepTensor,
    MultiplicityLinear,
    active_adjacency,
)
from ..geometry.atlas import AtlasConfig, PersistentOctreeAtlas
from ..manifold.barrier import BarrierConfig, BarrierProjector, FeasibilityReport
from ..manifold.flow import FlowConfig, RiemannianVectorField, SafeHeunIntegrator
from ..manifold.geometry import ManifoldState, spectral_box_spd
from ..mapping.manifold_mapping import (
    EvidenceParticles,
    GeometricEvidenceBuilder,
    ManifoldMappingConfig,
    ManifoldMappingOperator,
    MappingResult,
    sparse_view_reprojection_variance,
)
from ..readout.assets import AnalyticalReadoutConfig, AnalyticalSurfaceReadout, GaussianAsset, MeshAsset, write_gaussian_ply, write_mesh_glb
from ..readout.renderer import CameraBatch, CudaGaussianRenderer, ReferenceGaussianRenderer, RenderResult
from ..topology.strata import TopologySelection, TopologySelector, TopologySelectorConfig
from .vggt_adapter import (
    CameraAlignmentDiagnostics,
    VGGTAdapter,
    VGGTGeometryOutput,
    align_vggt_to_supervised_cameras,
)
from .trellis_prior import TrellisPriorAdapter, TrellisPriorMeasure


@dataclass(frozen=True)
class RobustnessPerturbation:
    camera_rotation_std: float = 0.003
    camera_translation_std: float = 0.002
    depth_log_std: float = 0.02
    confidence_log_std: float = 0.05
    active_prune_probability: float = 0.03


@dataclass(frozen=True)
class GraftGSConfig:
    feature_dim: int = 1024
    atlas: AtlasConfig = field(default_factory=AtlasConfig)
    mapping: ManifoldMappingConfig = field(default_factory=ManifoldMappingConfig)
    attention: GSTAConfig = field(default_factory=GSTAConfig)
    topology: TopologySelectorConfig = field(default_factory=TopologySelectorConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    barrier: BarrierConfig = field(default_factory=BarrierConfig)
    readout: AnalyticalReadoutConfig = field(default_factory=AnalyticalReadoutConfig)
    encoder_layers: int = 4
    transport_feature_iterations: int = 2
    refinement_rounds: int = 1
    run_flow: bool = True
    renderer_backend: str = "cuda"

    def __post_init__(self) -> None:
        if self.feature_dim < 1 or self.encoder_layers < 1:
            raise ValueError("feature width and encoder depth must be positive")
        if self.transport_feature_iterations < 1 or self.refinement_rounds < 0:
            raise ValueError("transport iterations must be positive and refinement rounds non-negative")
        if self.renderer_backend not in {"cuda", "reference"}:
            raise ValueError("renderer_backend must be 'cuda' or 'reference'")


@dataclass
class SceneOutput:
    evidence: object
    atlas: PersistentOctreeAtlas
    mapping: MappingResult
    topology: TopologySelection
    initial_state: ManifoldState
    final_state: ManifoldState
    gaussians: Optional[GaussianAsset]
    mesh: Optional[MeshAsset]
    feasibility_reports: List[FeasibilityReport] = field(default_factory=list)
    collision_pairs: Optional[Tensor] = None
    collision_face_pairs: Optional[Tensor] = None
    render: Optional[RenderResult] = None
    render_cameras: Optional[CameraBatch] = None
    atlas_rejected_evidence_count: int = 0
    atlas_rejected_evidence_mass: float = 0.0
    trellis_prior_support_count: int = 0
    trellis_prior_expected_mass: float = 0.0
    trellis_prior_sample_count: int = 0
    topology_occupancy: Optional[Tensor] = None
    topology_evidence_occupancy: Optional[Tensor] = None
    topology_shape_prior_probability: Optional[Tensor] = None
    encoder_activations: Optional[List[IrrepTensor]] = None

    def export(self, directory: str | Path, stem: str = "graft_gs") -> tuple[Path, Path]:
        if self.gaussians is None or self.mesh is None:
            raise RuntimeError(
                "this execution stage stops before analytical asset construction"
            )
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        ply = directory / f"{stem}.ply"
        glb = directory / f"{stem}.glb"
        write_gaussian_ply(ply, self.gaussians)
        write_mesh_glb(glb, self.mesh)
        return ply, glb


@dataclass
class GraftGSOutput:
    vggt: VGGTGeometryOutput
    scenes: List[SceneOutput]
    camera_alignment: Optional[CameraAlignmentDiagnostics] = None
    evidence_particles: Optional[List[EvidenceParticles]] = None
    execution_stage: str = "full"


class GraftGS(nn.Module):
    def __init__(
        self,
        vggt: VGGTAdapter,
        config: GraftGSConfig = GraftGSConfig(),
        trellis_prior: Optional[TrellisPriorAdapter] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vggt = vggt
        self.evidence_builder = GeometricEvidenceBuilder()
        self.mapping = ManifoldMappingOperator(config.feature_dim, config.mapping)
        self.encoder = nn.ModuleList([GaugeCovariantSparseTransportAttention(config.attention) for _ in range(config.encoder_layers)])
        for layer in self.encoder:
            for child in layer.modules():
                if isinstance(child, MultiplicityLinear):
                    nn.utils.parametrizations.spectral_norm(child, name="weight", n_power_iterations=1)
        self.topology = TopologySelector(config.topology)
        self.vector_field = RiemannianVectorField(config.flow, config.attention)
        self.integrator = SafeHeunIntegrator(config.flow.steps)
        self.readout = AnalyticalSurfaceReadout(config.readout)
        if config.renderer_backend == "cuda":
            self.renderer = CudaGaussianRenderer()
        elif config.renderer_backend == "reference":
            self.renderer = ReferenceGaussianRenderer()
        else:
            raise ValueError("renderer_backend must be 'cuda' or 'reference'")
        self.trellis_prior = trellis_prior

    def forward(
        self,
        images: Tensor,
        valid_mask: Optional[Tensor] = None,
        render_input_views: bool = False,
        distributed_synchronizer: Optional[object] = None,
        robustness: Optional[RobustnessPerturbation] = None,
        ground_truth_extrinsics: Optional[Tensor] = None,
        ground_truth_intrinsics: Optional[Tensor] = None,
        atlas_root_bounds: Optional[Tensor] = None,
        execution_stage: str = "full",
        trellis_prior_seed: int = 0,
        capture_distillation_activations: bool = False,
    ) -> GraftGSOutput:
        valid_execution_stages = {
            "evidence_calibration",
            "atlas_autoencoding",
            "flow_pretraining",
            "full",
        }
        if execution_stage not in valid_execution_stages:
            raise ValueError(
                f"execution_stage must be one of {sorted(valid_execution_stages)}"
            )
        with record_function("graft_gs/vggt_geometry"):
            vggt_output = self.vggt(images)
        if robustness is not None:
            vggt_output = self._perturb_geometry(vggt_output, robustness)
        camera_alignment = None
        if (ground_truth_extrinsics is None) != (ground_truth_intrinsics is None):
            raise ValueError("ground-truth camera supervision requires both extrinsics and intrinsics")
        if ground_truth_extrinsics is not None and ground_truth_intrinsics is not None:
            vggt_output, camera_alignment = align_vggt_to_supervised_cameras(
                vggt_output,
                ground_truth_extrinsics,
                ground_truth_intrinsics,
            )
        with record_function("graft_gs/evidence_lift"):
            particles = self.evidence_builder(
                vggt_output.images,
                vggt_output.depth,
                vggt_output.depth_confidence,
                vggt_output.extrinsics_world_to_camera,
                vggt_output.intrinsics,
                vggt_output.patch_features,
                valid_mask=valid_mask,
            )
        if execution_stage == "evidence_calibration":
            if distributed_synchronizer is not None and hasattr(
                distributed_synchronizer, "aggregate_evidence"
            ):
                particles = [
                    distributed_synchronizer.aggregate_evidence(evidence)
                    for evidence in particles
                ]
            return GraftGSOutput(
                vggt=vggt_output,
                scenes=[],
                camera_alignment=camera_alignment,
                evidence_particles=particles,
                execution_stage=execution_stage,
            )
        scenes: List[SceneOutput] = []
        if atlas_root_bounds is not None:
            atlas_root_bounds = atlas_root_bounds.to(
                device=vggt_output.images.device,
                dtype=vggt_output.depth.dtype,
            )
            if atlas_root_bounds.shape != (len(particles), 2, 3):
                raise ValueError("atlas_root_bounds must have shape [B,2,3]")
        for batch_index, evidence in enumerate(particles):
            if distributed_synchronizer is not None and hasattr(
                distributed_synchronizer, "aggregate_evidence"
            ):
                evidence = distributed_synchronizer.aggregate_evidence(evidence)
            atlas_position, atlas_mass = evidence.positions, evidence.mass
            if distributed_synchronizer is not None and not getattr(
                distributed_synchronizer, "maps_global_evidence", False
            ):
                atlas_position, atlas_mass = distributed_synchronizer.aggregate_atlas_measure(
                    atlas_position, atlas_mass
                )
            root_bounds = None
            prior_measure: Optional[TrellisPriorMeasure] = None
            rejected_evidence_count = 0
            rejected_evidence_mass = 0.0
            if atlas_root_bounds is not None:
                root_min, root_max = atlas_root_bounds[batch_index]
                in_bounds = torch.all(
                    (atlas_position >= root_min) & (atlas_position <= root_max), dim=-1
                )
                if not bool(torch.any(in_bounds)):
                    raise RuntimeError("no VGGT evidence lies inside the audited canonical atlas root")
                rejected_evidence_count = int((~in_bounds).sum().item())
                rejected_evidence_mass = float(atlas_mass[~in_bounds].sum().detach().cpu())
                atlas_position = atlas_position[in_bounds]
                atlas_mass = atlas_mass[in_bounds]
                root_bounds = (root_min, root_max)
            if self.trellis_prior is not None:
                if root_bounds is None:
                    root_bounds = PersistentOctreeAtlas.root_bounds_from_positions(
                        atlas_position,
                        self.config.atlas,
                    )
                with record_function("graft_gs/trellis_structure_prior"):
                    prior_images = vggt_output.images[batch_index]
                    if distributed_synchronizer is not None and hasattr(
                        distributed_synchronizer, "aggregate_prior_images"
                    ):
                        prior_images = distributed_synchronizer.aggregate_prior_images(
                            prior_images
                        )
                    should_sample = (
                        distributed_synchronizer is None
                        or not hasattr(
                            distributed_synchronizer,
                            "should_sample_trellis_prior",
                        )
                        or distributed_synchronizer.should_sample_trellis_prior()
                    )
                    if should_sample:
                        prior = self.trellis_prior.sample(
                            prior_images,
                            seed=int(trellis_prior_seed) + batch_index,
                        )
                        prior_measure = self.trellis_prior.support_measure(
                            prior,
                            root_bounds[0],
                            root_bounds[1],
                        )
                    if distributed_synchronizer is not None and hasattr(
                        distributed_synchronizer, "synchronize_trellis_prior_measure"
                    ):
                        prior_measure = distributed_synchronizer.synchronize_trellis_prior_measure(
                            prior_measure,
                            dtype=root_bounds[0].dtype,
                        )
            with record_function("graft_gs/atlas_initialize"):
                atlas = PersistentOctreeAtlas.from_evidence(
                    atlas_position,
                    atlas_mass,
                    self.config.atlas,
                    root_bounds=root_bounds,
                    prior_positions=(
                        prior_measure.positions if prior_measure is not None else None
                    ),
                    prior_mass=(prior_measure.mass if prior_measure is not None else None),
                    prior_mass_variance=(
                        prior_measure.mass_variance
                        if prior_measure is not None
                        else None
                    ),
                )
            if distributed_synchronizer is not None:
                atlas = distributed_synchronizer.synchronize_atlas(atlas)
            with record_function("graft_gs/sparse_uot_mapping"):
                mapping = self._map_with_feature_fixed_point(atlas, evidence)
            for _ in range(self.config.refinement_rounds):
                occupancy_entropy, reprojection_variance = self._refinement_statistics(
                    atlas,
                    mapping,
                    include_trellis_prior=prior_measure is not None,
                )
                split = atlas.refinement_mask(
                    occupancy_entropy=occupancy_entropy,
                    reprojection_variance=reprojection_variance,
                )
                if distributed_synchronizer is not None:
                    split = distributed_synchronizer.synchronize_split_mask(split)
                if not bool(torch.any(split)):
                    break
                with record_function("graft_gs/atlas_refine"):
                    atlas.refine(
                        atlas_position,
                        atlas_mass,
                        split_mask=split,
                        prior_positions=(
                            prior_measure.positions if prior_measure is not None else None
                        ),
                        prior_mass=(prior_measure.mass if prior_measure is not None else None),
                        prior_mass_variance=(
                            prior_measure.mass_variance
                            if prior_measure is not None
                            else None
                        ),
                    )
                if distributed_synchronizer is not None:
                    atlas = distributed_synchronizer.synchronize_atlas(atlas)
                with record_function("graft_gs/sparse_uot_remap"):
                    mapping = self._map_with_feature_fixed_point(atlas, evidence)
            if distributed_synchronizer is not None and not getattr(
                distributed_synchronizer, "maps_global_evidence", False
            ):
                mapping = distributed_synchronizer.reduce_mapping_statistics(mapping, atlas)
            with record_function("graft_gs/gauge_sparse_attention"):
                fields = IrrepTensor.from_packed(mapping.latent)
                encoder_activations = [fields] if capture_distillation_activations else None
                edge_ot_cost, edge_uncertainty = self._attention_edge_evidence(
                    atlas, mapping
                )
                for layer in self.encoder:
                    fields = layer(
                        atlas,
                        fields,
                        edge_ot_cost=edge_ot_cost,
                        edge_uncertainty=edge_uncertainty,
                    )
                    if encoder_activations is not None:
                        encoder_activations.append(fields)
                mapping.latent = fields.pack()
            area = torch.pi * atlas.chart_radii[mapping.graph.atlas_node_index].square()
            observed_occupancy = -torch.expm1(-mapping.transported_mass / area.clamp_min(1.0e-8))
            occupancy = observed_occupancy
            shape_prior_probability = None
            if robustness is not None and robustness.active_prune_probability > 0:
                prune = torch.rand_like(occupancy) < robustness.active_prune_probability
                occupancy = torch.where(prune, occupancy.new_full(occupancy.shape, 1.0e-6), occupancy)
            if self.trellis_prior is not None and prior_measure is not None:
                prior_probability = self.trellis_prior.node_probability(atlas)
                shape_prior_probability = self.trellis_prior.node_shape_probability(
                    atlas, prior_measure.sample_count
                )
                occupancy = self.trellis_prior.combine_observed_probability(occupancy, prior_probability)
            with record_function("graft_gs/topology_stratum"):
                topology = self.topology(
                    atlas,
                    occupancy.clamp(1.0e-6, 1.0 - 1.0e-6),
                    evidence_probability=observed_occupancy,
                    shape_prior_probability=shape_prior_probability,
                )
                topology, initial, projector, initial_report = self._select_feasible_stratum(
                    atlas, mapping, topology, occupancy
                )
            feasibility_reports = [initial_report]
            run_continuous_flow = self.config.run_flow and execution_stage in {
                "flow_pretraining",
                "full",
            }
            if run_continuous_flow:
                with record_function("graft_gs/barrier_riemannian_flow"):
                    final, integration_reports = self.integrator.integrate(
                        self.vector_field, atlas, initial, projector
                    )
                feasibility_reports.extend(integration_reports)
            else:
                final = initial
            gaussians: Optional[GaussianAsset] = None
            mesh: Optional[MeshAsset] = None
            if execution_stage != "flow_pretraining":
                with record_function("graft_gs/analytical_readout"):
                    gaussians, mesh = self.readout(atlas, final, mapping)
            render = None
            render_cameras = None
            if render_input_views:
                if gaussians is None:
                    raise ValueError(
                        "render_input_views is incompatible with flow_pretraining"
                    )
                render_extrinsics = (
                    ground_truth_extrinsics[batch_index].to(
                        device=vggt_output.images.device,
                        dtype=vggt_output.extrinsics_world_to_camera.dtype,
                    )
                    if ground_truth_extrinsics is not None
                    else vggt_output.extrinsics_world_to_camera[batch_index]
                )
                render_intrinsics = (
                    ground_truth_intrinsics[batch_index].to(
                        device=vggt_output.images.device,
                        dtype=vggt_output.intrinsics.dtype,
                    )
                    if ground_truth_intrinsics is not None
                    else vggt_output.intrinsics[batch_index]
                )
                camera = CameraBatch(
                    render_extrinsics,
                    render_intrinsics,
                    int(vggt_output.images.shape[-2]),
                    int(vggt_output.images.shape[-1]),
                )
                render_cameras = camera
                with record_function("graft_gs/gaussian_render"):
                    render = self.renderer(gaussians, camera)
            scenes.append(
                SceneOutput(
                    evidence=evidence,
                    atlas=atlas,
                    mapping=mapping,
                    topology=topology,
                    initial_state=initial,
                    final_state=final,
                    gaussians=gaussians,
                    mesh=mesh,
                    feasibility_reports=feasibility_reports,
                    collision_pairs=projector.nonlocal_pairs,
                    collision_face_pairs=projector.nonlocal_face_pairs,
                    render=render,
                    render_cameras=render_cameras,
                    atlas_rejected_evidence_count=rejected_evidence_count,
                    atlas_rejected_evidence_mass=rejected_evidence_mass,
                    trellis_prior_support_count=(
                        int(prior_measure.positions.shape[0])
                        if prior_measure is not None
                        else 0
                    ),
                    trellis_prior_expected_mass=(
                        float(prior_measure.mass.sum().detach().cpu())
                        if prior_measure is not None
                        else 0.0
                    ),
                    trellis_prior_sample_count=(
                        prior_measure.sample_count if prior_measure is not None else 0
                    ),
                    topology_occupancy=occupancy,
                    topology_evidence_occupancy=observed_occupancy,
                    topology_shape_prior_probability=shape_prior_probability,
                    encoder_activations=encoder_activations,
                )
            )
        return GraftGSOutput(
            vggt=vggt_output,
            scenes=scenes,
            camera_alignment=camera_alignment,
            evidence_particles=particles,
            execution_stage=execution_stage,
        )

    def _select_feasible_stratum(
        self,
        atlas: PersistentOctreeAtlas,
        mapping: MappingResult,
        topology: TopologySelection,
        occupancy: Tensor,
    ) -> tuple[TopologySelection, ManifoldState, BarrierProjector, FeasibilityReport]:
        """Choose the minimum-energy candidate inside the hard feasible set."""

        energy = torch.stack([candidate.total_energy for candidate in topology.candidates])
        failures: list[str] = []
        for index in torch.argsort(energy).tolist():
            topology.selected_index = int(index)
            candidate = topology.selected
            if (
                not candidate.manifold_incidence_valid
                or not candidate.orientation_consistent
                or not candidate.complex.manifold_incidence_valid()
                or not candidate.complex.orientation_consistent()
            ):
                failures.append(
                    f"{candidate.identifier}: invalid incidence/orientation"
                )
                continue
            state = self._state_from_mapping(
                atlas,
                mapping,
                topology,
                occupancy_probability=occupancy,
            )
            projector = BarrierProjector(state, self.config.barrier)
            report = projector.report(state)
            if report.feasible:
                return topology, state, projector, report
            failures.append(f"{topology.selected.identifier}: {report}")
        raise RuntimeError(
            "no proposed topology stratum has a strictly feasible transported embedding: "
            + "; ".join(failures)
        )

    def _map_with_feature_fixed_point(self, atlas: PersistentOctreeAtlas, evidence: object) -> MappingResult:
        mapping = self.mapping(atlas, evidence)
        for _ in range(max(0, self.config.transport_feature_iterations - 1)):
            invariant_scalar = mapping.latent[:, : self.config.attention.scalar_channels]
            projected_dim = self.config.mapping.feature_cost_dim
            if invariant_scalar.shape[-1] < projected_dim:
                invariant_scalar = torch.nn.functional.pad(
                    invariant_scalar, (0, projected_dim - invariant_scalar.shape[-1])
                )
            atlas_feature = invariant_scalar[:, :projected_dim]
            mapping = self.mapping(atlas, evidence, atlas_features=atlas_feature)
        return mapping

    @staticmethod
    def _attention_edge_evidence(
        atlas: PersistentOctreeAtlas,
        mapping: MappingResult,
    ) -> tuple[Tensor, Tensor]:
        r"""Lift bipartite UOT diagnostics onto atlas connection edges.

        For chart ``i`` the scalar

        ``c_i = (sum_j pi_ij C_ij) / (sum_j pi_ij)``

        is its conditional transport cost.  An undirected connection edge uses
        the symmetric mean ``(c_i+c_j)/2``; uncertainty analogously uses one
        minus the geometric mean of observation reliabilities.  Self edges are
        retained with their node values.  Both quantities are invariant under
        global SE(3) and local chart-gauge changes, so they are valid scalar
        attention biases.  ``log1p`` keeps high-cost outliers numerically
        bounded without detaching the transport/cost gradient path.
        """

        edge, active = active_adjacency(atlas)
        if not torch.equal(active, mapping.graph.atlas_node_index):
            raise RuntimeError(
                "transport rows and active atlas adjacency use inconsistent node ordering"
            )
        source = mapping.graph.source
        node_cost_numerator = mapping.cost.new_zeros(mapping.graph.source_count)
        node_cost_numerator.index_add_(
            0,
            source,
            mapping.plan * mapping.cost,
        )
        epsilon = torch.finfo(mapping.plan.dtype).eps
        node_cost = node_cost_numerator / mapping.transported_mass.clamp_min(epsilon)
        node_cost = torch.log1p(node_cost.clamp_min(0.0))
        reliability = mapping.observation_reliability.clamp(0.0, 1.0)
        edge_source, edge_target = edge
        edge_ot_cost = 0.5 * (
            node_cost[edge_source] + node_cost[edge_target]
        )
        edge_uncertainty = 1.0 - torch.sqrt(
            (reliability[edge_source] * reliability[edge_target]).clamp_min(0.0)
        )
        return edge_ot_cost, edge_uncertainty

    def _refinement_statistics(
        self,
        atlas: PersistentOctreeAtlas,
        mapping: MappingResult,
        include_trellis_prior: bool,
    ) -> tuple[Tensor, Tensor]:
        """Return active-chart entropy and exact sparse image disagreement.

        The evidence contract retains calibrated camera tables, allowing the
        plan-conditional chart observation in every supporting view to be
        compared in the image plane.  Residuals are measured in projected-cell
        units so the split threshold is dimensionless across resolution and
        focal length.
        """

        active = mapping.graph.atlas_node_index
        area = torch.pi * atlas.chart_radii[active].square()
        observed_probability = -torch.expm1(
            -mapping.transported_mass / area.clamp_min(1.0e-8)
        )
        probability = observed_probability
        if include_trellis_prior:
            if self.trellis_prior is None:
                raise RuntimeError("TRELLIS prior statistics requested without an adapter")
            probability = self.trellis_prior.combine_observed_probability(
                observed_probability,
                self.trellis_prior.node_probability(atlas),
            )
        probability = probability.clamp(1.0e-6, 1.0 - 1.0e-6)
        entropy = -(
            probability * torch.log(probability)
            + (1.0 - probability) * torch.log1p(-probability)
        )

        return entropy, sparse_view_reprojection_variance(atlas, mapping)

    @staticmethod
    def _perturb_geometry(output: VGGTGeometryOutput, perturbation: RobustnessPerturbation) -> VGGTGeometryOutput:
        from ..manifold.geometry import so3_exp

        extrinsics = output.extrinsics_world_to_camera.clone()
        rotation_noise = torch.randn_like(extrinsics[..., :3, 3]) * perturbation.camera_rotation_std
        translation_noise = torch.randn_like(extrinsics[..., :3, 3]) * perturbation.camera_translation_std
        delta_rotation = so3_exp(rotation_noise)
        extrinsics[..., :3, :3] = delta_rotation @ extrinsics[..., :3, :3]
        extrinsics[..., :3, 3] = torch.einsum("bkij,bkj->bki", delta_rotation, extrinsics[..., :3, 3]) + translation_noise
        depth = output.depth * torch.exp(torch.randn_like(output.depth) * perturbation.depth_log_std)
        confidence = output.depth_confidence * torch.exp(
            torch.randn_like(output.depth_confidence) * perturbation.confidence_log_std
        )
        return VGGTGeometryOutput(
            output.images,
            output.patch_features,
            extrinsics,
            output.intrinsics,
            depth,
            confidence,
            output.world_points,
            output.world_points_confidence,
        )

    @staticmethod
    def _state_from_mapping(
        atlas: PersistentOctreeAtlas,
        mapping: MappingResult,
        topology: TopologySelection,
        occupancy_probability: Optional[Tensor] = None,
    ) -> ManifoldState:
        complex_ = topology.selected.complex
        mapping_lookup = {int(node): i for i, node in enumerate(mapping.graph.atlas_node_index.tolist())}
        row = torch.tensor([mapping_lookup[int(node)] for node in complex_.atlas_node_index.tolist()], dtype=torch.int64, device=mapping.latent.device)
        position = mapping.transported_centers[row]
        rotation = atlas.chart_frames[complex_.atlas_node_index]
        metric = mapping.riemannian_metric[row]
        covariance_raw = torch.linalg.inv(metric)
        covariance = spectral_box_spd(covariance_raw, 1.0e-6, 0.25)
        if occupancy_probability is None:
            occupancy = -torch.expm1(
                -mapping.transported_mass[row]
                / (
                    torch.pi
                    * atlas.chart_radii[complex_.atlas_node_index].square()
                ).clamp_min(1.0e-8)
            )
        else:
            if occupancy_probability.shape != mapping.transported_mass.shape:
                raise ValueError(
                    "occupancy_probability must have one value per mapped active chart"
                )
            occupancy = occupancy_probability[row]
        opacity_logit = torch.logit(occupancy.clamp(1.0e-5, 1.0 - 1.0e-5))[:, None]
        appearance = mapping.latent.new_zeros((row.numel(), 48))
        if mapping.transported_color is not None:
            appearance[:, :3] = torch.logit(
                mapping.transported_color[row].clamp(1.0e-4, 1.0 - 1.0e-4)
            )
        state = ManifoldState(
            position=position,
            rotation=rotation,
            covariance=covariance,
            opacity_logit=opacity_logit,
            appearance=appearance,
            latent=mapping.latent[row],
            evidence_metric=metric,
            complex=complex_,
        )
        state.validate()
        return state


__all__ = ["GraftGS", "GraftGSConfig", "GraftGSOutput", "RobustnessPerturbation", "SceneOutput"]

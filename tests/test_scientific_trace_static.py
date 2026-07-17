"""PyTorch-independent guards for production-path scientific integration.

These are static tests, not numerical validation. They prevent a small set of
previously verified bypasses from silently returning after refactors.
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf8")


class ScientificProductionTraceStaticTest(unittest.TestCase):
    def test_attention_receives_transport_and_uncertainty_biases(self) -> None:
        pipeline = source("graft_gs/integration/pipeline.py")
        self.assertIn("edge_ot_cost=edge_ot_cost", pipeline)
        self.assertIn("edge_uncertainty=edge_uncertainty", pipeline)
        self.assertIn("mapping.plan * mapping.cost", pipeline)

    def test_refined_chart_fit_is_not_decorated_no_grad(self) -> None:
        tree = ast.parse(source("graft_gs/geometry/atlas.py"))
        methods = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods[node.name] = node
        for name in ("refine", "fit_active_charts"):
            decorators = [ast.unparse(value) for value in methods[name].decorator_list]
            self.assertNotIn("torch.no_grad()", decorators)
        atlas = source("graft_gs/geometry/atlas.py")
        self.assertIn("self.chart_centers.index_copy", atlas)
        self.assertIn("def partition_of_unity_metric", atlas)

    def test_training_phases_stop_at_their_required_stage(self) -> None:
        trainer = source("graft_gs/engine/trainer.py")
        expected = {
            'TrainingPhase.EVIDENCE_CALIBRATION: "evidence_calibration"',
            'TrainingPhase.ATLAS_AUTOENCODING: "atlas_autoencoding"',
            'TrainingPhase.RIEMANNIAN_FLOW: "flow_pretraining"',
        }
        for contract in expected:
            self.assertIn(contract, trainer)
        pipeline = source("graft_gs/integration/pipeline.py")
        self.assertIn('if execution_stage != "flow_pretraining"', pipeline)

    def test_checkpoint_format_has_rank_local_rng_and_objective(self) -> None:
        trainer = source("graft_gs/engine/trainer.py")
        self.assertIn('"format_version": 5', trainer)
        self.assertIn('"rank_rng_states": rank_rng_states', trainer)
        self.assertIn('"loss_weights": asdict(self.loss.weights)', trainer)
        self.assertIn("exact trainer resume requires the checkpoint world size", trainer)

    def test_topology_and_barrier_admissibility_are_hard_checks(self) -> None:
        topology = source("graft_gs/topology/strata.py")
        self.assertIn("def _orient_faces_consistently", topology)
        self.assertIn("not complex_.orientation_consistent()", topology)
        barrier = source("graft_gs/manifold/barrier.py")
        self.assertIn("minimum_linearized_margin", barrier)
        self.assertIn("control-barrier QP did not satisfy", barrier)

    def test_phase_f_uses_production_gradient_purification(self) -> None:
        trainer = source("graft_gs/engine/trainer.py")
        purifier = source("graft_gs/optimization/gradient_purification.py")
        self.assertIn("_backward_with_gradient_purification", trainer)
        self.assertIn("stable_objective = total - global_view_objective", trainer)
        self.assertIn("self._synchronize_gradients()", trainer)
        self.assertIn("def weighted_geometric_median", purifier)
        self.assertIn("def principal_subspace_projection", purifier)
        self.assertIn("fisher_norm_square", purifier)

    def test_phase_f_inner_maximizes_scale_and_dimensionless_margins(self) -> None:
        trainer = source("graft_gs/engine/trainer.py")
        quantization = source("graft_gs/optimization/quantization.py")
        losses = source("graft_gs/engine/losses.py")
        self.assertIn("set_worst_case_from_gradient", trainer)
        self.assertIn("self._restore_forward_rng(forward_rng_state)", trainer)
        self.assertIn("adversarial_log_scale_radius", quantization)
        self.assertIn("area / barrier_config.minimum_face_area - 1.0", losses)
        self.assertIn("relative_hardening_margin - all_margin", losses)

    def test_phase_e_captures_irreps_and_matches_manifold_jacobian(self) -> None:
        pipeline = source("graft_gs/integration/pipeline.py")
        trainer = source("graft_gs/engine/trainer.py")
        losses = source("graft_gs/engine/losses.py")
        geometry = source("graft_gs/manifold/geometry.py")
        self.assertIn("encoder_activations=encoder_activations", pipeline)
        self.assertIn("capture_distillation_activations=True", trainer)
        self.assertIn("gauge_covariant_activation_distillation", losses)
        self.assertIn("vector_field_jacobian_distillation", losses)
        self.assertIn("def spd_parallel_transport", geometry)

    def test_vggt_tracks_and_normals_are_derived_without_fake_heads(self) -> None:
        losses = source("graft_gs/engine/losses.py")
        adapter = source("graft_gs/integration/vggt_adapter.py")
        self.assertIn("def multiview_reprojection_cycle_loss", losses)
        self.assertIn("def vggt_depth_normal_field", losses)
        self.assertIn('terms["vggt_track_cycle"]', losses)
        self.assertIn('terms["vggt_depth_normal"]', losses)
        self.assertNotIn("fabricated_track_target", adapter)

    def test_offline_teacher_is_topology_fixed_and_dataset_typed(self) -> None:
        refiner = source("graft_gs/engine/teacher_refinement.py")
        dataset = source("graft_gs/data/meshfleet.py")
        script = source("scripts/refine_teacher_bundle.py")
        self.assertIn("BarrierProjector(self.base_state", refiner)
        self.assertIn("self.model.readout", refiner)
        self.assertIn("teacher_refined_fixed_stratum", refiner)
        self.assertIn("teacher_bundle_supervision_mask", dataset)
        self.assertIn("write_gaussian_ply", script)
        self.assertIn("write_mesh_glb", script)

    def test_learned_perceptual_path_is_hash_pinned_and_never_downloads(self) -> None:
        losses = source("graft_gs/engine/losses.py")
        trainer = source("graft_gs/engine/trainer.py")
        self.assertIn("class LearnedPerceptualPyramid", losses)
        self.assertIn("hashlib.sha256(path.read_bytes()).hexdigest()", losses)
        self.assertIn("vgg16(weights=None)", losses)
        self.assertIn("perceptual_checkpoint_sha256", trainer)

    def test_trellis_shape_prior_is_separate_from_observed_evidence(self) -> None:
        pipeline = source("graft_gs/integration/pipeline.py")
        topology = source("graft_gs/topology/strata.py")
        prior = source("graft_gs/integration/trellis_prior.py")
        self.assertIn("observed_occupancy =", pipeline)
        self.assertIn("evidence_probability=observed_occupancy", pipeline)
        self.assertIn("shape_prior_probability=shape_prior_probability", pipeline)
        self.assertIn("torch.log1p(-shape_prior)", topology)
        self.assertIn("def node_shape_probability", prior)

    def test_octree_refinement_uses_retained_camera_reprojection(self) -> None:
        mapping = source("graft_gs/mapping/manifold_mapping.py")
        pipeline = source("graft_gs/integration/pipeline.py")
        trainer = source("graft_gs/engine/trainer.py")
        self.assertIn("extrinsics_world_to_camera: Tensor", mapping)
        self.assertIn("def sparse_view_reprojection_variance", mapping)
        self.assertIn("projected_pixel - observed_pixel", mapping)
        self.assertIn("sparse_view_reprojection_variance(atlas, mapping)", pipeline)
        self.assertIn("gather_cameras(evidence.intrinsics)", trainer)
        self.assertNotIn("normalized view-disagreement proxy", pipeline)

    def test_topology_proposals_prioritize_persistence_events(self) -> None:
        topology = source("graft_gs/topology/strata.py")
        ablations = source("scripts/run_ablations.py")
        self.assertIn("def persistence_critical_occupancy_thresholds", topology)
        self.assertIn('(\"ph-critical\", critical_thresholds)', topology)
        self.assertIn("proposal_persistence = persistent_homology", topology)
        self.assertIn("crossing that event is", topology)
        self.assertIn("maximum_persistence_thresholds=0", ablations)

    def test_quantization_certificate_computes_metric_boundary_margin(self) -> None:
        barrier = source("graft_gs/manifold/barrier.py")
        quantization = source("graft_gs/optimization/quantization.py")
        meshfleet_inference = source("scripts/infer_meshfleet.py")
        overfit = source("scripts/overfit_meshfleet_object.py")
        self.assertIn("def topology_boundary_margin", barrier)
        self.assertIn('"va,vab,vb->"', barrier)
        self.assertIn("def certify_topology_quantization_step", quantization)
        self.assertIn("projector.topology_boundary_margin(state)", quantization)
        self.assertIn("quantization_topology_certificate", meshfleet_inference)
        self.assertNotIn("torch.inference_mode()", meshfleet_inference)
        self.assertNotIn("torch.inference_mode()", overfit)

    def test_flow_spectral_bound_is_not_dead_configuration(self) -> None:
        flow = source("graft_gs/manifold/flow.py")
        attention = source("graft_gs/equivariant/gsta.py")
        self.assertIn("child.set_operator_scale(config.spectral_bound)", flow)
        self.assertIn("self.operator_scale * torch.einsum", attention)

    def test_pytorch_24_reference_failures_have_production_repairs(self) -> None:
        atlas = source("graft_gs/geometry/atlas.py")
        losses = source("graft_gs/engine/losses.py")
        quantization = source("graft_gs/optimization/quantization.py")
        self.assertNotIn("torch.full_like(unique_codes, config.chart_radius_scale * side", atlas)
        self.assertIn("torch.ones_like(unique_codes, dtype=positions.dtype) * side", atlas)
        self.assertIn("torch.sqrt(world_squared + norm_epsilon.square())", losses)
        self.assertIn("torch.sqrt(pixel_squared + norm_epsilon.square())", losses)
        margin_position = quantization.index("margin = projector.topology_boundary_margin(state)")
        error_position = quantization.index("query_error_tensor = torch.as_tensor")
        self.assertLess(margin_position, error_position)
        self.assertIn("query_error, dtype=margin.dtype, device=margin.device", quantization)

    def test_server_validator_binds_exact_environment_and_remote_dataset(self) -> None:
        validator = source("scripts/validate_server.py")
        environment = source("scripts/validate_environment.py")
        self.assertIn("audit_environment(args.requirements)", validator)
        self.assertIn('[sys.executable, "-m", "pip", "check"]', validator)
        self.assertIn('record["accelerator"] = accelerator', validator)
        self.assertIn('details.get("torch_cuda") != "11.8"', validator)
        self.assertIn('"/mnt/sda2/hef/Base/dataset"', validator)
        self.assertNotIn("canonical schema ID", validator)
        self.assertIn("_inspect_manifest_contract(manifest, dataset_root, object_ids)", validator)
        self.assertIn("_manifest_requires_rebuild(args.rebuild_manifest, manifest_audit)", validator)
        self.assertIn("EXPECTED_MESHFLEET_SCHEMA", validator)
        self.assertIn('"--object-id-file"', validator)
        self.assertIn("manifest missing-ID inventory differs", validator)
        self.assertIn("dynamic dataset discovery produced an empty manifest", validator)
        self.assertIn('test_environment["GRAFT_GS_MESHFLEET_ROOT"]', validator)
        self.assertIn("unexpected_skip_reasons", validator)
        self.assertIn("requirements_sha256", environment)

    def test_six_rank_validator_records_distinct_a800_contract(self) -> None:
        validator = source("scripts/validate_ddp_server.py")
        self.assertIn("world_size != 6", validator)
        self.assertIn("audit_environment(args.requirements)", validator)
        self.assertIn("_accelerator_contract_errors(accelerator_details)", validator)
        self.assertIn("len(set(rank_keys)) != world_size", validator)
        self.assertIn("torch.cuda.set_device(local_rank)", validator)
        self.assertIn("successful_on_every_rank", validator)
        self.assertIn("dist.all_reduce(success, op=dist.ReduceOp.MIN)", validator)

    def test_six_gpu_training_launcher_cannot_bypass_pinned_interpreter(self) -> None:
        launcher = source("scripts/launch_a800_6gpu.sh")
        self.assertIn("/mnt/sda1/miniforge3/envs/CRAFT/bin/python", launcher)
        self.assertIn('"$ROOT/scripts/validate_environment.py"', launcher)
        self.assertIn('--requirements "$ROOT/requirements.txt"', launcher)
        self.assertIn('"$PYTHON_BIN" -m torch.distributed.run', launcher)
        self.assertNotIn("\ntorchrun \\", launcher)

    def test_training_uses_catalog_filtered_complete_meshfleet_records(self) -> None:
        meshfleet = source("graft_gs/data/meshfleet.py")
        trainer = source("scripts/train_a800.py")
        config = source("configs/graft_gs_a800_native.yaml")
        self.assertIn("def load_meshfleet_object_ids", meshfleet)
        self.assertIn("def meshfleet_record_admission_reasons", meshfleet)
        self.assertIn("object is absent from configured ID catalog", meshfleet)
        self.assertIn("dataset_object_id_catalog_sha256=object_id_digest", trainer)
        self.assertIn("dataset_coverage_", trainer)
        self.assertNotIn("object_id_file:", config)


if __name__ == "__main__":
    unittest.main()

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

    def test_implicit_sinkhorn_requires_forward_and_adjoint_convergence(self) -> None:
        mapping = source("graft_gs/mapping/manifold_mapping.py")
        configuration = source("graft_gs/engine/configuration.py")
        config = source("configs/graft_gs_a800_native.yaml")
        self.assertIn("sparse unbalanced Sinkhorn did not converge", mapping)
        self.assertIn("implicit Sinkhorn adjoint did not converge", mapping)
        self.assertIn("equation_source = lambda_source", mapping)
        self.assertIn("torch.linalg.solve_triangular", mapping)
        self.assertNotIn("precision = torch.linalg.inv(covariance)", mapping)
        tree = ast.parse(mapping)
        graph_builder = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "build_sparse_transport_graph"
        )
        self.assertIn(
            "torch.no_grad()",
            [ast.unparse(value) for value in graph_builder.decorator_list],
        )
        self.assertIn("convergence_check_interval", configuration)
        self.assertIn("convergence_check_interval: 8", config)

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
        self.assertIn('"format_version": 6', trainer)
        self.assertIn('"rank_rng_states": rank_rng_states', trainer)
        self.assertIn('"loss_weights": asdict(self.loss.weights)', trainer)
        self.assertIn("exact trainer resume requires the checkpoint world size", trainer)
        self.assertIn('"precision_float32_matmul"', trainer)

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

    def test_released_model_adapters_preserve_upstream_runtime_contracts(self) -> None:
        vggt = source("graft_gs/integration/vggt_adapter.py")
        trellis = source("graft_gs/integration/trellis_prior.py")
        external = source("graft_gs/integration/external.py")
        self.assertIn('import_external_module("vggt.models.vggt")', vggt)
        self.assertIn('import_external_module("vggt.utils.pose_enc")', vggt)
        self.assertIn("released four VGGT cached taps", vggt)
        self.assertIn("VGGT tensor inputs must use the released [0,1] RGB contract", vggt)
        self.assertIn('import_external_module("trellis.pipelines")', trellis)
        self.assertIn('mode="multidiffusion"', trellis)
        self.assertIn("TRELLIS tensor inputs must use the released [0,1] RGB contract", trellis)
        self.assertIn("decoded_resolutions.append(_decoded_structure_resolution(output))", trellis)
        self.assertIn("len(set(decoded_resolutions)) != 1", trellis)
        self.assertIn('values.view(torch.uint8).numpy().tobytes(order="C")', trellis)
        self.assertIn("self._sample_cache.popitem(last=False)", trellis)
        self.assertIn('DEFAULT_VGGT_CHECKPOINT = "facebook/VGGT-1B"', external)
        self.assertIn(
            'DEFAULT_TRELLIS_CHECKPOINT = "microsoft/TRELLIS-image-large"',
            external,
        )
        self.assertIn(
            'DEFAULT_VGGT_REPOSITORY_ROOT = Path("/mnt/sda2/hef/Base/vggt")',
            external,
        )
        self.assertIn(
            'DEFAULT_TRELLIS_REPOSITORY_ROOT = Path("/mnt/sda2/hef/Base/TRELLIS")',
            external,
        )
        import_boundary = external.index("def import_external_module")
        explicit_position = external.index("configured = repository_root", import_boundary)
        environment_position = external.index("configured = os.environ.get", import_boundary)
        default_position = external.index(
            "_DEFAULT_REPOSITORY_ROOT[package] / package", import_boundary
        )
        import_position = external.index(
            "module = importlib.import_module(module_name)", import_boundary
        )
        self.assertLess(explicit_position, environment_position)
        self.assertLess(environment_position, default_position)
        self.assertLess(default_position, import_position)

        tree = ast.parse(trellis)
        sample = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "sample"
        )
        sample_source = ast.get_source_segment(trellis, sample)
        self.assertIsNotNone(sample_source)
        self.assertNotIn(
            'models["sparse_structure_flow_model"].resolution',
            sample_source,
        )
        posterior_loop = next(
            node
            for node in ast.walk(sample)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "sample_index"
        )

        def injection_calls(node: ast.AST) -> list[ast.Call]:
            return [
                value
                for value in ast.walk(node)
                if isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "inject_sampler_multi_image"
            ]

        self.assertEqual(len(injection_calls(sample)), 1)
        self.assertEqual(len(injection_calls(posterior_loop)), 1)

    def test_same_object_ddp_samples_frozen_trellis_only_on_source_rank(self) -> None:
        trainer = source("graft_gs/engine/trainer.py")
        pipeline = source("graft_gs/integration/pipeline.py")
        distributed_test = source("tests/test_distributed_evidence.py")
        overfit = source("scripts/overfit_meshfleet_object.py")
        self.assertIn("def should_sample_trellis_prior", trainer)
        self.assertIn("non-source TRELLIS rank must not sample", trainer)
        self.assertIn("distributed_synchronizer.should_sample_trellis_prior()", pipeline)
        self.assertIn("prior_measure,\n                            dtype=root_bounds[0].dtype", pipeline)
        self.assertIn("if context.rank == 0\n            else None", distributed_test)
        self.assertIn("synchronize_object_atlas=True", overfit)
        self.assertIn("value.to(dtype=torch.int64, copy=True).contiguous()", trainer)
        self.assertIn(
            "dist_nn.broadcast(value.contiguous(), src=self.source_rank)",
            trainer,
        )
        self.assertIn("gauge_coordinate_fields", trainer)
        self.assertNotIn("reference + (value - value.detach())", trainer)
        self.assertIn("DDP atlas metadata mismatch before typed collectives", trainer)
        validator = source("scripts/validate_ddp_server.py")
        self.assertIn(
            "test_pca_frame_repeated_spectrum_has_finite_zero_gauge_gradient",
            validator,
        )
        self.assertIn(
            "test_pca_frame_distinct_spectrum_retains_finite_gradient",
            validator,
        )
        self.assertIn(
            "test_isotropic_chart_metric_has_finite_basis_free_backward",
            validator,
        )
        self.assertIn(
            "test_flat_chart_analytical_readout_backward_is_finite",
            validator,
        )
        self.assertIn(
            "test_spd_spectral_box_is_bounded_and_repeated_spectrum_safe",
            validator,
        )
        self.assertIn(
            "test_atlas_rejects_nonfinite_mass_with_specific_diagnostic",
            validator,
        )

    def test_remote_entry_points_resolve_checkpoint_environment_at_runtime(self) -> None:
        scripts = (
            "scripts/train_a800.py",
            "scripts/evaluate_meshfleet.py",
            "scripts/infer_meshfleet.py",
            "scripts/infer_multiview.py",
            "scripts/refine_teacher_bundle.py",
            "scripts/run_ablations.py",
            "scripts/overfit_meshfleet_object.py",
        )
        for script in scripts:
            content = source(script)
            self.assertIn("resolve_vggt_checkpoint", content, script)
            self.assertIn("resolve_trellis_checkpoint", content, script)
            self.assertNotIn("default=DEFAULT_VGGT_CHECKPOINT", content, script)
            self.assertNotIn("default=DEFAULT_TRELLIS_CHECKPOINT", content, script)
        for subtree in ("graft_gs", "scripts", "configs"):
            for path in (ROOT / subtree).rglob("*"):
                if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".sh"}:
                    content = path.read_text(encoding="utf8")
                    self.assertNotIn("D:\\VsCode", content, str(path))

    def test_external_preflight_selects_one_manifest_record_before_loading(self) -> None:
        content = source("scripts/validate_external_models.py")
        self.assertIn("load_meshfleet_manifest(manifest)", content)
        self.assertIn("meshfleet_record_admission_reasons(record, candidate_config)", content)
        self.assertIn("include_object_ids=(record.object_id,)", content)
        self.assertIn("MeshFleetObjectDataset(selected_config)", content)
        self.assertNotIn('split="train"', content)
        self.assertNotIn('split="test"', content)

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
        configuration = source("graft_gs/engine/configuration.py")
        server_config = source("configs/graft_gs_a800_native.yaml")
        losses = source("graft_gs/engine/losses.py")
        quantization = source("graft_gs/optimization/quantization.py")
        readout = source("graft_gs/readout/assets.py")
        pipeline = source("graft_gs/integration/pipeline.py")
        trainer = source("graft_gs/engine/trainer.py")
        self.assertNotIn("torch.full_like(unique_codes, config.chart_radius_scale * side", atlas)
        self.assertIn("torch.ones_like(unique_codes, dtype=positions.dtype) * side", atlas)
        self.assertIn("relative_eigengap * spectral_scale", atlas)
        self.assertIn("diagnostic_vectors + 0.0 * flat.sum", atlas)
        self.assertIn('"frame_relative_eigengap"', configuration)
        self.assertIn("frame_relative_eigengap: 1.0e-4", server_config)
        self.assertIn("_stratified_metric_eigh", readout)
        self.assertIn("tangent_factor.square()", readout)
        self.assertNotIn("torch.linalg.eigh(first_form)", readout)
        self.assertIn("spectral_box_spd(covariance_raw", pipeline)
        self.assertIn("def _clip_grad_norm_high_precision", trainer)
        self.assertIn("non-finite training tensors before", trainer)
        self.assertNotIn("torch.nn.utils.clip_grad_norm_", trainer)
        self.assertIn("post-step optimizer state", trainer)
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
        self.assertIn('record["upstream_repositories"] = upstream_repositories', validator)
        self.assertIn('"entrypoint_sha256"', validator)
        self.assertIn('"package_init_sha256"', validator)
        self.assertIn('details.get("torch_cuda") != "11.8"', validator)
        self.assertIn(
            '"/mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97"',
            validator,
        )
        self.assertNotIn("canonical schema ID", validator)
        self.assertIn("_inspect_manifest_contract(manifest, dataset_root, object_ids)", validator)
        self.assertIn("_manifest_requires_rebuild(args.rebuild_manifest, manifest_audit)", validator)
        self.assertIn("EXPECTED_MESHFLEET_SCHEMA", validator)
        self.assertIn('"--object-id-file"', validator)
        self.assertIn("manifest missing-ID inventory differs", validator)
        self.assertIn("dynamic dataset discovery produced an empty manifest", validator)
        self.assertIn('test_environment["GRAFT_GS_MESHFLEET_ROOT"]', validator)
        self.assertIn('"validate_external_models.py"', validator)
        self.assertIn('for component in ("vggt", "trellis")', validator)
        self.assertIn("unexpected_skip_reasons", validator)
        self.assertIn("requirements_sha256", environment)

    def test_visible_rank_validator_records_distinct_a800_contract(self) -> None:
        validator = source("scripts/validate_ddp_server.py")
        self.assertIn("visible_device_count != world_size", validator)
        self.assertIn('"visible_device_count": visible_device_count', validator)
        self.assertIn('"cuda_visible_devices": visible_device_mask', validator)
        self.assertIn('"multi_rank": world_size > 1', validator)
        self.assertIn("audit_environment(args.requirements)", validator)
        self.assertIn("_accelerator_contract_errors(accelerator_details)", validator)
        self.assertIn("len(set(rank_keys)) != world_size", validator)
        self.assertIn("torch.cuda.set_device(local_rank)", validator)
        self.assertIn("successful_on_every_rank", validator)
        self.assertIn("dist.all_reduce(success, op=dist.ReduceOp.MIN)", validator)
        self.assertIn("test_metric_minimal_restoration_enters_strict_feasible_set", validator)
        self.assertIn("test_adjoint_nonconvergence_is_not_silently_accepted", validator)

    def test_visible_gpu_training_launcher_cannot_bypass_pinned_interpreter(self) -> None:
        launcher = source("scripts/launch_a800_6gpu.sh")
        config = source("configs/graft_gs_a800_native.yaml")
        self.assertIn("/mnt/sda1/miniforge3/envs/CRAFT/bin/python", launcher)
        self.assertIn('"$ROOT/scripts/validate_environment.py"', launcher)
        self.assertIn('--requirements "$ROOT/requirements.txt"', launcher)
        self.assertIn('"$PYTHON_BIN" -m torch.distributed.run', launcher)
        self.assertIn("torch.cuda.device_count()", launcher)

    def test_a800_concurrency_uses_useful_views_and_early_rank_binding(self) -> None:
        trainer = source("graft_gs/engine/trainer.py")
        overfit = source("scripts/overfit_meshfleet_object.py")
        training = source("scripts/train_a800.py")
        launcher = source("scripts/launch_a800_6gpu.sh")
        config = source("configs/graft_gs_a800_native.yaml")
        self.assertIn("def bind_local_cuda_device", trainer)
        self.assertIn("def assert_local_cuda_allocator_ownership", trainer)
        self.assertIn("assert_local_cuda_allocator_ownership(self.context.device)", trainer)
        self.assertLess(
            overfit.index("device = bind_local_cuda_device(require_cuda=True)"),
            overfit.index("TrellisPriorAdapter.from_pretrained"),
        )
        self.assertLess(
            training.index("local_device = bind_local_cuda_device(require_cuda=True)"),
            training.index("TrellisPriorAdapter.from_pretrained"),
        )
        self.assertIn('"--views-per-rank"', overfit)
        self.assertIn("maximum_views = args.views_per_rank * world_size", overfit)
        self.assertIn('"--evaluation-views"', overfit)
        self.assertIn('"rank_performance": rank_performance', overfit)
        self.assertIn('"initial_feasibility": scene.feasibility_reports[0].__dict__', overfit)
        self.assertIn('"final_feasibility": scene.feasibility_reports[-1].__dict__', overfit)
        self.assertIn('"fixed_point_residual": transport.fixed_point_residual', overfit)
        selector = source("scripts/select_a800_view_budget.py")
        protocol = source("docs/A800_VALIDATION_PROTOCOL.md")
        self.assertIn("maximum_reserved_fraction", selector)
        self.assertIn("final feasibility certificate is missing", selector)
        self.assertIn("sparse transport is not certified converged", selector)
        self.assertIn("16 24 32 48 64", protocol)
        self.assertIn("select_a800_view_budget.py", protocol)
        self.assertIn("if trainer.context.rank != 0:", overfit)
        self.assertIn('"--maximum-views"', training)
        self.assertIn("dataset_maximum_views=maximum_views", training)
        self.assertLess(
            trainer.index("images, valid_mask, view_supervision = self._shard_object_views"),
            trainer.index("images = images.to("),
        )
        self.assertNotIn('torch.as_tensor(batch["images"], device=', trainer)
        self.assertGreaterEqual(
            trainer.count("images, valid_mask, view_supervision = self._shard_object_views"),
            2,
        )
        self.assertIn("non_blocking=True", trainer)
        self.assertIn("peak_reserved_memory_bytes", trainer)
        self.assertIn("local_views_per_second", trainer)
        self.assertIn("def _clip_grad_norm_high_precision", trainer)
        self.assertIn("dtype=torch.float64", trainer)
        save_start = trainer.index("def save_checkpoint")
        load_start = trainer.index("def load_checkpoint")
        checkpoint_source = trainer[save_start:load_start]
        self.assertLess(
            checkpoint_source.index("torch.save(payload, temporary)"),
            checkpoint_source.index("dist.broadcast(failed, src=0)"),
        )
        self.assertIn("distributed checkpoint commit failed", checkpoint_source)
        barrier = source("graft_gs/manifold/barrier.py")
        pipeline = source("graft_gs/integration/pipeline.py")
        configuration = source("graft_gs/engine/configuration.py")
        self.assertIn("def restore_feasible_embedding", barrier)
        self.assertIn("metric-minimal hard-constraint steps", barrier)
        self.assertIn("dual_check_interval", barrier)
        self.assertIn("diagnostic_minima = torch.stack", barrier)
        self.assertIn("projector.restore_feasible_embedding(state)", pipeline)
        self.assertIn('barrier = data.get("barrier", {})', configuration)
        self.assertIn("restoration_relative_margin", config)
        self.assertIn("--dataloader-workers", training)
        self.assertIn("dataloader_prefetch_factor", training)
        self.assertIn("--minimum-global-object-batch", training)
        self.assertIn("args.minimum_global_object_batch + world_size - 1", training)
        self.assertIn("find_unused_parameters: false", config)
        self.assertIn("dataloader_workers: 8", config)
        self.assertIn("dataloader_prefetch_factor: 4", config)
        self.assertIn("CUDA_VISIBLE_DEVICES must name the scheduler-assigned idle GPU subset", launcher)
        self.assertIn('--nproc-per-node="$NPROC_PER_NODE"', launcher)
        self.assertNotIn("--nproc-per-node=6", launcher)
        self.assertNotIn("world_size:", config)
        self.assertIn(
            "GRAFT_GS_VGGT_ROOT=${GRAFT_GS_VGGT_ROOT:-/mnt/sda2/hef/Base/vggt}",
            launcher,
        )
        self.assertIn(
            "GRAFT_GS_TRELLIS_ROOT=${GRAFT_GS_TRELLIS_ROOT:-/mnt/sda2/hef/Base/TRELLIS}",
            launcher,
        )
        self.assertNotIn("\ntorchrun \\", launcher)

    def test_a800_precision_and_mip_renderer_contracts_are_production_applied(self) -> None:
        precision = source("graft_gs/engine/precision.py")
        renderer = source("graft_gs/readout/renderer.py")
        barrier = source("graft_gs/manifold/barrier.py")
        trainer_entry = source("scripts/train_a800.py")
        config = source("configs/graft_gs_a800_native.yaml")
        self.assertIn("torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32", precision)
        self.assertIn("torch.set_float32_matmul_precision", precision)
        self.assertIn("precision_policy.apply()", trainer_entry)
        self.assertIn("backbone_dtype=precision_policy.backbone_dtype", trainer_entry)
        self.assertIn("float32_matmul_precision: highest", config)
        self.assertIn("allow_tf32: false", config)
        self.assertIn("cov3D_precomp=covariance_packed", renderer)
        self.assertIn("kernel_size=self.contract.kernel_size", renderer)
        self.assertIn("(2.0 * intrinsic[0, 2] + 1.0) / width", renderer)
        self.assertNotIn("rotations=quaternion", renderer)
        self.assertIn("state.position.detach().to(dtype=torch.float64)", barrier)
        self.assertIn("state.evidence_metric.detach().to(dtype=torch.float64)", barrier)

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

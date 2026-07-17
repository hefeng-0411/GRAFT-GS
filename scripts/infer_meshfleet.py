"""Checkpoint-backed inference on an audited MeshFleet/TRELLIS object record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from graft_gs.data import MeshFleetDatasetConfig, MeshFleetObjectDataset, meshfleet_single_object_collate
from graft_gs.engine import (
    load_graft_checkpoint,
    load_server_config,
    load_trellis_prior_config,
    validate_trellis_prior_policy,
)
from graft_gs.engine.losses import symmetric_surface_chamfer
from graft_gs.integration import GraftGS, TrellisPriorAdapter, VGGTAdapter
from graft_gs.manifold import BarrierProjector
from graft_gs.optimization import certify_topology_quantization_step


def _select_record(dataset: MeshFleetObjectDataset, object_id: str | None) -> int:
    if object_id is None:
        if len(dataset) != 1:
            raise ValueError("--object-id is required when the selected split has multiple usable records")
        return 0
    matches = [index for index, record in enumerate(dataset.records) if record.object_id == object_id]
    if not matches:
        raise ValueError(f"object {object_id!r} is absent or unusable in the selected split/view set")
    return matches[0]


def _save_rgb(path: Path, value: torch.Tensor) -> None:
    array = (
        value.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(np.asarray(array)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("graft_checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--object-id")
    parser.add_argument("--object-id-file", type=Path)
    parser.add_argument("--view-set", default="renders")
    parser.add_argument("--maximum-views", type=int, default=12)
    parser.add_argument("--vggt-checkpoint", default="facebook/VGGT-1B")
    parser.add_argument("--trellis-checkpoint")
    parser.add_argument("--disable-trellis-prior", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/graft_gs_a800_native.yaml"))
    parser.add_argument("--profile-trace", type=Path)
    parser.add_argument("--quantization-query-error", type=float)
    parser.add_argument("--vector-field-lipschitz-bound", type=float)
    args = parser.parse_args()
    if (args.quantization_query_error is None) != (
        args.vector_field_lipschitz_bound is None
    ):
        raise ValueError(
            "quantization certification requires both --quantization-query-error "
            "and --vector-field-lipschitz-bound"
        )

    model_config, _, _, dataset_config = load_server_config(args.config)
    configured_id_file = dataset_config.get("object_id_file")
    object_id_file = args.object_id_file or (
        Path(str(configured_id_file)) if configured_id_file is not None else None
    )
    prior_config = load_trellis_prior_config(args.config)
    use_prior = bool(prior_config["enabled_after_phase_a"]) and not args.disable_trellis_prior
    if use_prior and args.trellis_checkpoint is None:
        raise ValueError("inference requires --trellis-checkpoint unless the prior is explicitly disabled")
    image_size = dataset_config.get("image_size", [518, 518])
    dataset = MeshFleetObjectDataset(
        MeshFleetDatasetConfig(
            root=args.dataset_root,
            manifest=args.manifest,
            object_id_file=object_id_file,
            split=args.split,
            input_view_set=args.view_set,
            image_size=(int(image_size[0]), int(image_size[1])),
            minimum_views=2,
            maximum_views=args.maximum_views,
            view_selection="uniform",
            foreground_alpha_threshold=float(
                dataset_config.get("foreground_alpha_threshold", 0.5)
            ),
            surface_grid_resolution=int(dataset_config.get("surface_grid_resolution", 64)),
            load_surface_voxels=True,
            require_surface_voxels=True,
            require_requested_modalities=bool(
                dataset_config.get("require_requested_modalities", True)
            ),
            require_complete_input_view_set=bool(
                dataset_config.get("require_complete_input_view_set", True)
            ),
            require_normalization=bool(dataset_config.get("require_normalization", True)),
            require_render_mesh=bool(dataset_config.get("require_render_mesh", False)),
            topology_supervision_mode=str(
                dataset_config.get("topology_supervision_mode", "validated_or_repaired")
            ),
            minimum_topology_confidence=float(
                dataset_config.get("minimum_topology_confidence", 0.95)
            ),
        )
    )
    sample = dataset[_select_record(dataset, args.object_id)]
    batch = meshfleet_single_object_collate([sample])

    device = torch.device("cuda")
    prior = (
        TrellisPriorAdapter.from_pretrained(
            args.trellis_checkpoint,
            samples=int(prior_config["samples"]),
            sampler_steps=int(prior_config["sampler_steps"]),
            strength=float(prior_config["strength"]),
            minimum_probability=float(prior_config["minimum_probability"]),
            uncertainty_discount=float(prior_config["uncertainty_discount"]),
            device=device,
        )
        if use_prior
        else None
    )
    model = GraftGS(
        VGGTAdapter.from_pretrained(args.vggt_checkpoint, feature_dim=model_config.feature_dim),
        model_config,
        prior,
    )
    checkpoint_payload, checkpoint_report = load_graft_checkpoint(
        model,
        args.graft_checkpoint,
        map_location="cpu",
        strict=True,
    )
    trainer_metadata = checkpoint_payload.get("trainer_config", {})
    validate_trellis_prior_policy(
        checkpoint_payload,
        enabled=use_prior,
        samples=int(prior_config["samples"]),
        sampler_steps=int(prior_config["sampler_steps"]),
        strength=float(prior_config["strength"]),
        minimum_probability=float(prior_config["minimum_probability"]),
        uncertainty_discount=float(prior_config["uncertainty_discount"]),
    )
    if isinstance(trainer_metadata, dict):
        expected_manifest = trainer_metadata.get("dataset_manifest_sha256")
        actual_manifest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        if expected_manifest is not None and expected_manifest != actual_manifest:
            raise ValueError(
                "inference manifest differs from the dataset contract recorded in the checkpoint"
            )
    model = model.to(device).eval()
    images = batch["images"].to(device=device, dtype=torch.float32, non_blocking=True)
    evidence_mask = batch["evidence_mask"].to(device=device, non_blocking=True)
    extrinsics = batch["extrinsics_world_to_camera"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    intrinsics = batch["intrinsics"].to(device=device, dtype=torch.float32, non_blocking=True)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    profile_context = (
        torch.profiler.profile(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        if args.profile_trace is not None
        else torch.autograd.profiler.record_function("graft_gs/inference")
    )
    # Barrier projection and its exact JVP/Jacobian checks temporarily enable
    # autograd internally. `inference_mode` would create inference tensors that
    # cannot participate in those checks; `no_grad` avoids retaining the model
    # graph while preserving the certified local differentiation path.
    with torch.no_grad(), profile_context as profile:
        prior_seed = int.from_bytes(
            hashlib.sha256(str(sample["object_id"]).encode("utf8")).digest()[:8],
            "little",
        ) % (2**31 - 1)
        output = model(
            images,
            valid_mask=evidence_mask,
            render_input_views=True,
            ground_truth_extrinsics=extrinsics,
            ground_truth_intrinsics=intrinsics,
            atlas_root_bounds=batch["atlas_root_bounds"].to(device),
            trellis_prior_seed=prior_seed,
        )
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    if args.profile_trace is not None:
        args.profile_trace.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(args.profile_trace))
    scene = output.scenes[0]
    quantization_topology_certificate = None
    if args.quantization_query_error is not None:
        attention_temperature = min(
            model_config.attention.attention_temperature_scalar,
            model_config.attention.attention_temperature_vector,
            model_config.attention.attention_temperature_tensor,
        )
        integration_step = 1.0 / model_config.flow.steps
        certificate = certify_topology_quantization_step(
            BarrierProjector(scene.initial_state, model_config.barrier),
            scene.final_state,
            query_error=args.quantization_query_error,
            temperature=attention_temperature,
            vector_field_lipschitz=args.vector_field_lipschitz_bound,
            step_size=integration_step,
        )
        quantization_topology_certificate = certificate.to_dict() | {
            "attention_temperature": attention_temperature,
            "integration_step": integration_step,
            "lipschitz_bound_provenance": "explicit_server_measurement",
        }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    ply, glb = scene.export(args.output_directory, str(sample["object_id"]))
    render_directory = args.output_directory / "renders"
    render_directory.mkdir(parents=True, exist_ok=True)
    render_paths = []
    if scene.render is None:
        raise RuntimeError("MeshFleet inference requested input-view renders but produced none")
    for local_index, color in enumerate(scene.render.color):
        frame_index = int(sample["frame_indices"][local_index])
        path = render_directory / f"frame-{frame_index:04d}.png"
        _save_rgb(path, color)
        render_paths.append(str(path))
    target_surface = sample["surface_voxel_centers"].to(
        device=device, dtype=scene.gaussians.means.dtype
    )
    surface_chamfer = float(
        symmetric_surface_chamfer(scene.gaussians.means, target_surface).detach().cpu()
    )
    alignment = output.camera_alignment
    metrics = {
        "object_id": sample["object_id"],
        "split": sample["split"],
        "view_set": sample["view_set"],
        "frame_indices": sample["frame_indices"].tolist(),
        "seconds": seconds,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "profile_trace": str(args.profile_trace) if args.profile_trace is not None else None,
        "checkpoint": checkpoint_report.__dict__,
        "camera_alignment": {
            "center_rmse": alignment.center_rmse.tolist(),
            "rotation_geodesic": alignment.rotation_geodesic.tolist(),
            "intrinsic_log_focal_error": alignment.intrinsic_log_focal_error.tolist(),
            "scale": alignment.scale.tolist(),
        }
        if alignment is not None
        else None,
        "surface_chamfer_squared": surface_chamfer,
        "evidence_particles": scene.evidence.positions.shape[0],
        "atlas_rejected_evidence_count": scene.atlas_rejected_evidence_count,
        "atlas_rejected_evidence_mass": scene.atlas_rejected_evidence_mass,
        "trellis_prior_support_count": scene.trellis_prior_support_count,
        "trellis_prior_expected_mass": scene.trellis_prior_expected_mass,
        "trellis_prior_sample_count": scene.trellis_prior_sample_count,
        "prior_only_active_charts": int(
            torch.sum(
                (scene.atlas.evidence_mass[scene.atlas.active_indices] == 0)
                & (scene.atlas.prior_mass[scene.atlas.active_indices] > 0)
            ).item()
        ),
        "observation_reliability_mean": float(
            scene.mapping.observation_reliability.mean().detach().cpu()
        ),
        "active_charts": scene.atlas.num_active,
        "transport_edges": scene.mapping.graph.num_edges,
        "transport_iterations": scene.mapping.diagnostics.iterations,
        "transport_residual": scene.mapping.diagnostics.fixed_point_residual,
        "selected_topology": scene.topology.selected.identifier,
        "selected_topology_shape_prior_energy": float(
            scene.topology.selected.prior_energy.detach().cpu()
        ),
        "shape_prior_active": scene.topology_shape_prior_probability is not None,
        "betti": scene.topology.selected.betti,
        "feasibility": scene.feasibility_reports[-1].__dict__,
        "quantization_topology_certificate": quantization_topology_certificate,
        "gaussians": scene.gaussians.means.shape[0],
        "mesh_faces": scene.mesh.faces.shape[0],
        "ply": str(ply),
        "glb": str(glb),
        "renders": render_paths,
        "dataset_warnings": sample["dataset_warnings"],
        "topology_supervision": sample["topology_supervision"],
        "topology_supervision_active": bool(sample["topology_supervision_mask"]),
        "topology_target_masks": {
            "betti": bool(sample["topology_betti_supervision_mask"]),
            "persistence": bool(sample["topology_persistence_supervision_mask"]),
            "stratum": bool(sample["topology_stratum_supervision_mask"]),
            "source_manifold_certification": bool(
                sample["source_manifold_certification_mask"]
            ),
        },
        "topology_label_provenance": sample["topology_label_provenance"],
        "topology_activation_reason": sample["topology_activation_reason"],
    }
    metrics_path = args.output_directory / "meshfleet_inference_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

"""Create a robust topology-fixed analytical teacher pseudo-asset bundle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from graft_gs.data import (
    MeshFleetDatasetConfig,
    MeshFleetObjectDataset,
    meshfleet_single_object_collate,
)
from graft_gs.engine import (
    TeacherBundleConfig,
    TopologyFixedTeacherBundleRefiner,
    load_graft_checkpoint,
    load_server_config,
    load_trellis_prior_config,
    validate_trellis_prior_policy,
)
from graft_gs.engine.losses import multiview_reprojection_cycle_loss
from graft_gs.integration import (
    GraftGS,
    TrellisPriorAdapter,
    VGGTAdapter,
    resolve_trellis_checkpoint,
    resolve_vggt_checkpoint,
)
from graft_gs.readout.assets import write_gaussian_ply, write_mesh_glb


def _record_index(dataset: MeshFleetObjectDataset, object_id: str) -> int:
    matches = [
        index
        for index, record in enumerate(dataset.records)
        if record.object_id == object_id
    ]
    if len(matches) != 1:
        raise ValueError(f"object {object_id!r} is absent or duplicated in the manifest")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("graft_checkpoint", type=Path)
    parser.add_argument("object_id")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--object-id-file", type=Path)
    parser.add_argument("--maximum-views", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--vggt-checkpoint")
    parser.add_argument("--trellis-checkpoint")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/graft_gs_a800_native.yaml")
    )
    args = parser.parse_args()
    args.vggt_checkpoint = resolve_vggt_checkpoint(args.vggt_checkpoint)

    model_config, _, _, dataset_config = load_server_config(args.config)
    configured_id_file = dataset_config.get("object_id_file")
    object_id_file = args.object_id_file or (
        Path(str(configured_id_file)) if configured_id_file is not None else None
    )
    prior_config = load_trellis_prior_config(args.config)
    use_prior = bool(prior_config["enabled_after_phase_a"])
    if use_prior:
        args.trellis_checkpoint = resolve_trellis_checkpoint(args.trellis_checkpoint)
    image_size = dataset_config.get("image_size", [518, 518])
    dataset = MeshFleetObjectDataset(
        MeshFleetDatasetConfig(
            root=args.dataset_root,
            manifest=args.manifest,
            object_id_file=object_id_file,
            split=args.split,
            input_view_set=str(dataset_config.get("input_view_set", "renders")),
            image_size=(int(image_size[0]), int(image_size[1])),
            minimum_views=2,
            maximum_views=args.maximum_views,
            view_selection="uniform",
            foreground_alpha_threshold=float(
                dataset_config.get("foreground_alpha_threshold", 0.5)
            ),
            surface_grid_resolution=int(
                dataset_config.get("surface_grid_resolution", 64)
            ),
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
        )
    )
    sample = dataset[_record_index(dataset, args.object_id)]
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
        VGGTAdapter.from_pretrained(
            args.vggt_checkpoint, feature_dim=model_config.feature_dim
        ),
        model_config,
        prior,
    )
    payload, report = load_graft_checkpoint(
        model, args.graft_checkpoint, map_location="cpu", strict=True
    )
    validate_trellis_prior_policy(
        payload,
        enabled=use_prior,
        samples=int(prior_config["samples"]),
        sampler_steps=int(prior_config["sampler_steps"]),
        strength=float(prior_config["strength"]),
        minimum_probability=float(prior_config["minimum_probability"]),
        uncertainty_discount=float(prior_config["uncertainty_discount"]),
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    images = batch["images"].to(device=device, dtype=torch.float32)
    alpha = batch["alpha"].to(device=device, dtype=torch.float32)
    evidence_mask = batch["evidence_mask"].to(device=device)
    extrinsic = batch["extrinsics_world_to_camera"].to(
        device=device, dtype=torch.float32
    )
    intrinsic = batch["intrinsics"].to(device=device, dtype=torch.float32)
    seed = int.from_bytes(
        hashlib.sha256(args.object_id.encode("utf8")).digest()[:8], "little"
    ) % (2**31 - 1)
    with torch.no_grad():
        output = model(
            images,
            valid_mask=evidence_mask,
            render_input_views=True,
            ground_truth_extrinsics=extrinsic,
            ground_truth_intrinsics=intrinsic,
            atlas_root_bounds=batch["atlas_root_bounds"].to(device),
            trellis_prior_seed=seed,
        )
        cycle = multiview_reprojection_cycle_loss(output, evidence_mask)
    scene = output.scenes[0]
    if scene.render_cameras is None:
        raise RuntimeError("teacher initialization did not retain audited cameras")
    refinement_config = TeacherBundleConfig(
        iterations=args.iterations, learning_rate=args.learning_rate
    )
    refiner = TopologyFixedTeacherBundleRefiner(
        model, scene, scene.render_cameras, refinement_config
    ).to(device)
    result = refiner.refine(images[0], alpha[0], cycle)
    if not result.feasibility.feasible:
        raise RuntimeError("refined teacher bundle failed its final feasibility certificate")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    ply = args.output_directory / f"{args.object_id}.ply"
    glb = args.output_directory / f"{args.object_id}.glb"
    bundle = args.output_directory / f"{args.object_id}.teacher.pt"
    metadata_path = args.output_directory / f"{args.object_id}.teacher.json"
    write_gaussian_ply(ply, result.gaussians)
    write_mesh_glb(glb, result.mesh)
    state = result.state
    torch.save(
        {
            "schema": "graft_gs_teacher_bundle_v1",
            "object_id": args.object_id,
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "graft_checkpoint_sha256": hashlib.sha256(
                args.graft_checkpoint.read_bytes()
            ).hexdigest(),
            "checkpoint_phase": report.phase,
            "config": asdict(refinement_config),
            "teacher_confidence": result.teacher_confidence.cpu(),
            "topology_provenance": "teacher_refined_fixed_stratum",
            "state": {
                "position": state.position.cpu(),
                "rotation": state.rotation.cpu(),
                "covariance": state.covariance.cpu(),
                "opacity_logit": state.opacity_logit.cpu(),
                "appearance": state.appearance.cpu(),
                "latent": state.latent.cpu(),
                "evidence_metric": state.evidence_metric.cpu(),
                "atlas_node_index": state.complex.atlas_node_index.cpu(),
                "edges": state.complex.edges.cpu(),
                "faces": state.complex.faces.cpu(),
            },
            "cameras": {
                "extrinsics_world_to_camera": result.cameras.extrinsics_world_to_camera.cpu(),
                "intrinsics": result.cameras.intrinsics.cpu(),
                "height": result.cameras.height,
                "width": result.cameras.width,
            },
            "loss_history": result.loss_history,
        },
        bundle,
    )
    metadata = {
        "schema": "graft_gs_teacher_bundle_v1",
        "object_id": args.object_id,
        "teacher_confidence": float(result.teacher_confidence.cpu()),
        "reprojection_rmse": float(result.reprojection_rmse.cpu()),
        "topology_entropy": float(result.topology_entropy.cpu()),
        "cycle_residual": float(result.cycle_residual.cpu()),
        "feasibility": result.feasibility.__dict__,
        "topology": scene.topology.selected.identifier,
        "betti": scene.topology.selected.betti,
        "ply": str(ply),
        "glb": str(glb),
        "bundle": str(bundle),
        "iterations": args.iterations,
        "initial_loss": result.loss_history[0],
        "final_loss": result.loss_history[-1],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

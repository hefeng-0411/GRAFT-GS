"""One-object overfit using the audited MeshFleet supervision contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from graft_gs.data import MeshFleetDatasetConfig, MeshFleetObjectDataset, meshfleet_single_object_collate
from graft_gs.engine import (
    GraftGSTrainer,
    TrainerConfig,
    TrainingPhase,
    bind_local_cuda_device,
    load_loss_weights,
    load_precision_policy,
    load_server_config,
    load_trellis_prior_config,
)
from graft_gs.integration import (
    GraftGS,
    TrellisPriorAdapter,
    VGGTAdapter,
    resolve_trellis_checkpoint,
    resolve_vggt_checkpoint,
)


class RepeatedMapping:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def __iter__(self):
        while True:
            yield self.value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--object-id-file", type=Path)
    parser.add_argument("--object-id")
    parser.add_argument("--view-set", default="renders")
    parser.add_argument("--maximum-views", type=int)
    parser.add_argument(
        "--views-per-rank",
        type=int,
        help=(
            "same-object DDP view concurrency; the loader selects this value "
            "times WORLD_SIZE before deterministic rank sharding"
        ),
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--evaluation-views",
        type=int,
        default=24,
        help="deterministic view cap for rank-zero final asset evaluation",
    )
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.01)
    parser.add_argument("--vggt-checkpoint")
    parser.add_argument("--trellis-checkpoint")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/graft_gs_a800_native.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/meshfleet_overfit"))
    args = parser.parse_args()
    if args.steps < 1 or args.evaluation_views < 2:
        raise ValueError("--steps must be positive and --evaluation-views at least two")
    device = bind_local_cuda_device(require_cuda=True)
    args.vggt_checkpoint = resolve_vggt_checkpoint(args.vggt_checkpoint)

    model_config, training_config, distributed_config, dataset_config = (
        load_server_config(args.config)
    )
    precision_policy = load_precision_policy(args.config)
    precision_policy.apply()
    configured_id_file = dataset_config.get("object_id_file")
    object_id_file = args.object_id_file or (
        Path(str(configured_id_file)) if configured_id_file is not None else None
    )
    loss_weights = load_loss_weights(args.config)
    prior_config = load_trellis_prior_config(args.config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.maximum_views is not None and args.views_per_rank is not None:
        raise ValueError("choose --maximum-views or --views-per-rank, not both")
    if args.views_per_rank is not None:
        if args.views_per_rank < 2:
            raise ValueError("--views-per-rank must be at least two for camera alignment")
        maximum_views = args.views_per_rank * world_size
    else:
        maximum_views = int(
            args.maximum_views
            if args.maximum_views is not None
            else dataset_config.get("maximum_views", 24)
        )
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
            input_view_set=args.view_set,
            image_size=(int(image_size[0]), int(image_size[1])),
            minimum_views=2,
            maximum_views=maximum_views,
            view_selection="uniform",
            foreground_alpha_threshold=float(
                dataset_config.get("foreground_alpha_threshold", 0.5)
            ),
            surface_grid_resolution=int(dataset_config.get("surface_grid_resolution", 64)),
            load_surface_voxels=True,
            load_trellis_features=bool(
                dataset_config.get("load_trellis_features", False)
            ),
            load_trellis_latents=bool(
                dataset_config.get("load_trellis_latents", False)
            ),
            dino_pseudo_confidence=float(
                dataset_config.get("dino_pseudo_confidence", 0.5)
            ),
            trellis_latent_pseudo_confidence=float(
                dataset_config.get("trellis_latent_pseudo_confidence", 0.5)
            ),
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
    candidates = [
        index
        for index, record in enumerate(dataset.records)
        if args.object_id is None or record.object_id == args.object_id
    ]
    if len(candidates) != 1:
        raise ValueError("select exactly one usable record with --object-id")
    sample = meshfleet_single_object_collate([dataset[candidates[0]]])
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
            args.vggt_checkpoint,
            feature_dim=model_config.feature_dim,
            backbone_dtype=precision_policy.backbone_dtype,
        ),
        model_config,
        prior,
    )
    trainer = GraftGSTrainer(
        model,
        TrainerConfig(
            phase=TrainingPhase.ATLAS_AUTOENCODING,
            learning_rate=float(training_config.get("learning_rate", 1.0e-4)),
            maximum_gradient_norm=float(training_config.get("maximum_gradient_norm", 1.0)),
            find_unused_parameters=bool(
                distributed_config.get("find_unused_parameters", False)
            ),
            checkpoint_every=max(1, args.steps // 5),
            output_directory=str(args.output),
            # Every DDP rank receives this same object. Shard its views and
            # construct one global evidence/UOT/atlas state instead of treating
            # identical replicas as unrelated object-level batches.
            synchronize_object_atlas=True,
            dataset_manifest=str(args.manifest.resolve()),
            dataset_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            dataset_split=args.split,
            dataset_view_set=args.view_set,
            dataset_maximum_views=maximum_views,
            trellis_prior_checkpoint=args.trellis_checkpoint if use_prior else None,
            trellis_prior_samples=int(prior_config["samples"]) if use_prior else 0,
            trellis_prior_sampler_steps=int(prior_config["sampler_steps"])
            if use_prior
            else 0,
            trellis_prior_strength=float(prior_config["strength"]) if use_prior else 0.0,
            trellis_prior_minimum_probability=float(prior_config["minimum_probability"])
            if use_prior
            else 0.0,
            trellis_prior_uncertainty_discount=float(
                prior_config["uncertainty_discount"]
            )
            if use_prior
            else 0.0,
            precision_backbone=precision_policy.backbone,
            precision_geometric_state=precision_policy.geometric_state,
            precision_analytical_solve=precision_policy.analytical_solve,
            precision_diagnostics=precision_policy.diagnostics,
            precision_float32_matmul=precision_policy.float32_matmul_precision,
            precision_allow_tf32=precision_policy.allow_tf32,
            dino_relational_pseudo_supervision=bool(
                dataset_config.get("load_trellis_features", False)
            ),
            trellis_latent_relational_pseudo_supervision=bool(
                dataset_config.get("load_trellis_latents", False)
            ),
            dino_pseudo_confidence=float(
                dataset_config.get("dino_pseudo_confidence", 0.5)
            ),
            trellis_latent_pseudo_confidence=float(
                dataset_config.get("trellis_latent_pseudo_confidence", 0.5)
            ),
        ),
        loss_weights=loss_weights,
    )
    if args.initialize_from is not None:
        trainer.load_model_weights(args.initialize_from, strict=False)
    losses = []
    iterator = iter(RepeatedMapping(sample))
    for step in range(args.steps):
        metrics = trainer.train_step(next(iterator), step)
        trainer.microstep = step + 1
        trainer.batches_consumed_in_epoch = step + 1
        losses.append(metrics["total"])
        if trainer.global_step and trainer.global_step % trainer.config.checkpoint_every == 0:
            trainer.save_checkpoint(args.output / f"step-{trainer.global_step:08d}.pt")
    performance_fields = (
        "peak_memory_bytes",
        "peak_reserved_memory_bytes",
        "peak_allocated_fraction",
        "peak_reserved_fraction",
        "local_views",
        "local_views_per_second",
        "seconds",
    )
    local_performance = torch.tensor(
        [metrics[name] for name in performance_fields],
        device=trainer.context.device,
        dtype=torch.float64,
    )
    if trainer.context.distributed:
        gathered_performance = [
            torch.empty_like(local_performance)
            for _ in range(trainer.context.world_size)
        ]
        dist.all_gather(gathered_performance, local_performance)
    else:
        gathered_performance = [local_performance]
    rank_performance = [
        {
            "rank": rank,
            "logical_cuda_device": rank,
            **{
                name: float(value)
                for name, value in zip(performance_fields, rank_value.cpu().tolist())
            },
        }
        for rank, rank_value in enumerate(gathered_performance)
    ]
    window = min(10, len(losses))
    initial = sum(losses[:window]) / window
    final = sum(losses[-window:]) / window
    improvement = (initial - final) / max(abs(initial), 1.0e-12)
    trainer.save_checkpoint(args.output / "final.pt")
    if trainer.context.rank != 0:
        return
    trainer.module.eval()
    device = trainer.context.device
    total_evaluation_views = int(sample["images"].shape[1])
    evaluation_view_count = min(args.evaluation_views, total_evaluation_views)
    evaluation_index = torch.linspace(
        0,
        total_evaluation_views - 1,
        evaluation_view_count,
        dtype=torch.float64,
    ).round().to(torch.int64)
    evaluation_images = sample["images"].index_select(1, evaluation_index)
    evaluation_mask = sample["evidence_mask"].index_select(1, evaluation_index)
    evaluation_extrinsics = sample["extrinsics_world_to_camera"].index_select(
        1, evaluation_index
    )
    evaluation_intrinsics = sample["intrinsics"].index_select(1, evaluation_index)
    # Full GRAFT-GS evaluation includes barrier JVP/Jacobian checks. Keep the
    # ordinary graph disabled while allowing those local certified derivatives.
    with torch.no_grad():
        output = trainer.module(
            evaluation_images.to(device),
            valid_mask=evaluation_mask.to(device),
            render_input_views=True,
            ground_truth_extrinsics=evaluation_extrinsics.to(device),
            ground_truth_intrinsics=evaluation_intrinsics.to(device),
            atlas_root_bounds=sample["atlas_root_bounds"].to(device),
            trellis_prior_seed=trainer._trellis_prior_seed(sample),
        )
    scene = output.scenes[0]
    ply, glb = scene.export(args.output, "meshfleet_overfit")
    transport = scene.mapping.diagnostics
    summary = {
        "object_id": sample["object_id"],
        "steps": trainer.global_step,
        "initial_window_loss": initial,
        "final_window_loss": final,
        "relative_improvement": improvement,
        "losses": losses,
        "world_size": trainer.context.world_size,
        "global_training_views": maximum_views,
        "evaluation_views": evaluation_view_count,
        "rank_performance": rank_performance,
        "transport": {
            "iterations": transport.iterations,
            "fixed_point_residual": transport.fixed_point_residual,
            "effective_tolerance": transport.effective_tolerance,
            "converged": transport.converged,
            "minimum_source_transport_mass": float(
                transport.source_transport_mass.detach().amin().cpu()
            ),
            "minimum_target_transport_mass": float(
                transport.target_transport_mass.detach().amin().cpu()
            ),
            "objective": float(transport.objective.detach().cpu()),
        },
        "selected_topology": {
            "identifier": scene.topology.selected.identifier,
            "betti": list(scene.topology.selected.betti),
        },
        "initial_feasibility": scene.feasibility_reports[0].__dict__,
        "final_feasibility": scene.feasibility_reports[-1].__dict__,
        "ply": str(ply),
        "glb": str(glb),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "overfit_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf8"
    )
    if improvement < args.minimum_relative_improvement:
        raise RuntimeError(
            f"MeshFleet one-object overfit improved {improvement:.6f}, below required "
            f"{args.minimum_relative_improvement:.6f}"
        )


if __name__ == "__main__":
    main()

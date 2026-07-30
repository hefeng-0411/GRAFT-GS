"""torchrun entry point for native-precision visible-A800 staged training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from graft_gs.data import (
    DistributedViewCountBatchSampler,
    FolderMultiviewDataset,
    MANIFEST_SCHEMA,
    MeshFleetDatasetConfig,
    MeshFleetObjectDataset,
    load_meshfleet_object_ids,
    meshfleet_object_id_digest,
    meshfleet_object_collate,
    meshfleet_single_object_collate,
    folder_object_collate,
    single_object_collate,
    ViewCountBatchSampler,
)
from graft_gs.engine import (
    GraftGSTrainer,
    TrainerConfig,
    TrainingPhase,
    bind_local_cuda_device,
    load_graft_checkpoint,
    load_loss_weights,
    load_precision_policy,
    load_server_config,
    load_trellis_prior_config,
    validate_precision_policy,
)
from graft_gs.integration import (
    GraftGS,
    TrellisPriorAdapter,
    VGGTAdapter,
    resolve_trellis_checkpoint,
    resolve_vggt_checkpoint,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--object-id-file", type=Path)
    parser.add_argument("--dataset-format", choices=("auto", "meshfleet", "folders"), default="auto")
    parser.add_argument("--split", default="train")
    parser.add_argument("--phase", choices=list("ABCDEF"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--vggt-checkpoint")
    parser.add_argument("--trellis-checkpoint")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--teacher-bundle-root", type=Path)
    parser.add_argument("--perceptual-checkpoint", type=Path)
    parser.add_argument("--output", default="outputs/training")
    parser.add_argument("--same-object-view-shards", action="store_true")
    parser.add_argument(
        "--maximum-views",
        type=int,
        help=(
            "maximum images loaded per object before optional same-object view "
            "sharding (defaults to dataset.maximum_views)"
        ),
    )
    parser.add_argument("--dataloader-workers", type=int)
    parser.add_argument("--dataloader-prefetch-factor", type=int)
    parser.add_argument(
        "--object-batch-size",
        type=int,
        help=(
            "real objects processed per rank and forward pass; objects are "
            "grouped by exact view count so VGGT inputs are never padded"
        ),
    )
    parser.add_argument(
        "--batch-probe",
        type=Path,
        help=(
            "run one optimizer step, write per-rank memory/throughput JSON, "
            "and exit without saving a model checkpoint"
        ),
    )
    accumulation = parser.add_mutually_exclusive_group()
    accumulation.add_argument("--gradient-accumulation-steps", type=int)
    accumulation.add_argument(
        "--global-object-batch",
        type=int,
        help=(
            "require an exact optimizer batch across all ranks; the value must "
            "be divisible by WORLD_SIZE * object batch size"
        ),
    )
    accumulation.add_argument(
        "--minimum-global-object-batch",
        type=int,
        help=(
            "choose accumulation after accounting for WORLD_SIZE and the "
            "physical object batch; the realized batch is recorded by world "
            "size, object batch, and checkpointed accumulation count"
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/graft_gs_a800_native.yaml"))
    args = parser.parse_args()
    local_device = bind_local_cuda_device(require_cuda=True)
    args.vggt_checkpoint = resolve_vggt_checkpoint(args.vggt_checkpoint)
    phase = TrainingPhase(args.phase)
    model_config, training_config, distributed_config, dataset_config = load_server_config(args.config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    rank = int(os.environ.get("RANK", "0"))
    synchronize_object_atlas = args.same_object_view_shards or bool(
        distributed_config.get("synchronize_object_atlas", False)
    )
    object_batch_size = int(
        args.object_batch_size
        if args.object_batch_size is not None
        else training_config.get("object_batch_size", 1)
    )
    if object_batch_size < 1:
        raise ValueError("--object-batch-size must be positive")
    if synchronize_object_atlas and object_batch_size != 1:
        raise ValueError(
            "same-object view-sharded DDP requires --object-batch-size 1"
        )
    if args.global_object_batch is not None:
        if synchronize_object_atlas:
            raise ValueError(
                "--global-object-batch applies only to independent object-level DDP"
            )
        physical_global_batch = world_size * object_batch_size
        if (
            args.global_object_batch < physical_global_batch
            or args.global_object_batch % physical_global_batch
        ):
            raise ValueError(
                "--global-object-batch must be a positive multiple of "
                "WORLD_SIZE * object batch size"
            )
        gradient_accumulation_steps = (
            args.global_object_batch // physical_global_batch
        )
    elif args.minimum_global_object_batch is not None:
        if args.same_object_view_shards:
            raise ValueError(
                "--minimum-global-object-batch applies only to independent object-level DDP"
            )
        if args.minimum_global_object_batch < 1:
            raise ValueError("--minimum-global-object-batch must be positive")
        minimum_objects_per_rank = (
            args.minimum_global_object_batch + world_size - 1
        ) // world_size
        gradient_accumulation_steps = max(
            1,
            (minimum_objects_per_rank + object_batch_size - 1)
            // object_batch_size,
        )
    else:
        gradient_accumulation_steps = int(
            args.gradient_accumulation_steps
            if args.gradient_accumulation_steps is not None
            else training_config.get("gradient_accumulation_steps", 1)
        )
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation steps must be positive")
    maximum_views = int(
        args.maximum_views
        if args.maximum_views is not None
        else dataset_config.get("maximum_views", 24)
    )
    minimum_views = int(dataset_config.get("minimum_views", 2))
    if maximum_views < minimum_views:
        raise ValueError("--maximum-views must be at least dataset.minimum_views")
    precision_policy = load_precision_policy(args.config)
    precision_record = precision_policy.apply()
    loss_weights = load_loss_weights(args.config)
    prior_config = load_trellis_prior_config(args.config)
    configured_id_file = dataset_config.get("object_id_file")
    object_id_file = args.object_id_file or (
        Path(str(configured_id_file)) if configured_id_file is not None else None
    )
    object_ids = (
        load_meshfleet_object_ids(object_id_file)
        if object_id_file is not None
        else None
    )
    object_id_digest = (
        meshfleet_object_id_digest(object_ids) if object_ids is not None else None
    )
    use_prior = bool(prior_config["enabled_after_phase_a"]) and phase is not TrainingPhase.EVIDENCE_CALIBRATION
    if use_prior:
        args.trellis_checkpoint = resolve_trellis_checkpoint(args.trellis_checkpoint)
    prior_kwargs = {
        "samples": int(prior_config["samples"]),
        "sampler_steps": int(prior_config["sampler_steps"]),
        "strength": float(prior_config["strength"]),
        "minimum_probability": float(prior_config["minimum_probability"]),
        "uncertainty_discount": float(prior_config["uncertainty_discount"]),
        "maximum_conditioning_views": prior_config[
            "maximum_conditioning_views"
        ],
        "release_cuda_cache_after_sampling": bool(
            prior_config["release_cuda_cache_after_sampling"]
        ),
        "offload_cuda_pipeline_after_sampling": bool(
            prior_config["offload_cuda_pipeline_after_sampling"]
        ),
    }
    owns_trellis_sampling = (
        not synchronize_object_atlas or world_size == 1 or rank == 0
    )
    if use_prior and owns_trellis_sampling:
        prior = TrellisPriorAdapter.from_pretrained(
            args.trellis_checkpoint,
            **prior_kwargs,
            device=local_device,
        )
    elif use_prior:
        prior = TrellisPriorAdapter(pipeline=None, **prior_kwargs)
    else:
        prior = None
    adapter = VGGTAdapter.from_pretrained(
        args.vggt_checkpoint,
        feature_dim=model_config.feature_dim,
        backbone_dtype=precision_policy.backbone_dtype,
    )
    model = GraftGS(adapter, model_config, prior)
    teacher = None
    if phase is TrainingPhase.QUANTIZATION_DISTILLATION:
        if args.teacher is None:
            raise ValueError("Phase E requires --teacher")
        teacher = GraftGS(
            VGGTAdapter.from_pretrained(
                args.vggt_checkpoint,
                feature_dim=model_config.feature_dim,
                backbone_dtype=precision_policy.backbone_dtype,
            ),
            model_config,
            prior,
        )
        teacher_payload, _ = load_graft_checkpoint(
            teacher, args.teacher, map_location="cpu", strict=True
        )
        validate_precision_policy(teacher_payload, precision_policy)
        teacher_trainer = teacher_payload.get("trainer_config", {})
        if not isinstance(teacher_trainer, dict) or teacher_trainer.get(
            "trellis_prior_checkpoint"
        ) is None:
            raise ValueError("Phase-E teacher checkpoint lacks TRELLIS prior provenance")
        for field_name, expected in (
            ("trellis_prior_samples", int(prior_config["samples"])),
            ("trellis_prior_sampler_steps", int(prior_config["sampler_steps"])),
            ("trellis_prior_strength", float(prior_config["strength"])),
            (
                "trellis_prior_minimum_probability",
                float(prior_config["minimum_probability"]),
            ),
            (
                "trellis_prior_uncertainty_discount",
                float(prior_config["uncertainty_discount"]),
            ),
        ):
            if teacher_trainer.get(field_name) != expected:
                raise ValueError(f"Phase-E teacher prior policy differs at {field_name}")
    manifest_digest = None
    if args.manifest is not None:
        manifest_digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    teacher_bundle_digest = None
    if args.teacher_bundle_root is not None:
        bundle_files = sorted(args.teacher_bundle_root.glob("*.teacher.pt"))
        if not bundle_files:
            raise ValueError("--teacher-bundle-root contains no .teacher.pt files")
        digest = hashlib.sha256()
        for path in bundle_files:
            digest.update(path.name.encode("utf8"))
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        teacher_bundle_digest = digest.hexdigest()
    perceptual_digest = (
        hashlib.sha256(args.perceptual_checkpoint.read_bytes()).hexdigest()
        if args.perceptual_checkpoint is not None
        else None
    )
    trainer = GraftGSTrainer(
        model,
        TrainerConfig(
            phase=phase,
            learning_rate=float(training_config.get("learning_rate", 1.0e-4)),
            object_batch_size=object_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            maximum_gradient_norm=float(training_config.get("maximum_gradient_norm", 1.0)),
            find_unused_parameters=bool(
                distributed_config.get("find_unused_parameters", False)
            ),
            gradient_purification_enabled=bool(
                training_config.get("gradient_purification_enabled", True)
            ),
            gradient_purification_maximum_views=int(
                training_config.get("gradient_purification_maximum_views", 8)
            ),
            gradient_consensus_cosine=float(
                training_config.get("gradient_consensus_cosine", 0.2)
            ),
            gradient_consensus_relative_singular_value=float(
                training_config.get(
                    "gradient_consensus_relative_singular_value", 0.05
                )
            ),
            gradient_artifact_relative_singular_value=float(
                training_config.get(
                    "gradient_artifact_relative_singular_value", 0.1
                )
            ),
            gradient_weiszfeld_iterations=int(
                training_config.get("gradient_weiszfeld_iterations", 12)
            ),
            gradient_fisher_decay=float(
                training_config.get("gradient_fisher_decay", 0.95)
            ),
            gradient_fisher_damping=float(
                training_config.get("gradient_fisher_damping", 1.0e-6)
            ),
            gradient_fisher_radius=float(
                training_config.get("gradient_fisher_radius", 1.0)
            ),
            quantization_adversarial_log_scale_radius=float(
                training_config.get(
                    "quantization_adversarial_log_scale_radius", 0.05
                )
            ),
            topology_hardening_relative_margin=float(
                training_config.get("topology_hardening_relative_margin", 0.1)
            ),
            topology_hardening_temperature=float(
                training_config.get("topology_hardening_temperature", 0.1)
            ),
            output_directory=args.output,
            synchronize_object_atlas=synchronize_object_atlas,
            dataset_manifest=str(args.manifest.resolve()) if args.manifest is not None else None,
            dataset_manifest_sha256=manifest_digest,
            dataset_object_id_catalog=(
                str(object_id_file.resolve()) if object_id_file is not None else None
            ),
            dataset_object_id_catalog_sha256=object_id_digest,
            dataset_object_id_count=len(object_ids) if object_ids is not None else None,
            dataset_split=args.split if args.manifest is not None else None,
            dataset_view_set=str(dataset_config.get("input_view_set", "renders"))
            if args.manifest is not None
            else None,
            dataset_maximum_views=maximum_views,
            dataset_manifest_schema=MANIFEST_SCHEMA if args.manifest is not None else None,
            topology_supervision_mode=str(
                dataset_config.get("topology_supervision_mode", "validated_or_repaired")
            )
            if args.manifest is not None
            else None,
            minimum_topology_confidence=float(
                dataset_config.get("minimum_topology_confidence", 0.95)
            )
            if args.manifest is not None
            else None,
            teacher_checkpoint=str(args.teacher.resolve()) if args.teacher is not None else None,
            teacher_distillation_confidence=float(
                training_config.get("teacher_distillation_confidence", 1.0)
            ),
            teacher_topology_confidence=float(
                training_config.get("teacher_topology_confidence", 0.5)
            ),
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
            derive_mesh_depth_normals=bool(
                training_config.get("derive_mesh_depth_normals", True)
            ),
            require_mesh_depth_normals=bool(
                training_config.get("require_mesh_depth_normals", False)
            ),
            mesh_supervision_view_chunk_size=int(
                training_config.get("mesh_supervision_view_chunk_size", 2)
            ),
            renderer_checkpoint_views=bool(
                training_config.get("renderer_checkpoint_views", True)
            ),
            teacher_bundle_root=(
                str(args.teacher_bundle_root.resolve())
                if args.teacher_bundle_root is not None
                else None
            ),
            teacher_bundle_digest=teacher_bundle_digest,
            teacher_bundle_minimum_confidence=float(
                dataset_config.get("minimum_teacher_bundle_confidence", 0.25)
            ),
            perceptual_checkpoint=(
                str(args.perceptual_checkpoint.resolve())
                if args.perceptual_checkpoint is not None
                else None
            ),
            perceptual_checkpoint_sha256=perceptual_digest,
            precision_backbone=precision_policy.backbone,
            precision_geometric_state=precision_policy.geometric_state,
            precision_analytical_solve=precision_policy.analytical_solve,
            precision_diagnostics=precision_policy.diagnostics,
            precision_float32_matmul=precision_policy.float32_matmul_precision,
            precision_allow_tf32=precision_policy.allow_tf32,
        ),
        loss_weights=loss_weights,
        teacher=teacher,
    )
    if trainer.context.rank == 0:
        precision_path = Path(args.output) / "precision_policy.json"
        precision_path.parent.mkdir(parents=True, exist_ok=True)
        precision_path.write_text(
            json.dumps(
                precision_record,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf8",
        )
    dataset_format = args.dataset_format
    if dataset_format == "auto":
        dataset_format = (
            "meshfleet"
            if args.manifest is not None or (args.dataset / args.split / "renders").is_dir()
            else "folders"
        )
    if dataset_format == "meshfleet":
        if args.manifest is None:
            raise ValueError("MeshFleet training requires --manifest from scripts/build_meshfleet_manifest.py")
        image_size = dataset_config.get("image_size", [518, 518])
        dataset = MeshFleetObjectDataset(
            MeshFleetDatasetConfig(
                root=args.dataset,
                manifest=args.manifest,
                object_id_file=object_id_file,
                split=args.split,
                input_view_set=str(dataset_config.get("input_view_set", "renders")),
                image_size=(int(image_size[0]), int(image_size[1])),
                minimum_views=minimum_views,
                maximum_views=maximum_views,
                view_selection=str(dataset_config.get("view_selection", "random")),
                foreground_alpha_threshold=float(
                    dataset_config.get("foreground_alpha_threshold", 0.5)
                ),
                surface_grid_resolution=int(
                    dataset_config.get("surface_grid_resolution", 64)
                ),
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
                require_surface_voxels=bool(dataset_config.get("require_surface_voxels", True)),
                require_requested_modalities=bool(
                    dataset_config.get("require_requested_modalities", True)
                ),
                require_complete_input_view_set=bool(
                    dataset_config.get("require_complete_input_view_set", True)
                ),
                require_normalization=bool(
                    dataset_config.get("require_normalization", True)
                ),
                require_render_mesh=bool(dataset_config.get("require_render_mesh", False)),
                topology_supervision_mode=str(
                    dataset_config.get("topology_supervision_mode", "validated_or_repaired")
                ),
            minimum_topology_confidence=float(
                dataset_config.get("minimum_topology_confidence", 0.95)
            ),
            teacher_bundle_root=args.teacher_bundle_root,
            minimum_teacher_bundle_confidence=float(
                dataset_config.get("minimum_teacher_bundle_confidence", 0.25)
            ),
            require_teacher_bundle=(
                phase is TrainingPhase.RIEMANNIAN_FLOW
                and args.teacher_bundle_root is not None
            ),
            )
        )
        collate = (
            meshfleet_single_object_collate
            if object_batch_size == 1
            else meshfleet_object_collate
        )
        if trainer.context.rank == 0:
            coverage_path = Path(args.output) / f"dataset_coverage_{args.split}.json"
            coverage_path.parent.mkdir(parents=True, exist_ok=True)
            coverage_path.write_text(
                json.dumps(
                    {
                        "coverage": dataset.coverage,
                        "admitted_object_ids": [
                            record.object_id for record in dataset.records
                        ],
                        "excluded": dataset.excluded,
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf8",
            )
    else:
        dataset = FolderMultiviewDataset(
            args.dataset,
            require_target_state=phase is TrainingPhase.RIEMANNIAN_FLOW,
        )
        collate = (
            single_object_collate
            if object_batch_size == 1
            else folder_object_collate
        )
    sampler = None
    batch_sampler = None
    same_object = trainer.config.synchronize_object_atlas
    if object_batch_size > 1 and trainer.context.distributed:
        batch_sampler = DistributedViewCountBatchSampler(
            dataset,
            object_batch_size,
            num_replicas=trainer.context.world_size,
            rank=trainer.context.rank,
            shuffle=True,
            seed=trainer.config.seed,
        )
    elif object_batch_size > 1:
        batch_sampler = ViewCountBatchSampler(
            dataset,
            object_batch_size,
            shuffle=not same_object,
            seed=trainer.config.seed,
        )
    elif trainer.context.distributed and not same_object:
        sampler = DistributedSampler(
            dataset,
            num_replicas=trainer.context.world_size,
            rank=trainer.context.rank,
            shuffle=True,
        )
    dataloader_workers = int(
        args.dataloader_workers
        if args.dataloader_workers is not None
        else training_config.get("dataloader_workers", 8)
    )
    prefetch_factor = int(
        args.dataloader_prefetch_factor
        if args.dataloader_prefetch_factor is not None
        else training_config.get("dataloader_prefetch_factor", 4)
    )
    if dataloader_workers < 0 or prefetch_factor < 1:
        raise ValueError("dataloader workers must be non-negative and prefetch positive")
    loader_options = dict(
        dataset=dataset,
        num_workers=dataloader_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if dataloader_workers > 0 else None,
        # Workers are recreated after dataset.set_epoch so deterministic random
        # view subsets actually change between epochs.
        persistent_workers=False,
        collate_fn=collate,
    )
    if batch_sampler is not None:
        loader_options["batch_sampler"] = batch_sampler
    else:
        loader_options.update(
            batch_size=1,
            sampler=sampler,
            shuffle=sampler is None and not same_object,
        )
    loader = DataLoader(**loader_options)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif args.initialize_from:
        trainer.load_model_weights(args.initialize_from, strict=False)
    if args.batch_probe is not None:
        trainer.fit(loader, trainer.global_step + 1)
        if trainer.last_train_metrics is None:
            raise RuntimeError("batch probe completed without training metrics")
        local_probe = {
            "rank": trainer.context.rank,
            "logical_device": trainer.context.local_rank,
            "object_batch_size": object_batch_size,
            **trainer.last_train_metrics,
        }
        if trainer.context.distributed:
            probes: list[object] = [
                None for _ in range(trainer.context.world_size)
            ]
            dist.all_gather_object(probes, local_probe)
        else:
            probes = [local_probe]
        if trainer.context.rank == 0:
            typed_probes = [
                value for value in probes if isinstance(value, dict)
            ]
            maximum_seconds = max(float(value["seconds"]) for value in typed_probes)
            processed_objects = sum(
                int(value["local_scenes"]) for value in typed_probes
            )
            payload = {
                "schema": "graft-gs-object-batch-probe-v1",
                "world_size": trainer.context.world_size,
                "object_batch_size": object_batch_size,
                "global_objects_per_optimizer_step": (
                    object_batch_size
                    * trainer.context.world_size
                    * gradient_accumulation_steps
                ),
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "minimum_realized_object_batch_size": min(
                    int(value["local_scenes"]) for value in typed_probes
                ),
                "maximum_realized_object_batch_size": max(
                    int(value["local_scenes"]) for value in typed_probes
                ),
                "aggregate_objects_per_second": (
                    processed_objects / max(maximum_seconds, 1.0e-12)
                ),
                "maximum_peak_allocated_fraction": max(
                    float(value["peak_allocated_fraction"])
                    for value in typed_probes
                ),
                "maximum_peak_reserved_fraction": max(
                    float(value["peak_reserved_fraction"])
                    for value in typed_probes
                ),
                "minimum_ending_driver_free_fraction": min(
                    float(value["ending_driver_free_fraction"])
                    for value in typed_probes
                ),
                "ranks": typed_probes,
            }
            args.batch_probe.parent.mkdir(parents=True, exist_ok=True)
            args.batch_probe.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf8",
            )
        if trainer.context.distributed:
            dist.barrier()
            dist.destroy_process_group()
        return
    trainer.fit(loader, args.steps)
    trainer.save_checkpoint(Path(args.output) / "final.pt")


if __name__ == "__main__":
    main()

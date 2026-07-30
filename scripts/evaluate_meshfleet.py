"""Evaluate every admitted manifest object with deterministic multi-GPU sharding."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from graft_gs.data import (
    cost_balanced_distributed_indices,
    MeshFleetDatasetConfig,
    MeshFleetObjectDataset,
    meshfleet_object_collate,
    ViewCountBatchSampler,
)
from graft_gs.engine import (
    load_graft_checkpoint,
    load_precision_policy,
    load_server_config,
    load_trellis_prior_config,
    validate_precision_policy,
    validate_trellis_prior_policy,
)
from graft_gs.engine.losses import symmetric_surface_chamfer
from graft_gs.integration import (
    GraftGS,
    TrellisPriorAdapter,
    VGGTAdapter,
    resolve_trellis_checkpoint,
    resolve_vggt_checkpoint,
)


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf8", newline="\n") as file:
        file.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("graft_checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--splits", nargs="+", default=("test",))
    parser.add_argument("--object-id-file", type=Path)
    parser.add_argument("--view-set", default="renders")
    parser.add_argument("--maximum-views", type=int, default=12)
    parser.add_argument("--object-batch-size", type=int, default=1)
    parser.add_argument("--dataloader-workers", type=int)
    parser.add_argument("--dataloader-prefetch-factor", type=int)
    parser.add_argument(
        "--maximum-batches-per-rank",
        type=int,
        help="stop after this many local batches (used by isolated batch probes)",
    )
    parser.add_argument("--vggt-checkpoint")
    parser.add_argument("--trellis-checkpoint")
    parser.add_argument("--disable-trellis-prior", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/graft_gs_a800_native.yaml")
    )
    args = parser.parse_args()
    args.vggt_checkpoint = resolve_vggt_checkpoint(args.vggt_checkpoint)

    (
        model_config,
        training_config,
        distributed_config,
        dataset_config,
    ) = load_server_config(args.config)
    precision_policy = load_precision_policy(args.config)
    precision_policy.apply()
    prior_config = load_trellis_prior_config(args.config)
    configured_id_file = dataset_config.get("object_id_file")
    object_id_file = args.object_id_file or (
        Path(str(configured_id_file)) if configured_id_file is not None else None
    )
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("MeshFleet evaluation requires a CUDA environment")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=str(distributed_config.get("backend", "nccl")),
            init_method="env://",
            timeout=timedelta(
                seconds=int(
                    distributed_config.get("collective_timeout_seconds", 1800)
                )
            ),
        )
    if args.object_batch_size < 1:
        raise ValueError("--object-batch-size must be positive")
    if (
        args.maximum_batches_per_rank is not None
        and args.maximum_batches_per_rank < 1
    ):
        raise ValueError("--maximum-batches-per-rank must be positive")
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
        raise ValueError(
            "dataloader workers must be non-negative and prefetch positive"
        )

    use_prior = bool(prior_config["enabled_after_phase_a"]) and not args.disable_trellis_prior
    if use_prior:
        args.trellis_checkpoint = resolve_trellis_checkpoint(args.trellis_checkpoint)
    prior = (
        TrellisPriorAdapter.from_pretrained(
            args.trellis_checkpoint,
            samples=int(prior_config["samples"]),
            sampler_steps=int(prior_config["sampler_steps"]),
            strength=float(prior_config["strength"]),
            minimum_probability=float(prior_config["minimum_probability"]),
            uncertainty_discount=float(prior_config["uncertainty_discount"]),
            maximum_conditioning_views=prior_config[
                "maximum_conditioning_views"
            ],
            release_cuda_cache_after_sampling=bool(
                prior_config["release_cuda_cache_after_sampling"]
            ),
            offload_cuda_pipeline_after_sampling=bool(
                prior_config["offload_cuda_pipeline_after_sampling"]
            ),
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
    checkpoint, checkpoint_report = load_graft_checkpoint(
        model, args.graft_checkpoint, map_location="cpu", strict=True
    )
    validate_precision_policy(checkpoint, precision_policy)
    validate_trellis_prior_policy(
        checkpoint,
        enabled=use_prior,
        samples=int(prior_config["samples"]),
        sampler_steps=int(prior_config["sampler_steps"]),
        strength=float(prior_config["strength"]),
        minimum_probability=float(prior_config["minimum_probability"]),
        uncertainty_discount=float(prior_config["uncertainty_discount"]),
    )
    model = model.to(device).eval()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    rank_metrics = args.output_directory / f"metrics.rank-{rank:05d}.jsonl"
    rank_metrics.unlink(missing_ok=True)
    image_size = dataset_config.get("image_size", [518, 518])
    manifest_digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    failed = False
    total_admitted = 0
    local_batches = 0
    stop_after_probe = False

    for split in args.splits:
        dataset = MeshFleetObjectDataset(
            MeshFleetDatasetConfig(
                root=args.dataset_root,
                manifest=args.manifest,
                object_id_file=object_id_file,
                split=split,
                input_view_set=args.view_set,
                image_size=(int(image_size[0]), int(image_size[1])),
                minimum_views=int(dataset_config.get("minimum_views", 2)),
                maximum_views=args.maximum_views,
                view_selection="uniform",
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
                require_surface_voxels=True,
                require_requested_modalities=bool(
                    dataset_config.get("require_requested_modalities", True)
                ),
                require_complete_input_view_set=bool(
                    dataset_config.get("require_complete_input_view_set", True)
                ),
                require_normalization=bool(
                    dataset_config.get("require_normalization", True)
                ),
                require_render_mesh=bool(
                    dataset_config.get("require_render_mesh", False)
                ),
                topology_supervision_mode=str(
                    dataset_config.get(
                        "topology_supervision_mode", "validated_or_repaired"
                    )
                ),
                minimum_topology_confidence=float(
                    dataset_config.get("minimum_topology_confidence", 0.95)
                ),
            )
        )
        total_admitted += len(dataset)
        metadata = checkpoint.get("trainer_config", {})
        if isinstance(metadata, dict):
            expected_manifest = metadata.get("dataset_manifest_sha256")
            expected_catalog = metadata.get("dataset_object_id_catalog_sha256")
            if expected_manifest is not None and expected_manifest != manifest_digest:
                raise ValueError("evaluation manifest differs from checkpoint provenance")
            if (
                expected_catalog is not None
                and expected_catalog != dataset.object_id_catalog_sha256
            ):
                raise ValueError("evaluation object catalog differs from checkpoint provenance")

        local_indices = cost_balanced_distributed_indices(
            dataset,
            num_replicas=world_size,
            rank=rank,
        )
        batch_sampler = ViewCountBatchSampler(
            dataset,
            args.object_batch_size,
            indices=local_indices,
            shuffle=False,
            largest_first=args.maximum_batches_per_rank is not None,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=dataloader_workers,
            pin_memory=True,
            prefetch_factor=(
                prefetch_factor if dataloader_workers > 0 else None
            ),
            persistent_workers=dataloader_workers > 0,
            collate_fn=meshfleet_object_collate,
        )
        for batch in loader:
            object_value = batch["object_id"]
            object_ids = (
                [str(value) for value in object_value]
                if isinstance(object_value, (list, tuple))
                else [str(object_value)]
            )
            try:
                images = batch["images"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                evidence_mask = batch["evidence_mask"].to(
                    device=device,
                    non_blocking=True,
                )
                extrinsics = batch["extrinsics_world_to_camera"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                intrinsics = batch["intrinsics"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                prior_seed = tuple(
                    int.from_bytes(
                        hashlib.sha256(
                            object_id.encode("utf8")
                        ).digest()[:8],
                        "little",
                    )
                    % (2**31 - 1)
                    for object_id in object_ids
                )
                torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                with torch.no_grad():
                    output = model(
                        images,
                        valid_mask=evidence_mask,
                        render_input_views=False,
                        ground_truth_extrinsics=extrinsics,
                        ground_truth_intrinsics=intrinsics,
                        atlas_root_bounds=batch["atlas_root_bounds"].to(
                            device=device,
                            non_blocking=True,
                        ),
                        trellis_prior_seed=prior_seed,
                    )
                torch.cuda.synchronize(device)
                batch_seconds = time.perf_counter() - start
                peak_allocated = torch.cuda.max_memory_allocated(device)
                peak_reserved = torch.cuda.max_memory_reserved(device)
                driver_free, device_memory = torch.cuda.mem_get_info(device)
                surface_value = batch["surface_voxel_centers"]
                if isinstance(surface_value, (list, tuple)):
                    surfaces = list(surface_value)
                else:
                    dense_surface = torch.as_tensor(surface_value)
                    surfaces = (
                        [dense_surface]
                        if dense_surface.ndim == 2 and len(object_ids) == 1
                        else [
                            dense_surface[index]
                            for index in range(len(object_ids))
                        ]
                    )
                batch_metrics = []
                for index, (object_id, scene) in enumerate(
                    zip(object_ids, output.scenes)
                ):
                    object_output = args.output_directory / split / object_id
                    ply, glb = scene.export(object_output, object_id)
                    target = torch.as_tensor(surfaces[index]).to(
                        device=device,
                        dtype=scene.gaussians.means.dtype,
                        non_blocking=True,
                    )
                    batch_metrics.append(
                        {
                            "status": "ok",
                            "object_id": object_id,
                            "split": split,
                            "rank": rank,
                            "seconds": batch_seconds / len(object_ids),
                            "batch_seconds": batch_seconds,
                            "object_batch_size": len(object_ids),
                            "peak_memory_bytes": peak_allocated,
                            "peak_reserved_memory_bytes": peak_reserved,
                            "device_memory_bytes": device_memory,
                            "peak_allocated_fraction": (
                                peak_allocated / device_memory
                            ),
                            "peak_reserved_fraction": (
                                peak_reserved / device_memory
                            ),
                            "ending_driver_free_fraction": (
                                driver_free / device_memory
                            ),
                            "surface_chamfer_squared": float(
                                symmetric_surface_chamfer(
                                    scene.gaussians.means, target
                                )
                                .detach()
                                .cpu()
                            ),
                            "active_charts": scene.atlas.num_active,
                            "transport_edges": scene.mapping.graph.num_edges,
                            "selected_topology": scene.topology.selected.identifier,
                            "betti": scene.topology.selected.betti,
                            "gaussians": scene.gaussians.means.shape[0],
                            "mesh_faces": scene.mesh.faces.shape[0],
                            "ply": str(ply),
                            "glb": str(glb),
                            "checkpoint": checkpoint_report.__dict__,
                        }
                    )
            except Exception as error:
                failed = True
                for object_id in object_ids:
                    _append_jsonl(
                        rank_metrics,
                        {
                            "status": "error",
                            "object_id": object_id,
                            "split": split,
                            "rank": rank,
                            "object_batch_size": len(object_ids),
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                    )
                if not args.continue_on_error:
                    raise
            else:
                for metric in batch_metrics:
                    _append_jsonl(rank_metrics, metric)
            local_batches += 1
            if (
                args.maximum_batches_per_rank is not None
                and local_batches >= args.maximum_batches_per_rank
            ):
                stop_after_probe = True
                break
        if stop_after_probe:
            break

    if world_size > 1:
        # Do not let the final scalar status collective inherit outstanding
        # rank-local evaluation/export CUDA work.
        torch.cuda.synchronize(device)
        failure = torch.tensor([int(failed)], dtype=torch.int64, device=device)
        dist.all_reduce(failure, op=dist.ReduceOp.MAX)
        failed = bool(failure.item())
        dist.barrier()
    if rank == 0:
        all_metrics: list[dict[str, object]] = []
        for source_rank in range(world_size):
            all_metrics.extend(
                _read_jsonl(
                    args.output_directory / f"metrics.rank-{source_rank:05d}.jsonl"
                )
            )
        all_metrics.sort(key=lambda value: (str(value["split"]), str(value["object_id"])))
        merged = args.output_directory / "metrics.jsonl"
        merged.write_text(
            "".join(
                json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
                for value in all_metrics
            ),
            encoding="utf8",
        )
        ok = sum(value["status"] == "ok" for value in all_metrics)
        summary = {
            "catalog": (
                str(object_id_file.resolve()) if object_id_file is not None else None
            ),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest_digest,
            "splits": list(args.splits),
            "world_size": world_size,
            "maximum_object_batch_size_per_rank": args.object_batch_size,
            "maximum_batches_per_rank": args.maximum_batches_per_rank,
            "admitted_count": total_admitted,
            "evaluated_count": len(all_metrics),
            "success_count": ok,
            "failure_count": len(all_metrics) - ok,
            "complete": len(all_metrics) == total_admitted and ok == total_admitted,
        }
        (args.output_directory / "summary.json").write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

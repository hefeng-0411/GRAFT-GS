"""Checkpoint-backed VGGT/TRELLIS adapter validation on real MeshFleet views.

Each invocation validates one upstream model so its process releases all model
memory before the other checkpoint is loaded.  The object is selected from the
admitted manifest records by a deterministic policy, never by a hardcoded ID.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from graft_gs.data import (
    MeshFleetDatasetConfig,
    MeshFleetObjectDataset,
    ObjectManifestRecord,
    load_meshfleet_manifest,
    meshfleet_record_admission_reasons,
)
from graft_gs.engine import NativePrecisionPolicy
from graft_gs.integration import (
    TrellisPriorAdapter,
    VGGTAdapter,
    resolve_trellis_checkpoint,
    resolve_vggt_checkpoint,
)


def _real_multiview_sample(
    root: Path,
    manifest: Path,
    requested_object_id: str | None,
) -> tuple[dict[str, object], str]:
    candidates: list[tuple[ObjectManifestRecord, MeshFleetDatasetConfig]] = []
    for record in load_meshfleet_manifest(manifest):
        if requested_object_id is None or record.object_id == requested_object_id:
            candidate_config = MeshFleetDatasetConfig(
                root=root,
                manifest=manifest,
                split=record.split,
                include_object_ids=(record.object_id,),
                input_view_set="renders",
                image_size=(518, 518),
                minimum_views=2,
                maximum_views=2,
                view_selection="uniform",
                load_surface_voxels=False,
                require_surface_voxels=False,
                require_requested_modalities=False,
                require_complete_input_view_set=False,
                require_normalization=True,
            )
            if not meshfleet_record_admission_reasons(record, candidate_config):
                candidates.append((record, candidate_config))
    if not candidates:
        qualifier = (
            f" for object {requested_object_id}" if requested_object_id is not None else ""
        )
        raise RuntimeError(f"manifest has no admitted two-view smoke record{qualifier}")
    selected, selected_config = min(candidates, key=lambda value: value[0].object_id)
    dataset = MeshFleetObjectDataset(selected_config)
    policy = (
        "explicit_object_id" if requested_object_id is not None
        else "lexicographically_first_admitted_smoke_record"
    )
    return dataset[0], policy


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.all(torch.isfinite(value))):
        raise RuntimeError(f"{name} contains non-finite values")


def _validate_vggt(
    images: torch.Tensor,
    checkpoint: str,
) -> dict[str, object]:
    device = torch.device("cuda")
    adapter = VGGTAdapter.from_pretrained(
        checkpoint,
        feature_dim=64,
        freeze_backbone=True,
    ).to(device).eval()
    images = images[None].to(device)
    with torch.no_grad():
        output = adapter(images)
    expected_prefix = images.shape[:2]
    if output.extrinsics_world_to_camera.shape != (*expected_prefix, 3, 4):
        raise RuntimeError("VGGT adapter emitted an invalid extrinsic tensor contract")
    if output.intrinsics.shape != (*expected_prefix, 3, 3):
        raise RuntimeError("VGGT adapter emitted an invalid intrinsic tensor contract")
    if output.depth.shape[:2] != expected_prefix or output.depth.shape[-1] != 1:
        raise RuntimeError("VGGT adapter emitted an invalid depth tensor contract")
    if output.world_points.shape[:2] != expected_prefix or output.world_points.shape[-1] != 3:
        raise RuntimeError("VGGT adapter emitted an invalid point-map tensor contract")
    for name, value in (
        ("patch_features", output.patch_features),
        ("extrinsics", output.extrinsics_world_to_camera),
        ("intrinsics", output.intrinsics),
        ("depth", output.depth),
        ("depth_confidence", output.depth_confidence),
        ("world_points", output.world_points),
        ("world_points_confidence", output.world_points_confidence),
    ):
        _finite(name, value)
    rotation = output.extrinsics_world_to_camera[..., :3, :3].float()
    identity = torch.eye(3, device=device).expand_as(rotation)
    orthogonality_error = torch.linalg.matrix_norm(
        rotation @ rotation.transpose(-1, -2) - identity,
        ord="fro",
        dim=(-2, -1),
    ).max()
    determinant_error = (torch.linalg.det(rotation) - 1.0).abs().max()
    if float(orthogonality_error) > 5.0e-3 or float(determinant_error) > 5.0e-3:
        raise RuntimeError("VGGT camera head produced an invalid SO(3) rotation")
    return {
        "checkpoint": checkpoint,
        "upstream_provenance": adapter.upstream_provenance,
        "patch_features_shape": list(output.patch_features.shape),
        "depth_shape": list(output.depth.shape),
        "world_points_shape": list(output.world_points.shape),
        "maximum_rotation_orthogonality_error": float(orthogonality_error),
        "maximum_rotation_determinant_error": float(determinant_error),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _validate_trellis(
    images: torch.Tensor,
    checkpoint: str,
    samples: int,
    sampler_steps: int,
) -> dict[str, object]:
    device = torch.device("cuda")
    adapter = TrellisPriorAdapter.from_pretrained(
        checkpoint,
        samples=samples,
        sampler_steps=sampler_steps,
        device=device,
    )
    images = images.to(device)
    prior = adapter.sample(images, seed=1729)
    root_min = torch.full((3,), -0.5, device=device, dtype=torch.float32)
    root_max = torch.full((3,), 0.5, device=device, dtype=torch.float32)
    measure = adapter.support_measure(prior, root_min, root_max)
    if len(prior.coordinates) != samples:
        raise RuntimeError("TRELLIS posterior sample count differs from the request")
    if measure.sample_count != samples or measure.coordinates.shape[0] < 1:
        raise RuntimeError("TRELLIS support measure is empty or has invalid provenance")
    _finite("TRELLIS prior position", measure.positions)
    _finite("TRELLIS prior probability", measure.probability)
    return {
        "checkpoint": checkpoint,
        "upstream_provenance": adapter.upstream_provenance,
        "resolution": prior.resolution,
        "posterior_samples": len(prior.coordinates),
        "sample_support_counts": [int(value.shape[0]) for value in prior.coordinates],
        "union_support_count": int(measure.coordinates.shape[0]),
        "probability_minimum": float(measure.probability.min()),
        "probability_maximum": float(measure.probability.max()),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=("vggt", "trellis"))
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--object-id")
    parser.add_argument("--vggt-checkpoint")
    parser.add_argument("--trellis-checkpoint")
    parser.add_argument("--trellis-samples", type=int, default=2)
    parser.add_argument("--trellis-sampler-steps", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("external-model validation requires an A800 CUDA device")
    precision = NativePrecisionPolicy()
    precision_record = precision.apply()
    torch.cuda.reset_peak_memory_stats()
    sample, selection_policy = _real_multiview_sample(
        args.dataset_root.resolve(),
        args.manifest.resolve(),
        args.object_id,
    )
    images = sample["images"]
    if not isinstance(images, torch.Tensor):
        raise TypeError("MeshFleet smoke sample did not provide an image tensor")
    start = time.perf_counter()
    if args.component == "vggt":
        details = _validate_vggt(images, resolve_vggt_checkpoint(args.vggt_checkpoint))
    else:
        details = _validate_trellis(
            images,
            resolve_trellis_checkpoint(args.trellis_checkpoint),
            args.trellis_samples,
            args.trellis_sampler_steps,
        )
    record = {
        "valid": True,
        "component": args.component,
        "dataset_root": str(args.dataset_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "object_id": sample["object_id"],
        "selection_policy": selection_policy,
        "input_shape": list(images.shape),
        "seconds": time.perf_counter() - start,
        "precision": precision_record,
        "details": details,
    }
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf8")
    print(payload, end="")


if __name__ == "__main__":
    main()

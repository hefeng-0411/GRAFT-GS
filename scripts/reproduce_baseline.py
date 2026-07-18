"""Reproduce the untouched VGGT or TRELLIS inference path on the A800 server.

This script intentionally calls each upstream repository's released public API.
It does not route through GRAFT-GS, so its artifacts are valid baseline controls.
Run the two subcommands in their corresponding upstream environments when the
repositories use incompatible compiled dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from graft_gs.integration.external import (
    import_external_module,
    resolve_trellis_checkpoint,
    resolve_vggt_checkpoint,
)


def reproduce_vggt(images_directory: Path, output: Path, checkpoint: str) -> None:
    VGGT = getattr(import_external_module("vggt.models.vggt"), "VGGT")
    load_and_preprocess_images = getattr(
        import_external_module("vggt.utils.load_fn"), "load_and_preprocess_images"
    )
    pose_encoding_to_extri_intri = getattr(
        import_external_module("vggt.utils.pose_enc"),
        "pose_encoding_to_extri_intri",
    )

    paths = sorted(
        path for path in images_directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if len(paths) < 2:
        raise ValueError("VGGT baseline requires at least two ordered images")
    device = torch.device("cuda")
    images = load_and_preprocess_images([str(path) for path in paths]).to(device)
    model = VGGT.from_pretrained(checkpoint).eval().to(device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        prediction = model(images)
    extrinsics, intrinsics = pose_encoding_to_extri_intri(prediction["pose_enc"], images.shape[-2:])
    prediction["extrinsic"] = extrinsics
    prediction["intrinsic"] = intrinsics
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint": checkpoint,
            "image_paths": [str(path) for path in paths],
            "images": images.detach().cpu(),
            "prediction": {
                key: value.detach().float().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in prediction.items()
            },
        },
        output,
    )


def reproduce_trellis(image_path: Path, output_directory: Path, checkpoint: str, seed: int) -> None:
    from PIL import Image
    TrellisImageTo3DPipeline = getattr(
        import_external_module("trellis.pipelines"), "TrellisImageTo3DPipeline"
    )
    postprocessing_utils = import_external_module("trellis.utils.postprocessing_utils")

    pipeline = TrellisImageTo3DPipeline.from_pretrained(checkpoint)
    pipeline.cuda()
    outputs = pipeline.run(Image.open(image_path).convert("RGB"), seed=seed)
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs["gaussian"][0].save_ply(str(output_directory / "trellis_baseline.ply"))
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=0.95,
        texture_size=1024,
    )
    glb.export(str(output_directory / "trellis_baseline.glb"))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="baseline", required=True)
    vggt = subparsers.add_parser("vggt")
    vggt.add_argument("images", type=Path)
    vggt.add_argument("--output", type=Path, default=Path("outputs/baselines/vggt.pt"))
    vggt.add_argument("--checkpoint")
    trellis = subparsers.add_parser("trellis")
    trellis.add_argument("image", type=Path)
    trellis.add_argument("--output", type=Path, default=Path("outputs/baselines/trellis"))
    trellis.add_argument("--checkpoint")
    trellis.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.baseline == "vggt":
        reproduce_vggt(
            arguments.images,
            arguments.output,
            resolve_vggt_checkpoint(arguments.checkpoint),
        )
    else:
        reproduce_trellis(
            arguments.image,
            arguments.output,
            resolve_trellis_checkpoint(arguments.checkpoint),
            arguments.seed,
        )


if __name__ == "__main__":
    main()

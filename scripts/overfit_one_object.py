"""Executable one-object overfit verification using the staged trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from graft_gs.engine import GraftGSTrainer, TrainerConfig, TrainingPhase
from graft_gs.integration import GraftGS, GraftGSConfig, VGGTAdapter


class RepeatedObject:
    def __init__(self, images: torch.Tensor) -> None:
        self.images = images

    def __iter__(self):
        while True:
            yield {"images": self.images}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--checkpoint", default="facebook/VGGT-1B")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", default="outputs/one_object")
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.01)
    args = parser.parse_args()
    from vggt.utils.load_fn import load_and_preprocess_images

    paths = sorted(path for path in args.image_directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(paths) < 2:
        raise ValueError("one-object overfit requires at least two views")
    images = load_and_preprocess_images([str(path) for path in paths[:8]])[None]
    adapter = VGGTAdapter.from_pretrained(args.checkpoint)
    model = GraftGS(adapter, GraftGSConfig(run_flow=False))
    trainer = GraftGSTrainer(
        model,
        TrainerConfig(
            phase=TrainingPhase.ATLAS_AUTOENCODING,
            output_directory=args.output,
            checkpoint_every=max(1, args.steps // 5),
        ),
    )
    output_directory = Path(args.output)
    losses = []
    iterator = iter(RepeatedObject(images))
    for step in range(args.steps):
        metrics = trainer.train_step(next(iterator), step)
        trainer.microstep = step + 1
        trainer.batches_consumed_in_epoch = step + 1
        losses.append(metrics["total"])
        if trainer.global_step and trainer.global_step % trainer.config.checkpoint_every == 0:
            trainer.save_checkpoint(output_directory / f"step-{trainer.global_step:08d}.pt")
    window = min(10, len(losses))
    initial = sum(losses[:window]) / window
    final = sum(losses[-window:]) / window
    relative_improvement = (initial - final) / max(abs(initial), 1.0e-12)
    summary = {
        "steps": trainer.global_step,
        "initial_window_loss": initial,
        "final_window_loss": final,
        "relative_improvement": relative_improvement,
        "losses": losses,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "overfit_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    trainer.save_checkpoint(output_directory / "final.pt")
    trainer.module.eval()
    with torch.no_grad():
        result = trainer.module(images.to(trainer.context.device), render_input_views=True)
    ply, glb = result.scenes[0].export(output_directory, "overfit")
    summary.update(ply=str(ply), glb=str(glb))
    (output_directory / "overfit_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    if relative_improvement < args.minimum_relative_improvement:
        raise RuntimeError(
            f"one-object overfit failed: relative loss improvement {relative_improvement:.6f} "
            f"is below {args.minimum_relative_improvement:.6f}"
        )


if __name__ == "__main__":
    main()

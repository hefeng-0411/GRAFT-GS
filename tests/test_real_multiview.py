"""Checkpoint-backed real multiview integration test for the A800 server."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import torch

from graft_gs.engine import load_graft_checkpoint, validate_trellis_prior_policy
from graft_gs.integration import (
    GraftGS,
    GraftGSConfig,
    TrellisPriorAdapter,
    VGGTAdapter,
    import_external_module,
    resolve_trellis_checkpoint,
    resolve_vggt_checkpoint,
)


@unittest.skipUnless(os.environ.get("GRAFT_GS_REAL_IMAGE_DIR"), "set GRAFT_GS_REAL_IMAGE_DIR on the server")
class RealMultiviewTest(unittest.TestCase):
    def test_checkpoint_inference_export_and_reload(self) -> None:
        load_and_preprocess_images = getattr(
            import_external_module("vggt.utils.load_fn"),
            "load_and_preprocess_images",
        )

        image_directory = Path(os.environ["GRAFT_GS_REAL_IMAGE_DIR"])
        paths = sorted(
            path for path in image_directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        self.assertGreaterEqual(len(paths), 2)
        paths = paths[: int(os.environ.get("GRAFT_GS_REAL_VIEW_COUNT", "8"))]
        device = torch.device("cuda")
        adapter = VGGTAdapter.from_pretrained(resolve_vggt_checkpoint())
        prior = None
        if os.environ.get("GRAFT_GS_USE_TRELLIS_PRIOR", "0") == "1":
            prior = TrellisPriorAdapter.from_pretrained(
                resolve_trellis_checkpoint()
            )
        model = GraftGS(adapter, GraftGSConfig(), prior)
        checkpoint = os.environ.get("GRAFT_GS_CHECKPOINT")
        self.assertIsNotNone(checkpoint, "set GRAFT_GS_CHECKPOINT for checkpoint-backed inference")
        checkpoint_payload, checkpoint_report = load_graft_checkpoint(
            model,
            checkpoint,
            map_location="cpu",
            strict=True,
        )
        if prior is not None:
            validate_trellis_prior_policy(
                checkpoint_payload,
                enabled=True,
                samples=prior.samples,
                sampler_steps=prior.sampler_steps,
                strength=prior.strength,
                minimum_probability=prior.minimum_probability,
                uncertainty_discount=prior.uncertainty_discount,
            )
        self.assertIsNotNone(checkpoint_report.global_step)
        model = model.to(device).eval()
        images = load_and_preprocess_images([str(path) for path in paths]).to(device)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            output = model(images[None], render_input_views=True)
        scene = output.scenes[0]
        self.assertTrue(scene.atlas.validate().valid)
        scene.gaussians.validate()
        self.assertGreater(scene.mesh.faces.shape[0], 0)
        self.assertTrue(torch.all(torch.isfinite(scene.render.color)))
        with tempfile.TemporaryDirectory() as directory:
            ply, glb = scene.export(directory, "real_multiview")
            from plyfile import PlyData
            from pygltflib import GLTF2

            self.assertEqual(PlyData.read(ply)["vertex"].count, scene.gaussians.means.shape[0])
            self.assertEqual(len(GLTF2().load(str(glb)).meshes), 1)
        peak = torch.cuda.max_memory_allocated()
        self.assertGreater(peak, 0)

    @unittest.skipUnless(os.environ.get("GRAFT_GS_RUN_TRAINING_TESTS") == "1", "enable server training tests explicitly")
    def test_trainer_checkpoint_round_trip(self) -> None:
        from graft_gs.engine import GraftGSTrainer, TrainerConfig, TrainingPhase

        adapter = VGGTAdapter.from_pretrained(resolve_vggt_checkpoint())
        model = GraftGS(adapter, GraftGSConfig(run_flow=False))
        with tempfile.TemporaryDirectory() as directory:
            trainer = GraftGSTrainer(
                model,
                TrainerConfig(
                    phase=TrainingPhase.EVIDENCE_CALIBRATION,
                    output_directory=directory,
                ),
            )
            checkpoint = Path(directory) / "resume.pt"
            trainer.global_step = 37
            trainer.epoch = 5
            trainer.microstep = 81
            trainer.batches_consumed_in_epoch = 13
            trainer.save_checkpoint(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["format_version"], 5)
            self.assertEqual(payload["checkpoint_world_size"], trainer.context.world_size)
            self.assertEqual(len(payload["rank_rng_states"]), trainer.context.world_size)
            expected_random = torch.rand(8)
            parameter = next(parameter for parameter in trainer.module.parameters() if parameter.requires_grad)
            reference = parameter.detach().clone()
            with torch.no_grad():
                parameter.add_(1.0)
            trainer.load_checkpoint(checkpoint)
            self.assertEqual(trainer.global_step, 37)
            self.assertEqual(trainer.epoch, 5)
            self.assertEqual(trainer.microstep, 81)
            self.assertEqual(trainer.batches_consumed_in_epoch, 13)
            torch.testing.assert_close(parameter, reference)
            torch.testing.assert_close(torch.rand(8), expected_random)


if __name__ == "__main__":
    unittest.main()

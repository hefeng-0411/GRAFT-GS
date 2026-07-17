"""Dataset, camera-gauge, and derived-supervision contracts."""

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from graft_gs.data.meshfleet import (
    MeshFleetDatasetConfig,
    MeshFleetObjectDataset,
    intrinsics_from_fov,
    meshfleet_single_object_collate,
    opengl_c2w_to_opencv_c2w,
)
from graft_gs.engine.supervision import SurfaceTargetConfig, screened_surface_projection
from graft_gs.engine.losses import (
    GraftGSLoss,
    evidence_surface_calibration,
    provenance_weighted_topology_supervision,
)
from graft_gs.integration.vggt_adapter import VGGTGeometryOutput, align_vggt_to_supervised_cameras
from graft_gs.integration import GraftGS, GraftGSConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_MANIFEST = Path(
    os.environ.get(
        "GRAFT_GS_MESHFLEET_MANIFEST",
        str(PROJECT_ROOT / "data_manifests" / "meshfleet_local_audit.jsonl"),
    )
)
LOCAL_DATASET = Path(
    os.environ.get("GRAFT_GS_MESHFLEET_ROOT", r"D:\VsCode\MVG\Base\MeshFleet_TRELLIS")
)


def _audited_nonmanifold_record():
    records = [
        json.loads(line)
        for line in AUDIT_MANIFEST.read_text(encoding="utf8").splitlines()
        if line
    ]
    matches = [
        record for record in records
        if record.get("checks", {}).get("render_mesh_topology", {}).get(
            "connected_components"
        ) == 8
        and record.get("checks", {}).get("render_mesh_topology", {}).get(
            "nonmanifold_edge_count"
        ) == 313
    ]
    if len(matches) != 1:
        raise AssertionError(
            "audited non-manifold topology fixture must occur exactly once, "
            f"found {len(matches)}"
        )
    return matches[0]


def _audited_dataset_index(dataset: MeshFleetObjectDataset) -> int:
    audited_id = _audited_nonmanifold_record()["object_id"]
    matches = [
        index for index, record in enumerate(dataset.records)
        if record.object_id == audited_id
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"audited dataset object must occur exactly once, found {len(matches)}"
        )
    return matches[0]


def _w2c(rotation_c2w: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    rotation = rotation_c2w.transpose(-1, -2)
    return torch.cat((rotation, -(rotation @ center[..., None])), dim=-1)


class CameraContractTest(unittest.TestCase):
    def test_blender_to_opencv_axis_flip(self) -> None:
        c2w = opengl_c2w_to_opencv_c2w(torch.eye(4, dtype=torch.float64))
        expected = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64))
        torch.testing.assert_close(c2w, expected, atol=0.0, rtol=0.0)
        self.assertGreater(float(torch.linalg.det(c2w[:3, :3])), 0.0)

    def test_intrinsics_are_rescaled_in_pixels(self) -> None:
        intrinsic = intrinsics_from_fov(0.8, 0.6, 512, 512, 256, 384, dtype=torch.float64)
        self.assertAlmostEqual(float(intrinsic[0, 2]), 192.0)
        self.assertAlmostEqual(float(intrinsic[1, 2]), 128.0)
        self.assertAlmostEqual(float(intrinsic[2, 2]), 1.0)

    def test_scene_similarity_gauge_alignment(self) -> None:
        dtype = torch.float64
        center = torch.tensor(
            [[[-1.0, 0.0, 0.2], [0.5, 1.0, -0.1], [0.8, -0.7, 0.4]]], dtype=dtype
        )
        source_rotation = torch.eye(3, dtype=dtype).expand(1, 3, 3, 3).clone()
        source_extrinsic = _w2c(source_rotation, center)
        angle = torch.tensor(0.4, dtype=dtype)
        target_rotation = torch.tensor(
            [
                [torch.cos(angle), -torch.sin(angle), 0.0],
                [torch.sin(angle), torch.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=dtype,
        )
        scale = 1.7
        translation = torch.tensor([0.2, -0.3, 0.4], dtype=dtype)
        target_center = scale * torch.einsum("ij,bkj->bki", target_rotation, center) + translation
        target_c2w_rotation = target_rotation[None, None] @ source_rotation
        target_extrinsic = _w2c(target_c2w_rotation, target_center)
        intrinsic = torch.eye(3, dtype=dtype).expand(1, 3, 3, 3).clone()
        intrinsic[..., 0, 0] = 300.0
        intrinsic[..., 1, 1] = 300.0
        output = VGGTGeometryOutput(
            images=torch.zeros(1, 3, 3, 2, 2, dtype=dtype),
            patch_features=torch.zeros(1, 3, 1, 4, dtype=dtype),
            extrinsics_world_to_camera=source_extrinsic,
            intrinsics=intrinsic,
            depth=torch.ones(1, 3, 2, 2, 1, dtype=dtype),
            depth_confidence=torch.ones(1, 3, 2, 2, dtype=dtype),
            world_points=torch.zeros(1, 3, 2, 2, 3, dtype=dtype),
            world_points_confidence=torch.ones(1, 3, 2, 2, dtype=dtype),
        )
        aligned, diagnostics = align_vggt_to_supervised_cameras(
            output, target_extrinsic, intrinsic
        )
        torch.testing.assert_close(
            aligned.extrinsics_world_to_camera,
            target_extrinsic,
            atol=2.0e-10,
            rtol=2.0e-10,
        )
        torch.testing.assert_close(diagnostics.scale, torch.tensor([scale], dtype=dtype), atol=2.0e-10, rtol=2.0e-10)
        self.assertLess(float(diagnostics.center_rmse.max()), 2.0e-10)


class MeshFleetAuditTest(unittest.TestCase):
    def test_checked_manifest_records_physical_schema(self) -> None:
        record = _audited_nonmanifold_record()
        self.assertTrue(record["checks"]["feature_indices_equal_latent_coords"])
        self.assertTrue(record["checks"]["surface_voxel_indices_equal_feature_indices"])
        self.assertEqual(record["checks"]["surface_voxel_grid"]["maximum_center_residual"], 0.0)
        self.assertEqual(record["views"]["renders_cond"]["declared_frame_count"], 24)
        self.assertEqual(record["views"]["renders_cond"]["available_frame_count"], 1)
        topology = record["checks"]["render_mesh_topology"]
        self.assertEqual(topology["connected_components"], 8)
        self.assertEqual(topology["nonmanifold_edge_count"], 313)
        self.assertFalse(topology["hard_topology_supervision_admissible"])

    @unittest.skipUnless(LOCAL_DATASET.is_dir(), "audited MeshFleet_TRELLIS dataset is not mounted")
    def test_object_loader_uses_only_available_frames(self) -> None:
        dataset = MeshFleetObjectDataset(
            MeshFleetDatasetConfig(
                root=LOCAL_DATASET,
                manifest=AUDIT_MANIFEST,
                split="test",
                input_view_set="renders",
                image_size=(64, 64),
                maximum_views=3,
                load_surface_voxels=True,
                load_trellis_features=True,
                load_trellis_latents=True,
            )
        )
        sample = dataset[_audited_dataset_index(dataset)]
        self.assertEqual(tuple(sample["images"].shape), (3, 3, 64, 64))
        self.assertEqual(tuple(sample["valid_mask"].shape), (3, 1, 64, 64))
        self.assertEqual(sample["valid_mask"].dtype, torch.bool)
        self.assertEqual(tuple(sample["alpha"].shape), (3, 1, 64, 64))
        self.assertTrue(torch.equal(sample["valid_mask"], sample["evidence_mask"]))
        self.assertEqual(tuple(sample["surface_voxel_centers"].shape), (7996, 3))
        torch.testing.assert_close(
            sample["atlas_root_bounds"],
            torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]),
            atol=0.0,
            rtol=0.0,
        )
        self.assertTrue(torch.equal(sample["surface_voxel_indices"], sample["trellis_feature_indices"]))
        self.assertTrue(torch.equal(sample["trellis_feature_indices"], sample["trellis_latent_coords"]))
        self.assertEqual(
            sample["dino_pseudo_provenance"],
            "pretrained_dinov2_surface_feature",
        )
        self.assertEqual(
            sample["trellis_latent_pseudo_provenance"],
            "pretrained_trellis_structured_latent_encoder",
        )
        self.assertGreater(float(sample["dino_pseudo_confidence"]), 0.0)
        self.assertGreater(float(sample["trellis_latent_pseudo_confidence"]), 0.0)
        self.assertTrue(bool(sample["dino_pseudo_supervision_mask"]))
        self.assertTrue(bool(sample["trellis_latent_pseudo_supervision_mask"]))
        self.assertFalse(bool(sample["topology_supervision_mask"]))
        self.assertFalse(bool(sample["topology_betti_supervision_mask"]))
        self.assertFalse(bool(sample["topology_persistence_supervision_mask"]))
        self.assertFalse(bool(sample["topology_stratum_supervision_mask"]))
        self.assertFalse(bool(sample["source_manifold_certification_mask"]))
        self.assertEqual(float(sample["topology_supervision_confidence"]), 0.0)
        self.assertIsNone(sample["topology_target_betti_z2"])
        self.assertEqual(sample["topology_label_provenance"], "unavailable")
        batch = meshfleet_single_object_collate([sample])
        self.assertEqual(tuple(batch["extrinsics_world_to_camera"].shape), (1, 3, 3, 4))
        self.assertEqual(tuple(batch["atlas_root_bounds"].shape), (1, 2, 3))

    @unittest.skipUnless(
        LOCAL_DATASET.is_dir()
        and torch.cuda.is_available()
        and importlib.util.find_spec("nvdiffrast") is not None,
        "mesh target rasterization requires the A800 nvdiffrast environment",
    )
    def test_nonmanifold_mesh_still_derives_depth_and_normals(self) -> None:
        from graft_gs.data.mesh_supervision import MeshGroundTruthRasterizer

        dataset = MeshFleetObjectDataset(
            MeshFleetDatasetConfig(
                root=LOCAL_DATASET,
                manifest=AUDIT_MANIFEST,
                split="test",
                input_view_set="renders",
                image_size=(64, 64),
                maximum_views=2,
                load_surface_voxels=True,
            )
        )
        sample = dataset[_audited_dataset_index(dataset)]
        target = MeshGroundTruthRasterizer(torch.device("cuda"))(
            sample["modality_paths"]["render_mesh"],
            sample["extrinsics_world_to_camera"].cuda(),
            sample["intrinsics"].cuda(),
            64,
            64,
        )
        self.assertEqual(tuple(target.depth.shape), (2, 1, 64, 64))
        self.assertEqual(tuple(target.normal.shape), (2, 3, 64, 64))
        self.assertTrue(bool(torch.any(target.visibility)))
        self.assertTrue(torch.isfinite(target.depth).all())
        self.assertTrue(torch.isfinite(target.normal).all())

    def test_invalid_topology_cannot_activate_betti_loss(self) -> None:
        output = SimpleNamespace(
            vggt=SimpleNamespace(
                images=torch.zeros(1, 1, 3, 1, 1),
                depth=torch.zeros(1, 1, 1, 1, 1),
            ),
            scenes=[object()],
        )
        batch = {
            "topology_supervision_mask": torch.tensor([False]),
            "topology_betti_supervision_mask": torch.tensor([False]),
            "topology_supervision_confidence": torch.tensor([0.0]),
            "topology_label_provenance": "unavailable",
            "topology_target_betti_z2": None,
        }
        loss = provenance_weighted_topology_supervision(output, batch)
        self.assertEqual(float(loss), 0.0)
        with self.assertRaises(ValueError):
            provenance_weighted_topology_supervision(
                output,
                {**batch, "topology_target_betti_z2": torch.tensor([[0, 0, 0]])},
            )


class DerivedSurfaceTargetTest(unittest.TestCase):
    def test_screened_projection_satisfies_normal_equation(self) -> None:
        dtype = torch.float64
        reference = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=dtype
        )
        nearest = reference + torch.tensor([0.0, 0.2, 0.0], dtype=dtype)
        edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
        config = SurfaceTargetConfig(
            screening_weight=0.5,
            conjugate_gradient_iterations=32,
            conjugate_gradient_tolerance=1.0e-12,
        )
        result = screened_surface_projection(reference, nearest, edges, config)
        torch.testing.assert_close(result, nearest, atol=1.0e-11, rtol=1.0e-11)

    def test_quantization_aware_surface_likelihood_has_gradients(self) -> None:
        dtype = torch.float64
        position = torch.tensor([[0.01, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=dtype, requires_grad=True)
        covariance = (0.02 * torch.eye(3, dtype=dtype)).expand(2, 3, 3).clone().requires_grad_()
        confidence = torch.tensor([0.8, 0.2], dtype=dtype, requires_grad=True)
        surface = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=dtype)
        nll, brier = evidence_surface_calibration(
            position,
            covariance,
            confidence,
            surface,
            cell_size=1.0 / 64.0,
        )
        (nll + brier).backward()
        for value in (position.grad, covariance.grad, confidence.grad):
            self.assertIsNotNone(value)
            self.assertTrue(torch.isfinite(value).all())

    def test_phase_a_stops_after_evidence_and_backpropagates_calibration(self) -> None:
        class SyntheticVGGT(nn.Module):
            def forward(self, images: torch.Tensor) -> VGGTGeometryOutput:
                b, k, _, h, w = images.shape
                extrinsics = torch.eye(4, dtype=images.dtype, device=images.device)[:3].expand(b, k, 3, 4).clone()
                extrinsics[:, 1, 0, 3] = -0.2
                intrinsics = torch.eye(3, dtype=images.dtype, device=images.device).expand(b, k, 3, 3).clone()
                intrinsics[..., 0, 0] = 4.0
                intrinsics[..., 1, 1] = 4.0
                intrinsics[..., 0, 2] = w / 2
                intrinsics[..., 1, 2] = h / 2
                return VGGTGeometryOutput(
                    images=images,
                    patch_features=torch.ones(b, k, 4, 8, dtype=images.dtype, device=images.device),
                    extrinsics_world_to_camera=extrinsics,
                    intrinsics=intrinsics,
                    depth=torch.ones(b, k, h, w, 1, dtype=images.dtype, device=images.device),
                    depth_confidence=torch.full((b, k, h, w), 2.0, dtype=images.dtype, device=images.device),
                    world_points=torch.zeros(b, k, h, w, 3, dtype=images.dtype, device=images.device),
                    world_points_confidence=torch.ones(b, k, h, w, dtype=images.dtype, device=images.device),
                )

        model = GraftGS(
            SyntheticVGGT(),
            GraftGSConfig(feature_dim=8, encoder_layers=1, renderer_backend="reference"),
        ).double()
        images = torch.zeros(1, 2, 3, 4, 4, dtype=torch.float64)
        output = model(images, execution_stage="evidence_calibration")
        self.assertEqual(output.scenes, [])
        self.assertEqual(len(output.evidence_particles), 1)
        surface = output.evidence_particles[0].positions.detach().clone()
        total, terms = GraftGSLoss()(model, output, {
            "surface_voxel_centers": surface,
            "surface_cell_size": 1.0 / 64.0,
        }, "A")
        self.assertEqual(set(terms), {"surface_uncertainty_nll", "confidence_brier"})
        total.backward()
        gradients = [parameter.grad for parameter in model.evidence_builder.calibrator.parameters()]
        self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients))


if __name__ == "__main__":
    unittest.main()

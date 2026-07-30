"""Object-batching contracts that protect joint-view numerical semantics."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from graft_gs.data import (
    DistributedViewCountBatchSampler,
    meshfleet_object_collate,
    ViewCountBatchSampler,
)
from graft_gs.integration.vggt_adapter import (
    align_vggt_to_supervised_cameras,
    VGGTGeometryOutput,
)
from graft_gs.integration.trellis_prior import TrellisPriorAdapter


class _ViewDataset:
    def __init__(self, counts: list[int]) -> None:
        self.counts = counts

    def __len__(self) -> int:
        return len(self.counts)

    def view_count(self, index: int) -> int:
        return self.counts[index]


def _meshfleet_item(object_id: str, views: int, surface_points: int) -> dict[str, object]:
    return {
        "object_id": object_id,
        "images": torch.rand(views, 3, 8, 8),
        "alpha": torch.ones(views, 1, 8, 8),
        "evidence_mask": torch.ones(views, 1, 8, 8, dtype=torch.bool),
        "valid_mask": torch.ones(views, 1, 8, 8, dtype=torch.bool),
        "extrinsics_world_to_camera": torch.eye(4)[None, :3].repeat(
            views, 1, 1
        ),
        "intrinsics": torch.eye(3)[None].repeat(views, 1, 1),
        "atlas_root_bounds": torch.tensor(
            [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        ),
        "surface_voxel_centers": torch.rand(surface_points, 3),
        "surface_cell_size": 1.0 / 64.0,
        "topology_betti_supervision_mask": torch.tensor(False),
        "topology_target_betti_z2": None,
        "topology_label_provenance": "unavailable",
    }


class ObjectBatchCollateTest(unittest.TestCase):
    def test_equal_view_objects_batch_without_padding_variable_surfaces(self) -> None:
        batch = meshfleet_object_collate(
            [_meshfleet_item("a", 4, 7), _meshfleet_item("b", 4, 11)]
        )
        self.assertEqual(tuple(batch["images"].shape), (2, 4, 3, 8, 8))
        self.assertEqual(batch["object_id"], ["a", "b"])
        self.assertIsInstance(batch["surface_voxel_centers"], list)
        self.assertEqual(
            [value.shape[0] for value in batch["surface_voxel_centers"]],
            [7, 11],
        )
        self.assertIsNone(batch["topology_target_betti_z2"])

    def test_mixed_view_counts_are_rejected_instead_of_padded(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical view counts"):
            meshfleet_object_collate(
                [_meshfleet_item("a", 3, 7), _meshfleet_item("b", 4, 7)]
            )

    def test_view_count_samplers_emit_homogeneous_batches(self) -> None:
        dataset = _ViewDataset([3, 4, 3, 5, 4, 3, 5])
        local = list(
            ViewCountBatchSampler(dataset, 2, shuffle=True, seed=9)
        )
        self.assertTrue(local)
        self.assertEqual(len(local[0]), 2)
        for batch in local:
            self.assertEqual(
                len({dataset.view_count(index) for index in batch}), 1
            )

        rank_batches = [
            list(
                DistributedViewCountBatchSampler(
                    dataset,
                    2,
                    num_replicas=2,
                    rank=rank,
                    shuffle=True,
                    seed=9,
                )
            )
            for rank in range(2)
        ]
        self.assertEqual(len(rank_batches[0]), len(rank_batches[1]))
        for left, right in zip(*rank_batches):
            self.assertEqual(len(left), 2)
            self.assertEqual(len(right), 2)
            self.assertEqual(
                {dataset.view_count(index) for index in left},
                {dataset.view_count(index) for index in right},
            )


class MaskedCameraAlignmentTest(unittest.TestCase):
    @staticmethod
    def _geometry(centers: torch.Tensor) -> VGGTGeometryOutput:
        views = centers.shape[0]
        rotation = torch.eye(3).repeat(views, 1, 1)
        extrinsics = torch.cat((rotation, -centers[..., None]), dim=-1)[None]
        intrinsics = torch.eye(3)[None, None].repeat(1, views, 1, 1)
        return VGGTGeometryOutput(
            images=torch.zeros(1, views, 3, 2, 2),
            patch_features=torch.zeros(1, views, 1, 2),
            extrinsics_world_to_camera=extrinsics,
            intrinsics=intrinsics,
            depth=torch.ones(1, views, 2, 2, 1),
            depth_confidence=torch.ones(1, views, 2, 2),
            world_points=torch.zeros(1, views, 2, 2, 3),
            world_points_confidence=torch.ones(1, views, 2, 2),
        )

    def test_masked_extra_camera_does_not_change_alignment(self) -> None:
        centers = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        reference = self._geometry(centers)
        aligned, diagnostics = align_vggt_to_supervised_cameras(
            reference,
            reference.extrinsics_world_to_camera,
            reference.intrinsics,
        )
        padded_centers = torch.cat((centers, torch.tensor([[50.0, -20.0, 7.0]])))
        padded = self._geometry(padded_centers)
        padded_aligned, padded_diagnostics = align_vggt_to_supervised_cameras(
            padded,
            padded.extrinsics_world_to_camera,
            padded.intrinsics,
            valid_view_mask=torch.tensor([[True, True, True, False]]),
        )
        torch.testing.assert_close(
            padded_aligned.extrinsics_world_to_camera[:, :3],
            aligned.extrinsics_world_to_camera,
        )
        torch.testing.assert_close(
            padded_diagnostics.scale, diagnostics.scale
        )
        torch.testing.assert_close(
            padded_diagnostics.center_rmse, diagnostics.center_rmse
        )


class FrozenPriorBatchLifetimeTest(unittest.TestCase):
    def test_nested_sampling_session_offloads_once_at_outer_boundary(self) -> None:
        adapter = TrellisPriorAdapter(
            pipeline=None,
            samples=1,
            sampler_steps=1,
        )
        adapter._pipeline_device = torch.device("cuda", 0)
        with (
            mock.patch.object(adapter, "_offload_cuda_pipeline") as offload,
            mock.patch.object(adapter, "_release_inactive_cuda_cache") as release,
        ):
            with adapter.sampling_session():
                with adapter.sampling_session():
                    offload.assert_not_called()
                    release.assert_not_called()
            offload.assert_called_once_with(torch.device("cuda", 0))
            release.assert_called_once_with(torch.device("cuda", 0))


if __name__ == "__main__":
    unittest.main()

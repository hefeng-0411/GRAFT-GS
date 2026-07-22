"""Multi-rank A800 verification for the exact same-object evidence gather."""

from __future__ import annotations

import os
import random
import unittest
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

from graft_gs.engine.trainer import (
    AtlasDDPSynchronizer,
    DistributedContext,
    GraftGSTrainer,
    _broadcast_discrete_exact,
    _capture_rng_state,
    _restore_rng_state,
)
from graft_gs.geometry.atlas import AtlasConfig, PersistentOctreeAtlas
from graft_gs.mapping.manifold_mapping import EvidenceParticles
from graft_gs.integration.trellis_prior import TrellisPriorMeasure


class AtlasSynchronizationTransportTest(unittest.TestCase):
    def test_nonfinite_gradient_guard_fails_before_optimizer_step(self) -> None:
        trainer = object.__new__(GraftGSTrainer)
        trainer.context = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
        )
        trainer._assert_finite_tensors(
            "unit finite state", {"finite": torch.tensor([1.0, -2.0])}
        )
        with self.assertRaisesRegex(
            FloatingPointError, "bad_parameter"
        ):
            trainer._assert_finite_tensors(
                "optimizer step",
                {"bad_parameter": torch.tensor([float("nan")])},
            )

    def test_int64_transport_preserves_values_beyond_float_exactness(self) -> None:
        storage = torch.tensor(
            [
                [(1 << 53) + 1, (1 << 62) - 1],
                [(1 << 54) + 3, -((1 << 55) + 5)],
            ],
            dtype=torch.int64,
        )
        value = storage.T
        self.assertFalse(value.is_contiguous())
        local_before = value.clone()
        received = torch.tensor(
            [[101, 202], [303, 404]],
            dtype=torch.int64,
        )
        observed: list[torch.Tensor] = []

        def broadcast(tensor: torch.Tensor, src: int) -> None:
            self.assertEqual(src, 0)
            self.assertEqual(tensor.dtype, torch.int64)
            self.assertTrue(tensor.is_contiguous())
            observed.append(tensor.clone())
            tensor.copy_(received)

        with mock.patch(
            "graft_gs.engine.trainer.dist.broadcast",
            side_effect=broadcast,
        ):
            restored = _broadcast_discrete_exact(value, source_rank=0)
        self.assertTrue(torch.equal(restored, received))
        self.assertEqual(restored.dtype, torch.int64)
        self.assertEqual(len(observed), 1)
        self.assertTrue(torch.equal(observed[0], local_before))
        self.assertTrue(
            torch.equal(value, local_before),
            "collective transport aliased and overwrote rank-local int64 state",
        )

    def test_discrete_atlas_and_split_mask_use_nccl_safe_int64_transport(self) -> None:
        points = torch.tensor(
            [
                [-0.25, -0.20, -0.10],
                [0.20, -0.15, 0.10],
                [-0.10, 0.25, 0.15],
                [0.25, 0.20, -0.20],
            ],
            dtype=torch.float64,
        )
        atlas = PersistentOctreeAtlas.from_evidence(
            points,
            config=AtlasConfig(base_level=1, max_level=2),
        )
        discrete_names = (
            "levels",
            "morton_codes",
            "parent",
            "child_slot",
            "active",
            "point_count",
            "prior_point_count",
            "edge_index",
        )
        expected = {
            name: getattr(atlas, name).clone()
            for name in discrete_names
        }
        self.assertEqual(atlas.levels.dtype, torch.int16)
        self.assertEqual(atlas.child_slot.dtype, torch.int8)
        self.assertEqual(atlas.active.dtype, torch.bool)

        broadcast_dtypes: list[torch.dtype] = []

        def broadcast(tensor: torch.Tensor, src: int) -> None:
            self.assertEqual(src, 0)
            broadcast_dtypes.append(tensor.dtype)
            if not tensor.dtype.is_floating_point:
                self.assertEqual(
                    tensor.dtype,
                    torch.int64,
                    "NCCL-incompatible discrete dtype reached broadcast",
                )

        def all_gather_object(output: list[object], value: object) -> None:
            output[:] = [value, value]

        def all_reduce(tensor: torch.Tensor, op: object = None) -> None:
            del tensor, op

        context = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=2,
            device=torch.device("cpu"),
        )
        synchronizer = AtlasDDPSynchronizer(context)
        with (
            mock.patch(
                "graft_gs.engine.trainer.dist.all_gather_object",
                side_effect=all_gather_object,
            ),
            mock.patch(
                "graft_gs.engine.trainer.dist.broadcast",
                side_effect=broadcast,
            ),
            mock.patch(
                "graft_gs.engine.trainer.dist.all_reduce",
                side_effect=all_reduce,
            ),
            mock.patch(
                "graft_gs.engine.trainer.dist_nn.broadcast",
                side_effect=lambda tensor, src: tensor,
            ),
        ):
            synchronized = synchronizer.synchronize_atlas(atlas)
            split_mask = torch.tensor([True, False, True], dtype=torch.bool)
            synchronized_mask = synchronizer.synchronize_split_mask(split_mask)

        for name, reference in expected.items():
            value = getattr(synchronized, name)
            self.assertEqual(value.dtype, reference.dtype, name)
            self.assertTrue(torch.equal(value, reference), name)
        self.assertEqual(synchronized_mask.dtype, torch.bool)
        self.assertTrue(torch.equal(synchronized_mask, split_mask))
        self.assertIn(torch.int64, broadcast_dtypes)
        self.assertNotIn(torch.int16, broadcast_dtypes)
        self.assertNotIn(torch.int8, broadcast_dtypes)
        self.assertNotIn(torch.bool, broadcast_dtypes)

    def test_atlas_metadata_mismatch_fails_before_typed_collectives(self) -> None:
        atlas = PersistentOctreeAtlas.from_evidence(
            torch.tensor(
                [[-0.2, -0.1, 0.0], [0.2, 0.1, 0.0]],
                dtype=torch.float64,
            ),
            config=AtlasConfig(base_level=1, max_level=2),
        )
        context = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=2,
            device=torch.device("cpu"),
        )

        def disagree(output: list[object], value: object) -> None:
            output[:] = [value, {"incompatible_rank": 1}]

        with (
            mock.patch(
                "graft_gs.engine.trainer.dist.all_gather_object",
                side_effect=disagree,
            ),
            mock.patch(
                "graft_gs.engine.trainer.dist.broadcast",
            ) as broadcast,
        ):
            with self.assertRaisesRegex(RuntimeError, "metadata mismatch before typed collectives"):
                AtlasDDPSynchronizer(context).synchronize_atlas(atlas)
        broadcast.assert_not_called()


@unittest.skipUnless(int(os.environ.get("WORLD_SIZE", "1")) > 1, "launch with torchrun")
class DistributedEvidenceTest(unittest.TestCase):
    def test_source_atlas_broadcast_routes_global_evidence_gradient_to_every_rank(self) -> None:
        context = DistributedContext.initialize()
        synchronizer = AtlasDDPSynchronizer(context)
        local_points = torch.tensor(
            [
                [-0.31, -0.20, -0.09],
                [0.27, -0.14, 0.12],
                [-0.08, 0.29, 0.17],
                [0.23, 0.18, -0.24],
            ],
            dtype=torch.float32,
            device=context.device,
        ) + torch.tensor(
            [0.017 * context.rank, -0.011 * context.rank, 0.007 * context.rank],
            dtype=torch.float32,
            device=context.device,
        )
        local_points.requires_grad_(True)
        local_mass = torch.ones(
            local_points.shape[0], dtype=local_points.dtype, device=context.device
        )
        points, mass = synchronizer.aggregate_atlas_measure(local_points, local_mass)
        atlas = PersistentOctreeAtlas.from_evidence(
            points,
            mass,
            config=AtlasConfig(base_level=0, max_level=1),
        )

        # A pi tangent-gauge rotation represents the same chart geometry but
        # used to trigger the invalid raw-frame replica comparison.
        if context.rank != 0:
            gauge = torch.diag(
                torch.tensor(
                    [-1.0, -1.0, 1.0],
                    dtype=atlas.chart_frames.dtype,
                    device=context.device,
                )
            )
            atlas.chart_frames = atlas.chart_frames @ gauge

        continuous_names = (
            "root_min",
            "root_max",
            "cell_centers",
            "cell_sides",
            "chart_centers",
            "chart_frames",
            "chart_covariance",
            "curvature",
            "chart_radii",
            "evidence_mass",
            "prior_mass",
            "prior_mass_variance",
            "overlap_rotation",
            "overlap_translation",
        )
        source_reference: dict[str, torch.Tensor] = {}
        chart_frames_before: list[torch.Tensor] = []
        for name in continuous_names:
            local_value = getattr(atlas, name).detach()
            gathered_before = [
                torch.empty_like(local_value) for _ in range(context.world_size)
            ]
            dist.all_gather(gathered_before, local_value)
            source_reference[name] = gathered_before[0].clone()
            if name == "chart_frames":
                chart_frames_before = gathered_before
        self.assertTrue(
            any(
                not torch.equal(value, chart_frames_before[0])
                for value in chart_frames_before[1:]
            ),
            "rank-local gauge rotation did not change the test fixture",
        )

        synchronized = synchronizer.synchronize_atlas(atlas)
        for name in continuous_names:
            forward_value = getattr(synchronized, name)
            self.assertTrue(
                torch.equal(forward_value.detach(), source_reference[name]),
                f"continuous atlas field {name} differs from its source-rank value",
            )
            gathered_forward = [
                torch.empty_like(forward_value) for _ in range(context.world_size)
            ]
            dist.all_gather(gathered_forward, forward_value.detach())
            for value in gathered_forward:
                self.assertTrue(
                    torch.equal(value, gathered_forward[0]),
                    f"continuous atlas field {name} is not bitwise source-identical",
                )

        frame_probe = torch.tensor(
            [[0.7, -0.2, 0.1], [0.3, 0.4, -0.5], [-0.1, 0.2, 0.6]],
            dtype=local_points.dtype,
            device=context.device,
        )
        objective = (
            synchronized.chart_centers.square().sum()
            + 1.0e-3 * (synchronized.chart_frames * frame_probe).sum()
        )
        objective.backward()
        self.assertIsNotNone(local_points.grad)
        self.assertTrue(torch.all(torch.isfinite(local_points.grad)))
        self.assertGreater(float(local_points.grad.abs().sum()), 0.0)
        gathered_gradient = [
            torch.empty_like(local_points.grad)
            for _ in range(context.world_size)
        ]
        dist.all_gather(gathered_gradient, local_points.grad)
        self.assertTrue(
            all(
                bool(torch.all(torch.isfinite(value)))
                and float(value.abs().sum()) > 0.0
                for value in gathered_gradient
            ),
            "source-owned atlas gradient did not return through global all-gather",
        )

    def test_rank_local_rng_restore_does_not_collapse_streams(self) -> None:
        context = DistributedContext.initialize()
        seed = 1907 + context.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        local_state = _capture_rng_state(context.rank)
        gathered_states: list[object] = [None for _ in range(context.world_size)]
        dist.all_gather_object(gathered_states, local_state)

        expected_cpu = torch.rand(8)
        expected_cuda = torch.rand(8, device=context.device)
        expected_numpy = np.random.rand(8)
        expected_python = [random.random() for _ in range(8)]
        _ = torch.rand(31)
        _ = torch.rand(31, device=context.device)
        _ = np.random.rand(31)
        _ = [random.random() for _ in range(31)]
        _restore_rng_state(gathered_states[context.rank], context.rank)
        torch.testing.assert_close(torch.rand(8), expected_cpu, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            torch.rand(8, device=context.device),
            expected_cuda,
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_array_equal(np.random.rand(8), expected_numpy)
        self.assertEqual([random.random() for _ in range(8)], expected_python)

        rank_sample = expected_cuda[:1].clone()
        all_samples = [torch.empty_like(rank_sample) for _ in range(context.world_size)]
        dist.all_gather(all_samples, rank_sample)
        self.assertGreater(
            torch.unique(torch.cat(all_samples)).numel(),
            1,
            "rank-local streams collapsed to one sequence",
        )

    def test_complete_autograd_evidence_measure(self) -> None:
        context = DistributedContext.initialize()
        self.assertTrue(context.distributed)
        dtype = torch.float32
        rank = float(context.rank)
        position = torch.tensor(
            [[rank, 0.0, 1.0], [rank, 1.0, 1.0]], device=context.device, dtype=dtype
        ).requires_grad_()
        feature = torch.tensor(
            [[rank + 1.0, 2.0], [rank + 2.0, 3.0]], device=context.device, dtype=dtype
        ).requires_grad_()
        covariance = torch.eye(3, device=context.device, dtype=dtype).expand(2, 3, 3).clone()
        camera_extrinsic = (
            torch.eye(4, device=context.device, dtype=dtype)[:3]
            .expand(2, -1, -1)
            .clone()
            .requires_grad_()
        )
        camera_intrinsic = (
            torch.eye(3, device=context.device, dtype=dtype)
            .expand(2, -1, -1)
            .clone()
            .requires_grad_()
        )
        evidence = EvidenceParticles(
            positions=position,
            rays=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=context.device),
            features=feature,
            covariance=covariance,
            confidence=torch.full((2,), 0.8, device=context.device),
            mass=torch.full((2,), 0.5, device=context.device),
            view_index=torch.tensor([0, 1], dtype=torch.int64, device=context.device),
            pixel_uv=torch.tensor([[0.5, 0.5], [1.5, 0.5]], device=context.device),
            extrinsics_world_to_camera=camera_extrinsic,
            intrinsics=camera_intrinsic,
            depth_variance=torch.full((2,), 0.1, device=context.device),
            colors=torch.full((2, 3), 0.5, device=context.device),
        )
        global_evidence = AtlasDDPSynchronizer(context).aggregate_evidence(evidence)
        self.assertEqual(global_evidence.positions.shape[0], 2 * context.world_size)
        self.assertEqual(torch.unique(global_evidence.view_index).numel(), 2 * context.world_size)
        gathered_positions = [torch.empty_like(global_evidence.positions) for _ in range(context.world_size)]
        dist.all_gather(gathered_positions, global_evidence.positions.detach())
        for value in gathered_positions[1:]:
            torch.testing.assert_close(value, gathered_positions[0], atol=0.0, rtol=0.0)
        loss = (
            global_evidence.features.square().sum()
            + global_evidence.positions.square().sum()
            + global_evidence.extrinsics_world_to_camera.square().sum()
            + global_evidence.intrinsics.square().sum()
        )
        loss.backward()
        torch.testing.assert_close(
            feature.grad,
            2.0 * context.world_size * feature.detach(),
        )
        torch.testing.assert_close(
            position.grad,
            2.0 * context.world_size * position.detach(),
        )
        torch.testing.assert_close(
            camera_extrinsic.grad,
            2.0 * context.world_size * camera_extrinsic.detach(),
        )
        torch.testing.assert_close(
            camera_intrinsic.grad,
            2.0 * context.world_size * camera_intrinsic.detach(),
        )

    def test_rank_zero_hidden_support_broadcast(self) -> None:
        context = DistributedContext.initialize()
        local_count = 2
        coordinates = torch.arange(local_count * 3, device=context.device).reshape(
            local_count, 3
        )
        measure = (
            TrellisPriorMeasure(
                coordinates=coordinates,
                positions=coordinates.to(torch.float64) / 16.0 - 0.5,
                probability=torch.linspace(
                    0.6, 0.75, local_count, device=context.device, dtype=torch.float64
                ),
                mass=torch.linspace(
                    0.01, 0.02, local_count, device=context.device, dtype=torch.float64
                ),
                mass_variance=torch.linspace(
                    1.0e-5,
                    2.0e-5,
                    local_count,
                    device=context.device,
                    dtype=torch.float64,
                ),
                vote_count=torch.arange(
                    1, local_count + 1, dtype=torch.int64, device=context.device
                ),
                sample_count=8,
                resolution=64,
            )
            if context.rank == 0
            else None
        )
        synchronizer = AtlasDDPSynchronizer(context)
        self.assertEqual(synchronizer.should_sample_trellis_prior(), context.rank == 0)
        synchronized = synchronizer.synchronize_trellis_prior_measure(
            measure,
            dtype=torch.float64,
        )
        self.assertEqual(synchronized.positions.shape, (2, 3))
        self.assertEqual(synchronized.positions.dtype, torch.float64)
        self.assertEqual(synchronized.sample_count, 8)
        self.assertEqual(synchronized.resolution, 64)
        gathered = [
            torch.empty_like(synchronized.positions) for _ in range(context.world_size)
        ]
        dist.all_gather(gathered, synchronized.positions)
        for value in gathered[1:]:
            torch.testing.assert_close(value, gathered[0], atol=0.0, rtol=0.0)

    def test_trellis_conditioning_uses_all_view_shards(self) -> None:
        context = DistributedContext.initialize()
        local = torch.full(
            (context.rank + 1, 3, 2, 2),
            float(context.rank),
            device=context.device,
        )
        combined = AtlasDDPSynchronizer(context).aggregate_prior_images(local)
        self.assertEqual(
            combined.shape[0], context.world_size * (context.world_size + 1) // 2
        )
        cursor = 0
        for rank in range(context.world_size):
            torch.testing.assert_close(
                combined[cursor : cursor + rank + 1],
                torch.full_like(combined[cursor : cursor + rank + 1], float(rank)),
            )
            cursor += rank + 1


if __name__ == "__main__":
    unittest.main()

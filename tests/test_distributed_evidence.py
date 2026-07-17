"""Multi-rank A800 verification for the exact same-object evidence gather."""

from __future__ import annotations

import os
import random
import unittest

import numpy as np
import torch
import torch.distributed as dist

from graft_gs.engine.trainer import (
    AtlasDDPSynchronizer,
    DistributedContext,
    _capture_rng_state,
    _restore_rng_state,
)
from graft_gs.mapping.manifold_mapping import EvidenceParticles
from graft_gs.integration.trellis_prior import TrellisPriorMeasure


@unittest.skipUnless(int(os.environ.get("WORLD_SIZE", "1")) > 1, "launch with torchrun")
class DistributedEvidenceTest(unittest.TestCase):
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
        local_count = 2 if context.rank == 0 else 1
        coordinates = torch.arange(local_count * 3, device=context.device).reshape(
            local_count, 3
        )
        measure = TrellisPriorMeasure(
            coordinates=coordinates,
            positions=coordinates.to(torch.float32) / 16.0 - 0.5,
            probability=torch.linspace(
                0.6, 0.75, local_count, device=context.device
            ),
            mass=torch.linspace(0.01, 0.02, local_count, device=context.device),
            mass_variance=torch.linspace(
                1.0e-5, 2.0e-5, local_count, device=context.device
            ),
            vote_count=torch.arange(
                1, local_count + 1, dtype=torch.int64, device=context.device
            ),
            sample_count=8 + context.rank,
            resolution=64,
        )
        synchronized = AtlasDDPSynchronizer(
            context
        ).synchronize_trellis_prior_measure(measure)
        self.assertEqual(synchronized.positions.shape, (2, 3))
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

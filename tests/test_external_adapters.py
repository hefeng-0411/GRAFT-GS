"""CPU contract tests for released-model adapter boundary behavior."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from graft_gs.integration.trellis_prior import TrellisPriorAdapter
from graft_gs.integration.vggt_adapter import VGGTAdapter


class _NeverCalledAggregator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frame_blocks = nn.ModuleList()
        self.global_blocks = nn.ModuleList()
        self.cached_layer_indices = (4, 11, 17, 23)

    def forward(self, images: torch.Tensor):
        raise AssertionError("invalid VGGT input reached the upstream model")


class _MockVGGT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aggregator = _NeverCalledAggregator()
        self.camera_head = SimpleNamespace(token_norm=nn.LayerNorm(2048))
        self.depth_head = object()
        self.point_head = object()


class _MockTrellisPipeline:
    def __init__(self) -> None:
        self.models = {
            "image_cond_model": object(),
            "sparse_structure_flow_model": SimpleNamespace(resolution=8),
            "sparse_structure_decoder": object(),
        }
        self.sparse_structure_sampler = SimpleNamespace(sample=lambda *args: None)
        self.injection_count = 0
        self.active_injections = 0
        self.sample_count = 0

    def to(self, device):
        return self

    def get_cond(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "cond": torch.ones(images.shape[0], 1, 2),
            "neg_cond": torch.zeros(images.shape[0], 1, 2),
        }

    @contextmanager
    def inject_sampler_multi_image(self, name, views, steps, mode):
        self.injection_count += 1
        self.active_injections += 1
        try:
            yield
        finally:
            self.active_injections -= 1

    def sample_sparse_structure(self, condition, num_samples, parameters):
        if self.active_injections != 1:
            raise RuntimeError("posterior draw did not own exactly one injection context")
        self.sample_count += 1
        offset = self.sample_count % 4
        return torch.tensor(
            [[0, offset, 1, 2], [0, offset + 1, 1, 2]],
            dtype=torch.int32,
        )


class TrellisAdapterBoundaryTest(unittest.TestCase):
    def test_multi_image_context_is_recreated_for_every_posterior_draw(self) -> None:
        pipeline = _MockTrellisPipeline()
        adapter = TrellisPriorAdapter(pipeline, samples=3, sampler_steps=2)
        initial_rng = torch.random.get_rng_state()
        prior = adapter.sample(torch.zeros(2, 3, 8, 8), seed=7)
        self.assertEqual(len(prior.coordinates), 3)
        self.assertEqual(pipeline.injection_count, 3)
        self.assertEqual(pipeline.sample_count, 3)
        self.assertEqual(pipeline.active_injections, 0)
        self.assertTrue(torch.equal(initial_rng, torch.random.get_rng_state()))

    def test_exact_conditioning_cache_avoids_repeated_frozen_sampling(self) -> None:
        pipeline = _MockTrellisPipeline()
        adapter = TrellisPriorAdapter(
            pipeline,
            samples=2,
            sampler_steps=2,
            cache_entries=2,
        )
        images = torch.zeros(2, 3, 8, 8)
        first = adapter.sample(images, seed=19)
        first_calls = pipeline.sample_count
        second = adapter.sample(images.clone(), seed=19)
        self.assertEqual(pipeline.sample_count, first_calls)
        for expected, actual in zip(first.coordinates, second.coordinates):
            torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        changed = images.clone()
        changed[0, 0, 0, 0] = 1.0e-4
        adapter.sample(changed, seed=19)
        self.assertEqual(pipeline.sample_count, first_calls + adapter.samples)
        adapter.sample(images, seed=20)
        self.assertEqual(pipeline.sample_count, first_calls + 2 * adapter.samples)

    def test_cache_identity_hashes_bfloat16_as_exact_raw_bytes(self) -> None:
        adapter = TrellisPriorAdapter(
            _MockTrellisPipeline(),
            samples=1,
            sampler_steps=1,
        )
        images = torch.zeros(1, 3, 4, 4, dtype=torch.bfloat16)
        key = adapter._sample_cache_key(images, seed=3)
        self.assertEqual(key, adapter._sample_cache_key(images.clone(), seed=3))
        changed = images.clone()
        changed[0, 0, 0, 0] = torch.tensor(0.125, dtype=torch.bfloat16)
        self.assertNotEqual(key, adapter._sample_cache_key(changed, seed=3))

    def test_tensor_image_domain_is_rejected_before_upstream_sampling(self) -> None:
        adapter = TrellisPriorAdapter(_MockTrellisPipeline(), samples=1, sampler_steps=1)
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            adapter.sample(torch.full((1, 3, 4, 4), 1.1))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            images = torch.zeros(1, 3, 4, 4)
            images[0, 0, 0, 0] = float("nan")
            adapter.sample(images)
        with self.assertRaisesRegex(TypeError, "floating-point"):
            adapter.sample(torch.zeros(1, 3, 4, 4, dtype=torch.uint8))


class VGGTAdapterBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = VGGTAdapter(_MockVGGT(), feature_dim=8)

    def test_tensor_image_domain_is_rejected_before_upstream_inference(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            self.adapter(torch.full((1, 1, 3, 4, 4), 1.1))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            images = torch.zeros(1, 1, 3, 4, 4)
            images[0, 0, 0, 0, 0] = float("nan")
            self.adapter(images)
        with self.assertRaisesRegex(TypeError, "floating-point"):
            self.adapter(torch.zeros(1, 1, 3, 4, 4, dtype=torch.uint8))

    def test_empty_view_or_non_rgb_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one scene/view"):
            self.adapter(torch.zeros(1, 0, 3, 4, 4))
        with self.assertRaisesRegex(ValueError, "RGB channels"):
            self.adapter(torch.zeros(1, 1, 1, 4, 4))


if __name__ == "__main__":
    unittest.main()

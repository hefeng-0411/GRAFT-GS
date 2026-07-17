"""Numerical failure tests for production Phase-F gradient purification."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from graft_gs.optimization.gradient_purification import (
    GradientPurificationConfig,
    HilbertGradientPurifier,
    gradient_inner,
    project_to_consensus_cone,
    weighted_geometric_median,
)


def gradient(value: list[float]) -> tuple[torch.Tensor]:
    return (torch.tensor(value, dtype=torch.float64),)


class GradientPurificationTest(unittest.TestCase):
    def test_cone_projection_satisfies_boundary_and_is_nearest(self) -> None:
        axis = gradient([1.0, 0.0])
        sample = gradient([0.0, 1.0])
        projected, accepted = project_to_consensus_cone(
            sample, axis, minimum_cosine=0.5, epsilon=1.0e-12
        )
        self.assertFalse(bool(accepted))
        cosine = gradient_inner(projected, axis) / torch.sqrt(
            gradient_inner(projected, projected) * gradient_inner(axis, axis)
        )
        torch.testing.assert_close(cosine, torch.tensor(0.5, dtype=torch.float64))
        # Any feasible point on the same ray but farther from the analytical
        # boundary projection has a strictly larger Euclidean residual.
        alternative = gradient([0.5, 0.5 * 3.0**0.5])
        projected_error = gradient_inner(
            (projected[0] - sample[0],), (projected[0] - sample[0],)
        )
        alternative_error = gradient_inner(
            (alternative[0] - sample[0],), (alternative[0] - sample[0],)
        )
        self.assertLess(float(projected_error), float(alternative_error))

    def test_weighted_geometric_median_resists_minority_outlier(self) -> None:
        samples = (
            gradient([1.0, 0.0]),
            gradient([1.1, 0.05]),
            gradient([-20.0, 8.0]),
        )
        median, residual = weighted_geometric_median(
            samples,
            torch.tensor([1.0, 1.0, 0.05], dtype=torch.float64),
            iterations=64,
            smoothing=1.0e-10,
        )
        self.assertLess(float(torch.linalg.vector_norm(median[0] - samples[0][0])), 0.15)
        self.assertTrue(bool(torch.isfinite(residual)))

    def test_artifact_subspace_is_removed_and_fisher_state_roundtrips(self) -> None:
        parameter = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        config = GradientPurificationConfig(
            maximum_views=4,
            consensus_cosine=0.1,
            consensus_relative_singular_value=0.01,
            artifact_relative_singular_value=0.01,
            fisher_decay=0.5,
            fisher_damping=1.0,
            fisher_radius=100.0,
        )
        purifier = HilbertGradientPurifier((parameter,), config)
        views = (
            gradient([1.0, 0.8]),
            gradient([1.1, 0.9]),
            gradient([0.9, 1.0]),
        )
        artifacts = (
            gradient([0.0, 1.0]),
            gradient([0.0, 2.0]),
            gradient([0.0, -1.0]),
        )
        purified, diagnostics = purifier.purify(
            views, torch.ones(3, dtype=torch.float64), artifacts
        )
        self.assertGreater(float(purified[0][0]), 0.5)
        self.assertLess(abs(float(purified[0][1])), 1.0e-9)
        self.assertEqual(diagnostics.artifact_rank, 1)
        purifier.commit_fisher()
        state = purifier.state_dict()
        restored = HilbertGradientPurifier((parameter,), config)
        restored.load_state_dict(state)
        torch.testing.assert_close(restored.fisher[0], purifier.fisher[0])
        self.assertEqual(restored.committed_steps, 1)


if __name__ == "__main__":
    unittest.main()

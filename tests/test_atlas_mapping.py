"""Server-side verification for the persistent atlas and manifold OT lift.

These tests are intentionally backend-independent and use float64.  They are
designed to run on the A800 validation environment before sparse-kernel or
mixed-precision optimizations are enabled.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

import torch

from graft_gs.geometry.atlas import (
    AtlasConfig,
    PersistentOctreeAtlas,
    _right_handed_pca_frames,
    morton_decode,
    morton_encode,
)
from graft_gs.mapping.manifold_mapping import (
    EvidenceParticles,
    ImplicitSinkhornConfig,
    ImplicitUnbalancedSinkhorn,
    ManifoldMappingConfig,
    ManifoldMappingOperator,
    SparseTransportGraph,
    sparse_view_reprojection_variance,
)
from graft_gs.integration.trellis_prior import (
    TrellisPriorAdapter,
    TrellisStructurePrior,
)
from graft_gs.integration.pipeline import GraftGS
from graft_gs.equivariant.gsta import active_adjacency


def _surface_evidence(dtype: torch.dtype = torch.float64) -> EvidenceParticles:
    axis = torch.linspace(-0.4, 0.4, 5, dtype=dtype)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    z = 0.08 * (x.square() - y.square())
    positions = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
    count = positions.shape[0]
    rays = torch.nn.functional.normalize(positions + torch.tensor([0.0, 0.0, 2.0], dtype=dtype), dim=-1)
    feature_axis = torch.arange(12, dtype=dtype)
    features = torch.cos(positions[:, :1] * (feature_axis + 1)) + torch.sin(
        positions[:, 1:2] * (feature_axis + 0.5)
    )
    covariance = torch.diag(torch.tensor([2.0e-4, 2.0e-4, 8.0e-4], dtype=dtype)).expand(count, -1, -1).clone()
    confidence = torch.linspace(0.7, 0.95, count, dtype=dtype)
    mass = confidence * 1.0e-3
    return EvidenceParticles(
        positions=positions,
        rays=rays,
        features=features,
        covariance=covariance,
        confidence=confidence,
        mass=mass,
        view_index=torch.arange(count).remainder(3),
        pixel_uv=torch.stack((torch.arange(count, dtype=dtype), torch.zeros(count, dtype=dtype)), dim=-1),
        extrinsics_world_to_camera=torch.eye(4, dtype=dtype)[:3].expand(3, -1, -1).clone(),
        intrinsics=torch.eye(3, dtype=dtype).expand(3, -1, -1).clone(),
        depth_variance=covariance[:, 2, 2],
        colors=torch.sigmoid(features[:, :3]),
    )


class PersistentAtlasTest(unittest.TestCase):
    def test_atlas_rejects_nonfinite_mass_with_specific_diagnostic(self) -> None:
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=torch.float64
        )
        with self.assertRaisesRegex(ValueError, "mass contains non-finite"):
            PersistentOctreeAtlas.from_evidence(
                positions, torch.tensor([1.0, float("nan")], dtype=torch.float64)
            )

    def test_pca_frame_repeated_spectrum_has_finite_zero_gauge_gradient(self) -> None:
        covariance = torch.zeros(2, 3, 3, dtype=torch.float64, requires_grad=True)
        frame = _right_handed_pca_frames(covariance, 1.0e-10, 1.0e-4)
        eye = torch.eye(3, dtype=frame.dtype).expand_as(frame)
        torch.testing.assert_close(frame.transpose(-1, -2) @ frame, eye)
        torch.testing.assert_close(
            torch.linalg.det(frame), torch.ones(2, dtype=frame.dtype)
        )
        probe = torch.arange(18, dtype=frame.dtype).reshape_as(frame)
        gradient = torch.autograd.grad((frame * probe).sum(), covariance)[0]
        self.assertTrue(torch.all(torch.isfinite(gradient)))
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    def test_pca_frame_distinct_spectrum_retains_finite_gradient(self) -> None:
        covariance = torch.tensor(
            [
                [0.20, 0.01, -0.02],
                [0.01, 0.63, 0.03],
                [-0.02, 0.03, 1.17],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        frame = _right_handed_pca_frames(covariance, 1.0e-10, 1.0e-4)
        probe = torch.tensor(
            [[0.3, -0.7, 0.2], [0.4, 0.1, -0.2], [-0.5, 0.6, 0.8]],
            dtype=frame.dtype,
        )
        gradient = torch.autograd.grad((frame * probe).sum(), covariance)[0]
        self.assertTrue(torch.all(torch.isfinite(gradient)))
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_sparse_reprojection_variance_retains_camera_gradient(self) -> None:
        dtype = torch.float64
        position = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]], dtype=dtype)
        atlas = PersistentOctreeAtlas.from_evidence(
            position,
            config=AtlasConfig(base_level=0, max_level=1),
        )
        translation = torch.tensor(0.2, dtype=dtype, requires_grad=True)
        rotation = torch.eye(3, dtype=dtype).expand(2, -1, -1)
        camera_translation = torch.stack(
            (
                torch.zeros(3, dtype=dtype),
                torch.stack((translation, translation * 0.0, translation * 0.0)),
            )
        )
        extrinsic = torch.cat((rotation, camera_translation[..., None]), dim=-1)
        intrinsic = torch.tensor(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        ).expand(2, -1, -1).clone()
        evidence = EvidenceParticles(
            positions=position,
            rays=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=dtype),
            features=torch.zeros(2, 1, dtype=dtype),
            covariance=torch.eye(3, dtype=dtype).expand(2, -1, -1).clone(),
            confidence=torch.ones(2, dtype=dtype),
            mass=torch.ones(2, dtype=dtype),
            view_index=torch.tensor([0, 1], dtype=torch.int64),
            pixel_uv=torch.zeros(2, 2, dtype=dtype),
            extrinsics_world_to_camera=extrinsic,
            intrinsics=intrinsic,
            depth_variance=torch.ones(2, dtype=dtype),
        )
        graph = SparseTransportGraph(
            edge_index=torch.tensor([[0, 0], [0, 1]], dtype=torch.int64),
            atlas_node_index=atlas.active_indices,
            source_count=1,
            target_count=2,
            support_radius=torch.ones(1, dtype=dtype),
        )
        mapping = SimpleNamespace(
            evidence=evidence,
            graph=graph,
            plan=torch.ones(2, dtype=dtype),
            transported_centers=position[:1],
            transported_mass=torch.tensor([2.0], dtype=dtype),
        )
        variance = sparse_view_reprojection_variance(atlas, mapping)
        self.assertGreater(float(variance), 0.0)
        variance.sum().backward()
        self.assertIsNotNone(translation.grad)
        self.assertGreater(abs(float(translation.grad)), 0.0)

    def test_refined_continuous_charts_retain_evidence_gradients(self) -> None:
        evidence = _surface_evidence()
        evidence.positions = evidence.positions.detach().clone().requires_grad_(True)
        atlas = PersistentOctreeAtlas.from_evidence(
            evidence.positions,
            evidence.mass,
            AtlasConfig(base_level=0, max_level=2),
        )
        atlas.refine(
            evidence.positions,
            evidence.mass,
            split_mask=torch.tensor([True]),
        )
        active = atlas.active_indices
        objective = (
            atlas.chart_centers[active].square().sum()
            + 0.01 * atlas.chart_covariance[active].square().sum()
            + 0.01 * atlas.curvature[active].square().sum()
        )
        objective.backward()
        self.assertIsNotNone(evidence.positions.grad)
        self.assertTrue(torch.all(torch.isfinite(evidence.positions.grad)))
        self.assertGreater(float(evidence.positions.grad.abs().sum()), 0.0)

    def test_morton_round_trip(self) -> None:
        xyz = torch.tensor([[0, 0, 0], [1, 2, 3], [17, 9, 31], [1023, 511, 7]])
        self.assertTrue(torch.equal(morton_decode(morton_encode(xyz)), xyz))

    def test_persistent_split_and_chart_invariants(self) -> None:
        evidence = _surface_evidence()
        config = AtlasConfig(base_level=0, max_level=3, tau_geo=1.0e9, tau_curv=1.0e9)
        atlas = PersistentOctreeAtlas.from_evidence(evidence.positions, evidence.mass, config)
        root_key = (int(atlas.levels[0]), int(atlas.morton_codes[0]))
        activated = atlas.refine(evidence.positions, evidence.mass, split_mask=torch.tensor([True]))
        self.assertEqual(activated.numel(), 8)
        self.assertEqual(root_key, (int(atlas.levels[0]), int(atlas.morton_codes[0])))
        self.assertFalse(bool(atlas.active[0]))
        self.assertTrue(torch.all(atlas.parent[activated] == 0))
        validation = atlas.validate()
        self.assertTrue(validation.valid, validation)
        self.assertGreater(validation.min_chart_immersion_eigenvalue, 1.0e-6)
        restored = PersistentOctreeAtlas.from_checkpoint_payload(atlas.checkpoint_payload())
        for name in atlas._PERSISTENT_TENSORS:
            torch.testing.assert_close(getattr(restored, name), getattr(atlas, name))
        torch.testing.assert_close(restored.edge_index, atlas.edge_index)

    def test_chart_jacobian_matches_central_difference(self) -> None:
        evidence = _surface_evidence()
        atlas = PersistentOctreeAtlas.from_evidence(
            evidence.positions, evidence.mass, AtlasConfig(base_level=1, max_level=2)
        )
        node = int(atlas.active_indices[0])
        xi = torch.tensor([0.03, -0.02], dtype=torch.float64)
        analytic = atlas.chart_jacobian(node, xi)
        step = 1.0e-6
        columns = []
        for dimension in range(2):
            offset = torch.zeros(2, dtype=xi.dtype)
            offset[dimension] = step
            columns.append(
                (atlas.evaluate_chart(node, xi + offset) - atlas.evaluate_chart(node, xi - offset)) / (2 * step)
            )
        numerical = torch.stack(columns, dim=-1)
        torch.testing.assert_close(analytic, numerical, atol=2.0e-9, rtol=2.0e-9)

    def test_partition_of_unity_metric_is_spd_and_se3_covariant(self) -> None:
        evidence = _surface_evidence()
        atlas = PersistentOctreeAtlas.from_evidence(
            evidence.positions,
            evidence.mass,
            AtlasConfig(base_level=1, max_level=2),
        )
        active = atlas.active_indices
        diagonal = torch.stack(
            (
                torch.linspace(1.0, 2.0, active.numel(), dtype=torch.float64),
                torch.linspace(2.0, 3.0, active.numel(), dtype=torch.float64),
                torch.linspace(3.0, 4.0, active.numel(), dtype=torch.float64),
            ),
            dim=-1,
        )
        metric = torch.diag_embed(diagonal)
        query = evidence.positions[:7]
        interpolated = atlas.partition_of_unity_metric(
            query, metric, node_index=active
        )
        self.assertTrue(torch.all(torch.linalg.eigvalsh(interpolated) > 0))

        angle = torch.tensor(0.37, dtype=torch.float64)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotation = torch.stack(
            (
                torch.stack((cosine, -sine, angle.new_zeros(()))),
                torch.stack((sine, cosine, angle.new_zeros(()))),
                torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            )
        )
        translation = torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)
        transformed = copy.deepcopy(atlas)
        transformed.chart_centers = atlas.chart_centers @ rotation.T + translation
        transformed.cell_centers = atlas.cell_centers @ rotation.T + translation
        transformed.chart_frames = rotation @ atlas.chart_frames
        transformed_metric = rotation @ metric @ rotation.T
        transformed_value = transformed.partition_of_unity_metric(
            query @ rotation.T + translation,
            transformed_metric,
            node_index=active,
        )
        torch.testing.assert_close(
            transformed_value,
            rotation @ interpolated @ rotation.T,
            atol=2.0e-10,
            rtol=2.0e-10,
        )

    def test_trellis_prior_seeds_hidden_support_without_becoming_evidence(self) -> None:
        dtype = torch.float64
        prior = TrellisStructurePrior(
            coordinates=[
                torch.tensor([[6, 6, 6], [7, 6, 6]], dtype=torch.int64),
                torch.tensor([[6, 6, 6], [6, 7, 6]], dtype=torch.int64),
            ],
            resolution=8,
        )
        adapter = TrellisPriorAdapter(pipeline=None, samples=2, strength=0.4)
        measure = adapter.support_measure(
            prior,
            torch.full((3,), -0.5, dtype=dtype),
            torch.full((3,), 0.5, dtype=dtype),
        )
        self.assertEqual(measure.positions.shape[0], 3)
        self.assertEqual(measure.sample_count, 2)
        torch.testing.assert_close(
            measure.probability.sort().values,
            torch.tensor([0.5, 0.5, 5.0 / 6.0], dtype=dtype),
        )
        self.assertTrue(torch.all(measure.mass_variance >= 0))
        consistent = torch.argmax(measure.vote_count)
        singleton = torch.argmin(measure.vote_count)
        self.assertLess(
            float(measure.mass_variance[consistent]),
            float(measure.mass_variance[singleton]),
        )

        evidence = _surface_evidence(dtype)
        evidence.positions = evidence.positions * 0.2 - evidence.positions.new_tensor(
            [0.25, 0.25, 0.0]
        )
        config = AtlasConfig(base_level=2, max_level=3, tau_geo=1.0e9, tau_curv=1.0e9)
        atlas = PersistentOctreeAtlas.from_evidence(
            evidence.positions,
            evidence.mass,
            config,
            root_bounds=(
                torch.full((3,), -0.5, dtype=dtype),
                torch.full((3,), 0.5, dtype=dtype),
            ),
            prior_positions=measure.positions,
            prior_mass=measure.mass,
            prior_mass_variance=measure.mass_variance,
        )
        active = atlas.active_indices
        hidden = active[(atlas.evidence_mass[active] == 0) & (atlas.prior_mass[active] > 0)]
        self.assertGreater(hidden.numel(), 0)
        self.assertTrue(torch.all(atlas.point_count[hidden] == 0))
        self.assertTrue(torch.all(atlas.prior_point_count[hidden] > 0))

        operator = ManifoldMappingOperator(
            evidence.features.shape[-1],
            ManifoldMappingConfig(support_radius_factor=5.0),
        ).double()
        result = operator(atlas, evidence)
        self.assertEqual(result.graph.target_count, evidence.positions.shape[0])
        self.assertEqual(int(result.graph.target.max()), evidence.positions.shape[0] - 1)
        self.assertTrue(
            torch.all(
                (result.observation_reliability >= 0)
                & (result.observation_reliability < 1)
            )
        )
        observed_row = atlas.evidence_mass[result.graph.atlas_node_index] > 0
        self.assertTrue(
            torch.all(result.observation_reliability[observed_row] > 0)
        )
        source, target = result.graph.source, result.graph.target
        raw_center = torch.zeros_like(result.transported_centers)
        raw_center.index_add_(0, source, result.plan[:, None] * evidence.positions[target])
        raw_center = raw_center / result.transported_mass.clamp_min(
            torch.finfo(dtype).eps
        )[:, None]
        chart_center = atlas.chart_centers[result.graph.atlas_node_index]
        self.assertTrue(
            torch.all(
                torch.linalg.vector_norm(result.transported_centers - chart_center, dim=-1)
                <= torch.linalg.vector_norm(raw_center - chart_center, dim=-1) + 1.0e-12
            )
        )
        prior_probability = adapter.node_probability(atlas)
        shape_probability = adapter.node_shape_probability(atlas, measure.sample_count)
        zero_vote = atlas.prior_point_count[active] == 0
        if bool(torch.any(zero_vote)):
            torch.testing.assert_close(
                shape_probability[zero_vote],
                torch.full_like(shape_probability[zero_vote], 1.0 / 6.0),
            )
        observed = torch.linspace(0.1, 0.9, atlas.num_active, dtype=dtype)
        combined = adapter.combine_observed_probability(observed, prior_probability)
        self.assertTrue(torch.all(combined >= observed))
        restored = PersistentOctreeAtlas.from_checkpoint_payload(atlas.checkpoint_payload())
        torch.testing.assert_close(restored.prior_mass, atlas.prior_mass)
        torch.testing.assert_close(
            restored.prior_mass_variance, atlas.prior_mass_variance
        )
        torch.testing.assert_close(restored.prior_point_count, atlas.prior_point_count)
        legacy = copy.deepcopy(atlas.checkpoint_payload())
        legacy["format_version"] = 2
        del legacy["tensors"]["prior_mass"]
        del legacy["tensors"]["prior_point_count"]
        del legacy["tensors"]["prior_mass_variance"]
        migrated = PersistentOctreeAtlas.from_checkpoint_payload(legacy)
        self.assertTrue(torch.all(migrated.prior_mass == 0))
        self.assertTrue(torch.all(migrated.prior_point_count == 0))
        truncated = copy.deepcopy(atlas.checkpoint_payload())
        truncated["tensors"]["prior_mass"] = truncated["tensors"]["prior_mass"][:-1]
        with self.assertRaisesRegex(ValueError, "inconsistent persistent tensor lengths"):
            PersistentOctreeAtlas.from_checkpoint_payload(truncated)

    def test_transport_cost_and_uncertainty_reach_attention_adjacency(self) -> None:
        evidence = _surface_evidence()
        atlas = PersistentOctreeAtlas.from_evidence(
            evidence.positions,
            evidence.mass,
            AtlasConfig(base_level=1, max_level=2),
        )
        operator = ManifoldMappingOperator(
            evidence.features.shape[-1],
            ManifoldMappingConfig(support_radius_factor=4.0),
        ).double()
        mapping = operator(atlas, evidence)
        edge, active = active_adjacency(atlas)
        cost, uncertainty = GraftGS._attention_edge_evidence(atlas, mapping)
        self.assertTrue(torch.equal(active, mapping.graph.atlas_node_index))
        self.assertEqual(cost.shape, (edge.shape[1],))
        self.assertEqual(uncertainty.shape, (edge.shape[1],))
        self.assertTrue(torch.all(torch.isfinite(cost)))
        self.assertTrue(torch.all(cost >= 0))
        self.assertTrue(torch.all((uncertainty >= 0) & (uncertainty <= 1)))
        # This is a production gradient path, not detached logging metadata.
        (cost.mean() + uncertainty.mean()).backward()
        self.assertIsNotNone(operator.cost_model.raw_lambda_x.grad)
        self.assertGreater(float(operator.cost_model.raw_lambda_x.grad.abs()), 0.0)
        self.assertIsNotNone(operator.cost_model.raw_lambda_visibility.grad)
        self.assertGreater(
            float(operator.cost_model.raw_lambda_visibility.grad.abs()), 0.0
        )


class ImplicitSinkhornTest(unittest.TestCase):
    def test_sparse_all_edges_matches_dense_fixed_point_and_has_gradients(self) -> None:
        dtype = torch.float64
        n, m = 4, 5
        source, target = torch.meshgrid(torch.arange(n), torch.arange(m), indexing="ij")
        edge = torch.stack((source.reshape(-1), target.reshape(-1)))
        coordinate_source = torch.linspace(-1.0, 1.0, n, dtype=dtype)
        coordinate_target = torch.linspace(-0.8, 0.9, m, dtype=dtype)
        cost_matrix = (coordinate_source[:, None] - coordinate_target[None]).square()
        cost = cost_matrix.reshape(-1).clone().requires_grad_(True)
        a = torch.tensor([0.15, 0.35, 0.25, 0.25], dtype=dtype, requires_grad=True)
        b = torch.tensor([0.1, 0.2, 0.25, 0.3, 0.15], dtype=dtype, requires_grad=True)
        config = ImplicitSinkhornConfig(
            epsilon=0.08,
            tau_source=0.7,
            tau_target=0.6,
            max_iterations=1000,
            tolerance=1.0e-12,
            backward_max_iterations=1000,
            backward_tolerance=1.0e-12,
        )
        plan, diagnostics = ImplicitUnbalancedSinkhorn(config)(cost, a, b, edge)

        rho_a = config.tau_source / (config.tau_source + config.epsilon)
        rho_b = config.tau_target / (config.tau_target + config.epsilon)
        kernel = a[:, None] * b[None] * torch.exp(-cost_matrix / config.epsilon)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(1000):
            u_new = (a / (kernel @ v).clamp_min(torch.finfo(dtype).tiny)).pow(rho_a)
            v_new = (b / (kernel.transpose(0, 1) @ u_new).clamp_min(torch.finfo(dtype).tiny)).pow(rho_b)
            if max((u_new - u).abs().max(), (v_new - v).abs().max()) < 1.0e-12:
                u, v = u_new, v_new
                break
            u, v = u_new, v_new
        dense = u[:, None] * kernel * v[None]
        torch.testing.assert_close(plan.reshape(n, m), dense, atol=2.0e-10, rtol=2.0e-9)
        self.assertLess(diagnostics.fixed_point_residual, 1.0e-10)
        row = plan.reshape(n, m).sum(dim=1)
        column = plan.reshape(n, m).sum(dim=0)
        stationarity = (
            cost_matrix
            + config.epsilon * torch.log(plan.reshape(n, m) / (a[:, None] * b[None]))
            + config.tau_source * torch.log(row[:, None] / a[:, None])
            + config.tau_target * torch.log(column[None] / b[None])
        )
        torch.testing.assert_close(stationarity, torch.zeros_like(stationarity), atol=2.0e-9, rtol=0.0)
        loss = torch.sum(plan * torch.linspace(0.2, 1.1, plan.numel(), dtype=dtype))
        gradients = torch.autograd.grad(loss, (cost, a, b))
        for gradient in gradients:
            self.assertTrue(torch.all(torch.isfinite(gradient)))
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_implicit_backward_matches_finite_difference(self) -> None:
        dtype = torch.float64
        edge = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
        cost = torch.tensor([0.2, 0.5, 0.9, 0.7, 0.1, 0.4], dtype=dtype, requires_grad=True)
        source_mass = torch.tensor([0.55, 0.45], dtype=dtype, requires_grad=True)
        target_mass = torch.tensor([0.2, 0.5, 0.3], dtype=dtype, requires_grad=True)
        solver = ImplicitUnbalancedSinkhorn(
            ImplicitSinkhornConfig(
                epsilon=0.12,
                tau_source=0.8,
                tau_target=0.7,
                max_iterations=1500,
                tolerance=1.0e-13,
                backward_max_iterations=1500,
                backward_tolerance=1.0e-13,
            )
        )

        def function(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return solver(c, a, b, edge)[0]

        self.assertTrue(
            torch.autograd.gradcheck(
                function,
                (cost, source_mass, target_mass),
                eps=1.0e-6,
                atol=3.0e-5,
                rtol=3.0e-4,
            )
        )


class GaugeCovarianceTest(unittest.TestCase):
    def test_global_se3_equivariance(self) -> None:
        evidence = _surface_evidence()
        atlas = PersistentOctreeAtlas.from_evidence(
            evidence.positions, evidence.mass, AtlasConfig(base_level=1, max_level=2)
        )
        config = ManifoldMappingConfig(
            sinkhorn=ImplicitSinkhornConfig(max_iterations=800, tolerance=1.0e-11),
            support_radius_factor=4.0,
        )
        operator = ManifoldMappingOperator(feature_dim=evidence.features.shape[-1], config=config).double()
        reference = operator(atlas, evidence)

        angle = torch.tensor(0.63, dtype=torch.float64)
        c, s = torch.cos(angle), torch.sin(angle)
        rotation = torch.stack(
            (
                torch.stack((c, -s, torch.zeros_like(c))),
                torch.stack((s, c, torch.zeros_like(c))),
                torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            )
        )
        translation = torch.tensor([0.3, -0.2, 0.7], dtype=torch.float64)
        transformed_atlas = copy.deepcopy(atlas)
        transformed_atlas.chart_centers = atlas.chart_centers @ rotation.T + translation
        transformed_atlas.cell_centers = atlas.cell_centers @ rotation.T + translation
        transformed_atlas.chart_frames = rotation @ atlas.chart_frames
        transformed = copy.deepcopy(evidence)
        transformed.positions = evidence.positions @ rotation.T + translation
        transformed.rays = evidence.rays @ rotation.T
        transformed.covariance = rotation @ evidence.covariance @ rotation.T
        result = operator(transformed_atlas, transformed)

        torch.testing.assert_close(reference.cost, result.cost, atol=2.0e-9, rtol=2.0e-9)
        torch.testing.assert_close(reference.plan, result.plan, atol=2.0e-9, rtol=2.0e-9)
        torch.testing.assert_close(reference.latent, result.latent, atol=3.0e-8, rtol=3.0e-8)
        expected_centers = reference.transported_centers @ rotation.T + translation
        torch.testing.assert_close(expected_centers, result.transported_centers, atol=3.0e-9, rtol=3.0e-9)
        expected_metric = rotation @ reference.riemannian_metric @ rotation.T
        torch.testing.assert_close(expected_metric, result.riemannian_metric, atol=3.0e-8, rtol=3.0e-8)


if __name__ == "__main__":
    unittest.main()

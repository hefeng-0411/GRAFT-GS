"""Server-side invariant tests for cameras, equivariance, topology, and flow."""

from __future__ import annotations

import copy
import unittest

import torch

from graft_gs.engine.losses import (
    multiview_reprojection_cycle_loss,
    transport_packed_irrep,
    vggt_depth_normal_field,
)
from graft_gs.equivariant.gsta import (
    GSTAConfig,
    GaugeCovariantSparseTransportAttention,
    IrrepTensor,
    MultiplicityLinear,
    active_adjacency,
    l2_to_matrix,
    matrix_to_l2,
)
from graft_gs.geometry.atlas import AtlasConfig, PersistentOctreeAtlas
from graft_gs.integration.pipeline import GraftGSOutput
from graft_gs.integration.vggt_adapter import VGGTGeometryOutput
from graft_gs.manifold.barrier import BarrierConfig, BarrierProjector, triangle_distance_squared
from graft_gs.manifold.geometry import (
    ManifoldState,
    ManifoldTangent,
    geodesic_interpolate,
    retract,
    spectral_box_spd,
    so3_exp,
    so3_log,
    spd_geodesic,
    spd_parallel_transport,
)
from graft_gs.mapping.manifold_mapping import GeometricEvidenceBuilder
from graft_gs.optimization.quantization import certify_topology_quantization_step
from graft_gs.topology.strata import (
    SimplicialComplex,
    betti_numbers,
    persistence_critical_occupancy_thresholds,
    persistent_homology,
)


def _grid_atlas() -> PersistentOctreeAtlas:
    axis = torch.tensor([-0.35, 0.35], dtype=torch.float64)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    points = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
    atlas = PersistentOctreeAtlas.from_evidence(points, config=AtlasConfig(base_level=0, max_level=2))
    atlas.refine(points, split_mask=torch.tensor([True]))
    return atlas


class CameraConventionTest(unittest.TestCase):
    def test_unprojection_reprojection_identity_camera(self) -> None:
        dtype = torch.float64
        images = torch.linspace(0.1, 0.9, 4 * 4 * 3, dtype=dtype).reshape(1, 1, 3, 4, 4)
        depth = torch.full((1, 1, 4, 4, 1), 2.0, dtype=dtype)
        confidence = torch.full((1, 1, 4, 4), 5.0, dtype=dtype)
        extrinsic = torch.eye(4, dtype=dtype)[:3].reshape(1, 1, 3, 4)
        intrinsic = torch.tensor([[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]], dtype=dtype).reshape(1, 1, 3, 3)
        feature = torch.arange(4 * 8, dtype=dtype).reshape(1, 1, 4, 8)
        evidence = GeometricEvidenceBuilder().double()(images, depth, confidence, extrinsic, intrinsic, feature)[0]
        projected = evidence.positions[:, :2] / evidence.positions[:, 2:] * 4.0 + 2.0
        torch.testing.assert_close(projected, evidence.pixel_uv, atol=1.0e-10, rtol=1.0e-10)
        torch.testing.assert_close(evidence.positions[:, 2], torch.full((4,), 2.0, dtype=dtype))

    def test_vggt_derived_track_cycle_and_plane_normals(self) -> None:
        dtype = torch.float64
        height = width = 17
        depth = torch.full(
            (1, 2, height, width, 1), 2.0, dtype=dtype, requires_grad=True
        )
        extrinsic = torch.eye(4, dtype=dtype)[:3].reshape(1, 1, 3, 4).expand(1, 2, -1, -1).clone()
        intrinsic = torch.tensor(
            [[12.0, 0.0, 8.0], [0.0, 12.0, 8.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        ).reshape(1, 1, 3, 3).expand(1, 2, -1, -1).clone()
        geometry = VGGTGeometryOutput(
            images=torch.zeros(1, 2, 3, height, width, dtype=dtype),
            patch_features=torch.zeros(1, 2, 1, 1024, dtype=dtype),
            extrinsics_world_to_camera=extrinsic,
            intrinsics=intrinsic,
            depth=depth,
            depth_confidence=torch.full((1, 2, height, width, 1), 10.0, dtype=dtype),
            world_points=torch.zeros(1, 2, height, width, 3, dtype=dtype),
            world_points_confidence=torch.ones(1, 2, height, width, dtype=dtype),
        )
        output = GraftGSOutput(geometry, [])
        torch.testing.assert_close(
            multiview_reprojection_cycle_loss(output, sampling_stride=4),
            torch.zeros((), dtype=dtype),
            atol=2.0e-12,
            rtol=0.0,
        )
        normal, valid, confidence = vggt_depth_normal_field(output)
        self.assertTrue(bool(torch.all(valid[:, :, :, :-1, :-1])))
        self.assertFalse(bool(torch.any(valid[:, :, :, -1, :])))
        torch.testing.assert_close(
            normal[:, :, 2], torch.ones_like(normal[:, :, 2]), atol=2.0e-12, rtol=0.0
        )
        self.assertTrue(bool(torch.all(confidence > 0.9)))
        perturbed_depth = depth.clone()
        perturbed_depth[:, 1] = 2.02
        perturbed = GraftGSOutput(
            VGGTGeometryOutput(
                geometry.images,
                geometry.patch_features,
                extrinsic,
                intrinsic,
                perturbed_depth,
                geometry.depth_confidence,
                geometry.world_points,
                geometry.world_points_confidence,
            ),
            [],
        )
        cycle = multiview_reprojection_cycle_loss(perturbed, sampling_stride=4)
        self.assertGreater(float(cycle), 1.0e-5)
        cycle.backward()
        self.assertTrue(bool(torch.isfinite(depth.grad).all()))


class GaugeEquivarianceTest(unittest.TestCase):
    def test_multiplicity_spectral_policy_scale_is_effective(self) -> None:
        dtype = torch.float64
        layer = MultiplicityLinear(3).to(dtype=dtype)
        value = torch.arange(12, dtype=dtype).reshape(4, 3)
        reference = layer(value)
        layer.set_operator_scale(0.25)
        torch.testing.assert_close(layer(value), 0.25 * reference)

    def test_global_se3_and_local_gauge_covariance(self) -> None:
        atlas = _grid_atlas()
        v = atlas.num_active
        config = GSTAConfig(residual_step=0.2)
        layer = GaugeCovariantSparseTransportAttention(config).double()
        scalar = torch.cos(torch.arange(v * 60, dtype=torch.float64).reshape(v, 60) * 0.07)
        vector = torch.sin(torch.arange(v * 16 * 3, dtype=torch.float64).reshape(v, 16, 3) * 0.03)
        tensor = torch.cos(torch.arange(v * 4 * 5, dtype=torch.float64).reshape(v, 4, 5) * 0.05)
        fields = IrrepTensor(scalar, vector, tensor)
        edge, _ = active_adjacency(atlas)
        edge_ot_cost = torch.linspace(0.0, 1.0, edge.shape[1], dtype=torch.float64)
        edge_uncertainty = torch.linspace(1.0, 0.0, edge.shape[1], dtype=torch.float64)
        reference = layer(
            atlas,
            fields,
            edge_ot_cost=edge_ot_cost,
            edge_uncertainty=edge_uncertainty,
        )

        global_rotation = so3_exp(torch.tensor([0.2, -0.4, 0.3], dtype=torch.float64))
        translation = torch.tensor([0.3, 0.5, -0.2], dtype=torch.float64)
        transformed = copy.deepcopy(atlas)
        transformed.chart_centers = atlas.chart_centers @ global_rotation.T + translation
        transformed.cell_centers = atlas.cell_centers @ global_rotation.T + translation
        transformed.chart_frames = global_rotation @ atlas.chart_frames
        global_output = layer(
            transformed,
            fields,
            edge_ot_cost=edge_ot_cost,
            edge_uncertainty=edge_uncertainty,
        )
        torch.testing.assert_close(reference.pack(), global_output.pack(), atol=2.0e-9, rtol=2.0e-9)

        angle = torch.linspace(-0.7, 0.8, v, dtype=torch.float64)
        gauge = torch.zeros(v, 3, 3, dtype=torch.float64)
        gauge[:, 0, 0], gauge[:, 0, 1] = torch.cos(angle), -torch.sin(angle)
        gauge[:, 1, 0], gauge[:, 1, 1] = torch.sin(angle), torch.cos(angle)
        gauge[:, 2, 2] = 1.0
        gauged_atlas = copy.deepcopy(atlas)
        active = atlas.active_indices
        gauged_atlas.chart_frames[active] = atlas.chart_frames[active] @ gauge
        gauged_vector = torch.einsum("vji,vcj->vci", gauge, vector)
        tensor_matrix = l2_to_matrix(tensor)
        gauged_tensor = matrix_to_l2(gauge.transpose(-1, -2)[:, None] @ tensor_matrix @ gauge[:, None])
        gauged_output = layer(
            gauged_atlas,
            IrrepTensor(scalar, gauged_vector, gauged_tensor),
            edge_ot_cost=edge_ot_cost,
            edge_uncertainty=edge_uncertainty,
        )
        expected_vector = torch.einsum("vji,vcj->vci", gauge, reference.vector)
        expected_tensor = matrix_to_l2(
            gauge.transpose(-1, -2)[:, None] @ l2_to_matrix(reference.tensor) @ gauge[:, None]
        )
        torch.testing.assert_close(reference.scalar, gauged_output.scalar, atol=3.0e-8, rtol=3.0e-8)
        torch.testing.assert_close(expected_vector, gauged_output.vector, atol=3.0e-8, rtol=3.0e-8)
        torch.testing.assert_close(expected_tensor, gauged_output.tensor, atol=3.0e-8, rtol=3.0e-8)


class TopologyAndManifoldTest(unittest.TestCase):
    def test_spd_spectral_box_is_bounded_and_repeated_spectrum_safe(self) -> None:
        matrix = torch.stack(
            (
                0.01 * torch.eye(3, dtype=torch.float64),
                2.0 * torch.eye(3, dtype=torch.float64),
                torch.diag(torch.tensor([1.0e-8, 0.02, 3.0], dtype=torch.float64)),
            )
        ).requires_grad_(True)
        projected = spectral_box_spd(matrix, 1.0e-6, 0.25)
        eigenvalue = torch.linalg.eigvalsh(projected)
        self.assertTrue(bool(torch.all(eigenvalue >= 1.0e-6 - 1.0e-12)))
        self.assertTrue(bool(torch.all(eigenvalue <= 0.25 + 1.0e-12)))
        gradient = torch.autograd.grad(projected.square().sum(), matrix)[0]
        self.assertTrue(torch.all(torch.isfinite(gradient)))

    def test_persistence_critical_proposal_thresholds(self) -> None:
        diagrams = {
            0: torch.tensor([[0.1, 0.9]], dtype=torch.float64),
            1: torch.tensor([[0.25, 0.55]], dtype=torch.float64),
            2: torch.empty(0, 2, dtype=torch.float64),
        }
        thresholds = persistence_critical_occupancy_thresholds(
            diagrams, maximum_count=4, minimum_lifetime=0.1
        )
        self.assertEqual(len(thresholds), 4)
        for actual, expected in zip(thresholds, (0.9, 0.1, 0.75, 0.45)):
            self.assertAlmostEqual(actual, expected, places=12)

    def _tetra_complex(self) -> SimplicialComplex:
        return SimplicialComplex(
            torch.arange(4),
            torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
            torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
        )

    def test_exact_sphere_betti_and_persistence(self) -> None:
        complex_ = self._tetra_complex()
        self.assertEqual(betti_numbers(complex_), (1, 0, 1))
        self.assertTrue(complex_.manifold_incidence_valid())
        self.assertTrue(complex_.orientation_consistent())
        inconsistent = SimplicialComplex(
            complex_.atlas_node_index,
            complex_.edges,
            complex_.faces.clone(),
        )
        inconsistent.faces[0, [1, 2]] = inconsistent.faces[0, [2, 1]]
        self.assertFalse(inconsistent.orientation_consistent())
        filtration = torch.tensor([0.0, 0.2, 0.4, 0.6], dtype=torch.float64, requires_grad=True)
        diagram = persistent_homology(complex_, filtration)
        self.assertGreaterEqual(diagram[0].shape[0], 1)
        self.assertGreaterEqual(diagram[2].shape[0], 1)
        persistence_loss = sum(values.sum() for values in diagram.values())
        persistence_loss.backward()
        self.assertIsNotNone(filtration.grad)
        self.assertTrue(torch.all(torch.isfinite(filtration.grad)))
        self.assertGreater(float(filtration.grad.abs().sum()), 0.0)

    def test_so3_spd_geodesics_and_barrier(self) -> None:
        dtype = torch.float64
        rotation_0 = torch.eye(3, dtype=dtype).expand(4, -1, -1).clone()
        omega = torch.tensor([[0.2, -0.1, 0.3]], dtype=dtype).expand(4, -1)
        rotation_1 = rotation_0 @ so3_exp(omega)
        torch.testing.assert_close(so3_log(rotation_0.transpose(-1, -2) @ rotation_1), omega, atol=2.0e-9, rtol=2.0e-9)
        covariance_0 = torch.diag_embed(torch.tensor([[0.1, 0.2, 0.3]], dtype=dtype).expand(4, -1))
        covariance_1 = torch.diag_embed(torch.tensor([[0.3, 0.15, 0.4]], dtype=dtype).expand(4, -1))
        torch.testing.assert_close(spd_geodesic(covariance_0, covariance_1, 0.0), covariance_0)
        torch.testing.assert_close(spd_geodesic(covariance_0, covariance_1, 1.0), covariance_1)
        position = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]], dtype=dtype)
        complex_ = self._tetra_complex()
        state = ManifoldState(
            position,
            rotation_0,
            covariance_0,
            torch.zeros(4, 1, dtype=dtype),
            torch.zeros(4, 48, dtype=dtype),
            torch.zeros(4, 128, dtype=dtype),
            torch.eye(3, dtype=dtype).expand(4, -1, -1).clone(),
            complex_,
        )
        projector = BarrierProjector(state, BarrierConfig(minimum_separation=0.1))
        self.assertTrue(projector.report(state).feasible)
        topology_margin = projector.topology_boundary_margin(state)
        self.assertTrue(bool(torch.isfinite(topology_margin)))
        self.assertGreater(float(topology_margin), 0.0)
        scaled_metric_state = copy.deepcopy(state)
        scaled_metric_state.evidence_metric = 4.0 * state.evidence_metric
        scaled_margin = projector.topology_boundary_margin(scaled_metric_state)
        torch.testing.assert_close(
            scaled_margin, 2.0 * topology_margin, atol=2.0e-10, rtol=2.0e-10
        )
        certificate = certify_topology_quantization_step(
            projector,
            state,
            query_error=1.0e-4,
            temperature=1.0,
            vector_field_lipschitz=0.1,
            step_size=0.1,
        )
        self.assertTrue(bool(certificate.certified))
        self.assertEqual(
            certificate.to_dict()["topology_boundary_margin"],
            float(topology_margin),
        )
        state32 = ManifoldState(
            position.float(),
            rotation_0.float(),
            covariance_0.float(),
            torch.zeros(4, 1, dtype=torch.float32),
            torch.zeros(4, 48, dtype=torch.float32),
            torch.zeros(4, 128, dtype=torch.float32),
            torch.eye(3, dtype=torch.float32).expand(4, -1, -1).clone(),
            complex_,
        )
        projector32 = BarrierProjector(
            state32,
            BarrierConfig(minimum_separation=0.1),
        )
        self.assertTrue(projector32.report(state32).feasible)
        self.assertEqual(
            projector32.topology_boundary_margin(state32).dtype,
            torch.float64,
        )
        tangent = ManifoldTangent(
            position=-0.1 * position,
            rotation_body=torch.zeros(4, 3, dtype=dtype),
            covariance=torch.zeros(4, 3, 3, dtype=dtype),
            opacity_logit=torch.zeros(4, 1, dtype=dtype),
            appearance=torch.zeros(4, 48, dtype=dtype),
            latent=torch.zeros(4, 128, dtype=dtype),
        )
        safe, report = projector.project(state, tangent)
        next_state, accepted = projector.retract_with_backtracking(state, safe, 0.5)
        self.assertTrue(accepted.feasible)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(next_state.covariance) > 0))

    def test_affine_spd_parallel_transport_is_an_isometry(self) -> None:
        dtype = torch.float64
        start = torch.diag(torch.tensor([0.2, 0.7, 1.3], dtype=dtype))[None]
        rotation = so3_exp(torch.tensor([[0.2, -0.3, 0.1]], dtype=dtype))
        end_diagonal = torch.diag(torch.tensor([0.9, 0.3, 1.7], dtype=dtype))[None]
        end = rotation @ end_diagonal @ rotation.transpose(-1, -2)
        tangent = torch.tensor(
            [[[0.2, -0.1, 0.05], [-0.1, 0.3, 0.02], [0.05, 0.02, -0.15]]],
            dtype=dtype,
        )
        transported = spd_parallel_transport(start, end, tangent)
        start_inverse = torch.linalg.inv(start)
        end_inverse = torch.linalg.inv(end)
        start_norm = torch.einsum(
            "vij,vjk,vkl,vli->v", start_inverse, tangent, start_inverse, tangent
        )
        end_norm = torch.einsum(
            "vij,vjk,vkl,vli->v",
            end_inverse,
            transported,
            end_inverse,
            transported,
        )
        torch.testing.assert_close(start_norm, end_norm, atol=2.0e-10, rtol=2.0e-10)

    def test_packed_irrep_transport_round_trip(self) -> None:
        dtype = torch.float64
        packed = torch.linspace(-0.5, 0.7, 3 * 128, dtype=dtype).reshape(3, 128)
        source = so3_exp(
            torch.tensor(
                [[0.1, 0.2, -0.1], [-0.3, 0.2, 0.1], [0.05, -0.2, 0.4]],
                dtype=dtype,
            )
        )
        target = so3_exp(
            torch.tensor(
                [[-0.2, 0.1, 0.3], [0.15, -0.25, 0.2], [-0.1, 0.3, 0.2]],
                dtype=dtype,
            )
        )
        transported = transport_packed_irrep(packed, source, target)
        recovered = transport_packed_irrep(transported, target, source)
        torch.testing.assert_close(recovered, packed, atol=3.0e-10, rtol=3.0e-10)

    def test_collision_broad_phase_and_speed_certificate(self) -> None:
        dtype = torch.float64
        position = torch.tensor(
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.3, 0.0], [0.0, 0.3, 0.0]],
            dtype=dtype,
        )
        complex_ = SimplicialComplex(
            torch.arange(4),
            torch.tensor([[0, 1], [1, 2], [0, 2], [2, 3], [0, 3]]),
            torch.tensor([[0, 1, 2], [0, 2, 3]]),
        )
        state = ManifoldState(
            position=position,
            rotation=torch.eye(3, dtype=dtype).expand(4, -1, -1).clone(),
            covariance=0.1 * torch.eye(3, dtype=dtype).expand(4, -1, -1).clone(),
            opacity_logit=torch.zeros(4, 1, dtype=dtype),
            appearance=torch.zeros(4, 48, dtype=dtype),
            latent=torch.zeros(4, 128, dtype=dtype),
            evidence_metric=torch.eye(3, dtype=dtype).expand(4, -1, -1).clone(),
            complex=complex_,
        )
        config = BarrierConfig(
            minimum_separation=0.42,
            activation_margin=0.02,
            maximum_position_speed=0.1,
        )
        projector = BarrierProjector(state, config)
        self.assertEqual(projector.nonlocal_pairs.shape[0], 1)
        tangent = ManifoldTangent(
            position=torch.tensor(
                [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5], [1.0, 1.0, 0.0]],
                dtype=dtype,
            ),
            rotation_body=torch.zeros(4, 3, dtype=dtype),
            covariance=torch.zeros(4, 3, 3, dtype=dtype),
            opacity_logit=torch.zeros(4, 1, dtype=dtype),
            appearance=torch.zeros(4, 48, dtype=dtype),
            latent=torch.zeros(4, 128, dtype=dtype),
        )
        limited = projector._limit_position_speed(tangent)
        safe, projection_report = projector.project(state, tangent)
        self.assertLessEqual(
            float(torch.linalg.vector_norm(safe.position, dim=-1).max()),
            config.maximum_position_speed + 1.0e-12,
        )
        # The speed certificate must use a global positive rescaling.  A
        # vertex-wise clip could invalidate coupled CBF inequalities after the
        # projection even though each individual speed remains bounded.
        input_speed = torch.linalg.vector_norm(tangent.position, dim=-1)
        output_speed = torch.linalg.vector_norm(limited.position, dim=-1)
        ratio = output_speed / input_speed
        torch.testing.assert_close(
            ratio,
            torch.full_like(ratio, ratio[0]),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        self.assertGreaterEqual(
            projection_report.minimum_linearized_margin,
            -10.0 * config.dual_tolerance,
        )

    def test_metric_minimal_restoration_enters_strict_feasible_set(self) -> None:
        dtype = torch.float64
        separation = 8.6e-5
        position = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.0, separation],
                [0.2, 0.0, separation],
                [0.0, 0.2, separation],
            ],
            dtype=dtype,
            requires_grad=True,
        )
        complex_ = SimplicialComplex(
            torch.arange(6),
            torch.tensor(
                [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]],
                dtype=torch.int64,
            ),
            torch.tensor([[0, 1, 2], [3, 5, 4]], dtype=torch.int64),
        )
        state = ManifoldState(
            position=position,
            rotation=torch.eye(3, dtype=dtype).expand(6, -1, -1).clone(),
            covariance=0.1 * torch.eye(3, dtype=dtype).expand(6, -1, -1).clone(),
            opacity_logit=torch.zeros(6, 1, dtype=dtype),
            appearance=torch.zeros(6, 48, dtype=dtype),
            latent=torch.zeros(6, 128, dtype=dtype),
            evidence_metric=torch.eye(3, dtype=dtype).expand(6, -1, -1).clone(),
            complex=complex_,
        )
        config = BarrierConfig(
            minimum_separation=1.0e-4,
            maximum_position_speed=1.0e-3,
        )
        projector = BarrierProjector(state, config)
        initial = projector.report(state)
        self.assertFalse(initial.feasible)
        self.assertLess(initial.minimum_separation_margin, 0.0)
        restored, report = projector.restore_feasible_embedding(state)
        self.assertTrue(report.feasible)
        self.assertGreater(report.minimum_separation_margin, 0.0)
        self.assertGreater(report.restoration_iterations, 0)
        self.assertLessEqual(
            report.restoration_maximum_displacement,
            config.maximum_position_speed + 1.0e-12,
        )
        restored.position.square().sum().backward()
        self.assertIsNotNone(position.grad)
        self.assertTrue(torch.all(torch.isfinite(position.grad)))

    def test_triangle_collision_distance_detects_face_crossing(self) -> None:
        dtype = torch.float64
        left = torch.tensor(
            [[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]], dtype=dtype
        )
        crossing = torch.tensor(
            [[[0.0, -0.5, -1.0], [0.0, -0.5, 1.0], [0.0, 0.5, 0.0]]], dtype=dtype
        )
        separated = crossing + torch.tensor([4.0, 0.0, 0.0], dtype=dtype)
        torch.testing.assert_close(
            triangle_distance_squared(left, crossing), torch.zeros(1, dtype=dtype), atol=1.0e-12, rtol=0.0
        )
        self.assertGreater(float(triangle_distance_squared(left, separated)), 1.0)


if __name__ == "__main__":
    unittest.main()

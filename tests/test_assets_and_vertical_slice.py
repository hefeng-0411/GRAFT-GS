"""Analytical asset, serialization, rendering, and gradient verification."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import importlib.util
import unittest

import torch

from graft_gs.geometry.atlas import AtlasConfig, PersistentOctreeAtlas
from graft_gs.integration.pipeline import GraftGS
from graft_gs.engine.losses import (
    atlas_multilevel_consistency,
    atlas_overlap_consistency,
    conservative_tile_opacity_bound,
    generalized_transport_kl,
    minibatch_ot_flow_coupling,
    multiscale_perceptual_loss,
    structural_similarity_loss,
    surface_pseudo_relational_distillation,
)
from graft_gs.manifold.geometry import so3_exp
from graft_gs.mapping.manifold_mapping import (
    EvidenceParticles,
    ImplicitSinkhornConfig,
    ManifoldMappingConfig,
    ManifoldMappingOperator,
)
from graft_gs.readout.assets import (
    AnalyticalReadoutConfig,
    AnalyticalSurfaceReadout,
    write_gaussian_ply,
    write_mesh_glb,
)
from graft_gs.readout.renderer import (
    CameraBatch,
    CudaGaussianRenderer,
    RasterizationContract,
    ReferenceGaussianRenderer,
    _mip_filter_covariance,
)
from graft_gs.topology.strata import SimplicialComplex, TopologyCandidate, TopologySelection, betti_numbers


def _fixture() -> tuple[PersistentOctreeAtlas, object, TopologySelection]:
    dtype = torch.float64
    axis = torch.tensor([-0.4, 0.4], dtype=dtype)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    positions = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
    count = positions.shape[0]
    features = torch.stack([torch.sin((i + 1) * positions[:, i % 3]) for i in range(12)], dim=-1)
    positions.requires_grad_(True)
    features.requires_grad_(True)
    rays = torch.nn.functional.normalize(positions + torch.tensor([0.0, 0.0, 2.0], dtype=dtype), dim=-1)
    covariance = torch.eye(3, dtype=dtype).expand(count, -1, -1).clone() * 2.0e-3
    evidence = EvidenceParticles(
        positions=positions,
        rays=rays,
        features=features,
        covariance=covariance,
        confidence=torch.full((count,), 0.9, dtype=dtype),
        mass=torch.full((count,), 0.02, dtype=dtype),
        view_index=torch.arange(count).remainder(4),
        pixel_uv=torch.stack((torch.arange(count, dtype=dtype), torch.zeros(count, dtype=dtype)), dim=-1),
        extrinsics_world_to_camera=torch.eye(4, dtype=dtype)[:3].expand(4, -1, -1).clone(),
        intrinsics=torch.eye(3, dtype=dtype).expand(4, -1, -1).clone(),
        depth_variance=torch.full((count,), 2.0e-3, dtype=dtype),
        colors=torch.sigmoid(features[:, :3]),
    )
    atlas = PersistentOctreeAtlas.from_evidence(positions, evidence.mass, AtlasConfig(base_level=0, max_level=2))
    atlas.refine(positions, evidence.mass, torch.tensor([True]))
    operator = ManifoldMappingOperator(
        12,
        ManifoldMappingConfig(
            sinkhorn=ImplicitSinkhornConfig(max_iterations=800, tolerance=1.0e-11),
            support_radius_factor=4.0,
        ),
    ).double()
    mapping = operator(atlas, evidence)
    nodes = atlas.active_indices[torch.tensor([0, 1, 2, 4])]
    complex_ = SimplicialComplex(
        nodes,
        torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
        torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )
    zero = positions.new_zeros(())
    candidate = TopologyCandidate("tetra", complex_, betti_numbers(complex_), {0: positions.new_empty(0, 2), 1: positions.new_empty(0, 2), 2: positions.new_empty(0, 2)}, zero, zero, zero, zero, zero)
    selection = TopologySelection([candidate], positions.new_ones(1), 0)
    return atlas, mapping, selection


class AnalyticalAssetTest(unittest.TestCase):
    def test_camera_batch_rejects_non_opencv_or_non_so3_frames(self) -> None:
        extrinsic = torch.eye(4, dtype=torch.float64)[:3][None]
        intrinsic = torch.tensor(
            [[[12.0, 0.0, 8.0], [0.0, 12.0, 8.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float64,
        )
        CameraBatch(extrinsic, intrinsic, 16, 16)
        invalid_intrinsic = intrinsic.clone()
        invalid_intrinsic[:, 2, 0] = 0.1
        with self.assertRaises(ValueError):
            CameraBatch(extrinsic, invalid_intrinsic, 16, 16)
        reflection = extrinsic.clone()
        reflection[:, 0, 0] = -1.0
        with self.assertRaises(ValueError):
            CameraBatch(reflection, intrinsic, 16, 16)

    def test_cuda_projection_preserves_opencv_integer_pixel_centers(self) -> None:
        intrinsic = torch.tensor(
            [[612.0, 0.0, 251.25], [0.0, 598.0, 260.75], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        height, width = 518, 518
        point = torch.tensor([0.17, -0.11, 2.3, 1.0], dtype=torch.float64)
        projection = CudaGaussianRenderer._projection(
            intrinsic,
            height,
            width,
            0.01,
            100.0,
        )
        homogeneous = projection @ point
        ndc = homogeneous[:2] / homogeneous[3]
        cuda_pixel = torch.stack(
            (
                ((ndc[0] + 1.0) * width - 1.0) * 0.5,
                ((ndc[1] + 1.0) * height - 1.0) * 0.5,
            )
        )
        opencv_pixel = torch.stack(
            (
                intrinsic[0, 0] * point[0] / point[2] + intrinsic[0, 2],
                intrinsic[1, 1] * point[1] / point[2] + intrinsic[1, 2],
            )
        )
        torch.testing.assert_close(cuda_pixel, opencv_pixel, atol=1.0e-12, rtol=1.0e-12)

    def test_mip_filter_preserves_integrated_measure_and_gradients(self) -> None:
        covariance = torch.tensor(
            [[[0.4, 0.07], [0.07, 0.9]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        contract = RasterizationContract(kernel_size=0.1)
        filtered, peak = _mip_filter_covariance(covariance, contract)
        expected_filtered = covariance.detach() + 0.1 * torch.eye(2, dtype=torch.float64)
        expected_peak = torch.sqrt(
            torch.linalg.det(covariance.detach()).clamp_min(1.0e-6)
            / (torch.linalg.det(expected_filtered).clamp_min(1.0e-6) + 1.0e-6)
            + 1.0e-6
        )
        torch.testing.assert_close(filtered, expected_filtered)
        torch.testing.assert_close(peak, expected_peak)
        (filtered.square().sum() + peak.sum()).backward()
        self.assertIsNotNone(covariance.grad)
        self.assertTrue(torch.all(torch.isfinite(covariance.grad)))
        self.assertGreater(float(covariance.grad.abs().sum()), 0.0)

    def test_flow_minibatch_ot_couples_only_compatible_strata(self) -> None:
        atlas, mapping, selection = _fixture()
        base = GraftGS._state_from_mapping(atlas, mapping, selection)
        second = copy.deepcopy(base)
        second.position = base.position + 10.0
        target_far_from_first = copy.deepcopy(base)
        target_far_from_first.position = base.position + 9.0
        target_near_first = copy.deepcopy(base)
        target_near_first.position = base.position + 1.0
        scenes = [
            type("Scene", (), {"initial_state": base})(),
            type("Scene", (), {"initial_state": second})(),
        ]
        coupled = minibatch_ot_flow_coupling(
            scenes,
            [target_far_from_first, target_near_first],
        )
        self.assertIs(coupled[0], target_near_first)
        self.assertIs(coupled[1], target_far_from_first)

    def test_structural_and_fixed_perceptual_losses_have_real_failure_modes(self) -> None:
        target = torch.linspace(0.0, 1.0, 2 * 3 * 16 * 16, dtype=torch.float64).reshape(
            1, 2, 3, 16, 16
        )
        mask = torch.ones(1, 2, 1, 16, 16, dtype=torch.float64)
        torch.testing.assert_close(
            structural_similarity_loss(target, target, mask),
            torch.zeros((), dtype=torch.float64),
            atol=1.0e-12,
            rtol=0.0,
        )
        torch.testing.assert_close(
            multiscale_perceptual_loss(target, target, mask),
            torch.zeros((), dtype=torch.float64),
            atol=1.0e-12,
            rtol=0.0,
        )
        shifted = torch.roll(target, shifts=2, dims=-1)
        self.assertGreater(
            float(structural_similarity_loss(shifted, target, mask)), 0.01
        )
        self.assertGreater(
            float(multiscale_perceptual_loss(shifted, target, mask)), 1.0e-4
        )

    def test_unbalanced_transport_distillation_is_a_generalized_kl(self) -> None:
        teacher = torch.tensor([0.1, 0.25, 0.4], dtype=torch.float64)
        identical = teacher.clone().requires_grad_(True)
        torch.testing.assert_close(
            generalized_transport_kl(identical, teacher),
            torch.zeros((), dtype=torch.float64),
            atol=1.0e-14,
            rtol=0.0,
        )
        student = (0.5 * teacher).requires_grad_(True)
        loss = generalized_transport_kl(student, teacher)
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0.0)

    def test_gaussian_count_uses_curved_surface_area(self) -> None:
        atlas, mapping, selection = _fixture()
        state = GraftGS._state_from_mapping(atlas, mapping, selection)
        flat = copy.deepcopy(atlas)
        flat.curvature.zero_()
        curved = copy.deepcopy(flat)
        node = int(state.complex.atlas_node_index[0])
        curved.curvature[node] = torch.diag(
            torch.tensor([20.0, 20.0], dtype=torch.float64)
        )
        readout = AnalyticalSurfaceReadout(
            AnalyticalReadoutConfig(
                target_surface_area_per_gaussian=0.05,
                maximum_samples_per_chart=256,
            )
        ).double()
        flat_gaussians, _ = readout(flat, state, mapping)
        curved_gaussians, _ = readout(curved, state, mapping)
        flat_count = int(torch.sum(flat_gaussians.chart_index == 0))
        curved_count = int(torch.sum(curved_gaussians.chart_index == 0))
        self.assertGreater(curved_count, flat_count)

    def test_atlas_overlap_objective_is_se3_and_gauge_invariant(self) -> None:
        atlas, mapping, _ = _fixture()
        scene = type("Scene", (), {"atlas": atlas, "mapping": mapping})()
        reference = torch.cat(
            (
                torch.stack(atlas_overlap_consistency(scene)),
                atlas_multilevel_consistency(scene).reshape(1),
            )
        )

        global_rotation = so3_exp(
            torch.tensor([0.2, -0.3, 0.1], dtype=torch.float64)
        )
        translation = torch.tensor([0.4, -0.1, 0.2], dtype=torch.float64)
        transformed = copy.deepcopy(atlas)
        transformed.chart_centers = atlas.chart_centers @ global_rotation.T + translation
        transformed.cell_centers = atlas.cell_centers @ global_rotation.T + translation
        transformed.chart_frames = global_rotation @ atlas.chart_frames
        global_scene = type(
            "Scene", (), {"atlas": transformed, "mapping": mapping}
        )()
        torch.testing.assert_close(
            torch.cat(
                (
                    torch.stack(atlas_overlap_consistency(global_scene)),
                    atlas_multilevel_consistency(global_scene).reshape(1),
                )
            ),
            reference,
            atol=2.0e-9,
            rtol=2.0e-9,
        )

        gauged = copy.deepcopy(atlas)
        angle = torch.linspace(
            -0.5, 0.6, atlas.num_nodes, dtype=torch.float64
        )
        gauge = torch.zeros(atlas.num_nodes, 3, 3, dtype=torch.float64)
        gauge[:, 0, 0], gauge[:, 0, 1] = torch.cos(angle), -torch.sin(angle)
        gauge[:, 1, 0], gauge[:, 1, 1] = torch.sin(angle), torch.cos(angle)
        gauge[:, 2, 2] = 1.0
        gauged.chart_frames = atlas.chart_frames @ gauge
        tangent_gauge = gauge[:, :2, :2]
        gauged.curvature = (
            tangent_gauge.transpose(-1, -2)
            @ atlas.curvature
            @ tangent_gauge
        )
        gauge_scene = type("Scene", (), {"atlas": gauged, "mapping": mapping})()
        torch.testing.assert_close(
            torch.cat(
                (
                    torch.stack(atlas_overlap_consistency(gauge_scene)),
                    atlas_multilevel_consistency(gauge_scene).reshape(1),
                )
            ),
            reference,
            atol=3.0e-8,
            rtol=3.0e-8,
        )

    def test_relational_pseudo_labels_preserve_gauge_invariant_channel_geometry(self) -> None:
        atlas, mapping, selection = _fixture()
        state = GraftGS._state_from_mapping(atlas, mapping, selection)
        scene = type("Scene", (), {"atlas": atlas, "mapping": mapping})()
        surface = atlas.chart_centers[atlas.active_indices].detach()
        index = torch.arange(surface.shape[0], dtype=surface.dtype)
        dino = torch.stack(
            [torch.sin((frequency + 1) * index) for frequency in range(32)], dim=-1
        )
        trellis = torch.stack(
            [torch.cos((frequency + 0.5) * index) for frequency in range(8)], dim=-1
        )
        mapping.latent.retain_grad()
        surface_indices = torch.stack(
            (
                torch.arange(surface.shape[0]),
                torch.zeros(surface.shape[0], dtype=torch.int64),
                torch.zeros(surface.shape[0], dtype=torch.int64),
            ),
            dim=-1,
        )
        dino_loss, trellis_loss = surface_pseudo_relational_distillation(
            scene,
            {
                "surface_voxel_centers": surface,
                "surface_voxel_indices": surface_indices,
                "trellis_patchtokens": dino,
                "trellis_feature_indices": surface_indices.clone(),
                "dino_pseudo_supervision_mask": torch.tensor(True),
                "dino_pseudo_confidence": torch.tensor(0.5, dtype=surface.dtype),
                "dino_pseudo_provenance": "pretrained_dinov2_surface_feature",
                "trellis_latent_features": trellis,
                "trellis_latent_coords": surface_indices.clone(),
                "trellis_latent_pseudo_supervision_mask": torch.tensor(True),
                "trellis_latent_pseudo_confidence": torch.tensor(
                    0.5, dtype=surface.dtype
                ),
                "trellis_latent_pseudo_provenance": (
                    "pretrained_trellis_structured_latent_encoder"
                ),
            },
        )
        self.assertTrue(torch.isfinite(dino_loss + trellis_loss))
        self.assertGreater(float(dino_loss + trellis_loss), 0.0)
        (dino_loss + trellis_loss).backward()
        self.assertIsNotNone(mapping.latent.grad)
        self.assertGreater(float(mapping.latent.grad.abs().sum()), 0.0)

    def test_prior_occupancy_survives_topology_to_asset_state_boundary(self) -> None:
        atlas, mapping, selection = _fixture()
        occupancy = torch.linspace(
            0.2,
            0.8,
            mapping.transported_mass.shape[0],
            dtype=mapping.transported_mass.dtype,
        )
        state = GraftGS._state_from_mapping(
            atlas,
            mapping,
            selection,
            occupancy_probability=occupancy,
        )
        lookup = {
            int(node): index
            for index, node in enumerate(mapping.graph.atlas_node_index.tolist())
        }
        row = torch.tensor(
            [lookup[int(node)] for node in state.complex.atlas_node_index.tolist()]
        )
        torch.testing.assert_close(
            torch.sigmoid(state.opacity_logit[:, 0]),
            occupancy[row],
        )

    def test_nonfloating_spd_and_deterministic_reload(self) -> None:
        atlas, mapping, selection = _fixture()
        state = GraftGS._state_from_mapping(atlas, mapping, selection)
        gaussians, mesh = AnalyticalSurfaceReadout().double()(atlas, state, mapping)
        gaussians.validate()
        node = state.complex.atlas_node_index[gaussians.chart_index]
        curvature = atlas.curvature[node]
        xi = gaussians.chart_coordinates
        height = 0.5 * torch.einsum("gi,gij,gj->g", xi, curvature, xi)
        local = torch.cat((xi, height[:, None]), dim=-1)
        reconstructed = state.position[gaussians.chart_index] + torch.einsum(
            "gij,gj->gi", state.rotation[gaussians.chart_index], local
        )
        torch.testing.assert_close(gaussians.means, reconstructed, atol=1.0e-10, rtol=1.0e-10)
        with tempfile.TemporaryDirectory() as directory:
            first_ply, second_ply = Path(directory) / "a.ply", Path(directory) / "b.ply"
            first_glb, second_glb = Path(directory) / "a.glb", Path(directory) / "b.glb"
            write_gaussian_ply(first_ply, gaussians)
            write_gaussian_ply(second_ply, gaussians)
            write_mesh_glb(first_glb, mesh)
            write_mesh_glb(second_glb, mesh)
            self.assertEqual(first_ply.read_bytes(), second_ply.read_bytes())
            self.assertEqual(first_glb.read_bytes(), second_glb.read_bytes())
            from plyfile import PlyData
            from pygltflib import GLTF2

            loaded_ply = PlyData.read(first_ply)
            self.assertEqual(loaded_ply["vertex"].count, gaussians.means.shape[0])
            loaded_glb = GLTF2().load(str(first_glb))
            self.assertEqual(len(loaded_glb.meshes), 1)
            self.assertEqual(len(loaded_glb.accessors), 4)
            self.assertEqual(len(loaded_glb.materials), 1)
            self.assertEqual(loaded_glb.meshes[0].primitives[0].material, 0)

    def test_renderer_backward_reaches_surface_centers(self) -> None:
        atlas, mapping, selection = _fixture()
        state = GraftGS._state_from_mapping(atlas, mapping, selection)
        state.covariance.retain_grad()
        state.opacity_logit.retain_grad()
        gaussians, _ = AnalyticalSurfaceReadout().double()(atlas, state, mapping)
        gaussians.means.retain_grad()
        extrinsic = torch.eye(4, dtype=torch.float64)[:3]
        extrinsic[:, 3] = torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64)
        intrinsic = torch.tensor([[12.0, 0.0, 8.0], [0.0, 12.0, 8.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        cameras = CameraBatch(extrinsic[None], intrinsic[None], 16, 16)
        result = ReferenceGaussianRenderer()(gaussians, cameras)
        scene = type(
            "Scene",
            (),
            {"gaussians": gaussians, "render_cameras": cameras},
        )()
        opacity_bound = conservative_tile_opacity_bound(scene, tile_size=16)
        self.assertLessEqual(
            float(result.alpha.max()),
            float(opacity_bound.max()) + 1.0e-10,
        )
        loss = (
            result.color.square().mean()
            + result.alpha.mean()
            + result.depth.mean()
            + result.normal.square().mean()
        )
        loss.backward()
        self.assertIsNotNone(gaussians.means.grad)
        self.assertTrue(torch.all(torch.isfinite(gaussians.means.grad)))
        self.assertGreater(float(gaussians.means.grad.abs().sum()), 0.0)
        self.assertIsNotNone(mapping.evidence.positions.grad)
        self.assertIsNotNone(mapping.evidence.features.grad)
        self.assertTrue(torch.all(torch.isfinite(mapping.evidence.positions.grad)))
        self.assertTrue(torch.all(torch.isfinite(mapping.evidence.features.grad)))
        self.assertGreater(float(mapping.evidence.positions.grad.abs().sum()), 0.0)
        self.assertGreater(float(mapping.evidence.features.grad.abs().sum()), 0.0)
        self.assertIsNotNone(state.covariance.grad)
        self.assertIsNotNone(state.opacity_logit.grad)
        self.assertGreater(float(state.covariance.grad.abs().sum()), 0.0)
        self.assertGreater(float(state.opacity_logit.grad.abs().sum()), 0.0)

    @unittest.skipUnless(importlib.util.find_spec("diff_gaussian_rasterization"), "CUDA rasterizer is built only on the server")
    def test_cuda_reference_equivalence_small_scene(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        atlas, mapping, selection = _fixture()
        state = GraftGS._state_from_mapping(atlas, mapping, selection)
        gaussians, _ = AnalyticalSurfaceReadout().double()(atlas, state, mapping)
        gaussians = type(gaussians)(**{name: getattr(gaussians, name).float().cuda() for name in gaussians.__dataclass_fields__})
        gaussians.covariance.retain_grad()
        extrinsic = torch.eye(4, device="cuda")[:3]
        extrinsic[:, 3] = torch.tensor([0.0, 0.0, 3.0], device="cuda")
        intrinsic = torch.tensor([[12.0, 0.0, 7.25], [0.0, 12.0, 8.5], [0.0, 0.0, 1.0]], device="cuda")
        camera = CameraBatch(extrinsic[None], intrinsic[None], 16, 16)
        background = torch.tensor([0.15, 0.25, 0.35], device="cuda")
        reference = ReferenceGaussianRenderer()(gaussians, camera, background=background)
        optimized = CudaGaussianRenderer()(gaussians, camera, background=background)
        self.assertEqual(optimized.color.dtype, torch.float32)
        self.assertEqual(optimized.depth.dtype, torch.float32)
        torch.testing.assert_close(optimized.color, reference.color, atol=5.0e-2, rtol=8.0e-2)
        torch.testing.assert_close(optimized.alpha, reference.alpha, atol=5.0e-2, rtol=8.0e-2)
        visible = (optimized.alpha > 0.05) & (reference.alpha > 0.05)
        if bool(torch.any(visible)):
            torch.testing.assert_close(
                optimized.depth[visible],
                reference.depth[visible],
                atol=2.0e-2,
                rtol=2.0e-2,
            )
            cosine = torch.sum(optimized.normal * reference.normal, dim=1, keepdim=True).abs()
            self.assertGreater(float(cosine[visible].mean()), 0.9)
        (optimized.color.mean() + optimized.alpha.mean() + optimized.depth.mean()).backward()
        self.assertIsNotNone(gaussians.covariance.grad)
        self.assertTrue(torch.all(torch.isfinite(gaussians.covariance.grad)))
        self.assertGreater(float(gaussians.covariance.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

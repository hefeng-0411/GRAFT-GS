"""Source-level guards for the compiled linear-memory geometry path."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf8")


class CompiledGeometryStaticTest(unittest.TestCase):
    def test_active_package_contains_no_dense_cdist(self) -> None:
        offenders = [
            str(path.relative_to(ROOT))
            for path in (ROOT / "graft_gs").rglob("*.py")
            if "torch.cdist" in path.read_text(encoding="utf8")
        ]
        self.assertEqual(offenders, [])

    def test_transport_support_is_frnn_bounded_and_certified(self) -> None:
        mapping = source("graft_gs/mapping/manifold_mapping.py")
        atlas = source("graft_gs/geometry/atlas.py")
        primitives = source("graft_gs/kernels/geometry_primitives.py")
        config = source("configs/graft_gs_a800_native.yaml")
        self.assertIn("FRNNAtlasGraph", mapping)
        self.assertIn("class FRNNAtlasGraph", atlas)
        self.assertIn("frnn.frnn_grid_points(", primitives)
        self.assertIn("FRNN radius support saturated", primitives)
        self.assertIn("frnn_max_neighbors: 256", config)

    def test_fugw_fourth_order_term_is_a_keops_edge_reduction(self) -> None:
        primitives = source("graft_gs/kernels/geometry_primitives.py")
        mapping = source("graft_gs/mapping/manifold_mapping.py")
        self.assertIn("class KeOpsFUGWSolver", primitives)
        self.assertIn("class _KeOpsFUGWBCD(torch.autograd.Function)", primitives)
        self.assertIn("cross = (c1 * c2 * mass_f).sum(dim=1)", primitives)
        self.assertIn("self.fugw = KeOpsFUGWSolver", mapping)

    def test_losses_use_geomloss_and_keops_online_reductions(self) -> None:
        losses = source("graft_gs/engine/losses.py")
        primitives = source("graft_gs/kernels/geometry_primitives.py")
        self.assertIn("sinkhorn_surface_divergence(", losses)
        self.assertIn('backend="online"', primitives)
        self.assertIn("from pykeops.torch import LazyTensor", primitives)
        self.assertIn("keops_squared_distance_minima", losses)

    def test_production_renderer_is_one_batched_gsplat_call(self) -> None:
        renderer = source("graft_gs/readout/renderer.py")
        pipeline = source("graft_gs/integration/pipeline.py")
        production = renderer.split("class GsplatRenderer", 1)[1]
        self.assertEqual(production.count("rendered, alpha, metadata = rasterization("), 1)
        self.assertIn("packed=True", production)
        self.assertIn("quats=quaternions.contiguous()", production)
        self.assertIn("scales=scales.contiguous()", production)
        self.assertIn("extra_signals=world_normals", production)
        self.assertIn('renderer_backend: str = "gsplat"', pipeline)

    def test_nerfacc_adapter_uses_compiled_scan_and_occupancy_grid(self) -> None:
        primitives = source("graft_gs/kernels/geometry_primitives.py")
        self.assertIn("render_weight_from_density(", primitives)
        self.assertIn("accumulate_along_rays(", primitives)
        self.assertIn("OccGridEstimator", primitives)

    def test_local_extension_install_contract_is_explicit(self) -> None:
        requirements = source("requirements-geometry-local.txt")
        for relative in (
            "../extensions/keops/pykeops",
            "../extensions/geomloss",
            "../extensions/FRNN",
            "../extensions/gsplat",
            "../extensions/nerfacc",
        ):
            self.assertIn(relative, requirements)


if __name__ == "__main__":
    unittest.main()

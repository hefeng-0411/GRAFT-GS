"""CPU-only manifest checks that do not import PyTorch or model packages."""

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import sys
import struct
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_OBJECT_ID = "17a53839ae5da04c75ea21335d4bdc8ddc26b45f7bb9d0e18f5afaa397e43a17"
MANIFEST = Path(
    os.environ.get(
        "GRAFT_GS_MESHFLEET_MANIFEST",
        str(PROJECT_ROOT / "data_manifests" / "meshfleet_local_audit.jsonl"),
    )
)
SUMMARY = MANIFEST.with_suffix(MANIFEST.suffix + ".summary.json")
DATASET = Path(
    os.environ.get(
        "GRAFT_GS_MESHFLEET_ROOT",
        r"D:\VsCode\MVG\Base\MeshFleet_TRELLIS",
    )
)


def _records():
    return [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf8").splitlines()
        if line
    ]


def _canonical_record():
    matches = [
        record for record in _records()
        if record["object_id"] == CANONICAL_OBJECT_ID
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"canonical object must occur exactly once in manifest, found {len(matches)}"
        )
    return matches[0]


def _load_manifest_module():
    source = PROJECT_ROOT / "graft_gs" / "data" / "meshfleet.py"
    name = "meshfleet_manifest_static_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StaticMeshFleetManifestTest(unittest.TestCase):
    @staticmethod
    def _write_triangle_ply(path: Path, vertices, faces) -> None:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(vertices)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            f"element face {len(faces)}\n"
            "property list uchar uint vertex_indices\nend_header\n"
        ).encode("ascii")
        payload = bytearray(header)
        for vertex in vertices:
            payload.extend(struct.pack("<fff", *vertex))
        for face in faces:
            payload.extend(struct.pack("<BIII", 3, *face))
        path.write_bytes(payload)

    def test_records_are_relationally_verified(self) -> None:
        records = _records()
        self.assertGreater(len(records), 0)
        for record in records:
            self.assertEqual(record["schema"], "meshfleet-trellis-object-v2")
            self.assertTrue(record["object_id"])
            modalities = record["modalities"]
            if "features" in modalities and "latents" in modalities:
                self.assertTrue(record["checks"]["feature_indices_equal_latent_coords"])
            if "surface_voxels" in modalities:
                grid = record["checks"]["surface_voxel_grid"]
                self.assertEqual(grid["resolution"], 64)
                self.assertLessEqual(grid["maximum_center_residual"], 1.0e-6)
                self.assertTrue(grid["indices_in_bounds"])
                if "features" in modalities:
                    self.assertTrue(
                        record["checks"]["surface_voxel_indices_equal_feature_indices"]
                    )

    def test_canonical_raw_topology_is_diagnostic_not_a_label(self) -> None:
        record = _canonical_record()
        topology = record["checks"]["render_mesh_topology"]
        self.assertEqual(topology["vertex_count"], 78448)
        self.assertEqual(topology["edge_count"], 236075)
        self.assertEqual(topology["face_count"], 157592)
        self.assertEqual(topology["connected_components"], 8)
        self.assertEqual(topology["euler_characteristic"], -35)
        self.assertEqual(topology["boundary_edge_count"], 0)
        self.assertEqual(topology["nonmanifold_edge_count"], 313)
        self.assertEqual(topology["edge_incidence_histogram"], {"2": 235762, "4": 313})
        self.assertEqual(topology["maximum_edge_incidence"], 4)
        self.assertEqual(topology["isolated_vertex_count"], 0)
        self.assertEqual(topology["degenerate_face_count"], 0)
        self.assertFalse(topology["watertight"])
        self.assertFalse(topology["closed_two_manifold"])
        self.assertEqual(topology["orientability_status"], "indeterminate_nonmanifold")
        self.assertIsNone(topology["orientable"])
        contract = record["topology_supervision"]
        selected = contract["selected_label"]
        self.assertEqual(selected["status"], "unavailable")
        self.assertEqual(selected["provenance"], "unavailable")
        self.assertEqual(selected["confidence"], 0.0)
        for key in (
            "hard_topology_supervision_admissible",
            "hard_betti_supervision_admissible",
            "hard_persistence_supervision_admissible",
            "hard_stratum_supervision_admissible",
            "manifold_certification_admissible",
        ):
            self.assertFalse(selected[key])
        self.assertIsNone(selected["target_betti_z2"])
        self.assertIsNone(selected["target_persistence"])
        self.assertIsNone(selected["target_stratum"])
        self.assertTrue(contract["derived_topology_statistics"]["available"])
        self.assertFalse(contract["repaired_topology"]["available"])
        self.assertFalse(contract["teacher_pseudo_topology"]["available"])
        policy = _load_manifest_module()
        for target in ("topology", "betti", "persistence", "stratum", "manifold_certification"):
            self.assertFalse(policy.topology_supervision_is_admissible(contract, target))
        self.assertIn("surface_voxels", record["supervision"]["ground_truth"])
        self.assertIn("render_mesh_geometry", record["supervision"]["ground_truth"])
        self.assertNotIn("validated_topology_betti_z2", record["supervision"]["ground_truth"])

    def test_manifest_regeneration_is_deterministic(self) -> None:
        if not DATASET.is_dir():
            self.skipTest("audited MeshFleet_TRELLIS dataset is not mounted")
        module = _load_manifest_module()
        with tempfile.TemporaryDirectory() as directory:
            rebuilt = Path(directory) / "manifest.jsonl"
            module.build_meshfleet_manifest(
                DATASET,
                rebuilt,
                object_ids=(CANONICAL_OBJECT_ID,),
            )
            rebuilt_records = [
                json.loads(line)
                for line in rebuilt.read_text(encoding="utf8").splitlines()
                if line
            ]
            self.assertEqual(rebuilt_records, [_canonical_record()])

    def test_closed_oriented_tetrahedron_is_admissible(self) -> None:
        module = _load_manifest_module()
        vertices = (
            (1.0, 1.0, 1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (1.0, -1.0, -1.0),
        )
        faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tetra.ply"
            self._write_triangle_ply(path, vertices, faces)
            audit = module._triangle_mesh_topology(path)
        self.assertEqual(audit["connected_components"], 1)
        self.assertEqual(audit["edge_incidence_histogram"], {"2": 6})
        self.assertEqual(audit["euler_characteristic"], 2)
        self.assertTrue(audit["watertight"])
        self.assertTrue(audit["closed_two_manifold"])
        self.assertTrue(audit["orientable"])
        self.assertTrue(audit["orientation_consistent"])
        self.assertTrue(audit["hard_topology_supervision_admissible"])
        self.assertEqual(audit["betti_z2"], [1, 0, 1])

    def test_summary_matches_records(self) -> None:
        records = [line for line in MANIFEST.read_text(encoding="utf8").splitlines() if line]
        summary = json.loads(SUMMARY.read_text(encoding="utf8"))
        self.assertEqual(summary["record_count"], len(records))
        self.assertEqual(sum(summary["split_counts"].values()), len(records))

    def test_declared_and_physical_views_are_never_conflated(self) -> None:
        records = _records()
        for record in records:
            for view in record["views"].values():
                self.assertEqual(
                    view["declared_frame_count"],
                    view["available_frame_count"] + view["missing_frame_count"],
                )
                self.assertEqual(view["available_frame_count"], len(view["available_frames"]))
                self.assertEqual(view["missing_frame_count"], len(view["missing_frames"]))


if __name__ == "__main__":
    unittest.main()

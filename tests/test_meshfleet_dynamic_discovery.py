"""CPU-only tests for modality-centric MeshFleet intersection discovery."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "graft_gs_meshfleet_dynamic_test"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, PROJECT_ROOT / "graft_gs" / "data" / "meshfleet.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load MeshFleet dataset module")
MESHFLEET = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = MESHFLEET
SPEC.loader.exec_module(MESHFLEET)

DEFAULT_OPTIONAL_MODALITIES = MESHFLEET.DEFAULT_OPTIONAL_MODALITIES
DEFAULT_PRIMARY_MODALITIES = MESHFLEET.DEFAULT_PRIMARY_MODALITIES
DEFAULT_REQUIRED_MODALITIES = MESHFLEET.DEFAULT_REQUIRED_MODALITIES
MeshFleetDatasetConfig = MESHFLEET.MeshFleetDatasetConfig
MeshFleetObjectDataset = MESHFLEET.MeshFleetObjectDataset
build_meshfleet_manifest = MESHFLEET.build_meshfleet_manifest
load_meshfleet_manifest = MESHFLEET.load_meshfleet_manifest


TRAIN_COMPLETE = "1" * 64
TRAIN_INCOMPLETE = "2" * 64
TEST_COMPLETE = "3" * 64


def _write_required_object(root: Path, split: str, object_id: str) -> None:
    render = root / split / "renders" / object_id
    render.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (8, 8), (255, 64, 32, 255)).save(render / "000.png")
    (render / "transforms.json").write_text(
        json.dumps(
            {
                "camera_angle_x": 0.8,
                "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                "scale": 1.0,
                "offset": [0.0, 0.0, 0.0],
                "frames": [
                    {
                        "file_path": "000.png",
                        "transform_matrix": np.eye(4).tolist(),
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf8",
    )
    latent = root / split / "latents" / "slat-variant"
    latent.mkdir(parents=True, exist_ok=True)
    np.savez(
        latent / f"{object_id}.npz",
        coords=np.asarray([[1, 2, 3]], dtype=np.uint8),
        feats=np.asarray([[0.25, -0.5]], dtype=np.float32),
    )
    normalized = root / split / "mesh_normalized" / object_id
    normalized.mkdir(parents=True, exist_ok=True)
    (normalized / "bounding_box.json").write_text(
        json.dumps(
            {
                "min": [-0.5, -0.5, -0.5],
                "max": [0.5, 0.5, 0.5],
                "width": 1.0,
                "height": 1.0,
                "length": 1.0,
            },
            sort_keys=True,
        ),
        encoding="utf8",
    )
    (normalized / "mesh.glb").write_bytes(struct.pack("<4sII", b"glTF", 2, 12))


class DynamicMeshFleetDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "dataset"
        _write_required_object(self.root, "train", TRAIN_COMPLETE)
        _write_required_object(self.root, "test", TEST_COMPLETE)
        incomplete = self.root / "train" / "latents"
        incomplete.mkdir(parents=True, exist_ok=True)
        np.savez(
            incomplete / f"{TRAIN_INCOMPLETE}.npz",
            coords=np.zeros((1, 3), dtype=np.uint8),
            feats=np.zeros((1, 2), dtype=np.float32),
        )
        self.manifest = Path(temporary.name) / "manifest.jsonl"

    def test_intersection_sniffer_admits_all_complete_objects(self) -> None:
        summary = build_meshfleet_manifest(
            self.root,
            self.manifest,
            inspect_image_headers=True,
        )
        records = load_meshfleet_manifest(self.manifest)
        self.assertEqual(
            [(record.split, record.object_id) for record in records],
            [("train", TRAIN_COMPLETE), ("test", TEST_COMPLETE)],
        )
        self.assertEqual(summary["candidate_counts"], {"train": 2, "test": 1})
        self.assertEqual(summary["split_counts"], {"train": 1, "test": 1})
        self.assertEqual(summary["rejected_counts"], {"train": 1, "test": 0})
        self.assertEqual(
            summary["discovery_policy"]["primary_modalities"],
            list(DEFAULT_PRIMARY_MODALITIES),
        )
        self.assertEqual(
            summary["discovery_policy"]["required_modalities"],
            list(DEFAULT_REQUIRED_MODALITIES),
        )
        rejection = json.loads(
            self.manifest.with_suffix(".jsonl.rejected.jsonl").read_text(
                encoding="utf8"
            )
        )
        self.assertEqual(rejection["object_id"], TRAIN_INCOMPLETE)
        self.assertEqual(
            rejection["missing_required_modalities"],
            ["renders", "mesh_normalized"],
        )

    def test_optional_absence_is_explicit_and_never_rejects(self) -> None:
        build_meshfleet_manifest(self.root, self.manifest)
        records = load_meshfleet_manifest(self.manifest)
        for record in records:
            self.assertEqual(record.discovery["available_modalities"], [
                "renders", "latents", "mesh_normalized"
            ])
            self.assertEqual(
                record.discovery["missing_optional_modalities"],
                list(DEFAULT_OPTIONAL_MODALITIES),
            )
            latent_paths = record.discovery["structural_map"]["latents"]["npz"]
            self.assertEqual(len(latent_paths), 1)
            self.assertIn("latents/slat-variant/", latent_paths[0])

    def test_manifest_and_rejection_inventory_are_deterministic(self) -> None:
        build_meshfleet_manifest(self.root, self.manifest)
        first = self.manifest.read_bytes()
        first_rejected = self.manifest.with_suffix(
            ".jsonl.rejected.jsonl"
        ).read_bytes()
        build_meshfleet_manifest(self.root, self.manifest)
        self.assertEqual(self.manifest.read_bytes(), first)
        self.assertEqual(
            self.manifest.with_suffix(".jsonl.rejected.jsonl").read_bytes(),
            first_rejected,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "dataset tensor loading requires the declared PyTorch environment",
    )
    def test_loader_resolves_manifest_paths_from_configurable_root(self) -> None:
        build_meshfleet_manifest(self.root, self.manifest)
        dataset = MeshFleetObjectDataset(
            MeshFleetDatasetConfig(
                root=self.root,
                manifest=self.manifest,
                split="train",
                minimum_views=1,
                maximum_views=1,
                image_size=(8, 8),
                load_trellis_latents=True,
            )
        )
        sample = dataset[0]
        self.assertEqual(sample["object_id"], TRAIN_COMPLETE)
        self.assertTrue(
            Path(sample["modality_paths"]["latents"]).is_absolute()
        )
        self.assertEqual(tuple(sample["images"].shape), (1, 3, 8, 8))
        self.assertEqual(tuple(sample["trellis_latent_coords"].shape), (1, 3))

    def test_policy_can_promote_an_optional_modality_to_required(self) -> None:
        summary = build_meshfleet_manifest(
            self.root,
            self.manifest,
            required_modalities=(*DEFAULT_REQUIRED_MODALITIES, "voxels"),
            optional_modalities=tuple(
                item for item in DEFAULT_OPTIONAL_MODALITIES if item != "voxels"
            ),
        )
        self.assertEqual(summary["record_count"], 0)
        self.assertEqual(summary["rejected_counts"], {"train": 2, "test": 1})

    def test_ambiguous_optional_variants_are_recorded_not_guessed(self) -> None:
        for variant in ("dino-a", "dino-b"):
            directory = self.root / "train" / "features" / variant
            directory.mkdir(parents=True, exist_ok=True)
            np.savez(
                directory / f"{TRAIN_COMPLETE}.npz",
                indices=np.asarray([[1, 2, 3]], dtype=np.uint8),
                patchtokens=np.zeros((1, 4), dtype=np.float32),
            )
        build_meshfleet_manifest(self.root, self.manifest)
        record = load_meshfleet_manifest(self.manifest)[0]
        self.assertEqual(
            record.discovery["ambiguous_optional_artifacts"],
            {"features": ["npz"]},
        )
        self.assertNotIn("features", record.modalities)
        self.assertEqual(
            len(record.discovery["structural_map"]["features"]["npz"]), 2
        )
        self.assertTrue(
            any("model variants are ambiguous" in warning for warning in record.warnings)
        )

    def test_manifest_paths_cannot_escape_configured_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "escapes the dataset root"):
            MESHFLEET._resolve_manifest_path(self.root, "../outside.npz")


if __name__ == "__main__":
    unittest.main()

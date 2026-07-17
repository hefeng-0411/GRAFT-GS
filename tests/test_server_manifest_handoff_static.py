"""Pure-Python validation of remote manifest reuse and rebuild decisions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC = importlib.util.spec_from_file_location(
        "graft_gs_validate_server", SCRIPTS / "validate_server.py"
    )
    if SPEC is None or SPEC.loader is None:
        raise RuntimeError("cannot load validate_server.py")
    VALIDATOR = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(VALIDATOR)
finally:
    sys.path.pop(0)


class ServerManifestHandoffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "dataset"
        (self.root / "train").mkdir(parents=True)
        (self.root / "test").mkdir()
        self.manifest = Path(self.temporary.name) / "manifest.jsonl"

    def _write_contract(
        self,
        records: list[dict[str, object]],
        *,
        root: Path | None = None,
        schema: str | None = None,
        record_count: int | None = None,
    ) -> None:
        self.manifest.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf8",
        )
        summary = {
            "dataset_root": str(self.root if root is None else root),
            "schema": VALIDATOR.EXPECTED_MESHFLEET_SCHEMA if schema is None else schema,
            "record_count": len(records) if record_count is None else record_count,
        }
        self.manifest.with_suffix(".jsonl.summary.json").write_text(
            json.dumps(summary), encoding="utf8"
        )

    def _canonical(self, split: str = "test") -> dict[str, object]:
        return {"object_id": VALIDATOR.CANONICAL_OBJECT_ID, "split": split}

    def test_compatible_many_object_manifest_is_reused_and_selects_by_id(self) -> None:
        records = [
            {"object_id": "first-object-is-not-canonical", "split": "train"},
            self._canonical(),
            {"object_id": "later-object", "split": "test"},
        ]
        self._write_contract(records)
        audit = VALIDATOR._inspect_manifest_contract(self.manifest, self.root.resolve())
        self.assertTrue(audit["valid"], audit["errors"])
        self.assertFalse(VALIDATOR._manifest_requires_rebuild(False, audit))
        self.assertTrue(VALIDATOR._manifest_requires_rebuild(True, audit))
        self.assertEqual(audit["record_count"], 3)
        self.assertEqual(audit["canonical_split"], "test")

    def test_stale_schema_requests_rebuild(self) -> None:
        self._write_contract([self._canonical()], schema="meshfleet-v1")
        audit = VALIDATOR._inspect_manifest_contract(self.manifest, self.root.resolve())
        self.assertFalse(audit["valid"])
        self.assertTrue(VALIDATOR._manifest_requires_rebuild(False, audit))
        self.assertIn("manifest schema does not match the loader contract", audit["errors"])

    def test_record_count_mismatch_requests_rebuild(self) -> None:
        self._write_contract([self._canonical()], record_count=2)
        audit = VALIDATOR._inspect_manifest_contract(self.manifest, self.root.resolve())
        self.assertFalse(audit["valid"])
        self.assertTrue(VALIDATOR._manifest_requires_rebuild(False, audit))
        self.assertIn("manifest record count does not match its summary", audit["errors"])

    def test_dataset_root_mismatch_requests_rebuild(self) -> None:
        self._write_contract([self._canonical()], root=self.root.parent / "other")
        audit = VALIDATOR._inspect_manifest_contract(self.manifest, self.root.resolve())
        self.assertFalse(audit["valid"])
        self.assertTrue(VALIDATOR._manifest_requires_rebuild(False, audit))
        self.assertIn("manifest summary belongs to a different dataset root", audit["errors"])

    def test_missing_or_duplicate_canonical_identity_requests_rebuild(self) -> None:
        cases = (
            [{"object_id": "not-canonical", "split": "test"}],
            [self._canonical("train"), self._canonical("test")],
        )
        for records in cases:
            with self.subTest(count=len(records)):
                self._write_contract(records)
                audit = VALIDATOR._inspect_manifest_contract(
                    self.manifest, self.root.resolve()
                )
                self.assertFalse(audit["valid"])
                self.assertTrue(VALIDATOR._manifest_requires_rebuild(False, audit))
                self.assertTrue(
                    any("canonical object occurs" in error for error in audit["errors"])
                )

    def test_missing_or_malformed_summary_requests_rebuild(self) -> None:
        self.manifest.write_text(json.dumps(self._canonical()) + "\n", encoding="utf8")
        audit = VALIDATOR._inspect_manifest_contract(self.manifest, self.root.resolve())
        self.assertFalse(audit["valid"])
        self.assertTrue(VALIDATOR._manifest_requires_rebuild(False, audit))
        self.assertIn("manifest summary is missing", audit["errors"])
        self.manifest.with_suffix(".jsonl.summary.json").write_text("{", encoding="utf8")
        audit = VALIDATOR._inspect_manifest_contract(self.manifest, self.root.resolve())
        self.assertFalse(audit["valid"])
        self.assertTrue(any("summary is unreadable" in error for error in audit["errors"]))

    def test_a800_cuda118_bf16_contract_is_explicit(self) -> None:
        valid = {
            "cuda_available": True,
            "torch_cuda": "11.8",
            "bf16_supported": True,
            "devices": [{"name": "NVIDIA A800-SXM4-80GB"}],
        }
        self.assertEqual(VALIDATOR._accelerator_contract_errors(valid), [])
        cases = (
            ({**valid, "cuda_available": False}, "CUDA is unavailable"),
            ({**valid, "torch_cuda": "12.1"}, "CUDA 11.8"),
            ({**valid, "bf16_supported": False}, "BF16"),
            ({**valid, "devices": [{"name": "NVIDIA RTX 2060"}]}, "A800"),
        )
        for details, fragment in cases:
            with self.subTest(fragment=fragment):
                errors = VALIDATOR._accelerator_contract_errors(details)
                self.assertTrue(any(fragment in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()

"""Pure-Python tests for the exact remote environment contract."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "graft_gs_validate_environment", ROOT / "scripts" / "validate_environment.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load validate_environment.py")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
EXTERNAL_SPEC = importlib.util.spec_from_file_location(
    "graft_gs_external_contract",
    ROOT / "graft_gs" / "integration" / "external.py",
)
if EXTERNAL_SPEC is None or EXTERNAL_SPEC.loader is None:
    raise RuntimeError("cannot load integration/external.py")
EXTERNAL = importlib.util.module_from_spec(EXTERNAL_SPEC)
EXTERNAL_SPEC.loader.exec_module(EXTERNAL)


class ExactEnvironmentContractTest(unittest.TestCase):
    def _requirements(self, text: str) -> Path:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf8", delete=False
        )
        with temporary:
            temporary.write(text)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        return Path(temporary.name)

    def test_exact_pins_and_pep503_names(self) -> None:
        required = VALIDATOR.parse_pinned_requirements(
            self._requirements("My_Package==1.2.3\nother.package==4+cu118  # pinned\n")
        )
        self.assertEqual(set(required), {"my-package", "other-package"})
        comparison = VALIDATOR.compare_environment(
            required,
            {"my-package": ["1.2.3"], "other-package": ["4+cu118"]},
        )
        self.assertTrue(comparison["valid"])
        self.assertEqual(comparison["matched_count"], 2)

    def test_missing_and_mismatched_are_not_silently_accepted(self) -> None:
        required = VALIDATOR.parse_pinned_requirements(
            self._requirements("alpha==1\nbeta==2\n")
        )
        comparison = VALIDATOR.compare_environment(required, {"alpha": ["0.9"]})
        self.assertFalse(comparison["valid"])
        self.assertEqual(comparison["mismatched"][0]["name"], "alpha")
        self.assertEqual(comparison["missing"][0]["name"], "beta")

    def test_non_exact_or_conditional_requirement_is_rejected(self) -> None:
        for line in ("alpha>=1\n", "alpha==1; python_version > '3.10'\n", "-r base.txt\n"):
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    VALIDATOR.parse_pinned_requirements(self._requirements(line))

    def test_repository_contract_is_fully_exact_and_cuda_pin_is_preserved(self) -> None:
        required = VALIDATOR.parse_pinned_requirements(ROOT / "requirements.txt")
        self.assertEqual(len(required), 444)
        self.assertEqual(required["torch"]["version"], "2.4.0+cu118")
        self.assertEqual(required["torchvision"]["version"], "0.19.0+cu118")
        # ipykernel 7.3 declares jupyter-client>=8.9.0.  Pinning the legacy
        # 7.4.9 client makes an otherwise exact environment fail ``pip check``.
        self.assertEqual(required["ipykernel"]["version"], "7.3.0")
        self.assertEqual(required["jupyter-client"]["version"], "8.9.1")

    def test_checkpoint_resolution_is_cli_then_environment_then_official_default(self) -> None:
        names = (
            "GRAFT_GS_VGGT_CHECKPOINT",
            "VGGT_CHECKPOINT",
            "GRAFT_GS_TRELLIS_CHECKPOINT",
            "TRELLIS_CHECKPOINT",
        )
        clean = {name: os.environ[name] for name in names if name in os.environ}
        with patch.dict(os.environ, {}, clear=False):
            for name in names:
                os.environ.pop(name, None)
            self.assertEqual(EXTERNAL.resolve_vggt_checkpoint(), "facebook/VGGT-1B")
            self.assertEqual(
                EXTERNAL.resolve_trellis_checkpoint(),
                "microsoft/TRELLIS-image-large",
            )
            os.environ["VGGT_CHECKPOINT"] = "legacy-vggt"
            os.environ["GRAFT_GS_VGGT_CHECKPOINT"] = "deployment-vggt"
            self.assertEqual(EXTERNAL.resolve_vggt_checkpoint(), "deployment-vggt")
            self.assertEqual(
                EXTERNAL.resolve_vggt_checkpoint("explicit-vggt"),
                "explicit-vggt",
            )
            os.environ["TRELLIS_CHECKPOINT"] = "legacy-trellis"
            os.environ["GRAFT_GS_TRELLIS_CHECKPOINT"] = "deployment-trellis"
            self.assertEqual(
                EXTERNAL.resolve_trellis_checkpoint(),
                "deployment-trellis",
            )
        for name in names:
            if name in clean:
                os.environ[name] = clean[name]
            else:
                os.environ.pop(name, None)


if __name__ == "__main__":
    unittest.main()

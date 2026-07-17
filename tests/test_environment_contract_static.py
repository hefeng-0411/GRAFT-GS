"""Pure-Python tests for the exact remote environment contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "graft_gs_validate_environment", ROOT / "scripts" / "validate_environment.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load validate_environment.py")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


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


if __name__ == "__main__":
    unittest.main()

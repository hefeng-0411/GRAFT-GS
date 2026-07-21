"""Numerical policy tests for native A800 precision boundaries."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch

from graft_gs.engine import (
    NativePrecisionPolicy,
    load_precision_policy,
    validate_precision_policy,
)


ROOT = Path(__file__).resolve().parents[1]


class NativePrecisionPolicyTest(unittest.TestCase):
    def test_repository_policy_disables_tf32_and_preserves_state_precision(self) -> None:
        policy = load_precision_policy(ROOT / "configs" / "graft_gs_a800_native.yaml")
        self.assertEqual(policy.backbone, "bfloat16")
        self.assertEqual(policy.geometric_state, "float32")
        self.assertEqual(policy.analytical_solve, "float32")
        self.assertEqual(policy.diagnostics, "float64")
        self.assertEqual(policy.float32_matmul_precision, "highest")
        self.assertFalse(policy.allow_tf32)

    def test_runtime_flags_are_applied_and_reported(self) -> None:
        previous_precision = torch.get_float32_matmul_precision()
        previous_matmul = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn = torch.backends.cudnn.allow_tf32
        try:
            record = NativePrecisionPolicy().apply()
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertEqual(record["torch_float32_matmul_precision"], "highest")
            self.assertFalse(record["cuda_matmul_allow_tf32"])
            self.assertFalse(record["cudnn_allow_tf32"])
        finally:
            torch.set_float32_matmul_precision(previous_precision)
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul
            torch.backends.cudnn.allow_tf32 = previous_cudnn

    def test_precision_weakening_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NativePrecisionPolicy(allow_tf32=True)
        with self.assertRaises(ValueError):
            NativePrecisionPolicy(geometric_state="bfloat16")
        with self.assertRaises(ValueError):
            NativePrecisionPolicy(float32_matmul_precision="high")

    def test_checkpoint_precision_provenance_is_exact(self) -> None:
        policy = NativePrecisionPolicy()
        trainer = {
            "precision_backbone": policy.backbone,
            "precision_geometric_state": policy.geometric_state,
            "precision_analytical_solve": policy.analytical_solve,
            "precision_diagnostics": policy.diagnostics,
            "precision_float32_matmul": policy.float32_matmul_precision,
            "precision_allow_tf32": policy.allow_tf32,
        }
        validate_precision_policy(
            {
                "format_version": 6,
                "trainer_config": trainer,
                "precision_runtime": policy.apply(),
            },
            policy,
        )
        with self.assertRaises(ValueError):
            validate_precision_policy(
                {
                    "format_version": 6,
                    "trainer_config": {**trainer, "precision_allow_tf32": True},
                    "precision_runtime": policy.apply(),
                },
                policy,
            )
        with self.assertRaises(ValueError):
            validate_precision_policy({"format_version": 6}, policy)
        validate_precision_policy({"format_version": 5}, policy)
        with self.assertRaises(ValueError):
            validate_precision_policy(
                {"format_version": 5},
                NativePrecisionPolicy(backbone="float16"),
            )


if __name__ == "__main__":
    unittest.main()

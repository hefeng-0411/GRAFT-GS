"""Checkpoint compatibility boundaries for architecture and runtime policy."""

from __future__ import annotations

from dataclasses import asdict, replace
import unittest

from graft_gs.engine import (
    model_config_differences,
    validate_model_config_compatibility,
)
from graft_gs.equivariant.gsta import GSTAConfig
from graft_gs.integration import GraftGSConfig


class CheckpointConfigCompatibilityTest(unittest.TestCase):
    def test_legacy_checkpoint_may_omit_activation_checkpointing(self) -> None:
        current = GraftGSConfig(
            attention=GSTAConfig(activation_checkpointing=True),
        )
        checkpoint = asdict(current)
        del checkpoint["attention"]["activation_checkpointing"]
        ignored = validate_model_config_compatibility(
            checkpoint,
            current,
            context="test checkpoint",
        )
        self.assertEqual(len(ignored), 1)
        self.assertEqual(ignored[0]["path"], "attention.activation_checkpointing")
        self.assertFalse(ignored[0]["checkpoint_present"])
        self.assertTrue(ignored[0]["current_value"])

    def test_activation_checkpointing_value_difference_is_execution_only(self) -> None:
        checkpoint = GraftGSConfig(
            attention=GSTAConfig(activation_checkpointing=False),
        )
        current = GraftGSConfig(
            attention=GSTAConfig(activation_checkpointing=True),
        )
        self.assertFalse(model_config_differences(checkpoint, current))
        ignored = validate_model_config_compatibility(
            checkpoint,
            current,
            context="test checkpoint",
        )
        self.assertEqual(len(ignored), 1)

    def test_real_architecture_difference_is_rejected_with_path(self) -> None:
        checkpoint = GraftGSConfig()
        current = replace(checkpoint, encoder_layers=checkpoint.encoder_layers + 1)
        with self.assertRaisesRegex(
            ValueError,
            "encoder_layers.*checkpoint=4, current=5",
        ):
            validate_model_config_compatibility(
                checkpoint,
                current,
                context="test checkpoint",
            )


if __name__ == "__main__":
    unittest.main()

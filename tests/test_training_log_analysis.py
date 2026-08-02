"""Failure-classification tests for operational training logs."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_training_log import analyze


class TrainingLogAnalysisTest(unittest.TestCase):
    def test_legacy_parent_timeout_is_not_mislabeled_as_child_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.log"
            ready = {
                "rank": 0,
                "collective_stage": "backward",
                "local_nonfinite": False,
            }
            control = {"action": "SIGTERM_PROCESS_GROUP", "reason": "probe_timeout"}
            path.write_text(
                "GRAFT_GS_DDP_LOCAL_READY " + json.dumps(ready) + "\n"
                "Sampling: 100% 12/12\n"
                "GRAFT_GS_AUTOTUNE_PROBE_CONTROL " + json.dumps(control) + "\n",
                encoding="utf8",
            )
            report = analyze(path)
            self.assertEqual(
                report["failure_class"],
                "supervisor.legacy_fixed_wall_timeout",
            )
            self.assertTrue(report["synchronized_finite_backward_entry"])

    def test_cuda_oom_has_capacity_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.log"
            path.write_text(
                "torch.OutOfMemoryError: CUDA out of memory\n",
                encoding="utf8",
            )
            self.assertEqual(analyze(path)["failure_class"], "capacity.cuda_oom")


if __name__ == "__main__":
    unittest.main()

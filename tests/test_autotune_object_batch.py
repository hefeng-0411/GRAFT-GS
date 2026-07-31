"""Process-isolation and batch-policy regression tests for the GPU tuner."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

from scripts.autotune_object_batch import (
    _candidate_policy_failure,
    _gpu_environment,
    _run_and_tee,
    _single_microbatch_probe_arguments,
    _training_batch_policy,
)


class ObjectBatchAutotunePolicyTest(unittest.TestCase):
    def test_probe_replaces_but_launch_preserves_global_batch_policy(self) -> None:
        launch = [
            "/data",
            "--phase",
            "B",
            "--global-object-batch=32",
            "--steps",
            "50000",
        ]
        probe = _single_microbatch_probe_arguments(launch)
        self.assertIn("--global-object-batch=32", launch)
        self.assertNotIn("--global-object-batch=32", probe)
        self.assertEqual(probe[-2:], ["--gradient-accumulation-steps", "1"])
        self.assertEqual(
            _training_batch_policy(launch),
            ("--global-object-batch", 32),
        )

    def test_exact_global_batch_rejects_only_incompatible_candidates(self) -> None:
        policy = ("--global-object-batch", 32)
        self.assertIsNone(_candidate_policy_failure(policy, 4, 4))
        self.assertIsNotNone(_candidate_policy_failure(policy, 4, 3))
        self.assertIsNotNone(_candidate_policy_failure(policy, 4, 16))

    def test_direct_torchrun_environment_removes_deprecated_nccl_name(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"NCCL_ASYNC_ERROR_HANDLING": "0"},
            clear=True,
        ):
            environment, selected = _gpu_environment("4,5,6,7")
        self.assertEqual(selected, "4,5,6,7")
        self.assertNotIn("NCCL_ASYNC_ERROR_HANDLING", environment)
        self.assertEqual(environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"], "0")
        self.assertEqual(environment["TORCH_NCCL_DUMP_ON_TIMEOUT"], "1")
        self.assertEqual(environment["OMP_NUM_THREADS"], "1")


class ObjectBatchAutotuneProcessTest(unittest.TestCase):
    def test_first_oom_terminates_candidate_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "run.log"
            started = time.monotonic()
            return_code, oom, timed_out = _run_and_tee(
                [
                    sys.executable,
                    "-c",
                    (
                        "import time; "
                        "print('torch.OutOfMemoryError: CUDA out of memory', flush=True); "
                        "time.sleep(30)"
                    ),
                ],
                log_path,
                dict(os.environ),
                timeout_seconds=10.0,
                termination_grace_seconds=1.0,
            )
            elapsed = time.monotonic() - started
            self.assertNotEqual(return_code, 0)
            self.assertTrue(oom)
            self.assertFalse(timed_out)
            self.assertLess(elapsed, 5.0)
            self.assertIn("SIGTERM_PROCESS_GROUP", log_path.read_text())

    def test_silent_candidate_is_bounded_by_wall_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "run.log"
            started = time.monotonic()
            return_code, oom, timed_out = _run_and_tee(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                log_path,
                dict(os.environ),
                timeout_seconds=0.25,
                termination_grace_seconds=1.0,
            )
            elapsed = time.monotonic() - started
            self.assertNotEqual(return_code, 0)
            self.assertFalse(oom)
            self.assertTrue(timed_out)
            self.assertLess(elapsed, 5.0)
            self.assertIn("probe_timeout", log_path.read_text())


if __name__ == "__main__":
    unittest.main()

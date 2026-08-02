"""Process-isolation and batch-policy regression tests for the GPU tuner."""

from __future__ import annotations

import os
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

from scripts.autotune_object_batch import (
    SemanticProgressMonitor,
    _candidate_policy_failure,
    _gpu_environment,
    _probe_outcome,
    _run_and_tee,
    _runtime_policy_from_training_arguments,
    _single_microbatch_probe_arguments,
    _training_batch_policy,
)


class ObjectBatchAutotunePolicyTest(unittest.TestCase):
    def test_default_training_config_controls_semantic_probe_policy(self) -> None:
        watchdog, autotune = _runtime_policy_from_training_arguments(
            ["/data", "--phase", "B", "--steps", "10"]
        )
        self.assertEqual(watchdog["policy"], "semantic_progress")
        self.assertEqual(watchdog["bootstrap_timeout_seconds"], 1800)
        self.assertEqual(autotune["warmup_optimizer_steps"], 1)
        self.assertEqual(autotune["measurement_optimizer_steps"], 2)
        self.assertEqual(autotune["probe_trellis_cache_scope"], "candidate")

    def test_completed_stages_expand_budget_with_robust_q99_mad(self) -> None:
        monitor = SemanticProgressMonitor(
            started=0.0,
            bootstrap_timeout_seconds=10.0,
            no_progress_timeout_seconds=1.0,
            stage_timeout_seconds={"forward.test": 0.1},
            stage_latency_mad_multiplier=2.0,
        )
        sequence = 0
        now = 0.0
        for duration in (1.0, 2.0, 10.0):
            sequence += 1
            begin = {
                "schema": "graft-gs-progress-v1",
                "rank": 0,
                "sequence": sequence,
                "semantic_sequence": sequence,
                "semantic_progress": True,
                "stage": "forward.test",
                "event": "begin",
            }
            monitor.observe(
                "GRAFT_GS_PROGRESS " + json.dumps(begin),
                now,
            )
            sequence += 1
            end = {
                **begin,
                "sequence": sequence,
                "semantic_sequence": sequence,
                "event": "end",
            }
            monitor.observe(
                "GRAFT_GS_PROGRESS " + json.dumps(end),
                now + duration,
            )
            now += duration + 1.0
        self.assertEqual(monitor._stage_budget("forward.test"), 12.0)

    def test_candidate_outcomes_keep_failures_distinct(self) -> None:
        self.assertEqual(
            _probe_outcome(
                return_code=0,
                oom=False,
                timed_out=False,
                failure_class=None,
            ),
            "SUCCESS",
        )
        self.assertEqual(
            _probe_outcome(
                return_code=-15,
                oom=False,
                timed_out=False,
                failure_class="process.signal_failure",
            ),
            "EXTERNAL_TERMINATION",
        )
        self.assertEqual(
            _probe_outcome(
                return_code=1,
                oom=False,
                timed_out=True,
                failure_class="liveness.no_semantic_progress",
            ),
            "NO_PROGRESS",
        )

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

    def test_semantic_progress_allows_runtime_beyond_bootstrap_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "run.log"
            child = (
                "import json,time\n"
                "for i in range(8):\n"
                " p={'schema':'graft-gs-progress-v1','rank':0,'sequence':i+1,"
                "'semantic_sequence':i+1,'semantic_progress':True,"
                "'stage':'forward.test','event':'advance'}\n"
                " print('GRAFT_GS_PROGRESS '+json.dumps(p),flush=True)\n"
                " time.sleep(0.08)\n"
            )
            started = time.monotonic()
            return_code, oom, timed_out = _run_and_tee(
                [sys.executable, "-c", child],
                log_path,
                dict(os.environ),
                timeout_seconds=0.15,
                no_progress_timeout_seconds=0.2,
                termination_grace_seconds=1.0,
                world_size=1,
            )
            self.assertEqual(return_code, 0)
            self.assertFalse(oom)
            self.assertFalse(timed_out)
            self.assertGreater(time.monotonic() - started, 0.5)

    def test_heartbeats_do_not_mask_missing_semantic_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "run.log"
            child = (
                "import json,time\n"
                "for i in range(20):\n"
                " p={'schema':'graft-gs-progress-v1','rank':0,'sequence':i+1,"
                "'semantic_sequence':1,'semantic_progress':i==0,"
                "'stage':'forward.test','event':'begin' if i==0 else 'heartbeat'}\n"
                " print('GRAFT_GS_PROGRESS '+json.dumps(p),flush=True)\n"
                " time.sleep(0.05)\n"
            )
            return_code, oom, timed_out = _run_and_tee(
                [sys.executable, "-c", child],
                log_path,
                dict(os.environ),
                timeout_seconds=0.2,
                no_progress_timeout_seconds=0.2,
                termination_grace_seconds=1.0,
                world_size=1,
            )
            self.assertNotEqual(return_code, 0)
            self.assertFalse(oom)
            self.assertTrue(timed_out)
            timeout_lines = [
                line
                for line in log_path.read_text().splitlines()
                if line.startswith("GRAFT_GS_AUTOTUNE_PROGRESS_TIMEOUT ")
            ]
            self.assertEqual(len(timeout_lines), 1)
            payload = json.loads(timeout_lines[0].split(" ", 1)[1])
            self.assertEqual(payload["ranks"][0]["event"], "heartbeat")


if __name__ == "__main__":
    unittest.main()

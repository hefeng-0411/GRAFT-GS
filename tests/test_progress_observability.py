"""Regression tests for the semantic progress control-plane contract."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import time
import unittest

from graft_gs.observability import (
    PROGRESS_PREFIX,
    ProgressConfig,
    ProgressReporter,
    TrainingProfilerConfig,
)


class ProgressReporterTest(unittest.TestCase):
    def test_profiler_schedule_is_bounded_to_declared_steps(self) -> None:
        config = TrainingProfilerConfig(
            first_n_steps=5,
            wait_steps=1,
            warmup_steps=1,
        )
        self.assertEqual(config.active_steps, 3)
        with self.assertRaises(ValueError):
            TrainingProfilerConfig(
                first_n_steps=2,
                wait_steps=1,
                warmup_steps=1,
            )

    def test_stage_records_are_semantic_but_heartbeats_are_not(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            reporter = ProgressReporter(
                ProgressConfig(
                    heartbeat_interval_seconds=0.02,
                    include_cuda_memory=False,
                    profiler_ranges=False,
                ),
                rank=2,
                local_rank=1,
                world_size=4,
                device="cpu",
            )
            reporter.set_context(global_step=7, object_ids=["object-a"])
            with reporter.stage("forward.test", sparse_tokens=13):
                time.sleep(0.06)
            reporter.close()
        payloads = [
            json.loads(line[len(PROGRESS_PREFIX) :])
            for line in stream.getvalue().splitlines()
            if line.startswith(PROGRESS_PREFIX)
        ]
        self.assertGreaterEqual(len(payloads), 4)
        self.assertTrue(
            any(
                value["stage"] == "forward.test"
                and value["event"] == "begin"
                and value["semantic_progress"] is True
                for value in payloads
            )
        )
        self.assertTrue(
            any(
                value["event"] == "heartbeat"
                and value["semantic_progress"] is False
                for value in payloads
            )
        )
        sequences = [int(value["sequence"]) for value in payloads]
        self.assertEqual(sequences, sorted(set(sequences)))
        self.assertTrue(all(value["rank"] == 2 for value in payloads))


if __name__ == "__main__":
    unittest.main()

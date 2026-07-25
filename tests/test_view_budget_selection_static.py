"""PyTorch-independent tests for measured A800 view-budget selection."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_a800_view_budget", ROOT / "scripts" / "select_a800_view_budget.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SWEEP_SPEC = importlib.util.spec_from_file_location(
    "sweep_a800_view_budget", ROOT / "scripts" / "sweep_a800_view_budget.py"
)
assert SWEEP_SPEC is not None and SWEEP_SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SWEEP_SPEC)
SWEEP_SPEC.loader.exec_module(SWEEP)


def report(views: int, throughput: float, reserved: float) -> dict[str, object]:
    return {
        "world_size": 2,
        "evaluation_execution_stage": "atlas_autoencoding",
        "losses": [2.0, 1.9],
        "trellis_prior": {
            "enabled": True,
            "source_rank_only": True,
            "offload_cuda_pipeline_after_sampling": True,
            "maximum_conditioning_views": 16,
        },
        "rank_performance": [
            {
                "rank": rank,
                "local_views": views,
                "local_views_per_second": throughput,
                "peak_reserved_fraction": reserved,
                "peak_allocated_fraction": max(reserved - 0.10, 0.0),
                "peak_active_fraction": max(reserved - 0.10, 0.0),
                "ending_reserved_fraction": reserved,
                "ending_driver_free_fraction": 1.0 - reserved,
                "trellis_cache_release_performed": 1.0 if rank == 0 else 0.0,
                "trellis_pipeline_offload_performed": (
                    1.0 if rank == 0 else 0.0
                ),
                "trellis_pipeline_peak_allocated_before_bytes": (
                    1_000_000.0 if rank == 0 else 0.0
                ),
                "trellis_conditioning_views_available": (
                    float(2 * views) if rank == 0 else 0.0
                ),
                "trellis_conditioning_views_selected": (
                    float(min(16, 2 * views)) if rank == 0 else 0.0
                ),
                "cuda_stage_memory": {
                    stage: {
                        "allocated_fraction": max(reserved - 0.10, 0.0),
                        "reserved_fraction": reserved,
                        "driver_free_fraction": 1.0 - reserved,
                        "non_allocator_used_fraction": 0.0,
                    }
                    for stage in (
                        "input",
                        "trellis_prefetch",
                        "vggt_geometry",
                        "evidence_lift",
                        "trainer/before_backward",
                        "trainer/after_backward",
                    )
                },
            }
            for rank in range(2)
        ],
        "transport": {
            "converged": True,
            "fixed_point_residual": 1.0e-8,
            "effective_tolerance": 1.0e-7,
            "minimum_source_transport_mass": 0.1,
            "minimum_target_transport_mass": 0.1,
            "internal_minimum_log_plan": -20.0,
            "internal_solve_dtype": "float64",
            "storage_underflow_edges": 0,
            "storage_zero_source_rows": 0,
            "storage_zero_target_columns": 0,
            "storage_underflow_mass_fraction": 0.0,
            "storage_zero_source_mass_fraction": 0.0,
            "storage_zero_target_mass_fraction": 0.0,
            "storage_relative_l1_error": 1.0e-8,
            "edge_count": 100,
            "source_count": 10,
            "target_count": 20,
        },
        "final_feasibility": {
            "feasible": True,
            **{name: 0.01 for name in MODULE.FEASIBILITY_FIELDS},
        },
        "selected_topology": {
            "identifier": "support-endpoint-occ-0.0010-b1-0-0",
            "betti": [1, 0, 0],
            "persistence_cardinality": [[4, 4], [2, 2], [0, 0]],
            "persistence_matching_mode": ["exact", "exact", "exact"],
        },
        "rendering": {
            "backend": "cuda",
            "checkpoint_views": True,
        },
    }


class ViewBudgetSelectionTest(unittest.TestCase):
    def test_rejects_stale_report_that_executed_post_phase_b_flow(self) -> None:
        candidate = report(16, 1.0, 0.7)
        candidate["evaluation_execution_stage"] = "full"
        audited = MODULE.audit_report(candidate, 0.85)
        self.assertFalse(audited["admissible"])
        self.assertTrue(
            any("Phase-B" in reason for reason in audited["reasons"])
        )

    def test_visible_cuda_inventory_is_json_typed_and_rank_checked(self) -> None:
        payload = (
            '[{"free_bytes": 79000000000, "free_fraction": 0.9875, '
            '"logical_device": 0, "name": "NVIDIA A800", '
            '"total_bytes": 80000000000}]'
        )
        with (
            mock.patch.object(
                SWEEP.subprocess,
                "run",
                return_value=SimpleNamespace(stdout=payload),
            ),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            inventory = SWEEP.visible_cuda_inventory("/pinned/python")
        self.assertEqual(len(inventory), 1)
        self.assertEqual(inventory[0]["logical_device"], 0)
        with (
            mock.patch.object(
                SWEEP.subprocess,
                "run",
                return_value=SimpleNamespace(stdout=payload),
            ),
            mock.patch.dict(
                os.environ,
                {"GRAFT_GS_NPROC_PER_NODE": "2"},
                clear=True,
            ),
            self.assertRaisesRegex(RuntimeError, "does not match"),
        ):
            SWEEP.visible_cuda_inventory("/pinned/python")

    def test_sweep_command_is_complete_and_oom_detection_is_fail_closed(self) -> None:
        candidates = SWEEP.validate_candidates((16, 24, 32, 48, 64))
        self.assertEqual(candidates, (16, 24, 32, 48, 64))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            SWEEP.validate_candidates((16, 32, 24))
        self.assertTrue(
            SWEEP.log_reports_oom("RuntimeError: CUDA error: out of memory")
        )
        self.assertTrue(
            SWEEP.log_reports_oom("CUBLAS_STATUS_ALLOC_FAILED")
        )
        self.assertFalse(
            SWEEP.log_reports_oom("NCCL watchdog timed out")
        )
        command = SWEEP.build_overfit_command(
            python="/pinned/python",
            nproc_per_node=2,
            dataset_root=Path("/dataset"),
            manifest=Path("/manifest.jsonl"),
            object_id="object",
            config=Path("configs/graft_gs_a800_native.yaml"),
            vggt_checkpoint="vggt",
            trellis_checkpoint="trellis",
            views_per_rank=32,
            evaluation_views=24,
            steps=3,
            minimum_relative_improvement=-1.0,
            output=Path("/output"),
        )
        for option in (
            "--views-per-rank",
            "--evaluation-views",
            "--steps",
            "--minimum-relative-improvement",
            "--output",
        ):
            self.assertIn(option, command)
        source = (ROOT / "scripts" / "sweep_a800_view_budget.py").read_text(
            encoding="utf8"
        )
        self.assertIn("if oom and args.stop_after_oom:", source)
        self.assertNotIn("if oom and not args.continue_after_oom:", source)

    def test_empty_constraint_family_positive_infinity_is_admissible(self) -> None:
        value = report(16, 10.0, 0.3)
        value["final_feasibility"]["minimum_separation_margin"] = float("inf")
        self.assertTrue(MODULE.audit_report(value, 0.85)["admissible"])

    def test_selects_largest_safe_near_optimal_throughput_budget(self) -> None:
        candidates = []
        for views, speed, reserved in (
            (16, 10.0, 0.25),
            (32, 9.8, 0.45),
            (48, 8.0, 0.65),
            (64, 10.2, 0.91),
        ):
            candidate = MODULE.audit_report(report(views, speed, reserved), 0.85)
            candidate["views"] = views
            candidates.append(candidate)
        selected = MODULE.select_candidate(candidates, 0.97)
        self.assertEqual(selected["views"], 32)
        self.assertFalse(candidates[-1]["admissible"])

    def test_rejects_unconverged_or_infeasible_scientific_state(self) -> None:
        invalid = report(32, 10.0, 0.5)
        invalid["transport"]["converged"] = False
        invalid["final_feasibility"]["minimum_separation_margin"] = 0.0
        candidate = MODULE.audit_report(invalid, 0.85)
        self.assertFalse(candidate["admissible"])
        self.assertIn("sparse transport is not certified converged", candidate["reasons"])
        with self.assertRaisesRegex(RuntimeError, "no concurrency candidate"):
            MODULE.select_candidate([candidate], 0.97)

    def test_edge_underflow_count_is_diagnostic_when_discarded_mass_is_tiny(self) -> None:
        valid = report(32, 10.0, 0.5)
        valid["transport"]["storage_underflow_edges"] = 80
        valid["transport"]["storage_zero_source_rows"] = 5
        valid["transport"]["storage_underflow_mass_fraction"] = 1.0e-30
        valid["transport"]["storage_zero_source_mass_fraction"] = 1.0e-30
        candidate = MODULE.audit_report(valid, 0.85)
        self.assertTrue(candidate["admissible"])
        self.assertEqual(
            candidate["transport_storage_underflow_edge_fraction"],
            0.8,
        )

    def test_rejects_excessive_acknowledged_storage_mass_error(self) -> None:
        invalid = report(32, 10.0, 0.5)
        invalid["transport"]["storage_underflow_edges"] = 80
        invalid["transport"]["storage_underflow_mass_fraction"] = 1.0e-4
        invalid["transport"]["storage_relative_l1_error"] = 1.0e-4
        invalid["transport"]["storage_zero_source_mass_fraction"] = 1.0e-5
        candidate = MODULE.audit_report(invalid, 0.85)
        self.assertFalse(candidate["admissible"])
        self.assertIn(
            "transport FP32 storage relative-L1 error exceeds the configured accuracy limit",
            candidate["reasons"],
        )
        self.assertIn(
            "transport zero-marginal discarded mass exceeds the configured accuracy limit",
            candidate["reasons"],
        )

    def test_rejects_stale_allocator_proxy_report(self) -> None:
        invalid = report(24, 10.0, 0.4)
        del invalid["rank_performance"][0]["ending_reserved_fraction"]
        candidate = MODULE.audit_report(invalid, 0.85)
        self.assertFalse(candidate["admissible"])
        self.assertIn(
            "rank_performance row lacks numeric telemetry",
            candidate["reasons"],
        )

    def test_rejects_missing_or_violated_trellis_memory_policy(self) -> None:
        missing = report(16, 10.0, 0.3)
        missing.pop("trellis_prior")
        candidate = MODULE.audit_report(missing, 0.85)
        self.assertFalse(candidate["admissible"])
        self.assertTrue(
            any("memory policy" in reason for reason in candidate["reasons"])
        )
        exceeded = report(16, 10.0, 0.3)
        exceeded["rank_performance"][0][
            "trellis_conditioning_views_selected"
        ] = 17.0
        candidate = MODULE.audit_report(exceeded, 0.85)
        self.assertFalse(candidate["admissible"])
        self.assertTrue(
            any(
                "conditioning-view counts" in reason
                for reason in candidate["reasons"]
            )
        )

    def test_historical_peak_reserved_is_diagnostic_after_cache_release(self) -> None:
        value = report(24, 10.0, 0.4)
        value["rank_performance"][0]["peak_reserved_fraction"] = 0.99
        candidate = MODULE.audit_report(value, 0.85)
        self.assertTrue(candidate["admissible"])
        self.assertEqual(candidate["maximum_peak_reserved_fraction"], 0.99)

    def test_rejects_live_peak_and_missing_source_rank_cache_release(self) -> None:
        invalid = report(24, 10.0, 0.4)
        invalid["rank_performance"][0]["peak_allocated_fraction"] = 0.95
        invalid["rank_performance"][0]["peak_active_fraction"] = 0.96
        invalid["rank_performance"][0]["trellis_cache_release_performed"] = 0.0
        invalid["rank_performance"][0]["trellis_pipeline_offload_performed"] = 0.0
        candidate = MODULE.audit_report(
            invalid,
            0.85,
            maximum_allocated_fraction=0.90,
        )
        self.assertFalse(candidate["admissible"])
        self.assertIn(
            "rank 0 did not certify release of inactive frozen-TRELLIS allocator blocks",
            candidate["reasons"],
        )
        self.assertTrue(
            any("live peak fraction" in reason for reason in candidate["reasons"])
        )
        self.assertTrue(
            any("offload of frozen TRELLIS" in reason for reason in candidate["reasons"])
        )

    def test_rejects_stale_report_without_scalable_persistence_certificate(self) -> None:
        invalid = report(24, 10.0, 0.4)
        del invalid["selected_topology"]["persistence_matching_mode"]
        candidate = MODULE.audit_report(invalid, 0.85)
        self.assertFalse(candidate["admissible"])
        self.assertIn(
            "persistence matching modes are invalid or missing",
            candidate["reasons"],
        )

    def test_rejects_stale_report_without_renderer_memory_certificate(self) -> None:
        invalid = report(24, 10.0, 0.4)
        del invalid["rendering"]
        candidate = MODULE.audit_report(invalid, 0.85)
        self.assertFalse(candidate["admissible"])
        self.assertIn(
            "renderer memory-policy certificate is missing",
            candidate["reasons"],
        )


if __name__ == "__main__":
    unittest.main()

"""PyTorch-independent tests for measured A800 view-budget selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_a800_view_budget", ROOT / "scripts" / "select_a800_view_budget.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report(views: int, throughput: float, reserved: float) -> dict[str, object]:
    return {
        "world_size": 2,
        "losses": [2.0, 1.9],
        "rank_performance": [
            {
                "rank": rank,
                "local_views": views,
                "local_views_per_second": throughput,
                "peak_reserved_fraction": reserved,
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
            "edge_count": 100,
            "source_count": 10,
            "target_count": 20,
        },
        "final_feasibility": {
            "feasible": True,
            **{name: 0.01 for name in MODULE.FEASIBILITY_FIELDS},
        },
    }


class ViewBudgetSelectionTest(unittest.TestCase):
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

    def test_rejects_excessive_acknowledged_storage_underflow(self) -> None:
        invalid = report(32, 10.0, 0.5)
        invalid["transport"]["storage_underflow_edges"] = 8
        invalid["transport"]["storage_zero_source_rows"] = 1
        candidate = MODULE.audit_report(
            invalid,
            0.85,
            maximum_storage_underflow_fraction=0.05,
            maximum_zero_marginal_fraction=0.05,
        )
        self.assertFalse(candidate["admissible"])
        self.assertIn(
            "transport storage-underflow fraction exceeds the configured accuracy limit",
            candidate["reasons"],
        )
        self.assertIn(
            "transport zero-marginal fraction exceeds the configured accuracy limit",
            candidate["reasons"],
        )


if __name__ == "__main__":
    unittest.main()

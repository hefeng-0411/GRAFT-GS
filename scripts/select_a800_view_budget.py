"""Select a useful A800 multiview budget from measured overfit reports.

The selector deliberately treats VRAM as a feasibility constraint rather than
an objective.  A candidate is admissible only when every rank is represented,
all losses are finite, sparse transport converged, the final atlas embedding is
strictly feasible, and peak reserved memory remains below the requested limit.
Among admissible runs within ``throughput_fraction`` of the fastest aggregate
views/second, it chooses the largest measured per-rank view count to favor
multiview coverage without accepting a severe throughput regression.
"""

from __future__ import annotations

import argparse
import glob
import json
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence


FEASIBILITY_FIELDS = (
    "minimum_area_margin",
    "minimum_orientation_margin",
    "minimum_separation_margin",
    "minimum_covariance_margin",
    "maximum_covariance_margin",
)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(float(value))


def _positive_margin(value: object) -> bool:
    # ``+inf`` is the exact report convention for an empty constraint family
    # (for example, no nonlocal collision pair). It is a valid positive margin;
    # NaN, -inf, zero, and negative margins are not.
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) > 0
    )


def audit_report(
    report: Mapping[str, object],
    maximum_reserved_fraction: float,
) -> dict[str, object]:
    """Return a normalized candidate record with explicit rejection reasons."""

    reasons: list[str] = []
    world_size = int(report.get("world_size", 0))
    rank_rows = report.get("rank_performance")
    if world_size < 1:
        reasons.append("world_size is not positive")
    if not isinstance(rank_rows, list) or len(rank_rows) != world_size:
        reasons.append("rank_performance does not contain exactly one row per rank")
        rank_rows = []

    ranks: list[int] = []
    local_views: list[int] = []
    throughput: list[float] = []
    reserved: list[float] = []
    for row in rank_rows:
        if not isinstance(row, Mapping):
            reasons.append("rank_performance contains a malformed row")
            continue
        try:
            rank = int(row["rank"])
            views = int(float(row["local_views"]))
            views_per_second = float(row["local_views_per_second"])
            reserved_fraction = float(row["peak_reserved_fraction"])
        except (KeyError, TypeError, ValueError):
            reasons.append("rank_performance row lacks numeric telemetry")
            continue
        if views < 2:
            reasons.append(f"rank {rank} has fewer than two useful views")
        if not isfinite(views_per_second) or views_per_second <= 0:
            reasons.append(f"rank {rank} has invalid useful throughput")
        if not isfinite(reserved_fraction) or not 0 <= reserved_fraction <= 1:
            reasons.append(f"rank {rank} has invalid reserved-memory fraction")
        elif reserved_fraction > maximum_reserved_fraction:
            reasons.append(
                f"rank {rank} reserved fraction {reserved_fraction:.4f} exceeds "
                f"limit {maximum_reserved_fraction:.4f}"
            )
        ranks.append(rank)
        local_views.append(views)
        throughput.append(views_per_second)
        reserved.append(reserved_fraction)
    if ranks and sorted(ranks) != list(range(world_size)):
        reasons.append("rank identities are missing or duplicated")

    losses = report.get("losses")
    if not isinstance(losses, list) or not losses or not all(_finite_number(value) for value in losses):
        reasons.append("loss history is empty or non-finite")

    transport = report.get("transport")
    if not isinstance(transport, Mapping):
        reasons.append("transport certificate is missing")
    else:
        if transport.get("converged") is not True:
            reasons.append("sparse transport is not certified converged")
        for name in (
            "fixed_point_residual",
            "effective_tolerance",
            "minimum_source_transport_mass",
            "minimum_target_transport_mass",
        ):
            value = transport.get(name)
            if not _finite_number(value):
                reasons.append(f"transport field {name} is non-finite or missing")
        if _finite_number(transport.get("fixed_point_residual")) and _finite_number(
            transport.get("effective_tolerance")
        ) and float(transport["fixed_point_residual"]) > float(transport["effective_tolerance"]):
            reasons.append("transport residual exceeds its effective tolerance")
        for name in (
            "minimum_source_transport_mass",
            "minimum_target_transport_mass",
        ):
            if _finite_number(transport.get(name)) and float(transport[name]) <= 0:
                reasons.append(f"transport field {name} is not positive")

    feasibility = report.get("final_feasibility")
    if not isinstance(feasibility, Mapping):
        reasons.append("final feasibility certificate is missing")
    else:
        if feasibility.get("feasible") is not True:
            reasons.append("final embedding is not certified feasible")
        for name in FEASIBILITY_FIELDS:
            value = feasibility.get(name)
            if not _positive_margin(value):
                reasons.append(f"final feasibility field {name} is not positive")

    return {
        "admissible": not reasons,
        "reasons": sorted(set(reasons)),
        "world_size": world_size,
        "minimum_views_per_rank": min(local_views) if local_views else 0,
        "maximum_views_per_rank": max(local_views) if local_views else 0,
        "global_useful_views": sum(local_views),
        "aggregate_views_per_second": sum(throughput),
        "maximum_reserved_fraction": max(reserved) if reserved else float("inf"),
    }


def select_candidate(
    candidates: Sequence[Mapping[str, object]],
    throughput_fraction: float,
) -> Mapping[str, object]:
    if not 0 < throughput_fraction <= 1:
        raise ValueError("throughput_fraction must lie in (0,1]")
    admissible = [candidate for candidate in candidates if candidate.get("admissible") is True]
    if not admissible:
        raise RuntimeError("no concurrency candidate satisfies the scientific and memory gates")
    fastest = max(float(candidate["aggregate_views_per_second"]) for candidate in admissible)
    competitive = [
        candidate
        for candidate in admissible
        if float(candidate["aggregate_views_per_second"]) >= throughput_fraction * fastest
    ]
    return max(
        competitive,
        key=lambda candidate: (
            int(candidate["minimum_views_per_rank"]),
            float(candidate["aggregate_views_per_second"]),
            -float(candidate["maximum_reserved_fraction"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", help="JSON files or glob patterns")
    parser.add_argument("--maximum-reserved-fraction", type=float, default=0.85)
    parser.add_argument("--throughput-fraction", type=float, default=0.97)
    parser.add_argument("--output", type=Path, default=Path("outputs/concurrency/selection.json"))
    args = parser.parse_args()
    if not 0 < args.maximum_reserved_fraction < 1:
        raise ValueError("maximum-reserved-fraction must lie in (0,1)")

    paths = sorted(
        {
            Path(match)
            for pattern in args.reports
            for match in glob.glob(pattern, recursive=True)
        }
    )
    if not paths:
        raise FileNotFoundError("no view-budget reports matched the supplied paths")
    audited: list[dict[str, object]] = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"report root must be a mapping: {path}")
        candidate = audit_report(value, args.maximum_reserved_fraction)
        candidate["path"] = str(path)
        audited.append(candidate)
    selected = select_candidate(audited, args.throughput_fraction)
    output = {
        "schema": "graft-gs-a800-view-selection-v1",
        "maximum_reserved_fraction": args.maximum_reserved_fraction,
        "throughput_fraction": args.throughput_fraction,
        "candidates": audited,
        "selected": selected,
        "recommended_views_per_rank": selected["minimum_views_per_rank"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

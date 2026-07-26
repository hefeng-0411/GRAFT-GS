"""Select a useful A800 multiview budget from measured overfit reports.

The selector deliberately treats VRAM as a feasibility constraint rather than
an objective.  A candidate is admissible only when every rank is represented,
all losses are finite, sparse transport converged, the final atlas embedding is
strictly feasible, the live peak remains below the requested limit, and the
post-step caching-allocator state leaves explicit driver headroom.

``peak_reserved`` is retained as a diagnostic but is not a live-memory error
bound: PyTorch may reserve inactive blocks after a frozen upstream sampler.
Similarly, the count of FP32-zero OT edges is not a transport error bound.
Admission uses discarded transport mass and relative L1 storage error computed
against the FP64 log-domain solve.
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
    if value == "positive_infinity":
        return True
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) > 0
    )


def audit_report(
    report: Mapping[str, object],
    maximum_reserved_fraction: float,
    maximum_allocated_fraction: float = 0.90,
    minimum_driver_free_fraction: float = 0.05,
    maximum_storage_relative_l1_error: float = 1.0e-6,
    maximum_zero_marginal_mass_fraction: float = 1.0e-12,
) -> dict[str, object]:
    """Return a normalized candidate record with explicit rejection reasons."""

    reasons: list[str] = []
    if report.get("evaluation_execution_stage") != "atlas_autoencoding":
        reasons.append(
            "view-budget evaluation did not stop at the Phase-B "
            "atlas-autoencoding boundary"
        )
    world_size = int(report.get("world_size", 0))
    rank_rows = report.get("rank_performance")
    if world_size < 1:
        reasons.append("world_size is not positive")
    if not isinstance(rank_rows, list) or len(rank_rows) != world_size:
        reasons.append("rank_performance does not contain exactly one row per rank")
        rank_rows = []
    prior_policy = report.get("trellis_prior")
    configured_prior_view_cap = 0
    if not isinstance(prior_policy, Mapping):
        reasons.append("TRELLIS memory policy certificate is missing")
    else:
        configured_cap = prior_policy.get("maximum_conditioning_views")
        if (
            not isinstance(configured_cap, int)
            or isinstance(configured_cap, bool)
            or configured_cap < 1
        ):
            reasons.append("TRELLIS conditioning-view cap is invalid")
        else:
            configured_prior_view_cap = configured_cap
        if prior_policy.get("enabled") is not True:
            reasons.append("view-budget run did not enable the TRELLIS prior")
        if world_size > 1 and prior_policy.get("source_rank_only") is not True:
            reasons.append("same-object DDP TRELLIS ownership is not source-only")
        if (
            prior_policy.get("offload_cuda_pipeline_after_sampling")
            is not True
        ):
            reasons.append("TRELLIS CUDA pipeline offload policy was not active")

    ranks: list[int] = []
    local_views: list[int] = []
    throughput: list[float] = []
    peak_reserved: list[float] = []
    peak_allocated: list[float] = []
    peak_active: list[float] = []
    ending_reserved: list[float] = []
    ending_driver_free: list[float] = []
    conditioning_views: list[int] = []
    for row in rank_rows:
        if not isinstance(row, Mapping):
            reasons.append("rank_performance contains a malformed row")
            continue
        try:
            rank = int(row["rank"])
            views = int(float(row["local_views"]))
            views_per_second = float(row["local_views_per_second"])
            peak_reserved_fraction = float(row["peak_reserved_fraction"])
            peak_allocated_fraction = float(row["peak_allocated_fraction"])
            peak_active_fraction = float(row["peak_active_fraction"])
            ending_reserved_fraction = float(row["ending_reserved_fraction"])
            ending_driver_free_fraction = float(row["ending_driver_free_fraction"])
            cache_release_value = float(row["trellis_cache_release_performed"])
            pipeline_offload_value = float(
                row["trellis_pipeline_offload_performed"]
            )
            pipeline_prior_peak = float(
                row["trellis_pipeline_peak_allocated_before_bytes"]
            )
            available_prior_view_value = float(
                row["trellis_conditioning_views_available"]
            )
            selected_prior_view_value = float(
                row["trellis_conditioning_views_selected"]
            )
            if (
                not isfinite(pipeline_prior_peak)
                or pipeline_prior_peak < 0
                or not isfinite(available_prior_view_value)
                or not available_prior_view_value.is_integer()
                or not isfinite(selected_prior_view_value)
                or not selected_prior_view_value.is_integer()
            ):
                raise ValueError("TRELLIS lifetime telemetry is invalid")
            available_prior_views = int(available_prior_view_value)
            selected_prior_views = int(selected_prior_view_value)
            if cache_release_value not in (0.0, 1.0):
                raise ValueError("cache-release flag is not Boolean")
            if pipeline_offload_value not in (0.0, 1.0):
                raise ValueError("pipeline-offload flag is not Boolean")
            cache_release_performed = bool(cache_release_value)
            pipeline_offload_performed = bool(pipeline_offload_value)
        except (KeyError, TypeError, ValueError):
            reasons.append("rank_performance row lacks numeric telemetry")
            continue
        stage_memory = row.get("cuda_stage_memory")
        if not isinstance(stage_memory, Mapping):
            reasons.append(f"rank {rank} lacks CUDA stage-memory telemetry")
        else:
            for stage in (
                "input",
                "trellis_prefetch",
                "vggt_geometry",
                "evidence_lift",
                "trainer/before_backward",
                "trainer/after_backward",
            ):
                stage_value = stage_memory.get(stage)
                if not isinstance(stage_value, Mapping):
                    reasons.append(
                        f"rank {rank} lacks CUDA memory stage {stage}"
                    )
                    continue
                for field in (
                    "allocated_fraction",
                    "reserved_fraction",
                    "driver_free_fraction",
                    "non_allocator_used_fraction",
                ):
                    value = stage_value.get(field)
                    if (
                        not _finite_number(value)
                        or not 0 <= float(value) <= 1
                    ):
                        reasons.append(
                            f"rank {rank} stage {stage} has invalid {field}"
                        )
        if views < 2:
            reasons.append(f"rank {rank} has fewer than two useful views")
        if not isfinite(views_per_second) or views_per_second <= 0:
            reasons.append(f"rank {rank} has invalid useful throughput")
        for label, fraction in (
            ("peak reserved", peak_reserved_fraction),
            ("peak allocated", peak_allocated_fraction),
            ("peak active", peak_active_fraction),
            ("ending reserved", ending_reserved_fraction),
            ("ending driver free", ending_driver_free_fraction),
        ):
            if not isfinite(fraction) or not 0 <= fraction <= 1:
                reasons.append(f"rank {rank} has invalid {label}-memory fraction")
        live_peak_fraction = max(peak_allocated_fraction, peak_active_fraction)
        if live_peak_fraction > maximum_allocated_fraction:
            reasons.append(
                f"rank {rank} live peak fraction "
                f"{live_peak_fraction:.4f} exceeds live-memory limit "
                f"{maximum_allocated_fraction:.4f}"
            )
        if ending_reserved_fraction > maximum_reserved_fraction:
            reasons.append(
                f"rank {rank} ending reserved fraction "
                f"{ending_reserved_fraction:.4f} exceeds allocator-state limit "
                f"{maximum_reserved_fraction:.4f}"
            )
        if ending_driver_free_fraction < minimum_driver_free_fraction:
            reasons.append(
                f"rank {rank} ending driver-free fraction "
                f"{ending_driver_free_fraction:.4f} is below headroom limit "
                f"{minimum_driver_free_fraction:.4f}"
            )
        if rank == 0 and not cache_release_performed:
            reasons.append(
                "rank 0 did not certify release of inactive frozen-TRELLIS "
                "allocator blocks"
            )
        if rank == 0 and not pipeline_offload_performed:
            reasons.append(
                "rank 0 did not certify offload of frozen TRELLIS checkpoint "
                "weights before the differentiable GRAFT-GS graph"
            )
        if rank == 0:
            if pipeline_prior_peak <= 0:
                reasons.append(
                    "rank 0 lacks the frozen TRELLIS lifetime peak certificate"
                )
            if (
                available_prior_views < 1
                or selected_prior_views < 1
                or selected_prior_views > available_prior_views
                or (
                    configured_prior_view_cap > 0
                    and selected_prior_views > configured_prior_view_cap
                )
            ):
                reasons.append(
                    "rank 0 TRELLIS conditioning-view counts are invalid"
                )
            conditioning_views.append(selected_prior_views)
        ranks.append(rank)
        local_views.append(views)
        throughput.append(views_per_second)
        peak_reserved.append(peak_reserved_fraction)
        peak_allocated.append(peak_allocated_fraction)
        peak_active.append(peak_active_fraction)
        ending_reserved.append(ending_reserved_fraction)
        ending_driver_free.append(ending_driver_free_fraction)
    if ranks and sorted(ranks) != list(range(world_size)):
        reasons.append("rank identities are missing or duplicated")

    losses = report.get("losses")
    if not isinstance(losses, list) or not losses or not all(_finite_number(value) for value in losses):
        reasons.append("loss history is empty or non-finite")

    transport = report.get("transport")
    underflow_edge_fraction = float("inf")
    underflow_mass_fraction = float("inf")
    zero_source_mass_fraction = float("inf")
    zero_target_mass_fraction = float("inf")
    storage_relative_l1_error = float("inf")
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
            "internal_minimum_log_plan",
        ):
            value = transport.get(name)
            if not _finite_number(value):
                reasons.append(f"transport field {name} is non-finite or missing")
        if _finite_number(transport.get("fixed_point_residual")) and _finite_number(
            transport.get("effective_tolerance")
        ) and float(transport["fixed_point_residual"]) > float(transport["effective_tolerance"]):
            reasons.append("transport residual exceeds its effective tolerance")
        for name in ("minimum_source_transport_mass", "minimum_target_transport_mass"):
            if _finite_number(transport.get(name)) and float(transport[name]) < 0:
                reasons.append(f"transport field {name} is negative")
        if transport.get("internal_solve_dtype") != "float64":
            reasons.append("sparse transport was not solved in float64/log space")
        for name in (
            "storage_underflow_mass_fraction",
            "storage_zero_source_mass_fraction",
            "storage_zero_target_mass_fraction",
            "storage_relative_l1_error",
        ):
            value = transport.get(name)
            if (
                not _finite_number(value)
                or not 0 <= float(value) <= 1
            ):
                reasons.append(f"transport error certificate {name} is invalid or missing")
        if all(
            _finite_number(transport.get(name))
            for name in (
                "storage_underflow_mass_fraction",
                "storage_zero_source_mass_fraction",
                "storage_zero_target_mass_fraction",
                "storage_relative_l1_error",
            )
        ):
            underflow_mass_fraction = float(
                transport["storage_underflow_mass_fraction"]
            )
            zero_source_mass_fraction = float(
                transport["storage_zero_source_mass_fraction"]
            )
            zero_target_mass_fraction = float(
                transport["storage_zero_target_mass_fraction"]
            )
            storage_relative_l1_error = float(
                transport["storage_relative_l1_error"]
            )
            if storage_relative_l1_error > maximum_storage_relative_l1_error:
                reasons.append(
                    "transport FP32 storage relative-L1 error exceeds the "
                    "configured accuracy limit"
                )
            if max(
                zero_source_mass_fraction,
                zero_target_mass_fraction,
            ) > maximum_zero_marginal_mass_fraction:
                reasons.append(
                    "transport zero-marginal discarded mass exceeds the "
                    "configured accuracy limit"
                )
        count_fields = (
            "storage_underflow_edges",
            "storage_zero_source_rows",
            "storage_zero_target_columns",
            "edge_count",
            "source_count",
            "target_count",
        )
        parsed_counts: dict[str, int] = {}
        for name in count_fields:
            value = transport.get(name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                reasons.append(f"transport count {name} is invalid or missing")
            else:
                parsed_counts[name] = value
        if all(name in parsed_counts for name in count_fields):
            if min(
                parsed_counts["edge_count"],
                parsed_counts["source_count"],
                parsed_counts["target_count"],
            ) <= 0:
                reasons.append("transport graph cardinalities are not positive")
            else:
                underflow_edge_fraction = (
                    parsed_counts["storage_underflow_edges"]
                    / parsed_counts["edge_count"]
                )

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

    topology = report.get("selected_topology")
    if not isinstance(topology, Mapping):
        reasons.append("selected topology certificate is missing")
    else:
        modes = topology.get("persistence_matching_mode")
        cardinality = topology.get("persistence_cardinality")
        if (
            not isinstance(modes, list)
            or len(modes) != 3
            or any(mode not in {"exact", "sliced"} for mode in modes)
        ):
            reasons.append("persistence matching modes are invalid or missing")
        if (
            not isinstance(cardinality, list)
            or len(cardinality) != 3
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    for value in pair
                )
                for pair in cardinality
            )
        ):
            reasons.append("persistence diagram cardinalities are invalid or missing")

    rendering = report.get("rendering")
    if not isinstance(rendering, Mapping):
        reasons.append("renderer memory-policy certificate is missing")
    else:
        if rendering.get("backend") != "cuda":
            reasons.append("view-budget report did not use the production CUDA renderer")
        if rendering.get("checkpoint_views") is not True:
            reasons.append("CUDA renderer per-view checkpointing was not active")

    return {
        "admissible": not reasons,
        "reasons": sorted(set(reasons)),
        "world_size": world_size,
        "minimum_views_per_rank": min(local_views) if local_views else 0,
        "maximum_views_per_rank": max(local_views) if local_views else 0,
        "global_useful_views": sum(local_views),
        "aggregate_views_per_second": sum(throughput),
        "maximum_peak_reserved_fraction": (
            max(peak_reserved) if peak_reserved else None
        ),
        "maximum_peak_allocated_fraction": (
            max(peak_allocated) if peak_allocated else None
        ),
        "maximum_peak_active_fraction": (
            max(peak_active) if peak_active else None
        ),
        "maximum_ending_reserved_fraction": (
            max(ending_reserved) if ending_reserved else None
        ),
        "minimum_ending_driver_free_fraction": (
            min(ending_driver_free) if ending_driver_free else None
        ),
        "maximum_trellis_conditioning_views": (
            max(conditioning_views) if conditioning_views else 0
        ),
        "transport_storage_underflow_edge_fraction": underflow_edge_fraction,
        "transport_storage_underflow_mass_fraction": underflow_mass_fraction,
        "transport_storage_relative_l1_error": storage_relative_l1_error,
        "transport_zero_source_mass_fraction": zero_source_mass_fraction,
        "transport_zero_target_mass_fraction": zero_target_mass_fraction,
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
            -float(candidate["maximum_peak_allocated_fraction"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", help="JSON files or glob patterns")
    parser.add_argument("--maximum-reserved-fraction", type=float, default=0.85)
    parser.add_argument("--maximum-allocated-fraction", type=float, default=0.90)
    parser.add_argument("--minimum-driver-free-fraction", type=float, default=0.05)
    parser.add_argument(
        "--maximum-storage-relative-l1-error", type=float, default=1.0e-6
    )
    parser.add_argument(
        "--maximum-zero-marginal-mass-fraction", type=float, default=1.0e-12
    )
    parser.add_argument(
        "--maximum-storage-underflow-fraction",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--maximum-zero-marginal-fraction",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--throughput-fraction", type=float, default=0.97)
    parser.add_argument("--output", type=Path, default=Path("outputs/concurrency/selection.json"))
    args = parser.parse_args()
    if (
        args.maximum_storage_underflow_fraction is not None
        or args.maximum_zero_marginal_fraction is not None
    ):
        print(
            "warning: edge/row-count underflow limits are deprecated and ignored; "
            "selection uses FP64-referenced discarded mass and relative-L1 error",
            file=__import__("sys").stderr,
        )
    for name in (
        "maximum_reserved_fraction",
        "maximum_allocated_fraction",
    ):
        value = float(getattr(args, name))
        if not 0 < value < 1:
            raise ValueError(f"{name.replace('_', '-')} must lie in (0,1)")
    for name in (
        "minimum_driver_free_fraction",
        "maximum_storage_relative_l1_error",
        "maximum_zero_marginal_mass_fraction",
    ):
        value = float(getattr(args, name))
        if not 0 <= value < 1:
            raise ValueError(f"{name.replace('_', '-')} must lie in [0,1)")

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
        candidate = audit_report(
            value,
            args.maximum_reserved_fraction,
            args.maximum_allocated_fraction,
            args.minimum_driver_free_fraction,
            args.maximum_storage_relative_l1_error,
            args.maximum_zero_marginal_mass_fraction,
        )
        candidate["path"] = str(path)
        audited.append(candidate)
    selection_error: str | None = None
    try:
        selected = select_candidate(audited, args.throughput_fraction)
    except RuntimeError as error:
        selected = None
        selection_error = str(error)
    output = {
        "schema": "graft-gs-a800-view-selection-v5",
        "maximum_reserved_fraction": args.maximum_reserved_fraction,
        "maximum_allocated_fraction": args.maximum_allocated_fraction,
        "minimum_driver_free_fraction": args.minimum_driver_free_fraction,
        "maximum_storage_relative_l1_error": (
            args.maximum_storage_relative_l1_error
        ),
        "maximum_zero_marginal_mass_fraction": (
            args.maximum_zero_marginal_mass_fraction
        ),
        "ignored_legacy_count_gates": {
            "maximum_storage_underflow_fraction": (
                args.maximum_storage_underflow_fraction
            ),
            "maximum_zero_marginal_fraction": args.maximum_zero_marginal_fraction,
        },
        "throughput_fraction": args.throughput_fraction,
        "candidates": audited,
        "selected": selected,
        "recommended_views_per_rank": (
            selected["minimum_views_per_rank"] if selected is not None else None
        ),
        "selection_error": selection_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf8",
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    if selection_error is not None:
        rejection_summary = "; ".join(
            f"{candidate['path']}: {', '.join(candidate['reasons'])}"
            for candidate in audited
        )
        raise RuntimeError(
            f"{selection_error}; audit written to {args.output}; "
            f"candidate rejections: {rejection_summary}"
        )


if __name__ == "__main__":
    main()

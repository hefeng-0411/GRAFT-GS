"""Inspect phase-transfer compatibility without constructing CUDA models."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from graft_gs.engine import (
    MODEL_EXECUTION_POLICY_PATHS,
    load_server_config,
    model_config_differences,
    prepare_model_for_checkpoint,
)
from graft_gs.integration import GraftGS, VGGTAdapter


class _SchemaAggregator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frame_blocks = nn.ModuleList()
        self.global_blocks = nn.ModuleList()
        self.cached_layer_indices = (0, 1, 2, 3)


class _SchemaCameraHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(2048)


class _SchemaVGGT(nn.Module):
    """Minimal released-contract shell; upstream tensors are audited separately."""

    def __init__(self) -> None:
        super().__init__()
        self.aggregator = _SchemaAggregator()
        self.camera_head = _SchemaCameraHead()
        self.depth_head = nn.Identity()
        self.point_head = nn.Identity()


def _graft_state_schema(
    model_config: object,
    phase: str | None,
) -> dict[str, tuple[int, ...]]:
    feature_dim = int(getattr(model_config, "feature_dim"))
    model = GraftGS(
        VGGTAdapter(_SchemaVGGT(), feature_dim=feature_dim),
        model_config,
    )
    prepare_model_for_checkpoint(model, phase)
    return {
        key: tuple(value.shape)
        for key, value in model.state_dict().items()
        if not key.startswith("vggt.model.")
    }


def _checkpoint_graft_state_schema(
    state: Mapping[str, object],
    expected: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, tuple[int, ...]], tuple[str, ...]]:
    translated: dict[str, tuple[int, ...]] = {}
    ignored_upstream: list[str] = []
    for key, value in state.items():
        if key.startswith("vggt.model."):
            ignored_upstream.append(key)
            continue
        target_key = key
        if target_key not in expected and key.endswith(".weight"):
            parametrized = (
                key[: -len(".weight")]
                + ".parametrizations.weight.original"
            )
            if parametrized in expected:
                target_key = parametrized
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else ()
        translated[target_key] = shape
    return translated, tuple(ignored_upstream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/graft_gs_a800_native.yaml"),
    )
    parser.add_argument(
        "--require-source-phase",
        choices=tuple("ABCDEF"),
        help="fail unless the checkpoint declares this source phase",
    )
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint root must be a mapping")
    source_phase_value = payload.get("phase")
    source_phase = str(source_phase_value) if source_phase_value is not None else None
    if args.require_source_phase is not None and source_phase != args.require_source_phase:
        raise ValueError(
            "checkpoint source phase differs: "
            f"expected={args.require_source_phase}, checkpoint={source_phase}"
        )

    current_model, _, _, _ = load_server_config(args.config)
    checkpoint_model = payload.get("model_config")
    if not isinstance(checkpoint_model, Mapping):
        report = {
            "schema": "graft-gs-checkpoint-compatibility-v1",
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "compatible_without_retraining": None,
            "reason": "checkpoint_has_no_model_config",
            "source_phase": source_phase,
            "format_version": payload.get("format_version"),
            "global_step": payload.get("global_step"),
        }
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        raise SystemExit(2)

    current_mapping = asdict(current_model)
    all_differences = model_config_differences(
        checkpoint_model,
        current_mapping,
        include_execution_policy=True,
    )
    architectural_differences = tuple(
        record
        for record in all_differences
        if not bool(record["execution_policy"])
    )
    execution_differences = tuple(
        record for record in all_differences if bool(record["execution_policy"])
    )
    model_state = payload.get("model")
    expected_state = _graft_state_schema(current_model, source_phase)
    if isinstance(model_state, Mapping):
        checkpoint_state, ignored_upstream = _checkpoint_graft_state_schema(
            model_state,
            expected_state,
        )
        missing_state_keys = tuple(sorted(set(expected_state) - set(checkpoint_state)))
        unexpected_state_keys = tuple(
            sorted(set(checkpoint_state) - set(expected_state))
        )
        shape_mismatches = tuple(
            {
                "key": key,
                "checkpoint_shape": checkpoint_state[key],
                "current_shape": expected_state[key],
            }
            for key in sorted(set(checkpoint_state) & set(expected_state))
            if checkpoint_state[key] != expected_state[key]
        )
    else:
        ignored_upstream = ()
        missing_state_keys = tuple(sorted(expected_state))
        unexpected_state_keys = ()
        shape_mismatches = ()
    state_schema_compatible = not (
        missing_state_keys or unexpected_state_keys or shape_mismatches
    )
    compatible = not architectural_differences and state_schema_compatible
    report = {
        "schema": "graft-gs-checkpoint-compatibility-v1",
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "source_phase": source_phase,
        "format_version": payload.get("format_version"),
        "global_step": payload.get("global_step"),
        "model_state_tensor_count": (
            len(model_state) if isinstance(model_state, Mapping) else None
        ),
        "ignored_upstream_vggt_tensor_count": len(ignored_upstream),
        "graft_state_schema": {
            "compatible": state_schema_compatible,
            "expected_tensor_count": len(expected_state),
            "missing_keys": missing_state_keys,
            "unexpected_keys": unexpected_state_keys,
            "shape_mismatches": shape_mismatches,
        },
        "execution_policy_paths": sorted(MODEL_EXECUTION_POLICY_PATHS),
        "ignored_execution_policy_differences": execution_differences,
        "architectural_differences": architectural_differences,
        "compatible_without_retraining": compatible,
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not compatible:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Typed construction of server model/data/training configuration."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import yaml

from ..integration.pipeline import GraftGSConfig
from .losses import LossWeights


def load_server_config(
    path: str | Path,
) -> tuple[GraftGSConfig, dict[str, object], dict[str, object], dict[str, object]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    if not isinstance(data, dict):
        raise ValueError("server configuration root must be a mapping")
    base = GraftGSConfig()
    model = data.get("model", {})
    transport = data.get("transport", {})
    training = data.get("training", {})
    distributed = data.get("distributed", {})
    dataset = data.get("dataset", {})
    for name, value in (
        ("model", model),
        ("transport", transport),
        ("training", training),
        ("distributed", distributed),
        ("dataset", dataset),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"configuration section {name!r} must be a mapping")
    sinkhorn = replace(
        base.mapping.sinkhorn,
        epsilon=float(transport.get("epsilon", base.mapping.sinkhorn.epsilon)),
        tau_source=float(transport.get("tau_source", base.mapping.sinkhorn.tau_source)),
        tau_target=float(transport.get("tau_target", base.mapping.sinkhorn.tau_target)),
        max_iterations=int(transport.get("max_iterations", base.mapping.sinkhorn.max_iterations)),
        tolerance=float(transport.get("tolerance", base.mapping.sinkhorn.tolerance)),
    )
    config = replace(
        base,
        feature_dim=int(model.get("feature_dim", base.feature_dim)),
        atlas=replace(
            base.atlas,
            base_level=int(model.get("base_level", base.atlas.base_level)),
            max_level=int(model.get("max_level", base.atlas.max_level)),
        ),
        mapping=replace(
            base.mapping,
            sinkhorn=sinkhorn,
            retention_shrinkage=float(
                transport.get("retention_shrinkage", base.mapping.retention_shrinkage)
            ),
        ),
        attention=replace(
            base.attention,
            heads=int(model.get("attention_heads", base.attention.heads)),
        ),
        flow=replace(
            base.flow,
            layers=int(model.get("flow_layers", base.flow.layers)),
            steps=int(model.get("flow_steps", base.flow.steps)),
        ),
        encoder_layers=int(model.get("encoder_layers", base.encoder_layers)),
        transport_feature_iterations=int(
            model.get("transport_feature_iterations", base.transport_feature_iterations)
        ),
        refinement_rounds=int(model.get("refinement_rounds", base.refinement_rounds)),
        renderer_backend=str(model.get("renderer_backend", base.renderer_backend)),
    )
    return config, training, distributed, dataset


def load_trellis_prior_config(path: str | Path) -> dict[str, object]:
    """Load the fixed external hidden-support prior policy."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    value = data.get("trellis_prior", {}) if isinstance(data, dict) else {}
    if not isinstance(value, dict):
        raise ValueError("configuration section 'trellis_prior' must be a mapping")
    config: dict[str, object] = {
        "enabled_after_phase_a": bool(value.get("enabled_after_phase_a", False)),
        "samples": int(value.get("samples", 8)),
        "sampler_steps": int(value.get("sampler_steps", 12)),
        "strength": float(value.get("strength", 0.35)),
        "minimum_probability": float(value.get("minimum_probability", 0.0)),
        "uncertainty_discount": float(value.get("uncertainty_discount", 0.5)),
    }
    if int(config["samples"]) < 1 or int(config["sampler_steps"]) < 1:
        raise ValueError("TRELLIS prior samples/steps must be positive")
    if (
        float(config["strength"]) < 0
        or not 0 <= float(config["minimum_probability"]) < 1
        or float(config["uncertainty_discount"]) < 0
    ):
        raise ValueError("TRELLIS prior strength/threshold are outside their domains")
    return config


def load_loss_weights(path: str | Path) -> LossWeights:
    """Load explicit objective weights and reject dead configuration keys."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    value = data.get("loss", {}) if isinstance(data, dict) else {}
    if not isinstance(value, dict):
        raise ValueError("configuration section 'loss' must be a mapping")
    allowed = {item.name for item in fields(LossWeights)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown loss-weight keys: {unknown}")
    converted = {name: float(weight) for name, weight in value.items()}
    if any(weight < 0 for weight in converted.values()):
        raise ValueError("loss weights must be non-negative")
    return replace(LossWeights(), **converted)


__all__ = ["load_loss_weights", "load_server_config", "load_trellis_prior_config"]

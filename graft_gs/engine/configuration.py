"""Typed construction of server model/data/training configuration."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import yaml

from ..integration.pipeline import GraftGSConfig
from .losses import LossWeights
from .precision import NativePrecisionPolicy


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
    barrier = data.get("barrier", {})
    for name, value in (
        ("model", model),
        ("transport", transport),
        ("training", training),
        ("distributed", distributed),
        ("dataset", dataset),
        ("barrier", barrier),
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
        backward_max_iterations=int(
            transport.get(
                "backward_max_iterations",
                base.mapping.sinkhorn.backward_max_iterations,
            )
        ),
        backward_tolerance=float(
            transport.get(
                "backward_tolerance", base.mapping.sinkhorn.backward_tolerance
            )
        ),
        backward_damping=float(
            transport.get(
                "backward_damping", base.mapping.sinkhorn.backward_damping
            )
        ),
        mass_floor=float(
            transport.get("mass_floor", base.mapping.sinkhorn.mass_floor)
        ),
        convergence_check_interval=int(
            transport.get(
                "convergence_check_interval",
                base.mapping.sinkhorn.convergence_check_interval,
            )
        ),
    )
    config = replace(
        base,
        feature_dim=int(model.get("feature_dim", base.feature_dim)),
        atlas=replace(
            base.atlas,
            base_level=int(model.get("base_level", base.atlas.base_level)),
            max_level=int(model.get("max_level", base.atlas.max_level)),
            frame_epsilon=float(
                model.get("frame_epsilon", base.atlas.frame_epsilon)
            ),
            frame_relative_eigengap=float(
                model.get(
                    "frame_relative_eigengap",
                    base.atlas.frame_relative_eigengap,
                )
            ),
        ),
        mapping=replace(
            base.mapping,
            sinkhorn=sinkhorn,
            atlas_chunk_size=int(
                transport.get("atlas_chunk_size", base.mapping.atlas_chunk_size)
            ),
            evidence_chunk_size=int(
                transport.get(
                    "evidence_chunk_size", base.mapping.evidence_chunk_size
                )
            ),
            retention_shrinkage=float(
                transport.get("retention_shrinkage", base.mapping.retention_shrinkage)
            ),
        ),
        readout=replace(
            base.readout,
            metric_epsilon=float(
                model.get("readout_metric_epsilon", base.readout.metric_epsilon)
            ),
            metric_relative_eigengap=float(
                model.get(
                    "readout_metric_relative_eigengap",
                    base.readout.metric_relative_eigengap,
                )
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
        barrier=replace(
            base.barrier,
            minimum_face_area=float(
                barrier.get("minimum_face_area", base.barrier.minimum_face_area)
            ),
            minimum_orientation_cosine=float(
                barrier.get(
                    "minimum_orientation_cosine",
                    base.barrier.minimum_orientation_cosine,
                )
            ),
            minimum_separation=float(
                barrier.get("minimum_separation", base.barrier.minimum_separation)
            ),
            minimum_covariance_eigenvalue=float(
                barrier.get(
                    "minimum_covariance_eigenvalue",
                    base.barrier.minimum_covariance_eigenvalue,
                )
            ),
            maximum_covariance_eigenvalue=float(
                barrier.get(
                    "maximum_covariance_eigenvalue",
                    base.barrier.maximum_covariance_eigenvalue,
                )
            ),
            activation_margin=float(
                barrier.get("activation_margin", base.barrier.activation_margin)
            ),
            decay_rate=float(barrier.get("decay_rate", base.barrier.decay_rate)),
            dual_iterations=int(
                barrier.get("dual_iterations", base.barrier.dual_iterations)
            ),
            dual_tolerance=float(
                barrier.get("dual_tolerance", base.barrier.dual_tolerance)
            ),
            dual_regularization=float(
                barrier.get("dual_regularization", base.barrier.dual_regularization)
            ),
            dual_check_interval=int(
                barrier.get(
                    "dual_check_interval", base.barrier.dual_check_interval
                )
            ),
            maximum_backtracks=int(
                barrier.get("maximum_backtracks", base.barrier.maximum_backtracks)
            ),
            backtrack_factor=float(
                barrier.get("backtrack_factor", base.barrier.backtrack_factor)
            ),
            maximum_position_speed=float(
                barrier.get(
                    "maximum_position_speed", base.barrier.maximum_position_speed
                )
            ),
            restoration_iterations=int(
                barrier.get(
                    "restoration_iterations", base.barrier.restoration_iterations
                )
            ),
            restoration_relative_margin=float(
                barrier.get(
                    "restoration_relative_margin",
                    base.barrier.restoration_relative_margin,
                )
            ),
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


def load_precision_policy(path: str | Path) -> NativePrecisionPolicy:
    """Load and validate the process-wide native precision boundary."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    value = data.get("precision", {}) if isinstance(data, dict) else {}
    if not isinstance(value, dict):
        raise ValueError("configuration section 'precision' must be a mapping")
    allowed = {
        "backbone",
        "geometric_state",
        "analytical_solve",
        "diagnostics",
        "float32_matmul_precision",
        "allow_tf32",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown precision policy keys: {unknown}")
    return NativePrecisionPolicy(
        backbone=str(value.get("backbone", "bfloat16")),
        geometric_state=str(value.get("geometric_state", "float32")),
        analytical_solve=str(value.get("analytical_solve", "float32")),
        diagnostics=str(value.get("diagnostics", "float64")),
        float32_matmul_precision=str(
            value.get("float32_matmul_precision", "highest")
        ),
        allow_tf32=bool(value.get("allow_tf32", False)),
    )


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


__all__ = [
    "load_loss_weights",
    "load_precision_policy",
    "load_server_config",
    "load_trellis_prior_config",
]

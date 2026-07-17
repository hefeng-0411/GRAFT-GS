"""Phase-aware GRAFT-GS checkpoint reconstruction for training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ..integration.pipeline import GraftGS
from ..optimization.quantization import QuantizationConfig, apply_equivariant_qat


@dataclass(frozen=True)
class CheckpointLoadReport:
    phase: str | None
    global_step: int | None
    lora_modules: int
    qat_modules: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def prepare_model_for_checkpoint(model: GraftGS, phase: str | None) -> tuple[int, tuple[str, ...]]:
    """Recreate parameterization structure before loading checkpoint tensors."""

    lora_modules = 0
    qat_modules: tuple[str, ...] = ()
    if phase in {"D", "E", "F"}:
        lora_modules = model.vggt.install_late_lora()
    if phase in {"E", "F"}:
        qat_modules = tuple(
            apply_equivariant_qat(
                model,
                QuantizationConfig(bits=8, block_size=16, stochastic_rounding=True),
            )
        )
    return lora_modules, qat_modules


def load_graft_checkpoint(
    model: GraftGS,
    checkpoint: str | Path | Mapping[str, Any],
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    validate_model_config: bool = True,
) -> tuple[Mapping[str, Any], CheckpointLoadReport]:
    """Load a trainer payload or raw state dictionary without losing parametrizations."""

    payload: Mapping[str, Any]
    if isinstance(checkpoint, (str, Path)):
        loaded = torch.load(checkpoint, map_location=map_location, weights_only=False)
        if not isinstance(loaded, Mapping):
            raise TypeError("GRAFT-GS checkpoint must contain a mapping")
        payload = loaded
    else:
        payload = checkpoint
    phase_value = payload.get("phase")
    phase = str(phase_value) if phase_value is not None else None
    state_value = payload.get("model", payload)
    if not isinstance(state_value, Mapping):
        raise TypeError("checkpoint model state must be a mapping")
    if phase is None and any(".parametrizations.weight.0.a" in key for key in state_value):
        raise ValueError("raw LoRA state dictionaries require checkpoint phase metadata")
    if validate_model_config and "model_config" in payload:
        expected = asdict(model.config)
        if payload["model_config"] != expected:
            raise ValueError(
                "checkpoint model_config differs from the instantiated GRAFT-GS architecture"
            )
    lora_modules, qat_modules = prepare_model_for_checkpoint(model, phase)
    incompatible = model.load_state_dict(state_value, strict=strict)
    report = CheckpointLoadReport(
        phase=phase,
        global_step=int(payload["global_step"]) if "global_step" in payload else None,
        lora_modules=lora_modules,
        qat_modules=qat_modules,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )
    return payload, report


def validate_trellis_prior_policy(
    payload: Mapping[str, Any],
    *,
    enabled: bool,
    samples: int,
    sampler_steps: int,
    strength: float,
    minimum_probability: float,
    uncertainty_discount: float,
) -> None:
    """Verify fixed external prior provenance before inference/distillation."""

    if not enabled:
        return
    trainer = payload.get("trainer_config")
    if not isinstance(trainer, Mapping) or trainer.get("trellis_prior_checkpoint") is None:
        raise ValueError("checkpoint lacks active TRELLIS hidden-prior provenance")
    expected = {
        "trellis_prior_samples": samples,
        "trellis_prior_sampler_steps": sampler_steps,
        "trellis_prior_strength": strength,
        "trellis_prior_minimum_probability": minimum_probability,
        "trellis_prior_uncertainty_discount": uncertainty_discount,
    }
    for name, value in expected.items():
        if trainer.get(name) != value:
            raise ValueError(f"checkpoint TRELLIS hidden-prior policy differs at {name}")


__all__ = [
    "CheckpointLoadReport",
    "load_graft_checkpoint",
    "prepare_model_for_checkpoint",
    "validate_trellis_prior_policy",
]

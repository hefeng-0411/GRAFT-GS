"""Post-reference quantization, certificates, and equivalence instrumentation."""

from .quantization import (
    BlockwiseFakeQuantizer,
    QuantizationConfig,
    QuantizationTopologyCertificate,
    apply_equivariant_qat,
    attention_score_error_bound,
    certify_topology_quantization_step,
    quantization_scale_adversaries,
    topology_step_is_certified,
)
from .gradient_purification import (
    GradientPurificationConfig,
    GradientPurificationDiagnostics,
    HilbertGradientPurifier,
)

__all__ = [
    "BlockwiseFakeQuantizer",
    "GradientPurificationConfig",
    "GradientPurificationDiagnostics",
    "HilbertGradientPurifier",
    "QuantizationConfig",
    "QuantizationTopologyCertificate",
    "apply_equivariant_qat",
    "attention_score_error_bound",
    "certify_topology_quantization_step",
    "quantization_scale_adversaries",
    "topology_step_is_certified",
]

"""Executable native-precision policy for A800 training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class NativePrecisionPolicy:
    r"""Precision boundary required by the high-fidelity reference path.

    The released VGGT transformer is the only BF16 region.  Camera/depth
    heads, atlas/manifold storage, analytical readout, and the CUDA rasterizer
    remain FP32. Sparse UOT is the deliberate exception: its fixed-point
    potentials and implicit conditional probabilities use FP64/log space,
    while its returned geometric plan uses this policy's FP32 storage. FP64 is
    also used by strict feasibility/certification diagnostics. TF32 is disabled
    because its ten-bit product mantissa can perturb marginal SPD, topology,
    and transport decisions even though tensors report ``float32``.
    """

    backbone: str = "bfloat16"
    geometric_state: str = "float32"
    analytical_solve: str = "float32"
    diagnostics: str = "float64"
    float32_matmul_precision: str = "highest"
    allow_tf32: bool = False

    def __post_init__(self) -> None:
        if self.backbone not in {"bfloat16", "float16", "float32"}:
            raise ValueError("backbone precision must be bfloat16, float16, or float32")
        if self.geometric_state != "float32":
            raise ValueError("geometric/manifold state must remain float32 on the A800 path")
        if self.analytical_solve != "float32":
            raise ValueError("analytical solves must remain float32 on the A800 path")
        if self.diagnostics != "float64":
            raise ValueError("certification diagnostics must use float64")
        if self.float32_matmul_precision != "highest":
            raise ValueError("high-fidelity training requires highest float32 matmul precision")
        if self.allow_tf32:
            raise ValueError("TF32 is incompatible with the high-fidelity reference policy")

    @property
    def backbone_dtype(self) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.backbone]

    def apply(self) -> dict[str, object]:
        """Apply process-global PyTorch flags and return checkpoint provenance."""

        torch.set_float32_matmul_precision(self.float32_matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.allow_tf32 = self.allow_tf32
        return {
            **asdict(self),
            "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        }


__all__ = ["NativePrecisionPolicy"]

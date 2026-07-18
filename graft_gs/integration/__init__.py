"""Baseline-preserving VGGT/TRELLIS integration and end-to-end GRAFT-GS."""

from .vggt_adapter import (
    CameraAlignmentDiagnostics,
    VGGTAdapter,
    VGGTGeometryOutput,
    align_vggt_to_supervised_cameras,
)
from .pipeline import GraftGS, GraftGSConfig, GraftGSOutput, RobustnessPerturbation, SceneOutput
from .trellis_prior import TrellisPriorAdapter, TrellisPriorMeasure, TrellisStructurePrior
from .external import (
    DEFAULT_TRELLIS_CHECKPOINT,
    DEFAULT_VGGT_CHECKPOINT,
    import_external_module,
    resolve_trellis_checkpoint,
    resolve_vggt_checkpoint,
)

__all__ = [
    "GraftGS",
    "GraftGSConfig",
    "GraftGSOutput",
    "RobustnessPerturbation",
    "SceneOutput",
    "TrellisPriorAdapter",
    "TrellisPriorMeasure",
    "TrellisStructurePrior",
    "CameraAlignmentDiagnostics",
    "VGGTAdapter",
    "VGGTGeometryOutput",
    "align_vggt_to_supervised_cameras",
    "DEFAULT_TRELLIS_CHECKPOINT",
    "DEFAULT_VGGT_CHECKPOINT",
    "import_external_module",
    "resolve_trellis_checkpoint",
    "resolve_vggt_checkpoint",
]

"""Uncertainty-aware measure transport from image evidence to the atlas."""

from .manifold_mapping import (
    EvidenceParticles,
    GeometricEvidenceBuilder,
    ImplicitSinkhornConfig,
    ManifoldMappingConfig,
    ManifoldMappingOperator,
    MappingResult,
    SparseTransportGraph,
    sparse_view_reprojection_variance,
)

__all__ = [
    "EvidenceParticles",
    "GeometricEvidenceBuilder",
    "ImplicitSinkhornConfig",
    "ManifoldMappingConfig",
    "ManifoldMappingOperator",
    "MappingResult",
    "SparseTransportGraph",
    "sparse_view_reprojection_variance",
]

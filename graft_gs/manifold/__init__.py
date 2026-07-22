"""Product-manifold atlas state, flow matching, and feasibility control."""

from .geometry import (
    ManifoldState,
    ManifoldTangent,
    geodesic_interpolate,
    product_metric_squared,
    spectral_box_spd,
)
from .flow import FlowConfig, RiemannianFlowMatcher, RiemannianVectorField, SafeHeunIntegrator
from .barrier import BarrierConfig, BarrierProjector, FeasibilityReport

__all__ = [
    "BarrierConfig",
    "BarrierProjector",
    "FeasibilityReport",
    "FlowConfig",
    "ManifoldState",
    "ManifoldTangent",
    "RiemannianFlowMatcher",
    "RiemannianVectorField",
    "SafeHeunIntegrator",
    "geodesic_interpolate",
    "product_metric_squared",
    "spectral_box_spd",
]

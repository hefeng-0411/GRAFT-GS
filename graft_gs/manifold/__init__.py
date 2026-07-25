"""Product-manifold atlas state, flow matching, and feasibility control."""

from .geometry import (
    ManifoldState,
    ManifoldTangent,
    geodesic_interpolate,
    precision_to_bounded_covariance,
    product_metric_squared,
    spd_inverse_cholesky,
    spd_inverse_logdet_cholesky,
    spd_inverse_quadratic_trace,
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
    "precision_to_bounded_covariance",
    "product_metric_squared",
    "spd_inverse_cholesky",
    "spd_inverse_logdet_cholesky",
    "spd_inverse_quadratic_trace",
    "spectral_box_spd",
]

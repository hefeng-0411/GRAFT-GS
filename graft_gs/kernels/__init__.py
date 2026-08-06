"""Compiled hardware kernels and audited numerical entry points."""

from .fused_ssim import fused_ssim_loss, triton_ssim_available
from .geometry_primitives import (
    KeOpsFUGWConfig,
    KeOpsFUGWDiagnostics,
    KeOpsFUGWSolver,
    NerfaccOccupancyGrid,
    VolumetricRenderResult,
    fixed_radius_neighbors,
    geomloss_sinkhorn_divergence,
    keops_compact_partition,
    keops_fugw_tensor_product,
    keops_gaussian_reduction,
    keops_squared_distance_minima,
    nearest_neighbor_indices,
    nerfacc_volume_render,
)

__all__ = [
    "KeOpsFUGWConfig",
    "KeOpsFUGWDiagnostics",
    "KeOpsFUGWSolver",
    "NerfaccOccupancyGrid",
    "VolumetricRenderResult",
    "fixed_radius_neighbors",
    "fused_ssim_loss",
    "geomloss_sinkhorn_divergence",
    "keops_compact_partition",
    "keops_fugw_tensor_product",
    "keops_gaussian_reduction",
    "keops_squared_distance_minima",
    "nearest_neighbor_indices",
    "nerfacc_volume_render",
    "triton_ssim_available",
]

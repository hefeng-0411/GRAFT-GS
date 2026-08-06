"""Fail-fast ABI audit for the compiled geometry execution path."""

from __future__ import annotations

import inspect
import json
import platform
import sys


def validate_geometry_extensions() -> dict[str, object]:
    import torch

    errors: list[str] = []
    versions: dict[str, str] = {"torch": str(torch.__version__)}

    try:
        import frnn

        required = {"K", "r", "return_nn", "return_sorted"}
        parameters = set(inspect.signature(frnn.frnn_grid_points).parameters)
        if not required.issubset(parameters):
            errors.append("FRNN frnn_grid_points ABI is missing required arguments")
        versions["frnn"] = str(getattr(frnn, "__version__", "local"))
    except Exception as error:
        errors.append(f"FRNN import/ABI failure: {type(error).__name__}: {error}")

    try:
        import pykeops
        from pykeops.torch import LazyTensor  # noqa: F401

        versions["pykeops"] = str(getattr(pykeops, "__version__", "local"))
    except Exception as error:
        errors.append(f"KeOps import/ABI failure: {type(error).__name__}: {error}")

    try:
        import geomloss
        from geomloss import SamplesLoss  # noqa: F401

        versions["geomloss"] = str(getattr(geomloss, "__version__", "local"))
    except Exception as error:
        errors.append(f"GeomLoss import/ABI failure: {type(error).__name__}: {error}")

    try:
        import gsplat
        from gsplat.cuda._math import _rotmat_to_quat  # noqa: F401
        from gsplat.rendering import rasterization

        required = {
            "quats",
            "scales",
            "extra_signals",
            "global_z_order",
            "packed",
            "render_mode",
            "rasterize_mode",
            "sh_degree",
            "tile_size",
        }
        parameters = set(inspect.signature(rasterization).parameters)
        missing = sorted(required - parameters)
        if missing:
            errors.append(f"gsplat rasterization ABI is missing: {missing}")
        versions["gsplat"] = str(getattr(gsplat, "__version__", "local"))
    except Exception as error:
        errors.append(f"gsplat import/ABI failure: {type(error).__name__}: {error}")

    try:
        import nerfacc

        required = (
            "OccGridEstimator",
            "accumulate_along_rays",
            "render_weight_from_density",
        )
        missing = [name for name in required if not hasattr(nerfacc, name)]
        if missing:
            errors.append(f"nerfacc ABI is missing: {missing}")
        versions["nerfacc"] = str(getattr(nerfacc, "__version__", "local"))
    except Exception as error:
        errors.append(f"nerfacc import/ABI failure: {type(error).__name__}: {error}")

    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable; compiled production kernels cannot be exercised")
    return {
        "valid": not errors,
        "errors": errors,
        "versions": versions,
        "python": platform.python_version(),
        "executable": sys.executable,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


def main() -> None:
    result = validate_geometry_extensions()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["valid"] else 2)


if __name__ == "__main__":
    main()

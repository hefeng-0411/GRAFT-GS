"""Deployment-safe discovery of the released VGGT and TRELLIS packages.

The declared server checkouts are preferred when physically present. A
different checkout can be exposed explicitly through ``GRAFT_GS_VGGT_ROOT``
or ``GRAFT_GS_TRELLIS_ROOT``; an installed package is the portable fallback.
No workstation path or repository-relative sibling assumption is embedded in
production code.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Optional


DEFAULT_VGGT_CHECKPOINT = "facebook/VGGT-1B"
DEFAULT_TRELLIS_CHECKPOINT = "microsoft/TRELLIS-image-large"
DEFAULT_VGGT_REPOSITORY_ROOT = Path("/mnt/sda2/hef/Base/vggt")
DEFAULT_TRELLIS_REPOSITORY_ROOT = Path("/mnt/sda2/hef/Base/TRELLIS")

_ROOT_ENVIRONMENT = {
    "vggt": "GRAFT_GS_VGGT_ROOT",
    "trellis": "GRAFT_GS_TRELLIS_ROOT",
}
_DEFAULT_REPOSITORY_ROOT = {
    "vggt": DEFAULT_VGGT_REPOSITORY_ROOT,
    "trellis": DEFAULT_TRELLIS_REPOSITORY_ROOT,
}


def resolve_vggt_checkpoint(value: Optional[str | Path] = None) -> str:
    return _resolve_checkpoint(
        value,
        ("GRAFT_GS_VGGT_CHECKPOINT", "VGGT_CHECKPOINT"),
        DEFAULT_VGGT_CHECKPOINT,
    )


def resolve_trellis_checkpoint(value: Optional[str | Path] = None) -> str:
    return _resolve_checkpoint(
        value,
        ("GRAFT_GS_TRELLIS_CHECKPOINT", "TRELLIS_CHECKPOINT"),
        DEFAULT_TRELLIS_CHECKPOINT,
    )


def _resolve_checkpoint(
    value: Optional[str | Path], environment_names: tuple[str, ...], default: str
) -> str:
    if value is not None and str(value).strip():
        return str(value)
    for name in environment_names:
        configured = os.environ.get(name)
        if configured and configured.strip():
            return configured.strip()
    return default


def import_external_module(
    module_name: str,
    repository_root: Optional[str | Path] = None,
) -> ModuleType:
    """Import an upstream module from installation or an explicit checkout.

    ``repository_root`` is the checkout containing the top-level package
    directory, not the package directory itself. Environment configuration is
    consulted only when an explicit value is absent.
    """

    package = module_name.split(".", 1)[0]
    if package not in _ROOT_ENVIRONMENT:
        raise ValueError(f"unsupported external package {package!r}")
    configured = repository_root
    if configured is None:
        configured = os.environ.get(_ROOT_ENVIRONMENT[package])
    if (configured is None or not str(configured).strip()) and (
        _DEFAULT_REPOSITORY_ROOT[package] / package
    ).is_dir():
        configured = _DEFAULT_REPOSITORY_ROOT[package]
    root: Optional[Path] = None
    if configured is not None and str(configured).strip():
        root = Path(configured).expanduser().resolve()
        if not root.is_dir() or not (root / package).is_dir():
            raise FileNotFoundError(
                f"{_ROOT_ENVIRONMENT[package]} must name the checkout containing "
                f"the {package!r} package: {root}"
            )
        root_string = str(root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        importlib.invalidate_caches()
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != package:
            raise ImportError(
                f"{package} was found but dependency {error.name!r} is unavailable"
            ) from error
        environment = _ROOT_ENVIRONMENT[package]
        raise ImportError(
            f"cannot import {package}; install its checkout in the active environment "
            f"or set {environment} to the upstream repository root"
        ) from error
    if root is not None:
        origin_value = getattr(module, "__file__", None)
        if origin_value is not None:
            origin = Path(origin_value).resolve()
            try:
                origin.relative_to(root)
            except ValueError as error:
                raise ImportError(
                    f"imported {module_name} from {origin}, outside configured root {root}"
                ) from error
    return module


def external_module_provenance(module: ModuleType, checkpoint: str) -> dict[str, str]:
    origin = getattr(module, "__file__", None)
    return {
        "module": module.__name__,
        "module_file": str(Path(origin).resolve()) if origin is not None else "unknown",
        "checkpoint": checkpoint,
    }


__all__ = [
    "DEFAULT_TRELLIS_REPOSITORY_ROOT",
    "DEFAULT_TRELLIS_CHECKPOINT",
    "DEFAULT_VGGT_REPOSITORY_ROOT",
    "DEFAULT_VGGT_CHECKPOINT",
    "external_module_provenance",
    "import_external_module",
    "resolve_trellis_checkpoint",
    "resolve_vggt_checkpoint",
]

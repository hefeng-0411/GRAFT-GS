"""GRAFT-GS high-precision reference implementation.

The package deliberately keeps the original ``vggt`` and ``TRELLIS`` trees
unchanged.  Integration happens through explicit geometric contracts in this
package so both baseline entry points remain independently usable.
"""

from .geometry.atlas import AtlasConfig, AtlasValidation, PersistentOctreeAtlas

__all__ = ["AtlasConfig", "AtlasValidation", "PersistentOctreeAtlas"]


"""Static multiview datasets for staged GRAFT-GS training."""

from .meshfleet import (
    MANIFEST_SCHEMA,
    MeshFleetDatasetConfig,
    MeshFleetObjectDataset,
    ObjectManifestRecord,
    build_meshfleet_manifest,
    intrinsics_from_fov,
    load_meshfleet_manifest,
    meshfleet_single_object_collate,
    opengl_c2w_to_opencv_c2w,
    topology_supervision_is_admissible,
)
from .multiview import FolderMultiviewDataset, single_object_collate
from .mesh_supervision import MeshDerivedTargets, MeshGroundTruthRasterizer, TriangleSoup

__all__ = [
    "FolderMultiviewDataset",
    "MANIFEST_SCHEMA",
    "MeshFleetDatasetConfig",
    "MeshFleetObjectDataset",
    "MeshDerivedTargets",
    "MeshGroundTruthRasterizer",
    "ObjectManifestRecord",
    "TriangleSoup",
    "build_meshfleet_manifest",
    "intrinsics_from_fov",
    "load_meshfleet_manifest",
    "meshfleet_single_object_collate",
    "opengl_c2w_to_opencv_c2w",
    "topology_supervision_is_admissible",
    "single_object_collate",
]

"""Static multiview datasets for staged GRAFT-GS training."""

from .meshfleet import (
    DEFAULT_OPTIONAL_MODALITIES,
    DEFAULT_PRIMARY_MODALITIES,
    DEFAULT_REQUIRED_MODALITIES,
    MANIFEST_SCHEMA,
    MESHFLEET_MODALITIES,
    MeshFleetDatasetConfig,
    MeshFleetObjectDataset,
    ObjectManifestRecord,
    build_meshfleet_manifest,
    intrinsics_from_fov,
    load_meshfleet_manifest,
    load_meshfleet_object_ids,
    meshfleet_object_id_digest,
    meshfleet_record_admission_reasons,
    meshfleet_single_object_collate,
    opengl_c2w_to_opencv_c2w,
    topology_supervision_is_admissible,
)
from .multiview import FolderMultiviewDataset, single_object_collate
from .mesh_supervision import MeshDerivedTargets, MeshGroundTruthRasterizer, TriangleSoup

__all__ = [
    "DEFAULT_OPTIONAL_MODALITIES",
    "DEFAULT_PRIMARY_MODALITIES",
    "DEFAULT_REQUIRED_MODALITIES",
    "FolderMultiviewDataset",
    "MANIFEST_SCHEMA",
    "MESHFLEET_MODALITIES",
    "MeshFleetDatasetConfig",
    "MeshFleetObjectDataset",
    "MeshDerivedTargets",
    "MeshGroundTruthRasterizer",
    "ObjectManifestRecord",
    "TriangleSoup",
    "build_meshfleet_manifest",
    "intrinsics_from_fov",
    "load_meshfleet_manifest",
    "load_meshfleet_object_ids",
    "meshfleet_object_id_digest",
    "meshfleet_record_admission_reasons",
    "meshfleet_single_object_collate",
    "opengl_c2w_to_opencv_c2w",
    "topology_supervision_is_admissible",
    "single_object_collate",
]

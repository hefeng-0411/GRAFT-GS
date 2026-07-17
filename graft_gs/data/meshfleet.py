"""Audited MeshFleet-to-TRELLIS object dataset contract.

The source lineage is MeshFleet object acquisition followed by the released
TRELLIS ``dataset_toolkits`` preprocessing.  This module deliberately keeps
three kinds of supervision separate:

``ground_truth``
    Blender cameras, RGBA observations, the normalized render mesh, and the
    explicitly voxelized *surface* samples.
``derived``
    Quantities that can be deterministically rasterized or computed from the
    ground-truth mesh/cameras (depth, normals, topology, visibility).
``pseudo_label``
    DINOv2 patch tokens and pretrained TRELLIS structure/structured latents.

In particular, the sparse voxel PLY is not a solid occupancy grid and the
pretrained latent arrays are not geometric ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from math import tan
from pathlib import Path
import struct
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch
    from torch import Tensor


MANIFEST_SCHEMA = "meshfleet-trellis-object-v2"
_VIEW_MODALITIES = ("renders", "renders_cond", "renders_eval_70", "renders_eval_90")
_ARRAY_MODALITIES = ("features", "latents", "ss_latents")


@dataclass(frozen=True)
class MeshFleetDatasetConfig:
    """Configuration for a physical-file-verified object-level dataset.

    ``image_size`` is ``(height, width)``.  No view is repeated when fewer than
    ``maximum_views`` are available.  Records with fewer than ``minimum_views``
    in the selected camera set are excluded with an explicit diagnostic.
    """

    root: str | Path
    split: str = "train"
    manifest: Optional[str | Path] = None
    input_view_set: str = "renders"
    image_size: tuple[int, int] = (518, 518)
    minimum_views: int = 2
    maximum_views: Optional[int] = 12
    view_selection: str = "uniform"
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    foreground_alpha_threshold: float = 0.5
    surface_grid_resolution: int = 64
    load_surface_voxels: bool = True
    load_trellis_features: bool = False
    load_trellis_latents: bool = False
    load_structure_latent: bool = False
    dino_pseudo_confidence: float = 0.5
    trellis_latent_pseudo_confidence: float = 0.5
    require_surface_voxels: bool = False
    topology_supervision_mode: str = "validated_or_repaired"
    minimum_topology_confidence: float = 0.95
    verify_files_at_load: bool = True
    teacher_bundle_root: Optional[str | Path] = None
    minimum_teacher_bundle_confidence: float = 0.0
    require_teacher_bundle: bool = False
    seed: int = 17

    def __post_init__(self) -> None:
        if self.minimum_views < 1:
            raise ValueError("minimum_views must be positive")
        if self.maximum_views is not None and self.maximum_views < self.minimum_views:
            raise ValueError("maximum_views cannot be smaller than minimum_views")
        if len(self.image_size) != 2 or min(self.image_size) < 1:
            raise ValueError("image_size must contain positive (height, width)")
        if self.view_selection not in {"uniform", "random"}:
            raise ValueError("view_selection must be 'uniform' or 'random'")
        if len(self.background_rgb) != 3 or any(not 0.0 <= value <= 1.0 for value in self.background_rgb):
            raise ValueError("background_rgb must contain three values in [0, 1]")
        if not 0.0 <= self.foreground_alpha_threshold <= 1.0:
            raise ValueError("foreground_alpha_threshold must lie in [0, 1]")
        if self.surface_grid_resolution < 1:
            raise ValueError("surface_grid_resolution must be positive")
        if self.topology_supervision_mode not in {
            "disabled",
            "validated_raw_only",
            "validated_or_repaired",
        }:
            raise ValueError("unsupported topology_supervision_mode")
        if not 0.0 <= self.minimum_topology_confidence <= 1.0:
            raise ValueError("minimum_topology_confidence must lie in [0,1]")
        if not 0.0 <= self.dino_pseudo_confidence <= 1.0:
            raise ValueError("dino_pseudo_confidence must lie in [0,1]")
        if not 0.0 <= self.trellis_latent_pseudo_confidence <= 1.0:
            raise ValueError("trellis_latent_pseudo_confidence must lie in [0,1]")
        if not 0.0 <= self.minimum_teacher_bundle_confidence <= 1.0:
            raise ValueError("minimum teacher-bundle confidence must lie in [0,1]")
        if self.require_teacher_bundle and self.teacher_bundle_root is None:
            raise ValueError("required teacher bundles need an explicit root")


@dataclass
class ObjectManifestRecord:
    schema: str
    object_id: str
    split: str
    views: dict[str, dict[str, Any]] = field(default_factory=dict)
    modalities: dict[str, dict[str, Any]] = field(default_factory=dict)
    supervision: dict[str, list[str]] = field(default_factory=dict)
    topology_supervision: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObjectManifestRecord":
        record = cls(**dict(value))
        if record.schema != MANIFEST_SCHEMA:
            raise ValueError(f"unsupported MeshFleet manifest schema {record.schema!r}")
        return record


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _array_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype)}
            for key in sorted(archive.files)
        }
    return {"path": path, "arrays": arrays}


def _ply_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        lines: list[str] = []
        while True:
            raw = file.readline()
            if not raw:
                raise ValueError(f"truncated PLY header: {path}")
            line = raw.decode("ascii").strip()
            lines.append(line)
            if line == "end_header":
                break
    if lines[0] != "ply":
        raise ValueError(f"invalid PLY magic: {path}")
    elements: dict[str, int] = {}
    for line in lines:
        if line.startswith("element "):
            _, name, count = line.split()
            elements[name] = int(count)
    return {"format": lines[1].split()[1], "elements": elements}


def _read_ply_xyz_numpy(path: Path) -> np.ndarray:
    scalar = {
        "char": "b",
        "int8": "b",
        "uchar": "B",
        "uint8": "B",
        "short": "h",
        "int16": "h",
        "ushort": "H",
        "uint16": "H",
        "int": "i",
        "int32": "i",
        "uint": "I",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }
    with path.open("rb") as file:
        lines: list[str] = []
        while True:
            raw = file.readline()
            if not raw:
                raise ValueError(f"truncated PLY header: {path}")
            line = raw.decode("ascii").strip()
            lines.append(line)
            if line == "end_header":
                break
        format_name = lines[1].split()[1]
        vertex_count = 0
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in lines:
            if line.startswith("element "):
                _, name, count = line.split()
                in_vertex = name == "vertex"
                if in_vertex:
                    vertex_count = int(count)
            elif in_vertex and line.startswith("property "):
                fields = line.split()
                if fields[1] == "list":
                    raise ValueError("list property is invalid inside a PLY vertex element")
                properties.append((fields[1], fields[2]))
        names = [name for _, name in properties]
        if any(name not in names for name in ("x", "y", "z")):
            raise ValueError(f"PLY has no xyz vertex fields: {path}")
        if format_name == "ascii":
            rows = [file.readline().decode("ascii").split() for _ in range(vertex_count)]
            values = np.asarray(rows, dtype=np.float64)
        elif format_name in {"binary_little_endian", "binary_big_endian"}:
            try:
                codes = "".join(scalar[data_type] for data_type, _ in properties)
            except KeyError as error:
                raise ValueError(f"unsupported PLY scalar type {error.args[0]}") from error
            row = struct.Struct(("<" if format_name == "binary_little_endian" else ">") + codes)
            values = np.asarray([row.unpack(file.read(row.size)) for _ in range(vertex_count)])
        else:
            raise ValueError(f"unsupported PLY format {format_name}")
    return values[:, [names.index("x"), names.index("y"), names.index("z")]].astype(np.float64)


def _triangle_mesh_topology(path: Path) -> dict[str, Any]:
    """Audit raw triangle connectivity without assuming it is a manifold."""

    scalar_size = {
        "char": 1,
        "int8": 1,
        "uchar": 1,
        "uint8": 1,
        "short": 2,
        "int16": 2,
        "ushort": 2,
        "uint16": 2,
        "int": 4,
        "int32": 4,
        "uint": 4,
        "uint32": 4,
        "float": 4,
        "float32": 4,
        "double": 8,
        "float64": 8,
    }
    with path.open("rb") as file:
        lines: list[str] = []
        while True:
            raw = file.readline()
            if not raw:
                raise ValueError(f"truncated PLY header: {path}")
            line = raw.decode("ascii").strip()
            lines.append(line)
            if line == "end_header":
                break
        if lines[1].split()[1] != "binary_little_endian":
            return {"available": False, "reason": "topology audit currently requires binary_little_endian PLY"}
        element = None
        counts: dict[str, int] = {}
        properties: dict[str, list[list[str]]] = {}
        for line in lines:
            fields = line.split()
            if fields[:1] == ["element"]:
                element = fields[1]
                counts[element] = int(fields[2])
                properties[element] = []
            elif fields[:1] == ["property"] and element is not None:
                properties[element].append(fields[1:])
        vertex_count = counts.get("vertex", 0)
        face_count = counts.get("face", 0)
        vertex_row_size = 0
        for prop in properties.get("vertex", []):
            if prop[0] == "list":
                return {"available": False, "reason": "list-valued vertex property"}
            vertex_row_size += scalar_size[prop[0]]
        face_properties = properties.get("face", [])
        if face_properties != [["list", "uchar", "uint", "vertex_indices"]]:
            return {"available": False, "reason": f"unsupported face properties {face_properties}"}
        file.seek(vertex_count * vertex_row_size, 1)
        faces: list[tuple[int, ...]] = []
        arities: dict[int, int] = {}
        for _ in range(face_count):
            raw_count = file.read(1)
            if len(raw_count) != 1:
                raise ValueError(f"truncated face list in {path}")
            arity = struct.unpack("<B", raw_count)[0]
            indices = struct.unpack("<" + "I" * arity, file.read(4 * arity))
            faces.append(indices)
            arities[arity] = arities.get(arity, 0) + 1
    result: dict[str, Any] = {
        "available": True,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "face_arity_histogram": {str(key): value for key, value in sorted(arities.items())},
    }
    if set(arities) != {3}:
        result.update(
            closed_two_manifold=False,
            reason="non-triangular face topology requires an explicit triangulation policy",
        )
        return result
    triangle = np.asarray(faces, dtype=np.int64)
    invalid_index = np.any((triangle < 0) | (triangle >= vertex_count), axis=1)
    repeated_index = np.any(
        np.stack(
            (
                triangle[:, 0] == triangle[:, 1],
                triangle[:, 1] == triangle[:, 2],
                triangle[:, 2] == triangle[:, 0],
            ),
            axis=1,
        ),
        axis=1,
    )
    position = _read_ply_xyz_numpy(path)
    geometric_degenerate = np.zeros(face_count, dtype=bool)
    index_valid_triangle = triangle[~invalid_index]
    if index_valid_triangle.size:
        vertex = position[index_valid_triangle]
        double_area_squared = np.sum(
            np.cross(vertex[:, 1] - vertex[:, 0], vertex[:, 2] - vertex[:, 0]) ** 2,
            axis=1,
        )
        extent = np.ptp(position, axis=0)
        area_floor_squared = max(float(np.dot(extent, extent)) ** 2 * 1.0e-24, 1.0e-30)
        geometric_degenerate[~invalid_index] = double_area_squared <= area_floor_squared
    degenerate = invalid_index | repeated_index | geometric_degenerate
    valid = triangle[~degenerate]
    face_index = np.nonzero(~degenerate)[0]
    oriented_edge = np.concatenate(
        (valid[:, (0, 1)], valid[:, (1, 2)], valid[:, (2, 0)]), axis=0
    )
    oriented_face = np.tile(face_index, 3)
    orientation_sign = np.where(oriented_edge[:, 0] < oriented_edge[:, 1], 1, -1)
    edge = np.sort(oriented_edge, axis=1)
    unique_edge, inverse_edge, incidence = np.unique(
        edge, axis=0, return_inverse=True, return_counts=True
    )
    used = np.unique(valid)
    parent = np.arange(vertex_count, dtype=np.int64)
    rank = np.zeros(vertex_count, dtype=np.uint8)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left == right:
            return
        if rank[left] < rank[right]:
            left, right = right, left
        parent[right] = left
        if rank[left] == rank[right]:
            rank[left] += 1

    for left, right in unique_edge:
        union(int(left), int(right))
    used_component_roots = {find(int(index)) for index in used}
    boundary_edges = int(np.sum(incidence == 1))
    nonmanifold_edges = int(np.sum(incidence > 2))
    isolated_vertices = vertex_count - int(used.size)
    component_count = len(used_component_roots) + isolated_vertices
    euler = int(vertex_count - unique_edge.shape[0] + valid.shape[0])
    incidence_histogram = {
        str(int(value)): int(np.sum(incidence == value)) for value in np.unique(incidence)
    }
    two_face_edge = np.nonzero(incidence == 2)[0]
    order = np.argsort(inverse_edge, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(incidence)))
    same_direction_two_face_edges = 0
    dual_adjacency: list[list[tuple[int, int]]] = [[] for _ in range(face_count)]
    for edge_index in two_face_edge:
        member = order[offsets[edge_index] : offsets[edge_index + 1]]
        left_member, right_member = int(member[0]), int(member[1])
        left_face = int(oriented_face[left_member])
        right_face = int(oriented_face[right_member])
        same_direction = orientation_sign[left_member] == orientation_sign[right_member]
        same_direction_two_face_edges += int(same_direction)
        # A same-direction shared edge requires one face flip; an opposite
        # direction requires equal flip parity.
        parity = int(same_direction)
        dual_adjacency[left_face].append((right_face, parity))
        dual_adjacency[right_face].append((left_face, parity))
    two_face_orientation_consistent = same_direction_two_face_edges == 0
    orientable: Optional[bool]
    if nonmanifold_edges:
        orientable = None
        orientability_status = "indeterminate_nonmanifold"
        orientation_consistency_status = "invalid_nonmanifold"
        orientation_consistent: Optional[bool] = None
    else:
        parity_assignment = np.full(face_count, -1, dtype=np.int8)
        orientable = True
        for seed in face_index:
            if parity_assignment[seed] >= 0:
                continue
            parity_assignment[seed] = 0
            stack = [int(seed)]
            while stack and orientable:
                current = stack.pop()
                for neighbor, relation in dual_adjacency[current]:
                    expected = int(parity_assignment[current]) ^ relation
                    if parity_assignment[neighbor] < 0:
                        parity_assignment[neighbor] = expected
                        stack.append(neighbor)
                    elif int(parity_assignment[neighbor]) != expected:
                        orientable = False
                        break
        orientability_status = "orientable" if orientable else "nonorientable"
        orientation_consistent = two_face_orientation_consistent
        orientation_consistency_status = (
            "consistent" if orientation_consistent else "repairable_face_winding"
        )
    watertight = (
        valid.shape[0] > 0
        and not bool(np.any(degenerate))
        and boundary_edges == 0
        and nonmanifold_edges == 0
        and bool(np.all(incidence == 2))
    )
    closed_two_manifold = (
        watertight
        and isolated_vertices == 0
    )
    hard_topology_admissible = bool(
        closed_two_manifold and orientable is True and orientation_consistent is True
    )
    result.update(
        valid_triangle_count=int(valid.shape[0]),
        degenerate_face_count=int(np.sum(degenerate)),
        invalid_index_face_count=int(np.sum(invalid_index)),
        repeated_index_face_count=int(np.sum(repeated_index)),
        geometric_degenerate_face_count=int(np.sum(geometric_degenerate)),
        edge_count=int(unique_edge.shape[0]),
        unique_edge_count=int(unique_edge.shape[0]),
        edge_incidence_histogram=incidence_histogram,
        boundary_edge_count=boundary_edges,
        nonmanifold_edge_count=nonmanifold_edges,
        maximum_edge_incidence=int(incidence.max()) if incidence.size else 0,
        two_face_edge_count=int(two_face_edge.size),
        same_direction_two_face_edge_count=same_direction_two_face_edges,
        two_face_orientation_consistent=two_face_orientation_consistent,
        orientation_consistency_status=orientation_consistency_status,
        orientation_consistent=orientation_consistent,
        orientability_status=orientability_status,
        orientable=orientable,
        used_vertex_count=int(used.size),
        isolated_vertex_count=isolated_vertices,
        connected_components=component_count,
        euler_characteristic=euler,
        watertight=watertight,
        closed_two_manifold=closed_two_manifold,
        hard_topology_supervision_admissible=hard_topology_admissible,
    )
    if hard_topology_admissible:
        b0 = component_count
        b2 = b0
        result["betti_z2"] = [b0, b0 + b2 - euler, b2]
    return result


def _topology_supervision_contract(
    mesh_path: str,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify topology provenance without inventing a label."""

    audit_available = bool(audit.get("available", False))
    admissible = bool(audit.get("hard_topology_supervision_admissible", False))
    betti = audit.get("betti_z2") if admissible else None
    reason = (
        "raw connectivity passed closed, orientable, consistently oriented simplicial 2-manifold checks"
        if admissible
        else audit.get(
            "reason",
            "raw connectivity failed closed orientable simplicial 2-manifold validation",
        )
    )
    selected_status = "validated_topology_ground_truth" if admissible else "unavailable"
    confidence = 1.0 if admissible else 0.0
    return {
        "raw_source_mesh_topology": {
            "available": True,
            "path": mesh_path,
            "provenance": "direct_source_connectivity",
            "statistics": dict(audit),
        },
        "validated_topology_ground_truth": {
            "available": admissible,
            "provenance": "validated_raw_source_connectivity" if admissible else "unavailable",
            "confidence": confidence,
            "reason": reason,
            "target_betti_z2": betti,
            "target_persistence": None,
        },
        "repaired_topology": {
            "available": False,
            "provenance": "unavailable",
            "confidence": 0.0,
            "repair_method": None,
            "validation": None,
            "target_betti_z2": None,
            "target_persistence": None,
        },
        "derived_topology_statistics": {
            "available": audit_available,
            "provenance": "derived_from_raw_source_connectivity" if audit_available else "unavailable",
            "confidence": 1.0 if audit_available else 0.0,
            "statistics": dict(audit),
        },
        "teacher_pseudo_topology": {
            "available": False,
            "provenance": "unavailable",
            "confidence": 0.0,
            "teacher_checkpoint": None,
            "target_betti_z2": None,
            "target_persistence": None,
        },
        "selected_label": {
            "status": selected_status,
            "provenance": "validated_raw_source_connectivity" if admissible else "unavailable",
            "confidence": confidence,
            "reason": reason,
            "hard_topology_supervision_admissible": admissible,
            "hard_betti_supervision_admissible": admissible and betti is not None,
            # Persistent diagrams require an explicit filtration and stratum
            # labels require an atlas-to-source-complex correspondence. Neither
            # is fabricated from raw mesh connectivity.
            "hard_persistence_supervision_admissible": False,
            "hard_stratum_supervision_admissible": False,
            "manifold_certification_admissible": admissible,
            "target_betti_z2": betti,
            "target_persistence": None,
            "target_stratum": None,
        },
    }


def topology_supervision_is_admissible(
    contract: Mapping[str, Any],
    target: str,
) -> bool:
    """Pure policy check shared by manifest tests and model-side consumers."""

    key = {
        "topology": "hard_topology_supervision_admissible",
        "betti": "hard_betti_supervision_admissible",
        "persistence": "hard_persistence_supervision_admissible",
        "stratum": "hard_stratum_supervision_admissible",
        "manifold_certification": "manifold_certification_admissible",
    }.get(target)
    if key is None:
        raise ValueError(f"unknown topology supervision target {target!r}")
    selected = contract.get("selected_label")
    if not isinstance(selected, Mapping):
        return False
    admissible = bool(selected.get(key, False))
    if admissible:
        confidence = float(selected.get("confidence", 0.0))
        if not 0.0 < confidence <= 1.0:
            raise ValueError("admissible topology supervision requires confidence in (0,1]")
        if selected.get("provenance") in {None, "unavailable"}:
            raise ValueError("admissible topology supervision requires explicit provenance")
    return admissible


def _glb_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        magic, version, declared_length = struct.unpack("<4sII", file.read(12))
    if magic != b"glTF" or version != 2:
        raise ValueError(f"expected a glTF 2.0 binary asset: {path}")
    if declared_length != path.stat().st_size:
        raise ValueError(f"GLB declared length differs from physical size: {path}")
    return {"version": version, "byte_length": declared_length}


def _safe_frame_path(directory: Path, file_path: str) -> Optional[Path]:
    candidate = (directory / file_path).resolve()
    try:
        candidate.relative_to(directory.resolve())
    except ValueError as error:
        raise ValueError(f"camera frame escapes its object directory: {file_path}") from error
    return candidate if candidate.is_file() else None


def _audit_camera_set(directory: Path, root: Path, inspect_image_headers: bool) -> dict[str, Any]:
    transform_path = directory / "transforms.json"
    metadata = json.loads(transform_path.read_text(encoding="utf8"))
    frames = metadata.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"camera manifest has no frame list: {transform_path}")
    available: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    determinant_errors: list[float] = []
    orthogonality_errors: list[float] = []
    image_shapes: set[tuple[int, int, int]] = set()
    for index, frame in enumerate(frames):
        matrix = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"frame {index} has an invalid 4x4 transform in {transform_path}")
        rotation = matrix[:3, :3]
        determinant_errors.append(abs(float(np.linalg.det(rotation)) - 1.0))
        orthogonality_errors.append(float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))))
        path = _safe_frame_path(directory, str(frame.get("file_path", "")))
        entry = {"frame_index": index, "declared_file_path": str(frame.get("file_path", ""))}
        if path is None:
            missing.append(entry)
            continue
        entry["file"] = _relative(path, root)
        available.append(entry)
        if inspect_image_headers:
            with Image.open(path) as image:
                bands = len(image.getbands())
                image_shapes.add((image.height, image.width, bands))
    result: dict[str, Any] = {
        "directory": _relative(directory, root),
        "transforms": _relative(transform_path, root),
        "declared_frame_count": len(frames),
        "available_frame_count": len(available),
        "missing_frame_count": len(missing),
        "available_frames": available,
        "missing_frames": missing,
        "maximum_rotation_determinant_error": max(determinant_errors, default=0.0),
        "maximum_rotation_orthogonality_error": max(orthogonality_errors, default=0.0),
    }
    if inspect_image_headers:
        result["image_shapes_hwc"] = [list(shape) for shape in sorted(image_shapes)]
    for key in ("aabb", "scale", "offset"):
        if key in metadata:
            result[key] = metadata[key]
    return result


def _discover_object_ids(split_root: Path) -> set[str]:
    identifiers: set[str] = set()
    for modality in _VIEW_MODALITIES:
        directory = split_root / modality
        if directory.is_dir():
            identifiers.update(path.parent.name for path in directory.rglob("transforms.json"))
    for modality in _ARRAY_MODALITIES:
        directory = split_root / modality
        if directory.is_dir():
            identifiers.update(path.stem for path in directory.rglob("*.npz"))
    voxel_root = split_root / "voxels"
    if voxel_root.is_dir():
        identifiers.update(path.stem for path in voxel_root.rglob("*.ply"))
    normalized_root = split_root / "mesh_normalized"
    if normalized_root.is_dir():
        identifiers.update(path.parent.name for path in normalized_root.rglob("bounding_box.json"))
    return identifiers


def _unique_match(directory: Path, pattern: str, object_id: str) -> Optional[Path]:
    if not directory.is_dir():
        return None
    matches = sorted(path for path in directory.rglob(pattern) if path.stem == object_id or path.parent.name == object_id)
    if len(matches) > 1:
        raise ValueError(f"ambiguous {pattern} records for object {object_id}: {matches}")
    return matches[0] if matches else None


def build_meshfleet_manifest(
    root: str | Path,
    output_path: str | Path,
    splits: Sequence[str] = ("train", "test"),
    inspect_image_headers: bool = True,
) -> dict[str, Any]:
    """Audit all physical modalities and write deterministic JSON Lines records."""

    root = Path(root).resolve()
    output_path = Path(output_path)
    records: list[ObjectManifestRecord] = []
    split_counts: dict[str, int] = {}
    for split in splits:
        split_root = root / split
        if not split_root.is_dir():
            split_counts[split] = 0
            continue
        identifiers = sorted(_discover_object_ids(split_root))
        split_counts[split] = len(identifiers)
        for object_id in identifiers:
            record = ObjectManifestRecord(
                schema=MANIFEST_SCHEMA,
                object_id=object_id,
                split=split,
                supervision={
                    "ground_truth": [],
                    "derived": [],
                    "pseudo_label": [],
                },
            )
            for view_name in _VIEW_MODALITIES:
                manifest = _unique_match(split_root / view_name, "transforms.json", object_id)
                if manifest is None:
                    continue
                audit = _audit_camera_set(manifest.parent, root, inspect_image_headers)
                record.views[view_name] = audit
                record.supervision["ground_truth"].extend(
                    (f"{view_name}.rgba", f"{view_name}.camera_to_world_opengl")
                )
                if audit["missing_frame_count"]:
                    record.warnings.append(
                        f"{view_name}: {audit['missing_frame_count']} declared frames have no physical image"
                    )
            for modality in _ARRAY_MODALITIES:
                path = _unique_match(split_root / modality, "*.npz", object_id)
                if path is None:
                    continue
                metadata = _array_metadata(path)
                metadata["path"] = _relative(path, root)
                record.modalities[modality] = metadata
                record.supervision["pseudo_label"].append(modality)
            voxel = _unique_match(split_root / "voxels", "*.ply", object_id)
            if voxel is not None:
                record.modalities["surface_voxels"] = {
                    "path": _relative(voxel, root),
                    **_ply_header(voxel),
                    "semantics": "surface samples at centers of occupied cells in a canonical 64^3 grid",
                }
                record.supervision["ground_truth"].append("surface_voxels")
                record.supervision["derived"].extend(("surface_distance", "surface_occupancy"))
            render_mesh = _unique_match(split_root / "renders", "mesh.ply", object_id)
            if render_mesh is not None:
                topology_audit = _triangle_mesh_topology(render_mesh)
                render_mesh_path = _relative(render_mesh, root)
                record.topology_supervision = _topology_supervision_contract(
                    render_mesh_path,
                    topology_audit,
                )
                record.modalities["render_mesh"] = {
                    "path": render_mesh_path,
                    **_ply_header(render_mesh),
                    "topology_audit": topology_audit,
                }
                record.supervision["ground_truth"].append("render_mesh_geometry")
                record.supervision["derived"].extend(
                    (
                        "mesh_depth",
                        "mesh_normal",
                        "mesh_visibility",
                        "raw_mesh_connectivity_statistics",
                    )
                )
                record.checks["render_mesh_topology"] = topology_audit
                if topology_supervision_is_admissible(record.topology_supervision, "betti"):
                    record.supervision["ground_truth"].append("validated_topology_betti_z2")
                else:
                    record.warnings.append(
                        "raw render-mesh topology is inadmissible for hard Betti, persistence, stratum, and manifold-certification supervision"
                    )
            else:
                record.topology_supervision = {
                    "raw_source_mesh_topology": {
                        "available": False,
                        "path": None,
                        "provenance": "unavailable",
                        "statistics": None,
                    },
                    "validated_topology_ground_truth": {
                        "available": False,
                        "provenance": "unavailable",
                        "confidence": 0.0,
                        "reason": "render mesh unavailable",
                        "target_betti_z2": None,
                        "target_persistence": None,
                    },
                    "repaired_topology": {
                        "available": False,
                        "provenance": "unavailable",
                        "confidence": 0.0,
                        "repair_method": None,
                        "validation": None,
                        "target_betti_z2": None,
                        "target_persistence": None,
                    },
                    "derived_topology_statistics": {
                        "available": False,
                        "provenance": "unavailable",
                        "confidence": 0.0,
                        "statistics": None,
                    },
                    "teacher_pseudo_topology": {
                        "available": False,
                        "provenance": "unavailable",
                        "confidence": 0.0,
                        "teacher_checkpoint": None,
                        "target_betti_z2": None,
                        "target_persistence": None,
                    },
                    "selected_label": {
                        "status": "unavailable",
                        "provenance": "unavailable",
                        "confidence": 0.0,
                        "reason": "render mesh unavailable",
                        "hard_topology_supervision_admissible": False,
                        "hard_betti_supervision_admissible": False,
                        "hard_persistence_supervision_admissible": False,
                        "hard_stratum_supervision_admissible": False,
                        "manifold_certification_admissible": False,
                        "target_betti_z2": None,
                        "target_persistence": None,
                        "target_stratum": None,
                    },
                }
            bounding_box = _unique_match(split_root / "mesh_normalized", "bounding_box.json", object_id)
            normalized_mesh = bounding_box.parent / "mesh.glb" if bounding_box is not None else None
            if bounding_box is not None:
                record.modalities["normalized_bounding_box"] = {
                    "path": _relative(bounding_box, root),
                    "value": json.loads(bounding_box.read_text(encoding="utf8")),
                }
                record.supervision["ground_truth"].append("normalized_bounding_box")
            if normalized_mesh is not None and normalized_mesh.is_file():
                record.modalities["normalized_mesh"] = {
                    "path": _relative(normalized_mesh, root),
                    **_glb_header(normalized_mesh),
                }
                record.supervision["ground_truth"].append("normalized_mesh")
            if "features" in record.modalities and "latents" in record.modalities:
                with np.load(root / record.modalities["features"]["path"], allow_pickle=False) as archive:
                    feature_index = np.asarray(archive["indices"])
                with np.load(root / record.modalities["latents"]["path"], allow_pickle=False) as archive:
                    latent_index = np.asarray(archive["coords"])
                equal = bool(np.array_equal(feature_index, latent_index))
                record.checks["feature_indices_equal_latent_coords"] = equal
                if not equal:
                    record.warnings.append("DINO feature indices and TRELLIS latent coordinates differ")
            if voxel is not None:
                voxel_xyz = _read_ply_xyz_numpy(voxel)
                voxel_index = np.rint((voxel_xyz + 0.5) * 64.0 - 0.5).astype(np.int64)
                reconstructed = (voxel_index.astype(np.float64) + 0.5) / 64.0 - 0.5
                residual = float(np.max(np.abs(voxel_xyz - reconstructed)))
                in_bounds = bool(np.all((voxel_index >= 0) & (voxel_index < 64)))
                record.checks["surface_voxel_grid"] = {
                    "resolution": 64,
                    "maximum_center_residual": residual,
                    "indices_in_bounds": in_bounds,
                }
                if residual > 1.0e-6 or not in_bounds:
                    record.warnings.append("surface PLY is not on canonical 64^3 voxel centers")
                if "features" in record.modalities:
                    with np.load(root / record.modalities["features"]["path"], allow_pickle=False) as archive:
                        feature_index = np.asarray(archive["indices"])
                    equal = bool(np.array_equal(voxel_index, feature_index.astype(np.int64)))
                    record.checks["surface_voxel_indices_equal_feature_indices"] = equal
                    if not equal:
                        record.warnings.append("surface voxel and DINO feature sparse coordinates differ")
            records.append(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "schema": MANIFEST_SCHEMA,
        "dataset_root": str(root),
        "record_count": len(records),
        "split_counts": split_counts,
        "warning_count": sum(len(record.warnings) for record in records),
        "manifest": str(output_path.resolve()),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return summary


def load_meshfleet_manifest(path: str | Path) -> list[ObjectManifestRecord]:
    records: list[ObjectManifestRecord] = []
    with Path(path).open(encoding="utf8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                records.append(ObjectManifestRecord.from_dict(json.loads(line)))
            except Exception as error:
                raise ValueError(f"invalid manifest record at line {line_number}") from error
    return records


def _frame_angles(frame: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[float, float]:
    angle_x = frame.get("camera_angle_x", metadata.get("camera_angle_x"))
    angle_y = frame.get("camera_angle_y", metadata.get("camera_angle_y", angle_x))
    if angle_x is None or angle_y is None:
        raise ValueError("every camera frame requires camera_angle_x and camera_angle_y")
    angle_x, angle_y = float(angle_x), float(angle_y)
    if not 0.0 < angle_x < np.pi or not 0.0 < angle_y < np.pi:
        raise ValueError("camera field of view must lie in (0, pi)")
    return angle_x, angle_y


def opengl_c2w_to_opencv_c2w(matrix: Tensor) -> Tensor:
    """Convert Blender/OpenGL camera axes (+x,+y,-z view) to OpenCV axes."""

    import torch

    if matrix.shape[-2:] != (4, 4):
        raise ValueError("camera-to-world matrix must end in [4,4]")
    axis = torch.diag(matrix.new_tensor((1.0, -1.0, -1.0, 1.0)))
    return matrix @ axis


def intrinsics_from_fov(
    fov_x: float,
    fov_y: float,
    native_height: int,
    native_width: int,
    output_height: int,
    output_width: int,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Pixel-center intrinsic matrix after a direct image resize."""

    import torch

    dtype = torch.float32 if dtype is None else dtype
    intrinsic = torch.eye(3, dtype=dtype)
    intrinsic[0, 0] = 0.5 * native_width / tan(0.5 * fov_x)
    intrinsic[1, 1] = 0.5 * native_height / tan(0.5 * fov_y)
    intrinsic[0, 2] = 0.5 * native_width
    intrinsic[1, 2] = 0.5 * native_height
    scale = intrinsic.new_tensor((output_width / native_width, output_height / native_height))
    intrinsic[0] *= scale[0]
    intrinsic[1] *= scale[1]
    return intrinsic


def _read_surface_ply(path: Path) -> Tensor:
    """Read XYZ from an ASCII or binary little-endian vertex-only surface PLY."""

    import torch

    try:
        from plyfile import PlyData
    except ImportError as error:
        raise ImportError("loading surface voxels requires the declared plyfile dependency") from error
    vertex = PlyData.read(str(path))["vertex"]
    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float32, copy=False)
    return torch.from_numpy(np.array(points, copy=True))


class MeshFleetObjectDataset:
    """Configurable object-level loader for the audited TRELLIS preprocessing schema."""

    def __init__(self, config: MeshFleetDatasetConfig) -> None:
        self.config = config
        self.root = Path(config.root).resolve()
        manifest_path = Path(config.manifest) if config.manifest is not None else None
        if manifest_path is None:
            raise ValueError("MeshFleetObjectDataset requires an audited manifest; run build_meshfleet_manifest first")
        records = load_meshfleet_manifest(manifest_path)
        self.manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.teacher_bundle_root = (
            Path(config.teacher_bundle_root).resolve()
            if config.teacher_bundle_root is not None
            else None
        )
        self.records: list[ObjectManifestRecord] = []
        self.excluded: dict[str, str] = {}
        for record in records:
            if record.split != config.split:
                continue
            view = record.views.get(config.input_view_set)
            if view is None:
                self.excluded[record.object_id] = f"missing view set {config.input_view_set}"
                continue
            count = int(view.get("available_frame_count", 0))
            if count < config.minimum_views:
                self.excluded[record.object_id] = (
                    f"{config.input_view_set} has {count} physical frames; requires {config.minimum_views}"
                )
                continue
            if config.require_surface_voxels and "surface_voxels" not in record.modalities:
                self.excluded[record.object_id] = "missing required surface voxel PLY"
                continue
            if (
                config.require_teacher_bundle
                and self.teacher_bundle_root is not None
                and not (self.teacher_bundle_root / f"{record.object_id}.teacher.pt").is_file()
            ):
                self.excluded[record.object_id] = "missing required refined teacher bundle"
                continue
            self.records.append(record)
        if not self.records:
            detail = "; ".join(f"{key}: {value}" for key, value in sorted(self.excluded.items()))
            raise ValueError(f"no usable MeshFleet objects for split {config.split!r}. {detail}")
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _frame_inventory(self, record: ObjectManifestRecord) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        view = record.views[self.config.input_view_set]
        transform_path = self.root / view["transforms"]
        metadata = json.loads(transform_path.read_text(encoding="utf8"))
        by_index = {int(entry["frame_index"]): entry for entry in view["available_frames"]}
        inventory: list[dict[str, Any]] = []
        for index, frame in enumerate(metadata["frames"]):
            if index not in by_index:
                continue
            entry = dict(frame)
            entry["physical_file"] = by_index[index]["file"]
            entry["frame_index"] = index
            path = self.root / entry["physical_file"]
            if self.config.verify_files_at_load and not path.is_file():
                raise FileNotFoundError(f"manifest frame disappeared from disk: {path}")
            inventory.append(entry)
        return metadata, inventory

    def _select_frames(self, object_id: str, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        maximum = self.config.maximum_views
        if maximum is None or len(frames) <= maximum:
            return frames
        if self.config.view_selection == "uniform":
            index = np.linspace(0, len(frames) - 1, maximum, dtype=np.int64)
        else:
            digest = hashlib.sha256(f"{object_id}:{self.config.seed}:{self.epoch}".encode()).digest()
            generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            index = np.sort(generator.choice(len(frames), size=maximum, replace=False))
        return [frames[int(i)] for i in index]

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        record = self.records[index]
        metadata, inventory = self._frame_inventory(record)
        frames = self._select_frames(record.object_id, inventory)
        output_height, output_width = self.config.image_size
        images: list[Tensor] = []
        alphas: list[Tensor] = []
        evidence_masks: list[Tensor] = []
        c2w_cv: list[Tensor] = []
        w2c_cv: list[Tensor] = []
        intrinsics: list[Tensor] = []
        native_sizes: list[tuple[int, int]] = []
        frame_indices: list[int] = []
        background = torch.tensor(self.config.background_rgb, dtype=torch.float32)[:, None, None]
        for frame in frames:
            image_path = self.root / frame["physical_file"]
            with Image.open(image_path) as source:
                rgba = source.convert("RGBA")
                native_width, native_height = rgba.size
                resized = rgba.resize((output_width, output_height), Image.Resampling.LANCZOS)
                pixels = torch.from_numpy(np.array(resized, dtype=np.uint8, copy=True)).permute(2, 0, 1).float() / 255.0
            alpha = pixels[3:4].clamp(0.0, 1.0)
            rgb = pixels[:3] * alpha + background * (1.0 - alpha)
            matrix_gl = torch.as_tensor(frame["transform_matrix"], dtype=torch.float32)
            rotation_gl = matrix_gl[:3, :3]
            eye = torch.eye(3, dtype=rotation_gl.dtype)
            if not torch.allclose(rotation_gl.transpose(0, 1) @ rotation_gl, eye, atol=2.0e-4, rtol=0.0):
                raise ValueError(f"camera frame {frame['frame_index']} is not an orthogonal rigid transform")
            if not torch.allclose(torch.linalg.det(rotation_gl), rotation_gl.new_tensor(1.0), atol=2.0e-4, rtol=0.0):
                raise ValueError(f"camera frame {frame['frame_index']} rotation is not in SO(3)")
            matrix_cv = opengl_c2w_to_opencv_c2w(matrix_gl)
            inverse = torch.linalg.inv(matrix_cv)
            fov_x, fov_y = _frame_angles(frame, metadata)
            intrinsic = intrinsics_from_fov(
                fov_x,
                fov_y,
                native_height,
                native_width,
                output_height,
                output_width,
            )
            images.append(rgb)
            alphas.append(alpha)
            evidence_masks.append(alpha >= self.config.foreground_alpha_threshold)
            c2w_cv.append(matrix_cv)
            w2c_cv.append(inverse[:3])
            intrinsics.append(intrinsic)
            native_sizes.append((native_height, native_width))
            frame_indices.append(int(frame["frame_index"]))
        result: dict[str, object] = {
            "object_id": record.object_id,
            "split": record.split,
            "view_set": self.config.input_view_set,
            "images": torch.stack(images),
            "alpha": torch.stack(alphas),
            "evidence_mask": torch.stack(evidence_masks),
            # Compatibility with the generic pipeline: validity means that a
            # pixel may generate geometric evidence, not target opacity.
            "valid_mask": torch.stack(evidence_masks),
            "camera_to_world_opencv": torch.stack(c2w_cv),
            "extrinsics_world_to_camera": torch.stack(w2c_cv),
            "intrinsics": torch.stack(intrinsics),
            "frame_indices": torch.tensor(frame_indices, dtype=torch.int64),
            "native_sizes_hw": torch.tensor(native_sizes, dtype=torch.int64),
            "normalization": {
                key: metadata[key] for key in ("aabb", "scale", "offset") if key in metadata
            },
            "supervision_provenance": record.supervision,
            "topology_supervision": record.topology_supervision,
            "modality_paths": {
                name: str(self.root / value["path"])
                for name, value in record.modalities.items()
                if "path" in value
            },
            "dataset_warnings": tuple(record.warnings),
            "surface_grid_resolution": self.config.surface_grid_resolution,
            "surface_cell_size": 1.0 / self.config.surface_grid_resolution,
            "dino_pseudo_supervision_mask": torch.tensor(False, dtype=torch.bool),
            "trellis_latent_pseudo_supervision_mask": torch.tensor(
                False, dtype=torch.bool
            ),
            "dino_pseudo_confidence": torch.tensor(0.0, dtype=torch.float32),
            "trellis_latent_pseudo_confidence": torch.tensor(
                0.0, dtype=torch.float32
            ),
            "dino_pseudo_provenance": "unavailable",
            "trellis_latent_pseudo_provenance": "unavailable",
        }
        if "aabb" in metadata:
            atlas_root_bounds = torch.as_tensor(metadata["aabb"], dtype=torch.float32)
            if atlas_root_bounds.shape != (2, 3) or torch.any(
                atlas_root_bounds[1] <= atlas_root_bounds[0]
            ):
                raise ValueError("camera manifest aabb must define [minimum, maximum] in 3D")
            extent = atlas_root_bounds[1] - atlas_root_bounds[0]
            if not torch.allclose(extent, extent.max().expand_as(extent), atol=1.0e-6, rtol=0.0):
                raise ValueError("persistent octree root requires a cubic camera-manifest aabb")
            result["atlas_root_bounds"] = atlas_root_bounds
        selected_topology = record.topology_supervision.get("selected_label", {})
        manifest_topology_admissible = topology_supervision_is_admissible(
            record.topology_supervision,
            "topology",
        )
        topology_provenance = str(selected_topology.get("provenance", "unavailable"))
        topology_confidence = float(selected_topology.get("confidence", 0.0))
        permitted_provenance = {"validated_raw_source_connectivity"}
        if self.config.topology_supervision_mode == "validated_or_repaired":
            permitted_provenance.add("validated_repaired_topology")
        topology_mask = (
            self.config.topology_supervision_mode != "disabled"
            and manifest_topology_admissible
            and topology_provenance in permitted_provenance
            and topology_confidence >= self.config.minimum_topology_confidence
        )
        topology_target_masks = {
            target: topology_mask
            and topology_supervision_is_admissible(record.topology_supervision, target)
            for target in ("betti", "persistence", "stratum", "manifold_certification")
        }
        if topology_mask:
            topology_activation_reason = "admissible label passed configured provenance/confidence policy"
        elif not manifest_topology_admissible:
            topology_activation_reason = str(
                selected_topology.get("reason", "manifest topology label is inadmissible")
            )
        elif self.config.topology_supervision_mode == "disabled":
            topology_activation_reason = "topology supervision disabled by dataset configuration"
        elif topology_provenance not in permitted_provenance:
            topology_activation_reason = "topology provenance rejected by dataset configuration"
        else:
            topology_activation_reason = "topology confidence is below the configured threshold"
        result["topology_supervision_mask"] = torch.tensor(topology_mask, dtype=torch.bool)
        result["topology_betti_supervision_mask"] = torch.tensor(
            topology_target_masks["betti"], dtype=torch.bool
        )
        result["topology_persistence_supervision_mask"] = torch.tensor(
            topology_target_masks["persistence"], dtype=torch.bool
        )
        result["topology_stratum_supervision_mask"] = torch.tensor(
            topology_target_masks["stratum"], dtype=torch.bool
        )
        result["source_manifold_certification_mask"] = torch.tensor(
            topology_target_masks["manifold_certification"], dtype=torch.bool
        )
        result["topology_supervision_confidence"] = torch.tensor(
            topology_confidence,
            dtype=torch.float32,
        )
        result["topology_label_provenance"] = topology_provenance
        result["topology_activation_reason"] = topology_activation_reason
        result["topology_target_betti_z2"] = (
            torch.tensor(selected_topology["target_betti_z2"], dtype=torch.int64)
            if topology_target_masks["betti"]
            else None
        )
        self._load_sparse_modalities(record, result)
        result["teacher_bundle_supervision_mask"] = torch.tensor(False, dtype=torch.bool)
        result["teacher_bundle_confidence"] = torch.tensor(0.0, dtype=torch.float32)
        result["teacher_bundle_provenance"] = "unavailable"
        result["teacher_bundle_activation_reason"] = "teacher bundle root is not configured"
        if self.teacher_bundle_root is not None:
            bundle_path = self.teacher_bundle_root / f"{record.object_id}.teacher.pt"
            if bundle_path.is_file():
                from ..engine.teacher_refinement import load_teacher_bundle

                bundle = load_teacher_bundle(
                    bundle_path,
                    expected_object_id=record.object_id,
                    expected_manifest_sha256=self.manifest_sha256,
                    minimum_confidence=0.0,
                )
                active = (
                    bundle.confidence >= self.config.minimum_teacher_bundle_confidence
                )
                result["teacher_bundle_supervision_mask"] = torch.tensor(
                    active, dtype=torch.bool
                )
                result["teacher_bundle_confidence"] = torch.tensor(
                    bundle.confidence, dtype=torch.float32
                )
                result["teacher_bundle_provenance"] = bundle.topology_provenance
                result["teacher_bundle_activation_reason"] = (
                    "refined teacher passed identity, manifest, provenance, and confidence checks"
                    if active
                    else "refined teacher confidence is below the configured threshold"
                )
                if active:
                    result["teacher_target_state"] = bundle.state
                    result["target_state_provenance"] = bundle.topology_provenance
            else:
                result["teacher_bundle_activation_reason"] = "object teacher bundle is absent"
        return result

    def _load_sparse_modalities(self, record: ObjectManifestRecord, result: dict[str, object]) -> None:
        import torch

        cfg = self.config
        if cfg.load_surface_voxels and "surface_voxels" in record.modalities:
            points = _read_surface_ply(self.root / record.modalities["surface_voxels"]["path"])
            result["surface_voxel_centers"] = points
            resolution = float(cfg.surface_grid_resolution)
            grid = torch.round((points + 0.5) * resolution - 0.5).to(torch.int64)
            reconstructed = (grid.to(points.dtype) + 0.5) / resolution - 0.5
            residual = torch.max(torch.abs(points - reconstructed))
            if (
                float(residual) > 1.0e-6
                or torch.any(grid < 0)
                or torch.any(grid >= cfg.surface_grid_resolution)
            ):
                raise ValueError(
                    "surface voxel PLY is not aligned to configured canonical grid cell centers"
                )
            result["surface_voxel_indices"] = grid
        if cfg.load_trellis_features and "features" in record.modalities:
            with np.load(self.root / record.modalities["features"]["path"], allow_pickle=False) as archive:
                result["trellis_feature_indices"] = torch.from_numpy(np.array(archive["indices"], copy=True)).long()
                result["trellis_patchtokens"] = torch.from_numpy(np.array(archive["patchtokens"], copy=True))
            result["dino_pseudo_confidence"] = torch.tensor(
                cfg.dino_pseudo_confidence, dtype=torch.float32
            )
            result["dino_pseudo_supervision_mask"] = torch.tensor(
                cfg.dino_pseudo_confidence > 0, dtype=torch.bool
            )
            result["dino_pseudo_provenance"] = "pretrained_dinov2_surface_feature"
        if cfg.load_trellis_latents and "latents" in record.modalities:
            with np.load(self.root / record.modalities["latents"]["path"], allow_pickle=False) as archive:
                result["trellis_latent_coords"] = torch.from_numpy(np.array(archive["coords"], copy=True)).long()
                result["trellis_latent_features"] = torch.from_numpy(np.array(archive["feats"], copy=True))
            result["trellis_latent_pseudo_confidence"] = torch.tensor(
                cfg.trellis_latent_pseudo_confidence, dtype=torch.float32
            )
            result["trellis_latent_pseudo_supervision_mask"] = torch.tensor(
                cfg.trellis_latent_pseudo_confidence > 0, dtype=torch.bool
            )
            result["trellis_latent_pseudo_provenance"] = (
                "pretrained_trellis_structured_latent_encoder"
            )
        if cfg.load_structure_latent and "ss_latents" in record.modalities:
            with np.load(self.root / record.modalities["ss_latents"]["path"], allow_pickle=False) as archive:
                keys = sorted(archive.files)
                if keys != ["mean"]:
                    raise ValueError(f"unsupported structure latent fields: {keys}")
                result["trellis_structure_latent_mean"] = torch.from_numpy(np.array(archive["mean"], copy=True))
        if "surface_voxel_indices" in result and "trellis_feature_indices" in result:
            if not torch.equal(result["surface_voxel_indices"], result["trellis_feature_indices"]):
                raise ValueError("surface voxel and DINO feature coordinates are not aligned")
        if "trellis_feature_indices" in result and "trellis_latent_coords" in result:
            if not torch.equal(result["trellis_feature_indices"], result["trellis_latent_coords"]):
                raise ValueError("DINO feature and TRELLIS latent coordinates are not aligned")


def meshfleet_single_object_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    """Batch camera-aligned view tensors while retaining variable-size surfaces."""

    import torch

    if len(batch) != 1:
        raise ValueError("the reference GRAFT-GS trainer uses one variable-topology object per rank")
    item = dict(batch[0])
    for key in (
        "images",
        "alpha",
        "evidence_mask",
        "valid_mask",
        "camera_to_world_opencv",
        "extrinsics_world_to_camera",
        "intrinsics",
        "frame_indices",
        "native_sizes_hw",
        "atlas_root_bounds",
        "topology_supervision_mask",
        "topology_betti_supervision_mask",
        "topology_persistence_supervision_mask",
        "topology_stratum_supervision_mask",
        "source_manifold_certification_mask",
        "topology_supervision_confidence",
        "topology_target_betti_z2",
        "dino_pseudo_supervision_mask",
        "trellis_latent_pseudo_supervision_mask",
        "dino_pseudo_confidence",
        "trellis_latent_pseudo_confidence",
        "teacher_bundle_supervision_mask",
        "teacher_bundle_confidence",
    ):
        if key in item and item[key] is not None:
            item[key] = torch.as_tensor(item[key]).unsqueeze(0)
    teacher_target = item.pop("teacher_target_state", None)
    if teacher_target is not None:
        item["target_states"] = [teacher_target]
    return item


__all__ = [
    "MANIFEST_SCHEMA",
    "MeshFleetDatasetConfig",
    "MeshFleetObjectDataset",
    "ObjectManifestRecord",
    "build_meshfleet_manifest",
    "intrinsics_from_fov",
    "load_meshfleet_manifest",
    "meshfleet_single_object_collate",
    "opengl_c2w_to_opencv_c2w",
    "topology_supervision_is_admissible",
]

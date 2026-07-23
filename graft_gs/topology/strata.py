"""Structured topology-stratum inference on explicit finite complexes.

Topology proposal, selection, and topology-preserving refinement are separate
operations.  This module performs only the first two.  Betti numbers and
persistence are computed over Z2 from explicit boundary matrices; no continuous
flow is permitted to add/remove a cell after selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..equivariant.gsta import active_adjacency
from ..geometry.atlas import PersistentOctreeAtlas


@dataclass(frozen=True)
class TopologySelectorConfig:
    occupancy_thresholds: Tuple[float, ...] = (0.15, 0.3, 0.45, 0.6)
    minimum_vertices: int = 4
    triangle_planarity_cosine: float = 0.35
    minimum_triangle_area: float = 1.0e-10
    maximum_candidates: int = 12
    persistence_order: int = 2
    temperature: float = 0.1
    evidence_weight: float = 1.0
    persistence_weight: float = 0.5
    geometry_weight: float = 0.5
    complexity_weight: float = 1.0e-3
    boundary_weight: float = 0.05
    prior_weight: float = 0.25
    adaptive_threshold_quantiles: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    maximum_persistence_thresholds: int = 6
    minimum_persistence_lifetime: float = 0.02

    def __post_init__(self) -> None:
        if self.maximum_candidates < 1 or self.maximum_persistence_thresholds < 0:
            raise ValueError("topology candidate counts must be non-negative/positive")
        if self.minimum_persistence_lifetime < 0:
            raise ValueError("minimum persistence lifetime must be non-negative")
        if self.minimum_vertices < 3 or self.minimum_triangle_area <= 0:
            raise ValueError("topology complexes require vertices and positive face area")
        if not 0.0 <= self.triangle_planarity_cosine <= 1.0:
            raise ValueError("triangle planarity cosine must lie in [0,1]")
        if self.persistence_order < 1 or self.temperature <= 0:
            raise ValueError("persistence order and topology temperature must be positive")
        if min(
            self.evidence_weight,
            self.persistence_weight,
            self.geometry_weight,
            self.complexity_weight,
            self.prior_weight,
        ) <= 0 or self.boundary_weight < 0:
            raise ValueError("learned topology weights must be positive and boundary weight non-negative")
        if any(not 0.0 < value < 1.0 for value in self.occupancy_thresholds):
            raise ValueError("occupancy thresholds must lie in (0,1)")
        if any(not 0.0 < value < 1.0 for value in self.adaptive_threshold_quantiles):
            raise ValueError("adaptive topology quantiles must lie in (0,1)")


@dataclass
class SimplicialComplex:
    """A 2D simplicial complex embedded by selected atlas chart centers."""

    atlas_node_index: Tensor  # [Nv], persistent global atlas IDs
    edges: Tensor  # [Ne,2], local vertex IDs
    faces: Tensor  # [Nf,3], oriented local vertex IDs

    @property
    def num_vertices(self) -> int:
        return int(self.atlas_node_index.numel())

    @property
    def num_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

    def edge_incidence(self) -> Tensor:
        if self.num_edges == 0:
            return torch.empty(0, dtype=torch.int64, device=self.atlas_node_index.device)
        edge_map = {tuple(edge): i for i, edge in enumerate(self.edges.tolist())}
        count = torch.zeros(self.num_edges, dtype=torch.int64, device=self.edges.device)
        for face in self.faces.tolist():
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                key = (min(a, b), max(a, b))
                count[edge_map[key]] += 1
        return count

    def manifold_incidence_valid(self) -> bool:
        incidence = self.edge_incidence()
        return bool(torch.all((incidence >= 1) & (incidence <= 2))) if incidence.numel() else False

    def orientation_consistent(self) -> bool:
        """Whether every two-face edge is traversed in opposite directions."""

        orientation: Dict[Tuple[int, int], List[int]] = {}
        for face in self.faces.tolist():
            for a, b in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            ):
                key = (min(a, b), max(a, b))
                orientation.setdefault(key, []).append(1 if a < b else -1)
        return all(
            len(direction) <= 2
            and (len(direction) == 1 or direction[0] == -direction[1])
            for direction in orientation.values()
        ) and bool(orientation)

    def boundary_edge_count(self) -> int:
        return int(torch.sum(self.edge_incidence() == 1).item())


@dataclass
class TopologyCandidate:
    identifier: str
    complex: SimplicialComplex
    betti: Tuple[int, int, int]
    persistence: Dict[int, Tensor]
    evidence_energy: Tensor
    persistence_energy: Tensor
    geometry_energy: Tensor
    prior_energy: Tensor
    total_energy: Tensor
    manifold_incidence_valid: bool = True
    orientation_consistent: bool = True
    boundary_edge_count: int = 0


@dataclass
class TopologySelection:
    candidates: List[TopologyCandidate]
    probability: Tensor
    selected_index: int

    @property
    def selected(self) -> TopologyCandidate:
        return self.candidates[self.selected_index]


def _gf2_rank(columns: Iterable[int]) -> int:
    """Rank of binary column bitsets by pivot elimination."""

    pivots: Dict[int, int] = {}
    for column in columns:
        value = int(column)
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


def betti_numbers(complex_: SimplicialComplex) -> Tuple[int, int, int]:
    edge_columns = [(1 << int(a)) | (1 << int(b)) for a, b in complex_.edges.tolist()]
    edge_map = {tuple(edge): i for i, edge in enumerate(complex_.edges.tolist())}
    face_columns: List[int] = []
    for i, j, k in complex_.faces.tolist():
        bits = 0
        for a, b in ((i, j), (j, k), (k, i)):
            bits |= 1 << edge_map[(min(a, b), max(a, b))]
        face_columns.append(bits)
    rank_1 = _gf2_rank(edge_columns)
    rank_2 = _gf2_rank(face_columns)
    beta_0 = complex_.num_vertices - rank_1
    beta_1 = complex_.num_edges - rank_1 - rank_2
    beta_2 = complex_.num_faces - rank_2
    return beta_0, beta_1, beta_2


def persistent_homology(complex_: SimplicialComplex, vertex_filtration: Tensor) -> Dict[int, Tensor]:
    """Exact lower-star reduction over Z2 with piecewise gradients in values.

    Cell ordering is a discrete stratum decision and is intentionally detached;
    once the order/pairing is fixed, birth and death coordinates retain their
    autograd connection to ``vertex_filtration``.
    """

    device, dtype = vertex_filtration.device, vertex_filtration.dtype
    cells: List[Tuple[float, int, Tuple[int, ...], Tensor]] = []
    for i in range(complex_.num_vertices):
        value = vertex_filtration[i]
        cells.append((float(value.detach()), 0, (i,), value))
    for edge in complex_.edges.tolist():
        value = vertex_filtration[edge].max()
        cells.append((float(value.detach()), 1, tuple(edge), value))
    for face in complex_.faces.tolist():
        value = vertex_filtration[face].max()
        cells.append((float(value.detach()), 2, tuple(face), value))
    cells.sort(key=lambda item: (item[0], item[1], item[2]))
    cell_index = {
        (dimension, tuple(sorted(vertices))): i
        for i, (_, dimension, vertices, _) in enumerate(cells)
    }
    boundaries: List[int] = []
    for _, dimension, vertices, _ in cells:
        bits = 0
        if dimension == 1:
            for vertex in vertices:
                bits |= 1 << cell_index[(0, (vertex,))]
        elif dimension == 2:
            i, j, k = vertices
            for edge in ((i, j), (j, k), (k, i)):
                bits |= 1 << cell_index[(1, tuple(sorted(edge)))]
        boundaries.append(bits)
    reduced: List[int] = [0] * len(cells)
    low_to_column: Dict[int, int] = {}
    paired_births: set[int] = set()
    diagrams: Dict[int, List[Tensor]] = {0: [], 1: [], 2: []}
    for column_index, boundary in enumerate(boundaries):
        value = boundary
        while value:
            low = value.bit_length() - 1
            owner = low_to_column.get(low)
            if owner is None:
                break
            value ^= reduced[owner]
        reduced[column_index] = value
        if value:
            low = value.bit_length() - 1
            low_to_column[low] = column_index
            paired_births.add(low)
            birth = cells[low]
            death = cells[column_index]
            diagrams[birth[1]].append(torch.stack((birth[3], death[3])))
    if cells:
        maximum = torch.stack([cell[3] for cell in cells]).max()
    else:
        maximum = vertex_filtration.new_tensor(1.0)
    cap = maximum + torch.maximum(
        maximum.new_tensor(1.0e-6),
        0.05 * torch.maximum(maximum.new_tensor(1.0), maximum.abs()),
    )
    for index, (_, dimension, _, filtration) in enumerate(cells):
        if reduced[index] == 0 and index not in paired_births:
            diagrams[dimension].append(torch.stack((filtration, cap)))
    return {
        dimension: torch.stack(values) if values else torch.empty((0, 2), dtype=dtype, device=device)
        for dimension, values in diagrams.items()
    }


def persistence_wasserstein(left: Tensor, right: Tensor, order: int = 2) -> Tensor:
    """Exact finite-diagram Wasserstein assignment with diagonal matches."""

    n, m = left.shape[0], right.shape[0]
    if n == 0 and m == 0:
        return left.new_zeros(())
    size = n + m
    large = left.new_tensor(1.0e12)
    cost = left.new_full((size, size), large)
    if n and m:
        cost[:n, :m] = torch.cdist(left, right, p=2).pow(order)
    if n:
        diagonal_left = ((left[:, 1] - left[:, 0]).abs() / sqrt(2.0)).pow(order)
        cost[torch.arange(n), m + torch.arange(n)] = diagonal_left
    if m:
        diagonal_right = ((right[:, 1] - right[:, 0]).abs() / sqrt(2.0)).pow(order)
        cost[n + torch.arange(m), torch.arange(m)] = diagonal_right
    if n and m:
        cost[n:, m:] = 0
    # Selection is discrete, so the exact CPU Hungarian assignment does not
    # interrupt a required gradient path.
    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
    assignment = cost[torch.as_tensor(rows, device=cost.device), torch.as_tensor(columns, device=cost.device)]
    return assignment.sum().pow(1.0 / order)


def persistence_critical_occupancy_thresholds(
    diagrams: Mapping[int, Tensor],
    maximum_count: int,
    minimum_lifetime: float = 0.0,
) -> List[float]:
    """Return deterministic lower-star event thresholds, longest lived first.

    A lower-star persistence birth or finite death occurs at a vertex
    filtration value.  For filtration ``f=1-p_occ``, crossing that event is
    exactly an occupancy support change capable of changing homology.  Both
    endpoints of sufficiently persistent intervals are therefore stronger
    proposal cuts than distribution quantiles.  Endpoint selection is a
    detached discrete operation; candidate energies remain differentiable
    after the stratum is fixed.
    """

    if maximum_count < 0 or minimum_lifetime < 0:
        raise ValueError("threshold count and lifetime must be non-negative")
    ranked: List[Tuple[float, int, int, float]] = []
    for dimension in sorted(diagrams):
        diagram = diagrams[dimension]
        if diagram.ndim != 2 or diagram.shape[-1] != 2:
            raise ValueError("each persistence diagram must have shape [N,2]")
        for birth, death in diagram.detach().cpu().tolist():
            lifetime = float(death - birth)
            if lifetime < minimum_lifetime:
                continue
            for endpoint_index, filtration in enumerate((birth, death)):
                threshold = 1.0 - float(filtration)
                if 0.0 < threshold < 1.0:
                    ranked.append(
                        (-lifetime, int(dimension), endpoint_index, threshold)
                    )
    ranked.sort()
    selected: List[float] = []
    for _, _, _, threshold in ranked:
        if any(abs(threshold - retained) <= 1.0e-8 for retained in selected):
            continue
        selected.append(threshold)
        if len(selected) >= maximum_count:
            break
    return selected


def _orient_faces_consistently(
    faces: List[Tuple[int, int, int]],
) -> tuple[List[Tuple[int, int, int]], bool]:
    """Solve the Z2 orientation constraints of every face component."""

    edge_faces: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for face_index, face in enumerate(faces):
        for a, b in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            key = (min(a, b), max(a, b))
            edge_faces.setdefault(key, []).append(
                (face_index, 1 if a < b else -1)
            )
    if any(len(incident) > 2 for incident in edge_faces.values()):
        return faces, False
    adjacency: List[List[Tuple[int, int]]] = [[] for _ in faces]
    for incident in edge_faces.values():
        if len(incident) != 2:
            continue
        (left, left_direction), (right, right_direction) = incident
        relative_flip = -left_direction * right_direction
        adjacency[left].append((right, relative_flip))
        adjacency[right].append((left, relative_flip))
    flip = [0 for _ in faces]
    for root in range(len(faces)):
        if flip[root] != 0:
            continue
        flip[root] = 1
        stack = [root]
        while stack:
            left = stack.pop()
            for right, relative in adjacency[left]:
                required = relative * flip[left]
                if flip[right] == 0:
                    flip[right] = required
                    stack.append(right)
                elif flip[right] != required:
                    return faces, False
    oriented = [
        face if sign > 0 else (face[0], face[2], face[1])
        for face, sign in zip(faces, flip)
    ]
    return oriented, True


def _greedy_orientable_manifold_faces(
    proposals: Sequence[Tuple[float, Tuple[int, int, int]]],
) -> List[Tuple[int, int, int]]:
    """Select a deterministic high-quality orientable 2-subcomplex.

    Edge incidence and orientability are hereditary only under carefully
    ordered cell insertion.  Building every incidence-valid face and deleting
    the *entire* complex after one late orientation contradiction discards
    otherwise admissible surface components.  This routine instead maintains
    the Z2 face-orientation constraints with a parity union-find and rejects
    only the face that would make the selected complex non-orientable.

    ``parity[i]`` is the orientation sign of face/component ``i`` relative to
    its union-find parent.  For two incident faces, opposite traversal of their
    shared edge gives the constraint

    ``s_right = -direction_left * direction_right * s_left``.
    """

    parent: List[int] = []
    parity: List[int] = []
    faces: List[Tuple[int, int, int]] = []
    edge_faces: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    def find(index: int) -> Tuple[int, int]:
        root = index
        sign = 1
        while parent[root] != root:
            sign *= parity[root]
            root = parent[root]
        # Compression is safe because ``sign`` is the parity from ``index`` to
        # the root.  The small Python structure is a detached discrete stratum
        # decision and never participates in autograd.
        parent[index] = root
        parity[index] = sign
        return root, sign

    for _, face in sorted(proposals, reverse=True):
        directed_edges = (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        )
        face_edges = [(min(a, b), max(a, b)) for a, b in directed_edges]
        if any(len(edge_faces.get(edge, ())) >= 2 for edge in face_edges):
            continue

        # Required sign of the new face relative to each existing component.
        requirements: Dict[int, int] = {}
        admissible = True
        for (a, b), edge in zip(directed_edges, face_edges):
            incident = edge_faces.get(edge)
            if not incident:
                continue
            existing_face, existing_direction = incident[0]
            root, existing_sign = find(existing_face)
            new_direction = 1 if a < b else -1
            required = -existing_direction * new_direction * existing_sign
            previous = requirements.get(root)
            if previous is not None and previous != required:
                admissible = False
                break
            requirements[root] = required
        if not admissible:
            continue

        new_index = len(faces)
        parent.append(new_index)
        parity.append(1)
        if requirements:
            ordered = sorted(requirements.items())
            base_root, new_to_base = ordered[0]
            parent[new_index] = base_root
            parity[new_index] = new_to_base
            for root, new_to_root in ordered[1:]:
                # new = new_to_base * base = new_to_root * root
                # therefore root = new_to_root * new_to_base * base.
                parent[root] = base_root
                parity[root] = new_to_root * new_to_base

        faces.append(face)
        for (a, b), edge in zip(directed_edges, face_edges):
            direction = 1 if a < b else -1
            edge_faces.setdefault(edge, []).append((new_index, direction))

    oriented: List[Tuple[int, int, int]] = []
    for face_index, face in enumerate(faces):
        _, sign = find(face_index)
        oriented.append(face if sign > 0 else (face[0], face[2], face[1]))
    return oriented


def _surface_complex(
    atlas: PersistentOctreeAtlas,
    keep: Tensor,
    minimum_area: float,
    planarity_cosine: float,
) -> SimplicialComplex:
    active_edge, active_nodes = active_adjacency(atlas)
    nonself = active_edge[0] != active_edge[1]
    edges_all = active_edge[:, nonself].transpose(0, 1)
    keep_index = torch.nonzero(keep, as_tuple=False).flatten()
    old_to_new = torch.full((active_nodes.numel(),), -1, dtype=torch.int64, device=active_nodes.device)
    old_to_new[keep_index] = torch.arange(keep_index.numel(), device=active_nodes.device)
    valid_edge = keep[edges_all[:, 0]] & keep[edges_all[:, 1]]
    edges = old_to_new[edges_all[valid_edge]]
    edges = torch.sort(edges, dim=-1).values
    if edges.numel():
        linear = torch.unique(edges[:, 0] * keep_index.numel() + edges[:, 1], sorted=True)
        edges = torch.stack(
            (torch.div(linear, keep_index.numel(), rounding_mode="floor"), linear.remainder(keep_index.numel())), dim=-1
        )
    else:
        edges = torch.empty(0, 2, dtype=torch.int64, device=active_nodes.device)
    neighbors: List[set[int]] = [set() for _ in range(keep_index.numel())]
    for i, j in edges.tolist():
        neighbors[i].add(j)
        neighbors[j].add(i)
    nodes = active_nodes[keep_index]
    positions = atlas.chart_centers[nodes]
    normals = atlas.chart_frames[nodes, :, 2]
    proposals: List[Tuple[float, Tuple[int, int, int]]] = []
    for i in range(len(neighbors)):
        for j in sorted(x for x in neighbors[i] if x > i):
            for k in sorted(x for x in neighbors[i].intersection(neighbors[j]) if x > j):
                cross = torch.linalg.cross(positions[j] - positions[i], positions[k] - positions[i])
                area2 = torch.linalg.vector_norm(cross)
                if float(area2) <= 2.0 * minimum_area:
                    continue
                face_normal = cross / area2
                reference = F.normalize(normals[i] + normals[j] + normals[k], dim=0)
                cosine = torch.dot(face_normal, reference)
                if float(cosine.abs()) < planarity_cosine:
                    continue
                face = (i, k, j) if float(cosine) < 0 else (i, j, k)
                proposals.append((float(area2), face))
    # A greedy maximum-quality, parity-constrained 2-manifold subcomplex
    # prevents clique tetrahedra from assigning more than two faces to an edge
    # and rejects only orientation-conflicting cells.
    faces = _greedy_orientable_manifold_faces(proposals)
    if not faces:
        return SimplicialComplex(
            atlas_node_index=nodes.new_empty(0),
            edges=torch.empty(0, 2, dtype=torch.int64, device=nodes.device),
            faces=torch.empty(0, 3, dtype=torch.int64, device=nodes.device),
        )
    used_vertices = sorted({vertex for face in faces for vertex in face})
    reindex = {old: new for new, old in enumerate(used_vertices)}
    faces = [tuple(reindex[vertex] for vertex in face) for face in faces]
    used_edges = sorted(
        {
            edge
            for face in faces
            for edge in (
                (min(face[0], face[1]), max(face[0], face[1])),
                (min(face[1], face[2]), max(face[1], face[2])),
                (min(face[2], face[0]), max(face[2], face[0])),
            )
        }
    )
    return SimplicialComplex(
        atlas_node_index=nodes[torch.tensor(used_vertices, dtype=torch.int64, device=nodes.device)],
        edges=torch.tensor(used_edges, dtype=torch.int64, device=nodes.device).reshape(-1, 2),
        faces=torch.tensor(faces, dtype=torch.int64, device=nodes.device).reshape(-1, 3),
    )


class TopologySelector(nn.Module):
    """Propose a finite structured distribution and select one topology stratum."""

    def __init__(self, config: TopologySelectorConfig = TopologySelectorConfig()) -> None:
        super().__init__()
        self.config = config
        self.raw_evidence_weight = nn.Parameter(torch.tensor(float(config.evidence_weight)).log())
        self.raw_persistence_weight = nn.Parameter(torch.tensor(float(config.persistence_weight)).log())
        self.raw_geometry_weight = nn.Parameter(torch.tensor(float(config.geometry_weight)).log())
        self.raw_complexity_weight = nn.Parameter(torch.tensor(float(config.complexity_weight)).log())
        self.raw_prior_weight = nn.Parameter(torch.tensor(float(config.prior_weight)).log())

    def forward(
        self,
        atlas: PersistentOctreeAtlas,
        occupancy_probability: Tensor,
        evidence_probability: Optional[Tensor] = None,
        shape_prior_probability: Optional[Tensor] = None,
    ) -> TopologySelection:
        active = atlas.active_indices
        if occupancy_probability.shape != active.shape:
            raise ValueError("occupancy_probability must have one value per active chart")
        probability = occupancy_probability.clamp(1.0e-6, 1.0 - 1.0e-6)
        evidence = (
            probability
            if evidence_probability is None
            else evidence_probability.clamp(1.0e-6, 1.0 - 1.0e-6)
        )
        if evidence.shape != probability.shape:
            raise ValueError("evidence_probability must have one value per active chart")
        shape_prior = None
        if shape_prior_probability is not None:
            if shape_prior_probability.shape != probability.shape:
                raise ValueError("shape prior must have one probability per active chart")
            shape_prior = shape_prior_probability.clamp(1.0e-6, 1.0 - 1.0e-6)
        reference_complex = _surface_complex(
            atlas,
            torch.ones_like(probability, dtype=torch.bool),
            self.config.minimum_triangle_area,
            self.config.triangle_planarity_cosine,
        )
        if (
            reference_complex.num_vertices < self.config.minimum_vertices
            or reference_complex.num_edges == 0
            or reference_complex.num_faces == 0
        ):
            raise RuntimeError(
                "active atlas overlap graph cannot support a non-degenerate "
                "surface complex: "
                f"active_charts={active.numel()}, "
                f"reference_vertices={reference_complex.num_vertices}, "
                f"reference_edges={reference_complex.num_edges}, "
                f"reference_faces={reference_complex.num_faces}"
            )
        if (
            not reference_complex.manifold_incidence_valid()
            or not reference_complex.orientation_consistent()
        ):
            raise RuntimeError(
                "all-support atlas reference complex violates the hard "
                "manifold/orientation admissibility conditions"
            )
        global_to_active = torch.full(
            (atlas.num_nodes,), -1, dtype=torch.int64, device=active.device
        )
        global_to_active[active] = torch.arange(active.numel(), device=active.device)
        reference_index = global_to_active[reference_complex.atlas_node_index]
        reference_filtration = 1.0 - evidence[reference_index]
        reference_persistence = persistent_homology(reference_complex, reference_filtration)
        proposal_filtration = 1.0 - probability[reference_index]
        proposal_persistence = persistent_homology(
            reference_complex, proposal_filtration
        )
        candidates: List[TopologyCandidate] = []
        seen_complexes: set[Tuple[Tuple[int, ...], Tuple[Tuple[int, int, int], ...]]] = set()
        quantiles = probability.detach().new_tensor(
            self.config.adaptive_threshold_quantiles
        )
        if quantiles.numel() and bool(
            torch.any((quantiles <= 0) | (quantiles >= 1))
        ):
            raise ValueError("adaptive topology quantiles must lie in (0,1)")
        adaptive_thresholds = (
            torch.quantile(probability.detach(), quantiles).tolist()
            if quantiles.numel()
            else []
        )
        critical_thresholds = persistence_critical_occupancy_thresholds(
            proposal_persistence,
            self.config.maximum_persistence_thresholds,
            self.config.minimum_persistence_lifetime,
        )
        # The maximal observed/prior atlas support is the terminal filtration
        # stratum.  It must be proposed explicitly: quantiles and fixed
        # thresholds can all remove the overlap triangles when occupancy is
        # diffuse, even though the all-support atlas is a valid surface.  A
        # detached threshold strictly below min(p) represents that endpoint
        # without fabricating occupancy or bypassing structured selection.
        support_threshold = 0.5 * float(probability.detach().amin())
        threshold_records: List[Tuple[str, float]] = [
            ("support-endpoint", support_threshold)
        ]
        for source_name, values in (
            ("ph-critical", critical_thresholds),
            ("quantile", adaptive_thresholds),
            ("fixed", self.config.occupancy_thresholds),
        ):
            for value in values:
                threshold = float(value)
                if not 0.0 < threshold < 1.0:
                    continue
                if any(
                    abs(threshold - retained) <= 1.0e-8
                    for _, retained in threshold_records
                ):
                    continue
                threshold_records.append((source_name, threshold))
        rejection_counts = {
            "too_few_vertices": 0,
            "degenerate_complex": 0,
            "inadmissible_complex": 0,
            "duplicate_complex": 0,
        }
        support_sizes: List[Tuple[str, float, int]] = []
        for threshold_source, threshold in threshold_records:
            keep = probability >= threshold
            if int(keep.sum()) < self.config.minimum_vertices:
                top = torch.topk(probability, k=min(self.config.minimum_vertices, probability.numel())).indices
                keep = torch.zeros_like(keep)
                keep[top] = True
            kept_count = int(keep.sum())
            support_sizes.append((threshold_source, threshold, kept_count))
            if kept_count < self.config.minimum_vertices:
                rejection_counts["too_few_vertices"] += 1
                continue
            complex_ = (
                reference_complex
                if threshold_source == "support-endpoint"
                else _surface_complex(
                    atlas,
                    keep,
                    self.config.minimum_triangle_area,
                    self.config.triangle_planarity_cosine,
                )
            )
            if (
                complex_.num_vertices < self.config.minimum_vertices
                or complex_.num_faces == 0
                or complex_.num_edges == 0
            ):
                rejection_counts["degenerate_complex"] += 1
                continue
            if (
                not complex_.manifold_incidence_valid()
                or not complex_.orientation_consistent()
            ):
                # This is a discrete admissibility condition, not a soft loss:
                # an invalid cell complex cannot define a topology stratum for
                # the subsequent isotopy-preserving flow.
                rejection_counts["inadmissible_complex"] += 1
                continue
            complex_key = (
                tuple(int(value) for value in complex_.atlas_node_index.tolist()),
                tuple(tuple(int(value) for value in face) for face in complex_.faces.tolist()),
            )
            if complex_key in seen_complexes:
                rejection_counts["duplicate_complex"] += 1
                continue
            seen_complexes.add(complex_key)
            selected_index = global_to_active[complex_.atlas_node_index]
            selected_probability = evidence[selected_index]
            persistence = persistent_homology(complex_, 1.0 - selected_probability)
            betti = betti_numbers(complex_)
            selected_mask = torch.zeros_like(keep)
            selected_mask[selected_index] = True
            evidence_energy = -torch.sum(torch.log(evidence[selected_mask])) - torch.sum(
                torch.log1p(-evidence[~selected_mask])
            )
            persistence_energy = sum(
                persistence_wasserstein(persistence[dimension], reference_persistence[dimension], self.config.persistence_order)
                for dimension in range(3)
            )
            positions = atlas.chart_centers[complex_.atlas_node_index]
            face = complex_.faces
            cross = torch.linalg.cross(positions[face[:, 1]] - positions[face[:, 0]], positions[face[:, 2]] - positions[face[:, 0]])
            area = 0.5 * torch.linalg.vector_norm(cross, dim=-1)
            geometry_energy = torch.mean(1.0 / area.clamp_min(self.config.minimum_triangle_area))
            complexity = complex_.num_vertices + complex_.num_edges + complex_.num_faces
            complexity_energy = probability.new_tensor(float(complexity))
            prior_energy = probability.new_zeros(())
            if shape_prior is not None:
                prior_energy = -torch.sum(
                    torch.where(
                        selected_mask,
                        torch.log(shape_prior),
                        torch.log1p(-shape_prior),
                    )
                )
            total = (
                torch.exp(self.raw_evidence_weight) * evidence_energy
                + torch.exp(self.raw_persistence_weight) * persistence_energy
                + torch.exp(self.raw_geometry_weight) * geometry_energy
                + torch.exp(self.raw_complexity_weight) * complexity_energy
                + torch.exp(self.raw_prior_weight) * prior_energy
                + self.config.boundary_weight * complex_.boundary_edge_count()
            )
            candidates.append(
                TopologyCandidate(
                    identifier=f"{threshold_source}-occ-{threshold:.4f}-b{betti[0]}-{betti[1]}-{betti[2]}",
                    complex=complex_,
                    betti=betti,
                    persistence=persistence,
                    evidence_energy=evidence_energy,
                    persistence_energy=persistence_energy,
                    geometry_energy=geometry_energy,
                    prior_energy=prior_energy,
                    total_energy=total,
                    manifold_incidence_valid=complex_.manifold_incidence_valid(),
                    orientation_consistent=complex_.orientation_consistent(),
                    boundary_edge_count=complex_.boundary_edge_count(),
                )
            )
            if len(candidates) >= self.config.maximum_candidates:
                break
        if not candidates:
            probability_range = (
                float(probability.detach().amin()),
                float(probability.detach().amax()),
            )
            raise RuntimeError(
                "topology proposal produced no non-degenerate surface complex: "
                f"active_charts={active.numel()}, "
                f"probability_range={probability_range}, "
                f"reference=(V={reference_complex.num_vertices},"
                f"E={reference_complex.num_edges},F={reference_complex.num_faces}), "
                f"rejections={rejection_counts}, supports={support_sizes}"
            )
        energy = torch.stack([candidate.total_energy for candidate in candidates])
        distribution = torch.softmax(-energy / self.config.temperature, dim=0)
        selected = int(torch.argmin(energy).item())
        return TopologySelection(candidates, distribution, selected)


__all__ = [
    "SimplicialComplex",
    "TopologyCandidate",
    "TopologySelection",
    "TopologySelector",
    "TopologySelectorConfig",
    "betti_numbers",
    "persistence_wasserstein",
    "persistent_homology",
    "persistence_critical_occupancy_thresholds",
]

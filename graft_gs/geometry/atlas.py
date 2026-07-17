"""Persistent adaptive octree surface atlas.

This module implements the discrete/continuous state boundary required by
GRAFT-GS.  Octree mutation is deliberately non-differentiable and occurs only
between continuous optimization stages.  Chart centers, frames, curvature,
and all subsequent manifold state remain ordinary high-precision PyTorch
tensors and can participate in differentiable computation.

Conventions
-----------
* World coordinates are right handed.
* ``chart_frames[i]`` has columns ``(t1, t2, n)`` and lies in SO(3).
* Morton coordinates use x/y/z bits in positions 3b/(3b+1)/(3b+2).
* Nodes are persistent: a split deactivates a parent and appends/reactivates
  children; it never deletes or renumbers an existing node.
* ``active`` denotes the current leaf complex.  Inactive ancestors retain
  their fitted charts and statistics for multilevel losses and checkpointing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AtlasConfig:
    """Numerical and refinement policy for :class:`PersistentOctreeAtlas`."""

    base_level: int = 4
    max_level: int = 10
    root_padding: float = 0.05
    min_root_side: float = 1.0e-4
    min_points_per_chart: int = 6
    curvature_ridge: float = 1.0e-8
    frame_epsilon: float = 1.0e-10
    chart_radius_scale: float = 0.72
    overlap_scale: float = 1.05
    tau_geo: float = 0.08
    tau_curv: float = 0.35
    tau_occ: float = 0.55
    tau_repr: float = 2.0
    enforce_two_to_one: bool = True
    morton_bits: int = 20

    def __post_init__(self) -> None:
        if not 0 <= self.base_level <= self.max_level:
            raise ValueError("base_level must lie in [0, max_level]")
        if self.max_level > self.morton_bits:
            raise ValueError("max_level exceeds the configured Morton capacity")
        if self.root_padding < 0.0:
            raise ValueError("root_padding must be non-negative")
        if self.min_points_per_chart < 1:
            raise ValueError("min_points_per_chart must be positive")


@dataclass(frozen=True)
class AtlasValidation:
    """Measured atlas invariants; ``valid`` is true iff every margin passes."""

    valid: bool
    consistent_tensor_lengths: bool
    nonnegative_measures: bool
    unique_keys: bool
    valid_parents: bool
    active_are_leaves: bool
    balanced_two_to_one: bool
    symmetric_adjacency: bool
    max_frame_orthogonality_error: float
    min_frame_determinant: float
    min_chart_immersion_eigenvalue: float
    min_active_side: float


def morton_encode(xyz: Tensor, bits: int = 20) -> Tensor:
    """Encode non-negative integer ``[..., 3]`` coordinates as int64 Morton keys."""

    if xyz.shape[-1] != 3:
        raise ValueError("xyz must have final dimension 3")
    xyz = xyz.to(torch.int64)
    if torch.any(xyz < 0) or torch.any(xyz >= (1 << bits)):
        raise ValueError("Morton coordinate is outside the configured bit range")
    code = torch.zeros(xyz.shape[:-1], dtype=torch.int64, device=xyz.device)
    for bit in range(bits):
        code |= ((xyz[..., 0] >> bit) & 1) << (3 * bit)
        code |= ((xyz[..., 1] >> bit) & 1) << (3 * bit + 1)
        code |= ((xyz[..., 2] >> bit) & 1) << (3 * bit + 2)
    return code


def morton_decode(code: Tensor, bits: int = 20) -> Tensor:
    """Inverse of :func:`morton_encode`, returning int64 ``[..., 3]``."""

    code = code.to(torch.int64)
    xyz = torch.zeros((*code.shape, 3), dtype=torch.int64, device=code.device)
    for bit in range(bits):
        xyz[..., 0] |= ((code >> (3 * bit)) & 1) << bit
        xyz[..., 1] |= ((code >> (3 * bit + 1)) & 1) << bit
        xyz[..., 2] |= ((code >> (3 * bit + 2)) & 1) << bit
    return xyz


def _scatter_sum(values: Tensor, index: Tensor, size: int) -> Tensor:
    out = values.new_zeros((size, *values.shape[1:]))
    if values.numel() > 0:
        out.index_add_(0, index, values)
    return out


def _right_handed_pca_frames(covariance: Tensor, eps: float) -> Tensor:
    """Return stable SO(3) frames with the smallest-variance axis as normal."""

    eye = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
    evals, evecs = torch.linalg.eigh(covariance + eps * eye)
    normal = evecs[..., :, 0]
    tangent_1 = evecs[..., :, 2]
    tangent_2 = torch.linalg.cross(normal, tangent_1, dim=-1)
    tangent_2 = torch.nn.functional.normalize(tangent_2, dim=-1, eps=eps)
    tangent_1 = torch.linalg.cross(tangent_2, normal, dim=-1)
    tangent_1 = torch.nn.functional.normalize(tangent_1, dim=-1, eps=eps)
    normal = torch.nn.functional.normalize(normal, dim=-1, eps=eps)
    frame = torch.stack((tangent_1, tangent_2, normal), dim=-1)
    # The cross-product construction is right handed; the branch protects
    # against degenerate numerical eigenspaces without changing the normal.
    det = torch.linalg.det(frame)
    frame = frame.clone()
    frame[..., :, 1] = torch.where(
        (det < 0)[..., None], -frame[..., :, 1], frame[..., :, 1]
    )
    return frame


class PersistentOctreeAtlas(nn.Module):
    """Persistent multiresolution octree whose active leaves carry surface charts.

    Use :meth:`from_evidence` for initialization, :meth:`refine` only between
    optimizer stages, and :meth:`evaluate_chart`/``chart_jacobian`` inside the
    differentiable geometry path.
    """

    _PERSISTENT_TENSORS = (
        "levels",
        "morton_codes",
        "parent",
        "child_slot",
        "active",
        "cell_centers",
        "cell_sides",
        "chart_centers",
        "chart_frames",
        "chart_covariance",
        "curvature",
        "chart_radii",
        "evidence_mass",
        "prior_mass",
        "prior_mass_variance",
        "point_count",
        "prior_point_count",
    )

    def __init__(self, config: AtlasConfig, root_min: Tensor, root_max: Tensor) -> None:
        super().__init__()
        self.config = config
        dtype = root_min.dtype
        device = root_min.device
        self.register_buffer("root_min", root_min.reshape(3).clone())
        self.register_buffer("root_max", root_max.reshape(3).clone())
        self.register_buffer("levels", torch.empty(0, dtype=torch.int16, device=device))
        self.register_buffer("morton_codes", torch.empty(0, dtype=torch.int64, device=device))
        self.register_buffer("parent", torch.empty(0, dtype=torch.int64, device=device))
        self.register_buffer("child_slot", torch.empty(0, dtype=torch.int8, device=device))
        self.register_buffer("active", torch.empty(0, dtype=torch.bool, device=device))
        self.register_buffer("cell_centers", torch.empty(0, 3, dtype=dtype, device=device))
        self.register_buffer("cell_sides", torch.empty(0, dtype=dtype, device=device))
        self.register_buffer("chart_centers", torch.empty(0, 3, dtype=dtype, device=device))
        self.register_buffer("chart_frames", torch.empty(0, 3, 3, dtype=dtype, device=device))
        self.register_buffer("chart_covariance", torch.empty(0, 3, 3, dtype=dtype, device=device))
        self.register_buffer("curvature", torch.empty(0, 2, 2, dtype=dtype, device=device))
        self.register_buffer("chart_radii", torch.empty(0, dtype=dtype, device=device))
        self.register_buffer("evidence_mass", torch.empty(0, dtype=dtype, device=device))
        self.register_buffer("prior_mass", torch.empty(0, dtype=dtype, device=device))
        self.register_buffer("prior_mass_variance", torch.empty(0, dtype=dtype, device=device))
        self.register_buffer("point_count", torch.empty(0, dtype=torch.int64, device=device))
        self.register_buffer("prior_point_count", torch.empty(0, dtype=torch.int64, device=device))
        self.register_buffer("edge_index", torch.empty(2, 0, dtype=torch.int64, device=device))
        self.register_buffer("overlap_rotation", torch.empty(0, 3, 3, dtype=dtype, device=device))
        self.register_buffer("overlap_translation", torch.empty(0, 3, dtype=dtype, device=device))

    def get_extra_state(self) -> Mapping[str, object]:
        return {"config": asdict(self.config), "format_version": 4}

    def set_extra_state(self, state: Mapping[str, object]) -> None:
        config = state.get("config")
        if isinstance(config, Mapping):
            self.config = AtlasConfig(**dict(config))

    def checkpoint_payload(self) -> Mapping[str, object]:
        """Serialize variable-size persistent state without shape assumptions."""

        names = self._PERSISTENT_TENSORS + (
            "edge_index",
            "overlap_rotation",
            "overlap_translation",
        )
        return {
            "format_version": 4,
            "config": asdict(self.config),
            "root_min": self.root_min.detach().cpu(),
            "root_max": self.root_max.detach().cpu(),
            "tensors": {name: getattr(self, name).detach().cpu() for name in names},
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, object],
        device: Optional[torch.device | str] = None,
    ) -> "PersistentOctreeAtlas":
        """Restore exact Morton identity and all continuous chart state."""

        format_version = int(payload.get("format_version", -1))
        if format_version not in {2, 3, 4}:
            raise ValueError("unsupported persistent atlas checkpoint format")
        config = payload.get("config")
        tensors = payload.get("tensors")
        if not isinstance(config, Mapping) or not isinstance(tensors, Mapping):
            raise ValueError("atlas checkpoint is missing config or tensor state")
        move = lambda value: torch.as_tensor(value).to(device) if device is not None else torch.as_tensor(value)
        root_min = move(payload["root_min"])
        root_max = move(payload["root_max"])
        atlas = cls(AtlasConfig(**dict(config)), root_min, root_max)
        for name, value in tensors.items():
            if not hasattr(atlas, name):
                raise ValueError(f"unknown atlas checkpoint tensor {name!r}")
            setattr(atlas, name, move(value))
        if format_version == 2:
            # Version 2 predates typed hidden-support statistics. Preserve its
            # exact observed atlas and mark every node as having no prior.
            atlas.prior_mass = atlas.evidence_mass.new_zeros(atlas.num_nodes)
            atlas.prior_point_count = atlas.point_count.new_zeros(atlas.num_nodes)
        if format_version in {2, 3}:
            atlas.prior_mass_variance = atlas.evidence_mass.new_zeros(atlas.num_nodes)
        inconsistent = {
            name: int(getattr(atlas, name).shape[0])
            for name in atlas._PERSISTENT_TENSORS
            if getattr(atlas, name).shape[0] != atlas.num_nodes
        }
        if inconsistent:
            raise ValueError(
                f"atlas checkpoint has inconsistent persistent tensor lengths: {inconsistent}"
            )
        validation = atlas.validate()
        if not validation.valid:
            raise ValueError(f"restored atlas violates structural invariants: {validation}")
        return atlas

    @property
    def root_side(self) -> Tensor:
        return (self.root_max - self.root_min).max()

    @property
    def num_nodes(self) -> int:
        return int(self.levels.numel())

    @property
    def active_indices(self) -> Tensor:
        return torch.nonzero(self.active, as_tuple=False).flatten()

    @property
    def num_active(self) -> int:
        return int(self.active.sum().item())

    @staticmethod
    def root_bounds_from_positions(
        positions: Tensor,
        config: Optional[AtlasConfig] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Deterministically infer the cubic world root used by initialization."""

        if positions.ndim != 2 or positions.shape[-1] != 3 or positions.shape[0] == 0:
            raise ValueError("positions must be a non-empty [M,3] tensor")
        config = config or AtlasConfig()
        p_min, p_max = positions.amin(0), positions.amax(0)
        center = 0.5 * (p_min + p_max)
        side = torch.clamp((p_max - p_min).max(), min=config.min_root_side)
        side = side * (1.0 + 2.0 * config.root_padding)
        return center - 0.5 * side, center + 0.5 * side

    @classmethod
    def from_evidence(
        cls,
        positions: Tensor,
        mass: Optional[Tensor] = None,
        config: Optional[AtlasConfig] = None,
        root_bounds: Optional[Tuple[Tensor, Tensor]] = None,
        prior_positions: Optional[Tensor] = None,
        prior_mass: Optional[Tensor] = None,
        prior_mass_variance: Optional[Tensor] = None,
    ) -> "PersistentOctreeAtlas":
        """Build all occupied ancestors through ``base_level`` from particles.

        ``positions`` and ``mass`` are not detached during chart fitting, so the
        initialized continuous chart state preserves the identity gradient path
        from later rendering losses to the geometric evidence.
        """

        if positions.ndim != 2 or positions.shape[-1] != 3 or positions.shape[0] == 0:
            raise ValueError("positions must be a non-empty [M, 3] tensor")
        if not positions.dtype.is_floating_point:
            raise TypeError("positions must use a floating point dtype")
        config = config or AtlasConfig()
        mass = torch.ones(positions.shape[0], dtype=positions.dtype, device=positions.device) if mass is None else mass
        mass = mass.reshape(-1).to(dtype=positions.dtype, device=positions.device)
        if mass.shape[0] != positions.shape[0] or torch.any(mass < 0) or not bool(torch.any(mass > 0)):
            raise ValueError("mass must be non-negative and match positions")

        if (prior_positions is None) != (prior_mass is None):
            raise ValueError("prior_positions and prior_mass must be provided together")
        if prior_positions is None and prior_mass_variance is not None:
            raise ValueError("prior_mass_variance requires prior support")
        if prior_positions is None:
            prior_positions = positions.new_empty((0, 3))
            prior_mass = positions.new_empty((0,))
            prior_mass_variance = positions.new_empty((0,))
        else:
            prior_positions = prior_positions.to(dtype=positions.dtype, device=positions.device)
            prior_mass = prior_mass.reshape(-1).to(dtype=positions.dtype, device=positions.device)
            prior_mass_variance = (
                torch.zeros_like(prior_mass)
                if prior_mass_variance is None
                else prior_mass_variance.reshape(-1).to(
                    dtype=positions.dtype, device=positions.device
                )
            )
            if (
                prior_positions.ndim != 2
                or prior_positions.shape[1] != 3
                or prior_positions.shape[0] != prior_mass.shape[0]
                or torch.any(prior_mass <= 0)
                or prior_mass_variance.shape != prior_mass.shape
                or torch.any(prior_mass_variance < 0)
            ):
                raise ValueError("prior support must contain [P,3] positions and positive [P] mass")

        if root_bounds is None:
            root_min, root_max = cls.root_bounds_from_positions(positions, config)
        else:
            root_min, root_max = root_bounds
            root_min = root_min.to(dtype=positions.dtype, device=positions.device)
            root_max = root_max.to(dtype=positions.dtype, device=positions.device)
            extent = root_max - root_min
            if torch.any(extent <= 0) or not torch.allclose(extent, extent.max().expand_as(extent)):
                raise ValueError("root_bounds must define a non-empty cube")
        if prior_positions.shape[0] and not bool(
            torch.all((prior_positions >= root_min) & (prior_positions <= root_max))
        ):
            raise ValueError("TRELLIS prior support must lie inside the persistent atlas root")

        atlas = cls(config, root_min, root_max)
        node_chunks: Dict[str, list[Tensor]] = {name: [] for name in cls._PERSISTENT_TENSORS}
        key_to_index: Dict[Tuple[int, int], int] = {}
        running = 0
        initializer_positions = torch.cat((positions, prior_positions), dim=0)
        unit = ((initializer_positions - root_min) / (root_max - root_min)).clamp(
            0.0, 1.0 - torch.finfo(positions.dtype).eps
        )

        for level in range(config.base_level + 1):
            resolution = 1 << level
            xyz = torch.floor(unit * resolution).to(torch.int64)
            codes = morton_encode(xyz, config.morton_bits)
            unique_codes, inverse = torch.unique(codes, sorted=True, return_inverse=True)
            unique_xyz = morton_decode(unique_codes, config.morton_bits)
            observed_inverse = inverse[: positions.shape[0]]
            prior_inverse = inverse[positions.shape[0] :]
            count = torch.bincount(observed_inverse, minlength=unique_codes.numel())
            prior_count = torch.bincount(prior_inverse, minlength=unique_codes.numel())
            observed_wsum = _scatter_sum(mass, observed_inverse, unique_codes.numel())
            prior_wsum = _scatter_sum(prior_mass, prior_inverse, unique_codes.numel())
            prior_variance_sum = _scatter_sum(
                prior_mass_variance, prior_inverse, unique_codes.numel()
            )
            has_observation = count > 0
            # Prior positions fit only charts with no direct geometric
            # observation. This prevents a hallucinated support sample from
            # displacing or rotating an evidence-supported chart.
            fit_prior_mass = prior_mass * (~has_observation[prior_inverse]).to(prior_mass)
            fit_mass = torch.cat((mass, fit_prior_mass), dim=0)
            fit_wsum = _scatter_sum(fit_mass, inverse, unique_codes.numel()).clamp_min(
                config.frame_epsilon
            )
            means = _scatter_sum(
                fit_mass[:, None] * initializer_positions, inverse, unique_codes.numel()
            ) / fit_wsum[:, None]
            centered = initializer_positions - means[inverse]
            cov = _scatter_sum(
                fit_mass[:, None, None] * centered[:, :, None] * centered[:, None, :],
                inverse,
                unique_codes.numel(),
            ) / fit_wsum[:, None, None]
            frames = _right_handed_pca_frames(cov, config.frame_epsilon)
            fit_count = count + prior_count * (~has_observation).to(prior_count)
            curv = atlas._fit_curvature_grouped(
                initializer_positions, fit_mass, inverse, means, frames, fit_count
            )
            side = atlas.root_side / float(resolution)
            cell_centers = root_min + (unique_xyz.to(positions.dtype) + 0.5) * side
            parents = torch.full((unique_codes.numel(),), -1, dtype=torch.int64, device=positions.device)
            slots = torch.full((unique_codes.numel(),), -1, dtype=torch.int8, device=positions.device)
            if level > 0:
                parent_codes = morton_encode(unique_xyz // 2, config.morton_bits)
                for local, (pc, q) in enumerate(zip(parent_codes.tolist(), unique_xyz.tolist())):
                    parents[local] = key_to_index[(level - 1, int(pc))]
                    slots[local] = int((q[0] & 1) | ((q[1] & 1) << 1) | ((q[2] & 1) << 2))

            for local, code in enumerate(unique_codes.tolist()):
                key_to_index[(level, int(code))] = running + local
            running += int(unique_codes.numel())
            node_chunks["levels"].append(torch.full_like(unique_codes, level, dtype=torch.int16))
            node_chunks["morton_codes"].append(unique_codes)
            node_chunks["parent"].append(parents)
            node_chunks["child_slot"].append(slots)
            node_chunks["active"].append(torch.full_like(unique_codes, level == config.base_level, dtype=torch.bool))
            node_chunks["cell_centers"].append(cell_centers)
            node_chunks["cell_sides"].append(torch.full_like(unique_codes, side, dtype=positions.dtype))
            node_chunks["chart_centers"].append(means)
            node_chunks["chart_frames"].append(frames)
            node_chunks["chart_covariance"].append(cov)
            node_chunks["curvature"].append(curv)
            node_chunks["chart_radii"].append(torch.full_like(unique_codes, config.chart_radius_scale * side, dtype=positions.dtype))
            node_chunks["evidence_mass"].append(observed_wsum)
            node_chunks["prior_mass"].append(prior_wsum)
            node_chunks["prior_mass_variance"].append(prior_variance_sum)
            node_chunks["point_count"].append(count)
            node_chunks["prior_point_count"].append(prior_count)

        for name, chunks in node_chunks.items():
            setattr(atlas, name, torch.cat(chunks, dim=0))
        atlas.rebuild_adjacency()
        return atlas

    def _fit_curvature_grouped(
        self,
        positions: Tensor,
        mass: Tensor,
        inverse: Tensor,
        means: Tensor,
        frames: Tensor,
        count: Tensor,
    ) -> Tensor:
        """Weighted quadratic Monge-patch fit ``z=.5[x y]K[x y]^T``."""

        groups = means.shape[0]
        result = positions.new_zeros((groups, 2, 2))
        eye3 = torch.eye(3, dtype=positions.dtype, device=positions.device)
        for group in range(groups):
            if int(count[group]) < self.config.min_points_per_chart:
                continue
            mask = inverse == group
            local = (positions[mask] - means[group]) @ frames[group]
            x, y, z = local.unbind(-1)
            design = torch.stack((0.5 * x.square(), x * y, 0.5 * y.square()), dim=-1)
            w = mass[mask].clamp_min(0).sqrt()[:, None]
            lhs = (design * w).transpose(0, 1) @ (design * w)
            rhs = (design * w).transpose(0, 1) @ (z[:, None] * w)
            coeff = torch.linalg.solve(lhs + self.config.curvature_ridge * eye3, rhs).flatten()
            result[group] = torch.stack((coeff[[0, 1]], coeff[[1, 2]]))
        return result

    def _key_map(self, only_active: bool = False) -> Dict[Tuple[int, int], int]:
        mask = self.active if only_active else torch.ones_like(self.active)
        indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        levels = self.levels.tolist()
        codes = self.morton_codes.tolist()
        return {(int(levels[i]), int(codes[i])): i for i in indices}

    def assign_points(self, positions: Tensor) -> Tensor:
        """Assign each point to its deepest containing active leaf.

        Points outside the root cube are assigned to the nearest chart center;
        this keeps unbalanced OT support explicit without silently clamping
        background evidence into an unrelated boundary cell.
        """

        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape [M, 3]")
        assignments = torch.full((positions.shape[0],), -1, dtype=torch.int64, device=positions.device)
        inside = torch.all((positions >= self.root_min) & (positions <= self.root_max), dim=-1)
        key_map = self._key_map(only_active=True)
        denom = self.root_max - self.root_min
        for level in sorted(set(int(x) for x in self.levels[self.active].tolist()), reverse=True):
            unresolved = inside & (assignments < 0)
            if not torch.any(unresolved):
                break
            ids = torch.nonzero(unresolved, as_tuple=False).flatten()
            unit = ((positions[ids] - self.root_min) / denom).clamp(0.0, 1.0 - torch.finfo(positions.dtype).eps)
            xyz = torch.floor(unit * float(1 << level)).to(torch.int64)
            codes = morton_encode(xyz, self.config.morton_bits).tolist()
            values = [key_map.get((level, int(code)), -1) for code in codes]
            assignments[ids] = torch.tensor(values, dtype=torch.int64, device=positions.device)

        unresolved = assignments < 0
        if torch.any(unresolved):
            active = self.active_indices
            query = positions[unresolved]
            best_distance = torch.full((query.shape[0],), torch.inf, dtype=query.dtype, device=query.device)
            best_index = torch.zeros(query.shape[0], dtype=torch.int64, device=query.device)
            for start in range(0, active.numel(), 4096):
                ids = active[start : start + 4096]
                distance = torch.cdist(query, self.chart_centers[ids])
                value, local = distance.min(dim=1)
                improve = value < best_distance
                best_distance = torch.where(improve, value, best_distance)
                best_index = torch.where(improve, ids[local], best_index)
            assignments[unresolved] = best_index
        return assignments

    def refinement_mask(
        self,
        occupancy_entropy: Optional[Tensor] = None,
        reprojection_variance: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate the four specification split criteria on active charts."""

        active = self.active_indices
        sides = self.cell_sides[active]
        # Use the measured residual covariance rather than a cell-size proxy;
        # otherwise every equal-level chart receives the same split pressure.
        kappa = torch.linalg.matrix_norm(self.curvature[active], ord=2)
        trace_covariance = self.chart_covariance[active].diagonal(dim1=-2, dim2=-1).sum(-1)
        split = trace_covariance > self.config.tau_geo * sides.square()
        split |= kappa * sides > self.config.tau_curv
        if occupancy_entropy is not None:
            if occupancy_entropy.shape != active.shape:
                raise ValueError("occupancy_entropy must have one value per active cell")
            split |= occupancy_entropy > self.config.tau_occ
        if reprojection_variance is not None:
            if reprojection_variance.shape != active.shape:
                raise ValueError("reprojection_variance must have one value per active cell")
            split |= reprojection_variance > self.config.tau_repr
        split &= self.levels[active] < self.config.max_level
        return split

    def _balanced_split_closure(self, requested: Tensor) -> Tensor:
        selected = set(int(i) for i in requested.tolist())
        if not self.config.enforce_two_to_one:
            return torch.tensor(sorted(selected), dtype=torch.int64, device=self.levels.device)
        changed = True
        while changed:
            changed = False
            for i, j in self.edge_index.transpose(0, 1).tolist():
                if i in selected and int(self.levels[j]) < int(self.levels[i]):
                    if int(self.levels[j]) < self.config.max_level and j not in selected:
                        selected.add(j)
                        changed = True
        return torch.tensor(sorted(selected), dtype=torch.int64, device=self.levels.device)

    def refine(
        self,
        positions: Tensor,
        mass: Optional[Tensor] = None,
        split_mask: Optional[Tensor] = None,
        occupancy_entropy: Optional[Tensor] = None,
        reprojection_variance: Optional[Tensor] = None,
        prior_positions: Optional[Tensor] = None,
        prior_mass: Optional[Tensor] = None,
        prior_mass_variance: Optional[Tensor] = None,
    ) -> Tensor:
        """Persistently split selected leaves and refit their active child charts.

        Returns global indices of all newly activated children. Morton keys,
        activation, and point assignment are discrete stratum choices. The
        subsequent chart fit is nevertheless differentiable conditional on
        those choices, preserving gradients to evidence positions and masses.
        """

        active_before = self.active_indices
        if split_mask is None:
            split_mask = self.refinement_mask(occupancy_entropy, reprojection_variance)
        if split_mask.shape != active_before.shape:
            raise ValueError("split_mask must have one boolean per active cell")
        requested = active_before[split_mask]
        requested = self._balanced_split_closure(requested)
        requested = requested[self.levels[requested] < self.config.max_level]
        if requested.numel() == 0:
            return requested

        existing = self._key_map(only_active=False)
        activated: list[int] = []
        append: Dict[str, list[Tensor]] = {name: [] for name in self._PERSISTENT_TENSORS}
        next_index = self.num_nodes
        for parent_idx in requested.tolist():
            level = int(self.levels[parent_idx]) + 1
            parent_xyz = morton_decode(self.morton_codes[parent_idx], self.config.morton_bits)
            children_xyz = torch.stack(
                [2 * parent_xyz + torch.tensor(o, device=parent_xyz.device) for o in product((0, 1), repeat=3)]
            )
            children_codes = morton_encode(children_xyz, self.config.morton_bits)
            self.active[parent_idx] = False
            for slot, (xyz, code) in enumerate(zip(children_xyz, children_codes)):
                key = (level, int(code))
                if key in existing:
                    child_idx = existing[key]
                    self.active[child_idx] = True
                    activated.append(child_idx)
                    continue
                side = self.root_side / float(1 << level)
                cell_center = self.root_min + (xyz.to(self.root_min.dtype) + 0.5) * side
                append["levels"].append(torch.tensor([level], dtype=torch.int16, device=self.levels.device))
                append["morton_codes"].append(code.reshape(1))
                append["parent"].append(torch.tensor([parent_idx], dtype=torch.int64, device=self.levels.device))
                append["child_slot"].append(torch.tensor([slot], dtype=torch.int8, device=self.levels.device))
                append["active"].append(torch.ones(1, dtype=torch.bool, device=self.levels.device))
                append["cell_centers"].append(cell_center.reshape(1, 3))
                append["cell_sides"].append(side.reshape(1))
                append["chart_centers"].append(cell_center.reshape(1, 3))
                append["chart_frames"].append(self.chart_frames[parent_idx].reshape(1, 3, 3))
                append["chart_covariance"].append(self.chart_covariance[parent_idx].reshape(1, 3, 3))
                append["curvature"].append(self.curvature[parent_idx].reshape(1, 2, 2))
                append["chart_radii"].append((self.config.chart_radius_scale * side).reshape(1))
                append["evidence_mass"].append(self.evidence_mass.new_zeros(1))
                append["prior_mass"].append(self.prior_mass.new_zeros(1))
                append["prior_mass_variance"].append(
                    self.prior_mass_variance.new_zeros(1)
                )
                append["point_count"].append(self.point_count.new_zeros(1))
                append["prior_point_count"].append(self.prior_point_count.new_zeros(1))
                existing[key] = next_index
                activated.append(next_index)
                next_index += 1

        for name, chunks in append.items():
            if chunks:
                setattr(self, name, torch.cat((getattr(self, name), torch.cat(chunks, dim=0)), dim=0))
        mass = torch.ones(positions.shape[0], dtype=positions.dtype, device=positions.device) if mass is None else mass
        self.fit_active_charts(
            positions,
            mass,
            prior_positions=prior_positions,
            prior_mass=prior_mass,
            prior_mass_variance=prior_mass_variance,
        )
        self.rebuild_adjacency()
        return torch.tensor(sorted(activated), dtype=torch.int64, device=self.levels.device)

    def fit_active_charts(
        self,
        positions: Tensor,
        mass: Tensor,
        prior_positions: Optional[Tensor] = None,
        prior_mass: Optional[Tensor] = None,
        prior_mass_variance: Optional[Tensor] = None,
    ) -> None:
        """Refit from evidence, using the prior only for unobserved leaves."""

        if (prior_positions is None) != (prior_mass is None):
            raise ValueError("prior_positions and prior_mass must be supplied together")
        if prior_positions is None and prior_mass_variance is not None:
            raise ValueError("prior_mass_variance requires prior support")
        if prior_positions is None:
            prior_positions = positions.new_empty((0, 3))
            prior_mass = mass.new_empty((0,))
            prior_mass_variance = mass.new_empty((0,))
        else:
            prior_positions = prior_positions.to(positions)
            prior_mass = prior_mass.reshape(-1).to(mass)
            prior_mass_variance = (
                torch.zeros_like(prior_mass)
                if prior_mass_variance is None
                else prior_mass_variance.reshape(-1).to(mass)
            )
            if (
                prior_positions.shape != (prior_mass.numel(), 3)
                or torch.any(prior_mass <= 0)
                or prior_mass_variance.shape != prior_mass.shape
                or torch.any(prior_mass_variance < 0)
            ):
                raise ValueError(
                    "prior support requires [P,3] positions, positive mass, and non-negative variance"
                )

        assignment = self.assign_points(positions)
        prior_assignment = self.assign_points(prior_positions) if prior_positions.shape[0] else assignment.new_empty(0)
        active = self.active_indices
        global_to_local = torch.full((self.num_nodes,), -1, dtype=torch.int64, device=self.levels.device)
        global_to_local[active] = torch.arange(active.numel(), device=self.levels.device)
        local = global_to_local[assignment]
        valid = local >= 0
        local = local[valid]
        observed_position = positions[valid]
        observed_mass = mass[valid].to(positions)
        prior_local = global_to_local[prior_assignment]
        prior_valid = prior_local >= 0
        prior_local = prior_local[prior_valid]
        prior_position = prior_positions[prior_valid]
        prior_weight = prior_mass[prior_valid].to(positions)
        count = torch.bincount(local, minlength=active.numel())
        prior_count = torch.bincount(prior_local, minlength=active.numel())
        wsum = _scatter_sum(observed_mass, local, active.numel())
        prior_wsum = _scatter_sum(prior_weight, prior_local, active.numel())
        prior_variance_sum = _scatter_sum(
            prior_mass_variance[prior_valid].to(positions),
            prior_local,
            active.numel(),
        )
        has_points = count > 0
        retained_prior_weight = prior_weight * (~has_points[prior_local]).to(prior_weight)
        fit_position = torch.cat((observed_position, prior_position), dim=0)
        fit_weight = torch.cat((observed_mass, retained_prior_weight), dim=0)
        fit_local = torch.cat((local, prior_local), dim=0)
        fit_wsum = _scatter_sum(fit_weight, fit_local, active.numel())
        means = _scatter_sum(
            fit_weight[:, None] * fit_position, fit_local, active.numel()
        ) / fit_wsum.clamp_min(self.config.frame_epsilon)[:, None]
        has_fit_points = (count + prior_count * (~has_points).to(prior_count)) > 0
        means = torch.where(has_fit_points[:, None], means, self.chart_centers[active])
        centered = fit_position - means[fit_local]
        cov = _scatter_sum(
            fit_weight[:, None, None] * centered[:, :, None] * centered[:, None, :],
            fit_local,
            active.numel(),
        )
        cov = cov / fit_wsum.clamp_min(self.config.frame_epsilon)[:, None, None]
        frames = _right_handed_pca_frames(cov, self.config.frame_epsilon)
        frames = torch.where(has_fit_points[:, None, None], frames, self.chart_frames[active])
        cov = torch.where(has_fit_points[:, None, None], cov, self.chart_covariance[active])
        fit_count = count + prior_count * (~has_points).to(prior_count)
        curv = self._fit_curvature_grouped(
            fit_position, fit_weight, fit_local, means, frames, fit_count
        )
        curv = torch.where(
            (fit_count >= self.config.min_points_per_chart)[:, None, None],
            curv,
            self.curvature[active],
        )
        # Functional indexed writes preserve the conditional continuous
        # autograd graph. In-place assignment to buffers created during a
        # refined forward either detaches the values or corrupts saved tensors.
        self.chart_centers = self.chart_centers.index_copy(0, active, means)
        self.chart_frames = self.chart_frames.index_copy(0, active, frames)
        self.chart_covariance = self.chart_covariance.index_copy(0, active, cov)
        self.curvature = self.curvature.index_copy(0, active, curv)
        self.evidence_mass = self.evidence_mass.index_copy(0, active, wsum)
        self.prior_mass = self.prior_mass.index_copy(0, active, prior_wsum)
        self.prior_mass_variance = self.prior_mass_variance.index_copy(
            0,
            active,
            prior_variance_sum,
        )
        self.point_count = self.point_count.index_copy(0, active, count)
        self.prior_point_count = self.prior_point_count.index_copy(
            0,
            active,
            prior_count,
        )

    @torch.no_grad()
    def rebuild_adjacency(self) -> None:
        """Build symmetric sparse overlap adjacency without a dense V×V tensor."""

        active = self.active_indices.tolist()
        active_map = self._key_map(only_active=True)
        levels = self.levels.tolist()
        codes = self.morton_codes.tolist()
        pairs: set[Tuple[int, int]] = set()
        offsets_same = list(product((-1, 0, 1), repeat=3))
        offsets_fine = list(product((-1, 0, 1, 2), repeat=3))
        for i in active:
            level = int(levels[i])
            xyz = morton_decode(torch.tensor(codes[i]), self.config.morton_bits).tolist()
            candidates: set[int] = set()
            for candidate_level in (level - 1, level, level + 1):
                if candidate_level < 0 or candidate_level > self.config.max_level:
                    continue
                if candidate_level == level:
                    bases, offsets = xyz, offsets_same
                elif candidate_level == level - 1:
                    bases, offsets = [q // 2 for q in xyz], offsets_same
                else:
                    bases, offsets = [2 * q for q in xyz], offsets_fine
                resolution = 1 << candidate_level
                for offset in offsets:
                    q = [bases[d] + offset[d] for d in range(3)]
                    if min(q) < 0 or max(q) >= resolution:
                        continue
                    code = int(morton_encode(torch.tensor(q), self.config.morton_bits))
                    j = active_map.get((candidate_level, code))
                    if j is not None and j != i:
                        candidates.add(j)
            for j in candidates:
                if j < i:
                    continue
                delta = (self.cell_centers[i] - self.cell_centers[j]).abs()
                reach = 0.5 * self.config.overlap_scale * (self.cell_sides[i] + self.cell_sides[j])
                if bool(torch.all(delta <= reach + self.config.frame_epsilon)):
                    pairs.add((i, j))

        directed = sorted([(i, j) for i, j in pairs for i, j in ((i, j), (j, i))])
        if directed:
            edge = torch.tensor(directed, dtype=torch.int64, device=self.levels.device).transpose(0, 1)
            source, target = edge
            rotation = self.chart_frames[source].transpose(-1, -2) @ self.chart_frames[target]
            translation = torch.einsum(
                "eji,ej->ei",
                self.chart_frames[source],
                self.chart_centers[target] - self.chart_centers[source],
            )
        else:
            edge = torch.empty(2, 0, dtype=torch.int64, device=self.levels.device)
            rotation = self.chart_frames.new_empty((0, 3, 3))
            translation = self.chart_centers.new_empty((0, 3))
        self.edge_index = edge
        self.overlap_rotation = rotation
        self.overlap_translation = translation

    def evaluate_chart(self, node_index: Tensor | int, xi: Tensor) -> Tensor:
        """Evaluate ``φ_i(xi)`` for local coordinates ``[..., 2]``."""

        center = self.chart_centers[node_index]
        frame = self.chart_frames[node_index]
        curvature = self.curvature[node_index]
        height = 0.5 * torch.einsum("...i,...ij,...j->...", xi, curvature, xi)
        local = torch.cat((xi, height[..., None]), dim=-1)
        return center + torch.einsum("...ij,...j->...i", frame, local)

    def chart_jacobian(self, node_index: Tensor | int, xi: Tensor) -> Tensor:
        """Analytical ``∂φ_i/∂xi`` with shape ``[..., 3, 2]``."""

        frame = self.chart_frames[node_index]
        curvature = self.curvature[node_index]
        slope = torch.einsum("...ij,...j->...i", curvature, xi)
        local_jacobian = torch.stack(
            (
                torch.stack((torch.ones_like(slope[..., 0]), torch.zeros_like(slope[..., 0]), slope[..., 0]), dim=-1),
                torch.stack((torch.zeros_like(slope[..., 1]), torch.ones_like(slope[..., 1]), slope[..., 1]), dim=-1),
            ),
            dim=-1,
        )
        return frame @ local_jacobian

    def first_fundamental_form(self, node_index: Tensor | int, xi: Tensor) -> Tensor:
        jacobian = self.chart_jacobian(node_index, xi)
        return jacobian.transpose(-1, -2) @ jacobian

    def partition_of_unity_metric(
        self,
        query: Tensor,
        node_metric: Tensor,
        node_index: Optional[Tensor] = None,
        support_scale: float = 2.0,
        chunk_size: int = 2048,
    ) -> Tensor:
        r"""Evaluate a smooth SPD metric field over overlapping charts.

        Compact ``C-infinity`` bump weights

        ``w_i(x) proportional exp(-1/(1-r_i(x)^2))`` for ``r_i<1``

        form a partition of unity. A deterministic nearest-chart fallback is
        used only outside the union of supports. A positive weighted sum of SPD
        node metrics remains SPD, so no eigenvalue repair is needed.
        """

        if query.ndim != 2 or query.shape[1] != 3:
            raise ValueError("metric queries must have shape [Q,3]")
        if support_scale <= 0 or chunk_size < 1:
            raise ValueError("metric support scale and chunk size must be positive")
        nodes = self.active_indices if node_index is None else node_index
        if nodes.ndim != 1 or nodes.dtype != torch.int64:
            raise ValueError("node_index must be an int64 vector")
        if node_metric.shape != (nodes.numel(), 3, 3):
            raise ValueError("node_metric must have shape [V,3,3]")
        if nodes.numel() == 0:
            raise ValueError("partition of unity requires at least one chart")
        if query.shape[0] == 0:
            return node_metric.new_empty((0, 3, 3))
        center = self.chart_centers[nodes]
        support = support_scale * self.chart_radii[nodes]
        output = []
        for start in range(0, query.shape[0], chunk_size):
            value = query[start : start + chunk_size]
            distance = torch.cdist(value, center)
            normalized = distance / support[None].clamp_min(1.0e-12)
            inside = normalized < 1.0
            denominator = (1.0 - normalized.square()).clamp_min(1.0e-12)
            bump = torch.where(
                inside,
                torch.exp(-1.0 / denominator),
                torch.zeros_like(normalized),
            )
            total = bump.sum(dim=-1, keepdim=True)
            nearest = torch.nn.functional.one_hot(
                distance.argmin(dim=-1),
                num_classes=nodes.numel(),
            ).to(bump)
            weight = torch.where(
                total > torch.finfo(bump.dtype).tiny,
                bump / total.clamp_min(torch.finfo(bump.dtype).tiny),
                nearest,
            )
            metric = torch.einsum("qv,vij->qij", weight, node_metric)
            output.append(0.5 * (metric + metric.transpose(-1, -2)))
        return torch.cat(output, dim=0)

    def chart_immersion_margin(self, delta_j: float = 1.0e-6, samples_per_axis: int = 3) -> Tensor:
        """Minimum sampled eigenvalue of ``JᵀJ`` minus ``delta_j`` per active chart."""

        active = self.active_indices
        grid = torch.linspace(-1.0, 1.0, samples_per_axis, dtype=self.root_min.dtype, device=self.root_min.device)
        uv = torch.cartesian_prod(grid, grid)
        radius = self.chart_radii[active]
        xi = uv[None] * radius[:, None, None]
        nodes = active[:, None].expand(-1, uv.shape[0])
        metric = self.first_fundamental_form(nodes, xi)
        return torch.linalg.eigvalsh(metric)[..., 0].amin(dim=1) - delta_j

    def validate(self, immersion_delta: float = 1.0e-6) -> AtlasValidation:
        """Measure structural assumptions used by topology-preservation claims."""

        consistent_tensor_lengths = all(
            getattr(self, name).shape[0] == self.num_nodes
            for name in self._PERSISTENT_TENSORS
        )
        nonnegative_measures = bool(
            torch.all(self.evidence_mass >= 0)
            and torch.all(self.prior_mass >= 0)
            and torch.all(self.prior_mass_variance >= 0)
            and torch.all(self.point_count >= 0)
            and torch.all(self.prior_point_count >= 0)
        )
        keys = list(zip(self.levels.tolist(), self.morton_codes.tolist()))
        unique_keys = len(keys) == len(set(keys))
        key_map = {key: i for i, key in enumerate(keys)}
        valid_parents = True
        for i in range(self.num_nodes):
            if int(self.levels[i]) == 0:
                valid_parents &= int(self.parent[i]) == -1
                continue
            xyz = morton_decode(self.morton_codes[i], self.config.morton_bits)
            pcode = int(morton_encode(xyz // 2, self.config.morton_bits))
            expected = key_map.get((int(self.levels[i]) - 1, pcode), -2)
            valid_parents &= int(self.parent[i]) == expected
        active_set = set(self.active_indices.tolist())
        active_are_leaves = not any(int(self.parent[j]) in active_set for j in range(self.num_nodes))
        balanced = all(abs(int(self.levels[i]) - int(self.levels[j])) <= 1 for i, j in self.edge_index.transpose(0, 1).tolist())
        edges = set(map(tuple, self.edge_index.transpose(0, 1).tolist()))
        symmetric = all((j, i) in edges for i, j in edges)
        active = self.active_indices
        frames = self.chart_frames[active]
        eye = torch.eye(3, dtype=frames.dtype, device=frames.device)
        orth_error = torch.linalg.matrix_norm(frames.transpose(-1, -2) @ frames - eye, ord="fro", dim=(-2, -1))
        max_orth = float(orth_error.max().item()) if orth_error.numel() else 0.0
        min_det = float(torch.linalg.det(frames).min().item()) if frames.numel() else 1.0
        margins = self.chart_immersion_margin(immersion_delta)
        min_imm = float((margins + immersion_delta).min().item()) if margins.numel() else float("inf")
        min_side = float(self.cell_sides[active].min().item()) if active.numel() else 0.0
        valid = (
            consistent_tensor_lengths
            and nonnegative_measures
            and unique_keys
            and valid_parents
            and active_are_leaves
            and balanced
            and symmetric
            and max_orth < 1.0e-5
            and min_det > 0.999
            and min_imm > immersion_delta
            and min_side > 0.0
        )
        return AtlasValidation(
            valid=valid,
            consistent_tensor_lengths=consistent_tensor_lengths,
            nonnegative_measures=nonnegative_measures,
            unique_keys=unique_keys,
            valid_parents=valid_parents,
            active_are_leaves=active_are_leaves,
            balanced_two_to_one=balanced,
            symmetric_adjacency=symmetric,
            max_frame_orthogonality_error=max_orth,
            min_frame_determinant=min_det,
            min_chart_immersion_eigenvalue=min_imm,
            min_active_side=min_side,
        )


__all__ = [
    "AtlasConfig",
    "AtlasValidation",
    "PersistentOctreeAtlas",
    "morton_decode",
    "morton_encode",
]

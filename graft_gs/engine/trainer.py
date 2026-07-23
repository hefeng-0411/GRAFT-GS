"""Six-phase native-precision DDP trainer for the A800 execution target."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from ..geometry.atlas import PersistentOctreeAtlas
from ..integration.pipeline import GraftGS, GraftGSOutput, RobustnessPerturbation
from ..integration.trellis_prior import TrellisPriorMeasure
from ..mapping.manifold_mapping import EvidenceParticles, MappingResult
from ..optimization.gradient_purification import (
    Gradient,
    GradientPurificationConfig,
    HilbertGradientPurifier,
)
from ..optimization.quantization import (
    QuantizationConfig,
    apply_equivariant_qat,
    quantization_scale_adversaries,
)
from .losses import (
    GraftGSLoss,
    LearnedPerceptualPyramid,
    LossWeights,
    distillation_loss,
    view_conditioned_objectives,
)
from .precision import NativePrecisionPolicy


class TrainingPhase(str, Enum):
    EVIDENCE_CALIBRATION = "A"
    ATLAS_AUTOENCODING = "B"
    RIEMANNIAN_FLOW = "C"
    END_TO_END = "D"
    QUANTIZATION_DISTILLATION = "E"
    TOPOLOGY_HARDENING = "F"


@dataclass(frozen=True)
class TrainerConfig:
    phase: TrainingPhase = TrainingPhase.EVIDENCE_CALIBRATION
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-2
    gradient_accumulation_steps: int = 1
    maximum_gradient_norm: float = 1.0
    gradient_purification_enabled: bool = True
    gradient_purification_maximum_views: int = 8
    gradient_consensus_cosine: float = 0.2
    gradient_consensus_relative_singular_value: float = 0.05
    gradient_artifact_relative_singular_value: float = 0.1
    gradient_weiszfeld_iterations: int = 12
    gradient_fisher_decay: float = 0.95
    gradient_fisher_damping: float = 1.0e-6
    gradient_fisher_radius: float = 1.0
    quantization_adversarial_log_scale_radius: float = 0.05
    topology_hardening_relative_margin: float = 0.1
    topology_hardening_temperature: float = 0.1
    checkpoint_every: int = 1000
    validate_every: int = 500
    log_every: int = 10
    output_directory: str = "outputs/training"
    find_unused_parameters: bool = True
    synchronize_object_atlas: bool = False
    seed: int = 17
    dataset_manifest: Optional[str] = None
    dataset_manifest_sha256: Optional[str] = None
    dataset_object_id_catalog: Optional[str] = None
    dataset_object_id_catalog_sha256: Optional[str] = None
    dataset_object_id_count: Optional[int] = None
    dataset_split: Optional[str] = None
    dataset_view_set: Optional[str] = None
    dataset_maximum_views: Optional[int] = None
    dataset_manifest_schema: Optional[str] = None
    topology_supervision_mode: Optional[str] = None
    minimum_topology_confidence: Optional[float] = None
    teacher_checkpoint: Optional[str] = None
    teacher_distillation_confidence: float = 1.0
    teacher_topology_confidence: float = 0.5
    trellis_prior_checkpoint: Optional[str] = None
    trellis_prior_samples: int = 0
    trellis_prior_sampler_steps: int = 0
    trellis_prior_strength: float = 0.0
    trellis_prior_minimum_probability: float = 0.0
    trellis_prior_uncertainty_discount: float = 0.0
    dino_relational_pseudo_supervision: bool = False
    trellis_latent_relational_pseudo_supervision: bool = False
    dino_pseudo_confidence: float = 0.0
    trellis_latent_pseudo_confidence: float = 0.0
    derive_mesh_depth_normals: bool = True
    require_mesh_depth_normals: bool = False
    mesh_supervision_view_chunk_size: int = 2
    teacher_bundle_root: Optional[str] = None
    teacher_bundle_digest: Optional[str] = None
    teacher_bundle_minimum_confidence: float = 0.0
    perceptual_checkpoint: Optional[str] = None
    perceptual_checkpoint_sha256: Optional[str] = None
    precision_backbone: str = "bfloat16"
    precision_geometric_state: str = "float32"
    precision_analytical_solve: str = "float32"
    precision_diagnostics: str = "float64"
    precision_float32_matmul: str = "highest"
    precision_allow_tf32: bool = False

    def __post_init__(self) -> None:
        NativePrecisionPolicy(
            backbone=self.precision_backbone,
            geometric_state=self.precision_geometric_state,
            analytical_solve=self.precision_analytical_solve,
            diagnostics=self.precision_diagnostics,
            float32_matmul_precision=self.precision_float32_matmul,
            allow_tf32=self.precision_allow_tf32,
        )
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.maximum_gradient_norm <= 0:
            raise ValueError("maximum_gradient_norm must be positive")
        GradientPurificationConfig(
            maximum_views=self.gradient_purification_maximum_views,
            consensus_cosine=self.gradient_consensus_cosine,
            consensus_relative_singular_value=self.gradient_consensus_relative_singular_value,
            artifact_relative_singular_value=self.gradient_artifact_relative_singular_value,
            weiszfeld_iterations=self.gradient_weiszfeld_iterations,
            fisher_decay=self.gradient_fisher_decay,
            fisher_damping=self.gradient_fisher_damping,
            fisher_radius=self.gradient_fisher_radius,
        )
        if self.quantization_adversarial_log_scale_radius < 0:
            raise ValueError("quantization scale-adversary radius must be non-negative")
        if self.topology_hardening_relative_margin < 0:
            raise ValueError("topology hardening margin must be non-negative")
        if self.topology_hardening_temperature <= 0:
            raise ValueError("topology hardening temperature must be positive")
        if not 0.0 <= self.teacher_topology_confidence <= 1.0:
            raise ValueError("teacher_topology_confidence must lie in [0,1]")
        if not 0.0 <= self.teacher_distillation_confidence <= 1.0:
            raise ValueError("teacher_distillation_confidence must lie in [0,1]")
        if self.phase is TrainingPhase.QUANTIZATION_DISTILLATION and self.teacher_checkpoint is None:
            raise ValueError("Phase E requires explicit teacher checkpoint provenance")
        if self.minimum_topology_confidence is not None and not 0.0 <= self.minimum_topology_confidence <= 1.0:
            raise ValueError("minimum_topology_confidence must lie in [0,1]")
        if self.dataset_maximum_views is not None and self.dataset_maximum_views < 1:
            raise ValueError("dataset_maximum_views must be positive when configured")
        if self.trellis_prior_checkpoint is not None:
            if self.trellis_prior_samples < 1 or self.trellis_prior_sampler_steps < 1:
                raise ValueError("active TRELLIS prior requires positive samples and steps")
            if self.trellis_prior_strength < 0 or not 0 <= self.trellis_prior_minimum_probability < 1:
                raise ValueError("TRELLIS prior strength/threshold are outside their domains")
            if self.trellis_prior_uncertainty_discount < 0:
                raise ValueError("TRELLIS prior uncertainty discount must be non-negative")
        if not 0 <= self.dino_pseudo_confidence <= 1:
            raise ValueError("DINO pseudo confidence must lie in [0,1]")
        if not 0 <= self.trellis_latent_pseudo_confidence <= 1:
            raise ValueError("TRELLIS latent pseudo confidence must lie in [0,1]")
        if self.mesh_supervision_view_chunk_size < 1:
            raise ValueError("mesh supervision view chunk size must be positive")
        if not 0 <= self.teacher_bundle_minimum_confidence <= 1:
            raise ValueError("teacher bundle minimum confidence must lie in [0,1]")
        if (self.teacher_bundle_root is None) != (self.teacher_bundle_digest is None):
            raise ValueError("teacher bundle root and digest must be configured together")
        if (self.perceptual_checkpoint is None) != (
            self.perceptual_checkpoint_sha256 is None
        ):
            raise ValueError("perceptual checkpoint path and SHA-256 must be paired")


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @classmethod
    def initialize(cls) -> "DistributedContext":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = bind_local_cuda_device()
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        return cls(rank, local_rank, world_size, device)

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def bind_local_cuda_device(*, require_cuda: bool = False) -> torch.device:
    """Bind torchrun's local device before any checkpoint allocates CUDA state.

    Loading a CUDA checkpoint before ``torch.cuda.set_device(LOCAL_RANK)`` can
    leave one process owning allocations on both logical device zero and its
    assigned rank.  That wastes memory, defeats process isolation, and can OOM
    an otherwise lightly loaded A800.  This helper is intentionally safe to
    call again when the trainer initializes the process group.
    """

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError("the requested distributed path requires CUDA")
        return torch.device("cpu")
    device_count = torch.cuda.device_count()
    if not 0 <= local_rank < device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside the {device_count} devices exposed "
            "by CUDA_VISIBLE_DEVICES"
        )
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def assert_local_cuda_allocator_ownership(device: torch.device) -> None:
    """Reject process-local allocator state on another visible CUDA device.

    This inspects only allocations owned by the current Python process; memory
    belonging to other jobs is intentionally outside its scope. A one-process-
    per-GPU DDP rank has no legitimate reason to reserve storage on a sibling
    device before wrapping its local model.
    """

    if device.type != "cuda":
        return
    local_index = device.index
    if local_index is None:
        local_index = torch.cuda.current_device()
    foreign = {
        index: {
            "allocated": int(torch.cuda.memory_allocated(index)),
            "reserved": int(torch.cuda.memory_reserved(index)),
        }
        for index in range(torch.cuda.device_count())
        if index != local_index
        and (
            torch.cuda.memory_allocated(index) != 0
            or torch.cuda.memory_reserved(index) != 0
        )
    }
    if foreign:
        raise RuntimeError(
            "this DDP process owns CUDA allocator state on non-local devices; "
            f"LOCAL_RANK={local_index}, foreign={foreign}. Bind LOCAL_RANK before "
            "loading VGGT, TRELLIS, or any CUDA checkpoint."
        )


@torch.no_grad()
def _clip_grad_norm_high_precision(
    parameters: Sequence[nn.Parameter],
    maximum_norm: float,
) -> Tensor:
    """Clip a mixed-precision gradient vector using an FP64 global norm.

    PyTorch's foreach norm follows gradient dtypes. Squaring a finite FP32
    gradient can overflow in FP32 before clipping, and BF16 reductions discard
    small components. The A800 reference instead accumulates the Euclidean norm
    in FP64, then applies one common coefficient in each gradient's dtype.
    """

    entries = [
        (parameter, parameter.grad)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not entries:
        return torch.zeros((), dtype=torch.float64)
    device = entries[0][1].device
    squared_norm = torch.zeros((), dtype=torch.float64, device=device)
    for _, gradient in entries:
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        squared_norm.add_(values.detach().to(dtype=torch.float64).square().sum())
    norm = torch.sqrt(squared_norm)
    coefficient = torch.clamp(
        norm.new_tensor(maximum_norm) / norm.clamp_min(torch.finfo(norm.dtype).tiny),
        max=1.0,
    )
    for parameter, gradient in entries:
        if gradient.is_sparse:
            gradient = gradient.coalesce()
            parameter.grad = gradient
            values = gradient.values()
        else:
            values = gradient
        values.mul_(coefficient.to(device=values.device, dtype=values.dtype))
    return norm


def _capture_rng_state(rank: int) -> dict[str, object]:
    return {
        "rank": rank,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng_state(state: Mapping[str, object], rank: int) -> None:
    if int(state.get("rank", -1)) != rank:
        raise ValueError("rank-local RNG state has invalid rank provenance")
    torch_state = state["torch"]
    if not isinstance(torch_state, Tensor):
        raise TypeError("Torch RNG state must be a tensor")
    torch.set_rng_state(torch_state.cpu())
    cuda_state = state["cuda"]
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _broadcast_discrete_exact(value: Tensor, source_rank: int) -> Tensor:
    """Broadcast discrete state through NCCL's portable int64 domain.

    PyTorch 2.4's NCCL process group rejects ``torch.int16`` (``Short``), and
    backend support for bool/int8 metadata is not a stable serialization
    contract. Octree levels, child slots, activity masks, Morton identities,
    counts, and connectivity are all exactly representable as int64. A checked
    round trip restores the original storage dtype after communication.
    """

    if value.dtype.is_floating_point or value.dtype.is_complex:
        raise TypeError("exact discrete broadcast requires a non-floating tensor")
    if value.numel() == 0:
        # Atlas metadata equality proves every rank has the same empty field;
        # avoid relying on backend-specific zero-count collective behavior.
        return value.clone()
    # ``to(dtype)`` may alias an already-int64 input. The independent buffer is
    # essential: otherwise source broadcast would overwrite the rank-local
    # value before the exact mismatch check. Contiguity also covers transposed
    # connectivity such as ``edge_index``.
    transport = value.to(dtype=torch.int64, copy=True).contiguous()
    dist.broadcast(transport, src=source_rank)
    restored = transport.to(dtype=value.dtype)
    if not torch.equal(restored.to(dtype=torch.int64), transport):
        raise OverflowError(
            f"distributed discrete value cannot round-trip through {value.dtype}"
        )
    return restored


class AtlasDDPSynchronizer:
    """Synchronize discrete Morton state and continuous transport statistics.

    This mode is used when ranks hold disjoint view shards of the *same* object.
    Standard object-level DDP leaves it disabled because different objects should
    not share atlas states.
    """

    def __init__(self, context: DistributedContext, source_rank: int = 0) -> None:
        self.context = context
        self.source_rank = source_rank
        self.maps_global_evidence = True

    def aggregate_evidence(self, evidence: EvidenceParticles) -> EvidenceParticles:
        """Autograd-all-gather the complete measure for one mathematically global UOT solve."""

        if not self.context.distributed:
            return evidence
        local_size = torch.tensor(
            [evidence.positions.shape[0]], dtype=torch.int64, device=evidence.positions.device
        )
        sizes = [torch.empty_like(local_size) for _ in range(self.context.world_size)]
        dist.all_gather(sizes, local_size)
        counts = [int(value.item()) for value in sizes]
        maximum = max(counts)

        def gather(value: Tensor) -> Tensor:
            padding_shape = (maximum - value.shape[0],) + value.shape[1:]
            padded = torch.cat((value, value.new_zeros(padding_shape)), dim=0)
            if value.dtype.is_floating_point:
                gathered = dist_nn.all_gather(padded)
            else:
                gathered_list = [torch.empty_like(padded) for _ in range(self.context.world_size)]
                dist.all_gather(gathered_list, padded)
                gathered = tuple(gathered_list)
            return torch.cat(
                [rank_value[:count] for rank_value, count in zip(gathered, counts)], dim=0
            )

        local_view_count = torch.tensor(
            [evidence.extrinsics_world_to_camera.shape[0]],
            dtype=torch.int64,
            device=evidence.positions.device,
        )
        view_counts = [torch.empty_like(local_view_count) for _ in range(self.context.world_size)]
        dist.all_gather(view_counts, local_view_count)
        view_offsets = []
        running = 0
        for value in view_counts:
            view_offsets.append(running)
            running += int(value.item())
        padded_view = torch.cat(
            (
                evidence.view_index,
                evidence.view_index.new_zeros(maximum - evidence.view_index.shape[0]),
            ),
            dim=0,
        )
        gathered_view = [torch.empty_like(padded_view) for _ in range(self.context.world_size)]
        dist.all_gather(gathered_view, padded_view)
        global_view = torch.cat(
            [
                value[:count] + offset
                for value, count, offset in zip(gathered_view, counts, view_offsets)
            ],
            dim=0,
        )
        camera_counts = [int(value.item()) for value in view_counts]
        maximum_cameras = max(camera_counts)

        def gather_cameras(value: Tensor) -> Tensor:
            padding_shape = (maximum_cameras - value.shape[0],) + value.shape[1:]
            padded = torch.cat((value, value.new_zeros(padding_shape)), dim=0)
            gathered = dist_nn.all_gather(padded)
            return torch.cat(
                [rank_value[:count] for rank_value, count in zip(gathered, camera_counts)],
                dim=0,
            )

        color_presence = torch.tensor(
            [int(evidence.colors is not None)], dtype=torch.int64, device=evidence.positions.device
        )
        gathered_presence = [torch.empty_like(color_presence) for _ in range(self.context.world_size)]
        dist.all_gather(gathered_presence, color_presence)
        if len({int(value.item()) for value in gathered_presence}) != 1:
            raise RuntimeError("DDP ranks disagree on evidence color availability")
        global_evidence = EvidenceParticles(
            positions=gather(evidence.positions),
            rays=gather(evidence.rays),
            features=gather(evidence.features),
            covariance=gather(evidence.covariance),
            confidence=gather(evidence.confidence),
            mass=gather(evidence.mass),
            view_index=global_view,
            pixel_uv=gather(evidence.pixel_uv),
            extrinsics_world_to_camera=gather_cameras(
                evidence.extrinsics_world_to_camera
            ),
            intrinsics=gather_cameras(evidence.intrinsics),
            depth_variance=gather(evidence.depth_variance),
            colors=gather(evidence.colors) if evidence.colors is not None else None,
        )
        global_evidence.validate()
        return global_evidence

    def aggregate_atlas_measure(self, position: Tensor, mass: Tensor) -> tuple[Tensor, Tensor]:
        """Autograd-aware variable-length all-gather of geometric evidence."""

        if not self.context.distributed:
            return position, mass
        local_size = torch.tensor([position.shape[0]], dtype=torch.int64, device=position.device)
        sizes = [torch.empty_like(local_size) for _ in range(self.context.world_size)]
        dist.all_gather(sizes, local_size)
        counts = [int(value.item()) for value in sizes]
        maximum = max(counts)
        padded_position = torch.cat(
            (position, position.new_zeros((maximum - position.shape[0], 3))), dim=0
        )
        padded_mass = torch.cat((mass, mass.new_zeros(maximum - mass.shape[0])), dim=0)
        gathered_position = dist_nn.all_gather(padded_position)
        gathered_mass = dist_nn.all_gather(padded_mass)
        global_position = torch.cat(
            [value[:count] for value, count in zip(gathered_position, counts)], dim=0
        )
        global_mass = torch.cat(
            [value[:count] for value, count in zip(gathered_mass, counts)], dim=0
        )
        return global_position, global_mass

    def synchronize_trellis_prior_measure(
        self,
        measure: Optional[TrellisPriorMeasure],
        dtype: torch.dtype = torch.float32,
    ) -> TrellisPriorMeasure:
        """Broadcast source-only hidden support before persistent atlas creation.

        TRELLIS is frozen and sampled under ``no_grad``. Running it on every
        same-object rank is therefore redundant rather than an autograd path.
        Non-source ranks pass ``None`` and allocate from broadcast metadata.
        """

        if not self.context.distributed:
            if measure is None:
                raise ValueError("single-rank TRELLIS synchronization requires a measure")
            return measure
        if self.context.rank == self.source_rank:
            if measure is None:
                raise ValueError("the TRELLIS source rank did not produce a prior measure")
            metadata_values = (
                measure.positions.shape[0],
                measure.sample_count,
                measure.resolution,
            )
        else:
            if measure is not None:
                raise ValueError("non-source TRELLIS rank must not sample a redundant prior")
            metadata_values = (0, 0, 0)
        metadata = torch.tensor(
            metadata_values,
            dtype=torch.int64,
            device=self.context.device,
        )
        dist.broadcast(metadata, src=self.source_rank)
        count, sample_count, resolution = map(int, metadata.tolist())

        def synchronize(
            value: Optional[Tensor],
            shape: tuple[int, ...],
            value_dtype: torch.dtype,
        ) -> Tensor:
            if self.context.rank == self.source_rank:
                if value is None:
                    raise RuntimeError("source TRELLIS tensor is unavailable")
                tensor = value.to(
                    device=self.context.device,
                    dtype=value_dtype,
                ).contiguous()
                if tuple(tensor.shape) != shape:
                    raise RuntimeError("rank-zero TRELLIS prior has inconsistent metadata")
            else:
                tensor = torch.empty(
                    shape,
                    dtype=value_dtype,
                    device=self.context.device,
                )
            dist.broadcast(tensor, src=self.source_rank)
            return tensor

        synchronized = TrellisPriorMeasure(
            coordinates=synchronize(
                measure.coordinates if measure is not None else None,
                (count, 3),
                torch.int64,
            ),
            positions=synchronize(
                measure.positions if measure is not None else None,
                (count, 3),
                dtype,
            ),
            probability=synchronize(
                measure.probability if measure is not None else None,
                (count,),
                dtype,
            ),
            mass=synchronize(
                measure.mass if measure is not None else None,
                (count,),
                dtype,
            ),
            mass_variance=synchronize(
                measure.mass_variance if measure is not None else None,
                (count,),
                dtype,
            ),
            vote_count=synchronize(
                measure.vote_count if measure is not None else None,
                (count,),
                torch.int64,
            ),
            sample_count=sample_count,
            resolution=resolution,
        )
        synchronized.validate()
        return synchronized

    def should_sample_trellis_prior(self) -> bool:
        """Whether this rank owns frozen TRELLIS sampling in same-object DDP."""

        return not self.context.distributed or self.context.rank == self.source_rank

    def aggregate_prior_images(self, images: Tensor) -> Tensor:
        """Gather disjoint same-object view shards for TRELLIS conditioning."""

        if not self.context.distributed:
            return images
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("TRELLIS conditioning images must have shape [K,3,H,W]")
        local_count = torch.tensor(
            [images.shape[0]], dtype=torch.int64, device=images.device
        )
        counts_tensor = [torch.empty_like(local_count) for _ in range(self.context.world_size)]
        dist.all_gather(counts_tensor, local_count)
        counts = [int(value.item()) for value in counts_tensor]
        maximum = max(counts)
        padded = torch.cat(
            (
                images.detach(),
                images.new_zeros((maximum - images.shape[0], *images.shape[1:])),
            ),
            dim=0,
        )
        gathered = [torch.empty_like(padded) for _ in range(self.context.world_size)]
        dist.all_gather(gathered, padded)
        return torch.cat(
            [rank_images[:count] for rank_images, count in zip(gathered, counts)],
            dim=0,
        )

    def synchronize_atlas(self, atlas: PersistentOctreeAtlas) -> PersistentOctreeAtlas:
        """Select one source-exact nonlinear atlas and preserve global gradients.

        Same-object ranks first construct the same global evidence measure with
        an autograd-aware all-gather.  Atlas chart fitting contains discrete cell
        assignment and gauge-valued PCA eigenvectors, so independently computed
        floating chart coordinates need not be bitwise equal even when their
        geometry is equivalent.  The source rank therefore owns this nonlinear
        realization.  ``dist_nn.broadcast`` reduces all downstream rank losses
        to that source in backward; the preceding differentiable all-gather then
        routes atlas derivatives to every rank's local evidence graph.
        """

        if not self.context.distributed:
            return atlas
        names = (
            "root_min",
            "root_max",
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
            "edge_index",
            "overlap_rotation",
            "overlap_translation",
        )
        signature = {
            "config": asdict(atlas.config),
            "fields": tuple(
                (
                    name,
                    tuple(getattr(atlas, name).shape),
                    str(getattr(atlas, name).dtype),
                    getattr(atlas, name).device.type,
                )
                for name in names
            ),
        }
        gathered: list[object] = [None for _ in range(self.context.world_size)]
        dist.all_gather_object(gathered, signature)
        if any(value != signature for value in gathered):
            raise RuntimeError(
                f"DDP atlas metadata mismatch before typed collectives: {gathered}"
            )
        device_mismatch = torch.tensor(
            int(
                any(
                    getattr(atlas, name).device != self.context.device
                    for name in names
                )
            ),
            dtype=torch.int64,
            device=self.context.device,
        )
        dist.all_reduce(device_mismatch, op=dist.ReduceOp.MAX)
        if int(device_mismatch.item()):
            raise RuntimeError(
                "DDP atlas fields are not resident on their rank-local context device"
            )
        # These fields contain local chart-gauge coordinates. Raw elementwise
        # equality is neither gauge invariant nor expected from independent
        # eigensolver calls. Their authoritative source realization is still
        # checked for finiteness and for all atlas structural invariants below.
        gauge_coordinate_fields = {
            "chart_frames",
            "curvature",
            "overlap_rotation",
            "overlap_translation",
        }
        for name in names:
            value = getattr(atlas, name)
            if value.dtype.is_floating_point:
                if self.maps_global_evidence:
                    # This is the exact derivative of one common atlas, unlike
                    # a straight-through identity through a potentially
                    # different rank-local PCA gauge.
                    # Empty overlap tensors carry no state or derivative and
                    # need no backend collective. Metadata equality guarantees
                    # that every rank takes this branch together.
                    synchronized = (
                        dist_nn.broadcast(value.contiguous(), src=self.source_rank)
                        if value.numel()
                        else value
                    )
                    reference = synchronized.detach()
                    invalid = torch.tensor(
                        int(
                            not bool(torch.all(torch.isfinite(value.detach())))
                            or not bool(torch.all(torch.isfinite(reference)))
                        ),
                        dtype=torch.int64,
                        device=value.device,
                    )
                    dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
                    if int(invalid.item()):
                        raise RuntimeError(
                            f"global-evidence DDP produced non-finite atlas field {name}"
                        )
                    if name not in gauge_coordinate_fields:
                        if value.dtype in {torch.float16, torch.bfloat16}:
                            absolute_tolerance, relative_tolerance = 2.0e-3, 5.0e-3
                        elif value.dtype == torch.float32:
                            absolute_tolerance, relative_tolerance = 2.0e-5, 5.0e-5
                        else:
                            absolute_tolerance, relative_tolerance = 1.0e-10, 5.0e-10
                        difference = torch.abs(value.detach() - reference)
                        scale = torch.maximum(value.detach().abs(), reference.abs())
                        outside = difference > (
                            absolute_tolerance + relative_tolerance * scale
                        )
                        mismatch = torch.tensor(
                            int(bool(torch.any(outside))),
                            dtype=torch.int64,
                            device=value.device,
                        )
                        dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
                        if int(mismatch.item()):
                            local_error = (
                                difference.max()
                                if difference.numel()
                                else value.new_zeros(())
                            )
                            dist.all_reduce(local_error, op=dist.ReduceOp.MAX)
                            raise RuntimeError(
                                "global-evidence DDP produced inconsistent "
                                f"gauge-invariant atlas field {name}: maximum "
                                f"absolute error {float(local_error):.3e}"
                            )
                else:
                    synchronized = (
                        dist_nn.broadcast(value.contiguous(), src=self.source_rank)
                        if value.numel()
                        else value
                    )
            else:
                reference = _broadcast_discrete_exact(value, self.source_rank)
                mismatch = torch.tensor(
                    int(not torch.equal(value, reference)),
                    dtype=torch.int64,
                    device=value.device,
                )
                dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
                if self.maps_global_evidence and int(mismatch.item()):
                    raise RuntimeError(f"global-evidence DDP produced inconsistent discrete atlas field {name}")
                synchronized = reference
            setattr(atlas, name, synchronized)
        validation = atlas.validate()
        if not validation.valid:
            raise RuntimeError(
                f"synchronized DDP atlas violates structural invariants: {validation}"
            )
        return atlas

    def synchronize_split_mask(self, split_mask: Tensor) -> Tensor:
        if not self.context.distributed:
            return split_mask
        size = torch.tensor([split_mask.numel()], dtype=torch.int64, device=self.context.device)
        dist.broadcast(size, src=self.source_rank)
        if split_mask.numel() != int(size.item()):
            split_mask = torch.empty(int(size.item()), dtype=torch.bool, device=self.context.device)
        return _broadcast_discrete_exact(split_mask, self.source_rank)

    def reduce_mapping_statistics(self, mapping: MappingResult, atlas: PersistentOctreeAtlas) -> MappingResult:
        """Legacy sharded approximation; rejected when complete evidence was gathered."""

        if not self.context.distributed:
            return mapping
        if self.maps_global_evidence:
            raise RuntimeError("global-evidence UOT must not all-reduce already-global mapping statistics")
        node_key = tuple(map(int, mapping.graph.atlas_node_index.tolist()))
        gathered: list[object] = [None for _ in range(self.context.world_size)]
        dist.all_gather_object(gathered, node_key)
        if any(key != node_key for key in gathered):
            raise RuntimeError("DDP atlas key alignment failed before continuous-state reduction")
        mass = mapping.transported_mass
        position_numerator = mapping.transported_centers * mass[:, None]
        metric_numerator = mapping.riemannian_metric * mass[:, None, None]
        latent_numerator = mapping.latent * mass[:, None]
        reliability_numerator = mapping.observation_reliability * mass
        color_numerator = (
            mapping.transported_color * mass[:, None]
            if mapping.transported_color is not None
            else None
        )
        mass = dist_nn.all_reduce(mass, op=dist.ReduceOp.SUM)
        position_numerator = dist_nn.all_reduce(position_numerator, op=dist.ReduceOp.SUM)
        metric_numerator = dist_nn.all_reduce(metric_numerator, op=dist.ReduceOp.SUM)
        latent_numerator = dist_nn.all_reduce(latent_numerator, op=dist.ReduceOp.SUM)
        reliability_numerator = dist_nn.all_reduce(
            reliability_numerator, op=dist.ReduceOp.SUM
        )
        if color_numerator is not None:
            color_numerator = dist_nn.all_reduce(color_numerator, op=dist.ReduceOp.SUM)
        denominator = mass.clamp_min(1.0e-8)
        mapping.transported_centers = position_numerator / denominator[:, None]
        mapping.riemannian_metric = metric_numerator / denominator[:, None, None]
        mapping.latent = latent_numerator / denominator[:, None]
        mapping.observation_reliability = reliability_numerator / denominator
        if color_numerator is not None:
            mapping.transported_color = color_numerator / denominator[:, None]
        mapping.transported_mass = mass
        return mapping


def _set_trainable_phase(model: GraftGS, phase: TrainingPhase) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules: tuple[nn.Module, ...]
    if phase is TrainingPhase.EVIDENCE_CALIBRATION:
        modules = (model.evidence_builder.calibrator,)
    elif phase is TrainingPhase.ATLAS_AUTOENCODING:
        modules = (model.evidence_builder, model.mapping, model.encoder, model.topology)
    elif phase is TrainingPhase.RIEMANNIAN_FLOW:
        modules = (model.vector_field,)
    elif phase is TrainingPhase.END_TO_END:
        modules = (model.evidence_builder, model.mapping, model.encoder, model.topology, model.vector_field, model.vggt.feature_projection)
    elif phase is TrainingPhase.QUANTIZATION_DISTILLATION:
        modules = (model.encoder, model.vector_field, model.vggt.feature_projection)
    else:
        modules = (model.evidence_builder, model.mapping, model.encoder, model.topology, model.vector_field)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if phase in {
        TrainingPhase.END_TO_END,
        TrainingPhase.QUANTIZATION_DISTILLATION,
        TrainingPhase.TOPOLOGY_HARDENING,
    }:
        for parameter in model.vggt.lora_parameters():
            parameter.requires_grad_(True)


def _execution_stage_for_phase(phase: TrainingPhase) -> str:
    return {
        TrainingPhase.EVIDENCE_CALIBRATION: "evidence_calibration",
        TrainingPhase.ATLAS_AUTOENCODING: "atlas_autoencoding",
        TrainingPhase.RIEMANNIAN_FLOW: "flow_pretraining",
        TrainingPhase.END_TO_END: "full",
        TrainingPhase.QUANTIZATION_DISTILLATION: "full",
        TrainingPhase.TOPOLOGY_HARDENING: "full",
    }[phase]


def _phase_renders_input_views(phase: TrainingPhase) -> bool:
    return phase in {
        TrainingPhase.ATLAS_AUTOENCODING,
        TrainingPhase.END_TO_END,
        TrainingPhase.QUANTIZATION_DISTILLATION,
        TrainingPhase.TOPOLOGY_HARDENING,
    }


class GraftGSTrainer:
    def __init__(
        self,
        model: GraftGS,
        config: TrainerConfig = TrainerConfig(),
        loss_weights: LossWeights = LossWeights(),
        teacher: Optional[GraftGS] = None,
    ) -> None:
        self.config = config
        self.precision_policy = NativePrecisionPolicy(
            backbone=config.precision_backbone,
            geometric_state=config.precision_geometric_state,
            analytical_solve=config.precision_analytical_solve,
            diagnostics=config.precision_diagnostics,
            float32_matmul_precision=config.precision_float32_matmul,
            allow_tf32=config.precision_allow_tf32,
        )
        self.precision_record = self.precision_policy.apply()
        adapter_dtype = getattr(model.vggt, "backbone_dtype", None)
        if adapter_dtype is not None and adapter_dtype != self.precision_policy.backbone_dtype:
            raise ValueError(
                "VGGT adapter backbone dtype differs from the trainer precision policy"
            )
        self.context = DistributedContext.initialize()
        assert_local_cuda_allocator_ownership(self.context.device)
        self._seed_everything(config.seed + self.context.rank)
        if config.phase in {
            TrainingPhase.END_TO_END,
            TrainingPhase.QUANTIZATION_DISTILLATION,
            TrainingPhase.TOPOLOGY_HARDENING,
        }:
            model.vggt.install_late_lora()
        _set_trainable_phase(model, config.phase)
        self.qat_modules: list[str] = []
        if config.phase in {
            TrainingPhase.QUANTIZATION_DISTILLATION,
            TrainingPhase.TOPOLOGY_HARDENING,
        }:
            self.qat_modules = apply_equivariant_qat(
                model,
                QuantizationConfig(
                    bits=8,
                    block_size=16,
                    stochastic_rounding=True,
                    adversarial_log_scale_radius=(
                        config.quantization_adversarial_log_scale_radius
                        if config.phase is TrainingPhase.TOPOLOGY_HARDENING
                        else 0.0
                    ),
                ),
            )
        self.module = model.to(self.context.device)
        self.teacher = teacher.to(self.context.device).eval() if teacher is not None else None
        if self.teacher is not None:
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)
        self.model: nn.Module = self.module
        if self.context.distributed:
            self.model = DistributedDataParallel(
                self.module,
                device_ids=[self.context.local_rank],
                output_device=self.context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=config.find_unused_parameters,
                gradient_as_bucket_view=True,
            )
        parameters = [parameter for parameter in self.module.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError(f"phase {config.phase.value} selected no trainable parameters")
        self.trainable_parameters = tuple(parameters)
        self.optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
        learned_perceptual = (
            LearnedPerceptualPyramid.from_checkpoint(
                config.perceptual_checkpoint,
                expected_sha256=config.perceptual_checkpoint_sha256,
            ).to(self.context.device)
            if config.perceptual_checkpoint is not None
            else None
        )
        self.loss = GraftGSLoss(loss_weights, learned_perceptual).to(
            self.context.device
        )
        self.gradient_purifier = None
        if (
            config.phase is TrainingPhase.TOPOLOGY_HARDENING
            and config.gradient_purification_enabled
        ):
            self.gradient_purifier = HilbertGradientPurifier(
                self.trainable_parameters,
                GradientPurificationConfig(
                    maximum_views=config.gradient_purification_maximum_views,
                    consensus_cosine=config.gradient_consensus_cosine,
                    consensus_relative_singular_value=config.gradient_consensus_relative_singular_value,
                    artifact_relative_singular_value=config.gradient_artifact_relative_singular_value,
                    weiszfeld_iterations=config.gradient_weiszfeld_iterations,
                    fisher_decay=config.gradient_fisher_decay,
                    fisher_damping=config.gradient_fisher_damping,
                    fisher_radius=config.gradient_fisher_radius,
                ),
            )
        self.synchronizer = AtlasDDPSynchronizer(self.context) if config.synchronize_object_atlas else None
        self.global_step = 0
        self.epoch = 0
        self.microstep = 0
        self.batches_consumed_in_epoch = 0
        self.output_directory = Path(config.output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_directory / "metrics.jsonl"
        self._mesh_supervisor: Optional[object] = None

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _assert_finite_tensors(
        self,
        stage: str,
        tensors: Mapping[str, Tensor],
    ) -> None:
        """Collectively reject non-finite loss, gradient, or optimizer state.

        Gradient clipping cannot repair NaN/Inf: its scale factor becomes
        non-finite and corrupts every parameter in the following optimizer
        step.  This guard is collective so no rank exits while peers remain in
        a later NCCL operation.
        """

        # Build one asynchronous device indicator first. Converting every
        # parameter predicate to Python bool would serialize the CUDA stream
        # hundreds of times per step and materially depress A800 utilization.
        failure = torch.zeros((), dtype=torch.int64, device=self.context.device)
        for name, value in tensors.items():
            inspected = value.coalesce().values() if value.is_sparse else value
            del name
            indicator = torch.any(~torch.isfinite(inspected.detach())).to(
                device=self.context.device,
                dtype=torch.int64,
            )
            failure = torch.maximum(failure, indicator)
        if self.context.distributed:
            dist.all_reduce(failure, op=dist.ReduceOp.MAX)
        if not int(failure.item()):
            return
        bad = []
        for name, value in tensors.items():
            inspected = value.coalesce().values() if value.is_sparse else value
            if not bool(torch.all(torch.isfinite(inspected.detach()))):
                bad.append(name)
        if self.context.distributed:
            gathered: list[object] = [None for _ in range(self.context.world_size)]
            dist.all_gather_object(
                gathered,
                {"rank": self.context.rank, "nonfinite": bad[:64], "count": len(bad)},
            )
        else:
            gathered = [{"rank": 0, "nonfinite": bad[:64], "count": len(bad)}]
        raise FloatingPointError(
            f"non-finite training tensors before {stage}: {gathered}"
        )

    def train_step(self, batch: Mapping[str, object], microstep: int = 0) -> dict[str, float]:
        self.model.train()
        # In same-object DDP the collated CPU sample contains the union of all
        # rank views. Select the deterministic local shard before transferring
        # tensors, otherwise every process briefly materializes the entire
        # global image/camera set on its assigned A800.
        images = torch.as_tensor(batch["images"])
        valid_mask = batch.get("valid_mask")
        if valid_mask is not None:
            valid_mask = torch.as_tensor(valid_mask)
        view_supervision = self._view_supervision(batch)
        images, valid_mask, view_supervision = self._shard_object_views(
            images, valid_mask, view_supervision
        )
        images = images.to(
            device=self.context.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        if valid_mask is not None:
            valid_mask = valid_mask.to(
                device=self.context.device,
                non_blocking=True,
            )
        view_supervision = {
            name: value.to(
                device=self.context.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            for name, value in view_supervision.items()
        }
        atlas_root_bounds = self._atlas_root_bounds(batch)
        trellis_prior_seed = self._trellis_prior_seed(batch)
        robustness = None
        if self.config.phase is TrainingPhase.TOPOLOGY_HARDENING:
            if images.shape[1] > 2:
                keep = torch.rand(images.shape[1], device=images.device) >= 0.15
                keep[:2] = True
                images = images[:, keep]
                if valid_mask is not None:
                    valid_mask = valid_mask[:, keep]
                view_supervision = {
                    name: value[:, keep] for name, value in view_supervision.items()
                }
            images = (images + 0.01 * torch.randn_like(images)).clamp(0.0, 1.0)
            robustness = RobustnessPerturbation()
        should_step = (
            microstep % self.config.gradient_accumulation_steps
            == self.config.gradient_accumulation_steps - 1
        )
        manual_phase_f_sync = (
            self.context.distributed
            and self.config.phase is TrainingPhase.TOPOLOGY_HARDENING
            and self.gradient_purifier is not None
        )
        synchronize = self.context.distributed and (
            not should_step or manual_phase_f_sync
        )
        context = self.model.no_sync() if synchronize and isinstance(self.model, DistributedDataParallel) else nullcontext()
        if self.context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.context.device)
        start = time.perf_counter()
        scale_adversaries = quantization_scale_adversaries(self.module)
        for adversary in scale_adversaries:
            adversary.reset_adversary()
        forward_rng_state = self._capture_forward_rng() if scale_adversaries else None
        # These targets are immutable functions of the audited mesh and camera
        # manifests. Render them before the trainable forward graph exists so
        # nvdiffrast's transient binning/geometry workspace cannot overlap the
        # model's peak activation state.
        mesh_targets = self._derive_mesh_targets(
            batch, view_supervision, images.shape[-2:]
        )

        def forward_model() -> GraftGSOutput:
            return self.model(
                images,
                valid_mask=valid_mask,
                render_input_views=_phase_renders_input_views(self.config.phase),
                distributed_synchronizer=self.synchronizer,
                robustness=robustness,
                ground_truth_extrinsics=view_supervision.get("extrinsics_world_to_camera"),
                ground_truth_intrinsics=view_supervision.get("intrinsics"),
                atlas_root_bounds=atlas_root_bounds,
                execution_stage=_execution_stage_for_phase(self.config.phase),
                trellis_prior_seed=trellis_prior_seed,
                capture_distillation_activations=(
                    self.config.phase is TrainingPhase.QUANTIZATION_DISTILLATION
                ),
            )

        with context:
            output = forward_model()
            loss_batch = dict(batch)
            loss_batch["valid_mask"] = valid_mask
            loss_batch["evidence_mask"] = valid_mask
            if "alpha" in view_supervision:
                loss_batch["alpha"] = view_supervision["alpha"]
            loss_batch.update(mesh_targets)
            if self.config.phase is TrainingPhase.TOPOLOGY_HARDENING:
                loss_batch["feasibility_relative_margin"] = (
                    self.config.topology_hardening_relative_margin
                )
                loss_batch["feasibility_relative_temperature"] = (
                    self.config.topology_hardening_temperature
                )
            total, terms = self.loss(self.module, output, loss_batch, self.config.phase.value)
            if self.config.phase is TrainingPhase.QUANTIZATION_DISTILLATION:
                if self.teacher is None:
                    raise RuntimeError("Phase E requires a frozen teacher")
                with torch.no_grad():
                    teacher_output = self.teacher(
                        images,
                        valid_mask=valid_mask,
                        render_input_views=True,
                        distributed_synchronizer=self.synchronizer,
                        ground_truth_extrinsics=view_supervision.get("extrinsics_world_to_camera"),
                        ground_truth_intrinsics=view_supervision.get("intrinsics"),
                        atlas_root_bounds=atlas_root_bounds,
                        trellis_prior_seed=trellis_prior_seed,
                        capture_distillation_activations=True,
                    )
                distill = distillation_loss(
                    output,
                    teacher_output,
                    self.loss.weights,
                    teacher_confidence=self.config.teacher_distillation_confidence,
                    teacher_topology_confidence=self.config.teacher_topology_confidence,
                    student_model=self.module,
                    teacher_model=self.teacher,
                )
                terms["distill"] = distill
                total = total + distill
            self._assert_finite_tensors(
                "backward",
                {"total": total, **terms},
            )
            scale_adversary_metrics: dict[str, float] = {}
            if scale_adversaries:
                adversarial_tensors = [
                    module.adversarial_log_scale for module in scale_adversaries
                ]
                scale_gradient = torch.autograd.grad(
                    total,
                    adversarial_tensors,
                    retain_graph=False,
                    allow_unused=True,
                )
                for module, gradient in zip(scale_adversaries, scale_gradient):
                    module.set_worst_case_from_gradient(gradient)
                if forward_rng_state is None:
                    raise RuntimeError("scale adversary lost the forward RNG snapshot")
                self._restore_forward_rng(forward_rng_state)
                output = forward_model()
                total, terms = self.loss(
                    self.module, output, loss_batch, self.config.phase.value
                )
                finite_gradient = [
                    gradient.detach().to(torch.float32)
                    for gradient in scale_gradient
                    if gradient is not None
                ]
                scale_adversary_metrics = {
                    "hardening/scale_adversaries": float(len(scale_adversaries)),
                    "hardening/scale_gradient_norm": float(
                        torch.linalg.vector_norm(torch.stack(finite_gradient)).cpu()
                    )
                    if finite_gradient
                    else 0.0,
                    "hardening/relative_margin": self.config.topology_hardening_relative_margin,
                }
            self._assert_finite_tensors(
                "final backward",
                {"total": total, **terms},
            )
            purification_metrics: dict[str, float] = {}
            if self.gradient_purifier is not None:
                purification_metrics = self._backward_with_gradient_purification(
                    total,
                    terms,
                    output,
                    loss_batch,
                )
            else:
                (total / self.config.gradient_accumulation_steps).backward()
        for adversary in scale_adversaries:
            adversary.reset_adversary()
        gradient_norm = 0.0
        if should_step:
            if manual_phase_f_sync:
                self._synchronize_gradients()
            if self.gradient_purifier is not None:
                self.gradient_purifier.commit_fisher(
                    self._distributed_tensor_mean
                    if self.context.distributed
                    else None
                )
            named_gradient = {
                name: parameter.grad
                for name, parameter in self.module.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
            }
            self._assert_finite_tensors("gradient clipping", named_gradient)
            named_parameter = {
                name: parameter
                for name, parameter in self.module.named_parameters()
                if parameter.requires_grad
            }
            self._assert_finite_tensors("optimizer step", named_parameter)
            gradient_norm = float(
                _clip_grad_norm_high_precision(
                    self.trainable_parameters,
                    self.config.maximum_gradient_norm,
                ).cpu()
            )
            self.optimizer.step()
            self._assert_finite_tensors("post-step parameter state", named_parameter)
            optimizer_tensor = {
                f"parameter_{parameter_index}.{state_name}": state_value
                for parameter_index, parameter in enumerate(self.trainable_parameters)
                for state_name, state_value in self.optimizer.state.get(parameter, {}).items()
                if isinstance(state_value, Tensor)
            }
            self._assert_finite_tensors(
                "post-step optimizer state", optimizer_tensor
            )
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
        elapsed = time.perf_counter() - start
        if self.context.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(self.context.device)
            peak_reserved_memory = torch.cuda.max_memory_reserved(self.context.device)
            device_memory = torch.cuda.get_device_properties(
                self.context.device
            ).total_memory
        else:
            peak_memory = 0
            peak_reserved_memory = 0
            device_memory = 0
        metrics = {name: float(value.detach().cpu()) for name, value in terms.items()}
        if output.scenes:
            metrics.update(
                trellis_prior_support_count=float(
                    sum(scene.trellis_prior_support_count for scene in output.scenes)
                    / len(output.scenes)
                ),
                trellis_prior_expected_mass=float(
                    sum(scene.trellis_prior_expected_mass for scene in output.scenes)
                    / len(output.scenes)
                ),
                prior_only_active_charts=float(
                    sum(
                        int(
                            torch.sum(
                                (scene.atlas.evidence_mass[scene.atlas.active_indices] == 0)
                                & (scene.atlas.prior_mass[scene.atlas.active_indices] > 0)
                            ).item()
                        )
                        for scene in output.scenes
                    )
                    / len(output.scenes)
                ),
                observation_reliability_mean=float(
                    torch.stack(
                        [scene.mapping.observation_reliability.mean() for scene in output.scenes]
                    )
                    .mean()
                    .detach()
                    .cpu()
                ),
            )
        metrics.update(
            total=float(total.detach().cpu()),
            gradient_norm=gradient_norm,
            seconds=elapsed,
            peak_memory_bytes=float(peak_memory),
            peak_reserved_memory_bytes=float(peak_reserved_memory),
            device_memory_bytes=float(device_memory),
            peak_allocated_fraction=(
                float(peak_memory / device_memory) if device_memory else 0.0
            ),
            peak_reserved_fraction=(
                float(peak_reserved_memory / device_memory) if device_memory else 0.0
            ),
            local_scenes=float(images.shape[0]),
            local_views=float(images.shape[0] * images.shape[1]),
            local_views_per_second=float(
                images.shape[0] * images.shape[1] / max(elapsed, 1.0e-12)
            ),
        )
        metrics.update(purification_metrics)
        metrics.update(scale_adversary_metrics)
        if should_step and self.global_step % self.config.log_every == 0:
            self._log(metrics)
        return metrics

    def _capture_forward_rng(self) -> dict[str, Tensor]:
        state = {"cpu": torch.get_rng_state()}
        if self.context.device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state(self.context.device)
        return state

    def _restore_forward_rng(self, state: Mapping[str, Tensor]) -> None:
        torch.set_rng_state(state["cpu"].cpu())
        if self.context.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda"].cpu(), self.context.device)

    def _backward_with_gradient_purification(
        self,
        total: Tensor,
        terms: Mapping[str, Tensor],
        output: GraftGSOutput,
        batch: Mapping[str, object],
    ) -> dict[str, float]:
        """Replace only view-conditioned gradients by robust consensus gradients."""

        if self.gradient_purifier is None:
            raise RuntimeError("gradient purifier is not configured")
        view_terms = {
            "render": self.loss.weights.render,
            "ssim": self.loss.weights.ssim,
            "perceptual": self.loss.weights.perceptual,
            "mask": self.loss.weights.mask,
            "mesh_depth": self.loss.weights.mesh_depth,
            "mesh_normal": self.loss.weights.mesh_normal,
            "vggt_depth_reprojection": self.loss.weights.vggt_depth_reprojection,
            "vggt_depth_normal": self.loss.weights.vggt_depth_normal,
            "tile_opacity": self.loss.weights.tile_opacity,
        }
        global_view_objective = sum(
            weight * terms[name] for name, weight in view_terms.items()
        )
        stable_objective = total - global_view_objective
        per_view = view_conditioned_objectives(
            self.module,
            output,
            batch,
            self.loss.weights,
            self.loss.learned_perceptual,
        )
        flat_reliability = per_view.reliability.reshape(-1)
        valid = torch.nonzero(flat_reliability > 0, as_tuple=False).flatten()
        if valid.numel() < 2:
            raise RuntimeError(
                "Phase-F gradient purification requires at least two valid rendered views"
            )
        maximum = self.gradient_purifier.config.maximum_views
        if valid.numel() > maximum:
            order = torch.argsort(flat_reliability[valid], descending=True, stable=True)
            valid = valid[order[:maximum]]
        local_objective = per_view.objective.reshape(-1)[valid]
        artifact_delta = per_view.artifact_delta.reshape(-1)[valid]
        reliability = flat_reliability[valid]
        view_gradients: list[Gradient] = []
        artifact_gradients: list[Gradient] = []
        for index in range(local_objective.numel()):
            view_gradients.append(
                tuple(
                    torch.autograd.grad(
                        local_objective[index],
                        self.trainable_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                )
            )
            artifact_gradients.append(
                tuple(
                    torch.autograd.grad(
                        artifact_delta[index],
                        self.trainable_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                )
            )
        purified, diagnostics = self.gradient_purifier.purify(
            view_gradients,
            reliability,
            artifact_gradients,
        )
        (stable_objective / self.config.gradient_accumulation_steps).backward()
        for parameter, component in zip(self.trainable_parameters, purified):
            if component is None:
                continue
            contribution = component / self.config.gradient_accumulation_steps
            if parameter.grad is None:
                parameter.grad = contribution.detach().clone()
            else:
                parameter.grad.add_(contribution.detach())
        return {
            "gradient_purification/retained_views": float(diagnostics.retained_views),
            "gradient_purification/consensus_rank": float(diagnostics.consensus_rank),
            "gradient_purification/artifact_rank": float(diagnostics.artifact_rank),
            "gradient_purification/cone_acceptance": float(
                diagnostics.cone_acceptance_fraction.detach().cpu()
            ),
            "gradient_purification/median_residual": float(
                diagnostics.median_residual.detach().cpu()
            ),
            "gradient_purification/fisher_norm": float(
                diagnostics.fisher_norm.detach().cpu()
            ),
            "gradient_purification/fisher_scale": float(
                diagnostics.fisher_scale.detach().cpu()
            ),
            "gradient_purification/raw_view_objective": float(
                local_objective.mean().detach().cpu()
            ),
        }

    @torch.no_grad()
    def _distributed_tensor_mean(self, value: Tensor) -> Tensor:
        reduced = value.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced / self.context.world_size

    @torch.no_grad()
    def _synchronize_gradients(self, bucket_elements: int = 8_000_000) -> None:
        """Average purified rank-local gradients in deterministic dense buckets."""

        presence = torch.tensor(
            [parameter.grad is not None for parameter in self.trainable_parameters],
            device=self.context.device,
            dtype=torch.int32,
        )
        dist.all_reduce(presence, op=dist.ReduceOp.MAX)
        active = [
            parameter
            for index, parameter in enumerate(self.trainable_parameters)
            if bool(presence[index])
        ]
        for dtype in sorted({parameter.dtype for parameter in active}, key=str):
            group = [parameter for parameter in active if parameter.dtype == dtype]
            bucket: list[nn.Parameter] = []
            elements = 0
            for parameter in group:
                if bucket and elements + parameter.numel() > bucket_elements:
                    self._reduce_gradient_bucket(bucket)
                    bucket = []
                    elements = 0
                bucket.append(parameter)
                elements += parameter.numel()
            if bucket:
                self._reduce_gradient_bucket(bucket)

    @torch.no_grad()
    def _reduce_gradient_bucket(self, parameters: Sequence[nn.Parameter]) -> None:
        flat = torch.cat(
            [
                (
                    parameter.grad.reshape(-1)
                    if parameter.grad is not None
                    else torch.zeros_like(parameter).reshape(-1)
                )
                for parameter in parameters
            ]
        )
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.context.world_size)
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            value = flat[offset : offset + count].view_as(parameter)
            if parameter.grad is None:
                parameter.grad = value.clone()
            else:
                parameter.grad.copy_(value)
            offset += count

    @torch.no_grad()
    def validate(self, loader: Iterable[Mapping[str, object]], maximum_batches: Optional[int] = None) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        count = 0
        for batch in loader:
            images = torch.as_tensor(batch["images"])
            valid_mask = batch.get("valid_mask")
            if valid_mask is not None:
                valid_mask = torch.as_tensor(valid_mask)
            view_supervision = self._view_supervision(batch)
            images, valid_mask, view_supervision = self._shard_object_views(
                images, valid_mask, view_supervision
            )
            images = images.to(
                device=self.context.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            if valid_mask is not None:
                valid_mask = valid_mask.to(
                    device=self.context.device,
                    non_blocking=True,
                )
            view_supervision = {
                name: value.to(
                    device=self.context.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                for name, value in view_supervision.items()
            }
            atlas_root_bounds = self._atlas_root_bounds(batch)
            trellis_prior_seed = self._trellis_prior_seed(batch)
            mesh_targets = self._derive_mesh_targets(
                batch, view_supervision, images.shape[-2:]
            )
            output = self.model(
                images,
                valid_mask=valid_mask,
                render_input_views=_phase_renders_input_views(self.config.phase),
                distributed_synchronizer=self.synchronizer,
                ground_truth_extrinsics=view_supervision.get("extrinsics_world_to_camera"),
                ground_truth_intrinsics=view_supervision.get("intrinsics"),
                atlas_root_bounds=atlas_root_bounds,
                execution_stage=_execution_stage_for_phase(self.config.phase),
                trellis_prior_seed=trellis_prior_seed,
            )
            loss_batch = dict(batch)
            loss_batch["valid_mask"] = valid_mask
            loss_batch["evidence_mask"] = valid_mask
            if "alpha" in view_supervision:
                loss_batch["alpha"] = view_supervision["alpha"]
            loss_batch.update(mesh_targets)
            total, terms = self.loss(self.module, output, loss_batch, self.config.phase.value)
            values = {**terms, "total": total}
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            count += 1
            if maximum_batches is not None and count >= maximum_batches:
                break
        if self.context.distributed:
            for name in sorted(totals):
                value = torch.tensor(totals[name], dtype=torch.float64, device=self.context.device)
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                totals[name] = float(value.cpu())
            count_tensor = torch.tensor(count, dtype=torch.int64, device=self.context.device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            count = int(count_tensor.item())
        if count == 0:
            raise ValueError("validation loader produced no batches")
        metrics = {f"validation/{name}": value / count for name, value in totals.items()}
        self._log(metrics)
        return metrics

    def _shard_object_views(
        self,
        images: Tensor,
        valid_mask: Optional[Tensor],
        view_supervision: Mapping[str, Tensor],
    ) -> tuple[Tensor, Optional[Tensor], dict[str, Tensor]]:
        """Give every rank a deterministic view shard of one common object."""

        if not self.config.synchronize_object_atlas or not self.context.distributed:
            return images, valid_mask, dict(view_supervision)
        has_cameras = {
            "extrinsics_world_to_camera",
            "intrinsics",
        }.issubset(view_supervision)
        minimum_per_rank = 2 if has_cameras else 1
        if images.shape[1] < minimum_per_rank * self.context.world_size:
            raise ValueError(
                "same-object DDP requires enough views on every rank for its camera gauge solve; "
                f"received {images.shape[1]} views for {self.context.world_size} ranks and "
                f"requires {minimum_per_rank} per rank"
            )
        view = slice(self.context.rank, None, self.context.world_size)
        images = images[:, view]
        if valid_mask is not None:
            valid_mask = valid_mask[:, view]
        return images, valid_mask, {
            name: value[:, view] for name, value in view_supervision.items()
        }

    def _view_supervision(self, batch: Mapping[str, object]) -> dict[str, Tensor]:
        available = {
            name: torch.as_tensor(batch[name])
            for name in ("extrinsics_world_to_camera", "intrinsics", "alpha")
            if name in batch and batch[name] is not None
        }
        camera_keys = {"extrinsics_world_to_camera", "intrinsics"}
        if bool(camera_keys.intersection(available)) and not camera_keys.issubset(available):
            raise ValueError("camera supervision batch must provide both extrinsics and intrinsics")
        for name, value in available.items():
            if value.ndim < 2 or value.shape[:2] != torch.as_tensor(batch["images"]).shape[:2]:
                raise ValueError(f"view-aligned field {name} does not match image [B,K] dimensions")
        return available

    def _atlas_root_bounds(self, batch: Mapping[str, object]) -> Optional[Tensor]:
        value = batch.get("atlas_root_bounds")
        if value is None:
            return None
        bounds = torch.as_tensor(value).to(
            device=self.context.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        if bounds.ndim == 2:
            bounds = bounds[None]
        if bounds.ndim != 3 or bounds.shape[-2:] != (2, 3):
            raise ValueError("atlas_root_bounds must have shape [2,3] or [B,2,3]")
        return bounds

    @staticmethod
    def _trellis_prior_seed(batch: Mapping[str, object]) -> int:
        """Stable object-level seed shared by all ranks and teacher/student."""

        value = batch.get("object_id", "graft-gs-unidentified-object")
        if isinstance(value, (list, tuple)):
            value = "\x1f".join(str(item) for item in value)
        digest = hashlib.sha256(str(value).encode("utf8")).digest()
        return int.from_bytes(digest[:8], "little") % (2**31 - 1)

    def _derive_mesh_targets(
        self,
        batch: Mapping[str, object],
        view_supervision: Mapping[str, Tensor],
        image_size: tuple[int, int],
    ) -> dict[str, object]:
        render_phase = self.config.phase in {
            TrainingPhase.ATLAS_AUTOENCODING,
            TrainingPhase.END_TO_END,
            TrainingPhase.QUANTIZATION_DISTILLATION,
            TrainingPhase.TOPOLOGY_HARDENING,
        }
        if not self.config.derive_mesh_depth_normals or not render_phase:
            return {}
        paths = batch.get("modality_paths")
        mesh_path = paths.get("render_mesh") if isinstance(paths, Mapping) else None
        cameras_available = {
            "extrinsics_world_to_camera",
            "intrinsics",
        }.issubset(view_supervision)
        if mesh_path is None or not cameras_available:
            if self.config.require_mesh_depth_normals:
                raise ValueError(
                    "configured mesh depth/normal supervision requires render_mesh path and audited cameras"
                )
            return {}
        if self._mesh_supervisor is None:
            from ..data.mesh_supervision import MeshGroundTruthRasterizer

            self._mesh_supervisor = MeshGroundTruthRasterizer(
                self.context.device,
                view_chunk_size=self.config.mesh_supervision_view_chunk_size,
            )
        extrinsics = view_supervision["extrinsics_world_to_camera"]
        intrinsics = view_supervision["intrinsics"]
        if extrinsics.shape[0] != 1:
            raise ValueError("mesh-derived reference path expects one variable-topology object per rank")
        with torch.no_grad():
            target = self._mesh_supervisor(
                mesh_path,
                extrinsics[0],
                intrinsics[0],
                int(image_size[0]),
                int(image_size[1]),
            )
        return {
            "mesh_depth_target": target.depth[None],
            "mesh_normal_target": target.normal[None],
            "mesh_visibility_mask": target.visibility[None],
            "mesh_normal_validity": target.normal_validity[None],
            "mesh_normal_provenance": target.normal_provenance,
        }

    def fit(
        self,
        train_loader: Iterable[Mapping[str, object]],
        steps: int,
        validation_loader: Optional[Iterable[Mapping[str, object]]] = None,
    ) -> None:
        while self.global_step < steps:
            dataset = getattr(train_loader, "dataset", None)
            if dataset is not None and hasattr(dataset, "set_epoch"):
                dataset.set_epoch(self.epoch)
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)
            completed_epoch = True
            for batch_index, batch in enumerate(train_loader):
                if batch_index < self.batches_consumed_in_epoch:
                    continue
                previous_step = self.global_step
                self.train_step(batch, self.microstep)
                self.microstep += 1
                self.batches_consumed_in_epoch = batch_index + 1
                stepped = self.global_step != previous_step
                if stepped and self.global_step % self.config.checkpoint_every == 0:
                    self.save_checkpoint(self.output_directory / f"step-{self.global_step:08d}.pt")
                if (
                    validation_loader is not None
                    and stepped
                    and self.global_step % self.config.validate_every == 0
                ):
                    self.validate(validation_loader)
                if self.global_step >= steps:
                    completed_epoch = False
                    break
            if completed_epoch:
                self.epoch += 1
                self.batches_consumed_in_epoch = 0

    def save_checkpoint(self, path: str | Path) -> None:
        # RNG streams are rank-local by construction (the trainer seed is
        # offset by rank).  Saving only rank zero and restoring that state on
        # every process collapses those streams after resume and is not exact
        # DDP continuation.  Checkpointing is therefore a collective operation
        # in distributed mode; only serialization remains rank-zero-only.
        local_rng_state = _capture_rng_state(self.context.rank)
        if self.context.distributed:
            rank_rng_states: list[object] = [None for _ in range(self.context.world_size)]
            dist.all_gather_object(rank_rng_states, local_rng_state)
        else:
            rank_rng_states = [local_rng_state]
        save_exception: Optional[BaseException] = None
        save_error: Optional[str] = None
        if self.context.rank == 0:
            try:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".tmp")
                payload = {
                    "format_version": 6,
                    "global_step": self.global_step,
                    "epoch": self.epoch,
                    "microstep": self.microstep,
                    "batches_consumed_in_epoch": self.batches_consumed_in_epoch,
                    "phase": self.config.phase.value,
                    "model": self.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "checkpoint_world_size": self.context.world_size,
                    "rank_rng_states": rank_rng_states,
                    "trainer_config": asdict(self.config),
                    "precision_runtime": self.precision_record,
                    "loss_weights": asdict(self.loss.weights),
                    "model_config": asdict(self.module.config),
                    "gradient_purifier": (
                        self.gradient_purifier.state_dict()
                        if self.gradient_purifier is not None
                        else None
                    ),
                }
                torch.save(payload, temporary)
                os.replace(temporary, path)
            except BaseException as error:
                save_exception = error
                save_error = f"{type(error).__name__}: {error}"
        if self.context.distributed:
            # This broadcast is the checkpoint commit fence. No non-source
            # rank may enter the next forward/NCCL collective until rank zero
            # has atomically installed the file or reported its failure.
            failed = torch.tensor(
                [int(save_error is not None)],
                dtype=torch.int64,
                device=self.context.device,
            )
            dist.broadcast(failed, src=0)
            if int(failed.item()):
                reports: list[object] = [None for _ in range(self.context.world_size)]
                dist.all_gather_object(
                    reports,
                    {"rank": self.context.rank, "checkpoint_error": save_error},
                )
                message = next(
                    (
                        str(report["checkpoint_error"])
                        for report in reports
                        if isinstance(report, Mapping)
                        and report.get("checkpoint_error") is not None
                    ),
                    "unknown rank-zero checkpoint failure",
                )
                if save_exception is not None:
                    raise RuntimeError(
                        f"distributed checkpoint commit failed: {message}"
                    ) from save_exception
                raise RuntimeError(f"distributed checkpoint commit failed: {message}")
        elif save_exception is not None:
            raise save_exception

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.context.device, weights_only=False)
        format_version = payload.get("format_version")
        if format_version not in {1, 2, 3, 4, 5, 6}:
            raise ValueError("unsupported checkpoint format")
        if payload["phase"] != self.config.phase.value:
            raise ValueError("checkpoint phase does not match the configured trainer phase")
        if "model_config" in payload and payload["model_config"] != asdict(self.module.config):
            raise ValueError("checkpoint model configuration does not match the trainer model")
        if format_version >= 4 and payload.get("loss_weights") != asdict(self.loss.weights):
            raise ValueError("checkpoint loss weights do not match the configured objective")
        if format_version >= 6 and payload.get("precision_runtime") != self.precision_record:
            raise ValueError("checkpoint runtime precision record differs from the active process")
        if format_version >= 4:
            checkpoint_world_size = int(payload.get("checkpoint_world_size", -1))
            if checkpoint_world_size != self.context.world_size:
                raise ValueError(
                    "exact trainer resume requires the checkpoint world size to match; "
                    f"checkpoint={checkpoint_world_size}, current={self.context.world_size}"
                )
            rank_rng_states = payload.get("rank_rng_states")
            if not isinstance(rank_rng_states, list) or len(rank_rng_states) != checkpoint_world_size:
                raise ValueError("checkpoint rank-local RNG state is missing or inconsistent")
            rng_state = rank_rng_states[self.context.rank]
            if not isinstance(rng_state, Mapping) or int(rng_state.get("rank", -1)) != self.context.rank:
                raise ValueError("checkpoint rank-local RNG state has invalid rank provenance")
        if self.gradient_purifier is not None:
            if format_version < 5 or payload.get("gradient_purifier") is None:
                raise ValueError(
                    "exact Phase-F resume requires format-5 gradient-purifier state"
                )
            purifier_state = payload["gradient_purifier"]
            if not isinstance(purifier_state, Mapping):
                raise ValueError("gradient-purifier checkpoint state is malformed")
            if purifier_state.get("config") != self.gradient_purifier.config.__dict__:
                raise ValueError("checkpoint gradient-purifier configuration differs")
        elif format_version >= 5 and payload.get("gradient_purifier") is not None:
            raise ValueError("checkpoint contains an active purifier but trainer does not")
        checkpoint_trainer = payload.get("trainer_config", {})
        if isinstance(checkpoint_trainer, Mapping):
            checkpoint_manifest = checkpoint_trainer.get("dataset_manifest_sha256")
            if (
                checkpoint_manifest is not None
                and self.config.dataset_manifest_sha256 is not None
                and checkpoint_manifest != self.config.dataset_manifest_sha256
            ):
                raise ValueError("checkpoint dataset manifest digest differs from current training data")
            resume_policy_fields = (
                "gradient_accumulation_steps",
                "maximum_gradient_norm",
                "find_unused_parameters",
                "gradient_purification_enabled",
                "gradient_purification_maximum_views",
                "gradient_consensus_cosine",
                "gradient_consensus_relative_singular_value",
                "gradient_artifact_relative_singular_value",
                "gradient_weiszfeld_iterations",
                "gradient_fisher_decay",
                "gradient_fisher_damping",
                "gradient_fisher_radius",
                "quantization_adversarial_log_scale_radius",
                "topology_hardening_relative_margin",
                "topology_hardening_temperature",
                "synchronize_object_atlas",
                "seed",
                "dataset_manifest_schema",
                "dataset_object_id_catalog_sha256",
                "dataset_object_id_count",
                "dataset_maximum_views",
                "topology_supervision_mode",
                "minimum_topology_confidence",
                "teacher_checkpoint",
                "teacher_distillation_confidence",
                "teacher_topology_confidence",
                "trellis_prior_checkpoint",
                "trellis_prior_samples",
                "trellis_prior_sampler_steps",
                "trellis_prior_strength",
                "trellis_prior_minimum_probability",
                "trellis_prior_uncertainty_discount",
                "dino_relational_pseudo_supervision",
                "trellis_latent_relational_pseudo_supervision",
                "dino_pseudo_confidence",
                "trellis_latent_pseudo_confidence",
                "derive_mesh_depth_normals",
                "require_mesh_depth_normals",
                "mesh_supervision_view_chunk_size",
                "teacher_bundle_root",
                "teacher_bundle_digest",
                "teacher_bundle_minimum_confidence",
                "perceptual_checkpoint",
                "perceptual_checkpoint_sha256",
                "precision_backbone",
                "precision_geometric_state",
                "precision_analytical_solve",
                "precision_diagnostics",
                "precision_float32_matmul",
                "precision_allow_tf32",
            )
            for field_name in resume_policy_fields:
                if field_name not in checkpoint_trainer:
                    current_value = getattr(self.config, field_name)
                    legacy_default = {
                        "teacher_distillation_confidence": 1.0,
                        "precision_backbone": "bfloat16",
                        "precision_geometric_state": "float32",
                        "precision_analytical_solve": "float32",
                        "precision_diagnostics": "float64",
                        "precision_float32_matmul": "highest",
                        "precision_allow_tf32": False,
                    }.get(field_name, object())
                    if format_version < 6 and current_value == legacy_default:
                        continue
                    if current_value in {None, False, 0, 0.0}:
                        continue
                    raise ValueError(
                        f"checkpoint predates active resume policy field {field_name}"
                    )
                checkpoint_value = checkpoint_trainer.get(field_name)
                current_value = getattr(self.config, field_name)
                if checkpoint_value != current_value:
                    raise ValueError(
                        f"checkpoint resume policy differs at {field_name}"
                    )
        self.module.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload["global_step"])
        self.epoch = int(payload.get("epoch", 0))
        self.microstep = int(payload.get("microstep", self.global_step * self.config.gradient_accumulation_steps))
        self.batches_consumed_in_epoch = int(payload.get("batches_consumed_in_epoch", 0))
        if self.gradient_purifier is not None:
            self.gradient_purifier.load_state_dict(payload["gradient_purifier"])
        if format_version >= 4:
            _restore_rng_state(rng_state, self.context.rank)
        else:
            # Legacy checkpoints contain only rank-zero RNG state.  They remain
            # loadable for compatibility but cannot substantiate exact
            # multi-rank continuation.
            torch.set_rng_state(payload["torch_rng"].cpu())
            if torch.cuda.is_available() and payload["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
            np.random.set_state(payload["numpy_rng"])
            random.setstate(payload["python_rng"])

    def load_model_weights(self, path: str | Path, strict: bool = True) -> None:
        """Initialize a new training phase from the preceding phase checkpoint."""

        payload = torch.load(path, map_location=self.context.device, weights_only=False)
        if isinstance(payload, Mapping) and "model_config" in payload:
            if payload["model_config"] != asdict(self.module.config):
                raise ValueError("phase initialization checkpoint uses a different model configuration")
        source_trainer = payload.get("trainer_config", {}) if isinstance(payload, Mapping) else {}
        if isinstance(source_trainer, Mapping):
            precision_fields = (
                "precision_backbone",
                "precision_geometric_state",
                "precision_analytical_solve",
                "precision_diagnostics",
                "precision_float32_matmul",
                "precision_allow_tf32",
            )
            source_format = int(payload.get("format_version", 0))
            for field_name in precision_fields:
                if field_name not in source_trainer:
                    if source_format >= 6:
                        raise ValueError(
                            f"phase initialization lacks precision provenance at {field_name}"
                        )
                    continue
                if source_trainer.get(field_name) != getattr(self.config, field_name):
                    raise ValueError(
                        f"phase initialization changes native precision at {field_name}"
                    )
        if self.config.trellis_prior_checkpoint is not None and isinstance(source_trainer, Mapping):
            source_prior = source_trainer.get("trellis_prior_checkpoint")
            source_phase = str(payload.get("phase", "")) if isinstance(payload, Mapping) else ""
            if source_prior is None and source_phase != TrainingPhase.EVIDENCE_CALIBRATION.value:
                raise ValueError("non-Phase-A initialization checkpoint lacks TRELLIS prior provenance")
            if source_prior is not None:
                for field_name in (
                    "trellis_prior_samples",
                    "trellis_prior_sampler_steps",
                    "trellis_prior_strength",
                    "trellis_prior_minimum_probability",
                    "trellis_prior_uncertainty_discount",
                ):
                    if source_trainer.get(field_name) != getattr(self.config, field_name):
                        raise ValueError(
                            f"phase initialization changes hidden-prior policy at {field_name}"
                        )
        state = payload["model"] if isinstance(payload, Mapping) and "model" in payload else payload
        target = self.module.state_dict()
        translated = {}
        for key, value in state.items():
            if key in target:
                translated[key] = value
                continue
            if key.endswith(".weight"):
                parametrized = key[: -len(".weight")] + ".parametrizations.weight.original"
                if parametrized in target:
                    translated[parametrized] = value
                    continue
            translated[key] = value
        incompatible = self.module.load_state_dict(translated, strict=strict)
        if not strict and self.context.rank == 0:
            record = {
                "event": "phase_weight_initialization",
                "source": str(path),
                "missing_keys": incompatible.missing_keys,
                "unexpected_keys": incompatible.unexpected_keys,
            }
            with self.log_path.open("a", encoding="utf8") as file:
                file.write(json.dumps(record, sort_keys=True) + "\n")

    def _log(self, metrics: Mapping[str, float]) -> None:
        if self.context.rank != 0:
            return
        record = {"step": self.global_step, "phase": self.config.phase.value, **metrics}
        with self.log_path.open("a", encoding="utf8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = [
    "AtlasDDPSynchronizer",
    "assert_local_cuda_allocator_ownership",
    "bind_local_cuda_device",
    "DistributedContext",
    "GraftGSTrainer",
    "TrainerConfig",
    "TrainingPhase",
]

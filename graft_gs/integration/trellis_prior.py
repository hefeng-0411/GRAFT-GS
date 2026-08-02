"""Use TRELLIS structure generation as a discrete hidden-surface prior.

TRELLIS does not decode final Gaussians or a mesh in GRAFT-GS.  Its sampled
sparse structures define a prior measure over canonical occupied cells. That
measure is aligned to the evidence root cube and combined with observed atlas
occupancy as an additive surface hazard before topology proposal.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch import Tensor

from ..geometry.atlas import PersistentOctreeAtlas
from ..observability import ProgressReporter
from .external import (
    external_module_provenance,
    import_external_module,
    resolve_trellis_checkpoint,
)


def _cached_torch_hub_checkout(repository: str) -> Path:
    """Resolve an already-installed Torch Hub checkout without network access.

    Torch 2.4 resolves an unqualified ``owner/repository`` through GitHub before
    checking the local checkout.  TRELLIS uses precisely that form for DINOv2,
    so an otherwise complete server cache still fails when GitHub is transiently
    unavailable.  GRAFT-GS server execution is intentionally checkpoint-frozen:
    select a local checkout deterministically and fail closed if it is absent.
    """

    parts = repository.split("/")
    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError("Torch Hub repository must have the form 'owner/name'")
    owner, name = parts
    hub_root = Path(torch.hub.get_dir()).expanduser().resolve()
    candidates = (
        hub_root / f"{owner}_{name}_main",
        hub_root / f"{owner}_{name}_master",
    )
    complete = [
        path
        for path in candidates
        if path.is_dir() and (path / "hubconf.py").is_file()
    ]
    if not complete:
        expected = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"offline TRELLIS initialization requires a cached {repository} "
            f"Torch Hub checkout containing hubconf.py; expected one of: {expected}"
        )
    # The released DINOv2 reference is ``main``.  Candidate order is therefore
    # meaningful and independent of filesystem enumeration order.
    return complete[0]


@contextmanager
def _offline_torch_hub_repository(repository: str) -> Iterator[Path]:
    """Redirect one upstream Hub repository to its exact cached checkout.

    The patch is deliberately scoped to TRELLIS construction and restored even
    on failure.  Other repositories retain ordinary Torch Hub semantics.
    """

    checkout = _cached_torch_hub_checkout(repository)
    original_load = torch.hub.load

    def load_from_cache(
        repo_or_dir: object,
        model: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        if repo_or_dir != repository:
            return original_load(repo_or_dir, model, *args, **kwargs)
        # These options belong to remote repository resolution.  Passing them
        # to a ``source='local'`` load is either meaningless or rejected by
        # older Torch releases.
        kwargs.pop("force_reload", None)
        kwargs.pop("trust_repo", None)
        kwargs.pop("skip_validation", None)
        kwargs.pop("source", None)
        return original_load(
            str(checkout),
            model,
            *args,
            source="local",
            **kwargs,
        )

    torch.hub.load = load_from_cache
    try:
        yield checkout
    finally:
        torch.hub.load = original_load


def _decoded_structure_resolution(value: object) -> int:
    """Return the cubic spatial extent of one TRELLIS decoder output.

    The released pipeline samples a dense latent at the flow model's
    ``resolution`` and then upsamples it in ``sparse_structure_decoder`` before
    taking ``argwhere``.  Consequently the flow resolution is not the integer
    coordinate domain of the returned sparse structure.
    """

    if not isinstance(value, Tensor):
        raise TypeError("TRELLIS decoded sparse-structure output must be a tensor")
    if value.ndim != 5:
        raise ValueError(
            "TRELLIS decoded sparse-structure output must have shape [B,C,D,H,W]"
        )
    if tuple(value.shape[:2]) != (1, 1):
        raise ValueError(
            "one-sample TRELLIS decoding requires one batch and one occupancy channel"
        )
    spatial_shape = tuple(int(size) for size in value.shape[-3:])
    if min(spatial_shape) < 1 or len(set(spatial_shape)) != 1:
        raise ValueError(
            "TRELLIS decoder output must use a non-empty cubic grid"
        )
    return spatial_shape[0]


def _python_source_tree_digest(root: Path) -> str:
    """Hash the complete imported pipeline package, not only ``__init__.py``."""

    digest = hashlib.sha256()
    sources = sorted(
        (path for path in root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not sources:
        return "unavailable"
    for path in sources:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _explicit_decoder_resolution(decoder: object) -> Optional[int]:
    """Read an unambiguous output-grid declaration for non-Module adapters."""

    value = getattr(decoder, "output_resolution", None)
    if type(value) is not int or value < 1:
        return None
    return value


@dataclass
class TrellisStructurePrior:
    coordinates: list[Tensor]
    resolution: int

    def validate(self) -> None:
        if self.resolution < 1:
            raise ValueError("TRELLIS structure resolution must be positive")
        if not self.coordinates:
            raise ValueError("TRELLIS structure prior requires at least one posterior sample")
        for sample in self.coordinates:
            if sample.ndim != 2 or sample.shape[1] != 3:
                raise ValueError("each TRELLIS structure sample must have shape [P,3]")
            if sample.dtype.is_floating_point:
                raise TypeError("TRELLIS structure coordinates must use an integer dtype")
            if torch.any(sample < 0) or torch.any(sample >= self.resolution):
                raise ValueError("TRELLIS structure coordinate lies outside its declared grid")


@dataclass
class TrellisPriorMeasure:
    """Sparse empirical support measure in the persistent atlas world gauge.

    ``probability`` is the Jeffreys-posterior mean for a cell that appeared in
    at least one TRELLIS sample. ``mass`` is probability times fine-cell area;
    it initializes atlas support statistics but is never appended to the image
    evidence measure used as the Sinkhorn target marginal.
    """

    coordinates: Tensor
    positions: Tensor
    probability: Tensor
    mass: Tensor
    mass_variance: Tensor
    vote_count: Tensor
    sample_count: int
    resolution: int

    def validate(self) -> None:
        count = self.coordinates.shape[0]
        if type(self.resolution) is not int or self.resolution < 1:
            raise ValueError("TRELLIS prior measure resolution must be positive")
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise ValueError("TRELLIS prior measure sample count must be positive")
        if tuple(self.coordinates.shape) != (count, 3):
            raise ValueError("prior coordinates must have shape [P,3]")
        if self.coordinates.dtype.is_floating_point:
            raise TypeError("prior coordinates must use an integer dtype")
        if torch.any(self.coordinates < 0) or torch.any(
            self.coordinates >= self.resolution
        ):
            raise ValueError("prior coordinates lie outside the decoded grid")
        if tuple(self.positions.shape) != (count, 3):
            raise ValueError("prior positions must have shape [P,3]")
        for name in ("probability", "mass", "mass_variance", "vote_count"):
            if tuple(getattr(self, name).shape) != (count,):
                raise ValueError(f"prior {name} must have shape [P]")
        if not torch.all(torch.isfinite(self.positions)):
            raise ValueError("TRELLIS prior positions contain non-finite values")
        for name in ("probability", "mass", "mass_variance"):
            if not torch.all(torch.isfinite(getattr(self, name))):
                raise ValueError(f"TRELLIS prior {name} contains non-finite values")
        if torch.any(self.probability <= 0) or torch.any(self.probability >= 1):
            raise ValueError("Jeffreys prior probabilities must lie strictly inside (0,1)")
        if torch.any(self.mass <= 0):
            raise ValueError("TRELLIS prior support mass must be positive")
        if torch.any(self.mass_variance < 0):
            raise ValueError("TRELLIS prior mass variance must be non-negative")
        if self.vote_count.dtype.is_floating_point:
            raise TypeError("TRELLIS prior vote counts must use an integer dtype")
        if torch.any(self.vote_count < 1) or torch.any(
            self.vote_count > self.sample_count
        ):
            raise ValueError("TRELLIS prior vote count is outside [1,sample_count]")


class TrellisPriorAdapter:
    def __init__(
        self,
        pipeline: object,
        samples: int = 8,
        sampler_steps: int = 12,
        strength: float = 0.35,
        minimum_probability: float = 0.0,
        uncertainty_discount: float = 0.5,
        cache_entries: int = 64,
        maximum_conditioning_views: Optional[int] = None,
        release_cuda_cache_after_sampling: bool = True,
        offload_cuda_pipeline_after_sampling: bool = True,
        persistent_cache_directory: Optional[str | Path] = None,
        persistent_cache_namespace: Optional[str] = None,
        persistent_cache_maximum_bytes: int = 64 * 1024**3,
    ) -> None:
        if samples < 1 or sampler_steps < 1:
            raise ValueError("TRELLIS prior samples and sampler steps must be positive")
        if strength < 0 or not 0.0 <= minimum_probability < 1.0 or uncertainty_discount < 0:
            raise ValueError("TRELLIS prior strength/threshold are outside their domains")
        if cache_entries < 0:
            raise ValueError("TRELLIS prior cache_entries must be non-negative")
        if (
            maximum_conditioning_views is not None
            and (
                isinstance(maximum_conditioning_views, bool)
                or not isinstance(maximum_conditioning_views, int)
                or maximum_conditioning_views < 1
            )
        ):
            raise ValueError(
                "TRELLIS maximum_conditioning_views must be a positive integer "
                "or None"
            )
        if not isinstance(release_cuda_cache_after_sampling, bool):
            raise TypeError("release_cuda_cache_after_sampling must be Boolean")
        if not isinstance(offload_cuda_pipeline_after_sampling, bool):
            raise TypeError("offload_cuda_pipeline_after_sampling must be Boolean")
        if persistent_cache_maximum_bytes < 1:
            raise ValueError("persistent TRELLIS cache byte bound must be positive")
        if (persistent_cache_directory is None) != (
            persistent_cache_namespace is None
        ):
            raise ValueError(
                "persistent TRELLIS cache directory and namespace must be paired"
            )
        self.pipeline = pipeline
        if pipeline is not None:
            self._validate_upstream_contract(pipeline)
        self.samples = samples
        self.sampler_steps = sampler_steps
        self.strength = strength
        self.minimum_probability = minimum_probability
        self.uncertainty_discount = uncertainty_discount
        self.cache_entries = cache_entries
        self.maximum_conditioning_views = maximum_conditioning_views
        self.release_cuda_cache_after_sampling = release_cuda_cache_after_sampling
        self.offload_cuda_pipeline_after_sampling = (
            offload_cuda_pipeline_after_sampling
        )
        self._pipeline_device: Optional[torch.device] = None
        self.progress_reporter: Optional[ProgressReporter] = None
        self._sampling_session_depth = 0
        self._sample_cache: OrderedDict[str, TrellisStructurePrior] = OrderedDict()
        self.persistent_cache_maximum_bytes = persistent_cache_maximum_bytes
        self._persistent_cache_namespace = persistent_cache_namespace
        self._persistent_cache_directory = (
            Path(persistent_cache_directory).expanduser().resolve()
            / str(persistent_cache_namespace)
            if persistent_cache_directory is not None
            else None
        )
        if self._persistent_cache_directory is not None:
            self._persistent_cache_directory.mkdir(parents=True, exist_ok=True)
        self.last_cuda_cache_release: dict[str, int | bool] = {
            "performed": False,
            "allocated_before_bytes": 0,
            "reserved_before_bytes": 0,
            "allocated_after_bytes": 0,
            "reserved_after_bytes": 0,
            "released_reserved_bytes": 0,
        }
        self.last_cuda_pipeline_offload: dict[str, int | bool] = {
            "performed": False,
            "allocated_before_bytes": 0,
            "peak_allocated_before_bytes": 0,
            "allocated_after_bytes": 0,
            "released_allocated_bytes": 0,
        }
        self.last_conditioning_view_count: dict[str, int] = {
            "available": 0,
            "selected": 0,
        }

    @contextmanager
    def sampling_session(self) -> Iterator[None]:
        """Defer CUDA offload/cache release across consecutive object samples.

        The pipeline remains frozen and every posterior call is unchanged. A
        multi-object forward merely avoids moving the same weights CPU→CUDA→CPU
        between adjacent samples, then restores the original lifetime boundary
        before VGGT begins.
        """

        self._sampling_session_depth += 1
        try:
            yield
        finally:
            self._sampling_session_depth -= 1
            if self._sampling_session_depth < 0:
                raise RuntimeError("TRELLIS sampling session depth underflow")
            if (
                self._sampling_session_depth == 0
                and self._pipeline_device is not None
                and self._pipeline_device.type == "cuda"
            ):
                device = self._pipeline_device
                offload_context = (
                    self.progress_reporter.stage("forward.trellis.offload")
                    if self.progress_reporter is not None
                    else nullcontext()
                )
                with offload_context:
                    self._offload_cuda_pipeline(device)
                release_context = (
                    self.progress_reporter.stage(
                        "forward.trellis.release_allocator_cache"
                    )
                    if self.progress_reporter is not None
                    else nullcontext()
                )
                with release_context:
                    self._release_inactive_cuda_cache(device)

    @staticmethod
    def _validate_upstream_contract(pipeline: object) -> None:
        methods = (
            "get_cond",
            "sample_sparse_structure",
            "inject_sampler_multi_image",
            "to",
        )
        missing = [name for name in methods if not callable(getattr(pipeline, name, None))]
        if missing:
            raise TypeError(
                "TRELLIS pipeline lacks required released methods: "
                + ",".join(missing)
            )
        models = getattr(pipeline, "models", None)
        required_models = {
            "image_cond_model",
            "sparse_structure_flow_model",
            "sparse_structure_decoder",
        }
        if not isinstance(models, dict) or not required_models <= set(models):
            raise TypeError("TRELLIS pipeline lacks required image/structure models")
        sampler = getattr(pipeline, "sparse_structure_sampler", None)
        if sampler is None or not callable(getattr(sampler, "sample", None)):
            raise TypeError("TRELLIS sparse-structure sampler is unavailable")
        latent_resolution = getattr(
            models["sparse_structure_flow_model"], "resolution", None
        )
        if type(latent_resolution) is not int or latent_resolution < 1:
            raise ValueError("TRELLIS sparse-structure latent resolution is invalid")
        decoder = models["sparse_structure_decoder"]
        if not callable(getattr(decoder, "register_forward_hook", None)) and (
            _explicit_decoder_resolution(decoder) is None
        ):
            raise TypeError(
                "TRELLIS sparse-structure decoder must expose its decoded grid "
                "through a forward hook or positive output_resolution"
            )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Optional[str] = None,
        samples: int = 8,
        sampler_steps: int = 12,
        strength: float = 0.35,
        minimum_probability: float = 0.0,
        uncertainty_discount: float = 0.5,
        cache_entries: int = 64,
        maximum_conditioning_views: Optional[int] = None,
        release_cuda_cache_after_sampling: bool = True,
        offload_cuda_pipeline_after_sampling: bool = True,
        persistent_cache_directory: Optional[str | Path] = None,
        persistent_cache_maximum_bytes: int = 64 * 1024**3,
        device: Optional[torch.device | str] = None,
    ) -> "TrellisPriorAdapter":
        checkpoint = resolve_trellis_checkpoint(checkpoint)
        module = import_external_module("trellis.pipelines")
        pipeline_class = getattr(module, "TrellisImageTo3DPipeline")
        # TRELLIS' released constructor calls
        # ``torch.hub.load('facebookresearch/dinov2', ...)``.  Resolve that
        # frozen dependency from the server cache without a GitHub probe.
        with _offline_torch_hub_repository("facebookresearch/dinov2"):
            pipeline = pipeline_class.from_pretrained(checkpoint)
        target_device = torch.device("cuda") if device is None else torch.device(device)
        initial_device = (
            torch.device("cpu")
            if offload_cuda_pipeline_after_sampling
            and target_device.type == "cuda"
            else target_device
        )
        pipeline.to(initial_device)
        provenance = external_module_provenance(module, checkpoint)
        module_path = Path(provenance["module_file"])
        source_digest = (
            _python_source_tree_digest(module_path.parent)
            if module_path.is_file()
            else "unavailable"
        )
        namespace_payload = {
            **provenance,
            "module_sha256": source_digest,
            "samples": samples,
            "sampler_steps": sampler_steps,
        }
        persistent_namespace = (
            hashlib.sha256(
                json.dumps(namespace_payload, sort_keys=True).encode("utf8")
            ).hexdigest()
            if persistent_cache_directory is not None
            else None
        )
        adapter = cls(
            pipeline,
            samples,
            sampler_steps,
            strength,
            minimum_probability,
            uncertainty_discount,
            cache_entries,
            maximum_conditioning_views,
            release_cuda_cache_after_sampling,
            offload_cuda_pipeline_after_sampling,
            persistent_cache_directory,
            persistent_namespace,
            persistent_cache_maximum_bytes,
        )
        adapter._pipeline_device = initial_device
        adapter.upstream_provenance = provenance
        adapter.persistent_cache_provenance = namespace_payload
        return adapter

    @torch.no_grad()
    def sample(self, scene_images: Tensor, seed: int = 0) -> TrellisStructurePrior:
        if scene_images.ndim != 4:
            raise ValueError("scene_images must have shape [K,3,H,W]")
        if scene_images.shape[0] < 1 or scene_images.shape[1] != 3:
            raise ValueError("scene_images must contain at least one RGB view")
        if not scene_images.dtype.is_floating_point:
            raise TypeError("TRELLIS image conditioning requires floating-point RGB")
        if not bool(torch.all(torch.isfinite(scene_images))):
            raise ValueError("TRELLIS image conditioning contains non-finite values")
        if bool(torch.any(scene_images < 0)) or bool(torch.any(scene_images > 1)):
            raise ValueError("TRELLIS tensor inputs must use the released [0,1] RGB contract")
        available_views = int(scene_images.shape[0])
        scene_images = self._select_conditioning_views(scene_images)
        self.last_conditioning_view_count = {
            "available": available_views,
            "selected": int(scene_images.shape[0]),
        }
        cache_key = self._sample_cache_key(scene_images, seed)
        if cache_key is not None and cache_key in self._sample_cache:
            if self.progress_reporter is not None:
                self.progress_reporter.event(
                    "forward.trellis.cache",
                    "hit",
                    cache_key=cache_key,
                    seed=seed,
                    selected_views=int(scene_images.shape[0]),
                )
            cached = self._sample_cache.pop(cache_key)
            self._sample_cache[cache_key] = cached
            return TrellisStructurePrior(
                [value.to(device=scene_images.device).clone() for value in cached.coordinates],
                cached.resolution,
            )
        if cache_key is not None:
            persistent = self._load_persistent_sample(cache_key)
            if persistent is not None:
                if self.progress_reporter is not None:
                    self.progress_reporter.event(
                        "forward.trellis.cache",
                        "persistent_hit",
                        cache_key=cache_key,
                        seed=seed,
                        selected_views=int(scene_images.shape[0]),
                    )
                self._remember_sample(cache_key, persistent)
                return TrellisStructurePrior(
                    [
                        value.to(device=scene_images.device).clone()
                        for value in persistent.coordinates
                    ],
                    persistent.resolution,
                )
        if self.pipeline is None:
            raise RuntimeError(
                "this synchronized TRELLIS proxy cannot sample; only the "
                "designated distributed source rank may own the checkpoint"
            )
        if self.progress_reporter is not None:
            self.progress_reporter.event(
                "forward.trellis.cache",
                "miss",
                cache_key=cache_key,
                seed=seed,
                selected_views=int(scene_images.shape[0]),
                posterior_draws=self.samples,
                sampler_steps=self.sampler_steps,
            )
        device_context = (
            self.progress_reporter.stage("forward.trellis.device_load")
            if self.progress_reporter is not None
            else nullcontext()
        )
        with device_context:
            self._ensure_pipeline_device(scene_images.device)
        conditioning_context = (
            self.progress_reporter.stage(
                "forward.trellis.conditioning",
                selected_views=int(scene_images.shape[0]),
            )
            if self.progress_reporter is not None
            else nullcontext()
        )
        with conditioning_context:
            condition = self.pipeline.get_cond(scene_images)
        if not isinstance(condition, dict) or not {"cond", "neg_cond"} <= set(condition):
            raise TypeError("TRELLIS get_cond must return cond and neg_cond tensors")
        condition["neg_cond"] = condition["neg_cond"][:1]
        structures = []
        decoded_resolutions: list[int] = []
        decoder = self.pipeline.models["sparse_structure_decoder"]
        register_hook = getattr(decoder, "register_forward_hook", None)
        hook_handle = None
        if callable(register_hook):
            def record_decoded_resolution(
                _module: object,
                _arguments: tuple[object, ...],
                output: object,
            ) -> None:
                decoded_resolutions.append(_decoded_structure_resolution(output))

            hook_handle = register_hook(record_decoded_resolution)
        parameters = {"steps": self.sampler_steps}
        try:
            for sample_index in range(self.samples):
                # The released injector installs run-local sampler state. Recreate
                # that state for every posterior draw instead of reusing a context
                # whose callback counters/conditioning schedule have been consumed.
                context = self.pipeline.inject_sampler_multi_image(
                    "sparse_structure_sampler",
                    scene_images.shape[0],
                    self.sampler_steps,
                    mode="multidiffusion",
                ) if scene_images.shape[0] > 1 else nullcontext()
                progress_context = (
                    self.progress_reporter.stage(
                        "forward.trellis.posterior_draw",
                        draw_index=sample_index,
                        draw_count=self.samples,
                        sampler_steps=self.sampler_steps,
                        seed=seed + sample_index,
                    )
                    if self.progress_reporter is not None
                    else nullcontext()
                )
                with progress_context, context:
                    devices = [scene_images.device] if scene_images.is_cuda else []
                    # TRELLIS does not expose a generator argument. Isolate its
                    # sampling RNG so topology priors cannot perturb flow-time or
                    # training augmentation randomness in the surrounding model.
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(seed + sample_index)
                        coordinates = self.pipeline.sample_sparse_structure(
                            condition, 1, parameters
                        )
                    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
                        raise ValueError(
                            "TRELLIS sparse_structure output must have shape [P,4]"
                        )
                    if torch.any(coordinates[:, 0] != 0):
                        raise ValueError(
                            "one-sample TRELLIS structure contains a nonzero batch index"
                        )
                    structures.append(coordinates[:, 1:].to(torch.int64))
                if self.progress_reporter is not None:
                    self.progress_reporter.event(
                        "forward.trellis.posterior_draw",
                        "decoded",
                        draw_index=sample_index,
                        draw_count=self.samples,
                        support_points=int(structures[-1].shape[0]),
                    )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        if hook_handle is not None:
            if len(decoded_resolutions) != len(structures):
                raise RuntimeError(
                    "TRELLIS sparse-structure decoder did not execute exactly once "
                    "per posterior draw"
                )
            if len(set(decoded_resolutions)) != 1:
                raise RuntimeError(
                    "TRELLIS decoder changed grid resolution "
                    "between posterior draws"
                )
            resolution = decoded_resolutions[0]
        else:
            resolution = _explicit_decoder_resolution(decoder)
            if resolution is None:  # guarded by _validate_upstream_contract
                raise RuntimeError("TRELLIS decoded structure resolution is unavailable")
        prior = TrellisStructurePrior(structures, resolution)
        prior.validate()
        if self.progress_reporter is not None:
            self.progress_reporter.event(
                "forward.trellis.sample",
                "complete",
                posterior_draws=len(structures),
                decoded_resolution=resolution,
                total_support_points=sum(int(value.shape[0]) for value in structures),
            )
        if cache_key is not None:
            cached_prior = TrellisStructurePrior(
                [value.detach().to(device="cpu").clone() for value in prior.coordinates],
                prior.resolution,
            )
            self._remember_sample(cache_key, cached_prior)
            self._store_persistent_sample(cache_key, cached_prior)
        # TRELLIS sampling is a frozen, no-grad upstream operation. Its diffusion
        # workspaces are dead here, but PyTorch's caching allocator otherwise
        # keeps those blocks reserved on the source DDP rank. That rank-local
        # cache can consume almost the entire A800 and starve later native CUDA
        # rasterizers even though the active GRAFT-GS state is much smaller.
        # Emptying the cache cannot free live tensors (including ``prior`` or
        # model parameters), so this changes allocator ownership only—not
        # values, precision, gradients, or posterior samples.
        del condition
        if "coordinates" in locals():
            del coordinates
        if self._sampling_session_depth == 0:
            self._offload_cuda_pipeline(scene_images.device)
            self._release_inactive_cuda_cache(scene_images.device)
        return prior

    def _select_conditioning_views(self, scene_images: Tensor) -> Tensor:
        """Select a deterministic coverage subset for the frozen shape prior.

        Every input view still enters VGGT and the calibrated geometric
        evidence measure. TRELLIS is not observed evidence: it supplies a
        stochastic hidden-surface prior whose multi-image conditioning
        workspace grows with view count. Uniform endpoint-preserving sampling
        bounds that prior-only workspace without changing RGB supervision,
        cameras, transport mass, or any differentiable GRAFT-GS state.
        """

        maximum = self.maximum_conditioning_views
        count = int(scene_images.shape[0])
        if maximum is None or count <= maximum:
            return scene_images
        if maximum == 1:
            index = torch.tensor(
                [count // 2],
                dtype=torch.int64,
                device=scene_images.device,
            )
        else:
            numerator = torch.arange(
                maximum,
                dtype=torch.int64,
                device=scene_images.device,
            ) * (count - 1)
            index = torch.div(
                numerator,
                maximum - 1,
                rounding_mode="floor",
            )
        if int(torch.unique(index).numel()) != maximum:
            raise RuntimeError("TRELLIS conditioning subset contains duplicates")
        return scene_images.index_select(0, index)

    def _ensure_pipeline_device(self, device: torch.device) -> None:
        if self.pipeline is None:
            raise RuntimeError("TRELLIS sampling pipeline is unavailable")
        device = torch.device(device)
        if self._pipeline_device == device:
            return
        self.pipeline.to(device)
        self._pipeline_device = device

    def _offload_cuda_pipeline(self, sampling_device: torch.device) -> None:
        if (
            self.pipeline is None
            or not self.offload_cuda_pipeline_after_sampling
            or sampling_device.type != "cuda"
            or not torch.cuda.is_available()
        ):
            self.last_cuda_pipeline_offload = {
                "performed": False,
                "allocated_before_bytes": 0,
                "peak_allocated_before_bytes": 0,
                "allocated_after_bytes": 0,
                "released_allocated_bytes": 0,
            }
            return
        torch.cuda.synchronize(sampling_device)
        allocated_before = int(torch.cuda.memory_allocated(sampling_device))
        peak_allocated_before = int(
            torch.cuda.max_memory_allocated(sampling_device)
        )
        self.pipeline.to(torch.device("cpu"))
        self._pipeline_device = torch.device("cpu")
        torch.cuda.synchronize(sampling_device)
        allocated_after = int(torch.cuda.memory_allocated(sampling_device))
        if allocated_after > allocated_before:
            raise RuntimeError(
                "offloading TRELLIS increased live CUDA allocation accounting"
            )
        self.last_cuda_pipeline_offload = {
            "performed": True,
            "allocated_before_bytes": allocated_before,
            "peak_allocated_before_bytes": peak_allocated_before,
            "allocated_after_bytes": allocated_after,
            "released_allocated_bytes": allocated_before - allocated_after,
        }

    def _release_inactive_cuda_cache(self, device: torch.device) -> None:
        if (
            not self.release_cuda_cache_after_sampling
            or device.type != "cuda"
            or not torch.cuda.is_available()
        ):
            self.last_cuda_cache_release = {
                "performed": False,
                "allocated_before_bytes": 0,
                "reserved_before_bytes": 0,
                "allocated_after_bytes": 0,
                "reserved_after_bytes": 0,
                "released_reserved_bytes": 0,
            }
            return
        torch.cuda.synchronize(device)
        allocated_before = int(torch.cuda.memory_allocated(device))
        reserved_before = int(torch.cuda.memory_reserved(device))
        torch.cuda.empty_cache()
        allocated_after = int(torch.cuda.memory_allocated(device))
        reserved_after = int(torch.cuda.memory_reserved(device))
        if allocated_after != allocated_before:
            raise RuntimeError(
                "empty_cache changed live TRELLIS allocation accounting; "
                "the frozen-prior lifetime boundary is invalid"
            )
        self.last_cuda_cache_release = {
            "performed": True,
            "allocated_before_bytes": allocated_before,
            "reserved_before_bytes": reserved_before,
            "allocated_after_bytes": allocated_after,
            "reserved_after_bytes": reserved_after,
            "released_reserved_bytes": max(reserved_before - reserved_after, 0),
        }

    def _sample_cache_key(self, scene_images: Tensor, seed: int) -> Optional[str]:
        """Hash exact conditioning values; cache hits never approximate images."""

        if self.cache_entries == 0:
            return None
        values = scene_images.detach().to(device="cpu").contiguous()
        digest = hashlib.sha256()
        digest.update(str(tuple(values.shape)).encode("ascii"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(
            f"{int(seed)}\0{self.samples}\0{self.sampler_steps}\0".encode("ascii")
        )
        digest.update(values.view(torch.uint8).numpy().tobytes(order="C"))
        return digest.hexdigest()

    def _remember_sample(
        self,
        cache_key: str,
        prior: TrellisStructurePrior,
    ) -> None:
        if self.cache_entries == 0:
            return
        self._sample_cache[cache_key] = TrellisStructurePrior(
            [value.detach().to(device="cpu").clone() for value in prior.coordinates],
            prior.resolution,
        )
        while len(self._sample_cache) > self.cache_entries:
            self._sample_cache.popitem(last=False)

    @contextmanager
    def _persistent_cache_lock(self) -> Iterator[None]:
        directory = self._persistent_cache_directory
        if directory is None:
            yield
            return
        lock_path = directory / ".cache.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _load_persistent_sample(
        self,
        cache_key: str,
    ) -> Optional[TrellisStructurePrior]:
        directory = self._persistent_cache_directory
        if directory is None:
            return None
        path = directory / f"{cache_key}.pt"
        with self._persistent_cache_lock():
            if not path.is_file():
                return None
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except BaseException as error:
                raise RuntimeError(
                    f"invalid persistent TRELLIS cache entry {path}"
                ) from error
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "graft-gs-trellis-exact-cache-v1"
            or payload.get("namespace") != self._persistent_cache_namespace
            or payload.get("cache_key") != cache_key
            or not isinstance(payload.get("coordinates"), list)
        ):
            raise RuntimeError(f"persistent TRELLIS cache provenance mismatch: {path}")
        prior = TrellisStructurePrior(
            coordinates=payload["coordinates"],
            resolution=int(payload["resolution"]),
        )
        prior.validate()
        return prior

    def _store_persistent_sample(
        self,
        cache_key: str,
        prior: TrellisStructurePrior,
    ) -> None:
        directory = self._persistent_cache_directory
        if directory is None:
            return
        path = directory / f"{cache_key}.pt"
        with self._persistent_cache_lock():
            if path.is_file():
                return
            temporary = directory / f".{cache_key}.{os.getpid()}.tmp"
            try:
                torch.save(
                    {
                        "schema": "graft-gs-trellis-exact-cache-v1",
                        "namespace": self._persistent_cache_namespace,
                        "cache_key": cache_key,
                        "resolution": prior.resolution,
                        "coordinates": [
                            value.detach().to(device="cpu").clone()
                            for value in prior.coordinates
                        ],
                    },
                    temporary,
                )
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            entries = sorted(
                directory.glob("*.pt"),
                key=lambda item: (item.stat().st_mtime_ns, item.name),
            )
            total_bytes = sum(item.stat().st_size for item in entries)
            for entry in entries:
                if total_bytes <= self.persistent_cache_maximum_bytes:
                    break
                size = entry.stat().st_size
                entry.unlink()
                total_bytes -= size

    def support_measure(
        self,
        prior: TrellisStructurePrior,
        root_min: Tensor,
        root_max: Tensor,
        minimum_probability: Optional[float] = None,
    ) -> TrellisPriorMeasure:
        """Convert sampled structures into a deterministic sparse area measure."""

        prior.validate()
        minimum_probability = (
            self.minimum_probability if minimum_probability is None else minimum_probability
        )
        if not 0.0 <= minimum_probability < 1.0:
            raise ValueError("minimum_probability must lie in [0,1)")
        root_min = root_min.reshape(3)
        root_max = root_max.to(device=root_min.device, dtype=root_min.dtype).reshape(3)
        extent = root_max - root_min
        if torch.any(extent <= 0) or not torch.allclose(
            extent, extent.max().expand_as(extent), atol=1.0e-6, rtol=0.0
        ):
            raise ValueError("TRELLIS support requires a non-empty cubic atlas root")
        linear_samples = []
        for sample in prior.coordinates:
            coordinates = sample.to(device=root_min.device, dtype=torch.int64)
            linear_samples.append(
                torch.unique(
                    (coordinates[:, 0] * prior.resolution + coordinates[:, 1])
                    * prior.resolution
                    + coordinates[:, 2],
                    sorted=True,
                )
            )
        linear = torch.cat(linear_samples, dim=0)
        unique_linear, votes = torch.unique(linear, sorted=True, return_counts=True)
        x = torch.div(unique_linear, prior.resolution**2, rounding_mode="floor")
        remainder = unique_linear.remainder(prior.resolution**2)
        y = torch.div(remainder, prior.resolution, rounding_mode="floor")
        z = remainder.remainder(prior.resolution)
        unique_coordinates = torch.stack((x, y, z), dim=-1)
        alpha = votes.to(root_min.dtype) + 0.5
        beta = len(prior.coordinates) - votes.to(root_min.dtype) + 0.5
        probability = alpha / (alpha + beta)
        probability_variance = alpha * beta / (
            (alpha + beta).square() * (alpha + beta + 1.0)
        )
        retain = probability >= minimum_probability
        unique_coordinates = unique_coordinates[retain]
        votes = votes[retain]
        probability = probability[retain]
        probability_variance = probability_variance[retain]
        if unique_coordinates.shape[0] == 0:
            raise RuntimeError("TRELLIS posterior threshold removed every sampled support cell")
        positions = root_min + (
            (unique_coordinates.to(root_min.dtype) + 0.5) / float(prior.resolution)
        ) * extent
        fine_cell_area = (extent.max() / float(prior.resolution)).square()
        measure = TrellisPriorMeasure(
            coordinates=unique_coordinates,
            positions=positions,
            probability=probability,
            mass=probability * fine_cell_area,
            mass_variance=probability_variance * fine_cell_area.pow(2),
            vote_count=votes,
            sample_count=len(prior.coordinates),
            resolution=prior.resolution,
        )
        measure.validate()
        return measure

    def node_probability(self, atlas: PersistentOctreeAtlas) -> Tensor:
        """Return active-chart probability from persistent prior surface mass."""
        active = atlas.active_indices
        area = torch.pi * atlas.chart_radii[active].square()
        conservative_mass = (
            atlas.prior_mass[active]
            - self.uncertainty_discount
            * torch.sqrt(atlas.prior_mass_variance[active].clamp_min(0.0))
        ).clamp_min(0.0)
        hazard = conservative_mass / area.clamp_min(torch.finfo(area.dtype).eps)
        return -torch.expm1(-hazard)

    def node_shape_probability(
        self,
        atlas: PersistentOctreeAtlas,
        sample_count: int,
    ) -> Tensor:
        """Jeffreys-smoothed active-node Bernoulli field for topology energy.

        ``node_probability`` is a surface-mass hazard and is exactly zero where
        no admitted prior point landed.  For a shape likelihood, zero votes are
        evidence of absence but not certainty; the Beta(1/2,1/2) posterior mean
        is ``0.5/(S+1)`` after ``S`` structure samples.
        """

        if sample_count < 1:
            raise ValueError("shape probability requires a positive sample count")
        probability = self.node_probability(atlas)
        active = atlas.active_indices
        zero_vote_probability = probability.new_tensor(0.5 / (sample_count + 1.0))
        probability = torch.where(
            atlas.prior_point_count[active] > 0,
            probability,
            zero_vote_probability,
        )
        return probability.clamp(1.0e-6, 1.0 - 1.0e-6)

    def combine_observed_probability(self, observed: Tensor, prior_probability: Tensor) -> Tensor:
        if observed.shape != prior_probability.shape:
            raise ValueError("observed and prior occupancy must share the active atlas support")
        if self.strength < 0:
            raise ValueError("TRELLIS prior strength must be non-negative")
        # Independent surface hazards compose by multiplying absence
        # probabilities. A missing/weak prior can never erase observed
        # geometry, unlike adding negative logits for probabilities below .5.
        observed = observed.clamp(0.0, 1.0)
        prior_probability = prior_probability.clamp(0.0, 1.0 - 1.0e-7)
        prior_absence = (1.0 - prior_probability).pow(self.strength)
        return 1.0 - (1.0 - observed) * prior_absence


__all__ = ["TrellisPriorAdapter", "TrellisPriorMeasure", "TrellisStructurePrior"]

"""Use TRELLIS structure generation as a discrete hidden-surface prior.

TRELLIS does not decode final Gaussians or a mesh in GRAFT-GS.  Its sampled
sparse structures define a prior measure over canonical occupied cells. That
measure is aligned to the evidence root cube and combined with observed atlas
occupancy as an additive surface hazard before topology proposal.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..geometry.atlas import PersistentOctreeAtlas


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
        if tuple(self.coordinates.shape) != (count, 3):
            raise ValueError("prior coordinates must have shape [P,3]")
        if tuple(self.positions.shape) != (count, 3):
            raise ValueError("prior positions must have shape [P,3]")
        for name in ("probability", "mass", "mass_variance", "vote_count"):
            if tuple(getattr(self, name).shape) != (count,):
                raise ValueError(f"prior {name} must have shape [P]")
        if torch.any(self.probability <= 0) or torch.any(self.probability >= 1):
            raise ValueError("Jeffreys prior probabilities must lie strictly inside (0,1)")
        if torch.any(self.mass <= 0):
            raise ValueError("TRELLIS prior support mass must be positive")
        if torch.any(self.mass_variance < 0):
            raise ValueError("TRELLIS prior mass variance must be non-negative")


class TrellisPriorAdapter:
    def __init__(
        self,
        pipeline: object,
        samples: int = 8,
        sampler_steps: int = 12,
        strength: float = 0.35,
        minimum_probability: float = 0.0,
        uncertainty_discount: float = 0.5,
    ) -> None:
        if samples < 1 or sampler_steps < 1:
            raise ValueError("TRELLIS prior samples and sampler steps must be positive")
        if strength < 0 or not 0.0 <= minimum_probability < 1.0 or uncertainty_discount < 0:
            raise ValueError("TRELLIS prior strength/threshold are outside their domains")
        self.pipeline = pipeline
        self.samples = samples
        self.sampler_steps = sampler_steps
        self.strength = strength
        self.minimum_probability = minimum_probability
        self.uncertainty_discount = uncertainty_discount

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "microsoft/TRELLIS-image-large",
        samples: int = 8,
        sampler_steps: int = 12,
        strength: float = 0.35,
        minimum_probability: float = 0.0,
        uncertainty_discount: float = 0.5,
        device: Optional[torch.device | str] = None,
    ) -> "TrellisPriorAdapter":
        from trellis.pipelines import TrellisImageTo3DPipeline

        pipeline = TrellisImageTo3DPipeline.from_pretrained(checkpoint)
        pipeline.to(torch.device("cuda") if device is None else torch.device(device))
        return cls(
            pipeline,
            samples,
            sampler_steps,
            strength,
            minimum_probability,
            uncertainty_discount,
        )

    @torch.no_grad()
    def sample(self, scene_images: Tensor, seed: int = 0) -> TrellisStructurePrior:
        if scene_images.ndim != 4:
            raise ValueError("scene_images must have shape [K,3,H,W]")
        condition = self.pipeline.get_cond(scene_images)
        condition["neg_cond"] = condition["neg_cond"][:1]
        structures = []
        parameters = {"steps": self.sampler_steps}
        sampler = self.pipeline.sparse_structure_sampler
        context = self.pipeline.inject_sampler_multi_image(
            "sparse_structure_sampler",
            scene_images.shape[0],
            self.sampler_steps,
            mode="multidiffusion",
        ) if scene_images.shape[0] > 1 else nullcontext()
        with context:
            for sample_index in range(self.samples):
                devices = [scene_images.device] if scene_images.is_cuda else []
                # TRELLIS does not expose a generator argument. Isolate its
                # sampling RNG so topology priors cannot perturb flow-time or
                # training augmentation randomness in the surrounding model.
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed + sample_index)
                    coordinates = self.pipeline.sample_sparse_structure(condition, 1, parameters)
                if coordinates.ndim != 2 or coordinates.shape[1] != 4:
                    raise ValueError("TRELLIS sparse_structure output must have shape [P,4]")
                if torch.any(coordinates[:, 0] != 0):
                    raise ValueError("one-sample TRELLIS structure contains a nonzero batch index")
                structures.append(coordinates[:, 1:].to(torch.int64))
        resolution = int(self.pipeline.models["sparse_structure_flow_model"].resolution)
        prior = TrellisStructurePrior(structures, resolution)
        prior.validate()
        return prior

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

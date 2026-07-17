"""Run structural GRAFT-GS ablations without changing the defining reference path."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from graft_gs.engine import (
    load_graft_checkpoint,
    load_server_config,
    load_trellis_prior_config,
    validate_trellis_prior_policy,
)
from graft_gs.integration import GraftGS, TrellisPriorAdapter, VGGTAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--vggt-checkpoint", default="facebook/VGGT-1B")
    parser.add_argument("--trellis-checkpoint")
    parser.add_argument("--graft-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/graft_gs_a800_native.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ablations.json"))
    args = parser.parse_args()
    from vggt.utils.load_fn import load_and_preprocess_images

    paths = sorted(path for path in args.image_directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})[:8]
    if len(paths) < 2:
        raise ValueError("ablations require at least two real views")
    device = torch.device("cuda")
    images = load_and_preprocess_images([str(path) for path in paths])[None].to(device)
    base, _, _, _ = load_server_config(args.config)
    prior_config = load_trellis_prior_config(args.config)
    if bool(prior_config["enabled_after_phase_a"]) and args.trellis_checkpoint is None:
        raise ValueError("reference ablation requires --trellis-checkpoint")
    prior = (
        TrellisPriorAdapter.from_pretrained(
            args.trellis_checkpoint,
            samples=int(prior_config["samples"]),
            sampler_steps=int(prior_config["sampler_steps"]),
            strength=float(prior_config["strength"]),
            minimum_probability=float(prior_config["minimum_probability"]),
            uncertainty_discount=float(prior_config["uncertainty_discount"]),
            device=device,
        )
        if args.trellis_checkpoint
        else None
    )
    adapter = VGGTAdapter.from_pretrained(
        args.vggt_checkpoint, feature_dim=base.feature_dim
    )
    variants = {
        "reference": (base, prior),
        "no_hidden_surface_prior": (base, None),
        "no_continuous_flow": (replace(base, run_flow=False), prior),
        "high_entropy_transport": (replace(
            base,
            mapping=replace(base.mapping, sinkhorn=replace(base.mapping.sinkhorn, epsilon=0.12)),
        ), prior),
        "near_balanced_transport": (replace(
            base,
            mapping=replace(
                base.mapping,
                sinkhorn=replace(base.mapping.sinkhorn, tau_source=20.0, tau_target=20.0),
            ),
        ), prior),
        "fixed_coarse_atlas": (replace(base, refinement_rounds=0), prior),
        "no_transport_uncertainty_attention_bias": (
            replace(
                base,
                attention=replace(
                    base.attention,
                    ot_bias_weight=0.0,
                    uncertainty_bias_weight=0.0,
                ),
            ),
            prior,
        ),
        "fixed_topology_thresholds_only": (
            replace(
                base,
                topology=replace(
                    base.topology,
                    adaptive_threshold_quantiles=(),
                    maximum_persistence_thresholds=0,
                ),
            ),
            prior,
        ),
    }
    records = {}
    for name, (config, variant_prior) in variants.items():
        model = GraftGS(adapter, config, variant_prior)
        checkpoint_payload, checkpoint_report = load_graft_checkpoint(
            model,
            args.graft_checkpoint,
            map_location="cpu",
            strict=True,
            # Ablations intentionally alter non-shape hyperparameters while
            # preserving the trained parameter contract.
            validate_model_config=name == "reference",
        )
        validate_trellis_prior_policy(
            checkpoint_payload,
            enabled=variant_prior is not None,
            samples=int(prior_config["samples"]),
            sampler_steps=int(prior_config["sampler_steps"]),
            strength=float(prior_config["strength"]),
            minimum_probability=float(prior_config["minimum_probability"]),
            uncertainty_discount=float(prior_config["uncertainty_discount"]),
        )
        model = model.to(device).eval()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.no_grad():
            scene = model(images).scenes[0]
        torch.cuda.synchronize()
        barrier = scene.atlas.chart_immersion_margin()
        records[name] = {
            "seconds": time.perf_counter() - start,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "active_charts": scene.atlas.num_active,
            "transport_edges": scene.mapping.graph.num_edges,
            "transport_residual": scene.mapping.diagnostics.fixed_point_residual,
            "trellis_prior_support_count": scene.trellis_prior_support_count,
            "trellis_prior_expected_mass": scene.trellis_prior_expected_mass,
            "prior_only_active_charts": int(
                torch.sum(
                    (scene.atlas.evidence_mass[scene.atlas.active_indices] == 0)
                    & (scene.atlas.prior_mass[scene.atlas.active_indices] > 0)
                ).item()
            ),
            "observation_reliability_mean": float(
                scene.mapping.observation_reliability.mean().detach().cpu()
            ),
            "betti": scene.topology.selected.betti,
            "minimum_immersion_margin": float(barrier.min()),
            "feasibility": scene.feasibility_reports[-1].__dict__,
            "atlas_rejected_evidence_count": scene.atlas_rejected_evidence_count,
            "atlas_rejected_evidence_mass": scene.atlas_rejected_evidence_mass,
            "gaussians": int(scene.gaussians.means.shape[0]),
            "faces": int(scene.mesh.faces.shape[0]),
            "checkpoint_phase": checkpoint_report.phase,
            "checkpoint_step": checkpoint_report.global_step,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()

"""Native-precision A800 multiview inference and deterministic asset export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from graft_gs.engine import (
    load_graft_checkpoint,
    load_precision_policy,
    load_server_config,
    load_trellis_prior_config,
    validate_precision_policy,
    validate_trellis_prior_policy,
)
from graft_gs.integration import (
    GraftGS,
    TrellisPriorAdapter,
    VGGTAdapter,
    import_external_module,
    resolve_trellis_checkpoint,
    resolve_vggt_checkpoint,
)
from graft_gs.manifold import BarrierProjector
from graft_gs.optimization import certify_topology_quantization_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--vggt-checkpoint")
    parser.add_argument("--trellis-checkpoint")
    parser.add_argument("--graft-checkpoint", type=Path)
    parser.add_argument("--allow-untrained-graft-heads", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/graft_gs_a800_native.yaml"))
    parser.add_argument("--maximum-views", type=int, default=8)
    parser.add_argument("--render-input-views", action="store_true")
    parser.add_argument("--quantization-query-error", type=float)
    parser.add_argument("--vector-field-lipschitz-bound", type=float)
    args = parser.parse_args()
    args.vggt_checkpoint = resolve_vggt_checkpoint(args.vggt_checkpoint)
    if (args.quantization_query_error is None) != (
        args.vector_field_lipschitz_bound is None
    ):
        raise ValueError(
            "quantization certification requires both --quantization-query-error "
            "and --vector-field-lipschitz-bound"
        )
    load_and_preprocess_images = getattr(
        import_external_module("vggt.utils.load_fn"), "load_and_preprocess_images"
    )

    paths = sorted(path for path in args.image_directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if len(paths) < 2:
        raise ValueError("multiview inference requires at least two images")
    paths = paths[: args.maximum_views]
    device = torch.device("cuda")
    model_config, _, _, _ = load_server_config(args.config)
    precision_policy = load_precision_policy(args.config)
    precision_policy.apply()
    adapter = VGGTAdapter.from_pretrained(
        args.vggt_checkpoint,
        feature_dim=model_config.feature_dim,
        backbone_dtype=precision_policy.backbone_dtype,
    )
    prior_config = load_trellis_prior_config(args.config)
    use_prior = bool(prior_config["enabled_after_phase_a"])
    if use_prior:
        args.trellis_checkpoint = resolve_trellis_checkpoint(args.trellis_checkpoint)
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
        if use_prior
        else None
    )
    model = GraftGS(adapter, model_config, prior)
    checkpoint_report = None
    if args.graft_checkpoint is not None:
        checkpoint_payload, checkpoint_report = load_graft_checkpoint(
            model,
            args.graft_checkpoint,
            map_location="cpu",
            strict=True,
        )
        validate_precision_policy(checkpoint_payload, precision_policy)
        validate_trellis_prior_policy(
            checkpoint_payload,
            enabled=prior is not None,
            samples=int(prior_config["samples"]),
            sampler_steps=int(prior_config["sampler_steps"]),
            strength=float(prior_config["strength"]),
            minimum_probability=float(prior_config["minimum_probability"]),
            uncertainty_discount=float(prior_config["uncertainty_discount"]),
        )
    elif not args.allow_untrained_graft_heads:
        raise ValueError(
            "real inference requires --graft-checkpoint; use --allow-untrained-graft-heads only for architecture smoke tests"
        )
    model = model.to(device).eval()
    images = load_and_preprocess_images([str(path) for path in paths]).to(device)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        output = model(images[None], render_input_views=args.render_input_views)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ply, glb = output.scenes[0].export(args.output_directory)
    scene = output.scenes[0]
    quantization_topology_certificate = None
    if args.quantization_query_error is not None:
        attention_temperature = min(
            model_config.attention.attention_temperature_scalar,
            model_config.attention.attention_temperature_vector,
            model_config.attention.attention_temperature_tensor,
        )
        integration_step = 1.0 / model_config.flow.steps
        certificate = certify_topology_quantization_step(
            BarrierProjector(scene.initial_state, model_config.barrier),
            scene.final_state,
            query_error=args.quantization_query_error,
            temperature=attention_temperature,
            vector_field_lipschitz=args.vector_field_lipschitz_bound,
            step_size=integration_step,
        )
        quantization_topology_certificate = certificate.to_dict() | {
            "attention_temperature": attention_temperature,
            "integration_step": integration_step,
            "lipschitz_bound_provenance": "explicit_server_measurement",
        }
    render_paths = []
    if scene.render is not None:
        from torchvision.utils import save_image

        render_directory = args.output_directory / "renders"
        render_directory.mkdir(parents=True, exist_ok=True)
        for view, image in enumerate(scene.render.color):
            path = render_directory / f"view-{view:04d}.png"
            save_image(image.clamp(0.0, 1.0), path)
            render_paths.append(str(path))
    metrics = {
        "seconds": elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "active_charts": scene.atlas.num_active,
        "transport_edges": scene.mapping.graph.num_edges,
        "transport_iterations": scene.mapping.diagnostics.iterations,
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
        "feasibility": scene.feasibility_reports[-1].__dict__,
        "quantization_topology_certificate": quantization_topology_certificate,
        "gaussians": scene.gaussians.means.shape[0],
        "faces": scene.mesh.faces.shape[0],
        "ply": str(ply),
        "glb": str(glb),
        "renders": render_paths,
        "checkpoint": checkpoint_report.__dict__ if checkpoint_report is not None else None,
    }
    metrics_path = args.output_directory / "inference_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf8")
    print(metrics)


if __name__ == "__main__":
    main()

# Exact dynamically visible A800 validation protocol

This protocol defines server-ready commands; none of the resulting measurements
are claimed until their logs and artifacts exist. Run from the GRAFT-GS root in
the pinned VGGT/TRELLIS/PyTorch environment.

## Required environment

```bash
export GRAFT_GS_VGGT_CHECKPOINT=facebook/VGGT-1B
export GRAFT_GS_TRELLIS_CHECKPOINT=microsoft/TRELLIS-image-large
export VGGT_CHECKPOINT="$GRAFT_GS_VGGT_CHECKPOINT"
export TRELLIS_CHECKPOINT="$GRAFT_GS_TRELLIS_CHECKPOINT"
export GRAFT_GS_CHECKPOINT=/checkpoints/graft-gs-phase-f.pt
export GRAFT_GS_REAL_IMAGE_DIR=/data/real_multiview_object/images
export GRAFT_GS_MESHFLEET_ROOT=/mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97
export GRAFT_GS_MESHFLEET_MANIFEST=$PWD/outputs/validation/meshfleet_server.jsonl
export GRAFT_GS_VGGT_ROOT=/mnt/sda2/hef/Base/vggt
export GRAFT_GS_TRELLIS_ROOT=/mnt/sda2/hef/Base/TRELLIS
export GRAFT_GS_TEACHER_BUNDLES=/data/graft_gs_teacher_bundles
export GRAFT_GS_RUN_TRAINING_TESTS=1
export GRAFT_GS_PYTHON=/mnt/sda1/miniforge3/envs/CRAFT/bin/python
export PYTHONHASHSEED=0
```

The two checkpoint variables may be omitted when the released checkpoints already
exist in the default Hugging Face cache. Resolution is CLI override, then the
`GRAFT_GS_*` variable above, then the compatible legacy upstream variable,
then the official identifier. The two repository-root variables bind imports
to the exact upstream project code declared for this server; the same paths are
automatically preferred when present, while explicit environment values retain
precedence.

`validate_server.py` requires `vggt/__init__.py` plus `demo_gradio.py` and
`trellis/__init__.py` plus `app.py` under those roots. Their package/entrypoint
SHA-256 values are recorded in `upstream_repositories` before any model is
loaded, and the resolved roots are propagated to every child process.

Before importing PyTorch, verify that the active interpreter has every one of
the 444 exact versions pinned by the repository. This check is metadata-only;
it never installs or upgrades packages:

```bash
/mnt/sda1/miniforge3/envs/CRAFT/bin/python scripts/validate_environment.py \
  --requirements requirements.txt \
  --output outputs/validation/environment.json
/mnt/sda1/miniforge3/envs/CRAFT/bin/python -m pip check
```

A nonzero result is an environment failure. Synchronize that conda environment
from `requirements.txt` before validation; do not interpret tests from a
different dependency set as repository validation.

`validate_server.py` then performs a subprocess-isolated runtime probe and
records the PyTorch/CUDA version, visible A800 names, compute capabilities,
memory, and BF16 support. The reference path requires the pinned CUDA 11.8
build and at least one scheduler-visible A800. Do not rewrite a
scheduler-provided `CUDA_VISIBLE_DEVICES`: every launcher derives its process count from
`torch.cuda.device_count()` after that mask has been applied.

Build the manifest from the mounted data and retain its SHA-256 digest:

```bash
python scripts/build_meshfleet_manifest.py \
  "$GRAFT_GS_MESHFLEET_ROOT" "$GRAFT_GS_MESHFLEET_MANIFEST"
sha256sum "$GRAFT_GS_MESHFLEET_MANIFEST" | tee outputs/manifest.sha256
```

The builder scans the modality trees once per split, forms candidate IDs from
`latents` and `mesh_normalized`, and admits the intersection containing
`renders`, `latents`, and `mesh_normalized`. Missing optional modalities are
recorded per object; rejected candidates and their missing required modalities
are written to `meshfleet_server.jsonl.rejected.jsonl`. No object ID catalog or
example-object ordering is assumed. Select a complete training fixture only
when a single-object command requires one:

```bash
export GRAFT_GS_OBJECT_ID=$($GRAFT_GS_PYTHON -c '
import json, os
p = os.environ["GRAFT_GS_MESHFLEET_MANIFEST"]
records = [json.loads(x) for x in open(p) if x.strip()]
ids = sorted(r["object_id"] for r in records if r["split"] == "train")
if not ids: raise SystemExit("manifest contains no admitted train object")
print(ids[0])')
```

## Rank ownership and useful-concurrency sweep

Use exactly one process for every device already exposed by the scheduler. Do
not launch multiple ranks per A800 and do not rewrite the scheduler mask:

```bash
export GRAFT_GS_NPROC_PER_NODE=$($GRAFT_GS_PYTHON -c \
  'import torch; print(torch.cuda.device_count())')
test "$GRAFT_GS_NPROC_PER_NODE" -ge 1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ranks=$GRAFT_GS_NPROC_PER_NODE"

$GRAFT_GS_PYTHON -m unittest \
  tests.test_distributed_evidence.AtlasSynchronizationTransportTest.test_local_rank_is_bound_before_checkpoint_allocation \
  tests.test_distributed_evidence.AtlasSynchronizationTransportTest.test_local_rank_rejects_foreign_cuda_allocator_reservation \
  tests.test_distributed_evidence.AtlasSynchronizationTransportTest.test_local_rank_accepts_exclusively_local_cuda_allocator_state \
  -v
```

After confirming that no previous GRAFT-GS launch remains, sweep distinct views
per rank. The global loader admits `views_per_rank * world_size` views, then
shards the CPU sample before its non-blocking CUDA transfer:

```bash
for VIEWS_PER_RANK in 8 12 16; do
  RUN_DIR="outputs/concurrency/${GRAFT_GS_OBJECT_ID}/vpr-${VIEWS_PER_RANK}"
  mkdir -p "$RUN_DIR"
  "$GRAFT_GS_PYTHON" -m torch.distributed.run \
    --standalone --nnodes=1 \
    --nproc-per-node="$GRAFT_GS_NPROC_PER_NODE" \
    scripts/overfit_meshfleet_object.py \
    "$GRAFT_GS_MESHFLEET_ROOT" "$GRAFT_GS_MESHFLEET_MANIFEST" \
    --split train --object-id "$GRAFT_GS_OBJECT_ID" \
    --config configs/graft_gs_a800_native.yaml \
    --vggt-checkpoint "$VGGT_CHECKPOINT" \
    --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
    --views-per-rank "$VIEWS_PER_RANK" \
    --evaluation-views 24 --steps 2 \
    --minimum-relative-improvement -1 \
    --output "$RUN_DIR" 2>&1 | tee "$RUN_DIR/run.log"
done
```

In a second terminal, verify that each PID appears on one GPU only:

```bash
watch -n 2 'nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
  --format=csv,noheader,nounits | sort -k2,2n'
```

Inspect measured per-rank records instead of using occupancy as a proxy:

```bash
$GRAFT_GS_PYTHON - <<'PY'
import glob, json
for path in sorted(glob.glob("outputs/concurrency/*/vpr-*/overfit_metrics.json")):
    report = json.load(open(path))
    print(path)
    for row in report["rank_performance"]:
        print(
            " rank", row["rank"], "views", int(row["local_views"]),
            "views/s", round(row["local_views_per_second"], 3),
            "allocated", round(row["peak_allocated_fraction"], 3),
            "reserved", round(row["peak_reserved_fraction"], 3),
        )
PY
```

Select the largest budget that increases aggregate useful views/s, keeps every
finite-state guard passing, and retains headroom. Initially reject a two-step
Phase-B setting above 0.85 peak reserved fraction; Phase-D/F refinement peaks
can be larger. If 16 views/rank remains below 0.70 and improves views/s, repeat
once with 24. Never reserve dummy memory merely to reach 80 GiB.

For corpus training, use ordinary object-level DDP (omit
`--same-object-view-shards`). Every visible GPU receives a different complete
object, while `--maximum-views` controls its local geometric view budget:

```bash
bash scripts/launch_a800_6gpu.sh \
  "$GRAFT_GS_MESHFLEET_ROOT" B 1000 \
  --manifest "$GRAFT_GS_MESHFLEET_MANIFEST" --split train \
  --maximum-views 24 --dataloader-workers 8 \
  --dataloader-prefetch-factor 4 \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --output outputs/training/phase-b
```

The view budget is checkpoint provenance; changing it requires a new run or
phase initialization, not an exact in-epoch resume.

## High-precision reference suite

The returned server environment had `jupyter_client==7.4.9`, while the exact
repository contract pins `jupyter_client==8.9.1` and `ipykernel==7.3.0`
requires at least 8.9.0. Synchronize the exact pin before model validation:

```bash
/mnt/sda1/miniforge3/envs/CRAFT/bin/python -m pip install \
  --no-deps --upgrade jupyter_client==8.9.1
/mnt/sda1/miniforge3/envs/CRAFT/bin/python scripts/validate_environment.py \
  --requirements requirements.txt
/mnt/sda1/miniforge3/envs/CRAFT/bin/python -m pip check
```

```bash
/mnt/sda1/miniforge3/envs/CRAFT/bin/python scripts/validate_server.py \
  --requirements requirements.txt \
  --dataset-root /mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97 \
  --manifest outputs/validation/meshfleet_server.jsonl \
  --output outputs/validation/reference.json 2>&1 | tee outputs/validation/reference.log
```

This must execute—not skip—the numerical UOT, equivariance, manifold, barrier,
analytical asset, renderer-backward, checkpoint, MeshFleet real-contract, and
checkpoint-backed multiview tests. Any NaN, failed gradient assertion, skipped
test lacking an explicitly unavailable optional backend, or nonzero exit status
is a failure.

The validator injects the audited dataset root and manifest into the test
subprocess. It rejects dataset/backend skips; only the separate visible-rank DDP
launch and a separately configured real-image/checkpoint run may remain skipped
in this single-GPU reference command.

Before the unittest suite, the validator runs
`scripts/validate_external_models.py` twice in isolated processes. The VGGT
pass loads the resolved checkpoint, runs the production adapter on two real
manifest-selected views, verifies finite camera/depth/point outputs and SO(3)
margins, and records peak VRAM. The TRELLIS pass loads its resolved checkpoint,
runs two multi-image sparse-structure posterior draws, forms the Jeffreys
support measure, and records resolution/support/probability bounds and peak
VRAM. Selection is by an explicit object ID when supplied to that script, or a
documented lexicographic smoke-record policy; no canonical ID or first-manifest
assumption is used. The policy evaluates the production task-admission
predicate first and passes the winning ID through the bounded runtime selector,
so only one object is constructed even when the manifest contains the complete
remote corpus.

Manifest reuse is conditional on resolved-root equality, schema
`meshfleet-trellis-object-v2`, the modality-centric intersection policy,
summary/JSONL and split-count equality, valid 64-hex IDs, train/test disjointness,
and the discovered-object digest. Any failed condition invokes the deterministic
builder before importing the dataset. Use
`--rebuild-manifest` to force a fresh full-corpus audit even when those identity
checks pass.

## Reference/CUDA renderer equivalence

Using the TRELLIS mip-splatting `diff_gaussian_rasterization` already built on
the server:

```bash
/mnt/sda1/miniforge3/envs/CRAFT/bin/python -m unittest \
  tests.test_assets_and_vertical_slice.AnalyticalAssetTest.test_cuda_reference_equivalence_small_scene -v \
  2>&1 | tee outputs/validation/renderer_equivalence.log
```

The test now uses off-axis intrinsics and a nonblack background. Required
tolerances are encoded in the test: RGB/alpha `atol=5e-2, rtol=8e-2`, visible
depth `atol=rtol=2e-2`, and mean visible normal cosine above `0.9`. It must also
prove that the loaded extension exposes TRELLIS' `kernel_size` and
`subpixel_offset` ABI.

## TRELLIS latent/decoded-grid contract

The released structure flow samples a 16-cubed latent, but its decoder emits a
64-cubed occupancy field. Validate the decoder-observed coordinate contract
before any overfit or staged run:

```bash
$GRAFT_GS_PYTHON -m unittest -v tests.test_external_adapters \
  2>&1 | tee outputs/validation/trellis_contract_cpu.log

$GRAFT_GS_PYTHON scripts/validate_external_models.py \
  trellis "$GRAFT_GS_MESHFLEET_ROOT" "$GRAFT_GS_MESHFLEET_MANIFEST" \
  --object-id "$GRAFT_GS_TEST_OBJECT_ID" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --trellis-samples 2 --trellis-sampler-steps 2 \
  --output outputs/validation/trellis.json \
  2>&1 | tee outputs/validation/trellis.log
```

The JSON must report `resolution: 64`, finite support values, coordinates
inside `[0,63]`, and no fallback to `max(coordinate)+1`. The flow model's 16 is
retained only as latent-model metadata.

## Exact checkpoint and next-step replay

```bash
GRAFT_GS_RUN_TRAINING_TESTS=1 \
  python -m unittest \
  tests.test_real_multiview.RealMultiviewTest.test_trainer_checkpoint_round_trip -v \
  2>&1 | tee outputs/validation/checkpoint_single_gpu.log
```

The checkpoint must be format 6, restore model/optimizer/counters/objective,
native precision policy,
and reproduce the next Torch random sample exactly.

## Same-object distributed evidence and rank-local RNG

```bash
export GRAFT_GS_NPROC_PER_NODE=$($GRAFT_GS_PYTHON -c 'import torch; print(torch.cuda.device_count())')
test "$GRAFT_GS_NPROC_PER_NODE" -ge 1
$GRAFT_GS_PYTHON -m torch.distributed.run --standalone \
  --nproc-per-node="$GRAFT_GS_NPROC_PER_NODE" \
  scripts/validate_ddp_server.py \
  --requirements requirements.txt \
  --output outputs/validation/ddp_environment.json \
  2>&1 | tee outputs/validation/ddp_visible_rank.log
```

The suite must show identical global evidence/prior inputs while retaining
nonzero rank-local autograd. It also restores each rank's Torch CPU/CUDA,
NumPy, and Python stream and verifies the rank-local streams do not collapse. A
format-6 trainer checkpoint must additionally reject a different resume world
size before mutating model state. Its JSON preflight must contain exactly one
distinct host/local-rank/A800 record per scheduler-visible device; success is
reduced across all ranks, not inferred from rank zero alone. A one-device mask
validates the single-rank fallback but does not constitute multi-rank DDP
evidence; retain a `multi_rank=true` report when collective equivalence is the
claim under test.

The suite also executes a real persistent-atlas collective. Torch 2.4 NCCL
cannot broadcast `torch.int16` directly, so every discrete field must travel
through an independent contiguous int64 buffer and restore its exact original
dtype/value; the test includes noncontiguous and greater-than-2^53 identities.
Every continuous field must be bitwise source-identical in the forward pass.
The source-owned autograd broadcast must reduce all downstream rank losses and
the global evidence all-gather must return finite nonzero gradients to every
rank's local evidence, including when non-source ranks deliberately choose an
equivalent pi-rotated tangent gauge. Gauge-independent fields retain explicit
replica checks; raw PCA frame/curvature coordinates are not equality-tested.
Any `Short` collective, metadata mismatch, nonfinite field, or zero local
evidence gradient fails this gate. The CPU portion also requires a finite zero
gauge derivative at a repeated spectrum and finite nonzero frame derivatives
for a separated spectrum.

Before repeating checkpoint-backed overfit, run the focused spectral and
finite-state regressions with the exact pinned interpreter:

```bash
$GRAFT_GS_PYTHON -m unittest -v \
  tests.test_assets_and_vertical_slice.AnalyticalAssetTest.test_isotropic_chart_metric_has_finite_basis_free_backward \
  tests.test_assets_and_vertical_slice.AnalyticalAssetTest.test_flat_chart_analytical_readout_backward_is_finite \
  tests.test_geometry_invariants.TopologyAndManifoldTest.test_spd_spectral_box_is_bounded_and_repeated_spectrum_safe \
  tests.test_distributed_evidence.AtlasSynchronizationTransportTest.test_nonfinite_gradient_guard_fails_before_optimizer_step \
  tests.test_atlas_mapping.PersistentAtlasTest.test_atlas_rejects_nonfinite_mass_with_specific_diagnostic \
  2>&1 | tee outputs/validation/phase_b_finite_gradient.log
```

All five must pass. The same numerical cases are included in
`validate_ddp_server.py` on every visible rank. A non-finite failure is not
recoverable by increasing clipping or loosening a tolerance; retain the named
rank/tensor diagnostic and restart from the last checkpoint created before the
failed optimizer update.

## Offline teacher bundle refinement

```bash
python scripts/refine_teacher_bundle.py \
  "$GRAFT_GS_MESHFLEET_ROOT" "$GRAFT_GS_MESHFLEET_MANIFEST" \
  outputs/phase_d/final.pt \
  "$GRAFT_GS_OBJECT_ID" \
  "$GRAFT_GS_TEACHER_BUNDLES" --split test \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  2>&1 | tee outputs/validation/teacher_bundle.log
```

The run must retain the initial persistent complex, finish with positive hard
feasibility margins, emit a confidence in `[0,1]`, and independently reload
the atlas-derived PLY/GLB and typed `.teacher.pt` bundle. Corpus-scale Phase-C
training requires generating the same schema for every admitted train object.

## One-object overfit

```bash
$GRAFT_GS_PYTHON -m torch.distributed.run --standalone \
  --nproc-per-node="$GRAFT_GS_NPROC_PER_NODE" scripts/overfit_meshfleet_object.py \
  "$GRAFT_GS_MESHFLEET_ROOT" "$GRAFT_GS_MESHFLEET_MANIFEST" \
  --object-id "$GRAFT_GS_OBJECT_ID" \
  --config configs/graft_gs_a800_native.yaml \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --steps 1000 --output outputs/overfit_fixture \
  2>&1 | tee outputs/overfit_fixture/run.log
```

For the bounded two-step recovery smoke, keep artifacts and the tee in the
same directory (the `--output` argument is required; assigning `SMOKE_DIR`
alone does not change the script default):

```bash
SMOKE_DIR="outputs/overfit_smoke/${GRAFT_GS_TRAIN_OBJECT_ID}_strict_restore"
mkdir -p "$SMOKE_DIR"
$GRAFT_GS_PYTHON -m torch.distributed.run --standalone --nnodes=1 \
  --nproc-per-node="$GRAFT_GS_NPROC_PER_NODE" \
  scripts/overfit_meshfleet_object.py \
  "$GRAFT_GS_MESHFLEET_ROOT" "$GRAFT_GS_MESHFLEET_MANIFEST" \
  --split train --object-id "$GRAFT_GS_TRAIN_OBJECT_ID" \
  --config configs/graft_gs_a800_native.yaml \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --steps 2 --views-per-rank 12 --evaluation-views 24 \
  --minimum-relative-improvement -1 --output "$SMOKE_DIR" \
  2>&1 | tee "$SMOKE_DIR/run.log"
```

Delete nothing and do not reuse `step-00000001.pt` from the pre-repair failed
smoke: its optimizer step was not protected by the finite-state gate.

Required artifacts: periodic and final checkpoints, `metrics.jsonl`, decreasing
overfit objective, input-view renders, deterministic PLY/GLB, reload metrics,
and no activation of inadmissible hard raw-mesh topology loss. Topology
expectations are taken from each record's provenance-aware contract, never
from the selected object's ID.

Before that smoke, run the strict restoration, implicit-UOT, and high-precision
gradient-norm gate against the deployed source:

```bash
mkdir -p outputs/validation
"$GRAFT_GS_PYTHON" -m unittest \
  tests.test_geometry_invariants.TopologyAndManifoldTest.test_metric_minimal_restoration_enters_strict_feasible_set \
  tests.test_atlas_mapping.ImplicitSinkhornTest.test_solver_rejects_invalid_measure_and_nonconvergence \
  tests.test_atlas_mapping.ImplicitSinkhornTest.test_adjoint_nonconvergence_is_not_silently_accepted \
  tests.test_atlas_mapping.ImplicitSinkhornTest.test_sparse_all_edges_matches_dense_fixed_point_and_has_gradients \
  tests.test_atlas_mapping.ImplicitSinkhornTest.test_implicit_backward_matches_finite_difference \
  tests.test_distributed_evidence.AtlasSynchronizationTransportTest.test_high_precision_gradient_norm_does_not_overflow_before_clipping \
  -v 2>&1 | tee outputs/validation/strict_numerics.log
```

Retain `strict_numerics.log`, `run.log`, `metrics.json`, the final checkpoint,
rank performance records, and every final feasibility/restoration field. A
`find_unused_parameters=True` warning, evaluation on a nonzero rank, an
unconverged UOT diagnostic, or a non-positive recertified margin means the
deployed source/config is stale or the run failed.

## Full staged training and exact phase boundaries

```bash
bash scripts/launch_a800_6gpu.sh "$GRAFT_GS_MESHFLEET_ROOT" A 20000 --manifest "$GRAFT_GS_MESHFLEET_MANIFEST" --split train --output outputs/phase_a
bash scripts/launch_a800_6gpu.sh "$GRAFT_GS_MESHFLEET_ROOT" B 50000 --manifest "$GRAFT_GS_MESHFLEET_MANIFEST" --split train --trellis-checkpoint "$TRELLIS_CHECKPOINT" --initialize-from outputs/phase_a/final.pt --output outputs/phase_b
bash scripts/launch_a800_6gpu.sh "$GRAFT_GS_MESHFLEET_ROOT" C 50000 --manifest "$GRAFT_GS_MESHFLEET_MANIFEST" --split train --trellis-checkpoint "$TRELLIS_CHECKPOINT" --teacher-bundle-root "$GRAFT_GS_TEACHER_BUNDLES" --initialize-from outputs/phase_b/final.pt --output outputs/phase_c
bash scripts/launch_a800_6gpu.sh "$GRAFT_GS_MESHFLEET_ROOT" D 100000 --manifest "$GRAFT_GS_MESHFLEET_MANIFEST" --split train --trellis-checkpoint "$TRELLIS_CHECKPOINT" --initialize-from outputs/phase_c/final.pt --output outputs/phase_d
bash scripts/launch_a800_6gpu.sh "$GRAFT_GS_MESHFLEET_ROOT" E 50000 --manifest "$GRAFT_GS_MESHFLEET_MANIFEST" --split train --trellis-checkpoint "$TRELLIS_CHECKPOINT" --teacher outputs/phase_d/final.pt --initialize-from outputs/phase_d/final.pt --output outputs/phase_e
bash scripts/launch_a800_6gpu.sh "$GRAFT_GS_MESHFLEET_ROOT" F 30000 --manifest "$GRAFT_GS_MESHFLEET_MANIFEST" --split train --trellis-checkpoint "$TRELLIS_CHECKPOINT" --initialize-from outputs/phase_e/final.pt --output outputs/phase_f
```

Profiler traces must demonstrate: A has no atlas scene; B has no vector-field
integration; C has no Gaussian/mesh/readout/render; D-F execute the full path.

## Real multiview inference, assets, time, and memory

```bash
python scripts/infer_multiview.py \
  "$GRAFT_GS_REAL_IMAGE_DIR" outputs/real_multiview \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --graft-checkpoint "$GRAFT_GS_CHECKPOINT" --render-input-views \
  2>&1 | tee outputs/real_multiview/run.log

python scripts/infer_meshfleet.py \
  "$GRAFT_GS_MESHFLEET_ROOT" "$GRAFT_GS_MESHFLEET_MANIFEST" \
  "$GRAFT_GS_CHECKPOINT" outputs/meshfleet_inference \
  --object-id "$GRAFT_GS_OBJECT_ID" \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --quantization-query-error "$MEASURED_QK_ERROR" \
  --vector-field-lipschitz-bound "$MEASURED_FIELD_LIPSCHITZ_BOUND" \
  --profile-trace outputs/meshfleet_inference/trace.json \
  2>&1 | tee outputs/meshfleet_inference/run.log
```

Retain reported wall time, peak allocated CUDA bytes, active charts, UOT edge
count/iterations/residual, prior support, selected Betti tuple, every feasibility
margin, Gaussian/face counts, renders, PLY, GLB, and independent reload reports.
The two quantization arguments must come from the same pinned quantized
checkpoint/server precision path. Retain all emitted inequality terms; do not
interpret `certified=true` as unconditional beyond the recorded Lipschitz,
support-stratum, and barrier assumptions.

Corpus evaluation must use the complete admitted test split rather than a
sample-ID allowlist:

```bash
$GRAFT_GS_PYTHON -m torch.distributed.run --standalone \
  --nproc-per-node="$GRAFT_GS_NPROC_PER_NODE" \
  scripts/evaluate_meshfleet.py \
  "$GRAFT_GS_MESHFLEET_ROOT" "$GRAFT_GS_MESHFLEET_MANIFEST" \
  "$GRAFT_GS_CHECKPOINT" outputs/meshfleet_evaluation \
  --splits test --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  2>&1 | tee outputs/meshfleet_evaluation/run.log
```

## Ablations

```bash
python scripts/run_ablations.py \
  "$GRAFT_GS_REAL_IMAGE_DIR" --output outputs/ablations.json \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --graft-checkpoint "$GRAFT_GS_CHECKPOINT" \
  --config configs/graft_gs_a800_native.yaml \
  2>&1 | tee outputs/ablations.log
```

At minimum compare full, no hidden prior, no transport feature fixed point, no
flow, and reduced topology proposal variants. Add explicit toggles for new
OT/uncertainty attention bias, overlap/multilevel loss, adaptive refinement,
tile-opacity bound, and quantization only after the reference run passes.

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

## Offload-v4 failure regression gate (2026-07-25)

The supplied vpr-8, vpr-12, and vpr-24 logs predate the zero-reliability,
offline-Hub, sparse-barrier, and phase-boundary repairs. Run this focused gate
before another long sweep:

```bash
set -euo pipefail
mkdir -p outputs/validation
"$GRAFT_GS_PYTHON" -m unittest -v \
  tests.test_atlas_mapping.PersistentAtlasTest.test_attention_uncertainty_has_finite_zero_reliability_gradient \
  tests.test_atlas_mapping.ImplicitSinkhornTest.test_float32_storage_underflow_uses_log_domain_float64_reference \
  tests.test_atlas_mapping.ImplicitSinkhornTest.test_adjoint_rejects_nonfinite_positive_mass_cotangent \
  tests.test_atlas_mapping.ImplicitSinkhornTest.test_sparse_all_edges_matches_dense_fixed_point_and_has_gradients \
  tests.test_atlas_mapping.ImplicitSinkhornTest.test_implicit_backward_matches_finite_difference \
  tests.test_geometry_invariants.TopologyAndManifoldTest.test_metric_minimal_restoration_enters_strict_feasible_set \
  tests.test_external_adapters.TrellisAdapterBoundaryTest.test_dinov2_torch_hub_load_is_strictly_redirected_to_cache \
  tests.test_external_adapters.TrellisAdapterBoundaryTest.test_missing_dinov2_cache_fails_without_calling_network_loader \
  2>&1 | tee outputs/validation/offload_v4_numerics.log

"$GRAFT_GS_PYTHON" scripts/validate_external_models.py \
  trellis \
  "$GRAFT_GS_MESHFLEET_ROOT" \
  "$GRAFT_GS_MESHFLEET_MANIFEST" \
  --object-id "$GRAFT_GS_TEST_OBJECT_ID" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --trellis-samples 2 \
  --trellis-sampler-steps 2 \
  --output outputs/validation/trellis_offline_cache.json \
  2>&1 | tee outputs/validation/trellis_offline_cache.log
```

The first command must pass all eight tests. The sparse-barrier regression
constructs the former dense Jacobian only inside a small test oracle and
compares every row, the exact Gram product, its certified infinity-norm upper
bound, restoration feasibility, and backward finiteness. Production
`barrier.py` must contain no call to `torch.autograd.functional.jacobian`.
The second command must complete with networking disabled; absence of
`facebookresearch_dinov2_main/hubconf.py` under `torch.hub.get_dir()` is a
provisioning failure, not permission to download mutable source.

Then create a fresh schema-v5 sweep root. Do not reuse any pre-repair
`overfit_metrics.json`:

```bash
set -euo pipefail
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
SWEEP_ROOT="outputs/concurrency/${GRAFT_GS_TRAIN_OBJECT_ID}/sparse-barrier-${RUN_TAG}"

"$GRAFT_GS_PYTHON" scripts/sweep_a800_view_budget.py \
  "$GRAFT_GS_MESHFLEET_ROOT" \
  "$GRAFT_GS_MESHFLEET_MANIFEST" \
  --object-id "$GRAFT_GS_TRAIN_OBJECT_ID" \
  --config configs/graft_gs_a800_native.yaml \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --views-per-rank 8 12 16 24 32 48 64 \
  --evaluation-views 24 \
  --steps 3 \
  --minimum-relative-improvement -1 \
  --maximum-reserved-fraction 0.85 \
  --maximum-allocated-fraction 0.90 \
  --minimum-driver-free-fraction 0.05 \
  --minimum-initial-driver-free-fraction 0.95 \
  --maximum-storage-relative-l1-error 1e-6 \
  --maximum-zero-marginal-mass-fraction 1e-12 \
  --throughput-fraction 0.97 \
  --output "$SWEEP_ROOT" \
  2>&1 | tee "$SWEEP_ROOT.log"
```

Every admitted report must contain
`"evaluation_execution_stage": "atlas_autoencoding"`. Selection schema v5
rejects a missing value or `"full"`. This Phase-B sweep does not validate
continuous flow; retain a separate Phase-C/D profile showing positive final
margins and linear barrier memory before claiming full-flow scalability.

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

Before a full model sweep, the lightweight progress/NCCL path can be checked
without loading VGGT or TRELLIS:

```bash
CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc-per-node=2 \
  scripts/validate_progress_ddp.py \
  --iterations 4 \
  --output outputs/validation/progress_ddp_2gpu.json
```

This is a communication and observability smoke only; it does not validate
Phase-B memory, throughput, gradients, or accuracy.

For ordinary object-level DDP, tune the per-rank object batch independently
for every training phase after the view budget is fixed. The tuner takes the
physical indices/UUIDs explicitly, launches every candidate in a fresh process
group, rejects allocator/driver headroom failures, and optionally launches the
selected full run:

```bash
"$GRAFT_GS_PYTHON" scripts/autotune_object_batch.py \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --candidates 1 2 4 8 \
  --output outputs/concurrency/object-batch-phase-d \
  --maximum-allocated-fraction 0.85 \
  --maximum-reserved-fraction 0.88 \
  --minimum-driver-free-fraction 0.08 \
  --probe-timeout-seconds 1800 \
  --probe-no-progress-timeout-seconds 900 \
  --probe-warmup-steps 1 \
  --probe-measurement-steps 2 \
  --launch -- \
  "$GRAFT_GS_MESHFLEET_ROOT" --phase D --steps 100000 \
  --manifest "$GRAFT_GS_MESHFLEET_MANIFEST" --split train \
  --global-object-batch 32 \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --initialize-from outputs/phase_c/final.pt --output outputs/phase_d
```

Do not reuse a selected batch across phases without measuring it: Phase E
retains teacher/student state and Phase F replays a hardening forward. Do not
enable multi-object batching with `--same-object-view-shards`; that mode
constructs one global nonlinear UOT problem by definition. If a fixed global
optimizer batch is part of the experiment, pass `--global-object-batch`; the
trainer requires exact divisibility by physical objects times world size and
adjusts accumulation without changing the optimizer batch.

Object-batch probes deliberately replace the accumulation policy with one
microbatch per optimizer step. A warmup materializes buckets and AdamW state;
later optimizer steps provide steady-state throughput, while memory admission
uses all steps. This avoids sampling solely to fill the production accumulated
batch. The selected launch uses the original argument list, so
`--global-object-batch 32` remains exactly 32. On the first OOM the tuner
terminates the entire torchrun process group and skips larger physical batches
by default; `--continue-after-oom` is an explicit diagnostic escape hatch.

`--probe-timeout-seconds` is now only the bootstrap bound before any semantic
worker record. There is no total candidate wall deadline. Every later stage has
a no-progress budget, and meaningful stage/draw/layer/backward advancement
resets it. After three completed observations, the active threshold is the
larger of the configured stage minimum and empirical Q99 + 6×MAD. Heartbeats
never reset it. Before terminating a frozen stage the tuner writes a rank-state
table to `GRAFT_GS_AUTOTUNE_PROGRESS_TIMEOUT`.

Repeated upstream `Sampling: 12/12` bars count TRELLIS posterior draws (eight
per uncached object under the server configuration), not training iterations
and not an OOM retry. Exact conditioning/seed/checkpoint hits are retained in a
bounded atomic cache; probes use candidate-local scope by default so timing
remains comparable. Use `GRAFT_GS_PROGRESS` together
with the structured `GRAFT_GS_AUTOTUNE_CANDIDATE_START`, `_END`, and
`GRAFT_GS_AUTOTUNE_PROBE_CONTROL` records to distinguish sampler work from a
candidate transition or forced termination.

Repeat this exact procedure in every deployment pool. In particular, an A6000
selection is not admissible evidence for an A100 run, and A100 capacity must be
recorded because 40-GiB and 80-GiB variants require different candidates.

After selection, require a fresh 200-optimizer-step Phase-B soak on that exact
pool before starting 50,000 steps:

```bash
TUNING_ROOT=outputs/concurrency/object-batch-phase-b
SELECTED_BATCH=$(
  "$GRAFT_GS_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_object_batch_size"])' \
  "$TUNING_ROOT/selection.json"
)
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
SOAK_ROOT="outputs/validation/phase-b-200-${RUN_TAG}"

"$GRAFT_GS_PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="$GRAFT_GS_NPROC_PER_NODE" \
  scripts/train_a800.py \
  "$GRAFT_GS_MESHFLEET_ROOT" --phase B --steps 200 \
  --manifest "$GRAFT_GS_MESHFLEET_MANIFEST" --split train \
  --global-object-batch 32 --object-batch-size "$SELECTED_BATCH" \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --trellis-cache-directory "$TUNING_ROOT/production_trellis_exact_cache" \
  --initialize-from outputs/phase_a/final.pt --output "$SOAK_ROOT" \
  2>&1 | tee "$SOAK_ROOT.log"

test -f "$SOAK_ROOT/final.pt"
test "$(tail -n 1 "$SOAK_ROOT/metrics.jsonl" | \
  "$GRAFT_GS_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["step"])')" \
  -eq 200
! grep -E "GRAFT_GS_NONFINITE|GRAFT_GS_TRAINING_PROGRESS_TIMEOUT|OutOfMemoryError|Watchdog caught collective operation timeout" \
  "$SOAK_ROOT.log"
```

Retain the selection, initial CUDA inventory, every candidate log/probe, the
soak log, metrics, precision policy, dataset coverage, and final checkpoint.
Repeat from a fresh tuning/soak root on four A6000s and on four production
A100/A800s; do not transfer the selected physical batch between pools.

Capture a short per-rank PyTorch profiler trace with the already selected batch
and the same Phase-B workload. This is a separate fresh output because it adds
intentional profiling overhead:

```bash
PROFILE_TAG=$(date -u +%Y%m%dT%H%M%SZ)
PROFILE_ROOT="outputs/validation/phase-b-profile-${PROFILE_TAG}"

"$GRAFT_GS_PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="$GRAFT_GS_NPROC_PER_NODE" \
  scripts/train_a800.py \
  "$GRAFT_GS_MESHFLEET_ROOT" --phase B --steps 6 \
  --manifest "$GRAFT_GS_MESHFLEET_MANIFEST" --split train \
  --global-object-batch 32 --object-batch-size "$SELECTED_BATCH" \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --trellis-cache-directory "$TUNING_ROOT/production_trellis_exact_cache" \
  --initialize-from outputs/phase_a/final.pt --output "$PROFILE_ROOT" \
  --profile --profile-output "$PROFILE_ROOT/profiler" \
  2>&1 | tee "$PROFILE_ROOT.log"
```

The default bounded schedule is wait 1, warmup 1, active 3, repeat 1. Retain
all four trace files and compare rank-local data wait, frozen sampling,
forward/recompute, backward/NCCL overlap, optimizer, and allocator peaks before
making any kernel or DDP-bucket claim.

Use the corresponding fresh-process probe for full-corpus testing:

```bash
"$GRAFT_GS_PYTHON" scripts/autotune_evaluation_batch.py \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --candidates 1 2 4 8 \
  --output outputs/concurrency/evaluation-batch \
  --launch -- \
  "$GRAFT_GS_MESHFLEET_ROOT" "$GRAFT_GS_MESHFLEET_MANIFEST" \
  outputs/phase_f/final.pt outputs/evaluation --splits test
```

Both selection JSON files are execution evidence and should be retained with
the checkpoint/evaluation report.

Run the exact regressions for the two failures exposed by the first
16/24/32/48/64 sweep before spending time on TRELLIS sampling:

```bash
mkdir -p outputs/validation
"$GRAFT_GS_PYTHON" -m unittest -v \
  tests.test_atlas_mapping.ImplicitSinkhornTest.test_float32_storage_underflow_uses_log_domain_float64_reference \
  tests.test_atlas_mapping.ImplicitSinkhornTest.test_sparse_all_edges_matches_dense_fixed_point_and_has_gradients \
  tests.test_atlas_mapping.ImplicitSinkhornTest.test_implicit_backward_matches_finite_difference \
  tests.test_geometry_invariants.TopologyAndManifoldTest.test_diffuse_occupancy_retains_all_support_filtration_stratum \
  tests.test_geometry_invariants.TopologyAndManifoldTest.test_large_persistence_matching_is_linear_memory_symmetric_and_differentiable \
  tests.test_meshfleet_contract.MeshFleetAuditTest.test_nonmanifold_mesh_still_derives_depth_and_normals \
  tests.test_assets_and_vertical_slice.AnalyticalAssetTest.test_cuda_view_checkpoint_matches_forward_and_gradients \
  tests.test_external_adapters.TrellisAdapterBoundaryTest.test_hidden_prior_view_cap_is_deterministic_and_endpoint_covering \
  tests.test_external_adapters.TrellisAdapterBoundaryTest.test_frozen_pipeline_offload_releases_live_cuda_weight_allocation \
  tests.test_external_adapters.TrellisAdapterBoundaryTest.test_frozen_sampler_releases_only_inactive_cuda_allocator_blocks \
  tests.test_external_adapters.TrellisAdapterBoundaryTest.test_cache_release_rejects_a_live_allocation_lifetime_change \
  2>&1 | tee outputs/validation/concurrency_numerics.log
```

All eleven must pass. The underflow regression is deliberately disconnected and
contains a positive exact UOT component near `exp(-196)`: FP32 may store that
entry as zero, but the FP64/log-domain fixed point and implicit conditional
probabilities must remain finite. It also bounds the discarded mass and total
relative L1 storage error against the FP64 plan; the number of underflowed
edges is diagnostic only. The two allocator tests establish that the frozen
TRELLIS lifetime boundary returns inactive cached blocks without changing one
live allocated byte. The source-offload test additionally requires the frozen
checkpoint's live allocation to leave CUDA, and the conditioning test proves
the prior-only cap is deterministic and endpoint-covering. The topology
regression requires a valid,
orientable all-support filtration stratum when every ordinary fixed threshold
would remove the overlap triangles. The persistence regression patches
`torch.cdist` to fail and then exercises a 600-by-600 diagram through the
linear-memory sliced path, including symmetry, identity, and finite gradients.
The MeshFleet raster test compares one-view chunks against the former two-view
batch for depth, normals, visibility, and normal validity. This is the
numerical boundary for moving immutable mesh targets before the trainable graph
and bounding nvdiffrast workspace. The CUDA renderer test independently
requires per-view activation checkpointing to preserve all four outputs and
analytical-Gaussian gradients relative to the uncheckpointed operator.

After confirming that no previous GRAFT-GS launch remains, sweep distinct views
per rank. The global loader admits `views_per_rank * world_size` views, then
shards the CPU sample before its non-blocking CUDA transfer:

```bash
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
SWEEP_ROOT="outputs/concurrency/${GRAFT_GS_TRAIN_OBJECT_ID}/checkpoint-${RUN_TAG}"

"$GRAFT_GS_PYTHON" scripts/sweep_a800_view_budget.py \
  "$GRAFT_GS_MESHFLEET_ROOT" \
  "$GRAFT_GS_MESHFLEET_MANIFEST" \
  --object-id "$GRAFT_GS_TRAIN_OBJECT_ID" \
  --config configs/graft_gs_a800_native.yaml \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --views-per-rank 8 12 16 24 32 48 64 \
  --evaluation-views 24 \
  --steps 3 \
  --minimum-relative-improvement -1 \
  --maximum-reserved-fraction 0.85 \
  --maximum-allocated-fraction 0.90 \
  --minimum-driver-free-fraction 0.05 \
  --minimum-initial-driver-free-fraction 0.95 \
  --maximum-storage-relative-l1-error 1e-6 \
  --maximum-zero-marginal-mass-fraction 1e-12 \
  --throughput-fraction 0.97 \
  --output "$SWEEP_ROOT" \
  2>&1 | tee "$SWEEP_ROOT.log"
```

The driver uses the pinned interpreter without shell interpolation, requires a
fresh root, writes `initial_cuda_memory.json`, rejects a preoccupied visible
GPU before loading checkpoints, and retains every exact child command/log.
It evaluates every requested candidate by default because atlas/topology/
visibility memory is object-dependent and measured peaks are non-monotone.
`--stop-after-oom` is an explicit time-saving diagnostic mode, not the
scientific selection default.

In a second terminal, verify that each PID appears on one GPU only:

```bash
watch -n 2 'nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
  --format=csv,noheader,nounits | sort -k2,2n'
```

Select from the measured per-rank records instead of using occupancy as a
proxy. The sweep driver already runs the selector. It rejects non-finite loss,
unconverged UOT, non-positive hard feasibility margins, incomplete rank
telemetry, stale reports without the CUDA view-checkpoint certificate, more
than `1e-6` FP32 plan relative-L1 error, more than `1e-12` discarded
zero-marginal mass, a live allocated/active peak above 90%, a post-step
allocator reservation above 85%, or less than 5% driver-visible headroom.
Historical frozen-prior peak is serialized separately from the differentiable
graph peak. The source rank must explicitly certify checkpoint host offload,
inactive-cache release, and positive available/selected conditioning counts.
Every rank must provide per-stage PyTorch and non-allocator-visible CUDA
telemetry. When none pass,
`selection.json` is still written with every rejection reason before the
driver fails:

```bash
export GRAFT_GS_SELECTED_VIEWS_PER_RANK=$(
  "$GRAFT_GS_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["recommended_views_per_rank"])' \
  "$SWEEP_ROOT/selection.json"
)
echo "selected useful views per rank: $GRAFT_GS_SELECTED_VIEWS_PER_RANK"
```

The default 0.97 throughput fraction chooses the largest view count whose
aggregate useful throughput is within 3% of the fastest admissible run. Never
reserve dummy memory merely to reach 80 GiB. Repeat the sweep in Phase D before
full training because flow, rendering, and refined atlases have higher peaks.
Every admitted report must record `internal_solve_dtype=float64`, the minimum
log-plan value, COO cardinalities, exact storage-zero counts, their FP64 mass
fractions, total storage relative-L1 error, peak allocated/active/reserved
bytes, ending allocated/reserved bytes, driver-free bytes, source-rank TRELLIS
offload/cache-release/conditioning certificates, and CUDA stage memory. Retain
every `GRAFT_GS_CUDA_FAILURE_DIAGNOSTICS=` JSON line from a failed candidate.

For corpus training, use ordinary object-level DDP (omit
`--same-object-view-shards`). Every visible GPU receives a different complete
object, while `--maximum-views` controls its local geometric view budget:

```bash
bash scripts/launch_a800_6gpu.sh \
  "$GRAFT_GS_MESHFLEET_ROOT" B 1000 \
  --manifest "$GRAFT_GS_MESHFLEET_MANIFEST" --split train \
  --maximum-views "$GRAFT_GS_SELECTED_VIEWS_PER_RANK" \
  --minimum-global-object-batch 6 \
  --dataloader-workers 8 \
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
`--output "$SMOKE_DIR"` is mandatory. Setting `SMOKE_DIR` and teeing the log
does not change the script's artifact destination.

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

## Fast posterior-measure backward gate (2026-07-25)

The previous full sweep already established that 24 views is the first
numerical failure and that the failure is not OOM. Do not immediately repeat
the multi-hour seven-candidate sweep. Validate the repaired primitives first:

```bash
mkdir -p outputs/validation
"$GRAFT_GS_PYTHON" -m unittest -v \
  tests.test_atlas_mapping.PersistentAtlasTest.test_zero_transport_rows_use_finite_atlas_posterior_moments \
  tests.test_atlas_mapping.PersistentAtlasTest.test_named_gradient_boundary_rejects_without_replacement \
  tests.test_atlas_mapping.PersistentAtlasTest.test_transport_cost_and_uncertainty_reach_attention_adjacency \
  tests.test_atlas_mapping.PersistentAtlasTest.test_attention_uncertainty_has_finite_zero_reliability_gradient \
  tests.test_geometry_invariants.GaugeEquivarianceTest.test_coincident_connection_edges_have_finite_center_gradient \
  tests.test_geometry_invariants.TopologyAndManifoldTest.test_exact_persistence_identity_has_finite_zero_gradient \
  tests.test_assets_and_vertical_slice.AnalyticalAssetTest.test_exact_surface_chamfer_match_has_finite_zero_gradient \
  tests.test_assets_and_vertical_slice.AnalyticalAssetTest.test_flat_chart_analytical_readout_backward_is_finite \
  tests.test_atlas_mapping.ImplicitSinkhornTest.test_float32_storage_underflow_uses_log_domain_float64_reference \
  tests.test_atlas_mapping.ImplicitSinkhornTest.test_implicit_backward_matches_finite_difference \
  2>&1 | tee outputs/validation/posterior_measure_numerics.log
```

All ten tests must pass. The supplied A800 run passed all ten in 2.556 seconds,
and the subsequent 24-view gate completed an optimizer step, final checkpoint,
PLY, and GLB. The 32-view gate then localized its failure to the full
`[112,3,3]` chart-metric cotangent. Do not repeat the passed 24-view run.

Two subsequent 32-view runs narrowed the defect further. Replacing the backend
factorization backward with the exact inverse pullback did not close it:
`state_initialization.riemannian_metric` still received 1,008 non-finite
values, most recently with maximum finite magnitude `5.650222e-08`.
One identified instability was the unbounded inverse-before-box composition;
neither that path nor the reported failure was A800 memory or external
checkpoint execution. Production now
uses a fused bounded rational precision-to-covariance map in state
initialization and readout, plus a shared exact joint evidence
inverse/log-determinant primitive. Validate the complete closure:

```bash
set -euo pipefail
mkdir -p outputs/validation
"$GRAFT_GS_PYTHON" -m unittest -v \
  tests.test_geometry_invariants.TopologyAndManifoldTest.test_spd_cholesky_inverse_contractions_are_exact_and_finite \
  tests.test_geometry_invariants.TopologyAndManifoldTest.test_spd_inverse_float32_112_chart_cotangent_is_finite \
  tests.test_geometry_invariants.TopologyAndManifoldTest.test_spd_inverse_analytical_pullback_passes_gradcheck \
  tests.test_geometry_invariants.TopologyAndManifoldTest.test_bounded_precision_covariance_closes_112_chart_backward \
  tests.test_geometry_invariants.TopologyAndManifoldTest.test_bounded_precision_covariance_is_so3_covariant \
  tests.test_geometry_invariants.TopologyAndManifoldTest.test_sparse_gram_bound_zero_padding_has_finite_metric_gradient \
  tests.test_geometry_invariants.TopologyAndManifoldTest.test_metric_minimal_restoration_enters_strict_feasible_set \
  tests.test_assets_and_vertical_slice.AnalyticalAssetTest.test_flat_chart_analytical_readout_backward_is_finite \
  tests.test_assets_and_vertical_slice.AnalyticalAssetTest.test_anisotropic_metric_readout_backward_is_finite \
  2>&1 | tee outputs/validation/bounded_spd_numerics.log
```

All nine tests must pass. The supplied v1 rational-map gate subsequently
proved the bounded covariance microproblem finite but exposed the remaining
restoration branch: fixed-width sparse constraint rows contain exact
zero-covector padding, whose literal dual-norm square root generated
`0*infinity` in the metric backward. The two added tests cover the padding and
the complete restoration metric gradient. Then rerun only the unresolved
32-view boundary.
The executable itself performs the same 112-chart CUDA stress before resolving
or loading VGGT/TRELLIS, including the padded sparse dual norm; require its
versioned pass marker:

```bash
set -euo pipefail
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
TOTAL_VIEWS=32
RUN_DIR="outputs/spd_metric_gate/${GRAFT_GS_TRAIN_OBJECT_ID}/${RUN_TAG}/views-${TOTAL_VIEWS}"
mkdir -p "$RUN_DIR"
"$GRAFT_GS_PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 \
  --nproc-per-node="$GRAFT_GS_NPROC_PER_NODE" \
  scripts/overfit_meshfleet_object.py \
  "$GRAFT_GS_MESHFLEET_ROOT" "$GRAFT_GS_MESHFLEET_MANIFEST" \
  --split train --object-id "$GRAFT_GS_TRAIN_OBJECT_ID" \
  --config configs/graft_gs_a800_native.yaml \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --maximum-views "$TOTAL_VIEWS" \
  --evaluation-views 2 \
  --steps 1 \
  --minimum-relative-improvement -1 \
  --output "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/run.log"

grep -F "GRAFT_GS_NUMERICAL_PREFLIGHT=phase-b-rational-spd-zero-dual-v2:passed" \
  "$RUN_DIR/run.log"
test -f "$RUN_DIR/final.pt"
test -f "$RUN_DIR/meshfleet_overfit.ply"
test -f "$RUN_DIR/meshfleet_overfit.glb"
```

The supplied v2 A800 run passed this gate at 32 views: the preflight marker,
finite backward/optimizer step, checkpoint, atlas-autoencoding evaluation,
PLY, GLB, converged FP64 UOT, and positive feasibility margins were all
present. Do not rerun this expensive one-step gate unless the mapped,
restoration, renderer, or trainer numerical path changes. Proceed to the
phase-aware production smoke/exact-resume checks below and then staged
training.

`--maximum-views` fixes the global same-object evidence count even when
`CUDA_VISIBLE_DEVICES` exposes a dynamic number of ranks; `--views-per-rank`
would multiply the numerical problem by `WORLD_SIZE` and would not reproduce
the one-A800 boundary. A pass requires the versioned preflight marker, one
finite optimizer step, a committed checkpoint, finite FP64/log-UOT
diagnostics, positive feasibility margins,
`evaluation_execution_stage=atlas_autoencoding`, and silence from every named
`chart_writer.*`, `transport_attention.*`, `state_initialization.*`, and
`analytical_readout.*` cotangent boundary.

After the 32-view repair gate passes, start production with the configured 24-view object-level
Phase-B path. The previously selected 16 views/rank remains the conservative
fallback if training must begin before a fresh performance sweep; it is not a
reason to reduce precision or losses. Run the full schema-v5 sweep later only
to justify increasing concurrency, not to re-establish the corrected
mathematics.

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

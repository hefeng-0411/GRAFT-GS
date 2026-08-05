# GRAFT-GS research implementation

This package implements a static-3D Gauge-Riemannian Atlas Flow Transformer
beside the unmodified `vggt/` and `TRELLIS/` baseline trees.

## Server target

- Linux, one DDP process per explicitly selected NVIDIA Ampere GPU
- test execution profile: four RTX A6000 48 GB GPUs
- production training profile: physical GPUs 4 and 5, two A100 80 GB devices
  (A800 remains supported)
- PyTorch 2.4 or newer with CUDA/NCCL
- native BF16 VGGT aggregation
- FP32 OT, charts, manifold state, barriers, analytical solves, and export
- optional FP64 invariant diagnostics
- a server-built `diff_gaussian_rasterization` extension for training renders
- Triton-backed fused SSIM for bounded-memory Phase B/D/E/F supervision

The launcher derives its world size from `CUDA_VISIBLE_DEVICES`; historical
script/config names retain `a800` for compatibility and do not force that GPU
model or a six-rank world.

## Tensor path

```text
images                  [B,K,3,518,518]
VGGT cached taps         4 x [B,K,1374,2048]
orthogonal patch field   [B,K,1369,1024]
camera extrinsics        [B,K,3,4] OpenCV world-to-camera
intrinsics               [B,K,3,3] pixels
depth/confidence         [B,K,518,518,(1)]
evidence particles       variable [M]
active atlas charts      variable [V]
sparse UOT support        [2,E_OT]
local irreps             60(0e)+16(1o)+4(2e) = 128 scalars
selected complex         vertices [Nv], edges [Ne,2], faces [Nf,3]
manifold state           R3 x SO(3) x SPD(3) x R x appearance x latent
surface Gaussians        means/covariance/SH/opacity [G]
mesh                     vertices [Nv,3], faces [Nf,3]
```

The 60 scalar channels comprise the document's `48(0e)+12(0e)` blocks. They
are stored contiguously so equivariant multiplicity maps cannot consume vector
components accidentally.

## Installation on the server

Install the existing VGGT and TRELLIS requirements/checkpoints, build the CUDA
rasterizer, then install this package from the combined repository:

```bash
python -m pip install -e .
```

No baseline source file is patched by this package.

The production-size SSIM kernel and its recomputing adjoint are documented in
[`docs/FUSED_SSIM_MEMORY.md`](docs/FUSED_SSIM_MEMORY.md). Run its allocator and
gradient-parity gate on each deployment topology before a long training job.

## Validation

```bash
python scripts/validate_server.py --output outputs/validation.json
export CUDA_VISIBLE_DEVICES=4,5
torchrun --standalone --nproc-per-node=2 scripts/validate_ddp_server.py
```

For the real checkpoint-backed test:

```bash
export GRAFT_GS_REAL_IMAGE_DIR=/data/object/views
export VGGT_CHECKPOINT=/checkpoints/VGGT-1B
export TRELLIS_CHECKPOINT=/checkpoints/TRELLIS-image-large
export GRAFT_GS_CHECKPOINT=/checkpoints/graft-gs-phase-f.pt
export GRAFT_GS_MESHFLEET_ROOT=/data/MeshFleet_TRELLIS
export GRAFT_GS_MESHFLEET_MANIFEST=$PWD/data_manifests/meshfleet_server.jsonl
python scripts/validate_server.py
```

Untouched upstream control paths can be reproduced independently:

```bash
python scripts/reproduce_baseline.py vggt /data/object/views --output outputs/baselines/vggt.pt
python scripts/reproduce_baseline.py trellis /data/object/reference.png --output outputs/baselines/trellis
```

## Inference

```bash
python scripts/infer_multiview.py /data/object/views outputs/object \
  --vggt-checkpoint /checkpoints/VGGT-1B \
  --graft-checkpoint /checkpoints/graft-gs-phase-f.pt \
  --trellis-checkpoint /checkpoints/TRELLIS-image-large \
  --render-input-views
```

## Staged visible-GPU training

First audit the physical MeshFleet data. The manifest stores relative paths,
reconciles declared/available views, and gates topology labels:

```bash
python scripts/build_meshfleet_manifest.py /data/MeshFleet_TRELLIS \
  data_manifests/meshfleet_server.jsonl
```

Discovery is dynamic and modality-centric. Candidate IDs are collected once
from `latents` and `mesh_normalized`; the default manifest intersection requires
`renders`, `latents`, and a complete normalized mesh directory. DINO features,
structure latents, conditional/evaluation renders, and surface voxels are
optional at discovery time and are recorded explicitly when present or absent.
Training phases apply their own stronger admission policy through the dataset
configuration. Use repeated `--primary-modality`, `--required-modality`, and
`--optional-modality` flags only when intentionally changing this contract.

Run phases in order and initialize each new phase from the preceding model:

```bash
export CUDA_VISIBLE_DEVICES=4,5
export GRAFT_GS_EXPECTED_GPU_COUNT=2
export GRAFT_GS_EXPECTED_GPU_NAME=A100

bash scripts/launch_a800_6gpu.sh /data/MeshFleet_TRELLIS A 20000 --manifest data_manifests/meshfleet_server.jsonl --split train --output outputs/phase_a
bash scripts/launch_a800_6gpu.sh /data/MeshFleet_TRELLIS B 50000 --manifest data_manifests/meshfleet_server.jsonl --split train --trellis-checkpoint "$TRELLIS_CHECKPOINT" --initialize-from outputs/phase_a/final.pt --output outputs/phase_b
bash scripts/launch_a800_6gpu.sh /data/MeshFleet_TRELLIS C 50000 --manifest data_manifests/meshfleet_server.jsonl --split train --trellis-checkpoint "$TRELLIS_CHECKPOINT" --initialize-from outputs/phase_b/final.pt --output outputs/phase_c
bash scripts/launch_a800_6gpu.sh /data/MeshFleet_TRELLIS D 100000 --manifest data_manifests/meshfleet_server.jsonl --split train --trellis-checkpoint "$TRELLIS_CHECKPOINT" --initialize-from outputs/phase_c/final.pt --output outputs/phase_d
bash scripts/launch_a800_6gpu.sh /data/MeshFleet_TRELLIS E 50000 --manifest data_manifests/meshfleet_server.jsonl --split train --trellis-checkpoint "$TRELLIS_CHECKPOINT" --teacher outputs/phase_d/final.pt --initialize-from outputs/phase_d/final.pt --output outputs/phase_e
bash scripts/launch_a800_6gpu.sh /data/MeshFleet_TRELLIS F 30000 --manifest data_manifests/meshfleet_server.jsonl --split train --trellis-checkpoint "$TRELLIS_CHECKPOINT" --initialize-from outputs/phase_e/final.pt --output outputs/phase_f
```

The complete current-revision procedure—including old-checkpoint compatibility,
batch-8 probes, phase-specific tuning, exact resume, inference, and corpus
evaluation—is in
[`docs/TRAINING_AND_TESTING_TUTORIAL.md`](docs/TRAINING_AND_TESTING_TUTORIAL.md).
Before discarding a pre-OOM Phase A checkpoint, run
`scripts/inspect_checkpoint_compatibility.py`; activation checkpointing is an
execution policy and does not by itself require Phase A retraining.

The runtime stage boundary is explicit: A stops after calibrated evidence, B
stops after atlas/topology/readout and never runs the flow, C runs constrained
flow matching but does not build/render assets, and D--F execute the complete
path. Objective weights, including overlap/multilevel, structural image,
reprojection, and tile-opacity terms, are read from the YAML `loss` section and
are part of exact checkpoint compatibility.

Use `--same-object-view-shards` to make all ranks iterate the same object order;
the trainer deterministically partitions its views, autograd-all-gathers the
complete evidence measure, and replicates one global sparse UOT/atlas solve.
This is the high-precision reference because summing rank-local nonlinear UOT
barycenters is not a global UOT solve. In ordinary DDP, ranks
process different objects and only model gradients synchronize. Server defaults
are read from `configs/graft_gs_a800_native.yaml` and can be replaced with
`--config`.

The released geometry model on this path is `vggt.models.vggt.VGGT`—VGGT,
not “VGDT”. Phase B freezes its upstream backbone while retaining the declared
trainable GRAFT-GS evidence, mapping, attention, and topology modules.

The launcher derives one process per GPU from the active
`CUDA_VISIBLE_DEVICES`; it never assumes a fixed GPU count. Use
`--maximum-views N` for the ordinary object-level per-rank view budget. For the
same-object overfit diagnostic, use `--views-per-rank N`; its global sample
contains `N * WORLD_SIZE` views and is sharded before CUDA transfer. Do not add
multiple ranks per GPU or dummy allocations to fill 80 GiB.

Ordinary object-level DDP can batch multiple independent objects in one rank
with `--object-batch-size N`. Batches are grouped by their exact number of
views: the loader never pads a scene into VGGT's joint-view attention, and
variable surfaces/topologies remain separate per-object values. Losses retain
their mean reduction, precision policy, complete view set, and per-object
TRELLIS seed. Same-object view-sharded DDP deliberately remains batch size one.
When `--minimum-global-object-batch` is used, accumulation is reduced as the
physical object batch grows.

The server configuration enables exact GSTA activation recomputation for large
physical object batches. It removes the dominant edge-by-channel tensors from
the retained graph without changing the sparse gauge operator, resolution, or
gradient. Use `--gsta-activation-checkpointing` (or the explicit `--no-...`
override for an A/B measurement). The derivation, measured retained-storage
reduction, and one-GPU CUDA acceptance command are in
[`docs/PROTOCOL9_GSTA_MEMORY.md`](docs/PROTOCOL9_GSTA_MEMORY.md).

Every ordinary DDP batch, including batch size one, is also cohort-balanced by
the manifest's render-face and sparse-surface cardinalities. This prevents one
rank from receiving a tail-complexity Phase-B object while peers wait. Before a
scalar finite-state health collective, each rank completes its own queued CUDA
work; therefore NCCL's timeout measures communication rather than mesh/atlas
kernel latency. The default process-group timeout is 1800 seconds and can be
overridden with `--collective-timeout-seconds`. A rank/object warning is emitted
after 120 seconds of local CUDA delay, configurable with
`--straggler-warning-seconds`.

Select a training batch on the exact phase, dataset, and explicit GPU IDs with
fresh-process probes:

```bash
export GRAFT_GS_PYTHON=/mnt/sda1/miniforge3/envs/CRAFT/bin/python
"$GRAFT_GS_PYTHON" scripts/autotune_object_batch.py \
  --gpus 4,5 --candidates 1 2 4 8 \
  --probe-timeout-seconds 1800 \
  --probe-no-progress-timeout-seconds 900 \
  --probe-warmup-steps 1 --probe-measurement-steps 2 \
  --output outputs/batch_tuning/phase_d --launch -- \
  /data/MeshFleet_TRELLIS --phase D --steps 100000 \
  --manifest data_manifests/meshfleet_server.jsonl --split train \
  --global-object-batch 32 \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --initialize-from outputs/phase_c/final.pt --output outputs/phase_d
```

Each candidate gets a new process group and allocator and starts with the
largest estimated-work cohort. One optimizer step warms model/optimizer/DDP
state and two later steps measure steady-state throughput by default; all three
participate in peak-memory admission. The probe uses one physical microbatch
per optimizer step, so it materializes activations, buckets, gradients, and
AdamW state without replaying enough objects to fill the production accumulated
batch. The real launch retains the original accumulation argument, including
an exact `--global-object-batch`; the probe-only override therefore does not
change training optimization or checkpoint semantics.

Each candidate has a bounded TRELLIS cache under the tuning output. A hit
requires exact conditioning tensor bytes, seed, sampling policy, checkpoint
identity, and upstream source digest. Files are atomically committed under a
process lock and deterministically evicted at the configured byte bound. The
default candidate-local scope preserves comparable end-to-end timings;
`--probe-trellis-cache-scope shared` is available only when intentionally
benchmarking a pre-cached workload.

The selector rejects OOMs and candidates outside allocated, reserved, or
driver-free headroom, then chooses the largest batch within 3% of the best
measured object throughput. The first detected CUDA allocation failure
terminates the complete candidate process group immediately. Larger physical
batches are skipped after that monotonic capacity failure unless
`--continue-after-oom` is explicitly supplied. There is no total-wall-time
candidate kill. Structured per-rank events identify model load, data fetch,
each TRELLIS draw, VGGT, atlas/transport/attention/topology/render, loss,
backward sentinels, optimizer, collectives, and checkpointing. A stage is
terminated only after its configured interval with no semantic advancement.
That interval is a conservative minimum; after three completions the supervisor
also evaluates empirical Q99 + 6×MAD latency and uses the larger budget.
Heartbeats expose liveness and memory but do not reset that deadline. The
timeout record contains the last state and workload of every rank.
With `--launch`, the same semantic supervisor remains around the full training
process and writes `production.log` plus `launch_result.json`; supervision does
not stop after batch selection. Evaluated candidates expose `SUCCESS`, `OOM`,
`NONFINITE`, `NO_PROGRESS`, `COLLECTIVE_FAILURE`, `DATA_FAILURE`,
`MODEL_FAILURE`, or `EXTERNAL_TERMINATION` independently of detailed diagnosis.

For a bounded first-step CPU/CUDA trace, add `--profile` to
`scripts/train_a800.py`. The server profile waits one optimizer step, warms up
one, records the next three with shapes and allocator activity, then stops
collecting; each rank writes a separate Chrome/TensorBoard trace under
`OUTPUT/profiler` (or `--profile-output PATH`). Ordinary training leaves this
disabled.

TRELLIS emits one `Sampling: 12/12` progress display for each posterior draw;
with the default eight draws it is normal to see eight displays per uncached
object and rank. These displays are frozen-prior work, not optimizer steps or
an OOM retry loop. `GRAFT_GS_PROGRESS` records the draw index and cache result.
`GRAFT_GS_AUTOTUNE_CANDIDATE_START`, `_END`, and
`GRAFT_GS_AUTOTUNE_PROBE_CONTROL` delimit the actual candidate lifecycle.
The selected physical batch and resulting global batch are checkpointed.
Changing the global optimizer batch can change optimization statistics. Use
`--global-object-batch N` to retain an exact existing optimizer batch; tuner
candidates that do not divide that global batch are rejected. Use
`--minimum-global-object-batch` only when rounding upward is acceptable.

Tune independently on each hardware pool. An A6000 result must not be copied to
an A100 pool (or between 40-GiB and 80-GiB A100 variants): use the exact visible
GPU IDs, phase, manifest, view budget, and checkpoint with a fresh output
directory on each server. The resulting production launch still uses one DDP
rank per visible GPU.

Evaluation uses the same exact-view batching and deterministic per-object
sharding. It can be launched directly on predetermined GPUs:

```bash
bash scripts/launch_meshfleet_evaluation.sh 0,2,3,5 \
  /data/MeshFleet_TRELLIS data_manifests/meshfleet_server.jsonl \
  outputs/phase_f/final.pt outputs/evaluation \
  --splits test --object-batch-size 4
```

Or measure evaluation batches independently before launching the full corpus:

```bash
"$GRAFT_GS_PYTHON" scripts/autotune_evaluation_batch.py \
  --gpus 0,2,3,5 --candidates 1 2 4 8 \
  --output outputs/batch_tuning/evaluation --launch -- \
  /data/MeshFleet_TRELLIS data_manifests/meshfleet_server.jsonl \
  outputs/phase_f/final.pt outputs/evaluation --splits test
```

The measured view sweep, allocator-ownership check, headroom criterion, and
exact commands are in `docs/A800_VALIDATION_PROTOCOL.md`.

Phases B--F use the configured fixed TRELLIS structure generator as a
Beta-Bernoulli hidden-surface prior. Its checkpoint and sampling/uncertainty
policy are stored in every training checkpoint. The audited DINOv2 and TRELLIS
surface arrays are loaded only as confidence-gated relational pseudo labels;
their channels are never concatenated with or directly regressed from GRAFT-GS
features. Set the corresponding dataset flags or `trellis_prior` policy to
disabled for explicit ablations.

## Scientific records

- `docs/RESEARCH_DECISIONS.md`
- `docs/REPOSITORY_AUDIT.md`
- `docs/MATHEMATICAL_ASSUMPTIONS.md`
- `docs/DEVIATIONS_FROM_SPEC.md`
- `docs/DATASET_AUDIT.md`
- `docs/SPECIFICATION_TRACEABILITY.md`
- `docs/UNRESOLVED_BLOCKERS.md`
- `docs/A800_VALIDATION_PROTOCOL.md`
- `docs/MULTI_GPU_EXECUTION_TOPOLOGY.md`
- `docs/DDP_TRAINING_INCIDENT_ANALYSIS.md`
- `IMPLEMENTATION_LEDGER.md`
- `VALIDATION_LEDGER.md`

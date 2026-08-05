# Current-revision training and testing tutorial

This tutorial targets one process per explicitly selected A6000, A100, or A800
GPU. Historical filenames containing `a800` do not hardcode a GPU model or a
six-rank world size.

## 1. Shell contract

Run every command from the deployed repository root:

```bash
set -euo pipefail
cd /mnt/sda2/hef/Base/GRAFT-GS

export GRAFT_GS_PYTHON=/mnt/sda1/miniforge3/envs/CRAFT/bin/python
export GRAFT_GS_DATASET=/mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97
export GRAFT_GS_MANIFEST=$PWD/data_manifests/meshfleet_server.jsonl
export GRAFT_GS_CONFIG=$PWD/configs/graft_gs_a800_native.yaml
export GRAFT_GS_GPUS=4,5
export CUDA_VISIBLE_DEVICES=$GRAFT_GS_GPUS
export GRAFT_GS_EXPECTED_GPU_COUNT=2
export GRAFT_GS_EXPECTED_GPU_NAME=A100

export GRAFT_GS_VGGT_ROOT=/mnt/sda2/hef/Base/vggt
export GRAFT_GS_TRELLIS_ROOT=/mnt/sda2/hef/Base/TRELLIS
export VGGT_CHECKPOINT=facebook/VGGT-1B
export TRELLIS_CHECKPOINT=microsoft/TRELLIS-image-large

export GRAFT_GS_NPROC_PER_NODE=$(
  "$GRAFT_GS_PYTHON" -c 'import torch; print(torch.cuda.device_count())'
)
test "$GRAFT_GS_NPROC_PER_NODE" -eq 2
mkdir -p outputs/validation
```

The CUDA visibility remap is intentional: torchrun local rank 0 owns physical
GPU 4 and local rank 1 owns physical GPU 5. Confirm both selected devices are
A100s before loading any checkpoint:

```bash
nvidia-smi -i "$GRAFT_GS_GPUS" \
  --query-gpu=index,name,memory.total,memory.free,compute_cap \
  --format=csv,noheader

"$GRAFT_GS_PYTHON" - <<'PY'
import torch
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
for logical in range(2):
    name = torch.cuda.get_device_name(logical)
    assert "A100" in name, (logical, name)
    print(f"logical_cuda={logical} physical_cuda={4 + logical} name={name}")
PY
```

Checkpoint identifiers may be replaced by local paths. Do not set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on the logged A800 software
stack: that runtime reports the feature unsupported.

Record the exact deployed source before training:

```bash
git status --short
git rev-parse HEAD | tee outputs/validation/source_commit.txt
```

A dirty source tree is not reproducible. Either commit the intended changes or
record a patch before running production.

## 2. Environment and dataset preflight

Audit dependencies, upstream checkouts, CUDA, and the existing manifest:

```bash
"$GRAFT_GS_PYTHON" scripts/validate_server.py \
  --requirements requirements.txt \
  --dataset-root "$GRAFT_GS_DATASET" \
  --manifest "$GRAFT_GS_MANIFEST" \
  --vggt-root "$GRAFT_GS_VGGT_ROOT" \
  --trellis-root "$GRAFT_GS_TRELLIS_ROOT" \
  --output outputs/validation/server.json
```

Build a new manifest only if the dataset changed or no audited manifest exists:

```bash
"$GRAFT_GS_PYTHON" scripts/build_meshfleet_manifest.py \
  "$GRAFT_GS_DATASET" "$GRAFT_GS_MANIFEST" \
  --splits train val test
```

Validate rank-to-device ownership and the NCCL/scientific contract:

```bash
"$GRAFT_GS_PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 \
  --nproc-per-node="$GRAFT_GS_NPROC_PER_NODE" \
  scripts/validate_ddp_server.py \
  --requirements requirements.txt \
  --output outputs/validation/ddp_environment.json
```

All explicitly selected GPUs must be idle before the DDP or tuning commands.

Verify that the training interpreter contains Triton, then run the exact-shape
SSIM value/adjoint/allocation gate on one isolated GPU. Phase B/D/E/F refuses
to start without this backend because the eager CUDA SSIM path has an
unbounded full-image tape at the target batch.

```bash
"$GRAFT_GS_PYTHON" -c \
  'import torch, triton; print(torch.__version__, triton.__version__)'

CUDA_VISIBLE_DEVICES=4 "$GRAFT_GS_PYTHON" \
  scripts/benchmark_fused_ssim.py \
  --batch 8 --views 24 --height 518 --width 518 \
  --output outputs/validation/fused_ssim_gpu4.json
```

Require a zero exit status, positive `peak_reduction_bytes`, loss absolute
error at most `2e-7`, and gradient relative L2 at most `5e-6`. The repository's
measured A6000 result is in
[`FUSED_SSIM_MEMORY.md`](FUSED_SSIM_MEMORY.md); repeat the gate on the A100/A800
rather than extrapolating an sm_86 measurement to sm_80.

## 3. Decide whether an old Phase A checkpoint can be reused

The OOM revision added `attention.activation_checkpointing`, which is an
execution policy with no parameter or buffer. Current loaders ignore only this
declared execution-only difference and still reject every real architectural
difference.

Inspect the actual production checkpoint without loading VGGT or TRELLIS:

```bash
"$GRAFT_GS_PYTHON" scripts/inspect_checkpoint_compatibility.py \
  outputs/phase_a/final.pt \
  --config "$GRAFT_GS_CONFIG" \
  --require-source-phase A \
  | tee outputs/validation/phase_a_compatibility.json
```

Reuse Phase A when all of the following are true:

- `compatible_without_retraining` is `true`;
- `architectural_differences` is empty;
- `graft_state_schema.compatible` is `true`;
- the only ignored difference is
  `attention.activation_checkpointing` being absent or false in the checkpoint
  and true now.

Retrain or explicitly migrate Phase A if the inspector exits nonzero. Never
silence differences in feature width, irrep multiplicities, encoder depth,
atlas/flow/barrier/readout configuration, or GRAFT state tensor keys/shapes.
If the report passes, preserve the old `outputs/phase_a/final.pt`, skip the
Phase A command in section 7, and begin with the Phase B probe.

## 4. Checkpoint semantics

- `--initialize-from PATH` transfers model weights into a new phase. Optimizer,
  step, epoch, sampler, and RNG state start fresh.
- `--resume PATH` is an exact continuation of the same phase. It restores model,
  optimizer, counters, gradient purifier, and rank-local RNG state. World size,
  dataset provenance, precision, objective, physical batch, and accumulation
  policy must match.
- Never pass both. For resume, `--steps` is the target global step, not a number
  of additional steps.
- Phase E requires both `--teacher outputs/phase_d/final.pt` and
  `--initialize-from outputs/phase_d/final.pt`.

## 5. Phase B batch-8 repair gate

After the compatibility report passes, rerun the exact failing physical batch
in a fresh output. Two ranks times eight objects gives physical global batch
16. `--global-object-batch 32` therefore selects exactly two gradient-
accumulation microsteps per optimizer step; the former global optimizer batch
and loss scaling remain unchanged:

```bash
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
PROBE_ROOT="outputs/validation/phase-b-batch8-$RUN_TAG"
mkdir -p "$PROBE_ROOT"

CUDA_VISIBLE_DEVICES="$GRAFT_GS_GPUS" \
bash scripts/launch_a800_6gpu.sh \
  "$GRAFT_GS_DATASET" B 50000 \
  --config "$GRAFT_GS_CONFIG" \
  --manifest "$GRAFT_GS_MANIFEST" --split train \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --initialize-from outputs/phase_a/final.pt \
  --object-batch-size 8 --global-object-batch 32 \
  --gsta-activation-checkpointing \
  --batch-probe "$PROBE_ROOT/probe.json" \
  --batch-probe-warmup-steps 1 \
  --batch-probe-measurement-steps 2 \
  --output "$PROBE_ROOT/training" \
  2>&1 | tee "$PROBE_ROOT/run.log"
```

The process-start records must report
`"ssim_backend":"triton_recomputing_adjoint"`. The probe must contain both
ranks, three finite optimizer steps, safe peak
allocated/reserved/driver-free fractions, and no OOM, collective error, or
non-finite diagnostic. A successful checkpoint load only resolves the current
configuration error; it does not by itself prove that the later batch-8 forward
fits.

Then run a 200-step soak before committing to 50,000 steps:

```bash
SOAK_ROOT="outputs/validation/phase-b-batch8-soak-$RUN_TAG"
CUDA_VISIBLE_DEVICES="$GRAFT_GS_GPUS" \
bash scripts/launch_a800_6gpu.sh \
  "$GRAFT_GS_DATASET" B 200 \
  --config "$GRAFT_GS_CONFIG" \
  --manifest "$GRAFT_GS_MANIFEST" --split train \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --initialize-from outputs/phase_a/final.pt \
  --object-batch-size 8 --global-object-batch 32 \
  --gsta-activation-checkpointing \
  --output "$SOAK_ROOT" \
  2>&1 | tee "$SOAK_ROOT.log"

test -f "$SOAK_ROOT/final.pt"
! rg -n "OutOfMemoryError|NONFINITE|Watchdog caught|DistBackendError" \
  "$SOAK_ROOT.log"
```

## 6. Throughput-optimal batch selection

Physical batch sizes must be tuned independently for every phase and hardware
pool. The following Phase B sweep preserves global optimizer batch 32 and does
not launch full training:

```bash
TUNE_B="outputs/batch_tuning/phase-b-$(date -u +%Y%m%dT%H%M%SZ)"
"$GRAFT_GS_PYTHON" scripts/autotune_object_batch.py \
  --gpus "$GRAFT_GS_GPUS" --candidates 1 2 4 8 \
  --probe-warmup-steps 1 --probe-measurement-steps 2 \
  --output "$TUNE_B" -- \
  "$GRAFT_GS_DATASET" --phase B --steps 50000 \
  --config "$GRAFT_GS_CONFIG" \
  --manifest "$GRAFT_GS_MANIFEST" --split train \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --initialize-from outputs/phase_a/final.pt \
  --global-object-batch 32 --gsta-activation-checkpointing \
  --output outputs/phase_b

BATCH_B=$(
  "$GRAFT_GS_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_object_batch_size"])' \
  "$TUNE_B/selection.json"
)
echo "selected Phase B physical batch: $BATCH_B"
```

Repeat the sweep for A, C, D, E, and F using that phase's command from the next
section. Use fresh tuning directories. Phase E must be tuned separately because
the frozen teacher and student coexist; Phase F has a separate hardening replay
and gradient-purification memory profile. Never transfer an A6000 selection to
an A100/A800 pool.

## 7. Full staged training

Set each `BATCH_*` from its phase-specific tuning report. `1` is the safe
configuration fallback, not a throughput claim:

```bash
: "${BATCH_A:=1}" "${BATCH_B:=1}" "${BATCH_C:=1}"
: "${BATCH_D:=1}" "${BATCH_E:=1}" "${BATCH_F:=1}"
```

Train in order:

Use a fresh output directory for every new phase. If a path below already
contains artifacts from a failed or different revision, choose a new suffixed
path and update every subsequent `--initialize-from` reference; do not append a
new phase to ambiguous metrics/checkpoints. Reuse the same directory only for
the exact `--resume` workflow in section 8.

```bash
CUDA_VISIBLE_DEVICES="$GRAFT_GS_GPUS" bash scripts/launch_a800_6gpu.sh \
  "$GRAFT_GS_DATASET" A 20000 \
  --config "$GRAFT_GS_CONFIG" --manifest "$GRAFT_GS_MANIFEST" --split train \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --object-batch-size "$BATCH_A" --global-object-batch 32 \
  --gsta-activation-checkpointing --output outputs/phase_a

CUDA_VISIBLE_DEVICES="$GRAFT_GS_GPUS" bash scripts/launch_a800_6gpu.sh \
  "$GRAFT_GS_DATASET" B 50000 \
  --config "$GRAFT_GS_CONFIG" --manifest "$GRAFT_GS_MANIFEST" --split train \
  --vggt-checkpoint "$VGGT_CHECKPOINT" --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --initialize-from outputs/phase_a/final.pt \
  --object-batch-size "$BATCH_B" --global-object-batch 32 \
  --gsta-activation-checkpointing --output outputs/phase_b

CUDA_VISIBLE_DEVICES="$GRAFT_GS_GPUS" bash scripts/launch_a800_6gpu.sh \
  "$GRAFT_GS_DATASET" C 50000 \
  --config "$GRAFT_GS_CONFIG" --manifest "$GRAFT_GS_MANIFEST" --split train \
  --vggt-checkpoint "$VGGT_CHECKPOINT" --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --initialize-from outputs/phase_b/final.pt \
  --object-batch-size "$BATCH_C" --global-object-batch 32 \
  --gsta-activation-checkpointing --output outputs/phase_c

CUDA_VISIBLE_DEVICES="$GRAFT_GS_GPUS" bash scripts/launch_a800_6gpu.sh \
  "$GRAFT_GS_DATASET" D 100000 \
  --config "$GRAFT_GS_CONFIG" --manifest "$GRAFT_GS_MANIFEST" --split train \
  --vggt-checkpoint "$VGGT_CHECKPOINT" --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --initialize-from outputs/phase_c/final.pt \
  --object-batch-size "$BATCH_D" --global-object-batch 32 \
  --gsta-activation-checkpointing --output outputs/phase_d

CUDA_VISIBLE_DEVICES="$GRAFT_GS_GPUS" bash scripts/launch_a800_6gpu.sh \
  "$GRAFT_GS_DATASET" E 50000 \
  --config "$GRAFT_GS_CONFIG" --manifest "$GRAFT_GS_MANIFEST" --split train \
  --vggt-checkpoint "$VGGT_CHECKPOINT" --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --teacher outputs/phase_d/final.pt \
  --initialize-from outputs/phase_d/final.pt \
  --object-batch-size "$BATCH_E" --global-object-batch 32 \
  --gsta-activation-checkpointing --output outputs/phase_e

CUDA_VISIBLE_DEVICES="$GRAFT_GS_GPUS" bash scripts/launch_a800_6gpu.sh \
  "$GRAFT_GS_DATASET" F 30000 \
  --config "$GRAFT_GS_CONFIG" --manifest "$GRAFT_GS_MANIFEST" --split train \
  --vggt-checkpoint "$VGGT_CHECKPOINT" --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --initialize-from outputs/phase_e/final.pt \
  --object-batch-size "$BATCH_F" --global-object-batch 32 \
  --gsta-activation-checkpointing --output outputs/phase_f
```

If offline teacher bundles have been generated, add
`--teacher-bundle-root "$GRAFT_GS_TEACHER_BUNDLES"` to Phase C. Omitting that
optional flag retains the ordinary Phase C objective.

## 8. Exact same-phase resume

Use the same GPU count, physical batch, global batch, manifest, precision,
prior, and phase arguments. This example continues Phase B to target step
50,000:

```bash
CUDA_VISIBLE_DEVICES="$GRAFT_GS_GPUS" bash scripts/launch_a800_6gpu.sh \
  "$GRAFT_GS_DATASET" B 50000 \
  --config "$GRAFT_GS_CONFIG" --manifest "$GRAFT_GS_MANIFEST" --split train \
  --vggt-checkpoint "$VGGT_CHECKPOINT" --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --resume outputs/phase_b/step-00020000.pt \
  --object-batch-size "$BATCH_B" --global-object-batch 32 \
  --gsta-activation-checkpointing --output outputs/phase_b
```

Do not use `--initialize-from` for interruption recovery.

An exact checkpoint created with four DDP ranks cannot be resumed with this
two-rank topology because world size and accumulation policy are checkpointed.
Phase-to-phase `--initialize-from` remains valid: for example, a compatible
Phase-A model trained on four ranks can initialize a fresh two-rank Phase-B
run because optimizer, sampler, and rank-local RNG state are deliberately not
transferred.

## 9. Local numerical and GSTA memory tests

Run CPU/static validation before deployment:

```bash
"$GRAFT_GS_PYTHON" -m unittest \
  tests.test_fused_ssim \
  tests.test_checkpoint_config_compatibility \
  tests.test_geometry_invariants \
  tests.test_object_batching \
  tests.test_autotune_object_batch \
  tests.test_scientific_trace_static

"$GRAFT_GS_PYTHON" scripts/benchmark_gsta_memory.py \
  --device cpu --vertices 8192 --layers 4 \
  --output outputs/validation/gsta-retention-cpu.json
```

Measure actual peak allocation on one idle GPU from each hardware pool:

```bash
CUDA_VISIBLE_DEVICES=4 "$GRAFT_GS_PYTHON" \
  scripts/benchmark_gsta_memory.py \
  --device cuda --vertices 8192 --layers 4 \
  --output outputs/validation/gsta-memory-gpu.json
```

The benchmark exits nonzero if checkpointing fails numerical tolerances or does
not reduce both retained storage and CUDA peak allocation.

The repository-wide discovery command is:

```bash
"$GRAFT_GS_PYTHON" -m unittest discover -s tests -p 'test_*.py'
```

In this workspace audit, the focused 88-test SSIM/checkpoint/GSTA/DDP/tuning
gate passes. Full discovery runs 235 tests with 11
environment skips but also exposes five pre-existing, unrelated failures: three
non-leaf-tensor `deepcopy` errors in analytical asset tests, one perceptual-loss
threshold assertion, and one TRELLIS-prior monotonicity assertion. None is a
regression of the checkpoint/SSIM repair, but the repository-wide release
gate must not be reported as fully green until those independent failures are
resolved.

## 10. Single-object inference

Use one explicitly selected idle GPU and a final Phase F checkpoint:

```bash
export GRAFT_GS_OBJECT_ID=<64-hex-object-id>
CUDA_VISIBLE_DEVICES=4 "$GRAFT_GS_PYTHON" scripts/infer_meshfleet.py \
  "$GRAFT_GS_DATASET" "$GRAFT_GS_MANIFEST" \
  outputs/phase_f/final.pt outputs/inference/$GRAFT_GS_OBJECT_ID \
  --config "$GRAFT_GS_CONFIG" --split test \
  --object-id "$GRAFT_GS_OBJECT_ID" --maximum-views 12 \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --profile-trace outputs/inference/$GRAFT_GS_OBJECT_ID/trace.json
```

Do not provide quantization-certificate bounds unless they were measured on the
same checkpoint and numerical path.

For an ordinary directory of multiview images:

```bash
CUDA_VISIBLE_DEVICES=4 "$GRAFT_GS_PYTHON" scripts/infer_multiview.py \
  /data/object/views outputs/inference/multiview \
  --config "$GRAFT_GS_CONFIG" --maximum-views 12 \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --graft-checkpoint outputs/phase_f/final.pt \
  --render-input-views
```

## 11. Full multi-GPU test evaluation

Tune evaluation batch size independently:

```bash
EVAL_TUNE="outputs/batch_tuning/evaluation-$(date -u +%Y%m%dT%H%M%SZ)"
"$GRAFT_GS_PYTHON" scripts/autotune_evaluation_batch.py \
  --gpus "$GRAFT_GS_GPUS" --candidates 1 2 4 8 \
  --output "$EVAL_TUNE" --launch -- \
  "$GRAFT_GS_DATASET" "$GRAFT_GS_MANIFEST" \
  outputs/phase_f/final.pt outputs/evaluation \
  --config "$GRAFT_GS_CONFIG" --splits test --maximum-views 12 \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT"
```

Alternatively, run a predetermined evaluation batch directly:

```bash
bash scripts/launch_meshfleet_evaluation.sh "$GRAFT_GS_GPUS" \
  "$GRAFT_GS_DATASET" "$GRAFT_GS_MANIFEST" \
  outputs/phase_f/final.pt outputs/evaluation \
  --config "$GRAFT_GS_CONFIG" --splits test \
  --maximum-views 12 --object-batch-size 4 \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT"
```

Require one successful record per admitted test object in
`outputs/evaluation/metrics.jsonl`, no rank-local error records, finite metrics,
and valid PLY/GLB assets. Use fresh output and tuning directories for every
checkpoint/hardware comparison.

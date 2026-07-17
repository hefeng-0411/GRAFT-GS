# Exact six-A800 validation protocol

This protocol defines server-ready commands; none of the resulting measurements
are claimed until their logs and artifacts exist. Run from the GRAFT-GS root in
the pinned VGGT/TRELLIS/PyTorch environment.

## Required environment

```bash
export VGGT_CHECKPOINT=/checkpoints/VGGT-1B
export TRELLIS_CHECKPOINT=/checkpoints/TRELLIS-image-large
export GRAFT_GS_CHECKPOINT=/checkpoints/graft-gs-phase-f.pt
export GRAFT_GS_REAL_IMAGE_DIR=/data/real_multiview_object/images
export GRAFT_GS_MESHFLEET_ROOT=/mnt/sda2/hef/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97
export GRAFT_GS_MESHFLEET_MANIFEST=$PWD/outputs/validation/meshfleet_server.jsonl
export GRAFT_GS_TEACHER_BUNDLES=/data/graft_gs_teacher_bundles
export GRAFT_GS_RUN_TRAINING_TESTS=1
export GRAFT_GS_PYTHON=/mnt/sda1/miniforge3/envs/CRAFT/bin/python
export PYTHONHASHSEED=0
```

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
build and at least one visible A800. With `CUDA_VISIBLE_DEVICES=0`, one visible
device is correct here; the dedicated `torchrun --nproc-per-node=6` suite
validates all six devices and collectives separately.

Build the manifest from the mounted data and retain its SHA-256 digest:

```bash
python scripts/build_meshfleet_manifest.py \
  "$GRAFT_GS_MESHFLEET_ROOT" "$GRAFT_GS_MESHFLEET_MANIFEST"
sha256sum "$GRAFT_GS_MESHFLEET_MANIFEST" | tee outputs/manifest.sha256
```

## High-precision reference suite

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/sda1/miniforge3/envs/CRAFT/bin/python scripts/validate_server.py \
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
subprocess. It rejects dataset/backend skips; only the separate six-rank DDP
launch and a separately configured real-image/checkpoint run may remain skipped
in this single-GPU reference command.

Manifest reuse is conditional on all of: resolved root equality, schema
`meshfleet-trellis-object-v2`, summary/JSONL record-count equality, readable
JSON objects, and exactly one occurrence of the canonical schema ID. Any failed
condition invokes the deterministic builder before importing the dataset. Use
`--rebuild-manifest` to force a fresh full-corpus audit even when those identity
checks pass.

## Reference/CUDA renderer equivalence

After building `diff_gaussian_rasterization` on the server:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unittest \
  tests.test_assets_and_vertical_slice.AnalyticalAssetTest.test_cuda_reference_equivalence_small_scene -v \
  2>&1 | tee outputs/validation/renderer_equivalence.log
```

Required tolerances are encoded in the test: RGB/alpha `atol=5e-2, rtol=8e-2`
and mean visible normal cosine above `0.9`.

## Exact checkpoint and next-step replay

```bash
CUDA_VISIBLE_DEVICES=0 GRAFT_GS_RUN_TRAINING_TESTS=1 \
  python -m unittest \
  tests.test_real_multiview.RealMultiviewTest.test_trainer_checkpoint_round_trip -v \
  2>&1 | tee outputs/validation/checkpoint_single_gpu.log
```

The checkpoint must be format 5, restore model/optimizer/counters/objective,
and reproduce the next Torch random sample exactly.

## Same-object distributed evidence and rank-local RNG

```bash
$GRAFT_GS_PYTHON -m torch.distributed.run --standalone --nproc-per-node=6 \
  scripts/validate_ddp_server.py \
  --requirements requirements.txt \
  --output outputs/validation/ddp_environment.json \
  2>&1 | tee outputs/validation/ddp_six_rank.log
```

The suite must show identical global evidence/prior inputs while retaining
nonzero rank-local autograd. It also restores each rank's Torch CPU/CUDA,
NumPy, and Python stream and verifies the six streams do not collapse. A
format-5 trainer checkpoint must additionally reject a different resume world
size before mutating model state. Its JSON preflight must contain six distinct
host/local-rank assignments and six A800 records; success is reduced across all
ranks, not inferred from rank zero alone.

## Offline teacher bundle refinement

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/refine_teacher_bundle.py \
  "$GRAFT_GS_MESHFLEET_ROOT" "$GRAFT_GS_MESHFLEET_MANIFEST" \
  outputs/phase_d/final.pt \
  17a53839ae5da04c75ea21335d4bdc8ddc26b45f7bb9d0e18f5afaa397e43a17 \
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
torchrun --standalone --nproc-per-node=6 scripts/overfit_meshfleet_object.py \
  "$GRAFT_GS_MESHFLEET_ROOT" "$GRAFT_GS_MESHFLEET_MANIFEST" \
  --object-id 17a53839ae5da04c75ea21335d4bdc8ddc26b45f7bb9d0e18f5afaa397e43a17 \
  --config configs/graft_gs_a800_native.yaml \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --steps 1000 --output outputs/overfit_canonical \
  2>&1 | tee outputs/overfit_canonical/run.log
```

Required artifacts: periodic and final checkpoints, `metrics.jsonl`, decreasing
overfit objective, input-view renders, deterministic PLY/GLB, reload metrics,
and no activation of hard raw-mesh topology loss. Because this canonical mesh
is nonmanifold, success does not include matching its raw Betti numbers.

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
CUDA_VISIBLE_DEVICES=0 python scripts/infer_multiview.py \
  "$GRAFT_GS_REAL_IMAGE_DIR" outputs/real_multiview \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --trellis-checkpoint "$TRELLIS_CHECKPOINT" \
  --graft-checkpoint "$GRAFT_GS_CHECKPOINT" --render-input-views \
  2>&1 | tee outputs/real_multiview/run.log

CUDA_VISIBLE_DEVICES=0 python scripts/infer_meshfleet.py \
  "$GRAFT_GS_MESHFLEET_ROOT" "$GRAFT_GS_MESHFLEET_MANIFEST" \
  "$GRAFT_GS_CHECKPOINT" outputs/meshfleet_inference \
  --object-id 17a53839ae5da04c75ea21335d4bdc8ddc26b45f7bb9d0e18f5afaa397e43a17 \
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

## Ablations

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_ablations.py \
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

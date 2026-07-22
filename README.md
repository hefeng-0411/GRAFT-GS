# GRAFT-GS research implementation

This package implements a static-3D Gauge-Riemannian Atlas Flow Transformer
beside the unmodified `vggt/` and `TRELLIS/` baseline trees.

## Server target

- Linux, six NVIDIA A800 80 GB GPUs
- PyTorch 2.4 or newer with CUDA/NCCL
- native BF16 VGGT aggregation
- FP32 OT, charts, manifold state, barriers, analytical solves, and export
- optional FP64 invariant diagnostics
- a server-built `diff_gaussian_rasterization` extension for training renders

The local RTX 2060 environment is not a validation target.

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

## Validation

```bash
python scripts/validate_server.py --output outputs/validation.json
torchrun --standalone --nproc-per-node=6 scripts/validate_ddp_server.py
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

## Staged six-GPU training

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
bash scripts/launch_a800_6gpu.sh /data/MeshFleet_TRELLIS A 20000 --manifest data_manifests/meshfleet_server.jsonl --split train --output outputs/phase_a
bash scripts/launch_a800_6gpu.sh /data/MeshFleet_TRELLIS B 50000 --manifest data_manifests/meshfleet_server.jsonl --split train --trellis-checkpoint "$TRELLIS_CHECKPOINT" --initialize-from outputs/phase_a/final.pt --output outputs/phase_b
bash scripts/launch_a800_6gpu.sh /data/MeshFleet_TRELLIS C 50000 --manifest data_manifests/meshfleet_server.jsonl --split train --trellis-checkpoint "$TRELLIS_CHECKPOINT" --initialize-from outputs/phase_b/final.pt --output outputs/phase_c
bash scripts/launch_a800_6gpu.sh /data/MeshFleet_TRELLIS D 100000 --manifest data_manifests/meshfleet_server.jsonl --split train --trellis-checkpoint "$TRELLIS_CHECKPOINT" --initialize-from outputs/phase_c/final.pt --output outputs/phase_d
bash scripts/launch_a800_6gpu.sh /data/MeshFleet_TRELLIS E 50000 --manifest data_manifests/meshfleet_server.jsonl --split train --trellis-checkpoint "$TRELLIS_CHECKPOINT" --teacher outputs/phase_d/final.pt --initialize-from outputs/phase_d/final.pt --output outputs/phase_e
bash scripts/launch_a800_6gpu.sh /data/MeshFleet_TRELLIS F 30000 --manifest data_manifests/meshfleet_server.jsonl --split train --trellis-checkpoint "$TRELLIS_CHECKPOINT" --initialize-from outputs/phase_e/final.pt --output outputs/phase_f
```

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

The launcher derives one process per GPU from the active
`CUDA_VISIBLE_DEVICES`; it never assumes that all six A800s are idle. Use
`--maximum-views N` for the ordinary object-level per-rank view budget. For the
same-object overfit diagnostic, use `--views-per-rank N`; its global sample
contains `N * WORLD_SIZE` views and is sharded before CUDA transfer. Do not add
multiple ranks per GPU or dummy allocations to fill 80 GiB. The measured
8/12/16-view sweep, allocator-ownership check, headroom criterion, and exact
commands are in `docs/A800_VALIDATION_PROTOCOL.md`.

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
- `IMPLEMENTATION_LEDGER.md`
- `VALIDATION_LEDGER.md`

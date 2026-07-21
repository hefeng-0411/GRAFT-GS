#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 DATASET PHASE STEPS [extra train_a800.py arguments...]" >&2
  exit 2
fi

DATASET=$1
PHASE=$2
STEPS=$3
shift 3

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${GRAFT_GS_PYTHON:-/mnt/sda1/miniforge3/envs/CRAFT/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "GRAFT-GS server interpreter is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must name the scheduler-assigned idle GPU subset" >&2
  exit 2
fi

# Bind server training to explicit upstream source checkouts. Environment
# overrides support relocated mirrors; the declared paths are the defaults.
export GRAFT_GS_VGGT_ROOT=${GRAFT_GS_VGGT_ROOT:-/mnt/sda2/hef/Base/vggt}
export GRAFT_GS_TRELLIS_ROOT=${GRAFT_GS_TRELLIS_ROOT:-/mnt/sda2/hef/Base/TRELLIS}
if [[ ! -f "$GRAFT_GS_VGGT_ROOT/vggt/__init__.py" || ! -f "$GRAFT_GS_VGGT_ROOT/demo_gradio.py" ]]; then
  echo "declared VGGT checkout is incomplete: $GRAFT_GS_VGGT_ROOT" >&2
  exit 2
fi
if [[ ! -f "$GRAFT_GS_TRELLIS_ROOT/trellis/__init__.py" || ! -f "$GRAFT_GS_TRELLIS_ROOT/app.py" ]]; then
  echo "declared TRELLIS checkout is incomplete: $GRAFT_GS_TRELLIS_ROOT" >&2
  exit 2
fi

"$PYTHON_BIN" "$ROOT/scripts/validate_environment.py" \
  --requirements "$ROOT/requirements.txt" \
  --output "$ROOT/outputs/validation/training_environment.json"

# torch.cuda.device_count() is authoritative after CUDA_VISIBLE_DEVICES has
# been applied and also supports UUID-based masks. Do not parse the mask as a
# comma-separated list or assume that all six physical A800s are idle.
NPROC_PER_NODE=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')
if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "CUDA_VISIBLE_DEVICES exposes no usable CUDA device: $NPROC_PER_NODE" >&2
  exit 2
fi

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$NPROC_PER_NODE" \
  "$ROOT/scripts/train_a800.py" "$DATASET" --phase "$PHASE" --steps "$STEPS" "$@"

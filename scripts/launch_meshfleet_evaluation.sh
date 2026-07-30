#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 GPU_IDS DATASET_ROOT MANIFEST CHECKPOINT OUTPUT [evaluation arguments...]" >&2
  exit 2
fi

GPU_IDS=$1
shift
if [[ ! "$GPU_IDS" =~ ^[A-Za-z0-9:._,/-]+$ ]]; then
  echo "GPU_IDS contains unsupported characters: $GPU_IDS" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=$GPU_IDS

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${GRAFT_GS_PYTHON:-/mnt/sda1/miniforge3/envs/CRAFT/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "GRAFT-GS server interpreter is not executable: $PYTHON_BIN" >&2
  exit 2
fi

NPROC_PER_NODE=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')
if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "the explicit GPU set exposes no usable CUDA device: $NPROC_PER_NODE" >&2
  exit 2
fi

export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-${NCCL_ASYNC_ERROR_HANDLING:-1}}
unset NCCL_ASYNC_ERROR_HANDLING

"$PYTHON_BIN" "$ROOT/scripts/validate_environment.py" \
  --requirements "$ROOT/requirements.txt" \
  --output "$ROOT/outputs/validation/evaluation_environment.json"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$NPROC_PER_NODE" \
  "$ROOT/scripts/evaluate_meshfleet.py" "$@"

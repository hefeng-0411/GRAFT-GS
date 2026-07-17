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

"$PYTHON_BIN" "$ROOT/scripts/validate_environment.py" \
  --requirements "$ROOT/requirements.txt" \
  --output "$ROOT/outputs/validation/training_environment.json"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=6 \
  "$ROOT/scripts/train_a800.py" "$DATASET" --phase "$PHASE" --steps "$STEPS" "$@"

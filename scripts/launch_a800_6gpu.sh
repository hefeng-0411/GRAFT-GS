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

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=6 \
  scripts/train_a800.py "$DATASET" --phase "$PHASE" --steps "$STEPS" "$@"


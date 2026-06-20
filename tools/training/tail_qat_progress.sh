#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
RUN_DIR="${QAT_RUN_DIR:-artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8}"
INTERVAL="${QAT_CHECKPOINT_INTERVAL:-5}"
REFRESH_SECONDS="${QAT_REFRESH_SECONDS:-0}"

if [[ "${QAT_CLEAR:-1}" == "1" ]] && command -v clear >/dev/null 2>&1; then
  clear
fi

exec "$PY" tools/training/rwkv7_progress.py \
  --run-dir "$RUN_DIR" \
  --next-checkpoint \
  --checkpoint-interval "$INTERVAL" \
  --follow \
  --refresh-seconds "$REFRESH_SECONDS" \
  "$@"

#!/bin/bash
# path-to-125 L8 iter-5 — K=4 vector gate, NO curriculum.
#
# Tests whether iter 4's K=2 success scales to K=4 when the head is
# trained on K=4 chain rollouts from step 0 (no curriculum holding it
# back). Vector gate + K=4 training signal = the bet.
#
# Risk: K=4 phase under contention was 22-76 s/step in iter 3; with
# no Claude running and iter 4 finished, expect 5-15 s/step. Full run
# ~2-4 hours.
#
# Usage: tools/eagle4_iter5_k4_vector.sh [--nohup]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PYTHON="/Users/scammermike/Downloads/dismantle/eagle4/.venv/bin/python"
CKPT_DIR="eagle4/checkpoints/eagle4_iter5_k4_vector"
LOG_DIR="reports/path_to_90/_levers"
LOG_FILE="$LOG_DIR/l8_train.log"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

CMD=(
  "$VENV_PYTHON" eagle4/eagle4.py train
  --parquet training_data/c2_hidden/eagle4_v0/shard_*.parquet
  --frozen eagle4/v2lite_frozen.npz
  --ckpt-dir "$CKPT_DIR"
  --epochs 2
  --multi-step-k 4
  --multi-step-decay 0.7
  --chain-h-high
  --target-warmup-steps 500
  --multi-step-aux-decay 0.05
  --gate-init 0.1
  --gate-lr-multiplier 10.0
  --gate-shape vector
  --lr-schedule cosine
  --lr-min-ratio 0.1
  # No --k-curriculum: K=4 from step 0 so vector gate trains under real
  # K=4 chain rollouts.
)

if [ "${1:-}" = "--nohup" ]; then
  echo "[iter5] launching nohup K=4 vector training; log=$LOG_FILE"
  nohup nice -n 19 taskpolicy -b "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
  TRAIN_PID=$!
  cat >"$LOG_DIR/l8_status.json" <<JSON
{
  "pid": $TRAIN_PID,
  "started_at": "$(date -u +%FT%TZ)",
  "ckpt_dir": "$CKPT_DIR",
  "log_file": "$LOG_FILE",
  "iter_name": "iter5_k4_vector",
  "chain_k_for_smoke": 4,
  "recipe": "K=4 vector gate, no curriculum, all iter-4 patches kept"
}
JSON
  echo "[iter5] pid=$TRAIN_PID"
  exit 0
fi

echo "[iter5] foreground run"
nice -n 19 taskpolicy -b "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

#!/bin/bash
# path-to-125 L8 iter-6 — K=4 vector gate + fix-(h) chain-rollout
# regularizer. Fallback if iter 5 K=4 plateaus below 25% accept.
#
# Adds `--chain-reg-weight 0.1` which subtracts
# `0.1 * mean((draft_h_k+1 - draft_h_k)^2)` from each chain step's
# loss — actively rewards the head for evolving draft_hidden across
# rollouts. Together with vector gate, attacks both axes of the
# chain-decode failure mode.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PYTHON="/Users/scammermike/Downloads/dismantle/eagle4/.venv/bin/python"
CKPT_DIR="eagle4/checkpoints/eagle4_iter6_k4_chainreg"
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
  --chain-reg-weight 0.1
)

if [ "${1:-}" = "--nohup" ]; then
  echo "[iter6] launching nohup K=4 + chain-reg training"
  nohup nice -n 19 taskpolicy -b "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
  TRAIN_PID=$!
  cat >"$LOG_DIR/l8_status.json" <<JSON
{
  "pid": $TRAIN_PID,
  "started_at": "$(date -u +%FT%TZ)",
  "ckpt_dir": "$CKPT_DIR",
  "log_file": "$LOG_FILE",
  "iter_name": "iter6_k4_chainreg",
  "chain_k_for_smoke": 4,
  "recipe": "K=4 vector + chain-reg 0.1"
}
JSON
  echo "[iter6] pid=$TRAIN_PID"
  exit 0
fi

echo "[iter6] foreground run"
nice -n 19 taskpolicy -b "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

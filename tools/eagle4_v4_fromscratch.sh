#!/bin/bash
# path-to-125 L8L — launch the Eagle4 v4 from-scratch retrain.
#
# Trains EagleHead with --gate-init 0.1 and no --resume, in line with
# closeout § Branch 3 fix (e): a non-trivial initial residual_gate
# forces the head to either learn to use block_output or actively zero
# the gate, rather than starting near zero and staying there.
#
# Usage:
#   tools/eagle4_v4_fromscratch.sh           # foreground (blocking)
#   tools/eagle4_v4_fromscratch.sh --nohup   # background via nohup
#
# Output lands in reports/path_to_90/_levers/l8_train.log (regardless
# of mode). Status pings every ~200 steps (printed by the train loop)
# are visible in the log; grep for "chain_accept" or "loss=" to track.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PYTHON="/Users/scammermike/Downloads/dismantle/eagle4/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  echo "[eagle4_v4_fromscratch] mlx venv not found at $VENV_PYTHON" >&2
  exit 1
fi

CKPT_DIR="eagle4/checkpoints/eagle4_v4_fromscratch"
LOG_DIR="reports/path_to_90/_levers"
LOG_FILE="$LOG_DIR/l8_train.log"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

# Recipe per path-to-125 NEXT_SESSION_PROMPT §3 L8 and closeout § Branch 3 fix (e).
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
  --multi-step-aux-decay 0.3
  --gate-init 0.1
)

# Background launch: nohup, write start metadata to l8_status.json,
# tail the pid to the same file so the next attended session can
# poll without blocking. Foreground launch streams to stdout AND tee
# into the log.
if [ "${1:-}" = "--nohup" ]; then
  echo "[eagle4_v4_fromscratch] launching nohup background training; log=$LOG_FILE"
  echo "[eagle4_v4_fromscratch] glob expansion:"
  ls training_data/c2_hidden/eagle4_v0/shard_*.parquet | head -3
  echo "  ... ($(ls training_data/c2_hidden/eagle4_v0/shard_*.parquet | wc -l | tr -d ' ') shards total)"
  nohup nice -n 19 taskpolicy -b "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
  TRAIN_PID=$!
  cat >"$LOG_DIR/l8_status.json" <<JSON
{
  "pid": $TRAIN_PID,
  "started_at": "$(date -u +%FT%TZ)",
  "ckpt_dir": "$CKPT_DIR",
  "log_file": "$LOG_FILE",
  "command": $(printf '%s\n' "${CMD[@]}" | python3 -c "import json,sys; print(json.dumps([l.rstrip() for l in sys.stdin]))"),
  "recipe": "path-to-125 L8 from-scratch retrain with --gate-init 0.1",
  "expected_wall_clock_hours_contended": "10-15",
  "expected_wall_clock_hours_clean": "3-4",
  "stop_when": "epochs=2 complete OR you Ctrl-C after the first chain_accept readout"
}
JSON
  echo "[eagle4_v4_fromscratch] pid=$TRAIN_PID written to $LOG_DIR/l8_status.json"
  echo "[eagle4_v4_fromscratch] tail -f $LOG_FILE to watch progress."
  echo "[eagle4_v4_fromscratch] kill $TRAIN_PID to stop."
  exit 0
fi

echo "[eagle4_v4_fromscratch] foreground run (tee → $LOG_FILE)"
nice -n 19 taskpolicy -b "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

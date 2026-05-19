#!/usr/bin/env bash
# path-to-125 smoke pipeline (k=2, heldout shard, 5 epochs).
# Tolerant of Claude.app being open — small training run to validate
# the --chain-h-high patch directionally without burning hours of
# contended GPU. Auto-runs tau_eval after training.

set -euo pipefail

cd /Users/scammermike/Downloads/dismantle/.claude/worktrees/dreamy-golick-d54ff8
mkdir -p reports/path_to_90/_pipeline

VENV=/Users/scammermike/Downloads/dismantle/eagle4/.venv/bin/python3
SHARD=eagle4/data/v2lite_3layer_heldout/shard_00000.parquet
CKPT=eagle4/checkpoints/eagle4_v4_smoke
TRAIN_LOG=reports/path_to_90/_pipeline/train.log
TAU_LOG=reports/path_to_90/_pipeline/tau_eval.log

echo "[$(date +%H:%M:%S)] starting training (k=2, 5 epochs, heldout shard)"
"$VENV" eagle4/eagle4.py train \
  --parquet "$SHARD" \
  --frozen eagle4/v2lite_frozen.npz \
  --ckpt-dir "$CKPT" \
  --resume eagle4/checkpoints/eagle4_v3/best.npz \
  --epochs 5 \
  --multi-step-k 2 \
  --multi-step-decay 0.7 \
  --chain-h-high \
  --target-warmup-steps 100 \
  > "$TRAIN_LOG" 2>&1
echo "[$(date +%H:%M:%S)] training done"

echo "[$(date +%H:%M:%S)] running tau_eval"
"$VENV" eagle4/tau_eval.py eval \
  --ckpt "$CKPT/latest.npz" \
  --frozen eagle4/v2lite_frozen.npz \
  --parquet "$SHARD" \
  --depth 4 \
  > "$TAU_LOG" 2>&1
echo "[$(date +%H:%M:%S)] tau_eval done"

echo "[$(date +%H:%M:%S)] pipeline complete"
if command -v osascript >/dev/null 2>&1; then
  osascript -e 'display notification "smoke train+eval complete" with title "dismantle"' || true
fi

#!/usr/bin/env bash
# path-to-125 chain-train iter5 — 5 shards × k=4 with aux-decay 0.3.
# Goal: break the gate-plateau by tapering MSE aux through chain steps,
# letting residual_gate grow so the chain block actually contributes.

set -euo pipefail

cd /Users/scammermike/Downloads/dismantle/.claude/worktrees/dreamy-golick-d54ff8
mkdir -p reports/path_to_90/_pipeline

VENV=/Users/scammermike/Downloads/dismantle/eagle4/.venv/bin/python3
CKPT=eagle4/checkpoints/eagle4_v4_chain_iter5
HELDOUT=eagle4/data/v2lite_3layer_heldout/shard_00000.parquet
TRAIN_LOG=reports/path_to_90/_pipeline/iter5_train.log
TAU_LOG=reports/path_to_90/_pipeline/iter5_tau_eval.log
SMOKE_V3_LOG=reports/path_to_90/_pipeline/iter5_smoke_v3.log
SMOKE_V4_LOG=reports/path_to_90/_pipeline/iter5_smoke_v4.log

echo "[$(date +%H:%M:%S)] iter5 starting (5 shards × k=4 × aux-decay 0.3)"
"$VENV" eagle4/eagle4.py train \
  --parquet \
    training_data/c2_hidden/eagle4_v0/shard_00000.parquet \
    training_data/c2_hidden/eagle4_v0/shard_00001.parquet \
    training_data/c2_hidden/eagle4_v0/shard_00002.parquet \
    training_data/c2_hidden/eagle4_v0/shard_00003.parquet \
    training_data/c2_hidden/eagle4_v0/shard_00004.parquet \
  --frozen eagle4/v2lite_frozen.npz \
  --ckpt-dir "$CKPT" \
  --resume eagle4/checkpoints/eagle4_v3/best.npz \
  --epochs 1 \
  --multi-step-k 4 \
  --multi-step-decay 0.7 \
  --multi-step-aux-decay 0.3 \
  --chain-h-high \
  --target-warmup-steps 50 \
  > "$TRAIN_LOG" 2>&1
echo "[$(date +%H:%M:%S)] iter5 training done"

echo "[$(date +%H:%M:%S)] tau_eval"
"$VENV" eagle4/tau_eval.py eval \
  --ckpt "$CKPT/latest.npz" \
  --frozen eagle4/v2lite_frozen.npz \
  --parquet "$HELDOUT" \
  --depth 4 \
  > "$TAU_LOG" 2>&1
echo "[$(date +%H:%M:%S)] tau_eval done"

cp profiles/deepseek-v2-lite-q4.m3pro18.json /tmp/_iter5_profile_backup.json
"$VENV" -c "
import json
p = json.load(open('profiles/deepseek-v2-lite-q4.m3pro18.json'))
p['selected']['verify_kernels'] = 'parallel-k'
json.dump(p, open('profiles/deepseek-v2-lite-q4.m3pro18.json','w'), indent=2)
"

echo "[$(date +%H:%M:%S)] chain-decode smoke v3"
EAGLE4_CHAIN_K=4 DISMANTLE_SPEC_LOG=1 nice -n 19 ./target/release/dismantle generate \
  --weights models/deepseek-v2-lite-q4.gguf \
  --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
  --prompt "The capital of France is" \
  --max-new-tokens 32 --temperature 0 \
  --speculate eagle4 \
  --draft-head eagle4/checkpoints/eagle4_v3/best.npz \
  --eagle4-frozen eagle4/v2lite_frozen.npz \
  > "$SMOKE_V3_LOG" 2>&1 || true

echo "[$(date +%H:%M:%S)] chain-decode smoke v4_chain_iter5"
EAGLE4_CHAIN_K=4 DISMANTLE_SPEC_LOG=1 nice -n 19 ./target/release/dismantle generate \
  --weights models/deepseek-v2-lite-q4.gguf \
  --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
  --prompt "The capital of France is" \
  --max-new-tokens 32 --temperature 0 \
  --speculate eagle4 \
  --draft-head "$CKPT/latest.npz" \
  --eagle4-frozen eagle4/v2lite_frozen.npz \
  > "$SMOKE_V4_LOG" 2>&1 || true

cp /tmp/_iter5_profile_backup.json profiles/deepseek-v2-lite-q4.m3pro18.json

echo "[$(date +%H:%M:%S)] iter5 complete"
if command -v osascript >/dev/null 2>&1; then
  osascript -e 'display notification "iter5 chain train + eval complete" with title "dismantle"' || true
fi

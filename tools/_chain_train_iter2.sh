#!/usr/bin/env bash
# path-to-125 chain-train iteration 2 — 10 shards × 1 epoch × k=2.
# Clears the alpha warmup (target_warmup_steps=200) and sees proper
# convergence. Auto-runs tau_eval + chain-decode A/B smoke after.

set -euo pipefail

cd /Users/scammermike/Downloads/dismantle/.claude/worktrees/dreamy-golick-d54ff8
mkdir -p reports/path_to_90/_pipeline

VENV=/Users/scammermike/Downloads/dismantle/eagle4/.venv/bin/python3
CKPT=eagle4/checkpoints/eagle4_v4_chain_iter2
HELDOUT=eagle4/data/v2lite_3layer_heldout/shard_00000.parquet
TRAIN_LOG=reports/path_to_90/_pipeline/iter2_train.log
TAU_LOG=reports/path_to_90/_pipeline/iter2_tau_eval.log
SMOKE_V3_LOG=reports/path_to_90/_pipeline/iter2_smoke_v3.log
SMOKE_V4_LOG=reports/path_to_90/_pipeline/iter2_smoke_v4.log

echo "[$(date +%H:%M:%S)] iter2 starting (10 shards × 1 epoch × k=2)"
"$VENV" eagle4/eagle4.py train \
  --parquet \
    training_data/c2_hidden/eagle4_v0/shard_00000.parquet \
    training_data/c2_hidden/eagle4_v0/shard_00001.parquet \
    training_data/c2_hidden/eagle4_v0/shard_00002.parquet \
    training_data/c2_hidden/eagle4_v0/shard_00003.parquet \
    training_data/c2_hidden/eagle4_v0/shard_00004.parquet \
    training_data/c2_hidden/eagle4_v0/shard_00005.parquet \
    training_data/c2_hidden/eagle4_v0/shard_00006.parquet \
    training_data/c2_hidden/eagle4_v0/shard_00007.parquet \
    training_data/c2_hidden/eagle4_v0/shard_00008.parquet \
    training_data/c2_hidden/eagle4_v0/shard_00009.parquet \
  --frozen eagle4/v2lite_frozen.npz \
  --ckpt-dir "$CKPT" \
  --resume eagle4/checkpoints/eagle4_v3/best.npz \
  --epochs 1 \
  --multi-step-k 2 \
  --multi-step-decay 0.7 \
  --chain-h-high \
  --target-warmup-steps 200 \
  > "$TRAIN_LOG" 2>&1
echo "[$(date +%H:%M:%S)] iter2 training done"

echo "[$(date +%H:%M:%S)] tau_eval"
"$VENV" eagle4/tau_eval.py eval \
  --ckpt "$CKPT/latest.npz" \
  --frozen eagle4/v2lite_frozen.npz \
  --parquet "$HELDOUT" \
  --depth 4 \
  > "$TAU_LOG" 2>&1
echo "[$(date +%H:%M:%S)] tau_eval done"

# Chain-decode A/B smoke vs v3 baseline (in parallel-k profile).
# Temporarily flip profile, restore at end.
cp profiles/deepseek-v2-lite-q4.m3pro18.json /tmp/_iter2_profile_backup.json
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

echo "[$(date +%H:%M:%S)] chain-decode smoke v4_chain_iter2"
EAGLE4_CHAIN_K=4 DISMANTLE_SPEC_LOG=1 nice -n 19 ./target/release/dismantle generate \
  --weights models/deepseek-v2-lite-q4.gguf \
  --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
  --prompt "The capital of France is" \
  --max-new-tokens 32 --temperature 0 \
  --speculate eagle4 \
  --draft-head "$CKPT/latest.npz" \
  --eagle4-frozen eagle4/v2lite_frozen.npz \
  > "$SMOKE_V4_LOG" 2>&1 || true

cp /tmp/_iter2_profile_backup.json profiles/deepseek-v2-lite-q4.m3pro18.json

echo "[$(date +%H:%M:%S)] iter2 complete"
if command -v osascript >/dev/null 2>&1; then
  osascript -e 'display notification "iter2 chain train + eval complete" with title "dismantle"' || true
fi

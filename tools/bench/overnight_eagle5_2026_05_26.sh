#!/usr/bin/env bash
# tools/bench/overnight_eagle5_2026_05_26.sh
#
# Overnight chain: corpus rebuild → Eagle5 v2 train → τ-at-depth eval →
# paired bench. Optimised for ~2.5-3 hr wall on M3 Pro 18 GB.
#
# Launch:
#   nohup tools/bench/overnight_eagle5_2026_05_26.sh \
#     > reports/overnight_eagle5_2026_05_26.log 2>&1 & disown
#
# Resumable: build_corpus.py skips existing shards. eagle5_train.py
# accepts --resume; if you need to restart, point it at the latest
# checkpoint and re-launch.

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG="reports/overnight_eagle5_2026_05_26.log"
mkdir -p reports
exec > >(tee -a "$LOG") 2>&1
echo "[overnight] start $(date -u +%FT%TZ)"

step() {
  local name="$1"; shift
  echo "[overnight] ▶ $name  $(date -u +%FT%TZ)"
  "$@"
  local rc=$?
  echo "[overnight] ◀ $name rc=$rc  $(date -u +%FT%TZ)"
  if [[ $rc -ne 0 ]]; then
    echo "[overnight] HALT — $name failed"
    exit $rc
  fi
}

# (1) Corpus rebuild  ~30-45 min (batch=8 vs default batch=1)
step corpus \
  nice -n 19 taskpolicy -b python3 tools/training/build_corpus.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --dataset HuggingFaceH4/ultrachat_200k \
    --max-sequences 3000 \
    --batch-size 8 \
    --max-tokens-per-seq 2048 \
    --shard-size 32 \
    --capture all \
    --out artifacts/calibration/v2_lite_corpus

# (2) Eagle5 v2 train  ~1.5 hr (3 epochs not 5; batch 24 not 16)
step train \
  nice -n 19 taskpolicy -b python3 tools/training/eagle5_train.py \
    --corpus-dir artifacts/calibration/v2_lite_corpus \
    --frozen     eagle4/v2lite_frozen.npz \
    --ckpt-dir   checkpoints/eagle5_v2 \
    --epochs 3 --batch-size 24 --seq-len 16 --lr 3e-4 \
    --sparsity-head proxy --seed 0

# (3) τ-at-depth eval (K=1..8 acceptance)  ~10 min
step tau_eval \
  bash -c "nice -n 19 taskpolicy -b python3 tools/training/eagle5_tau_eval.py \
    --ckpt    checkpoints/eagle5_v2/head_final.safetensors \
    --frozen  eagle4/v2lite_frozen.npz \
    --corpus  artifacts/calibration/v2_lite_corpus \
    > reports/eagle5_tau_2026_05_26.txt 2>&1"

# (4) cargo build + Eagle5 paired bench (DeepSeek-V2-Lite, K=2,4,8)  ~20 min
step build_release nice -n 19 cargo build --release -p dismantle
step paired_bench \
  bash -c "EAGLE5_HEAD=checkpoints/eagle5_v2/head_final.safetensors \
    TOKENS=64 TRIALS=5 \
    nice -n 19 taskpolicy -b ./tools/bench/eagle5_paired_bench.sh \
    > reports/eagle5_paired_2026_05_26.txt 2>&1"

# (5) W4A8 per-channel calibration dump (Qwen-3B, separate workstream)
#     Independent of Eagle5 — runs after to avoid Metal contention.
#     Produces reports/w4a8_per_channel_calibration_2026_05_26.json
if [[ -f tests/calibration/w4a8_per_channel_calibrate.rs ]] \
   || cargo test -p dismantle-core --tests --list 2>/dev/null \
        | grep -q w4a8_per_channel_calibrate; then
  step w4a8_calibration \
    bash -c "nice -n 19 cargo test --release -p dismantle-core \
      --test w4a8_per_channel_calibrate -- --nocapture --ignored \
      > reports/w4a8_per_channel_calibration_2026_05_26.txt 2>&1"
else
  echo "[overnight] skip w4a8_calibration — test not yet present"
fi

echo "[overnight] ALL DONE $(date -u +%FT%TZ)"

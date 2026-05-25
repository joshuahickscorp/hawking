#!/usr/bin/env bash
# tools/bench/overnight_eagle5_2026_05_26.sh — MEMORY-SAFE RECOVERY CONFIG
#
# Overnight chain (~3.5-4 hr wall on M3 Pro 18 GB):
#   1. Corpus rebuild       3000 seqs, batch=4, max_tok=1024   ~1.5-2 hr
#   2. Eagle5 v2 train      3 epochs, batch=24                  ~1 hr
#   3. τ-at-depth eval      K=1..8 acceptance                   ~10 min
#   4. build release dismantle                                  ~1 min
#   5. Eagle5 paired bench  TOKENS=64, n=10 trials              ~15 min
#   6. Eagle5 paired bench  TOKENS=256, n=10 trials             ~30 min
#   7. W4A8 LM_HEAD calibration on Qwen-3B                      ~1 min
#   8. Lookahead n-gram parity sweep on Qwen-3B                 ~10 min
#
# History: The MAXED 8h config (batch=8, max-tokens=2048, 10000 seqs)
# OOMed at the first batch (18.15 GB resident + 6.25 GB transient >
# 22.64 GB MPS ceiling) after 1h 15min of model load. Recovery drops
# batch=8→4 and max-tokens=2048→1024 (~1.6 GB transient, well clear
# of the ceiling); reverts seqs and epochs to the SAFE-config values
# so the chain completes inside the remaining budget.
#
# Launch:
#   nohup tools/bench/overnight_eagle5_2026_05_26.sh \
#     > reports/overnight_eagle5_2026_05_26.log 2>&1 & disown
#
# Resumable: build_corpus.py skips existing shards (--skip-existing on
# by default). eagle5_train.py accepts --resume; on a restart, comment
# out completed steps or point --resume at the latest checkpoint.

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG="reports/overnight_eagle5_2026_05_26.log"
mkdir -p reports
exec > >(tee -a "$LOG") 2>&1
echo "[overnight] start $(date -u +%FT%TZ)  CONFIG=MEMORY_SAFE_RECOVERY"

# Pin python3 to the python.org 3.12 framework where pip installed the
# deps (torch, transformers, datasets, pyarrow, mlx, accelerate, ...).
# Without this, on macOS with Homebrew installed `python3` may resolve
# to Homebrew Python 3.14 which has none of those packages and the
# corpus step halts with "missing python deps: torch, transformers, ...".
PYBIN_PINNED="/Library/Frameworks/Python.framework/Versions/3.12/bin"
if [[ -x "$PYBIN_PINNED/python3" ]]; then
  export PATH="$PYBIN_PINNED:$PATH"
  echo "[overnight] using python3 = $PYBIN_PINNED/python3"
else
  echo "[overnight] WARN — pinned python at $PYBIN_PINNED/python3 missing; falling back to PATH python3 = $(command -v python3)"
fi
python3 --version 2>&1 | sed 's/^/[overnight] /'

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

# Soft step — failure is logged but does NOT halt the chain. Used for
# the ride-along quality checks at the end, so a flaky lookahead test
# doesn't lose us the Eagle5 head.
soft_step() {
  local name="$1"; shift
  echo "[overnight] ▶ $name (soft)  $(date -u +%FT%TZ)"
  "$@"
  local rc=$?
  echo "[overnight] ◀ $name rc=$rc  $(date -u +%FT%TZ)"
  if [[ $rc -ne 0 ]]; then
    echo "[overnight] WARN — $name failed (soft, continuing)"
  fi
}

# (1) Corpus rebuild  ~1.5-2 hr (3000 seqs, batch=4, max_tok=1024)
#     Resumable: --skip-existing is on by default.
#     Memory: per-batch transient ≈ (1024/2048)·(4/8)·6.25 GB = 1.6 GB;
#     resident 18.15 GB + transient 1.6 GB = 19.75 GB, ~3 GB clear of
#     the 22.64 GB MPS ceiling.
step corpus \
  nice -n 19 taskpolicy -b python3 tools/training/build_corpus.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --dataset HuggingFaceH4/ultrachat_200k \
    --max-sequences 3000 \
    --batch-size 4 \
    --max-tokens-per-seq 1024 \
    --shard-size 32 \
    --capture all \
    --out artifacts/calibration/v2_lite_corpus

# (2) Eagle5 v2 train  ~2-2.5 hr (5 epochs over 20k Colab corpus, batch=24)
#     Bumped from 3→5 epochs because the Colab-built corpus has ~7× more
#     samples than the killed laptop attempt. More passes are useful with
#     bigger data; loss plateau in path_to_50_complete.md was at epoch 2
#     for a 3k corpus, expected ~epoch 4 for 20k.
step train \
  nice -n 19 taskpolicy -b python3 tools/training/eagle5_train.py \
    --corpus-dir artifacts/calibration/v2_lite_corpus \
    --frozen     eagle4/v2lite_frozen.npz \
    --ckpt-dir   checkpoints/eagle5_v2 \
    --epochs 5 --batch-size 24 --seq-len 16 --lr 3e-4 \
    --sparsity-head proxy --seed 0

# (3) τ-at-depth eval (K=1..8 acceptance)  ~15 min
step tau_eval \
  bash -c "nice -n 19 taskpolicy -b python3 tools/training/eagle5_tau_eval.py \
    --ckpt    checkpoints/eagle5_v2/head_final.safetensors \
    --frozen  eagle4/v2lite_frozen.npz \
    --corpus  artifacts/calibration/v2_lite_corpus \
    > reports/eagle5_tau_2026_05_26.txt 2>&1"

# (4) build release dismantle for the paired bench
step build_release nice -n 19 cargo build --release -p dismantle

# (5) Eagle5 paired bench — short-output regime (64 tok, n=10)
#     Catches per-token overhead amortization; should give tightest
#     CI on the K=2/4/8 dec_tps delta.
step paired_bench_64 \
  bash -c "EAGLE5_HEAD=checkpoints/eagle5_v2/head_final.safetensors \
    TOKENS=64 TRIALS=10 \
    nice -n 19 taskpolicy -b ./tools/bench/eagle5_paired_bench.sh \
    > reports/eagle5_paired_64tok_2026_05_26.txt 2>&1"

# (6) Eagle5 paired bench — longer-output regime (256 tok, n=10)
#     Tests whether acceptance rate holds (or degrades) on extended
#     generation. Eagle5's draft accuracy can drift over long contexts;
#     this surfaces it.
step paired_bench_256 \
  bash -c "EAGLE5_HEAD=checkpoints/eagle5_v2/head_final.safetensors \
    TOKENS=256 TRIALS=10 \
    nice -n 19 taskpolicy -b ./tools/bench/eagle5_paired_bench.sh \
    > reports/eagle5_paired_256tok_2026_05_26.txt 2>&1"

# ── Ride-along quality checks (soft — don't halt on flake) ───────────
#    Each runs on the GPU after Eagle5 releases it; independent of
#    DeepSeek so no contention with anything earlier.

# (7) W4A8 per-channel LM_HEAD calibration dump on Qwen-3B
#     ~16 sec. Refreshes reports/w4a8_lmhead_calibration_2026_05_26.json
#     for tomorrow's attended wire-up session.
soft_step w4a8_calibration \
  bash -c "nice -n 19 cargo test --release -p dismantle-core \
    --test w4a8_per_channel_calibrate -- --nocapture --ignored \
    > reports/w4a8_per_channel_calibration_2026_05_26.txt 2>&1"

# (8) Lookahead n-gram parity sweep on Qwen-3B
#     ~10 min. Confirms the parity-bug fix (memory/lookahead_resurrected
#     _2026_05_26.md) isn't prompt-dependent — currently only validated
#     on 2 prompts. Test runs DISMANTLE_LOOKAHEAD=N greedy vs baseline.
soft_step lookahead_parity \
  bash -c "nice -n 19 cargo test --release -p dismantle-core \
    --test qwen_lookahead_parity -- --nocapture --ignored \
    > reports/qwen_lookahead_parity_2026_05_26.txt 2>&1"

echo "[overnight] ALL DONE $(date -u +%FT%TZ)"

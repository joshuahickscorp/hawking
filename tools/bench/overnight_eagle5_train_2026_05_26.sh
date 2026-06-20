#!/usr/bin/env bash
# tools/bench/overnight_eagle5_train_2026_05_26.sh
#
# Post-corpus overnight chain. Assumes corpus is already on disk at
# artifacts/calibration/v2_lite_corpus/ (built on Colab, downloaded here).
# Runs train → τ-eval → bench → ride-alongs.
#
# Tuned for the actual corpus we got from Colab: 16,208 sequences (1013
# shards × 16 seqs/shard) — between SAFE-config 3k and MAXED 20k. At
# this size 3 epochs sits past the loss plateau without overfitting.
#
# Launch:
#   nohup tools/bench/overnight_eagle5_train_2026_05_26.sh \
#     > reports/overnight_eagle5_train_2026_05_26.log 2>&1 & disown
#
# Total wall: ~2-3 hr on M3 Pro 18 GB.
#   train             ~1-2 hr  (MLX, head is tiny so it's quick)
#   τ-eval            ~10 min
#   build release      ~1 min
#   paired bench 64    ~15 min
#   paired bench 256   ~30 min
#   W4A8 calibration   ~1 min (soft)
#   lookahead parity   ~10 min (soft)

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG="reports/overnight_eagle5_train_2026_05_26.log"
mkdir -p reports
exec > >(tee -a "$LOG") 2>&1
echo "[overnight] start $(date -u +%FT%TZ)  CONFIG=POST_COLAB_CORPUS"

# Pin python3 to the python.org 3.12 framework where pip installed mlx etc.
PYBIN="/Library/Frameworks/Python.framework/Versions/3.12/bin"
if [[ -x "$PYBIN/python3" ]]; then
  export PATH="$PYBIN:$PATH"
fi
python3 --version 2>&1 | sed 's/^/[overnight] /'

# Sanity: confirm the corpus is present
CORPUS=artifacts/calibration/v2_lite_corpus
shard_count=$(ls -1 "$CORPUS"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
echo "[overnight] corpus: $shard_count shards at $CORPUS"
if [[ "$shard_count" -lt 100 ]]; then
  echo "[overnight] HALT — too few shards. Did the Colab download land here?"
  exit 1
fi

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

soft_step() {
  local name="$1"; shift
  echo "[overnight] ▶ $name (soft)  $(date -u +%FT%TZ)"
  "$@"
  local rc=$?
  echo "[overnight] ◀ $name rc=$rc  $(date -u +%FT%TZ)"
  if [[ $rc -ne 0 ]]; then
    echo "[overnight] WARN — $name failed (continuing)"
  fi
}

# (1) Train Eagle5 v2 head  ~1-2 hr
#     3 epochs is sweet spot for 16k seqs (plateau-past, no overfit).
step train \
  nice -n 19 taskpolicy -b python3 tools/training/eagle5_train.py \
    --corpus-dir "$CORPUS" \
    --frozen     eagle4/v2lite_frozen.npz \
    --ckpt-dir   checkpoints/eagle5_v2 \
    --epochs 3 --batch-size 24 --seq-len 16 --lr 3e-4 \
    --sparsity-head proxy --seed 0

# (2) τ-at-depth eval (K=1..8 acceptance)  ~10 min
step tau_eval \
  bash -c "nice -n 19 taskpolicy -b python3 tools/training/eagle5_tau_eval.py \
    --ckpt    checkpoints/eagle5_v2/head_final.safetensors \
    --frozen  eagle4/v2lite_frozen.npz \
    --corpus  $CORPUS \
    > reports/eagle5_tau_2026_05_26.txt 2>&1"

# (3) build release dismantle
step build_release nice -n 19 cargo build --release -p dismantle

# (4) Paired bench — short-output regime (64 tok, n=10)
step paired_bench_64 \
  bash -c "EAGLE5_HEAD=checkpoints/eagle5_v2/head_final.safetensors \
    TOKENS=64 TRIALS=10 \
    nice -n 19 taskpolicy -b ./tools/bench/eagle5_paired_bench.sh \
    > reports/eagle5_paired_64tok_2026_05_26.txt 2>&1"

# (5) Paired bench — long-output regime (256 tok, n=10)
step paired_bench_256 \
  bash -c "EAGLE5_HEAD=checkpoints/eagle5_v2/head_final.safetensors \
    TOKENS=256 TRIALS=10 \
    nice -n 19 taskpolicy -b ./tools/bench/eagle5_paired_bench.sh \
    > reports/eagle5_paired_256tok_2026_05_26.txt 2>&1"

# ── Ride-along quality checks (soft) ─────────────────────────────────

# (6) W4A8 LM_HEAD per-channel calibration on Qwen-3B
soft_step w4a8_calibration \
  bash -c "nice -n 19 cargo test --release -p hawking-core \
    --test w4a8_per_channel_calibrate -- --nocapture --ignored \
    > reports/w4a8_per_channel_calibration_2026_05_26.txt 2>&1"

# (7) Lookahead n-gram parity sweep on Qwen-3B
soft_step lookahead_parity \
  bash -c "nice -n 19 cargo test --release -p hawking-core \
    --test qwen_lookahead_parity -- --nocapture --ignored \
    > reports/qwen_lookahead_parity_2026_05_26.txt 2>&1"

echo "[overnight] ALL DONE $(date -u +%FT%TZ)"
echo
echo "Artifacts:"
echo "  trained head: checkpoints/eagle5_v2/head_final.safetensors"
echo "  τ-at-K eval:  reports/eagle5_tau_2026_05_26.txt"
echo "  bench 64:     reports/eagle5_paired_64tok_2026_05_26.txt"
echo "  bench 256:    reports/eagle5_paired_256tok_2026_05_26.txt"
echo "  W4A8 cal:     reports/w4a8_per_channel_calibration_2026_05_26.txt"
echo "  lookahead:    reports/qwen_lookahead_parity_2026_05_26.txt"

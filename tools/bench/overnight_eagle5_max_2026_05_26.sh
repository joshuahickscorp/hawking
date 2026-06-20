#!/usr/bin/env bash
# tools/bench/overnight_eagle5_max_2026_05_26.sh — MAXED post-corpus chain
#
# Replaces the single-config overnight with a 2-variant comparison:
#   A. sparsity-head=proxy  (default; auxiliary loss)
#   B. sparsity-head=off    (no auxiliary loss, simpler head)
#
# Both train on the same Colab-built 16k-seq corpus, get τ-eval'd at
# K=1..8, then we pick the winner (highest K=4 acceptance) and run
# paired bench at 64 + 256 tokens. Ride-alongs at the tail.
#
# Total wall ~5-6 hr on M3 Pro. Fills the 5-6 hr downtime window.
#
# Launch:
#   nohup tools/bench/overnight_eagle5_max_2026_05_26.sh \
#     > reports/overnight_eagle5_max_2026_05_26.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG="reports/overnight_eagle5_max_2026_05_26.log"
mkdir -p reports
exec > >(tee -a "$LOG") 2>&1
echo "[overnight] start $(date -u +%FT%TZ)  CONFIG=POST_COLAB_MAX_2VAR"

PYBIN="/Library/Frameworks/Python.framework/Versions/3.12/bin"
if [[ -x "$PYBIN/python3" ]]; then export PATH="$PYBIN:$PATH"; fi
python3 --version 2>&1 | sed 's/^/[overnight] /'

CORPUS=artifacts/calibration/v2_lite_corpus
shard_count=$(ls -1 "$CORPUS"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
echo "[overnight] corpus: $shard_count shards at $CORPUS"
[[ "$shard_count" -lt 100 ]] && { echo "[overnight] HALT — too few shards"; exit 1; }

step() {
  local name="$1"; shift
  echo "[overnight] ▶ $name  $(date -u +%FT%TZ)"
  "$@"
  local rc=$?
  echo "[overnight] ◀ $name rc=$rc  $(date -u +%FT%TZ)"
  if [[ $rc -ne 0 ]]; then echo "[overnight] HALT — $name failed"; exit $rc; fi
}

soft_step() {
  local name="$1"; shift
  echo "[overnight] ▶ $name (soft)  $(date -u +%FT%TZ)"
  "$@"
  local rc=$?
  echo "[overnight] ◀ $name rc=$rc  $(date -u +%FT%TZ)"
  [[ $rc -ne 0 ]] && echo "[overnight] WARN — $name failed (continuing)"
}

# (1A) Train variant A — sparsity-head=proxy (default)
step train_proxy \
  nice -n 19 taskpolicy -b python3 tools/training/eagle5_train.py \
    --corpus-dir "$CORPUS" \
    --frozen     eagle4/v2lite_frozen.npz \
    --ckpt-dir   checkpoints/eagle5_v2_proxy \
    --epochs 5 --batch-size 24 --seq-len 16 --lr 3e-4 \
    --max-rows 5000 --max-row-tokens 128 \
    --sparsity-head proxy --seed 0

# (1B) Train variant B — sparsity-head=off (no aux loss)
step train_off \
  nice -n 19 taskpolicy -b python3 tools/training/eagle5_train.py \
    --corpus-dir "$CORPUS" \
    --frozen     eagle4/v2lite_frozen.npz \
    --ckpt-dir   checkpoints/eagle5_v2_off \
    --epochs 5 --batch-size 24 --seq-len 16 --lr 3e-4 \
    --max-rows 5000 --max-row-tokens 128 \
    --sparsity-head off --seed 0

# (2A) τ-eval variant A
step tau_eval_proxy \
  bash -c "nice -n 19 taskpolicy -b python3 tools/training/eagle5_tau_eval.py \
    --ckpt    checkpoints/eagle5_v2_proxy/head_final.safetensors \
    --frozen  eagle4/v2lite_frozen.npz \
    --corpus  $CORPUS \
    > reports/eagle5_tau_proxy_2026_05_26.txt 2>&1"

# (2B) τ-eval variant B
step tau_eval_off \
  bash -c "nice -n 19 taskpolicy -b python3 tools/training/eagle5_tau_eval.py \
    --ckpt    checkpoints/eagle5_v2_off/head_final.safetensors \
    --frozen  eagle4/v2lite_frozen.npz \
    --corpus  $CORPUS \
    > reports/eagle5_tau_off_2026_05_26.txt 2>&1"

# (3) Pick winner by K=4 acceptance (cheap inline awk-grep)
echo "[overnight] ▶ pick_winner  $(date -u +%FT%TZ)"
PROXY_K4=$(grep -E '^K=4|K\s*=\s*4' reports/eagle5_tau_proxy_2026_05_26.txt | head -1 | grep -oE '[0-9]+\.[0-9]+%' | head -1 | tr -d '%')
OFF_K4=$(grep -E '^K=4|K\s*=\s*4' reports/eagle5_tau_off_2026_05_26.txt | head -1 | grep -oE '[0-9]+\.[0-9]+%' | head -1 | tr -d '%')
echo "[overnight]   variant proxy:  K=4 accept = ${PROXY_K4:-NA}%"
echo "[overnight]   variant off:    K=4 accept = ${OFF_K4:-NA}%"

# Default to proxy if grep failed; otherwise pick higher K=4
WINNER=proxy
if [[ -n "${OFF_K4:-}" ]] && [[ -n "${PROXY_K4:-}" ]]; then
  awk_cmp=$(awk -v p="$PROXY_K4" -v o="$OFF_K4" 'BEGIN { print (o>p) ? "off" : "proxy" }')
  WINNER="$awk_cmp"
fi
WINNER_CKPT="checkpoints/eagle5_v2_${WINNER}/head_final.safetensors"
echo "[overnight]   WINNER = $WINNER  (ckpt=$WINNER_CKPT)"

# Symlink for legacy compat
ln -sf "eagle5_v2_${WINNER}" checkpoints/eagle5_v2_winner
echo "[overnight] ◀ pick_winner rc=0  $(date -u +%FT%TZ)"

# (4) build release dismantle for paired bench
step build_release nice -n 19 cargo build --release -p dismantle

# (5) Paired bench — winner @ 64 tokens
step paired_bench_64 \
  bash -c "EAGLE5_HEAD=$WINNER_CKPT TOKENS=64 TRIALS=10 \
    nice -n 19 taskpolicy -b ./tools/bench/eagle5_paired_bench.sh \
    > reports/eagle5_paired_64tok_2026_05_26.txt 2>&1"

# (6) Paired bench — winner @ 256 tokens
step paired_bench_256 \
  bash -c "EAGLE5_HEAD=$WINNER_CKPT TOKENS=256 TRIALS=10 \
    nice -n 19 taskpolicy -b ./tools/bench/eagle5_paired_bench.sh \
    > reports/eagle5_paired_256tok_2026_05_26.txt 2>&1"

# ── Ride-along (soft) ────────────────────────────────────────────────
soft_step w4a8_calibration \
  bash -c "nice -n 19 cargo test --release -p hawking-core \
    --test w4a8_per_channel_calibrate -- --nocapture --ignored \
    > reports/w4a8_per_channel_calibration_2026_05_26.txt 2>&1"

soft_step lookahead_parity \
  bash -c "nice -n 19 cargo test --release -p hawking-core \
    --test qwen_lookahead_parity -- --nocapture --ignored \
    > reports/qwen_lookahead_parity_2026_05_26.txt 2>&1"

echo "[overnight] ALL DONE $(date -u +%FT%TZ)"
echo
echo "Summary:"
echo "  WINNER:     $WINNER (K=4 proxy=$PROXY_K4% vs off=$OFF_K4%)"
echo "  Head:       checkpoints/eagle5_v2_${WINNER}/head_final.safetensors"
echo "  τ-eval:     reports/eagle5_tau_{proxy,off}_2026_05_26.txt"
echo "  Bench 64:   reports/eagle5_paired_64tok_2026_05_26.txt"
echo "  Bench 256:  reports/eagle5_paired_256tok_2026_05_26.txt"
echo "  W4A8 cal:   reports/w4a8_per_channel_calibration_2026_05_26.txt"
echo "  Lookahead:  reports/qwen_lookahead_parity_2026_05_26.txt"

#!/usr/bin/env bash
# tools/bench/overnight_eagle5_grid_2026_05_26.sh
#
# 4-variant Eagle5 v2 hyperparameter grid: {sparsity proxy/off} × {lr 3e-4/1e-3}.
# Trains all four heads, τ-evals each, picks the highest K=4 acceptance, and
# paired-benches the winner at 64 + 256 tokens. Ride-alongs at the tail.
#
# Trained 5 epochs each on a 5000-row × 128-token subsample of the Colab corpus.
# Total wall ~4.3 hr on M3 Pro 18 GB. Fits the user's 6 hr downtime budget
# with margin.
#
# Variants:
#   A. proxy_3e4 : sparsity=proxy, lr=3e-4   (baseline — original config)
#   B. proxy_1e3 : sparsity=proxy, lr=1e-3   (faster learning)
#   C. off_3e4   : sparsity=off,   lr=3e-4   (no aux loss)
#   D. off_1e3   : sparsity=off,   lr=1e-3   (combined)
#
# Launch:
#   nohup tools/bench/overnight_eagle5_grid_2026_05_26.sh \
#     > reports/overnight_eagle5_grid_2026_05_26.log 2>&1 & disown

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG="reports/overnight_eagle5_grid_2026_05_26.log"
mkdir -p reports
exec > >(tee -a "$LOG") 2>&1
echo "[overnight] start $(date -u +%FT%TZ)  CONFIG=POST_COLAB_GRID_4VAR"

PYBIN="/Library/Frameworks/Python.framework/Versions/3.12/bin"
[[ -x "$PYBIN/python3" ]] && export PATH="$PYBIN:$PATH"
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

# Train one variant: $1=name $2=sparsity $3=lr
train_variant() {
  local name="$1" spar="$2" lr="$3"
  step "train_$name" \
    nice -n 19 taskpolicy -b python3 tools/training/eagle5_train.py \
      --corpus-dir "$CORPUS" \
      --frozen     eagle4/v2lite_frozen.npz \
      --ckpt-dir   "checkpoints/eagle5_v2_$name" \
      --epochs 5 --batch-size 24 --seq-len 16 --lr "$lr" \
      --max-rows 5000 --max-row-tokens 128 \
      --sparsity-head "$spar" --seed 0
}

# τ-eval one variant: $1=name → reports/eagle5_tau_${name}_*.txt
eval_variant() {
  local name="$1"
  step "tau_eval_$name" \
    bash -c "nice -n 19 taskpolicy -b python3 tools/training/eagle5_tau_eval.py \
      --ckpt    checkpoints/eagle5_v2_${name}/head_final.safetensors \
      --frozen  eagle4/v2lite_frozen.npz \
      --corpus  $CORPUS \
      > reports/eagle5_tau_${name}_2026_05_26.txt 2>&1"
}

# (1) Train all 4 variants
train_variant proxy_3e4 proxy 3e-4
train_variant proxy_1e3 proxy 1e-3
train_variant off_3e4   off   3e-4
train_variant off_1e3   off   1e-3

# (2) τ-eval all 4
eval_variant proxy_3e4
eval_variant proxy_1e3
eval_variant off_3e4
eval_variant off_1e3

# (3) Pick winner by K=4 acceptance
echo "[overnight] ▶ pick_winner  $(date -u +%FT%TZ)"
read_k4() {
  grep -E '^K=4|K\s*=\s*4' "reports/eagle5_tau_${1}_2026_05_26.txt" 2>/dev/null \
    | head -1 | grep -oE '[0-9]+\.[0-9]+%' | head -1 | tr -d '%'
}
P3=$(read_k4 proxy_3e4)
P1=$(read_k4 proxy_1e3)
O3=$(read_k4 off_3e4)
O1=$(read_k4 off_1e3)
echo "[overnight]   proxy_3e4 K=4 = ${P3:-NA}%"
echo "[overnight]   proxy_1e3 K=4 = ${P1:-NA}%"
echo "[overnight]   off_3e4   K=4 = ${O3:-NA}%"
echo "[overnight]   off_1e3   K=4 = ${O1:-NA}%"

WINNER=proxy_3e4
BEST=${P3:-0}
for cand in "proxy_1e3:${P1:-0}" "off_3e4:${O3:-0}" "off_1e3:${O1:-0}"; do
  name="${cand%%:*}"; val="${cand##*:}"
  cmp=$(awk -v b="$BEST" -v v="$val" 'BEGIN { print (v>b) ? 1 : 0 }')
  if [[ "$cmp" == "1" ]]; then
    BEST="$val"; WINNER="$name"
  fi
done
WINNER_CKPT="checkpoints/eagle5_v2_${WINNER}/head_final.safetensors"
ln -sf "eagle5_v2_${WINNER}" checkpoints/eagle5_v2_winner
echo "[overnight]   WINNER = $WINNER (K=4 = $BEST%)"
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
  bash -c "nice -n 19 cargo test --release -p dismantle-core \
    --test w4a8_per_channel_calibrate -- --nocapture --ignored \
    > reports/w4a8_per_channel_calibration_2026_05_26.txt 2>&1"

soft_step lookahead_parity \
  bash -c "nice -n 19 cargo test --release -p dismantle-core \
    --test qwen_lookahead_parity -- --nocapture --ignored \
    > reports/qwen_lookahead_parity_2026_05_26.txt 2>&1"

echo "[overnight] ALL DONE $(date -u +%FT%TZ)"
echo
echo "Hyperparameter grid summary:"
printf "  proxy + lr 3e-4   K=4=%s%%\n" "${P3:-NA}"
printf "  proxy + lr 1e-3   K=4=%s%%\n" "${P1:-NA}"
printf "  off   + lr 3e-4   K=4=%s%%\n" "${O3:-NA}"
printf "  off   + lr 1e-3   K=4=%s%%\n" "${O1:-NA}"
echo "  WINNER: $WINNER (K=4 = $BEST%)"
echo
echo "Artifacts:"
echo "  4 trained heads in checkpoints/eagle5_v2_{proxy_3e4,proxy_1e3,off_3e4,off_1e3}/"
echo "  symlink:           checkpoints/eagle5_v2_winner"
echo "  τ-eval reports:    reports/eagle5_tau_{4 names}_2026_05_26.txt"
echo "  bench (winner):    reports/eagle5_paired_{64,256}tok_2026_05_26.txt"
echo "  W4A8 cal:          reports/w4a8_per_channel_calibration_2026_05_26.txt"
echo "  lookahead:         reports/qwen_lookahead_parity_2026_05_26.txt"

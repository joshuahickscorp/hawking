#!/usr/bin/env bash
# tools/bench/overnight_eagle5_grid_resume_2026_05_26.sh
#
# Resume after training finished but tau_eval halted (filename mismatch:
# trainer saved latest.npz, eval expected head_final.safetensors). The 4
# trained heads are intact and symlinked.
#
# Skips: training × 4 (already done, ~3.5 hr)
# Runs: tau_eval × 4 + build + paired bench × 2 + ride-alongs
# Total wall: ~1.5-1.7 hr from launch.

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG="reports/overnight_eagle5_grid_resume_2026_05_26.log"
mkdir -p reports
exec > >(tee -a "$LOG") 2>&1
echo "[overnight] resume start $(date -u +%FT%TZ)"

PYBIN="/Library/Frameworks/Python.framework/Versions/3.12/bin"
[[ -x "$PYBIN/python3" ]] && export PATH="$PYBIN:$PATH"

CORPUS=artifacts/calibration/v2_lite_corpus

# Verify all 4 trained heads are present (via the symlinks we just made)
for v in proxy_3e4 proxy_1e3 off_3e4 off_1e3; do
  ckpt="checkpoints/eagle5_v2_${v}/head_final.safetensors"
  if [[ ! -e "$ckpt" ]]; then
    echo "[overnight] HALT — missing $ckpt"; exit 1
  fi
done
echo "[overnight] all 4 trained heads present ✓"

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

eval_variant() {
  local name="$1"
  step "tau_eval_$name" \
    bash -c "nice -n 19 taskpolicy -b python3 tools/training/eagle5_tau_eval.py \
      --ckpt    checkpoints/eagle5_v2_${name}/head_final.safetensors \
      --frozen  eagle4/v2lite_frozen.npz \
      --corpus  $CORPUS \
      > reports/eagle5_tau_${name}_2026_05_26.txt 2>&1"
}

# τ-eval all 4 (was the halt point)
eval_variant proxy_3e4
eval_variant proxy_1e3
eval_variant off_3e4
eval_variant off_1e3

# Pick winner
echo "[overnight] ▶ pick_winner  $(date -u +%FT%TZ)"
read_k4() {
  grep -E '^K=4|K\s*=\s*4' "reports/eagle5_tau_${1}_2026_05_26.txt" 2>/dev/null \
    | head -1 | grep -oE '[0-9]+\.[0-9]+%' | head -1 | tr -d '%'
}
P3=$(read_k4 proxy_3e4); P1=$(read_k4 proxy_1e3)
O3=$(read_k4 off_3e4);   O1=$(read_k4 off_1e3)
echo "[overnight]   proxy_3e4 K=4 = ${P3:-NA}%"
echo "[overnight]   proxy_1e3 K=4 = ${P1:-NA}%"
echo "[overnight]   off_3e4   K=4 = ${O3:-NA}%"
echo "[overnight]   off_1e3   K=4 = ${O1:-NA}%"

WINNER=proxy_3e4
BEST=${P3:-0}
for cand in "proxy_1e3:${P1:-0}" "off_3e4:${O3:-0}" "off_1e3:${O1:-0}"; do
  name="${cand%%:*}"; val="${cand##*:}"
  cmp=$(awk -v b="$BEST" -v v="$val" 'BEGIN { print (v>b) ? 1 : 0 }')
  if [[ "$cmp" == "1" ]]; then BEST="$val"; WINNER="$name"; fi
done
WINNER_CKPT="checkpoints/eagle5_v2_${WINNER}/head_final.safetensors"
ln -sf "eagle5_v2_${WINNER}" checkpoints/eagle5_v2_winner
echo "[overnight]   WINNER = $WINNER (K=4 = $BEST%)"
echo "[overnight] ◀ pick_winner rc=0  $(date -u +%FT%TZ)"

# Build + paired benches
step build_release nice -n 19 cargo build --release -p dismantle

step paired_bench_64 \
  bash -c "EAGLE5_HEAD=$WINNER_CKPT TOKENS=64 TRIALS=10 \
    nice -n 19 taskpolicy -b ./tools/bench/eagle5_paired_bench.sh \
    > reports/eagle5_paired_64tok_2026_05_26.txt 2>&1"

step paired_bench_256 \
  bash -c "EAGLE5_HEAD=$WINNER_CKPT TOKENS=256 TRIALS=10 \
    nice -n 19 taskpolicy -b ./tools/bench/eagle5_paired_bench.sh \
    > reports/eagle5_paired_256tok_2026_05_26.txt 2>&1"

# Ride-alongs (soft)
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
echo "Grid summary:"
printf "  proxy + lr 3e-4   K=4=%s%%\n" "${P3:-NA}"
printf "  proxy + lr 1e-3   K=4=%s%%\n" "${P1:-NA}"
printf "  off   + lr 3e-4   K=4=%s%%\n" "${O3:-NA}"
printf "  off   + lr 1e-3   K=4=%s%%\n" "${O1:-NA}"
echo "  WINNER: $WINNER (K=4 = $BEST%)"

#!/usr/bin/env bash
# path-to-100 Step 2B — chain-K=4 acceptance-distribution harness.
#
# Background (reports/path_to_90/plans/path_to_100_repath.md §Step 2B):
#   Clean-window bench 2026-05-20 shows eagle4 / parallel-k / K=4 at
#   7.52 dec_tps vs off=26.87. With K=4 the math floor for break-even
#   with off (outer_step ≈ 1.5× off_step) is mean_accept ≈ 0.5; for
#   path-to-50 it's ≈2.0; for path-to-90 it's ≈4.0. Current 7.52 means
#   EITHER outer_step is far worse than 1.5× off OR mean_accept ≈ 0.
#
# This harness captures the existing [spec/eagle4-chain] log emitted
# from deepseek_v2.rs line ~1762 across 3 prompts and parses:
#
#   * accept distribution        (histogram: bins 0..=K=4)
#   * mean acceptance            (sum(first_reject) / outer_iters)
#   * median outer_step_ms       (median of step= field)
#   * implied break-even tps     ((1+mean_accept)/median_outer_step_s)
#   * Gate 1: mean_accept < 0.5  → head architectural wall
#   * Gate 2: mean_accept ≥ 1.0 + outer_step ≥ 1.5× off_step
#                                → glue is the cost; L5 Lever B target
#
# Configs measured:
#   1. off / sequential / K=1                (re-baseline for off_step_ms)
#   2. eagle4 / parallel-k / K=4 (production chain config — the regressor)
#
# REQUIRES: Claude.app quit (Cmd-Q). Refuses if pgrep finds Claude alive.
#
# Outputs:
#   reports/path_to_90/_bench_step2b_<TS>/
#     summary.txt          — accept histogram + medians + gate verdicts
#     chain_log.txt        — full [spec/eagle4-chain] capture (3 prompts)
#     chain_steps.csv      — per-outer-iter: prompt, accept, step_ms

set -euo pipefail

usage() {
  cat <<'EOF'
path_to_100_step2b.sh — chain-K=4 acceptance distribution harness

Usage:
  ./tools/bench/path_to_100_step2b.sh           # default: K=4, parallel-k
  EAGLE4_CHAIN_K=8 ./tools/bench/path_to_100_step2b.sh  # try K=8
  ./tools/bench/path_to_100_step2b.sh --help

Configs measured:
  1. off / sequential / K=1   (baseline for off_step_ms break-even calc)
  2. eagle4 / parallel-k / K=4 (chain config — captures [spec/eagle4-chain])

Prerequisites:
  - Claude.app quit (Cmd-Q)
  - models/deepseek-v2-lite-q4.gguf present
  - eagle4/v2lite_frozen.npz + eagle4/checkpoints/eagle4_v3/best.npz present

Outputs land under reports/path_to_90/_bench_step2b_<timestamp>/.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if pgrep -i "Claude" >/dev/null 2>&1; then
  echo "ERROR: Claude is still running. Quit Claude.app (Cmd-Q) before benching." >&2
  echo "       Contended GPU produces 4-5x inflated dec_tps — useless data." >&2
  exit 2
fi

if pgrep -f "slm" >/dev/null 2>&1; then
  echo "WARN: slm process detected — pause it (kill -STOP <pid>) or wait until idle." >&2
  sleep 30
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

WEIGHTS="$REPO_ROOT/models/deepseek-v2-lite-q4.gguf"
PROFILE_SEQ="$REPO_ROOT/profiles/deepseek-v2-lite-q4.m3pro18.json"
PROFILE_PK="/tmp/path_to_100_step2b_profile_parallelk.json"
FROZEN_NPZ="$REPO_ROOT/eagle4/v2lite_frozen.npz"
DRAFT_NPZ="${EAGLE4_CKPT:-$REPO_ROOT/eagle4/checkpoints/eagle4_v3/best.npz}"
DISMANTLE="$REPO_ROOT/target/release/dismantle"
CHAIN_K="${EAGLE4_CHAIN_K:-4}"

if [[ ! -x "$DISMANTLE" ]]; then
  echo "ERROR: $DISMANTLE missing. Build first: cargo build --release --workspace" >&2
  exit 3
fi
for f in "$WEIGHTS" "$PROFILE_SEQ" "$FROZEN_NPZ" "$DRAFT_NPZ"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing artifact: $f" >&2
    exit 4
  fi
done

# Build the parallel-k variant of the profile (mirrors path_to_125_bench.sh).
python3 -c "
import json
p = json.load(open('$PROFILE_SEQ'))
p['selected']['verify_kernels'] = 'parallel-k'
json.dump(p, open('$PROFILE_PK','w'), indent=2)
"

TS="$(date +%Y%m%dT%H%M%S)"
OUTDIR="$REPO_ROOT/reports/path_to_90/_bench_step2b_${TS}"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.txt"
CHAIN_LOG="$OUTDIR/chain_log.txt"
STEPS_CSV="$OUTDIR/chain_steps.csv"

PROMPTS=(
  "The quick brown fox"
  "Write a Python function to compute Fibonacci numbers"
  "Summarize the plot of Hamlet in three sentences"
)
TOKENS=64

echo "=== path-to-100 Step 2B — chain-K=$CHAIN_K acceptance @ $TS ===" | tee "$SUMMARY"
echo "draft head: $DRAFT_NPZ" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# --- 1. off baseline (single trial per prompt — we only need a median step_ms reference) ---
echo "--- baseline: off / sequential / K=1 (for break-even calc) ---" | tee -a "$SUMMARY"
OFF_TPS_SUM=0
OFF_TPS_N=0
for prompt in "${PROMPTS[@]}"; do
  out=$(nice -n 19 "$DISMANTLE" generate \
          --weights "$WEIGHTS" --kernel-profile "$PROFILE_SEQ" \
          --prompt "$prompt" --max-new-tokens "$TOKENS" --temperature 0 \
          2>&1)
  dec_tps=$(echo "$out" | grep -oE 'dec_tps=[0-9.]+' | head -1 | cut -d= -f2)
  dec_tps="${dec_tps:-0}"
  printf '  prompt=%-32s off_dec_tps=%s\n' "$(echo "$prompt" | cut -c1-32)" "$dec_tps" | tee -a "$SUMMARY"
  OFF_TPS_SUM=$(python3 -c "print($OFF_TPS_SUM + $dec_tps)")
  OFF_TPS_N=$((OFF_TPS_N + 1))
done
OFF_TPS_MEAN=$(python3 -c "print($OFF_TPS_SUM / $OFF_TPS_N if $OFF_TPS_N else 0)")
OFF_STEP_MS=$(python3 -c "print(1000.0 / $OFF_TPS_MEAN if $OFF_TPS_MEAN else 0)")
echo "  off_dec_tps_mean=$OFF_TPS_MEAN  off_step_ms=$OFF_STEP_MS" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# --- 2. eagle4 chain-K=N with spec_log capture ---
echo "--- eagle4 / parallel-k / K=$CHAIN_K with DISMANTLE_SPEC_LOG=1 ---" | tee -a "$SUMMARY"
: > "$CHAIN_LOG"
echo "prompt_idx,accept,draft_actual_k,step_ms,emit,inst_tps" > "$STEPS_CSV"

pidx=0
for prompt in "${PROMPTS[@]}"; do
  pidx=$((pidx + 1))
  echo "  [prompt $pidx] $(echo "$prompt" | cut -c1-50)" | tee -a "$SUMMARY"
  out=$(env DISMANTLE_SPEC_LOG=1 EAGLE4_CHAIN_K="$CHAIN_K" nice -n 19 "$DISMANTLE" generate \
          --weights "$WEIGHTS" --kernel-profile "$PROFILE_PK" \
          --prompt "$prompt" --max-new-tokens "$TOKENS" --temperature 0 \
          --speculate eagle4 --draft-head "$DRAFT_NPZ" --eagle4-frozen "$FROZEN_NPZ" \
          2>&1)
  # Capture full chain log for this prompt.
  echo "=== prompt $pidx: $prompt ===" >> "$CHAIN_LOG"
  echo "$out" | grep '\[spec/eagle4-chain\]' >> "$CHAIN_LOG" || true
  echo "" >> "$CHAIN_LOG"

  # Also extract dec_tps from the [stats] line for sanity.
  dec_tps=$(echo "$out" | grep -oE 'dec_tps=[0-9.]+' | head -1 | cut -d= -f2)
  printf '    chain_dec_tps=%s\n' "${dec_tps:-?}" | tee -a "$SUMMARY"

  # Parse each chain line into the CSV.
  echo "$out" | grep '\[spec/eagle4-chain\]' | \
    sed -E "s/.*K=([0-9]+) accept=([0-9]+)\/([0-9]+) step=([0-9.]+)ms emit=([0-9]+) tps=([0-9.]+).*/${pidx},\2,\3,\4,\5,\6/" \
    >> "$STEPS_CSV" || true
done
echo "" | tee -a "$SUMMARY"

# --- 3. analysis ---
echo "=== analysis ===" | tee -a "$SUMMARY"
python3 - "$STEPS_CSV" "$OFF_STEP_MS" "$CHAIN_K" <<'PY' | tee -a "$SUMMARY"
import sys, csv, statistics, collections
csv_path, off_step_ms, K = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
rows = list(csv.DictReader(open(csv_path)))
if not rows:
    print("(no chain log lines parsed — instrumentation may not be firing)")
    sys.exit(0)

accepts = [int(r['accept']) for r in rows]
step_ms = [float(r['step_ms']) for r in rows]
emits   = [int(r['emit']) for r in rows]
inst_tps = [float(r['inst_tps']) for r in rows]
n = len(rows)

# Distribution histogram over accept bins.
hist = collections.Counter(accepts)
print("")
print(f"  outer_iters captured       : {n}")
print(f"  off_step_ms (single-tok)   : {off_step_ms:.2f}")
print(f"  chain_K                    : {K}")
print("")
print(f"  accept distribution:")
total = sum(hist.values())
for bin in range(K+1):
    cnt = hist.get(bin, 0)
    pct = 100.0 * cnt / total if total else 0
    bar = '#' * int(pct/2)
    print(f"    accept={bin}/{K}  count={cnt:4d}  pct={pct:5.1f}%  {bar}")

mean_accept   = statistics.mean(accepts)
median_step   = statistics.median(step_ms)
mean_step     = statistics.mean(step_ms)
mean_emit     = statistics.mean(emits)
median_tps    = statistics.median(inst_tps)
implied_tps   = (1 + mean_accept) / (median_step / 1000.0)

print("")
print(f"  mean_accept                : {mean_accept:.3f} / {K}")
print(f"  median_outer_step_ms       : {median_step:.2f}")
print(f"  mean_outer_step_ms         : {mean_step:.2f}")
print(f"  median_inst_tps            : {median_tps:.2f}")
print(f"  implied_chain_tps          : (1+{mean_accept:.3f})/{median_step:.2f}ms = {implied_tps:.2f}")
print(f"  mean_emit                  : {mean_emit:.3f} tokens/outer")

# Break-even math
need_accept_breakeven = (median_step / off_step_ms) - 1.0  # to match off_dec_tps
print(f"  step_inflation             : {median_step/off_step_ms:.2f}× off_step_ms")
print(f"  acceptance to break-even   : {need_accept_breakeven:.2f}  (mean_accept must reach this to match off_dec_tps)")

# Gates
print("")
print("=== gate verdict ===")
gate1 = mean_accept < 0.5
gate2_glue = (mean_accept >= 1.0) and (median_step / off_step_ms >= 1.5)
gate_breakeven = mean_accept >= need_accept_breakeven

print(f"  Gate 1 (mean_accept < 0.5)               : {'FAIL → head architectural wall' if gate1 else 'pass'}")
print(f"  Gate 2 (accept ≥ 1.0 AND step ≥ 1.5×off) : {'FAIL → glue/L5-B target' if gate2_glue else 'pass'}")
print(f"  Gate break-even (accept ≥ {need_accept_breakeven:.2f})       : {'pass — chain beats off' if gate_breakeven else 'FAIL — chain loses to off'}")
print("")
if gate1:
    print("  Verdict: HEAD IS THE WALL. The draft head's K=4 acceptance is")
    print("  below the no-accept floor required for chain-K=4 to even break")
    print("  even with off-mode. F.2 already ruled out medusa K=8 acceptance;")
    print("  this confirms the eagle4_v3 head is similarly capped. Path-to-100")
    print("  via chain-K=4 is closed. Options: F.3 (medusa rust port retry),")
    print("  F.5 (hybrid tree), or new draft head architecture/training.")
elif gate2_glue:
    print("  Verdict: GLUE IS THE COST. Acceptance is non-zero but the outer")
    print("  step is bloated >1.5× off_step. The head proposes are not the")
    print("  bottleneck. Next implementation target: L5 Lever B (chain-step")
    print("  pipelining), argbuf rollup, persistent threads for the K head")
    print("  proposes, and reducing the seed-forward / verify-batch glue.")
elif gate_breakeven:
    print("  Verdict: CHAIN IS WORKING. mean_accept clears the break-even")
    print("  threshold; chain-K=4 outperforms off-mode. Focus shifts to")
    print("  further amortization — bigger K, persistent threads, head")
    print("  distillation for higher acceptance.")
else:
    print("  Verdict: PARTIAL — acceptance is between Gate 1 and break-even.")
    print("  Head has some signal but not enough at this K. Could try lower K")
    print("  (K=2) to see if acceptance per K improves, or invest in head")
    print("  quality (Track 2 head retrain).")
PY

echo "" | tee -a "$SUMMARY"
echo "=== outputs ===" | tee -a "$SUMMARY"
echo "summary:    $SUMMARY"     | tee -a "$SUMMARY"
echo "chain_log:  $CHAIN_LOG"   | tee -a "$SUMMARY"
echo "steps_csv:  $STEPS_CSV"   | tee -a "$SUMMARY"

if command -v osascript >/dev/null 2>&1; then
  osascript -e 'display notification "Step 2B bench complete" with title "dismantle"' || true
fi
echo ""
echo "Bench complete. See $SUMMARY"

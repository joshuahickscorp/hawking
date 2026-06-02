#!/usr/bin/env bash
# =============================================================================
# tools/bench/final_analysis.sh — ONE turnkey clean-room final analysis.
#
#   ⚠⚠⚠  RUN THIS WITH CLAUDE CODE FULLY QUIT.  ⚠⚠⚠
#
# Prints ABSOLUTE metrics (dec_tps, J/tok, quality) for the paradigm-shift
# build. A running Claude/agent session inflates throughput ~4-5x, so these
# numbers are only meaningful on a clean machine. (Paired A/B via ab_lever.sh
# is contamination-robust and can run anytime; this batch is the absolute one.)
#
# Runs, in sequence (~15-25 min total):
#   1. ABSOLUTE ANCHOR  — clean decode tps + J/tok + the Q3 §A byte-cut proxy
#   2. f16-KV ENERGY    — J/tok baseline vs DISMANTLE_QWEN_F16_KV=1 @1024 tok
#                         (the one genuinely open question: is f16-KV a real
#                          energy lever, or footprint-only? lower J/tok = real)
#   3. QUALITY          — f16-scales (--profile fast) + f16-KV token/logit drift
#
# Usage:
#   tools/bench/final_analysis.sh            # the full analysis
#   tools/bench/final_analysis.sh --quick    # skip §3 quality (faster)
#
# Everything is tee'd to reports/bench/final_analysis_<ts>.log and a best-effort
# summary is printed at the end. JSON artifacts land in reports/{bench,quality}/.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

TS=$(date +%Y%m%dT%H%M%S)
LOG="reports/bench/final_analysis_${TS}.log"
mkdir -p reports/bench reports/quality
# Tee all output (stdout+stderr) to the log.
exec > >(tee "$LOG") 2>&1

banner() { printf '\n\n=================== %s ===================\n' "$1"; }

echo "dismantle final analysis — $(date)"
echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git rev-parse --short HEAD 2>/dev/null)"
echo "log:    $LOG"
echo "NOTE:   absolute metrics — only valid with Claude Code QUIT."

banner "0/3  build (idempotent; fast if already built)"
cargo build --release --workspace 2>&1 | tail -2

banner "1/3  ABSOLUTE ANCHOR  (clean decode tps + J/tok + Q3 §A proxy)"
tools/bench/clean_room_batch.sh || echo "[warn] clean_room_batch.sh returned non-zero (read its output above)"

banner "2/3  f16-KV ENERGY QUESTION  (J/tok, baseline vs f16-KV @1024 tok)"
echo "------ baseline (f16-KV OFF) ------"
tools/bench/phase_joules.sh --tokens 1024 || echo "[warn] phase_joules baseline returned non-zero"
echo "------ f16-KV ON (DISMANTLE_QWEN_F16_KV=1) ------"
DISMANTLE_QWEN_F16_KV=1 tools/bench/phase_joules.sh --tokens 1024 || echo "[warn] phase_joules f16-KV returned non-zero"

if [[ "$QUICK" == 0 ]]; then
  banner "3/3  QUALITY  (f16-scales / --profile fast, and f16-KV)"
  tools/bench/quality_oracle.sh --lever DISMANTLE_QWEN_PREDEC_F16SCALES --label f16scales || echo "[warn] quality f16scales non-zero"
  tools/bench/quality_oracle.sh --lever DISMANTLE_QWEN_F16_KV --label f16kv --long || echo "[warn] quality f16kv non-zero"
else
  echo; echo "(--quick: skipped §3 quality)"
fi

sleep 1   # let tee flush before we grep the log for the summary
banner "SUMMARY  (best-effort — full numbers in the sections above)"
echo "--- throughput / energy ---"
grep -hiE "dec_tps|decode_tps|tokens_per_sec|J/tok|joules|GB/s|W GPU|W pkg|peak|0\.[0-9]+ J" "$LOG" \
  | grep -viE "warning|note:|help:|^#" | sort -u | head -40 || true
echo "--- quality verdicts ---"
grep -hiE "PASS|FAIL|identical|cosine|drift|verdict" "$LOG" \
  | grep -viE "warning|note:|help:" | head -25 || true

echo
echo "DONE.  full log: $LOG"
echo "JSON:  reports/quality/oracle_f16scales.json  reports/quality/oracle_f16kv.json  reports/bench/*.json"
echo "KEY READ — §2: is f16-KV's J/tok LOWER with the flag on?"
echo "   lower  -> f16-KV is a real ENERGY lever (footprint + energy)"
echo "   higher -> footprint-only (the f16 dequant compute eats the DRAM saving)"

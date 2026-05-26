#!/usr/bin/env bash
# tools/bench/build_qwen3b_kernel_profile.sh
#
# Generates the Qwen-3B kernel profile via `dismantle autotune`. The
# Track E paired bench (w4a8_lmhead_per_channel_bench.sh) needs a
# Qwen-3B profile to pass via --kernel-profile, but only the V2-Lite
# profile ships in profiles/ today.
#
# `dismantle autotune` runs a deterministic kernel-variant search using
# the model's actual shapes, then writes a JSON profile that hard-codes
# the winner kernel for each (op, shape) pair. The shader-hash in the
# profile is stamped from the current build so the in-flight build's
# shader recompilation is automatically captured.
#
# Wall time: ~10-30 min depending on how exhaustive the autotune sweep
# is (--max-hours bounds it; we cap at 1 hr to be safe).
#
# Launch:
#   nohup tools/bench/build_qwen3b_kernel_profile.sh \
#     > reports/build_qwen3b_kernel_profile.log 2>&1 & disown

set -euo pipefail
cd "$(dirname "$0")/../.."

WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
OUT="${OUT:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
MAX_HOURS="${MAX_HOURS:-1}"
LOG="${LOG:-reports/qwen3b_autotune_$(date +%Y%m%d_%H%M%S).jsonl}"

if [[ ! -f "$WEIGHTS" ]]; then
    echo "❌ weights missing: $WEIGHTS" >&2
    exit 1
fi
if [[ ! -x ./target/release/dismantle ]]; then
    echo "❌ dismantle binary not built — run 'cargo build --release -p dismantle' first" >&2
    exit 1
fi

mkdir -p reports profiles

echo "[autotune] weights:   $WEIGHTS"
echo "[autotune] output:    $OUT"
echo "[autotune] max-hours: $MAX_HOURS"
echo "[autotune] log:       $LOG"
echo "[autotune] start:     $(date -u +%FT%TZ)"

nice -n 19 taskpolicy -b ./target/release/dismantle autotune \
    --weights "$WEIGHTS" \
    --profile m3-pro-18gb \
    --max-hours "$MAX_HOURS" \
    --out "$OUT" \
    --log "$LOG"

echo "[autotune] done:      $(date -u +%FT%TZ)"
echo
echo "Resulting profile:"
ls -lh "$OUT"
echo
echo "Use it for the Track E bench:"
echo "  KERNEL_PROFILE=$OUT \\"
echo "  bash tools/bench/w4a8_lmhead_per_channel_bench.sh"

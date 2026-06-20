#!/usr/bin/env bash
# tools/bench/mst_capture.sh — wrap a dismantle bench invocation in
# `xcrun xctrace record --template "Metal System Trace"` and write a
# `.trace` bundle to traces/<timestamp>/.
#
# Purpose: Phase 1 prerequisite. Captures
#   • per-layer dispatch latency
#   • per-token GPU/CPU bandwidth utilization vs the 150 GB/s ceiling
#   • routed-expert cache hit rate (via Metal counters)
#
# Use the produced .trace bundle in Instruments (GUI) or `xctrace export`
# to extract dispatch counts — feeds the ICB design (dispatch budget
# estimate). NOT a perf-shippable measurement on its own; recording
# overhead taints dec_tps.
#
# Usage:
#   ./tools/bench/mst_capture.sh                    # wraps quick_bench defaults
#   ./tools/bench/mst_capture.sh -- bench --suite decode --max-new-tokens 32 \
#       --weights models/deepseek-v2-lite-q4.gguf \
#       --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json
#
# Env:
#   WEIGHTS   default model weights (when no `--` args given)
#   PROFILE   default kernel profile
#   TOKENS    default token count for the default invocation
#   TRACES_DIR  output dir (default: traces/)
#   BIN       dismantle binary path (default: target/release/hawking)

set -euo pipefail
cd "$(dirname "$0")/../.."

if ! command -v xcrun >/dev/null 2>&1; then
    echo "error: xcrun not found. MST requires macOS + Xcode command-line tools." >&2
    exit 1
fi
# Don't pipe `xctrace list templates` through grep -q: with `set -o pipefail`,
# grep's early exit makes xctrace receive SIGPIPE and report a non-zero status,
# which would falsely trigger this branch. Capture output first, then grep.
_xctrace_templates="$(xcrun xctrace list templates 2>/dev/null || true)"
if ! printf '%s\n' "$_xctrace_templates" | grep -q "Metal System Trace"; then
    echo "error: xctrace 'Metal System Trace' template missing. Install Xcode (not just CLT)." >&2
    exit 1
fi

WEIGHTS="${WEIGHTS:-models/deepseek-v2-lite-q4.gguf}"
PROFILE="${PROFILE:-profiles/deepseek-v2-lite-q4.m3pro18.json}"
TOKENS="${TOKENS:-32}"
BIN="${BIN:-./target/release/hawking}"
TRACES_DIR="${TRACES_DIR:-traces}"

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not built. Run: cargo build --release -p hawking" >&2
    exit 1
fi

if pgrep -f "Claude.app" >/dev/null 2>&1; then
    echo "⚠️  Claude desktop app is running — trace counters will reflect contention." >&2
    echo "   For clean dispatch-count numbers, Cmd+Q Claude first." >&2
fi

mkdir -p "$TRACES_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$TRACES_DIR/mst_${STAMP}.trace"

if [ "$#" -gt 0 ] && [ "$1" = "--" ]; then
    shift
    CMD=("$BIN" "$@")
else
    CMD=("$BIN" bench --backend hawking --suite decode \
        --weights "$WEIGHTS" --trials 1 --max-new-tokens "$TOKENS" \
        --kernel-profile "$PROFILE")
fi

echo "=== MST capture ==="
echo "  output:  $OUT"
echo "  command: ${CMD[*]}"
echo

xcrun xctrace record \
    --template "Metal System Trace" \
    --output "$OUT" \
    --launch -- "${CMD[@]}"

echo
echo "Trace written to: $OUT"
echo "Open in Instruments:    open '$OUT'"
echo "Export dispatch table:  xcrun xctrace export --input '$OUT' --xpath '/trace-toc/run[1]/data/table[@schema=\"metal-gpu-channel-events\"]'"

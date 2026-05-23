#!/usr/bin/env bash
#
# tools/bisect/bisect_v2_lite.sh — git bisect runner for V2-Lite perf regressions.
#
# Returns 0 if V2-Lite dec_tps >= threshold, 1 otherwise.
# Used as: git bisect run tools/bisect/bisect_v2_lite.sh <threshold>
#
# Example:
#   git bisect start
#   git bisect bad HEAD             # current bad state
#   git bisect good <last_good>     # last known-good commit
#   git bisect run tools/bisect/bisect_v2_lite.sh 17.0
#   git bisect reset
#
# Each step: build (~45s) + autotune (~5s) + 4-trial bench (~3-5 min).
# For 10 commits between good and bad: ~35-60 min total.
#
# See tools/bisect/README.md for full workflow.

set -euo pipefail
cd "$(dirname "$0")/../.."

THRESHOLD="${1:-17.0}"
WEIGHTS="${WEIGHTS:-models/deepseek-v2-lite-q4.gguf}"
PROFILE="${BISECT_PROFILE:-/tmp/dismantle_bisect_profile.json}"
BIN="./target/release/dismantle"

log() { printf '[bisect] %s\n' "$*" >&2; }

log "=== bisect step at $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown') ==="
log "  threshold: ${THRESHOLD} dec_tps"

# ── 1. Build at this commit ───────────────────────────────────────────────────
log "building..."
if ! cargo build --release --workspace -q 2>/dev/null; then
    log "BUILD FAILED — marking as BAD (build failure = regression)"
    exit 1
fi
log "build OK"

# ── 2. Autotune profile for this commit's shaders ────────────────────────────
log "autotuning (shader hash may have changed)..."
if ! "$BIN" autotune \
        --weights "$WEIGHTS" \
        --out "$PROFILE" \
        -q 2>/dev/null; then
    # Autotune failure: try with existing profile if available, else fail.
    DEFAULT_PROFILE="profiles/deepseek-v2-lite-q4.m3pro18.json"
    if [[ -f "$DEFAULT_PROFILE" ]]; then
        log "autotune failed — falling back to $DEFAULT_PROFILE"
        PROFILE="$DEFAULT_PROFILE"
    else
        log "autotune failed and no fallback — marking as BAD"
        exit 1
    fi
fi
log "profile ready: $PROFILE"

# ── 3. Quick 4-trial bench ────────────────────────────────────────────────────
log "running 4-trial bench (TRIALS=4 TOKENS=24)..."
BENCH_OUT=$(TRIALS=4 TOKENS=24 ANCHOR=999 PROFILE="$PROFILE" \
    bash tools/bench/coexist_bench.sh \
    --quiet --profile "$PROFILE" 2>/dev/null || echo "")

MEDIAN=$(printf '%s\n' "$BENCH_OUT" | grep "^median:" | awk '{print $2}')

if [[ -z "$MEDIAN" ]]; then
    log "bench produced no median — marking as BAD"
    exit 1
fi

log "median: ${MEDIAN} dec_tps  (threshold: ${THRESHOLD})"

if awk -v med="$MEDIAN" -v thr="$THRESHOLD" 'BEGIN { exit (med >= thr) ? 0 : 1 }'; then
    log "GOOD (${MEDIAN} >= ${THRESHOLD})"
    exit 0
else
    log "BAD  (${MEDIAN} < ${THRESHOLD})"
    exit 1
fi

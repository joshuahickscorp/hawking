#!/usr/bin/env bash
# tools/training/overnight_path_to_50_bench.sh
#
# Post-eagle5-training bench stage. Designed to run as a follow-on to
# tools/training/overnight_path_to_50.sh once that pipeline reports
# current_stage == complete && state == done.
#
# Reads:
#   artifacts/runs/overnight/extended_status.json  (gate: training must be done)
#
# Writes:
#   artifacts/runs/overnight/path_to_50_bench_status.json   (heartbeat)
#   artifacts/runs/path_to_50_matrix/<utc>/report.md         (the matrix output)
#   artifacts/runs/overnight/path_to_50_bench.log            (full log)
#
# Refuses to run if Claude is live (uses tools/bench/path_to_50_matrix.sh
# which is --strict by default).
#
# Intended invocation:
#   1. After overnight_path_to_50.sh finishes
#   2. After user has quit Claude (or via launchd at a known quiet window)
#   3. Manual:  bash tools/training/overnight_path_to_50_bench.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="artifacts/runs/overnight"
mkdir -p "$LOG_DIR"
STATUS_PATH="$LOG_DIR/path_to_50_bench_status.json"
LOG_PATH="$LOG_DIR/path_to_50_bench.log"
TRAINING_STATUS="$LOG_DIR/extended_status.json"

START_EPOCH=$(date +%s)

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

write_status() {
    local stage="$1" state="$2" note="${3:-}"
    local now uptime free_gb
    now=$(date +%s)
    uptime=$((now - START_EPOCH))
    free_gb=$(df -g . | awk 'NR==2 {print $4}')
    cat > "$STATUS_PATH.tmp" <<EOF
{
  "ts": "$(stamp)",
  "pipeline_pid": $$,
  "current_stage": "$stage",
  "state": "$state",
  "note": "$note",
  "uptime_seconds": $uptime,
  "free_disk_gb": $free_gb
}
EOF
    mv "$STATUS_PATH.tmp" "$STATUS_PATH"
}

log() {
    echo "[$(stamp)] $*" | tee -a "$LOG_PATH"
}

# --- Gate: training must be done ---
log "=== path_to_50 bench stage starting ==="
if [[ ! -f "$TRAINING_STATUS" ]]; then
    log "WARN: no $TRAINING_STATUS — proceeding (assumed manual invocation)"
else
    cur_stage=$(jq -r '.current_stage // "unknown"' "$TRAINING_STATUS")
    cur_state=$(jq -r '.state // "unknown"' "$TRAINING_STATUS")
    if [[ "$cur_stage" != "complete" ]] || [[ "$cur_state" != "done" ]]; then
        log "❌ training not complete (stage=$cur_stage state=$cur_state). Aborting."
        write_status "gate" "failed" "training pipeline still running"
        exit 65
    fi
    log "training gate clear: stage=$cur_stage state=$cur_state"
fi

# --- Gate: no Claude (relaxable) ---
# STRICT_CLEAN_BENCH=1 (default) refuses to run when Claude.app is live —
# absolute numbers would be contaminated 4-5×. Set STRICT_CLEAN_BENCH=0 to
# proceed anyway; paired deltas (lever vs baseline) cancel contamination per
# memory feedback_bench_with_claude_open.md.
: "${STRICT_CLEAN_BENCH:=1}"
if [[ "$STRICT_CLEAN_BENCH" = "1" ]] && pgrep -f "Claude.app" > /dev/null 2>&1; then
    log "❌ Claude.app is running — bench numbers would be contaminated. Aborting."
    log "   (set STRICT_CLEAN_BENCH=0 to override; paired deltas still valid.)"
    write_status "gate" "failed" "Claude.app live"
    exit 64
fi
if [[ "$STRICT_CLEAN_BENCH" = "0" ]]; then
    log "⚠️  STRICT_CLEAN_BENCH=0 — absolute tps numbers contaminated; trust paired deltas only"
fi

# --- Pause hook (between every stage) ---
wait_if_paused() {
    local where="$1"
    if [[ -f artifacts/runs/PAUSE ]]; then
        log "⏸  PAUSED at $where (touch artifacts/runs/RESUME or rm artifacts/runs/PAUSE to continue)"
        write_status "paused" "running" "at $where"
        while [[ -f artifacts/runs/PAUSE ]]; do
            sleep 10
            if [[ -f artifacts/runs/RESUME ]]; then
                log "▶  RESUME signal — clearing pause"
                rm -f artifacts/runs/PAUSE artifacts/runs/RESUME
                break
            fi
        done
        log "▶  resumed at $where"
    fi
}

# --- Stage 1: cargo build (in case worktree changes haven't been built) ---
wait_if_paused "before-build"
log "=== build ==="
write_status "build" "running"
if ! cargo build --release -p dismantle >> "$LOG_PATH" 2>&1; then
    log "❌ cargo build failed"
    write_status "build" "failed" "see log"
    exit 1
fi
write_status "build" "done"

# --- Stage 2: parity tests for every landed lever ---
# These are CPU-only and cheap (~1-2 min each). If parity is broken,
# bench numbers downstream are meaningless — abort early.
wait_if_paused "before-parity"
log "=== parity gate ==="
write_status "parity" "running"

PARITY_TESTS=(
    # Session B: spec-decode parity (NGram baseline + exact-shared if test exists)
    "v1_1_phase4D_spec_exact_mode"
    # Existing greedy parity (any regression here means something is fundamentally broken)
    "integration_greedy_64"
)

# Optional parity tests for A and C — included only if the test file exists
# (so this script doesn't have to be edited when A/C land).
[[ -f crates/dismantle-core/tests/mixed_precision_parity.rs ]] && \
    PARITY_TESTS+=("mixed_precision_parity")
[[ -f crates/dismantle-core/tests/q8_kv_parity.rs ]] && \
    PARITY_TESTS+=("q8_kv_parity")

parity_fail=0
for t in "${PARITY_TESTS[@]}"; do
    log "  parity: $t"
    if ! cargo test --release -p dismantle-core --test "$t" >> "$LOG_PATH" 2>&1; then
        log "  ❌ parity FAILED: $t"
        parity_fail=$((parity_fail + 1))
    else
        log "  ✓ parity OK: $t"
    fi
done

if [[ "$parity_fail" -gt 0 ]]; then
    log "❌ $parity_fail parity test(s) failed. Aborting before bench."
    write_status "parity" "failed" "$parity_fail tests failed"
    exit 2
fi
write_status "parity" "done"

# --- Stage 3: bench matrix ---
wait_if_paused "before-bench"
log "=== bench matrix ==="
write_status "bench_matrix" "running"
if ! bash tools/bench/path_to_50_matrix.sh >> "$LOG_PATH" 2>&1; then
    log "❌ bench matrix failed"
    write_status "bench_matrix" "failed" "see log"
    exit 3
fi
write_status "bench_matrix" "done"

# --- Wrap up ---
REPORT="artifacts/runs/path_to_50_matrix/latest/report.md"
log "=== complete ==="
log "report: $REPORT"
write_status "complete" "done" "report at $REPORT"

# Echo report tail for the cron monitor / user
if [[ -f "$REPORT" ]]; then
    echo
    echo "--- report tail ---"
    tail -n 40 "$REPORT"
fi

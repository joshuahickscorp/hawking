#!/usr/bin/env bash
# stage0_capture.sh — path-to-90 execution-plan step 1.
#
# Captures clean baseline tok/s + Metal System Trace bundle so the
# next attended Claude Code session can compute bandwidth efficiency
# vs M3 Pro's 150 GB/s theoretical ceiling.
#
# RUN THIS AFTER Cmd-Q'ing the Claude desktop app. The CLI Claude
# session that asked you to run this is dead; you'll start a new one
# afterward and it will pick up from the marker files below.
#
# Usage:
#   cd /Users/scammermike/Downloads/dismantle/.claude/worktrees/dreamy-golick-d54ff8
#   bash tools/bench/stage0_capture.sh
#
# Markers (for the next Claude session to read):
#   reports/path_to_90/_stage0_capture/STATUS.log  ← progress log, tail -f friendly
#   reports/path_to_90/_stage0_capture/raw.json    ← parsed metrics
#   reports/path_to_90/_stage0_capture/mst.trace   ← Instruments bundle
#   reports/path_to_90/_stage0_capture/DONE        ← created on success
#   reports/path_to_90/_stage0_capture/FAILED      ← created on failure
#
# Estimated runtime: 4-8 minutes (model loads ~5-10s × 4 runs + 64-tok decodes + xctrace).

set -uo pipefail
cd "$(dirname "$0")/../.."

OUT_DIR="reports/path_to_90/_stage0_capture"
mkdir -p "$OUT_DIR"
STATUS="$OUT_DIR/STATUS.log"
RAW="$OUT_DIR/raw.json"
TRACE="$OUT_DIR/mst.trace"
DONE_MARKER="$OUT_DIR/DONE"
FAIL_MARKER="$OUT_DIR/FAILED"

rm -f "$DONE_MARKER" "$FAIL_MARKER"
: > "$STATUS"

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$STATUS"
}

notify() {
    local title="$1" body="$2" sound="$3"
    osascript -e "display notification \"$body\" with title \"$title\" sound name \"$sound\"" 2>/dev/null || true
}

fail() {
    log "FAILED: $*"
    touch "$FAIL_MARKER"
    notify "path-to-90 step 1" "stage0 FAILED: $*" "Sosumi"
    (say "stage zero capture failed" &) 2>/dev/null || true
    exit 1
}

trap 'fail "interrupted (signal)"' INT TERM

log "=== stage0_capture starting ==="
log "Worktree: $(pwd)"
log "Branch:   $(git branch --show-current)"
log "Commit:   $(git rev-parse HEAD)"
log ""

# ── 1) verify Claude.app is fully quit ─────────────────────────────
if pgrep -f "/Applications/Claude.app/Contents/MacOS/Claude" >/dev/null 2>&1; then
    log "Claude.app processes still running:"
    pgrep -af "/Applications/Claude.app/Contents/MacOS/Claude" | head -5 | tee -a "$STATUS"
    fail "Cmd-Q the Claude desktop app first, then re-run this script."
fi
log "✓ Claude.app: not running"

# ── 2) pause slm overnight trainer (resumes on exit) ──────────────
SLM_PID=$(pgrep -f "overnight_shift.py" 2>/dev/null | head -1 || true)
PAUSED_PIDS=""
if [[ -n "$SLM_PID" ]]; then
    SLM_PGID=$(ps -o pgid= -p "$SLM_PID" 2>/dev/null | tr -d ' ')
    PAUSED_PIDS=$(pgrep -g "$SLM_PGID" 2>/dev/null | tr '\n' ' ')
    log "Pausing slm tree (pgid=$SLM_PGID): $PAUSED_PIDS"
    kill -STOP $PAUSED_PIDS 2>/dev/null || true
    sleep 2
    log "✓ slm paused"
else
    log "✓ slm: not running"
fi

restore_slm() {
    if [[ -n "$PAUSED_PIDS" ]]; then
        log "Resuming slm: $PAUSED_PIDS"
        kill -CONT $PAUSED_PIDS 2>/dev/null || true
    fi
}
trap 'restore_slm' EXIT

# ── 3) preflight: build, weights, profile ─────────────────────────
log ""
log "Building --release (incremental) ..."
if ! cargo build --release --workspace 2>>"$STATUS" >/dev/null; then
    fail "cargo build failed — see STATUS.log"
fi
log "✓ build OK"

BIN="./target/release/dismantle"
WEIGHTS="models/deepseek-v2-lite-q4.gguf"
PROFILE="profiles/deepseek-v2-lite-q4.m3pro18.json"
TOKENS=64

[[ -x "$BIN" ]]    || fail "missing binary: $BIN"
[[ -f "$WEIGHTS" ]] || fail "missing weights: $WEIGHTS"
[[ -f "$PROFILE" ]] || fail "missing profile: $PROFILE"

log "✓ binary:  $BIN"
log "✓ weights: $WEIGHTS ($(du -h "$WEIGHTS" | cut -f1))"
log "✓ profile: $PROFILE"

# ── 4) clean dec_tps bench: 3 trials × 64 tokens ──────────────────
log ""
log "── Clean dec_tps bench (3 trials × $TOKENS tokens) ──"
TRIALS_TPS=()
TRIALS_PREFILL=()
for i in 1 2 3; do
    OUT="$OUT_DIR/bench_t${i}.json"
    log "  trial $i/3 ..."
    if ! "$BIN" bench --backend dismantle --suite decode \
            --weights "$WEIGHTS" --trials 1 --max-new-tokens "$TOKENS" \
            --kernel-profile "$PROFILE" --json "$OUT" >>"$STATUS" 2>&1; then
        fail "bench trial $i failed — see STATUS.log"
    fi
    TPS=$(jq -r '.results.trial_stats[0].decode_tps // 0' "$OUT")
    PRE=$(jq -r '.results.trial_stats[0].prefill_tps // 0' "$OUT")
    log "    decode_tps=$TPS  prefill_tps=$PRE"
    TRIALS_TPS+=("$TPS")
    TRIALS_PREFILL+=("$PRE")
done
MEDIAN_TPS=$(printf '%s\n' "${TRIALS_TPS[@]}" | sort -n | sed -n '2p')
log "✓ median clean dec_tps = $MEDIAN_TPS"

# ── 5) Metal System Trace capture (1 run, 64 tokens) ──────────────
log ""
log "── Metal System Trace (xctrace) ──"
rm -rf "$TRACE"
MST_BENCH_JSON="$OUT_DIR/bench_under_mst.json"
log "Recording trace ... (xctrace may take ~30-90s)"
if ! xctrace record \
        --template "Metal System Trace" \
        --output "$TRACE" \
        --launch -- \
        "$BIN" bench --backend dismantle --suite decode \
            --weights "$WEIGHTS" --trials 1 --max-new-tokens "$TOKENS" \
            --kernel-profile "$PROFILE" --json "$MST_BENCH_JSON" \
        >>"$STATUS" 2>&1; then
    fail "xctrace record failed — likely permissions (System Settings → Privacy & Security → Developer Tools → Terminal)"
fi
log "✓ trace bundle: $TRACE ($(du -sh "$TRACE" | cut -f1))"
MST_TPS=$(jq -r '.results.trial_stats[0].decode_tps // 0' "$MST_BENCH_JSON" 2>/dev/null || echo "0")
log "  dec_tps under MST instrumentation: $MST_TPS"

# ── 6) export trace TOC + every schema we know to try ─────────────
log ""
log "── Exporting trace data ──"
if ! xctrace export --input "$TRACE" --toc --output "$OUT_DIR/toc.xml" >>"$STATUS" 2>&1; then
    log "WARN: --toc export failed; trace bundle is still on disk for manual inspection"
else
    log "✓ TOC: $OUT_DIR/toc.xml"
    # Auto-discover schemas from the TOC and export each.
    SCHEMAS=$(grep -oE 'schema="[^"]+"' "$OUT_DIR/toc.xml" | sort -u | sed 's/schema="//;s/"$//')
    log "Schemas found in TOC:"
    printf '%s\n' "$SCHEMAS" | sed 's/^/    /' | tee -a "$STATUS"
    for s in $SCHEMAS; do
        safe=$(echo "$s" | tr '/:' '__')
        xctrace export --input "$TRACE" \
            --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"$s\"]" \
            --output "$OUT_DIR/schema_${safe}.xml" >>"$STATUS" 2>&1 \
            && log "  ✓ exported $s" \
            || log "  ✗ failed   $s"
    done
fi

# ── 7) write raw.json summary ─────────────────────────────────────
TRIALS_JSON=$(printf '%s,' "${TRIALS_TPS[@]}" | sed 's/,$//')
PREFILL_JSON=$(printf '%s,' "${TRIALS_PREFILL[@]}" | sed 's/,$//')
cat >"$RAW" <<JSON
{
  "captured_at": "$(date -Iseconds)",
  "git_sha": "$(git rev-parse HEAD)",
  "branch": "$(git branch --show-current)",
  "host": {
    "model": "M3 Pro 18 GB",
    "theoretical_bandwidth_gbps": 150
  },
  "bench_clean": {
    "tokens_per_trial": $TOKENS,
    "trials_decode_tps": [$TRIALS_JSON],
    "trials_prefill_tps": [$PREFILL_JSON],
    "median_decode_tps": $MEDIAN_TPS
  },
  "bench_under_mst": {
    "decode_tps": $MST_TPS,
    "json": "$MST_BENCH_JSON"
  },
  "trace": {
    "bundle": "$TRACE",
    "toc": "$OUT_DIR/toc.xml",
    "schemas_dir": "$OUT_DIR"
  }
}
JSON
log "✓ raw summary: $RAW"

# ── 8) DONE ────────────────────────────────────────────────────────
log ""
log "=== stage0_capture DONE ==="
log ""
log "Next: relaunch Claude Code in the worktree and tell it:"
log "  'step 1 done — read reports/path_to_90/_stage0_capture/'"

touch "$DONE_MARKER"
notify "path-to-90 step 1" "stage0 capture complete — return to Claude" "Glass"
(say "stage zero capture complete" &) 2>/dev/null || true
exit 0

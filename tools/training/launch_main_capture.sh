#!/usr/bin/env bash
# Pre-staged main 100K capture launcher (path-to-90 C2 recovery, 2026-05-17).
#
# Sequence — when smoke validates:
#   1. cd into this worktree
#   2. Run this script (no args)
#   3. Capture launches detached, pipeline_loop launches detached
#   4. Both write to training_data/c2_hidden/eagle3_v0/
#
# To halt: touch training_data/c2_hidden/eagle3_v0/pipeline/HALT
set -euo pipefail

WT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$WT"

DISMANTLE="./target/release/dismantle"
WEIGHTS="models/deepseek-v2-lite-q4.gguf"
PROFILE="profiles/deepseek-v2-lite-q4.m3pro18.json"
SAMPLES="tests/data/ultrachat_100k_union.jsonl"
OUT_DIR="training_data/c2_hidden/eagle3_v0"
SHARD_PREFIX="$OUT_DIR/shard_000"
PIPELINE_DIR="$OUT_DIR/pipeline"

# Sanity checks (fail loud rather than launch into a wedge)
[ -x "$DISMANTLE" ]      || { echo "MISSING: $DISMANTLE — run cargo build --release --workspace"; exit 1; }
[ -f "$WEIGHTS" ]        || { echo "MISSING: $WEIGHTS (symlink models/ from main repo if absent)"; exit 1; }
[ -f "$SAMPLES" ]        || { echo "MISSING: $SAMPLES — build the union first"; exit 1; }
[ -f "$PROFILE" ]        || { echo "MISSING: $PROFILE"; exit 1; }

mkdir -p "$OUT_DIR" "$PIPELINE_DIR"

# Don't double-launch. Use PID files instead of pgrep regexes — pgrep -f
# matches against the full argv of every process including shell wrappers
# (status monitors, the harness invoking us, etc.) that may contain the
# patterns we're looking for as literal substrings, leading to false
# positives. A PID file is unambiguous.
CAPTURE_PIDFILE="$OUT_DIR/capture.pid"
LOOP_PIDFILE="$PIPELINE_DIR/loop.pid"

check_pidfile() {
    # $1 = pidfile path, $2 = human name
    [ -f "$1" ] || return 1
    local pid
    pid=$(cat "$1" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    if kill -0 "$pid" 2>/dev/null; then
        echo "$2 already running (pid $pid per $1) — aborting"
        return 0
    fi
    # Stale pidfile — clean up
    rm -f "$1"
    return 1
}
if check_pidfile "$CAPTURE_PIDFILE" "capture"; then exit 2; fi
if check_pidfile "$LOOP_PIDFILE" "pipeline_loop"; then exit 2; fi

# Launch capture: 100K cap (--max-samples), --resume (pick up if .bin exists),
# --no-lm-head (teacher-forced training), nice+taskpolicy for coexist.
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching capture (--max-samples 100000 --resume)"
nohup nice -n 19 taskpolicy -b "$DISMANTLE" capture-hidden \
    --weights "$WEIGHTS" \
    --samples "$SAMPLES" \
    --out "$SHARD_PREFIX" \
    --max-tokens 128 --max-samples 100000 --no-lm-head --resume \
    --kernel-profile "$PROFILE" \
    >> "$SHARD_PREFIX.log" 2>&1 < /dev/null &
disown
CAPTURE_PID=$!
echo "$CAPTURE_PID" > "$CAPTURE_PIDFILE"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] capture PID: $CAPTURE_PID (pidfile: $CAPTURE_PIDFILE)"

# Give capture 3s to fail-fast on bad args before launching the loop
sleep 3
if ! ps -p $CAPTURE_PID >/dev/null 2>&1; then
    echo "capture exited within 3s — see $SHARD_PREFIX.log"
    rm -f "$CAPTURE_PIDFILE"
    tail -20 "$SHARD_PREFIX.log"
    exit 3
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching pipeline_loop"
nohup bash tools/training/pipeline_loop.sh \
    > "$PIPELINE_DIR/loop.log" 2>&1 < /dev/null &
disown
LOOP_PID=$!
echo "$LOOP_PID" > "$LOOP_PIDFILE"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pipeline_loop PID: $LOOP_PID (pidfile: $LOOP_PIDFILE)"

echo
echo "Both launched. Monitor via:"
echo "  tail -f $SHARD_PREFIX.log"
echo "  tail -f $PIPELINE_DIR/loop.log"
echo "Halt the pipeline:"
echo "  touch $PIPELINE_DIR/HALT"

#!/usr/bin/env bash
# tools/training/run_corpus_autonomous.sh
#
# Self-restarting watchdog wrapper around build_corpus.py.
#
# Designed so a coding agent session can be the only supervisor — the user
# kicks this script once and walks away. The script:
#
#   - Loops forever, kicking build_corpus.py with --skip-existing
#     (default). Each parquet shard is idempotent: crashed runs resume
#     from the next unfinished shard with no data loss.
#   - Writes a JSON heartbeat to $HEARTBEAT_PATH every 30 s while
#     build_corpus.py is alive. The agent reads this to track progress.
#   - Sleeps with exponential backoff after crashes (cap 5 min) so OOM
#     loops don't spin the CPU.
#   - Exits cleanly when $STOP_FLAG_PATH exists, when --max-sequences
#     completes, or when EXIT_AFTER_HOURS elapses.
#
# Env knobs (all optional — sensible defaults baked in):
#   MAX_SEQUENCES        default 10000 — the plan's full target
#   MAX_TOKENS_PER_SEQ   default 512
#   SHARD_SIZE           default 32
#   MODEL                default deepseek-ai/DeepSeek-V2-Lite-Chat
#   DATASET              default HuggingFaceH4/ultrachat_200k
#   DEVICE               default mps
#   DTYPE                default float16
#   OUT_DIR              default artifacts/calibration/v2_lite_corpus
#   HEARTBEAT_PATH       default artifacts/calibration/heartbeat.json
#   STOP_FLAG_PATH       default artifacts/calibration/STOP
#   LOG_PATH             default artifacts/calibration/overnight.log
#   PID_PATH             default artifacts/calibration/overnight.pid
#   VENV_PATH            default .venv-calibration (created if missing)
#   EXIT_AFTER_HOURS     default 0 (= run until completion)
#
# Usage:
#   ./tools/training/run_corpus_autonomous.sh                       # default 10k
#   MAX_SEQUENCES=200 ./tools/training/run_corpus_autonomous.sh     # smaller overnight
#   nohup ./tools/training/run_corpus_autonomous.sh > /dev/null 2>&1 &  # detached
#
# Stop cleanly: `touch artifacts/calibration/STOP`
# Kill hard:    `kill $(cat artifacts/calibration/overnight.pid)`

set -uo pipefail
cd "$(dirname "$0")/../.."

MAX_SEQUENCES="${MAX_SEQUENCES:-10000}"
MAX_TOKENS_PER_SEQ="${MAX_TOKENS_PER_SEQ:-512}"
SHARD_SIZE="${SHARD_SIZE:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V2-Lite-Chat}"
DATASET="${DATASET:-HuggingFaceH4/ultrachat_200k}"
DEVICE="${DEVICE:-mps}"
DTYPE="${DTYPE:-float16}"
OUT_DIR="${OUT_DIR:-artifacts/calibration/v2_lite_corpus}"
HEARTBEAT_PATH="${HEARTBEAT_PATH:-artifacts/calibration/heartbeat.json}"
STOP_FLAG_PATH="${STOP_FLAG_PATH:-artifacts/calibration/STOP}"
LOG_PATH="${LOG_PATH:-artifacts/calibration/overnight.log}"
PID_PATH="${PID_PATH:-artifacts/calibration/overnight.pid}"
VENV_PATH="${VENV_PATH:-.venv-calibration}"
EXIT_AFTER_HOURS="${EXIT_AFTER_HOURS:-0}"

mkdir -p "$(dirname "$HEARTBEAT_PATH")"

# Detach into background if requested via `&`; this script logs to LOG_PATH.
echo "$$" > "$PID_PATH"
START_EPOCH=$(date +%s)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_PATH"; }

write_heartbeat() {
    # $1 = status string, $2 = build pid or "-"
    local status="$1"
    local build_pid="$2"
    local n_shards
    n_shards=$(ls "$OUT_DIR"/shard_*.parquet 2>/dev/null | wc -l | tr -d ' ')
    local newest_shard_mtime
    newest_shard_mtime=$(ls -t "$OUT_DIR"/shard_*.parquet 2>/dev/null | head -1 | xargs -I{} stat -f%m {} 2>/dev/null || echo 0)
    local now
    now=$(date +%s)
    local uptime=$((now - START_EPOCH))
    cat > "$HEARTBEAT_PATH.tmp" <<EOF
{
  "ts": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "watchdog_pid": $$,
  "build_pid": "$build_pid",
  "status": "$status",
  "uptime_seconds": $uptime,
  "shards_complete": $n_shards,
  "max_sequences": $MAX_SEQUENCES,
  "shard_size": $SHARD_SIZE,
  "shards_target": $(( (MAX_SEQUENCES + SHARD_SIZE - 1) / SHARD_SIZE )),
  "newest_shard_mtime_epoch": $newest_shard_mtime,
  "out_dir": "$OUT_DIR",
  "log_path": "$LOG_PATH"
}
EOF
    mv "$HEARTBEAT_PATH.tmp" "$HEARTBEAT_PATH"
}

# Ensure the venv exists; if not, the user (or an agent session) needs
# to run Step 1 of the runbook first. Don't auto-pip-install here — that
# can be a multi-GB download and the user should consent.
if [ ! -d "$VENV_PATH" ]; then
    log "FATAL: venv $VENV_PATH not found. Run Step 1 of reports/overnight_calibration_corpus_runbook.md first."
    write_heartbeat "fatal_no_venv" "-"
    exit 1
fi

# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

log "watchdog start: MAX_SEQUENCES=$MAX_SEQUENCES MAX_TOKENS_PER_SEQ=$MAX_TOKENS_PER_SEQ DEVICE=$DEVICE"
write_heartbeat "starting" "-"

BACKOFF_SEC=10
MAX_BACKOFF_SEC=300
RESTARTS=0

while true; do
    if [ -f "$STOP_FLAG_PATH" ]; then
        log "STOP flag at $STOP_FLAG_PATH — exiting cleanly."
        write_heartbeat "stopped_by_user" "-"
        rm -f "$STOP_FLAG_PATH"
        exit 0
    fi

    if [ "$EXIT_AFTER_HOURS" != "0" ]; then
        ELAPSED_HOURS=$(( ( $(date +%s) - START_EPOCH ) / 3600 ))
        if [ "$ELAPSED_HOURS" -ge "$EXIT_AFTER_HOURS" ]; then
            log "Reached EXIT_AFTER_HOURS=$EXIT_AFTER_HOURS — exiting cleanly."
            write_heartbeat "stopped_time_limit" "-"
            exit 0
        fi
    fi

    log "launching build_corpus.py (restart #$RESTARTS)"
    write_heartbeat "launching" "-"

    # Wall-clock optimization #5 (2026-05-22): pass --skip-rows = number
    # of completed shards × shard size. On a fresh start this is 0; on
    # restart after N shards landed, we walk forward in the source dataset
    # instead of re-sampling row 0. Eliminates the duplicate-tail bug that
    # wasted ~60% of the 2026-05-21 corpus capture.
    N_SHARDS_DONE=$(ls "$OUT_DIR"/shard_*.parquet 2>/dev/null | wc -l | tr -d ' ')
    SKIP_ROWS=$((N_SHARDS_DONE * SHARD_SIZE))
    log "resume cursor: $N_SHARDS_DONE shards on disk → --skip-rows $SKIP_ROWS"

    # Run build_corpus.py and stream output to log.
    python3 tools/training/build_corpus.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --max-sequences "$MAX_SEQUENCES" \
        --max-tokens-per-seq "$MAX_TOKENS_PER_SEQ" \
        --shard-size "$SHARD_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --quantize-intermediates int8 \
        --skip-rows "$SKIP_ROWS" \
        --out "$OUT_DIR" \
        >> "$LOG_PATH" 2>&1 &
    BUILD_PID=$!
    log "build_corpus.py pid=$BUILD_PID"
    write_heartbeat "running" "$BUILD_PID"

    # Heartbeat-while-alive loop.
    while kill -0 "$BUILD_PID" 2>/dev/null; do
        sleep 30
        write_heartbeat "running" "$BUILD_PID"
        if [ -f "$STOP_FLAG_PATH" ]; then
            log "STOP flag during run — terminating build_pid=$BUILD_PID."
            kill "$BUILD_PID" 2>/dev/null || true
            sleep 5
            kill -9 "$BUILD_PID" 2>/dev/null || true
            write_heartbeat "stopped_by_user" "-"
            rm -f "$STOP_FLAG_PATH"
            exit 0
        fi
    done

    wait "$BUILD_PID" 2>/dev/null
    EXIT_CODE=$?
    log "build_corpus.py exited code=$EXIT_CODE"

    if [ "$EXIT_CODE" -eq 0 ]; then
        # Completed normally — likely all shards finished.
        log "Run completed cleanly. Heartbeat -> done."
        write_heartbeat "done" "-"
        exit 0
    fi

    # Crashed — back off and retry. --skip-existing means progress isn't lost.
    RESTARTS=$((RESTARTS + 1))
    write_heartbeat "crashed_backing_off" "-"
    log "crash #$RESTARTS — sleeping ${BACKOFF_SEC}s before restart"
    sleep "$BACKOFF_SEC"
    BACKOFF_SEC=$((BACKOFF_SEC * 2))
    if [ "$BACKOFF_SEC" -gt "$MAX_BACKOFF_SEC" ]; then
        BACKOFF_SEC="$MAX_BACKOFF_SEC"
    fi
done

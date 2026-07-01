#!/usr/bin/env bash
# tools/training/corpus_monitor.sh
#
# One-shot status check for the autonomous corpus build. The coding
# agent's monitor session calls this every 15–30 min (via /loop or manually)
# and reads the JSON-ish summary on stdout.
#
# Three classes of output:
#   - HEALTHY: shards increasing, watchdog + build PIDs alive, log
#     growing. No action needed.
#   - STALLED: watchdog alive but no new shard for > 2 hours. Likely
#     an MPS hang inside build_corpus.py. Monitor agent should kill
#     the build PID (watchdog will auto-restart).
#   - DEAD: watchdog process gone. Monitor agent should relaunch the
#     watchdog with `nohup ./tools/training/run_corpus_autonomous.sh &`.
#   - DONE: heartbeat status == "done". No action; report completion.
#
# Exit code: 0 = healthy, 1 = stalled, 2 = dead, 3 = done. Lets a
# /loop hook branch on $?.

set -uo pipefail
cd "$(dirname "$0")/../.."

HEARTBEAT_PATH="${HEARTBEAT_PATH:-artifacts/calibration/heartbeat.json}"
LOG_PATH="${LOG_PATH:-artifacts/calibration/overnight.log}"
PID_PATH="${PID_PATH:-artifacts/calibration/overnight.pid}"
OUT_DIR="${OUT_DIR:-artifacts/calibration/v2_lite_corpus}"
STALL_THRESHOLD_SEC="${STALL_THRESHOLD_SEC:-7200}"  # 2 hours

NOW=$(date +%s)

# --- Heartbeat ----
if [ ! -f "$HEARTBEAT_PATH" ]; then
    echo "{\"status\": \"no_heartbeat\", \"action\": \"relaunch_watchdog\"}"
    exit 2
fi

# Extract a few fields via python (jq may not be installed).
STATUS=$(python3 -c "import json; print(json.load(open('$HEARTBEAT_PATH'))['status'])" 2>/dev/null || echo "parse_error")
WD_PID=$(python3 -c "import json; print(json.load(open('$HEARTBEAT_PATH'))['watchdog_pid'])" 2>/dev/null || echo "0")
BUILD_PID=$(python3 -c "import json; print(json.load(open('$HEARTBEAT_PATH'))['build_pid'])" 2>/dev/null || echo "-")
HB_TS=$(python3 -c "import json; print(json.load(open('$HEARTBEAT_PATH'))['ts'])" 2>/dev/null || echo "-")
SHARDS=$(python3 -c "import json; print(json.load(open('$HEARTBEAT_PATH'))['shards_complete'])" 2>/dev/null || echo "0")
SHARDS_TARGET=$(python3 -c "import json; print(json.load(open('$HEARTBEAT_PATH'))['shards_target'])" 2>/dev/null || echo "0")
NEWEST_MTIME=$(python3 -c "import json; print(json.load(open('$HEARTBEAT_PATH'))['newest_shard_mtime_epoch'])" 2>/dev/null || echo "0")

# Watchdog alive?
WD_ALIVE="false"
if [ "$WD_PID" -gt 0 ] && kill -0 "$WD_PID" 2>/dev/null; then
    WD_ALIVE="true"
fi

# Build alive?
BUILD_ALIVE="false"
if [ "$BUILD_PID" != "-" ] && [ "$BUILD_PID" != "0" ] && kill -0 "$BUILD_PID" 2>/dev/null; then
    BUILD_ALIVE="true"
fi

# Time since last shard
if [ "$NEWEST_MTIME" -gt 0 ]; then
    SECS_SINCE_LAST_SHARD=$((NOW - NEWEST_MTIME))
else
    SECS_SINCE_LAST_SHARD=-1
fi

# Disk
FREE_GB=$(df -g . | awk 'NR==2 {print $4}')
CORPUS_BYTES=$(du -sk "$OUT_DIR" 2>/dev/null | awk '{print $1 * 1024}')

# Log progress: last 5 non-empty lines
LOG_TAIL=$(tail -10 "$LOG_PATH" 2>/dev/null | grep -v '^$' | tail -5 | sed 's/"/\\"/g' | tr '\n' '|' || echo "")

# --- Classify ---
ACTION="none"
EXIT_CODE=0

if [ "$STATUS" = "done" ]; then
    ACTION="report_completion"
    EXIT_CODE=3
elif [ "$STATUS" = "fatal_no_venv" ]; then
    # Watchdog can't help — Step 1 of the runbook (pip install) is the
    # blocker. Don't loop-relaunch; tell the operator.
    ACTION="human_setup_required_step_1"
    EXIT_CODE=2
elif [ "$WD_ALIVE" = "false" ]; then
    ACTION="relaunch_watchdog"
    EXIT_CODE=2
elif [ "$BUILD_ALIVE" = "false" ] && [ "$STATUS" = "running" ]; then
    # Stale heartbeat — watchdog should self-recover; if not, escalate.
    ACTION="watch_next_tick"
    EXIT_CODE=1
elif [ "$SECS_SINCE_LAST_SHARD" -gt "$STALL_THRESHOLD_SEC" ] && [ "$BUILD_ALIVE" = "true" ]; then
    ACTION="kill_build_let_watchdog_restart"
    EXIT_CODE=1
fi

cat <<EOF
{
  "status": "$STATUS",
  "watchdog_pid": $WD_PID,
  "watchdog_alive": $WD_ALIVE,
  "build_pid": "$BUILD_PID",
  "build_alive": $BUILD_ALIVE,
  "heartbeat_ts": "$HB_TS",
  "shards_complete": $SHARDS,
  "shards_target": $SHARDS_TARGET,
  "percent_complete": $(python3 -c "print(round(100 * $SHARDS / max($SHARDS_TARGET, 1), 1))" 2>/dev/null || echo "0"),
  "secs_since_last_shard": $SECS_SINCE_LAST_SHARD,
  "stall_threshold_sec": $STALL_THRESHOLD_SEC,
  "free_disk_gb": $FREE_GB,
  "corpus_bytes": ${CORPUS_BYTES:-0},
  "action": "$ACTION",
  "log_tail": "$LOG_TAIL"
}
EOF

exit $EXIT_CODE

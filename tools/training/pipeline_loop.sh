#!/usr/bin/env bash
# pipeline_loop.sh — calls advance_pipeline.sh every N seconds.
#
# Run via:
#   nohup tools/training/pipeline_loop.sh > training_data/c2_hidden/eagle3_v0/pipeline/loop.log 2>&1 < /dev/null &
#   disown
#
# Or once per laptop wake via launchd:
#   cp tools/training/com.user.dismantle.pipeline.plist ~/Library/LaunchAgents/
#   launchctl load ~/Library/LaunchAgents/com.user.dismantle.pipeline.plist
#
# Stops when:
#   - ALL_DONE marker is created by advance_pipeline.sh
#   - HALT marker is touched manually
#   - advance_pipeline.sh exits with code 2 (stage failure)

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

PIPELINE_DIR="${PIPELINE_DIR:-training_data/c2_hidden/eagle3_v0/pipeline}"
INTERVAL="${INTERVAL:-60}"  # seconds between checks
M_ALL="$PIPELINE_DIR/ALL_DONE"
M_HALT="$PIPELINE_DIR/HALT"
ADVANCE="$PROJECT_ROOT/tools/training/advance_pipeline.sh"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s loop] %s\n' "$(ts)" "$*"; }

log "starting; interval=${INTERVAL}s; advance=$ADVANCE"

while true; do
  if [ -f "$M_ALL" ]; then
    log "ALL_DONE present; exiting loop"
    exit 0
  fi
  if [ -f "$M_HALT" ]; then
    log "HALT present; exiting loop"
    exit 0
  fi
  bash "$ADVANCE"
  rc=$?
  if [ "$rc" -eq 2 ]; then
    log "advance_pipeline.sh failed with stage error; exiting loop"
    exit 2
  fi
  sleep "$INTERVAL"
done

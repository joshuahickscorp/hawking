#!/usr/bin/env bash
# Persistent supervisor for the 7B frontier ladder.
#
# The ladder itself is detached and checkpointed. This outer loop handles the
# practical overnight failure mode: if the ladder exits before the frontier is
# actually exhausted, plant an autopilot inject if possible and relaunch.
set -uo pipefail
cd "$HOME/Downloads/hawking" || exit 2

RUN="${RUN:-7b_frontier}"
INTERVAL="${KEEPALIVE_INTERVAL:-300}"
LOCK="reports/cron/${RUN}.lock"
HEALTH_PID="reports/cron/${RUN}_health.pid"
CAFFEINATE_PID="reports/cron/${RUN}_caffeinate.pid"
KEEPALIVE_PID="reports/cron/${RUN}_keepalive.pid"
CONDUCTOR_PID="reports/cron/${RUN}_conductor.pid"
RUNLOG="reports/cron/${RUN}_run.log"
INJECT="reports/cron/${RUN}_inject.py"
LOG="reports/cron/${RUN}_keepalive.log"

mkdir -p reports/cron
echo "$$" > "$KEEPALIVE_PID"

cleanup_pid() {
  if [ -f "$KEEPALIVE_PID" ] && [ "$(cat "$KEEPALIVE_PID" 2>/dev/null)" = "$$" ]; then
    rm -f "$KEEPALIVE_PID"
  fi
}
trap cleanup_pid EXIT

live_pid() {
  local file="$1"
  [ -f "$file" ] || return 0
  local p
  p=$(cat "$file" 2>/dev/null)
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"
}

frontier_live() { live_pid "$LOCK"; }
health_live() { live_pid "$HEALTH_PID"; }
caffeinate_live() { live_pid "$CAFFEINATE_PID"; }
conductor_live() { live_pid "$CONDUCTOR_PID"; }
run_done_marker() { tail -80 "$RUNLOG" 2>/dev/null | grep -q '# done ->'; }

echo "# keepalive start $(date) interval=${INTERVAL}s" >> "$LOG"
while true; do
  if [ -n "$(frontier_live)" ]; then
    if [ -z "$(health_live)" ] || [ -z "$(caffeinate_live)" ] || [ -z "$(conductor_live)" ]; then
      echo "$(date '+%H:%M:%S') live frontier but missing health/caffeinate/conductor; adopting" >> "$LOG"
      ./run_7b_frontier.sh launch >> "$LOG" 2>&1 || true
    fi
  else
    python3.12 tools/condense/frontier_autopilot.py \
      --outbase "reports/cron/${RUN}" --emit-inject >> "$LOG" 2>&1 || true
    if run_done_marker && [ ! -f "$INJECT" ]; then
      echo "$(date '+%H:%M:%S') frontier done and autopilot has no inject; keepalive exit" >> "$LOG"
      break
    fi
    echo "$(date '+%H:%M:%S') frontier not live; relaunching" >> "$LOG"
    ./run_7b_frontier.sh launch >> "$LOG" 2>&1 || true
  fi
  sleep "$INTERVAL"
done

#!/bin/bash
# Supervisor wrapper for the resident Genesis body + optimization loop. launchd
# restarts this on crash and at login; this script handles what launchd does not:
# log growth, a stale lock left by a killed run, and coordinated child shutdown.
#
#   tools/genesis_forever.sh            # run (launchd calls this)
#   touch workspace/ops/GENESIS_STOP    # halt after the current tick
#   rm     workspace/ops/GENESIS_STOP   # resume
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO/workspace/ops/ascent-daemon.log"
RESIDENT_LOG="$REPO/workspace/ops/genesis-resident.log"
STOP="$REPO/workspace/ops/GENESIS_STOP"
LOCK=/tmp/hawking-gpu-lane.lock
RESIDENT_CLIENT="$REPO/tools/agentos/genesis_resident.py"

mkdir -p "$(dirname "$LOG")"

# Rotate before appending: an unattended loop writes a JSON line per tick forever,
# and a multi-GB log on a box that has hit 0 bytes free twice is a real hazard.
if [ -f "$LOG" ] && [ "$(stat -f %z "$LOG" 2>/dev/null || echo 0)" -gt 52428800 ]; then
  mv -f "$LOG" "$LOG.1"
fi

# A killed run can leave the GPU lock held by a pid that no longer exists. The lock
# script self-heals on a dead pid, but only for a lock that HAS a pid file.
if [ -d "$LOCK" ] && [ ! -f "$LOCK/pid" ]; then
  rm -rf "$LOCK"
elif [ -f "$LOCK/pid" ] && ! kill -0 "$(cat "$LOCK/pid" 2>/dev/null)" 2>/dev/null; then
  rm -rf "$LOCK"
fi

if [ -f "$STOP" ]; then
  echo "{\"stopfile\": true, \"note\": \"GENESIS_STOP present, not starting\"}" >> "$LOG"
  exit 0
fi

# Seat the lineage if it is not seated. Never zero valid Genesis, including after a
# reboot that wiped nothing but started us from cold.
if [ ! -f "$REPO/receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json" ]; then
  /usr/bin/env python3 "$REPO/tools/genesis_seat.py" seat >> "$LOG" 2>&1
fi

resident_pid=""
daemon_pid=""
resident_health_failed_at=0

cleanup() {
  trap - EXIT INT TERM
  if [ -n "$daemon_pid" ] && kill -0 "$daemon_pid" 2>/dev/null; then
    kill -TERM "$daemon_pid" 2>/dev/null || true
    wait "$daemon_pid" 2>/dev/null || true
  fi
  if [ -n "$resident_pid" ] && kill -0 "$resident_pid" 2>/dev/null; then
    /usr/bin/env python3 "$RESIDENT_CLIENT" stop >/dev/null 2>&1 || \
      kill -TERM "$resident_pid" 2>/dev/null || true
    wait "$resident_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_resident() {
  # Artifact, tokenizer, and executable are selected by the seated CURRENT
  # identity inside genesis_resident.py.  Do not pin G0 paths here: after a
  # protected runtime/representation handoff that would restart a healthy
  # process against the wrong generation.
  /usr/bin/env python3 "$RESIDENT_CLIENT" serve \
    --repo "$REPO" \
    --lineage "$REPO/receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json" \
    --max-seq-len 8192 >> "$RESIDENT_LOG" 2>&1 &
  resident_pid=$!

  attempts=0
  while ! /usr/bin/env python3 "$RESIDENT_CLIENT" health >/dev/null 2>&1; do
    if [ -f "$STOP" ]; then
      return 1
    fi
    if ! kill -0 "$resident_pid" 2>/dev/null; then
      echo '{"resident": "failed_before_health"}' >> "$LOG"
      resident_pid=""
      return 1
    fi
    attempts=$((attempts + 1))
    if [ "$attempts" -ge 600 ]; then
      echo '{"resident": "health_timeout_300s"}' >> "$LOG"
      kill -TERM "$resident_pid" 2>/dev/null || true
      wait "$resident_pid" 2>/dev/null || true
      resident_pid=""
      return 1
    fi
    sleep 0.5
  done
  return 0
}

# The optimization daemon must never fall back to a cold, separate model body in
# normal operation. Bring the one resident parent up first and prove it answers.
if ! /usr/bin/env python3 "$RESIDENT_CLIENT" health >/dev/null 2>&1; then
  if ! start_resident; then
    exit 1
  fi
fi

/usr/bin/env python3 "$REPO/tools/ascent_daemon.py" loop >> "$LOG" 2>&1 &
daemon_pid=$!

while true; do
  if [ -f "$STOP" ]; then
    exit 0
  fi
  if ! kill -0 "$daemon_pid" 2>/dev/null; then
    wait "$daemon_pid" 2>/dev/null
    exit $?
  fi
  if /usr/bin/env python3 "$RESIDENT_CLIENT" health >/dev/null 2>&1; then
    if [ "$resident_health_failed_at" -ne 0 ]; then
      echo '{"resident": "health_recovered_after_busy"}' >> "$LOG"
      resident_health_failed_at=0
    fi
  elif [ -n "$resident_pid" ] && kill -0 "$resident_pid" 2>/dev/null; then
    # The body currently serves requests serially. A long proposal makes the
    # health socket temporarily unavailable even though the same resident PID is
    # alive and owns the GPU work. Do not turn a busy body into a cold restart.
    now_s=$(date +%s)
    if [ "$resident_health_failed_at" -eq 0 ]; then
      resident_health_failed_at=$now_s
      echo '{"resident": "health_busy_process_alive"}' >> "$LOG"
    elif [ $((now_s - resident_health_failed_at)) -ge 3600 ]; then
      echo '{"resident": "health_unavailable_3600s_process_alive"}' >> "$LOG"
      exit 1
    fi
  else
    # A lifecycle runtime handoff stops the old resident only after durable
    # worker rebind. Keep the optimization daemon alive while this supervisor
    # starts the executable/artifact named by the new CURRENT. A failed child
    # start is retried rather than tearing down the controller that can roll
    # state back to LAST_KNOWN_GOOD.
    echo '{"resident": "lost_while_daemon_live_restarting"}' >> "$LOG"
    resident_pid=""
    if ! start_resident; then
      echo '{"resident": "restart_failed_waiting_for_lineage_rollback"}' >> "$LOG"
      sleep 2
    fi
  fi
  sleep 5
done

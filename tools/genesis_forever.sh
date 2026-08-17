#!/bin/bash
# Supervisor wrapper for the Genesis loop. launchd restarts this on crash and at
# login; this script handles what launchd does not: log growth, a stale lock left
# by a killed run, and a stopfile so the loop can be halted without touching launchd.
#
#   tools/genesis_forever.sh            # run (launchd calls this)
#   touch workspace/ops/GENESIS_STOP    # halt after the current tick
#   rm     workspace/ops/GENESIS_STOP   # resume
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO/workspace/ops/ascent-daemon.log"
STOP="$REPO/workspace/ops/GENESIS_STOP"
LOCK=/tmp/hawking-gpu-lane.lock

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

exec /usr/bin/env python3 "$REPO/tools/ascent_daemon.py" loop >> "$LOG" 2>&1

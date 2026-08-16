#!/bin/bash
# Serializes GPU/memory-exclusive benchmark runs across parallel Grok lanes.
# Usage: gpu_lane_lock.sh <lane-name> <command...>
# ponytail: mkdir-atomic lock, no fairness/queueing. Swap for flock if lane count grows.
LOCK=/tmp/hawking-gpu-lane.lock
NAME="$1"; shift
DEADLINE=$(( $(date +%s) + 5400 ))
until mkdir "$LOCK" 2>/dev/null; do
  if [ -f "$LOCK/pid" ] && ! kill -0 "$(cat "$LOCK/pid")" 2>/dev/null; then
    rm -rf "$LOCK"; continue
  fi
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "gpu_lane_lock: timeout waiting, held by $(cat "$LOCK/owner" 2>/dev/null)" >&2; exit 75; }
  sleep 5
done
echo $$ > "$LOCK/pid"; echo "$NAME" > "$LOCK/owner"
trap 'rm -rf "$LOCK"' EXIT INT TERM
"$@"

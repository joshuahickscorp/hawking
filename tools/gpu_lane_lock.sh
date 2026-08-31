#!/bin/bash
# Serializes GPU/memory-exclusive benchmark runs across parallel Grok lanes.
# Usage: gpu_lane_lock.sh <lane-name> <command...>
# ponytail: mkdir-atomic lock, no fairness/queueing. Swap for flock if lane count grows.
LOCK=${HAWKING_GPU_LANE_LOCK:-/tmp/hawking-gpu-lane.lock}
NAME="$1"; shift
DEADLINE=$(( $(date +%s) + 5400 ))
until mkdir "$LOCK" 2>/dev/null; do
  # A NON-DIRECTORY at the lock path is not a lock, it is a wedge. mkdir can
  # never succeed against an existing regular file, and the stale-owner branch
  # below tests "$LOCK/pid" which is unreachable when $LOCK is a file - so the
  # rm -rf that clears a dead owner could never fire and every caller spun to
  # its deadline and exited 75. This has now happened TWICE on this box: a
  # 0-byte file appeared at 05:58, was cleared at 06:21, and another appeared at
  # 08:41. "Nothing in this repo creates it as a file" was WRONG when written:
  # test_gpu_lane_lock_wedge.sh planted the wedge at this very path to prove the
  # branch below works, and rm -rf'd the path first - so the test could destroy a
  # live lane's lock, and an interrupted run left the wedge behind. It now drives
  # HAWKING_GPU_LANE_LOCK at a temp path instead. That accounts for 08:41; 05:58
  # predates the test and stays unexplained, which is why the reclaim stays.
  # A path that is not a directory cannot be holding anything: remove it, retry.
  if [ -e "$LOCK" ] && [ ! -d "$LOCK" ]; then
    echo "gpu_lane_lock: reclaiming non-directory at $LOCK (wedge, not a lock)" >&2
    rm -f "$LOCK"; continue
  fi
  if [ -f "$LOCK/pid" ] && ! kill -0 "$(cat "$LOCK/pid")" 2>/dev/null; then
    rm -rf "$LOCK"; continue
  fi
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "gpu_lane_lock: timeout waiting, held by $(cat "$LOCK/owner" 2>/dev/null)" >&2; exit 75; }
  sleep 5
done
echo $$ > "$LOCK/pid"; echo "$NAME" > "$LOCK/owner"
trap 'rm -rf "$LOCK"' EXIT INT TERM
"$@"

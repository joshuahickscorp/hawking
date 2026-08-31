#!/bin/bash
# The GPU lane lock must reclaim a non-directory at its path.
#
# A 0-byte regular file appeared at /tmp/hawking-gpu-lane.lock twice on this box
# (05:58, cleared 06:21; again at 08:41). mkdir can never succeed against an
# existing file, and the stale-owner branch tests "$LOCK/pid" - unreachable when
# the lock path is a file - so the rm -rf that clears a dead owner could never fire.
# Every caller spun to its 5400s deadline and exited 75, which means GPU lanes
# were either timing out or running unserialised.
set -u
# NEVER touch the production lock path. This test plants a wedge and rm -rf's the
# path; pointed at the production lock it could destroy a live lane's lock
# and, if interrupted, leave behind the exact 0-byte wedge it exists to reclaim.
export HAWKING_GPU_LANE_LOCK="${TMPDIR:-/tmp}/gpu-lane-lock-wedgetest.$$"
LOCK="$HAWKING_GPU_LANE_LOCK"
trap 'rm -rf "$LOCK"' EXIT
LOCK_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/tools/gpu_lane_lock.sh"
fail() { echo "FAIL: $1" >&2; exit 1; }

# 1. a wedge (regular file) is reclaimed and the command still runs
rm -rf "$LOCK"
: > "$LOCK"
out="$("$LOCK_SCRIPT" wedgetest /bin/echo ran 2>/dev/null)" || fail "wedge not reclaimed"
[ "$out" = "ran" ] || fail "command did not run after reclaim (got: $out)"

# 2. the lock is released afterwards
[ -e "$LOCK" ] && fail "lock not released after exit"

# 3. a LIVE holder is still respected - the wedge branch must not steal a real lock
mkdir "$LOCK"
echo $$ > "$LOCK"/pid
echo livetest > "$LOCK"/owner
timeout 8 "$LOCK_SCRIPT" intruder /bin/echo stolen >/dev/null 2>&1 && fail "a live lock was stolen"
[ -d "$LOCK" ] || fail "live lock was destroyed"
rm -rf "$LOCK"

echo "ok: wedge reclaimed, lock released, live holder respected"

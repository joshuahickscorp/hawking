#!/usr/bin/env bash
# Odyssey-I RESIDENT driver: loops continuously — reap finished lanes, launch the
# next rungs, keep the box working. Not a 15-min cron burst; a resident that ticks
# every TICK_SECS so there is (almost) always a model runner PID in flight while
# the deterministic descent-ladder well has work.
# Launchd template (KeepAlive): workspace/campaign/odyssey/com.hawking.odyssey.plist
# Do not load that plist from this script.
set -uo pipefail   # not -e: one bad tick must never kill the resident

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# launchd runs with a minimal environment: restore HOME + a full PATH so git/date/
# python3 resolve. (grok is intentionally NOT in the loop, but keep PATH complete.)
export HOME="${HOME:-/Users/scammermike}"
export PATH="$HOME/.grok/bin:$HOME/.claude-grok/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# Drive the box HARD into the user-set swap budget for full utilization: smaller
# OS reserve + lower page-starve floor let memgate keep more lanes in flight so
# RAM stays near-full instead of sawtoothing. SWAP_MAX_GIB (30) stays the hard
# cap that prevents OOM — memgate refuses once projected swap approaches it.
export ODYSSEY_RESERVE_GIB="${ODYSSEY_RESERVE_GIB:-6}"
export ODYSSEY_FREE_RAM_RESERVE_GIB="${ODYSSEY_FREE_RAM_RESERVE_GIB:-1}"

PY="${ODYSSEY_PYTHON:-/Library/Frameworks/Python.framework/Versions/3.12/bin/python3}"
if [ ! -x "$PY" ]; then
  PY=python3
fi

LOG="$ROOT/workspace/campaign/odyssey/driver.log"
mkdir -p "$(dirname "$LOG")"

WINDOW_SECS="${ODYSSEY_WINDOW_SECS:-300}"  # tight-loop window per process (~5 min), commit after
INNER_SLEEP="${ODYSSEY_INNER_SLEEP:-3}"    # seconds between reap+refill inside the tight loop
# Single-resident lock: if another resident holds it, exit (launchd KeepAlive keeps one).
LOCK="$ROOT/workspace/campaign/odyssey/.resident.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  # stale lock if owner PID is gone
  if [ -f "$LOCK/pid" ] && ! kill -0 "$(cat "$LOCK/pid" 2>/dev/null)" 2>/dev/null; then
    rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null || exit 0
  else
    echo "$(date '+%FT%T%z') resident already running; exit" >>"$LOG"; exit 0
  fi
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT INT TERM

commit_data() {
  git add -- receipts/odyssey-i \
    workspace/campaign/odyssey/ODYSSEY_COMPLETIONS.json \
    workspace/campaign/odyssey/ODYSSEY_STATE.json \
    workspace/campaign/odyssey/RUN_LOG.jsonl \
    workspace/campaign/odyssey/TRANSFER_MATRIX.json \
    workspace/campaign/odyssey/GRAVITY_RULEBASE.json \
    workspace/campaign/odyssey/NEGATIVE_SCIENCE.json \
    workspace/campaign/odyssey/patients >/dev/null 2>&1 || true
  git commit -q -m "odyssey-i resident: autonomous cycle $(date '+%FT%T%z')" >/dev/null 2>&1 \
    && echo "-- committed data --" || true
}

echo "== odyssey RESIDENT up $(date '+%FT%T%z') pid=$$ window=${WINDOW_SECS}s inner=${INNER_SLEEP}s ==" >>"$LOG"
tick=0
while true; do
  tick=$((tick+1))
  ts="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  {
    echo "== window $tick $ts (tight ${WINDOW_SECS}s loop) =="
    # TIGHT internal loop: one process reaps + refills lanes every INNER_SLEEP
    # for WINDOW_SECS, so lanes stay CONSTANTLY full (no per-tick startup, no
    # gap between waves). --max-lanes = ceiling; memgate (RAM) + disk are the
    # real limits. --grok-lanes 0 = deterministic science only.
    ODYSSEY_HEADROOM_ADMIT=1 "$PY" tools/odyssey_ctl.py cycle --go \
      --max-lanes 14 --grok-lanes 0 \
      --loop-secs "$WINDOW_SECS" --inner-sleep "$INNER_SLEEP"
    echo "-- window rc=$? --"
  } >>"$LOG" 2>&1
  commit_data >>"$LOG" 2>&1
done

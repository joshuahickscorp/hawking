#!/usr/bin/env bash
# ============================================================================
#  7B quality-ladder — fire-and-forget overnight runner.
#
#  Run this ONCE, then close the terminal AND Claude. Everything below reparents
#  to launchd (PPID 1) and survives disconnect — closing Claude actually HELPS
#  (frees ~3GB RAM for the baker).
#
#      ./run_7b_ladder.sh            # launch / resume the ladder + arm the backstop
#      ./run_7b_ladder.sh status     # how far it's got (safe to run anytime)
#      ./run_7b_ladder.sh kill       # stop everything
#
#  It is IDEMPOTENT: if a ladder is already running it adopts it (won't restart
#  and lose the in-flight bake); after a reboot it relaunches from the last
#  checkpoint. Results stream to reports/cron/7b_ladder.jsonl as each config
#  finishes (f16 + 4/3/2/1-AWQ + 1-RHT + residuals).
#
#  Memory/disk safety (this is a 7B on a 19GB Mac — the edge):
#    * the baker is fed ONE chunk of tensors at a time (BAKE_CHUNKS=8) so it
#      never accumulates the full 28GB F32 recon that was OOM-killing it;
#    * a detached watchdog SIGTERMs the baker if free disk ever drops < 6GB
#      (macOS dynamic swap collapses the instant the baker dies), then the
#      ladder's own guard logs that config as skipped and CONTINUES.
#  Nothing here can hard-OOM the machine; worst case a single config is skipped.
# ============================================================================
set -uo pipefail
cd "$HOME/Downloads/hawking" || { echo "repo not found at ~/Downloads/hawking"; exit 2; }
PY=python3.12
LOCK=reports/cron/7b_ladder.lock
JSONL=reports/cron/7b_ladder.jsonl
HEALTH=reports/cron/7b_ladder_health.log
mkdir -p reports/cron

cmd="${1:-launch}"

_live_ladder() {   # echo PID if a ladder process holds a live lock, else nothing
  [ -f "$LOCK" ] || return 0
  local p; p=$(cat "$LOCK" 2>/dev/null)
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"
}

case "$cmd" in
  status)
    p=$(_live_ladder)
    echo "ladder: ${p:-not running}"
    [ -n "$(pgrep -f ladder_health.sh)" ] && echo "backstop: armed" || echo "backstop: down"
    if [ -f "$JSONL" ]; then
      echo "--- results so far ---"
      $PY - "$JSONL" <<'PY'
import json,sys
for ln in open(sys.argv[1]):
    try:
        r=json.loads(ln)
    except: continue
    if "ppl" in r: print(f"  {r['config']:9s} {r.get('eff_bpw','?'):>6} bpw   +{r.get('degr_pct','?')}%  vs f16")
    elif "error" in r: print(f"  {r['config']:9s} (retry) {r['error'][:60]}")
PY
    fi
    [ -f "$HEALTH" ] && { echo "--- disk/swap (last) ---"; tail -1 "$HEALTH"; }
    ;;

  kill)
    $PY tools/condense/ladder_launch.py kill 2>/dev/null
    pkill -f ladder_health.sh 2>/dev/null
    pkill -9 -f quantize-model 2>/dev/null
    echo "stopped ladder + backstop + any baker"
    ;;

  launch|*)
    p=$(_live_ladder)
    if [ -n "$p" ]; then
      echo "ladder already running (PID $p) — adopting it, not restarting (keeps the in-flight bake)"
    else
      echo "launching ladder (detached, resumes from checkpoint)…"
      $PY tools/condense/ladder_launch.py
    fi
    if [ -n "$(pgrep -f ladder_health.sh)" ]; then
      echo "backstop already armed"
    else
      echo "arming OOM/disk backstop (detached)…"
      nohup bash tools/condense/ladder_health.sh >/dev/null 2>&1 &
      disown 2>/dev/null || true
    fi
    sleep 12            # let the ladder finish importing torch + write its lock before status
    echo ""
    "$0" status
    echo ""
    echo "✅ Safe to close this terminal AND Claude now — both reparent to launchd."
    echo "   Check anytime:  ./run_7b_ladder.sh status"
    ;;
esac

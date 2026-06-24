#!/usr/bin/env bash
# ============================================================================
#  7B FRONTIER — comprehensive 4/3/2/1-bit local sweep, fire-and-forget.
#
#  Extends the first ladder across the whole bit-frontier:
#    * mixed-precision (attn-high / FFN-low) — the measured density lever
#    * 3-bit alpha sweep (AWQ .25/.5/.75 + RHT) — push the usable floor
#    * 2-bit rescue attempts (alpha + RHT) — can anything below 3-bit be saved?
#    * 4-RHT for completeness
#  The 6 already-measured results (f16, 4/3/2/1-AWQ, 1-RHT) are SEEDED so they
#  skip — only the ~10 new configs run. Long run (~10-15h) → detached; you close
#  Claude, I sample/update and watch for optimizations.
#
#      ./run_7b_frontier.sh            # launch / resume + arm backstop
#      ./run_7b_frontier.sh status     # results so far (run anytime)
#      ./run_7b_frontier.sh kill       # stop everything
#
#  Same OOM control as the first ladder: chunked baking (RAM bounded), per-config
#  disk guard, detached watchdog that kills the baker if free disk < 6GB, and
#  caffeinate so the machine doesn't sleep. Nothing here can hard-OOM the machine.
# ============================================================================
set -uo pipefail
cd "$HOME/Downloads/hawking" || { echo "repo not found at ~/Downloads/hawking"; exit 2; }
PY=python3.12
RUN=7b_frontier
LOCK=reports/cron/$RUN.lock
JSONL=reports/cron/$RUN.jsonl
HEALTH=reports/cron/${RUN}_health.log
SEED=reports/cron/7b_ladder.jsonl          # the first ladder's results to carry over
mkdir -p reports/cron
export LADDER_MODEL="$HOME/Downloads/hawking/scratch/qwen-7b"
export LADDER_LABEL=7B LADDER_SET=frontier LADDER_OUT=reports/cron/$RUN
export LADDER_LOCK=$LOCK LADDER_HEALTH=$HEALTH
export DOCTOR_DEVICE=cpu DOCTOR_DTYPE=bfloat16 PYTHONUNBUFFERED=1 STRAND_NO_GPU=1
export BAKE_CHUNKS=8 BAKE_THREADS=8

cmd="${1:-launch}"

_live() { [ -f "$LOCK" ] || return 0; local p; p=$(cat "$LOCK" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"; }

_seed() {   # carry over the already-measured configs so they skip (only if no progress yet)
  [ -f "$JSONL" ] && return 0
  [ -f "$SEED" ] || return 0
  $PY - "$SEED" "$JSONL" <<'PY'
import json,sys
keep={"f16","4-AWQ","3-AWQ","2-AWQ","1-AWQ","1-RHT"}
out=open(sys.argv[2],"w")
for ln in open(sys.argv[1]):
    try: r=json.loads(ln)
    except: continue
    if r.get("config") in keep and "ppl" in r: out.write(json.dumps(r)+"\n")
out.close()
PY
  echo "seeded $(wc -l < "$JSONL" | tr -d ' ') known results into $JSONL (they will skip)"
}

case "$cmd" in
  status)
    p=$(_live); echo "frontier: ${p:-not running}"
    [ -n "$(pgrep -f ladder_health.sh)" ] && echo "backstop: armed" || echo "backstop: down"
    if [ -f "$JSONL" ]; then echo "--- results so far ---"; $PY - "$JSONL" <<'PY'
import json,sys
for ln in open(sys.argv[1]):
    try: r=json.loads(ln)
    except: continue
    if "ppl" in r: print(f"  {r['config']:9s} {r.get('eff_bpw','?'):>6} bpw   +{r.get('degr_pct','?')}%")
    elif "error" in r: print(f"  {r['config']:9s} (retry) {r['error'][:55]}")
PY
    fi
    [ -f "$HEALTH" ] && { echo "--- disk/swap (last) ---"; tail -1 "$HEALTH"; } ;;

  kill)
    $PY tools/condense/ladder_launch.py kill 2>/dev/null
    pkill -f ladder_health.sh 2>/dev/null; pkill -9 -f quantize-model 2>/dev/null
    echo "stopped frontier + backstop + baker" ;;

  launch|*)
    _seed
    p=$(_live)
    if [ -n "$p" ]; then echo "frontier already running (PID $p) — adopting, not restarting"
    else echo "launching frontier (detached, resumes from checkpoint)…"; $PY tools/condense/ladder_launch.py; fi
    if [ -n "$(pgrep -f ladder_health.sh)" ]; then echo "backstop already armed"
    else echo "arming OOM/disk backstop…"; nohup bash tools/condense/ladder_health.sh >/dev/null 2>&1 & disown 2>/dev/null || true; fi
    # keep the machine awake until the run's PID exits
    rp=$(cat reports/cron/$RUN.pid 2>/dev/null)
    [ -n "$rp" ] && { nohup caffeinate -is -w "$rp" >/dev/null 2>&1 & disown 2>/dev/null || true; }
    sleep 12; echo ""; "$0" status
    echo ""; echo "✅ Safe to close terminal + Claude — reparents to launchd, stays awake on AC."
    echo "   Check anytime:  ./run_7b_frontier.sh status" ;;
esac

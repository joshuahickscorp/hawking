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
HEALTH_PID=reports/cron/${RUN}_health.pid
CAFFEINATE_PID=reports/cron/${RUN}_caffeinate.pid
KEEPALIVE_PID=reports/cron/${RUN}_keepalive.pid
CONDUCTOR_PID=reports/cron/${RUN}_conductor.pid
VERIFIER_PID=reports/cron/${RUN}_verifier.pid
SEED=reports/cron/7b_ladder.jsonl          # the first ladder's results to carry over
mkdir -p reports/cron
export LADDER_MODEL="$HOME/Downloads/hawking/scratch/qwen-7b"
export LADDER_LABEL=7B LADDER_SET=frontier LADDER_OUT=reports/cron/$RUN
export LADDER_LOCK=$LOCK LADDER_HEALTH=$HEALTH LADDER_HEALTH_PID=$HEALTH_PID
export LADDER_RUN_LOG=reports/cron/${RUN}_run.log
export LADDER_SIGMA=reports/cron/${RUN}_sigma.safetensors
export DOCTOR_DEVICE=cpu DOCTOR_DTYPE=bfloat16 PYTHONUNBUFFERED=1 STRAND_NO_GPU=1
export DOCTOR_THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 8)
export DOCTOR_GRAD_ACCUM=4
# engage all cores (P+E) for CPU math — user keeps the machine free for this run.
# BLAS backends each gate on their own env var; set them all to the logical core count.
NCPU=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 8)
export OMP_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU \
       VECLIB_MAXIMUM_THREADS=$NCPU NUMEXPR_NUM_THREADS=$NCPU
export DOCTOR_SAVE_MODE=adapter
export DOCTOR_TIMEOUT=${DOCTOR_TIMEOUT:-28800}
export DOCTOR_SWAP_CEIL=${DOCTOR_SWAP_CEIL:-12000}
export DOCTOR_SWAP_HARD_CEIL=${DOCTOR_SWAP_HARD_CEIL:-18000}
export DOCTOR_TERMINATE_GRACE=${DOCTOR_TERMINATE_GRACE:-600}
export DOCTOR_USE_PARTIAL=${DOCTOR_USE_PARTIAL:-1}
export KD_TOPK=${KD_TOPK:-64}
export BAKE_CHUNKS=8 BAKE_THREADS=$NCPU
PT=/tmp/ppl24k.txt
[ -f "$PT" ] || cat README.md docs/plans/*.md 2>/dev/null | head -c 24000 > "$PT"
export PPL_TEXT=$PT

cmd="${1:-launch}"

_live() { [ -f "$LOCK" ] || return 0; local p; p=$(cat "$LOCK" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"; }
_live_health() { [ -f "$HEALTH_PID" ] || return 0; local p; p=$(cat "$HEALTH_PID" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"; }
_live_caffeinate() { [ -f "$CAFFEINATE_PID" ] || return 0; local p; p=$(cat "$CAFFEINATE_PID" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"; }
_live_keepalive() { [ -f "$KEEPALIVE_PID" ] || return 0; local p; p=$(cat "$KEEPALIVE_PID" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"; }
_live_conductor() { [ -f "$CONDUCTOR_PID" ] || return 0; local p; p=$(cat "$CONDUCTOR_PID" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"; }
_live_verifier() { p=$(pgrep -f frontier_verifier.py | head -1); [ -n "$p" ] && echo "$p"; }
_arm_health() {
  $PY - <<'PY'
import os, subprocess
env = os.environ.copy()
subprocess.Popen(
    ["bash", "tools/condense/ladder_health.sh"],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    env=env, start_new_session=True,
)
PY
}
_arm_caffeinate() {
  local rp="$1"
  [ -n "$rp" ] || return 0
  RUN="$RUN" RP="$rp" $PY - <<'PY'
import os, pathlib, subprocess
run = os.environ["RUN"]
rp = os.environ["RP"]
root = pathlib.Path("reports/cron")
root.mkdir(parents=True, exist_ok=True)
log = open(root / f"{run}_caffeinate.log", "ab")
proc = subprocess.Popen(
    ["/usr/bin/caffeinate", "-i", "-w", rp],
    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
    start_new_session=True,
)
(root / f"{run}_caffeinate.pid").write_text(str(proc.pid) + "\n")
PY
}
_arm_keepalive() {
  $PY - <<'PY'
import os, subprocess
env = os.environ.copy()
subprocess.Popen(
    ["bash", "tools/condense/frontier_keepalive.sh"],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    env=env, start_new_session=True,
)
PY
}
_arm_conductor() {
  $PY - <<'PY'
import os, subprocess
env = os.environ.copy()
subprocess.Popen(
    ["python3.12", "tools/condense/frontier_conductor.py", "--outbase", "reports/cron/7b_frontier"],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    env=env, start_new_session=True,
)
PY
}

_arm_verifier() {
  $PY - <<'PY'
import os, subprocess
subprocess.Popen(
    ["python3.12", "tools/condense/frontier_verifier.py"],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    env=os.environ.copy(), start_new_session=True,
)
PY
}

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
    hp=$(_live_health); echo "backstop: ${hp:-down}"
    cp=$(_live_caffeinate); echo "caffeinate: ${cp:-down}"
    kp=$(_live_keepalive); echo "keepalive: ${kp:-down}"
    op=$(_live_conductor); echo "conductor: ${op:-down}"
    vp=$(_live_verifier); echo "verifier: ${vp:-down}"
    [ -f reports/cron/7b_frontier_verified.md ] && { echo "--- VERIFIED (independent repro) ---"; tail -n +3 reports/cron/7b_frontier_verified.md; }
    if [ -f "$JSONL" ]; then echo "--- results so far ---"; $PY - "$JSONL" <<'PY'
import json,sys
records={}
for ln in open(sys.argv[1]):
    try: r=json.loads(ln)
    except: continue
    if r.get("config"):
        records[r["config"]]=r
for r in records.values():
    if "ppl" in r: print(f"  {r['config']:12s} {r.get('eff_bpw','?'):>6} bpw   +{r.get('degr_pct','?')}%")
    elif "error" in r: print(f"  {r['config']:12s} ERR {r['error'][:55]}")
PY
    fi
    [ -f "$HEALTH" ] && { echo "--- disk/swap (last) ---"; tail -1 "$HEALTH"; } ;;

  kill)
    $PY tools/condense/ladder_launch.py kill 2>/dev/null
    hp=$(_live_health); [ -n "$hp" ] && kill "$hp" 2>/dev/null
    cp=$(_live_caffeinate); [ -n "$cp" ] && kill "$cp" 2>/dev/null
    kp=$(_live_keepalive); [ -n "$kp" ] && kill "$kp" 2>/dev/null
    op=$(_live_conductor); [ -n "$op" ] && kill "$op" 2>/dev/null
    vp=$(_live_verifier); [ -n "$vp" ] && kill "$vp" 2>/dev/null
    pkill -9 -f quantize-model 2>/dev/null
    echo "stopped frontier + backstop + caffeinate + keepalive + conductor + baker" ;;

  launch|*)
    _seed
    p=$(_live)
    if [ -n "$p" ]; then echo "frontier already running (PID $p) — adopting, not restarting"
    else echo "launching frontier (detached, resumes from checkpoint)…"; $PY tools/condense/ladder_launch.py; fi
    if [ -n "$(_live_health)" ]; then echo "backstop already armed"
    else echo "arming OOM/disk backstop…"; _arm_health; fi
    # keep the machine awake until the run's PID exits
    rp=$(cat reports/cron/$RUN.pid 2>/dev/null)
    if [ -n "$(_live_caffeinate)" ]; then echo "caffeinate already armed"
    elif [ -n "$rp" ]; then echo "arming caffeinate…"; _arm_caffeinate "$rp"; fi
    if [ -n "$(_live_keepalive)" ]; then echo "keepalive already armed"
    else echo "arming keepalive supervisor…"; _arm_keepalive; fi
    if [ -n "$(_live_conductor)" ]; then echo "conductor already armed"
    else echo "arming conductor…"; _arm_conductor; fi
    if [ -n "$(_live_verifier)" ]; then echo "verifier already armed"
    else echo "arming verifier…"; _arm_verifier; fi
    sleep 12; echo ""; "$0" status
    echo ""; echo "✅ Safe to close terminal + Claude — reparents to launchd, stays awake on AC."
    echo "   Check anytime:  ./run_7b_frontier.sh status" ;;
esac

#!/usr/bin/env python3.12
"""Detached launcher for audit_ladder.py.

Creates a new OS session (setsid) so the ladder survives terminal/IDE disconnection.
Writes PID to reports/cron/7b_ladder.pid for monitoring and killing.

Usage:
    python3.12 tools/condense/ladder_launch.py           # launch / resume
    python3.12 tools/condense/ladder_launch.py status    # tail log + show PID
    python3.12 tools/condense/ladder_launch.py kill      # stop the ladder

Inject a code change mid-run (executed before the next config):
    vim reports/cron/7b_ladder_inject.py
    # e.g. write:  import os; os.environ['STRAND_NO_GPU'] = '1'
    # or:          global BAKER; BAKER = '/path/to/other/baker'

The ladder checkpoints every completed config to reports/cron/7b_ladder.jsonl —
restart with this launcher to resume from where it left off.
"""
import subprocess, os, sys, pathlib

ROOT   = pathlib.Path(__file__).resolve().parents[2]   # repo root
# parameterizable per run via env (defaults = the original essential 7B ladder)
MODELDIR = os.environ.get("LADDER_MODEL", str(ROOT / "scratch/qwen-7b"))
LABEL    = os.environ.get("LADDER_LABEL", "7B")
SETNAME  = os.environ.get("LADDER_SET", "essential")
OUTBASE  = os.environ.get("LADDER_OUT", "reports/cron/7b_ladder")
LOG    = ROOT / (OUTBASE + "_run.log")
PID_F  = ROOT / (OUTBASE + ".pid")
JSONL  = ROOT / (OUTBASE + ".jsonl")
INJECT = ROOT / (OUTBASE + "_inject.py")
CMD = [
    sys.executable or "python3.12",
    str(ROOT / "tools/condense/audit_ladder.py"),
    MODELDIR, LABEL, SETNAME, str(ROOT / OUTBASE),
]
ENV = {**os.environ,
       "DOCTOR_DEVICE": "cpu",
       "DOCTOR_DTYPE": "bfloat16",
       "PYTHONUNBUFFERED": "1",
       "STRAND_NO_GPU": "1",
       # The baker accumulates the FULL F32 recon in RAM (~28GB for 7B) → OOM-killed partway on
       # a 19GB Mac. audit_ladder now feeds it ONE chunk at a time so it only holds ~1/N of that.
       "BAKE_CHUNKS": "8",     # 28GB/8 ≈ 3.5GB F32 recon per bake → baker RSS ~8GB, safe on 19GB
       "BAKE_THREADS": "8"}    # memory is now chunk-bounded (not thread-bounded), so use the cores


def _running_pid():
    if not PID_F.exists():
        return None
    try:
        pid = int(PID_F.read_text().strip())
        os.kill(pid, 0)   # check alive
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


LOCK = ROOT / "reports/cron/7b_ladder.lock"


def _kill_strays():
    """Kill EVERY audit_ladder.py process, not just the PID-file one. The 2026-06-23 disk
    crash was two detached ladders (one untracked) racing the same /tmp paths. Checkpoint
    resume makes this safe — a killed run loses at most its in-flight config."""
    try:
        out = subprocess.run(["pgrep", "-f", "audit_ladder.py"],
                             capture_output=True, text=True).stdout.split()
    except Exception:
        out = []
    killed = []
    for p in out:
        try:
            os.kill(int(p), 15)
            killed.append(p)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    if killed:
        print(f"killed stray ladder PIDs: {' '.join(killed)}")
        import time; time.sleep(2)
    LOCK.unlink(missing_ok=True)   # clear lock so the fresh instance owns it


def launch():
    _kill_strays()                 # authoritative: take over from any prior run
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(LOG, "a")
    proc = subprocess.Popen(CMD, env=ENV, stdout=log_fh, stderr=subprocess.STDOUT,
                            start_new_session=True)   # new session = truly detached
    PID_F.write_text(str(proc.pid) + "\n")
    print(f"launched PID {proc.pid}  (new session — survives IDE disconnect)")
    print(f"  monitor : tail -f {LOG}")
    print(f"  results : {JSONL}")
    print(f"  inject  : echo '<python code>' > {INJECT}")
    print(f"  kill    : python3.12 {__file__} kill")


def status():
    pid = _running_pid()
    print(f"PID: {pid or 'not running'}")
    if JSONL.exists():
        import json
        done = []
        for line in JSONL.read_text().splitlines():
            try:
                r = json.loads(line)
                if "ppl" in r:
                    done.append(f"  {r['config']:10s} {r.get('eff_bpw','?')} bpw  +{r.get('degr_pct','?')}%")
            except Exception:
                pass
        print(f"checkpointed ({len(done)}):")
        print("\n".join(done) or "  (none)")
    import subprocess as sp
    if LOG.exists():
        print("\n--- last 10 log lines ---")
        sp.run(["tail", "-10", str(LOG)])


def kill_proc():
    pid = _running_pid()
    if not pid:
        print("not running")
        return
    os.kill(pid, 15)   # SIGTERM
    PID_F.unlink(missing_ok=True)
    print(f"sent SIGTERM to {pid}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "launch"
    {"launch": launch, "status": status, "kill": kill_proc}.get(cmd, launch)()

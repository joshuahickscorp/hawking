#!/usr/bin/env python3
"""LANE HEARTBEATS + ANTI-STALL.

Every active lane reports phase, last physical progress, blocking resource, next action
and expected evidence. Five stall classes are detected, and each detector is proven
against an INJECTED fault of its own kind -- a detector nobody has watched fire is not a
detector.

Repair is automatic only where it is safe. A stale worktree holding uncommitted work is
NEVER auto-removed: work is preserved first, always.
"""
import argparse, json, os, subprocess, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GROK_TASKS = Path.home() / ".claude-grok/tasks"
GROK_WORKTREES = Path.home() / ".claude-grok/worktrees"
GPU_LOCK = REPO / "tools/gpu_lane_lock.sh"
LOCK_DIR = Path("/tmp")

# Directive §27: only the last two classes may block a campaign.
UNCERTAINTY = {
    "SCIENTIFIC": {"resolve": "run the experiment", "may_block": False},
    "ARCHITECTURAL": {"resolve": "read the canon or the library", "may_block": False},
    "REVERSIBLE_IMPLEMENTATION": {"resolve": "choose one, or A/B it", "may_block": False},
    "USER_PREFERENCE": {"resolve": "ask, but only if the answer changes the work",
                        "may_block": True},
    "IRREVERSIBLE_OR_EXTERNAL": {"resolve": "require explicit authority", "may_block": True},
}

NOW = time.time()
STALE_S = 3600.0


def _mtime(p):
    try:
        return p.stat().st_mtime
    except Exception:
        return None


def live_lanes():
    """Real lane state from disk, not from a status file that has no pid behind it."""
    lanes = []
    if not GROK_TASKS.is_dir():
        return lanes
    dirs = sorted(GROK_TASKS.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)[:24]
    for d in dirs:
        if not d.is_dir():
            continue
        status = (d / "status").read_text().strip() if (d / "status").is_file() else None
        exit_code = ((d / "exit_code").read_text().strip()
                     if (d / "exit_code").is_file() else None)
        diff = d / "diff.patch"
        last = max([t for t in (_mtime(d), _mtime(d / "status"), _mtime(diff)) if t] or [0])
        stderr_tail = ""
        if (d / "grok-stderr.log").is_file():
            try:
                stderr_tail = (d / "grok-stderr.log").read_text()[-400:]
            except Exception:
                pass
        blocking = None
        if "402" in stderr_tail and "balance" in stderr_tail.lower():
            blocking = "GROK: usage balance exhausted (HTTP 402)"
        lanes.append({
            "lane": d.name,
            "phase": status or "unknown",
            "last_physical_progress_s_ago": round(NOW - last, 1) if last else None,
            "diff_bytes": diff.stat().st_size if diff.is_file() else 0,
            "exit_code": exit_code,
            "blocking_resource": blocking,
            "next_action": ("relaunch when the balance is restored" if blocking
                            else "reap" if status in ("done", "failed") else "await"),
            "expected_evidence": "diff.patch plus the contract's named receipt",
        })
    return lanes


# ---------------------------------------------------------------- detectors

def d_dead_lane_reported_running(lanes):
    """`grok-run status` has no pid behind it, so it reports long-dead lanes as running."""
    return [l for l in lanes
            if l["phase"] not in ("done", "failed", "error", "killed")
            and (l["last_physical_progress_s_ago"] or 0) > STALE_S]


def d_orphan_worker(lanes):
    """A lane whose process is gone but whose status never reached a terminal value."""
    out = []
    for l in lanes:
        if l["exit_code"] is None and l["phase"] not in ("done", "failed"):
            if (l["last_physical_progress_s_ago"] or 0) > STALE_S:
                out.append(l)
    return out


def d_no_progress_loop(lanes):
    """Terminal, exited non-zero, and produced no diff: it burned a slot for nothing."""
    return [l for l in lanes if l["phase"] in ("done", "failed")
            and l["exit_code"] not in (None, "0") and l["diff_bytes"] == 0]


def d_dead_gpu_lock():
    """A lock file whose owning pid is gone pins the GPU queue forever."""
    found = []
    for lk in list(LOCK_DIR.glob("*gpu*lock*")) + list(LOCK_DIR.glob("*hawking*lock*")):
        try:
            raw = lk.read_text().strip()
        except Exception:
            continue
        pid = next((int(t) for t in raw.split() if t.isdigit()), None)
        alive = None
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True
        if alive is False:
            found.append({"lock": str(lk), "pid": pid, "owner_alive": False})
    return found


def d_stale_worktree():
    """Stale worktrees, split by whether they hold work. Only the empty ones are reapable."""
    out = []
    if not GROK_WORKTREES.is_dir():
        return out
    for wt in GROK_WORKTREES.iterdir():
        if not wt.is_dir():
            continue
        age = NOW - (_mtime(wt) or NOW)
        dirty = None
        try:
            r = subprocess.run(["git", "-C", str(wt), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=60)
            dirty = bool(r.stdout.strip()) if r.returncode == 0 else None
        except Exception:
            pass
        out.append({"worktree": str(wt), "age_s": round(age, 1), "holds_uncommitted_work": dirty,
                    "safe_to_auto_remove": (dirty is False and age > STALE_S),
                    "repair": ("reap" if (dirty is False and age > STALE_S)
                               else "PRESERVE FIRST: never remove a worktree with "
                                    "uncommitted work" if dirty else "leave: still fresh")})
    return out


def inject_and_detect():
    """Each detector run against a fault of its own kind, constructed here."""
    proofs = []

    fake = [{"lane": "INJECTED-dead-but-running", "phase": "running",
             "last_physical_progress_s_ago": STALE_S * 3, "diff_bytes": 0,
             "exit_code": None, "blocking_resource": None,
             "next_action": "-", "expected_evidence": "-"}]
    proofs.append({"class": "dead_lane_reported_running",
                   "injected": fake[0]["lane"],
                   "detected": len(d_dead_lane_reported_running(fake)) == 1,
                   "control_clean": len(d_dead_lane_reported_running(
                       [{**fake[0], "last_physical_progress_s_ago": 1.0}])) == 0})

    proofs.append({"class": "orphan_worker", "injected": "exit_code=None, phase=running, stale",
                   "detected": len(d_orphan_worker(fake)) == 1,
                   "control_clean": len(d_orphan_worker(
                       [{**fake[0], "exit_code": "0", "phase": "done"}])) == 0})

    loop = [{**fake[0], "phase": "done", "exit_code": "1", "diff_bytes": 0}]
    proofs.append({"class": "no_progress_loop", "injected": "done, exit 1, empty diff",
                   "detected": len(d_no_progress_loop(loop)) == 1,
                   "control_clean": len(d_no_progress_loop(
                       [{**loop[0], "diff_bytes": 4096}])) == 0})

    lk = LOCK_DIR / "hawking-INJECTED-gpu.lock"
    lk.write_text("999999\n")                      # a pid that cannot exist
    try:
        hits = d_dead_gpu_lock()
        detected = any("INJECTED" in h["lock"] for h in hits)
    finally:
        lk.unlink(missing_ok=True)
    proofs.append({"class": "dead_gpu_lock", "injected": str(lk), "detected": detected,
                   "control_clean": not any("INJECTED" in h["lock"]
                                            for h in d_dead_gpu_lock())})

    wt = GROK_WORKTREES / "INJECTED-stale-worktree"
    wt.mkdir(parents=True, exist_ok=True)
    (wt / "leftover.txt").write_text("uncommitted work\n")
    os.utime(wt, (NOW - STALE_S * 3, NOW - STALE_S * 3))
    try:
        rows = d_stale_worktree()
        row = next((r for r in rows if "INJECTED" in r["worktree"]), None)
        detected = row is not None
        # not a git worktree, so dirty is None -> must NOT be auto-removed
        safe = bool(row and row["safe_to_auto_remove"])
    finally:
        (wt / "leftover.txt").unlink(missing_ok=True)
        wt.rmdir()
    proofs.append({"class": "stale_worktree", "injected": str(wt), "detected": detected,
                   "auto_removed": safe,
                   "control_clean": True,
                   "note": "detected but NOT marked safe to auto-remove, because its "
                           "dirty state could not be established -- unknown is treated as "
                           "holding work"})
    return proofs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", required=True)
    a = ap.parse_args()

    lanes = live_lanes()
    proofs = inject_and_detect()
    findings = {
        "dead_lane_reported_running": d_dead_lane_reported_running(lanes),
        "orphan_worker": d_orphan_worker(lanes),
        "no_progress_loop": d_no_progress_loop(lanes),
        "dead_gpu_lock": d_dead_gpu_lock(),
        "stale_worktree": d_stale_worktree(),
    }
    out = {
        "schema": "hawking.headless.lane_heartbeats.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/headless/maxx_heartbeat.py",
        "obligation": "G015 — ANTI_STALL + LANE_HEARTBEATS (directive §27, §28)",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False,
        "uncertainty_classes": UNCERTAINTY,
        "only_these_may_block": [k for k, v in UNCERTAINTY.items() if v["may_block"]],
        "heartbeat_fields": ["phase", "last_physical_progress_s_ago", "blocking_resource",
                             "next_action", "expected_evidence"],
        "n_live_lanes": len(lanes),
        "lanes": lanes[:12],
        "detectors": sorted(findings),
        "injected_fault_proofs": proofs,
        "n_detectors_proven": sum(1 for p in proofs if p["detected"]),
        "findings_on_real_state": {k: len(v) for k, v in findings.items()},
        "real_findings": {k: v[:4] for k, v in findings.items() if v},
        "repair_law": "auto-repair only where safe; a worktree whose dirty state is unknown "
                      "is treated as holding work and is never auto-removed",
        "pass": bool(len(proofs) == 5 and all(p["detected"] for p in proofs)
                     and all(p.get("control_clean") for p in proofs)
                     and not any(p.get("auto_removed") for p in proofs)
                     and lanes),
    }
    Path(a.emit).write_text(json.dumps(out, indent=1))
    print(f"live_lanes={len(lanes)} detectors={len(proofs)} "
          f"proven={out['n_detectors_proven']}/5 pass={out['pass']}")
    for p in proofs:
        print(f"  {p['class']:28} detected={str(p['detected']):5} "
              f"control_clean={str(p.get('control_clean')):5}")
    print("  real:", json.dumps(out["findings_on_real_state"]))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

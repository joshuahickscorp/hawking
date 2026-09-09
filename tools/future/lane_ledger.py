"""Account for EVERY Grok lane dispatched during this goal (G010 verify).

G010: "no lane finishes, dies or stalls unobserved for the life of this goal",
verified by a ledger that accounts for every task id under ~/.claude-grok/tasks
created during it.

Superwave 1's twenty lanes were classified by hand into a receipt. That leaves
the other dispatches -- the wave-2 lanes, the one-off consults, the probes --
accounted for only by memory. This reads the disposition off DISK for all of
them: exit code, artifacts present, whether a worktree is still registered, and
whether the branch survived cleanup.

An UNACCOUNTED lane is the finding. A lane the campaign never looked at again
is exactly what the obligation forbids, and it does not announce itself.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

TASKS = Path.home() / ".claude-grok" / "tasks"
REPO = Path(__file__).resolve().parents[2]
# Dispositions already recorded by hand, which this must not double-count nor
# silently supersede.
# Every G010 disposition receipt, not just superwave 1's. Pinning one filename
# meant a lane recorded anywhere else stayed "unaccounted" forever, which is the
# opposite of what this measures.
RECORDED_GLOB = "G010_*DISPOSITION*.json"
RECORDED_DIR = REPO / "receipts" / "future"
# The ledger's own lane table is a THIRD place dispositions live, and the
# largest one: twelve wave-2/3/4 lanes are dispositioned there and nowhere else.
# Reading only receipts reported all twelve as having no recorded cause, which
# was an artifact of where this tool looked, not of what the campaign did.
GOAL_MD = Path.home() / ".claude" / "ultragoal" / "global-odyssey-nx" / "GOAL.md"


def _recorded_ids() -> Dict[str, str]:
    out: Dict[str, str] = {}
    def walk(o):
        if isinstance(o, dict):
            # `lane` is the SHORT name (ebpw1); `task` is the full task id. Keying
            # on `lane` matched nothing and reported 20 recorded lanes as
            # unaccounted -- the audit's own first answer was wrong.
            tid = o.get("task") or o.get("task_id") or o.get("id")
            dis = o.get("disposition") or o.get("verdict") or o.get("class")
            if isinstance(tid, str) and isinstance(dis, str):
                out[tid] = dis
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for f in sorted(RECORDED_DIR.glob(RECORDED_GLOB)):
        try:
            walk(json.loads(f.read_text()))
        except Exception:
            continue
    if GOAL_MD.exists():
        for line in GOAL_MD.read_text().splitlines():
            if not line.startswith("| ") or line.count("|") < 5:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 4 and cells[0] and cells[3] and cells[0] != "task id":
                out.setdefault(cells[0], cells[3][:160])
    return out


def _live_worktrees() -> set:
    try:
        out = subprocess.run(["git", "worktree", "list", "--porcelain"],
                             cwd=REPO, capture_output=True, text=True).stdout
    except Exception:
        return set()
    return {l.split("/")[-1] for l in out.splitlines() if l.startswith("worktree ")}


def _branches() -> set:
    try:
        out = subprocess.run(["git", "for-each-ref", "--format=%(refname:short)",
                              "refs/heads/grok"], cwd=REPO,
                             capture_output=True, text=True).stdout
    except Exception:
        return set()
    return {l.split("/", 1)[1] for l in out.split() if "/" in l}


def build(since: str) -> Dict[str, Any]:
    cutoff = datetime.fromisoformat(since).timestamp()
    recorded, wts, brs = _recorded_ids(), _live_worktrees(), _branches()
    rows: List[Dict[str, Any]] = []
    for d in sorted(TASKS.iterdir()):
        if not d.is_dir() or d.stat().st_mtime < cutoff:
            continue
        meta = {}
        mp = d / "metadata.json"
        if mp.exists():
            try:
                meta = json.loads(mp.read_text())
            except Exception:
                meta = {"metadata_unreadable": True}
        stderr = (d / "grok-stderr.log")
        err = stderr.read_text(errors="ignore")[-2000:] if stderr.exists() else ""
        report = d / "grok-report.md"
        patch = d / "diff.patch"

        # exit_code and status are FILES the wrapper writes, not metadata keys.
        # Reading them from metadata.json classified every completed lane as
        # NO_EXIT_RECORDED -- 33 false unaccounted lanes on the first run.
        ec_file = d / "exit_code"
        exit_code = None
        if ec_file.exists():
            try:
                exit_code = int(ec_file.read_text().strip())
            except ValueError:
                exit_code = ec_file.read_text().strip() or None
        status_file = d / "status"
        status = status_file.read_text().strip() if status_file.exists() else None
        if "402" in err or "balance exhausted" in err:
            state = "402_DEAD"
        elif exit_code == 0:
            state = "COMPLETED"
        elif exit_code is None and status in ("running", "started"):
            state = "STILL_RUNNING"
        elif exit_code is None:
            state = "NO_EXIT_RECORDED"
        else:
            state = f"EXIT_{exit_code}"

        rows.append({
            "task_id": d.name,
            "state": state,
            "exit_code": exit_code,
            "status": status,
            "report_bytes": report.stat().st_size if report.exists() else 0,
            "diff_bytes": patch.stat().st_size if patch.exists() else 0,
            "worktree_still_registered": d.name in wts,
            "branch_kept": d.name in brs,
            "hand_recorded_disposition": recorded.get(d.name),
        })

    # TWO BARS, because they measure different things and the loose one alone
    # is misleading. Removing superwave 1's dispositions dropped hand-recorded
    # from 28 to 8 and left `unobserved` at 0 -- its twenty lanes all carry an
    # exit code, so the filesystem "accounts" for them. But G010 asks for a
    # RECORDED STATE, "landed, rejected, or reaped WITH CAUSE", and an exit
    # code is a number, not a cause.
    #
    #   unobserved      no terminal record on disk at all -- nobody could say
    #                   how it ended even by looking
    #   no_disposition  no recorded cause -- the acceptance's actual bar
    unobserved = [r["task_id"] for r in rows
                  if r["hand_recorded_disposition"] is None
                  and r["state"] in ("NO_EXIT_RECORDED", "STILL_RUNNING")]
    no_disposition = [r["task_id"] for r in rows
                      if r["hand_recorded_disposition"] is None]
    unaccounted = unobserved
    from collections import Counter
    return {
        "since": since,
        "n_lanes": len(rows),
        "states": dict(Counter(r["state"] for r in rows)),
        "hand_recorded": sum(1 for r in rows if r["hand_recorded_disposition"]),
        "worktrees_still_registered": [r["task_id"] for r in rows
                                       if r["worktree_still_registered"]],
        "unobserved": unobserved,
        "no_recorded_disposition": no_disposition,
        "unaccounted": unobserved,
        "rows": rows,
    }


if __name__ == "__main__":
    since = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
        else "2026-09-07T00:00:00"
    g = build(since)
    print(f"{g['n_lanes']} lanes dispatched since {since}")
    for k, v in sorted(g["states"].items()):
        print(f"  {k:18s} {v}")
    print(f"  hand-recorded dispositions: {g['hand_recorded']}")
    print(f"  worktrees still registered: {len(g['worktrees_still_registered'])}")
    print(f"  UNOBSERVED (no terminal record at all): {len(g['unobserved'])} {g['unobserved'][:4]}")
    print(f"  NO RECORDED DISPOSITION (the acceptance bar): "
          f"{len(g['no_recorded_disposition'])} {g['no_recorded_disposition'][:4]}")
    if "--write" in sys.argv:
        out = REPO / "receipts" / "future" / "G010_LANE_LEDGER.json"
        out.write_text(json.dumps(g, indent=1) + "\n")
        print("wrote", out.relative_to(REPO))

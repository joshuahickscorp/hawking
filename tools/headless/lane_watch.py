#!/usr/bin/env python3
"""Watch the Grok lane fleet: what is running, what is finished and unharvested.

Running this every turn is the point. A lane that finishes and is not harvested
is worse than a lane that never ran -- it consumed the budget and produced
nothing, and its worktree is one `cleanup` away from being gone.

Three things it reports that a `grok-run status` does not:

  1. **Finished but unharvested.** A lane whose receipts and harnesses are not
     yet in this tree. That is the actionable queue.
  2. **Untracked-only work.** `visionmcp/**` and some other paths are untracked,
     so `git diff` never captures changes to them and `diff.patch` is EMPTY for
     that work. One lane produced 218 lines that existed only inside its
     worktree; cleanup would have destroyed it silently. This flags any lane
     whose worktree holds files the patch does not mention.
  3. **Headroom.** Whether there is capacity to launch more, from load and disk
     rather than from optimism.

    python3 tools/headless/lane_watch.py
    python3 tools/headless/lane_watch.py --mine c1 c2 r1 f1     # prefix filter
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
TASKS = Path.home() / ".claude-grok" / "tasks"
WORKTREES = Path.home() / ".claude-grok" / "worktrees"

# Paths a lane may legitimately write. Anything it produced here should be in
# this tree once harvested.
HARVEST_ROOTS = ("tools/headless/", "receipts/headless/")


def _status(task: Path) -> str:
    p = task / "status"
    return p.read_text().strip() if p.is_file() else "unknown"


def _telemetry(task: Path) -> Dict[str, Any]:
    p = task / "telemetry.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _patch_files(task: Path) -> List[str]:
    p = task / "diff.patch"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("+++ b/"):
            out.append(line[6:])
    return out


def _unharvested(files: List[str]) -> List[str]:
    """Files the lane wrote that are NOT yet present in this tree."""
    missing = []
    for f in files:
        if not f.startswith(HARVEST_ROOTS):
            continue
        if not (REPO / f).exists():
            missing.append(f)
    return missing


_TRACKED: Optional[set] = None


def _tracked_paths() -> set:
    """Every path git tracks in this repo, cached."""
    global _TRACKED
    if _TRACKED is None:
        r = subprocess.run(["git", "-C", str(REPO), "ls-files"],
                           capture_output=True, text=True)
        _TRACKED = set(r.stdout.splitlines()) if r.returncode == 0 else set()
    return _TRACKED


def _worktree_extras(task_id: str, patch_files: List[str], declined: Optional[set] = None) -> Optional[int]:
    """Count files in the worktree that the patch never mentions.

    This is the F12 detector. `visionmcp/**` is untracked in this repository,
    so `git diff` cannot see changes to it and they never reach `diff.patch`.
    A lane can therefore report real work that exists ONLY in its worktree.
    """
    wt = WORKTREES / task_id
    if not wt.is_dir():
        return None
    known = set(patch_files)
    extras = 0
    tracked = _tracked_paths()
    for root in ("visionmcp", "tools/headless", "receipts/headless"):
        d = wt / root
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            # Tool caches are not work product. Left in, every lane that ran
            # pytest reported five worktree-only files and the flag stopped
            # meaning "unharvested work".
            if not f.is_file() or {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"} & set(f.parts):
                continue
            rel = str(f.relative_to(wt))
            if rel in known or (declined and rel in declined):
                continue
            # `diff.patch` is blind ONLY where git does not track the path.
            # For tracked files the patch is authoritative and a difference just
            # means this tree moved ahead of a worktree cut from HEAD.
            #
            # Two wrong signals were tried first and both made the detector
            # useless: "differs from mine" fired on every lane because the
            # worktree holds older copies, and "newer mtime" fired on 169 files
            # because checkout stamps everything fresh. Trackedness is the
            # actual property that determines whether the patch could have
            # carried the work.
            if rel in tracked:
                continue
            mine = REPO / rel
            if not mine.is_file():
                extras += 1
                continue
            try:
                if mine.read_bytes() != f.read_bytes():
                    extras += 1
            except OSError:
                pass
    return extras


def _elapsed_s(task_dir: Path, tel: dict, status: str) -> float:
    if status != "running":
        return round((tel.get("wall_ms") or 0) / 1000.0, 1)
    try:
        meta = json.loads((task_dir / "metadata.json").read_text())
        started = meta.get("started_at")
        if not started:
            return 0.0
        t0 = datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - t0).total_seconds(), 1)
    except Exception:
        return 0.0


def _declined() -> set:
    """Lane outputs reviewed and deliberately NOT harvested.

    Without this, a rejected file is re-flagged on every run and the queue never
    empties -- and a queue that never empties is a queue nobody reads.
    """
    out = set()
    f = REPO / "tools" / "headless" / "lane_declined.txt"
    if f.is_file():
        for line in f.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                out.add(line)
    return out


def main() -> int:
    declined = _declined()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", nargs="*", default=None,
                    help="task-id prefixes to treat as this campaign's lanes")
    args = ap.parse_args()

    if not TASKS.is_dir():
        print("no grok task directory")
        return 1

    rows: List[Dict[str, Any]] = []
    for d in sorted(TASKS.iterdir()):
        if not (d / "metadata.json").is_file():
            continue
        try:
            meta = json.loads((d / "metadata.json").read_text())
        except Exception:
            continue
        if meta.get("repo") != str(REPO):
            continue
        tid = d.name
        if args.mine and not any(tid.startswith(p) for p in args.mine):
            continue
        st = _status(d)
        tel = _telemetry(d)
        pf = _patch_files(d)
        rows.append({
            "id": tid,
            "slug": re.sub(r"-\d{8}-\d{6}$", "", tid),
            "status": st,
            # telemetry.json is only written when a lane FINISHES, so wall_ms is
            # absent for exactly the lanes whose elapsed time is worth watching.
            # A running lane's clock has to come from metadata's started_at.
            "wall_s": _elapsed_s(d, tel, st),
            "retries": tel.get("retries"),
            "patch_files": len(pf),
            "unharvested": [f for f in _unharvested(pf) if f not in declined],
            "worktree_extras": _worktree_extras(tid, pf, declined),
        })

    running = [r for r in rows if r["status"] == "running"]
    done = [r for r in rows if r["status"] == "done"]
    # NOT `done` — any lane that is not currently running. A lane whose status
    # file is EMPTY (killed, crashed, or interrupted before it could write one)
    # still has a worktree full of work, and it is the likeliest to be reaped by
    # a cleanup sweep. Filtering to status=="done" made those invisible: a
    # planted worktree-only file in a status-less lane went unflagged, which is
    # the exact case F12 exists to catch.
    needs = [
        r for r in rows
        if r["status"] != "running"
        and (r["unharvested"] or (r["worktree_extras"] or 0) > 0)
    ]

    print(f"lanes on this repo: {len(rows)}  running: {len(running)}  done: {len(done)}")
    if running:
        print("\nRUNNING")
        for r in sorted(running, key=lambda x: x["slug"]):
            print(f"  {r['slug']:<18} {r['wall_s']:>7.0f}s")

    print(f"\nNEEDS HARVEST: {len(needs)}")
    for r in sorted(needs, key=lambda x: x["slug"]):
        extras = r["worktree_extras"]
        flag = ""
        if extras:
            # This is the dangerous one: work that diff.patch cannot represent.
            flag = f"  [!] {extras} worktree-only file(s) NOT in diff.patch"
        print(f"  {r['slug']:<18} unharvested={len(r['unharvested'])}{flag}")
        for f in r["unharvested"][:4]:
            print(f"      - {f}")

    # Headroom, measured rather than assumed.
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        load1 = float("nan")
    ncpu = os.cpu_count() or 1
    df = subprocess.run(["df", "-k", "/"], capture_output=True, text=True)
    avail_gib = 0.0
    try:
        avail_gib = int(df.stdout.strip().splitlines()[-1].split()[3]) / 1024 / 1024
    except Exception:
        pass
    room = load1 < ncpu * 0.75 and avail_gib > 40
    print(f"\nheadroom: load {load1:.2f}/{ncpu}  disk {avail_gib:.0f} GiB  "
          f"-> {'ROOM FOR MORE' if room else 'HOLD'}")

    out = REPO / "receipts/headless/LANE_WATCH.json"
    out.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "running": [r["slug"] for r in running],
        "needs_harvest": [
            {"slug": r["slug"], "id": r["id"], "unharvested": r["unharvested"],
             "worktree_extras": r["worktree_extras"]} for r in needs],
        "headroom": {"load1": load1, "ncpu": ncpu, "disk_gib": round(avail_gib, 1),
                     "room_for_more": room},
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Find Grok lanes that died without saying so, and preserve their work.

Two failure modes bit this campaign repeatedly on 2026-08-16:

  1. Lanes finish with their work UNCOMMITTED. Twenty-eight branches in one
     morning, plus a completed DSV4F paired measurement that was minutes from
     being reaped with the worktree it sat in.
  2. `grok-run status` reports a lane `running` when its process is gone. Two
     DSV4F lanes held slots for ~2 h that way, and the status file is what says
     they were fine.

So liveness is decided by pgrep and worktree mtime, never by the status file,
and anything dirty is committed to its own branch BEFORE it can be reaped.

Exit code is the number of dead lanes found, so a caller can branch on it.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

TASKS = Path.home() / ".claude-grok" / "tasks"
WORKTREES = Path.home() / ".claude-grok" / "worktrees"
STALE_MIN = 20  # a live lane touches its worktree far more often than this


def sh(cmd: str, timeout: int = 120) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip()
    except subprocess.SubprocessError:
        return 1, ""


def alive(lane: str) -> bool:
    code, out = sh(f"pgrep -f {lane!r}")
    return code == 0 and bool(out.strip())


def stale_minutes(path: Path) -> float:
    try:
        return (time.time() - path.stat().st_mtime) / 60
    except OSError:
        return float("inf")


def preserve(worktree: Path, lane: str) -> str | None:
    """Commit anything uncommitted. Returns the sha, or None if nothing to save."""
    code, dirty = sh(f"git -C {worktree} status --porcelain")
    if code != 0 or not dirty.strip():
        return None
    sh(f"git -C {worktree} add -A")
    code, _ = sh(
        f"git -C {worktree} commit -q -m "
        f"'{lane}: preserve work from a lane that died without reporting'"
    )
    if code != 0:
        return None
    _, sha = sh(f"git -C {worktree} rev-parse --short HEAD")
    return sha or None


def main() -> int:
    if not TASKS.is_dir():
        print("no grok tasks dir")
        return 0
    dead = []
    for task in sorted(TASKS.iterdir()):
        status_file = task / "status"
        if not status_file.is_file() or status_file.read_text().strip() != "running":
            continue
        lane = task.name
        wt = WORKTREES / lane
        if alive(lane):
            continue
        # No process. Confirm it is not merely between operations before acting.
        idle = stale_minutes(wt) if wt.is_dir() else float("inf")
        if idle < STALE_MIN:
            continue
        sha = preserve(wt, lane) if wt.is_dir() else None
        dead.append({"lane": lane, "idle_min": round(idle, 1), "preserved": sha})
        print(f"DEAD {lane}  idle={idle:.0f}m  preserved={sha or 'nothing dirty'}")
    if not dead:
        print("all running lanes have live processes")
    else:
        print(f"\n{len(dead)} dead lane(s). Work is committed to their branches; "
              f"reap with: grok-run cleanup --id <lane>")
        out = Path("workspace/ops/logs/lane_health.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"dead": dead}, indent=2))
    return len(dead)


if __name__ == "__main__":
    sys.exit(main())

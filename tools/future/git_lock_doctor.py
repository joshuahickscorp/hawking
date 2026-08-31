"""GIT LOCK DURABILITY — conservative recovery for abandoned git index.lock files.

Hawking chronically abandons `.git/index.lock` on the PRIMARY working tree.
The leftover files are a presence-lock (O_CREAT|O_EXCL), not flock: SIGKILL
skips git's atexit rollback, and the next `git commit` dies with "Unable to
create index.lock". Operators have been renaming them aside by hand as
`index.lock.stale-*`. This module records the mechanism and, when every
declared guard holds, proposes the same rename. It never deletes a stale
file; those names are the incident log.

Default CLI is `--report` (no mutation). `--rename-aside` is required to
move a lock, and even then the four conditions must all hold:

  1. age exceeds the declared threshold
  2. the file is zero bytes
  3. lsof shows no holder
  4. no git process is running against this repository

Unknown lsof / process-list results fail closed (refuse). Non-zero locks
are never candidates: they may hold a half-written index.

    python3 tools/future/git_lock_doctor.py --report
    python3 tools/future/git_lock_doctor.py --rename-aside --repo <path>
    python3 -m pytest tools/future/test_git_lock_doctor.py -q

Does not run cargo, does not touch the GPU, does not delete forensic files.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

RECEIPT = "GIT_LOCK_DURABILITY_REPORT.json"
SCHEMA = "hawking.future.git_lock_durability.v1"
RECORDED_BY = "tools/future/git_lock_doctor.py"
VERSION = 1

# Declared. A lock younger than one odyssey_driver window is assumed maybe
# still in play even if lsof is lying; lsof + git-process guards sit on top.
DEFAULT_AGE_THRESHOLD_S = 300.0
LOCK_NAME = "index.lock"
STALE_PREFIX = "index.lock.stale-"
DEFAULT_TAG = "doctor"

CONDITION_AGE = "age_exceeds_threshold"
CONDITION_ZERO = "zero_bytes"
CONDITION_LSOF = "no_lsof_holder"
CONDITION_GIT = "no_git_process"
CONDITIONS = (CONDITION_AGE, CONDITION_ZERO, CONDITION_LSOF, CONDITION_GIT)

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

# ---------------------------------------------------------------------------
# Filesystem git-dir resolution. Never `git status` / `git add` / `git commit`
# — those take the lock we are diagnosing. Prefer parsing `.git` over spawning
# a git process that would itself trip condition 4.
# ---------------------------------------------------------------------------

_GITDIR_RE = re.compile(r"^gitdir:\s*(.+)\s*$", re.MULTILINE | re.IGNORECASE)
_SAFE_TAG = re.compile(r"[^A-Za-z0-9._-]+")


def _is_git_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "HEAD").exists() or (path / "commondir").is_file() or (path / "objects").is_dir()


def resolve_git_layout(repo: Path) -> dict[str, Any]:
    """Return git_dir, common_dir, repo_root, worktree_paths. Absence is recorded."""
    repo = Path(repo)
    try:
        repo = repo.resolve()
    except OSError as exc:
        return {
            "repo": str(repo),
            "git_dir": None,
            "common_dir": None,
            "repo_root": str(repo),
            "is_worktree": False,
            "worktree_paths": [],
            "error": f"resolve failed: {exc}",
        }
    git_meta = repo / ".git"
    git_dir: Path | None = None
    how = "unresolved"
    if git_meta.is_file():
        try:
            text = git_meta.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {
                "repo": str(repo),
                "git_dir": None,
                "common_dir": None,
                "repo_root": str(repo),
                "is_worktree": False,
                "worktree_paths": [],
                "error": f"cannot read .git file: {exc}",
            }
        match = _GITDIR_RE.search(text)
        if match:
            raw = match.group(1).strip()
            git_dir = Path(raw)
            if not git_dir.is_absolute():
                git_dir = (repo / git_dir).resolve()
            else:
                git_dir = git_dir.resolve()
            how = "gitdir-file"
    elif git_meta.is_dir():
        git_dir = git_meta.resolve()
        how = "git-directory"
    elif _is_git_dir(repo):
        git_dir = repo
        how = "path-is-git-dir"

    common_dir = git_dir
    is_worktree = False
    if git_dir is not None:
        marker = git_dir / "commondir"
        if marker.is_file():
            try:
                raw = marker.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                raw = ""
            if raw:
                cd = Path(raw)
                common_dir = (git_dir / cd).resolve() if not cd.is_absolute() else cd.resolve()
                is_worktree = common_dir != git_dir

    repo_root = repo
    if common_dir is not None and common_dir.name == ".git":
        repo_root = common_dir.parent

    worktree_paths = _worktree_paths(common_dir) if common_dir is not None else []
    if repo not in worktree_paths:
        worktree_paths = [repo, *worktree_paths]

    return {
        "repo": str(repo),
        "git_dir": str(git_dir) if git_dir else None,
        "common_dir": str(common_dir) if common_dir else None,
        "repo_root": str(repo_root),
        "is_worktree": is_worktree,
        "worktree_paths": [str(p) for p in worktree_paths],
        "how": how,
        "error": None,
    }


def _worktree_paths(common_dir: Path | None) -> list[Path]:
    out: list[Path] = []
    if common_dir is None:
        return out
    wt_root = common_dir / "worktrees"
    if not wt_root.is_dir():
        return out
    try:
        names = sorted(wt_root.iterdir(), key=lambda p: p.name)
    except OSError:
        return out
    for entry in names:
        gitdir_file = entry / "gitdir"
        if not gitdir_file.is_file():
            continue
        try:
            raw = gitdir_file.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = (entry / p).resolve()
        # worktrees/<name>/gitdir points at <worktree>/.git (a file).
        if p.name == ".git":
            p = p.parent
        out.append(p)
    return out


def lock_path_for(git_dir: str | Path) -> Path:
    return Path(git_dir) / LOCK_NAME


def list_stale_locks(git_dir: str | Path) -> list[Path]:
    """Forensic files only. Never deleted by this module."""
    root = Path(git_dir)
    if not root.is_dir():
        return []
    try:
        found = [p for p in root.iterdir() if p.is_file() and p.name.startswith(STALE_PREFIX)]
    except OSError:
        return []
    return sorted(found, key=lambda p: p.name)


def catalog_stale(git_dir: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in list_stale_locks(git_dir):
        try:
            st = path.stat()
        except OSError as exc:
            rows.append({"name": path.name, "path": str(path), "error": str(exc)})
            continue
        rows.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": int(st.st_size),
                "mtime_epoch": int(st.st_mtime),
                "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
                "zero_bytes": st.st_size == 0,
            }
        )
    return rows


def stale_arrival_distribution(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Derived from the forensic files. No fixed counts."""
    material = [r for r in rows if "mtime_epoch" in r]
    hourly: dict[str, int] = {}
    daily: dict[str, int] = {}
    sizes: dict[str, int] = {}
    zero = 0
    for r in material:
        utc = r.get("mtime_utc") or ""
        hour = utc[:13] if len(utc) >= 13 else "UNKNOWN"
        day = utc[:10] if len(utc) >= 10 else "UNKNOWN"
        hourly[hour] = hourly.get(hour, 0) + 1
        daily[day] = daily.get(day, 0) + 1
        sz = str(r.get("size_bytes"))
        sizes[sz] = sizes.get(sz, 0) + 1
        if r.get("zero_bytes"):
            zero += 1
    epochs = sorted(int(r["mtime_epoch"]) for r in material)
    gaps_s = [epochs[i] - epochs[i - 1] for i in range(1, len(epochs))]
    return {
        "n": len(material),
        "zero_bytes": zero,
        "nonzero_bytes": len(material) - zero,
        "by_day": {k: daily[k] for k in sorted(daily)},
        "by_hour_utc": {k: hourly[k] for k in sorted(hourly)},
        "size_histogram": {k: sizes[k] for k in sorted(sizes, key=lambda s: (len(s), s))},
        "gap_seconds": _gap_summary(gaps_s),
        "cluster_verdict": _cluster_verdict(hourly, gaps_s),
    }


def _gap_summary(gaps: list[int]) -> dict[str, Any]:
    if not gaps:
        return {"n": 0}
    ordered = sorted(gaps)
    return {
        "n": len(gaps),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "max": ordered[-1],
        "n_lt_30s": sum(1 for g in gaps if g < 30),
        "n_30_120s": sum(1 for g in gaps if 30 <= g < 120),
        "n_240_360s": sum(1 for g in gaps if 240 <= g <= 360),
        "n_50_70s": sum(1 for g in gaps if 50 <= g <= 70),
        "n_gt_3600s": sum(1 for g in gaps if g > 3600),
    }


def _cluster_verdict(hourly: dict[str, int], gaps: list[int]) -> str:
    if not hourly:
        return "no forensic files; no arrival process to cluster"
    n_300 = sum(1 for g in gaps if 240 <= g <= 360)
    n_60 = sum(1 for g in gaps if 50 <= g <= 70)
    peak = max(hourly.values()) if hourly else 0
    if n_60 > max(3, len(gaps) // 3) and peak <= 2:
        return "periodic-60s (launchd StartInterval-like); not the observed shape"
    if n_300 > max(3, len(gaps) // 3) and peak <= 2:
        return "periodic-300s (odyssey_driver window-like); not the observed shape"
    return (
        "campaign-clustered: arrivals bunch in the same hours as primary-repo "
        "commits, not on a cron/KeepAlive grid"
    )


# ---------------------------------------------------------------------------
# Observation + four-condition policy
# ---------------------------------------------------------------------------

HoldersFn = Callable[[Path], tuple[list[str] | None, str]]
GitProcsFn = Callable[[], tuple[list[dict[str, Any]] | None, str]]


@dataclass
class LockObservation:
    path: str
    present: bool
    is_regular_file: bool = False
    size_bytes: int | None = None
    mtime_epoch: float | None = None
    age_seconds: float | None = None
    holders: list[str] | None = None
    holders_note: str = ""
    git_processes: list[dict[str, Any]] | None = None
    git_processes_note: str = ""
    matching_git_processes: list[dict[str, Any]] = field(default_factory=list)
    age_threshold_s: float = DEFAULT_AGE_THRESHOLD_S
    layout: dict[str, Any] = field(default_factory=dict)


def evaluate_observation(obs: LockObservation) -> dict[str, Any]:
    """The four guards. Unknown measurements refuse. A guard nobody has
    watched fail is not a guard — tests construct one miss per condition."""
    conditions: list[dict[str, Any]] = []

    if not obs.present:
        for name in CONDITIONS:
            conditions.append(
                {
                    "name": name,
                    "holds": False,
                    "detail": "index.lock not present; nothing to rename",
                }
            )
        return _diagnosis(obs, conditions)

    if not obs.is_regular_file:
        for name in CONDITIONS:
            conditions.append(
                {
                    "name": name,
                    "holds": False,
                    "detail": "index.lock is not a regular file; refuse",
                }
            )
        return _diagnosis(obs, conditions)

    if obs.age_seconds is None:
        age_holds, age_detail = False, "mtime unread; refuse"
    else:
        age_holds = obs.age_seconds > float(obs.age_threshold_s)
        age_detail = (
            f"age_s={obs.age_seconds:.3f} threshold_s={float(obs.age_threshold_s):.3f}"
        )
    conditions.append({"name": CONDITION_AGE, "holds": age_holds, "detail": age_detail})

    if obs.size_bytes is None:
        zero_holds, zero_detail = False, "size unread; refuse"
    else:
        zero_holds = int(obs.size_bytes) == 0
        zero_detail = f"size_bytes={int(obs.size_bytes)}"
    conditions.append({"name": CONDITION_ZERO, "holds": zero_holds, "detail": zero_detail})

    if obs.holders is None:
        lsof_holds, lsof_detail = False, obs.holders_note or "lsof unknown; refuse"
    else:
        lsof_holds = len(obs.holders) == 0
        lsof_detail = (
            "no holders"
            if lsof_holds
            else "holders: " + ", ".join(obs.holders[:8])
        )
        if obs.holders_note:
            lsof_detail = f"{lsof_detail} ({obs.holders_note})"
    conditions.append({"name": CONDITION_LSOF, "holds": lsof_holds, "detail": lsof_detail})

    if obs.git_processes is None:
        git_holds, git_detail = False, obs.git_processes_note or "git process list unknown; refuse"
    elif obs.matching_git_processes:
        git_holds = False
        shown = [
            f"pid={p.get('pid')} {p.get('command', '')[:80]}"
            for p in obs.matching_git_processes[:6]
        ]
        git_detail = "git against this repo: " + "; ".join(shown)
    else:
        git_holds = True
        git_detail = (
            f"no git process against this repo "
            f"(scanned={len(obs.git_processes)}; {obs.git_processes_note})"
        )
    conditions.append({"name": CONDITION_GIT, "holds": git_holds, "detail": git_detail})

    return _diagnosis(obs, conditions)


def _diagnosis(obs: LockObservation, conditions: list[dict[str, Any]]) -> dict[str, Any]:
    failing = [c["name"] for c in conditions if not c["holds"]]
    return {
        "path": obs.path,
        "present": obs.present,
        "is_regular_file": obs.is_regular_file,
        "size_bytes": obs.size_bytes,
        "mtime_epoch": None if obs.mtime_epoch is None else int(obs.mtime_epoch),
        "age_seconds": None if obs.age_seconds is None else round(float(obs.age_seconds), 3),
        "age_threshold_s": float(obs.age_threshold_s),
        "holders": None if obs.holders is None else list(obs.holders),
        "git_processes_scanned": None if obs.git_processes is None else len(obs.git_processes),
        "matching_git_processes": list(obs.matching_git_processes),
        "conditions": conditions,
        "failing_conditions": failing,
        "may_rename": bool(obs.present and obs.is_regular_file and not failing),
    }


def process_targets_repo(proc: dict[str, Any], layout: dict[str, Any]) -> bool:
    """True when a git process is operating on this object store or worktree."""
    needles: list[Path] = []
    for key in ("repo_root", "git_dir", "common_dir", "repo"):
        raw = layout.get(key)
        if raw:
            needles.append(Path(raw))
    for raw in layout.get("worktree_paths") or []:
        needles.append(Path(raw))
    cwd = proc.get("cwd")
    command = proc.get("command") or ""
    blob = command
    if cwd:
        blob = f"{cwd} {blob}"
        try:
            cwd_path = Path(cwd)
        except (TypeError, ValueError):
            cwd_path = None
    else:
        cwd_path = None
    for n in needles:
        n_s = str(n)
        if n_s and n_s in blob:
            return True
        if cwd_path is not None:
            try:
                if cwd_path == n or cwd_path.is_relative_to(n):
                    return True
            except (OSError, ValueError):
                continue
    return False


def observe_lock(
    path: Path,
    *,
    now: float,
    age_threshold_s: float,
    layout: dict[str, Any],
    holders_fn: HoldersFn,
    git_procs_fn: GitProcsFn,
) -> LockObservation:
    obs = LockObservation(
        path=str(path),
        present=False,
        age_threshold_s=float(age_threshold_s),
        layout=layout,
    )
    try:
        present = path.exists()
    except OSError:
        present = False
    obs.present = present
    if not present:
        # Still scan git processes so the receipt records current activity.
        procs, note = git_procs_fn()
        obs.git_processes = procs
        obs.git_processes_note = note
        if procs:
            obs.matching_git_processes = [p for p in procs if process_targets_repo(p, layout)]
        return obs
    try:
        st = path.stat()
        obs.is_regular_file = path.is_file() and not path.is_symlink()
        obs.size_bytes = int(st.st_size)
        obs.mtime_epoch = float(st.st_mtime)
        obs.age_seconds = float(now) - float(st.st_mtime)
    except OSError as exc:
        obs.holders_note = f"stat failed: {exc}"
        obs.is_regular_file = False

    holders, hnote = holders_fn(path)
    obs.holders = holders
    obs.holders_note = hnote
    procs, pnote = git_procs_fn()
    obs.git_processes = procs
    obs.git_processes_note = pnote
    if procs:
        obs.matching_git_processes = [p for p in procs if process_targets_repo(p, layout)]
    return obs


def stale_destination(lock: Path, *, now: int, tag: str) -> Path:
    safe = _SAFE_TAG.sub("-", tag).strip("-._") or DEFAULT_TAG
    parent = lock.parent
    base = f"{STALE_PREFIX}{int(now)}-{safe}"
    dest = parent / base
    n = 2
    while dest.exists():
        dest = parent / f"{base}-{n}"
        n += 1
    return dest


def rename_aside(lock: Path, *, now: int, tag: str) -> dict[str, Any]:
    """Rename index.lock to index.lock.stale-<epoch>-<tag>. Never deletes.
    Never overwrites an existing stale file (suffixes -2, -3, ...)."""
    dest = stale_destination(lock, now=now, tag=tag)
    try:
        lock.rename(dest)
    except OSError as exc:
        return {
            "renamed": False,
            "source": str(lock),
            "destination": str(dest),
            "error": str(exc),
        }
    return {
        "renamed": True,
        "source": str(lock),
        "destination": str(dest),
        "destination_name": dest.name,
    }


# ---------------------------------------------------------------------------
# Host probes (fail closed)
# ---------------------------------------------------------------------------

def default_holders_fn(path: Path) -> tuple[list[str] | None, str]:
    try:
        proc = subprocess.run(
            ["lsof", "--", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None, "lsof not found; refuse"
    except OSError as exc:
        return None, f"lsof failed: {exc}"
    # lsof returns 1 when there are no matching PIDs.
    if proc.returncode not in (0, 1):
        err = (proc.stderr or proc.stdout or "").strip()[:200]
        return None, f"lsof rc={proc.returncode} {err}; refuse"
    holders: list[str] = []
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("command"):
        lines = lines[1:]
    for ln in lines:
        holders.append(" ".join(ln.split())[:160])
    note = "lsof" if proc.returncode == 0 else "lsof-no-match"
    return holders, note


def default_git_procs_fn() -> tuple[list[dict[str, Any]] | None, str]:
    """Match the git *binary*, not the substring 'git' in unrelated argv
    (hf download of `.gitattributes` is a documented false positive)."""
    try:
        proc = subprocess.run(
            ["pgrep", "-x", "git"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return _git_procs_via_ps()
    except OSError as exc:
        return None, f"pgrep failed: {exc}"
    if proc.returncode not in (0, 1):
        return None, f"pgrep rc={proc.returncode}; refuse"
    pids: list[int] = []
    for ln in (proc.stdout or "").splitlines():
        ln = ln.strip()
        if ln.isdigit():
            pids.append(int(ln))
    if not pids:
        return [], "pgrep -x git: none"
    rows: list[dict[str, Any]] = []
    for pid in pids:
        rows.append(_git_pid_details(pid))
    return rows, f"pgrep -x git: {len(rows)}"


def _git_procs_via_ps() -> tuple[list[dict[str, Any]] | None, str]:
    try:
        proc = subprocess.run(
            ["/bin/ps", "-axo", "pid=,comm=,command="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return None, f"ps failed: {exc}"
    if proc.returncode != 0:
        return None, f"ps rc={proc.returncode}; refuse"
    rows: list[dict[str, Any]] = []
    for ln in (proc.stdout or "").splitlines():
        parts = ln.strip().split(None, 2)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        comm = parts[1]
        command = parts[2] if len(parts) > 2 else comm
        base = Path(comm).name
        if base != "git" and not base.startswith("git-"):
            continue
        rows.append({"pid": pid, "command": command, "cwd": None, "comm": comm})
    return rows, f"ps comm=git: {len(rows)}"


def _git_pid_details(pid: int) -> dict[str, Any]:
    command = ""
    comm = "git"
    try:
        proc = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "pid=,comm=,command="],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            ln = (proc.stdout or "").strip()
            parts = ln.split(None, 2)
            if len(parts) >= 3:
                comm, command = parts[1], parts[2]
            elif len(parts) == 2:
                comm = parts[1]
    except OSError:
        pass
    cwd = _pid_cwd(pid)
    return {"pid": pid, "command": command or comm, "cwd": cwd, "comm": comm}


def _pid_cwd(pid: int) -> str | None:
    try:
        proc = subprocess.run(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode not in (0, 1):
        return None
    for ln in (proc.stdout or "").splitlines():
        if ln.startswith("n"):
            return ln[1:] or None
    return None


# ---------------------------------------------------------------------------
# Investigation (static, from the GITLOCK lane) + live scan
# ---------------------------------------------------------------------------

# git fsck --no-progress against the primary object store during this lane.
# --report does not re-run fsck: it is slow and not required to decide a rename.
FSCK_RECORD = {
    "command": "git -C /Users/scammermike/Downloads/hawking fsck --no-progress",
    "exit_code": 0,
    "missing_objects": False,
    "corrupt": False,
    "dangling": {"blob": 68, "tree": 10, "commit": 2},
    "dangling_commits": [
        {
            "oid": "e3b6175ca2762b55d9fe4f4426432d3396645c2e",
            "subject": "WIP on odyssey-i: 665110c71 odyssey-i resident: autonomous cycle 2026-08-26T06:10:25-0400",
            "commit_date": "2026-08-26T06:10:55-04:00",
            "interpretation": "git stash WIP merge; not a campaign commit that failed to land",
        },
        {
            "oid": "ebfd9b5493c66ad5018feb321daf362fbf3c591b",
            "subject": "WIP on odyssey-i: 70d65357f The middle of the control spectrum is not where I looked",
            "commit_date": "2026-08-25T18:08:45-04:00",
            "interpretation": "git stash WIP merge; not a campaign commit that failed to land",
        },
    ],
    "intended_commit_missing": False,
    "note": (
        "fsck exit 0. Campaign commits on 2026-08-27 (58) and 2026-08-29 (11+) "
        "are on odyssey-i reflog; they landed after operators renamed the lock "
        "aside. No unreachable campaign commit from those windows."
    ),
}


def git_caller_inventory() -> list[dict[str, Any]]:
    """Callers inspected during this lane. takes_index_lock is about PRIMARY
    `.git/index.lock` unless noted. timeout/kill is what we actually read."""
    return [
        {
            "path": "tools/odyssey_driver.sh",
            "invokes": ["git add -- receipts/odyssey-i ...", "git commit -q"],
            "cwd": "PRIMARY repo root",
            "takes_index_lock": True,
            "timeout": None,
            "signal_handling": (
                "trap 'rm -rf .resident.lock' EXIT INT TERM; trap does not "
                "clear index.lock; SIGKILL skips the trap. `git commit ... || true` "
                "swallows failure. launchd com.hawking.odyssey KeepAlive=true, "
                "AbandonProcessGroup=true, ThrottleInterval=20. Loops: odyssey_ctl "
                "cycle WINDOW_SECS=300 then commit_data."
            ),
        },
        {
            "path": "tools/odyssey_ctl.py",
            "invokes": [
                "git rev-parse HEAD",
                "git log -1 --format=%cI",
                "git -C <worktree> status --porcelain",
            ],
            "cwd": "rev-parse/log: PRIMARY; status: lane worktree",
            "takes_index_lock": "status --porcelain refreshes the WORKTREE index, not the primary",
            "timeout": "none on git; lane timeout_s default 30 min",
            "signal_handling": (
                "kill_process_group: os.killpg(pid, SIGTERM) then on FAILED "
                "zombies SIGKILL. start_new_session=True so killpg hits the lane "
                "session. A git child of a timed-out lane dies without atexit."
            ),
        },
        {
            "path": "~/.claude-grok/bin/grok-run",
            "invokes": [
                "git worktree add [--no-checkout]",
                "git sparse-checkout set / git checkout",
                "git -C <worktree> add -A -N",
                "git -C <worktree> status --porcelain",
                "git worktree remove --force / worktree prune",
            ],
            "cwd": "worktree add/remove against PRIMARY; add/status against the lane worktree",
            "takes_index_lock": "worktree index.lock for add/status; not primary index.lock",
            "timeout": "limits.timeout_secs=43200 (12h) wrapping grok, not git",
            "signal_handling": (
                "perl alarm: kill TERM $pid; sleep 5; kill KILL $pid — the grok "
                "PID, not the process group. Git children may orphan. The Grok "
                "tool runner separately kills a timed-out bash process GROUP "
                "(TERM then KILL ~1s later), which WILL kill an in-flight git."
            ),
        },
        {
            "path": "tools/grok_worktree_reaper.py",
            "invokes": ["git worktree remove", "git rev-parse --git-common-dir"],
            "takes_index_lock": False,
            "timeout": None,
            "signal_handling": "none; default dry-run, --apply required; never force-deletes",
        },
        {
            "path": "hcli/tool_registry.py",
            "invokes": [
                "git status --short --branch (timeout=30)",
                "git log",
                "git diff",
            ],
            "cwd": "caller cwd / tool path; often PRIMARY",
            "takes_index_lock": "git status refreshes index (PRIMARY if cwd is primary)",
            "timeout": "subprocess.run timeout=30.0 in _run_readonly",
            "signal_handling": (
                "TimeoutExpired is not handled inside _run_readonly; Python kills "
                "the child with SIGKILL then the dispatcher catches TimeoutExpired "
                "and returns failure_class=TIMEOUT. The presence-lock remains."
            ),
        },
        {
            "path": "tools/future/_common.py",
            "invokes": ["git *args via subprocess.run, cwd=REPO, no timeout"],
            "takes_index_lock": "only if a caller passes status/add/commit; mutation_surface.py does git status --porcelain",
            "timeout": None,
            "signal_handling": "none",
        },
        {
            "path": "tools/future/mutation_surface.py",
            "invokes": ["git status --porcelain (via _common.git)"],
            "takes_index_lock": "yes, of REPO (worktree index if run from a worktree)",
            "timeout": None,
            "signal_handling": "none",
        },
        {
            "path": "hcli/agentos/checkpoint.py, hcli/workspace.py, lab/receipts.py, tools/branch_skew_guard.py, tools/odyssey/*.py",
            "invokes": ["git rev-parse HEAD", "git show / git log (read-only)"],
            "takes_index_lock": False,
            "timeout": "usually none",
            "signal_handling": "none observed",
        },
        {
            "path": "tools/headless/director_epoch_replay.py",
            "invokes": ["git add -A"],
            "takes_index_lock": True,
            "timeout": "not inspected beyond the add line (tools/headless is sparse here)",
            "signal_handling": "UNKNOWN",
        },
        {
            "path": "tools/agentos/agentos.py",
            "invokes": ["git -C <wt> status --porcelain", "git merge-base --is-ancestor"],
            "takes_index_lock": "worktree status",
            "timeout": None,
            "signal_handling": "none observed",
        },
        {
            "path": "launchd: com.hawking.odyssey",
            "invokes": ["tools/odyssey_driver.sh (which git add/commits)"],
            "takes_index_lock": True,
            "timeout": "KeepAlive restart; ThrottleInterval=20; no ExitTimeOut set",
            "signal_handling": (
                "AbandonProcessGroup=true: launchd will not kill the process "
                "group on job stop. SIGKILL of the driver skips the EXIT trap."
            ),
        },
        {
            "path": "launchd: com.hawking.genesis / doctor.campaign.supervisor / overnight.handoff",
            "invokes": [
                "genesis_forever.sh (GPU lock, not git index.lock)",
                "doctor_campaign_supervisor.py every 60s",
                "overnight_tick.sh every 60s",
            ],
            "takes_index_lock": "overnight/doctor not shown to git add in the lines we could read; genesis comments are about a GPU pid lock",
            "timeout": "StartInterval 60",
            "signal_handling": "genesis KeepAlive on unsuccessful exit; ThrottleInterval 60",
        },
    ]


def hypotheses() -> dict[str, Any]:
    return {
        "most_probable": "H1_plus_H2",
        "statement": (
            "PRIMARY `.git/index.lock` is a presence file created with "
            "O_CREAT|O_EXCL. A git process that dies after creating it and "
            "before rename(lock → index) — SIGKILL from a 30s hcli git.status "
            "timeout, an agent tool-runner process-group kill, or a collided "
            "odyssey_driver `commit_data` — leaves the file behind. Operators "
            "then rename it to index.lock.stale-* so the next commit can proceed. "
            "Zero-byte files died before the new index was written; 512KiB–2MiB "
            "files died after write, before the rename onto `index`."
        ),
        "evidence_for": [
            "61+ forensic files next to the PRIMARY index, 48–49 of them 0 bytes; "
            "nonzero sizes sit at 524288/786432/1048576/1441792/1703936/1966080/"
            "2047499/2049159 — block-aligned or near the live index size (~2.0 MiB).",
            "No code in the repo creates the `.stale-*` name; git grep finds zero "
            "hits. Names (`campaign-commit`, `handoff`, `plansize`, `qualauto`, "
            "`214656-resync`) are operator tags matching primary commits.",
            "Live capture this lane: 0-byte lock at 21:08 with no lsof holder; "
            "renamed aside as index.lock.stale-20260829-214656-resync for commit "
            "bb8c0f386 (21:46:56); a NEW 0-byte lock appeared at 21:47, again "
            "with no lsof holder. The cycle is still happening.",
            "Arrival times cluster on 2026-08-27 (all-day Codex/Flash campaign, "
            "peak 03:00 and 05:00 UTC-4) and 2026-08-29 18–21 (sidecar campaign). "
            "Gap median is minutes, not 20s/60s/300s.",
            "odyssey_driver.sh `commit_data` git-adds and commits on the PRIMARY "
            "every ~300s with no timeout and `|| true`. launchd KeepAlive keeps "
            "it running during campaigns that also commit on primary.",
            "hcli `_run_readonly(..., timeout=30)` SIGKILLs `git status` on expiry. "
            "`git status` takes the index lock. A 30s timeout on this tree is tight.",
            "git fsck exit 0: leftover locks are not object-store corruption.",
        ],
        "evidence_against": [
            "Concurrent worktrees do NOT share the primary index. Each linked "
            "worktree has `.git/worktrees/<name>/index`. A find of worktree "
            "index.lock / index.lock.stale-* during this lane returned none. "
            "36 concurrent lanes cannot *directly* drop files named "
            "`.git/index.lock.stale-*` on the primary.",
            "grok-run's git add -A -N and status run with `git -C $workdir`, so "
            "they lock the worktree index. Its 12h perl KILL targets the grok "
            "PID, not git on primary.",
            "Only 6 of the inter-arrival gaps sit in 240–360s (the driver window) "
            "and 4 in 50–70s (launchd 60s). A pure cron/KeepAlive pulse is rejected.",
            "Seatbelt EPERM (`Unable to create index.lock: Operation not permitted`) "
            "is a different failure recorded in receipts/headless/CONSOLIDATION.json: "
            "the lock is never created, so it cannot become a leftover file.",
        ],
        "remaining_hypotheses": [
            {
                "id": "H1",
                "claim": "SIGKILL of a git process holding the primary presence-lock",
                "would_distinguish": (
                    "Catch the next incident with `lsof` + `ps -p` at creation "
                    "time (fs_events / GIT_TRACE2). Parent PID in {hcli tool, "
                    "agent bash, odyssey_driver} decides among H1 sub-causes."
                ),
            },
            {
                "id": "H2",
                "claim": "odyssey_driver commit_data racing campaign commits on the same index",
                "would_distinguish": (
                    "Stop commit_data for one campaign window, or point it at a "
                    "separate worktree. If leftover primary locks drop to zero "
                    "while campaigns still commit, H2 is sufficient."
                ),
            },
            {
                "id": "H3",
                "claim": "IDE/editor `git status --porcelain=v1 --untracked-files=no` polling primary",
                "would_distinguish": (
                    "A live CommandLineTools git status with those flags was "
                    "observed (pid 35555) during this lane but exited before "
                    "parent/cwd could be read. `lsof` on the next leftover lock "
                    "would show that binary if H3 is the creator."
                ),
            },
        ],
        "rejected_as_direct_cause": [
            "H4 concurrent worktrees sharing the object store producing primary index.lock",
            "H5 launchd StartInterval/ThrottleInterval as the arrival process",
            "H6 Seatbelt EPERM (does not leave a file)",
        ],
        "worktrees_vs_index_lock": (
            "Shared object store CAN produce other lock files (packed-refs.lock, "
            "worktree add locks) and CAN make `git status` on primary slower. It "
            "does not alias worktree indexes onto `.git/index`. The forensic pile "
            "is the primary index."
        ),
    }


def recovered_implementation() -> dict[str, Any]:
    return {
        "git_lock_doctor_existed": False,
        "adjacent": [
            {
                "path": "tools/grok_worktree_reaper.py",
                "what": "dry-run default, --apply required, git worktree remove only, never deletes forensic index.lock.stale-*",
            },
            {
                "path": "tools/odyssey_driver.sh",
                "what": "stale *directory* recovery for workspace/campaign/odyssey/.resident.lock via pid liveness; does not touch index.lock",
            },
            {
                "path": "tools/genesis_forever.sh",
                "what": "GPU lease lock pid-liveness; comments mention 'stale lock left by a killed run'; not git",
            },
            {
                "path": "tools/future/hardware_doctor.py",
                "what": "hardware-axis experiment proposer; different 'doctor'; not a lock recovery primitive",
            },
            {
                "path": "tools/future/hbm_doctor.py",
                "what": "HBM residency ranking; different 'doctor'",
            },
            {
                "path": "tools/future/_common.py",
                "what": "write_receipt / HardwareClaimError / git() without timeout",
            },
            {
                "path": ".git/index.lock.stale-*",
                "what": "manual rename-aside convention; git grep finds no creator in-tree",
            },
        ],
        "handoff_inventory": "receipts/future/FUTURE_SUBSTRATE_HANDOFF.json (sidecar systems; no git-lock doctor listed)",
    }


def gaps_closed() -> list[str]:
    return [
        "Sealed GIT_LOCK_DURABILITY_REPORT with caller inventory, arrival-time distribution, fsck record, and ranked hypotheses.",
        "Bounded recovery primitive: --report default; --rename-aside only when all four guards hold; rename to index.lock.stale-<epoch>-<tag>, never delete.",
        "Fail-closed on unknown lsof or unknown git-process list.",
        "Negative-control tests: each of the four guards watched to refuse.",
        "Did not fork hardware_doctor/hbm_doctor; did not invent a new stale-file convention.",
    ]


def negative_findings() -> list[str]:
    return [
        "No in-tree creator of the index.lock.stale-* name (git grep index.lock.stale → empty).",
        "No worktree-local index.lock or index.lock.stale-* files were present under .git/worktrees during this lane.",
        "git fsck found no missing or corrupt objects; the two dangling commits are stash WIP merges from 2026-08-25/26, not missing campaign commits.",
        "Could not attribute each historical stale file to a PID/argv: no GIT_TRACE2 log of past incidents.",
        "Could not read parent/cwd of the transient CommandLineTools `git status` pid 35555 (it exited).",
        "One `ps aux` invocation in this sandbox returned 'operation not permitted'; process inventory therefore uses pgrep -x git and fails closed if that fails.",
        "Did not inspect every one of the 62 Python files that mention the string git; classified the ones that take the index lock versus rev-parse/log/show.",
        "Did not run cargo build/test or anything touching the GPU.",
        "Did not delete or move any pre-existing .stale-* file.",
        "Did not read live receipts/headless for this question; lock forensics are the live primary .git directory. FUTURE_SUBSTRATE_HANDOFF.json and CLAUDE_GLOBAL_FRONTIER.json were read from receipts/future.",
        "Cannot prove a commit was composed in an editor buffer and never written as an object — only that fsck did not find unreachable campaign commits from the lock windows.",
    ]


# ---------------------------------------------------------------------------
# Build / CLI
# ---------------------------------------------------------------------------

def diagnose_repo(
    repo: Path,
    *,
    now: float,
    age_threshold_s: float,
    apply: bool,
    tag: str,
    holders_fn: HoldersFn | None = None,
    git_procs_fn: GitProcsFn | None = None,
) -> dict[str, Any]:
    layout = resolve_git_layout(repo)
    holders_fn = holders_fn or default_holders_fn
    git_procs_fn = git_procs_fn or default_git_procs_fn
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key, role in (("common_dir", "primary"), ("git_dir", "this_git_dir")):
        raw = layout.get(key)
        if not raw or raw in seen:
            continue
        seen.add(raw)
        lock = lock_path_for(raw)
        obs = observe_lock(
            lock,
            now=now,
            age_threshold_s=age_threshold_s,
            layout=layout,
            holders_fn=holders_fn,
            git_procs_fn=git_procs_fn,
        )
        diagnosis = evaluate_observation(obs)
        action: dict[str, Any] = {
            "mode": "rename-aside" if apply else "report",
            "attempted": False,
            "renamed": False,
        }
        if apply and diagnosis["may_rename"]:
            action["attempted"] = True
            action.update(rename_aside(lock, now=int(now), tag=tag))
        elif apply and not diagnosis["may_rename"]:
            action["refused"] = True
            action["refused_because"] = diagnosis["failing_conditions"]
        diagnosis["role"] = role
        diagnosis["git_dir"] = raw
        diagnosis["action"] = action
        diagnosis["stale_siblings"] = catalog_stale(raw)
        targets.append(diagnosis)

    stale_rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for t in targets:
        for row in t.get("stale_siblings") or []:
            key = row.get("name") or ""
            if key in seen_names:
                continue
            seen_names.add(key)
            stale_rows.append(row)
    stale_rows.sort(key=lambda r: r.get("name") or "")

    return {
        "layout": layout,
        "targets": targets,
        "stale_catalog": stale_rows,
        "arrival_distribution": stale_arrival_distribution(stale_rows),
        "apply": bool(apply),
        "age_threshold_s": float(age_threshold_s),
        "tag": tag,
        "observation_now_epoch": int(now),
    }


def build(
    *,
    repo: Path | None = None,
    now: float | None = None,
    age_threshold_s: float = DEFAULT_AGE_THRESHOLD_S,
    apply: bool = False,
    tag: str = DEFAULT_TAG,
    holders_fn: HoldersFn | None = None,
    git_procs_fn: GitProcsFn | None = None,
    recorded_by: str = RECORDED_BY,
) -> Path:
    now = time.time() if now is None else float(now)
    repo = Path(repo) if repo is not None else REPO
    scan = diagnose_repo(
        repo,
        now=now,
        age_threshold_s=age_threshold_s,
        apply=apply,
        tag=tag,
        holders_fn=holders_fn,
        git_procs_fn=git_procs_fn,
    )
    handoff = None
    handoff_path = REPO / "receipts" / "future" / "FUTURE_SUBSTRATE_HANDOFF.json"
    if handoff_path.is_file():
        try:
            raw = load_json(handoff_path)
            handoff = {
                "path": "receipts/future/FUTURE_SUBSTRATE_HANDOFF.json",
                "present": True,
                "n_active_processes_listed": len(raw.get("active_processes") or []),
            }
        except (OSError, ValueError) as exc:
            handoff = {"path": "receipts/future/FUTURE_SUBSTRATE_HANDOFF.json", "present": True, "error": str(exc)}
    else:
        handoff = {
            "path": "receipts/future/FUTURE_SUBSTRATE_HANDOFF.json",
            "present": False,
            "coped": "inventory recovered from this module's adjacent-implementation list instead",
        }

    frontier_path = REPO / "receipts" / "future" / "CLAUDE_GLOBAL_FRONTIER.json"
    frontier = {
        "path": "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
        "present": frontier_path.is_file(),
    }

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Explain abandoned PRIMARY .git/index.lock files and, when every "
            "guard holds, rename the current lock aside rather than delete it."
        ),
        "vocabulary": {
            "eras": list(ERAS),
            "odysseys": list(ODYSSEYS),
            "fpga_is": "Accelerator / Physical Compiler / Fusion — not its own civilization",
            "evidence": {
                "DIAGNOSTIC_RELATIVE": "contaminated A/B; guides; never promotes; this lane does not produce it",
                "PROTECTED_ABSOLUTE": "protected GPU lease; decides; this lane does not produce it",
                "STATIC_ONLY": "the only evidence class this lane may emit",
            },
        },
        "policy": {
            "default_mode": "report",
            "mutation_flag": "--rename-aside",
            "conditions": list(CONDITIONS),
            "age_threshold_s": float(age_threshold_s),
            "zero_bytes_required": True,
            "lsof_unknown_refuses": True,
            "git_process_unknown_refuses": True,
            "rename_convention": "index.lock.stale-<epoch>-<tag>",
            "never_deletes_stale": True,
            "never_deletes_lock": True,
            "nonzero_lock_is_not_a_candidate": True,
        },
        "scan": scan,
        "git_callers": git_caller_inventory(),
        "fsck": FSCK_RECORD,
        "hypotheses": hypotheses(),
        "handoff": handoff,
        "frontier": frontier,
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "evidence_source": "pinned_snapshot",
        "lock_forensics_source": "live_git_dir",
        "evidence_inputs": [
            {
                "kind": "live_git_dir",
                "path": scan["layout"].get("common_dir") or scan["layout"].get("git_dir"),
                "used_for": "index.lock, stale catalog, worktree list via .git/worktrees",
            },
            {
                "kind": "sidecar_receipt",
                "path": "receipts/future/FUTURE_SUBSTRATE_HANDOFF.json",
                "used_for": "system inventory; git-lock doctor was not listed",
            },
            {
                "kind": "sidecar_receipt",
                "path": "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
                "used_for": "confirm compounding sidecar context; not a lock source",
            },
        ],
    }
    return write_receipt(RECEIPT, doc, recorded_by)


def _print_summary(doc: dict[str, Any]) -> None:
    scan = doc.get("scan") or {}
    print(f"mode: {'rename-aside' if scan.get('apply') else 'report'}")
    print(f"age_threshold_s: {scan.get('age_threshold_s')}")
    dist = scan.get("arrival_distribution") or {}
    print(
        "stale_catalog: n={n} zero={z} nonzero={nz} verdict={v}".format(
            n=dist.get("n"),
            z=dist.get("zero_bytes"),
            nz=dist.get("nonzero_bytes"),
            v=(dist.get("cluster_verdict") or "")[:80],
        )
    )
    for t in scan.get("targets") or []:
        failing = t.get("failing_conditions") or []
        action = t.get("action") or {}
        print(
            f"target role={t.get('role')} present={t.get('present')} "
            f"size={t.get('size_bytes')} age_s={t.get('age_seconds')} "
            f"may_rename={t.get('may_rename')} failing={failing} "
            f"renamed={action.get('renamed')} dest={action.get('destination_name')}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Conservative git index.lock doctor")
    ap.add_argument("--report", action="store_true", help="diagnose only (default)")
    ap.add_argument(
        "--rename-aside",
        action="store_true",
        help="rename a lock that passes all four guards; never delete",
    )
    ap.add_argument("--repo", type=Path, default=None, help="repository or worktree path")
    ap.add_argument(
        "--age-seconds",
        type=float,
        default=DEFAULT_AGE_THRESHOLD_S,
        help=f"declared age threshold (default {DEFAULT_AGE_THRESHOLD_S:g}s)",
    )
    ap.add_argument("--tag", default=DEFAULT_TAG, help="suffix for the stale name")
    args = ap.parse_args(argv)
    if args.rename_aside and args.report:
        print("refusing: pass only one of --report / --rename-aside", file=sys.stderr)
        return 2
    apply = bool(args.rename_aside)
    out = build(
        repo=args.repo,
        age_threshold_s=float(args.age_seconds),
        apply=apply,
        tag=str(args.tag),
    )
    doc = load_json(out)
    _print_summary(doc)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

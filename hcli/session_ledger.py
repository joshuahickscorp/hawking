"""Uncommitted work should never be silently lost.

An interactive session can see `git status` any time it likes; an ultragoal
run cannot, because nobody is watching the terminal. `SessionLedger` gives
both the same three numbers -- files changed, lines changed, minutes since
the last commit -- so the interactive side can OFFER a commit and the
resident side can RECORD one was due, without either side re-deriving git
plumbing of its own.

It deliberately reuses `hcli.landing.IntegrationVerifier`'s own git
subprocess wrapper (`_run`) and porcelain parser (`_parse_status`) instead of
writing a second one: landing.py is the one file in this repo that owns "how
do we talk to git safely" and a second parser here is exactly the kind of
drift a prior audit of this repo found and fixed (see its own module
docstring). This module never calls `git add`, `git commit`, or `git push`
itself -- it only reads.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .landing import IntegrationVerifier
from .paths import find_repo_root

PathLike = Union[str, "os.PathLike[str]"]

# One dict, one place, each number justified. HCLI_LEDGER_FILES /
# HCLI_LEDGER_LINES / HCLI_LEDGER_MINUTES override at runtime.
_ENV_PREFIX = "HCLI_LEDGER_"
_DEFAULT_THRESHOLDS: Dict[str, float] = {
    # A diff touching more files than this is usually several concerns
    # bundled together -- past the point a single commit reviews well.
    "files": 8,
    # ~400 changed lines is roughly where a diff stops fitting in one read;
    # this repo's own review habits already treat that size as needing care.
    "lines": 400,
    # Half an hour of uncommitted work survives a crash or a killed process
    # badly -- this repo has lost accumulated work to a killed worker before.
    "minutes": 30,
}


def _thresholds_from_env() -> Dict[str, float]:
    out = dict(_DEFAULT_THRESHOLDS)
    for key in out:
        raw = os.environ.get(_ENV_PREFIX + key.upper())
        if raw is None:
            continue
        try:
            out[key] = float(raw)
        except ValueError:
            continue
    return out


_SHORTSTAT_INSERT_RE = re.compile(r"(\d+) insertion")
_SHORTSTAT_DELETE_RE = re.compile(r"(\d+) deletion")


def _parse_shortstat(text: str) -> Tuple[int, int]:
    ins = _SHORTSTAT_INSERT_RE.search(text)
    dele = _SHORTSTAT_DELETE_RE.search(text)
    return (int(ins.group(1)) if ins else 0, int(dele.group(1)) if dele else 0)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{int(minutes)}m"
    return f"{minutes / 60.0:.1f}h"


def discover_repo_root(workspace: PathLike) -> Path:
    """The real git top-level for `workspace`.

    `find_repo_root` (paths.py) is shaped to find THIS repo specifically --
    it requires the hawking-only markers (`tools/headless`, `crates`,
    `Cargo.toml`) and silently falls back to the hawking checkout containing
    the running code for any tree that lacks them. That is correct for the
    rest of HCLI, which only ever runs inside this repo, but it is wrong
    here: pointed at a scratch or unrelated git repo it would silently
    resolve to the live hawking checkout instead, which is exactly the tree
    `/land push` and `/land merge` must never touch by accident. Ask git
    itself first; fall back to `find_repo_root` only when `workspace` is not
    inside any git repository at all.
    """
    probe = IntegrationVerifier()._run(Path(workspace), "rev-parse", "--show-toplevel")
    if probe.returncode == 0 and probe.stdout.strip():
        return Path(probe.stdout.strip())
    return find_repo_root(workspace)


def changed_paths(repo_root: PathLike) -> List[str]:
    """Every path git considers changed (tracked or untracked).

    Reuses `IntegrationVerifier._parse_status` so this and `/land`'s own
    allowlist-gathering can never disagree with what LandingService itself
    will see when it re-derives the same status a moment later.
    """
    verifier = IntegrationVerifier()
    status = verifier._run(Path(repo_root), "status", "--porcelain")
    if status.returncode != 0:
        return []
    return verifier._parse_status(status.stdout)


class SessionLedger:
    """Cheap, read-only view of one repo's uncommitted work."""

    def __init__(self, workspace: PathLike, repo_root: Optional[PathLike] = None) -> None:
        self.workspace = Path(workspace)
        self.repo_root = Path(repo_root) if repo_root is not None else discover_repo_root(self.workspace)
        self.thresholds = _thresholds_from_env()
        self._verifier = IntegrationVerifier()
        # What the last offer looked like, so an unchanged state does not
        # re-prompt every turn. None means "nothing offered yet" (also the
        # reset state once the tree goes clean again).
        self._last_offered: Optional[Tuple[int, int, int, int]] = None

    def snapshot(self) -> Dict[str, Any]:
        """Uncommitted work right now. Never raises -- a git call that fails
        (no repo, no commits yet, no upstream) reports zero/None rather than
        propagating, since this is display data, not a gate."""
        try:
            return self._snapshot()
        except Exception:
            return {
                "files_changed": 0, "insertions": 0, "deletions": 0,
                "untracked": 0, "commits_ahead": None, "seconds_since_commit": None,
            }

    def _snapshot(self) -> Dict[str, Any]:
        status = self._verifier._run(self.repo_root, "status", "--porcelain")
        lines = status.stdout.splitlines() if status.returncode == 0 else []
        untracked = sum(1 for line in lines if line[:2] == "??")
        files_changed = sum(1 for line in lines if line.strip() and line[:2] != "??")

        insertions = deletions = 0
        diffstat = self._verifier._run(self.repo_root, "diff", "--shortstat", "HEAD")
        if diffstat.returncode == 0:
            insertions, deletions = _parse_shortstat(diffstat.stdout)

        commits_ahead: Optional[int] = None
        upstream = self._verifier._run(self.repo_root, "rev-list", "--left-right", "--count", "@{u}...HEAD")
        if upstream.returncode == 0:
            parts = upstream.stdout.split()
            if len(parts) == 2 and parts[1].isdigit():
                commits_ahead = int(parts[1])

        seconds_since_commit: Optional[float] = None
        last_commit = self._verifier._run(self.repo_root, "log", "-1", "--format=%ct")
        if last_commit.returncode == 0 and last_commit.stdout.strip():
            try:
                seconds_since_commit = max(0.0, time.time() - float(last_commit.stdout.strip()))
            except ValueError:
                seconds_since_commit = None

        return {
            "files_changed": files_changed,
            "insertions": insertions,
            "deletions": deletions,
            "untracked": untracked,
            "commits_ahead": commits_ahead,
            "seconds_since_commit": seconds_since_commit,
        }

    def should_prompt(
        self, *, at_exit: bool = False, snapshot: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """(prompt?, human reason).

        True once files/lines/minutes crosses a threshold, or -- regardless
        of thresholds -- when `at_exit=True` and anything is uncommitted at
        all, so a session never ends silently forgetting real work. Refuses
        to repeat itself: once a given set of numbers has been offered, the
        same numbers stay quiet on later turns (not at exit -- that is the
        operator's last chance to see them) until they actually move.
        """
        snap = snapshot if snapshot is not None else self.snapshot()
        dirty = snap["files_changed"] > 0 or snap["untracked"] > 0
        if not dirty:
            self._last_offered = None
            return False, "working tree is clean"

        changed_lines = snap["insertions"] + snap["deletions"]
        # Threshold counts every dirty path, tracked or not -- a pile of new
        # untracked files is exactly the kind of accumulated work this exists
        # to surface, even though `snapshot()` reports them separately.
        total_files = snap["files_changed"] + snap["untracked"]
        reasons: List[str] = []
        if total_files >= self.thresholds["files"]:
            reasons.append(f"{total_files} files changed (>= {int(self.thresholds['files'])})")
        if changed_lines >= self.thresholds["lines"]:
            reasons.append(f"{changed_lines} lines changed (>= {int(self.thresholds['lines'])})")
        since = snap["seconds_since_commit"]
        if since is not None and since >= self.thresholds["minutes"] * 60:
            reasons.append(f"{int(since // 60)}m since last commit (>= {int(self.thresholds['minutes'])}m)")

        if not reasons and not at_exit:
            return False, "below thresholds"
        if not reasons:
            reasons.append("session ending with uncommitted work")

        signature = (snap["files_changed"], snap["insertions"], snap["deletions"], snap["untracked"])
        if not at_exit and signature == self._last_offered:
            return False, "already offered; numbers unchanged since last prompt"
        self._last_offered = signature
        return True, "; ".join(reasons)

    def render(self, snapshot: Optional[Dict[str, Any]] = None) -> List[str]:
        """Display lines: the numbers an unwatched session would not see."""
        snap = snapshot if snapshot is not None else self.snapshot()
        lines = [
            f"{snap['files_changed']} file(s) changed",
            f"+{snap['insertions']}/-{snap['deletions']} lines",
            f"{snap['untracked']} untracked",
        ]
        if snap["seconds_since_commit"] is not None:
            lines.append(f"{_fmt_duration(snap['seconds_since_commit'])} since last commit")
        if snap["commits_ahead"] is not None:
            lines.append(f"{snap['commits_ahead']} commit(s) ahead of upstream")
        return lines


__all__ = ["SessionLedger", "changed_paths", "discover_repo_root"]

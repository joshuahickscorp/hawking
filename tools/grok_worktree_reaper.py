#!/usr/bin/env python3
"""Grok worktree reaper — remove only clean Hawking-owned worktrees.

Safety tool for the shared ~/.claude-grok/worktrees directory (used by multiple
repos). Default is dry-run; --apply is required to actually remove anything.

HARD INVARIANTS:
  - Removal happens ONLY via ``git worktree remove`` (never filesystem force-delete).
  - A non-zero exit from ``git worktree remove`` is a stop signal for that path —
    never an escalation to force delete.
  - Non-Hawking worktrees are always skipped.
  - When unsure about cleanliness, treat as dirty.

stdlib only. No network.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ACTIONS = frozenset(
    {
        "skip-not-worktree",
        "skip-not-hawking",
        "skip-dirty",
        "would-remove",
        "removed",
        "refused-stop",
    }
)

_GITDIR_RE = re.compile(r"^gitdir:\s*(.+)\s*$", re.MULTILINE | re.IGNORECASE)


def _run_git(
    *args: str,
    cwd: str | Path | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=check,
    )


def free_gib(path: Path) -> float:
    """Return free disk space in GiB for the filesystem containing *path*."""
    # shutil.disk_usage is inspection-only (not a removal API).
    import shutil

    try:
        usage = shutil.disk_usage(str(path if path.exists() else path.parent))
    except OSError:
        return -1.0
    return usage.free / float(1024**3)


def realpath(path: str | Path) -> str:
    return os.path.realpath(str(path))


def resolve_owner_repo(worktree: Path) -> tuple[str | None, str]:
    """Resolve the main repository root that owns *worktree*.

    Returns (owner_repo_root | None, reason_if_unresolved).
    """
    git_meta = worktree / ".git"
    if not git_meta.exists():
        return None, "not a git worktree: missing .git"

    # Linked worktree: .git is a file containing "gitdir: <MAIN>/.git/worktrees/<name>"
    if git_meta.is_file():
        try:
            text = git_meta.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return None, f"cannot read .git file: {exc}"
        match = _GITDIR_RE.search(text)
        if not match:
            return None, "unrecognized .git file format (no gitdir: line)"
        gitdir = match.group(1).strip()
        # gitdir may be relative to the worktree
        if not os.path.isabs(gitdir):
            gitdir = os.path.join(str(worktree), gitdir)
        gitdir = realpath(gitdir)
        # Expected: <MAIN>/.git/worktrees/<name>
        marker = f"{os.sep}.git{os.sep}worktrees{os.sep}"
        idx = gitdir.rfind(marker)
        if idx != -1:
            main_root = gitdir[:idx]
            if main_root and os.path.isdir(main_root):
                return realpath(main_root), "resolved via .git gitdir"

        # Fallback from the gitdir path itself
        # .../.git/worktrees/<name> -> parent x3 is main root if structure matches
        parts = Path(gitdir).parts
        try:
            wt_idx = parts.index("worktrees")
            if wt_idx >= 1 and parts[wt_idx - 1] == ".git":
                main_root = str(Path(*parts[: wt_idx - 1]))
                if os.path.isdir(main_root):
                    return realpath(main_root), "resolved via .git gitdir path parts"
        except ValueError:
            pass

    # Fallback / cross-check: git-common-dir -> parent of that is main .git dir's parent
    proc = _run_git("-C", str(worktree), "rev-parse", "--git-common-dir")
    if proc.returncode != 0:
        return None, f"not a git worktree: rev-parse failed: {proc.stderr.strip()}"
    common = proc.stdout.strip()
    if not common:
        return None, "empty git-common-dir"
    if not os.path.isabs(common):
        common = os.path.join(str(worktree), common)
    common = realpath(common)
    # common is typically <MAIN>/.git
    if os.path.basename(common) == ".git":
        main_root = os.path.dirname(common)
    else:
        # bare or unusual layout: parent of common dir
        main_root = os.path.dirname(common)
    if not main_root or not os.path.isdir(main_root):
        return None, f"cannot derive main repo root from git-common-dir={common}"
    return realpath(main_root), "resolved via git-common-dir"


def _head_sha(worktree: Path) -> str | None:
    proc = _run_git("-C", str(worktree), "rev-parse", "HEAD")
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def _head_reachable_from_main(main_repo: str, head: str) -> bool | None:
    """Return True if HEAD is contained in some main-repo ref.

    None means unsure (caller must treat as dirty).
    """
    proc = _run_git("-C", main_repo, "for-each-ref", "--format=%(objectname)")
    if proc.returncode != 0:
        return None
    tips = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not tips:
        # No refs at all — conservative dirty unless HEAD is unborn edge case
        return None
    for tip in tips:
        check = _run_git(
            "-C", main_repo, "merge-base", "--is-ancestor", head, tip
        )
        if check.returncode == 0:
            return True
        # returncode 1 => not an ancestor; other codes => error
        if check.returncode not in (0, 1):
            return None
    return False


def is_worktree_clean(worktree: Path, main_repo: str) -> tuple[bool, str]:
    """Conservative cleanliness check.

    Clean only if porcelain is empty AND HEAD is reachable from a main-repo ref.
    When unsure, treat as dirty.
    """
    status = _run_git("-C", str(worktree), "status", "--porcelain")
    if status.returncode != 0:
        return False, f"git status failed: {status.stderr.strip()}"
    if status.stdout.strip():
        return False, "uncommitted or untracked changes present"

    head = _head_sha(worktree)
    if head is None:
        return False, "cannot resolve HEAD"

    reachable = _head_reachable_from_main(main_repo, head)
    if reachable is None:
        return False, "unable to verify HEAD reachability from main refs (treat as dirty)"
    if not reachable:
        return False, "HEAD is not contained in any main-repo ref (would lose commits)"

    return True, "clean: empty porcelain and HEAD reachable from main refs"


def remove_worktree(main_repo: str, worktree: Path) -> tuple[bool, str]:
    """Attempt ``git -C <MAIN> worktree remove <D>``.

    On failure: return (False, stderr). Never force-deletes.
    """
    proc = _run_git("-C", main_repo, "worktree", "remove", str(worktree))
    if proc.returncode == 0:
        return True, "removed via git worktree remove"
    err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
    return False, err


def classify_worktree(
    worktree: Path,
    hawking_repo: str,
    apply: bool,
) -> dict[str, Any]:
    """Classify one immediate child and optionally remove it."""
    record: dict[str, Any] = {
        "path": str(worktree),
        "owner_repo": None,
        "is_hawking": False,
        "clean": False,
        "action": "skip-not-worktree",
        "reason": "",
    }

    if not worktree.is_dir():
        record["action"] = "skip-not-worktree"
        record["reason"] = "not a directory"
        return record

    owner, resolve_reason = resolve_owner_repo(worktree)
    if owner is None:
        record["action"] = "skip-not-worktree"
        record["reason"] = resolve_reason
        return record

    record["owner_repo"] = owner
    is_hawking = realpath(owner) == realpath(hawking_repo)
    record["is_hawking"] = is_hawking

    if not is_hawking:
        record["action"] = "skip-not-hawking"
        record["reason"] = f"owner is not Hawking repo ({owner})"
        return record

    clean, clean_reason = is_worktree_clean(worktree, owner)
    record["clean"] = clean
    if not clean:
        record["action"] = "skip-dirty"
        record["reason"] = clean_reason
        return record

    if not apply:
        record["action"] = "would-remove"
        record["reason"] = "dry-run: would run git worktree remove"
        return record

    ok, msg = remove_worktree(owner, worktree)
    if ok:
        record["action"] = "removed"
        record["reason"] = msg
        return record

    record["action"] = "refused-stop"
    record["reason"] = f"git worktree remove refused: {msg}"
    return record


def scan_worktrees(
    hawking_repo: str,
    worktrees_dir: Path,
    apply: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not worktrees_dir.is_dir():
        return records

    children = sorted(
        (p for p in worktrees_dir.iterdir() if not p.name.startswith(".")),
        key=lambda p: p.name,
    )
    for child in children:
        if not child.is_dir():
            continue
        records.append(classify_worktree(child, hawking_repo, apply=apply))
    return records


def build_report(
    records: list[dict[str, Any]],
    *,
    hawking_repo: str,
    worktrees_dir: Path,
    apply: bool,
    min_free_gib: float,
    free_before: float,
    free_after: float,
) -> dict[str, Any]:
    return {
        "repo": realpath(hawking_repo),
        "worktrees_dir": realpath(worktrees_dir) if worktrees_dir.exists() else str(worktrees_dir),
        "apply": apply,
        "min_free_gib": min_free_gib,
        "free_gib_before": free_before,
        "free_gib_after": free_after,
        "records": records,
        "summary": {
            "total": len(records),
            **{
                action: sum(1 for r in records if r["action"] == action)
                for action in sorted(ACTIONS)
            },
        },
    }


def summary_line(report: dict[str, Any]) -> str:
    s = report["summary"]
    return (
        f"grok_worktree_reaper: apply={report['apply']} "
        f"total={s['total']} "
        f"removed={s.get('removed', 0)} "
        f"would-remove={s.get('would-remove', 0)} "
        f"skip-not-hawking={s.get('skip-not-hawking', 0)} "
        f"skip-dirty={s.get('skip-dirty', 0)} "
        f"skip-not-worktree={s.get('skip-not-worktree', 0)} "
        f"refused-stop={s.get('refused-stop', 0)} "
        f"free_gib_before={report['free_gib_before']:.2f} "
        f"free_gib_after={report['free_gib_after']:.2f}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_wt = Path.home() / ".claude-grok" / "worktrees"
    p = argparse.ArgumentParser(
        description=(
            "Remove only clean Hawking-owned git worktrees from a shared "
            "worktrees directory. Default is dry-run."
        )
    )
    p.add_argument(
        "--repo",
        required=True,
        help="Absolute path to the Hawking repository root",
    )
    p.add_argument(
        "--worktrees-dir",
        default=str(default_wt),
        help=f"Directory whose immediate children are candidate worktrees (default: {default_wt})",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually run git worktree remove (default: dry-run)",
    )
    p.add_argument(
        "--min-free-gib",
        type=float,
        default=15.0,
        help="Emergency floor recorded in the report (does not block --apply; default: 15)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit the JSON report to stdout (always emitted; flag kept for CLI clarity)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hawking_repo = realpath(args.repo)
    worktrees_dir = Path(args.worktrees_dir).expanduser()

    if not os.path.isdir(hawking_repo):
        print(f"error: --repo is not a directory: {hawking_repo}", file=sys.stderr)
        return 2

    free_before = free_gib(worktrees_dir)
    records = scan_worktrees(hawking_repo, worktrees_dir, apply=args.apply)
    free_after = free_gib(worktrees_dir)

    report = build_report(
        records,
        hawking_repo=hawking_repo,
        worktrees_dir=worktrees_dir,
        apply=args.apply,
        min_free_gib=args.min_free_gib,
        free_before=free_before,
        free_after=free_after,
    )

    # JSON report to stdout; human summary to stderr
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    print(summary_line(report), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

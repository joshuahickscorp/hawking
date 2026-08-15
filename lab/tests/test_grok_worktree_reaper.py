"""Safety tests for tools/grok_worktree_reaper.py.

Acceptance: the reaper must refuse cross-repo and dirty worktrees, never
force-delete when git worktree remove refuses, and only remove clean
Hawking-owned worktrees via ``git worktree remove``.

Everything runs under tmp_path — never touches ~/.claude-grok or the real
Hawking checkout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REAPER = REPO_ROOT / "tools" / "grok_worktree_reaper.py"


def _git(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=check,
    )


def _init_repo(path: Path, name: str = "init") -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git("init", str(path))
    _git("-C", str(path), "config", "user.email", "reaper-test@example.com")
    _git("-C", str(path), "config", "user.name", "Reaper Test")
    # Avoid "main"/"master" ambiguity across git versions
    _git("-C", str(path), "checkout", "-B", "main")
    (path / "README.md").write_text(f"# {name}\n", encoding="utf-8")
    _git("-C", str(path), "add", "README.md")
    _git("-C", str(path), "commit", "-m", f"init {name}")


def _run_reaper(
    *,
    repo: Path,
    worktrees_dir: Path,
    apply: bool = False,
) -> tuple[dict, str, int]:
    cmd = [
        sys.executable,
        str(REAPER),
        "--repo",
        str(repo),
        "--worktrees-dir",
        str(worktrees_dir),
        "--json",
    ]
    if apply:
        cmd.append("--apply")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"reaper exited {proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    report = json.loads(proc.stdout)
    return report, proc.stderr, proc.returncode


def _by_name(records: list[dict], name: str) -> dict:
    matches = [r for r in records if Path(r["path"]).name == name]
    assert len(matches) == 1, f"expected one record for {name!r}, got {matches!r}"
    return matches[0]


def test_source_contains_no_force_delete_apis() -> None:
    """Static invariant: no unconditional filesystem removal of worktree paths."""
    src = REAPER.read_text(encoding="utf-8")
    assert "rmtree" not in src
    assert "rm -rf" not in src
    assert "rm -fr" not in src
    # No subprocess call that shells out to rm for deletion
    assert '["rm"' not in src
    assert "['rm'" not in src
    assert "os.remove" not in src
    assert "os.unlink" not in src
    assert "pathlib.Path.unlink" not in src
    assert ".unlink(" not in src
    # Removal must go through git worktree remove only
    assert "worktree" in src and "remove" in src


def test_reaper_safety_paths(tmp_path: Path) -> None:
    """(a)(b)(c)(d) acceptance scenarios under an isolated worktrees dir."""
    hawking = tmp_path / "hawking-repo"
    other = tmp_path / "other-repo"
    wtdir = tmp_path / "shared-worktrees"
    wtdir.mkdir()

    _init_repo(hawking, "hawking")
    _init_repo(other, "other")

    # (a) worktree of a NON-Hawking repo
    other_wt = wtdir / "other-wt"
    _git("-C", str(other), "worktree", "add", "-b", "other-branch", str(other_wt))

    # (b) Hawking-owned worktree with an uncommitted change
    dirty_wt = wtdir / "dirty-wt"
    _git("-C", str(hawking), "worktree", "add", "-b", "dirty-branch", str(dirty_wt))
    (dirty_wt / "uncommitted.txt").write_text("dirt\n", encoding="utf-8")

    # (c) Hawking-owned worktree that git worktree remove will refuse:
    #     clean porcelain + reachable HEAD, but locked.
    locked_wt = wtdir / "locked-wt"
    _git("-C", str(hawking), "worktree", "add", "-b", "locked-branch", str(locked_wt))
    _git("-C", str(hawking), "worktree", "lock", str(locked_wt), "--reason", "test-lock")

    # (d) clean Hawking-owned worktree
    clean_wt = wtdir / "clean-wt"
    _git("-C", str(hawking), "worktree", "add", "-b", "clean-branch", str(clean_wt))

    # --- dry-run ---
    dry, _stderr, _rc = _run_reaper(repo=hawking, worktrees_dir=wtdir, apply=False)
    assert dry["apply"] is False
    assert "free_gib_before" in dry and "free_gib_after" in dry

    a = _by_name(dry["records"], "other-wt")
    assert a["action"] == "skip-not-hawking"
    assert a["is_hawking"] is False

    b = _by_name(dry["records"], "dirty-wt")
    assert b["action"] == "skip-dirty"
    assert b["is_hawking"] is True
    assert b["clean"] is False

    c = _by_name(dry["records"], "locked-wt")
    assert c["is_hawking"] is True
    assert c["clean"] is True
    assert c["action"] == "would-remove"  # dry-run does not attempt remove

    d = _by_name(dry["records"], "clean-wt")
    assert d["action"] == "would-remove"
    assert d["is_hawking"] is True
    assert d["clean"] is True

    # All dirs still present after dry-run
    assert other_wt.is_dir()
    assert dirty_wt.is_dir()
    assert locked_wt.is_dir()
    assert clean_wt.is_dir()

    # --- apply ---
    applied, _stderr2, _rc2 = _run_reaper(repo=hawking, worktrees_dir=wtdir, apply=True)
    assert applied["apply"] is True

    a2 = _by_name(applied["records"], "other-wt")
    assert a2["action"] == "skip-not-hawking"
    assert other_wt.is_dir(), "non-Hawking worktree must survive --apply"

    b2 = _by_name(applied["records"], "dirty-wt")
    assert b2["action"] == "skip-dirty"
    assert dirty_wt.is_dir(), "dirty Hawking worktree must survive --apply"

    c2 = _by_name(applied["records"], "locked-wt")
    assert c2["action"] == "refused-stop"
    assert locked_wt.is_dir(), (
        "when git worktree remove refuses, dir must still exist (no force delete)"
    )

    d2 = _by_name(applied["records"], "clean-wt")
    assert d2["action"] == "removed"
    assert not clean_wt.exists(), "clean Hawking worktree should be removed"

    # git worktree list on hawking must no longer show clean-wt
    listing = _git("-C", str(hawking), "worktree", "list", "--porcelain")
    assert str(clean_wt.resolve()) not in listing.stdout
    # locked and dirty still registered
    assert str(locked_wt.resolve()) in listing.stdout
    assert str(dirty_wt.resolve()) in listing.stdout

    # other repo worktree untouched
    other_listing = _git("-C", str(other), "worktree", "list", "--porcelain")
    assert str(other_wt.resolve()) in other_listing.stdout


def test_skip_non_worktree_directory(tmp_path: Path) -> None:
    hawking = tmp_path / "hawking-repo"
    wtdir = tmp_path / "shared-worktrees"
    wtdir.mkdir()
    _init_repo(hawking, "hawking")

    plain = wtdir / "not-a-worktree"
    plain.mkdir()
    (plain / "notes.txt").write_text("hello\n", encoding="utf-8")

    report, _, _ = _run_reaper(repo=hawking, worktrees_dir=wtdir, apply=True)
    rec = _by_name(report["records"], "not-a-worktree")
    assert rec["action"] == "skip-not-worktree"
    assert plain.is_dir()


def test_min_free_gib_does_not_block_apply(tmp_path: Path) -> None:
    """Even with a high min-free-gib floor, --apply still runs (frees disk)."""
    hawking = tmp_path / "hawking-repo"
    wtdir = tmp_path / "shared-worktrees"
    wtdir.mkdir()
    _init_repo(hawking, "hawking")

    clean_wt = wtdir / "clean-wt"
    _git("-C", str(hawking), "worktree", "add", "-b", "clean-branch", str(clean_wt))

    cmd = [
        sys.executable,
        str(REAPER),
        "--repo",
        str(hawking),
        "--worktrees-dir",
        str(wtdir),
        "--apply",
        "--min-free-gib",
        "999999",
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    report = json.loads(proc.stdout)
    assert report["min_free_gib"] == 999999
    rec = _by_name(report["records"], "clean-wt")
    assert rec["action"] == "removed"
    assert not clean_wt.exists()

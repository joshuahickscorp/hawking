"""hcli.checkpoint against REAL scratch git repositories (never the Hawking
repo itself), including one real ``git worktree add`` tree -- a worktree's
``.git`` is a file, not a directory, and that is the only thing that proves
find_repo_root / checkpoint / restore actually cope with it.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from hcli._test_git import scratch_repo

import pytest

from hcli.checkpoint import CheckpointError, checkpoint, list_checkpoints, restore_checkpoint
from hcli.paths import find_repo_root


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True, timeout=10)


def _repo(tmp_path: Path, name: str = "repo") -> Path:
    return scratch_repo(
        tmp_path / name,
        email="hcli-checkpoint-test@example.com",
        name="hcli-checkpoint-test",
        filename="tracked.txt",
        body="v1\n",
    )


def _porcelain(repo: Path) -> str:
    return _run("status", "--porcelain", cwd=repo).stdout


def _head(repo: Path) -> str:
    return _run("rev-parse", "HEAD", cwd=repo).stdout.strip()


# --- checkpoint() never disturbs the working tree ----------------------------


def test_checkpoint_does_not_disturb_working_tree(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("v1\nDIRTY\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")

    before_status = _porcelain(repo)
    before_head = _head(repo)

    result = checkpoint(repo, "mid-edit")

    assert result["commit_sha"]
    assert result["parent_sha"] == before_head
    # The whole point: the live tree and the real index are untouched.
    assert _porcelain(repo) == before_status
    assert _head(repo) == before_head
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "v1\nDIRTY\n"


def test_checkpoint_on_fresh_repo_with_no_head_yet(tmp_path):
    repo = tmp_path / "empty_repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "x@x.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "x"], cwd=repo, check=True)
    (repo / "a.txt").write_text("a\n", encoding="utf-8")

    result = checkpoint(repo, "first-ever")
    assert result["parent_sha"] is None
    assert result["commit_sha"]


# --- checkpoint() captures dirty + untracked, respects .gitignore ------------


def test_restore_recovers_dirty_and_untracked_but_not_gitignored(tmp_path):
    repo = _repo(tmp_path)
    (repo / ".gitignore").write_text("ignored.log\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "gitignore"], cwd=repo, check=True)

    (repo / "tracked.txt").write_text("v1\nDIRTY\n", encoding="utf-8")
    (repo / "new_untracked.txt").write_text("brand new\n", encoding="utf-8")
    (repo / "ignored.log").write_text("should not be captured\n", encoding="utf-8")

    result = checkpoint(repo, "snapshot-1")

    dest = tmp_path / "restore-1"
    restored = restore_checkpoint(repo, result["checkpoint_id"], into=dest)

    assert restored["commit_sha"] == result["commit_sha"]
    assert (dest / "tracked.txt").read_text(encoding="utf-8") == "v1\nDIRTY\n"
    assert (dest / "new_untracked.txt").read_text(encoding="utf-8") == "brand new\n"
    assert not (dest / "ignored.log").exists(), "gitignored files must NOT be captured"
    assert restored["files_restored"] == 3  # tracked.txt, .gitignore, new_untracked.txt


# --- list_checkpoints ---------------------------------------------------------


def test_list_checkpoints_newest_first_with_labels(tmp_path):
    repo = _repo(tmp_path)
    first = checkpoint(repo, "alpha")
    time.sleep(0.01)
    second = checkpoint(repo, "beta")

    listed = list_checkpoints(repo)
    ids = [c["checkpoint_id"] for c in listed]
    assert ids.index(second["checkpoint_id"]) < ids.index(first["checkpoint_id"])
    by_id = {c["checkpoint_id"]: c for c in listed}
    assert by_id[first["checkpoint_id"]]["label"] == "alpha"
    assert by_id[second["checkpoint_id"]]["label"] == "beta"
    assert by_id[first["checkpoint_id"]]["commit_sha"] == first["commit_sha"]


def test_list_checkpoints_empty_when_none_taken(tmp_path):
    repo = _repo(tmp_path)
    assert list_checkpoints(repo) == []


# --- restore_checkpoint refusals ----------------------------------------------


def test_restore_refuses_the_live_working_tree(tmp_path):
    repo = _repo(tmp_path)
    result = checkpoint(repo, "snap")
    with pytest.raises(CheckpointError, match="live working tree"):
        restore_checkpoint(repo, result["checkpoint_id"], into=repo)


def test_restore_refuses_a_path_inside_the_live_tree(tmp_path):
    repo = _repo(tmp_path)
    result = checkpoint(repo, "snap")
    # Never created before this call, so it IS empty -- only the "inside
    # repo_root" guard can be what stops this.
    nested = repo / "some" / "new_restore_dir"
    with pytest.raises(CheckpointError, match="live working tree"):
        restore_checkpoint(repo, result["checkpoint_id"], into=nested)


def test_restore_refuses_a_nonempty_destination(tmp_path):
    repo = _repo(tmp_path)
    result = checkpoint(repo, "snap")
    dest = tmp_path / "occupied"
    dest.mkdir()
    (dest / "already_here.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(CheckpointError, match="not empty"):
        restore_checkpoint(repo, result["checkpoint_id"], into=dest)


def test_restore_unknown_checkpoint_id_raises(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(CheckpointError, match="no such checkpoint"):
        restore_checkpoint(repo, "does-not-exist", into=tmp_path / "wherever")


def test_checkpoint_refuses_non_git_directory(tmp_path):
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    with pytest.raises(CheckpointError, match="not a git repository"):
        checkpoint(plain, "x")


# --- real git worktree: .git is a FILE here, not a directory -----------------


def test_checkpoint_and_restore_against_a_real_worktree(tmp_path):
    main_repo = _repo(tmp_path, name="main")
    # find_repo_root identifies the HAWKING repo specifically by structural
    # marker (tools/headless or Cargo.toml), not "any git repo" -- give this
    # scratch repo that marker so the assertion below exercises the real
    # worktree-detection path instead of falling through to paths.py's own
    # on-disk location.
    (main_repo / "tools" / "headless").mkdir(parents=True)
    (main_repo / "tools" / "headless" / "marker.txt").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "tools"], cwd=main_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "hawking marker"], cwd=main_repo, check=True)
    _run("branch", "wt-branch", cwd=main_repo)
    wt = tmp_path / "linked-worktree"
    subprocess.run(
        ["git", "-C", str(main_repo), "worktree", "add", "-q", str(wt), "wt-branch"],
        check=True,
    )

    # Confirm the premise: .git under a linked worktree is a file.
    assert (wt / ".git").is_file()
    assert not (wt / ".git").is_dir()

    # find_repo_root must resolve the worktree's OWN root, not the main repo's.
    assert find_repo_root(wt) == wt.resolve()

    (wt / "tracked.txt").write_text("v1\nedited-in-worktree\n", encoding="utf-8")
    (wt / "new_in_worktree.txt").write_text("hello\n", encoding="utf-8")
    before_status = _porcelain(wt)
    before_head = _head(wt)

    result = checkpoint(wt, "worktree-snapshot")

    # Working tree of the LINKED worktree is untouched.
    assert _porcelain(wt) == before_status
    assert _head(wt) == before_head

    # The checkpoint ref lives in the shared object store: visible from the
    # main checkout too, not just the worktree that made it.
    main_listing = list_checkpoints(main_repo)
    assert any(c["commit_sha"] == result["commit_sha"] for c in main_listing)

    dest = tmp_path / "restored-from-worktree"
    restored = restore_checkpoint(wt, result["checkpoint_id"], into=dest)
    assert restored["commit_sha"] == result["commit_sha"]
    assert (dest / "tracked.txt").read_text(encoding="utf-8") == "v1\nedited-in-worktree\n"
    assert (dest / "new_in_worktree.txt").read_text(encoding="utf-8") == "hello\n"

"""Mission.fingerprint must not read the whole workspace.

The change-detector `Mission.run()` calls before the first WorkUnit -- and
`_heartbeat` calls again every 30s -- used to `os.walk` the workspace and
`read_bytes()` EVERY file. On a repo carrying model artifacts and activation
captures that never finished, so the resident daemon sat at body=LOADING
burning CPU on `.f32le` dumps and never reached a model call. In a git
worktree the same question is answered from git's index.
"""
from __future__ import annotations

import subprocess
import time

import pytest

from hcli.mission import Mission


def _git(tmp_path, *args):
    return subprocess.run(
        ["git", "-C", str(tmp_path), *args], capture_output=True, check=True
    )


@pytest.fixture()
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "seed.txt").write_text("seed\n")
    _git(tmp_path, "add", "seed.txt")
    _git(tmp_path, "commit", "-qm", "seed")
    return tmp_path


def test_a_git_workspace_is_not_read_byte_by_byte(repo, monkeypatch):
    """The load-bearing guard: no file contents are read in a git worktree."""
    import pathlib

    def explode(self, *a, **k):  # pragma: no cover - only fires on regression
        raise AssertionError(f"fingerprint read file contents: {self}")

    mission = Mission(repo, goal="a goal")
    monkeypatch.setattr(pathlib.Path, "read_bytes", explode)
    assert len(mission.fingerprint()) == 20


def test_the_fingerprint_still_moves_when_the_tree_changes(repo):
    mission = Mission(repo, goal="a goal")
    before = mission.fingerprint()

    (repo / "new.txt").write_text("added\n")
    assert mission.fingerprint() != before, "an untracked file must register"

    after_add = mission.fingerprint()
    time.sleep(0.01)
    (repo / "seed.txt").write_text("edited once\n")
    once = mission.fingerprint()
    assert once != after_add, "a modified tracked file must register"

    # A SECOND edit of an already-dirty file: bare `git status` output is
    # identical here, so size/mtime is what keeps the detector honest.
    time.sleep(0.01)
    (repo / "seed.txt").write_text("edited twice, and longer\n")
    assert mission.fingerprint() != once, "a re-edit of a dirty file must register"


def test_a_plain_directory_still_uses_the_content_walk(tmp_path):
    (tmp_path / "a.txt").write_text("one\n")
    mission = Mission(tmp_path, goal="a goal")
    before = mission.fingerprint()
    (tmp_path / "a.txt").write_text("two\n")
    assert mission.fingerprint() != before
    assert mission._git_fingerprint(tmp_path) is None

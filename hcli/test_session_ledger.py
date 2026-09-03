"""SessionLedger: the numbers behind "did i always have to tell it commit
push merge?" An interactive session can see accumulated uncommitted work any
time by asking; an unattended mission slice cannot, because nobody is
watching. This module is the one place both read the same three signals
(files changed, lines changed, minutes since the last commit) from real git,
never a mock.

Also covers `/land`'s merge and push verbs (`CommandHandler._land_merge`,
`_land_push`), per the task's own instruction to keep those tests here
rather than starting a second scratch-repo test module for them. Both reuse
`hcli.landing.IntegrationVerifier`'s git subprocess wrapper the same way
`SessionLedger` does, so one real git repo fixture serves every test in this
file.

Runnable two ways:

    python3 -m pytest hcli/test_session_ledger.py -q
    python3 hcli/test_session_ledger.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hcli._test_git import scratch_repo

from hcli.commands import CommandHandler
from hcli.session_ledger import SessionLedger, changed_paths, discover_repo_root


def _repo(tmp_path: Path, name: str = "repo") -> Path:
    """A scratch git repository, never the Hawking repo, fresh per test."""
    return scratch_repo(
        tmp_path / name,
        email="ledger-test@example.com",
        name="ledger-test",
        filename="f.txt",
        body="a\nb\nc\n",
    )


def _head(repo: Path, ref: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", ref],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, timeout=10,
    )


class _StubController:
    """Just enough for `_workspace_root` (commands.py) to find the repo."""

    def __init__(self, repo: Path) -> None:
        self.workspace_root = str(repo)


# --- discover_repo_root: the landmine this module exists to avoid ------------


def test_discover_repo_root_finds_the_scratch_repo_not_the_hawking_checkout(tmp_path):
    """`find_repo_root` (paths.py) is shaped to find HCLI's own checkout and
    silently falls back to it for any tree lacking hawking's markers -- a
    scratch repo has none. `discover_repo_root` must resolve to the scratch
    repo itself, never to whatever repo the test happens to run inside."""
    repo = _repo(tmp_path)
    resolved = discover_repo_root(repo)
    assert resolved == repo.resolve(), (resolved, repo)


# --- snapshot: the numbers must be right --------------------------------------


def test_snapshot_numbers_are_right(tmp_path):
    repo = _repo(tmp_path)
    (repo / "f.txt").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")  # +2/-0
    (repo / "new_untracked.txt").write_text("x\n", encoding="utf-8")

    snap = SessionLedger(repo, repo_root=repo).snapshot()

    assert snap["files_changed"] == 1, snap
    assert snap["insertions"] == 2, snap
    assert snap["deletions"] == 0, snap
    assert snap["untracked"] == 1, snap
    assert snap["commits_ahead"] is None, "no upstream configured"
    assert snap["seconds_since_commit"] is not None and snap["seconds_since_commit"] >= 0, snap


def test_snapshot_counts_deletions_too(tmp_path):
    repo = _repo(tmp_path)
    (repo / "f.txt").write_text("a\n", encoding="utf-8")  # removes b, c

    snap = SessionLedger(repo, repo_root=repo).snapshot()

    assert snap["insertions"] == 0, snap
    assert snap["deletions"] == 2, snap


def test_snapshot_on_a_clean_tree_is_all_zero(tmp_path):
    repo = _repo(tmp_path)
    snap = SessionLedger(repo, repo_root=repo).snapshot()
    assert snap["files_changed"] == 0
    assert snap["untracked"] == 0
    assert snap["insertions"] == 0
    assert snap["deletions"] == 0


def test_snapshot_never_raises_when_there_is_no_repo_at_all(tmp_path):
    empty = tmp_path / "not_a_repo"
    empty.mkdir()
    snap = SessionLedger(empty, repo_root=empty).snapshot()
    assert snap == {
        "files_changed": 0, "insertions": 0, "deletions": 0,
        "untracked": 0, "commits_ahead": None, "seconds_since_commit": None,
    }


# --- should_prompt: fires on threshold, stays quiet on repeats ---------------


def test_should_prompt_stays_quiet_below_every_threshold(tmp_path):
    repo = _repo(tmp_path)
    (repo / "f.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")  # 1 file, 1 line

    ledger = SessionLedger(repo, repo_root=repo)
    prompt, reason = ledger.should_prompt()

    assert prompt is False, reason
    assert reason == "below thresholds"


def test_should_prompt_fires_once_the_files_threshold_is_crossed(tmp_path):
    repo = _repo(tmp_path)
    for i in range(9):
        (repo / f"n{i}.txt").write_text("x\n", encoding="utf-8")  # 9 untracked

    ledger = SessionLedger(repo, repo_root=repo)
    prompt, reason = ledger.should_prompt()

    assert prompt is True
    assert "files changed" in reason and "9" in reason, reason


def test_should_prompt_does_not_refire_on_the_same_unchanged_state(tmp_path):
    repo = _repo(tmp_path)
    for i in range(9):
        (repo / f"n{i}.txt").write_text("x\n", encoding="utf-8")
    ledger = SessionLedger(repo, repo_root=repo)

    first = ledger.should_prompt()
    second = ledger.should_prompt()
    third = ledger.should_prompt()

    assert first == (True, "9 files changed (>= 8)")
    assert second == (False, "already offered; numbers unchanged since last prompt")
    assert third == second, "must stay quiet, not just once but every repeat"


def test_should_prompt_fires_again_once_the_numbers_actually_move(tmp_path):
    repo = _repo(tmp_path)
    for i in range(9):
        (repo / f"n{i}.txt").write_text("x\n", encoding="utf-8")
    ledger = SessionLedger(repo, repo_root=repo)
    assert ledger.should_prompt()[0] is True
    assert ledger.should_prompt()[0] is False, "must not refire yet"

    (repo / "n9.txt").write_text("x\n", encoding="utf-8")  # numbers moved
    prompt, reason = ledger.should_prompt()
    assert prompt is True, "10 untracked files must re-offer after 9 already was"
    assert "10" in reason


def test_should_prompt_resets_once_the_tree_goes_clean(tmp_path):
    repo = _repo(tmp_path)
    for i in range(9):
        (repo / f"n{i}.txt").write_text("x\n", encoding="utf-8")
    ledger = SessionLedger(repo, repo_root=repo)
    assert ledger.should_prompt()[0] is True
    assert ledger.should_prompt()[0] is False

    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "checkpoint"], check=True, capture_output=True)
    assert ledger.should_prompt() == (False, "working tree is clean")

    for i in range(9):
        (repo / f"m{i}.txt").write_text("x\n", encoding="utf-8")
    prompt, reason = ledger.should_prompt()
    assert prompt is True, "a fresh dirty tree with the SAME numbers must offer again after going clean"


def test_should_prompt_at_exit_fires_below_every_threshold(tmp_path):
    """The session-ending case: even one small uncommitted file must be
    surfaced, because there will be no next turn to catch it."""
    repo = _repo(tmp_path)
    (repo / "f.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
    ledger = SessionLedger(repo, repo_root=repo)

    assert ledger.should_prompt()[0] is False, "not at exit: below thresholds stays quiet"
    prompt, reason = ledger.should_prompt(at_exit=True)
    assert prompt is True
    assert reason == "session ending with uncommitted work"


def test_should_prompt_at_exit_stays_quiet_on_a_clean_tree(tmp_path):
    repo = _repo(tmp_path)
    ledger = SessionLedger(repo, repo_root=repo)
    assert ledger.should_prompt(at_exit=True) == (False, "working tree is clean")


# --- changed_paths reuses landing's own porcelain parser ----------------------


def test_changed_paths_matches_what_landing_itself_would_see(tmp_path):
    repo = _repo(tmp_path)
    (repo / "f.txt").write_text("changed\n", encoding="utf-8")
    (repo / "u.txt").write_text("new\n", encoding="utf-8")

    paths = changed_paths(repo)
    assert set(paths) == {"f.txt", "u.txt"}


# --- /land merge: fast-forward only, never checks anything out ---------------


def test_land_merge_refuses_a_target_that_is_not_a_strict_ancestor(tmp_path):
    repo = _repo(tmp_path)
    _git(repo, "branch", "other")
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "other"], check=True)
    (repo / "f.txt").write_text("only on other\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-am", "diverge"], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "main"], check=True)
    before = _head(repo, "other")

    handler = CommandHandler(_StubController(repo))
    text = handler.handle("/land merge other")

    assert handler.last_value["merged"] is False
    assert handler.last_value["reason"] == "NOT_FAST_FORWARD", handler.last_value
    assert "not a strict ancestor" in text
    assert _head(repo, "other") == before, "a refused merge must not move the branch pointer"
    assert _git(repo, "symbolic-ref", "--short", "HEAD").stdout.strip() == "main", (
        "a refused merge must never check anything out"
    )


def test_land_merge_fast_forwards_a_strict_ancestor_without_checking_it_out(tmp_path):
    repo = _repo(tmp_path)
    _git(repo, "branch", "old")  # old == the initial commit, a strict ancestor of what main becomes
    (repo / "f.txt").write_text("main moved on\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-am", "advance main"], check=True)
    head = _head(repo, "main")
    assert _head(repo, "old") != head

    handler = CommandHandler(_StubController(repo))
    text = handler.handle("/land merge old")

    assert handler.last_value == {"merged": True, "target": "old", "advanced_by": 1}, handler.last_value
    assert "Fast-forwarded" in text
    assert _head(repo, "old") == head, "old must now point at main's HEAD"
    assert _git(repo, "symbolic-ref", "--short", "HEAD").stdout.strip() == "main", (
        "merging must never check the target branch out -- a live worker respawns from these files"
    )


def test_land_merge_refuses_an_unknown_branch(tmp_path):
    repo = _repo(tmp_path)
    handler = CommandHandler(_StubController(repo))
    text = handler.handle("/land merge does-not-exist")
    assert handler.last_value["reason"] == "NO_SUCH_BRANCH"
    assert "no such branch" in text


# --- /land push: only the explicit verb ever reaches a remote ----------------


def _with_bare_remote(repo: Path, tmp_path: Path) -> Path:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", str(remote)], check=True)
    subprocess.run(["git", "-C", str(repo), "push", "-q", "-u", "origin", "main"], check=True)
    return remote


def _with_a_passing_test(repo: Path) -> None:
    """/land's default test_command is `python3 -m pytest hcli/ -q` -- give
    the scratch repo a trivial `hcli/` so that command has something real to
    run and pass, instead of pytest's "no tests collected" usage error.

    The directory itself is seeded and committed first: an untracked
    directory is one `git status --porcelain` line ("?? hcli/"), not one per
    file, which does not match the per-file allowlist `/land` builds. Only
    the test file inside it is left as the actual uncommitted addition.
    """
    hcli_dir = repo / "hcli"
    hcli_dir.mkdir(exist_ok=True)
    keep = hcli_dir / ".gitkeep"
    if not keep.exists():
        keep.write_text("", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "hcli/.gitkeep"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "seed hcli dir"], check=True, capture_output=True)
    (hcli_dir / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")


def test_land_commit_never_pushes(tmp_path):
    repo = _repo(tmp_path)
    remote = _with_bare_remote(repo, tmp_path)
    remote_before = _head(remote, "main")
    (repo / "f.txt").write_text("a checkpoint worth landing\n", encoding="utf-8")
    _with_a_passing_test(repo)

    handler = CommandHandler(_StubController(repo))
    result = handler._land_commit("test checkpoint")
    assert handler.last_value.get("landed") is True, handler.last_value

    assert _head(repo) != remote_before, "sanity: the local commit must have actually landed"
    assert _head(remote, "main") == remote_before, (
        "a bare /land commit must never reach the remote -- only /land push may"
    )


def test_land_push_reaches_the_remote_only_when_typed(tmp_path):
    repo = _repo(tmp_path)
    remote = _with_bare_remote(repo, tmp_path)
    (repo / "f.txt").write_text("a checkpoint worth landing\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-am", "checkpoint"], check=True)
    local_head = _head(repo)
    assert _head(remote, "main") != local_head

    handler = CommandHandler(_StubController(repo))
    text = handler.handle("/land push")

    assert handler.last_value["pushed"] is True, handler.last_value
    assert text == "Pushed."
    assert _head(remote, "main") == local_head, "the explicit verb must actually reach the remote"


def test_land_push_without_a_remote_fails_and_stays_local(tmp_path):
    repo = _repo(tmp_path)  # no remote configured at all
    handler = CommandHandler(_StubController(repo))
    text = handler.handle("/land push")
    assert handler.last_value["pushed"] is False
    assert "Push failed" in text


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in sorted(globals().items()):
            if not name.startswith("test_"):
                continue
            case_dir = Path(tmp) / name
            case_dir.mkdir()
            fn(case_dir)
            print(f"ok  {name}")
    print("all green")

"""G117 tests: a gate that logs and continues is not a gate.

Every fixture is a REAL git repo in tmp_path, not a mock, because the checks are
about what git actually reports.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_safety as ss  # noqa: E402


def _run(*args, cwd):
    subprocess.run(args, cwd=str(cwd), capture_output=True, check=False)


def _repo(tmp_path, name, dirty=False, untracked=False):
    p = tmp_path / name
    p.mkdir()
    _run("git", "init", "-q", cwd=p)
    _run("git", "config", "user.email", "t@t", cwd=p)
    _run("git", "config", "user.name", "T", cwd=p)
    (p / "a.txt").write_text("one\n")
    _run("git", "add", "a.txt", cwd=p)
    _run("git", "commit", "-qm", "init", cwd=p)
    if dirty:
        (p / "a.txt").write_text("two\n")
    if untracked:
        (p / "b.txt").write_text("new\n")
    return p


def test_a_missing_target_is_nothing_to_delete(tmp_path):
    g = ss.gate(tmp_path / "does-not-exist")
    assert g["decision"] == "NOTHING_TO_DELETE"
    assert g["failures"] == []


def test_a_dirty_target_is_refused(tmp_path):
    g = ss.gate(_repo(tmp_path, "dirty", dirty=True))
    assert g["decision"] == "REFUSED"
    assert "dirty_paths" in g["blocking_failures"]


def test_untracked_files_alone_are_enough_to_refuse(tmp_path):
    """Untracked work is the easiest to lose and the hardest to notice."""
    g = ss.gate(_repo(tmp_path, "untracked", untracked=True))
    assert g["decision"] == "REFUSED"
    assert "dirty_paths" in g["blocking_failures"]


def test_an_unreachable_head_is_refused(tmp_path):
    """A clean external clone whose commits canonical has never seen."""
    g = ss.gate(_repo(tmp_path, "clean"))
    assert g["decision"] == "REFUSED"
    checks = {f["check"] for f in g["failures"]}
    assert "unreachable_commits" in checks
    assert "external_clone" in checks


def test_require_safe_to_delete_RAISES_it_does_not_warn(tmp_path):
    with pytest.raises(ss.DeletionRefused, match="forgotten work"):
        ss.require_safe_to_delete(_repo(tmp_path, "boom", dirty=True))


def test_a_non_git_directory_is_allowed(tmp_path):
    """The gate guards WORK, not every folder. A plain directory holds no
    commits and no git state to lose."""
    d = tmp_path / "plain"
    d.mkdir()
    (d / "x.txt").write_text("hi")
    g = ss.gate(d)
    assert g["decision"] == "ALLOWED"
    assert g["failures"] == []


def test_an_unregistered_target_gets_no_preservation_credit(tmp_path):
    p = ss.preservation_for(_repo(tmp_path, "stranger"))
    assert p["registered"] is False
    assert p["history_recoverable"] is False
    assert "nothing claims to have captured it" in p["why"]


def test_the_real_recovery_source_is_registered_and_refused():
    """hawking-copy: 17 GB, HEAD unreachable from main, 517 dirty and 4061
    untracked across 55 branches. Nothing has been deleted and nothing may be."""
    g = ss.gate("~/Downloads/hawking-copy")
    if not g["target"]["exists"]:
        pytest.skip("hawking-copy is not on this machine")
    assert g["decision"] == "REFUSED"
    assert g["blocking_failures"] == ["dirty_paths"]
    assert g["target"]["external_clone"] is True
    assert g["target"]["head_reachable_from_canonical"] is False


def test_history_preserved_is_not_work_preserved():
    """The distinction that keeps 17 GB undeletable. Collapsing the two would
    license exactly the deletion this gate exists to stop."""
    p = ss.preservation_for("~/Downloads/hawking-copy")
    if not p["registered"]:
        pytest.skip("not registered on this machine")
    assert p["history_recoverable"] is True
    assert p["working_files_preserved"] is False
    for f in ss.gate("~/Downloads/hawking-copy")["failures"]:
        if f["check"] == "unreachable_commits":
            assert f["recoverable_anyway"] is True
        if f["check"] == "dirty_paths":
            assert f["recoverable_anyway"] is False


def test_the_preserved_ref_actually_exists():
    p = ss.preservation_for("~/Downloads/hawking-copy")
    if not p["registered"]:
        pytest.skip("not registered")
    assert p["preserved_sha"], "the preserved ref must resolve, not just be named"


def test_the_build_reports_nothing_deleted_and_says_why():
    d = ss.build()
    assert d["nothing_deleted"] is True
    assert set(d["checks"]) == set(ss.CHECKS)
    assert "RAISES" in d["the_gate_is_a_refusal_not_a_warning"]
    assert "History preserved is not work preserved" in d["why_17_gb_is_still_on_disk"]


def test_every_registered_source_names_a_preserved_ref():
    for r in ss.REGISTERED:
        assert r["preserved_ref"].startswith("refs/preserved/")

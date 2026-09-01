"""Resident code tools: wrap existing machinery; refuse escape; never land red.

The load-bearing proofs are the ones a mock cannot stand in for: a real git
worktree under .worktrees/ (not a sibling), PATCH refusing a path outside it,
and one create → patch → test → diff → rollback cycle on a real tree.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.future import resident_code_tools as rct
from tools.future import sandbox as sb
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


def _seed(root: Path) -> Path:
    (root / "scratch.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "test_scratch.py").write_text(
        "from scratch import VALUE\n\n"
        "def test_value():\n"
        "    assert VALUE == 1\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "scratch.py", "test_scratch.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "seed scratch"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return root


@pytest.fixture
def seeded(tmp_path: Path) -> Path:
    root = sb.init_fixture_repo(tmp_path / "canonical")
    _seed(root)
    yield root
    subprocess.run(
        ["git", "-C", str(root), "worktree", "prune"],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def worktree(seeded: Path) -> Path:
    name = "rct-test"
    created = rct.invoke("CREATE_WORKTREE", name=name, repo=str(seeded))
    assert created["ok"], created.get("error")
    wt = Path(created["worktree"])
    try:
        yield wt
    finally:
        if wt.exists():
            try:
                rct.invoke("ROLLBACK", worktree=str(wt), remove_worktree=True)
            except rct.CodeToolsRefused:
                porcelain = subprocess.run(
                    ["git", "--no-optional-locks", "-C", str(wt), "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if porcelain.returncode == 0 and not porcelain.stdout.strip():
                    subprocess.run(
                        ["git", "-C", str(seeded), "worktree", "remove", str(wt)],
                        capture_output=True,
                        text=True,
                        check=False,
                    )


def test_operations_are_the_required_ten():
    assert rct.OPERATIONS == (
        "SEARCH_CODE",
        "READ_CODE",
        "CREATE_WORKTREE",
        "PATCH",
        "BUILD",
        "TEST",
        "RUN",
        "DIFF",
        "LAND_THROUGH_GATE",
        "ROLLBACK",
    )
    assert set(rct.WRAPS) == set(rct.OPERATIONS)
    assert set(rct._DISPATCH) == set(rct.OPERATIONS)


def test_unknown_operation_raises():
    with pytest.raises(rct.CodeToolsRefused, match="unknown operation"):
        rct.invoke("CHOOSE_THE_HYPOTHESIS")


def test_missing_inputs_raise_rather_than_defaulting():
    with pytest.raises(rct.CodeToolsRefused, match="pattern is required"):
        rct.invoke("SEARCH_CODE", root="tools/future")
    with pytest.raises(rct.CodeToolsRefused, match="root is required"):
        rct.invoke("SEARCH_CODE", pattern="VALUE")
    with pytest.raises(rct.CodeToolsRefused, match="path is required"):
        rct.invoke("READ_CODE")
    with pytest.raises(rct.CodeToolsRefused, match="name is required"):
        rct.invoke("CREATE_WORKTREE")
    with pytest.raises(rct.CodeToolsRefused, match="worktree is required"):
        rct.invoke("PATCH", path="x.py", content="x\n")
    with pytest.raises(rct.CodeToolsRefused, match="paths is required"):
        rct.invoke("LAND_THROUGH_GATE", message_file="/dev/null")
    with pytest.raises(rct.CodeToolsRefused, match="message_file is required"):
        rct.invoke("LAND_THROUGH_GATE", paths=["tools/future/resident_code_tools.py"])


def test_search_code_wraps_existing_fs_search(seeded: Path):
    r = rct.invoke("SEARCH_CODE", pattern="VALUE = 1", root=str(seeded), glob="*.py")
    assert r["ok"] is True
    assert r["op"] == "SEARCH_CODE"
    assert r["n_matches"] >= 1
    assert any("scratch.py" in str(m.get("path")) for m in r["matches"])
    assert "fs.search" in r["wrapped"]


def test_read_code_wraps_existing_fs_read(seeded: Path):
    r = rct.invoke("READ_CODE", path="scratch.py", worktree=str(seeded))
    assert r["ok"] is True
    assert "VALUE = 1" in r["content"]
    assert r["source"] == "disk"
    assert "fs.read" in r["wrapped"]


def test_read_missing_file_is_a_result_not_an_exception(seeded: Path):
    r = rct.invoke("READ_CODE", path="no_such_file.py", worktree=str(seeded))
    assert r["ok"] is False
    assert r["status"] == "FAILED"
    assert "not found" in r["error"]


def test_worktree_is_created_under_dot_worktrees_not_as_a_sibling(seeded: Path):
    name = "rct-place"
    r = rct.invoke("CREATE_WORKTREE", name=name, repo=str(seeded))
    try:
        assert r["ok"] is True, r.get("error")
        wt = Path(r["worktree"])
        assert wt.parent.name == ".worktrees"
        assert os.path.realpath(str(wt.parent.parent)) == os.path.realpath(str(seeded))
        sibling = Path(seeded).parent / name
        assert os.path.realpath(str(wt)) != os.path.realpath(str(sibling))
        assert not sibling.exists()
        assert r["under_dot_worktrees"] is True
        assert r["not_sibling"] is True
        listed = subprocess.run(
            ["git", "-C", str(seeded), "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert os.path.realpath(str(wt)) in listed or str(wt) in listed
    finally:
        if r.get("ok"):
            subprocess.run(
                ["git", "-C", str(seeded), "worktree", "remove", r["worktree"]],
                capture_output=True,
                text=True,
                check=False,
            )


def test_unsafe_worktree_name_is_refused(seeded: Path):
    with pytest.raises(rct.CodeToolsRefused, match="safe path component"):
        rct.invoke("CREATE_WORKTREE", name="../sibling", repo=str(seeded))
    with pytest.raises(rct.CodeToolsRefused, match="safe path component"):
        rct.invoke("CREATE_WORKTREE", name="foo/bar", repo=str(seeded))
    assert not (Path(seeded).parent / "sibling").exists()


def test_patch_refuses_a_path_outside_the_worktree(worktree: Path, tmp_path: Path):
    victim = tmp_path / "outside.txt"
    victim.write_text("safe\n", encoding="utf-8")
    with pytest.raises(rct.CodeToolsRefused, match="escapes worktree"):
        rct.invoke(
            "PATCH",
            worktree=str(worktree),
            path=str(victim),
            content="pwned\n",
        )
    assert victim.read_text(encoding="utf-8") == "safe\n"
    with pytest.raises(rct.CodeToolsRefused, match="escapes worktree"):
        rct.invoke(
            "PATCH",
            worktree=str(worktree),
            path="../outside.txt",
            content="pwned\n",
        )


def test_run_refuses_a_path_outside_the_worktree(worktree: Path, tmp_path: Path):
    victim = tmp_path / "secret.txt"
    victim.write_text("secret\n", encoding="utf-8")
    with pytest.raises(rct.CodeToolsRefused, match="escapes worktree"):
        rct.invoke("RUN", worktree=str(worktree), argv=["cat", str(victim)])


def test_failed_build_is_a_result_not_an_exception(worktree: Path):
    made = rct.invoke(
        "PATCH",
        worktree=str(worktree),
        operations=[{"op": "create", "path": "bad.py", "content": "def (\n"}],
    )
    assert made["ok"] is True
    built = rct.invoke("BUILD", worktree=str(worktree), paths=["bad.py"])
    assert built["ok"] is False
    assert built["status"] == "FAILED"
    assert built["op"] == "BUILD"
    assert "compile" in (built.get("wrapped") or "")


def test_failed_test_is_a_result_not_an_exception(worktree: Path):
    rct.invoke(
        "PATCH",
        worktree=str(worktree),
        path="test_scratch.py",
        old_text="    assert VALUE == 1\n",
        new_text="    assert VALUE == 999\n",
    )
    tested = rct.invoke("TEST", worktree=str(worktree), paths=["test_scratch.py"])
    assert tested["ok"] is False
    assert tested["status"] == "FAILED"
    assert tested["op"] == "TEST"


def test_land_through_gate_never_lands_red(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    called = {"land": 0}

    def fake_check(paths):
        return {
            "green": False,
            "verdict": "RED",
            "tests": {"why": "forced-red"},
            "receipts": {"malformed": []},
        }

    def fake_land(*_a, **_k):
        called["land"] += 1
        raise AssertionError("land must not be called when the gate is red")

    monkeypatch.setattr(rct.integration_gate, "check", fake_check)
    monkeypatch.setattr(rct.integration_gate, "land", fake_land)
    msg = tmp_path / "m.txt"
    msg.write_text("should never land\n")
    result = rct.invoke(
        "LAND_THROUGH_GATE",
        paths=["tools/future/resident_code_tools.py"],
        message_file=str(msg),
    )
    assert result["ok"] is False
    assert result["committed"] is False
    assert called["land"] == 0
    assert "RED" in (result.get("error") or "") or result.get("green") is False


def test_land_through_gate_refuses_known_red(tmp_path: Path):
    msg = tmp_path / "m.txt"
    msg.write_text("no\n")
    with pytest.raises(rct.CodeToolsRefused, match="never land red"):
        rct.invoke(
            "LAND_THROUGH_GATE",
            paths=["tools/future/resident_code_tools.py"],
            message_file=str(msg),
            known_red="please",
        )


def test_land_through_gate_calls_the_existing_gate_on_green(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls = []

    monkeypatch.setattr(
        rct.integration_gate,
        "check",
        lambda paths: {
            "green": True,
            "verdict": "GREEN",
            "tests": {"why": "ok"},
            "receipts": {"malformed": []},
        },
    )

    def fake_land(paths, message_file, known_red=None):
        calls.append(
            {"paths": list(paths), "message_file": message_file, "known_red": known_red}
        )
        return {"green": True, "committed": True, "known_red": known_red}

    monkeypatch.setattr(rct.integration_gate, "land", fake_land)
    msg = tmp_path / "m.txt"
    msg.write_text("land\n")
    result = rct.invoke(
        "LAND_THROUGH_GATE",
        paths=["tools/future/resident_code_tools.py"],
        message_file=str(msg),
    )
    assert result["ok"] is True
    assert result["committed"] is True
    assert calls and calls[0]["known_red"] is None
    src = Path(rct.__file__).read_text(encoding="utf-8")
    assert "integration_gate.check" in src
    assert "integration_gate.land" in src


def test_rollback_refuses_to_delete_uncommitted_work(worktree: Path):
    (worktree / "uncommitted.txt").write_text("do not delete me\n", encoding="utf-8")
    with pytest.raises(rct.CodeToolsRefused, match="uncommitted"):
        rct.invoke("ROLLBACK", worktree=str(worktree), remove_worktree=True)
    assert (worktree / "uncommitted.txt").is_file()
    assert (worktree / "uncommitted.txt").read_text(encoding="utf-8") == "do not delete me\n"
    (worktree / "uncommitted.txt").unlink()


def test_real_worktree_create_patch_test_diff_rollback(seeded: Path):
    """Not a mock: git worktree add, a real patch, pytest, git diff, rollback."""
    name = "rct-e2e"
    created = rct.invoke("CREATE_WORKTREE", name=name, repo=str(seeded))
    assert created["ok"] is True, created.get("error")
    wt = Path(created["worktree"])
    assert wt.parent.name == ".worktrees"
    assert (Path(seeded).parent / name).exists() is False
    try:
        patched = rct.invoke(
            "PATCH",
            worktree=str(wt),
            path="scratch.py",
            old_text="VALUE = 1\n",
            new_text="VALUE = 1  # resident-code-tools-e2e\n",
        )
        assert patched["ok"] is True, patched.get("error")
        assert "scratch.py" in (patched.get("paths") or patched.get("changed") or [])
        body = (wt / "scratch.py").read_text(encoding="utf-8")
        assert "resident-code-tools-e2e" in body

        tested = rct.invoke("TEST", worktree=str(wt), paths=["test_scratch.py"])
        assert tested["ok"] is True, tested.get("error") or tested.get("stdout")
        assert tested.get("returncode") == 0

        differed = rct.invoke("DIFF", worktree=str(wt), paths=["scratch.py"])
        assert differed["ok"] is True, differed.get("error")
        assert differed["nonempty"] is True
        assert "resident-code-tools-e2e" in (differed.get("stdout") or "")

        rolled = rct.invoke("ROLLBACK", worktree=str(wt), remove_worktree=True)
        assert rolled["ok"] is True, rolled.get("error")
        assert rolled["restored"] is True
        assert rolled["removed"] is True
        assert not wt.exists()
    except Exception:
        if wt.exists():
            subprocess.run(
                ["git", "-C", str(wt), "checkout", "--", "scratch.py", "test_scratch.py"],
                capture_output=True,
                text=True,
                check=False,
            )
            subprocess.run(
                ["git", "-C", str(seeded), "worktree", "remove", str(wt)],
                capture_output=True,
                text=True,
                check=False,
            )
        raise


def test_build_writes_a_sealed_receipt():
    out = rct.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "RESIDENT_CODE_TOOLS.json"
    assert doc["schema"] == rct.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["does_not_choose_science"] is True
    assert doc["not_a_new_agent_framework"] is True
    assert doc["land_never_red"] is True
    assert doc["never_deletes_uncommitted_work"] is True
    assert doc["never_sibling"] is True
    assert doc["placement"] == "<repo>/.worktrees/<name>"
    assert doc["operations"] == list(rct.OPERATIONS)
    assert doc["hermetic_proof"]["real_not_mock"] is True
    assert doc["hermetic_proof"]["create"]["under_dot_worktrees"] is True
    assert doc["hermetic_proof"]["patch_outside_refused"] is True
    assert doc["land_proof"]["land_called"] is False
    assert doc["land_proof"]["known_red_refused"] is True
    _assert_no_hardware_claims(doc)
    for field in HARDWARE_FIELDS:
        assert field not in doc or doc[field] in (None, "UNKNOWN")

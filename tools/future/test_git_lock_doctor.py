"""Git lock doctor: four-condition refuse, rename-aside, sealed receipt.

Temp-dir fixtures only for mutation tests. Live-repo assertions check that
the module COPES with either lock-present or lock-absent and records which
path it took — they do not encode this sparse checkout.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.future import git_lock_doctor as gld
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims

NOW = 1_788_050_000.0
THRESHOLD = 300.0


def _git_layout(tmp_path: Path) -> Path:
    """A fake git dir with HEAD. No `git init` — the doctor parses filesystem."""
    git_dir = tmp_path / "repo" / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / "repo" / "README").write_text("x\n")
    return git_dir


def _write_lock(git_dir: Path, *, size: int, mtime: float) -> Path:
    lock = git_dir / "index.lock"
    lock.write_bytes(b"\0" * size if size else b"")
    if size == 0:
        lock.write_bytes(b"")
    os.utime(lock, (mtime, mtime))
    return lock


def _holders_none(path: Path):
    return [], "test-lsof-empty"


def _holders_one(path: Path):
    return ["git 111 cwd"], "test-lsof-held"


def _holders_unknown(path: Path):
    return None, "lsof not found; refuse"


def _git_none():
    return [], "test-pgrep-empty"


def _git_unknown():
    return None, "pgrep failed; refuse"


def _git_against(repo: Path):
    def _fn():
        return (
            [{"pid": 222, "command": "git commit -q", "cwd": str(repo)}],
            "test-pgrep-hit",
        )

    return _fn


def _obs(**over) -> gld.LockObservation:
    base = dict(
        path="/tmp/fake/.git/index.lock",
        present=True,
        is_regular_file=True,
        size_bytes=0,
        mtime_epoch=NOW - 1000,
        age_seconds=1000.0,
        holders=[],
        holders_note="test",
        git_processes=[],
        git_processes_note="test",
        matching_git_processes=[],
        age_threshold_s=THRESHOLD,
        layout={},
    )
    base.update(over)
    return gld.LockObservation(**base)


def test_entry_point_emits_sealed_receipt():
    """The module's CLI runs and write_receipt seals GIT_LOCK_DURABILITY_REPORT."""
    proc = subprocess.run(
        [sys.executable, str(Path(gld.__file__)), "--report"],
        cwd=str(gld.REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = RECEIPTS / gld.RECEIPT
    assert out.is_file()
    doc = json.loads(out.read_text())
    assert doc["schema"] == gld.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["claim_boundary"]
    _assert_no_hardware_claims(doc)
    assert doc["policy"]["default_mode"] == "report"
    assert doc["policy"]["never_deletes_stale"] is True
    assert list(c["name"] for t in doc["scan"]["targets"] for c in t["conditions"])
    for t in doc["scan"]["targets"]:
        assert set(c["name"] for c in t["conditions"]) == set(gld.CONDITIONS)
        assert t["action"]["renamed"] is False
        assert t["action"]["mode"] == "report"
    assert "git_callers" in doc
    assert "fsck" in doc
    assert doc["fsck"]["exit_code"] == 0
    assert "hypotheses" in doc
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert doc["evidence_source"] in ("pinned_snapshot", "live_headless")
    assert doc["lock_forensics_source"] == "live_git_dir"
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    import hashlib

    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]


def test_report_mode_never_renames_even_when_all_four_hold(tmp_path: Path):
    git_dir = _git_layout(tmp_path)
    lock = _write_lock(git_dir, size=0, mtime=NOW - 1000)
    stale = git_dir / "index.lock.stale-already"
    stale.write_bytes(b"")
    scan = gld.diagnose_repo(
        git_dir.parent,
        now=NOW,
        age_threshold_s=THRESHOLD,
        apply=False,
        tag="doctor",
        holders_fn=_holders_none,
        git_procs_fn=_git_none,
    )
    assert scan["targets"]
    t = scan["targets"][0]
    assert t["may_rename"] is True
    assert t["action"]["renamed"] is False
    assert t["action"]["attempted"] is False
    assert lock.is_file()
    assert stale.is_file()
    assert lock.stat().st_size == 0


def test_rename_aside_when_all_four_hold(tmp_path: Path):
    git_dir = _git_layout(tmp_path)
    lock = _write_lock(git_dir, size=0, mtime=NOW - 1000)
    preexisting = git_dir / "index.lock.stale-20260827-campaign"
    preexisting.write_bytes(b"")
    scan = gld.diagnose_repo(
        git_dir.parent,
        now=NOW,
        age_threshold_s=THRESHOLD,
        apply=True,
        tag="doctor",
        holders_fn=_holders_none,
        git_procs_fn=_git_none,
    )
    t = scan["targets"][0]
    assert t["may_rename"] is True
    assert t["action"]["renamed"] is True
    dest = Path(t["action"]["destination"])
    assert dest.name.startswith("index.lock.stale-")
    assert dest.is_file()
    assert dest.stat().st_size == 0
    assert not lock.exists()
    assert preexisting.is_file(), "existing forensic stale files must not be deleted"
    names = {p.name for p in git_dir.iterdir()}
    assert "index.lock.stale-20260827-campaign" in names


def test_rename_refused_when_any_guard_fails_does_not_move_lock(tmp_path: Path):
    git_dir = _git_layout(tmp_path)
    lock = _write_lock(git_dir, size=16, mtime=NOW - 1000)
    scan = gld.diagnose_repo(
        git_dir.parent,
        now=NOW,
        age_threshold_s=THRESHOLD,
        apply=True,
        tag="doctor",
        holders_fn=_holders_none,
        git_procs_fn=_git_none,
    )
    t = scan["targets"][0]
    assert t["may_rename"] is False
    assert t["action"].get("refused") is True
    assert gld.CONDITION_ZERO in t["failing_conditions"]
    assert lock.is_file()
    assert lock.stat().st_size == 16


@pytest.mark.parametrize(
    "over,failing",
    [
        ({"age_seconds": 1.0, "mtime_epoch": NOW - 1}, gld.CONDITION_AGE),
        ({"size_bytes": 524288}, gld.CONDITION_ZERO),
        ({"holders": ["git 9 /repo"]}, gld.CONDITION_LSOF),
        (
            {
                "matching_git_processes": [
                    {"pid": 7, "command": "git add -- receipts/odyssey-i", "cwd": "/repo"}
                ],
                "git_processes": [
                    {"pid": 7, "command": "git add -- receipts/odyssey-i", "cwd": "/repo"}
                ],
            },
            gld.CONDITION_GIT,
        ),
    ],
)
def test_negative_control_each_guard_refuses(over, failing):
    """Four cases, one miss each. A guard nobody has watched fail is not a guard."""
    obs = _obs(**over)
    d = gld.evaluate_observation(obs)
    assert d["may_rename"] is False
    assert failing in d["failing_conditions"], d
    named = [c for c in d["conditions"] if c["name"] == failing]
    assert named and named[0]["holds"] is False
    # The other three still hold — the refusal is specifically this guard.
    others = [c for c in d["conditions"] if c["name"] != failing]
    assert others and all(c["holds"] for c in others), d["conditions"]


def test_negative_control_four_tempdir_cases(tmp_path: Path):
    """Construct four live files in a temp git dir; each fails exactly one guard."""
    cases = []

    young = _git_layout(tmp_path / "young")
    _write_lock(young, size=0, mtime=NOW - 10)
    cases.append((young, _holders_none, _git_none, gld.CONDITION_AGE))

    nonzero = _git_layout(tmp_path / "nonzero")
    _write_lock(nonzero, size=32, mtime=NOW - 1000)
    cases.append((nonzero, _holders_none, _git_none, gld.CONDITION_ZERO))

    held = _git_layout(tmp_path / "held")
    _write_lock(held, size=0, mtime=NOW - 1000)
    cases.append((held, _holders_one, _git_none, gld.CONDITION_LSOF))

    busy = _git_layout(tmp_path / "busy")
    _write_lock(busy, size=0, mtime=NOW - 1000)
    cases.append((busy, _holders_none, _git_against(busy.parent), gld.CONDITION_GIT))

    for git_dir, holders_fn, git_fn, failing in cases:
        lock = git_dir / "index.lock"
        scan = gld.diagnose_repo(
            git_dir.parent,
            now=NOW,
            age_threshold_s=THRESHOLD,
            apply=True,
            tag="doctor",
            holders_fn=holders_fn,
            git_procs_fn=git_fn,
        )
        t = scan["targets"][0]
        assert t["may_rename"] is False, failing
        assert failing in t["failing_conditions"], (failing, t)
        assert t["action"].get("renamed") is False
        assert lock.is_file(), f"refused lock must stay in place ({failing})"


def test_unknown_lsof_fail_closed(tmp_path: Path):
    git_dir = _git_layout(tmp_path)
    lock = _write_lock(git_dir, size=0, mtime=NOW - 1000)
    d = gld.evaluate_observation(
        _obs(holders=None, holders_note="lsof not found; refuse")
    )
    assert d["may_rename"] is False
    assert gld.CONDITION_LSOF in d["failing_conditions"]
    scan = gld.diagnose_repo(
        git_dir.parent,
        now=NOW,
        age_threshold_s=THRESHOLD,
        apply=True,
        tag="doctor",
        holders_fn=_holders_unknown,
        git_procs_fn=_git_none,
    )
    assert scan["targets"][0]["may_rename"] is False
    assert lock.is_file()


def test_unknown_git_process_list_fail_closed():
    d = gld.evaluate_observation(
        _obs(git_processes=None, git_processes_note="pgrep failed; refuse")
    )
    assert d["may_rename"] is False
    assert gld.CONDITION_GIT in d["failing_conditions"]


def test_absent_lock_is_recorded_not_an_error(tmp_path: Path):
    git_dir = _git_layout(tmp_path)
    scan = gld.diagnose_repo(
        git_dir.parent,
        now=NOW,
        age_threshold_s=THRESHOLD,
        apply=True,
        tag="doctor",
        holders_fn=_holders_none,
        git_procs_fn=_git_none,
    )
    t = scan["targets"][0]
    assert t["present"] is False
    assert t["may_rename"] is False
    assert t["action"].get("renamed") is False
    assert not (git_dir / "index.lock").exists()
    # The module coped; it did not invent a lock to remove.


def test_stale_destination_never_clobbers(tmp_path: Path):
    git_dir = _git_layout(tmp_path)
    lock = git_dir / "index.lock"
    lock.write_bytes(b"")
    taken = git_dir / f"index.lock.stale-{int(NOW)}-doctor"
    taken.write_bytes(b"keep")
    dest = gld.stale_destination(lock, now=int(NOW), tag="doctor")
    assert dest != taken
    assert dest.name.endswith("-2")
    result = gld.rename_aside(lock, now=int(NOW), tag="doctor")
    assert result["renamed"] is True
    assert Path(result["destination"]).name.endswith("-2")
    assert taken.read_bytes() == b"keep"


def test_catalog_and_distribution_derived_from_disk(tmp_path: Path):
    git_dir = _git_layout(tmp_path)
    a = git_dir / "index.lock.stale-20260827-0306"
    b = git_dir / "index.lock.stale-20260827-0310"
    a.write_bytes(b"")
    b.write_bytes(b"x" * 524288)
    os.utime(a, (NOW - 4000, NOW - 4000))
    os.utime(b, (NOW - 2000, NOW - 2000))
    rows = gld.catalog_stale(git_dir)
    assert [r["name"] for r in rows] == sorted(r["name"] for r in rows)
    assert len(rows) == 2
    dist = gld.stale_arrival_distribution(rows)
    assert dist["n"] == 2
    assert dist["zero_bytes"] == 1
    assert dist["nonzero_bytes"] == 1
    assert "524288" in dist["size_histogram"]
    assert dist["gap_seconds"]["n"] == 1


def test_process_targets_repo_does_not_match_unrelated_cwd():
    layout = {
        "repo_root": "/Users/scammermike/Downloads/hawking",
        "git_dir": "/Users/scammermike/Downloads/hawking/.git",
        "common_dir": "/Users/scammermike/Downloads/hawking/.git",
        "repo": "/Users/scammermike/Downloads/hawking",
        "worktree_paths": [],
    }
    other = {"pid": 1, "command": "git status", "cwd": "/tmp/unrelated"}
    mine = {
        "pid": 2,
        "command": "git add -- receipts/odyssey-i",
        "cwd": "/Users/scammermike/Downloads/hawking",
    }
    substr = {
        "pid": 3,
        "command": "hf download org/model .gitattributes",
        "cwd": "/Volumes/corpdrive",
    }
    assert gld.process_targets_repo(other, layout) is False
    assert gld.process_targets_repo(mine, layout) is True
    assert gld.process_targets_repo(substr, layout) is False


def test_hf_gitattributes_is_not_a_git_binary():
    """pgrep -x git must not treat 'hf download … .gitattributes' as git."""
    rows, note = [], "pgrep -x git: none"
    obs = _obs(git_processes=rows, git_processes_note=note, matching_git_processes=[])
    d = gld.evaluate_observation(obs)
    assert d["may_rename"] is True
    assert note  # path recorded


def test_build_receipt_fields_and_no_hardware(tmp_path: Path):
    git_dir = _git_layout(tmp_path)
    _write_lock(git_dir, size=0, mtime=NOW - 10)
    out = gld.build(
        repo=git_dir.parent,
        now=NOW,
        age_threshold_s=THRESHOLD,
        apply=False,
        holders_fn=_holders_none,
        git_procs_fn=_git_none,
        recorded_by="tools/future/test_git_lock_doctor.py",
    )
    assert out.parent == RECEIPTS
    assert out.name == gld.RECEIPT
    doc = json.loads(out.read_text())
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        # Presence of the key as a nested hardware number is already refused
        # by _assert_no_hardware_claims. The doctor must not even advertise them.
        assert key not in doc
    assert doc["hypotheses"]["most_probable"] == "H1_plus_H2"
    assert doc["recovered_implementation"]["git_lock_doctor_existed"] is False
    assert doc["fsck"]["intended_commit_missing"] is False
    assert doc["scan"]["apply"] is False
    # Copes with a present 0-byte lock in the temp repo: records present True.
    assert any(t["present"] is True for t in doc["scan"]["targets"])


def test_cli_rename_aside_isolated(tmp_path: Path, monkeypatch):
    git_dir = _git_layout(tmp_path)
    lock = _write_lock(git_dir, size=0, mtime=NOW - 1000)

    def _holders(path):
        return [], "cli-lsof"

    def _git():
        return [], "cli-pgrep"

    monkeypatch.setattr(gld, "default_holders_fn", _holders)
    monkeypatch.setattr(gld, "default_git_procs_fn", _git)
    rc = gld.main(
        [
            "--rename-aside",
            "--repo",
            str(git_dir.parent),
            "--age-seconds",
            "300",
            "--tag",
            "testdoc",
        ]
    )
    assert rc == 0
    assert not lock.exists()
    stale = list(git_dir.glob("index.lock.stale-*-testdoc"))
    assert len(stale) == 1


def test_main_refuses_both_flags(capsys):
    rc = gld.main(["--report", "--rename-aside"])
    assert rc == 2


def test_worktree_gitdir_file_resolves_common_dir(tmp_path: Path):
    primary = tmp_path / "hawking"
    common = primary / ".git"
    common.mkdir(parents=True)
    (common / "HEAD").write_text("ref: refs/heads/odyssey-i\n")
    (common / "objects").mkdir()
    wt = tmp_path / "lane"
    wt.mkdir()
    wt_gitdir = common / "worktrees" / "lane"
    wt_gitdir.mkdir(parents=True)
    (wt_gitdir / "commondir").write_text("../..\n")
    (wt_gitdir / "gitdir").write_text(str(wt / ".git") + "\n")
    (wt / ".git").write_text(f"gitdir: {wt_gitdir}\n")
    layout = gld.resolve_git_layout(wt)
    assert layout["is_worktree"] is True
    assert Path(layout["common_dir"]) == common.resolve()
    assert Path(layout["git_dir"]) == wt_gitdir.resolve()
    _write_lock(common, size=0, mtime=NOW - 1000)
    scan = gld.diagnose_repo(
        wt,
        now=NOW,
        age_threshold_s=THRESHOLD,
        apply=False,
        tag="doctor",
        holders_fn=_holders_none,
        git_procs_fn=_git_none,
    )
    roles = {t["role"] for t in scan["targets"]}
    assert "primary" in roles
    primary_t = next(t for t in scan["targets"] if t["role"] == "primary")
    assert primary_t["present"] is True
    assert primary_t["may_rename"] is True

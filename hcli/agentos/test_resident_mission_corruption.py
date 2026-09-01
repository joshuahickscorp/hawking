"""A corrupt (but present) mission/state.json must not read as "no work".

chaos-recovery probe (2026-09-01): ``_mission_has_work`` caught
``json.JSONDecodeError`` alongside ``FileNotFoundError`` and returned False
for both.  The supervisor's spawn gate is
``mission_pending or inbox_pending or not mission_path.is_file()`` -- so a
mission file that EXISTS but is garbled (a truncated/garbled write mid
mutation) made every term False and the supervisor went IDLE forever with no
error, no restart, and no operator-visible signal, even though
``agent.recover_mission()`` (via ``hcli.mission.load_state``) already raises
a clean ``MissionCorruptError`` for exactly this file. That correct error was
never reached because the shallow pre-check silently absorbed it first.

The fix makes the pre-check ask the SAME question ``recover_mission()`` asks
(``hcli.mission.load_state``), so a corrupt file is classified as "has work"
(spawn a worker, let it fail loudly) rather than "no work" (freeze silently).
"""
from __future__ import annotations

import json
from pathlib import Path

from hcli.agentos.resident import _mission_has_work


def _mission_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".hcli" / "mission"
    d.mkdir(parents=True)
    return d


def test_no_mission_file_is_genuinely_no_work(tmp_path):
    assert _mission_has_work(tmp_path) is False


def test_valid_mission_with_no_pending_units_is_no_work(tmp_path):
    d = _mission_dir(tmp_path)
    (d / "state.json").write_text(
        json.dumps({"units": {"done1": {"status": "complete"}}}),
        encoding="utf-8",
    )
    assert _mission_has_work(tmp_path) is False


def test_valid_mission_with_a_pending_unit_is_work(tmp_path):
    d = _mission_dir(tmp_path)
    (d / "state.json").write_text(
        json.dumps({"units": {"u1": {"status": "pending"}}}),
        encoding="utf-8",
    )
    assert _mission_has_work(tmp_path) is True


def test_corrupt_json_is_treated_as_work_not_absence(tmp_path):
    # A present-but-garbled file (e.g. a torn write) must trigger a worker
    # spawn so recover_mission()'s MissionCorruptError actually surfaces,
    # instead of freezing the resident in IDLE with no signal.
    d = _mission_dir(tmp_path)
    (d / "state.json").write_text("{not valid json truncated", encoding="utf-8")
    assert _mission_has_work(tmp_path) is True


def test_non_object_json_root_is_also_treated_as_work(tmp_path):
    # load_state() raises MissionCorruptError for this too (root not a dict);
    # the pre-check must agree, not diverge into a second classifier.
    d = _mission_dir(tmp_path)
    (d / "state.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert _mission_has_work(tmp_path) is True


if __name__ == "__main__":
    import tempfile

    for fn in [
        test_no_mission_file_is_genuinely_no_work,
        test_valid_mission_with_no_pending_units_is_no_work,
        test_valid_mission_with_a_pending_unit_is_work,
        test_corrupt_json_is_treated_as_work_not_absence,
        test_non_object_json_root_is_also_treated_as_work,
    ]:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")

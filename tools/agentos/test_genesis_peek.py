"""Focused, process-free checks for genesis_peek's runtime truth reporting."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PEEK = REPO / "tools" / "genesis_peek.py"


def _load():
    spec = importlib.util.spec_from_file_location("genesis_peek", PEEK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


peek = _load()


def test_process_snapshot_classifies_daemon_and_resident_without_substring_phantoms() -> None:
    rows = [
        {
            "pid": 11,
            "ppid": 1,
            "state": "S",
            "alive": True,
            "command": "/usr/bin/python3 /repo/tools/ascent_daemon.py loop",
        },
        {
            "pid": 12,
            "ppid": 1,
            "state": "S",
            "alive": True,
            "command": "/repo/workspace/ops/build/rust/release/genesis-resident --socket /tmp/x",
        },
        {
            "pid": 13,
            "ppid": 1,
            "state": "S",
            "alive": True,
            "command": "rg ascent_daemon.py loop genesis-resident",
        },
        {
            "pid": 14,
            "ppid": 1,
            "state": "Z",
            "alive": False,
            "command": "/usr/bin/python3 /repo/tools/ascent_daemon.py loop",
        },
    ]
    assert peek.daemon_pids(rows) == [11]
    assert peek.resident_pids(rows) == [12]


def test_gpu_lock_reports_owner_and_distinguishes_held_from_stale(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    assert peek.gpu_lock_state(lock, alive=lambda _pid: True)["state"] == "FREE"
    lock.mkdir()
    (lock / "owner").write_text("honest-roof\n")
    (lock / "pid").write_text("4242\n")
    assert peek.gpu_lock_state(lock, alive=lambda pid: pid == 4242) == {
        "state": "HELD",
        "owner": "honest-roof",
        "pid": 4242,
    }
    assert peek.gpu_lock_state(lock, alive=lambda _pid: False)["state"] == "STALE"


def test_gpu_lock_with_missing_pid_is_incomplete_not_free(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    lock.mkdir()
    (lock / "owner").write_text("half-written\n")
    assert peek.gpu_lock_state(lock, alive=lambda _pid: True) == {
        "state": "INCOMPLETE",
        "owner": "half-written",
        "pid": None,
    }


def test_lineage_flags_are_compared_to_each_other_and_resident_reality() -> None:
    mismatch, issues = peek.lineage_reality({"live": True, "launched": False}, False)
    assert mismatch == "MISMATCH"
    assert "CURRENT live/launch flags disagree" in issues
    assert "CURRENT.live does not match resident body" in issues

    mismatch, issues = peek.lineage_reality({"live": True, "launched": False}, True)
    assert mismatch == "MISMATCH"
    assert "CURRENT.launched does not match resident body" in issues

    assert peek.lineage_reality({"live": True, "launched": True}, True) == ("MATCH", [])


def test_agentos_dispatch_record_is_cross_checked_with_process_liveness(
    tmp_path: Path, monkeypatch
) -> None:
    registry = tmp_path / "workers.json"
    registry.write_text(json.dumps({"workers": []}))
    controller = tmp_path / "agentos-controller.json"
    controller.write_text(
        json.dumps({"status": "running", "pid": 4242, "worker_id": "gravity"})
    )
    monkeypatch.setattr(peek, "WORKER_REGISTRY", registry)
    monkeypatch.setattr(peek, "CANDIDATE_ROOT", tmp_path / "candidates")
    monkeypatch.setattr(peek, "LIFECYCLE_CONTROLLER", tmp_path / "lifecycle.json")
    monkeypatch.setattr(peek, "AGENTOS_CONTROLLER", controller)
    monkeypatch.setattr(peek, "pid_alive", lambda _pid: False)

    status = peek.agentos_status()
    assert status["dispatch"] == {
        "status": "running",
        "pid": 4242,
        "worker_id": "gravity",
        "runtime_state": "EXITED",
    }

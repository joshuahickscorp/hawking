"""Negative controls for tools/future/detached.py.

A guard nobody has watched fail is not a guard. These tests spawn real
processes (/bin/sleep, python3 -c), never mocks. They do not encode this
sparse checkout: identity and presence are recorded as the path taken.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hcli.resources import pid_is_alive
from hcli.workunit import DEFAULT_RETRY_BUDGET, WorkUnit
from tools.future import detached as d
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


def _sleep(name: str, seconds: int, **extra):
    return {
        "id": name,
        "role": "science",
        "description": f"sleep {seconds}",
        "command": ["/bin/sleep", str(seconds)],
        "resource_class": "LIGHT_CONTROL",
        "verifier": "future.detached.child_terminal_classified",
        "classification": "STATIC_ONLY",
        **extra,
    }


@pytest.fixture
def sup(tmp_path):
    supervisor = d.DetachedSupervisor(tmp_path)
    yield supervisor
    supervisor.reap_all()


def test_entry_point_runs_and_seals_receipt():
    out = d.selftest()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "DETACHED_EXECUTION.json"
    assert doc["schema"] == d.SCHEMA
    assert doc["schema"] == "hawking.future.detached.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert "resident_callable" in doc
    callable_doc = doc["resident_callable"]
    assert callable_doc["callable"] is True
    assert callable_doc["workunit"]["id"]
    assert callable_doc["receipt"].endswith("DETACHED_EXECUTION.json")
    assert callable_doc["fail_closed"]
    assert doc["proofs_all_passed"] is True
    assert set(doc["terminal_classes"]) == set(d.TERMINAL_CLASSES)
    assert "unknown" in doc["terminal_classes"]
    assert "VI" not in "".join(doc["eras"])
    assert all("Odyssey IV" not in item and not item.startswith("IV ") for item in doc["odysseys"])
    assert len(doc["eras"]) == 5
    assert len(doc["odysseys"]) == 3
    _assert_no_hardware_claims(doc)
    WorkUnit.from_dict(dict(callable_doc["workunit"]))


def test_launch_returns_while_child_still_running(sup):
    t0 = time.monotonic()
    rec = sup.launch(_sleep("not-blocked", 8))
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"launch blocked for {elapsed:.3f}s"
    assert rec.get("terminal") is None
    supervisor_pid = rec.get("supervisor_pid")
    work_pid = rec.get("pid")
    supervisor_alive = isinstance(supervisor_pid, int) and pid_is_alive(supervisor_pid)
    work_alive = isinstance(work_pid, int) and pid_is_alive(work_pid)
    assert supervisor_alive or work_alive
    if isinstance(work_pid, int) and rec.get("start_token"):
        assert d.identity_status(rec) == "match"
    cancelled = sup.cancel(rec["job_id"])
    assert cancelled["terminal"] == "cancelled"
    if isinstance(work_pid, int):
        assert not pid_is_alive(work_pid)


def test_killed_mid_flight_classified_crashed_not_hanging(sup):
    rec = sup.launch(_sleep("killed-mid", 20))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not rec.get("pid"):
        rec = sup.inspect(rec["job_id"])
        time.sleep(0.02)
    pid = rec.get("pid")
    assert isinstance(pid, int) and pid > 0
    t0 = time.monotonic()
    os.kill(pid, signal.SIGKILL)
    terminal = sup.wait_terminal(rec["job_id"], timeout_s=4.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 6.0, f"inspect hung for {elapsed:.3f}s after kill"
    assert terminal["terminal"] == "crashed"
    assert terminal["terminal"] != "unknown"
    assert not pid_is_alive(pid)


def test_stale_pid_reused_is_not_the_original_child(sup, tmp_path):
    other = subprocess.Popen(
        ["/bin/sleep", "20"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    rec = None
    try:
        rec = sup.launch(_sleep("original-child", 20))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not rec.get("start_token"):
            rec = sup.inspect(rec["job_id"])
            time.sleep(0.02)
        forged = dict(rec)
        forged["pid"] = other.pid
        status = d.identity_status(forged)
        # Cope with either: tokens present (reused) or missing (unknown).
        # Never treat a mismatched/unproven identity as the original child.
        assert status != "match"
        assert status in {"reused", "unknown"}
        copy_id = "forged-reuse"
        planted = dict(forged)
        planted["job_id"] = copy_id
        planted["terminal"] = None
        planted["state"] = "RUNNING"
        planted["cancel_requested"] = False
        planted["launch_refused"] = False
        from hcli.persist import atomic_write_json

        atomic_write_json(sup._path(copy_id), planted)
        cancelled = sup.cancel(copy_id)
        assert pid_is_alive(other.pid), "cancel must not kill the unrelated live pid"
        assert cancelled.get("terminal") in {"unknown", None}
        adopted = sup.adopt(copy_id)
        assert adopted.get("terminal") != "completed-with-receipt"
        assert adopted.get("terminal") != "completed-without-receipt"
        if adopted.get("state") == "RUNNING":
            assert adopted.get("identity_status") != "match"
    finally:
        if rec:
            try:
                sup.cancel(rec["job_id"])
            except Exception:
                d._terminate_session(rec.get("pid"))
        d._terminate_session(other.pid)
        try:
            other.wait(timeout=2)
        except subprocess.TimeoutExpired:
            other.kill()


def test_cancel_reaps_child(sup):
    rec = sup.launch(_sleep("cancel-me", 20))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not rec.get("pid"):
        rec = sup.inspect(rec["job_id"])
        time.sleep(0.02)
    pid = rec.get("pid")
    cancelled = sup.cancel(rec["job_id"])
    assert cancelled["terminal"] == "cancelled"
    if isinstance(pid, int):
        assert not pid_is_alive(pid)


def test_timeout_reaps_child(sup):
    rec = sup.launch(_sleep("timeout-me", 20, timeout_s=0.25))
    terminal = sup.wait_terminal(rec["job_id"], timeout_s=5.0)
    assert terminal["terminal"] == "timed_out"
    pid = terminal.get("pid") or rec.get("pid")
    if isinstance(pid, int):
        assert not pid_is_alive(pid)


def test_completed_with_and_without_receipt(sup, tmp_path):
    expected = tmp_path / "results" / "child.json"
    with_rec = sup.launch(
        {
            "id": "with-receipt",
            "command": [
                sys.executable,
                "-c",
                "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('{\"ok\": true}\\n')",
                str(expected),
            ],
            "output_receipt_path": str(expected),
            "resource_class": "LIGHT_CONTROL",
            "role": "science",
            "description": "write receipt",
            "verifier": "future.detached.child_terminal_classified",
        }
    )
    without_rec = sup.launch(
        {
            "id": "without-receipt",
            "command": [sys.executable, "-c", "raise SystemExit(0)"],
            "resource_class": "LIGHT_CONTROL",
            "role": "science",
            "description": "exit 0 no receipt",
            "verifier": "future.detached.child_terminal_classified",
        }
    )
    with_term = sup.wait_terminal(with_rec["job_id"], timeout_s=5.0)
    without_term = sup.wait_terminal(without_rec["job_id"], timeout_s=5.0)
    assert with_term["terminal"] == "completed-with-receipt"
    assert without_term["terminal"] == "completed-without-receipt"
    assert expected.is_file()


def test_dead_pid_without_exit_is_unknown_not_guessed(sup, tmp_path):
    rec = {
        "schema": d.SUPERVISION_SCHEMA,
        "job_id": "ghost",
        "argv": ["/bin/sleep", "1"],
        "cwd": str(tmp_path),
        "state": "RUNNING",
        "terminal": None,
        "pid": 2**30,
        "start_token": "0.000000",
        "timeout_s": None,
        "cancel_requested": False,
        "launch_refused": False,
        "expected_receipt_path": str(tmp_path / "results" / "missing.json"),
        "returncode": None,
        "crash_reason": None,
        "started_at": time.time(),
        "log_paths": {
            "stdout": str(tmp_path / "detached" / "logs" / "g.stdout.log"),
            "stderr": str(tmp_path / "detached" / "logs" / "g.stderr.log"),
        },
        "resource_lease": d._empty_resource_lease(),
        "retry_policy": d._retry_policy_of({}),
    }
    from hcli.persist import atomic_write_json

    atomic_write_json(sup._path("ghost"), rec)
    adopted = sup.adopt("ghost")
    assert adopted["terminal"] == "unknown"
    assert adopted["terminal"] not in {
        "crashed",
        "completed-with-receipt",
        "completed-without-receipt",
    }


def test_restarted_supervisor_readopts_live_child(sup, tmp_path):
    rec = sup.launch(_sleep("readopt", 12))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not rec.get("start_token"):
        rec = sup.inspect(rec["job_id"])
        time.sleep(0.02)
    if not rec.get("pid") or not rec.get("start_token"):
        pytest.skip("start_token unavailable on this host; identity fails closed to unknown")
    other = d.DetachedSupervisor(tmp_path)
    adopted = other.adopt(rec["job_id"])
    assert adopted["identity_status"] == "match"
    assert adopted.get("terminal") is None
    assert pid_is_alive(rec["pid"])
    cancelled = other.cancel(rec["job_id"])
    assert cancelled["terminal"] == "cancelled"
    assert not pid_is_alive(rec["pid"])


def test_unsafe_commands_are_refused_and_do_not_spawn(sup):
    trials = [
        (["cargo", "build"], "LIGHT_CONTROL"),
        (["cargo", "test", "--release"], "LIGHT_CONTROL"),
        (["/bin/sleep", "1"], "GPU_EXCLUSIVE"),
        (["flock", ".hcli/locks/protected-accelerator-bench.lock", "echo", "x"], "LIGHT_CONTROL"),
        (["xcrun", "metal", "foo.metal"], "LIGHT_CONTROL"),
        (
            [sys.executable, "-m", "hcli", "agentos", "protected-accelerator-bench"],
            "LIGHT_CONTROL",
        ),
    ]
    for argv, rc in trials:
        with pytest.raises(d.UnsafeCommandError) as ei:
            sup.launch(
                {
                    "id": "unsafe",
                    "command": argv,
                    "resource_class": rc,
                    "role": "science",
                    "description": "must refuse",
                    "verifier": "future.detached.child_terminal_classified",
                }
            )
        assert ei.value.record is not None
        assert ei.value.record["pid"] is None
        assert ei.value.record["state"] == "SLEEPING"
        assert ei.value.record["launch_refused"] is True


def test_harmless_python_dash_c_is_allowed(sup):
    rec = sup.launch(
        {
            "id": "py-ok",
            "command": [sys.executable, "-c", "raise SystemExit(0)"],
            "resource_class": "LIGHT_CONTROL",
            "role": "science",
            "description": "harmless python",
            "verifier": "future.detached.child_terminal_classified",
        }
    )
    term = sup.wait_terminal(rec["job_id"], timeout_s=5.0)
    assert term["terminal"] == "completed-without-receipt"


def test_supervision_record_has_required_fields(sup):
    rec = sup.launch(_sleep("fields", 8, timeout_s=5.0))
    try:
        for key in (
            "pid",
            "argv",
            "cwd",
            "started_at",
            "log_paths",
            "timeout_s",
            "retry_policy",
            "expected_receipt_path",
            "cancel_requested",
            "resource_lease",
            "crash_reason",
            "job_id",
            "supervisor_pid",
        ):
            assert key in rec, key
        assert rec["argv"][0] == "/bin/sleep"
        assert rec["timeout_s"] == 5.0
        assert rec["retry_policy"]["policy"]
        assert rec["retry_policy"]["max_attempts"] == DEFAULT_RETRY_BUDGET
        assert rec["resource_lease"]["held"] is False
        assert rec["resource_lease"]["refused"] is True
        assert Path(rec["log_paths"]["stdout"]).parent.exists()
        assert rec["supervisor_pid"]
    finally:
        sup.cancel(rec["job_id"])


def test_emit_workunit_round_trips_hcli():
    row = d.emit_resident_workunit()
    wu = WorkUnit.from_dict(dict(row))
    assert wu.id == row["id"]
    assert wu.resource_class == "LIGHT_CONTROL"
    assert row["may_promote"] is False
    assert row["command"]


def test_classify_unknown_is_a_real_outcome():
    rec = {"expected_receipt_path": "/no/such/receipt.json"}
    assert d.classify_terminal(rec, returncode=None) == "unknown"
    assert d.classify_terminal(rec, returncode=0) == "completed-without-receipt"
    assert d.classify_terminal(rec, returncode=9) == "crashed"
    assert d.classify_terminal(rec, returncode=0, cancelled=True) == "cancelled"
    assert d.classify_terminal(rec, returncode=0, timed_out=True) == "timed_out"


def test_module_launch_function(tmp_path):
    rec = d.launch(_sleep("mod-launch", 8), workspace=tmp_path)
    try:
        assert rec.get("terminal") is None
        assert rec.get("supervisor_pid")
    finally:
        d.cancel(rec["job_id"], workspace=tmp_path)

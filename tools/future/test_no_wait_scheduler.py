"""Negative controls for tools/future/no_wait_scheduler.py.

A guard nobody has watched fail is not a guard. These tests spawn real
processes and read timestamps. They do not encode this sparse checkout.

Mandatory negatives:
- an interval where a detached handle is open and another unit completes;
  overlap is asserted from timestamps, not from intent
- a cancelled handle must actually stop; the process is gone
- with no independent work available, the scheduler reports BLOCKED with the
  dependency named — never spin
- a detached job whose receipt never appears is process_failed, never forgotten
"""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import pytest

from hcli.resources import pid_is_alive
from hcli.workunit import WorkUnit
from tools.future import no_wait_scheduler as nws
from tools.future import workgraph as wg
from tools.future._common import RECEIPTS, _assert_no_hardware_claims
from tools.future.detached import UnsafeCommandError
from tools.future.wakeup import POLL_ALIASES, Watcher


def _unit(name: str, receipt: Path, *, sleep_s: float | None = None, **over):
    body = "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('{\"ok\": true}\\n')"
    if sleep_s is not None:
        body = (
            "import sys,time; from pathlib import Path; "
            f"time.sleep({float(sleep_s)}); "
            "Path(sys.argv[1]).write_text('{\"ok\": true}\\n')"
        )
    row = {
        "id": name,
        "role": "science",
        "description": f"test unit {name}",
        "command": [sys.executable, "-c", body, str(receipt)],
        "resource_class": "LIGHT_CONTROL",
        "output_receipt_path": str(receipt),
        "verifier": "future.no_wait_scheduler.ingest_ready",
        "classification": "STATIC_ONLY",
        "timeout_s": over.pop("timeout_s", 30.0),
        "dependencies": list(over.pop("dependencies", ())),
    }
    row.update(over)
    return row


@pytest.fixture
def sched(tmp_path: Path):
    s = nws.NoWaitScheduler(tmp_path)
    yield s
    s.reap_all()


# ---------------------------------------------------------------------------
# Receipt / entry point
# ---------------------------------------------------------------------------


def test_build_emits_sealed_receipt():
    out = nws.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "NO_WAIT_SCHEDULER.json"
    assert doc["schema"] == nws.SCHEMA
    assert doc["schema"] == "hawking.future.no_wait_scheduler.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["proofs_all_passed"] is True
    assert doc["overlap_is_an_interval"] is True
    assert doc["does_not_call_subprocess_run_for_work"] is True
    assert doc["proofs"]["overlap_interval"]["passed"] is True
    assert doc["proofs"]["overlap_interval"]["slow_open_at_fast_finish"] is True
    assert doc["proofs"]["cancel_stops"]["passed"] is True
    assert doc["proofs"]["blocked_names_dependency"]["passed"] is True
    assert doc["proofs"]["missing_receipt_is_process_failed"]["passed"] is True
    assert doc["proofs"]["rediscover_adopts"]["passed"] is True
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert "resident_callable" in doc
    callable_doc = doc["resident_callable"]
    assert callable_doc["entry_point"]
    assert callable_doc["workunit"]
    assert callable_doc["receipt"].endswith("NO_WAIT_SCHEDULER.json")
    assert callable_doc["frontier"] == "FT.HCLI_SELF.no-launch"
    assert callable_doc["fails_closed"]
    assert "VI" not in "".join(doc["eras"])
    assert all("Odyssey IV" not in item and not item.startswith("IV ") for item in doc["odysseys"])
    _assert_no_hardware_claims(doc)
    WorkUnit.from_dict(dict(callable_doc["workunit_row"]))


def test_ast_module_is_parseable_and_has_no_stubs():
    src = Path(nws.__file__).read_text()
    compile(src, nws.__file__, "exec")
    for needle in ("TODO", "NotImplementedError", "pytest.skip"):
        assert needle not in src
    assert "subprocess.run(" not in src


def test_poll_source_is_not_a_loop():
    src = inspect.getsource(nws.NoWaitScheduler.poll)
    assert "sleep" not in src
    assert "while " not in src
    src_ingest = inspect.getsource(nws.NoWaitScheduler.ingest_ready)
    assert "sleep" not in src_ingest
    assert "wait_terminal" not in src_ingest


# ---------------------------------------------------------------------------
# Fail closed on absent inputs
# ---------------------------------------------------------------------------


def test_absent_workspace_is_refused():
    with pytest.raises(nws.SchedulerError) as exc:
        nws._require_workspace(None)
    assert exc.value.fault == "workspace_required"


def test_absent_unit_is_refused(tmp_path: Path):
    with pytest.raises(nws.SchedulerError) as exc:
        nws.expected_receipt(None, workspace=tmp_path)
    assert exc.value.fault == "unit_required"
    with pytest.raises(nws.SchedulerError) as exc2:
        nws.launch_detached(None, workspace=tmp_path)
    assert exc2.value.fault == "unit_required"


def test_unit_without_id_is_refused(tmp_path: Path):
    with pytest.raises(nws.SchedulerError) as exc:
        nws.expected_receipt(
            {"command": ["/bin/sleep", "1"], "role": "science"},
            workspace=tmp_path,
        )
    assert exc.value.fault == "unit_required"


def test_cancel_missing_handle_is_refused(sched: nws.NoWaitScheduler):
    with pytest.raises(nws.SchedulerError) as exc:
        sched.cancel({"job_id": "no-such-job"})
    assert exc.value.fault == "handle_required"


def test_poll_empty_handles_is_empty(sched: nws.NoWaitScheduler):
    assert sched.poll([]) == []
    landed = sched.ingest_ready([])
    assert landed["landed"] == []
    assert landed["still_open"] == []
    assert landed["forgotten"] == []


# ---------------------------------------------------------------------------
# THE proof: overlap interval from timestamps
# ---------------------------------------------------------------------------


def test_overlap_interval_from_timestamps_not_intent(tmp_path: Path):
    proof = nws.prove_overlap_interval(tmp_path)
    assert proof["passed"] is True
    assert proof["slow_open_at_fast_progress"] is True
    assert proof["slow_open_at_fast_finish"] is True
    assert proof["fast_ingest"] == nws.INGESTED
    assert proof["fast_finished_at"] is not None
    assert proof["slow_launched_at"] <= proof["fast_launched_at"] <= proof["fast_finished_at"]
    assert proof["started_marker_present"] is True
    assert proof["fast_receipt_present"] is True
    assert proof["runnable_status_while_slow_open"] == nws.RUNNABLE


def test_launch_returns_before_child_finishes(sched: nws.NoWaitScheduler, tmp_path: Path):
    receipt = tmp_path / "results" / "slow.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    handle = sched.launch_detached(_unit("still-running", receipt, sleep_s=8.0))
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"launch_detached blocked for {elapsed:.3f}s"
    assert handle.get("terminal") is None
    assert handle.get("job_id")
    assert Path(handle["expected_receipt_path"]).parent.exists()
    contract = sched._load_contract(handle["job_id"])
    assert contract is not None
    assert contract["path"] == handle["expected_receipt_path"]
    assert contract["absent_is"] == nws.PROCESS_FAILED
    snaps = sched.poll([handle])
    assert snaps[0]["ingest"] == nws.OPEN
    cancelled = sched.cancel(handle)
    assert cancelled["terminal"] == "cancelled"
    assert cancelled["process_gone"] is True


# ---------------------------------------------------------------------------
# Cancel actually stops
# ---------------------------------------------------------------------------


def test_cancelled_handle_process_is_gone(tmp_path: Path):
    proof = nws.prove_cancel_stops(tmp_path)
    assert proof["passed"] is True
    assert proof["terminal"] == "cancelled"
    assert proof["process_gone"] is True
    assert proof["pid_still_alive"] is False


def test_cancel_of_live_sleep_reaps_pid(sched: nws.NoWaitScheduler, tmp_path: Path):
    receipt = tmp_path / "results" / "c.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    handle = sched.launch_detached(_unit("reap-me", receipt, sleep_s=20.0, timeout_s=60.0))
    deadline = time.monotonic() + 2.0
    pid = handle.get("pid")
    while time.monotonic() < deadline and not (isinstance(pid, int) and pid > 0):
        snaps = sched.poll([handle])
        pid = snaps[0].get("pid") if snaps else None
        time.sleep(0.02)
    cancelled = sched.cancel(handle)
    assert cancelled["terminal"] == "cancelled"
    assert cancelled["process_gone"] is True
    if isinstance(pid, int) and pid > 0:
        assert not pid_is_alive(pid)


# ---------------------------------------------------------------------------
# BLOCKED with the dependency named — never spin
# ---------------------------------------------------------------------------


def test_no_independent_work_reports_blocked_with_named_dependency(tmp_path: Path):
    proof = nws.prove_blocked_names_dependency(tmp_path)
    assert proof["passed"] is True
    assert proof["status"] == nws.BLOCKED
    assert proof["named_dependency"] == "WU.PRE"
    assert proof["spin"] is False
    assert proof["n_block_events"] >= 1


def test_blocked_from_candidate_list_names_the_dependency(
    sched: nws.NoWaitScheduler, tmp_path: Path
):
    receipt = tmp_path / "results" / "a.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    a = _unit("WU.A", receipt, sleep_s=8.0)
    b = {
        "id": "WU.B",
        "dependencies": ["WU.A"],
        "role": "science",
        "description": "depends on A",
    }
    handle = sched.launch_detached(a)
    try:
        view = sched.runnable_now([handle], candidates=[a, b])
        assert view["status"] == nws.BLOCKED
        assert view["named_dependency"] == "WU.A"
        assert view["runnable"] == []
        assert view["spin"] is False
        assert view["whole_resident_wait"] is True
        assert sched.block_events
        assert sched.block_events[-1]["named_dependency"] == "WU.A"
    finally:
        sched.cancel(handle)


def test_independent_peer_is_runnable_while_handle_is_open(
    sched: nws.NoWaitScheduler, tmp_path: Path
):
    receipt = tmp_path / "results" / "a.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    a = _unit("WU.A", receipt, sleep_s=8.0)
    c = {"id": "WU.C", "dependencies": [], "description": "independent"}
    b = {"id": "WU.B", "dependencies": ["WU.A"], "description": "waits on A"}
    handle = sched.launch_detached(a)
    try:
        view = sched.runnable_now([handle], candidates=[a, b, c])
        assert view["status"] == nws.RUNNABLE
        assert view["spin"] is False
        names = [nws._unit_id_of(u) for u in view["runnable"]]
        assert "WU.C" in names
        assert "WU.A" not in names
        assert "WU.B" not in names
        assert view["named_dependency"] is None
    finally:
        sched.cancel(handle)


def test_nothing_in_flight_and_nothing_to_run_is_idle_not_blocked():
    view = nws.runnable_now([], candidates=[])
    assert view["status"] == nws.IDLE
    assert view["spin"] is False
    assert view["named_dependency"] is None


def test_graph_independent_unit_runs_while_predecessor_is_open(
    sched: nws.NoWaitScheduler, tmp_path: Path
):
    graph = wg.WorkGraph(workspace=tmp_path, ncpu=2)
    for uid, deps in (("WU.PRE", []), ("WU.DEP", ["WU.PRE"]), ("WU.PEER", [])):
        graph.admit(
            wg.make_unit(
                id=uid,
                role="science",
                description=uid,
                dependencies=deps,
                resource_lane="CPU_ANALYSIS",
                mutation_scope=[],
                verifier=f"future.no_wait.{uid}",
                expected_information_gain=2,
                cost_units=1,
                requires_hardware=False,
            )
        )
    receipt = tmp_path / "results" / "pre.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    handle = sched.launch_detached(_unit("WU.PRE", receipt, sleep_s=8.0))
    try:
        view = sched.runnable_now([handle], graph=graph)
        assert view["status"] == nws.RUNNABLE
        names = [nws._unit_id_of(u) for u in view["runnable"]]
        assert "WU.PEER" in names
        assert "WU.DEP" not in names
        assert "WU.PRE" not in names
    finally:
        sched.cancel(handle)


# ---------------------------------------------------------------------------
# Missing receipt → process_failed, never forgotten
# ---------------------------------------------------------------------------


def test_missing_receipt_times_out_as_process_failed(tmp_path: Path):
    proof = nws.prove_missing_receipt_is_process_failed(tmp_path)
    assert proof["passed"] is True
    assert proof["ingest"] == nws.PROCESS_FAILED
    assert proof["receipt_present"] is False
    assert proof["forgotten"] == []
    assert proof["still_known_after_rediscover"] is True


def test_exit_zero_without_receipt_is_process_failed(
    sched: nws.NoWaitScheduler, tmp_path: Path
):
    expected = tmp_path / "results" / "missing.json"
    expected.parent.mkdir(parents=True, exist_ok=True)
    unit = {
        "id": "no-receipt",
        "role": "science",
        "description": "exits 0 without writing the contract path",
        "command": [sys.executable, "-c", "raise SystemExit(0)"],
        "resource_class": "LIGHT_CONTROL",
        "output_receipt_path": str(expected),
        "verifier": "future.no_wait_scheduler.ingest_ready",
        "classification": "STATIC_ONLY",
        "timeout_s": 10.0,
        "dependencies": [],
    }
    handle = sched.launch_detached(unit)
    deadline = time.monotonic() + 5.0
    terminal = None
    while time.monotonic() < deadline:
        snaps = sched.poll([handle])
        terminal = snaps[0].get("terminal")
        if terminal is not None:
            break
        time.sleep(0.05)
    landed = sched.ingest_ready([handle])
    assert landed["forgotten"] == []
    assert len(landed["landed"]) == 1
    row = landed["landed"][0]
    assert row["ingest"] == nws.PROCESS_FAILED
    assert row["job_id"] == handle["job_id"]
    assert not expected.is_file()
    assert terminal in {
        "completed-without-receipt",
        "timed_out",
        "crashed",
        "unknown",
    }


def test_ingest_orders_by_landing_time(sched: nws.NoWaitScheduler, tmp_path: Path):
    first_path = tmp_path / "results" / "first.json"
    second_path = tmp_path / "results" / "second.json"
    first_path.parent.mkdir(parents=True, exist_ok=True)
    h1 = sched.launch_detached(_unit("first", first_path, sleep_s=2.0))
    h2 = sched.launch_detached(_unit("second", second_path, sleep_s=0.05))
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        landed = sched.ingest_ready([h1, h2])
        if landed["n_landed"] == 2:
            ids = [r["job_id"] for r in landed["landed"]]
            assert ids[0] == h2["job_id"]
            assert ids[1] == h1["job_id"]
            assert all(r["ingest"] == nws.INGESTED for r in landed["landed"])
            return
        time.sleep(0.05)
    raise AssertionError("both units did not land within the window")


def test_missing_supervision_record_is_process_failed_not_forgotten(
    sched: nws.NoWaitScheduler,
):
    snaps = sched.poll([{"job_id": "ghost-never-launched"}])
    assert len(snaps) == 1
    assert snaps[0]["ingest"] == nws.PROCESS_FAILED
    assert snaps[0]["state"] == "missing"
    landed = sched.ingest_ready([{"job_id": "ghost-never-launched"}])
    assert landed["forgotten"] == []
    assert landed["landed"][0]["ingest"] == nws.PROCESS_FAILED


# ---------------------------------------------------------------------------
# Rediscover adopts; never relaunches
# ---------------------------------------------------------------------------


def test_rediscover_adopts_live_child(tmp_path: Path):
    proof = nws.prove_rediscover_adopts(tmp_path)
    assert proof["passed"] is True
    assert proof["relaunched"] is False


def test_fresh_scheduler_on_same_workspace_adopts(
    sched: nws.NoWaitScheduler, tmp_path: Path
):
    receipt = tmp_path / "results" / "live.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    handle = sched.launch_detached(_unit("live-child", receipt, sleep_s=12.0, timeout_s=40.0))
    other = nws.NoWaitScheduler(tmp_path)
    try:
        report = other.rediscover()
        assert report["relaunched"] is False
        ids = [h.get("job_id") for h in report["handles"]]
        assert handle["job_id"] in ids
        mine = next(h for h in report["handles"] if h["job_id"] == handle["job_id"])
        assert mine.get("relaunched") is False
        snaps = other.poll([mine])
        assert snaps[0]["ingest"] in {nws.OPEN, nws.INGESTED, nws.PROCESS_FAILED}
    finally:
        other.cancel(handle)


# ---------------------------------------------------------------------------
# Unsafe launch still fails closed; GPU is not seized to stay busy
# ---------------------------------------------------------------------------


def test_gpu_unit_is_refused_not_run_to_fill_the_clock(
    sched: nws.NoWaitScheduler, tmp_path: Path
):
    with pytest.raises(UnsafeCommandError) as exc:
        sched.launch_detached(
            {
                "id": "gpu-seizure",
                "command": ["/bin/sleep", "1"],
                "resource_class": "GPU_EXCLUSIVE",
                "role": "science",
                "description": "must sleep, not run",
                "verifier": "future.no_wait_scheduler.ingest_ready",
            }
        )
    assert exc.value.record is not None
    assert exc.value.record["pid"] is None
    assert exc.value.record["state"] == "SLEEPING"


# ---------------------------------------------------------------------------
# Wakeup still refuses model poll; this poll is the supervisor entry
# ---------------------------------------------------------------------------


def test_wakeup_still_refuses_model_poll_aliases(tmp_path: Path):
    watcher = Watcher(tmp_path / "ledger.json", root=tmp_path)
    for name in POLL_ALIASES:
        with pytest.raises(AttributeError):
            getattr(watcher, name)


def test_poll_is_cheap(sched: nws.NoWaitScheduler, tmp_path: Path):
    receipt = tmp_path / "results" / "p.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    handle = sched.launch_detached(_unit("poll-cheap", receipt, sleep_s=8.0))
    try:
        t0 = time.monotonic()
        snaps = sched.poll([handle, handle, handle])
        elapsed = time.monotonic() - t0
        assert elapsed < 1.5, f"poll looped: {elapsed:.3f}s"
        assert len(snaps) == 3
        assert all(s["job_id"] == handle["job_id"] for s in snaps)
    finally:
        sched.cancel(handle)


def test_frontier_path_copes_when_book_is_or_is_not_loadable():
    view = nws.runnable_now([])
    assert view["status"] in {nws.RUNNABLE, nws.IDLE, nws.REFUSED, nws.BLOCKED}
    assert view.get("spin") is False
    if view["status"] == nws.REFUSED:
        assert view["why"]
    if view["status"] == nws.BLOCKED:
        assert view["named_dependency"]
    if view["status"] == nws.RUNNABLE:
        assert view["runnable"]


def test_expected_receipt_names_the_path_without_spawning(tmp_path: Path):
    receipt = tmp_path / "results" / "named.json"
    contract = nws.expected_receipt(
        _unit("named", receipt),
        workspace=tmp_path,
    )
    assert contract["schema"] == nws.CONTRACT_SCHEMA
    assert contract["path"]
    assert contract["absent_is"] == nws.PROCESS_FAILED
    assert contract["required_schema"] is None
    assert contract["unit_id"] == "named"
    assert not Path(contract["path"]).is_file()

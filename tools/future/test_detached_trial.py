"""G008 detached-work trial: a real timeline, not a fixture.

A synthetic test does not satisfy the obligation. These tests spawn real
processes through tools/future/detached.py, assert overlap from timestamps,
and read the campaign receipt produced by a real run. They do not encode
this sparse checkout.
"""
from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import pytest

from hcli.resources import pid_is_alive
from hcli.workunit import WorkUnit
from tools.future import detached as d
from tools.future import detached_trial as dt
from tools.future._common import RECEIPTS, _assert_no_hardware_claims
from tools.future.wakeup import COMPLETED, POLL_ALIASES, Watcher


def _sleep_free(src: str) -> None:
    assert "/bin/sleep" not in src


# ---------------------------------------------------------------------------
# Cheap: source, guards, WorkUnit. No 128s child.
# ---------------------------------------------------------------------------


def test_module_consumes_detached_and_does_not_fork_it():
    src = Path(dt.__file__).read_text(encoding="utf-8")
    run_src = inspect.getsource(dt.run_real_trial)
    assert "from tools.future import detached as d" in src
    assert dt.d is d
    assert "def supervise(" not in src
    assert "Popen(" not in run_src
    assert "subprocess.run" not in run_src
    assert "wait_terminal(" not in run_src
    _sleep_free(inspect.getsource(dt.child_work))
    _sleep_free(inspect.getsource(dt.make_child_unit))


def test_poll_aliases_remain_refused_on_wakeup_watcher(tmp_path: Path):
    watcher = Watcher(tmp_path / "ledger.json", root=tmp_path)
    for name in POLL_ALIASES:
        with pytest.raises(AttributeError):
            getattr(watcher, name)


def test_emit_workunit_round_trips_hcli():
    row = dt.emit_resident_workunit()
    wu = WorkUnit.from_dict(dict(row))
    assert wu.id == row["id"]
    assert wu.resource_class == "LIGHT_CONTROL"
    assert row["may_promote"] is False
    assert row["command"]


def test_1h_shaped_timeline_fails_the_guards():
    """Negative control: the AUTONOMY_TIMELINE_1h shape must FAIL."""
    same_25 = [f"FT.SAME.{i:02d}" for i in range(25)]
    timeline = [
        {"kind": "idea_rejected", "t_s": 7.0 + i * 0.04, "payload": {"table": "fixed"}}
        for i in range(222)
    ]
    timeline.append({"kind": "CHILD_LAUNCHED", "t_s": 0.1, "payload": {}})
    timeline.append({"kind": "NEXT_DECISION", "t_s": 119.0, "payload": {}})
    guards = dt.evaluate_autonomy_1h_guards(
        timeline,
        elapsed_s=3600.0,
        refill_sets=[same_25, same_25, same_25, same_25],
        refusals=[],
        specimen_ingests=29,
    )
    assert guards["passed"] is False
    assert guards["rejection_table"]["passed"] is False
    assert guards["refill_novelty"]["passed"] is False
    assert guards["decisions_after_two_minutes"]["passed"] is False
    assert guards["specimen_verification_not_looped"] is False


def test_short_run_fails_two_minute_guard_honestly():
    """A 10s run cannot claim it beat the 1h decision cutoff."""
    timeline = [
        {"kind": "CHILD_LAUNCHED", "t_s": 0.0, "payload": {}},
        {"kind": "INDEPENDENT_STARTED", "t_s": 0.1, "payload": {}},
        {"kind": "INDEPENDENT_COMPLETED", "t_s": 1.0, "payload": {}},
        {"kind": "WORK_REFILLED", "t_s": 1.1, "payload": {}},
        {"kind": "WORK_REFILLED", "t_s": 4.0, "payload": {}},
        {"kind": "RECEIPT_WAKEUP", "t_s": 9.5, "payload": {}},
        {"kind": "NEXT_DECISION", "t_s": 9.6, "payload": {}},
    ]
    guards = dt.evaluate_autonomy_1h_guards(
        timeline,
        elapsed_s=10.0,
        refill_sets=[["a", "b"], ["c"], ["d"]],
        refusals=[],
        specimen_ingests=0,
    )
    assert guards["refill_novelty"]["passed"] is True
    assert guards["rejection_table"]["passed"] is True
    assert guards["decisions_after_two_minutes"]["passed"] is False
    assert guards["passed"] is False


def test_passing_guard_shape_requires_novel_refills_and_late_decisions():
    timeline = [
        {"kind": "CHILD_LAUNCHED", "t_s": 0.05, "payload": {}},
        {"kind": "INDEPENDENT_STARTED", "t_s": 0.2, "payload": {}},
        {"kind": "WORK_REFILLED", "t_s": 0.2, "payload": {}},
        {"kind": "INDEPENDENT_COMPLETED", "t_s": 5.0, "payload": {}},
        {"kind": "WORK_REFILLED", "t_s": 5.1, "payload": {}},
        {"kind": "RECEIPT_WAKEUP", "t_s": 128.0, "payload": {}},
        {"kind": "NEXT_DECISION", "t_s": 128.05, "payload": {}},
    ]
    guards = dt.evaluate_autonomy_1h_guards(
        timeline,
        elapsed_s=130.0,
        refill_sets=[["WU.A", "WU.B"], ["WU.C"], ["WU.D"]],
        refusals=[{"unit_id": "WU.X", "t_s": 0.3}, {"unit_id": "WU.Y", "t_s": 40.0}],
        specimen_ingests=0,
    )
    assert guards["passed"] is True
    assert guards["rejection_table"]["passed"] is True
    assert guards["refill_novelty"]["passed"] is True
    assert guards["decisions_after_two_minutes"]["passed"] is True


# ---------------------------------------------------------------------------
# Live short trial: real pid, overlap, kqueue wakeup, bound. ~7s.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def short_trial(tmp_path_factory):
    result = dt.run_real_trial(
        tmp_path_factory.mktemp("short-trial"),
        child_duration_s=6.0,
        independent_duration_s=0.6,
        bound=2,
        exceed_attempts=1,
        exceed_second_after_s=99.0,
    )
    return result


def test_live_child_is_a_real_os_process(short_trial):
    child = short_trial["child"]
    assert isinstance(child["pid"], int) and child["pid"] > 1
    assert child["start_token"]
    assert child["launch_return_s"] < 2.0
    assert child["expected_receipt_path"]
    assert child["receipt_present"] is True
    assert child["start_new_session"] is True
    assert child["terminal"] in {
        "completed-with-receipt",
        "completed-without-receipt",
        "cancelled",
        "timed_out",
        "crashed",
        "unknown",
        None,
    }
    # The child of this short trial should have finished.
    assert child["terminal"] == "completed-with-receipt"
    assert not pid_is_alive(int(child["pid"]))


def test_live_independent_units_complete_while_child_runs(short_trial):
    n_wait = short_trial["independent"]["n_completed_during_wait"]
    n_started = short_trial["independent"]["n_started"]
    assert n_started >= 2
    assert n_wait >= 1
    child_launch = next(e for e in short_trial["timeline"] if e["kind"] == "CHILD_LAUNCHED")
    wakeup = next(e for e in short_trial["timeline"] if e["kind"] == "RECEIPT_WAKEUP")
    during = [
        e
        for e in short_trial["timeline"]
        if e["kind"] == "INDEPENDENT_COMPLETED"
        and child_launch["t_s"] <= e["t_s"] <= wakeup["t_s"]
    ]
    assert during, "overlap must be a timestamp interval, not intent"
    ids = [e["payload"].get("unit_id") for e in during]
    assert all(str(i).startswith("WU.DETACHED_TRIAL.ind.") for i in ids if i)


def test_live_idle_runnable_seconds_is_reported(short_trial):
    idle = short_trial["idle_runnable_seconds"]
    assert isinstance(idle, (int, float))
    assert idle >= 0.0
    assert idle < dt.IDLE_DEFECT_S
    assert short_trial["idle_is_autonomy_defect"] is False


def test_live_wakeup_is_kqueue_not_a_coincident_poll(short_trial):
    wu = short_trial["wakeup"]
    assert wu is not None
    assert str(wu["mechanism"]).startswith("kqueue")
    assert wu["poll"] is False
    assert wu["wait_terminal"] is False
    latency = wu.get("kqueue_latency_s")
    assert isinstance(latency, (int, float))
    # A 1s+ poll coinciding with landing would sit near the poll interval.
    # kqueue on this host is milliseconds; allow 2s of supervisor-scheduling slack.
    assert latency < 2.0
    reaction = wu.get("supervisor_reaction_latency_s")
    assert isinstance(reaction, (int, float))
    assert reaction < 2.0
    states = []
    for item in wu.get("dispatch") or []:
        if isinstance(item, dict):
            states.append(item.get("state"))
    assert COMPLETED in states
    nexts = [e for e in short_trial["timeline"] if e["kind"] == "NEXT_DECISION"]
    assert nexts
    wakeup_t = next(e["t_s"] for e in short_trial["timeline"] if e["kind"] == "RECEIPT_WAKEUP")
    assert nexts[0]["t_s"] >= wakeup_t - 1e-6


def test_live_concurrency_bound_refuses_overflow(short_trial):
    conc = short_trial["concurrency"]
    assert conc["bound"] == 2
    assert conc["held"] is True
    assert conc["n_exceed_attempts"] >= 1
    assert conc["refusals"]
    for row in conc["refusals"]:
        assert row.get("spawned") is False
        assert row.get("pid") is None
        assert row.get("in_flight") >= conc["bound"] or row.get("live_before", conc["bound"]) >= conc["bound"]


def test_live_refills_are_novel(short_trial):
    sets = short_trial["refills"]["sets"]
    assert len(sets) >= 2
    guards = dt.evaluate_autonomy_1h_guards(
        short_trial["timeline"],
        elapsed_s=short_trial["elapsed_s"],
        refill_sets=sets,
        refusals=short_trial["concurrency"]["refusals"],
        specimen_ingests=0,
    )
    assert guards["refill_novelty"]["passed"] is True
    assert guards["rejection_table"]["passed"] is True


def test_live_bounded_launcher_does_not_spawn_above_bound(tmp_path: Path):
    sup = d.DetachedSupervisor(tmp_path)
    launcher = dt.BoundedIndependentLauncher(sup, bound=2)
    launched = []
    try:
        for i in range(2):
            unit = dt.make_independent_unit(tmp_path, i, duration_s=8.0)
            rec = launcher.launch(unit)
            launched.append(rec)
            assert rec.get("pid")
            assert pid_is_alive(int(rec["pid"]))
        extra = dt.make_independent_unit(tmp_path, 99, duration_s=8.0)
        with pytest.raises(dt.ConcurrencyBoundError) as ei:
            launcher.launch(extra)
        assert ei.value.bound == 2
        assert ei.value.in_flight == 2
        assert ei.value.unit_id == extra["id"]
        assert launcher.n_live() == 2
        for rec in launched:
            assert pid_is_alive(int(rec["pid"]))
    finally:
        sup.reap_all()


# ---------------------------------------------------------------------------
# Campaign receipt: the obligation's deliverable. ~128s real child.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def campaign_doc():
    path = dt.selftest()
    assert path.parent == RECEIPTS
    assert path.name == dt.RECEIPT
    return json.loads(path.read_text(encoding="utf-8"))


def test_campaign_receipt_is_sealed_static_only(campaign_doc):
    doc = campaign_doc
    assert doc["schema"] == dt.SCHEMA
    assert doc["schema"] == "hawking.future.detached_trial.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["synthetic"] is False
    assert doc["fixture"] is False
    assert "VI" not in "".join(doc["eras"])
    assert all("Odyssey IV" not in item and not item.startswith("IV ") for item in doc["odysseys"])
    _assert_no_hardware_claims(doc)
    WorkUnit.from_dict(dict(doc["resident_callable"]["workunit"]))


def test_campaign_timeline_is_a_real_run_not_a_fixture(campaign_doc):
    doc = campaign_doc
    child = doc["child"]
    assert isinstance(child["pid"], int) and child["pid"] > 1
    assert child["start_token"]
    assert isinstance(doc["t0_unix"], (int, float))
    # selftest just ran this process; a planted fixture would be hours/days old.
    assert abs(time.time() - float(doc["t0_unix"])) < 600.0
    assert float(doc["elapsed_s"]) > dt.TWO_MINUTES_S
    assert float(doc["elapsed_s"]) >= 0.9 * dt.DEFAULT_CHILD_DURATION_S
    ts = [float(e["t_s"]) for e in doc["timeline"]]
    assert ts == sorted(ts)
    assert ts[0] >= 0.0
    kinds = [e["kind"] for e in doc["timeline"]]
    assert "CHILD_LAUNCHED" in kinds
    assert "INDEPENDENT_STARTED" in kinds
    assert "INDEPENDENT_COMPLETED" in kinds
    assert "RECEIPT_WAKEUP" in kinds
    assert "NEXT_DECISION" in kinds
    assert "WORK_REFILLED" in kinds
    launched = next(e for e in doc["timeline"] if e["kind"] == "CHILD_LAUNCHED")
    assert launched["payload"]["pid"] == child["pid"]
    assert launched["payload"]["start_token"] == child["start_token"]
    assert launched["payload"]["terminal_at_return"] is None
    assert launched["t_unix"] >= doc["t0_unix"] - 0.01


def test_campaign_idle_runnable_seconds_reported(campaign_doc):
    idle = campaign_doc["idle_runnable_seconds"]
    assert isinstance(idle, (int, float))
    assert idle >= 0.0
    assert "idle_runnable_seconds" in campaign_doc
    assert campaign_doc["idle_is_autonomy_defect"] is False
    assert idle <= dt.IDLE_DEFECT_S
    assert campaign_doc["proofs"]["idle_runnable_seconds"]["value"] == idle


def test_campaign_wakeup_latency_measured(campaign_doc):
    wu = campaign_doc["wakeup"]
    assert wu is not None
    assert str(wu["mechanism"]).startswith("kqueue")
    assert wu["poll"] is False
    latency = wu["kqueue_latency_s"]
    reaction = wu["supervisor_reaction_latency_s"]
    assert isinstance(latency, (int, float))
    assert isinstance(reaction, (int, float))
    assert latency < 2.0
    assert reaction < 2.0
    assert wu.get("receipt_written_at")
    assert campaign_doc["kqueue_armed_before_launch"] is True
    nexts = [e for e in campaign_doc["timeline"] if e["kind"] == "NEXT_DECISION"]
    wakes = [e for e in campaign_doc["timeline"] if e["kind"] == "RECEIPT_WAKEUP"]
    assert wakes and nexts
    assert nexts[0]["t_s"] >= wakes[0]["t_s"] - 1e-6


def test_campaign_concurrency_bound_enforced(campaign_doc):
    conc = campaign_doc["concurrency"]
    assert conc["bound"] == dt.SAFE_IN_FLIGHT_BOUND
    assert conc["bound"] == 2
    assert conc["held"] is True
    assert conc["n_exceed_attempts"] >= 1
    assert conc["refusals"]
    for row in conc["refusals"]:
        assert row["spawned"] is False
        assert row["pid"] is None
    assert campaign_doc["safe_in_flight_bound"] == 2


def test_campaign_autonomy_1h_guards_reported_either_way(campaign_doc):
    guards = campaign_doc["autonomy_1h_guards"]
    assert "passed" in guards
    assert "rejection_table" in guards
    assert "refill_novelty" in guards
    assert "decisions_after_two_minutes" in guards
    assert guards["rejection_table"]["passed"] is True
    assert guards["refill_novelty"]["passed"] is True
    late = guards["decisions_after_two_minutes"]
    assert late["elapsed_s"] > dt.TWO_MINUTES_S
    assert late["n_decisions_after_120"] >= 1
    assert late["last_decision_t_s"] > dt.TWO_MINUTES_S
    assert late["passed"] is True
    assert guards["specimen_verification_ingests"] == 0
    assert guards["passed"] is True
    assert campaign_doc["verdict"] == "PASS"
    assert campaign_doc["unmet"] == []
    assert campaign_doc["proofs_all_passed"] is True


def test_campaign_independent_progress_covers_the_wait(campaign_doc):
    n_wait = campaign_doc["independent"]["n_completed_during_wait"]
    assert n_wait >= 2
    wakeup_t = next(e["t_s"] for e in campaign_doc["timeline"] if e["kind"] == "RECEIPT_WAKEUP")
    assert wakeup_t > dt.TWO_MINUTES_S
    during = [
        e
        for e in campaign_doc["timeline"]
        if e["kind"] == "INDEPENDENT_COMPLETED" and e["t_s"] < wakeup_t
    ]
    assert len(during) >= 2
    # Progress is not bunched only at t=0: at least one completion after 60s.
    assert any(e["t_s"] > 60.0 for e in during)


def test_campaign_does_not_call_wait_terminal_or_subprocess_run(campaign_doc):
    src = campaign_doc["source_guards"]
    assert src["run_real_trial_calls_wait_terminal"] is False
    assert src["run_real_trial_calls_subprocess_run"] is False
    assert src["run_real_trial_calls_popen"] is False
    assert src["module_imports_detached"] is True
    assert src["module_imports_wakeup"] is True
    assert campaign_doc["does_not_call_wait_terminal_on_child"] is True
    assert campaign_doc["consumes_detached_py"] is True
    assert campaign_doc["does_not_fork_detached_py"] is True

"""NO-WAIT ORCHESTRATION is FAIL, not a slow resident.

These tests pin the three-way verdict in both directions, the runnable-safe-work
rule one clause at a time, and the real on-disk timelines — AUTONOMY_TIMELINE_1h
and DETACHED_WORK_TRIAL — not fixtures. The 1h last-2864s tail is adjudicated
from that timeline's events. A would-PASS trial cannot PASS through the public
API while FAIL_NO_WAIT_ORCHESTRATION holds.
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.future import improvement_trial as it
from tools.future import no_wait_orchestration as nwo
from tools.future import no_wait_scheduler as nws
from tools.future._common import RECEIPTS, REPO, _assert_no_hardware_claims


TIMELINE_1H = REPO / nwo.TIMELINE_1H_REL
DETACHED = REPO / nwo.DETACHED_TRIAL_REL
POWER_TORTURE_TIMELINE = REPO / nwo.POWER_TORTURE_TIMELINE_REL


def _ev(kind: str, t_s: float, payload: dict | None = None, **extra) -> dict:
    row = {"kind": kind, "t_s": t_s, "payload": dict(payload or {})}
    row.update(extra)
    return row


def _unit(
    uid: str,
    *,
    resource_class: str = "STATIC_ANALYSIS",
    dependencies: list | None = None,
    status: str | None = None,
    wake_condition: str | None = None,
    command: list | None = None,
    frontier_id: str | None = None,
    description: str | None = None,
) -> dict:
    row = {
        "id": uid,
        "resource_class": resource_class,
        "dependencies": list(dependencies or []),
    }
    if status is not None:
        row["status"] = status
    if wake_condition is not None:
        row["wake_condition"] = wake_condition
    if command is not None:
        row["command"] = command
    if frontier_id is not None:
        row["frontier_id"] = frontier_id
    if description is not None:
        row["description"] = description
    return row


def _launch(uid: str, t_s: float, **unit_kw) -> dict:
    return _ev("workunit_launched", t_s, {"unit": _unit(uid, **unit_kw)})


def _ingest(uid: str, t_s: float) -> dict:
    return _ev("result_ingested", t_s, {"unit_id": uid})


def _fail(uid: str, t_s: float) -> dict:
    return _ev("process_failed", t_s, {"unit_id": uid})


def _sleep(rc: str, t_s: float, wake: str) -> dict:
    return _ev(
        "workunit_sleeping",
        t_s,
        {"resource_class": rc, "wake_condition": wake},
    )


def _next_work(t_s: float, ids: list[str], *, n: int | None = None, source: str = "frontiers.refill") -> dict:
    return _ev(
        "next_work_left",
        t_s,
        {"ids": ids, "unit_ids": ids, "n": n if n is not None else len(ids), "source": source},
    )


# ---------------------------------------------------------------------------
# Runnable safe work — one test per clause
# ---------------------------------------------------------------------------


def test_gpu_lease_blocked_unit_is_not_runnable():
    world = nwo.World(lease_held=False)
    for rc in ("GPU_EXCLUSIVE", "GPU_DECODE", "GPU_DIRTY_OK"):
        hit = nwo.is_runnable_safe_work(_unit("WU.GPU", resource_class=rc), world)
        assert hit["runnable"] is False, (rc, hit)
        assert hit["clause"] == "RESOURCE"
        assert "lease" in hit["reason"].lower()


def test_gpu_unit_is_runnable_only_when_lease_held():
    unit = _unit("WU.GPU", resource_class="GPU_EXCLUSIVE")
    denied = nwo.is_runnable_safe_work(unit, nwo.World(lease_held=False))
    allowed = nwo.is_runnable_safe_work(unit, nwo.World(lease_held=True))
    assert denied["runnable"] is False
    assert allowed["runnable"] is True
    assert allowed["clause"] == "RUNNABLE"


def test_parked_ane_fpga_gpu_protected_are_not_runnable_without_qualification():
    world = nwo.World(lease_held=False)
    for rc in ("ANE", "FPGA", "GPU_PROTECTED"):
        hit = nwo.is_runnable_safe_work(_unit(f"WU.{rc}", resource_class=rc), world)
        assert hit["runnable"] is False, (rc, hit)
        assert hit["clause"] == "RESOURCE"


def test_resource_free_and_deps_met_is_runnable():
    world = nwo.World(completed_ids=frozenset({"WU.PRE"}))
    hit = nwo.is_runnable_safe_work(
        _unit("WU.CPU", resource_class="STATIC_ANALYSIS", dependencies=["WU.PRE"]),
        world,
    )
    assert hit["runnable"] is True
    assert hit["clause"] == "RUNNABLE"
    light = nwo.is_runnable_safe_work(_unit("WU.LIGHT", resource_class="LIGHT_CONTROL"), nwo.World())
    assert light["runnable"] is True


def test_unmet_dependencies_are_not_runnable():
    world = nwo.World(in_flight_ids=frozenset({"WU.PRE"}))
    hit = nwo.is_runnable_safe_work(
        _unit("WU.NEXT", dependencies=["WU.PRE"]),
        world,
    )
    assert hit["runnable"] is False
    assert hit["clause"] == "DEPENDENCIES"
    assert "WU.PRE" in hit["unmet_dependencies"]


def test_sleeping_unit_with_unsatisfied_wake_is_not_runnable():
    wake = "a qualified Metal GPU and Metal compiler, plus a real HCLI lease"
    unit = _unit("WU.SLEEP", resource_class="STATIC_ANALYSIS", status="SLEEPING", wake_condition=wake)
    hit = nwo.is_runnable_safe_work(unit, nwo.World())
    assert hit["runnable"] is False
    assert hit["clause"] == "SLEEP"
    parked = nwo.is_runnable_safe_work(
        _unit("WU.LANE", resource_class="STATIC_ANALYSIS"),
        nwo.World(sleeping_resources={"STATIC_ANALYSIS": wake}),
    )
    assert parked["runnable"] is False
    assert parked["clause"] == "SLEEP"


def test_sleeping_unit_with_satisfied_wake_is_runnable():
    wake = "lease acquired"
    unit = _unit("WU.WOKE", resource_class="LIGHT_CONTROL", status="SLEEPING", wake_condition=wake)
    hit = nwo.is_runnable_safe_work(
        unit, nwo.World(satisfied_wake_conditions=frozenset({wake}))
    )
    assert hit["runnable"] is True


def test_in_flight_and_completed_units_are_not_runnable():
    inflight = nwo.is_runnable_safe_work(
        _unit("WU.RUN"), nwo.World(in_flight_ids=frozenset({"WU.RUN"}))
    )
    done = nwo.is_runnable_safe_work(
        _unit("WU.DONE"), nwo.World(completed_ids=frozenset({"WU.DONE"}))
    )
    assert inflight["runnable"] is False and inflight["clause"] == "IDENTITY"
    assert done["runnable"] is False and done["clause"] == "IDENTITY"


def test_fpga_engine_sim_as_static_analysis_is_runnable():
    """FPGA.engine-sim launched as STATIC_ANALYSIS is CPU simulation, not parked FPGA."""
    hit = nwo.is_runnable_safe_work(
        _unit("WU.SIM", resource_class="STATIC_ANALYSIS", frontier_id="FT.FPGA.engine-sim")
    )
    assert hit["runnable"] is True


def test_static_analysis_of_gpu_kernels_is_runnable():
    hit = nwo.is_runnable_safe_work(
        _unit(
            "WU.WARN",
            resource_class="STATIC_ANALYSIS",
            frontier_id="FT.GPU_KERNELS.static-warnings",
        )
    )
    assert hit["runnable"] is True


def test_lease_seizure_command_is_not_runnable():
    hit = nwo.is_runnable_safe_work(
        _unit(
            "WU.FLOCK",
            resource_class="LIGHT_CONTROL",
            command=["flock", "/tmp/protected-accelerator-bench.lock", "true"],
        )
    )
    assert hit["runnable"] is False
    assert hit["clause"] == "SAFETY"


def test_rule_text_names_every_clause_the_tests_pin():
    rule = nwo.RUNNABLE_SAFE_WORK_RULE
    for token in (
        "GPU lease",
        "dependencies",
        "SLEEPING",
        "wake_condition",
        "STATIC_ANALYSIS",
        "resource is free",
    ):
        assert token.lower() in rule.lower() or token in rule


# ---------------------------------------------------------------------------
# Three-way verdict, both directions
# ---------------------------------------------------------------------------


def test_blocked_with_runnable_work_is_fail_not_slow():
    events = [
        _launch("WU.HASH", 0, description="recompute digests"),
        _next_work(0, ["WU.CPU"]),
        _ingest("WU.HASH", 100),
        _launch("WU.CPU", 100, dependencies=[]),
        _ingest("WU.CPU", 101),
    ]
    report = nwo.classify(events)
    assert report["verdict"] == nwo.FAIL_NO_WAIT_ORCHESTRATION, report["reason"]
    assert report["verdict"] != nwo.SLOW_BUT_CORRECT
    assert report["n_forcing_intervals"] >= 1
    first = report["forcing_intervals"][0]
    assert first["start_s"] == 0
    assert first["end_s"] == 100
    assert "WU.CPU" in first["runnable_ids"]
    assert "waiting on subprocess" in first["loop_doing"]


def test_blocked_with_nothing_runnable_is_slow_but_correct_not_fail():
    wake = "a qualified Metal GPU and Metal compiler, plus a real HCLI lease"
    events = [
        _sleep("GPU_PROTECTED", 0, wake),
        _sleep("ANE", 0, wake),
        _sleep("FPGA", 0, wake),
        _next_work(0, []),
        _launch("WU.HASH", 1, description="only remaining CPU unit"),
        _ingest("WU.HASH", 501),
    ]
    report = nwo.classify(events)
    assert report["verdict"] == nwo.SLOW_BUT_CORRECT, (
        report["reason"],
        report["forcing_intervals"],
        report["slow_intervals"],
    )
    assert report["verdict"] != nwo.FAIL_NO_WAIT_ORCHESTRATION
    assert report["n_forcing_intervals"] == 0
    assert report["n_slow_intervals"] >= 1
    slow = report["slow_intervals"][0]
    assert slow["duration_s"] == 500
    assert slow["n_runnable"] == 0


def test_gpu_queue_during_wait_does_not_become_fail():
    """A GPU-lease unit sitting in the queue is not runnable safe work."""
    events = [
        _launch("WU.HASH", 0),
        _next_work(0, ["WU.GPU"]),
        _ingest("WU.HASH", 80),
        _launch("WU.GPU", 80, resource_class="GPU_EXCLUSIVE"),
        _ingest("WU.GPU", 81),
    ]
    # The later GPU launch is not runnable-safe (no lease), so it must not force FAIL.
    report = nwo.classify(events)
    assert report["verdict"] == nwo.SLOW_BUT_CORRECT, (
        report["verdict"],
        report["reason"],
        report["forcing_intervals"],
    )


def test_handle_wait_with_runnable_ids_is_fail():
    events = [
        _launch("WU.A", 0),
        _ev(
            "HANDLE_WAIT",
            0,
            {"handle_id": "WU.A", "wait_s": 59.0, "runnable_unit_ids": ["WU.B"]},
        ),
        _ingest("WU.A", 59),
    ]
    report = nwo.classify(events)
    assert report["verdict"] == nwo.FAIL_NO_WAIT_ORCHESTRATION
    ids = report["forcing_intervals"][0]["runnable_ids"]
    assert "WU.B" in ids


def test_handle_wait_with_empty_runnable_is_slow_but_correct():
    events = [
        _launch("WU.A", 0),
        _ev("HANDLE_WAIT", 0, {"handle_id": "WU.A", "wait_s": 500.0, "runnable_unit_ids": []}),
        _ingest("WU.A", 500),
    ]
    report = nwo.classify(events)
    assert report["verdict"] == nwo.SLOW_BUT_CORRECT, report


def test_no_blocked_interval_is_pass():
    events = [
        _ev(
            "CHILD_LAUNCHED",
            0.1,
            {"unit_id": "WU.CHILD", "pid": 1},
        ),
        _ev("INDEPENDENT_STARTED", 0.2, {"unit_id": "WU.IND.0"}),
        _ev("INDEPENDENT_STARTED", 0.3, {"unit_id": "WU.IND.1"}),
        _ev("INDEPENDENT_COMPLETED", 1.0, {"unit_id": "WU.IND.0"}),
        _ev("INDEPENDENT_STARTED", 1.1, {"unit_id": "WU.IND.2"}),
        _ev("CHILD_TERMINAL", 5.0, {"unit_id": "WU.CHILD"}),
        _ev("INDEPENDENT_COMPLETED", 5.2, {"unit_id": "WU.IND.1"}),
        _ev("INDEPENDENT_COMPLETED", 5.3, {"unit_id": "WU.IND.2"}),
    ]
    report = nwo.classify(events)
    assert report["verdict"] == nwo.PASS, (report["reason"], report["forcing_intervals"])
    assert report["n_forcing_intervals"] == 0


def test_dependent_next_launch_does_not_count_as_runnable_during_wait():
    events = [
        _launch("WU.PRE", 0),
        _ingest("WU.PRE", 40),
        _launch("WU.NEXT", 40, dependencies=["WU.PRE"]),
        _ingest("WU.NEXT", 41),
    ]
    report = nwo.classify(events)
    assert report["verdict"] == nwo.SLOW_BUT_CORRECT, (
        report["verdict"],
        report["forcing_intervals"],
        report["reason"],
    )


def test_verdict_is_exactly_one_of_three():
    assert nwo.VERDICTS == (
        nwo.PASS,
        nwo.FAIL_NO_WAIT_ORCHESTRATION,
        nwo.SLOW_BUT_CORRECT,
    )
    for source in (
        [_launch("A", 0), _ingest("A", 10)],
        [_launch("A", 0), _next_work(0, ["B"]), _ingest("A", 10), _launch("B", 10)],
        [
            _ev("CHILD_LAUNCHED", 0, {"unit_id": "C"}),
            _ev("INDEPENDENT_STARTED", 0.1, {"unit_id": "I"}),
        ],
    ):
        report = nwo.classify(source)
        assert report["verdict"] in nwo.VERDICTS


# ---------------------------------------------------------------------------
# Scheduler composition
# ---------------------------------------------------------------------------


def test_independent_runnable_now_filters_gpu_out_of_scheduler_runnable():
    blocked = [{"job_id": "slow", "unit_id": "WU.SLOW"}]
    cpu = _unit("WU.CPU", resource_class="STATIC_ANALYSIS")
    gpu = _unit("WU.GPU", resource_class="GPU_EXCLUSIVE")
    view = nwo.independent_runnable_now(blocked, candidates=[cpu, gpu], world=nwo.World())
    ids = {_unit_id(u) for u in view["runnable"]}
    assert "WU.CPU" in ids
    assert "WU.GPU" not in ids
    assert view["status"] == nws.RUNNABLE
    assert view["scheduler_status_before_safe_filter"] == nws.RUNNABLE


def _unit_id(row) -> str:
    return str(row.get("id") or row.get("unit_id") or "")


def test_scheduler_blocked_when_only_dependent_candidate_remains():
    blocked = [{"job_id": "pre", "unit_id": "WU.PRE"}]
    dep = _unit("WU.NEXT", dependencies=["WU.PRE"])
    view = nwo.independent_runnable_now(blocked, candidates=[dep], world=nwo.World())
    assert view["status"] == nws.BLOCKED
    assert view["runnable"] == []
    assert view.get("named_dependency")


# ---------------------------------------------------------------------------
# Real timelines
# ---------------------------------------------------------------------------


def test_disk_timelines_are_real_receipts_not_fixtures():
    assert TIMELINE_1H.is_file(), TIMELINE_1H
    assert DETACHED.is_file(), DETACHED
    one_h = json.loads(TIMELINE_1H.read_text())
    detached = json.loads(DETACHED.read_text())
    assert one_h.get("trial") == "1h" or one_h.get("duration_s") == 3600
    assert isinstance(one_h.get("events"), list) and len(one_h["events"]) > 100
    assert detached.get("schema") == "hawking.future.detached_trial.v1"
    assert detached.get("fixture") is False
    assert isinstance(detached.get("timeline"), list) and len(detached["timeline"]) > 10


def test_1h_timeline_is_fail_no_wait_orchestration():
    report = nwo.classify(TIMELINE_1H)
    assert report["verdict"] == nwo.FAIL_NO_WAIT_ORCHESTRATION, report["reason"]
    assert report["fixtures"] is False
    assert report["n_forcing_intervals"] >= 1
    forcing = report["forcing_intervals"]
    # The 491s hashing wait (t_s 202 -> 693) is on this timeline.
    hashing = [
        row
        for row in forcing
        if abs(float(row["start_s"]) - 202) < 1e-6 and abs(float(row["end_s"]) - 693) < 1e-6
    ]
    assert hashing, (
        "expected the 202->693s subprocess wait as a forcing interval, got "
        + json.dumps(
            [{"start_s": r["start_s"], "end_s": r["end_s"], "waited_unit": r["waited_unit"],
              "n_runnable": r["n_runnable"]} for r in forcing[:12]],
            indent=2,
        )
    )
    row = hashing[0]
    assert row["duration_s"] == 491
    assert row["n_runnable"] >= 1
    assert "waiting on subprocess" in row["loop_doing"]
    assert row["waited_unit"] == "WU.AUTONOMY.specimen_verify.139"
    assert all("start_s" in r and "end_s" in r and "loop_doing" in r and "runnable" in r for r in forcing)


def test_1h_tail_adjudicated_from_timeline_not_intuition():
    report = nwo.classify(TIMELINE_1H)
    tail = report["tail"]
    assert tail["applies"] is True
    assert tail["named_duration_s"] == nwo.TAIL_S
    assert abs(float(tail["end_s"]) - float(report["elapsed_s"])) < 1e-6
    assert abs(float(tail["start_s"]) - (float(report["elapsed_s"]) - nwo.TAIL_S)) < 1e-6
    assert tail["verdict"] in nwo.VERDICTS
    # Parked ANE/FPGA/GPU_PROTECTED at t=0 must not be counted runnable.
    parked = tail["parked_sleeping_not_runnable"]
    assert {p["resource_class"] for p in parked} >= {"ANE", "FPGA", "GPU_PROTECTED"}
    assert all(p["counted_runnable"] is False for p in parked)
    leftover = tail["leftover_at_end"]
    assert leftover is not None
    assert leftover["n"] == 32
    assert leftover["note"] and "never because work ran out" in leftover["note"]
    # The tail is an I/O-bound loop: no refills inside it.
    assert tail["n_refills_in_tail"] == 0
    assert tail["last_refill_t_s"] is not None
    assert tail["last_refill_t_s"] < tail["start_s"]
    # Honest adjudication: remaining independent verifies + leftover CPU-safe
    # frontier sat while the loop waited. That is FAIL, not SLOW_BUT_CORRECT.
    assert tail["verdict"] == nwo.FAIL_NO_WAIT_ORCHESTRATION, (
        tail["why"],
        tail["n_forcing_intervals"],
        (tail["forcing_intervals"] or [{}])[0] if tail["forcing_intervals"] else None,
    )
    assert tail["not"] == nwo.SLOW_BUT_CORRECT
    assert tail["n_forcing_intervals"] >= 1
    last = tail["forcing_intervals"][-1]
    assert last["n_runnable"] >= 1
    # Leftover CPU-safe frontier ids must appear on some tail forcing interval
    # (the last wait reports them at t_s 3629).
    leftover_ids = set(leftover["ids"] or [])
    named = {i for row in tail["forcing_intervals"] for i in row["runnable_ids"]}
    assert leftover_ids & named or any(
        r["n_runnable"] >= 1 for r in tail["forcing_intervals"]
    )


def test_detached_timeline_is_pass():
    report = nwo.classify(DETACHED)
    assert report["verdict"] == nwo.PASS, (
        report["reason"],
        report["n_forcing_intervals"],
        report["forcing_intervals"][:4],
        report["n_slow_intervals"],
    )
    assert report["n_forcing_intervals"] == 0
    # Detached launches are not a blocked loop; independent work ran.
    detached_doc = json.loads(DETACHED.read_text())
    assert detached_doc["idle_runnable_seconds"] == 0.0
    assert detached_doc["safe_in_flight_bound"] == 2
    assert (detached_doc.get("child") or {}).get("pid")


def test_power_torture_timeline_absent_is_reported_not_invented():
    replay = nwo.replay_disk_timelines()
    row = replay["power_torture_timeline"]
    if POWER_TORTURE_TIMELINE.is_file():
        assert row["present"] is True
        assert row["verdict"] in nwo.VERDICTS
    else:
        assert row["present"] is False
        assert row["verdict"] is None
        assert "POWER_TORTURE_TIMELINE" in row["reason"]
        assert row["path"] == nwo.POWER_TORTURE_TIMELINE_REL


def test_replay_disk_timelines_are_not_fixtures():
    replay = nwo.replay_disk_timelines()
    assert replay["fixtures"] is False
    assert replay["autonomy_1h"]["verdict"] == nwo.FAIL_NO_WAIT_ORCHESTRATION
    assert replay["detached_work_trial"]["verdict"] == nwo.PASS


# ---------------------------------------------------------------------------
# Improvement-trial public API (does not edit the trial file)
# ---------------------------------------------------------------------------


def test_apply_to_trial_verdict_blocks_pass_when_fail_no_wait_holds():
    judged = {"verdict": "PASS", "unmet": [], "conditions": [], "reason": "would pass"}
    events = [
        _launch("WU.HASH", 0),
        _ev(
            "HANDLE_WAIT",
            0,
            {"handle_id": "WU.HASH", "wait_s": 12.0, "runnable_unit_ids": ["WU.OTHER"]},
        ),
        _ingest("WU.HASH", 12),
    ]
    gated = nwo.apply_to_trial_verdict(judged, events)
    assert gated["verdict"] != "PASS"
    assert gated["verdict"] == "FAIL"
    assert gated["failed_on_no_wait_orchestration"] is True
    assert "no_wait_orchestration" in gated["unmet"]
    assert gated["no_wait_orchestration"]["verdict"] == nwo.FAIL_NO_WAIT_ORCHESTRATION


def test_apply_to_trial_verdict_does_not_fail_slow_but_correct():
    judged = {"verdict": "PASS", "unmet": [], "conditions": [], "reason": "would pass"}
    events = [
        _launch("WU.HASH", 0),
        _ev("HANDLE_WAIT", 0, {"handle_id": "WU.HASH", "wait_s": 500.0, "runnable_unit_ids": []}),
        _ingest("WU.HASH", 500),
    ]
    gated = nwo.apply_to_trial_verdict(judged, events)
    assert gated["no_wait_orchestration"]["verdict"] == nwo.SLOW_BUT_CORRECT
    assert gated["failed_on_no_wait_orchestration"] is False
    assert gated["verdict"] == "PASS"
    assert "no_wait_orchestration" not in gated["unmet"]


def _sneaky_passing_record() -> it.TrialRecord:
    """A record the unwrapped judge PASSes, but this obligation FAILs.

    HANDLE_WAIT of 59s is below improvement_trial's 477s / majority-window
    lines, so eval_no_open_handle_wait does not fire. The architecture
    failure is still present: runnable work existed while the loop waited.
    """
    record = it.passing_skeleton()
    last = record.events[-1] if record.events else {}
    t_s = float(last.get("t_s") or 0.0)
    seq = int(last.get("seq") or 0) + 1
    record.events = list(record.events) + [
        {
            "seq": seq,
            "t_s": t_s,
            "kind": "HANDLE_WAIT",
            "payload": {
                "handle_id": "WU.IMPROVEMENT.sneaky",
                "wait_s": 59.0,
                "runnable_unit_ids": ["group_size_1024", "deltanet_organ_vs_isolated_kernel"],
            },
        }
    ]
    return record


def test_sneaky_passing_skeleton_is_pass_for_unwrapped_judge_and_fail_for_the_gate():
    record = _sneaky_passing_record()
    original = getattr(it.judge, "_no_wait_original", None)
    if original is None:
        # Auto-install wraps on import. Peek through the wrapper.
        original = getattr(it.judge, "_no_wait_original", it.judge)
    unwrapped = original(record) if original is not it.judge or not getattr(it.judge, "_no_wait_orchestration_wrapped", False) else original(record)
    # The unwrapped judge must still PASS this record — otherwise the 59s
    # wait leaked into another guard and the distinction is untestable.
    assert unwrapped["verdict"] == "PASS", (unwrapped["unmet"], unwrapped["reason"])
    gated = nwo.judge_improvement_trial(record)
    assert gated["verdict"] == "FAIL"
    assert gated["failed_on_no_wait_orchestration"] is True
    assert gated["no_wait_orchestration"]["verdict"] == nwo.FAIL_NO_WAIT_ORCHESTRATION


def test_passing_skeleton_still_passes_through_the_gate():
    record = it.passing_skeleton()
    gated = nwo.judge_improvement_trial(record)
    assert gated["verdict"] == "PASS", (gated["unmet"], gated["reason"], gated.get("no_wait_orchestration"))
    assert gated["failed_on_no_wait_orchestration"] is False
    assert gated["no_wait_orchestration"]["verdict"] in {nwo.PASS, nwo.SLOW_BUT_CORRECT}


def test_open_handle_477s_control_is_fail_no_wait_orchestration():
    record = it.CONTROL_FACTORIES["open_handle_wait"]()
    report = nwo.classify(record)
    assert report["verdict"] == nwo.FAIL_NO_WAIT_ORCHESTRATION, report["reason"]
    waits = [e for e in record.events if e.get("kind") == "HANDLE_WAIT"]
    assert waits
    assert float((waits[0].get("payload") or {}).get("wait_s") or 0) >= 477
    assert (waits[0].get("payload") or {}).get("runnable_unit_ids")
    gated = nwo.judge_improvement_trial(record)
    assert gated["verdict"] == "FAIL"
    assert gated["failed_on_no_wait_orchestration"] is True


def test_install_wraps_improvement_trial_judge_without_editing_the_file():
    src = Path(it.__file__).read_text(encoding="utf-8")
    assert "no_wait_orchestration" not in src
    nwo.install_into_improvement_trial()
    assert getattr(it.judge, "_no_wait_orchestration_wrapped", False) is True
    record = _sneaky_passing_record()
    judged = it.judge(record)
    assert judged["verdict"] == "FAIL"
    assert judged["failed_on_no_wait_orchestration"] is True


def test_measure_is_classify():
    events = [_launch("A", 0), _ingest("A", 3)]
    assert nwo.measure(events)["verdict"] == nwo.classify(events)["verdict"]


def test_eval_no_wait_orchestration_matches_trial_condition_shape():
    row = nwo.eval_no_wait_orchestration(it.passing_skeleton())
    assert set(row) >= {"id", "met", "detail", "cites"}
    assert row["id"] == "no_wait_orchestration"
    assert row["met"] is True
    bad = nwo.eval_no_wait_orchestration(_sneaky_passing_record())
    assert bad["met"] is False
    assert bad["verdict"] == nwo.FAIL_NO_WAIT_ORCHESTRATION


# ---------------------------------------------------------------------------
# Receipt / entry point
# ---------------------------------------------------------------------------


def test_entry_point_seals_receipt_with_three_way_verdict_and_real_timelines():
    path = nwo.build()
    assert path.parent == RECEIPTS
    assert path.name == nwo.RECEIPT
    doc = json.loads(path.read_text())
    assert doc["schema"] == nwo.SCHEMA
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["seal_sha256"]
    assert doc["three_way_verdict"] == list(nwo.VERDICTS)
    assert "RESOURCE" in doc["runnable_safe_work_rule"] or "resource" in doc["runnable_safe_work_rule"]
    assert doc["replay"]["fixtures"] is False
    assert doc["replay"]["autonomy_1h"]["verdict"] == nwo.FAIL_NO_WAIT_ORCHESTRATION
    assert doc["replay"]["detached_work_trial"]["verdict"] == nwo.PASS
    tail = doc["one_h_tail"]
    assert tail["applies"] is True
    assert tail["verdict"] == nwo.FAIL_NO_WAIT_ORCHESTRATION
    assert tail["named_duration_s"] == 2864.0
    forcing = doc["replay"]["autonomy_1h"]["forcing_intervals"]
    assert forcing
    assert all("start_s" in r and "end_s" in r and "loop_doing" in r and "runnable" in r for r in forcing)
    cited = doc["detached_measured"]["cited_from_receipt"]
    assert cited["copied_as_strings_not_remeasured"] is True
    assert cited["cited_idle_runnable_seconds"] == "0.0"
    assert cited["cited_safe_in_flight_bound"] == "2"
    assert doc["api_for_callers"]["classify"]
    assert doc["api_for_callers"]["apply_to_trial_verdict"]
    assert "tools/future/improvement_trial.py" in doc["does_not_edit"]
    assert "tools/future/no_wait_scheduler.py" in doc["does_not_edit"]
    assert doc["verdict"] == "PASS"
    _assert_no_hardware_claims(doc)


def test_does_not_claim_to_edit_the_composed_modules():
    src = Path(nwo.__file__).read_text(encoding="utf-8")
    for path in (
        "tools/future/no_wait_scheduler.py",
        "tools/future/detached_trial.py",
        "tools/future/autonomy_degeneracy.py",
        "tools/future/improvement_trial.py",
    ):
        assert path in src
    doc = json.loads(nwo.build().read_text())
    for path in (
        "tools/future/no_wait_scheduler.py",
        "tools/future/detached_trial.py",
        "tools/future/autonomy_degeneracy.py",
        "tools/future/improvement_trial.py",
    ):
        assert path in doc["does_not_edit"]


def test_a_launch_claiming_detached_without_proof_is_still_a_sync_wait():
    """The payload is the driver's own word for it. detached_started is evidence.

    emit_detached_started refuses to write unless the job has a live pid, so a
    matching detached_started is proof the loop did not block. Accepting the
    payload claim alone would let a driver relabel its way out of every forcing
    interval.
    """
    events = [
        {"kind": "workunit_launched", "t_s": 0.0,
         "payload": {"unit": {"id": "WU.A"}, "launch": "detached"}},
        # no detached_started for WU.A
        {"kind": "result_ingested", "t_s": 9.0, "payload": {"unit_id": "WU.A"}},
    ]
    assert nwo._verified_detached_ids(events) == set()


def test_a_launch_with_a_matching_detached_started_is_proven_non_blocking():
    events = [
        {"kind": "workunit_launched", "t_s": 0.0,
         "payload": {"unit": {"id": "WU.A"}, "launch": "detached"}},
        {"kind": "detached_started", "t_s": 0.1, "payload": {"unit_id": "WU.A"}},
        {"kind": "result_ingested", "t_s": 9.0, "payload": {"unit_id": "WU.A"}},
    ]
    assert nwo._verified_detached_ids(events) == {"WU.A"}


def test_an_unmarked_launch_is_never_treated_as_detached():
    events = [
        {"kind": "workunit_launched", "t_s": 0.0, "payload": {"unit": {"id": "WU.B"}}},
        {"kind": "detached_started", "t_s": 0.1, "payload": {"unit_id": "WU.B"}},
    ]
    assert nwo._verified_detached_ids(events) == set()

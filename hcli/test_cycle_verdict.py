"""A run verdict has to be able to go green, and has to refuse to.

"The daemon is up" was true for 553 minutes while `accepted` stayed 0. Every
criterion here is a defect that actually happened, so each test builds the
defect and checks the verdict names it.

Two failure modes this file guards against in the verdict ITSELF:

* an UNREACHABLE bar -- counting receipts from before the fix that resolved them
  meant `structured_output_ok` could never clear, and a criterion that can never
  pass trains the reader to ignore the whole report;
* UNKNOWN quietly rounding to PASS -- a criterion with no evidence has not been
  satisfied, it has not been checked.
"""
from __future__ import annotations

import json
import time

import pytest

from hcli.cycle_verdict import FAIL, PASS, UNKNOWN, evaluate, render


def _workspace(tmp_path, *, mission=None, daemon=None, dag=None, log=None, events=None):
    hcli = tmp_path / ".hcli"
    (hcli / "mission").mkdir(parents=True)
    (hcli / "resident").mkdir(parents=True)
    if mission is not None:
        (hcli / "mission" / "state.json").write_text(json.dumps(mission))
    if daemon is not None:
        (hcli / "resident" / "state.json").write_text(json.dumps(daemon))
    if dag is not None:
        (hcli / "dag.json").write_text(json.dumps(dag))
    if log:
        (hcli / "mission" / "mission.log").write_text(
            "\n".join(json.dumps(r) for r in log)
        )
    if events:
        (hcli / "mission" / "events.jsonl").write_text(
            "\n".join(json.dumps(r) for r in events)
        )
    return tmp_path


def _healthy_mission(**over):
    base = {
        "id": "m1",
        "phase": "running",
        "accepted_count": 1,
        "started_at": time.time() - 60,
        "units": {"u1": {"status": "completed"}},
    }
    base.update(over)
    return base


def test_a_fully_good_run_has_no_failing_criterion(tmp_path):
    """Note what a PASS requires: a clean receipt FROM THIS MISSION.

    Without one, structured_output_ok is UNKNOWN and the run verdict is
    UNKNOWN -- not PASS. That is the point of the criterion.
    """
    ws = _workspace(
        tmp_path,
        mission=_healthy_mission(),
        daemon={"failure_streak": 0, "config": {"max_restarts": 3}},
        dag={"repair_counts": {}},
        log=[{"event": "dispatch"}],
        events=[
            {"type": "tool_invoked", "data": {"tool": "fs.read", "ok": True}},
            {"type": "model_call_finished", "data": {"goal_id": "g", "ok": True}},
            {"type": "tool_call_finished", "data": {"tool": "fs.read", "elapsed_s": 0.003}},
        ],
    )
    receipts = tmp_path / ".hcli" / "receipts"
    receipts.mkdir()
    (receipts / "clean.json").write_text(json.dumps({
        "structured_output": {"exhausted": False, "attempts": 1},
        # Fast AND reusing: green means both.
        "model_calls": [
            {"prompt_tokens": 3000, "wall_s": 2.0, "prefix_reused_tokens": 2400},
        ],
    }))
    report = evaluate(ws)
    assert not report["failed"], report["criteria"]
    assert report["measured"]["effective_prompt_tps"] >= 100
    assert report["measured"]["realized_reuse_fraction"] >= 0.5
    # One model call per goal, so there is no prior turn to reuse from and
    # kv_reuse is NOT APPLICABLE rather than green. The run verdict is UNKNOWN,
    # which is the honest answer: nothing here was checked and failed.
    assert report["criteria"]["kv_reuse"]["verdict"] == UNKNOWN
    assert "no prior turn" in report["criteria"]["kv_reuse"]["detail"]


def test_alive_but_accepting_nothing_is_a_FAIL(tmp_path):
    """The 553-minute case. Everything else green, nothing done."""
    ws = _workspace(
        tmp_path,
        mission=_healthy_mission(accepted_count=0, units={"u1": {"status": "ready"}}),
        daemon={"failure_streak": 0, "config": {"max_restarts": 3}},
        dag={"repair_counts": {}},
        log=[{"event": "dispatch"}],
        events=[{"type": "tool_invoked", "data": {"ok": True}}],
    )
    report = evaluate(ws)
    assert report["verdict"] == FAIL
    assert "progress" in report["failed"]
    assert "alive is not progress" in report["criteria"]["progress"]["detail"]


def test_a_cancelled_mission_is_named_as_needing_a_human(tmp_path):
    ws = _workspace(tmp_path, mission=_healthy_mission(phase="cancelled"))
    report = evaluate(ws)
    assert report["criteria"]["mission_advanceable"]["verdict"] == FAIL
    assert "needs a human" in report["criteria"]["mission_advanceable"]["detail"]


def test_an_inherited_repair_budget_is_caught(tmp_path):
    """The defect that kept accepted at 0 across missions."""
    ws = _workspace(
        tmp_path,
        mission=_healthy_mission(accepted_count=0, units={}),
        dag={"repair_counts": {"G001.work": 1, "G002.work": 1}},
    )
    report = evaluate(ws)
    row = report["criteria"]["repair_budget_unspent"]
    assert row["verdict"] == FAIL
    assert "depth 0" in row["detail"]


def test_a_depth_zero_repair_refusal_is_caught(tmp_path):
    ws = _workspace(
        tmp_path,
        mission=_healthy_mission(),
        log=[{"event": "repair_exhausted", "id": "G001.work", "depth": 0}],
    )
    row = evaluate(ws)["criteria"]["repair_reached_the_unit"]
    assert row["verdict"] == FAIL
    assert "G001.work" in row["detail"]


def test_a_tool_loop_is_caught(tmp_path):
    events = [{"type": "tool_invoked", "data": {"ok": False}} for _ in range(4)]
    events += [{"type": "tool_call_repeated", "data": {}} for _ in range(3)]
    ws = _workspace(tmp_path, mission=_healthy_mission(), events=events)
    row = evaluate(ws)["criteria"]["no_tool_loops"]
    assert row["verdict"] == FAIL
    assert "3 of 7 requested calls were duplicates" in row["detail"]


def test_old_receipts_do_not_hold_the_bar_down(tmp_path):
    """The unreachable-bar guard.

    A receipt written BEFORE this mission started is not this mission's
    failure. Counting it made `structured_output_ok` permanently red.
    """
    ws = _workspace(tmp_path, mission=_healthy_mission())
    receipts = tmp_path / ".hcli" / "receipts"
    receipts.mkdir()
    stale = receipts / "old.json"
    stale.write_text(json.dumps({"structured_output": {"exhausted": True}}))
    import os

    started = json.loads((tmp_path / ".hcli" / "mission" / "state.json").read_text())[
        "started_at"
    ]
    os.utime(stale, (started - 3600, started - 3600))

    row = evaluate(ws)["criteria"]["structured_output_ok"]
    assert row["verdict"] != FAIL, "a pre-mission receipt must not hold the bar down"


def test_a_receipt_from_THIS_mission_does_hold_the_bar(tmp_path):
    """Negative control for the one above: scoping is not silencing."""
    ws = _workspace(tmp_path, mission=_healthy_mission())
    receipts = tmp_path / ".hcli" / "receipts"
    receipts.mkdir()
    (receipts / "now.json").write_text(
        json.dumps({"structured_output": {"exhausted": True}})
    )
    row = evaluate(ws)["criteria"]["structured_output_ok"]
    assert row["verdict"] == FAIL
    assert "exhausted" in row["detail"]


def test_unknown_never_rounds_to_pass(tmp_path):
    ws = _workspace(tmp_path)  # nothing on disk at all
    report = evaluate(ws)
    assert report["verdict"] == UNKNOWN
    assert report["verdict"] != PASS
    assert "has not been checked" in report["note"]


def test_a_burning_restart_budget_is_caught(tmp_path):
    ws = _workspace(
        tmp_path,
        mission=_healthy_mission(),
        daemon={"failure_streak": 2, "config": {"max_restarts": 3}},
    )
    row = evaluate(ws)["criteria"]["worker_stable"]
    assert row["verdict"] == FAIL
    assert "2 of 3" in row["detail"]


def test_render_puts_the_verdict_first(tmp_path):
    ws = _workspace(tmp_path, mission=_healthy_mission())
    text = render(evaluate(ws))
    assert text.splitlines()[0].startswith("RUN VERDICT:")


def test_a_repair_this_mission_EARNED_is_not_called_inherited(tmp_path):
    """The false-FAIL guard.

    A unit that failed and legitimately got a repair carries a count. Reading
    that as an inherited budget fires the moment the repair machinery works,
    which is the unreachable bar again in a different hat.
    """
    ws = _workspace(
        tmp_path,
        mission=_healthy_mission(
            accepted_count=0,
            units={
                "G001.work": {"status": "failed"},
                "G001.work.repair.1": {"status": "ready", "repairs": "G001.work"},
            },
        ),
        dag={"repair_counts": {"G001.work": 1}},
    )
    row = evaluate(ws)["criteria"]["repair_budget_unspent"]
    assert row["verdict"] == PASS, row["detail"]
    assert "earned by this mission" in row["detail"]


def test_a_count_with_no_repair_unit_is_still_caught(tmp_path):
    """Negative control: separating earned from inherited is not silencing."""
    ws = _workspace(
        tmp_path,
        mission=_healthy_mission(accepted_count=0, units={"G001.work": {"status": "ready"}}),
        dag={"repair_counts": {"G001.work": 1}},
    )
    row = evaluate(ws)["criteria"]["repair_budget_unspent"]
    assert row["verdict"] == FAIL
    assert "no repair unit in this mission" in row["detail"]


def test_a_slow_tool_is_a_FAIL_even_when_it_works(tmp_path):
    """`fs.list` took 28.1 s and returned successfully. Working is not the bar."""
    ws = _workspace(
        tmp_path,
        mission=_healthy_mission(),
        events=[{"type": "tool_call_finished", "data": {"tool": "fs.list", "elapsed_s": 28.1}}],
    )
    row = evaluate(ws)["criteria"]["tools_fast"]
    assert row["verdict"] == FAIL
    assert "28100" in row["detail"] or "28100.0" in row["detail"]


def test_budgets_are_named_and_overridable(tmp_path):
    """A budget buried in a comparison is one nobody can find or revise."""
    from hcli.cycle_verdict import DEFAULT_BUDGETS

    assert set(DEFAULT_BUDGETS) >= {
        "tool_p95_ms",
        "mean_rounds_per_goal",
        "effective_prompt_tps",
        "realized_reuse_fraction",
    }
    ws = _workspace(
        tmp_path,
        mission=_healthy_mission(),
        events=[{"type": "tool_call_finished", "data": {"tool": "x", "elapsed_s": 0.2}}],
    )
    assert evaluate(ws)["criteria"]["tools_fast"]["verdict"] == FAIL
    assert evaluate(ws, {"tool_p95_ms": 500.0})["criteria"]["tools_fast"]["verdict"] == PASS


def test_unmeasured_kv_reuse_is_not_reported_as_zero(tmp_path):
    ws = _workspace(tmp_path, mission=_healthy_mission())
    row = evaluate(ws)["criteria"]["kv_reuse"]
    assert row["verdict"] == UNKNOWN
    assert "NOT zero" in row["detail"]

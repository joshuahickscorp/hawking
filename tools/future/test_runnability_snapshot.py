"""A count taken at the wait, not the driver's claim about it.

Two 30m timelines emit the identical pre-gap signal - exhausted True, n 0, ids
[] - and mean opposite things. In the archived 477 s case the claim was FALSE:
twelve frontiers held novel work the driver had missed. So a judge keyed on
`exhausted` acquits the original defect, which the negative control proved when
I tried exactly that.
"""
from __future__ import annotations

import pytest

from tools.future import runnability_snapshot as rs


def _f(fid, entries):
    return {"id": fid, "entry_ids": entries}


def test_a_snapshot_that_did_not_look_is_refused():
    with pytest.raises(rs.SnapshotRefused, match="did not look"):
        rs.snapshot(None, already_launched=[], scar_dead=[], t_s=1.0)


def test_runnable_excludes_scar_dead_and_already_launched():
    snap = rs.snapshot(
        [_f("FT.A", ["a1", "a2", "a3"]), _f("FT.B", ["b1"])],
        already_launched=["a1"],
        scar_dead=["a2"],
        t_s=81.0,
    )
    assert snap["n_runnable"] == 2  # a3 and b1
    by = {r["frontier_id"]: r for r in snap["runnable_by_frontier"]}
    assert by["FT.A"]["n_entries"] == 3
    assert by["FT.A"]["n_after_scars"] == 2
    assert by["FT.A"]["n_not_yet_launched"] == 1


def test_a_wait_with_nothing_runnable_is_justified():
    snap = rs.snapshot([_f("FT.A", ["a1"])], already_launched=["a1"],
                       scar_dead=[], t_s=81.0)
    got = rs.classify_wait(snap, waiting_on=["job-1"])
    assert got["verdict"] == rs.WAIT_JUSTIFIED
    assert got["n_runnable"] == 0
    assert "not idling" in got["why"]


def test_a_wait_with_runnable_work_is_the_failure_and_names_it():
    snap = rs.snapshot([_f("FT.A", ["a1", "a2"]), _f("FT.B", ["b1"])],
                       already_launched=["a1"], scar_dead=[], t_s=81.0)
    got = rs.classify_wait(snap, waiting_on=["job-1"])
    assert got["verdict"] == rs.IDLE_WITH_RUNNABLE_WORK
    assert got["n_runnable"] == 2
    fids = {r["frontier_id"] for r in got["runnable_by_frontier"]}
    assert fids == {"FT.A", "FT.B"}, "the failure must name which frontiers held work"


def test_a_wait_with_no_open_handle_is_not_a_wait():
    snap = rs.snapshot([_f("FT.A", ["a1"])], already_launched=[], scar_dead=[], t_s=1.0)
    got = rs.classify_wait(snap, waiting_on=[])
    assert "not a wait; it is an end" in got["why"]


def test_a_missing_snapshot_is_unevidenced_not_passed():
    """The state both 30m timelines are in. Neither convicted nor acquitted."""
    got = rs.classify_wait({}, waiting_on=["job-1"])
    assert got["verdict"] == rs.UNEVIDENCED
    assert "cannot say" in got["why"]


def test_the_receipt_refuses_to_retrofit_a_verdict_onto_old_timelines():
    cb = rs.build()["claim_boundary"]
    assert "does not retrofit" in cb
    assert "UNEVIDENCED" in cb


def test_the_two_timelines_are_recorded_as_indistinguishable():
    d = rs.why_the_two_timelines_are_indistinguishable()
    assert "exhausted True, n 0, ids []" in d["archived_477s"]["signal"]
    assert "exhausted True, n 0, ids []" in d["run_three_482s"]["signal"]
    assert "FALSE" in d["archived_477s"]["truth"]
    assert "acquits the archived defect" in d["conclusion"]


def test_the_driver_takes_a_snapshot_at_the_wait():
    """An instrument nobody calls is not an instrument.

    The wait path emits runnability_snapshot BEFORE emit_idle_justified and
    passes the runnable ids into it, so the refusal and the evidence agree.
    """
    from tools.future import autonomy_run as ar

    src = open(ar.__file__, encoding="utf-8").read()
    snap_at = src.index('doc = _emit(doc, "runnability_snapshot", snap, t_s=t())')
    idle_at = src.index("doc = emit_idle_justified(")
    assert snap_at < idle_at, "the snapshot must be taken before the wait is justified"
    between = src[snap_at:idle_at]
    assert len(between) < 400, "the snapshot and the justification must stay adjacent"
    assert "runnable_ids=[" in src[idle_at:idle_at + 500]


def test_a_snapshot_failure_never_kills_a_run():
    """Evidence is worth having; it is not worth losing thirty minutes over."""
    from tools.future import autonomy_run as ar

    src = open(ar.__file__, encoding="utf-8").read()
    i = src.index('doc = _emit(doc, "runnability_snapshot", snap, t_s=t())')
    guard = src[max(0, i - 600):i]
    assert "a snapshot must never kill a run" in guard
    assert "except Exception" in guard

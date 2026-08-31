"""The 477-second idle must be structurally impossible, not merely detected.

G037: "the driver must consult refill before sleeping on open handles and must
justify any idle it does take; the judge needs a no_idle_while_work_exists
condition whose negative control is the archived 30m timeline itself."

The archived 30m run drained its queue at t=88 and did nothing until t=565, then
ended with next_work_left naming TWELVE frontiers that still held novel work.
eval_never_conversational_wait passed it because that evaluator matches
conversational phrases and event kinds and NEVER an idle interval.

These pin the DRIVER half. emit_idle_justified already refuses the four ways a
wait can be dishonest; what nothing pinned is that the wait loop RE-ASKS rather
than justifying once and sleeping through a frontier that later fills - which is
exactly the shape of a 477-second hole.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.future.autonomy_run import EmitRefused, emit_idle_justified  # noqa: E402


def _doc():
    return {"events": []}


def test_waiting_while_novel_work_exists_is_refused():
    """The core of the obligation: novel work means take it, never wait."""
    with pytest.raises(EmitRefused, match="refill returned novel work"):
        emit_idle_justified(
            _doc(),
            asked=[{"frontier_id": "F.A", "returned": "novel"}],
            waiting_on=[{"job_id": "j1", "pid": 1}],
            t_s=88,
        )


def test_a_wait_with_no_open_handle_is_an_end_not_a_wait():
    with pytest.raises(EmitRefused, match="not a wait; it is an end"):
        emit_idle_justified(_doc(), asked=[{"frontier_id": "F.A"}], waiting_on=[], t_s=88)


def test_idle_without_asking_the_frontier_is_refused():
    """A 477-second gap with no survey is the failure this exists to prevent."""
    with pytest.raises(EmitRefused, match="frontiers were not asked"):
        emit_idle_justified(_doc(), asked=None, waiting_on=[{"job_id": "j1"}], t_s=88)


def test_a_dry_survey_with_an_open_handle_is_a_legitimate_wait():
    doc = emit_idle_justified(
        _doc(),
        asked=[{"frontier_id": "F.A"}, {"frontier_id": "F.B"}],
        waiting_on=[{"job_id": "j1", "pid": 4242}],
        t_s=88,
    )
    ev = doc["events"][-1]
    payload = ev.get("payload") or ev
    assert payload["n_novel"] == 0
    assert payload["n_asked"] == 2
    assert payload["frontiers_asked"] == ["F.A", "F.B"]
    assert payload["waiting_on"][0]["pid"] == 4242


def test_the_survey_is_carried_not_summarised():
    """next_work_left named twelve frontiers holding novel work and the judge
    could not see them. The justification must carry the frontiers it asked, by
    id, so a later reader can check the claim rather than trust it."""
    asked = [{"frontier_id": f"F.{i}"} for i in range(12)]
    doc = emit_idle_justified(
        _doc(), asked=asked, waiting_on=[{"job_id": "j1"}], t_s=88
    )
    payload = doc["events"][-1].get("payload") or doc["events"][-1]
    assert payload["frontiers_asked"] == [f"F.{i}" for i in range(12)]
    assert len(payload["returned"]) == 12


def test_one_novel_frontier_among_many_still_refuses():
    """Eleven dry frontiers do not license sleeping through the twelfth."""
    asked = [{"frontier_id": f"F.{i}"} for i in range(11)]
    asked.append({"frontier_id": "F.11", "returned": "novel"})
    with pytest.raises(EmitRefused, match="refill returned novel work"):
        emit_idle_justified(_doc(), asked=asked, waiting_on=[{"job_id": "j1"}], t_s=88)


# ---------------------------------------------------------------------------
# NOVEL is not the same question as RUNNABLE.
#
# The 30m frozen trial surveyed all 32 frontiers, got n_novel 0, and this
# function accepted the wait - so the trial's own verify() scored
# no_idle_while_work_exists as PASS. no_wait_orchestration.classify read the SAME
# timeline as FAIL with 17 forcing intervals, the first a 1s inline wait on
# WU.AUTONOMY.negative_index.45 while ngram_school and status_causality were
# RUNNABLE. ngram_school was IN that survey. Refill said "nothing novel"; the
# frontier still held work that could run.
# ---------------------------------------------------------------------------


def test_a_dry_refill_does_not_license_sleeping_through_runnable_work():
    """The exact 30m disagreement, as a test."""
    with pytest.raises(EmitRefused, match="RUNNABLE"):
        emit_idle_justified(
            _doc(),
            asked=[{"frontier_id": "FT.MODEL_REPRESENTATION.ngram-school"}],
            waiting_on=[{"job_id": "WU.AUTONOMY.negative_index.45"}],
            t_s=490,
            runnable_ids=["ngram_school", "status_causality"],
        )


def test_the_refusal_names_what_was_runnable():
    """A refusal that does not say WHAT could have run is not actionable."""
    with pytest.raises(EmitRefused, match="ngram_school"):
        emit_idle_justified(
            _doc(),
            asked=[{"frontier_id": "F.A"}],
            waiting_on=[{"job_id": "j1"}],
            t_s=490,
            runnable_ids=["ngram_school"],
        )


def test_nothing_runnable_is_still_a_legitimate_wait():
    """The stricter question must not make every wait illegal."""
    doc = emit_idle_justified(
        _doc(),
        asked=[{"frontier_id": "F.A"}],
        waiting_on=[{"job_id": "j1", "pid": 7}],
        t_s=490,
        runnable_ids=[],
    )
    payload = doc["events"][-1].get("payload") or doc["events"][-1]
    assert payload["n_runnable"] == 0
    assert payload["runnable_checked"] is True


def test_an_unchecked_runnable_set_is_recorded_as_unchecked():
    """Callers that cannot supply the set must not look like callers that
    supplied an empty one - silence and zero are different claims."""
    doc = emit_idle_justified(
        _doc(), asked=[{"frontier_id": "F.A"}], waiting_on=[{"job_id": "j1"}], t_s=490
    )
    payload = doc["events"][-1].get("payload") or doc["events"][-1]
    assert payload["runnable_checked"] is False

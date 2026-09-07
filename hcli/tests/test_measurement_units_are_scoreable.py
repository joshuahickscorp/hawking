"""A round that MEASURES could never be accepted, and its receipt said "evidence: none".

The campaign asks HCLI to be a scientist: read the ledger, choose a specimen, measure
it, report. Four autonomous rounds have now run. Every one was recorded

    validation = {"ok": True, "kind": "read_only", "evidence": "none",
                  "accepted_work": False}

hardcoded on the answer path, no matter what the round actually did. Round 4 made
SEVEN model calls and its prompt grew from 5,015 to 5,651 tokens between calls, which
only happens when tool observations are accumulating -- and the receipt still said
evidence: none. The verifier can see mutations and nothing else, so a measurement
WorkUnit is unscoreable BY CONSTRUCTION and G006's streak could never leave zero.

Two things must both stay true, and they pull in opposite directions:

  - `evidence: "none"` is FALSE when the engine ran tools and got output back. A tool
    observation is machine-produced, not a model claim.
  - The scar that hardcoded line was written for is real: five autonomous rounds
    returned answers claiming a function was "already present ... the test passes as
    expected", the name existed nowhere, and all five were recorded ok=True. An
    ANSWER must never become accepted MUTATION work.

So: record what actually happened, keep `is_accepted_work` exactly as strict as it
is, and add a SEPARATE predicate for measurement. Nothing here weakens the mutation
gate; the tests below assert that directly.
"""
from __future__ import annotations

import pytest

from hcli.engine import Engine, is_accepted_work, is_accepted_measurement


OK_OBS = [{"tool": "odyssey.ledger", "ok": True, "repeat": False, "text": "{...}"}]
BAD_OBS = [{"tool": "lake.census", "ok": False, "repeat": False, "text": "KeyError"}]


def test_an_answer_backed_by_tool_output_is_not_evidence_none():
    """The engine ran the tool. Calling that "none" is factually wrong."""
    v = Engine._answer_validation(OK_OBS, rounds=3, failed_rounds=0, closure="observation_budget_8")
    assert v["evidence"] != "none", v
    assert v["evidence"] == "tool_observations"
    assert v["observations_ok"] == 1
    assert v["tools_used"] == ["odyssey.ledger"]


def test_an_answer_with_no_observations_still_says_none():
    """The fabricated-completion case is unchanged: no tools, no evidence."""
    v = Engine._answer_validation([], rounds=0, failed_rounds=0, closure=None)
    assert v["evidence"] == "none"
    assert v["observations_ok"] == 0


def test_an_answer_whose_only_observations_FAILED_says_none():
    """A round that only ever got errors back measured nothing."""
    v = Engine._answer_validation(BAD_OBS, rounds=1, failed_rounds=1, closure="failed_call")
    assert v["evidence"] == "none", v
    assert v["observations_failed"] == 1


def test_the_mutation_gate_is_not_weakened_by_any_of_this():
    """THE SCAR CONTROL. An answer with tool output must still not be accepted WORK."""
    v = Engine._answer_validation(OK_OBS, rounds=3, failed_rounds=0, closure=None)
    assert v["accepted_work"] is False
    assert is_accepted_work(v) is False, "an answer became accepted mutation work"
    # and the five-fabricated-completions shape is still rejected
    assert is_accepted_work({"ok": True}) is False
    assert is_accepted_work({"ok": True, "kind": "read_only",
                             "checks": [{"kind": "test", "exit_code": 0}]}) is False


def test_a_measurement_unit_is_recognisable_at_all():
    """The whole point: measurement work must be SCOREABLE, separately."""
    good = Engine._answer_validation(OK_OBS, rounds=2, failed_rounds=0, closure=None)
    assert is_accepted_measurement(good) is True
    assert is_accepted_measurement(Engine._answer_validation([], 0, 0, None)) is False
    assert is_accepted_measurement(Engine._answer_validation(BAD_OBS, 1, 1, "failed_call")) is False
    # a mutation validation is not a measurement, and must not be double-counted
    assert is_accepted_measurement({"ok": True, "checks": [{"kind": "test", "exit_code": 0}]}) is False


def test_the_receipt_says_why_the_tool_loop_closed():
    """Four rounds ended closed and NONE of them could say why. That is the defect."""
    v = Engine._answer_validation(OK_OBS, rounds=6, failed_rounds=2, closure="failed_call")
    assert v["tool_rounds"] == 6
    assert v["failed_rounds"] == 2
    assert v["closure_reason"] == "failed_call"
    # and an unclosed loop must not invent a reason
    assert Engine._answer_validation(OK_OBS, 1, 0, None)["closure_reason"] is None


def test_the_answer_path_actually_calls_it():
    """A builder nothing calls is not a fix. Grep the CALL SITE, not the def.

    The first version of this change defined `_answer_validation` and left the
    hardcoded dict in place at the receipt site. Every test above still passed and
    every receipt still said evidence: none.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "engine.py").read_text()
    assert src.count("self._answer_validation(") >= 1, (
        "_answer_validation is defined but the answer path never invokes it")
    assert '"evidence": "none",\n                        "accepted_work": False,' not in src, (
        "the hardcoded evidence-none validation is back on the answer path")

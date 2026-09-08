"""A round that MEASURED still reported the closing instruction back to us.

Round 5 was the first to work: 5 tool rounds, 7 successful observations, all four
campaign tools used -- odyssey.ledger, campaign.guard, lake.census, odyssey.anatomy --
and it closed on the OBSERVATION BUDGET, not on a failure. It was still going when the
allowance ran out.

Its final answer was:

    "Tool access is closed for this round; no tool calls are emitted. Next: call
     odyssey.ledger {"owed_only": true} ..."

and that string was recorded in the receipt as a verified_fact with source
engine_result. Seven real measurements were taken and none of them reached the report.

The closing prompt asked for exactly that. It said "TOOL ACCESS IS CLOSED FOR THIS
ROUND. Do not emit tool_calls. Use the observations already present and return the
shortest valid answer or mutation now." A model handed that, with observations
attached, answers about the closure -- which is the shortest valid answer.

When observations EXIST the closing turn must ask for the FINDING: what was measured,
on what, what the numbers were, what it means. When there are none, the original text
is right and must not change -- an empty round has nothing to report and inventing a
finding is the failure mode the whole receipt discipline exists to prevent.
"""
from __future__ import annotations

from hcli.engine import Engine
from hcli.tool_registry import default_tool_registry
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _engine():
    e = Engine.__new__(Engine)
    e._tools_cached = default_tool_registry(REPO, repo_root=REPO)
    return e


OK = [{"tool": "odyssey.anatomy", "ok": True, "repeat": False, "text": '{"deficit_pct": 13.4}'}]
BAD = [{"tool": "lake.census", "ok": False, "repeat": False, "text": "KeyError"}]


def test_a_closed_turn_with_measurements_is_asked_to_REPORT_them():
    p = _engine()._prompt_with_observations("resume the odyssey", OK, tools_allowed=False)
    assert "TOOL ACCESS IS CLOSED" in p, "the closure must still be stated"
    low = p.lower()
    # it must ask for the substance, not merely for brevity
    assert "report" in low or "state what you measured" in low, p[-600:]
    assert "measured" in low, "the closing turn never asks what was measured"
    assert "do not describe the tool state" in low, (
        "nothing stops it answering about the closure itself, which is what happened")


def test_a_closed_turn_with_NO_observations_is_unchanged():
    """An empty round has nothing to report. Asking it to report invites invention."""
    p = _engine()._prompt_with_observations("resume the odyssey", [], tools_allowed=False)
    assert "TOOL ACCESS IS CLOSED" in p
    assert "shortest" in p.lower()
    assert "do not describe the tool state" not in p.lower(), (
        "the reporting instruction leaked into a round that observed nothing")


def test_only_FAILED_observations_do_not_trigger_a_report_request():
    """A round that only got errors back measured nothing to report."""
    p = _engine()._prompt_with_observations("resume the odyssey", BAD, tools_allowed=False)
    assert "do not describe the tool state" not in p.lower(), (
        "a round whose every observation failed was asked to report a measurement")

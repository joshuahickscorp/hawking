"""The report request was wired to a path that never receives observations.

376bd2aa4 made the closed turn ask for the finding when at least one observation
succeeded. Round 6 then ran SIX tool rounds with SIX successful observations and ZERO
failures across five tools -- and still answered "Tool access is closed for this round;
no tool calls are emitted."

The guard was correct and unreachable. Both closing paths build their prompt with an
EMPTY observation list and supply the observations separately as `trailing`:

    final_prompt = self._prompt_with_observations(
        prompt, [], compact_catalog=False, tools_allowed=False)
    final_trailing = self._observations_block(
        self._compact_closed_observations(observations), final=True)

so `any(o["ok"] for o in observations)` was evaluated against `[]` every time. A
capability nothing calls with the data does not exist, which is the rule this codebase
keeps re-learning.

The fix cannot be "pass the observations in": the closed branch RENDERS whatever it is
given, and they are already travelling as trailing, so that would duplicate them in the
prompt the round pays prefill on. The decision and the rendering are separate concerns,
so `measured` is passed separately from the observations themselves.
"""
from __future__ import annotations

from pathlib import Path

from hcli.engine import Engine
from hcli.tool_registry import default_tool_registry

REPO = Path(__file__).resolve().parents[2]


def _engine():
    e = Engine.__new__(Engine)
    e._tools_cached = default_tool_registry(REPO, repo_root=REPO)
    return e


def test_the_final_turn_can_be_told_it_measured_without_rendering_observations():
    """The closing paths pass no observations. They must still be able to say so."""
    p = _engine()._prompt_with_observations(
        "resume the odyssey", [], tools_allowed=False, measured=True)
    low = p.lower()
    assert "report the finding" in low, p[-500:]
    assert "do not describe the tool state" in low
    # and it must NOT have rendered an observations block it was not given
    assert "-----" not in p or "observ" not in low.split("report the finding")[0][-200:], p[-800:]


def test_measured_false_is_the_old_behaviour():
    p = _engine()._prompt_with_observations(
        "resume the odyssey", [], tools_allowed=False, measured=False)
    assert "report the finding" not in p.lower()
    assert "TOOL ACCESS IS CLOSED" in p


def test_observations_passed_directly_still_work():
    """The other caller passes real observations; that path must not regress."""
    ok = [{"tool": "odyssey.anatomy", "ok": True, "repeat": False, "text": "{}"}]
    p = _engine()._prompt_with_observations("resume", ok, tools_allowed=False)
    assert "report the finding" in p.lower()


def test_BOTH_closing_paths_pass_measured():
    """The defect was a call site, so the call sites are what this asserts.

    Grepping the source is the only way to catch a guard that is correct and never
    reached: every unit test of the guard itself passed while round 6 measured six
    things and reported none of them.
    """
    src = (Path(__file__).resolve().parents[1] / "engine.py").read_text()
    # the two closed-turn calls must both hand `measured=` in
    # One path rebuilds the prompt and passes measured=; the other reuses the stable
    # cognition_prompt for prefix reuse and must carry the instruction in its tail.
    assert "measured=_measured" in src, (
        "the rebuilt closing prompt does not say whether anything was measured")
    assert "_tail + \"\\n\\n\" + self.REPORT_THE_FINDING" in src, (
        "the budget-exhausted closing path -- the one round 5 took -- still never asks "
        "for the finding")
    assert src.count("_measured = any(") == 2, (
        f"expected both closing paths to compute _measured, found {src.count('_measured = any(')}")

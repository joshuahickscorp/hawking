"""A wrong argument is a normal step of exploration, not a confused plan.

The Odyssey's science round is given four tools and told to CHOOSE ITS OWN TARGET.
Its first move is necessarily a guess -- which slug, which snapshot path -- and the
registry answers a wrong guess with a readable error. That error is the useful part:
`lake.census {"slug": "Qwen--Qwen3-30B-A3B"}` returns
`KeyError: 'Qwen--Qwen3-30B-A3B is not in receipts/future/modellake-index/catalog.json'`,
which names the catalog to read next.

The loop closed the tool catalog on ANY failed call in a round. Three consecutive
autonomous rounds ended with `tool_catalog_mode: "none"` and zero measurements, and
the receipt could not say why -- from outside it looked like the model refusing to
act. The policy is right for a repair unit, where a path miss means the plan is
confused. It is wrong for an exploration unit, where the first miss is how the space
gets mapped.

So: the first failing round is FORGIVEN and its error text goes back as an
observation. A second failing round still closes -- a model that cannot use the
error it was just handed is the confused plan the original rule was written for.
"""
from __future__ import annotations

import os

import pytest

from hcli.engine import Engine


def _engine(max_observations=8, rounds=6):
    e = Engine.__new__(Engine)
    e._cancelled = False
    e.MAX_TOOL_ROUNDS = rounds
    e.MAX_TOOL_OBSERVATIONS = max_observations
    e._agentic_execution = False
    e._tools_closed_for_round = False
    e._last_model_text = ""
    e._prompt_with_observations = lambda *a, **k: "PROMPT"
    # **kw because this stubs an INTERNAL method and pinning its exact signature
    # makes any new keyword a TypeError inside the loop -- which aborts the round
    # after one call and reads as "the loop stopped after the first failed call".
    e._observations_block = lambda obs, final=False, **kw: "OBS"
    e._compact_closed_observations = lambda obs: obs
    e._sanitize_result = lambda v: v
    e._emit = lambda *a, **k: None
    e._write_receipt = lambda **k: "receipt"
    e._compiled_workunit_count = lambda compiled: 0
    return e


def _drive(engine, ok_pattern, tools=("lake.census", "odyssey.ledger", "odyssey.anatomy")):
    """Run the real loop. ok_pattern[i] decides whether round i's call succeeds."""
    ran = []
    replies = [
        {"kind": "tool_use", "content": "", "tool_calls": [{"tool": t, "arguments": []}]}
        for t in tools
    ] + [{"kind": "answer", "content": "measured", "tool_calls": []}]
    seen = []

    def call_model(*a, **k):
        seen.append(1)
        return replies[min(len(seen) - 1, len(replies) - 1)]

    def run_tool_calls(calls, goal_id):
        name = calls[0]["tool"]
        ran.append(name)
        ok = ok_pattern[min(len(ran) - 1, len(ok_pattern) - 1)]
        return [{"tool": name, "ok": ok, "repeat": False,
                 "text": "measured" if ok else "KeyError: not in catalog.json"}]

    engine._call_model = call_model
    engine._run_tool_calls = run_tool_calls
    result = engine.execute("resume the odyssey", evidence=[], compiled={})
    return ran, result


def test_one_failed_tool_call_does_not_end_the_round():
    """The first miss is how the space gets mapped."""
    ran, result = _drive(_engine(), ok_pattern=[False, True, True])
    assert ran[:2] == ["lake.census", "odyssey.ledger"], (
        f"the loop stopped after the first failed call: ran={ran}")
    assert result["kind"] == "answer"


def test_a_second_failed_round_still_closes_the_catalog():
    """Forgiveness is one round, not a licence to spin."""
    ran, _ = _drive(_engine(), ok_pattern=[False, False, True])
    assert ran == ["lake.census", "odyssey.ledger"], (
        f"two failing rounds did not close the loop: ran={ran}")


def test_the_observation_budget_still_closes_the_loop():
    """The forgiveness must not reopen the budget the budget exists to enforce."""
    ran, _ = _drive(_engine(max_observations=2), ok_pattern=[True, True, True])
    assert len(ran) == 2, f"observation budget 2 did not close the loop: ran={ran}"


def test_the_failure_tolerance_is_configurable_and_zero_restores_the_old_rule():
    """Zero tolerance is the pre-existing behaviour, kept reachable on purpose."""
    os.environ["HCLI_TOOL_FAILURE_TOLERANCE"] = "0"
    try:
        e = _engine()
        e.TOOL_FAILURE_TOLERANCE = 0
        ran, _ = _drive(e, ok_pattern=[False, True, True])
        assert ran == ["lake.census"], f"zero tolerance did not close on the first failure: {ran}"
    finally:
        os.environ.pop("HCLI_TOOL_FAILURE_TOLERANCE", None)

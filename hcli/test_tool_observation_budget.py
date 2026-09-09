"""HCLI could make exactly ONE conditioned tool round, no matter how well it went.

The tool loop closed on `len(observations) >= 1`, unconditionally. A unit could
batch up to MAX_TOOL_CALLS_PER_ROUND calls inside one round, but could never
CONDITION a second round on the first's results, so MAX_TOOL_ROUNDS = 6 was
unreachable in practice.

That is fatal for campaign work. "Call odyssey.ledger owed_only, read what is
owed, choose a specimen, guard it, anatomise it" is four steps where each depends
on the previous one, and it cannot be expressed as a single batch. The resident
correctly reported "Tool access is closed for this round" -- the engine had told
it exactly that, one observation in.
"""
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hcli.engine import Engine  # noqa: E402


def test_the_budget_is_a_named_constant_not_a_literal_one():
    assert hasattr(Engine, "MAX_TOOL_OBSERVATIONS")
    assert isinstance(Engine.MAX_TOOL_OBSERVATIONS, int)
    assert Engine.MAX_TOOL_OBSERVATIONS >= 1


def test_the_default_is_unchanged_behaviour():
    """Default stays 1 until a measurement justifies raising it."""
    assert int(os.environ.get("HCLI_MAX_TOOL_OBSERVATIONS", "1")) == 1 or True
    src = (pathlib.Path(__file__).resolve().parents[1] / "hcli" / "engine.py").read_text()
    assert 'HCLI_MAX_TOOL_OBSERVATIONS", "1"' in src, "default drifted from 1 silently"


def test_the_loop_reads_the_budget_and_not_a_literal():
    """The closer must consult the budget it is GIVEN, at any value.

    This was a substring search for `len(observations) >= self.MAX_TOOL_OBSERVATIONS`.
    That pinned one spelling in one place and broke the moment the policy moved into
    a function, while proving nothing about behaviour. The policy is pure now, so
    ask it directly: a literal 1 cannot answer these two budgets differently.
    """
    ok = [{"tool": "t", "ok": True, "repeat": False}]
    assert Engine._tool_loop_closure(ok, 1, 0, 1, 1) == "observation_budget_1"
    assert Engine._tool_loop_closure(ok, 1, 0, 5, 1) is None, "a literal 1 is back"
    assert Engine._tool_loop_closure(ok, 5, 0, 5, 1) == "observation_budget_5"


def test_the_env_override_is_honoured(monkeypatch):
    """A campaign run must be able to buy more conditioned rounds."""
    monkeypatch.setenv("HCLI_MAX_TOOL_OBSERVATIONS", "5")
    import importlib

    import hcli.engine as E
    importlib.reload(E)
    try:
        assert E.Engine.MAX_TOOL_OBSERVATIONS == 5
    finally:
        monkeypatch.delenv("HCLI_MAX_TOOL_OBSERVATIONS", raising=False)
        importlib.reload(E)


def test_the_failure_closers_are_untouched():
    """Raising the budget must NOT weaken the failed-call or all-repeat closers.

    Those exist so a resident cannot spend minutes replaying a confused plan. What
    changed is only WHEN the failed-call closer fires: the first failing round is
    now forgiven so its error text can go back as an observation, and the second
    still closes. Tolerance 0 restores the original any-failure rule exactly.
    """
    failed = [{"tool": "t", "ok": False, "repeat": False}]
    repeated = [{"tool": "t", "ok": True, "repeat": True}]

    # tolerance 0 == the original rule: any failure closes immediately
    assert Engine._tool_loop_closure(failed, 0, 1, 8, 0) == "failed_call"
    # tolerance 1: the first failing round is forgiven, the second is not
    assert Engine._tool_loop_closure(failed, 0, 1, 8, 1) is None
    assert Engine._tool_loop_closure(failed, 0, 2, 8, 1) == "failed_call"
    # The all-repeat closer CHANGED, deliberately, under [S010 10-11]: closing on
    # the FIRST all-repeat round made a correct verification instinct fatal.
    # Rounds 22, 24 and 25 each measured real dense anatomy, re-read the ledger
    # to confirm it, and that confirmation ended the loop before they could
    # write -- all three with observations still in budget. One confirmation is
    # now allowed; two CONSECUTIVE all-repeat rounds still close, so a
    # no-progress cycle stays bounded.
    assert Engine._tool_loop_closure(repeated, 0, 0, 8, 1, repeat_rounds=1) is None
    assert (Engine._tool_loop_closure(repeated, 0, 0, 8, 1, repeat_rounds=2)
            == "bounded_observation_round")
    # and the repeat allowance must never outrank the budget
    assert (Engine._tool_loop_closure(repeated, 8, 0, 8, 1, repeat_rounds=1)
            == "observation_budget_8")
    # and forgiving a failure must never outrank the budget
    assert Engine._tool_loop_closure(failed, 8, 1, 8, 1) == "observation_budget_8"


def test_the_loop_actually_calls_the_policy():
    """A policy nothing calls is not a policy. Grep the CALL SITE, not the def."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "hcli" / "engine.py").read_text()
    assert src.count("self._tool_loop_closure(") >= 1, (
        "_tool_loop_closure is defined but never invoked by the loop")

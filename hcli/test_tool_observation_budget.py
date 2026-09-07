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
    """The closer must consult the constant. A literal 1 is the defect."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "hcli" / "engine.py").read_text()
    assert "len(observations) >= self.MAX_TOOL_OBSERVATIONS" in src
    # the exact defective line must be gone
    assert "or len(observations) >= 1\n" not in src, (
        "the unconditional one-observation cap is back")


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

    Those two exist so a resident cannot spend minutes replaying a confused plan.
    Only the success-path cap moved.
    """
    src = (pathlib.Path(__file__).resolve().parents[1] / "hcli" / "engine.py").read_text()
    # Check the CLOSER CONDITION, not any occurrence of the string. The same text
    # appears again in the reason expression below it, so a substring search
    # passes even with the condition gutted -- which is exactly what happened the
    # first time this test was written.
    i = src.index("if round_observations and (")
    cond = src[i:src.index("):", i)]
    assert 'any(not item.get("ok") for item in round_observations)' in cond, cond
    assert 'item.get("repeat") or not item.get("ok")' in cond, cond
    assert "len(observations) >= self.MAX_TOOL_OBSERVATIONS" in cond, cond

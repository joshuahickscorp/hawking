"""A tool call costs a millisecond. Being asked again costs minutes.

Measured on one live goal: 3 model calls, 11 tool invocations, 5 of them
failures, and the failures were the SAME call re-issued -- fs.read with an
argument that had already been rejected, fs.search with no `pattern`. Each
repeat cost a whole round, and a round is a 92-385 s model call against a 1-3 ms
tool.

Two defects behind that:

* `MAX_TOOL_ROUNDS` was used for two different budgets -- how many rounds the
  loop may take AND how many calls one round may execute. Those costs differ by
  five orders of magnitude, so capping calls-per-round at the round budget
  priced milliseconds like minutes.
* Nothing noticed a repeat. The model cannot break a loop it cannot see.
"""
from __future__ import annotations

import json

import pytest

from hcli.engine import HCLI_RESULT_SCHEMA, Engine


def test_the_two_budgets_are_separate_and_priced_differently():
    assert Engine.MAX_TOOL_ROUNDS == 6
    assert Engine.MAX_TOOL_CALLS_PER_ROUND > Engine.MAX_TOOL_ROUNDS, (
        "a round is a model call and a tool call is a millisecond; the per-round "
        "tool budget must not be capped at the round budget"
    )


def test_the_schema_lets_the_model_ask_for_what_the_executor_will_run():
    """A schema narrower than the executor is a cap the model cannot see past."""
    declared = HCLI_RESULT_SCHEMA["properties"]["tool_calls"]["maxItems"]
    assert declared == Engine.MAX_TOOL_CALLS_PER_ROUND


def test_the_prompt_tells_the_model_what_a_round_costs():
    from hcli.engine import _SYSTEM_PROMPT

    assert "ONE REPLY" in _SYSTEM_PROMPT
    assert "millisecond" in _SYSTEM_PROMPT
    assert str(Engine.MAX_TOOL_CALLS_PER_ROUND) in _SYSTEM_PROMPT


class _CountingRegistry:
    """Counts real executions so a suppressed repeat is provable, not assumed."""

    def __init__(self, ok: bool = True) -> None:
        self.invocations: list = []
        self._ok = ok

    def get(self, name):
        return type("Spec", (), {"input_schema": {"properties": {"path": {"type": "string"}}}})()

    def invoke(self, name, args):
        self.invocations.append((name, dict(args)))
        return type(
            "Result", (), {"ok": self._ok, "value": "CONTENT", "error": "BOOM"}
        )()


def _engine(registry):
    eng = Engine.__new__(Engine)
    eng._tools_cached = registry
    eng._tool_calls_seen = {}
    eng._emit = lambda *a, **k: None
    eng.MAX_EVIDENCE_CHARS_PER_FILE = 4000
    return eng


def _call(path):
    return {"tool": "fs.read", "arguments": [{"name": "path", "value": path}]}


def test_an_identical_call_is_not_executed_twice():
    reg = _CountingRegistry()
    eng = _engine(reg)
    first = eng._run_tool_calls([_call("a.py")], "g")
    second = eng._run_tool_calls([_call("a.py")], "g")

    assert len(reg.invocations) == 1, "the repeat was executed again"
    assert first[0].get("repeat") is not True
    assert second[0]["repeat"] is True
    assert "REPEAT" in second[0]["text"]
    assert "CONTENT" in second[0]["text"], "the prior answer must still be given"


def test_a_repeated_FAILURE_says_it_will_keep_failing():
    """The loop this exists to break: five identical rejected calls."""
    reg = _CountingRegistry(ok=False)
    eng = _engine(reg)
    eng._run_tool_calls([_call("hcli")], "g")
    again = eng._run_tool_calls([_call("hcli")], "g")

    assert len(reg.invocations) == 1
    text = again[0]["text"]
    assert "FAILED then and fails now" in text
    assert "Change the arguments or answer" in text


def test_a_different_argument_is_a_different_call(_=None):
    """Negative control: suppression must not swallow real work."""
    reg = _CountingRegistry()
    eng = _engine(reg)
    eng._run_tool_calls([_call("a.py")], "g")
    eng._run_tool_calls([_call("b.py")], "g")
    assert len(reg.invocations) == 2


def test_many_calls_in_one_round_all_run():
    reg = _CountingRegistry()
    eng = _engine(reg)
    batch = [_call(f"f{i}.py") for i in range(12)]
    out = eng._run_tool_calls(batch, "g")
    assert len(reg.invocations) == 12, "one round must be able to do a round's worth"
    assert len(out) == 12


def test_the_per_round_cap_still_bites():
    reg = _CountingRegistry()
    eng = _engine(reg)
    batch = [_call(f"f{i}.py") for i in range(Engine.MAX_TOOL_CALLS_PER_ROUND + 5)]
    eng._run_tool_calls(batch, "g")
    assert len(reg.invocations) == Engine.MAX_TOOL_CALLS_PER_ROUND

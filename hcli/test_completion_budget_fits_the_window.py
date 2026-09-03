"""prompt + max_new_tokens must never exceed the runtime's window.

The runtime refused whole calls in 11 ms:

    llama-server request failed: prompt has 5792 tokens and max_new_tokens is
    2612; resident max_seq_len is 8192

Two causes, both here. The floor was applied with `max(_MAX_TOKENS_FLOOR, ...)`,
so a request whose prompt had already consumed the context still asked for 512
completion tokens and overflowed. And the budget was resolved BEFORE
`contract.apply` injected the schema instruction, so the payload grew by ~713
tokens after the number was fixed.

A request that cannot fit is not worth making. Asking for a floor that does not
exist turns a tight fit into a hard failure.
"""
from __future__ import annotations

import pytest

from hcli.engine import _CTX_ESTIMATE_MARGIN, _MAX_TOKENS_FLOOR, Engine


def _engine(ctx: int = 8192):
    eng = Engine.__new__(Engine)
    eng.config = type("C", (), {"model_tokens": lambda self: (None, None)})()
    eng._context_budget = lambda: type("B", (), {"per_request_ctx": ctx})()
    return eng


@pytest.mark.parametrize("prompt_tokens", [1, 1000, 4000, 5580, 5792, 7000, 8100, 8192])
def test_the_sum_never_exceeds_the_window(prompt_tokens):
    eng = _engine()
    max_new, _ = eng._resolve_max_tokens(prompt_tokens)
    assert prompt_tokens + max_new <= 8192 + 1, (
        f"prompt {prompt_tokens} + max_new {max_new} overflows the window"
    )
    assert max_new >= 1, "a request must be allowed to produce something"


def test_a_prompt_that_fills_the_window_is_clamped_not_floored():
    """The exact defect: the floor pushed the request past the window."""
    eng = _engine()
    max_new, source = eng._resolve_max_tokens(8100)
    assert max_new < _MAX_TOKENS_FLOOR
    assert source == "derived_clamped_to_window"


def test_a_roomy_prompt_still_gets_a_generous_budget():
    """Negative control: clamping must not starve the normal case."""
    eng = _engine()
    max_new, source = eng._resolve_max_tokens(1000)
    assert max_new > 4000
    assert source == "derived"


def test_a_margin_is_left_for_estimator_disagreement():
    """Being over by one token costs the entire call."""
    eng = _engine()
    max_new, _ = eng._resolve_max_tokens(4000)
    assert 4000 + max_new <= 8192 - _CTX_ESTIMATE_MARGIN + 1


def test_an_explicit_configured_budget_still_wins():
    """Operator intent is not overridden by the derivation."""
    eng = _engine()
    eng.config = type(
        "C", (), {"model_tokens": lambda self: (1234, "config")}
    )()
    assert eng._resolve_max_tokens(1000) == (1234, "config")


def test_the_posted_estimate_lowers_a_stale_budget():
    """The budget must follow the prompt that is actually sent.

    It was resolved before contract.apply grew the payload; re-measuring is the
    only place the true size is known.
    """
    eng = _engine()
    eng._last_call_plan = {"max_tokens": 6000}
    eng._context_efficiency = {}
    payload = {
        "messages": [{"role": "user", "content": "x" * (5792 * 3)}],
        "max_tokens": 6000,
    }
    eng._commit_posted_prompt_estimate(payload)
    assert payload["max_tokens"] < 6000
    assert payload["max_tokens"] + 5792 <= 8192


def test_a_budget_that_already_fits_is_left_alone():
    """Negative control: only ever lower it, never raise it."""
    eng = _engine()
    eng._last_call_plan = {"max_tokens": 100}
    eng._context_efficiency = {}
    payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
    eng._commit_posted_prompt_estimate(payload)
    assert payload["max_tokens"] == 100

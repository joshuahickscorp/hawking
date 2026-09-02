"""An explicit completion budget must survive the native adapter.

The engine resolves a per-call budget from the native context profile (8192
total, 5632 usable) and sends it as payload["max_tokens"]. The adapter treated
`generation.max_new_tokens` -- a DEFAULT for callers that ask for nothing, and
2048 when HCLI_HAWKING_MAX_NEW_TOKENS is unset -- as a hard ceiling over that
explicit request. Every sovereign work unit died at exactly 2048 tokens with
finish_reason=length, and the receipt said "hit the 6310-token completion
budget": a number the model never reached.
"""
from __future__ import annotations

import pytest

from hcli.engine import _truncation_message
from hcli.hawking_native import HawkingNativeConfig, HawkingNativeConnector


def _runtime(default_max_new_tokens: int = 2048, max_seq_len: int = 8192):
    cfg = HawkingNativeConfig(
        binary="/nonexistent/resident",
        artifact_root="/nonexistent/artifact",
        tokenizer="/nonexistent/tokenizer.json",
        max_seq_len=max_seq_len,
        generation={"max_new_tokens": default_max_new_tokens},
    )
    return HawkingNativeConnector.__new__(HawkingNativeConnector), cfg


def _limits(payload, prompt_tokens, *, default=2048, max_seq_len=8192):
    runtime, cfg = _runtime(default, max_seq_len)
    runtime.config = cfg
    return HawkingNativeConnector._limits(runtime, payload, prompt_tokens)


def test_explicit_budget_is_not_capped_by_the_config_default():
    """The exact production shape: engine asks 6310, default is 2048.

    6310 does not itself fit an 8192 window behind a 2122-token prompt, so the
    honest answer is the context bound (6062), not the 2048 default. What must
    never happen again is the request collapsing to the default.
    """
    granted, _seq, _clamped = _limits({"max_tokens": 6310}, 2122)
    context_bound = 8192 - 2122 - 8
    assert granted == context_bound, (
        f"granted {granted}; an explicit request must be bounded by the "
        f"context window ({context_bound}), by nothing else"
    )
    assert granted != 2048, "the request collapsed to the config default again"
    assert granted > 2048 * 2, (
        f"granted {granted}; the whole defect was a 3x cut to 2048"
    )


def test_an_explicit_budget_that_fits_is_granted_whole():
    """Separates 'bounded by context' from 'capped by the default'."""
    granted, _seq, clamped = _limits({"max_tokens": 4096}, 512, default=2048)
    assert granted == 4096, (
        f"granted {granted}; 4096 fits the window behind a 512-token prompt "
        "and exceeds the 2048 default, so the default must not bind it"
    )
    assert clamped is False, "an honoured request must not report itself clamped"


def test_the_context_window_still_bounds_an_explicit_request():
    """The fix must not remove the only bound that is real."""
    granted, _seq, clamped = _limits({"max_tokens": 999_999}, 2122, max_seq_len=8192)
    assert granted == 8192 - 2122 - 8, (
        f"granted {granted}; a request larger than the context must be bounded "
        "by the context window"
    )
    assert clamped is True, "a genuinely bounded request must report clamped"


def test_a_caller_that_asks_for_nothing_still_gets_the_configured_default():
    granted, _seq, _clamped = _limits({}, 100, default=777)
    assert granted == 777, (
        f"granted {granted}; with no explicit max_tokens the configured "
        "default is the right answer"
    )


def test_a_prompt_that_cannot_fit_raises_rather_than_granting_zero():
    with pytest.raises(Exception) as exc:
        _limits({"max_tokens": 512}, 8191, max_seq_len=8192)
    assert "no generation token fits" in str(exc.value)


def test_the_truncation_message_names_the_real_ceiling():
    """A runtime that stops short of the budget must say so.

    Reporting only the requested budget made a clamping runtime look identical
    to a model that genuinely exhausted its budget -- which is why the 2048
    ceiling stayed invisible in every receipt.
    """
    short = _truncation_message(6310, 2048, 2122)
    assert "2048" in short, "the message omits what the model actually produced"
    assert "SHORT" in short, (
        "the message does not distinguish a clamping runtime from an exhausted "
        f"budget: {short!r}"
    )
    exhausted = _truncation_message(6310, 6310, 2122)
    assert "SHORT" not in exhausted, (
        "a genuinely exhausted budget must NOT be reported as a runtime clamp; "
        f"got {exhausted!r}"
    )

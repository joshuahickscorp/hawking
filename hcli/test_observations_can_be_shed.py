"""Observations are re-derivable context and must be sheddable.

Twelve model calls succeeded at ~2,600 prompt tokens and the thirteenth was
refused: `demand 14135 exceeds per-request ctx 8192`. Observations accumulate
across tool rounds, and the reduction ladder had no lever on them -- it would
drop evidence to ZERO and then the durable checkpoint, while the block that
actually caused the overflow sat untouched, because moving observations into
`trailing` (for prefix stability) also moved them outside the ladder.

Order matters and is deliberate: evidence first, then the checkpoint, and only
then observations. A file snapshot can be re-read for free; a tool call has
already been paid for.

Newest are kept. The last tool result is the one the model asked for and is
about to reason over; the earliest are the ones it has already used.
"""
from __future__ import annotations

import pytest

from hcli.engine import Engine, _join_observations, _observation_blocks


def _obs(n):
    return [{"tool": f"tool{i}", "ok": True, "text": f"result-{i} " + "x" * 40} for i in range(n)]


def _rendered(n):
    return Engine.__new__(Engine)._observations_block(_obs(n))


def test_the_rendered_block_splits_back_into_one_entry_per_tool():
    assert len(_observation_blocks(_rendered(5))) == 5


def test_shedding_keeps_the_NEWEST_results():
    blocks = _observation_blocks(_rendered(4))
    kept = _join_observations(blocks[-2:])
    assert "result-3" in kept and "result-2" in kept
    assert "result-0" not in kept and "result-1" not in kept


def test_the_reader_is_told_that_earlier_results_were_dropped():
    """A silently shortened context is a context the model cannot reason about."""
    kept = _join_observations(_observation_blocks(_rendered(4))[-1:])
    assert "earlier tool results dropped" in kept


def test_an_empty_or_absent_tail_yields_nothing_to_shed():
    assert _observation_blocks("") == []
    assert _observation_blocks("some prompt with no observations") == []
    assert _join_observations([]) == ""


def test_the_ladder_sheds_evidence_before_observations(tmp_path):
    """A file snapshot is cheaper to re-derive than a tool call already paid for."""
    import inspect

    src = inspect.getsource(Engine._fit_payload_to_budget)
    ev_at = src.index("evidence 0 + no checkpoint")
    obs_at = src.index("observations {keep_n}")
    assert ev_at < obs_at, "observations must be the LAST thing shed"


def test_shedding_actually_shrinks_the_payload():
    """The load-bearing property: fewer blocks must mean fewer characters."""
    blocks = _observation_blocks(_rendered(8))
    full = _join_observations(blocks)
    half = _join_observations(blocks[-4:])
    one = _join_observations(blocks[-1:])
    assert len(full) > len(half) > len(one)


def test_one_observation_can_never_exceed_the_window():
    """The blocker shedding could not solve.

    MAX_EVIDENCE_CHARS_PER_FILE was 24,000 characters -- about 8,000 tokens --
    against a usable input of 5,632. A single fs.read of a large file was 1.4x
    the entire context, so the ladder could shed every OTHER observation and
    still be over. Measured: demand stuck at 12,469 against a 8,192 window.
    """
    eng = Engine.__new__(Engine)
    eng.MAX_EVIDENCE_CHARS_PER_FILE = 24000
    eng._context_budget = lambda: type("B", (), {"usable_input_tokens": 5632})()

    clamped = eng._clamp_observation("x" * 30000)
    assert len(clamped) < 5632 * 3 // 2, "one result must not dominate the window"
    assert "truncated to fit the context window" in clamped, (
        "silent truncation hides that there was more"
    )


def test_a_small_result_is_returned_exactly(tmp_path):
    """Negative control: clamping must not touch what already fits."""
    eng = Engine.__new__(Engine)
    eng.MAX_EVIDENCE_CHARS_PER_FILE = 24000
    eng._context_budget = lambda: type("B", (), {"usable_input_tokens": 5632})()
    assert eng._clamp_observation("VALUE = 1\n") == "VALUE = 1\n"


def test_the_clamp_follows_the_window_rather_than_a_constant():
    """A bigger window should permit bigger results, without a code change."""
    eng = Engine.__new__(Engine)
    eng.MAX_EVIDENCE_CHARS_PER_FILE = 24000

    eng._context_budget = lambda: type("B", (), {"usable_input_tokens": 5632})()
    small = len(eng._clamp_observation("x" * 30000))
    eng._context_budget = lambda: type("B", (), {"usable_input_tokens": 30000})()
    large = len(eng._clamp_observation("x" * 30000))
    assert large > small


def test_a_broken_budget_falls_back_to_the_constant():
    """Telemetry must never be the thing that ends a goal."""
    eng = Engine.__new__(Engine)
    eng.MAX_EVIDENCE_CHARS_PER_FILE = 500

    def boom():
        raise RuntimeError("no budget")

    eng._context_budget = boom
    # Assert the PAYLOAD is bounded, not payload-plus-notice. The notice is an
    # explanation of what to do next and is allowed to grow; the thing the
    # fallback exists to bound is how much observation text survives.
    clamped = eng._clamp_observation("x" * 5000)
    payload = clamped.split("\n[...", 1)[0]
    assert len(payload) <= 500, len(payload)
    assert "truncated" in clamped

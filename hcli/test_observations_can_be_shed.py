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

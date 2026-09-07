"""The round reported "No observations were provided" while its receipt said five.

Round 8 ran five tool rounds with FIVE successful observations and zero failures, used
odyssey.ledger and campaign.guard, and its closing answer was:

    "No observations were provided; no measurement was made or recorded."

Its receipt says observations_ok: 5. Both are true, from different vantage points: the
engine collected five, and the closing turn was handed almost none. Its final call was
1,553 prompt tokens against 4,700-5,800 on every earlier call -- the observations were
not in it.

`_observation_floor` is the cause, and it is a GOOD mechanism used in the wrong place.
It is a forward-only index so each retrieval turn is the previous turn plus an append,
which is what a KV prefix can reuse; without it five consecutive calls sat pinned at
1,398 reused tokens while the prompts grew past 4,700. But the FIRST ladder attempt
applies that advanced floor, so once it moves the closing turn can never see anything
before it.

The closing turn is a DECISION turn, not another retrieval turn. Its whole job is to
report on everything the round measured, and it runs once. Prefix reuse is worth less
there than the round's only product, so it starts from observation zero. The observations
are already compacted to CLOSED_OBSERVATION_CHARS each, so the full tail is small.
"""
from __future__ import annotations

import pytest

from hcli.engine import Engine, _observation_blocks


def _trailing(n: int) -> str:
    obs = [{"tool": f"t{i}", "ok": True, "text": f"result-{i}"} for i in range(n)]
    e = Engine.__new__(Engine)
    return e._observations_block(obs, final=True)


def test_the_fixture_itself_splits_into_the_blocks_it_claims():
    """Guard the guard: a header mismatch would make every assertion below vacuous."""
    assert len(_observation_blocks(_trailing(5))) == 5


def test_a_retrieval_turn_still_advances_the_floor():
    """The prefix-reuse mechanism must survive this change."""
    e = Engine.__new__(Engine)
    e._tools_closed_for_round = False
    e._observation_floor = 3
    assert e._observation_start(_observation_blocks(_trailing(5))) == 3


def test_the_DECISION_turn_starts_from_observation_zero():
    e = Engine.__new__(Engine)
    e._tools_closed_for_round = True
    e._observation_floor = 4
    start = e._observation_start(_observation_blocks(_trailing(5)))
    assert start == 0, (
        f"the closing turn starts at observation {start}, so it cannot report on the "
        f"ones before it -- which is how a round measured five things and answered "
        f"'No observations were provided'")


def test_the_floor_never_exceeds_the_blocks_it_has():
    e = Engine.__new__(Engine)
    e._tools_closed_for_round = False
    e._observation_floor = 99
    assert e._observation_start(_observation_blocks(_trailing(3))) == 2
    assert e._observation_start([]) == 0


def test_the_fit_ladder_actually_calls_it():
    """A helper the ladder does not use is not a fix.

    Restoring the old inline `min(self._observation_floor, ...)` at the call site left
    every test above green, because they all exercise the helper directly. That is the
    same shape as the report request that was correct and unreachable for two rounds.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "engine.py").read_text()
    assert "floor = self._observation_start(blocks)" in src, (
        "_fit_payload_to_budget computes its own floor again and ignores the decision turn")
    assert 'floor = min(getattr(self, "_observation_floor", 0)' not in src, (
        "the inline floor is back alongside the helper; two copies will drift")

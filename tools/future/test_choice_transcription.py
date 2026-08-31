"""A typo in a copied id is not a judgment failure.

The 30m run's steady state: two candidates on the menu, the model picked the
right one for the right reason - gain 2 against gain 1, cited correctly - and
wrote "WU.PROBE.decode_arith cost". One space where the id has an underscore.
The tools refused it as an invented id, nothing launched, the frontier never
advanced, and that same question was asked 94 more times.

Refusing invented ids is correct and stays. Requiring character-perfect
transcription of a long opaque identifier, and then scoring the typo as a
failure to decide, measures the ask instead of the chooser.
"""
from __future__ import annotations

from tools.future import model_bearing as mb

BY_ID = {
    "WU.PROBE.decode_arith_cost": {"id": "WU.PROBE.decode_arith_cost"},
    "WU.SUBAGENT.receipt_wait_probe": {"id": "WU.SUBAGENT.receipt_wait_probe"},
}


def test_the_exact_reply_from_the_run_resolves():
    got, repaired = mb.resolve_choice_id("WU.PROBE.decode_arith cost", BY_ID)
    assert got == "WU.PROBE.decode_arith_cost"
    assert repaired == "WU.PROBE.decode_arith cost", "the repair must be recorded"


def test_an_exact_id_is_not_marked_repaired():
    got, repaired = mb.resolve_choice_id("WU.PROBE.decode_arith_cost", BY_ID)
    assert got == "WU.PROBE.decode_arith_cost"
    assert repaired is None


def test_an_invented_id_is_still_refused():
    got, repaired = mb.resolve_choice_id("WU.TOTALLY.invented", BY_ID)
    assert got == "WU.TOTALLY.invented"
    assert repaired is None
    assert got not in BY_ID, "an invented id must not resolve to anything"


def test_an_ambiguous_fold_is_refused_rather_than_guessed():
    """Better no launch than the wrong one."""
    colliding = {"WU.A_B": {}, "WU.A-B": {}}
    got, repaired = mb.resolve_choice_id("WU.AB", colliding)
    assert got == "WU.AB" and repaired is None
    assert got not in colliding


def test_empty_and_whitespace_resolve_to_nothing():
    for raw in ("", "   ", None):
        got, repaired = mb.resolve_choice_id(raw, BY_ID)
        assert got not in BY_ID and repaired is None


def test_case_and_punctuation_fold_but_letters_do_not():
    assert mb.resolve_choice_id("wu.probe.decodearithcost", BY_ID)[0] == (
        "WU.PROBE.decode_arith_cost"
    )
    # a real letter difference is a different id, not a typo to absorb
    assert mb.resolve_choice_id("WU.PROBE.decode_arith_costs", BY_ID)[0] not in BY_ID

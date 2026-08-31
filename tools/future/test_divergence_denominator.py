"""Divergence has a denominator, and it is not the number of asks.

The first honest 30m model-bearing run asked choose() 56 times and diverged 0
times, which reads as "the model agreed 56 independent times". Only FIVE of
those asks were distinct - one prompt fired 52 times - and those 5 distinct
prompts produced 5 distinct replies. At temperature 0 a deterministic body
re-asked a byte-identical question answers it identically by construction. The
repeat count is a fact about the frontier; reporting it as agreement makes it
look like a fact about the model's judgment, 11x overstated.
"""
from __future__ import annotations

from tools.future import model_bearing as mb


def _choose(digest: str, *, diverged: bool = False):
    return {
        "kind": "choose",
        "cognition": mb.AVAILABLE,
        "prompt_sha256": digest,
        "diverged_from_fixed_policy": diverged,
        "chose": {"id": "WU.X"} if diverged else None,
        "counts_as_decision": True,
        "reason": "because",
        "model_decided": "WU.X",
    }


def test_repeated_asks_do_not_inflate_the_denominator():
    log = [_choose("aaa") for _ in range(52)] + [_choose(f"d{i}") for i in range(4)]
    got = mb.materially_participated(log)
    scope = got["divergence_scope"]
    assert scope["n_choose_asks"] == 56
    assert scope["n_distinct_choose_asks"] == 5
    assert scope["n_repeated_choose_asks"] == 51


def test_the_finding_states_the_distinct_count_not_the_total():
    log = [_choose("aaa") for _ in range(52)] + [_choose(f"d{i}") for i in range(4)]
    finding = mb.materially_participated(log)["publishable_finding"]
    assert "5 distinct prompts" in finding
    assert "56 asks" in finding


def test_distinct_asks_still_counted_when_all_differ():
    log = [_choose(f"h{i}") for i in range(7)]
    scope = mb.materially_participated(log)["divergence_scope"]
    assert scope["n_distinct_choose_asks"] == 7
    assert scope["n_repeated_choose_asks"] == 0


def test_a_real_divergence_still_counts():
    log = [_choose("a"), _choose("b", diverged=True)]
    got = mb.materially_participated(log)
    assert got["divergence_count"] == 1
    assert got["publishable_finding"] is None


def _probe(i: int, extra: str = "WU.PROBE.decode_arith_cost"):
    return {
        "kind": "choose",
        "cognition": mb.AVAILABLE,
        "prompt_sha256": f"h{i}",
        "tools_established": {"ids": [f"WU.HAWKING.health_probe.{i:03d}", extra]},
        "diverged_from_fixed_policy": False,
        "chose": None,
        "counts_as_decision": True,
        "reason": "because",
        "model_decided": "x",
    }


def test_an_indexed_probe_series_is_ONE_question():
    """44 of the 51-launch run's units were WU.HAWKING.health_probe.NNN.

    Distinct prompts is a better denominator than total asks and still the wrong
    one: a frontier that refills itself with indexed probes produces fifty
    distinct prompts and one distinct question.
    """
    scope = mb.materially_participated([_probe(i) for i in range(44)])["divergence_scope"]
    assert scope["n_choose_asks"] == 44
    assert scope["n_distinct_choose_asks"] == 44, "prompts really are distinct"
    assert scope["n_distinct_question_kinds"] == 1, "and they are all one question"


def test_genuinely_different_candidate_sets_count_separately():
    """Different FAMILIES are different questions; different indices are not.

    WU.REAL.0 and WU.REAL.1 would collapse to one kind, and that is the rule
    working: an indexed series is one question. Only a different family counts.
    """
    families = ["exec", "gravity", "doctor", "tooling", "specimen"]
    log = [_probe(i, extra=f"WU.{f.upper()}.probe") for i, f in enumerate(families)]
    scope = mb.materially_participated(log)["divergence_scope"]
    assert scope["n_distinct_question_kinds"] == 5


def test_the_finding_states_the_kind_count_first():
    finding = mb.materially_participated([_probe(i) for i in range(44)])["publishable_finding"]
    assert "1 DISTINCT QUESTION KINDS" in finding
    assert "44 distinct prompts" in finding

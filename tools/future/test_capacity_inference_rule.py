"""The user should not have needed to interpret G009.

S025 §35 is a test of S022 itself: seeing one stream at ~361 GB/s and three at
~580, HCLI should infer that capacity exceeds single-stream exposure and generate
the intra-token questions - without a steer. §36 generalises it and §37 turns the
route this campaign actually took into policy.
"""
from __future__ import annotations

import pytest

from tools.future import capacity_inference_rule as rule


def _lvl(n, tps):
    return {"concurrency": n, "aggregate_decode_tps": tps}


def test_it_fires_on_the_g009_shape():
    got = rule.fires_on([_lvl(1, 36.6), _lvl(2, 51.2)], semantics_comparable=True)
    assert got["fired"] is True
    assert got["inference"] == rule.UNDERUTILIZATION


def test_a_gain_inside_noise_does_not_fire():
    got = rule.fires_on([_lvl(1, 36.6), _lvl(2, 37.5)], semantics_comparable=True)
    assert got["fired"] is False
    assert got["inference"] == rule.NO_INFERENCE


def test_incomparable_semantics_refuse_rather_than_infer():
    """More aggregate throughput from DIFFERENT work is not underutilisation."""
    with pytest.raises(rule.RuleRefused, match="comparable work"):
        rule.fires_on([_lvl(1, 36.6), _lvl(2, 90.0)], semantics_comparable=False)


def test_no_baseline_refuses():
    with pytest.raises(rule.RuleRefused, match="nothing to be under-utilised against"):
        rule.fires_on([_lvl(2, 51.2)], semantics_comparable=True)


def test_a_zero_baseline_refuses_rather_than_dividing():
    with pytest.raises(rule.RuleRefused, match="nothing measurable"):
        rule.fires_on([_lvl(1, 0), _lvl(2, 51.2)], semantics_comparable=True)


def test_it_generates_competing_explanations_and_names_no_cause():
    fired = rule.fire()
    assert fired["fired"] is True
    classes = {q["class"] for q in fired["generated_questions"]}
    assert classes == {"dependency", "occupancy", "latency_hiding",
                       "memory_level_parallelism", "scheduler_topology"}
    assert "Naming the winner before measuring" in fired["does_not_name_a_cause"]


def test_the_parent_class_is_marked_as_needing_a_split():
    q = next(x for x in rule.questions() if x["class"] == "latency_hiding")
    assert "must be split, never answered" in q["discriminator"]


def test_the_experiment_policy_records_the_route_actually_taken():
    p = rule.experiment_policy_lesson()
    seq = " ".join(p["sequence_taken"])
    assert "contiguity killed" in seq and "dispatch count killed" in seq
    assert "concurrency exposed capacity" in seq
    assert "before spending hours polishing static streaming" in p["policy"]
    assert "three schools were killed one at a time" in p["cost_of_not_having_it"]


def test_every_generated_question_carries_a_discriminator():
    for q in rule.questions():
        assert len(q["discriminator"]) > 40, q["class"]

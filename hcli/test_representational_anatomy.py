"""G028: the anatomy must be repeatable, and its verdicts must be falsifiable."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "future"))
from representational_anatomy import (hypotheses, SHARING_LIVE_BELOW, LOWRANK_LIVE_BELOW)


def test_o003_numbers_produce_the_verdicts_actually_measured():
    a = {"cross_expert": [{"ratio": 0.9825, "n": 64}],
         "cross_layer": {"ratio": 0.9569, "n": 26},
         "within_expert": {"ratio": 0.6162, "rank_90": 794}}
    h = " ".join(hypotheses(a))
    assert "cross-expert sharing is DEAD" in h
    assert "low-rank factorisation is DEAD" in h
    assert "NO LINEAR STRUCTURE AVAILABLE" in h


def test_the_second_specimen_reaches_the_same_verdict():
    a = {"cross_expert": [{"ratio": 0.9796, "n": 128}],
         "within_expert": {"ratio": 0.6085, "rank_90": 476}}
    assert "cross-expert sharing is DEAD" in " ".join(hypotheses(a))


def test_a_specimen_that_CONTRADICTS_the_prior_is_reported_as_live():
    """The prior must be able to die. A low ratio has to flip the verdict."""
    a = {"cross_expert": [{"ratio": 0.62, "n": 64}],
         "within_expert": {"ratio": 0.21, "rank_90": 90}}
    h = " ".join(hypotheses(a))
    assert "cross-expert sharing is LIVE" in h, h
    assert "contradicts the family prior" in h, h
    assert "low-rank is LIVE" in h, h
    assert "NO LINEAR STRUCTURE" not in h


def test_thresholds_are_the_published_ones():
    assert SHARING_LIVE_BELOW == 0.95 and LOWRANK_LIVE_BELOW == 0.40


def test_no_hypotheses_without_measurements():
    assert hypotheses({}) == []

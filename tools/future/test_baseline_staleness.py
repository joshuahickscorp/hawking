"""G138 tests: the detector must not over-claim, and must not go stale itself."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_staleness as bs  # noqa: E402


def test_the_current_baseline_is_read_not_typed():
    c = bs.current()
    m = json.loads((bs.REPO / bs.CURRENT_REL).read_text())["measured"]
    assert c["gpu_ms_per_token"] == m["gpu_ms_per_token"]
    assert c["gpu_tps"] == m["gpu_tps"]


def test_without_a_current_baseline_nothing_is_called_stale(monkeypatch):
    """Reporting every receipt stale is worse than no report."""
    monkeypatch.setattr(bs, "CURRENT_REL", "receipts/future/NO_SUCH.json")
    with pytest.raises(bs.StalenessRefused, match="worse than no report"):
        bs.current()


def test_the_number_match_does_not_fire_on_a_longer_number():
    """27.2896 must not match 127.28964, and 36.644 must not match 136.6441."""
    assert bs._occurrences({"a": 27.2896}, "27.2896")
    assert not bs._occurrences({"a": 127.28964}, "27.2896")
    assert not bs._occurrences({"a": "36.6441"}, "36.644")


def test_a_value_under_a_historical_key_is_history():
    doc = {"supersedes": {"stale_ladder_ms": 28.722}}
    assert bs._is_historical(bs._occurrences(doc, "28.722")) is True


def test_ONE_live_occurrence_makes_it_live():
    """A receipt that explains the supersession in prose AND still divides by
    the old number is a live consumer with a good alibi."""
    doc = {"supersedes": {"was": 28.722}, "token_ms": 28.722}
    assert bs._is_historical(bs._occurrences(doc, "28.722")) is False


def test_entitlement_is_structural_not_a_hand_kept_list():
    """A hand-maintained list of entitled receipts would itself go stale, which
    is the exact failure this module exists to catch."""
    src = Path(bs.__file__).read_text()
    assert "historical_owners" not in src
    assert "HISTORICAL_KEY" in src


def test_every_row_carries_where_the_value_appears():
    for r in bs.scan():
        assert r["occurrence_paths"], r["receipt"]
        for v, paths in r["occurrence_paths"].items():
            assert v in r["superseded_values_present"]
            assert paths


def test_an_unreadable_producer_is_not_reported_clean():
    """An unanswered question is not a pass. This is how a stale ledger survives
    an audit."""
    verdicts = {r["verdict"] for r in bs.scan()}
    assert "PRODUCER_UNKNOWN" not in {"HISTORICAL_OWNER"}
    assert bs._reads_current(None) is None
    assert bs._reads_current("tools/future/NO_SUCH_MODULE.py") is None
    w = bs.what_this_does_not_do()
    assert "not as passing" in w["producer_unknown_is_not_clean"]


def test_a_producer_that_reads_the_current_baseline_is_detected():
    assert bs._reads_current("tools/future/gap_ledger_60.py") is True
    assert bs._reads_current("tools/future/path_to_71.py") is True


def test_it_is_labelled_a_review_list_not_a_defect_list():
    """Calling all 22 defects would be the same over-claiming this catches."""
    r = bs.report()
    assert "over-claiming" in r["this_is_a_REVIEW_LIST_not_a_DEFECT_LIST"]
    assert "n_needing_review" in r


def test_it_does_not_rewrite_receipts():
    w = bs.what_this_does_not_do()
    assert w["does_not_rewrite"] is True
    assert "receipt-only fix" in w["why"]


def test_the_three_hand_found_cases_are_recorded_with_their_deltas():
    found = bs.build()["found_by_hand_three_times_first"]
    assert len(found) == 3
    joined = " ".join(found)
    assert "10.6229" in joined and "5.2797" in joined
    assert "49.8" in joined and "62.90" in joined
    assert "34.7%" in joined and "114.6%" in joined


def test_the_two_ledgers_i_already_fixed_now_read_current():
    """GAP_LEDGER_60 and PATH_TO_71 were the first two found by hand."""
    rows = {r["receipt"]: r for r in bs.scan()}
    if "GAP_LEDGER_60.json" in rows:
        assert rows["GAP_LEDGER_60.json"]["producer_reads_current_baseline"] is True

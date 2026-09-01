"""G114 tests: the metric must be computed, and must be able to look bad."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claude_interventions as ci  # noqa: E402


def test_the_register_is_a_real_input_not_a_decoration():
    r = ci.register()
    src = json.loads((ci.REPO / ci.REGISTER_REL).read_text())
    assert r["n_operations"] == len(src["register"])
    assert r["n_claude_owned"] + r["n_handed_off"] == r["n_operations"]
    assert ci.metric()["register_fraction_handed_off"] == r["fraction_handed_off"]


def test_a_missing_register_refuses_rather_than_asserting(monkeypatch):
    monkeypatch.setattr(ci, "REGISTER_REL", "receipts/future/NO_SUCH.json")
    with pytest.raises(ci.MetricRefused, match="asserted rather than computed"):
        ci.register()


def test_frontier_moves_come_from_git_with_a_clock():
    fm = ci.frontier_moves()
    assert fm["n"] > 0
    assert fm["span_hours"] > 0
    assert fm["last_unix"] > fm["first_unix"]
    assert fm["distinct_obligations"] <= fm["n"]


def test_only_obligation_bearing_commits_count_as_frontier_moves():
    assert ci.OBLIGATION.search("feat(G126): x")
    assert ci.OBLIGATION.search("fix(G21,G129): y")
    assert not ci.OBLIGATION.search("chore: tidy up")
    assert not ci.OBLIGATION.search("feat(future): no obligation named")


def test_only_EXECUTED_resident_work_counts_as_a_decision():
    """Accepted-but-unlaunched means the resident chose and nothing happened.
    Counting it would credit the resident for work the harness dropped."""
    rd = ci.resident_decisions()
    rows = [json.loads(l) for l in
            (ci.REPO / ci.SOVEREIGN_LOG_REL).read_text().splitlines() if l.strip()]
    its = [r for r in rows if "n" in r]
    ran = [r for r in its if any(x.get("ran") for x in (r.get("results") or []))]
    assert rd["n_executed"] == len(ran)
    assert rd["n_executed"] <= rd["n_parsed"] <= rd["n_iterations"]


def test_the_per_hour_rate_is_withheld_while_entries_are_unclocked():
    """Back-dating an entry to complete a rate is fabricating a measurement."""
    rd = ci.resident_decisions()
    if rd["n_unclocked"]:
        assert rd["per_hour"] is None
        assert rd["per_hour_status"] == "UNCLOCKED_ENTRIES_PRESENT"
        assert "fabricating" in rd["why_no_rate"]
    else:
        assert rd["per_hour_status"] == "MEASURED"


def test_the_producer_now_stamps_an_absolute_clock():
    src = (Path(ci.__file__).parent / "hcli_sovereign.py").read_text()
    assert 'rec.setdefault("unix", time.time())' in src, (
        "the per-hour half of this obligation is unmeasurable without it"
    )


def test_the_metric_is_arithmetic_over_the_two_streams():
    m, fm, rd = ci.metric(), ci.frontier_moves(), ci.resident_decisions()
    assert m["frontier_moves"] == fm["n"]
    assert m["resident_decided"] == rd["n_executed"]
    assert m["claude_interventions"] == max(fm["n"] - rd["n_executed"], 0)
    assert m["interventions_per_frontier_move"] == pytest.approx(
        m["claude_interventions"] / m["frontier_moves"], abs=5e-4)


def test_the_metric_does_not_flatter():
    """It should read high today, and the receipt should say the true figure is
    higher still. A metric that cannot report a bad number is not a metric."""
    m = ci.metric()
    assert m["interventions_per_frontier_move"] > 0.5
    assert "higher than this figure, not lower" in \
        m["this_is_an_upper_bound_on_resident_credit"]


def test_one_reading_is_not_called_a_trend():
    f = ci.is_it_falling()
    assert f["verdict"] == "NOT_YET_ANSWERABLE"
    assert "one point" in f["why"]
    assert f["what_would_make_it_falling"]


def test_the_register_share_and_the_move_share_are_kept_distinct():
    """An operation can have an owner on disk and still be exercised by Claude."""
    m = ci.metric()
    assert "more optimistic question" in m["reading"]
    assert m["register_fraction_handed_off"] != \
        1 - m["interventions_per_frontier_move"]

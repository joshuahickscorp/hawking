"""G133 tests: a decomposition that does not reconcile is not a decomposition."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import organ_decomposition_sealed as o  # noqa: E402


def _patch(monkeypatch, rel, mutate):
    d = json.loads((o.REPO / rel).read_text())
    mutate(d)
    real = o._load
    monkeypatch.setattr(o, "_load", lambda r: d if r == rel else real(r))


def test_the_sealed_rows_reconcile_with_the_measured_token():
    r = o.reconciliation()
    assert r["reconciles"] is True
    assert r["residual_relative"] <= o.MAX_RECONCILE_REL
    assert abs(r["residual_ms"]) < 0.2


def test_the_previous_table_did_not_reconcile_and_that_is_recorded():
    """The reason this measurement was taken at all."""
    prev = o.reconciliation()["the_previous_table_did_not"]
    assert prev["was_above_its_baseline_by_ms"] > 4.0
    assert "no longer runs" in prev["why"]


def test_a_decomposition_that_does_not_sum_to_its_total_is_refused(monkeypatch):
    def mutate(d):
        # The LARGEST organ, so the sum actually moves. Mutating whichever row
        # happens to be first would pass on embedding at 0.0088 ms.
        big = max(d["isolated_organs"]["organs"], key=lambda r: r["gpu_ns_median"])
        big["gpu_ns_median"] *= 4
    _patch(monkeypatch, o.NEW_REL, mutate)
    with pytest.raises(o.DecompositionRefused, match="not a decomposition"):
        o.reconciliation()


def test_a_changed_organ_set_is_refused_rather_than_differenced(monkeypatch):
    def mutate(d):
        d["isolated_organs"]["organs"].append(
            {"organ": "brand_new", "gpu_ns_median": 1000})
    _patch(monkeypatch, o.NEW_REL, mutate)
    with pytest.raises(o.DecompositionRefused, match="not a delta"):
        o.table()


def test_shares_sum_to_one():
    assert sum(r["share_of_sealed_token"] for r in o.table()) \
        == pytest.approx(1.0, abs=2e-3)


def test_deltas_are_new_minus_pre():
    for r in o.table():
        assert r["delta_ms"] == pytest.approx(r["sealed_ms"] - r["pre_ms"], abs=5e-4)


def test_the_table_is_ranked_by_sealed_cost():
    ms = [r["sealed_ms"] for r in o.table()]
    assert ms == sorted(ms, reverse=True)
    assert o.table()[0]["organ"] == "mlp_gate_up"


def test_most_of_the_saving_landed_in_mlp():
    """An MLP dequant lever that saved its time somewhere else would mean the
    arms were not matched."""
    w = o.where_the_saving_landed()
    assert w["mlp_share_of_the_saving"] > 0.7
    assert w["total_delta_ms"] < 0


def test_the_organs_nothing_touched_did_not_move():
    w = o.where_the_saving_landed()
    assert set(w["organs_that_did_not_move"]) >= {"sampling", "embedding"}


def test_everything_outside_mlp_and_deltanet_cannot_supply_the_gap():
    """The decisive fact for where to look next."""
    r = o.what_remains()
    assert r["deleting_everything_else_perfectly_still_misses_60_by_ms"] > 0
    assert r["everything_else_ms"] < r["gap_to_60_ms"]
    assert r["mlp_share"] > 0.5


def test_the_gap_figures_come_from_the_protected_absolute():
    a = json.loads((o.REPO / o.ABSOLUTE_REL).read_text())["measured"]
    base = float(a["gpu_ms_per_token"])
    r = o.what_remains()
    assert r["gap_to_60_ms"] == pytest.approx(base - 1000.0 / 60.0, abs=5e-4)
    assert r["gap_to_71_ms"] == pytest.approx(base - 1000.0 / 71.0, abs=5e-4)


def test_a_missing_raw_refuses(monkeypatch):
    monkeypatch.setattr(o, "NEW_REL", "receipts/future/NO_SUCH.json")
    with pytest.raises(o.DecompositionRefused, match="not on disk"):
        o.table()

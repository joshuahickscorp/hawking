"""G075's reprofile must be a measurement, not a remembered number.

The stale 28.722 ms ladder was the whole reason this exists. Three things had to
be true before the number could be taken and none of them were listed: a rebuild
(widen_f4's selection path postdated every binary on disk), the release profile
rather than release-fast (whose Cargo.toml forbids benchmarking with it), and a
protected window (ModelLake SIGSTOPped, because absolutes move with load).
"""
from __future__ import annotations

import pytest

from tools.future import resident_reprofile as rr


def test_the_arms_are_token_identical():
    """A speed delta between different outputs is not a speedup."""
    ab = rr.arms()
    assert ab["token_identical"] is True
    assert ab["decode_steps"] == 48


def test_the_measured_delta_reproduces_the_claimed_win():
    """widen_f4 was landed on a claimed 1.0245 ms. This is an independent run."""
    ab = rr.arms()
    assert ab["delta_ms"] == pytest.approx(1.09, abs=0.05)
    assert ab["widen_f4_ms_per_token"] < ab["baseline_ms_per_token"]


def test_the_ladder_baseline_actually_moved():
    doc = rr.build()
    assert doc["supersedes"]["stale_ladder_ms"] == 28.722
    assert doc["decode_wall_ms_per_token"] < 28.722
    assert doc["supersedes"]["ms_removed"] > 1.0


def test_widen_f4_is_proven_by_a_dispatched_kernel_not_by_strings():
    """strings(1) was the wrong probe: widen_f4 is only the as_str() display
    form and fat LTO removes it. The kernel name in the run is the evidence."""
    org = rr.organs()
    assert org["widen_f4_kernel_dispatched"] is True


def test_the_organ_rows_reconcile_to_the_measured_gpu_token():
    org = rr.organs()
    # organs are measured in isolation, so they need not sum exactly to the
    # in-graph token, but they must be within a few percent or the census is
    # measuring something other than the token.
    assert abs(org["organ_ms_sum"] - org["decode_gpu_ms_per_token"]) < 1.0
    assert org["dispatches"] == 628


def test_the_floor_control_reports_its_own_reliability():
    """An empty command buffer is often below timestamp resolution. That is
    information about the floor, not a reason to discard the census."""
    noop = rr.organs()["noop_floor_control"]
    assert noop["n_reps_with_timestamp"] >= 1
    assert noop["n_reps_with_timestamp"] + noop["n_reps_below_timestamp_resolution"] == noop["n_reps"]


def test_a_missing_raw_refuses_rather_than_assembling_from_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(rr, "GREEDY_BASELINE", tmp_path / "nope.json")
    with pytest.raises(rr.ReprofileRefused, match="not on disk"):
        rr.arms()


def test_the_receipt_is_not_offered_as_a_clean_absolute():
    doc = rr.build()
    assert doc["timing_label"] == "DIRTY_ENGINEERING"
    assert doc["protected_window"]["modellake_sigstopped"] is True
    assert doc["binary"]["profile"] == "release"

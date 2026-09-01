"""G131 tests: an absolute must be refusable."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sealed_default_absolute as a  # noqa: E402


def _patch(monkeypatch, mutate):
    d = json.loads((a.REPO / a.RAW_REL).read_text())
    mutate(d)
    real = a._load
    monkeypatch.setattr(a, "_load", lambda r: d if r == a.RAW_REL else real(r))


def test_the_measured_window_is_steady():
    m = a.measured()
    assert m["n_reps"] >= a.MIN_REPS
    assert m["gpu_rep_spread"] <= a.MAX_SPREAD
    assert m["gpu_ms_min"] <= m["gpu_ms_per_token"] <= m["gpu_ms_max"]


def test_tps_is_derived_from_the_measured_ms():
    m = a.measured()
    assert m["gpu_tps"] == pytest.approx(1000.0 / m["gpu_ms_per_token"], abs=5e-3)
    assert m["wall_tps"] == pytest.approx(1000.0 / m["wall_ms_per_token"], abs=5e-3)


def test_host_gap_is_wall_minus_gpu_not_a_field():
    m = a.measured()
    assert m["host_gap_ms"] == pytest.approx(
        m["wall_ms_per_token"] - m["gpu_ms_per_token"], abs=5e-4)


def test_it_corroborates_the_prior_protected_window():
    c = a.corroboration()
    assert c["agrees"] is True
    assert c["relative_difference"] <= a.CORROBORATION_REL


def test_a_pinned_arm_is_refused(monkeypatch):
    _patch(monkeypatch, lambda d: d.__setitem__(
        "the_state_kernel_was_never_pinned", False))
    with pytest.raises(a.AbsoluteRefused, match="pinned an arm"):
        a.raw()


def test_an_inherited_lever_is_refused(monkeypatch):
    _patch(monkeypatch, lambda d: d.__setitem__("levers_unset", False))
    with pytest.raises(a.AbsoluteRefused, match="pinned an arm or inherited"):
        a.raw()


def test_too_few_reps_is_refused(monkeypatch):
    _patch(monkeypatch, lambda d: d.__setitem__(
        "complete_token_gpu_ns_medians", d["complete_token_gpu_ns_medians"][:3]))
    with pytest.raises(a.AbsoluteRefused, match="reps"):
        a.raw()


def test_an_unsteady_window_is_refused(monkeypatch):
    def mutate(d):
        v = list(d["complete_token_gpu_ns_medians"])
        v[0] = int(v[0] * 1.5)
        d["complete_token_gpu_ns_medians"] = v
    _patch(monkeypatch, mutate)
    with pytest.raises(a.AbsoluteRefused, match="not a steady window"):
        a.measured()


def test_a_disagreeing_prior_window_is_reported_not_hidden(monkeypatch):
    real = a._load
    def fake(rel):
        d = real(rel)
        if rel == a.PRIOR_LEASE_REL:
            d = json.loads(json.dumps(d))
            d["measured"]["bitcast_gpu_ms"] = 30.0
        return d
    monkeypatch.setattr(a, "_load", fake)
    c = a.corroboration()
    assert c["agrees"] is False
    assert c["relative_difference"] > a.CORROBORATION_REL


def test_the_wall_figure_is_never_the_headline():
    doc = a.build()
    m = doc["measured"]
    assert doc["headline_gpu_tps"] == m["gpu_tps"]
    assert doc["headline_gpu_tps"] != m["wall_tps"]
    assert "host_gap_is_not_the_resident" in doc


def test_the_host_gap_spans_more_than_the_gpu_figure_moves():
    """The whole reason wall is not promoted."""
    h = a.host_gap_is_not_the_resident()
    c = a.corroboration()
    assert h["span"] > 1.5
    assert c["relative_difference"] < 0.01


def test_the_lease_records_the_worker_that_exited():
    """An absolute that hides a disturbed download is not an honest absolute."""
    L = a.lease()
    w = L["one_worker_exited_after_resume"]
    assert w["work_lost"] is None
    assert "AFTER the window closed" in w["when"]
    assert "resumes rather than restarts" in w["partial_intact"]
    assert "SIGSTOP and SIGCONT only" in L["nothing_was_killed"]


def test_the_gap_to_the_targets_is_computed_from_the_measured_gpu_ms():
    doc = a.build()
    g = doc["measured"]["gpu_ms_per_token"]
    assert doc["still_short_of_60_by_ms"] == pytest.approx(g - 1000.0 / 60.0, abs=5e-4)
    assert doc["still_short_of_71_by_ms"] == pytest.approx(g - 1000.0 / 71.0, abs=5e-4)


def test_identity_is_cited_not_reproven():
    doc = a.build()
    assert doc["identity_verified_separately"]["receipt"] == a.VERIFIED_REL
    assert "does not re-prove identity" in doc["identity_verified_separately"]["note"]

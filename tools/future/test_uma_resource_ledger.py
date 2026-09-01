"""Memory is read live, and the peak is predicted before a load, not after.

The guarded failure is a scheduler that discovers OOM by allocating. Every
reading here comes from vm_stat and sysctl, and an unavailable reading REFUSES
rather than defaulting to a number that would admit everything.
"""
from __future__ import annotations

import pytest

from tools.future import uma_resource_ledger as ul


def test_memory_is_read_live_and_is_internally_consistent():
    m = ul.memory()
    assert m["total_bytes"] > 0
    assert m["free_bytes"] <= m["reclaimable_bytes"] <= m["total_bytes"]
    assert m["admissible_bytes"] <= m["reclaimable_bytes"]


def test_headroom_is_actually_withheld():
    m = ul.memory()
    assert m["admissible_bytes"] < m["reclaimable_bytes"], \
        "reserving headroom must reduce what is admissible"
    assert m["headroom_reserved_gb"] > 0
    assert "option value" in m["headroom_is_reserved_because"]


def test_an_unavailable_reading_refuses_rather_than_admitting_everything(monkeypatch):
    def boom(*a, **k):
        raise OSError("no vm_stat here")
    monkeypatch.setattr(ul.subprocess, "run", boom)
    with pytest.raises(ul.ResourceRefused, match="vm_stat is unavailable"):
        ul.memory()


def test_an_unreadable_volume_refuses(monkeypatch):
    monkeypatch.setattr(ul, "LAKE_VOLUME", "/no/such/volume")
    with pytest.raises(ul.ResourceRefused, match="is not readable"):
        ul.disk()


def test_a_specimen_larger_than_memory_is_refused_before_the_load():
    p = ul.predict_peak(10 ** 13)
    assert p["fits"] is False
    assert p["exceeds_total_memory"] is True
    assert p["margin_gb"] < 0


def test_a_small_specimen_is_admitted():
    p = ul.predict_peak(1_000_000_000)
    assert p["fits"] is True
    assert p["exceeds_total_memory"] is False


def test_source_bytes_is_reported_as_two_bounds_not_an_answer():
    p = ul.predict_peak(1_000_000_000)
    assert "OVERSTATE what a read costs" in p["source_bytes_is_an_upper_bound_on_the_load"]
    assert "at least this much" in p["and_a_lower_bound_on_the_executor"]
    assert "Neither bound is the answer" in p["and_a_lower_bound_on_the_executor"]


def test_most_of_the_sealed_library_does_not_fit():
    b = ul.build()
    assert b["n_sealed_that_do_not_fit"] > 0
    assert b["n_sealed_exceeding_total_memory"] > 0, \
        "at least one sealed specimen exceeds total memory outright"


def test_the_residency_score_prices_a_measured_reload():
    r = ul.residency_score(source_bytes=61_090_000_000, expected_reuses=3,
                           scientific_priority=1.0)
    assert r["reload_seconds"] > 0
    assert "measured" in r["the_reload_cost_is_measured_not_assumed"]
    assert "not from a nominal bandwidth" in r["the_reload_cost_is_measured_not_assumed"]


def test_more_reuses_raise_residency_value_and_more_bytes_lower_it():
    a = ul.residency_score(source_bytes=10 ** 10, expected_reuses=1,
                           scientific_priority=1.0)
    b = ul.residency_score(source_bytes=10 ** 10, expected_reuses=5,
                           scientific_priority=1.0)
    c = ul.residency_score(source_bytes=10 ** 11, expected_reuses=1,
                           scientific_priority=1.0)
    assert b["residency_value"] > a["residency_value"]
    assert c["residency_value"] < b["residency_value"]


def test_the_residency_formula_is_declared_not_frozen():
    r = ul.residency_score(source_bytes=10 ** 10, expected_reuses=1,
                           scientific_priority=1.0)
    assert "declines to freeze" in r["not_frozen"]
    assert "replaced when eviction data exists" in r["not_frozen"]


def test_loads_and_downloads_may_overlap_on_measured_evidence():
    r = ul.build()["lane_overlap_rule"]
    assert "SPECIMEN_LOAD_COST" in r["measured_in"]
    assert "1.23x" in r["rule"] and "142x" in r["rule"]
    assert "Do not serialize a load behind a download" in r["rule"]


def test_the_lease_rule_records_the_supervisor_lesson():
    p = ul.build()["protected_lease_overrides"]
    assert "EXCEPTION window" in p["rule"]
    assert "supervisor must be suspended first" in p["rule"]


def test_the_module_admits_it_cannot_see_the_metal_working_set():
    w = ul.build()["what_this_does_not_know"]
    assert "Metal's working set" in w
    assert "does not pretend to decide the marginal ones" in w

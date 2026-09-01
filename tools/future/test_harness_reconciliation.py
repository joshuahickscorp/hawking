"""G125 tests: the decomposition must be arithmetic, and the gate must be able to say NO."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness_reconciliation as hr  # noqa: E402


def test_the_wall_gap_is_exactly_the_gpu_gap_plus_the_host_gap():
    d = hr.decomposition()
    assert abs(d["gpu_component_ms"] + d["host_component_ms"]
               - d["wall_disagreement_ms"]) < 5e-4


def test_host_share_and_gpu_gap_are_computed_from_the_receipts_not_hardcoded():
    h = hr.harnesses()
    c, l = h["canonical"], h["lease"]
    d = hr.decomposition()
    assert d["gpu_component_ms"] == pytest.approx(c["gpu_ms"] - l["gpu_ms"], abs=5e-4)
    assert d["host_component_ms"] == pytest.approx(c["host_ms"] - l["host_ms"], abs=5e-4)


def test_host_ms_is_derived_never_read_from_a_field():
    for v in hr.harnesses().values():
        assert v["host_ms"] == pytest.approx(v["wall_ms"] - v["gpu_ms"], abs=5e-4)


def test_the_gpu_halves_actually_agree_on_the_real_receipts():
    d = hr.decomposition()
    assert d["gpu_halves_agree"] is True
    assert d["gpu_relative_difference"] <= hr.GPU_AGREEMENT_REL
    assert d["gap_is_host_borne"] is True


def test_a_disagreeing_gpu_half_refuses_to_promote(monkeypatch):
    """The gate must be capable of NO, or it is not a gate."""
    real = hr.harnesses()
    real["lease"]["gpu_ms"] = real["canonical"]["gpu_ms"] - 2.0
    real["lease"]["host_ms"] = round(
        real["lease"]["wall_ms"] - real["lease"]["gpu_ms"], 4)
    monkeypatch.setattr(hr, "harnesses", lambda: real)
    out = hr.what_is_promotable()
    assert out["gpu_absolute"] == "NOT_PROMOTABLE"
    assert "instrument conflict" in out["why"]


def test_wall_tps_is_never_a_single_number():
    out = hr.what_is_promotable()
    assert out["wall_tps"] == "RANGE_ONLY"
    lo, hi = out["wall_tps_range"]
    assert lo < hi
    assert "decode_wall_tps" not in out


def test_promoted_gpu_ms_lies_between_the_two_harnesses():
    h, out = hr.harnesses(), hr.what_is_promotable()
    lo, hi = sorted([h["canonical"]["gpu_ms"], h["lease"]["gpu_ms"]])
    assert lo <= out["decode_gpu_ms_per_token"] <= hi


def test_a_lease_that_timed_a_different_arm_is_refused(monkeypatch):
    """Differencing two harnesses that timed different arms would call a real
    lever instrument noise. That must refuse, not average."""
    lease = json.loads((hr.REPO / hr.LEASE_REL).read_text())
    lease["measured"]["arm"] = "baseline"
    real_load = hr._load
    monkeypatch.setattr(
        hr, "_load",
        lambda rel: lease if rel == hr.LEASE_REL else real_load(rel))
    with pytest.raises(hr.ReconciliationRefused, match="DIFFERENT arms"):
        hr.harnesses()


def test_a_missing_input_refuses_rather_than_defaulting(monkeypatch):
    monkeypatch.setattr(hr, "CANON_REL", "receipts/future/NO_SUCH_BUDGET.json")
    with pytest.raises(hr.ReconciliationRefused, match="not on disk"):
        hr.harnesses()


def test_the_build_claims_no_new_hardware_number():
    doc = hr.build()
    assert doc["evidence_class"] == "DERIVED_FROM_SEALED_RECEIPTS"
    assert "No GPU lease" in doc["no_new_measurement"]
    assert set(doc["inputs"]) == {hr.CANON_REL, hr.LEASE_REL}


def test_the_original_refusal_is_preserved_not_overturned():
    out = hr.what_is_promotable()
    assert "still holds" in out["the_original_refusal_was_right"]

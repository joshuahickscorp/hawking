"""G126 tests: a gate that cannot refuse is a rubber stamp."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lever_promotion_gate as g  # noqa: E402


def _patch(monkeypatch, rel, mutate):
    """Replace one input receipt with a mutated copy, leave the rest real."""
    doc = json.loads((g.REPO / rel).read_text())
    mutate(doc)
    real = g._load
    monkeypatch.setattr(g, "_load", lambda r: doc if r == rel else real(r))


def test_all_seven_preconditions_hold_on_the_real_receipts():
    checks = g.preconditions()
    assert len(checks) == 7
    assert [c["id"] for c in checks if not c["holds"]] == []


def test_the_live_decision_is_licensed():
    d = g.decision()
    assert d["verdict"] == "LICENSED_TO_FLIP_THE_DEFAULTS"
    assert d["n_preconditions"] == 7


def test_a_contaminated_window_refuses(monkeypatch):
    _patch(monkeypatch, g.LEASE_REL,
           lambda d: d["lease"].__setitem__("evidence_class", "SELF_MEASURED_DIRTY"))
    d = g.decision()
    assert d["verdict"] == "REFUSED"
    assert d["defaults_unchanged"] is True
    assert "LEASE_EVIDENCE_CLASS_IS_PROTECTED_ABSOLUTE" in d["failed_preconditions"]


def test_a_nonzero_fallback_refuses(monkeypatch):
    _patch(monkeypatch, g.LEASE_REL,
           lambda d: d["measured"].__setitem__("fallbacks", 1))
    assert g.decision()["verdict"] == "REFUSED"


def test_a_non_token_identical_arm_refuses(monkeypatch):
    _patch(monkeypatch, g.LEASE_REL,
           lambda d: d["measured"].__setitem__("token_identical", False))
    assert g.decision()["verdict"] == "REFUSED"


def test_an_unreconciled_harness_gap_refuses(monkeypatch):
    _patch(monkeypatch, g.RECON_REL,
           lambda d: d["what_is_promotable"].__setitem__(
               "gpu_absolute", "NOT_PROMOTABLE"))
    d = g.decision()
    assert d["verdict"] == "REFUSED"
    assert "THE_HARNESS_DISAGREEMENT_IS_NOT_AN_INSTRUMENT_CONFLICT" \
        in d["failed_preconditions"]


def test_a_lease_control_that_is_not_widen_f4_refuses(monkeypatch):
    """If the control were the 628 incumbent, the two savings WOULD add and the
    arithmetic below would be wrong. The gate must notice."""
    _patch(monkeypatch, g.LEASE_REL,
           lambda d: d["measured"].__setitem__("arm", "baseline"))
    d = g.decision()
    assert d["verdict"] == "REFUSED"
    assert "WIDEN_F4_IS_THE_LEASE_CONTROL_NOT_AN_ADDEND" in d["failed_preconditions"]


def test_a_missing_input_refuses_rather_than_defaulting(monkeypatch):
    monkeypatch.setattr(g, "RECON_REL", "receipts/future/NO_SUCH.json")
    with pytest.raises(g.PromotionRefused, match="fake completion"):
        g.preconditions()


def test_the_savings_are_not_summed():
    dc = g.the_double_count_this_refuses()
    lease = json.loads((g.REPO / g.LEASE_REL).read_text())
    assert dc["correct_saving_ms"] == float(lease["measured"]["gpu_ms_saved"])
    assert dc["tempting_wrong_sum_ms"] > dc["correct_saving_ms"]
    assert g.decision()["ms_saved"] == dc["correct_saving_ms"]


def test_the_gate_does_not_claim_promotion(monkeypatch):
    d = g.decision()
    assert d["verdict"] != "PROMOTED"
    assert "post-flip" in d["what_this_does_not_license"]


def test_every_named_default_points_at_a_selector_that_exists():
    for spec in g.DEFAULTS:
        src = (g.REPO / spec["file"]).read_text()
        leaf = spec["selector"].split("::")[-1]
        assert leaf in src, f"{spec['selector']} not found in {spec['file']}"


def test_the_gap_to_60_is_computed_from_the_measured_arm():
    d = g.decision()
    assert d["still_short_of_60_by_ms"] == pytest.approx(
        d["sealed_default_after_ms"] - 1000.0 / 60.0, abs=5e-4)
    assert d["still_short_of_71_by_ms"] > d["still_short_of_60_by_ms"]


def test_no_hardware_number_is_minted_here():
    doc = g.build()
    assert doc["evidence_class"] == "DERIVED_FROM_SEALED_RECEIPTS"
    assert "no GPU lease was taken" in doc["no_new_measurement"]

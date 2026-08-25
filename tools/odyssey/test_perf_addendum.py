"""G005 pins."""
import json
from pathlib import Path

import pytest

RH = Path(__file__).resolve().parents[2] / "receipts/headless"
A = RH / "QWEN_PERFORMANCE_ADDENDUM.json"
Q = RH / "QWEN_PERFORMANCE_QUALIFICATION.json"
pytestmark = pytest.mark.skipif(not A.is_file() or not Q.is_file(),
                                reason="G005 receipts not built")


def add():
    return json.load(open(A))


def test_active_bytes_are_recomputed_from_payload_not_the_design_constant():
    """Both bodies publish an identical design active_bytes_per_token despite a 1.29 GB
    payload difference. Physical figures must not inherit it."""
    d = add()
    v = d["physical_vectors"]
    clean = v["clean-2.60"]["ARTIFACT_PHYSICAL_active_bytes_per_token"]
    va = v["variantA-2.98"]["ARTIFACT_PHYSICAL_active_bytes_per_token"]
    assert clean != va, "two bodies of different size share an active-bytes figure"
    assert va > clean


def test_the_frozen_constant_is_reported_as_a_divergence_not_hidden():
    d = add()
    va = d["physical_vectors"]["variantA-2.98"]
    assert va["design_matches_physical"] is False
    assert va["design_vs_physical_rel_error"] > 0.1


def test_a_tiny_relative_error_is_not_reported_as_a_mismatch():
    """clean's design figure is off by 1.5e-5. Calling that the same kind of failure as
    variantA's 13.5% would make the flag useless."""
    d = add()
    c = d["physical_vectors"]["clean-2.60"]
    assert c["design_vs_physical_rel_error"] < 1e-4
    assert c["design_matches_physical"] is True


def test_model_reachable_roof_is_per_body_and_below_the_device_roof():
    d = add()
    roofs = {k: v.get("RUNTIME_MEASURED_model_reachable_gb_s")
             for k, v in d["physical_vectors"].items() if v.get("available")}
    assert len(roofs) >= 2
    assert len(set(roofs.values())) == len(roofs), "a roof was copied between bodies"
    for r in roofs.values():
        assert 0 < r < 778.8, "model-reachable roof must sit under DEVICE_MEASURED_SUSTAINED"


def test_concurrency_used_paired_reps_and_publishes_spread():
    """A single Metal sweep is page-cache confounded; two of them disagreed by 33%."""
    d = add()["concurrency"]
    assert "ALTERNATING PAIRED" in d["method"]
    for c, v in d["levels"].items():
        assert len(v["reps"]) >= 3, f"c{c} has too few reps to show a spread"
        assert "spread_pct" in v


def test_equilibrium_is_the_highest_REPRODUCIBLE_level_not_the_highest_median():
    d = add()["concurrency"]
    eq, peak = int(d["equilibrium_concurrency"]), int(d["peak_median_level"])
    assert d["levels"][str(eq)]["spread_pct"] < 15
    if peak != eq:
        assert d["levels"][str(peak)]["spread_pct"] >= 15, (
            "a level with a higher median AND a tight spread should have been the "
            "equilibrium")


def test_the_refuted_single_sweep_is_recorded_not_deleted():
    d = add()["concurrency"]
    s = d["superseded_single_sweeps"]
    assert "refute" in s["why_discarded"]
    assert s["sweep_2"]["c4"] < s["sweep_2"]["c2"], "the refuted claim is not preserved"
    assert d["levels"]["4"]["median_aggregate_tps"] > d["levels"]["2"]["median_aggregate_tps"]


def test_collapse_is_attributed_to_a_measured_mechanism():
    d = add()["concurrency"]["collapse"]
    assert d["observed_at"], "no collapse level recorded"
    assert "working set" in d["mechanism"]


def test_wus_per_hour_is_zero_because_the_bodies_are_capability_dead():
    d = add()["verified_wus_per_hour"]
    assert d["value"] == 0.0
    assert "3/43" in d["basis"]
    assert "G039" in d["canonical_definition_pending"]


def test_variantA_short_generation_anomaly_is_declared():
    """Its TPOT came from 175 sampled steps against 285 for the others."""
    d = add()["anomaly_variantA_short_generation"]
    assert "2 tokens" in d["observed"]
    assert "NOT comparable" in d["consequence"]


def test_qualification_no_longer_claims_a_finally_block_resume():
    q = json.load(open(Q))
    m = q["protected_window"]["io_paused_for_the_window"]["mechanism"]
    assert "finally block" not in m
    assert "protected_window.py" in m

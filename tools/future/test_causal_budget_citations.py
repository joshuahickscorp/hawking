"""The 71 ledger must read its numbers, not remember them.

G072: "the ledger recomputes from landed receipts rather than from hand-entered
numbers, and a test proves it refuses to sum overlapping levers."

Every constant used to be a literal with a receipt PATH beside it. That is a
citation, not a recomputation: when the receipt moved, the literal stayed and
nothing said so. One drift was already sitting there - the ledger carried
was_worth_tps 1.24 while WALL_GPU_RECONCILIATION.derived says 1.214.
"""
from __future__ import annotations

import json

import pytest

from tools.future import causal_budget_71 as cb


def test_every_citation_resolves_from_disk():
    got = cb.resolve_all()
    assert len(got) == len(cb.CITATIONS)
    assert got["host_gap_ms"] == pytest.approx(0.9894, abs=1e-9)
    assert got["group_size_1024_bytes"] == 1_002_700_800


def test_drift_raises_rather_than_keeping_the_literal():
    with pytest.raises(cb.CitationDrift):
        cb.resolve(
            "receipts/future/WALL_GPU_RECONCILIATION.json",
            ["derived", "host_gap_ms_per_token"],
            0.5,
        )


def test_missing_receipt_raises_and_never_falls_back():
    with pytest.raises(cb.CitationMissing):
        cb.resolve("receipts/future/NO_SUCH_RECEIPT.json", ["x"], 1.0)


def test_missing_field_raises():
    with pytest.raises(cb.CitationMissing):
        cb.resolve(
            "receipts/future/WALL_GPU_RECONCILIATION.json",
            ["derived", "not_a_field"],
            1.0,
        )


def test_selector_must_match_exactly_one_row():
    with pytest.raises(cb.CitationMissing):
        cb.resolve(
            "receipts/future/MLP_AUXILIARY_INFORMATION.json",
            ["group_size_curve", {"group_size": 999999}, "bytes_eliminated_vs_incumbent"],
            0.0,
        )


def test_non_numeric_citation_is_refused():
    with pytest.raises(cb.CitationMissing):
        cb.resolve("receipts/future/MLP_AUXILIARY_INFORMATION.json", ["schema"], 1.0)


def test_build_cannot_emit_without_resolving_citations():
    d = cb.build()
    assert set(d["citations_resolved"]) == {c["id"] for c in cb.CITATIONS}
    assert len(d["citations"]) == len(cb.CITATIONS)
    json.dumps(d)  # the receipt must stay serialisable


def test_the_drift_that_was_actually_there_is_gone():
    """was_worth_tps 1.24 was never in the receipt; it says 1.214."""
    row = next(r for r in cb.REFUTED_LEVERS if r["id"] == "eliminate_all_host_gap")
    assert row["was_worth_tps"] == 1.214
    assert cb.resolve_all()["host_gap_worth_tps"] == pytest.approx(1.214, abs=1e-9)


def test_the_residual_is_the_same_run_remainder_not_a_cross_receipt_subtraction():
    """0.321 was 0.095 of remainder plus 0.226 of trace overhead and variation.

    I computed the residual by subtracting ORGAN_BANDWIDTH's covered 27.733 from
    WALL_GPU_RECONCILIATION's 28.054 and called the difference unattributed GPU
    time. Different runs, and the organ run has the region trace ON at a measured
    1.8% of GPU time. The honest remainder comes from the run that measured both
    the parts and the total.
    """
    r = cb.causal_residual()
    assert r["gpu_residual_ms"] == pytest.approx(0.095, abs=1e-6)
    assert cb.UNATTRIBUTED_GPU_MS == pytest.approx(r["gpu_residual_ms"], abs=1e-9)
    # same-run identity: covered + remainder == the traced total
    assert r["organ_ms_covered_in_receipt"] + r["gpu_residual_ms"] == pytest.approx(
        r["traced_gpu_ms"], abs=1e-6
    )


def test_the_traced_untraced_gap_is_labelled_overhead_not_physics():
    r = cb.causal_residual()
    assert r["traced_vs_untraced_ms"] == pytest.approx(0.226, abs=1e-3)
    assert r["trace_overhead_pct"] == pytest.approx(1.8, abs=1e-6)
    assert r["gpu_residual_ms"] < r["traced_vs_untraced_ms"], (
        "if the remainder ever exceeds the run-to-run gap, the comparison is "
        "no longer dominated by overhead and this framing must be revisited"
    )


def test_the_remainder_is_named_not_miscellaneous():
    """G072: the residual must not stay a miscellaneous bucket."""
    r = cb.causal_residual()
    named = r["gpu_residual_is_named_not_mysterious"]
    for part in ("norms", "embedding row", "A_log", "dt_bias"):
        assert part in named, f"the remainder does not name {part}"


def test_host_gap_is_still_a_subtraction():
    r = cb.causal_residual()
    assert r["wall_ms"] - r["untraced_gpu_ms"] == pytest.approx(r["host_gap_ms"], abs=1e-9)


def test_the_ladder_baseline_moved_to_the_measured_current_body():
    """G072: when a win lands, the baseline MOVES. It has."""
    b = cb.causal_residual()["baseline_moved"]
    assert b["current_body_ms"] == pytest.approx(27.2896, abs=1e-3)
    assert b["current_body_tps"] == pytest.approx(36.644, abs=1e-2)
    assert b["ms_removed_since"] > 1.0
    # and it must say what did NOT move with it
    assert "PER-ORGAN ms in ORGANS are from the pre-widen_f4" in b[
        "what_is_still_from_the_old_census"
    ]


def test_organs_do_not_explain_all_of_traced_gpu_time():
    r = cb.causal_residual()
    assert r["organs_explain_of_traced_gpu"] < 1.0, "zero would mean fitted, not measured"
    assert r["organs_explain_of_traced_gpu"] > 0.99, "coverage is 99.97% of bytes"


def test_the_next_prey_is_not_the_remainder():
    """ORGAN_BANDWIDTH's finding is that the loss is uniform, not localised."""
    prey = cb.causal_residual()["next_prey"]
    assert "no hot organ" in prey
    assert "341.9" in prey and "360.0" in prey


def test_every_lever_is_billed_at_its_own_stream_rate():
    """The organ average was never a byte-class rate.

    ECONOMICS_CALIBRATION dropped fractions of each stream and timed it:
    codes_keep_50 is faster than 2*MAD, aux_keep_50 is NOT. Billing an auxiliary
    byte at the MLP organ average credited quantize_aux_u8 with 1.99 TPS and
    group_size_256 with 3.08 TPS. Both are 0.00.
    """
    for lever in cb.BYTE_LEVERS:
        assert "stream_class" in lever, lever["id"]
        assert lever["stream_class"] in cb.STREAM_MS_PER_GB
    aux = [l for l in cb.BYTE_LEVERS if l["stream_class"] == "broadcast_aux"]
    assert aux, "the auxiliary levers must stay on the record, priced at zero"
    for lever in aux:
        assert cb.lever_ms_saved(lever) == 0.0, f"{lever['id']} is credited time"


def test_the_code_body_is_the_only_priced_byte_lever():
    codes = [l for l in cb.BYTE_LEVERS if l["stream_class"] == "weight_codes"]
    assert codes
    assert cb.lever_ms_saved(codes[0]) == pytest.approx(0.152, abs=1e-3)


def test_the_refuted_group_size_levers_are_kept_and_labelled():
    """Deleting them would hide what was once claimed for them."""
    by = {l["id"]: l for l in cb.BYTE_LEVERS}
    for cid in ("group_size_256", "group_size_1024"):
        assert "CAPABILITY_REFUTED" in by[cid]["status"]
        assert cb.lever_ms_saved(by[cid]) == 0.0


def test_no_byte_lever_rung_beats_the_demonstrated_regime_by_more_than_the_floor():
    doc = cb.build()
    demo = next(r["ms"] for r in doc["ladder"] if r["rung"].endswith("497.4 GB/s"))
    for row in doc["ladder"]:
        if not row["rung"].startswith("demonstrated regime + "):
            continue
        assert demo - row["ms"] <= 0.16, f"{row['rung']} claims {demo - row['ms']} ms"

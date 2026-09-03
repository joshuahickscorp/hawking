"""Unit tests for Q80 coherence-deep analysis (no GPU, no weights)."""
from __future__ import annotations

from lab.operators.q80_coherence_deep_probe import (
    REQUIRED_REVERSE_STRING_IDS,
    analyze_drift,
    complete_bpw,
    decide_verdict,
    summarize_organs,
)


def _layer(i: int, mixed_rel: float, nulls: list[float]) -> dict:
    return {
        "layer": i,
        "in_span": True,
        "true_residual_growth": 1.05,
        "mixed": {
            "last_token_rel_l2": mixed_rel,
            "last_token_cosine": 0.99,
        },
        "nulls": [
            {"seed": 20260816 + k, "metrics": {"last_token_rel_l2": v}}
            for k, v in enumerate(nulls)
        ],
    }


def test_complete_bpw_identity():
    got = complete_bpw(1.190335440839333, 8.0)
    assert abs(got - 1.3924671422461694) < 1e-12


def test_analyze_full_depth_no_extrapolation_verdict_fields():
    layers = [_layer(i, 0.2 * (1.01**i), [0.18, 0.19, 0.21, 0.20, 0.22]) for i in range(48)]
    analysis = analyze_drift({"result": {"layers": layers, "logits": {}}}, {}, {"gate_proj": {"frac_underdetermined": 0.9}})
    assert analysis["n_measured_layers"] == 48
    assert "diagnostic_only_4layer_extrapolation_at_48" in analysis
    assert analysis["windows"]["rel_l2_at_47"] == analysis["span_end_mixed_rel_l2"]
    assert len(analysis["null_last_token_rel_l2_mean"]) == 48


def test_not_separated_when_mixed_inside_null_envelope():
    layers = [_layer(i, 0.20, [0.18, 0.19, 0.21, 0.20, 0.22]) for i in range(48)]
    analysis = analyze_drift({"result": {"layers": layers, "logits": {}}}, {}, {"gate_proj": {}})
    assert analysis["separated_from_null"] is False


def test_separated_when_mixed_above_null_max():
    layers = [_layer(i, 0.50, [0.18, 0.19, 0.21, 0.20, 0.22]) for i in range(48)]
    analysis = analyze_drift({"result": {"layers": layers, "logits": {}}}, {}, {"gate_proj": {}})
    assert analysis["separated_from_null"] is True


def test_generation_required_ids_is_go():
    decision = decide_verdict(
        {"n_measured_layers": 48, "span_end_mixed_rel_l2": 2.0, "mixed_geo_growth_full_depth": 1.2, "separated_from_null": True},
        {"generated_token_ids": REQUIRED_REVERSE_STRING_IDS, "generated_text": "def reverse"},
    )
    assert decision["verdict"] == "GO"


def test_gibberish_generation_is_nogo():
    decision = decide_verdict(
        {"n_measured_layers": 48, "span_end_mixed_rel_l2": 0.1, "mixed_geo_growth_full_depth": 1.0, "separated_from_null": True},
        {"generated_token_ids": [1, 1, 1, 1], "generated_text": "aaaa"},
    )
    assert decision["verdict"] == "NO_GO"


def test_summarize_organs_flags_clamps():
    rows = [
        {
            "hgravs_rank_clamped": True,
            "down_cold_left_bf16": False,
            "down_bpw": 0.9,
            "gate_bpw": 1.1,
            "up_bpw": 1.2,
            "n_fit_rows": 40,
        },
        {
            "hgravs_rank_clamped": False,
            "down_cold_left_bf16": True,
            "down_bpw": 16.0,
            "gate_bpw": 1.1,
            "up_bpw": 1.2,
            "n_fit_rows": 0,
        },
    ]
    meta = summarize_organs(rows)
    assert meta["hgravs_rank_clamped"] == 1
    assert meta["down_cold_left_bf16"] == 1
    assert meta["organs_rows_lt_160"] == 2

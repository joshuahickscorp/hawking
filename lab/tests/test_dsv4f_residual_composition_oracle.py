"""Calibration and closed-form tests for the DSV4F residual-composition oracle."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from lab.operators.dsv4f_residual_composition_oracle import (
    COMPOSITION_FLOOR,
    DEFAULT_ACTIVATIONS,
    DEFAULT_JSON,
    N_LAYERS_DSV4F,
    N_LAYERS_Q30,
    analyze_hidden,
    break_even_organ_cosine,
    build_receipt,
    closed_form_break_even_constant_r,
    make_constant_gain_stream,
    make_orthogonal_increment_stream,
    naive_necessary_cosine,
    naive_product,
    residual_identity_layer_cosine,
    residual_identity_product,
    stream_metrics,
)
from lab.receipts import verify


def test_naive_screens_match_closed_form() -> None:
    assert naive_necessary_cosine(N_LAYERS_DSV4F) == 0.5 ** (1.0 / 43)
    assert naive_necessary_cosine(N_LAYERS_Q30) == 0.5 ** (1.0 / 48)
    assert naive_product(0.80, 43) == 0.80**43
    assert naive_product(0.84, 43) == 0.84**43
    assert naive_necessary_cosine(43) == pytest.approx(0.9840095252215965)
    assert naive_necessary_cosine(48) == pytest.approx(0.9856631986401876)


def test_identity_stream_gain_is_exactly_one() -> None:
    rng = np.random.default_rng(0)
    hidden = make_constant_gain_stream(
        gain=1.0, n_states=8, n_seq=4, dim=16, rng=rng
    )
    metrics = stream_metrics(hidden)
    assert np.allclose(metrics["gains"], 1.0)
    assert np.allclose(metrics["increment_ratios"], 0.0)
    assert residual_identity_product(0.80, metrics["increment_ratios"].mean(axis=1)) == 1.0
    assert break_even_organ_cosine(metrics["increment_ratios"].mean(axis=1)) == 0.0


def test_expansive_1_2_stream_matches_closed_form() -> None:
    rng = np.random.default_rng(1)
    hidden = make_constant_gain_stream(
        gain=1.2, n_states=8, n_seq=4, dim=16, rng=rng
    )
    metrics = stream_metrics(hidden)
    assert np.allclose(metrics["gains"], 1.2)
    assert np.allclose(metrics["increment_ratios"], 0.2)
    assert np.allclose(metrics["alignments"], 1.0)
    # Conservative α=0 bound uses r only.
    r = 0.2
    n = 43
    expected_be = closed_form_break_even_constant_r(r, n)
    ratios = [r] * n
    got_be = break_even_organ_cosine(ratios)
    assert got_be == pytest.approx(expected_be, rel=1e-9, abs=1e-12)
    assert residual_identity_product(expected_be, ratios) == pytest.approx(
        COMPOSITION_FLOOR, rel=1e-9, abs=1e-12
    )
    # Layer factor itself matches (1 + c r^2) / (1 + r^2).
    for c in (0.75, 0.84, 0.99):
        got = residual_identity_layer_cosine(c, r, alignment=0.0)
        want = (1.0 + c * r * r) / (1.0 + r * r)
        assert got == pytest.approx(want, rel=1e-12, abs=1e-15)


def test_orthogonal_increment_gain_and_break_even_match_closed_form() -> None:
    rng = np.random.default_rng(2)
    r = 0.4
    hidden = make_orthogonal_increment_stream(
        increment_ratio=r, n_states=6, n_seq=5, dim=32, rng=rng
    )
    metrics = stream_metrics(hidden)
    expected_gain = math.sqrt(1.0 + r * r)
    assert np.allclose(metrics["gains"], expected_gain, rtol=1e-12, atol=1e-12)
    assert np.allclose(metrics["increment_ratios"], r, rtol=1e-12, atol=1e-12)
    assert np.allclose(metrics["alignments"], 0.0, atol=1e-12)
    n = 43
    expected_be = (COMPOSITION_FLOOR ** (1.0 / n) * (1.0 + r * r) - 1.0) / (r * r)
    ratios = [r] * n
    assert break_even_organ_cosine(ratios) == pytest.approx(expected_be, rel=1e-9, abs=1e-12)
    assert residual_identity_product(expected_be, ratios) == pytest.approx(
        COMPOSITION_FLOOR, rel=1e-9, abs=1e-12
    )
    assert residual_identity_product(0.80, ratios) == pytest.approx(
        ((1.0 + 0.80 * r * r) / (1.0 + r * r)) ** n, rel=1e-12, abs=1e-15
    )


def test_measured_alignment_formula_recovers_parallel_and_orthogonal() -> None:
    r = 0.25
    c = 0.9
    orthogonal = residual_identity_layer_cosine(c, r, alignment=0.0)
    assert orthogonal == pytest.approx((1.0 + c * r * r) / (1.0 + r * r))
    parallel = residual_identity_layer_cosine(c, r, alignment=1.0)
    want = (1.0 + c * r) / math.sqrt(1.0 + 2.0 * c * r + r * r)
    assert parallel == pytest.approx(want)


def test_analyze_hidden_on_synthetic_expansive_stream() -> None:
    rng = np.random.default_rng(3)
    hidden = make_constant_gain_stream(
        gain=1.2, n_states=8, n_seq=3, dim=8, rng=rng
    )
    out = analyze_hidden(hidden, n_organs=7)
    assert out["gain"]["gmean"] == pytest.approx(1.2)
    assert out["gain"]["regime"] == "expansive"
    assert out["increment"]["mean"] == pytest.approx(0.2)
    assert out["naive_product_model"]["necessary_organ_cosine"] == pytest.approx(
        0.5 ** (1.0 / 7)
    )


def test_real_export_break_even_and_band_reject() -> None:
    activations = DEFAULT_ACTIVATIONS
    if not (activations / "L00.npy").is_file():
        return
    from lab.operators.dsv4f_residual_composition_oracle import load_late_hidden_stack

    hidden, _ = load_late_hidden_stack(activations)
    out = analyze_hidden(hidden, n_organs=N_LAYERS_DSV4F)
    ident = out["residual_identity_model"]
    naive = out["naive_product_model"]
    assert naive["necessary_organ_cosine"] == pytest.approx(0.5 ** (1.0 / 43))
    assert ident["c_0_80_end_to_end"] < COMPOSITION_FLOOR
    assert ident["c_0_84_end_to_end"] < COMPOSITION_FLOOR
    assert naive["c_0_80_end_to_end"] < 1e-3
    assert naive["c_0_84_end_to_end"] < 1e-3
    # Locked from the first sealed run of this instrument. Drift here means
    # the math changed, not the dump (the dump is read-only).
    assert ident["break_even_alpha_zero_n43"] == pytest.approx(0.8615764681875107, rel=1e-12)
    assert ident["c_0_80_end_to_end"] == pytest.approx(0.3639802924540813, rel=1e-12)
    assert ident["c_0_84_end_to_end"] == pytest.approx(0.44765898343805866, rel=1e-12)
    assert out["sub_1_5_static_verdict"]["family_verdict"] == "REJECT"
    assert out["geometry"]["n_unique_streams"] == 13
    assert out["gain"]["regime"] == "expansive"


def test_sealed_receipt_round_trip_if_present() -> None:
    if not DEFAULT_JSON.is_file():
        return
    payload = json.loads(DEFAULT_JSON.read_text())
    verify(payload, label="DSV4F_RESIDUAL_COMPOSITION_ORACLE")
    assert payload["schema"] == "hawking.dsv4f.residual_composition_oracle.v1"
    assert payload["claim_boundary"]["is_rejection_instrument_only"] is True
    assert payload["claim_boundary"]["cannot_promote_coherence"] is True
    assert payload["claim_boundary"]["measured_on"]["max_seq_len"] == 3
    assert payload["claim_boundary"]["measured_on"]["tokens_total"] == 96
    rebuilt = build_receipt(activations_dir=DEFAULT_ACTIVATIONS)
    assert rebuilt["residual_identity_model"]["break_even_alpha_zero_n43"] == payload[
        "residual_identity_model"
    ]["break_even_alpha_zero_n43"]
    assert rebuilt["residual_identity_model"]["c_0_80_end_to_end"] == payload[
        "residual_identity_model"
    ]["c_0_80_end_to_end"]
    assert rebuilt["residual_identity_model"]["c_0_84_end_to_end"] == payload[
        "residual_identity_model"
    ]["c_0_84_end_to_end"]

"""Parity ladder: token-id equality is necessary and NOT sufficient.

Load-bearing negatives a guard nobody has watched fail is not a guard:

  * token-id equality alone cannot yield PASS
  * argmax agreement is not logit parity
  * a byte-mismatch count is not a characterisation
  * WITHIN_TOLERANCE without MLP_ERROR_BUDGET numbers is refused
  * a candidate's verdict is the weakest applicable rung
  * fold_addqx algebra is exact over reals and not f32
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tools.future import parity_ladder as pl
from tools.future._common import HARDWARE_FIELDS, RECEIPTS
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def _f32(n: int, *, scale: float = 0.1, offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(n)
    return (rng.normal(offset, scale, size=n)).astype(np.float32)


# ---------------------------------------------------------------------------
# The obligation.
# ---------------------------------------------------------------------------


def test_token_id_equality_alone_cannot_yield_pass():
    """NEGATIVE CONTROL: matching tokens are not a candidate PASS."""
    token = pl.rung_token_ids([11, 22, 33, 44], [11, 22, 33, 44])
    assert token["verdict"] == pl.BIT_IDENTICAL
    assert token["never_sufficient"] is True
    rungs = pl.empty_rungs(dense_reason=pl.qwen38_is_dense_reason())
    rungs["token_ids"] = token
    with pytest.raises(pl.TokenIdAloneIsNotParity, match="NOT SUFFICIENT"):
        pl.judge_candidate(
            rungs,
            available={
                "intermediate_buffers": False,
                "route_ids": False,
                "hidden_state": False,
                "final_logits": False,
                "token_ids": True,
            },
        )


def test_token_ids_identical_with_differing_intermediates_is_refuse_not_pass():
    """The fold_addqx shape: tokens match, intermediates do not, no tolerance."""
    inc = _f32(64)
    cand = inc + np.float32(0.5)
    char = pl.characterize_f32(inc, cand, compared_against="unit")
    intermediates = pl.rung_intermediate_buffers({"gate": char}, tolerance=None)
    assert intermediates["verdict"] == pl.DIFFERS
    rungs = {
        "intermediate_buffers": intermediates,
        "route_ids": pl.rung_not_applicable("route_ids", pl.qwen38_is_dense_reason()),
        "hidden_state": pl.rung_not_applicable("hidden_state", "not this test"),
        "final_logits": pl.rung_not_applicable("final_logits", "not this test"),
        "token_ids": pl.rung_token_ids([1, 2, 3], [1, 2, 3]),
    }
    # hidden/logits marked available=False so N/A is honest here.
    verdict = pl.judge_candidate(
        rungs,
        available={
            "intermediate_buffers": True,
            "route_ids": False,
            "hidden_state": False,
            "final_logits": False,
            "token_ids": True,
        },
    )
    assert verdict["verdict"] == pl.REFUSE
    assert "PASS" not in verdict["verdict"]
    assert verdict["promote_to_bit_identical"] is False
    assert verdict["promote_to_default_on"] is False
    assert verdict["rungs"]["token_ids"] == pl.BIT_IDENTICAL
    assert verdict["rungs"]["intermediate_buffers"] == pl.DIFFERS


def test_argmax_agreement_alone_is_not_logit_parity():
    with pytest.raises(pl.ArgmaxIsNotParity, match="KL"):
        pl.report_logit_agreement(
            kl_nats=None, top_k_agreement=None, argmax_agreement=1.0
        )
    with pytest.raises(pl.ArgmaxIsNotParity):
        pl.report_logit_agreement(
            kl_nats=None, top_k_agreement=0.99, argmax_agreement=1.0
        )
    with pytest.raises(pl.ArgmaxIsNotParity):
        pl.rung_final_logits({"argmax_agreement": 1.0, "kl_nats": None, "top_k_agreement": None})


def test_logit_agreement_is_kl_and_topk_not_argmax():
    rng = np.random.default_rng(0)
    inc = rng.normal(size=(4, 32)).astype(np.float32)
    cand = inc.copy()
    for i in range(inc.shape[0]):
        am = int(np.argmax(inc[i]))
        cand[i] = inc[i] + rng.normal(scale=1.5, size=inc.shape[1]).astype(np.float32)
        cand[i, am] = float(inc[i, am]) + 8.0
    agr = pl.logit_agreement_from_arrays(inc, cand, k=5)
    assert agr["argmax_agreement"] == pytest.approx(1.0)
    assert agr["argmax_is_not_parity"] is True
    assert agr["kl_nats"] > 0.0
    assert "kl_nats" in agr["parity_quantities"]
    assert "top_k_agreement" in agr["parity_quantities"]
    same = pl.logit_agreement_from_arrays(inc, inc, k=5)
    assert same["kl_nats"] == pytest.approx(0.0, abs=1e-12)
    assert same["top_k_agreement"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Characterisation vs a byte count.
# ---------------------------------------------------------------------------


def test_byte_count_alone_is_not_a_characterisation():
    """The shape FOLD_ADDQX_AB stopped at is refused here."""
    with pytest.raises(pl.CountOnlyDiffRefuse, match="magnitude"):
        pl.require_characterisation(
            {
                "n_bytes_compared": 69632,
                "n_mismatch_bytes": 22309,
                "bit_identical": False,
                "first_mismatch_index": 0,
            },
            label="layer0_gate",
        )


def test_characterisation_reports_magnitude_and_cause():
    inc = _f32(256, scale=1.0)
    # 1-ULP perturbation on a handful of elements: source-order scale.
    cand = inc.copy()
    bits = cand.view(np.uint32)
    bits[:40] = bits[:40] + np.uint32(1)
    cand = bits.view(np.float32)
    char = pl.characterize_f32(inc, cand, compared_against="unit ulp")
    for field in pl.REQUIRED_CHAR_FIELDS:
        assert field in char
    assert char["n_mismatch_bytes"] > 0
    assert char["n_float_mismatch"] == 40
    assert char["max_abs"] is not None
    assert char["rel_l2"] is not None
    assert char["rel_l2"] < pl.COMFORTABLY_USABLE_REL_L2
    assert char["bit_identical"] is False
    assert char["cause"] == pl.CAUSE_SOURCE_ORDER_FMA
    assert char["samples"]
    assert char["ulp_histogram"]["eq1"] == 40


def test_large_rel_l2_is_a_different_computation():
    inc = _f32(128, scale=1.0)
    cand = inc * np.float32(4.0)
    char = pl.characterize_f32(inc, cand, compared_against="unit wrecked")
    assert char["cause"] == pl.CAUSE_DIFFERENT_COMPUTATION
    assert char["rel_l2"] > pl.ALL_LAYERS_STRUCTURED_REL_L2
    rung = pl.rung_intermediate_buffers(
        {"gate": char}, tolerance=pl.fold_addqx_tolerance()
    )
    assert rung["verdict"] == pl.DIFFERS


# ---------------------------------------------------------------------------
# Weakest rung, every rung reports, tolerance justification.
# ---------------------------------------------------------------------------


def test_every_rung_reports_its_own_verdict():
    inc = _f32(32)
    char = pl.characterize_f32(inc, inc, compared_against="unit identical")
    agr = pl.logit_agreement_from_arrays(inc[:8], inc[:8], k=5)
    rungs = {
        "intermediate_buffers": pl.rung_intermediate_buffers({"gate": char}),
        "route_ids": pl.rung_route_ids(
            None, None, organ_has_routes=False, reason_if_absent="dense"
        ),
        "hidden_state": pl.rung_hidden_state(char),
        "final_logits": pl.rung_final_logits(agr),
        "token_ids": pl.rung_token_ids([7, 8], [7, 8]),
    }
    for name in pl.RUNGS:
        assert rungs[name]["name"] == name
        assert rungs[name]["verdict"] in pl.RUNG_VERDICTS
    assert rungs["route_ids"]["verdict"] == pl.NOT_APPLICABLE
    assert rungs["intermediate_buffers"]["verdict"] == pl.BIT_IDENTICAL
    assert rungs["token_ids"]["never_sufficient"] is True


def test_weakest_rung_wins():
    inc = _f32(64, scale=1.0)
    # Tiny association-scale noise, well under 0.01 rel L2.
    cand = inc.copy()
    bits = cand.view(np.uint32)
    bits[::3] = bits[::3] + np.uint32(1)
    cand = bits.view(np.float32)
    char = pl.characterize_f32(inc, cand, compared_against="unit fma")
    tol = pl.fold_addqx_tolerance()
    assert char["cause"] == pl.CAUSE_SOURCE_ORDER_FMA
    assert char["rel_l2"] <= tol["rel_l2_bar"]
    agr = pl.logit_agreement_from_arrays(inc[:16], cand[:16], k=5)
    # Force logits to a known-good pair for the rung (KL of 16 noise dims
    # can be large); use identical logits so the weakest rung is intermediates.
    agr_same = pl.logit_agreement_from_arrays(inc[:16], inc[:16], k=5)
    hidden_same = pl.characterize_f32(inc, inc, compared_against="hidden same")
    rungs = {
        "intermediate_buffers": pl.rung_intermediate_buffers({"gate": char}, tolerance=tol),
        "route_ids": pl.rung_not_applicable("route_ids", "dense"),
        "hidden_state": pl.rung_hidden_state(hidden_same, tolerance=tol),
        "final_logits": pl.rung_final_logits(agr_same, tolerance=tol),
        "token_ids": pl.rung_token_ids([1, 1, 1], [1, 1, 1]),
    }
    cap = {
        "ran": True,
        "pass": True,
        "reason": "unit",
        "mean_kl_nats": 0.0,
        "mean_top_k_agreement": 1.0,
    }
    verdict = pl.judge_candidate(
        rungs,
        capability=cap,
        available={
            "intermediate_buffers": True,
            "route_ids": False,
            "hidden_state": True,
            "final_logits": True,
            "token_ids": True,
        },
        candidate="unit_weakest",
    )
    assert rungs["intermediate_buffers"]["verdict"] == pl.WITHIN_TOLERANCE
    assert rungs["token_ids"]["verdict"] == pl.BIT_IDENTICAL
    assert rungs["hidden_state"]["verdict"] == pl.BIT_IDENTICAL
    assert verdict["verdict"] == pl.PASS_JUSTIFIED_TOLERANCE
    assert verdict["weakest_rung"] == "intermediate_buffers"
    assert verdict["promote_to_bit_identical"] is False
    assert verdict["promote_to_default_on"] is True
    # Silence unused.
    assert agr["argmax_is_not_parity"] is True


def test_unjustified_tolerance_is_refused():
    inc = _f32(32)
    cand = inc.copy()
    bits = cand.view(np.uint32)
    bits[:4] = bits[:4] + np.uint32(1)
    cand = bits.view(np.float32)
    char = pl.characterize_f32(inc, cand, compared_against="unit")
    # Not bit-identical, no tolerance supplied → DIFFERS, not a quiet WITHIN.
    rung = pl.rung_intermediate_buffers({"gate": char}, tolerance=None)
    assert rung["verdict"] == pl.DIFFERS
    shifted = inc[:8].copy()
    shifted[0] = shifted[0] + np.float32(0.05)
    agr = pl.logit_agreement_from_arrays(inc[:8], shifted, k=5)
    assert agr["kl_nats"] > 0.0
    assert agr["kl_nats"] < pl.USABLE_KL
    # Logits agree on the usable bars but are not bit-identical: a
    # tolerance is required rather than a quiet WITHIN_TOLERANCE.
    with pytest.raises(pl.UnjustifiedTolerance):
        pl.rung_final_logits(agr, tolerance=None)


def test_tolerance_is_justified_against_mlp_error_budget():
    bars = pl.mlp_error_budget_bars()
    assert bars["all_layers_structured_tolerated_relative_l2"] == pytest.approx(0.03)
    assert bars["comfortably_usable_relative_l2"] == pytest.approx(0.01)
    assert bars["usable_last_token_kl"] == pytest.approx(0.10)
    assert bars["usable_mean_top5"] == pytest.approx(0.80)
    assert bars["argmax_is_not_parity"] is True
    tol = pl.fold_addqx_tolerance(bars)
    assert tol["rel_l2_bar"] == pytest.approx(0.01)
    assert tol["kl_bar"] == pytest.approx(0.10)
    assert "0.03" in tol["justification"]
    assert "0.01" in tol["justification"]
    assert "MLP_ERROR_BUDGET" in tol["justification"]
    assert tol["permitted_cause"] == pl.CAUSE_SOURCE_ORDER_FMA


def test_capability_spot_check_is_required_for_tolerance_pass():
    inc = _f32(64, scale=1.0)
    cand = inc.copy()
    bits = cand.view(np.uint32)
    bits[::5] = bits[::5] + np.uint32(1)
    cand = bits.view(np.float32)
    char = pl.characterize_f32(inc, cand, compared_against="unit")
    tol = pl.fold_addqx_tolerance()
    agr = pl.logit_agreement_from_arrays(inc[:8], inc[:8], k=5)
    hidden = pl.characterize_f32(inc, inc, compared_against="h")
    rungs = {
        "intermediate_buffers": pl.rung_intermediate_buffers({"gate": char}, tolerance=tol),
        "route_ids": pl.rung_not_applicable("route_ids", "dense"),
        "hidden_state": pl.rung_hidden_state(hidden, tolerance=tol),
        "final_logits": pl.rung_final_logits(agr, tolerance=tol),
        "token_ids": pl.rung_token_ids([3, 4], [3, 4]),
    }
    avail = {
        "intermediate_buffers": True,
        "route_ids": False,
        "hidden_state": True,
        "final_logits": True,
        "token_ids": True,
    }
    with pytest.raises(pl.CapabilitySpotCheckMissing):
        pl.judge_candidate(rungs, capability=None, available=avail)
    with pytest.raises(pl.CapabilitySpotCheckMissing):
        pl.judge_candidate(rungs, capability={"ran": False}, available=avail)
    refused = pl.judge_candidate(
        rungs, capability={"ran": True, "pass": False, "reason": "broke"}, available=avail
    )
    assert refused["verdict"] == pl.REFUSE
    assert refused["promote_to_default_on"] is False


def test_route_ids_are_not_applicable_on_dense_qwen38():
    rung = pl.rung_route_ids(
        None,
        None,
        organ_has_routes=False,
        reason_if_absent=pl.qwen38_is_dense_reason(),
    )
    assert rung["verdict"] == pl.NOT_APPLICABLE
    assert "dense" in rung["reason"].lower()
    assert "moe" in rung["reason"].lower()


def test_incomplete_available_surface_is_refused():
    token = pl.rung_token_ids([1], [1])
    rungs = pl.empty_rungs(dense_reason="dense")
    rungs["token_ids"] = token
    with pytest.raises(pl.IncompleteLadder, match="intermediate_buffers"):
        pl.judge_candidate(
            rungs,
            available={
                "intermediate_buffers": True,  # available but N/A
                "route_ids": False,
                "hidden_state": False,
                "final_logits": False,
                "token_ids": True,
            },
        )


# ---------------------------------------------------------------------------
# fold_addqx algebra.
# ---------------------------------------------------------------------------


def test_fold_addqx_is_exact_over_reals_and_not_f32():
    algebra = pl.fold_addqx_algebra()
    assert algebra["identity"] == pl.FOLD_ADDQX_IDENTITY
    assert algebra["over_reals"]["exact"] is True
    assert algebra["over_f32"]["matches_production"] is False
    assert algebra["over_f32"]["n_tiles_differ"] > 0
    cx = algebra["over_f32"]["counterexample"]
    assert cx["packed16"] == 65535
    assert cx["abs_err"] > 0.0
    # The cheapen receipt's counterexample is ~9.5e-7; stay in that class.
    assert cx["abs_err"] < 1e-5
    assert algebra["source_order_permits"] is True
    assert algebra["cause"] == pl.CAUSE_SOURCE_ORDER_FMA


def test_fold_addqx_cpu_unpack_matches_canonical_counterexample_class():
    x = [0.7] * 8
    prod = pl.unpack8_production(65535, 0.3, 0.1, x)
    fold = pl.unpack8_fold_addqx(65535, 0.3, 0.1, x)
    assert prod != fold
    lhs, rhs = pl.unpack8_reals(65535, 0.3, 0.1, x)
    assert abs(lhs - rhs) < 1e-12


# ---------------------------------------------------------------------------
# Receipt / selftest.
# ---------------------------------------------------------------------------


def test_selftest_watches_the_refusals_fire():
    result = pl.selftest()
    assert result["ok"] is True
    assert result["token_id_alone_raises"] is True
    assert result["argmax_alone_raises"] is True
    assert result["count_only_raises"] is True
    assert result["algebra_over_reals"] is True
    assert result["algebra_over_f32_matches"] is False


def test_build_static_receipt_does_not_promote_fold_addqx_to_bit_identical():
    doc = pl.build(raw=None, measured=False)
    assert doc["schema"] == pl.SCHEMA
    assert doc["token_id_equality_alone_cannot_pass"] is True
    assert doc["argmax_is_not_parity"] is True
    assert doc["rung_order"] == list(pl.RUNGS)
    assert doc["primitive"] in ATLAS_PRIMITIVES
    assert doc["fold_addqx"]["not_promoted_to_bit_identical"] is True
    assert "fold_addqx_ab.py" in doc["does_not_edit"][0]
    assert "q80_mixed_decode.metal" in doc["does_not_edit"][1]
    # No hardware field invented.
    blob = json.dumps(doc)
    for field in HARDWARE_FIELDS:
        # cited ms lives under a non-hardware key from FOLD_ADDQX_AB.
        assert f'"{field}":' not in blob or field not in (
            "tps",
            "gpu_ns",
            "accepted_tps",
        )


def test_capability_from_probe_refuses_argmax_only_rows():
    with pytest.raises(pl.ArgmaxIsNotParity):
        pl.capability_from_probe(
            [{"id": "x", "logits": {"argmax_agreement": 1.0}}]
        )


def test_pass_bit_identical_requires_every_applicable_rung():
    inc = _f32(16)
    char = pl.characterize_f32(inc, inc, compared_against="same")
    agr = pl.logit_agreement_from_arrays(inc, inc, k=5)
    rungs = {
        "intermediate_buffers": pl.rung_intermediate_buffers({"gate": char}),
        "route_ids": pl.rung_not_applicable("route_ids", "dense"),
        "hidden_state": pl.rung_hidden_state(char),
        "final_logits": pl.rung_final_logits(agr),
        "token_ids": pl.rung_token_ids([9], [9]),
    }
    v = pl.judge_candidate(
        rungs,
        available={
            "intermediate_buffers": True,
            "route_ids": False,
            "hidden_state": True,
            "final_logits": True,
            "token_ids": True,
        },
    )
    assert v["verdict"] == pl.PASS_BIT_IDENTICAL
    assert v["promote_to_bit_identical"] is True


def test_receipt_file_if_present_has_every_rung():
    path = RECEIPTS / "PARITY_LADDER.json"
    if not path.is_file():
        pytest.skip("PARITY_LADDER.json not written yet")
    doc = json.loads(path.read_text())
    assert doc["schema"] == pl.SCHEMA
    assert doc["token_id_equality_alone_cannot_pass"] is True
    rungs = (doc.get("fold_addqx") or {}).get("rungs")
    if rungs is None:
        pytest.skip("static receipt without live rungs")
    for name in pl.RUNGS:
        assert name in rungs
        assert rungs[name]["verdict"] in pl.RUNG_VERDICTS
    j = doc["fold_addqx"]["judgement"]
    assert j["promote_to_bit_identical"] is False
    assert j["verdict"] in pl.CANDIDATE_VERDICTS
    gate = doc["fold_addqx"].get("layer0_gate_characterisation")
    if gate:
        for field in ("max_abs", "rel_l2", "cause", "n_mismatch_bytes"):
            assert field in gate
        assert gate["n_mismatch_bytes"] != 0 or gate["bit_identical"] is True

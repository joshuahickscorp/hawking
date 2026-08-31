"""Tests for the DeltaNet multi-step authority.

A guard nobody has watched fail is not a guard. Load-bearing refusals:

1. Collapsing state/output/logit into one error raises, it is not scored.
2. A one-step-only number raises rather than returning a verdict.
3. Every named horizon is RUN or SKIPPED; silent omission raises.
4. Argmax agreement is not parity.
5. Train-set rows cannot be reported as held-out.
6. A candidate without bytes_removed AND bytes_added is not a candidate.
"""
from __future__ import annotations

import json

import pytest

from tools.future import deltanet_multistep as dnm
from tools.future import executable_economics as ee
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims
from tools.future.physical_primitives import ATLAS_PRIMITIVES


TINY_HEADS = 2
TINY_DIM = 8
TINY_VOCAB = 16
TINY_REQUIRED = (1, 4, 16)
TINY_PLUS = (64, 128, 256)


@pytest.fixture(scope="module")
def built_receipt():
    """One --build for the receipt tests. The curve is the expensive part."""
    rc = dnm.main(["--build"])
    assert rc == 0
    path = RECEIPTS / dnm.RECEIPT
    return json.loads(path.read_text())


def _tiny_seq(n_tokens: int = 32, *, split: str = "hold", prompt_id: str = "hold:00"):
    x = dnm.make_fixture_hidden(n_tokens, hidden=dnm.HIDDEN, seed=dnm.RNG_SEED)
    return {
        "prompt_id": prompt_id,
        "layer": 38,
        "split": split,
        "n_tokens": n_tokens,
        "x": x,
        "capability_domain": "plain-prose",
    }


def _tiny_bundle(n_tokens: int = 32):
    x = dnm.make_fixture_hidden(n_tokens)
    return dnm.coefficients_from_hidden(
        x, n_heads=TINY_HEADS, dim=TINY_DIM, seed=dnm.RNG_SEED
    )


def _tiny_logit():
    return dnm.logit_matrix(
        n_heads=TINY_HEADS, dim=TINY_DIM, vocab=TINY_VOCAB, seed=dnm.RNG_SEED
    )


def _econ_zero(*, status: str = "EXISTING_LEVER"):
    return {
        "bytes_removed": 0,
        "bytes_added": 0,
        "extra_flops_per_output_element": 0.0,
        "dispatch_delta": 0.0,
        "consuming_primitive": "LocalStateMachine",
        "stream_class": "activation",
        "status": status,
        "reusable_family": False,
        "high_information_falsifier": True,
        "dense_rematerialization": dnm.DIRECT_CONSUME,
    }


# ---------------------------------------------------------------------------
# Collapse / one-step / skip / argmax. The load-bearing refusals.
# ---------------------------------------------------------------------------


def test_collapsed_error_is_refused_not_scored():
    """NEGATIVE CONTROL: one combined error is not a curve."""
    with pytest.raises(dnm.CollapsedSeriesRefuse, match="three separate series"):
        dnm.require_separate_series({"horizon": 4, "error": 0.01})

    with pytest.raises(dnm.CollapsedSeriesRefuse, match="missing"):
        dnm.require_separate_series(
            {
                "horizon": 4,
                "state_error": {"relative_l2": 0.01},
            }
        )

    with pytest.raises(dnm.CollapsedSeriesRefuse, match="collapsed to float"):
        dnm.require_separate_series(
            {
                "horizon": 4,
                "state_error": 0.01,
                "output_error": 0.01,
                "logit_effect": 0.01,
            }
        )

    ok = {
        "horizon": 4,
        "state_error": {"relative_l2": 0.0, "cosine": 1.0},
        "output_error": {"relative_l2": 0.0, "cosine": 1.0},
        "logit_effect": {
            "relative_l2": 0.0,
            "cosine": 1.0,
            "argmax_agreement": 1.0,
            "argmax_is_not_parity": True,
        },
    }
    dnm.require_separate_series(ok)


def test_one_step_only_raises_rather_than_reporting_a_verdict():
    """A one-step number is not admissible. Raise, do not return NO."""
    rec = {
        "horizon": 1,
        "state_error": {"relative_l2": 1e-8, "cosine": 1.0},
        "output_error": {"relative_l2": 1e-8, "cosine": 1.0},
        "logit_effect": {
            "relative_l2": 1e-8,
            "cosine": 1.0,
            "argmax_agreement": 1.0,
            "argmax_is_not_parity": True,
        },
    }
    curve = {
        "horizons_run": [1],
        "per_horizon": [rec],
    }
    with pytest.raises(dnm.OneStepOnlyRefuse, match="not admissible") as caught:
        dnm.demand_fitted_heldout(curve, required=dnm.REQUIRED_HORIZONS)
    assert "verdict" not in str(caught.value).lower() or "REFUSED" in str(caught.value)
    assert "REFUSED" in str(caught.value)

    with pytest.raises(dnm.OneStepOnlyRefuse):
        dnm.demand_fitted_heldout({"horizons_run": [], "per_horizon": []})

    # Evaluating a 1-token sequence under the production required list also
    # refuses rather than minting FITTED_HELDOUT or MEASURED_NEGATIVE.
    seq = _tiny_seq(n_tokens=1)
    with pytest.raises(dnm.OneStepOnlyRefuse):
        dnm.evaluate_candidate(
            cand_id="one_step_probe",
            step=dnm.identity_step,
            sequences=[seq],
            n_heads=TINY_HEADS,
            dim=TINY_DIM,
            vocab=TINY_VOCAB,
            required=dnm.REQUIRED_HORIZONS,
            plus=dnm.PLUS_HORIZONS,
            economics=_econ_zero(),
            report_as="fixture",
        )


def test_skipped_horizons_are_named_not_silently_omitted():
    bundle = _tiny_bundle(n_tokens=20)
    curve = dnm.roll_curve(
        candidate_step=dnm.identity_step,
        coeffs=bundle,
        w_logit=_tiny_logit(),
        required=TINY_REQUIRED,
        plus=TINY_PLUS,
        skip_for_cost=(256,),
        n_heads=TINY_HEADS,
        dim=TINY_DIM,
    )
    named = {row["horizon"]: row for row in curve["horizons_named"]}
    for h in list(TINY_REQUIRED) + list(TINY_PLUS):
        assert h in named, f"horizon {h} silently omitted"
    assert named[1]["status"] == dnm.RUN
    assert named[4]["status"] == dnm.RUN
    assert named[16]["status"] == dnm.RUN
    assert named[64]["status"] == "SKIPPED"
    assert named[64]["reason"] == dnm.SKIPPED_INSUFFICIENT_SEQUENCE
    assert named[256]["status"] == "SKIPPED"
    assert named[256]["reason"] == dnm.SKIPPED_FOR_COST
    assert 256 not in curve["horizons_run"]

    with pytest.raises(dnm.SilentOmissionRefuse, match="neither RUN nor SKIPPED"):
        dnm.require_named_horizons(
            required=(1, 4, 16),
            plus=(),
            run=[1, 4],
            skipped=[],
        )


def test_argmax_agreement_is_not_parity():
    with pytest.raises(dnm.ArgmaxIsNotParity, match="not parity"):
        dnm.require_logit_effect({"argmax_agreement": 1.0})

    with pytest.raises(dnm.ArgmaxIsNotParity, match="argmax_is_not_parity"):
        dnm.require_logit_effect(
            {
                "relative_l2": 0.4,
                "argmax_agreement": 1.0,
                "argmax_is_not_parity": False,
            }
        )

    # A candidate that preserves argmax while drifting in logit space cannot
    # be FITTED_HELDOUT even if state and output are quiet.
    bundle = _tiny_bundle(n_tokens=20)
    drifted = dnm.roll_curve(
        candidate_step=dnm.identity_step,
        coeffs=bundle,
        w_logit=_tiny_logit(),
        required=TINY_REQUIRED,
        plus=(),
        logit_offset=7.5,
        n_heads=TINY_HEADS,
        dim=TINY_DIM,
    )
    for rec in drifted["per_horizon"]:
        assert rec["logit_effect"]["argmax_agreement"] == 1.0
        assert rec["logit_effect"]["relative_l2"] > dnm.LOGIT_BAR
        assert rec["state_error"]["relative_l2"] <= 1e-5
    verdict = dnm.demand_fitted_heldout(drifted, required=TINY_REQUIRED)
    assert verdict["status"] == dnm.MEASURED_NEGATIVE
    assert verdict["argmax_is_not_parity"] is True
    assert any("argmax_survived" in r for r in verdict["reasons"])


def test_train_set_cannot_be_reported_as_held_out():
    seq = _tiny_seq(split="train", prompt_id="train:00")
    with pytest.raises(dnm.TrainReportedAsHeldOut, match="held_out"):
        dnm.evaluate_candidate(
            cand_id="leaky",
            step=dnm.identity_step,
            sequences=[seq],
            n_heads=TINY_HEADS,
            dim=TINY_DIM,
            vocab=TINY_VOCAB,
            required=TINY_REQUIRED,
            plus=(),
            economics=_econ_zero(),
            report_as="held_out",
        )


def test_candidate_without_economics_is_refused():
    with pytest.raises(dnm.MissingEconomics, match="compression ratio") as caught:
        dnm.score_candidate_economics(
            cand_id="ratio_only",
            bytes_removed=1_000,
            bytes_added=None,
            consuming_primitive="LocalStateMachine",
            stream_class="activation",
        )
    assert caught.value.cand_id == "ratio_only"
    assert "bytes_added" in caught.value.missing

    with pytest.raises(ee.IncompleteEconomics):
        ee.score(bytes_removed=1_000_000)


# ---------------------------------------------------------------------------
# Curves on the cheap controls. Tiny geometry, real numbers.
# ---------------------------------------------------------------------------


def test_identity_control_is_flat_zero_on_all_three_series():
    bundle = _tiny_bundle(n_tokens=24)
    curve = dnm.roll_curve(
        candidate_step=dnm.identity_step,
        coeffs=bundle,
        w_logit=_tiny_logit(),
        required=TINY_REQUIRED,
        plus=(),
        n_heads=TINY_HEADS,
        dim=TINY_DIM,
    )
    assert curve["horizons_run"] == [1, 4, 16]
    assert curve["shape"]["overall"] == dnm.FLAT_ZERO
    for key in dnm.SERIES_KEYS:
        assert curve["shape"]["by_series"][key] == dnm.FLAT_ZERO
        for row in curve["series"][key]:
            assert row["relative_l2"] <= dnm.FLAT_ZERO_ABS
    for rec in curve["per_horizon"]:
        dnm.require_separate_series(rec)
        assert rec["logit_effect"]["argmax_agreement"] == 1.0
    verdict = dnm.demand_fitted_heldout(curve, required=TINY_REQUIRED)
    assert verdict["status"] == dnm.FITTED_HELDOUT
    assert verdict["one_step_only_admissible"] is False


def test_truncated_state_reports_three_series_and_a_shape():
    bundle = _tiny_bundle(n_tokens=24)
    step = dnm.make_truncated_state_step(2)
    curve = dnm.roll_curve(
        candidate_step=step,
        coeffs=bundle,
        w_logit=_tiny_logit(),
        required=TINY_REQUIRED,
        plus=(),
        n_heads=TINY_HEADS,
        dim=TINY_DIM,
    )
    assert curve["horizons_run"] == [1, 4, 16]
    for rec in curve["per_horizon"]:
        dnm.require_separate_series(rec)
        for key in dnm.SERIES_KEYS:
            assert "relative_l2" in rec[key]
        assert "argmax_agreement" in rec["logit_effect"]
        assert rec["logit_effect"]["argmax_is_not_parity"] is True
    # Three series, not one: the values are allowed to differ.
    last = curve["per_horizon"][-1]
    triple = (
        last["state_error"]["relative_l2"],
        last["output_error"]["relative_l2"],
        last["logit_effect"]["relative_l2"],
    )
    assert all(isinstance(v, float) for v in triple)
    shape = curve["shape"]["overall"]
    assert shape in {dnm.PLATEAU, dnm.COMPOUNDING, dnm.ONSET_THEN_PLATEAU, dnm.FLAT_ZERO}
    verdict = dnm.demand_fitted_heldout(curve, required=TINY_REQUIRED)
    assert verdict["status"] in {dnm.FITTED_HELDOUT, dnm.MEASURED_NEGATIVE}
    # Rank-2 of dim-8 is a real truncation; one-step quietness is not enough
    # if the curve compounds or onsets after the rank fills.
    if shape in {dnm.COMPOUNDING, dnm.ONSET_THEN_PLATEAU}:
        assert verdict["status"] == dnm.MEASURED_NEGATIVE


def test_lower_rank_transition_runs_and_names_horizons():
    basis = dnm.orthonormal_basis(TINY_DIM, 2, seed=dnm.RNG_SEED)
    bundle = _tiny_bundle(n_tokens=24)
    curve = dnm.roll_curve(
        candidate_step=dnm.make_lower_rank_transition_step(basis),
        coeffs=bundle,
        w_logit=_tiny_logit(),
        required=TINY_REQUIRED,
        plus=(64,),
        n_heads=TINY_HEADS,
        dim=TINY_DIM,
    )
    named = {row["horizon"]: row["status"] for row in curve["horizons_named"]}
    assert named[1] == dnm.RUN
    assert named[16] == dnm.RUN
    assert named[64] == "SKIPPED"
    assert curve["shape"]["overall"] in {
        dnm.PLATEAU,
        dnm.COMPOUNDING,
        dnm.ONSET_THEN_PLATEAU,
        dnm.FLAT_ZERO,
    }
    for key in dnm.SERIES_KEYS:
        assert [row["horizon"] for row in curve["series"][key]] == curve["horizons_run"]


def test_incomplete_required_horizons_cannot_be_fitted_heldout():
    bundle = _tiny_bundle(n_tokens=20)
    curve = dnm.roll_curve(
        candidate_step=dnm.identity_step,
        coeffs=bundle,
        w_logit=_tiny_logit(),
        required=dnm.REQUIRED_HORIZONS,
        plus=dnm.PLUS_HORIZONS,
        skip_for_cost=(),
        n_heads=TINY_HEADS,
        dim=TINY_DIM,
    )
    assert 1 in curve["horizons_run"]
    assert 256 not in curve["horizons_run"]
    skipped_h = {row["horizon"] for row in curve["horizons_skipped"]}
    assert 64 in skipped_h
    assert 256 in skipped_h
    with pytest.raises(dnm.IncompleteHorizonsRefuse) as caught:
        dnm.demand_fitted_heldout(curve, required=dnm.REQUIRED_HORIZONS)
    assert 64 in caught.value.missing or 256 in caught.value.missing


def test_classify_shape_distinguishes_plateau_from_compounding():
    assert dnm.classify_shape({1: 0.0, 4: 0.0, 16: 0.0, 64: 0.0}) == dnm.FLAT_ZERO
    assert dnm.classify_shape({1: 0.01, 4: 0.011, 16: 0.012, 64: 0.012}) == dnm.PLATEAU
    assert dnm.classify_shape({1: 0.01, 4: 0.05, 16: 0.2, 64: 0.8}) == dnm.COMPOUNDING
    # Exact until rank fills, then a residual that sits: the one-step lie.
    assert (
        dnm.classify_shape({1: 0.0, 4: 0.0, 16: 0.0, 64: 0.10, 128: 0.09, 256: 0.08})
        == dnm.ONSET_THEN_PLATEAU
    )
    assert dnm.shape_supports_promotion(dnm.PLATEAU) is True
    assert dnm.shape_supports_promotion(dnm.COMPOUNDING) is False
    assert dnm.shape_supports_promotion(dnm.ONSET_THEN_PLATEAU) is False

    # The truncated-S lie, as a constructed curve: perfect through 16, residual
    # at 64. One-step would have said FITTED; the authority must say no.
    recs = []
    for h, err in ((1, 0.0), (4, 0.0), (16, 0.0), (64, 0.10), (128, 0.09), (256, 0.08)):
        recs.append(
            {
                "horizon": h,
                "state_error": {"relative_l2": err, "cosine": 1.0 - err},
                "output_error": {"relative_l2": err, "cosine": 1.0 - err},
                "logit_effect": {
                    "relative_l2": err,
                    "cosine": 1.0 - err,
                    "argmax_agreement": 1.0,
                    "argmax_is_not_parity": True,
                },
            }
        )
    onset = {
        "horizons_run": [1, 4, 16, 64, 128, 256],
        "per_horizon": recs,
    }
    verdict = dnm.demand_fitted_heldout(onset, required=dnm.REQUIRED_HORIZONS)
    assert verdict["status"] == dnm.MEASURED_NEGATIVE
    assert verdict["shape"]["overall"] == dnm.ONSET_THEN_PLATEAU
    assert any("argmax_survived@64" in r for r in verdict["reasons"])


# ---------------------------------------------------------------------------
# Economics + discovery.
# ---------------------------------------------------------------------------


def test_economics_score_both_sides_and_atlas_primitive():
    claim = dnm.generated_transition_claimed_economics()
    row = dnm.score_candidate_economics(
        cand_id=dnm.GENERATED_TRANSITION,
        bytes_removed=claim["bytes_removed"],
        bytes_added=claim["bytes_added"],
        extra_flops_per_output_element=0.0,
        dispatch_delta=float(claim["dispatch_delta"]),
        consuming_primitive=str(claim["consuming_primitive"]),
        stream_class=str(claim.get("stream_class") or "weight_codes"),
        status=dnm.OPEN,
        reusable_family=True,
        high_information_falsifier=True,
    )
    assert row["bytes_removed"] == 2_139_096_960
    assert row["bytes_added"]["total"] == 4_548_560
    assert row["bytes_added_supplied"] is True
    assert row["consuming_primitive"] in ATLAS_PRIMITIVES
    assert row["organ"] == "deltanet"
    assert "compression_ratio" not in row
    # Predicted ms/token is arithmetic over cited organ times, not a bench.
    assert row["predicted_ms_saved"] > 0.0
    assert row["lane_material"]["clears"] is True


def test_truncated_state_byte_model_removes_s_and_adds_factors():
    specs = {s["id"]: s for s in dnm.cheap_control_specs()}
    tr = specs[dnm.TRUNCATED_STATE]["economics"]
    row = dnm.score_candidate_economics(
        cand_id=dnm.TRUNCATED_STATE,
        bytes_removed=tr["bytes_removed"],
        bytes_added=tr["bytes_added"],
        consuming_primitive=tr["consuming_primitive"],
        stream_class=tr["stream_class"],
        reusable_family=True,
        high_information_falsifier=True,
        status=dnm.OPEN,
    )
    assert row["bytes_removed"] == dnm.REC_STATE_RESIDENT
    assert row["bytes_added"]["total"] > 0
    assert row["bytes_added"]["total"] < row["bytes_removed"]
    assert row["consuming_primitive"] == "LocalStateMachine"


def test_discover_landed_candidates_records_absence_not_a_fake_curve():
    disc = dnm.discover_landed_candidates()
    assert "landed" in disc
    assert disc["candidate_id"] == dnm.GENERATED_TRANSITION
    assert disc["status"] in {dnm.NOT_LANDED, "LANDED"}
    assert len(disc["receipts_probed"]) == len(dnm.LANDING_RECEIPT_RELS)
    for hit in disc["receipts_probed"]:
        assert hit["source"] in {"disk", "git:HEAD", "missing", "unreadable", "git-unreadable"}


# ---------------------------------------------------------------------------
# Receipt.
# ---------------------------------------------------------------------------


def test_build_emits_sealed_receipt_with_three_series_and_the_rule(built_receipt):
    out = RECEIPTS / dnm.RECEIPT
    assert out.parent == RECEIPTS
    assert out.name == dnm.RECEIPT
    doc = built_receipt
    assert doc["schema"] == dnm.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    _assert_no_hardware_claims(doc)

    acc = doc["acceptance"]
    assert acc["one_step_only_admissible"] is False
    assert acc["argmax_is_not_parity"] is True
    assert acc["series"] == list(dnm.SERIES_KEYS)
    assert acc["required_horizons"] == list(dnm.REQUIRED_HORIZONS)
    assert acc["plus_horizons"] == list(dnm.PLUS_HORIZONS)
    assert "one-step-only number is not admissible" in acc["fitted_heldout_rule"]
    assert "FITTED_HELDOUT" in acc["fitted_heldout_rule"]
    assert acc["held_out_unit"] == "prompt_id"
    assert acc["silent_horizon_omission"] == "REFUSED"

    # Cited figures, never stored under hardware-field keys.
    cited = doc["cited"]
    assert cited["cited_deltanet_bytes"] == 2_961_659_904
    assert cited["cited_deltanet_ms"] == pytest.approx(8.227)
    assert cited["cited_token_ms"] == pytest.approx(28.722)
    assert cited["cited_resident_tps"] == pytest.approx(34.82)
    assert "tps" not in cited
    assert "token_ns" not in cited

    cap = doc["capability_residual_budget"]
    assert cap["not_a_win"] is True
    assert cap["share_of_token"] == pytest.approx(0.0028028380503877935)
    assert cap["bytes"] == 27_688_960

    ids = [c["id"] for c in doc["controls"]]
    assert ids == [dnm.IDENTITY, dnm.TRUNCATED_STATE, dnm.LOWER_RANK_TRANSITION]
    for row in doc["controls"]:
        curve = row["curve"]
        named = {h["horizon"]: h for h in curve["horizons_named"]}
        for h in list(dnm.REQUIRED_HORIZONS) + list(dnm.PLUS_HORIZONS):
            assert h in named, f"{row['id']} silently omitted horizon {h}"
            assert named[h]["status"] in {dnm.RUN, "SKIPPED"}
        assert set(curve["series"]) == set(dnm.SERIES_KEYS)
        for rec in curve["per_horizon"]:
            dnm.require_separate_series(rec)
        assert row["economics"]["bytes_added_supplied"] is True
        assert row["one_step_only_admissible"] is False
        assert row["argmax_is_not_parity"] is True
        assert row["evidence_class"] == "STATIC_ONLY"

    ident = next(c for c in doc["controls"] if c["id"] == dnm.IDENTITY)
    assert ident["curve"]["shape"]["overall"] == dnm.FLAT_ZERO
    assert ident["status"] == dnm.CONTROL

    # Load-bearing finding of this lane: rank-16 truncation of 128x128 S is
    # exact through horizon 16 (rank has not filled) and then sits. A
    # one-step number would have promoted it. The authority must not.
    trunc = next(c for c in doc["controls"] if c["id"] == dnm.TRUNCATED_STATE)
    by_h = {int(r["horizon"]): r for r in trunc["curve"]["per_horizon"]}
    quiet = 1.0e-6  # still 10,000x below STATE_BAR; rank has not filled
    for h in (1, 4, 16):
        assert by_h[h]["state_error"]["relative_l2"] <= quiet
        assert by_h[h]["output_error"]["relative_l2"] <= quiet
        assert by_h[h]["logit_effect"]["relative_l2"] <= quiet
    assert by_h[64]["state_error"]["relative_l2"] > dnm.STATE_BAR
    assert trunc["curve"]["shape"]["overall"] == dnm.ONSET_THEN_PLATEAU
    assert trunc["status"] == dnm.MEASURED_NEGATIVE
    assert trunc["verdict"]["status"] == dnm.MEASURED_NEGATIVE

    low = next(c for c in doc["controls"] if c["id"] == dnm.LOWER_RANK_TRANSITION)
    assert low["status"] == dnm.MEASURED_NEGATIVE
    assert low["curve"]["per_horizon"][0]["state_error"]["relative_l2"] > dnm.STATE_BAR

    gen = doc["generated_transition"]
    assert gen["id"] == dnm.GENERATED_TRANSITION
    assert gen["one_step_only_admissible"] is False
    assert gen["status"] in {dnm.NOT_LANDED, dnm.UNMEASURED}
    if not gen["landed"]:
        assert gen["curve"] is None
        assert gen["economics"]["bytes_removed"] == 2_139_096_960
        assert gen["economics"]["bytes_added"]["total"] == 4_548_560

    assert doc["answers"]["is_one_step_admissible"]["one_step_only_admissible"] is False
    assert doc["answers"]["is_argmax_parity"]["argmax_is_not_parity"] is True


def test_module_entrypoint_runs(built_receipt):
    assert built_receipt["schema"] == dnm.SCHEMA
    assert built_receipt["seal_sha256"]
    assert (RECEIPTS / dnm.RECEIPT).is_file()


def test_selftest_aliases_build():
    assert dnm.selftest is dnm.build


def test_hardware_fields_stay_non_numeric_on_the_receipt(built_receipt):
    _assert_no_hardware_claims(built_receipt)
    for key in HARDWARE_FIELDS:
        assert key not in built_receipt.get("cited", {})


def test_identity_evaluate_uses_three_series_on_a_fixture():
    seq = _tiny_seq(n_tokens=24)
    row = dnm.evaluate_candidate(
        cand_id=dnm.IDENTITY,
        step=dnm.identity_step,
        sequences=[seq],
        n_heads=TINY_HEADS,
        dim=TINY_DIM,
        vocab=TINY_VOCAB,
        required=TINY_REQUIRED,
        plus=(),
        economics=_econ_zero(),
        report_as="fixture",
    )
    assert row["status"] == dnm.CONTROL
    assert row["curve"]["shape"]["overall"] == dnm.FLAT_ZERO
    assert row["verdict"]["status"] == dnm.FITTED_HELDOUT
    assert set(row["curve"]["series"]) == set(dnm.SERIES_KEYS)


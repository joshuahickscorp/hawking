"""Sealed HWIR predictions: refuse, score, tamper. PREHARDWARE. No board."""
from __future__ import annotations

import copy

import pytest

from tools.future import hwir
from tools.future._common import RECEIPTS, load_json


def _minimal_kwargs(**overrides):
    coeffs = hwir._subset_coefficients(["hbm_bytes_per_modelled_cycle"])
    body = dict(
        id="unit.hbm.beat",
        plan="inbound-u50-qgemv",
        quantity="hbm_bytes_per_modelled_cycle",
        predicted_value=1024,
        units="bytes/modelled_cycle",
        model_coefficients=coeffs,
        depends_on=["hbm_bytes_per_modelled_cycle"],
        tolerance={"kind": "relative", "value": 0.50},
        falsification_condition=(
            "An observation of HBM bytes per modelled cycle that differs from "
            "1024 by more than 50% relative falsifies hbm_bytes_per_modelled_cycle."
        ),
        implicated_coefficient="hbm_bytes_per_modelled_cycle",
        evidence_tier="COST_MODEL",
    )
    body.update(overrides)
    return body


def test_prediction_without_falsification_is_refused():
    with pytest.raises(hwir.PredictionRefused, match="falsification_condition"):
        hwir.seal_prediction(**_minimal_kwargs(falsification_condition=""))
    with pytest.raises(hwir.PredictionRefused, match="falsification_condition"):
        hwir.seal_prediction(**_minimal_kwargs(falsification_condition="   "))
    with pytest.raises(hwir.PredictionRefused, match="falsification_condition"):
        hwir.seal_prediction(**_minimal_kwargs(falsification_condition=None))
    with pytest.raises(hwir.PredictionRefused, match="falsification_condition"):
        hwir.seal_prediction(**_minimal_kwargs(falsification_condition=True))
    sealed = hwir.seal_prediction(**_minimal_kwargs())
    assert sealed["content_sha256"]
    assert sealed["falsification_condition"]
    assert sealed["wake_condition"] == hwir.WAKE_U50_PRESENT
    hwir.verify_prediction_seal(sealed)
    hwir.assert_no_hardware_measured(sealed)


def test_synthetic_rehearsal_falsifies_divergent_and_names_coefficient():
    report = hwir.run_synthetic_arrival_rehearsal(write=False)
    assert report["kind"] == "SYNTHETIC_ARRIVAL_REHEARSAL"
    assert report["not_an_arrival"] is True
    assert report["not_a_board_measurement"] is True
    assert report["synthetic_rehearsal"] is True
    assert report["hardware_measured"] is False
    assert "NOT AN ARRIVAL" in report["label"]
    hwir.assert_no_hardware_measured(report)
    assert "HARDWARE_MEASURED" not in hwir.collect_evidence_tiers(report)

    by_id = {s["prediction_id"]: s for s in report["score"]["scores"]}
    hbm = by_id[hwir.PRED_HBM_BEAT]
    assert hbm["verdict"] == "FALSIFIED"
    assert hbm["implicated_coefficient"] == "hbm_bytes_per_modelled_cycle"
    assert hbm["synthetic_rehearsal"] is True
    assert hbm["not_an_arrival"] is True

    fabric = by_id[hwir.PRED_FABRIC_BEAT]
    assert fabric["verdict"] == "FALSIFIED"
    assert fabric["implicated_coefficient"] == "fabric_bytes_per_modelled_cycle"

    host = by_id[hwir.PRED_HOST_BEAT]
    assert host["verdict"] == "CONFIRMED"
    assert host["implicated_coefficient"] is None

    planning = by_id[hwir.PRED_PLAN_HBM_CYCLES]
    assert planning["verdict"] == "FALSIFIED"
    assert planning["implicated_coefficient"] == "hbm_bytes_per_modelled_cycle"

    named = set(report["implicated_coefficients"])
    assert named == {"hbm_bytes_per_modelled_cycle", "fabric_bytes_per_modelled_cycle"}
    assert hwir.PRED_HBM_BEAT in report["falsified_ids"]
    assert hwir.PRED_FABRIC_BEAT in report["falsified_ids"]
    assert hwir.PRED_PLAN_HBM_CYCLES in report["falsified_ids"]
    assert hwir.PRED_HOST_BEAT in report["confirmed_ids"]
    for score in report["score"]["scores"]:
        assert score["verdict"] in hwir.SCORE_VERDICTS
        assert score["hardware_measured"] is False
        if score["verdict"] == "FALSIFIED":
            assert score["implicated_coefficient"]
            assert score["implicated_coefficient"] in score["depends_on"]


def test_tampered_sealed_prediction_is_rejected():
    """Editing a sealed prediction without invalidating its seal is detected.

    MUTATION_CHECK target: hwir._seal_is_valid. Replacing its body with
    `return True` must make this test FAIL. Restore after the check.
    """
    pred = hwir.inbound_board_predictions()[0]
    hwir.verify_prediction_seal(pred)
    tampered = copy.deepcopy(pred)
    tampered["predicted_value"] = float(pred["predicted_value"]) * 2 + 1
    assert tampered["content_sha256"] == pred["content_sha256"]
    assert tampered["predicted_value"] != pred["predicted_value"]
    with pytest.raises(hwir.TamperedPrediction):
        hwir.verify_prediction_seal(tampered)
    with pytest.raises(hwir.TamperedPrediction):
        hwir.score_prediction_set(
            [tampered],
            {
                tampered["id"]: {
                    "value": tampered["predicted_value"],
                    "units": tampered["units"],
                }
            },
            synthetic_rehearsal=True,
        )


def test_reseal_after_edit_is_a_new_prediction_not_a_quiet_reaim():
    pred = hwir.seal_prediction(**_minimal_kwargs())
    edited = hwir.seal_prediction(
        **_minimal_kwargs(predicted_value=256)
    )
    assert edited["content_sha256"] != pred["content_sha256"]
    hwir.verify_prediction_seal(edited)
    # Scoring uses the sealed predicted_value, never observation['predicted_value'].
    obs = {
        pred["id"]: {
            "value": 1024,
            "units": pred["units"],
            "predicted_value": 256,
        }
    }
    scored = hwir.score_prediction_set([pred], obs, synthetic_rehearsal=True)
    row = scored["scores"][0]
    assert row["predicted_value"] == 1024
    assert row["observed_value"] == 1024
    assert row["verdict"] == "CONFIRMED"


def test_real_scoring_refused_without_u50():
    preds = hwir.inbound_board_predictions()
    with pytest.raises(hwir.ScoringRefused, match="U50_PRESENT"):
        hwir.score_prediction_set(
            preds,
            hwir.synthetic_rehearsal_observations(),
            synthetic_rehearsal=False,
        )
    wake = hwir.load_u50_wake_condition()
    assert wake["present"] is False
    assert wake["wake_condition"] == hwir.WAKE_U50_PRESENT
    assert len(wake["gates_carrying_wake"]) == 12


def test_inbound_predictions_are_sealed_with_tolerance_and_falsifier():
    preds = hwir.inbound_board_predictions()
    assert len(preds) == 12
    ids = [p["id"] for p in preds]
    assert len(ids) == len(set(ids))
    required = {
        hwir.PRED_HBM_BEAT,
        hwir.PRED_FABRIC_BEAT,
        hwir.PRED_HOST_BEAT,
        hwir.PRED_PLAN_HBM_CYCLES,
    }
    assert required <= set(ids)
    for pred in preds:
        hwir.verify_prediction_seal(pred)
        assert pred["schema"] == hwir.PREDICTION_SCHEMA
        assert pred["wake_condition"] == hwir.WAKE_U50_PRESENT
        assert pred["falsification_condition"].strip()
        assert pred["tolerance"]["kind"] in hwir.TOLERANCE_KINDS
        assert pred["implicated_coefficient"] in pred["depends_on"]
        assert pred["hardware_measured"] is False
        assert pred["prehardware"] is True
        assert pred["evidence_tier"] in hwir.EVIDENCE_TIERS
        assert pred["evidence_tier"] != "HARDWARE_MEASURED"
        hwir.assert_no_hardware_measured(pred)


def test_units_mismatch_is_not_quietly_confirmed():
    pred = hwir.seal_prediction(**_minimal_kwargs())
    scored = hwir.score_prediction_set(
        [pred],
        {pred["id"]: {"value": 1024, "units": "furlongs/fortnight"}},
        synthetic_rehearsal=True,
    )
    row = scored["scores"][0]
    assert row["verdict"] == "REFUSED"
    assert row["implicated_coefficient"] is None
    assert "units mismatch" in row["reason"]


def test_missing_observation_is_unpinned_not_confirmed():
    pred = hwir.seal_prediction(**_minimal_kwargs())
    scored = hwir.score_prediction_set([pred], {}, synthetic_rehearsal=True)
    row = scored["scores"][0]
    assert row["verdict"] == "UNPINNED"
    assert row["implicated_coefficient"] is None


def test_no_hardware_measured_in_prediction_receipts():
    sealed_path = hwir.write_sealed_predictions_receipt()
    rehearsal = hwir.run_synthetic_arrival_rehearsal(write=True)
    sealed = load_json(sealed_path)
    hwir.assert_no_hardware_measured(sealed)
    hwir.assert_no_hardware_measured(rehearsal)
    assert sealed["kind"] == "SEALED_PREDICTION_SET"
    assert sealed["wake_condition"] == hwir.WAKE_U50_PRESENT
    assert sealed["u50"]["present"] is False
    assert rehearsal["kind"] == "SYNTHETIC_ARRIVAL_REHEARSAL"
    assert rehearsal["not_an_arrival"] is True
    written = load_json(RECEIPTS / hwir.REHEARSAL_RECEIPT)
    assert written["kind"] == "SYNTHETIC_ARRIVAL_REHEARSAL"
    assert written["not_an_arrival"] is True
    assert "NOT AN ARRIVAL" in written["label"]
    assert written["hardware_measured"] is False
    assert "HARDWARE_MEASURED" not in hwir.collect_evidence_tiers(written)


def test_emit_hardware_measured_still_illegal():
    with pytest.raises(hwir.IllegalEvidenceTier):
        hwir.seal_prediction(**_minimal_kwargs(evidence_tier="HARDWARE_MEASURED"))

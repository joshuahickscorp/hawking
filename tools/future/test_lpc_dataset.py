"""Tests for the LPC data contract and the negative controls on abstention/authority."""
from __future__ import annotations

import json

import pytest

from tools.future import lpc_baselines as lb
from tools.future import lpc_dataset as ds
from tools.future._common import RECEIPTS, HardwareClaimError, _assert_no_hardware_claims


def test_build_emits_sealed_receipt():
    out = ds.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "LPC_DATASET.json"
    assert doc["schema"] == ds.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert doc["inventory"]["complete"] == doc["ingest"]["inventory"]["complete"]
    _assert_no_hardware_claims(doc)


def test_receipt_refuses_hardware_field_numbers():
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims({"complete_token_ns": 1.0})


def test_missing_required_key_is_rejected():
    row = ds.row_template(model="x", contamination_class="STATIC_ONLY")
    row.pop("organ_fingerprint")
    verdict = ds.validate_row(row)
    assert verdict["status"] == "REJECTED"
    assert "organ_fingerprint" in verdict["missing"]
    assert verdict["complete"] is False


def test_valid_with_nulls_and_reasons():
    row = ds.row_template(
        model="qwen3.8-27b-sealed-3.14",
        contamination_class="STATIC_ONLY",
        reasons_for_missing="UNMEASURED",
    )
    verdict = ds.validate_row(row)
    assert verdict["status"] == "VALID"
    assert verdict["complete"] is False
    assert row["latency"] is None
    assert row["absence_reasons"]["latency"] == "UNMEASURED"


def test_silent_null_is_invalid():
    row = ds.row_template(
        reasons_for_missing=None,
        model="x",
        contamination_class="STATIC_ONLY",
    )
    verdict = ds.validate_row(row)
    assert verdict["status"] == "INVALID_NULL"
    assert verdict["why"] == "silent_null"
    assert "latency" in verdict["fields"]


def test_unknown_reason_is_invalid():
    row = ds.row_template(reasons_for_missing="not-a-real-reason", model="x")
    verdict = ds.validate_row(row)
    assert verdict["status"] == "INVALID_NULL"
    assert verdict["why"] == "unknown_reason"


def test_null_is_never_imputed_to_zero():
    row = ds.row_template(
        contamination_class="STATIC_ONLY",
        dispatches=None,
        absence_reasons={"dispatches": "UNMEASURED"},
    )
    assert ds.as_numeric(row, "dispatches") is None
    with pytest.raises(ds.ImputationError, match="refusing to impute 0"):
        ds.forbid_zero_imputation(row, "dispatches")


def test_measured_zero_is_preserved():
    row = ds.row_template(dispatches=0, contamination_class="STATIC_ONLY")
    assert ds.as_numeric(row, "dispatches") == 0
    assert ds.forbid_zero_imputation(row, "dispatches") == 0
    verdict = ds.validate_row(row)
    assert verdict["status"] == "VALID"


def test_complete_row_requires_protected_and_all_fields():
    complete = lb._complete_fixture()
    assert ds.validate_row(complete)["complete"] is True
    diagnostic = lb._complete_fixture(contamination_class="DIAGNOSTIC_RELATIVE")
    verdict = ds.validate_row(diagnostic)
    assert verdict["status"] == "VALID"
    assert verdict["complete"] is False


def test_ingest_scoreboard_fixture_does_not_invent_organs():
    doc = {
        "schema": "hawking.accelerator.scoreboard.v1",
        "rows": [
            {
                "receipt": "fixture/protected.json",
                "model": "qwen3.8-27b-sealed-3.14",
                "backend": "hawking_native",
                "representation": "native-packed",
                "machine": "Apple M3 Ultra",
                "benchmark_class": "QUALIFIED_PROTECTED",
                "complete_token_ns": 121.0,
                "dispatches": 0,
                "fallback_count": 0,
                "capability_verified": None,
                "resident_bytes": None,
                "executable_id": "abc",
            },
            {
                "receipt": "fixture/diagnostic.json",
                "model": "qwen3.8-27b-sealed-3.14",
                "backend": "native",
                "benchmark_class": "DIAGNOSTIC_CONTAMINATED",
                "gpu_ns_per_token": 50.0,
                "dispatches": 15700,
            },
        ],
    }
    rows = ds.ingest_scoreboard(doc)
    assert len(rows) == 2
    protected, diagnostic = rows
    assert ds.validate_row(protected)["status"] == "VALID"
    assert ds.validate_row(protected)["complete"] is False
    assert protected["organ_fingerprint"] is None
    assert protected["absence_reasons"]["organ_fingerprint"] == "NOT_IN_SOURCE"
    assert protected["layout"] is None
    assert protected["contamination_class"] == "PROTECTED_ABSOLUTE"
    assert protected["dispatches"] == 0
    assert protected["latency"] == 121.0
    assert ds.as_numeric(protected, "resident_bytes") is None
    assert diagnostic["contamination_class"] == "DIAGNOSTIC_RELATIVE"
    assert diagnostic["latency"] == 50.0
    census = ds.classify_rows(rows)
    assert census["complete"] == 0
    assert census["valid"] == 2


def test_ingest_queue_fixture_all_metrics_null_with_reasons():
    doc = {
        "candidates": [
            {
                "candidate_id": "qwen27-affine2-splitk4",
                "model": "Qwen27",
                "affected_physical_region": "Qwen27 HGRAVF01 affine Q2 GEMV",
                "exact_mutation": {"child_fusion_env": {"SPLITK": "4"}},
                "measurements": {
                    "status": "NOT_MEASURED",
                    "resident_bytes": None,
                    "absence_reasons": {
                        "resident_bytes": "awaiting native protected complete-token receipt",
                    },
                },
            }
        ],
        "work_units": [
            {
                "candidate_id": "qwen27-affine2-splitk4",
                "preferred_backend": "metal",
            }
        ],
    }
    rows = ds.ingest_queue(doc)
    assert len(rows) == 1
    row = rows[0]
    assert ds.validate_row(row)["status"] == "VALID"
    assert row["model"] == "Qwen27"
    assert row["organ_fingerprint"] == "Qwen27 HGRAVF01 affine Q2 GEMV"
    assert row["backend"] == "metal"
    assert row["fusion"] == {"SPLITK": "4"}
    assert row["latency"] is None
    assert row["absence_reasons"]["latency"] == "AWAITING_PROTECTED_RECEIPT"
    assert row["contamination_class"] == "STATIC_ONLY"
    assert ds.classify_rows(rows)["complete"] == 0


def test_ingest_budget_fixture_organs_are_valid_and_unmeasured():
    doc = {
        "model": "qwen3.8-27b-sealed-3.14",
        "baseline": {"representation": "native-packed sealed control"},
        "organs": [
            {
                "organ": "mlp",
                "actual": {"resident_bytes": None, "gpu_ns_per_token": None},
                "absence_reasons": {
                    "resident_bytes": "awaiting native protected complete-token receipt",
                },
            }
        ],
    }
    rows = ds.ingest_budget(doc)
    assert len(rows) == 1
    assert rows[0]["organ_fingerprint"] == "mlp"
    assert rows[0]["contamination_class"] == "STATIC_ONLY"
    assert ds.as_numeric(rows[0], "latency") is None
    assert ds.validate_row(rows[0])["complete"] is False


def test_live_ingest_does_not_claim_complete_rows_without_sources():
    rows, report = ds.ingest_from_disk()
    census = report["inventory"]
    assert census["n"] == len(rows)
    assert census["complete"] == 0
    for src in report["sources"]:
        assert src["complete"] == 0
        assert "on_disk" in src and "in_git_HEAD" in src


def test_predictor_abstains_outside_support():
    """NEGATIVE CONTROL: far queries must ABSTAIN, not extrapolate."""
    train = [
        lb._complete_fixture(row_id="t0", organ_fingerprint="mlp", latency=10.0),
        lb._complete_fixture(row_id="t1", organ_fingerprint="attention", latency=12.0),
    ]
    in_support = lb._complete_fixture(row_id="q-near", organ_fingerprint="mlp", latency=None)
    in_support["absence_reasons"] = {"latency": "UNMEASURED"}
    near = lb.predict(in_support, train, method="nearest")
    assert near.status == lb.PREDICTED
    assert near.value == 10.0
    assert near.uncertainty is not None and near.uncertainty > 0

    far = lb._complete_fixture(
        row_id="q-far",
        model="not-a-hawking-model",
        organ_fingerprint="vacuum-chamber",
        representation="imaginary-nr",
        machine_genome="other-soc",
        physical_graph_identity="no-such-graph",
        backend="not-a-backend",
        layout="scrambled",
        tile="tg1",
        grouping="none",
        fusion="everything",
        persistent_resources="none",
        active_bytes=1e12,
        resident_bytes=1e12,
        dispatches=1e9,
        synchronization=1e9,
        latency=None,
        absence_reasons={"latency": "UNMEASURED"},
    )
    distance = lb.row_distance(train[0], far)
    assert distance is not None
    assert distance > lb.SUPPORT_RADIUS
    refused = lb.predict(far, train, method="nearest")
    assert refused.status == lb.ABSTAIN
    assert refused.value is None
    assert refused.as_tuple() == lb.ABSTAIN
    assert refused.reason == "outside_support"


def test_authority_protected_outranks_confident_model():
    """NEGATIVE CONTROL: a confident model cannot override PROTECTED_ABSOLUTE."""
    model = lb.Prediction(
        status=lb.PREDICTED,
        value=1.0,
        uncertainty=lb.UNCERTAINTY_FLOOR,
        method="nearest",
        reason=None,
    )
    measurement = {"contamination_class": "PROTECTED_ABSOLUTE", "value": 999.0}
    decided = lb.resolve_authority(model, measurement)
    assert decided["value"] == 999.0
    assert decided["source"] == "PROTECTED_ABSOLUTE"
    assert decided["model_prediction_ignored"] is True
    assert decided["model_value"] == 1.0
    assert decided["value"] != model.value

    with pytest.raises(lb.AuthorityError, match="not PROTECTED_ABSOLUTE"):
        lb.resolve_authority(
            model,
            {"contamination_class": "DIAGNOSTIC_RELATIVE", "value": 2.0},
        )
    with pytest.raises(lb.AuthorityError, match="null value"):
        lb.resolve_authority(
            model,
            {"contamination_class": "PROTECTED_ABSOLUTE", "value": None},
        )

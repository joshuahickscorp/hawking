"""Tests for the empirical MLP functional-rank probe.

Load-bearing negatives: a train-set error must not be reportable as held-out,
and a byte figure must come from executable_economics.score, not a ratio.
A guard nobody has watched fail is not a guard.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tools.future import executable_economics as ee
from tools.future import mlp_functional_rank as mfr
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    _assert_no_hardware_claims,
)


def test_train_error_cannot_be_reported_as_held_out():
    """NEGATIVE CONTROL: split='train' must refuse, not relabel."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=(16, 8)).astype(np.float32)
    yhat = y + 0.01
    err = mfr.relative_output_error(y, yhat)
    with pytest.raises(mfr.TrainReportedAsHeldOut) as caught:
        mfr.report_held_out_error(err, split="train", n_rows=16, rank=4, method="ols")
    assert "REFUSED" in str(caught.value)
    assert "TRAIN_REPORTED_AS_HELD_OUT" in caught.value.codes
    assert caught.value.result["split"] == "train"

    with pytest.raises(mfr.TrainReportedAsHeldOut):
        mfr.report_held_out_error(err, split="TRAIN", n_rows=16)

    with pytest.raises(mfr.TrainReportedAsHeldOut):
        mfr.report_held_out_error(err, split="fit", n_rows=16)


def test_hold_package_that_includes_a_train_prompt_is_refused():
    """NEGATIVE CONTROL: a leaked prompt id is not a held-out number."""
    with pytest.raises(mfr.TrainReportedAsHeldOut) as caught:
        mfr.report_held_out_error(
            0.02,
            split="hold",
            n_rows=8,
            prompt_ids=["code:00", "code:14"],
            train_prompt_ids=["code:00", "code:01"],
            hold_prompt_ids=["code:14", "code:15"],
        )
    assert "TRAIN_REPORTED_AS_HELD_OUT" in caught.value.codes
    assert "code:00" in caught.value.result["train_prompt_ids_in_hold"]

    with pytest.raises(mfr.TrainReportedAsHeldOut) as caught2:
        mfr.report_held_out_error(
            0.02,
            split="hold",
            n_rows=8,
            train_prompt_ids=["code:14"],
            hold_prompt_ids=["code:14", "code:15"],
        )
    assert "HELD_OUT_PROMPT_LEAK" in caught2.value.codes


def test_genuine_hold_error_is_labeled_held_out():
    y = np.ones((10, 4), dtype=np.float32)
    pack = mfr.report_held_out_error(
        mfr.relative_output_error(y, y),
        split="hold",
        n_rows=10,
        prompt_ids=["h:00"],
        train_prompt_ids=["t:00"],
        hold_prompt_ids=["h:00"],
        rank=3,
        method="identity",
    )
    assert pack["split"] == "hold"
    assert pack["held_out"] is True
    assert pack["relative_error"] == pytest.approx(0.0)
    assert pack["n_rows"] == 10
    assert "train" not in pack["split"]


def test_byte_figures_come_from_executable_economics_not_a_ratio():
    """A compression ratio without executable economics is not a candidate."""
    with pytest.raises(ee.IncompleteEconomics, match="bytes_added"):
        ee.score(bytes_removed=ee.MLP_ACTIVE_BYTES)

    ledger = mfr.rank_byte_ledger(32)
    for key in ee.BYTES_ADDED_FIELDS:
        assert key in ledger
    assert ledger["residuals"] == 0
    assert ledger["generator"] > 0
    assert ledger["metadata"] > 0
    # The helper must go through ee.score — compare field-by-field.
    scored = mfr.score_rank_bytes(32, family="factorized_swiglu", status="OPEN")
    direct = ee.score(
        bytes_removed=ee.MLP_ACTIVE_BYTES,
        bytes_added=ledger,
        extra_flops_per_output_element=0.0,
        organ="mlp",
        stream_class="weight_codes",
        consuming_primitive="TiledProjection",
        reusable_family=True,
        high_information_falsifier=True,
        status="OPEN",
    )
    assert scored["scored_by"] == "tools/future/executable_economics.py::score"
    assert scored["bytes_removed"] == ee.MLP_ACTIVE_BYTES
    assert scored["bytes_added"]["generator"] == ledger["generator"]
    assert scored["bytes_added"]["residuals"] == ledger["residuals"]
    assert scored["bytes_added"]["metadata"] == ledger["metadata"]
    assert scored["bytes_added_total"] == sum(ledger[k] for k in ee.BYTES_ADDED_FIELDS)
    assert scored["net_bytes"] == direct["net_bytes"]
    assert scored["verdict"] == direct["verdict"]
    # Residual is in the ledger even when it is zero. A ratio would omit it.
    assert "residuals" in scored["bytes_added"]
    assert scored["incumbent_mlp_bytes"] == 5_347_795_776


def test_affordable_rank_cap_counts_metadata_not_just_generator():
    cap = mfr._affordable_rank_cap()
    under = mfr.rank_byte_ledger(cap)
    over = mfr.rank_byte_ledger(cap + 1)
    assert sum(under.values()) < ee.MLP_ACTIVE_BYTES
    assert sum(over.values()) >= ee.MLP_ACTIVE_BYTES
    scored = mfr.score_rank_bytes(cap, family="factorized_swiglu", status="MEASURED_NEGATIVE")
    assert scored["bytes_added_total"] == sum(under[k] for k in ee.BYTES_ADDED_FIELDS)
    assert scored["net_bytes"] < 0
    assert scored["bytes_added"]["metadata"] > 0
    assert scored["bytes_added"]["residuals"] == 0


def test_linear_map_ledger_is_also_scored_by_economics():
    scored = mfr.score_rank_bytes(64, family="linear_map", status="OPEN")
    assert scored["scored_by"] == "tools/future/executable_economics.py::score"
    assert scored["bytes_added"]["residuals"] == 0
    assert scored["bytes_removed"] == ee.MLP_ACTIVE_BYTES
    assert scored["bytes_added"]["generator"] == mfr.linear_map_byte_ledger(64)["generator"]


def test_effective_dimension_and_relative_error_on_a_known_spectrum():
    eig = np.array([8.0, 1.0, 0.5, 0.4, 0.1], dtype=np.float64)
    # total 10. 80% of 10 is 8 → first 1. 90% of 10 is 9 → first 2 (8+1).
    assert mfr.effective_dimension(eig, 0.80) == 1
    assert mfr.effective_dimension(eig, 0.90) == 2
    assert mfr.effective_dimension(eig, 0.95) == 3
    assert mfr.effective_dimension(eig, 0.99) == 4
    assert mfr.effective_dimension(eig, 1.00) == 5
    assert mfr.participation_ratio(eig) == pytest.approx((10.0 ** 2) / (8.0 ** 2 + 1 + 0.25 + 0.16 + 0.01))

    y = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)  # norms 5 and 0; mean 2.5
    yhat = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    assert mfr.relative_output_error(y, yhat) == pytest.approx(0.0)
    yhat2 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    assert mfr.relative_output_error(y, yhat2) == pytest.approx(1.0)


def test_activation_weighted_and_raw_svd_both_reported_and_they_differ():
    """Decaying X + full-rank W: weighted rank-r beats raw SVD on hold error."""
    fx = mfr.make_decaying_linear_fixture(n=120, d=20, hold=30, decay=2.0, seed=7)
    # Orthogonal W so raw singular values are all equal (energy is full-dim)
    # while activation-weighted energy follows the decaying X.
    q, _ = np.linalg.qr(np.random.default_rng(11).normal(size=(fx["d"], fx["d"])))
    w = q.astype(np.float32)
    fx["W"] = w
    fx["y_tr"] = fx["x_tr"] @ w.T
    fx["y_ho"] = fx["x_ho"] @ w.T
    x2 = mfr.second_moment_basis(fx["x_tr"])
    spec = mfr.organ_spectra(fx["W"], x2)
    raw_90 = mfr.effective_dimension(spec["raw_s"] ** 2, 0.90)
    w_90 = mfr.effective_dimension(spec["w_s"] ** 2, 0.90)
    assert raw_90 > w_90
    assert w_90 < fx["d"]
    assert raw_90 >= int(0.85 * fx["d"])

    r = max(3, w_90)
    y_w = mfr.apply_weighted_organ(fx["x_ho"], spec, r)
    y_r = mfr.apply_raw_organ(fx["x_ho"], spec["W"], spec["raw_V"], r)
    err_w = mfr.relative_output_error(fx["y_ho"], y_w)
    err_r = mfr.relative_output_error(fx["y_ho"], y_r)
    pack_w = mfr.report_held_out_error(
        err_w,
        split="hold",
        n_rows=len(fx["y_ho"]),
        train_prompt_ids=fx["train_prompt_ids"],
        hold_prompt_ids=fx["hold_prompt_ids"],
        rank=r,
        method="activation_weighted",
    )
    pack_r = mfr.report_held_out_error(
        err_r,
        split="hold",
        n_rows=len(fx["y_ho"]),
        train_prompt_ids=fx["train_prompt_ids"],
        hold_prompt_ids=fx["hold_prompt_ids"],
        rank=r,
        method="raw_svd",
    )
    assert pack_w["held_out"] is True and pack_r["held_out"] is True
    assert pack_w["relative_error"] < pack_r["relative_error"]
    # Full rank recovers the linear map on both.
    y_full_w = mfr.apply_weighted_organ(fx["x_ho"], spec, fx["d"])
    y_full_r = mfr.apply_raw_organ(fx["x_ho"], spec["W"], spec["raw_V"], fx["d"])
    assert mfr.relative_output_error(fx["y_ho"], y_full_w) < 1e-4
    assert mfr.relative_output_error(fx["y_ho"], y_full_r) < 1e-4


def test_first_rank_crossing_and_none_when_never():
    sweep = [
        {"rank": 8, "factorized_weighted_held_out_relative_error": 0.40},
        {"rank": 32, "factorized_weighted_held_out_relative_error": 0.12},
        {"rank": 64, "factorized_weighted_held_out_relative_error": 0.02},
    ]
    assert mfr.first_rank_at_or_below(sweep, 0.10, error_key="factorized_weighted_held_out_relative_error") == 64
    assert mfr.first_rank_at_or_below(sweep, 0.03, error_key="factorized_weighted_held_out_relative_error") == 64
    assert mfr.first_rank_at_or_below(sweep, 0.01, error_key="factorized_weighted_held_out_relative_error") is None
    assert mfr.first_rank_at_or_below(sweep, 0.50, error_key="factorized_weighted_held_out_relative_error") == 8


def test_selftest_function_proves_the_refusals():
    result = mfr.selftest()
    assert result["train_reported_as_held_out_refused"] is True
    assert result["held_out_prompt_leak_refused"] is True
    assert result["genuine_hold_accepted"] is True
    assert result["bytes_removed_without_added_refused"] is True
    assert result["economics_scored_by"] == "tools/future/executable_economics.py::score"
    assert result["ledger_has_residual_field"] is True
    assert result["identity_hold_relative_error"] == pytest.approx(0.0)


def test_receipt_if_present_is_held_out_and_economics_scored():
    path = RECEIPTS / mfr.RECEIPT
    if not path.is_file():
        pytest.skip("MLP_FUNCTIONAL_RANK.json not written yet")
    doc = json.loads(path.read_text())
    assert doc["schema"] == mfr.SCHEMA
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    _assert_no_hardware_claims(doc)
    for field in HARDWARE_FIELDS:
        assert field not in doc
    assert doc["seal_sha256"]
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    import hashlib

    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    assert doc["selftest"]["train_reported_as_held_out_refused"] is True
    assert doc["organ"]["incumbent_mlp_bytes"] == 5_347_795_776
    assert "TRAIN_REPORTED_AS_HELD_OUT" in doc["anti_fabrication"]["detectors"]

    for layer in doc["layers"]:
        for point in layer["held_out_sweep"]:
            for key, value in point.items():
                if "error" in key:
                    assert "held_out" in key, (
                        f"layer {layer['layer']} reports {key} without held_out in the name"
                    )
                    assert value is None or value >= 0.0
        # The split on the layer record is hold-only.
        assert layer["split"]["unit"] == "prompt_id"
        assert layer["split"]["disjoint"] is True

    for th, row in doc["crossings"].items():
        econ = row.get("economics")
        if econ is None:
            continue
        assert econ["scored_by"] == "tools/future/executable_economics.py::score"
        for key in ee.BYTES_ADDED_FIELDS:
            assert key in econ["bytes_added"]
        assert econ["bytes_removed"] == ee.MLP_ACTIVE_BYTES
        assert "residuals" in econ["bytes_added"]
    cap_econ = doc["affordable_cap_economics"]
    assert cap_econ["scored_by"] == "tools/future/executable_economics.py::score"
    assert cap_econ["bytes_added"]["residuals"] == 0
    assert cap_econ["incumbent_mlp_bytes"] == ee.MLP_ACTIVE_BYTES

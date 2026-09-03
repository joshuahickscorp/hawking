"""Tests for the shared MLP program census.

Load-bearing negatives a guard nobody has watched fail is not a guard:

  * a candidate whose shared-basis bytes are not billed is REFUSED
  * a train-set figure cannot be reported as held-out
  * a consumer that rematerializes dense W cannot be reported live
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools.future import executable_economics as ee
from tools.future import mlp_shared_program as msp
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    _assert_no_hardware_claims,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def _fixture(**kwargs):
    return msp.make_fixture_xy(**kwargs)


def test_unbilled_shared_basis_is_refused():
    """NEGATIVE CONTROL: a free shared basis is a fabrication, not a candidate."""
    br = msp.byte_breakdown(
        shape=msp.SHARED_BOTH, rank_in=8, rank_out=8, residual_k=4, n_layers=64
    )
    added = msp.bytes_added_from_breakdown(br)
    # Honest ledger must pass.
    msp.validate_billing(
        {"shape": msp.SHARED_BOTH, "byte_breakdown": br, "bytes_added": added}
    )
    assert added["embeddings"] == (
        br["shared_input_basis_bytes"] + br["shared_output_basis_bytes"]
    )
    assert added["embeddings"] > 0
    assert added["total"] == sum(added[k] for k in ee.BYTES_ADDED_FIELDS)

    stolen = dict(added)
    stolen["embeddings"] = 0
    stolen["total"] = sum(stolen[k] for k in ee.BYTES_ADDED_FIELDS)
    with pytest.raises(msp.UnbilledSharedBasis, match="fabrication|not billed"):
        msp.validate_billing(
            {
                "shape": msp.SHARED_BOTH,
                "byte_breakdown": br,
                "bytes_added": stolen,
            }
        )

    # Bases used, bytes recorded as zero in the breakdown itself.
    zero_br = dict(br)
    zero_br["shared_input_basis_bytes"] = 0
    zero_br["shared_output_basis_bytes"] = 0
    with pytest.raises(msp.UnbilledSharedBasis, match="free in the receipt"):
        msp.validate_billing(
            {
                "shape": msp.SHARED_INPUT,
                "byte_breakdown": zero_br,
                "bytes_added": {
                    "embeddings": 0,
                    "generator": 1,
                    "residuals": 0,
                    "metadata": 0,
                    "state": 0,
                    "total": 1,
                },
            }
        )


def test_shared_input_and_output_each_require_their_basis_billed():
    br_in = msp.byte_breakdown(shape=msp.SHARED_INPUT, rank_in=16, rank_out=16)
    assert br_in["shared_input_basis_bytes"] > 0
    assert br_in["shared_output_basis_bytes"] == 0
    msp.validate_billing(
        {
            "shape": msp.SHARED_INPUT,
            "byte_breakdown": br_in,
            "bytes_added": msp.bytes_added_from_breakdown(br_in),
        }
    )
    br_out = msp.byte_breakdown(shape=msp.SHARED_OUTPUT, rank_in=16, rank_out=16)
    assert br_out["shared_output_basis_bytes"] > 0
    assert br_out["shared_input_basis_bytes"] == 0
    msp.validate_billing(
        {
            "shape": msp.SHARED_OUTPUT,
            "byte_breakdown": br_out,
            "bytes_added": msp.bytes_added_from_breakdown(br_out),
        }
    )
    # SHARED_INPUT with the output-only ledger is a fabrication.
    with pytest.raises(msp.UnbilledSharedBasis):
        msp.validate_billing(
            {
                "shape": msp.SHARED_INPUT,
                "byte_breakdown": br_out,
                "bytes_added": msp.bytes_added_from_breakdown(br_out),
            }
        )


def test_train_set_figure_cannot_be_reported_as_held_out():
    """NEGATIVE CONTROL: a fit-set number labelled held-out must refuse."""
    fx = _fixture()
    # Honest hold score is legal.
    ho = msp.function_error(fx["Yho"], fx["Yho"], split="hold", report_as="held_out")
    assert ho["held_out_split"] == "hold"
    assert ho["error_authority"] == "held_out_relative_l2"
    assert ho["held_out_relative_l2"] == pytest.approx(0.0, abs=1e-12)

    tr = msp.function_error(fx["Ytr"], fx["Ytr"], split="train", report_as="train")
    assert "held_out_relative_l2" not in tr
    assert tr["train_split"] == "train"

    with pytest.raises(msp.TrainReportedAsHeldOut, match="cannot be reported as held-out"):
        msp.function_error(fx["Ytr"], fx["Ytr"], split="train", report_as="held_out")

    with pytest.raises(msp.TrainReportedAsHeldOut):
        msp.function_error(fx["Ytr"], fx["Ytr"], split="train", report_as="hold")

    forged = {
        "held_out_relative_l2": tr["train_relative_l2_diagnostic"],
        "held_out_split": "train",
        "error_authority": "held_out_relative_l2",
    }
    with pytest.raises(msp.TrainReportedAsHeldOut, match="held_out_split"):
        msp.validate_error_authority(forged)

    leaked = {
        "held_out_relative_l2": 0.01,
        "held_out_split": "hold",
        "fitted_on": "hold",
        "error_authority": "held_out_relative_l2",
    }
    with pytest.raises(msp.TrainReportedAsHeldOut):
        msp.validate_error_authority(leaked)


def test_weight_reconstruction_is_not_authority():
    row = {
        "held_out_relative_l2": 0.1,
        "held_out_split": "hold",
        "error_authority": "weight_reconstruction_error",
    }
    with pytest.raises(msp.SharedProgramRefuse, match="not held_out_relative_l2"):
        msp.validate_error_authority(row)


def test_remat_consumer_dies_immediately():
    """NEGATIVE CONTROL: rebuild-W-then-GEMV cannot be reported live."""
    fx = _fixture()
    sketch = msp.native_consumer_sketch(msp.SHARED_BOTH, rematerialize_dense_W=True)
    assert sketch["status"] == msp.REJECTED_DENSE_REMAT
    assert sketch["rematerialize_dense_W"] is True
    assert msp.consumer_status(sketch) == msp.REJECTED_DENSE_REMAT
    with pytest.raises(msp.RematConsumer, match="REJECTED_DENSE_REMAT"):
        msp.emit_candidate(
            shape=msp.SHARED_BOTH,
            rank_in=4,
            rank_out=4,
            residual_k=0,
            program="linear",
            pred_tr=fx["Ytr"],
            pred_ho=fx["Yho"],
            y_tr=fx["Ytr"],
            y_ho=fx["Yho"],
            consumer=sketch,
            n_layers=2,
        )
    dead = {
        "status": msp.REJECTED_DENSE_REMAT,
        "consumer_status": msp.REJECTED_DENSE_REMAT,
        "held_out_relative_l2": 0.0,
    }
    assert msp.surviving_candidates([dead]) == []


def test_native_consumer_is_an_atlas_primitive_and_direct():
    for shape in msp.SHAPES:
        sketch = msp.native_consumer_sketch(shape, residual_k=32)
        assert sketch["primitive"] in ATLAS_PRIMITIVES
        assert sketch["consumes_directly"] is True
        assert sketch["rematerialize_dense_W"] is False
        assert msp.consumer_status(sketch) == msp.DIRECT_CONSUME
        for name in sketch["also"]:
            assert name in ATLAS_PRIMITIVES


def test_honest_emit_bills_the_basis_and_scores_held_out():
    fx = _fixture()
    row = msp.emit_candidate(
        shape=msp.SHARED_INPUT,
        rank_in=4,
        rank_out=4,
        residual_k=0,
        program="linear",
        pred_tr=fx["Ytr"],
        pred_ho=fx["Yho"],
        y_tr=fx["Ytr"],
        y_ho=fx["Yho"],
        consumer=msp.native_consumer_sketch(msp.SHARED_INPUT),
        n_layers=2,
    )
    assert row["held_out_split"] == "hold"
    assert row["error_authority"] == "held_out_relative_l2"
    assert row["weight_reconstruction_error"] is None
    assert row["bytes_added"]["embeddings"] > 0
    assert row["byte_breakdown"]["shared_input_basis_bytes"] > 0
    assert set(ee.BYTES_ADDED_FIELDS) <= set(row["bytes_added"])
    assert "predicted_ms_saved" in row["economics"]
    assert row["economics"]["bytes_added_total"] == row["bytes_added"]["total"]
    # Train diagnostic is present under a different key, not as held-out.
    assert "train_relative_l2_diagnostic" in row
    assert row["held_out_relative_l2"] != row.get("train_split")


def test_underdetermined_rank_is_refused():
    fx = _fixture(n_train=6, n_hold=4, hidden=16, rank=3)
    with pytest.raises(msp.UnderdeterminedFit, match="n_fit"):
        msp.fit_shared_input(
            fx["Xtr"], fx["Ytr"], fx["Xho"], fx["Yho"], rank=16, program="linear"
        )


def test_mean_l2_ratio_is_the_contract_metric_not_frobenius():
    rng = np.random.default_rng(0)
    target = rng.standard_normal((8, 5)).astype(np.float32)
    pred = target * 0.5
    rel = msp.mean_l2_ratio(pred, target)
    fro = msp.relative_frobenius(pred, target)
    assert rel == pytest.approx(0.5, rel=1e-6)
    # Frobenius of a stacked matrix is a different aggregator.
    assert fro == pytest.approx(0.5, rel=1e-6)
    # A row-heterogeneous error: mean-L2 and Frobenius must be allowed to differ.
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0], dtype=np.float32)
    tgt = rng.standard_normal((8, 5)).astype(np.float32) * scale[:, None]
    err = np.zeros_like(tgt)
    err[-1] = tgt[-1]
    pred2 = tgt - err
    rel2 = msp.mean_l2_ratio(pred2, tgt)
    fro2 = msp.relative_frobenius(pred2, tgt)
    assert rel2 != pytest.approx(fro2, rel=1e-4)


def test_selftest_fires_the_three_guards():
    out = msp.selftest()
    assert out["held_out_leak_refused"] is True
    assert out["unbilled_shared_basis_refused"] is True
    assert out["remat_consumer_refused"] is True


def test_consult_index_does_not_refuse_the_proposal_families():
    index = msp.consult_index()
    assert index["proceed"] is True
    assert index["proposal_refused"] == []
    families = {q["hypothesis_family"] for q in index["queries"]}
    assert "shared_input_transforms" in families
    assert "function_replacement" in families
    assert "synthetic_activation" in families


def test_build_emits_sealed_receipt():
    out = msp.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "MLP_SHARED_PROGRAM.json"
    assert doc["schema"] == msp.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["gpu_authority"] is False
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    _assert_no_hardware_claims(doc)
    for field in HARDWARE_FIELDS:
        assert field not in doc
    assert doc["selftest"]["held_out_leak_refused"] is True
    assert doc["selftest"]["unbilled_shared_basis_refused"] is True
    assert doc["metric"]["authority"] == "held_out_relative_l2"
    assert doc["go_wider"] is False
    assert doc["n_survivors"] == 0
    assert doc["survivors"] == []


def test_receipt_errors_are_held_out_and_bases_are_billed():
    out = msp.build()
    doc = json.loads(out.read_text())
    assert doc["candidates"]
    shapes = {c["shape"] for c in doc["candidates"]}
    assert shapes == set(msp.SHAPES)
    for row in doc["candidates"]:
        assert row["held_out_split"] == "hold"
        assert row["error_authority"] == "held_out_relative_l2"
        assert "held_out_relative_l2" in row
        assert "train_relative_l2_diagnostic" in row
        assert row["weight_reconstruction_error"] is None
        assert row["bytes_added"]["embeddings"] > 0
        msp.validate_billing(row)
        msp.validate_error_authority(row)
        for key in ee.BYTES_ADDED_FIELDS:
            assert key in row["bytes_added"]
        assert row["economics"]["bytes_removed"] == ee.MLP_ACTIVE_BYTES
        assert row["status"] == msp.MEASURED_NEGATIVE
        assert row["consumer"]["primitive"] in ATLAS_PRIMITIVES
        assert row["consumer_status"] == msp.DIRECT_CONSUME
        assert float(row["held_out_relative_l2"]) >= msp.HELD_OUT_KILL_REL
        # Train diagnostic must not be laundered as the held-out figure:
        # they are different keys, and on this corpus they are different numbers.
        assert row["held_out_relative_l2"] != row["train_relative_l2_diagnostic"]
    for verdict in doc["shape_verdicts"]:
        assert verdict["status"] == msp.MEASURED_NEGATIVE
        assert verdict["native_consumer"]["primitive"] in ATLAS_PRIMITIVES
        assert verdict["clears_s020_time_bar_if_function_held"] is True
    for rec in doc["oracle_output_pca"]:
        assert rec["held_out_split"] == "hold"
        assert rec["held_out_relative_l2"] >= msp.HELD_OUT_KILL_REL
    assert doc["baselines"]["held_out_split"] == "hold"
    assert doc["baselines"]["zero_held_out_relative_l2"] == pytest.approx(1.0, abs=1e-6)
    assert doc["corpus"]["split_unit"] == "prompt_id"
    assert doc["corpus"]["disjoint"] is True
    assert doc["index"]["proceed"] is True


def test_economics_projection_uses_the_shared_scorer():
    br = msp.byte_breakdown(shape=msp.SHARED_BOTH, rank_in=32, rank_out=32, residual_k=32)
    added = msp.bytes_added_from_breakdown(br)
    scored = ee.score(
        bytes_removed=ee.MLP_ACTIVE_BYTES,
        bytes_added={k: added[k] for k in ee.BYTES_ADDED_FIELDS},
        organ="mlp",
        stream_class="weight_codes",
        consuming_primitive="TiledProjection",
        reusable_family=True,
        high_information_falsifier=True,
        status=msp.OPEN,
    )
    assert scored["bytes_added"]["embeddings"] == added["embeddings"]
    assert scored["s020_section_20"]["clears_time_bar"] is True
    assert added["total"] < ee.MLP_ACTIVE_BYTES
    # A free basis would have been a different (fabricated) net.
    free = dict(added)
    free["embeddings"] = 0
    free_scored = ee.score(
        bytes_removed=ee.MLP_ACTIVE_BYTES,
        bytes_added={k: free[k] for k in ee.BYTES_ADDED_FIELDS},
        organ="mlp",
        stream_class="weight_codes",
        consuming_primitive="TiledProjection",
        status=msp.OPEN,
    )
    assert free_scored["net_bytes"] < scored["net_bytes"]


def test_function_error_matches_the_two_metrics_it_fused():
    """Sharing the upcast must not move either metric.

    function_error called mean_l2_ratio and relative_frobenius back to back on
    the same pair; each cast pred and target to float64 and each formed
    pred - target, so four large temporaries existed where two suffice. The
    fused form must return EXACTLY what the standalone functions return -- these
    are contract metrics, and a value that drifts under a refactor can no longer
    be compared against a sealed receipt.
    """
    import numpy as np

    from tools.future import mlp_shared_program as m

    rng = np.random.default_rng(9)
    for shape in ((512, 301), (97, 64), (33, 8)):
        pred = rng.standard_normal(shape, dtype=np.float32)
        target = rng.standard_normal(shape, dtype=np.float32)

        got = m.function_error(pred, target, split="hold", report_as="held_out")

        assert got["held_out_relative_l2"] == m._r(m.mean_l2_ratio(pred, target)), shape
        assert got["held_out_relative_fro_diagnostic"] == m._r(
            m.relative_frobenius(pred, target)
        ), shape

    # The refusal must survive the fusion: a shape mismatch still refuses.
    import pytest

    with pytest.raises(m.SharedProgramRefuse):
        m.function_error(
            rng.standard_normal((4, 5), dtype=np.float32),
            rng.standard_normal((4, 6), dtype=np.float32),
            split="hold",
            report_as="held_out",
        )

"""Tests for the full-width structured MLP operator census.

Load-bearing negatives a guard nobody has watched fail is not a guard:

  * a train-set figure cannot be reported as held-out
  * a held-out figure without the mean-predictor baseline is REFUSED
  * a complete ledger at or above the incumbent cannot be a byte win
  * a consumer that rematerializes dense W cannot be reported live
  * a used factor with 0 billed bytes is REFUSED
  * a named r-bottleneck family cannot be a candidate
  * a distilled hidden width below the input width is a bottleneck in disguise
  * every byte figure must come from executable_economics.score
  * the distilled control ran and is reported whatever it is
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools.future import executable_economics as ee
from tools.future import mlp_shared_program as msp
from tools.future import mlp_structured_operator as mso
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    _assert_no_hardware_claims,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def _fixture(**kwargs):
    return mso.make_fixture_xy(**kwargs)


def test_train_set_figure_cannot_be_reported_as_held_out():
    fx = _fixture()
    ho = mso.function_error(fx["Yho"], fx["Yho"], split="hold", report_as="held_out")
    assert ho["held_out_split"] == "hold"
    assert ho["error_authority"] == "held_out_relative_l2"
    assert ho["held_out_relative_l2"] == pytest.approx(0.0, abs=1e-12)

    tr = mso.function_error(fx["Ytr"], fx["Ytr"], split="train", report_as="train")
    assert "held_out_relative_l2" not in tr
    assert tr["train_split"] == "train"

    with pytest.raises(mso.TrainReportedAsHeldOut, match="cannot be reported as held-out"):
        mso.function_error(fx["Ytr"], fx["Ytr"], split="train", report_as="held_out")

    forged = {
        "held_out_relative_l2": tr["train_relative_l2_diagnostic"],
        "held_out_split": "train",
        "error_authority": "held_out_relative_l2",
        "mean_held_out_relative_l2": 0.97,
    }
    with pytest.raises(mso.TrainReportedAsHeldOut):
        mso.validate_error_authority(forged)


def test_omitting_the_mean_predictor_baseline_is_refused():
    """NEGATIVE CONTROL: a held-out number not beside the mean is not a result."""
    with pytest.raises(mso.BaselineOmitted, match="mean-predictor baseline"):
        mso.validate_baseline(
            {
                "held_out_relative_l2": 0.4,
                "held_out_split": "hold",
                "error_authority": "held_out_relative_l2",
            }
        )
    fx = _fixture()
    with pytest.raises(mso.BaselineOmitted, match="mean-predictor"):
        mso.emit_candidate(
            family=mso.KRONECKER,
            program="tensor_product",
            pred_tr=fx["Ytr"],
            pred_ho=fx["Yho"],
            y_tr=fx["Ytr"],
            y_ho=fx["Yho"],
            consumer=mso.native_consumer_sketch(
                mso.KRONECKER,
                factor_p=fx["factor_p"],
                factor_q=fx["factor_q"],
                hidden=fx["hidden"],
                n_layers=2,
            ),
            mean_held_out_relative_l2=None,
            factor_p=fx["factor_p"],
            factor_q=fx["factor_q"],
            hidden=fx["hidden"],
            n_layers=2,
            meta_ho=fx["hold_meta"],
        )


def test_over_incumbent_cannot_be_reported_as_a_byte_win():
    """NEGATIVE CONTROL: a ledger >= incumbent is not a byte win."""
    br = mso.byte_breakdown(
        mso.DISTILLED,
        depth=2,
        width=mso.HIDDEN,
        hidden=mso.HIDDEN,
        n_layers=mso.N_LAYERS,
        program="two_layer_silu",
    )
    added = mso.bytes_added_from_breakdown(br)
    assert added["total"] >= mso.INCUMBENT_MLP_BYTES
    assert added["total"] == sum(added[k] for k in ee.BYTES_ADDED_FIELDS)
    with pytest.raises(mso.ExceedsIncumbent, match="byte win"):
        mso.report_as_byte_win(
            {
                "bytes_added": added,
                "exceeds_incumbent": True,
                "byte_win": True,
            }
        )
    fx = _fixture(hidden=16)
    # Honest emit of an over-incumbent distilled setting (full geometry).
    x_tr = np.zeros((8, mso.HIDDEN), dtype=np.float32)
    y_tr = np.zeros((8, mso.HIDDEN), dtype=np.float32)
    x_ho = np.zeros((4, mso.HIDDEN), dtype=np.float32)
    y_ho = np.ones((4, mso.HIDDEN), dtype=np.float32)
    mean_held = mso.mean_l2_ratio(np.zeros_like(y_ho), y_ho)
    row = mso.emit_candidate(
        family=mso.DISTILLED,
        program="two_layer_silu",
        pred_tr=np.zeros_like(y_tr),
        pred_ho=np.zeros_like(y_ho),
        y_tr=y_tr,
        y_ho=y_ho,
        consumer=mso.native_consumer_sketch(
            mso.DISTILLED, depth=2, width=mso.HIDDEN, n_layers=mso.N_LAYERS
        ),
        mean_held_out_relative_l2=mean_held,
        depth=2,
        width=mso.HIDDEN,
        hidden=mso.HIDDEN,
        n_layers=mso.N_LAYERS,
        id_suffix="two_layer_silu_over",
    )
    assert row["exceeds_incumbent"] is True
    assert row["byte_win"] is False
    assert "byte_win_refused" in row
    with pytest.raises(mso.ExceedsIncumbent):
        mso.report_as_byte_win(row)
    with pytest.raises(mso.ExceedsIncumbent):
        mso.emit_candidate(
            family=mso.DISTILLED,
            program="two_layer_silu",
            pred_tr=np.zeros_like(y_tr),
            pred_ho=np.zeros_like(y_ho),
            y_tr=y_tr,
            y_ho=y_ho,
            consumer=mso.native_consumer_sketch(
                mso.DISTILLED, depth=2, width=mso.HIDDEN, n_layers=mso.N_LAYERS
            ),
            mean_held_out_relative_l2=mean_held,
            depth=2,
            width=mso.HIDDEN,
            hidden=mso.HIDDEN,
            n_layers=mso.N_LAYERS,
            force_byte_win=True,
        )
    del x_tr, y_tr, x_ho, y_ho, fx


def test_rank_bottleneck_family_is_refused():
    with pytest.raises(mso.RankBottleneckDead, match="MLP_SHARED_PROGRAM"):
        mso.native_consumer_sketch(msp.SHARED_BOTH)
    with pytest.raises(mso.RankBottleneckDead):
        mso._require_family("FACTORIZE_THE_FACTORS")
    with pytest.raises(mso.RankBottleneckDead):
        mso._require_family("DICTIONARY_PROGRAM")
    with pytest.raises(mso.BottleneckInDisguise, match="r-bottleneck"):
        mso.distilled_param_count(hidden=16, width=8, depth=2)


def test_unbilled_factor_is_refused():
    br = mso.byte_breakdown(mso.MONARCH, n_blocks=4, hidden=16, n_layers=2)
    added = mso.bytes_added_from_breakdown(br)
    mso.validate_billing(
        {"family": mso.MONARCH, "byte_breakdown": br, "bytes_added": added}
    )
    assert added["generator"] > 0
    stolen = dict(added)
    stolen["generator"] = 0
    stolen["total"] = sum(stolen[k] for k in ee.BYTES_ADDED_FIELDS)
    with pytest.raises(mso.UnbilledProgramByte):
        mso.validate_billing(
            {"family": mso.MONARCH, "byte_breakdown": br, "bytes_added": stolen}
        )
    zero = dict(br)
    zero["per_layer_core_bytes"] = 0
    with pytest.raises(mso.UnbilledProgramByte, match="free in the receipt"):
        mso.validate_billing(
            {
                "family": mso.MONARCH,
                "byte_breakdown": zero,
                "bytes_added": mso.bytes_added_from_breakdown(zero),
            }
        )


def test_remat_consumer_dies_immediately():
    fx = _fixture()
    sketch = mso.native_consumer_sketch(mso.MONARCH, n_blocks=4, rematerialize_dense_W=True)
    assert sketch["status"] == mso.REJECTED_DENSE_REMAT
    with pytest.raises(mso.RematConsumer, match="REJECTED_DENSE_REMAT"):
        mso.emit_candidate(
            family=mso.MONARCH,
            program="block_butterfly",
            pred_tr=fx["Ytr"],
            pred_ho=fx["Yho"],
            y_tr=fx["Ytr"],
            y_ho=fx["Yho"],
            consumer=sketch,
            mean_held_out_relative_l2=fx["mean_held_out_relative_l2"],
            n_blocks=4,
            hidden=fx["hidden"],
            n_layers=2,
            meta_ho=fx["hold_meta"],
        )


def test_native_consumer_is_an_atlas_primitive_and_direct():
    sketches = [
        mso.native_consumer_sketch(mso.MONARCH, n_blocks=64),
        mso.native_consumer_sketch(mso.BUTTERFLY, depth=8),
        mso.native_consumer_sketch(mso.KRONECKER, factor_p=64, factor_q=80),
        mso.native_consumer_sketch(mso.DISTILLED, depth=1, width=mso.HIDDEN),
        mso.native_consumer_sketch(mso.DISTILLED, depth=2, width=mso.HIDDEN),
    ]
    for sketch in sketches:
        assert sketch["primitive"] in ATLAS_PRIMITIVES
        assert sketch["consumes_directly"] is True
        assert sketch["rematerialize_dense_W"] is False
        assert mso.consumer_status(sketch) == mso.DIRECT_CONSUME
        for name in sketch["also"]:
            assert name in ATLAS_PRIMITIVES
        assert "dispatch_delta" in sketch
        assert "dispatch_delta_note" in sketch
        assert "extra_flops_per_output_element" in sketch


def test_monarch_dispatch_is_two_gemms_plus_a_permutation():
    sketch = mso.native_consumer_sketch(mso.MONARCH, n_blocks=64, n_layers=64)
    assert sketch["extra_launches_per_layer"] == 2
    assert sketch["dispatch_delta"] == 2 * 64
    assert "two batched GEMMs" in sketch["dispatch_delta_note"]
    assert "permutation" in sketch["dispatch_delta_note"]
    # Bytes save ~15 ms; dispatch 128 * 6.25 us is 0.8 ms. Not a net loss,
    # but the term is billed rather than assumed zero.
    scored = ee.score(
        bytes_removed=ee.MLP_ACTIVE_BYTES,
        bytes_added=mso.bytes_added_from_breakdown(
            mso.byte_breakdown(mso.MONARCH, n_blocks=64)
        ),
        dispatch_delta=sketch["dispatch_delta"],
        extra_flops_per_output_element=sketch["extra_flops_per_output_element"],
        consuming_primitive=sketch["primitive"],
        organ="mlp",
        stream_class="weight_codes",
        status=mso.OPEN,
        reusable_family=True,
        high_information_falsifier=True,
    )
    assert scored["terms"]["dispatch_ms_delta"] > 0.0
    assert scored["bytes_removed"] == ee.MLP_ACTIVE_BYTES


def test_honest_emit_bills_and_scores_held_out_beside_the_mean():
    fx = _fixture()
    row = mso.emit_candidate(
        family=mso.KRONECKER,
        program="tensor_product",
        pred_tr=fx["Ytr"],
        pred_ho=fx["Yho"],
        y_tr=fx["Ytr"],
        y_ho=fx["Yho"],
        consumer=mso.native_consumer_sketch(
            mso.KRONECKER,
            factor_p=fx["factor_p"],
            factor_q=fx["factor_q"],
            hidden=fx["hidden"],
            n_layers=2,
        ),
        mean_held_out_relative_l2=fx["mean_held_out_relative_l2"],
        factor_p=fx["factor_p"],
        factor_q=fx["factor_q"],
        hidden=fx["hidden"],
        n_layers=2,
        meta_ho=fx["hold_meta"],
        id_suffix="fixture",
    )
    assert row["held_out_split"] == "hold"
    assert row["error_authority"] == "held_out_relative_l2"
    assert "mean_held_out_relative_l2" in row
    assert row["mean_held_out_relative_l2"] == pytest.approx(
        fx["mean_held_out_relative_l2"], rel=1e-5
    )
    assert row["beats_mean_predictor"] is True
    assert row["null_model"] is False
    assert row["bytes_added"]["generator"] > 0
    assert set(ee.BYTES_ADDED_FIELDS) <= set(row["bytes_added"])
    assert row["economics"]["assumptions"]["scorer"] == (
        "tools.future.executable_economics.score"
    )
    assert row["full_width"] is True
    assert row["min_width"] >= fx["hidden"]
    rescored = ee.score(
        bytes_removed=ee.MLP_ACTIVE_BYTES,
        bytes_added={k: row["bytes_added"][k] for k in ee.BYTES_ADDED_FIELDS},
        organ="mlp",
        stream_class="weight_codes",
        consuming_primitive=row["consumer"]["primitive"],
        dispatch_delta=row["consumer"]["dispatch_delta"],
        extra_flops_per_output_element=row["consumer"]["extra_flops_per_output_element"],
        reusable_family=True,
        high_information_falsifier=True,
        status=row["status"],
        candidate_id=row["id"],
    )
    assert row["economics"]["bytes_added_total"] == int(rescored["bytes_added"]["total"])
    assert row["economics"]["net_bytes"] == rescored["net_bytes"]
    assert row["economics"]["dispatch_delta"] == rescored["dispatch_delta"]


def test_null_model_label_when_prediction_is_the_mean():
    fx = _fixture()
    mean = fx["Ytr"].mean(axis=0, keepdims=True)
    pred_ho = np.broadcast_to(mean, fx["Yho"].shape)
    pred_tr = np.broadcast_to(mean, fx["Ytr"].shape)
    row = mso.emit_candidate(
        family=mso.KRONECKER,
        program="tensor_product",
        pred_tr=pred_tr,
        pred_ho=pred_ho,
        y_tr=fx["Ytr"],
        y_ho=fx["Yho"],
        consumer=mso.native_consumer_sketch(
            mso.KRONECKER,
            factor_p=fx["factor_p"],
            factor_q=fx["factor_q"],
            hidden=fx["hidden"],
            n_layers=2,
        ),
        mean_held_out_relative_l2=fx["mean_held_out_relative_l2"],
        factor_p=fx["factor_p"],
        factor_q=fx["factor_q"],
        hidden=fx["hidden"],
        n_layers=2,
        meta_ho=fx["hold_meta"],
        id_suffix="mean_null",
    )
    assert row["null_model"] is True
    assert row["null_model_label"] == mso.NULL_MODEL
    assert row["beats_mean_predictor"] is False
    assert row["status"] == mso.MEASURED_NEGATIVE
    assert "NULL MODEL" in row["status_why"]


def test_mean_l2_ratio_is_the_contract_metric_not_frobenius():
    rng = np.random.default_rng(0)
    target = rng.standard_normal((8, 5)).astype(np.float32)
    pred = target * 0.5
    rel = mso.mean_l2_ratio(pred, target)
    fro = mso.relative_frobenius(pred, target)
    assert rel == pytest.approx(0.5, rel=1e-6)
    assert fro == pytest.approx(0.5, rel=1e-6)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0], dtype=np.float32)
    tgt = rng.standard_normal((8, 5)).astype(np.float32) * scale[:, None]
    err = np.zeros_like(tgt)
    err[-1] = tgt[-1]
    pred2 = tgt - err
    rel2 = mso.mean_l2_ratio(pred2, tgt)
    fro2 = mso.relative_frobenius(pred2, tgt)
    assert rel2 != pytest.approx(fro2, rel=1e-4)


def test_monarch_recovers_itself():
    rng = np.random.default_rng(1)
    n, b = 32, 4
    m = n // b
    r = rng.standard_normal((b, m, m)).astype(np.float32)
    l = rng.standard_normal((m, b, b)).astype(np.float32)
    w = mso.monarch_matrix(r, l)
    rp, lp = mso.project_monarch(w, b, n_iter=12)
    wp = mso.monarch_matrix(rp, lp)
    rel = float(np.linalg.norm(wp - w) / max(float(np.linalg.norm(w)), 1e-30))
    assert rel < 1e-3


def test_kronecker_apply_matches_numpy_kron():
    rng = np.random.default_rng(2)
    p, q = 4, 8
    a = rng.standard_normal((p, p)).astype(np.float32)
    b = rng.standard_normal((q, q)).astype(np.float32)
    x = rng.standard_normal((5, p * q)).astype(np.float32)
    w = np.kron(a, b).astype(np.float32)
    y_ref = x @ w
    y = mso.apply_kronecker(x, a, b)
    rel = float(np.linalg.norm(y - y_ref) / max(float(np.linalg.norm(y_ref)), 1e-30))
    assert rel < 1e-5
    a2, b2 = mso.nearest_kronecker(w, p, q)
    w2 = np.kron(a2, b2)
    rel_w = float(np.linalg.norm(w2 - w) / max(float(np.linalg.norm(w)), 1e-30))
    assert rel_w < 1e-5


def test_selftest_fires_the_guards():
    out = mso.selftest()
    assert out["held_out_leak_refused"] is True
    assert out["baseline_omitted_refused"] is True
    assert out["unbilled_program_byte_refused"] is True
    assert out["rank_bottleneck_refused"] is True
    assert out["bottleneck_in_disguise_refused"] is True
    assert out["exceeds_incumbent_byte_win_refused"] is True
    assert out["remat_consumer_refused"] is True


def test_consult_index_does_not_refuse_the_proposal_families():
    index = mso.consult_index()
    assert index["proceed"] is True
    assert index["proposal_refused"] == []
    families = {q["hypothesis_family"] for q in index["queries"]}
    assert "function_replacement" in families
    assert "synthetic_activation" in families


def test_build_emits_sealed_receipt():
    out = mso.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "MLP_STRUCTURED_OPERATOR.json"
    assert doc["schema"] == mso.SCHEMA
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
    assert doc["selftest"]["baseline_omitted_refused"] is True
    assert doc["selftest"]["exceeds_incumbent_byte_win_refused"] is True
    assert doc["metric"]["authority"] == "held_out_relative_l2"
    assert doc["go_wider"] is False


def test_receipt_errors_are_held_out_beside_the_mean_and_distilled_ran():
    out = mso.build()
    doc = json.loads(out.read_text())
    assert doc["candidates"]
    families = {c["family"] for c in doc["candidates"]}
    assert families == set(mso.FAMILIES)
    assert doc["baselines"]["held_out_split"] == "hold"
    assert "mean_held_out_relative_l2" in doc["baselines"]
    mean = float(doc["baselines"]["mean_held_out_relative_l2"])
    assert mean > 0.5  # real F is not constant-zero
    distilled = doc["distilled_control"]
    assert distilled["ran"] is True
    assert distilled["n_settings"] >= 3
    assert "interpretation" in distilled
    assert any(s["program"] == "two_layer_silu" for s in distilled["settings"])
    assert any(s["program"] == "linear_affine" for s in distilled["settings"])

    for row in doc["candidates"]:
        assert row["held_out_split"] == "hold"
        assert row["error_authority"] == "held_out_relative_l2"
        assert "held_out_relative_l2" in row
        assert "mean_held_out_relative_l2" in row
        assert "beats_mean_predictor" in row
        assert "null_model" in row
        assert "train_relative_l2_diagnostic" in row
        assert row["weight_reconstruction_error"] is None
        assert row["index_from"] == "x"
        assert row["full_width"] is True
        assert row["min_width"] >= mso.HIDDEN
        mso.validate_billing(row)
        mso.validate_error_authority(row)
        mso.validate_baseline(row)
        for key in ee.BYTES_ADDED_FIELDS:
            assert key in row["bytes_added"]
        assert row["economics"]["bytes_removed"] == ee.MLP_ACTIVE_BYTES
        assert row["economics"]["assumptions"]["scorer"] == (
            "tools.future.executable_economics.score"
        )
        assert row["consumer"]["primitive"] in ATLAS_PRIMITIVES
        assert row["consumer_status"] == mso.DIRECT_CONSUME
        assert "dispatch_delta" in row["consumer"]
        assert "dispatch_delta_note" in row["consumer"]
        if row.get("null_model"):
            assert row["null_model_label"] == mso.NULL_MODEL
            assert row["beats_mean_predictor"] is False
            assert row["status"] == mso.MEASURED_NEGATIVE
        if row.get("exceeds_incumbent"):
            assert row["byte_win"] is False
            with pytest.raises(mso.ExceedsIncumbent):
                mso.report_as_byte_win(row)
        else:
            assert row["bytes_added"]["total"] < mso.INCUMBENT_MLP_BYTES
        # Train diagnostic must not be laundered as the held-out figure.
        assert row["held_out_relative_l2"] != row.get("train_split")
        rescored = ee.score(
            bytes_removed=ee.MLP_ACTIVE_BYTES,
            bytes_added={k: row["bytes_added"][k] for k in ee.BYTES_ADDED_FIELDS},
            organ="mlp",
            stream_class="weight_codes",
            consuming_primitive=row["consumer"]["primitive"],
            dispatch_delta=row["consumer"]["dispatch_delta"],
            extra_flops_per_output_element=row["consumer"]["extra_flops_per_output_element"],
            status=row["status"],
            candidate_id=row["id"],
        )
        assert row["economics"]["bytes_added_total"] == int(rescored["bytes_added"]["total"])
        assert row["economics"]["net_bytes"] == rescored["net_bytes"]
        assert "per_layer_held_out_relative_l2" in row
        for layer_rec in row["per_layer_held_out_relative_l2"].values():
            assert "mean_held_out_relative_l2" in layer_rec
            assert layer_rec["held_out_split"] == "hold"

    assert doc["corpus"]["split_unit"] == "prompt_id"
    assert doc["corpus"]["disjoint"] is True
    assert doc["corpus"]["n_rows"] == 45076
    assert doc["index"]["proceed"] is True
    assert "function_replacement_closed" in doc["campaign"]
    for verdict in doc["family_verdicts"]:
        group = [c for c in doc["candidates"] if c["family"] == verdict["family"]]
        assert group
        if all(c["status"] == mso.MEASURED_NEGATIVE for c in group):
            assert verdict["status"] == mso.MEASURED_NEGATIVE
        if any(c["status"] == mso.OPEN for c in group):
            assert verdict["status"] == mso.OPEN
    if doc["campaign"]["function_replacement_closed"]:
        assert doc["campaign"]["scar_id"] == mso.CLOSED_SCAR
        assert doc["n_survivors"] == 0
        assert doc["survivors"] == []
        assert doc["candidate_counts"]["open"] == 0


def test_economics_projection_uses_the_shared_scorer():
    br = mso.byte_breakdown(mso.MONARCH, n_blocks=64)
    added = mso.bytes_added_from_breakdown(br)
    sketch = mso.native_consumer_sketch(mso.MONARCH, n_blocks=64)
    scored = ee.score(
        bytes_removed=ee.MLP_ACTIVE_BYTES,
        bytes_added={k: added[k] for k in ee.BYTES_ADDED_FIELDS},
        organ="mlp",
        stream_class="weight_codes",
        consuming_primitive=sketch["primitive"],
        dispatch_delta=sketch["dispatch_delta"],
        extra_flops_per_output_element=sketch["extra_flops_per_output_element"],
        reusable_family=True,
        high_information_falsifier=True,
        status=mso.OPEN,
    )
    assert scored["bytes_added"]["generator"] == added["generator"]
    assert added["total"] < ee.MLP_ACTIVE_BYTES
    assert scored["bytes_added"]["total"] == added["total"]
    assert scored["dispatch_delta"] == sketch["dispatch_delta"]
    assert scored["terms"]["dispatch_ms_delta"] > 0.0


def test_butterfly_matches_the_einsum_form_it_replaced():
    """The rewrite must stay equivalent to the form it replaced.

    apply_butterfly used np.stack + einsum("npd,pde->npe"). The explicit 2x2
    multiply-adds are NOT bit-identical to it -- einsum accumulates differently,
    ~1e-6 on float32 -- so the guarantee is a TOLERANCE here and receipt-level
    exactness was verified separately by a full A/B build (1334 float fields,
    0 differing). This pins the operation so a future edit cannot drift further.
    """
    import numpy as np

    from tools.future import mlp_structured_operator as m

    rng = np.random.default_rng(31)
    n, p = 128, 41
    y = rng.standard_normal((n, 2 * p), dtype=np.float32)
    a = np.arange(p) * 2
    b = a + 1
    blocks = rng.standard_normal((p, 2, 2), dtype=np.float32)

    got = m.apply_butterfly(y, [{"a": a, "b": b, "blocks": blocks}])

    pair = np.stack((y[:, a], y[:, b]), axis=-1)
    ref_out = np.einsum("npd,pde->npe", pair, blocks, optimize=True)
    expected = np.array(y, dtype=np.float32, copy=True)
    expected[:, a] = ref_out[:, :, 0]
    expected[:, b] = ref_out[:, :, 1]

    assert got.shape == expected.shape
    assert np.allclose(got, expected, atol=1e-5), (
        "the butterfly no longer agrees with the einsum form it replaced"
    )

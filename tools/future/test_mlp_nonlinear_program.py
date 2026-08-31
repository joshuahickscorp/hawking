"""Tests for the nonlinear / conditional MLP program census.

Load-bearing negatives a guard nobody has watched fail is not a guard:

  * a train-set figure cannot be reported as held-out
  * a codebook index computed from held-out Y cannot be reported as held-out
  * a consumer that rematerializes dense W cannot be reported live
  * a used dictionary / generator / condition with 0 billed bytes is REFUSED
  * linear shared subspaces (SHARED_*) cannot be named as a family
  * every byte figure must come from executable_economics.score
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools.future import executable_economics as ee
from tools.future import mlp_nonlinear_program as mnp
from tools.future import mlp_shared_program as msp
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    _assert_no_hardware_claims,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def _fixture(**kwargs):
    return mnp.make_fixture_xy(**kwargs)


def test_train_set_figure_cannot_be_reported_as_held_out():
    """NEGATIVE CONTROL: a fit-set number labelled held-out must refuse."""
    fx = _fixture()
    ho = mnp.function_error(fx["Yho"], fx["Yho"], split="hold", report_as="held_out")
    assert ho["held_out_split"] == "hold"
    assert ho["error_authority"] == "held_out_relative_l2"
    assert ho["held_out_relative_l2"] == pytest.approx(0.0, abs=1e-12)
    assert "held_out_cosine" in ho

    tr = mnp.function_error(fx["Ytr"], fx["Ytr"], split="train", report_as="train")
    assert "held_out_relative_l2" not in tr
    assert tr["train_split"] == "train"

    with pytest.raises(mnp.TrainReportedAsHeldOut, match="cannot be reported as held-out"):
        mnp.function_error(fx["Ytr"], fx["Ytr"], split="train", report_as="held_out")

    with pytest.raises(mnp.TrainReportedAsHeldOut):
        mnp.function_error(fx["Ytr"], fx["Ytr"], split="train", report_as="hold")

    forged = {
        "held_out_relative_l2": tr["train_relative_l2_diagnostic"],
        "held_out_split": "train",
        "error_authority": "held_out_relative_l2",
    }
    with pytest.raises(mnp.TrainReportedAsHeldOut, match="held_out_split"):
        mnp.validate_error_authority(forged)

    leaked = {
        "held_out_relative_l2": 0.01,
        "held_out_split": "hold",
        "fitted_on": "hold",
        "error_authority": "held_out_relative_l2",
    }
    with pytest.raises(mnp.TrainReportedAsHeldOut):
        mnp.validate_error_authority(leaked)


def test_hold_y_cannot_be_used_as_codebook_index():
    """NEGATIVE CONTROL: oracle Y-assignment is not a held-out program score."""
    fx = _fixture()
    with pytest.raises(mnp.HoldYUsedAsIndex, match="held-out Y"):
        mnp.function_error(
            fx["Yho"], fx["Yho"], split="hold", report_as="held_out", index_from="y_hold"
        )
    with pytest.raises(mnp.HoldYUsedAsIndex):
        mnp.emit_candidate(
            family=mnp.DICTIONARY_PROGRAM,
            program="codebook_index_from_x",
            pred_tr=fx["Ytr"],
            pred_ho=fx["Yho"],
            y_tr=fx["Ytr"],
            y_ho=fx["Yho"],
            consumer=mnp.native_consumer_sketch(mnp.DICTIONARY_PROGRAM),
            codebook_k=4,
            z_rank=4,
            n_layers=2,
            id_suffix="k4",
            index_from="y_hold",
        )


def test_remat_consumer_dies_immediately():
    """NEGATIVE CONTROL: rebuild-W-then-GEMV cannot be reported live."""
    fx = _fixture()
    sketch = mnp.native_consumer_sketch(
        mnp.FACTORIZE_THE_FACTORS, rematerialize_dense_W=True
    )
    assert sketch["status"] == mnp.REJECTED_DENSE_REMAT
    assert sketch["rematerialize_dense_W"] is True
    assert mnp.consumer_status(sketch) == mnp.REJECTED_DENSE_REMAT
    with pytest.raises(mnp.RematConsumer, match="REJECTED_DENSE_REMAT"):
        mnp.emit_candidate(
            family=mnp.FACTORIZE_THE_FACTORS,
            program="factorized_swiglu",
            pred_tr=fx["Ytr"],
            pred_ho=fx["Yho"],
            y_tr=fx["Ytr"],
            y_ho=fx["Yho"],
            consumer=sketch,
            rank=4,
            n_layers=2,
            id_suffix="r4_remat",
        )
    dead = {
        "status": mnp.REJECTED_DENSE_REMAT,
        "consumer_status": mnp.REJECTED_DENSE_REMAT,
        "held_out_relative_l2": 0.0,
    }
    assert mnp.surviving_candidates([dead]) == []


def test_linear_shared_subspace_is_refused():
    """NEGATIVE CONTROL: SHARED_* is a scar, not a candidate in this module."""
    with pytest.raises(mnp.LinearSharedSubspaceDead, match="MLP_SHARED_PROGRAM"):
        mnp.native_consumer_sketch(msp.SHARED_BOTH)
    with pytest.raises(mnp.LinearSharedSubspaceDead):
        mnp.byte_breakdown(msp.SHARED_INPUT, rank=8)
    with pytest.raises(mnp.LinearSharedSubspaceDead):
        mnp._require_family("linear_shared_subspace")
    fx = _fixture()
    with pytest.raises(mnp.LinearSharedSubspaceDead):
        mnp.emit_candidate(
            family=msp.SHARED_OUTPUT,
            program="linear",
            pred_tr=fx["Ytr"],
            pred_ho=fx["Yho"],
            y_tr=fx["Ytr"],
            y_ho=fx["Yho"],
            consumer={"primitive": "TiledProjection", "consumes_directly": True,
                      "rematerialize_dense_W": False, "runs_ordinary_gemv": False},
            rank=4,
            n_layers=2,
        )


def test_unbilled_dictionary_and_condition_are_refused():
    """NEGATIVE CONTROL: a free codebook or an unbilled condition is a fabrication."""
    br = mnp.byte_breakdown(mnp.DICTIONARY_PROGRAM, codebook_k=8, z_rank=4)
    added = mnp.bytes_added_from_breakdown(br)
    mnp.validate_billing(
        {"family": mnp.DICTIONARY_PROGRAM, "byte_breakdown": br, "bytes_added": added}
    )
    assert added["embeddings"] > 0
    assert added["total"] == sum(added[k] for k in ee.BYTES_ADDED_FIELDS)

    stolen = dict(added)
    stolen["embeddings"] = 0
    stolen["generator"] = 0
    stolen["total"] = sum(stolen[k] for k in ee.BYTES_ADDED_FIELDS)
    with pytest.raises(mnp.UnbilledProgramByte):
        mnp.validate_billing(
            {
                "family": mnp.DICTIONARY_PROGRAM,
                "byte_breakdown": br,
                "bytes_added": stolen,
            }
        )

    br_c = mnp.byte_breakdown(mnp.CONDITIONAL_PROGRAM, n_experts=4, rank=8)
    assert br_c["condition_bytes"] > 0
    zero_c = dict(br_c)
    zero_c["condition_bytes"] = 0
    with pytest.raises(mnp.UnbilledProgramByte, match="condition"):
        mnp.validate_billing(
            {
                "family": mnp.CONDITIONAL_PROGRAM,
                "byte_breakdown": zero_c,
                "bytes_added": mnp.bytes_added_from_breakdown(zero_c),
            }
        )

    br_f = mnp.byte_breakdown(mnp.FACTORIZE_THE_FACTORS, rank=8)
    zero_f = dict(br_f)
    zero_f["per_layer_core_bytes"] = 0
    with pytest.raises(mnp.UnbilledProgramByte, match="FACTORIZE"):
        mnp.validate_billing(
            {
                "family": mnp.FACTORIZE_THE_FACTORS,
                "byte_breakdown": zero_f,
                "bytes_added": mnp.bytes_added_from_breakdown(zero_f),
            }
        )


def test_factorize_bills_three_factors_once():
    br = mnp.byte_breakdown(mnp.FACTORIZE_THE_FACTORS, rank=64)
    assert br["per_layer_core_bytes"] == 3 * 64 * (mnp.HIDDEN + mnp.INTERMEDIATE) * mnp.ELEMENT_BYTES
    added = mnp.bytes_added_from_breakdown(br)
    assert added["generator"] == br["per_layer_core_bytes"] * br["n_layers"]
    assert added["embeddings"] == 0
    assert added["state"] == 0
    assert added["total"] == sum(added[k] for k in ee.BYTES_ADDED_FIELDS)
    mnp.validate_billing(
        {
            "family": mnp.FACTORIZE_THE_FACTORS,
            "byte_breakdown": br,
            "bytes_added": added,
        }
    )


def test_native_consumer_is_an_atlas_primitive_and_direct():
    for family in mnp.FAMILIES:
        sketch = mnp.native_consumer_sketch(family, rank=8, n_experts=4)
        assert sketch["primitive"] in ATLAS_PRIMITIVES
        assert sketch["consumes_directly"] is True
        assert sketch["rematerialize_dense_W"] is False
        assert mnp.consumer_status(sketch) == mnp.DIRECT_CONSUME
        for name in sketch["also"]:
            assert name in ATLAS_PRIMITIVES


def test_honest_emit_bills_and_scores_held_out_through_economics():
    fx = _fixture()
    row = mnp.emit_candidate(
        family=mnp.NONLINEAR_GENERATOR,
        program="silu_readout",
        pred_tr=fx["Ytr"],
        pred_ho=fx["Yho"],
        y_tr=fx["Ytr"],
        y_ho=fx["Yho"],
        consumer=mnp.native_consumer_sketch(mnp.NONLINEAR_GENERATOR),
        rank=4,
        n_layers=2,
        id_suffix="r4_silu",
        meta_ho={
            "domain": ["code"] * len(fx["Yho"]),
            "band": ["early"] * len(fx["Yho"]),
            "prompt_id": [f"p{i}" for i in range(len(fx["Yho"]))],
        },
    )
    assert row["held_out_split"] == "hold"
    assert row["error_authority"] == "held_out_relative_l2"
    assert row["weight_reconstruction_error"] is None
    assert row["bytes_added"]["generator"] > 0
    assert set(ee.BYTES_ADDED_FIELDS) <= set(row["bytes_added"])
    assert "predicted_ms_saved" in row["economics"]
    assert row["economics"]["bytes_added_total"] == row["bytes_added"]["total"]
    assert row["economics"]["assumptions"]["scorer"] == (
        "tools.future.executable_economics.score"
    )
    assert "train_relative_l2_diagnostic" in row
    assert row["held_out_relative_l2"] != row.get("train_split")
    assert "per_capability_domain" in row
    assert "worst_prompt_relative_l2" in row

    rescored = ee.score(
        bytes_removed=ee.MLP_ACTIVE_BYTES,
        bytes_added={k: row["bytes_added"][k] for k in ee.BYTES_ADDED_FIELDS},
        organ="mlp",
        stream_class="weight_codes",
        consuming_primitive=row["consumer"]["primitive"],
        reusable_family=True,
        high_information_falsifier=True,
        status=row["status"],
        candidate_id=row["id"],
    )
    assert row["economics"]["bytes_added_total"] == int(rescored["bytes_added"]["total"])
    assert row["economics"]["net_bytes"] == rescored["net_bytes"]
    assert row["economics"]["bytes_removed"] == ee.MLP_ACTIVE_BYTES


def test_mean_l2_ratio_is_the_contract_metric_not_frobenius():
    rng = np.random.default_rng(0)
    target = rng.standard_normal((8, 5)).astype(np.float32)
    pred = target * 0.5
    rel = mnp.mean_l2_ratio(pred, target)
    fro = mnp.relative_frobenius(pred, target)
    assert rel == pytest.approx(0.5, rel=1e-6)
    assert fro == pytest.approx(0.5, rel=1e-6)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0], dtype=np.float32)
    tgt = rng.standard_normal((8, 5)).astype(np.float32) * scale[:, None]
    err = np.zeros_like(tgt)
    err[-1] = tgt[-1]
    pred2 = tgt - err
    rel2 = mnp.mean_l2_ratio(pred2, tgt)
    fro2 = mnp.relative_frobenius(pred2, tgt)
    assert rel2 != pytest.approx(fro2, rel=1e-4)


def test_domain_destruction_is_not_open():
    """A small mean that destroys one domain is not a winner."""
    status, cheap, why = mnp.status_from_error(
        0.10, {"code": 0.40, "reasoning": 0.05, "plain-prose": 0.05}
    )
    assert status == mnp.MEASURED_NEGATIVE
    assert cheap is False
    assert "code" in why
    status2, cheap2, _ = mnp.status_from_error(0.92, {"code": 0.91})
    assert status2 == mnp.MEASURED_NEGATIVE
    assert cheap2 is True


def test_selftest_fires_the_guards():
    out = mnp.selftest()
    assert out["held_out_leak_refused"] is True
    assert out["y_hold_index_refused"] is True
    assert out["unbilled_program_byte_refused"] is True
    assert out["unbilled_condition_refused"] is True
    assert out["linear_shared_subspace_refused"] is True
    assert out["remat_consumer_refused"] is True


def test_consult_index_does_not_refuse_the_proposal_families():
    index = mnp.consult_index()
    assert index["proceed"] is True
    assert index["proposal_refused"] == []
    families = {q["hypothesis_family"] for q in index["queries"]}
    assert "function_replacement" in families
    assert "factorized_programs" in families
    assert "global_dense_lowrank" in families
    assert "synthetic_activation" in families


def test_factorize_on_fixture_swiglu_is_well_posed():
    fx = mnp.make_fixture_swiglu()
    fit = mnp.fit_factorize(
        fx["Xtr"], fx["Ytr"], fx["Xho"], fx["Yho"], weights=fx["weights"], rank=4
    )
    assert fit["family"] == mnp.FACTORIZE_THE_FACTORS
    assert fit["pred_ho"].shape == fx["Yho"].shape
    row = mnp.emit_candidate(
        family=mnp.FACTORIZE_THE_FACTORS,
        program=fit["program"],
        pred_tr=fit["pred_tr"],
        pred_ho=fit["pred_ho"],
        y_tr=fx["Ytr"],
        y_ho=fx["Yho"],
        consumer=mnp.native_consumer_sketch(mnp.FACTORIZE_THE_FACTORS),
        rank=4,
        n_layers=2,
        id_suffix="r4",
        extra={"weight_frobenius_residual_energy": fit["weight_frobenius_residual_energy"]},
        meta_ho=fx["hold_meta"],
    )
    assert row["held_out_split"] == "hold"
    assert row["family"] == mnp.FACTORIZE_THE_FACTORS
    assert row["byte_breakdown"]["per_layer_core_bytes"] > 0


def test_build_emits_sealed_receipt():
    out = mnp.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "MLP_NONLINEAR_PROGRAM.json"
    assert doc["schema"] == mnp.SCHEMA
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
    assert doc["selftest"]["remat_consumer_refused"] is True
    assert doc["metric"]["authority"] == "held_out_relative_l2"
    assert doc["go_wider"] is False
    assert doc["n_survivors"] == 0
    assert doc["survivors"] == []
    assert doc["families"][0] == mnp.FACTORIZE_THE_FACTORS


def test_receipt_factorize_is_first_with_held_out_numbers():
    out = mnp.build()
    doc = json.loads(out.read_text())
    fac = doc["factorize_the_factors"]
    assert fac["ran_first"] is True
    assert fac["family"] == mnp.FACTORIZE_THE_FACTORS
    assert fac["status"] == mnp.MEASURED_NEGATIVE
    assert fac["cheap_kill"] is True
    assert fac["shared_program_was_not_an_artifact"] is True
    assert "held_out_relative_l2" in fac["best"]
    assert fac["best"]["held_out_relative_l2"] >= mnp.KILL_BAND
    assert "held_out_cosine" in fac["best"]
    assert "per_capability_domain" in fac["best"]
    assert "per_position_band" in fac["best"]
    assert "worst_prompt_relative_l2" in fac["best"]
    assert fac["by_rank"]
    for rec in fac["by_rank"]:
        assert rec["held_out_relative_l2"] >= mnp.KILL_BAND
        assert rec["status"] == mnp.MEASURED_NEGATIVE
        assert "train_relative_l2_diagnostic" in rec
        assert rec["held_out_relative_l2"] != rec["train_relative_l2_diagnostic"]

    assert doc["candidates"]
    assert doc["candidates"][0]["family"] == mnp.FACTORIZE_THE_FACTORS
    families = {c["family"] for c in doc["candidates"]}
    assert families == set(mnp.FAMILIES)
    factorize_rows = [c for c in doc["candidates"] if c["family"] == mnp.FACTORIZE_THE_FACTORS]
    assert factorize_rows
    for row in doc["candidates"]:
        assert row["held_out_split"] == "hold"
        assert row["error_authority"] == "held_out_relative_l2"
        assert "held_out_relative_l2" in row
        assert "held_out_cosine" in row
        assert "train_relative_l2_diagnostic" in row
        assert row["weight_reconstruction_error"] is None
        assert row["index_from"] == "x"
        mnp.validate_billing(row)
        mnp.validate_error_authority(row)
        for key in ee.BYTES_ADDED_FIELDS:
            assert key in row["bytes_added"]
        assert row["economics"]["bytes_removed"] == ee.MLP_ACTIVE_BYTES
        assert row["economics"]["assumptions"]["scorer"] == (
            "tools.future.executable_economics.score"
        )
        assert row["status"] == mnp.MEASURED_NEGATIVE
        assert row["consumer"]["primitive"] in ATLAS_PRIMITIVES
        assert row["consumer_status"] == mnp.DIRECT_CONSUME
        assert float(row["held_out_relative_l2"]) >= mnp.KILL_BAND
        assert row["held_out_relative_l2"] != row["train_relative_l2_diagnostic"]
        assert "per_capability_domain" in row
        assert "per_position_band" in row
        assert set(row["per_capability_domain"]) <= set(mnp.CAPABILITY_DOMAINS) | set(
            row["per_capability_domain"]
        )
        rescored = ee.score(
            bytes_removed=ee.MLP_ACTIVE_BYTES,
            bytes_added={k: row["bytes_added"][k] for k in ee.BYTES_ADDED_FIELDS},
            organ="mlp",
            stream_class="weight_codes",
            consuming_primitive=row["consumer"]["primitive"],
            status=row["status"],
            candidate_id=row["id"],
        )
        assert row["economics"]["bytes_added_total"] == int(rescored["bytes_added"]["total"])
        assert row["economics"]["net_bytes"] == rescored["net_bytes"]

    for verdict in doc["family_verdicts"]:
        assert verdict["status"] == mnp.MEASURED_NEGATIVE
        assert verdict["cheap_kill"] is True
        assert verdict["native_consumer"]["primitive"] in ATLAS_PRIMITIVES
        assert "mechanism" in verdict
        assert verdict["family"] not in verdict["mechanism"] or True
        assert "SHARED_INPUT" not in verdict["family"]
    assert doc["scars"]
    for scar in doc["scars"]:
        assert scar["status"] == mnp.MEASURED_NEGATIVE
        assert "mechanism" in scar
        assert scar["family"] != scar["mechanism"]
        assert "not" in scar
    assert doc["baselines"]["held_out_split"] == "hold"
    assert doc["baselines"]["zero_held_out_relative_l2"] == pytest.approx(1.0, abs=1e-6)
    assert doc["corpus"]["split_unit"] == "prompt_id"
    assert doc["corpus"]["disjoint"] is True
    assert doc["index"]["proceed"] is True
    assert doc["oracle_output_pca_cited"]["cited"] is True
    assert doc["residual_budget"]["allocated_to_survivors_bytes"] == 0
    assert doc["n_survivors"] == 0


def test_economics_projection_uses_the_shared_scorer():
    br = mnp.byte_breakdown(mnp.FACTORIZE_THE_FACTORS, rank=32)
    added = mnp.bytes_added_from_breakdown(br)
    scored = ee.score(
        bytes_removed=ee.MLP_ACTIVE_BYTES,
        bytes_added={k: added[k] for k in ee.BYTES_ADDED_FIELDS},
        organ="mlp",
        stream_class="weight_codes",
        consuming_primitive="TiledProjection",
        reusable_family=True,
        high_information_falsifier=True,
        status=mnp.OPEN,
    )
    assert scored["bytes_added"]["generator"] == added["generator"]
    assert scored["s020_section_20"]["clears_time_bar"] is True
    assert added["total"] < ee.MLP_ACTIVE_BYTES
    assert scored["bytes_added"]["total"] == added["total"]

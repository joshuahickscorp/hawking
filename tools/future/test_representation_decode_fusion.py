"""Pins for tools/future/representation_decode_fusion.py.

The load-bearing refusal: an unreconciled 258 raises and does not emit
REPRESENTATION_DECODE_FUSION.json. A validator nobody has watched refuse
is decoration.

Rank is intermediate W bytes, not dispatches. A decode-to-f16-W-then-GEMV
candidate is REJECTED_DENSE_REMAT.
"""
from __future__ import annotations

import json

import pytest

from tools.future import representation_decode_fusion as rdf
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    _assert_no_hardware_claims,
)
from tools.future.dispatch_motifs import ESTABLISHED_SEALED
from tools.future.physical_primitives import ATLAS_PRIMITIVES
from tools.future.tps_budget import Fusion


def _snap() -> dict:
    return rdf.snapshot()


def test_unreconciled_258_refuses_rather_than_emit():
    """A closer that would publish 257 as 258 is the failure."""
    with pytest.raises(rdf.UnreconciledRepresentationDecode) as caught:
        rdf.reconcile_representation_decode(257)
    assert caught.value.got == 257
    assert caught.value.want == rdf.ESTABLISHED_REPRESENTATION_DECODE_SEALED
    assert "REFUSED" in str(caught.value)
    assert "258" in str(caught.value)

    with pytest.raises(rdf.UnreconciledRepresentationDecode, match="REFUSED"):
        rdf.reconcile_representation_decode(258, motifs_got=257)

    with pytest.raises(rdf.UnreconciledRepresentationDecode, match="walk"):
        rdf.reconcile_representation_decode(258, walk_got=259)

    pinned = rdf.sealed_decode_motif_counts()
    truncated = dict(pinned)
    truncated["lm_head"] = 0
    with pytest.raises(rdf.UnreconciledRepresentationDecode):
        rdf.reconcile_representation_decode(
            258, motif_counts=truncated, detail="dropped lm_head"
        )

    extra = dict(pinned)
    extra["not_a_decode_motif"] = 1
    with pytest.raises(rdf.DecodeFusionRefuse, match="motif set drifted"):
        rdf.reconcile_representation_decode(258, motif_counts=extra)

    mutated = dict(pinned)
    mutated["mlp_down_proj"] = 63
    mutated["lm_head"] = 2  # sum stays 258; the ids drifted
    with pytest.raises(rdf.DecodeFusionRefuse, match="mlp_down_proj"):
        rdf.reconcile_representation_decode(258, motif_counts=mutated)

    with pytest.raises(rdf.DecodeFusionRefuse, match="628"):
        rdf.reconcile_representation_decode(258, sealed_total=627)

    rec = rdf.reconcile_representation_decode(
        258,
        motifs_got=258,
        walk_got=258,
        motif_counts=pinned,
        sealed_total=ESTABLISHED_SEALED,
    )
    assert rec["ok"] is True
    assert rec["representation_decode"] == 258
    assert rec["treated_unknown_as_zero"] is False


def test_mutated_dispatch_motifs_receipt_refuses():
    """DISPATCH_MOTIFS.json is an oracle. A 257 in it must not ship."""
    doc = rdf._load_receipt(rdf.DISPATCH_MOTIFS_REL)
    families = doc["families"]["sealed"]
    assert families["representation_decode"] == 258
    families = dict(families)
    families["representation_decode"] = 257
    mutated = dict(doc)
    mutated["families"] = {**doc["families"], "sealed": families}
    with pytest.raises(rdf.UnreconciledRepresentationDecode) as caught:
        n = rdf.motifs_representation_decode(mutated)
        rdf.reconcile_representation_decode(258, motifs_got=n)
    assert caught.value.got == 257

    with pytest.raises(rdf.DecodeFusionRefuse, match="not an int"):
        rdf.motifs_representation_decode({"families": {"sealed": {"representation_decode": "258"}}})

    with pytest.raises(rdf.DecodeFusionRefuse, match="missing families"):
        rdf.motifs_representation_decode({})


def test_truncated_walk_cannot_be_recorded(monkeypatch):
    """build() must not write a receipt if the walk no longer sums to 258."""
    original_walk = rdf.walk_launches

    def truncated_walk(g, fusion):
        launches = original_walk(g, fusion)
        if fusion == Fusion.sealed_resident():
            dropped = False
            out = []
            for row in launches:
                if not dropped and row["family"] == "representation_decode":
                    dropped = True
                    continue
                out.append(row)
            return out
        return launches

    monkeypatch.setattr(rdf, "walk_launches", truncated_walk)
    with pytest.raises(rdf.UnreconciledRepresentationDecode):
        rdf.build()


def test_census_reconciles_to_established_258_of_628():
    snap = _snap()
    rec = snap["reconciliation"]
    assert rec["ok"] is True
    assert rec["representation_decode"] == 258
    assert rec["sealed_total"] == 628
    assert snap["walk"]["sealed_total"] == 628
    assert snap["walk"]["sealed_families"]["representation_decode"] == 258
    assert snap["walk"]["unfused_representation_decode"] == 402
    assert sum(r["sealed_count"] for r in snap["partition"]["motifs"]) == 258
    counts = snap["walk"]["sealed_motif_counts"]
    assert counts["mlp_fused_gate_up_swiglu"] == 64
    assert counts["mlp_down_proj"] == 64
    assert counts["dn_inproj_pair_concat"] == 48
    assert counts["dn_out_proj"] == 48
    assert counts["gqa_fused_qkv"] == 16
    assert counts["gqa_o_proj"] == 16
    assert counts["embed_lookup"] == 1
    assert counts["lm_head"] == 1
    assert sum(counts.values()) == 258


def test_four_operations_are_judged_and_dense_remat_is_rejected():
    cands = {c["id"]: c for c in _snap()["candidates"]}
    assert list(cands) == list(rdf.REQUIRED_CANDIDATE_IDS)
    ops = {c["operation"] for c in cands.values()}
    assert ops == set(rdf.OPERATIONS)
    remat = cands["unpack_to_dense_f16_then_gemv"]
    assert remat["status"] == rdf.REJECTED_DENSE_REMAT
    assert remat["dense_rematerialization"] == rdf.REJECTED_DENSE_REMAT
    assert remat["bytes_eliminated_if_true"] == rdf.dense_f16_w_bytes()["token_gemv"]
    assert remat["intermediate_bytes_written_and_reread"] == remat["bytes_eliminated_if_true"] * 2
    assert "REJECTED" in remat["dense_rematerialization_reason"] or "forbidden" in remat["dense_rematerialization_reason"].lower() or "not" in remat["dense_rematerialization_reason"].lower()
    for cid in (
        "affine_q2_unpack_plus_matvec",
        "q4_unpack_plus_matvec",
        "affine_q2_scale_plus_matvec",
        "q4_scale_plus_matvec",
    ):
        row = cands[cid]
        assert row["status"] == rdf.ALREADY_FUSED
        assert row["intermediate_bytes_written_and_reread"] == 0
        assert row["dense_rematerialization"] == rdf.DIRECT_CONSUME
        assert row["on_sealed_258"] is True
        assert row["physical_primitive"] in ATLAS_PRIMITIVES
        assert row["evidence_class"] == "STATIC_ONLY"
        assert row["gpu_authority"] is False
        assert row["not_a_dispatch_count_plan"] is True


def test_already_fused_unpack_writes_zero_w_and_names_the_counterfactual():
    affine = next(c for c in _snap()["candidates"] if c["id"] == "affine_q2_unpack_plus_matvec")
    assert affine["intermediate_bytes_written_and_reread"] == 0
    assert affine["bytes_eliminated_vs_split_decode"] == rdf.f16_bytes(rdf.mlp_params())
    assert affine["bytes_eliminated_vs_split_decode"] == 34_225_520_640
    assert affine["bytes_eliminated_if_true"] == 0
    q4 = next(c for c in _snap()["candidates"] if c["id"] == "q4_unpack_plus_matvec")
    assert q4["intermediate_bytes_written_and_reread"] == 0
    assert q4["bytes_eliminated_vs_split_decode"] == (
        rdf.f16_bytes(rdf.attention_gemv_params()) + rdf.f16_bytes(rdf.lm_head_params())
    )


def test_ranking_is_by_bytes_not_dispatches():
    snap = _snap()
    ranking = snap["ranking"]
    assert ranking["rank_by"] == "bytes_eliminated_vs_split_decode"
    assert ranking["not_by"] == "dispatches_removed"
    assert ranking["top_legal"] == "affine_q2_unpack_plus_matvec"
    assert ranking["top_status"] == rdf.ALREADY_FUSED
    assert ranking["top_bytes_eliminated_vs_split_decode"] == 34_225_520_640
    assert ranking["cheapest_falsifier_for_top"]
    assert "dense_w_materialized" in ranking["cheapest_falsifier_for_top"]
    assert "unpack_to_dense_f16_then_gemv" in ranking["rejected_dense_remat"]
    assert ranking["order"][0] == "affine_q2_unpack_plus_matvec"
    # Q4 has more sealed launches (130 vs 128) and fewer W bytes. Byte rank
    # must not flip to launch rank.
    assert ranking["launch_order_differs_from_byte_order"] is True
    cands = {c["id"]: c for c in snap["candidates"]}
    assert cands["q4_unpack_plus_matvec"]["sealed_launches"] > cands["affine_q2_unpack_plus_matvec"]["sealed_launches"]
    assert (
        cands["q4_unpack_plus_matvec"]["bytes_eliminated_vs_split_decode"]
        < cands["affine_q2_unpack_plus_matvec"]["bytes_eliminated_vs_split_decode"]
    )
    assert ranking["order"].index("affine_q2_unpack_plus_matvec") < ranking["order"].index(
        "q4_unpack_plus_matvec"
    )


def test_hgravs_mid_is_open_blocked_and_zero_on_this_artifact():
    row = next(c for c in _snap()["candidates"] if c["id"] == "hgravs_two_stage_mid")
    assert row["status"] == rdf.OPEN
    assert row["on_sealed_258"] is False
    assert row["sealed_launches"] == 0
    assert row["bytes_eliminated_if_true"] == 0
    assert row["blocked_today"] is True
    assert "512" in row["metal_blockers"]["threadgroup_capacity"]
    assert str(rdf.HIDDEN) in row["metal_blockers"]["threadgroup_capacity"]
    assert row["bytes_per_hgravs_gemv_write_plus_reread"] == 1280
    packing = _snap()["packing"]
    assert packing["hgravs_gemv_tensors_in_kernel_geometry_census"] == 0


def test_hgravs_x_cap_is_below_this_models_k():
    assert rdf.HGRAVS_X_CAP == 512
    assert rdf.HGRAVS_X_CAP < rdf.HIDDEN
    assert rdf.HGRAVS_X_CAP < rdf.INTERMEDIATE
    assert rdf.HGRAVS_X_CAP < rdf.O_PROJ_COLS


def test_codebook_and_residual_are_not_this_artifact():
    by_id = {c["id"]: c for c in _snap()["candidates"]}
    assert by_id["codebook_lookup_plus_accumulate"]["status"] == rdf.NOT_THIS_ARTIFACT
    assert by_id["sparse_residual_decode_plus_consume"]["status"] == rdf.NOT_THIS_ARTIFACT
    assert by_id["rice_index_expansion_at_upload"]["status"] == rdf.LOAD_TIME_ONLY
    assert by_id["codebook_lookup_plus_accumulate"]["sealed_launches"] == 0
    assert by_id["sparse_residual_decode_plus_consume"]["dense_rematerialization"] == rdf.DIRECT_CONSUME


def test_activation_intermediates_are_not_ranked_as_w():
    act = _snap()["activation_intermediates_not_weight_decode"]
    assert act["are_decoded_weights"] is False
    ids = [r["id"] for r in act["rows"]]
    assert "mlp_act" in ids
    assert "hgravs_mid" in ids
    mlp_act = next(r for r in act["rows"] if r["id"] == "mlp_act")
    assert mlp_act["bytes_per_launch"] == 17_408 * 4
    assert mlp_act["write_plus_reread_per_token"] == 17_408 * 4 * 64 * 2
    ranking = _snap()["ranking"]
    assert ranking["top_legal"] != "mlp_act"
    assert all(c["id"] != "mlp_act" for c in _snap()["candidates"])


def test_helper_markers_fail_closed_when_missing():
    dead = rdf.helper_markers({rdf.AFFINE2_SHADER: "not the affine path"})
    assert dead["ok"] is False
    assert "affine2_never_writes_dense_w" in dead["missing"]
    live = rdf.helper_markers()
    assert live["ok"] is True
    assert not live["missing"]
    assert live["required_present"]["decode_two_stage_unbound"] is True
    assert live["required_present"]["q80_two_stage_kernel"] is True


def test_every_candidate_has_metal_blockers_and_a_falsifier():
    for row in _snap()["candidates"]:
        assert set(row["metal_blockers"]) == set(rdf.METAL_BLOCKER_KEYS)
        assert row["cheapest_falsifier"]
        assert row["mechanism"]
        assert row["what_is_materialized"]
        assert row["not_a_dispatch_count_plan"] is True
        assert row["evidence_class"] == "STATIC_ONLY"
        if row["status"] != rdf.REJECTED_DENSE_REMAT:
            assert row["physical_primitive"] in ATLAS_PRIMITIVES


def test_dispatch_count_refutation_is_cited_not_reopened():
    snap = _snap()
    cited = snap["established"]["dispatch_count_is_not_the_350GBs_cause"]
    assert cited["finding_id"] == "DISPATCH_COUNT_DOES_NOT_EXPLAIN_THE_DIFFERENCE"
    assert snap["established"]["not_a_fuse_to_reduce_the_integer"] is True
    assert "negative" in cited["what"].lower() or "refuted" in cited["what"].lower()


def test_no_hardware_claims_and_receipt_is_static_only():
    out = rdf.build()
    assert out.parent == RECEIPTS
    assert out.name == rdf.RECEIPT
    doc = json.loads(out.read_text())
    assert doc["schema"] == rdf.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    _assert_no_hardware_claims(doc)
    for field in HARDWARE_FIELDS:
        # Nested hardware numbers are refused by _assert_no_hardware_claims.
        assert True
    assert doc["ranking"]["top_legal"] == "affine_q2_unpack_plus_matvec"
    assert doc["reconciliation"]["representation_decode"] == 258


def test_module_entrypoint_writes_receipt():
    rc = rdf.main(["--record"])
    assert rc == 0
    doc = json.loads((RECEIPTS / rdf.RECEIPT).read_text())
    assert doc["schema"] == rdf.SCHEMA
    assert doc["seal_sha256"]


def test_geometry_pins():
    assert rdf.mlp_params() == 17_112_760_320
    assert rdf.lm_head_params() == 248_320 * 5_120
    assert rdf.dense_f16_w_bytes()["mlp"] == 34_225_520_640
    assert rdf.dense_f16_w_bytes()["lm_head"] == 2_542_796_800
    assert rdf.attention_gemv_params() == (
        48 * 16384 * 5120
        + 48 * 96 * 5120
        + 48 * 5120 * 6144
        + 16 * 12288 * 5120
        + 16 * 1024 * 5120
        + 16 * 1024 * 5120
        + 16 * 5120 * 6144
    )


def test_answer_does_not_treat_258_as_a_separate_unpack_pass():
    answer = _snap()["answer"]["why_is_it_separate"].lower()
    assert "in-register" in answer or "are* the consume" in _snap()["answer"]["why_is_it_separate"]
    assert "not because a decode pass" in _snap()["answer"]["why_is_it_separate"]
    assert "REJECTED_DENSE_REMAT" in _snap()["answer"]["what_would_look_like_separate_and_is_forbidden"]

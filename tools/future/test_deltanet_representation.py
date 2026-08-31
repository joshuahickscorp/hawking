"""Tests for the DeltaNet representation census of the 2.96 GB.

A guard nobody has watched fail is not a guard. The load-bearing refusal:
per-tensor DeltaNet bytes that do not sum to 2,961,659,904 raise, they do
not silently pass.
"""
from __future__ import annotations

import json

import pytest

from tools.future import deltanet_representation as dnr
from tools.future._common import RECEIPTS, _assert_no_hardware_claims
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def test_organ_targets_sum_to_deltanet_bar():
    assert (
        dnr.QKVZ_ACTIVE_TARGET
        + dnr.OUT_ACTIVE_TARGET
        + dnr.BA_ACTIVE_TARGET
        + dnr.CONV_ACTIVE_TARGET
        == dnr.DELTANET_ACTIVE_TARGET
    )
    assert dnr.DELTANET_ACTIVE_TARGET == 2_961_659_904


def test_unreconciled_deltanet_raises_and_does_not_silently_pass():
    """NEGATIVE CONTROL: broken per-tensor accounting must refuse."""
    with pytest.raises(dnr.UnreconciledDeltaNet) as caught:
        dnr.reconcile_deltanet(0)
    assert caught.value.got == 0
    assert caught.value.want == dnr.DELTANET_ACTIVE_TARGET
    assert "REFUSED" in str(caught.value)
    assert str(dnr.DELTANET_ACTIVE_TARGET) in str(caught.value)

    with pytest.raises(dnr.UnreconciledDeltaNet):
        dnr.reconcile_deltanet(dnr.DELTANET_ACTIVE_TARGET - 1)

    with pytest.raises(dnr.UnreconciledDeltaNet):
        dnr.reconcile_deltanet(dnr.DELTANET_ACTIVE_TARGET + 1)

    rows, _geo = dnr.census_rows()
    mutated = [dict(r) for r in rows]
    mutated[0]["stored_bytes"] = int(mutated[0]["stored_bytes"]) + 1
    mutated[0]["code_bytes"] = int(mutated[0]["code_bytes"]) + 1
    with pytest.raises(dnr.UnreconciledDeltaNet) as caught2:
        dnr.accounting_from_rows(mutated)
    assert caught2.value.got == dnr.DELTANET_ACTIVE_TARGET + 1
    assert caught2.value.want == dnr.DELTANET_ACTIVE_TARGET


def test_accounting_reconciles_to_recorded_deltanet_total():
    snap = dnr.accounting()
    assert snap["stored_bytes"] == dnr.DELTANET_ACTIVE_TARGET
    assert snap["header_bytes"] + snap["scale_bytes"] + snap["bias_bytes"] + snap["code_bytes"] == (
        dnr.DELTANET_ACTIVE_TARGET
    )
    assert snap["bias_bytes"] == 0
    assert snap["reconciled"] is True
    assert snap["n_tensors"] == 192
    organs = snap["by_organ"]
    assert organs["attention.linear_qkvz"]["stored_bytes"] == dnr.QKVZ_ACTIVE_TARGET
    assert organs["attention.linear_out"]["stored_bytes"] == dnr.OUT_ACTIVE_TARGET
    assert organs["attention.linear_ba"]["stored_bytes"] == dnr.BA_ACTIVE_TARGET
    assert organs["attention.linear_conv1d"]["stored_bytes"] == dnr.CONV_ACTIVE_TARGET
    assert organs["attention.linear_qkvz"]["n_tensors"] == 48
    # Codes dominate; headers are not the 2.96 GB.
    assert snap["code_share"] > 0.9
    assert snap["header_bytes"] < 10_000
    assert snap["header_share"] < 0.001
    # linear_qkvz is bigger than GQA (891,292,160) + lm_head (675,430,440).
    assert dnr.QKVZ_ACTIVE_TARGET > 891_292_160 + 675_430_440
    assert abs(snap["share_of_token"] - 0.2998) < 1e-4


def test_hq30uq4_and_f32v2_are_the_incumbent_representations():
    rows, geo = dnr.census_rows()
    qkvz = [r for r in rows if r["organ"] == "attention.linear_qkvz"]
    out = [r for r in rows if r["organ"] == "attention.linear_out"]
    ba = [r for r in rows if r["organ"] == "attention.linear_ba"]
    conv = [r for r in rows if r["organ"] == "attention.linear_conv1d"]
    assert len(qkvz) == len(out) == len(ba) == len(conv) == 48
    assert {r["representation"] for r in qkvz + out + ba} == {"hq30uq4_uniform_q4"}
    assert {r["bits"] for r in qkvz + out + ba} == {4}
    assert {r["group_size"] for r in qkvz + out + ba} == {64}
    assert {r["header_bytes"] for r in qkvz + out + ba} == {40}
    assert {r["bias_bytes"] for r in qkvz + out + ba} == {0}
    assert {r["representation"] for r in conv} == {"f32v2_le"}
    assert {r["bits"] for r in conv} == {32}
    assert {r["header_bytes"] for r in conv} == {8}
    gate0 = next(r for r in qkvz if r["layer"] == 0)
    assert gate0["shape"] == [geo["qkvz_rows"], geo["qkvz_cols"]] == [16384, 5120]
    assert gate0["stored_bytes"] == 44_564_520
    assert gate0["code_bytes"] + gate0["scale_bytes"] + gate0["header_bytes"] == 44_564_520
    conv0 = next(r for r in conv if r["layer"] == 0)
    assert conv0["shape"] == [geo["conv_channels"], geo["conv_kernel"], 1]
    assert conv0["elements"] == 40_960


def test_qkvz_subblocks_cover_the_fused_tensor_and_reconcile():
    geo = dnr.load_dn_geometry()
    idx = dnr.fused_qkvz_row_indices(geo)
    union = set()
    for name in dnr.SUBBLOCKS:
        rows = idx[name]
        as_set = set(int(x) for x in rows)
        assert len(as_set) == rows.size
        assert union.isdisjoint(as_set)
        union |= as_set
    assert union == set(range(int(geo["qkvz_rows"])))
    assert idx["q"].size == idx["k"].size == 2048
    assert idx["v"].size == idx["z"].size == 6144
    parts = dnr.qkvz_subblock_parts(geo)
    payload = sum(parts[s]["payload_bytes"] for s in dnr.SUBBLOCKS)
    header = parts["header"]["header_bytes"]
    assert payload + header == dnr.QKVZ_ACTIVE_TARGET
    # z is 37.5% of rows — same byte mass as v — and is the only block
    # that cannot corrupt rec_state.
    assert parts["z"]["row_share_of_qkvz"] == 0.375
    assert parts["q"]["row_share_of_qkvz"] == 0.125


def test_per_layer_is_uniform_and_only_linear_attention():
    snap = dnr.accounting()
    layers = snap["per_layer"]
    assert len(layers) == 48
    sizes = {int(r["stored_bytes"]) for r in layers}
    assert sizes == {dnr.DELTANET_ACTIVE_TARGET // 48}
    for layer in layers:
        assert layer["kind"] == "linear_attention"
        organs = set(layer["organs"])
        assert organs == set(dnr.DN_ORGANS)


def test_geometry_state_is_not_added_to_the_2gb_bar():
    snap = dnr.accounting()
    state = snap["geometry_state"]
    assert state["in_catalog"] is False
    assert state["in_2gb_bar"] is False
    rec = state["recurrent_state"]
    assert rec["elements_per_layer"] == 48 * 128 * 128
    assert rec["resident_bytes"] == rec["elements_per_layer"] * 4 * 48
    assert rec["rw_bytes_per_token"] == rec["resident_bytes"] * 2
    assert snap["adjacent_not_in_2gb"]["in_2gb_bar"] is False
    assert snap["adjacent_not_in_2gb"]["stored_bytes"] > 0
    assert snap["stored_bytes"] + snap["adjacent_not_in_2gb"]["stored_bytes"] != snap["stored_bytes"]
    # The 2.96 GB bar is weights only.
    assert snap["stored_bytes"] == dnr.DELTANET_ACTIVE_TARGET


def test_z_does_not_enter_gated_delta_and_equal_precision_is_not_the_default():
    info = dnr.independent_information(dnr.load_dn_geometry())
    assert info["roles"]["z"]["enters_gated_delta"] is False
    assert info["roles"]["z"]["enters_conv"] is False
    assert info["roles"]["z"]["enters_state_update"] is False
    assert info["roles"]["q"]["enters_gated_delta"] is True
    assert info["roles"]["k"]["enters_state_update"] is True
    assert info["roles"]["v"]["enters_state_update"] is True
    assert info["equal_precision_default"]["justified_by_consume"] is False
    assert info["equal_precision_default"]["heterogeneous_allocation_available"] is True
    assert info["equal_precision_default"]["physical_primitive"] in ATLAS_PRIMITIVES


def _snap() -> dict:
    return dnr.snapshot(consult_index=False)


def test_measurements_do_not_claim_equal_sensitivity_from_equal_packing():
    meas = _snap()["measurements"]
    assert meas["n_tensors_measured"] == 48
    rel = {
        s: meas["by_subblock"][s]["requant_relfro_vs_incumbent_q4"]["q3"]
        for s in dnr.SUBBLOCKS
    }
    # Packing is uniform: all four sit near 0.22. That is not a ranking.
    for s, v in rel.items():
        assert 0.15 < v < 0.30, (s, v)
    spread = max(rel.values()) - min(rel.values())
    assert spread < 0.03
    # Cross-layer scales are not a shared transform.
    for s in dnr.SUBBLOCKS:
        mean = meas["cross_layer_per_row_mean_scale_pearson"][s]["mean"]
        assert abs(mean) < 0.05
    # Energy share roughly tracks row share; packing does not pick a winner.
    energy = meas["scale_l2_energy_share_mean"]
    assert abs(energy["q"] - 0.125) < 0.03
    assert abs(energy["z"] - 0.375) < 0.03


def test_candidates_cover_the_contract_questions():
    cands = _snap()["candidates"]
    ids = [c["id"] for c in cands]
    assert ids == list(dnr.REQUIRED_CANDIDATE_IDS)
    allowed_status = {
        dnr.ALREADY_FALSIFIED,
        dnr.MEASURED_NEGATIVE,
        dnr.OPEN,
        dnr.EXISTING_LEVER,
    }
    allowed_remat = {
        dnr.DIRECT_CONSUME,
        dnr.REJECTED_DENSE_REMAT,
        dnr.DEPENDS_ON_LOWERING,
    }
    for row in cands:
        assert row["mechanism"]
        assert row["byte_model"]
        assert row["cheapest_falsifier"]
        assert row["dense_rematerialization"] in allowed_remat
        assert row["status"] in allowed_status
        assert row["evidence_class"] == "STATIC_ONLY"
        assert row["gpu_authority"] is False
        if row["dense_rematerialization"] != dnr.REJECTED_DENSE_REMAT:
            assert row["physical_primitive"] in ATLAS_PRIMITIVES


def test_prize_and_scars_are_honest():
    by_id = {c["id"]: c for c in _snap()["candidates"]}
    het = by_id["heterogeneous_qkvz_bits"]
    assert het["status"] == dnr.OPEN
    assert het["capability"] == "UNMEASURED"
    assert het["measured"]["z_enters_gated_delta"] is False
    assert het["bytes_eliminated_if_true"] == het["bytes_eliminated_breakdown"]["z_codes_to_3bit"]
    assert het["dense_rematerialization"] == dnr.DIRECT_CONSUME
    assert by_id["gravity_family_on_dn_weights"]["status"] == dnr.ALREADY_FALSIFIED
    assert any(
        c.get("scar_id") == "NNS-019"
        for c in by_id["gravity_family_on_dn_weights"]["citations"]
    )
    assert by_id["share_or_merge_state_across_depth"]["status"] == dnr.ALREADY_FALSIFIED
    assert any(
        c.get("scar_id") == "QN-STATE-MERGING"
        for c in by_id["share_or_merge_state_across_depth"]["citations"]
    )
    assert by_id["shared_transforms_across_layers"]["status"] == dnr.MEASURED_NEGATIVE
    assert by_id["fused_update_consume"]["status"] == dnr.EXISTING_LEVER
    assert by_id["fused_update_consume"]["bytes_eliminated_if_true"] == 0
    assert by_id["generated_coefficients"]["dense_rematerialization"] == dnr.REJECTED_DENSE_REMAT


def test_no_candidate_is_silent_dense_w_remat():
    cands = _snap()["candidates"]
    remat = {c["id"] for c in cands if c["dense_rematerialization"] == dnr.REJECTED_DENSE_REMAT}
    assert remat == {"generated_coefficients"}
    gen = next(c for c in cands if c["id"] == "generated_coefficients")
    assert "dense" in gen["dense_rematerialization_reason"].lower()


def test_answers_section_does_not_assume_equal_precision():
    ans = _snap()["answers"]
    equal = ans["do_qkvz_subblocks_deserve_equal_precision"]
    assert equal["answer"].startswith("NO")
    assert equal["available"] is True
    assert equal["z_enters_gated_delta"] is False
    assert ans["what_are_the_bytes"]["stored_bytes"] == dnr.DELTANET_ACTIVE_TARGET
    assert ans["what_are_the_bytes"]["bias_bytes"] == 0
    assert ans["what_does_the_recurrent_state_need"]["share_or_merge_status"] == (
        dnr.ALREADY_FALSIFIED
    )
    assert ans["is_there_an_existing_fuse_of_update_and_consume"]["status"] == (
        dnr.EXISTING_LEVER
    )


def test_negative_index_is_queried_and_does_not_launder_mlp_scars():
    snap = dnr.snapshot(consult_index=True)
    by_id = {c["id"]: c for c in snap["candidates"]}
    shared = by_id["shared_transforms_across_layers"]
    assert shared["status"] == dnr.MEASURED_NEGATIVE
    assert shared.get("cousin_not_this_object") is True
    merge = by_id["share_or_merge_state_across_depth"]
    assert merge["status"] == dnr.ALREADY_FALSIFIED
    # Index may attach QN-STATE-MERGING; it must not flip an OPEN prize.
    het = by_id["heterogeneous_qkvz_bits"]
    assert het["status"] == dnr.OPEN


def test_build_emits_sealed_receipt():
    out = dnr.build(consult_index=True)
    assert out.parent == RECEIPTS
    assert out.name == "DELTANET_REPRESENTATION.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == dnr.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    _assert_no_hardware_claims(doc)
    assert doc["accounting"]["stored_bytes"] == dnr.DELTANET_ACTIVE_TARGET
    assert doc["accounting"]["reconciled"] is True
    assert [c["id"] for c in doc["candidates"]] == list(dnr.REQUIRED_CANDIDATE_IDS)
    assert doc["candidate_counts"]["n"] == len(dnr.REQUIRED_CANDIDATE_IDS)
    assert any(c["id"] == "heterogeneous_qkvz_bits" for c in doc["candidates"])


def test_module_entrypoint_runs_and_emits_sealed_receipt():
    rc = dnr.main(["--build"])
    assert rc == 0
    doc = json.loads((RECEIPTS / dnr.RECEIPT).read_text())
    assert doc["schema"] == dnr.SCHEMA
    assert doc["seal_sha256"]
    assert doc["accounting"]["stored_bytes"] == dnr.DELTANET_ACTIVE_TARGET


def test_selftest_aliases_build():
    assert dnr.selftest is dnr.build or dnr.selftest().name == dnr.RECEIPT


def test_q4_parts_match_a_real_tensor():
    parts = dnr.q4_parts(16384, 5120)
    assert parts["stored_bytes"] == 44_564_520
    assert parts["header_bytes"] == 40
    assert parts["scale_bytes"] == 2_621_440
    assert parts["code_bytes"] == 41_943_040
    assert parts["bias_bytes"] == 0
    conv = dnr.f32v2_parts(40_960)
    assert conv["stored_bytes"] == 163_848

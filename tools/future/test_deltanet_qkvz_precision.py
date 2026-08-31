"""Tests for heterogeneous precision inside DeltaNet linear_qkvz.

A guard nobody has watched fail is not a guard. Two load-bearing refusals:

1. q/k/v/z byte accounting that does not reassemble to 2,139,096,960 raises.
2. A bit reduction is never reported as supported on entropy alone.
"""
from __future__ import annotations

import json

import pytest

from tools.future import deltanet_qkvz_precision as dqp
from tools.future import deltanet_representation as dnr
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def test_unreconciled_qkvz_raises_and_does_not_silently_pass():
    """NEGATIVE CONTROL: broken qkvz accounting must refuse."""
    with pytest.raises(dqp.UnreconciledQkvz) as caught:
        dqp.reconcile_qkvz(0)
    assert caught.value.got == 0
    assert caught.value.want == 2_139_096_960
    assert caught.value.want == dqp.QKVZ_ACTIVE_TARGET
    assert "REFUSED" in str(caught.value)
    assert str(2_139_096_960) in str(caught.value)

    with pytest.raises(dqp.UnreconciledQkvz):
        dqp.reconcile_qkvz(dqp.QKVZ_ACTIVE_TARGET - 1)

    with pytest.raises(dqp.UnreconciledQkvz):
        dqp.reconcile_qkvz(dqp.QKVZ_ACTIVE_TARGET + 1)

    with pytest.raises(dqp.UnreconciledQkvz) as caught2:
        dqp.qkvz_subblock_payload_sum(
            {
                "q": {"payload_bytes": 1},
                "k": {"payload_bytes": 1},
                "v": {"payload_bytes": 1},
                "z": {"payload_bytes": 1},
                "header": {"header_bytes": 0},
            }
        )
    assert caught2.value.got == 4
    assert caught2.value.want == dqp.QKVZ_ACTIVE_TARGET


def test_bit_reduction_is_never_supported_on_entropy_alone():
    """A low H(q) does not license a drop without a sensitivity measurement."""
    low = dqp.decide_supported_bit_reduction(
        candidate_bits=3,
        incumbent_bits=4,
        H_q_bits=2.0,
        sensitivity=None,
    )
    assert low["supported"] is False
    assert low["reason"] == dqp.ENTROPY_ALONE_INSUFFICIENT
    assert low["sensitivity_measured"] is False
    assert low["lossless_possible"] is True

    high = dqp.decide_supported_bit_reduction(
        candidate_bits=3,
        H_q_bits=3.5,
        sensitivity=None,
    )
    assert high["supported"] is False
    assert high["reason"] == dqp.ENTROPY_ALONE_INSUFFICIENT
    assert high["lossless_possible"] is False

    missing_flag = dqp.decide_supported_bit_reduction(
        candidate_bits=3,
        H_q_bits=2.0,
        sensitivity={"gated_cosine_min": 0.999, "writes_rec_state": False},
    )
    assert missing_flag["supported"] is False
    assert missing_flag["reason"] == dqp.ENTROPY_ALONE_INSUFFICIENT
    assert missing_flag["sensitivity_measured"] is False

    measured_false = dqp.decide_supported_bit_reduction(
        candidate_bits=3,
        H_q_bits=2.0,
        sensitivity={"measured": False, "gated_cosine_min": 0.999},
    )
    assert measured_false["supported"] is False
    assert measured_false["reason"] == dqp.ENTROPY_ALONE_INSUFFICIENT


def test_sensitivity_can_support_only_when_measured_and_clears_bars():
    ok = dqp.decide_supported_bit_reduction(
        candidate_bits=3,
        H_q_bits=3.5,
        sensitivity={
            "measured": True,
            "gated_cosine_min": 0.995,
            "writes_rec_state": False,
            "rec_state_identical": True,
            "rec_state_relfro_max": 0.0,
        },
    )
    assert ok["supported"] is True
    assert ok["reason"] == dqp.SENSITIVITY_CLEARS_BAR
    assert ok["sensitivity_measured"] is True
    assert ok["lossless_possible"] is False

    injured = dqp.decide_supported_bit_reduction(
        candidate_bits=3,
        H_q_bits=2.0,
        sensitivity={
            "measured": True,
            "gated_cosine_min": 0.95,
            "writes_rec_state": False,
            "rec_state_identical": True,
            "rec_state_relfro_max": 0.0,
        },
    )
    assert injured["supported"] is False
    assert injured["reason"] == dqp.GATED_COSINE_BELOW_BAR
    assert injured["sensitivity_measured"] is True

    writes = dqp.decide_supported_bit_reduction(
        candidate_bits=3,
        H_q_bits=2.0,
        sensitivity={
            "measured": True,
            "gated_cosine_min": 0.999,
            "writes_rec_state": True,
            "rec_state_identical": False,
            "rec_state_relfro_max": 0.2,
        },
    )
    assert writes["supported"] is False
    assert writes["reason"] == dqp.REC_STATE_INJURED


def test_subblock_payloads_reconcile_to_qkvz_organ_total():
    geo = dnr.load_dn_geometry()
    parts = dnr.qkvz_subblock_parts(geo)
    got = dqp.qkvz_subblock_payload_sum(parts)
    assert got == 2_139_096_960
    assert parts["q"]["payload_bytes"] + parts["k"]["payload_bytes"] == parts["v"]["payload_bytes"] / 3 * 2 or (
        parts["q"]["payload_bytes"] * 3 == parts["v"]["payload_bytes"]
    )
    assert parts["q"]["payload_bytes"] == parts["k"]["payload_bytes"]
    assert parts["v"]["payload_bytes"] == parts["z"]["payload_bytes"]
    assert parts["z"]["row_share_of_qkvz"] == 0.375
    assert parts["q"]["row_share_of_qkvz"] == 0.125


def test_accounting_reconciles_to_recorded_qkvz_total():
    snap = dqp.accounting()
    assert snap["stored_bytes"] == 2_139_096_960
    assert snap["reconciled"] is True
    assert snap["n_tensors"] == 48
    assert snap["bias_bytes"] == 0
    assert snap["header_bytes"] + snap["scale_bytes"] + snap["bias_bytes"] + snap["code_bytes"] == (
        snap["stored_bytes"]
    )
    assert snap["incumbent_bits"] == 4
    dqp.qkvz_subblock_payload_sum(snap["qkvz_subblocks"])
    mutated = dict(snap["qkvz_subblocks"])
    q = dict(mutated["q"])
    q["payload_bytes"] = int(q["payload_bytes"]) + 1
    mutated["q"] = q
    with pytest.raises(dqp.UnreconciledQkvz) as caught:
        dqp.qkvz_subblock_payload_sum(mutated)
    assert caught.value.got == 2_139_096_960 + 1


def _snap() -> dict:
    return dqp.snapshot(consult_index=False)


def test_entropy_method_matches_mlp_code_information_and_does_not_pick_a_winner():
    ent = _snap()["entropy"]
    assert ent["n_tensors_measured"] == 48
    assert ent["code_bytes_read"] == _snap()["accounting"]["code_bytes"]
    assert 3.40 < ent["H_q_bits"] < 3.60
    assert ent["lossless_q3_impossible"] is True
    assert ent["mi_within_byte_bits"] < 0.05
    assert abs(ent["H_q_given_prev_within_byte"] - ent["H_q_bits"]) < 0.05
    assert abs(ent["H_byte_bits"] - ent["H_byte_if_iid_q"]) < 0.05
    # Missing nibble 0 is histogram bias, not a 3-bit island.
    assert ent["q_hist"][0] == 0
    for s in dqp.SUBBLOCKS:
        block = ent["by_subblock"][s]
        assert 3.40 < block["H_q_bits"] < 3.60
        assert block["lossless_q3_impossible"] is True
        assert block["nibble_0_unused"] is True
        assert block["mi_within_byte_bits"] < 0.05
        mi = block["cross_layer"]["mi_bits"]["mean"]
        assert mi < 0.01
        match = block["cross_layer"]["match"]["mean"]
        indep = block["cross_layer"]["independent_match"]["mean"]
        assert abs(match - indep) < 0.02
    spread = max(ent["by_subblock"][s]["H_q_bits"] for s in dqp.SUBBLOCKS) - min(
        ent["by_subblock"][s]["H_q_bits"] for s in dqp.SUBBLOCKS
    )
    assert spread < 0.05
    assert ent["sample_unique_frac"]["min"] > 0.99


def test_sensitivity_distinguishes_consume_roles_and_does_not_license_a_drop():
    snap = _snap()
    sens = snap["sensitivity"]
    assert sens["measured"] is True
    assert sens["gpu_authority"] is False
    assert sens["probe"]["generate_gate"] is False
    z = sens["by_subblock"]["z"]
    q = sens["by_subblock"]["q"]
    k = sens["by_subblock"]["k"]
    v = sens["by_subblock"]["v"]
    assert z["rec_state_identical"] is True
    assert z["rec_out_h_identical"] is True
    assert z["writes_rec_state"] is False
    assert q["rec_state_identical"] is True
    assert q["writes_rec_state"] is False
    assert q["rec_out_h_identical"] is False
    assert k["writes_rec_state"] is True
    assert v["writes_rec_state"] is True
    assert k["rec_state_identical"] is False
    assert v["rec_state_identical"] is False
    assert k["rec_state_relfro_max"] > dqp.STATE_RELFRO_BAR
    assert v["rec_state_relfro_max"] > dqp.STATE_RELFRO_BAR
    for s in dqp.SUBBLOCKS:
        assert sens["by_subblock"][s]["gated_cosine_min"] < dqp.COSINE_BAR


def test_allocation_keeps_four_bits_and_eliminates_zero_bytes():
    alloc = _snap()["allocation"]
    assert alloc["total_bytes_eliminated"] == 0
    assert alloc["any_supported"] is False
    assert alloc["cheapest_relative_win"] == "z"
    assert set(alloc["not_worth_touching"]) == set(dqp.SUBBLOCKS)
    for s in dqp.SUBBLOCKS:
        row = alloc["by_subblock"][s]
        assert row["bits"] == 4
        assert row["supported"] is False
        assert row["bytes_eliminated"] == 0
        assert row["sensitivity_measured"] is True
        assert row["reason"] != dqp.ENTROPY_ALONE_INSUFFICIENT
        assert row["consume"]["physical_primitive"] in ATLAS_PRIMITIVES
        if row["supported"] and not row["sensitivity_measured"]:
            raise AssertionError("supported drop without sensitivity")
    assert alloc["qkvz_bytes_after"] == 2_139_096_960
    assert alloc["token_bytes_after"] == dqp.TOKEN_ACTIVE_TARGET
    assert alloc["share_of_token_eliminated"] == 0.0


def test_allocation_refuses_to_mark_supported_when_sensitivity_is_omitted():
    acc, ent, _sens, info = dqp._measured()
    alloc = dqp.allocation_from_measurements(acc, ent, None, info)
    assert alloc["any_supported"] is False
    assert alloc["total_bytes_eliminated"] == 0
    for s in dqp.SUBBLOCKS:
        row = alloc["by_subblock"][s]
        assert row["supported"] is False
        assert row["reason"] == dqp.ENTROPY_ALONE_INSUFFICIENT
        assert row["sensitivity_measured"] is False
        assert row["bits"] == 4


def test_z_is_cheapest_relative_and_k_v_q_are_not_worth_touching():
    ans = _snap()["answers"]
    cheap = ans["which_subblock_is_the_cheapest_real_win"]
    assert cheap["cheapest_relative"] == "z"
    assert cheap["licensed_win"] is None
    assert set(ans["which_is_not_worth_touching"]["not_worth_touching"]) == set(dqp.SUBBLOCKS)
    bits = ans["what_is_the_heterogeneous_allocation"]
    assert bits["total_bytes_eliminated"] == 0
    assert bits["bits"] == {"q": 4, "k": 4, "v": 4, "z": 4}
    assert ans["does_sensitivity_license_heterogeneous_bits"]["status"] == dqp.MEASURED_NEGATIVE
    assert ans["do_the_subblocks_have_different_code_entropy"]["status"] == dqp.MEASURED_NEGATIVE


def test_candidates_cover_the_contract_and_name_a_physical_primitive():
    cands = _snap()["candidates"]
    ids = [c["id"] for c in cands]
    assert ids == list(dqp.REQUIRED_CANDIDATE_IDS)
    legal_status = {
        dqp.ALREADY_FALSIFIED,
        dqp.MEASURED_NEGATIVE,
        dqp.OPEN,
        dqp.UNMEASURED,
    }
    legal_remat = {
        dqp.DIRECT_CONSUME,
        dqp.REJECTED_DENSE_REMAT,
        dqp.DEPENDS_ON_LOWERING,
    }
    for row in cands:
        assert row["mechanism"]
        assert row["byte_model"]
        assert row["cheapest_falsifier"]
        assert row["dense_rematerialization"] in legal_remat
        assert row["status"] in legal_status
        assert row["evidence_class"] == "STATIC_ONLY"
        assert row["gpu_authority"] is False
        assert row["physical_primitive"] in ATLAS_PRIMITIVES
    by_id = {c["id"]: c for c in cands}
    assert by_id["heterogeneous_qkvz_bits"]["status"] == dqp.MEASURED_NEGATIVE
    assert by_id["heterogeneous_qkvz_bits"]["bytes_eliminated_if_true"] == 0
    assert by_id["z_only_q3"]["status"] == dqp.MEASURED_NEGATIVE
    assert by_id["q_readout_q3"]["status"] == dqp.MEASURED_NEGATIVE
    assert by_id["k_state_q3"]["status"] == dqp.MEASURED_NEGATIVE
    assert by_id["v_state_q3"]["status"] == dqp.MEASURED_NEGATIVE
    assert by_id["uniform_q3_qkvz"]["status"] == dqp.MEASURED_NEGATIVE
    assert by_id["entropy_coded_qkvz_codes"]["status"] == dqp.OPEN
    assert by_id["entropy_coded_qkvz_codes"]["support"] == "SHANNON_GAP_MEASURED_KERNEL_UNMEASURED"
    assert by_id["heterogeneous_qkvz_bits"]["dense_rematerialization"] == dqp.DIRECT_CONSUME
    assert "NNS-019" in {c["scar_id"] for c in by_id["heterogeneous_qkvz_bits"]["citations"]}
    assert "NNS-029" in {c["scar_id"] for c in by_id["heterogeneous_qkvz_bits"]["citations"]}
    assert "NNS-022" in {c["scar_id"] for c in by_id["entropy_coded_qkvz_codes"]["citations"]}


def test_shader_binding_z_does_not_enter_gated_delta():
    info = _snap()["independent_information"]
    assert info["roles"]["z"]["enters_gated_delta"] is False
    assert info["roles"]["z"]["enters_conv"] is False
    assert info["roles"]["z"]["enters_state_update"] is False
    assert info["roles"]["q"]["enters_gated_delta"] is True
    assert info["roles"]["k"]["enters_state_update"] is True
    assert info["roles"]["v"]["enters_state_update"] is True
    assert info["equal_precision_default"]["justified_by_consume"] is False


def test_build_emits_sealed_receipt():
    out = dqp.build(consult_index=True)
    assert out.parent == RECEIPTS
    assert out.name == "DELTANET_QKVZ_PRECISION.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == dqp.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    _assert_no_hardware_claims(doc)
    assert doc["accounting"]["stored_bytes"] == 2_139_096_960
    assert doc["accounting"]["reconciled"] is True
    assert doc["allocation"]["total_bytes_eliminated"] == 0
    assert [c["id"] for c in doc["candidates"]] == list(dqp.REQUIRED_CANDIDATE_IDS)
    assert doc["candidate_counts"]["n"] == len(dqp.REQUIRED_CANDIDATE_IDS)
    assert doc["allocation"]["by_subblock"]["z"]["sensitivity_measured"] is True
    for s in dqp.SUBBLOCKS:
        assert doc["allocation"]["by_subblock"][s]["supported"] is False


def test_module_entrypoint_runs_and_emits_sealed_receipt():
    rc = dqp.main(["--build"])
    assert rc == 0
    doc = json.loads((RECEIPTS / dqp.RECEIPT).read_text())
    assert doc["schema"] == dqp.SCHEMA
    assert doc["seal_sha256"]
    assert doc["accounting"]["stored_bytes"] == 2_139_096_960


def test_selftest_aliases_build():
    assert dqp.selftest is dqp.build or dqp.selftest().name == dqp.RECEIPT


def test_hardware_fields_stay_non_numeric_on_the_receipt():
    out = dqp.build(consult_index=False)
    doc = json.loads(out.read_text())
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:

        def walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    here = f"{path}.{k}" if path else k
                    if k in HARDWARE_FIELDS:
                        assert not isinstance(v, (int, float)) or isinstance(v, bool), here
                    walk(v, here)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")

        walk(doc)


def test_q3_code_byte_model_is_exact():
    z_code = 754_974_720
    assert dqp.bytes_eliminated_codes(z_code, 3) == z_code // 4
    q_code = 251_658_240
    assert dqp.bytes_eliminated_codes(q_code, 3) == q_code // 4
    assert dqp.q4_code_bytes_at_bits(z_code, 4) == z_code
    assert dqp.q4_code_bytes_at_bits(z_code, 3) + dqp.bytes_eliminated_codes(z_code, 3) == z_code

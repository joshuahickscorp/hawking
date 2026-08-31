"""Tests for the DeltaNet recurrent-state function census.

A guard nobody has watched fail is not a guard. Two load-bearing refusals:

1. Geometry-derived S that does not equal 150,994,944 raises.
2. A candidate without BOTH bytes_removed and bytes_added is refused —
   a compression ratio with no executable economics is not a candidate.
"""
from __future__ import annotations

import json

import pytest

from tools.future import deltanet_state_function as dsf
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def test_unreconciled_state_raises_and_does_not_silently_pass():
    """NEGATIVE CONTROL: broken S accounting must refuse."""
    with pytest.raises(dsf.UnreconciledState) as caught:
        dsf.reconcile_state(0)
    assert caught.value.got == 0
    assert caught.value.want == 150_994_944
    assert caught.value.want == dsf.REC_STATE_RESIDENT
    assert "REFUSED" in str(caught.value)
    assert str(150_994_944) in str(caught.value)

    with pytest.raises(dsf.UnreconciledState):
        dsf.reconcile_state(dsf.REC_STATE_RESIDENT - 1)

    with pytest.raises(dsf.UnreconciledState):
        dsf.reconcile_state(dsf.REC_STATE_RESIDENT + 1)

    assert dsf.reconcile_state(dsf.REC_STATE_RESIDENT) == 150_994_944


def test_candidate_without_bytes_removed_and_added_is_refused():
    """A compression ratio with no executable economics is not a candidate."""
    with pytest.raises(dsf.MissingEconomics) as caught:
        dsf.require_economics({"id": "ratio_only", "compression_ratio": 4.0})
    assert caught.value.cand_id == "ratio_only"
    assert "bytes_removed" in caught.value.missing
    assert "bytes_added" in caught.value.missing
    assert "compression ratio is not a candidate" in str(caught.value)
    assert "REFUSED" in str(caught.value)

    with pytest.raises(dsf.MissingEconomics) as caught2:
        dsf.require_economics({"id": "removed_only", "bytes_removed": 100})
    assert caught2.value.missing == ("bytes_added",)

    with pytest.raises(dsf.MissingEconomics) as caught3:
        dsf.require_economics({"id": "added_only", "bytes_added": 10})
    assert caught3.value.missing == ("bytes_removed",)

    with pytest.raises(dsf.MissingEconomics):
        dsf.require_economics({"id": "eliminated_only", "bytes_eliminated_if_true": 1_000})

    with pytest.raises(dsf.MissingEconomics):
        dsf.require_economics(None)

    # Both present as ints is accepted.
    assert dsf.require_economics({"id": "ok", "bytes_removed": 100, "bytes_added": 10}) == (100, 10)

    # Both present as breakdowns with total is accepted.
    got = dsf.require_economics(
        {
            "id": "ok_dict",
            "bytes_removed": dsf.removed(catalog_weights=50, state=50),
            "bytes_added": dsf.added(generator=7, metadata=3),
        }
    )
    assert got == (100, 10)

    # finalize_candidates refuses a row that only has a ratio.
    ratio_row = {
        "id": "ratio_only",
        "name": "x",
        "mechanism": "x",
        "byte_model": "4x",
        "compression_ratio": 4.0,
        "extra_flops": {"per_token": 0, "formula": "0", "relative_to": "x"},
        "dispatch_change": {"incumbent": 1, "candidate": 1, "delta": 0, "surface": "x"},
        "physical_primitive": "LocalStateMachine",
        "cheapest_falsifier": "x",
        "dense_rematerialization": dsf.DIRECT_CONSUME,
        "status": dsf.OPEN,
        "evidence_class": "STATIC_ONLY",
    }
    with pytest.raises(dsf.MissingEconomics) as caught4:
        dsf.finalize_candidates([ratio_row])
    assert "bytes_removed" in str(caught4.value)


def test_breakdown_parts_must_sum_to_total():
    with pytest.raises(dsf.StateFunctionRefuse):
        dsf.require_economics(
            {
                "id": "bad_sum",
                "bytes_removed": {
                    "catalog_weights": 1,
                    "state": 1,
                    "other": 1,
                    "total": 99,
                },
                "bytes_added": 0,
            }
        )


def test_rec_state_geometry_reconciles_to_recorded_total():
    assert dsf.rec_state_resident_bytes() == 150_994_944
    assert 48 * 48 * 128 * 128 * 4 == 150_994_944
    geo = dsf.geometry()
    conv = dsf.conv_state_resident_bytes(conv_channels=int(geo["conv_channels"]))
    assert conv == dsf.CONV_STATE_RESIDENT
    inc = dsf.incumbent_bytes()
    assert inc["reconciled"] is True
    assert inc["catalog_weights"]["total"] == dsf.DELTANET_ACTIVE_TARGET
    assert inc["catalog_weights"]["attention.linear_qkvz"] == dsf.QKVZ_ACTIVE_TARGET
    assert inc["state"]["recurrent_resident"] == 150_994_944
    assert inc["state"]["in_catalog"] is False
    assert inc["state"]["in_2gb_bar"] is False
    assert abs(inc["catalog_weights"]["share_of_token"] - 0.2998) < 1e-4


def test_incumbent_operator_is_rank1_plus_decay_and_z_is_out():
    op = dsf.incumbent_operator()
    assert op["parameterisation_not_requirement"] is True
    assert op["update_rank_per_token"] == 1
    assert op["state_rank_capacity_per_head"] == 128
    assert op["every_element_rw_every_token"] is True
    assert op["z_enters_gated_delta"] is False
    assert op["roles"]["z"]["enters_state_update"] is False
    assert op["roles"]["k"]["enters_state_update"] is True
    assert op["roles"]["v"]["enters_state_update"] is True
    assert op["roles"]["q"]["enters_state_update"] is False
    assert op["physical_primitive"] in ATLAS_PRIMITIVES
    assert "judged_on" in op
    assert "logits" in op["judged_on"]


def test_gated_delta_flops_are_exact():
    per = dsf.gated_delta_flops_per_token()
    # 8*kd*vd + 2*vd per head, 48 heads, 48 layers.
    per_head = 8 * 128 * 128 + 2 * 128
    assert per_head == 131_328
    assert per == 48 * 48 * per_head
    flops = dsf.incumbent_flops()
    assert flops["gated_delta_per_token"] == per
    assert flops["qkvz_gemv_per_token"] == 2 * 16384 * 5120 * 48
    assert flops["qkvz_gemv_per_token"] > flops["gated_delta_per_token"]


def test_q4_one_layer_qkvz_matches_recorded_tensor():
    assert dsf.q4_stored(16384, 5120, n_layers=1) == 44_564_520
    assert dsf.q4_stored(16384, 5120, n_layers=48) == dsf.QKVZ_ACTIVE_TARGET
    assert dsf.f32v2_stored(40_960, n_layers=1) == 163_848


def _snap() -> dict:
    return dsf.snapshot(consult_index=False)


def test_candidates_cover_the_contract_questions_and_carry_economics():
    cands = _snap()["candidates"]
    ids = [c["id"] for c in cands]
    assert ids == list(dsf.REQUIRED_CANDIDATE_IDS)
    legal_status = {
        dsf.ALREADY_FALSIFIED,
        dsf.MEASURED_NEGATIVE,
        dsf.OPEN,
        dsf.EXISTING_LEVER,
        dsf.UNMEASURED,
    }
    legal_remat = {
        dsf.DIRECT_CONSUME,
        dsf.REJECTED_DENSE_REMAT,
        dsf.DEPENDS_ON_LOWERING,
    }
    for row in cands:
        for field in dsf.REQUIRED_CANDIDATE_FIELDS:
            assert field in row, field
        removed, added = dsf.require_economics(row)
        assert removed == row["bytes_removed"]["total"]
        assert added == row["bytes_added"]["total"]
        assert row["net_bytes"] == removed - added
        assert "compression_ratio" not in row
        assert row["extra_flops"]["per_token"] is not None
        assert isinstance(row["extra_flops"]["per_token"], int)
        assert "incumbent" in row["dispatch_change"]
        assert "candidate" in row["dispatch_change"]
        assert "delta" in row["dispatch_change"]
        assert row["dense_rematerialization"] in legal_remat
        assert row["status"] in legal_status
        assert row["evidence_class"] == "STATIC_ONLY"
        assert row["gpu_authority"] is False
        if row["dense_rematerialization"] != dsf.REJECTED_DENSE_REMAT:
            assert row["physical_primitive"] in ATLAS_PRIMITIVES
        for part in dsf.REMOVED_PARTS:
            assert part in row["bytes_removed"]
        for part in dsf.ADDED_PARTS:
            assert part in row["bytes_added"]


def test_already_falsified_are_cited_not_rederived():
    by_id = {c["id"]: c for c in _snap()["candidates"]}
    merge = by_id["share_or_merge_state_across_depth"]
    assert merge["status"] == dsf.ALREADY_FALSIFIED
    assert any(c.get("scar_id") == "QN-STATE-MERGING" for c in merge["citations"])
    emit = by_id["emit_w_then_ordinary_gemv"]
    assert emit["status"] == dsf.ALREADY_FALSIFIED
    assert emit["dense_rematerialization"] == dsf.REJECTED_DENSE_REMAT
    assert emit["net_bytes"] == 0
    bits = by_id["qkvz_bit_descent"]
    assert bits["status"] == dsf.MEASURED_NEGATIVE
    assert bits["licensed"] is False
    assert any(c.get("scar_id") == "NNS-019" for c in bits["citations"])
    assert any(c.get("scar_id") == "NNS-029" for c in bits["citations"])


def test_fused_update_consume_is_existing_lever_with_zero_catalog_bytes():
    fused = next(c for c in _snap()["candidates"] if c["id"] == "fused_update_consume")
    assert fused["status"] == dsf.EXISTING_LEVER
    assert fused["bytes_removed"]["total"] == 0
    assert fused["bytes_added"]["total"] == 0
    assert fused["extra_flops"]["per_token"] == 0
    assert fused["dispatch_change"]["incumbent"] == 628
    assert fused["dispatch_change"]["candidate"] == 580
    assert fused["dispatch_change"]["delta"] == -48
    assert fused["physical_primitive"] in ATLAS_PRIMITIVES
    scar_ids = {c.get("scar_id") for c in fused["citations"]}
    assert "BA_DELTA_AB" in scar_ids
    assert "dn_layer_state_machine" in scar_ids


def test_scale_field_sharing_is_measured_negative_and_w_is_unmeasured():
    shared = next(c for c in _snap()["candidates"] if c["id"] == "shared_transforms_w_unmeasured")
    assert shared["status"] == dsf.OPEN
    assert shared["measured"]["scale_field_sharing"] == dsf.MEASURED_NEGATIVE
    assert shared["measured"]["W_sharing"] == dsf.UNMEASURED
    assert shared["bytes_removed"]["total"] == dsf.QKVZ_ACTIVE_TARGET
    assert shared["bytes_added"]["total"] > 0
    assert shared["bytes_added"]["total"] < shared["bytes_removed"]["total"]
    assert shared["net_bytes"] > 0


def test_smaller_state_is_not_a_closed_rank_r_store():
    spec = _snap()["spectrum"]
    assert spec["measured"] is True
    assert spec["trained_function_rank"] == dsf.UNMEASURED
    assert spec["n_tokens"] == 256
    e16 = spec["energy_at_rank"]["16"]["mean"]
    assert 0.0 < e16 <= 1.0
    smaller = next(c for c in _snap()["candidates"] if c["id"] == "smaller_state_machine")
    assert smaller["status"] == dsf.OPEN
    assert smaller["licensed"] is False
    assert smaller["bytes_removed"]["state"] == dsf.REC_STATE_RESIDENT + dsf.CONV_STATE_RESIDENT
    assert smaller["bytes_added"]["state"] < smaller["bytes_removed"]["state"]
    # Rank-r of the present S is not closed; d=64 is the closed named point.
    assert "not closed" in smaller["name"].lower() or "not closed" in smaller["mechanism"].lower()


def test_learned_recurrence_drops_qkv_and_adds_a_generator():
    learned = next(c for c in _snap()["candidates"] if c["id"] == "learned_recurrence")
    assert learned["bytes_removed"]["catalog_weights"] > 0
    assert learned["bytes_added"]["generator"] > 0
    assert learned["bytes_added"]["state"] > 0
    assert learned["bytes_added"]["total"] < learned["bytes_removed"]["total"]
    assert learned["dense_rematerialization"] == dsf.DIRECT_CONSUME
    assert any(c.get("scar_id") == "NNS-015" for c in learned["citations"])


def test_conditional_recurrence_is_zero_catalog_and_does_not_retry_merge():
    cond = next(c for c in _snap()["candidates"] if c["id"] == "conditional_recurrence")
    assert cond["bytes_removed"]["total"] == 0
    assert cond["bytes_added"]["total"] == 0
    assert cond["rw_bytes"]["skip_fraction"] == dsf.UNMEASURED
    assert cond["physical_primitive"] == "ConditionalPhysicalProgram"
    assert any(c.get("scar_id") == "QN-STATE-MERGING" for c in cond["citations"])


def test_answers_do_not_treat_the_parameterisation_as_a_requirement():
    ans = _snap()["answers"]
    need = ans["what_update_does_the_model_need"]
    assert need["parameterisation_not_requirement"] is True
    assert need["z_enters_gated_delta"] is False
    assert ans["is_fused_update_consume_already_one_op"]["status"] == dsf.EXISTING_LEVER
    assert ans["is_fused_update_consume_already_one_op"]["catalog_bytes_eliminated"] == 0
    assert ans["does_every_token_need_the_full_update"]["layer_merge_status"] == (
        dsf.ALREADY_FALSIFIED
    )
    assert ans["can_W_be_shared_across_layers"]["scale_field"] == dsf.MEASURED_NEGATIVE
    assert ans["can_W_be_shared_across_layers"]["W"] == dsf.UNMEASURED
    assert ans["what_is_already_falsified"]["emit_w"] == dsf.ALREADY_FALSIFIED
    assert ans["what_is_already_falsified"]["qkvz_bits"] == dsf.MEASURED_NEGATIVE


def test_no_candidate_is_silent_dense_w_remat_except_the_rejected_row():
    cands = _snap()["candidates"]
    remat = {c["id"] for c in cands if c["dense_rematerialization"] == dsf.REJECTED_DENSE_REMAT}
    assert remat == {"emit_w_then_ordinary_gemv"}


def test_negative_index_is_queried_with_and_without_model():
    snap = dsf.snapshot(consult_index=True)
    by_id = {c["id"]: c for c in snap["candidates"]}
    merge = by_id["share_or_merge_state_across_depth"]
    assert merge["status"] == dsf.ALREADY_FALSIFIED
    for row in snap["candidates"]:
        assert row["index_query_modes"] == ["with_model", "without_model"]
    # GENERAL_PHYSICAL scars refuse regardless of the model named.
    from tools.future.negative_index import refuse_if_dead

    gp = refuse_if_dead(
        {
            "model": "qwen3.8-27b",
            "hypothesis_family": "prefill_over_generated_token_denominator",
        }
    )
    gp_no_model = refuse_if_dead(
        {"hypothesis_family": "prefill_over_generated_token_denominator"}
    )
    assert gp and gp.get("refused")
    assert gp_no_model and gp_no_model.get("refused")
    assert gp.get("level") == "GENERAL_PHYSICAL"
    probe = dsf.probe_general_physical()
    assert len(probe) == 2 * len(dsf.GENERAL_PHYSICAL_PROBE_FAMILIES)
    assert all(h["refused"] for h in probe)
    assert {h["query_mode"] for h in probe} == {"with_model", "without_model"}
    assert all(h.get("level") == "GENERAL_PHYSICAL" for h in probe)


def test_build_emits_sealed_receipt():
    out = dsf.build(consult_index=True)
    assert out.parent == RECEIPTS
    assert out.name == "DELTANET_STATE_FUNCTION.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == dsf.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    _assert_no_hardware_claims(doc)
    assert doc["accounting"]["catalog_weights"]["total"] == dsf.DELTANET_ACTIVE_TARGET
    assert doc["accounting"]["state"]["recurrent_resident"] == 150_994_944
    assert [c["id"] for c in doc["candidates"]] == list(dsf.REQUIRED_CANDIDATE_IDS)
    assert doc["candidate_counts"]["n"] == len(dsf.REQUIRED_CANDIDATE_IDS)
    assert doc["index_consultation"]["modes"] == ["with_model", "without_model"]
    assert doc["index_consultation"]["general_physical_probe_all_refused"] is True
    assert doc["index_consultation"]["with_model_hits"] > 0
    assert doc["index_consultation"]["without_model_hits"] > 0
    for row in doc["candidates"]:
        dsf.require_economics(row)
    # Open levers still carry both sides of the byte model.
    for lever in doc["open_byte_levers"]:
        assert lever["bytes_removed"] is not None
        assert lever["bytes_added"] is not None


def test_module_entrypoint_runs_and_emits_sealed_receipt():
    rc = dsf.main(["--build"])
    assert rc == 0
    doc = json.loads((RECEIPTS / dsf.RECEIPT).read_text())
    assert doc["schema"] == dsf.SCHEMA
    assert doc["seal_sha256"]
    assert doc["accounting"]["reconciled"] is True


def test_selftest_aliases_build():
    assert dsf.selftest is dsf.build or dsf.selftest().name == dsf.RECEIPT


def test_hardware_fields_stay_non_numeric_on_the_receipt():
    out = dsf.build(consult_index=True)
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


def test_narrow_d64_qkvz_is_half_payload_plus_headers():
    narrow = dsf.narrow_mixer_bytes(64)
    assert narrow["qkvz_rows"] == 8192
    # Half the payload of 44,564,520 - 40, plus a 40-byte header, times 48.
    per = (44_564_520 - 40) // 2 + 40
    assert per == 22_282_280
    assert narrow["qkvz"] == per * 48
    assert narrow["rec_state"] == 48 * 48 * 64 * 64 * 4
    assert narrow["rec_state"] == 37_748_736

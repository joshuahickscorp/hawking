"""Causal budget: UNKNOWN stays UNKNOWN; unreachable milestones are flagged.

A validator nobody has watched refuse is decoration. These tests watch
UNKNOWN coerced to a number, a pretty-sum closer, and a milestone that
would have to delete more than all MLP bytes.
"""
from __future__ import annotations

import json

import pytest

from tools.future import tps_budget as G
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, _assert_no_hardware_claims, write_receipt


def test_unknown_category_stays_unknown_and_is_never_coerced_to_a_number():
    decomp = G.decompose_decode_ms(dispatch_count=628)
    syn = decomp["categories"]["synchronization"]
    assert syn["ms"] == G.UNKNOWN
    assert syn["status"] == G.UNKNOWN
    assert syn["coerced_to_number"] is False
    assert not isinstance(syn["ms"], (int, float))

    for ident in (
        "state_bytes",
        "low_bit_decode_cost",
        "useful_arithmetic",
        "dispatch_cost",
        "command_encoder_cost",
        "cpu_submission",
        "state_transition",
    ):
        row = decomp["categories"][ident]
        assert row["ms"] == G.UNKNOWN, ident
        assert not isinstance(row["ms"], (int, float)), ident

    with pytest.raises(G.BudgetRefuse, match="UNKNOWN"):
        G.as_ms(syn["ms"], what="synchronization")
    with pytest.raises(G.BudgetRefuse, match="UNKNOWN"):
        G.as_ms(None, what="missing")
    with pytest.raises(G.BudgetRefuse, match="UNATTRIBUTED"):
        G.as_ms(G.UNATTRIBUTED, what="remainder")

    known = G.sum_known_ms(decomp["categories"])
    assert known["treated_unknown_as_zero"] is False
    assert "synchronization" in known["skipped_unknown"]
    assert "synchronization" not in known["included"]
    # A closer that would zero UNKNOWN to make 28.17 pretty is the failure.
    with pytest.raises(G.BudgetRefuse, match="UNKNOWN"):
        G.close_arithmetic_by_zeroing_unknown(
            decomp["categories"],
            decomp["token_ms_derived_from_established_decode_rate"],
        )


def test_milestone_arithmetic_flags_unreachable():
    unreachable = G.milestone_row(
        200,
        roof_gb_s=G.ESTABLISHED_CLEAN_ROOF_GB_S,
        active_bytes=G.ESTABLISHED_ACTIVE_BYTES,
        non_mlp_fraction=G.ESTABLISHED_NON_MLP_FRACTION,
    )
    assert unreachable["status"] == G.UNREACHABLE
    assert unreachable["status"] == "ARITHMETICALLY_UNREACHABLE_BY_MLP_ALONE"
    assert unreachable["remaining_mlp_fraction"] < 0
    assert unreachable["required_total_byte_fraction"] < G.ESTABLISHED_NON_MLP_FRACTION

    reachable = G.milestone_row(
        100,
        roof_gb_s=G.ESTABLISHED_CLEAN_ROOF_GB_S,
        active_bytes=G.ESTABLISHED_ACTIVE_BYTES,
        non_mlp_fraction=G.ESTABLISHED_NON_MLP_FRACTION,
    )
    assert reachable["status"] != G.UNREACHABLE
    assert reachable["remaining_mlp_fraction"] >= 0
    assert reachable["status"] == G.REQUIRES_MLP

    no_cut = G.milestone_row(
        50,
        roof_gb_s=G.ESTABLISHED_CLEAN_ROOF_GB_S,
        active_bytes=G.ESTABLISHED_ACTIVE_BYTES,
        non_mlp_fraction=G.ESTABLISHED_NON_MLP_FRACTION,
    )
    assert no_cut["status"] == G.NO_MLP_CUT
    assert no_cut["required_total_byte_fraction"] > 1.0

    ladder = G.milestone_ladder()
    by_t = {int(r["milestone_tokens_per_second"]): r for r in ladder}
    assert set(by_t) == {50, 71, 100, 125, 150}
    assert by_t[150]["status"] != G.UNREACHABLE
    assert by_t[150]["remaining_mlp_fraction"] >= 0


def test_dispatch_count_is_rederived_from_encode_sites():
    geo = G.load_geometry()
    unfused = G.count_dispatches_per_decoded_token(geo, G.Fusion.env_unset_default())
    sealed = G.count_dispatches_per_decoded_token(geo, G.Fusion.sealed_resident())
    assert unfused["not_copied_from_a_receipt"] is True
    assert sealed["not_copied_from_a_receipt"] is True
    assert unfused["total"] == 964
    assert sealed["total"] == 628
    assert unfused["n_dn_layers"] == 48
    assert unfused["n_gqa_layers"] == 16
    # Savings from the sealed fusion set, re-derived not copied:
    # mlp swiglu 2*64, gqa qkv 2*16, dn inproj 48, add_rmsnorm 2*64.
    assert sealed["total"] == unfused["total"] - (2 * 64) - (2 * 16) - 48 - (2 * 64)
    assert unfused["by_kind"]["embed"] == 1
    assert unfused["by_kind"]["terminal"] == 3
    assert sealed["by_kind"]["terminal"] == 2  # final rms folded into last add_rms


def test_record_seals_receipt_without_hardware_keys():
    out = G.record()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == G.RECEIPT
    assert doc["schema"] == G.SCHEMA
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["gpu_authority"] is False
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["dispatch_count"]["current_production_for_this_budget"] == 628
    assert doc["dispatch_count"]["source_default_env_unset"]["total"] == 964
    assert doc["dispatch_count"]["not_copied_from_a_receipt"] is True
    remainder = doc["decode_ms_decomposition"]["categories"]["measured_remainder"]
    assert remainder["status"] == G.UNATTRIBUTED
    assert remainder["ms"] == G.UNATTRIBUTED
    assert remainder["folded_into_named_categories"] is False
    syn = doc["decode_ms_decomposition"]["categories"]["synchronization"]
    assert syn["ms"] == G.UNKNOWN
    assert not isinstance(syn["ms"], (int, float))
    assert doc["resident_callable"]["gpu_authority"] is False
    assert doc["resident_callable"]["evidence_class"] == "STATIC_ONLY"
    _assert_no_hardware_claims(doc)

    def walk(node: object, path: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                    raise AssertionError(f"{here} = {v!r} is a hardware field")
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)


def test_writing_a_hardware_named_field_raises():
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "_TPS_BUDGET_HARDWARE_PROBE.json",
            {"schema": "probe", "tps": 71.0},
            "tools/future/test_tps_budget.py",
        )
    with pytest.raises(HardwareClaimError):
        G.refuse_hardware_measurement("gpu_ns")


def test_decode_path_markers_fail_closed_when_missing():
    markers = G.decode_path_markers("not the decode path")
    assert markers["ok"] is False
    assert markers["missing"]
    live = G.decode_path_markers()
    assert live["ok"] is True
    assert live["skip_terminal_in_this_tree"] is False

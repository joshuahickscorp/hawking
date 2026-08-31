"""Pins for tools/future/dispatch_motifs.py.

The load-bearing refusal: an unreconciled motif census raises MotifRefuse
and does not emit DISPATCH_MOTIFS.json. A validator nobody has watched
refuse is decoration.
"""
from __future__ import annotations

import json

import pytest

from tools.future import dispatch_motifs as dm
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    HardwareClaimError,
    _assert_no_hardware_claims,
    write_receipt,
)
from tools.future.tps_budget import Fusion, count_dispatches_per_decoded_token, load_geometry


def _walk_both():
    geo = load_geometry()
    sealed = dm.walk_launches(geo, Fusion.sealed_resident())
    unfused = dm.walk_launches(geo, Fusion.env_unset_default())
    return geo, sealed, unfused


def test_unreconciled_census_refuses_rather_than_emit():
    """A closer that would publish 627 as 628 is the failure."""
    geo, sealed, unfused = _walk_both()
    sealed_counts = dm.cluster_launches(sealed)
    unfused_counts = dm.cluster_launches(unfused)

    # Truncated walk: drop one sealed launch, census must refuse.
    truncated = dm.cluster_launches(sealed[:-1])
    with pytest.raises(dm.MotifRefuse, match="unreconciled"):
        dm.reconcile_census(truncated, unfused_counts)

    # Wrong expected totals, even if the walk is internally consistent.
    with pytest.raises(dm.MotifRefuse, match="unreconciled"):
        dm.reconcile_census(
            sealed_counts,
            unfused_counts,
            sealed_expected=627,
            unfused_expected=964,
        )

    # Unknown motif id.
    bogus = dict(sealed_counts)
    bogus["not_a_motif"] = 1
    with pytest.raises(dm.MotifRefuse, match="unreconciled"):
        dm.reconcile_census(bogus, unfused_counts)

    # Missing catalog id.
    missing = dict(sealed_counts)
    del missing["argmax"]
    with pytest.raises(dm.MotifRefuse, match="unreconciled"):
        dm.reconcile_census(missing, unfused_counts)

    # The live walk still reconciles; refusal did not poison it.
    rec = dm.reconcile_census(sealed_counts, unfused_counts)
    assert rec["ok"] is True
    assert rec["sealed_sum"] == 628
    assert rec["unfused_sum"] == 964
    assert rec["fusion_removed"] == 336
    assert rec["treated_unknown_as_zero"] is False


def test_truncated_walk_cannot_be_recorded(tmp_path, monkeypatch):
    """record() must not write a receipt if the walk no longer sums to 628."""
    geo, sealed, unfused = _walk_both()
    monkeypatch.setattr(
        dm,
        "walk_launches",
        lambda geo, fusion: (
            sealed[:-1]
            if fusion == Fusion.sealed_resident()
            else unfused
        ),
    )
    with pytest.raises(dm.MotifRefuse):
        dm.build()


def test_census_sums_to_established_628_and_964():
    geo, sealed, unfused = _walk_both()
    assert len(sealed) == dm.ESTABLISHED_SEALED == 628
    assert len(unfused) == dm.ESTABLISHED_UNFUSED == 964
    sealed_counts = dm.cluster_launches(sealed)
    unfused_counts = dm.cluster_launches(unfused)
    rec = dm.reconcile_census(sealed_counts, unfused_counts)
    assert rec["sealed_sum"] == 628
    assert rec["unfused_sum"] == 964
    assert rec["fusion_removed"] == 336
    # Independent oracle: the coarse tps_budget walk.
    assert count_dispatches_per_decoded_token(geo, Fusion.sealed_resident())["total"] == 628
    assert count_dispatches_per_decoded_token(geo, Fusion.env_unset_default())["total"] == 964
    assert rec["sealed_families"] == {
        "representation_decode": 258,
        "state_update": 112,
        "producer_consumer": 65,
        "norm": 49,
        "residual": 128,
        "routing": 16,
        "elementwise": 0,
    }
    assert rec["unfused_families"] == {
        "representation_decode": 402,
        "state_update": 112,
        "producer_consumer": 65,
        "norm": 177,
        "residual": 128,
        "routing": 16,
        "elementwise": 64,
    }
    assert sum(rec["sealed_families"].values()) == 628
    assert sum(rec["unfused_families"].values()) == 964


def test_high_frequency_motif_counts_are_the_layer_repetition():
    _, sealed, unfused = _walk_both()
    s = dm.cluster_launches(sealed)
    u = dm.cluster_launches(unfused)
    # Sealed 64-wide: MLP suffix + both fused residuals.
    assert s["mlp_fused_gate_up_swiglu"] == 64
    assert s["mlp_down_proj"] == 64
    assert s["mlp_add_residual_rmsnorm"] == 64
    assert s["mixer_add_residual_rmsnorm"] == 64
    # Sealed 48-wide: DeltaNet sequence.
    assert s["dn_inproj_pair_concat"] == 48
    assert s["dn_rearrange_conv"] == 48
    assert s["dn_ba_to_decay"] == 48
    assert s["dn_gated_delta"] == 48
    assert s["dn_gated_rmsnorm"] == 48
    assert s["dn_out_proj"] == 48
    # Sealed 16-wide: GQA sequence.
    assert s["gqa_fused_qkv"] == 16
    assert s["gqa_qk_norm_rope_cache"] == 16
    assert s["gqa_mha_decode"] == 16
    assert s["gqa_sigmoid_gate"] == 16
    assert s["gqa_o_proj"] == 16
    # Singletons.
    assert s["embed_lookup"] == 1
    assert s["mixer_input_rmsnorm"] == 1  # layer 0 only
    assert s["lm_head"] == 1
    assert s["argmax"] == 1
    assert s["final_rmsnorm"] == 0
    assert s["mlp_swiglu"] == 0
    assert s["dn_gated_delta_fused_ba"] == 0
    # Unfused extras that fusion removed as kinds.
    assert u["mixer_input_rmsnorm"] == 64
    assert u["mlp_post_attn_rmsnorm"] == 64
    assert u["final_rmsnorm"] == 1
    assert u["mlp_gate_proj"] == 64
    assert u["mlp_up_proj"] == 64
    assert u["mlp_swiglu"] == 64
    assert u["dn_inproj_qkvz"] == 48
    assert u["dn_inproj_ba"] == 48
    assert u["gqa_q_proj"] == 16
    assert u["gqa_k_proj"] == 16
    assert u["gqa_v_proj"] == 16
    assert u["mixer_add_residual"] == 64
    assert u["mlp_add_residual"] == 64
    assert u["dn_inproj_pair_concat"] == 0
    assert u["gqa_fused_qkv"] == 0
    assert u["mlp_fused_gate_up_swiglu"] == 0


def test_layer0_is_dn_and_gqa_is_every_fourth():
    geo, sealed, _ = _walk_both()
    interval = int(geo["QWEN38_FULL_ATTENTION_INTERVAL"])
    assert interval == 4
    tiles = dm.four_layer_tile(sealed, interval)
    assert tiles["n_tiles"] == 16
    assert tiles["first_tile_dispatches"] == 40
    assert tiles["later_tile_dispatches"] == [39]
    assert tiles["later_tiles_identical"] is True
    assert tiles["tiles"][0]["kinds"] == ["dn", "dn", "dn", "gqa"]
    assert tiles["tiles"][0]["per_layer"] == [11, 10, 10, 9]
    assert tiles["tiles"][1]["per_layer"] == [10, 10, 10, 9]
    # 40 + 15*39 = 625 layer launches; + embed + lm_head + argmax = 628.
    assert 40 + 15 * 39 + 3 == 628


def test_ba_delta_inner_cut_is_48_not_the_region():
    geo = load_geometry()
    fused = Fusion(
        mlp="swiglu",
        gqa_qkv=True,
        dn_inproj=True,
        add_rmsnorm=True,
        ba_delta=True,
        argmax_two_pass=False,
    )
    launches = dm.walk_launches(geo, fused)
    assert len(launches) == 580
    counts = dm.cluster_launches(launches)
    assert counts["dn_ba_to_decay"] == 0
    assert counts["dn_gated_delta"] == 0
    assert counts["dn_gated_delta_fused_ba"] == 48


def test_cited_marginal_carries_the_caveat_and_is_not_a_measurement():
    product = dm.cited_marginal_product_ms(48)
    assert product["cited_marginal_us"] == 6.25
    assert product["product_us"] == 300.0
    assert product["product_ms"] == 0.3
    assert product["status"] == "CITED_MARGINAL_NOT_EXTRAPOLATED"
    assert product["not_a_hardware_measurement"] is True
    assert "336" in product["caveat"]
    assert "do not assume" in product["caveat"].lower() or "Do not assume" in product["caveat"]
    assert "remaining 628" in product["caveat"]


def test_residual_standalone_removes_zero_and_mlp_suffix_removes_128():
    _, sealed, _ = _walk_both()
    candidates = {row["id"]: row for row in dm.region_candidates(dm.cluster_launches(sealed))}
    residual = candidates["residual_rmsnorm_is_a_boundary_not_a_region"]
    assert residual["dispatches_removed"] == 0
    assert residual["judgment"] == "NO_PRODUCER_CONSUMER_BOUNDARY"
    assert residual["blocked_today"] is True
    mlp = candidates["mlp_suffix_representation_region"]
    assert mlp["sealed_dispatches"] == 192
    assert mlp["launches_after"] == 64
    assert mlp["dispatches_removed"] == 128
    assert mlp["blocked_today"] is False
    dn = candidates["dn_layer_state_machine"]
    assert dn["sealed_dispatches"] == 337
    assert dn["launches_after"] == 48
    assert dn["dispatches_removed"] == 289
    gqa = candidates["gqa_layer_static_skeleton"]
    assert gqa["sealed_dispatches"] == 96
    assert gqa["dispatches_removed"] == 80
    assert gqa["blocked_today"] is True
    token = candidates["token_graph_persistent_executor"]
    assert token["dispatches_removed"] == 627
    assert token["not_628_to_500"] is True
    ba = candidates["dn_ba_delta_existing_lever"]
    assert ba["dispatches_removed"] == 48
    assert ba["judgment"] == "EXISTING_FUSION_LEVER_NOT_A_NEW_REGION"


def test_every_candidate_names_metal_blockers_and_how_many_it_removes():
    _, sealed, _ = _walk_both()
    candidates = dm.region_candidates(dm.cluster_launches(sealed))
    assert candidates
    for row in candidates:
        assert set(row["metal_blockers"]) == set(dm.METAL_BLOCKERS)
        assert isinstance(row["dispatches_removed"], int)
        assert row["dispatches_removed"] >= 0
        assert row["icb_is_the_wrong_textbook_for_the_6_25us_class"] is True
        assert row["cheapest_falsifier"]
        assert row["form"] in {
            "static_skeleton_with_dynamic_token_or_route_slots",
            "graph_replay_equivalent",
            "representation_native_region",
            "long_lived_state_machine",
            "producer_consumer_boundary",
        }


def test_ranking_names_cheapest_falsifier_for_the_top_unblocked_region():
    _, sealed, _ = _walk_both()
    candidates = dm.region_candidates(dm.cluster_launches(sealed))
    ranking = dm.rank_candidates(candidates)
    assert ranking["top_unblocked_region"] == "dn_layer_state_machine"
    assert ranking["top_dispatches_removed"] == 289
    assert ranking["cheapest_falsifier_for_top"]
    assert "FUSE_BA_DELTA" in ranking["cheapest_falsifier_for_top"]
    # Existing 48-launch lever is recorded but is not the top region.
    assert "dn_ba_delta_existing_lever" in ranking["order"]
    assert ranking["order_unblocked_regions"][0] == "dn_layer_state_machine"
    # 628 -> 500 is not the goalpost any candidate claims.
    by_id = {row["id"]: row for row in candidates}
    assert all(row["not_628_to_500"] for row in by_id.values())


def test_every_motif_has_a_region_judgment_and_none_is_standalone():
    assert set(dm.MOTIF_REGION) == {row["id"] for row in dm.MOTIF_CATALOG}
    for spec in dm.MOTIF_CATALOG:
        judgment = dm.MOTIF_REGION[spec["id"]]
        assert judgment["standalone"] is False, spec["id"]
        assert judgment["absorbed_by"]
        assert judgment["why"]
        if judgment["form"] is not None:
            assert judgment["form"] in {
                "static_skeleton_with_dynamic_token_or_route_slots",
                "graph_replay_equivalent",
                "representation_native_region",
                "long_lived_state_machine",
            }


def test_mha_is_the_dynamic_shape_and_sigmoid_is_not_expert_routing():
    spec = dm.CATALOG_BY_ID["gqa_mha_decode"]
    assert "threadgroup_memory" in spec["dynamic_slots"]
    assert "seq_len" in spec["dynamic_slots"]
    gate = dm.CATALOG_BY_ID["gqa_sigmoid_gate"]
    assert gate["family"] == "routing"
    assert "not expert routing" in gate["what"]


def test_helper_markers_fail_closed_when_missing():
    dead = dm.helper_markers("not the decode path")
    assert dead["ok"] is False
    assert dead["missing"]
    live = dm.helper_markers()
    assert live["ok"] is True
    assert not live["missing"]


def test_argmax_two_pass_is_refused_as_production():
    geo = load_geometry()
    with pytest.raises(dm.MotifRefuse, match="two-pass"):
        dm.walk_launches(
            geo,
            Fusion(
                mlp="swiglu",
                gqa_qkv=True,
                dn_inproj=True,
                add_rmsnorm=True,
                argmax_two_pass=True,
            ),
        )


def test_unknown_motif_id_from_walk_refuses():
    with pytest.raises(dm.MotifRefuse, match="unknown motif"):
        dm.cluster_launches([{"motif_id": "ghost_kernel"}])


def test_record_seals_receipt_without_hardware_keys():
    out = dm.record()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == dm.RECEIPT
    assert doc["schema"] == dm.SCHEMA
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["gpu_authority"] is False
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["reconciliation"]["sealed_sum"] == 628
    assert doc["reconciliation"]["unfused_sum"] == 964
    assert doc["walk"]["sealed_launches"] == 628
    assert doc["walk"]["unfused_launches"] == 964
    assert doc["ranking"]["top_unblocked_region"] == "dn_layer_state_machine"
    assert doc["established"]["do_not_extrapolate_linearly_to_zero"] is True
    assert doc["established"]["do_not_assume_remaining_628_cost_the_same"] is True
    motif_sealed = sum(row["sealed_count"] for row in doc["motifs"])
    motif_unfused = sum(row["unfused_count"] for row in doc["motifs"])
    assert motif_sealed == 628
    assert motif_unfused == 964
    hf = {row["id"]: row["sealed_count"] for row in doc["high_frequency_motifs"]}
    assert hf["mlp_fused_gate_up_swiglu"] == 64
    assert hf["dn_gated_delta"] == 48
    assert hf["gqa_mha_decode"] == 16
    assert all(n >= 16 for n in hf.values())
    assert doc["high_frequency_sealed_sum"] == 624
    assert "embed_lookup" not in hf
    assert "mixer_input_rmsnorm" not in hf
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
            "_DISPATCH_MOTIFS_HARDWARE_PROBE.json",
            {"schema": "probe", "tps": 40.79},
            "tools/future/test_dispatch_motifs.py",
        )


def test_build_is_the_in_memory_document_and_main_records():
    doc = dm.build()
    assert doc["schema"] == dm.SCHEMA
    assert doc["reconciliation"]["ok"] is True
    rc = dm.main(["--build"])
    assert rc == 0
    assert (RECEIPTS / dm.RECEIPT).is_file()

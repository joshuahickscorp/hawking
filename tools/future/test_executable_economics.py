"""Pins for tools/future/executable_economics.py.

A ratio without bytes_added is not a candidate. A candidate without a
declared stream class is not a candidate. Unique aux bytes are not billed
at the organ average.
"""
from __future__ import annotations

import json

import pytest
from pathlib import Path

from tools.future import executable_economics as ee
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    _assert_no_hardware_claims,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def test_bytes_removed_without_bytes_added_is_refused():
    """A compression ratio without executable economics is not a candidate."""
    with pytest.raises(ee.IncompleteEconomics, match="bytes_added"):
        ee.score(bytes_removed=1_000_000)
    with pytest.raises(ee.IncompleteEconomics, match="executable economics"):
        ee.score(bytes_removed=ee.MLP_ACTIVE_BYTES, bytes_added=None)
    with pytest.raises(ee.IncompleteEconomics, match="bytes_added"):
        ee.score(bytes_removed=0)


def test_undeclared_stream_class_is_refused():
    """Defaulting to the organ average is how the aux-u8 overcredit hid."""
    with pytest.raises(ee.IncompleteEconomics, match="stream_class"):
        ee.score(bytes_removed=1_000_000, bytes_added=0, organ="mlp")
    with pytest.raises(ee.IncompleteEconomics, match="organ average"):
        ee.score(
            bytes_removed=534_773_760,
            bytes_added=0,
            organ="mlp",
            consuming_primitive="FusedDecodeCompute",
        )
    with pytest.raises(ee.EconomicsRefuse, match="unknown stream_class"):
        ee.score(
            bytes_removed=1_000,
            bytes_added=0,
            stream_class="organ_average",
        )


def test_explicit_zero_bytes_added_is_a_complete_claim():
    row = ee.score(
        bytes_removed=1_000_000,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
    )
    assert row["ok"] is True
    assert row["bytes_added_supplied"] is True
    assert row["bytes_added"]["total"] == 0
    assert row["bytes_removed"] == 1_000_000
    assert row["stream_class"] == ee.STREAM_CLASS_WEIGHT_CODES
    assert row["terms"]["byte_ms_delta"] < 0.0


def test_mlp_full_removal_with_extra_flops_scores_both_terms():
    """Removing 100% of MLP bytes AND adding 100 extra FLOPs per output
    element must move the score on both terms, not on bytes alone.
    """
    bytes_only = ee.score(
        bytes_removed=ee.MLP_ACTIVE_BYTES,
        bytes_added=0,
        extra_flops_per_output_element=0.0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
        stream_gb_s=ee.MLP_GB_S,
        stream_on_critical_path=True,
    )
    both = ee.score(
        bytes_removed=ee.MLP_ACTIVE_BYTES,
        bytes_added=0,
        extra_flops_per_output_element=100.0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
        stream_gb_s=ee.MLP_GB_S,
        stream_on_critical_path=True,
    )
    assert bytes_only["terms"]["byte_ms_delta"] < 0.0
    assert bytes_only["terms"]["flop_ms_delta"] == 0.0
    assert both["terms"]["flop_ms_delta"] > 0.0
    assert both["terms"]["byte_ms_delta"] == pytest.approx(
        bytes_only["terms"]["byte_ms_delta"]
    )
    assert both["n_output_elements"] == ee.ORGAN_OUTPUT_ELEMENTS["mlp"]
    assert both["n_output_elements"] == 2_555_904
    assert both["predicted_ms_delta"] > bytes_only["predicted_ms_delta"]
    assert both["predicted_token_ms"] > bytes_only["predicted_token_ms"]
    assert both["predicted_ms_delta"] != both["terms"]["byte_ms_delta"]
    assert both["predicted_ms_delta"] == pytest.approx(
        both["terms"]["byte_ms_delta"] + both["terms"]["flop_ms_delta"]
    )
    assert bytes_only["terms"]["byte_ms_delta"] == pytest.approx(
        -ee.bytes_to_ms(ee.MLP_ACTIVE_BYTES, ee.MLP_GB_S)
    )
    assert bytes_only["terms"]["byte_ms_delta"] == pytest.approx(-ee.MLP_MS, abs=1e-3)


def test_broadcast_aux_bytes_are_not_billed_at_the_organ_average():
    aux = ee.score(
        bytes_removed=534_773_760,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=ee.STREAM_CLASS_BROADCAST_AUX,
        candidate_id="quantize_aux_u8",
        reusable_family=True,
    )
    codes = ee.score(
        bytes_removed=534_773_760,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
    )
    # Aux unique bytes are cache-served. Paired 50% drop was inside noise.
    assert aux["terms"]["byte_ms_delta"] == pytest.approx(0.0, abs=1e-9)
    assert aux["predicted_ms_saved"] == pytest.approx(0.0, abs=1e-9)
    assert aux["assumptions"]["stream_on_critical_path"] is False
    # The same unique-byte count as codes is still a real save.
    assert codes["predicted_ms_saved"] > 0.0
    assert codes["assumptions"]["stream_on_critical_path"] is True
    # Old organ-average would have billed ~1.55 ms for this aux removal.
    organ_avg = 534_773_760 / ee.MLP_GB_S * 1e-6
    assert organ_avg == pytest.approx(1.5541, abs=1e-3)
    assert aux["predicted_ms_saved"] < 0.1 * organ_avg


def test_bytes_added_five_fields_reduce_the_save():
    removed = 534_773_760  # quantize_aux_u8, but billed as codes here
    no_add = ee.score(
        bytes_removed=removed,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
    )
    with_add = ee.score(
        bytes_removed=removed,
        bytes_added={
            "generator": 10_000_000,
            "embeddings": 1_000_000,
            "residuals": 2_000_000,
            "metadata": 3_000_000,
            "state": 4_000_000,
        },
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
    )
    assert with_add["bytes_added"]["total"] == 20_000_000
    assert with_add["net_bytes"] == removed * -1 + 20_000_000
    assert with_add["terms"]["byte_added_ms"] > 0.0
    assert with_add["predicted_ms_saved"] < no_add["predicted_ms_saved"]
    for key in ee.BYTES_ADDED_FIELDS:
        assert key in with_add["bytes_added"]


def test_one_percent_bar_and_family_override():
    # Under the bar even as codes (binding stream, calibrated rate).
    small = ee.score(
        bytes_removed=1_000_000,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
        candidate_id="tiny_header_pack",
    )
    assert small["s020_section_20"]["clears_time_bar"] is False
    assert small["verdict"] == "IMMATERIAL"

    over = ee.score(
        bytes_removed=ee.MLP_CODE_BYTES,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
    )
    assert over["s020_section_20"]["clears_time_bar"] is True
    assert over["verdict"] == "MATERIAL"
    assert "clears_s020_section_20_time_bar" in over["verdict_reasons"]

    family = ee.score(
        bytes_removed=1_000_000,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
        reusable_family=True,
        candidate_id="a_family",
    )
    assert family["s020_section_20"]["clears_time_bar"] is False
    assert family["verdict"] == "MATERIAL"
    assert "reusable_representation_family" in family["verdict_reasons"]

    falsifier = ee.score(
        bytes_removed=0,
        bytes_added=0,
        high_information_falsifier=True,
        candidate_id="a_falsifier",
        stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
    )
    assert falsifier["verdict"] == "MATERIAL"
    assert "high_information_falsifier" in falsifier["verdict_reasons"]


def test_dead_status_is_immaterial_even_if_the_byte_save_was_large():
    row = ee.score(
        bytes_removed=ee.MLP_CODE_BYTES,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
        reusable_family=True,
        status="MEASURED_NEGATIVE",
    )
    assert row["live"] is False
    assert row["verdict"] == "IMMATERIAL"
    assert row["predicted_ms_saved"] > ee.S020_SECTION_20_BAR_MS


def test_new_representation_bandwidth_is_an_assumption_with_a_range():
    row = ee.score(
        bytes_removed=100_000_000,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="TiledProjection",
        stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
    )
    assume = row["assumptions"]
    assert assume["bandwidth_regime"] == "unknown_new_representation"
    assert assume["bandwidth_is_assumption"] is True
    lo, hi = assume["bandwidth_gb_s_range"]
    assert lo < hi
    assert lo == pytest.approx(ee.AFFINE_Q2_GB_S_AT_5MB)
    assert hi == pytest.approx(ee.LM_HEAD_GB_S)
    assert hi < ee.CLEAN_GEMV_GB_S
    dlo, dhi = row["predicted_ms_delta_range"]
    assert dlo <= dhi
    assert row["predicted_tps_range"][0] <= row["predicted_tps_range"][1]
    assert "ASSUMPTION" in assume["bandwidth_note"]


def test_affine_q2_is_not_dressed_as_497():
    row = ee.score(
        bytes_removed=534_773_760,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
    )
    assume = row["assumptions"]
    assert assume["bandwidth_regime"] == "affine_q2_family"
    lo, hi = assume["bandwidth_gb_s_range"]
    assert hi == pytest.approx(ee.AFFINE_Q2_SATURATED_GB_S)
    assert hi < ee.LM_HEAD_GB_S
    assert assume["bandwidth_gb_s_nominal"] == pytest.approx(ee.MLP_GB_S)


def test_dispatch_class_is_class_dependent():
    mlp = ee.score(
        bytes_removed=0,
        bytes_added=0,
        dispatch_delta=-48,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        dispatch_class="mlp_gqa_norm_fusion",
        stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
    )
    dn = ee.score(
        bytes_removed=0,
        bytes_added=0,
        dispatch_delta=-48,
        organ="deltanet",
        consuming_primitive="FusedDecodeCompute",
        dispatch_class="deltanet_ba",
        stream_class=ee.STREAM_CLASS_ACTIVATION,
    )
    assert mlp["terms"]["dispatch_ms_delta"] == pytest.approx(-48 * 6.25 / 1000.0)
    assert dn["terms"]["dispatch_ms_delta"] == pytest.approx(-48 * 2.884 / 1000.0)
    assert mlp["terms"]["dispatch_ms_delta"] != dn["terms"]["dispatch_ms_delta"]


def test_unknown_primitive_is_still_scored_with_a_range():
    row = ee.score(
        bytes_removed=10_000,
        bytes_added={"metadata": 100},
        organ="mlp",
        consuming_primitive="NotAnAtlasPrimitive",
        stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
    )
    assert row["ok"] is True
    assert row["assumptions"]["bandwidth_is_assumption"] is True
    assert row["consuming_primitive"] == "NotAnAtlasPrimitive"


def test_bytes_removed_past_the_organ_is_refused():
    with pytest.raises(ee.EconomicsRefuse, match="exceeds"):
        ee.score(
            bytes_removed=ee.MLP_ACTIVE_BYTES + 1,
            bytes_added=0,
            organ="mlp",
            stream_class=ee.STREAM_CLASS_WEIGHT_CODES,
        )


def test_recorded_catalog_covers_aux_code_deltanet_and_dispatch_size():
    ranked = ee.rank_recorded()
    ids = [r["id"] for r in ranked]
    assert len(ids) == len(set(ids))

    aux = json.loads((RECEIPTS / "MLP_AUXILIARY_INFORMATION.json").read_text())
    code = json.loads((RECEIPTS / "MLP_CODE_INFORMATION.json").read_text())
    dn = json.loads((RECEIPTS / "DELTANET_REPRESENTATION.json").read_text())
    for src in (aux, code, dn):
        for cand in src["candidates"]:
            assert cand["id"] in ids, cand["id"]

    assert "surviving_dispatch_size_amortize_sub20mb" in ids
    assert "dispatch_size_concat_to_lm_head_mb" in ids
    assert "group_size_256" in ids
    assert "group_size_512" in ids
    assert "group_size_1024" in ids
    assert "quantize_aux_u8" in ids
    assert "entropy_coded_code_stream" in ids
    assert "pack_headers" in ids
    assert "function_replacement" in ids
    assert "fused_update_consume" in ids

    by_id = {r["id"]: r for r in ranked}
    assert by_id["quantize_aux_u8"]["stream_class"] == ee.STREAM_CLASS_BROADCAST_AUX
    assert by_id["quantize_aux_u8"]["verdict"] == "MATERIAL"
    assert by_id["quantize_aux_u8"]["predicted_ms_saved"] == pytest.approx(0.0, abs=1e-6)
    assert by_id["quantize_aux_u8"]["delta_from_legacy_ms_saved"] == pytest.approx(
        -1.5541, abs=1e-3
    )
    assert "reusable_representation_family" in by_id["quantize_aux_u8"]["verdict_reasons"]
    assert by_id["quantize_aux_u8"]["s020_section_20"]["clears_time_bar"] is False

    assert by_id["group_size_1024"]["live"] is False
    assert by_id["group_size_1024"]["verdict"] == "IMMATERIAL"
    assert by_id["group_size_1024"]["status"] == "GRANULARITY_REFUTED"
    assert by_id["group_size_1024"]["stream_class"] == ee.STREAM_CLASS_BROADCAST_AUX
    assert by_id["group_size_1024"]["predicted_ms_saved"] == pytest.approx(0.0, abs=1e-6)
    assert by_id["group_size_256"]["live"] is False
    assert by_id["group_size_256"]["status"] == "GRANULARITY_REFUTED"

    assert by_id["entropy_coded_code_stream"]["verdict"] == "MATERIAL"
    assert by_id["entropy_coded_code_stream"]["stream_class"] == ee.STREAM_CLASS_WEIGHT_CODES
    assert by_id["entropy_coded_code_stream"]["predicted_ms_saved"] > 0.0
    # Binding-stream re-price, not a wipe.
    assert by_id["entropy_coded_code_stream"]["predicted_ms_saved"] == pytest.approx(
        0.1520, abs=5e-3
    )
    assert by_id["entropy_coded_code_stream"]["legacy_organ_average_ms_saved"] == pytest.approx(
        0.8070, abs=1e-3
    )

    assert by_id["pack_headers"]["verdict"] == "IMMATERIAL"
    assert by_id["surviving_dispatch_size_amortize_sub20mb"]["verdict"] == "MATERIAL"
    assert "reusable_representation_family" in by_id[
        "surviving_dispatch_size_amortize_sub20mb"
    ]["verdict_reasons"] or "high_information_falsifier" in by_id[
        "surviving_dispatch_size_amortize_sub20mb"
    ]["verdict_reasons"]
    assert by_id["dispatch_size_concat_to_lm_head_mb"]["live"] is False
    assert by_id["dispatch_size_concat_to_lm_head_mb"]["verdict"] == "IMMATERIAL"

    saves = [r["predicted_ms_saved"] for r in ranked]
    assert saves == sorted(saves, reverse=True)

    fused = by_id["fused_update_consume"]
    assert fused["dispatch_delta"] == -48
    assert fused["assumptions"]["dispatch_class"] == "deltanet_ba"
    assert fused["terms"]["dispatch_ms_delta"] == pytest.approx(
        -48 * 2.884 / 1000.0, abs=1e-3
    )


def test_group_size_1024_is_repriced_and_capability_dead():
    by_id = {r["id"]: r for r in ee.rank_recorded()}
    row = by_id["group_size_1024"]
    assert row["bytes_removed"] == 1_002_700_800
    assert row["bytes_added_total"] == 0
    assert row["stream_class"] == ee.STREAM_CLASS_BROADCAST_AUX
    assert row["predicted_ms_saved"] == pytest.approx(0.0, abs=1e-6)
    assert row["legacy_organ_average_ms_saved"] == pytest.approx(2.9140, abs=1e-3)
    assert row["delta_from_legacy_ms_saved"] == pytest.approx(-2.9140, abs=1e-3)
    assert row["live"] is False


def test_every_ranked_row_reports_its_delta_from_the_old_model():
    for row in ee.rank_recorded():
        assert "legacy_organ_average_ms_saved" in row
        assert "delta_from_legacy_ms_saved" in row
        assert "stream_class" in row
        assert row["stream_class"] in ee.STREAM_CLASS_NAMES
        assert row["delta_from_legacy_ms_saved"] == pytest.approx(
            row["predicted_ms_saved"] - row["legacy_organ_average_ms_saved"],
            abs=1e-3,
        )


def test_rejected_dense_remat_is_scored_on_added_bytes_not_removal_alone():
    by_id = {r["id"]: r for r in ee.rank_recorded()}
    gen = by_id["generated_tensors"]
    assert gen["status"] == "REJECTED_DENSE_REMAT"
    assert gen["bytes_added_total"] == ee.MLP_PARAMS * ee.F16_BYTES
    assert gen["net_bytes"] > 0
    assert gen["predicted_ms_saved"] < 0.0
    assert gen["verdict"] == "IMMATERIAL"


def test_atlas_primitives_are_the_vocabulary():
    for name in (
        "FusedDecodeCompute",
        "TiledProjection",
        "LocalStateMachine",
        "LayoutTransform",
        "MoveOrRecompute",
        "DirectRoutedAccumulate",
        "ConditionalPhysicalProgram",
    ):
        assert name in ATLAS_PRIMITIVES


def test_aux_u8_lut_is_predicted_not_a_win():
    lut = ee.score(
        bytes_removed=534_773_760,
        bytes_added={"metadata": 393_216},
        extra_flops_per_output_element=0.0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        stream_class=ee.STREAM_CLASS_BROADCAST_AUX,
        reusable_family=True,
    )
    assert lut["predicted_ms_delta"] == pytest.approx(0.0, abs=1e-9)
    assert lut["predicted_ms_saved"] == pytest.approx(0.0, abs=1e-9)
    assert lut["predicted_ms_delta"] >= 0.0  # not a win
    assert lut["s020_section_20"]["clears_time_bar"] is False


def test_three_real_abs_are_retrodicted():
    doc = ee.build()
    by = {r["id"]: r for r in doc["retrodictions"]}
    widen = by["DELTANET_WIDEN_AB"]
    assert widen["measured_ms_delta"] == pytest.approx(-1.0245, abs=1e-4)
    assert widen["predicted_ms_delta"] == pytest.approx(-0.1384, abs=1e-3)
    assert widen["predicted_is_win"] is True
    assert widen["measured_is_win"] is True

    cheapen = by["MLP_DECODE_CHEAPEN"]
    assert cheapen["bytes_removed"] == 0
    assert cheapen["predicted_ms_delta"] == pytest.approx(0.0, abs=1e-9)
    assert cheapen["measured_ms_delta"] == pytest.approx(-1.745, abs=1e-3)
    assert cheapen["measured_gb_s"]["production"] == pytest.approx(329.2)
    assert cheapen["measured_gb_s"]["fold_addqx"] == pytest.approx(370.9)

    lut = by["AUX_U8_LUT"]
    assert lut["stream_class"] == ee.STREAM_CLASS_BROADCAST_AUX
    assert lut["predicted_is_win"] is False
    assert lut["measured_is_win"] is False
    assert lut["must_predict_not_a_win"] is True
    assert lut["legacy_predicted_is_win"] is True
    assert lut["legacy_organ_average_ms_saved"] == pytest.approx(1.553, abs=1e-2)
    assert lut["measured_ms_delta"] > 0.0  # slower


def test_build_emits_sealed_static_only_receipt():
    path = ee.record()
    assert path.parent == RECEIPTS
    assert path.name == "EXECUTABLE_ECONOMICS.json"
    doc = json.loads(path.read_text())
    assert doc["schema"] == ee.SCHEMA
    assert doc["version"] == 2
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["seal_sha256"]
    assert doc["guards"]["bytes_removed_without_bytes_added_refused"] is True
    assert doc["guards"]["stream_class_undeclared_refused"] is True
    assert doc["guards"]["mlp_full_removal_scores_both_terms"] is True
    assert doc["guards"]["mlp_full_removal_flop_ms_at_100"] > 0.0
    assert doc["guards"]["aux_u8_lut_predicted_not_a_win"] is True
    assert doc["n_candidates"] == len(doc["candidates_ranked"])
    assert doc["n_live_material"] >= 8
    assert "quantize_aux_u8" in doc["live_material_ranked"]
    assert "group_size_1024" not in doc["live_material_ranked"]
    assert "entropy_coded_code_stream" in doc["live_material_ranked"]
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        assert key not in doc
    cited = doc["measured_constants_cited"]
    assert cited["affine_q2_saturated_gb_s"] == pytest.approx(376.7)
    assert cited["lm_head_gb_s"] == pytest.approx(497.4)
    # 0.528 was a fact about the 28.722 ms pre-widen_f4 anchor this file used to
    # hard-code. G131 promoted three levers and measured 21.9464 ms GPU, so the
    # reduction needed to reach 71 fell to 0.402. The INVARIANT is that it is
    # derived from the live baseline, not that it equals any particular number -
    # a pinned figure fails the moment the body gets faster, which punishes
    # exactly the progress this campaign is for.
    assert cited["gpu_reduction_for_71_at_current_executor"] == pytest.approx(
        1.0 - (1000.0 / 71.0 - ee.CITED_HOST_MS) / ee.CITED_GPU_MS, abs=1e-4
    )
    assert 0.0 < cited["gpu_reduction_for_71_at_current_executor"] < 1.0
    cal = doc["calibration"]
    assert cal["stream_classes"]["broadcast_aux"]["on_critical_path"] is False
    assert cal["stream_classes"]["weight_codes"]["on_critical_path"] is True
    assert cal["loadavg"]


def test_the_baseline_is_read_from_the_promoted_absolute_not_hard_coded():
    """28.722 was hard-coded through TWO rebases, so every predicted_token_ms in
    this gate was 6.8 ms stale and every predicted TPS with it."""
    import json as _j
    assert ee.CITED_BASIS == "GPU_SEALED_DEFAULT"
    m = _j.loads(
        (ee.REPO / "receipts/future/SEALED_DEFAULT_ABSOLUTE.json").read_text()
    )["measured"]
    assert ee.CITED_GPU_MS == m["gpu_ms_per_token"]
    assert ee.CITED_HOST_MS == m["host_gap_ms"]
    assert ee.CITED_TOKEN_MS == pytest.approx(
        ee.CITED_GPU_MS + ee.CITED_HOST_MS, abs=1e-9)
    src = Path(ee.__file__).read_text()
    assert "CITED_TOKEN_MS = 28.722" not in src


def test_calibration_receipt_is_a_real_gpu_run():
    path = RECEIPTS / ee.CALIBRATION_RECEIPT
    doc = json.loads(path.read_text())
    assert doc["schema"] == ee.CALIBRATION_SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["took_gpu_lease"] is True
    assert doc["metal_device"]
    assert "M3" in str(doc["metal_device"]) or "Apple" in str(doc["metal_device"])
    assert doc["loadavg"]
    assert doc["layer"] == 3
    classes = doc["stream_classes"]
    assert classes["broadcast_aux"]["on_critical_path"] is False
    assert classes["broadcast_aux"]["within_noise_at_50pct"] is True
    assert classes["weight_codes"]["on_critical_path"] is True
    assert classes["weight_codes"]["paired_dt_ns_at_50pct"] < 0
    assert classes["activation"]["probe_on_critical_path"] is True
    # Absolute GPU ns is under load; the verdict is the paired dt.
    assert doc["concurrent_load"]["loadavg"] == doc["loadavg"]


def test_every_ranked_row_carries_a_bandwidth_range():
    for row in ee.rank_recorded():
        lo, hi = row["assumptions"]["bandwidth_gb_s_range"]
        assert lo > 0.0 and hi > 0.0
        assert lo <= hi
        assert "bandwidth_regime" in row["assumptions"]
        assert "bandwidth_note" in row["assumptions"]
        dlo, dhi = row["predicted_ms_delta_range"]
        assert dlo <= dhi

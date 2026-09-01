"""Geometry table: the compiler consults it; an unmeasured shape is named."""
from __future__ import annotations

import json

import pytest

from tools.future import geometry_table as gt


WEIGHT = 27_852_800


def _launch(
    ident: str,
    *,
    gb_s: float,
    tg: int,
    rows_per_tg: int,
    weight_bytes: int = WEIGHT,
    packing: str | None = None,
    family: str = "launch",
) -> dict:
    gpu_ns = int(round(weight_bytes / gb_s)) if gb_s > 0 else 0
    tpr = tg // max(rows_per_tg, 1)
    return {
        "id": ident,
        "kernel": f"geo_table_{ident}",
        "family": family,
        "threads_per_threadgroup": tg,
        "rows_per_threadgroup": rows_per_tg,
        "threads_per_row": tpr,
        "stream_packing": packing,
        "weight_bytes": weight_bytes,
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [gpu_ns],
        "gpu_us_median": gpu_ns / 1e3,
        "effective_gb_s": gb_s,
        "bit_identical_vs_production_geo": ident == gt.PRODUCTION_GEO_ID,
    }


def _shape(
    organ: str,
    rows: int,
    cols: int,
    dtype: str,
    *,
    winner_gb: float = 330.0,
    runner_gb: float = 310.0,
    winner_id: str = "tg128_r2",
    winner_tg: int = 128,
    winner_rpt: int = 2,
    runner_id: str = "tg64_r2",
    runner_tg: int = 64,
    runner_rpt: int = 2,
    extra: list[dict] | None = None,
    packing: list[dict] | None = None,
    weight_bytes: int = WEIGHT,
) -> dict:
    launch = [
        _launch(winner_id, gb_s=winner_gb, tg=winner_tg, rows_per_tg=winner_rpt, weight_bytes=weight_bytes),
        _launch(runner_id, gb_s=runner_gb, tg=runner_tg, rows_per_tg=runner_rpt, weight_bytes=weight_bytes),
    ]
    if winner_id != gt.PRODUCTION_GEO_ID and runner_id != gt.PRODUCTION_GEO_ID:
        launch.append(
            _launch(
                gt.PRODUCTION_GEO_ID,
                gb_s=min(winner_gb, runner_gb) - 5.0,
                tg=128,
                rows_per_tg=2,
                weight_bytes=weight_bytes,
            )
        )
    if extra:
        launch.extend(extra)
    out = {
        "organ": organ,
        "rows": rows,
        "cols": cols,
        "dtype": dtype,
        "weight_bytes": weight_bytes,
        "launch": launch,
    }
    if packing is not None:
        out["packing"] = packing
    return out


def _hot_raw(*, vary: bool = True) -> dict:
    shapes = []
    for i, hot in enumerate(gt.HOT_DIMENSIONS):
        if vary:
            # Two col-widths pick different TGs so a non-flat table is representable.
            if hot["cols"] == 17408:
                winner_id, tg, rpt, wgb = "tg64_r2", 64, 2, 340.0
                runner_id, runner_tg, runner_rpt, rgb = "tg128_r2", 128, 2, 300.0
            elif hot["rows"] <= 128:
                winner_id, tg, rpt, wgb = "tg64_r1", 64, 1, 120.0
                runner_id, runner_tg, runner_rpt, rgb = "tg128_r2", 128, 2, 90.0
            else:
                winner_id, tg, rpt, wgb = "tg128_r2", 128, 2, 322.0
                runner_id, runner_tg, runner_rpt, rgb = "tg64_r2", 64, 2, 309.0
        else:
            winner_id, tg, rpt, wgb = "tg128_r2", 128, 2, 320.0
            runner_id, runner_tg, runner_rpt, rgb = "tg64_r2", 64, 2, 300.0
        packing = None
        if hot["dtype"].startswith("affine2"):
            packing = [
                _launch(
                    "mlp_2_2_2_32",
                    gb_s=308.3,
                    tg=128,
                    rows_per_tg=2,
                    packing="mlp_2_2_2_32",
                    family="packing",
                    weight_bytes=hot["rows"] * hot["cols"],  # unused scale; overwritten
                ),
                _launch(
                    "mid_2_4_32",
                    gb_s=526.6,
                    tg=128,
                    rows_per_tg=2,
                    packing="mid_2_4_32",
                    family="packing",
                ),
            ]
        shapes.append(
            _shape(
                hot["organ"],
                hot["rows"],
                hot["cols"],
                hot["dtype"],
                winner_gb=wgb,
                runner_gb=rgb,
                winner_id=winner_id,
                winner_tg=tg,
                winner_rpt=rpt,
                runner_id=runner_id,
                runner_tg=runner_tg,
                runner_rpt=runner_rpt,
                packing=packing,
            )
        )
    return {
        "schema": "hawking.future.geometry_table.raw.v1",
        "warmup": 3,
        "reps": 7,
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "git_head": "fixture",
        "artifact_root": "/Users/scammermike/noetic/NOETIC_PARENT_A",
        "measured_at": "2026-08-31T12:00:00Z",
        "gpu_lane_lock_held": True,
        "concurrent_load": {"loadavg": "{ 1.00 1.00 1.00 }"},
        "shapes": shapes,
        "does_not_edit_production_shaders": True,
    }


def test_every_hot_dimension_names_provenance():
    assert len(gt.HOT_DIMENSIONS) >= 8
    organs = {h["organ"] for h in gt.HOT_DIMENSIONS}
    for required in (
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "linear_attn.in_proj_qkvz",
        "linear_attn.in_proj_ba",
        "linear_attn.out_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "lm_head",
    ):
        assert required in organs
    for hot in gt.HOT_DIMENSIONS:
        assert hot["rows"] > 0 and hot["cols"] > 0
        assert hot["provenance"], hot["organ"]
        assert any("qwen38_geometry.rs" in p or "KERNEL_GEOMETRY.json" in p for p in hot["provenance"])
        assert hot["cols"] in (5120, 6144, 17408) or hot["organ"] == "lm_head"


def test_synthetic_grid_point_is_refused():
    with pytest.raises(gt.NotAHotDimension):
        gt.assert_hot_dimension(7, 13, "affine2_q2", "probe.synthetic")
    raw = _hot_raw()
    raw["shapes"].append(
        _shape("probe.synthetic", 7, 13, "affine2_q2", winner_gb=10.0, runner_gb=9.0)
    )
    with pytest.raises(gt.NotAHotDimension):
        gt.measurement_from_raw(raw)


def test_absent_shape_is_named_unmeasured_not_silent_tpr64():
    measured = gt.measurement_from_raw(_hot_raw())
    doc = gt.build(measured)
    decision = gt.consult(1, 1, "affine2_q2", "not.an.organ", table=doc)
    assert decision["status"] == gt.UNMEASURED_SHAPE
    assert decision["fallback"] == gt.UNMEASURED_SHAPE
    assert decision["winner"] is None
    assert decision["geometry"] is None
    assert "geo_tpr64_tg128" not in str(decision.get("winner"))
    assert decision["production_incumbent_cited_not_selected"] == "geo_tpr64_tg128"
    assert "silent" in decision["why"].lower() or "unmeasured" in decision["why"].lower()


def test_consult_refuses_a_default_argument():
    measured = gt.measurement_from_raw(_hot_raw())
    doc = gt.build(measured)
    with pytest.raises(gt.SilentDefaultRefused):
        gt.consult(1, 1, "affine2_q2", "missing", table=doc, default="geo_tpr64_tg128")


def test_planner_alias_hits_a_measured_shape():
    measured = gt.measurement_from_raw(_hot_raw())
    doc = gt.build(measured)
    hit = gt.planner_geometry(17408, 5120, "affine2_q2", "mlp.gate_proj", table=doc)
    assert hit["status"] == gt.STATUS_HIT
    assert hit["winner"]["threads_per_threadgroup"] == 128
    assert hit["runner_up"]["effective_gb_s"] > 0
    assert hit["fallback"] is None


def test_winner_and_runner_up_required():
    raw = _hot_raw()
    raw["shapes"][0]["launch"] = [raw["shapes"][0]["launch"][0]]
    with pytest.raises(gt.MissingSweep, match="runner-up"):
        gt.measurement_from_raw(raw)


def test_refuted_accumulator_chain_is_not_rerun():
    raw = _hot_raw()
    raw["shapes"][0]["launch"].append(
        _launch("ilp8", gb_s=327.4, tg=128, rows_per_tg=2, family="ilp")
    )
    with pytest.raises(gt.RefutedDiscriminatorRerun, match="discriminator"):
        gt.measurement_from_raw(raw)


def test_refuted_working_set_is_not_rerun():
    raw = _hot_raw()
    raw["shapes"][0]["ilp"] = []
    raw["register_pressure"] = [
        _launch("ws32", gb_s=332.4, tg=128, rows_per_tg=2, family="working_set")
    ]
    with pytest.raises(gt.RefutedDiscriminatorRerun):
        gt.measurement_from_raw(raw)


def test_pack_38_merge_is_not_rerun():
    raw = _hot_raw()
    raw["shapes"][0]["packing"] = [
        _launch("mlp_2_2_2_32", gb_s=308.3, tg=128, rows_per_tg=2, packing="mlp_2_2_2_32", family="packing"),
        _launch("pack_38", gb_s=45.6, tg=128, rows_per_tg=2, packing="pack_38", family="pack_38"),
    ]
    with pytest.raises(gt.RefutedDiscriminatorRerun):
        gt.measurement_from_raw(raw)


def test_empty_gpu_sample_is_refused():
    with pytest.raises(gt.EmptyGpuSample):
        gt.effective_gb_s(100, 0)


def test_flat_table_is_reported_as_flat():
    measured = gt.measurement_from_raw(_hot_raw(vary=False))
    assert measured["flat"]["flat"] is True
    doc = gt.build(measured)
    assert doc["geometry_is_a_constant_for_this_resident"] is True
    assert doc["verdict"] == "GEOMETRY_IS_A_CONSTANT_FOR_THIS_RESIDENT"
    assert "answered in the negative" in doc["finding"]


def test_nonflat_table_is_a_table():
    measured = gt.measurement_from_raw(_hot_raw(vary=True))
    assert measured["flat"]["flat"] is False
    doc = gt.build(measured)
    assert doc["verdict"] == "GEOMETRY_IS_A_TABLE"
    assert doc["geometry_is_a_constant_for_this_resident"] is False
    down = gt.consult(5120, 17408, "affine2_q2", "mlp.down_proj", table=doc)
    gate = gt.consult(17408, 5120, "affine2_q2", "mlp.gate_proj", table=doc)
    assert down["winner"]["threads_per_threadgroup"] != gate["winner"]["threads_per_threadgroup"]


def test_each_row_carries_winner_runner_up_and_gbs():
    measured = gt.measurement_from_raw(_hot_raw())
    for row in measured["shapes"]:
        assert row["winner"]["effective_gb_s"] > 0
        assert row["runner_up"]["effective_gb_s"] > 0
        assert row["winner"]["gpu_ns_median"] > 0
        assert row["runner_up"]["id"] != row["winner"]["id"] or (
            row["winner"]["effective_gb_s"] == row["runner_up"]["effective_gb_s"]
        )
        assert row["provenance"]
        assert "s016_projection" in row
        assert "token_ms_projected" in row["s016_projection"]


def test_s016_ranks_token_ns_above_bandwidth():
    measured = gt.measurement_from_raw(_hot_raw())
    doc = gt.build(measured)
    above = " ".join(doc["s016"]["above_bandwidth"]).lower()
    assert "token_ns" in above
    assert "accepted tps" in above
    assert "capability" in above
    assert "executable cost" in above
    assert "slower tokens" in doc["s016"]["not_the_objective"][1]


def test_specialized_cols_are_the_two_mlp_widths():
    assert list(gt.SPECIALIZED_COLS) == [5120, 17408]
    down = next(h for h in gt.HOT_DIMENSIONS if h["organ"] == "mlp.down_proj")
    gate = next(h for h in gt.HOT_DIMENSIONS if h["organ"] == "mlp.gate_proj")
    assert down["cols"] == 17408
    assert gate["cols"] == 5120


def test_ba_is_the_occupancy_starved_hot_dimension():
    ba = next(h for h in gt.HOT_DIMENSIONS if h["organ"] == "linear_attn.in_proj_ba")
    assert ba["rows"] == 96
    assert any("occupancy" in p.lower() for p in ba["provenance"])


def test_receipt_on_disk_if_present_matches_the_contract():
    if not gt.RECEIPT.is_file():
        pytest.skip("GEOMETRY_TABLE.json not written yet")
    doc = json.loads(gt.RECEIPT.read_text())
    assert doc["schema"] == gt.SCHEMA
    assert "table" in doc
    assert doc["fallback_for_absent_shape"] == gt.UNMEASURED_SHAPE
    assert doc["planner_entry"] == "tools.future.geometry_table.consult"
    assert doc["does_not_edit_production_shaders"] is True
    assert "accumulator_chain" in doc["refuted_discriminators"]
    assert "working_set" in doc["refuted_discriminators"]
    assert "stream_merge_beyond_mid_2_4_32" in doc["refuted_discriminators"]
    for key, row in doc["table"].items():
        k = row["key"]
        gt.assert_hot_dimension(k["rows"], k["cols"], k["dtype"], k["organ"])
        assert row["winner"]["effective_gb_s"] > 0
        assert row["runner_up"]["effective_gb_s"] > 0
        assert row["provenance"]
        # Hardware numbers must be placeable in time.
        assert "measurement_provenance" in doc
        # Consult the on-disk table the way the planner will.
        hit = gt.consult(k["rows"], k["cols"], k["dtype"], k["organ"], table=doc)
        assert hit["status"] == gt.STATUS_HIT
        assert hit["winner"]["id"] == row["winner"]["id"]
    miss = gt.consult(3, 5, "affine2_q2", "absent.organ", table=doc)
    assert miss["status"] == gt.UNMEASURED_SHAPE
    assert miss["fallback"] == gt.UNMEASURED_SHAPE
    assert miss["geometry"] is None
    if doc.get("geometry_is_a_constant_for_this_resident"):
        assert "negative" in doc["finding"].lower() or "constant" in doc["finding"].lower()
    assert doc["s016"]["primary_objective"]
    # Provenance: lock + time + load.
    prov = doc["measurement_provenance"]
    assert "measured_at" in prov
    assert "gpu_lane_lock_held" in prov
    assert "loadavg" in prov


def test_the_body_is_read_from_the_promoted_absolute_not_hard_coded():
    """28.722 was the pre-widen_f4 anchor and survived two rebases. A geometry
    table priced against a body that no longer runs ranks shapes by a stale
    share."""
    import json
    assert gt.CITED_BODY_BASIS == "GPU_SEALED_DEFAULT"
    m = json.loads(
        (gt.REPO / "receipts/future/SEALED_DEFAULT_ABSOLUTE.json").read_text()
    )["measured"]
    assert gt.TOKEN_MS == m["gpu_ms_per_token"]
    assert gt.TOKEN_TPS == m["gpu_tps"]


def test_the_organ_figures_come_from_the_sealed_decomposition():
    import json
    rows = {r["organ"]: float(r["sealed_ms"]) for r in json.loads(
        (gt.REPO / "receipts/future/ORGAN_DECOMPOSITION_SEALED.json").read_text()
    )["table"]}
    assert gt.MLP_MS == rows["mlp_gate_up"] + rows["mlp_down"]
    assert gt.DELTANET_MS == rows["deltanet"]
    assert gt.LM_HEAD_MS == rows["lm_head"]


def test_the_cited_organs_fit_inside_the_cited_token():
    """The pre-promotion set did not: its organs summed to 26.7013 against a
    21.9464 body."""
    total = gt.MLP_MS + gt.DELTANET_MS + gt.GQA_MS + gt.LM_HEAD_MS
    assert total < gt.TOKEN_MS


def test_the_constant_cannot_creep_back():
    from pathlib import Path
    src = Path(gt.__file__).read_text()
    assert "TOKEN_MS = 28.722" not in src

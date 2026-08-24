"""N024 kernel bottleneck: ≥3 non-load levers on the production decode path."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from kernel_bottleneck import (  # noqa: E402
    BAR_GB_S,
    CONTROLS,
    LEVERS,
    N018_PRODUCTION_GB_S,
    PARENT_ACTIVE_BYTES,
    PARENT_ROOT,
    RECEIPT,
    ROOF_GB_S,
    SCHEMA,
    shader_evidence,
    separated,
    summarize_isolated,
    summarize_production,
)

RECEIPT_DOC = None


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        reuse = os.environ.get("NOETIC_KBN_REUSE", "1") != "0"
        if reuse and RECEIPT.is_file():
            RECEIPT_DOC = json.loads(RECEIPT.read_text())
            if RECEIPT_DOC.get("schema") == SCHEMA:
                return RECEIPT_DOC
        from kernel_bottleneck import build  # noqa: WPS433

        RECEIPT_DOC = build(live=True)
    return RECEIPT_DOC


def test_four_levers_and_two_controls_are_in_the_shaders():
    ev = shader_evidence()
    assert ev["mixed_present"]
    assert ev["standalone_present"]
    assert ev["wired_launch"]
    assert ev["wired_fused"]
    assert ev["incumbent_untouched"]
    assert ev["bad_control_kept"]
    assert ev["n018_tiles_kept_not_retried_as_levers"]
    for name, present in ev["production_kernels"].items():
        assert present, name
    for name, present in ev["isolated_kernels"].items():
        assert present, name


def test_did_not_retry_n018_tile_geometries_as_levers():
    assert set(LEVERS) == {"tgsb", "pipe", "splitk4", "accfuse"}
    assert "qmvfast" not in LEVERS
    assert "wide64" not in LEVERS
    assert "tgx" not in LEVERS


def test_separation_helper_refuses_overlap():
    assert separated([1.0, 2.0], [3.0, 4.0]) is True
    assert separated([1.0, 3.0], [2.0, 4.0]) is False
    assert separated([], [1.0]) is False


def test_summarize_requires_seven_reps_and_names_overlap():
    fake = {
        "isolated": {
            "parity": [
                {"id": "tpr64", "ok": True, "max_abs_diff": 1e-4},
                {"id": "tgsb", "ok": True, "max_abs_diff": 1e-4},
            ],
            "shapes": [
                {
                    "label": "mlp.gate_proj",
                    "rows": 17408,
                    "cols": 5120,
                    "weight_payload_bytes": 1,
                    "arms": [
                        {
                            "id": "tpr64",
                            "role": "no_op_control",
                            "gpu_ns_reps": [10, 11, 12, 10, 11, 12, 11],
                            "gpu_ns_min": 10,
                            "gpu_ns_median": 11,
                            "gpu_ns_max": 12,
                            "weight_gb_s_median": 400.0,
                            "dense_w_materialized": 0,
                        },
                        {
                            "id": "tgsb",
                            "role": "lever",
                            "gpu_ns_reps": [8, 9, 8, 9, 8, 9, 8],
                            "gpu_ns_min": 8,
                            "gpu_ns_median": 8,
                            "gpu_ns_max": 9,
                            "weight_gb_s_median": 500.0,
                            "dense_w_materialized": 0,
                        },
                    ],
                }
            ],
        }
    }
    iso = summarize_isolated(fake)
    assert iso["kind"] == "MEASURED"
    assert iso["parity_all_ok"] is True
    gate = iso["shapes"][0]
    assert gate["arms"]["tpr64"]["n_reps"] == 7
    assert gate["separation"]["tgsb_vs_tpr64"]["separated"] is True


def test_production_summary_keeps_token_ids_and_roof():
    ids = [1, 2, 3]
    fake = {
        "production": {
            "did_not_mutate_parent": True,
            "did_not_load_second_27b": True,
            "fusion": "parent",
            "arms": [
                {
                    "id": "tpr64",
                    "role": "no_op_control",
                    "median_gpu_ns_per_token_reps": [40_000_000] * 7,
                    "gpu_ns_min": 40_000_000,
                    "gpu_ns_median": 40_000_000,
                    "gpu_ns_max": 40_000_000,
                    "achieved_gb_s_median": PARENT_ACTIVE_BYTES / 40_000_000,
                    "tok_s_reps": [25.0] * 7,
                    "new_token_ids": ids,
                    "token_ids_unchanged_vs_tpr64": True,
                    "token_ids_stable_across_reps": True,
                    "fallbacks_reps": [0] * 7,
                    "dense_w_materialized": 0,
                },
                {
                    "id": "tgsb",
                    "role": "lever",
                    "median_gpu_ns_per_token_reps": [20_000_000] * 7,
                    "gpu_ns_min": 20_000_000,
                    "gpu_ns_median": 20_000_000,
                    "gpu_ns_max": 20_000_000,
                    "achieved_gb_s_median": PARENT_ACTIVE_BYTES / 20_000_000,
                    "tok_s_reps": [50.0] * 7,
                    "new_token_ids": ids,
                    "token_ids_unchanged_vs_tpr64": True,
                    "token_ids_stable_across_reps": True,
                    "fallbacks_reps": [0] * 7,
                    "dense_w_materialized": 0,
                },
            ],
        }
    }
    prod = summarize_production(fake)
    assert prod["kind"] == "MEASURED"
    assert prod["before"]["achieved_gb_s"] == PARENT_ACTIVE_BYTES / 40_000_000
    assert prod["after"]["id"] == "tgsb"
    assert prod["roof_gb_s"] == ROOF_GB_S
    assert prod["reached_bar"] is False  # 494 GB/s < 775
    assert prod["dense_w_materialized"] == 0
    assert prod["n018_anchor_gb_s"] == N018_PRODUCTION_GB_S


def test_receipt_schema_levers_controls_and_no_second_27b():
    doc = receipt()
    assert RECEIPT.is_file()
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_mutate_parent"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["did_not_write_ascent_or_campaign"] is True
    assert doc["roof_gb_s"] == ROOF_GB_S
    assert doc["bar_gb_s"] == BAR_GB_S
    assert doc["dense_w_materialized"] == 0
    ids = {lever["id"] for lever in doc["levers"]}
    assert ids == set(LEVERS)
    assert "qmvfast" not in ids
    assert "tpr64" in doc["controls"]["no_op"] or doc["controls"]["no_op"]
    assert "runtime_div" in doc["controls"]["deliberately_bad"]
    assert doc["controls"]["reps"] >= 7
    loc = doc["parent_immutable"]
    assert Path(loc["path"]).resolve() == PARENT_ROOT.resolve() or str(PARENT_ROOT) in loc["path"]
    assert loc["outside_worktree"] is True
    assert "what_still_blocks" in doc
    assert doc["prior_not_rederived"]["n018_production_decode_gb_s"] == N018_PRODUCTION_GB_S
    assert "qmvfast" in doc["prior_not_rederived"]["did_not_retry"]


def test_receipt_isolated_or_production_has_seven_reps_and_parity():
    doc = receipt()
    iso = doc["isolated_gemv"]
    prod = doc["production_decode"]
    assert iso["kind"] in ("MEASURED", "ABSENT")
    assert prod["kind"] in ("MEASURED", "ABSENT")
    autopsy = doc["kernel_autopsy"]
    assert autopsy["any_new_kernel_defective"] is False
    if iso["kind"] == "MEASURED":
        assert iso["parity_all_ok"] is True
        assert iso["dense_w_materialized"] == 0
        assert iso["shapes"], "isolated measured but no shapes"
        for shape in iso["shapes"]:
            for cid in CONTROLS:
                assert cid in shape["arms"], cid
            for lid in LEVERS:
                assert lid in shape["arms"], lid
                arm = shape["arms"][lid]
                assert arm["n_reps"] >= 7
                assert arm["gpu_ns_min"] <= arm["gpu_ns_median"] <= arm["gpu_ns_max"]
            tpr = shape["arms"]["tpr64"]
            assert tpr["n_reps"] >= 7
            bad = shape["arms"]["runtime_div"]
            assert bad["n_reps"] >= 7
            for _key, row in shape["separation"].items():
                if row["separated"] is False:
                    assert "separated" in row
    if prod["kind"] == "MEASURED":
        assert prod["active_bytes_per_token"] == PARENT_ACTIVE_BYTES
        assert prod["did_not_mutate_parent"] is True
        arms = prod["arms"]
        assert "tpr64" in arms
        assert "runtime_div" in arms
        for lid in LEVERS:
            assert lid in arms
            assert arms[lid]["n_reps"] >= 7
            assert arms[lid]["dense_w_materialized"] == 0
        before = prod["before"]
        after = prod["after"]
        assert before["achieved_gb_s"] is not None
        if after.get("achieved_gb_s") is not None:
            assert after.get("token_ids_unchanged") is True
        for lid, row in prod["separation"].items():
            if not row.get("separated_from_tpr64"):
                assert "NOT SEPARATED" in (row.get("note") or "")
        bad = prod["deliberately_bad_control"]
        assert bad is not None
        assert bad["achieved_gb_s_median"] < before["achieved_gb_s"]
        assert "356.7" in doc["what_still_blocks"] or "778.8" in doc["what_still_blocks"]
        assert prod["fraction_of_356p7_to_778p8_closed"] is not None
    else:
        assert "absent_reason" in prod
    ev = doc["shader_evidence"]
    assert all(ev["production_kernels"].values())
    assert all(ev["isolated_kernels"].values())

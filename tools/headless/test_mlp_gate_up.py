"""N030 MLP_GATE_UP: non-load fused gate_up_swiglu autopsy."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from mlp_gate_up import (  # noqa: E402
    BAD,
    BAD_PROD,
    LEVER,
    N018_TILES_NOT_RETRIED,
    N024_NOT_RETRIED,
    N025_MLP_GATE_UP_GAP_SHARE,
    NO_OP,
    PARENT_ACTIVE_BYTES,
    PARENT_ROOT,
    RECEIPT,
    ROOF_GB_S,
    SCHEMA,
    separated,
    shader_evidence,
    summarize_isolated,
    summarize_production,
)

RECEIPT_DOC = None


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        reuse = os.environ.get("NOETIC_MLP_GATE_UP_REUSE", "1") != "0"
        if reuse and RECEIPT.is_file():
            RECEIPT_DOC = json.loads(RECEIPT.read_text())
            if RECEIPT_DOC.get("schema") == SCHEMA:
                return RECEIPT_DOC
        from mlp_gate_up import build, write_receipt  # noqa: WPS433

        RECEIPT_DOC = build(live=True)
        write_receipt(RECEIPT_DOC)
    return RECEIPT_DOC


def test_did_not_retune_n018_tiles_or_retry_n024_levers():
    assert LEVER == "biasprep"
    assert LEVER not in N018_TILES_NOT_RETRIED
    assert LEVER not in N024_NOT_RETRIED
    assert "qmvfast" not in (LEVER, NO_OP, BAD)
    ev = shader_evidence()
    assert ev["n018_tiles_kept_not_retried_as_levers"]
    assert ev["n024_levers_kept_not_retried"]
    assert ev["wired_biasprep"]
    assert ev["wired_drop"]
    assert ev["incumbent_untouched"]
    assert ev["same_tpr64_occupancy"]
    for name, present in ev["production_kernels"].items():
        assert present, name
    for name, present in ev["isolated_kernels"].items():
        assert present, name


def test_separation_helper_refuses_overlap():
    assert separated([1.0, 2.0], [3.0, 4.0]) is True
    assert separated([1.0, 3.0], [2.0, 4.0]) is False
    assert separated([], [1.0]) is False


def test_summarize_isolated_requires_seven_reps_and_names_overlap():
    fake = {
        "isolated": {
            "parity": [
                {"id": "tpr64", "ok": True, "max_abs_diff": 1e-4},
                {"id": "biasprep", "ok": True, "max_abs_diff": 1e-4},
                {"id": "dropbias", "ok": False, "rejected": True, "must_fail": True},
            ],
            "shape": {"rows": 17408, "cols": 5120, "label": "mlp.gate_up_swiglu"},
            "arms": [
                {
                    "id": "tpr64",
                    "role": "no_op_control",
                    "gpu_ns_reps": [10, 11, 12, 10, 11, 12, 11],
                    "gpu_ns_min": 10,
                    "gpu_ns_median": 11,
                    "gpu_ns_max": 12,
                    "dense_w_materialized": 0,
                },
                {
                    "id": "biasprep",
                    "role": "lever",
                    "gpu_ns_reps": [8, 9, 8, 9, 8, 9, 8],
                    "gpu_ns_min": 8,
                    "gpu_ns_median": 8,
                    "gpu_ns_max": 9,
                    "dense_w_materialized": 0,
                },
                {
                    "id": "acc_only",
                    "role": "component_accumulate",
                    "gpu_ns_reps": [9] * 7,
                    "gpu_ns_median": 9,
                    "dense_w_materialized": 0,
                },
                {
                    "id": "decode_probe",
                    "role": "component_packed_decode",
                    "gpu_ns_reps": [4] * 7,
                    "gpu_ns_median": 4,
                    "dense_w_materialized": 0,
                },
                {
                    "id": "addr_probe",
                    "role": "component_load_only",
                    "gpu_ns_reps": [3] * 7,
                    "gpu_ns_median": 3,
                    "dense_w_materialized": 0,
                },
                {
                    "id": "launch64",
                    "role": "component_64_launches",
                    "gpu_ns_reps": [700] * 7,
                    "gpu_ns_median": 700,
                    "dense_w_materialized": 0,
                },
                {
                    "id": "xsum64",
                    "role": "component_xsum",
                    "gpu_ns_reps": [1] * 7,
                    "gpu_ns_median": 1,
                    "dense_w_materialized": 0,
                },
            ],
        }
    }
    iso = summarize_isolated(fake)
    assert iso["kind"] == "MEASURED"
    assert iso["parity_all_ok"] is True
    assert iso["dropbias_rejected"] is True
    assert iso["arms"]["tpr64"]["n_reps"] == 7
    assert iso["separation"]["biasprep_vs_tpr64"]["separated"] is True
    assert iso["non_load_profile"]["fused_swiglu_gpu_ns_median"] == 11
    overlap = {
        "isolated": {
            "parity": [{"id": "tpr64", "ok": True}],
            "arms": [
                {
                    "id": "tpr64",
                    "gpu_ns_reps": [10, 12, 11, 10, 12, 11, 10],
                    "dense_w_materialized": 0,
                },
                {
                    "id": "biasprep",
                    "gpu_ns_reps": [11, 13, 12, 11, 13, 12, 11],
                    "dense_w_materialized": 0,
                },
            ],
        }
    }
    ov = summarize_isolated(overlap)
    assert ov["separation"]["biasprep_vs_tpr64"]["separated"] is False
    assert "NOT SEPARATED" in ov["separation"]["biasprep_vs_tpr64"]["note"]


def test_summarize_production_ranks_complete_token_ns_not_gbs():
    ids = [1, 2, 3]
    fake = {
        "production": {
            "did_not_mutate_parent": True,
            "did_not_load_second_27b": True,
            "fusion": "580",
            "expected_dispatches": 580,
            "arms": [
                {
                    "id": "tpr64",
                    "role": "no_op_control",
                    "complete_token_ns_reps": [30_000_000] * 7,
                    "complete_token_ns_median": 30_000_000,
                    "median_gpu_ns_per_token_reps": [28_000_000] * 7,
                    "tok_s_reps": [33.0] * 7,
                    "new_token_ids": ids,
                    "token_ids_unchanged_vs_tpr64": True,
                    "token_ids_stable_across_reps": True,
                    "fallbacks_reps": [0] * 7,
                    "dense_w_materialized": 0,
                },
                {
                    "id": "biasprep",
                    "role": "lever",
                    "complete_token_ns_reps": [28_000_000] * 7,
                    "complete_token_ns_median": 28_000_000,
                    "median_gpu_ns_per_token_reps": [27_000_000] * 7,
                    "tok_s_reps": [35.0] * 7,
                    "new_token_ids": ids,
                    "token_ids_unchanged_vs_tpr64": True,
                    "token_ids_stable_across_reps": True,
                    "fallbacks_reps": [0] * 7,
                    "dense_w_materialized": 0,
                },
                {
                    "id": "biasprep_drop",
                    "role": "deliberately_bad_control",
                    "complete_token_ns_reps": [40_000_000] * 7,
                    "complete_token_ns_median": 40_000_000,
                    "new_token_ids": [9, 9, 9],
                    "token_ids_unchanged_vs_tpr64": False,
                    "fallbacks_reps": [0] * 7,
                    "dense_w_materialized": 0,
                },
            ],
        }
    }
    prod = summarize_production(fake)
    assert prod["kind"] == "MEASURED"
    assert prod["ranking_metric"] == "COMPLETE_TOKEN_NS"
    assert prod["after"]["id"] == "biasprep"
    assert prod["after"]["kept_incumbent_tpr64"] is False
    assert prod["deliberately_bad_control"]["rejected"] is True
    overlap = {
        "production": {
            "arms": [
                {
                    "id": "tpr64",
                    "complete_token_ns_reps": [30e6, 31e6, 29e6, 30e6, 31e6, 29e6, 30e6],
                    "complete_token_ns_median": 30e6,
                    "new_token_ids": ids,
                    "token_ids_unchanged_vs_tpr64": True,
                    "fallbacks_reps": [0] * 7,
                    "dense_w_materialized": 0,
                },
                {
                    "id": "biasprep",
                    "complete_token_ns_reps": [29e6, 32e6, 30e6, 29e6, 32e6, 30e6, 29e6],
                    "complete_token_ns_median": 30e6,
                    "new_token_ids": ids,
                    "token_ids_unchanged_vs_tpr64": True,
                    "fallbacks_reps": [0] * 7,
                    "dense_w_materialized": 0,
                },
                {
                    "id": "biasprep_drop",
                    "complete_token_ns_reps": [40e6] * 7,
                    "token_ids_unchanged_vs_tpr64": False,
                    "dense_w_materialized": 0,
                },
            ]
        }
    }
    ov = summarize_production(overlap)
    assert ov["after"]["kept_incumbent_tpr64"] is True
    assert "NOT SEPARATED" in (ov["separation"]["biasprep_vs_tpr64"]["note"] or "")


def test_receipt_schema_controls_and_no_second_27b():
    doc = receipt()
    assert RECEIPT.is_file()
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_mutate_parent"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["did_not_write_ascent_or_campaign"] is True
    assert doc["ranking_metric"] == "COMPLETE_TOKEN_NS"
    assert doc["roof_gb_s"] == ROOF_GB_S
    assert doc["n025_mlp_gate_up_gap_share"] == N025_MLP_GATE_UP_GAP_SHARE
    assert doc["dense_w_materialized"] == 0
    assert doc["lever"]["id"] == LEVER
    assert doc["lever"]["not_load_geometry"] is True
    assert doc["controls"]["no_op"] == NO_OP
    assert BAD in doc["controls"]["deliberately_bad"] or BAD_PROD in doc["controls"]["deliberately_bad"]
    assert doc["controls"]["reps"] >= 7
    loc = doc["parent_immutable"]
    assert Path(loc["path"]).resolve() == PARENT_ROOT.resolve() or str(PARENT_ROOT) in loc["path"]
    assert loc["outside_worktree"] is True
    prior = doc["prior_not_rederived"]
    assert prior["n024_tile_ruled_out"] is True
    assert prior["did_not_retry_n018_tiles"] == list(N018_TILES_NOT_RETRIED)
    assert prior["did_not_retry_n024_levers"] == list(N024_NOT_RETRIED)


def test_receipt_isolated_or_production_has_seven_reps_parity_and_autopsy():
    doc = receipt()
    iso = doc["isolated_gemv"]
    prod = doc["production_decode"]
    assert iso["kind"] in ("MEASURED", "ABSENT")
    assert prod["kind"] in ("MEASURED", "ABSENT")
    autopsy = doc["kernel_autopsy"]
    assert autopsy["any_new_kernel_defective"] is False
    if iso["kind"] == "MEASURED":
        assert iso["parity_all_ok"] is True
        assert iso["dropbias_rejected"] is True
        assert iso["dense_w_materialized"] == 0
        assert iso["did_not_load_second_27b"] is True
        for cid in (NO_OP, LEVER, BAD, "acc_only", "decode_probe", "addr_probe", "launch64"):
            assert cid in iso["arms"], cid
            arm = iso["arms"][cid]
            assert arm["n_reps"] >= 7
            if arm.get("gpu_ns_min") is not None:
                assert arm["gpu_ns_min"] <= arm["gpu_ns_median"] <= arm["gpu_ns_max"]
        prof = iso["non_load_profile"]
        assert "fused_swiglu_gpu_ns_median" in prof
        assert "packed_decode_gpu_ns_median" in prof
        assert "launch64_gpu_ns_median" in prof
        sep = iso["separation"]["biasprep_vs_tpr64"]
        if not sep["separated"]:
            assert "NOT SEPARATED" in (sep.get("note") or "")
    if prod["kind"] == "MEASURED":
        assert prod["ranking_metric"] == "COMPLETE_TOKEN_NS"
        assert prod["active_bytes_per_token"] == PARENT_ACTIVE_BYTES
        assert prod["did_not_mutate_parent"] is True
        arms = prod["arms"]
        assert NO_OP in arms
        assert LEVER in arms
        assert BAD_PROD in arms or BAD in arms
        for lid in (NO_OP, LEVER):
            assert arms[lid]["n_reps"] >= 7
            assert arms[lid]["dense_w_materialized"] == 0
        bad_arm = arms.get(BAD_PROD) or arms.get(BAD)
        assert bad_arm is not None
        assert bad_arm["n_reps"] >= 7
        assert bad_arm["dense_w_materialized"] == 0
        before = prod["before"]
        after = prod["after"]
        assert before["complete_token_ns_median"] is not None
        if not after.get("kept_incumbent_tpr64"):
            assert after.get("token_ids_unchanged") is True
            assert after.get("complete_token_ns_median") <= before["complete_token_ns_median"]
        sep = prod["separation"]["biasprep_vs_tpr64"]
        if not sep.get("separated"):
            assert "NOT SEPARATED" in (sep.get("note") or "")
        bad = prod["deliberately_bad_control"]
        assert bad["rejected"] is True
    else:
        assert "absent_reason" in prod
    ev = doc["shader_evidence"]
    assert all(ev["production_kernels"].values())
    assert all(ev["isolated_kernels"].values())
    assert "COMPLETE_TOKEN_NS" in doc["answer"] or "complete_token_ns" in doc["answer"]
    assert "N024" in doc["answer"] or "tile" in doc["answer"]

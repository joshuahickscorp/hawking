"""N025 organ bandwidth: per-organ GPU ns/GB/s and a dispatch cut below 628."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from organ_bandwidth import (  # noqa: E402
    BASELINE_DISPATCHES,
    CANDIDATE_DISPATCHES,
    GAP_GB_S,
    KERNEL_BAD,
    KERNEL_GOOD,
    N018_PRODUCTION_GB_S,
    ORGANS,
    PARENT_ACTIVE_BYTES,
    PARENT_ROOT,
    RECEIPT,
    ROOF_GB_S,
    SCHEMA,
    map_ledger_row,
    organ_bytes_from_ledger,
    separated,
    shader_evidence,
)

RECEIPT_DOC = None


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        reuse = os.environ.get("NOETIC_ORGAN_REUSE", "1") != "0"
        if reuse and RECEIPT.is_file():
            RECEIPT_DOC = json.loads(RECEIPT.read_text())
            if RECEIPT_DOC.get("schema") == SCHEMA:
                return RECEIPT_DOC
        from organ_bandwidth import build, write_receipt  # noqa: WPS433

        RECEIPT_DOC = build(live=True)
        write_receipt(RECEIPT_DOC)
    return RECEIPT_DOC


def test_separation_helper_refuses_overlap():
    assert separated([1.0, 2.0], [3.0, 4.0]) is True
    assert separated([1.0, 3.0], [2.0, 4.0]) is False
    assert separated([], [1.0]) is False


def test_ledger_rows_cover_the_eight_organs_disjointly():
    bytes_by = organ_bytes_from_ledger()
    assert set(bytes_by) == set(ORGANS)
    n = sum(b["n"] for b in bytes_by.values())
    assert n == 756, n
    assert bytes_by["q4_remainder"]["n"] == 64  # 48 DN out_proj + 16 GQA o_proj
    assert bytes_by["mlp_gate_up"]["n"] == 128  # 64 swiglu + 64 post rms
    assert bytes_by["mlp_down"]["n"] == 128  # 64 down + 64 residual
    w = sum(b["weight_read"] for b in bytes_by.values())
    assert w > 9_000_000_000
    # MLP is affine2; Q4 remainder is the leftover geo_tpr64 GEMVs.
    assert bytes_by["q4_remainder"]["weight_read"] > 1_000_000_000
    assert bytes_by["mlp_gate_up"]["weight_read"] > bytes_by["q4_remainder"]["weight_read"]


def test_map_row_q4_remainder_is_out_proj_only():
    assert (
        map_ledger_row({"organ": "linear_attn.out_proj", "mixer": "deltanet"})
        == "q4_remainder"
    )
    assert map_ledger_row({"organ": "self_attn.o_proj", "mixer": "gqa"}) == "q4_remainder"
    assert map_ledger_row({"organ": "self_attn.qkv", "mixer": "gqa"}) == "gqa_attention"
    assert map_ledger_row({"organ": "linear_attn.in_proj", "mixer": "deltanet"}) == "deltanet"
    assert map_ledger_row({"organ": "mixer.norm", "mixer": "gqa"}) == "gqa_attention"
    assert map_ledger_row({"organ": "mixer.norm", "mixer": "deltanet"}) == "deltanet"


def test_fused_kernels_declared_and_default_off():
    ev = shader_evidence()
    assert ev["shader_present"]
    assert ev["all_kernels_declared"], ev
    assert ev["wired"]
    assert ev["default_off"]
    assert ev["organ_isolate"]
    assert ev["does_not_write_dense_w"] is True


def test_receipt_schema_organs_roof_and_no_second_27b():
    doc = receipt()
    assert RECEIPT.is_file()
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_mutate_parent"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["roof_gb_s"] == ROOF_GB_S
    assert doc["n018_production_gb_s"] == N018_PRODUCTION_GB_S
    assert abs(doc["gap_gb_s"] - GAP_GB_S) < 0.2
    assert doc["dense_w_materialized"] == 0
    assert doc["prior_not_rederived"]["did_not_retry_mlp_tile"] is True
    assert doc["prior_not_rederived"]["n017_dram_roof_gb_s"] == ROOF_GB_S
    assert set(doc["organs_named"]) == set(ORGANS)
    loc = doc["parent_immutable"]
    assert Path(loc["path"]).resolve() == PARENT_ROOT.resolve() or str(PARENT_ROOT) in loc["path"]
    assert loc["outside_worktree"] is True
    assert "10+ GiB" in doc["occupancy"]["note"] or "27B" in doc["occupancy"]["note"]
    law = doc["causal_benchmark_law"]
    assert law["kernel_identity"] == KERNEL_GOOD
    assert KERNEL_BAD in law["bad_control"]
    assert "628" in law["dispatch_count"] and "580" in law["dispatch_count"]


def test_receipt_attributes_organs_or_names_absent():
    doc = receipt()
    att = doc["organ_attribution"]
    assert att["kind"] in ("MEASURED", "ABSENT")
    if att["kind"] != "MEASURED":
        assert att.get("absent_reason")
        return
    assert att["roof_gb_s"] == ROOF_GB_S
    assert att["production_active_bytes"] == PARENT_ACTIVE_BYTES
    assert att["n018_anchor_gb_s"] == N018_PRODUCTION_GB_S
    assert att["dense_w_materialized"] == 0
    assert att["scale_onto_production"] is not None
    assert att["largest_roof_gap_organ"] in ORGANS
    assert att["next_optimization_target"] == att["largest_roof_gap_organ"]
    assert att["noop_empty"]["did_not_score"] is True
    organs = att["organs"]
    assert set(organs) == set(ORGANS)
    for name, row in organs.items():
        assert row["status"] == "MEASURED", name
        assert row["n_reps"] >= 7, name
        assert row["gpu_ns_min"] <= row["gpu_ns_median_isolated"] <= row["gpu_ns_max"]
        assert row["scaled_gpu_ns"] is not None
        assert row["fraction_of_roof_gap"] is not None
        assert row["dense_w_materialized"] == 0
        assert row["roof_gb_s"] == ROOF_GB_S
    fracs = [organs[n]["fraction_of_roof_gap"] for n in ORGANS]
    assert abs(sum(fracs) - 1.0) < 1e-6
    ns_ranks = [r["organ"] for r in att["ranked_by_ns"]]
    gap_ranks = [r["organ"] for r in att["ranked_by_fraction_of_roof_gap"]]
    assert ns_ranks[0] == att["largest_ns_organ"]
    assert gap_ranks[0] == att["largest_roof_gap_organ"]
    ns_vals = [organs[n]["scaled_gpu_ns"] for n in ns_ranks]
    assert ns_vals == sorted(ns_vals, reverse=True)
    # Winner of the roof-gap is the next target; named in one_line.
    assert att["largest_roof_gap_organ"] in doc["one_line"]
    assert "778.8" in doc["one_line"] or "422" in doc["one_line"]


def test_receipt_dispatch_below_628_or_measured_why():
    doc = receipt()
    red = doc["dispatch_reduction"]
    why = doc["no_further_or_the_cut"]
    assert red["kind"] in ("measured_reduction", "measured_no_win", "ABSENT")
    assert why["kind"] == red["kind"] or red["kind"] == "ABSENT"
    if red.get("kind") == "ABSENT":
        assert red.get("absent_reason")
        return
    assert red["dense_w_materialized"] == 0
    assert red["parent_dispatches"] == 756
    assert KERNEL_GOOD in (red.get("sentinel") or {}).get("kernel", KERNEL_GOOD)
    if red.get("measured") and red.get("token_ids_unchanged"):
        assert red["candidate_dispatches"] < BASELINE_DISPATCHES
        assert red["candidate_dispatches"] == CANDIDATE_DISPATCHES or (
            isinstance(red["candidate_dispatches"], int)
            and red["candidate_dispatches"] < BASELINE_DISPATCHES
        )
        assert red["token_ids_before"] == red["token_ids_after"]
        assert len(red["token_ids_after"]) == 16
        assert red["noop_control"]["did_not_score"] is True
        assert red["bad_control"]["rejected"] is True
        par = red["parity"]
        assert par.get("max_abs_diff") is not None
        assert par["max_abs_diff"] < 1e-4
        assert red["bad_control"]["kernel"] == KERNEL_BAD
        if red.get("gpu_ns_separated") is False and red.get("gpu_ns_before_reps"):
            assert "NOT SEPARATED" in (red.get("note") or "")
        assert "628" in doc["one_line"] or str(red["to"]) in doc["one_line"]
    else:
        assert red.get("why") or why.get("why")
        assert "628" in (red.get("why") or "") or "628" in (why.get("note") or doc["one_line"])


def test_kernel_autopsy_did_not_flag_new_kernels_defective():
    doc = receipt()
    autopsy = doc["kernel_autopsy"]
    assert autopsy["any_new_kernel_defective"] is False
    ev = doc["shader_evidence"]
    assert ev["all_kernels_declared"]
    assert ev["wired"]

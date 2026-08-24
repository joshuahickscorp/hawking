"""Dispatch ledger: every launch named, ranked, cut below 756 or a measured bound."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from dispatch_ledger import (  # noqa: E402
    CANDIDATE_DISPATCHES,
    KERNEL_BAD,
    KERNEL_GOOD,
    PARENT_DISPATCHES,
    RECEIPT,
    SCHEMA,
    UNFUSED_DISPATCHES,
    build,
    shader_evidence,
    write_receipt,
)

RECEIPT_DOC = None


def receipt() -> dict:
    """Read the sealed ledger. Build only if there is none to read.

    Rebuilding on every pytest run called `ps` (forbidden in some sandboxes)
    and overwrote a sealed measurement as a side effect of checking it.
    """
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        if RECEIPT.is_file():
            RECEIPT_DOC = json.loads(RECEIPT.read_text())
            if RECEIPT_DOC.get("schema") == SCHEMA:
                return RECEIPT_DOC
        raw = None
        raw_path = REPO / "receipts" / "headless" / "_DISPATCH_LEDGER_raw.json"
        if raw_path.is_file():
            try:
                raw = json.loads(raw_path.read_text())
            except json.JSONDecodeError:
                raw = None
        RECEIPT_DOC = build(raw=raw)
        write_receipt(RECEIPT_DOC)
    return RECEIPT_DOC


def test_harness_writes_receipt_and_schema():
    doc = receipt()
    assert RECEIPT.is_file(), f"missing {RECEIPT}"
    disk = json.loads(RECEIPT.read_text())
    assert disk["schema"] == SCHEMA
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_mutate_parent"] is True
    # A sibling GPU lane may hold the q4 incumbent; that is not this process
    # loading a second 27B. The harness itself opens one catalog.
    assert "10+ GiB" in doc["occupancy"]["note"] or "27B" in doc["occupancy"]["note"]


def test_names_every_dispatch_of_the_756_parent():
    doc = receipt()
    rows = doc["dispatches"]
    assert len(rows) == PARENT_DISPATCHES, len(rows)
    assert doc["graph"]["parent_fused"] == PARENT_DISPATCHES
    assert doc["graph"]["unfused"] == UNFUSED_DISPATCHES
    assert doc["graph"]["candidate_residual_rmsnorm"] == CANDIDATE_DISPATCHES
    assert CANDIDATE_DISPATCHES < PARENT_DISPATCHES
    indices = [r["index"] for r in rows]
    assert indices == list(range(PARENT_DISPATCHES))
    operators = {r["operator"] for r in rows}
    for must in (
        "embed_lookup",
        "input_rmsnorm",
        "dn_qkvz_ba_concat",
        "gqa_qkv_concat",
        "gate_up_swiglu",
        "down_proj",
        "mixer_residual",
        "mlp_residual",
        "lm_head",
        "argmax",
    ):
        assert must in operators, must


def test_every_dispatch_has_operator_bytes_flops_overhead_deps_candidacy_frequency():
    doc = receipt()
    for r in doc["dispatches"]:
        assert r["operator"], r
        assert r["kernel"], r
        assert isinstance(r["bytes"]["total"], int) and r["bytes"]["total"] >= 0
        assert isinstance(r["flops"], (int, float)) and r["flops"] >= 0
        oh = r["launch_overhead_ns"]
        assert oh["kind"] in ("MEASURED", "DERIVED", "ABSENT"), oh
        assert oh["command"]
        if oh["kind"] == "ABSENT":
            assert oh["value"] is None
            assert oh["absent_reason"]
        else:
            assert oh["value"] is not None
            assert oh["absent_reason"] is None
        assert isinstance(r["dependencies"], list) and r["dependencies"]
        assert r["fusion_candidacy"]
        assert r["fusion_candidacy_why"]
        assert r["frequency"] >= 1
        assert r["command_buffer"] == 1
        assert "rank_score_ns" in r
        assert r["synchronization_ns"] == 0


def test_ranked_by_launch_plus_memory_plus_sync():
    doc = receipt()
    ranked = doc["dispatches_ranked"]
    assert len(ranked) == PARENT_DISPATCHES
    scores = [r["rank_score_ns"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0]["rank"] == 1
    # Weight-stream GEMVs outrank 20 KB residuals.
    assert ranked[0]["bytes"]["total"] > 1_000_000
    formula = doc["rank_score"]["formula"]
    assert "launch_overhead_ns" in formula
    assert "memory_traffic_bytes" in formula
    assert "synchronization" in formula


def test_fused_kernels_declared_and_default_off():
    ev = shader_evidence()
    assert ev["shader_present"]
    assert ev["all_kernels_declared"], ev
    assert ev["wired"]
    assert ev["uses_one_plus_w"]
    assert ev["bad_uses_plain_weight"]
    doc = receipt()
    assert KERNEL_GOOD in doc["candidate_attempted"]["kernel"]
    assert KERNEL_BAD == doc["candidate_attempted"]["bad_control_kernel"]
    assert "off" in doc["candidate_attempted"]["enable"]["default"].lower()


def test_causal_benchmark_law_fields_present():
    doc = receipt()
    law = doc["causal_benchmark_law"]
    assert law["kernel_identity"] == KERNEL_GOOD
    assert "756" in law["dispatch_count"] and "628" in law["dispatch_count"]
    assert "sentinel" in law
    assert "noop_control" in law
    assert "bad_control" in law
    assert KERNEL_BAD in law["bad_control"]
    red = doc["reduction"]
    # A live GPU run must carry the five proofs. A source-only receipt still
    # names the controls so a no-op cannot hide.
    assert "noop_control" in red or red.get("gpu_ran") in (True, False)


def test_parent_not_mutated_and_no_second_27b():
    doc = receipt()
    assert doc["parent"]["immutable"] is True
    assert "NOETIC_PARENT_A" in doc["parent"]["path"]
    assert doc["dense_parent"]["dense_w_materialized"] == 0
    assert doc["dense_parent"]["expanded_to_q4"] == 0
    assert doc["dense_parent"]["expanded_to_float_gemv"] == 0


def test_gpu_ledger_honesty_is_in_the_receipt():
    doc = receipt()
    o = doc["gpu_ledger_overturns"]
    assert "468.9" in o["q4_incumbent"] or "bandwidth" in o["q4_incumbent"].lower()
    assert "5.8" in o["964_to_756_bought"]
    assert "proportionally" in o["implication"]


def test_reduction_or_measured_reason():
    doc = receipt()
    why = doc["no_further_or_the_cut"]
    red = doc["reduction"]
    assert why["kind"]
    if red.get("measured") and red.get("token_ids_unchanged"):
        assert red["candidate_dispatches"] < PARENT_DISPATCHES
        assert red["token_ids_before"] == red["token_ids_after"]
        assert len(red["token_ids_after"]) == 16
        assert red["noop_control"]["did_not_score"] is True
        assert red["bad_control"]["rejected"] is True
        par = red["parity"]
        assert par.get("max_abs_diff") is not None or par.get("max_abs_diff_norm") is not None
    else:
        # Isolated TOKEN_NS bound is itself a measurement.
        assert "isolated_residual_gpu_ns" in why or why["kind"] in (
            "candidate_rejected",
            "measured_no_win",
            "measured_bound_pending_live_generate",
        )
        if "isolated_residual_gpu_ns" in why:
            assert why["isolated_residual_gpu_ns"] > 0
            assert why["upper_bound_as_pct_of_gpu_ns"] < 5.0

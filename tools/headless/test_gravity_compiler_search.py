"""Gravity compiler search: a candidate is representation AND execution.

pytest tools/headless -q must write receipts/headless/GRAVITY_COMPILER_SEARCH.json
and exit 0. Two things have to actually happen in this process:

  1. score() on a candidate that names no kernel raises ScoringRefused
     and does not return a number (no default kernel).
  2. credit_kernel_win() on a faster-per-dispatch kernel attached to the
     Q4 reconstruct-then-GEMM lowering raises KernelWinRefused.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from gravity_compiler_search import (  # noqa: E402
    KIND_ABSENT,
    RECEIPT,
    SCHEMA,
    KernelWinRefused,
    ScoringRefused,
    build,
    candidate_q4_geo_tpr64,
    candidate_q4_reconstruct_gemv,
    candidate_q4_serial,
    candidate_uv_cache,
    candidate_uv_fused,
    candidate_without_kernel,
    compile_candidates,
    credit_kernel_win,
    kernel_catalog,
    per_dispatch_kernel_compute_ns,
    representation_key,
    score,
    try_credit_kernel_win,
    try_score,
    write_receipt,
)

RECEIPT_DOC = None


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        RECEIPT_DOC = build()
        write_receipt(RECEIPT_DOC)
    return RECEIPT_DOC


def test_harness_writes_receipt():
    doc = receipt()
    assert RECEIPT.is_file(), f"missing {RECEIPT}"
    on_disk = json.loads(RECEIPT.read_text())
    assert on_disk["schema"] == SCHEMA
    assert doc["schema"] == SCHEMA


def test_compiler_emits_at_least_two_q4_candidates_that_differ_in_execution():
    compiled = compile_candidates()
    q4 = [c for c in compiled if c["representation"]["family"] == "grouped_absmax_q4"]
    assert len(q4) >= 2
    reps = {c["identity"]["representation_fingerprint"] for c in q4}
    execs = {c["identity"]["execution_fingerprint"] for c in q4}
    cands = {c["identity"]["candidate_fingerprint"] for c in q4}
    kernels = {c["execution"]["kernel"] for c in q4}
    assert len(reps) == 1, "Q4 candidates must hold the stored representation fixed"
    assert len(execs) == len(q4)
    assert len(cands) == len(q4)
    assert len(kernels) >= 2


def test_same_weights_different_kernels_are_different_candidates():
    a = candidate_q4_geo_tpr64()
    b = candidate_q4_serial()
    assert representation_key(a["representation"]) == representation_key(b["representation"])
    assert a["identity"]["representation_fingerprint"] == b["identity"]["representation_fingerprint"]
    assert a["execution"]["kernel"] != b["execution"]["kernel"]
    assert a["identity"]["execution_fingerprint"] != b["identity"]["execution_fingerprint"]
    assert a["identity"]["candidate_fingerprint"] != b["identity"]["candidate_fingerprint"]


def test_reconstruct_holds_q4_representation_and_changes_the_kernel():
    fused = candidate_q4_geo_tpr64()
    recon = candidate_q4_reconstruct_gemv()
    assert fused["identity"]["representation_fingerprint"] == recon["identity"]["representation_fingerprint"]
    assert fused["execution"]["kernel"] == "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128"
    assert recon["execution"]["kernel"] == "gemv_simdgroup_f32"
    assert recon["execution"]["runtime_graph"]["prologue_kernel"] == "qwen_uniform_q4_decode_vector"
    assert recon["label"] == "ORACLE"


def test_compiler_never_emits_a_kernel_less_candidate():
    for c in compile_candidates():
        assert c["execution"]["kernel"], c["id"]
        scored = score(c)
        assert scored["scored"] is True


def test_score_raises_without_a_kernel_and_does_not_return_a_number():
    cand = candidate_without_kernel()
    assert cand["execution"]["kernel"] is None
    with pytest.raises(ScoringRefused) as ei:
        score(cand)
    payload = ei.value.payload
    assert payload["defaulted_kernel"] is None
    assert payload["kernel_supplied"] is None
    assert "default" in payload["why"].lower() or "refusing" in payload["why"].lower()
    # Returning a numeric score here is the bug. try_score must not invent one.
    attempt = try_score(cand)
    assert attempt["refused"] is True
    assert attempt["scored"] is False
    assert attempt["defaulted_kernel"] is None
    assert "result" not in attempt or attempt.get("result") is None


@pytest.mark.parametrize("sent", ["default", "auto", "", "none", "unspecified", "implicit"])
def test_score_refuses_kernel_sentinels(sent):
    cand = candidate_without_kernel()
    cand["execution"] = dict(cand["execution"])
    cand["execution"]["kernel"] = sent
    with pytest.raises(ScoringRefused) as ei:
        score(cand)
    assert ei.value.payload["defaulted_kernel"] is None
    assert ei.value.payload["kernel_supplied"] == sent


def test_score_refuses_a_candidate_with_no_execution_half():
    cand = candidate_q4_geo_tpr64()
    cand = dict(cand)
    cand["execution"] = None
    with pytest.raises(ScoringRefused) as ei:
        score(cand)
    assert ei.value.payload["defaulted_kernel"] is None


def test_reconstruct_gemv_is_faster_per_dispatch_on_the_compute_floor():
    fused = candidate_q4_geo_tpr64()
    recon = candidate_q4_reconstruct_gemv()
    fused_pd = per_dispatch_kernel_compute_ns(fused)
    recon_pd = per_dispatch_kernel_compute_ns(recon)
    assert recon_pd < fused_pd
    ratio = fused_pd / recon_pd
    # Native operator: executable FLOPs / GEMV-MAC FLOPs ≈ 1.50
    assert 1.45 < ratio < 1.55, ratio


def test_kernel_win_is_refused_when_representation_traffic_dominates():
    fused = candidate_q4_geo_tpr64()
    recon = candidate_q4_reconstruct_gemv()
    with pytest.raises(KernelWinRefused) as ei:
        credit_kernel_win(recon, fused)
    payload = ei.value.payload
    assert payload["faster_per_dispatch"] is True
    assert payload["traffic_dominates"] is True
    assert payload["same_stored_representation"] is True
    assert payload["decision"] == "KERNEL_WIN_REFUSED"
    assert "traffic" in payload["why"].lower() or "byte" in payload["why"].lower()
    # try_ wrapper records the refusal rather than turning it into WIN
    attempt = try_credit_kernel_win(recon, fused)
    assert attempt["refused"] is True
    assert attempt["credited"] is False
    assert attempt["decision"] == "KERNEL_WIN_REFUSED"


def test_serial_is_not_a_flop_kernel_win_against_tpr64():
    """Same algorithm FLOPs; occupancy is a different axis, not a compute-floor win."""
    fused = candidate_q4_geo_tpr64()
    serial = candidate_q4_serial()
    attempt = try_credit_kernel_win(serial, fused)
    assert attempt["decision"] == "NOT_FASTER_PER_DISPATCH"
    assert attempt["refused"] is False


def test_every_candidate_reports_storage_and_active_bpw():
    doc = receipt()
    assert doc["bpw_every_candidate"]
    for row in doc["bpw_every_candidate"]:
        assert isinstance(row["stored_bpw"], (int, float))
        assert isinstance(row["active_bpw_fused"], (int, float))
        assert row["active_bpw_cached_dense"] is None or isinstance(
            row["active_bpw_cached_dense"], (int, float)
        )
        assert row["which"]


def test_uv_0485_stored_is_16_active_when_cached():
    fused = candidate_uv_fused()
    cached = candidate_uv_cache()
    assert fused["identity"]["representation_fingerprint"] == cached["identity"]["representation_fingerprint"]
    assert abs(fused["representation"]["bpw"]["stored_bpw"] - 0.0485) < 1e-4
    assert abs(fused["representation"]["bpw"]["active_bpw_fused"] - 0.0485) < 1e-4
    assert cached["representation"]["bpw"]["active_bpw_cached_dense"] == 16.0
    assert fused["execution"]["kernel"] != cached["execution"]["kernel"]
    # Organ-level: token_ns is ABSENT, not 0
    fused_score = score(fused)
    assert fused_score["token_ns"]["kind"] == KIND_ABSENT
    assert fused_score["token_ns"]["value"] is None
    assert fused_score["token_ns"]["absent_reason"]


def test_absent_is_never_written_as_zero():
    doc = receipt()
    seen_absent = False
    for cell in _walk_cells(doc):
        if cell.get("kind") == KIND_ABSENT:
            seen_absent = True
            assert cell["value"] is None
            assert cell["absent_reason"]
        elif cell.get("value") == 0:
            assert cell.get("kind") != KIND_ABSENT
    gpu = doc["gpu_live_this_run"]["gpu_wall_ns_per_token"]
    assert gpu["kind"] == KIND_ABSENT
    assert gpu["value"] is None
    assert seen_absent


def test_required_kernels_are_declared_in_shaders():
    cat = kernel_catalog()
    assert cat["all_required_declared"], cat["missing_required"]
    for name, info in cat["required"].items():
        assert info["declared"], name
        assert info["on_disk"], name


def test_receipt_records_both_demonstrations_and_self_check():
    doc = receipt()
    demo = doc["demonstration_scoring_without_kernel"]
    assert demo["refused"] is True
    assert demo["scored"] is False
    assert demo["defaulted_kernel"] is None
    assert demo["all_sentinel_probes_refused"] is True
    kw = doc["demonstration_kernel_win_refused_when_traffic_dominates"]
    assert kw["decision"] == "KERNEL_WIN_REFUSED"
    assert kw["refused"] is True
    assert kw["payload"]["faster_per_dispatch"] is True
    assert kw["payload"]["traffic_dominates"] is True
    sc = doc["self_check"]
    assert all(sc.values()), [k for k, v in sc.items() if not v]
    assert doc["identity_checks"]["q4_fused_vs_serial_same_representation"] is True
    assert doc["identity_checks"]["q4_fused_vs_serial_different_candidate"] is True
    assert "Did not load a second 27B." in doc["what_i_did_not_do"]


def test_native_operator_shape_matches_the_1_50_3_49_7_34_tradeoff():
    doc = receipt()
    s = doc["native_operator_shape"]
    assert abs(s["flops_ratio_executable_over_source"] - 1.50) < 0.01
    assert abs(s["ops_ratio_executable_over_source"] - 3.49) < 0.02
    assert abs(s["dram_ratio_source_over_executable"] - 7.34) < 0.05
    assert s["dispatch_count_both"] == 964


def _walk_cells(obj):
    if isinstance(obj, dict):
        if "kind" in obj and "value" in obj:
            yield obj
        for v in obj.values():
            yield from _walk_cells(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_cells(v)

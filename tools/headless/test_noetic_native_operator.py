"""Native-operator receipt: SOURCE vs EXECUTABLE, ORACLE labelled.

pytest tools/headless -q must write receipts/headless/NOETIC_NATIVE_OPERATOR.json
and exit 0. Oracle = reconstruct-then-matmul, or peak temp reaching a 2-D parent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from noetic_native_operator import (  # noqa: E402
    KIND_ABSENT,
    METRICS,
    RECEIPT,
    SCHEMA,
    build,
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


def test_source_and_executable_columns_side_by_side():
    doc = receipt()
    cols = doc["columns"]
    assert set(cols) == set(METRICS)
    for metric in METRICS:
        assert set(cols[metric]) == {"SOURCE", "EXECUTABLE"}, metric
        for side in ("SOURCE", "EXECUTABLE"):
            cell = cols[metric][side]
            assert "value" in cell and "kind" in cell and "unit" in cell, (metric, side)
            if cell["kind"] == KIND_ABSENT:
                assert cell["value"] is None
                assert cell["absent_reason"]
            else:
                assert cell["value"] is not None


def test_every_path_has_the_five_metrics_side_by_side():
    doc = receipt()
    assert doc["paths"], "no paths measured"
    for path in doc["paths"]:
        for metric in METRICS:
            pair = path["columns"][metric]
            assert set(pair) == {"SOURCE", "EXECUTABLE"}, (path["id"], metric)


def test_every_oracle_path_is_labelled_ORACLE():
    doc = receipt()
    assert doc["oracle_path_ids"], "expected at least one ORACLE control"
    for path in doc["paths"]:
        is_oracle = (
            path["materializes_dense_w_then_ordinary_matmul"]
            or path["peak_temp_reaches_parent_tensor_shape"]
        )
        if is_oracle:
            assert path["label"] == "ORACLE", path["id"]
            assert path["id"] in doc["oracle_path_ids"]
        else:
            assert path["label"] == "NATIVE", path["id"]
            assert path["id"] in doc["native_path_ids"]
    assert doc["every_oracle_path_labelled_ORACLE"] is True
    assert doc["discriminator_holds"] is True


def test_peak_temp_reaching_parent_forces_ORACLE():
    doc = receipt()
    hits = 0
    for path in doc["paths"]:
        if path["parent"]["rank"] != 2:
            continue
        peak = path["columns"]["peak_temporary_materialization"]["EXECUTABLE"]
        if peak["kind"] == KIND_ABSENT:
            continue
        if peak["value"] >= path["parent"]["bytes"]:
            hits += 1
            assert path["label"] == "ORACLE", path["id"]
            assert path["peak_temp_reaches_parent_tensor_shape"] is True
    assert hits >= 2, "expected reconstruct GEMM and UV-cache to hit parent shape"


def test_fused_production_is_NATIVE_and_does_not_write_parent_W():
    doc = receipt()
    prod = next(p for p in doc["paths"] if p["id"] == "qwen38_uniform_q4_fused")
    assert prod["label"] == "NATIVE"
    assert prod["on_production_token"] is True
    assert prod["materializes_dense_w_then_ordinary_matmul"] is False
    peak = prod["columns"]["peak_temporary_materialization"]["EXECUTABLE"]["value"]
    assert peak < prod["parent"]["bytes"]
    # FLOPs/ops/dispatches/DRAM/peak all present on both columns
    for metric in METRICS:
        for side in ("SOURCE", "EXECUTABLE"):
            cell = prod["columns"][metric][side]
            assert cell["kind"] != KIND_ABSENT
            assert isinstance(cell["value"], (int, float))


def test_decode_vector_then_gemm_is_ORACLE():
    doc = receipt()
    p = next(x for x in doc["paths"] if x["id"] == "qwen38_q4_decode_vector_then_gemm")
    assert p["label"] == "ORACLE"
    assert p["materializes_dense_w_then_ordinary_matmul"] is True
    assert p["peak_temp_reaches_parent_tensor_shape"] is True
    assert p["on_production_token"] is False


def test_hgravs_two_stage_native_reconstruct_oracle():
    doc = receipt()
    native = next(p for p in doc["paths"] if p["id"] == "q80_hgravs01_two_stage")
    oracle = next(p for p in doc["paths"] if p["id"] == "q80_hgravs01_reconstruct_then_gemm")
    assert native["label"] == "NATIVE"
    assert oracle["label"] == "ORACLE"
    npeak = native["columns"]["peak_temporary_materialization"]["EXECUTABLE"]["value"]
    opeak = oracle["columns"]["peak_temporary_materialization"]["EXECUTABLE"]["value"]
    assert npeak < native["parent"]["bytes"]
    assert opeak >= oracle["parent"]["bytes"]
    # two-stage does less GEMV work than dense, reconstruct does not hide that
    nf = native["columns"]["flops"]["EXECUTABLE"]["value"]
    sf = native["columns"]["flops"]["SOURCE"]["value"]
    assert nf < sf


def test_0485_regime_fused_native_cache_oracle():
    doc = receipt()
    r = doc["regime_0485"]
    assert abs(r["stored_bpw"] - 0.0485) < 1e-4
    assert abs(r["active_bpw_fused"] - 0.0485) < 1e-4
    assert r["active_bpw_cached_dense"] == 16.0
    assert r["fused_label"] == "NATIVE"
    assert r["cached_label"] == "ORACLE"
    fused = next(p for p in doc["paths"] if p["id"] == r["fused_path_id"])
    cached = next(p for p in doc["paths"] if p["id"] == r["cached_path_id"])
    assert fused["label"] == "NATIVE"
    assert cached["label"] == "ORACLE"
    assert fused["regime"]["which"].startswith("0.0485")
    assert cached["peak_temp_reaches_parent_tensor_shape"] is True


def test_rank1_rmsnorm_decode_is_not_ORACLE():
    doc = receipt()
    p = next(x for x in doc["paths"] if x["id"] == "qwen30_rmsnorm_vector_decode")
    assert p["parent"]["rank"] == 1
    assert p["label"] == "NATIVE"
    assert p["peak_temp_reaches_parent_tensor_shape"] is False


def test_absent_is_never_written_as_zero():
    doc = receipt()
    seen_absent = False
    for path in doc["paths"]:
        for metric in METRICS:
            for side in ("SOURCE", "EXECUTABLE"):
                cell = path["columns"][metric][side]
                if cell["kind"] == KIND_ABSENT:
                    seen_absent = True
                    assert cell["value"] is None, (path["id"], metric, side)
                    assert cell["absent_reason"]
                elif cell["value"] == 0:
                    # 0 is allowed only as a measured zero
                    assert cell["kind"] != KIND_ABSENT
    # GPU live of this run is ABSENT, not 0
    gpu = doc["gpu_live_this_run"]["gpu_wall_ns_per_token"]
    assert gpu["kind"] == KIND_ABSENT
    assert gpu["value"] is None
    assert seen_absent  # PQ codebook bytes / index ops at least


def test_decode_vector_kernels_are_named_and_not_on_qwen38_fused_path():
    doc = receipt()
    names = {k["name"] for k in doc["shader_evidence"]["decode_vector_kernels"]}
    assert "qwen_uniform_q4_decode_vector" in names
    assert "qwen_complete_binary_decode_vector" in names
    fused = next(p for p in doc["paths"] if p["id"] == "qwen38_uniform_q4_fused")
    assert fused["label"] == "NATIVE"
    oracle = next(p for p in doc["paths"] if p["id"] == "qwen38_q4_decode_vector_then_gemm")
    assert oracle["label"] == "ORACLE"


def test_q80_forbidden_path_is_in_source():
    ev = receipt()["shader_evidence"]
    assert ev["q80_forbidden_token_path_comment"] is True
    assert ev["q80_cpu_two_stage_named"] is True
    assert ev["q80_decode_vector_refuses_dense_w"] is True
    assert ev["shaders"]["q80_mixed_decode.metal"]["needles"]["threadgroup float mid[kRankCap]"] >= 0


def test_self_check_all_true():
    doc = receipt()
    failed = [k for k, v in doc["self_check"].items() if v is not True]
    assert not failed, failed

"""HYBRID_OPERATOR: binary bulk + distributed correction as ONE fused native operator."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from bytes_frontier import Q2F_BPW, MLP_ELEMENTS  # noqa: E402
from hybrid_operator import (  # noqa: E402
    BINARY_BPW,
    K_SHARED,
    NOOP_KERNELS,
    PRODUCTION_KERNELS,
    Q2F_COMPLETE_NS,
    RANK_BUDGET,
    RECEIPT,
    SCHEMA,
    SHADER,
    lowrank_bill,
    rank_for_extra_bpw,
    shader_autopsy,
    shared_hybrid_bill,
    uv_bytes,
)
from kernel_competence import kernel_bodies, params_of, screen_kernel, strip_comments  # noqa: E402
from shared_basis_kernel import fused_bpw  # noqa: E402


def test_binary_bulk_is_1_25_and_q2f_is_2_25():
    assert abs(BINARY_BPW - 1.25) < 1e-12
    assert abs(Q2F_BPW - 2.25) < 1e-12
    assert Q2F_COMPLETE_NS == 27_547_874


def test_lowrank_r8_bills_correction_and_stays_under_q2f():
    b = lowrank_bill(8)
    assert b["binary_bytes"] == MLP_ELEMENTS * 1.25 / 8.0
    assert b["correction_bytes"] == float(uv_bytes(8))
    assert b["correction_bytes"] > 0
    assert b["active_bytes"] == b["binary_bytes"] + b["correction_bytes"]
    assert b["active_bpw"] > BINARY_BPW
    assert b["active_bpw"] < Q2F_BPW
    assert b["below_q2f_bpw"] is True
    assert b["dense_w"] == 0.0
    assert b["complete_ebpw"] > b["active_bpw"]
    # extra bits are the correction, not a hidden plane
    assert abs(b["correction_bpw"] - (b["active_bpw"] - BINARY_BPW)) < 1e-12


def test_lowrank_r32_still_under_q2f_r256_is_not():
    r32 = lowrank_bill(32)
    r256 = lowrank_bill(256)
    assert r32["active_bpw"] < Q2F_BPW
    assert r256["active_bpw"] > Q2F_BPW
    assert r256["correction_bpw"] > 1.0
    # the 1.0-bpw extra budget is r<=247
    assert RANK_BUDGET == 247
    assert abs(rank_for_extra_bpw(1.0) - (MLP_ELEMENTS / (8.0 * uv_bytes(1)))) < 1e-9
    assert rank_for_extra_bpw(1.0) > 247
    assert rank_for_extra_bpw(1.0) < 248
    assert lowrank_bill(RANK_BUDGET)["active_bpw"] < Q2F_BPW
    assert lowrank_bill(RANK_BUDGET + 1)["active_bpw"] > Q2F_BPW


def test_shared_k2_hybrid_bills_binary_plus_amortized_bases_under_q2f():
    sh = shared_hybrid_bill(2)
    extra = fused_bpw(2)
    assert abs(extra["active_bpw"] - 0.53125) < 1e-12
    assert abs(sh["active_bpw"] - (BINARY_BPW + extra["active_bpw"])) < 1e-12
    assert sh["active_bpw"] < Q2F_BPW
    assert sh["below_q2f_bpw"] is True
    assert sh["correction_bytes"] == extra["active_bytes"]
    assert sh["basis_sign_bytes"] > 0
    assert sh["scale_bytes"] > sh["basis_sign_bytes"]
    assert sh["dense_w"] == 0.0
    k4 = shared_hybrid_bill(4)
    assert k4["active_bpw"] > Q2F_BPW
    assert k4["below_q2f_bpw"] is False


def test_correction_is_distributed_not_sparse_or_island():
    # low-rank r and shared bases are full-width: every output channel gets a term.
    b = lowrank_bill(8)
    # 3 organs × (rows+cols) × r × f16
    assert b["correction_bytes"] == float(64 * 3 * 2 * 8 * (17408 + 5120))
    sh = shared_hybrid_bill(2)
    # signs once for the whole plane, not a CSR of islands
    assert sh["basis_sign_bytes"] == fused_bpw(2)["basis_sign_bytes"]


def test_production_kernels_exist_group64_shift_no_bind_time_rank():
    auto = shader_autopsy()
    assert auto["all_present"], auto["kernels_present"]
    assert auto["uses_shift_not_div"]
    assert auto["dense_w_written"] is False
    src = strip_comments(SHADER.read_text())
    for name in PRODUCTION_KERNELS + NOOP_KERNELS:
        assert f"kernel void {name}(" in src
        params = params_of(src, name)
        assert "constant uint& rows" not in params
        assert "constant uint& cols" not in params
        assert "constant uint& rank" not in params
        assert "constant uint& group_size" not in params
    assert "ocol >> 6u" in src or "col >> 6u" in src or ">> 6u" in src


def test_fused_kernels_are_competence_clear():
    src = strip_comments(SHADER.read_text())
    for name, body in kernel_bodies(src):
        r = screen_kernel(name, body, params_of(src, name))
        assert r["verdict"] == "CLEAR", f"{name} {r['verdict']} {r['findings']}"
        assert not r["findings"]
    auto = shader_autopsy()
    assert auto["all_clear"]
    assert auto["production_all_clear"]
    assert auto["production_no_bind_time_shape"]


def test_lowrank_reconstruct_is_distributed_and_beats_binary_on_gaussian():
    from fractional_bit_canon import _binary_meanabs, snap_f16

    rng = np.random.RandomState(0)
    w = rng.randn(64, 128).astype(np.float32)
    wb = _binary_meanabs(w, 64)
    u, s, vh = np.linalg.svd(w - wb, full_matrices=False)
    k = 8
    what = wb + (snap_f16(u[:, :k]) * snap_f16(s[:k])) @ snap_f16(vh[:k])
    assert what.shape == w.shape
    assert np.isfinite(what).all()
    err_b = float(np.linalg.norm(wb - w))
    err_h = float(np.linalg.norm(what - w))
    assert err_h < err_b
    # every row of U is used — not an island of channels
    assert np.count_nonzero(np.any(np.abs(u[:, :k]) > 0, axis=1)) == 64


def test_receipt_reports_two_fused_hybrids_ebpw_ns_rung_parity_dense_w():
    assert RECEIPT.is_file(), (
        "receipts/headless/HYBRID_OPERATOR.json missing — "
        "run python3 tools/headless/hybrid_operator.py"
    )
    doc = json.loads(RECEIPT.read_text())
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["did_not_mutate_noetic_parent_a"] is True
    assert doc["dense_w"] == 0
    assert doc["dense_w_materialized"] == 0
    assert doc["competent"] is True
    auto = doc["kernel_autopsy"]
    assert auto["production_all_clear"]
    assert auto["dense_w_written"] is False
    assert doc["parity"]["ok"] is True
    assert doc["parity"]["noop_diverges"] is True
    reps = doc["representations"]
    hybrids = [
        r
        for r in reps
        if r.get("fused_native_operator")
        and r.get("correction") in {"lowrank", "shared_basis"}
    ]
    assert len(hybrids) >= 2
    kinds = {r["correction"] for r in hybrids}
    assert "lowrank" in kinds
    assert "shared_basis" in kinds
    for h in hybrids:
        assert h["dense_w"] == 0
        assert h["distributed"] is True
        assert h["not_sparse"] is True
        assert h["not_island"] is True
        assert h["active_bytes"] > 0
        assert h["complete_ebpw"] > 0
        assert h["correction_bytes"] > 0
        assert h["one_dispatch_per_gemv"] is True
        ct = h["COMPLETE_TOKEN_NS"]
        gpu = ct["mlp_graph_gpu_ns"]
        assert gpu.get("n", 0) >= 7
        assert gpu["min"] <= gpu["median"] <= gpu["max"]
        assert ct.get("median") is not None
        assert ct.get("reps", 0) >= 7
        assert h["parity"]["ok"] is True
        ctrl = h.get("control") or {}
        assert ctrl.get("label") in {"SEPARATED", "NOT SEPARATED"}
        if ctrl.get("overlap") is True:
            assert ctrl["label"] == "NOT SEPARATED"
        rung = (h.get("composition") or {}).get("rung") or doc["composition_ladder"]["rung"]
        assert rung in {
            "local_functional_probe",
            "held_out_activation",
            "adjacent_layers",
            "short_chain",
            "complete_organ",
            "complete_token",
            "coherent_generation",
        }
        if h.get("counts_as_win"):
            assert rung == "coherent_generation"
            assert h["below_q2f_bpw"] is True
            assert h["faster_than_q2f_27_55ms"] is True
    assert isinstance(doc["coherent_hybrid_beats_q2f"], bool)
    assert doc["finding"]["reason"]
    assert doc["q2f_baseline"]["complete_token_ns"] == Q2F_COMPLETE_NS
    assert doc["q2f_baseline"]["bpw"] == Q2F_BPW
    assert doc["held_out_lowrank"].get("real_activations") is True
    assert doc["held_out_lowrank"].get("not_gaussian") is True
    if doc["coherent_hybrid_beats_q2f"]:
        assert any(h.get("counts_as_win") for h in hybrids)
    else:
        assert "2.25" in doc["answer"] or "q2f" in doc["answer"]
        assert doc["finding"]["died_at"] or "floor" in doc["finding"]["reason"].lower()

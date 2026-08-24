"""SHARED_BASIS_KERNEL: competent fused operator for the 0.53-bpw shared binary MLP."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from kernel_competence import kernel_bodies, params_of, screen_kernel, strip_comments  # noqa: E402
from shared_basis_kernel import (  # noqa: E402
    FUSED_PRODUCTION,
    N032_TWOPASS_DISPATCHES,
    RECEIPT,
    SCHEMA,
    SHADER,
    fused_bpw,
    shader_autopsy,
)


def test_k2_64_layer_bills_0_53_not_two():
    k2 = fused_bpw(2, n_layers=64, group=64)
    assert abs(k2["active_bpw"] - 0.53125) < 1e-12
    assert k2["active_bytes"] > 0
    assert k2["scale_bytes"] > k2["basis_sign_bytes"]
    k2_2L = fused_bpw(2, n_layers=2, group=64)
    # Signs are stored once; amortizing over 2 layers instead of 64 raises bpw.
    assert k2_2L["active_bpw"] > k2["active_bpw"]


def test_k_curve_coefficient_cost_does_not_vanish():
    k2 = fused_bpw(2, 64, 64)
    k4 = fused_bpw(4, 64, 64)
    k8 = fused_bpw(8, 64, 64)
    assert k4["active_bpw"] > k2["active_bpw"]
    assert k8["active_bpw"] > k4["active_bpw"]
    # 64-layer K=8 is ~2.125 bpw, still near q2f, not a free lunch.
    assert k8["active_bpw"] < 2.3


def test_fused_kernels_exist_and_are_group64_shifts():
    src = strip_comments(SHADER.read_text())
    for name in FUSED_PRODUCTION:
        assert f"kernel void {name}(" in src
    assert "ocol >> 6u" in src
    for name, body in kernel_bodies(src):
        params = params_of(src, name)
        blob = params + "\n" + body
        assert "constant uint& rows" not in blob
        assert "constant uint& group_size" not in blob
        assert "constant uint& cols" not in blob
    # Two-pass per-group barrier must not be the production path.
    assert "for (uint g = 0u; g < GPR; ++g)" not in src


def test_fused_kernels_are_competence_clear():
    src = strip_comments(SHADER.read_text())
    for name, body in kernel_bodies(src):
        r = screen_kernel(name, body, params_of(src, name))
        assert r["verdict"] == "CLEAR", f"{name} {r['verdict']} {r['findings']}"
        assert not r["findings"]
    auto = shader_autopsy()
    assert auto["all_clear"]
    assert auto["production_no_runtime_div"]
    assert auto["production_no_runtime_loop"]
    assert auto["production_no_dynamic_inner_branch"]
    assert auto["production_no_bind_time_shape"]


def test_receipt_reports_competent_kernel_before_after_and_ladder():
    assert RECEIPT.is_file(), (
        "receipts/headless/SHARED_BASIS_KERNEL.json missing — "
        "run python3 tools/headless/shared_basis_kernel.py"
    )
    doc = json.loads(RECEIPT.read_text())
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["did_not_mutate_noetic_parent_a"] is True
    assert doc["dense_w_materialized"] == 0
    assert doc["dense_w"] == 0
    assert doc["competent"] is True
    auto = doc["kernel_autopsy"]
    assert auto["production_all_clear"]
    assert auto["production_no_runtime_div"]
    assert auto["production_no_runtime_loop"]
    assert auto["production_no_dynamic_inner_branch"]
    assert auto["production_no_bind_time_shape"]
    assert doc["parity"]["ok"] is True
    assert doc["parity"]["noop_diverges"] is True

    before = doc["before"]["COMPLETE_TOKEN_NS"]
    after = doc["after"]["COMPLETE_TOKEN_NS"]
    for arm in (before, after):
        gpu = arm["mlp_graph_gpu_ns"]
        assert gpu.get("n", 0) >= 7
        assert gpu["min"] <= gpu["median"] <= gpu["max"]
        assert arm.get("median") is not None
        assert arm["reps"] >= 7

    disp = doc["dispatches"]
    assert disp["before"] == N032_TWOPASS_DISPATCHES or disp["before"] >= 384
    assert disp["after_fused_per_gemv"] < N032_TWOPASS_DISPATCHES
    assert disp["driven_down_from_384"] is True

    assert doc["active_bytes_per_token"] > 0
    assert abs(doc["active_bpw"] - 0.53125) < 1e-6
    assert "byte_win_translates_to_token_ns" in doc
    assert isinstance(doc["byte_win_translates_to_token_ns"], bool)
    assert doc["finding"]["reason"]
    assert "rung" in doc["composition_ladder"]

    ctrl = doc["controls"]
    assert ctrl["label"] in {"SEPARATED", "NOT SEPARATED"}
    if ctrl["label"] == "NOT SEPARATED":
        assert ctrl["overlap"] is True
    else:
        assert ctrl["serial"]["overlap_with_fused"] is False
        assert ctrl["noop"]["overlap_with_fused"] is False

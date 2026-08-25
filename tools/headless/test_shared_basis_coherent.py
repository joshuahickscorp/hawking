"""SHARED_BASIS_COHERENT: K-sweep on the full model + protected islands."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from kernel_competence import kernel_bodies, params_of, screen_kernel, strip_comments  # noqa: E402
from shared_basis_coherent import (  # noqa: E402
    K_MAX,
    K_SWEEP,
    PRODUCTION_KERNELS,
    Q2F_BPW,
    Q2F_COMPLETE_NS,
    RECEIPT,
    SCHEMA,
    SHADER,
    complete_mlp_bytes,
    pack_pm1,
    reconstruct_what,
    unpack_pm1,
)
from shared_basis_kernel import fused_bpw  # noqa: E402


def test_k8_64_layer_bills_2_125_below_q2f():
    k8 = fused_bpw(8, n_layers=64, group=64)
    assert abs(k8["active_bpw"] - 2.125) < 1e-12
    assert k8["active_bpw"] < Q2F_BPW
    k2 = fused_bpw(2, n_layers=64, group=64)
    assert abs(k2["active_bpw"] - 0.53125) < 1e-12
    acc = complete_mlp_bytes(8)
    assert abs(acc["active_bpw"] - 2.125) < 1e-12
    assert acc["complete_ebpw"] > 0
    assert acc["active_bytes"] == k8["active_bytes"]
    assert acc["dram_bytes_per_token"] > acc["active_bytes"]


def test_k16_bills_above_q2f_so_cannot_win_density():
    k16 = fused_bpw(16, n_layers=64, group=64)
    assert abs(k16["active_bpw"] - 4.25) < 1e-12
    assert k16["active_bpw"] > Q2F_BPW


def test_islands_are_billed_and_not_a_free_lunch():
    base = complete_mlp_bytes(2, n_protected=0)
    mix = complete_mlp_bytes(2, n_protected=12, protected_bpw=Q2F_BPW)
    assert mix["active_bytes"] > base["active_bytes"]
    assert mix["n_protected_tensors"] == 12
    assert mix["protected_bytes"] > 0
    assert mix["basis_sign_bytes"] == base["basis_sign_bytes"]
    # 12/192 tensors at 2.25 plus K=2 shared should stay under q2f body bpw.
    assert mix["active_bpw"] < Q2F_BPW
    full = complete_mlp_bytes(2, n_protected=192)
    assert full["active_bpw"] > mix["active_bpw"]


def test_pack_roundtrip_little_endian_matches_kernel():
    rng = np.random.RandomState(0)
    w = rng.randn(8, 64).astype(np.float32)
    packed = pack_pm1(w)
    recon = unpack_pm1(packed, 8, 64)
    assert recon.shape == (8, 64)
    assert set(np.unique(recon).tolist()) <= {-1.0, 1.0}
    assert np.all((w >= 0) == (recon > 0))
    # LSB is column 0, matching rust `1 << (flat & 7)`.
    assert packed.shape[0] == (8 * 64) // 8


def test_reconstruct_what_is_sum_of_scaled_bases():
    rows, cols, g, k = 4, 64, 64, 2
    b0 = np.ones((rows, cols), dtype=np.float32)
    b1 = np.where(np.arange(cols) % 2 == 0, 1.0, -1.0).astype(np.float32)
    b1 = np.broadcast_to(b1, (rows, cols)).copy()
    signs = np.stack([pack_pm1(b0), pack_pm1(b1)])
    scales = np.zeros((k, rows, cols // g), dtype=np.float16)
    scales[0] = np.float16(0.5)
    scales[1] = np.float16(0.25)
    what = reconstruct_what(signs, scales, k, rows, cols, g)
    expect = 0.5 * b0 + 0.25 * b1
    assert np.allclose(what, expect, atol=1e-4)


def test_k8_production_kernels_exist_and_are_competence_clear():
    src = strip_comments(SHADER.read_text())
    for name in PRODUCTION_KERNELS:
        assert f"kernel void {name}(" in src, name
    assert "ocol >> 6u" in src
    for name in (
        "shared_binary_k8_fused_stream_c5120_tpr64_tg128",
        "shared_binary_k8_fused_stream_c17408_tpr64_tg128",
        "shared_binary_k4_fused_stream_c5120_tpr64_tg128",
        "shared_binary_k16_fused_stream_c5120_tpr64_tg128",
    ):
        body = next(b for n, b in kernel_bodies(src) if n == name)
        r = screen_kernel(name, body, params_of(src, name))
        assert r["verdict"] == "CLEAR", f"{name} {r['verdict']} {r['findings']}"
        assert "constant uint& rows" not in params_of(src, name)
        assert "constant uint& group_size" not in params_of(src, name)


def test_receipt_reports_full_model_operating_point():
    assert RECEIPT.is_file(), (
        "receipts/headless/SHARED_BASIS_COHERENT.json missing — "
        "run python3 tools/headless/shared_basis_coherent.py"
    )
    doc = json.loads(RECEIPT.read_text())
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["did_not_mutate_noetic_parent_a"] is True
    assert doc["dense_w"] == 0
    assert doc["dense_w_materialized"] == 0
    assert doc["competent"] is True
    op = doc["operating_point"]
    assert op["k"] in K_SWEEP or op["k"] == K_MAX
    assert op["active_bytes_per_token"] > 0
    assert op["complete_ebpw"] > 0
    ns = doc["COMPLETE_TOKEN_NS"]
    assert ns["reps"] >= 7
    assert ns["min"] <= ns["median"] <= ns["max"] or ns.get("mlp_graph_gpu_ns", {}).get("n", 0) >= 7
    gpu = ns["mlp_graph_gpu_ns"]
    assert gpu.get("n", 0) >= 7
    assert doc["parity"]["ok"] is True
    assert doc["parity"]["noop_diverges"] is True
    ctrl = doc["controls"]
    assert ctrl["label"] in {"SEPARATED", "NOT SEPARATED"}
    assert "rung" in doc["composition_ladder"]
    assert "token_ids" in doc
    assert isinstance(doc["coherent_shared_basis_beats_q2f"], bool)
    assert doc["finding"]["reason"]
    assert doc["q2f_baseline"]["complete_token_ns"] == Q2F_COMPLETE_NS
    assert doc["q2f_baseline"]["bpw"] == Q2F_BPW
    # Either a coherent point below both bars, or a measured reason not.
    if doc["coherent_shared_basis_beats_q2f"]:
        assert op["below_q2f_bpw"] is True
        assert op["below_q2f_ns"] is True
        assert op["coherent"] is True
        assert doc["composition_ladder"]["rung"] == "coherent_generation"
    else:
        assert "No coherent shared-basis point beats q2f" in doc["answer"] or (
            "not coherent" in doc["finding"]["reason"]
            or "is not below" in doc["finding"]["reason"]
        )
    islands = doc["protected_islands"]
    assert "marginal" in islands
    assert islands["shared_k"] == 2
    assert islands["marginal"][0]["n_protected_tensors"] == 0

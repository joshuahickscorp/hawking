"""BYTES_FRONTIER: native sub-2.25-bpw MLP representations, no dense W."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from bytes_frontier import (  # noqa: E402
    GROUP,
    KERNELS,
    MLP_ELEMENTS,
    Q2F_BPW,
    RECEIPT,
    SCHEMA,
    SHADER,
    TRIT_PACK,
    bpw_binary,
    bpw_q2f,
    bpw_ternary,
    pack_ternary_5in8,
    reconstruct_ternary,
    residual_bytes,
    shared_k2_bytes,
    shader_evidence,
)


def test_q2f_baseline_is_2_25():
    assert abs(bpw_q2f() - 2.25) < 1e-12
    assert abs(bpw_q2f() - Q2F_BPW) < 1e-12


def test_ternary_5in8_g64_bills_1_85_not_one():
    assert abs(bpw_ternary() - (TRIT_PACK + 16.0 / GROUP)) < 1e-12
    assert abs(bpw_ternary() - 1.85) < 1e-12
    assert bpw_ternary() < Q2F_BPW
    assert bpw_ternary() != 1.0


def test_binary_g64_bills_1_25():
    assert abs(bpw_binary() - 1.25) < 1e-12
    assert bpw_binary() < Q2F_BPW


def test_shared_k2_active_bpw_is_below_2_25_and_below_independent_binary():
    sh = shared_k2_bytes()
    assert sh["active_bpw"] < Q2F_BPW
    assert sh["active_bpw"] < bpw_binary()
    assert sh["basis_sign_bytes"] > 0
    assert sh["scale_bytes"] > sh["basis_sign_bytes"]
    # Amortised signs; coefficients do not vanish at 64 layers.
    assert sh["active_bytes"] / MLP_ELEMENTS * 8 > 0.4


def test_residual_2pct_bills_indices_and_is_not_a_free_lunch():
    res = residual_bytes(0.02)
    bin_bytes = MLP_ELEMENTS * bpw_binary() / 8.0
    assert res["csr_bytes"] > 0
    assert res["active_bytes"] > bin_bytes
    # 2% u32+f16 on 17e9 elems is a lot of index bytes; may exceed 2.25.
    assert res["nnz_frac"] == 0.02


def test_ternary_5in8_roundtrip_and_is_not_deletion():
    rng = np.random.RandomState(0)
    w = rng.randn(8, 320).astype(np.float32)
    codes, scales, acc = pack_ternary_5in8(w, 64)
    assert acc["scales_counted"] is True
    assert abs(acc["storage_bpw"] - 1.85) < 1e-12
    assert acc["zero_bytes_skipped"] is False
    recon = reconstruct_ternary(codes, scales, 320, 64)
    assert recon.shape == w.shape
    assert np.isfinite(recon).all()
    assert np.count_nonzero(recon) > 0
    rel = float(np.linalg.norm(recon - w) / np.linalg.norm(w))
    assert rel < 0.7
    # Codes are a 5-in-8 pack, not 2-bit.
    assert codes.dtype == np.uint8
    assert codes.max() <= 242  # 3^5 - 1


def test_ternary_refuses_ragged_cols():
    w = np.ones((4, 100), dtype=np.float32)
    with pytest.raises(Exception):
        pack_ternary_5in8(w, 64)


def test_native_kernels_exist_and_are_group64_shifts():
    evid = shader_evidence()
    assert evid["all_present"], evid["kernels_present"]
    src = SHADER.read_text()
    for k in KERNELS:
        assert f"kernel void {k}(" in src
    assert evid["no_bind_time_group_size_in_ternary_geo"]
    assert evid["uses_shift_not_div_for_group"]
    assert "col / group_size" not in src
    # Production geo kernels must not take bind-time group_size.
    for name in (
        "ternary_5in8_g64_matvec_geo_c5120_tpr64_tg128",
        "binary_g64_matvec_geo_c5120_tpr64_tg128",
        "q2f_g64_matvec_geo_c5120_tpr64_tg128",
        "shared_binary_k2_group_dots_c5120_g64_tpr64_tg128",
        "binary_sparse_fused_geo_c5120_tpr64_tg128",
    ):
        a = src.find(f"kernel void {name}(")
        assert a >= 0
        b = src.find("kernel void ", a + 10)
        body = src[a : b if b > a else None]
        assert "constant uint& group_size" not in body


def test_receipt_reports_two_cheaper_reps_complete_token_and_dense_w():
    assert RECEIPT.is_file(), (
        "receipts/headless/BYTES_FRONTIER.json missing — "
        "run python3 tools/headless/bytes_frontier.py"
    )
    doc = json.loads(RECEIPT.read_text())
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["did_not_mutate_noetic_parent_a"] is True
    assert doc["dense_w_materialized"] == 0
    assert doc["dense_w_is_a_counter"] is True
    assert abs(doc["n021_baseline"]["bpw"] - 2.25) < 1e-12
    reps = doc["representations"]
    cheaper = [r for r in reps if r.get("lower_than_q2f_2_25")]
    assert len(cheaper) >= 2, "need >=2 representations below 2.25 active bpw"
    for r in cheaper:
        assert r["active_bytes_per_token"] > 0
        assert r["dram_bytes_per_token"] > r["active_bytes_per_token"]
        ct = r["COMPLETE_TOKEN_NS"]
        gpu = ct["mlp_graph_gpu_ns"]
        assert gpu.get("n", 0) >= 7
        assert gpu.get("min") is not None
        assert gpu.get("median") is not None
        assert gpu.get("max") is not None
        assert gpu["min"] <= gpu["median"] <= gpu["max"]
        assert ct.get("median") is not None
        assert r["dense_w_materialized"] == 0
        assert "rung" in r["coherence"]
        assert r["parity"]["ok"] is True
        assert "toward_roof_729_7" in r
        assert "moved" in r["toward_roof_729_7"]
    assert doc["kernel_competence"]["any_geo_defective"] is False
    assert isinstance(doc["finding"]["fewer_bytes_moved_token_ns_toward_729_7"], bool)
    assert doc["finding"]["per_representation"]

"""NOETIC_FUSED_SUBBIT: 2-bit affine MLP on a fused operator graph."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from first_noetic_executable import Q4_INCUMBENT_EBPW  # noqa: E402
from noetic_fused_subbit import (  # noqa: E402
    AFFINE2_EBPW,
    BEFORE_DISPATCHES,
    INCUMBENT_TOK_S,
    KERNELS,
    MIXED_SHADER,
    Q4_SHADER,
    RECEIPT,
    SCHEMA,
    shader_evidence,
    theoretical_after,
)

RECEIPT_DOC = None


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        assert RECEIPT.is_file(), (
            f"missing {RECEIPT} — run python3 tools/headless/noetic_fused_subbit.py"
        )
        RECEIPT_DOC = json.loads(RECEIPT.read_text())
    return RECEIPT_DOC


def test_theoretical_fusion_cuts_dispatches_below_964():
    assert theoretical_after("pair", False, False) == 900
    assert theoretical_after("swiglu", False, False) == 836
    assert theoretical_after("swiglu", True, True) == 756
    assert theoretical_after("swiglu", True, True) < BEFORE_DISPATCHES
    assert BEFORE_DISPATCHES == 964


def test_fused_affine2_kernels_exist_and_are_wired():
    ev = shader_evidence()
    assert ev["shader_present"], f"missing {MIXED_SHADER}"
    assert ev["all_kernels_declared"], ev["kernel_needles"]
    assert ev["wired_in_encode_dense_mlp_mixed"]
    assert ev["wired_in_encode_gqa_mixed"]
    assert ev["wired_in_encode_deltanet_mixed"]
    assert ev["production_964_untouched"]
    assert ev["specialized_g64_shift"]
    assert ev["runtime_div_kept_as_diagnostic"]
    mixed = MIXED_SHADER.read_text(encoding="utf-8")
    q4 = Q4_SHADER.read_text(encoding="utf-8") if Q4_SHADER.is_file() else ""
    combined = mixed + "\n" + q4
    for name in KERNELS:
        assert f"kernel void {name}(" in combined, name


def test_receipt_schema_ebpw_and_no_second_27b():
    doc = receipt()
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_write_under_models"] is True
    rep = doc["representation"]
    assert abs(rep["complete_ebpw"] - AFFINE2_EBPW) < 1e-6
    assert abs(rep["q4_incumbent_complete_physical_bpw"] - Q4_INCUMBENT_EBPW) < 1e-9
    assert abs(Q4_INCUMBENT_EBPW - 4.252735126866492) < 1e-9
    assert rep["n_affine"] == 192
    assert rep["not_the_incumbent"] is True
    assert doc["operator_graph"]["fused"] is True
    assert doc["operator_graph"]["not_the_source"] is True


def test_receipt_reports_dispatches_tok_s_16_tokens():
    doc = receipt()
    disp = doc["dispatches_per_token"]
    assert disp["before"] == BEFORE_DISPATCHES or disp["theoretical"]["unfused"] == BEFORE_DISPATCHES
    after = disp.get("measured_after") if disp.get("measured_after") is not None else disp["after"]
    before = disp.get("measured_before") if disp.get("measured_before") is not None else disp["before"]
    assert after < before or after < BEFORE_DISPATCHES
    assert disp["command_buffers"] == 1
    assert "TokenCommandBuffer.dispatch_count" in doc["counting_method"]
    tok = doc["decode_tok_s"]
    assert tok["incumbent"] == INCUMBENT_TOK_S or abs(tok["incumbent"] - INCUMBENT_TOK_S) < 1e-6
    after_arm = tok.get("after") or {}
    assert after_arm.get("tok_s_mean") is not None or after_arm.get("tok_s_reps")
    ids = after_arm.get("new_token_ids") or []
    text = after_arm.get("generated_text_verbatim")
    assert len(ids) == 16, f"expected 16 tokens, got {ids!r}"
    assert isinstance(text, str)
    verbatim = doc["verbatim"]
    assert verbatim["prompt"]
    assert verbatim["after"]["new_token_ids"] == ids
    assert verbatim["after"]["generated_text"] == text


def test_receipt_parity_dense_parent_and_every_config():
    doc = receipt()
    par = doc["parity"]
    mlp = par.get("mlp_gate_up_swiglu") or {}
    assert "max_abs_diff" in mlp or par.get("max_abs_diff") is not None
    dense = doc["dense_parent"]
    assert dense["dense_w_materialized"] == 0
    assert dense["expanded_to_q4"] == 0
    assert dense["expanded_to_float_gemv"] == 0
    tried = doc["configs_tried"]
    assert len(tried) >= 2
    names = " ".join(c.get("id", "") for c in tried).lower()
    assert "unfused" in names
    assert "swiglu" in names
    why = doc["why_affine2_g64_was_slower"]
    assert why.get("status") in {"measured", "not_measured"}
    if why.get("status") == "measured":
        assert why.get("per_dispatch")
        gate = next(
            (r for r in why["per_dispatch"] if r.get("label") == "gate_up"),
            why["per_dispatch"][0],
        )
        assert "q4_g64_gpu_ns" in gate
        assert "affine2_g64_specialized_gpu_ns" in gate
        assert "affine2_g64_runtime_div_gpu_ns" in gate
    assert doc["verdict"]
    assert set(KERNELS) <= set(doc["kernels"])
    assert doc["gpu_ran"] is True, doc.get("verdict")

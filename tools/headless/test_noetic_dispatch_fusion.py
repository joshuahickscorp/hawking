"""Dispatch fusion: dispatches per token BEFORE/AFTER, tok/s, 16 tokens, parity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from noetic_dispatch_fusion import (  # noqa: E402
    BEFORE_DISPATCHES,
    KERNELS,
    RECEIPT,
    SCHEMA,
    SHADER,
    theoretical_after,
    shader_evidence,
)

RECEIPT_DOC = None


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        assert RECEIPT.is_file(), (
            f"missing {RECEIPT} — run python3 tools/headless/noetic_dispatch_fusion.py"
        )
        RECEIPT_DOC = json.loads(RECEIPT.read_text())
    return RECEIPT_DOC


def test_theoretical_fusion_cuts_dispatches_below_964():
    assert theoretical_after("pair", False, False) == 900
    assert theoretical_after("swiglu", False, False) == 836
    assert theoretical_after("swiglu", True, True) == 756
    assert theoretical_after("swiglu", True, True) < BEFORE_DISPATCHES
    assert BEFORE_DISPATCHES == 964


def test_fused_kernels_exist_in_shader_and_are_wired():
    ev = shader_evidence()
    assert ev["shader_present"], f"missing {SHADER}"
    assert ev["all_kernels_declared"], ev["kernel_needles"]
    assert ev["wired_in_encode_dense_mlp"]
    assert ev["wired_in_encode_gqa"]
    assert ev["wired_in_encode_deltanet"]
    assert ev["production_964_untouched"]
    assert ev["workhorse_unchanged"]
    text = SHADER.read_text(encoding="utf-8")
    for name in KERNELS:
        assert f"kernel void {name}(" in text


def test_receipt_schema_and_counting_method():
    doc = receipt()
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_write_under_models"] is True
    disp = doc["dispatches_per_token"]
    assert disp["before"] == BEFORE_DISPATCHES or disp["theoretical"]["unfused"] == BEFORE_DISPATCHES
    assert disp["after"] < disp["before"] or (
        disp.get("measured_after") is not None
        and disp.get("measured_before") is not None
        and disp["measured_after"] < disp["measured_before"]
    )
    assert "TokenCommandBuffer.dispatch_count" in doc["counting_method"]
    assert disp["command_buffers"] == 1
    assert len(doc["fusions_attempted"]) >= 3
    assert set(KERNELS) <= set(doc["kernels"])


def test_receipt_reports_tok_s_spread_and_16_tokens():
    doc = receipt()
    assert doc["gpu_ran"] is True, doc.get("verdict")
    after = doc["decode_tok_s"]["after"]
    before = doc["decode_tok_s"]["before"]
    assert before is not None, "unfused decode missing"
    assert after is not None, "fused decode missing"
    assert before.get("tok_s_reps"), before
    assert after.get("tok_s_reps"), after
    ids = after.get("new_token_ids") or []
    text = after.get("generated_text_verbatim")
    assert len(ids) == 16, f"expected 16 tokens, got {ids!r}"
    assert isinstance(text, str)
    verbatim = doc["verbatim"]
    assert verbatim["prompt"]
    assert verbatim["after"]["new_token_ids"] == ids
    assert verbatim["after"]["generated_text"] == text


def test_receipt_parity_and_dense_parent_counters():
    doc = receipt()
    par = doc["parity"]
    mlp = par.get("mlp_gate_up_swiglu") or {}
    assert "max_abs_diff" in mlp or par.get("max_abs_diff") is not None
    dense = doc["dense_parent"]
    assert dense["dense_w_materialized"] == 0
    assert dense["expanded_to_q4"] == 0
    assert dense["expanded_to_float_gemv"] == 0


def test_receipt_reports_every_fusion_and_a_verdict():
    doc = receipt()
    attempted = " ".join(doc["fusions_attempted"]).lower()
    assert "swiglu" in attempted or "gate" in attempted
    assert "qkv" in attempted
    assert doc["verdict"]
    # A slower fusion is still a valid result; the receipt must say so.
    if doc["decode_tok_s"]["after"] and doc["decode_tok_s"]["before"]:
        a = doc["decode_tok_s"]["after"].get("tok_s_mean")
        b = doc["decode_tok_s"]["before"].get("tok_s_mean")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a < b:
            assert "SLOWER" in doc["verdict"] or "slower" in doc["verdict"].lower()

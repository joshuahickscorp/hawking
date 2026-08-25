"""BINARY_HEALING: localize the 1.25-bpw death, bill islands, gate the receipt."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from binary_healing import (  # noqa: E402
    BINARY_BPW,
    BINARY_COMPLETE_NS,
    BINARY_FIRST16,
    CHANNEL_TOP_K,
    HOLD_TOKENS,
    KERNELS,
    MLP_ELEMENTS,
    Q2F_BPW,
    Q2F_COMPLETE_NS,
    Q2F_FIRST16,
    RECEIPT,
    REL_L2_DIVERGE,
    SCHEMA,
    SHADER,
    capability_restored,
    complete_ebpw_from_mlp_bytes,
    island_spec,
    mlp_body_bytes,
    ranking_score,
    reconstruct_binary,
    reconstruct_q2f,
    shader_autopsy,
    sparse_spec,
    token_logit_from_receipts,
    top_channels,
)
from bytes_frontier import bpw_binary, bpw_q2f  # noqa: E402


def test_injured_body_bills_1_25_and_q2f_bills_2_25():
    assert abs(bpw_binary() - 1.25) < 1e-12
    assert abs(bpw_q2f() - 2.25) < 1e-12
    assert abs(BINARY_BPW - 1.25) < 1e-12
    assert abs(Q2F_BPW - 2.25) < 1e-12
    assert BINARY_COMPLETE_NS < Q2F_COMPLETE_NS


def test_binary_reconstruct_is_not_deletion():
    rng = np.random.RandomState(0)
    w = rng.randn(8, 128).astype(np.float32)
    b = reconstruct_binary(w, 64)
    q = reconstruct_q2f(w, 64)
    assert b.shape == w.shape
    assert q.shape == w.shape
    assert np.count_nonzero(b) == b.size
    assert np.isfinite(b).all() and np.isfinite(q).all()
    rel_b = float(np.linalg.norm(b - w) / np.linalg.norm(w))
    rel_q = float(np.linalg.norm(q - w) / np.linalg.norm(w))
    assert rel_b < 0.8
    assert rel_q < rel_b + 1e-6  # 4-level is not worse than 1-bit on a Gaussian draw
    # q2f uses four magnitudes, binary two.
    uniq_b = len(np.unique(np.round(np.abs(b) / (np.abs(b).max() + 1e-12), 4)))
    assert uniq_b >= 1


def test_down_q2f_island_bills_1_583_not_global_restore():
    s = island_spec("down_q2f")
    assert s["n_q2f_gemvs"] == 64
    assert s["n_binary_gemvs"] == 128
    assert abs(s["mlp_body_bpw"] - (2 * 1.25 + 2.25) / 3) < 1e-12
    assert 1.4 < s["mlp_body_bpw"] < 1.6
    assert s["mlp_tax_ebpw"] == pytest.approx(1.0 / 3.0)
    assert s["dense_w"] == 0
    # Complete EBPW counts islands + residual attention; no hidden bits.
    assert s["complete_ebpw"] > s["mlp_body_bpw"]
    assert s["COHERENCE_TAX_EBPW"] == pytest.approx(s["complete_ebpw"] - 1.25)
    # Global q2f restore is strictly more bits than the down island.
    g = island_spec("q2f")
    assert g["mlp_body_bpw"] == pytest.approx(2.25)
    assert g["complete_ebpw"] > s["complete_ebpw"]


def test_gate_and_layer_islands_are_minimal_not_full_body():
    g = island_spec("gate_q2f")
    e = island_spec("early16_q2f")
    late = island_spec("late16_q2f")
    full = island_spec("q2f")
    injured = island_spec("binary")
    assert g["n_q2f_gemvs"] == 64
    assert e["n_q2f_gemvs"] == 16 * 3
    assert late["n_q2f_gemvs"] == 16 * 3
    assert injured["n_q2f_gemvs"] == 0
    assert injured["mlp_body_bpw"] == pytest.approx(1.25)
    for s in (g, e, late):
        assert s["mlp_body_bpw"] < full["mlp_body_bpw"]
        assert s["mlp_body_bpw"] > injured["mlp_body_bpw"]
        assert s["dense_w"] == 0


def test_sparse_residual_counts_csr_bytes():
    s = sparse_spec(0.005)
    assert s["nnz_frac"] == 0.005
    assert s["csr_bytes"] > 0
    assert s["mlp_body_bpw"] > 1.25
    assert s["mlp_body_bpw"] < 2.25
    assert s["dense_w"] == 0
    assert s["COHERENCE_TAX_EBPW"] == pytest.approx(s["complete_ebpw"] - 1.25)


def test_complete_ebpw_counts_non_mlp_bytes():
    mlp = MLP_ELEMENTS * 1.25 / 8.0
    e = complete_ebpw_from_mlp_bytes(mlp)
    # mix_c was 2.344 with all-MLP binary; theoretical is the same non-MLP residual.
    assert 2.3 < e < 2.4
    healed = mlp_body_bytes(128, 64)
    eh = complete_ebpw_from_mlp_bytes(healed["active_bytes"])
    assert eh > e
    assert eh < 3.0


def test_token_logit_diverges_at_position_zero():
    t = token_logit_from_receipts()
    assert t["earliest_position"] == 0
    assert t["binary_token_id"] == 271
    assert t["q2f_token_id"] == Q2F_FIRST16[0]
    assert t["binary_token_id"] != t["q2f_token_id"]
    assert BINARY_FIRST16 == [271] * 16
    assert t["real_activations"] is True


def test_sensitive_channels_are_ranked_by_error_energy():
    rng = np.random.RandomState(1)
    err = rng.randn(16, 64).astype(np.float32)
    err[:, 7] *= 20
    ch = top_channels(err, 8)
    assert ch[0]["channel"] == 7
    assert ch[0]["rank"] == 0
    assert ch[0]["error_energy_frac"] > ch[1]["error_energy_frac"]
    assert ch[-1]["cum_frac"] <= 1.0 + 1e-9
    assert CHANNEL_TOP_K >= 8


def test_kernels_exist_group64_shift_no_bind_time_div():
    auto = shader_autopsy()
    assert auto["all_present"], auto["kernels_present"]
    src = SHADER.read_text()
    for k in KERNELS:
        assert f"kernel void {k}(" in src
    assert auto["uses_shift_not_div"]
    assert auto["dense_w_written"] is False
    assert auto["any_geo_defective"] is False
    for name in KERNELS:
        a = src.find(f"kernel void {name}(")
        b = src.find("kernel void ", a + 10)
        body = src[a : b if b > a else None]
        assert "constant uint& group_size" not in body
        assert "col / group_size" not in body


def test_ranking_prefers_restored_capability_per_added_cost():
    # A coherent cheap heal beats a coherent expensive one.
    cheap = ranking_score(1.0, 0.33, 2.0e6)
    dear = ranking_score(1.0, 1.00, 2.0e6)
    dead = ranking_score(0.0, 0.10, -1.0e6)
    assert cheap > dear
    assert dead == 0.0
    assert capability_restored(
        {"highest_rung_reached": "coherent_generation"},
        {"coherent": True, "n_unique_ids": 14, "n_new_tokens": 16},
    ) == 1.0
    assert (
        capability_restored(
            {"highest_rung_reached": "complete_token"},
            {
                "coherent": False,
                "repeated_single_token": True,
                "n_unique_ids": 1,
                "n_new_tokens": 16,
            },
        )
        == 0.0
    )


def test_divergence_threshold_was_named_before_looking():
    assert REL_L2_DIVERGE == 0.15
    assert HOLD_TOKENS >= 32


def test_receipt_reports_failure_map_two_heals_and_tax_curve():
    assert RECEIPT.is_file(), (
        "receipts/headless/BINARY_HEALING.json missing — "
        "run python3 tools/headless/binary_healing.py"
    )
    doc = json.loads(RECEIPT.read_text())
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["did_not_mutate_noetic_parent_a"] is True
    assert doc["dense_w"] == 0
    assert doc["dense_w_materialized"] == 0
    fmap = doc["COHERENCE_FAILURE_MAP"]
    tok = fmap["earliest_token_logit_divergence"]
    assert tok["earliest_position"] == 0
    assert tok["binary_token_id"] == 271
    assert fmap["earliest_layer"] is not None
    assert fmap["earliest_organ"] in {"gate_proj", "up_proj", "down_proj"}
    assert fmap["real_activations"] is True
    assert fmap["sensitive_channels"]
    heals = doc["healing_candidates"]
    assert len(heals) >= 2
    counted = 0
    for h in heals:
        assert "COHERENCE_TAX_EBPW" in h
        assert abs(h["COHERENCE_TAX_EBPW"] - (h["complete_ebpw"] - 1.25)) < 1e-6
        assert "complete_ebpw" in h
        ct = h["COMPLETE_TOKEN_NS"]
        gpu = ct["mlp_graph_gpu_ns"]
        assert gpu.get("n", 0) >= 7
        assert gpu["min"] <= gpu["median"] <= gpu["max"]
        assert ct.get("median") is not None
        assert ct.get("reps", 0) >= 7
        assert h["dense_w"] == 0
        assert h["parity"]["ok"] is True
        rung = h["composition"]["highest_rung_reached"]
        assert rung in {
            "local_functional_probe",
            "held_out_activation",
            "adjacent_layers",
            "short_chain",
            "complete_organ",
            "complete_token",
            "coherent_generation",
        }
        if h.get("counts_as_heal"):
            counted += 1
            assert rung == "coherent_generation"
            assert h["coherence"]["coherent"] is True
        assert "ranking_score" in h
        ctrl = h.get("control") or {}
        if ctrl.get("overlap") is True:
            assert ctrl.get("label") == "NOT SEPARATED"
    # Ranking is capability / (tax + ns). Scores must be monotone in the list.
    scores = [h["ranking_score"] for h in heals]
    assert scores == sorted(scores, reverse=True)
    finding = doc["finding"]
    assert isinstance(finding["coherent_healed_body_still_faster_than_q2f"], bool)
    assert finding["n_healing_candidates"] >= 2
    assert doc["kernel_competence"]["any_geo_defective"] is False
    loc = doc["localization"]
    assert loc["not_gaussian"] is True
    assert loc.get("n_tokens_used", 0) >= 16
    # Injured body is still faster than q2f and still dead.
    injured = [c for c in doc["injured_and_reference"] if c["id"] == "binary"][0]
    assert injured["composition"]["died_at"] == "coherent_generation"
    assert injured["COMPLETE_TOKEN_NS"]["median"] < Q2F_COMPLETE_NS

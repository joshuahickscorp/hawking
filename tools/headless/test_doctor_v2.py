"""Doctor v2: physical computation per organ, not bits per tensor.

The harness must write receipts/headless/DOCTOR_V2_PRESCRIPTION.json for at
least one REAL organ of the qualified parent, carrying measured functional
sensitivity, discovered shared structure, competing candidates with executable
bytes, route cost, estimated DRAM, expected token_ns, and a ranking by
expected verified gain per experiment cost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from noetic_doctor_v2 import (  # noqa: E402
    RECEIPT,
    grouped_codes_and_scales,
    grouped_recon,
    pack_codes,
    pack_grouped_tensor,
    prescribe,
    storage_pack_grouped,
)


def test_binary_codec_is_not_deletion():
    """bits=1 is a sign code. The absmax parameterization with qmax=0 is the zero tensor."""
    rng = np.random.RandomState(0)
    w = rng.randn(64, 1024).astype(np.float32)
    out = grouped_recon(w, bits=1, group=1024)
    assert np.count_nonzero(out) == out.size
    rel = np.linalg.norm(out - w) / np.linalg.norm(w)
    assert 0.50 <= rel <= 0.75, rel


def test_binary_g1024_storage_bpw_is_1_015625():
    pack = storage_pack_grouped(1, 1024, numel=5120 * 17408, rows=5120, cols=17408)
    assert abs(pack["storage_bpw"] - 1.015625) < 1e-12
    assert pack["fused_active_bytes"] == pack["storage_bytes"]
    assert pack["decoded_f16_active_bpw"] == 16.0


def test_error_falls_as_bits_rise():
    w = np.random.RandomState(1).randn(128, 64).astype(np.float32)
    rel = [
        float(np.linalg.norm(grouped_recon(w, b, 64) - w) / np.linalg.norm(w))
        for b in (1, 2, 3, 4)
    ]
    assert rel == sorted(rel, reverse=True), rel


def test_zero_bits_is_still_deletion():
    w = np.random.RandomState(2).randn(32, 64).astype(np.float32)
    assert not np.any(grouped_recon(w, 0, 64))


def test_pack_roundtrip_bytes_are_real():
    w = np.random.RandomState(1).randn(8, 64).astype(np.float32)
    blob = pack_grouped_tensor("t", w, bits=4, group=64)
    assert b"DOCV2PK1" in blob
    assert len(blob) > 64
    codes, _ = grouped_codes_and_scales(w, 4, 64)
    packed = pack_codes(codes, 4)
    assert len(packed) == (codes.size + 1) // 2


@pytest.fixture(scope="session")
def prescription():
    rec = prescribe()
    assert RECEIPT.is_file(), "harness did not write DOCTOR_V2_PRESCRIPTION.json"
    disk = json.loads(RECEIPT.read_text())
    assert disk["schema"] == rec["schema"]
    return disk


def test_receipt_written(prescription):
    assert RECEIPT.name == "DOCTOR_V2_PRESCRIPTION.json"
    assert prescription["schema"] == "hawking.headless.doctor_v2_prescription.v1"
    assert prescription["question"] == "what physical computation should this ORGAN perform"
    assert prescription["rejected_question"] == "how many bits should this tensor get"


def test_at_least_one_real_organ_of_qualified_parent(prescription):
    organs = prescription["organs"]
    assert len(organs) >= 1
    parent = prescription["qualified_parent"]
    assert "qwen3.8" in parent or "qwen38" in parent
    real = [o for o in organs if o.get("real_organ_of_qualified_parent")]
    assert real, "no REAL organ of the qualified parent"
    for o in real:
        assert o["parent_tensors"], o["organ_id"]
        assert all("language_model.layers" in n for n in o["parent_tensors"])


def test_capture_is_real_not_gaussian(prescription):
    cap = prescription["capture"]
    assert cap["gaussian_proxy_used"] is False
    assert cap["not_gaussian"] is True
    assert cap["not_llama_server"] is True
    assert prescription["live_27b_policy"]["did_not_load_second_27b"] is True
    assert prescription["live_27b_policy"]["did_not_contact_llama_server"] is True


def test_attention_and_mlp_get_different_prescriptions(prescription):
    organs = {o["kind"]: o for o in prescription["organs"]}
    assert "mlp_swiglu" in organs
    assert "attention_gqa" in organs
    mlp_p = organs["mlp_swiglu"]["prescription"]["physical_computation"]
    gqa_p = organs["attention_gqa"]["prescription"]["physical_computation"]
    assert mlp_p != gqa_p
    assert prescription["prescriptions_differ"] is True
    assert prescription["organs_need_different_prescriptions"] is True
    # Evidence the contract named.
    mlp_id = organs["mlp_swiglu"]["prescription"]["candidate_id"]
    gqa_id = organs["attention_gqa"]["prescription"]["candidate_id"]
    assert "lowrank" not in mlp_id
    assert "q3" not in gqa_id


def test_functional_sensitivity_measured_on_real_x(prescription):
    for o in prescription["organs"]:
        s = o["measured_functional_sensitivity"]
        assert s["gaussian_proxy_used"] is False
        assert "identity" in s or "q_proj_identity" in s
        assert s["scale_trap_rejected"] is True
        assert s["null"]
        trap = s.get("scaled_0p01_W") or s.get("q_proj_scaled_0p01_W")
        assert trap["gain"] < 0.05
        assert trap["cosine_null"] is not None
        # Linear maps (GQA q_proj) still exhibit cosine≈1 at 0.01*W — that is why
        # cosine is not the gate. SwiGLU is not linear, so organ cosine may drop;
        # gain is the rejection either way.
        if o["kind"] != "mlp_swiglu":
            assert trap["cosine"] > 0.99


def test_shared_structure_discovered_or_none(prescription):
    for o in prescription["organs"]:
        sh = o["discovered_shared_structure"]
        finding = sh.get("finding") or sh.get("shipping_conclusion")
        assert finding, o["organ_id"]
        assert sh["gaussian_proxy_used"] is False
        assert "g035" in json.dumps(sh).lower() or "NONE" in sh.get("shipping_conclusion", "")
        # A finding of NONE is a result. G035 already recorded shared_beats_independent=false.
        priors = sh.get("priors") or {}
        if "g035_shared_beats_independent" in priors:
            assert priors["g035_shared_beats_independent"] is False


def test_at_least_three_candidates_with_executable_bytes(prescription):
    for o in prescription["organs"]:
        cands = o["candidates"]
        assert len(cands) >= 3, o["organ_id"]
        for c in cands:
            ex = c["bytes"]["executable_bytes"]
            assert ex["materialized"] is True
            assert ex["n_bytes"] > 0
            assert len(ex["sha256"]) == 64
            assert ex["magic"] == "DOCV2PK1"
            assert c["bytes"]["storage_bytes"] > 0
            assert c["bytes"]["fused_active_bytes"] > 0
            assert c["physical_computation"]


def test_storage_and_active_reported_separately(prescription):
    assert prescription["accounting_discipline"]["storage_and_active_reported_separately"] is True
    for o in prescription["organs"]:
        for c in o["candidates"]:
            b = c["bytes"]
            assert "storage_bytes" in b and "fused_active_bytes" in b
            assert "decoded_f16_active_bytes" in b
            assert b["decoded_f16_active_bpw"] == 16.0
            assert b["storage_bpw"] != b["decoded_f16_active_bpw"]


def test_null_stated_for_every_quality_number(prescription):
    assert prescription["accounting_discipline"]["null_stated_for_every_quality_number"] is True
    for o in prescription["organs"]:
        for c in o["candidates"]:
            q = c["quality"]
            assert q["null_kind"]
            assert "cosine_null" in q
            assert "rel_fro_null" in q
            assert "gain_null" in q
            assert "scale_aware_null" in q


def test_route_cost_dram_token_ns(prescription):
    for o in prescription["organs"]:
        assert "token_ns_incumbent" in o
        assert o["token_ns_incumbent"]["ns_per_token"] > 0
        for c in o["candidates"]:
            assert "route_cost" in c
            assert "estimated_dram" in c
            assert c["estimated_dram"]["estimated_dram_bytes"] > 0
            assert "expected_token_ns" in c
            assert c["expected_token_ns"]["expected_token_ns"] > 0
            assert c["expected_token_ns"]["null"]


def test_ranking_is_gain_per_cost_not_quality_alone(prescription):
    for o in prescription["organs"]:
        rule = o["ranking_rule"]
        assert rule["sorts_by"] == "expected_verified_gain_per_experiment_cost"
        assert rule["quality_order_coincided"] is False, (
            f"{o['organ_id']} ranked by quality alone — that is a wishlist"
        )
        ranks = o["ranking"]
        assert ranks == sorted(ranks, key=lambda r: r["rank"])
        gpc = [r["gain_per_experiment_cost"] for r in ranks]
        assert gpc == sorted(gpc, reverse=True)
        # Top experiment is not the incumbent (gain 0).
        assert ranks[0]["id"] != "incumbent_q4_g64"


def test_grounded_in_named_receipts(prescription):
    prior = prescription["prior_science"]
    assert prior["attention_floor"]["organ_floor_moved_below_4.125"] is False
    assert prior["composition"]["operator_class_beats_bit_count"] is True
    assert abs(prior["composition"]["binary_g1024_bpw"] - 1.015625) < 1e-9
    assert prior["g035"]["shared_beats_independent_any_pair"] is False
    g034 = prior["g034"]["lowrank_over_flat_q3_error"]
    assert g034 is not None and 2.7 < g034 < 3.2
    q80 = prior["q80"]["gate_proj_pairwise_cosine_mean"]
    assert q80 is not None and abs(q80 - 0.004) < 0.002

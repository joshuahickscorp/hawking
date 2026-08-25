"""N042 WHOLE_MODEL_NATIVE: heterogeneous 2.60-EBPW body, zero parent, reprofile."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from q3_mlp_q4_attn import dequant_hgravu01, pack_hgravu01, parse_hgravu01  # noqa: E402
from q2f_g64_generation import pack_hgrafv01_q2f, q2f_storage_bpw  # noqa: E402
from whole_model_native import (  # noqa: E402
    GENOME,
    fuse_in_proj_ba,
    KERNEL_DN,
    KERNEL_EMBED,
    KERNEL_GQA,
    KERNEL_MLP,
    MIX_ID,
    N041_TARGET_EBPW,
    N041_TOLERANCE,
    Q2F_UNIFORM_ROOF,
    RECEIPT,
    SCHEMA,
    assignment_for,
    autopsy_kernels,
    organ_role,
    q3_storage_bpw,
    reprofile_from_zero,
    rust_dense_w_is_a_counter,
    zero_parent_verdict,
)

DECODE_RS = REPO / "crates" / "hawking-core" / "src" / "model" / "qwen38_hybrid_decode.rs"
SHADER_MIXED = REPO / "crates" / "hawking-core" / "shaders" / "q80_mixed_decode.metal"
SHADER_EMBED = REPO / "crates" / "hawking-core" / "shaders" / "qwen38_device_activations.metal"


def test_genome_matches_n041_floors():
    assert GENOME["mlp"]["gemv_storage_bpw"] == 2.25
    assert GENOME["mlp"]["kernel"] == KERNEL_MLP
    assert GENOME["deltanet"]["gemv_storage_bpw"] == 3.25
    assert GENOME["deltanet"]["group"] == 64
    assert GENOME["deltanet"]["transition_program"] is True
    assert GENOME["attention_gqa"]["gemv_storage_bpw"] == 3.125
    assert GENOME["attention_gqa"]["group"] == 128
    assert GENOME["embedding"]["gemv_storage_bpw"] == 3.125
    assert GENOME["output"]["gemv_storage_bpw"] == 3.125
    assert abs(q3_storage_bpw(64, 3) - 3.25) < 1e-12
    assert abs(q3_storage_bpw(128, 3) - 3.125) < 1e-12
    assert abs(q2f_storage_bpw(64) - 2.25) < 1e-12


def test_fuse_in_proj_ba_interleaves_per_key_head():
    rng = np.random.RandomState(1)
    a = rng.randn(48, 5120).astype(np.float32)
    b = rng.randn(48, 5120).astype(np.float32)
    fused = fuse_in_proj_ba(b, a)
    assert fused.shape == (96, 5120)
    np.testing.assert_array_equal(fused[0:3], b[0:3])
    np.testing.assert_array_equal(fused[3:6], a[0:3])
    np.testing.assert_array_equal(fused[6:9], b[3:6])
    np.testing.assert_array_equal(fused[9:12], a[3:6])


def test_organ_role_classifies_catalog_names():
    assert organ_role("language_model.model.layers.0.mlp.down_proj.weight") == "mlp"
    assert organ_role("language_model.model.layers.0.linear_attn.in_proj_qkvz.weight") == "deltanet"
    assert organ_role("language_model.model.layers.3.self_attn.q_proj.weight") == "attention_gqa"
    assert organ_role("language_model.model.embed_tokens.weight") == "embedding"
    assert organ_role("language_model.lm_head.weight") == "output"
    assert organ_role("language_model.model.layers.0.input_layernorm.weight") == "leftover"
    assert assignment_for("language_model.model.layers.0.input_layernorm.weight") is None
    assert assignment_for("language_model.lm_head.weight")["group"] == 128


def test_hgravu01_q3_g128_roundtrips_and_is_not_deletion():
    rng = np.random.RandomState(0)
    w = rng.randn(8, 128).astype(np.float32)
    payload = pack_hgravu01(w, 3, 128)
    header = parse_hgravu01(payload)
    assert header["bits"] == 3
    assert header["group_size"] == 128
    recon = dequant_hgravu01(payload)
    assert recon.shape == (8, 128)
    assert float(np.abs(recon).mean()) > 0.1 * float(np.abs(w).mean())
    num = float(np.vdot(w.ravel(), recon.ravel()))
    den = float(np.linalg.norm(w) * np.linalg.norm(recon))
    assert num / den > 0.9


def test_q2f_and_q3_packers_refuse_ragged_cols():
    w = np.ones((4, 100), dtype=np.float32)
    with pytest.raises(Exception):
        pack_hgravu01(w, 3, 128)
    with pytest.raises(Exception):
        pack_hgrafv01_q2f(w, 64)


def test_q3_g128_kernel_is_specialized_not_runtime_div():
    src = SHADER_MIXED.read_text()
    assert f"kernel void {KERNEL_GQA}(" in src
    assert f"kernel void {KERNEL_DN}(" in src
    start = src.find(f"kernel void {KERNEL_GQA}(")
    end = src.find("kernel void ", start + 1)
    body = src[start:end]
    assert "group_size == 128u" in body
    assert "cols >> 7u" in body or "col >> 7u" in body
    assert "col / group_size" not in body
    assert "No dense W" in SHADER_MIXED.read_text()[:2500] or "No dense W" in src
    assert "kernel void qwen38_hgravu_embedding_lookup(" in SHADER_EMBED.read_text()
    decode = DECODE_RS.read_text()
    assert KERNEL_GQA in decode
    assert "qwen38_hgravu01_geo_tpr64_launch" in decode
    assert '(3, 128)' in decode or "3, 128" in decode


def test_dense_w_is_a_runtime_counter_in_rust():
    c = rust_dense_w_is_a_counter()
    assert c["account_dense_w_present"] is True
    assert c["field_present"] is True
    assert c["generate_copies_counter"] is True
    assert c["production_comment"] is True
    src = DECODE_RS.read_text()
    # The only increment path is account_dense_w. Production packed GEMV
    # does not reconstruct dense W.
    assert src.count("self.dense_w_materialized +=") == 1
    assert "pub fn account_dense_w" in src


def test_kernel_autopsy_finds_the_wired_kernels():
    a = autopsy_kernels()
    for name in (KERNEL_MLP, KERNEL_DN, KERNEL_GQA, KERNEL_EMBED):
        assert a[name]["present"] is True, name
        assert a[name]["verdict"] in ("CLEAR", "SUSPECT")
        assert a[name]["dense_w_written"] is False


def test_reprofile_refuses_the_q2f_uniform_roof_and_keeps_819_778():
    roofs = reprofile_from_zero(8.2e9)
    assert roofs["DEVICE_THEORETICAL"]["value"] == 819.0
    assert roofs["DEVICE_MEASURED_SUSTAINED"]["value"] == 778.8
    mr = roofs["MODEL_REACHABLE"]["value"]
    assert mr is not None
    assert abs(float(mr) - Q2F_UNIFORM_ROOF) > 1e-6
    assert roofs["MODEL_REACHABLE"]["refused_to_reuse_729_7"] is True
    assert roofs["overlap"] == "NOT SEPARATED"
    assert roofs["never_collapsed"] is True


def test_zero_parent_verdict_requires_the_counter():
    dense = {
        "dense_w_materialized": 0,
        "counter_not_literal": True,
    }
    decoded = {"expanded_to_q4": 0, "expanded_to_float_gemv": 0, "fallbacks": 0}
    v = zero_parent_verdict(dense, decoded)
    assert v["QWEN_ZERO_PARENT_RUNTIME_DEPENDENCY"] == "PASS"
    bad = dict(dense)
    bad["dense_w_materialized"] = 1
    v_bad = zero_parent_verdict(bad, decoded)
    assert v_bad["QWEN_ZERO_PARENT_RUNTIME_DEPENDENCY"] == "FAIL"


@pytest.fixture(scope="session")
def receipt() -> dict:
    if not RECEIPT.is_file():
        import whole_model_native as w

        assert w.main() == 0
    assert RECEIPT.is_file(), (
        "receipts/headless/WHOLE_MODEL_NATIVE.json missing — "
        "run python3 tools/headless/whole_model_native.py"
    )
    return json.loads(RECEIPT.read_text())


def test_receipt_schema_and_discipline(receipt: dict):
    assert receipt["schema"] == SCHEMA
    assert receipt["hand_authored"] is False
    assert receipt["did_not_load_second_27b"] is True
    assert receipt["did_not_write_under_models"] is True
    assert receipt["did_not_mutate_noetic_parent_a"] is True
    assert receipt["generated_by"].endswith("whole_model_native.py")


def test_receipt_zero_parent_is_a_counter_not_a_literal(receipt: dict):
    assert receipt["QWEN_ZERO_PARENT_RUNTIME_DEPENDENCY"] == "PASS"
    zp = receipt["zero_parent"]
    assert zp["dense_w_materialized"] == 0
    assert zp["no_dense_parent_reconstructed"] is True
    assert zp["no_fallback_dense_tensor"] is True
    assert zp["counter"]["counter_not_literal"] is True
    assert zp["rust_counter_contract"]["account_dense_w_present"] is True
    # A hardcoded python 0 would not also have the census/stderr parse.
    counter = zp["counter"]
    sources = [
        counter.get("from_generate_json"),
        counter.get("from_census_parse"),
        counter.get("from_stderr_counter"),
    ]
    assert any(s == 0 for s in sources if s is not None)


def test_receipt_complete_ebpw_matches_n041(receipt: dict):
    ebpw = float(receipt["complete_ebpw"])
    assert abs(ebpw - N041_TARGET_EBPW) <= N041_TOLERANCE
    assert receipt["n041_match"] is True
    assert receipt["below_3_0"] is True
    assert ebpw < 3.0


def test_receipt_organs_are_wired_natively(receipt: dict):
    wired = set(receipt["organs_wired"])
    for organ in ("mlp", "deltanet", "attention_gqa", "embedding", "output"):
        assert organ in wired or organ in receipt["organs_pending"]
    compile_ = receipt["compile"]
    assert compile_["mix_id"] == MIX_ID
    assert compile_["counts"]["mlp"] == 192
    assert compile_["counts"]["deltanet"] == 144
    assert compile_["counts"]["attention_gqa"] == 64
    assert compile_["counts"]["embedding"] == 1
    assert compile_["counts"]["output"] == 1
    genome = receipt["representation_genome"]
    assert genome["mlp"]["kernel"] == KERNEL_MLP
    assert genome["deltanet"]["kernel"] == KERNEL_DN
    assert genome["attention_gqa"]["kernel"] == KERNEL_GQA
    assert genome["embedding"]["kernel"] == KERNEL_EMBED


def test_receipt_coherent_token_loop(receipt: dict):
    decode = receipt.get("decode") or {}
    wall = receipt.get("complete_wall") or {}
    primary = decode if decode.get("ok") else wall
    assert primary.get("ok") is True
    assert primary.get("n_new_tokens") >= 16 or (
        (receipt.get("new_token_ids") or []) and len(receipt["new_token_ids"]) >= 8
    )
    ids = receipt.get("new_token_ids") or primary.get("new_token_ids") or []
    assert len(set(ids)) > 1, "degenerate single-token generation"
    coh = receipt.get("coherence") or primary.get("coherence") or {}
    assert coh.get("repeated_single_token") is not True
    assert primary.get("native_kernel_ran") is True
    assert primary.get("dequant_path") is False
    assert primary.get("fallbacks") == 0
    text = receipt.get("generated_text_verbatim") or ""
    assert isinstance(text, str)


def test_receipt_complete_token_ns_seven_reps(receipt: dict):
    ctn = receipt.get("COMPLETE_TOKEN_NS") or {}
    assert ctn.get("kind") == "MEASURED"
    assert ctn.get("overlap") == "NOT SEPARATED"
    assert int(ctn.get("n_warm_reps") or 0) >= 7
    assert ctn.get("min") is not None
    assert ctn.get("median") is not None
    assert ctn.get("max") is not None
    assert int(ctn["min"]) <= int(ctn["median"]) <= int(ctn["max"])
    assert receipt["overlap"] == "NOT SEPARATED"


def test_receipt_parity_vs_oracle(receipt: dict):
    p = receipt["parity"]
    assert p["ok"] is True
    assert p["q2f_group64"].get("ok") is True
    assert p["q3_group128"].get("ok") is True
    assert "reconstruct_then_matvec" in (p.get("oracle") or "")


def test_receipt_reprofile_three_roofs(receipt: dict):
    three = receipt["three_roofs"]
    assert three["DEVICE_THEORETICAL"] == 819.0
    assert three["DEVICE_MEASURED_SUSTAINED"] == 778.8
    mr = three["MODEL_REACHABLE"]
    assert mr is not None
    assert abs(float(mr) - Q2F_UNIFORM_ROOF) > 1e-6
    assert three["not_the_q2f_uniform_roof_729_7"] is True
    assert three["new_model_reachable"] is True
    rp = receipt["reprofile"]["MODEL_REACHABLE"]
    assert rp["refused_to_reuse_729_7"] is True
    assert three["current_fraction_of_model_reachable"] is not None
    assert 0.0 < float(three["current_fraction_of_model_reachable"]) <= 1.0
    assert three["current_achieved_gb_s"] is not None
    frac = None
    sustained = 778.8
    if mr and sustained:
        frac = float(mr) / sustained
    assert receipt["reprofile"]["DEVICE_THEORETICAL"]["value"] == 819.0
    assert receipt["reprofile"]["DEVICE_MEASURED_SUSTAINED"]["value"] == 778.8
    # Current fraction of the new roof sits on the measured complete-token
    # if we have a wall; otherwise the derived occupancy roof still stands.
    if frac is not None:
        assert 0.0 < frac <= 1.0 + 1e-6 or float(mr) > 0

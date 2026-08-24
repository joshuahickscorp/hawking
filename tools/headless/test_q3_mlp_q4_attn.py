"""Q3-MLP g64 + Q4-attention mix: packer contract and native-decode receipt."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from first_noetic_executable import (  # noqa: E402
    PARENT_PARAMS,
    Q4_INCUMBENT_EBPW,
    judge_coherence,
    write_catalog,
)
from q3_mlp_q4_attn import (  # noqa: E402
    BITS_Q3,
    GROUP_Q3,
    MAGIC_UNIFORM,
    MIX_ID,
    NATIVE_Q3_KERNEL,
    RECEIPT,
    SCHEMA,
    SCHEMA_UNIFORM,
    dequant_hgravu01,
    extract_unsigned,
    is_mlp_proj,
    mix_recipe,
    pack_hgravu01,
    pack_unsigned_lsb,
    parse_census,
    parse_hgravu01,
    q3_storage_bpw,
)


def test_q3_g64_storage_bpw_is_3_25():
    assert abs(q3_storage_bpw(64, 3) - 3.25) < 1e-12
    assert q3_storage_bpw() < Q4_INCUMBENT_EBPW
    assert q3_storage_bpw() > 3.0


def test_lsb_3bit_pack_matches_extract_unsigned():
    codes = np.arange(8, dtype=np.uint8) % 7
    packed = pack_unsigned_lsb(codes, 3)
    assert len(packed) == 3
    recovered = [extract_unsigned(packed, i, 3) for i in range(8)]
    assert recovered == codes.tolist()


def test_hgravu01_roundtrips_and_is_not_deletion():
    rng = np.random.RandomState(0)
    w = rng.randn(8, 64).astype(np.float32)
    payload = pack_hgravu01(w, 3, 64)
    assert payload[:8] == MAGIC_UNIFORM
    header = parse_hgravu01(payload)
    assert header["schema"] == SCHEMA_UNIFORM
    assert header["shape"] == [8, 64]
    assert header["bits"] == 3
    assert header["group_size"] == 64
    assert header["groups"] == 8
    assert header["bound"] == 3
    recon = dequant_hgravu01(payload)
    assert recon.shape == (8, 64)
    # Grouped absmax q3 is a real code, not a zero tensor.
    assert float(np.abs(recon).mean()) > 0.1 * float(np.abs(w).mean())
    # Cosine should be high on this small random draw.
    num = float(np.vdot(w.ravel(), recon.ravel()))
    den = float(np.linalg.norm(w) * np.linalg.norm(recon))
    assert den > 0
    assert num / den > 0.95


def test_hgravu01_refuses_ragged_cols():
    w = np.ones((4, 100), dtype=np.float32)
    with pytest.raises(Exception):
        pack_hgravu01(w, 3, 64)


def test_catalog_magic_for_hgravu01(tmp_path: Path):
    payload = pack_hgravu01(np.ones((4, 64), dtype=np.float32), 3, 64)
    dest = tmp_path / "t.hgravu01"
    dest.write_bytes(payload)
    rec = {
        "name": "language_model.model.layers.0.mlp.down_proj.weight",
        "codec": 3,
        "organ": 2,
        "shape": [4, 64],
        "elements": 256,
        "segment_id": 0,
        "offset": 0,
        "nbytes": len(payload),
        "sha256": "ab" * 32,
        "codec_bpw": 3.25,
    }
    seg = {
        "id": 0,
        "filename": "t.hgravu01",
        "bytes": len(payload),
        "sha256": "ab" * 32,
    }
    blob = write_catalog(tmp_path / "catalog.hq38m20", [rec], [seg])
    assert blob[:8] == b"HQ38M20\0"
    version, n_tensors, n_segments = struct.unpack_from("<III", blob, 8)
    assert version == 1
    assert n_tensors == 1
    assert n_segments == 1


def test_recipe_covers_all_mlp_and_leaves_attention_at_q4():
    r = mix_recipe()
    assert r["id"] == MIX_ID
    assert r["organs"] == ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    assert r["layers"] == list(range(64))
    assert r["group"] == GROUP_Q3
    assert r["bits"] == BITS_Q3
    assert "HQ30UQ4" in r["attention"]
    assert "do not apply the MLP" in r["do_not"].lower() or "do not apply" in r["do_not"]
    assert r["native_kernel"] == NATIVE_Q3_KERNEL
    assert is_mlp_proj("language_model.model.layers.3.mlp.gate_proj.weight")
    assert not is_mlp_proj("language_model.model.layers.3.self_attn.q_proj.weight")
    assert not is_mlp_proj("language_model.model.layers.0.linear_attn.out_proj.weight")


def test_repeated_token_is_not_coherent():
    bad = judge_coherence("\n" * 16, [198] * 16)
    assert bad["coherent"] is False
    assert bad["repeated_single_token"] is True
    ids = [
        248068,
        198,
        760,
        1156,
        6587,
        264,
        11346,
        11,
        58655,
        15673,
        314,
        1204,
        264,
        18826,
        27545,
        264,
    ]
    ok = judge_coherence("<think>\nThe user wants a detailed, prose explanation", ids)
    assert ok["coherent"] is True


def test_census_parser_reads_expanded_counters():
    line = (
        "qwen38-decode mixed census: tensors=755 binary=0 residual=0 "
        "hgravs=0 uniform=192 q4=210 f32=353 refused=0 expanded_to_q4=0 "
        "expanded_to_float_gemv=0"
    )
    census = parse_census(line)
    assert census is not None
    assert census["uniform"] == 192
    assert census["q4"] == 210
    assert census["expanded_to_q4"] == 0
    assert census["expanded_to_float_gemv"] == 0
    assert census["binary"] == 0
    affine = (
        "qwen38-decode mixed census: tensors=755 binary=0 residual=0 "
        "hgravs=0 uniform=192 affine=0 q4=210 f32=353 refused=0 "
        "expanded_to_q4=0 expanded_to_float_gemv=0"
    )
    c2 = parse_census(affine)
    assert c2 is not None
    assert c2["uniform"] == 192
    assert c2["affine"] == 0
    assert c2["expanded_to_q4"] == 0


def test_receipt_reports_ebpw_verbatim_decode_and_native_kernel():
    assert RECEIPT.is_file(), (
        "receipts/headless/NOETIC_Q3_MLP_Q4_ATTN.json missing — "
        "run python3 tools/headless/q3_mlp_q4_attn.py"
    )
    doc = json.loads(RECEIPT.read_text())
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["parent_params"] == PARENT_PARAMS
    assert doc["native_q3_kernel"] == NATIVE_Q3_KERNEL
    compile_ = doc["compile"]
    assert compile_["n_q3_mlp"] == 192
    assert abs(compile_["q3_storage_bpw"] - 3.25) < 1e-12
    assert compile_["complete_ebpw"] < Q4_INCUMBENT_EBPW
    assert abs(compile_["q4_incumbent_complete_physical_bpw"] - Q4_INCUMBENT_EBPW) < 1e-9
    for key in ("storage_bpw", "active_bpw", "complete_ebpw"):
        assert key in compile_
        assert key in compile_["beside_q4"]
        assert abs(compile_["beside_q4"][key]["q4_incumbent"] - Q4_INCUMBENT_EBPW) < 1e-9
    decode = doc["decode"]
    assert decode["ok"] is True
    assert decode["n_new_tokens"] >= 16
    assert isinstance(decode["generated_text_verbatim"], str)
    assert decode["prompt"]
    assert decode["tok_s"] is not None and decode["tok_s"] > 0
    assert "dense_w_materialized" in decode
    assert "fallbacks" in decode
    assert "expanded_to_q4" in decode
    census = decode["census"]
    assert census is not None
    assert census["expanded_to_q4"] == 0
    assert census["expanded_to_float_gemv"] == 0
    assert census["uniform"] == 192
    assert decode["dequant_path"] is False
    assert decode["native_kernel_ran"] is True
    chosen = doc["chosen"]
    assert chosen["n_new_tokens"] >= 16
    assert chosen["native_kernel_ran"] is True
    assert chosen["complete_ebpw"] < Q4_INCUMBENT_EBPW
    # Sixteen copies of one token is not coherence; the receipt must say so.
    coh = chosen["coherence"]
    if coh["repeated_single_token"]:
        assert coh["coherent"] is False
        assert "not coherence" in doc["generation_finding"].lower() or (
            "NOT COHERENT" in doc["generation_finding"]
        )
    # Attention stayed q4: recipe says so, and no MLP name is an attention tensor.
    for name in compile_["q3_tensors"]:
        assert is_mlp_proj(name)
        assert "self_attn" not in name
        assert "linear_attn" not in name

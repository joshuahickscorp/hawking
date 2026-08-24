"""AFFINE2_NATIVE_MLP: native 2-bit affine whole-MLP executable."""
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

from affine2_native_mlp import (  # noqa: E402
    GROUP_AFFINE,
    MAGIC_AFFINE,
    PARENT_PARAMS,
    Q4_INCUMBENT_EBPW,
    RECEIPT,
    SCHEMA,
    SCHEMA_AFFINE,
    affine_storage_bpw,
    pack_hgrafv01,
    parse_hgrafv01,
    reconstruct_hgrafv01,
)
from first_noetic_executable import write_catalog  # noqa: E402


def test_affine_g32_storage_bpw_is_3():
    assert abs(affine_storage_bpw(32) - 3.0) < 1e-12
    assert abs(affine_storage_bpw(64) - 2.5) < 1e-12
    assert abs(affine_storage_bpw(64, bias=False) - 2.25) < 1e-12
    assert affine_storage_bpw(32) == 2.0 + 16 / 32 + 16 / 32


def test_hgrafv01_roundtrips_and_is_q_times_scale_plus_bias():
    rng = np.random.RandomState(0)
    w = rng.randn(8, 64).astype(np.float32)
    payload = pack_hgrafv01(w, 32)
    assert payload[:8] == MAGIC_AFFINE
    header = parse_hgrafv01(payload)
    assert header["schema"] == SCHEMA_AFFINE
    assert header["shape"] == [8, 64]
    assert header["group_size"] == 32
    assert header["groups"] == 16
    recon = reconstruct_hgrafv01(payload)
    assert recon.shape == (8, 64)
    # Codes stay in {0,1,2,3}; reconstruction is exact for those codes.
    header_len = struct.unpack_from("<I", payload, 8)[0]
    body = payload[12 + header_len :]
    scale_bytes = int(header["scale_bytes"])
    bias_bytes = int(header["bias_bytes"])
    packed = np.frombuffer(body[scale_bytes + bias_bytes :], dtype=np.uint8)
    assert packed.size == 8 * 64 // 4
    # Every reconstructed value equals q*scale+bias of its group.
    assert np.isfinite(recon).all()
    # Fitted affine should be closer than throwing the weights away.
    rel = np.linalg.norm(recon - w) / np.linalg.norm(w)
    assert rel < 0.5


def test_hgrafv01_refuses_ragged_cols():
    w = np.ones((4, 100), dtype=np.float32)
    with pytest.raises(Exception):
        pack_hgrafv01(w, 32)


def test_catalog_magic_and_codec_5(tmp_path: Path):
    payload = pack_hgrafv01(np.ones((4, 64), dtype=np.float32), 32)
    dest = tmp_path / "t.hgrafv01"
    dest.write_bytes(payload)
    rec = {
        "name": "language_model.model.layers.0.mlp.gate_proj.weight",
        "codec": 5,
        "organ": 0,
        "shape": [4, 64],
        "elements": 256,
        "segment_id": 0,
        "offset": 0,
        "nbytes": len(payload),
        "sha256": "ab" * 32,
        "codec_bpw": 3.0,
    }
    seg = {
        "id": 0,
        "filename": "t.hgrafv01",
        "bytes": len(payload),
        "sha256": "ab" * 32,
    }
    blob = write_catalog(tmp_path / "catalog.hq38m20", [rec], [seg])
    assert blob[:8] == b"HQ38M20\0"
    version, n_tensors, n_segments = struct.unpack_from("<III", blob, 8)
    assert version == 1
    assert n_tensors == 1
    assert n_segments == 1


def test_existing_kernel_files_are_present():
    shader = REPO / "crates" / "hawking-core" / "shaders" / "affine2_group32_matvec.metal"
    mixed = REPO / "crates" / "hawking-core" / "shaders" / "q80_mixed_decode.metal"
    assert shader.is_file()
    src = shader.read_text()
    assert "kernel void affine2_group32_matvec(" in src
    assert "kernel void affine2_group32_matvec_geo_tpr64_tg128(" in src
    mixed_src = mixed.read_text()
    assert "kernel void qwen_affine_q2_group32_matvec(" in mixed_src
    assert "kernel void qwen_affine_q2_group32_matvec_geo_tpr64_tg128(" in mixed_src


def test_receipt_reports_ebpw_verbatim_decode_and_parity():
    assert RECEIPT.is_file(), (
        "receipts/headless/AFFINE2_NATIVE_MLP.json missing — "
        "run python3 tools/headless/affine2_native_mlp.py"
    )
    doc = json.loads(RECEIPT.read_text())
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["parent_params"] == PARENT_PARAMS
    assert doc["kernel_already_existed"]["did_not_write_a_new_kernel"] is True
    compile_ = doc["compile"]
    assert compile_["n_affine"] == 192
    assert abs(compile_["affine_tensor_storage_bpw"] - 3.0) < 1e-12
    assert compile_["affine_group"] == GROUP_AFFINE
    assert compile_["complete_ebpw"] < Q4_INCUMBENT_EBPW
    assert (
        abs(compile_["q4_incumbent_complete_physical_bpw"] - Q4_INCUMBENT_EBPW) < 1e-9
    )
    billing = compile_["affine_bpw_billing"]
    assert billing["codes_bpw"] == 2.0
    assert abs(billing["scale_bpw"] - 0.5) < 1e-12
    assert abs(billing["bias_bpw"] - 0.5) < 1e-12
    assert abs(billing["total_bpw"] - 3.0) < 1e-12
    cmp_ = compile_["comparison_not_run_on_this_kernel"]
    assert abs(cmp_["bias_free_group64_bpw"] - 2.25) < 1e-12
    assert abs(cmp_["affine_group64_bpw"] - 2.5) < 1e-12
    chosen = doc["chosen"]
    assert chosen, "no mix produced a native decode"
    assert chosen["n_new_tokens"] >= 16
    assert isinstance(chosen["generated_text_verbatim"], str)
    assert chosen["prompt"]
    for key in ("storage_bpw", "active_bpw", "complete_ebpw"):
        assert key in chosen
        assert abs(chosen["q4_incumbent_complete_physical_bpw"] - Q4_INCUMBENT_EBPW) < 1e-9
    assert chosen["native_kernel_ran"] is True
    assert chosen["dequant_path"] is False
    assert chosen["fallbacks"] == 0
    assert chosen["dense_w_materialized"] == 0
    assert chosen.get("expanded_to_q4") == 0
    assert chosen.get("expanded_to_float_gemv") == 0
    assert "qwen_affine_q2_group32_matvec" in (chosen.get("bind") or "")
    decode = doc["decode"]
    assert decode["ok"] is True
    assert "expanded_to_q4=0" in (decode.get("census") or "") or decode.get("census")
    parity = doc["parity"]
    assert parity["ok"] is True
    assert str(parity.get("status", "")).upper() == "PASS" or "PASS" in parity.get("stdout", "")

"""AFFINE2_G64_LSFIT: LS-fitted affine-2 at group 64."""
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

from affine2_g64_lsfit import (  # noqa: E402
    AFFINE2_G32_EBPW,
    GROUP_AFFINE,
    MAGIC_AFFINE,
    MIX_ID,
    NATIVE_KERNEL_GEO,
    PARENT_PARAMS,
    Q3_EBPW,
    Q4_INCUMBENT_EBPW,
    RECEIPT,
    SCHEMA,
    SCHEMA_AFFINE,
    affine_storage_bpw,
    current_fit_is_least_squares,
    fit_affine_ls,
    fit_affine_minmax,
    mse,
    pack_hgrafv01,
    parse_hgrafv01,
    reconstruct_from_qsb,
    reconstruct_hgrafv01,
)
from first_noetic_executable import write_catalog  # noqa: E402


def test_affine_g64_storage_bpw_is_2_5():
    assert abs(affine_storage_bpw(64) - 2.5) < 1e-12
    assert abs(affine_storage_bpw(32) - 3.0) < 1e-12
    assert abs(affine_storage_bpw(64, bias=False) - 2.25) < 1e-12
    assert affine_storage_bpw(64) == 2.0 + 16 / 64 + 16 / 64


def test_current_g32_fit_is_not_least_squares():
    # Reading the packer is the refutation check. minmax/range is not LS.
    assert current_fit_is_least_squares() is False


def test_ls_beats_minmax_on_a_skewed_group():
    rng = np.random.RandomState(0)
    g = rng.randn(8, 4, 64).astype(np.float32) * 0.05
    g[:, :, -1] = 4.0  # one outlier wastes the minmax range
    s_mm, b_mm, q_mm = fit_affine_minmax(g)
    s_ls, b_ls, q_ls, n_iters = fit_affine_ls(g)
    recon_mm = reconstruct_from_qsb(q_mm, s_mm, b_mm)
    recon_ls = reconstruct_from_qsb(q_ls, s_ls, b_ls)
    assert mse(recon_ls, g) < mse(recon_mm, g)
    assert n_iters >= 1
    # Not a no-op: at least one group changed relative to minmax.
    assert not (np.array_equal(q_ls, q_mm) and np.allclose(s_ls, s_mm) and np.allclose(b_ls, b_mm))


def test_hgrafv01_ls_roundtrips_q_times_scale_plus_bias():
    rng = np.random.RandomState(1)
    w = rng.randn(8, 128).astype(np.float32)
    payload = pack_hgrafv01(w, 64, fit="ls")
    assert payload[:8] == MAGIC_AFFINE
    header = parse_hgrafv01(payload)
    assert header["schema"] == SCHEMA_AFFINE
    assert header["shape"] == [8, 128]
    assert header["group_size"] == 64
    assert header["groups"] == 16
    assert header["fit"] == "ls"
    recon = reconstruct_hgrafv01(payload)
    assert recon.shape == (8, 128)
    header_len = struct.unpack_from("<I", payload, 8)[0]
    body = payload[12 + header_len :]
    scale_bytes = int(header["scale_bytes"])
    bias_bytes = int(header["bias_bytes"])
    packed = np.frombuffer(body[scale_bytes + bias_bytes :], dtype=np.uint8)
    assert packed.size == 8 * 128 // 4
    assert np.isfinite(recon).all()
    rel = np.linalg.norm(recon - w) / np.linalg.norm(w)
    assert rel < 0.5


def test_hgrafv01_refuses_ragged_cols():
    w = np.ones((4, 100), dtype=np.float32)
    with pytest.raises(Exception):
        pack_hgrafv01(w, 64)


def test_catalog_magic_and_codec_5(tmp_path: Path):
    payload = pack_hgrafv01(np.ones((4, 64), dtype=np.float32), 64)
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
        "codec_bpw": 2.5,
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


def test_existing_kernel_family_accepts_group_64():
    shader = REPO / "crates" / "hawking-core" / "shaders" / "affine2_group32_matvec.metal"
    mixed = REPO / "crates" / "hawking-core" / "shaders" / "q80_mixed_decode.metal"
    assert shader.is_file()
    src = shader.read_text()
    assert "kernel void affine2_group32_matvec(" in src
    assert "kernel void affine2_group32_matvec_geo_tpr64_tg128(" in src
    assert "group_size == 32u || group_size == 64u" in src
    mixed_src = mixed.read_text()
    assert "kernel void qwen_affine_q2_group32_matvec(" in mixed_src
    assert "kernel void qwen_affine_q2_group32_matvec_geo_tpr64_tg128(" in mixed_src
    assert "group_size == 32u || group_size == 64u" in mixed_src


def test_receipt_reports_ebpw_verbatim_decode_and_parity():
    assert RECEIPT.is_file(), (
        "receipts/headless/AFFINE2_G64_LSFIT.json missing — "
        "run python3 tools/headless/affine2_g64_lsfit.py"
    )
    doc = json.loads(RECEIPT.read_text())
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["parent_params"] == PARENT_PARAMS
    assert doc["hypothesis"]["already_least_squares"] is False
    assert doc["hypothesis"]["refuted_by_reading_the_packer"] is False
    assert doc["kernel_family"]["did_not_write_a_new_codec_family"] is True
    assert doc["kernel_family"]["group_size_now"] == 64
    compile_ = doc["compile"]
    assert compile_["n_affine"] == 192
    assert abs(compile_["affine_tensor_storage_bpw"] - 2.5) < 1e-12
    assert compile_["affine_group"] == GROUP_AFFINE
    assert compile_["complete_ebpw"] < Q4_INCUMBENT_EBPW
    assert abs(compile_["q4_incumbent_complete_physical_bpw"] - Q4_INCUMBENT_EBPW) < 1e-9
    billing = compile_["affine_bpw_billing"]
    assert billing["codes_bpw"] == 2.0
    assert abs(billing["scale_bpw"] - 0.25) < 1e-12
    assert abs(billing["bias_bpw"] - 0.25) < 1e-12
    assert abs(billing["total_bpw"] - 2.5) < 1e-12
    assert compile_["how_scale_bias_were_currently_chosen"]["already_least_squares"] is False
    ls_probe = (compile_.get("ls_fit") or {}).get("probe") or {}
    assert ls_probe.get("ls_beats_minmax") is True
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
    census = chosen.get("census") or {}
    if isinstance(census, dict):
        assert census.get("affine") == 192
        assert census.get("expanded_to_q4") == 0
        assert census.get("expanded_to_float_gemv") == 0
    bind = chosen.get("bind") or ""
    assert NATIVE_KERNEL_GEO in bind or "HGRAVF01 affine2" in bind
    decode = doc["decode"]
    assert decode["ok"] is True
    assert decode["tok_s"] is not None and decode["tok_s"] > 0
    parity = doc["parity"]
    assert parity["ok"] is True
    assert str(parity.get("status", "")).upper() == "PASS" or "PASS" in (parity.get("stdout") or "")
    assert "max_abs_diff" in parity
    table = doc["comparison"]
    names = [row["codec"] for row in table]
    assert names == ["q4 incumbent", "q3 g64", "affine2 g32", "affine2 g64 LS"]
    assert abs(table[0]["complete_ebpw"] - Q4_INCUMBENT_EBPW) < 1e-9
    assert abs(table[1]["complete_ebpw"] - Q3_EBPW) < 1e-9
    assert abs(table[2]["complete_ebpw"] - AFFINE2_G32_EBPW) < 1e-9
    assert abs(table[3]["complete_ebpw"] - compile_["complete_ebpw"]) < 1e-12
    assert table[3]["text_verbatim"] == chosen["generated_text_verbatim"]
    assert MIX_ID in compile_["mix_id"]
    coh = chosen["coherence"]
    if coh.get("repeated_single_token"):
        assert coh["coherent"] is False

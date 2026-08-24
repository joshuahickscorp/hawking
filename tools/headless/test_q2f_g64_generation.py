"""Q2F_G64_GENERATION: 4-level LS-fitted 2-bit at group 64, asked to generate."""
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

from fractional_bit_canon import _fourlevel_fitted  # noqa: E402
from first_noetic_executable import Q4_INCUMBENT_EBPW, write_catalog  # noqa: E402
from q2f_g64_generation import (  # noqa: E402
    GROUP_Q2F,
    LEADER_EBPW,
    MAGIC_AFFINE,
    MIX_ID,
    NATIVE_KERNEL_Q2F_GEO,
    PARENT_PARAMS,
    Q2F_REPR,
    RECEIPT,
    SCHEMA,
    SCHEMA_AFFINE,
    fit_q2f,
    pack_hgrafv01_q2f,
    parse_hgrafv01_q2f,
    q2f_storage_bpw,
    reconstruct_hgrafv01_q2f,
)


def test_q2f_storage_bpw_is_2_25():
    assert abs(q2f_storage_bpw(64) - 2.25) < 1e-12
    assert q2f_storage_bpw(64) == 2.0 + 16 / 64
    assert q2f_storage_bpw(64) < 2.5


def test_fit_q2f_codes_are_four_level_and_reconstruct_as_q_minus_1_5_times_delta():
    rng = np.random.RandomState(0)
    w = rng.randn(8, 128).astype(np.float32)
    q, delta, probe = fit_q2f(w, 64)
    assert q.shape == (8, 2, 64)
    assert set(np.unique(q).tolist()) <= {0, 1, 2, 3}
    recon = (q.astype(np.float32) - np.float32(1.5)) * delta[..., None]
    rel = np.linalg.norm(recon.reshape(8, 128) - w) / np.linalg.norm(w)
    assert rel < 0.5
    assert probe["ls_iters"] >= 1
    ref = _fourlevel_fitted(w, 64)
    # CORRECTED 2026-08-24. This originally asserted the native 4-level fit was
    # COARSER than the composition reference, which was true only because that
    # reference was cheating: `rint(x*2)/2` emitted SEVEN levels, not four, so it
    # was not a 2-bit code at all. The reference has been fixed to snap to the
    # four legal codes, and the relationship INVERTED -- both are now true 4-level
    # and the native iterative assign+LS fit is the BETTER one.
    assert np.isfinite(ref).all()
    # Both must now be genuinely 4-level.
    g = 64
    O = ref.reshape(-1, g)
    amax = np.abs(O).max(-1, keepdims=True)
    scale = np.where(amax > 0, amax / 1.5, 1.0)
    assert len(np.unique(np.round(O / scale, 4))) <= 4, "the reference must be a 2-bit code"
    # And iterating assignment against a refitted delta is worth something.
    assert probe["q2f_4level_rel_l2"] <= probe["fourlevel_fitted_rel_l2"] + 1e-6, (
        "the native iterative fit should be at least as good as a single reassignment"
    )


def test_hgrafv01_q2f_roundtrips_with_zero_bias_bytes():
    rng = np.random.RandomState(1)
    w = rng.randn(8, 128).astype(np.float32)
    payload, _probe = pack_hgrafv01_q2f(w, 64)
    assert payload[:8] == MAGIC_AFFINE
    header = parse_hgrafv01_q2f(payload)
    assert header["schema"] == SCHEMA_AFFINE
    assert header["representation"] == Q2F_REPR
    assert header["shape"] == [8, 128]
    assert header["group_size"] == 64
    assert header["bias_bytes"] == 0
    recon = reconstruct_hgrafv01_q2f(payload)
    assert recon.shape == (8, 128)
    assert np.isfinite(recon).all()
    rel = np.linalg.norm(recon - w) / np.linalg.norm(w)
    assert rel < 0.5
    header_len = struct.unpack_from("<I", payload, 8)[0]
    body = payload[12 + header_len :]
    assert len(body) == int(header["scale_bytes"]) + int(header["code_bytes"])


def test_hgrafv01_q2f_refuses_ragged_cols():
    w = np.ones((4, 100), dtype=np.float32)
    with pytest.raises(Exception):
        pack_hgrafv01_q2f(w, 64)


def test_catalog_magic_and_codec_5(tmp_path: Path):
    payload, _ = pack_hgrafv01_q2f(np.ones((4, 64), dtype=np.float32), 64)
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
        "codec_bpw": 2.25,
    }
    seg = {
        "id": 0,
        "filename": "t.hgrafv01",
        "bytes": len(payload),
        "sha256": "ab" * 32,
    }
    blob = write_catalog(tmp_path / "catalog.hq38m20", [rec], [seg])
    assert blob[:8] == b"HQ38M20\0"


def test_q2f_kernels_are_group64_specialized_not_runtime_div():
    mixed = (REPO / "crates/hawking-core/shaders/q80_mixed_decode.metal").read_text()
    stand = (REPO / "crates/hawking-core/shaders/affine2_group32_matvec.metal").read_text()
    assert "kernel void qwen_q2f_group64_matvec_geo_tpr64_tg128(" in mixed
    assert "kernel void qwen_q2f_group64_matvec(" in mixed
    assert "const uint group = col >> 6u;" in mixed
    assert "(float(q) - 1.5f) * delta" in mixed or "(float(q) - 1.5f)" in mixed
    # Production q2f geo kernel must not take bind-time group_size.
    geo_start = mixed.find("kernel void qwen_q2f_group64_matvec_geo_tpr64_tg128(")
    geo_end = mixed.find("kernel void qwen_q2f_group64_matvec_gate_up_geo_tpr64_tg128(", geo_start + 1)
    geo = mixed[geo_start:geo_end]
    assert "constant uint& group_size" not in geo
    assert "col / group_size" not in geo
    assert "kernel void q2f_group64_matvec_geo_tpr64_tg128(" in stand


def test_receipt_reports_16_tokens_ebpw_and_counters():
    assert RECEIPT.is_file(), (
        "receipts/headless/Q2F_G64_GENERATION.json missing — "
        "run python3 tools/headless/q2f_g64_generation.py"
    )
    doc = json.loads(RECEIPT.read_text())
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["parent_params"] == PARENT_PARAMS
    assert doc["codec"]["group"] == GROUP_Q2F
    assert abs(doc["codec"]["billing_bpw"] - 2.25) < 1e-12
    compile_ = doc["compile"]
    assert compile_["n_q2f"] == 192
    assert abs(compile_["q2f_tensor_storage_bpw"] - 2.25) < 1e-12
    assert compile_["q2f_group"] == GROUP_Q2F
    assert compile_["complete_ebpw"] < Q4_INCUMBENT_EBPW
    assert compile_["complete_ebpw"] < LEADER_EBPW
    assert abs(compile_["q4_incumbent_complete_physical_bpw"] - Q4_INCUMBENT_EBPW) < 1e-9
    assert abs(compile_["leader_complete_ebpw"] - LEADER_EBPW) < 1e-9
    billing = compile_["q2f_bpw_billing"]
    assert billing["codes_bpw"] == 2.0
    assert abs(billing["scale_bpw"] - 0.25) < 1e-12
    assert billing["bias_bpw"] == 0.0
    chosen = doc["chosen"]
    assert chosen, "no mix produced a native decode"
    assert chosen["n_new_tokens"] >= 16
    assert isinstance(chosen["generated_text_verbatim"], str)
    assert chosen["prompt"]
    for key in ("storage_bpw", "active_bpw", "complete_ebpw"):
        assert key in chosen
    assert abs(chosen["q4_incumbent_complete_physical_bpw"] - Q4_INCUMBENT_EBPW) < 1e-9
    assert abs(chosen["leader_complete_ebpw"] - LEADER_EBPW) < 1e-9
    assert chosen["native_kernel_ran"] is True
    assert chosen["dequant_path"] is False
    assert chosen["fallbacks"] == 0
    assert chosen["dense_w_materialized"] == 0
    assert chosen.get("expanded_to_q4") == 0
    assert chosen.get("expanded_to_float_gemv") == 0
    assert chosen.get("dispatches_per_token") is not None
    census = chosen.get("census") or {}
    if isinstance(census, dict):
        assert census.get("affine") == 192
        assert census.get("expanded_to_q4") == 0
        assert census.get("expanded_to_float_gemv") == 0
    bind = chosen.get("bind") or ""
    assert NATIVE_KERNEL_Q2F_GEO in bind or "q2f" in bind
    decode = doc["decode"]
    assert decode["ok"] is True
    assert decode["tok_s"] is not None and decode["tok_s"] > 0
    parity = doc["parity"]
    assert parity["ok"] is True
    assert str(parity.get("status", "")).upper() == "PASS" or "PASS" in (
        parity.get("stdout") or ""
    )
    assert "max_abs_diff" in parity
    competence = doc["kernel_competence"]
    assert competence.get("any_q2f_defective") is False
    table = doc["comparison"]
    names = [row["codec"] for row in table]
    assert "q4 incumbent" in names[0]
    assert "leader" in names[1].lower() or "PARENT_A" in names[1]
    assert "q2_4level" in names[2] or "this arm" in names[2]
    assert abs(table[0]["complete_ebpw"] - Q4_INCUMBENT_EBPW) < 1e-9
    assert abs(table[1]["complete_ebpw"] - LEADER_EBPW) < 1e-9
    assert abs(table[2]["complete_ebpw"] - compile_["complete_ebpw"]) < 1e-12
    assert table[2]["text_verbatim"] == chosen["generated_text_verbatim"]
    assert MIX_ID in compile_["mix_id"]
    tried = doc["configs_tried"]
    ids = [c["id"] for c in tried]
    assert "q2f_bias_free_unfused" in ids
    assert "q2f_reuse_affine2_unfused" in ids
    coh = chosen["coherence"]
    if coh.get("repeated_single_token"):
        assert coh["coherent"] is False

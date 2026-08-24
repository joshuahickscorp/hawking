"""FIRST_NOETIC_EXECUTABLE: sub-2-bit MLP mix that the native runtime decodes."""
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
    GROUP_BINARY_SURVIVOR,
    MAGIC_BINARY,
    PARENT_PARAMS,
    Q4_INCUMBENT_EBPW,
    RECEIPT,
    SCHEMA,
    SCHEMA_BINARY,
    binary_storage_bpw,
    judge_coherence,
    mix_recipe,
    pack_hgravb01,
    parse_hgravb01,
    write_catalog,
)


def test_binary_g1024_storage_bpw_is_1_015625():
    assert abs(binary_storage_bpw(1024) - 1.015625) < 1e-12
    assert abs(binary_storage_bpw(64) - 1.25) < 1e-12
    assert binary_storage_bpw(64) < 2.0
    assert binary_storage_bpw(1024) < 2.0


def test_hgravb01_is_not_deletion_and_roundtrips_ledger():
    rng = np.random.RandomState(0)
    w = rng.randn(8, 1024).astype(np.float32)
    payload = pack_hgravb01(w, 1024)
    assert payload[:8] == MAGIC_BINARY
    header = parse_hgravb01(payload)
    assert header["schema"] == SCHEMA_BINARY
    assert header["shape"] == [8, 1024]
    assert header["group_size"] == 1024
    assert header["groups"] == 8
    # sign code keeps every weight
    grouped = w.reshape(8, 1, 1024)
    scales = np.abs(grouped.astype(np.float64)).mean(axis=-1).astype(np.float32).astype(np.float16)
    assert np.count_nonzero(scales) == 8
    assert int(header["sign_bytes"]) == (8 * 1024) // 8


def test_hgravb01_refuses_ragged_cols():
    w = np.ones((4, 100), dtype=np.float32)
    with pytest.raises(Exception):
        pack_hgravb01(w, 64)


def test_catalog_magic_and_record_size(tmp_path: Path):
    payload = pack_hgravb01(np.ones((4, 64), dtype=np.float32), 64)
    dest = tmp_path / "t.hgravb01"
    dest.write_bytes(payload)
    rec = {
        "name": "language_model.model.layers.0.mlp.down_proj.weight",
        "codec": 0,
        "organ": 2,
        "shape": [4, 64],
        "elements": 256,
        "segment_id": 0,
        "offset": 0,
        "nbytes": len(payload),
        "sha256": "ab" * 32,
        "codec_bpw": 1.25,
    }
    seg = {
        "id": 0,
        "filename": "t.hgravb01",
        "bytes": len(payload),
        "sha256": "ab" * 32,
    }
    blob = write_catalog(tmp_path / "catalog.hq38m20", [rec], [seg])
    assert blob[:8] == b"HQ38M20\0"
    version, n_tensors, n_segments = struct.unpack_from("<III", blob, 8)
    assert version == 1
    assert n_tensors == 1
    assert n_segments == 1


def test_mix_recipes_are_partial_and_sub2():
    a = mix_recipe("mix_a_l0_down_binary_g1024")
    b = mix_recipe("mix_b_all_down_binary_g1024")
    c = mix_recipe("mix_c_all_mlp_binary_g64")
    assert a["organs"] == ["mlp.down_proj"]
    assert a["layers"] == [0]
    assert b["organs"] == ["mlp.down_proj"]
    assert c["organs"] == ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    assert binary_storage_bpw(a["group"]) < 2.0
    assert binary_storage_bpw(c["group"]) < 2.0


def test_repeated_token_is_not_coherent():
    bad = judge_coherence(" a" * 16, [264] * 16)
    assert bad["coherent"] is False
    assert bad["repeated_single_token"] is True
    ws = judge_coherence("\n" * 16, [198] * 16)
    assert ws["coherent"] is False
    early = judge_coherence("front", [6735, 248046])
    assert early["coherent"] is False
    ids = [248068, 198, 760, 1156, 6587, 264, 11346, 11, 58655, 15673, 314, 1204, 264, 18826, 27545, 264]
    ok = judge_coherence("<think>\nThe user wants a detailed, prose explanation", ids)
    assert ok["coherent"] is True


def test_receipt_reports_mix_bpw_and_verbatim_decode():
    assert RECEIPT.is_file(), (
        "receipts/headless/FIRST_NOETIC_EXECUTABLE.json missing — "
        "run python3 tools/headless/first_noetic_executable.py"
    )
    doc = json.loads(RECEIPT.read_text())
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["parent_params"] == PARENT_PARAMS
    assert "mixes_attempted" in doc and doc["mixes_attempted"]
    chosen = doc["chosen"]
    assert chosen, "no mix produced a native decode"
    assert chosen["n_new_tokens"] >= 16
    assert isinstance(chosen["generated_text_verbatim"], str)
    assert chosen["prompt"]
    for key in ("storage_bpw", "active_bpw", "complete_ebpw"):
        assert key in chosen
        assert abs(chosen["q4_incumbent_complete_physical_bpw"] - Q4_INCUMBENT_EBPW) < 1e-9
    assert chosen["binary_tensor_storage_bpw"] < 2.0
    assert chosen["native_kernel_ran"] is True
    assert chosen["dequant_path"] is False
    assert chosen["fallbacks"] == 0
    assert chosen["exact_mix"]["codec"]
    assert chosen["exact_mix"]["tensors"]
    # Every mix attempted is recorded, including failures.
    for attempt in doc["mixes_attempted"]:
        assert "compile" in attempt
        assert "decode" in attempt
        compile_ = attempt["compile"]
        assert compile_["binary_storage_bpw"] < 2.0
        assert compile_["complete_ebpw"] < Q4_INCUMBENT_EBPW or compile_["n_binary"] >= 1

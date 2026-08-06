"""Executable residual-PQ payload grammar tests (no model download required)."""
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PACKER_PATH = REPO / "tools" / "llama_residual_pq_pack.py"
SPEC = importlib.util.spec_from_file_location("llama_residual_pq_pack", PACKER_PATH)
assert SPEC and SPEC.loader
packer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packer)


def test_residual_payload_has_canonical_executable_header_and_billed_size() -> None:
    weight = np.sin(np.arange(16 * 16, dtype=np.float32).reshape(16, 16) / 9.0)
    blob, report = packer.serialize_residual(
        weight, dim=8, stages=2, card=4, seed=7, iterations=1,
        batch_rows=32, reservoir_rows=32,
    )
    assert blob[:8] == b"LLM52RPK"
    assert struct.unpack_from("<HHH", blob, 8) == (8, 2, 4)
    assert blob[34] == 0 and blob[35] == 2
    # 64-byte header + 2*[4][8] fp16 + 16 rows * 2 chunks * 2 stages * 2 bits.
    assert len(blob) == 64 + 2 * 4 * 8 * 2 + (16 * 2 * 2 * 2 + 7) // 8
    assert report["active_bytes"] == len(blob)
    assert 0.0 <= report["relative_weight_error"] <= 1.0


def test_required_runtime_tensor_set_is_complete_for_one_layer() -> None:
    expected = packer.expected_names(1)
    assert "model.embed_tokens.weight" in expected
    assert "model.layers.0.self_attn.q_proj.weight" in expected
    assert "model.layers.0.mlp.down_proj.weight" in expected
    assert "model.norm.weight" in expected

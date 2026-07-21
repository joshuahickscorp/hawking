from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np
import pytest


TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_reference as kimi  # noqa: E402


def official_snapshot() -> Path:
    return (Path.home() / ".cache/huggingface/hub/models--moonshotai--Kimi-K2.6"
            / "snapshots" / kimi.REVISION)


def test_official_rope_layout_matches_reference_permutation() -> None:
    value = mx.array(np.arange(8, dtype=np.float32).reshape(1, 2, 4))
    actual = np.asarray(kimi.official_rope_layout(value))
    expected = np.array([[[0, 2, 1, 3], [4, 6, 5, 7]]], dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_tokenizer_round_trips_official_chat_protocol() -> None:
    source = official_snapshot()
    if not (source / "tiktoken.model").exists():
        pytest.skip("official tokenizer has not landed")
    tokenizer = kimi.KimiTokenizer(source)
    for text in (
        "2 + 2 =",
        kimi.KimiTokenizer.user_prompt("Reply with exactly OK.", thinking=False),
        kimi.KimiTokenizer.user_prompt("Think briefly.", thinking=True),
    ):
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_metal_native_int4_matches_manual_first_output() -> None:
    source = official_snapshot()
    shard_path = kimi.shard_path(source, 2)
    if not shard_path.exists():
        pytest.skip("first official routed-expert shard has not landed")
    shard = kimi.TensorShard(shard_path)
    base = f"{kimi.PREFIX}.layers.1.mlp.experts.0.down_proj"
    shape = [int(v) for v in np.asarray(shard.numpy(base + ".weight_shape")).flat]
    assert shape == [7168, 2048]
    packed = shard.numpy(base + ".weight_packed", unsigned_packed=True)
    scales = np.asarray(shard.numpy(base + ".weight_scale"), dtype=np.float32)
    rng = np.random.default_rng(260621)
    source_x = rng.standard_normal((1, shape[1]), dtype=np.float32)
    # Float32 isolates decode/affine correctness from expected BF16 accumulation error.
    x = mx.array(source_x, dtype=mx.float32)
    actual = kimi.quantized_linear(x, shard, base)
    mx.eval(actual)

    row_words = np.asarray(packed[0], dtype=np.uint32)
    shifts = np.arange(8, dtype=np.uint32) * 4
    signed = ((row_words[:, None] >> shifts[None, :]) & 0xF).astype(np.int8) - 8
    signed = signed.reshape(-1)[:shape[1]].astype(np.float32)
    dequantized = signed * np.repeat(scales[0], 32)[:shape[1]]
    x_bf16 = np.asarray(x.astype(mx.float32))[0]
    expected = float(x_bf16 @ dequantized)
    np.testing.assert_allclose(float(np.asarray(actual.astype(mx.float32))[0, 0]),
                               expected, rtol=2e-4, atol=2e-3)


def test_deterministic_signature_excludes_runtime_measurements() -> None:
    template = {
        "source": {"revision": kimi.REVISION}, "token_ids": [1, 2],
        "top_token_ids": [3], "top_logits": [1.25],
        "layers": [{"hidden_sha256": "a", "moe": {"route_indices": [[1]]},
                    "seconds": 1.0}], "runtime_seconds": 2.0,
    }
    changed = json.loads(json.dumps(template))
    changed["layers"][0]["seconds"] = 99.0
    changed["runtime_seconds"] = 101.0
    assert kimi.deterministic_signature(template) == kimi.deterministic_signature(changed)


def test_canonical_seal_preserves_utf8_probe_text() -> None:
    value = {"top_token_text": ["用户"], "status": "PASS"}
    expected = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode()
    assert kimi.canonical(value) == expected
    assert hashlib.sha256(kimi.canonical(value)).hexdigest() == hashlib.sha256(expected).hexdigest()

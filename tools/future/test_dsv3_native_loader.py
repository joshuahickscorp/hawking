"""CPU-only contract tests for the native DeepSeek/Kimi loader.

These tests deliberately do not open Kimi weights or acquire MPS.  They cover
the two failure modes that previously made ``--check`` overclaim compatibility:
MLX packed affine tensors and Kimi's stacked ``switch_mlp`` serialization.
"""
from types import SimpleNamespace

import torch

from tools.future.dsv3_native_loader import (
    _dequantize_mlx_affine,
    _mapping_for,
)


def _pack(values, bits):
    per_word = 32 // bits
    mask = (1 << bits) - 1
    words = []
    for start in range(0, len(values), per_word):
        word = 0
        for offset, value in enumerate(values[start:start + per_word]):
            word |= (int(value) & mask) << (offset * bits)
        words.append(word)
    return torch.tensor([words], dtype=torch.uint32)


def test_mlx_affine_unpack_is_low_bits_first_and_grouped():
    packed = _pack(range(32), bits=4)
    scales = torch.ones((1, 2), dtype=torch.float32)
    biases = torch.zeros((1, 2), dtype=torch.float32)
    decoded = _dequantize_mlx_affine(
        packed, scales, biases, (1, 32), dtype=torch.float32
    )
    expected = torch.tensor(list(range(16)) + list(range(16)), dtype=torch.float32)
    assert torch.equal(decoded, expected.reshape(1, 32))


def test_kimi_switch_mlp_maps_to_native_fused_experts_and_consumes_metadata():
    cfg = SimpleNamespace(
        num_hidden_layers=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        n_routed_experts=2,
    )
    want = {
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.experts.gate_up_proj",
        "model.layers.1.mlp.experts.down_proj",
        "model.layers.1.mlp.gate.weight",
    }
    have = {
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.gate.weight",
    }
    for stem in ("gate_proj", "up_proj", "down_proj"):
        base = f"model.layers.1.mlp.switch_mlp.{stem}"
        have.update({f"{base}.weight", f"{base}.scales", f"{base}.biases"})

    mapping = _mapping_for(want, have, cfg)
    assert mapping["model.layers.0.mlp.gate_proj.weight"]["kind"] == "direct"
    assert mapping["model.layers.1.mlp.experts.gate_up_proj"]["kind"] == "switch_fused"
    assert mapping["model.layers.1.mlp.experts.down_proj"]["kind"] == "direct"
    assert set(mapping["model.layers.1.mlp.experts.down_proj"]["sources"]) == {
        "model.layers.1.mlp.switch_mlp.down_proj.weight",
        "model.layers.1.mlp.switch_mlp.down_proj.scales",
        "model.layers.1.mlp.switch_mlp.down_proj.biases",
    }


def test_partial_packed_group_is_not_treated_as_direct():
    cfg = SimpleNamespace(
        num_hidden_layers=1,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        n_routed_experts=2,
    )
    target = "model.layers.0.mlp.experts.gate_up_proj"
    base = "model.layers.0.mlp.switch_mlp.gate_proj"
    have = {f"{base}.weight", f"{base}.scales"}
    mapping = _mapping_for({target}, have, cfg)
    assert target not in mapping

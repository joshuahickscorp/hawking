#!/usr/bin/env python3.12
"""Primitive and state tests for the inspectable GLM-5.2 reference forward."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest


CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_reference as ref  # noqa: E402


def _tiny_config(*, layers: int, mlp_types: list[str], indexer_types: list[str]):
    from transformers import GlmMoeDsaConfig

    config = GlmMoeDsaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        moe_intermediate_size=4,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        q_lora_rank=4,
        kv_lora_rank=2,
        qk_rope_head_dim=2,
        qk_nope_head_dim=2,
        v_head_dim=2,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        indexer_types=indexer_types,
        mlp_layer_types=mlp_types,
        layer_types=["deepseek_sparse_attention"] * layers,
        max_position_embeddings=32,
        rope_parameters={"rope_theta": 10_000.0, "rope_type": "default"},
        rms_norm_eps=1e-5,
        attention_bias=False,
        attention_dropout=0.0,
        num_nextn_predict_layers=1,
    )
    config._attn_implementation = "eager"
    return config


def _official_state_with_separate_experts(model, config) -> dict[str, np.ndarray]:
    state = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    intermediate = config.moe_intermediate_size
    for layer, kind in enumerate(config.mlp_layer_types):
        if kind != "sparse":
            continue
        prefix = f"model.layers.{layer}.mlp.experts"
        fused = state[f"{prefix}.gate_up_proj"]
        down = state[f"{prefix}.down_proj"]
        for expert in range(config.n_routed_experts):
            state[f"{prefix}.{expert}.gate_proj.weight"] = fused[expert, :intermediate]
            state[f"{prefix}.{expert}.up_proj.weight"] = fused[expert, intermediate:]
            state[f"{prefix}.{expert}.down_proj.weight"] = down[expert]
    return state


def test_rmsnorm_accumulates_in_float32() -> None:
    x = np.array([[[3.0, 4.0]]], dtype=np.float16)
    weight = np.array([2.0, 0.5], dtype=np.float16)
    actual = ref.rmsnorm(x, weight, 0.0)
    expected = np.array([[[1.6970563, 0.56568545]]], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.dtype == np.float32


def test_interleaved_rope_returns_concatenated_components() -> None:
    positions = np.array([[0, 1]], dtype=np.int64)
    cos, sin = ref.rope_cos_sin(positions, 4, 10_000.0)
    q = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4)
    actual, _ = ref.apply_interleaved_rope(q, q, cos, sin)
    # Position zero has no trigonometric ambiguity.  The pinned implementation
    # returns [even components, odd components], not [x0,x1,x2,x3].
    np.testing.assert_array_equal(actual[0, 0, 0], [0.0, 2.0, 1.0, 3.0])
    assert not np.array_equal(actual[0, 0, 0], q[0, 0, 0])


def test_router_corrects_selection_but_uses_raw_sigmoid_weights() -> None:
    hidden = np.zeros((1, 1, 2), dtype=np.float32)
    weight = np.zeros((4, 2), dtype=np.float32)
    correction = np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32)
    _logits, topk_weights, topk_indices = ref.router_topk(
        hidden,
        weight,
        correction,
        top_k=2,
        num_groups=1,
        topk_groups=1,
        normalize=True,
        scaling_factor=2.5,
    )
    assert 3 in topk_indices.ravel()
    # Every uncorrected sigmoid is 0.5, so correction cannot alter the gathered
    # mixture amplitudes.  Two equal selected weights normalize to 1.25 each.
    np.testing.assert_allclose(topk_weights, 1.25, rtol=0, atol=1e-7)
    np.testing.assert_allclose(topk_weights.sum(axis=-1), 2.5, rtol=0, atol=1e-7)


def test_indexer_is_causal_and_cache_extends() -> None:
    rng = np.random.default_rng(11)
    hidden = rng.normal(size=(1, 3, 4)).astype(np.float32)
    q_resid = rng.normal(size=(1, 3, 2)).astype(np.float32)
    positions = np.array([[0, 1, 2]], dtype=np.int64)
    cos, sin = ref.rope_cos_sin(positions, 2, 10_000.0)
    kwargs = {
        "wq_b": rng.normal(size=(8, 2)).astype(np.float32),
        "wk": rng.normal(size=(4, 4)).astype(np.float32),
        "k_norm_weight": np.ones(4, dtype=np.float32),
        "k_norm_bias": np.zeros(4, dtype=np.float32),
        "weights_proj": rng.normal(size=(2, 4)).astype(np.float32),
        "n_heads": 2,
        "head_dim": 4,
        "rotary_dim": 2,
        "topk": 2,
    }
    indices, scores, keys = ref.indexer_topk(
        hidden,
        q_resid,
        position_ids=positions,
        cos=cos,
        sin=sin,
        **kwargs,
    )
    assert keys.shape == (1, 3, 4)
    for query, position in enumerate(positions[0]):
        chosen = indices[0, query]
        finite = np.isfinite(scores[0, query, chosen])
        assert np.all(chosen[finite] <= position)
        assert np.count_nonzero(finite) == min(2, int(position) + 1)

    next_hidden = rng.normal(size=(1, 2, 4)).astype(np.float32)
    next_q = rng.normal(size=(1, 2, 2)).astype(np.float32)
    next_positions = np.array([[3, 4]], dtype=np.int64)
    next_cos, next_sin = ref.rope_cos_sin(next_positions, 2, 10_000.0)
    next_indices, _next_scores, next_keys = ref.indexer_topk(
        next_hidden,
        next_q,
        position_ids=next_positions,
        cos=next_cos,
        sin=next_sin,
        past_keys=keys,
        **kwargs,
    )
    assert next_keys.shape == (1, 5, 4)
    assert np.all(next_indices[0, 0] <= 3)
    assert np.all(next_indices[0, 1] <= 4)


def test_canonical_pairs_remove_unsorted_slot_ambiguity() -> None:
    a = ref.canonical_topk_pairs(
        np.array([[[3, 1]]]), np.array([[[0.4, 0.6]]], dtype=np.float32)
    )
    b = ref.canonical_topk_pairs(
        np.array([[[1, 3]]]), np.array([[[0.6, 0.4]]], dtype=np.float32)
    )
    assert a == b


def test_shared_indexshare_without_predecessor_fails_closed() -> None:
    # The guard is exercised before any shared-layer tensor access.
    with pytest.raises(ref.Glm52ReferenceError, match="previous full-layer"):
        ref.attention_forward(
            np.zeros((1, 1, 2), dtype=np.float32),
            {},
            0,
            {
                "num_attention_heads": 1,
                "q_lora_rank": 1,
                "kv_lora_rank": 1,
                "qk_nope_head_dim": 1,
                "qk_rope_head_dim": 2,
                "v_head_dim": 1,
                "rms_norm_eps": 1e-5,
                "rope_parameters": {"rope_theta": 10_000},
            },
            np.array([[0]], dtype=np.int64),
            ref.ReferenceCache(),
            indexer_type="shared",
            previous_topk=None,
        )


def test_interleaved_rope_matches_pinned_transformers() -> None:
    torch = pytest.importorskip("torch")
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        apply_rotary_pos_emb_interleave,
    )

    rng = np.random.default_rng(21)
    q = rng.normal(size=(2, 3, 5, 8)).astype(np.float32)
    k = rng.normal(size=(2, 1, 5, 8)).astype(np.float32)
    positions = np.broadcast_to(np.arange(5)[None, :], (2, 5))
    cos, sin = ref.rope_cos_sin(positions, 8, 8_000_000.0)
    actual_q, actual_k = ref.apply_interleaved_rope(q, k, cos, sin)
    expected_q, expected_k = apply_rotary_pos_emb_interleave(
        torch.from_numpy(q),
        torch.from_numpy(k),
        torch.from_numpy(cos),
        torch.from_numpy(sin),
    )
    np.testing.assert_allclose(actual_q, expected_q.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_k, expected_k.numpy(), rtol=1e-6, atol=1e-6)


def test_router_matches_pinned_transformers_after_canonicalization() -> None:
    torch = pytest.importorskip("torch")
    from transformers import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaTopkRouter

    config = GlmMoeDsaConfig(
        hidden_size=8,
        n_routed_experts=8,
        num_experts_per_tok=3,
        n_group=2,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
    )
    module = GlmMoeDsaTopkRouter(config).eval()
    rng = np.random.default_rng(22)
    hidden = rng.normal(size=(2, 4, 8)).astype(np.float32)
    weight = rng.normal(size=(8, 8)).astype(np.float32)
    correction = rng.normal(scale=0.2, size=(8,)).astype(np.float32)
    with torch.no_grad():
        module.weight.copy_(torch.from_numpy(weight))
        module.e_score_correction_bias.copy_(torch.from_numpy(correction))
        expected_logits, expected_weights, expected_indices = module(torch.from_numpy(hidden))
    logits, weights, indices = ref.router_topk(
        hidden,
        weight,
        correction,
        top_k=3,
        num_groups=2,
        topk_groups=1,
        normalize=True,
        scaling_factor=2.5,
    )
    np.testing.assert_allclose(logits.reshape(-1, 8), expected_logits.numpy(), rtol=2e-6, atol=2e-6)
    actual_pairs = ref.canonical_topk_pairs(indices, weights)
    expected_pairs = ref.canonical_topk_pairs(expected_indices.numpy(), expected_weights.numpy())
    for actual_row, expected_row in zip(actual_pairs, expected_pairs):
        assert [item[0] for item in actual_row] == [item[0] for item in expected_row]
        np.testing.assert_allclose(
            [item[1] for item in actual_row],
            [item[1] for item in expected_row],
            rtol=2e-6,
            atol=2e-6,
        )


def test_indexer_indices_match_pinned_transformers() -> None:
    torch = pytest.importorskip("torch")
    from transformers import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer

    config = GlmMoeDsaConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=4,
        qk_rope_head_dim=2,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
    )
    module = GlmMoeDsaIndexer(config, 0).eval()
    rng = np.random.default_rng(23)
    hidden = rng.normal(size=(1, 5, 8)).astype(np.float32)
    q_resid = rng.normal(size=(1, 5, 4)).astype(np.float32)
    weights = {
        "wq_b": rng.normal(size=(8, 4)).astype(np.float32),
        "wk": rng.normal(size=(4, 8)).astype(np.float32),
        "k_norm_weight": rng.normal(loc=1.0, scale=0.1, size=(4,)).astype(np.float32),
        "k_norm_bias": rng.normal(scale=0.1, size=(4,)).astype(np.float32),
        "weights_proj": rng.normal(size=(2, 8)).astype(np.float32),
    }
    positions = np.arange(5, dtype=np.int64)[None, :]
    cos, sin = ref.rope_cos_sin(positions, 2, 8_000_000.0)
    with torch.no_grad():
        module.wq_b.weight.copy_(torch.from_numpy(weights["wq_b"]))
        module.wk.weight.copy_(torch.from_numpy(weights["wk"]))
        module.k_norm.weight.copy_(torch.from_numpy(weights["k_norm_weight"]))
        module.k_norm.bias.copy_(torch.from_numpy(weights["k_norm_bias"]))
        module.weights_proj.weight.copy_(torch.from_numpy(weights["weights_proj"]))
        expected = module(
            torch.from_numpy(hidden),
            torch.from_numpy(q_resid),
            (torch.from_numpy(cos), torch.from_numpy(sin)),
            None,
            torch.from_numpy(positions),
            None,
        ).numpy()
    actual, _scores, _keys = ref.indexer_topk(
        hidden,
        q_resid,
        position_ids=positions,
        cos=cos,
        sin=sin,
        n_heads=2,
        head_dim=4,
        rotary_dim=2,
        topk=2,
        **weights,
    )
    for actual_row, expected_row in zip(actual.reshape(-1, 2), expected.reshape(-1, 2)):
        np.testing.assert_array_equal(np.sort(actual_row), np.sort(expected_row))


def test_full_dense_indexshare_logits_match_pinned_transformers() -> None:
    torch = pytest.importorskip("torch")
    from transformers import GlmMoeDsaForCausalLM

    config = _tiny_config(
        layers=2,
        mlp_types=["dense", "dense"],
        indexer_types=["full", "shared"],
    )
    torch.manual_seed(31)
    model = GlmMoeDsaForCausalLM(config).eval()
    ids = torch.tensor([[1, 3, 5, 7, 2]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits.numpy()
    actual, _cache, trace = ref.main_forward(
        ids.numpy(), _official_state_with_separate_experts(model, config), config.to_dict()
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-4)
    np.testing.assert_array_equal(
        trace["layers"][0]["attention"]["topk_indices"],
        trace["layers"][1]["attention"]["topk_indices"],
    )


def test_full_sparse_moe_logits_match_pinned_transformers() -> None:
    torch = pytest.importorskip("torch")
    from transformers import GlmMoeDsaForCausalLM

    config = _tiny_config(layers=1, mlp_types=["sparse"], indexer_types=["full"])
    torch.manual_seed(32)
    model = GlmMoeDsaForCausalLM(config).eval()
    ids = torch.tensor([[1, 3, 5, 7, 2]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits.numpy()
    state = _official_state_with_separate_experts(model, config)
    actual, _cache, trace = ref.main_forward(ids.numpy(), state, config.to_dict())
    # NumPy accumulates expert contributions in a deterministic expert-major
    # order while PyTorch's kernel is token/slot ordered; float32 associativity
    # therefore needs a small absolute allowance.
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-4)
    np.testing.assert_allclose(
        trace["layers"][0]["mlp"]["topk_weights"].sum(axis=-1),
        2.5,
        rtol=2e-6,
        atol=2e-6,
    )


def test_reference_prefill_and_tokenwise_cache_logits_match() -> None:
    torch = pytest.importorskip("torch")
    from transformers import GlmMoeDsaForCausalLM

    config = _tiny_config(
        layers=2,
        mlp_types=["dense", "dense"],
        indexer_types=["full", "shared"],
    )
    torch.manual_seed(33)
    model = GlmMoeDsaForCausalLM(config).eval()
    state = _official_state_with_separate_experts(model, config)
    ids = np.array([[1, 3, 5, 7, 2]], dtype=np.int64)
    prefill, _prefill_cache, _trace = ref.main_forward(ids, state, config.to_dict())
    cache = ref.ReferenceCache()
    pieces = []
    for token in ids[0]:
        logits, cache, _trace = ref.main_forward(
            np.array([[token]], dtype=np.int64), state, config.to_dict(), cache=cache
        )
        pieces.append(logits)
    tokenwise = np.concatenate(pieces, axis=1)
    np.testing.assert_allclose(tokenwise, prefill, rtol=3e-3, atol=1.5e-4)
    assert cache.sequence_length() == ids.shape[1]

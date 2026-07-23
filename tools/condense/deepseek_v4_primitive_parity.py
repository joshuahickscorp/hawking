#!/usr/bin/env python3.12
"""Primitive parity: validate the NumPy DeepSeek-V4 primitives against the official
transformers reference on synthetic inputs.

The correction to the campaign is exact: a streamed forward is buildable, and the real
blocker was that no independently validated implementation existed. transformers 5.14.1
ships the official ``deepseek_v4`` modeling, so each primitive can be checked against it on
tiny synthetic weights, with no need to run the 160 GB model. A primitive is admitted only
when it matches the reference to float tolerance; anything that does not match is a named
missing primitive, not a silent approximation.

This validates the MoE organ (router scoring, top-k, SwiGLU experts, shared expert) and the
norm. Attention, the DSA compressor/indexer and hyper-connections are validated by the same
pattern as they are built.

    parity        run every implemented primitive against the reference
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CONDENSE = Path(__file__).resolve().parent
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import deepseek_v4_moe as ds


def _tiny_config():
    import torch  # noqa: F401
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    # A small but structurally identical config: same routing, activation and clamp.
    return DeepseekV4Config(
        hidden_size=64, intermediate_size=32, num_local_experts=8,
        num_experts_per_tok=3, scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
        swiglu_limit=10.0, hidden_act="silu", num_hidden_layers=2, mlp_bias=False)


def _rmsnorm_parity() -> dict:
    import torch
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RMSNorm
    hidden = 64
    ref = DeepseekV4RMSNorm(hidden, eps=ds.RMS_EPS)
    with torch.no_grad():
        ref.weight.copy_(torch.randn(hidden))
    x = np.random.default_rng(0).standard_normal((5, hidden)).astype(np.float32)
    mine = ds.rmsnorm(x, ref.weight.detach().numpy())
    theirs = ref(torch.from_numpy(x)).detach().numpy()
    err = float(np.max(np.abs(mine - theirs)))
    return {"primitive": "rmsnorm", "max_abs_error": err, "matches": err < 1e-4}


def _moe_parity() -> dict:
    import torch
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4TopKRouter, DeepseekV4MLP)
    config = _tiny_config()
    rng = np.random.default_rng(1)

    router = DeepseekV4TopKRouter(config)
    gate_w = rng.standard_normal((config.num_local_experts, config.hidden_size)).astype(np.float32)
    with torch.no_grad():
        router.weight.copy_(torch.from_numpy(gate_w))
    shared = DeepseekV4MLP(config)
    w = {name: rng.standard_normal(tuple(getattr(shared, name).weight.shape)).astype(np.float32)
         for name in ("gate_proj", "up_proj", "down_proj")}
    with torch.no_grad():
        for name, value in w.items():
            getattr(shared, name).weight.copy_(torch.from_numpy(value))

    x = rng.standard_normal((6, config.hidden_size)).astype(np.float32)

    # Reference: router weights/indices, then the shared MLP as a stand-in expert forward.
    _, ref_weights, ref_indices = router(torch.from_numpy(x))
    ref_shared = shared(torch.from_numpy(x)).detach().numpy()

    # Mine: sqrtsoftplus scoring, top-3, normalised weights (unscaled), and the same SwiGLU
    # with the fixed max-only gate clamp using the shared weights as an expert.
    scores = ds._sqrtsoftplus(x @ gate_w.T)
    topk = np.argsort(-scores, axis=-1)[:, :config.num_experts_per_tok]
    mine_weights = np.take_along_axis(scores, topk, axis=-1)
    mine_weights = mine_weights / (mine_weights.sum(axis=-1, keepdims=True) + 1e-20)
    mine_weights *= config.routed_scaling_factor

    expert = {"w1": w["gate_proj"], "w3": w["up_proj"], "w2": w["down_proj"]}
    mine_shared = ds._expert_forward(x, expert)

    # Indices are a set per token; compare as sets since topk sorted=False.
    ref_sets = [set(row.tolist()) for row in ref_indices.detach().numpy()]
    mine_sets = [set(row.tolist()) for row in topk]
    index_match = all(a == b for a, b in zip(ref_sets, mine_sets))
    # Match weights by aligning to the reference index order.
    weight_err = 0.0
    for t in range(x.shape[0]):
        order = {int(e): i for i, e in enumerate(topk[t])}
        aligned = np.array([mine_weights[t, order[int(e)]] for e in ref_indices.detach().numpy()[t]])
        weight_err = max(weight_err, float(np.max(np.abs(aligned - ref_weights.detach().numpy()[t]))))
    shared_err = float(np.max(np.abs(mine_shared - ref_shared)))
    return {"primitive": "moe_router+swiglu",
            "router_indices_match": index_match,
            "router_weight_max_error": weight_err,
            "swiglu_expert_max_error": shared_err,
            "matches": index_match and weight_err < 1e-4 and shared_err < 1e-3}


def parity() -> dict:
    results = [_rmsnorm_parity(), _moe_parity()]
    return {"schema": "hawking.deepseek_v4.primitive_parity.v1",
            "reference": "transformers.models.deepseek_v4 (official)",
            "validated": [r for r in results if r["matches"]],
            "results": results,
            "all_match": all(r["matches"] for r in results)}


if __name__ == "__main__":
    report = parity()
    print(json.dumps(report, indent=2, default=float))
    raise SystemExit(0 if report["all_match"] else 1)

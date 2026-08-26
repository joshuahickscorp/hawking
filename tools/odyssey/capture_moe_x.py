#!/usr/bin/env python3
"""Capture EXACT real activations from the first K layers of a MoE specimen.

Layers 0..K-1 of a transformer do not depend on any later layer, so a truncated model's
activations at those depths are exact, not an approximation. That is what makes this a
legitimate probe rather than a proxy -- and proxies are what the Qwen campaign proved
must never be used to rank compression candidates.

Runs on the vision python (torch + transformers); the default interpreter has neither.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

PROMPTS = [
    "Explain, in ordinary prose and at length, how a compiler turns a for-loop into basic blocks and then into machine code.",
    "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    p = a[len(a)//2]\n",
    '{"tool": "search", "arguments": {"query": "unified memory bandwidth", "limit": 5}}',
    "The capital of France is Paris, and the capital of Japan is",
    "$ grep -rn 'threadgroup' crates/hawking-core/shaders/*.metal | head -20",
]


def load_truncated(model_dir, n_layers):
    """Truncate to the first K layers. Layers 0..K-1 do not depend on any later layer, so
    their activations are EXACT rather than approximate. The class is resolved from the
    config so this works on any supported architecture, not only the MoE it was written
    for."""
    cfg = AutoConfig.from_pretrained(model_dir)
    cfg.num_hidden_layers = n_layers
    cfg._attn_implementation = "eager"
    cls = (Qwen3MoeForCausalLM if cfg.model_type == "qwen3_moe"
           else AutoModelForCausalLM._model_mapping[type(cfg)])
    # meta + to_empty() allocates UNINITIALIZED memory for parameters AND buffers, and
    # load_state_dict only fills what the checkpoint carries. Non-persistent buffers --
    # which Falcon-H1's Mamba path has and Qwen3-MoE did not -- stay as garbage, and the
    # forward returns NaN with n_missing reporting 0. Constructing on CPU lets __init__
    # initialize the buffers; the random parameter init it also does is immediately
    # overwritten by load_state_dict.
    model = cls(cfg)
    idx = json.load(open(Path(model_dir) / "model.safetensors.index.json"))["weight_map"]
    want = {}
    for name in model.state_dict():
        if name in idx:
            want.setdefault(idx[name], []).append(name)
    sd, loaded = {}, 0
    for shard, names in sorted(want.items()):
        with safe_open(Path(model_dir) / shard, framework="pt") as f:
            for n in names:
                sd[n] = f.get_tensor(n)
                loaded += 1

    # transformers >=5 stores MoE experts FUSED -- experts.gate_up_proj (E, 2*inter, hidden)
    # and experts.down_proj (E, hidden, inter) -- while the checkpoint stores them per
    # expert. Without this mapping the expert weights silently stay at their random init
    # and every activation past layer 0 is captured from a model that is partly noise.
    # That is precisely the "calibrated on gibberish" failure, so the mapping is verified
    # numerically below rather than trusted.
    fused = 0
    by_shard = {}
    has_experts = any(".mlp.experts." in n for n in idx)
    for name, shard in (idx.items() if has_experts else []):
        if ".mlp.experts." in name:
            li = int(name.split(".layers.")[1].split(".")[0])
            if li < n_layers:
                by_shard.setdefault(shard, []).append(name)
    parts = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(Path(model_dir) / shard, framework="pt") as f:
            for n in names:
                parts[n] = f.get_tensor(n)
    for li in (range(n_layers) if has_experts else []):
        gu, dn = [], []
        for e in range(cfg.num_experts):
            g = parts[f"model.layers.{li}.mlp.experts.{e}.gate_proj.weight"]
            u = parts[f"model.layers.{li}.mlp.experts.{e}.up_proj.weight"]
            d = parts[f"model.layers.{li}.mlp.experts.{e}.down_proj.weight"]
            gu.append(torch.cat([g, u], 0))
            dn.append(d)
        sd[f"model.layers.{li}.mlp.experts.gate_up_proj"] = torch.stack(gu, 0)
        sd[f"model.layers.{li}.mlp.experts.down_proj"] = torch.stack(dn, 0)
        fused += 2
    del parts

    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    if not has_experts:
        model.eval()
        return model, cfg, {"n_tensors_loaded": loaded, "n_expert_stacks_fused": 0,
                            "n_missing": len(missing), "n_unexpected": len(unexpected),
                            "missing_examples": list(missing)[:5],
                            "fused_mapping_max_abs_diff_vs_per_expert": 0.0,
                            "fused_mapping_verified": True,
                            "note": "dense architecture: no expert stacks to fuse"}
    # Numerical proof the fused mapping is right: run expert 0 of layer 0 by hand from the
    # per-expert checkpoint tensors and compare against the loaded fused weights.
    with safe_open(Path(model_dir) / idx[f"model.layers.0.mlp.experts.0.gate_proj.weight"],
                   framework="pt") as f:
        g0 = f.get_tensor("model.layers.0.mlp.experts.0.gate_proj.weight").float()
        u0 = f.get_tensor("model.layers.0.mlp.experts.0.up_proj.weight").float()
        d0 = f.get_tensor("model.layers.0.mlp.experts.0.down_proj.weight").float()
    torch.manual_seed(0)
    x = torch.randn(1, cfg.hidden_size)
    ref = torch.nn.functional.silu(x @ g0.T) * (x @ u0.T) @ d0.T
    ex = model.model.layers[0].mlp.experts
    gu = ex.gate_up_proj[0].float()
    dw = ex.down_proj[0].float()
    got = torch.nn.functional.silu(x @ gu[:cfg.moe_intermediate_size].T) \
        * (x @ gu[cfg.moe_intermediate_size:].T) @ dw.T
    err = float((ref - got).abs().max())
    if err > 1e-3:
        raise SystemExit(f"fused expert mapping is wrong: max_abs_diff={err}")
    return model, cfg, {"n_tensors_loaded": loaded, "n_expert_stacks_fused": fused,
                        "n_missing": len(missing), "n_unexpected": len(unexpected),
                        "missing_examples": list(missing)[:5],
                        "fused_mapping_max_abs_diff_vs_per_expert": err,
                        "fused_mapping_verified": err <= 1e-3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    model, cfg, load_info = load_truncated(a.model_dir, a.layers)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    t_load = time.time() - t0

    caught = {}

    def mk(i):
        def hook(mod, args, kwargs=None):
            h = args[0] if args else None
            if h is not None:
                caught.setdefault(i, []).append(h.detach().reshape(-1, h.shape[-1]).float())
        return hook

    # An organ is a ROLE, not a tensor name: Qwen calls the MLP `mlp`, Falcon-H1 calls it
    # `feed_forward`. Resolve by role rather than assuming the name the first specimen used.
    MLP_NAMES = ("mlp", "feed_forward", "ffn", "mlp_block")
    layers = model.model.layers

    def mlp_of(layer):
        for n in MLP_NAMES:
            if hasattr(layer, n):
                return getattr(layer, n), n
        raise SystemExit(f"no MLP submodule on {type(layer).__name__}; "
                         f"tried {MLP_NAMES}, has {[k for k, _ in layer.named_children()]}")

    hooked = []
    handles = []
    for i in range(a.layers):
        mod, nm = mlp_of(layers[i])
        hooked.append(nm)
        handles.append(mod.register_forward_pre_hook(mk(i)))
    ids_seen = 0
    with torch.no_grad():
        for p in PROMPTS:
            enc = tok(p, return_tensors="pt")
            model(**enc)
            ids_seen += int(enc["input_ids"].numel())
    for h in handles:
        h.remove()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model_dir": a.model_dir, "n_layers_captured": a.layers,
            "exact_because": "layers 0..K-1 do not depend on any later layer, so truncation "
                             "does not change their activations",
            "n_prompts": len(PROMPTS), "n_tokens": ids_seen,
            "load_wall_s": round(t_load, 1), "load": load_info,
            "model_type": cfg.model_type,
            "mlp_submodule_name": hooked[0] if hooked else None,
            "hidden_size": cfg.hidden_size,
            "num_experts": getattr(cfg, "num_experts", None),
            "num_experts_per_tok": getattr(cfg, "num_experts_per_tok", None),
            "moe_intermediate_size": getattr(cfg, "moe_intermediate_size", None),
            "intermediate_size": getattr(cfg, "intermediate_size", None),
            "layers": {}}
    bad = []
    for i, chunks in sorted(caught.items()):
        X = torch.cat(chunks, 0).numpy().astype(np.float32)
        if not np.isfinite(X).all():
            bad.append({"layer": i, "n_nonfinite": int((~np.isfinite(X)).sum()),
                        "of": int(X.size)})
            continue
        np.save(out / f"X_layer{i}.npy", X)
        meta["layers"][str(i)] = {"file": f"X_layer{i}.npy", "shape": list(X.shape),
                                  "dtype": "float32",
                                  "abs_mean": float(np.abs(X).mean()),
                                  "std": float(X.std())}
    if bad:
        # A capture with NaN in it would silently poison every downstream ranking, which is
        # the "calibrated on gibberish" failure in a new costume. Refuse to write it.
        meta["nonfinite_layers"] = bad
        (out / "CAPTURE.json").write_text(json.dumps(meta, indent=1))
        raise SystemExit(f"REFUSING to write a capture with non-finite activations: {bad}")
    meta["all_layers_finite"] = True
    (out / "CAPTURE.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "layers"}, indent=1))
    for k, v in meta["layers"].items():
        print(" layer", k, v["shape"], "abs_mean", round(v["abs_mean"], 5))
    return 0


if __name__ == "__main__":
    sys.exit(main())

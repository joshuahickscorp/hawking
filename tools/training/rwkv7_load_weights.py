"""Strict safetensors -> RWKV7Model loader.

Maps HF `RWKV7ForCausalLM` tensor names onto the pure-torch module, casting
BF16 -> fp32. "Strict" means: every model parameter is assigned exactly once
from the checkpoint, and every consumed checkpoint tensor is accounted for. Any
missing param or leftover (unconsumed) checkpoint tensor raises.

HF name map (per the safetensors dump):
  globals:  model.embeddings.weight, model.norm.{weight,bias}, lm_head.weight
  layer 0:  model.layers.0.pre_norm.{weight,bias}            (LN0, layer 0 only)
  per blk:  attn_norm.{weight,bias}, ffn_norm.{weight,bias}
  attn.:    {r,k,v,o}_proj.weight; x_{r,w,k,v,a,g}[1,1,n];
            w_lora.lora.{0,2}.weight + lora.2.bias  (w1,w2,w0)
            a_lora.lora.{0,2}.weight + lora.2.bias  (a1,a2,a0)
            v_lora.lora.{0,2}.weight + lora.2.bias  (v1,v2,v0, layer>0 only)
            g_lora.lora.{0,2}.weight                (g1,g2, NO bias)
            k_k, k_a, r_k[H,D], g_norm.{weight,bias}
  ffn.:     x_k, key.weight[n_ff,n], value.weight[n,n_ff]
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from rwkv7_torch_model import RWKV7Config, RWKV7Model


def load_rwkv7(
    safetensors_path: str | Path,
    cfg: RWKV7Config | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> RWKV7Model:
    cfg = cfg or RWKV7Config()
    model = RWKV7Model(cfg)

    # Read every checkpoint tensor as fp32 up front.
    ckpt: dict[str, torch.Tensor] = {}
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            ckpt[key] = f.get_tensor(key).to(torch.float32)

    consumed: set[str] = set()

    def take(name: str) -> torch.Tensor:
        if name not in ckpt:
            raise KeyError(f"checkpoint missing tensor: {name}")
        consumed.add(name)
        return ckpt[name]

    sd: dict[str, torch.Tensor] = {}

    # ---- globals ----
    sd["embeddings.weight"] = take("model.embeddings.weight")
    sd["norm_w"] = take("model.norm.weight")
    sd["norm_b"] = take("model.norm.bias")
    sd["lm_head.weight"] = take("lm_head.weight")

    # ---- per-layer ----
    for li in range(cfg.n_layer):
        p = f"model.layers.{li}."
        m = f"layers.{li}."

        if li == 0:
            sd[m + "pre_norm_w"] = take(p + "pre_norm.weight")
            sd[m + "pre_norm_b"] = take(p + "pre_norm.bias")
        sd[m + "attn_norm_w"] = take(p + "attn_norm.weight")
        sd[m + "attn_norm_b"] = take(p + "attn_norm.bias")
        sd[m + "ffn_norm_w"] = take(p + "ffn_norm.weight")
        sd[m + "ffn_norm_b"] = take(p + "ffn_norm.bias")

        a = p + "attn."
        ma = m + "attn."
        sd[ma + "r_proj.weight"] = take(a + "r_proj.weight")
        sd[ma + "k_proj.weight"] = take(a + "k_proj.weight")
        sd[ma + "v_proj.weight"] = take(a + "v_proj.weight")
        sd[ma + "o_proj.weight"] = take(a + "o_proj.weight")

        # lerp coeffs stored [1,1,n] -> flatten to [n].
        for slot in ("r", "w", "k", "v", "a", "g"):
            sd[ma + f"x_{slot}"] = take(a + f"x_{slot}").reshape(-1)

        # decay LoRA
        sd[ma + "w1"] = take(a + "w_lora.lora.0.weight")
        sd[ma + "w2"] = take(a + "w_lora.lora.2.weight")
        sd[ma + "w0"] = take(a + "w_lora.lora.2.bias")
        # iclr LoRA
        sd[ma + "a1"] = take(a + "a_lora.lora.0.weight")
        sd[ma + "a2"] = take(a + "a_lora.lora.2.weight")
        sd[ma + "a0"] = take(a + "a_lora.lora.2.bias")
        # value-residual LoRA (layer > 0 only)
        if li > 0:
            sd[ma + "v1"] = take(a + "v_lora.lora.0.weight")
            sd[ma + "v2"] = take(a + "v_lora.lora.2.weight")
            sd[ma + "v0"] = take(a + "v_lora.lora.2.bias")
        # gate LoRA (no bias)
        sd[ma + "g1"] = take(a + "g_lora.lora.0.weight")
        sd[ma + "g2"] = take(a + "g_lora.lora.2.weight")

        sd[ma + "k_k"] = take(a + "k_k")
        sd[ma + "k_a"] = take(a + "k_a")
        sd[ma + "r_k"] = take(a + "r_k")  # [H, D]
        sd[ma + "g_norm_w"] = take(a + "g_norm.weight")
        sd[ma + "g_norm_b"] = take(a + "g_norm.bias")

        fn = p + "ffn."
        mf = m + "ffn."
        sd[mf + "x_k"] = take(fn + "x_k")
        sd[mf + "key.weight"] = take(fn + "key.weight")
        sd[mf + "value.weight"] = take(fn + "value.weight")

    # ---- strict assignment ----
    own = dict(model.named_parameters())
    own.update(dict(model.named_buffers()))
    missing = sorted(set(own) - set(sd))
    extra = sorted(set(sd) - set(own))
    if missing:
        raise RuntimeError(f"loader: {len(missing)} model params got no checkpoint tensor: {missing[:8]}")
    if extra:
        raise RuntimeError(f"loader: {len(extra)} mapped keys not in model: {extra[:8]}")

    # Shape check + copy.
    for name, dst in own.items():
        src = sd[name]
        if tuple(src.shape) != tuple(dst.shape):
            raise RuntimeError(f"loader: shape mismatch {name}: ckpt {tuple(src.shape)} vs model {tuple(dst.shape)}")
        with torch.no_grad():
            dst.copy_(src)

    leftover = sorted(set(ckpt) - consumed)
    if leftover:
        raise RuntimeError(f"loader: {len(leftover)} checkpoint tensors unconsumed: {leftover[:8]}")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "models/rwkv7-g1-04-hf/model.safetensors"
    m = load_rwkv7(path)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"loaded RWKV7Model strict from {path}: {n_params/1e6:.1f}M params, {m.cfg.n_layer} layers")

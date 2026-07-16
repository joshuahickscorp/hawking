#!/usr/bin/env python3
"""Dump an Eagle6 head forward-pass fixture for Rust parity testing.

Loads a trained head from safetensors, runs the PyTorch forward on a
fixed-seed (prev_token, residual, intermediate) input, and writes a
JSON fixture that the Rust parity test consumes.

The fixture intentionally stores the full logits vector (so the Rust
test can compute L2 distance + a top-K comparison) but uses base64-
encoded raw f32 bytes for compactness — JSON with 151936 ascii floats
is ~3 MB, which would clog the repo. Base64 of raw bytes is ~810 KB,
acceptable as a test fixture.

Example:
  python3 tools/eagle5_forward_dump.py \\
    --head $HOME/Downloads/head_final.safetensors \\
    --out crates/hawking-core/tests/eagle5_parity_q3b.json \\
    --seed 0xea91e5
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--head", type=Path, required=True, help="Path to head_final.safetensors")
    p.add_argument("--out", type=Path, required=True, help="Output JSON fixture path")
    p.add_argument("--seed", type=lambda s: int(s, 0), default=0xea91e5,
                   help="Seed for the deterministic input vector (int, e.g. 0xea91e5)")
    p.add_argument("--prev-token", type=int, default=13,
                   help="prev_token id to feed (default 13, an arbitrary in-vocab id)")
    args = p.parse_args()

    try:
        import numpy as np
        import torch
    except ImportError as e:
        print(f"ERROR: needs numpy + torch in this env: {e}", file=sys.stderr)
        return 1

    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: pip install safetensors", file=sys.stderr)
        return 1

    # Load the head's safetensors directly (don't rely on the trainer's
    # Python module — this script must be runnable from a clean env).
    with safe_open(str(args.head), framework="pt") as f:
        meta = f.metadata() or {}
        hidden_dim = int(meta["hidden_dim"])
        vocab_size = int(meta["vocab_size"])
        n_heads = int(meta["n_heads"])
        ff_mult = float(meta["ff_mult"])
        num_blocks = int(meta["num_blocks"])
        ff_dim = int(round(hidden_dim * ff_mult))

        def t(name: str) -> torch.Tensor:
            return f.get_tensor(name)

        # Frozen tensors (will be converted to f32 for compute parity).
        token_embd_f16 = t("_token_embd")  # (hidden, vocab) f16
        lm_head_f16 = t("_lm_head")        # (hidden, vocab) f16
        output_norm = t("_output_norm")    # (hidden,) f32
        in_proj = t("in_proj.weight")      # (hidden, 3*hidden) f32
        residual_gate = t("residual_gate").item()  # scalar

        def block_keys(prefix: str) -> dict:
            return {
                "attn_norm": t(f"{prefix}attn_norm"),
                "q_proj": t(f"{prefix}q_proj.weight"),
                "k_proj": t(f"{prefix}k_proj.weight"),
                "v_proj": t(f"{prefix}v_proj.weight"),
                "out_proj": t(f"{prefix}out_proj.weight"),
                "mlp_norm": t(f"{prefix}mlp_norm"),
                "mlp_gate": t(f"{prefix}mlp.gate.weight"),
                "mlp_up": t(f"{prefix}mlp.up.weight"),
                "mlp_down": t(f"{prefix}mlp.down.weight"),
            }

        blocks = [block_keys("block.")]
        for i in range(num_blocks - 1):
            blocks.append(block_keys(f"extra_blocks.{i}."))

    print(f"Loaded head: hidden={hidden_dim} vocab={vocab_size} "
          f"n_heads={n_heads} ff_mult={ff_mult} num_blocks={num_blocks}")

    # Build a deterministic input vector using numpy's seeded PRNG.
    # Real verifier residual/intermediate at the capture layer would
    # be on the order of N(0, 1) post-norm; we use that scale here.
    rng = np.random.default_rng(args.seed)
    residual = rng.standard_normal(hidden_dim).astype(np.float32) * 0.5
    intermediate = rng.standard_normal(hidden_dim).astype(np.float32) * 0.5

    residual_t = torch.from_numpy(residual)
    intermediate_t = torch.from_numpy(intermediate)

    # Forward — bit-for-bit replicate the Eagle5Head.forward in
    # eagle5_train_pytorch.py:192-226 at B=1, S=1.

    RMS_EPS = 1e-6

    def rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + RMS_EPS)
        return (x32 * rms * w.float()).to(x.dtype)

    def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # nn.Linear without bias: y = x @ w.T
        return x @ w.t()

    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def attn_s1(h: torch.Tensor, block: dict) -> torch.Tensor:
        head_dim = hidden_dim // n_heads
        q = linear(h, block["q_proj"]).view(n_heads, head_dim)
        k = linear(h, block["k_proj"]).view(n_heads, head_dim)
        v = linear(h, block["v_proj"]).view(n_heads, head_dim)
        # S=1: scores per head = q·k / sqrt(d), single scalar each.
        # Mask 0, softmax([scalar]) = 1. attn_per_head = v.
        # We do the explicit matmul for parity clarity.
        scale = 1.0 / (head_dim ** 0.5)
        # scores: (n_heads, 1, 1) — but at S=1 just compute q·k
        _scores = (q * k).sum(dim=-1, keepdim=True) * scale  # (n_heads, 1)
        # softmax over the single key axis is 1.0; multiply by v.
        attn = v  # (n_heads, head_dim)
        attn_flat = attn.reshape(hidden_dim)
        return linear(attn_flat, block["out_proj"])

    # (1) prev_embed: column `prev_token` of token_embd (hidden,)
    prev_embed_f32 = token_embd_f16[:, args.prev_token].float()

    # (2) concat
    x = torch.cat([prev_embed_f32, residual_t, intermediate_t], dim=-1)

    # (3) in_proj
    x = linear(x, in_proj)

    # (4) blocks
    for b in blocks:
        h = rms_norm(x, b["attn_norm"])
        a = attn_s1(h, b)
        x = x + a
        h = rms_norm(x, b["mlp_norm"])
        m_gate = linear(h, b["mlp_gate"])
        m_up = linear(h, b["mlp_up"])
        m_act = silu(m_gate) * m_up
        m_out = linear(m_act, b["mlp_down"])
        x = x + m_out

    # (5) baseline
    baseline = rms_norm(residual_t, output_norm)
    # (6) draft_hidden = baseline + residual_gate * x
    draft_hidden = baseline + residual_gate * x
    # (7) logits = draft_hidden @ lm_head
    logits = draft_hidden @ lm_head_f16.float()  # (vocab,)

    logits_np = logits.detach().cpu().numpy().astype(np.float32)
    argmax = int(np.argmax(logits_np))
    top_k = 16
    top_idx = np.argpartition(-logits_np, top_k)[:top_k]
    top_idx = top_idx[np.argsort(-logits_np[top_idx])]
    top_vals = logits_np[top_idx].tolist()
    print(f"argmax token id = {argmax}, top logit = {logits_np[argmax]:.4f}")
    print(f"top-{top_k} indices = {top_idx[:8].tolist()}")
    print(f"logits L2 = {float(np.sqrt((logits_np ** 2).sum())):.4f}")

    fixture = {
        "schema": "eagle5-forward-parity-v1",
        "head_path": str(args.head),
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "n_heads": n_heads,
        "ff_mult": ff_mult,
        "num_blocks": num_blocks,
        "seed": args.seed,
        "prev_token": args.prev_token,
        "residual_b64": base64.b64encode(residual.tobytes()).decode("ascii"),
        "intermediate_b64": base64.b64encode(intermediate.tobytes()).decode("ascii"),
        # Reference logits packed as raw little-endian f32 bytes, b64.
        "logits_b64": base64.b64encode(logits_np.tobytes()).decode("ascii"),
        "argmax": argmax,
        "top_k": top_k,
        "top_indices": top_idx.tolist(),
        "top_values": top_vals,
        "logits_l2": float(np.sqrt((logits_np ** 2).sum())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")
    # Quick sanity: residual b64 round-trips.
    assert struct.unpack(
        f"<{hidden_dim}f", base64.b64decode(fixture["residual_b64"])
    )[0] == residual[0]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

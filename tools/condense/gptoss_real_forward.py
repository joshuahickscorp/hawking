#!/usr/bin/env python3.12
"""Real full-model GPT-OSS-120B forward - bounded-streaming, parity-correct activation.

This is the missing G4 instrument. Every forward in the campaign so far (Second-Light, Gravity,
G0-G3) was a from-config APPROXIMATION on synthetic Harmony-ish token ids, measuring only the
RELATIVE original-vs-packed divergence, and used the repo's non-parity `_swiglu` (split-half +
plain SiLU), which is valid only because the wrong activation CANCELS in a relative comparison.

This module produces REAL logits on REAL tokenizer output by:
  1. threading the residual stream through all 36 blocks (block N output feeds block N+1), instead
     of the isolated from-embedding approximation each gate used;
  2. using the CORRECT gpt-oss expert activation, taken verbatim from transformers 5.6.2
     modeling_gpt_oss.GptOssExperts._apply_gate: interleaved gate/up (`[..., ::2]`,`[..., 1::2]`),
     gate.clamp(max=7), up.clamp(-7,7), glu = gate*sigmoid(1.702*gate), out = (up+1)*glu;
  3. final RMSNorm + unembed -> real logits over the vocab.

It stays BOUNDED-MEMORY: one block plus a handful of experts resident at a time (~65 GB read per
forward, no offload). Attention/RoPE/sinks reuse the PROVEN primitives in gptoss_block +
gptoss_moe_runtime (attention math already matches transformers eager_attention_forward on sinks).

HONESTY BOUNDARY. This is a from-config forward. RoPE convention and the mlp1 gate/up interleave
ordering (OpenAI original checkpoint vs HF converted) are validated EMPIRICALLY by coherent
next-token prediction on real prompts (`--smoke`). If output is incoherent, a convention is wrong;
fix and re-smoke before any capability claim. No capability is claimed by this module alone.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Callable

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gptoss_block as blk          # rmsnorm, _rope, N_Q/N_KV/HEAD_DIM  (proven primitives)
import gptoss_moe_runtime as rt     # ProvenanceReader, load_router, load_expert

HIDDEN = 2880
N_LAYERS = 36
TOP_K = 4
N_Q, N_KV, HEAD_DIM = blk.N_Q, blk.N_KV, blk.HEAD_DIM
ALPHA = 1.702
LIMIT = 7.0


# ── correct gpt-oss expert activation (transformers modeling_gpt_oss parity) ──────────────
def apply_gate(gate_up: np.ndarray) -> np.ndarray:
    """gate_up: [..., 2*intermediate] with INTERLEAVED gate/up. Returns [..., intermediate]."""
    gate = gate_up[..., ::2]
    up = gate_up[..., 1::2]
    gate = np.minimum(gate, LIMIT)
    up = np.clip(up, -LIMIT, LIMIT)
    glu = gate * (1.0 / (1.0 + np.exp(-ALPHA * gate)))
    return (up + 1.0) * glu


# ── generalized block-N attention (verbatim from the frozen G3 controller path) ───────────
def block_n_attention(reader: Any, n: int, x: np.ndarray) -> np.ndarray:
    """x:[seq, HIDDEN] -> attn_out:[seq, HIDDEN]. GQA + RoPE + causal + per-head sinks. Matches
    transformers eager_attention_forward (concat sink logit, subtract max, softmax, drop sink)."""
    seq = x.shape[0]
    nrm = reader.bf16(f"block.{n}.attn.norm.scale")
    qkvw = reader.bf16(f"block.{n}.attn.qkv.weight")
    qkvb = reader.bf16(f"block.{n}.attn.qkv.bias")
    outw = reader.bf16(f"block.{n}.attn.out.weight")
    outb = reader.bf16(f"block.{n}.attn.out.bias")
    sinks = reader.bf16(f"block.{n}.attn.sinks")

    h = blk.rmsnorm(x, nrm)
    qkv = h @ qkvw.T + qkvb
    q = qkv[:, :N_Q * HEAD_DIM].reshape(seq, N_Q, HEAD_DIM)
    k = qkv[:, N_Q * HEAD_DIM:(N_Q + N_KV) * HEAD_DIM].reshape(seq, N_KV, HEAD_DIM)
    v = qkv[:, (N_Q + N_KV) * HEAD_DIM:].reshape(seq, N_KV, HEAD_DIM)
    pos = np.arange(seq)
    q = blk._rope(q, pos); k = blk._rope(k, pos)
    grp = N_Q // N_KV
    scale = 1.0 / np.sqrt(HEAD_DIM)
    causal = np.triu(np.full((seq, seq), -1e30, dtype=np.float32), 1)
    out = np.zeros((seq, N_Q, HEAD_DIM), dtype=np.float32)
    for hh in range(N_Q):
        kv = hh // grp
        scores = (q[:, hh] @ k[:, kv].T) * scale + causal
        aug = np.concatenate([scores, np.full((seq, 1), sinks[hh], np.float32)], axis=1)
        aug -= aug.max(axis=1, keepdims=True)
        w = np.exp(aug); w /= w.sum(axis=1, keepdims=True)
        out[:, hh] = w[:, :seq] @ v[:, kv]
    return out.reshape(seq, N_Q * HEAD_DIM) @ outw.T + outb


# ── the real full-model forward ───────────────────────────────────────────────────────────
class RealForward:
    """Bounded-streaming real forward. expert_hook(block, expert, w) -> w lets a caller substitute
    packed/decoded experts (for the G4 original-vs-packed comparison); None = source-native."""

    def __init__(self, manifest_path: str | None = None):
        self.reader = rt.ProvenanceReader(manifest_path) if manifest_path else rt.ProvenanceReader()
        self._emb: np.ndarray | None = None
        self._unemb: np.ndarray | None = None

    def source_present(self) -> bool:
        s = self.reader.by_name.get("block.0.mlp.gate.weight")
        from pathlib import Path
        return s is not None and Path(s["shard_path"]).exists()

    @property
    def embedding(self) -> np.ndarray:
        if self._emb is None:
            self._emb = self.reader.bf16("embedding.weight")          # [vocab, HIDDEN]
        return self._emb

    @property
    def unembedding(self) -> np.ndarray:
        if self._unemb is None:
            self._unemb = self.reader.bf16("unembedding.weight")      # [vocab, HIDDEN]
        return self._unemb

    def _moe_block(self, reader: Any, n: int, mlp_in: np.ndarray,
                   expert_hook: Callable | None) -> np.ndarray:
        """mlp_in:[seq, HIDDEN] -> moe_out:[seq, HIDDEN]. Streams + caches experts within the block."""
        router = rt.load_router(reader, n)
        seq = mlp_in.shape[0]
        logits = mlp_in @ router["weight"].T + router["bias"]         # [seq, 128]
        out = np.zeros_like(mlp_in)
        cache: dict[int, dict[str, np.ndarray]] = {}
        for p in range(seq):
            lg = logits[p]
            idx = np.argsort(-lg)[:TOP_K]
            w = lg[idx]; w = np.exp(w - w.max()); w = w / w.sum()
            x = mlp_in[p]
            acc = np.zeros(HIDDEN, dtype=np.float32)
            for e, gw in zip(idx, w):
                e = int(e)
                ex = cache.get(e)
                if ex is None:
                    ex = rt.load_expert(reader, n, e)
                    if expert_hook is not None:
                        ex = expert_hook(n, e, ex)
                    cache[e] = ex
                h = ex["mlp1"] @ x + ex["mlp1_bias"]                  # [5760]
                a = apply_gate(h)                                     # [HIDDEN]
                y = ex["mlp2"] @ a + ex["mlp2_bias"]                  # [HIDDEN]
                acc += gw * y
            out[p] = acc
        return out

    def logits_for(self, token_ids: list[int], *, positions: str = "last",
                   max_blocks: int = N_LAYERS, expert_hook: Callable | None = None,
                   progress: bool = False) -> np.ndarray:
        reader = self.reader
        x = np.ascontiguousarray(self.embedding[token_ids], dtype=np.float32)   # [seq, HIDDEN]
        for n in range(max_blocks):
            t0 = time.time()
            x = x + block_n_attention(reader, n, x)                              # attention residual
            mlp_norm = reader.bf16(f"block.{n}.mlp.norm.scale")
            mlp_in = blk.rmsnorm(x, mlp_norm)
            x = x + self._moe_block(reader, n, mlp_in, expert_hook)              # MoE residual
            if progress:
                print(f"  block {n:2d}/{max_blocks-1}  {time.time()-t0:5.1f}s  "
                      f"rms={float(np.sqrt(np.mean(x**2))):.3f}", flush=True)
        if max_blocks < N_LAYERS:
            return x                                                             # partial: hidden state
        x = blk.rmsnorm(x, reader.bf16("norm.scale"))                           # final norm
        sel = x if positions == "all" else x[-1:]
        return sel @ self.unembedding.T                                         # [.., vocab]


def _smoke(prompt: str, n_next: int = 5, max_blocks: int = N_LAYERS) -> dict:
    from tokenizers import Tokenizer
    tk_path = os.path.join(os.path.dirname(_HERE), "..", "models", "gpt-oss-120b", "tokenizer.json")
    tk_path = os.path.abspath(os.path.join(_HERE, "..", "..", "models", "gpt-oss-120b", "tokenizer.json"))
    tk = Tokenizer.from_file(tk_path)
    ids = tk.encode(prompt).ids
    fwd = RealForward()
    if not fwd.source_present():
        return {"ok": None, "reason": "source absent"}
    t0 = time.time()
    logits = fwd.logits_for(ids, positions="last", max_blocks=max_blocks, progress=True)
    dt = time.time() - t0
    lg = logits[-1].astype(np.float64)
    finite = bool(np.isfinite(lg).all())
    top = np.argsort(-lg)[:n_next].tolist()
    top_tokens = [tk.decode([int(t)]) for t in top]
    top_probs = np.exp(lg[top] - lg.max()); top_probs = (top_probs / np.exp(lg - lg.max()).sum())
    return {
        "ok": finite,
        "prompt": prompt,
        "n_tokens_in": len(ids),
        "max_blocks": max_blocks,
        "forward_seconds": round(dt, 1),
        "logits_finite": finite,
        "logits_shape": list(logits.shape),
        "top_next_tokens": [{"id": int(t), "token": repr(tok), "prob": round(float(p), 4)}
                            for t, tok, p in zip(top, top_tokens, top_probs)],
    }


def main(argv: list[str] | None = None) -> int:
    import json
    ap = argparse.ArgumentParser(description="Real full-model GPT-OSS-120B forward.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n-next", type=int, default=8)
    ap.add_argument("--max-blocks", type=int, default=N_LAYERS)
    args = ap.parse_args(argv)
    if args.smoke:
        print(json.dumps(_smoke(args.prompt, n_next=args.n_next, max_blocks=args.max_blocks),
                         indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

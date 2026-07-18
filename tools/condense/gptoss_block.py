#!/usr/bin/env python3.12
"""Bounded single-block GPT-OSS-120B forward: produce the REAL post-attention MoE input.

The F2 output-divergence measurement is far more faithful when the MoE sees the true residual stream
(post-attention, post-mlp-norm) rather than raw token embeddings. This module runs block 0's
attention over a short token SEQUENCE (so tokens actually attend to one another) and returns the
genuine MoE-input activations per position.

Geometry (from the source config): hidden 2880, 64 query heads / 8 KV heads (GQA), head_dim 64,
RoPE theta 150000, sliding_window 128, attention sinks (one learned logit per query head),
rms_norm_eps 1e-5, top-k 4.

Honesty / validity boundary:
  * This is a from-config forward, NOT HF-parity-validated. Absolute activations may differ from a
    reference in RoPE convention / SwiGLU clamp / eps.
  * It is used to compute the RELATIVE output divergence between original and packed experts, where
    the shared residual input and shared reference SwiGLU largely CANCEL those approximations - the
    residual only needs realistic scale and in-context mixing, which this provides.
  * Short sequences (< initial_context 4096, < sliding_window 128) keep RoPE-scaling and windowing
    inactive, minimizing from-config risk.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gptoss_moe_runtime as rt   # noqa: E402

HIDDEN, N_Q, N_KV, HEAD_DIM = 2880, 64, 8, 64
ROPE_THETA, RMS_EPS = 150000.0, 1e-5


def rmsnorm(x: np.ndarray, scale: np.ndarray, eps: float = RMS_EPS) -> np.ndarray:
    """RMSNorm over the last dim. x: [..., H]."""
    ms = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)) * scale


def _rope(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Rotate-half RoPE. x: [seq, n_heads, head_dim]."""
    half = HEAD_DIM // 2
    freqs = ROPE_THETA ** (-np.arange(half, dtype=np.float32) / half)   # [half]
    ang = np.outer(pos.astype(np.float32), freqs)                       # [seq, half]
    cos = np.cos(ang)[:, None, :]; sin = np.sin(ang)[:, None, :]        # [seq,1,half]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def block0_attention(reader: rt.ProvenanceReader, x: np.ndarray) -> np.ndarray:
    """Block-0 attention over a sequence x:[seq, HIDDEN] -> attn_out:[seq, HIDDEN]. GQA + RoPE +
    causal + per-head attention sinks."""
    seq = x.shape[0]
    nrm = reader.bf16("block.0.attn.norm.scale")
    qkvw = reader.bf16("block.0.attn.qkv.weight")   # [5120, 2880]
    qkvb = reader.bf16("block.0.attn.qkv.bias")     # [5120]
    outw = reader.bf16("block.0.attn.out.weight")   # [2880, 4096]
    outb = reader.bf16("block.0.attn.out.bias")     # [2880]
    sinks = reader.bf16("block.0.attn.sinks")       # [64]

    h = rmsnorm(x, nrm)                              # [seq, 2880]
    qkv = h @ qkvw.T + qkvb                          # [seq, 5120]
    q = qkv[:, :N_Q * HEAD_DIM].reshape(seq, N_Q, HEAD_DIM)                 # [seq,64,64]
    k = qkv[:, N_Q * HEAD_DIM:(N_Q + N_KV) * HEAD_DIM].reshape(seq, N_KV, HEAD_DIM)
    v = qkv[:, (N_Q + N_KV) * HEAD_DIM:].reshape(seq, N_KV, HEAD_DIM)
    pos = np.arange(seq)
    q = _rope(q, pos); k = _rope(k, pos)
    grp = N_Q // N_KV                               # 8 query heads per kv head
    scale = 1.0 / np.sqrt(HEAD_DIM)
    causal = np.triu(np.full((seq, seq), -1e30, dtype=np.float32), 1)      # j>i masked
    out = np.zeros((seq, N_Q, HEAD_DIM), dtype=np.float32)
    for hh in range(N_Q):
        kv = hh // grp
        scores = (q[:, hh] @ k[:, kv].T) * scale + causal                  # [seq,seq]
        # attention sink: an extra column with the per-head learned logit; its weight is discarded
        aug = np.concatenate([scores, np.full((seq, 1), sinks[hh], np.float32)], axis=1)
        aug -= aug.max(axis=1, keepdims=True)
        w = np.exp(aug); w /= w.sum(axis=1, keepdims=True)
        out[:, hh] = w[:, :seq] @ v[:, kv]                                  # sink column contributes 0
    attn = out.reshape(seq, N_Q * HEAD_DIM) @ outw.T + outb                 # [seq, 2880]
    return attn


def block0_moe_inputs(reader: rt.ProvenanceReader, token_ids: list[int],
                      embeddings: np.ndarray | None = None) -> np.ndarray:
    """Return the REAL block-0 MoE-input activations for a token sequence: embed -> attention ->
    residual -> mlp.norm. embeddings may be passed in to avoid reloading the 2.3GB matrix."""
    if embeddings is None:
        emb = reader.bf16("embedding.weight")
        x = np.ascontiguousarray(emb[token_ids], dtype=np.float32)
        del emb
    else:
        x = np.ascontiguousarray(embeddings[token_ids], dtype=np.float32)
    attn = block0_attention(reader, x)
    resid = x + attn                                                       # post-attention residual
    mlp_norm = reader.bf16("block.0.mlp.norm.scale")
    return rmsnorm(resid, mlp_norm)                                        # the true MoE input


def selftest() -> dict:
    """Shape + finiteness + scale sanity on the real source (skips if absent)."""
    from pathlib import Path
    reader = rt.ProvenanceReader()
    if not Path(reader.by_name["block.0.mlp.gate.weight"]["shard_path"]).exists():
        return {"ok": None, "reason": "source absent"}
    ids = [1, 100, 500, 2000, 42, 7]
    mi = block0_moe_inputs(reader, ids)
    return {"ok": bool(np.isfinite(mi).all()), "shape": list(mi.shape),
            "mean_abs": round(float(np.mean(np.abs(mi))), 4),
            "rms": round(float(np.sqrt(np.mean(mi ** 2))), 4)}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2))

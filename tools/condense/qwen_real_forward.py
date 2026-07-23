#!/usr/bin/env python3.12
"""Real full-model Qwen3-235B-A22B (qwen3_moe) forward - bounded-streaming, parity-shaped.

The Qwen analog of gptoss_real_forward.py. It threads the residual stream through all 94 blocks
(block N output feeds block N+1), streams one MoE expert at a time straight from the sharded
safetensors byte ranges (mmap, PressureAwareCache), runs a final RMSNorm + untied lm_head, and
returns REAL logits over the 151936-token vocab. An expert_hook lets a caller substitute
packed/decoded experts for the Gravity/Doctor original-vs-packed compression test. NLL/perplexity
are provided for capability scoring.

CRITICAL: this is Qwen3-MoE, NOT gpt-oss. The activation math differs on every axis that matters:

  * Expert MLP is the STANDARD SwiGLU with THREE separate projections per expert:
        down_proj( silu(gate_proj(x)) * up_proj(x) )
    NOT gpt-oss's single interleaved gate/up tensor with clamp(+-7) and (up+1)*glu.
  * RoPE is the STANDARD rotate_half convention (theta = rope_theta from config), applied AFTER
    the optional per-head q_norm/k_norm RMSNorm on head_dim. No attention sinks, no qkv bias
    (attention_bias == false).
  * Router is softmax-over-ALL-experts FIRST, then top-k, then (norm_topk_prob) renormalize the k
    selected weights - the transformers Qwen3MoeSparseMoeBlock convention. gpt-oss instead softmaxes
    only over the top-k logits.
  * GQA repeat_kv: 64 query heads share 4 KV heads (grp = 16).

Geometry (models/qwen3-235b-a22b/_meta/config.json, rev ac9c66cc...):
  qwen3_moe, 94 layers, hidden 4096, head_dim 128 (decoupled: q_proj out = 64*128 = 8192),
  64 Q / 4 KV heads, 128 experts, top-8, moe_intermediate 1536, vocab 151936, bf16,
  tie_word_embeddings false, rms_norm_eps 1e-6, rope_theta 5e6, q_norm/k_norm present.

HONESTY BOUNDARY. The real-source forward is UNTESTED-PENDING-SOURCE: the 438 GiB weight set is not
staged locally (only models/qwen3-235b-a22b/_meta/ metadata exists), so logits_for() has never been
run against the true shards. What IS tested (test_qwen_real_forward.py) is a tiny SYNTHETIC TWIN:
a 2-layer / 4-expert / hidden-16 qwen3_moe with random bf16 tensors written to real safetensors
shards, exercising the exact byte-range streaming loader + full forward chain + SwiGLU + router +
RoPE + q/k-norm end to end in milliseconds. No capability is claimed by this module.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bounded_cache import PressureAwareCache  # noqa: E402

# safetensors dtype tag -> (numpy dtype, bytes-per-element) for the read path we exercise.
_ST_BYTES = {"BF16": 2, "F16": 2, "F32": 4}

DEFAULT_SOURCE = Path("models/qwen3-235b-a22b")
DEFAULT_META = Path("models/qwen3-235b-a22b/_meta")


# ── bf16 <-> f32 bit ops ────────────────────────────────────────────────────────────────────
def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """BF16 stored as uint16 (top 16 bits of fp32) -> fp32 (little-endian host)."""
    u16 = np.asarray(bits, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    """fp32 -> BF16 uint16 by truncation (top 16 bits). Round-trips exactly through the reader."""
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    return (u32 >> np.uint32(16)).astype(np.uint16)


# ── correct Qwen3-MoE primitives ─────────────────────────────────────────────────────────────
def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Qwen3 RMSNorm over the last dim, computed in fp32. x:[..., D], weight:[D]."""
    xf = x.astype(np.float32)
    ms = np.mean(xf * xf, axis=-1, keepdims=True)
    return (xf / np.sqrt(ms + eps)) * weight.astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """STANDARD Qwen3 SwiGLU: silu(gate) * up. gate/up are the two SEPARATE projections' outputs.
    (Contrast gpt-oss apply_gate: interleaved split + clamp(+-7) + (up+1)*glu.)"""
    return silu(gate) * up


def rope(x: np.ndarray, pos: np.ndarray, head_dim: int, theta: float) -> np.ndarray:
    """Standard rotate_half RoPE. x:[seq, n_heads, head_dim] -> rotated, same shape."""
    half = head_dim // 2
    freqs = theta ** (-np.arange(half, dtype=np.float32) / half)          # [half]
    ang = np.outer(pos.astype(np.float32), freqs)                         # [seq, half]
    cos = np.cos(ang)[:, None, :]                                         # [seq,1,half]
    sin = np.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    # standard: q*cos + rotate_half(q)*sin, rotate_half = [-x2, x1]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def route_topk(logits: np.ndarray, top_k: int, norm_topk_prob: bool) -> tuple[np.ndarray, np.ndarray]:
    """Qwen3MoeSparseMoeBlock routing: softmax over ALL experts, take top-k, optionally renormalize.
    logits:[n_experts] -> (idx[k], weight[k]). softmax-first is monotone so argsort on logits==probs."""
    m = logits.max()
    probs = np.exp(logits - m)
    probs /= probs.sum()
    idx = np.argsort(-probs)[:top_k]
    w = probs[idx].astype(np.float32)
    if norm_topk_prob:
        w = w / max(float(w.sum()), 1e-20)
    return idx, w


# ── bounded-streaming safetensors reader (real HF sharded checkpoint) ─────────────────────────
class SafetensorsIndexReader:
    """Reads ONE tensor at a time straight from its shard's byte range (mmap), bf16 -> fp32.

    Never materializes a whole shard: it parses each shard's safetensors header (8-byte length +
    JSON) once, mmaps the file, and slices the exact [data_start+start : data_start+end] range of a
    named tensor. Bounded-memory by construction; the OS pages only the touched ranges. The 438 GiB
    Qwen3 source is NOT staged locally, so this path is untested-pending-source against the real
    weights (source_present() gates it); the synthetic twin exercises it byte-for-byte.
    """

    def __init__(self, source_dir: str | os.PathLike[str],
                 index_path: str | os.PathLike[str] | None = None):
        self.source_dir = Path(source_dir)
        if index_path is None:
            cand = self.source_dir / "model.safetensors.index.json"
            index_path = cand if cand.is_file() else DEFAULT_META / "model.safetensors.index.json"
        with open(index_path) as fh:
            idx = json.load(fh)
        self.weight_map: dict[str, str] = idx["weight_map"]
        self.metadata: dict[str, Any] = idx.get("metadata", {})
        self._headers: dict[str, tuple[dict[str, Any], int]] = {}   # shard -> (entries, data_start)
        self._mmaps: dict[str, mmap.mmap] = {}
        self._files: dict[str, Any] = {}

    # -- name / presence -------------------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self.weight_map

    def source_present(self) -> bool:
        """True iff every referenced shard file exists on disk."""
        return all((self.source_dir / s).is_file() for s in set(self.weight_map.values()))

    def shard_of(self, name: str) -> str:
        if name not in self.weight_map:
            raise KeyError(f"tensor not in index weight_map: {name}")
        return self.weight_map[name]

    # -- shard header / mmap ---------------------------------------------------------------
    def _shard(self, shard_file: str) -> tuple[dict[str, Any], int, mmap.mmap]:
        if shard_file not in self._headers:
            path = self.source_dir / shard_file
            fh = open(path, "rb")
            prefix = fh.read(8)
            if len(prefix) != 8:
                raise IOError(f"truncated safetensors length: {path}")
            hlen = struct.unpack("<Q", prefix)[0]
            header = fh.read(hlen)
            if len(header) != hlen:
                raise IOError(f"truncated safetensors header: {path}")
            entries = json.loads(header)
            entries.pop("__metadata__", None)
            self._headers[shard_file] = (entries, 8 + hlen)
            self._files[shard_file] = fh
            self._mmaps[shard_file] = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        entries, data_start = self._headers[shard_file]
        return entries, data_start, self._mmaps[shard_file]

    def _entry(self, name: str) -> tuple[dict[str, Any], int, mmap.mmap, dict[str, Any]]:
        entries, data_start, mm = self._shard(self.shard_of(name))
        row = entries.get(name)
        if row is None:
            raise KeyError(f"{name!r} absent from shard {self.shard_of(name)} header")
        return entries, data_start, mm, row

    # -- tensor reads ----------------------------------------------------------------------
    def bf16(self, name: str) -> np.ndarray:
        """Full tensor -> fp32 (bf16 source). One bounded mmap slice, no whole-shard load."""
        _, data_start, mm, row = self._entry(name)
        dtype, shape, (start, end) = row["dtype"], tuple(row["shape"]), row["data_offsets"]
        if dtype != "BF16":
            raise ValueError(f"{name}: expected BF16, got {dtype}")
        u16 = np.frombuffer(mm, dtype=np.uint16, count=(end - start) // 2,
                            offset=data_start + start)
        return bf16_bits_to_f32(u16).reshape(shape)

    def bf16_rows(self, name: str, rows: list[int]) -> np.ndarray:
        """Gather specific leading-dim rows of a 2D bf16 tensor without materializing the whole
        matrix (used for the embedding table). Returns [len(rows), cols] fp32."""
        _, data_start, mm, row = self._entry(name)
        shape, (start, _end) = tuple(row["shape"]), row["data_offsets"]
        if len(shape) != 2:
            raise ValueError(f"bf16_rows needs a 2D tensor, {name} is {shape}")
        _nrow, ncol = shape
        base = data_start + start
        out = np.empty((len(rows), ncol), dtype=np.float32)
        for i, r in enumerate(rows):
            off = base + int(r) * ncol * 2
            u16 = np.frombuffer(mm, dtype=np.uint16, count=ncol, offset=off)
            out[i] = bf16_bits_to_f32(u16)
        return out

    def close(self) -> None:
        for mm in self._mmaps.values():
            try:
                mm.close()
            except Exception:
                pass
        for fh in self._files.values():
            try:
                fh.close()
            except Exception:
                pass
        self._mmaps.clear(); self._files.clear(); self._headers.clear()


# ── geometry ─────────────────────────────────────────────────────────────────────────────────
class QwenGeometry:
    """Config-driven geometry; used for both the real 235B and the synthetic twin."""

    def __init__(self, config: dict[str, Any]):
        self.hidden = int(config["hidden_size"])
        self.n_layers = int(config["num_hidden_layers"])
        self.n_heads = int(config["num_attention_heads"])
        self.n_kv = int(config["num_key_value_heads"])
        self.head_dim = int(config.get("head_dim") or (self.hidden // self.n_heads))
        self.n_experts = int(config["num_experts"])
        self.top_k = int(config["num_experts_per_tok"])
        self.moe_inter = int(config["moe_intermediate_size"])
        self.vocab = int(config["vocab_size"])
        self.eps = float(config.get("rms_norm_eps", 1e-6))
        self.rope_theta = float(config.get("rope_theta", 1e6))
        self.norm_topk_prob = bool(config.get("norm_topk_prob", True))
        self.tie = bool(config.get("tie_word_embeddings", False))

    @property
    def q_out(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_out(self) -> int:
        return self.n_kv * self.head_dim


# ── the real full-model forward ────────────────────────────────────────────────────────────────
class QwenRealForward:
    """Bounded-streaming real Qwen3-MoE forward. expert_hook(layer, expert, w) -> w lets a caller
    substitute packed/decoded experts (for the original-vs-packed comparison); None = source-native.
    """

    def __init__(self, reader: SafetensorsIndexReader, geom: QwenGeometry,
                 cache: PressureAwareCache | None = None):
        self.reader = reader
        self.g = geom
        self.cache = cache if cache is not None else PressureAwareCache(
            "qwen-experts", disk_path=str(reader.source_dir), verbose=False)
        # q/k per-head RMSNorm is optional in qwen3_moe; detect from the index.
        self.has_qk_norm = reader.has("model.layers.0.self_attn.q_norm.weight")

    def source_present(self) -> bool:
        return self.reader.source_present()

    def _lm_head_name(self) -> str:
        return "model.embed_tokens.weight" if self.g.tie else "lm_head.weight"

    # -- attention -------------------------------------------------------------------------
    def _attention(self, L: int, x: np.ndarray) -> np.ndarray:
        g, r = self.g, self.reader
        seq = x.shape[0]
        h = rmsnorm(x, r.bf16(f"model.layers.{L}.input_layernorm.weight"), g.eps)
        qw = r.bf16(f"model.layers.{L}.self_attn.q_proj.weight")      # [q_out, hidden]
        kw = r.bf16(f"model.layers.{L}.self_attn.k_proj.weight")      # [kv_out, hidden]
        vw = r.bf16(f"model.layers.{L}.self_attn.v_proj.weight")      # [kv_out, hidden]
        ow = r.bf16(f"model.layers.{L}.self_attn.o_proj.weight")      # [hidden, q_out]
        q = (h @ qw.T).reshape(seq, g.n_heads, g.head_dim)
        k = (h @ kw.T).reshape(seq, g.n_kv, g.head_dim)
        v = (h @ vw.T).reshape(seq, g.n_kv, g.head_dim)
        if self.has_qk_norm:                                          # per-head RMSNorm on head_dim
            qn = r.bf16(f"model.layers.{L}.self_attn.q_norm.weight")
            kn = r.bf16(f"model.layers.{L}.self_attn.k_norm.weight")
            q = rmsnorm(q, qn, g.eps)
            k = rmsnorm(k, kn, g.eps)
        pos = np.arange(seq)
        q = rope(q, pos, g.head_dim, g.rope_theta)                    # AFTER q/k norm
        k = rope(k, pos, g.head_dim, g.rope_theta)
        grp = g.n_heads // g.n_kv
        scale = 1.0 / np.sqrt(g.head_dim)
        causal = np.triu(np.full((seq, seq), -1e30, dtype=np.float32), 1)
        out = np.zeros((seq, g.n_heads, g.head_dim), dtype=np.float32)
        for hh in range(g.n_heads):
            kv = hh // grp                                            # GQA repeat_kv
            scores = (q[:, hh] @ k[:, kv].T) * scale + causal
            scores -= scores.max(axis=1, keepdims=True)
            w = np.exp(scores); w /= w.sum(axis=1, keepdims=True)
            out[:, hh] = w @ v[:, kv]
        return out.reshape(seq, g.q_out) @ ow.T                       # [seq, hidden]

    # -- experts / MoE ---------------------------------------------------------------------
    def _load_expert(self, L: int, e: int) -> dict[str, np.ndarray]:
        r = self.reader
        return {
            "gate": r.bf16(f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight"),  # [moe_inter, hid]
            "up":   r.bf16(f"model.layers.{L}.mlp.experts.{e}.up_proj.weight"),    # [moe_inter, hid]
            "down": r.bf16(f"model.layers.{L}.mlp.experts.{e}.down_proj.weight"),  # [hid, moe_inter]
        }

    def _get_expert(self, L: int, e: int, expert_hook: Callable | None) -> dict[str, np.ndarray]:
        key = (L, e)
        ex = self.cache.get(key)
        if ex is None:
            ex = self._load_expert(L, e)
            if expert_hook is not None:
                ex = expert_hook(L, e, ex)
            self.cache.put(key, ex)
        return ex

    def _moe(self, L: int, x: np.ndarray, expert_hook: Callable | None) -> np.ndarray:
        g, r = self.g, self.reader
        seq = x.shape[0]
        gate_w = r.bf16(f"model.layers.{L}.mlp.gate.weight")          # router [n_experts, hidden]
        logits = x @ gate_w.T                                         # [seq, n_experts]
        out = np.zeros_like(x)
        for p in range(seq):
            idx, wts = route_topk(logits[p], g.top_k, g.norm_topk_prob)
            xp = x[p]
            acc = np.zeros(g.hidden, dtype=np.float32)
            for e, gw in zip(idx, wts):
                ex = self._get_expert(L, int(e), expert_hook)
                a = swiglu(ex["gate"] @ xp, ex["up"] @ xp)            # STANDARD SwiGLU
                acc += gw * (ex["down"] @ a)
            out[p] = acc
        return out

    # -- full forward ----------------------------------------------------------------------
    def logits_for(self, token_ids: list[int], *, positions: str = "last",
                   max_blocks: int | None = None, expert_hook: Callable | None = None,
                   progress: bool = False) -> np.ndarray:
        g, r = self.g, self.reader
        nb = g.n_layers if max_blocks is None else min(max_blocks, g.n_layers)
        x = r.bf16_rows("model.embed_tokens.weight", list(token_ids))   # [seq, hidden] bounded gather
        for L in range(nb):
            t0 = time.time()
            x = x + self._attention(L, x)                               # attention residual
            h = rmsnorm(x, r.bf16(f"model.layers.{L}.post_attention_layernorm.weight"), g.eps)
            x = x + self._moe(L, h, expert_hook)                        # MoE residual
            if progress:
                print(f"  block {L:2d}/{nb-1}  {time.time()-t0:5.1f}s  "
                      f"rms={float(np.sqrt(np.mean(x**2))):.3f}", flush=True)
        if nb < g.n_layers:
            return x                                                    # partial: raw hidden state
        x = rmsnorm(x, r.bf16("model.norm.weight"), g.eps)              # final norm
        sel = x if positions == "all" else x[-1:]
        lm = r.bf16(self._lm_head_name())                              # [vocab, hidden] (untied)
        return sel @ lm.T                                              # [.., vocab]

    def nll(self, token_ids: list[int], *, expert_hook: Callable | None = None,
            max_blocks: int | None = None) -> dict[str, float]:
        """Next-token NLL / perplexity over the sequence (teacher-forced)."""
        if len(token_ids) < 2:
            raise ValueError("need >= 2 tokens for next-token NLL")
        logits = self.logits_for(token_ids, positions="all", expert_hook=expert_hook,
                                 max_blocks=max_blocks)                # [seq, vocab]
        pred = logits[:-1].astype(np.float64)                         # predict token t+1 from t
        tgt = np.asarray(token_ids[1:], dtype=np.int64)
        pred -= pred.max(axis=-1, keepdims=True)
        logZ = np.log(np.exp(pred).sum(axis=-1))                      # [seq-1]
        chosen = pred[np.arange(pred.shape[0]), tgt]
        nll = float(np.mean(logZ - chosen))
        return {"nll": nll, "perplexity": float(np.exp(nll)), "n_pred": int(pred.shape[0])}


# ── real-source builder + smoke (UNTESTED-PENDING-SOURCE) ─────────────────────────────────────
def from_source(source_dir: str | os.PathLike[str] = DEFAULT_SOURCE,
                meta_dir: str | os.PathLike[str] = DEFAULT_META) -> QwenRealForward:
    """Build a forward over the real 235B checkpoint. Reads only config + index metadata to
    construct geometry; the weight shards are streamed lazily and are NOT required to exist yet."""
    with open(Path(meta_dir) / "config.json") as fh:
        config = json.load(fh)
    idx = Path(source_dir) / "model.safetensors.index.json"
    reader = SafetensorsIndexReader(source_dir, idx if idx.is_file()
                                    else Path(meta_dir) / "model.safetensors.index.json")
    return QwenRealForward(reader, QwenGeometry(config))


def _smoke(prompt: str, n_next: int = 8, max_blocks: int | None = None) -> dict:
    """Real-source smoke. Returns ok=None with a reason if the 438 GiB shards are not staged."""
    fwd = from_source()
    if not fwd.source_present():
        return {"ok": None, "reason": "Qwen3-235B source shards not present (only _meta staged); "
                "real-source forward is untested-pending-source"}
    from tokenizers import Tokenizer  # type: ignore
    tk = Tokenizer.from_file(str(DEFAULT_META / "tokenizer.json"))
    ids = tk.encode(prompt).ids
    t0 = time.time()
    logits = fwd.logits_for(ids, positions="last", max_blocks=max_blocks, progress=True)
    dt = time.time() - t0
    lg = logits[-1].astype(np.float64)
    top = np.argsort(-lg)[:n_next].tolist()
    probs = np.exp(lg[top] - lg.max()); probs /= np.exp(lg - lg.max()).sum()
    return {"ok": bool(np.isfinite(lg).all()), "prompt": prompt, "n_tokens_in": len(ids),
            "forward_seconds": round(dt, 1), "logits_shape": list(logits.shape),
            "top_next_tokens": [{"id": int(t), "token": tk.decode([int(t)]), "prob": round(float(p), 4)}
                                for t, p in zip(top, probs)]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Real full-model Qwen3-235B-A22B (qwen3_moe) forward.")
    ap.add_argument("--smoke", action="store_true", help="real-source smoke (needs staged shards)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n-next", type=int, default=8)
    ap.add_argument("--max-blocks", type=int, default=None)
    args = ap.parse_args(argv)
    if args.smoke:
        print(json.dumps(_smoke(args.prompt, n_next=args.n_next, max_blocks=args.max_blocks),
                         indent=2, default=str))
    else:
        print(json.dumps({"module": "qwen_real_forward",
                          "real_source": "untested-pending-source (438 GiB shards not staged)",
                          "note": "run pytest tools/condense/tests/test_qwen_real_forward.py"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

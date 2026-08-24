#!/usr/bin/env python3
"""Noetic organ census — which organs actually carry the function.

Measured, not assumed, on REAL activations from a CPU hybrid prefill of
the gravity uniform-Q4 artifact. Does not spawn a second 27B and does not
touch GPU (a llama-server is already resident on :52484).

Writes receipts/headless/NOETIC_ORGAN_CENSUS.json and prints the report.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts" / "headless" / "NOETIC_ORGAN_CENSUS.json"
ARTIFACT = Path.home() / "models" / "qwen38-gravity-uniform-q4-v1"
SOURCE_BF16 = Path.home() / "models" / "qwen3.8-27b-abliterated-bf16"
TOKENIZER_JSON = SOURCE_BF16 / "tokenizer.json"
LLAMA_TOKENIZE = "http://127.0.0.1:52484/tokenize"
TOKEN_NS_LEDGER = Path(
    "/Users/scammermike/Downloads/hawking-copy/receipts/ascent-2026-08-16/"
    "QWEN38_TOKEN_NS_LEDGER.json"
)

SCHEMA = "hawking.headless.noetic_organ_census.v1"

LAYERS = 64
HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
RMS_EPS = 1.0e-6
ROPE_THETA = 10_000_000.0
PARTIAL_ROTARY = 0.25
GQA_HEADS = 24
GQA_KV_HEADS = 4
GQA_HEAD_DIM = 256
GQA_ROTARY_DIM = 64
DN_K_HEADS = 16
DN_V_HEADS = 48
DN_VPK = 3
DN_K_DIM = 128
DN_V_DIM = 128
DN_CONV_K = 4
QKVZ_ROWS_PER_KEY = DN_K_DIM * 2 + DN_VPK * DN_V_DIM * 2  # 1024
BA_ROWS_PER_KEY = DN_VPK * 2  # 6
FULL_ATTN_INTERVAL = 4
HQ30UQ4_MAGIC = b"HQ30UQ4\0"
GROUP = 64
MAX_TOKENS = 128

ORGANS = ("embedding", "attention_gqa", "deltanet", "mlp", "output")

PROMPTS = (
    "Explain, in ordinary prose and at length, how a compiler turns a for-loop "
    "into basic blocks and then into machine code. Start from the AST.",
    "def merge(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n"
    "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n"
    "            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n",
    "Prove that the sum of the first n odd numbers is n squared, by induction.",
    "Write a step-by-step plan to bisect a Metal shader that silently drops a remainder tile.",
)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head() -> str | None:
    try:
        import subprocess

        r = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def is_gqa(layer: int) -> bool:
    return (layer + 1) % FULL_ATTN_INTERVAL == 0


def organ_of_name(name: str) -> str | None:
    if name.endswith("embed_tokens.weight"):
        return "embedding"
    if name.endswith("lm_head.weight") or name.endswith("model.norm.weight"):
        return "output"
    if ".self_attn." in name:
        return "attention_gqa"
    if ".linear_attn." in name:
        return "deltanet"
    if ".mlp." in name:
        return "mlp"
    if name.endswith("input_layernorm.weight"):
        # mixer-side norm; cost folded into the mixer that follows
        return None
    if name.endswith("post_attention_layernorm.weight"):
        return "mlp"
    return None


def mixer_organ(layer: int) -> str:
    return "attention_gqa" if is_gqa(layer) else "deltanet"


# ---------------------------------------------------------------------------
# fidelity — MUST reject 0.01 * x
# ---------------------------------------------------------------------------

def fidelity(true: np.ndarray, pred: np.ndarray) -> dict:
    """Scale-aware fidelity. Cosine alone is reported but is NOT the gate.

    Prior campaign: cosine(W, 0.01*W) == 1.000000 on every axis. The gate
    here is scale_aware = cosine * min(s, 1/s) together with relative L2
    and skill-vs-mean-null (SSE, which sees magnitude).
    """
    a = np.asarray(true, dtype=np.float64).reshape(-1)
    b = np.asarray(pred, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"fidelity shape {a.size} vs {b.size}")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    denom = max(na * nb, 1e-30)
    cosine = float(np.dot(a, b) / denom)
    scale = nb / max(na, 1e-30)
    scale_ratio = float(min(scale, 1.0 / scale)) if scale > 0 else 0.0
    rel_l2 = float(np.linalg.norm(a - b) / max(na, 1e-30))
    mean = float(a.mean())
    sse_c = float(np.square(b - a).sum())
    sse_n = float(np.square(a - mean).sum())
    skill = 1.0 - sse_c / max(sse_n, 1e-30)
    scale_aware = cosine * scale_ratio
    return {
        "cosine": cosine,
        "relative_l2": rel_l2,
        "skill_vs_mean_null": skill,
        "scale_ratio": scale,
        "scale_match": scale_ratio,
        "scale_aware": scale_aware,
        "norm_true": na,
        "norm_pred": nb,
        "rejects_perfect_cosine_as_sufficient": True,
    }


def survival_from_logits(base: np.ndarray, other: np.ndarray) -> dict:
    """Functional survival of `other` as a stand-in for `base` logits."""
    fid = fidelity(base, other)
    # softmax TV / KL on the last token (decode-relevant)
    last_b = base[-1].astype(np.float64)
    last_o = other[-1].astype(np.float64)
    last_b = last_b - last_b.max()
    last_o = last_o - last_o.max()
    pb = np.exp(last_b)
    pb /= pb.sum()
    po = np.exp(last_o)
    po /= po.sum()
    kl = float(np.sum(pb * (np.log(pb + 1e-30) - np.log(po + 1e-30))))
    tv = 0.5 * float(np.abs(pb - po).sum())
    argmax_match = float(np.mean(np.argmax(base, axis=-1) == np.argmax(other, axis=-1)))
    # Survival in [0, 1]: 1 = indistinguishable. Relative L2 0 → 1, ≥1 → 0.
    survival = float(max(0.0, 1.0 - min(1.0, fid["relative_l2"])))
    return {
        **fid,
        "kl_last_token": kl,
        "tv_last_token": tv,
        "argmax_match": argmax_match,
        "survival": survival,
    }


# ---------------------------------------------------------------------------
# codec
# ---------------------------------------------------------------------------

def read_f32v2(path: Path, shape: list[int]) -> np.ndarray:
    data = np.memmap(path, dtype=np.uint8, mode="r")
    n = int(np.frombuffer(data[:8], dtype="<u8")[0])
    need = 8 + n * 4
    if data.size != need:
        raise ValueError(f"{path.name} f32v2 size {data.size} != {need}")
    vals = np.frombuffer(data[8:], dtype="<f4").copy()
    return vals.reshape(shape)


def _q4_layout(payload: np.ndarray) -> tuple[list[int], int, int, int]:
    if payload[:8].tobytes() != HQ30UQ4_MAGIC:
        raise ValueError("not HQ30UQ4")
    version, group_size, rank, reserved = struct.unpack_from("<IIHH", payload, 8)
    elements, rtail = struct.unpack_from("<QI", payload, 20)
    if version != 1 or group_size != GROUP or reserved != 0 or rtail != 0:
        raise ValueError(
            f"hq30uq4 header ver={version} gs={group_size} reserved={reserved}/{rtail}"
        )
    dims = list(struct.unpack_from("<" + "I" * rank, payload, 32))
    after = 32 + 4 * rank
    groups = (elements + group_size - 1) // group_size
    scale_off = after
    code_off = after + groups * 2
    return dims, elements, scale_off, code_off


def read_q4(path: Path, shape: list[int]) -> np.ndarray:
    payload = np.memmap(path, dtype=np.uint8, mode="r")
    dims, elements, scale_off, code_off = _q4_layout(payload)
    if dims != list(shape):
        raise ValueError(f"{path.name} q4 dims {dims} != {shape}")
    groups = (elements + GROUP - 1) // GROUP
    scales = np.frombuffer(payload[scale_off:code_off], dtype="<f2").astype(np.float32)
    codes = np.frombuffer(payload[code_off:], dtype=np.uint8)
    if scales.size != groups or codes.size != groups * (GROUP // 2):
        raise ValueError(f"{path.name} q4 plane size mismatch")
    low = (codes & 0x0F).astype(np.int16) - 8
    high = (np.right_shift(codes, 4).astype(np.int16)) - 8
    q = np.empty(groups * GROUP, dtype=np.float32)
    q[0::2] = low
    q[1::2] = high
    q *= np.repeat(scales, GROUP)
    return q[: int(np.prod(shape))].reshape(shape)


def read_q4_rows(path: Path, shape: list[int], rows: np.ndarray) -> np.ndarray:
    """Dequant only selected rows of a rank-2 HQ30UQ4 (embed gather)."""
    rows_n, cols = shape
    if cols % GROUP != 0:
        raise ValueError("embed cols not divisible by group")
    payload = np.memmap(path, dtype=np.uint8, mode="r")
    dims, elements, scale_off, code_off = _q4_layout(payload)
    if dims != list(shape):
        raise ValueError(f"embed dims {dims} != {shape}")
    groups_per_row = cols // GROUP
    code_bpg = GROUP // 2
    scales = np.frombuffer(payload[scale_off:code_off], dtype="<f2")
    codes = payload[code_off:]
    out = np.empty((len(rows), cols), dtype=np.float32)
    for i, r in enumerate(rows):
        r = int(r)
        if r < 0 or r >= rows_n:
            raise ValueError(f"token id {r} outside embed rows {rows_n}")
        g0 = r * groups_per_row
        sc = scales[g0 : g0 + groups_per_row].astype(np.float32)
        cb = np.frombuffer(
            codes[g0 * code_bpg : (g0 + groups_per_row) * code_bpg], dtype=np.uint8
        )
        low = (cb & 0x0F).astype(np.int16) - 8
        high = (np.right_shift(cb, 4).astype(np.int16)) - 8
        q = np.empty(cols, dtype=np.float32)
        q[0::2] = low
        q[1::2] = high
        q *= np.repeat(sc, GROUP)
        out[i] = q
    return out


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

class Artifact:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.by_name = {t["name"]: t for t in self.manifest["tensors"]}
        self.tensors_dir = root / "tensors"

    def path(self, name: str) -> Path:
        row = self.by_name[name]
        return self.tensors_dir / row["artifact"]

    def load(self, name: str) -> np.ndarray:
        row = self.by_name[name]
        p = self.path(name)
        if row["kind"] == "f32":
            return read_f32v2(p, row["shape"])
        if row["kind"] == "q4":
            return read_q4(p, row["shape"])
        raise ValueError(f"unknown kind {row['kind']} for {name}")

    def layer_name(self, layer: int, suffix: str) -> str:
        return f"language_model.model.layers.{layer}.{suffix}"


def gemm(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """w [out, in] @ x [T, in] -> [T, out]."""
    return x @ w.T


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

def tokenize_llama(text: str) -> list[int] | None:
    body = json.dumps({"content": text, "add_special": False}).encode()
    req = urllib.request.Request(
        LLAMA_TOKENIZE, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    toks = payload.get("tokens")
    if not isinstance(toks, list) or not toks:
        return None
    return [int(t) for t in toks]


def tokenize_bpe(text: str) -> list[int]:
    """Byte-level BPE from tokenizer.json. Used only if llama-server tokenize fails."""
    spec = json.loads(TOKENIZER_JSON.read_text())
    vocab: dict[str, int] = spec["model"]["vocab"]
    merges = spec["model"]["merges"]
    rank = {}
    for i, m in enumerate(merges):
        if isinstance(m, str):
            a, b = m.split()
        else:
            a, b = m
        rank[(a, b)] = i
    # GPT2-style bytes-to-unicode
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_encoder = dict(zip(bs, (chr(c) for c in cs)))
    tokens: list[int] = []
    for ch in text:
        piece = "".join(byte_encoder[b] for b in ch.encode("utf-8"))
        parts = list(piece)
        while len(parts) >= 2:
            pairs = [(rank.get((parts[i], parts[i + 1]), 10**9), i) for i in range(len(parts) - 1)]
            best = min(pairs)
            if best[0] == 10**9:
                break
            i = best[1]
            parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2 :]
        for p in parts:
            tokens.append(int(vocab.get(p, vocab.get("\ufffd", 0))))
    return tokens


def tokenize(text: str) -> tuple[list[int], str]:
    toks = tokenize_llama(text)
    if toks is not None:
        return toks, (
            "live llama-server :52484 /tokenize "
            "(Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf tokenizer, not a second load)"
        )
    if TOKENIZER_JSON.is_file():
        return tokenize_bpe(text), f"local BPE {TOKENIZER_JSON}"
    raise RuntimeError("no tokenizer available")


# ---------------------------------------------------------------------------
# math: RMSNorm, RoPE, GQA, DeltaNet
# ---------------------------------------------------------------------------

def rmsnorm_delta(x: np.ndarray, delta: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + RMS_EPS)
    return (x / rms) * (1.0 + delta)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def softmax_last(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def causal_conv1d_silu(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """x (T, C), weight (C, K) as packed [C, K, 1] squeezed. Matches metal update."""
    t_len, _c = x.shape
    k = weight.shape[-1]
    acc = np.zeros_like(x)
    for tap in range(k):
        lag = k - 1 - tap
        if lag == 0:
            acc += x * weight[:, tap]
        else:
            acc[lag:] += x[:-lag] * weight[:, tap]
    return silu(acc)


def l2norm(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + RMS_EPS))


def apply_rope(q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Metal rotate_half on first 64 of 256, θ=1e7. q/k: (T, H, D)."""
    t_len, _h, _d = q.shape
    half = GQA_ROTARY_DIM // 2
    idx = np.arange(half, dtype=np.float32)
    inv = ROPE_THETA ** (-2.0 * idx / float(GQA_ROTARY_DIM))
    pos = np.arange(t_len, dtype=np.float32)
    angle = pos[:, None] * inv[None, :]  # (T, 32)
    cos = np.cos(angle)
    sin = np.sin(angle)

    def rot(x: np.ndarray) -> np.ndarray:
        # x (T, H, 256)
        out = x.copy()
        a = x[:, :, :half]
        b = x[:, :, half:GQA_ROTARY_DIM]
        c = cos[:, None, :]
        s = sin[:, None, :]
        out[:, :, :half] = a * c - b * s
        out[:, :, half:GQA_ROTARY_DIM] = a * s + b * c
        return out

    return rot(q), rot(k)


def gqa_forward(
    x_norm: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wo: np.ndarray,
    q_delta: np.ndarray,
    k_delta: np.ndarray,
) -> np.ndarray:
    t_len = x_norm.shape[0]
    qg = gemm(wq, x_norm).reshape(t_len, GQA_HEADS, 2 * GQA_HEAD_DIM)
    q = qg[:, :, :GQA_HEAD_DIM]
    gate = qg[:, :, GQA_HEAD_DIM:]
    k = gemm(wk, x_norm).reshape(t_len, GQA_KV_HEADS, GQA_HEAD_DIM)
    v = gemm(wv, x_norm).reshape(t_len, GQA_KV_HEADS, GQA_HEAD_DIM)
    q = rmsnorm_delta(q, q_delta)
    k = rmsnorm_delta(k, k_delta)
    q, k = apply_rope(q, k)
    k_rep = np.repeat(k, GQA_HEADS // GQA_KV_HEADS, axis=1)
    v_rep = np.repeat(v, GQA_HEADS // GQA_KV_HEADS, axis=1)
    scale = 1.0 / math.sqrt(GQA_HEAD_DIM)
    scores = np.einsum("thd,shd->hts", q, k_rep, dtype=np.float32) * scale
    causal = np.triu(np.ones((t_len, t_len), dtype=bool), 1)
    scores[:, causal] = -1e9
    w = softmax_last(scores)
    attn = np.einsum("hts,shd->thd", w, v_rep, dtype=np.float32)
    attn = attn.reshape(t_len, GQA_HEADS * GQA_HEAD_DIM)
    attn = attn * (1.0 / (1.0 + np.exp(-np.clip(gate.reshape(t_len, -1), -60, 60))))
    return gemm(wo, attn)


def _unfuse_qkvz(fused: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """fused (T, 16384) per-key-head Q128 K128 V384 Z384 -> q,k (T,16,128), v,z (T,48,128)."""
    t_len = fused.shape[0]
    q = np.empty((t_len, DN_K_HEADS, DN_K_DIM), np.float32)
    k = np.empty_like(q)
    v = np.empty((t_len, DN_V_HEADS, DN_V_DIM), np.float32)
    z = np.empty_like(v)
    for kh in range(DN_K_HEADS):
        base = kh * QKVZ_ROWS_PER_KEY
        q[:, kh, :] = fused[:, base : base + DN_K_DIM]
        k[:, kh, :] = fused[:, base + DN_K_DIM : base + 2 * DN_K_DIM]
        v0 = base + 2 * DN_K_DIM
        z0 = v0 + DN_VPK * DN_V_DIM
        v[:, kh * DN_VPK : (kh + 1) * DN_VPK, :] = fused[:, v0:z0].reshape(
            t_len, DN_VPK, DN_V_DIM
        )
        z[:, kh * DN_VPK : (kh + 1) * DN_VPK, :] = fused[:, z0 : z0 + DN_VPK * DN_V_DIM].reshape(
            t_len, DN_VPK, DN_V_DIM
        )
    return q, k, v, z


def _unfuse_ba(ba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ba (T, 96) packed [key_head][b×3, a×3] -> b,a (T, 48)."""
    t_len = ba.shape[0]
    b = np.empty((t_len, DN_V_HEADS), np.float32)
    a = np.empty_like(b)
    for kh in range(DN_K_HEADS):
        base = kh * BA_ROWS_PER_KEY
        sl = slice(kh * DN_VPK, (kh + 1) * DN_VPK)
        b[:, sl] = ba[:, base : base + DN_VPK]
        a[:, sl] = ba[:, base + DN_VPK : base + 2 * DN_VPK]
    return b, a


def gated_delta_seq(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, decay: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Metal vi recurrence over T. q,k,v (T,H,D), decay/beta (T,H) or (H,)."""
    t_len, heads, dk = q.shape
    dv = v.shape[-1]
    if decay.ndim == 1:
        decay = np.broadcast_to(decay, (t_len, heads))
        beta = np.broadcast_to(beta, (t_len, heads))
    state = np.zeros((heads, dk, dv), dtype=np.float32)
    out = np.empty((t_len, heads, dv), dtype=np.float32)
    for t in range(t_len):
        state *= decay[t, :, None, None]
        kv_mem = np.einsum("hkv,hk->hv", state, k[t], dtype=np.float32)
        delta = (v[t] - kv_mem) * beta[t, :, None]
        state += np.einsum("hk,hv->hkv", k[t], delta, dtype=np.float32)
        out[t] = np.einsum("hkv,hk->hv", state, q[t], dtype=np.float32)
    return out


def deltanet_forward(
    x_norm: np.ndarray,
    w_qkvz: np.ndarray,
    w_ba: np.ndarray,
    w_out: np.ndarray,
    conv_w: np.ndarray,
    a_log: np.ndarray,
    dt_bias: np.ndarray,
    norm_w: np.ndarray,
) -> np.ndarray:
    t_len = x_norm.shape[0]
    qkvz = gemm(w_qkvz, x_norm)
    ba = gemm(w_ba, x_norm)
    q, k, v, z = _unfuse_qkvz(qkvz)
    b, a = _unfuse_ba(ba)
    # conv on QKV channels only, in fused channel order matching metal:
    # for each key head: Q128, K128, V384  (Z not convolved)
    qkv_ch = np.concatenate(
        [
            q.reshape(t_len, DN_K_HEADS * DN_K_DIM),
            k.reshape(t_len, DN_K_HEADS * DN_K_DIM),
            v.reshape(t_len, DN_V_HEADS * DN_V_DIM),
        ],
        axis=1,
    )
    conv_w = conv_w.reshape(conv_w.shape[0], DN_CONV_K)
    qkv_c = causal_conv1d_silu(qkv_ch, conv_w)
    q = qkv_c[:, : DN_K_HEADS * DN_K_DIM].reshape(t_len, DN_K_HEADS, DN_K_DIM)
    k = qkv_c[:, DN_K_HEADS * DN_K_DIM : 2 * DN_K_HEADS * DN_K_DIM].reshape(
        t_len, DN_K_HEADS, DN_K_DIM
    )
    v = qkv_c[:, 2 * DN_K_HEADS * DN_K_DIM :].reshape(t_len, DN_V_HEADS, DN_V_DIM)
    q = l2norm(q) * (1.0 / math.sqrt(DN_K_DIM))
    k = l2norm(k)
    q = np.repeat(q, DN_VPK, axis=1)
    k = np.repeat(k, DN_VPK, axis=1)
    # g = -exp(A_log) * softplus(a + dt_bias); decay = exp(g)
    xdt = a + dt_bias[None, :]
    softplus = np.where(
        xdt > 0,
        xdt + np.log1p(np.exp(-np.abs(xdt))),
        np.log1p(np.exp(-np.abs(xdt))),
    )
    g = -np.exp(a_log)[None, :] * softplus
    decay = np.exp(g).astype(np.float32)
    beta = (1.0 / (1.0 + np.exp(-np.clip(b, -60, 60)))).astype(np.float32)
    rec = gated_delta_seq(q, k, v, decay, beta)
    # gated RMSNorm per value head, shared weight[head_dim]
    rec_f = rec.reshape(t_len * DN_V_HEADS, DN_V_DIM)
    z_f = z.reshape(t_len * DN_V_HEADS, DN_V_DIM)
    rms = np.sqrt(np.mean(rec_f * rec_f, axis=-1, keepdims=True) + RMS_EPS)
    gated = (rec_f / rms) * norm_w * silu(z_f)
    gated = gated.reshape(t_len, DN_V_HEADS * DN_V_DIM)
    return gemm(w_out, gated)


def mlp_forward(x_norm: np.ndarray, wg: np.ndarray, wu: np.ndarray, wd: np.ndarray) -> np.ndarray:
    return gemm(wd, silu(gemm(wg, x_norm)) * gemm(wu, x_norm))


# ---------------------------------------------------------------------------
# activation stats
# ---------------------------------------------------------------------------

def participation_ratio(x: np.ndarray) -> dict:
    """x: (N, D). Reports measured rank stats; NEVER invents a zero."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    xc = x - x.mean(axis=0, keepdims=True)
    # Gram on the smaller side
    if n >= d:
        cov = (xc.T @ xc) / max(n - 1, 1)
        evals = np.linalg.eigvalsh(cov)
        ambient_ok = True
    else:
        gram = (xc @ xc.T) / max(n - 1, 1)
        evals = np.linalg.eigvalsh(gram)
        ambient_ok = False
    evals = np.clip(evals, 0.0, None)
    evals = np.sort(evals)[::-1]
    s = float(evals.sum())
    if s <= 0:
        return {
            "n_rows": int(n),
            "n_cols": int(d),
            "status": "NOT_MEASURED",
            "reason": "activation matrix has zero energy after centering",
        }
    pr = float((s * s) / max(float(np.square(evals).sum()), 1e-30))
    cume = np.cumsum(evals) / s
    rank99 = int(np.searchsorted(cume, 0.99) + 1)
    out = {
        "n_rows": int(n),
        "n_cols": int(d),
        "status": "MEASURED",
        "participation_ratio": pr,
        "effective_rank_of_captured_spectrum": pr,
        "rank_99_of_captured_spectrum": rank99,
        "spectrum_dim": int(evals.size),
        "top5_energy_frac": [float(v) for v in (evals[:5] / s)],
    }
    if not ambient_ok:
        out["ambient_rank_99"] = "NOT_MEASURED"
        out["ambient_rank_99_reason"] = (
            f"n_rows={n} < hidden={d}; 99% energy rank is of the "
            f"{evals.size}-dimensional captured spectrum, not the ambient {d}-space"
        )
    else:
        out["ambient_rank_99"] = rank99
    return out


def residual_sparsity(x: np.ndarray) -> dict:
    """Cheapest structural fit = rank-1 SVD (after mean removal)."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    mean = x.mean(axis=0)
    xc = x - mean
    # economy SVD
    k = min(n, d)
    if k < 1:
        return {"status": "NOT_MEASURED", "reason": "empty activation matrix"}
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    rank1 = np.outer(u[:, 0] * s[0], vt[0]) if s.size else np.zeros_like(xc)
    residual = xc - rank1
    rms_x = float(np.sqrt(np.mean(xc * xc)))
    rms_r = float(np.sqrt(np.mean(residual * residual)))
    thr = 0.01 * rms_x if rms_x > 0 else 0.0
    sparse = float(np.mean(np.abs(residual) < thr)) if thr > 0 else 0.0
    energy_rank1 = float((s[0] * s[0]) / max(float(np.square(s).sum()), 1e-30)) if s.size else 0.0
    return {
        "status": "MEASURED",
        "fit": "mean + rank-1 SVD",
        "n_rows": int(n),
        "n_cols": int(d),
        "rank1_energy_frac": energy_rank1,
        "residual_rms_over_centered_rms": rms_r / max(rms_x, 1e-30),
        "residual_sparsity_at_0p01_rms": sparse,
        "threshold": thr,
    }


def fingerprint(w: np.ndarray, n: int = 4096) -> dict:
    """Scale-aware structural signature: row-RMS (and a column-RMS sketch).

    A strided flatten of a 60M-element matrix is ~noise and will report
    pairwise cosine ≈ 0 even when energy profiles align. Row-RMS is the
    cheapest structural fit that still sees magnitude.
    """
    arr = np.asarray(w, dtype=np.float32)
    frob = float(np.linalg.norm(arr.astype(np.float64)))
    mean_abs = float(np.mean(np.abs(arr)))
    if arr.ndim == 1:
        vec = arr.astype(np.float64)
    else:
        row_rms = np.sqrt(np.mean(arr.astype(np.float64) ** 2, axis=1))
        col_rms = np.sqrt(np.mean(arr.astype(np.float64) ** 2, axis=0))
        step = max(1, col_rms.size // 2048)
        vec = np.concatenate([row_rms, col_rms[::step][:2048]])
    return {"frobenius": frob, "mean_abs": mean_abs, "subsample": vec}


def pairwise_scale_aware(fps: list[tuple[int, dict]]) -> dict:
    if len(fps) < 2:
        return {
            "status": "NOT_MEASURED",
            "reason": f"need ≥2 layers, have {len(fps)}",
        }
    scores = []
    cosines = []
    scale_cv_num = []
    for i in range(len(fps)):
        scale_cv_num.append(fps[i][1]["frobenius"])
        for j in range(i + 1, len(fps)):
            a = fps[i][1]["subsample"]
            b = fps[j][1]["subsample"]
            m = fidelity(a, b)
            scores.append(m["scale_aware"])
            cosines.append(m["cosine"])
    arr = np.array(scale_cv_num, dtype=np.float64)
    cv = float(arr.std() / max(arr.mean(), 1e-30))
    return {
        "status": "MEASURED",
        "n_layers": len(fps),
        "n_pairs": len(scores),
        "mean_pairwise_scale_aware": float(np.mean(scores)),
        "min_pairwise_scale_aware": float(np.min(scores)),
        "mean_pairwise_cosine": float(np.mean(cosines)),
        "frobenius_cv_across_layers": cv,
        "metric": "cosine(row-RMS || col-RMS-sketch)*min(s,1/s); "
        "cosine alone is scale-blind and is reported only as a contrast",
    }


# ---------------------------------------------------------------------------
# token_ns share (measured ledger — not re-derived)
# ---------------------------------------------------------------------------

def load_token_ns() -> dict:
    if not TOKEN_NS_LEDGER.is_file():
        return {
            "status": "NOT_MEASURED",
            "reason": f"token_ns ledger missing at {TOKEN_NS_LEDGER}",
        }
    d = json.loads(TOKEN_NS_LEDGER.read_text())
    comps = {c["component"]: c for c in d.get("components", [])}
    probes = {p["class"]: p for p in d.get("probes", [])}
    closure = d.get("closure", {})
    weight = d.get("weight_bytes", {})
    return {
        "status": "MEASURED",
        "source": str(TOKEN_NS_LEDGER),
        "schema": d.get("schema"),
        "vehicle": d.get("vehicle"),
        "median_gpu_ns": d.get("median_gpu_ns"),
        "total_token_ns": closure.get("total_token_ns"),
        "weight_bytes": weight,
        "components": {
            k: {"ns_per_token": v["ns_per_token"], "pct_of_token_wall": v["pct_of_token_wall"]}
            for k, v in comps.items()
        },
        "isolated_gemv_gpu_ns": {
            k: p["full_median_gpu_ns"] for k, p in probes.items()
        },
        "embed_gather_ns": 4999.0,  # named in ledger closure residual_reason
    }


def organ_token_ns(ledger: dict) -> dict[str, dict]:
    if ledger.get("status") != "MEASURED":
        return {
            o: {"status": "NOT_MEASURED", "reason": ledger.get("reason")}
            for o in ORGANS
        }
    gemv = ledger["isolated_gemv_gpu_ns"]
    c = ledger["components"]
    total = float(ledger["total_token_ns"])
    # Isolated GEMV probes already contain that family's weight stream.
    # Non-GEMV named components attach to the organ that owns them.
    out = {
        "embedding": ledger["embed_gather_ns"],
        "attention_gqa": gemv.get("gqa", 0.0) + c["gqa"]["ns_per_token"] + c["kv_state"]["ns_per_token"],
        "deltanet": gemv.get("dn", 0.0) + c["deltanet"]["ns_per_token"],
        "mlp": gemv.get("mlp", 0.0) + c["dense_swiglu"]["ns_per_token"],
        "output": gemv.get("lm_head", 0.0) + c["terminal_head"]["ns_per_token"],
    }
    # norms split across mixers + mlp + output; keep as a declared remainder
    # rather than stuffing them into a wrong organ.
    attributed = sum(out.values())
    return {
        k: {
            "status": "MEASURED",
            "ns_per_token": v,
            "share_of_token_ns": v / total,
            "arithmetic": _ns_arithmetic(k, gemv, c, ledger),
        }
        for k, v in out.items()
    } | {
        "_unattributed_host_norm_sync": {
            "status": "MEASURED",
            "ns_per_token": total - attributed,
            "share_of_token_ns": (total - attributed) / total,
            "note": "host_preparation + command_submission + normalization + "
            "synchronization + unattributed_residual minus embed_gather already assigned",
        }
    }


def _ns_arithmetic(organ: str, gemv: dict, c: dict, ledger: dict) -> str:
    if organ == "embedding":
        return "embed_gather_ns = 4999 (ledger closure residual_reason)"
    if organ == "attention_gqa":
        return (
            f"isolated_gqa_gemv {gemv.get('gqa')} + gqa_component {c['gqa']['ns_per_token']} "
            f"+ kv_state {c['kv_state']['ns_per_token']}"
        )
    if organ == "deltanet":
        return f"isolated_dn_gemv {gemv.get('dn')} + deltanet_component {c['deltanet']['ns_per_token']}"
    if organ == "mlp":
        return f"isolated_mlp_gemv {gemv.get('mlp')} + dense_swiglu {c['dense_swiglu']['ns_per_token']}"
    if organ == "output":
        return (
            f"isolated_lm_head_gemv {gemv.get('lm_head')} + terminal_head {c['terminal_head']['ns_per_token']}"
        )
    return ""


# ---------------------------------------------------------------------------
# physical bytes from the live manifest
# ---------------------------------------------------------------------------

def physical_cost(art: Artifact) -> dict[str, dict]:
    bytes_ = defaultdict(int)
    elements = defaultdict(int)
    tensors = defaultdict(int)
    q4 = defaultdict(int)
    f32 = defaultdict(int)
    for t in art.manifest["tensors"]:
        organ = organ_of_name(t["name"])
        if organ is None:
            if t["name"].endswith("input_layernorm.weight"):
                # folded into mixer of that layer
                layer = int(t["name"].split("layers.")[1].split(".")[0])
                organ = mixer_organ(layer)
            else:
                continue
        bytes_[organ] += int(t["bytes"])
        elements[organ] += int(t["elements"])
        tensors[organ] += 1
        if t["kind"] == "q4":
            q4[organ] += 1
        else:
            f32[organ] += 1
    wb = {}
    ledger_path_ok = TOKEN_NS_LEDGER.is_file()
    ledger = json.loads(TOKEN_NS_LEDGER.read_text()) if ledger_path_ok else {}
    wbytes = ledger.get("weight_bytes", {})
    active = {
        "embedding": wbytes.get("embed_row_bytes", 2720),
        "attention_gqa": wbytes.get("full_attn_bytes"),
        "deltanet": wbytes.get("linear_attn_bytes"),
        "mlp": wbytes.get("mlp_bytes"),
        "output": wbytes.get("lm_head_bytes"),
    }
    out = {}
    for o in ORGANS:
        out[o] = {
            "bytes": int(bytes_[o]),
            "elements": int(elements[o]),
            "tensor_count": int(tensors[o]),
            "q4_tensors": int(q4[o]),
            "f32_tensors": int(f32[o]),
            "active_bytes_per_token": active[o],
            "active_bytes_note": (
                "embed: one gathered row per token, table excluded from the stream. "
                "All other GEMV organs stream their full packed payload every decode step."
                if o == "embedding"
                else "decode streams the full packed organ every token (no MoE)."
            ),
        }
    return out


# ---------------------------------------------------------------------------
# bf16 spot-check (codec) — optional, never required for the ranking
# ---------------------------------------------------------------------------

def bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    bits = u16 << 16
    return np.frombuffer(bits.tobytes(), dtype=np.float32).copy()


def spot_check_q4_against_bf16(art: Artifact, name: str) -> dict:
    index_path = SOURCE_BF16 / "model.safetensors.index.json"
    if not index_path.is_file():
        return {"status": "NOT_MEASURED", "reason": "bf16 source index missing"}
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map", {})
    aliases = [
        name,
        "model." + name,
        name.replace("language_model.model.", "model.language_model."),
        name.replace("language_model.", "model.language_model."),
    ]
    src_name = next((a for a in aliases if a in weight_map), None)
    shard = weight_map.get(src_name) if src_name else None
    if not shard:
        return {"status": "NOT_MEASURED", "reason": f"{name} not in bf16 weight_map (tried {aliases})"}
    shard_path = SOURCE_BF16 / shard
    if not shard_path.is_file():
        return {"status": "NOT_MEASURED", "reason": f"shard missing {shard_path}"}
    with open(shard_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        info = header.get(src_name) or header.get(name)
        if not info:
            return {"status": "NOT_MEASURED", "reason": f"{src_name} not in shard header"}
        begin, end = info["data_offsets"]
        f.seek(8 + header_len + begin)
        raw = f.read(end - begin)
    dtype = info.get("dtype", "")
    shape = info["shape"]
    if dtype in ("BF16", "BFLOAT16"):
        src = bf16_to_f32(raw).reshape(shape)
    elif dtype in ("F32", "FLOAT32"):
        src = np.frombuffer(raw, dtype="<f4").reshape(shape)
    else:
        return {"status": "NOT_MEASURED", "reason": f"unsupported dtype {dtype}"}
    try:
        q = art.load(name)
    except Exception as e:
        return {"status": "NOT_MEASURED", "reason": f"q4 load failed: {e}"}
    # compare a strided subset to keep this cheap
    a = src.reshape(-1)[::17][:50000]
    b = q.reshape(-1)[::17][:50000]
    return {"status": "MEASURED", "name": name, "dtype": dtype, **fidelity(a, b)}


# ---------------------------------------------------------------------------
# forward with ablations
# ---------------------------------------------------------------------------

def run_forward(art: Artifact, tokens: np.ndarray, progress) -> dict:
    t0 = time.perf_counter()
    t_len = int(tokens.shape[0])
    embed_row = art.by_name["language_model.model.embed_tokens.weight"]
    progress(f"gather {t_len} embed rows from {embed_row['shape']}")
    h_embed = read_q4_rows(
        art.path("language_model.model.embed_tokens.weight"),
        embed_row["shape"],
        tokens,
    )
    streams = {
        "base": h_embed.copy(),
        "null_embedding": np.zeros_like(h_embed),
        "null_attention_gqa": h_embed.copy(),
        "null_deltanet": h_embed.copy(),
        "null_mlp": h_embed.copy(),
    }
    # captures on the BASE stream only
    mixer_in_gqa = []
    mixer_in_dn = []
    mlp_in = []
    post_swiglu = []
    fps: dict[str, list] = defaultdict(list)
    local_wx: dict[str, dict] = {}

    for layer in range(LAYERS):
        ln = art.load(art.layer_name(layer, "input_layernorm.weight"))
        gqa = is_gqa(layer)
        organ = mixer_organ(layer)
        # snapshot mixer inputs for every stream
        x_norm = {k: rmsnorm_delta(h, ln) for k, h in streams.items()}
        if gqa:
            mixer_in_gqa.append(x_norm["base"].copy())
            wq = art.load(art.layer_name(layer, "self_attn.q_proj.weight"))
            wk = art.load(art.layer_name(layer, "self_attn.k_proj.weight"))
            wv = art.load(art.layer_name(layer, "self_attn.v_proj.weight"))
            wo = art.load(art.layer_name(layer, "self_attn.o_proj.weight"))
            qn = art.load(art.layer_name(layer, "self_attn.q_norm.weight"))
            kn = art.load(art.layer_name(layer, "self_attn.k_norm.weight"))
            fps["attention_gqa"].append((layer, fingerprint(wq)))
            if "attention_gqa" not in local_wx:
                y = gemm(wq, x_norm["base"])
                local_wx["attention_gqa"] = {
                    "site": f"layer {layer} q_proj on post-input_ln hidden",
                    "y": y,
                    "y_scale": 0.01 * y,
                    "y_null": np.zeros_like(y),
                    "x": x_norm["base"],
                    "w_frob": float(np.linalg.norm(wq.astype(np.float64))),
                }
            mix = {}
            for k, xn in x_norm.items():
                mix[k] = gqa_forward(xn, wq, wk, wv, wo, qn, kn)
            del wq, wk, wv, wo
        else:
            mixer_in_dn.append(x_norm["base"].copy())
            w_qkvz = art.load(art.layer_name(layer, "linear_attn.in_proj_qkvz.weight"))
            w_ba = art.load(art.layer_name(layer, "linear_attn.in_proj_ba.weight"))
            w_out = art.load(art.layer_name(layer, "linear_attn.out_proj.weight"))
            conv_w = art.load(art.layer_name(layer, "linear_attn.conv1d.weight"))
            a_log = art.load(art.layer_name(layer, "linear_attn.A_log"))
            dt_bias = art.load(art.layer_name(layer, "linear_attn.dt_bias"))
            nrm = art.load(art.layer_name(layer, "linear_attn.norm.weight"))
            fps["deltanet"].append((layer, fingerprint(w_qkvz)))
            if "deltanet" not in local_wx:
                y = gemm(w_qkvz, x_norm["base"])
                local_wx["deltanet"] = {
                    "site": f"layer {layer} in_proj_qkvz on post-input_ln hidden",
                    "y": y,
                    "y_scale": 0.01 * y,
                    "y_null": np.zeros_like(y),
                    "x": x_norm["base"],
                    "w_frob": float(np.linalg.norm(w_qkvz.astype(np.float64))),
                }
            mix = {}
            for k, xn in x_norm.items():
                mix[k] = deltanet_forward(
                    xn, w_qkvz, w_ba, w_out, conv_w, a_log, dt_bias, nrm
                )
            del w_qkvz, w_ba, w_out, conv_w

        for k, h in streams.items():
            add = mix[k]
            if k == "null_attention_gqa" and gqa:
                add = np.zeros_like(add)
            if k == "null_deltanet" and not gqa:
                add = np.zeros_like(add)
            streams[k] = h + add

        pn = art.load(art.layer_name(layer, "post_attention_layernorm.weight"))
        mlp_x = {k: rmsnorm_delta(h, pn) for k, h in streams.items()}
        mlp_in.append(mlp_x["base"].copy())
        wg = art.load(art.layer_name(layer, "mlp.gate_proj.weight"))
        wu = art.load(art.layer_name(layer, "mlp.up_proj.weight"))
        wd = art.load(art.layer_name(layer, "mlp.down_proj.weight"))
        fps["mlp"].append((layer, fingerprint(wd)))
        pre_base = silu(gemm(wg, mlp_x["base"])) * gemm(wu, mlp_x["base"])
        post_swiglu.append(pre_base.copy())
        if "mlp" not in local_wx:
            y = gemm(wd, pre_base)
            local_wx["mlp"] = {
                "site": f"layer {layer} down_proj on real post-SwiGLU",
                "y": y,
                "y_scale": 0.01 * y,
                "y_null": np.zeros_like(y),
                "x": pre_base,
                "w_frob": float(np.linalg.norm(wd.astype(np.float64))),
            }
        mlp_out = {"base": gemm(wd, pre_base)}
        for k, xn in mlp_x.items():
            if k == "base":
                continue
            mlp_out[k] = mlp_forward(xn, wg, wu, wd)
        del wg, wu, wd
        for k, h in streams.items():
            add = mlp_out[k]
            if k == "null_mlp":
                add = np.zeros_like(add)
            streams[k] = h + add
        if layer % 8 == 0 or layer == LAYERS - 1:
            progress(
                f"layer {layer:02d}/{LAYERS-1} {organ}  "
                f"h_rms={float(np.sqrt(np.mean(streams['base']**2))):.4f}  "
                f"{time.perf_counter()-t0:.1f}s"
            )

    # fingerprints for embed / output
    fps["embedding"].append(
        (0, fingerprint(h_embed))
    )  # activation fingerprint; weight fingerprint separately
    w_embed_fp = fingerprint(
        read_q4_rows(
            art.path("language_model.model.embed_tokens.weight"),
            art.by_name["language_model.model.embed_tokens.weight"]["shape"],
            np.unique(tokens)[:32],
        )
    )
    fps["embedding"] = [(0, w_embed_fp)]

    final_delta = art.load("language_model.model.norm.weight")
    h_final = {k: rmsnorm_delta(h, final_delta) for k, h in streams.items()}
    progress("lm_head dequant + GEMM (248320 x 5120)")
    w_head = art.load("language_model.lm_head.weight")
    fps["output"].append((0, fingerprint(w_head)))
    if "output" not in local_wx:
        y = gemm(w_head, h_final["base"])
        local_wx["output"] = {
            "site": "lm_head on final RMSNorm hidden",
            "y": y,
            "y_scale": 0.01 * y,
            "y_null": np.zeros_like(y),
            "x": h_final["base"],
            "w_frob": float(np.linalg.norm(w_head.astype(np.float64))),
        }
    if "embedding" not in local_wx:
        # organ output of embed IS the gathered rows
        y = h_embed
        local_wx["embedding"] = {
            "site": "embed_tokens gather of real token ids",
            "y": y,
            "y_scale": 0.01 * y,
            "y_null": np.zeros_like(y),
            "x": tokens.astype(np.float32)[:, None],  # ids, not a linear map
            "w_frob": None,
        }
    logits = {k: gemm(w_head, h) for k, h in h_final.items()}
    logits["null_output"] = np.zeros_like(logits["base"])
    del w_head
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_s": elapsed,
        "logits": logits,
        "h_embed": h_embed,
        "h_final_base": h_final["base"],
        "mixer_in_gqa": np.concatenate(mixer_in_gqa, axis=0) if mixer_in_gqa else None,
        "mixer_in_dn": np.concatenate(mixer_in_dn, axis=0) if mixer_in_dn else None,
        "mlp_in": np.concatenate(mlp_in, axis=0),
        "post_swiglu": np.concatenate(post_swiglu, axis=0) if post_swiglu else None,
        "fingerprints": fps,
        "local_wx": local_wx,
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def main() -> int:
    started = time.perf_counter()
    watched_fail: list[str] = []
    progress = lambda m: print(f"[progress] {m}", file=sys.stderr, flush=True)

    if not (ARTIFACT / "manifest.json").is_file():
        print("FATAL: gravity artifact missing", ARTIFACT)
        return 2
    art = Artifact(ARTIFACT)
    man = art.manifest

    # --- identity ---
    file_count = sum(1 for _ in ARTIFACT.rglob("*") if _.is_file())
    byte_count = 0
    for p in ARTIFACT.rglob("*"):
        if p.is_file():
            byte_count += p.stat().st_size
    n_tensors = int(man["tensor_count"])
    n_params = int(man["source_weight_elements"])
    bpw = float(man["complete_physical_bpw"])

    progress(f"artifact {file_count} files, {byte_count} bytes, {n_tensors} tensors")

    # --- tokenize ---
    token_ids: list[int] = []
    tok_source = None
    for p in PROMPTS:
        ids, src = tokenize(p)
        tok_source = src
        token_ids.extend(ids)
        if len(token_ids) >= MAX_TOKENS:
            break
    token_ids = token_ids[:MAX_TOKENS]
    if len(token_ids) < 8:
        raise RuntimeError(f"too few tokens: {len(token_ids)}")
    tokens = np.array(token_ids, dtype=np.int64)
    capture_source = {
        "kind": "cpu_hybrid_prefill_of_gravity_q4_on_real_token_ids",
        "artifact": str(ARTIFACT),
        "tokenizer": tok_source,
        "n_tokens": int(tokens.size),
        "token_ids_head": token_ids[:16],
        "prompts": list(PROMPTS),
        "math": (
            "Qwen3_5 hybrid: Gated-DeltaNet (fused QKVZ/BA, causal conv+silu, "
            "L2 Q/K, recurrent gated delta, gated RMSNorm, out_proj) and GQA "
            "(q_proj+output_gate, qk RMSNorm as 1+δ, rotate_half RoPE θ=1e7 "
            "on first 64 of 256, sigmoid gate, o_proj) then SwiGLU MLP. "
            "Norms packed as hf_gemma_delta. Weights are the gravity HQ30UQ4/"
            "f32v2 artifact, dequantized per tensor on CPU. No GPU, no second 27B."
        ),
        "not_synthetic": True,
        "gaussian_proxy_used": False,
        "resident_llama_server": "127.0.0.1:52484 used as tokenizer only",
    }
    progress(f"tokens={tokens.size} via {tok_source}")

    # --- physical cost ---
    cost = physical_cost(art)
    ledger = load_token_ns()
    ns_share = organ_token_ns(ledger)

    # --- scale-rejection on REAL embed rows (before the long forward) ---
    progress("scale-rejection probe on real embed rows")
    y_embed = read_q4_rows(
        art.path("language_model.model.embed_tokens.weight"),
        art.by_name["language_model.model.embed_tokens.weight"]["shape"],
        tokens,
    )
    scale_probe = {
        "site": "embed_tokens[real token ids]  — real activations, not gaussian",
        "identity": fidelity(y_embed, y_embed),
        "scaled_0p01": fidelity(y_embed, 0.01 * y_embed),
        "zero": fidelity(y_embed, np.zeros_like(y_embed)),
    }
    sp = scale_probe["scaled_0p01"]
    if sp["cosine"] > 0.999 and sp["scale_aware"] > 0.5:
        watched_fail.append(
            "scale-aware metric FAILED to reject 0.01*embed — this is the cosine-blindness bug"
        )
    else:
        watched_fail.append(
            f"cosine(embed, 0.01*embed)={sp['cosine']:.6f} looks perfect; "
            f"scale_aware={sp['scale_aware']:.4f} and relative_l2={sp['relative_l2']:.4f} "
            f"reject it. This is the failure mode the metric exists to catch."
        )

    # codec spot check
    progress("codec spot-check vs bf16 source (strided)")
    codec_check = spot_check_q4_against_bf16(
        art, "language_model.model.layers.0.mlp.gate_proj.weight"
    )
    if codec_check.get("status") == "MEASURED" and codec_check["relative_l2"] > 0.3:
        watched_fail.append(
            f"q4 vs bf16 gate_proj L0 relative_l2={codec_check['relative_l2']:.3f} "
            f"(cosine={codec_check['cosine']:.3f}) — dequant may be wrong; ranking still uses "
            "the packed artifact as the vehicle under test"
        )

    # --- forward ---
    progress("CPU hybrid prefill with 5 null streams + base")
    try:
        fwd = run_forward(art, tokens, progress)
        forward_status = "MEASURED"
        forward_reason = None
    except Exception as e:
        watched_fail.append(f"CPU hybrid prefill raised {type(e).__name__}: {e}")
        fwd = None
        forward_status = "NOT_MEASURED"
        forward_reason = f"{type(e).__name__}: {e}"

    organs_out: dict[str, dict] = {}
    ranking_rows = []

    if fwd is None:
        for o in ORGANS:
            organs_out[o] = {
                "status": "NOT_MEASURED",
                "reason": forward_reason,
                "physical": cost[o],
                "token_ns": ns_share.get(o, {"status": "NOT_MEASURED"}),
            }
    else:
        logits = fwd["logits"]
        base_logits = logits["base"]
        null_key = {
            "embedding": "null_embedding",
            "attention_gqa": "null_attention_gqa",
            "deltanet": "null_deltanet",
            "mlp": "null_mlp",
            "output": "null_output",
        }
        act_by_organ = {
            "embedding": fwd["h_embed"],
            "attention_gqa": fwd["mixer_in_gqa"],
            "deltanet": fwd["mixer_in_dn"],
            "mlp": fwd["mlp_in"],
            "output": fwd["h_final_base"],
        }
        act_site = {
            "embedding": "embed_tokens[token_ids]  (T, 5120)",
            "attention_gqa": "post-input_ln hidden at every GQA layer, concatenated (16T, 5120)",
            "deltanet": "post-input_ln hidden at every DeltaNet layer, concatenated (48T, 5120)",
            "mlp": "post-attention_ln hidden concatenated over 64 layers (64T, 5120); "
            "down_proj Wx is scored on real post-SwiGLU at layer 0",
            "output": "final RMSNorm hidden (T, 5120) — lm_head input",
        }

        for o in ORGANS:
            wx = fwd["local_wx"].get(o)
            local = None
            if wx is not None:
                m_id = fidelity(wx["y"], wx["y"])
                m_sc = fidelity(wx["y"], wx["y_scale"])
                m_z = fidelity(wx["y"], wx["y_null"])
                local = {
                    "site": wx["site"],
                    "identity": m_id,
                    "scaled_0p01_W": m_sc,
                    "null_W": m_z,
                    "scaled_rejected": bool(
                        m_sc["scale_aware"] < 0.5 and m_sc["relative_l2"] > 0.5
                    ),
                }
            acts = act_by_organ[o]
            if acts is None:
                rank = {
                    "status": "NOT_MEASURED",
                    "reason": f"no activations captured for {o}",
                }
                spars = rank
            else:
                rank = participation_ratio(acts)
                spars = residual_sparsity(acts)
            shared = pairwise_scale_aware(fwd["fingerprints"].get(o, []))
            if o == "embedding":
                shared = {
                    "status": "NOT_MEASURED",
                    "reason": "embedding is a single table, not a per-layer family; "
                    "cross-layer alignment is undefined",
                }
            if o == "output":
                # lm_head is a single table; final norm is a vector. Cross-layer N/A.
                shared = {
                    "status": "NOT_MEASURED",
                    "reason": "output is lm_head + final norm, not a per-layer family",
                }

            null_surv = survival_from_logits(base_logits, logits[null_key[o]])
            function_lost = 1.0 - null_surv["survival"]
            b = cost[o]["bytes"]
            active_b = cost[o]["active_bytes_per_token"] or b
            per_stored = function_lost / b if b else None
            per_active = function_lost / active_b if active_b else None
            organs_out[o] = {
                "status": "MEASURED",
                "capture_site": act_site[o],
                "physical": cost[o],
                "token_ns": ns_share.get(o),
                "functional_sensitivity_local": local,
                "null_representation": {
                    "probe": "zero this organ's residual (embed: zero rows; output: zero logits; "
                    "mixer/MLP: drop residual add)",
                    "consequence": null_surv,
                    "function_lost": function_lost,
                    "reading": (
                        "near-zero function_lost means the organ can disappear with little "
                        "output change on this prompt set — a bigger finding than 'compresses well'"
                        if function_lost < 0.15
                        else "zeroing this organ moves the logits; it carries function on this capture"
                    ),
                },
                "shared_structure_across_layers": shared,
                "activation_rank": rank,
                "residual_sparsity_after_cheapest_fit": spars,
            }
            ranking_rows.append(
                {
                    "organ": o,
                    "function_lost": function_lost,
                    "bytes": b,
                    "active_bytes_per_token": active_b,
                    "function_lost_per_stored_byte": per_stored,
                    "function_lost_per_active_byte": per_active,
                    "survival_null": null_surv["survival"],
                    "relative_l2_null": null_surv["relative_l2"],
                    "kl_last_token": null_surv["kl_last_token"],
                    "argmax_match_null": null_surv["argmax_match"],
                    "arithmetic_stored": (
                        f"(1 - survival_null) / bytes = (1 - {null_surv['survival']:.6f}) / {b}"
                    ),
                    "arithmetic_active": (
                        f"(1 - survival_null) / active_bytes_per_token = "
                        f"(1 - {null_surv['survival']:.6f}) / {active_b}"
                    ),
                }
            )

        ranking_rows.sort(
            key=lambda r: (r["function_lost_per_stored_byte"] or -1.0), reverse=True
        )
        for i, r in enumerate(ranking_rows, 1):
            r["rank_by_function_per_stored_byte"] = i
        by_active = sorted(
            ranking_rows,
            key=lambda r: (r["function_lost_per_active_byte"] or -1.0),
            reverse=True,
        )
        for i, r in enumerate(by_active, 1):
            r["rank_by_function_per_active_byte"] = i

        watched_fail.append(
            "v1 activation capture "
            "(workspace/campaign/records/runs/qwen38-27b/activation-capture-v1) "
            "is not on disk. Recaptured from the packed artifact rather than replaying numbers."
        )
        watched_fail.append(
            "native gravity greedy decode (QWEN38_GRAVITY_NATIVE) collapsed to token 150910 "
            "with replacement characters. That run is not used as functional ground truth."
        )
        final_rms = float(np.sqrt(np.mean(fwd["h_final_base"] ** 2)))
        watched_fail.append(
            f"CPU residual-stream RMS at the last layer was {final_rms:.3f} "
            "(started ~0.21 after embed+L0). Native Metal decode is the bit-identity "
            "oracle and is currently collapsed (QWEN38_GRAVITY_NATIVE); this CPU path "
            "is internally consistent for organ ablations but is not claimed bit-identical "
            "to the GPU kernels."
        )
        watched_fail.append(
            "torch MPS is unavailable in this process; census ran on CPU. "
            "A second 27B was not loaded. llama-server :52484 stayed resident."
        )

    # baseline note
    watched_fail.append(
        "protected HCLI suite on hawking-copy with HCLI_SWAP_CEILING_GIB=64: "
        "463 passed, 2 skipped (contract named 464 passed, 1 skipped). "
        "This worktree is a sparse checkout; tools/haider is not materialized here "
        "and git sparse-checkout add is forbidden, so the suite cannot be reproduced "
        "inside this lane. Headless hcli_* tests ImportError without hcli."
    )

    receipt = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "git_head": git_head(),
        "artifact": {
            "root": str(ARTIFACT),
            "files": file_count,
            "bytes": byte_count,
            "tensor_count": n_tensors,
            "parameter_count": n_params,
            "complete_physical_bpw": bpw,
            "codec_families": {
                "grouped_absmax_q4_group64": int(man["q4_tensors"]),
                "raw_f32": int(man["f32_tensors"]),
            },
            "manifest_schema": man.get("schema"),
            "anchors_not_rederived": {
                "measured_tps": 32.73,
                "measured_ms_per_token": 30.606,
                "machine": "Apple M3 Ultra, 60 GPU cores, 103079215104 B, Metal 4",
                "measured_roof_GB_s": 595.9,
                "kernel_binding": "38 dispatched against 554 declared (G104 NX genome)",
                "decode_source": "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
                "two_servers_tps": 3.986,
                "one_server_tps": 33.47,
            },
        },
        "capture": capture_source,
        "forward_status": forward_status,
        "forward_reason": forward_reason,
        "forward_elapsed_s": None if fwd is None else fwd["elapsed_s"],
        "scale_rejection": {
            "statement": "0.01*W must not score perfect. Cosine is scale-invariant; "
            "scale_aware = cosine * min(s,1/s) and relative_l2 are the gates.",
            "probe": scale_probe,
            "passes": bool(
                scale_probe["scaled_0p01"]["scale_aware"] < 0.5
                and scale_probe["scaled_0p01"]["relative_l2"] > 0.5
                and scale_probe["identity"]["scale_aware"] > 0.99
            ),
        },
        "codec_spot_check_vs_bf16": codec_check,
        "prior_science_respected": {
            "never_synthetic_activations": True,
            "cosine_is_scale_blind": True,
            "raw_activation_cosine_null_around_0p898": (
                "not re-derived; cited. Local skill-vs-mean-null is the comparable statistic here."
            ),
            "complete_bpw_does_not_predict_coherence": True,
            "attention_set_the_bpw_floor": True,
            "mlp_3p25_first_coherent_low_bpw": True,
        },
        "organs": organs_out,
        "ranking": {
            "definition": (
                "marginal functional survival per marginal physical byte, estimated from the "
                "NULL representation: function_lost = 1 - survival(logits_base, logits_organ_zeroed); "
                "preciousness = function_lost / bytes. Higher = an allocator should keep these bytes. "
                "An organ that can be zeroed with little loss ranks last even if it 'compresses well'."
            ),
            "by_stored_byte": ranking_rows,
            "by_active_byte_per_token": sorted(
                ranking_rows,
                key=lambda r: r.get("rank_by_function_per_active_byte") or 99,
            )
            if ranking_rows
            else [],
        },
        "token_ns_ledger": {
            "status": ledger.get("status"),
            "source": ledger.get("source"),
            "total_token_ns": ledger.get("total_token_ns"),
        },
        "what_i_watched_fail": watched_fail,
        "wall_s": time.perf_counter() - started,
    }

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=False, default=str) + "\n")

    # -------- stdout report --------
    print("=" * 78)
    print("NOETIC ORGAN CENSUS")
    print("=" * 78)
    print(f"schema: {SCHEMA}")
    print(f"artifact: {ARTIFACT}")
    print(
        f"identity: {file_count} files, {byte_count} bytes, {n_tensors} tensors, "
        f"{n_params} params, {bpw:.6f} BPW"
    )
    print(
        f"codec: grouped_absmax 4-bit/group-64 × {man['q4_tensors']}, "
        f"raw_f32 × {man['f32_tensors']}"
    )
    print()
    print("CAPTURE SOURCE")
    print(f"  {capture_source['kind']}")
    print(f"  tokenizer: {capture_source['tokenizer']}")
    print(f"  n_tokens: {capture_source['n_tokens']}  ids[:8]={capture_source['token_ids_head'][:8]}")
    print(f"  math: {capture_source['math']}")
    print(f"  gaussian_proxy_used: {capture_source['gaussian_proxy_used']}")
    print()
    print("SCALE REJECTION  (0.01*W must not score perfect)")
    print(
        f"  identity  cosine={scale_probe['identity']['cosine']:.6f}  "
        f"scale_aware={scale_probe['identity']['scale_aware']:.6f}  "
        f"rel_l2={scale_probe['identity']['relative_l2']:.6f}"
    )
    print(
        f"  0.01*W    cosine={sp['cosine']:.6f}  "
        f"scale_aware={sp['scale_aware']:.6f}  "
        f"rel_l2={sp['relative_l2']:.6f}  "
        f"skill={sp['skill_vs_mean_null']:.6f}"
    )
    print(
        f"  zero      cosine={scale_probe['zero']['cosine']:.6f}  "
        f"rel_l2={scale_probe['zero']['relative_l2']:.6f}"
    )
    print(f"  gate passes: {receipt['scale_rejection']['passes']}")
    print()
    print("CODEC SPOT-CHECK vs BF16 source (strided, not the ranking)")
    if codec_check.get("status") == "MEASURED":
        print(
            f"  {codec_check['name']}: cosine={codec_check['cosine']:.6f}  "
            f"rel_l2={codec_check['relative_l2']:.6f}  scale_aware={codec_check['scale_aware']:.6f}"
        )
    else:
        print(f"  NOT_MEASURED: {codec_check.get('reason')}")
    print()
    print(f"FORWARD: {forward_status}" + (f" ({forward_reason})" if forward_reason else ""))
    if fwd is not None:
        print(f"  elapsed {fwd['elapsed_s']:.1f}s")
    print()

    for o in ORGANS:
        rec = organs_out[o]
        print("-" * 78)
        print(f"ORGAN  {o}")
        print(f"  status: {rec.get('status')}")
        if rec.get("status") != "MEASURED":
            print(f"  reason: {rec.get('reason')}")
            print(f"  physical bytes: {cost[o]['bytes']}")
            continue
        ph = rec["physical"]
        print(
            f"  physical: {ph['bytes']} bytes  {ph['elements']} elements  "
            f"{ph['tensor_count']} tensors (q4={ph['q4_tensors']} f32={ph['f32_tensors']})"
        )
        print(
            f"  active_bytes_per_token: {ph['active_bytes_per_token']}  "
            f"({ph['active_bytes_note']})"
        )
        tn = rec.get("token_ns") or {}
        if tn.get("status") == "MEASURED":
            print(
                f"  token_ns: {tn['ns_per_token']:.1f} ns  "
                f"share={tn['share_of_token_ns']*100:.2f}%  [{tn['arithmetic']}]"
            )
        else:
            print(f"  token_ns: NOT_MEASURED ({tn.get('reason')})")
        print(f"  capture_site: {rec['capture_site']}")
        loc = rec.get("functional_sensitivity_local")
        if loc:
            sc = loc["scaled_0p01_W"]
            nz = loc["null_W"]
            print(f"  local Wx site: {loc['site']}")
            print(
                f"    0.01*W: cosine={sc['cosine']:.6f} scale_aware={sc['scale_aware']:.6f} "
                f"rel_l2={sc['relative_l2']:.6f} rejected={loc['scaled_rejected']}"
            )
            print(
                f"    NULL W: cosine={nz['cosine']:.6f} scale_aware={nz['scale_aware']:.6f} "
                f"rel_l2={nz['relative_l2']:.6f} skill={nz['skill_vs_mean_null']:.6f}"
            )
        nr = rec["null_representation"]
        csq = nr["consequence"]
        print("  NULL REPRESENTATION (organ disappears)")
        print(f"    probe: {nr['probe']}")
        print(
            f"    survival={csq['survival']:.6f}  rel_l2={csq['relative_l2']:.6f}  "
            f"kl_last={csq['kl_last_token']:.4f}  tv_last={csq['tv_last_token']:.4f}  "
            f"argmax_match={csq['argmax_match']:.3f}  cosine={csq['cosine']:.6f}  "
            f"scale_aware={csq['scale_aware']:.6f}"
        )
        print(f"    function_lost={nr['function_lost']:.6f}  {nr['reading']}")
        sh = rec["shared_structure_across_layers"]
        if sh.get("status") == "MEASURED":
            print(
                f"  shared structure: mean scale_aware={sh['mean_pairwise_scale_aware']:.4f}  "
                f"min={sh['min_pairwise_scale_aware']:.4f}  mean_cosine={sh['mean_pairwise_cosine']:.4f}  "
                f"frob_cv={sh['frobenius_cv_across_layers']:.4f}  pairs={sh['n_pairs']}"
            )
        else:
            print(f"  shared structure: NOT_MEASURED ({sh.get('reason')})")
        rk = rec["activation_rank"]
        if rk.get("status") == "MEASURED":
            amb = rk.get("ambient_rank_99", rk.get("rank_99_of_captured_spectrum"))
            print(
                f"  activation rank: n={rk['n_rows']}×{rk['n_cols']}  "
                f"PR={rk['participation_ratio']:.2f}  rank99_captured={rk['rank_99_of_captured_spectrum']}  "
                f"ambient_rank99={amb}"
            )
            if rk.get("ambient_rank_99_reason"):
                print(f"    {rk['ambient_rank_99_reason']}")
        else:
            print(f"  activation rank: NOT_MEASURED ({rk.get('reason')})")
        spz = rec["residual_sparsity_after_cheapest_fit"]
        if spz.get("status") == "MEASURED":
            print(
                f"  residual sparsity after {spz['fit']}: "
                f"rank1_energy={spz['rank1_energy_frac']:.4f}  "
                f"resid_rms/x={spz['residual_rms_over_centered_rms']:.4f}  "
                f"frac_|r|<0.01rms={spz['residual_sparsity_at_0p01_rms']:.4f}"
            )
        else:
            print(f"  residual sparsity: NOT_MEASURED ({spz.get('reason')})")

    print()
    print("=" * 78)
    print("RANKING  — marginal functional survival per marginal physical byte")
    print("  function_lost = 1 - survival(logits_base, logits_organ_zeroed)")
    print("  preciousness  = function_lost / bytes")
    print("  higher = these bytes buy more function; an allocator keeps them first")
    print("=" * 78)
    if not ranking_rows:
        print("  NOT_MEASURED: forward did not complete")
    else:
        print("  by stored byte:")
        for r in ranking_rows:
            print(
                f"    #{r['rank_by_function_per_stored_byte']}  {r['organ']:<16}  "
                f"lost={r['function_lost']:.6f}  bytes={r['bytes']}  "
                f"lost/byte={r['function_lost_per_stored_byte']:.4e}  "
                f"  {r['arithmetic_stored']}"
            )
        print("  by active byte per token:")
        for r in sorted(ranking_rows, key=lambda x: x["rank_by_function_per_active_byte"]):
            print(
                f"    #{r['rank_by_function_per_active_byte']}  {r['organ']:<16}  "
                f"lost={r['function_lost']:.6f}  active={r['active_bytes_per_token']}  "
                f"lost/active={r['function_lost_per_active_byte']:.4e}  "
                f"  {r['arithmetic_active']}"
            )

    print()
    print("## WHAT I WATCHED FAIL")
    for line in watched_fail:
        print(f"  - {line}")
    print()
    print(f"receipt: {RECEIPT}")
    print(f"wall_s: {receipt['wall_s']:.1f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)

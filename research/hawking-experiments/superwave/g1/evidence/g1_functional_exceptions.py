#!/usr/bin/env python3
"""Functional-exception topology search on real Qwen3.8 BF16 + 256-token capture.

CPU / numpy only. No GPU. No generate. No pack. No live-organism contact.

W ≈ Q_low + E_sparse. Exceptions ranked by activation-conditioned functional
consequence (fit-split), not |W|. Downstream amplification multiplies the
score when units are compared across layers.

Writes /tmp/g1_functional_exceptions.json and a machine-readable log.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import resource
import struct
import time
from pathlib import Path

import numpy as np

SRC = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16")
CAP = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1")
OUT = Path("/tmp/g1_functional_exceptions.json")

HIDDEN = 5120
INTERMEDIATE = 17408
N_LAYERS = 64
N_TOKENS = 256
VOCAB = 248320
N_PARAMS = 26_895_998_464

KEY_HEADS = 16
VALUES_PER_KEY = 3
KEY_DIM = 128
VALUE_DIM = 128
LINEAR_VALUE_HEADS = 48
GQA_HEADS = 24
GQA_KV = 4
GQA_HEAD_DIM = 256

# even-fit / odd-hold matches forensics + mse-scale-rule
FIT = np.arange(0, N_TOKENS, 2)
HOLD = np.arange(1, N_TOKENS, 2)

# cited from residual_walk, re-verified in this run's hidden census
CITED_L63_MEAN_TOKEN_YD_OVER_X = 2.6039602756500244

LAYERS = (0, 3, 6, 15, 32, 47, 63)

# inventory mass (language-only G0 catalog) — g1-heterogeneous-allocation.md
MASS = {
    "gate": 5_704_253_440,
    "up": 5_704_253_440,
    "down": 5_704_253_440,
    "dn_qkvz": 4_026_531_840,
    "dn_out": 1_509_949_440,
    "embed": 1_271_398_400,
    "lm_head": 1_271_398_400,
    "gqa_q": 1_006_632_960,
    "gqa_o": 503_316_480,
    "gqa_k": 83_886_080,
    "gqa_v": 83_886_080,
    "dn_ba": 23_592_960,
    "small_f32": 2_646_144,
}

VALUE_BITS = 16.0  # bf16 exception payload


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_gb():.3f}G {msg}", flush=True)


def sha256_file(path: Path, nbytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        if nbytes is None:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(fh.read(nbytes))
    return h.hexdigest()


def mean_row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
        b = b[None, :]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 1e-12
    if not np.any(ok):
        return 0.0
    return float(np.mean(num[ok] / den[ok]))


def min_row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
        b = b[None, :]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 1e-12
    if not np.any(ok):
        return 0.0
    return float(np.min(num[ok] / den[ok]))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(a))
    return num / den if den > 1e-12 else num


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


# ---------------------------------------------------------------------------
# safetensors BF16
# ---------------------------------------------------------------------------

_HEADER_CACHE: dict[Path, dict] = {}
_WMAP = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]


def read_header(shard: Path) -> dict:
    if shard not in _HEADER_CACHE:
        with shard.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            _HEADER_CACHE[shard] = json.loads(fh.read(n))
    return _HEADER_CACHE[shard]


def load_tensor(name: str) -> np.ndarray:
    shard = SRC / _WMAP[name]
    header = read_header(shard)
    info = header[name]
    dtype = info.get("dtype", "BF16")
    shape = tuple(int(x) for x in info["shape"])
    lo, hi = info["data_offsets"]
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        fh.seek(8 + n + lo)
        raw = fh.read(hi - lo)
    if dtype in ("BF16", "BFLOAT16"):
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return np.ascontiguousarray(u32.view(np.float32).reshape(shape))
    if dtype in ("F32", "FLOAT32"):
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    raise RuntimeError(f"unsupported dtype {dtype} for {name}")


def load_hidden(layer: int) -> np.ndarray:
    path = CAP / "hidden" / f"L{layer:02d}.f32"
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != N_TOKENS * HIDDEN:
        raise RuntimeError(f"hidden L{layer} size {raw.size}")
    return np.ascontiguousarray(raw.reshape(N_TOKENS, HIDDEN))


def tname(layer: int, suffix: str) -> str:
    return f"language_model.model.layers.{layer}.{suffix}"


def is_gqa(layer: int) -> bool:
    return (layer + 1) % 4 == 0


# ---------------------------------------------------------------------------
# codecs (production family)
# ---------------------------------------------------------------------------

def group_pad(W: np.ndarray, group_size: int) -> tuple[np.ndarray, int]:
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    n = int(flat.size)
    groups = (n + group_size - 1) // group_size
    padded = np.zeros((groups, group_size), dtype=np.float32)
    padded.reshape(-1)[:n] = flat
    return padded, n


def uniform_absmax_recon(W: np.ndarray, bits: int, group_size: int = 64) -> np.ndarray:
    """HQ30UQ4 family: flat C-order groups, f16(absmax/bound), RTN."""
    bound = (1 << (bits - 1)) - 1
    if bound <= 0:
        raise ValueError(f"uniform bits={bits} is degenerate")
    padded, n = group_pad(W, group_size)
    absmax = np.max(np.abs(padded), axis=1)
    scale = (absmax / float(bound)).astype(np.float16).astype(np.float32)
    den = np.where(scale > 0.0, scale, 1.0)
    qmin = -(1 << (bits - 1))
    qmax = bound
    codes = np.rint(padded / den[:, None]).clip(qmin, qmax)
    recon = (codes * scale[:, None]).reshape(-1)[:n]
    return recon.reshape(W.shape).astype(np.float32)


def binary_meanabs_recon(W: np.ndarray, group_size: int = 128) -> np.ndarray:
    """HGRAVB01: sign × group mean-abs."""
    padded, n = group_pad(W, group_size)
    mean = np.mean(np.abs(padded), axis=1).astype(np.float32)
    recon = (np.sign(padded) * mean[:, None]).reshape(-1)[:n]
    return recon.reshape(W.shape).astype(np.float32)


def body_bpw(kind: str) -> float:
    return {
        "none": 0.0,
        "binary_g128": 1.0 + 16.0 / 128.0,
        "q2_g64": 2.0 + 16.0 / 64.0,
        "q3_g64": 3.0 + 16.0 / 64.0,
        "q4_g64": 4.0 + 16.0 / 64.0,
        "q2_g128": 2.0 + 16.0 / 128.0,
        "q3_g128": 3.0 + 16.0 / 128.0,
    }[kind]


def quantize(W: np.ndarray, kind: str) -> np.ndarray:
    if kind == "none":
        return np.zeros_like(W, dtype=np.float32)
    if kind == "binary_g128":
        return binary_meanabs_recon(W, 128)
    if kind == "q2_g64":
        return uniform_absmax_recon(W, 2, 64)
    if kind == "q3_g64":
        return uniform_absmax_recon(W, 3, 64)
    if kind == "q4_g64":
        return uniform_absmax_recon(W, 4, 64)
    if kind == "q2_g128":
        return uniform_absmax_recon(W, 2, 128)
    if kind == "q3_g128":
        return uniform_absmax_recon(W, 3, 128)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# index encodings
# ---------------------------------------------------------------------------

def elias_gamma_bits(v: np.ndarray) -> int:
    v = np.maximum(v.astype(np.int64, copy=False), 1)
    return int((2 * np.floor(np.log2(v)).astype(np.int64) + 1).sum())


def rice_bits(v: np.ndarray, k: int) -> int:
    v = np.maximum(v.astype(np.int64, copy=False), 0)
    q = v >> k
    return int(q.sum() + v.size * (1 + k))


def best_rice_bits(v: np.ndarray) -> tuple[int, int]:
    if v.size == 0:
        return 0, 0
    mean = float(v.mean()) if v.size else 1.0
    k0 = max(0, int(round(math.log2(max(mean, 1.0)))))
    best = (10**18, 0)
    for k in range(max(0, k0 - 3), k0 + 6):
        b = rice_bits(v, k)
        if b < best[0]:
            best = (b, k)
    return best


def index_from_positions(pos: np.ndarray, n: int) -> dict:
    """Exact index encodings for a set of linearized positions in [0, n)."""
    pos = np.unique(np.asarray(pos, dtype=np.int64))
    k = int(pos.size)
    if k == 0:
        return {
            "k": 0,
            "dense_bitmap_bits": n,
            "rice_bits": 0,
            "elias_gamma_bits": 0,
            "fixed_log2n_bits": 0,
            "rice_k": 0,
            "log2n": int(math.ceil(math.log2(max(n, 2)))),
            "cheapest_bits": 0,
            "cheapest": "none",
        }
    deltas = np.empty(k, dtype=np.int64)
    deltas[0] = int(pos[0]) + 1
    if k > 1:
        deltas[1:] = np.diff(pos)
    gamma = elias_gamma_bits(deltas)
    rice_b, rice_k = best_rice_bits(deltas)
    log2n = int(math.ceil(math.log2(max(n, 2))))
    fixed = k * log2n
    # occupied-group bitmap G=64
    G = 64
    n_groups = (n + G - 1) // G
    occ = np.zeros(n_groups, dtype=bool)
    occ[pos // G] = True
    n_occ = int(occ.sum())
    occ_bits = n_groups + n_occ * G
    candidates = {
        "dense_bitmap": n,
        "rice": rice_b,
        "elias_gamma": gamma,
        "fixed_log2n": fixed,
        "occupied_bitmap_g64": occ_bits,
    }
    cheapest = min(candidates, key=candidates.get)
    return {
        "k": k,
        "dense_bitmap_bits": n,
        "rice_bits": rice_b,
        "elias_gamma_bits": gamma,
        "fixed_log2n_bits": fixed,
        "occupied_bitmap_g64_bits": occ_bits,
        "occupied_groups_g64": n_occ,
        "rice_k": rice_k,
        "log2n": log2n,
        "cheapest_bits": int(candidates[cheapest]),
        "cheapest": cheapest,
    }


def structured_index_bits(k: int, n_units: int) -> int:
    if k == 0 or n_units <= 1:
        return 0
    return int(k * math.ceil(math.log2(n_units)))


# ---------------------------------------------------------------------------
# X construction
# ---------------------------------------------------------------------------

def fuse_qkvz(qkv: np.ndarray, z: np.ndarray) -> np.ndarray:
    qkv = np.ascontiguousarray(qkv, dtype=np.float32)
    z = np.ascontiguousarray(z, dtype=np.float32)
    value_rows = VALUES_PER_KEY * VALUE_DIM
    qkvz_per_key = KEY_DIM * 2 + value_rows * 2
    fused = np.empty((KEY_HEADS * qkvz_per_key, HIDDEN), dtype=np.float32)
    for kh in range(KEY_HEADS):
        dst = kh * qkvz_per_key
        q_src = kh * KEY_DIM
        k_src = KEY_HEADS * KEY_DIM + kh * KEY_DIM
        v_src = KEY_HEADS * KEY_DIM * 2 + kh * value_rows
        z_src = kh * value_rows
        fused[dst : dst + KEY_DIM] = qkv[q_src : q_src + KEY_DIM]
        fused[dst + KEY_DIM : dst + 2 * KEY_DIM] = qkv[k_src : k_src + KEY_DIM]
        fused[dst + 2 * KEY_DIM : dst + 2 * KEY_DIM + value_rows] = qkv[v_src : v_src + value_rows]
        fused[dst + 2 * KEY_DIM + value_rows : dst + qkvz_per_key] = z[z_src : z_src + value_rows]
    return fused


def deltanet_out_proxy(X: np.ndarray, W_qkvz: np.ndarray) -> np.ndarray:
    y = X @ W_qkvz.T
    value_rows = VALUES_PER_KEY * VALUE_DIM
    per_key = KEY_DIM * 2 + value_rows * 2
    y3 = y.reshape(X.shape[0], KEY_HEADS, per_key)
    v = y3[:, :, KEY_DIM * 2 : KEY_DIM * 2 + value_rows].reshape(X.shape[0], -1)
    z = y3[:, :, KEY_DIM * 2 + value_rows :].reshape(X.shape[0], -1)
    return np.ascontiguousarray(v * silu(z), dtype=np.float32)


def gqa_out_proxy(X: np.ndarray, W_q: np.ndarray, W_v: np.ndarray) -> np.ndarray:
    qg = X @ W_q.T
    v = X @ W_v.T
    qg = qg.reshape(X.shape[0], GQA_HEADS, 2, GQA_HEAD_DIM)
    gate = sigmoid(qg[:, :, 1, :])
    v = v.reshape(X.shape[0], GQA_KV, GQA_HEAD_DIM)
    v_rep = np.repeat(v, GQA_HEADS // GQA_KV, axis=1)
    return np.ascontiguousarray(
        (v_rep * gate).reshape(X.shape[0], GQA_HEADS * GQA_HEAD_DIM), dtype=np.float32
    )


# ---------------------------------------------------------------------------
# amplification census
# ---------------------------------------------------------------------------

def amplification_census() -> dict:
    log("amplification census: load 64 hidden files")
    norms = np.zeros((N_LAYERS, N_TOKENS), dtype=np.float64)
    rms = np.zeros(N_LAYERS, dtype=np.float64)
    col_rms = np.zeros((N_LAYERS, HIDDEN), dtype=np.float64)
    for L in range(N_LAYERS):
        h = load_hidden(L).astype(np.float64, copy=False)
        norms[L] = np.linalg.norm(h, axis=1)
        rms[L] = float(np.sqrt(np.mean(h * h)))
        col_rms[L] = np.sqrt(np.mean(h * h, axis=0))
        del h
    # local A_l = mean_t ||h_{l+1}|| / ||h_l||
    A_token = np.zeros(N_LAYERS - 1, dtype=np.float64)
    A_rms = np.zeros(N_LAYERS - 1, dtype=np.float64)
    for L in range(N_LAYERS - 1):
        den = np.maximum(norms[L], 1e-20)
        A_token[L] = float(np.mean(norms[L + 1] / den))
        A_rms[L] = float(rms[L + 1] / max(rms[L], 1e-20))
    # remaining gain to L63
    rem_token = np.zeros(N_LAYERS, dtype=np.float64)
    rem_rms = np.zeros(N_LAYERS, dtype=np.float64)
    for L in range(N_LAYERS):
        den = np.maximum(norms[L], 1e-20)
        rem_token[L] = float(np.mean(norms[63] / den))
        rem_rms[L] = float(rms[63] / max(rms[L], 1e-20))
    # product of subsequent A_rms from L to 62
    rem_prod = np.ones(N_LAYERS, dtype=np.float64)
    acc = 1.0
    for L in range(N_LAYERS - 2, -1, -1):
        acc *= float(A_rms[L])
        rem_prod[L] = acc
    rem_prod[63] = 1.0

    # residual-channel persistence (reconfirm 3994/3456/310)
    med = np.median(col_rms, axis=1, keepdims=True)
    hot10 = col_rms >= (10.0 * np.maximum(med, 1e-20))
    n_hot10 = hot10.sum(axis=0)
    mean_rms = col_rms.mean(axis=0)
    rank = np.argsort(-mean_rms)

    cited = json.loads((CAP / "capture-result.json").read_text())
    rec = {
        "n_layers": N_LAYERS,
        "n_tokens": N_TOKENS,
        "hidden": HIDDEN,
        "capture_sha256_self": cited.get("sha256_self"),
        "capture_status": cited.get("status"),
        "rms": [float(x) for x in rms],
        "A_token_mean": [float(x) for x in A_token],
        "A_rms": [float(x) for x in A_rms],
        "remaining_token_mean_to_L63": [float(x) for x in rem_token],
        "remaining_rms_to_L63": [float(x) for x in rem_rms],
        "remaining_prod_A_rms": [float(x) for x in rem_prod],
        "L0_to_L63_rms_ratio": float(rms[63] / max(rms[0], 1e-20)),
        "L62_to_L63_rms_ratio": float(rms[63] / max(rms[62], 1e-20)),
        "L62_to_L63_token_mean_ratio": float(A_token[62]),
        "mean_A_token": float(A_token.mean()),
        "mean_A_rms": float(A_rms.mean()),
        "n_layers_A_rms_gt_1": int((A_rms > 1.0).sum()),
        "n_layers_A_token_gt_1": int((A_token > 1.0).sum()),
        "act_rank_by_mean_rms_top8": [int(x) for x in rank[:8]],
        "ch3994_n_hot10": int(n_hot10[3994]),
        "ch3456_n_hot10": int(n_hot10[3456]),
        "ch310_n_hot10": int(n_hot10[310]),
        "persist_hot10_ge4": int((n_hot10 >= 4).sum()),
        "cited_L63_mean_token_yd_over_x": CITED_L63_MEAN_TOKEN_YD_OVER_X,
        "cited_source": "/tmp/g1_screen_vs_generate.json residual_walk.summary.max_yd_over_x AND per_layer[63].mean_token_yd_over_x",
    }
    log(
        f"amp L0_rms={rms[0]:.4f} L63_rms={rms[63]:.4f} "
        f"L0→L63={rec['L0_to_L63_rms_ratio']:.4f} "
        f"A_rms>1 on {rec['n_layers_A_rms_gt_1']}/63 steps "
        f"hot10>=4 n={rec['persist_hot10_ge4']} rank[:3]={rec['act_rank_by_mean_rms_top8'][:3]}"
    )
    return rec, rem_rms, rem_token


# ---------------------------------------------------------------------------
# topology ranking + curves
# ---------------------------------------------------------------------------

def energy_cover_count(scores: np.ndarray, fracs=(0.5, 0.9, 0.99)) -> dict:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    total = float(s.sum())
    if total <= 0:
        return {f"n{int(f*100)}": 0 for f in fracs} | {"total": 0.0, "gini": 0.0}
    order = np.argsort(-s)
    csum = np.cumsum(s[order])
    out = {"total": total}
    for f in fracs:
        out[f"n{int(round(f*100))}"] = int(np.searchsorted(csum, f * total) + 1)
    # Gini of scores
    x = np.sort(s)
    n = x.size
    if n and x[-1] > 0:
        out["gini"] = float((2.0 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n)
    else:
        out["gini"] = 0.0
    return out


def hold_scores(Y: np.ndarray, Yhat: np.ndarray) -> dict:
    return {
        "hold_cosine": mean_row_cosine(Y[HOLD], Yhat[HOLD]),
        "hold_min_row": min_row_cosine(Y[HOLD], Yhat[HOLD]),
        "hold_rel_l2": rel_l2(Y[HOLD], Yhat[HOLD]),
        "fit_cosine": mean_row_cosine(Y[FIT], Yhat[FIT]),
        "all_cosine": mean_row_cosine(Y, Yhat),
    }


def overlap_topk(a: np.ndarray, b: np.ndarray, k: int) -> int:
    ka = set(np.argsort(-a)[:k].tolist())
    kb = set(np.argsort(-b)[:k].tolist())
    return int(len(ka & kb))


def ladder_from_n(n: int, extra=()) -> list[int]:
    ks = {0, 1, 2, 3, 4, 8, 16, 32, 42, 64}
    p = 128
    while p < n:
        ks.add(p)
        p *= 2
    for e in extra:
        if 0 <= e <= n:
            ks.add(int(e))
    ks.add(n)
    return sorted(k for k in ks if 0 <= k <= n)


def apply_columns(Yhat: np.ndarray, X: np.ndarray, dW: np.ndarray, cols: np.ndarray) -> None:
    if cols.size == 0:
        return
    Yhat += X[:, cols] @ dW[:, cols].T


def apply_rows(Yhat: np.ndarray, Y: np.ndarray, rows: np.ndarray) -> None:
    if rows.size == 0:
        return
    Yhat[:, rows] = Y[:, rows]


def apply_elems(Yhat: np.ndarray, X: np.ndarray, dW: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> None:
    if ii.size == 0:
        return
    # Yhat[t, ii[p]] += X[t, jj[p]] * dW[ii[p], jj[p]]
    contrib = X[:, jj] * dW[ii, jj][None, :]
    np.add.at(Yhat, (slice(None), ii), contrib)


def curve_from_running(
    *,
    Y: np.ndarray,
    Yq: np.ndarray,
    apply_batch,
    order: np.ndarray,
    ks: list[int],
    n_elem: int,
    value_elems_fn,
    index_fn,
    body: float,
) -> list[dict]:
    """Walk prefix of `order`, apply new units, score hold at each k in ks."""
    Yhat = Yq.copy()
    points = []
    prev = 0
    kset = set(ks)
    # always include 0
    if 0 not in kset:
        ks = [0] + list(ks)
    for k in ks:
        if k > prev:
            apply_batch(Yhat, order[prev:k])
            prev = k
        hs = hold_scores(Y, Yhat)
        n_exact = int(value_elems_fn(k))
        idx = index_fn(order[:k])
        index_bits = int(idx["cheapest_bits"]) if "cheapest_bits" in idx else int(idx.get("bits", 0))
        value_bits = n_exact * VALUE_BITS
        extra_bpw = (index_bits + value_bits) / float(n_elem)
        rec = {
            "k": int(k),
            "n_exact": n_exact,
            "frac_exact": n_exact / float(n_elem),
            "body_bpw": body,
            "index_bits": index_bits,
            "value_bits": value_bits,
            "index_bpw": index_bits / float(n_elem),
            "value_bpw": value_bits / float(n_elem),
            "complete_bpw": body + extra_bpw,
            "index": idx,
            **hs,
        }
        points.append(rec)
    return points


def crossing(points: list[dict], bar: float) -> dict | None:
    for p in points:
        if p.get("hold_cosine", -1.0) >= bar:
            n_exact = int(p.get("n_exact", 0))
            return {
                "bar": bar,
                "k": p.get("k"),
                "n_exact": n_exact,
                "frac_exact": p.get("frac_exact"),
                "complete_bpw": p.get("complete_bpw"),
                "hold_cosine": p["hold_cosine"],
                "body_bpw": p.get("body_bpw"),
            }
    return None


def head_spec(kind: str, K: int, M: int) -> tuple[int, int] | None:
    """Return (n_heads, head_width along K) or None."""
    if kind in ("out", "o") and K == 6144:
        if True:  # DN and GQA both 6144
            # DN: 48 x 128; GQA: 24 x 256. Caller passes mixer.
            return None
    return None


# ---------------------------------------------------------------------------
# per-tensor
# ---------------------------------------------------------------------------

def score_tensor(
    *,
    label: str,
    kind: str,
    layer: int,
    W: np.ndarray,
    X: np.ndarray,
    H: np.ndarray | None,
    rem_amp: float,
    bases: list[str],
    deep_elem: bool,
) -> dict:
    M, K = int(W.shape[0]), int(W.shape[1])
    n = M * K
    log(f"score {label} shape=({M},{K}) X={tuple(X.shape)} rem_amp={rem_amp:.4f}")
    X = np.ascontiguousarray(X, dtype=np.float32)
    W = np.ascontiguousarray(W, dtype=np.float32)
    assert X.shape[1] == K, f"{label} X.K={X.shape[1]} W.K={K}"

    Y = X @ W.T
    x_energy_fit = np.sum(np.square(X[FIT], dtype=np.float64), axis=0)
    x_energy_all = np.sum(np.square(X, dtype=np.float64), axis=0)
    x_energy_hold = np.sum(np.square(X[HOLD], dtype=np.float64), axis=0)
    # first128 vs last128 rank stability (activation energy)
    x_e_a = np.sum(np.square(X[:128], dtype=np.float64), axis=0)
    x_e_b = np.sum(np.square(X[128:], dtype=np.float64), axis=0)

    # mixer / residual diagnostics
    x_cover = energy_cover_count(x_energy_all)
    w_col = np.sum(np.square(W, dtype=np.float64), axis=0)
    w_row = np.sum(np.square(W, dtype=np.float64), axis=1)

    residual = None
    if H is not None and M == HIDDEN:
        residual = {
            "write_rms": float(np.sqrt(np.mean(np.square(Y, dtype=np.float64)))),
            "hidden_rms": float(np.sqrt(np.mean(np.square(H, dtype=np.float64)))),
        }
        residual["write_over_hidden_rms"] = residual["write_rms"] / max(residual["hidden_rms"], 1e-20)
        residual["write_dot_hidden_cos"] = mean_row_cosine(Y, H)

    # head geometry
    mixer = "gqa" if is_gqa(layer) else "dn"
    heads = None
    if kind in ("out", "o") and K == 6144:
        if mixer == "dn":
            heads = (LINEAR_VALUE_HEADS, 128)
        else:
            heads = (GQA_HEADS, 256)
    elif kind == "q" and K == HIDDEN and M == 12288:
        heads = (GQA_HEADS, HIDDEN)  # heads along output; treat as 24 row-blocks of 512
        # actually q is [12288, 5120] = 24 * 512 output rows. Head is along M, not K.
        heads = None
        q_heads = (GQA_HEADS, 512)  # output-row heads
    else:
        q_heads = None
    if kind == "q" and M == 12288:
        q_heads = (GQA_HEADS, 512)
    else:
        q_heads = None

    # residual-axis identity
    residual_axis = None
    if K == HIDDEN:
        residual_axis = "col"  # reads residual
    elif M == HIDDEN:
        residual_axis = "row"  # writes residual

    rec = {
        "label": label,
        "kind": kind,
        "layer": layer,
        "mixer": mixer,
        "shape": [M, K],
        "elements": n,
        "rem_amp_rms_to_L63": float(rem_amp),
        "x_energy_cover": x_cover,
        "x_col_max_over_median": float(np.max(x_energy_all) / max(np.median(x_energy_all), 1e-20)),
        "rows_per_dim": float(X.shape[0]) / float(K),
        "fit_n": int(FIT.size),
        "hold_n": int(HOLD.size),
        "site": "CAPTURED_REAL_BF16_POST_NORM_HIDDEN"
        if K == HIDDEN
        else ("mixer_proxy_v_silu_z_or_repeat_v_sigmoid_qgate" if K == 6144 else "reconstructed_swiglu"),
        "residual_axis": residual_axis,
        "residual": residual,
        "rank_stability_x_energy_top42_first128_last128": overlap_topk(x_e_a, x_e_b, 42),
        "bases": {},
    }

    # compile-time residual set
    CT = np.array([3994, 3456, 310], dtype=np.int64)

    for bkind in bases:
        t0 = time.perf_counter()
        Wq = quantize(W, bkind)
        dW = (W - Wq).astype(np.float32, copy=False)
        Yq = X @ Wq.T
        body = body_bpw(bkind)
        hs0 = hold_scores(Y, Yq)
        wcos = mean_row_cosine(W.reshape(1, -1), Wq.reshape(1, -1))

        # functional scores on FIT
        # column: ||X_fit[:,j]||^2 * ||dW[:,j]||^2   (exact column contribution energy)
        dw_col = np.sum(np.square(dW, dtype=np.float64), axis=0)
        col_func = x_energy_fit * dw_col
        col_w = w_col  # |W| column energy
        col_dw = dw_col  # residual-magnitude columns
        col_x = x_energy_fit

        # row: ||dY_fit[:,i]||^2 = actual output-row error energy
        dY = Y - Yq
        row_func = np.sum(np.square(dY[FIT], dtype=np.float64), axis=0)
        row_w = w_row
        row_dw = np.sum(np.square(dW, dtype=np.float64), axis=1)

        # element scores: dW[i,j]^2 * x_energy_fit[j]
        # materialize float32
        elem_score = (np.square(dW) * x_energy_fit[None, :].astype(np.float32)).astype(np.float32, copy=False)

        # block G=64 along K (row-major groups; same as quant groups when K%64==0)
        Gblk = 64
        padK = ((K + Gblk - 1) // Gblk) * Gblk
        nblk_row = padK // Gblk
        if padK != K:
            es = np.zeros((M, padK), dtype=np.float32)
            es[:, :K] = elem_score
        else:
            es = elem_score
        block_score = es.reshape(M, nblk_row, Gblk).sum(axis=2).reshape(-1)  # (M*nblk_row,)
        n_blocks = int(block_score.size)

        # heads along K
        head_score = None
        n_heads = 0
        head_w = 0
        if heads is not None:
            n_heads, head_w = heads
            assert n_heads * head_w == K
            head_score = col_func.reshape(n_heads, head_w).sum(axis=1)

        # q output-row heads
        qhead_score = None
        if q_heads is not None:
            n_qh, qh_w = q_heads
            qhead_score = row_func.reshape(n_qh, qh_w).sum(axis=1)

        # covers + overlaps
        col_cover = energy_cover_count(col_func)
        row_cover = energy_cover_count(row_func)
        elem_cover = energy_cover_count(elem_score)
        block_cover = energy_cover_count(block_score)

        ov_col_42 = {
            "func_vs_W": overlap_topk(col_func, col_w, 42),
            "func_vs_dW": overlap_topk(col_func, col_dw, 42),
            "func_vs_X": overlap_topk(col_func, col_x, 42),
            "W_vs_X": overlap_topk(col_w, col_x, 42),
        }
        ov_row_42 = {
            "func_vs_W": overlap_topk(row_func, row_w, 42),
            "func_vs_dW": overlap_topk(row_func, row_dw, 42),
        }

        # channel 3994 ranks
        def rank_of(scores: np.ndarray, idx: int) -> int:
            return int(np.sum(scores > scores[idx]) + 1)

        ch_ranks = {}
        if residual_axis == "col" and K == HIDDEN:
            for c in (3994, 3456, 310):
                ch_ranks[str(c)] = {
                    "func": rank_of(col_func, c),
                    "W": rank_of(col_w, c),
                    "X": rank_of(col_x, c),
                    "func_share": float(col_func[c] / max(col_func.sum(), 1e-20)),
                }
        if residual_axis == "row" and M == HIDDEN:
            for c in (3994, 3456, 310):
                ch_ranks[str(c)] = {
                    "func": rank_of(row_func, c),
                    "W": rank_of(row_w, c),
                    "func_share": float(row_func[c] / max(row_func.sum(), 1e-20)),
                }

        # residual-channel scores (for shared-set aggregation)
        if residual_axis == "col" and K == HIDDEN:
            resid_func = col_func.copy()
        elif residual_axis == "row" and M == HIDDEN:
            resid_func = row_func.copy()
        else:
            resid_func = None

        # ---- topology curves ----
        topologies = {}

        # COLUMNS
        col_order = np.argsort(-col_func)
        extra = [col_cover.get("n50", 0), col_cover.get("n90", 0), col_cover.get("n99", 0)]
        col_ks = ladder_from_n(K, extra=extra)
        # denser near the start
        col_ks = sorted(set(col_ks) | set(range(0, min(K, 96), 1)))

        def apply_col_batch(Yhat, batch):
            apply_columns(Yhat, X, dW, batch)

        def val_cols(k):
            return k * M

        def idx_cols(units):
            bits = structured_index_bits(int(units.size), K)
            pos = units.astype(np.int64)  # column ids
            rice = index_from_positions(pos, K)
            return {
                "cheapest_bits": min(bits, rice["cheapest_bits"] if units.size else 0) if units.size else 0,
                "structured_log2_bits": bits,
                "unit_rice": rice,
                "cheapest": "structured_or_rice",
            }

        topologies["col_func"] = curve_from_running(
            Y=Y, Yq=Yq, apply_batch=apply_col_batch, order=col_order,
            ks=col_ks, n_elem=n, value_elems_fn=val_cols, index_fn=idx_cols, body=body,
        )

        # |W| column control (same apply, different order)
        col_order_w = np.argsort(-col_w)
        topologies["col_W"] = curve_from_running(
            Y=Y, Yq=Yq, apply_batch=apply_col_batch, order=col_order_w,
            ks=[k for k in col_ks if k <= 256 or k in (0, 42, 64, 128, 256, 512, 1024, K)],
            n_elem=n, value_elems_fn=val_cols, index_fn=idx_cols, body=body,
        )

        # ROWS
        row_order = np.argsort(-row_func)
        extra_r = [row_cover.get("n50", 0), row_cover.get("n90", 0), row_cover.get("n99", 0)]
        row_ks = ladder_from_n(M, extra=extra_r)
        row_ks = sorted(set(row_ks) | set(range(0, min(M, 64), 1)))

        def apply_row_batch(Yhat, batch):
            apply_rows(Yhat, Y, batch)

        def val_rows(k):
            return k * K

        def idx_rows(units):
            bits = structured_index_bits(int(units.size), M)
            rice = index_from_positions(units.astype(np.int64), M)
            return {
                "cheapest_bits": min(bits, rice["cheapest_bits"] if units.size else 0) if units.size else 0,
                "structured_log2_bits": bits,
                "unit_rice": rice,
                "cheapest": "structured_or_rice",
            }

        topologies["row_func"] = curve_from_running(
            Y=Y, Yq=Yq, apply_batch=apply_row_batch, order=row_order,
            ks=row_ks, n_elem=n, value_elems_fn=val_rows, index_fn=idx_rows, body=body,
        )
        row_order_w = np.argsort(-row_w)
        topologies["row_W"] = curve_from_running(
            Y=Y, Yq=Yq, apply_batch=apply_row_batch, order=row_order_w,
            ks=[k for k in row_ks if k <= 128 or k in (0, 1, 3, 8, 32, 64, 128, 256, M)],
            n_elem=n, value_elems_fn=val_rows, index_fn=idx_rows, body=body,
        )

        # BLOCKS 64
        blk_order = np.argsort(-block_score)
        extra_b = [block_cover.get("n50", 0), block_cover.get("n90", 0), block_cover.get("n99", 0)]
        max_blk_walk = min(n_blocks, 8192)
        blk_ks = [k for k in ladder_from_n(max_blk_walk, extra=extra_b) if k <= max_blk_walk]
        if n_blocks not in blk_ks and n_blocks <= 4096:
            blk_ks.append(n_blocks)
            blk_ks = sorted(set(blk_ks))

        def apply_blk_batch(Yhat, batch):
            # each unit is linearized (row * nblk_row + g)
            if batch.size == 0:
                return
            rows = (batch // nblk_row).astype(np.int64)
            gs = (batch % nblk_row).astype(np.int64)
            j0 = gs * Gblk
            # clip to K
            for r, j in zip(rows.tolist(), j0.tolist()):
                j1 = min(j + Gblk, K)
                if j >= K:
                    continue
                Yhat[:, r] += X[:, j:j1] @ dW[r, j:j1]

        def val_blk(k):
            # last group may be short; charge Gblk (upper bound) except we clip later
            return k * Gblk

        def idx_blk(units):
            bits = structured_index_bits(int(units.size), n_blocks)
            rice = index_from_positions(units.astype(np.int64), n_blocks)
            return {
                "cheapest_bits": min(bits, rice["cheapest_bits"] if units.size else 0) if units.size else 0,
                "structured_log2_bits": bits,
                "unit_rice": rice,
                "cheapest": "structured_or_rice",
            }

        topologies["block64_func"] = curve_from_running(
            Y=Y, Yq=Yq, apply_batch=apply_blk_batch, order=blk_order,
            ks=blk_ks, n_elem=n, value_elems_fn=val_blk, index_fn=idx_blk, body=body,
        )

        # HEADS
        if head_score is not None:
            hd_order = np.argsort(-head_score)
            hd_ks = list(range(0, n_heads + 1))
            hd_cover = energy_cover_count(head_score)

            def apply_hd_batch(Yhat, batch):
                if batch.size == 0:
                    return
                cols = []
                for h in batch.tolist():
                    cols.extend(range(h * head_w, (h + 1) * head_w))
                apply_columns(Yhat, X, dW, np.asarray(cols, dtype=np.int64))

            def val_hd(k):
                return k * head_w * M

            def idx_hd(units):
                bits = structured_index_bits(int(units.size), n_heads)
                return {"cheapest_bits": bits, "structured_log2_bits": bits, "cheapest": "head_id"}

            topologies["head_func"] = curve_from_running(
                Y=Y, Yq=Yq, apply_batch=apply_hd_batch, order=hd_order,
                ks=hd_ks, n_elem=n, value_elems_fn=val_hd, index_fn=idx_hd, body=body,
            )
            # |W| heads
            hd_w = w_col.reshape(n_heads, head_w).sum(axis=1)
            topologies["head_W"] = curve_from_running(
                Y=Y, Yq=Yq, apply_batch=apply_hd_batch, order=np.argsort(-hd_w),
                ks=hd_ks, n_elem=n, value_elems_fn=val_hd, index_fn=idx_hd, body=body,
            )
        else:
            hd_cover = None

        if qhead_score is not None:
            n_qh, qh_w = q_heads
            qh_order = np.argsort(-qhead_score)
            qh_ks = list(range(0, n_qh + 1))

            def apply_qh_batch(Yhat, batch):
                if batch.size == 0:
                    return
                rows = []
                for h in batch.tolist():
                    rows.extend(range(h * qh_w, (h + 1) * qh_w))
                apply_rows(Yhat, Y, np.asarray(rows, dtype=np.int64))

            def val_qh(k):
                return k * qh_w * K

            def idx_qh(units):
                bits = structured_index_bits(int(units.size), n_qh)
                return {"cheapest_bits": bits, "structured_log2_bits": bits, "cheapest": "qhead_id"}

            topologies["qhead_func"] = curve_from_running(
                Y=Y, Yq=Yq, apply_batch=apply_qh_batch, order=qh_order,
                ks=qh_ks, n_elem=n, value_elems_fn=val_qh, index_fn=idx_qh, body=body,
            )

        # ELEMENTS
        flat = elem_score.reshape(-1)
        k_elem_max = min(n, 500_000 if deep_elem else 200_000)
        # top-k indices
        if k_elem_max >= n:
            elem_idx = np.argsort(-flat)
        else:
            part = np.argpartition(-flat, k_elem_max)[:k_elem_max]
            elem_idx = part[np.argsort(-flat[part])]
        extra_e = [elem_cover.get("n50", 0), elem_cover.get("n90", 0), elem_cover.get("n99", 0)]
        extra_e = [e for e in extra_e if e <= k_elem_max]
        elem_ks = ladder_from_n(k_elem_max, extra=extra_e)
        # add percent rungs
        for frac in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
            ke = int(round(frac * n))
            if 0 < ke <= k_elem_max:
                elem_ks.append(ke)
        elem_ks = sorted(set(elem_ks))

        def apply_el_batch(Yhat, batch):
            ii = (batch // K).astype(np.int64)
            jj = (batch % K).astype(np.int64)
            apply_elems(Yhat, X, dW, ii, jj)

        def val_el(k):
            return k

        def idx_el(units):
            rice = index_from_positions(units.astype(np.int64), n)
            return rice

        topologies["elem_func"] = curve_from_running(
            Y=Y, Yq=Yq, apply_batch=apply_el_batch, order=elem_idx,
            ks=elem_ks, n_elem=n, value_elems_fn=val_el, index_fn=idx_el, body=body,
        )

        # |W| element control, smaller ladder
        w_flat = np.abs(W).reshape(-1)
        if k_elem_max >= n:
            w_idx = np.argsort(-w_flat)
        else:
            part = np.argpartition(-w_flat, k_elem_max)[:k_elem_max]
            w_idx = part[np.argsort(-w_flat[part])]
        w_elem_ks = [k for k in elem_ks if k <= 65536 or k in (0, 1024, 4096, 16384, 65536, 131072)]
        topologies["elem_W"] = curve_from_running(
            Y=Y, Yq=Yq, apply_batch=apply_el_batch, order=w_idx,
            ks=w_elem_ks, n_elem=n, value_elems_fn=val_el, index_fn=idx_el, body=body,
        )

        # COMPILE-TIME residual channels {3994,3456,310} prefix
        ct_points = []
        if residual_axis == "col" and K == HIDDEN:
            Yhat = Yq.copy()
            for kk, c in enumerate(CT, start=1):
                apply_columns(Yhat, X, dW, np.array([c], dtype=np.int64))
                hs = hold_scores(Y, Yhat)
                n_exact = kk * M
                extra_bpw = (n_exact * VALUE_BITS) / float(n)  # index 0
                ct_points.append(
                    {
                        "k": kk,
                        "channels": [int(x) for x in CT[:kk]],
                        "n_exact": n_exact,
                        "frac_exact": n_exact / float(n),
                        "index_bits": 0,
                        "body_bpw": body,
                        "complete_bpw": body + extra_bpw,
                        **hs,
                    }
                )
        elif residual_axis == "row" and M == HIDDEN:
            Yhat = Yq.copy()
            for kk, c in enumerate(CT, start=1):
                apply_rows(Yhat, Y, np.array([c], dtype=np.int64))
                hs = hold_scores(Y, Yhat)
                n_exact = kk * K
                extra_bpw = (n_exact * VALUE_BITS) / float(n)
                ct_points.append(
                    {
                        "k": kk,
                        "channels": [int(x) for x in CT[:kk]],
                        "n_exact": n_exact,
                        "frac_exact": n_exact / float(n),
                        "index_bits": 0,
                        "body_bpw": body,
                        "complete_bpw": body + extra_bpw,
                        **hs,
                    }
                )
        topologies["channel_compile_time"] = ct_points

        # residual-channel curve (func rank on residual axis) — 0 index if later shared
        if resid_func is not None:
            axis_n = HIDDEN
            resid_order = np.argsort(-resid_func)
            rks = [0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
            rks = [k for k in rks if k <= axis_n]

            if residual_axis == "col":

                def apply_rs(Yhat, batch):
                    apply_columns(Yhat, X, dW, batch)

                def val_rs(k):
                    return k * M
            else:

                def apply_rs(Yhat, batch):
                    apply_rows(Yhat, Y, batch)

                def val_rs(k):
                    return k * K

            def idx_rs(units):
                # per-tensor residual list still needs an index unless compile-time shared
                bits = structured_index_bits(int(units.size), HIDDEN)
                return {"cheapest_bits": bits, "structured_log2_bits": bits, "cheapest": "channel_id"}

            topologies["resid_channel_func"] = curve_from_running(
                Y=Y, Yq=Yq, apply_batch=apply_rs, order=resid_order,
                ks=rks, n_elem=n, value_elems_fn=val_rs, index_fn=idx_rs, body=body,
            )
            # zero-index variant (shared compile-time): same hold, index_bits=0
            zpts = []
            for p in topologies["resid_channel_func"]:
                q = dict(p)
                q["index_bits"] = 0
                q["index_bpw"] = 0.0
                q["complete_bpw"] = body + q["value_bpw"]
                q["index_note"] = "zero_if_compile_time_shared"
                zpts.append(q)
            topologies["resid_channel_func_zero_index"] = zpts

        # residual-proxy on write tensors
        rproxy = None
        if H is not None and M == HIDDEN:
            Rt = H + Y
            Rq = H + Yq
            rproxy = {
                "k0_cosine": mean_row_cosine(Rt, Rq),
                "k0_rel_l2": rel_l2(Rt, Rq),
            }
            # after exacting top-42 func cols and top-1 func row
            Yhat = Yq.copy()
            apply_columns(Yhat, X, dW, col_order[:42])
            rproxy["top42_col_func_cosine"] = mean_row_cosine(Rt, H + Yhat)
            Yhat = Yq.copy()
            apply_rows(Yhat, Y, row_order[:1])
            rproxy["top1_row_func_cosine"] = mean_row_cosine(Rt, H + Yhat)

        # crossings
        crosses = {}
        for tname_, pts in topologies.items():
            if not pts or "hold_cosine" not in pts[0]:
                continue
            crosses[tname_] = {
                "0.990": crossing(pts, 0.990),
                "0.995": crossing(pts, 0.995),
            }

        # cheapest point that beats uniform of same total size
        # uniform holds at this tensor: we have hs0 for this base; collect after all bases
        beat = {}
        for tname_, pts in topologies.items():
            if not pts or "hold_cosine" not in pts[0]:
                continue
            # vs this base's k=0 (always beats if exceptions help)
            # recorded later vs other uniforms
            beat[tname_] = {
                "first_hold_ge_k0_plus_1e-3": next(
                    (
                        {"k": p["k"], "hold_cosine": p["hold_cosine"], "complete_bpw": p["complete_bpw"]}
                        for p in pts
                        if p["hold_cosine"] >= hs0["hold_cosine"] + 1e-3
                    ),
                    None,
                )
            }

        # concentration of functional energy inside top columns (elem vs col)
        top_cols = col_order[: min(42, K)]
        # within those columns, gini of |dW[i,j]| across rows
        if top_cols.size:
            sub = np.abs(dW[:, top_cols])
            row_frac = np.sort(sub, axis=0)
            # rows to cover 90% of |dW| per top col (mean)
            cover90 = []
            for ci in range(top_cols.size):
                v = np.square(dW[:, top_cols[ci]], dtype=np.float64)
                tot = float(v.sum())
                if tot <= 0:
                    cover90.append(0)
                    continue
                csum = np.cumsum(np.sort(v)[::-1])
                cover90.append(int(np.searchsorted(csum, 0.9 * tot) + 1))
            col_internal = {
                "top42_mean_rows_to_90pct_dw2": float(np.mean(cover90)) if cover90 else 0.0,
                "top42_median_rows_to_90pct_dw2": float(np.median(cover90)) if cover90 else 0.0,
                "top1_rows_to_90pct_dw2": int(cover90[0]) if cover90 else 0,
            }
        else:
            col_internal = {}

        # how many heads own the top-42 func cols
        head_own = None
        if heads is not None:
            n_heads, head_w = heads
            top42 = col_order[: min(42, K)]
            hid = (top42 // head_w).astype(int)
            head_own = {
                "n_heads_owning_top42_func_cols": int(np.unique(hid).size),
                "n_heads": n_heads,
                "head_width": head_w,
                "top16_func_heads": [int(x) for x in np.argsort(-head_score)[:16]],
                "head_energy_cover": energy_cover_count(head_score),
            }

        base_rec = {
            "body_bpw": body,
            "weight_cosine": wcos,
            "k0": hs0,
            "col_func_cover": col_cover,
            "row_func_cover": row_cover,
            "elem_func_cover": elem_cover,
            "block64_func_cover": block_cover,
            "head_func_cover": hd_cover,
            "overlap_top42_cols": ov_col_42,
            "overlap_top42_rows": ov_row_42,
            "channel_ranks": ch_ranks,
            "col_internal_row_cover": col_internal,
            "head_ownership": head_own,
            "residual_proxy": rproxy,
            "top16_col_func": [int(x) for x in col_order[:16]],
            "top16_col_W": [int(x) for x in np.argsort(-col_w)[:16]],
            "top16_col_X": [int(x) for x in np.argsort(-col_x)[:16]],
            "top8_row_func": [int(x) for x in row_order[:8]],
            "top8_row_W": [int(x) for x in np.argsort(-row_w)[:8]],
            "crossings": crosses,
            "beat_vs_k0": beat,
            "curves": {k: _thin_curve(v) for k, v in topologies.items()},
            "resid_func_top32": [int(x) for x in np.argsort(-resid_func)[:32]] if resid_func is not None else None,
            "resid_func_scores_top32": [float(resid_func[i]) for i in np.argsort(-resid_func)[:32]]
            if resid_func is not None
            else None,
            "resid_func_sum": float(resid_func.sum()) if resid_func is not None else None,
            "resid_func_all": resid_func.astype(np.float64) if resid_func is not None else None,
            "wall_s": time.perf_counter() - t0,
        }
        rec["bases"][bkind] = base_rec
        log(
            f"  {bkind} k0_hold={hs0['hold_cosine']:.6f} wcos={wcos:.6f} "
            f"col_n90={col_cover.get('n90')} elem_n90={elem_cover.get('n90')} "
            f"ov_func_W_42={ov_col_42['func_vs_W']} ov_W_X_42={ov_col_42['W_vs_X']} "
            f"c3994={ch_ranks.get('3994')} wall={base_rec['wall_s']:.1f}s"
        )
        del Wq, dW, Yq, dY, elem_score, es, flat
        gc.collect()

    del Y, W, X
    gc.collect()
    return rec


def _thin_curve(points: list[dict]) -> list[dict]:
    """Drop bulky nested rice dumps from every point; keep cheapest bits."""
    out = []
    for p in points:
        q = {k: v for k, v in p.items() if k != "index"}
        idx = p.get("index")
        if isinstance(idx, dict):
            q["index_cheapest"] = idx.get("cheapest")
            q["index_cheapest_bits"] = idx.get("cheapest_bits")
            q["index_rice_bits"] = idx.get("rice_bits")
            q["index_structured_bits"] = idx.get("structured_log2_bits")
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# X cache per layer
# ---------------------------------------------------------------------------

def layer_inputs(layer: int, need: set[str]) -> dict:
    """Load hidden + build mixer / swiglu X as required. Returns dict of arrays."""
    H = load_hidden(layer)
    out = {"H": H}
    if need & {"gate", "up", "qkv", "z", "q", "k", "v"}:
        out["H"] = H
    if "down" in need:
        log(f"L{layer} build down X (SwiGLU recon)")
        Wg = load_tensor(tname(layer, "mlp.gate_proj.weight"))
        g = H @ Wg.T
        del Wg
        Wu = load_tensor(tname(layer, "mlp.up_proj.weight"))
        u = H @ Wu.T
        del Wu
        out["down"] = np.ascontiguousarray(silu(g) * u, dtype=np.float32)
        del g, u
        gc.collect()
    if "out" in need:
        log(f"L{layer} build out/o mixer-proxy X")
        if is_gqa(layer):
            Wq = load_tensor(tname(layer, "self_attn.q_proj.weight"))
            Wv = load_tensor(tname(layer, "self_attn.v_proj.weight"))
            out["out"] = gqa_out_proxy(H, Wq, Wv)
            del Wq, Wv
        else:
            Wqkv = load_tensor(tname(layer, "linear_attn.in_proj_qkv.weight"))
            Wz = load_tensor(tname(layer, "linear_attn.in_proj_z.weight"))
            fused = fuse_qkvz(Wqkv, Wz)
            del Wqkv, Wz
            out["out"] = deltanet_out_proxy(H, fused)
            del fused
        gc.collect()
    return out


def tensor_jobs() -> list[tuple]:
    """(layer, kind, suffix, xkey, deep_elem)."""
    jobs = []
    for L in LAYERS:
        jobs.append((L, "gate", "mlp.gate_proj.weight", "H", L in (0, 32, 63)))
        jobs.append((L, "up", "mlp.up_proj.weight", "H", L in (0, 32)))
        jobs.append((L, "down", "mlp.down_proj.weight", "down", L in (0, 32, 63)))
        if is_gqa(L):
            jobs.append((L, "q", "self_attn.q_proj.weight", "H", L in (3, 63)))
            jobs.append((L, "o", "self_attn.o_proj.weight", "out", L in (3, 63)))
            if L in (3, 63):
                jobs.append((L, "v", "self_attn.v_proj.weight", "H", False))
        else:
            jobs.append((L, "qkv", "linear_attn.in_proj_qkv.weight", "H", L in (0, 32)))
            jobs.append((L, "out", "linear_attn.out_proj.weight", "out", L in (0, 32)))
    return jobs


def bases_for(kind: str, layer: int) -> list[str]:
    # always Q2, Q3; Q4 on a subset as incumbent; binary on a subset as cheap base
    b = ["q2_g64", "q3_g64"]
    if (kind, layer) in {
        ("out", 0),
        ("o", 63),
        ("gate", 0),
        ("gate", 32),
        ("down", 0),
        ("down", 63),
        ("qkv", 0),
        ("q", 3),
    }:
        b.append("q4_g64")
        b.append("binary_g128")
    if (kind, layer) == ("out", 0):
        b.append("none")
        b.append("q2_g128")
        b.append("q3_g128")
    return b


# ---------------------------------------------------------------------------
# complete-BPW projection + compounding
# ---------------------------------------------------------------------------

def class_of(kind: str, mixer: str) -> str:
    return {
        "gate": "gate",
        "up": "up",
        "down": "down",
        "qkv": "dn_qkvz",  # qkv only; ba separate
        "out": "dn_out",
        "q": "gqa_q",
        "o": "gqa_o",
        "v": "gqa_v",
    }[kind]


def project_complete(alloc: dict[str, float]) -> float:
    """alloc: class -> physical BPW. Missing classes stay at G0 Q4 4.25 / f32 32."""
    g0 = {
        "gate": 4.25,
        "up": 4.25,
        "down": 4.25,
        "dn_qkvz": 4.25,
        "dn_out": 4.25001,
        "embed": 4.25,
        "lm_head": 4.25,
        "gqa_q": 4.25001,
        "gqa_o": 4.25001,
        "gqa_k": 4.25006,
        "gqa_v": 4.25006,
        "dn_ba": 4.25065,
        "small_f32": 32.0,
    }
    bits = 0.0
    for k, e in MASS.items():
        b = alloc.get(k, g0[k])
        bits += e * b
    return bits / float(N_PARAMS)


def main() -> None:
    t_all = time.perf_counter()
    cited = json.loads((CAP / "capture-result.json").read_text())
    cap_sha = sha256_file(CAP / "capture-result.json")
    l00_sha = sha256_file(CAP / "hidden" / "L00.f32")
    log(f"capture-result.json sha256={cap_sha} sha256_self={cited.get('sha256_self')}")
    log(f"L00.f32 sha256={l00_sha}")

    # Q4 self-check L0 out_proj
    W = load_tensor(tname(0, "linear_attn.out_proj.weight"))
    Wq = quantize(W, "q4_g64")
    wcos = mean_row_cosine(W.reshape(1, -1), Wq.reshape(1, -1))
    # excess kurtosis
    c = W.astype(np.float64).reshape(-1)
    c = c - c.mean()
    m2 = float(np.mean(c * c))
    m4 = float(np.mean(c * c * c * c))
    kurt = m4 / (m2 * m2) - 3.0
    log(f"Q4 self-check L0.out weight_cosine={wcos:.8f} kurt={kurt:.4f}")
    del W, Wq, c
    gc.collect()

    amp, rem_rms, rem_token = amplification_census()

    smoke = os.environ.get("G1_FE_SMOKE") == "1"
    if smoke:
        global LAYERS
        LAYERS = (0,)
    jobs = tensor_jobs()
    if smoke:
        jobs = [j for j in jobs if j[1] == "out"]
    # group jobs by layer to reuse X
    by_layer: dict[int, list] = {}
    for job in jobs:
        by_layer.setdefault(job[0], []).append(job)

    tensors = []
    shared_scores = {L: None for L in LAYERS}  # resid_func * rem_amp^2 per layer, accumulated

    # accumulate amp-weighted residual-channel scores across tensors
    shared_acc = np.zeros(HIDDEN, dtype=np.float64)
    shared_acc_n = 0

    for L in LAYERS:
        need = set()
        for _, kind, _, xkey, _ in by_layer[L]:
            need.add(xkey if xkey != "H" else "gate")
            if xkey == "down":
                need.add("down")
            if xkey == "out":
                need.add("out")
        xin = layer_inputs(L, need)
        rem = float(rem_rms[L])
        for layer, kind, suffix, xkey, deep in by_layer[L]:
            label = f"L{layer}.{kind}"
            W = load_tensor(tname(layer, suffix))
            X = xin["H"] if xkey == "H" else xin[xkey]
            H = xin["H"] if W.shape[0] == HIDDEN else None
            rec = score_tensor(
                label=label,
                kind=kind,
                layer=layer,
                W=W,
                X=X,
                H=H,
                rem_amp=rem,
                bases=bases_for(kind, layer),
                deep_elem=deep,
            )
            # pull residual scores out of last computed base (q3 preferred)
            bases = rec["bases"]
            pick = "q3_g64" if "q3_g64" in bases else next(iter(bases))
            rf = bases[pick].pop("resid_func_all", None)
            if rf is not None:
                w = (rem ** 2)
                shared_acc += rf * w
                shared_acc_n += 1
                rec["bases"][pick]["resid_func_amp_weight"] = float(w)
            tensors.append(rec)
            # checkpoint
            payload_partial = {"n_tensors_done": len(tensors), "last": label}
            OUT.write_text(json.dumps({"partial": payload_partial}, indent=2))
            del W
            gc.collect()
        del xin
        gc.collect()

    # shared residual-channel ranking
    shared_rank = np.argsort(-shared_acc)
    shared = {
        "n_tensors_contributing": shared_acc_n,
        "score_is": "sum_tensors (resid_func * remaining_rms_to_L63^2) on Q3 body",
        "top32": [int(x) for x in shared_rank[:32]],
        "top32_scores": [float(shared_acc[i]) for i in shared_rank[:32]],
        "rank_3994": int(np.sum(shared_acc > shared_acc[3994]) + 1),
        "rank_3456": int(np.sum(shared_acc > shared_acc[3456]) + 1),
        "rank_310": int(np.sum(shared_acc > shared_acc[310]) + 1),
        "overlap_top3_with_3994_3456_310": int(
            len(set(shared_rank[:3].tolist()) & {3994, 3456, 310})
        ),
        "overlap_top8_with_3994_3456_310": int(
            len(set(shared_rank[:8].tolist()) & {3994, 3456, 310})
        ),
        "energy_cover": energy_cover_count(shared_acc),
    }
    log(
        f"shared residual channels top8={shared['top32'][:8]} "
        f"rank3994={shared['rank_3994']} rank3456={shared['rank_3456']} rank310={shared['rank_310']} "
        f"top3∩ct={shared['overlap_top3_with_3994_3456_310']}"
    )

    # second pass: evaluate shared top-k on each tensor at Q3 (reload)
    log("second pass: shared residual-channel hold at Q3")
    shared_eval = []
    for L in LAYERS:
        jobs_L = [j for j in by_layer[L] if j[1] in ("gate", "up", "down", "qkv", "q", "out", "o", "v")]
        need = set()
        for _, kind, _, xkey, _ in jobs_L:
            if xkey == "down":
                need.add("down")
            elif xkey == "out":
                need.add("out")
            else:
                need.add("gate")
        xin = layer_inputs(L, need)
        for layer, kind, suffix, xkey, _ in jobs_L:
            W = load_tensor(tname(layer, suffix))
            M, K = W.shape
            residual_axis = "col" if K == HIDDEN else ("row" if M == HIDDEN else None)
            if residual_axis is None:
                del W
                continue
            X = xin["H"] if xkey == "H" else xin[xkey]
            Wq = quantize(W, "q3_g64")
            dW = (W - Wq).astype(np.float32, copy=False)
            Y = X @ W.T
            Yq = X @ Wq.T
            body = body_bpw("q3_g64")
            pts = []
            Yhat = Yq.copy()
            prev = 0
            for k in (1, 3, 8, 16, 32):
                batch = shared_rank[prev:k]
                if residual_axis == "col":
                    apply_columns(Yhat, X, dW, batch)
                    n_exact = k * M
                else:
                    apply_rows(Yhat, Y, batch)
                    n_exact = k * K
                hs = hold_scores(Y, Yhat)
                extra = (n_exact * VALUE_BITS) / float(M * K)  # index 0
                pts.append(
                    {
                        "k": k,
                        "channels": [int(x) for x in shared_rank[:k]],
                        "n_exact": n_exact,
                        "index_bits": 0,
                        "complete_bpw": body + extra,
                        **hs,
                    }
                )
                prev = k
            shared_eval.append(
                {
                    "label": f"L{layer}.{kind}",
                    "k0_hold": hold_scores(Y, Yq)["hold_cosine"],
                    "points": pts,
                }
            )
            log(
                f"  shared {layer}.{kind} k0={shared_eval[-1]['k0_hold']:.6f} "
                f"k1={pts[0]['hold_cosine']:.6f} k3={pts[1]['hold_cosine']:.6f} "
                f"k32={pts[-1]['hold_cosine']:.6f}"
            )
            del W, Wq, dW, Y, Yq, Yhat
            gc.collect()
        del xin
        gc.collect()

    # summary tables
    def summarize_tensor(t: dict) -> dict:
        out = {
            "label": t["label"],
            "shape": t["shape"],
            "rem_amp": t["rem_amp_rms_to_L63"],
            "x_n50": t["x_energy_cover"].get("n50"),
            "x_n90": t["x_energy_cover"].get("n90"),
            "x_n99": t["x_energy_cover"].get("n99"),
            "bases": {},
        }
        for bk, br in t["bases"].items():
            bd = {
                "body_bpw": br["body_bpw"],
                "k0_hold": br["k0"]["hold_cosine"],
                "k0_rel_l2": br["k0"]["hold_rel_l2"],
                "weight_cosine": br["weight_cosine"],
                "ov42_func_W": br["overlap_top42_cols"]["func_vs_W"],
                "ov42_W_X": br["overlap_top42_cols"]["W_vs_X"],
                "ov42_func_X": br["overlap_top42_cols"]["func_vs_X"],
                "col_n90": br["col_func_cover"].get("n90"),
                "elem_n90": br["elem_func_cover"].get("n90"),
                "row_n90": br["row_func_cover"].get("n90"),
                "ch3994": br["channel_ranks"].get("3994"),
                "cross_0.995": {},
                "cross_0.990": {},
            }
            for tn, cr in br["crossings"].items():
                bd["cross_0.995"][tn] = cr.get("0.995")
                bd["cross_0.990"][tn] = cr.get("0.990")
            out["bases"][bk] = bd
        return out

    summaries = [summarize_tensor(t) for t in tensors]

    # cheapest base+topo reaching 0.995 per tensor
    cheapest_0995 = []
    for t in tensors:
        best = None
        for bk, br in t["bases"].items():
            for tn, cr in br["crossings"].items():
                hit = cr.get("0.995")
                if hit is None:
                    continue
                cand = {
                    "label": t["label"],
                    "base": bk,
                    "topology": tn,
                    **hit,
                }
                if best is None or cand["complete_bpw"] < best["complete_bpw"]:
                    best = cand
        cheapest_0995.append(best if best is not None else {"label": t["label"], "hit": False})

    # does any cheap base (binary or q2) hit 0.995 on the hard tensors?
    hard = [s for s in cheapest_0995 if isinstance(s, dict)]

    # compounding model using measured write residual-proxy at Q3 and rem_amp
    # e_{l+1} = A_l e_l + q_l ; we approximate final ≈ sum_l rem[l] * q_l
    # q_l from write tensors' k0 hold rel-L2 * write_rms, when available
    write_q = []
    for t in tensors:
        if t["kind"] not in ("down", "out", "o"):
            continue
        if "q3_g64" not in t["bases"]:
            continue
        br = t["bases"]["q3_g64"]
        rp = br.get("residual_proxy") or {}
        write_q.append(
            {
                "label": t["label"],
                "layer": t["layer"],
                "kind": t["kind"],
                "rem_amp": t["rem_amp_rms_to_L63"],
                "k0_hold": br["k0"]["hold_cosine"],
                "k0_rel_l2": br["k0"]["hold_rel_l2"],
                "rproxy_k0": rp.get("k0_cosine"),
                "write_over_hidden": (t.get("residual") or {}).get("write_over_hidden_rms"),
                "final_weight": float(t["rem_amp_rms_to_L63"]) * float(br["k0"]["hold_rel_l2"]),
            }
        )

    # complete BPW recipes from measured crossings
    # Recipe A: Q3 body everywhere GEMV, no exceptions
    rec_q3 = project_complete(
        {k: 3.25 for k in ("gate", "up", "down", "dn_qkvz", "dn_out", "gqa_q", "gqa_o", "gqa_k", "gqa_v", "dn_ba")}
    )
    rec_q2 = project_complete(
        {k: 2.25 for k in ("gate", "up", "down", "dn_qkvz", "dn_out", "gqa_q", "gqa_o", "gqa_k", "gqa_v", "dn_ba")}
    )
    rec_bin = project_complete(
        {k: 1.125 for k in ("gate", "up", "down", "dn_qkvz", "dn_out", "gqa_q", "gqa_o", "gqa_k", "gqa_v", "dn_ba")}
    )
    rec_q4 = project_complete({})  # all G0
    # Recipe: Q3 + compile-time k=3 residual channels both directions
    # island mass from channel-3994 lane: 5,252,608 elems per k, index 0
    island_per_k = 5_252_608
    rec_q3_ct3 = rec_q3 + (3 * island_per_k * 16.0) / float(N_PARAMS)

    # estimate exception BPW to hit 0.995 from sample (median over tensors that have a hit)
    hits_q3_col = []
    hits_q3_elem = []
    hits_q2_elem = []
    hits_q3_row = []
    hits_q3_head = []
    for t in tensors:
        for bk, tn, bucket in (
            ("q3_g64", "col_func", hits_q3_col),
            ("q3_g64", "elem_func", hits_q3_elem),
            ("q3_g64", "row_func", hits_q3_row),
            ("q3_g64", "head_func", hits_q3_head),
            ("q2_g64", "elem_func", hits_q2_elem),
        ):
            if bk not in t["bases"]:
                continue
            hit = t["bases"][bk]["crossings"].get(tn, {}).get("0.995")
            if hit:
                bucket.append({"label": t["label"], **hit})

    def med_bpw(bucket):
        if not bucket:
            return None
        return float(np.median([x["complete_bpw"] for x in bucket]))

    projection = {
        "N": N_PARAMS,
        "G0_complete_bpw": 4.252735126866492,
        "all_q4_language_gemv": rec_q4,
        "all_q3_gemv_embed_q4": rec_q3,
        "all_q2_gemv_embed_q4": rec_q2,
        "all_binary_gemv_embed_q4": rec_bin,
        "all_q3_plus_ct3_island": rec_q3_ct3,
        "island_per_k_elems_cited": island_per_k,
        "median_q3_col_func_bpw_at_0.995": med_bpw(hits_q3_col),
        "median_q3_elem_func_bpw_at_0.995": med_bpw(hits_q3_elem),
        "median_q3_row_func_bpw_at_0.995": med_bpw(hits_q3_row),
        "median_q3_head_func_bpw_at_0.995": med_bpw(hits_q3_head),
        "median_q2_elem_func_bpw_at_0.995": med_bpw(hits_q2_elem),
        "n_hit_q3_col": len(hits_q3_col),
        "n_hit_q3_elem": len(hits_q3_elem),
        "n_hit_q3_row": len(hits_q3_row),
        "n_hit_q3_head": len(hits_q3_head),
        "n_hit_q2_elem": len(hits_q2_elem),
        "n_tensors": len(tensors),
        "hits_q3_col": hits_q3_col,
        "hits_q3_elem": hits_q3_elem,
        "hits_q2_elem": hits_q2_elem,
    }

    # if we applied the per-class median complete_bpw at 0.995 (elem_func Q3)
    # onto that class — only where we have a sample
    class_bpw_q3_elem = {}
    class_pts: dict[str, list[float]] = {}
    for t in tensors:
        if "q3_g64" not in t["bases"]:
            continue
        hit = t["bases"]["q3_g64"]["crossings"].get("elem_func", {}).get("0.995")
        if not hit:
            continue
        cls = class_of(t["kind"], t["mixer"])
        class_pts.setdefault(cls, []).append(hit["complete_bpw"])
    for cls, vs in class_pts.items():
        class_bpw_q3_elem[cls] = float(np.median(vs))
    projection["class_median_q3_elem_0.995"] = class_bpw_q3_elem
    if class_bpw_q3_elem:
        projection["projected_complete_if_class_medians_q3_elem"] = project_complete(class_bpw_q3_elem)

    # beat-uniform: for L0.out, compare Q3+topo vs Q4 at 4.25 and Q3 at 3.25
    beat_uniform = []
    for t in tensors:
        row = {"label": t["label"]}
        q4h = t["bases"].get("q4_g64", {}).get("k0", {}).get("hold_cosine")
        q3h = t["bases"].get("q3_g64", {}).get("k0", {}).get("hold_cosine")
        q2h = t["bases"].get("q2_g64", {}).get("k0", {}).get("hold_cosine")
        binh = t["bases"].get("binary_g128", {}).get("k0", {}).get("hold_cosine")
        row["uniform"] = {"q4": q4h, "q3": q3h, "q2": q2h, "binary": binh}
        # cheapest combo that (hold >= q4h) and complete_bpw < 4.25
        best = None
        for bk, br in t["bases"].items():
            if bk == "q4_g64":
                continue
            for tn, pts in br["curves"].items():
                for p in pts:
                    if q4h is not None and p["hold_cosine"] >= q4h and p["complete_bpw"] < 4.25:
                        if best is None or p["complete_bpw"] < best["complete_bpw"]:
                            best = {
                                "base": bk,
                                "topology": tn,
                                "k": p["k"],
                                "hold_cosine": p["hold_cosine"],
                                "complete_bpw": p["complete_bpw"],
                                "beats": "q4_hold_at_lower_bpw",
                            }
                    if q3h is not None and p["hold_cosine"] >= q3h and p["complete_bpw"] < 3.25:
                        # record separately
                        pass
        row["cheapest_match_q4_hold_below_4.25"] = best
        # cheapest that matches Q3 hold below 3.25
        best3 = None
        for bk, br in t["bases"].items():
            if bk in ("q3_g64", "q3_g128", "q4_g64"):
                continue
            for tn, pts in br["curves"].items():
                for p in pts:
                    if q3h is not None and p["hold_cosine"] >= q3h and p["complete_bpw"] < 3.25:
                        if best3 is None or p["complete_bpw"] < best3["complete_bpw"]:
                            best3 = {
                                "base": bk,
                                "topology": tn,
                                "k": p["k"],
                                "hold_cosine": p["hold_cosine"],
                                "complete_bpw": p["complete_bpw"],
                                "beats": "q3_hold_at_lower_bpw",
                            }
        row["cheapest_match_q3_hold_below_3.25"] = best3
        beat_uniform.append(row)

    # product-of-holds projection (MLP 192) using sample geomean
    def geomean_hold(kind: str, bk: str) -> float | None:
        vs = []
        for t in tensors:
            if t["kind"] != kind or bk not in t["bases"]:
                continue
            vs.append(t["bases"][bk]["k0"]["hold_cosine"])
        if not vs:
            return None
        return float(np.exp(np.mean(np.log(np.maximum(vs, 1e-12)))))

    prod_proj = {
        "note": "PROJECTED: geomean of this lane's sampled layers, raised to 64 (or 192). Not a 64-layer census.",
        "sample_layers": list(LAYERS),
        "cited_mlp192_q3_prod": 0.009305905311825565,
        "cited_mlp192_q4_prod": 0.4078534106896186,
        "cited_q3_gate_geomean_64": 0.9803139293361185,
        "cited_q4_gate_geomean_64": 0.9962828319111336,
    }
    for kind in ("gate", "up", "down"):
        for bk in ("q2_g64", "q3_g64", "q4_g64", "binary_g128"):
            gm = geomean_hold(kind, bk)
            if gm is None:
                continue
            prod_proj[f"sample_geomean_{kind}_{bk}"] = gm
            prod_proj[f"projected_prod64_{kind}_{bk}"] = gm ** 64

    # kernel notes (static)
    kernel = {
        "body": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128 (and uniform_qn g=128 for Q2/Q3 production)",
        "reject": "packed-then-expand-to-Q4-then-generic-GEMV",
        "consume": {
            "col / resid_channel / head": "pack-time zero of selected input columns; post-GEMV saxpy y += x[c] * W_exact[:,c]. geo_tpr64 unchanged.",
            "row": "pack-time skip/zero of selected output rows; post-GEMV exact dots y[r] = dot(W_exact[r], x). Partner-row TG still exists (row 3994 is TG 1997 team 0).",
            "elem": "CSR of (row,col,bf16) ; Metal gather-axpy epilogue. Highest index cost. Only native if k is tiny.",
            "block64": "side buffer of promoted groups; group-aligned with HQ30UQ4. Epilogue dequant-add of those 64-wide row slices.",
            "compile_time_channel": "index bits 0; k saxpy/dot of compile-time ids. Same as g1-channel-3994-island.md.",
        },
    }

    payload = {
        "schema": "hawking.g1.functional_exceptions.v1",
        "wall_s": time.perf_counter() - t_all,
        "rss_max_gb": rss_gb(),
        "identity": {
            "bf16": str(SRC),
            "capture": str(CAP),
            "capture_sha256_self": cited.get("sha256_self"),
            "capture_result_file_sha256": cap_sha,
            "L00_f32_sha256": l00_sha,
            "q4_selfcheck_L0_out_weight_cosine": wcos,
            "q4_selfcheck_L0_out_excess_kurtosis": kurt,
            "fit": "even 128",
            "hold": "odd 128",
            "ranking": "fit-split functional scores; capture is thick enough to RANK, not to estimate a production scale plane",
            "N_params": N_PARAMS,
            "G0_complete_bpw": 4.252735126866492,
        },
        "amplification": amp,
        "shared_residual_channels": shared,
        "shared_eval_q3": shared_eval,
        "tensor_summaries": summaries,
        "cheapest_0.995": cheapest_0995,
        "write_error_amp_weighted": write_q,
        "projection": projection,
        "beat_uniform": beat_uniform,
        "product_projection": prod_proj,
        "kernel": kernel,
        "tensors": tensors,
    }
    # numpy types → python
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        raise TypeError(type(o))

    raw = json.dumps(payload, default=conv)
    OUT.write_text(raw)
    log(f"WROTE {OUT} bytes={len(raw)} wall={payload['wall_s']:.1f}s rss_max={payload['rss_max_gb']:.3f}G")


if __name__ == "__main__":
    main()

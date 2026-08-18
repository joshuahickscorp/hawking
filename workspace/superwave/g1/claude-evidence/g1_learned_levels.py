#!/usr/bin/env python3
"""Activation-conditioned level-placement search for Qwen3.8.

CPU / numpy only. No GPU, no generate, no pack, no resident touch.
Writes /tmp/g1-learned-levels/*.json and a running log.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import struct
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

SRC = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16")
CAP = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1")
OUT = Path("/tmp/g1-learned-levels")
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN = 5120
INTERMEDIATE = 17408
N_TOK = 256
FIT_N = 192
HOLD_N = 64
G64 = 64
G128 = 128

N_MODEL = 26_895_998_464
E_MLP = 17_112_760_320
E_ATTN = 7_237_795_840
E_TAB = 2_542_796_800
E_SMALL = 2_645_504
B_TAB_Q4 = 4.250000251691366
B_SMALL_F32 = 32.00853977162764

KEY_HEADS = 16
VALUES_PER_KEY = 3
KEY_DIM = 128
VALUE_DIM = 128
GQA_HEADS = 24
GQA_KV = 4
GQA_HEAD_DIM = 256

SCALE_MULTS = np.array([0.50, 0.65, 0.75, 0.82, 0.88, 0.92, 0.96, 1.00, 1.15, 1.30], dtype=np.float32)
MU_GRID = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float32)
POW_GRID = np.array([0.45, 0.60, 0.75, 1.00, 1.35, 1.80, 2.40], dtype=np.float32)
CHUNK = 16384


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


_LOG_FH = open(OUT / "run.log", "a")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] rss={rss_gb():.3f}G {msg}"
    print(line, flush=True)
    _LOG_FH.write(line + "\n")
    _LOG_FH.flush()


def snap_f16(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def flat_cosine(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(left @ right)
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    if den <= 1e-12:
        return 1.0 if num == 0.0 else 0.0
    return num / den


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    ref = np.asarray(a, dtype=np.float64).reshape(-1)
    hat = np.asarray(b, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(ref))
    if nrm <= 1e-12:
        return 0.0
    return float(np.linalg.norm(ref - hat) / nrm)


def mean_row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.ascontiguousarray(a, dtype=np.float64)
    b64 = np.ascontiguousarray(b, dtype=np.float64)
    if a64.ndim == 1:
        a64 = a64[None, :]
        b64 = b64[None, :]
    num = np.sum(a64 * b64, axis=1)
    den = np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1)
    ok = den > 1e-12
    if not np.any(ok):
        return 0.0
    return float(np.mean(num[ok] / den[ok]))


def silu(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.ascontiguousarray(x, dtype=np.float32), -80.0, 80.0)))


# ---------------------------------------------------------------------------
# I/O
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
    if raw.size != N_TOK * HIDDEN:
        raise RuntimeError(f"hidden L{layer} size {raw.size}")
    return np.ascontiguousarray(raw.reshape(N_TOK, HIDDEN))


def tname(layer: int, suffix: str) -> str:
    return f"language_model.model.layers.{layer}.{suffix}"


def is_gqa(layer: int) -> bool:
    return (layer + 1) % 4 == 0


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
    return np.ascontiguousarray((v_rep * gate).reshape(X.shape[0], GQA_HEADS * GQA_HEAD_DIM), dtype=np.float32)


def down_x(Xh: np.ndarray, layer: int) -> np.ndarray:
    Wg = load_tensor(tname(layer, "mlp.gate_proj.weight"))
    g = Xh @ Wg.T
    del Wg
    Wu = load_tensor(tname(layer, "mlp.up_proj.weight"))
    u = Xh @ Wu.T
    del Wu
    return np.ascontiguousarray(silu(g) * u, dtype=np.float32)


def mixer_x(Xh: np.ndarray, layer: int) -> np.ndarray:
    if is_gqa(layer):
        Wq = load_tensor(tname(layer, "self_attn.q_proj.weight"))
        Wv = load_tensor(tname(layer, "self_attn.v_proj.weight"))
        out = gqa_out_proxy(Xh, Wq, Wv)
        del Wq, Wv
        return out
    Wqkv = load_tensor(tname(layer, "linear_attn.in_proj_qkv.weight"))
    Wz = load_tensor(tname(layer, "linear_attn.in_proj_z.weight"))
    Wqkvz = fuse_qkvz(Wqkv, Wz)
    del Wqkv, Wz
    out = deltanet_out_proxy(Xh, Wqkvz)
    del Wqkvz
    return out


# ---------------------------------------------------------------------------
# grouping / assignment
# ---------------------------------------------------------------------------

def group_pad(W: np.ndarray, g: int) -> tuple[np.ndarray, int]:
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    n = int(flat.size)
    groups = (n + g - 1) // g
    padded = np.zeros((groups, g), dtype=np.float32)
    padded.reshape(-1)[:n] = flat
    return padded, n


def col_energy_for_groups(energy: np.ndarray, n_out: int, n_in: int, g: int) -> np.ndarray:
    """Per-element activation energy tiled to (n_groups, g) in C-order of W[out,in]."""
    if n_in % g != 0:
        raise RuntimeError(f"n_in={n_in} not divisible by g={g}")
    n_blocks = n_in // g
    e = energy.astype(np.float64).reshape(n_blocks, g)
    tiled = np.broadcast_to(e, (n_out, n_blocks, g)).reshape(-1, g)
    return np.ascontiguousarray(tiled, dtype=np.float64)


def assign_levels_chunked(padded: np.ndarray, levels: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """recon[i,j] = levels[q] * scale[i], q = argmin_k (w - levels[k]*s)^2."""
    ng, gsz = padded.shape
    k = int(levels.size)
    recon = np.empty_like(padded)
    lev = np.ascontiguousarray(levels, dtype=np.float32)
    for s0 in range(0, ng, CHUNK):
        s1 = min(ng, s0 + CHUNK)
        w = padded[s0:s1]
        s = scale[s0:s1]
        cand = lev[None, None, :] * s[:, None, None]
        err = np.square(w[:, :, None] - cand)
        q = err.argmin(axis=2)
        recon[s0:s1] = lev[q] * s[:, None]
    return recon


def kmeans_1d(x: np.ndarray, k: int, weights: np.ndarray | None = None, n_iter: int = 12) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros(k, dtype=np.float32)
    qs = np.linspace(0.0, 100.0, k)
    c = np.percentile(x, qs).astype(np.float64)
    w = None if weights is None else np.ascontiguousarray(weights, dtype=np.float64).reshape(-1)
    for _ in range(n_iter):
        if k == 1:
            if w is None:
                c[0] = float(np.mean(x))
            else:
                c[0] = float(np.dot(x, w) / max(float(np.sum(w)), 1e-20))
            break
        edges = 0.5 * (c[:-1] + c[1:])
        q = np.searchsorted(edges, x)
        for i in range(k):
            m = q == i
            if not np.any(m):
                continue
            if w is None:
                c[i] = float(np.mean(x[m]))
            else:
                ww = w[m]
                sw = float(np.sum(ww))
                if sw > 0:
                    c[i] = float(np.dot(x[m], ww) / sw)
        c.sort()
    return c.astype(np.float32)


def lloyd_groups(
    padded: np.ndarray,
    k: int,
    weights: np.ndarray | None = None,
    n_iter: int = 4,
) -> np.ndarray:
    ng, gsz = padded.shape
    recon = np.empty_like(padded)
    for s0 in range(0, ng, CHUNK):
        s1 = min(ng, s0 + CHUNK)
        w = padded[s0:s1]
        lo = w.min(axis=1)
        hi = w.max(axis=1)
        t = np.linspace(0.0, 1.0, k, dtype=np.float32)
        c = lo[:, None] + (hi - lo)[:, None] * t[None, :]
        ww = None if weights is None else weights[s0:s1]
        for _ in range(n_iter):
            if k == 1:
                if ww is None:
                    c[:, 0] = w.mean(axis=1)
                else:
                    den = ww.sum(axis=1)
                    c[:, 0] = np.where(den > 0, (w * ww).sum(axis=1) / np.maximum(den, 1e-20), c[:, 0])
                break
            edges = 0.5 * (c[:, :-1] + c[:, 1:])
            q = np.sum(w[:, :, None] >= edges[:, None, :], axis=2)
            for ki in range(k):
                mask = q == ki
                if ww is None:
                    den = mask.sum(axis=1).astype(np.float32)
                    num = np.where(mask, w, 0.0).sum(axis=1)
                else:
                    mwt = np.where(mask, ww, 0.0)
                    den = mwt.sum(axis=1)
                    num = (mwt * w).sum(axis=1)
                c[:, ki] = np.where(den > 0, num / np.maximum(den, 1e-20), c[:, ki])
            c.sort(axis=1)
        if k == 1:
            recon[s0:s1] = c[:, 0:1]
        else:
            edges = 0.5 * (c[:, :-1] + c[:, 1:])
            q = np.sum(w[:, :, None] >= edges[:, None, :], axis=2)
            recon[s0:s1] = np.take_along_axis(c, q, axis=1)
    return recon


def best_scale_for_levels(
    padded: np.ndarray,
    levels: np.ndarray,
    amax: np.ndarray,
    wcost: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Search s = m * amax / max|L|. Minimize weighted SSE (or unweighted)."""
    lev_max = float(np.max(np.abs(levels))) if levels.size else 1.0
    denom = max(lev_max, 1e-12)
    best_cost = None
    best_s = None
    best_recon = None
    for m in SCALE_MULTS:
        s = snap_f16((amax * float(m)) / denom)
        recon = assign_levels_chunked(padded, levels, s)
        err = padded - recon
        if wcost is None:
            cost = float(np.sum(np.square(err, dtype=np.float64)))
        else:
            cost = float(np.sum(np.square(err, dtype=np.float64) * wcost))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_s = s
            best_recon = recon
    assert best_s is not None and best_recon is not None and best_cost is not None
    return best_recon, best_s, best_cost


def unpad(recon_g: np.ndarray, n: int, shape: tuple[int, ...]) -> np.ndarray:
    return recon_g.reshape(-1)[:n].reshape(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# codecs
# ---------------------------------------------------------------------------

def recon_uniform_clip(W: np.ndarray, bits: int, g: int) -> np.ndarray:
    """HGRAVU01: scale=amax/bound, q clipped to [-bound, bound]. Wastes 1 of 2^b codes."""
    if bits < 2:
        raise ValueError("uniform_clip bits>=2")
    bound = (1 << (bits - 1)) - 1
    padded, n = group_pad(W, g)
    amax = np.max(np.abs(padded), axis=1)
    scale = snap_f16(amax / max(bound, 1))
    den = np.where(scale > 0.0, scale, 1.0)
    q = np.rint(padded / den[:, None]).clip(-bound, bound)
    return unpad(q.astype(np.float32) * scale[:, None], n, W.shape)


def recon_uniform_asymm(W: np.ndarray, bits: int, g: int) -> np.ndarray:
    """Q4-style full grid: q in [qmin, qmax] = [-2^(b-1), 2^(b-1)-1]. Uses all codes."""
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    padded, n = group_pad(W, g)
    amax = np.max(np.abs(padded), axis=1)
    # scale so qmax * s covers +amax; negative has one extra step
    scale = snap_f16(amax / max(qmax, 1))
    den = np.where(scale > 0.0, scale, 1.0)
    q = np.rint(padded / den[:, None]).clip(qmin, qmax)
    return unpad(q.astype(np.float32) * scale[:, None], n, W.shape)


def recon_binary_meanabs(W: np.ndarray, g: int = G128) -> np.ndarray:
    padded, n = group_pad(W, g)
    scale = snap_f16(np.mean(np.abs(padded), axis=1, dtype=np.float64).astype(np.float32))
    signs = np.where(padded >= 0.0, 1.0, -1.0).astype(np.float32)
    return unpad(signs * scale[:, None], n, W.shape)


def recon_binary_absmax(W: np.ndarray, g: int = G64) -> np.ndarray:
    padded, n = group_pad(W, g)
    scale = snap_f16(np.max(np.abs(padded), axis=1))
    signs = np.where(padded >= 0.0, 1.0, -1.0).astype(np.float32)
    return unpad(signs * scale[:, None], n, W.shape)


def recon_onesided(W: np.ndarray, g: int, positive: bool) -> np.ndarray:
    padded, n = group_pad(W, g)
    if positive:
        body = np.maximum(padded, 0.0)
        scale = snap_f16(np.max(body, axis=1))
        q = (padded > 0.5 * scale[:, None]).astype(np.float32)
        return unpad(q * scale[:, None], n, W.shape)
    body = np.minimum(padded, 0.0)
    scale = snap_f16(-np.min(body, axis=1))
    q = (padded < -0.5 * scale[:, None]).astype(np.float32)
    return unpad(-q * scale[:, None], n, W.shape)


def recon_ternary_t07(W: np.ndarray, g: int = G128, tmult: float = 0.7) -> np.ndarray:
    padded, n = group_pad(W, g)
    base = np.mean(np.abs(padded), axis=1, dtype=np.float64).astype(np.float32)
    thr = snap_f16(base * float(tmult))
    active = np.abs(padded) >= thr[:, None]
    selected = np.where(active, np.abs(padded), 0.0)
    count = np.maximum(active.sum(axis=1), 1)
    scale = snap_f16((selected.sum(axis=1) / count).astype(np.float32))
    recon = np.where(~active, 0.0, np.where(padded >= 0.0, 1.0, -1.0)) * scale[:, None]
    return unpad(recon.astype(np.float32), n, W.shape)


def recon_ternary_sym(W: np.ndarray, g: int, scale_mode: str, wcost: np.ndarray | None) -> np.ndarray:
    padded, n = group_pad(W, g)
    amax = np.max(np.abs(padded), axis=1)
    levels = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    if scale_mode == "amax":
        s = snap_f16(amax)
        recon = assign_levels_chunked(padded, levels, s)
    else:
        recon, _, _ = best_scale_for_levels(padded, levels, amax, wcost)
    return unpad(recon, n, W.shape)


def recon_ternary_asymm_two_scale(W: np.ndarray, g: int, wcost: np.ndarray | None) -> np.ndarray:
    """{-a, 0, +b} with independent pos/neg scales. Same BPW as t0.7 (2 x f16 / group)."""
    padded, n = group_pad(W, g)
    pos = np.maximum(padded, 0.0)
    neg = np.maximum(-padded, 0.0)
    amax_p = np.max(pos, axis=1)
    amax_n = np.max(neg, axis=1)
    best = None
    best_recon = None
    for mp in (0.70, 0.85, 1.00):
        for mn in (0.70, 0.85, 1.00):
            sp = snap_f16(amax_p * mp)
            sn = snap_f16(amax_n * mn)
            # 3-way nearest: 0, +sp, -sn
            c0 = np.square(padded)
            cp = np.square(padded - sp[:, None])
            cn = np.square(padded + sn[:, None])
            recon = np.zeros_like(padded)
            use_p = (cp <= c0) & (cp <= cn)
            use_n = (cn < c0) & (cn < cp)
            recon = np.where(use_p, sp[:, None], recon)
            recon = np.where(use_n, -sn[:, None], recon)
            err = padded - recon
            if wcost is None:
                cost = float(np.sum(np.square(err, dtype=np.float64)))
            else:
                cost = float(np.sum(np.square(err, dtype=np.float64) * wcost))
            if best is None or cost < best:
                best = cost
                best_recon = recon
    assert best_recon is not None
    return unpad(best_recon, n, W.shape)


def recon_fixed_grid(W: np.ndarray, g: int, units: np.ndarray, wcost: np.ndarray | None) -> np.ndarray:
    """units are integer/float steps; scale searched against amax."""
    padded, n = group_pad(W, g)
    amax = np.max(np.abs(padded), axis=1)
    recon, _, _ = best_scale_for_levels(padded, units.astype(np.float32), amax, wcost)
    return unpad(recon, n, W.shape)


def recon_minmax_affine(W: np.ndarray, bits: int, g: int) -> np.ndarray:
    k = 1 << bits
    padded, n = group_pad(W, g)
    lo = padded.min(axis=1)
    hi = padded.max(axis=1)
    scale = snap_f16((hi - lo) / max(k - 1, 1))
    zp = snap_f16(lo)
    den = np.where(scale > 0.0, scale, 1.0)
    q = np.rint((padded - zp[:, None]) / den[:, None]).clip(0, k - 1)
    return unpad(zp[:, None] + q.astype(np.float32) * scale[:, None], n, W.shape)


def recon_mu_law(W: np.ndarray, bits: int, g: int, wcost: np.ndarray | None) -> tuple[np.ndarray, float]:
    """Companding: z=asinh(mu*w/s)/asinh(mu), uniform quantize, expand. Search mu on fit cost."""
    k = 1 << bits
    padded, n = group_pad(W, g)
    amax = np.max(np.abs(padded), axis=1)
    s = snap_f16(np.maximum(amax, 1e-12))
    wn = padded / s[:, None]
    best = None
    best_recon = None
    best_mu = None
    # mid-tread codes centered on 0
    u = (np.arange(k, dtype=np.float32) - 0.5 * (k - 1)) / (0.5 * (k - 1))
    for mu in MU_GRID:
        # reconstruction levels in [-1,1]
        levels = np.sinh(float(mu) * u) / math.sinh(float(mu))
        recon_n = assign_levels_chunked(wn.astype(np.float32), levels.astype(np.float32), np.ones(padded.shape[0], dtype=np.float32))
        recon = recon_n * s[:, None]
        err = padded - recon
        if wcost is None:
            cost = float(np.sum(np.square(err, dtype=np.float64)))
        else:
            cost = float(np.sum(np.square(err, dtype=np.float64) * wcost))
        if best is None or cost < best:
            best = cost
            best_recon = recon
            best_mu = float(mu)
    assert best_recon is not None and best_mu is not None
    return unpad(best_recon.astype(np.float32), n, W.shape), best_mu


def recon_power_spacing(W: np.ndarray, bits: int, g: int, wcost: np.ndarray | None) -> tuple[np.ndarray, float]:
    k = 1 << bits
    padded, n = group_pad(W, g)
    amax = np.max(np.abs(padded), axis=1)
    u = (np.arange(k, dtype=np.float32) - 0.5 * (k - 1)) / max(0.5 * (k - 1), 1e-6)
    best = None
    best_recon = None
    best_p = None
    for p in POW_GRID:
        levels = np.sign(u) * np.power(np.abs(u), float(p))
        recon, _, cost = best_scale_for_levels(padded, levels.astype(np.float32), amax, wcost)
        if best is None or cost < best:
            best = cost
            best_recon = recon
            best_p = float(p)
    assert best_recon is not None and best_p is not None
    return unpad(best_recon, n, W.shape), best_p


def recon_log_pow2(W: np.ndarray, bits: int, g: int, wcost: np.ndarray | None) -> np.ndarray:
    """Zero plus signed powers of two. K=2^bits. For bits=2: {0, ±s, ±2s} uses 5>4 so {0,+s,-s,+2s}."""
    k = 1 << bits
    if bits == 1:
        units = np.array([-1.0, 1.0], dtype=np.float32)
    elif bits == 2:
        units = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    else:
        # 0, ±1, ±2, ±4, and +8 if needed
        pos = [2.0**i for i in range(0, 4)]
        units_list = [0.0]
        for p in pos:
            units_list.append(p)
            units_list.append(-p)
        units = np.array(units_list[:k], dtype=np.float32)
        if units.size < k:
            units = np.concatenate([units, np.array([8.0] * (k - units.size), dtype=np.float32)])
    return recon_fixed_grid(W, g, units, wcost)


def sample_normed(
    padded: np.ndarray,
    amax: np.ndarray,
    wcost: np.ndarray | None,
    n_sample: int = 400_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Sample groups uniformly, keep per-element act-weights for 1-D k-means.

    Do not rng.choice over 89e6 elements with a p-vector. That stalls.
    Groups already cycle every input-column block, so a uniform group
    sample covers the column set. Activation weighting is applied in
    the k-means update, not in the group draw.
    """
    den = np.where(amax > 0.0, amax, 1.0)
    ng, gsz = padded.shape
    rng = rng or np.random.default_rng(0)
    n_keep = min(ng, max(2048, int(n_sample) // max(gsz, 1)))
    gi = rng.choice(ng, size=n_keep, replace=False)
    xs = (padded[gi] / den[gi, None]).reshape(-1).astype(np.float32)
    if wcost is None:
        return xs, None
    return xs, wcost[gi].reshape(-1).astype(np.float64)


def recon_learned_shared(
    W: np.ndarray,
    k: int,
    g: int,
    wcost: np.ndarray | None,
    pin_zero: bool = False,
    scale_search: bool = True,
) -> tuple[np.ndarray, list[float]]:
    padded, n = group_pad(W, g)
    amax = np.max(np.abs(padded), axis=1)
    xs, ws = sample_normed(padded, amax, wcost)
    levels = kmeans_1d(xs, k, weights=ws, n_iter=12)
    if pin_zero:
        # replace the level closest to 0 with exact 0, re-fit others around it
        levels = np.sort(levels)
        iz = int(np.argmin(np.abs(levels)))
        levels[iz] = 0.0
    if scale_search:
        recon, _, _ = best_scale_for_levels(padded, levels, amax, wcost)
    else:
        s = snap_f16(amax)
        recon = assign_levels_chunked(padded, levels, s)
    return unpad(recon, n, W.shape), [float(x) for x in levels]


def recon_lloyd(W: np.ndarray, k: int, g: int, wcost: np.ndarray | None) -> np.ndarray:
    padded, n = group_pad(W, g)
    recon = lloyd_groups(padded, k, weights=wcost, n_iter=4)
    return unpad(recon, n, W.shape)


# ---------------------------------------------------------------------------
# BPW
# ---------------------------------------------------------------------------

def bpw_codes(n: int, g: int, bits: float, n_f16_per_group: int, extra_f16: int = 0) -> float:
    groups = math.ceil(n / g)
    return (groups * (bits * g + 16.0 * n_f16_per_group) + 16.0 * extra_f16) / float(n)


def decode_class(name: str) -> str:
    if name.startswith("lloyd_"):
        return "TABLE"
    if name.startswith("learned_"):
        return "REGISTER_LUT"
    return "REGISTER"


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score(W: np.ndarray, Wh: np.ndarray, X_hold: np.ndarray | None, Y_hold: np.ndarray | None, R_hold: np.ndarray | None) -> dict:
    rec = {
        "weight_cosine": flat_cosine(W, Wh),
        "weight_rel_l2": rel_l2(W, Wh),
    }
    if X_hold is not None and Y_hold is not None:
        Yh = X_hold @ Wh.T
        rec["hold_output_cosine"] = flat_cosine(Y_hold, Yh)
        rec["hold_output_rel_l2"] = rel_l2(Y_hold, Yh)
        rec["hold_mean_row_cosine"] = mean_row_cosine(Y_hold, Yh)
        if R_hold is not None and Y_hold.shape[1] == R_hold.shape[1]:
            rec["residual_proxy_cosine"] = flat_cosine(R_hold + Y_hold, R_hold + Yh)
            rec["residual_proxy_rel_l2"] = rel_l2(R_hold + Y_hold, R_hold + Yh)
            rec["write_rms_over_r_rms"] = float(
                np.sqrt(np.mean(np.square(Y_hold, dtype=np.float64)))
                / max(np.sqrt(np.mean(np.square(R_hold, dtype=np.float64))), 1e-12)
            )
        del Yh
    return rec


# ---------------------------------------------------------------------------
# variant lists
# ---------------------------------------------------------------------------

def run_variants(
    W: np.ndarray,
    X_fit: np.ndarray | None,
    X_hold: np.ndarray | None,
    Y_hold: np.ndarray | None,
    R_hold: np.ndarray | None,
    *,
    want_lloyd: bool,
) -> list[dict]:
    n_out, n_in = W.shape
    n = int(W.size)
    rows: list[dict] = []

    def add(name: str, family: str, bits_nom: float, g: int, n_f16: int, extra_f16: int, Wh: np.ndarray, extra: dict | None = None) -> None:
        rec = {
            "name": name,
            "family": family,
            "bits_nominal": bits_nom,
            "group": g,
            "decode": decode_class(name),
            "complete_bpw": bpw_codes(n, g, bits_nom if name != "ternary_packed5" else 1.6, n_f16, extra_f16),
            "n_f16_per_group": n_f16,
            "extra_f16": extra_f16,
        }
        # packing alternatives for ternary
        if family == "ternary":
            rec["complete_bpw_2bit_store"] = bpw_codes(n, g, 2.0, n_f16, extra_f16)
            rec["complete_bpw_packed5"] = bpw_codes(n, g, 1.6, n_f16, extra_f16)
            rec["complete_bpw_shannon"] = bpw_codes(n, g, math.log2(3.0), n_f16, extra_f16)
        rec.update(score(W, Wh, X_hold, Y_hold, R_hold))
        if extra:
            rec.update(extra)
        rows.append(rec)
        log(f"    {name} cos={rec.get('hold_output_cosine'):.6f} bpw={rec['complete_bpw']:.4f} {rec['decode']}")

    wcost64 = None
    wcost128 = None
    if X_fit is not None and X_fit.shape[1] == n_in and n_in % 64 == 0:
        energy = np.sum(np.square(X_fit, dtype=np.float64), axis=0)
        wcost64 = col_energy_for_groups(energy, n_out, n_in, G64)
        if n_in % 128 == 0:
            wcost128 = col_energy_for_groups(energy, n_out, n_in, G128)

    # ----- 1-bit -----
    add("binary_meanabs_g128", "1bit", 1.0, G128, 1, 0, recon_binary_meanabs(W, G128))
    add("binary_absmax_g64", "1bit", 1.0, G64, 1, 0, recon_binary_absmax(W, G64))
    add("onesided_pos_g64", "1bit", 1.0, G64, 1, 0, recon_onesided(W, G64, True))
    add("onesided_neg_g64", "1bit", 1.0, G64, 1, 0, recon_onesided(W, G64, False))
    Wh, lev = recon_learned_shared(W, 2, G64, None, scale_search=True)
    add("learned_2_weight_g64", "1bit", 1.0, G64, 1, 2, Wh, {"levels_norm": lev})
    Wh, lev = recon_learned_shared(W, 2, G64, wcost64, scale_search=True)
    add("learned_2_act_g64", "1bit", 1.0, G64, 1, 2, Wh, {"levels_norm": lev})
    if want_lloyd:
        add("lloyd_2_act_g64", "1bit", 1.0, G64, 2, 0, recon_lloyd(W, 2, G64, wcost64))

    # ----- ternary -----
    add("ternary_t0.7_g128", "ternary", 2.0, G128, 2, 0, recon_ternary_t07(W, G128, 0.7))
    add("ternary_sym_amax_g64", "ternary", 1.584962500721156, G64, 1, 0, recon_ternary_sym(W, G64, "amax", wcost64))
    add("ternary_sym_mse_g64", "ternary", 1.584962500721156, G64, 1, 0, recon_ternary_sym(W, G64, "mse", None))
    add("ternary_sym_act_g64", "ternary", 1.584962500721156, G64, 1, 0, recon_ternary_sym(W, G64, "act", wcost64))
    add("ternary_asymm_2s_g64", "ternary", 2.0, G64, 2, 0, recon_ternary_asymm_two_scale(W, G64, wcost64))
    Wh, lev = recon_learned_shared(W, 3, G64, None, scale_search=True)
    add("learned_3_weight_g64", "ternary", 1.584962500721156, G64, 1, 3, Wh, {"levels_norm": lev})
    Wh, lev = recon_learned_shared(W, 3, G64, wcost64, scale_search=True)
    add("learned_3_act_g64", "ternary", 1.584962500721156, G64, 1, 3, Wh, {"levels_norm": lev})
    Wh, lev = recon_learned_shared(W, 3, G64, wcost64, pin_zero=True, scale_search=True)
    add("learned_3_act_pin0_g64", "ternary", 1.584962500721156, G64, 1, 3, Wh, {"levels_norm": lev})
    if want_lloyd:
        add("lloyd_3_act_g64", "ternary", 1.584962500721156, G64, 3, 0, recon_lloyd(W, 3, G64, wcost64))

    # ----- 2-bit -----
    add("uniform_q2_clip_g64", "2bit", 2.0, G64, 1, 0, recon_uniform_clip(W, 2, G64))
    add("uniform_q2_asymm_g64", "2bit", 2.0, G64, 1, 0, recon_uniform_asymm(W, 2, G64))
    add("zh_m3m101_g64", "2bit", 2.0, G64, 1, 0, recon_fixed_grid(W, G64, np.array([-3, -1, 0, 1], dtype=np.float32), wcost64))
    add("zh_m1013_g64", "2bit", 2.0, G64, 1, 0, recon_fixed_grid(W, G64, np.array([-1, 0, 1, 3], dtype=np.float32), wcost64))
    add("zh_m2012_g64", "2bit", 2.0, G64, 1, 0, recon_fixed_grid(W, G64, np.array([-2, 0, 1, 2], dtype=np.float32), wcost64))
    add("pow2_m1012_g64", "2bit", 2.0, G64, 1, 0, recon_log_pow2(W, 2, G64, wcost64))
    Wh, mu = recon_mu_law(W, 2, G64, wcost64)
    add("mulaw_q2_g64", "2bit", 2.0, G64, 1, 0, Wh, {"mu": mu})
    Wh, p = recon_power_spacing(W, 2, G64, wcost64)
    add("power_q2_g64", "2bit", 2.0, G64, 1, 0, Wh, {"p": p})
    add("minmax_q2_g64", "2bit", 2.0, G64, 2, 0, recon_minmax_affine(W, 2, G64))
    Wh, lev = recon_learned_shared(W, 4, G64, None, scale_search=True)
    add("learned_4_weight_g64", "2bit", 2.0, G64, 1, 4, Wh, {"levels_norm": lev})
    Wh, lev = recon_learned_shared(W, 4, G64, wcost64, scale_search=True)
    add("learned_4_act_g64", "2bit", 2.0, G64, 1, 4, Wh, {"levels_norm": lev})
    if want_lloyd:
        add("lloyd_4_act_g64", "2bit", 2.0, G64, 4, 0, recon_lloyd(W, 4, G64, wcost64))
        add("lloyd_4_weight_g64", "2bit", 2.0, G64, 4, 0, recon_lloyd(W, 4, G64, None))

    # ----- 3-bit -----
    add("uniform_q3_clip_g64", "3bit", 3.0, G64, 1, 0, recon_uniform_clip(W, 3, G64))
    add("uniform_q3_asymm_g64", "3bit", 3.0, G64, 1, 0, recon_uniform_asymm(W, 3, G64))
    add("zh_pow2_q3_g64", "3bit", 3.0, G64, 1, 0, recon_log_pow2(W, 3, G64, wcost64))
    Wh, mu = recon_mu_law(W, 3, G64, wcost64)
    add("mulaw_q3_g64", "3bit", 3.0, G64, 1, 0, Wh, {"mu": mu})
    Wh, p = recon_power_spacing(W, 3, G64, wcost64)
    add("power_q3_g64", "3bit", 3.0, G64, 1, 0, Wh, {"p": p})
    add("minmax_q3_g64", "3bit", 3.0, G64, 2, 0, recon_minmax_affine(W, 3, G64))
    # zero-heavy 3-bit register set: denser near 0
    add(
        "zh_dense0_q3_g64",
        "3bit",
        3.0,
        G64,
        1,
        0,
        recon_fixed_grid(W, G64, np.array([-8, -3, -1, 0, 1, 2, 4, 8], dtype=np.float32), wcost64),
    )
    Wh, lev = recon_learned_shared(W, 8, G64, None, scale_search=True)
    add("learned_8_weight_g64", "3bit", 3.0, G64, 1, 8, Wh, {"levels_norm": lev})
    Wh, lev = recon_learned_shared(W, 8, G64, wcost64, scale_search=True)
    add("learned_8_act_g64", "3bit", 3.0, G64, 1, 8, Wh, {"levels_norm": lev})
    # parametric fit of the act-learned levels
    if lev:
        u = (np.arange(8, dtype=np.float32) - 3.5) / 3.5
        tgt = np.array(lev, dtype=np.float64)
        tgt = tgt / max(float(np.max(np.abs(tgt))), 1e-12)
        best_mu_err = 1e9
        best_mu = 1.0
        for mu in MU_GRID:
            pred = np.sinh(float(mu) * u) / math.sinh(float(mu))
            err = float(np.max(np.abs(pred - tgt)))
            if err < best_mu_err:
                best_mu_err = err
                best_mu = float(mu)
        levels_mu = (np.sinh(best_mu * u) / math.sinh(best_mu)).astype(np.float32)
        padded, nn = group_pad(W, G64)
        amax = np.max(np.abs(padded), axis=1)
        recon, _, _ = best_scale_for_levels(padded, levels_mu, amax, wcost64)
        add(
            "mulaw_fit_learned8_g64",
            "3bit",
            3.0,
            G64,
            1,
            0,
            unpad(recon, nn, W.shape),
            {"mu": best_mu, "max_abs_level_err_vs_learned": best_mu_err, "learned_levels": lev},
        )
    if want_lloyd:
        add("lloyd_8_act_g64", "3bit", 3.0, G64, 8, 0, recon_lloyd(W, 8, G64, wcost64))
        add("lloyd_8_weight_g64", "3bit", 3.0, G64, 8, 0, recon_lloyd(W, 8, G64, None))

    # g=128 check on the two 3-bit leaders (uniform + learned act)
    add("uniform_q3_clip_g128", "3bit", 3.0, G128, 1, 0, recon_uniform_clip(W, 3, G128))
    Wh, lev = recon_learned_shared(W, 8, G128, wcost128, scale_search=True)
    add("learned_8_act_g128", "3bit", 3.0, G128, 1, 8, Wh, {"levels_norm": lev})

    return rows


def weight_stats(W: np.ndarray) -> dict:
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    # sample for percentiles on huge tensors
    if flat.size > 2_000_000:
        rng = np.random.default_rng(0)
        samp = flat[rng.choice(flat.size, size=2_000_000, replace=False)]
    else:
        samp = flat
    abs_s = np.abs(samp)
    med = float(np.median(abs_s))
    return {
        "shape": [int(x) for x in W.shape],
        "elements": int(flat.size),
        "mean": float(np.mean(samp)),
        "mean_abs": float(np.mean(abs_s)),
        "median_abs": med,
        "p99_abs": float(np.percentile(abs_s, 99)),
        "p999_abs": float(np.percentile(abs_s, 99.9)),
        "max_abs": float(np.max(abs_s)),
        "frac_abs_lt_1e-3": float(np.mean(abs_s < 1e-3)),
        "frac_abs_lt_median": float(np.mean(abs_s < med)) if med > 0 else 0.0,
        "sign_frac_pos": float(np.mean(samp > 0)),
        "skew": float(np.mean(((samp - np.mean(samp)) / (np.std(samp) + 1e-12)) ** 3)),
    }


# ---------------------------------------------------------------------------
# tensor schedule
# ---------------------------------------------------------------------------

PHASE1 = [
    (0, "mlp.gate_proj.weight", "hidden", True),
    (0, "mlp.up_proj.weight", "hidden", False),
    (0, "mlp.down_proj.weight", "swiglu", True),
    (0, "linear_attn.in_proj_qkv.weight", "hidden", False),
    (0, "linear_attn.out_proj.weight", "mixer", True),
    (3, "self_attn.q_proj.weight", "hidden", False),
    (3, "self_attn.o_proj.weight", "mixer", True),
    (31, "mlp.up_proj.weight", "hidden", False),
    (31, "mlp.down_proj.weight", "swiglu", True),
    (54, "mlp.down_proj.weight", "swiglu", False),
    (58, "mlp.down_proj.weight", "swiglu", True),
    (62, "mlp.down_proj.weight", "swiglu", True),
    (63, "mlp.gate_proj.weight", "hidden", False),
    (63, "mlp.down_proj.weight", "swiglu", True),
    (63, "self_attn.o_proj.weight", "mixer", False),
]


def load_site_x(layer: int, site: str, Xh: np.ndarray) -> np.ndarray:
    if site == "hidden":
        return Xh
    if site == "swiglu":
        return down_x(Xh, layer)
    if site == "mixer":
        return mixer_x(Xh, layer)
    raise RuntimeError(site)


def calib() -> dict:
    log("CALIB L0 gate/down Q3 + binary last-64 flattened")
    Xh = load_hidden(0)
    X_hold = Xh[FIT_N:]
    Wg = load_tensor(tname(0, "mlp.gate_proj.weight"))
    Yg = X_hold @ Wg.T
    Wh = recon_uniform_clip(Wg, 3, G64)
    Yh = X_hold @ Wh.T
    gate_q3 = flat_cosine(Yg, Yh)
    Whb = recon_binary_meanabs(Wg, G128)
    Yhb = X_hold @ Whb.T
    gate_bin = flat_cosine(Yg, Yhb)
    del Wg, Wh, Whb, Yg, Yh, Yhb
    Xd = down_x(Xh, 0)
    Xd_hold = Xd[FIT_N:]
    Wd = load_tensor(tname(0, "mlp.down_proj.weight"))
    Yd = Xd_hold @ Wd.T
    Whd = recon_uniform_clip(Wd, 3, G64)
    Yhd = Xd_hold @ Whd.T
    down_q3 = flat_cosine(Yd, Yhd)
    del Wd, Whd, Yd, Yhd, Xd, Xh
    rec = {
        "L0_gate_q3_hold_flat_cosine": gate_q3,
        "L0_gate_q3_target": 0.982098354690,
        "L0_gate_q3_abs_err": abs(gate_q3 - 0.982098354690),
        "L0_gate_binary_hold_flat_cosine": gate_bin,
        "L0_gate_binary_target": 0.861852194430,
        "L0_gate_binary_abs_err": abs(gate_bin - 0.861852194430),
        "L0_down_q3_hold_flat_cosine": down_q3,
        "L0_down_q3_target": 0.992290431331,
        "L0_down_q3_abs_err": abs(down_q3 - 0.992290431331),
        "split": "last64_of_256",
        "metric": "flattened_output_cosine",
    }
    log(f"CALIB {json.dumps(rec)}")
    return rec


def amplification() -> dict:
    log("AMPLIFICATION from captured post-norm hiddens + write gains")
    rms = []
    ratios = []
    h_prev = None
    per = []
    for L in range(64):
        h = load_hidden(L)
        r = float(np.sqrt(np.mean(np.square(h, dtype=np.float64))))
        rms.append(r)
        token_rms = np.sqrt(np.mean(np.square(h, dtype=np.float64), axis=1))
        rec = {"layer": L, "hidden_rms": r, "hidden_mean_token_rms": float(np.mean(token_rms))}
        if h_prev is not None:
            prev_tok = np.sqrt(np.mean(np.square(h_prev, dtype=np.float64), axis=1))
            ratio_tok = token_rms / np.maximum(prev_tok, 1e-12)
            rec["h_l_over_h_lm1_mean"] = float(np.mean(ratio_tok))
            rec["h_l_over_h_lm1_median"] = float(np.median(ratio_tok))
            rec["fro_ratio"] = float(np.linalg.norm(h) / max(np.linalg.norm(h_prev), 1e-12))
            ratios.append(rec["h_l_over_h_lm1_mean"])
        per.append(rec)
        h_prev = h
    # write gains on a layer grid
    write_layers = [0, 3, 15, 31, 47, 54, 58, 62, 63]
    writes = []
    for L in write_layers:
        Xh = load_hidden(L)
        Wd = load_tensor(tname(L, "mlp.down_proj.weight"))
        Xd = down_x(Xh, L)
        Yd = Xd @ Wd.T
        del Wd, Xd
        down_rms = float(np.sqrt(np.mean(np.square(Yd, dtype=np.float64))))
        h_rms = float(np.sqrt(np.mean(np.square(Xh, dtype=np.float64))))
        rec = {
            "layer": L,
            "down_write_rms": down_rms,
            "hidden_rms": h_rms,
            "down_over_hidden": down_rms / max(h_rms, 1e-12),
            "down_over_hidden_mean_token": float(
                np.mean(np.linalg.norm(Yd, axis=1) / np.maximum(np.linalg.norm(Xh, axis=1), 1e-12))
            ),
        }
        if is_gqa(L):
            Wo = load_tensor(tname(L, "self_attn.o_proj.weight"))
        else:
            Wo = load_tensor(tname(L, "linear_attn.out_proj.weight"))
        Xm = mixer_x(Xh, L)
        Yo = Xm @ Wo.T
        del Wo, Xm
        rec["attn_write_rms"] = float(np.sqrt(np.mean(np.square(Yo, dtype=np.float64))))
        rec["attn_over_hidden"] = rec["attn_write_rms"] / max(h_rms, 1e-12)
        comb = Yd + Yo
        rec["combined_write_rms"] = float(np.sqrt(np.mean(np.square(comb, dtype=np.float64))))
        rec["combined_over_hidden"] = rec["combined_write_rms"] / max(h_rms, 1e-12)
        rec["combined_over_hidden_mean_token"] = float(
            np.mean(np.linalg.norm(comb, axis=1) / np.maximum(np.linalg.norm(Xh, axis=1), 1e-12))
        )
        rec["gqa"] = is_gqa(L)
        writes.append(rec)
        del Yd, Yo, comb, Xh
        log(f"write-gain L{L} {rec['combined_over_hidden']:.4f} down={rec['down_over_hidden']:.4f} attn={rec['attn_over_hidden']:.4f}")
    out = {
        "hidden_rms": rms,
        "growth_mean": float(np.mean(ratios)),
        "growth_min": float(np.min(ratios)),
        "growth_max": float(np.max(ratios)),
        "L63_over_L0_rms": rms[63] / max(rms[0], 1e-12),
        "per_layer_growth": per,
        "write_gains": writes,
        "label": "PROXY_from_captured_post_norm_hidden_and_reconstructed_writes",
    }
    (OUT / "amplification.json").write_text(json.dumps(out, indent=2))
    log(f"AMPLIFICATION L63/L0 rms={out['L63_over_L0_rms']:.6f} mean_step={out['growth_mean']:.6f}")
    return out


def run_phase1() -> list[dict]:
    results = []
    for layer, suffix, site, want_lloyd in PHASE1:
        label = f"L{layer}.{suffix}"
        log(f"PHASE1 {label} site={site} lloyd={want_lloyd}")
        t0 = time.time()
        Xh = load_hidden(layer)
        X = load_site_x(layer, site, Xh)
        W = load_tensor(tname(layer, suffix))
        X_fit, X_hold = X[:FIT_N], X[FIT_N:]
        Y_hold = X_hold @ W.T
        R_hold = Xh[FIT_N:] if Y_hold.shape[1] == HIDDEN else None
        stats = weight_stats(W)
        rows = run_variants(W, X_fit, X_hold, Y_hold, R_hold, want_lloyd=want_lloyd)
        rec = {
            "label": label,
            "layer": layer,
            "suffix": suffix,
            "site": site,
            "shape": [int(x) for x in W.shape],
            "x_shape": [int(x) for x in X.shape],
            "rows_per_dim": float(X.shape[0]) / float(W.shape[1]),
            "weight_stats": stats,
            "variants": rows,
            "wall_s": time.time() - t0,
            "rss_gb": rss_gb(),
        }
        results.append(rec)
        path = OUT / f"phase1_{label.replace('.', '_')}.json"
        path.write_text(json.dumps(rec))
        # brief ranking
        by_fam: dict[str, list] = {}
        for r in rows:
            by_fam.setdefault(r["family"], []).append(r)
        for fam, vs in by_fam.items():
            vs_sorted = sorted(vs, key=lambda z: -z.get("hold_output_cosine", -1))
            top = vs_sorted[0]
            base = next((z for z in vs if "uniform" in z["name"] or z["name"].startswith("binary_meanabs") or z["name"] == "ternary_t0.7_g128"), vs[0])
            log(
                f"  {fam}: best={top['name']} cos={top.get('hold_output_cosine'):.6f} "
                f"bpw={top['complete_bpw']:.4f} dec={top['decode']} "
                f"vs {base['name']} {base.get('hold_output_cosine'):.6f}"
            )
        del W, X, Xh, Y_hold, R_hold
    (OUT / "phase1.json").write_text(json.dumps({"tensors": results}, indent=2))
    return results


def run_phase2() -> list[dict]:
    """All 64 downs + layer-grid gates at 2/3 bit: uniform_clip, best register, learned_act."""
    methods_3 = [
        ("uniform_q3_clip_g64", lambda W, wc: recon_uniform_clip(W, 3, G64), 3.0, G64, 1, 0),
        ("uniform_q3_asymm_g64", lambda W, wc: recon_uniform_asymm(W, 3, G64), 3.0, G64, 1, 0),
        ("mulaw_q3_g64", lambda W, wc: recon_mu_law(W, 3, G64, wc)[0], 3.0, G64, 1, 0),
        ("learned_8_act_g64", lambda W, wc: recon_learned_shared(W, 8, G64, wc, scale_search=True)[0], 3.0, G64, 1, 8),
    ]
    methods_2 = [
        ("uniform_q2_clip_g64", lambda W, wc: recon_uniform_clip(W, 2, G64), 2.0, G64, 1, 0),
        ("uniform_q2_asymm_g64", lambda W, wc: recon_uniform_asymm(W, 2, G64), 2.0, G64, 1, 0),
        ("ternary_t0.7_g128", lambda W, wc: recon_ternary_t07(W, G128, 0.7), 2.0, G128, 2, 0),
        ("learned_4_act_g64", lambda W, wc: recon_learned_shared(W, 4, G64, wc, scale_search=True)[0], 2.0, G64, 1, 4),
        ("ternary_sym_act_g64", lambda W, wc: recon_ternary_sym(W, G64, "act", wc), 1.584962500721156, G64, 1, 0),
    ]
    out = []
    for layer in range(64):
        log(f"PHASE2 down L{layer}")
        t0 = time.time()
        Xh = load_hidden(layer)
        X = down_x(Xh, layer)
        W = load_tensor(tname(layer, "mlp.down_proj.weight"))
        X_fit, X_hold = X[:FIT_N], X[FIT_N:]
        Y_hold = X_hold @ W.T
        energy = np.sum(np.square(X_fit, dtype=np.float64), axis=0)
        wc = col_energy_for_groups(energy, W.shape[0], W.shape[1], G64)
        rec = {"layer": layer, "organ": "down_proj", "variants": []}
        for name, fn, bits, g, nf, ex in methods_3 + methods_2:
            Wh = fn(W, wc)
            row = {
                "name": name,
                "family": "3bit" if bits >= 3 else ("ternary" if "ternary" in name else "2bit"),
                "decode": decode_class(name),
                "complete_bpw": bpw_codes(W.size, g, bits, nf, ex),
            }
            row.update(score(W, Wh, X_hold, Y_hold, Xh[FIT_N:]))
            rec["variants"].append(row)
            del Wh
        rec["wall_s"] = time.time() - t0
        rec["rss_gb"] = rss_gb()
        out.append(rec)
        # also a few gates
        if layer in (0, 6, 15, 22, 31, 47, 58, 63):
            log(f"PHASE2 gate L{layer}")
            Wg = load_tensor(tname(layer, "mlp.gate_proj.weight"))
            Xg_hold = Xh[FIT_N:]
            Yg = Xg_hold @ Wg.T
            energy_g = np.sum(np.square(Xh[:FIT_N], dtype=np.float64), axis=0)
            wcg = col_energy_for_groups(energy_g, Wg.shape[0], Wg.shape[1], G64)
            grec = {"layer": layer, "organ": "gate_proj", "variants": []}
            for name, fn, bits, g, nf, ex in methods_3 + methods_2:
                Wh = fn(Wg, wcg)
                row = {
                    "name": name,
                    "family": "3bit" if bits >= 3 else ("ternary" if "ternary" in name else "2bit"),
                    "decode": decode_class(name),
                    "complete_bpw": bpw_codes(Wg.size, g, bits, nf, ex),
                }
                row.update(score(Wg, Wh, Xg_hold, Yg, None))
                grec["variants"].append(row)
                del Wh
            out.append(grec)
            del Wg, Yg
        del W, X, Xh, Y_hold
        (OUT / "phase2.json").write_text(json.dumps({"tensors": out}))
    return out


def main() -> None:
    t0 = time.time()
    log("START g1-learned-levels")
    cal = calib()
    if cal["L0_gate_q3_abs_err"] > 2e-5:
        raise RuntimeError(f"calibration failed gate q3 {cal}")
    if cal["L0_down_q3_abs_err"] > 2e-5:
        raise RuntimeError(f"calibration failed down q3 {cal}")
    amp = amplification()
    p1 = run_phase1()
    p2 = run_phase2()
    summary = {
        "schema": "hawking.g1.learned_levels.v1",
        "wall_s": time.time() - t0,
        "rss_max_gb": rss_gb(),
        "calibration": cal,
        "amplification_path": str(OUT / "amplification.json"),
        "phase1_n": len(p1),
        "phase2_n": len(p2),
        "fit_hold": {"fit": FIT_N, "hold": HOLD_N, "metric": "flattened_output_cosine_last64"},
        "claim_boundary": {
            "no_gpu": True,
            "no_generate": True,
            "no_pack": True,
            "capture_underdetermined": True,
            "rows_per_dim_gate_up": 256 / 5120,
            "rows_per_dim_down": 256 / 17408,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"DONE wall={summary['wall_s']:.1f}s rss={summary['rss_max_gb']:.3f}G")


if __name__ == "__main__":
    main()

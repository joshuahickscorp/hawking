#!/usr/bin/env python3
"""Invertible DOF search on Qwen3.8-27B. CPU/numpy only. No GPU, no generate, no pack.

Enumerates exact pack-time maps, kills the ones the architecture forbids, and
searches the surviving family for a compressibility win at fixed absmax-Qn width.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import struct
import time
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, "/tmp")

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")

SRC = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16")
CAP = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1")
OUT = Path("/tmp/g1_invertible_dof.json")
LOG = Path("/tmp/g1_invertible_dof.log")

HIDDEN = 5120
INTER = 17408
GQA_HEADS = 24
GQA_KV = 4
GQA_HEAD_DIM = 256
GQA_ROTARY = 64
G = 64
SOURCE_N = 26_895_998_464
RMS_EPS = 1.0e-6
ROPE_THETA = 10_000_000.0
S_CLIP = (1e-4, 1e4)

# ---------------------------------------------------------------------------
# io / codec (same family as /tmp/qwen38_out_proj_forensics.py)
# ---------------------------------------------------------------------------

_HEADER_CACHE: dict[Path, dict] = {}
_WMAP = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
_LOG_FH = None


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] rss={rss_gb():.3f}G {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    raw = np.fromfile(CAP / "hidden" / f"L{layer:02d}.f32", dtype="<f4")
    if raw.size != 256 * HIDDEN:
        raise RuntimeError(f"hidden L{layer} size {raw.size}")
    return np.ascontiguousarray(raw.reshape(256, HIDDEN))


def tname(layer: int, suffix: str) -> str:
    return f"language_model.model.layers.{layer}.{suffix}"


def is_gqa(layer: int) -> bool:
    return (layer + 1) % 4 == 0


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


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


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(a))
    return num / den if den > 1e-12 else num


def excess_kurtosis(flat: np.ndarray) -> float:
    x = np.asarray(flat, dtype=np.float64).reshape(-1)
    c = x - float(np.mean(x))
    m2 = float(np.mean(c * c))
    m4 = float(np.mean(c * c * c * c))
    return (m4 / (m2 * m2) - 3.0) if m2 > 0 else 0.0


def group_pad(W: np.ndarray, group_size: int) -> tuple[np.ndarray, int]:
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    n = int(flat.size)
    groups = (n + group_size - 1) // group_size
    padded = np.zeros((groups, group_size), dtype=np.float32)
    padded.reshape(-1)[:n] = flat
    return padded, n


def uniform_absmax_recon(W: np.ndarray, bits: int, group_size: int = 64) -> np.ndarray:
    bound = (1 << (bits - 1)) - 1
    padded, n = group_pad(W, group_size)
    absmax = np.max(np.abs(padded), axis=1)
    scale = absmax / max(bound, 1)
    den = np.where(scale > 0.0, scale, 1.0)
    codes = np.rint(padded / den[:, None]).clip(-bound, bound)
    recon = (codes * scale[:, None]).reshape(-1)[:n]
    return recon.reshape(W.shape).astype(np.float32)


def group_util(W: np.ndarray, group_size: int = 64) -> dict:
    padded, _ = group_pad(W, group_size)
    abs_g = np.abs(padded)
    absmax = np.max(abs_g, axis=1)
    med = np.median(abs_g, axis=1)
    nz = absmax > 0
    ratio = np.zeros_like(absmax)
    ratio[nz] = med[nz] / absmax[nz]
    dominated = (absmax > 8.0 * np.maximum(med, 1e-20)) & nz
    return {
        "n_groups": int(padded.shape[0]),
        "mean_med_over_absmax": float(np.mean(ratio[nz])) if np.any(nz) else 0.0,
        "median_med_over_absmax": float(np.median(ratio[nz])) if np.any(nz) else 0.0,
        "frac_groups_absmax_gt_8x_median": float(np.mean(dominated)),
        "mean_absmax": float(np.mean(absmax[nz])) if np.any(nz) else 0.0,
    }


def weight_cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return num / den if den > 0 else 0.0


def gemv(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Y = X @ W.T  with W [out, in], X [n, in]."""
    return np.ascontiguousarray(X @ W.T, dtype=np.float32)


def hold_cosine(X: np.ndarray, W: np.ndarray, Wq: np.ndarray) -> float:
    return mean_row_cosine(gemv(X, W), gemv(X, Wq))


def normalize_s(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    s = np.clip(np.abs(s), S_CLIP[0], S_CLIP[1])
    g = float(np.exp(np.mean(np.log(s))))
    if g > 0:
        s = s / g
    return np.clip(s, S_CLIP[0], S_CLIP[1]).astype(np.float64)


def axis_absmax(W: np.ndarray, axis: int) -> np.ndarray:
    return np.max(np.abs(W), axis=axis).astype(np.float64)


def axis_rms(W: np.ndarray, axis: int) -> np.ndarray:
    return np.sqrt(np.mean(np.square(W, dtype=np.float64), axis=axis))


def pack_groups_greedy(scores: np.ndarray, g: int = 64) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    n = int(scores.size)
    n_g = n // g
    order = np.argsort(-scores)
    groups: list[list[int]] = [[] for _ in range(n_g)]
    gmax = np.zeros(n_g, dtype=np.float64)
    gcount = np.zeros(n_g, dtype=np.int32)
    for idx in order:
        space = np.flatnonzero(gcount < g)
        pick = int(space[np.argmin(gmax[space])])
        groups[pick].append(int(idx))
        gmax[pick] = max(gmax[pick], scores[idx])
        gcount[pick] += 1
    out = np.empty(n, dtype=np.int64)
    pos = 0
    for grp in groups:
        out[pos : pos + len(grp)] = np.asarray(grp, dtype=np.int64)
        pos += len(grp)
    return out


def interleave_high_low(scores: np.ndarray, g: int = 64) -> np.ndarray:
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    n = int(order.size)
    n_g = n // g
    out = np.empty(n, dtype=np.int64)
    hi = 0
    lo = n - 1
    for gi in range(n_g):
        take_hi = 1
        take_lo = g - take_hi
        sl = gi * g
        out[sl : sl + take_hi] = order[hi : hi + take_hi]
        hi += take_hi
        out[sl + take_hi : sl + g] = order[lo - take_lo + 1 : lo + 1]
        lo -= take_lo
    return out


def stride_round_robin(scores: np.ndarray, g: int = 64) -> np.ndarray:
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    n = int(order.size)
    n_g = n // g
    # place rank r into group r % n_g, slot r // n_g
    out = np.empty(n, dtype=np.int64)
    for r, idx in enumerate(order):
        gi = r % n_g
        slot = r // n_g
        out[gi * g + slot] = idx
    return out


# ---------------------------------------------------------------------------
# exactness
# ---------------------------------------------------------------------------

def qk_norm_rope_one(
    raw: np.ndarray, gamma: np.ndarray, slot: int, rotary: int = GQA_ROTARY
) -> np.ndarray:
    """Match qwen38_gqa_qk_norm_rope_cache_f32 for one head. raw, gamma [256]."""
    raw = np.asarray(raw, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    inv_rms = 1.0 / np.sqrt(np.mean(raw * raw) + RMS_EPS)
    normed = raw * inv_rms * gamma
    out = normed.copy()
    half = rotary // 2
    if rotary > 0:
        freq = np.arange(half, dtype=np.float64)
        inv_f = ROPE_THETA ** (-2.0 * freq / float(rotary))
        angle = float(slot) * inv_f
        c = np.cos(angle)
        s = np.sin(angle)
        a = normed[:half]
        b = normed[half:rotary]
        out[:half] = a * c - b * s
        out[half:rotary] = b * c + a * s
    return out.astype(np.float64)


def residual_rmsnorm(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    if x.ndim == 1:
        inv = 1.0 / np.sqrt(np.mean(x * x) + RMS_EPS)
        return (x * inv * gamma).astype(np.float32)
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + RMS_EPS)
    return (x * inv * gamma).astype(np.float32)


def maxabs(a: np.ndarray) -> float:
    return float(np.max(np.abs(a)))


def exactness_suite() -> dict:
    rng = np.random.default_rng(0)
    out: dict = {"synthetic": {}, "real_L0": {}, "real_L3": {}}

    # --- MLP diagonal exact, general M inexact, permutation exact ---
    n, d, m = 8, 32, 48
    X = rng.normal(size=(n, d)).astype(np.float32)
    Wg = rng.normal(size=(m, d)).astype(np.float32) * 0.05
    Wu = rng.normal(size=(m, d)).astype(np.float32) * 0.05
    Wd = rng.normal(size=(d, m)).astype(np.float32) * 0.05
    y0 = (silu(X @ Wg.T) * (X @ Wu.T)) @ Wd.T
    s = np.exp(rng.normal(size=m) * 0.5)
    Wu2 = (s[:, None] * Wu).astype(np.float32)
    Wd2 = (Wd / s[None, :]).astype(np.float32)
    y_diag = (silu(X @ Wg.T) * (X @ Wu2.T)) @ Wd2.T
    out["synthetic"]["mlp_diag_rel_l2"] = rel_l2(y0, y_diag)
    out["synthetic"]["mlp_diag_maxabs"] = maxabs(y0 - y_diag)

    P = rng.permutation(m)
    y_perm = (silu(X @ Wg[P].T) * (X @ Wu[P].T)) @ Wd[:, P].T
    out["synthetic"]["mlp_perm_rel_l2"] = rel_l2(y0, y_perm)

    M = rng.normal(size=(m, m)).astype(np.float64)
    M += 0.1 * np.eye(m)
    Minv = np.linalg.inv(M)
    WuM = (M @ Wu).astype(np.float32)
    WdM = (Wd @ Minv.T).astype(np.float32) if False else (Wd @ Minv).astype(np.float32)
    y_M = (silu(X @ Wg.T) * (X @ WuM.T)) @ WdM.T
    out["synthetic"]["mlp_fullM_rel_l2"] = rel_l2(y0, y_M)
    out["synthetic"]["mlp_fullM_is_exact"] = bool(rel_l2(y0, y_M) < 1e-5)

    # gate diagonal is NOT exact (silu)
    sg = np.exp(rng.normal(size=m) * 0.5)
    Wg2 = (sg[:, None] * Wg).astype(np.float32)
    Wd_g = (Wd / sg[None, :]).astype(np.float32)
    y_g = (silu(X @ Wg2.T) * (X @ Wu.T)) @ Wd_g.T
    out["synthetic"]["mlp_gate_diag_rel_l2"] = rel_l2(y0, y_g)
    out["synthetic"]["mlp_gate_diag_is_exact"] = bool(rel_l2(y0, y_g) < 1e-5)

    # --- GQA v/o + sigmoid gate: diagonal exact, full M inexact, perm+gate exact ---
    hd, kv, qh, seq = 16, 2, 6, 5
    v = rng.normal(size=(seq, kv, hd)).astype(np.float64)
    gate = rng.normal(size=(seq, qh, hd)).astype(np.float64)
    attn = rng.random(size=(seq, qh, seq))
    attn = attn / attn.sum(axis=2, keepdims=True)
    # mix v over sequence per kv head, repeat to q heads
    mixed = np.einsum("sqt,tkd->sqkd", attn.reshape(seq, qh, seq), v)
    # simpler proxy matching production kernel: identity-attn then * sigmoid(gate)
    v_rep = np.repeat(v, qh // kv, axis=1)
    gated = v_rep * sigmoid(gate.astype(np.float32)).astype(np.float64)
    o = rng.normal(size=(d, qh * hd)).astype(np.float64) * 0.05
    y_vo = gated.reshape(seq, -1) @ o.T

    s_vo = np.exp(rng.normal(size=(kv, hd)) * 0.4)
    v2 = v * s_vo[None, :, :]
    s_rep = np.repeat(s_vo, qh // kv, axis=0).reshape(-1)
    o2 = o / s_rep[None, :]
    v2_rep = np.repeat(v2, qh // kv, axis=1)
    gated2 = v2_rep * sigmoid(gate.astype(np.float32)).astype(np.float64)
    y_vo_d = gated2.reshape(seq, -1) @ o2.T
    out["synthetic"]["vo_diag_rel_l2"] = rel_l2(y_vo, y_vo_d)

    Mhd = rng.normal(size=(hd, hd)) + 0.5 * np.eye(hd)
    Minv_hd = np.linalg.inv(Mhd)
    vM = np.einsum("skd,de->ske", v, Mhd)
    vM_rep = np.repeat(vM, qh // kv, axis=1)
    gatedM = vM_rep * sigmoid(gate.astype(np.float32)).astype(np.float64)
    # o absorbs Minv on each q-head block
    oM = o.copy()
    for h in range(qh):
        oM[:, h * hd : (h + 1) * hd] = o[:, h * hd : (h + 1) * hd] @ Minv_hd
    y_vo_M = gatedM.reshape(seq, -1) @ oM.T
    out["synthetic"]["vo_fullM_rel_l2"] = rel_l2(y_vo, y_vo_M)
    out["synthetic"]["vo_fullM_is_exact"] = bool(rel_l2(y_vo, y_vo_M) < 1e-5)

    Pdim = rng.permutation(hd)
    vP = v[:, :, Pdim]
    gateP = gate[:, :, Pdim]
    vP_rep = np.repeat(vP, qh // kv, axis=1)
    gatedP = vP_rep * sigmoid(gateP.astype(np.float32)).astype(np.float64)
    oP = np.empty_like(o)
    for h in range(qh):
        oP[:, h * hd : (h + 1) * hd] = o[:, h * hd : (h + 1) * hd][:, Pdim]
    y_vo_P = gatedP.reshape(seq, -1) @ oP.T
    out["synthetic"]["vo_perm_gate_rel_l2"] = rel_l2(y_vo, y_vo_P)

    # --- QK-norm + RoPE: general M inexact; signed perm of non-rotary + gamma perm exact ---
    q = rng.normal(size=256)
    k = rng.normal(size=256)
    qg = 0.8 + 0.4 * rng.random(256)
    kg = 0.8 + 0.4 * rng.random(256)
    slot = 7
    qh0 = qk_norm_rope_one(q, qg, slot)
    kh0 = qk_norm_rope_one(k, kg, slot)
    score0 = float(np.dot(qh0, kh0))

    M256 = rng.normal(size=(256, 256))
    M256 += np.eye(256)
    MinvT = np.linalg.inv(M256).T
    qhM = qk_norm_rope_one(M256 @ q, qg, slot)
    khM = qk_norm_rope_one(MinvT @ k, kg, slot)
    out["synthetic"]["qk_fullM_pre_norm_score_rel"] = abs(float(np.dot(qhM, khM)) - score0) / (
        abs(score0) + 1e-12
    )

    # orthogonal on all 256, gamma unmoved
    U, _ = np.linalg.qr(rng.normal(size=(256, 256)))
    qhU = qk_norm_rope_one(U @ q, qg, slot)
    khU = qk_norm_rope_one(U @ k, kg, slot)
    out["synthetic"]["qk_orth_pre_norm_score_rel"] = abs(float(np.dot(qhU, khU)) - score0) / (
        abs(score0) + 1e-12
    )

    # permute non-rotary dims + permute gamma
    nr = np.arange(GQA_ROTARY, 256)
    Pnr = rng.permutation(nr.size)
    q2 = q.copy()
    k2 = k.copy()
    qg2 = qg.copy()
    kg2 = kg.copy()
    q2[nr] = q[nr][Pnr]
    k2[nr] = k[nr][Pnr]
    qg2[nr] = qg[nr][Pnr]
    kg2[nr] = kg[nr][Pnr]
    qhP = qk_norm_rope_one(q2, qg2, slot)
    khP = qk_norm_rope_one(k2, kg2, slot)
    out["synthetic"]["qk_nr_perm_score_rel"] = abs(float(np.dot(qhP, khP)) - score0) / (
        abs(score0) + 1e-12
    )

    # isotropic per-head scale of raw q is eaten by RMSNorm
    qh_s = qk_norm_rope_one(3.7 * q, qg, slot)
    out["synthetic"]["qk_isotropic_scale_rel"] = rel_l2(qh0, qh_s)

    # --- norm-boundary diagonal exact ---
    x = rng.normal(size=(n, d)).astype(np.float64)
    gamma = 0.5 + rng.random(d)
    W = rng.normal(size=(m, d)).astype(np.float64) * 0.05
    y_n = residual_rmsnorm(x, gamma) @ W.T
    s_n = np.exp(rng.normal(size=d) * 0.4)
    gamma2 = gamma * s_n
    W2 = W / s_n[None, :]
    y_n2 = residual_rmsnorm(x, gamma2) @ W2.T
    out["synthetic"]["norm_boundary_rel_l2"] = rel_l2(y_n, y_n2)

    # residual-space orthogonal (skip-add identity) is NOT exact through RMSNorm+gamma
    U2, _ = np.linalg.qr(rng.normal(size=(d, d)))
    y_u = residual_rmsnorm(x @ U2.T, gamma) @ (W @ U2).T
    out["synthetic"]["resid_orth_rel_l2"] = rel_l2(y_n, y_u)
    out["synthetic"]["resid_orth_is_exact"] = bool(rel_l2(y_n, y_u) < 1e-5)

    # --- real L0 MLP diagonal ---
    log("exactness real L0 mlp")
    X0 = load_hidden(0)[:32]
    Wg = load_tensor(tname(0, "mlp.gate_proj.weight"))
    Wu = load_tensor(tname(0, "mlp.up_proj.weight"))
    Wd = load_tensor(tname(0, "mlp.down_proj.weight"))
    y0 = (silu(X0 @ Wg.T) * (X0 @ Wu.T)) @ Wd.T
    s = normalize_s(np.sqrt(axis_absmax(Wd, 0) / np.maximum(axis_absmax(Wu, 1), 1e-12)))
    Wu2 = (Wu * s[:, None]).astype(np.float32)
    Wd2 = (Wd / s[None, :]).astype(np.float32)
    y1 = (silu(X0 @ Wg.T) * (X0 @ Wu2.T)) @ Wd2.T
    out["real_L0"]["mlp_diag_rel_l2"] = rel_l2(y0, y1)
    out["real_L0"]["mlp_diag_maxabs"] = maxabs(y0 - y1)
    out["real_L0"]["mlp_diag_hold_unquant"] = mean_row_cosine(y0, y1)
    del Wg, Wu, Wd, Wu2, Wd2, y0, y1
    gc.collect()

    # --- real L3 v/o diagonal ---
    log("exactness real L3 vo")
    X3 = load_hidden(3)[:32]
    Wq = load_tensor(tname(3, "self_attn.q_proj.weight"))
    Wv = load_tensor(tname(3, "self_attn.v_proj.weight"))
    Wo = load_tensor(tname(3, "self_attn.o_proj.weight"))
    proxy0 = gqa_vo_proxy(X3, Wq, Wv)
    y0 = proxy0 @ Wo.T
    s_vo = vo_s_from_absmax(Wv, Wo)
    Wv2, Wo2 = apply_vo_s(Wv, Wo, s_vo)
    proxy1 = gqa_vo_proxy(X3, Wq, Wv2)
    y1 = proxy1 @ Wo2.T
    out["real_L3"]["vo_diag_rel_l2"] = rel_l2(y0, y1)
    out["real_L3"]["vo_diag_hold_unquant"] = mean_row_cosine(y0, y1)
    del Wq, Wv, Wo, Wv2, Wo2, proxy0, proxy1, y0, y1
    gc.collect()

    log(f"exactness done {json.dumps({k: out[k] for k in out})}")
    return out


def gqa_vo_proxy(X: np.ndarray, Wq: np.ndarray, Wv: np.ndarray) -> np.ndarray:
    qg = gemv(X, Wq).reshape(X.shape[0], GQA_HEADS, 2, GQA_HEAD_DIM)
    gate = sigmoid(qg[:, :, 1, :])
    v = gemv(X, Wv).reshape(X.shape[0], GQA_KV, GQA_HEAD_DIM)
    v_rep = np.repeat(v, GQA_HEADS // GQA_KV, axis=1)
    return np.ascontiguousarray((v_rep * gate).reshape(X.shape[0], GQA_HEADS * GQA_HEAD_DIM), dtype=np.float32)


def vo_s_from_absmax(Wv: np.ndarray, Wo: np.ndarray) -> np.ndarray:
    """s[kv, dim] balances v-row absmax against the 6 o-columns that consume it."""
    v = np.abs(Wv.reshape(GQA_KV, GQA_HEAD_DIM, -1))
    v_am = v.max(axis=2)  # [4, 256]
    o = np.abs(Wo.reshape(Wo.shape[0], GQA_HEADS, GQA_HEAD_DIM))
    # o columns for kv head h, dim d: heads 6h..6h+5, dim d
    o_am = np.zeros((GQA_KV, GQA_HEAD_DIM), dtype=np.float64)
    rep = GQA_HEADS // GQA_KV
    for h in range(GQA_KV):
        o_am[h] = o[:, h * rep : (h + 1) * rep, :].max(axis=(0, 1))
    s = np.sqrt(o_am / np.maximum(v_am, 1e-12))
    return normalize_s(s.reshape(-1)).reshape(GQA_KV, GQA_HEAD_DIM)


def apply_vo_s(Wv: np.ndarray, Wo: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Wv2 = Wv.reshape(GQA_KV, GQA_HEAD_DIM, -1) * s[:, :, None]
    Wo2 = Wo.copy()
    rep = GQA_HEADS // GQA_KV
    s_rep = np.repeat(s, rep, axis=0)  # [24, 256]
    Wo2 = Wo2.reshape(Wo.shape[0], GQA_HEADS, GQA_HEAD_DIM) / s_rep[None, :, :]
    return (
        np.ascontiguousarray(Wv2.reshape(Wv.shape), dtype=np.float32),
        np.ascontiguousarray(Wo2.reshape(Wo.shape), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# MLP site
# ---------------------------------------------------------------------------

def mlp_s_recipes(Wu: np.ndarray, Wd: np.ndarray, h_fit: np.ndarray | None) -> dict[str, np.ndarray]:
    up_row_am = np.maximum(axis_absmax(Wu, 1), 1e-12)
    dn_col_am = np.maximum(axis_absmax(Wd, 0), 1e-12)
    up_row_rms = np.maximum(axis_rms(Wu, 1), 1e-12)
    dn_col_rms = np.maximum(axis_rms(Wd, 0), 1e-12)
    rec: dict[str, np.ndarray] = {
        "identity": np.ones(INTER, dtype=np.float64),
        "absmax_a25": normalize_s((dn_col_am / up_row_am) ** 0.25),
        "absmax_a50": normalize_s((dn_col_am / up_row_am) ** 0.50),
        "absmax_a75": normalize_s((dn_col_am / up_row_am) ** 0.75),
        "absmax_a100": normalize_s(dn_col_am / up_row_am),
        "rms_a50": normalize_s((dn_col_rms / up_row_rms) ** 0.50),
        "col_equalize": normalize_s(dn_col_am),
        "col_equalize_floor": normalize_s(np.maximum(dn_col_am, np.percentile(dn_col_am, 10))),
    }
    if h_fit is not None:
        h_rms = np.maximum(axis_rms(h_fit, 0), 1e-12)
        rec["act_h_a50"] = normalize_s((h_rms / np.median(h_rms)) ** 0.50)
        rec["act_down_awq_a50"] = normalize_s((dn_col_am / h_rms) ** 0.50)
    return rec


def apply_mlp_s(Wu: np.ndarray, Wd: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.ascontiguousarray(Wu * s[:, None], dtype=np.float32),
        np.ascontiguousarray(Wd / s[None, :], dtype=np.float32),
    )


def apply_mlp_perm(Wg: np.ndarray, Wu: np.ndarray, Wd: np.ndarray, p: np.ndarray):
    return Wg[p], Wu[p], Wd[:, p]


def mlp_composition(X: np.ndarray, Wg, Wu, Wd) -> np.ndarray:
    return (silu(gemv(X, Wg)) * gemv(X, Wu)) @ Wd.T


def eval_mlp_pair(
    X_hold: np.ndarray,
    Wg: np.ndarray,
    Wu: np.ndarray,
    Wd: np.ndarray,
    bits: int,
    y_ref: np.ndarray | None = None,
) -> dict:
    Wg_q = uniform_absmax_recon(Wg, bits)
    Wu_q = uniform_absmax_recon(Wu, bits)
    Wd_q = uniform_absmax_recon(Wd, bits)
    y_q = mlp_composition(X_hold, Wg_q, Wu_q, Wd_q)
    if y_ref is None:
        y_ref = mlp_composition(X_hold, Wg, Wu, Wd)
    h = silu(gemv(X_hold, Wg)) * gemv(X_hold, Wu)
    h_q = silu(gemv(X_hold, Wg_q)) * gemv(X_hold, Wu_q)
    return {
        "bits": bits,
        "comp_hold": mean_row_cosine(y_ref, y_q),
        "comp_rel_l2": rel_l2(y_ref, y_q),
        "gate_hold": hold_cosine(X_hold, Wg, Wg_q),
        "up_hold": hold_cosine(X_hold, Wu, Wu_q),
        "down_hold": mean_row_cosine(h @ Wd.T, h @ Wd_q.T),
        "down_hold_hq": mean_row_cosine(h @ Wd.T, h_q @ Wd_q.T),
        "wcos_gate": weight_cos(Wg, Wg_q),
        "wcos_up": weight_cos(Wu, Wu_q),
        "wcos_down": weight_cos(Wd, Wd_q),
        "down_kurtosis": excess_kurtosis(Wd),
        "down_q_kurtosis": excess_kurtosis(Wd_q),
        "util_down": group_util(Wd),
        "util_up": group_util(Wu),
    }


def phase_mlp(layers: list[int], recipes_on: list[int], bits_list=(2, 3, 4)) -> dict:
    result: dict = {"layers": {}, "recipe_wins": {}}
    # recipe search
    vote: dict[str, list[float]] = {}
    for layer in recipes_on:
        log(f"mlp recipe search L{layer}")
        X = load_hidden(layer)
        even = np.arange(0, 256, 2)
        odd = np.arange(1, 256, 2)
        Wg = load_tensor(tname(layer, "mlp.gate_proj.weight"))
        Wu = load_tensor(tname(layer, "mlp.up_proj.weight"))
        Wd = load_tensor(tname(layer, "mlp.down_proj.weight"))
        h_fit = silu(gemv(X[even], Wg)) * gemv(X[even], Wu)
        y_ref = mlp_composition(X[odd], Wg, Wu, Wd)
        recs = mlp_s_recipes(Wu, Wd, h_fit)
        # permutation candidates (apply, then identity s)
        scores = axis_absmax(Wd, 0)
        recs_perm = {
            "perm_greedy": pack_groups_greedy(scores),
            "perm_interleave": interleave_high_low(scores),
            "perm_stride": stride_round_robin(scores),
        }
        layer_rec = {"diag": {}, "perm": {}}
        base = eval_mlp_pair(X[odd], Wg, Wu, Wd, 3, y_ref=y_ref)
        layer_rec["identity_q3"] = {k: base[k] for k in ("comp_hold", "down_hold", "wcos_down", "util_down")}
        best_name, best_delta = "identity", 0.0
        for name, s in recs.items():
            Wu2, Wd2 = apply_mlp_s(Wu, Wd, s)
            ev = eval_mlp_pair(X[odd], Wg, Wu2, Wd2, 3, y_ref=y_ref)
            ev["s_geomean"] = float(np.exp(np.mean(np.log(s))))
            ev["s_max"] = float(np.max(s))
            ev["s_min"] = float(np.min(s))
            ev["s_p99_over_p1"] = float(np.percentile(s, 99) / max(np.percentile(s, 1), 1e-12))
            layer_rec["diag"][name] = ev
            vote.setdefault(name, []).append(ev["comp_hold"] - base["comp_hold"])
            if ev["comp_hold"] - base["comp_hold"] > best_delta:
                best_delta = ev["comp_hold"] - base["comp_hold"]
                best_name = name
            del Wu2, Wd2
        for name, p in recs_perm.items():
            Wg2, Wu2, Wd2 = apply_mlp_perm(Wg, Wu, Wd, p)
            ev = eval_mlp_pair(X[odd], Wg2, Wu2, Wd2, 3, y_ref=y_ref)
            layer_rec["perm"][name] = ev
            vote.setdefault(name, []).append(ev["comp_hold"] - base["comp_hold"])
            del Wg2, Wu2, Wd2
        # perm_greedy + col_equalize
        p = recs_perm["perm_greedy"]
        Wg2, Wu2, Wd2 = apply_mlp_perm(Wg, Wu, Wd, p)
        s = normalize_s(axis_absmax(Wd2, 0))
        Wu3, Wd3 = apply_mlp_s(Wu2, Wd2, s)
        ev = eval_mlp_pair(X[odd], Wg2, Wu3, Wd3, 3, y_ref=y_ref)
        layer_rec["perm"]["perm_greedy+col_equalize"] = ev
        vote.setdefault("perm_greedy+col_equalize", []).append(ev["comp_hold"] - base["comp_hold"])
        del Wg2, Wu2, Wd2, Wu3, Wd3, Wg, Wu, Wd, X, h_fit, y_ref
        gc.collect()
        result["layers"][str(layer)] = {"recipe_search": layer_rec, "best_diag": best_name}
        log(
            f"L{layer} id_q3={base['comp_hold']:.6f} best_diag={best_name} "
            f"d={best_delta:+.6f} perm_greedy={layer_rec['perm']['perm_greedy']['comp_hold']:.6f}"
        )

    result["recipe_mean_delta_q3"] = {k: float(np.mean(v)) for k, v in vote.items()}
    # pick best weight-only diag (exclude act_*)
    wo = {k: v for k, v in result["recipe_mean_delta_q3"].items() if not k.startswith("act_")}
    winner = max(wo, key=wo.get) if wo else "identity"
    result["winner_weight_only"] = winner
    result["winner_mean_delta_q3"] = wo.get(winner, 0.0)
    log(f"mlp winner={winner} mean_dQ3={result['winner_mean_delta_q3']:+.6f} all={result['recipe_mean_delta_q3']}")

    # apply winner + identity across the full layer list at Q2/Q3/Q4
    for layer in layers:
        log(f"mlp sweep L{layer}")
        X = load_hidden(layer)
        odd = np.arange(1, 256, 2)
        Wg = load_tensor(tname(layer, "mlp.gate_proj.weight"))
        Wu = load_tensor(tname(layer, "mlp.up_proj.weight"))
        Wd = load_tensor(tname(layer, "mlp.down_proj.weight"))
        y_ref = mlp_composition(X[odd], Wg, Wu, Wd)
        even = np.arange(0, 256, 2)
        h_fit = silu(gemv(X[even], Wg)) * gemv(X[even], Wu)
        recs = mlp_s_recipes(Wu, Wd, h_fit)
        # reconstruct winner transform
        def transform(name: str):
            if name == "identity":
                return Wg, Wu, Wd
            if name.startswith("perm_greedy+"):
                p = pack_groups_greedy(axis_absmax(Wd, 0))
                Wg2, Wu2, Wd2 = apply_mlp_perm(Wg, Wu, Wd, p)
                s = recs["col_equalize"] if "col_equalize" in name else recs.get("absmax_a50")
                # col_equalize must be recomputed in the permuted basis
                s = normalize_s(axis_absmax(Wd2, 0))
                Wu3, Wd3 = apply_mlp_s(Wu2, Wd2, s)
                return Wg2, Wu3, Wd3
            if name.startswith("perm_"):
                scores = axis_absmax(Wd, 0)
                if name == "perm_greedy":
                    p = pack_groups_greedy(scores)
                elif name == "perm_interleave":
                    p = interleave_high_low(scores)
                else:
                    p = stride_round_robin(scores)
                return apply_mlp_perm(Wg, Wu, Wd, p)
            s = recs[name]
            Wu2, Wd2 = apply_mlp_s(Wu, Wd, s)
            return Wg, Wu2, Wd2

        cell = result["layers"].setdefault(str(layer), {})
        cell["sweep"] = {}
        for name in ("identity", winner):
            Wg2, Wu2, Wd2 = transform(name)
            cell["sweep"][name] = {}
            for b in bits_list:
                cell["sweep"][name][f"q{b}"] = eval_mlp_pair(X[odd], Wg2, Wu2, Wd2, b, y_ref=y_ref)
            # diagnostics
            if name != "identity":
                cell["sweep"][name]["down_kurtosis"] = excess_kurtosis(Wd2)
                cell["sweep"][name]["down_row3994_rms_xmed"] = None
                row_rms = axis_rms(Wd2, 1)
                med = float(np.median(row_rms))
                if Wd2.shape[0] > 3994 and med > 0:
                    cell["sweep"][name]["down_row3994_rms_xmed"] = float(row_rms[3994] / med)
                    cell["sweep"][name]["down_kurtosis_drop3994"] = excess_kurtosis(np.delete(Wd2, 3994, axis=0))
            else:
                row_rms = axis_rms(Wd, 1)
                med = float(np.median(row_rms))
                cell["sweep"][name]["down_kurtosis"] = excess_kurtosis(Wd)
                if Wd.shape[0] > 3994 and med > 0:
                    cell["sweep"][name]["down_row3994_rms_xmed"] = float(row_rms[3994] / med)
                    cell["sweep"][name]["down_kurtosis_drop3994"] = excess_kurtosis(np.delete(Wd, 3994, axis=0))
            if name != "identity":
                del Wg2, Wu2, Wd2
        id3 = cell["sweep"]["identity"]["q3"]["comp_hold"]
        w3 = cell["sweep"][winner]["q3"]["comp_hold"]
        id4 = cell["sweep"]["identity"]["q4"]["comp_hold"]
        w4 = cell["sweep"][winner]["q4"]["comp_hold"]
        log(
            f"L{layer} MLP id Q2/3/4="
            f"{cell['sweep']['identity']['q2']['comp_hold']:.6f}/"
            f"{id3:.6f}/{id4:.6f}  {winner} "
            f"{cell['sweep'][winner]['q2']['comp_hold']:.6f}/{w3:.6f}/{w4:.6f}  "
            f"dQ3={w3-id3:+.6f} dQ4={w4-id4:+.6f} q3_vs_id_q4={w3-id4:+.6f}"
        )
        del Wg, Wu, Wd, X, y_ref, h_fit
        gc.collect()
    return result


# ---------------------------------------------------------------------------
# GQA v/o
# ---------------------------------------------------------------------------

def vo_recipes(Wv: np.ndarray, Wo: np.ndarray, X_fit: np.ndarray | None, Wq: np.ndarray | None) -> dict[str, np.ndarray]:
    v = Wv.reshape(GQA_KV, GQA_HEAD_DIM, -1)
    v_am = np.maximum(v.max(axis=2), 1e-12) if False else np.maximum(np.max(np.abs(v), axis=2), 1e-12)
    o = np.abs(Wo).reshape(Wo.shape[0], GQA_HEADS, GQA_HEAD_DIM)
    o_am = np.zeros((GQA_KV, GQA_HEAD_DIM), dtype=np.float64)
    rep = GQA_HEADS // GQA_KV
    for h in range(GQA_KV):
        o_am[h] = o[:, h * rep : (h + 1) * rep, :].max(axis=(0, 1))
        o_am[h] = np.maximum(o_am[h], 1e-12)
    rec = {
        "identity": np.ones((GQA_KV, GQA_HEAD_DIM), dtype=np.float64),
        "absmax_a50": normalize_s(np.sqrt(o_am / v_am).reshape(-1)).reshape(GQA_KV, GQA_HEAD_DIM),
        "absmax_a100": normalize_s((o_am / v_am).reshape(-1)).reshape(GQA_KV, GQA_HEAD_DIM),
        "col_equalize": normalize_s(o_am.reshape(-1)).reshape(GQA_KV, GQA_HEAD_DIM),
    }
    if X_fit is not None and Wq is not None:
        proxy = gqa_vo_proxy(X_fit, Wq, Wv)  # [n, 6144]
        pr = proxy.reshape(proxy.shape[0], GQA_HEADS, GQA_HEAD_DIM)
        # rms per kv-head dim: mean over the 6 repeats
        rms = np.zeros((GQA_KV, GQA_HEAD_DIM), dtype=np.float64)
        for h in range(GQA_KV):
            sl = pr[:, h * rep : (h + 1) * rep, :]
            rms[h] = np.sqrt(np.mean(np.square(sl, dtype=np.float64), axis=(0, 1)))
        rms = np.maximum(rms, 1e-12)
        rec["act_a50"] = normalize_s((rms / np.median(rms)).reshape(-1)).reshape(GQA_KV, GQA_HEAD_DIM)
    return rec


def eval_vo(
    X_hold: np.ndarray,
    Wq: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    bits: int,
    y_ref: np.ndarray | None,
    proxy_ref: np.ndarray | None,
) -> dict:
    Wq_q = uniform_absmax_recon(Wq, bits)
    Wv_q = uniform_absmax_recon(Wv, bits)
    Wo_q = uniform_absmax_recon(Wo, bits)
    proxy_q = gqa_vo_proxy(X_hold, Wq_q, Wv_q)
    y_q = proxy_q @ Wo_q.T
    if y_ref is None:
        proxy_ref = gqa_vo_proxy(X_hold, Wq, Wv)
        y_ref = proxy_ref @ Wo.T
    return {
        "bits": bits,
        "comp_hold": mean_row_cosine(y_ref, y_q),
        "comp_rel_l2": rel_l2(y_ref, y_q),
        "v_hold": hold_cosine(X_hold, Wv, Wv_q),
        "o_hold_exact_proxy": mean_row_cosine(proxy_ref @ Wo.T, proxy_ref @ Wo_q.T),
        "wcos_v": weight_cos(Wv, Wv_q),
        "wcos_o": weight_cos(Wo, Wo_q),
        "o_kurtosis": excess_kurtosis(Wo),
        "util_o": group_util(Wo),
        "util_v": group_util(Wv),
    }


def phase_vo(layers: list[int], bits_list=(2, 3, 4)) -> dict:
    result: dict = {"layers": {}, "recipe_mean_delta_q3": {}}
    vote: dict[str, list[float]] = {}
    search_layers = [L for L in layers if L in {3, 15, 31, 47, 63}]
    if not search_layers:
        search_layers = layers[:3]
    for layer in search_layers:
        log(f"vo recipe L{layer}")
        X = load_hidden(layer)
        even, odd = np.arange(0, 256, 2), np.arange(1, 256, 2)
        Wq = load_tensor(tname(layer, "self_attn.q_proj.weight"))
        Wv = load_tensor(tname(layer, "self_attn.v_proj.weight"))
        Wo = load_tensor(tname(layer, "self_attn.o_proj.weight"))
        recs = vo_recipes(Wv, Wo, X[even], Wq)
        proxy_ref = gqa_vo_proxy(X[odd], Wq, Wv)
        y_ref = proxy_ref @ Wo.T
        base = eval_vo(X[odd], Wq, Wv, Wo, 3, y_ref, proxy_ref)
        cell = {"diag": {}, "identity_q3": {k: base[k] for k in ("comp_hold", "o_hold_exact_proxy", "wcos_o")}}
        for name, s in recs.items():
            Wv2, Wo2 = apply_vo_s(Wv, Wo, s)
            ev = eval_vo(X[odd], Wq, Wv2, Wo2, 3, y_ref, proxy_ref)
            cell["diag"][name] = ev
            vote.setdefault(name, []).append(ev["comp_hold"] - base["comp_hold"])
            del Wv2, Wo2
        result["layers"][str(layer)] = {"recipe_search": cell}
        del Wq, Wv, Wo, X
        gc.collect()
        log(f"L{layer} vo id_q3={base['comp_hold']:.6f}")

    result["recipe_mean_delta_q3"] = {k: float(np.mean(v)) for k, v in vote.items()}
    wo = {k: v for k, v in result["recipe_mean_delta_q3"].items() if not k.startswith("act_")}
    winner = max(wo, key=wo.get) if wo else "identity"
    result["winner_weight_only"] = winner
    log(f"vo winner={winner} {result['recipe_mean_delta_q3']}")

    for layer in layers:
        log(f"vo sweep L{layer}")
        X = load_hidden(layer)
        even, odd = np.arange(0, 256, 2), np.arange(1, 256, 2)
        Wq = load_tensor(tname(layer, "self_attn.q_proj.weight"))
        Wv = load_tensor(tname(layer, "self_attn.v_proj.weight"))
        Wo = load_tensor(tname(layer, "self_attn.o_proj.weight"))
        recs = vo_recipes(Wv, Wo, X[even], Wq)
        proxy_ref = gqa_vo_proxy(X[odd], Wq, Wv)
        y_ref = proxy_ref @ Wo.T
        cell = result["layers"].setdefault(str(layer), {})
        cell["sweep"] = {}
        for name in ("identity", winner):
            Wv2, Wo2 = apply_vo_s(Wv, Wo, recs[name])
            cell["sweep"][name] = {}
            for b in bits_list:
                cell["sweep"][name][f"q{b}"] = eval_vo(X[odd], Wq, Wv2, Wo2, b, y_ref, proxy_ref)
            if name != "identity":
                del Wv2, Wo2
            else:
                del Wv2, Wo2
        id3 = cell["sweep"]["identity"]["q3"]["comp_hold"]
        w3 = cell["sweep"][winner]["q3"]["comp_hold"]
        id4 = cell["sweep"]["identity"]["q4"]["comp_hold"]
        log(
            f"L{layer} VO id Q3/Q4={id3:.6f}/{id4:.6f} {winner} "
            f"{w3:.6f}/{cell['sweep'][winner]['q4']['comp_hold']:.6f} dQ3={w3-id3:+.6f}"
        )
        del Wq, Wv, Wo, X
        gc.collect()
    return result


# ---------------------------------------------------------------------------
# norm-boundary (post-norm readers share gamma)
# ---------------------------------------------------------------------------

def eval_readers(X_hold: np.ndarray, X_hold_scaled: np.ndarray, weights: dict[str, np.ndarray], bits: int, y_ref: dict[str, np.ndarray]) -> dict:
    out = {}
    for name, W in weights.items():
        Wq = uniform_absmax_recon(W, bits)
        yq = gemv(X_hold_scaled, Wq)
        out[name] = {
            "hold": mean_row_cosine(y_ref[name], yq),
            "wcos": weight_cos(W, Wq),
            "util": group_util(W),
        }
        del Wq
    holds = [v["hold"] for v in out.values()]
    out["min_hold"] = float(min(holds)) if holds else 0.0
    out["mean_hold"] = float(np.mean(holds)) if holds else 0.0
    return out


def phase_norm(layers: list[int], bits_list=(3, 4)) -> dict:
    result: dict = {"layers": {}}
    for layer in layers:
        log(f"norm-boundary L{layer} gqa={is_gqa(layer)}")
        X = load_hidden(layer)  # captured post-norm (input_ln site UNCONFIRMED)
        even, odd = np.arange(0, 256, 2), np.arange(1, 256, 2)
        gamma_in = load_tensor(tname(layer, "input_layernorm.weight"))
        gamma_post = load_tensor(tname(layer, "post_attention_layernorm.weight"))
        # mixer readers of input_ln
        if is_gqa(layer):
            readers_in = {
                "q": load_tensor(tname(layer, "self_attn.q_proj.weight")),
                "k": load_tensor(tname(layer, "self_attn.k_proj.weight")),
                "v": load_tensor(tname(layer, "self_attn.v_proj.weight")),
            }
        else:
            readers_in = {
                "qkv": load_tensor(tname(layer, "linear_attn.in_proj_qkv.weight")),
                "z": load_tensor(tname(layer, "linear_attn.in_proj_z.weight")),
            }
        readers_mlp = {
            "gate": load_tensor(tname(layer, "mlp.gate_proj.weight")),
            "up": load_tensor(tname(layer, "mlp.up_proj.weight")),
        }

        def col_s(readers: dict[str, np.ndarray], X_fit: np.ndarray | None) -> dict[str, np.ndarray]:
            # pooled absmax across readers
            am = None
            rms = None
            for W in readers.values():
                a = axis_absmax(W, 0)
                r = axis_rms(W, 0)
                am = a if am is None else np.maximum(am, a)
                rms = r if rms is None else np.maximum(rms, r)
            rec = {
                "identity": np.ones(HIDDEN, dtype=np.float64),
                "col_equalize": normalize_s(am),
                "rms_equalize": normalize_s(rms),
            }
            if X_fit is not None:
                xr = np.maximum(axis_rms(X_fit, 0), 1e-12)
                rec["act_a50"] = normalize_s((xr / np.median(xr)) ** 0.50)
                rec["act_a100"] = normalize_s(xr / np.median(xr))
                rec["smooth_a50"] = normalize_s((xr ** 0.50) / np.maximum(am, 1e-12) ** 0.50)
            return rec

        def apply_col(readers, s):
            return {k: np.ascontiguousarray(W / s[None, :], dtype=np.float32) for k, W in readers.items()}

        y_ref_in = {k: gemv(X[odd], W) for k, W in readers_in.items()}
        y_ref_mlp = {k: gemv(X[odd], W) for k, W in readers_mlp.items()}
        rec_in = col_s(readers_in, X[even])
        rec_mlp = col_s(readers_mlp, X[even])
        cell = {"input_ln": {}, "post_ln": {}, "gamma_in_stats": _vec_stats(gamma_in), "gamma_post_stats": _vec_stats(gamma_post)}
        for site, readers, recs, y_ref, _gamma in (
            ("input_ln", readers_in, rec_in, y_ref_in, gamma_in),
            ("post_ln", readers_mlp, rec_mlp, y_ref_mlp, gamma_post),
        ):
            for name, s in recs.items():
                # X' = X * s  (because y' = y * s when gamma *= s)
                Xs = (X[odd] * s[None, :]).astype(np.float32)
                Ws = apply_col(readers, s)
                cell[site][name] = {}
                for b in bits_list:
                    cell[site][name][f"q{b}"] = eval_readers(X[odd], Xs, Ws, b, y_ref)
                del Ws, Xs
        result["layers"][str(layer)] = cell
        id3 = cell["post_ln"]["identity"]["q3"]["mean_hold"]
        ce3 = cell["post_ln"]["col_equalize"]["q3"]["mean_hold"]
        log(
            f"L{layer} post_ln mean_hold Q3 id={id3:.6f} col_eq={ce3:.6f} d={ce3-id3:+.6f} "
            f"min id={cell['post_ln']['identity']['q3']['min_hold']:.6f} "
            f"col_eq={cell['post_ln']['col_equalize']['q3']['min_hold']:.6f}"
        )
        del readers_in, readers_mlp, X
        gc.collect()
    return result


def _vec_stats(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    return {
        "min": float(np.min(v)),
        "median": float(np.median(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "n_zero": int(np.sum(v == 0)),
    }


# ---------------------------------------------------------------------------
# DeltaNet linear_attn.norm (128) vs out_proj shared dim
# ---------------------------------------------------------------------------

def phase_dn_norm(layers: list[int], bits_list=(3, 4)) -> dict:
    result = {"layers": {}}
    for layer in layers:
        if is_gqa(layer):
            continue
        log(f"dn-norm L{layer}")
        Wnorm = load_tensor(tname(layer, "linear_attn.norm.weight"))  # [128]
        Wo = load_tensor(tname(layer, "linear_attn.out_proj.weight"))  # [5120, 6144]
        # columns i, 128+i, ... share scale
        o = np.abs(Wo).reshape(Wo.shape[0], 48, 128)
        o_am = np.maximum(o.max(axis=(0, 1)), 1e-12)  # [128]
        recs = {
            "identity": np.ones(128, dtype=np.float64),
            "col_equalize": normalize_s(o_am),
            "absmax_a50": normalize_s(np.sqrt(o_am / np.maximum(np.abs(Wnorm), 1e-12))),
        }
        X = load_hidden(layer)
        # mixer proxy from fused qkvz
        Wqkv = load_tensor(tname(layer, "linear_attn.in_proj_qkv.weight"))
        Wz = load_tensor(tname(layer, "linear_attn.in_proj_z.weight"))
        # reuse forensics fuse
        from qwen38_out_proj_forensics import fuse_qkvz, deltanet_out_proxy

        Wf = fuse_qkvz(Wqkv, Wz)
        odd = np.arange(1, 256, 2)
        proxy = deltanet_out_proxy(X[odd], Wf)
        y_ref = proxy @ Wo.T
        cell = {"norm_stats": _vec_stats(Wnorm), "recipes": {}}
        for name, s in recs.items():
            # scale shared dim: out columns 48 copies, norm *= s
            Wo2 = Wo.reshape(Wo.shape[0], 48, 128) / s[None, None, :]
            Wo2 = np.ascontiguousarray(Wo2.reshape(Wo.shape), dtype=np.float32)
            # proxy is pre-norm mixer; the actual path is gated-rmsnorm then o.
            # We cannot replay gated-rmsnorm without rec state. Score o under
            # the same proxy as forensics, with column scales folded. Function
            # of (norm, o) is exact only through the real gated-rms; proxy hold
            # is a RANKER. Label it.
            cell["recipes"][name] = {}
            for b in bits_list:
                Woq = uniform_absmax_recon(Wo2, b)
                cell["recipes"][name][f"q{b}"] = {
                    "o_hold_proxy": mean_row_cosine(y_ref, proxy @ Woq.T),
                    "wcos_o": weight_cos(Wo2, Woq),
                    "util_o": group_util(Wo2),
                }
                del Woq
            del Wo2
        result["layers"][str(layer)] = cell
        log(
            f"L{layer} dn-norm proxy Q3 id={cell['recipes']['identity']['q3']['o_hold_proxy']:.6f} "
            f"col_eq={cell['recipes']['col_equalize']['q3']['o_hold_proxy']:.6f}"
        )
        del Wqkv, Wz, Wf, Wo, X
        gc.collect()
    return result


# ---------------------------------------------------------------------------
# q/k: exactness on real L3 + runtime-M experiment
# ---------------------------------------------------------------------------

def phase_qk(layers: list[int]) -> dict:
    result = {"layers": {}}
    for layer in layers:
        log(f"qk L{layer}")
        X = load_hidden(layer)
        odd = np.arange(1, 256, 2)
        Wq = load_tensor(tname(layer, "self_attn.q_proj.weight"))
        Wk = load_tensor(tname(layer, "self_attn.k_proj.weight"))
        qn = load_tensor(tname(layer, "self_attn.q_norm.weight"))
        kn = load_tensor(tname(layer, "self_attn.k_norm.weight"))
        q_act = gemv(X[odd], Wq).reshape(-1, GQA_HEADS, 2, GQA_HEAD_DIM)
        k_act = gemv(X[odd], Wk).reshape(-1, GQA_KV, GQA_HEAD_DIM)
        q_raw = q_act[:, :, 0, :]
        # scores after QK-norm+RoPE, slot=0 (relative; we compare pairs)
        ntok = q_raw.shape[0]
        scores0 = np.zeros((ntok, GQA_HEADS), dtype=np.float64)
        for t in range(min(ntok, 32)):
            for h in range(GQA_HEADS):
                kh = h // (GQA_HEADS // GQA_KV)
                qh = qk_norm_rope_one(q_raw[t, h], qn, slot=t)
                khv = qk_norm_rope_one(k_act[t, kh], kn, slot=t)
                scores0[t, h] = np.dot(qh, khv)

        # 1) fold random invertible M into W, NO runtime inverse: scores must move
        rng = np.random.default_rng(layer + 7)
        M = rng.normal(size=(256, 256))
        M += 1.5 * np.eye(256)
        MinvT = np.linalg.inv(M).T
        Wq2 = Wq.reshape(GQA_HEADS, 2, GQA_HEAD_DIM, HIDDEN).copy()
        Wk2 = Wk.reshape(GQA_KV, GQA_HEAD_DIM, HIDDEN).copy()
        for h in range(GQA_KV):
            Wk2[h] = MinvT @ Wk2[h]
            for r in range(GQA_HEADS // GQA_KV):
                hh = h * (GQA_HEADS // GQA_KV) + r
                Wq2[hh, 0] = M @ Wq2[hh, 0]
        Wq2 = Wq2.reshape(Wq.shape).astype(np.float32)
        Wk2 = Wk2.reshape(Wk.shape).astype(np.float32)
        q2 = gemv(X[odd], Wq2).reshape(-1, GQA_HEADS, 2, GQA_HEAD_DIM)[:, :, 0, :]
        k2 = gemv(X[odd], Wk2).reshape(-1, GQA_KV, GQA_HEAD_DIM)
        scoresM = np.zeros_like(scores0)
        for t in range(min(ntok, 32)):
            for h in range(GQA_HEADS):
                kh = h // (GQA_HEADS // GQA_KV)
                scoresM[t, h] = np.dot(
                    qk_norm_rope_one(q2[t, h], qn, slot=t),
                    qk_norm_rope_one(k2[t, kh], kn, slot=t),
                )
        rel = np.abs(scoresM[:32] - scores0[:32]) / (np.abs(scores0[:32]) + 1e-12)

        # 2) same M, runtime inverse AFTER gemv BEFORE qk-norm: scores must match
        # q_raw_rec = M^{-1} @ (M @ q_raw) = q_raw
        scores_rt = np.zeros_like(scores0)
        Minv = np.linalg.inv(M)
        MT = M.T
        for t in range(min(ntok, 32)):
            for h in range(GQA_HEADS):
                kh = h // (GQA_HEADS // GQA_KV)
                q_rec = Minv @ q2[t, h]
                k_rec = MT @ k2[t, kh]
                scores_rt[t, h] = np.dot(
                    qk_norm_rope_one(q_rec, qn, slot=t),
                    qk_norm_rope_one(k_rec, kn, slot=t),
                )
        rel_rt = np.abs(scores_rt[:32] - scores0[:32]) / (np.abs(scores0[:32]) + 1e-12)

        # 3) non-rotary permutation + gamma permute, no runtime: should be exact
        nr = np.arange(GQA_ROTARY, GQA_HEAD_DIM)
        P = rng.permutation(nr.size)
        WqP = Wq.reshape(GQA_HEADS, 2, GQA_HEAD_DIM, HIDDEN).copy()
        WkP = Wk.reshape(GQA_KV, GQA_HEAD_DIM, HIDDEN).copy()
        qnP = qn.copy()
        knP = kn.copy()
        WqP[:, 0, nr] = WqP[:, 0, nr[P]]
        WkP[:, nr] = WkP[:, nr[P]]
        qnP[nr] = qnP[nr[P]]
        knP[nr] = knP[nr[P]]
        WqP = WqP.reshape(Wq.shape).astype(np.float32)
        WkP = WkP.reshape(Wk.shape).astype(np.float32)
        qP = gemv(X[odd], WqP).reshape(-1, GQA_HEADS, 2, GQA_HEAD_DIM)[:, :, 0, :]
        kP = gemv(X[odd], WkP).reshape(-1, GQA_KV, GQA_HEAD_DIM)
        scoresP = np.zeros_like(scores0)
        for t in range(min(ntok, 32)):
            for h in range(GQA_HEADS):
                kh = h // (GQA_HEADS // GQA_KV)
                scoresP[t, h] = np.dot(
                    qk_norm_rope_one(qP[t, h], qnP, slot=t),
                    qk_norm_rope_one(kP[t, kh], knP, slot=t),
                )
        rel_p = np.abs(scoresP[:32] - scores0[:32]) / (np.abs(scores0[:32]) + 1e-12)

        # 4) runtime-M compressibility: per-KV-head whitening of concatenated q/k rows
        # M = Λ^{-1/2} V^T of the 256-d output-row Gram. Fold into W, quant, undo in float.
        rt = {}
        for bits in (2, 3, 4):
            Wq_id_q = uniform_absmax_recon(Wq, bits)
            Wk_id_q = uniform_absmax_recon(Wk, bits)
            hold_q_id = hold_cosine(X[odd], Wq, Wq_id_q)
            hold_k_id = hold_cosine(X[odd], Wk, Wk_id_q)
            # whitened
            WqW = Wq.reshape(GQA_HEADS, 2, GQA_HEAD_DIM, HIDDEN).copy()
            WkW = Wk.reshape(GQA_KV, GQA_HEAD_DIM, HIDDEN).copy()
            Ms = []
            for h in range(GQA_KV):
                blocks = [WkW[h]]
                for r in range(GQA_HEADS // GQA_KV):
                    hh = h * (GQA_HEADS // GQA_KV) + r
                    blocks.append(WqW[hh, 0])
                cat = np.concatenate(blocks, axis=1)  # [256, 5120*7]
                # Gram 256x256
                Grm = cat @ cat.T
                # eigh
                evals, evecs = np.linalg.eigh(Grm.astype(np.float64))
                evals = np.clip(evals, 1e-8, None)
                Mw = (evecs * (evals ** -0.5)[None, :]) @ evecs.T  # V Λ^{-1/2} V^T, symmetric
                Ms.append(Mw)
                WkW[h] = Mw @ WkW[h]
                for r in range(GQA_HEADS // GQA_KV):
                    hh = h * (GQA_HEADS // GQA_KV) + r
                    WqW[hh, 0] = Mw @ WqW[hh, 0]
            WqW = WqW.reshape(Wq.shape).astype(np.float32)
            WkW = WkW.reshape(Wk.shape).astype(np.float32)
            WqWq = uniform_absmax_recon(WqW, bits)
            WkWq = uniform_absmax_recon(WkW, bits)
            # recover in original basis (runtime inverse = M^{-1} = V Λ^{1/2} V^T)
            Wq_rec = WqWq.reshape(GQA_HEADS, 2, GQA_HEAD_DIM, HIDDEN).copy()
            Wk_rec = WkWq.reshape(GQA_KV, GQA_HEAD_DIM, HIDDEN).copy()
            for h in range(GQA_KV):
                Minv_w = np.linalg.inv(Ms[h])
                Wk_rec[h] = Minv_w @ Wk_rec[h]
                for r in range(GQA_HEADS // GQA_KV):
                    hh = h * (GQA_HEADS // GQA_KV) + r
                    Wq_rec[hh, 0] = Minv_w @ Wq_rec[hh, 0]
            Wq_rec = Wq_rec.reshape(Wq.shape).astype(np.float32)
            Wk_rec = Wk_rec.reshape(Wk.shape).astype(np.float32)
            rt[f"q{bits}"] = {
                "q_hold_id": hold_q_id,
                "k_hold_id": hold_k_id,
                "q_hold_whiten_then_undo": hold_cosine(X[odd], Wq, Wq_rec),
                "k_hold_whiten_then_undo": hold_cosine(X[odd], Wk, Wk_rec),
                "wcos_q_id": weight_cos(Wq, Wq_id_q),
                "wcos_k_id": weight_cos(Wk, Wk_id_q),
                "wcos_q_whiten": weight_cos(WqW, WqWq),
                "wcos_k_whiten": weight_cos(WkW, WkWq),
            }
            del Wq_id_q, Wk_id_q, WqW, WkW, WqWq, WkWq, Wq_rec, Wk_rec

        # 5) Hadamard on head_dim (runtime, same protocol)
        H = hadamard_256()
        WqH = Wq.reshape(GQA_HEADS, 2, GQA_HEAD_DIM, HIDDEN).copy()
        WkH = Wk.reshape(GQA_KV, GQA_HEAD_DIM, HIDDEN).copy()
        for h in range(GQA_HEADS):
            WqH[h, 0] = H @ WqH[h, 0]
        for h in range(GQA_KV):
            WkH[h] = H @ WkH[h]
        WqH = WqH.reshape(Wq.shape).astype(np.float32)
        WkH = WkH.reshape(Wk.shape).astype(np.float32)
        had = {}
        for bits in (3, 4):
            WqHq = uniform_absmax_recon(WqH, bits)
            WkHq = uniform_absmax_recon(WkH, bits)
            Wq_rec = WqHq.reshape(GQA_HEADS, 2, GQA_HEAD_DIM, HIDDEN).copy()
            Wk_rec = WkHq.reshape(GQA_KV, GQA_HEAD_DIM, HIDDEN).copy()
            for h in range(GQA_HEADS):
                Wq_rec[h, 0] = H @ Wq_rec[h, 0]  # H=H^{-1}
            for h in range(GQA_KV):
                Wk_rec[h] = H @ Wk_rec[h]
            Wq_rec = Wq_rec.reshape(Wq.shape).astype(np.float32)
            Wk_rec = Wk_rec.reshape(Wk.shape).astype(np.float32)
            had[f"q{bits}"] = {
                "q_hold_had_then_undo": hold_cosine(X[odd], Wq, Wq_rec),
                "k_hold_had_then_undo": hold_cosine(X[odd], Wk, Wk_rec),
                "q_hold_id": hold_cosine(X[odd], Wq, uniform_absmax_recon(Wq, bits)),
                "k_hold_id": hold_cosine(X[odd], Wk, uniform_absmax_recon(Wk, bits)),
            }
            del WqHq, WkHq, Wq_rec, Wk_rec

        result["layers"][str(layer)] = {
            "fullM_pre_norm_score_rel_mean": float(np.mean(rel)),
            "fullM_pre_norm_score_rel_max": float(np.max(rel)),
            "fullM_runtime_undo_score_rel_mean": float(np.mean(rel_rt)),
            "fullM_runtime_undo_score_rel_max": float(np.max(rel_rt)),
            "nr_perm_score_rel_mean": float(np.mean(rel_p)),
            "nr_perm_score_rel_max": float(np.max(rel_p)),
            "q_norm_stats": _vec_stats(qn),
            "k_norm_stats": _vec_stats(kn),
            "runtime_whiten": rt,
            "runtime_hadamard": had,
            "runtime_cost_projected": {
                "gqa_layers": 16,
                "heads_q": 24,
                "heads_k": 4,
                "dim": 256,
                "fma_per_token": 16 * (24 + 4) * 256 * 256,
                "m_storage_bytes_f32_per_kv_head": 256 * 256 * 4,
                "m_storage_bytes_all": 16 * 4 * 256 * 256 * 4,
                "complete_bpw_of_M_f32": 8 * (16 * 4 * 256 * 256 * 4) / SOURCE_N,
            },
        }
        log(
            f"L{layer} qk fullM_rel_mean={float(np.mean(rel)):.4f} "
            f"runtime_undo={float(np.mean(rel_rt)):.2e} nr_perm={float(np.mean(rel_p)):.2e} "
            f"Q3 q_hold id={rt['q3']['q_hold_id']:.6f} whiten={rt['q3']['q_hold_whiten_then_undo']:.6f}"
        )
        del Wq, Wk, Wq2, Wk2, WqP, WkP, X
        gc.collect()
    return result


def hadamard_256() -> np.ndarray:
    n = 256
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    H /= np.sqrt(n)
    return H


# ---------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------

def class_elements() -> dict:
    mlp = 64 * (INTER * HIDDEN * 2 + HIDDEN * INTER)  # gate+up+down
    gqa_vo = 16 * (1024 * HIDDEN + HIDDEN * 6144)
    gqa_qk = 16 * (12288 * HIDDEN + 1024 * HIDDEN)
    dn_in = 48 * (10240 * HIDDEN + 6144 * HIDDEN)
    dn_out = 48 * (HIDDEN * 6144)
    return {
        "source_n": SOURCE_N,
        "mlp_gate_up_down": mlp,
        "mlp_down_only": 64 * HIDDEN * INTER,
        "mlp_up_only": 64 * INTER * HIDDEN,
        "gqa_v_o": gqa_vo,
        "gqa_q_k": gqa_qk,
        "dn_in_qkv_z": dn_in,
        "dn_out": dn_out,
        "g0_complete_bpw": 4.252735126866492,
        "nominal_qn_g64_bpw": {2: 2.25, 3: 3.25, 4: 4.25},
    }


def main() -> None:
    global _LOG_FH
    t0 = time.time()
    LOG.write_text("")
    _LOG_FH = LOG.open("a")
    log("start invertible-dof")
    report = {
        "src": str(SRC),
        "cap": str(CAP),
        "protocol": {
            "hold_rows": "odd 128 of 256-token capture",
            "fit_rows": "even 128, only for act-aware s",
            "codec": "HGRAVU01 absmax g=64 flat C-order (= per-row along K; all K % 64 == 0)",
            "metric_primary": "composition hold = mean row-cosine of site output vs BF16",
            "capture_sha256_self_claimed": "fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512",
            "capture_note": "256 tokens, ranks reliably, magnitudes not; weight-only s does not use it",
        },
        "geometry": class_elements(),
    }

    report["exactness"] = exactness_suite()

    mlp_search = [0, 3, 15, 32, 47, 63]
    mlp_sweep = [0, 1, 3, 7, 8, 15, 16, 24, 31, 32, 40, 47, 48, 55, 62, 63]
    report["mlp"] = phase_mlp(mlp_sweep, mlp_search)

    vo_layers = [3, 7, 15, 31, 47, 63]
    report["gqa_vo"] = phase_vo(vo_layers)

    norm_layers = [0, 3, 15, 32, 47, 63]
    report["norm_boundary"] = phase_norm(norm_layers)

    report["dn_norm_o"] = phase_dn_norm([0, 16, 32, 48, 62])

    report["gqa_qk"] = phase_qk([3, 31, 63])

    report["wall_s"] = time.time() - t0
    report["rss_max_gb"] = rss_gb()
    report["script"] = str(Path(__file__).resolve())
    OUT.write_text(json.dumps(report, indent=2, default=float))
    report["json_sha256"] = sha256_file(OUT)
    # rewrite with hash
    OUT.write_text(json.dumps(report, indent=2, default=float))
    log(f"wrote {OUT} wall={report['wall_s']:.1f}s rss_max={report['rss_max_gb']:.3f}G sha={report['json_sha256']}")
    _LOG_FH.close()


if __name__ == "__main__":
    main()

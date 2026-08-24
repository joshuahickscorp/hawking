#!/usr/bin/env python3
"""Hierarchical correction ladder on real Qwen3.8-27B tensors.

W ≈ B + C1 + C2_sparse, each level fit to the residual of the previous
against activation-conditioned output error on a held-out split.

CPU / numpy only. No GPU, no generate, no pack, no repo writes.
"""
from __future__ import annotations

import argparse
import gc
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
G0_MANIFEST = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json")
OUT_JSON = Path("/tmp/g1_hierarchical_correction.json")
OUT_LOG = Path("/tmp/g1_hierarchical_correction.log")

HIDDEN = 5120
INTERMEDIATE = 17408
N_TOK = 256
FIT = slice(0, 192)
HOLD = slice(192, 256)
G_HIER = 128

# Language mass from G0 / geometry (cited, verified against index in this run).
N_LANG = 26_895_998_464
MASS = {
    "mlp.gate_proj": 5_704_253_440,
    "mlp.up_proj": 5_704_253_440,
    "mlp.down_proj": 5_704_253_440,
    "attn": 7_237_795_840,
    "embed_lm_head": 2_542_796_800,
    "small": 2_645_504,
}

LAYERS_FULL = (0, 3, 15, 31, 32, 47, 63)
LAYERS_PROBE = (0, 3, 63)


def rss_gb() -> float:
    # macOS ru_maxrss is bytes; Linux is KiB.
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if raw > 1e9:  # bytes
        return raw / (1024.0 ** 3)
    return raw / (1024.0 ** 2)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] rss={rss_gb():.3f}G {msg}"
    print(line, flush=True)
    with OUT_LOG.open("a") as fh:
        fh.write(line + "\n")


def f16(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def is_gqa(layer: int) -> bool:
    return (layer + 1) % 4 == 0


def tname(layer: int, suffix: str) -> str:
    return f"language_model.model.layers.{layer}.{suffix}"


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
    if raw.size != N_TOK * HIDDEN:
        raise RuntimeError(f"hidden L{layer} size {raw.size}")
    return np.ascontiguousarray(raw.reshape(N_TOK, HIDDEN))


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def weight_metrics(src: np.ndarray, rec: np.ndarray) -> dict:
    a = src.reshape(-1).astype(np.float64, copy=False)
    b = rec.reshape(-1).astype(np.float64, copy=False)
    dot = float(np.dot(a, b))
    na = float(np.dot(a, a)) ** 0.5
    nb = float(np.dot(b, b)) ** 0.5
    err = a - b
    ne = float(np.dot(err, err)) ** 0.5
    return {
        "weight_cosine": dot / (na * nb + 1e-30),
        "weight_rel_l2": ne / (na + 1e-30),
    }


def output_metrics(Y: np.ndarray, Yh: np.ndarray) -> dict:
    out = {}
    for tag, sl in (("fit", FIT), ("hold", HOLD), ("all", slice(None))):
        y = Y[sl].astype(np.float64, copy=False)
        yh = Yh[sl].astype(np.float64, copy=False)
        yf = y.reshape(-1)
        yhf = yh.reshape(-1)
        dot = float(np.dot(yf, yhf))
        na = float(np.dot(yf, yf)) ** 0.5
        nb = float(np.dot(yhf, yhf)) ** 0.5
        ne = float(np.dot(yf - yhf, yf - yhf)) ** 0.5
        yn = np.linalg.norm(y, axis=1)
        yhn = np.linalg.norm(yh, axis=1)
        rc = (y * yh).sum(axis=1) / (yn * yhn + 1e-30)
        out[f"{tag}_output_cosine"] = dot / (na * nb + 1e-30)
        out[f"{tag}_output_rel_l2"] = ne / (na + 1e-30)
        out[f"{tag}_mean_row_cosine"] = float(rc.mean()) if rc.size else None
        out[f"{tag}_min_row_cosine"] = float(rc.min()) if rc.size else None
    return out


def score_pair(W: np.ndarray, rec: np.ndarray, X: np.ndarray | None, Y: np.ndarray | None) -> dict:
    d = weight_metrics(W, rec)
    if X is not None and Y is not None:
        Yh = X @ rec.T
        d.update(output_metrics(Y, Yh))
        del Yh
    else:
        d["hold_output_cosine"] = None
        d["hold_output_rel_l2"] = None
    return d


# ---------------------------------------------------------------------------
# grouping (K-axis, one row = sequence of groups) and flat-C (incumbent)
# ---------------------------------------------------------------------------

def n_groups(inn: int, G: int) -> int:
    return (inn + G - 1) // G


def k_group_view(W: np.ndarray, G: int) -> tuple[np.ndarray, int, int]:
    """Pad K to multiple of G. Returns (out, n_g, G), original inn."""
    out, inn = W.shape
    ng = n_groups(inn, G)
    pad = ng * G
    if pad == inn:
        return W.reshape(out, ng, G), inn, ng
    buf = np.zeros((out, pad), dtype=W.dtype)
    buf[:, :inn] = W
    return buf.reshape(out, ng, G), inn, ng


def apply_k_scale(P: np.ndarray, scales: np.ndarray, G: int) -> np.ndarray:
    """P (out,inn) ±1, scales (out,ng) -> P * scale_g."""
    out, inn = P.shape
    ng = scales.shape[1]
    rec = np.empty_like(P)
    rec[:, : ng * G] = (P[:, : ng * G].reshape(out, ng, G) * scales[:, :, None]).reshape(out, ng * G)
    if inn > ng * G:
        rec[:, ng * G :] = P[:, ng * G :] * scales[:, -1:]
    return rec


def apply_k_offset(offset: np.ndarray, inn: int, G: int) -> np.ndarray:
    out, ng = offset.shape
    rec = np.empty((out, inn), dtype=np.float32)
    rec[:, : ng * G] = np.broadcast_to(offset[:, :, None], (out, ng, G)).reshape(out, ng * G)
    if inn > ng * G:
        rec[:, ng * G :] = offset[:, -1:]
    return rec


def meanabs_scales_k(W: np.ndarray, G: int) -> np.ndarray:
    g, inn, ng = k_group_view(W, G)
    return f16(np.mean(np.abs(g), axis=2))


def sign_pattern(W: np.ndarray) -> np.ndarray:
    return np.where(W >= 0.0, np.float32(1.0), np.float32(-1.0))


def pair_pattern(W: np.ndarray) -> np.ndarray:
    """One sign per consecutive K-pair, broadcast to both weights."""
    out, inn = W.shape
    P = np.empty_like(W)
    even = inn - (inn % 2)
    a = W[:, 0:even:2]
    b = W[:, 1:even:2]
    s = np.where((a + b) >= 0.0, np.float32(1.0), np.float32(-1.0))
    P[:, 0:even:2] = s
    P[:, 1:even:2] = s
    if inn % 2:
        P[:, -1] = np.where(W[:, -1] >= 0.0, np.float32(1.0), np.float32(-1.0))
    return P


def binary_k(W: np.ndarray, G: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    P = sign_pattern(W)
    S = meanabs_scales_k(W, G)
    rec = apply_k_scale(P, S, G)
    bpw = 1.0 + 16.0 / G
    return P, S, rec, bpw


def pair_binary_k(W: np.ndarray, G: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    P = pair_pattern(W)
    S = meanabs_scales_k(W, G)
    rec = apply_k_scale(P, S, G)
    bpw = 0.5 + 16.0 / G
    return P, S, rec, bpw


def flat_c_binary(W: np.ndarray, G: int) -> np.ndarray:
    flat = W.reshape(-1)
    n = int(flat.size)
    groups = (n + G - 1) // G
    pad = np.zeros(groups * G, dtype=np.float32)
    pad[:n] = flat
    g = pad.reshape(groups, G)
    scales = f16(np.mean(np.abs(g), axis=1))
    rec = (np.where(g >= 0.0, 1.0, -1.0) * scales[:, None]).reshape(-1)[:n]
    return rec.reshape(W.shape).astype(np.float32)


def uniform_k(W: np.ndarray, bits: int, G: int) -> np.ndarray:
    bound = (1 << (bits - 1)) - 1
    g, inn, ng = k_group_view(W, G)
    amax = np.max(np.abs(g), axis=2)
    scale = f16(amax / max(bound, 1))
    den = np.where(scale > 0.0, scale, 1.0)
    q = np.rint(g / den[:, :, None]).clip(-bound, bound)
    rec = (q * scale[:, :, None]).reshape(W.shape[0], ng * G)[:, :inn]
    return rec.astype(np.float32)


def uniform_flat_c(W: np.ndarray, bits: int, G: int) -> np.ndarray:
    bound = (1 << (bits - 1)) - 1
    flat = W.reshape(-1)
    n = int(flat.size)
    groups = (n + G - 1) // G
    pad = np.zeros(groups * G, dtype=np.float32)
    pad[:n] = flat
    g = pad.reshape(groups, G)
    amax = np.max(np.abs(g), axis=1)
    scale = f16(amax / max(bound, 1))
    den = np.where(scale > 0.0, scale, 1.0)
    q = np.rint(g / den[:, None]).clip(-bound, bound)
    rec = (q * scale[:, None]).reshape(-1)[:n]
    return rec.reshape(W.shape).astype(np.float32)


def ternary_k(W: np.ndarray, thresh_mult: float, G: int) -> tuple[np.ndarray, float, float]:
    """Production ternary: 2-bit codes + fp16 threshold + fp16 scale per group."""
    g, inn, ng = k_group_view(W, G)
    base = np.mean(np.abs(g), axis=2)
    thr = f16(base * thresh_mult)
    active = np.abs(g) >= thr[:, :, None]
    selected = np.where(active, np.abs(g), 0.0)
    count = np.maximum(active.sum(axis=2), 1)
    scales = f16(selected.sum(axis=2) / count)
    rec = np.where(active, np.where(g >= 0.0, 1.0, -1.0) * scales[:, :, None], 0.0)
    rec = rec.reshape(W.shape[0], ng * G)[:, :inn].astype(np.float32)
    density = float(active.reshape(-1)[: W.size].mean())
    bpw = 2.0 + 32.0 / G
    return rec, bpw, density


# ---------------------------------------------------------------------------
# activation-fitted scale / offset on a fixed sign (or pair-sign) pattern
# ---------------------------------------------------------------------------

def fit_scale_offset(
    W: np.ndarray, P: np.ndarray, X: np.ndarray, G: int, sl: slice
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per K-group 2-param LS: min ||Xg (w - s p - c 1)|| on the fit split.

    Well-determined: 2 params, |sl| tokens. Uses activations as a ranking+fit
    of two scalars, not a 5120-dim plane.
    """
    Xf = X[sl]
    out, inn = W.shape
    ng = n_groups(inn, G)
    scales = np.zeros((out, ng), dtype=np.float32)
    offs = np.zeros((out, ng), dtype=np.float32)
    diag = {
        "n_groups": ng,
        "n_singular_2x2": 0,
        "n_ok": 0,
        "median_scale": None,
        "median_off_over_scale": None,
    }
    Tf = Xf.shape[0]
    for g in range(ng):
        a0 = g * G
        a1 = min(inn, a0 + G)
        Xg = Xf[:, a0:a1]  # (Tf, G')
        Wg = W[:, a0:a1]
        Pg = P[:, a0:a1]
        # a = Xg @ w^T → (Tf, out); b = Xg @ p^T; s = Xg @ 1
        A = Xg @ Wg.T
        B = Xg @ Pg.T
        Svec = Xg.sum(axis=1)  # (Tf,)
        G11 = np.einsum("to,to->o", B, B)
        G12 = B.T @ Svec
        G22 = float(np.dot(Svec, Svec))
        r1 = np.einsum("to,to->o", B, A)
        r2 = A.T @ Svec
        det = G11 * G22 - G12 * G12
        ok = np.abs(det) > 1e-12 * (np.abs(G11) * G22 + 1.0)
        s = np.zeros(out, dtype=np.float64)
        c = np.zeros(out, dtype=np.float64)
        s[ok] = (G22 * r1[ok] - G12[ok] * r2[ok]) / det[ok]
        c[ok] = (G11[ok] * r2[ok] - G12[ok] * r1[ok]) / det[ok]
        # scale-only fallback
        bad = ~ok
        if np.any(bad):
            den = G11[bad]
            s[bad] = np.where(den > 1e-18, r1[bad] / den, 0.0)
            c[bad] = 0.0
            diag["n_singular_2x2"] += int(bad.sum())
        diag["n_ok"] += int(ok.sum())
        scales[:, g] = f16(s.astype(np.float32))
        offs[:, g] = f16(c.astype(np.float32))
        del A, B, Xg, Wg, Pg
    sc = np.abs(scales).reshape(-1)
    oc = np.abs(offs).reshape(-1)
    diag["median_scale"] = float(np.median(sc)) if sc.size else None
    ratio = oc / (sc + 1e-12)
    diag["median_off_over_scale"] = float(np.median(ratio)) if ratio.size else None
    return scales, offs, diag


def fit_scale_only(W: np.ndarray, P: np.ndarray, X: np.ndarray, G: int, sl: slice) -> np.ndarray:
    Xf = X[sl]
    out, inn = W.shape
    ng = n_groups(inn, G)
    scales = np.zeros((out, ng), dtype=np.float32)
    for g in range(ng):
        a0 = g * G
        a1 = min(inn, a0 + G)
        Xg = Xf[:, a0:a1]
        A = Xg @ W[:, a0:a1].T
        B = Xg @ P[:, a0:a1].T
        G11 = np.einsum("to,to->o", B, B)
        r1 = np.einsum("to,to->o", B, A)
        s = np.where(G11 > 1e-18, r1 / G11, 0.0)
        scales[:, g] = f16(s.astype(np.float32))
        del A, B, Xg
    return scales


def reconstruct_affine(P: np.ndarray, scales: np.ndarray, offs: np.ndarray | None, G: int) -> np.ndarray:
    rec = apply_k_scale(P, scales, G)
    if offs is not None:
        rec = rec + apply_k_offset(offs, P.shape[1], G)
    return rec


# ---------------------------------------------------------------------------
# rank-2 corrections
# ---------------------------------------------------------------------------

def rsvd(A: np.ndarray, k: int, p: int = 6, q: int = 1, rng=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Halko rSVD. A is (m,n) float32. Returns U(m,k), S(k), Vt(k,n) float32."""
    rng = np.random.default_rng(0) if rng is None else rng
    m, n = A.shape
    Omega = rng.standard_normal((n, k + p), dtype=np.float32)
    Y = A @ Omega
    for _ in range(q):
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y.astype(np.float64), mode="reduced")
    Q = Q.astype(np.float32)
    B = Q.T @ A
    Ub, S, Vt = np.linalg.svd(B.astype(np.float64), full_matrices=False)
    U = (Q @ Ub[:, :k].astype(np.float32)).astype(np.float32)
    return U, S[:k].astype(np.float32), Vt[:k].astype(np.float32)


def aw_rank2_factors(R: np.ndarray, rms_x: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-RMS-weighted rank-k of residual. Ranking-weighted, well-determined.

    R_w = R * rms[None,:]; R_w ≈ U S Vt; R ≈ U S (Vt/rms).
    Returns U (out,k), V (inn,k) so C1 = U @ V.T, and S.
    """
    rms = np.asarray(rms_x, dtype=np.float32)
    rms = np.where(rms > 1e-12, rms, 1.0)
    Rw = R * rms[None, :]
    U, S, Vt = rsvd(Rw, k=k, p=6, q=1)
    del Rw
    V = (Vt / rms[None, :]).T * S[None, :]
    return U, V.astype(np.float32), S


def fn_rank2_factors(
    Y_err_fit: np.ndarray, X_fit: np.ndarray, k: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Functional rank-k of residual output on the fit split.

    M = Y_err_fit.T = (out, Tf). SVD is exact (Tf=192).
    V is the min-norm solution of X_fit V ≈ right-vectors. UNDERDETERMINED
    when inn > Tf (5120 > 192). Magnitude of V is a plane estimate; the
    left subspace U is a ranking of output directions.
    """
    M = np.ascontiguousarray(Y_err_fit.T, dtype=np.float32)  # out × Tf
    # economy SVD via M M^T if out is huge? Tf=192, out up to 17408.
    # SVD of M is (out, 192) — numpy handles this.
    U, S, Vt = np.linalg.svd(M.astype(np.float64), full_matrices=False)
    energy = (S * S)
    tot = float(energy.sum()) + 1e-30
    spec = {str(i + 1): float(energy[: i + 1].sum() / tot) for i in range(min(16, S.size))}
    Uk = U[:, :k].astype(np.float32) * S[:k].astype(np.float32)  # out × k
    rhs = Vt[:k].T.astype(np.float32)  # Tf × k
    # min-norm lstsq: X_fit (Tf, inn) * V (inn, k) ≈ rhs
    V, residuals, rank, sv = np.linalg.lstsq(X_fit.astype(np.float64), rhs.astype(np.float64), rcond=None)
    info = {
        "spectrum_cum": spec,
        "s0": float(S[0]) if S.size else None,
        "s1": float(S[1]) if S.size > 1 else None,
        "lstsq_rank": int(rank),
        "lstsq_sv_min": float(sv.min()) if len(sv) else None,
        "lstsq_sv_max": float(sv.max()) if len(sv) else None,
        "underdetermined": bool(X_fit.shape[1] > X_fit.shape[0]),
        "rows_per_dim": float(X_fit.shape[0] / X_fit.shape[1]),
    }
    return Uk, V.astype(np.float32), S[:16].astype(np.float32), info


def apply_rank(X: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Y += (X @ V) @ U.T  without forming UV^T."""
    return (X @ V) @ U.T


def rank_bpw(out: int, inn: int, k: int) -> float:
    return (16.0 * k * (out + inn)) / (out * inn)


# ---------------------------------------------------------------------------
# C2 sparse exact
# ---------------------------------------------------------------------------

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


def index_bits_sorted(pos: np.ndarray, n: int) -> dict:
    if pos.size == 0:
        return {"rice": 0, "rice_k": 0, "fixed_log2n": 0, "elias_gamma": 0}
    pos = np.sort(pos.astype(np.int64))
    deltas = np.empty_like(pos)
    deltas[0] = pos[0] + 1
    deltas[1:] = np.maximum(pos[1:] - pos[:-1], 1)
    rice, rk = best_rice_bits(deltas)
    log2n = int(math.ceil(math.log2(max(n, 2))))
    elias = int((2 * np.floor(np.log2(np.maximum(deltas, 1))) + 1).sum())
    return {
        "rice": int(rice),
        "rice_k": int(rk),
        "fixed_log2n": int(pos.size * log2n),
        "elias_gamma": elias,
    }


def c2_entries(
    W: np.ndarray,
    R: np.ndarray,
    rms_x: np.ndarray | None,
    frac: float,
    how: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (flat_pos, values_f16) for top-frac residual entries."""
    n = int(R.size)
    k = max(1, int(math.ceil(n * frac))) if frac > 0 else 0
    if k == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32), {"k": 0, "frac": 0.0}
    if how == "absR":
        score = np.abs(R).reshape(-1)
    elif how == "act_weighted":
        if rms_x is None:
            score = np.abs(R).reshape(-1)
        else:
            score = (np.abs(R) * rms_x[None, :]).reshape(-1)
    else:
        raise ValueError(how)
    pos = np.argpartition(score, -k)[-k:]
    vals = f16(R.reshape(-1)[pos])
    cost = index_bits_sorted(pos, n)
    cost.update(
        {
            "k": int(k),
            "frac": float(k / n),
            "value_bits": int(k * 16),
            "how": how,
        }
    )
    cost["bpw_rice"] = (cost["value_bits"] + cost["rice"]) / n
    cost["bpw_fixed"] = (cost["value_bits"] + cost["fixed_log2n"]) / n
    return pos.astype(np.int64), vals, cost


def apply_entries(base: np.ndarray, pos: np.ndarray, vals: np.ndarray) -> np.ndarray:
    rec = base.copy()
    if pos.size:
        rec.reshape(-1)[pos] += vals
    return rec


def c2_columns(W: np.ndarray, R: np.ndarray, rms_x: np.ndarray | None, n_cols: int) -> tuple[np.ndarray, dict]:
    inn = R.shape[1]
    col_score = np.linalg.norm(R, axis=0)
    if rms_x is not None:
        col_score = col_score * rms_x
    n_cols = min(n_cols, inn)
    idx = np.argpartition(col_score, -n_cols)[-n_cols:]
    rec_add = np.zeros_like(R)
    rec_add[:, idx] = R[:, idx]
    out = R.shape[0]
    bits = n_cols * out * 16 + n_cols * math.ceil(math.log2(max(inn, 2)))
    return rec_add, {
        "n_cols": int(n_cols),
        "bpw": bits / R.size,
        "idx_head": [int(i) for i in idx[:8]],
        "score_head": [float(col_score[i]) for i in idx[:8]],
    }


def c2_rows(W: np.ndarray, R: np.ndarray, Y_err_fit: np.ndarray | None, n_rows: int) -> tuple[np.ndarray, dict]:
    out, inn = R.shape
    if Y_err_fit is not None:
        row_score = np.linalg.norm(Y_err_fit, axis=0)  # Y is (T,out)
    else:
        row_score = np.linalg.norm(R, axis=1)
    n_rows = min(n_rows, out)
    idx = np.argpartition(row_score, -n_rows)[-n_rows:]
    rec_add = np.zeros_like(R)
    rec_add[idx, :] = R[idx, :]
    bits = n_rows * inn * 16 + n_rows * math.ceil(math.log2(max(out, 2)))
    return rec_add, {
        "n_rows": int(n_rows),
        "bpw": bits / R.size,
        "idx_head": [int(i) for i in idx[:8]],
        "score_head": [float(row_score[i]) for i in idx[:8]],
        "has_3994": bool(3994 in set(int(i) for i in idx)) if out > 3994 else False,
    }


# ---------------------------------------------------------------------------
# amplification
# ---------------------------------------------------------------------------

def measure_amplification() -> dict:
    rms = []
    mean_norm = []
    col3994_energy = []
    col3994_rms = []
    ratios = []
    prev = None
    for L in range(64):
        H = load_hidden(L)
        nrm = np.linalg.norm(H.astype(np.float64), axis=1)
        mean_norm.append(float(nrm.mean()))
        rms.append(float(np.sqrt(np.mean(H.astype(np.float64) ** 2))))
        e = H[:, 3994].astype(np.float64)
        tot = float(np.sum(H.astype(np.float64) ** 2)) + 1e-30
        col3994_energy.append(float(np.sum(e * e) / tot))
        col3994_rms.append(float(np.sqrt(np.mean(e * e))))
        if prev is not None:
            pn = np.linalg.norm(prev.astype(np.float64), axis=1)
            ratios.append(float(np.mean(nrm / (pn + 1e-30))))
        prev = H
        del H
    return {
        "mean_hidden_norm": mean_norm,
        "rms_hidden": rms,
        "A_l_mean_norm_ratio": ratios,  # length 63, A_l = ||h_{l+1}||/||h_l||
        "col3994_energy_frac": col3994_energy,
        "col3994_rms": col3994_rms,
        "A_mean": float(np.mean(ratios)),
        "A_63_over_0": float(mean_norm[63] / (mean_norm[0] + 1e-30)),
        "contract_A63_cited": 2.6039602756500244,
        "A63_measured_norm_over_A0": float(mean_norm[63] / (mean_norm[0] + 1e-30)),
    }


# ---------------------------------------------------------------------------
# per-tensor ladder
# ---------------------------------------------------------------------------

ENTRY_FRACS = (3e-4, 1e-3, 3e-3)
COL_COUNTS = (1, 3, 8, 32)
ROW_COUNTS = (1, 3, 8, 32)


def run_ladder(W: np.ndarray, X: np.ndarray | None, tag: str, deep: bool) -> dict:
    t0 = time.time()
    out, inn = W.shape
    n = int(W.size)
    rec_out: dict = {
        "tag": tag,
        "shape": [int(out), int(inn)],
        "elements": n,
        "rows_per_dim": (float(192 / inn) if X is not None else None),
        "x_site": None if X is None else "provided",
        "rungs": {},
        "flats": {},
        "spectra": {},
        "c2": {},
        "diag": {},
    }

    Y = None
    rms_x = None
    if X is not None:
        if X.shape != (N_TOK, inn):
            raise RuntimeError(f"{tag} X shape {X.shape} != {(N_TOK, inn)}")
        Y = X @ W.T
        rms_x = np.sqrt(np.mean(X[FIT].astype(np.float64) ** 2, axis=0)).astype(np.float32)
        rec_out["x_rms_mean"] = float(rms_x.mean())
        rec_out["x_rms_max"] = float(rms_x.max())
        rec_out["x_rms_argmax"] = int(rms_x.argmax())
        xn = np.linalg.norm(X.astype(np.float64), axis=1)
        yn = np.linalg.norm(Y.astype(np.float64), axis=1)
        rec_out["local_amp_mean"] = float(np.mean(yn / (xn + 1e-30)))
        rec_out["local_amp_hold"] = float(np.mean(yn[HOLD] / (xn[HOLD] + 1e-30)))
        rec_out["y_hold_rms"] = float(np.sqrt(np.mean(Y[HOLD].astype(np.float64) ** 2)))

    def bill(name: str, rec: np.ndarray, bpw: float, extra: dict | None = None):
        d = score_pair(W, rec, X, Y)
        d["bpw"] = float(bpw)
        if extra:
            d.update(extra)
        rec_out["rungs"][name] = d
        hc = d.get("hold_output_cosine")
        log(
            f"  {tag:28s} {name:22s} bpw={bpw:7.4f} "
            f"wcos={d['weight_cosine']:.5f} hold={None if hc is None else f'{hc:.5f}'}"
        )
        return d

    # ----- flats (K-axis) -----
    flats = [
        ("bin_k_g128", lambda: binary_k(W, 128)[2], 1.0 + 16.0 / 128.0),
        ("bin_k_g64", lambda: binary_k(W, 64)[2], 1.0 + 16.0 / 64.0),
        ("bin_k_g32", lambda: binary_k(W, 32)[2], 1.0 + 16.0 / 32.0),
        ("q2_k_g256", lambda: uniform_k(W, 2, 256), 2.0 + 16.0 / 256.0),
        ("q2_k_g128", lambda: uniform_k(W, 2, 128), 2.0 + 16.0 / 128.0),
        ("q2_k_g64", lambda: uniform_k(W, 2, 64), 2.0 + 16.0 / 64.0),
        ("q3_k_g128", lambda: uniform_k(W, 3, 128), 3.0 + 16.0 / 128.0),
        ("q3_k_g64", lambda: uniform_k(W, 3, 64), 3.0 + 16.0 / 64.0),
        ("q4_k_g64", lambda: uniform_k(W, 4, 64), 4.0 + 16.0 / 64.0),
    ]
    if deep:
        rec_t, bpw_t, dens_t = ternary_k(W, 0.7, 128)
        rec_out["flats"]["ternary_t0.7_g128"] = score_pair(W, rec_t, X, Y)
        rec_out["flats"]["ternary_t0.7_g128"]["bpw"] = bpw_t
        rec_out["flats"]["ternary_t0.7_g128"]["density"] = dens_t
        del rec_t
    for name, fn, bpw in flats:
        rec = fn()
        d = score_pair(W, rec, X, Y)
        d["bpw"] = bpw
        rec_out["flats"][name] = d
        del rec

    # incumbent flat-C self-check only when deep or L0-sized
    if deep:
        rec = flat_c_binary(W, 128)
        d = score_pair(W, rec, X, Y)
        d["bpw"] = 1.0 + 16.0 / 128.0
        rec_out["flats"]["bin_flatC_g128"] = d
        del rec
        rec = uniform_flat_c(W, 4, 64)
        d = score_pair(W, rec, X, Y)
        d["bpw"] = 4.0 + 16.0 / 64.0
        rec_out["flats"]["q4_flatC_g64"] = d
        del rec

    # ----- B: binary K-axis g128 (primary cheap 1-bit) -----
    P, Sma, B, bpw_B = binary_k(W, G_HIER)
    bill("B_bin_g128", B, bpw_B, {"level": "B", "form": "sign*meanabs_k_g128"})

    # ----- B_pair: under-1 BPW -----
    Pp, Smap, Bp, bpw_Bp = pair_binary_k(W, G_HIER)
    bill("B_pair_g128", Bp, bpw_Bp, {"level": "B", "form": "pair_sign*meanabs_k_g128"})

    if X is None:
        rec_out["wall_s"] = time.time() - t0
        return rec_out

    # ----- same-bit act-scale refinement of B -----
    Sact = fit_scale_only(W, P, X, G_HIER, FIT)
    B_as = reconstruct_affine(P, Sact, None, G_HIER)
    bill("B_actscale", B_as, bpw_B, {"level": "B", "form": "sign*act_scale_k_g128", "extra_bpw": 0.0})

    # ----- C1: act scale + offset (justified structured) -----
    Sso, Cso, diag_so = fit_scale_offset(W, P, X, G_HIER, FIT)
    rec_out["diag"]["c1_scale_offset"] = diag_so
    BC1 = reconstruct_affine(P, Sso, Cso, G_HIER)
    bpw_C1 = bpw_B + 16.0 / G_HIER  # extra fp16 offset plane
    bill("B_C1_soff", BC1, bpw_C1, {"level": "B+C1", "form": "sign*(act_s)+act_offset", "c1_bpw": 16.0 / G_HIER})

    # pair + C1
    Ssop, Csop, diag_sop = fit_scale_offset(W, Pp, X, G_HIER, FIT)
    rec_out["diag"]["c1_pair_scale_offset"] = diag_sop
    BpC1 = reconstruct_affine(Pp, Ssop, Csop, G_HIER)
    bpw_pC1 = bpw_Bp + 16.0 / G_HIER
    bill("Bp_C1_soff", BpC1, bpw_pC1, {"level": "B+C1", "form": "pair_sign*(act_s)+act_offset"})

    # residual after B+C1
    R1 = W - BC1
    Y_bc1 = X @ BC1.T
    Y_err = Y - Y_bc1

    # functional spectrum of residual output (fit)
    _, _, Sfn, fn_info = fn_rank2_factors(Y_err[FIT], X[FIT], k=2)
    rec_out["spectra"]["fn_residual_after_BC1"] = fn_info
    rec_out["spectra"]["fn_S_head"] = [float(x) for x in Sfn[:8]]

    # also spectrum of residual after B only
    Y_b = X @ B.T
    _, _, _, fn_info_B = fn_rank2_factors((Y - Y_b)[FIT], X[FIT], k=2)
    rec_out["spectra"]["fn_residual_after_B"] = fn_info_B

    # ----- C1b: aw-rank2 on R1 (ranking-weighted, well determined) -----
    Uaw, Vaw, Saw = aw_rank2_factors(R1, rms_x, k=2)
    rec_out["spectra"]["aw_rank2_S"] = [float(x) for x in Saw]
    Y_aw = Y_bc1 + apply_rank(X, Uaw, Vaw)
    # reconstruct for weight metrics
    C_aw = Uaw @ Vaw.T
    BC1aw = BC1 + C_aw
    bpw_aw = bpw_C1 + rank_bpw(out, inn, 2)
    d = weight_metrics(W, BC1aw)
    d.update(output_metrics(Y, Y_aw))
    d["bpw"] = bpw_aw
    d["level"] = "B+C1+awr2"
    d["c1b_bpw"] = rank_bpw(out, inn, 2)
    rec_out["rungs"]["B_C1_awr2"] = d
    log(
        f"  {tag:28s} {'B_C1_awr2':22s} bpw={bpw_aw:7.4f} "
        f"wcos={d['weight_cosine']:.5f} hold={d['hold_output_cosine']:.5f}"
    )

    # ----- C1b-fn: functional rank2 (UNDERDETERMINED V) -----
    Ufn, Vfn, _, _ = fn_rank2_factors(Y_err[FIT], X[FIT], k=2)
    Y_fn = Y_bc1 + apply_rank(X, Ufn, Vfn)
    C_fn = Ufn @ Vfn.T
    BC1fn = BC1 + C_fn
    bpw_fn = bpw_C1 + rank_bpw(out, inn, 2)
    d = weight_metrics(W, BC1fn)
    d.update(output_metrics(Y, Y_fn))
    d["bpw"] = bpw_fn
    d["level"] = "B+C1+fnr2"
    d["underdetermined"] = True
    rec_out["rungs"]["B_C1_fnr2"] = d
    log(
        f"  {tag:28s} {'B_C1_fnr2':22s} bpw={bpw_fn:7.4f} "
        f"wcos={d['weight_cosine']:.5f} hold={d['hold_output_cosine']:.5f}"
    )

    # generalization of fit left-subspace onto hold residual
    M_hold = Y_err[HOLD].T.astype(np.float64)
    Uo = Ufn / (np.linalg.norm(Ufn, axis=0, keepdims=True) + 1e-30)
    num = float(np.sum((Uo.T.astype(np.float64) @ M_hold) ** 2))
    den = float(np.sum(M_hold ** 2)) + 1e-30
    rec_out["spectra"]["fn_U_hold_energy_frac"] = num / den

    # ----- C2 on residual after B+C1 (primary 3-level) -----
    # entries
    for frac in ENTRY_FRACS:
        for how in ("act_weighted", "absR"):
            pos, vals, cost = c2_entries(W, R1, rms_x, frac, how)
            rec = apply_entries(BC1, pos, vals)
            key = f"B_C1_e{frac:g}_{how}"
            bill(key, rec, bpw_C1 + cost["bpw_rice"], {"level": "B+C1+C2", "c2": cost})
            rec_out["c2"][key] = cost
            del rec
            # skip-C1: B + same entries of (W-B)
            if how == "act_weighted":
                Rb = W - B
                posb, valsb, costb = c2_entries(W, Rb, rms_x, frac, how)
                recb = apply_entries(B, posb, valsb)
                keyb = f"B_e{frac:g}_{how}"
                bill(keyb, recb, bpw_B + costb["bpw_rice"], {"level": "B+C2", "c2": costb})
                del recb, Rb

    # columns / rows (structured sparse)
    for nc in COL_COUNTS:
        add, meta = c2_columns(W, R1, rms_x, nc)
        rec = BC1 + add
        key = f"B_C1_cols{nc}"
        bill(key, rec, bpw_C1 + meta["bpw"], {"level": "B+C1+C2", "c2": meta})
        rec_out["c2"][key] = meta
        del rec, add
    for nr in ROW_COUNTS:
        add, meta = c2_rows(W, R1, Y_err[FIT], nr)
        rec = BC1 + add
        key = f"B_C1_rows{nr}"
        bill(key, rec, bpw_C1 + meta["bpw"], {"level": "B+C1+C2", "c2": meta})
        rec_out["c2"][key] = meta
        del rec, add

    # C2 on top of B+C1+awr2 at 1e-3 act-weighted (does rank2 leave a cheaper tail?)
    R2 = W - BC1aw
    pos, vals, cost = c2_entries(W, R2, rms_x, 1e-3, "act_weighted")
    rec = apply_entries(BC1aw, pos, vals)
    bill("B_C1_awr2_e0.001", rec, bpw_aw + cost["bpw_rice"], {"level": "B+C1+awr2+C2", "c2": cost})
    del rec, R2

    # pair-B + C1 + C2 1e-3 (under-1 base track)
    Rp = W - BpC1
    pos, vals, cost = c2_entries(W, Rp, rms_x, 1e-3, "act_weighted")
    rec = apply_entries(BpC1, pos, vals)
    bill("Bp_C1_e0.001", rec, bpw_pC1 + cost["bpw_rice"], {"level": "Bp+C1+C2", "c2": cost})
    del rec, Rp

    rec_out["wall_s"] = time.time() - t0
    rec_out["rss_gb"] = rss_gb()
    # free
    del P, Sma, B, Pp, Bp, B_as, BC1, BpC1, R1, Y, Y_bc1, Y_err, Y_aw, Y_fn
    del C_aw, C_fn, BC1aw, BC1fn, Uaw, Vaw, Ufn, Vfn
    gc.collect()
    return rec_out


# ---------------------------------------------------------------------------
# mixer proxies (labeled)
# ---------------------------------------------------------------------------

def deltanet_out_x(H: np.ndarray, Wqkv: np.ndarray, Wz: np.ndarray) -> np.ndarray:
    """X for out_proj: v * silu(z). NOT the recurrent mixer. PROXY."""
    Yqkv = H @ Wqkv.T  # (T, 10240) = Q 2048 | K 2048 | V 6144
    V = Yqkv[:, 4096:10240]
    Z = H @ Wz.T
    return (V * silu(Z)).astype(np.float32)


def gqa_o_x(H: np.ndarray, Wq: np.ndarray, Wv: np.ndarray) -> np.ndarray:
    """X for o_proj: repeat(v) * sigmoid(q_gate). NOT softmax mix. PROXY."""
    Yq = H @ Wq.T  # (T, 12288) = 24 * (q 256 + gate 256)
    Yv = H @ Wv.T  # (T, 1024) = 4 * 256
    T = H.shape[0]
    qg = Yq.reshape(T, 24, 2, 256)[:, :, 1, :]  # (T,24,256)
    v = Yv.reshape(T, 4, 256)
    v_rep = np.repeat(v, 6, axis=1)  # 4*6=24
    return (v_rep.reshape(T, 6144) * sigmoid(qg.reshape(T, 6144))).astype(np.float32)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def self_check() -> dict:
    """Reproduce known L0 gate numbers (flat-C production family)."""
    name = tname(0, "mlp.gate_proj.weight")
    W = load_tensor(name)
    X = load_hidden(0)
    Y = X @ W.T
    rec4 = uniform_flat_c(W, 4, 64)
    recb = flat_c_binary(W, 128)
    d4 = score_pair(W, rec4, X, Y)
    db = score_pair(W, recb, X, Y)
    # K-axis variants
    rec4k = uniform_k(W, 4, 64)
    recbk = binary_k(W, 128)[2]
    d4k = score_pair(W, rec4k, X, Y)
    dbk = score_pair(W, recbk, X, Y)
    out = {
        "tensor": name,
        "shape": list(W.shape),
        "q4_flatC_g64": d4,
        "bin_flatC_g128": db,
        "q4_k_g64": d4k,
        "bin_k_g128": dbk,
        "cited_q4_weight_cosine": 0.9941447925762601,
        "cited_q4_rel_l2": 0.10873951632127579,
        "cited_bin_weight_cosine": 0.7983613876884476,
        "q4_weight_cosine_abs_err_vs_cited": abs(d4["weight_cosine"] - 0.9941447925762601),
        "bin_weight_cosine_abs_err_vs_cited": abs(db["weight_cosine"] - 0.7983613876884476),
    }
    log(
        f"SELFCHECK L0 gate q4_flatC wcos={d4['weight_cosine']:.8f} "
        f"Δ={out['q4_weight_cosine_abs_err_vs_cited']:.3e} "
        f"bin_flatC wcos={db['weight_cosine']:.8f} "
        f"Δ={out['bin_weight_cosine_abs_err_vs_cited']:.3e}"
    )
    del W, X, Y, rec4, recb, rec4k, recbk
    gc.collect()
    return out


def identity() -> dict:
    cfg = json.loads((SRC / "config.json").read_text())
    tc = cfg["text_config"]
    cap = json.loads((CAP / "capture-result.json").read_text())
    g0 = json.loads(G0_MANIFEST.read_text())
    import hashlib

    def sha(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    return {
        "src": str(SRC),
        "config_model_type": cfg.get("model_type"),
        "text_model_type": tc.get("model_type"),
        "num_hidden_layers": tc.get("num_hidden_layers"),
        "hidden_size": tc.get("hidden_size"),
        "intermediate_size": tc.get("intermediate_size"),
        "vocab_size": tc.get("vocab_size"),
        "capture_status": cap.get("status"),
        "capture_sha256_self": cap.get("sha256_self"),
        "capture_file_sha256": sha(CAP / "capture-result.json"),
        "L00_sha256": sha(CAP / "hidden" / "L00.f32"),
        "g0_complete_physical_bpw": g0.get("complete_physical_bpw"),
        "g0_manifest_sha256": sha(G0_MANIFEST),
        "g0_source_weight_elements": g0.get("source_weight_elements"),
        "index_n_tensors": len(_WMAP),
        "python": os.popen("~/.grok-vision/bin/python -c 'import sys; print(sys.version)'").read().strip(),
        "numpy": np.__version__,
    }


def process_layer(layer: int, deep: bool, want: set[str]) -> list[dict]:
    rows = []
    H = load_hidden(layer)
    # MLP
    if "gate" in want:
        Wg = load_tensor(tname(layer, "mlp.gate_proj.weight"))
        rows.append(run_ladder(Wg, H, f"L{layer}.gate", deep))
        Wg_keep = Wg if "down" in want else None
        if Wg_keep is None:
            del Wg
            gc.collect()
    else:
        Wg_keep = None
    if "up" in want:
        Wu = load_tensor(tname(layer, "mlp.up_proj.weight"))
        rows.append(run_ladder(Wu, H, f"L{layer}.up", deep))
        Wu_keep = Wu if "down" in want else None
        if Wu_keep is None:
            del Wu
            gc.collect()
    else:
        Wu_keep = None
    if "down" in want:
        if Wg_keep is None:
            Wg_keep = load_tensor(tname(layer, "mlp.gate_proj.weight"))
        if Wu_keep is None:
            Wu_keep = load_tensor(tname(layer, "mlp.up_proj.weight"))
        Xd = silu(H @ Wg_keep.T) * (H @ Wu_keep.T)
        del Wg_keep, Wu_keep
        gc.collect()
        Wd = load_tensor(tname(layer, "mlp.down_proj.weight"))
        rows.append(run_ladder(Wd, Xd.astype(np.float32), f"L{layer}.down", deep))
        del Wd, Xd
        gc.collect()

    if is_gqa(layer):
        if "q" in want:
            Wq = load_tensor(tname(layer, "self_attn.q_proj.weight"))
            rows.append(run_ladder(Wq, H, f"L{layer}.q", deep))
            Wq_keep = Wq if "o" in want else None
            if Wq_keep is None:
                del Wq
                gc.collect()
        else:
            Wq_keep = None
        if "v" in want:
            Wv = load_tensor(tname(layer, "self_attn.v_proj.weight"))
            rows.append(run_ladder(Wv, H, f"L{layer}.v", deep))
            Wv_keep = Wv if "o" in want else None
            if Wv_keep is None:
                del Wv
                gc.collect()
        else:
            Wv_keep = None
        if "o" in want:
            if Wq_keep is None:
                Wq_keep = load_tensor(tname(layer, "self_attn.q_proj.weight"))
            if Wv_keep is None:
                Wv_keep = load_tensor(tname(layer, "self_attn.v_proj.weight"))
            Xo = gqa_o_x(H, Wq_keep, Wv_keep)
            del Wq_keep, Wv_keep
            gc.collect()
            Wo = load_tensor(tname(layer, "self_attn.o_proj.weight"))
            r = run_ladder(Wo, Xo, f"L{layer}.o", deep)
            r["x_site"] = "PROXY_gqa_repeat_v_times_sigmoid_qgate"
            rows.append(r)
            del Wo, Xo
            gc.collect()
        if "k" in want:
            Wk = load_tensor(tname(layer, "self_attn.k_proj.weight"))
            rows.append(run_ladder(Wk, H, f"L{layer}.k", deep))
            del Wk
            gc.collect()
    else:
        if "qkv" in want:
            Wqkv = load_tensor(tname(layer, "linear_attn.in_proj_qkv.weight"))
            rows.append(run_ladder(Wqkv, H, f"L{layer}.in_qkv", deep))
            Wqkv_keep = Wqkv if "out" in want else None
            if Wqkv_keep is None:
                del Wqkv
                gc.collect()
        else:
            Wqkv_keep = None
        if "out" in want:
            if Wqkv_keep is None:
                Wqkv_keep = load_tensor(tname(layer, "linear_attn.in_proj_qkv.weight"))
            Wz = load_tensor(tname(layer, "linear_attn.in_proj_z.weight"))
            Xo = deltanet_out_x(H, Wqkv_keep, Wz)
            del Wqkv_keep, Wz
            gc.collect()
            Wo = load_tensor(tname(layer, "linear_attn.out_proj.weight"))
            r = run_ladder(Wo, Xo, f"L{layer}.out", deep)
            r["x_site"] = "PROXY_dn_v_times_silu_z"
            rows.append(r)
            del Wo, Xo
            gc.collect()
    del H
    gc.collect()
    return rows


def summarize(rows: list[dict], amp: dict) -> dict:
    """Mass-weighted and product screens; pick best C2 per tensor; vs flat."""

    def hold(d):
        return d.get("hold_output_cosine")

    # per-tensor best hierarchy vs best flat at <= that BPW
    comparisons = []
    products = {k: 1.0 for k in (
        "B_bin_g128", "B_actscale", "B_C1_soff", "B_C1_awr2",
        "q3_k_g64", "q4_k_g64", "bin_k_g128", "q2_k_g64",
    )}
    product_n = {k: 0 for k in products}
    class_acc: dict[str, dict] = {}

    def cls_of(tag: str) -> str:
        if tag.endswith(".gate"):
            return "mlp.gate_proj"
        if tag.endswith(".up"):
            return "mlp.up_proj"
        if tag.endswith(".down"):
            return "mlp.down_proj"
        return "attn"

    for row in rows:
        if hold(row["rungs"].get("B_bin_g128", {})) is None:
            continue
        for k in products:
            src = row["rungs"] if k in row["rungs"] else row["flats"]
            if k in src and hold(src[k]) is not None:
                products[k] *= float(hold(src[k]))
                product_n[k] += 1
        # best hierarchical C2 under 1.6 BPW
        best_h = None
        for name, d in row["rungs"].items():
            if d.get("hold_output_cosine") is None:
                continue
            if best_h is None or (
                d["hold_output_cosine"], -d["bpw"]
            ) > (best_h[1]["hold_output_cosine"], -best_h[1]["bpw"]):
                best_h = (name, d)
        # best flat at each hierarchical rung
        rungs_cmp = {}
        for rname in ("B_bin_g128", "B_actscale", "B_C1_soff", "B_C1_awr2"):
            if rname not in row["rungs"]:
                continue
            rd = row["rungs"][rname]
            rh = hold(rd)
            rbpw = rd["bpw"]
            best_flat = None
            for fname, fd in row["flats"].items():
                if hold(fd) is None:
                    continue
                if fd["bpw"] <= rbpw + 1e-9:
                    if best_flat is None or hold(fd) > hold(best_flat[1]):
                        best_flat = (fname, fd)
            # also closest flat regardless of <=
            closest = None
            for fname, fd in row["flats"].items():
                if hold(fd) is None:
                    continue
                gap = abs(fd["bpw"] - rbpw)
                if closest is None or gap < closest[2]:
                    closest = (fname, fd, gap)
            rungs_cmp[rname] = {
                "hier_hold": rh,
                "hier_bpw": rbpw,
                "best_flat_le": None
                if best_flat is None
                else {
                    "name": best_flat[0],
                    "hold": hold(best_flat[1]),
                    "bpw": best_flat[1]["bpw"],
                    "hier_minus_flat": None if rh is None else rh - hold(best_flat[1]),
                },
                "closest_flat": None
                if closest is None
                else {
                    "name": closest[0],
                    "hold": hold(closest[1]),
                    "bpw": closest[1]["bpw"],
                    "hier_minus_flat": None if rh is None else rh - hold(closest[1]),
                },
            }
        # C2 rungs vs flats
        for rname, rd in row["rungs"].items():
            if not rname.startswith("B_C1_e") and not rname.startswith("B_C1_cols") and not rname.startswith("B_C1_rows"):
                continue
            if hold(rd) is None:
                continue
            rbpw = rd["bpw"]
            best_flat = None
            for fname, fd in row["flats"].items():
                if hold(fd) is None:
                    continue
                if fd["bpw"] <= rbpw + 1e-9:
                    if best_flat is None or hold(fd) > hold(best_flat[1]):
                        best_flat = (fname, fd)
            rungs_cmp[rname] = {
                "hier_hold": hold(rd),
                "hier_bpw": rbpw,
                "best_flat_le": None
                if best_flat is None
                else {
                    "name": best_flat[0],
                    "hold": hold(best_flat[1]),
                    "bpw": best_flat[1]["bpw"],
                    "hier_minus_flat": hold(rd) - hold(best_flat[1]),
                },
            }
        comparisons.append({"tag": row["tag"], "cls": cls_of(row["tag"]), "rungs": rungs_cmp})

        c = cls_of(row["tag"])
        class_acc.setdefault(c, []).append(row)

    # element-weighted class means for key rungs
    class_means = {}
    key_rungs = [
        "B_bin_g128",
        "B_pair_g128",
        "B_actscale",
        "B_C1_soff",
        "B_C1_awr2",
        "B_C1_fnr2",
        "B_C1_e0.001_act_weighted",
        "B_e0.001_act_weighted",
        "Bp_C1_soff",
        "Bp_C1_e0.001",
        "B_C1_awr2_e0.001",
    ]
    for c, lst in class_acc.items():
        elems = [r["elements"] for r in lst]
        te = sum(elems)
        cm = {"n": len(lst), "elements": te}
        for kr in key_rungs:
            hs, bs, ws = [], [], []
            for r, e in zip(lst, elems):
                d = r["rungs"].get(kr)
                if not d or d.get("hold_output_cosine") is None:
                    continue
                hs.append((d["hold_output_cosine"], e))
                bs.append((d["bpw"], e))
                ws.append((d["weight_cosine"], e))
            if hs:
                cm[kr] = {
                    "hold_elem": sum(a * e for a, e in hs) / sum(e for _, e in hs),
                    "bpw_elem": sum(a * e for a, e in bs) / sum(e for _, e in bs),
                    "wcos_elem": sum(a * e for a, e in ws) / sum(e for _, e in ws),
                    "n": len(hs),
                }
        # flats
        for fk in ("bin_k_g128", "bin_k_g64", "bin_k_g32", "q2_k_g64", "q3_k_g64", "q4_k_g64", "ternary_t0.7_g128"):
            hs, bs = [], []
            for r, e in zip(lst, elems):
                d = r["flats"].get(fk)
                if not d or d.get("hold_output_cosine") is None:
                    continue
                hs.append((d["hold_output_cosine"], e))
                bs.append((d["bpw"], e))
            if hs:
                cm[f"flat:{fk}"] = {
                    "hold_elem": sum(a * e for a, e in hs) / sum(e for _, e in hs),
                    "bpw_elem": sum(a * e for a, e in bs) / sum(e for _, e in bs),
                    "n": len(hs),
                }
        class_means[c] = cm

    # hierarchy vs flat win count at B+C1 (1.25 BPW vs bin_g64 1.25)
    wins = {"B_C1_soff_vs_bin_g64": {"win": 0, "lose": 0, "tie": 0, "deltas": []}}
    for row in rows:
        h = row["rungs"].get("B_C1_soff")
        f = row["flats"].get("bin_k_g64")
        if not h or not f or hold(h) is None or hold(f) is None:
            continue
        delta = hold(h) - hold(f)
        wins["B_C1_soff_vs_bin_g64"]["deltas"].append({"tag": row["tag"], "delta": delta})
        if delta > 5e-4:
            wins["B_C1_soff_vs_bin_g64"]["win"] += 1
        elif delta < -5e-4:
            wins["B_C1_soff_vs_bin_g64"]["lose"] += 1
        else:
            wins["B_C1_soff_vs_bin_g64"]["tie"] += 1

    # more matchups
    matchups = [
        ("B_actscale", "bin_k_g128", 5e-4),
        ("B_C1_e0.001_act_weighted", "bin_k_g32", 5e-4),
        ("B_C1_e0.001_act_weighted", "q2_k_g256", 5e-4),
        ("B_C1_soff", "q2_k_g256", 5e-4),
        ("B_bin_g128", "bin_k_g128", 5e-4),
        ("B_C1_awr2", "bin_k_g64", 5e-4),
    ]
    for hr, fr, thr in matchups:
        key = f"{hr}_vs_{fr}"
        acc = {"win": 0, "lose": 0, "tie": 0, "deltas": []}
        for row in rows:
            h = row["rungs"].get(hr) or row["flats"].get(hr)
            f = row["flats"].get(fr) or row["rungs"].get(fr)
            if not h or not f or hold(h) is None or hold(f) is None:
                continue
            delta = hold(h) - hold(f)
            acc["deltas"].append({"tag": row["tag"], "delta": delta, "h_bpw": h["bpw"], "f_bpw": f["bpw"]})
            if delta > thr:
                acc["win"] += 1
            elif delta < -thr:
                acc["lose"] += 1
            else:
                acc["tie"] += 1
        wins[key] = acc

    # chain projection using write tensors (down, out, o)
    write_q = []
    for row in rows:
        if not (row["tag"].endswith(".down") or row["tag"].endswith(".out") or row["tag"].endswith(".o")):
            continue
        layer = int(row["tag"].split(".")[0][1:])
        q = {}
        for kr in ("B_C1_soff", "B_C1_e0.001_act_weighted", "B_bin_g128"):
            d = row["rungs"].get(kr)
            if d and d.get("hold_output_rel_l2") is not None:
                q[kr] = d["hold_output_rel_l2"]
        for kr in ("q3_k_g64", "q4_k_g64"):
            d = row["flats"].get(kr)
            if d and d.get("hold_output_rel_l2") is not None:
                q[kr] = d["hold_output_rel_l2"]
        write_q.append({"tag": row["tag"], "layer": layer, "q": q})

    # A_l from amp, project e_final = sum_l q_l * prod_{k=l}^{L-1} A_k
    A = amp.get("A_l_mean_norm_ratio") or []
    chain = {}
    if A and write_q:
        # use measured layers only, treat A_cum from that layer to 63
        for kr in ("B_bin_g128", "B_C1_soff", "B_C1_e0.001_act_weighted", "q3_k_g64", "q4_k_g64"):
            acc = 0.0
            nterm = 0
            for w in write_q:
                if kr not in w["q"]:
                    continue
                L = w["layer"]
                gain = 1.0
                for k in range(L, 63):
                    if k < len(A):
                        gain *= A[k]
                acc += w["q"][kr] * gain
                nterm += 1
            chain[kr] = {"sum_q_times_gain": acc, "n": nterm}

    return {
        "n_scored": sum(1 for r in rows if hold(r["rungs"].get("B_bin_g128", {})) is not None),
        "products": {k: {"product": products[k], "n": product_n[k]} for k in products},
        "class_means": class_means,
        "wins": {k: {kk: vv for kk, vv in v.items() if kk != "deltas"} | {"n_deltas": len(v.get("deltas", [])), "mean_delta": (float(np.mean([d["delta"] for d in v["deltas"]])) if v.get("deltas") else None)} for k, v in wins.items()},
        "win_deltas": {k: v.get("deltas") for k, v in wins.items()},
        "comparisons": comparisons,
        "write_injection": write_q,
        "chain_projection": chain,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--skip-amp", action="store_true")
    args = ap.parse_args()
    if OUT_LOG.exists():
        OUT_LOG.unlink()
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    t_all = time.time()
    ident = identity()
    log(f"identity {json.dumps({k: ident[k] for k in ('config_model_type','hidden_size','intermediate_size','g0_complete_physical_bpw','g0_manifest_sha256','capture_sha256_self')})}")

    chk = self_check()

    amp = {}
    if not args.skip_amp:
        log("amplification pass")
        amp = measure_amplification()
        log(
            f"amp A_mean={amp['A_mean']:.4f} A63/A0_norm={amp['A_63_over_0']:.4f} "
            f"L63_col3994_e={amp['col3994_energy_frac'][63]:.4f} L0_col3994_e={amp['col3994_energy_frac'][0]:.4f}"
        )

    rows: list[dict] = []
    if args.probe:
        # L0 gate, L0 down, L3 q, L63 down, L0 out — enough to see the ladder
        plan = [
            (0, True, {"gate", "down", "out"}),
            (3, True, {"q"}),
            (63, True, {"down"}),
        ]
    else:
        plan = []
        for L in LAYERS_FULL:
            if is_gqa(L):
                want = {"gate", "up", "down", "q", "v", "o"}
            else:
                want = {"gate", "up", "down", "qkv", "out"}
            plan.append((L, True, want))

    for L, deep, want in plan:
        log(f"LAYER {L} want={sorted(want)} gqa={is_gqa(L)}")
        rows.extend(process_layer(L, deep, want))
        # incremental save
        payload = {
            "identity": ident,
            "self_check": chk,
            "amplification": amp,
            "rows": rows,
            "probe": bool(args.probe),
            "wall_s": time.time() - t_all,
            "rss_max_gb": rss_gb(),
        }
        OUT_JSON.write_text(json.dumps(payload))

    summary = summarize(rows, amp)
    payload = {
        "identity": ident,
        "self_check": chk,
        "amplification": amp,
        "rows": rows,
        "summary": summary,
        "probe": bool(args.probe),
        "wall_s": time.time() - t_all,
        "rss_max_gb": rss_gb(),
        "method": {
            "fit": "tokens 0:192",
            "hold": "tokens 192:256",
            "B": "K-axis sign * group mean-abs, G=128 (also pair-sign 0.5 b/w)",
            "C1": "per-group act-fitted scale (replaces mean-abs) + act-fitted offset; optional aw-rank2 / fn-rank2",
            "C2": "exact f16 residual at functionally selected entries/cols/rows",
            "grouping": "per-output-row groups along K (native GEMV walk)",
            "fn_rank2": "UNDERDETERMINED V (192 x inn lstsq); U is ranking",
            "aw_rank2": "column-RMS ranking-weighted rSVD; well-determined",
            "down_X": "reconstructed silu(H Wg.T)*(H Wu.T), not captured SwiGLU",
            "out_X": "mixer proxy, not recurrent/softmax mix",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    log(f"DONE wall={time.time()-t_all:.1f}s rss_max={rss_gb():.3f}G n_rows={len(rows)} -> {OUT_JSON}")
    # print compact summary
    print(json.dumps(summary.get("class_means", {}), indent=2)[:4000])
    print("WINS", json.dumps(summary.get("wins", {}), indent=2)[:3000])
    print("PRODUCTS", json.dumps(summary.get("products", {}), indent=2))
    print("CHAIN", json.dumps(summary.get("chain_projection", {}), indent=2))


if __name__ == "__main__":
    main()

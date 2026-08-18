#!/usr/bin/env python3
"""Generative regime: shared G + per-site z_l, scored on LAYER FUNCTION.

Not the falsified family (per-tensor low-rank + quantized residual of W).
CPU only. Real BF16 + real 256-token capture. No GPU, no resident touch.

Writes /tmp/g1_generative.json incrementally.
"""
from __future__ import annotations

import json
import math
import os
import resource
import struct
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

MODEL_DIR = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
)
CAPTURE_DIR = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
)
OUT = Path(os.environ.get("OUT", "/tmp/g1_generative.json"))
LOG = Path(os.environ.get("LOG", "/tmp/g1_generative.log"))

N_SOURCE = 26_895_998_464
N_MLP = 17_112_760_320
N_ATTN = 7_237_795_840
N_TAB = 2_542_796_800
N_TOKENS = 256
HIDDEN = 5120
INTERMEDIATE = 17408
FIT_N = 192
HOLD_N = 64
PROMPT_FIT_N = 185
PROMPT_HOLD_N = 71
PAD = 32768  # FWHT / FFT pad (next pow2 that covers 17408)
HEADER_BYTES = 256
ALIGN = 256

LAYERS_MLP = (0, 3, 15, 31, 47, 58, 63)
LAYERS_GQA = (3, 15, 31, 63)
LAYERS_DN = (0, 16, 32, 48)
RANKS = (8, 16, 32, 64, 96, 128, 160, 192)
HAD_RANKS = (1, 2, 4, 8)
SEED_TRIALS = 16


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')} rss={rss_gb():.3f}GiB] {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def dump(obj) -> None:
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=2))
    tmp.replace(OUT)


def cosine(a, b) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(left @ right)
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    if den <= 1e-12:
        return 1.0 if num == 0.0 else 0.0
    return num / den


def rel_l2(ref, hat) -> float:
    r = np.asarray(ref, dtype=np.float64).reshape(-1)
    h = np.asarray(hat, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(r))
    if n <= 1e-12:
        return 0.0
    return float(np.linalg.norm(r - h) / n)


def silu(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def bill_bytes(payload: int, header: int = HEADER_BYTES) -> int:
    return align_up(header + payload)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

_HEADER_CACHE: dict[Path, dict] = {}
_WEIGHT_MAP: dict[str, str] | None = None


def load_weight_map() -> dict[str, str]:
    global _WEIGHT_MAP
    if _WEIGHT_MAP is None:
        idx = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
        _WEIGHT_MAP = dict(idx["weight_map"])
    return _WEIGHT_MAP


def load_tensor(name: str) -> np.ndarray:
    weight_map = load_weight_map()
    shard = MODEL_DIR / weight_map[name]
    if shard not in _HEADER_CACHE:
        with shard.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            _HEADER_CACHE[shard] = json.loads(fh.read(n))
    info = _HEADER_CACHE[shard][name]
    dtype = info.get("dtype", "BF16")
    shape = tuple(int(x) for x in info["shape"])
    lo, hi = info["data_offsets"]
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        fh.seek(8 + n + lo)
        raw = fh.read(hi - lo)
    if dtype not in ("BF16", "BFLOAT16"):
        raise RuntimeError(f"{name} dtype {dtype}")
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return np.ascontiguousarray(u32.view(np.float32).reshape(shape))


def load_hidden(layer: int) -> np.ndarray:
    path = CAPTURE_DIR / "hidden" / f"L{layer:02d}.f32"
    x = np.fromfile(path, dtype=np.float32)
    if x.size != N_TOKENS * HIDDEN:
        raise RuntimeError(f"{path} size {x.size}")
    return np.ascontiguousarray(x.reshape(N_TOKENS, HIDDEN), dtype=np.float32)


def tensor_name(layer: int, suffix: str) -> str:
    return f"language_model.model.layers.{layer}.{suffix}"


# ---------------------------------------------------------------------------
# splits / metrics
# ---------------------------------------------------------------------------

def split_last64(X: np.ndarray):
    return X[:FIT_N], X[FIT_N:]


def split_prompt(X: np.ndarray):
    return X[:PROMPT_FIT_N], X[PROMPT_FIT_N:]


def output_scores(W: np.ndarray, X: np.ndarray, Yhat: np.ndarray) -> dict:
    Y = X @ W.T
    return {
        "cosine": cosine(Y, Yhat),
        "rel_l2": rel_l2(Y, Yhat),
        "y_rms": float(np.sqrt(np.mean(np.square(Y, dtype=np.float64)))),
        "yhat_rms": float(np.sqrt(np.mean(np.square(Yhat, dtype=np.float64)))),
    }


def weight_cosine_from_ZV(W: np.ndarray, Z: np.ndarray, V: np.ndarray) -> dict:
    # Ŵ = Z V.T, V columns orthonormal
    wv = W @ V
    inner = float(np.einsum("ij,ij->", wv, Z, dtype=np.float64))
    nw = float(np.linalg.norm(W.reshape(-1).astype(np.float64)))
    nh = float(np.linalg.norm(Z.reshape(-1).astype(np.float64)))  # ||Z|| = ||Ŵ|| if V ortho
    return {
        "weight_cosine": (inner / (nw * nh)) if nw * nh > 1e-12 else 0.0,
        "weight_rel_l2_ortho_proxy": float(math.sqrt(max(0.0, 2.0 - 2.0 * (inner / (nw * nh) if nw * nh > 1e-12 else 0.0)))),
        "w_rms": nw / math.sqrt(W.size),
    }


# ---------------------------------------------------------------------------
# structured generators
# ---------------------------------------------------------------------------

def fwht_unnormalized(a: np.ndarray) -> np.ndarray:
    """In-place-style unnormalized Sylvester FWHT on last axis. n power of 2."""
    x = np.array(a, dtype=np.float32, copy=True)
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"fwht n={n} not pow2")
    h = 1
    lead = int(np.prod(x.shape[:-1]))
    x = x.reshape(lead, n)
    while h < n:
        x = x.reshape(lead, n // (2 * h), 2, h)
        u = x[:, :, 0, :].copy()
        v = x[:, :, 1, :]
        x[:, :, 0, :] = u + v
        x[:, :, 1, :] = u - v
        x = x.reshape(lead, n)
        h *= 2
    return x.reshape(a.shape)


def hadamard_signs(rows: np.ndarray, k: int) -> np.ndarray:
    j = np.arange(k, dtype=np.int64)
    bits = np.bitwise_count(rows.astype(np.int64)[:, None] & j[None, :])
    return np.where(bits & 1, np.float32(-1.0), np.float32(1.0))


def apply_Wh(W: np.ndarray, vec: np.ndarray, bs: int = 256) -> np.ndarray:
    """(W ⊙ H) @ vec, H Sylvester cropped."""
    m, k = W.shape
    out = np.empty(m, dtype=np.float32)
    for i0 in range(0, m, bs):
        i1 = min(m, i0 + bs)
        H = hadamard_signs(np.arange(i0, i1), k)
        out[i0:i1] = (W[i0:i1] * H) @ vec
    return out


def apply_Wh_T(W: np.ndarray, vec: np.ndarray, bs: int = 256) -> np.ndarray:
    """(W ⊙ H).T @ vec"""
    m, k = W.shape
    acc = np.zeros(k, dtype=np.float64)
    for i0 in range(0, m, bs):
        i1 = min(m, i0 + bs)
        H = hadamard_signs(np.arange(i0, i1), k)
        acc += ((W[i0:i1] * H).T @ vec[i0:i1]).astype(np.float64)
    return acc.astype(np.float32)


def hadamard_factors(W: np.ndarray, rank: int, iters: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Rank-r of (W ⊙ H) via sequential power-method deflation. A (M,r), B (r,K)."""
    m, k = W.shape
    A = np.zeros((m, rank), dtype=np.float32)
    B = np.zeros((rank, k), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for r in range(rank):
        b = rng.standard_normal(k).astype(np.float32)
        b /= float(np.linalg.norm(b)) + 1e-12
        a = np.zeros(m, dtype=np.float32)
        for _ in range(iters):
            a = apply_Wh(W, b)
            if r:
                a -= A[:, :r] @ (B[:r] @ b)
            b = apply_Wh_T(W, a)
            if r:
                b -= B[:r].T @ (A[:, :r].T @ a)
            nrm = float(np.linalg.norm(b))
            if nrm <= 1e-12:
                break
            b /= nrm
        a = apply_Wh(W, b)
        if r:
            a -= A[:, :r] @ (B[:r] @ b)
        A[:, r] = a
        B[r] = b
    return A, B


def apply_hadamard_factors(X: np.ndarray, A: np.ndarray, B: np.ndarray, pad: int = PAD) -> np.ndarray:
    """y = sum_r A[:,r] ⊙ (H @ pad(B[r] ⊙ x))[:M]"""
    t, k = X.shape
    m = A.shape[0]
    y = np.zeros((t, m), dtype=np.float32)
    for r in range(A.shape[1]):
        u = np.zeros((t, pad), dtype=np.float32)
        u[:, :k] = X * B[r]
        u = fwht_unnormalized(u)
        y += u[:, :m] * A[:, r]
    return y


def rademacher_row(i: int, k: int, seed: int) -> np.ndarray:
    z = (np.uint64(seed) ^ (np.uint64(i) * np.uint64(0x9E3779B97F4A7C15))) + np.arange(k, dtype=np.uint64) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return np.where((z >> np.uint64(63)) == 0, np.float32(1.0), np.float32(-1.0))


def apply_Gh(W: np.ndarray, vec: np.ndarray, seed: int, bs: int = 256) -> np.ndarray:
    m, k = W.shape
    out = np.empty(m, dtype=np.float32)
    for i0 in range(0, m, bs):
        i1 = min(m, i0 + bs)
        G = np.stack([rademacher_row(i, k, seed) for i in range(i0, i1)], axis=0)
        out[i0:i1] = (W[i0:i1] * G) @ vec
    return out


def apply_Gh_T(W: np.ndarray, vec: np.ndarray, seed: int, bs: int = 256) -> np.ndarray:
    m, k = W.shape
    acc = np.zeros(k, dtype=np.float64)
    for i0 in range(0, m, bs):
        i1 = min(m, i0 + bs)
        G = np.stack([rademacher_row(i, k, seed) for i in range(i0, i1)], axis=0)
        acc += ((W[i0:i1] * G).T @ vec[i0:i1]).astype(np.float64)
    return acc.astype(np.float32)


def prng_rank1(W: np.ndarray, seed: int, iters: int = 5) -> tuple[np.ndarray, np.ndarray]:
    m, k = W.shape
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(k).astype(np.float32)
    b /= float(np.linalg.norm(b)) + 1e-12
    a = np.zeros(m, dtype=np.float32)
    for _ in range(iters):
        a = apply_Gh(W, b, seed)
        b = apply_Gh_T(W, a, seed)
        nrm = float(np.linalg.norm(b))
        if nrm <= 1e-12:
            break
        b /= nrm
    a = apply_Gh(W, b, seed)
    return a, b


def apply_prng_sandwich(X: np.ndarray, a: np.ndarray, b: np.ndarray, seed: int) -> np.ndarray:
    """y_i = a_i * <G_i, b ⊙ x>  — no fast transform; used only for quality."""
    t, k = X.shape
    m = a.shape[0]
    xb = X * b
    y = np.empty((t, m), dtype=np.float32)
    for i in range(m):
        g = rademacher_row(i, k, seed)
        y[:, i] = (xb @ g) * a[i]
    return y


def next_pow2(n: int) -> int:
    p = 1
    while p < int(n):
        p *= 2
    return p


def circulant_rowscale_fit(W: np.ndarray, n_fft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Ŵ[i,j] = a[i] * c[(j - (i % n_fft)) % n_fft] cropped to K.
    c from mean row (padded); a LS per row."""
    m, k = W.shape
    if n_fft is None:
        n_fft = next_pow2(k)
    c = np.zeros(n_fft, dtype=np.float32)
    c[:k] = W.mean(axis=0)
    # build one circulant action template via FFT of c
    cf = np.fft.rfft(c, n=n_fft)
    a = np.empty(m, dtype=np.float32)
    js = np.arange(k)
    for i in range(m):
        # row of circulant: c[(j - i) mod n]
        shift = i % n_fft
        # c_row[j] = c[(j - shift) % n_fft]
        # generate via roll
        crow = np.roll(c, shift)[:k]
        denom = float(np.dot(crow, crow)) + 1e-12
        a[i] = float(np.dot(W[i], crow)) / denom
    return a, c


def apply_circulant_rowscale(X: np.ndarray, a: np.ndarray, c: np.ndarray, n_fft: int | None = None) -> np.ndarray:
    t, k = X.shape
    m = a.shape[0]
    if n_fft is None:
        n_fft = int(c.shape[0])
    xp = np.zeros((t, n_fft), dtype=np.float32)
    xp[:, :k] = X
    xf = np.fft.rfft(xp, n=n_fft, axis=1)
    cf = np.fft.rfft(c, n=n_fft)
    conv = np.fft.irfft(xf * cf, n=n_fft, axis=1).astype(np.float32)
    # y[i] = a[i] * conv[..., i % n_fft]  but conv is on input, this is c⋆x not shift-per-row
    # Honest apply of Ŵ[i,j] = a[i] * c[(j - i) % n]:
    # y[i] = a[i] * sum_j c[(j-i)%n] x[j] = a[i] * (c_flipped ⋆ x)[i]
    # Using circular corr: y_full = irfft(conj(cf)*xf) then y[i]=a[i]*y_full[i]
    corr = np.fft.irfft(np.conjugate(cf) * xf, n=n_fft, axis=1).astype(np.float32)
    idx = np.arange(m) % n_fft
    return a[None, :] * corr[:, idx]


def fourier_multiplier_fit(W: np.ndarray, n_fft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """y = a ⊙ irfft(λ ⊙ rfft(pad(x))), λ from mean-row spectrum, a LS."""
    m, k = W.shape
    if n_fft is None:
        n_fft = next_pow2(k)
    mean_row = np.zeros(n_fft, dtype=np.float32)
    mean_row[:k] = W.mean(axis=0)
    lam = np.fft.rfft(mean_row, n=n_fft)
    # C[i,j] independent of i in this family (same circulant for every row) → rank-1-ish
    # Use: Ŵ[i] = a[i] * irfft(λ)[:k]  which IS rank-1. Still a structured baseline.
    basis = np.fft.irfft(lam, n=n_fft).astype(np.float32)[:k]
    denom = float(np.dot(basis, basis)) + 1e-12
    a = (W @ basis) / denom
    return a.astype(np.float32), lam


def apply_fourier_multiplier(X: np.ndarray, a: np.ndarray, lam: np.ndarray, n_fft: int | None = None) -> np.ndarray:
    t, k = X.shape
    if n_fft is None:
        n_fft = (lam.shape[0] - 1) * 2
    xp = np.zeros((t, n_fft), dtype=np.float32)
    xp[:, :k] = X
    yk = np.fft.irfft(np.fft.rfft(xp, n=n_fft, axis=1) * lam, n=n_fft, axis=1).astype(np.float32)
    # rank-1 structured: y = (yk[:,:k] @ 1-vector?) no: Ŵ = a outer basis, y = (X @ basis) * a
    basis = np.fft.irfft(lam, n=n_fft).astype(np.float32)[:k]
    coeff = X @ basis
    return coeff[:, None] * a[None, :]


# ---------------------------------------------------------------------------
# subspace
# ---------------------------------------------------------------------------

def thin_right_basis(X: np.ndarray, rank: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """V (D,r) orthonormal right basis of X (T,D). Exact thin SVD if min(T,D)<=384 else rSVD."""
    Xc = np.ascontiguousarray(X, dtype=np.float32)
    t, d = Xc.shape
    r = min(int(rank), t, d)
    if min(t, d) <= 384:
        _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        return vt[:r].T.copy(), s.astype(np.float64)
    p = min(16, max(0, min(t, d) - r))
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((d, r + p)).astype(np.float32)
    y = Xc @ omega
    qf, _ = np.linalg.qr(y, mode="reduced")
    y = Xc @ (Xc.T @ qf)
    qf, _ = np.linalg.qr(y, mode="reduced")
    b = qf.T @ Xc
    _u, s, vt = np.linalg.svd(b, full_matrices=False)
    return vt[:r].T.copy(), s.astype(np.float64)


def energy_in_V(X: np.ndarray, V: np.ndarray) -> float:
    num = float(np.square(X @ V, dtype=np.float64).sum())
    den = float(np.square(X, dtype=np.float64).sum())
    if den <= 1e-18:
        return 1.0
    return num / den


def subspace_overlap(Va: np.ndarray, Vb: np.ndarray) -> float:
    r = min(Va.shape[1], Vb.shape[1])
    m = Va[:, :r].T @ Vb[:, :r]
    return float(np.square(m, dtype=np.float64).sum()) / float(r)


def rsvd_right(W: np.ndarray, rank: int, seed: int = 0, q: int = 1) -> np.ndarray:
    """Orthonormal right basis of top-r of W (weight space)."""
    m, k = W.shape
    r = min(int(rank), m, k)
    p = min(16, max(0, min(m, k) - r))
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((k, r + p)).astype(np.float32)
    y = W @ omega
    qf, _ = np.linalg.qr(y, mode="reduced")
    for _ in range(q):
        y = W @ (W.T @ qf)
        qf, _ = np.linalg.qr(y, mode="reduced")
    b = qf.T @ W
    _u, _s, vt = np.linalg.svd(b, full_matrices=False)
    return vt[:r].T.copy()


def apply_ZV(X: np.ndarray, Z: np.ndarray, V: np.ndarray) -> np.ndarray:
    # y = Z @ (V.T @ x)
    return (X @ V) @ Z.T


def uniform_q4(W: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    gsz = 64
    groups = math.ceil(flat.size / gsz)
    pad = groups * gsz - flat.size
    padded = np.pad(flat, (0, pad)) if pad else flat
    padded = padded.reshape(groups, gsz)
    bound = 7.0
    scales = (np.max(np.abs(padded), axis=1) / bound).astype(np.float16).astype(np.float32)
    den = np.where(scales > 0.0, scales, 1.0)
    q = np.rint(padded / den[:, None]).clip(-bound, bound)
    recon = (q.astype(np.float32) * scales[:, None]).reshape(-1)[: flat.size]
    return recon.reshape(W.shape)


def binary_recon(W: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    gsz = 128
    groups = math.ceil(flat.size / gsz)
    pad = groups * gsz - flat.size
    padded = np.pad(flat, (0, pad)) if pad else flat
    padded = padded.reshape(groups, gsz)
    scales = np.mean(np.abs(padded), axis=1, dtype=np.float64).astype(np.float16).astype(np.float32)
    signs = np.where(padded >= 0.0, 1.0, -1.0).astype(np.float32)
    return (signs * scales[:, None]).reshape(-1)[: flat.size].reshape(W.shape)


def q4_bytes(rows: int, cols: int) -> int:
    groups = math.ceil(cols / 64)
    return rows * groups * 34


# ---------------------------------------------------------------------------
# phase 1 — activation geometry (no W)
# ---------------------------------------------------------------------------

def phase_activation(result: dict) -> None:
    log("PHASE1 activation geometry")
    xs = [load_hidden(i) for i in range(64)]
    Xall = np.stack(xs, axis=0)  # 64 x 256 x 5120
    result["activation"] = {
        "n_layers": 64,
        "shape_per_layer": [N_TOKENS, HIDDEN],
        "bytes_all": int(Xall.nbytes),
        "capture_schema": "hawking.ascension.qwen38_bf16_post_swiglu_activation_capture.v1",
        "capture_status": "CAPTURED_REAL_BF16_POST_NORM_HIDDEN",
        "prompt_token_cuts": [0, 57, 117, 185, 246, 256],
        "note": "RANKS a sharp set. Does not estimate magnitudes. mixer_x never captured.",
    }
    # per-layer spectra on full 256 and on fit/hold
    per = []
    Vs_full = []
    Vs_fit = []
    for i in range(64):
        X = xs[i]
        _u, s_full, vt_full = np.linalg.svd(X, full_matrices=False)
        e = np.square(s_full, dtype=np.float64)
        e = e / e.sum()
        cum = np.cumsum(e)
        Xf, Xh = split_last64(X)
        Vf, sf = thin_right_basis(Xf, 192)
        Xp, Xhp = split_prompt(X)
        Vp, _ = thin_right_basis(Xp, 185)
        rec = {
            "layer": i,
            "rms": float(np.sqrt(np.mean(np.square(X, dtype=np.float64)))),
            "energy_cum": {str(r): float(cum[r - 1]) for r in (1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 256)},
            "hold64_energy_in_fit192": {str(r): energy_in_V(Xh, Vf[:, :r]) for r in RANKS},
            "hold_prompt_energy_in_fit185": {str(r): energy_in_V(Xhp, Vp[:, : min(r, Vp.shape[1])]) for r in RANKS if r <= 185},
        }
        per.append(rec)
        Vs_full.append(vt_full[:32].T.copy())
        Vs_fit.append(Vf[:, :32].copy())
        if i in (0, 3, 15, 31, 47, 58, 63):
            log(f"  L{i} e64={cum[63]:.4f} e192={cum[191]:.4f} hold64@r64={rec['hold64_energy_in_fit192']['64']:.4f} hold64@r192={rec['hold64_energy_in_fit192']['192']:.4f}")
    result["activation"]["per_layer"] = per

    # shared V from stacked fit tokens of all 64 layers
    log("  stacked fit SVD (rSVD)")
    Xfit_stack = np.concatenate([x[:FIT_N] for x in xs], axis=0)  # 12288 x 5120
    Xhold_stack = np.concatenate([x[FIT_N:] for x in xs], axis=0)
    Vshared, s_shared = thin_right_basis(Xfit_stack, 192)
    log("  stacked fit SVD done")
    e_s = np.square(s_shared, dtype=np.float64)
    e_s = e_s / e_s.sum()
    cum_s = np.cumsum(e_s)
    shared = {
        "n_fit_rows": int(Xfit_stack.shape[0]),
        "rows_per_dim": float(Xfit_stack.shape[0] / HIDDEN),
        "energy_cum_shared_fit": {str(r): float(cum_s[r - 1]) for r in RANKS},
        "hold_stack_energy_in_shared": {str(r): energy_in_V(Xhold_stack, Vshared[:, :r]) for r in RANKS},
    }
    # per-layer hold energy in SHARED V (not that layer's V)
    per_shared = []
    for i in range(64):
        Xh = xs[i][FIT_N:]
        per_shared.append({
            "layer": i,
            "hold64_energy_in_shared": {str(r): energy_in_V(Xh, Vshared[:, :r]) for r in RANKS},
            "hold64_energy_in_self": per[i]["hold64_energy_in_fit192"],
        })
    shared["per_layer_hold"] = per_shared
    # leave-one-out for sample layers: V from other 63
    loo = []
    for i in LAYERS_MLP:
        log(f"  loo V layer {i}")
        others = np.concatenate([xs[j][:FIT_N] for j in range(64) if j != i], axis=0)
        Vo, _ = thin_right_basis(others, 192)
        Xh = xs[i][FIT_N:]
        Xall_i = xs[i]
        loo.append({
            "layer": i,
            "hold64_energy_in_loo": {str(r): energy_in_V(Xh, Vo[:, :r]) for r in RANKS},
            "full256_energy_in_loo": {str(r): energy_in_V(Xall_i, Vo[:, :r]) for r in RANKS},
        })
    shared["leave_one_layer_out"] = loo
    result["activation"]["shared"] = shared
    log(f"  shared fit energy@64={cum_s[63]:.4f} @192={cum_s[191]:.4f} hold_stack@64={shared['hold_stack_energy_in_shared']['64']:.4f} @192={shared['hold_stack_energy_in_shared']['192']:.4f}")

    # adjacent / far subspace overlap of top-32 activation PCs
    adj = []
    far = []
    for i in range(63):
        adj.append(subspace_overlap(Vs_fit[i], Vs_fit[i + 1]))
    for i in range(48):
        far.append(subspace_overlap(Vs_fit[i], Vs_fit[i + 16]))
    result["activation"]["subspace_overlap_top32_fit"] = {
        "adjacent_mean": float(np.mean(adj)),
        "adjacent_min": float(np.min(adj)),
        "adjacent_max": float(np.max(adj)),
        "d16_mean": float(np.mean(far)),
        "d16_min": float(np.min(far)),
        "d16_max": float(np.max(far)),
        "null_k32": 32.0 / 5120.0,
        "adjacent": [float(x) for x in adj],
    }
    log(f"  act PC32 overlap adj mean={np.mean(adj):.4f} d16 mean={np.mean(far):.4f} null={32/5120:.4f}")

    # prompt-hold shared V
    Xfit_p = np.concatenate([x[:PROMPT_FIT_N] for x in xs], axis=0)
    Xhold_p = np.concatenate([x[PROMPT_FIT_N:] for x in xs], axis=0)
    Vp, _ = thin_right_basis(Xfit_p, 185)
    result["activation"]["shared_prompt_split"] = {
        "n_fit": int(Xfit_p.shape[0]),
        "n_hold": int(Xhold_p.shape[0]),
        "hold_stack_energy": {str(r): energy_in_V(Xhold_p, Vp[:, : min(r, Vp.shape[1])]) for r in RANKS if r <= 185},
    }
    # stash Vshared in result? too big. Recompute later from same data.
    result["activation"]["rss_gb"] = rss_gb()
    dump(result)
    # keep xs in closure via result path; return stack pieces we need
    result["_cache"] = {
        "Vshared_fit64": Vshared,  # not written (ndarray stripped? we dump before cache)
    }
    # Don't put ndarrays into dumped result. Keep locally by storing on function attribute.
    phase_activation.Vshared = Vshared
    phase_activation.xs = xs
    log("PHASE1 done")


# ---------------------------------------------------------------------------
# phase 2 — per-tensor functional generators
# ---------------------------------------------------------------------------

def score_family_ZV(W, X_fit, X_hold, X_probe, Z, V, extra=None) -> dict:
    rec = {
        "fit": output_scores(W, X_fit, apply_ZV(X_fit, Z, V)),
        "hold": output_scores(W, X_hold, apply_ZV(X_hold, Z, V)),
        "probe_otherX": output_scores(W, X_probe, apply_ZV(X_probe, Z, V)),
        "weight": weight_cosine_from_ZV(W, Z, V),
        "rank": int(V.shape[1]),
    }
    if extra:
        rec.update(extra)
    return rec


def phase_tensors(result: dict) -> None:
    log("PHASE2 per-tensor generators")
    xs = phase_activation.xs
    Vshared = phase_activation.Vshared
    # a far-layer probe X (L40 hold) used as OOD activation for every tensor
    X_probe_default = xs[40][FIT_N:]
    rng = np.random.default_rng(1)
    jobs = []
    for L in LAYERS_MLP:
        jobs.append((L, "mlp.gate_proj", "mlp.gate_proj.weight", "hidden", None))
        jobs.append((L, "mlp.up_proj", "mlp.up_proj.weight", "hidden", None))
        jobs.append((L, "mlp.down_proj", "mlp.down_proj.weight", "swiglu", None))
    for L in LAYERS_GQA:
        jobs.append((L, "full.q_proj", "self_attn.q_proj.weight", "hidden", None))
        jobs.append((L, "full.k_proj", "self_attn.k_proj.weight", "hidden", None))
    for L in LAYERS_DN:
        jobs.append((L, "lin.in_proj_qkv", "linear_attn.in_proj_qkv.weight", "hidden", None))
        jobs.append((L, "lin.out_proj", "linear_attn.out_proj.weight", "mixer_x_missing", None))

    tensor_rows = []
    swiglu_cache: dict[int, np.ndarray] = {}
    block_cache: dict[int, dict] = {}

    for L, cls, suffix, site, _ in jobs:
        name = tensor_name(L, suffix)
        log(f"  load {name}")
        if site == "mixer_x_missing":
            tensor_rows.append({
                "name": name,
                "class": cls,
                "layer": L,
                "site": site,
                "status": "REFUSED",
                "reason": "mixer_x never captured; out_proj/o_proj input is 6144. Function not scored.",
            })
            continue
        W = load_tensor(name)
        m, k = W.shape
        if site == "hidden":
            X = xs[L]
            if X.shape[1] != k:
                tensor_rows.append({"name": name, "status": "REFUSED", "reason": f"X dim {X.shape[1]} != W cols {k}"})
                del W
                continue
        else:
            if L not in swiglu_cache:
                log(f"    reconstruct SwiGLU X L{L}")
                Wg = load_tensor(tensor_name(L, "mlp.gate_proj.weight"))
                Wu = load_tensor(tensor_name(L, "mlp.up_proj.weight"))
                Xh = xs[L]
                Yg = Xh @ Wg.T
                Yu = Xh @ Wu.T
                swiglu_cache[L] = silu(Yg) * Yu
                block_cache.setdefault(L, {})
                block_cache[L]["Yg"] = Yg
                block_cache[L]["Yu"] = Yu
                block_cache[L]["X"] = Xh
                del Wg, Wu
            X = swiglu_cache[L]
            if X.shape[1] != k:
                tensor_rows.append({"name": name, "status": "REFUSED", "reason": f"swiglu dim {X.shape[1]} != {k}"})
                del W
                continue

        X_fit, X_hold = split_last64(X)
        # probe: same-shape activations from another layer if hidden, else gaussian
        if site == "hidden":
            X_probe = X_probe_default
        else:
            # different layer swiglu if present else gaussian matching rms
            other = 63 if L != 63 else 0
            if other in swiglu_cache and swiglu_cache[other].shape[1] == k:
                X_probe = swiglu_cache[other][FIT_N:]
            else:
                rms = float(np.sqrt(np.mean(np.square(X_hold, dtype=np.float64))))
                X_probe = (rng.standard_normal((HOLD_N, k)) * rms).astype(np.float32)

        row = {
            "name": name,
            "class": cls,
            "layer": L,
            "site": site,
            "shape": [m, k],
            "elems": int(m * k),
            "rows_per_in_dim_capture": float(N_TOKENS / k),
            "rows_per_in_dim_fit": float(FIT_N / k),
            "q4_bytes": q4_bytes(m, k),
            "families": {},
        }

        # calibration codecs (weight-only)
        if cls in ("mlp.gate_proj", "mlp.down_proj") and L in (0, 63):
            Wb = binary_recon(W)
            Wq = uniform_q4(W)
            row["families"]["binary_g128"] = {
                "fit": output_scores(W, X_fit, X_fit @ Wb.T),
                "hold": output_scores(W, X_hold, X_hold @ Wb.T),
                "probe_otherX": output_scores(W, X_probe, X_probe @ Wb.T),
                "weight_cosine": cosine(W, Wb),
                "store_bytes": bill_bytes(m * k // 8 + (m * k // 128) * 2),
                "kind": "weight_codec_control",
            }
            row["families"]["uniform_q4_g64"] = {
                "fit": output_scores(W, X_fit, X_fit @ Wq.T),
                "hold": output_scores(W, X_hold, X_hold @ Wq.T),
                "probe_otherX": output_scores(W, X_probe, X_probe @ Wq.T),
                "weight_cosine": cosine(W, Wq),
                "store_bytes": bill_bytes(q4_bytes(m, k)),
                "kind": "weight_codec_control",
            }
            del Wb, Wq
            log(f"    calib {cls} L{L} binary_hold={row['families']['binary_g128']['hold']['cosine']:.6f} q4_hold={row['families']['uniform_q4_g64']['hold']['cosine']:.6f}")

        # --- act-PCA per-layer (input projector) ---
        Vself, _s = thin_right_basis(X_fit, 192)
        fam_act = {}
        for r in RANKS:
            if r > Vself.shape[1]:
                continue
            V = Vself[:, :r]
            Z = W @ V
            rec = score_family_ZV(W, X_fit, X_hold, X_probe, Z, V)
            rec["store"] = {
                "shared_V_bytes_if_amortized": bill_bytes(r * k * 2),
                "per_site_Z_f16_bytes": bill_bytes(m * r * 2),
                "per_site_Z_q4_bytes": bill_bytes(q4_bytes(m, r) if r >= 64 else m * r * 2),
                "kind": "per_layer_act_pca",
            }
            rec["fit_energy_X_in_V"] = energy_in_V(X_fit, V)
            rec["hold_energy_X_in_V"] = energy_in_V(X_hold, V)
            rec["probe_energy_X_in_V"] = energy_in_V(X_probe, V)
            fam_act[str(r)] = rec
        row["families"]["act_pca_self"] = fam_act

        # --- shared act-PCA (hidden-side only; down uses its own 17408-d V) ---
        if k == HIDDEN:
            fam_sh = {}
            for r in RANKS:
                V = Vshared[:, :r]
                Z = W @ V
                rec = score_family_ZV(W, X_fit, X_hold, X_probe, Z, V)
                rec["store"] = {
                    "shared_V_bytes_once": bill_bytes(r * k * 2),
                    "per_site_Z_f16_bytes": bill_bytes(m * r * 2),
                    "kind": "shared_act_pca",
                }
                rec["fit_energy_X_in_V"] = energy_in_V(X_fit, V)
                rec["hold_energy_X_in_V"] = energy_in_V(X_hold, V)
                rec["probe_energy_X_in_V"] = energy_in_V(X_probe, V)
                fam_sh[str(r)] = rec
            row["families"]["act_pca_shared64"] = fam_sh

        # --- weight-space rSVD (dead family, scored on FUNCTION) ---
        fam_w = {}
        for r in (8, 32, 64, 160):
            Vw = rsvd_right(W, r)
            Z = W @ Vw
            rec = score_family_ZV(W, X_fit, X_hold, X_probe, Z, Vw)
            rec["store"] = {
                "per_site_factors_f16_bytes": bill_bytes(r * (m + k) * 2),
                "kind": "per_tensor_weight_rsvd",
                "family_note": "falsified_construction_scored_on_function",
            }
            rec["fit_energy_X_in_V"] = energy_in_V(X_fit, Vw)
            rec["hold_energy_X_in_V"] = energy_in_V(X_hold, Vw)
            fam_w[str(r)] = rec
        row["families"]["weight_rsvd"] = fam_w

        # --- Hadamard-domain low rank (structured G implicit) ---
        do_had = not (cls.endswith("k_proj"))
        fam_h = {}
        A_h = B_h = None
        if do_had:
            A_h, B_h = hadamard_factors(W, max(HAD_RANKS), iters=4, seed=0)
        for r in HAD_RANKS:
            if not do_had:
                break
            A, B = A_h[:, :r], B_h[:r]
            Yfit = apply_hadamard_factors(X_fit, A, B)
            Yhold = apply_hadamard_factors(X_hold, A, B)
            Yprobe = apply_hadamard_factors(X_probe, A, B)
            rec = {
                "fit": output_scores(W, X_fit, Yfit),
                "hold": output_scores(W, X_hold, Yhold),
                "probe_otherX": output_scores(W, X_probe, Yprobe),
                "rank": r,
                "store": {
                    "per_site_bytes_f16": bill_bytes(r * (m + k) * 2),
                    "shared_G_bytes": 0,
                    "kind": "hadamard_factors_implicit_H",
                },
            }
            # weight cosine via inner product
            inner = 0.0
            for t in range(r):
                inner += float(np.dot(apply_Wh(W, B[t]), A[:, t]))
            nw = float(np.linalg.norm(W.reshape(-1).astype(np.float64)))
            nh = float(math.sqrt(sum(float(np.dot(A[:, t], A[:, t])) * float(np.dot(B[t], B[t])) * k for t in range(r))))
            rec["weight_cosine"] = (inner / (nw * nh)) if nw * nh > 1e-12 else 0.0
            fam_h[str(r)] = rec
        if do_had:
            row["families"]["hadamard_rankr"] = fam_h
            log(f"    hadamard r1 hold={fam_h['1']['hold']['cosine']:.4f} r8 hold={fam_h['8']['hold']['cosine']:.4f}")

        # --- procedural Rademacher sandwich r=1 (L0/L63 gate only; apply is O(M) python) ---
        if cls == "mlp.gate_proj" and L in (0, 63):
            seeds = [0xC0FFEE, 1, 2, 7, 99]
            prng_rows = []
            best = None
            for sd in seeds:
                a, b = prng_rank1(W, seed=sd, iters=3)
                rec = {
                    "seed": int(sd),
                    "hold": output_scores(W, X_hold, apply_prng_sandwich(X_hold, a, b, sd)),
                    "fit": output_scores(W, X_fit, apply_prng_sandwich(X_fit, a, b, sd)),
                }
                prng_rows.append(rec)
                if best is None or rec["hold"]["cosine"] > best["hold"]["cosine"]:
                    best = rec
            row["families"]["prng_sandwich_r1"] = {
                "n_seeds": len(seeds),
                "best_seed": best,
                "all_hold_cosines": [p["hold"]["cosine"] for p in prng_rows],
                "store": {"per_site_bytes_f16": bill_bytes((m + k) * 2), "seed_bytes": 8, "kind": "procedural_rademacher_sandwich"},
                "kernel_note": "per-element hash has no FWHT; generate-then-MAC is the 460us-class death unless replaced by a structured transform",
            }

        # --- circulant / fourier structured (cheap fourier always; circulant LS on a subset) ---
        a_f, lam = fourier_multiplier_fit(W)
        row["families"]["fourier_rank1"] = {
            "fit": output_scores(W, X_fit, apply_fourier_multiplier(X_fit, a_f, lam)),
            "hold": output_scores(W, X_hold, apply_fourier_multiplier(X_hold, a_f, lam)),
            "probe_otherX": output_scores(W, X_probe, apply_fourier_multiplier(X_probe, a_f, lam)),
            "store": {"per_site_bytes_f16": bill_bytes((m + k) * 2), "kind": "fourier_multiplier_rank1"},
        }
        if (cls == "mlp.gate_proj" and L in (0, 31, 63)) or (cls == "mlp.down_proj" and L in (0, 63)):
            a_c, c_c = circulant_rowscale_fit(W)
            row["families"]["circulant_rowscale"] = {
                "fit": output_scores(W, X_fit, apply_circulant_rowscale(X_fit, a_c, c_c)),
                "hold": output_scores(W, X_hold, apply_circulant_rowscale(X_hold, a_c, c_c)),
                "probe_otherX": output_scores(W, X_probe, apply_circulant_rowscale(X_probe, a_c, c_c)),
                "store": {"per_site_bytes_f16": bill_bytes((m + 8192) * 2), "shared_G_bytes": 0, "kind": "circulant_rowscale_fft8192"},
            }
            log(f"    circ hold={row['families']['circulant_rowscale']['hold']['cosine']:.4f} fourier1 hold={row['families']['fourier_rank1']['hold']['cosine']:.4f} act192 hold={fam_act['192']['hold']['cosine']:.4f} wsvd64 hold={fam_w['64']['hold']['cosine']:.4f}")
        else:
            log(f"    fourier1 hold={row['families']['fourier_rank1']['hold']['cosine']:.4f} act192 hold={fam_act['192']['hold']['cosine']:.4f} wsvd64 hold={fam_w['64']['hold']['cosine']:.4f}")

        # Q4(Z) of act-PCA r=64/128: codes cheaper, same apply
        if k == HIDDEN:
            for r in (64, 128):
                V = Vself[:, :r]
                Z = W @ V
                Zq = uniform_q4(Z)
                rec = score_family_ZV(W, X_fit, X_hold, X_probe, Zq, V)
                rec["store"] = {
                    "per_site_Z_q4_bytes": bill_bytes(q4_bytes(m, r)),
                    "V_bytes": bill_bytes(r * k * 2),
                    "kind": "act_pca_Z_uniform_q4",
                }
                row["families"][f"act_pca_self_r{r}_Zq4"] = rec

        tensor_rows.append(row)
        if cls == "mlp.gate_proj":
            block_cache.setdefault(L, {})
            block_cache[L]["Wg"] = W
            block_cache[L]["Ah_g"] = A_h
            block_cache[L]["Bh_g"] = B_h
        elif cls == "mlp.up_proj":
            block_cache.setdefault(L, {})
            block_cache[L]["Wu"] = W
            block_cache[L]["Ah_u"] = A_h
            block_cache[L]["Bh_u"] = B_h
        elif cls == "mlp.down_proj":
            block_cache.setdefault(L, {})
            block_cache[L]["Wd"] = W
            block_cache[L]["Ah_d"] = A_h
            block_cache[L]["Bh_d"] = B_h
            block_cache[L]["Xsw"] = X
            score_one_block(result, L, block_cache[L], xs, Vshared)
            for key in list(block_cache[L].keys()):
                del block_cache[L][key]
            if L in swiglu_cache:
                del swiglu_cache[L]
        else:
            del W
        result["tensors"] = tensor_rows
        result["rss_gb"] = rss_gb()
        dump({k: v for k, v in result.items() if not k.startswith("_")})

    result["tensors"] = tensor_rows
    phase_tensors.block_cache = block_cache
    phase_tensors.swiglu_cache = swiglu_cache
    log("PHASE2 done")


# ---------------------------------------------------------------------------
# phase 3 — shared G transfer of WEIGHT right-basis + cross-layer act V
# ---------------------------------------------------------------------------

def phase_shared_transfer(result: dict) -> None:
    log("PHASE3 shared transfer")
    xs = phase_activation.xs
    Vshared = phase_activation.Vshared
    rows = []
    # For gate: V_w from L0 applied to L15/L31/L63; V_act from L0 applied to those
    src = 0
    Wsrc = load_tensor(tensor_name(src, "mlp.gate_proj.weight"))
    Vw_src = rsvd_right(Wsrc, 64)
    Xsrc = xs[src]
    Va_src, _ = thin_right_basis(Xsrc[:FIT_N], 192)
    del Wsrc
    for L in (3, 15, 31, 47, 63):
        W = load_tensor(tensor_name(L, "mlp.gate_proj.weight"))
        Xf, Xh = split_last64(xs[L])
        rec = {"layer": L, "class": "mlp.gate_proj"}
        for name, V in (("weight_V_from_L0_r64", Vw_src), ("act_V_from_L0_r64", Va_src[:, :64]), ("act_V_shared64_r64", Vshared[:, :64]), ("act_V_self_r64", thin_right_basis(Xf, 64)[0])):
            Z = W @ V
            rec[name] = {
                "hold": output_scores(W, Xh, apply_ZV(Xh, Z, V)),
                "fit": output_scores(W, Xf, apply_ZV(Xf, Z, V)),
                "hold_energy_X_in_V": energy_in_V(Xh, V),
                "weight": weight_cosine_from_ZV(W, Z, V),
            }
        rows.append(rec)
        log(f"  L{L} gate hold self={rec['act_V_self_r64']['hold']['cosine']:.4f} shared={rec['act_V_shared64_r64']['hold']['cosine']:.4f} L0act={rec['act_V_from_L0_r64']['hold']['cosine']:.4f} L0w={rec['weight_V_from_L0_r64']['hold']['cosine']:.4f}")
        del W
    result["shared_transfer_gate"] = rows
    dump({k: v for k, v in result.items() if not k.startswith("_")})
    log("PHASE3 done")


# ---------------------------------------------------------------------------
# phase 4 — MLP block function
# ---------------------------------------------------------------------------

def block_out(X, Wg, Wu, Wd):
    return (silu(X @ Wg.T) * (X @ Wu.T)) @ Wd.T


def score_one_block(result: dict, L: int, cache: dict, xs, Vshared) -> None:
    log(f"  MLP block L{L}")
    X = xs[L]
    Xf, Xh = split_last64(X)
    Wg, Wu, Wd = cache["Wg"], cache["Wu"], cache["Wd"]
    Yh = block_out(Xh, Wg, Wu, Wd)
    Yf = block_out(Xf, Wg, Wu, Wd)
    rec = {"layer": L, "hold_y_rms": float(np.sqrt(np.mean(np.square(Yh, dtype=np.float64))))}

    def pack(Yhat_h, Yhat_f, store):
        return {
            "hold_cosine": cosine(Yh, Yhat_h),
            "hold_rel_l2": rel_l2(Yh, Yhat_h),
            "fit_cosine": cosine(Yf, Yhat_f),
            "store": store,
        }

    Wgq, Wuq, Wdq = uniform_q4(Wg), uniform_q4(Wu), uniform_q4(Wd)
    rec["q4_all"] = pack(block_out(Xh, Wgq, Wuq, Wdq), block_out(Xf, Wgq, Wuq, Wdq), {
        "bytes": 3 * q4_bytes(INTERMEDIATE, HIDDEN),
        "kind": "uniform_q4_g64",
    })
    del Wgq, Wuq, Wdq

    Xsw = cache["Xsw"]
    Xsw_f, Xsw_h = split_last64(Xsw)
    Vd, _ = thin_right_basis(Xsw_f, 192)
    for r in (32, 64, 128, 192):
        Vg = Vshared[:, :r]
        Zg = Wg @ Vg
        Zu = Wu @ Vg
        Vdd = Vd[:, :r]
        Zd = Wd @ Vdd

        def gen_block(Xin, Zg=Zg, Zu=Zu, Vg=Vg, Zd=Zd, Vdd=Vdd):
            yg = apply_ZV(Xin, Zg, Vg)
            yu = apply_ZV(Xin, Zu, Vg)
            xsw = silu(yg) * yu
            return apply_ZV(xsw, Zd, Vdd)

        rec[f"actpca_r{r}_all"] = pack(gen_block(Xh), gen_block(Xf), {
            "shared_V_hidden_bytes": bill_bytes(r * HIDDEN * 2),
            "Z_gate_up_bytes": 2 * bill_bytes(INTERMEDIATE * r * 2),
            "V_down_bytes": bill_bytes(r * INTERMEDIATE * 2),
            "Z_down_bytes": bill_bytes(HIDDEN * r * 2),
            "kind": "shared_act_pca_gateup_self_down",
        })

        def gen_gu_exact_d(Xin, Zg=Zg, Zu=Zu, Vg=Vg):
            yg = apply_ZV(Xin, Zg, Vg)
            yu = apply_ZV(Xin, Zu, Vg)
            return (silu(yg) * yu) @ Wd.T

        rec[f"actpca_r{r}_gu_exact_down"] = pack(gen_gu_exact_d(Xh), gen_gu_exact_d(Xf), {"kind": "isolate_swiglu"})
        rec[f"actpca_r{r}_exact_gu_gen_down"] = pack(
            apply_ZV(Xsw_h, Zd, Vdd),
            apply_ZV(Xsw_f, Zd, Vdd),
            {"kind": "isolate_down"},
        )

    for r in (1, 8):
        def had_block(Xin, r=r):
            yg = apply_hadamard_factors(Xin, cache["Ah_g"][:, :r], cache["Bh_g"][:r])
            yu = apply_hadamard_factors(Xin, cache["Ah_u"][:, :r], cache["Bh_u"][:r])
            xsw = silu(yg) * yu
            return apply_hadamard_factors(xsw, cache["Ah_d"][:, :r], cache["Bh_d"][:r])

        rec[f"hadamard_r{r}_all"] = pack(had_block(Xh), had_block(Xf), {
            "per_mlp_bytes_f16": 3 * bill_bytes(r * (INTERMEDIATE + HIDDEN) * 2),
            "kind": "hadamard_rankr_all",
        })

    Vwg = rsvd_right(Wg, 64)
    Vwu = rsvd_right(Wu, 64)
    Vwd = rsvd_right(Wd, 64)
    Zg, Zu, Zd = Wg @ Vwg, Wu @ Vwu, Wd @ Vwd

    def wsvd_block(Xin):
        yg = apply_ZV(Xin, Zg, Vwg)
        yu = apply_ZV(Xin, Zu, Vwu)
        xsw = silu(yg) * yu
        return apply_ZV(xsw, Zd, Vwd)

    rec["weight_rsvd_r64_all"] = pack(wsvd_block(Xh), wsvd_block(Xf), {
        "bytes": 3 * bill_bytes(64 * (INTERMEDIATE + HIDDEN) * 2),
        "kind": "dead_family_on_block",
    })
    result.setdefault("mlp_block", []).append(rec)
    log(
        f"  L{L} block hold q4={rec['q4_all']['hold_cosine']:.4f} "
        f"act64={rec['actpca_r64_all']['hold_cosine']:.4f} "
        f"act192={rec['actpca_r192_all']['hold_cosine']:.4f} "
        f"had1={rec['hadamard_r1_all']['hold_cosine']:.4f} "
        f"wsvd64={rec['weight_rsvd_r64_all']['hold_cosine']:.4f}"
    )


def phase_block(result: dict) -> None:
    n = len(result.get("mlp_block", []))
    log(f"PHASE4 MLP block already scored inline n={n}")
    if n == 0:
        log("PHASE4 EMPTY — no inline block scores")


# ---------------------------------------------------------------------------
# phase 5 — accounting + kernel arithmetic
# ---------------------------------------------------------------------------

def phase_accounting(result: dict) -> None:
    log("PHASE5 accounting + kernel arithmetic")
    # Q4 incumbents
    gate_q4 = q4_bytes(INTERMEDIATE, HIDDEN)
    down_q4 = q4_bytes(HIDDEN, INTERMEDIATE)
    mlp_q4 = 64 * (2 * gate_q4 + down_q4)
    # cited roofs (NOT remeasured GPU)
    addr_gbps = 639.25
    payload = 13_611_663_360
    addr_ns = 21_293_102.5
    token_ns = 39_326_090
    codebook_us = 460.0
    disc_gate_q4_ns = 15500.0
    disc_gate_f32_ns = 15125.0

    def stream_us(nbytes: float, gbps: float = addr_gbps) -> float:
        return 1e6 * nbytes / (gbps * 1e9)

    families = []
    for r in RANKS:
        v_h = bill_bytes(r * HIDDEN * 2)
        z_gu = bill_bytes(INTERMEDIATE * r * 2)
        v_d = bill_bytes(r * INTERMEDIATE * 2)
        z_d = bill_bytes(HIDDEN * r * 2)
        # gate+up share V; 64 layers; down has own V per layer (swiglu basis not shared cheaply)
        mlp_bytes = v_h + 64 * (2 * z_gu + v_d + z_d)
        # if we also share nothing on down V across layers
        complete_bytes_rest_q4 = (payload - mlp_q4) + mlp_bytes  # GEMV payload substitution, not complete artifact
        # honest complete BPW: replace MLP Q4 bytes in G0-like budget
        # G0 complete BPW 4.2527 on N_SOURCE. We only have GEMV payload here.
        # Complete artifact also has embed table, norms, catalog.
        # Report class BPW on MLP elems and a SUBSTITUTION complete BPW:
        #   (G0_bytes - mlp_q4 + mlp_bytes) * 8 / N_SOURCE
        g0_bpw = 4.252735126866492
        g0_bytes = g0_bpw * N_SOURCE / 8.0
        sub_bytes = g0_bytes - mlp_q4 + mlp_bytes
        families.append({
            "name": f"shared_actpca_r{r}_mlp_f16Z",
            "r": r,
            "mlp_bytes": mlp_bytes,
            "mlp_class_bpw": 8.0 * mlp_bytes / N_MLP,
            "substitution_complete_bpw": 8.0 * sub_bytes / N_SOURCE,
            "g0_complete_bpw": g0_bpw,
            "delta_complete_bpw": 8.0 * sub_bytes / N_SOURCE - g0_bpw,
            "per_gate_stream_bytes": z_gu,  # V cached
            "q4_gate_stream_bytes": gate_q4,
            "stream_us_gate_at_639gbps": stream_us(z_gu),
            "stream_us_q4_gate_at_639gbps": stream_us(gate_q4),
            "flops_apply_gate": r * (HIDDEN + INTERMEDIATE),
            "flops_q4_gate": INTERMEDIATE * HIDDEN,
            "label": "PROJECTED_storage_DERIVED_from_MEASURED_shapes",
        })
        # Z at Q4
        z_gu_q = bill_bytes(q4_bytes(INTERMEDIATE, r))
        z_d_q = bill_bytes(q4_bytes(HIDDEN, r))
        mlp_q = v_h + 64 * (2 * z_gu_q + v_d + z_d_q)
        sub_q = g0_bytes - mlp_q4 + mlp_q
        families.append({
            "name": f"shared_actpca_r{r}_mlp_q4Z",
            "r": r,
            "mlp_bytes": mlp_q,
            "mlp_class_bpw": 8.0 * mlp_q / N_MLP,
            "substitution_complete_bpw": 8.0 * sub_q / N_SOURCE,
            "per_gate_stream_bytes": z_gu_q,
            "stream_us_gate_at_639gbps": stream_us(z_gu_q),
            "label": "PROJECTED",
        })

    for r in HAD_RANKS:
        per = bill_bytes(r * (INTERMEDIATE + HIDDEN) * 2)
        mlp_bytes = 64 * 3 * per
        g0_bpw = 4.252735126866492
        g0_bytes = g0_bpw * N_SOURCE / 8.0
        sub_bytes = g0_bytes - mlp_q4 + mlp_bytes
        # apply: r FWHTs of PAD + r*M scales
        fwht_adds = r * PAD * int(math.log2(PAD))
        families.append({
            "name": f"hadamard_r{r}_mlp",
            "r": r,
            "mlp_bytes": mlp_bytes,
            "mlp_class_bpw": 8.0 * mlp_bytes / N_MLP,
            "substitution_complete_bpw": 8.0 * sub_bytes / N_SOURCE,
            "per_gate_stream_bytes": per,
            "stream_us_gate_at_639gbps": stream_us(per),
            "fwht_adds_per_gate": fwht_adds,
            "flops_q4_gate": INTERMEDIATE * HIDDEN,
            "label": "PROJECTED",
        })

    # PRNG per-element generate cost ESTIMATE
    hash_ops_per_w = 10  # splitmix-class
    prng_ops_gate = INTERMEDIATE * HIDDEN * hash_ops_per_w
    families.append({
        "name": "prng_per_element_gate",
        "mlp_bytes": 64 * 3 * bill_bytes((INTERMEDIATE + HIDDEN) * 2 + 8),
        "hash_ops_per_gate": prng_ops_gate,
        "cited_codebook_lookup_gemv_us": codebook_us,
        "verdict_exec": "KILLS_on_time_if_per_element_hash",
        "label": "ESTIMATED_ops_CITED_460us",
    })

    result["accounting"] = {
        "n_source": N_SOURCE,
        "n_mlp": N_MLP,
        "q4_gate_bytes": gate_q4,
        "q4_mlp_bytes": mlp_q4,
        "cited_addr_gbps": addr_gbps,
        "cited_addr_gbps_source": "g1-direct-gemv-geometry.md sealed addressing 639.25 GB/s",
        "cited_payload_bytes": payload,
        "cited_addr_ns": addr_ns,
        "cited_token_ns": token_ns,
        "cited_codebook_gemv_us": codebook_us,
        "cited_disc_gate_q4_ns": disc_gate_q4_ns,
        "cited_disc_gate_f32_ns": disc_gate_f32_ns,
        "disc_note": "15us discriminator is launch-dominated / cache-resident, NOT unique-once. Streaming comparison uses 639.25 GB/s.",
        "families": families,
        "kernel_paths": {
            "act_pca_ZV": {
                "native": "new kernel: s = V^T x (r x K GEMV, V resident if shared); y = Z s (M x r GEMV). MAC immediately. Never materialize M x K.",
                "reject": "packed-then-expand-to-Q4-then-generic-GEMV",
            },
            "hadamard": {
                "native": "new kernel: u = b ⊙ x; FWHT_pad(u); y = a ⊙ u[:M]. Shared memory FWHT. Never materialize W.",
                "reject": "generate W then GEMV",
            },
            "circulant": {
                "native": "new kernel: y = a ⊙ iFFT(λ ⊙ FFT(x_pad)). Metal has no stock FFT; length-8192/32768 FFT is writeable but not free.",
                "reject": "materialize circulant",
            },
            "prng_per_element": {
                "native": "thread generates G[i,j] via hash, FMA into acc. Same class as rejected 460us codebook GEMV.",
                "verdict": "KILLS on this machine",
            },
        },
    }
    result["rss_max_gb"] = rss_gb()
    result["wall_s"] = time.time() - result["t0"]
    dump({k: v for k, v in result.items() if not k.startswith("_")})
    log("PHASE5 done")


def main():
    if LOG.exists():
        LOG.unlink()
    t0 = time.time()
    result = {
        "schema": "hawking.g1.generative_representation.v1",
        "t0_human": time.strftime("%Y-%m-%d %H:%M:%S"),
        "t0": t0,
        "identity": {
            "model": "Qwen3.8-27B",
            "bf16": str(MODEL_DIR),
            "capture": str(CAPTURE_DIR),
            "n_source": N_SOURCE,
        },
        "method": {
            "objective": "layer function F_l(X) not ||W-What||",
            "hold": "last-64 of 256 (descent-comparable); also prompt 185/71 in phase1",
            "capture_limit": "256 tokens; ranks a sharp set; does not estimate magnitudes; mixer_x missing",
            "not_the_dead_family": [
                "G is shared (act-PCA V amortized across sites)",
                "score is output agreement not weight reconstruction",
                "z_l may be M x r with r << min(M,K) because G carries the activation basis",
            ],
        },
    }
    # identity smoke
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    result["identity"]["config_model_type"] = cfg.get("model_type")
    result["identity"]["num_hidden_layers"] = cfg.get("text_config", cfg).get("num_hidden_layers") if isinstance(cfg.get("text_config"), dict) else cfg.get("num_hidden_layers")
    log(f"start model_type={result['identity'].get('config_model_type')}")
    dump(result)

    phase_activation(result)
    phase_tensors(result)
    phase_shared_transfer(result)
    phase_block(result)
    phase_accounting(result)
    log(f"ALL DONE wall={time.time()-t0:.1f}s rss_max={rss_gb():.3f}GiB -> {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""N047 — Doctor diagnosis engine (S026 §66, §67; DOC-DIAGNOSIS; CPU).

Diagnosis PREDICTS which S026 §65 representation families are plausible per
organ. It does not certify, does not reopen the closed 2.25 MLP floor, and
does not touch the GPU. Features are measured on REAL parent tensors streamed
one at a time from the qualified BF16 parent. ~/noetic/NOETIC_PARENT_A is
read-only (catalog fingerprint before/after).

    python3 tools/headless/doctor_diagnosis.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

RECEIPT = REPO / "receipts" / "headless" / "DOCTOR_DIAGNOSIS.json"
DOCS = REPO / "docs" / "ultragoals" / "DOCTOR_DIAGNOSIS.md"
GENERATOR = "tools/headless/doctor_diagnosis.py"
SCHEMA = "hawking.headless.doctor_diagnosis.v1"
OBLIGATION = (
    "N047 — DOCTOR_DIAGNOSIS (S026 §66, §67; DOC-DIAGNOSIS; CPU). The engine "
    "that ranks technique families per organ from measured diagnostic features "
    "on the real parent tensors, with an AVOID list citing negative science."
)

PARENT_A = Path(
    os.environ.get("NOETIC_PARENT_A_ROOT", str(Path.home() / "noetic" / "NOETIC_PARENT_A"))
)
PARENT_BF16_CANDIDATES = [
    Path(os.environ.get("QWEN38_PARENT_BF16", str(Path.home() / "models/qwen3.8-27b-abliterated-bf16"))),
    Path("/Users/scammermike/models/qwen3.8-27b-abliterated-bf16"),
]
N043_CANDIDATES = [
    REPO / "receipts/headless/DOCTOR_TECHNIQUE_REGISTRY.json",
    Path("/Users/scammermike/Downloads/hawking/receipts/headless/DOCTOR_TECHNIQUE_REGISTRY.json"),
    Path("/Users/scammermike/.claude-grok/worktrees/n043registry-20260824-181218/receipts/headless/DOCTOR_TECHNIQUE_REGISTRY.json"),
]
N046_CANDIDATES = [
    REPO / "receipts/headless/LITERATURE_FRONTIER.json",
    Path("/Users/scammermike/.claude-grok/worktrees/n046literature-20260824-181501/receipts/headless/LITERATURE_FRONTIER.json"),
    Path("/Users/scammermike/Downloads/hawking/receipts/headless/LITERATURE_FRONTIER.json"),
]

LAYERS = 64
HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
GQA_HEADS = 24
GQA_KV_HEADS = 4
GQA_HEAD_DIM = 256
DN_K_HEADS = 16
DN_V_HEADS = 48
DN_HEAD_DIM = 128
FULL_ATTN_INTERVAL = 4
PARENT_PARAMS = 26_895_998_464

SVD_K = 48
SVD_NITER = 1
SKETCH_DIM = 32
KMEANS_K_SMALL = 16
KMEANS_K_LARGE = 64
KMEANS_ROWS = 2048
KMEANS_ITERS = 10
GROUP_VQ = 8
GROUP_ABSMAX = 64
NEAR_ZERO_REL = 1e-3
OUTLIER_MAD_K = 6.0
OUTLIER_STD_K = 3.0
SEED = 0xD047
EMBED_ROW_SAMPLE = 4096
EMBED_ROW_BATCH = 2048

# S026 §65 competing families. Labels are the steer text.
FAMILIES: tuple[tuple[str, str], ...] = (
    ("scalar_quantization", "scalar quantization"),
    ("binary", "binary"),
    ("ternary", "ternary"),
    ("trit_plane", "trit-plane"),
    ("vector_codebook", "vector codebook"),
    ("additive_codebook", "additive codebook"),
    ("adaptive_codebook", "adaptive codebook"),
    ("shared_basis", "shared basis"),
    ("tensor_factorization", "tensor factorization"),
    ("low_rank", "low-rank"),
    ("low_rank_sparse", "low-rank + sparse"),
    ("generated_coefficients", "generated coefficients"),
    ("structured_pruning", "structured pruning"),
    ("routed_structure", "routed structure"),
    ("protected_islands", "protected islands"),
)
FAMILY_LABEL = {k: v for k, v in FAMILIES}

FAMILY_TO_N043 = {
    "binary": "onebit",
    "ternary": "twla",
    "trit_plane": "ptqtp",
    "vector_codebook": "vptq",
    "additive_codebook": "aqlm",
    "adaptive_codebook": "aqlm",
    "low_rank": "caldera",
    "low_rank_sparse": "caldera",
    "structured_pruning": "prosparse",
    "routed_structure": "mixture_of_depths",
    "protected_islands": "squeezellm",
}

# Default layer samples: early / quarter / mid / three-quarter / late.
DEFAULT_MLP_LAYERS = (0, 15, 31, 47, 63)
DEFAULT_GQA_LAYERS = (3, 15, 31, 47, 63)
DEFAULT_DN_LAYERS = (1, 14, 30, 46, 62)

# Negative-science receipts that exist in this worktree (must stay real paths).
R_SHARED_C = "receipts/headless/SHARED_BASIS_COHERENT.json"
R_SHARED_K = "receipts/headless/SHARED_BASIS_KERNEL.json"
R_C1 = "receipts/headless/C1SHAREDBASIS_DESIGN.json"
R_C2 = "receipts/headless/C2TENSOROP_DESIGN.json"
R_C3 = "receipts/headless/C3LOWRANKSPARSE_DESIGN.json"
R_C4 = "receipts/headless/C4CODEBOOK_DESIGN.json"
R_C5 = "receipts/headless/C5STRUCTTRANSFORM_DESIGN.json"
R_BYTES = "receipts/headless/BYTES_FRONTIER.json"
R_BINARY = "receipts/headless/BINARY_HEALING.json"
R_HYBRID = "receipts/headless/HYBRID_OPERATOR.json"
R_TERNARY = "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json"
R_ONEBIT = "receipts/headless/ONEBIT_FAMILIES.json"
R_FIRST = "receipts/headless/FIRST_NOETIC_EXECUTABLE.json"
R_GEN = "receipts/headless/GENERATED_WEIGHTS_RETEST.json"
R_NNS = "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json"
R_FLOORS = "receipts/headless/ORGAN_DENSITY_FLOORS.json"
R_FRONTIERS = "receipts/headless/ORGAN_FRONTIERS.json"
R_FRACTIONAL = "receipts/headless/FRACTIONAL_BIT_CANON.json"

CATALOG_MAGIC = b"HQ38M20\0"
CATALOG_VERSION = 1
RECORD_SIZE = 128


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, timeout=20
        ).strip()
    except Exception:
        return "unknown"


def j(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [j(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return j(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, float):
        if not np.isfinite(x):
            return None
        return float(x)
    if isinstance(x, (bool, int, str)) or x is None:
        return x
    return str(x)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def first_existing(cands: list[Path]) -> Path | None:
    for p in cands:
        if p.is_file():
            return p
    return None


def find_parent_bf16() -> Path:
    for p in PARENT_BF16_CANDIDATES:
        if (p / "model.safetensors.index.json").is_file():
            return p
    raise FileNotFoundError("qualified parent bf16 not found")


def gqa_layers() -> tuple[int, ...]:
    return tuple(i for i in range(LAYERS) if (i + 1) % FULL_ATTN_INTERVAL == 0)


def dn_layers() -> tuple[int, ...]:
    g = set(gqa_layers())
    return tuple(i for i in range(LAYERS) if i not in g)


def tname(layer: int, kind: str) -> str:
    return f"model.language_model.layers.{layer}.{kind}"


# ---------------------------------------------------------------------------
# safetensors stream (one tensor at a time; never the 27B)
# ---------------------------------------------------------------------------


_INDEX_CACHE: dict[str, str] | None = None
_HEADER_CACHE: dict[str, dict[str, Any]] = {}


def weight_index(parent: Path) -> dict[str, str]:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = load_json(parent / "model.safetensors.index.json")["weight_map"]
    return _INDEX_CACHE


def _header_for_shard(shard: Path) -> dict[str, Any]:
    key = str(shard)
    hit = _HEADER_CACHE.get(key)
    if hit is not None:
        return hit
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        header["__header_bytes"] = 8 + n
    _HEADER_CACHE[key] = header
    return header


def bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load_tensor(parent: Path, name: str) -> np.ndarray:
    """Stream one tensor from a parent shard. Does not load the 27B."""
    shard = parent / weight_index(parent)[name]
    header = _header_for_shard(shard)
    meta = header[name]
    start, end = meta["data_offsets"]
    with open(shard, "rb") as f:
        f.seek(header["__header_bytes"] + start)
        raw = f.read(end - start)
    dtype = meta["dtype"]
    shape = tuple(meta["shape"])
    if dtype == "BF16":
        return np.array(bf16_to_f32(raw).reshape(shape), dtype=np.float32, copy=True)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape).copy()
    raise ValueError(f"{name} dtype {dtype}")


def load_tensor_row_sample(parent: Path, name: str, n_rows: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Stream a uniform row sample of a rank-2 BF16 tensor. For embed / lm_head."""
    shard = parent / weight_index(parent)[name]
    header = _header_for_shard(shard)
    meta = header[name]
    shape = tuple(meta["shape"])
    if len(shape) != 2:
        raise ValueError(f"{name} not rank-2")
    rows, cols = int(shape[0]), int(shape[1])
    take = int(min(n_rows, rows))
    idx = np.sort(rng.choice(rows, size=take, replace=False).astype(np.int64))
    start, _end = meta["data_offsets"]
    base = header["__header_bytes"] + start
    item = 2 if meta["dtype"] == "BF16" else 4
    row_bytes = cols * item
    chunks: list[np.ndarray] = []
    with open(shard, "rb") as f:
        for i in idx:
            f.seek(base + int(i) * row_bytes)
            raw = f.read(row_bytes)
            if meta["dtype"] == "BF16":
                chunks.append(bf16_to_f32(raw))
            elif meta["dtype"] == "F32":
                chunks.append(np.frombuffer(raw, dtype="<f4").copy())
            else:
                raise ValueError(meta["dtype"])
    W = np.stack(chunks, axis=0).astype(np.float32, copy=False)
    return W, {
        "full_shape": [rows, cols],
        "sampled_rows": take,
        "sampled_row_indices_head": [int(x) for x in idx[:8]],
        "sampling": "uniform_without_replacement",
    }


def iter_rows_stats(parent: Path, name: str, batch: int = EMBED_ROW_BATCH) -> dict:
    """One streaming pass over a rank-2 tensor for moments / sparsity / range."""
    shard = parent / weight_index(parent)[name]
    header = _header_for_shard(shard)
    meta = header[name]
    shape = tuple(meta["shape"])
    rows, cols = int(shape[0]), int(shape[1])
    start, _end = meta["data_offsets"]
    base = header["__header_bytes"] + start
    item = 2 if meta["dtype"] == "BF16" else 4
    row_bytes = cols * item
    n = 0
    mean = 0.0
    m2 = 0.0
    abs_sum = 0.0
    n_near = 0
    wmin = 0.0
    wmax = 0.0
    first = True
    with open(shard, "rb") as f:
        for r0 in range(0, rows, batch):
            r1 = min(rows, r0 + batch)
            f.seek(base + r0 * row_bytes)
            raw = f.read((r1 - r0) * row_bytes)
            if meta["dtype"] == "BF16":
                block = bf16_to_f32(raw)
            else:
                block = np.frombuffer(raw, dtype="<f4")
            x = block.astype(np.float64, copy=False)
            bmin = float(x.min())
            bmax = float(x.max())
            if first:
                wmin, wmax, first = bmin, bmax, False
            else:
                wmin = min(wmin, bmin)
                wmax = max(wmax, bmax)
            abs_sum += float(np.abs(x).sum())
            # Welford over the block via parallel merge of a batch mean.
            bn = int(x.size)
            bmean = float(x.mean())
            bm2 = float(np.square(x - bmean).sum())
            delta = bmean - mean
            tot = n + bn
            mean = mean + delta * (bn / tot)
            m2 = m2 + bm2 + delta * delta * n * bn / tot
            n = tot
            std_guess = float(np.sqrt(m2 / max(n, 1)))
            # sparsity vs running std is slightly biased on early batches;
            # recompute against final std below using a cheap abs threshold
            # relative to max. Record both.
            n_near += int(np.count_nonzero(np.abs(x) < 1e-8))
            del x, block
    var = m2 / max(n - 1, 1)
    std = float(np.sqrt(max(var, 0.0)))
    return {
        "n": n,
        "shape": [rows, cols],
        "mean": float(mean),
        "std": std,
        "var": float(var),
        "mean_abs": abs_sum / max(n, 1),
        "min": wmin,
        "max": wmax,
        "frac_abs_lt_1e8": n_near / max(n, 1),
        "frobenius_sq": float(m2 + n * mean * mean),
        "streamed_rows": True,
    }


# ---------------------------------------------------------------------------
# feature extractors (pure numpy; unit-tested)
# ---------------------------------------------------------------------------


def excess_kurtosis(x: np.ndarray) -> float:
    """Fisher excess kurtosis. Gaussian null = 0."""
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    n = v.size
    if n < 8:
        return 0.0
    mean = float(v.mean())
    c = v - mean
    m2 = float(np.mean(c * c))
    if m2 <= 1e-30:
        return 0.0
    m4 = float(np.mean(c * c * c * c))
    g2 = m4 / (m2 * m2) - 3.0
    return g2


def mad(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    med = float(np.median(v))
    return float(np.median(np.abs(v - med)))


def randomized_svd(
    W: np.ndarray, k: int, rng: np.random.Generator, niter: int = SVD_NITER
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Range-finder SVD. Returns U (m,k), S (k,), Vh (k,n) float32/float64."""
    W32 = np.asarray(W, dtype=np.float32)
    m, n = W32.shape
    k = int(max(1, min(k, m, n)))
    p = int(min(k + 8, min(m, n)))
    Omega = rng.standard_normal((n, p)).astype(np.float32)
    Y = W32 @ Omega
    Q, _ = np.linalg.qr(Y.astype(np.float64), mode="reduced")
    for _ in range(max(0, niter)):
        Z = W32.T @ Q.astype(np.float32)
        Y = W32 @ Z
        Q, _ = np.linalg.qr(Y.astype(np.float64), mode="reduced")
    B = Q.T @ W32.astype(np.float64)
    Uhat, S, Vh = np.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat
    kk = int(min(k, S.size))
    return U[:, :kk].astype(np.float32), S[:kk].astype(np.float64), Vh[:kk].astype(np.float32)


def spectrum_from_singular(
    S: np.ndarray, frobenius_sq: float, full_min_dim: int
) -> dict:
    s2 = np.asarray(S, dtype=np.float64) ** 2
    captured = float(s2.sum())
    frob = float(max(frobenius_sq, captured, 1e-30))
    residual = max(0.0, frob - captured)
    k = int(S.size)
    nrest = max(int(full_min_dim) - k, 1)
    # Do NOT collapse residual energy into one atom — that makes a full-rank
    # matrix look rank-1. Spread leftover energy uniformly over uncomputed
    # modes (maximum-entropy completion given only ||W||_F^2 and top-k).
    rest = np.full(nrest, residual / nrest, dtype=np.float64)
    energy = np.concatenate([s2, rest])
    tot = float(energy.sum())
    p = energy / max(tot, 1e-30)
    pr = float((tot * tot) / max(float(np.square(energy).sum()), 1e-30))
    ent = float(-np.sum(p * np.log(p + 1e-30)))
    erank = float(np.exp(ent))
    c = np.cumsum(s2) / max(frob, 1e-30)
    def _rank_at(th: float) -> int | None:
        if c.size == 0 or float(c[-1]) < th:
            return None  # not reached within computed k
        return int(np.searchsorted(c, th) + 1)
    cap_frac = captured / frob
    return {
        "k_computed": k,
        "singular_values_head": [float(x) for x in S[:12]],
        "captured_energy_frac": cap_frac,
        "residual_energy_frac": residual / frob,
        "participation_ratio": pr,
        "entropy_effective_rank": erank,
        "effective_rank_frac_of_min_dim": erank / max(full_min_dim, 1),
        "flat_spectrum_implied_rank": (k / cap_frac) if cap_frac > 1e-9 else None,
        "rank_for_90pct_energy": _rank_at(0.90),
        "rank_for_99pct_energy": _rank_at(0.99),
        "min_dim": int(full_min_dim),
        "spectrum_is_truncated": True,
        "residual_model": "uniform_over_uncomputed_modes",
        "null": "a full-rank Gaussian of the same shape has participation_ratio ≈ min(m,n)",
    }


def kmeans_rel_fro(
    X: np.ndarray, k: int, rng: np.random.Generator, n_iter: int = KMEANS_ITERS
) -> dict:
    """k-means++ then Lloyd. Returns reconstruction rel_fro vs the data matrix."""
    X64 = np.asarray(X, dtype=np.float64)
    n, d = X64.shape
    if n == 0 or d == 0:
        return {"k": 0, "rel_fro": 1.0, "n": 0, "d": 0}
    k = int(max(1, min(k, n)))
    centers = np.empty((k, d), dtype=np.float64)
    centers[0] = X64[int(rng.integers(n))]
    closest = np.full(n, np.inf, dtype=np.float64)
    for j in range(1, k):
        diff = X64 - centers[j - 1]
        dist = np.einsum("ij,ij->i", diff, diff)
        closest = np.minimum(closest, dist)
        s = float(closest.sum())
        if s <= 1e-30:
            centers[j] = X64[int(rng.integers(n))]
            continue
        p = closest / s
        centers[j] = X64[int(rng.choice(n, p=p))]
    x2 = np.einsum("ij,ij->i", X64, X64)
    for _ in range(n_iter):
        c2 = np.einsum("ij,ij->i", centers, centers)
        dots = X64 @ centers.T
        labels = np.argmin(x2[:, None] + c2[None, :] - 2.0 * dots, axis=1)
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = X64[mask].mean(axis=0)
    c2 = np.einsum("ij,ij->i", centers, centers)
    dots = X64 @ centers.T
    labels = np.argmin(x2[:, None] + c2[None, :] - 2.0 * dots, axis=1)
    recon = centers[labels]
    denom = float(np.linalg.norm(X64))
    rel = float(np.linalg.norm(X64 - recon) / max(denom, 1e-30))
    return {
        "k": k,
        "n": int(n),
        "d": int(d),
        "rel_fro": rel,
        "affinity": float(max(0.0, 1.0 - rel)),
        "null_rel_fro": "1-mean codebook (k=1) is the scale-only null; reported separately when computed",
    }


def grouped_meanabs_sign_rel_fro(W: np.ndarray, group: int = GROUP_ABSMAX) -> float:
    """HGRAVB01-style: per-group mean(|w|) * sign. Weight-space only."""
    w = np.asarray(W, dtype=np.float32).reshape(-1)
    n = int(w.size)
    g = int(group)
    pad = (-n) % g
    if pad:
        w = np.concatenate([w, np.zeros(pad, dtype=np.float32)])
    blocks = w.reshape(-1, g)
    scale = np.mean(np.abs(blocks), axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-12)
    q = np.sign(blocks).astype(np.float32)
    q[q == 0] = 1.0
    recon = (q * scale).reshape(-1)[:n].astype(np.float64)
    orig = np.asarray(W, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(orig - recon) / max(float(np.linalg.norm(orig)), 1e-30))


def grouped_absmax_rel_fro(W: np.ndarray, bits: int, group: int = GROUP_ABSMAX) -> float:
    """Weight-space grouped absmax reconstruction error. Diagnosis, not a gate."""
    w = np.asarray(W, dtype=np.float32).reshape(-1)
    n = int(w.size)
    g = int(group)
    pad = (-n) % g
    if pad:
        w = np.concatenate([w, np.zeros(pad, dtype=np.float32)])
    blocks = w.reshape(-1, g)
    scale = np.max(np.abs(blocks), axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-12)
    orig = np.asarray(W, dtype=np.float64).reshape(-1)
    if bits <= 0:
        recon = np.zeros_like(blocks)
    elif bits == 1:
        q = np.where(blocks >= 0, 1.0, -1.0).astype(np.float32)
        recon = q * scale
    else:
        qmax = float((1 << (bits - 1)) - 1)
        q = np.clip(np.round(blocks / scale * qmax), -qmax, qmax)
        recon = (q / qmax) * scale
    recon_f = recon.reshape(-1)[:n].astype(np.float64)
    return float(np.linalg.norm(orig - recon_f) / max(float(np.linalg.norm(orig)), 1e-30))


def subsample_entries(W: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    flat = np.asarray(W, dtype=np.float32).reshape(-1)
    if flat.size <= n:
        return flat.astype(np.float64)
    idx = rng.choice(flat.size, size=n, replace=False)
    return flat[idx].astype(np.float64)


def head_blocks(W: np.ndarray, n_heads: int, head_dim: int, axis: str) -> np.ndarray | None:
    """Reshape a projection into (n_heads, head_dim, rest) then flatten per head."""
    W = np.asarray(W, dtype=np.float32)
    if W.ndim != 2 or n_heads <= 1:
        return None
    m, n = W.shape
    if axis == "out":
        # rows are heads * head_dim (* maybe gate)
        if m % (n_heads * head_dim) == 0:
            extra = m // (n_heads * head_dim)
            H = W.reshape(n_heads, extra, head_dim, n)
            return H[:, 0, :, :].reshape(n_heads, -1)
        if m % n_heads == 0:
            return W.reshape(n_heads, -1)
    if axis == "in":
        if n % (n_heads * head_dim) == 0:
            extra = n // (n_heads * head_dim)
            H = W.reshape(m, n_heads, extra, head_dim)
            return np.transpose(H[:, :, 0, :], (1, 0, 2)).reshape(n_heads, -1)
        if n % n_heads == 0:
            return np.transpose(W.reshape(m, n_heads, -1), (1, 0, 2)).reshape(n_heads, -1)
    return None


def pairwise_mean_cosine(rows: np.ndarray) -> dict:
    X = np.asarray(rows, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return {"n": int(n), "mean_cosine": None, "min_cosine": None, "max_cosine": None}
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    U = X / norms
    G = U @ U.T
    iu = np.triu_indices(n, k=1)
    vals = G[iu]
    return {
        "n": int(n),
        "mean_cosine": float(vals.mean()),
        "min_cosine": float(vals.min()),
        "max_cosine": float(vals.max()),
        "null": "independent Gaussian heads → cosine ≈ 0",
    }


def features_of_matrix(
    W: np.ndarray,
    *,
    rng: np.random.Generator,
    head_spec: dict | None = None,
    sample_note: dict | None = None,
    compute_svd: bool = True,
) -> dict:
    """DIAGNOSIS FEATURES for one 2-D weight matrix. Pure CPU."""
    W = np.asarray(W, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError(f"expected rank-2, got {W.shape}")
    m, n = int(W.shape[0]), int(W.shape[1])
    fro2 = float(np.square(W.astype(np.float64)).sum())
    sample = subsample_entries(W, 1_000_000, rng)
    std = float(sample.std())
    med_abs = float(np.median(np.abs(sample)))
    mad_s = mad(sample)
    kurt = excess_kurtosis(sample)
    outlier_mad = float(np.mean(np.abs(sample - float(np.median(sample))) > OUTLIER_MAD_K * max(mad_s, 1e-30)))
    outlier_std = float(np.mean(np.abs(sample - float(sample.mean())) > OUTLIER_STD_K * max(std, 1e-30)))
    near = float(np.mean(np.abs(sample) < NEAR_ZERO_REL * max(std, 1e-30)))
    abs_x = np.abs(sample)
    gini = None
    if abs_x.size >= 8:
        srt = np.sort(abs_x)
        # Gini of |w|: 0 = equal magnitude, 1 = all mass on one entry.
        idx = np.arange(1, srt.size + 1, dtype=np.float64)
        gini = float(
            (2.0 * np.sum(idx * srt) / (srt.size * max(float(srt.sum()), 1e-30))) - (srt.size + 1) / srt.size
        )

    bits = {}
    for b in (1, 2, 3, 4):
        bits[str(b)] = grouped_absmax_rel_fro(W, b, GROUP_ABSMAX)
    # Campaign binary is mean-abs * sign, not absmax. Absmax 1-bit can have
    # rel_fro > 1 on near-Gaussian weights (the max overshoots the bulk).
    bits["1_meanabs"] = grouped_meanabs_sign_rel_fro(W, GROUP_ABSMAX)

    svd = None
    U = S = Vh = None
    if compute_svd:
        U, S, Vh = randomized_svd(W, SVD_K, rng, SVD_NITER)
        svd = spectrum_from_singular(S, fro2, min(m, n))

    # codebook: k-means on a row sample (vector quant of output channels)
    n_take = min(KMEANS_ROWS, m)
    row_idx = rng.choice(m, size=n_take, replace=False) if m > n_take else np.arange(m)
    rows = W[row_idx].astype(np.float32, copy=False)
    km1 = kmeans_rel_fro(rows, 1, rng, n_iter=6)
    km16 = kmeans_rel_fro(rows, min(KMEANS_K_SMALL, n_take), rng)
    km64 = kmeans_rel_fro(rows, min(KMEANS_K_LARGE, n_take), rng)
    # group-VQ along the last axis
    g = GROUP_VQ
    if n % g == 0:
        G = W.reshape(m, n // g, g).reshape(-1, g)
        if G.shape[0] > KMEANS_ROWS:
            G = G[rng.choice(G.shape[0], size=KMEANS_ROWS, replace=False)]
        gv16 = kmeans_rel_fro(G, min(KMEANS_K_SMALL, G.shape[0]), rng)
        gv64 = kmeans_rel_fro(G, min(KMEANS_K_LARGE, G.shape[0]), rng)
    else:
        gv16 = gv64 = {"k": None, "rel_fro": None, "affinity": None, "note": "last dim not divisible by group"}

    heads = None
    if head_spec:
        blocks = head_blocks(W, int(head_spec["n_heads"]), int(head_spec["head_dim"]), head_spec["axis"])
        if blocks is not None:
            heads = pairwise_mean_cosine(blocks)
            heads["geometry"] = {
                "n_heads": int(head_spec["n_heads"]),
                "head_dim": int(head_spec["head_dim"]),
                "axis": head_spec["axis"],
                "block_shape": [int(blocks.shape[0]), int(blocks.shape[1])],
            }

    sketch_omega = rng.standard_normal((n, SKETCH_DIM)).astype(np.float32)
    sketch = (W @ sketch_omega).mean(axis=0).astype(np.float64)

    feat = {
        "shape": [m, n],
        "n_weights": int(m * n),
        "sample": sample_note or {"kind": "full_matrix", "n_for_moments": int(sample.size)},
        "weight_distribution": {
            "mean": float(sample.mean()),
            "std": std,
            "mean_abs": float(np.mean(np.abs(sample))),
            "median_abs": med_abs,
            "min": float(sample.min()),
            "max": float(sample.max()),
            "excess_kurtosis": kurt,
            "excess_kurtosis_null": 0.0,
            "excess_kurtosis_null_kind": "gaussian",
            "outlier_frac_6mad": outlier_mad,
            "outlier_frac_3std": outlier_std,
            "mad": mad_s,
            "gini_abs": gini,
        },
        "sparsity": {
            "frac_near_zero": near,
            "near_zero_rule": f"|w| < {NEAR_ZERO_REL} * std",
            "gini_abs": gini,
        },
        "rank_spectrum": svd,
        "codebook_affinity": {
            "row_kmeans_k1": km1,
            "row_kmeans_k16": km16,
            "row_kmeans_k64": km64,
            "group8_kmeans_k16": gv16,
            "group8_kmeans_k64": gv64,
            "note": "weight-space k-means; activation-aware VQ is unmeasured (no second 27B decode)",
        },
        "grouped_absmax_rel_fro_g64": bits,
        "cross_head_similarity": heads,
        "frobenius_sq": fro2,
    }
    aux = {"U": U, "S": S, "Vh": Vh, "sketch": sketch, "shape": (m, n)}
    return feat, aux


def mean_or_none(xs: list[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    if not v:
        return None
    return float(sum(v) / len(v))


def project_shared(W: np.ndarray, Vh: np.ndarray) -> float:
    """Affinity of W onto another layer's right singular basis. 1 - rel_fro."""
    W64 = np.asarray(W, dtype=np.float32)
    V = np.asarray(Vh, dtype=np.float32)
    # W ≈ (W @ V.T) @ V   with V = Vh  (k, n)
    recon = (W64 @ V.T) @ V
    denom = float(np.linalg.norm(W64.astype(np.float64)))
    rel = float(np.linalg.norm((W64 - recon).astype(np.float64)) / max(denom, 1e-30))
    return float(max(0.0, 1.0 - rel))


def project_shared_left(W: np.ndarray, U: np.ndarray) -> float:
    W64 = np.asarray(W, dtype=np.float32)
    U32 = np.asarray(U, dtype=np.float32)
    recon = U32 @ (U32.T @ W64)
    denom = float(np.linalg.norm(W64.astype(np.float64)))
    rel = float(np.linalg.norm((W64 - recon).astype(np.float64)) / max(denom, 1e-30))
    return float(max(0.0, 1.0 - rel))


# ---------------------------------------------------------------------------
# organ plan
# ---------------------------------------------------------------------------


def sample_layers_from_env(default: tuple[int, ...], which: str) -> tuple[int, ...]:
    env = os.environ.get("DOCTOR_DIAGNOSIS_LAYERS")
    if env:
        n = int(env)
        if n <= 0:
            return default
        if which == "gqa":
            pool = gqa_layers()
        elif which == "dn":
            pool = dn_layers()
        else:
            pool = tuple(range(LAYERS))
        if n >= len(pool):
            return pool
        # even stride through the legal pool, always include first+last
        if n == 1:
            return (pool[len(pool) // 2],)
        idx = np.linspace(0, len(pool) - 1, n).round().astype(int)
        seen = []
        for i in idx:
            if pool[int(i)] not in seen:
                seen.append(pool[int(i)])
        return tuple(seen)
    return default


def organ_plan() -> list[dict]:
    mlp_L = sample_layers_from_env(DEFAULT_MLP_LAYERS, "mlp")
    gqa_L = sample_layers_from_env(DEFAULT_GQA_LAYERS, "gqa")
    dn_L = sample_layers_from_env(DEFAULT_DN_LAYERS, "dn")
    return [
        {
            "organ_id": "mlp_gate_up",
            "library_organ": "mlp_gate_up",
            "kinds": ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
            "layers": mlp_L,
            "head_spec": None,
            "incumbent": {"family": "scalar quantization", "codec": "q2f / affine2_g64", "bpw": 2.25},
        },
        {
            "organ_id": "mlp_down",
            "library_organ": "mlp_down",
            "kinds": ["mlp.down_proj.weight"],
            "layers": mlp_L,
            "head_spec": None,
            "incumbent": {"family": "scalar quantization", "codec": "q2f / affine2_g64", "bpw": 2.25},
        },
        {
            "organ_id": "gqa_q",
            "library_organ": "gqa_attention",
            "kinds": ["self_attn.q_proj.weight"],
            "layers": gqa_L,
            "head_spec": {"n_heads": GQA_HEADS, "head_dim": GQA_HEAD_DIM, "axis": "out"},
            "incumbent": {"family": "scalar quantization", "codec": "ws_rtn_q3_g128", "bpw": 3.125},
            "note": "q_proj is 12288 = 24 heads * 256 * 2 (attn_output_gate); cross-head uses the query slice",
        },
        {
            "organ_id": "gqa_k",
            "library_organ": "gqa_attention",
            "kinds": ["self_attn.k_proj.weight"],
            "layers": gqa_L,
            "head_spec": {"n_heads": GQA_KV_HEADS, "head_dim": GQA_HEAD_DIM, "axis": "out"},
            "incumbent": {"family": "scalar quantization", "codec": "ws_rtn_q3_g128", "bpw": 3.125},
        },
        {
            "organ_id": "gqa_v",
            "library_organ": "gqa_attention",
            "kinds": ["self_attn.v_proj.weight"],
            "layers": gqa_L,
            "head_spec": {"n_heads": GQA_KV_HEADS, "head_dim": GQA_HEAD_DIM, "axis": "out"},
            "incumbent": {"family": "scalar quantization", "codec": "ws_rtn_q3_g128", "bpw": 3.125},
        },
        {
            "organ_id": "gqa_o",
            "library_organ": "gqa_attention",
            "kinds": ["self_attn.o_proj.weight"],
            "layers": gqa_L,
            "head_spec": {"n_heads": GQA_HEADS, "head_dim": GQA_HEAD_DIM, "axis": "in"},
            "incumbent": {"family": "scalar quantization", "codec": "ws_rtn_q3_g128", "bpw": 3.125},
        },
        {
            "organ_id": "deltanet_in_proj",
            "library_organ": "deltanet",
            "kinds": ["linear_attn.in_proj_qkv.weight"],
            "layers": dn_L,
            "head_spec": {"n_heads": DN_K_HEADS, "head_dim": DN_HEAD_DIM, "axis": "out"},
            "incumbent": {"family": "scalar quantization", "codec": "ws_rtn_q3_g64", "bpw": 3.25},
            "note": "primary in_proj is in_proj_qkv (q/k/v concatenated). in_proj_z/a/b are smaller and not mixed into this organ's rank.",
        },
        {
            "organ_id": "deltanet_out_proj",
            "library_organ": "deltanet",
            "kinds": ["linear_attn.out_proj.weight"],
            "layers": dn_L,
            "head_spec": {"n_heads": DN_V_HEADS, "head_dim": DN_HEAD_DIM, "axis": "in"},
            "incumbent": {"family": "scalar quantization", "codec": "ws_rtn_q3_g64", "bpw": 3.25},
        },
        {
            "organ_id": "embed",
            "library_organ": "embed",
            "kinds": None,
            "single_tensor": "model.language_model.embed_tokens.weight",
            "layers": (),
            "head_spec": None,
            "incumbent": {"family": "scalar quantization", "codec": "ws_rtn_q3_g128", "bpw": 3.125},
            "row_sample": True,
        },
        {
            "organ_id": "lm_head",
            "library_organ": "lm_head",
            "kinds": None,
            "single_tensor": "lm_head.weight",
            "layers": (),
            "head_spec": None,
            "incumbent": {"family": "scalar quantization", "codec": "ws_rtn_q3_g128", "bpw": 3.125},
            "row_sample": True,
        },
    ]


# ---------------------------------------------------------------------------
# parent-A catalog (read-only)
# ---------------------------------------------------------------------------


def parent_a_fingerprint(root: Path) -> dict:
    cat = root / "catalog.hq38m20"
    mix = root / "MIX_REPORT.json"
    out = {
        "path": str(root),
        "catalog": str(cat),
        "catalog_present": cat.is_file(),
        "mix_report_present": mix.is_file(),
    }
    if cat.is_file():
        st = cat.stat()
        out["catalog_sha256"] = sha256_file(cat)
        out["catalog_size"] = int(st.st_size)
        out["catalog_mtime_ns"] = int(st.st_mtime_ns)
    if mix.is_file():
        out["mix_sha256"] = sha256_file(mix)
        out["mix_size"] = int(mix.stat().st_size)
    return out


def inspect_parent_a_catalog(root: Path) -> dict:
    """Read-only HQ38M20 walk. Does not open weight segments."""
    cat = root / "catalog.hq38m20"
    raw = cat.read_bytes()
    if raw[:8] != CATALOG_MAGIC:
        raise ValueError("catalog magic is not HQ38M20")
    version = struct.unpack_from("<I", raw, 8)[0]
    n_tensors = struct.unpack_from("<I", raw, 12)[0]
    n_segments = struct.unpack_from("<I", raw, 16)[0]
    name_blob_bytes = struct.unpack_from("<I", raw, 24)[0]
    cursor = 32
    for _ in range(n_segments):
        name_len = struct.unpack_from("<H", raw, cursor + 2)[0]
        cursor += 44 + name_len
    table = raw[cursor : cursor + n_tensors * RECORD_SIZE]
    cursor += n_tensors * RECORD_SIZE
    name_blob = raw[cursor : cursor + name_blob_bytes]
    codecs: dict[str, int] = {}
    organs_present = {oid: 0 for oid in (
        "mlp_gate_up", "mlp_down", "gqa_q", "gqa_k", "gqa_v", "gqa_o",
        "deltanet_in_proj", "deltanet_out_proj", "embed", "lm_head",
    )}
    names_head = []
    for i in range(n_tensors):
        rec = table[i * RECORD_SIZE : (i + 1) * RECORD_SIZE]
        name_off = struct.unpack_from("<I", rec, 0)[0]
        name_len = struct.unpack_from("<H", rec, 4)[0]
        codec = rec[6]
        name = name_blob[name_off : name_off + name_len].decode("utf-8")
        codecs[str(codec)] = codecs.get(str(codec), 0) + 1
        if i < 8:
            names_head.append(name)
        if "mlp.gate_proj" in name or "mlp.up_proj" in name:
            organs_present["mlp_gate_up"] += 1
        elif "mlp.down_proj" in name:
            organs_present["mlp_down"] += 1
        elif "self_attn.q_proj" in name:
            organs_present["gqa_q"] += 1
        elif "self_attn.k_proj" in name:
            organs_present["gqa_k"] += 1
        elif "self_attn.v_proj" in name:
            organs_present["gqa_v"] += 1
        elif "self_attn.o_proj" in name:
            organs_present["gqa_o"] += 1
        elif "linear_attn.in_proj" in name:
            organs_present["deltanet_in_proj"] += 1
        elif "linear_attn.out_proj" in name:
            organs_present["deltanet_out_proj"] += 1
        elif "embed_tokens" in name:
            organs_present["embed"] += 1
        elif name.endswith("lm_head.weight") or name == "output.weight":
            organs_present["lm_head"] += 1
    return {
        "magic": "HQ38M20",
        "version": int(version),
        "n_tensors": int(n_tensors),
        "n_segments": int(n_segments),
        "codecs": codecs,
        "organs_present_in_catalog": organs_present,
        "names_head": names_head,
        "weights_not_dequantized": True,
        "why": (
            "NOETIC_PARENT_A is the sealed MIX (affine2 MLP + HQ30UQ4 attn). "
            "Diagnosis needs the REAL parent tensors, so GEMVs are streamed from "
            "the qualified BF16 parent. This catalog is read-only evidence that "
            "the named organs exist on the sealed leader."
        ),
    }


# ---------------------------------------------------------------------------
# N043 / N046 / negative science loaders
# ---------------------------------------------------------------------------


def load_n043() -> dict:
    path = first_existing(N043_CANDIDATES)
    if path is None:
        try:
            raw = subprocess.check_output(
                ["git", "show", "HEAD:receipts/headless/DOCTOR_TECHNIQUE_REGISTRY.json"],
                cwd=REPO, timeout=30,
            )
            return {"loaded": True, "path": "git:HEAD:receipts/headless/DOCTOR_TECHNIQUE_REGISTRY.json",
                    "doc": json.loads(raw)}
        except Exception:
            return {"loaded": False, "path": None, "doc": None}
    return {"loaded": True, "path": str(path), "doc": load_json(path)}


def load_n046() -> dict:
    path = first_existing(N046_CANDIDATES)
    if path is None:
        return {"loaded": False, "path": None, "doc": None}
    return {"loaded": True, "path": str(path), "doc": load_json(path)}


def n043_by_id(n043: dict) -> dict[str, dict]:
    doc = n043.get("doc") or {}
    out = {}
    for t in doc.get("techniques") or []:
        tid = ((t.get("technique_identity") or {}).get("id"))
        if tid:
            out[tid] = t
    return out


def n043_verdict(tech: dict | None) -> dict:
    if not tech:
        return {"status": "ABSENT_FROM_REGISTRY", "literature_is": "HYPOTHESIS"}
    v = tech.get("current_verdict") or {}
    return {
        "status": v.get("status"),
        "literature_is": v.get("literature_is") or "HYPOTHESIS",
        "hawking_receipts": v.get("hawking_receipts") or [],
        "technique_id": (tech.get("technique_identity") or {}).get("id"),
        "short_name": (tech.get("technique_identity") or {}).get("short_name"),
    }


def literature_for_organ(n046: dict, organ_id: str) -> list[dict]:
    doc = n046.get("doc") or {}
    if not doc:
        return []
    lib = {
        "mlp_gate_up": "mlp",
        "mlp_down": "mlp",
        "gqa_q": "gqa",
        "gqa_k": "gqa",
        "gqa_v": "gqa",
        "gqa_o": "gqa",
        "deltanet_in_proj": "delta",
        "deltanet_out_proj": "delta",
        "embed": "embed",
        "lm_head": "lm_head",
    }[organ_id]
    hits = []
    for t in doc.get("techniques") or []:
        organ = str(t.get("hawking_organ") or "").lower()
        if lib not in organ and organ_id not in organ:
            # some rows say "mlp_gate_up" etc.
            if not any(s in organ for s in (lib, organ_id.replace("_", " "))):
                continue
        if t.get("ADD_TO_REGISTRY") not in (True, "yes", "YES"):
            continue
        hits.append({
            "name": t.get("name"),
            "arxiv_id": t.get("arxiv_id"),
            "family": t.get("family"),
            "ADD_TO_REGISTRY": True,
            "literature_status": "HYPOTHESIS",
            "not_authority": True,
        })
    # recommended_additions may not carry organ; skip if already collected
    return hits[:8]


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------


def _num(d: dict, *path, default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def score_families(agg: dict, organ_id: str) -> list[dict]:
    """Score §65 families from measured features. Higher = more plausible as a probe."""
    kurt = float(_num(agg, "weight_distribution", "excess_kurtosis", default=0.0) or 0.0)
    out_mad = float(_num(agg, "weight_distribution", "outlier_frac_6mad", default=0.0) or 0.0)
    sparse = float(_num(agg, "sparsity", "frac_near_zero", default=0.0) or 0.0)
    er_frac = float(_num(agg, "rank_spectrum", "effective_rank_frac_of_min_dim", default=1.0) or 1.0)
    captured = float(_num(agg, "rank_spectrum", "captured_energy_frac", default=0.0) or 0.0)
    share = float(_num(agg, "shared_basis_affinity", "mean_right_affinity", default=0.0) or 0.0)
    xlayer = float(_num(agg, "cross_layer_similarity", "mean_sketch_cosine", default=0.0) or 0.0)
    cb16 = float(_num(agg, "codebook_affinity", "row_kmeans_k16", "affinity", default=0.0) or 0.0)
    cb64 = float(_num(agg, "codebook_affinity", "row_kmeans_k64", "affinity", default=0.0) or 0.0)
    gv16 = float(_num(agg, "codebook_affinity", "group8_kmeans_k16", "affinity", default=0.0) or 0.0)
    b1 = float(_num(agg, "grouped_absmax_rel_fro_g64", "1_meanabs", default=None) or
               _num(agg, "grouped_absmax_rel_fro_g64", "1", default=1.0) or 1.0)
    b2 = float(_num(agg, "grouped_absmax_rel_fro_g64", "2", default=1.0) or 1.0)
    b3 = float(_num(agg, "grouped_absmax_rel_fro_g64", "3", default=1.0) or 1.0)
    xhead = _num(agg, "cross_head_similarity", "mean_cosine", default=None)
    xhead_v = float(xhead) if xhead is not None else None
    layer_hetero = 1.0 - max(0.0, min(1.0, abs(xlayer)))

    # Scores in [0, 100]. These are predicted expected-value, not certificates.
    scores: dict[str, float] = {k: 0.0 for k, _ in FAMILIES}

    # Incumbent scalar quantization is always a live baseline.
    scores["scalar_quantization"] = 55.0 + 20.0 * max(0.0, 1.0 - b3) + 5.0

    # Binary / ternary: weight-space grouped-absmax friendliness.
    scores["binary"] = 15.0 + 70.0 * max(0.0, 1.0 - b1) - 15.0 * min(1.0, abs(kurt) / 10.0)
    scores["ternary"] = 20.0 + 70.0 * max(0.0, 1.0 - b2) - 8.0 * min(1.0, abs(kurt) / 10.0)
    scores["trit_plane"] = 0.65 * scores["ternary"] + 10.0 * max(0.0, 1.0 - captured)

    # Codebooks
    scores["vector_codebook"] = 10.0 + 80.0 * max(cb16, gv16)
    scores["additive_codebook"] = 8.0 + 70.0 * cb64 + 10.0 * gv16
    # Adaptive codebooks need BOTH a codebook fit AND layer heterogeneity.
    # Heterogeneity alone is not codebook affinity.
    scores["adaptive_codebook"] = (
        scores["vector_codebook"] * 0.85 + 15.0 * layer_hetero * max(cb16, gv16)
    )

    # Shared basis: needs BOTH affinity and cross-layer cosine.
    scores["shared_basis"] = 80.0 * (0.5 * share + 0.5 * max(0.0, xlayer))

    # Rank families
    lowrank_signal = max(0.0, 1.0 - min(1.0, er_frac / 0.35)) * captured
    scores["low_rank"] = 85.0 * lowrank_signal
    scores["tensor_factorization"] = 0.8 * scores["low_rank"]
    scores["low_rank_sparse"] = scores["low_rank"] * 0.6 + 30.0 * min(1.0, sparse / 0.2)

    scores["generated_coefficients"] = 5.0 + 20.0 * share  # sharing is necessary, never sufficient
    scores["structured_pruning"] = 5.0 + 80.0 * min(1.0, sparse / 0.3)
    scores["routed_structure"] = 8.0 + 20.0 * layer_hetero
    if xhead_v is not None:
        # near-orthogonal heads = NOT redundant (N050 will own elimination).
        scores["routed_structure"] += 10.0 * max(0.0, 1.0 - abs(xhead_v))
        scores["structured_pruning"] += 5.0 * max(0.0, xhead_v)  # similar heads might prune

    scores["protected_islands"] = 15.0 + 70.0 * min(1.0, out_mad / 0.05) + 15.0 * min(1.0, max(0.0, kurt) / 8.0)

    # Organ-specific prior: DeltaNet paper transfer is weaker.
    if organ_id.startswith("deltanet"):
        for k in ("binary", "ternary", "trit_plane", "shared_basis"):
            scores[k] *= 0.75
        scores["routed_structure"] += 5.0
    if organ_id in {"embed", "lm_head"}:
        scores["shared_basis"] *= 0.4  # one tensor, no layers to share
        scores["routed_structure"] *= 0.3
        scores["structured_pruning"] += 10.0  # dead rows are a tokenizer question (N045)
    if organ_id.startswith("gqa"):
        scores["protected_islands"] += 5.0

    ranked = []
    for fid, label in FAMILIES:
        ranked.append({
            "family_id": fid,
            "family": label,
            "score": float(max(0.0, scores[fid])),
            "role": "incumbent" if fid == "scalar_quantization" else "probe",
            "predicts_not_certifies": True,
            "n043_technique": FAMILY_TO_N043.get(fid),
        })
    ranked.sort(key=lambda r: (-r["score"], r["family_id"]))
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return ranked


def diagnosis_lines(agg: dict, organ_id: str, ranked: list[dict]) -> list[dict]:
    lines = []
    er_frac = float(_num(agg, "rank_spectrum", "effective_rank_frac_of_min_dim", default=1.0) or 1.0)
    captured = float(_num(agg, "rank_spectrum", "captured_energy_frac", default=0.0) or 0.0)
    if er_frac > 0.35 or captured < 0.55:
        lines.append({
            "text": "low-rank weak",
            "source": "measured",
            "feature": "rank_spectrum.effective_rank_frac_of_min_dim",
            "value": er_frac,
            "why": f"entropy effective rank is {er_frac:.3f} of min(m,n) with captured energy {captured:.3f} in top-{SVD_K}",
        })
    elif er_frac > 0.15:
        lines.append({
            "text": "moderate rank; low-rank is a partial fit",
            "source": "measured",
            "feature": "rank_spectrum.effective_rank_frac_of_min_dim",
            "value": er_frac,
            "why": f"effective rank fraction {er_frac:.3f}",
        })
    else:
        lines.append({
            "text": "effective rank low; factorization is plausible as a probe",
            "source": "measured",
            "feature": "rank_spectrum.effective_rank_frac_of_min_dim",
            "value": er_frac,
            "why": f"effective rank fraction {er_frac:.3f}",
        })

    kurt = float(_num(agg, "weight_distribution", "excess_kurtosis", default=0.0) or 0.0)
    out_mad = float(_num(agg, "weight_distribution", "outlier_frac_6mad", default=0.0) or 0.0)
    if kurt > 3.0 or out_mad > 0.01:
        lines.append({
            "text": "rotation may reduce outliers",
            "source": "measured",
            "feature": "weight_distribution.excess_kurtosis",
            "value": kurt,
            "why": f"excess kurtosis {kurt:.3f} (gaussian null 0), outlier_frac_6mad {out_mad:.4f}",
        })
    else:
        lines.append({
            "text": "weight distribution is not outlier-dominated",
            "source": "measured",
            "feature": "weight_distribution.excess_kurtosis",
            "value": kurt,
            "why": f"excess kurtosis {kurt:.3f}, outlier_frac_6mad {out_mad:.4f}",
        })

    share = float(_num(agg, "shared_basis_affinity", "mean_right_affinity", default=0.0) or 0.0)
    if share > 0.6:
        aff_txt = "high"
    elif share > 0.3:
        aff_txt = "moderate"
    else:
        aff_txt = "low"
    lines.append({
        "text": f"shared basis local affinity {aff_txt}",
        "source": "measured",
        "feature": "shared_basis_affinity.mean_right_affinity",
        "value": share,
        "why": f"mean right-basis reconstruction affinity {share:.3f} (1 - rel_fro onto other-layer Vh)",
    })

    if organ_id in {"mlp_gate_up", "mlp_down"}:
        lines.append({
            "text": "binary fast but coherence-sensitive",
            "source": "prior_science",
            "feature": "grouped_absmax_rel_fro_g64.1",
            "value": _num(agg, "grouped_absmax_rel_fro_g64", "1"),
            "why": "BYTES_FRONTIER: binary_g64 moved token_ns; BINARY_HEALING: injury is uniform, 0 heals coherent",
            "receipts": [R_BYTES, R_BINARY],
        })

    cb = float(_num(agg, "codebook_affinity", "row_kmeans_k16", "affinity", default=0.0) or 0.0)
    lines.append({
        "text": (
            "codebook (k-means) affinity "
            + ("high" if cb > 0.6 else "moderate" if cb > 0.35 else "low")
        ),
        "source": "measured",
        "feature": "codebook_affinity.row_kmeans_k16.affinity",
        "value": cb,
        "why": f"row k-means k=16 affinity {cb:.3f} (1 - rel_fro); weight-space only",
    })

    sparse = float(_num(agg, "sparsity", "frac_near_zero", default=0.0) or 0.0)
    lines.append({
        "text": "sparsity " + ("material" if sparse > 0.15 else "low"),
        "source": "measured",
        "feature": "sparsity.frac_near_zero",
        "value": sparse,
        "why": f"frac_near_zero {sparse:.4f}",
    })

    xhead = _num(agg, "cross_head_similarity", "mean_cosine")
    if xhead is not None:
        lines.append({
            "text": (
                "cross-head similarity "
                + ("high (possible redundancy)" if abs(float(xhead)) > 0.5
                   else "low (heads not interchangeable)")
            ),
            "source": "measured",
            "feature": "cross_head_similarity.mean_cosine",
            "value": float(xhead),
            "why": f"mean pairwise head cosine {float(xhead):.3f}",
        })
    return lines


def avoid_list(organ_id: str, agg: dict) -> list[dict]:
    """AVOID entries cite negative science. Diagnosis does not re-run the dead experiment."""
    share = float(_num(agg, "shared_basis_affinity", "mean_right_affinity", default=0.0) or 0.0)
    items: list[dict] = []

    def add(family, experiment, reason, receipts, nns=None, applies=True):
        if not applies:
            return
        items.append({
            "family": family,
            "experiment": experiment,
            "reason": reason,
            "negative_science": list(receipts),
            "nns_ids": list(nns or []),
            "predicts_not_certifies": True,
        })

    mlp = organ_id in {"mlp_gate_up", "mlp_down"}
    gqa = organ_id.startswith("gqa")
    dn = organ_id.startswith("deltanet")

    add(
        "shared basis",
        "identical shared-K2 experiment",
        (
            "SHARED_BASIS_COHERENT is dead: no coherent shared-basis point beats q2f "
            "on both density and ns; K=2 dies at held_out_activation. C1 killed the "
            "family on FIDELITY (G035 shared_beats_independent=false), not on a missing "
            "kernel (SHARED_BASIS_KERNEL is competent). "
            f"This organ's measured local affinity is {share:.3f}, which does not "
            "reopen a fidelity-dead family."
        ),
        [R_SHARED_C, R_SHARED_K, R_C1, R_NNS],
        ["NNS-004", "NNS-012", "NNS-013"],
        applies=mlp or True,  # do not transplant K2 onto any organ without a new diagnosis that beats G035
    )
    add(
        "low-rank",
        "weight-space / activation-aware low-rank as a density replacement of q2f/q3",
        "G034 / C3 / NNS-014 / NNS-016: low-rank is 2.93× q3 error at matched bits; "
        "activation-aware functional low-rank does not beat q3 held-out. Hybrid residual never heals.",
        [R_C3, R_HYBRID, R_NNS],
        ["NNS-014", "NNS-016"],
        applies=mlp,
    )
    add(
        "low-rank + sparse",
        "fused low-rank + sparse correction kernel on Qwen3.8 MLP",
        "C3: NOT_WORTH_BUILDING. Fusion guaranteed save is ~0.007% of token_ns; the approximation family is already refuted as a byte/quality lever.",
        [R_C3, R_HYBRID, R_NNS],
        ["NNS-015"],
        applies=mlp,
    )
    add(
        "tensor factorization",
        "TT / Kronecker / Tucker of a single GEMV as the resident operator",
        "C2 / G034: tensor operators refuted at the coherent point; 223 rows <0.5 bpw had 0 healthy.",
        [R_C2, R_NNS],
        ["NNS-016"],
        applies=mlp or gqa or dn,
    )
    add(
        "binary",
        "unrotated 1.25-bpw binary as a resident MLP",
        "BYTES_FRONTIER: binary_g64 is fast (moved token_ns toward the roof). BINARY_HEALING: uniformly injured, 0 islands restore coherent generation. S026 §14: CURRENT BINARY = incoherent, not 'all binary is impossible'. Retest only after a coordinate change (N044) or as a DRAFT (N049, §63).",
        [R_BYTES, R_BINARY, R_ONEBIT, R_FIRST],
        [],
        applies=mlp,
    )
    add(
        "ternary",
        "unrotated ternary as a reopen of the 2.25 MLP floor",
        "NOETIC_COMPOSITION_WHOLEMODEL_TERNARY closed the unrotated family (argmax flip). QWEN_MLP_2_25 stays CLOSED for that family (S026 §11). A rotated ternary is a DIFFERENT family (N044).",
        [R_TERNARY, R_FRACTIONAL, R_BYTES],
        [],
        applies=mlp,
    )
    add(
        "vector codebook",
        "PQ/VQ of RAW frozen weights as a path to ~1 bpw capability",
        "NNS-017: raw-weight PQ/VQ is not a route to one bit that preserves capability (dominant failure organ = gate). C4: NOT_WORTH_BUILDING_THE_QWEN38_PORT of gravity_pq. Activation-aware additive codebooks remain a HYPOTHESIS (AQLM/VPTQ RELATED_NEGATIVE, not the same experiment).",
        [R_C4, R_NNS, R_ONEBIT],
        ["NNS-017"],
        applies=mlp,
    )
    add(
        "generated coefficients",
        "procedural / generated parameters as stored-bit replacement",
        "G042 / GENERATED_WEIGHTS_RETEST / NNS-024: generated weights REFUTED (GENERATED_BPW=0).",
        [R_GEN, R_C5, R_NNS],
        ["NNS-024"],
        applies=True,
    )
    add(
        "structured pruning",
        "unstructured activation sparsity as a decode win under the coherent floor",
        "NNS-029: activation sparsity is not a clean path under the Qwen3.8 coherent floor. Sparse only if structured + a competent fused path (S026 §27-28; N033/CSR lesson).",
        [R_NNS],
        ["NNS-029"],
        applies=mlp,
    )
    if dn:
        add(
            "scalar quantization",
            "blind transplant of transformer PTQ papers onto DeltaNet",
            "S026 §103: paper techniques are NOT assumed to transfer from transformers. Diagnose DeltaNet first (this receipt). Quamba2 etc. stay HYPOTHESIS (N046).",
            [R_NNS, R_FRONTIERS],
            [],
            applies=True,
        )
    if organ_id in {"embed", "lm_head"}:
        add(
            "structured pruning",
            "ASCII-only vocabulary prune as the default",
            "S026 §32: ASCII-only is NOT the default. Tokenizer gravity (N045) owns token-inflation accounting. Weight kurtosis here does not license dropping rows.",
            [R_FLOORS],
            [],
            applies=True,
        )
    return items


def prescription_steps(
    organ_id: str, agg: dict, ranked: list[dict], n043_map: dict[str, dict], lit: list[dict]
) -> list[dict]:
    """Concrete §67 PRESCRIPTION steps. Ranked by expected value of the next experiment."""
    steps: list[dict] = []
    kurt = float(_num(agg, "weight_distribution", "excess_kurtosis", default=0.0) or 0.0)
    out_mad = float(_num(agg, "weight_distribution", "outlier_frac_6mad", default=0.0) or 0.0)
    probes = [r for r in ranked if r["role"] == "probe" and r["family_id"] not in
              {"generated_coefficients", "tensor_factorization"}]
    # drop families we AVOID as resident-probes for this organ; they can still be listed as 'avoided'
    avoid_ids = {a["family"] for a in avoid_list(organ_id, agg)}
    live = [r for r in probes if r["family"] not in avoid_ids or r["family_id"] in
            {"protected_islands", "adaptive_codebook", "trit_plane"}]
    # Always keep protected islands / rotation even if related-negative.

    n = 1
    if kurt > 2.0 or out_mad > 0.008 or organ_id in {"mlp_gate_up", "mlp_down", "gqa_q", "gqa_o"}:
        spin = n043_verdict(n043_map.get("spinquant"))
        steps.append({
            "rank": n,
            "action": "learn function-preserving rotation",
            "family": "coordinate transform (not a §65 storage family; enables them)",
            "motivating_feature": "weight_distribution.excess_kurtosis",
            "motivating_value": kurt,
            "why": (
                f"excess kurtosis {kurt:.3f}, outlier_frac_6mad {out_mad:.4f}. "
                "S026 §8/§10: search T such that representation_cost(T(W)) << cost(W)."
            ),
            "n043": spin,
            "literature_hypotheses": [x for x in lit if "rotat" in str(x.get("family") or "").lower()][:3],
            "next_obligation": "N044 COORDINATE_TRANSFORM_PROBE" if organ_id.startswith("mlp") else None,
            "predicts_not_certifies": True,
        })
        n += 1

    retest = []
    for r in live[:4]:
        if r["family_id"] in {"binary", "ternary", "trit_plane"}:
            retest.append(r["family"])
    if organ_id in {"mlp_gate_up", "mlp_down"}:
        retest = ["ternary", "binary"] + [x for x in retest if x not in {"ternary", "binary"}]
    if retest:
        steps.append({
            "rank": n,
            "action": "retest " + "/".join(retest[:3]) + " in the resulting coordinates",
            "family": retest[0] if retest else "ternary",
            "motivating_feature": "grouped_absmax_rel_fro_g64",
            "motivating_value": _num(agg, "grouped_absmax_rel_fro_g64"),
            "why": (
                "Unrotated binary/ternary are closed as residents on MLP. "
                "S026 §11: a coordinate change is a legal reopen condition. "
                "This is a probe, not a certificate that the floor moves."
            ),
            "n043": n043_verdict(n043_map.get("twla" if "ternary" in retest else "onebit")),
            "predicts_not_certifies": True,
        })
        n += 1

    steps.append({
        "rank": n,
        "action": "protect top sensitivity channels",
        "family": "protected islands",
        "motivating_feature": "weight_distribution.outlier_frac_6mad",
        "motivating_value": out_mad,
        "why": (
            "SqueezeLLM-style dense+sparse: outliers/kurtosis identify WHERE bits "
            "must remain. N051 owns the sensitivity curve; this diagnosis only "
            "flags that the weight distribution is not uniform-safe."
        ),
        "n043": n043_verdict(n043_map.get("squeezellm")),
        "predicts_not_certifies": True,
    })
    n += 1

    top_probe = next((r for r in live if r["family_id"] not in {"protected_islands"}), None)
    steps.append({
        "rank": n,
        "action": "compile fused native operator",
        "family": (top_probe or {}).get("family") or "scalar quantization",
        "motivating_feature": "incumbent kernel competency gate (S026 §90)",
        "motivating_value": None,
        "why": (
            "A representation cannot be condemned until its native kernel is competent. "
            "Do not reconstruct-to-dense. Trit-plane/codebook/sparse must execute native "
            "(S026 §17, §27, §28)."
        ),
        "n043": None,
        "predicts_not_certifies": True,
    })
    n += 1

    if organ_id.startswith("gqa_k") or organ_id.startswith("gqa_v"):
        steps.append({
            "rank": n,
            "action": "do not confuse this weight diagnosis with KIVI/MiniCache/H2O (those are STATE; N048)",
            "family": "scalar quantization",
            "motivating_feature": "cross_head_similarity.mean_cosine",
            "motivating_value": _num(agg, "cross_head_similarity", "mean_cosine"),
            "why": "KIVI/MiniCache/H2O act on KV/state, not on q/k/v/o storage. UNTESTED in N043.",
            "n043": n043_verdict(n043_map.get("kivi")),
            "predicts_not_certifies": True,
        })
    if organ_id.startswith("deltanet"):
        steps.append({
            "rank": n,
            "action": "treat DeltaNet as its own doctor (parameter + transition + state)",
            "family": "scalar quantization",
            "motivating_feature": "rank_spectrum.effective_rank_frac_of_min_dim",
            "motivating_value": _num(agg, "rank_spectrum", "effective_rank_frac_of_min_dim"),
            "why": "S026 §103. Quamba2 / Sparse Delta Memory are N046 HYPOTHESES, not Metal results.",
            "literature_hypotheses": lit[:3],
            "predicts_not_certifies": True,
        })
    if organ_id in {"embed", "lm_head"}:
        steps.append({
            "rank": n,
            "action": "route vocabulary elimination to N045 (token-inflation), not to this weight diagnosis",
            "family": "structured pruning",
            "motivating_feature": "sparsity.frac_near_zero",
            "motivating_value": _num(agg, "sparsity", "frac_near_zero"),
            "why": "LM head removes work from EVERY decode step (S026 §36) but only after honest TOKEN_INFLATION.",
            "predicts_not_certifies": True,
        })
    return steps


def section_67(organ_id: str, lines: list[dict], steps: list[dict], avoid: list[dict]) -> dict:
    return {
        "ORGAN": organ_id,
        "DIAGNOSIS": [ln["text"] for ln in lines],
        "PRESCRIPTION": [f"{s['rank']}. {s['action']}" for s in steps],
        "AVOID": [
            {"item": a["experiment"], "reason": a["reason"], "family": a["family"],
             "negative_science": a["negative_science"]}
            for a in avoid
        ],
    }


# ---------------------------------------------------------------------------
# aggregation + organ run
# ---------------------------------------------------------------------------


def _mean_dict(dicts: list[dict], keys: tuple[str, ...]) -> dict:
    out: dict[str, Any] = {}
    for k in keys:
        vals = []
        for d in dicts:
            if not isinstance(d, dict) or k not in d:
                continue
            v = d[k]
            if isinstance(v, (int, float)) and v is not None:
                vals.append(float(v))
        out[k] = float(sum(vals) / len(vals)) if vals else None
    return out


def aggregate_features(per: list[dict], shared: dict, xlayer: dict) -> dict:
    dists = [p["weight_distribution"] for p in per if p.get("weight_distribution")]
    spars = [p["sparsity"] for p in per if p.get("sparsity")]
    ranks = [p["rank_spectrum"] for p in per if p.get("rank_spectrum")]
    cbs = [p["codebook_affinity"] for p in per if p.get("codebook_affinity")]
    bits = [p.get("grouped_absmax_rel_fro_g64") or {} for p in per]
    heads = [p["cross_head_similarity"] for p in per if p.get("cross_head_similarity")]

    def mean_key(rows, *path):
        vals = []
        for r in rows:
            v = r
            ok = True
            for p in path:
                if not isinstance(v, dict) or p not in v:
                    ok = False
                    break
                v = v[p]
            if ok and isinstance(v, (int, float)) and v is not None:
                vals.append(float(v))
        return float(sum(vals) / len(vals)) if vals else None

    bit_mean = {}
    for b in ("1", "2", "3", "4", "1_meanabs"):
        bit_mean[b] = mean_key(bits, b)

    return {
        "n_tensors_measured": len(per),
        "weight_distribution": {
            "mean": mean_key(dists, "mean"),
            "std": mean_key(dists, "std"),
            "mean_abs": mean_key(dists, "mean_abs"),
            "median_abs": mean_key(dists, "median_abs"),
            "excess_kurtosis": mean_key(dists, "excess_kurtosis"),
            "excess_kurtosis_null": 0.0,
            "excess_kurtosis_null_kind": "gaussian",
            "outlier_frac_6mad": mean_key(dists, "outlier_frac_6mad"),
            "outlier_frac_3std": mean_key(dists, "outlier_frac_3std"),
            "gini_abs": mean_key(dists, "gini_abs"),
            "min": mean_key(dists, "min"),
            "max": mean_key(dists, "max"),
        },
        "sparsity": {
            "frac_near_zero": mean_key(spars, "frac_near_zero"),
            "near_zero_rule": f"|w| < {NEAR_ZERO_REL} * std",
            "gini_abs": mean_key(spars, "gini_abs"),
        },
        "rank_spectrum": {
            "k_computed": SVD_K,
            "captured_energy_frac": mean_key(ranks, "captured_energy_frac"),
            "residual_energy_frac": mean_key(ranks, "residual_energy_frac"),
            "participation_ratio": mean_key(ranks, "participation_ratio"),
            "entropy_effective_rank": mean_key(ranks, "entropy_effective_rank"),
            "effective_rank_frac_of_min_dim": mean_key(ranks, "effective_rank_frac_of_min_dim"),
            "rank_for_90pct_energy": mean_key(ranks, "rank_for_90pct_energy"),
            "rank_for_99pct_energy": mean_key(ranks, "rank_for_99pct_energy"),
            "flat_spectrum_implied_rank": mean_key(ranks, "flat_spectrum_implied_rank"),
            "spectrum_is_truncated": True,
            "residual_model": "uniform_over_uncomputed_modes",
            "null": "full-rank Gaussian participation_ratio ≈ min(m,n)",
        },
        "codebook_affinity": {
            "row_kmeans_k1": {
                "affinity": mean_key(cbs, "row_kmeans_k1", "affinity"),
                "rel_fro": mean_key(cbs, "row_kmeans_k1", "rel_fro"),
            },
            "row_kmeans_k16": {
                "affinity": mean_key(cbs, "row_kmeans_k16", "affinity"),
                "rel_fro": mean_key(cbs, "row_kmeans_k16", "rel_fro"),
            },
            "row_kmeans_k64": {
                "affinity": mean_key(cbs, "row_kmeans_k64", "affinity"),
                "rel_fro": mean_key(cbs, "row_kmeans_k64", "rel_fro"),
            },
            "group8_kmeans_k16": {
                "affinity": mean_key(cbs, "group8_kmeans_k16", "affinity"),
                "rel_fro": mean_key(cbs, "group8_kmeans_k16", "rel_fro"),
            },
            "group8_kmeans_k64": {
                "affinity": mean_key(cbs, "group8_kmeans_k64", "affinity"),
                "rel_fro": mean_key(cbs, "group8_kmeans_k64", "rel_fro"),
            },
            "note": "weight-space k-means; not activation-aware",
        },
        "grouped_absmax_rel_fro_g64": bit_mean,
        "cross_head_similarity": {
            "mean_cosine": mean_key(heads, "mean_cosine"),
            "min_cosine": mean_key(heads, "min_cosine"),
            "max_cosine": mean_key(heads, "max_cosine"),
            "n_heads": heads[0].get("n") if heads else None,
            "null": "independent Gaussian heads → cosine ≈ 0",
        } if heads else None,
        "shared_basis_affinity": shared,
        "cross_layer_similarity": xlayer,
    }


def diagnose_organ(parent: Path, spec: dict, rng: np.random.Generator) -> dict:
    organ_id = spec["organ_id"]
    per: list[dict] = []
    snaps: list[dict] = []
    tensors_seen: list[str] = []
    recon_right: list[float] = []
    recon_left: list[float] = []
    t0 = time.time()

    def consume(name: str, W: np.ndarray, sample_note: dict | None = None) -> None:
        feat, aux = features_of_matrix(
            W, rng=rng, head_spec=spec.get("head_spec"), sample_note=sample_note
        )
        feat["tensor"] = name
        kind_key = name.split(".layers.")[-1].split(".", 1)[-1] if ".layers." in name else name
        aux["kind_key"] = kind_key
        # Reconstruction affinity onto earlier layers of the SAME tensor kind,
        # computed while W is resident (one tensor at a time).
        pair_r, pair_l = [], []
        for prev in snaps:
            if prev.get("kind_key") != kind_key:
                continue
            if prev["Vh"] is None or aux["Vh"] is None:
                continue
            if prev["shape"] != aux["shape"]:
                continue
            pair_r.append(project_shared(W, prev["Vh"]))
            if prev["U"] is not None:
                pair_l.append(project_shared_left(W, prev["U"]))
        feat["shared_basis_vs_previous"] = {
            "n_previous": len(pair_r),
            "mean_right_recon_affinity": float(sum(pair_r) / len(pair_r)) if pair_r else None,
            "mean_left_recon_affinity": float(sum(pair_l) / len(pair_l)) if pair_l else None,
        }
        recon_right.extend(pair_r)
        recon_left.extend(pair_l)
        per.append(feat)
        snaps.append(aux)
        tensors_seen.append(name)

    if spec.get("single_tensor"):
        name = spec["single_tensor"]
        print(f"  [{organ_id}] stream {name}", flush=True)
        if spec.get("row_sample"):
            # moments over ALL rows (chunked); SVD/k-means on a row sample
            moments = iter_rows_stats(parent, name)
            W, note = load_tensor_row_sample(parent, name, EMBED_ROW_SAMPLE, rng)
            note["full_pass_moments"] = {
                "mean": moments["mean"],
                "std": moments["std"],
                "mean_abs": moments["mean_abs"],
                "min": moments["min"],
                "max": moments["max"],
                "frac_abs_lt_1e8": moments["frac_abs_lt_1e8"],
                "n": moments["n"],
            }
            consume(name, W, note)
            # overwrite distribution with full-pass moments (more honest)
            per[-1]["weight_distribution"]["mean"] = moments["mean"]
            per[-1]["weight_distribution"]["std"] = moments["std"]
            per[-1]["weight_distribution"]["mean_abs"] = moments["mean_abs"]
            per[-1]["weight_distribution"]["min"] = moments["min"]
            per[-1]["weight_distribution"]["max"] = moments["max"]
            per[-1]["sparsity"]["frac_abs_lt_1e8_full"] = moments["frac_abs_lt_1e8"]
            per[-1]["frobenius_sq_full_pass"] = moments["frobenius_sq"]
            per[-1]["frobenius_sq_row_sample"] = per[-1]["frobenius_sq"]
            if per[-1].get("rank_spectrum"):
                # Spectrum is of the ROW SAMPLE. Do not divide by full-matrix
                # ||W||_F^2 — that would make captured_energy_frac look ~0.
                per[-1]["rank_spectrum"]["computed_on"] = "uniform_row_sample"
                per[-1]["rank_spectrum"]["full_matrix_shape"] = moments["shape"]
            del W
        else:
            W = load_tensor(parent, name)
            consume(name, W)
            del W
    else:
        for layer in spec["layers"]:
            for kind in spec["kinds"]:
                name = tname(layer, kind)
                print(f"  [{organ_id}] L{layer} {kind}", flush=True)
                W = load_tensor(parent, name)
                consume(name, W, {"kind": "full_matrix", "layer": int(layer)})
                del W

    # cross-layer sketch cosine + shared-basis affinities
    right_aff: list[float] = []
    left_aff: list[float] = []
    sketch_cos: list[float] = []
    for i in range(len(snaps)):
        for j in range(i + 1, len(snaps)):
            a, b = snaps[i], snaps[j]
            if a.get("kind_key") != b.get("kind_key"):
                continue
            if a["Vh"] is None or b["Vh"] is None:
                continue
            if a["shape"][1] != b["shape"][1] or a["shape"][0] != b["shape"][0]:
                continue
            # sketches
            sa, sb = a["sketch"], b["sketch"]
            denom = float(np.linalg.norm(sa) * np.linalg.norm(sb))
            sketch_cos.append(float(np.dot(sa, sb) / max(denom, 1e-30)))
            # We no longer have W; cannot project now. Compute affinities
            # while we still had W? We didn't store W. Recompute from
            # stored U/Vh overlap instead (subspace affinity).
            # Principal-angle proxy: ||Vh_a Vh_b.T||_F^2 / k
            Va, Vb = a["Vh"], b["Vh"]
            k = min(Va.shape[0], Vb.shape[0])
            M = Va[:k] @ Vb[:k].T
            right_aff.append(float(np.square(M).sum() / max(k, 1)))
            Ua, Ub = a["U"], b["U"]
            if Ua is not None and Ub is not None and Ua.shape[0] == Ub.shape[0]:
                kU = min(Ua.shape[1], Ub.shape[1])
                MU = Ua[:, :kU].T @ Ub[:, :kU]
                left_aff.append(float(np.square(MU).sum() / max(kU, 1)))

    shared = {
        "mean_right_affinity": (
            float(sum(recon_right) / len(recon_right)) if recon_right
            else (float(sum(right_aff) / len(right_aff)) if right_aff else None)
        ),
        "mean_left_affinity": (
            float(sum(recon_left) / len(recon_left)) if recon_left
            else (float(sum(left_aff) / len(left_aff)) if left_aff else None)
        ),
        "mean_right_grassmann": float(sum(right_aff) / len(right_aff)) if right_aff else None,
        "mean_left_grassmann": float(sum(left_aff) / len(left_aff)) if left_aff else None,
        "n_recon_pairs": len(recon_right),
        "n_pairs": len(right_aff),
        "method": (
            "right/left reconstruction affinity = 1 - rel_fro of W onto another "
            "layer's truncated SVD_K basis, computed while each tensor is resident; "
            "also Grassmann ||Va Vb^T||_F^2 / k on the same bases"
        ),
        "null": "independent Gaussians → recon affinity ≈ captured_energy_frac of the donor basis, Grassmann ≈ k/min_dim",
        "note": (
            "This is local subspace overlap, not G035 function-space "
            "shared_beats_independent. Moderate overlap does not reopen C1."
        ),
    }
    xlayer = {
        "mean_sketch_cosine": float(sum(sketch_cos) / len(sketch_cos)) if sketch_cos else None,
        "n_pairs": len(sketch_cos),
        "sketch_dim": SKETCH_DIM,
        "null": "independent Gaussians → cosine ≈ 0",
    }
    if spec.get("single_tensor"):
        shared = {
            "mean_right_affinity": None,
            "mean_left_affinity": None,
            "n_pairs": 0,
            "method": None,
            "note": "single tensor (no cross-layer pairs)",
        }
        xlayer = {
            "mean_sketch_cosine": None,
            "n_pairs": 0,
            "note": "single tensor (no cross-layer pairs)",
        }

    agg = aggregate_features(per, shared, xlayer)
    return {
        "organ_id": organ_id,
        "library_organ": spec["library_organ"],
        "parent_tensors": tensors_seen,
        "layers": list(spec.get("layers") or []),
        "real_organ_of_qualified_parent": True,
        "incumbent": spec.get("incumbent"),
        "note": spec.get("note"),
        "diagnosis_features": agg,
        "per_tensor": per,
        "wall_s": time.time() - t0,
    }


def attach_prescription(organ: dict, n043_map: dict[str, dict], lit: list[dict]) -> dict:
    agg = organ["diagnosis_features"]
    ranked = score_families(agg, organ["organ_id"])
    lines = diagnosis_lines(agg, organ["organ_id"], ranked)
    avoid = avoid_list(organ["organ_id"], agg)
    steps = prescription_steps(organ["organ_id"], agg, ranked, n043_map, lit)
    organ["diagnosis"] = lines
    organ["ranked_families"] = ranked
    organ["prescription"] = steps
    organ["avoid"] = avoid
    organ["section_67"] = section_67(organ["organ_id"], lines, steps, avoid)
    organ["predicts_not_certifies"] = True
    # every recommendation cites a diagnostic feature
    organ["every_recommendation_cites_a_diagnostic_feature"] = all(
        s.get("motivating_feature") for s in steps
    )
    organ["every_avoid_cites_negative_science"] = all(
        a.get("negative_science") for a in avoid
    )
    return organ


# ---------------------------------------------------------------------------
# receipt
# ---------------------------------------------------------------------------


def unmeasured_block() -> dict:
    return {
        "hessian_sensitivity_proxy": {
            "status": "ABSENT",
            "reason": "N051 owns activation-aware second-order sensitivity; this lane does not run a second 27B decode",
        },
        "activation_distribution": {
            "status": "ABSENT",
            "reason": "no second 27B decode; weight-space diagnosis only. Real-X functional probes live in N011/N040/N044",
        },
        "cross_expert_similarity": {
            "status": "ABSENT",
            "reason": "Qwen3.8 is not MoE; no routed experts",
        },
        "state_redundancy": {
            "status": "ABSENT",
            "reason": "N048 STATE_GRAVITY owns KV/DeltaNet state",
        },
        "token_frequency": {
            "status": "ABSENT",
            "reason": "N045 TOKENIZER_GRAVITY owns vocabulary statistics",
        },
        "vocabulary_compositionality": {
            "status": "ABSENT",
            "reason": "N045 TOKENIZER_GRAVITY",
        },
    }


def build_docs() -> str:
    return """# Doctor Diagnosis

S026 §66, §67. Family: DOC-DIAGNOSIS. Obligation N047.

Machine receipt: `receipts/headless/DOCTOR_DIAGNOSIS.json`, generated by
`tools/headless/doctor_diagnosis.py`. This document is the law; the JSON is
the census. Numbers live in the JSON.

Diagnosis PREDICTS which S026 §65 representation family is plausible per
organ. It does not certify. It does not reopen `QWEN_MLP_2_25` for the
unrotated family (S026 §11). Every prescription cites the diagnostic
feature that motivates it. Every AVOID cites a negative-science receipt.

## Features measured (S026 §66)

On real parent tensors, streamed one at a time from the qualified BF16
parent (CPU, numpy). `~/noetic/NOETIC_PARENT_A` is read-only.

- weight distribution + kurtosis / outlier fraction
- singular-value / rank spectrum (truncated randomized SVD + residual energy)
- cross-layer similarity (Rademacher sketch cosine)
- cross-head similarity (GQA q/k/v/o, DeltaNet in/out)
- shared-basis affinity (Grassmann overlap of top right/left singular vectors)
- codebook (k-means) affinity
- sparsity
- grouped-absmax weight-space error at 1/2/3/4 bits (binary/ternary friendliness)

Unmeasured on purpose (named ABSENT, never invented): Hessian/sensitivity
(N051), activations (no second 27B decode), experts (not MoE), state (N048),
token frequency / vocabulary (N045).

## Prescription format (S026 §67)

Per organ:

    ORGAN / DIAGNOSIS / PRESCRIPTION / AVOID

AVOID is mandatory. Example: do not rerun identical shared-K2 on MLP —
`SHARED_BASIS_COHERENT` is dead even if local subspace affinity looks
moderate. Moderate affinity is not a reopen.

## What this is not

Not N043 (the paper registry). Not N046 (the literature survey). Not N044
(the rotation experiment). This engine consumes those and the negative
science library and emits a ranked, organ-local hypothesis list.
"""


def diagnose(*, parent: Path | None = None, write: bool = True) -> dict:
    t0 = time.time()
    parent = parent or find_parent_bf16()
    rng = np.random.default_rng(SEED)
    fp_before = parent_a_fingerprint(PARENT_A) if PARENT_A.is_dir() else {"path": str(PARENT_A), "catalog_present": False}
    catalog = inspect_parent_a_catalog(PARENT_A) if fp_before.get("catalog_present") else None

    n043 = load_n043()
    n046 = load_n046()
    n043_map = n043_by_id(n043)

    organs = []
    for spec in organ_plan():
        print(f"== organ {spec['organ_id']} ==", flush=True)
        organ = diagnose_organ(parent, spec, rng)
        lit = literature_for_organ(n046, spec["organ_id"]) if n046.get("loaded") else []
        attach_prescription(organ, n043_map, lit)
        # drop bulky per-tensor sketches already aggregated; keep per_tensor features
        organs.append(organ)

    fp_after = parent_a_fingerprint(PARENT_A) if PARENT_A.is_dir() else fp_before
    mutated = False
    if fp_before.get("catalog_sha256") and fp_after.get("catalog_sha256"):
        mutated = fp_before["catalog_sha256"] != fp_after["catalog_sha256"] or (
            fp_before.get("mix_sha256") != fp_after.get("mix_sha256")
        )

    # §110 report rollup
    ranked_global = []
    for o in organs:
        top = next((r for r in o["ranked_families"] if r["role"] == "probe"), None)
        ranked_global.append({
            "organ": o["organ_id"],
            "top_probe": None if top is None else {
                "family": top["family"], "score": top["score"], "rank": top["rank"]
            },
            "section_67": o["section_67"],
        })

    receipt = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "git_head": git_head(),
        "generated_by": GENERATOR,
        "obligation": OBLIGATION,
        "s026": ["§65", "§66", "§67", "§5", "§11", "§76", "§110"],
        "family": "DOC-DIAGNOSIS",
        "phase": "A",
        "hand_authored": False,
        "predicts_not_certifies": True,
        "diagnosis_predicts_plausible_families": True,
        "diagnosis_does_not_certify": True,
        "qwen_mlp_2_25_stays_closed_for_unrotated_family": True,
        "literature_is": "HYPOTHESIS",
        "literature_is_not_authority": True,
        "did_not_touch_gpu": True,
        "did_not_run_cargo_or_metal_benchmarks": True,
        "did_not_run_second_27b_decode": True,
        "did_not_load_second_27b": True,
        "did_not_mutate_noetic_parent_a": not mutated,
        "did_not_write_under_models": True,
        "pure_cpu_numpy": True,
        "python": sys.executable,
        "numpy": np.__version__,
        "seed": SEED,
        "one_line": (
            "Per-organ diagnosis on real Qwen3.8 parent tensors: rank spectrum, "
            "outliers, cross-layer/head, shared-basis and codebook affinity, sparsity "
            "→ ranked §65 families with a negative-science AVOID list."
        ),
        "question": "which S026 §65 representation family is plausible per organ, and what should we AVOID?",
        "rejected_question": "how many bits should this tensor get",
        "qualified_parent": str(parent),
        "parent_streamed_one_tensor_at_a_time": True,
        "sealed_leader": {
            "path": str(PARENT_A),
            "read_only": True,
            "fingerprint_before": fp_before,
            "fingerprint_after": fp_after,
            "mutated": mutated,
            "catalog": catalog,
            "weights_streamed_from": "qualified BF16 parent (real tensors), not dequantized mix codes",
        },
        "n043": {
            "loaded": n043["loaded"],
            "path": n043["path"],
            "n_techniques": None if not n043["loaded"] else n043["doc"].get("n_techniques"),
            "literature_is": "HYPOTHESIS",
        },
        "n046": {
            "loaded": n046["loaded"],
            "path": n046["path"],
            "n_recommended_additions": None if not n046["loaded"] else n046["doc"].get("n_recommended_additions"),
            "literature_is": "HYPOTHESIS",
        },
        "s026_65_families": [label for _id, label in FAMILIES],
        "organs_required": [
            "mlp_gate_up", "mlp_down",
            "gqa_q", "gqa_k", "gqa_v", "gqa_o",
            "deltanet_in_proj", "deltanet_out_proj",
            "embed", "lm_head",
        ],
        "unmeasured": unmeasured_block(),
        "geometry": {
            "layers": LAYERS,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "vocab": VOCAB,
            "gqa_layers": list(gqa_layers()),
            "dn_layers": list(dn_layers()),
            "parent_params": PARENT_PARAMS,
        },
        "method": {
            "svd": f"randomized range-finder k={SVD_K} niter={SVD_NITER}; residual energy from ||W||_F^2",
            "kmeans": f"k-means++ Lloyd k={KMEANS_K_SMALL}/{KMEANS_K_LARGE} on up to {KMEANS_ROWS} rows; group-{GROUP_VQ} VQ",
            "shared_basis": "Grassmann affinity of truncated right/left singular subspaces across sampled layers",
            "cross_layer": f"mean Rademacher sketch (dim {SKETCH_DIM}) cosine",
            "outliers": f"fraction |w-median| > {OUTLIER_MAD_K} MAD and |w-mean| > {OUTLIER_STD_K} std on up to 1e6-entry sample",
            "embed_lm_head": f"full-pass moments + uniform {EMBED_ROW_SAMPLE}-row sample for SVD/k-means",
            "never_gaussian_proxy_as_the_measurement": True,
            "synthetic_used_only_in_unit_tests": True,
        },
        "organs": organs,
        "section_110": {
            "DIAGNOSIS": "per-organ diagnosis_features + diagnosis[]",
            "PHYSICAL_BOTTLENECK": "ABSENT (N027/N030/N031 own token_ns; this lane is CPU features)",
            "INFORMATION_BOTTLENECK": ranked_global,
            "STATE_BOTTLENECK": "ABSENT (N048)",
            "TOKENIZER_BOTTLENECK": "ABSENT (N045)",
            "BEHAVIOR_TABULA_STATE": "ABSENT (Tabula is a distinct axis, S026 §71)",
            "KNOWN_RELEVANT_TECHNIQUES": "N043 registry + N046 additions, cited per organ",
            "NEGATIVE_SCIENCE": "AVOID lists cite receipts/headless/* and NNS-* ids",
            "RANKED_PRESCRIPTIONS": [o["section_67"] for o in organs],
            "EXPECTED_EBPW_DELTA": "ABSENT (diagnosis does not invent a byte win)",
            "EXPECTED_BYTE_DELTA": "ABSENT",
            "EXPECTED_NS_DELTA": "ABSENT",
            "EXPERIMENT_COST": "next experiments named (N044/N045/N048/N051); this lane is CPU-only",
            "CONFIDENCE": "predicted from weight-space features + prior science; not a capability certificate",
        },
        "docs": "docs/ultragoals/DOCTOR_DIAGNOSIS.md",
        "wall_s": time.time() - t0,
        "written_to": str(RECEIPT),
    }
    receipt = j(receipt)
    if write:
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n")
        DOCS.parent.mkdir(parents=True, exist_ok=True)
        DOCS.write_text(build_docs())
        print(f"wrote {RECEIPT} wall_s={receipt['wall_s']:.1f}", flush=True)
    return receipt


def main() -> int:
    diagnose(write=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""CPU representation sweep: product / residual / lattice / trellis on real Qwen3.8 attention.

Read-only on the bf16 shards and the activation capture. No GPU, no pack, no model write.
"""
from __future__ import annotations

import json
import math
import os
import resource
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from sklearn.cluster import MiniBatchKMeans

    HAVE_SK = True
except Exception:
    HAVE_SK = False

BF16 = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
)
ACT = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
)
OUT = Path("/tmp/g1_vq_results.json")

# Attention quality bar from receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_VERDICT.json
BAR_Q4 = 0.990
BAR_INTEREST = 0.970

SEED = 20260817


def rss_gb() -> float:
    # macOS: ru_maxrss is bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def now() -> float:
    return time.perf_counter()


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    bits = u16.astype(np.uint32) << 16
    return bits.view(np.float32).copy()


def load_index() -> dict:
    return json.loads((BF16 / "model.safetensors.index.json").read_text())


def parse_st_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def load_bf16_matrix(name: str, weight_map: dict) -> np.ndarray:
    shard = BF16 / weight_map[name]
    header, data0 = parse_st_header(shard)
    info = header[name]
    if info["dtype"] != "BF16":
        raise RuntimeError(f"{name} dtype {info['dtype']}")
    shape = tuple(info["shape"])
    begin, end = info["data_offsets"]
    n = int(np.prod(shape))
    with open(shard, "rb") as f:
        f.seek(data0 + begin)
        raw = f.read(end - begin)
    u16 = np.frombuffer(raw, dtype="<u2", count=n)
    return bf16_to_f32(u16).reshape(shape)


def load_hidden(layer: int) -> np.ndarray:
    p = ACT / "hidden" / f"L{layer:02d}.f32"
    arr = np.fromfile(p, dtype=np.float32)
    if arr.size != 256 * 5120:
        raise RuntimeError(f"{p} size {arr.size}")
    return arr.reshape(256, 5120)


def weight_stats(W: np.ndarray) -> dict:
    flat = W.reshape(-1).astype(np.float64, copy=False)
    absv = np.abs(flat)
    mean = float(flat.mean())
    var = float(((flat - mean) ** 2).mean())
    std = math.sqrt(var)
    rms = float(np.sqrt((flat * flat).mean()))
    # excess kurtosis
    if var > 0:
        m4 = float((((flat - mean) ** 4).mean()))
        kurt = m4 / (var * var) - 3.0
    else:
        kurt = 0.0
    return {
        "shape": [int(x) for x in W.shape],
        "elements": int(W.size),
        "dtype": "float32_from_bf16",
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": mean,
        "std": std,
        "rms": rms,
        "mean_abs": float(absv.mean()),
        "median_abs": float(np.median(absv)),
        "p99_abs": float(np.quantile(absv, 0.99)),
        "max_abs": float(absv.max()),
        "excess_kurtosis": kurt,
        "frac_zero": float((flat == 0).mean()),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64, copy=False)
    b = b.reshape(-1).astype(np.float64, copy=False)
    na = float(np.dot(a, a))
    nb = float(np.dot(b, b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / math.sqrt(na * nb))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64, copy=False)
    d = a - b.reshape(-1).astype(np.float64, copy=False)
    na = float(np.dot(a, a))
    if na <= 0:
        return 0.0
    return float(math.sqrt(float(np.dot(d, d)) / na))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False)
    return float(np.mean(d * d))


def snr_db(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64, copy=False)
    d = a - b.reshape(-1).astype(np.float64, copy=False)
    ps = float(np.dot(a, a))
    ns = float(np.dot(d, d))
    if ns <= 0:
        return float("inf")
    if ps <= 0:
        return 0.0
    return float(10.0 * math.log10(ps / ns))


def row_cosines(Y: np.ndarray, Yh: np.ndarray) -> tuple[float, float]:
    y = Y.astype(np.float64, copy=False)
    yh = Yh.astype(np.float64, copy=False)
    num = np.sum(y * yh, axis=1)
    den = np.sqrt(np.sum(y * y, axis=1) * np.sum(yh * yh, axis=1))
    c = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return float(c.mean()), float(c.min())


def score(W: np.ndarray, What: np.ndarray, X: np.ndarray | None) -> dict:
    out = {
        "weight_cosine": cosine(W, What),
        "weight_rel_l2": rel_l2(W, What),
        "weight_mse": mse(W, What),
        "weight_snr_db": snr_db(W, What),
        "weight_max_abs_err": float(np.max(np.abs(W - What))),
    }
    if X is not None:
        # X: (T, cols)
        Y = W @ X.T
        Yh = What @ X.T
        mean_r, min_r = row_cosines(Y, Yh)
        out.update(
            {
                "output_cosine": cosine(Y, Yh),
                "output_cosine_mean_row": mean_r,
                "output_cosine_min_row": min_r,
                "output_rel_l2": rel_l2(Y, Yh),
                "n_x_rows": int(X.shape[0]),
                "clears_0p990": bool(mean_r >= BAR_Q4 and cosine(Y, Yh) >= BAR_Q4),
                "clears_0p970": bool(cosine(Y, Yh) >= BAR_INTEREST),
            }
        )
        del Y, Yh
    else:
        out.update(
            {
                "output_cosine": None,
                "note": "no mixer-site X in capture; weight-space only",
            }
        )
    return out


def assign_batch(X: np.ndarray, C: np.ndarray, batch: int = 65536) -> np.ndarray:
    n = X.shape[0]
    labels = np.empty(n, dtype=np.int32)
    c2 = np.sum(C * C, axis=1)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        dots = X[s:e] @ C.T
        x2 = np.sum(X[s:e] * X[s:e], axis=1, keepdims=True)
        dist = x2 + c2[None, :] - 2.0 * dots
        labels[s:e] = np.argmin(dist, axis=1)
    return labels


def kmeans_shared(X: np.ndarray, K: int, *, seed: int, train_n: int = 262144, iters: int = 20) -> np.ndarray:
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if n > train_n:
        idx = rng.choice(n, size=train_n, replace=False)
        train = np.ascontiguousarray(X[idx])
    else:
        train = np.ascontiguousarray(X)
    if HAVE_SK and K < train.shape[0]:
        km = MiniBatchKMeans(
            n_clusters=K,
            batch_size=min(8192, train.shape[0]),
            max_iter=iters,
            n_init=1,
            random_state=seed,
            reassignment_ratio=0.01,
        )
        km.fit(train)
        return km.cluster_centers_.astype(np.float32)
    # numpy fallback: kmeans++ on up to 32k then lloyd
    m = min(len(train), 32768)
    sample = train if len(train) <= m else train[rng.choice(len(train), m, replace=False)]
    C = np.empty((K, sample.shape[1]), dtype=np.float32)
    C[0] = sample[int(rng.integers(len(sample)))]
    closest = np.full(len(sample), np.inf, dtype=np.float64)
    for k in range(1, K):
        diff = sample - C[k - 1]
        dist = np.sum(diff * diff, axis=1)
        closest = np.minimum(closest, dist)
        tot = float(closest.sum())
        if tot <= 0:
            C[k:] = sample[rng.choice(len(sample), K - k, replace=True)]
            break
        C[k] = sample[rng.choice(len(sample), p=closest / tot)]
    for _ in range(iters):
        labels = assign_batch(sample, C, batch=32768)
        for k in range(K):
            msk = labels == k
            if msk.any():
                C[k] = sample[msk].mean(axis=0)
    return C


def pca_rotate(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (X @ R, R) with R orthonormal, principal axes. Fold later via C @ R.T."""
    # X: (n, d)
    xc = X - X.mean(axis=0, keepdims=True)
    # thin SVD on covariance via X^T X
    d = X.shape[1]
    if X.shape[0] > 65536:
        rng = np.random.default_rng(SEED)
        xs = xc[rng.choice(X.shape[0], 65536, replace=False)]
    else:
        xs = xc
    cov = (xs.T @ xs) / max(len(xs) - 1, 1)
    w, V = np.linalg.eigh(cov)
    R = np.fliplr(V).astype(np.float32)  # descending
    return (X @ R).astype(np.float32), R


def reshape_sub(W: np.ndarray, d: int) -> np.ndarray:
    rows, cols = W.shape
    if cols % d:
        raise ValueError(f"cols {cols} not divisible by d={d}")
    S = cols // d
    return W.reshape(rows, S, d)


def pq_shared(W: np.ndarray, d: int, K: int, *, opq: bool = False) -> tuple[np.ndarray, dict]:
    rows, cols = W.shape
    S = cols // d
    V = reshape_sub(W, d).reshape(-1, d)
    R = None
    train_src = V
    if opq:
        train_src, R = pca_rotate(V)
    C = kmeans_shared(train_src, K, seed=SEED + d + K)
    labels = assign_batch(train_src, C)
    recon_v = C[labels]
    if opq:
        recon_v = recon_v @ R.T
        C_store = C @ R.T  # folded: decode is still one gather
    else:
        C_store = C
    What = recon_v.reshape(rows, cols)
    n_cb = 1
    bits_idx = math.ceil(math.log2(K))
    index_bits = rows * S * bits_idx
    codebook_bits = n_cb * K * d * 16
    payload_bytes = (index_bits + 7) // 8 + (codebook_bits + 7) // 8
    meta = {
        "family": "PQ_shared" + ("_OPQfold" if opq else ""),
        "d_sub": d,
        "K": K,
        "subspaces": S,
        "n_codebooks": n_cb,
        "index_bits_per_subvector": bits_idx,
        "index_bits": index_bits,
        "codebook_bits_fp16": codebook_bits,
        "payload_bytes": payload_bytes,
        "bpw": (index_bits + codebook_bits) / W.size,
        "table_bytes": (codebook_bits + 7) // 8,
        "register_only": False,
        "decode": {
            "per_subvector": {
                "sequential_index_extract": 1,
                "random_codebook_gathers": 1,
                "gathered_values": d,
                "fma": d,
            },
            "per_weight": {
                "random_gathers": 1.0 / d,
                "fma": 1.0,
            },
            "table": f"one fp16 codebook {K}x{d} = {K * d * 2} bytes, reused across all rows and subspaces",
        },
    }
    return What.astype(np.float32), meta


def pq_per_subspace(W: np.ndarray, d: int, K: int) -> tuple[np.ndarray, dict]:
    rows, cols = W.shape
    S = cols // d
    Vs = reshape_sub(W, d)  # (rows, S, d)
    What = np.empty_like(W)
    C_all = np.empty((S, K, d), dtype=np.float32)
    # train each subspace independently (n = rows)
    for s in range(S):
        X = np.ascontiguousarray(Vs[:, s, :])
        C = kmeans_shared(X, K, seed=SEED + 1000 + s, train_n=min(rows, 65536), iters=12)
        C_all[s] = C
        labels = assign_batch(X, C, batch=max(rows, 1))
        What[:, s * d : (s + 1) * d] = C[labels]
    bits_idx = math.ceil(math.log2(K))
    index_bits = rows * S * bits_idx
    codebook_bits = S * K * d * 16
    meta = {
        "family": "PQ_per_subspace",
        "d_sub": d,
        "K": K,
        "subspaces": S,
        "n_codebooks": S,
        "index_bits_per_subvector": bits_idx,
        "index_bits": index_bits,
        "codebook_bits_fp16": codebook_bits,
        "payload_bytes": (index_bits + 7) // 8 + (codebook_bits + 7) // 8,
        "bpw": (index_bits + codebook_bits) / W.size,
        "table_bytes": (codebook_bits + 7) // 8,
        "register_only": False,
        "decode": {
            "per_subvector": {
                "sequential_index_extract": 1,
                "random_codebook_gathers": 1,
                "gathered_values": d,
                "fma": d,
            },
            "per_weight": {"random_gathers": 1.0 / d, "fma": 1.0},
            "table": f"{S} fp16 codebooks {K}x{d} = {S * K * d * 2} bytes (working set)",
        },
    }
    return What, meta


def gravity_vq(W: np.ndarray, D: int, K: int) -> tuple[np.ndarray, dict]:
    """S=1 gravity-pq geometry: one codebook, D-wide chunks."""
    rows, cols = W.shape
    nchunk = cols // D
    V = W.reshape(-1, D)
    C = kmeans_shared(V, K, seed=SEED + 7000 + D + K)
    labels = assign_batch(V, C)
    What = C[labels].reshape(rows, cols)
    bits_idx = math.ceil(math.log2(K))
    index_bits = rows * nchunk * bits_idx
    codebook_bits = K * D * 16
    meta = {
        "family": "gravity_VQ_S1",
        "D": D,
        "K": K,
        "nchunk": nchunk,
        "subspaces": 1,
        "index_bits_per_chunk": bits_idx,
        "index_bits": index_bits,
        "codebook_bits_fp16": codebook_bits,
        "payload_bytes": (index_bits + 7) // 8 + (codebook_bits + 7) // 8,
        "bpw": (index_bits + codebook_bits) / W.size,
        "table_bytes": (codebook_bits + 7) // 8,
        "register_only": False,
        "matches_shipping_kernel": "crates/hawking-core/shaders/gravity_pq.metal::gravity_pq_matvec (S=1)",
        "decode": {
            "per_chunk": {
                "sequential_index_extract": 1,
                "random_codebook_gathers": 1,
                "gathered_values": D,
                "fma": D,
            },
            "per_weight": {"random_gathers": 1.0 / D, "fma": 1.0},
            "table": f"one fp16 codebook {K}x{D} = {K * D * 2} bytes, reused across every row",
        },
    }
    return What, meta


def residual_vq(W: np.ndarray, d: int, K: int, stages: int) -> tuple[np.ndarray, dict]:
    rows, cols = W.shape
    S = cols // d
    V = reshape_sub(W, d).reshape(-1, d).copy()
    residual = V.copy()
    acc = np.zeros_like(V)
    codebooks = []
    for st in range(stages):
        C = kmeans_shared(residual, K, seed=SEED + 3000 + st + d * 17 + K)
        labels = assign_batch(residual, C)
        recon = C[labels]
        acc += recon
        residual -= recon
        codebooks.append(C)
    What = acc.reshape(rows, cols)
    bits_idx = math.ceil(math.log2(K))
    index_bits = rows * S * stages * bits_idx
    codebook_bits = stages * K * d * 16
    meta = {
        "family": "RVQ_shared",
        "d_sub": d,
        "K": K,
        "stages": stages,
        "subspaces": S,
        "index_bits": index_bits,
        "codebook_bits_fp16": codebook_bits,
        "payload_bytes": (index_bits + 7) // 8 + (codebook_bits + 7) // 8,
        "bpw": (index_bits + codebook_bits) / W.size,
        "table_bytes": (codebook_bits + 7) // 8,
        "register_only": False,
        "matches_shipping_kernel": "crates/hawking-core/shaders/gravity_pq.metal::gravity_residual_pq_matvec",
        "decode": {
            "per_subvector": {
                "sequential_index_extract": stages,
                "random_codebook_gathers": stages,
                "gathered_values": stages * d,
                "fma": stages * d,
                "stage_dependencies": stages,
            },
            "per_weight": {"random_gathers": stages / d, "fma": float(stages)},
            "table": f"{stages} fp16 codebooks {K}x{d} = {stages * K * d * 2} bytes",
        },
    }
    return What, meta


def uniform_qn(W: np.ndarray, bits: int, group: int = 64) -> tuple[np.ndarray, dict]:
    """Exact HGRAVU01 rule: scale = maxabs/bound, bound=(1<<(bits-1))-1, q in [-bound,bound]."""
    flat = W.reshape(-1)
    groups = math.ceil(flat.size / group)
    pad = groups * group - flat.size
    padded = np.pad(flat, (0, pad), constant_values=0.0).reshape(groups, group)
    bound = (1 << (bits - 1)) - 1
    scales = (np.max(np.abs(padded), axis=1) / max(bound, 1)).astype(np.float16).astype(np.float32)
    den = np.where(scales > 0, scales, 1.0)
    q = np.rint(padded / den[:, None]).clip(-bound, bound)
    recon = (q * scales[:, None]).reshape(-1)[: flat.size].reshape(W.shape)
    index_bits = W.size * bits
    scale_bits = groups * 16
    meta = {
        "family": f"HGRAVU01_q{bits}_g{group}",
        "bits": bits,
        "group": group,
        "index_bits": index_bits,
        "scale_bits": scale_bits,
        "payload_bytes": (index_bits + 7) // 8 + (scale_bits + 7) // 8,
        "bpw": (index_bits + scale_bits) / W.size,
        "table_bytes": 0,
        "register_only": True,
        "matches_shipping_kernel": "crates/hawking-core/shaders/qwen_uniform_q4.metal::qwen_uniform_q4_group64_matvec",
        "decode": {
            "per_weight": {
                "sequential_nibble_or_bitfield": 1,
                "random_gathers": 0,
                "scale_broadcast": f"1 fp16 / {group} weights",
                "fma": 1,
            },
            "table": "none",
        },
    }
    return recon.astype(np.float32), meta


def binary_g(W: np.ndarray, group: int = 128) -> tuple[np.ndarray, dict]:
    flat = W.reshape(-1)
    groups = math.ceil(flat.size / group)
    pad = groups * group - flat.size
    padded = np.pad(flat, (0, pad), constant_values=0.0).reshape(groups, group)
    scales = np.mean(np.abs(padded), axis=1).astype(np.float16).astype(np.float32)
    recon = (np.where(padded >= 0, 1.0, -1.0) * scales[:, None]).reshape(-1)[: flat.size].reshape(W.shape)
    index_bits = W.size  # 1 sign bit
    scale_bits = groups * 16
    meta = {
        "family": f"HGRAVB01_binary_g{group}",
        "group": group,
        "index_bits": index_bits,
        "scale_bits": scale_bits,
        "payload_bytes": (index_bits + 7) // 8 + (scale_bits + 7) // 8,
        "bpw": (index_bits + scale_bits) / W.size,
        "table_bytes": 0,
        "register_only": True,
        "decode": {
            "per_weight": {"sign_bit": 1, "random_gathers": 0, "fma": 1},
            "table": "none",
        },
    }
    return recon.astype(np.float32), meta


def nearest_dn(Y: np.ndarray) -> np.ndarray:
    R = np.rint(Y)
    odd = (np.rint(R.sum(axis=1)).astype(np.int64) & 1) != 0
    if not np.any(odd):
        return R
    err = np.abs(Y - R)
    flip = err.argmax(axis=1)
    idx = np.arange(Y.shape[0])
    adj = np.where(Y[idx, flip] >= R[idx, flip], 1.0, -1.0)
    R = R.copy()
    R[odd, flip[odd]] += adj[odd]
    return R


def nearest_e8(Y: np.ndarray) -> np.ndarray:
    f0 = nearest_dn(Y)
    f1 = nearest_dn(Y - 0.5) + 0.5
    d0 = np.sum((Y - f0) ** 2, axis=1)
    d1 = np.sum((Y - f1) ** 2, axis=1)
    return np.where((d0 <= d1)[:, None], f0, f1)


def lattice_group(W: np.ndarray, kind: str, b_bits: int, group: int = 64) -> tuple[np.ndarray, dict]:
    """Group-scaled D4 / E8 / Z^d. Register-only. Scale amortized like Q4 (g=64)."""
    d = {"D4": 4, "E8": 8, "Z4": 4, "Z8": 8}[kind]
    flat = W.reshape(-1)
    # process as (nvec, d)
    if flat.size % d:
        raise ValueError("size not divisible by lattice dim")
    V = flat.reshape(-1, d).copy()
    # group G weights => G/d lattice vectors share one fp16 scale
    if group % d:
        raise ValueError("group must be multiple of lattice dim")
    vecs_per_group = group // d
    nvec = V.shape[0]
    ng = math.ceil(nvec / vecs_per_group)
    pad = ng * vecs_per_group - nvec
    if pad:
        V = np.pad(V, ((0, pad), (0, 0)), constant_values=0.0)
    G = V.reshape(ng, vecs_per_group, d)
    # scale so that after /scale, typical coords fit in signed b-bit range
    radius = (1 << (b_bits - 1)) - 1
    gflat = G.reshape(ng, -1)
    scales = (np.max(np.abs(gflat), axis=1) / max(radius, 1)).astype(np.float16).astype(np.float32)
    den = np.where(scales > 0, scales, 1.0)
    Y = G / den[:, None, None]
    Y2 = Y.reshape(-1, d)
    if kind == "D4":
        Q = nearest_dn(Y2)
    elif kind == "E8":
        Q = nearest_e8(Y2)
    else:
        Q = np.rint(Y2)
    lo = -radius
    hi = radius
    # E8 half-integers: store 2*coord in the integer range, so clip at radius after *1
    Q = np.clip(Q, lo, hi)
    recon = (Q.reshape(ng, vecs_per_group, d) * scales[:, None, None]).reshape(-1, d)[:nvec]
    What = recon.reshape(W.shape)
    # bits: D4/E8 can drop 1 parity bit per vector; Z^d cannot.
    parity_save = 1 if kind in {"D4", "E8"} else 0
    # E8 also needs 1 coset bit (integer vs half-int) if we encode the lattice index tightly.
    coset_bits = 1 if kind == "E8" else 0
    bits_per_vec = d * b_bits - parity_save + coset_bits
    nvec_real = W.size // d
    index_bits = nvec_real * bits_per_vec
    scale_bits = math.ceil(W.size / group) * 16
    meta = {
        "family": f"lattice_{kind}_b{b_bits}_g{group}",
        "lattice": kind,
        "coord_bits": b_bits,
        "dim": d,
        "group": group,
        "parity_bit_saved": parity_save,
        "coset_bits": coset_bits,
        "bits_per_vector": bits_per_vec,
        "index_bits": index_bits,
        "scale_bits": scale_bits,
        "payload_bytes": (index_bits + 7) // 8 + (scale_bits + 7) // 8,
        "bpw": (index_bits + scale_bits) / W.size,
        "table_bytes": 0,
        "register_only": True,
        "decode": {
            "per_vector": {
                "sequential_bitfield_extract": 1,
                "random_gathers": 0,
                "integer_to_float": d,
                "parity_or_coset_restore": 1 if (parity_save or coset_bits) else 0,
                "scale_mul": d,
                "fma_if_gemv": d,
            },
            "table": "none — lattice point is a function of the index bits",
        },
    }
    return What.astype(np.float32), meta


def hadamard_lattice(W: np.ndarray, bits: int, group: int = 128) -> tuple[np.ndarray, dict]:
    """Repo HGRAVH01: Walsh-Hadamard then uniform integer lattice. Register-only transform."""
    if group & (group - 1):
        raise ValueError("hadamard group must be power of two")
    flat = W.reshape(-1)
    groups = math.ceil(flat.size / group)
    pad = groups * group - flat.size
    padded = np.pad(flat, (0, pad), constant_values=0.0).reshape(groups, group)
    work = padded.astype(np.float32, copy=True)
    width = group
    stride = 1
    while stride < width:
        view = work.reshape(groups, width // (2 * stride), 2, stride)
        left = view[:, :, 0, :].copy()
        right = view[:, :, 1, :].copy()
        view[:, :, 0, :] = left + right
        view[:, :, 1, :] = left - right
        stride *= 2
    transformed = work / math.sqrt(width)
    bound = (1 << (bits - 1)) - 1
    scales = (np.max(np.abs(transformed), axis=1) / max(bound, 1)).astype(np.float16).astype(np.float32)
    den = np.where(scales > 0, scales, 1.0)
    q = np.rint(transformed / den[:, None]).clip(-bound, bound)
    coeff = q * scales[:, None]
    # inverse = same transform
    work = coeff.astype(np.float32, copy=True)
    stride = 1
    while stride < width:
        view = work.reshape(groups, width // (2 * stride), 2, stride)
        left = view[:, :, 0, :].copy()
        right = view[:, :, 1, :].copy()
        view[:, :, 0, :] = left + right
        view[:, :, 1, :] = left - right
        stride *= 2
    restored = (work / math.sqrt(width)).reshape(-1)[: flat.size].reshape(W.shape)
    index_bits = W.size * bits
    scale_bits = groups * 16
    meta = {
        "family": f"HGRAVH01_hadamard_q{bits}_g{group}",
        "bits": bits,
        "group": group,
        "index_bits": index_bits,
        "scale_bits": scale_bits,
        "payload_bytes": (index_bits + 7) // 8 + (scale_bits + 7) // 8,
        "bpw": (index_bits + scale_bits) / W.size,
        "table_bytes": 0,
        "register_only": True,
        "decode": {
            "per_group": {
                "sequential_bitfield": group,
                "walsh_hadamard_butterflies": int(group * math.log2(group)),
                "random_gathers": 0,
            },
            "table": "none",
        },
    }
    return restored.astype(np.float32), meta


def inv_norm_cdf(p: np.ndarray) -> np.ndarray:
    """Acklam rational, vectorized. Same coefficients as strand-quant codebook.rs."""
    A = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.38357751867269e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    B = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    C = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    D = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )
    p = np.clip(p, 1e-12, 1 - 1e-12)
    plow = 0.02425
    out = np.empty_like(p, dtype=np.float64)
    left = p < plow
    right = p > 1 - plow
    mid = ~left & ~right
    if np.any(left):
        q = np.sqrt(-2.0 * np.log(p[left]))
        out[left] = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / (
            ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q) + 1.0
        )
    if np.any(right):
        q = np.sqrt(-2.0 * np.log(1.0 - p[right]))
        out[right] = -(
            (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / (((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q) + 1.0)
        )
    if np.any(mid):
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (
            (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        ) / ((((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r) + 1.0)
    return out


def hash_state(s: np.ndarray, l_bits: int) -> np.ndarray:
    mask = (1 << l_bits) - 1
    r = max(l_bits // 2, 1)
    h = s & mask
    h = (h ^ (h >> r)) & mask
    h = (h * np.uint64(0x2545F4914F6CDD1D)) & mask
    h = (h ^ (h >> r)) & mask
    h = (h * np.uint64(0x9E3779B97F4A7C15)) & mask
    return h & mask


def gaussian_codebook(l_bits: int) -> np.ndarray:
    n = 1 << l_bits
    s = np.arange(n, dtype=np.uint64)
    rnk = hash_state(s, l_bits)
    p = (rnk.astype(np.float64) + 0.5) / n
    return inv_norm_cdf(p).astype(np.float32)


def vector_trellis_computed(W: np.ndarray, d: int, k_bits: int, group: int = 64) -> tuple[np.ndarray, dict]:
    """k bits per d-vector. Reconstruction = group_scale * computed Gaussian axis-aligned code.

    Independent steps (not Viterbi). Register-only: index -> hash -> Acklam.
    """
    rows, cols = W.shape
    S = cols // d
    V = reshape_sub(W, d)  # (rows, S, d)
    n = 1 << k_bits
    # codebook: d independent quantiles from different hashes of the same index
    cb = np.empty((n, d), dtype=np.float32)
    base = gaussian_codebook(k_bits)
    for j in range(d):
        # rotate the permutation by j so axes are not identical
        cb[:, j] = base[np.roll(np.arange(n), j)]
    # normalize codewords to unit rms so scale is meaningful
    cb /= max(float(np.sqrt(np.mean(cb * cb))), 1e-12)
    # group scale over `group` weights
    flat = W.reshape(-1)
    ng = math.ceil(flat.size / group)
    pad = ng * group - flat.size
    padded = np.pad(flat, (0, pad)).reshape(ng, group)
    scales = (np.sqrt(np.mean(padded * padded, axis=1))).astype(np.float16).astype(np.float32)
    # assign each subvector by nearest computed code * local scale
    # local scale: use the group that contains the first element of the subvector
    What = np.empty_like(W)
    # For speed: ignore per-group variation inside assign — use per-subvector rms
    Vrms = np.sqrt(np.mean(V * V, axis=2, keepdims=True))
    Vn = np.divide(V, Vrms, out=np.zeros_like(V), where=Vrms > 0)
    flatn = Vn.reshape(-1, d)
    labels = assign_batch(flatn, cb)
    recon = (cb[labels].reshape(rows, S, d) * Vrms)
    What = recon.reshape(rows, cols)
    # honest payload: we would store k bits per subvector + one scale per group
    # (assignment above used per-subvector rms which is NOT stored — that's encode-oracle.
    # Re-decode with stored group scales only.)
    recon2 = cb[labels].reshape(rows, S, d)
    # apply group scales in weight order
    recon2_flat = recon2.reshape(-1)
    # map each weight to its group scale
    gidx = np.arange(W.size) // group
    What = (recon2_flat * scales[gidx]).reshape(W.shape)
    index_bits = rows * S * k_bits
    scale_bits = ng * 16
    meta = {
        "family": f"trellis_computed_d{d}_k{k_bits}_g{group}",
        "d_sub": d,
        "k_bits": k_bits,
        "n_codewords": n,
        "index_bits": index_bits,
        "scale_bits": scale_bits,
        "payload_bytes": (index_bits + 7) // 8 + (scale_bits + 7) // 8,
        "bpw": (index_bits + scale_bits) / W.size,
        "table_bytes": 0,
        "register_only": True,
        "decode": {
            "per_subvector": {
                "sequential_index": 1,
                "hash_ops": 4,
                "acklam_rationals": d,
                "random_gathers": 0,
                "fma": d,
            },
            "table": "none — codebook is a pure function of (index, axis). ALU-heavy.",
            "note": "strand-quant CodebookMode::ComputedAcklam is the scalar analogue (codebook.rs)",
        },
    }
    return What.astype(np.float32), meta


def viterbi_tcq_sample(w: np.ndarray, k_bits: int, l_bits: int, block: int = 256) -> dict:
    """True 1D TCQ Viterbi on a sample. Scalar: cannot go below 1 bpw."""
    qcb = gaussian_codebook(l_bits)
    # unit-normalize
    qcb = qcb / max(float(np.std(qcb)), 1e-12)
    nstates = 1 << l_bits
    ninp = 1 << k_bits
    mask = nstates - 1
    n = (len(w) // block) * block
    w = w[:n].astype(np.float32)
    recon = np.empty(n, dtype=np.float32)
    sse = 0.0
    pwr = 0.0
    for b0 in range(0, n, block):
        blk = w[b0 : b0 + block]
        scale = float(np.sqrt(np.mean(blk * blk)) or 1.0)
        target = blk / scale
        # dp
        inf = 1e30
        cost = np.full(nstates, inf, dtype=np.float64)
        cost[0] = 0.0
        prev = np.zeros((block, nstates), dtype=np.int16)
        prev_sym = np.zeros((block, nstates), dtype=np.int8)
        for t in range(block):
            new = np.full(nstates, inf, dtype=np.float64)
            xt = float(target[t])
            for s in range(nstates):
                c0 = cost[s]
                if c0 >= inf:
                    continue
                base = (s << k_bits) & mask
                for sym in range(ninp):
                    ns = base | sym
                    e = xt - float(qcb[ns])
                    nc = c0 + e * e
                    if nc < new[ns]:
                        new[ns] = nc
                        prev[t, ns] = s
                        prev_sym[t, ns] = sym
            cost = new
        # backtrack
        s = int(np.argmin(cost))
        path = np.empty(block, dtype=np.int32)
        for t in range(block - 1, -1, -1):
            path[t] = s
            s = int(prev[t, s])
        rec = qcb[path] * scale
        recon[b0 : b0 + block] = rec
        d = blk.astype(np.float64) - rec.astype(np.float64)
        sse += float(np.dot(d, d))
        pwr += float(np.dot(blk.astype(np.float64), blk.astype(np.float64)))
    # bpw: k bits/weight + 16-bit scale / block + L bits init state / block
    bpw = k_bits + (16 + l_bits) / block
    return {
        "family": f"TCQ_viterbi_k{k_bits}_L{l_bits}_blk{block}",
        "n_weights": n,
        "weight_rel_l2": math.sqrt(sse / pwr) if pwr > 0 else 0.0,
        "weight_snr_db": 10 * math.log10(pwr / sse) if sse > 0 else float("inf"),
        "weight_cosine": cosine(w, recon),
        "bpw": bpw,
        "register_only": True,
        "table_bytes": 0,
        "decode": {
            "per_weight": {
                "state_shift_or": 1,
                "acklam_or_hashed_quantile": 1,
                "random_gathers": 0,
                "fma": 1,
            },
            "note": "scalar TCQ cannot go below 1 bit/weight; vector form needed for sub-bit",
        },
    }


def cross_layer_pq(train_W: np.ndarray, test_W: np.ndarray, d: int, K: int) -> tuple[np.ndarray, dict]:
    Vtr = reshape_sub(train_W, d).reshape(-1, d)
    C = kmeans_shared(Vtr, K, seed=SEED + 9000)
    Vte = reshape_sub(test_W, d).reshape(-1, d)
    labels = assign_batch(Vte, C)
    What = C[labels].reshape(test_W.shape)
    bits_idx = math.ceil(math.log2(K))
    S = test_W.shape[1] // d
    index_bits = test_W.shape[0] * S * bits_idx
    # codebook amortized over TWO tensors of this shape, as a lower bound on sharing
    codebook_bits = K * d * 16
    codebook_bits_shared48 = codebook_bits / 48.0  # if one codebook served all 48 DeltaNet twins
    meta = {
        "family": f"PQ_shared_crosslayer_d{d}_K{K}",
        "d_sub": d,
        "K": K,
        "bpw_codebook_on_test_tensor_only": (index_bits + codebook_bits) / test_W.size,
        "bpw_if_codebook_shared_across_48": (index_bits + codebook_bits_shared48) / test_W.size,
        "index_bits": index_bits,
        "codebook_bits_fp16": codebook_bits,
        "register_only": False,
    }
    return What, meta


def print_row(tag: str, rec: dict) -> None:
    oc = rec.get("output_cosine")
    oc_s = f"{oc:8.5f}" if isinstance(oc, float) else "    n/a "
    clr = rec.get("clears_0p990")
    clr_s = "PASS" if clr else ("fail" if clr is False else "  - ")
    print(
        f"{tag:<42} bpw={rec['bpw']:7.4f} w_cos={rec['weight_cosine']:.5f} "
        f"w_rel={rec['weight_rel_l2']:.4f} out_cos={oc_s} {clr_s} "
        f"reg={int(rec['register_only'])} tab={rec.get('table_bytes', 0):8d} "
        f"rss={rss_gb():.3f}GB",
        flush=True,
    )


TENSORS = [
    # name, layer for X, functional?, grid
    (
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        0,
        True,
        "full",
    ),
    (
        "language_model.model.layers.0.linear_attn.in_proj_z.weight",
        0,
        True,
        "flag",
    ),
    (
        "language_model.model.layers.0.linear_attn.out_proj.weight",
        0,
        False,
        "full",
    ),
    (
        "language_model.model.layers.3.self_attn.q_proj.weight",
        3,
        True,
        "full",
    ),
    (
        "language_model.model.layers.3.self_attn.k_proj.weight",
        3,
        True,
        "flag",
    ),
    (
        "language_model.model.layers.3.self_attn.v_proj.weight",
        3,
        True,
        "flag",
    ),
    (
        "language_model.model.layers.3.self_attn.o_proj.weight",
        3,
        False,
        "flag",
    ),
    (
        "language_model.model.layers.32.linear_attn.in_proj_qkv.weight",
        32,
        True,
        "flag",
    ),
    (
        "language_model.model.layers.63.self_attn.q_proj.weight",
        63,
        True,
        "flag",
    ),
]


def configs_for(grid: str, cols: int) -> list:
    # only d that divide cols
    def ok(d):
        return cols % d == 0

    full = []
    # scalar baselines
    full += [("uniform", 4, 64), ("uniform", 3, 64), ("uniform", 2, 64), ("binary", 128, None)]
    # PQ shared
    for d, K in [(2, 256), (4, 16), (4, 256), (4, 1024), (8, 16), (8, 256), (8, 1024), (16, 256), (16, 1024)]:
        if ok(d):
            full.append(("pq_shared", d, K))
    # OPQ on the interesting mid-rate points
    for d, K in [(4, 256), (8, 256)]:
        if ok(d):
            full.append(("pq_opq", d, K))
    # per-subspace (S independent k-means). Only d=8 K=256 — S=640 is already expensive.
    if ok(8):
        full.append(("pq_per", 8, 256))
    # gravity S=1 VQ
    for D, K in [(4, 256), (8, 256), (16, 256), (32, 256)]:
        if ok(D):
            full.append(("gvq", D, K))
    # RVQ
    for d, K, st in [(8, 256, 2), (8, 16, 4), (16, 256, 2), (16, 256, 3), (4, 256, 2)]:
        if ok(d):
            full.append(("rvq", d, K, st))
    # lattices
    for kind, b in [("D4", 2), ("D4", 3), ("E8", 1), ("E8", 2), ("Z4", 2), ("Z4", 3), ("Z8", 2)]:
        d = 4 if "4" in kind else 8
        if ok(d):
            full.append(("lattice", kind, b))
    # hadamard lattice (repo HGRAVH01)
    full += [("hadamard", 4, 128), ("hadamard", 3, 128), ("hadamard", 2, 128)]
    # computed trellis
    for d, k in [(8, 8), (8, 4), (4, 8), (16, 8)]:
        if ok(d):
            full.append(("trellis", d, k))

    if grid == "full":
        return full
    # flagship subset
    flag = [
        ("uniform", 4, 64),
        ("uniform", 3, 64),
        ("binary", 128, None),
        ("pq_shared", 4, 256),
        ("pq_shared", 8, 256),
        ("pq_opq", 8, 256),
        ("gvq", 8, 256),
        ("rvq", 8, 256, 2),
        ("lattice", "D4", 2),
        ("lattice", "E8", 2),
        ("hadamard", 4, 128),
        ("trellis", 8, 8),
    ]
    out = []
    for c in flag:
        if c[0] in {"pq_shared", "pq_opq", "gvq", "rvq", "trellis"}:
            if ok(c[1]):
                out.append(c)
        elif c[0] == "lattice":
            d = 4 if "4" in c[1] else 8
            if ok(d):
                out.append(c)
        else:
            out.append(c)
    return out


def run_one(kind, W, spec) -> tuple[np.ndarray, dict]:
    if kind == "uniform":
        return uniform_qn(W, spec[0], spec[1])
    if kind == "binary":
        return binary_g(W, spec[0])
    if kind == "pq_shared":
        return pq_shared(W, spec[0], spec[1], opq=False)
    if kind == "pq_opq":
        return pq_shared(W, spec[0], spec[1], opq=True)
    if kind == "pq_per":
        return pq_per_subspace(W, spec[0], spec[1])
    if kind == "gvq":
        return gravity_vq(W, spec[0], spec[1])
    if kind == "rvq":
        return residual_vq(W, spec[0], spec[1], spec[2])
    if kind == "lattice":
        return lattice_group(W, spec[0], spec[1], group=64)
    if kind == "hadamard":
        return hadamard_lattice(W, spec[0], spec[1])
    if kind == "trellis":
        return vector_trellis_computed(W, spec[0], spec[1], group=64)
    raise ValueError(kind)


def main() -> int:
    t0 = now()
    print(f"python {sys.version.split()[0]} numpy {np.__version__} sklearn={HAVE_SK}", flush=True)
    print(f"bf16={BF16} exists={BF16.is_dir()}", flush=True)
    print(f"act={ACT} exists={ACT.is_dir()}", flush=True)
    idx = load_index()
    wm = idx["weight_map"]
    results = {
        "schema": "hawking.superwave.g1.vector_quantization.v1",
        "date": "2026-08-17",
        "host_note": "CPU only. No GPU, no pack, no inference.",
        "source_weights": str(BF16),
        "source_activations": str(ACT),
        "quality_bar_output_cosine": BAR_Q4,
        "bar_source": "receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_VERDICT.json",
        "sklearn": HAVE_SK,
        "tensors": [],
        "viterbi_sample": None,
        "cross_layer": None,
    }

    hidden_cache: dict[int, np.ndarray] = {}
    W_l0_qkv = None
    W_l32_qkv = None

    for name, layer, functional, grid in TENSORS:
        print(f"\n===== {name} grid={grid} =====", flush=True)
        t1 = now()
        W = load_bf16_matrix(name, wm)
        print(f"  loaded {W.shape} in {now()-t1:.2f}s rss={rss_gb():.3f}GB", flush=True)
        stats = weight_stats(W)
        X = None
        if functional:
            if layer not in hidden_cache:
                hidden_cache[layer] = load_hidden(layer)
            X = hidden_cache[layer]
            if X.shape[1] != W.shape[1]:
                print(f"  SKIP functional: X cols {X.shape[1]} != W cols {W.shape[1]}", flush=True)
                X = None
        rec = {
            "tensor": name,
            "layer": layer,
            "grid": grid,
            "stats": stats,
            "functional_x": "hidden_post_norm" if X is not None else None,
            "variants": [],
        }
        if name.endswith("layers.0.linear_attn.in_proj_qkv.weight"):
            W_l0_qkv = W.copy()
        if name.endswith("layers.32.linear_attn.in_proj_qkv.weight"):
            W_l32_qkv = W.copy()

        for cfg in configs_for(grid, W.shape[1]):
            kind = cfg[0]
            spec = cfg[1:]
            tag = kind + "_" + "_".join(str(s) for s in spec if s is not None)
            t2 = now()
            try:
                What, meta = run_one(kind, W, spec)
            except Exception as e:
                print(f"  FAIL {tag}: {e}", flush=True)
                rec["variants"].append({"tag": tag, "error": repr(e)})
                continue
            metrics = score(W, What, X)
            row = {**meta, **metrics, "wall_s": now() - t2, "tag": tag}
            rec["variants"].append(row)
            print_row(tag, row)
            del What
        rec["wall_s"] = now() - t1
        results["tensors"].append(rec)
        if name.endswith("layers.32.linear_attn.in_proj_qkv.weight"):
            pass
        else:
            if not name.endswith("layers.0.linear_attn.in_proj_qkv.weight"):
                del W

    # cross-layer codebook transfer L0 -> L32
    if W_l0_qkv is not None and W_l32_qkv is not None:
        print("\n===== CROSS-LAYER PQ L0 codebook -> L32 in_proj_qkv =====", flush=True)
        X = hidden_cache.get(32)
        if X is None:
            X = load_hidden(32)
            hidden_cache[32] = X
        What, meta = cross_layer_pq(W_l0_qkv, W_l32_qkv, 8, 256)
        metrics = score(W_l32_qkv, What, X)
        results["cross_layer"] = {**meta, **metrics}
        print_row("cross_L0toL32_pq_d8_K256", {**meta, **metrics, "register_only": False, "table_bytes": meta["codebook_bits_fp16"] // 8})
        del What

    # Viterbi TCQ sample on L0 qkv
    if W_l0_qkv is not None:
        print("\n===== VITERBI TCQ sample (1D, cannot go sub-bit) =====", flush=True)
        sample = W_l0_qkv.reshape(-1)[: 256 * 256]  # 65536 weights, 256 blocks
        vits = []
        for k, L in [(1, 5), (2, 6), (3, 7)]:
            t2 = now()
            r = viterbi_tcq_sample(sample, k, L, block=256)
            r["wall_s"] = now() - t2
            vits.append(r)
            print(
                f"  {r['family']} bpw={r['bpw']:.4f} w_cos={r['weight_cosine']:.5f} "
                f"w_rel={r['weight_rel_l2']:.4f} rss={rss_gb():.3f}GB {r['wall_s']:.2f}s",
                flush=True,
            )
        results["viterbi_sample"] = vits

    results["wall_s"] = now() - t0
    results["peak_rss_gb"] = rss_gb()
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT} wall={results['wall_s']:.1f}s peak_rss={results['peak_rss_gb']:.3f}GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

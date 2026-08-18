#!/usr/bin/env python3
"""Cross-layer structure constructions that shared-basis / gen+residual did not test.

CPU only. Real Qwen3.8-27B BF16. No Metal, no generate, no repo writes.
Peak RSS must stay under 15 GiB.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

ROOT = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
)
ACT = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/"
    "activation-capture-v1"
)
OUT = Path(os.environ.get("OUT", "/tmp/g1_cross_layer_structure.json"))
LOG = Path(os.environ.get("LOG", "/tmp/g1_cross_layer_structure.log"))

HIDDEN = 5120
N_LAYERS = 64
G64 = 64
PREFIX = "language_model.model.layers.{layer}."
T0 = time.time()


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')} t={time.time()-T0:7.1f}s rss={rss_gb():.2f}GiB] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))


def dump(obj) -> None:
    OUT.write_text(json.dumps(obj, indent=2, default=_jsonable))


def sha256_file(path: Path, cap: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if cap is None:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        else:
            h.update(f.read(cap))
    return h.hexdigest()


def parse_all_headers(root: Path) -> dict:
    table = {}
    for shard in sorted(root.glob("model-*.safetensors")):
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            table[name] = {
                "path": str(shard),
                "header_nbytes": n,
                "dtype": info["dtype"],
                "shape": tuple(info["shape"]),
                "begin": int(begin),
                "end": int(end),
            }
    return table


def mmap_bf16_u16(info) -> np.memmap:
    if info["dtype"] != "BF16":
        raise RuntimeError(f"expected BF16, got {info['dtype']}")
    n = int(np.prod(info["shape"]))
    return np.memmap(
        info["path"],
        dtype="<u2",
        mode="r",
        offset=8 + info["header_nbytes"] + info["begin"],
        shape=(n,),
    )


def bf16u16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load_f32(info) -> np.ndarray:
    raw = mmap_bf16_u16(info)
    return bf16u16_to_f32(np.asarray(raw)).reshape(info["shape"]).copy()


def iter_row_tiles(info, tile: int = 256):
    rows, cols = info["shape"]
    raw = mmap_bf16_u16(info)
    for r0 in range(0, rows, tile):
        r1 = min(rows, r0 + tile)
        sl = raw[r0 * cols : r1 * cols]
        yield r0, r1, bf16u16_to_f32(np.asarray(sl)).reshape(r1 - r0, cols)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    if na == 0.0:
        return float("nan")
    return float(np.linalg.norm(a - b) / na)


def w1_from_hist(h1: np.ndarray, h2: np.ndarray, lo: float, hi: float) -> float:
    """W1 between two count histograms on a uniform grid [lo, hi]."""
    p = h1.astype(np.float64)
    q = h2.astype(np.float64)
    ps, qs = p.sum(), q.sum()
    if ps == 0 or qs == 0:
        return float("nan")
    p = p / ps
    q = q / qs
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    dx = (hi - lo) / h1.size
    return float(np.sum(np.abs(cdf_p - cdf_q)) * dx)


def lloyd_1d(x: np.ndarray, k: int, iters: int = 12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.linspace(-1.0, 1.0, k).astype(np.float32)
    qs = np.linspace(0.5, 99.5, k)
    levels = np.percentile(x, qs).astype(np.float64)
    for _ in range(iters):
        edges = 0.5 * (levels[:-1] + levels[1:])
        idx = np.searchsorted(edges, x)
        counts = np.bincount(idx, minlength=k).astype(np.float64)
        sums = np.bincount(idx, weights=x.astype(np.float64), minlength=k)
        nonempty = counts > 0
        levels[nonempty] = sums[nonempty] / counts[nonempty]
        # keep empty slots by linear interpolation so they do not collapse
        if not nonempty.all():
            filled = np.where(nonempty)[0]
            empty = np.where(~nonempty)[0]
            levels[empty] = np.interp(empty, filled, levels[filled])
    return levels.astype(np.float32)


def uniform_levels(bits: int) -> np.ndarray:
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    return (np.arange(qmin, qmax + 1, dtype=np.float32) / float(qmax))


def apply_levels_chunked(
    W: np.ndarray, levels: np.ndarray, G: int = G64, chunk_groups: int = 16384
) -> tuple[float, float, float]:
    """Return (rel_l2, cosine, mse) of group-absmax + shared-level reconstruction."""
    flat = np.ascontiguousarray(W, dtype=np.float32).ravel()
    n = int(flat.size)
    pad = (-n) % G
    if pad:
        work = np.empty(n + pad, dtype=np.float32)
        work[:n] = flat
        work[n:] = 0.0
    else:
        work = flat
    groups = work.reshape(-1, G)
    ng = groups.shape[0]
    levels = np.sort(np.asarray(levels, dtype=np.float32))
    edges = 0.5 * (levels[:-1] + levels[1:])
    sse = 0.0
    sww = 0.0
    swh = 0.0
    shh = 0.0
    for g0 in range(0, ng, chunk_groups):
        g1 = min(ng, g0 + chunk_groups)
        blk = groups[g0:g1]
        amax = np.max(np.abs(blk), axis=1, keepdims=True)
        amax = np.maximum(amax, 1e-12)
        u = blk / amax
        q = np.searchsorted(edges, u)
        recon = levels[q] * amax
        # ignore pad on last group
        if g1 == ng and pad:
            valid = blk.copy()
            recon_v = recon.copy()
            valid[-1, G - pad :] = 0.0
            recon_v[-1, G - pad :] = 0.0
            diff = (valid - recon_v).astype(np.float64)
            ww = valid.astype(np.float64)
            hh = recon_v.astype(np.float64)
        else:
            diff = (blk - recon).astype(np.float64)
            ww = blk.astype(np.float64)
            hh = recon.astype(np.float64)
        sse += float(np.square(diff).sum())
        sww += float(np.square(ww).sum())
        swh += float((ww * hh).sum())
        shh += float(np.square(hh).sum())
    rel = float(np.sqrt(sse) / np.sqrt(sww)) if sww > 0 else float("nan")
    cos = float(swh / (np.sqrt(sww) * np.sqrt(shh))) if sww > 0 and shh > 0 else float("nan")
    mse = float(sse / n)
    return rel, cos, mse


def group_amax_and_hist(W: np.ndarray, G: int = G64, nbins: int = 64):
    flat = np.ascontiguousarray(W, dtype=np.float32).ravel()
    n = int(flat.size)
    pad = (-n) % G
    if pad:
        work = np.empty(n + pad, dtype=np.float32)
        work[:n] = flat
        work[n:] = 0.0
    else:
        work = flat
    groups = work.reshape(-1, G)
    amax = np.max(np.abs(groups), axis=1, keepdims=True)
    amax_safe = np.maximum(amax, 1e-12)
    u = groups / amax_safe
    if pad:
        u = u.copy()
        u[-1, G - pad :] = 0.0
        amax = amax.copy()
        # last group's amax still valid on real cells
    hist, _ = np.histogram(u.ravel()[:n], bins=nbins, range=(-1.0, 1.0))
    return amax.ravel().astype(np.float32), hist.astype(np.int64), u.ravel()[:n]


def q4_codes_and_scales(W: np.ndarray, G: int = G64, bits: int = 4):
    """G0-style absmax uniform codes. Returns int8 codes (qmin..qmax) and f32 scales."""
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    flat = np.ascontiguousarray(W, dtype=np.float32).ravel()
    n = int(flat.size)
    pad = (-n) % G
    if pad:
        work = np.empty(n + pad, dtype=np.float32)
        work[:n] = flat
        work[n:] = 0.0
    else:
        work = flat
    groups = work.reshape(-1, G)
    amax = np.max(np.abs(groups), axis=1, keepdims=True)
    scale = np.maximum(amax / float(qmax), 1e-12)
    q = np.rint(groups / scale)
    q = np.clip(q, qmin, qmax).astype(np.int8)
    if pad:
        q = q.copy()
        q[-1, G - pad :] = 0
    return q, scale.astype(np.float32).ravel(), n, pad


def sample_unit_patches(W: np.ndarray, d: int, n_keep: int, rng: np.random.Generator):
    rows, cols = W.shape
    if cols % d != 0:
        usable_cols = cols - (cols % d)
        W = W[:, :usable_cols]
        cols = usable_cols
    n_per_row = cols // d
    n_total = rows * n_per_row
    take = min(n_keep, n_total)
    idx = rng.choice(n_total, size=take, replace=False)
    row = idx // n_per_row
    col0 = (idx % n_per_row) * d
    patches = np.empty((take, d), dtype=np.float32)
    for i in range(take):
        patches[i] = W[row[i], col0[i] : col0[i] + d]
    rms = np.sqrt(np.mean(np.square(patches), axis=1, keepdims=True))
    rms = np.maximum(rms, 1e-12)
    templates = patches / rms
    return templates, rms.ravel()


def hash_templates(templates: np.ndarray, bits_per_dim: int = 2) -> np.ndarray:
    """Deterministic 2-bit/dim hash of unit templates. bins from N(0, 1/sqrt(d))."""
    d = templates.shape[1]
    sigma = 1.0 / np.sqrt(d)
    # 4 bins: (-inf,-σ], (-σ,0], (0,σ], (σ,+inf)
    edges = np.array([-sigma, 0.0, sigma], dtype=np.float32)
    q = np.searchsorted(edges, templates)
    # pack first 16 dims into uint32 (2 bits each)
    use = min(d, 16)
    acc = np.zeros(templates.shape[0], dtype=np.uint32)
    for i in range(use):
        acc |= (q[:, i].astype(np.uint32) & 3) << (2 * i)
    return acc


def kmeans_fit(X: np.ndarray, k: int, iters: int = 8, seed: int = 0):
    """Mini Lloyd on rows of X. Returns centroids (k, d)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)
    pick = rng.choice(n, size=k, replace=False)
    C = X[pick].astype(np.float32).copy()
    Xf = X.astype(np.float32, copy=False)
    for _ in range(iters):
        # assign
        # chunked distances
        assign = np.empty(n, dtype=np.int32)
        bs = 4096
        C64 = C.astype(np.float64)
        nC2 = np.square(C64).sum(axis=1)
        for i0 in range(0, n, bs):
            i1 = min(n, i0 + bs)
            blk = Xf[i0:i1].astype(np.float64)
            # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c
            nX2 = np.square(blk).sum(axis=1, keepdims=True)
            d2 = nX2 + nC2[None, :] - 2.0 * (blk @ C64.T)
            assign[i0:i1] = d2.argmin(axis=1).astype(np.int32)
        # update
        new = np.zeros_like(C, dtype=np.float64)
        cnt = np.bincount(assign, minlength=k).astype(np.float64)
        for j in range(k):
            mask = assign == j
            if mask.any():
                new[j] = Xf[mask].mean(axis=0)
            else:
                new[j] = C[j]
        C = new.astype(np.float32)
    return C


def kmeans_assign_relmse(X: np.ndarray, C: np.ndarray) -> dict:
    n = X.shape[0]
    bs = 4096
    C64 = C.astype(np.float64)
    nC2 = np.square(C64).sum(axis=1)
    sse = 0.0
    sxx = 0.0
    sxc = 0.0
    used = np.zeros(C.shape[0], dtype=np.int64)
    for i0 in range(0, n, bs):
        i1 = min(n, i0 + bs)
        blk = X[i0:i1].astype(np.float64)
        nX2 = np.square(blk).sum(axis=1, keepdims=True)
        d2 = nX2 + nC2[None, :] - 2.0 * (blk @ C64.T)
        a = d2.argmin(axis=1)
        rec = C64[a]
        sse += float(np.square(blk - rec).sum())
        sxx += float(np.square(blk).sum())
        sxc += float((blk * rec).sum())
        used += np.bincount(a, minlength=C.shape[0])
    return {
        "rel_mse": float(sse / sxx) if sxx > 0 else float("nan"),
        "rel_l2": float(np.sqrt(sse) / np.sqrt(sxx)) if sxx > 0 else float("nan"),
        "n": int(n),
        "k": int(C.shape[0]),
        "n_used_centroids": int((used > 0).sum()),
        "frac_used": float((used > 0).mean()),
    }


def wx_cosine(W: np.ndarray, What_flat_or_fn, X: np.ndarray) -> float:
    """cos(X @ W.T, X @ What.T) on the provided token rows. W is (out, in)."""
    # we reconstruct What by applying levels outside and pass What
    raise NotImplementedError


def reconstruct_with_levels(W: np.ndarray, levels: np.ndarray, G: int = G64) -> np.ndarray:
    flat = np.ascontiguousarray(W, dtype=np.float32).ravel()
    n = int(flat.size)
    pad = (-n) % G
    if pad:
        work = np.empty(n + pad, dtype=np.float32)
        work[:n] = flat
        work[n:] = 0.0
    else:
        work = flat.copy()
    groups = work.reshape(-1, G)
    levels = np.sort(np.asarray(levels, dtype=np.float32))
    edges = 0.5 * (levels[:-1] + levels[1:])
    out = np.empty_like(groups)
    chunk = 8192
    ng = groups.shape[0]
    for g0 in range(0, ng, chunk):
        g1 = min(ng, g0 + chunk)
        blk = groups[g0:g1]
        amax = np.maximum(np.max(np.abs(blk), axis=1, keepdims=True), 1e-12)
        u = blk / amax
        q = np.searchsorted(edges, u)
        out[g0:g1] = levels[q] * amax
    rec = out.ravel()[:n].reshape(W.shape)
    return rec


def matvec_cos(W: np.ndarray, What: np.ndarray, X: np.ndarray) -> dict:
    """X: (T, in). W, What: (out, in)."""
    Y = X.astype(np.float32) @ W.T
    Yh = X.astype(np.float32) @ What.T
    return {
        "out_cosine": cosine(Y, Yh),
        "out_rel_l2": rel_l2(Y, Yh),
        "n_tokens": int(X.shape[0]),
        "in_dim": int(X.shape[1]),
        "out_dim": int(W.shape[0]),
        "rows_per_in_dim": float(X.shape[0]) / float(X.shape[1]),
    }


# ---------------------------------------------------------------------------
# Class map
# ---------------------------------------------------------------------------

MLP_LAYERS = list(range(N_LAYERS))
GQA_LAYERS = [i for i in range(N_LAYERS) if (i + 1) % 4 == 0]
DN_LAYERS = [i for i in range(N_LAYERS) if (i + 1) % 4 != 0]

PROBE_MLP = [0, 1, 15, 16, 31, 32, 47, 48, 62, 63]
PROBE_GQA = [3, 7, 15, 31, 47, 63]
PROBE_DN = [0, 1, 16, 32, 48, 62]

CLASSES = [
    {
        "name": "mlp.gate_proj",
        "suffix": "mlp.gate_proj.weight",
        "layers": MLP_LAYERS,
        "probe": PROBE_MLP,
        "family": "mlp",
    },
    {
        "name": "mlp.up_proj",
        "suffix": "mlp.up_proj.weight",
        "layers": MLP_LAYERS,
        "probe": PROBE_MLP,
        "family": "mlp",
    },
    {
        "name": "mlp.down_proj",
        "suffix": "mlp.down_proj.weight",
        "layers": MLP_LAYERS,
        "probe": PROBE_MLP,
        "family": "mlp",
    },
    {
        "name": "self_attn.q_proj",
        "suffix": "self_attn.q_proj.weight",
        "layers": GQA_LAYERS,
        "probe": PROBE_GQA,
        "family": "gqa",
    },
    {
        "name": "self_attn.k_proj",
        "suffix": "self_attn.k_proj.weight",
        "layers": GQA_LAYERS,
        "probe": PROBE_GQA,
        "family": "gqa",
    },
    {
        "name": "self_attn.v_proj",
        "suffix": "self_attn.v_proj.weight",
        "layers": GQA_LAYERS,
        "probe": PROBE_GQA,
        "family": "gqa",
    },
    {
        "name": "self_attn.o_proj",
        "suffix": "self_attn.o_proj.weight",
        "layers": GQA_LAYERS,
        "probe": PROBE_GQA,
        "family": "gqa",
    },
    {
        "name": "linear_attn.in_proj_qkv",
        "suffix": "linear_attn.in_proj_qkv.weight",
        "layers": DN_LAYERS,
        "probe": PROBE_DN,
        "family": "dn",
    },
    {
        "name": "linear_attn.in_proj_z",
        "suffix": "linear_attn.in_proj_z.weight",
        "layers": DN_LAYERS,
        "probe": PROBE_DN,
        "family": "dn",
    },
    {
        "name": "linear_attn.out_proj",
        "suffix": "linear_attn.out_proj.weight",
        "layers": DN_LAYERS,
        "probe": PROBE_DN,
        "family": "dn",
    },
]

ADJ_PAIRS_MLP = [(0, 1), (15, 16), (31, 32), (47, 48), (62, 63)]
D16_PAIRS_MLP = [(0, 16), (15, 31), (31, 47), (47, 63)]
GQA_ADJ = list(zip(GQA_LAYERS[:-1], GQA_LAYERS[1:]))
GQA_ADJ_PROBE = [(3, 7), (15, 19), (31, 35), (47, 51), (59, 63)]
GQA_D16 = [(3, 19), (15, 31), (31, 47), (47, 63)]  # 4 GQA steps = 16 layers
DN_ADJ_PROBE = [(0, 1), (4, 5), (16, 17), (32, 33), (48, 49), (61, 62)]
DN_D16 = [(0, 16), (16, 32), (32, 48)]


def tensor_name(layer: int, suffix: str) -> str:
    return f"language_model.model.layers.{layer}.{suffix}"


def load_hidden(layer: int) -> np.ndarray:
    p = ACT / "hidden" / f"L{layer:02d}.f32"
    arr = np.fromfile(p, dtype=np.float32)
    return arr.reshape(256, HIDDEN)


def profile_from_tiles(info) -> dict:
    rows, cols = info["shape"]
    row_ssq = np.zeros(rows, dtype=np.float64)
    col_ssq = np.zeros(cols, dtype=np.float64)
    hist = np.zeros(64, dtype=np.int64)
    fro2 = 0.0
    n = 0
    amax_ssq = 0.0
    amax_sum = 0.0
    n_groups = 0
    sample_u = []
    rng = np.random.default_rng(0xA11CE)
    for r0, r1, tile in iter_row_tiles(info, tile=256):
        t = tile.astype(np.float32, copy=False)
        row_ssq[r0:r1] = np.square(t.astype(np.float64)).sum(axis=1)
        col_ssq += np.square(t.astype(np.float64)).sum(axis=0)
        fro2 += float(np.square(t.astype(np.float64)).sum())
        n += t.size
        # group hist along last axis, groups of 64
        c_use = cols - (cols % G64)
        if c_use > 0:
            g = t[:, :c_use].reshape(t.shape[0], c_use // G64, G64)
            am = np.max(np.abs(g), axis=2, keepdims=True)
            am = np.maximum(am, 1e-12)
            u = g / am
            h, _ = np.histogram(u.ravel(), bins=64, range=(-1.0, 1.0))
            hist += h
            amax_ssq += float(np.square(am.astype(np.float64)).sum())
            amax_sum += float(am.astype(np.float64).sum())
            n_groups += int(am.size)
            # reservoir a bit of u for later (not all layers)
            if len(sample_u) < 8:
                flat_u = u.ravel()
                take = min(65536, flat_u.size)
                sel = rng.choice(flat_u.size, size=take, replace=False)
                sample_u.append(flat_u[sel].astype(np.float32))
    row_rms = np.sqrt(row_ssq / cols)
    col_rms = np.sqrt(col_ssq / rows)
    return {
        "shape": [int(rows), int(cols)],
        "n": int(n),
        "fro": float(np.sqrt(fro2)),
        "mean_abs": None,
        "row_rms": row_rms.astype(np.float32),
        "col_rms": col_rms.astype(np.float32),
        "hist_u": hist,
        "amax_mean": float(amax_sum / n_groups) if n_groups else float("nan"),
        "amax_rms": float(np.sqrt(amax_ssq / n_groups)) if n_groups else float("nan"),
        "sample_u": np.concatenate(sample_u) if sample_u else np.zeros(0, np.float32),
    }


def pairwise_profile_stats(vecs: dict[int, np.ndarray], pairs: list[tuple[int, int]]) -> list:
    out = []
    rng = np.random.default_rng(7)
    for i, j in pairs:
        if i not in vecs or j not in vecs:
            continue
        a, b = vecs[i], vecs[j]
        ones = np.ones_like(a, dtype=np.float64)
        ac = a.astype(np.float64) - float(np.mean(a))
        bc = b.astype(np.float64) - float(np.mean(b))
        bshuf = b.copy()
        rng.shuffle(bshuf)
        out.append(
            {
                "i": int(i),
                "j": int(j),
                "d": int(j - i),
                "cosine": cosine(a, b),
                "cosine_centered": cosine(ac, bc),
                "cosine_vs_ones_i": cosine(a, ones),
                "cosine_shuffled": cosine(a, bshuf),
                "rel_l2": rel_l2(a, b),
                "cv_i": float(np.std(a) / (np.mean(np.abs(a)) + 1e-12)),
            }
        )
    return out


def stack_energy(vecs: dict[int, np.ndarray], max_rank: int = 8) -> dict:
    layers = sorted(vecs)
    M = np.stack([vecs[l] for l in layers], axis=0).astype(np.float64)  # L x D
    M0 = M - M.mean(axis=0, keepdims=True)
    # SVD of L x D, L is 16..64, cheap
    try:
        _, s, _ = np.linalg.svd(M0, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"ok": False}
    e = s ** 2
    tot = float(e.sum()) if e.size else 0.0
    frac = (np.cumsum(e) / tot).tolist() if tot > 0 else []
    # also uncentered rank-1 (shared template * per-layer scale)
    # M ≈ u[:,None] * v[None,:]
    # take top left/right of uncentered
    try:
        _, s_u, _ = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        s_u = s
    e_u = s_u ** 2
    tot_u = float(e_u.sum()) if e_u.size else 0.0
    frac_u = (np.cumsum(e_u) / tot_u).tolist() if tot_u > 0 else []
    return {
        "ok": True,
        "n_layers": int(len(layers)),
        "dim": int(M.shape[1]),
        "centered_energy_frac": frac[:max_rank],
        "uncentered_energy_frac": frac_u[:max_rank],
        "s0_over_s1_centered": float(s[0] / s[1]) if s.size > 1 and s[1] > 0 else None,
        "mean_vec_std": float(M.std()),
        "mean_layer_norm": float(np.linalg.norm(M, axis=1).mean()),
    }


def positional_pair(info_a, info_b, bits: int = 4) -> dict:
    Wa = load_f32(info_a)
    Wb = load_f32(info_b)
    assert Wa.shape == Wb.shape
    qa, sa, n, pad = q4_codes_and_scales(Wa, bits=bits)
    qb, sb, _, _ = q4_codes_and_scales(Wb, bits=bits)
    qa_f = qa.ravel()[:n]
    qb_f = qb.ravel()[:n]
    # signs of raw weights
    sa_w = np.sign(Wa.ravel())
    sb_w = np.sign(Wb.ravel())
    nz = (sa_w != 0) & (sb_w != 0)
    sign_agree = float((sa_w[nz] == sb_w[nz]).mean()) if nz.any() else float("nan")
    code_equal = float((qa_f == qb_f).mean())
    code_l1 = float(np.abs(qa_f.astype(np.int16) - qb_f.astype(np.int16)).mean())
    # independent baseline: permute B codes in blocks of 4096 to keep hist, destroy site
    rng = np.random.default_rng(123)
    qb_perm = qb_f.copy()
    rng.shuffle(qb_perm)
    code_equal_shuf = float((qa_f == qb_perm).mean())
    code_l1_shuf = float(np.abs(qa_f.astype(np.int16) - qb_perm.astype(np.int16)).mean())
    # scale cosine + trivial-positive controls
    scale_cos = cosine(sa, sb)
    ones = np.ones_like(sa)
    sa_c = sa.astype(np.float64) - float(sa.mean())
    sb_c = sb.astype(np.float64) - float(sb.mean())
    sb_shuf = sb.copy()
    rng.shuffle(sb_shuf)
    scale_cos_centered = cosine(sa_c, sb_c)
    scale_cos_ones_a = cosine(sa, ones)
    scale_cos_shuf = cosine(sa, sb_shuf)
    scale_cv = float(sa.std() / (sa.mean() + 1e-12))
    # weight cosine / rel delta (site-aligned) — cheap confirmation
    w_cos = cosine(Wa, Wb)
    w_rel = rel_l2(Wa, Wb)
    # residual of delta vs residual of B after Q4 of each
    # Q4(B) error vs Q4(B-A)+A error would need extra; skip if we already know w_rel~sqrt2
    out = {
        "shape": list(Wa.shape),
        "n": int(n),
        "bits": int(bits),
        "sign_agree_nonzero": sign_agree,
        "frac_both_nonzero": float(nz.mean()),
        "code_equal": code_equal,
        "code_equal_shuffled": code_equal_shuf,
        "code_equal_minus_shuffle": float(code_equal - code_equal_shuf),
        "code_l1": code_l1,
        "code_l1_shuffled": code_l1_shuf,
        "scale_cosine": scale_cos,
        "scale_cosine_centered": scale_cos_centered,
        "scale_cosine_vs_ones": scale_cos_ones_a,
        "scale_cosine_shuffled": scale_cos_shuf,
        "scale_cv": scale_cv,
        "weight_cosine": w_cos,
        "weight_rel_l2": w_rel,
        "independent_unit_rel_l2": 1.4142135623730951,
    }
    del Wa, Wb, qa, qb, qa_f, qb_f
    gc.collect()
    return out


def head_structure_gqa_q(info, layer: int) -> dict:
    """q_proj 12288 x 5120 = 24 heads * (q 256 + gate 256)."""
    W = load_f32(info)
    rows, cols = W.shape
    assert rows == 12288 and cols == 5120, W.shape
    heads_q = []
    heads_g = []
    for h in range(24):
        block = W[h * 512 : (h + 1) * 512]
        heads_q.append(block[:256])
        heads_g.append(block[256:])
    def pairwise_mean_cos(mats):
        cs = []
        for i in range(len(mats)):
            for j in range(i + 1, len(mats)):
                cs.append(cosine(mats[i], mats[j]))
        arr = np.asarray(cs, dtype=np.float64)
        return {
            "n_pairs": int(arr.size),
            "mean": float(arr.mean()),
            "p95": float(np.quantile(arr, 0.95)),
            "max": float(arr.max()),
            "min": float(arr.min()),
        }

    def rms_profile_cos(mats):
        rms = [np.sqrt(np.mean(np.square(m.astype(np.float64)), axis=1)) for m in mats]
        return pairwise_mean_cos(rms)

    # shared levels across heads: fit on head 0, apply to others (sample)
    def levels_transfer(mats, bits=4):
        u0 = []
        for m in mats[:1]:
            _, _, u = group_amax_and_hist(m)
            u0.append(u)
        u0 = np.concatenate(u0)
        rng = np.random.default_rng(0)
        if u0.size > 400_000:
            u0 = u0[rng.choice(u0.size, 400_000, replace=False)]
        lv_priv = []
        lv0 = lloyd_1d(u0, 1 << bits)
        rels_shared = []
        rels_priv = []
        for m in mats:
            _, _, u = group_amax_and_hist(m)
            if u.size > 400_000:
                us = u[rng.choice(u.size, 400_000, replace=False)]
            else:
                us = u
            lv = lloyd_1d(us, 1 << bits)
            lv_priv.append(lv)
            # score on this head full
            r_s, _, _ = apply_levels_chunked(m, lv0)
            r_p, _, _ = apply_levels_chunked(m, lv)
            rels_shared.append(r_s)
            rels_priv.append(r_p)
        return {
            "bits": bits,
            "shared_from_head0_rel_l2_mean": float(np.mean(rels_shared)),
            "private_rel_l2_mean": float(np.mean(rels_priv)),
            "shared_over_private": float(np.mean(rels_shared) / np.mean(rels_priv)),
            "level_l2_vs_head0_mean": float(
                np.mean([np.linalg.norm(lv.astype(np.float64) - lv0.astype(np.float64)) for lv in lv_priv])
            ),
        }

    out = {
        "layer": layer,
        "q_weight_cos": pairwise_mean_cos(heads_q),
        "gate_weight_cos": pairwise_mean_cos(heads_g),
        "q_rowrms_cos": rms_profile_cos(heads_q),
        "gate_rowrms_cos": rms_profile_cos(heads_g),
        "q_shared_levels": levels_transfer(heads_q, 4),
        "gate_shared_levels": levels_transfer(heads_g, 4),
    }
    # template hash overlap across q heads
    rng = np.random.default_rng(1)
    sets = []
    for m in heads_q:
        t, _ = sample_unit_patches(m, d=16, n_keep=4096, rng=rng)
        sets.append(set(hash_templates(t).tolist()))
    jacc = []
    for i in range(24):
        for j in range(i + 1, 24):
            a, b = sets[i], sets[j]
            u = len(a | b)
            jacc.append(len(a & b) / u if u else 0.0)
    out["q_template_jaccard_mean"] = float(np.mean(jacc))
    out["q_template_jaccard_max"] = float(np.max(jacc))
    del W
    gc.collect()
    return out


def head_structure_dn_qkv(info, layer: int) -> dict:
    """in_proj_qkv 10240 x 5120 = Q 16x128 | K 16x128 | V 48x128."""
    W = load_f32(info)
    assert W.shape == (10240, 5120), W.shape
    q = [W[i * 128 : (i + 1) * 128] for i in range(16)]
    k = [W[2048 + i * 128 : 2048 + (i + 1) * 128] for i in range(16)]
    v = [W[4096 + i * 128 : 4096 + (i + 1) * 128] for i in range(48)]

    def pairwise_mean_cos(mats, max_pairs=200):
        cs = []
        n = len(mats)
        for i in range(n):
            for j in range(i + 1, n):
                cs.append(cosine(mats[i], mats[j]))
                if len(cs) >= max_pairs:
                    break
            if len(cs) >= max_pairs:
                break
        arr = np.asarray(cs, dtype=np.float64)
        return {
            "n_pairs": int(arr.size),
            "mean": float(arr.mean()),
            "p95": float(np.quantile(arr, 0.95)),
            "max": float(arr.max()),
            "min": float(arr.min()),
        }

    def rms_profile_cos(mats, max_pairs=200):
        rms = [np.sqrt(np.mean(np.square(m.astype(np.float64)), axis=1)) for m in mats]
        return pairwise_mean_cos(rms, max_pairs=max_pairs)

    def levels_xfer(mats, bits=4, n_eval=8):
        rng = np.random.default_rng(2)
        _, _, u0 = group_amax_and_hist(mats[0])
        if u0.size > 200_000:
            u0 = u0[rng.choice(u0.size, 200_000, replace=False)]
        lv0 = lloyd_1d(u0, 1 << bits)
        rels_s, rels_p = [], []
        eval_idx = list(range(min(n_eval, len(mats))))
        for i in eval_idx:
            r_s, _, _ = apply_levels_chunked(mats[i], lv0)
            _, _, ui = group_amax_and_hist(mats[i])
            if ui.size > 200_000:
                ui = ui[rng.choice(ui.size, 200_000, replace=False)]
            lvi = lloyd_1d(ui, 1 << bits)
            r_p, _, _ = apply_levels_chunked(mats[i], lvi)
            rels_s.append(r_s)
            rels_p.append(r_p)
        return {
            "bits": bits,
            "n_eval": len(eval_idx),
            "shared_from_head0_rel_l2_mean": float(np.mean(rels_s)),
            "private_rel_l2_mean": float(np.mean(rels_p)),
            "shared_over_private": float(np.mean(rels_s) / np.mean(rels_p)),
        }

    out = {
        "layer": layer,
        "q_weight_cos": pairwise_mean_cos(q),
        "k_weight_cos": pairwise_mean_cos(k),
        "v_weight_cos": pairwise_mean_cos(v),
        "q_rowrms_cos": rms_profile_cos(q),
        "k_rowrms_cos": rms_profile_cos(k),
        "v_rowrms_cos": rms_profile_cos(v),
        "q_shared_levels": levels_xfer(q),
        "k_shared_levels": levels_xfer(k),
        "v_shared_levels": levels_xfer(v, n_eval=8),
    }
    del W
    gc.collect()
    return out


def embed_lm_head_alignment(table: dict) -> dict:
    e_info = table["language_model.model.embed_tokens.weight"]
    h_info = table["language_model.lm_head.weight"]
    assert e_info["shape"] == h_info["shape"], (e_info["shape"], h_info["shape"])
    rows, cols = e_info["shape"]
    dot = 0.0
    ee = 0.0
    hh = 0.0
    d2 = 0.0
    # also row-rms cosine
    e_row = np.zeros(rows, dtype=np.float64)
    h_row = np.zeros(rows, dtype=np.float64)
    tile = 1024
    e_raw = mmap_bf16_u16(e_info)
    h_raw = mmap_bf16_u16(h_info)
    for r0 in range(0, rows, tile):
        r1 = min(rows, r0 + tile)
        et = bf16u16_to_f32(np.asarray(e_raw[r0 * cols : r1 * cols])).reshape(r1 - r0, cols)
        ht = bf16u16_to_f32(np.asarray(h_raw[r0 * cols : r1 * cols])).reshape(r1 - r0, cols)
        et64 = et.astype(np.float64)
        ht64 = ht.astype(np.float64)
        dot += float((et64 * ht64).sum())
        ee += float(np.square(et64).sum())
        hh += float(np.square(ht64).sum())
        d2 += float(np.square(et64 - ht64).sum())
        e_row[r0:r1] = np.sqrt(np.square(et64).sum(axis=1) / cols)
        h_row[r0:r1] = np.sqrt(np.square(ht64).sum(axis=1) / cols)
    cos = float(dot / (np.sqrt(ee) * np.sqrt(hh)))
    rel = float(np.sqrt(d2) / np.sqrt(ee))
    return {
        "shape": [int(rows), int(cols)],
        "weight_cosine": cos,
        "weight_rel_l2": rel,
        "row_rms_cosine": cosine(e_row, h_row),
        "embed_fro": float(np.sqrt(ee)),
        "lm_head_fro": float(np.sqrt(hh)),
        "tied_would_save_elements": int(rows * cols),
        "tied_save_complete_bpw_if_drop_one_at_g0": float(rows * cols * 4.252735126866492 / 26895998464),
    }


def activation_gravity() -> dict:
    norms = []
    rms_tok = []
    rms_dim = []
    # per-token L2 and per-dim rms
    prev = None
    ratios = []
    for L in range(N_LAYERS):
        H = load_hidden(L)  # 256 x 5120
        tok = np.linalg.norm(H.astype(np.float64), axis=1)
        dim = np.sqrt(np.mean(np.square(H.astype(np.float64)), axis=0))
        rec = {
            "layer": L,
            "mean_token_l2": float(tok.mean()),
            "std_token_l2": float(tok.std()),
            "mean_hidden_rms": float(np.sqrt(np.mean(np.square(H.astype(np.float64))))),
            "max_abs": float(np.max(np.abs(H))),
        }
        norms.append(rec)
        if prev is not None:
            ratios.append(rec["mean_token_l2"] / prev["mean_token_l2"])
        prev = rec
        rms_tok.append(tok.astype(np.float32))
        rms_dim.append(dim.astype(np.float32))
    # pairwise hidden cosine adjacent (token-averaged)
    adj_cos = []
    for L in range(N_LAYERS - 1):
        a = load_hidden(L)
        b = load_hidden(L + 1)
        # per-token cosine then mean
        cs = []
        for t in range(a.shape[0]):
            cs.append(cosine(a[t], b[t]))
        adj_cos.append(float(np.mean(cs)))
    # effective rank of stacked activations — UNDERDETERMINED
    # use every 4th layer to keep 16*256=4096 rows
    stack = np.concatenate([load_hidden(L) for L in range(0, 64, 4)], axis=0).astype(np.float32)
    # 4096 x 5120
    stack0 = stack - stack.mean(axis=0, keepdims=True)
    # randomized energy via Gram of 4096 (cheap)
    G = stack0.astype(np.float64) @ stack0.astype(np.float64).T  # 4096^2
    ev = np.linalg.eigvalsh(G)
    ev = np.clip(ev[::-1], 0, None)
    tot = float(ev.sum())
    frac = (np.cumsum(ev) / tot).tolist() if tot > 0 else []
    # k90
    k90 = int(np.searchsorted(np.cumsum(ev) / tot, 0.90)) + 1 if tot > 0 else None
    k99 = int(np.searchsorted(np.cumsum(ev) / tot, 0.99)) + 1 if tot > 0 else None
    return {
        "n_tokens": 256,
        "hidden": 5120,
        "rows_per_dim": 256 / 5120,
        "label": "UNDERDETERMINED_FOR_SCALE_OK_FOR_RANKING",
        "per_layer": norms,
        "adjacent_token_l2_ratio_mean": float(np.mean(ratios)),
        "adjacent_token_l2_ratio_min": float(np.min(ratios)),
        "adjacent_token_l2_ratio_max": float(np.max(ratios)),
        "L63_over_L00_mean_token_l2": float(norms[63]["mean_token_l2"] / norms[0]["mean_token_l2"]),
        "L63_over_L00_mean_hidden_rms": float(norms[63]["mean_hidden_rms"] / norms[0]["mean_hidden_rms"]),
        "adjacent_hidden_cosine_mean": float(np.mean(adj_cos)),
        "adjacent_hidden_cosine_min": float(np.min(adj_cos)),
        "adjacent_hidden_cosine_max": float(np.max(adj_cos)),
        "stacked_every4_n_rows": int(stack.shape[0]),
        "stacked_energy_frac_top": frac[:32],
        "stacked_k90": k90,
        "stacked_k99": k99,
        "capture_schema_note": "post-norm hidden per layer, 256 tokens",
    }


def bits_complete_shared_levels(n_weights: int, n_tensors: int, bits: int, G: int, n_levels: int) -> dict:
    groups = int(np.ceil(n_weights / G))
    # per-tensor private learned: groups*(bits*G + 16) + 16*n_levels
    priv = groups * (bits * G + 16) + 16 * n_levels
    # shared levels: groups*(bits*G + 16) + 16*n_levels  (levels once, not per tensor)
    # but this function is called per class total
    shared = groups * (bits * G + 16) + 16 * n_levels  # levels once for whole class
    return {
        "n_weights": int(n_weights),
        "n_tensors": int(n_tensors),
        "bits": bits,
        "G": G,
        "private_bpw": priv / n_weights,
        "shared_levels_bpw": shared / n_weights,
        "level_tax_private_bpw": (16 * n_levels * n_tensors) / n_weights,
        "level_tax_shared_bpw": (16 * n_levels) / n_weights,
        "delta_bpw_from_sharing_levels": (16 * n_levels * (n_tensors - 1)) / n_weights,
    }


def bits_complete_dict(n_weights: int, d: int, K: int, scale_bits: int = 16) -> dict:
    n_sites = n_weights / d
    idx_bits = float(np.ceil(np.log2(K)))
    book_bits = K * d * 16  # f16 centroids
    payload = n_sites * idx_bits + n_sites * scale_bits + book_bits
    return {
        "d": d,
        "K": K,
        "n_sites": n_sites,
        "index_bpw": idx_bits / d,
        "scale_bpw": scale_bits / d,
        "book_bpw": book_bits / n_weights,
        "complete_bpw_no_residual": payload / n_weights,
        "note": "no residual. If templates do not reconstruct, add residual codec on top.",
    }


def process_class(table: dict, spec: dict, X_by_layer: dict, report: dict) -> None:
    name = spec["name"]
    log(f"=== class {name} ===")
    row_rms = {}
    col_rms = {}
    hists = {}
    fro = {}
    shapes = {}
    sample_u = {}
    private_levels = {2: {}, 3: {}, 4: {}}
    private_score = {2: {}, 3: {}, 4: {}}
    uniform_score = {2: {}, 3: {}, 4: {}}
    patch_sets = {}
    patch_templates = {}
    wx_scores = []

    # Pass 1: all layers profiles (tiles). Probe layers also collect u-sample + patches + private levels + scores.
    for L in spec["layers"]:
        key = tensor_name(L, spec["suffix"])
        info = table[key]
        shapes[L] = tuple(info["shape"])
        prof = profile_from_tiles(info)
        row_rms[L] = prof["row_rms"]
        col_rms[L] = prof["col_rms"]
        hists[L] = prof["hist_u"]
        fro[L] = prof["fro"]
        if L in spec["probe"]:
            sample_u[L] = prof["sample_u"]
            # load once for private codec + patches
            W = load_f32(info)
            rng = np.random.default_rng(10 + L)
            # extra u sample from full W for level fit (better than tile scraps)
            _, _, u_all = group_amax_and_hist(W)
            if u_all.size > 500_000:
                fit = u_all[rng.choice(u_all.size, 500_000, replace=False)]
                hold = u_all[rng.choice(u_all.size, 500_000, replace=False)]
            else:
                fit = u_all
                hold = u_all
            sample_u[L] = fit
            for bits in (2, 3, 4):
                lv = lloyd_1d(fit, 1 << bits)
                private_levels[bits][L] = lv
                r_p, c_p, _ = apply_levels_chunked(W, lv)
                r_u, c_u, _ = apply_levels_chunked(W, uniform_levels(bits))
                private_score[bits][L] = {"rel_l2": r_p, "cosine": c_p}
                uniform_score[bits][L] = {"rel_l2": r_u, "cosine": c_u}
            # patches
            tmpl, _ = sample_unit_patches(W, d=16, n_keep=16384, rng=rng)
            patch_templates[L] = tmpl
            patch_sets[L] = set(hash_templates(tmpl).tolist())
            # WX on first/last probe if GEMV against hidden
            if W.shape[1] == HIDDEN and L in (spec["probe"][0], spec["probe"][-1]) and L in X_by_layer:
                X = X_by_layer[L][192:]  # holdout 64 tokens
                for bits, tag in ((4, "priv4"), (2, "priv2")):
                    What = reconstruct_with_levels(W, private_levels[bits][L])
                    rec = matvec_cos(W, What, X)
                    rec.update({"layer": L, "class": name, "rule": tag, "bits": bits})
                    wx_scores.append(rec)
                    del What
                # also uniform 4
                What = reconstruct_with_levels(W, uniform_levels(4))
                rec = matvec_cos(W, What, X)
                rec.update({"layer": L, "class": name, "rule": "uniform4", "bits": 4})
                wx_scores.append(rec)
                del What
            del W, u_all, fit, hold
            gc.collect()
            log(
                f"  L{L} {info['shape']} priv4={private_score[4][L]['rel_l2']:.5f} "
                f"uni4={uniform_score[4][L]['rel_l2']:.5f} "
                f"priv2={private_score[2][L]['rel_l2']:.5f}"
            )
        else:
            if L % 8 == 0:
                log(f"  L{L} profile only {info['shape']}")

    # alignment: hist W1 vs L_first, row/col rms cosines
    Lref = spec["layers"][0]
    hist_w1 = []
    for L in spec["layers"]:
        hist_w1.append(
            {
                "layer": L,
                "w1_u_vs_Lref": w1_from_hist(hists[L], hists[Lref], -1.0, 1.0),
            }
        )

    # pairs for profiles
    if spec["family"] == "mlp":
        adj_pairs = ADJ_PAIRS_MLP
        d16_pairs = D16_PAIRS_MLP
        all_adj = [(i, i + 1) for i in range(63)]
    elif spec["family"] == "gqa":
        adj_pairs = GQA_ADJ_PROBE
        d16_pairs = GQA_D16
        all_adj = list(zip(spec["layers"][:-1], spec["layers"][1:]))
    else:
        adj_pairs = DN_ADJ_PROBE
        d16_pairs = DN_D16
        all_adj = [(spec["layers"][i], spec["layers"][i + 1]) for i in range(len(spec["layers"]) - 1)]
        # only true layer-adjacent among DN (skip GQA holes): keep those with j-i==1
        all_adj = [(i, j) for (i, j) in all_adj if j - i == 1]

    row_adj = pairwise_profile_stats(row_rms, all_adj)
    col_adj = pairwise_profile_stats(col_rms, all_adj)
    row_d16 = pairwise_profile_stats(row_rms, d16_pairs)
    col_d16 = pairwise_profile_stats(col_rms, d16_pairs)

    def mean_cos(pairs, key="cosine"):
        if not pairs:
            return None
        return float(np.mean([p[key] for p in pairs]))

    # shared subspace of profiles
    row_svd = stack_energy(row_rms)
    col_svd = stack_energy(col_rms)

    # pooled levels from all probe fit samples
    shared_levels = {}
    shared_score = {2: {}, 3: {}, 4: {}}
    xref_score = {2: {}, 3: {}, 4: {}}  # apply Lref private to others
    pooled = {}
    for bits in (2, 3, 4):
        cat = np.concatenate([sample_u[L] for L in spec["probe"] if L in sample_u])
        rng = np.random.default_rng(99)
        if cat.size > 800_000:
            cat = cat[rng.choice(cat.size, 800_000, replace=False)]
        pooled[bits] = lloyd_1d(cat, 1 << bits)
        shared_levels[bits] = pooled[bits]
        L0p = spec["probe"][0]
        lv_ref = private_levels[bits][L0p]
        # score on probe tensors: reload
        for L in spec["probe"]:
            info = table[tensor_name(L, spec["suffix"])]
            W = load_f32(info)
            r_s, c_s, _ = apply_levels_chunked(W, pooled[bits])
            r_x, c_x, _ = apply_levels_chunked(W, lv_ref)
            shared_score[bits][L] = {"rel_l2": r_s, "cosine": c_s}
            xref_score[bits][L] = {"rel_l2": r_x, "cosine": c_x}
            # WX shared4 on first/last
            if (
                bits == 4
                and W.shape[1] == HIDDEN
                and L in (spec["probe"][0], spec["probe"][-1])
                and L in X_by_layer
            ):
                X = X_by_layer[L][192:]
                What = reconstruct_with_levels(W, pooled[bits])
                rec = matvec_cos(W, What, X)
                rec.update({"layer": L, "class": name, "rule": "shared4_pooled", "bits": 4})
                wx_scores.append(rec)
                del What
            del W
            gc.collect()

    # level-vector agreement
    level_l2 = {}
    for bits in (2, 3, 4):
        L0p = spec["probe"][0]
        a = private_levels[bits][L0p].astype(np.float64)
        ds = []
        for L, lv in private_levels[bits].items():
            ds.append({"layer": L, "l2": float(np.linalg.norm(lv.astype(np.float64) - a))})
        level_l2[bits] = ds

    # patches: jaccard + kmeans transfer
    jacc_adj = []
    for i, j in adj_pairs:
        if i in patch_sets and j in patch_sets:
            a, b = patch_sets[i], patch_sets[j]
            u = len(a | b)
            jacc_adj.append(
                {
                    "i": i,
                    "j": j,
                    "jaccard": (len(a & b) / u) if u else 0.0,
                    "n_i": len(a),
                    "n_j": len(b),
                    "n_inter": len(a & b),
                }
            )
    jacc_d16 = []
    for i, j in d16_pairs:
        if i in patch_sets and j in patch_sets:
            a, b = patch_sets[i], patch_sets[j]
            u = len(a | b)
            jacc_d16.append(
                {
                    "i": i,
                    "j": j,
                    "jaccard": (len(a | b) and len(a & b) / u) or 0.0,
                    "n_i": len(a),
                    "n_j": len(b),
                    "n_inter": len(a & b),
                }
            )
    # unique rate vs gaussian
    uniq = {L: len(s) / 16384 for L, s in patch_sets.items()}

    kmeans_xfer = {}
    L0p = spec["probe"][0]
    Llast = spec["probe"][-1]
    if L0p in patch_templates and Llast in patch_templates:
        for K in (64, 256, 1024):
            C = kmeans_fit(patch_templates[L0p], K, iters=8, seed=7)
            self_s = kmeans_assign_relmse(patch_templates[L0p], C)
            xfer_s = kmeans_assign_relmse(patch_templates[Llast], C)
            Cpriv = kmeans_fit(patch_templates[Llast], K, iters=8, seed=8)
            priv_s = kmeans_assign_relmse(patch_templates[Llast], Cpriv)
            kmeans_xfer[K] = {
                "fit_layer": L0p,
                "eval_layer": Llast,
                "self": self_s,
                "transfer": xfer_s,
                "private_on_eval": priv_s,
                "transfer_over_private": float(xfer_s["rel_l2"] / priv_s["rel_l2"])
                if priv_s["rel_l2"]
                else None,
            }
            log(
                f"  kmeans K={K} self={self_s['rel_l2']:.4f} "
                f"xfer L{L0p}->L{Llast}={xfer_s['rel_l2']:.4f} "
                f"priv={priv_s['rel_l2']:.4f}"
            )

    # gaussian null patches of same d
    rng = np.random.default_rng(12345)
    g_shape = shapes[L0p]
    # small gaussian matrix
    Gmat = rng.normal(0.0, 0.01, size=(min(512, g_shape[0]), g_shape[1])).astype(np.float32)
    gt, _ = sample_unit_patches(Gmat, d=16, n_keep=16384, rng=rng)
    gset = set(hash_templates(gt).tolist())
    gauss_unique = len(gset) / 16384
    Cg = kmeans_fit(gt, 256, iters=6, seed=1)
    gauss_km = kmeans_assign_relmse(gt, Cg)

    # positional codes on a few pairs
    pos = {"adj": [], "d16": []}
    for tag, pairs in (("adj", adj_pairs[:3]), ("d16", d16_pairs[:3])):
        for i, j in pairs:
            ki = tensor_name(i, spec["suffix"])
            kj = tensor_name(j, spec["suffix"])
            if ki not in table or kj not in table:
                continue
            rec = positional_pair(table[ki], table[kj], bits=4)
            rec.update({"i": i, "j": j, "d": j - i})
            pos[tag].append(rec)
            log(
                f"  pos {tag} L{i}-L{j} code_eq={rec['code_equal']:.4f} "
                f"shuf={rec['code_equal_shuffled']:.4f} "
                f"sign={rec['sign_agree_nonzero']:.4f} "
                f"wcos={rec['weight_cosine']:.5f} "
                f"scos={rec['scale_cosine']:.4f}"
            )

    n_weights_class = int(np.prod(shapes[spec["layers"][0]])) * len(spec["layers"])
    bpw_levels = {
        bits: bits_complete_shared_levels(
            n_weights_class, len(spec["layers"]), bits, G64, 1 << bits
        )
        for bits in (2, 3, 4)
    }
    bpw_dict = {
        f"d16_K{K}": bits_complete_dict(n_weights_class, 16, K) for K in (64, 256, 1024)
    }

    def summarize_score(d):
        rels = [v["rel_l2"] for v in d.values()]
        return {
            "n": len(rels),
            "rel_l2_mean": float(np.mean(rels)),
            "rel_l2_min": float(np.min(rels)),
            "rel_l2_max": float(np.max(rels)),
            "per_layer": {str(k): v for k, v in d.items()},
        }

    report["classes"][name] = {
        "family": spec["family"],
        "n_layers": len(spec["layers"]),
        "probe": spec["probe"],
        "shape": list(shapes[spec["layers"][0]]),
        "n_weights_class": n_weights_class,
        "hist_w1_vs_Lref": hist_w1,
        "hist_w1_mean": float(np.mean([h["w1_u_vs_Lref"] for h in hist_w1])),
        "hist_w1_max": float(np.max([h["w1_u_vs_Lref"] for h in hist_w1])),
        "row_rms_adj_cos_mean": mean_cos(row_adj),
        "row_rms_adj_cos_centered_mean": mean_cos(row_adj, "cosine_centered"),
        "col_rms_adj_cos_mean": mean_cos(col_adj),
        "col_rms_adj_cos_centered_mean": mean_cos(col_adj, "cosine_centered"),
        "row_rms_d16_cos_mean": mean_cos(row_d16),
        "row_rms_d16_cos_centered_mean": mean_cos(row_d16, "cosine_centered"),
        "col_rms_d16_cos_mean": mean_cos(col_d16),
        "col_rms_d16_cos_centered_mean": mean_cos(col_d16, "cosine_centered"),
        "row_rms_adj": row_adj,
        "col_rms_adj": col_adj,
        "row_rms_d16": row_d16,
        "col_rms_d16": col_d16,
        "row_profile_svd": row_svd,
        "col_profile_svd": col_svd,
        "private_levels": {
            str(b): {str(L): lv.tolist() for L, lv in private_levels[b].items()}
            for b in (2, 3, 4)
        },
        "pooled_levels": {str(b): pooled[b].tolist() for b in (2, 3, 4)},
        "level_l2_vs_Lref": level_l2,
        "private_score": {str(b): summarize_score(private_score[b]) for b in (2, 3, 4)},
        "uniform_score": {str(b): summarize_score(uniform_score[b]) for b in (2, 3, 4)},
        "shared_pooled_score": {str(b): summarize_score(shared_score[b]) for b in (2, 3, 4)},
        "shared_Lref_score": {str(b): summarize_score(xref_score[b]) for b in (2, 3, 4)},
        "wx": wx_scores,
        "patch_unique_frac": uniq,
        "patch_jaccard_adj": jacc_adj,
        "patch_jaccard_d16": jacc_d16,
        "patch_kmeans": kmeans_xfer,
        "gauss_patch_unique_frac": gauss_unique,
        "gauss_kmeans256": gauss_km,
        "positional": pos,
        "bpw_shared_levels": bpw_levels,
        "bpw_dict": bpw_dict,
    }
    # drop heavy arrays from this function scope
    del row_rms, col_rms, patch_templates
    gc.collect()
    dump(report)
    log(f"=== done {name} rss={rss_gb():.2f} ===")


def main():
    LOG.write_text("")
    log("start")
    table = parse_all_headers(ROOT)
    log(f"headers {len(table)}")
    cfg = json.loads((ROOT / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    identity = {
        "config_model_type": cfg.get("model_type"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "hidden_size": text.get("hidden_size"),
        "intermediate_size": text.get("intermediate_size"),
        "num_experts": text.get("num_experts"),
        "l0_gate": list(table[tensor_name(0, "mlp.gate_proj.weight")]["shape"]),
        "l0_gate_dtype": table[tensor_name(0, "mlp.gate_proj.weight")]["dtype"],
        "n_header_tensors": len(table),
        "bf16_root": str(ROOT),
        "act_root": str(ACT),
        "script": __file__,
    }
    log(f"identity {identity}")

    report = {
        "schema": "hawking.g1.cross_layer_structure.v1",
        "identity": identity,
        "constructions": [
            "shared_quant_levels_codes_stay_per_layer",
            "shared_row_col_rms_profiles",
            "positional_q4_code_delta",
            "shared_unit_patch_dictionary",
            "shared_block_templates_hash",
            "within_layer_head_levels_and_templates",
            "embed_vs_lm_head",
            "activation_stream_gravity",
        ],
        "not_retried": [
            "flattened_weight_shared_svd_basis — already FALSIFIED in g1-shared-basis.md",
            "generator_plus_residual_as_constructed — already FALSIFIED in g1-generator-residual.md",
        ],
        "classes": {},
        "gravity": None,
        "embed_lm_head": None,
        "heads": {},
        "rss_max_gb": None,
        "elapsed_s": None,
    }
    dump(report)

    log("activation gravity")
    report["gravity"] = activation_gravity()
    dump(report)
    g = report["gravity"]
    log(
        f"gravity L63/L00 token_l2={g['L63_over_L00_mean_token_l2']:.4f} "
        f"adj_cos={g['adjacent_hidden_cosine_mean']:.4f} "
        f"k90={g['stacked_k90']}"
    )

    log("embed vs lm_head")
    report["embed_lm_head"] = embed_lm_head_alignment(table)
    dump(report)
    log(f"embed-lm_head {report['embed_lm_head']}")

    # load needed activations (64 * 5MB = 320MB — keep only probe layers)
    need_X = set(PROBE_MLP + PROBE_GQA + PROBE_DN)
    X_by_layer = {L: load_hidden(L) for L in sorted(need_X)}
    log(f"loaded {len(X_by_layer)} hidden captures")

    for spec in CLASSES:
        process_class(table, spec, X_by_layer, report)

    log("heads GQA q L3/L31/L63")
    report["heads"]["gqa_q"] = {}
    for L in (3, 31, 63):
        info = table[tensor_name(L, "self_attn.q_proj.weight")]
        report["heads"]["gqa_q"][str(L)] = head_structure_gqa_q(info, L)
        dump(report)
        log(f"  gqa q L{L} {report['heads']['gqa_q'][str(L)]['q_weight_cos']}")

    log("heads DN qkv L0/L32/L62")
    report["heads"]["dn_qkv"] = {}
    for L in (0, 32, 62):
        info = table[tensor_name(L, "linear_attn.in_proj_qkv.weight")]
        report["heads"]["dn_qkv"][str(L)] = head_structure_dn_qkv(info, L)
        dump(report)
        log(f"  dn qkv L{L} qcos={report['heads']['dn_qkv'][str(L)]['q_weight_cos']['mean']}")

    report["rss_max_gb"] = rss_gb()
    report["elapsed_s"] = time.time() - T0
    dump(report)
    log(f"DONE elapsed={report['elapsed_s']:.1f}s rss_max={report['rss_max_gb']:.3f}GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

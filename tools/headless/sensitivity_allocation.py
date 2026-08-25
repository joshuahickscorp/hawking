#!/usr/bin/env python3
"""N051 SENSITIVITY_ALLOCATION: value-of-information map for N041 (S026 §26, §3).

Activation-aware second-order PROXY on real captured activations, streamed from
the parent BF16 tensors on CPU. Ranks organs / layers / channels; classifies
information disposable / cheap / ordinary / sensitive / critical; emits a
marginal capability-gain-per-bit curve at the SAME total bits as the N041
2.5969 complete-EBPW allocation so a later allocator can spend bits where they
matter and starve where they do not.

This is a PROXY. It RANKS. It does NOT certify composition, and it does NOT
claim a new whole-model floor. The 2.5969 number stays N041's.

    python3 tools/headless/sensitivity_allocation.py
    python3 -m pytest tools/headless -q

Pure CPU. No GPU, no cargo, no Metal, no second 27B decode. Parent tensors are
read-only, one at a time. Does not mutate NOETIC_PARENT_A.
"""
from __future__ import annotations

import gc
import json
import math
import os
import struct
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from organ_frontiers import (  # noqa: E402
    CAPTURE_CANDIDATES,
    HARDCODED_FAMILIES,
    HIDDEN,
    PARENT_CANDIDATES,
    VOCAB,
    find_capture,
    find_parent,
    find_tokenizer,
    git_head,
    load_X,
    load_tensor,
    reconstruct_token_ids,
    split_from_manifest,
    tensor_name,
    weight_index,
)

SCHEMA = "hawking.headless.sensitivity_allocation.v1"
RECEIPT = REPO / "receipts" / "headless" / "SENSITIVITY_ALLOCATION.json"
GENERATOR = "tools/headless/sensitivity_allocation.py"
OBLIGATION = (
    "N051 — SENSITIVITY_ALLOCATION (S026 §26, §3; DOC-REPRESENTATION; CPU). "
    "Value of information per organ / layer / channel via an activation-aware "
    "second-order proxy (diagonal Fisher / activation-weighted magnitude on "
    "real activations, streamed). Classify disposable/cheap/ordinary/sensitive/"
    "critical. Produce the marginal capability-gain-per-bit curve N041 consumes "
    "to move below uniform per-organ floors, refining the 2.5969 complete-EBPW "
    "allocation. Sensitivity is a proxy: it ranks, it does not certify."
)
VISION_PY = Path.home() / ".grok-vision" / "bin" / "python"

SOURCE_PARAM_COUNT = 26_895_998_464
LAYERS = 64
INTERMEDIATE = 17_408
N_SUB = 256
ROW_CHUNK = 1024
SEED = 20260824
LAPLACE = 1.0
MIN_REGION_PARAMS = 1_000_000
TOP_K = 8
LEFTOVER_BPW = 32.0  # N040 leftover f32 packing; cited, not re-derived

CLASSES = ("disposable", "cheap", "ordinary", "sensitive", "critical")
CLASS_RANKS = (0.05, 0.25, 0.75, 0.95)
BIT_LEVELS = (1.25, 1.85, 2.25, 3.125, 3.25, 4.125, 4.25)

# N041 uniform-per-organ floors. CITED, not re-derived. Complete EBPW 2.5969.
N041_FLOOR = {
    "mlp": 2.25,
    "deltanet": 3.2614879737507687,
    "gqa": 3.126550820024315,
    "embedding": 3.125,
    "output": 3.125,
}
N041_COMPLETE_EBPW = 2.596888
N041_RECEIPT = "receipts/headless/WHOLE_MODEL_RECOMPOSE.json"
N040_RECEIPT = "receipts/headless/ORGAN_DENSITY_FLOORS.json"

# Residual-injection depth weights. CITED from global_allocator / gravity_error_chain.
Q_INJECT = {
    0: 1.597e-04,
    7: 1.910e-04,
    15: 2.715e-04,
    23: 4.281e-04,
    31: 4.634e-04,
    39: 5.279e-04,
    47: 6.534e-04,
    55: 9.065e-04,
    63: 2.577e-03,
}
_QL = sorted(Q_INJECT)

NEGATIVE_SCIENCE = (
    "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
    "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json",
    "receipts/headless/BINARY_HEALING.json",
    "receipts/headless/HYBRID_OPERATOR.json",
    "receipts/headless/ONEBIT_FAMILIES.json",
    "receipts/headless/SHARED_BASIS_COHERENT.json",
    "receipts/headless/ORGAN_DENSITY_FLOORS.json",
    "receipts/headless/ORGAN_FRONTIERS.json",
)

GQA_LAYERS = tuple(range(3, LAYERS, 4))
DN_VPK = 3
DN_K_HEADS = 16
DN_K_DIM = 128
DN_V_DIM = 128
GQA_HEADS = 24
GQA_KV_HEADS = 4
GQA_HEAD_DIM = 256


# ---------------------------------------------------------------------------
# import-safe arithmetic (no torch, no 27B, no GPU)
# ---------------------------------------------------------------------------

def q_inject(layer: int | None) -> float:
    if layer is None:
        return Q_INJECT[0]
    if layer <= _QL[0]:
        return Q_INJECT[_QL[0]]
    if layer >= _QL[-1]:
        return Q_INJECT[_QL[-1]]
    lo = max(l for l in _QL if l <= layer)
    hi = min(l for l in _QL if l >= layer)
    if lo == hi:
        return Q_INJECT[lo]
    t = (layer - lo) / (hi - lo)
    return Q_INJECT[lo] * (1.0 - t) + Q_INJECT[hi] * t


def q_mult(layer: int | None) -> float:
    """Residual-sensitivity relative to L0. Embed uses L0, lm_head L63."""
    return q_inject(layer) / q_inject(0)


def fisher_diag(X) -> "object":
    """Diagonal Fisher / Hessian proxy for a linear map Y = X W^T: E[x_j^2]."""
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    return np.mean(np.square(X, dtype=np.float64), axis=0)


def remaining_mse(voi: float, bpw: float) -> float:
    """RTN grouped-quant MSE proxy: VoI / 4^{bpw}. Does not certify composition."""
    return float(voi) / (4.0 ** float(bpw))


def mse_drop(voi: float, bpw_from: float, bpw_to: float) -> float:
    return remaining_mse(voi, bpw_from) - remaining_mse(voi, bpw_to)


def pearson(x, y) -> float:
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if den <= 0.0:
        return 0.0
    return float(np.dot(x, y) / den)


def spearman(x, y) -> float:
    import numpy as np

    def rank(a):
        return np.argsort(np.argsort(np.asarray(a), kind="mergesort"))

    return pearson(rank(x), rank(y))


def class_cuts(values) -> tuple[float, float, float, float]:
    import numpy as np

    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    q = np.quantile(v, CLASS_RANKS)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def classify(value: float, cuts: tuple[float, float, float, float]) -> str:
    if value <= cuts[0]:
        return "disposable"
    if value <= cuts[1]:
        return "cheap"
    if value <= cuts[2]:
        return "ordinary"
    if value <= cuts[3]:
        return "sensitive"
    return "critical"


def classify_array(values, cuts: tuple[float, float, float, float]):
    import numpy as np

    v = np.asarray(values, dtype=np.float64)
    out = np.empty(v.shape, dtype=object)
    out[v <= cuts[0]] = "disposable"
    out[(v > cuts[0]) & (v <= cuts[1])] = "cheap"
    out[(v > cuts[1]) & (v <= cuts[2])] = "ordinary"
    out[(v > cuts[2]) & (v <= cuts[3])] = "sensitive"
    out[v > cuts[3]] = "critical"
    return out


def score_linear(W, d) -> dict:
    """Second-order VoI for Y = X W^T, F_jj = d_j = E[x_j^2].

    VoI_ij = 0.5 * d_j * W_ij^2. Local (un-depth-weighted).
    """
    import numpy as np

    W = np.asarray(W, dtype=np.float32)
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    if W.ndim != 2:
        raise ValueError(f"expected 2D W, got {W.shape}")
    n_out, n_in = W.shape
    if d.shape[0] != n_in:
        raise ValueError(f"d length {d.shape[0]} != n_in {n_in}")
    voi_out = np.empty(n_out, dtype=np.float64)
    voi_in = np.zeros(n_in, dtype=np.float64)
    row_fro2 = np.empty(n_out, dtype=np.float64)
    w_fro2 = 0.0
    step = 512
    for i0 in range(0, n_out, step):
        sl = np.asarray(W[i0 : i0 + step], dtype=np.float32)
        w2 = np.square(sl, dtype=np.float64)
        row_fro2[i0 : i0 + sl.shape[0]] = w2.sum(axis=1)
        voi_out[i0 : i0 + sl.shape[0]] = 0.5 * (w2 * d).sum(axis=1)
        voi_in += 0.5 * d * w2.sum(axis=0)
        w_fro2 += float(w2.sum())
    local_voi = float(voi_out.sum())
    return {
        "n_out": int(n_out),
        "n_in": int(n_in),
        "n_params": int(n_out * n_in),
        "local_voi": local_voi,
        "voi_out": voi_out,
        "voi_in": voi_in,
        "row_fro2": row_fro2,
        "w_fro2": w_fro2,
        "weight_only_voi": 0.5 * float(d.mean()) * w_fro2,
        "d_mean": float(d.mean()),
        "d_min": float(d.min()),
        "d_max": float(d.max()),
        "d_participation": _participation(d),
        "spearman_channel_vs_weight": spearman(voi_out, row_fro2) if n_out >= 4 else None,
    }


def score_embed_rows(row_iter, counts, n_rows: int, n_cols: int) -> dict:
    """Gather embedding: VoI_token = 0.5 * p(token) * ||row||^2."""
    import numpy as np

    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    if counts.shape[0] != n_rows:
        raise ValueError("token count length != vocab rows")
    n = float(counts.sum())
    p_token = (counts + LAPLACE) / (n + LAPLACE * n_rows)
    voi_out = np.empty(n_rows, dtype=np.float64)
    row_fro2 = np.empty(n_rows, dtype=np.float64)
    w_fro2 = 0.0
    for i0, rows in row_iter:
        sl = np.asarray(rows, dtype=np.float32)
        w2 = np.square(sl, dtype=np.float64).sum(axis=1)
        voi_out[i0 : i0 + sl.shape[0]] = 0.5 * p_token[i0 : i0 + sl.shape[0]] * w2
        row_fro2[i0 : i0 + sl.shape[0]] = w2
        w_fro2 += float(w2.sum())
    local_voi = float(voi_out.sum())
    return {
        "n_out": int(n_rows),
        "n_in": int(n_cols),
        "n_params": int(n_rows * n_cols),
        "local_voi": local_voi,
        "voi_out": voi_out,
        "voi_in": None,
        "row_fro2": row_fro2,
        "counts": counts,
        "w_fro2": w_fro2,
        "weight_only_voi": 0.5 * float(p_token.mean()) * w_fro2,
        "d_mean": float(p_token.mean()),
        "d_min": float(p_token.min()),
        "d_max": float(p_token.max()),
        "d_participation": _participation(p_token),
    }


def _slice_channels(scored: dict, mask, *, d_mean: float | None = None) -> dict:
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    voi_out = np.asarray(scored["voi_out"])[mask]
    n_out = int(mask.sum())
    n_in = int(scored["n_in"])
    row_fro2 = np.asarray(scored["row_fro2"])[mask] if "row_fro2" in scored else None
    if row_fro2 is not None:
        w_fro2 = float(row_fro2.sum())
        rho_ch = spearman(voi_out, row_fro2) if n_out >= 4 else None
    else:
        w_fro2 = float(scored["w_fro2"]) * (n_out / max(int(scored["n_out"]), 1))
        rho_ch = None
    dm = float(scored["d_mean"] if d_mean is None else d_mean)
    return {
        "n_out": n_out,
        "n_in": n_in,
        "n_params": n_out * n_in,
        "local_voi": float(voi_out.sum()),
        "voi_out": voi_out,
        "voi_in": None,
        "w_fro2": w_fro2,
        "weight_only_voi": 0.5 * dm * w_fro2,
        "d_mean": dm,
        "d_min": float(scored["d_min"]),
        "d_max": float(scored["d_max"]),
        "d_participation": float(scored["d_participation"]),
        "spearman_channel_vs_weight": rho_ch,
    }


def score_vector(w, d=None) -> dict:
    import numpy as np

    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if d is None:
        d = np.ones_like(w)
    else:
        d = np.asarray(d, dtype=np.float64).reshape(-1)
        if d.shape[0] != w.shape[0]:
            d = np.ones_like(w)
    voi_out = 0.5 * d * (w * w)
    w_fro2 = float((w * w).sum())
    return {
        "n_out": int(w.size),
        "n_in": 1,
        "n_params": int(w.size),
        "local_voi": float(voi_out.sum()),
        "voi_out": voi_out,
        "voi_in": None,
        "w_fro2": w_fro2,
        "weight_only_voi": 0.5 * w_fro2,
        "d_mean": float(d.mean()),
        "d_min": float(d.min()),
        "d_max": float(d.max()),
        "d_participation": _participation(d),
    }


def _participation(d) -> float:
    import numpy as np

    x = np.asarray(d, dtype=np.float64)
    s = float((x * x).sum())
    if s <= 0.0:
        return 0.0
    return float((x.sum() ** 2) / s / max(x.size, 1))


def snap_level(bpw: float, levels: tuple[float, ...] = BIT_LEVELS) -> float:
    return min(levels, key=lambda x: abs(x - float(bpw)))


def greedy_fill(items: list[dict], budget_bits: float, levels: tuple[float, ...] = BIT_LEVELS) -> dict:
    """Spend a fixed bit budget on highest-marginal items, starting at levels[0].

    remaining_mse = voi / 4^{bpw}. Equal total bits vs a uniform allocation is
    the caller's job (pass that budget). Discrete levels; leftover bits are
    applied as a fractional last step so the budget is hit.
    """
    if not items:
        return {"assignment": [], "bits": 0.0, "steps": 0, "hit_budget": True, "curve": []}
    n = len(items)
    idx = [0] * n
    n_p = [float(it["n_params"]) for it in items]
    voi = [float(it["voi"]) for it in items]
    bits = sum(n_p[i] * levels[0] for i in range(n))
    curve = [
        {
            "bits": bits,
            "remaining_mse_proxy": sum(remaining_mse(voi[i], levels[0]) for i in range(n)),
            "step": 0,
        }
    ]
    steps = 0
    last_record = 0

    def marg(i: int) -> tuple[float, float]:
        k = idx[i]
        if k + 1 >= len(levels):
            return (-1.0, 0.0)
        b0, b1 = levels[k], levels[k + 1]
        dbits = n_p[i] * (b1 - b0)
        if dbits <= 0.0:
            return (-1.0, 0.0)
        drop = mse_drop(voi[i], b0, b1)
        return (drop / dbits, dbits)

    guard = 0
    max_steps = n * len(levels) + 4
    while guard < max_steps:
        guard += 1
        best_i = -1
        best_m = 0.0
        best_cost = 0.0
        for i in range(n):
            m, cost = marg(i)
            if cost <= 0.0 or bits + cost > budget_bits + 1e-3:
                continue
            if m > best_m:
                best_m, best_i, best_cost = m, i, cost
        if best_i < 0:
            break
        idx[best_i] += 1
        bits += best_cost
        steps += 1
        if steps - last_record >= 40 or bits >= budget_bits - 1.0:
            curve.append(
                {
                    "bits": bits,
                    "remaining_mse_proxy": sum(
                        remaining_mse(voi[i], levels[idx[i]]) for i in range(n)
                    ),
                    "step": steps,
                }
            )
            last_record = steps

    # Spend leftover bits as a fractional promotion on the current best item.
    remaining = budget_bits - bits
    if remaining > 1.0:
        best_i = -1
        best_m = 0.0
        for i in range(n):
            m, cost = marg(i)
            if cost <= 0.0:
                continue
            if m > best_m:
                best_m, best_i = m, i
        if best_i >= 0:
            k = idx[best_i]
            b0, b1 = levels[k], levels[k + 1]
            span = n_p[best_i] * (b1 - b0)
            frac = min(1.0, remaining / span) if span > 0 else 0.0
            bpw = b0 + frac * (b1 - b0)
            bits += n_p[best_i] * (bpw - b0)
            assignment_extra = {best_i: bpw}
        else:
            assignment_extra = {}
    else:
        assignment_extra = {}

    assignment = []
    for i, it in enumerate(items):
        bpw = assignment_extra[i] if i in assignment_extra else float(levels[idx[i]])
        rec = dict(it)
        rec["recommended_bpw"] = bpw
        rec["recommended_bits"] = float(it["n_params"]) * bpw
        rec["remaining_mse_proxy"] = remaining_mse(float(it["voi"]), bpw)
        rec["uniform_remaining_mse_proxy"] = remaining_mse(
            float(it["voi"]), float(it["uniform_bpw"])
        )
        assignment.append(rec)
    mse_end = sum(a["remaining_mse_proxy"] for a in assignment)
    curve.append({"bits": bits, "remaining_mse_proxy": mse_end, "step": steps + 1})
    return {
        "assignment": assignment,
        "bits": bits,
        "steps": steps,
        "hit_budget": abs(bits - budget_bits) / max(budget_bits, 1.0) < 1e-4
        or bits <= budget_bits + 1.0,
        "curve": curve,
        "budget_bits": budget_bits,
    }


def take_sub(X, idx, n: int = N_SUB, seed: int = SEED):
    import numpy as np

    idx = np.asarray(idx)
    if idx.size == 0:
        raise ValueError("empty subsample index")
    rng = np.random.RandomState(seed)
    if idx.size > n:
        pick = np.sort(rng.choice(idx, n, replace=False))
    else:
        pick = idx
    return X[pick]


def silu(x):
    import numpy as np

    x = np.clip(np.asarray(x, dtype=np.float32), -40.0, 40.0)
    return x * (1.0 / (1.0 + np.exp(-x)))


def sigmoid(x):
    import numpy as np

    x = np.clip(np.asarray(x, dtype=np.float32), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def gemm_xt(X, W):
    import numpy as np

    a = np.ascontiguousarray(X, dtype=np.float32)
    b = np.ascontiguousarray(W, dtype=np.float32)
    return a @ b.T


def gqa_out_from_y(Yq, Yv):
    import numpy as np

    qg = np.asarray(Yq, dtype=np.float32).reshape(-1, GQA_HEADS, 2, GQA_HEAD_DIM)
    gate = sigmoid(qg[:, :, 1, :])
    v = np.asarray(Yv, dtype=np.float32).reshape(-1, GQA_KV_HEADS, GQA_HEAD_DIM)
    v_rep = np.repeat(v, GQA_HEADS // GQA_KV_HEADS, axis=1)
    return np.ascontiguousarray((v_rep * gate).reshape(Yq.shape[0], GQA_HEADS * GQA_HEAD_DIM))


def dn_out_from_y(Yqkv, Yz):
    import numpy as np

    v = np.asarray(Yqkv, dtype=np.float32)[:, 4096:10240]
    z = np.asarray(Yz, dtype=np.float32)
    return np.ascontiguousarray(v * silu(z))


def j(x):
    if isinstance(x, dict):
        return {k: j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [j(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, (int, str, bool)) or x is None:
        return x
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            if x.size > 64:
                return {
                    "n": int(x.size),
                    "mean": float(np.mean(x)),
                    "p50": float(np.quantile(x, 0.5)),
                    "p95": float(np.quantile(x, 0.95)),
                }
            return x.tolist()
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
    except Exception:
        pass
    return str(x)


def citation_exists(rel: str) -> bool:
    p = REPO / rel
    if p.is_file():
        return True
    r = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-e", f"HEAD:{rel}"],
        capture_output=True,
    )
    return r.returncode == 0


# ---------------------------------------------------------------------------
# safetensors streaming (embed / lm_head only; others fit in RAM as f32)
# ---------------------------------------------------------------------------

_HEADER_CACHE: dict[str, dict] = {}


def _shard_header(parent: Path, shard_name: str) -> tuple[dict, int]:
    key = str(parent / shard_name)
    if key in _HEADER_CACHE:
        return _HEADER_CACHE[key]
    path = parent / shard_name
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    rec = (header, 8 + n)
    _HEADER_CACHE[key] = rec
    return rec


def tensor_meta(parent: Path, name: str) -> dict:
    shard = weight_index(parent)[name]
    header, data_start = _shard_header(parent, shard)
    meta = header[name]
    return {
        "name": name,
        "shard": shard,
        "dtype": meta["dtype"],
        "shape": tuple(meta["shape"]),
        "data_offsets": tuple(meta["data_offsets"]),
        "data_start": data_start,
        "path": parent / shard,
    }


def iter_matrix_rows(parent: Path, name: str, row_chunk: int = ROW_CHUNK):
    import numpy as np

    meta = tensor_meta(parent, name)
    shape = meta["shape"]
    if len(shape) != 2:
        raise ValueError(f"{name} not 2D: {shape}")
    rows, cols = int(shape[0]), int(shape[1])
    dtype = meta["dtype"]
    if dtype == "BF16":
        es = 2
    elif dtype == "F16":
        es = 2
    elif dtype == "F32":
        es = 4
    else:
        raise ValueError(f"{name} dtype {dtype}")
    base = meta["data_start"] + meta["data_offsets"][0]
    with open(meta["path"], "rb") as f:
        for i0 in range(0, rows, row_chunk):
            i1 = min(rows, i0 + row_chunk)
            f.seek(base + i0 * cols * es)
            raw = f.read((i1 - i0) * cols * es)
            if dtype == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16)
                f32 = (u16.astype(np.uint32) << 16).view(np.float32)
                arr = np.array(f32.reshape(i1 - i0, cols), dtype=np.float32, copy=True)
            elif dtype == "F16":
                arr = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(i1 - i0, cols)
            else:
                arr = np.frombuffer(raw, dtype="<f4").copy().reshape(i1 - i0, cols)
            yield i0, arr


def layers_to_score() -> tuple[int, ...]:
    env = os.environ.get("N051_LAYERS", "").strip()
    if env:
        return tuple(int(x) for x in env.split(",") if x.strip() != "")
    return tuple(range(LAYERS))


# ---------------------------------------------------------------------------
# tokenizer / token prior (embed)
# ---------------------------------------------------------------------------

class _IdsTok:
    def __init__(self, table: dict[str, list[int]]):
        self.table = table

    def encode(self, text: str):
        class _R:
            def __init__(self, ids):
                self.ids = ids

        if text not in self.table:
            return _R([])
        return _R(self.table[text])


def encode_prompts(tok_path: Path, texts: list[str]) -> list[list[int]] | None:
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tok_path))
        return [tok.encode(t).ids for t in texts]
    except Exception:
        pass
    if VISION_PY.is_file():
        code = (
            "import json,sys\n"
            "from tokenizers import Tokenizer\n"
            "tok=Tokenizer.from_file(sys.argv[1])\n"
            "print(json.dumps([tok.encode(t).ids for t in json.load(sys.stdin)]))\n"
        )
        r = subprocess.run(
            [str(VISION_PY), "-c", code, str(tok_path)],
            input=json.dumps(texts).encode(),
            capture_output=True,
            timeout=180,
        )
        if r.returncode == 0:
            return json.loads(r.stdout.decode())
    return None


def token_prior(parent: Path, cap: Path, manifest: dict) -> dict:
    import numpy as np

    tok_path = find_tokenizer() or (parent / "tokenizer.json")
    texts = []
    for fam, prompts in HARDCODED_FAMILIES.items():
        texts.extend(prompts)
    encoded = encode_prompts(tok_path, texts) if tok_path.is_file() else None
    counts = np.zeros(VOCAB, dtype=np.float64)
    info = {
        "status": "UNIFORM_PRIOR",
        "tokenizer": str(tok_path) if tok_path.is_file() else None,
        "n_ids": 0,
        "n_unique": 0,
        "aligned_families": [],
        "failed_families": [],
        "note": "no reconstructed token ids; embed uses uniform Laplace over the vocab",
    }
    if encoded is None:
        p = (counts + LAPLACE) / (counts.sum() + LAPLACE * VOCAB)
        info["p_unseen"] = float(p[0])
        return {"p": p, "counts": counts, "info": info}
    table: dict[str, list[int]] = {}
    k = 0
    for fam, prompts in HARDCODED_FAMILIES.items():
        for prompt in prompts:
            table[prompt] = encoded[k]
            k += 1
    rec = reconstruct_token_ids(_IdsTok(table), manifest)
    ids = list(rec.get("fit_ids") or []) + list(rec.get("hold_ids") or [])
    for t in ids:
        ti = int(t)
        if 0 <= ti < VOCAB:
            counts[ti] += 1.0
    n = float(counts.sum())
    p = (counts + LAPLACE) / (n + LAPLACE * VOCAB)
    info = {
        "status": "MEASURED" if n > 0 else "UNIFORM_PRIOR",
        "tokenizer": str(tok_path),
        "n_ids": int(n),
        "n_unique": int((counts > 0).sum()),
        "aligned_families": rec.get("aligned_families") or [],
        "failed_families": rec.get("failed_families") or [],
        "vocab": VOCAB,
        "laplace": LAPLACE,
        "note": (
            "token frequencies from capture_diverse2 reconstructed prompt ids "
            "(HARDCODED_FAMILIES + parent tokenizer.json). Laplace +1. Unseen "
            "rows are LOCKED at the N041 embed floor (S026 §109: the capture "
            "long tail is not evidence of disposable information). This is the "
            "capture distribution, not the pretraining distribution."
        ),
    }
    return {"p": p, "counts": counts, "info": info}


# ---------------------------------------------------------------------------
# per-tensor packaging
# ---------------------------------------------------------------------------

def _pack_score(
    *,
    organ: str,
    layer: int | None,
    kind: str,
    name: str,
    scored: dict,
    activation_site: str,
    activation_site_status: str,
    locked: bool,
    uniform_bpw: float,
    extra: dict | None = None,
) -> dict:
    n_in = int(scored["n_in"])
    voi_out = scored["voi_out"]
    per = voi_out / max(n_in, 1)
    depth = q_mult(layer)
    local = float(scored["local_voi"])
    rec = {
        "organ": organ,
        "layer": layer,
        "kind": kind,
        "name": name,
        "n_out": int(scored["n_out"]),
        "n_in": n_in,
        "n_params": int(scored["n_params"]),
        "local_voi": local,
        "depth_q_mult": depth,
        "depth_weighted_voi": local * depth,
        "voi_per_param": (local * depth) / max(int(scored["n_params"]), 1),
        "local_voi_per_param": local / max(int(scored["n_params"]), 1),
        "weight_only_voi": float(scored["weight_only_voi"]),
        "w_fro2": float(scored["w_fro2"]),
        "d_mean": float(scored["d_mean"]),
        "d_min": float(scored["d_min"]),
        "d_max": float(scored["d_max"]),
        "d_participation": float(scored["d_participation"]),
        "activation_site": activation_site,
        "activation_site_status": activation_site_status,
        "locked": locked,
        "uniform_bpw": float(uniform_bpw),
        "channel_voi_per_param": per * depth,
        "voi_out": voi_out,
        "spearman_channel_vs_weight": scored.get("spearman_channel_vs_weight"),
    }
    if extra:
        rec.update(extra)
    return rec


def _summarize_channels(rec: dict, cuts: tuple[float, float, float, float]) -> dict:
    import numpy as np

    per = np.asarray(rec.pop("channel_voi_per_param"), dtype=np.float64)
    voi_out = np.asarray(rec.pop("voi_out"), dtype=np.float64)
    depth = float(rec["depth_q_mult"])
    labels = classify_array(per, cuts)
    n_in = int(rec["n_in"])
    counts = {c: 0 for c in CLASSES}
    voi_mass = {c: 0.0 for c in CLASSES}
    params = {c: 0 for c in CLASSES}
    for lab, vpp, v in zip(labels, per, voi_out):
        counts[str(lab)] += 1
        voi_mass[str(lab)] += float(v) * depth
        params[str(lab)] += n_in
    # top / bottom channels
    order = np.argsort(per)
    bottom = []
    top = []
    for i in order[:TOP_K]:
        bottom.append(
            {
                "channel": int(i),
                "class": str(labels[i]),
                "voi_per_param": float(per[i]),
                "voi": float(voi_out[i]) * depth,
            }
        )
    for i in order[-TOP_K:][::-1]:
        top.append(
            {
                "channel": int(i),
                "class": str(labels[i]),
                "voi_per_param": float(per[i]),
                "voi": float(voi_out[i]) * depth,
            }
        )
    rec["class"] = classify(float(rec["voi_per_param"]), cuts)
    rec["channel_class_counts"] = counts
    rec["channel_class_voi"] = voi_mass
    rec["channel_class_params"] = params
    rec["top_critical_channels"] = top
    rec["bottom_disposable_channels"] = bottom
    rec["channel_voi_per_param_p50"] = float(np.quantile(per, 0.5)) if per.size else 0.0
    rec["channel_voi_per_param_p95"] = float(np.quantile(per, 0.95)) if per.size else 0.0
    rec["n_channels"] = int(per.size)
    return rec


def _bucket_items(rec: dict) -> list[dict]:
    items = []
    if rec["locked"]:
        return items
    for cls in CLASSES:
        n_p = int(rec["channel_class_params"].get(cls, 0))
        if n_p <= 0:
            continue
        items.append(
            {
                "id": f"L{rec['layer']}.{rec['kind']}.{cls}"
                if rec["layer"] is not None
                else f"{rec['kind']}.{cls}",
                "organ": rec["organ"],
                "layer": rec["layer"],
                "tensor": rec["kind"],
                "class": cls,
                "n_params": n_p,
                "voi": float(rec["channel_class_voi"].get(cls, 0.0)),
                "uniform_bpw": float(rec["uniform_bpw"]),
                "diagnostic": (
                    "diagonal Fisher F_jj=E[x_j^2] on capture_diverse2; "
                    "VoI=0.5 F W^2 * q_mult(layer). PROXY, ranks, does not certify."
                ),
                "negative_science": list(NEGATIVE_SCIENCE),
            }
        )
    return items


# ---------------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------------

def _is_gqa(layer: int) -> bool:
    return layer in GQA_LAYERS


def measure(parent: Path, cap: Path) -> dict:
    import numpy as np

    t0 = time.time()
    manifest = json.loads((cap / "manifest.json").read_text())
    n_tokens = int(manifest.get("total_tokens") or 0)
    X0 = load_X(cap, 0)
    if n_tokens <= 0:
        n_tokens = int(X0.shape[0])
    fit_idx, hold_idx = split_from_manifest(manifest, n_tokens)
    prior = token_prior(parent, cap, manifest)
    layers = layers_to_score()

    scores: list[dict] = []
    skipped_visual = 0
    skipped_mtp = 0
    idx = weight_index(parent)
    for name in idx:
        if "model.visual" in name:
            skipped_visual += 1
        elif name.startswith("mtp."):
            skipped_mtp += 1

    # scale-trap demonstration on a tiny real-shaped matrix (not 27B)
    rng = np.random.RandomState(SEED)
    Wtrap = rng.randn(32, 64).astype(np.float32)
    Xtrap = rng.randn(48, 64).astype(np.float32)
    dtrap = fisher_diag(Xtrap)
    trap_id = score_linear(Wtrap, dtrap)
    trap_001 = score_linear(0.01 * Wtrap, dtrap)
    trap_z = score_linear(Wtrap * 0.0, dtrap)

    print(
        f"parent={parent}\ncapture={cap} n_tokens={n_tokens} layers={layers}\n"
        f"token_prior={prior['info']['status']} unique={prior['info']['n_unique']}",
        flush=True,
    )

    for layer in layers:
        X = load_X(cap, layer)
        d_hidden = fisher_diag(X)
        X_sub = take_sub(X, fit_idx, N_SUB, seed=SEED + layer)
        site_full = f"post_attn_norm L{layer} full {X.shape[0]} tokens"
        organ_mix = "gqa" if _is_gqa(layer) else "deltanet"
        mix_floor = N041_FLOOR[organ_mix]
        Yq = Yv = Yqkv = Yz = None

        if _is_gqa(layer):
            Wq = load_tensor(parent, tensor_name(layer, "self_attn.q_proj.weight"))
            scores.append(
                _pack_score(
                    organ="gqa",
                    layer=layer,
                    kind="q_proj",
                    name=tensor_name(layer, "self_attn.q_proj.weight"),
                    scored=score_linear(Wq, d_hidden),
                    activation_site=site_full + " (PROXY vs input_layernorm; real distribution)",
                    activation_site_status="PROXY_SITE",
                    locked=False,
                    uniform_bpw=mix_floor,
                )
            )
            Yq = gemm_xt(X_sub, Wq)
            del Wq
            Wk = load_tensor(parent, tensor_name(layer, "self_attn.k_proj.weight"))
            scores.append(
                _pack_score(
                    organ="gqa",
                    layer=layer,
                    kind="k_proj",
                    name=tensor_name(layer, "self_attn.k_proj.weight"),
                    scored=score_linear(Wk, d_hidden),
                    activation_site=site_full + " (PROXY vs input_layernorm; real distribution)",
                    activation_site_status="PROXY_SITE",
                    locked=False,
                    uniform_bpw=mix_floor,
                )
            )
            del Wk
            Wv = load_tensor(parent, tensor_name(layer, "self_attn.v_proj.weight"))
            scores.append(
                _pack_score(
                    organ="gqa",
                    layer=layer,
                    kind="v_proj",
                    name=tensor_name(layer, "self_attn.v_proj.weight"),
                    scored=score_linear(Wv, d_hidden),
                    activation_site=site_full + " (PROXY vs input_layernorm; real distribution)",
                    activation_site_status="PROXY_SITE",
                    locked=False,
                    uniform_bpw=mix_floor,
                )
            )
            Yv = gemm_xt(X_sub, Wv)
            del Wv
            o_in = gqa_out_from_y(Yq, Yv)
            d_o = fisher_diag(o_in)
            Wo = load_tensor(parent, tensor_name(layer, "self_attn.o_proj.weight"))
            scores.append(
                _pack_score(
                    organ="gqa",
                    layer=layer,
                    kind="o_proj",
                    name=tensor_name(layer, "self_attn.o_proj.weight"),
                    scored=score_linear(Wo, d_o),
                    activation_site=(
                        f"gqa_out_proxy L{layer} subsample {o_in.shape[0]} "
                        "(repeat(v)*sigmoid(q_gate) from streamed parent; not a second decode)"
                    ),
                    activation_site_status="PROXY_SUBSAMPLE",
                    locked=False,
                    uniform_bpw=mix_floor,
                    extra={"n_sub": int(o_in.shape[0])},
                )
            )
            del Wo, o_in, Yq, Yv
            for kind in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
                w = load_tensor(parent, tensor_name(layer, kind))
                scores.append(
                    _pack_score(
                        organ="gqa",
                        layer=layer,
                        kind=kind.split(".")[1],
                        name=tensor_name(layer, kind),
                        scored=score_vector(w, d_hidden if w.reshape(-1).size == d_hidden.size else None),
                        activation_site=site_full + " (norm; PROXY using post_attn energy)",
                        activation_site_status="PROXY_SITE",
                        locked=True,
                        uniform_bpw=LEFTOVER_BPW,
                    )
                )
                del w
        else:
            Wqkv = load_tensor(parent, tensor_name(layer, "linear_attn.in_proj_qkv.weight"))
            scores.append(
                _pack_score(
                    organ="deltanet",
                    layer=layer,
                    kind="in_proj_qkv",
                    name=tensor_name(layer, "linear_attn.in_proj_qkv.weight"),
                    scored=score_linear(Wqkv, d_hidden),
                    activation_site=site_full + " (PROXY vs input_layernorm; real distribution)",
                    activation_site_status="PROXY_SITE",
                    locked=False,
                    uniform_bpw=mix_floor,
                )
            )
            Yqkv = gemm_xt(X_sub, Wqkv)
            del Wqkv
            Wz = load_tensor(parent, tensor_name(layer, "linear_attn.in_proj_z.weight"))
            scores.append(
                _pack_score(
                    organ="deltanet",
                    layer=layer,
                    kind="in_proj_z",
                    name=tensor_name(layer, "linear_attn.in_proj_z.weight"),
                    scored=score_linear(Wz, d_hidden),
                    activation_site=site_full + " (PROXY vs input_layernorm; real distribution)",
                    activation_site_status="PROXY_SITE",
                    locked=False,
                    uniform_bpw=mix_floor,
                )
            )
            Yz = gemm_xt(X_sub, Wz)
            del Wz
            for kind in ("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"):
                Wab = load_tensor(parent, tensor_name(layer, kind))
                scores.append(
                    _pack_score(
                        organ="deltanet",
                        layer=layer,
                        kind=kind.split(".")[1],
                        name=tensor_name(layer, kind),
                        scored=score_linear(Wab, d_hidden),
                        activation_site=site_full + " (PROXY vs input_layernorm; real distribution)",
                        activation_site_status="PROXY_SITE",
                        locked=False,
                        uniform_bpw=mix_floor,
                    )
                )
                del Wab
            o_in = dn_out_from_y(Yqkv, Yz)
            d_o = fisher_diag(o_in)
            Wo = load_tensor(parent, tensor_name(layer, "linear_attn.out_proj.weight"))
            scores.append(
                _pack_score(
                    organ="deltanet",
                    layer=layer,
                    kind="out_proj",
                    name=tensor_name(layer, "linear_attn.out_proj.weight"),
                    scored=score_linear(Wo, d_o),
                    activation_site=(
                        f"deltanet_out_proxy L{layer} subsample {o_in.shape[0]} "
                        "(v*silu(z) from streamed in_proj; NOT the recurrent S mix)"
                    ),
                    activation_site_status="PROXY_SUBSAMPLE",
                    locked=False,
                    uniform_bpw=mix_floor,
                    extra={"n_sub": int(o_in.shape[0]), "recurrent_state_not_captured": True},
                )
            )
            del Wo, o_in, Yqkv, Yz
            for kind, force_crit in (
                ("linear_attn.A_log", True),
                ("linear_attn.dt_bias", True),
                ("linear_attn.conv1d.weight", True),
                ("linear_attn.norm.weight", False),
            ):
                w = load_tensor(parent, tensor_name(layer, kind))
                d_use = d_hidden if w.reshape(-1).size == d_hidden.size else None
                rec = _pack_score(
                    organ="deltanet",
                    layer=layer,
                    kind=kind.split(".", 1)[-1],
                    name=tensor_name(layer, kind),
                    scored=score_vector(w, d_use),
                    activation_site=(
                        "WEIGHT_ONLY leftover (recurrent A_log/dt_bias/conv have no "
                        "captured state activations); S024 §32 keeps these high precision"
                        if force_crit
                        else site_full + " (norm PROXY)"
                    ),
                    activation_site_status="WEIGHT_ONLY" if force_crit else "PROXY_SITE",
                    locked=True,
                    uniform_bpw=LEFTOVER_BPW,
                    extra={"policy_critical": force_crit},
                )
                scores.append(rec)
                del w

        # MLP: gate/up on full hidden Fisher; down on subsampled SwiGLU intermediate
        Wg = load_tensor(parent, tensor_name(layer, "mlp.gate_proj.weight"))
        scores.append(
            _pack_score(
                organ="mlp",
                layer=layer,
                kind="gate_proj",
                name=tensor_name(layer, "mlp.gate_proj.weight"),
                scored=score_linear(Wg, d_hidden),
                activation_site=site_full,
                activation_site_status="MEASURED",
                locked=False,
                uniform_bpw=N041_FLOOR["mlp"],
            )
        )
        Yg = gemm_xt(X_sub, Wg)
        del Wg
        Wu = load_tensor(parent, tensor_name(layer, "mlp.up_proj.weight"))
        scores.append(
            _pack_score(
                organ="mlp",
                layer=layer,
                kind="up_proj",
                name=tensor_name(layer, "mlp.up_proj.weight"),
                scored=score_linear(Wu, d_hidden),
                activation_site=site_full,
                activation_site_status="MEASURED",
                locked=False,
                uniform_bpw=N041_FLOOR["mlp"],
            )
        )
        Yu = gemm_xt(X_sub, Wu)
        del Wu
        h = silu(Yg) * Yu
        d_down = fisher_diag(h)
        del Yg, Yu
        Wd = load_tensor(parent, tensor_name(layer, "mlp.down_proj.weight"))
        scores.append(
            _pack_score(
                organ="mlp",
                layer=layer,
                kind="down_proj",
                name=tensor_name(layer, "mlp.down_proj.weight"),
                scored=score_linear(Wd, d_down),
                activation_site=(
                    f"swiglu intermediate L{layer} subsample {h.shape[0]} "
                    "(silu(X W_gate^T)*(X W_up^T) from streamed parent; not a second decode)"
                ),
                activation_site_status="PROXY_SUBSAMPLE",
                locked=False,
                uniform_bpw=N041_FLOOR["mlp"],
                extra={"n_sub": int(h.shape[0])},
            )
        )
        del Wd, h
        for kind in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            w = load_tensor(parent, tensor_name(layer, kind))
            scores.append(
                _pack_score(
                    organ="mlp" if "post_attention" in kind else organ_mix,
                    layer=layer,
                    kind=kind,
                    name=tensor_name(layer, kind),
                    scored=score_vector(w, d_hidden),
                    activation_site=site_full + " (RMSNorm PROXY using post_attn energy)",
                    activation_site_status="PROXY_SITE",
                    locked=True,
                    uniform_bpw=LEFTOVER_BPW,
                )
            )
            del w
        del X, d_hidden, X_sub
        gc.collect()
        print(
            f"  L{layer:02d} {_is_gqa(layer) and 'GQA' or 'DN'} "
            f"n_scores={len(scores)} elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

    # embed: token-row Fisher from reconstructed capture ids.
    # S026 §109: unseen-in-capture rows are NOT disposable. Lock them at the
    # N041 embed floor so the allocator cannot steal the long tail.
    print("embed_tokens ...", flush=True)
    import numpy as np

    emb_name = "model.language_model.embed_tokens.weight"
    meta_e = tensor_meta(parent, emb_name)
    erows, ecols = int(meta_e["shape"][0]), int(meta_e["shape"][1])
    counts = prior["counts"]
    emb = score_embed_rows(iter_matrix_rows(parent, emb_name), counts, erows, ecols)
    seen = np.asarray(counts) > 0
    unseen = ~seen
    if int(seen.sum()) > 0:
        scores.append(
            _pack_score(
                organ="embedding",
                layer=0,
                kind="embed_tokens.seen",
                name=emb_name,
                scored=_slice_channels(emb, seen),
                activation_site=(
                    "token one-hot with p(token) from capture_diverse2 reconstructed "
                    f"ids ({prior['info']['status']}); SEEN rows only"
                ),
                activation_site_status=prior["info"]["status"],
                locked=False,
                uniform_bpw=N041_FLOOR["embedding"],
                extra={"n_seen_rows": int(seen.sum()), "s026_109_long_tail_locked": False},
            )
        )
    if int(unseen.sum()) > 0:
        scores.append(
            _pack_score(
                organ="embedding",
                layer=0,
                kind="embed_tokens.unseen",
                name=emb_name,
                scored=_slice_channels(emb, unseen),
                activation_site=(
                    "UNMEASURED_IN_CAPTURE. Laplace would call these disposable; "
                    "S026 §109 forbids starving the long tail the calibration set "
                    "did not exercise. Locked at the N041 embedding floor."
                ),
                activation_site_status="UNMEASURED_IN_CAPTURE",
                locked=True,
                uniform_bpw=N041_FLOOR["embedding"],
                extra={
                    "n_unseen_rows": int(unseen.sum()),
                    "unmeasured_in_capture": True,
                    "s026_109_long_tail_locked": True,
                    "policy_class": "ordinary",
                },
            )
        )
    gc.collect()

    # lm_head: standard linear, d = last-layer post_attn (PROXY for final hidden)
    print("lm_head ...", flush=True)
    X_last = load_X(cap, max(layers))
    d_last = fisher_diag(X_last)
    lm_name = "lm_head.weight"
    meta_l = tensor_meta(parent, lm_name)
    # stream rows of lm_head with d_last on the hidden axis
    import numpy as np

    n_out, n_in = int(meta_l["shape"][0]), int(meta_l["shape"][1])
    voi_out = np.empty(n_out, dtype=np.float64)
    voi_in = np.zeros(n_in, dtype=np.float64)
    row_fro2 = np.empty(n_out, dtype=np.float64)
    w_fro2 = 0.0
    for i0, rows in iter_matrix_rows(parent, lm_name):
        w2 = np.square(np.asarray(rows, dtype=np.float32), dtype=np.float64)
        row_fro2[i0 : i0 + rows.shape[0]] = w2.sum(axis=1)
        voi_out[i0 : i0 + rows.shape[0]] = 0.5 * (w2 * d_last).sum(axis=1)
        voi_in += 0.5 * d_last * w2.sum(axis=0)
        w_fro2 += float(w2.sum())
    lm_scored = {
        "n_out": n_out,
        "n_in": n_in,
        "n_params": n_out * n_in,
        "local_voi": float(voi_out.sum()),
        "voi_out": voi_out,
        "voi_in": voi_in,
        "row_fro2": row_fro2,
        "w_fro2": w_fro2,
        "weight_only_voi": 0.5 * float(d_last.mean()) * w_fro2,
        "d_mean": float(d_last.mean()),
        "d_min": float(d_last.min()),
        "d_max": float(d_last.max()),
        "d_participation": _participation(d_last),
        "spearman_channel_vs_weight": spearman(voi_out, row_fro2),
    }
    scores.append(
        _pack_score(
            organ="output",
            layer=max(layers),
            kind="lm_head",
            name=lm_name,
            scored=lm_scored,
            activation_site=(
                f"post_attn_norm L{max(layers)} full {X_last.shape[0]} "
                "(PROXY for final hidden: last MLP + model.norm were not captured; "
                "no second 27B decode)"
            ),
            activation_site_status="PROXY_SITE",
            locked=False,
            uniform_bpw=N041_FLOOR["output"],
        )
    )
    del X_last, d_last
    gc.collect()

    # final norm leftover
    if "model.language_model.norm.weight" in idx:
        w = load_tensor(parent, "model.language_model.norm.weight")
        Xn = load_X(cap, max(layers))
        scores.append(
            _pack_score(
                organ="output",
                layer=max(layers),
                kind="final_norm",
                name="model.language_model.norm.weight",
                scored=score_vector(w, fisher_diag(Xn)),
                activation_site=f"post_attn_norm L{max(layers)} (PROXY for pre-final-norm)",
                activation_site_status="PROXY_SITE",
                locked=True,
                uniform_bpw=LEFTOVER_BPW,
            )
        )
        del w, Xn

    return {
        "scores": scores,
        "manifest": manifest,
        "n_tokens": n_tokens,
        "fit_idx_n": int(len(fit_idx)),
        "hold_idx_n": int(len(hold_idx)),
        "prior": prior["info"],
        "layers": list(layers),
        "skipped_visual": skipped_visual,
        "skipped_mtp": skipped_mtp,
        "scale_trap": {
            "identity_voi": trap_id["local_voi"],
            "scaled_0p01_voi": trap_001["local_voi"],
            "zero_voi": trap_z["local_voi"],
            "ratio_0p01_over_identity": trap_001["local_voi"] / max(trap_id["local_voi"], 1e-30),
            "expected_ratio": 1e-4,
            "rejects_scaled_artifact": True,
            "note": (
                "VoI is second-order in W, so 0.01*W has 1e-4 the VoI. Cosine would "
                "score 1.0 on 0.01*W; this proxy does not."
            ),
        },
        "elapsed_s": time.time() - t0,
    }


def assemble(raw: dict, parent: Path, cap: Path) -> dict:
    import numpy as np

    scores = raw["scores"]
    organ_vals: dict[str, list] = defaultdict(list)
    gemv_vals = []
    rho_ch = []
    for rec in scores:
        v = rec.get("channel_voi_per_param")
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float64)
        if rec.get("spearman_channel_vs_weight") is not None:
            rho_ch.append(float(rec["spearman_channel_vs_weight"]))
        if rec["locked"]:
            continue
        organ_vals[rec["organ"]].append(arr)
        if rec["organ"] != "embedding":
            gemv_vals.append(arr)
    organ_cuts = {
        o: class_cuts(np.concatenate(vs)) for o, vs in organ_vals.items() if vs
    }
    if gemv_vals:
        all_ch = np.concatenate(gemv_vals)
    else:
        all_ch = np.zeros(1, dtype=np.float64)
    cuts = class_cuts(all_ch)
    for rec in scores:
        use = organ_cuts.get(rec["organ"], cuts)
        _summarize_channels(rec, use)
        if rec.get("policy_critical"):
            rec["class"] = "critical"
        if rec.get("unmeasured_in_capture") or rec.get("policy_class"):
            rec["class"] = rec.get("policy_class") or "ordinary"

    items: list[dict] = []
    for rec in scores:
        items.extend(_bucket_items(rec))

    budget_bits = sum(float(it["n_params"]) * float(it["uniform_bpw"]) for it in items)
    filled = greedy_fill(items, budget_bits)
    assignment = filled["assignment"]

    uniform_mse = sum(remaining_mse(float(it["voi"]), float(it["uniform_bpw"])) for it in items)
    greedy_mse = sum(a["remaining_mse_proxy"] for a in assignment)
    greedy_bits = sum(a["recommended_bits"] for a in assignment)

    # organ / layer aggregates
    def _agg(rows, key_fn):
        out = {}
        for r in rows:
            k = key_fn(r)
            b = out.setdefault(
                k,
                {
                    "n_params": 0,
                    "local_voi": 0.0,
                    "depth_weighted_voi": 0.0,
                    "uniform_bits": 0.0,
                    "recommended_bits": 0.0,
                    "class_params": {c: 0 for c in CLASSES},
                },
            )
            b["n_params"] += int(r["n_params"])
            b["local_voi"] += float(r["local_voi"])
            b["depth_weighted_voi"] += float(r["depth_weighted_voi"])
            b["uniform_bits"] += float(r["n_params"]) * float(r["uniform_bpw"])
            for c in CLASSES:
                b["class_params"][c] += int(r["channel_class_counts"].get(c, 0)) * int(r["n_in"])
        return out

    by_organ_scores = _agg(scores, lambda r: r["organ"])
    rec_bits_organ = defaultdict(float)
    rec_n_organ = defaultdict(int)
    rec_bits_unlocked = defaultdict(float)
    rec_n_unlocked = defaultdict(int)
    for a in assignment:
        rec_bits_organ[a["organ"]] += a["recommended_bits"]
        rec_n_organ[a["organ"]] += int(a["n_params"])
        rec_bits_unlocked[a["organ"]] += a["recommended_bits"]
        rec_n_unlocked[a["organ"]] += int(a["n_params"])
    for rec in scores:
        if rec["locked"]:
            rec_bits_organ[rec["organ"]] += int(rec["n_params"]) * float(rec["uniform_bpw"])
            rec_n_organ[rec["organ"]] += int(rec["n_params"])
    organ_table = []
    for organ, b in by_organ_scores.items():
        n_all = rec_n_organ.get(organ, 0)
        n_unlock = rec_n_unlocked.get(organ, 0)
        rec_bpw = rec_bits_organ[organ] / n_all if n_all else None
        rec_bpw_unlocked = rec_bits_unlocked[organ] / n_unlock if n_unlock else None
        uni_bpw = N041_FLOOR.get(organ)
        organ_table.append(
            {
                "organ": organ,
                "n_params_scored": b["n_params"],
                "n_params_unlocked": n_unlock,
                "local_voi": b["local_voi"],
                "depth_weighted_voi": b["depth_weighted_voi"],
                "voi_per_param": b["depth_weighted_voi"] / max(b["n_params"], 1),
                "uniform_bpw_cited": uni_bpw,
                "recommended_bpw": rec_bpw,
                "recommended_bpw_unlocked": rec_bpw_unlocked,
                "below_uniform": (rec_bpw is not None and uni_bpw is not None and rec_bpw < uni_bpw - 1e-6),
                "delta_bpw": (rec_bpw - uni_bpw) if rec_bpw is not None and uni_bpw is not None else None,
                "class_params": b["class_params"],
            }
        )
    organ_table.sort(key=lambda r: r["voi_per_param"])

    # layer ranking (unlocked GEMV only)
    layer_map: dict[int, dict] = {}
    for rec in scores:
        if rec["locked"] or rec["layer"] is None:
            continue
        b = layer_map.setdefault(
            int(rec["layer"]),
            {"layer": int(rec["layer"]), "n_params": 0, "depth_weighted_voi": 0.0, "organs": set()},
        )
        b["n_params"] += int(rec["n_params"])
        b["depth_weighted_voi"] += float(rec["depth_weighted_voi"])
        b["organs"].add(rec["organ"])
    rec_bits_layer = defaultdict(float)
    rec_n_layer = defaultdict(int)
    uni_bits_layer = defaultdict(float)
    for a in assignment:
        if a["layer"] is None:
            continue
        rec_bits_layer[int(a["layer"])] += a["recommended_bits"]
        rec_n_layer[int(a["layer"])] += int(a["n_params"])
        uni_bits_layer[int(a["layer"])] += float(a["n_params"]) * float(a["uniform_bpw"])
    layer_table = []
    for L, b in layer_map.items():
        n = rec_n_layer.get(L, 0)
        rec_bpw = rec_bits_layer[L] / n if n else None
        uni_bpw = uni_bits_layer[L] / n if n else None
        layer_table.append(
            {
                "layer": L,
                "organs": sorted(b["organs"]),
                "n_params": b["n_params"],
                "depth_weighted_voi": b["depth_weighted_voi"],
                "voi_per_param": b["depth_weighted_voi"] / max(b["n_params"], 1),
                "q_mult": q_mult(L),
                "uniform_bpw": uni_bpw,
                "recommended_bpw": rec_bpw,
                "below_uniform": (
                    rec_bpw is not None and uni_bpw is not None and rec_bpw < uni_bpw - 1e-6
                ),
            }
        )
    layer_table.sort(key=lambda r: r["voi_per_param"])

    # tensor ranking among large unlocked GEMVs (embed table is a separate organ)
    big = [
        rec
        for rec in scores
        if (not rec["locked"])
        and int(rec["n_params"]) >= MIN_REGION_PARAMS
        and not str(rec["kind"]).startswith("embed_tokens")
    ]
    big_sorted = sorted(big, key=lambda r: r["voi_per_param"])
    least = []
    for rec in big_sorted[:12]:
        least.append(
            {
                "name": rec["name"],
                "organ": rec["organ"],
                "layer": rec["layer"],
                "kind": rec["kind"],
                "n_params": rec["n_params"],
                "class": rec["class"],
                "voi_per_param": rec["voi_per_param"],
                "depth_weighted_voi": rec["depth_weighted_voi"],
                "uniform_bpw": rec["uniform_bpw"],
                "activation_site_status": rec["activation_site_status"],
                "why": (
                    "lowest depth-weighted VoI per parameter among GEMV tensors "
                    f"with n_params >= {MIN_REGION_PARAMS}; cheapest to compress further "
                    "UNDER THIS PROXY (ranks, does not certify composition)"
                ),
                "negative_science": list(NEGATIVE_SCIENCE),
            }
        )
    most = []
    for rec in big_sorted[-12:][::-1]:
        most.append(
            {
                "name": rec["name"],
                "organ": rec["organ"],
                "layer": rec["layer"],
                "kind": rec["kind"],
                "n_params": rec["n_params"],
                "class": rec["class"],
                "voi_per_param": rec["voi_per_param"],
                "depth_weighted_voi": rec["depth_weighted_voi"],
                "uniform_bpw": rec["uniform_bpw"],
                "activation_site_status": rec["activation_site_status"],
                "why": (
                    "highest depth-weighted VoI per parameter among GEMV tensors "
                    f"with n_params >= {MIN_REGION_PARAMS}; spend bits here first "
                    "UNDER THIS PROXY"
                ),
                "negative_science": list(NEGATIVE_SCIENCE),
            }
        )

    # activation-awareness vs weight-only: CHANNEL grain (tensor grain is ~1
    # whenever d is similar across tensors of the same site).
    aa = [float(r["local_voi"]) for r in scores if not r["locked"]]
    wo = [float(r["weight_only_voi"]) for r in scores if not r["locked"]]
    rho_tensor = spearman(aa, wo) if len(aa) >= 4 else None
    rho = float(np.mean(rho_ch)) if rho_ch else rho_tensor

    rec_organ_bpw = {o["organ"]: o["recommended_bpw"] for o in organ_table}
    below = [o["organ"] for o in organ_table if o.get("below_uniform")]
    above = [
        o["organ"]
        for o in organ_table
        if o.get("recommended_bpw") is not None
        and o.get("uniform_bpw_cited") is not None
        and o["recommended_bpw"] > o["uniform_bpw_cited"] + 1e-6
    ]

    leftover_n = sum(
        int(r["n_params"]) for r in scores if r["locked"] and float(r["uniform_bpw"]) == LEFTOVER_BPW
    )
    locked_bits = sum(int(r["n_params"]) * float(r["uniform_bpw"]) for r in scores if r["locked"])
    gemv_n = sum(int(r["n_params"]) for r in scores if not r["locked"])
    illustrative_complete = (greedy_bits + locked_bits) / SOURCE_PARAM_COUNT
    uniform_complete = (budget_bits + locked_bits) / SOURCE_PARAM_COUNT

    curve = []
    for pt in filled["curve"]:
        curve.append(
            {
                "gemv_bits": pt["bits"],
                "complete_ebpw": (pt["bits"] + locked_bits) / SOURCE_PARAM_COUNT,
                "remaining_mse_proxy": pt["remaining_mse_proxy"],
                "step": pt["step"],
            }
        )

    # strip bulky arrays already popped; make JSON-safe tensor table
    tensor_table = []
    for rec in scores:
        tensor_table.append(
            {
                "name": rec["name"],
                "organ": rec["organ"],
                "layer": rec["layer"],
                "kind": rec["kind"],
                "n_params": rec["n_params"],
                "n_out": rec["n_out"],
                "n_in": rec["n_in"],
                "class": rec["class"],
                "local_voi": rec["local_voi"],
                "depth_weighted_voi": rec["depth_weighted_voi"],
                "voi_per_param": rec["voi_per_param"],
                "weight_only_voi": rec["weight_only_voi"],
                "activation_site": rec["activation_site"],
                "activation_site_status": rec["activation_site_status"],
                "locked": rec["locked"],
                "uniform_bpw": rec["uniform_bpw"],
                "channel_class_counts": rec["channel_class_counts"],
                "channel_class_params": rec["channel_class_params"],
                "channel_voi_per_param_p50": rec["channel_voi_per_param_p50"],
                "channel_voi_per_param_p95": rec["channel_voi_per_param_p95"],
                "d_participation": rec["d_participation"],
                "d_mean": rec["d_mean"],
                "d_min": rec["d_min"],
                "d_max": rec["d_max"],
                "top_critical_channels": rec["top_critical_channels"][:4],
                "bottom_disposable_channels": rec["bottom_disposable_channels"][:4],
            }
        )

    n041_items = []
    for a in assignment:
        n041_items.append(
            {
                "id": a["id"],
                "organ": a["organ"],
                "layer": a["layer"],
                "tensor": a["tensor"],
                "class": a["class"],
                "n_params": a["n_params"],
                "voi": a["voi"],
                "voi_per_param": float(a["voi"]) / max(int(a["n_params"]), 1),
                "uniform_bpw": a["uniform_bpw"],
                "recommended_bpw": a["recommended_bpw"],
                "marginal_at_uniform": mse_drop(
                    float(a["voi"]), float(a["uniform_bpw"]), float(a["uniform_bpw"]) + 1.0
                )
                / max(int(a["n_params"]), 1),
                "diagnostic": a["diagnostic"],
                "negative_science": a["negative_science"],
            }
        )

    return {
        "cuts": {
            "on": (
                "per-organ rank cuts on depth-weighted VoI/param of unlocked channels; "
                "global values below are GEMV-only (embed excluded so the vocab tail "
                "cannot dominate percentiles). Embed unseen rows are locked (S026 §109)."
            ),
            "ranks": list(CLASS_RANKS),
            "values": {
                "p05_disposable_max": cuts[0],
                "p25_cheap_max": cuts[1],
                "p75_ordinary_max": cuts[2],
                "p95_sensitive_max": cuts[3],
            },
            "per_organ": {
                o: {
                    "p05": c[0],
                    "p25": c[1],
                    "p75": c[2],
                    "p95": c[3],
                }
                for o, c in organ_cuts.items()
            },
            "n_channels": int(all_ch.size),
            "classes": list(CLASSES),
        },
        "organ_table": organ_table,
        "layer_table": layer_table,
        "tensor_table": tensor_table,
        "least_sensitive_regions": least,
        "most_sensitive_regions": most,
        "spearman_activation_aware_vs_weight_only": rho,
        "spearman_tensor_grain": rho_tensor,
        "spearman_channel_grain_mean": float(np.mean(rho_ch)) if rho_ch else None,
        "assignment": {
            "budget_bits": budget_bits,
            "greedy_bits": greedy_bits,
            "uniform_remaining_mse_proxy": uniform_mse,
            "greedy_remaining_mse_proxy": greedy_mse,
            "greedy_beats_uniform_mse_proxy": greedy_mse < uniform_mse,
            "relative_mse_drop": (uniform_mse - greedy_mse) / max(uniform_mse, 1e-30),
            "hit_budget": filled["hit_budget"],
            "n_items": len(assignment),
            "bit_levels": list(BIT_LEVELS),
            "mse_model": "remaining_mse = depth_weighted_VoI / 4^{bpw} (RTN grouped-quant PROXY)",
        },
        "n041_consumer": {
            "schema": "hawking.headless.sensitivity_allocation.n041_consumer.v1",
            "does_not_claim_new_whole_model_floor": True,
            "sensitivity_is_a_proxy": True,
            "ranks_does_not_certify": True,
            "baseline_complete_ebpw_cited": N041_COMPLETE_EBPW,
            "baseline_source": N041_RECEIPT,
            "uniform_organ_floors_cited": dict(N041_FLOOR),
            "proxy": "diagonal_fisher_activation_weighted_magnitude",
            "formula": "VoI_ij = 0.5 * E[x_j^2] * W_ij^2 * q_mult(layer); embed uses 0.5 p(token) ||row||^2",
            "marginal_mse_model": "remaining_mse = VoI / 4^{bpw}",
            "recommended_organ_bpw": rec_organ_bpw,
            "below_uniform_organs": below,
            "above_uniform_organs": above,
            "illustrative_complete_ebpw_at_same_gemv_bits": illustrative_complete,
            "uniform_complete_ebpw_from_measured_n": uniform_complete,
            "leftover_n": leftover_n,
            "leftover_bpw_cited": LEFTOVER_BPW,
            "locked_bits": locked_bits,
            "gemv_n": gemv_n,
            "items": n041_items,
            "curve": curve,
            "least_sensitive_regions": least,
            "most_sensitive_regions": most,
            "note": (
                "Total GEMV bits are held at the N041 uniform-per-organ mix. "
                "Recommended bpw is a reallocation, not a new floor. N041 must still "
                "run composition screening before moving any organ below its measured floor. "
                "1.25 binary / 1.85 ternary failed whole-MLP composition; starving a "
                "channel to those densities is a ranking, not a license."
            ),
        },
        "all_ch_mean": float(all_ch.mean()) if all_ch.size else 0.0,
        "all_ch_p50": float(np.quantile(all_ch, 0.5)) if all_ch.size else 0.0,
    }


def build_receipt(parent: Path, cap: Path, raw: dict, assembled: dict) -> dict:
    py = sys.executable
    citations = {
        "n041": N041_RECEIPT,
        "n040": N040_RECEIPT,
        "q_inject": "tools/headless/global_allocator.py (CITED Q_INJECT from gravity_error_chain)",
        "capture": str(cap),
        "parent": str(parent),
        "negative_science": list(NEGATIVE_SCIENCE),
        "activation_source": (
            "workspace/campaign/phaseB/capture_diverse2 via "
            f"{cap}: post_attn_norm, {raw['n_tokens']} real tokens, BF16 parent MLX "
            "full-model forward. Not Gaussian. Mixer in_proj uses this site as a "
            "PROXY (true site is input_layernorm). down_proj / out_proj activations "
            "are reconstructed on a 256-token fit subsample from streamed parent "
            "weights. lm_head uses last-layer post_attn as a PROXY for final hidden. "
            "embed uses reconstructed capture token frequencies. No second 27B decode."
        ),
    }
    missing = [c for c in NEGATIVE_SCIENCE if not citation_exists(c)]
    doc = {
        "schema": SCHEMA,
        "obligation": OBLIGATION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git_head(),
        "generated_by": GENERATOR,
        "hand_authored": False,
        "python": py,
        "parent": str(parent),
        "did_not_load_second_27b": True,
        "did_not_touch_gpu": True,
        "did_not_run_cargo_or_metal_benchmarks": True,
        "did_not_mutate_noetic_parent_a": True,
        "did_not_write_under_models": True,
        "sensitivity_is_a_proxy": True,
        "ranks_does_not_certify": True,
        "does_not_claim_new_whole_model_floor": True,
        "cited_n041_complete_ebpw": N041_COMPLETE_EBPW,
        "cited_n041_source": N041_RECEIPT,
        "proxy": {
            "name": "diagonal_fisher_activation_weighted_magnitude",
            "formula": "VoI_ij = 0.5 * E_t[x_{t j}^2] * W_{ij}^2 ; capability_VoI = VoI * q_mult(layer)",
            "fisher": "For Y=XW^T under a locally-quadratic (MSE / Gaussian) model, the Hessian/Fisher diagonal on W_ij is E[x_j^2].",
            "not_full_fisher": True,
            "not_composition_certificate": True,
            "nulls": {
                "gaussian_isotropic_X": "E[x_j^2] constant in j; ranking collapses to |W|. Real d_participation << 1 is evidence the capture is not that null.",
                "weight_only": "0.5 mean(d) ||W||_F^2. Spearman vs activation-aware is reported; disagreement is the value of the activation term.",
                "cosine_scale_trap": "0.01*W scores cosine 1.0; this VoI scores 1e-4 of identity.",
            },
        },
        "capture": {
            "path": str(cap),
            "site": "post_attn_norm",
            "n_tokens": raw["n_tokens"],
            "n_fit": raw["fit_idx_n"],
            "n_hold": raw["hold_idx_n"],
            "n_sub_reconstructed_intermediates": N_SUB,
            "hidden": HIDDEN,
            "not_gaussian": True,
            "not_llama_server_teacher": True,
            "split_rule": "last 3 prompts/family = hold (organ_frontiers split_from_manifest)",
            "source_note": citations["activation_source"],
            "token_prior": raw["prior"],
        },
        "scale_trap": raw["scale_trap"],
        "q_inject": {
            "table": {str(k): v for k, v in Q_INJECT.items()},
            "source": "CITED from tools/headless/global_allocator.py / tools/gravity_error_chain.py",
            "not_rederived": True,
            "q_mult_L0": 1.0,
            "q_mult_L63": q_mult(63),
        },
        "n_layers_scored": len(raw["layers"]),
        "layers_scored": raw["layers"],
        "skipped_visual_tensors": raw["skipped_visual"],
        "skipped_mtp_tensors": raw["skipped_mtp"],
        "visual_and_mtp_not_in_n041_closure": True,
        "elapsed_s": raw["elapsed_s"],
        "class_cuts": assembled["cuts"],
        "organs_ranked_least_sensitive_first": assembled["organ_table"],
        "layers_ranked_least_sensitive_first": assembled["layer_table"],
        "least_sensitive_regions": assembled["least_sensitive_regions"],
        "most_sensitive_regions": assembled["most_sensitive_regions"],
        "spearman_activation_aware_vs_weight_only": assembled[
            "spearman_activation_aware_vs_weight_only"
        ],
        "spearman_tensor_grain": assembled.get("spearman_tensor_grain"),
        "spearman_channel_grain_mean": assembled.get("spearman_channel_grain_mean"),
        "s026_109_unseen_embed_locked": True,
        "assignment": assembled["assignment"],
        "n041_consumer": assembled["n041_consumer"],
        "tensors": assembled["tensor_table"],
        "citations": citations,
        "citations_missing": missing,
        "negative_science": [
            {
                "path": p,
                "present": citation_exists(p),
                "why": "starving below a measured organ floor is a ranking, not a reopen of a closed composition failure",
            }
            for p in NEGATIVE_SCIENCE
        ],
        "headline": _headline(assembled),
    }
    return doc


def _headline(assembled: dict) -> str:
    organs = assembled["organ_table"]
    least = assembled["least_sensitive_regions"]
    below = assembled["n041_consumer"]["below_uniform_organs"]
    above = assembled["n041_consumer"]["above_uniform_organs"]
    least_s = ", ".join(
        f"L{r['layer']}.{r['kind']}" for r in least[:5]
    ) if least else "(none)"
    organ_s = ", ".join(
        f"{o['organ']} rec={o['recommended_bpw']:.3f} (uni {o['uniform_bpw_cited']})"
        for o in organs
        if o.get("recommended_bpw") is not None
    )
    return (
        f"PROXY map at equal bits to N041 {N041_COMPLETE_EBPW:.4f} complete EBPW. "
        f"Organs below uniform: {below or 'none'}; above: {above or 'none'}. "
        f"{organ_s}. Least-sensitive GEMV regions (cheapest to compress further, "
        f"n_params>={MIN_REGION_PARAMS}): {least_s}. This is not a new floor."
    )


def main() -> int:
    parent = find_parent()
    cap = find_capture()
    print("N051 SENSITIVITY_ALLOCATION — CPU stream, no GPU, no second 27B", flush=True)
    raw = measure(parent, cap)
    assembled = assemble(raw, parent, cap)
    doc = build_receipt(parent, cap, raw, assembled)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(j(doc), indent=2) + "\n")
    print(doc["headline"], flush=True)
    print(f"receipt: {RECEIPT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

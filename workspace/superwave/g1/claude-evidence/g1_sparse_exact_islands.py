#!/usr/bin/env python3
"""CPU-only sparse-exact-island screen on real Qwen3.8 attention tensors.

Does not touch GPU, does not write into the repo, does not pack an artifact.
"""
from __future__ import annotations

import gc
import json
import os
import resource
import time
from collections import OrderedDict

import numpy as np

SRC = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
ACT = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1/hidden"
OUT = "/tmp/g1_sparse_exact_islands.json"

FRACS = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
BASES = [
    ("none", None, None, 0.0),
    ("binary_g128", "binary", 128, 1.0 + 16.0 / 128.0),
    ("uniform_q2_g64", 2, 64, 2.0 + 16.0 / 64.0),
    ("uniform_q3_g64", 3, 64, 3.0 + 16.0 / 64.0),
]
VALUE_BITS = 16  # bf16 exact island payload
G_BITMAP = (64, 128)
FIXED_SLOTS = (1, 2, 4, 8)
FIXED_G = 64

TENSORS = [
    # layer, kind, suffix, act_file or None
    (0, "delta", "linear_attn.in_proj_qkv.weight", "L00.f32"),
    (0, "delta", "linear_attn.out_proj.weight", None),
    (32, "delta", "linear_attn.in_proj_qkv.weight", "L32.f32"),
    (32, "delta", "linear_attn.out_proj.weight", None),
    (3, "gqa", "self_attn.q_proj.weight", "L03.f32"),
    (3, "gqa", "self_attn.o_proj.weight", None),
    (3, "gqa", "self_attn.v_proj.weight", "L03.f32"),
    (63, "gqa", "self_attn.q_proj.weight", "L63.f32"),
    (63, "gqa", "self_attn.o_proj.weight", None),
    (63, "gqa", "self_attn.v_proj.weight", "L63.f32"),
]


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def tensor_name(layer: int, suffix: str) -> str:
    return f"language_model.model.layers.{layer}.{suffix}"


def load_index():
    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    return idx["weight_map"]


_HEADER_CACHE = {}


def read_header(shard_path: str):
    if shard_path in _HEADER_CACHE:
        return _HEADER_CACHE[shard_path]
    with open(shard_path, "rb") as f:
        ln = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(ln))
    _HEADER_CACHE[shard_path] = (ln, header)
    return ln, header


def load_bf16(name: str, weight_map) -> np.ndarray:
    shard = os.path.join(SRC, weight_map[name])
    hlen, header = read_header(shard)
    info = header[name]
    dtype = info["dtype"]
    if dtype not in ("BF16", "BFLOAT16"):
        raise RuntimeError(f"{name} dtype {dtype}")
    begin, end = info["data_offsets"]
    off = 8 + hlen + begin
    nbytes = end - begin
    with open(shard, "rb") as f:
        f.seek(off)
        raw = f.read(nbytes)
    u16 = np.frombuffer(raw, dtype="<u2")
    f32 = np.frombuffer((u16.astype(np.uint32) << 16).tobytes(), dtype=np.float32).copy()
    return f32.reshape(info["shape"])


def load_act(fn: str) -> np.ndarray:
    raw = np.fromfile(os.path.join(ACT, fn), dtype=np.float32)
    if raw.size != 256 * 5120:
        raise RuntimeError(f"{fn} size {raw.size}")
    return raw.reshape(256, 5120)


def metrics(src: np.ndarray, rec: np.ndarray) -> dict:
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
        "weight_rmse": ne / (len(a) ** 0.5),
    }


def output_metrics(W: np.ndarray, Wrec: np.ndarray, X: np.ndarray) -> dict:
    # W is [out, in]; X is [T, in]
    Y = X @ W.T
    Yh = X @ Wrec.T
    out = {}
    for tag, sl in (("all", slice(None)), ("fit", slice(0, 192)), ("hold", slice(192, 256))):
        y = Y[sl].astype(np.float64, copy=False)
        yh = Yh[sl].astype(np.float64, copy=False)
        yf = y.reshape(-1)
        yhf = yh.reshape(-1)
        dot = float(np.dot(yf, yhf))
        na = float(np.dot(yf, yf)) ** 0.5
        nb = float(np.dot(yhf, yhf)) ** 0.5
        err = yf - yhf
        ne = float(np.dot(err, err)) ** 0.5
        # per-row cosine min
        yn = np.linalg.norm(y, axis=1)
        yhn = np.linalg.norm(yh, axis=1)
        rc = np.sum(y * yh, axis=1) / (yn * yhn + 1e-30)
        out[f"{tag}_output_cosine"] = dot / (na * nb + 1e-30)
        out[f"{tag}_output_rel_l2"] = ne / (na + 1e-30)
        out[f"{tag}_output_cosine_min_row"] = float(rc.min()) if rc.size else None
    return out


def _group_view(flat: np.ndarray, G: int):
    n = flat.size
    n_pad = ((n + G - 1) // G) * G
    if n_pad == n:
        return flat.reshape(-1, G), n, False
    buf = np.zeros(n_pad, dtype=flat.dtype)
    buf[:n] = flat
    return buf.reshape(-1, G), n, True


def uniform_qn(x: np.ndarray, bits: int, G: int, exclude: np.ndarray | None) -> np.ndarray:
    flat = x.reshape(-1)
    g, n, padded = _group_view(flat, G)
    if exclude is not None:
        eflat = exclude.reshape(-1)
        if padded:
            em = np.zeros(g.size, dtype=bool)
            em[:n] = eflat
            emg = em.reshape(g.shape)
        else:
            emg = eflat.reshape(g.shape)
        work = np.abs(g, dtype=np.float32)
        work[emg] = 0.0
        amax = work.max(axis=1, keepdims=True)
    else:
        emg = None
        amax = np.abs(g).max(axis=1, keepdims=True)
    nlevels = 1 << bits
    qmax = nlevels // 2 - 1
    qmin = -(nlevels // 2)
    scale = np.where(amax <= 0.0, 1.0, amax / qmax).astype(np.float32)
    scale = scale.astype(np.float16).astype(np.float32)
    q = np.rint(g / scale)
    np.clip(q, qmin, qmax, out=q)
    rec = q * scale
    if emg is not None:
        rec = np.where(emg, g, rec)
    return rec.reshape(-1)[:n].reshape(x.shape)


def binary_meanabs(x: np.ndarray, G: int, exclude: np.ndarray | None) -> np.ndarray:
    flat = x.reshape(-1)
    g, n, padded = _group_view(flat, G)
    if exclude is not None:
        eflat = exclude.reshape(-1)
        if padded:
            em = np.zeros(g.size, dtype=bool)
            em[:n] = eflat
            emg = em.reshape(g.shape)
        else:
            emg = eflat.reshape(g.shape)
        abs_g = np.abs(g)
        abs_g[emg] = 0.0
        denom = (~emg).sum(axis=1, keepdims=True).astype(np.float32)
        mean = abs_g.sum(axis=1, keepdims=True) / np.maximum(denom, 1.0)
    else:
        emg = None
        mean = np.abs(g).mean(axis=1, keepdims=True)
    rec = np.sign(g) * mean
    if emg is not None:
        rec = np.where(emg, g, rec)
    return rec.reshape(-1)[:n].reshape(x.shape)


def quantize(x: np.ndarray, spec, exclude: np.ndarray | None) -> np.ndarray:
    kind, G, _bpw = spec[1], spec[2], spec[3]
    if kind is None:
        rec = np.zeros_like(x)
        if exclude is not None:
            rec = np.where(exclude, x, rec)
        return rec
    if kind == "binary":
        return binary_meanabs(x, G, exclude)
    return uniform_qn(x, int(kind), int(G), exclude)


def elias_gamma_bits(v: np.ndarray) -> int:
    # v >= 1
    v = np.maximum(v.astype(np.int64, copy=False), 1)
    # 2*floor(log2(v))+1
    return int((2 * np.floor(np.log2(v)).astype(np.int64) + 1).sum())


def rice_bits(v: np.ndarray, k: int) -> int:
    v = np.maximum(v.astype(np.int64, copy=False), 0)
    q = v >> k
    return int(q.sum() + v.size * (1 + k))


def best_rice_bits(v: np.ndarray) -> tuple[int, int]:
    if v.size == 0:
        return 0, 0
    mean = float(v.mean()) if v.size else 1.0
    k0 = max(0, int(round(np.log2(max(mean, 1.0)))))
    best = None
    for k in range(max(0, k0 - 3), k0 + 5):
        b = rice_bits(v, k)
        if best is None or b < best[0]:
            best = (b, k)
    return best


def index_costs(mask: np.ndarray, n: int) -> dict:
    k = int(mask.sum())
    if k == 0:
        return {
            "k": 0,
            "frac": 0.0,
            "dense_bitmap_bpw": 1.0,
            "occupied_bitmap_g64_bpw": 1.0 / 64.0,
            "occupied_bitmap_g128_bpw": 1.0 / 128.0,
            "delta_elias_gamma_bpw": 0.0,
            "delta_rice_bpw": 0.0,
            "delta_fixed_log2n_bpw": 0.0,
            "group_local_g64_bpw": 8.0 / 64.0,
            "value_bpw_bf16": 0.0,
        }
    pos = np.flatnonzero(mask)
    # deltas: first position as pos+1 so all >= 1
    deltas = np.empty(k, dtype=np.int64)
    deltas[0] = int(pos[0]) + 1
    if k > 1:
        deltas[1:] = np.diff(pos)
    gamma = elias_gamma_bits(deltas)
    rice_b, rice_k = best_rice_bits(deltas)
    log2n = int(np.ceil(np.log2(n)))
    out = {
        "k": k,
        "frac": k / n,
        "dense_bitmap_bpw": 1.0,
        "delta_elias_gamma_bpw": gamma / n,
        "delta_rice_bpw": rice_b / n,
        "delta_rice_k": rice_k,
        "delta_fixed_log2n_bpw": k * log2n / n,
        "value_bpw_bf16": k * VALUE_BITS / n,
        "log2n": log2n,
    }
    for G in G_BITMAP:
        n_groups = (n + G - 1) // G
        # occupancy bit per group + G-bit mask on occupied groups
        occupied = 0
        # count groups with any set bit
        # pad mask
        mpad = np.zeros(n_groups * G, dtype=bool)
        mpad[:n] = mask
        occ = mpad.reshape(n_groups, G).any(axis=1)
        occupied = int(occ.sum())
        bits = n_groups + occupied * G
        out[f"occupied_bitmap_g{G}_bpw"] = bits / n
        out[f"occupied_groups_g{G}"] = occupied
        out[f"n_groups_g{G}"] = n_groups
        # CSR group-local: 8-bit count per group + ceil(log2(G)) local idx
        local_bits = int(np.ceil(np.log2(G)))
        # use 8-bit counts (max 255; G<=128)
        csr_bits = n_groups * 8 + k * local_bits
        out[f"group_local_g{G}_bpw"] = csr_bits / n
    return out


def structure_report(mask: np.ndarray, shape: tuple[int, int]) -> dict:
    rows, cols = shape
    m = mask.reshape(rows, cols)
    k = int(m.sum())
    row_c = m.sum(axis=1)
    col_c = m.sum(axis=0)
    if k == 0:
        return {
            "k": 0,
            "rows_occupied": 0,
            "cols_occupied": 0,
            "frac_rows_occupied": 0.0,
            "frac_cols_occupied": 0.0,
        }
    row_order = np.argsort(-row_c)
    col_order = np.argsort(-col_c)

    def share(counts, order, frac):
        n_take = max(1, int(np.ceil(frac * counts.size)))
        return float(counts[order[:n_take]].sum() / k)

    def cover_frac(counts, order, target):
        csum = np.cumsum(counts[order])
        need = int(np.searchsorted(csum, target * k, side="left")) + 1
        return min(need, counts.size)

    # gini of row counts among occupied
    occ_rows = row_c[row_c > 0]
    if occ_rows.size:
        s = np.sort(occ_rows.astype(np.float64))
        n = s.size
        gini = float((2.0 * np.arange(1, n + 1) * s).sum() / (n * s.sum()) - (n + 1) / n)
    else:
        gini = 0.0
    return {
        "k": k,
        "rows_occupied": int((row_c > 0).sum()),
        "cols_occupied": int((col_c > 0).sum()),
        "frac_rows_occupied": float((row_c > 0).mean()),
        "frac_cols_occupied": float((col_c > 0).mean()),
        "max_per_row": int(row_c.max()),
        "max_per_col": int(col_c.max()),
        "mean_per_occupied_row": float(occ_rows.mean()) if occ_rows.size else 0.0,
        "row_gini": gini,
        "row_share_hottest_0p1pct": share(row_c, row_order, 0.001),
        "row_share_hottest_1pct": share(row_c, row_order, 0.01),
        "row_share_hottest_5pct": share(row_c, row_order, 0.05),
        "row_share_hottest_10pct": share(row_c, row_order, 0.10),
        "col_share_hottest_0p1pct": share(col_c, col_order, 0.001),
        "col_share_hottest_1pct": share(col_c, col_order, 0.01),
        "col_share_hottest_5pct": share(col_c, col_order, 0.05),
        "col_share_hottest_10pct": share(col_c, col_order, 0.10),
        "n_rows_to_cover_50pct_outliers": cover_frac(row_c, row_order, 0.50),
        "n_rows_to_cover_90pct_outliers": cover_frac(row_c, row_order, 0.90),
        "n_rows_to_cover_99pct_outliers": cover_frac(row_c, row_order, 0.99),
        "n_cols_to_cover_50pct_outliers": cover_frac(col_c, col_order, 0.50),
        "n_cols_to_cover_90pct_outliers": cover_frac(col_c, col_order, 0.90),
        "n_cols_to_cover_99pct_outliers": cover_frac(col_c, col_order, 0.99),
        "n_rows": rows,
        "n_cols": cols,
    }


def weight_stats(W: np.ndarray) -> dict:
    a = np.abs(W.reshape(-1))
    med = float(np.median(a))
    return {
        "shape": list(W.shape),
        "elements": int(W.size),
        "min": float(W.min()),
        "max": float(W.max()),
        "mean": float(W.mean()),
        "std": float(W.std()),
        "rms": float(np.sqrt(np.mean(W.astype(np.float64) ** 2))),
        "median_abs": med,
        "p90_abs": float(np.quantile(a, 0.90)),
        "p99_abs": float(np.quantile(a, 0.99)),
        "p999_abs": float(np.quantile(a, 0.999)),
        "p9999_abs": float(np.quantile(a, 0.9999)),
        "max_abs": float(a.max()),
        "dynamic_range_max_over_median": float(a.max() / med) if med > 0 else None,
        "frac_abs_gt_10x_median": float((a > 10.0 * med).mean()) if med > 0 else None,
        "frac_abs_gt_100x_median": float((a > 100.0 * med).mean()) if med > 0 else None,
        "excess_kurtosis": float(
            np.mean(((W.reshape(-1) - W.mean()) / (W.std() + 1e-30)) ** 4) - 3.0
        ),
    }


def topk_mask_from_score(score: np.ndarray, frac: float) -> np.ndarray:
    n = score.size
    k = int(round(frac * n))
    mask = np.zeros(n, dtype=bool)
    if k <= 0:
        return mask.reshape(score.shape)
    if k >= n:
        mask[:] = True
        return mask.reshape(score.shape)
    # argpartition
    flat = score.reshape(-1)
    idx = np.argpartition(flat, -k)[-k:]
    mask[idx] = True
    return mask.reshape(score.shape)


def r4(x):
    if isinstance(x, float):
        return float(f"{x:.8g}")
    return x


def compact(d):
    if isinstance(d, dict):
        return {k: compact(v) for k, v in d.items()}
    if isinstance(d, list):
        return [compact(v) for v in d]
    if isinstance(d, float):
        return r4(d)
    if isinstance(d, (np.floating,)):
        return r4(float(d))
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


def process_tensor(layer, kind, suffix, act_fn, weight_map, X_cache):
    name = tensor_name(layer, suffix)
    t0 = time.time()
    W = load_bf16(name, weight_map)
    assert W.ndim == 2
    stats = weight_stats(W)
    X = None
    if act_fn is not None and W.shape[1] == 5120:
        if act_fn not in X_cache:
            X_cache[act_fn] = load_act(act_fn)
        X = X_cache[act_fn]
    rows, cols = W.shape
    n = W.size
    absW = np.abs(W)

    # precompute base reconstructions at frac=0 for residual scores
    base_recons = {}
    for spec in BASES:
        rec0 = quantize(W, spec, None)
        base_recons[spec[0]] = rec0

    curves = []
    # selection policies
    for sel_name, score_of in (
        ("absW", lambda rec0: absW),
        ("residual", lambda rec0: np.abs(W - rec0)),
    ):
        for spec in BASES:
            rec0 = base_recons[spec[0]]
            score = score_of(rec0)
            for mode in ("overlay", "refit"):
                if spec[0] == "none" and mode == "overlay":
                    # overlay on zeros == refit on zeros
                    if sel_name != "absW":
                        continue
                for frac in FRACS:
                    if frac == 0.0 and (mode == "refit" or sel_name == "residual"):
                        # identical to overlay/absW at 0
                        if not (sel_name == "absW" and mode == "overlay"):
                            continue
                    mask = topk_mask_from_score(score, frac)
                    if mode == "overlay":
                        rec = rec0.copy()
                        rec[mask] = W[mask]
                    else:
                        rec = quantize(W, spec, mask)
                    m = metrics(W, rec)
                    row = {
                        "tensor": name,
                        "layer": layer,
                        "kind": kind,
                        "suffix": suffix,
                        "base": spec[0],
                        "base_bpw": spec[3],
                        "selection": sel_name,
                        "mode": mode,
                        "frac": frac,
                        **m,
                    }
                    if X is not None:
                        row.update(output_metrics(W, rec, X))
                    ic = index_costs(mask.reshape(-1), n)
                    row["index"] = ic
                    # complete scheme bpw for each encoding
                    val = ic["value_bpw_bf16"]
                    encodings = {
                        "dense_bitmap": spec[3] + ic["dense_bitmap_bpw"] + val,
                        "occupied_bitmap_g64": spec[3]
                        + ic.get("occupied_bitmap_g64_bpw", 0.0)
                        + val,
                        "occupied_bitmap_g128": spec[3]
                        + ic.get("occupied_bitmap_g128_bpw", 0.0)
                        + val,
                        "delta_elias_gamma": spec[3]
                        + ic.get("delta_elias_gamma_bpw", 0.0)
                        + val,
                        "delta_rice": spec[3] + ic.get("delta_rice_bpw", 0.0) + val,
                        "delta_fixed_log2n": spec[3]
                        + ic.get("delta_fixed_log2n_bpw", 0.0)
                        + val,
                        "group_local_g64": spec[3]
                        + ic.get("group_local_g64_bpw", 0.0)
                        + val,
                    }
                    row["scheme_bpw"] = encodings
                    if frac in (1e-4, 1e-3, 1e-2, 3e-2) and sel_name == "absW" and mode == "refit":
                        row["structure"] = structure_report(mask, W.shape)
                    elif frac in (1e-3, 1e-2) and sel_name == "absW" and mode == "overlay":
                        row["structure"] = structure_report(mask, W.shape)
                    curves.append(row)
                    del rec
            del score

    # fixed-slot-per-group (local top-S by |W|, refit, binary + q2)
    fixed_rows = []
    for spec in BASES:
        if spec[0] not in ("binary_g128", "uniform_q2_g64", "none"):
            continue
        G = FIXED_G
        g, nn, padded = _group_view(absW.reshape(-1), G)
        # argsort per group
        order = np.argsort(-g, axis=1)  # largest first
        for S in FIXED_SLOTS:
            mask_g = np.zeros_like(g, dtype=bool)
            take = order[:, :S]
            rows_i = np.arange(g.shape[0])[:, None]
            mask_g[rows_i, take] = True
            # if a slot is padding, ignore
            if padded:
                # last group padded zeros should not be "kept" unless they are real
                last = g.shape[0] - 1
                valid = nn - last * G
                mask_g[last, valid:] = False
            mask = mask_g.reshape(-1)[:n].reshape(W.shape)
            rec = quantize(W, spec, mask)
            m = metrics(W, rec)
            k = int(mask.sum())
            local_bits = int(np.ceil(np.log2(G)))
            n_groups = g.shape[0]
            # pay S slots always
            index_bits = n_groups * S * local_bits
            value_bits = n_groups * S * VALUE_BITS
            # if last group has fewer real slots, still charged (fixed)
            row = {
                "tensor": name,
                "layer": layer,
                "kind": kind,
                "suffix": suffix,
                "base": spec[0],
                "base_bpw": spec[3],
                "S": S,
                "G": G,
                "k_actual": k,
                "frac_actual": k / n,
                **m,
                "index_bpw": index_bits / n,
                "value_bpw": value_bits / n,
                "scheme_bpw": spec[3] + (index_bits + value_bits) / n,
            }
            if X is not None:
                row.update(output_metrics(W, rec, X))
            fixed_rows.append(row)
            del rec, mask

    # row-island: keep hottest R output rows fully exact, refit binary/q2
    row_islands = []
    row_rms = np.sqrt(np.mean(W.astype(np.float64) ** 2, axis=1))
    row_order = np.argsort(-row_rms)
    for spec in BASES:
        if spec[0] not in ("binary_g128", "uniform_q2_g64"):
            continue
        for rfrac in (0.001, 0.003, 0.01, 0.03, 0.10):
            R = max(1, int(round(rfrac * rows)))
            mask = np.zeros(W.shape, dtype=bool)
            mask[row_order[:R], :] = True
            rec = quantize(W, spec, mask)
            m = metrics(W, rec)
            # index: list of row ids (log2(rows) each) + those rows stored bf16
            idx_bpw = R * int(np.ceil(np.log2(rows))) / n
            val_bpw = R * cols * VALUE_BITS / n
            row = {
                "tensor": name,
                "base": spec[0],
                "base_bpw": spec[3],
                "row_frac": rfrac,
                "R": R,
                **m,
                "index_bpw": idx_bpw,
                "value_bpw": val_bpw,
                "scheme_bpw": spec[3] + idx_bpw + val_bpw,
            }
            if X is not None:
                row.update(output_metrics(W, rec, X))
            row_islands.append(row)
            del rec, mask

    rec_q4 = uniform_qn(W, 4, 64, None)
    q4 = metrics(W, rec_q4)
    if X is not None:
        q4.update(output_metrics(W, rec_q4, X))
    q4["base_bpw"] = 4.0 + 16.0 / 64.0

    elapsed = time.time() - t0
    result = {
        "name": name,
        "layer": layer,
        "kind": kind,
        "suffix": suffix,
        "act": act_fn,
        "stats": stats,
        "q4_g64": q4,
        "curves": curves,
        "fixed_slot": fixed_rows,
        "row_islands": row_islands,
        "wall_s": elapsed,
        "rss_mb_after": rss_mb(),
    }
    del W, absW, rec_q4
    for v in base_recons.values():
        del v
    gc.collect()
    return result


def main():
    t0 = time.time()
    weight_map = load_index()
    X_cache = {}
    tensors = []
    print(f"start rss_mb={rss_mb():.1f}", flush=True)
    for layer, kind, suffix, act_fn in TENSORS:
        name = tensor_name(layer, suffix)
        print(f"== {name} ==", flush=True)
        r = process_tensor(layer, kind, suffix, act_fn, weight_map, X_cache)
        print(
            f"   shape={r['stats']['shape']} kurt={r['stats']['excess_kurtosis']:.3f} "
            f"q4_cos={r['q4_g64']['weight_cosine']:.6f} wall={r['wall_s']:.1f}s rss={r['rss_mb_after']:.1f}",
            flush=True,
        )
        tensors.append(r)
        gc.collect()

    payload = {
        "schema": "hawking.g1.sparse_exact_islands.v1",
        "date": "2026-08-17",
        "source_dir": SRC,
        "activation_dir": ACT,
        "machine": "Apple M3 Ultra, CPU-only numpy, no GPU",
        "claim_boundary": {
            "not_a_kernel": True,
            "not_a_gpu_benchmark": True,
            "not_a_generation_coherence_claim": True,
            "used_real_bf16_weights": True,
            "used_real_captured_hiddens_where_in_dim_matches": True,
            "out_proj_weight_space_only": True,
            "exact_island_values_costed_as_bf16": True,
        },
        "method": {
            "q4_family": "asymmetric absmax group-64, scale=max_abs/7 in fp16, codes rint clamp [-8,7] (matches pack_uniform_q4_group64)",
            "qn_family": "same family, bits=2/3, qmax=2^(b-1)-1, qmin=-2^(b-1)",
            "binary": "sign * group mean-abs, G=128 (HGRAVB01)",
            "overlay": "quantize all, then restore selected entries to source f32",
            "refit": "exclude selected entries from group scale/mean, restore them exact",
            "selection_absW": "global top-k by |W|",
            "selection_residual": "global top-k by |W-Q0(W)| on the unpatched base",
            "output_site": "X = captured post-norm hidden [256,5120]; Y=X@W.T; fit=0:192 hold=192:256",
        },
        "peak_rss_mb": rss_mb(),
        "wall_s": time.time() - t0,
        "tensors": compact(tensors),
    }
    with open(OUT, "w") as f:
        json.dump(payload, f)
    print(f"wrote {OUT} bytes={os.path.getsize(OUT)} wall={payload['wall_s']:.1f}s peak_rss_mb={payload['peak_rss_mb']:.1f}")


if __name__ == "__main__":
    main()

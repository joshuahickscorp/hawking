#!/usr/bin/env python3
"""G1 residual-channel 3994 island: existence, generality, payoff, kernel map.

CPU / numpy only. No GPU, no generate, no pack, no live-organism contact.
Writes /tmp/g1_channel_3994_island.json and prints a machine-readable log.
"""
from __future__ import annotations

import gc
import json
import os
import resource
import struct
import time
from pathlib import Path

import numpy as np

SRC = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16")
CAP = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1")
OUT = Path("/tmp/g1_channel_3994_island.json")

HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
N_LAYERS = 64
N_TOKENS = 256
KEY_HEADS = 16
VALUES_PER_KEY = 3
KEY_DIM = 128
VALUE_DIM = 128
GQA_HEADS = 24
GQA_KV = 4
GQA_HEAD_DIM = 256
GROUP = 64
EPS = 1.0e-6
ISLAND0 = 3994
SOURCE_ELEMS = 26_895_998_464
G0_BPW = 4.252735126866492

LAYERS = (0, 3, 6, 15, 32, 47, 63)
K_SWEEP = (0, 1, 2, 3, 4, 8, 16, 32)
BITS = (2, 3)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_gb():.3f}G {msg}", flush=True)


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


def excess_kurtosis(flat: np.ndarray) -> float:
    x = np.asarray(flat, dtype=np.float64).reshape(-1)
    c = x - float(np.mean(x))
    m2 = float(np.mean(c * c))
    m4 = float(np.mean(c * c * c * c))
    return (m4 / (m2 * m2) - 3.0) if m2 > 0 else 0.0


_HEADER_CACHE: dict[Path, dict] = {}
_WMAP = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]


def read_header(shard: Path) -> dict:
    if shard not in _HEADER_CACHE:
        with shard.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            _HEADER_CACHE[shard] = json.loads(fh.read(n))
    return _HEADER_CACHE[shard]


def tensor_info(name: str) -> tuple[Path, dict, int]:
    shard = SRC / _WMAP[name]
    header = read_header(shard)
    info = header[name]
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
    return shard, info, 8 + n


def load_tensor(name: str) -> np.ndarray:
    shard, info, data_start = tensor_info(name)
    dtype = info.get("dtype", "BF16")
    shape = tuple(int(x) for x in info["shape"])
    lo, hi = info["data_offsets"]
    with shard.open("rb") as fh:
        fh.seek(data_start + lo)
        raw = fh.read(hi - lo)
    if dtype in ("BF16", "BFLOAT16"):
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return np.ascontiguousarray(u32.view(np.float32).reshape(shape))
    if dtype in ("F32", "FLOAT32"):
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    raise RuntimeError(f"unsupported dtype {dtype} for {name}")


def load_row(name: str, row: int) -> np.ndarray:
    shard, info, data_start = tensor_info(name)
    dtype = info.get("dtype", "BF16")
    shape = tuple(int(x) for x in info["shape"])
    itemsize = 2 if dtype in ("BF16", "BFLOAT16", "F16") else 4
    cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    lo, _ = info["data_offsets"]
    with shard.open("rb") as fh:
        fh.seek(data_start + lo + row * cols * itemsize)
        raw = fh.read(cols * itemsize)
    if dtype in ("BF16", "BFLOAT16"):
        u16 = np.frombuffer(raw, dtype=np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32).copy()
    if dtype in ("F32", "FLOAT32"):
        return np.frombuffer(raw, dtype=np.float32).copy()
    raise RuntimeError(f"unsupported dtype {dtype}")


def load_column(name: str, col: int) -> np.ndarray:
    shard, info, data_start = tensor_info(name)
    dtype = info.get("dtype", "BF16")
    shape = tuple(int(x) for x in info["shape"])
    itemsize = 2 if dtype in ("BF16", "BFLOAT16", "F16") else 4
    rows = int(shape[0])
    cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    lo, _ = info["data_offsets"]
    out = np.empty(rows, dtype=np.float32)
    with shard.open("rb") as fh:
        base = data_start + lo
        if dtype in ("BF16", "BFLOAT16"):
            for r in range(rows):
                fh.seek(base + (r * cols + col) * itemsize)
                u16 = np.frombuffer(fh.read(2), dtype=np.uint16)[0]
                out[r] = np.array([u16], dtype=np.uint16).astype(np.uint32).__ilshift__(16).view(np.float32)[0]
        elif dtype in ("F32", "FLOAT32"):
            for r in range(rows):
                fh.seek(base + (r * cols + col) * itemsize)
                out[r] = np.frombuffer(fh.read(4), dtype=np.float32)[0]
        else:
            raise RuntimeError(dtype)
    return out


def load_column_fast(name: str, col: int) -> np.ndarray:
    """Read one column via full-tensor mmap-ish frombuffer; faster than per-row seek on SSD."""
    W = load_tensor(name)
    col_v = np.ascontiguousarray(W[:, col])
    del W
    return col_v


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


def uniform_absmax_recon(W: np.ndarray, bits: int, group_size: int = GROUP) -> np.ndarray:
    bound = (1 << (bits - 1)) - 1
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    n = int(flat.size)
    groups = (n + group_size - 1) // group_size
    padded = np.zeros((groups, group_size), dtype=np.float32)
    padded.reshape(-1)[:n] = flat
    absmax = np.max(np.abs(padded), axis=1)
    scale = absmax / max(bound, 1)
    den = np.where(scale > 0.0, scale, 1.0)
    codes = np.rint(padded / den[:, None]).clip(-bound, bound)
    recon = (codes * scale[:, None]).reshape(-1)[:n]
    return recon.reshape(W.shape).astype(np.float32)


def uniform_absmax_refit_zero_cols(W: np.ndarray, bits: int, cols: np.ndarray, group_size: int = GROUP) -> np.ndarray:
    """Quantize after zeroing selected input columns, then restore those columns exactly."""
    body = np.array(W, dtype=np.float32, copy=True)
    if cols.size:
        body[:, cols] = 0.0
    recon = uniform_absmax_recon(body, bits, group_size)
    if cols.size:
        recon[:, cols] = W[:, cols]
    return recon


def rmsnorm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Match qwen80_residual_rmsnorm_f32: out = x * rsqrt(mean(x^2)+eps) * (1+w)."""
    ms = np.mean(np.square(x, dtype=np.float64), axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + EPS)
    return (x * inv * (1.0 + w.astype(np.float64))).astype(np.float32)


def deltanet_out_proxy(X: np.ndarray, W_qkv: np.ndarray, W_z: np.ndarray) -> np.ndarray:
    yq = X @ W_qkv.T
    yz = X @ W_z.T
    value_rows = VALUES_PER_KEY * VALUE_DIM
    v = yq[:, KEY_HEADS * KEY_DIM * 2 :].reshape(X.shape[0], -1)
    z = yz
    return np.ascontiguousarray(v * silu(z), dtype=np.float32)


def gqa_out_proxy(X: np.ndarray, W_q: np.ndarray, W_v: np.ndarray) -> np.ndarray:
    qg = X @ W_q.T
    v = X @ W_v.T
    qg = qg.reshape(X.shape[0], GQA_HEADS, 2, GQA_HEAD_DIM)
    gate = sigmoid(qg[:, :, 1, :])
    v = v.reshape(X.shape[0], GQA_KV, GQA_HEAD_DIM)
    v_rep = np.repeat(v, GQA_HEADS // GQA_KV, axis=1)
    return np.ascontiguousarray((v_rep * gate).reshape(X.shape[0], GQA_HEADS * GQA_HEAD_DIM), dtype=np.float32)


def down_x(X: np.ndarray, layer: int) -> np.ndarray:
    Wg = load_tensor(tname(layer, "mlp.gate_proj.weight"))
    g = X @ Wg.T
    del Wg
    Wu = load_tensor(tname(layer, "mlp.up_proj.weight"))
    u = X @ Wu.T
    del Wu
    gc.collect()
    return np.ascontiguousarray(silu(g) * u, dtype=np.float32)


def hold_idx() -> np.ndarray:
    return np.arange(N_TOKENS) % 2 == 1


def score_pair(Y: np.ndarray, Yq: np.ndarray, hold: np.ndarray, residual_out: bool) -> dict:
    err = (Y - Yq).astype(np.float64)
    mse = np.mean(np.square(err))
    ch_err = np.sqrt(np.mean(np.square(err, dtype=np.float64), axis=0))
    ch_true = np.sqrt(np.mean(np.square(Y, dtype=np.float64), axis=0))
    rel = np.divide(ch_err, np.maximum(ch_true, 1e-20))
    mse_share = np.square(ch_err) / max(float(np.sum(np.square(ch_err))), 1e-30)
    rec = {
        "hold_cosine": mean_row_cosine(Y[hold], Yq[hold]),
        "hold_min_cosine": min_row_cosine(Y[hold], Yq[hold]),
        "all_cosine": mean_row_cosine(Y, Yq),
        "rel_l2": rel_l2(Y, Yq),
        "mse": float(mse),
        "ch_rel_err_median": float(np.median(rel)),
        "ch_rel_err_max": float(np.max(rel)),
        "ch_rel_err_max_ch": int(np.argmax(rel)),
    }
    # Residual channel 3994 is an OUTPUT index only on 5120-wide write tensors.
    if residual_out and Y.shape[1] == HIDDEN:
        rec["ch3994_rel_err"] = float(rel[ISLAND0])
        rec["ch3994_mse_share"] = float(mse_share[ISLAND0])
    else:
        rec["ch3994_rel_err"] = None
        rec["ch3994_mse_share"] = None
    return rec


# ---------------------------------------------------------------------------
# 1. Activation census
# ---------------------------------------------------------------------------

def activation_census(meta: dict) -> dict:
    log("activation census start")
    prompts = meta["prompts"]
    bounds = []
    off = 0
    for p in prompts:
        n = int(p["n_tokens"])
        bounds.append((off, off + n, p["prompt"]))
        off += n
    all_rms = np.zeros((N_LAYERS, HIDDEN), dtype=np.float64)
    layers = []
    tok_top1 = np.zeros(N_LAYERS, dtype=np.int32)
    tok_hot10 = np.zeros(N_LAYERS, dtype=np.int32)
    half_agree = []
    prompt_ranks = []
    for li in range(N_LAYERS):
        x = load_hidden(li).astype(np.float64, copy=False)
        ch_rms = np.sqrt(np.mean(np.square(x), axis=0))
        all_rms[li] = ch_rms
        med = float(np.median(ch_rms))
        order = np.argsort(ch_rms)[::-1]
        rank = np.empty(HIDDEN, dtype=np.int32)
        rank[order] = np.arange(1, HIDDEN + 1)
        energy = np.square(ch_rms)
        energy_frac = energy / max(float(energy.sum()), 1e-30)
        rec = {
            "layer": li,
            "rms": float(np.sqrt(np.mean(np.square(x)))),
            "ch_rms_median": med,
            "ch_max_over_median": float(ch_rms.max() / med) if med else None,
            "n_hot4": int(np.sum(ch_rms > 4.0 * med)),
            "n_hot10": int(np.sum(ch_rms > 10.0 * med)),
            "n_hot20": int(np.sum(ch_rms > 20.0 * med)),
            "top5": [{"ch": int(i), "rms": float(ch_rms[i]), "xmed": float(ch_rms[i] / med), "energy_frac": float(energy_frac[i])} for i in order[:5]],
            "ch3994": {
                "rms": float(ch_rms[ISLAND0]),
                "xmed": float(ch_rms[ISLAND0] / med) if med else None,
                "rank": int(rank[ISLAND0]),
                "energy_frac": float(energy_frac[ISLAND0]),
                "hot4": bool(ch_rms[ISLAND0] > 4.0 * med),
                "hot10": bool(ch_rms[ISLAND0] > 10.0 * med),
            },
            "tracked": {},
        }
        for ch in (3994, 3456, 310, 3842, 1689, 2631, 3532):
            rec["tracked"][str(ch)] = {
                "rms": float(ch_rms[ch]),
                "xmed": float(ch_rms[ch] / med) if med else None,
                "rank": int(rank[ch]),
                "energy_frac": float(energy_frac[ch]),
            }
        # per-token
        tok_abs = np.abs(x)
        tok_med = np.median(tok_abs, axis=1)
        tok_rank1 = np.argmax(tok_abs, axis=1)
        tok_top1[li] = int(np.sum(tok_rank1 == ISLAND0))
        tok_hot10[li] = int(np.sum(tok_abs[:, ISLAND0] > 10.0 * tok_med))
        rec["ch3994"]["n_tokens_top1"] = int(tok_top1[li])
        rec["ch3994"]["n_tokens_hot10"] = int(tok_hot10[li])
        # half split even/odd tokens
        for label, sl in (("even", x[0::2]), ("odd", x[1::2]), ("first128", x[:128]), ("last128", x[128:])):
            r = np.sqrt(np.mean(np.square(sl), axis=0))
            m = float(np.median(r))
            rec[f"split_{label}_ch3994_rank"] = int(np.sum(r > r[ISLAND0]) + 1)
            rec[f"split_{label}_ch3994_xmed"] = float(r[ISLAND0] / m) if m else None
            rec[f"split_{label}_top1"] = int(np.argmax(r))
        half_agree.append(rec["split_first128_top1"] == rec["split_last128_top1"])
        # prompt split
        pr = []
        for a, b, prompt in bounds:
            r = np.sqrt(np.mean(np.square(x[a:b]), axis=0))
            m = float(np.median(r))
            pr.append({
                "span": [a, b],
                "prompt": prompt[:48],
                "ch3994_rank": int(np.sum(r > r[ISLAND0]) + 1),
                "ch3994_xmed": float(r[ISLAND0] / m) if m else None,
                "top1": int(np.argmax(r)),
            })
        rec["per_prompt"] = pr
        # shared prefix (first 45 tokens of capture are the common system template on prompts 0-3)
        pref = x[:45]
        r = np.sqrt(np.mean(np.square(pref), axis=0))
        m = float(np.median(r))
        rec["prefix45_ch3994_rank"] = int(np.sum(r > r[ISLAND0]) + 1)
        rec["prefix45_ch3994_xmed"] = float(r[ISLAND0] / m) if m else None
        rec["prefix45_top1"] = int(np.argmax(r))
        layers.append(rec)
        log(
            f"act L{li:02d} rms={rec['rms']:.4f} 3994 rank={rec['ch3994']['rank']} "
            f"xmed={rec['ch3994']['xmed']:.2f} efrac={rec['ch3994']['energy_frac']:.4f} "
            f"tok_top1={tok_top1[li]}/256 tok_hot10={tok_hot10[li]}/256"
        )
        del x
    med_per = np.median(all_rms, axis=1, keepdims=True)
    hot4 = all_rms > 4.0 * med_per
    hot10 = all_rms > 10.0 * med_per
    persist4 = hot4.sum(axis=0)
    persist10 = hot10.sum(axis=0)
    mean_rms = all_rms.mean(axis=0)
    rank_mean = np.argsort(mean_rms)[::-1]
    top_persist = [
        {
            "ch": int(i),
            "n_hot4": int(persist4[i]),
            "n_hot10": int(persist10[i]),
            "mean_rms": float(mean_rms[i]),
            "mean_energy_frac": float((np.square(all_rms[:, i]) / np.maximum(np.square(all_rms).sum(axis=1), 1e-30)).mean()),
        }
        for i in np.argsort(persist4)[::-1][:12]
    ]
    act_rank_list = [int(i) for i in rank_mean[:64]]
    log(f"act persist hot10>=4: {int((persist10 >= 4).sum())} channels; top persist {top_persist[:5]}")
    log(f"act mean-rms rank[:8]={act_rank_list[:8]}")
    return {
        "n_tokens": N_TOKENS,
        "hidden": HIDDEN,
        "n_layers": N_LAYERS,
        "site": meta.get("status"),
        "schema": meta.get("schema"),
        "sha256_self": meta.get("sha256_self"),
        "prompts": [p.get("prompt") for p in prompts],
        "prompt_bounds": [{"lo": a, "hi": b, "prompt": p} for a, b, p in bounds],
        "layers": layers,
        "cross_layer": {
            "n_hot4_ge8": int((persist4 >= 8).sum()),
            "n_hot4_ge16": int((persist4 >= 16).sum()),
            "n_hot10_ge4": int((persist10 >= 4).sum()),
            "n_hot10_ge8": int((persist10 >= 8).sum()),
            "max_hot4": int(persist4.max()),
            "max_hot10": int(persist10.max()),
            "ch3994_n_hot4": int(persist4[ISLAND0]),
            "ch3994_n_hot10": int(persist10[ISLAND0]),
            "ch3994_mean_rms": float(mean_rms[ISLAND0]),
            "ch3994_mean_rank": float(np.mean([layers[i]["ch3994"]["rank"] for i in range(N_LAYERS)])),
            "top_persist_hot4": top_persist,
            "act_rank_by_mean_rms": act_rank_list,
            "n_layers_first128_last128_top1_agree": int(sum(half_agree)),
            "n_tokens_3994_is_top1_mean": float(tok_top1.mean()),
            "n_tokens_3994_hot10_mean": float(tok_hot10.mean()),
        },
    }


# ---------------------------------------------------------------------------
# 2. Weight-native sample + column 3994 on measured layers
# ---------------------------------------------------------------------------

def write_row_stats(W: np.ndarray) -> dict:
    out_rms = np.sqrt(np.mean(np.square(W, dtype=np.float64), axis=1))
    med = float(np.median(out_rms))
    order = np.argsort(out_rms)[::-1]
    rank = np.empty(W.shape[0], dtype=np.int32)
    rank[order] = np.arange(1, W.shape[0] + 1)
    row = W[ISLAND0].astype(np.float64)
    body = np.delete(W, ISLAND0, axis=0)
    return {
        "shape": [int(x) for x in W.shape],
        "excess_kurtosis": excess_kurtosis(W),
        "excess_kurtosis_drop_row3994": excess_kurtosis(body),
        "row3994_rms": float(out_rms[ISLAND0]),
        "row3994_xmed": float(out_rms[ISLAND0] / med) if med else None,
        "row3994_rank": int(rank[ISLAND0]),
        "row3994_mean_abs": float(np.mean(np.abs(row))),
        "row3994_max_abs": float(np.max(np.abs(row))),
        "out_rms_median": med,
        "out_n_hot4": int(np.sum(out_rms > 4.0 * med)),
        "out_n_hot10": int(np.sum(out_rms > 10.0 * med)),
        "top5": [{"ch": int(i), "rms": float(out_rms[i]), "xmed": float(out_rms[i] / med)} for i in order[:5]],
        "frac_frob_in_row3994": float(np.sum(np.square(row)) / max(float(np.sum(np.square(W, dtype=np.float64))), 1e-30)),
    }


def read_col_stats(W: np.ndarray) -> dict:
    in_rms = np.sqrt(np.mean(np.square(W, dtype=np.float64), axis=0))
    med = float(np.median(in_rms))
    order = np.argsort(in_rms)[::-1]
    rank = np.empty(W.shape[1], dtype=np.int32)
    rank[order] = np.arange(1, W.shape[1] + 1)
    col = W[:, ISLAND0].astype(np.float64)
    return {
        "shape": [int(x) for x in W.shape],
        "col3994_rms": float(in_rms[ISLAND0]),
        "col3994_xmed": float(in_rms[ISLAND0] / med) if med else None,
        "col3994_rank": int(rank[ISLAND0]),
        "col3994_mean_abs": float(np.mean(np.abs(col))),
        "col3994_max_abs": float(np.max(np.abs(col))),
        "in_rms_median": med,
        "in_n_hot4": int(np.sum(in_rms > 4.0 * med)),
        "in_n_hot10": int(np.sum(in_rms > 10.0 * med)),
        "in_max_over_median": float(in_rms.max() / med) if med else None,
        "in_max_ch": int(np.argmax(in_rms)),
        "frac_frob_in_col3994": float(np.sum(np.square(col)) / max(float(np.sum(np.square(W, dtype=np.float64))), 1e-30)),
        "top5_in": [{"ch": int(i), "rms": float(in_rms[i]), "xmed": float(in_rms[i] / med)} for i in order[:5]],
    }


def weight_sample() -> dict:
    log("weight sample start")
    writes = []
    reads = []
    for layer in LAYERS:
        gqa = is_gqa(layer)
        w_specs = [
            ("down", tname(layer, "mlp.down_proj.weight")),
            ("out", tname(layer, "self_attn.o_proj.weight" if gqa else "linear_attn.out_proj.weight")),
        ]
        r_specs = [
            ("gate", tname(layer, "mlp.gate_proj.weight")),
            ("up", tname(layer, "mlp.up_proj.weight")),
            ("in", tname(layer, "self_attn.q_proj.weight" if gqa else "linear_attn.in_proj_qkv.weight")),
        ]
        for cls, name in w_specs:
            W = load_tensor(name)
            rec = {"layer": layer, "class": cls, "name": name, **write_row_stats(W), **{f"in_{k}": v for k, v in read_col_stats(W).items() if k != "shape"}}
            writes.append(rec)
            log(
                f"Wwrite L{layer}.{cls} kurt={rec['excess_kurtosis']:.2f} "
                f"drop3994={rec['excess_kurtosis_drop_row3994']:.2f} "
                f"row3994 rank={rec['row3994_rank']} xmed={rec['row3994_xmed']:.3f} "
                f"frob={rec['frac_frob_in_row3994']:.5f}"
            )
            del W
            gc.collect()
        for cls, name in r_specs:
            W = load_tensor(name)
            rec = {"layer": layer, "class": cls, "name": name, **read_col_stats(W)}
            reads.append(rec)
            log(
                f"Wread  L{layer}.{cls} col3994 rank={rec['col3994_rank']} "
                f"xmed={rec['col3994_xmed']:.3f} frob={rec['frac_frob_in_col3994']:.5f} "
                f"in_max_ch={rec['in_max_ch']}"
            )
            del W
            gc.collect()
    # norms + embed col + lm_head col
    norms = []
    for layer in range(N_LAYERS):
        for kind, suffix in (("input", "input_layernorm.weight"), ("post", "post_attention_layernorm.weight")):
            w = load_tensor(tname(layer, suffix))
            med = float(np.median(np.abs(w)))
            norms.append({
                "layer": layer,
                "kind": kind,
                "w3994": float(w[ISLAND0]),
                "w3994_over_median_abs": float(abs(w[ISLAND0]) / med) if med else None,
                "rank_abs": int(np.sum(np.abs(w) > abs(w[ISLAND0])) + 1),
                "median_abs": med,
                "max_abs": float(np.max(np.abs(w))),
                "max_ch": int(np.argmax(np.abs(w))),
            })
    w = load_tensor("language_model.model.norm.weight")
    med = float(np.median(np.abs(w)))
    final_norm = {
        "w3994": float(w[ISLAND0]),
        "w3994_over_median_abs": float(abs(w[ISLAND0]) / med) if med else None,
        "rank_abs": int(np.sum(np.abs(w) > abs(w[ISLAND0])) + 1),
        "median_abs": med,
        "formula": "out = x * rsqrt(mean(x^2)+eps) * (1+w)   [qwen80_residual_rmsnorm_f32]",
    }
    log(f"final_norm w3994={final_norm['w3994']:.6f} rank_abs={final_norm['rank_abs']}")
    # embed / lm_head input column 3994 vs all (chunked)
    def table_col_stats(name: str) -> dict:
        shard, info, data_start = tensor_info(name)
        shape = tuple(int(x) for x in info["shape"])
        rows, cols = int(shape[0]), int(shape[1])
        lo, _ = info["data_offsets"]
        itemsize = 2
        in_sumsq = np.zeros(cols, dtype=np.float64)
        col3994 = np.empty(rows, dtype=np.float32)
        chunk = 2048
        with shard.open("rb") as fh:
            for r0 in range(0, rows, chunk):
                r1 = min(rows, r0 + chunk)
                fh.seek(data_start + lo + r0 * cols * itemsize)
                raw = fh.read((r1 - r0) * cols * itemsize)
                u16 = np.frombuffer(raw, dtype=np.uint16)
                x = (u16.astype(np.uint32) << 16).view(np.float32).reshape(r1 - r0, cols)
                xd = x.astype(np.float64, copy=False)
                in_sumsq += np.square(xd).sum(axis=0)
                col3994[r0:r1] = x[:, ISLAND0]
                del raw, u16, x, xd
        in_rms = np.sqrt(in_sumsq / rows)
        med = float(np.median(in_rms))
        order = np.argsort(in_rms)[::-1]
        return {
            "name": name,
            "shape": [rows, cols],
            "col3994_rms": float(in_rms[ISLAND0]),
            "col3994_xmed": float(in_rms[ISLAND0] / med),
            "col3994_rank": int(np.sum(in_rms > in_rms[ISLAND0]) + 1),
            "in_max_ch": int(order[0]),
            "in_max_xmed": float(in_rms[order[0]] / med),
            "in_n_hot4": int(np.sum(in_rms > 4.0 * med)),
            "top5_in": [{"ch": int(i), "rms": float(in_rms[i]), "xmed": float(in_rms[i] / med)} for i in order[:5]],
            "col3994_mean_abs": float(np.mean(np.abs(col3994))),
        }

    log("embed column census")
    embed = table_col_stats("language_model.model.embed_tokens.weight")
    log(f"embed col3994 rank={embed['col3994_rank']} xmed={embed['col3994_xmed']:.3f} max_ch={embed['in_max_ch']}")
    log("lm_head column census")
    lm_head = table_col_stats("language_model.lm_head.weight")
    log(f"lm_head col3994 rank={lm_head['col3994_rank']} xmed={lm_head['col3994_xmed']:.3f} max_ch={lm_head['in_max_ch']}")
    return {
        "writes": writes,
        "reads": reads,
        "norms": norms,
        "final_norm": final_norm,
        "embed": embed,
        "lm_head": lm_head,
    }


# ---------------------------------------------------------------------------
# 3. Payoff sweep
# ---------------------------------------------------------------------------

def col_mse_share(X: np.ndarray, W: np.ndarray, Wq: np.ndarray, cols: np.ndarray, Y_err_norm2: float) -> float:
    if cols.size == 0 or Y_err_norm2 <= 0:
        return 0.0
    dW = (W[:, cols] - Wq[:, cols]).astype(np.float64)
    contrib = X[:, cols].astype(np.float64) @ dW.T
    return float(np.sum(np.square(contrib)) / Y_err_norm2)


def sweep_one(label: str, W: np.ndarray, X: np.ndarray, direction: str, act_rank: list[int], hold: np.ndarray, R: np.ndarray | None) -> dict:
    """direction: 'in_col' (protect input columns) or 'out_row' (protect output rows)."""
    Y = X @ W.T
    residual_out = direction == "out_row"
    out = {
        "label": label,
        "direction": direction,
        "shape": [int(x) for x in W.shape],
        "x_shape": [int(x) for x in X.shape],
        "weight_cosine_q": {},
        "curves": {},
        "q4": None,
        "refit": {},
        "random_control": {},
        "weight_rank_control": {},
    }
    Wq4 = uniform_absmax_recon(W, 4)
    Y4 = X @ Wq4.T
    out["q4"] = {
        "weight_cosine": mean_row_cosine(W.reshape(1, -1), Wq4.reshape(1, -1)),
        **score_pair(Y, Y4, hold, residual_out),
    }
    if R is not None and R.shape[1] == Y.shape[1]:
        out["q4"]["residual_proxy_cosine"] = mean_row_cosine(R[hold] + Y[hold], R[hold] + Y4[hold])
    log(f"  {label} Q4 hold_cos={out['q4']['hold_cosine']:.6f} ch3994_rel={out['q4']['ch3994_rel_err']}")
    del Wq4, Y4

    rng = np.random.default_rng(3994)
    random_rank = rng.permutation(HIDDEN).tolist()
    # weight-native ranking along the protected axis
    if direction == "in_col":
        axis_rms = np.sqrt(np.mean(np.square(W, dtype=np.float64), axis=0))
    else:
        axis_rms = np.sqrt(np.mean(np.square(W, dtype=np.float64), axis=1))
    w_rank = np.argsort(axis_rms)[::-1]
    w_rank_list = [int(i) for i in w_rank[:64]]

    for bits in BITS:
        Wq0 = uniform_absmax_recon(W, bits)
        Y0 = X @ Wq0.T
        wcos = mean_row_cosine(W.reshape(1, -1), Wq0.reshape(1, -1))
        out["weight_cosine_q"][str(bits)] = wcos
        err0 = (Y - Y0).astype(np.float64)
        err0_n2 = float(np.sum(np.square(err0)))
        # MSE share of column 3994 on the un-protected quant (read direction)
        share3994 = None
        if direction == "in_col" and W.shape[1] > ISLAND0:
            share3994 = col_mse_share(X, W, Wq0, np.array([ISLAND0]), err0_n2)
        curves = []
        for k in K_SWEEP:
            chans = np.array(act_rank[:k], dtype=np.int64) if k else np.array([], dtype=np.int64)
            if direction == "in_col":
                chans = chans[chans < W.shape[1]]
                Yq = Y0.copy()
                if chans.size:
                    dW = (W[:, chans] - Wq0[:, chans]).astype(np.float64)
                    Yq = (Yq.astype(np.float64) + X[:, chans].astype(np.float64) @ dW.T).astype(np.float32)
            else:
                chans = chans[chans < W.shape[0]]
                Yq = Y0.copy()
                if chans.size:
                    Yq[:, chans] = Y[:, chans]
            rec = score_pair(Y, Yq, hold, residual_out)
            rec["k"] = int(k)
            rec["channels"] = [int(c) for c in chans]
            rec["includes_3994"] = bool(ISLAND0 in chans)
            rec["weight_cosine"] = wcos
            if direction == "in_col":
                rec["col3994_mse_share_at_k0"] = share3994
                if chans.size:
                    rec["protected_mse_share_at_k0"] = col_mse_share(X, W, Wq0, chans, err0_n2)
            if R is not None and R.shape[1] == Y.shape[1]:
                rec["residual_proxy_cosine"] = mean_row_cosine(R[hold] + Y[hold], R[hold] + Yq[hold])
            # X energy in protected set
            if direction == "in_col" and chans.size:
                rec["x_energy_frac_protected"] = float(
                    np.sum(np.square(X[:, chans], dtype=np.float64)) / max(float(np.sum(np.square(X, dtype=np.float64))), 1e-30)
                )
            elif direction == "out_row" and chans.size:
                rec["w_frob_frac_protected"] = float(
                    np.sum(np.square(W[chans], dtype=np.float64)) / max(float(np.sum(np.square(W, dtype=np.float64))), 1e-30)
                )
            curves.append(rec)
        out["curves"][str(bits)] = curves
        log(
            f"  {label} Q{bits} k=0 hold={curves[0]['hold_cosine']:.6f} "
            f"k=1 hold={curves[1]['hold_cosine']:.6f} "
            f"k=3 hold={curves[3]['hold_cosine']:.6f} "
            f"ch3994_rel k0={curves[0]['ch3994_rel_err']} "
            f"col_mse_share={share3994}"
        )
        # refit for Q3 k=1 and k=3 on in_col only
        if bits == 3 and direction == "in_col":
            for k in (1, 3):
                chans = np.array(act_rank[:k], dtype=np.int64)
                chans = chans[chans < W.shape[1]]
                Wr = uniform_absmax_refit_zero_cols(W, bits, chans)
                Yr = X @ Wr.T
                out["refit"][f"q3_k{k}"] = {
                    "k": k,
                    "channels": [int(c) for c in chans],
                    "weight_cosine": mean_row_cosine(W.reshape(1, -1), Wr.reshape(1, -1)),
                    **score_pair(Y, Yr, hold, residual_out),
                }
                del Wr, Yr
        # random control Q3
        if bits == 3:
            rc = []
            for k in K_SWEEP:
                chans = np.array(random_rank[:k], dtype=np.int64) if k else np.array([], dtype=np.int64)
                if direction == "in_col":
                    chans = chans[chans < W.shape[1]]
                    Yq = Y0.copy()
                    if chans.size:
                        dW = (W[:, chans] - Wq0[:, chans]).astype(np.float64)
                        Yq = (Yq.astype(np.float64) + X[:, chans].astype(np.float64) @ dW.T).astype(np.float32)
                else:
                    chans = chans[chans < W.shape[0]]
                    Yq = Y0.copy()
                    if chans.size:
                        Yq[:, chans] = Y[:, chans]
                rc.append({"k": int(k), "channels": [int(c) for c in chans[:8]], **score_pair(Y, Yq, hold, residual_out)})
            out["random_control"]["3"] = rc
            # weight-rank control Q3
            wc = []
            for k in K_SWEEP:
                chans = np.array(w_rank_list[:k], dtype=np.int64) if k else np.array([], dtype=np.int64)
                if direction == "in_col":
                    chans = chans[chans < W.shape[1]]
                    Yq = Y0.copy()
                    if chans.size:
                        dW = (W[:, chans] - Wq0[:, chans]).astype(np.float64)
                        Yq = (Yq.astype(np.float64) + X[:, chans].astype(np.float64) @ dW.T).astype(np.float32)
                else:
                    chans = chans[chans < W.shape[0]]
                    Yq = Y0.copy()
                    if chans.size:
                        Yq[:, chans] = Y[:, chans]
                wc.append({
                    "k": int(k),
                    "channels": [int(c) for c in chans[:8]],
                    "includes_3994": bool(ISLAND0 in chans),
                    **score_pair(Y, Yq, hold, residual_out),
                })
            out["weight_rank_control"]["3"] = wc
        del Wq0, Y0
        gc.collect()
    del Y
    gc.collect()
    return out


def payoff(act_rank: list[int]) -> dict:
    log("payoff sweep start")
    hold = hold_idx()
    results = []
    for layer in LAYERS:
        log(f"===== layer {layer} gqa={is_gqa(layer)} =====")
        Xh = load_hidden(layer)
        gqa = is_gqa(layer)
        # READ tensors: protect input columns
        read_specs = [
            ("gate", tname(layer, "mlp.gate_proj.weight")),
            ("up", tname(layer, "mlp.up_proj.weight")),
            ("in", tname(layer, "self_attn.q_proj.weight" if gqa else "linear_attn.in_proj_qkv.weight")),
        ]
        for cls, name in read_specs:
            log(f"eval {name}")
            W = load_tensor(name)
            rec = sweep_one(f"L{layer}.{cls}", W, Xh, "in_col", act_rank, hold, R=None)
            rec["layer"] = layer
            rec["class"] = cls
            rec["name"] = name
            rec["mixer"] = "gqa" if gqa else "delta"
            results.append(rec)
            del W
            gc.collect()
        # WRITE down
        log(f"eval down L{layer}")
        Xd = down_x(Xh, layer)
        W = load_tensor(tname(layer, "mlp.down_proj.weight"))
        rec = sweep_one(f"L{layer}.down", W, Xd, "out_row", act_rank, hold, R=Xh)
        rec["layer"] = layer
        rec["class"] = "down"
        rec["name"] = tname(layer, "mlp.down_proj.weight")
        rec["mixer"] = "gqa" if gqa else "delta"
        rec["x_site"] = "reconstructed_swiglu_from_captured_hidden"
        results.append(rec)
        del W, Xd
        gc.collect()
        # WRITE out
        log(f"eval out L{layer}")
        if gqa:
            Wq = load_tensor(tname(layer, "self_attn.q_proj.weight"))
            Wv = load_tensor(tname(layer, "self_attn.v_proj.weight"))
            Xo = gqa_out_proxy(Xh, Wq, Wv)
            del Wq, Wv
            oname = tname(layer, "self_attn.o_proj.weight")
            xsite = "gqa_repeat(v)*sigmoid(q_gate)_proxy"
        else:
            Wqkv = load_tensor(tname(layer, "linear_attn.in_proj_qkv.weight"))
            Wz = load_tensor(tname(layer, "linear_attn.in_proj_z.weight"))
            Xo = deltanet_out_proxy(Xh, Wqkv, Wz)
            del Wqkv, Wz
            oname = tname(layer, "linear_attn.out_proj.weight")
            xsite = "deltanet_v*silu(z)_from_split_qkv_z_proxy"
        W = load_tensor(oname)
        rec = sweep_one(f"L{layer}.out", W, Xo, "out_row", act_rank, hold, R=Xh)
        rec["layer"] = layer
        rec["class"] = "out"
        rec["name"] = oname
        rec["mixer"] = "gqa" if gqa else "delta"
        rec["x_site"] = xsite
        results.append(rec)
        del W, Xo, Xh
        gc.collect()
    return {"layers": list(LAYERS), "k_sweep": list(K_SWEEP), "bits": list(BITS), "hold": "odd_rows_of_256", "tensors": results}


# ---------------------------------------------------------------------------
# 4. lm_head chunked Q3 + protect
# ---------------------------------------------------------------------------

def lm_head_payoff(act_rank: list[int]) -> dict:
    log("lm_head payoff (chunked)")
    name = "language_model.lm_head.weight"
    Xh = load_hidden(63)
    wn = load_tensor("language_model.model.norm.weight")
    X = rmsnorm(Xh, wn)
    hold = hold_idx()
    shard, info, data_start = tensor_info(name)
    rows, cols = int(info["shape"][0]), int(info["shape"][1])
    lo, _ = info["data_offsets"]
    chunk = 2048
    ks = (0, 1, 3, 8)
    bits = 3
    # accumulators per k: sum of row-cosine on hold
    acc = {k: {"num": 0.0, "n": 0, "rel2_num": 0.0, "rel2_den": 0.0} for k in ks}
    acc["q4"] = {"num": 0.0, "n": 0, "rel2_num": 0.0, "rel2_den": 0.0}
    with shard.open("rb") as fh:
        for r0 in range(0, rows, chunk):
            r1 = min(rows, r0 + chunk)
            fh.seek(data_start + lo + r0 * cols * 2)
            raw = fh.read((r1 - r0) * cols * 2)
            u16 = np.frombuffer(raw, dtype=np.uint16)
            W = (u16.astype(np.uint32) << 16).view(np.float32).reshape(r1 - r0, cols)
            Y = X @ W.T
            Wq4 = uniform_absmax_recon(W, 4)
            Y4 = X @ Wq4.T
            acc["q4"]["num"] += mean_row_cosine(Y[hold], Y4[hold]) * (r1 - r0)
            acc["q4"]["n"] += (r1 - r0)
            acc["q4"]["rel2_num"] += float(np.sum(np.square(Y - Y4, dtype=np.float64)))
            acc["q4"]["rel2_den"] += float(np.sum(np.square(Y, dtype=np.float64)))
            Wq = uniform_absmax_recon(W, bits)
            Y0 = X @ Wq.T
            for k in ks:
                chans = np.array(act_rank[:k], dtype=np.int64) if k else np.array([], dtype=np.int64)
                Yq = Y0.copy()
                if chans.size:
                    dW = (W[:, chans] - Wq[:, chans]).astype(np.float64)
                    Yq = (Yq.astype(np.float64) + X[:, chans].astype(np.float64) @ dW.T).astype(np.float32)
                acc[k]["num"] += mean_row_cosine(Y[hold], Yq[hold]) * (r1 - r0)
                acc[k]["n"] += (r1 - r0)
                acc[k]["rel2_num"] += float(np.sum(np.square(Y - Yq, dtype=np.float64)))
                acc[k]["rel2_den"] += float(np.sum(np.square(Y, dtype=np.float64)))
            del W, Wq, Wq4, Y, Y0, Y4, raw, u16
            if r0 % 32768 == 0:
                log(f"  lm_head rows {r0}/{rows}")
    def pack(a):
        return {
            "hold_cosine_mass_mean": a["num"] / max(a["n"], 1),
            "rel_l2": (a["rel2_num"] ** 0.5) / max(a["rel2_den"] ** 0.5, 1e-30),
            "n_rows": a["n"],
        }
    out = {
        "x_site": "L63_hidden then final RMSNorm (1+w); NOT a confirmed lm_head capture",
        "q4": pack(acc["q4"]),
        "q3": {str(k): pack(acc[k]) for k in ks},
        "channels": {str(k): act_rank[:k] for k in ks},
    }
    log(f"lm_head Q4 hold={out['q4']['hold_cosine_mass_mean']:.6f} Q3 k0={out['q3']['0']['hold_cosine_mass_mean']:.6f} k1={out['q3']['1']['hold_cosine_mass_mean']:.6f}")
    return out


# ---------------------------------------------------------------------------
# 5. Complete BPW
# ---------------------------------------------------------------------------

def complete_bpw(k: int, body_bpw_by_class: dict) -> dict:
    """Exact arithmetic on geometry. Island values bf16, index bits = 0 (compile-time set)."""
    # residual-touching GEMV island elements per protected channel
    down = 64 * INTERMEDIATE
    gate = 64 * INTERMEDIATE
    up = 64 * INTERMEDIATE
    o = 64 * 6144  # 48 lin_o + 16 o
    qkv = 48 * 10240
    z = 48 * 6144
    q = 16 * 12288
    kv = 16 * 1024 * 2
    ab = 48 * 48 * 2
    lm = VOCAB
    emb = VOCAB
    per_k = down + gate + up + o + qkv + z + q + kv + ab + lm + emb
    island_elems = k * per_k
    # GEMV mass by class (source elements)
    mlp = 3 * 64 * INTERMEDIATE * HIDDEN  # gate+up+down
    attn_delta = 48 * ((10240 + 6144 + 6144) * HIDDEN + 48 * HIDDEN * 2)  # qkv+z+o + a+b
    # o is 5120*6144, a/b 48*5120
    attn_delta = (
        48 * 10240 * HIDDEN
        + 48 * 6144 * HIDDEN
        + 48 * HIDDEN * 6144
        + 48 * 48 * HIDDEN
        + 48 * 48 * HIDDEN
    )
    attn_gqa = (
        16 * 12288 * HIDDEN
        + 16 * 1024 * HIDDEN
        + 16 * 1024 * HIDDEN
        + 16 * HIDDEN * 6144
    )
    embed = VOCAB * HIDDEN
    lm_head = VOCAB * HIDDEN
    # small f32 (G0)
    f32_elems = (
        64 * HIDDEN  # input rms
        + 64 * HIDDEN  # post
        + 16 * 256  # q_norm
        + 16 * 256  # k_norm
        + 48 * 128  # lin_norm
        + HIDDEN  # final
        + 48 * 48  # A_log
        + 48 * 48  # dt_bias
        + 48 * 10240 * 4  # conv1d
    )
    gemv_elems = mlp + attn_delta + attn_gqa + embed + lm_head
    other = SOURCE_ELEMS - gemv_elems - f32_elems

    def bits_for(body_map):
        # body_map keys: mlp, attn, embed, lm_head  (nominal bits+scale, i.e. 2.25 / 3.25 / 4.25)
        island_bits = 16.0 * island_elems
        def body(n_elems, bpw):
            return bpw * n_elems
        # island elems are subtracted from their class
        # allocate island mass to classes
        isl_mlp = k * (down + gate + up)
        isl_attn = k * (o + qkv + z + q + kv + ab)
        isl_emb = k * emb
        isl_lm = k * lm
        b = 0.0
        b += body(mlp - isl_mlp, body_map["mlp"])
        b += body(attn_delta + attn_gqa - isl_attn, body_map["attn"])
        b += body(embed - isl_emb, body_map["embed"])
        b += body(lm_head - isl_lm, body_map["lm_head"])
        b += island_bits
        b += 32.0 * f32_elems
        # leftover (should be ~0 language-other); charge G0 4.25 if any
        if other > 0:
            b += G0_BPW * other
        return b / SOURCE_ELEMS

    recipes = {
        "all_q2_plus_island": {"mlp": 2.25, "attn": 2.25, "embed": 2.25, "lm_head": 2.25},
        "all_q3_plus_island": {"mlp": 3.25, "attn": 3.25, "embed": 3.25, "lm_head": 3.25},
        "mlp_q2_attn_q4_emb_q4": {"mlp": 2.25, "attn": 4.25, "embed": 4.25, "lm_head": 4.25},
        "mlp_q3_attn_q3_emb_q4": {"mlp": 3.25, "attn": 3.25, "embed": 4.25, "lm_head": 4.25},
        "mlp_q3_attn_q4_emb_q4": {"mlp": 3.25, "attn": 4.25, "embed": 4.25, "lm_head": 4.25},
        "g0_q4_plus_island": {"mlp": 4.25, "attn": 4.25, "embed": 4.25, "lm_head": 4.25},
    }
    out_recipes = {}
    for name, m in recipes.items():
        bpw = bits_for(m)
        out_recipes[name] = {
            "body": m,
            "complete_bpw": bpw,
            "delta_vs_g0": bpw - G0_BPW,
            "label": "PROJECTED_arithmetic_on_geometry_not_a_packed_artifact",
        }
    return {
        "k": k,
        "index_bits": 0,
        "island_value_bits": 16,
        "per_channel_elems": per_k,
        "island_elems": island_elems,
        "island_frac_of_params": island_elems / SOURCE_ELEMS,
        "island_added_bpw_vs_q3_body": island_elems * (16.0 - 3.25) / SOURCE_ELEMS,
        "mass": {
            "mlp": mlp,
            "attn_delta": attn_delta,
            "attn_gqa": attn_gqa,
            "embed": embed,
            "lm_head": lm_head,
            "f32_elems": f32_elems,
            "gemv_elems": gemv_elems,
            "other": other,
            "source_elems": SOURCE_ELEMS,
        },
        "breakdown_per_k": {
            "down_rows": down,
            "gate_cols": gate,
            "up_cols": up,
            "o_lin_o_rows": o,
            "qkv_cols": qkv,
            "z_cols": z,
            "q_cols": q,
            "kv_cols": kv,
            "ab_cols": ab,
            "lm_head_cols": lm,
            "embed_cols": emb,
        },
        "recipes": out_recipes,
    }


def kernel_geometry() -> dict:
    # 3994 even → team 0 of TG 1997 among 5120-row organs
    row = ISLAND0
    tg = row // 2
    team = row % 2
    # input column walk: col = lane_in_row * 8; stride 512; lane_in_row = split*32 + simd_lane
    # 3994 = 3992 + 2; 3992 / 8 = 499; 499 = lane + 64*t → lane = 499 % 64 = 51
    col = ISLAND0
    group = col // GROUP
    local = col - group * GROUP
    aligned8 = col - (col % 8)
    lane_word = aligned8 // 8
    lane_in_row = lane_word % 64
    split = lane_in_row // 32
    simd_lane = lane_in_row % 32
    return {
        "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
        "source": "crates/hawking-core/shaders/qwen_uniform_q4.metal:183-221",
        "tg": 128,
        "simdgroups_per_tg": 4,
        "rows_per_tg": 2,
        "threads_per_row": 64,
        "group_size": 64,
        "unpack_width": 8,
        "col_stride": 512,
        "row_3994": {
            "row": row,
            "threadgroup_id": tg,
            "team": team,
            "partner_row": row + (1 if team == 0 else -1),
            "note": "one of two rows owned by TG 1997; overwriting output[3994] after the GEMV does not drop the TG",
        },
        "col_3994": {
            "col": col,
            "group": group,
            "local": local,
            "aligned8": aligned8,
            "lane_in_row": lane_in_row,
            "split": split,
            "simd_lane": simd_lane,
            "note": "lives in the middle of an 8-unpack (local 26); in-kernel special-case diverges the packed load",
        },
        "col_3456": {
            "col": 3456,
            "group": 3456 // 64,
            "local": 0,
            "note": "group-aligned",
        },
        "col_310": {
            "col": 310,
            "group": 310 // 64,
            "local": 310 % 64,
        },
        "preferred_branch": (
            "pack-time zero the protected columns/rows in the Qn body; "
            "leave geo_tpr64_tg128 unchanged; "
            "epilogue: for each protected input channel c, y += x[c] * W_exact[:, c]; "
            "for each protected output row r, y[r] = dot(W_exact[r, :], x). "
            "Common-case Qn path has zero extra branches and zero per-weight indirection."
        ),
        "common_case_cost": (
            "zero inside the 401 geo_tpr64 GEMVs. "
            "Epilogue cost PROJECTED: k saxpys of length `rows` (read organs) plus k dots of length `cols` (write organs). "
            "k=1 gate saxpy is 17408 bf16 = 34816 B; k=1 down row-dot is 17408 bf16 = 34816 B. "
            "Not a token-level measurement."
        ),
    }


def main():
    t0 = time.time()
    meta = json.load(open(CAP / "capture-result.json"))
    payload = {
        "schema": "g1.channel3994.island.v1",
        "t0": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_bf16": str(SRC),
        "capture": str(CAP),
        "limits": "cpu_numpy_only_no_gpu_no_generate_peak_rss_under_20G",
    }
    # Q4 self-check
    log("Q4 self-check L0 lin_o")
    W = load_tensor(tname(0, "linear_attn.out_proj.weight"))
    Wq = uniform_absmax_recon(W, 4)
    wcos = mean_row_cosine(W.reshape(1, -1), Wq.reshape(1, -1))
    payload["q4_selfcheck"] = {
        "tensor": tname(0, "linear_attn.out_proj.weight"),
        "weight_cosine": wcos,
        "excess_kurtosis": excess_kurtosis(W),
        "expect_weight_cosine": 0.993541,
        "expect_kurtosis": 149.3577,
        "match_probe_7digits": abs(wcos - 0.993541491119444) < 1e-6,
    }
    log(f"Q4 self-check weight_cosine={wcos:.8f} kurt={payload['q4_selfcheck']['excess_kurtosis']:.4f}")
    del W, Wq
    gc.collect()

    def ckpt(stage: str) -> None:
        payload["wall_s"] = time.time() - t0
        payload["rss_max_gb"] = rss_gb()
        payload["stage"] = stage
        with open(OUT, "w") as f:
            json.dump(payload, f)
        log(f"checkpoint {stage} -> {OUT}")

    payload["activation"] = activation_census(meta)
    act_rank = payload["activation"]["cross_layer"]["act_rank_by_mean_rms"]
    log(f"protected-set rank[:8] = {act_rank[:8]}")
    ckpt("activation")

    payload["weight_sample"] = weight_sample()
    ckpt("weight_sample")
    payload["payoff"] = payoff(act_rank)
    ckpt("payoff")
    payload["lm_head_payoff"] = lm_head_payoff(act_rank)
    payload["complete_bpw"] = {str(k): complete_bpw(k, {}) for k in K_SWEEP}
    payload["kernel"] = kernel_geometry()
    payload["wall_s"] = time.time() - t0
    payload["rss_max_gb"] = rss_gb()
    payload["stage"] = "done"
    with open(OUT, "w") as f:
        json.dump(payload, f)
    log(f"WROTE {OUT} wall={payload['wall_s']:.1f}s rss_max={payload['rss_max_gb']:.3f}G")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Q4 group-128 MSE candidate screen.

CPU/numpy only. No GPU, no pack, no generate, no resident touch.
Primary hold = last 64 of the 256-token capture (same tokens as G0 S0).
Fit = first 192. Scale search snaps s to f16 before rint (stored-f16 authority).

Writes incremental JSONL + a final JSON under /tmp.
"""
from __future__ import annotations

import json
import math
import resource
import struct
import time
from pathlib import Path

import numpy as np

SRC = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16")
CAP = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1")
MAN = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json")
OUT_JSON = Path("/tmp/g1_q4_mse_g128_screen.json")
OUT_JSONL = Path("/tmp/g1_q4_mse_g128_screen.jsonl")
LOG = Path("/tmp/g1_q4_mse_g128_screen.log")

HIDDEN = 5120
N_TOKENS = 256
FIT_N = 192
HOLD_N = 64
N_PARAMS = 26_895_998_464
G0_BYTES = 14_297_694_680
G0_S0 = 0.4078534106896186
KEY_HEADS = 16
VALUES_PER_KEY = 3
KEY_DIM = 128
VALUE_DIM = 128
GQA_HEADS = 24
GQA_KV = 4
GQA_HEAD_DIM = 256
MSE_MULT = (0.50, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 2.00)
BOUND4 = 7

_HEADER_CACHE: dict[Path, dict] = {}
_WMAP = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] rss={rss_gb():.3f}G {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


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


def flat_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """G0 S0 metric: flattened output cosine (g1-mlp-floor / g1-screen-vs-generate)."""
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(left @ right)
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    if den <= 1e-12:
        return 1.0 if num == 0.0 else 0.0
    return num / den


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


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


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
    if raw.size != N_TOKENS * HIDDEN:
        raise RuntimeError(f"hidden L{layer} size {raw.size}")
    return np.ascontiguousarray(raw.reshape(N_TOKENS, HIDDEN))


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


def fuse_ba(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate([b, a], axis=0), dtype=np.float32)


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


def n_blocks(n_in: int, g: int) -> int:
    return (n_in + g - 1) // g


def hq30_bytes(elements: int, rank: int, g: int) -> int:
    ng = (elements + g - 1) // g
    code_b = (g * 4 + 7) // 8
    return 32 + 4 * rank + ng * (2 + code_b)


def absmax_recon(W: np.ndarray, g: int, snap_f16: bool = True) -> np.ndarray:
    n_out, n_in = W.shape
    Wh = np.empty_like(W)
    w = np.ascontiguousarray(W, dtype=np.float32)
    for bi in range(n_blocks(n_in, g)):
        lo = bi * g
        hi = min(lo + g, n_in)
        blk = w[:, lo:hi]
        amax = np.max(np.abs(blk), axis=1)
        s = amax / BOUND4
        if snap_f16:
            s = s.astype(np.float16).astype(np.float32)
        den = np.where(s > 0.0, s, 1.0)
        codes = np.clip(np.rint(blk / den[:, None]), -BOUND4, BOUND4)
        codes = np.where((s > 0.0)[:, None], codes, 0.0)
        Wh[:, lo:hi] = codes * s[:, None]
    return Wh


def mse_recon(
    W: np.ndarray,
    X_fit: np.ndarray,
    g: int,
    snap_f16: bool = True,
    row_chunk: int = 0,
) -> tuple[np.ndarray, dict]:
    n_out, n_in = W.shape
    if X_fit.shape[1] != n_in:
        raise RuntimeError(f"X_fit in {X_fit.shape[1]} != W in {n_in}")
    Wh = np.empty_like(W)
    picked = np.zeros(len(MSE_MULT), dtype=np.int64)
    improved = 0
    n_groups = 0
    X64 = np.ascontiguousarray(X_fit, dtype=np.float64)
    t0 = time.time()
    one_i = MSE_MULT.index(1.0)
    chunks = [(0, n_out)] if row_chunk <= 0 else [
        (s, min(s + row_chunk, n_out)) for s in range(0, n_out, row_chunk)
    ]
    for rs, re in chunks:
        W64 = np.ascontiguousarray(W[rs:re], dtype=np.float64)
        n_rows = re - rs
        for bi in range(n_blocks(n_in, g)):
            lo = bi * g
            hi = min(lo + g, n_in)
            Xg = X64[:, lo:hi]
            Grm = Xg.T @ Xg
            w = W64[:, lo:hi]
            amax = np.max(np.abs(w), axis=1)
            s0 = amax / BOUND4
            best_c = np.full(n_rows, np.inf, dtype=np.float64)
            best_s = s0.copy()
            best_i = np.full(n_rows, one_i, dtype=np.int32)
            zero = s0 <= 0.0
            if np.any(~zero):
                for i, m in enumerate(MSE_MULT):
                    s = s0 * m
                    if snap_f16:
                        s = s.astype(np.float16).astype(np.float64)
                    den = np.where(s > 0.0, s, 1.0)
                    codes = np.clip(np.rint(w / den[:, None]), -BOUND4, BOUND4)
                    codes = np.where((s > 0.0)[:, None], codes, 0.0)
                    e = w - codes * s[:, None]
                    c = np.sum((e @ Grm) * e, axis=1)
                    better = (c < best_c) & (~zero)
                    if np.any(better):
                        best_c = np.where(better, c, best_c)
                        best_s = np.where(better, s, best_s)
                        best_i = np.where(better, i, best_i)
            den = np.where(best_s > 0.0, best_s, 1.0)
            codes = np.clip(np.rint(w / den[:, None]), -BOUND4, BOUND4)
            codes = np.where((best_s > 0.0)[:, None], codes, 0.0)
            Wh[rs:re, lo:hi] = (codes * best_s[:, None]).astype(np.float32)
            for i in range(len(MSE_MULT)):
                picked[i] += int(np.sum(best_i == i))
            improved += int(np.sum((best_i != one_i) & (~zero)))
            n_groups += n_rows
    meta = {
        "n_groups": int(n_groups),
        "n_groups_not_absmax": int(improved),
        "frac_groups_not_absmax": float(improved) / float(max(n_groups, 1)),
        "n_picked_per_multiplier": [int(x) for x in picked],
        "wall_s": time.time() - t0,
        "snap_f16": bool(snap_f16),
    }
    return Wh, meta


def score_hold(W: np.ndarray, Wh: np.ndarray, X_hold: np.ndarray) -> dict:
    y = X_hold @ W.T
    yh = X_hold @ Wh.T
    return {
        "hold_flat_cosine": flat_cosine(y, yh),
        "hold_mean_row_cosine": mean_row_cosine(y, yh),
        "hold_output_cosine_min_row": min_row_cosine(y, yh),
    }


def capture_token_ids() -> np.ndarray:
    cap = json.loads((CAP / "capture-result.json").read_text())
    ids: list[int] = []
    for p in cap["prompts"]:
        ids.extend(int(x) for x in p["ids"])
    if len(ids) != N_TOKENS:
        raise RuntimeError(f"capture ids {len(ids)} != {N_TOKENS}")
    return np.asarray(ids, dtype=np.int64)


def product(xs: list[float]) -> float:
    p = 1.0
    for x in xs:
        p *= float(x)
    return p


def append_jsonl(row: dict) -> None:
    with OUT_JSONL.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def already_done() -> set[str]:
    done: set[str] = set()
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done.add(row["tensor_id"])
    return done


def compute_bpw() -> dict:
    man = json.loads(MAN.read_text())
    q4 = [t for t in man["tensors"] if t["kind"] == "q4"]
    f32 = [t for t in man["tensors"] if t["kind"] == "f32"]
    if len(q4) != 402 or len(f32) != 353:
        raise RuntimeError(f"unexpected catalog {len(q4)} q4 {len(f32)} f32")
    rem = {t["elements"] % 128 for t in q4}
    g64_bytes = sum(t["bytes"] for t in q4)
    g128_bytes = sum(hq30_bytes(t["elements"], len(t["shape"]), 128) for t in q4)
    f32_bytes = sum(t["bytes"] for t in f32)
    # formula check
    mismatches = sum(
        1 for t in q4 if hq30_bytes(t["elements"], len(t["shape"]), 64) != t["bytes"]
    )
    scale_g64 = sum((t["elements"] // 64) * 2 for t in q4)
    scale_g128 = sum((t["elements"] // 128) * 2 for t in q4)
    codes = sum(t["elements"] // 2 for t in q4)
    headers = 40 * len(q4)
    by_class = {}
    for t in q4:
        tail = t["name"].split(".")[-2] if t["name"].endswith(".weight") else t["name"]
        if "layers" in t["name"]:
            if "mlp.gate" in t["name"]:
                cls = "mlp.gate_proj"
            elif "mlp.up" in t["name"]:
                cls = "mlp.up_proj"
            elif "mlp.down" in t["name"]:
                cls = "mlp.down_proj"
            elif "in_proj_qkvz" in t["name"]:
                cls = "attn.in_proj_qkvz"
            elif "in_proj_ba" in t["name"]:
                cls = "attn.in_proj_ba"
            elif "linear_attn.out_proj" in t["name"]:
                cls = "attn.out_proj"
            elif "self_attn.q_proj" in t["name"]:
                cls = "attn.q_proj"
            elif "self_attn.k_proj" in t["name"]:
                cls = "attn.k_proj"
            elif "self_attn.v_proj" in t["name"]:
                cls = "attn.v_proj"
            elif "self_attn.o_proj" in t["name"]:
                cls = "attn.o_proj"
            else:
                cls = tail
        elif "embed_tokens" in t["name"]:
            cls = "embed_tokens"
        elif "lm_head" in t["name"]:
            cls = "lm_head"
        else:
            cls = tail
        rec = by_class.setdefault(cls, {"n": 0, "elements": 0, "g64_bytes": 0, "g128_bytes": 0})
        rec["n"] += 1
        rec["elements"] += t["elements"]
        rec["g64_bytes"] += t["bytes"]
        rec["g128_bytes"] += hq30_bytes(t["elements"], len(t["shape"]), 128)
    complete_g64 = 8.0 * (g64_bytes + f32_bytes) / N_PARAMS
    complete_g128 = 8.0 * (g128_bytes + f32_bytes) / N_PARAMS
    estimate_save = 400_343_040
    estimate_bpw = 8.0 * (G0_BYTES - estimate_save) / N_PARAMS
    return {
        "N": N_PARAMS,
        "q4_tensors": 402,
        "f32_tensors": 353,
        "q4_E_mod_128": sorted(rem),
        "g64_formula_mismatches": mismatches,
        "g0_payload_bytes": g64_bytes + f32_bytes,
        "g0_complete_bpw": complete_g64,
        "q4_g64_bytes": g64_bytes,
        "q4_g128_bytes": g128_bytes,
        "f32_bytes": f32_bytes,
        "g128_payload_bytes": g128_bytes + f32_bytes,
        "g128_complete_bpw": complete_g128,
        "scale_g64_bytes": scale_g64,
        "scale_g128_bytes": scale_g128,
        "scale_saved_bytes": scale_g64 - scale_g128,
        "code_bytes_unchanged": codes,
        "header_bytes_unchanged": headers,
        "contract_estimate_scale_save_bytes": estimate_save,
        "contract_estimate_complete_bpw": estimate_bpw,
        "delta_vs_estimate_bytes": (scale_g64 - scale_g128) - estimate_save,
        "delta_vs_estimate_bpw": complete_g128 - estimate_bpw,
        "delta_vs_g0_bpw": complete_g128 - complete_g64,
        "by_class": by_class,
        "note": (
            "Estimate halved only the no-embed GEMV scale plane 800,686,080/2. "
            "Real: all 402 Q4 tensors have E%128==0, so embed+lm_head scales also halve."
        ),
    }


def score_tensor(
    tensor_id: str,
    layer,
    role: str,
    family: str,
    W: np.ndarray,
    X: np.ndarray,
    x_site: str,
    snap_f16: bool,
    fit_idx: np.ndarray,
    hold_idx: np.ndarray,
    split: str,
) -> dict:
    X_fit = np.ascontiguousarray(X[fit_idx], dtype=np.float32)
    X_hold = np.ascontiguousarray(X[hold_idx], dtype=np.float32)
    t0 = time.time()
    n_out, n_in = W.shape
    row_chunk = 8192 if n_out >= 20000 else 0
    W_abs = absmax_recon(W, 64, snap_f16=snap_f16)
    s_abs = score_hold(W, W_abs, X_hold)
    del W_abs
    W_m64, meta64 = mse_recon(W, X_fit, 64, snap_f16=snap_f16, row_chunk=row_chunk)
    s_m64 = score_hold(W, W_m64, X_hold)
    del W_m64
    W_m128, meta128 = mse_recon(W, X_fit, 128, snap_f16=snap_f16, row_chunk=row_chunk)
    s_m128 = score_hold(W, W_m128, X_hold)
    del W_m128
    e = n_out * n_in
    row = {
        "tensor_id": tensor_id,
        "layer": layer,
        "role": role,
        "family": family,
        "W_shape": [n_out, n_in],
        "elements": e,
        "x_site": x_site,
        "split": split,
        "n_fit": int(fit_idx.size),
        "n_hold": int(hold_idx.size),
        "snap_f16": snap_f16,
        "g0_q4_g64_absmax": s_abs["hold_flat_cosine"],
        "g0_q4_g64_absmax_mean_row": s_abs["hold_mean_row_cosine"],
        "g0_q4_g64_absmax_min_row": s_abs["hold_output_cosine_min_row"],
        "q4_g64_mse": s_m64["hold_flat_cosine"],
        "q4_g64_mse_mean_row": s_m64["hold_mean_row_cosine"],
        "q4_g64_mse_min_row": s_m64["hold_output_cosine_min_row"],
        "q4_g128_mse": s_m128["hold_flat_cosine"],
        "q4_g128_mse_mean_row": s_m128["hold_mean_row_cosine"],
        "q4_g128_mse_min_row": s_m128["hold_output_cosine_min_row"],
        "delta_g128_mse_minus_g0": s_m128["hold_flat_cosine"] - s_abs["hold_flat_cosine"],
        "loses_vs_g0": bool(s_m128["hold_flat_cosine"] < s_abs["hold_flat_cosine"]),
        "g64_bytes": hq30_bytes(e, 2, 64),
        "g128_bytes": hq30_bytes(e, 2, 128),
        "scale_saved_bytes": (e // 64) * 2 - (e // 128) * 2,
        "mse64_meta": meta64,
        "mse128_meta": meta128,
        "wall_s": time.time() - t0,
    }
    return row


def verify_known_cells() -> dict:
    """Reproduce two sealed cells before the 402-wide run."""
    out = {}
    # L0 gate, last-64, g64 absmax f16 — MLP floor
    W = load_tensor(tname(0, "mlp.gate_proj.weight"))
    X = load_hidden(0)
    X_hold = X[FIT_N : FIT_N + HOLD_N]
    Wh = absmax_recon(W, 64, snap_f16=True)
    sc = score_hold(W, Wh, X_hold)
    out["L0.gate_q4_g64_absmax_last64"] = {
        "measured_flat": sc["hold_flat_cosine"],
        "measured_mean_row": sc["hold_mean_row_cosine"],
        "cited_flat": 0.9966467936497861,
        "abs_err_flat": abs(sc["hold_flat_cosine"] - 0.9966467936497861),
    }
    del W, Wh
    # L47 o, even/odd 128, g128 MSE no-snap — attention stack
    Wq = load_tensor(tname(47, "self_attn.q_proj.weight"))
    Wv = load_tensor(tname(47, "self_attn.v_proj.weight"))
    Wo = load_tensor(tname(47, "self_attn.o_proj.weight"))
    X = load_hidden(47)
    Xp = gqa_out_proxy(X, Wq, Wv)
    even = np.arange(0, N_TOKENS, 2)
    odd = np.arange(1, N_TOKENS, 2)
    Wh, meta = mse_recon(Wo, Xp[even], 128, snap_f16=False)
    sc = score_hold(Wo, Wh, Xp[odd])
    out["L47.o_q4_g128_mse_oddeven_nosnap"] = {
        "measured_mean_row": sc["hold_mean_row_cosine"],
        "measured_flat": sc["hold_flat_cosine"],
        "cited_mean_row": 0.9900882450327616,
        "abs_err_mean_row": abs(sc["hold_mean_row_cosine"] - 0.9900882450327616),
        "meta": meta,
    }
    return out


def run(verify_only: bool = False) -> dict:
    if LOG.exists():
        # keep one log; append
        pass
    log("start")
    bpw = compute_bpw()
    log(
        f"bpw g0={bpw['g0_complete_bpw']:.15f} g128={bpw['g128_complete_bpw']:.15f} "
        f"save={bpw['scale_saved_bytes']} vs_est={bpw['delta_vs_estimate_bytes']}"
    )
    verify = verify_known_cells()
    log(f"verify {json.dumps(verify)}")
    if verify_only:
        return {"bpw": bpw, "verify": verify}

    done = already_done()
    log(f"resume done={len(done)}")
    last64 = np.arange(FIT_N, FIT_N + HOLD_N)
    first192 = np.arange(0, FIT_N)
    rows: list[dict] = []
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    # language GEMVs, fused G0 catalog
    for layer in range(64):
        H = load_hidden(layer)
        Wg = load_tensor(tname(layer, "mlp.gate_proj.weight"))
        Wu = load_tensor(tname(layer, "mlp.up_proj.weight"))
        Wd = load_tensor(tname(layer, "mlp.down_proj.weight"))
        jobs = [
            (f"L{layer}.gate_proj", "gate_proj", "mlp", Wg, H, "captured_post_norm_hidden"),
            (f"L{layer}.up_proj", "up_proj", "mlp", Wu, H, "captured_post_norm_hidden"),
        ]
        # down X = SwiGLU(H)
        x_sw = silu(H @ Wg.T) * (H @ Wu.T)
        jobs.append((f"L{layer}.down_proj", "down_proj", "mlp", Wd, x_sw, "reconstructed_swiglu"))
        if is_gqa(layer):
            Wq = load_tensor(tname(layer, "self_attn.q_proj.weight"))
            Wk = load_tensor(tname(layer, "self_attn.k_proj.weight"))
            Wv = load_tensor(tname(layer, "self_attn.v_proj.weight"))
            Wo = load_tensor(tname(layer, "self_attn.o_proj.weight"))
            jobs += [
                (f"L{layer}.q_proj", "q_proj", "attn", Wq, H, "captured_post_norm_hidden"),
                (f"L{layer}.k_proj", "k_proj", "attn", Wk, H, "captured_post_norm_hidden"),
                (f"L{layer}.v_proj", "v_proj", "attn", Wv, H, "captured_post_norm_hidden"),
            ]
            xo = gqa_out_proxy(H, Wq, Wv)
            jobs.append((f"L{layer}.o_proj", "o_proj", "attn", Wo, xo, "derived_gqa_mixer_proxy"))
        else:
            Wqkv = load_tensor(tname(layer, "linear_attn.in_proj_qkv.weight"))
            Wz = load_tensor(tname(layer, "linear_attn.in_proj_z.weight"))
            Wa = load_tensor(tname(layer, "linear_attn.in_proj_a.weight"))
            Wb = load_tensor(tname(layer, "linear_attn.in_proj_b.weight"))
            Wo = load_tensor(tname(layer, "linear_attn.out_proj.weight"))
            Wqkvz = fuse_qkvz(Wqkv, Wz)
            Wba = fuse_ba(Wb, Wa)
            jobs += [
                (f"L{layer}.in_proj_qkvz", "in_proj_qkvz", "attn", Wqkvz, H, "captured_post_norm_hidden"),
                (f"L{layer}.in_proj_ba", "in_proj_ba", "attn", Wba, H, "captured_post_norm_hidden"),
            ]
            xo = deltanet_out_proxy(H, Wqkvz)
            jobs.append((f"L{layer}.out_proj", "out_proj", "attn", Wo, xo, "derived_deltanet_mixer_proxy"))
            del Wqkv, Wz, Wa, Wb
        for tid, role, fam, W, X, site in jobs:
            if tid in done:
                continue
            row = score_tensor(tid, layer, role, fam, W, X, site, True, first192, last64, "last64")
            append_jsonl(row)
            rows.append(row)
            done.add(tid)
            log(
                f"{tid:22s} g0={row['g0_q4_g64_absmax']:.9f} "
                f"m64={row['q4_g64_mse']:.9f} m128={row['q4_g128_mse']:.9f} "
                f"d={row['delta_g128_mse_minus_g0']:+.6f} lose={int(row['loses_vs_g0'])} "
                f"{row['wall_s']:.2f}s"
            )
        del Wg, Wu, Wd, H, x_sw

    # lm_head on L63 hidden (unconfirmed final-norm; same site as mse-scale-rule)
    if "lm_head" not in done:
        W = load_tensor("language_model.lm_head.weight")
        X = load_hidden(63)
        row = score_tensor("lm_head", None, "lm_head", "table", W, X, "L63_post_norm_unconfirmed_final", True, first192, last64, "last64")
        append_jsonl(row)
        rows.append(row)
        done.add("lm_head")
        log(f"lm_head m128={row['q4_g128_mse']:.9f} {row['wall_s']:.2f}s")
        del W

    # embed: gather observed token rows. Not a GEMV. Scored separately.
    if "embed_tokens" not in done:
        W = load_tensor("language_model.model.embed_tokens.weight")
        ids = capture_token_ids()
        X = load_hidden(0)
        t0 = time.time()
        Wh64 = absmax_recon(W, 64, snap_f16=True)
        Wh128, meta128 = mse_recon(W, X[first192], 128, snap_f16=True, row_chunk=8192)
        hold_ids = ids[FIT_N : FIT_N + HOLD_N]
        e = int(W.size)
        g0 = flat_cosine(W[hold_ids], Wh64[hold_ids])
        m128 = flat_cosine(W[hold_ids], Wh128[hold_ids])
        row = {
            "tensor_id": "embed_tokens",
            "layer": None,
            "role": "embed_tokens",
            "family": "table",
            "W_shape": [int(W.shape[0]), int(W.shape[1])],
            "elements": e,
            "x_site": "gathered_hold_token_rows; MSE X=L0_hidden UNDERDETERMINED (63 unique vocab rows)",
            "split": "last64",
            "n_fit": FIT_N,
            "n_hold": HOLD_N,
            "snap_f16": True,
            "g0_q4_g64_absmax": g0,
            "g0_q4_g64_absmax_mean_row": mean_row_cosine(W[hold_ids], Wh64[hold_ids]),
            "g0_q4_g64_absmax_min_row": min_row_cosine(W[hold_ids], Wh64[hold_ids]),
            "q4_g64_mse": None,
            "q4_g64_mse_mean_row": None,
            "q4_g64_mse_min_row": None,
            "q4_g128_mse": m128,
            "q4_g128_mse_mean_row": mean_row_cosine(W[hold_ids], Wh128[hold_ids]),
            "q4_g128_mse_min_row": min_row_cosine(W[hold_ids], Wh128[hold_ids]),
            "delta_g128_mse_minus_g0": m128 - g0,
            "loses_vs_g0": bool(m128 < g0),
            "g64_bytes": hq30_bytes(e, 2, 64),
            "g128_bytes": hq30_bytes(e, 2, 128),
            "scale_saved_bytes": (e // 64) * 2 - (e // 128) * 2,
            "mse128_meta": meta128,
            "n_unique_hold_ids": int(len(set(int(x) for x in hold_ids))),
            "n_unique_fit_ids": int(len(set(int(x) for x in ids[:FIT_N]))),
            "wall_s": time.time() - t0,
        }
        append_jsonl(row)
        rows.append(row)
        done.add("embed_tokens")
        log(f"embed gather g0={g0:.9f} m128={m128:.9f} {row['wall_s']:.2f}s")
        del W, Wh128, Wh64

    mlp = [r for r in rows if r["family"] == "mlp"]
    attn = [r for r in rows if r["family"] == "attn"]
    tables = [r for r in rows if r["family"] == "table"]
    summary = {
        "schema": "hawking.g1.q4_mse_g128_candidate_screen.v1",
        "date": time.strftime("%Y-%m-%d"),
        "n_rows": len(rows),
        "n_mlp": len(mlp),
        "n_attn": len(attn),
        "n_tables": len(tables),
        "split": "fit=first192 hold=last64",
        "snap_f16": True,
        "g0_s0_cited": G0_S0,
        "s0_metric": "flat_output_cosine last64 (same as G0 0.4078534106896186)",
        "s0_192_g0_recomputed": product([r["g0_q4_g64_absmax"] for r in mlp]) if len(mlp) == 192 else None,
        "s0_192_g128_mse": product([r["q4_g128_mse"] for r in mlp]) if len(mlp) == 192 else None,
        "s0_192_g64_mse": product([r["q4_g64_mse"] for r in mlp]) if len(mlp) == 192 else None,
        "min_192_g0": min(r["g0_q4_g64_absmax"] for r in mlp) if mlp else None,
        "min_192_g128_mse": min(r["q4_g128_mse"] for r in mlp) if mlp else None,
        "min_192_g64_mse": min(r["q4_g64_mse"] for r in mlp) if mlp else None,
        "s0_192_g128_mse_mean_row": product([r["q4_g128_mse_mean_row"] for r in mlp]) if len(mlp) == 192 else None,
        "min_192_g128_mse_mean_row": min(r["q4_g128_mse_mean_row"] for r in mlp) if mlp else None,
        "s0_208_attn_g128_mse": product([r["q4_g128_mse"] for r in attn]) if len(attn) == 208 else None,
        "min_208_attn_g128_mse": min(r["q4_g128_mse"] for r in attn) if attn else None,
        "s0_208_attn_g128_mse_mean_row": product([r["q4_g128_mse_mean_row"] for r in attn]) if len(attn) == 208 else None,
        "min_208_attn_g128_mse_mean_row": min(r["q4_g128_mse_mean_row"] for r in attn) if attn else None,
        "s0_400_gemv_g128_mse": product([r["q4_g128_mse"] for r in mlp + attn]) if len(mlp) + len(attn) == 400 else None,
        "min_400_g128_mse": min(r["q4_g128_mse"] for r in mlp + attn) if mlp or attn else None,
        "n_lose_vs_g0": sum(1 for r in rows if r.get("loses_vs_g0")),
        "lose_ids": [r["tensor_id"] for r in rows if r.get("loses_vs_g0")],
        "bpw": bpw,
        "verify": verify,
        "rss_max_gb": rss_gb(),
    }
    if len(mlp) == 192:
        summary["s0_192_delta_vs_g0_cited"] = summary["s0_192_g128_mse"] - G0_S0
        summary["s0_192_ratio_vs_g0_cited"] = summary["s0_192_g128_mse"] / G0_S0
    payload = {"summary": summary, "rows": rows}
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    log(f"wrote {OUT_JSON} n={len(rows)} s0_192={summary.get('s0_192_g128_mse')} min={summary.get('min_192_g128_mse')}")
    return payload


def process_layer(layer: int) -> list[dict]:
    """Score one layer's G0-fused GEMVs. Safe under spawn."""
    H = load_hidden(layer)
    Wg = load_tensor(tname(layer, "mlp.gate_proj.weight"))
    Wu = load_tensor(tname(layer, "mlp.up_proj.weight"))
    Wd = load_tensor(tname(layer, "mlp.down_proj.weight"))
    last64 = np.arange(FIT_N, FIT_N + HOLD_N)
    first192 = np.arange(0, FIT_N)
    jobs = [
        (f"L{layer}.gate_proj", "gate_proj", "mlp", Wg, H, "captured_post_norm_hidden"),
        (f"L{layer}.up_proj", "up_proj", "mlp", Wu, H, "captured_post_norm_hidden"),
    ]
    x_sw = silu(H @ Wg.T) * (H @ Wu.T)
    jobs.append((f"L{layer}.down_proj", "down_proj", "mlp", Wd, x_sw, "reconstructed_swiglu"))
    if is_gqa(layer):
        Wq = load_tensor(tname(layer, "self_attn.q_proj.weight"))
        Wk = load_tensor(tname(layer, "self_attn.k_proj.weight"))
        Wv = load_tensor(tname(layer, "self_attn.v_proj.weight"))
        Wo = load_tensor(tname(layer, "self_attn.o_proj.weight"))
        jobs += [
            (f"L{layer}.q_proj", "q_proj", "attn", Wq, H, "captured_post_norm_hidden"),
            (f"L{layer}.k_proj", "k_proj", "attn", Wk, H, "captured_post_norm_hidden"),
            (f"L{layer}.v_proj", "v_proj", "attn", Wv, H, "captured_post_norm_hidden"),
        ]
        xo = gqa_out_proxy(H, Wq, Wv)
        jobs.append((f"L{layer}.o_proj", "o_proj", "attn", Wo, xo, "derived_gqa_mixer_proxy"))
    else:
        Wqkv = load_tensor(tname(layer, "linear_attn.in_proj_qkv.weight"))
        Wz = load_tensor(tname(layer, "linear_attn.in_proj_z.weight"))
        Wa = load_tensor(tname(layer, "linear_attn.in_proj_a.weight"))
        Wb = load_tensor(tname(layer, "linear_attn.in_proj_b.weight"))
        Wo = load_tensor(tname(layer, "linear_attn.out_proj.weight"))
        Wqkvz = fuse_qkvz(Wqkv, Wz)
        Wba = fuse_ba(Wb, Wa)
        jobs += [
            (f"L{layer}.in_proj_qkvz", "in_proj_qkvz", "attn", Wqkvz, H, "captured_post_norm_hidden"),
            (f"L{layer}.in_proj_ba", "in_proj_ba", "attn", Wba, H, "captured_post_norm_hidden"),
        ]
        xo = deltanet_out_proxy(H, Wqkvz)
        jobs.append((f"L{layer}.out_proj", "out_proj", "attn", Wo, xo, "derived_deltanet_mixer_proxy"))
    out = []
    for tid, role, fam, W, X, site in jobs:
        row = score_tensor(tid, layer, role, fam, W, X, site, True, first192, last64, "last64")
        out.append(row)
    return out


def run_parallel(workers: int = 8) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    done = already_done()
    # layers still missing any of their tensors
    need = []
    for layer in range(64):
        names = [f"L{layer}.gate_proj", f"L{layer}.up_proj", f"L{layer}.down_proj"]
        if is_gqa(layer):
            names += [f"L{layer}.q_proj", f"L{layer}.k_proj", f"L{layer}.v_proj", f"L{layer}.o_proj"]
        else:
            names += [f"L{layer}.in_proj_qkvz", f"L{layer}.in_proj_ba", f"L{layer}.out_proj"]
        if any(n not in done for n in names):
            need.append(layer)
    log(f"parallel workers={workers} layers_needed={need}")
    if not need:
        run(verify_only=False)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_layer, L): L for L in need}
        for fut in as_completed(futs):
            L = futs[fut]
            try:
                rows = fut.result()
            except Exception as e:
                log(f"LAYER {L} FAIL {e!r}")
                raise
            for row in rows:
                if row["tensor_id"] in already_done():
                    continue
                append_jsonl(row)
                log(
                    f"{row['tensor_id']:22s} g0={row['g0_q4_g64_absmax']:.9f} "
                    f"m64={row['q4_g64_mse']:.9f} m128={row['q4_g128_mse']:.9f} "
                    f"d={row['delta_g128_mse_minus_g0']:+.6f} lose={int(row['loses_vs_g0'])} "
                    f"{row['wall_s']:.2f}s"
                )
    # tables + summarize
    run(verify_only=False)


def run_layer_list(layers: list[int], shard: Path) -> None:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    shard.parent.mkdir(parents=True, exist_ok=True)
    for L in layers:
        t0 = time.time()
        rows = process_layer(L)
        with shard.open("a") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        log(f"shard {shard.name} L{L} n={len(rows)} {time.time()-t0:.1f}s")


if __name__ == "__main__":
    import sys
    if "--verify" in sys.argv:
        run(verify_only=True)
    elif "--layers" in sys.argv:
        i = sys.argv.index("--layers")
        layers = [int(x) for x in sys.argv[i + 1].split(",") if x]
        shard = Path("/tmp/g1_q4_mse_g128_shards") / f"L{layers[0]}.jsonl"
        if "--shard" in sys.argv:
            shard = Path(sys.argv[sys.argv.index("--shard") + 1])
        run_layer_list(layers, shard)
    elif "--workers" in sys.argv:
        i = sys.argv.index("--workers")
        n = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 8
        run_parallel(workers=n)
    else:
        run(verify_only=False)

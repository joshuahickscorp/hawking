#!/usr/bin/env python3
"""G1 attention stack: MSE-optimal group scale + residual island + geometry + bitwidth.

CPU / numpy only. No GPU, no generate, no pack, no resident contact.
Writes /tmp/g1_attention_2bpw_stack.json and a running log.
"""
from __future__ import annotations

import gc
import json
import math
import resource
import struct
import time
from pathlib import Path

import numpy as np

SRC = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16")
CAP = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1")
OUT = Path("/tmp/g1_attention_2bpw_stack.json")
LOG = Path("/tmp/g1_attention_2bpw_stack.log")

HIDDEN = 5120
N_TOKENS = 256
N_LAYERS = 64
KEY_HEADS = 16
VALUES_PER_KEY = 3
KEY_DIM = 128
VALUE_DIM = 128
GQA_HEADS = 24
GQA_KV = 4
GQA_HEAD_DIM = 256

N = 26_895_998_464
E_MLP = 17_112_760_320
E_ATTN = 7_237_795_840
E_TAB = 2_542_796_800
E_SMALL = 2_645_504
TAB_BYTES_Q4 = 1_350_860_880  # G0 MEASURED embed+lm_head payload
SMALL_BYTES = 10_584_840  # G0 MEASURED f32 class payload
B_TAB_Q4 = 8.0 * TAB_BYTES_Q4 / E_TAB
B_SMALL = 8.0 * SMALL_BYTES / E_SMALL
B_MLP_LO = 0.848
B_MLP_HI = 0.989
G0_BPW = 4.252735126866492
ATTN_FUSED_TENSORS = 208
HEADER_BYTES = 40
ISLAND0 = 3994
# Contract inversion: (1.5*N - E_mlp*0.848 - 8*TAB_BYTES - 8*SMALL_BYTES) / E_attn
# = 2.064157091228481 exactly under integer-byte tables/small.

# Compile-time residual island, then mean-RMS continuation (filled at runtime).
ISLAND_SEED = [3994, 3456, 310]

LAYERS = (0, 3, 6, 7, 15, 32, 47, 63)
BITS = (2, 3, 4)
GROUPS = (16, 32, 48, 64, 128, 256, 512)
K_SWEEP = (0, 1, 3, 8, 32)
MSE_MULT = [0.50, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 2.00]
BINARY_GROUPS = (16, 32, 64, 128, 256)

# Fused attention GEMV census (G0 catalog names).
ATTN_SHAPES = (
    # count, rows, cols, island_axis ('row' write / 'col' read)
    (48, 16384, 5120, "col"),  # in_proj_qkvz
    (48, 96, 5120, "col"),  # in_proj_ba
    (48, 5120, 6144, "row"),  # out_proj
    (16, 12288, 5120, "col"),  # q_proj
    (16, 1024, 5120, "col"),  # k_proj
    (16, 1024, 5120, "col"),  # v_proj
    (16, 5120, 6144, "row"),  # o_proj
)

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


def bound_for(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def n_blocks(n_in: int, g: int) -> int:
    return (n_in + g - 1) // g


def absmax_recon_perrow(W: np.ndarray, bits: int, g: int) -> np.ndarray:
    """Per-row K-axis absmax RTN. Equals flat C-order when n_in % g == 0."""
    b = bound_for(bits)
    n_out, n_in = W.shape
    Wh = np.empty_like(W)
    for bi in range(n_blocks(n_in, g)):
        lo = bi * g
        hi = min(lo + g, n_in)
        w = W[:, lo:hi]
        amax = np.max(np.abs(w), axis=1)
        s = amax / max(b, 1)
        den = np.where(s > 0.0, s, 1.0)
        codes = np.clip(np.rint(w / den[:, None]), -b, b)
        codes = np.where((s > 0.0)[:, None], codes, 0.0)
        Wh[:, lo:hi] = codes * s[:, None]
    return Wh


def group_mse_recon(W: np.ndarray, X_fit: np.ndarray, bits: int, g: int, multipliers: list[float]) -> tuple[np.ndarray, dict]:
    """Per-group s = argmin_m ||X_g (w - q(w, s0*m))|| via 8-multiplier search.

    Surviving scale rule from g1-scale-contradiction.md. Vectorized over rows.
    """
    b = bound_for(bits)
    n_out, n_in = W.shape
    if X_fit.shape[1] != n_in:
        raise RuntimeError(f"X_fit in {X_fit.shape[1]} != W in {n_in}")
    Wh = np.empty_like(W)
    picked = np.zeros(len(multipliers), dtype=np.int64)
    improved = 0
    n_groups = 0
    W64 = np.ascontiguousarray(W, dtype=np.float64)
    X64 = np.ascontiguousarray(X_fit, dtype=np.float64)
    t0 = time.time()
    one_i = multipliers.index(1.0) if 1.0 in multipliers else 0
    for bi in range(n_blocks(n_in, g)):
        lo = bi * g
        hi = min(lo + g, n_in)
        Xg = X64[:, lo:hi]
        Grm = Xg.T @ Xg
        w = W64[:, lo:hi]
        amax = np.max(np.abs(w), axis=1)
        s0 = amax / max(b, 1)
        best_c = np.full(n_out, np.inf, dtype=np.float64)
        best_s = s0.copy()
        best_i = np.full(n_out, one_i, dtype=np.int32)
        zero = s0 <= 0.0
        if np.any(~zero):
            for i, m in enumerate(multipliers):
                s = s0 * m
                den = np.where(s > 0.0, s, 1.0)
                codes = np.clip(np.rint(w / den[:, None]), -b, b)
                codes = np.where((s > 0.0)[:, None], codes, 0.0)
                e = w - codes * s[:, None]
                c = np.sum((e @ Grm) * e, axis=1)
                better = (c < best_c) & (~zero)
                if np.any(better):
                    best_c = np.where(better, c, best_c)
                    best_s = np.where(better, s, best_s)
                    best_i = np.where(better, i, best_i)
        den = np.where(best_s > 0.0, best_s, 1.0)
        codes = np.clip(np.rint(w / den[:, None]), -b, b)
        codes = np.where((best_s > 0.0)[:, None], codes, 0.0)
        Wh[:, lo:hi] = (codes * best_s[:, None]).astype(np.float32)
        for i in range(len(multipliers)):
            picked[i] += int(np.sum(best_i == i))
        improved += int(np.sum((best_i != one_i) & (~zero)))
        n_groups += n_out
    meta = {
        "n_groups": int(n_groups),
        "n_groups_not_absmax": int(improved),
        "frac_groups_not_absmax": float(improved) / float(max(n_groups, 1)),
        "n_picked_per_multiplier": [int(x) for x in picked],
        "multipliers": multipliers,
        "wall_s": time.time() - t0,
    }
    return Wh, meta


def alpha_recon(W: np.ndarray, X_fit: np.ndarray, bits: int, g: int, alpha: float) -> np.ndarray:
    """AWQ-style fold with exponent alpha. alpha=0.25 is the cheap surviving proxy."""
    s = np.sqrt(np.mean(np.square(X_fit, dtype=np.float64), axis=0))
    s = np.maximum(s, 1e-8)
    if alpha != 1.0:
        s = np.power(s, float(alpha))
    s32 = s.astype(np.float32)
    Ws = W * s32[None, :]
    Whs = absmax_recon_perrow(Ws, bits, g)
    return (Whs / s32[None, :]).astype(np.float32)


def binary_meanabs_recon(W: np.ndarray, g: int) -> np.ndarray:
    """HGRAVB01: sign(w>=0) * f16(mean_abs) per-row K groups."""
    n_out, n_in = W.shape
    Wh = np.empty_like(W)
    for bi in range(n_blocks(n_in, g)):
        lo = bi * g
        hi = min(lo + g, n_in)
        w = W[:, lo:hi]
        scale = np.mean(np.abs(w), axis=1, dtype=np.float64).astype(np.float16).astype(np.float32)
        sgn = np.where(w >= 0.0, 1.0, -1.0)
        Wh[:, lo:hi] = sgn * scale[:, None]
    return Wh


def binary_mse_recon(W: np.ndarray, X_fit: np.ndarray, g: int) -> np.ndarray:
    """Closed-form MSE-optimal scale for sign(w>=0)*s against X_fit Gram."""
    n_out, n_in = W.shape
    Wh = np.empty_like(W)
    W64 = np.ascontiguousarray(W, dtype=np.float64)
    X64 = np.ascontiguousarray(X_fit, dtype=np.float64)
    for bi in range(n_blocks(n_in, g)):
        lo = bi * g
        hi = min(lo + g, n_in)
        Xg = X64[:, lo:hi]
        Grm = Xg.T @ Xg
        w = W64[:, lo:hi]
        sgn = np.where(w >= 0.0, 1.0, -1.0)
        num = np.sum((sgn @ Grm) * w, axis=1)
        den = np.sum((sgn @ Grm) * sgn, axis=1)
        s = np.divide(num, np.maximum(den, 1e-30))
        # keep non-negative; a negative LS scale is a flipped sign, reject
        s = np.maximum(s, 0.0)
        Wh[:, lo:hi] = (sgn * s[:, None]).astype(np.float32)
    return Wh


def rice_q1_rms_2pct_recon(W: np.ndarray, g: int = 128, ratio: float = 0.02) -> np.ndarray:
    """Quality reconstruction of HGRAVR02 rice_q1_rms_2pct (index coding does not change values)."""
    # Binary base is FLAT C-order groups, matching _binary_parts.
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    groups = math.ceil(flat.size / g)
    padded = np.zeros((groups, g), dtype=np.float32)
    padded.reshape(-1)[: flat.size] = flat
    scales = np.mean(np.abs(padded), axis=1, dtype=np.float64).astype(np.float16).astype(np.float32)
    base = (np.where(padded >= 0.0, 1.0, -1.0) * scales[:, None]).reshape(-1)[: flat.size]
    residual = flat - base
    count = max(1, int(math.ceil(flat.size * ratio)))
    idx = np.argpartition(np.abs(residual), -count)[-count:]
    sel = residual[idx]
    stat = float(np.sqrt(np.mean(np.square(sel)))) if sel.size else 0.0
    scale16 = np.float16(stat if stat > 0 else 1.0)
    scale_f = float(scale16)
    rec = base.copy()
    rec[idx] = np.where(sel >= 0.0, scale_f, -scale_f)
    return rec.reshape(W.shape).astype(np.float32)


def apply_island(Wh: np.ndarray, W: np.ndarray, axis: str, chs: list[int]) -> np.ndarray:
    if not chs:
        return Wh
    out = Wh.copy()
    if axis == "row":
        valid = [c for c in chs if 0 <= c < W.shape[0]]
        if valid:
            out[valid, :] = W[valid, :]
    else:
        valid = [c for c in chs if 0 <= c < W.shape[1]]
        if valid:
            out[:, valid] = W[:, valid]
    return out


def island_elems_attn(k: int, ranked: list[int]) -> int:
    """Exact bf16 island mass on fused attention GEMVs for the compile-time set of size k."""
    chs = ranked[:k]
    total = 0
    for count, rows, cols, axis in ATTN_SHAPES:
        if axis == "row":
            n = sum(1 for c in chs if c < rows)
            total += count * n * cols
        else:
            n = sum(1 for c in chs if c < cols)
            total += count * n * rows
    return int(total)


def payload_bytes_qn(bits: int, g: int) -> int:
    """HQ30-family payload: 40 B header + f16 scale + padded codes per per-row group."""
    total = 0
    code_b = math.ceil(bits * g / 8)
    for count, rows, cols, _axis in ATTN_SHAPES:
        ng = rows * n_blocks(cols, g)
        total += count * (HEADER_BYTES + ng * (2 + code_b))
    return int(total)


def payload_bytes_binary(g: int) -> int:
    """HGRAVB01-style: 40 B header stand-in + f16 scale + 1 sign bit, per-row groups."""
    total = 0
    for count, rows, cols, _axis in ATTN_SHAPES:
        ng = rows * n_blocks(cols, g)
        # last-group sign bits charged as full g (conservative pad)
        sign_bytes = math.ceil(g / 8)
        total += count * (HEADER_BYTES + ng * (2 + sign_bytes))
    return int(total)


def attn_bpw(body_bytes: int, island_elems: int) -> dict:
    island_bytes = island_elems * 2  # bf16
    bits_body = 8 * body_bytes
    bits_island = 16 * island_elems
    b_attn = (bits_body + bits_island) / float(E_ATTN)
    b_attn_body = bits_body / float(E_ATTN)

    def complete(b_mlp: float) -> float:
        return (
            E_MLP * b_mlp
            + E_ATTN * b_attn
            + 8.0 * TAB_BYTES_Q4
            + 8.0 * SMALL_BYTES
        ) / float(N)

    return {
        "body_bytes": int(body_bytes),
        "island_elems": int(island_elems),
        "island_bytes_bf16": int(island_bytes),
        "b_attn": float(b_attn),
        "b_attn_body_only": float(b_attn_body),
        "complete_mlp_0p848": float(complete(B_MLP_LO)),
        "complete_mlp_0p989": float(complete(B_MLP_HI)),
        "index_bits": 0,
    }


def inversion_table() -> dict:
    ceiling_bits = 1.5 * N
    tab_bits = 8.0 * TAB_BYTES_Q4
    small_bits = 8.0 * SMALL_BYTES

    def max_attn(b_mlp: float) -> float:
        rem = ceiling_bits - E_MLP * b_mlp - tab_bits - small_bits
        return rem / float(E_ATTN)

    return {
        "N": N,
        "E_mlp": E_MLP,
        "E_attn": E_ATTN,
        "E_tab": E_TAB,
        "E_small": E_SMALL,
        "tab_bytes_q4": TAB_BYTES_Q4,
        "small_bytes": SMALL_BYTES,
        "b_tab_q4": B_TAB_Q4,
        "b_small": B_SMALL,
        "complete_ceiling": 1.5,
        "max_b_attn_mlp_0p848": float(max_attn(B_MLP_LO)),
        "max_b_attn_mlp_0p989": float(max_attn(B_MLP_HI)),
        "threshold_about_2p07": 2.07,
        "formula": "(1.5*N - E_mlp*b_mlp - 8*1350860880 - 8*10584840) / E_attn",
    }


def score_hold(Y: np.ndarray, Yh: np.ndarray, W: np.ndarray, Wh: np.ndarray, R: np.ndarray | None) -> dict:
    rec = {
        "output_cosine": mean_row_cosine(Y, Yh),
        "output_cosine_min_row": min_row_cosine(Y, Yh),
        "output_rel_l2": rel_l2(Y, Yh),
        "weight_cosine": mean_row_cosine(W.reshape(1, -1), Wh.reshape(1, -1)),
        "weight_rel_l2": rel_l2(W, Wh),
        "space": "output_hold_odd_rows",
    }
    if R is not None and Y.shape[1] == R.shape[1]:
        rec["residual_proxy_cosine"] = mean_row_cosine(R + Y, R + Yh)
        rec["residual_proxy_rel_l2"] = rel_l2(R + Y, R + Yh)
        rec["write_rms_over_residual_rms"] = float(
            np.sqrt(np.mean(np.square(Y, dtype=np.float64)))
            / max(np.sqrt(np.mean(np.square(R, dtype=np.float64))), 1e-12)
        )
    else:
        rec["residual_proxy_cosine"] = None
        rec["residual_proxy_rel_l2"] = None
        rec["write_rms_over_residual_rms"] = None
    return rec


def ranked_island_channels() -> tuple[list[int], dict]:
    """Mean-RMS rank across 64 captured post-norm hiddens. Seed the honest set."""
    all_rms = np.zeros((N_LAYERS, HIDDEN), dtype=np.float64)
    l7_nz = None
    for li in range(N_LAYERS):
        x = load_hidden(li).astype(np.float64, copy=False)
        all_rms[li] = np.sqrt(np.mean(np.square(x), axis=0))
        if li == 7:
            l7_nz = int(np.count_nonzero(x[:, ISLAND0]))
        del x
    mean_rms = all_rms.mean(axis=0)
    order = [int(i) for i in np.argsort(mean_rms)[::-1]]
    # Honest compile-time set: 3994, 3456, 310 first, then mean-RMS continuation.
    ranked: list[int] = []
    for c in ISLAND_SEED:
        if c not in ranked:
            ranked.append(c)
    for c in order:
        if c not in ranked:
            ranked.append(c)
    meta = {
        "mean_rms_top16": [{"ch": int(c), "mean_rms": float(mean_rms[c])} for c in order[:16]],
        "seed": ISLAND_SEED,
        "ranked_head32": ranked[:32],
        "l7_ch3994_n_nonzero": l7_nz,
        "site": "CAPTURED_REAL_BF16_POST_NORM_HIDDEN",
        "n_tokens": N_TOKENS,
        "note": "256-token capture is underdetermined for magnitude; rank of the 3-set is compile-time and weight-native.",
    }
    return ranked, meta


def tensor_plan() -> list[dict]:
    plan = []
    for layer in LAYERS:
        if is_gqa(layer):
            plan.append({"layer": layer, "role": "q_proj", "suffix": "self_attn.q_proj.weight", "axis": "col", "x": "hidden"})
            plan.append({"layer": layer, "role": "k_proj", "suffix": "self_attn.k_proj.weight", "axis": "col", "x": "hidden"})
            plan.append({"layer": layer, "role": "v_proj", "suffix": "self_attn.v_proj.weight", "axis": "col", "x": "hidden"})
            plan.append({"layer": layer, "role": "o_proj", "suffix": "self_attn.o_proj.weight", "axis": "row", "x": "mixer_gqa"})
        else:
            plan.append({"layer": layer, "role": "in_proj_qkv", "suffix": "linear_attn.in_proj_qkv.weight", "axis": "col", "x": "hidden"})
            plan.append({"layer": layer, "role": "in_proj_z", "suffix": "linear_attn.in_proj_z.weight", "axis": "col", "x": "hidden"})
            plan.append({"layer": layer, "role": "out_proj", "suffix": "linear_attn.out_proj.weight", "axis": "row", "x": "mixer_dn"})
    return plan


def load_x_for(item: dict, Xh: np.ndarray) -> tuple[np.ndarray, str]:
    layer = item["layer"]
    kind = item["x"]
    if kind == "hidden":
        return Xh, "CAPTURED_REAL_BF16_POST_NORM_HIDDEN"
    if kind == "mixer_dn":
        Wqkv = load_tensor(tname(layer, "linear_attn.in_proj_qkv.weight"))
        Wz = load_tensor(tname(layer, "linear_attn.in_proj_z.weight"))
        fused = fuse_qkvz(Wqkv, Wz)
        del Wqkv, Wz
        Xm = deltanet_out_proxy(Xh, fused)
        del fused
        return Xm, "DERIVED_MIXER_PROXY_V_SILU_Z_NOT_CAPTURED_MIXER_X"
    if kind == "mixer_gqa":
        Wq = load_tensor(tname(layer, "self_attn.q_proj.weight"))
        Wv = load_tensor(tname(layer, "self_attn.v_proj.weight"))
        Xm = gqa_out_proxy(Xh, Wq, Wv)
        del Wq, Wv
        return Xm, "DERIVED_MIXER_PROXY_REPEAT_V_SIGMOID_GATE_NOT_CAPTURED_MIXER_X"
    raise RuntimeError(kind)


def dump(obj: dict) -> None:
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(OUT)


def run_sanity(ranked: list[int]) -> dict:
    """Reproduce the two stacked wins and the g=48 cell before the sweep."""
    log("SANITY L0 out_proj")
    Xh = load_hidden(0)
    W = load_tensor(tname(0, "linear_attn.out_proj.weight"))
    Wqkv = load_tensor(tname(0, "linear_attn.in_proj_qkv.weight"))
    Wz = load_tensor(tname(0, "linear_attn.in_proj_z.weight"))
    Xm = deltanet_out_proxy(Xh, fuse_qkvz(Wqkv, Wz))
    del Wqkv, Wz
    hold = np.arange(1, N_TOKENS, 2)
    fit = np.arange(0, N_TOKENS, 2)
    Xo, Xf = Xm[hold], Xm[fit]
    Rh = Xh[hold]
    Y = Xo @ W.T
    rec: dict = {
        "W_shape": [int(x) for x in W.shape],
        "kurtosis": float(excess_kurtosis(W)),
        "kurtosis_drop_row3994": float(excess_kurtosis(np.delete(W, ISLAND0, axis=0))),
        "row3994_rank_rms": int(1 + np.sum(np.sqrt(np.mean(np.square(W, dtype=np.float64), axis=1)) > np.sqrt(np.mean(np.square(W[ISLAND0], dtype=np.float64))))),
    }
    Wh = absmax_recon_perrow(W, 3, 64)
    Yh = Xo @ Wh.T
    rec["q3_g64_absmax"] = score_hold(Y, Yh, W, Wh, Rh)
    log(f"  q3_g64_absmax out={rec['q3_g64_absmax']['output_cosine']:.8f} (want 0.95310345)")

    Wh_i = apply_island(Wh, W, "row", ranked[:1])
    Yh = Xo @ Wh_i.T
    rec["q3_g64_absmax_k1"] = score_hold(Y, Yh, W, Wh_i, Rh)
    log(f"  q3_g64_absmax_k1 out={rec['q3_g64_absmax_k1']['output_cosine']:.8f} (want 0.961951)")
    del Wh, Wh_i

    Wh = absmax_recon_perrow(W, 3, 48)
    Yh = Xo @ Wh.T
    rec["q3_g48_absmax"] = score_hold(Y, Yh, W, Wh, Rh)
    log(f"  q3_g48_absmax out={rec['q3_g48_absmax']['output_cosine']:.8f} (want 0.96015278)")
    del Wh

    Wh, meta = group_mse_recon(W, Xf, 3, 64, MSE_MULT)
    Yh = Xo @ Wh.T
    rec["q3_g64_mse"] = score_hold(Y, Yh, W, Wh, Rh)
    rec["q3_g64_mse"]["mse_meta"] = meta
    log(f"  q3_g64_mse out={rec['q3_g64_mse']['output_cosine']:.8f} (want 0.97460) frac_moved={meta['frac_groups_not_absmax']:.4f}")

    Wh_i = apply_island(Wh, W, "row", ranked[:1])
    Yh = Xo @ Wh_i.T
    rec["q3_g64_mse_k1"] = score_hold(Y, Yh, W, Wh_i, Rh)
    log(f"  q3_g64_mse_k1 out={rec['q3_g64_mse_k1']['output_cosine']:.8f}")
    del Wh, Wh_i

    Wh = absmax_recon_perrow(W, 4, 64)
    rec["q4_g64_absmax_weight_cosine"] = mean_row_cosine(W.reshape(1, -1), Wh.reshape(1, -1))
    Yh = Xo @ Wh.T
    rec["q4_g64_absmax"] = score_hold(Y, Yh, W, Wh, Rh)
    log(f"  q4_g64_absmax out={rec['q4_g64_absmax']['output_cosine']:.8f} w={rec['q4_g64_absmax_weight_cosine']:.8f}")
    del Wh, W, Xm, Xh, Y
    rec["rss_gb"] = rss_gb()
    return rec


def score_config_grid(W, Xf, Xo, Rh, axis, ranked, bits_list, groups, do_binary: bool) -> list[dict]:
    rows: list[dict] = []
    Y = Xo @ W.T
    # Qn family
    for bits in bits_list:
        for g in groups:
            if Xf.shape[1] % g != 0 and g not in (48,):
                # still legal (short last group); keep
                pass
            bodies = {}
            mse_meta = None
            t0 = time.time()
            bodies["absmax"] = absmax_recon_perrow(W, bits, g)
            t_abs = time.time()
            bodies["mse"] = None
            try:
                bodies["mse"], mse_meta = group_mse_recon(W, Xf, bits, g, MSE_MULT)
            except Exception as exc:  # noqa: BLE001
                log(f"    MSE fail bits={bits} g={g}: {exc}")
            t_mse = time.time()
            bodies["a025"] = alpha_recon(W, Xf, bits, g, 0.25)
            t_a = time.time()
            body_bytes = payload_bytes_qn(bits, g)
            for scale_name, Wh0 in bodies.items():
                if Wh0 is None:
                    continue
                for k in K_SWEEP:
                    Wh = apply_island(Wh0, W, axis, ranked[:k]) if k else Wh0
                    Yh = Xo @ Wh.T
                    sc = score_hold(Y, Yh, W, Wh, Rh)
                    ie = island_elems_attn(k, ranked)
                    bpw = attn_bpw(body_bytes, ie)
                    row = {
                        "family": "uniform_qn",
                        "bits": int(bits),
                        "g": int(g),
                        "scale": scale_name,
                        "k": int(k),
                        "island_chs": ranked[:k],
                        **sc,
                        **bpw,
                        "quantize_wall_s": {
                            "absmax": t_abs - t0,
                            "mse": (t_mse - t_abs) if scale_name == "mse" else None,
                            "a025": t_a - t_mse,
                        }.get(scale_name),
                    }
                    if scale_name == "mse" and mse_meta is not None and k == 0:
                        row["mse_meta"] = mse_meta
                    rows.append(row)
                    if Wh is not Wh0:
                        del Wh, Yh
                    else:
                        del Yh
            for Wh0 in bodies.values():
                del Wh0
            log(
                f"    q{bits} g={g} abs={t_abs-t0:.2f}s mse={t_mse-t_abs:.2f}s a025={t_a-t_mse:.2f}s "
                f"abs_out={next(r['output_cosine'] for r in rows if r['bits']==bits and r['g']==g and r['scale']=='absmax' and r['k']==0):.5f} "
                f"mse_out={next((r['output_cosine'] for r in rows if r['bits']==bits and r['g']==g and r['scale']=='mse' and r['k']==0), float('nan')):.5f}"
            )
            gc.collect()

    if do_binary:
        for g in BINARY_GROUPS:
            t0 = time.time()
            bodies = {
                "meanabs": binary_meanabs_recon(W, g),
                "mse": binary_mse_recon(W, Xf, g),
            }
            body_bytes = payload_bytes_binary(g)
            for scale_name, Wh0 in bodies.items():
                for k in K_SWEEP:
                    Wh = apply_island(Wh0, W, axis, ranked[:k]) if k else Wh0
                    Yh = Xo @ Wh.T
                    sc = score_hold(Y, Yh, W, Wh, Rh)
                    ie = island_elems_attn(k, ranked)
                    bpw = attn_bpw(body_bytes, ie)
                    rows.append(
                        {
                            "family": "binary",
                            "bits": 1,
                            "g": int(g),
                            "scale": scale_name,
                            "k": int(k),
                            "island_chs": ranked[:k],
                            **sc,
                            **bpw,
                        }
                    )
                    if Wh is not Wh0:
                        del Wh, Yh
                    else:
                        del Yh
            log(f"    binary g={g} meanabs={next(r['output_cosine'] for r in rows if r['family']=='binary' and r['g']==g and r['scale']=='meanabs' and r['k']==0):.5f} mse={next(r['output_cosine'] for r in rows if r['family']=='binary' and r['g']==g and r['scale']=='mse' and r['k']==0):.5f} wall={time.time()-t0:.2f}s")
            del bodies
            gc.collect()

        # Existing 1.291 attention encoding, quality only; BPW from packed artifact.
        t0 = time.time()
        Wh0 = rice_q1_rms_2pct_recon(W, g=128, ratio=0.02)
        for k in (0, 1, 3):
            Wh = apply_island(Wh0, W, axis, ranked[:k]) if k else Wh0
            Yh = Xo @ Wh.T
            sc = score_hold(Y, Yh, W, Wh, Rh)
            # MEASURED packed attention rice BPW; island additive on top.
            ie = island_elems_attn(k, ranked)
            b_attn = 1.2877935788805008 + (16.0 * ie) / float(E_ATTN)
            rows.append(
                {
                    "family": "rice_q1_rms_2pct",
                    "bits": None,
                    "g": 128,
                    "scale": "binary_meanabs_plus_top2pct_sign_rms",
                    "k": int(k),
                    "island_chs": ranked[:k],
                    **sc,
                    "b_attn": float(b_attn),
                    "b_attn_body_only": 1.2877935788805008,
                    "island_elems": int(ie),
                    "bpw_source": "MEASURED mixed-sub15-v1 attention_gemv_rice + DERIVED island",
                }
            )
            if Wh is not Wh0:
                del Wh, Yh
            else:
                del Yh
        log(f"    rice_q1_rms_2pct k0 out={rows[-3]['output_cosine']:.5f} wall={time.time()-t0:.2f}s")
        del Wh0
    del Y
    return rows


def summarize(all_rows: list[dict]) -> dict:
    """Mass-unweighted across scored GEMVs. Also worst-cell and under-2.07 pick."""
    # group by config key
    from collections import defaultdict

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        key = (r["family"], r.get("bits"), r["g"], r["scale"], r["k"])
        buckets[key].append(r)

    summary = []
    for key, rs in buckets.items():
        family, bits, g, scale, k = key
        outs = [x["output_cosine"] for x in rs]
        mins = [x["output_cosine_min_row"] for x in rs]
        ws = [x["weight_cosine"] for x in rs]
        b_attns = [x.get("b_attn") for x in rs if x.get("b_attn") is not None]
        b_attn = float(b_attns[0]) if b_attns else None
        n_ge = int(sum(1 for c in outs if c >= 0.99))
        rec = {
            "family": family,
            "bits": bits,
            "g": g,
            "scale": scale,
            "k": k,
            "n_tensors": len(rs),
            "mean_output_cosine": float(np.mean(outs)),
            "min_output_cosine": float(np.min(outs)),
            "max_output_cosine": float(np.max(outs)),
            "n_ge_0p99": n_ge,
            "all_ge_0p99": bool(n_ge == len(rs)),
            "mean_output_cosine_min_row": float(np.mean(mins)),
            "min_output_cosine_min_row": float(np.min(mins)),
            "mean_weight_cosine": float(np.mean(ws)),
            "b_attn": b_attn,
            "under_2p07": bool(b_attn is not None and b_attn <= 2.07),
            "under_2p064157": bool(b_attn is not None and b_attn <= 2.064157091228481),
            "worst": min(rs, key=lambda x: x["output_cosine"])["tensor_id"] if rs and "tensor_id" in rs[0] else None,
        }
        # attach complete BPW if present
        if rs[0].get("complete_mlp_0p848") is not None:
            rec["complete_mlp_0p848"] = rs[0]["complete_mlp_0p848"]
            rec["complete_mlp_0p989"] = rs[0]["complete_mlp_0p989"]
        summary.append(rec)

    def sort_key(r: dict) -> tuple:
        return (
            0 if r["under_2p07"] else 1,
            -(r["min_output_cosine"]),
            r["b_attn"] if r["b_attn"] is not None else 99.0,
        )

    summary.sort(key=sort_key)

    under = [r for r in summary if r["under_2p07"]]
    best_under = None
    if under:
        # best = highest min-cell output cosine, then highest mean, then lowest bpw
        under_sorted = sorted(under, key=lambda r: (-r["min_output_cosine"], -r["mean_output_cosine"], r["b_attn"] or 99))
        best_under = under_sorted[0]

    cleared = [r for r in summary if r["all_ge_0p99"]]
    floor = None
    if cleared:
        floor = min(cleared, key=lambda r: r["b_attn"] if r["b_attn"] is not None else 99.0)

    # also mean>=0.99 floor (weaker bar)
    mean_cleared = [r for r in summary if r["mean_output_cosine"] >= 0.99]
    mean_floor = None
    if mean_cleared:
        mean_floor = min(mean_cleared, key=lambda r: r["b_attn"] if r["b_attn"] is not None else 99.0)

    return {
        "n_configs": len(summary),
        "best_under_2p07": best_under,
        "lowest_bpw_all_cells_ge_0p99": floor,
        "lowest_bpw_mean_ge_0p99": mean_floor,
        "configs": summary,
    }


def main() -> None:
    LOG.write_text("")
    t0 = time.time()
    inv = inversion_table()
    log(f"inversion max_b_attn mlp0.848={inv['max_b_attn_mlp_0p848']:.15f} mlp0.989={inv['max_b_attn_mlp_0p989']:.15f}")

    ranked, rank_meta = ranked_island_channels()
    log(f"island ranked[:8]={ranked[:8]} L7.3994_nz={rank_meta['l7_ch3994_n_nonzero']}")

    out: dict = {
        "schema": "hawking.g1.attention_2bpw_stack.v1",
        "protocol": {
            "metric": "mean row-cosine of Y_hold = X_hold @ W.T, odd rows of 256-token capture",
            "fit": "even rows (128)",
            "hold": "odd rows (128)",
            "scale_rule": "per-group s=argmin_m ||X_g (w-q(w,s0*m))||, m in {0.50,0.70,0.85,1.00,1.15,1.30,1.50,2.00}",
            "island": "compile-time residual channels, overlay exact bf16, index bits=0",
            "grouping": "per-row along K; equals flat C-order iff K%g==0",
            "mixer_x": "NEVER CAPTURED. out_proj/o_proj use derived mixer-site proxy.",
            "no_gpu": True,
        },
        "inversion": inv,
        "island_rank": rank_meta,
        "sanity": None,
        "tensors": [],
        "rows": [],
    }
    dump(out)

    sanity = run_sanity(ranked)
    out["sanity"] = sanity
    dump(out)

    plan = tensor_plan()
    all_rows: list[dict] = []
    for i, item in enumerate(plan):
        layer, role = item["layer"], item["role"]
        tid = f"L{layer}.{role}"
        log(f"===== {tid} ({i+1}/{len(plan)}) mixer={item['x']} =====")
        Xh = load_hidden(layer)
        W = load_tensor(tname(layer, item["suffix"]))
        X, site = load_x_for(item, Xh)
        if X.shape[1] != W.shape[1]:
            raise RuntimeError(f"{tid} X.in {X.shape[1]} != W.in {W.shape[1]}")
        hold = np.arange(1, X.shape[0], 2)
        fit = np.arange(0, X.shape[0], 2)
        Xo, Xf = X[hold], X[fit]
        Rh = Xh[hold] if W.shape[0] == HIDDEN else None
        rec = {
            "tensor_id": tid,
            "layer": layer,
            "role": role,
            "mixer": "gqa" if is_gqa(layer) else "delta_net",
            "W_shape": [int(x) for x in W.shape],
            "X_shape": [int(x) for x in X.shape],
            "X_site": site,
            "island_axis": item["axis"],
            "n_fit": int(fit.size),
            "n_hold": int(hold.size),
            "kurtosis": float(excess_kurtosis(W)),
        }
        # Binary + rice only on a representative subset to keep wall in budget,
        # but Qn stack on every tensor. Existing-candidate rice/binary on every
        # tensor: rice is one recon, cheap enough. Binary is 5 g * 2 scales.
        do_binary = True
        rows = score_config_grid(W, Xf, Xo, Rh, item["axis"], ranked, BITS, GROUPS, do_binary)
        for r in rows:
            r["tensor_id"] = tid
            r["layer"] = layer
            r["role"] = role
            r["X_site"] = site
            r["mixer"] = rec["mixer"]
            r["W_shape"] = rec["W_shape"]
        rec["n_rows"] = len(rows)
        # keep only compact per-tensor highlights in the tensor list
        highlights = []
        for r in rows:
            if r["family"] == "uniform_qn" and r["g"] in (48, 64, 256) and r["k"] in (0, 1, 3) and r["scale"] in ("absmax", "mse"):
                highlights.append(
                    {
                        "cfg": f"q{r['bits']}_g{r['g']}_{r['scale']}_k{r['k']}",
                        "output_cosine": r["output_cosine"],
                        "weight_cosine": r["weight_cosine"],
                        "b_attn": r.get("b_attn"),
                    }
                )
        rec["highlights"] = highlights
        out["tensors"].append(rec)
        all_rows.extend(rows)
        out["rows"] = all_rows
        out["wall_s"] = time.time() - t0
        out["rss_max_gb"] = rss_gb()
        dump(out)
        log(f"  wrote {len(rows)} rows rss={rss_gb():.3f}G")
        del W, X, Xh, Xo, Xf, rows
        gc.collect()

    out["summary"] = summarize(all_rows)
    out["wall_s"] = time.time() - t0
    out["rss_max_gb"] = rss_gb()
    dump(out)
    bu = out["summary"]["best_under_2p07"]
    fl = out["summary"]["lowest_bpw_all_cells_ge_0p99"]
    log(f"BEST under 2.07: {bu}")
    log(f"0.99 FLOOR (all cells): {fl}")
    log(f"DONE wall={out['wall_s']:.1f}s rss_max={out['rss_max_gb']:.3f}G n_rows={len(all_rows)}")


if __name__ == "__main__":
    main()

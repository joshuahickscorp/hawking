#!/usr/bin/env python3
"""Qwen3.8 group-partition geometry sweep. CPU/numpy only. No GPU, no generate.

Writes /tmp/g1_group_partition_geometry_sweep.json and prints a machine log.
"""
from __future__ import annotations

import json
import math
import resource
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16")
CAP = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
)
MANIFEST = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json"
)
OUT = Path("/tmp/g1_group_partition_geometry_sweep.json")

HIDDEN = 5120
KEY_HEADS = 16
VALUES_PER_KEY = 3
KEY_DIM = 128
VALUE_DIM = 128
GQA_HEADS = 24
GQA_KV = 4
GQA_HEAD_DIM = 256
LINEAR_VALUE_HEADS = 48
LINEAR_VALUE_DIM = 128
N_ALL = 26_895_998_464
F32_BYTES = 10_584_840
HEADER_RANK2 = 40
TPR = 64
TG = 128
ROWS_PER_TG = 2
THREAD_TILE = 8
STRIDE = TPR * THREAD_TILE  # 512
INCUMBENT_G = 64

LAYERS_DN = [0, 8, 16, 32, 48, 60]
LAYERS_GQA = [3, 15, 31, 47, 63]
GROUP_SIZES = [8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 256]
BITS_MAIN = (3, 4)
OFFSETS = (0, 16, 32, 48, 64)
OFFSET_GS = (32, 48, 64, 96, 128)
AXIS2_GS = (16, 32, 48, 64, 128)
AXIS2_BITS = (3, 4)
TWO_D_TILES = ((2, 32), (2, 48), (2, 64), (2, 128), (4, 64))
SLOT_GS = (32, 64, 128)

# Layers that also get offset / other-axis / gaussian-X discriminators.
DEEP_LAYERS = {0, 3, 32, 63}


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] rss_max={rss_gb():.3f}G {msg}", flush=True)


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
    if raw.size != 256 * HIDDEN:
        raise RuntimeError(f"hidden L{layer} size {raw.size}")
    return np.ascontiguousarray(raw.reshape(256, HIDDEN))


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


# ---------------------------------------------------------------------------
# complete BPW + kernel mapping
# ---------------------------------------------------------------------------

def code_bytes_per_group(bits: int, g: int) -> int:
    return (bits * g + 7) // 8


def n_groups_1d(length: int, g: int, phase: int) -> int:
    if length <= 0:
        return 0
    if phase <= 0:
        return (length + g - 1) // g
    if phase >= length:
        return 1
    return 1 + (length - phase + g - 1) // g


def tensor_payload_bytes(
    rows: int,
    cols: int,
    bits: int,
    *,
    axis: str,
    g: int,
    phase: int = 0,
    gm: int = 1,
    gk: int | None = None,
) -> int:
    """HQ30-style: rank-2 header + f16 scale/group + per-group full-slot codes."""
    if axis == "K_per_row":
        ng = n_groups_1d(cols, g, phase) * rows
        cpg = code_bytes_per_group(bits, g)
    elif axis == "K_flat":
        ng = n_groups_1d(rows * cols, g, 0)
        cpg = code_bytes_per_group(bits, g)
    elif axis == "M_per_col":
        ng = n_groups_1d(rows, g, phase) * cols
        cpg = code_bytes_per_group(bits, g)
    elif axis == "2D":
        assert gk is not None
        ng_r = n_groups_1d(rows, gm, 0)
        ng_c = n_groups_1d(cols, gk, phase)
        ng = ng_r * ng_c
        cpg = code_bytes_per_group(bits, gm * gk)
    else:
        raise ValueError(axis)
    return HEADER_RANK2 + ng * (2 + cpg)


def tensor_complete_bpw(rows: int, cols: int, bits: int, **kw) -> float:
    n = rows * cols
    return 8.0 * tensor_payload_bytes(rows, cols, bits, **kw) / float(n)


def model_complete_bpw(q4_shapes: list[tuple[int, int]], bits: int, **kw) -> dict:
    q_bytes = 0
    pad_groups = 0
    exact_groups = 0
    for rows, cols in q4_shapes:
        q_bytes += tensor_payload_bytes(rows, cols, bits, **kw)
        axis = kw.get("axis", "K_per_row")
        g = kw["g"]
        phase = kw.get("phase", 0)
        if axis == "K_per_row":
            ng = n_groups_1d(cols, g, phase)
            exact = cols % g == 0 and phase == 0
        elif axis == "M_per_col":
            ng = n_groups_1d(rows, g, phase)
            exact = rows % g == 0 and phase == 0
        elif axis == "2D":
            gm = kw["gm"]
            gk = kw["gk"]
            exact = (rows % gm == 0) and (cols % gk == 0) and phase == 0
        else:
            exact = (rows * cols) % g == 0
        if exact:
            exact_groups += 1
        else:
            pad_groups += 1
    total = q_bytes + F32_BYTES
    return {
        "q4_file_bytes": int(q_bytes),
        "f32_file_bytes": int(F32_BYTES),
        "payload_bytes": int(total),
        "complete_bpw": 8.0 * total / float(N_ALL),
        "tensors_exact_div": int(exact_groups),
        "tensors_need_short_group": int(pad_groups),
        "label": "MEASURED_FROM_402_SHAPES",
    }


def kernel_verdict(
    *,
    axis: str,
    bits: int,
    g: int,
    phase: int = 0,
    gm: int = 1,
    gk: int | None = None,
    cols: int | None = None,
) -> dict:
    """Can geo_tpr64_tg128's 64-thread × 8-wide K walk consume this without
    cross-lane shuffle or a second pass?

    Cheap means: each 8-wide tile shares one scale, scale is a function of
    (row, col) this thread already knows, codes are a fixed-size per-group
    blob this thread can load alone.
    """
    rec = {
        "axis": axis,
        "bits": bits,
        "g": g,
        "phase": phase,
        "gm": gm,
        "gk": gk,
        "tpr": TPR,
        "tg": TG,
        "rows_per_tg": ROWS_PER_TG,
        "thread_tile": THREAD_TILE,
        "stride": STRIDE,
    }
    if axis == "M_per_col":
        rec["class"] = "NOT_CHEAP"
        rec["reason"] = (
            "M-axis: the 8 consecutive K elements a thread unpacks have 8 "
            "distinct scales. unpack8 dies. No shuffle strictly required "
            "(thread can load 8 scales) but it is not the cheap mapping."
        )
        return rec
    if axis == "K_flat":
        if cols is not None and cols % g == 0 and phase == 0:
            rec["class"] = "SAME_AS_K_PER_ROW"
            rec["reason"] = "K divides g and phase=0, so flat groups never leave a row."
        else:
            rec["class"] = "NOT_CHEAP"
            rec["reason"] = (
                "Flat C-order groups straddle output rows. The kernel indexes "
                "scales as row*groups_per_row + col/g. A scale shared by the "
                "end of row r and the start of row r+1 needs a second pass "
                "or a different index."
            )
            return rec
        axis = "K_per_row"
    gk_eff = gk if axis == "2D" else g
    phase_eff = phase
    eight_ok = (gk_eff % THREAD_TILE == 0) and (phase_eff % THREAD_TILE == 0)
    if not eight_ok:
        rec["class"] = "NOT_CHEAP"
        rec["reason"] = (
            f"8-wide tile straddles a group (gk={gk_eff}, phase={phase_eff}). "
            "Needs multi-scale unpack or a cross-lane shuffle."
        )
        return rec
    if axis == "2D":
        rec["class"] = "CHEAP_REWRITE"
        rec["reason"] = (
            f"2D gm={gm} gk={gk_eff}: each 8-pack shares one scale; "
            f"scale index (row/{gm}, col/{gk_eff}). Production TG already "
            "owns 2 consecutive rows, so gm=2 is a natural map. Constant "
            "change + scale layout change; no shuffle, no second pass."
        )
        return rec
    if bits == 4:
        rec["class"] = "CHEAP"
        rec["reason"] = (
            f"Q4 K-axis g={g} phase={phase}: same 64-tpr / 8-unpack / "
            "stride-512 walk. Change GROUP_SIZE and CODE_BYTES_PER_GROUP. "
            "uint load stays aligned because g%8==0 and phase%8==0."
        )
    else:
        rec["class"] = "CHEAP_UNPACK_REWRITE"
        rec["reason"] = (
            f"Q{bits} K-axis g={g} phase={phase}: the 64-tpr / 8-tile map "
            "still owns one scale per tile, but the nibble-uint unpack is "
            "Q4-specific. Bit-packed Qn needs a new unpack, not a shuffle "
            "and not a second pass."
        )
    return rec


# ---------------------------------------------------------------------------
# quantizers
# ---------------------------------------------------------------------------

def _absmax_last(arr: np.ndarray, bits: int) -> np.ndarray:
    bound = float((1 << (bits - 1)) - 1)
    absmax = np.max(np.abs(arr), axis=-1, keepdims=True)
    scale = absmax / bound
    den = np.where(scale > 0.0, scale, 1.0)
    codes = np.rint(arr / den).clip(-bound, bound)
    return (codes * scale).astype(np.float32)


def quant_k_per_row(W: np.ndarray, bits: int, g: int, phase: int = 0) -> np.ndarray:
    W = np.ascontiguousarray(W, dtype=np.float32)
    rows, cols = W.shape
    out = np.empty_like(W)
    if phase > 0:
        out[:, :phase] = _absmax_last(W[:, :phase], bits)
        rest = W[:, phase:]
        dest0 = phase
    else:
        rest = W
        dest0 = 0
    kr = rest.shape[1]
    nfull = kr // g
    rem = kr - nfull * g
    if nfull:
        block = rest[:, : nfull * g].reshape(rows, nfull, g)
        q = _absmax_last(block, bits).reshape(rows, nfull * g)
        out[:, dest0 : dest0 + nfull * g] = q
    if rem:
        out[:, dest0 + nfull * g :] = _absmax_last(rest[:, nfull * g :], bits)
    return out


def quant_k_flat(W: np.ndarray, bits: int, g: int) -> np.ndarray:
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    n = int(flat.size)
    groups = (n + g - 1) // g
    padded = np.zeros((groups, g), dtype=np.float32)
    padded.reshape(-1)[:n] = flat
    q = _absmax_last(padded, bits).reshape(-1)[:n]
    return q.reshape(W.shape)


def quant_m_per_col(W: np.ndarray, bits: int, g: int, phase: int = 0) -> np.ndarray:
    return quant_k_per_row(W.T, bits, g, phase).T.copy()


def quant_2d(W: np.ndarray, bits: int, gm: int, gk: int) -> np.ndarray:
    W = np.ascontiguousarray(W, dtype=np.float32)
    rows, cols = W.shape
    out = np.empty_like(W)
    r0 = (rows // gm) * gm
    c0 = (cols // gk) * gk
    if r0 and c0:
        t = W[:r0, :c0].reshape(r0 // gm, gm, c0 // gk, gk)
        bound = float((1 << (bits - 1)) - 1)
        absmax = np.max(np.abs(t), axis=(1, 3), keepdims=True)
        scale = absmax / bound
        den = np.where(scale > 0.0, scale, 1.0)
        codes = np.rint(t / den).clip(-bound, bound)
        out[:r0, :c0] = (codes * scale).reshape(r0, c0)
    if c0 < cols:
        out[:r0, c0:] = quant_k_per_row(W[:r0, c0:], bits, gk, 0)
    if r0 < rows:
        out[r0:, :] = quant_k_per_row(W[r0:, :], bits, gk, 0)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_output(Y: np.ndarray, Yq: np.ndarray, W: np.ndarray, Wq: np.ndarray) -> dict:
    wc = mean_row_cosine(W.reshape(1, -1), Wq.reshape(1, -1))
    wr = rel_l2(W, Wq)
    oc = mean_row_cosine(Y, Yq)
    ol = rel_l2(Y, Yq)
    return {
        "weight_cosine": wc,
        "weight_rel_l2": wr,
        "output_cosine": oc,
        "output_cosine_min_row": min_row_cosine(Y, Yq),
        "output_rel_l2": ol,
        "output_over_weight_rel_l2": (ol / wr) if wr > 1e-12 else None,
    }


def head_slot_error(W: np.ndarray, Wq: np.ndarray, head_dim: int, g: int) -> dict:
    rows, cols = W.shape
    rec = {
        "head_dim": head_dim,
        "group_size": g,
        "n_heads": cols // head_dim if head_dim else None,
        "in_dim_divisible_by_g": cols % g == 0,
        "head_divisible_by_g": (head_dim % g == 0) if head_dim else None,
        "groups_aligned_to_head": bool(head_dim and (head_dim % g == 0) and (cols % g == 0)),
        "groups_straddle_head": bool(head_dim and not (head_dim % g == 0)),
    }
    if not rec["groups_aligned_to_head"]:
        return rec
    gph = head_dim // g
    n_heads = cols // head_dim
    err2 = np.square(W.astype(np.float64) - Wq.astype(np.float64))
    e = err2.reshape(rows, n_heads, gph, g)
    slot = e.mean(axis=(0, 1, 3))
    rec["groups_per_head"] = int(gph)
    rec["mean_sq_err_by_intra_head_slot"] = [float(x) for x in slot]
    rec["slot_max_over_min"] = float(np.max(slot) / np.min(slot)) if np.min(slot) > 0 else None
    return rec


def score_config(
    *,
    W: np.ndarray,
    Wq: np.ndarray,
    Y: np.ndarray,
    X_hold: np.ndarray,
    bits: int,
    axis: str,
    g: int,
    phase: int = 0,
    gm: int = 1,
    gk: int | None = None,
    head_dim: int | None = None,
    want_slots: bool = False,
) -> dict:
    Yq = X_hold @ Wq.T
    rows, cols = W.shape
    rec = score_output(Y, Yq, W, Wq)
    rec.update(
        {
            "bits": bits,
            "axis": axis,
            "g": g,
            "phase": phase,
            "gm": gm,
            "gk": gk,
            "tensor_complete_bpw": tensor_complete_bpw(
                rows, cols, bits, axis=axis, g=g, phase=phase, gm=gm, gk=gk
            ),
            "body_bpw_if_exact": bits + 16.0 / float((gm * (gk or g))),
            "K_divisible_by_g": (cols % (gk or g) == 0) and phase == 0,
            "kernel": kernel_verdict(
                axis=axis, bits=bits, g=g, phase=phase, gm=gm, gk=gk, cols=cols
            ),
        }
    )
    if want_slots and head_dim is not None and axis == "K_per_row" and phase == 0:
        rec["head_slots"] = head_slot_error(W, Wq, head_dim, g)
    return rec


# ---------------------------------------------------------------------------
# per-tensor
# ---------------------------------------------------------------------------

def sweep_tensor(
    *,
    label: str,
    role: str,
    organ: str,
    layer: int,
    W: np.ndarray,
    X: np.ndarray,
    head_dim: int | None,
    deep: bool,
) -> dict:
    log(f"eval {label} shape={tuple(W.shape)} organ={organ} deep={deep}")
    hold = np.arange(1, X.shape[0], 2)
    X_hold = X[hold]
    Y = X_hold @ W.T
    rows, cols = W.shape
    rec: dict = {
        "label": label,
        "role": role,
        "organ": organ,
        "layer": layer,
        "shape": [int(rows), int(cols)],
        "mixer": "gqa" if is_gqa(layer) else "delta_net",
        "head_dim": head_dim,
        "hold_n": int(hold.size),
        "hold_rule": "odd rows of 256-token capture, matching g1-out-proj-forensics",
        "configs": [],
    }

    # Phase 1: K-axis, phase 0, all group sizes, Q3 and Q4.
    for bits in BITS_MAIN:
        for g in GROUP_SIZES:
            Wq = quant_k_per_row(W, bits, g, 0)
            want_slots = head_dim is not None and bits == 3 and g in SLOT_GS
            rec["configs"].append(
                score_config(
                    W=W,
                    Wq=Wq,
                    Y=Y,
                    X_hold=X_hold,
                    bits=bits,
                    axis="K_per_row",
                    g=g,
                    phase=0,
                    head_dim=head_dim,
                    want_slots=want_slots,
                )
            )
            del Wq

    if not deep:
        return rec

    # Phase 2: alignment offset vs head boundaries (Q3, selected g).
    for g in OFFSET_GS:
        for phase in OFFSETS:
            if phase == 0:
                continue  # already have phase 0
            if phase >= g:
                continue
            Wq = quant_k_per_row(W, 3, g, phase)
            rec["configs"].append(
                score_config(
                    W=W,
                    Wq=Wq,
                    Y=Y,
                    X_hold=X_hold,
                    bits=3,
                    axis="K_per_row",
                    g=g,
                    phase=phase,
                    head_dim=head_dim,
                )
            )
            del Wq

    # Phase 3: other axis and 2D tiles.
    for bits in AXIS2_BITS:
        for g in AXIS2_GS:
            Wq = quant_m_per_col(W, bits, g, 0)
            rec["configs"].append(
                score_config(
                    W=W,
                    Wq=Wq,
                    Y=Y,
                    X_hold=X_hold,
                    bits=bits,
                    axis="M_per_col",
                    g=g,
                    phase=0,
                )
            )
            del Wq
        for gm, gk in TWO_D_TILES:
            Wq = quant_2d(W, bits, gm, gk)
            rec["configs"].append(
                score_config(
                    W=W,
                    Wq=Wq,
                    Y=Y,
                    X_hold=X_hold,
                    bits=bits,
                    axis="2D",
                    g=gk,
                    phase=0,
                    gm=gm,
                    gk=gk,
                )
            )
            del Wq

    # Flat C-order only when it differs from per-row (K % g != 0), Q3.
    for g in (48, 96, 40, 24):
        if cols % g == 0:
            continue
        Wq = quant_k_flat(W, 3, g)
        rec["configs"].append(
            score_config(
                W=W,
                Wq=Wq,
                Y=Y,
                X_hold=X_hold,
                bits=3,
                axis="K_flat",
                g=g,
                phase=0,
                head_dim=head_dim,
            )
        )
        del Wq

    # Gaussian X discriminator: is a g=48 win weight-structure or act-structure?
    rng = np.random.default_rng(38)
    rms = float(np.sqrt(np.mean(np.square(X_hold, dtype=np.float64))))
    Xg = rng.standard_normal((X_hold.shape[0], X_hold.shape[1]), dtype=np.float32)
    Xg *= np.float32(rms)
    Yg = Xg @ W.T
    gauss = []
    for g in (32, 48, 64, 96, 128):
        Wq = quant_k_per_row(W, 3, g, 0)
        Yq = Xg @ Wq.T
        gauss.append(
            {
                "g": g,
                "bits": 3,
                "axis": "K_per_row",
                "output_cosine": mean_row_cosine(Yg, Yq),
                "weight_cosine": mean_row_cosine(W.reshape(1, -1), Wq.reshape(1, -1)),
            }
        )
        del Wq
    rec["gaussian_x_q3"] = {
        "note": "iid N(0, rms(X_hold)^2), same hold_n; discriminator only",
        "x_rms": rms,
        "rows": gauss,
    }
    return rec


def dump(obj: dict) -> None:
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(OUT)


def main() -> None:
    t0 = time.time()
    if rss_gb() > 16.0:
        raise RuntimeError(f"RSS already {rss_gb():.3f}G before start")

    manifest = json.loads(MANIFEST.read_text())
    q4_shapes = [tuple(t["shape"]) for t in manifest["tensors"] if t["kind"] == "q4"]
    q4_elems = sum(a * b for a, b in q4_shapes)
    log(f"manifest q4 tensors={len(q4_shapes)} elems={q4_elems}")

    unique_k = sorted({k for _, k in q4_shapes})
    unique_mk = sorted(set(q4_shapes))
    div_table = []
    for g in GROUP_SIZES:
        row = {"g": g, "divides_all_K": all(k % g == 0 for k in unique_k)}
        for k in unique_k:
            row[f"K{k}_rem"] = k % g
        div_table.append(row)

    recipes = []
    for bits in (2, 3, 4):
        for g in GROUP_SIZES:
            recipes.append(
                {
                    "name": f"global_q{bits}_g{g}_K",
                    "bits": bits,
                    **model_complete_bpw(q4_shapes, bits, axis="K_per_row", g=g, phase=0),
                    "kernel": kernel_verdict(axis="K_per_row", bits=bits, g=g, phase=0),
                }
            )
        for gm, gk in TWO_D_TILES:
            recipes.append(
                {
                    "name": f"global_q{bits}_2d_{gm}x{gk}",
                    "bits": bits,
                    **model_complete_bpw(
                        q4_shapes, bits, axis="2D", g=gk, phase=0, gm=gm, gk=gk
                    ),
                    "kernel": kernel_verdict(
                        axis="2D", bits=bits, g=gk, phase=0, gm=gm, gk=gk
                    ),
                }
            )

    # Hybrid: g=48 only where 48 | K (the 64 out/o tensors), incumbent g=64 else.
    def hybrid_bytes(bits: int, special_k: int, special_g: int, default_g: int) -> dict:
        q_bytes = 0
        n_special = 0
        for rows, cols in q4_shapes:
            g = special_g if cols == special_k else default_g
            if cols == special_k:
                n_special += 1
            q_bytes += tensor_payload_bytes(
                rows, cols, bits, axis="K_per_row", g=g, phase=0
            )
        total = q_bytes + F32_BYTES
        return {
            "name": f"hybrid_q{bits}_g{special_g}_on_K{special_k}_else_g{default_g}",
            "bits": bits,
            "n_special_tensors": n_special,
            "q4_file_bytes": int(q_bytes),
            "payload_bytes": int(total),
            "complete_bpw": 8.0 * total / float(N_ALL),
            "kernel_special": kernel_verdict(
                axis="K_per_row", bits=bits, g=special_g, phase=0
            ),
            "kernel_default": kernel_verdict(
                axis="K_per_row", bits=bits, g=default_g, phase=0
            ),
        }

    recipes.append(hybrid_bytes(4, 6144, 48, 64))
    recipes.append(hybrid_bytes(3, 6144, 48, 64))
    recipes.append(hybrid_bytes(4, 6144, 32, 64))
    recipes.append(hybrid_bytes(3, 6144, 32, 64))

    # Incumbent check.
    inc = model_complete_bpw(q4_shapes, 4, axis="K_per_row", g=64, phase=0)
    log(
        f"incumbent reconstructed complete_bpw={inc['complete_bpw']:.15f} "
        f"q4_bytes={inc['q4_file_bytes']} (manifest q4=14287109840)"
    )

    results: dict = {
        "schema": "hawking.g1.qwen38_group_partition_geometry.v1",
        "date": "2026-08-17",
        "source": str(SRC),
        "activation": str(CAP),
        "manifest": str(MANIFEST),
        "claim_boundary": {
            "no_gpu": True,
            "no_generate": True,
            "no_pack": True,
            "not_a_token_level_claim": True,
            "activations_are_real_bf16_post_norm_hidden": True,
            "out_proj_x_is_mixer_site_proxy": True,
            "down_x_is_reconstructed_swiglu": True,
            "holdout": "odd rows of 256",
        },
        "codec": {
            "name": "uniform absmax RTN, HQ30UQ4 / HGRAVU01 family",
            "bound": "2^(bits-1)-1",
            "scale": "absmax/bound stored f16 in the BPW bill, f32 in this recon",
            "layout_default": "per-row groups along K",
        },
        "kernel_constants": {
            "name": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
            "tpr": TPR,
            "tg": TG,
            "rows_per_tg": ROWS_PER_TG,
            "thread_tile": THREAD_TILE,
            "stride": STRIDE,
            "incumbent_g": INCUMBENT_G,
            "source": "crates/hawking-core/shaders/qwen_uniform_q4.metal:181-221",
        },
        "ground_truth": {
            "complete_bpw_incumbent_measured": 4.252735126866492,
            "n_params": N_ALL,
            "f32_bytes": F32_BYTES,
            "q4_file_bytes_manifest": 14_287_109_840,
            "reconstructed_incumbent": inc,
        },
        "divisibility": {"unique_K": unique_k, "unique_MK": unique_mk, "table": div_table},
        "model_complete_bpw_recipes": recipes,
        "sanity": {},
        "tensors": [],
        "wall_s": None,
        "rss_max_gb": None,
    }
    dump(results)

    # Sanity: L0 out_proj must reproduce wave-1 g48=0.9602 / g64=0.9531.
    log("SANITY L0 out_proj Q3 g48/g64")
    X0 = load_hidden(0)
    Wqkv = load_tensor(tname(0, "linear_attn.in_proj_qkv.weight"))
    Wz = load_tensor(tname(0, "linear_attn.in_proj_z.weight"))
    Wqkvz = fuse_qkvz(Wqkv, Wz)
    del Wqkv, Wz
    X_out0 = deltanet_out_proxy(X0, Wqkvz)
    del Wqkvz
    Wo0 = load_tensor(tname(0, "linear_attn.out_proj.weight"))
    hold = np.arange(1, 256, 2)
    Y0 = X_out0[hold] @ Wo0.T
    sanity = {}
    for g in (48, 64, 96, 128):
        Wq = quant_k_per_row(Wo0, 3, g, 0)
        Yq = X_out0[hold] @ Wq.T
        sanity[f"q3_g{g}"] = {
            "output_cosine": mean_row_cosine(Y0, Yq),
            "weight_cosine": mean_row_cosine(Wo0.reshape(1, -1), Wq.reshape(1, -1)),
        }
        del Wq
    results["sanity"] = {
        "tensor": "L0.linear_attn.out_proj",
        "wave1_claimed": {"q3_g48": 0.9602, "q3_g64": 0.9531, "q3_g96": 0.9434, "q3_g128": 0.9443},
        "measured_this_lane": sanity,
    }
    log(f"SANITY {json.dumps(sanity)}")
    g64 = sanity["q3_g64"]["output_cosine"]
    if abs(g64 - 0.9531) > 0.003:
        raise RuntimeError(f"sanity failed L0 out_proj Q3 g64 cosine={g64} expected ~0.9531")
    dump(results)

    layers = sorted(set(LAYERS_DN + LAYERS_GQA))
    for layer in layers:
        deep = layer in DEEP_LAYERS
        log(f"===== layer {layer} gqa={is_gqa(layer)} deep={deep} =====")
        Xh = load_hidden(layer)

        # MLP: gate, up, then reconstructed down X.
        Wg = load_tensor(tname(layer, "mlp.gate_proj.weight"))
        results["tensors"].append(
            sweep_tensor(
                label=f"L{layer}.mlp.gate_proj",
                role="gate_proj",
                organ="mlp",
                layer=layer,
                W=Wg,
                X=Xh,
                head_dim=None,
                deep=deep,
            )
        )
        dump(results)
        Wu = load_tensor(tname(layer, "mlp.up_proj.weight"))
        results["tensors"].append(
            sweep_tensor(
                label=f"L{layer}.mlp.up_proj",
                role="up_proj",
                organ="mlp",
                layer=layer,
                W=Wu,
                X=Xh,
                head_dim=None,
                deep=deep,
            )
        )
        dump(results)
        X_down = np.ascontiguousarray(silu(Xh @ Wg.T) * (Xh @ Wu.T), dtype=np.float32)
        del Wg, Wu
        Wd = load_tensor(tname(layer, "mlp.down_proj.weight"))
        results["tensors"].append(
            sweep_tensor(
                label=f"L{layer}.mlp.down_proj",
                role="down_proj",
                organ="mlp",
                layer=layer,
                W=Wd,
                X=X_down,
                head_dim=None,
                deep=deep,
            )
        )
        del Wd, X_down
        dump(results)

        if is_gqa(layer):
            Wq = load_tensor(tname(layer, "self_attn.q_proj.weight"))
            results["tensors"].append(
                sweep_tensor(
                    label=f"L{layer}.self_attn.q_proj",
                    role="q_proj",
                    organ="gqa",
                    layer=layer,
                    W=Wq,
                    X=Xh,
                    head_dim=None,
                    deep=deep,
                )
            )
            dump(results)
            Wv = load_tensor(tname(layer, "self_attn.v_proj.weight"))
            results["tensors"].append(
                sweep_tensor(
                    label=f"L{layer}.self_attn.v_proj",
                    role="v_proj",
                    organ="gqa",
                    layer=layer,
                    W=Wv,
                    X=Xh,
                    head_dim=GQA_HEAD_DIM,
                    deep=deep,
                )
            )
            X_out = gqa_out_proxy(Xh, Wq, Wv)
            del Wq, Wv
            dump(results)
            Wo = load_tensor(tname(layer, "self_attn.o_proj.weight"))
            results["tensors"].append(
                sweep_tensor(
                    label=f"L{layer}.self_attn.o_proj",
                    role="o_proj",
                    organ="gqa",
                    layer=layer,
                    W=Wo,
                    X=X_out,
                    head_dim=GQA_HEAD_DIM,
                    deep=deep,
                )
            )
            del Wo, X_out
            dump(results)
        else:
            Wqkv = load_tensor(tname(layer, "linear_attn.in_proj_qkv.weight"))
            Wz = load_tensor(tname(layer, "linear_attn.in_proj_z.weight"))
            Wqkvz = fuse_qkvz(Wqkv, Wz)
            del Wqkv, Wz
            results["tensors"].append(
                sweep_tensor(
                    label=f"L{layer}.linear_attn.in_proj_qkvz",
                    role="in_proj_qkvz",
                    organ="delta_net",
                    layer=layer,
                    W=Wqkvz,
                    X=Xh,
                    head_dim=None,
                    deep=deep,
                )
            )
            X_out = deltanet_out_proxy(Xh, Wqkvz)
            del Wqkvz
            dump(results)
            Wo = load_tensor(tname(layer, "linear_attn.out_proj.weight"))
            results["tensors"].append(
                sweep_tensor(
                    label=f"L{layer}.linear_attn.out_proj",
                    role="out_proj",
                    organ="delta_net",
                    layer=layer,
                    W=Wo,
                    X=X_out,
                    head_dim=LINEAR_VALUE_DIM,
                    deep=deep,
                )
            )
            del Wo, X_out
            dump(results)
        del Xh
        if rss_gb() > 18.0:
            raise RuntimeError(f"RSS {rss_gb():.3f}G exceeded 18G cap")

    results["wall_s"] = time.time() - t0
    results["rss_max_gb"] = rss_gb()
    dump(results)
    log(f"DONE tensors={len(results['tensors'])} wall_s={results['wall_s']:.1f} rss={rss_gb():.3f}G")


if __name__ == "__main__":
    main()

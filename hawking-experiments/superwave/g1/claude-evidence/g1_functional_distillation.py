#!/usr/bin/env python3
"""Functional distillation vs weight-space codec fit.

CPU / numpy only. No GPU, no generate, no pack, no resident touch.
One tensor resident at a time. Peak RSS target << 15 GB.

Writes:
  /tmp/g1-functional-distillation/run.jsonl
  /tmp/g1-functional-distillation/report.json
  /tmp/g1-functional-distillation/run.log
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import resource
import struct
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

ART = Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b")
SRC = ART / "bf16"
CAP = ART / "activation-capture-v1"
G0 = ART / "uniform-q4-v1"
OUT = Path("/tmp/g1-functional-distillation")
OUT.mkdir(parents=True, exist_ok=True)
JSONL = OUT / "run.jsonl"
REPORT = OUT / "report.json"
LOG = OUT / "run.log"

HIDDEN = 5120
INTER = 17408
N_TOKENS = 256
N_PARAMS = 26_895_998_464
E_MLP = 17_112_760_320
E_ATTN = 7_237_795_840
E_TAB = 2_542_796_800
E_SMALL = 2_645_504
G0_BPW = 4.252735126866492
G0_S0 = 0.4078534106896186
F32_BYTES = 10_584_840
KEY_HEADS = 16
VALUES_PER_KEY = 3
KEY_DIM = 128
VALUE_DIM = 128
GQA_HEADS = 24
GQA_KV = 4
GQA_HEAD_DIM = 256
MULT = (0.50, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 2.00)
PROMPT_LENS = (57, 60, 68, 61, 10)

LAYERS = (0, 3, 15, 16, 31, 32, 47, 48, 58, 62, 63)
ROLES = ("gate", "up", "down", "out")
# extra late-collapse downs already in LAYERS via 58, 62

_HEADER_CACHE: dict[Path, dict] = {}
_WMAP = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
_G0_MAN = json.loads((G0 / "manifest.json").read_text())
_G0_BY_NAME = {t["name"]: t for t in _G0_MAN["tensors"]}


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] rss={rss_gb():.3f}G {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def append_jsonl(row: dict) -> None:
    with JSONL.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def already_done() -> set[str]:
    done: set[str] = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" in row:
                done.add(row["id"])
    return done


def flat_cosine(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(left @ right)
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    if den <= 1e-12:
        return 1.0 if num == 0.0 else 0.0
    return num / den


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
    ref = np.asarray(a, dtype=np.float64).reshape(-1)
    hat = np.asarray(b, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(ref))
    if nrm <= 1e-12:
        return 0.0
    return float(np.linalg.norm(ref - hat) / nrm)


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


def prompt_slices() -> list[tuple[int, int]]:
    out = []
    s = 0
    for n in PROMPT_LENS:
        out.append((s, s + n))
        s += n
    return out


def split_indices() -> dict[str, dict[str, np.ndarray]]:
    all_i = np.arange(N_TOKENS)
    s0_fit = all_i[:192]
    s0_hold = all_i[192:]
    even = all_i[0::2]
    odd = all_i[1::2]
    sl = prompt_slices()
    # first 3 prompts fit, last 2 hold
    p_fit = np.concatenate([np.arange(a, b) for a, b in sl[:3]])
    p_hold = np.concatenate([np.arange(a, b) for a, b in sl[3:]])
    return {
        "s0": {"fit": s0_fit, "hold": s0_hold},
        "evenodd": {"fit": even, "hold": odd},
        "prompt": {"fit": p_fit, "hold": p_hold},
    }


def family_bounds(bits: int, family: str) -> tuple[int, int, int]:
    """Return (qmin, qmax, bound_for_scale)."""
    if family == "hq30":
        if bits != 4:
            raise RuntimeError("hq30 family is Q4 only")
        return -8, 7, 7
    if family == "hgravu":
        bound = (1 << (bits - 1)) - 1
        return -bound, bound, bound
    if family == "binary":
        return -1, 1, 1
    raise RuntimeError(family)


def payload_bytes(n_elem: int, bits: int, g: int, family: str, extra_level_bits: int = 0) -> int:
    """Complete physical bytes of one GEMV, HQ30-family header 40 B.

    codes: ceil(bits * g / 8) per group; scale: 2 B per group.
    learned extra levels billed separately via extra_level_bits.
    """
    ng = (n_elem + g - 1) // g
    if family == "binary":
        # HGRAVB01: 1 bit + f16 scale / 128. mixed-2p0 L0 gate MEASURED 12534021
        # formula used here: header 40 + ng * (2 + ceil(g/8))
        code_b = (g + 7) // 8
        return 40 + ng * (2 + code_b) + extra_level_bits // 8
    code_b = (bits * g + 7) // 8
    return 40 + ng * (2 + code_b) + extra_level_bits // 8


def complete_bpw_if_class(bytes_per_tensor: int, n_tensors: int, class_elems: int, rest_bytes: int) -> float:
    return 8.0 * (bytes_per_tensor * n_tensors + rest_bytes) / N_PARAMS


def absmax_scales(W: np.ndarray, bits: int, g: int, family: str, snap_f16: bool = True) -> np.ndarray:
    qmin, qmax, bound = family_bounds(bits, family)
    n_out, n_in = W.shape
    n_blocks = (n_in + g - 1) // g
    W3 = np.zeros((n_out, n_blocks, g), dtype=np.float32)
    W3.reshape(n_out, -1)[:, :n_in] = W
    if family == "binary":
        s = np.mean(np.abs(W3), axis=2)
    else:
        amax = np.max(np.abs(W3), axis=2)
        s = amax / max(bound, 1)
    if snap_f16:
        s = s.astype(np.float16).astype(np.float32)
    return s


def quant_with_scales(
    W: np.ndarray, scales: np.ndarray, bits: int, g: int, family: str
) -> np.ndarray:
    qmin, qmax, bound = family_bounds(bits, family)
    n_out, n_in = W.shape
    n_blocks = (n_in + g - 1) // g
    W3 = np.zeros((n_out, n_blocks, g), dtype=np.float32)
    W3.reshape(n_out, -1)[:, :n_in] = W
    s = scales.reshape(n_out, n_blocks)
    if family == "binary":
        signs = np.where(W3 >= 0.0, 1.0, -1.0)
        recon = signs * s[:, :, None]
    else:
        den = np.where(s > 0.0, s, 1.0)
        codes = np.clip(np.rint(W3 / den[:, :, None]), qmin, qmax)
        codes = np.where((s > 0.0)[:, :, None], codes, 0.0)
        recon = codes * s[:, :, None]
    return np.ascontiguousarray(recon.reshape(n_out, -1)[:, :n_in], dtype=np.float32)


def hq30_absmax_recon(W: np.ndarray, g: int = 64) -> np.ndarray:
    s = absmax_scales(W, 4, g, "hq30", snap_f16=True)
    return quant_with_scales(W, s, 4, g, "hq30")


def decode_hq30uq4_rows(path: Path, rows: np.ndarray) -> np.ndarray:
    with path.open("rb") as fh:
        header = fh.read(40)
        if header[:8] != b"HQ30UQ4\0":
            raise RuntimeError(f"bad magic {header[:8]!r}")
        group_size = struct.unpack_from("<I", header, 12)[0]
        rank = struct.unpack_from("<H", header, 16)[0]
        nrows = struct.unpack_from("<I", header, 32)[0]
        ncols = struct.unpack_from("<I", header, 36)[0]
        n_groups = (nrows * ncols) // group_size
        gpr = ncols // group_size
        scale_off = 40
        code_b = (group_size * 4 + 7) // 8
        code_off = 40 + n_groups * 2
        out = np.empty((len(rows), ncols), dtype=np.float32)
        for i, r in enumerate(rows):
            r = int(r)
            fh.seek(scale_off + r * gpr * 2)
            scales = np.frombuffer(fh.read(gpr * 2), dtype="<f2").astype(np.float32)
            fh.seek(code_off + r * gpr * code_b)
            codes = np.frombuffer(fh.read(gpr * code_b), dtype=np.uint8)
            lo = (codes & 0x0F).astype(np.int16) - 8
            hi = (codes >> 4).astype(np.int16) - 8
            q = np.empty(ncols, dtype=np.int16)
            q[0::2] = lo
            q[1::2] = hi
            out[i] = q.astype(np.float32) * np.repeat(scales, group_size)
    return out


def score_outputs(Y_ref: np.ndarray, Y_hat: np.ndarray) -> dict:
    return {
        "flat_cosine": flat_cosine(Y_ref, Y_hat),
        "mean_row_cosine": mean_row_cosine(Y_ref, Y_hat),
        "min_row_cosine": min_row_cosine(Y_ref, Y_hat),
        "rel_l2": rel_l2(Y_ref, Y_hat),
        "err_norm": float(np.linalg.norm(np.asarray(Y_ref, dtype=np.float64) - np.asarray(Y_hat, dtype=np.float64))),
        "ref_norm": float(np.linalg.norm(np.asarray(Y_ref, dtype=np.float64))),
    }


def matvec_scores(W: np.ndarray, Wh: np.ndarray, X: np.ndarray, idx: np.ndarray) -> dict:
    Xi = np.ascontiguousarray(X[idx], dtype=np.float32)
    Y = Xi @ W.T
    Yh = Xi @ Wh.T
    rec = score_outputs(Y, Yh)
    rec["n"] = int(idx.size)
    del Xi, Y, Yh
    return rec


def search_scales(
    W: np.ndarray,
    bits: int,
    g: int,
    family: str,
    X_fit: np.ndarray | None,
    teacher: np.ndarray | None,
    multipliers: tuple[float, ...] = MULT,
    snap_f16: bool = True,
    row_chunk: int = 2048,
) -> tuple[np.ndarray, dict]:
    """Per-group scale search.

    teacher is None and X_fit is None -> weight MSE (||w - q||^2).
    teacher is set and X_fit is set -> ||X (teacher - q(W,s))||^2 = e^T G e
      with e = teacher_group - q(W_group, s).
    """
    t0 = time.time()
    qmin, qmax, bound = family_bounds(bits, family)
    n_out, n_in = W.shape
    if n_in % g != 0:
        raise RuntimeError(f"in_dim {n_in} not divisible by {g}")
    n_blocks = n_in // g
    n_groups = n_out * n_blocks
    chosen = np.empty((n_out, n_blocks), dtype=np.float32)
    picked = np.zeros(len(multipliers), dtype=np.int64)
    Wf = np.ascontiguousarray(W, dtype=np.float32)
    Tf = None if teacher is None else np.ascontiguousarray(teacher, dtype=np.float32)
    X64 = None if X_fit is None else np.ascontiguousarray(X_fit, dtype=np.float64)
    grams = None
    if X64 is not None:
        grams = np.empty((n_blocks, g, g), dtype=np.float64)
        for b in range(n_blocks):
            xb = X64[:, b * g : (b + 1) * g]
            grams[b] = xb.T @ xb
    one_i = multipliers.index(1.0) if 1.0 in multipliers else 0
    for r0 in range(0, n_out, row_chunk):
        r1 = min(n_out, r0 + row_chunk)
        Wc = Wf[r0:r1].reshape(r1 - r0, n_blocks, g)
        if family == "binary":
            s0 = np.mean(np.abs(Wc), axis=2)
        else:
            amax = np.max(np.abs(Wc), axis=2)
            s0 = amax / max(bound, 1)
        Tc = Wc if Tf is None else Tf[r0:r1].reshape(r1 - r0, n_blocks, g)
        best_c = np.full((r1 - r0, n_blocks), np.inf, dtype=np.float64)
        best_s = s0.astype(np.float32).copy()
        best_i = np.full((r1 - r0, n_blocks), one_i, dtype=np.int16)
        zero = s0 <= 0.0
        for i, m in enumerate(multipliers):
            s = (s0 * float(m)).astype(np.float32)
            if snap_f16:
                s = s.astype(np.float16).astype(np.float32)
            if family == "binary":
                signs = np.where(Wc >= 0.0, 1.0, -1.0)
                q = signs * s[:, :, None]
            else:
                den = np.where(s > 0.0, s, 1.0)
                codes = np.clip(np.rint(Wc / den[:, :, None]), qmin, qmax)
                codes = np.where((s > 0.0)[:, :, None], codes, 0.0)
                q = codes * s[:, :, None]
            e = Tc.astype(np.float64) - q.astype(np.float64)
            if grams is None:
                cost = np.sum(e * e, axis=2)
            else:
                ge = np.einsum("bkl,cbl->cbk", grams, e, optimize=True)
                cost = np.einsum("cbk,cbk->cb", ge, e, optimize=True)
            cost = np.where(zero, np.inf, cost)
            # keep absmax (m=1) on exact-zero groups
            better = cost < best_c
            best_c = np.where(better, cost, best_c)
            best_s = np.where(better, s, best_s)
            best_i = np.where(better, i, best_i)
        # restore zero groups to 0
        best_s = np.where(zero, np.float32(0.0), best_s)
        chosen[r0:r1] = best_s
        for i in range(len(multipliers)):
            picked[i] += int(np.sum(best_i == i))
        del Wc, Tc, s0, best_c, best_s, best_i
    n_not = int(n_groups - picked[one_i]) if 1.0 in multipliers else int(n_groups)
    n_down = int(sum(int(picked[i]) for i, m in enumerate(multipliers) if m < 1.0))
    meta = {
        "n_groups": int(n_groups),
        "n_blocks": int(n_blocks),
        "frac_groups_not_absmax": float(n_not) / float(max(n_groups, 1)),
        "frac_groups_smaller_than_absmax": float(n_down) / float(max(n_groups, 1)),
        "n_picked_per_multiplier": [int(x) for x in picked],
        "multipliers": [float(m) for m in multipliers],
        "wall_s": time.time() - t0,
        "snap_f16": bool(snap_f16),
        "objective": "weight_mse" if grams is None else "func_eGe",
        "teacher_is_self": bool(teacher is None or teacher is W),
    }
    return chosen, meta


def ls_scales(
    W: np.ndarray,
    bits: int,
    g: int,
    family: str,
    X_fit: np.ndarray,
    teacher: np.ndarray,
    n_iter: int = 3,
    snap_f16: bool = True,
) -> tuple[np.ndarray, dict]:
    """Iterative least-squares scale given codes from current s.

    Per group / row: s <- (u·v)/(u·u) with u = Xg @ codes, v = Xg @ teacher_row.
    Codes always come from W (the quantized source), teacher is the map we match.
    """
    t0 = time.time()
    qmin, qmax, bound = family_bounds(bits, family)
    n_out, n_in = W.shape
    n_blocks = n_in // g
    s = absmax_scales(W, bits, g, family, snap_f16=snap_f16)
    X64 = np.ascontiguousarray(X_fit, dtype=np.float64)
    Wf = np.ascontiguousarray(W, dtype=np.float32)
    Tf = np.ascontiguousarray(teacher, dtype=np.float32)
    hist = []
    for it in range(n_iter):
        moved = 0
        for b in range(n_blocks):
            lo, hi = b * g, (b + 1) * g
            w = Wf[:, lo:hi]
            t = Tf[:, lo:hi]
            Xg = X64[:, lo:hi]
            sb = s[:, b]
            if family == "binary":
                signs = np.where(w >= 0.0, 1.0, -1.0)
                codes = signs
            else:
                den = np.where(sb > 0.0, sb, 1.0)
                codes = np.clip(np.rint(w / den[:, None]), qmin, qmax)
                codes = np.where((sb > 0.0)[:, None], codes, 0.0)
            # U[tok, row] = Xg[tok, :] · codes[row, :]
            U = Xg @ codes.T.astype(np.float64)
            V = Xg @ t.T.astype(np.float64)
            num = np.sum(U * V, axis=0)
            denu = np.sum(U * U, axis=0)
            s_new = np.where(denu > 1e-20, num / denu, sb.astype(np.float64)).astype(np.float32)
            s_new = np.maximum(s_new, 0.0)
            if snap_f16:
                s_new = s_new.astype(np.float16).astype(np.float32)
            moved += int(np.sum(s_new != sb))
            s[:, b] = s_new
        hist.append(int(moved))
    meta = {
        "n_iter": int(n_iter),
        "n_groups_moved_per_iter": hist,
        "wall_s": time.time() - t0,
        "snap_f16": bool(snap_f16),
        "objective": "func_ls",
    }
    return s, meta


def kmeans_1d(samples: np.ndarray, k: int, n_iter: int = 10, weights: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if weights is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        w = np.maximum(w, 0.0)
    # init: quantiles
    qs = np.linspace(0.0, 1.0, k)
    # weighted quantile approx via sort
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    csum = np.cumsum(ws)
    tot = float(csum[-1]) if csum.size else 1.0
    centers = np.empty(k, dtype=np.float64)
    for i, q in enumerate(qs):
        centers[i] = xs[int(np.searchsorted(csum, q * tot, side="left").clip(0, xs.size - 1))]
    for _ in range(n_iter):
        d = np.abs(x[:, None] - centers[None, :])
        lab = np.argmin(d, axis=1)
        for j in range(k):
            m = lab == j
            if not np.any(m):
                continue
            ww = w[m]
            sw = float(ww.sum())
            if sw > 0:
                centers[j] = float(np.dot(x[m], ww) / sw)
        centers.sort()
    return centers.astype(np.float32)


def learned_shared_recon(
    W: np.ndarray,
    bits: int,
    g: int,
    X_fit: np.ndarray | None,
    n_sample: int = 400_000,
    n_iter: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Shared 2^b-level codebook on values / per-group absmax.

    Weight-space: unweighted k-means.
    Functional proxy: sample weights = column energy of X (broadcast).
    """
    t0 = time.time()
    k = 1 << bits
    n_out, n_in = W.shape
    n_blocks = n_in // g
    W3 = W.reshape(n_out, n_blocks, g)
    amax = np.max(np.abs(W3), axis=2)
    den = np.where(amax > 0.0, amax, 1.0)
    wn = W3 / den[:, :, None]
    rng = np.random.default_rng(0)
    flat = wn.reshape(-1)
    n = int(flat.size)
    take = min(n_sample, n)
    idx = rng.choice(n, size=take, replace=False)
    samp = flat[idx]
    if X_fit is None:
        centers = kmeans_1d(samp, k, n_iter=n_iter, weights=None)
        kind = "weight_kmeans"
    else:
        # column energy, one value per input dim, broadcast over rows
        energy = np.sum(np.square(X_fit, dtype=np.float64), axis=0)
        # each sample's input column = (linear index % (n_blocks*g)) % n_in, but
        # W is [out, in] C-order: linear = out*(n_blocks*g) + block*g + gi
        # in = block*g + gi
        in_idx = idx % (n_blocks * g)
        ww = energy[in_idx]
        centers = kmeans_1d(samp, k, n_iter=n_iter, weights=ww)
        kind = "func_colenergy_kmeans"
    # assign in row chunks — a full (out, blocks, g, k) cube is 11 GB at Q4 gate
    Wh = np.empty((n_out, n_in), dtype=np.float32)
    chunk = 256
    for r0 in range(0, n_out, chunk):
        r1 = min(n_out, r0 + chunk)
        sl = wn[r0:r1]
        d = np.abs(sl[..., None].astype(np.float32) - centers[None, None, None, :])
        lab = np.argmin(d, axis=3)
        recon = centers[lab] * amax[r0:r1, :, None]
        Wh[r0:r1] = recon.reshape(r1 - r0, n_in)
        del d, lab, recon, sl
    extra_bits = 16 * k  # fp16 levels once per tensor
    meta = {
        "kind": kind,
        "levels": [float(x) for x in centers],
        "n_sample": int(take),
        "n_iter": int(n_iter),
        "extra_level_bits": int(extra_bits),
        "wall_s": time.time() - t0,
    }
    del wn, amax, W3, samp
    return Wh, centers, meta


def exact_columns(W: np.ndarray, Wh_body: np.ndarray, cols: np.ndarray) -> np.ndarray:
    out = Wh_body.copy()
    out[:, cols] = W[:, cols]
    return out


def col_energy(X: np.ndarray) -> np.ndarray:
    return np.sum(np.square(X, dtype=np.float64), axis=0)


def load_role_weight(layer: int, role: str) -> tuple[np.ndarray, str, str]:
    if role == "gate":
        return load_tensor(tname(layer, "mlp.gate_proj.weight")), tname(layer, "mlp.gate_proj.weight"), "post_norm_hidden"
    if role == "up":
        return load_tensor(tname(layer, "mlp.up_proj.weight")), tname(layer, "mlp.up_proj.weight"), "post_norm_hidden"
    if role == "down":
        return load_tensor(tname(layer, "mlp.down_proj.weight")), tname(layer, "mlp.down_proj.weight"), "reconstructed_swiglu"
    if role == "out":
        if is_gqa(layer):
            return (
                load_tensor(tname(layer, "self_attn.o_proj.weight")),
                tname(layer, "self_attn.o_proj.weight"),
                "gqa_mixer_proxy_repeat_v_sigmoid_qgate",
            )
        return (
            load_tensor(tname(layer, "linear_attn.out_proj.weight")),
            tname(layer, "linear_attn.out_proj.weight"),
            "dn_mixer_proxy_v_silu_z",
        )
    raise RuntimeError(role)


def load_role_X(layer: int, role: str, H: np.ndarray, cache: dict) -> np.ndarray:
    key = (layer, role)
    if key in cache:
        return cache[key]
    if role in ("gate", "up"):
        X = H
    elif role == "down":
        ck = (layer, "swiglu")
        if ck not in cache:
            Wg = load_tensor(tname(layer, "mlp.gate_proj.weight"))
            Wu = load_tensor(tname(layer, "mlp.up_proj.weight"))
            cache[ck] = np.ascontiguousarray(silu(H @ Wg.T) * (H @ Wu.T), dtype=np.float32)
            del Wg, Wu
        X = cache[ck]
    elif role == "out":
        if is_gqa(layer):
            Wq = load_tensor(tname(layer, "self_attn.q_proj.weight"))
            Wv = load_tensor(tname(layer, "self_attn.v_proj.weight"))
            X = gqa_out_proxy(H, Wq, Wv)
            del Wq, Wv
        else:
            Wqkv = load_tensor(tname(layer, "linear_attn.in_proj_qkv.weight"))
            Wz = load_tensor(tname(layer, "linear_attn.in_proj_z.weight"))
            fused = fuse_qkvz(Wqkv, Wz)
            del Wqkv, Wz
            X = deltanet_out_proxy(H, fused)
            del fused
    else:
        raise RuntimeError(role)
    cache[key] = X
    return X


def g0_artifact_path(name: str) -> Path:
    rec = _G0_BY_NAME[name]
    return G0 / "tensors" / rec["artifact"]


def codec_spec(bits: int, g: int, family: str) -> dict:
    if family == "binary":
        nom = 1.0 + 16.0 / float(g)
        return {"bits": 1, "g": g, "family": family, "nominal_bpw": nom, "qmin": -1, "qmax": 1}
    qmin, qmax, bound = family_bounds(bits, family)
    nom = float(bits) + 16.0 / float(g)
    return {
        "bits": bits,
        "g": g,
        "family": family,
        "nominal_bpw": nom,
        "qmin": qmin,
        "qmax": qmax,
        "bound": bound,
    }


def score_cell(
    W: np.ndarray,
    Wh: np.ndarray,
    Wg0: np.ndarray,
    X: np.ndarray,
    splits: dict,
    H: np.ndarray | None,
    is_write: bool,
) -> dict:
    out: dict = {"weight": score_outputs(W, Wh)}
    # precompute full Y once
    Y = X @ W.T
    Yh = X @ Wh.T
    Yg = X @ Wg0.T
    for split_name, sp in splits.items():
        hold = sp["hold"]
        rec = {
            "vs_bf16": score_outputs(Y[hold], Yh[hold]),
            "vs_g0": score_outputs(Yg[hold], Yh[hold]),
            "g0_vs_bf16": score_outputs(Y[hold], Yg[hold]),
            "n_hold": int(hold.size),
        }
        if is_write and H is not None and Y.shape[1] == H.shape[1]:
            rec["residual_proxy_vs_bf16"] = score_outputs(H[hold] + Y[hold], H[hold] + Yh[hold])
            rec["residual_proxy_vs_g0"] = score_outputs(H[hold] + Yg[hold], H[hold] + Yh[hold])
            rec["write_rms_over_H_rms"] = float(
                np.sqrt(np.mean(np.square(Y[hold], dtype=np.float64)))
                / max(np.sqrt(np.mean(np.square(H[hold], dtype=np.float64))), 1e-12)
            )
        out[split_name] = rec
    del Y, Yh, Yg
    return out


def run_calibration(splits: dict) -> dict:
    log("CALIBRATION start")
    rec: dict = {}
    # 1. G0 packed decode vs requant on L0 gate first 4 rows
    name = tname(0, "mlp.gate_proj.weight")
    W = load_tensor(name)
    Wg0 = hq30_absmax_recon(W, 64)
    packed = decode_hq30uq4_rows(g0_artifact_path(name), np.arange(4))
    rec["g0_pack_vs_requant_4rows"] = {
        "max_abs": float(np.max(np.abs(packed - Wg0[:4]))),
        "flat_cosine": flat_cosine(packed, Wg0[:4]),
        "name": name,
        "artifact": _G0_BY_NAME[name]["artifact"],
    }
    log(f"cal G0 pack vs requant max_abs={rec['g0_pack_vs_requant_4rows']['max_abs']:.3e}")
    del packed

    # 2. L0 out_proj Q3 even/odd mean-row (forensics)
    H = load_hidden(0)
    cache: dict = {}
    Wout = load_tensor(tname(0, "linear_attn.out_proj.weight"))
    Xout = load_role_X(0, "out", H, cache)
    s = absmax_scales(Wout, 3, 64, "hgravu", snap_f16=False)
    Wh = quant_with_scales(Wout, s, 3, 64, "hgravu")
    hold = splits["evenodd"]["hold"]
    Y = Xout[hold] @ Wout.T
    Yh = Xout[hold] @ Wh.T
    rec["l0_out_q3_absmax_f32scale_evenodd"] = {
        "mean_row_cosine": mean_row_cosine(Y, Yh),
        "flat_cosine": flat_cosine(Y, Yh),
        "cited_forensics": 0.9531034548050097,
    }
    log(f"cal L0 out Q3 absmax mean_row={rec['l0_out_q3_absmax_f32scale_evenodd']['mean_row_cosine']:.12f}")
    del Y, Yh, Wh, s

    # 3. L0 gate Q3 last-64 flat (mlp-floor)
    Wg = W
    Xg = H
    s = absmax_scales(Wg, 3, 64, "hgravu", snap_f16=True)
    Wh = quant_with_scales(Wg, s, 3, 64, "hgravu")
    hold = splits["s0"]["hold"]
    Y = Xg[hold] @ Wg.T
    Yh = Xg[hold] @ Wh.T
    rec["l0_gate_q3_absmax_f16_s0"] = {
        "flat_cosine": flat_cosine(Y, Yh),
        "cited_mlp_floor": 0.982098354690,
    }
    log(f"cal L0 gate Q3 s0 flat={rec['l0_gate_q3_absmax_f16_s0']['flat_cosine']:.12f}")
    del Y, Yh, Wh, s, W, Wg0, Wout, Wg, H
    gc.collect()
    rec["rss_gb"] = rss_gb()
    rec["capture"] = {
        "sha256_self": json.loads((CAP / "capture-result.json").read_text())["sha256_self"],
        "receipt_sha256": sha256_file(CAP / "capture-result.json"),
        "L00_sha256": sha256_file(CAP / "hidden" / "L00.f32"),
        "n_tokens": N_TOKENS,
        "prompt_lens": list(PROMPT_LENS),
        "status": "CAPTURED_REAL_BF16_POST_NORM_HIDDEN",
    }
    rec["g0_manifest_bpw"] = _G0_MAN["complete_physical_bpw"]
    return rec


def describe_tensor(layer: int, role: str, W: np.ndarray, X: np.ndarray) -> dict:
    n_out, n_in = W.shape
    return {
        "layer": layer,
        "role": role,
        "gqa": bool(is_gqa(layer)),
        "shape": [int(n_out), int(n_in)],
        "elements": int(W.size),
        "x_shape": [int(X.shape[0]), int(X.shape[1])],
        "rows_per_dim": float(X.shape[0]) / float(n_in),
        "rows_per_dim_s0_fit": 192.0 / float(n_in),
        "rows_per_dim_prompt_fit": 185.0 / float(n_in),
    }


def fit_and_score(
    *,
    cell_id: str,
    layer: int,
    role: str,
    W: np.ndarray,
    Wg0: np.ndarray,
    X: np.ndarray,
    H: np.ndarray,
    bits: int,
    g: int,
    family: str,
    objective: str,
    splits: dict,
    source: str = "bf16",
) -> dict:
    spec = codec_spec(bits, g, family)
    Ws = Wg0 if source == "g0" else W
    fit = splits["s0"]["fit"]  # default fit split for searches; evenodd/prompt re-fit below for key cells
    X_fit = np.ascontiguousarray(X[fit], dtype=np.float32)
    t0 = time.time()
    if objective == "weight_absmax":
        scales = absmax_scales(Ws, bits, g, family, snap_f16=True)
        meta = {"kind": "weight_absmax"}
    elif objective == "weight_mse":
        scales, meta = search_scales(Ws, bits, g, family, None, None, MULT, True)
    elif objective == "func_bf16":
        scales, meta = search_scales(Ws, bits, g, family, X_fit, W, MULT, True)
    elif objective == "func_g0":
        scales, meta = search_scales(Ws, bits, g, family, X_fit, Wg0, MULT, True)
    elif objective == "func_ls_bf16":
        scales, meta = ls_scales(Ws, bits, g, family, X_fit, W, n_iter=3, snap_f16=True)
    elif objective == "func_ls_g0":
        scales, meta = ls_scales(Ws, bits, g, family, X_fit, Wg0, n_iter=3, snap_f16=True)
    else:
        raise RuntimeError(objective)
    Wh = quant_with_scales(Ws, scales, bits, g, family)
    scored = score_cell(W, Wh, Wg0, X, splits, H, is_write=(role in ("down", "out")))
    nbytes = payload_bytes(int(W.size), bits if family != "binary" else 1, g, family)
    row = {
        "id": cell_id,
        "kind": "cell",
        "layer": layer,
        "role": role,
        "source": source,
        "objective": objective,
        "codec": spec,
        "bytes": nbytes,
        "nominal_bpw": spec["nominal_bpw"],
        "meta": meta,
        "scores": scored,
        "wall_s": time.time() - t0,
        "rss_gb": rss_gb(),
        "fit_split_for_search": "s0_first192",
        "x_site": None,
    }
    del Wh, scales, X_fit
    return row


def refit_on_split(
    W: np.ndarray,
    Wg0: np.ndarray,
    X: np.ndarray,
    bits: int,
    g: int,
    family: str,
    objective: str,
    fit_idx: np.ndarray,
    source: str = "bf16",
) -> np.ndarray:
    Ws = Wg0 if source == "g0" else W
    X_fit = np.ascontiguousarray(X[fit_idx], dtype=np.float32)
    if objective == "weight_absmax":
        scales = absmax_scales(Ws, bits, g, family, snap_f16=True)
    elif objective == "weight_mse":
        scales, _ = search_scales(Ws, bits, g, family, None, None, MULT, True)
    elif objective == "func_bf16":
        scales, _ = search_scales(Ws, bits, g, family, X_fit, W, MULT, True)
    elif objective == "func_g0":
        scales, _ = search_scales(Ws, bits, g, family, X_fit, Wg0, MULT, True)
    elif objective == "func_ls_bf16":
        scales, _ = ls_scales(Ws, bits, g, family, X_fit, W, n_iter=3, snap_f16=True)
    elif objective == "func_ls_g0":
        scales, _ = ls_scales(Ws, bits, g, family, X_fit, Wg0, n_iter=3, snap_f16=True)
    else:
        raise RuntimeError(objective)
    Wh = quant_with_scales(Ws, scales, bits, g, family)
    del scales, X_fit
    return Wh


def capture_size_sweep(
    layer: int,
    role: str,
    W: np.ndarray,
    Wg0: np.ndarray,
    X: np.ndarray,
    bits: int,
    g: int,
    family: str,
) -> dict:
    hold = np.arange(192, 256)
    sizes = (16, 32, 64, 96, 128, 160, 192)
    rows = []
    Y = X[hold] @ W.T
    Yg = X[hold] @ Wg0.T
    for n in sizes:
        fit = np.arange(n)
        for obj in ("weight_absmax", "weight_mse", "func_bf16", "func_g0"):
            Wh = refit_on_split(W, Wg0, X, bits, g, family, obj, fit)
            Yh = X[hold] @ Wh.T
            rows.append(
                {
                    "n_fit": int(n),
                    "objective": obj,
                    "vs_bf16": score_outputs(Y, Yh),
                    "vs_g0": score_outputs(Yg, Yh),
                    "rows_per_g": float(n) / float(g),
                    "rows_per_dim": float(n) / float(W.shape[1]),
                }
            )
            del Wh, Yh
    return {"layer": layer, "role": role, "bits": bits, "g": g, "family": family, "rows": rows}


def exception_experiment(
    layer: int,
    role: str,
    W: np.ndarray,
    Wg0: np.ndarray,
    X: np.ndarray,
    splits: dict,
    bits: int = 3,
    g: int = 64,
) -> dict:
    X_fit = X[splits["s0"]["fit"]]
    X_hold = X[splits["s0"]["hold"]]
    s = absmax_scales(W, bits, g, "hgravu", snap_f16=True)
    body = quant_with_scales(W, s, bits, g, "hgravu")
    e_w = np.linalg.norm(W, axis=0)
    e_x = col_energy(X_fit)
    e_func = col_energy(X_fit) * np.linalg.norm(W - Wg0, axis=0)
    Y = X_hold @ W.T
    Yg = X_hold @ Wg0.T
    Yb = X_hold @ body.T
    base = {"vs_bf16": score_outputs(Y, Yb), "vs_g0": score_outputs(Yg, Yb)}
    ks = [8, 32, 42, 128]
    rows = []
    n_in = W.shape[1]
    for k in ks:
        for tag, energy in (("weight_mag", e_w), ("x_energy", e_x), ("func_x_times_dw", e_func)):
            cols = np.argpartition(energy, -k)[-k:]
            cols.sort()
            Wh = exact_columns(W, body, cols)
            Yh = X_hold @ Wh.T
            extra_bpw = 16.0 * float(k) / float(n_in)  # zero index bits, predetermined
            rows.append(
                {
                    "k": k,
                    "rule": tag,
                    "extra_bpw_zero_index": extra_bpw,
                    "complete_nominal": bits + 16.0 / g + extra_bpw,
                    "overlap_with_x_energy": int(len(set(cols.tolist()) & set(np.argpartition(e_x, -k)[-k:].tolist()))),
                    "vs_bf16": score_outputs(Y, Yh),
                    "vs_g0": score_outputs(Yg, Yh),
                }
            )
            del Wh, Yh
    # Jaccard even/odd of x_energy top-42
    e_even = col_energy(X[splits["evenodd"]["fit"]])
    e_odd = col_energy(X[splits["evenodd"]["hold"]])
    jacc = {}
    for k in ks:
        a = set(np.argpartition(e_even, -k)[-k:].tolist())
        b = set(np.argpartition(e_odd, -k)[-k:].tolist())
        jacc[str(k)] = len(a & b) / max(len(a | b), 1)
    return {
        "layer": layer,
        "role": role,
        "bits": bits,
        "base": base,
        "rows": rows,
        "jaccard_even_odd_x_energy": jacc,
        "top42_weight_cap_x": int(
            len(set(np.argpartition(e_w, -42)[-42:].tolist()) & set(np.argpartition(e_x, -42)[-42:].tolist()))
        ),
    }


def partition_experiment(
    layer: int,
    role: str,
    W: np.ndarray,
    Wg0: np.ndarray,
    X: np.ndarray,
    splits: dict,
    bits: int = 3,
    g: int = 64,
) -> dict:
    """Same g=64, two partitions: native K-order vs X-energy-sorted columns."""
    hold = splits["s0"]["hold"]
    fit = splits["s0"]["fit"]
    Y = X[hold] @ W.T
    Yg = X[hold] @ Wg0.T
    # native
    s = absmax_scales(W, bits, g, "hgravu", True)
    Wh = quant_with_scales(W, s, bits, g, "hgravu")
    native_abs = score_outputs(Y, X[hold] @ Wh.T)
    native_g0 = score_outputs(Yg, X[hold] @ Wh.T)
    # energy-sorted: permute columns, quantize, invert
    energy = col_energy(X[fit])
    perm = np.argsort(energy)[::-1]
    Wp = W[:, perm]
    Xp = X[:, perm]
    Wg0p = Wg0[:, perm]
    s = absmax_scales(Wp, bits, g, "hgravu", True)
    Whp = quant_with_scales(Wp, s, bits, g, "hgravu")
    # also functional on permuted
    s_f, _ = search_scales(Wp, bits, g, "hgravu", Xp[fit], Wp, MULT, True)
    Whp_f = quant_with_scales(Wp, s_f, bits, g, "hgravu")
    # unpermute
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    Wh_natperm = Whp[:, inv]
    Wh_funcperm = Whp_f[:, inv]
    return {
        "layer": layer,
        "role": role,
        "bits": bits,
        "g": g,
        "native_absmax_vs_bf16": native_abs,
        "native_absmax_vs_g0": native_g0,
        "energy_sorted_absmax_vs_bf16": score_outputs(Y, X[hold] @ Wh_natperm.T),
        "energy_sorted_absmax_vs_g0": score_outputs(Yg, X[hold] @ Wh_natperm.T),
        "energy_sorted_func_vs_bf16": score_outputs(Y, X[hold] @ Wh_funcperm.T),
        "energy_sorted_func_vs_g0": score_outputs(Yg, X[hold] @ Wh_funcperm.T),
        "note": "same g, same BPW; column gather required on device",
    }


def composed_mlp(
    layer: int,
    H: np.ndarray,
    splits: dict,
) -> dict:
    """Full SwiGLU write under G0 vs replacing one tensor at a time."""
    Wg = load_tensor(tname(layer, "mlp.gate_proj.weight"))
    Wu = load_tensor(tname(layer, "mlp.up_proj.weight"))
    Wd = load_tensor(tname(layer, "mlp.down_proj.weight"))
    Wg0 = hq30_absmax_recon(Wg, 64)
    Wu0 = hq30_absmax_recon(Wu, 64)
    Wd0 = hq30_absmax_recon(Wd, 64)
    hold = splits["s0"]["hold"]
    Hh = H[hold]

    def mlp(Wg_, Wu_, Wd_):
        return (silu(Hh @ Wg_.T) * (Hh @ Wu_.T)) @ Wd_.T

    Y_bf = mlp(Wg, Wu, Wd)
    Y_g0 = mlp(Wg0, Wu0, Wd0)
    out = {
        "layer": layer,
        "g0_vs_bf16": score_outputs(Y_bf, Y_g0),
        "amp_write_over_H": float(
            np.mean(np.linalg.norm(Y_bf, axis=1) / np.maximum(np.linalg.norm(Hh, axis=1), 1e-12))
        ),
        "amp_write_over_H_g0": float(
            np.mean(np.linalg.norm(Y_g0, axis=1) / np.maximum(np.linalg.norm(Hh, axis=1), 1e-12))
        ),
        "replacements": [],
    }
    for bits in (2, 3, 4):
        for role, W, W0 in (("gate", Wg, Wg0), ("up", Wu, Wu0), ("down", Wd, Wd0)):
            X_map = {
                "gate": Hh,
                "up": Hh,
                "down": silu(Hh @ Wg.T) * (Hh @ Wu.T),
            }
            # we need full-X for search; use all tokens then apply hold
            X_full = {
                "gate": H,
                "up": H,
                "down": silu(H @ Wg.T) * (H @ Wu.T),
            }[role]
            for obj in ("weight_absmax", "func_bf16", "func_g0"):
                Wh = refit_on_split(W, W0, X_full, bits, 64, "hgravu" if bits < 4 else "hq30", obj, splits["s0"]["fit"])
                if role == "gate":
                    Yh = mlp(Wh, Wu0, Wd0)
                elif role == "up":
                    Yh = mlp(Wg0, Wh, Wd0)
                else:
                    Yh = mlp(Wg0, Wu0, Wh)
                out["replacements"].append(
                    {
                        "bits": bits,
                        "role": role,
                        "objective": obj,
                        "vs_bf16": score_outputs(Y_bf, Yh),
                        "vs_g0": score_outputs(Y_g0, Yh),
                    }
                )
                del Wh, Yh
    del Wg, Wu, Wd, Wg0, Wu0, Wd0, Y_bf, Y_g0
    gc.collect()
    return out


def amplification_table() -> dict:
    rows = []
    prev = None
    for layer in range(64):
        H = load_hidden(layer)
        nrm = np.linalg.norm(H, axis=1)
        rec = {
            "layer": layer,
            "mean_H_norm": float(np.mean(nrm)),
            "mean_abs": float(np.mean(np.abs(H))),
            "rms": float(np.sqrt(np.mean(np.square(H, dtype=np.float64)))),
        }
        if prev is not None:
            rec["mean_H_norm_ratio_vs_prev"] = float(np.mean(nrm / np.maximum(prev, 1e-12)))
        # write amps for selected layers only (need weights)
        rows.append(rec)
        prev = nrm
        del H
    # detailed write amp on selected layers
    detail = []
    for layer in LAYERS:
        H = load_hidden(layer)
        cache: dict = {}
        Wd = load_tensor(tname(layer, "mlp.down_proj.weight"))
        Xd = load_role_X(layer, "down", H, cache)
        Yd = Xd @ Wd.T
        rec = {
            "layer": layer,
            "down_mean_Y_over_H": float(
                np.mean(np.linalg.norm(Yd, axis=1) / np.maximum(np.linalg.norm(H, axis=1), 1e-12))
            ),
            "down_mean_Y_over_X": float(
                np.mean(np.linalg.norm(Yd, axis=1) / np.maximum(np.linalg.norm(Xd, axis=1), 1e-12))
            ),
            "down_rms_Y_over_rms_H": float(
                np.sqrt(np.mean(np.square(Yd, dtype=np.float64)))
                / max(np.sqrt(np.mean(np.square(H, dtype=np.float64))), 1e-12)
            ),
        }
        Wout, _, _ = load_role_weight(layer, "out")
        Xo = load_role_X(layer, "out", H, cache)
        Yo = Xo @ Wout.T
        rec["out_mean_Y_over_H"] = float(
            np.mean(np.linalg.norm(Yo, axis=1) / np.maximum(np.linalg.norm(H, axis=1), 1e-12))
        )
        rec["out_rms_Y_over_rms_H"] = float(
            np.sqrt(np.mean(np.square(Yo, dtype=np.float64)))
            / max(np.sqrt(np.mean(np.square(H, dtype=np.float64))), 1e-12)
        )
        detail.append(rec)
        del H, Wd, Wout, Xd, Xo, Yd, Yo
        gc.collect()
    # chain product of H-norm ratios
    ratios = [r["mean_H_norm_ratio_vs_prev"] for r in rows if "mean_H_norm_ratio_vs_prev" in r]
    prod = 1.0
    for x in ratios:
        prod *= x
    return {
        "per_layer_H": rows,
        "write_detail": detail,
        "product_H_norm_ratios_1_to_63": prod,
        "layer63_mean_H_over_layer0": float(rows[63]["mean_H_norm"] / max(rows[0]["mean_H_norm"], 1e-12)),
    }


def loo_prompt_stability(layer: int, role: str, W: np.ndarray, Wg0: np.ndarray, X: np.ndarray) -> dict:
    sl = prompt_slices()
    bits, g, family = 3, 64, "hgravu"
    rows = []
    for leave in range(5):
        fit = np.concatenate([np.arange(a, b) for i, (a, b) in enumerate(sl) if i != leave])
        hold = np.arange(*sl[leave])
        for obj in ("weight_absmax", "func_bf16", "func_g0"):
            Wh = refit_on_split(W, Wg0, X, bits, g, family, obj, fit)
            Y = X[hold] @ W.T
            Yh = X[hold] @ Wh.T
            Yg = X[hold] @ Wg0.T
            rows.append(
                {
                    "leave_prompt": leave,
                    "n_fit": int(fit.size),
                    "n_hold": int(hold.size),
                    "objective": obj,
                    "vs_bf16": score_outputs(Y, Yh),
                    "vs_g0": score_outputs(Yg, Yh),
                }
            )
            del Wh
    return {"layer": layer, "role": role, "rows": rows}


def main() -> None:
    t_all = time.time()
    LOG.write_text("")
    log("start functional distillation")
    splits = split_indices()
    done = already_done()
    log(f"resume done={len(done)}")

    cal = None
    if "calibration" not in done:
        cal = run_calibration(splits)
        append_jsonl({"id": "calibration", "kind": "calibration", **cal})
        done.add("calibration")
    else:
        log("skip calibration")

    # Main grid
    objectives_main = ("weight_absmax", "weight_mse", "func_bf16", "func_g0")
    bit_cfgs = [
        (2, 64, "hgravu"),
        (3, 64, "hgravu"),
        (4, 64, "hq30"),
        (3, 128, "hgravu"),
        (4, 128, "hq30"),
    ]
    binary_cfg = (1, 128, "binary")

    x_cache: dict = {}
    current_layer = None
    H = None
    for layer in LAYERS:
        if current_layer != layer:
            x_cache.clear()
            gc.collect()
            H = load_hidden(layer)
            current_layer = layer
            log(f"layer {layer} hidden loaded")
        for role in ROLES:
            W, name, x_site = load_role_weight(layer, role)
            X = load_role_X(layer, role, H, x_cache)
            Wg0 = hq30_absmax_recon(W, 64)
            desc_id = f"desc/L{layer}/{role}"
            if desc_id not in done:
                desc = describe_tensor(layer, role, W, X)
                desc["id"] = desc_id
                desc["kind"] = "tensor_desc"
                desc["name"] = name
                desc["x_site"] = x_site
                append_jsonl(desc)
                done.add(desc_id)
            cfgs = list(bit_cfgs) + [binary_cfg]
            for bits, g, family in cfgs:
                objs = list(objectives_main)
                if bits == 3 and g == 64 and family == "hgravu":
                    objs = objs + ["func_ls_bf16", "func_ls_g0"]
                for obj in objs:
                    cid = f"cell/L{layer}/{role}/b{bits}/g{g}/{family}/{obj}/src=bf16"
                    if cid in done:
                        continue
                    log(f"fit {cid} shape={W.shape}")
                    row = fit_and_score(
                        cell_id=cid,
                        layer=layer,
                        role=role,
                        W=W,
                        Wg0=Wg0,
                        X=X,
                        H=H,
                        bits=bits,
                        g=g,
                        family=family,
                        objective=obj,
                        splits=splits,
                        source="bf16",
                    )
                    row["x_site"] = x_site
                    row["name"] = name
                    append_jsonl(row)
                    done.add(cid)
                    s0 = row["scores"]["s0"]
                    log(
                        f"  s0 vs_bf16={s0['vs_bf16']['flat_cosine']:.6f} "
                        f"vs_g0={s0['vs_g0']['flat_cosine']:.6f} "
                        f"row={s0['vs_bf16']['mean_row_cosine']:.6f} "
                        f"wall={row['wall_s']:.2f}"
                    )
            # source=g0 requantize ablation
            if (layer, role) in ((0, "gate"), (0, "down"), (63, "down"), (32, "out"), (62, "down")):
                for obj in ("weight_absmax", "func_g0", "func_bf16"):
                    cid = f"cell/L{layer}/{role}/b3/g64/hgravu/{obj}/src=g0"
                    if cid in done:
                        continue
                    log(f"fit {cid} (requantize G0)")
                    row = fit_and_score(
                        cell_id=cid,
                        layer=layer,
                        role=role,
                        W=W,
                        Wg0=Wg0,
                        X=X,
                        H=H,
                        bits=3,
                        g=64,
                        family="hgravu",
                        objective=obj,
                        splits=splits,
                        source="g0",
                    )
                    row["x_site"] = x_site
                    row["name"] = name
                    append_jsonl(row)
                    done.add(cid)
            # learned shared levels
            if (layer, role) in ((0, "gate"), (0, "down"), (0, "out"), (32, "out"), (63, "down"), (62, "down"), (31, "gate")):
                for bits in (2, 3, 4):
                    for kind in ("weight", "func"):
                        cid = f"levels/L{layer}/{role}/b{bits}/{kind}"
                        if cid in done:
                            continue
                        log(f"levels {cid}")
                        X_fit = X[splits["s0"]["fit"]]
                        Wh, centers, meta = learned_shared_recon(
                            W, bits, 64, None if kind == "weight" else X_fit
                        )
                        scored = score_cell(W, Wh, Wg0, X, splits, H, role in ("down", "out"))
                        extra_bits = 16 * (1 << bits)
                        nbytes = payload_bytes(int(W.size), bits, 64, "hgravu", extra_level_bits=extra_bits)
                        append_jsonl(
                            {
                                "id": cid,
                                "kind": "levels",
                                "layer": layer,
                                "role": role,
                                "bits": bits,
                                "g": 64,
                                "objective": f"learned_shared_{kind}",
                                "levels": [float(x) for x in centers],
                                "meta": meta,
                                "bytes": nbytes,
                                "nominal_bpw": bits + 16.0 / 64.0 + extra_bits / float(W.size),
                                "scores": scored,
                                "rss_gb": rss_gb(),
                            }
                        )
                        done.add(cid)
                        del Wh, centers
                        gc.collect()
            # exceptions
            if (layer, role) in ((0, "out"), (32, "out"), (63, "out"), (0, "down"), (63, "down"), (62, "down"), (0, "gate")):
                cid = f"except/L{layer}/{role}/q3"
                if cid not in done:
                    log(f"exceptions {cid}")
                    rec = exception_experiment(layer, role, W, Wg0, X, splits, bits=3, g=64)
                    rec["id"] = cid
                    rec["kind"] = "exceptions"
                    append_jsonl(rec)
                    done.add(cid)
            # partitions
            if (layer, role) in ((0, "gate"), (0, "out"), (63, "down"), (32, "out")):
                cid = f"part/L{layer}/{role}/q3"
                if cid not in done:
                    log(f"partition {cid}")
                    rec = partition_experiment(layer, role, W, Wg0, X, splits, bits=3, g=64)
                    rec["id"] = cid
                    rec["kind"] = "partition"
                    append_jsonl(rec)
                    done.add(cid)
            # capture-size sweep
            if (layer, role) in ((0, "gate"), (0, "down"), (0, "out"), (32, "out"), (63, "down"), (62, "down"), (58, "down")):
                cid = f"nsweep/L{layer}/{role}/q3"
                if cid not in done:
                    log(f"n-sweep {cid}")
                    rec = capture_size_sweep(layer, role, W, Wg0, X, bits=3, g=64, family="hgravu")
                    rec["id"] = cid
                    rec["kind"] = "n_sweep"
                    append_jsonl(rec)
                    done.add(cid)
            # prompt LOO
            if (layer, role) in ((0, "gate"), (0, "out"), (63, "down"), (62, "down")):
                cid = f"loo/L{layer}/{role}/q3"
                if cid not in done:
                    log(f"loo {cid}")
                    rec = loo_prompt_stability(layer, role, W, Wg0, X)
                    rec["id"] = cid
                    rec["kind"] = "loo"
                    append_jsonl(rec)
                    done.add(cid)
            del W, Wg0
            gc.collect()

        # composed MLP on a subset of layers
        if layer in (0, 16, 32, 48, 58, 62, 63):
            cid = f"mlp/L{layer}"
            if cid not in done:
                log(f"composed MLP L{layer}")
                rec = composed_mlp(layer, H, splits)
                rec["id"] = cid
                rec["kind"] = "composed_mlp"
                append_jsonl(rec)
                done.add(cid)

    if "amplification" not in done:
        log("amplification table")
        rec = amplification_table()
        rec["id"] = "amplification"
        rec["kind"] = "amplification"
        append_jsonl(rec)
        done.add("amplification")

    # assemble report
    cells = []
    extras = []
    cal_out = None
    amp = None
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            k = row.get("kind")
            if k == "cell":
                cells.append(row)
            elif k == "calibration":
                cal_out = row
            elif k == "amplification":
                amp = row
            else:
                extras.append(row)

    report = {
        "schema": "hawking.g1.functional_distillation.v1",
        "wall_s": time.time() - t_all,
        "rss_max_gb": rss_gb(),
        "n_cells": len(cells),
        "n_extras": len(extras),
        "calibration": cal_out,
        "amplification_present": amp is not None,
        "g0_s0_cited": G0_S0,
        "g0_bpw_cited": G0_BPW,
        "splits": {
            "s0": {"fit": "0:192", "hold": "192:256", "n_fit": 192, "n_hold": 64, "leak": "prompt4 split across fit/hold"},
            "evenodd": {"fit": "even 128", "hold": "odd 128", "n_fit": 128, "n_hold": 128, "leak": "every prompt"},
            "prompt": {"fit": "prompts0-2 = 185", "hold": "prompts3-4 = 71", "n_fit": 185, "n_hold": 71, "leak": "none"},
        },
        "jsonl": str(JSONL),
        "jsonl_sha256": sha256_file(JSONL) if JSONL.exists() else None,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    REPORT.write_text(json.dumps(report, indent=2))
    log(f"done cells={len(cells)} extras={len(extras)} wall={report['wall_s']:.1f}s rss={rss_gb():.3f}G")


if __name__ == "__main__":
    main()

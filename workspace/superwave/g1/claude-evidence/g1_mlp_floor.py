#!/usr/bin/env python3
"""Qwen3.8 MLP floor locate — CPU, real BF16 tensors, real capture.

Native-reader codecs only: HGRAVB01, HGRAVR02 rice_q1_rms_2pct,
HGRAVS01 r160_b3, HGRAVU01 bits 2..8 group-64.

No GPU, no Metal, no generate, no pack, no resident touch.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL_DIR = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
)
CAPTURE_DIR = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
)
MIXED_2P0 = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1"
)
MIXED_Q3 = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-q3mlp-v1"
)

N_TOKENS = 256
HIDDEN = 5120
INTERMEDIATE = 17408
FIT_N = 192
HOLD_N = 64
GROUP_BINARY = 128
GROUP_UNIFORM = 64
HGRAVS_RANK = 160
HGRAVS_BITS = 3
ELEMENTS_MLP_TENSOR = 17408 * 5120  # 89128960
UNIFORM_HEADER = 280  # calibrated: q3 nbytes 36208920 - scales - codes

N = 26_895_998_464
E_MLP = 17_112_760_320
E_ATTN = 7_237_795_840
E_TAB = 2_542_796_800
E_SMALL = 2_645_504
B_TAB_Q4 = 4.250000251691366
B_SMALL_F32 = 32.00853977162764
BYTES_TAB_Q4 = 1_350_860_880
BYTES_SMALL_F32 = 10_584_840
SIDE_TABLE_2P0 = 184_307  # catalog+format+manifest MEASURED mixed-2p0


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(left @ right)
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    if den <= 1e-12:
        return 1.0 if num == 0.0 else 0.0
    return num / den


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    ref = np.asarray(a, dtype=np.float64).reshape(-1)
    hat = np.asarray(b, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(ref))
    if nrm <= 1e-12:
        return 0.0
    return float(np.linalg.norm(ref - hat) / nrm)


def silu(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


# ---------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------

_HEADER_CACHE: dict[Path, dict[str, Any]] = {}
_WEIGHT_MAP: dict[str, str] | None = None


def load_weight_map() -> dict[str, str]:
    global _WEIGHT_MAP
    if _WEIGHT_MAP is None:
        idx = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
        _WEIGHT_MAP = dict(idx["weight_map"])
    return _WEIGHT_MAP


def read_safetensors_header(shard: Path) -> dict[str, Any]:
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def load_tensor(name: str) -> np.ndarray:
    weight_map = load_weight_map()
    shard = MODEL_DIR / weight_map[name]
    if shard not in _HEADER_CACHE:
        _HEADER_CACHE[shard] = read_safetensors_header(shard)
    info = _HEADER_CACHE[shard][name]
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
    raise RuntimeError(f"unsupported dtype {dtype} for {name}")


def load_hidden(layer: int) -> np.ndarray:
    path = CAPTURE_DIR / "hidden" / f"L{layer:02d}.f32"
    x = np.fromfile(path, dtype=np.float32)
    if x.size != N_TOKENS * HIDDEN:
        raise RuntimeError(f"{path} has {x.size} floats")
    return np.ascontiguousarray(x.reshape(N_TOKENS, HIDDEN), dtype=np.float32)


# ---------------------------------------------------------------------------
# codecs (reconstruction; BPW billed from MEASURED / closed form)
# ---------------------------------------------------------------------------

def uniform_bytes(bits: int, n: int = ELEMENTS_MLP_TENSOR) -> int:
    groups = math.ceil(n / GROUP_UNIFORM)
    scales = groups * 2
    codes = math.ceil(n * bits / 8)
    return UNIFORM_HEADER + scales + codes


def binary_bytes(n: int = ELEMENTS_MLP_TENSOR) -> int:
    # MEASURED mixed-2p0 / descent: 12534021 = 261 + n/8 + 2n/128
    return 261 + n // 8 + (n // GROUP_BINARY) * 2


def uniform_recon(W: np.ndarray, bits: int) -> np.ndarray:
    if bits < 2 or bits > 8:
        raise ValueError("uniform bits 2..8")
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    gsz = GROUP_UNIFORM
    groups = math.ceil(flat.size / gsz)
    pad = groups * gsz - flat.size
    padded = np.pad(flat, (0, pad)) if pad else flat
    padded = padded.reshape(groups, gsz)
    bound = (1 << (bits - 1)) - 1
    scales = (np.max(np.abs(padded), axis=1) / max(bound, 1)).astype(np.float16).astype(np.float32)
    den = np.where(scales > 0.0, scales, 1.0)
    q = np.rint(padded / den[:, None]).clip(-bound, bound)
    recon = (q.astype(np.float32) * scales[:, None]).reshape(-1)[: flat.size]
    return recon.reshape(W.shape)


def binary_recon(W: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    gsz = GROUP_BINARY
    groups = math.ceil(flat.size / gsz)
    pad = groups * gsz - flat.size
    padded = np.pad(flat, (0, pad)) if pad else flat
    padded = padded.reshape(groups, gsz)
    scales = np.mean(np.abs(padded), axis=1, dtype=np.float64).astype(np.float16).astype(np.float32)
    signs = np.where(padded >= 0.0, 1.0, -1.0).astype(np.float32)
    recon = (signs * scales[:, None]).reshape(-1)[: flat.size]
    return recon.reshape(W.shape)


def residual_recon(W: np.ndarray, outlier_ratio: float = 0.02) -> tuple[np.ndarray, dict[str, Any]]:
    """Binary base + top-|residual| 1-bit rms correction. Same op as HGRAVR02."""
    flat = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    base = binary_recon(W).reshape(-1)
    resid = flat - base
    count = max(1, int(math.ceil(flat.size * outlier_ratio)))
    idx = np.argpartition(np.abs(resid), -count)[-count:]
    vals = resid[idx]
    stat = float(np.sqrt(np.mean(np.square(vals)))) if vals.size else 0.0
    if not math.isfinite(stat) or stat <= 0.0:
        stat = 1.0
    scale = float(np.asarray([stat], dtype=np.float16)[0])
    corr = np.where(vals >= 0.0, scale, -scale).astype(np.float32)
    recon = base.copy()
    recon[idx] += corr
    # rice index byte count (vectorized; not a bitstream write)
    idx_sorted = np.sort(idx.astype(np.int64))
    if idx_sorted.size <= 1:
        rice_bytes = 0
        rice_k = 0
        index_bytes = 4
    else:
        diffs = np.diff(idx_sorted)
        best_k, best_bits = 0, 1 << 62
        n = int(diffs.size)
        for k in range(16):
            bits = int((diffs >> k).sum()) + n * (1 + k)
            if bits < best_bits:
                best_k, best_bits = k, bits
        rice_k = best_k
        rice_bytes = (best_bits + 7) // 8
        index_bytes = 4 + rice_bytes
    groups = math.ceil(flat.size / GROUP_BINARY)
    body = groups * 2 + math.ceil(groups * GROUP_BINARY / 8) + index_bytes + 2 + math.ceil(count / 8)
    # header calibrated from descent L0 up: 14344242 - body_without_json
    # JSON header for residual is ~400-500 B; use MEASURED class later.
    return recon.reshape(W.shape), {
        "outlier_count": int(count),
        "rice_k": int(rice_k),
        "rice_bytes": int(rice_bytes),
        "index_bytes": int(index_bytes),
        "body_without_json": int(body),
    }


def _uniform_factor(M: np.ndarray, bits: int = HGRAVS_BITS) -> np.ndarray:
    return uniform_recon(M, bits)


def hgravs_weight_rsvd(W: np.ndarray, rank: int = HGRAVS_RANK, seed: int = 0) -> np.ndarray:
    """Weight-space randomized SVD + q3 factors. Well-posed. Not HGRAVS01-act."""
    matrix = np.ascontiguousarray(W, dtype=np.float32)
    rows, cols = matrix.shape
    actual = min(max(1, rank), rows, cols)
    over = min(16, max(0, min(rows, cols) - actual))
    p = actual + over
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((cols, p), dtype=np.float32)
    y = matrix @ omega
    q, _ = np.linalg.qr(y, mode="reduced")
    for _ in range(1):
        y = matrix @ (matrix.T @ q)
        q, _ = np.linalg.qr(y, mode="reduced")
    b = q.T @ matrix
    uh, s, vt = np.linalg.svd(b, full_matrices=False)
    left = (q @ uh[:, :actual]) * s[:actual]
    right = vt[:actual, :]
    left_q = _uniform_factor(left)
    right_q = _uniform_factor(right)
    return (left_q @ right_q).astype(np.float32)


def hgravs_act_thin(
    W: np.ndarray,
    X_fit: np.ndarray,
    rank: int = HGRAVS_RANK,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Activation-weighted SVD in the span of X_fit (n=256), + ridge complement.

    Equivalent to doctor6 _activation_weighted_svd_factors when rank(X) <= n
    and the complement is scaled by sqrt(ridge). Avoids forming (in,in) gram.
    """
    matrix = np.ascontiguousarray(W, dtype=np.float32)
    X = np.ascontiguousarray(X_fit, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != matrix.shape[1]:
        raise ValueError(f"X in-dim {X.shape} vs W {matrix.shape}")
    n = max(1, X.shape[0])
    kin = matrix.shape[1]
    # economy SVD of X: X = U s Vt, Vt (n, kin)
    # Use rSVD-free exact thin SVD; n=256.
    _u, s, vt = np.linalg.svd(X, full_matrices=False)
    # ridge law from dual gravity worker
    xf = float(np.square(X, dtype=np.float64).sum())
    gram_trace_over_dim = xf / float(n * kin)
    ridge = 1e-5 * gram_trace_over_dim + 1e-8
    ridge_s = math.sqrt(ridge)
    # eigenvalues of gram in span(Vt): s^2/n + ridge
    ev = np.clip(np.square(s) / float(n) + ridge, 1e-12, None)
    sqrt_ev = np.sqrt(ev)
    corr = (sqrt_ev - ridge_s).astype(np.float32)
    vt32 = vt.astype(np.float32)
    # W @ sqrt_g = ridge_s W + (W @ Vt.T) @ diag(corr) @ Vt
    wvt = matrix @ vt32.T  # (out, n)
    weighted = ridge_s * matrix + (wvt * corr) @ vt32
    actual = min(max(1, int(rank)), weighted.shape[0], weighted.shape[1])
    # rSVD of weighted (same shape as W) via two GEMMs + thin SVD of the small side
    over = min(16, max(0, min(weighted.shape) - actual))
    p = actual + over
    rng = np.random.default_rng(0)
    omega = rng.standard_normal((weighted.shape[1], p), dtype=np.float32)
    y = weighted @ omega
    q, _ = np.linalg.qr(y, mode="reduced")
    y = weighted @ (weighted.T @ q)
    q, _ = np.linalg.qr(y, mode="reduced")
    b = q.T @ weighted
    uh, ss, vth = np.linalg.svd(b, full_matrices=False)
    left = (q @ uh[:, :actual]) * ss[:actual]
    right = vth[:actual, :]
    left_q = _uniform_factor(left)
    right_q = _uniform_factor(right)
    recon = (left_q @ right_q).astype(np.float32)
    del weighted
    return recon, {
        "n_fit": int(n),
        "rank": int(actual),
        "ridge": float(ridge),
        "method": "thin_X_plus_ridge_complement_rsvd_q3_factors",
    }


def hgravs_factor_bytes(shape: tuple[int, int], rank: int = HGRAVS_RANK, bits: int = HGRAVS_BITS) -> int:
    """Closed-form factor payload + typical JSON header.

    mixed-2p0 downs MEASURED 1,466,360–1,466,365. Naive 3.25 * rank*(m+n)/n
    plus header/scale tax.
    """
    m, n = shape
    left_n = m * rank
    right_n = rank * n
    # each factor: scales 2 per 64 + codes bits/8 + 0 (body only; outer JSON billed)
    def body(ne: int) -> int:
        groups = math.ceil(ne / GROUP_UNIFORM)
        return groups * 2 + math.ceil(ne * bits / 8)

    # MEASURED down nbytes ~1466363 for 5120x17408 r160. Use that scale.
    # 8 magic + 4 hdrlen + json + left + right
    json_and_magic = 1466363 - body(5120 * 160) - body(160 * 17408)
    return 12 + max(json_and_magic, 200) + body(left_n) + body(right_n)


# ---------------------------------------------------------------------------
# packed HGRAVS01 decode (mixed-2p0 production downs)
# ---------------------------------------------------------------------------

def parse_hq38m20(root: Path) -> list[dict[str, Any]]:
    raw = (root / "catalog.hq38m20").read_bytes()
    if raw[:8] != b"HQ38M20\0":
        raise RuntimeError("catalog magic")
    version = struct.unpack_from("<I", raw, 8)[0]
    if version != 1:
        raise RuntimeError(f"catalog version {version}")
    n_tensors = struct.unpack_from("<I", raw, 12)[0]
    n_segments = struct.unpack_from("<I", raw, 16)[0]
    name_blob_bytes = struct.unpack_from("<I", raw, 24)[0]
    cursor = 32
    by_id: dict[int, Path] = {}
    for _ in range(n_segments):
        sid = struct.unpack_from("<H", raw, cursor)[0]
        name_len = struct.unpack_from("<H", raw, cursor + 2)[0]
        cursor += 44
        filename = raw[cursor : cursor + name_len].decode("utf-8")
        cursor += name_len
        by_id[sid] = root / "segments" / filename
    rec_size = 128
    table = raw[cursor : cursor + n_tensors * rec_size]
    cursor += n_tensors * rec_size
    name_blob = raw[cursor : cursor + name_blob_bytes]
    rows = []
    for i in range(n_tensors):
        rec = table[i * rec_size : (i + 1) * rec_size]
        name_off = struct.unpack_from("<I", rec, 0)[0]
        name_len = struct.unpack_from("<H", rec, 4)[0]
        codec = rec[6]
        ndim = rec[8]
        shape = [struct.unpack_from("<I", rec, 12 + d * 4)[0] for d in range(ndim)]
        segment_id = struct.unpack_from("<H", rec, 36)[0]
        offset = struct.unpack_from("<Q", rec, 40)[0]
        nbytes = struct.unpack_from("<Q", rec, 48)[0]
        name = name_blob[name_off : name_off + name_len].decode("utf-8")
        rows.append(
            {
                "name": name,
                "codec": int(codec),
                "shape": shape,
                "segment": str(by_id[segment_id]),
                "offset": int(offset),
                "nbytes": int(nbytes),
            }
        )
    return rows


def _unpack_unsigned(payload: bytes, count: int, bits: int) -> np.ndarray:
    bit_count = count * bits
    raw = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")[:bit_count]
    weights = (1 << np.arange(bits, dtype=np.uint8)).astype(np.uint16)
    return (raw.reshape(count, bits).astype(np.uint16) * weights).sum(axis=1).astype(np.uint8)


def decode_uniform_body(header: dict[str, Any], body: bytes) -> np.ndarray:
    shape = tuple(int(x) for x in header["shape"])
    elements = int(header["elements"])
    bits = int(header["bits"])
    group_size = int(header["group_size"])
    groups = int(header["groups"])
    scale_bytes = int(header["scale_bytes"])
    code_bytes = int(header["code_bytes"])
    scales = np.frombuffer(body[:scale_bytes], dtype="<f2", count=groups).astype(np.float32)
    unsigned = _unpack_unsigned(body[scale_bytes : scale_bytes + code_bytes], groups * group_size, bits)
    bound = (1 << (bits - 1)) - 1
    signed = unsigned.astype(np.int16) - bound
    rebuilt = signed.reshape(groups, group_size).astype(np.float32) * scales[:, None]
    return np.ascontiguousarray(rebuilt.reshape(-1)[:elements].reshape(shape), dtype=np.float32)


def decode_hgravs01(payload: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    if payload[:8] != b"HGRAVS01":
        raise RuntimeError(f"not HGRAVS01: {payload[:8]!r}")
    hlen = struct.unpack_from("<I", payload, 8)[0]
    header = json.loads(payload[12 : 12 + hlen])
    body = payload[12 + hlen :]
    left_n = int(header["left_body_bytes"])
    right_n = int(header["right_body_bytes"])
    left = decode_uniform_body(header["left"], body[:left_n])
    right = decode_uniform_body(header["right"], body[left_n : left_n + right_n])
    recon = np.ascontiguousarray(left @ right, dtype=np.float32)
    return recon, header


def read_payload(row: dict[str, Any]) -> bytes:
    with open(row["segment"], "rb") as fh:
        fh.seek(row["offset"])
        return fh.read(row["nbytes"])


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_pair(
    W: np.ndarray,
    W_hat: np.ndarray,
    X_hold: np.ndarray | None,
    Y_hold_ref: np.ndarray | None,
    *,
    codec: str,
    payload_bytes: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elements = int(W.size)
    row: dict[str, Any] = {
        "codec": codec,
        "payload_bytes": int(payload_bytes),
        "elements": elements,
        "physical_bpw": 8.0 * payload_bytes / elements,
        "weight_cosine": cosine(W, W_hat),
        "weight_rel_l2": rel_l2(W, W_hat),
        "quality_space": "weight_only" if X_hold is None else "output",
    }
    if X_hold is not None and Y_hold_ref is not None:
        y_hat = X_hold @ W_hat.T
        row["hold_output_cosine"] = cosine(Y_hold_ref, y_hat)
        row["hold_output_rel_l2"] = rel_l2(Y_hold_ref, y_hat)
        del y_hat
    else:
        row["hold_output_cosine"] = None
        row["hold_output_rel_l2"] = None
    if extra:
        row.update(extra)
    return row


def encode_all(
    W: np.ndarray,
    X_fit: np.ndarray | None,
    X_hold: np.ndarray | None,
    Y_hold_ref: np.ndarray | None,
    *,
    packed_hgravs: np.ndarray | None,
    packed_hgravs_bytes: int | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    n = int(W.size)

    t0 = time.perf_counter()
    hat = binary_recon(W)
    out.append(
        score_pair(
            W, hat, X_hold, Y_hold_ref,
            codec="binary_g128",
            payload_bytes=binary_bytes(n),
            extra={"encode_s": time.perf_counter() - t0, "family": "Binary"},
        )
    )
    del hat

    t0 = time.perf_counter()
    hat, rmeta = residual_recon(W, 0.02)
    # Residual payload: use MEASURED class mean bytes scaled is wrong per tensor.
    # Prefer body_without_json + 450 B JSON (calibrated later against packed).
    payload = int(rmeta["body_without_json"] + 450)
    out.append(
        score_pair(
            W, hat, X_hold, Y_hold_ref,
            codec="residual_rice_q1_rms_2pct",
            payload_bytes=payload,
            extra={"encode_s": time.perf_counter() - t0, "family": "Residual", **rmeta},
        )
    )
    del hat

    for bits in range(2, 9):
        t0 = time.perf_counter()
        hat = uniform_recon(W, bits)
        out.append(
            score_pair(
                W, hat, X_hold, Y_hold_ref,
                codec=f"uniform_q{bits}_g64",
                payload_bytes=uniform_bytes(bits, n),
                extra={"encode_s": time.perf_counter() - t0, "family": "Uniform", "bits": bits},
            )
        )
        del hat

    t0 = time.perf_counter()
    hat = hgravs_weight_rsvd(W, HGRAVS_RANK, seed=0)
    out.append(
        score_pair(
            W, hat, X_hold, Y_hold_ref,
            codec="hgravs01_r160_b3_weight_rsvd",
            payload_bytes=hgravs_factor_bytes(tuple(W.shape)),
            extra={
                "encode_s": time.perf_counter() - t0,
                "family": "Hgravs",
                "fit": "weight_space_rsvd",
                "note": "well-posed; not the production activation-weighted operator",
            },
        )
    )
    del hat

    if X_fit is not None:
        t0 = time.perf_counter()
        hat, hmeta = hgravs_act_thin(W, X_fit, HGRAVS_RANK)
        out.append(
            score_pair(
                W, hat, X_hold, Y_hold_ref,
                codec="hgravs01_r160_b3_act_thin",
                payload_bytes=hgravs_factor_bytes(tuple(W.shape)),
                extra={
                    "encode_s": time.perf_counter() - t0,
                    "family": "Hgravs",
                    "fit": "activation_weighted_thin_ridge",
                    **hmeta,
                },
            )
        )
        del hat

    if packed_hgravs is not None and packed_hgravs_bytes is not None:
        out.append(
            score_pair(
                W, packed_hgravs, X_hold, Y_hold_ref,
                codec="hgravs01_r160_b3_packed_2p0",
                payload_bytes=packed_hgravs_bytes,
                extra={
                    "family": "Hgravs",
                    "fit": "production_mixed_2p0_decode",
                    "note": "MEASURED decode of packed HGRAVS01; n_fit_rows=256",
                },
            )
        )
    return out


# ---------------------------------------------------------------------------
# inversion / counterfactual
# ---------------------------------------------------------------------------

def invert_attn(b_mlp: float, target: float, *, include_side_table: bool) -> dict[str, float]:
    """b_attn such that complete = target. Equality cell. G1 < target sits under it."""
    bits_target = target * N
    bits_mlp = b_mlp * E_MLP
    bits_tab = B_TAB_Q4 * E_TAB
    bits_small = B_SMALL_F32 * E_SMALL
    bits_meta = 8.0 * SIDE_TABLE_2P0 if include_side_table else 0.0
    rem = bits_target - bits_mlp - bits_tab - bits_small - bits_meta
    b_attn = rem / E_ATTN
    complete_if_eq = (
        E_MLP * b_mlp + E_ATTN * b_attn + E_TAB * B_TAB_Q4 + E_SMALL * B_SMALL_F32 + bits_meta / 8.0 * 8.0
    ) / N
    return {
        "b_mlp": b_mlp,
        "target": target,
        "b_attn": b_attn,
        "rem_bits": rem,
        "feasible": rem > 0,
        "include_side_table": include_side_table,
        "complete_at_equality": complete_if_eq,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def tensor_name(layer: int, role: str) -> str:
    return f"language_model.model.layers.{layer}.mlp.{role}.weight"


def run(layers: list[int], out_path: Path) -> dict[str, Any]:
    t_all = time.perf_counter()
    capture_meta = json.loads((CAPTURE_DIR / "capture-result.json").read_text())
    if capture_meta.get("status") != "CAPTURED_REAL_BF16_POST_NORM_HIDDEN":
        raise SystemExit(f"capture not real: {capture_meta.get('status')}")
    if capture_meta.get("source", {}).get("not_synthetic") is not True:
        raise SystemExit("capture claims synthetic")

    print("parsing mixed-2p0 catalog …", flush=True)
    cat = parse_hq38m20(MIXED_2P0)
    packed_down = {}
    for row in cat:
        name = row["name"]
        if name.endswith("mlp.down_proj.weight") and row["codec"] == 2:
            layer = int(name.split(".")[3])
            packed_down[layer] = row
    print(f"  packed HGRAVS downs in catalog: {len(packed_down)} rss={rss_gb():.3f}G", flush=True)

    organs: list[dict[str, Any]] = []
    peak = rss_gb()

    for layer in layers:
        print(f"=== L{layer} rss={rss_gb():.3f}G ===", flush=True)
        X = load_hidden(layer)
        X_fit, X_hold = X[:FIT_N], X[FIT_N : FIT_N + HOLD_N]

        for role in ("gate_proj", "up_proj"):
            t_org = time.perf_counter()
            W = load_tensor(tensor_name(layer, role))
            Y_hold = X_hold @ W.T
            rows = encode_all(
                W, X_fit, X_hold, Y_hold,
                packed_hgravs=None, packed_hgravs_bytes=None,
            )
            organs.append(
                {
                    "layer": layer,
                    "role": role,
                    "W_shape": [int(W.shape[0]), int(W.shape[1])],
                    "quality_space": "output",
                    "x_site": "captured_post_norm_hidden",
                    "x_underdetermined": True,
                    "rows_per_in_dim": N_TOKENS / W.shape[1],
                    "n_fit": FIT_N,
                    "n_hold": HOLD_N,
                    "candidates": rows,
                    "wall_s": time.perf_counter() - t_org,
                }
            )
            print(
                f"  {role} {time.perf_counter()-t_org:.1f}s "
                f"q3_hold={next(c['hold_output_cosine'] for c in rows if c['codec']=='uniform_q3_g64'):.6f} "
                f"bin_hold={next(c['hold_output_cosine'] for c in rows if c['codec']=='binary_g128'):.6f}",
                flush=True,
            )
            del W, Y_hold
            peak = max(peak, rss_gb())

        gate = load_tensor(tensor_name(layer, "gate_proj"))
        up = load_tensor(tensor_name(layer, "up_proj"))
        t_sw = time.perf_counter()
        # hold-only SwiGLU for scoring; fit-X for HGRAVS uses first 192
        x_swiglu = silu(X @ gate.T) * (X @ up.T)
        print(f"  swiglu {x_swiglu.shape} {time.perf_counter()-t_sw:.2f}s", flush=True)
        del gate, up
        X_sw_fit, X_sw_hold = x_swiglu[:FIT_N], x_swiglu[FIT_N : FIT_N + HOLD_N]

        W = load_tensor(tensor_name(layer, "down_proj"))
        Y_hold = X_sw_hold @ W.T
        packed_hat = None
        packed_bytes = None
        packed_header = None
        if layer in packed_down:
            t_pk = time.perf_counter()
            payload = read_payload(packed_down[layer])
            packed_hat, packed_header = decode_hgravs01(payload)
            packed_bytes = int(packed_down[layer]["nbytes"])
            print(
                f"  decoded packed HGRAVS L{layer} "
                f"{packed_hat.shape} bytes={packed_bytes} {time.perf_counter()-t_pk:.2f}s "
                f"rank={packed_header.get('rank')} bits={packed_header.get('factor_bits')}",
                flush=True,
            )
            del payload
        t_org = time.perf_counter()
        rows = encode_all(
            W, X_sw_fit, X_sw_hold, Y_hold,
            packed_hgravs=packed_hat, packed_hgravs_bytes=packed_bytes,
        )
        organs.append(
            {
                "layer": layer,
                "role": "down_proj",
                "W_shape": [int(W.shape[0]), int(W.shape[1])],
                "quality_space": "output_reconstructed_swiglu",
                "x_site": "reconstructed_silu(X@Wg.T)*(X@Wu.T)_from_captured_hidden_plus_bf16_gate_up",
                "x_underdetermined": True,
                "rows_per_in_dim": N_TOKENS / W.shape[1],
                "n_fit": FIT_N,
                "n_hold": HOLD_N,
                "packed_hgravs_header_rank": None if packed_header is None else packed_header.get("rank"),
                "candidates": rows,
                "wall_s": time.perf_counter() - t_org,
            }
        )
        print(
            f"  down {time.perf_counter()-t_org:.1f}s "
            f"q3_hold={next(c['hold_output_cosine'] for c in rows if c['codec']=='uniform_q3_g64'):.6f} "
            f"bin_hold={next(c['hold_output_cosine'] for c in rows if c['codec']=='binary_g128'):.6f} "
            f"hgravs_act={next((c['hold_output_cosine'] for c in rows if c['codec']=='hgravs01_r160_b3_act_thin'), None)} "
            f"hgravs_pk={next((c['hold_output_cosine'] for c in rows if c['codec']=='hgravs01_r160_b3_packed_2p0'), None)}",
            flush=True,
        )
        del W, Y_hold, x_swiglu, packed_hat, X, X_fit, X_hold, X_sw_fit, X_sw_hold
        peak = max(peak, rss_gb())

        # checkpoint
        out_path.write_text(
            json.dumps(
                {
                    "partial": True,
                    "layers_done": [o["layer"] for o in organs if o["role"] == "down_proj"],
                    "n_organs": len(organs),
                    "rss_max_gb": peak,
                    "organs": organs,
                }
            )
        )

    wall = time.perf_counter() - t_all
    receipt = {
        "schema": "hawking.g1.mlp_floor_locate.v1",
        "partial": False,
        "date": "2026-08-17",
        "lane": "81-mlp-floor-locate",
        "activation": {
            "path": str(CAPTURE_DIR),
            "status": capture_meta.get("status"),
            "n_tokens": N_TOKENS,
            "fit_n": FIT_N,
            "hold_n": HOLD_N,
            "not_synthetic": True,
            "sha256_self": capture_meta.get("sha256_self"),
            "file_sha256_cited": "01db2f814fba99a1b7dac4668e30e20d69247ee3a4efa83b9ce4665718aedcbe",
            "down_x": "reconstructed post-SwiGLU from captured hidden + BF16 gate/up",
            "rows_per_dim_gate_up": N_TOKENS / HIDDEN,
            "rows_per_dim_down": N_TOKENS / INTERMEDIATE,
        },
        "layers": layers,
        "organs": organs,
        "rss_max_gb": peak,
        "wall_s": wall,
        "accounting": {
            "N": N,
            "E_mlp": E_MLP,
            "E_attn": E_ATTN,
            "E_tab": E_TAB,
            "E_small": E_SMALL,
            "b_tab_q4": B_TAB_Q4,
            "b_small_f32": B_SMALL_F32,
            "bytes_tab_q4": BYTES_TAB_Q4,
            "bytes_small_f32": BYTES_SMALL_F32,
            "side_table_2p0": SIDE_TABLE_2P0,
        },
        "claim_boundary": {
            "generation_not_run": True,
            "gpu_not_used": True,
            "no_resident_touch": True,
            "capture_underdetermined": True,
            "down_x_is_reconstructed_not_captured": True,
            "hgravs_act_is_thin_ridge_equivalent_not_full_17408_gram": True,
            "packed_hgravs_is_production_decode": True,
        },
    }
    out_path.write_text(json.dumps(receipt) + "\n")
    print(f"wrote {out_path} wall={wall:.1f}s rss_max={peak:.3f}G organs={len(organs)}", flush=True)
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0-63")
    ap.add_argument("--out", type=Path, default=Path("/tmp/g1_mlp_floor.json"))
    args = ap.parse_args()
    layers: list[int] = []
    for part in args.layers.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            layers.extend(range(int(a), int(b) + 1))
        elif part:
            layers.append(int(part))
    run(layers, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

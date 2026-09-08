"""Within-tensor low-rank NR: W ~= A @ B, optional residual, optional unequal quant.

The anatomy verdicts that called this family dead were wrong in both directions:
the 2048x768 expert organ sits 13.49% below a matched null (structure EXISTS)
and a rank-495 float factor stores 0.886x dense (factoring CAN pay in bytes).
rank_90 is 90% of spectral ENERGY, not of behaviour. This module answers the
other two questions honestly: whether a factored form lands at the same
complete EBPW as an incumbent affine/PQ arm, and whether capability survives.

Capability is a CONJUNCTION (perplexity AND n-gram diversity). Reconstruction
error is a cheap filter only -- this campaign has measured cosine improving
while capability worsened. A byte-count drop is not a win.

Every persistent byte is billed: factor payloads, per-group scales and
zero-points, residual VALUES and residual INDICES, decode metadata.
Source parameters are the dense organ m*n, never the factored element count
and never the scale tensors. include_scales=False is a control that lies in
the numerator on purpose; it is not an arm.

This is an NR, not an NX. Reconstruction rematerializes a dense W_hat; there
is no low-rank kernel. execution_complete is False.

    PYTHONPATH=tools/future python3.12 tools/future/lowrank_nr.py --selfcheck
    PYTHONPATH=tools/future python3.12 tools/future/lowrank_nr.py --compare
"""
from __future__ import annotations

import glob
import json
import math
import os
import struct
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dense_anatomy import factoring_report
from lake_scheme_census import FLOAT_DTYPES, read_header
from representational_anatomy import (
    LOWRANK_LIVE_DEFICIT_PCT,
    _effective_rank,
    break_even_rank,
)

# ---------------------------------------------------------------------------
# constants -- billed widths, not documentation
# ---------------------------------------------------------------------------

BF16_BYTES = 2
SCALE_BYTES = 2          # fp16 scale per group
ZERO_BYTES = 2           # fp16 zero-point per group
RESIDUAL_VALUE_BYTES = 2  # bf16 residual
RESIDUAL_INDEX_BYTES = 4  # uint32 linear index; k>0 with this at 0 is unbilled
# Decode header the factored form cannot be read without: rank, both factor
# (bits, group) pairs, residual k, flags. Dense bf16 does not need it.
METADATA_BYTES = 24
AFFINE_BITS = (2, 3, 4, 8)
AFFINE_GROUPS = (32, 64, 128)

# The organ the (wrong) "low-rank is DEAD / costs more than dense" text was about.
# That receipt is Qwen3-30B-A3B. gravity_outlier_eval's O003 is Kimi-VL-A3B --
# a different body, 2048x1408 not 2048x768. Both are loadable without Metal.
O003_ANATOMY_SNAP = (
    "/Users/scammermike/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-30B-A3B-Instruct-2507/"
    "snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
)
O003_ORGAN_KEY = "model.layers.0.mlp.experts.0.down_proj.weight"
O003_GRAVITY_SNAP = (
    "/Users/scammermike/.cache/huggingface/hub/"
    "models--moonshotai--Kimi-VL-A3B-Instruct/"
    "snapshots/398eede0903cd983a2bfa0cc634e9ac1d843f375"
)
O003_GRAVITY_KEY = "language_model.model.layers.10.mlp.experts.0.down_proj.weight"

PART_KEYS = (
    "A_payload", "A_scales", "A_zeros",
    "B_payload", "B_scales", "B_zeros",
    "residual_values", "residual_indices",
    "codebook", "pq_indices",
    "dense_payload",
    "metadata",
)


class AccountingError(RuntimeError):
    """Complete-byte arithmetic that does not reconcile is not a number."""


# ---------------------------------------------------------------------------
# bf16 round-trip -- arm A is the reference, not float32
# ---------------------------------------------------------------------------

def round_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 to bf16 and expand back. Matches dense_anatomy's shift decode."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    hi = ((u + bias) >> np.uint32(16)).astype(np.uint16)
    return (hi.astype(np.uint32) << np.uint32(16)).view(np.float32)


def packed_bytes(n_values: int, bits: int) -> int:
    """Ceil(n*bits / 8). A packed body billed as n bytes is the 4x-too-small defect."""
    if n_values < 0 or bits < 0:
        raise AccountingError(f"packed_bytes({n_values}, {bits})")
    return (int(n_values) * int(bits) + 7) // 8


def source_params_of(shape) -> int:
    """Denominator is the dense organ. Never r*(m+n), never scale tensors."""
    m, n = int(shape[0]), int(shape[1])
    return m * n


def empty_parts() -> dict[str, int]:
    return {k: 0 for k in PART_KEYS}


def finalize_parts(parts: dict[str, int], source_params: int,
                   include_scales: bool) -> tuple[dict[str, int], int, float]:
    """Bill every declared part. Unknown keys are hidden free information."""
    unknown = set(parts) - set(PART_KEYS)
    if unknown:
        raise AccountingError(f"undeclared parts (hidden free information): {sorted(unknown)}")
    missing = [k for k in PART_KEYS if k not in parts]
    if missing:
        raise AccountingError(f"parts dict missing keys: {missing}")
    out = {k: int(parts[k]) for k in PART_KEYS}
    if any(v < 0 for v in out.values()):
        raise AccountingError(f"negative part: {out}")
    if not include_scales:
        # Control: lie in the NUMERATOR. Reconstruction is unchanged.
        out["A_scales"] = 0
        out["A_zeros"] = 0
        out["B_scales"] = 0
        out["B_zeros"] = 0
    if int(source_params) <= 0:
        raise AccountingError("source_params must be the dense organ, got "
                              f"{source_params}")
    complete = int(sum(out.values()))
    ebpw = complete * 8.0 / int(source_params)
    return out, complete, ebpw


def residual_parts(k: int) -> tuple[int, int]:
    """Values AND indices. k>0 with index_bytes=0 is the unbilled-residual defect."""
    k = int(k)
    if k < 0:
        raise AccountingError(f"residual k={k}")
    if k == 0:
        return 0, 0
    return k * RESIDUAL_VALUE_BYTES, k * RESIDUAL_INDEX_BYTES


# ---------------------------------------------------------------------------
# affine quant along the last axis (incumbent, and per-factor)
# ---------------------------------------------------------------------------

def _n_groups(length: int, group: int) -> int:
    return (int(length) + int(group) - 1) // int(group)


def affine_dequant(x: np.ndarray, bits: int, group: int) -> tuple[np.ndarray, int, int]:
    """Asymmetric affine along last axis. Returns (reconstruction, n_groups, n_q).

    n_q counts padded group slots -- that is what a persistent packed body holds.
    Scale and zero are one fp16 each per group (32 bits/group), matching the
    incumbent affine accounting in gravity_outlier_eval (bits + 32/group).
    """
    bits, group = int(bits), int(group)
    if bits not in AFFINE_BITS:
        raise ValueError(f"affine bits {bits} not in {AFFINE_BITS}")
    if group not in AFFINE_GROUPS:
        raise ValueError(f"affine group {group} not in {AFFINE_GROUPS}")
    x = np.ascontiguousarray(x, dtype=np.float32)
    last = int(x.shape[-1])
    lead = int(np.prod(x.shape[:-1]))
    x2 = x.reshape(lead, last)
    n_g = _n_groups(last, group)
    pad = n_g * group - last
    if pad:
        x2 = np.pad(x2, ((0, 0), (0, pad)), mode="constant")
    xg = x2.reshape(lead, n_g, group)
    lo = xg.min(axis=-1, keepdims=True)
    hi = xg.max(axis=-1, keepdims=True)
    qmax = (1 << bits) - 1
    scale = np.maximum((hi - lo) / max(qmax, 1), 1e-12)
    q = np.rint((xg - lo) / scale).clip(0, qmax)
    rec = (q * scale + lo).reshape(lead, n_g * group)
    if pad:
        rec = rec[:, :last]
    n_groups = lead * n_g
    # Persistent body is the m*n values, not the padded group slots. Billing
    # pad would make a 64-wide organ at g=128 look 2x larger than it stores.
    n_q = lead * last
    return rec.reshape(x.shape).astype(np.float32), n_groups, n_q


def store_array(x: np.ndarray, bits: int, group: int | None) -> tuple[np.ndarray, int, int, int]:
    """(reconstruction, payload_bytes, n_groups, n_q). 32=f32, 16=bf16, else affine."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if bits == 32:
        rec = np.array(x, dtype=np.float32, copy=True)
        return rec, int(x.size) * 4, 0, int(x.size)
    if bits == 16:
        rec = round_bf16(x)
        return rec, int(x.size) * BF16_BYTES, 0, int(x.size)
    rec, n_groups, n_q = affine_dequant(x, bits, int(group))
    return rec, packed_bytes(n_q, bits), n_groups, n_q


# ---------------------------------------------------------------------------
# SVD factor / residual / reconstruction
# ---------------------------------------------------------------------------

def factorize(W: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rank-r: A is (m, r), B is (r, n), W ~= A @ B."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError(f"expected a 2D organ, got shape {W.shape}")
    m, n = int(W.shape[0]), int(W.shape[1])
    r = int(r)
    if r < 1:
        raise ValueError(f"rank must be >= 1, got {r}")
    r = min(r, m, n)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    A = (U[:, :r] * S[:r]).astype(np.float32, copy=False)
    B = Vt[:r, :].astype(np.float32, copy=False)
    return A, B


def apply_residual(base: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> np.ndarray:
    out = np.array(base, dtype=np.float32, copy=True)
    if idx.size:
        flat = out.reshape(-1)
        flat[idx] = flat[idx] + vals
    return out


def largest_error_residual(R: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    flat = np.ascontiguousarray(R, dtype=np.float32).reshape(-1)
    k = int(min(max(int(k), 0), flat.size))
    if k == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    idx = np.argpartition(np.abs(flat), -k)[-k:]
    idx = np.sort(idx.astype(np.int64, copy=False))
    vals = round_bf16(flat[idx])
    return idx, vals


def reconstruction_metrics(W: np.ndarray, H: np.ndarray) -> dict[str, float]:
    """Cheap filter. Not a capability number. Cosine can rise while the gate falls."""
    W64 = np.ascontiguousarray(W, dtype=np.float64).reshape(-1)
    H64 = np.ascontiguousarray(H, dtype=np.float64).reshape(-1)
    w2 = float(np.dot(W64, W64))
    h2 = float(np.dot(H64, H64))
    d = W64 - H64
    d2 = float(np.dot(d, d))
    dot = float(np.dot(W64, H64))
    rel_fro = math.sqrt(d2 / max(w2, 1e-30))
    denom = math.sqrt(max(w2 * h2, 1e-30))
    cosine = dot / denom
    mag = math.sqrt(h2 / max(w2, 1e-30))
    return {
        "rel_fro": float(rel_fro),
        "cosine": float(cosine),
        "magnitude_ratio": float(mag),
        "direction_similarity": float(cosine),
    }


# ---------------------------------------------------------------------------
# arm result
# ---------------------------------------------------------------------------

@dataclass
class ArmResult:
    arm: str
    spec: str
    rank: int | None
    shape: tuple[int, int]
    reconstruction: np.ndarray
    parts: dict[str, int]
    source_params: int
    complete_bytes: int
    complete_ebpw: float
    factor_precisions: dict[str, Any]
    residual_k: int
    include_scales: bool
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def stores_more_than_dense(self) -> bool:
        dense = self.source_params * BF16_BYTES
        return self.complete_bytes > dense


def _finish(arm: str, spec: str, rank: int | None, W: np.ndarray, rec: np.ndarray,
            parts: dict[str, int], prec: dict, k: int, include_scales: bool,
            notes: list[str] | None = None) -> ArmResult:
    src = source_params_of(W.shape)
    parts, complete, ebpw = finalize_parts(parts, src, include_scales)
    out = ArmResult(
        arm=arm, spec=spec, rank=rank, shape=(int(W.shape[0]), int(W.shape[1])),
        reconstruction=np.ascontiguousarray(rec, dtype=np.float32),
        parts=parts, source_params=src, complete_bytes=complete,
        complete_ebpw=ebpw, factor_precisions=prec, residual_k=int(k),
        include_scales=include_scales, notes=list(notes or []),
    )
    out.metrics = reconstruction_metrics(W, out.reconstruction)
    return out


def encode_dense_bf16(W: np.ndarray) -> ArmResult:
    """Arm A. Reference. Payload is m*n bf16. No decode metadata beyond the organ."""
    rec = round_bf16(W)
    parts = empty_parts()
    parts["dense_payload"] = int(np.asarray(W).size) * BF16_BYTES
    return _finish("A", "dense-bf16", None, W, rec, parts,
                   {"A": {"bits": 16, "group": 0}, "B": None}, 0, True)


def _factor_prec(bits_a: int, group_a: int | None, bits_b: int, group_b: int | None) -> dict:
    return {
        "A": {"bits": int(bits_a), "group": 0 if bits_a >= 16 else int(group_a or 0)},
        "B": {"bits": int(bits_b), "group": 0 if bits_b >= 16 else int(group_b or 0)},
    }


def encode_lowrank(
    W: np.ndarray,
    r: int,
    bits_a: int = 16,
    group_a: int | None = None,
    bits_b: int = 16,
    group_b: int | None = None,
    residual_k: int = 0,
    include_scales: bool = True,
    arm: str | None = None,
) -> ArmResult:
    """Arms B / C / D. Unequal factor precision is allowed and often right."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    A0, B0 = factorize(W, r)
    r_used = int(A0.shape[1])
    A, a_pay, a_ng, _ = store_array(A0, bits_a, group_a)
    B, b_pay, b_ng, _ = store_array(B0, bits_b, group_b)
    base = A @ B
    idx, vals = largest_error_residual(W - base, residual_k)
    rec = apply_residual(base, idx, vals)
    k = int(idx.size)
    parts = empty_parts()
    parts["A_payload"] = a_pay
    parts["B_payload"] = b_pay
    parts["A_scales"] = a_ng * SCALE_BYTES
    parts["A_zeros"] = a_ng * ZERO_BYTES
    parts["B_scales"] = b_ng * SCALE_BYTES
    parts["B_zeros"] = b_ng * ZERO_BYTES
    parts["residual_values"], parts["residual_indices"] = residual_parts(k)
    parts["metadata"] = METADATA_BYTES
    ga = 0 if bits_a >= 16 else int(group_a)
    gb = 0 if bits_b >= 16 else int(group_b)
    if bits_a >= 16 and bits_b >= 16 and k == 0:
        letter, tag = "B", f"lowrank-r{r_used}-f{bits_a}x{bits_b}"
    elif k == 0:
        letter, tag = "C", f"lowrank-r{r_used}-A{bits_a}g{ga}-B{bits_b}g{gb}"
    else:
        letter, tag = "D", f"lowrank-r{r_used}-A{bits_a}g{ga}-B{bits_b}g{gb}-k{k}"
    if arm:
        letter = arm
    notes = []
    if bits_a != bits_b or ga != gb:
        notes.append("unequal factor precision")
    if k:
        notes.append(f"sparse residual k={k} billed values AND indices")
    return _finish(letter, tag, r_used, W, rec, parts,
                   _factor_prec(bits_a, group_a, bits_b, group_b), k,
                   include_scales, notes)


def encode_affine(W: np.ndarray, bits: int, group: int,
                  include_scales: bool = True) -> ArmResult:
    """Arm E, affine incumbent. Same bits+32/group accounting as gravity_outlier_eval."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    rec, payload, n_g, _ = store_array(W, bits, group)
    parts = empty_parts()
    parts["dense_payload"] = payload
    parts["A_scales"] = n_g * SCALE_BYTES
    parts["A_zeros"] = n_g * ZERO_BYTES
    parts["metadata"] = METADATA_BYTES
    spec = f"affine-q{bits}-g{group}"
    return _finish("E", spec, None, W, rec, parts,
                   {"A": {"bits": int(bits), "group": int(group)}, "B": None},
                   0, include_scales)


def _index_bits_per(k: int) -> int:
    b = math.log2(int(k))
    if abs(b - round(b)) < 1e-12:
        return int(round(b))
    return int(math.ceil(b))


def encode_broken_magnitude(W: np.ndarray, scale: float = 0.01) -> ArmResult:
    """Deliberate negative control: 0.01*W. Cosine ~1, magnitude destroyed.

    gravity_gauntlet rejects this even with direction_similarity 0.999.
    Without Metal the magnitude_ratio is the evidence this arm is broken;
    the conjunction gate is what must fail it once Metal is available.
    """
    W = np.ascontiguousarray(W, dtype=np.float32)
    rec = round_bf16(W * np.float32(scale))
    parts = empty_parts()
    parts["dense_payload"] = int(W.size) * BF16_BYTES
    parts["metadata"] = METADATA_BYTES
    out = _finish("broken", f"broken-magnitude-{scale}", None, W, rec, parts,
                  {"A": {"bits": 16, "group": 0}, "B": None}, 0, True,
                  notes=["deliberate magnitude-destroyed negative control"])
    return out


def encode_pq(W: np.ndarray, sub_dim: int, codebook: int, seed: int = 0) -> ArmResult:
    """Arm E, product-quantization incumbent. Codebook is fp16, indices are packed."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    m, n = int(W.shape[0]), int(W.shape[1])
    d, K = int(sub_dim), int(codebook)
    if d < 1 or n % d:
        raise ValueError(f"sub_dim {d} does not divide n={n}")
    if K < 2:
        raise ValueError(f"codebook {K}")
    X = W.reshape(-1, d)
    n_sub = int(X.shape[0])
    rng = np.random.default_rng(seed)
    sample_n = min(200_000, n_sub)
    smp = X[rng.integers(0, n_sub, size=sample_n)]
    C = smp[rng.integers(0, sample_n, size=K)].copy()
    for _ in range(12):
        x2 = np.sum(smp * smp, axis=1, keepdims=True)
        c2 = np.sum(C * C, axis=1)
        a = np.argmin(x2 - 2.0 * (smp @ C.T) + c2[None, :], axis=1)
        for j in range(K):
            mask = a == j
            if np.any(mask):
                C[j] = smp[mask].mean(axis=0)
    C16 = round_bf16(C)
    rec_parts = []
    chunk = 65_536
    for s in range(0, n_sub, chunk):
        x = X[s:s + chunk]
        x2 = np.sum(x * x, axis=1, keepdims=True)
        c2 = np.sum(C16 * C16, axis=1)
        a = np.argmin(x2 - 2.0 * (x @ C16.T) + c2[None, :], axis=1)
        rec_parts.append(C16[a])
    rec = np.concatenate(rec_parts, axis=0).reshape(m, n).astype(np.float32)
    parts = empty_parts()
    parts["codebook"] = K * d * BF16_BYTES
    parts["pq_indices"] = packed_bytes(n_sub, _index_bits_per(K))
    parts["metadata"] = METADATA_BYTES
    spec = f"pq{d}k{K}"
    return _finish("E", spec, None, W, rec, parts,
                   {"form": "pq", "sub_dim": d, "codebook": K}, 0, True)


# ---------------------------------------------------------------------------
# matched-EBPW: land C or D on the incumbent's complete bytes
# ---------------------------------------------------------------------------

def encode_matched_residual(
    W: np.ndarray,
    incumbent: ArmResult,
    r: int,
    bits_a: int,
    group_a: int,
    bits_b: int,
    group_b: int,
) -> ArmResult:
    """Arm D: factor bytes first, leftover budget becomes billed residual entries.

    Residual cannot shrink a bill. If rank r already overshoots the incumbent,
    drop r until the factors fit, then spend the leftover on residual. A gap
    under one residual entry (6 bytes) is the matching quantum, not a miss.
    """
    per = RESIDUAL_VALUE_BYTES + RESIDUAL_INDEX_BYTES
    r_try = max(1, int(r))
    last_over: ArmResult | None = None
    while r_try >= 1:
        base = encode_lowrank(W, r_try, bits_a, group_a, bits_b, group_b, residual_k=0)
        leftover = int(incumbent.complete_bytes) - int(base.complete_bytes)
        if leftover >= 0:
            k = leftover // per
            out = encode_lowrank(W, r_try, bits_a, group_a, bits_b, group_b, residual_k=k)
            out.notes.append(
                f"matched residual k={k} at r={r_try} to incumbent {incumbent.spec} "
                f"({incumbent.complete_bytes} bytes); leftover after residual "
                f"{incumbent.complete_bytes - out.complete_bytes}B"
            )
            return out
        last_over = base
        if r_try == 1:
            break
        r_try = max(1, r_try - max(1, r_try // 8))
    assert last_over is not None
    last_over.notes.append(
        f"factor form still over incumbent at r=1 "
        f"({last_over.complete_bytes} vs {incumbent.complete_bytes})"
    )
    return last_over


def parse_incumbent_spec(spec: str) -> dict[str, Any]:
    """Affine q<bits>-g<group> or pq<sub>k<codebook>. Raises on anything else."""
    s = str(spec).strip()
    import re
    m = re.fullmatch(r"q(\d+)-g(\d+)", s, re.I)
    if m:
        bits, group = int(m.group(1)), int(m.group(2))
        if bits not in AFFINE_BITS:
            raise ValueError(f"incumbent bits {bits} not in {AFFINE_BITS}")
        if group not in AFFINE_GROUPS:
            raise ValueError(f"incumbent group {group} not in {AFFINE_GROUPS}")
        return {"form": "affine", "bits": bits, "group": group}
    m = re.fullmatch(r"pq(\d+)k(\d+)", s, re.I)
    if m:
        return {"form": "pq", "sub_dim": int(m.group(1)), "codebook": int(m.group(2))}
    raise ValueError(f"incumbent spec {spec!r} is not affine qN-gG or pqDkK")


def encode_incumbent(W: np.ndarray, spec: str) -> ArmResult:
    p = parse_incumbent_spec(spec)
    if p["form"] == "affine":
        return encode_affine(W, p["bits"], p["group"])
    return encode_pq(W, p["sub_dim"], p["codebook"])


# ---------------------------------------------------------------------------
# spectral evidence (the statistic the campaign already has)
# ---------------------------------------------------------------------------

def spectral_of(W: np.ndarray, seed: int = 5) -> dict[str, Any]:
    """Deficit against a same-shape null. Existence of structure, not a byte win."""
    s = _effective_rank(W, with_null=True, seed=seed)
    m, n = int(W.shape[0]), int(W.shape[1])
    fac = factoring_report(m, n, int(s["rank_90"]))
    s.update(fac)
    s["break_even_rank"] = break_even_rank([m, n])
    s["structure_real"] = float(s["deficit_pct"]) >= float(LOWRANK_LIVE_DEFICIT_PCT)
    return s


def structure_verdict(spectral: dict[str, Any]) -> str:
    """Bytes paying is arithmetic. Structure is a deficit. Neither is capability."""
    dfc = float(spectral["deficit_pct"])
    r90 = int(spectral["rank_90"])
    be = float(spectral["break_even_rank"])
    ratio = float(spectral.get("byte_ratio") or 0.0)
    if dfc < LOWRANK_LIVE_DEFICIT_PCT:
        return (f"low-rank structure is ABSENT ({dfc:.2f}% from a matched null): "
                f"a rank-{r90} factor at {ratio:.3f}x dense would be compressing noise, "
                f"not a win")
    side = "UNDER" if r90 < be else "OVER"
    return (f"low-rank structure is REAL ({dfc:.2f}% below a matched null); a "
            f"rank-{r90} factor stores {ratio:.3f}x dense ({side} the {be:.0f} "
            f"break-even rank). rank_90 is 90% of spectral ENERGY, not of behaviour: "
            f"this says the BYTES work, not that capability survives")


# ---------------------------------------------------------------------------
# capability -- match gravity_outlier_eval's gate, do not invent a proxy
# ---------------------------------------------------------------------------

def metal_status() -> tuple[bool, str]:
    """Can this process actually EVALUATE on the GPU?

    This set the default device to mx.cpu, allocated there, and returned
    "mlx cpu usable" -- a true sentence about the wrong device. It would answer True on
    a machine with no GPU at all, because it never touched one.

    A lane found that the expensive way. `mx.default_device()` reported Device(gpu, 0),
    which NAMES a default rather than loading a device, and the next real allocation
    raised "[metal::load_device] No Metal device available. This typically occurs in
    headless, sandboxed, or virtualized macOS sessions." The G004 capability arm is
    gated on this answer, so a check that cannot say no converts a resource refusal into
    a mid-run crash inside mlx_lm.

    The device is restored afterwards: this is a probe, not a mode change, and leaving
    the default pinned is how the original version poisoned every later allocation.
    """
    try:
        import mlx.core as mx
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    previous = None
    try:
        previous = mx.default_device()
    except Exception:
        pass
    try:
        # Allocate AND evaluate on the gpu device. Naming it is not loading it.
        x = mx.ones((8, 8), dtype=mx.float32) if hasattr(mx, "ones") else mx.zeros((8, 8))
        with mx.stream(mx.gpu):
            y = x + x
            mx.eval(y)
        return True, "metal usable: evaluated on mx.gpu"
    except Exception as e:
        return False, f"metal unavailable: {type(e).__name__}: {e}"
    finally:
        if previous is not None:
            try:
                mx.set_default_device(previous)
            except Exception:
                pass


def capability_record() -> dict[str, Any]:
    """The gate is ppl AND n-gram diversity. A proxy that ran is worse than this."""
    ok, why = metal_status()
    gate = {}
    try:
        import gravity_outlier_eval as G
        gate = {
            "R4_MAX": G.R4_MAX,
            "PPL_MAX": G.PPL_MAX,
            "BF16_PPL": G.BF16_PPL,
            "BF16_R4": G.BF16_R4,
            "conjunction": "perplexity AND n-gram diversity",
            "evaluator": "tools/future/gravity_outlier_eval.py::evaluate",
            "evaluator_specimen": "Kimi-VL-A3B (O003 gravity), not the Qwen organ",
        }
    except Exception as e:
        gate = {"import_error": f"{type(e).__name__}: {e}"}
    reason = (
        "capability not measured, because Metal is blocked "
        f"({why}). gravity_outlier_eval.evaluate loads Kimi-VL-A3B on MLX and "
        "cannot run here; this module does not invent a parallel scorer. "
        "The anatomy receipt organ is Qwen3-30B-A3B down_proj 2048x768; the "
        "evaluator's O003 organ is Kimi-VL-A3B down_proj 2048x1408. "
        "Reconstruction/cosine is NOT a substitute for the conjunction gate."
    )
    return {
        "status": "NOT_MEASURED",
        "capability_ok": None,
        "reason": reason,
        "metal_ok": ok,
        "metal_detail": why,
        "gate": gate,
    }


def evaluator_handle(arm: ArmResult, capability: dict[str, Any] | None = None) -> dict[str, Any]:
    """Receipt-shaped handle matching gravity_outlier_eval.evaluate's keys.

    A caller with Metal still cannot feed this to evaluate() without a new spec
    form in that file (DENY-WRITE). The reconstruction is the thing that caller
    would assign. This is an NR; rematerializing dense W_hat is not an NX.
    """
    cap = capability or capability_record()
    return {
        "schema": "hawking.future.lowrank_nr.v1",
        "spec": arm.spec,
        "arm": arm.arm,
        "representation_class": "WITHIN_TENSOR_LOWRANK_NR",
        "complete_ebpw": arm.complete_ebpw,
        "complete_bpw": arm.complete_ebpw,
        "stored_bytes": arm.complete_bytes,
        "accounting": {
            "complete_bytes": arm.complete_bytes,
            "parts": dict(arm.parts),
            "source_params": arm.source_params,
            "include_scales": arm.include_scales,
            "factor_precisions": arm.factor_precisions,
            "residual_k": arm.residual_k,
            "rank": arm.rank,
            "shape": list(arm.shape),
        },
        "magnitude_ratio": arm.metrics.get("magnitude_ratio"),
        "direction_similarity": arm.metrics.get("direction_similarity"),
        "rel_fro": arm.metrics.get("rel_fro"),
        "capability_ok": cap.get("capability_ok"),
        "capability_status": cap.get("status", "NOT_MEASURED"),
        "capability": cap,
        "execution_complete": False,
        "execution_note": ("no low-rank kernel; reconstruction rematerializes dense "
                           "W_hat for scoring; this is an NR, not an NX"),
        "nr_not_nx": True,
    }


# ---------------------------------------------------------------------------
# organ load / synthetic with a target rank_90
# ---------------------------------------------------------------------------

def load_bf16_tensor(path: str, key: str) -> np.ndarray:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        if key not in hdr:
            raise KeyError(f"{path}: no key {key}")
        e = hdr[key]
        start, end = e["data_offsets"]
        fh.seek(8 + n + int(start))
        raw = fh.read(int(end) - int(start))
    if e.get("dtype") not in FLOAT_DTYPES:
        raise TypeError(f"{key}: dtype {e.get('dtype')} is not a float payload")
    if e["dtype"] == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        arr = (u16.astype(np.uint32) << 16).view(np.float32)
    elif e["dtype"] == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif e["dtype"] == "F32":
        arr = np.frombuffer(raw, dtype=np.float32).copy()
    else:
        arr = np.frombuffer(raw, dtype=np.float64).astype(np.float32)
    return np.ascontiguousarray(arr.reshape(tuple(int(x) for x in e["shape"])),
                                dtype=np.float32)


def load_named_organ(snapshot: str, key: str) -> tuple[np.ndarray, str]:
    """One float payload, headers first. Raises FileNotFoundError if absent."""
    shards = sorted(glob.glob(os.path.join(snapshot, "**", "*.safetensors"),
                              recursive=True))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {snapshot}")
    for path in shards:
        hdr = read_header(path)
        if key in hdr:
            return load_bf16_tensor(path, key), f"{path}::{key}"
    raise FileNotFoundError(f"{key} not in {len(shards)} shards under {snapshot}")


def load_o003_organ(snapshot: str = O003_ANATOMY_SNAP,
                    key: str = O003_ORGAN_KEY) -> tuple[np.ndarray, str]:
    """Qwen3-30B-A3B expert down_proj -- the anatomy-receipt organ (2048x768)."""
    return load_named_organ(snapshot, key)


def load_o003_gravity_organ(snapshot: str = O003_GRAVITY_SNAP,
                            key: str = O003_GRAVITY_KEY) -> tuple[np.ndarray, str]:
    """Kimi-VL-A3B expert down_proj -- gravity_outlier_eval's O003 organ (2048x1408)."""
    return load_named_organ(snapshot, key)


def synthetic_with_rank90(m: int, n: int, r90: int, seed: int = 0) -> np.ndarray:
    """A matrix whose energy rank_90 is near `r90` -- the structure the organ has."""
    rng = np.random.default_rng(seed)
    k = min(int(m), int(n))
    r90 = int(min(max(int(r90), 1), k))
    U, _ = np.linalg.qr(rng.standard_normal((m, k), dtype=np.float64), mode="reduced")
    V, _ = np.linalg.qr(rng.standard_normal((n, k), dtype=np.float64), mode="reduced")
    # Geometric spectrum: binary-search q so cum energy hits 0.9 at r90.
    lo, hi = 1e-6, 0.999999
    for _ in range(60):
        q = 0.5 * (lo + hi)
        e = q ** np.arange(k, dtype=np.float64)
        tot = float(e.sum())
        got = int((np.cumsum(e) / max(tot, 1e-30) < 0.90).sum()) + 1
        if got > r90:
            hi = q
        else:
            lo = q
    e = (0.5 * (lo + hi)) ** np.arange(k, dtype=np.float64)
    S = np.sqrt(e)
    W = (U * S) @ V.T
    return np.ascontiguousarray(W, dtype=np.float32)


def exact_rank_matrix(m: int, n: int, r: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, r), dtype=np.float32)
    B = rng.standard_normal((r, n), dtype=np.float32)
    return A @ B


# ---------------------------------------------------------------------------
# comparison table -- the actual deliverable
# ---------------------------------------------------------------------------

def _prec_cell(prec: dict | None) -> str:
    if not prec:
        return "-"
    if prec.get("form") == "pq":
        return f"pq{prec['sub_dim']}k{prec['codebook']}"
    bits = prec.get("bits")
    group = prec.get("group")
    if bits == 32:
        return "f32"
    if bits == 16 or group in (0, None):
        return "bf16"
    return f"q{bits}g{group}"


def format_table(arms: list[ArmResult], capability: dict[str, Any],
                 spectral: dict[str, Any] | None = None,
                 organ_note: str = "") -> str:
    cap = capability.get("status", "NOT_MEASURED")
    why = capability.get("reason", "capability not measured")
    lines = []
    lines.append("=== matched-EBPW comparison ===")
    if organ_note:
        lines.append(organ_note)
    if spectral:
        lines.append(
            f"spectral: ratio={spectral.get('ratio')} null={spectral.get('null_ratio')} "
            f"deficit={spectral.get('deficit_pct')}% rank_90={spectral.get('rank_90')} "
            f"break_even={spectral.get('break_even_rank'):.2f} "
            f"structure_real={spectral.get('structure_real')}"
        )
        lines.append(structure_verdict(spectral))
    hdr = (f"{'arm':<4} {'rank':>6} {'A':<12} {'B':<12} {'k':>7} "
           f"{'bytes':>10} {'EBPW':>8} {'rel_fro':>9} {'cosine':>8}  capability")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for a in arms:
        rank = "-" if a.rank is None else str(a.rank)
        pa = _prec_cell(a.factor_precisions.get("A") if isinstance(a.factor_precisions.get("A"), dict)
                        else a.factor_precisions)
        pb = _prec_cell(a.factor_precisions.get("B") if isinstance(a.factor_precisions, dict)
                        else None)
        if a.arm == "E" and a.factor_precisions.get("form") == "pq":
            pa, pb = _prec_cell(a.factor_precisions), "-"
        rel = a.metrics.get("rel_fro", float("nan"))
        cos = a.metrics.get("cosine", float("nan"))
        lines.append(
            f"{a.arm:<4} {rank:>6} {pa:<12} {pb:<12} {a.residual_k:>7d} "
            f"{a.complete_bytes:>10d} {a.complete_ebpw:>8.4f} {rel:>9.4f} {cos:>8.4f}  {cap}"
        )
        lines.append(f"     spec={a.spec}  parts={{{_parts_brief(a.parts)}}}")
    lines.append("")
    lines.append(why)
    lines.append(
        "reconstruction is a cheap filter only; capability is the conjunction gate "
        "(perplexity AND n-gram diversity) and was NOT substituted by cosine/rel_fro."
    )
    lines.append(
        "WHAT THE APPROXIMATION LOSES (phenotype, not measured here): "
        "likelihood -- discarded tail singular vectors are not 10% of NLL; "
        "diversity -- rows live in an r-dimensional subspace, the same output-range "
        "narrowing the PQ scar measured as repetition; "
        "state -- this organ is FFN down_proj, state is not stored here; "
        "routing -- the router is a different organ, but collapsed expert outputs "
        "make downstream residual-stream directions less distinguishable."
    )
    return "\n".join(lines)


def _parts_brief(parts: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in parts.items() if v)


def compare_matched(
    W: np.ndarray,
    incumbent_spec: str = "q2-g128",
    r: int | None = None,
    organ_note: str = "",
) -> dict[str, Any]:
    """Arms A-E. C or D is landed at the incumbent's complete EBPW."""
    spectral = spectral_of(W)
    r90 = int(spectral["rank_90"])
    r_use = int(r) if r is not None else r90
    A = encode_dense_bf16(W)
    B = encode_lowrank(W, r_use, 16, None, 16, None, residual_k=0)
    E = encode_incumbent(W, incumbent_spec)
    broken = encode_broken_magnitude(W, 0.01)
    # Unequal precision is the campaign's stated prior (A/B do not share sensitivity).
    C_unequal = encode_lowrank(W, r_use, 4, 64, 2, 128, residual_k=0)
    # Matched pair: factors at the energy rank, leftover budget -> billed residual.
    D = encode_matched_residual(W, E, r_use, 2, 128, 2, 128)
    # Equal-quant factor form at break-even rank: same bits/group as E, so
    # r = mn/(m+n) matches E's bytes by arithmetic (no residual).
    r_be = int(math.ceil(break_even_rank([int(W.shape[0]), int(W.shape[1])])))
    p = parse_incumbent_spec(incumbent_spec)
    if p["form"] == "affine":
        C_matched = encode_lowrank(W, r_be, p["bits"], p["group"], p["bits"], p["group"],
                                   residual_k=0)
    else:
        C_matched = C_unequal
    cap = capability_record()
    arms = [A, B, C_unequal, C_matched, D, E, broken]
    # Primary matched pair is D vs E when D landed; else C_matched vs E.
    d_gap = abs(D.complete_bytes - E.complete_bytes)
    c_gap = abs(C_matched.complete_bytes - E.complete_bytes)
    pair = ("D", "E") if d_gap <= c_gap else ("C", "E")
    table = format_table(arms, cap, spectral, organ_note)
    return {
        "spectral": spectral,
        "arms": arms,
        "capability": cap,
        "table": table,
        "matched_pair": pair,
        "incumbent_spec": incumbent_spec,
        "handles": [evaluator_handle(a, cap) for a in arms],
        "verdict": structure_verdict(spectral),
    }


def public_arm(arm: ArmResult) -> dict[str, Any]:
    """JSON-safe arm: accounting and metrics, no reconstruction payload."""
    return {
        "arm": arm.arm,
        "spec": arm.spec,
        "rank": arm.rank,
        "shape": list(arm.shape),
        "source_params": arm.source_params,
        "complete_bytes": arm.complete_bytes,
        "complete_ebpw": arm.complete_ebpw,
        "factor_precisions": arm.factor_precisions,
        "residual_k": arm.residual_k,
        "include_scales": arm.include_scales,
        "parts": dict(arm.parts),
        "metrics": dict(arm.metrics),
        "stores_more_than_dense": arm.stores_more_than_dense,
        "notes": list(arm.notes),
    }


def kimi_expert0_index(snapshot: str = O003_GRAVITY_SNAP) -> dict[tuple[int, str], tuple[str, str, tuple[int, ...]]]:
    """(layer, proj) -> (key, path, shape) for expert 0 only."""
    import re
    pat = re.compile(
        r"language_model\.model\.layers\.(\d+)\.mlp\.experts\.0\.(\w+)\.weight$"
    )
    out: dict[tuple[int, str], tuple[str, str, tuple[int, ...]]] = {}
    shards = sorted(glob.glob(os.path.join(snapshot, "*.safetensors")))
    for path in shards:
        for k, e in read_header(path).items():
            m = pat.search(k)
            if m:
                out[(int(m.group(1)), m.group(2))] = (
                    k, path, tuple(int(x) for x in e["shape"]),
                )
    return out


def expert_depth_panel(snapshot: str = O003_GRAVITY_SNAP,
                       projs: tuple[str, ...] = ("down_proj", "gate_proj", "up_proj"),
                       ) -> dict[str, Any]:
    """Within-tensor deficit at first/mid/last layer for expert 0.

    One down_proj is not the organ. G004's layer-10 down_proj was ABSENT;
    late gate_proj on the same expert is REAL. The panel is the measurement
    of that organ-dependence.
    """
    idx = kimi_expert0_index(snapshot)
    if not idx:
        raise FileNotFoundError(f"no expert0 weights under {snapshot}")
    layers = sorted({L for L, _ in idx})
    pick = [layers[0], layers[len(layers) // 2], layers[-1]]
    rows = []
    for L in pick:
        for proj in projs:
            if (L, proj) not in idx:
                continue
            key, path, shape = idx[(L, proj)]
            W = load_bf16_tensor(path, key)
            spec = spectral_of(W)
            rows.append({
                "layer": L, "proj": proj, "key": key, "shape": list(shape),
                "deficit_pct": spec["deficit_pct"], "rank_90": spec["rank_90"],
                "break_even_rank": spec["break_even_rank"],
                "byte_ratio": spec["byte_ratio"], "pays": spec["pays"],
                "structure_real": spec["structure_real"],
                "ratio": spec["ratio"], "null_ratio": spec["null_ratio"],
            })
    return {
        "snapshot": snapshot,
        "layers_picked": pick,
        "n_layers": len(layers),
        "live_deficit_pct": LOWRANK_LIVE_DEFICIT_PCT,
        "rows": rows,
        "any_structure_real": any(r["structure_real"] for r in rows),
        "max_deficit_pct": max((r["deficit_pct"] for r in rows), default=0.0),
        "capability": capability_record(),
    }


def public_compare(cmp: dict[str, Any]) -> dict[str, Any]:
    return {
        "spectral": cmp["spectral"],
        "verdict": cmp.get("verdict"),
        "incumbent_spec": cmp["incumbent_spec"],
        "matched_pair": list(cmp["matched_pair"]),
        "capability": cmp["capability"],
        "arms": [public_arm(a) for a in cmp["arms"]],
        "table": cmp["table"],
    }


# ---------------------------------------------------------------------------
# controls (selfcheck + tests)
# ---------------------------------------------------------------------------

def float_factor_payload_bytes(m: int, n: int, r: int) -> int:
    return BF16_BYTES * int(r) * (int(m) + int(n))


def dense_payload_bytes(m: int, n: int) -> int:
    return BF16_BYTES * int(m) * int(n)


def _selfcheck() -> None:
    rng = np.random.default_rng(0)

    # Positive control: exactly rank-r reconstructs at rank r; deficit is large.
    W_pos = exact_rank_matrix(48, 32, 5, seed=1)
    rec_pos = encode_lowrank(W_pos, 5, 32, None, 32, None)
    assert rec_pos.metrics["rel_fro"] < 1e-5, rec_pos.metrics
    rec_bf = encode_lowrank(W_pos, 5, 16, None, 16, None)
    assert rec_bf.metrics["rel_fro"] < 1e-2, rec_bf.metrics
    spec_pos = spectral_of(W_pos)
    assert spec_pos["deficit_pct"] > 50.0, spec_pos
    assert spec_pos["structure_real"] is True, spec_pos
    assert "REAL" in structure_verdict(spec_pos)

    # Negative control: full-rank random, deficit near 0, factoring is not a win.
    W_neg = rng.standard_normal((64, 48), dtype=np.float32)
    spec_neg = spectral_of(W_neg)
    assert abs(spec_neg["deficit_pct"]) < 5.0, spec_neg
    assert spec_neg["structure_real"] is False, spec_neg
    v_neg = structure_verdict(spec_neg)
    assert "ABSENT" in v_neg and "win" in v_neg, v_neg
    rec_neg = encode_lowrank(W_neg, 6, 16, None, 16, None)
    assert rec_neg.metrics["rel_fro"] > 0.3, rec_neg.metrics
    assert rec_neg.source_params == 64 * 48, rec_neg.source_params

    # Break-even control: r >= mn/(m+n) stores MORE than dense.
    m, n = 2048, 768
    be = break_even_rank([m, n])
    assert abs(be - 558.54545) < 1e-3, be
    r_over = int(math.ceil(be))  # 559
    assert float_factor_payload_bytes(m, n, r_over) > dense_payload_bytes(m, n)
    assert float_factor_payload_bytes(m, n, r_over - 1) < dense_payload_bytes(m, n)
    W_be = rng.standard_normal((32, 24), dtype=np.float32)
    be_s = break_even_rank([32, 24])
    r_s = int(math.ceil(be_s))
    dense = encode_dense_bf16(W_be)
    fat = encode_lowrank(W_be, r_s, 16, None, 16, None)
    assert fat.complete_bytes > dense.complete_bytes, (fat.complete_bytes, dense.complete_bytes)
    assert fat.stores_more_than_dense is True

    # Accounting control: omit scales -> reported EBPW drops; reconstruction identical.
    W_q = rng.standard_normal((32, 64), dtype=np.float32)
    full = encode_lowrank(W_q, 8, 4, 32, 2, 32, residual_k=16, include_scales=True)
    omit = encode_lowrank(W_q, 8, 4, 32, 2, 32, residual_k=16, include_scales=False)
    assert full.parts["A_scales"] > 0 and full.parts["A_zeros"] > 0, full.parts
    assert omit.parts["A_scales"] == 0 and omit.parts["A_zeros"] == 0, omit.parts
    assert omit.complete_ebpw < full.complete_ebpw, (omit.complete_ebpw, full.complete_ebpw)
    assert omit.complete_bytes < full.complete_bytes
    np.testing.assert_allclose(omit.reconstruction, full.reconstruction, atol=0, rtol=0)
    # residual indices are billed
    assert full.residual_k == 16
    assert full.parts["residual_indices"] == 16 * RESIDUAL_INDEX_BYTES
    assert full.parts["residual_values"] == 16 * RESIDUAL_VALUE_BYTES
    # denominator is dense
    assert full.source_params == 32 * 64
    assert full.source_params != 8 * (32 + 64)
    # parts sum
    assert sum(full.parts.values()) == full.complete_bytes

    # Packed payload is bits, not one byte per quantized value.
    W_row = rng.standard_normal((1, 128), dtype=np.float32)
    aff = encode_affine(W_row, 2, 128)
    assert aff.parts["dense_payload"] == packed_bytes(128, 2), aff.parts
    assert aff.parts["A_scales"] == SCALE_BYTES
    assert aff.parts["A_zeros"] == ZERO_BYTES

    # Unequal precision is representable and changes the bill.
    u = encode_lowrank(W_q, 8, 4, 32, 2, 32)
    eq = encode_lowrank(W_q, 8, 4, 32, 4, 32)
    assert u.factor_precisions["A"]["bits"] != u.factor_precisions["B"]["bits"]
    assert u.complete_bytes != eq.complete_bytes

    # Matched-EBPW comparison prints both arms.
    W_cmp = synthetic_with_rank90(96, 64, r90=40, seed=2)
    cmp = compare_matched(W_cmp, incumbent_spec="q2-g128")
    letters = {a.arm for a in cmp["arms"]}
    assert {"A", "B", "C", "D", "E"} <= letters, letters
    print(cmp["table"])
    d = next(a for a in cmp["arms"] if a.arm == "D")
    e = next(a for a in cmp["arms"] if a.arm == "E")
    gap = abs(d.complete_bytes - e.complete_bytes)
    assert gap < 16, f"D/E not matched: D={d.complete_bytes} E={e.complete_bytes} gap={gap}"
    assert cmp["capability"]["status"] == "NOT_MEASURED"
    assert cmp["capability"]["capability_ok"] is None

    print("selfcheck OK -- positive/negative/break-even/accounting controls hold; "
          f"matched D {d.complete_bytes}B vs E {e.complete_bytes}B; "
          f"capability {cmp['capability']['status']}")


def _try_load(loader, label: str) -> tuple[np.ndarray, str] | None:
    try:
        W, where = loader()
        note = (f"organ: {where} shape={tuple(W.shape)} "
                f"source_params={source_params_of(W.shape)} ({label})")
        return W, note
    except Exception as e:
        print(f"# {label} UNAVAILABLE: {type(e).__name__}: {e}", flush=True)
        return None


def _run_compare(incumbent_spec: str = "q2-g128", as_json: bool = False) -> int:
    jobs = [
        (load_o003_gravity_organ,
         "Kimi-VL-A3B expert down_proj; gravity_outlier_eval O003 organ"),
        (load_o003_organ,
         "Qwen3-30B-A3B expert down_proj; the anatomy-receipt organ"),
    ]
    reports = []
    for loader, label in jobs:
        got = _try_load(loader, label)
        if got is None:
            continue
        W, note = got
        cmp = compare_matched(W, incumbent_spec=incumbent_spec, organ_note=note)
        reports.append(public_compare(cmp))
        if not as_json:
            print(cmp["table"])
            print(f"matched_pair: {cmp['matched_pair'][0]} vs {cmp['matched_pair'][1]}")
            print()
    if not reports:
        W = synthetic_with_rank90(2048, 768, r90=495, seed=0)
        note = "organ: SYNTHETIC 2048x768 rank_90-target 495 (neither real organ loaded)"
        cmp = compare_matched(W, incumbent_spec=incumbent_spec, organ_note=note)
        reports.append(public_compare(cmp))
        if not as_json:
            print(cmp["table"])
            print(f"matched_pair: {cmp['matched_pair'][0]} vs {cmp['matched_pair'][1]}")
    if as_json:
        print(json.dumps({"incumbent_spec": incumbent_spec, "organs": reports},
                         indent=1, default=str))
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        print()
        _run_compare(as_json="--json" in sys.argv)
    elif "--compare" in sys.argv:
        spec = "q2-g128"
        if "--incumbent" in sys.argv:
            spec = sys.argv[sys.argv.index("--incumbent") + 1]
        raise SystemExit(_run_compare(spec, as_json="--json" in sys.argv))
    elif "--panel" in sys.argv:
        panel = expert_depth_panel()
        if "--json" in sys.argv:
            print(json.dumps(panel, indent=1, default=str))
        else:
            print(f"kimi expert0 panel layers={panel['layers_picked']} "
                  f"n_layers={panel['n_layers']} max_deficit={panel['max_deficit_pct']} "
                  f"any_real={panel['any_structure_real']}")
            print(f"{'L':>4} {'proj':<12} {'def%':>8} {'r90':>5} {'be':>8} {'pays':>5} {'real':>5}")
            for r in panel["rows"]:
                print(f"{r['layer']:4d} {r['proj']:<12} {r['deficit_pct']:8.3f} "
                      f"{r['rank_90']:5d} {r['break_even_rank']:8.1f} "
                      f"{str(r['pays']):>5} {str(r['structure_real']):>5}")
            cap = panel["capability"]
            print(cap.get("status"), cap.get("reason", "")[:200])
        raise SystemExit(0)
    else:
        sys.stderr.write(
            "usage: lowrank_nr.py --selfcheck | --compare [--incumbent q2-g128] "
            "[--json] | --panel [--json]\n"
        )
        raise SystemExit(2)

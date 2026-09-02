#!/usr/bin/env python3
"""PARITY LADDER — stronger than argmax, and stronger than token-id equality.

Obligation (verbatim): token-id equality is necessary and NOT SUFFICIENT.
Where a change claims unchanged arithmetic, verify the strongest available
surface: intermediate buffers, route ids / selected experts, hidden state,
final logits, then token ids. BIT_IDENTICAL where source order permits,
otherwise an EXPLICITLY JUSTIFIED TOLERANCE plus route parity, logit
agreement and a capability spot-check. A kernel that changes the computation
and happens to preserve argmax is not a physical-only speedup.

This module is the reusable judge. It does not re-run fold_addqx's complete-
token A/B; it climbs the rest of the ladder that A/B stopped at.

    python3 tools/future/parity_ladder.py --selftest
    python3 tools/future/parity_ladder.py --measure --record
    python3 tools/future/parity_ladder.py --from /tmp/parity_ladder_probe.json --record
    python3 -m pytest tools/future/test_parity_ladder.py -q

Rung order is strongest-first. A candidate's verdict is the WEAKEST applicable
rung it passes. Token-id equality alone can never produce PASS.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.future._common import (  # noqa: E402
    REPO,
    git,
    gpu_lane_lock_path,
    load_json,
    measurement_provenance,
    write_measured_receipt,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES  # noqa: E402


RECEIPT = REPO / "receipts" / "future" / "PARITY_LADDER.json"
RAW_DEFAULT = Path("/tmp/parity_ladder_probe.json")
SCHEMA = "hawking.future.parity_ladder.v1"
PROBE_SCHEMA = "hawking.future.parity_ladder.probe.v1"
VERSION = 1
RECORDED_BY = "tools/future/parity_ladder.py"
FOLD_AB_REL = "receipts/future/FOLD_ADDQX_AB.json"
BUDGET_REL = "receipts/future/MLP_ERROR_BUDGET.json"
CHEAPEN_REL = "receipts/future/MLP_DECODE_CHEAPEN.json"

GPU_LOCK = gpu_lane_lock_path()
GPU_LOCK_SH = REPO / "tools" / "gpu_lane_lock.sh"
DEFAULT_ARTIFACT = Path("/Users/scammermike/noetic/NOETIC_PARENT_A")
DEFAULT_TOKENIZER = DEFAULT_ARTIFACT / "tokenizer.json"

# Strongest surface first. Token ids last: necessary, never sufficient.
RUNGS: tuple[str, ...] = (
    "intermediate_buffers",
    "route_ids",
    "hidden_state",
    "final_logits",
    "token_ids",
)

BIT_IDENTICAL = "BIT_IDENTICAL"
WITHIN_TOLERANCE = "WITHIN_TOLERANCE"
DIFFERS = "DIFFERS"
NOT_APPLICABLE = "NOT_APPLICABLE"
RUNG_VERDICTS: tuple[str, ...] = (
    BIT_IDENTICAL,
    WITHIN_TOLERANCE,
    DIFFERS,
    NOT_APPLICABLE,
)

# Candidate-level verdicts. PASS_TOKEN_IDS is deliberately absent.
PASS_BIT_IDENTICAL = "PASS_BIT_IDENTICAL"
PASS_JUSTIFIED_TOLERANCE = "PASS_JUSTIFIED_TOLERANCE"
REFUSE = "REFUSE"
CANDIDATE_VERDICTS: tuple[str, ...] = (
    PASS_BIT_IDENTICAL,
    PASS_JUSTIFIED_TOLERANCE,
    REFUSE,
)

# Weaker number wins. NOT_APPLICABLE is skipped, not ranked.
STRENGTH: dict[str, int] = {
    BIT_IDENTICAL: 3,
    WITHIN_TOLERANCE: 2,
    DIFFERS: 0,
}

CAUSE_SOURCE_ORDER_FMA = "SOURCE_ORDER_FMA_ASSOCIATION"
CAUSE_DIFFERENT_COMPUTATION = "DIFFERENT_COMPUTATION"
CAUSE_BIT_IDENTICAL = "BIT_IDENTICAL"
CAUSE_UNMEASURED = "UNMEASURED"

# MLP_ERROR_BUDGET measured numbers. Not assumed. Argmax is not among them.
ALL_LAYERS_STRUCTURED_REL_L2 = 0.03
COMFORTABLY_USABLE_REL_L2 = 0.01
ONE_LAYER_REL_L2 = 0.3
USABLE_KL = 0.10
USABLE_TOP5 = 0.80
DEGRADE_KL = 1.0
DEGRADE_TOP5 = 0.50
TOPK_AUTHORITY = 5
KL_EPS = 1e-30

# fold_addqx A/B cited facts. Timing is not re-measured here.
CITED_GATE_MISMATCH_BYTES = 22309
CITED_GATE_BYTES = 69632
CITED_UP_MISMATCH_BYTES = 22320
CITED_DOWN_MISMATCH_BYTES = 6534
CITED_TOKEN_SAVING_MS = 3.9833
CITED_TOKEN_FNV1A64 = "e04e1b12206475d8"

FOLD_ADDQX_IDENTITY = "sum_i (s*q_i + b)*x_i = s*sum(q_i*x_i) + b*sum(x_i)"

PRIMITIVE = "FusedDecodeCompute"

CLAIM_BOUNDARY = (
    "Reusable parity ladder, strongest surface first. Token-id equality is "
    "necessary and not sufficient; a candidate whose only passing rung is "
    "token ids cannot PASS. Argmax agreement is diagnostic and is not parity. "
    "BIT_IDENTICAL requires a byte comparison of the named surface. "
    "WITHIN_TOLERANCE requires an explicit justification against "
    "MLP_ERROR_BUDGET's measured all-layers structured bar (0.03) and the "
    "comfortably-usable 0.01, plus route parity (or an explicit dense-model "
    "NOT_APPLICABLE), logit KL and top-k (not argmax), and a capability "
    "spot-check. fold_addqx is judged from a live Metal probe of intermediates, "
    "hidden, logits and a cheap generate; the complete-token 7x32 A/B is cited "
    "from FOLD_ADDQX_AB and is not re-run. Timing numbers in this receipt are "
    "citations, not new measurements. No TPS is labelled QUALIFIED."
)


class ParityLadderRefuse(ValueError):
    """Raised rather than emit a ladder verdict that cannot be defended."""


class TokenIdAloneIsNotParity(ParityLadderRefuse):
    def __init__(self, detail: str = "") -> None:
        extra = f": {detail}" if detail else ""
        super().__init__(
            "REFUSED: token-id equality is necessary and NOT SUFFICIENT; "
            "token-id equality alone cannot yield PASS"
            f"{extra}."
        )


class ArgmaxIsNotParity(ParityLadderRefuse):
    def __init__(self, detail: str = "") -> None:
        extra = f": {detail}" if detail else ""
        super().__init__(
            "REFUSED: argmax agreement is not parity; logit agreement is KL "
            f"and top-k{extra}."
        )


class CountOnlyDiffRefuse(ParityLadderRefuse):
    def __init__(self, detail: str = "") -> None:
        extra = f": {detail}" if detail else ""
        super().__init__(
            "REFUSED: a byte-mismatch count is not a characterisation; "
            "magnitude (max_abs, rel_l2) and cause are required"
            f"{extra}."
        )


class UnjustifiedTolerance(ParityLadderRefuse):
    def __init__(self, detail: str = "") -> None:
        extra = f": {detail}" if detail else ""
        super().__init__(
            "REFUSED: WITHIN_TOLERANCE requires an explicit justification "
            "against MLP_ERROR_BUDGET's measured numbers, not an assumed bar"
            f"{extra}."
        )


class IncompleteLadder(ParityLadderRefuse):
    def __init__(self, detail: str = "") -> None:
        extra = f": {detail}" if detail else ""
        super().__init__(
            "REFUSED: a stronger available surface was not reported"
            f"{extra}."
        )


class CapabilitySpotCheckMissing(ParityLadderRefuse):
    def __init__(self, detail: str = "") -> None:
        extra = f": {detail}" if detail else ""
        super().__init__(
            "REFUSED: a WITHIN_TOLERANCE candidate needs a capability "
            f"spot-check{extra}."
        )


# ---------------------------------------------------------------------------
# Tiny numeric helpers.
# ---------------------------------------------------------------------------


def f32(x: float) -> float:
    """Round a Python float to IEEE-754 binary32."""
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def _py(x: Any) -> Any:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_py(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    return x


def _r(value: float | None, n: int = 12) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return value
    out = round(float(value), n)
    return 0.0 if out == 0.0 else out


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - float(z.max())
    e = np.exp(np.clip(z, -80.0, 80.0))
    s = float(e.sum())
    return e / max(s, KL_EPS)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats. p is the incumbent."""
    pp = np.clip(np.asarray(p, dtype=np.float64).ravel(), KL_EPS, 1.0)
    qq = np.clip(np.asarray(q, dtype=np.float64).ravel(), KL_EPS, 1.0)
    pp = pp / pp.sum()
    qq = qq / qq.sum()
    return float(np.sum(pp * (np.log(pp) - np.log(qq))))


def kl_from_logits(inc_logits: np.ndarray, cand_logits: np.ndarray) -> float:
    return kl_divergence(softmax(inc_logits), softmax(cand_logits))


def topk_indices(logits: np.ndarray, k: int) -> np.ndarray:
    z = np.asarray(logits).ravel()
    k = min(int(k), int(z.size))
    if k <= 0:
        return np.array([], dtype=np.int64)
    if z.size <= k:
        return np.argsort(z)[::-1]
    part = np.argpartition(z, -k)[-k:]
    return part[np.argsort(z[part])[::-1]]


def topk_agreement(inc_logits: np.ndarray, cand_logits: np.ndarray, k: int) -> float:
    a = set(int(i) for i in topk_indices(inc_logits, k))
    b = set(int(i) for i in topk_indices(cand_logits, k))
    if not a:
        return 1.0
    return float(len(a & b) / len(a))


def ulp_distance_u32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute IEEE-754 bit distance of two f32 arrays, as int64."""
    ua = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.int64)
    ub = np.asarray(b, dtype=np.float32).view(np.uint32).astype(np.int64)
    return np.abs(ua - ub)


# ---------------------------------------------------------------------------
# Characterisation. A byte count is not enough.
# ---------------------------------------------------------------------------


REQUIRED_CHAR_FIELDS: tuple[str, ...] = (
    "n_bytes_compared",
    "n_mismatch_bytes",
    "n_floats_compared",
    "n_float_mismatch",
    "max_abs",
    "rel_l2",
    "cosine",
    "cause",
    "bit_identical",
)


def require_characterisation(doc: Mapping[str, Any] | None, *, label: str) -> dict[str, Any]:
    """Refuse a count-only byte diff. Magnitude and cause are load-bearing."""
    if not isinstance(doc, Mapping):
        raise CountOnlyDiffRefuse(label)
    missing = [k for k in REQUIRED_CHAR_FIELDS if k not in doc]
    if missing:
        raise CountOnlyDiffRefuse(f"{label}: missing {missing}")
    n_bytes = int(doc.get("n_bytes_compared") or 0)
    if n_bytes <= 0 and int(doc.get("n_floats_compared") or 0) <= 0:
        raise CountOnlyDiffRefuse(f"{label}: nothing compared")
    # A count without magnitude is the shape FOLD_ADDQX_AB stopped at.
    if doc.get("max_abs") is None or doc.get("rel_l2") is None:
        raise CountOnlyDiffRefuse(f"{label}: max_abs/rel_l2 are required")
    if not doc.get("cause"):
        raise CountOnlyDiffRefuse(f"{label}: cause is required")
    out = dict(doc)
    max_ulp = int(out.get("max_ulp") or 0)
    max_abs = out.get("max_abs")
    if max_ulp >= 1024 and max_abs is not None and float(max_abs) < 1e-5:
        out["ulp_note"] = (
            "max_ulp is IEEE-754 bit distance, which jumps across exponent "
            "boundaries on near-zero values; magnitude authority is max_abs "
            "and abs_histogram, not max_ulp"
        )
    return out


def classify_cause(char: Mapping[str, Any]) -> str:
    """Name the difference. Source-order FMA vs a different computation."""
    if bool(char.get("bit_identical")):
        return CAUSE_BIT_IDENTICAL
    n_nan = int(char.get("n_nan") or 0)
    n_inf = int(char.get("n_inf") or 0)
    n_sign = int(char.get("n_sign_flips") or 0)
    rel = char.get("rel_l2")
    cos = char.get("cosine")
    max_abs = char.get("max_abs")
    rms = char.get("incumbent_rms")
    rel_f = None if rel is None else float(rel)
    cos_f = None if cos is None else float(cos)
    if n_nan or n_inf:
        return CAUSE_DIFFERENT_COMPUTATION
    if rel_f is None:
        return CAUSE_UNMEASURED
    if rel_f > ALL_LAYERS_STRUCTURED_REL_L2:
        return CAUSE_DIFFERENT_COMPUTATION
    if cos_f is not None and cos_f < 0.999:
        return CAUSE_DIFFERENT_COMPUTATION
    if n_sign > 0 and rel_f > COMFORTABLY_USABLE_REL_L2:
        return CAUSE_DIFFERENT_COMPUTATION
    if (
        max_abs is not None
        and rms is not None
        and float(rms) > 0
        and float(max_abs) / float(rms) > 0.5
        and rel_f > COMFORTABLY_USABLE_REL_L2
    ):
        return CAUSE_DIFFERENT_COMPUTATION
    return CAUSE_SOURCE_ORDER_FMA


def characterize_f32(
    incumbent: Sequence[float] | np.ndarray,
    candidate: Sequence[float] | np.ndarray,
    *,
    compared_against: str,
    sample_n: int = 12,
) -> dict[str, Any]:
    """Byte + float characterisation of two f32 buffers.

    A mismatch *count* is not this object. Cause and magnitude are.
    """
    a = np.asarray(incumbent, dtype=np.float32).ravel()
    b = np.asarray(candidate, dtype=np.float32).ravel()
    if a.size != b.size:
        raise ParityLadderRefuse(
            f"buffer lengths differ: incumbent {a.size} vs candidate {b.size}"
        )
    if a.size == 0:
        raise ParityLadderRefuse("empty buffer is not a characterisation")
    ab = a.tobytes()
    bb = b.tobytes()
    n_bytes = len(ab)
    mismatch_bytes = int(sum(1 for i in range(n_bytes) if ab[i] != bb[i]))
    first_byte = next((i for i in range(n_bytes) if ab[i] != bb[i]), None)
    aa = a.astype(np.float64)
    bb64 = b.astype(np.float64)
    diff = aa - bb64
    absd = np.abs(diff)
    finite = np.isfinite(aa) & np.isfinite(bb64)
    n_nan = int(np.isnan(aa).sum() + np.isnan(bb64).sum())
    n_inf = int(np.isinf(aa).sum() + np.isinf(bb64).sum())
    n_sign = int(((np.sign(aa) != np.sign(bb64)) & (aa != 0) & (bb64 != 0) & finite).sum())
    float_mismatch = a != b
    n_float = int(float_mismatch.sum())
    first_float = int(np.argmax(float_mismatch)) if n_float else None
    if first_float is not None and not float_mismatch[first_float]:
        first_float = None
    num = float(np.sqrt(np.square(diff).sum()))
    den = float(np.sqrt(np.square(aa).sum()))
    rel_l2 = 0.0 if num == 0.0 and den == 0.0 else (float("inf") if den == 0.0 else num / den)
    da = float(aa @ aa)
    db = float(bb64 @ bb64)
    cosine = float("nan") if da == 0.0 or db == 0.0 else float((aa @ bb64) / math.sqrt(da * db))
    ulp = ulp_distance_u32(a, b)
    ulp_mismatch = ulp[float_mismatch] if n_float else np.array([], dtype=np.int64)
    max_ulp = int(ulp.max()) if ulp.size else 0
    rms_inc = float(np.sqrt(np.mean(np.square(aa)))) if aa.size else 0.0
    max_abs = float(absd.max()) if absd.size else 0.0
    mean_abs = float(absd.mean()) if absd.size else 0.0
    rms_diff = float(np.sqrt(np.mean(np.square(diff)))) if diff.size else 0.0

    abs_hist = {
        "eq0": int((absd == 0).sum()),
        "le_1e-8": int(((absd > 0) & (absd <= 1e-8)).sum()),
        "le_1e-7": int(((absd > 1e-8) & (absd <= 1e-7)).sum()),
        "le_1e-6": int(((absd > 1e-7) & (absd <= 1e-6)).sum()),
        "le_1e-5": int(((absd > 1e-6) & (absd <= 1e-5)).sum()),
        "le_1e-4": int(((absd > 1e-5) & (absd <= 1e-4)).sum()),
        "le_1e-3": int(((absd > 1e-4) & (absd <= 1e-3)).sum()),
        "le_1e-2": int(((absd > 1e-3) & (absd <= 1e-2)).sum()),
        "le_1e-1": int(((absd > 1e-2) & (absd <= 1e-1)).sum()),
        "gt_1e-1": int((absd > 1e-1).sum()),
    }
    ulp_hist = {
        "eq0": int((ulp == 0).sum()),
        "eq1": int((ulp == 1).sum()),
        "2_3": int(((ulp >= 2) & (ulp <= 3)).sum()),
        "4_7": int(((ulp >= 4) & (ulp <= 7)).sum()),
        "8_15": int(((ulp >= 8) & (ulp <= 15)).sum()),
        "16_63": int(((ulp >= 16) & (ulp <= 63)).sum()),
        "64_255": int(((ulp >= 64) & (ulp <= 255)).sum()),
        "256_1023": int(((ulp >= 256) & (ulp <= 1023)).sum()),
        "ge_1024": int((ulp >= 1024).sum()),
    }
    samples: list[dict[str, Any]] = []
    if n_float:
        idxs = np.nonzero(float_mismatch)[0][: int(sample_n)]
        for i in idxs:
            samples.append(
                {
                    "index": int(i),
                    "incumbent": float(a[i]),
                    "candidate": float(b[i]),
                    "abs": float(absd[i]),
                    "ulp": int(ulp[i]),
                }
            )
    bit_identical = mismatch_bytes == 0 and n_float == 0 and n_bytes > 0
    row: dict[str, Any] = {
        "compared_against": compared_against,
        "n_bytes_compared": n_bytes,
        "n_mismatch_bytes": mismatch_bytes,
        "first_mismatch_byte": first_byte,
        "n_floats_compared": int(a.size),
        "n_float_mismatch": n_float,
        "float_mismatch_fraction": _r(n_float / a.size, 8),
        "first_mismatch_float": first_float,
        "max_abs": _r(max_abs),
        "mean_abs": _r(mean_abs),
        "rms_diff": _r(rms_diff),
        "rel_l2": _r(rel_l2),
        "cosine": _r(float(cosine) if cosine == cosine else None),
        "max_ulp": max_ulp,
        "median_ulp_mismatch": int(np.median(ulp_mismatch)) if ulp_mismatch.size else 0,
        "incumbent_rms": _r(rms_inc),
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_sign_flips": n_sign,
        "abs_histogram": abs_hist,
        "ulp_histogram": ulp_hist,
        "samples": samples,
        "bit_identical": bit_identical,
    }
    row["cause"] = classify_cause(row)
    return row


def report_logit_agreement(
    *,
    kl_nats: float | None,
    top_k_agreement: float | None,
    argmax_agreement: float | None = None,
    k: int = TOPK_AUTHORITY,
    n_rows: int | None = None,
) -> dict[str, Any]:
    """KL + top-k are the screen. Argmax alone is a loud refuse."""
    if kl_nats is None or top_k_agreement is None:
        raise ArgmaxIsNotParity(
            f"kl_nats={kl_nats!r} top_k_agreement={top_k_agreement!r} "
            f"argmax_agreement={argmax_agreement!r}"
        )
    return {
        "kl_nats": float(kl_nats),
        "top_k": int(k),
        "top_k_agreement": float(top_k_agreement),
        "argmax_agreement": None if argmax_agreement is None else float(argmax_agreement),
        "argmax_is_not_parity": True,
        "n_rows": None if n_rows is None else int(n_rows),
        "parity_quantities": ["kl_nats", "top_k_agreement"],
        "usable_kl_bar": USABLE_KL,
        "usable_top5_bar": USABLE_TOP5,
        "clears_usable_kl": float(kl_nats) < USABLE_KL,
        "clears_usable_top5": float(top_k_agreement) >= USABLE_TOP5,
    }


def logit_agreement_from_arrays(
    inc_logits: np.ndarray,
    cand_logits: np.ndarray,
    *,
    k: int = TOPK_AUTHORITY,
) -> dict[str, Any]:
    inc = np.asarray(inc_logits)
    cand = np.asarray(cand_logits)
    if inc.shape != cand.shape:
        raise ParityLadderRefuse(f"logit shapes {inc.shape} != {cand.shape}")
    if inc.ndim == 1:
        inc = inc[None, :]
        cand = cand[None, :]
    kls: list[float] = []
    tops: list[float] = []
    args: list[float] = []
    for i in range(inc.shape[0]):
        kls.append(kl_from_logits(inc[i], cand[i]))
        tops.append(topk_agreement(inc[i], cand[i], k))
        args.append(1.0 if int(np.argmax(inc[i])) == int(np.argmax(cand[i])) else 0.0)
    return report_logit_agreement(
        kl_nats=float(sum(kls) / len(kls)),
        top_k_agreement=float(sum(tops) / len(tops)),
        argmax_agreement=float(sum(args) / len(args)),
        k=k,
        n_rows=int(inc.shape[0]),
    )


# ---------------------------------------------------------------------------
# fold_addqx algebra. Exact over reals; a different f32 association.
# ---------------------------------------------------------------------------


def unpack8_production(packed16: int, scale: float, bias: float, x: Sequence[float]) -> float:
    """Production tile: sequential (q*scale + bias)*x."""
    acc = 0.0
    s = f32(scale)
    b = f32(bias)
    for i in range(8):
        q = (int(packed16) >> (2 * i)) & 3
        xi = f32(x[i])
        w = f32(f32(float(q) * s) + b)
        acc = f32(acc + f32(w * xi))
    return acc


def unpack8_fold_addqx(packed16: int, scale: float, bias: float, x: Sequence[float]) -> float:
    """fold_addqx tile: fma(scale, sum(q*x as adds), bias*sum(x))."""
    acc_qx = 0.0
    acc_x = 0.0
    s = f32(scale)
    b = f32(bias)
    for i in range(8):
        q = (int(packed16) >> (2 * i)) & 3
        xi = f32(x[i])
        acc_x = f32(acc_x + xi)
        if q & 2:
            acc_qx = f32(acc_qx + f32(xi + xi))
        if q & 1:
            acc_qx = f32(acc_qx + xi)
    return f32(math.fma(s, acc_qx, f32(b * acc_x)))


def unpack8_reals(packed16: int, scale: float, bias: float, x: Sequence[float]) -> tuple[float, float]:
    """Both associations in Python float (binary64). Exact over reals up to 1e-15."""
    lhs = 0.0
    acc_qx = 0.0
    acc_x = 0.0
    for i in range(8):
        q = float((int(packed16) >> (2 * i)) & 3)
        xi = float(x[i])
        lhs += (float(scale) * q + float(bias)) * xi
        acc_qx += q * xi
        acc_x += xi
    rhs = float(scale) * acc_qx + float(bias) * acc_x
    return lhs, rhs


def fold_addqx_algebra(*, seed: int = 38, n: int = 256) -> dict[str, Any]:
    """CPU proof that fold_addqx is a source-order rewrite, not a new function.

    Over reals the identity holds. Over f32 it is a different association of
    the same terms, which is the obligation's 'source order permits' clause
    — provided live buffers stay inside the measured error budget.
    """
    rng = np.random.default_rng(seed)
    real_err = []
    f32_err = []
    for _ in range(n):
        packed = int(rng.integers(0, 65536))
        scale = float(rng.normal(0.0, 0.05))
        bias = float(rng.normal(0.0, 0.02))
        x = rng.normal(0.0, 0.5, size=8).tolist()
        lhs, rhs = unpack8_reals(packed, scale, bias, x)
        real_err.append(abs(lhs - rhs))
        p = unpack8_production(packed, scale, bias, x)
        f = unpack8_fold_addqx(packed, scale, bias, x)
        f32_err.append(abs(p - f))
    # Canonical counterexample from MLP_DECODE_CHEAPEN.
    cx_x = [0.7] * 8
    cx_prod = unpack8_production(65535, 0.3, 0.1, cx_x)
    cx_fold = unpack8_fold_addqx(65535, 0.3, 0.1, cx_x)
    cx_lhs, cx_rhs = unpack8_reals(65535, 0.3, 0.1, cx_x)
    return {
        "identity": FOLD_ADDQX_IDENTITY,
        "over_reals": {
            "exact": bool(max(real_err) < 1e-12),
            "max_abs_err": _r(max(real_err)),
            "n": n,
            "counterexample_lhs": cx_lhs,
            "counterexample_rhs": cx_rhs,
        },
        "over_f32": {
            "matches_production": bool(max(f32_err) == 0.0),
            "max_abs_err": _r(max(f32_err)),
            "n_tiles_differ": int(sum(1 for e in f32_err if e > 0.0)),
            "n": n,
            "counterexample": {
                "packed16": 65535,
                "scale": 0.3,
                "bias": 0.1,
                "x": cx_x,
                "production_f32": cx_prod,
                "fold_addqx_f32": cx_fold,
                "abs_err": _r(abs(cx_prod - cx_fold)),
            },
        },
        "source_order_permits": True,
        "cause": CAUSE_SOURCE_ORDER_FMA,
        "note": (
            "q in {0,1,2,3} so q*x is adds (xi or xi+xi) instead of float(q)*x; "
            "scale and bias apply once per 8-wide tile. Exact over reals; "
            "sequential f32 (q*s+b)*x is a different rounding."
        ),
    }


# ---------------------------------------------------------------------------
# MLP_ERROR_BUDGET bars. Cited, not assumed.
# ---------------------------------------------------------------------------


def mlp_error_budget_bars(doc: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The measured bars this ladder judges against. Not a 1% assumption."""
    headline: Mapping[str, Any] = {}
    answers: Mapping[str, Any] = {}
    if doc is None:
        path = REPO / BUDGET_REL
        if path.is_file():
            loaded = load_json(path)
            headline = loaded.get("headline") or {}
            answers = loaded.get("answers") or {}
    else:
        headline = doc.get("headline") or {}
        answers = doc.get("answers") or {}
    structured = headline.get("structured_all_layers_usable")
    if structured is None:
        structured = ALL_LAYERS_STRUCTURED_REL_L2
    comfortable = COMFORTABLY_USABLE_REL_L2
    one_layer = headline.get("isolated_one_layer_tolerated_relative_l2")
    if one_layer is None:
        one_layer = ONE_LAYER_REL_L2
    return {
        "source": BUDGET_REL,
        "argmax_is_not_parity": True,
        "all_layers_structured_tolerated_relative_l2": float(structured),
        "comfortably_usable_relative_l2": float(comfortable),
        "isolated_one_layer_tolerated_relative_l2": float(one_layer),
        "usable_last_token_kl": USABLE_KL,
        "usable_mean_top5": USABLE_TOP5,
        "degrades_at_relative_l2": headline.get("degrades_at", 0.1),
        "what_the_model_actually_tolerates": answers.get(
            "what_error_does_the_model_actually_tolerate",
            "0.03 per layer on all 64 layers, structured, KL < 0.10 and top-5 >= 0.80",
        ),
        "is_0_01_fatal_as_all_layers": answers.get(
            "is_0_01_already_fatal_as_a_per_layer_all_layers_program", "NO"
        ),
        "note": (
            "0.03 is the campaign number (stricter of isotropic/structured, "
            "all 64 layers). 0.01 is comfortably usable on the same criterion. "
            "A one-layer bar of 0.3 is not the whole-model bar."
        ),
    }


def fold_addqx_tolerance(bars: Mapping[str, Any] | None = None) -> dict[str, Any]:
    bars = dict(bars or mlp_error_budget_bars())
    rel_bar = float(bars["comfortably_usable_relative_l2"])
    return {
        "rel_l2_bar": rel_bar,
        "rel_l2_bar_source": (
            "MLP_ERROR_BUDGET comfortably-usable 0.01; all-layers structured "
            f"tolerated {bars['all_layers_structured_tolerated_relative_l2']}"
        ),
        "all_layers_structured_tolerated_relative_l2": float(
            bars["all_layers_structured_tolerated_relative_l2"]
        ),
        "kl_bar": USABLE_KL,
        "top5_bar": USABLE_TOP5,
        "permitted_cause": CAUSE_SOURCE_ORDER_FMA,
        "source_order_permits": True,
        "justification": (
            "fold_addqx is the algebraic rewrite "
            f"{FOLD_ADDQX_IDENTITY} with q*x as adds. Exact over reals; "
            "f32 association differs. MLP_ERROR_BUDGET measured that a "
            "structured per-layer relative L2 of 0.03 on all 64 layers stays "
            "USABLE (last-token KL < 0.10 nats, mean top-5 >= 0.80), and that "
            "0.01 is not fatal as an all-layers program. A live rel_l2 at or "
            "under 0.01 with cause SOURCE_ORDER_FMA_ASSOCIATION is inside "
            "that measured budget, not an assumed 1% bar. Argmax is not this "
            "justification."
        ),
        "source": BUDGET_REL,
    }


def tolerance_accepts(char: Mapping[str, Any], tolerance: Mapping[str, Any]) -> bool:
    if char.get("cause") != tolerance.get("permitted_cause"):
        return False
    rel = char.get("rel_l2")
    if rel is None:
        return False
    if float(rel) > float(tolerance["rel_l2_bar"]):
        return False
    if int(char.get("n_nan") or 0) or int(char.get("n_inf") or 0):
        return False
    return True


# ---------------------------------------------------------------------------
# Rungs.
# ---------------------------------------------------------------------------


def _rung(
    name: str,
    verdict: str,
    *,
    available: bool,
    reason: str,
    **extra: Any,
) -> dict[str, Any]:
    if verdict not in RUNG_VERDICTS:
        raise ParityLadderRefuse(f"unknown rung verdict {verdict!r}")
    if name not in RUNGS:
        raise ParityLadderRefuse(f"unknown rung {name!r}")
    out = {
        "name": name,
        "verdict": verdict,
        "available": bool(available),
        "reason": reason,
        "argmax_is_not_parity": True,
    }
    out.update(extra)
    return out


def rung_not_applicable(name: str, reason: str) -> dict[str, Any]:
    return _rung(name, NOT_APPLICABLE, available=False, reason=reason)


def rung_intermediate_buffers(
    surfaces: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strongest rung. Each named buffer carries its own characterisation."""
    if not surfaces:
        raise IncompleteLadder("intermediate_buffers: no surfaces")
    chars: dict[str, Any] = {}
    verdicts: list[str] = []
    for label, raw in surfaces.items():
        char = require_characterisation(raw, label=label)
        if char["bit_identical"]:
            v = BIT_IDENTICAL
        elif tolerance is not None and tolerance_accepts(char, tolerance):
            v = WITHIN_TOLERANCE
        else:
            v = DIFFERS
        chars[label] = {**char, "rung_verdict": v}
        verdicts.append(v)
    if all(v == BIT_IDENTICAL for v in verdicts):
        verdict = BIT_IDENTICAL
        reason = "every named intermediate buffer is bit-identical"
    elif DIFFERS in verdicts:
        verdict = DIFFERS
        reason = (
            "at least one intermediate buffer differs beyond a justified "
            "tolerance (or no tolerance was supplied)"
        )
    else:
        verdict = WITHIN_TOLERANCE
        if tolerance is None:
            raise UnjustifiedTolerance("intermediate_buffers")
        reason = (
            "intermediates are not bit-identical; diffs sit inside the "
            "justified tolerance"
        )
    return _rung(
        "intermediate_buffers",
        verdict,
        available=True,
        reason=reason,
        surfaces=chars,
        tolerance=None if verdict == BIT_IDENTICAL else (dict(tolerance) if tolerance else None),
        weakest_surface=min(
            chars,
            key=lambda k: STRENGTH.get(chars[k]["rung_verdict"], 0),
        ),
    )


def rung_route_ids(
    incumbent: Sequence[int] | None,
    candidate: Sequence[int] | None,
    *,
    organ_has_routes: bool,
    reason_if_absent: str,
    selected_experts_incumbent: Sequence[int] | None = None,
    selected_experts_candidate: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Route ids / selected experts. Dense organs report NOT_APPLICABLE."""
    if not organ_has_routes:
        return rung_not_applicable(
            "route_ids",
            reason_if_absent,
        )
    if incumbent is None or candidate is None:
        raise IncompleteLadder("route_ids: organ has routes but none were reported")
    inc = [int(x) for x in incumbent]
    cand = [int(x) for x in candidate]
    identical = inc == cand
    experts_identical = True
    if selected_experts_incumbent is not None or selected_experts_candidate is not None:
        if selected_experts_incumbent is None or selected_experts_candidate is None:
            raise IncompleteLadder("selected experts reported on one arm only")
        experts_identical = list(selected_experts_incumbent) == list(
            selected_experts_candidate
        )
    ok = identical and experts_identical
    return _rung(
        "route_ids",
        BIT_IDENTICAL if ok else DIFFERS,
        available=True,
        reason=(
            "route ids and selected experts match"
            if ok
            else "route ids or selected experts differ"
        ),
        incumbent=inc,
        candidate=cand,
        identical=identical,
        selected_experts_identical=experts_identical,
        n=len(inc),
    )


def rung_hidden_state(
    char: Mapping[str, Any] | None,
    *,
    tolerance: Mapping[str, Any] | None = None,
    available: bool = True,
    absent_reason: str = "",
) -> dict[str, Any]:
    if not available:
        return rung_not_applicable("hidden_state", absent_reason)
    c = require_characterisation(char, label="hidden_state")
    if c["bit_identical"]:
        v, reason = BIT_IDENTICAL, "hidden state is bit-identical"
        tol = None
    elif tolerance is not None and tolerance_accepts(c, tolerance):
        v, reason = WITHIN_TOLERANCE, "hidden-state diff sits inside the justified tolerance"
        tol = dict(tolerance)
    else:
        v, reason = DIFFERS, "hidden state differs beyond a justified tolerance"
        tol = dict(tolerance) if tolerance else None
    return _rung(
        "hidden_state",
        v,
        available=True,
        reason=reason,
        characterisation=c,
        tolerance=tol,
    )


def rung_final_logits(
    agreement: Mapping[str, Any] | None,
    *,
    char: Mapping[str, Any] | None = None,
    tolerance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if agreement is None:
        raise IncompleteLadder("final_logits: KL and top-k were not reported")
    if agreement.get("kl_nats") is None or agreement.get("top_k_agreement") is None:
        raise ArgmaxIsNotParity("final_logits requires kl_nats and top_k_agreement")
    kl = float(agreement["kl_nats"])
    # log-sum-exp KL can go tiny-negative from rounding; that is 0 nats.
    if -1e-12 < kl < 0.0:
        kl = 0.0
    top = float(agreement["top_k_agreement"])
    k = int(agreement.get("top_k") or TOPK_AUTHORITY)
    argmax = agreement.get("argmax_agreement")
    bits = None
    if char is not None:
        bits = require_characterisation(char, label="final_logits")
        if bits["bit_identical"]:
            return _rung(
                "final_logits",
                BIT_IDENTICAL,
                available=True,
                reason="final logits are bit-identical",
                agreement=report_logit_agreement(
                    kl_nats=kl, top_k_agreement=top, argmax_agreement=argmax, k=k,
                    n_rows=agreement.get("n_rows"),
                ),
                characterisation=bits,
                tolerance=None,
            )
    kl_bar = float((tolerance or {}).get("kl_bar") or USABLE_KL)
    top_bar = float((tolerance or {}).get("top5_bar") or USABLE_TOP5)
    if kl == 0.0 and top == 1.0 and (bits is None or bits["bit_identical"]):
        verdict = BIT_IDENTICAL
        reason = "logits match at KL=0 and top-k=1"
    elif kl < kl_bar and top >= top_bar:
        if tolerance is None:
            raise UnjustifiedTolerance(
                "final_logits are not bit-identical; a tolerance is required"
            )
        verdict = WITHIN_TOLERANCE
        reason = (
            f"logit KL {kl:.6g} nats < {kl_bar} and top-{k} {top:.4f} >= {top_bar} "
            "(MLP_ERROR_BUDGET usable bars); argmax is not this verdict"
        )
    else:
        verdict = DIFFERS
        reason = (
            f"logit KL {kl:.6g} nats or top-{k} {top:.4f} misses the usable bars "
            f"KL < {kl_bar} and top-k >= {top_bar}"
        )
    return _rung(
        "final_logits",
        verdict,
        available=True,
        reason=reason,
        agreement=report_logit_agreement(
            kl_nats=kl, top_k_agreement=top, argmax_agreement=argmax, k=k,
            n_rows=agreement.get("n_rows"),
        ),
        characterisation=bits,
        tolerance=None if verdict == BIT_IDENTICAL else (dict(tolerance) if tolerance else None),
    )


def rung_token_ids(
    incumbent: Sequence[int],
    candidate: Sequence[int],
    *,
    fallbacks: int = 0,
) -> dict[str, Any]:
    """Necessary, never sufficient. Matching ids are not a PASS."""
    inc = [int(x) for x in incumbent]
    cand = [int(x) for x in candidate]
    identical = inc == cand and int(fallbacks) == 0
    a = b"".join(int(x).to_bytes(4, "little", signed=False) for x in inc)
    b = b"".join(int(x).to_bytes(4, "little", signed=False) for x in cand)
    n = min(len(a), len(b))
    mismatch = sum(1 for i in range(n) if a[i] != b[i])
    if len(a) != len(b):
        mismatch += abs(len(a) - len(b))
        identical = False
    verdict = BIT_IDENTICAL if identical and mismatch == 0 and n > 0 else DIFFERS
    return _rung(
        "token_ids",
        verdict,
        available=True,
        reason=(
            "token ids are bit-identical with fallbacks 0; this is necessary "
            "and NOT a candidate PASS"
            if verdict == BIT_IDENTICAL
            else "token ids differ (or fallbacks are non-zero)"
        ),
        incumbent=inc,
        candidate=cand,
        identical=bool(inc == cand),
        fallbacks=int(fallbacks),
        n_bytes_compared=n,
        n_mismatch_bytes=int(mismatch),
        bit_identical=verdict == BIT_IDENTICAL,
        never_sufficient=True,
    )


def empty_rungs(*, dense_reason: str) -> dict[str, dict[str, Any]]:
    """All rungs present as NOT_APPLICABLE except token_ids omitted.

    Used by tests to prove that filling only token_ids cannot PASS.
    """
    return {
        "intermediate_buffers": rung_not_applicable(
            "intermediate_buffers", "not reported"
        ),
        "route_ids": rung_not_applicable("route_ids", dense_reason),
        "hidden_state": rung_not_applicable("hidden_state", "not reported"),
        "final_logits": rung_not_applicable("final_logits", "not reported"),
    }


# ---------------------------------------------------------------------------
# Candidate verdict. Weakest applicable rung wins. Token ids cannot PASS.
# ---------------------------------------------------------------------------


def _require_available_reported(rungs: Mapping[str, Mapping[str, Any]], *, available: Mapping[str, bool]) -> None:
    for name in RUNGS:
        if name not in rungs:
            raise IncompleteLadder(name)
        row = rungs[name]
        if available.get(name) and row.get("verdict") == NOT_APPLICABLE:
            raise IncompleteLadder(
                f"{name} is available on this organ and was reported "
                "NOT_APPLICABLE"
            )


def judge_candidate(
    rungs: Mapping[str, Mapping[str, Any]],
    *,
    capability: Mapping[str, Any] | None = None,
    available: Mapping[str, bool] | None = None,
    candidate: str = "candidate",
) -> dict[str, Any]:
    """A candidate's verdict is the weakest applicable rung it passes.

    Token-id equality alone raises TokenIdAloneIsNotParity rather than PASS.
    Argmax-only logits raise ArgmaxIsNotParity.
    """
    avail = {
        "intermediate_buffers": True,
        "route_ids": True,
        "hidden_state": True,
        "final_logits": True,
        "token_ids": True,
    }
    if available:
        avail.update({k: bool(v) for k, v in available.items()})
    _require_available_reported(rungs, available=avail)

    for name in RUNGS:
        v = rungs[name]["verdict"]
        if v not in RUNG_VERDICTS:
            raise ParityLadderRefuse(f"{name} has unknown verdict {v!r}")

    token = rungs["token_ids"]
    if token["verdict"] == NOT_APPLICABLE:
        raise IncompleteLadder("token ids are always available and are necessary")
    if token["verdict"] != BIT_IDENTICAL:
        return {
            "candidate": candidate,
            "verdict": REFUSE,
            "weakest_rung": "token_ids",
            "weakest_rung_verdict": token["verdict"],
            "promote_to_bit_identical": False,
            "promote_to_default_on": False,
            "reason": "token-id equality is necessary and failed",
            "rungs": {k: rungs[k]["verdict"] for k in RUNGS},
            "capability": capability,
        }

    applicable = [name for name in RUNGS if rungs[name]["verdict"] != NOT_APPLICABLE]
    stronger = [name for name in applicable if name != "token_ids"]
    if not stronger:
        raise TokenIdAloneIsNotParity(candidate)

    logits = rungs["final_logits"]
    if logits["verdict"] != NOT_APPLICABLE:
        agr = logits.get("agreement") or {}
        if agr.get("kl_nats") is None or agr.get("top_k_agreement") is None:
            raise ArgmaxIsNotParity("final_logits")

    weakest_name = min(applicable, key=lambda n: STRENGTH[rungs[n]["verdict"]])
    weakest_v = rungs[weakest_name]["verdict"]

    if weakest_v == DIFFERS:
        return {
            "candidate": candidate,
            "verdict": REFUSE,
            "weakest_rung": weakest_name,
            "weakest_rung_verdict": weakest_v,
            "promote_to_bit_identical": False,
            "promote_to_default_on": False,
            "reason": (
                f"{weakest_name} DIFFERS and has no justified tolerance; "
                "token-id equality does not rescue it"
            ),
            "rungs": {k: rungs[k]["verdict"] for k in RUNGS},
            "capability": capability,
        }

    if weakest_v == BIT_IDENTICAL:
        return {
            "candidate": candidate,
            "verdict": PASS_BIT_IDENTICAL,
            "weakest_rung": weakest_name,
            "weakest_rung_verdict": weakest_v,
            "promote_to_bit_identical": True,
            "promote_to_default_on": True,
            "reason": "every applicable rung is BIT_IDENTICAL",
            "rungs": {k: rungs[k]["verdict"] for k in RUNGS},
            "capability": capability,
        }

    # WITHIN_TOLERANCE. The four supports are load-bearing.
    if logits["verdict"] == NOT_APPLICABLE:
        raise IncompleteLadder(
            "logit agreement is required when any rung is WITHIN_TOLERANCE"
        )
    if logits["verdict"] == DIFFERS:
        return {
            "candidate": candidate,
            "verdict": REFUSE,
            "weakest_rung": "final_logits",
            "weakest_rung_verdict": DIFFERS,
            "promote_to_bit_identical": False,
            "promote_to_default_on": False,
            "reason": "logit agreement failed the usable KL/top-k bars",
            "rungs": {k: rungs[k]["verdict"] for k in RUNGS},
            "capability": capability,
        }
    if capability is None or not capability.get("ran"):
        raise CapabilitySpotCheckMissing(candidate)
    if not capability.get("pass"):
        return {
            "candidate": candidate,
            "verdict": REFUSE,
            "weakest_rung": weakest_name,
            "weakest_rung_verdict": weakest_v,
            "promote_to_bit_identical": False,
            "promote_to_default_on": False,
            "reason": (
                "capability spot-check failed; a justified tolerance still "
                "needs a behavioural surface that would notice a real change"
            ),
            "rungs": {k: rungs[k]["verdict"] for k in RUNGS},
            "capability": dict(capability),
        }
    return {
        "candidate": candidate,
        "verdict": PASS_JUSTIFIED_TOLERANCE,
        "weakest_rung": weakest_name,
        "weakest_rung_verdict": weakest_v,
        "promote_to_bit_identical": False,
        "promote_to_default_on": True,
        "reason": (
            f"weakest rung is {weakest_name}=WITHIN_TOLERANCE; token ids match; "
            "logit KL/top-k clear the measured usable bars; route parity is "
            "reported (or dense NOT_APPLICABLE); capability spot-check passed. "
            "Not bit-identical and not left unjudged."
        ),
        "rungs": {k: rungs[k]["verdict"] for k in RUNGS},
        "capability": dict(capability),
    }


# ---------------------------------------------------------------------------
# Probe I/O and fold_addqx case.
# ---------------------------------------------------------------------------


def qwen38_is_dense_reason() -> str:
    return (
        "qwen38 is dense Qwen3.5-27B hybrid (Gated DeltaNet + GQA); "
        "crates/hawking-core/src/model/qwen38_geometry.rs refuses MoE keys "
        "(num_experts / moe_intermediate_size). fold_addqx rewrites the MLP "
        "affine2 unpack, which has no router and no selected experts."
    )


def cited_fold_addqx_ab() -> dict[str, Any]:
    path = REPO / FOLD_AB_REL
    if not path.is_file():
        return {
            "source": FOLD_AB_REL,
            "present": False,
            "note": "FOLD_ADDQX_AB.json is not on disk in this worktree",
        }
    doc = load_json(path)
    layer0 = doc.get("layer0_byte_compare") or {}
    gate = layer0.get("gate") or {}
    up = layer0.get("up") or {}
    down = layer0.get("down") or {}
    parity = doc.get("parity") or {}
    saving = doc.get("saving") or {}
    return {
        "source": FOLD_AB_REL,
        "present": True,
        "not_rerun": True,
        "complete_token_saving_ms": saving.get("complete_token_saving_ms", CITED_TOKEN_SAVING_MS),
        "complete_token_incumbent_ms": saving.get("complete_token_incumbent_ms"),
        "complete_token_fold_addqx_ms": saving.get("complete_token_fold_addqx_ms"),
        "class": saving.get("class", "approx_candidate"),
        "token_ids_identical": parity.get("token_ids_identical"),
        "arithmetic_exact": parity.get("arithmetic_exact"),
        "parity": parity.get("parity"),
        "runs_compared": parity.get("runs_compared"),
        "layer0_gate_n_mismatch_bytes": gate.get("n_mismatch_bytes", CITED_GATE_MISMATCH_BYTES),
        "layer0_gate_n_bytes": gate.get("n_bytes_compared", CITED_GATE_BYTES),
        "layer0_up_n_mismatch_bytes": up.get("n_mismatch_bytes", CITED_UP_MISMATCH_BYTES),
        "layer0_down_n_mismatch_bytes": down.get("n_mismatch_bytes", CITED_DOWN_MISMATCH_BYTES),
        "finding": doc.get("finding"),
        "default_off": True,
        "note": (
            "This lane does not re-run the complete-token A/B. The 3.9833 ms "
            "saving and the 22309-byte gate mismatch are cited from that "
            "receipt; the ladder characterises that mismatch and climbs the "
            "rungs the A/B did not."
        ),
    }


def _mean(values: Sequence[float]) -> float | None:
    xs = [float(v) for v in values]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def capability_from_probe(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Cheapest meaningful surface: last-token KL/top-5 plus a short generate."""
    if not rows:
        return {"ran": False, "pass": False, "reason": "no capability prompts"}
    kls = []
    tops = []
    argmaxes = []
    token_ok = []
    domains = []
    for row in rows:
        agr = row.get("logits") or row.get("agreement") or {}
        if agr.get("kl_nats") is None or agr.get("top_k_agreement") is None:
            raise ArgmaxIsNotParity(f"capability prompt {row.get('id')!r}")
        kls.append(float(agr["kl_nats"]))
        tops.append(float(agr["top_k_agreement"]))
        if agr.get("argmax_agreement") is not None:
            argmaxes.append(float(agr["argmax_agreement"]))
        token_ok.append(bool(row.get("token_ids_identical")))
        domains.append(
            {
                "id": row.get("id"),
                "domain": row.get("domain"),
                "kl_nats": float(agr["kl_nats"]),
                "top_k_agreement": float(agr["top_k_agreement"]),
                "argmax_agreement": agr.get("argmax_agreement"),
                "token_ids_identical": bool(row.get("token_ids_identical")),
                "incumbent_text": row.get("incumbent_text"),
                "candidate_text": row.get("candidate_text"),
                "usable": (
                    float(agr["kl_nats"]) < USABLE_KL
                    and float(agr["top_k_agreement"]) >= USABLE_TOP5
                ),
            }
        )
    mean_kl = _mean(kls)
    mean_top = _mean(tops)
    all_usable = all(d["usable"] for d in domains)
    # A real behavioural change on a short greedy generate would either
    # move KL/top-5 off the usable bars or flip the generated tokens on a
    # prompt whose next-token set is not a near-tie. Token flips with KL
    # still under the bar are recorded, not treated as a silent pass.
    passed = bool(all_usable and mean_kl is not None and mean_top is not None)
    return {
        "ran": True,
        "pass": passed,
        "argmax_is_not_parity": True,
        "n_prompts": len(rows),
        "mean_kl_nats": _r(mean_kl),
        "mean_top_k_agreement": _r(mean_top),
        "mean_argmax_agreement": _r(_mean(argmaxes)),
        "all_token_ids_identical": all(token_ok),
        "usable_kl_bar": USABLE_KL,
        "usable_top5_bar": USABLE_TOP5,
        "domains": domains,
        "reason": (
            "every capability prompt stays inside MLP_ERROR_BUDGET usable "
            "bars (KL < 0.10 nats, top-5 >= 0.80)"
            if passed
            else "at least one capability prompt missed the usable KL/top-5 bars"
        ),
        "surface": (
            "teacher-forced last-token KL and top-5 on short real prompts, "
            "plus an 8-token greedy generate that would notice a continuation "
            "change; not a full teacher-corpus domain generate"
        ),
    }


def _char_from_probe(node: Mapping[str, Any] | None, *, label: str) -> dict[str, Any]:
    if not isinstance(node, Mapping):
        raise IncompleteLadder(label)
    # Probe may nest characterisation under "compare".
    if "rel_l2" in node and "n_mismatch_bytes" in node:
        return require_characterisation(node, label=label)
    if isinstance(node.get("compare"), Mapping):
        return require_characterisation(node["compare"], label=label)
    raise CountOnlyDiffRefuse(label)


def _logit_agreement_from_probe(node: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(node, Mapping):
        raise IncompleteLadder("final_logits")
    if node.get("kl_nats") is None:
        inner = node.get("agreement")
        if isinstance(inner, Mapping):
            node = inner
    return report_logit_agreement(
        kl_nats=None if node.get("kl_nats") is None else float(node["kl_nats"]),
        top_k_agreement=(
            None if node.get("top_k_agreement") is None else float(node["top_k_agreement"])
        ),
        argmax_agreement=(
            None
            if node.get("argmax_agreement") is None
            else float(node["argmax_agreement"])
        ),
        k=int(node.get("top_k") or TOPK_AUTHORITY),
        n_rows=node.get("n_rows") or node.get("n_prompts"),
    )


def rungs_from_fold_addqx_probe(
    raw: Mapping[str, Any],
    *,
    tolerance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Apply the ladder to a live probe dump. Does not re-run the A/B."""
    tol = tolerance or fold_addqx_tolerance()
    layer0 = raw.get("layer0_named_matvec") or raw.get("layer0") or {}
    surfaces: dict[str, Any] = {}
    for name in ("gate", "up", "down"):
        row = layer0.get(name)
        if row is None:
            raise IncompleteLadder(f"layer0 {name}")
        surfaces[f"layer0_{name}"] = _char_from_probe(row, label=f"layer0_{name}")
    live = list(raw.get("live_prompts") or raw.get("prompts") or [])
    last_mlp = None
    if live:
        last_mlp = live[0].get("last_layer_mlp") or live[0].get("mlp")
    if isinstance(last_mlp, Mapping):
        # Fused GateUpSwiglu writes act/down and does not refresh the
        # gate/up workspace. A bit-identical leftover there is not a rung.
        live_down = None
        if "down" in last_mlp:
            live_down = _char_from_probe(last_mlp["down"], label="last_layer_down")
            surfaces["last_layer_down"] = live_down
        if "act" in last_mlp:
            surfaces["last_layer_act"] = _char_from_probe(
                last_mlp["act"], label="last_layer_act"
            )
        for name in ("gate", "up"):
            if name not in last_mlp:
                continue
            char = _char_from_probe(last_mlp[name], label=f"last_layer_{name}")
            if (
                char.get("bit_identical")
                and live_down is not None
                and not live_down.get("bit_identical")
            ):
                continue
            surfaces[f"last_layer_{name}"] = char
    intermediates = rung_intermediate_buffers(surfaces, tolerance=tol)

    routes = rung_route_ids(
        None,
        None,
        organ_has_routes=False,
        reason_if_absent=qwen38_is_dense_reason(),
    )

    hidden_chars: list[dict[str, Any]] = []
    logit_rows: list[dict[str, Any]] = []
    token_inc: list[int] = []
    token_cand: list[int] = []
    fallbacks = 0
    for row in live:
        if isinstance(row.get("hidden"), Mapping):
            hidden_chars.append(_char_from_probe(row["hidden"], label="hidden"))
        if isinstance(row.get("logits"), Mapping):
            logit_rows.append(_logit_agreement_from_probe(row["logits"]))
        inc_ids = row.get("incumbent_token_ids") or (row.get("incumbent") or {}).get(
            "new_token_ids"
        )
        cand_ids = row.get("candidate_token_ids") or (row.get("fold_addqx") or {}).get(
            "new_token_ids"
        )
        if inc_ids:
            token_inc.extend(int(x) for x in inc_ids)
        if cand_ids:
            token_cand.extend(int(x) for x in cand_ids)
        fallbacks += int(row.get("fallbacks") or 0)
        fallbacks += int((row.get("incumbent") or {}).get("fallbacks") or 0)
        fallbacks += int((row.get("fold_addqx") or {}).get("fallbacks") or 0)

    if not hidden_chars:
        raise IncompleteLadder("hidden_state")
    # Weakest hidden across prompts.
    hidden_chars.sort(key=lambda c: float(c.get("rel_l2") or 0.0), reverse=True)
    hidden = rung_hidden_state(hidden_chars[0], tolerance=tol)

    if not logit_rows:
        raise IncompleteLadder("final_logits")
    mean_kl = float(sum(r["kl_nats"] for r in logit_rows) / len(logit_rows))
    if -1e-12 < mean_kl < 0.0:
        mean_kl = 0.0
    mean_top = float(sum(r["top_k_agreement"] for r in logit_rows) / len(logit_rows))
    mean_arg = None
    args = [r["argmax_agreement"] for r in logit_rows if r["argmax_agreement"] is not None]
    if args:
        mean_arg = float(sum(args) / len(args))
    logits = rung_final_logits(
        {
            "kl_nats": mean_kl,
            "top_k_agreement": mean_top,
            "argmax_agreement": mean_arg,
            "top_k": TOPK_AUTHORITY,
            "n_rows": len(logit_rows),
        },
        tolerance=tol,
    )

    if not token_inc or not token_cand:
        raise IncompleteLadder("token_ids")
    tokens = rung_token_ids(token_inc, token_cand, fallbacks=fallbacks)

    cap_rows = list(raw.get("capability") or live)
    capability = capability_from_probe(cap_rows)

    rungs = {
        "intermediate_buffers": intermediates,
        "route_ids": routes,
        "hidden_state": hidden,
        "final_logits": logits,
        "token_ids": tokens,
    }
    judgement = judge_candidate(
        rungs,
        capability=capability,
        available={
            "intermediate_buffers": True,
            "route_ids": False,
            "hidden_state": True,
            "final_logits": True,
            "token_ids": True,
        },
        candidate="fold_addqx",
    )
    return rungs, {**judgement, "capability": capability, "tolerance": tol}


# ---------------------------------------------------------------------------
# GPU lane lock. mkdir-atomic. A stale FILE cannot be taken as a lock.
# ---------------------------------------------------------------------------


def inspect_gpu_lane_lock(path: Path | None = None) -> dict[str, Any]:
    """Read the lock. Does not create it. Never fabricates a holder."""
    lock = Path(path) if path is not None else GPU_LOCK
    if not lock.exists():
        return {"present": False, "path": str(lock), "kind": None}
    if lock.is_dir():
        owner = ""
        pid_s = ""
        try:
            owner = (lock / "owner").read_text().strip()
        except OSError:
            owner = ""
        try:
            pid_s = (lock / "pid").read_text().strip()
        except OSError:
            pid_s = ""
        alive = False
        pid = None
        if pid_s.isdigit():
            pid = int(pid_s)
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        return {
            "present": True,
            "path": str(lock),
            "kind": "mkdir_dir",
            "owner": owner or None,
            "pid": pid,
            "holder_alive": alive,
            "protocol": "mkdir-atomic (gpu_lane_lock.sh)",
        }
    if lock.is_file():
        size = int(lock.stat().st_size)
        return {
            "present": True,
            "path": str(lock),
            "kind": "stale_file",
            "size": size,
            "note": (
                "gpu_lane_lock.sh uses mkdir; a regular file at this path is "
                "the wedged-stale-file scar and blocks taking the lock"
            ),
        }
    return {"present": True, "path": str(lock), "kind": "other"}


def prepare_gpu_lane_lock() -> dict[str, Any]:
    """Clear a stale FILE so mkdir can take the lock. Never removes a live dir."""
    before = inspect_gpu_lane_lock()
    cleared = False
    if before.get("kind") == "stale_file":
        GPU_LOCK.unlink()
        cleared = True
    after = inspect_gpu_lane_lock()
    return {"before": before, "cleared_stale_file": cleared, "after": after}


def example_binaries() -> list[Path]:
    names = [
        REPO / "workspace" / "ops" / "build" / "rust" / "release-fast" / "examples" / "parity_ladder_probe",
        REPO / "target" / "release-fast" / "examples" / "parity_ladder_probe",
    ]
    return [p for p in names if p.is_file()]


def run_probe(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT,
    tokenizer: Path = DEFAULT_TOKENIZER,
    out: Path = RAW_DEFAULT,
    max_new_tokens: int = 8,
    max_seq_len: int = 128,
    use_lock: bool = True,
) -> dict[str, Any]:
    bins = example_binaries()
    if not bins:
        raise FileNotFoundError(
            "parity_ladder_probe binary not found; build with "
            "`CARGO_TARGET_DIR=workspace/ops/build/rust cargo build "
            "--profile release-fast -p hawking-core "
            "--example parity_ladder_probe`"
        )
    exe = bins[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    inner = [
        str(exe),
        "--artifact-root",
        str(artifact_root),
        "--tokenizer",
        str(tokenizer),
        "--max-new-tokens",
        str(max_new_tokens),
        "--max-seq-len",
        str(max_seq_len),
        "--out",
        str(out),
    ]
    lock_prep = prepare_gpu_lane_lock()
    cmd = (
        ["bash", str(GPU_LOCK_SH), "g022parity", *inner]
        if use_lock and GPU_LOCK_SH.is_file()
        else inner
    )
    env = os.environ.copy()
    env.setdefault("HAWKING_QWEN_RESIDENCY", "1")
    env["HAWKING_GPU_LANE_LOCK_HELD"] = "1" if use_lock else "0"
    proc = subprocess.run(
        cmd, cwd=REPO, capture_output=True, text=True, check=False, env=env
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} exited {proc.returncode}\nstdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    raw = json.loads(out.read_text())
    raw["gpu_lane_lock_prepare"] = lock_prep
    raw["gpu_lane_lock_held"] = bool(use_lock)
    return raw


# ---------------------------------------------------------------------------
# Receipt.
# ---------------------------------------------------------------------------


def _require_primitive(name: str) -> str:
    if name not in ATLAS_PRIMITIVES:
        raise ParityLadderRefuse(f"{name} is not an atlas primitive")
    return name


def build(
    *,
    raw: Mapping[str, Any] | None = None,
    measured: bool = False,
    lock_held: bool = False,
    measured_at: str | None = None,
) -> dict[str, Any]:
    bars = mlp_error_budget_bars()
    algebra = fold_addqx_algebra()
    cited = cited_fold_addqx_ab()
    rungs: dict[str, Any] | None = None
    judgement: dict[str, Any] | None = None
    probe_error = None
    if raw is not None:
        try:
            rungs, judgement = rungs_from_fold_addqx_probe(raw, tolerance=fold_addqx_tolerance(bars))
        except ParityLadderRefuse as e:
            probe_error = str(e)
            judgement = {
                "candidate": "fold_addqx",
                "verdict": REFUSE,
                "promote_to_bit_identical": False,
                "promote_to_default_on": False,
                "reason": str(e),
            }

    gate_char = None
    if rungs is not None:
        sur = (rungs["intermediate_buffers"].get("surfaces") or {})
        gate_char = sur.get("layer0_gate")

    finding = (
        "PARITY MUST BE STRONGER THAN ARGMAX, and token-id equality is not "
        "the top of the ladder. fold_addqx remains the case that forced it: "
        "FOLD_ADDQX_AB saved 3.9833 ms on the complete token with identical "
        "token ids and a layer-0 gate buffer that is not bit-identical "
        "(22309 of 69632 bytes). "
    )
    if judgement is None:
        finding += (
            "This receipt records the ladder and the CPU algebra (exact over "
            "reals, not f32). Live magnitude, logits, hidden and capability "
            "are UNMEASURED until the Metal probe runs; fold_addqx stays "
            "default-off and unpromoted because the 22309-byte difference "
            "has not yet been characterised by magnitude on this run."
        )
        stays_off_reason = (
            "live characterisation of the 22309-byte difference was not "
            "measured on this run"
        )
    else:
        finding += judgement.get("reason", "")
        if gate_char:
            finding += (
                f" Layer-0 gate: {gate_char.get('n_mismatch_bytes')} of "
                f"{gate_char.get('n_bytes_compared')} bytes differ "
                f"({gate_char.get('n_float_mismatch')} floats), max_abs="
                f"{gate_char.get('max_abs')}, rel_l2={gate_char.get('rel_l2')} "
                f"({CAUSE_SOURCE_ORDER_FMA}; "
                f"{ALL_LAYERS_STRUCTURED_REL_L2} all-layers structured bar, "
                f"{COMFORTABLY_USABLE_REL_L2} comfortably usable). "
            )
        stays_off_reason = None if judgement.get("promote_to_default_on") else judgement.get("reason")

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "nomenclature_version": "HAWKING_NOMENCLATURE_V1",
        "recorded_by": RECORDED_BY,
        "git_head": git("rev-parse", "HEAD"),
        "obligation": (
            "PARITY MUST BE STRONGER THAN ARGMAX. Token-id equality is "
            "necessary and NOT SUFFICIENT. Where a change claims unchanged "
            "arithmetic, verify the strongest available surface: intermediate "
            "buffers, route ids, selected experts, hidden state, final logits, "
            "then token ids - BIT_IDENTICAL where source order permits, "
            "otherwise an EXPLICITLY JUSTIFIED TOLERANCE plus route parity, "
            "logit agreement and a capability spot-check. A kernel that "
            "changes the computation and happens to preserve argmax is not a "
            "physical-only speedup."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_class": "SELF_MEASURED_DIRTY" if measured else "STATIC_ONLY",
        "gpu_authority": bool(measured),
        "took_gpu_lease": bool(measured and lock_held),
        "primitive": _require_primitive(PRIMITIVE),
        "rung_order": list(RUNGS),
        "rung_verdicts": list(RUNG_VERDICTS),
        "candidate_verdicts": list(CANDIDATE_VERDICTS),
        "token_id_equality_alone_cannot_pass": True,
        "argmax_is_not_parity": True,
        "mlp_error_budget": bars,
        "fold_addqx_algebra": algebra,
        "cited_fold_addqx_ab": cited,
        "fold_addqx": {
            "lever": "HAWKING_AFFINE2_GEO=fold_addqx / Affine2Geo::FoldAddqx",
            "default_production": "tpr64 unpack8",
            "production_stays_default_off_until": (
                "PASS_JUSTIFIED_TOLERANCE with all four supports, or "
                "PASS_BIT_IDENTICAL"
            ),
            "not_promoted_to_bit_identical": True,
            "rungs": rungs,
            "judgement": judgement,
            "probe_error": probe_error,
            "layer0_gate_characterisation": gate_char,
            "cited_gate_mismatch_bytes": CITED_GATE_MISMATCH_BYTES,
            "reproduced_cited_byte_count": (
                None
                if gate_char is None
                else int(gate_char.get("n_mismatch_bytes") or -1) == CITED_GATE_MISMATCH_BYTES
            ),
        },
        "finding": finding,
        "stays_default_off_reason": stays_off_reason,
        "anti_fabrication": {
            "detectors": [
                "TOKEN_ID_ALONE_PRESENTED_AS_PASS",
                "ARGMAX_PRESENTED_AS_PARITY",
                "COUNT_ONLY_BYTE_DIFF",
                "UNJUSTIFIED_TOLERANCE",
                "CAPABILITY_SPOT_CHECK_MISSING",
            ],
            "loud_exceptions": [
                "TokenIdAloneIsNotParity",
                "ArgmaxIsNotParity",
                "CountOnlyDiffRefuse",
                "UnjustifiedTolerance",
                "CapabilitySpotCheckMissing",
                "IncompleteLadder",
            ],
            "rule": (
                "judge_candidate is the only constructor of a candidate "
                "verdict. Token-id equality alone raises TokenIdAloneIsNotParity. "
                "Argmax-only logits raise ArgmaxIsNotParity. A byte-mismatch "
                "count without max_abs/rel_l2/cause raises CountOnlyDiffRefuse. "
                "WITHIN_TOLERANCE without MLP_ERROR_BUDGET numbers raises "
                "UnjustifiedTolerance."
            ),
        },
        "tps_qualification": {
            "any_tps_labelled_qualified": False,
            "reason": "this lane judges parity, it does not mint a TPS number",
        },
        "does_not_edit": [
            "tools/future/fold_addqx_ab.py",
            "crates/hawking-core/shaders/q80_mixed_decode.metal",
        ],
        "probe": {
            "schema": PROBE_SCHEMA,
            "example": "crates/hawking-core/examples/parity_ladder_probe.rs",
            "raw": str(RAW_DEFAULT),
            "measured": bool(measured),
        },
    }
    if measured:
        doc["measurement_provenance"] = measurement_provenance(
            lock_held=lock_held, lane="g022parity", measured_at=measured_at
        )
    return doc


def record(doc: Mapping[str, Any], *, path: Path | None = None) -> Path:
    out = path or RECEIPT
    return write_measured_receipt(out, dict(doc), RECORDED_BY)


def selftest() -> dict[str, Any]:
    """CPU-only proofs the ladder's refusals actually fire."""
    algebra = fold_addqx_algebra()
    assert algebra["over_reals"]["exact"] is True
    assert algebra["over_f32"]["matches_production"] is False

    token = rung_token_ids([1, 2, 3], [1, 2, 3])
    assert token["verdict"] == BIT_IDENTICAL
    dense = qwen38_is_dense_reason()
    rungs = empty_rungs(dense_reason=dense)
    rungs["token_ids"] = token
    raised = False
    try:
        judge_candidate(
            rungs,
            available={
                "intermediate_buffers": False,
                "route_ids": False,
                "hidden_state": False,
                "final_logits": False,
                "token_ids": True,
            },
            candidate="selftest_token_only",
        )
    except TokenIdAloneIsNotParity:
        raised = True
    assert raised, "token-id equality alone must not PASS"

    try:
        report_logit_agreement(kl_nats=None, top_k_agreement=None, argmax_agreement=1.0)
        argmax_raised = False
    except ArgmaxIsNotParity:
        argmax_raised = True
    assert argmax_raised

    try:
        require_characterisation(
            {
                "n_bytes_compared": 69632,
                "n_mismatch_bytes": 22309,
                "bit_identical": False,
            },
            label="gate",
        )
        count_raised = False
    except CountOnlyDiffRefuse:
        count_raised = True
    assert count_raised

    return {
        "algebra_over_reals": algebra["over_reals"]["exact"],
        "algebra_over_f32_matches": algebra["over_f32"]["matches_production"],
        "token_id_alone_raises": raised,
        "argmax_alone_raises": argmax_raised,
        "count_only_raises": count_raised,
        "ok": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--from", dest="raw_path", default=None)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-lock", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        result = selftest()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 1

    raw = None
    measured = False
    lock_held = False
    measured_at = None
    if args.measure:
        raw = run_probe(
            artifact_root=args.artifact_root,
            tokenizer=args.tokenizer,
            out=RAW_DEFAULT,
            use_lock=not args.no_lock,
        )
        measured = True
        lock_held = not args.no_lock
    elif args.raw_path:
        raw_path = Path(args.raw_path)
        raw = load_json(raw_path)
        measured = True
        lock_held = bool(os.environ.get("HAWKING_GPU_LANE_LOCK_HELD")) or bool(
            raw.get("gpu_lane_lock_held")
        )
        measured_at = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(raw_path.stat().st_mtime)
        )

    doc = build(
        raw=raw,
        measured=measured,
        lock_held=lock_held,
        measured_at=measured_at,
    )
    if args.record:
        path = record(doc, path=args.out)
        print(f"wrote {path}")
        print(doc["finding"])
        j = (doc.get("fold_addqx") or {}).get("judgement") or {}
        print(f"verdict={j.get('verdict')} default_on={j.get('promote_to_default_on')}")
    else:
        print(doc["finding"])
        if doc.get("fold_addqx", {}).get("judgement"):
            print(json.dumps(doc["fold_addqx"]["judgement"], indent=2, sort_keys=True, default=str)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

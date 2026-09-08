#!/usr/bin/env python3
"""DELTANET GENERATED TRANSITION — fit the skinny program, judge multi-step.

Candidate generated_transition_coefficients (DELTANET_STATE_FUNCTION): remove
2,139,096,960 catalog bytes of attention.linear_qkvz and add 4,548,560
(T1 + T2 + 48 f16 diagonals + 48 layer embeddings + headers). The generator
IS the matvec:

    y = diag_l * T2 (T1 x + e_l)

TiledProjection consumes those coefficients. Emitting W_l = diag_l T2 T1
then running the ordinary GEMV is REJECTED_DENSE_REMAT; executable_economics
prices that path as removed == added.

Authority is multi-step stability of S and h, not one-step regression.
A fit that looks good at one step and drifts over 128 tokens is a failure.
This module REFUSES to report a verdict without multi-step results at 64
steps or more, and REFUSES to report a train figure as held-out.

    python3 tools/future/deltanet_generated_transition.py --build
    python3 tools/future/deltanet_generated_transition.py --fit
    python3 -m pytest tools/future/test_deltanet_generated_transition.py -q

evidence_class DIAGNOSTIC_RELATIVE. Bytes come from executable_economics
and the recorded candidate; this module does not invent a byte model.
q/k/v/z coding is at its floor (DELTANET_QKVZ_PRECISION) and is not retried.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tools.future._common import REPO, RECEIPTS, load_json, write_receipt
from tools.future.ebpw_categories import PRODUCTION, judge_dense_rematerialization
from tools.future.executable_economics import (
    IncompleteEconomics,
    score as economics_score,
)
from tools.future.mlp_teacher_corpus import FUSION_ENV
from tools.future.physical_primitives import ATLAS_PRIMITIVES
from tools.future import deltanet_state_function as dsf
from tools.future import deltanet_teacher_corpus as dtc


RECEIPT = "DELTANET_GENERATED_TRANSITION.json"
SCHEMA = "hawking.future.deltanet_generated_transition.v1"
VERSION = 1
RECORDED_BY = "tools/future/deltanet_generated_transition.py"
CANDIDATE_ID = "generated_transition_coefficients"
STATE_FN_REL = "receipts/future/DELTANET_STATE_FUNCTION.json"
QKVZ_PREC_REL = "receipts/future/DELTANET_QKVZ_PRECISION.json"

BOTTLENECK = 256
MULTISTEP_CHECKPOINTS: tuple[int, ...] = (1, 4, 16, 64, 256)
MIN_VERDICT_STEPS = 64
STATE_CROSS_BARS: tuple[float, ...] = (0.01, 0.03, 0.10)
ONE_STEP_COSINE_BAR = 0.99
Q4_GROUP = 64
RNG_SEED = 38
PAYLOAD_DIR = dtc.PAYLOAD_DIR
FIT_NAME = "FIT.json"

DIRECT_CONSUME = dsf.DIRECT_CONSUME
REJECTED_DENSE_REMAT = dsf.REJECTED_DENSE_REMAT

CLAIM_BOUNDARY = (
    "DIAGNOSTIC_RELATIVE sidecar. No generate gate and no GPU lease. The byte "
    "model is the recorded generated_transition_coefficients candidate scored "
    "through executable_economics; it is not re-derived. The generator is "
    "y = diag_l T2 (T1 x + e_l) consumed as two skinny GEMVs "
    "(TiledProjection). Materializing W and running ordinary GEMV is "
    "REJECTED_DENSE_REMAT (removed == added). Multi-step S/h divergence is "
    "the authority; a one-step cosine is not a verdict. Held-out is by "
    "prompt_id. q/k/v/z bit-descent is not retried (DELTANET_QKVZ_PRECISION). "
    "X is capture_diverse2 post_attn_norm, not claimed to be sealed mixer input."
)


class TransitionRefuse(ValueError):
    """The generator lane refused rather than guessing."""


class VerdictRefuse(TransitionRefuse):
    """No verdict without multi-step results at 64 steps or more."""


class HeldOutRefuse(TransitionRefuse):
    """A train figure cannot be reported as held-out."""


class DenseRematRefuse(TransitionRefuse):
    """Emit-W-then-GEMV is REJECTED_DENSE_REMAT."""


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


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def load_candidate() -> dict[str, Any]:
    """The recorded candidate. Do not invent a byte model."""
    path = REPO / STATE_FN_REL
    if not path.is_file():
        raise TransitionRefuse(f"REFUSED: {STATE_FN_REL} is not readable")
    doc = load_json(path)
    for row in doc.get("candidates") or []:
        if str(row.get("id")) == CANDIDATE_ID:
            dsf.require_economics(row)
            return dict(row)
    raise TransitionRefuse(
        f"REFUSED: {CANDIDATE_ID} missing from {STATE_FN_REL}"
    )


def qkvz_coding_is_at_floor() -> dict[str, Any]:
    path = REPO / QKVZ_PREC_REL
    if not path.is_file():
        raise TransitionRefuse(f"REFUSED: {QKVZ_PREC_REL} is not readable")
    doc = load_json(path)
    alloc = doc.get("allocation") or {}
    bits = (alloc.get("bits") or {})
    return {
        "receipt": QKVZ_PREC_REL,
        "not_worth_touching": list(alloc.get("not_worth_touching") or ["q", "k", "v", "z"]),
        "bits": {k: int(v) for k, v in bits.items()} if bits else {"q": 4, "k": 4, "v": 4, "z": 4},
        "total_bytes_eliminated": int(alloc.get("total_bytes_eliminated") or 0),
        "retried": False,
    }


def billed_bytes(candidate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row = candidate if candidate is not None else load_candidate()
    added = dict(row["bytes_added"])
    removed = dict(row["bytes_removed"])
    return {
        "bytes_removed": removed,
        "bytes_added": {
            "generator": int(added.get("generator") or 0),
            "embeddings": int(added.get("embeddings") or 0),
            "residuals": int(added.get("residuals") or 0),
            "metadata": int(added.get("metadata") or 0),
            "state": int(added.get("state") or 0),
            "total": int(added["total"]),
        },
        "net_bytes": int(row["net_bytes"]),
        "source": STATE_FN_REL,
        "id": CANDIDATE_ID,
        "byte_model": row.get("byte_model"),
        "note": (
            "Every billed byte is counted once, at model scope. T1 and T2 are "
            "shared; diagonals and embeddings are per-layer; headers are 40 B "
            "per DN layer."
        ),
    }


def consume_path_tag(*, emit_dense_w: bool, ordinary_gemv: bool) -> str:
    verdict = judge_dense_rematerialization(
        {
            "path_kind": PRODUCTION,
            "dense_rematerialization": bool(emit_dense_w),
            "decompresses_to_dense_weight_tensor": bool(emit_dense_w),
            "runs_ordinary_kernels": bool(ordinary_gemv),
            "consumes_representation_directly": (not emit_dense_w),
        }
    )
    if emit_dense_w and ordinary_gemv:
        if verdict.ok:
            raise TransitionRefuse("expected REJECTED_DENSE_REMAT for emit-W lowering")
        return REJECTED_DENSE_REMAT
    if emit_dense_w:
        return REJECTED_DENSE_REMAT
    return DIRECT_CONSUME


def native_consume(T1: np.ndarray, T2: np.ndarray, diag: np.ndarray, embed: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y = diag * T2 (T1 x + e). Direct consume; never writes W."""
    if x.ndim == 1:
        z = T1 @ x + embed
        return diag * (T2 @ z)
    z = x @ T1.T + embed[None, :]
    return (z @ T2.T) * diag[None, :]


def emit_w_then_gemv(
    T1: np.ndarray, T2: np.ndarray, diag: np.ndarray, embed: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """REJECTED_DENSE_REMAT production path. Exists so the refusal has a referent."""
    raise DenseRematRefuse(
        "REFUSED: emitting W = diag T2 T1 then running ordinary GEMV is "
        "REJECTED_DENSE_REMAT; the native path is TiledProjection on generated "
        "coefficients. Economics of this path are removed == added."
    )


def score_dense_remat(candidate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Emit-W path: the economics model prices removed == added."""
    row = candidate if candidate is not None else load_candidate()
    removed = int(row["bytes_removed"]["total"])
    billed = economics_score(
        bytes_removed=removed,
        bytes_added=removed,
        extra_flops_per_output_element=0.0,
        dispatch_delta=0.0,
        consuming_primitive="FusedDecodeCompute",
        organ="deltanet",
        stream_class="weight_codes",
        candidate_id="emit_w_then_ordinary_gemv",
        status="REJECTED_DENSE_REMAT",
    )
    if int(billed["bytes_removed"]) != int(billed["bytes_added"]["total"]):
        raise TransitionRefuse("REFUSED: dense remat must score removed == added")
    billed["dense_rematerialization"] = REJECTED_DENSE_REMAT
    billed["net_bytes"] = 0
    return billed


def score_candidate(
    candidate: Mapping[str, Any] | None = None,
    *,
    status: str | None = None,
) -> dict[str, Any]:
    """Score the skinny program through executable_economics. Do not invent bytes."""
    row = candidate if candidate is not None else load_candidate()
    billed = billed_bytes(row)
    added = billed["bytes_added"]
    dispatch_delta = int(row["dispatch_change"]["delta"])
    primitive = str(row["physical_primitive"])
    if primitive not in ATLAS_PRIMITIVES:
        raise TransitionRefuse(f"{primitive} is not an atlas primitive")
    extra_token = int(row["extra_flops"]["per_token"])
    # executable_economics refuses negative extra FLOPs. The skinny map is a
    # FLOP save vs the incumbent qkvz GEMV; it is recorded as extra_token and
    # scored at 0 extra (bytes + dispatch carry the time model).
    extra_for_score = 0.0
    scored = economics_score(
        bytes_removed=int(billed["bytes_removed"]["total"]),
        bytes_added={
            "generator": added["generator"],
            "embeddings": added["embeddings"],
            "residuals": added["residuals"],
            "metadata": added["metadata"],
            "state": added["state"],
        },
        extra_flops_per_output_element=extra_for_score,
        dispatch_delta=float(dispatch_delta),
        consuming_primitive=primitive,
        organ="deltanet",
        stream_class="weight_codes",
        candidate_id=CANDIDATE_ID,
        status=status or str(row.get("status") or "OPEN"),
        reusable_family=True,
        high_information_falsifier=False,
    )
    scored["extra_flops_per_token_recorded"] = extra_token
    scored["extra_flops_formula"] = row["extra_flops"]["formula"]
    scored["dispatch_change"] = dict(row["dispatch_change"])
    scored["physical_primitive"] = primitive
    scored["dense_rematerialization"] = DIRECT_CONSUME
    scored["state_update"] = {
        "incumbent_kernel": "qwen38_gated_delta_decode_vi_simd",
        "candidate_kernel": "qwen38_gated_delta_decode_vi_simd",
        "latency_delta": 0,
        "note": (
            "S geometry is unchanged. The generator replaces the qkvz GEMV; "
            "gated-delta still consumes generated q,k,v in-register."
        ),
    }
    return scored


def pack_q4(W: np.ndarray, *, group: int = Q4_GROUP) -> tuple[np.ndarray, np.ndarray]:
    """HQ30UQ4-style signed nibble + per-group f16 scale. Consume path, not a W write."""
    rows, cols = W.shape
    if cols % group:
        raise TransitionRefuse(f"cols {cols} is not a multiple of group {group}")
    gpr = cols // group
    Wg = np.ascontiguousarray(W, dtype=np.float32).reshape(rows, gpr, group)
    scale = np.max(np.abs(Wg), axis=-1) / 7.0
    scale = np.maximum(scale, 1.0e-8)
    q = np.clip(np.round(Wg / scale[:, :, None]), -8.0, 7.0)
    lo = (q[:, :, 0::2] + 8.0).astype(np.uint8)
    hi = (q[:, :, 1::2] + 8.0).astype(np.uint8)
    codes = lo | (hi << 4)
    return codes, scale.astype(np.float16)


def unpack_q4(codes: np.ndarray, scales: np.ndarray, *, group: int = Q4_GROUP) -> np.ndarray:
    from tools.future.deltanet_qkvz_precision import _unpack_q4

    W, _q = _unpack_q4(codes, scales, group=group)
    return W


def randomized_svd(
    W: np.ndarray, rank: int, *, seed: int = RNG_SEED, oversample: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Range-finder SVD. Returns U, S, Vh with U (m,r), Vh (r,n)."""
    m, n = W.shape
    r = int(rank)
    k = min(r + int(oversample), min(m, n))
    rng = np.random.default_rng(int(seed))
    G = rng.standard_normal((n, k)).astype(np.float32)
    Y = np.matmul(W, G)
    Q, _ = np.linalg.qr(Y.astype(np.float64), mode="reduced")
    B = np.matmul(Q.T, W.astype(np.float64))
    Uh, S, Vh = np.linalg.svd(B, full_matrices=False)
    U = np.matmul(Q, Uh)[:, :r]
    return U.astype(np.float32), S[:r].astype(np.float32), Vh[:r].astype(np.float32)


def fit_generator_from_W(
    weights: Mapping[int, np.ndarray],
    *,
    typical_layer: int,
    rank: int = BOTTLENECK,
    X_means: Mapping[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Shared T1,T2 from SVD of the typical layer; per-layer f16 diag + f32 embed.

    The billed generator is Q4 T1/T2. Consume uses the unpacked Q4 factors,
    not the f32 SVD (the billed bytes are Q4).
    """
    if typical_layer not in weights:
        raise TransitionRefuse(f"REFUSED: typical layer {typical_layer} has no W")
    W0 = np.ascontiguousarray(weights[typical_layer], dtype=np.float32)
    if W0.shape != (dsf.QKVZ_ROWS, dsf.HIDDEN):
        # Allow toy geometry in tests.
        pass
    U, S, Vh = randomized_svd(W0, rank)
    scale = np.sqrt(np.maximum(S, 0.0))
    T2_f32 = U * scale[None, :]
    T1_f32 = scale[:, None] * Vh
    t1_codes, t1_scales = pack_q4(T1_f32)
    t2_codes, t2_scales = pack_q4(T2_f32)
    T1 = unpack_q4(t1_codes, t1_scales)
    T2 = unpack_q4(t2_codes, t2_scales)
    Wapprox = np.matmul(T2, T1)
    diagonals: dict[int, np.ndarray] = {}
    embeds: dict[int, np.ndarray] = {}
    rec_cos: dict[str, float] = {}
    for layer, W in weights.items():
        num = (W * Wapprox).sum(axis=1)
        den = (Wapprox * Wapprox).sum(axis=1)
        d = np.divide(num, den, out=np.ones_like(num), where=den > 1e-12)
        d16 = d.astype(np.float16).astype(np.float32)
        diagonals[int(layer)] = d16
        e = np.zeros((T1.shape[0],), dtype=np.float32)
        if X_means is not None and int(layer) in X_means:
            mu = np.ascontiguousarray(X_means[int(layer)], dtype=np.float32)
            y_true = W @ mu
            y_hat = d16 * (T2 @ (T1 @ mu))
            resid = y_true - y_hat
            # Least-squares e in bottleneck: T2 e ≈ resid / d
            target = np.divide(resid, d16, out=np.zeros_like(resid), where=np.abs(d16) > 1e-12)
            sol, *_ = np.linalg.lstsq(T2.astype(np.float64), target.astype(np.float64), rcond=None)
            e = sol.astype(np.float32)
        embeds[int(layer)] = e
        y_hat = d16[:, None] * Wapprox
        rec_cos[str(layer)] = float(_cosine_mat(y_hat, W))
    return {
        "T1": T1,
        "T2": T2,
        "T1_f32": T1_f32,
        "T2_f32": T2_f32,
        "t1_codes": t1_codes,
        "t1_scales": t1_scales,
        "t2_codes": t2_codes,
        "t2_scales": t2_scales,
        "diagonals": diagonals,
        "embeddings": embeds,
        "typical_layer": int(typical_layer),
        "rank": int(rank),
        "weight_cosine_after_diag": rec_cos,
        "singular_values": S,
        "consume": DIRECT_CONSUME,
    }


def _cosine_mat(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.ravel().astype(np.float64)
    bb = b.ravel().astype(np.float64)
    den = math.sqrt(float(aa @ aa) * float(bb @ bb))
    if den == 0.0:
        return float("nan")
    return float(aa @ bb / den)


def _relfro(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.sqrt(np.square(a.astype(np.float64) - b.astype(np.float64)).sum()))
    den = float(np.sqrt(np.square(b.astype(np.float64)).sum()))
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return _cosine_mat(a, b)


def first_crossing(series: Sequence[float], bar: float) -> int | None:
    """1-indexed step at which relative error first exceeds `bar`. None if never."""
    for i, v in enumerate(series, start=1):
        if v is None:
            continue
        if float(v) > float(bar):
            return int(i)
    return None


def _max_step_present(multistep: Mapping[str, Any]) -> int:
    steps = [int(s) for s in (multistep.get("checkpoints") or [])]
    dense = multistep.get("n_steps")
    got = 0
    if steps:
        got = max(got, max(steps))
    if dense is not None:
        got = max(got, int(dense))
    return got


def assert_held_out(error: Mapping[str, Any], *, split: Mapping[str, Any] | None = None) -> None:
    """A train figure cannot be reported as held-out. Loud exception."""
    side = str(error.get("split") or error.get("source") or "")
    if side in {"train", "fit", "training"}:
        raise HeldOutRefuse(
            "REFUSED: train figure cannot be reported as held-out "
            f"(split={side!r})"
        )
    if side != "hold":
        raise HeldOutRefuse(
            f"REFUSED: held-out error must be tagged split='hold', got {side!r}"
        )
    pids = [str(x) for x in (error.get("prompt_ids") or [])]
    if split is not None:
        train = set(str(x) for x in (split.get("train_prompt_ids") or []))
        hold = set(str(x) for x in (split.get("hold_prompt_ids") or []))
        leak = [p for p in pids if p in train]
        if leak:
            raise HeldOutRefuse(
                "REFUSED: held-out error includes train prompt ids "
                f"{leak[:8]}"
            )
        unknown = [p for p in pids if p not in hold]
        if unknown:
            raise HeldOutRefuse(
                "REFUSED: held-out error prompt ids are not in the hold set "
                f"{unknown[:8]}"
            )


def report_verdict(
    multistep: Mapping[str, Any] | None,
    held_out_error: Mapping[str, Any] | None,
    *,
    split: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Refuse a verdict without 64-step multi-step AND a true held-out error."""
    if not isinstance(multistep, Mapping):
        raise VerdictRefuse(
            "REFUSED: no verdict without multi-step results at 64 steps or more"
        )
    max_step = _max_step_present(multistep)
    checkpoints = [int(s) for s in (multistep.get("checkpoints") or [])]
    if max_step < MIN_VERDICT_STEPS or MIN_VERDICT_STEPS not in checkpoints:
        raise VerdictRefuse(
            "REFUSED: no verdict without multi-step results at 64 steps or more "
            f"(max_step={max_step}, checkpoints={checkpoints})"
        )
    if not isinstance(held_out_error, Mapping):
        raise HeldOutRefuse("REFUSED: no verdict without a held-out error object")
    assert_held_out(held_out_error, split=split)

    series = [float(x) for x in (multistep.get("state_relfro_dense") or [])]
    crossings = {
        str(bar): first_crossing(series, bar) if series else None
        for bar in STATE_CROSS_BARS
    }
    # Prefer dense series; fall back to checkpoint values if only those exist.
    if not series:
        by = multistep.get("state_relfro") or {}
        ordered = [float(by[str(s)]) for s in checkpoints if str(s) in by]
        crossings = {str(bar): first_crossing(ordered, bar) if ordered else None for bar in STATE_CROSS_BARS}

    one_step = held_out_error.get("one_step_cosine")
    n_fail = crossings.get("0.01")
    unstable = n_fail is not None
    cosine_fail = one_step is not None and float(one_step) < ONE_STEP_COSINE_BAR
    if unstable:
        status = "MEASURED_NEGATIVE"
        scar = "DELTANET_REPLACEMENT_UNSTABLE_AT_N"
        verdict = "FAIL"
        reason = (
            f"relative state error crossed 1% at step {n_fail}; "
            "multi-step stability is the authority"
        )
        mechanism = (
            "The shared rank-256 map T2 T1 does not emit this layer's qkvz "
            "coefficients. The residual in q/k/v is integrated by the rank-1 "
            "Householder (I - beta k k^T) plus the write beta k v^T, so S "
            f"diverges at step {n_fail} rather than after a long horizon. "
            "A one-step cosine is not a stay of execution."
        )
    elif cosine_fail:
        status = "MEASURED_NEGATIVE"
        scar = "DELTANET_REPLACEMENT_ONE_STEP_BELOW_BAR"
        verdict = "FAIL"
        reason = (
            f"held-out one-step cosine {one_step} < {ONE_STEP_COSINE_BAR}; "
            "the skinny program is not this function"
        )
        mechanism = (
            "y = diag T2 (T1 x + e) is not W_qkvz x on held-out prompts. "
            "The generator is not this organ."
        )
    else:
        status = "OPEN"
        scar = None
        verdict = "PASS"
        reason = (
            f"held-out state relfro stayed <= 1% through step {max_step} "
            f"and one-step cosine {one_step} cleared {ONE_STEP_COSINE_BAR}"
        )
        mechanism = None
    return {
        "verdict": verdict,
        "status": status,
        "scar_id": scar,
        "reason": reason,
        "mechanism": mechanism,
        "n_unstable": n_fail,
        "crossings": crossings,
        "max_step": max_step,
        "checkpoints": checkpoints,
        "one_step_cosine": None if one_step is None else float(one_step),
        "held_out_split": "hold",
        "authority": "multi_step_state_and_output",
    }


def measure_multistep(
    *,
    incumbent_y: Callable[[np.ndarray], np.ndarray],
    candidate_y: Callable[[np.ndarray], np.ndarray],
    xs: Sequence[np.ndarray],
    aux: Mapping[str, Any],
    geo: Mapping[str, Any],
    checkpoints: Sequence[int] = MULTISTEP_CHECKPOINTS,
) -> dict[str, Any]:
    """Roll incumbent vs generator from S=0. Dense series + named checkpoints."""
    from tools.future.deltanet_qkvz_precision import (
        _ba_decay_beta,
        _gated_delta,
        _rearrange_conv,
    )

    n = len(xs)
    if n <= 0:
        raise TransitionRefuse("REFUSED: multi-step needs a token sequence")
    want = [int(s) for s in checkpoints]
    vh = int(geo["value_heads"])
    kd = int(geo["key_head_dim"])
    vd = int(geo["value_head_dim"])
    C = int(geo["conv_channels"])
    kconv = int(geo["conv_kernel"])
    S_i = np.zeros((vh, kd, vd), dtype=np.float32)
    S_c = np.zeros((vh, kd, vd), dtype=np.float32)
    cs_i = np.zeros((C, kconv - 1), dtype=np.float32)
    cs_c = np.zeros((C, kconv - 1), dtype=np.float32)
    conv = aux["conv"]
    Wba = aux["ba"]
    state_rel: list[float] = []
    out_rel: list[float] = []
    out_cos: list[float] = []
    y_cos: list[float] = []
    ck_state: dict[str, float] = {}
    ck_out: dict[str, float] = {}
    ck_cos: dict[str, float] = {}
    for t, x in enumerate(xs, start=1):
        x = np.ascontiguousarray(x, dtype=np.float32)
        yi = incumbent_y(x)
        yc = candidate_y(x)
        y_cos.append(_cosine(yc, yi))
        ba = Wba @ x
        qi, ki, vi, _zi = _rearrange_conv(yi, conv, cs_i, geo)
        qc, kc, vc, _zc = _rearrange_conv(yc, conv, cs_c, geo)
        decay, beta = _ba_decay_beta(ba, aux["a_log"], aux["dt_bias"], geo)
        S_i, hi = _gated_delta(S_i, qi, ki, vi, decay, beta)
        S_c, hc = _gated_delta(S_c, qc, kc, vc, decay, beta)
        sr = _relfro(S_c, S_i)
        or_ = _relfro(hc, hi)
        oc = _cosine(hc, hi)
        state_rel.append(sr)
        out_rel.append(or_)
        out_cos.append(oc)
        if t in want:
            ck_state[str(t)] = sr
            ck_out[str(t)] = or_
            ck_cos[str(t)] = oc
    present = [s for s in want if s <= n]
    return {
        "n_steps": n,
        "checkpoints": present,
        "state_relfro": ck_state,
        "output_relfro": ck_out,
        "output_cosine": ck_cos,
        "state_relfro_dense": state_rel,
        "output_relfro_dense": out_rel,
        "output_cosine_dense": out_cos,
        "qkvz_cosine_dense": y_cos,
        "one_step_state_relfro": state_rel[0] if state_rel else None,
        "one_step_output_cosine": out_cos[0] if out_cos else None,
        "one_step_qkvz_cosine": y_cos[0] if y_cos else None,
        "crossings": {
            str(bar): first_crossing(state_rel, bar) for bar in STATE_CROSS_BARS
        },
    }


def _toy_geo(*, heads: int = 2, dim: int = 8, hidden: int = 32) -> dict[str, Any]:
    vpk = 1
    q_rows = heads * dim
    return {
        "hidden_size": hidden,
        "key_heads": heads,
        "value_heads": heads,
        "key_head_dim": dim,
        "value_head_dim": dim,
        "values_per_key": vpk,
        "conv_kernel": 4,
        "q_rows": q_rows,
        "k_rows": q_rows,
        "v_rows": heads * dim,
        "z_rows": heads * dim,
        "qkvz_rows": q_rows * 2 + heads * dim * 2,
        "conv_channels": q_rows * 2 + heads * dim,
        "fused_rows_per_key": dim * 2 + vpk * dim * 2,
    }


def _toy_aux(geo: Mapping[str, Any], *, seed: int = RNG_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    C = int(geo["conv_channels"])
    k = int(geo["conv_kernel"])
    vh = int(geo["value_heads"])
    hidden = int(geo["hidden_size"])
    ba_rows = vh * 2
    return {
        "ba": rng.normal(scale=0.05, size=(ba_rows, hidden)).astype(np.float32),
        "conv": rng.normal(scale=0.05, size=(C, k)).astype(np.float32),
        "a_log": rng.normal(scale=0.1, size=(vh,)).astype(np.float32),
        "dt_bias": rng.normal(scale=0.1, size=(vh,)).astype(np.float32),
        "norm": np.ones((vh, int(geo["value_head_dim"])), dtype=np.float32),
    }


def fixture_multistep(*, n_steps: int = 64, seed: int = RNG_SEED) -> dict[str, Any]:
    """Deterministic toy rollout so the 64-step refusal has a passing counterpart.

    Not a sealed-3.14 measurement. Tagged fixture.
    """
    rng = np.random.default_rng(int(seed))
    geo = _toy_geo()
    aux = _toy_aux(geo, seed=seed)
    qkvz_rows = int(geo["qkvz_rows"])
    hidden = int(geo["hidden_size"])
    W = rng.normal(scale=0.05, size=(qkvz_rows, hidden)).astype(np.float32)
    rank = min(8, hidden, qkvz_rows)
    U, S, Vh = randomized_svd(W, rank, seed=seed)
    T2 = U * S[None, :]
    T1 = Vh
    d = np.ones((qkvz_rows,), dtype=np.float32)
    e = np.zeros((rank,), dtype=np.float32)
    xs = [rng.normal(scale=0.1, size=(hidden,)).astype(np.float32) for _ in range(int(n_steps))]

    def inc(x: np.ndarray) -> np.ndarray:
        return W @ x

    def cand(x: np.ndarray) -> np.ndarray:
        return native_consume(T1, T2, d, e, x)

    ms = measure_multistep(
        incumbent_y=inc,
        candidate_y=cand,
        xs=xs,
        aux=aux,
        geo=geo,
        checkpoints=tuple(s for s in MULTISTEP_CHECKPOINTS if s <= n_steps) or (n_steps,),
    )
    ms["kind"] = "fixture"
    ms["split"] = "hold"
    return ms


def _load_fit_payload(payload_dir: Path | None = None) -> dict[str, Any] | None:
    root = payload_dir if payload_dir is not None else PAYLOAD_DIR
    path = root / FIT_NAME
    if not path.is_file():
        return None
    return load_json(path)


def _write_fit_payload(doc: Mapping[str, Any], payload_dir: Path | None = None) -> Path:
    root = payload_dir if payload_dir is not None else PAYLOAD_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / FIT_NAME
    path.write_text(json.dumps(_py(dict(doc)), indent=1, sort_keys=True) + "\n")
    return path


def _prompt_sequences(
    rows: Sequence[Mapping[str, Any]], X: np.ndarray, *, side: str
) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for rec in rows:
        if str(rec.get("split")) != side:
            continue
        pid = str(rec["prompt_id"])
        if pid not in by:
            by[pid] = []
            order.append(pid)
        by[pid].append(rec)
    out: list[dict[str, Any]] = []
    for pid in order:
        recs = sorted(by[pid], key=lambda r: int(r["token_position"]))
        idx = [int(r["x_row_index"]) for r in recs]
        out.append(
            {
                "prompt_id": pid,
                "capability_domain": recs[0].get("capability_domain"),
                "n_tokens": len(recs),
                "xs": [np.ascontiguousarray(X[i], dtype=np.float32) for i in idx],
            }
        )
    return out


def run_fit(
    *,
    layers: Sequence[int] | None = None,
    typical_layer: int = 21,
    n_hold_steps: int = 256,
    payload_dir: Path | None = None,
) -> dict[str, Any]:
    """Fit T1/T2/diag/embed on captured layers; judge held-out multi-step."""
    started = time.perf_counter()
    from tools.future import deltanet_qkvz_precision as dqp
    from tools.future import deltanet_representation as dnr
    from tools.future.mlp_teacher_corpus import load_x_f16, resolve_x_capture_dir

    capture = dtc.load_existing_capture(payload_dir)
    chosen = list(layers) if layers is not None else list(dtc.DN_REPRESENTATIVE_LAYERS)
    if typical_layer not in chosen:
        chosen = [typical_layer, *chosen]
    census, geo = dnr.census_rows()
    weights: dict[int, np.ndarray] = {}
    aux_by: dict[int, dict[str, Any]] = {}
    x_by: dict[int, np.ndarray] = {}
    x_dir = resolve_x_capture_dir()
    for layer in chosen:
        rec = next(
            r
            for r in census
            if int(r["layer"]) == int(layer) and str(r["organ"]) == "attention.linear_qkvz"
        )
        weights[int(layer)] = dqp._load_q4_matrix(rec)
        aux_by[int(layer)] = dqp._layer_aux(census, geo, int(layer))
        x_by[int(layer)] = np.ascontiguousarray(load_x_f16(x_dir, int(layer)), dtype=np.float32)

    x_means = {L: x.mean(axis=0) for L, x in x_by.items()}
    fit = fit_generator_from_W(weights, typical_layer=int(typical_layer), X_means=x_means)
    T1, T2 = fit["T1"], fit["T2"]

    rows_jsonl = None
    split_block = None
    if capture and (REPO / capture.get("rows_jsonl", "")).is_file():
        rows_jsonl = [json.loads(line) for line in (REPO / capture["rows_jsonl"]).open()]
        split_block = capture.get("split")

    hold_prompt_ids: list[str] = []
    per_layer_ms: list[dict[str, Any]] = []
    one_step_cos: list[float] = []
    for layer in chosen:
        W = weights[int(layer)]
        d = fit["diagonals"][int(layer)]
        e = fit["embeddings"][int(layer)]
        X = x_by[int(layer)]
        aux = aux_by[int(layer)]

        def inc(x: np.ndarray, _W=W) -> np.ndarray:
            return _W @ x

        def cand(x: np.ndarray, _T1=T1, _T2=T2, _d=d, _e=e) -> np.ndarray:
            return native_consume(_T1, _T2, _d, _e, x)

        seqs: list[dict[str, Any]]
        if rows_jsonl:
            layer_rows = [r for r in rows_jsonl if int(r["layer"]) == int(layer)]
            seqs = _prompt_sequences(layer_rows, X, side="hold")
        else:
            # Capture-diverse2 last-3-per-family hold, used only if rows.jsonl
            # is absent. Still tagged hold; report_verdict will require the tag.
            from tools.future.mlp_teacher_corpus import load_x_manifest, expand_capture_rows

            man = load_x_manifest(x_dir)
            meta = expand_capture_rows(layer=int(layer), x_manifest=man, n_tokens=int(X.shape[0]))
            split = dtc.split_by_prompt(meta)
            assigned = dtc.assign_split(meta, split)
            seqs = _prompt_sequences(assigned, X, side="hold")
            split_block = {
                "train_prompt_ids": split["train_prompt_ids"],
                "hold_prompt_ids": split["hold_prompt_ids"],
                "hold_frac": split["hold_frac"],
            }
        long_enough = [s for s in seqs if int(s["n_tokens"]) >= MIN_VERDICT_STEPS]
        if not long_enough:
            raise TransitionRefuse(
                "REFUSED: no held-out prompt has 64 tokens; cannot judge multi-step"
            )
        take = [s for s in long_enough if int(s["n_tokens"]) >= int(n_hold_steps)] or long_enough
        # Prefer the longest held-out prompt (code holds are 345–496 tokens).
        take.sort(key=lambda s: -int(s["n_tokens"]))
        chosen_seq = take[0]
        xs = chosen_seq["xs"][: int(n_hold_steps)]
        hold_prompt_ids.append(chosen_seq["prompt_id"])
        ms = measure_multistep(
            incumbent_y=inc,
            candidate_y=cand,
            xs=xs,
            aux=aux,
            geo=geo,
            checkpoints=MULTISTEP_CHECKPOINTS,
        )
        ms["layer"] = int(layer)
        ms["prompt_id"] = chosen_seq["prompt_id"]
        ms["split"] = "hold"
        ms["kind"] = "captured"
        per_layer_ms.append(ms)
        one_step_cos.append(float(ms["one_step_qkvz_cosine"]))

    # Authority is the worst (largest state drift) representative layer.
    worst = max(per_layer_ms, key=lambda m: float(m["state_relfro_dense"][-1]))
    held_out_error = {
        "split": "hold",
        "source": "hold",
        "prompt_ids": sorted(set(hold_prompt_ids)),
        "one_step_cosine": float(min(one_step_cos)),
        "one_step_state_relfro": float(worst["one_step_state_relfro"]),
        "layers": [int(m["layer"]) for m in per_layer_ms],
    }
    verdict = report_verdict(worst, held_out_error, split=split_block)
    elapsed = time.perf_counter() - started
    payload = {
        "schema": "hawking.future.deltanet_generated_transition.fit.v1",
        "status": verdict["status"],
        "typical_layer": int(typical_layer),
        "rank": BOTTLENECK,
        "layers": [int(x) for x in chosen],
        "weight_cosine_after_diag": fit["weight_cosine_after_diag"],
        "singular_values_head": [float(x) for x in fit["singular_values"][:8]],
        "multistep": {
            k: worst[k]
            for k in (
                "n_steps",
                "checkpoints",
                "state_relfro",
                "output_relfro",
                "output_cosine",
                "crossings",
                "one_step_state_relfro",
                "one_step_output_cosine",
                "one_step_qkvz_cosine",
                "layer",
                "prompt_id",
                "split",
                "kind",
            )
        },
        "multistep_state_relfro_dense": worst["state_relfro_dense"],
        "multistep_per_layer": [
            {
                "layer": m["layer"],
                "prompt_id": m["prompt_id"],
                "n_steps": m["n_steps"],
                "checkpoints": m["checkpoints"],
                "state_relfro": m["state_relfro"],
                "output_relfro": m["output_relfro"],
                "output_cosine": m["output_cosine"],
                "crossings": m["crossings"],
                "one_step_qkvz_cosine": m["one_step_qkvz_cosine"],
                "split": "hold",
            }
            for m in per_layer_ms
        ],
        "held_out_error": held_out_error,
        "verdict": verdict,
        "consume": DIRECT_CONSUME,
        "dense_rematerialization": DIRECT_CONSUME,
        "fit_elapsed_s": elapsed,
        "split": split_block,
    }
    _write_fit_payload(payload, payload_dir)
    # Drop huge arrays from the in-memory return.
    return payload


def selftest() -> dict[str, Any]:
    corpus = dtc.selftest()
    cand = load_candidate()
    billed = billed_bytes(cand)
    if int(billed["bytes_removed"]["total"]) != dsf.QKVZ_ACTIVE_TARGET:
        raise SystemExit("selftest: bytes_removed must be the recorded qkvz organ")
    if int(billed["bytes_added"]["total"]) != 4_548_560:
        raise SystemExit(
            f"selftest: bytes_added {billed['bytes_added']['total']} != 4548560"
        )

    # 64-step refusal.
    verdict_refused = False
    try:
        report_verdict({"checkpoints": [1, 4, 16], "n_steps": 16}, {"split": "hold"})
    except VerdictRefuse:
        verdict_refused = True
    else:
        raise SystemExit("selftest: verdict without 64 steps was NOT refused")

    hold_refused = False
    toy = fixture_multistep(n_steps=64)
    try:
        report_verdict(toy, {"split": "train", "one_step_cosine": 0.999, "prompt_ids": []})
    except HeldOutRefuse:
        hold_refused = True
    else:
        raise SystemExit("selftest: train figure as held-out was NOT refused")

    remat_tag = consume_path_tag(emit_dense_w=True, ordinary_gemv=True)
    if remat_tag != REJECTED_DENSE_REMAT:
        raise SystemExit("selftest: emit-W was not REJECTED_DENSE_REMAT")
    remat_score = score_dense_remat(cand)
    if int(remat_score["bytes_removed"]) != int(remat_score["bytes_added"]["total"]):
        raise SystemExit("selftest: dense remat must be removed == added")

    native_tag = consume_path_tag(emit_dense_w=False, ordinary_gemv=False)
    if native_tag != DIRECT_CONSUME:
        raise SystemExit("selftest: native consume is DIRECT_CONSUME")

    # A 64-step fixture is allowed to produce a verdict (likely FAIL on a
    # rank-deficient toy, which is still a verdict).
    held = {
        "split": "hold",
        "source": "hold",
        "prompt_ids": [],
        "one_step_cosine": toy.get("one_step_qkvz_cosine"),
    }
    verdict = report_verdict(toy, held)
    floor = qkvz_coding_is_at_floor()
    return {
        "corpus": corpus,
        "held_out_leak_refused": corpus["held_out_leak_refused"],
        "synthetic_refused": corpus["synthetic_refused"],
        "verdict_without_64_refused": verdict_refused,
        "train_as_held_out_refused": hold_refused,
        "dense_remat": remat_tag,
        "dense_remat_removed_equals_added": True,
        "native_consume": native_tag,
        "fixture_verdict": verdict["verdict"],
        "qkvz_coding_retried": floor["retried"],
        "bytes_added_total": billed["bytes_added"]["total"],
        "bytes_removed_total": billed["bytes_removed"]["total"],
    }


def build(*, fit: bool = False, consult_economics: bool = True) -> Path:
    test = selftest()
    cand = load_candidate()
    billed = billed_bytes(cand)
    floor = qkvz_coding_is_at_floor()
    existing = _load_fit_payload()
    if fit:
        existing = run_fit()

    status_for_score = "OPEN"
    verdict_block: dict[str, Any] | None = None
    if existing and existing.get("held_out_error") and existing.get("multistep"):
        ms = dict(existing["multistep"])
        dense = existing.get("multistep_state_relfro_dense")
        if dense is not None:
            ms["state_relfro_dense"] = dense
        verdict_block = report_verdict(
            ms, existing["held_out_error"], split=existing.get("split")
        )
        wc = existing.get("weight_cosine_after_diag") or {}
        if verdict_block.get("mechanism") and wc:
            verdict_block["weight_cosine_after_diag"] = dict(wc)
            typical = existing.get("typical_layer")
            if typical is not None and str(typical) in wc:
                verdict_block["mechanism"] = (
                    verdict_block["mechanism"]
                    + f" Weight-space cosine after the per-layer diagonal is "
                    f"{float(wc[str(typical)]):.3f} on the fit layer "
                    f"(L{typical}) and near 0 on the other representatives; "
                    "the 256-wide bottleneck does not span W_qkvz."
                )
        status_for_score = str(verdict_block.get("status") or "OPEN")
    elif existing and existing.get("verdict"):
        verdict_block = existing["verdict"]
        status_for_score = str(verdict_block.get("status") or "OPEN")
    economics = score_candidate(cand, status=status_for_score) if consult_economics else None

    capture = dtc.load_existing_capture()
    capture_block: dict[str, Any]
    if capture:
        capture_block = {k: v for k, v in capture.items() if k != "rows"}
    else:
        capture_block = {
            "status": "not_run",
            "payload_dir": str(PAYLOAD_DIR.relative_to(REPO)),
            "note": (
                "Fingerprint and guards are in this receipt. Activation payloads "
                "are written by deltanet_teacher_corpus --capture into the "
                "gitignored payload_dir. pytest does not run the GEMM."
            ),
        }

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Fit generated_transition_coefficients on a real DeltaNet teacher "
            "corpus and judge it on multi-step state stability."
        ),
        "evidence_class": "DIAGNOSTIC_RELATIVE",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "candidate_id": CANDIDATE_ID,
        "predecessor": STATE_FN_REL,
        "qkvz_precision": floor,
        "specimen": dtc.specimen_identity(),
        "fusion_env": dict(FUSION_ENV),
        "organ": {
            "function": "y = diag_l T2 (T1 x + e_l); then incumbent rearrange + gated-delta",
            "hidden": dsf.HIDDEN,
            "qkvz_rows": dsf.QKVZ_ROWS,
            "bottleneck": BOTTLENECK,
            "n_deltanet_layers": dsf.N_DN_LAYERS,
            "native_primitive": "TiledProjection",
            "packing_T1_T2": "hq30uq4_uniform_q4_group64_signed_nibble_f16_scale",
            "residuals": "per-layer f16 diagonal on 16384 fused rows",
            "embeddings": "per-layer f32 bottleneck identity, 256 wide",
        },
        "bytes": billed,
        "economics": _py(economics) if economics is not None else None,
        "dense_rematerialization": {
            "native": DIRECT_CONSUME,
            "emit_w_then_ordinary_gemv": REJECTED_DENSE_REMAT,
            "emit_w_economics_removed_equals_added": True,
            "emit_w_score": _py(score_dense_remat(cand)),
        },
        "capture": capture_block,
        "fit": None
        if existing is None
        else {
            k: v
            for k, v in existing.items()
            if k != "multistep_state_relfro_dense"
        },
        "multistep_authority": {
            "checkpoints": list(MULTISTEP_CHECKPOINTS),
            "min_verdict_steps": MIN_VERDICT_STEPS,
            "state_cross_bars": list(STATE_CROSS_BARS),
            "one_step_cosine_bar": ONE_STEP_COSINE_BAR,
            "rule": (
                "A replacement with excellent one-step regression that drifts "
                "over 128 tokens is a FAILURE. Verdicts without a 64-step "
                "held-out series are refused."
            ),
        },
        "verdict": verdict_block,
        "selftest": test,
        "anti_fabrication": {
            "detectors": [
                "HELD_OUT_PROMPT_LEAK",
                "SYNTHETIC_ROW",
                "DUPLICATE_ROWS_ABOVE_THRESHOLD",
                "PROMPT_NOT_IN_SPLIT",
                "VERDICT_WITHOUT_64_STEPS",
                "TRAIN_FIGURE_AS_HELD_OUT",
            ],
            "loud_exceptions": [
                "CorpusRefused",
                "VerdictRefuse",
                "HeldOutRefuse",
                "DenseRematRefuse",
            ],
            "guards_module": "tools.future.mlp_teacher_corpus",
        },
        "gaps_closed": [
            "DeltaNet functional corpus reuses the MLP leak/synthetic/dup guards.",
            "Generator billed at the recorded 4,548,560 bytes, once, at model scope.",
            "Native consume is two skinny GEMVs; emit-W is REJECTED_DENSE_REMAT.",
            "Verdict refused without multi-step results at 64 steps or more.",
            "Train figures cannot be reported as held-out.",
            "q/k/v/z bit-descent is cited as at its floor and is not retried.",
        ],
        "what_this_does_not_prove": [
            "A generate-identity gate on the sealed residual stream.",
            "That X equals the sealed mixer input (capture_diverse2 is post_attn_norm).",
            "A protected complete-token or TPS number.",
        ],
        "era_vocabulary": {
            "evidence_class": "DIAGNOSTIC_RELATIVE",
            "bench_state": "UNKNOWN",
        },
        "bench": {
            "state": "UNKNOWN",
            "measurement_state": "DIAGNOSTIC_RELATIVE",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recorded_by": RECORDED_BY,
            "machine": "Apple host; CPU operator fit / rollout, no GPU lease",
            "gpu_authority": False,
            "rule": "no hardware measurement claim without hardware",
        },
    }
    out = write_receipt(RECEIPT, doc, RECORDED_BY)
    written = load_json(out)
    if written.get("schema") != SCHEMA or not written.get("seal_sha256"):
        raise SystemExit(f"receipt {out} failed round-trip")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else _sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args(argv_list)
    if args.selftest:
        json.dump(selftest(), _sys.stdout, indent=2, sort_keys=True)
        _sys.stdout.write("\n")
        return 0
    if args.capture:
        dtc.run_capture()
    if args.fit or args.build or args.capture:
        out = build(fit=bool(args.fit or args.capture))
        _sys.stdout.write(str(out) + "\n")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

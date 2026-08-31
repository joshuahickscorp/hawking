#!/usr/bin/env python3
"""AUX CAPABILITY SCREEN — the 1.07 GB auxiliary levers, on real weights.

Three OPEN byte levers sit on the 71-TPS ladder with exact byte models and
capability UNMEASURED:

    group_size_1024   aux(G) = 4*n_params/G + 58176
    group_size_256
    quantize_aux_u8

This module requantizes the auxiliary stream for real (LS refit of
scale/zero at the coarser group, or a real u8 encode of the incumbent
f16 aux), then measures damage cheapest-first:

    weight-space   relative error of reconstructed W per layer
    organ-space    held-out SwiGLU error on the teacher corpus, prompt-split
    logit-space    KL and top-k agreement through the real LM head
                   (argmax agreement is reported and is not parity)

A HETEROGENEOUS variant uses the capability map as an allocator: coarse G
only where the map marked a region supported, incumbent G=64 elsewhere,
and the extra variable-G metadata is billed through executable_economics.

    python3 tools/future/aux_capability_screen.py --build
    python3 -m pytest tools/future/test_aux_capability_screen.py -q

evidence_class STATIC_ONLY. No GPU lease. Does not touch crates/.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tools.future import executable_economics as ee
from tools.future._common import REPO, RECEIPTS, load_json, write_receipt
from tools.future.mlp_auxiliary_information import (
    AUXILIARY_BYTES_TARGET,
    F16_BYTES,
    HGRAVF_MAGIC,
    INCUMBENT_GROUP,
    SCALE_PLUS_BIAS_F16,
    _read_f16,
    _read_u8,
    _unpack_q,
    group_size_byte_curve,
    mlp_records,
    parse_hgrafv01_header,
)
from tools.future.mlp_byte_census import (
    CatalogAbsent,
    load_geometry,
    load_sealed,
    resolve_artifact_root,
)
from tools.future.mlp_teacher_corpus import (
    HIDDEN,
    INTERMEDIATE,
    N_LAYERS,
    PAYLOAD_DIR,
    _matmul,
    silu,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


RECEIPT = "AUX_CAPABILITY_SCREEN.json"
SCHEMA = "hawking.future.aux_capability_screen.v1"
VERSION = 1
RECORDED_BY = "tools/future/aux_capability_screen.py"
EVIDENCE_CLASS = "STATIC_ONLY"

MAP_REL = "receipts/future/CAPABILITY_INFORMATION_MAP.json"
AUX_REL = "receipts/future/MLP_AUXILIARY_INFORMATION.json"
CENSUS_REL = "receipts/future/MLP_BYTE_CENSUS.json"
CORPUS_REL = "receipts/future/MLP_TEACHER_CORPUS.json"
BUDGET_REL = "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json"
PATH71_REL = "receipts/future/PATH_TO_71.json"

N_PARAMS = 17_112_760_320
N_TENSORS = 192
HEADER_BYTES = 58_176
HEADER_PER_TENSOR = HEADER_BYTES // N_TENSORS  # 303
N_GROUPS_INCUMBENT = N_PARAMS // INCUMBENT_GROUP  # 267_386_880

LEVER_IDS: tuple[str, ...] = (
    "group_size_1024",
    "group_size_256",
    "quantize_aux_u8",
)
HETERO_ID = "group_size_1024_heterogeneous"

OVERLAP_VOCAB: tuple[str, ...] = (
    "INDEPENDENT",
    "OVERLAPPING",
    "SUBSUMED",
    "MUTUALLY_EXCLUSIVE",
    "INTERACTING",
    "UNKNOWN",
)

FITTED_HELDOUT = "FITTED_HELDOUT"
REFUTED = "REFUTED"
PROSPECTIVE_ECONOMIC = "PROSPECTIVE_ECONOMIC"

DIRECT_CONSUME = "DIRECT_CONSUME"

# Damage bars. Weight-space 0.05 is well below NNS-029's uniform-Q2 injury
# (~0.58) and above the already-measured u8-log scale rel-fro (~0.004).
# Organ cosine matches the capability map. KL is nats vs the incumbent
# softmax; top-5 agreement is the share of the incumbent top-5 that the
# candidate keeps. Argmax is recorded and is not a bar.
WEIGHT_RELFRO_BAR = 0.05
WEIGHT_RELFRO_EARLY_STOP = 0.20
ORGAN_COSINE_BAR = 0.990
# Rel-fro is reported; the consume-path bar is cosine (capability map).
# 0.20 is a cousin of the weight-space early-stop and sits below the
# uniform-G organ injury (~0.31) and above a 0.03-W aux change (~0.055).
ORGAN_RELFRO_BAR = 0.20
LOGIT_KL_BAR = 0.10
TOPK = 5
TOPK_AGREE_BAR = 0.80
N_ITERS_LS = 4
RNG_SEED = 38
LOGIT_HOLD_ROWS = 16
# Organ-space uses the full hold set unless it exceeds this cap; the cap
# is a time bound, not a substitute for the prompt-split.
ORGAN_HOLD_ROW_CAP = 512

# Variable-G metadata. Incumbent tensors encode one G in the JSON header
# (already counted in the 58,176). Mixed tensors need a run table the
# incumbent does not have.
VAR_G_GLOBAL_HEADER_BYTES = 4  # version u16 + n_mixed u16
VAR_G_TENSOR_ID_BYTES = 4  # layer u16 + organ u8 + n_runs u8
VAR_G_RUN_BYTES = 12  # row0 u32, row1 u32, G u16, pad u16

CHANNEL_RE = re.compile(
    r"^L(\d+)\.(mlp\.(?:gate|up|down))(?:\.(all|channel\.rows_(\d+)_(\d+)))?$"
)

CORPUS_CANDIDATES: tuple[Path, ...] = (
    PAYLOAD_DIR,
    Path("/Users/scammermike/Downloads/hawking/workspace/ops/local/scratch/mlp_teacher_corpus"),
    Path("/Users/scammermike/Downloads/hawking-copy/workspace/ops/local/scratch/mlp_teacher_corpus"),
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate gate. "
    "Auxiliary requant is a CPU LS refit of scale/zero (or a real u8 encode "
    "of the incumbent f16 aux) on sealed-3.14 HGRAVF01 tensors. Weight-space "
    "is reconstructed-W rel-fro vs the incumbent affine. Organ-space is "
    "held-out SwiGLU error on the real teacher corpus, split by prompt_id. "
    "Logit-space is KL and top-k agreement of the real LM head applied to "
    "last-layer organ outputs on held-out prompts; it is not a full-stack "
    "generate identity. Argmax agreement is recorded and is not parity. "
    "Byte figures come from tools/future/executable_economics.py. "
    "evidence_class is STATIC_ONLY. gpu_authority is false."
)


class ScreenRefuse(ValueError):
    """The screen refused rather than guessing."""


class ArgmaxAloneParityRefuse(ScreenRefuse):
    """Argmax agreement is not a capability screen."""

    def __init__(self, detail: str = "") -> None:
        extra = f" ({detail})" if detail else ""
        super().__init__(
            "REFUSED: argmax agreement is not parity; KL and top-k agreement "
            f"are required{extra}"
        )


class IncompleteScreen(ScreenRefuse):
    """A lever is missing a damage stage and has no early-stop reason."""


class CorpusAbsent(ScreenRefuse):
    """Teacher corpus is not readable; organ-space cannot be synthesised."""


# ---------------------------------------------------------------------------
# Tiny numeric helpers.
# ---------------------------------------------------------------------------


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


def relfro(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    num = float(np.sqrt(np.square(aa - bb).sum()))
    den = float(np.sqrt(np.square(bb).sum()))
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    den = math.sqrt(float(aa @ aa) * float(bb @ bb))
    if den == 0.0:
        return float("nan")
    return float(aa @ bb / den)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        z = z - float(z.max())
        e = np.exp(np.clip(z, -80.0, 80.0))
        s = float(e.sum())
        return e / max(s, 1e-30)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(np.clip(z, -80.0, 80.0))
    s = e.sum(axis=-1, keepdims=True)
    return e / np.maximum(s, 1e-30)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats. p is the incumbent."""
    pp = np.asarray(p, dtype=np.float64).ravel()
    qq = np.asarray(q, dtype=np.float64).ravel()
    pp = np.clip(pp, 1e-30, 1.0)
    qq = np.clip(qq, 1e-30, 1.0)
    pp = pp / pp.sum()
    qq = qq / qq.sum()
    return float(np.sum(pp * (np.log(pp) - np.log(qq))))


def topk_indices(logits: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    z = np.asarray(logits)
    if z.size <= k:
        return np.argsort(z)[::-1]
    return np.argpartition(z, -k)[-k:]


def topk_agreement(inc_logits: np.ndarray, cand_logits: np.ndarray, k: int) -> float:
    a = set(int(i) for i in topk_indices(inc_logits, k))
    b = set(int(i) for i in topk_indices(cand_logits, k))
    if not a:
        return 1.0
    return float(len(a & b) / len(a))


# ---------------------------------------------------------------------------
# Exact byte model. aux(G) = 4*n_params/G + 58176.
# ---------------------------------------------------------------------------


def aux_bytes_at_group(
    group_size: int,
    *,
    n_params: int = N_PARAMS,
    n_tensors: int = N_TENSORS,
    header_per_tensor: int = HEADER_PER_TENSOR,
) -> int:
    g = int(group_size)
    if g <= 0:
        raise ScreenRefuse(f"group_size must be positive, got {g}")
    if n_params % g:
        raise ScreenRefuse(f"group_size {g} does not divide n_params {n_params}")
    return (n_params // g) * SCALE_PLUS_BIAS_F16 + n_tensors * header_per_tensor


def bytes_eliminated_at_group(group_size: int) -> int:
    return AUXILIARY_BYTES_TARGET - aux_bytes_at_group(group_size)


def u8_aux_bytes(*, n_groups: int = N_GROUPS_INCUMBENT, n_tensors: int = N_TENSORS) -> dict[str, int]:
    """u8 scale + u8 bias + two f16 endpoints per array per tensor."""
    u8_payload = int(n_groups) * 2
    endpoints = int(n_tensors) * 8  # 2 arrays * 2 f16
    incumbent_scale_bias = int(n_groups) * SCALE_PLUS_BIAS_F16
    return {
        "u8_payload_bytes": u8_payload,
        "endpoint_bytes": endpoints,
        "candidate_aux_scale_bias_bytes": u8_payload + endpoints,
        "incumbent_scale_bias_bytes": incumbent_scale_bias,
        "bytes_removed": incumbent_scale_bias - u8_payload,
        "bytes_added_metadata": endpoints,
    }


def variable_group_metadata_bytes(*, n_mixed_tensors: int, n_runs: int) -> int:
    """Extra bytes a variable-G consumer must store that incumbent G=64 does not."""
    if n_mixed_tensors < 0 or n_runs < 0:
        raise ScreenRefuse("metadata counts cannot be negative")
    if n_mixed_tensors == 0:
        return 0
    return (
        VAR_G_GLOBAL_HEADER_BYTES
        + int(n_mixed_tensors) * VAR_G_TENSOR_ID_BYTES
        + int(n_runs) * VAR_G_RUN_BYTES
    )


def aux_bytes_heterogeneous(
    runs: Sequence[Mapping[str, Any]],
    *,
    n_tensors_total: int = N_TENSORS,
    header_per_tensor: int = HEADER_PER_TENSOR,
) -> dict[str, int]:
    """Scale+bias bytes for a per-run group-size allocation, plus metadata.

    Each run is {n_rows, n_cols, group_size} on one tensor. Tensors that
    are not named keep incumbent G=64 and are not in `runs`; the caller
    supplies only the mixed tensor's runs. Remaining tensors are billed
    at incumbent G.
    """
    mixed_scale_bias = 0
    mixed_incumbent = 0
    n_runs = 0
    tensors: set[tuple[int, str]] = set()
    for run in runs:
        n_rows = int(run["n_rows"])
        n_cols = int(run["n_cols"])
        g = int(run["group_size"])
        if n_cols % g:
            raise ScreenRefuse(f"group_size {g} does not divide n_cols {n_cols}")
        n_groups = n_rows * (n_cols // g)
        mixed_scale_bias += n_groups * SCALE_PLUS_BIAS_F16
        if n_cols % INCUMBENT_GROUP:
            raise ScreenRefuse(
                f"incumbent group {INCUMBENT_GROUP} does not divide n_cols {n_cols}"
            )
        mixed_incumbent += n_rows * (n_cols // INCUMBENT_GROUP) * SCALE_PLUS_BIAS_F16
        n_runs += 1
        tensors.add((int(run["layer"]), str(run["organ"])))
    n_mixed = len(tensors)
    metadata = variable_group_metadata_bytes(n_mixed_tensors=n_mixed, n_runs=n_runs)
    # Remaining tensors keep incumbent aux.
    remaining_incumbent = (
        N_GROUPS_INCUMBENT * SCALE_PLUS_BIAS_F16 - mixed_incumbent
    )
    candidate_scale_bias = remaining_incumbent + mixed_scale_bias
    headers = n_tensors_total * header_per_tensor
    incumbent_aux = AUXILIARY_BYTES_TARGET
    candidate_aux = candidate_scale_bias + headers + metadata
    return {
        "mixed_tensors": n_mixed,
        "n_runs": n_runs,
        "metadata_bytes": metadata,
        "mixed_scale_bias_bytes": mixed_scale_bias,
        "mixed_incumbent_scale_bias_bytes": mixed_incumbent,
        "candidate_scale_bias_bytes": candidate_scale_bias,
        "candidate_aux_bytes": candidate_aux,
        "incumbent_aux_bytes": incumbent_aux,
        "bytes_removed": mixed_incumbent - mixed_scale_bias,
        "bytes_added_metadata": metadata,
        "net_bytes_saved": incumbent_aux - candidate_aux,
    }


# ---------------------------------------------------------------------------
# Real requant. LS affine-Q2 at a new group size; u8 encode of f16 aux.
# ---------------------------------------------------------------------------


def _affine_q2_ls_from_init(
    grouped: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    degener: np.ndarray,
    *,
    n_iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign q, solve [q, 1] for (scale, bias), repeat. `grouped` is float64."""
    n = float(grouped.shape[-1])
    wmin = grouped.min(axis=-1, keepdims=True)
    scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-20)
    bias = np.asarray(bias, dtype=np.float64)
    for _ in range(int(n_iters)):
        q = np.clip(np.rint((grouped - bias) / scale), 0.0, 3.0)
        sum_q = q.sum(axis=-1, keepdims=True)
        sum_q2 = np.square(q).sum(axis=-1, keepdims=True)
        sum_w = grouped.sum(axis=-1, keepdims=True)
        sum_qw = (q * grouped).sum(axis=-1, keepdims=True)
        det = sum_q2 * n - sum_q * sum_q
        s_new = (n * sum_qw - sum_q * sum_w) / np.maximum(det, 1e-20)
        b_new = (sum_q2 * sum_w - sum_q * sum_qw) / np.maximum(det, 1e-20)
        bad = (det[..., 0] < 1e-12) | degener
        keep = bad[..., None]
        scale = np.where(keep, scale, s_new)
        bias = np.where(keep, bias, b_new)
        scale = np.maximum(scale, 1e-20)
    q = np.clip(np.rint((grouped - bias) / scale), 0.0, 3.0)
    if np.any(degener):
        q[degener] = 0.0
        scale[degener] = 1e-20
        bias[degener] = wmin[degener]
    return q, scale, bias


def refit_affine_q2(
    W: np.ndarray,
    group_size: int,
    *,
    n_iters: int = N_ITERS_LS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LS refit of unsigned affine-Q2 (q in {0,1,2,3}) at `group_size`.

    Not an analytic error model: each group of `group_size` weights is
    fitted by (assign q, solve [q, 1] for scale and bias). Two inits
    (min/max and moment-matched) are run; the lower reconstruction error
    wins per group. Returns (W_hat, scale[rows, gpr], bias[rows, gpr]).
    """
    W32 = np.asarray(W, dtype=np.float32)
    if W32.ndim != 2:
        raise ScreenRefuse(f"W must be rank-2, got {W32.shape}")
    rows, cols = W32.shape
    g = int(group_size)
    if g <= 0 or cols % g:
        raise ScreenRefuse(f"group_size {g} does not divide cols {cols}")
    gpr = cols // g
    grouped = W32.reshape(rows, gpr, g).astype(np.float64, copy=False)
    wmin = grouped.min(axis=-1, keepdims=True)
    wmax = grouped.max(axis=-1, keepdims=True)
    degener = (wmax - wmin)[..., 0] <= 1e-20
    # Init A: min/max of the group. Wrong when a group does not hit q=0 and q=3.
    scale_a = np.maximum((wmax - wmin) / 3.0, 1e-20)
    bias_a = wmin.copy()
    # Init B: moment match. unsigned q has mean 1.5 and variance 1.25.
    mu = grouped.mean(axis=-1, keepdims=True)
    sd = grouped.std(axis=-1, keepdims=True)
    scale_b = np.maximum(sd / math.sqrt(1.25), 1e-20)
    bias_b = mu - 1.5 * scale_b
    q_a, s_a, b_a = _affine_q2_ls_from_init(
        grouped, scale_a, bias_a, degener, n_iters=n_iters
    )
    q_b, s_b, b_b = _affine_q2_ls_from_init(
        grouped, scale_b, bias_b, degener, n_iters=n_iters
    )
    rec_a = q_a * s_a + b_a
    rec_b = q_b * s_b + b_b
    err_a = np.square(rec_a - grouped).sum(axis=-1)
    err_b = np.square(rec_b - grouped).sum(axis=-1)
    take_b = err_b < err_a
    q = np.where(take_b[..., None], q_b, q_a)
    scale = np.where(take_b[..., None], s_b, s_a)
    bias = np.where(take_b[..., None], b_b, b_a)
    what = (q * scale + bias).astype(np.float32).reshape(rows, cols)
    return what, scale.astype(np.float32)[..., 0], bias.astype(np.float32)[..., 0]


def u8_log_encode(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-12, None)
    ls = np.log(clipped)
    lmin = float(ls.min())
    lmax = float(ls.max())
    if lmax <= lmin:
        return np.zeros(values.shape, dtype=np.uint8), lmin, lmax
    q = np.clip(np.round((ls - lmin) / (lmax - lmin) * 255.0), 0, 255).astype(np.uint8)
    return q, lmin, lmax


def u8_log_decode(q: np.ndarray, lmin: float, lmax: float) -> np.ndarray:
    if lmax <= lmin:
        return np.full(np.asarray(q).shape, math.exp(lmin), dtype=np.float32)
    return np.exp(lmin + np.asarray(q, dtype=np.float64) * ((lmax - lmin) / 255.0)).astype(
        np.float32
    )


def u8_linear_encode(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    v = np.asarray(values, dtype=np.float64)
    vmin = float(v.min())
    vmax = float(v.max())
    if vmax <= vmin:
        return np.zeros(values.shape, dtype=np.uint8), vmin, vmax
    q = np.clip(np.round((v - vmin) / (vmax - vmin) * 255.0), 0, 255).astype(np.uint8)
    return q, vmin, vmax


def u8_linear_decode(q: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.full(np.asarray(q).shape, vmin, dtype=np.float32)
    return (vmin + np.asarray(q, dtype=np.float64) * ((vmax - vmin) / 255.0)).astype(
        np.float32
    )


def requant_aux_u8(
    q: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep 2-bit codes; replace f16 scale/bias with a real u8 encode.

    Scale is log-minmax (positive); bias is linear minmax. Endpoints are
    per tensor, matching the OPEN candidate in MLP_AUXILIARY_INFORMATION.
    """
    s_q, s_lo, s_hi = u8_log_encode(scale)
    b_q, b_lo, b_hi = u8_linear_encode(bias)
    s_hat = u8_log_decode(s_q, s_lo, s_hi).reshape(scale.shape)
    b_hat = u8_linear_decode(b_q, b_lo, b_hi).reshape(bias.shape)
    what = (q.astype(np.float32) * s_hat[..., None] + b_hat[..., None]).reshape(
        q.shape[0], q.shape[1] * q.shape[2]
    )
    meta = {
        "scale_lmin": float(s_lo),
        "scale_lmax": float(s_hi),
        "bias_min": float(b_lo),
        "bias_max": float(b_hi),
        "n_u8_scale": int(s_q.size),
        "n_u8_bias": int(b_q.size),
        "endpoint_bytes": 8,
    }
    return what, meta


def load_affine_q2(path: Path) -> dict[str, Any]:
    """Incumbent packed tensor: W, q, scale, bias, header."""
    parsed = parse_hgrafv01_header(Path(path))
    if parsed["group_size"] != INCUMBENT_GROUP:
        raise ScreenRefuse(
            f"{path}: group_size {parsed['group_size']} != incumbent {INCUMBENT_GROUP}"
        )
    rows, cols = int(parsed["shape"][0]), int(parsed["shape"][1])
    gpr = cols // INCUMBENT_GROUP
    n = int(parsed["groups"])
    off = int(parsed["payload_off"])
    scale = _read_f16(Path(path), off, n).astype(np.float32).reshape(rows, gpr)
    bias = _read_f16(Path(path), off + int(parsed["scale_bytes"]), n).astype(
        np.float32
    ).reshape(rows, gpr)
    codes = _read_u8(
        Path(path),
        off + int(parsed["scale_bytes"]) + int(parsed["bias_bytes"]),
        int(parsed["code_bytes"]),
    )
    q = _unpack_q(codes.reshape(n, 16)).reshape(rows, gpr, INCUMBENT_GROUP).astype(
        np.float32
    )
    W = (q * scale[:, :, None] + bias[:, :, None]).reshape(rows, cols)
    return {
        "W": W,
        "q": q,
        "scale": scale,
        "bias": bias,
        "shape": [rows, cols],
        "groups": n,
        "header_bytes": int(parsed["header_bytes"]),
        "path": str(path),
        "magic": HGRAVF_MAGIC.decode() if isinstance(HGRAVF_MAGIC, bytes) else str(HGRAVF_MAGIC),
    }


# ---------------------------------------------------------------------------
# Logit parity. Argmax alone is a loud refuse, not a quiet pass.
# ---------------------------------------------------------------------------


def report_logit_parity(
    *,
    kl_nats: float | None,
    top_k_agreement: float | None,
    argmax_agreement: float | None,
    k: int = TOPK,
    n_rows: int | None = None,
) -> dict[str, Any]:
    """KL + top-k are the screen. Argmax is a side report.

    Passing only argmax_agreement raises ArgmaxAloneParityRefuse. A
    candidate that keeps argmax while drifting in KL has not been screened.
    """
    if kl_nats is None or top_k_agreement is None:
        raise ArgmaxAloneParityRefuse(
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
    }


def mean_logit_parity(
    inc_logits: np.ndarray,
    cand_logits: np.ndarray,
    *,
    k: int = TOPK,
) -> dict[str, Any]:
    """Pairwise KL / top-k / argmax over a batch of logit vectors."""
    inc = np.asarray(inc_logits)
    cand = np.asarray(cand_logits)
    if inc.shape != cand.shape:
        raise ScreenRefuse(f"logit shapes {inc.shape} != {cand.shape}")
    if inc.ndim == 1:
        inc = inc[None, :]
        cand = cand[None, :]
    kls: list[float] = []
    tops: list[float] = []
    args: list[float] = []
    for i in range(inc.shape[0]):
        p = softmax(inc[i])
        q = softmax(cand[i])
        kls.append(kl_divergence(p, q))
        tops.append(topk_agreement(inc[i], cand[i], k))
        args.append(1.0 if int(np.argmax(inc[i])) == int(np.argmax(cand[i])) else 0.0)
    return report_logit_parity(
        kl_nats=float(sum(kls) / len(kls)),
        top_k_agreement=float(sum(tops) / len(tops)),
        argmax_agreement=float(sum(args) / len(args)),
        k=k,
        n_rows=int(inc.shape[0]),
    )


# ---------------------------------------------------------------------------
# Sensitivity map as an allocator.
# ---------------------------------------------------------------------------


def load_capability_map(path: Path | None = None) -> dict[str, Any]:
    rel = Path(path) if path is not None else REPO / MAP_REL
    if not rel.is_file():
        raise ScreenRefuse(f"REFUSED: {MAP_REL} is not on disk")
    doc = load_json(rel)
    alloc = doc.get("allocation")
    if not isinstance(alloc, Mapping):
        raise ScreenRefuse("CAPABILITY_INFORMATION_MAP.allocation is missing")
    return doc


def parse_mlp_region_id(region_id: str) -> dict[str, Any] | None:
    match = CHANNEL_RE.match(str(region_id))
    if match is None:
        return None
    layer = int(match.group(1))
    organ = str(match.group(2))
    rest = match.group(3) or "all"
    if rest == "all":
        return {
            "id": str(region_id),
            "layer": layer,
            "organ": organ,
            "kind": "all",
            "row0": None,
            "row1": None,
        }
    return {
        "id": str(region_id),
        "layer": layer,
        "organ": organ,
        "kind": "channel",
        "row0": int(match.group(4)),
        "row1": int(match.group(5)),
    }


def mlp_sensitivity_slices(alloc: Mapping[str, Any]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    """Per (layer, organ), the finest measured slices and whether they may coarsen.

    Channel rows beat `.all` when both exist. Unlisted tensors stay
    UNMEASURED and the allocator refuses to coarsen them.
    """
    could = set(str(x) for x in (alloc.get("could_take_fewer_bits") or []))
    must = set(str(x) for x in (alloc.get("must_keep_or_gain") or []))
    by: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for rid in list(could) + list(must):
        parsed = parse_mlp_region_id(rid)
        if parsed is None:
            continue
        key = (int(parsed["layer"]), str(parsed["organ"]))
        parsed["supported"] = rid in could
        parsed["must_keep"] = rid in must
        parsed["sensitivity"] = "COULD_TAKE" if rid in could else "MUST_KEEP"
        by.setdefault(key, []).append(parsed)
    out: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for key, rows in by.items():
        channels = [r for r in rows if r["kind"] == "channel"]
        chosen = channels if channels else [r for r in rows if r["kind"] == "all"]
        # Stable by row start.
        chosen.sort(key=lambda r: (r["row0"] is None, r["row0"] or 0, r["id"]))
        out[key] = chosen
    return out


def allocate_group_runs(
    *,
    layer: int,
    organ: str,
    n_rows: int,
    n_cols: int,
    coarse_group: int,
    slices: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Coarse G only on COULD_TAKE slices; incumbent G on MUST_KEEP / UNMEASURED."""
    key = (int(layer), str(organ))
    measured = list(slices.get(key) or [])
    if not measured:
        return [
            {
                "layer": int(layer),
                "organ": str(organ),
                "row0": 0,
                "row1": int(n_rows),
                "n_rows": int(n_rows),
                "n_cols": int(n_cols),
                "group_size": INCUMBENT_GROUP,
                "sensitivity": "UNMEASURED",
                "why": "capability map did not measure this tensor; refusing to coarsen",
            }
        ]
    runs: list[dict[str, Any]] = []
    covered = np.zeros(int(n_rows), dtype=np.bool_)
    for sl in measured:
        if sl["kind"] == "all":
            a, b = 0, int(n_rows)
        else:
            a, b = int(sl["row0"]), int(sl["row1"])
        g = int(coarse_group) if sl["sensitivity"] == "COULD_TAKE" else INCUMBENT_GROUP
        runs.append(
            {
                "layer": int(layer),
                "organ": str(organ),
                "row0": a,
                "row1": b,
                "n_rows": b - a,
                "n_cols": int(n_cols),
                "group_size": g,
                "sensitivity": sl["sensitivity"],
                "region_id": sl["id"],
                "why": (
                    "capability map supported a bit drop here"
                    if sl["sensitivity"] == "COULD_TAKE"
                    else "capability map marked this region must-keep"
                ),
            }
        )
        covered[a:b] = True
    if not bool(covered.all()):
        # Gaps stay incumbent. Concatenate contiguous gaps into runs.
        i = 0
        while i < n_rows:
            if covered[i]:
                i += 1
                continue
            j = i + 1
            while j < n_rows and not covered[j]:
                j += 1
            runs.append(
                {
                    "layer": int(layer),
                    "organ": str(organ),
                    "row0": int(i),
                    "row1": int(j),
                    "n_rows": int(j - i),
                    "n_cols": int(n_cols),
                    "group_size": INCUMBENT_GROUP,
                    "sensitivity": "UNMEASURED",
                    "why": "row range not in the capability map; incumbent G",
                }
            )
            i = j
    runs.sort(key=lambda r: (r["row0"], r["row1"]))
    return runs


def error_concentrates_in_sensitive(
    *,
    must_keep_relfro: Sequence[float],
    could_take_relfro: Sequence[float],
) -> dict[str, Any]:
    """A cheap mean can hide a spike on the must-keep slices."""
    mk = [float(x) for x in must_keep_relfro if x is not None]
    ct = [float(x) for x in could_take_relfro if x is not None]
    mk_mean = float(sum(mk) / len(mk)) if mk else None
    ct_mean = float(sum(ct) / len(ct)) if ct else None
    concentrates = False
    why = "no paired must-keep / could-take W-space measurements"
    if mk_mean is not None and ct_mean is not None:
        # Sensitive regions taking more of the injury than the quiet ones.
        concentrates = mk_mean > max(ct_mean * 1.5, 1e-6) and mk_mean >= WEIGHT_RELFRO_BAR / 5.0
        why = (
            f"must-keep mean rel-fro {mk_mean:.6g} vs could-take {ct_mean:.6g}; "
            + (
                "injury concentrates where the map says sensitivity is high"
                if concentrates
                else "injury is not concentrated on the high-sensitivity slices"
            )
        )
    elif mk_mean is not None and ct_mean is None:
        concentrates = mk_mean >= WEIGHT_RELFRO_BAR
        why = (
            f"only must-keep slices were measured (mean rel-fro {mk_mean:.6g}); "
            "no quiet-region licence exists on the MLP body"
        )
    return {
        "concentrates_in_sensitive": bool(concentrates),
        "must_keep_relfro_mean": mk_mean,
        "could_take_relfro_mean": ct_mean,
        "n_must_keep": len(mk),
        "n_could_take": len(ct),
        "why": why,
    }


# ---------------------------------------------------------------------------
# Overlap vocabulary.
# ---------------------------------------------------------------------------


def overlap_relations() -> list[dict[str, Any]]:
    """Every pair the ladder might be tempted to add. Not a byte claim."""
    return [
        {
            "a": "group_size_256",
            "b": "group_size_1024",
            "relation": "MUTUALLY_EXCLUSIVE",
            "why": (
                "Two points on the same aux(G)=4*n_params/G+58176 curve. "
                "A tensor has one group size; the saves are not additive."
            ),
        },
        {
            "a": "quantize_aux_u8",
            "b": "group_size_1024",
            "relation": "OVERLAPPING",
            "why": (
                "Both attack the same 1.07 GB of f16 scale/bias. PATH_TO_71 "
                "already records they must not be summed."
            ),
        },
        {
            "a": "quantize_aux_u8",
            "b": "group_size_256",
            "relation": "OVERLAPPING",
            "why": "Same 1.07 GB auxiliary as group_size_256.",
        },
        {
            "a": "quantize_aux_u8",
            "b": "group_size_1024",
            "relation": "INTERACTING",
            "why": (
                "u8-at-G=1024 is a different packing, not the sum of the two "
                "byte claims. Composing them without a new fit is forbidden."
            ),
        },
        {
            "a": HETERO_ID,
            "b": "group_size_1024",
            "relation": "SUBSUMED",
            "why": (
                "Heterogeneous G=1024 is a restricted allocation of the same "
                "lever: coarse G only on map-supported slices. Its bytes are "
                "a subset of uniform G=1024."
            ),
        },
        {
            "a": HETERO_ID,
            "b": "quantize_aux_u8",
            "relation": "OVERLAPPING",
            "why": "Heterogeneous G still rewrites the same scale/bias arrays u8 would quantize.",
        },
    ]


def assert_overlap_vocab(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        rel = str(row.get("relation") or "")
        if rel not in OVERLAP_VOCAB:
            raise ScreenRefuse(
                f"REFUSED: overlap relation {rel!r} is not in {OVERLAP_VOCAB}"
            )


# ---------------------------------------------------------------------------
# Economics wrapper. A ratio without bytes_added is not a candidate.
# ---------------------------------------------------------------------------


def score_lever(
    *,
    lever_id: str,
    bytes_removed: int,
    bytes_added: Mapping[str, int] | int,
    extra_flops_per_output_element: float = 0.0,
    reusable_family: bool = True,
    status: str = "OPEN",
) -> dict[str, Any]:
    scored = ee.score(
        bytes_removed=int(bytes_removed),
        bytes_added=bytes_added,
        extra_flops_per_output_element=float(extra_flops_per_output_element),
        consuming_primitive="FusedDecodeCompute",
        bandwidth_regime="affine_q2_family",
        organ="mlp",
        stream_class="broadcast_aux",
        reusable_family=bool(reusable_family),
        candidate_id=str(lever_id),
        status=str(status),
    )
    proposal = {
        "id": lever_id,
        "source": AUX_REL,
        "status": status,
        "byte_model_incomplete": False,
        "capability": None,
        "note": None,
        "name": lever_id,
    }
    return ee._compact(scored, proposal)


# ---------------------------------------------------------------------------
# Packed organ + corpus.
# ---------------------------------------------------------------------------


def organ_index(root: Path | None = None) -> dict[tuple[int, str], dict[str, Any]]:
    recs = mlp_records(root=root)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for rec in recs:
        out[(int(rec["layer"]), str(rec["organ"]))] = rec
    if len(out) != N_TENSORS:
        raise ScreenRefuse(f"expected {N_TENSORS} MLP tensors, got {len(out)}")
    return out


def resolve_corpus_dir() -> Path:
    for path in CORPUS_CANDIDATES:
        if (path / "CAPTURE.json").is_file() and (path / "L63_x.f32").is_file():
            return path
    raise CorpusAbsent(
        "REFUSED: mlp_teacher_corpus payload is not readable; "
        "organ-space will not synthesise X (NNS-001). looked at "
        + ", ".join(str(p) for p in CORPUS_CANDIDATES)
    )


def load_hold_xy(
    layer: int,
    corpus_dir: Path,
    *,
    max_rows: int | None = ORGAN_HOLD_ROW_CAP,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Held-out (X, Y) for one layer, prompt-split. Never mixes train prompts."""
    rows_path = corpus_dir / "rows.jsonl"
    if not rows_path.is_file():
        raise CorpusAbsent(f"REFUSED: missing {rows_path}")
    hold_idx: list[int] = []
    hold_prompts: list[str] = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if int(rec.get("layer", -1)) != int(layer):
            continue
        if str(rec.get("split") or "") != "hold":
            continue
        if rec.get("synthetic"):
            raise ScreenRefuse(
                f"REFUSED: synthetic hold row {rec.get('row_id')} (NNS-001)"
            )
        hold_idx.append(int(rec["x_row_index"]))
        hold_prompts.append(str(rec["prompt_id"]))
    if not hold_idx:
        raise CorpusAbsent(f"REFUSED: no hold rows for layer {layer}")
    x_path = corpus_dir / f"L{int(layer):02d}_x.f32"
    y_path = corpus_dir / f"L{int(layer):02d}_y.f32"
    if not x_path.is_file() or not y_path.is_file():
        raise CorpusAbsent(f"REFUSED: missing {x_path} or {y_path}")
    x_all = np.fromfile(x_path, dtype="<f4")
    y_all = np.fromfile(y_path, dtype="<f4")
    if x_all.size % HIDDEN or y_all.size % HIDDEN:
        raise ScreenRefuse(f"layer {layer} f32 payload is not a multiple of {HIDDEN}")
    x_all = x_all.reshape(-1, HIDDEN)
    y_all = y_all.reshape(-1, HIDDEN)
    idx = np.asarray(hold_idx, dtype=np.int64)
    if max_rows is not None and idx.size > int(max_rows):
        rng = rng if rng is not None else np.random.default_rng(RNG_SEED)
        # Prompt-stratified: pick whole prompts until the cap, never a
        # partial-prompt leak into a smaller hold.
        by_prompt: dict[str, list[int]] = {}
        for i, pid in enumerate(hold_prompts):
            by_prompt.setdefault(pid, []).append(int(idx[i]))
        pids = sorted(by_prompt)
        rng.shuffle(pids)
        chosen: list[int] = []
        used_prompts: list[str] = []
        for pid in pids:
            nxt = by_prompt[pid]
            if len(chosen) + len(nxt) > int(max_rows) and chosen:
                break
            chosen.extend(nxt)
            used_prompts.append(pid)
            if len(chosen) >= int(max_rows):
                break
        idx = np.asarray(chosen, dtype=np.int64)
        hold_prompts = used_prompts
    return {
        "layer": int(layer),
        "X": np.ascontiguousarray(x_all[idx], dtype=np.float32),
        "Y": np.ascontiguousarray(y_all[idx], dtype=np.float32),
        "n_hold_available": int(len(hold_idx)),
        "n_hold_used": int(idx.size),
        "n_hold_prompts": len(set(hold_prompts)),
        "prompt_ids": sorted(set(hold_prompts)),
        "split": "hold",
        "split_unit": "prompt_id",
    }


def swiglu_from_weights(
    x: np.ndarray,
    Wg: np.ndarray,
    Wu: np.ndarray,
    Wd: np.ndarray,
) -> np.ndarray:
    gate = _matmul(x, Wg.T)
    up = _matmul(x, Wu.T)
    hidden = silu(gate) * up
    del gate, up
    return _matmul(hidden, Wd.T)


def apply_lever_to_packed(
    packed: Mapping[str, Any],
    *,
    kind: str,
    group_size: int | None = None,
    runs: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct incumbent W, then a real requant. Returns (W_hat, fit_meta)."""
    W = packed["W"]
    if kind == "group_size":
        if group_size is None:
            raise ScreenRefuse("group_size lever needs group_size")
        what, scale, bias = refit_affine_q2(W, int(group_size))
        return what, {
            "kind": "group_size",
            "group_size": int(group_size),
            "n_iters": N_ITERS_LS,
            "n_groups": int(scale.size),
        }
    if kind == "u8":
        what, meta = requant_aux_u8(packed["q"], packed["scale"], packed["bias"])
        meta = dict(meta)
        meta["kind"] = "u8"
        return what, meta
    if kind == "heterogeneous":
        if not runs:
            raise ScreenRefuse("heterogeneous lever needs runs")
        what = np.array(W, copy=True)
        n_coarse = 0
        for run in runs:
            a, b = int(run["row0"]), int(run["row1"])
            g = int(run["group_size"])
            if g == INCUMBENT_GROUP:
                continue
            sl, _, _ = refit_affine_q2(W[a:b], g)
            what[a:b] = sl
            n_coarse += b - a
        return what, {
            "kind": "heterogeneous",
            "n_runs": len(list(runs)),
            "n_rows_coarsened": int(n_coarse),
            "n_rows": int(W.shape[0]),
        }
    raise ScreenRefuse(f"unknown lever kind {kind!r}")


# ---------------------------------------------------------------------------
# LM head batch (real Q4 table, no generate).
# ---------------------------------------------------------------------------


def lm_head_batch(X: np.ndarray, *, chunk_rows: int = 4096) -> np.ndarray:
    from tools.future.capability_information_map import (
        hq30uq4_meta,
        unpack_q4_rows,
        _tensor,
    )

    rec = _tensor(None, "lm_head")
    meta = hq30uq4_meta(Path(rec["segment_path"]))
    vocab = int(meta["shape"][0])
    Xc = np.ascontiguousarray(X, dtype=np.float32)
    if Xc.ndim == 1:
        Xc = Xc[None, :]
    n = int(Xc.shape[0])
    out = np.empty((n, vocab), dtype=np.float32)
    for r0 in range(0, vocab, int(chunk_rows)):
        r1 = min(vocab, r0 + int(chunk_rows))
        W = unpack_q4_rows(meta, r0, r1)
        out[:, r0:r1] = _matmul(Xc, W.T)
        del W
    return out


# ---------------------------------------------------------------------------
# Stage bars / completeness.
# ---------------------------------------------------------------------------


def weight_fails(relfro_mean: float) -> bool:
    return float(relfro_mean) > WEIGHT_RELFRO_BAR


def organ_fails(*, cosine_v: float, relfro_v: float) -> bool:
    return float(cosine_v) < ORGAN_COSINE_BAR or float(relfro_v) > ORGAN_RELFRO_BAR


def logit_fails(parity: Mapping[str, Any]) -> bool:
    return (
        float(parity["kl_nats"]) > LOGIT_KL_BAR
        or float(parity["top_k_agreement"]) < TOPK_AGREE_BAR
    )


def assert_complete_lever(row: Mapping[str, Any]) -> None:
    """Every lever carries three stages or an explicit early-stop reason."""
    cid = str(row.get("id") or "?")
    reason = row.get("early_stop_reason")
    for stage in ("weight_space", "organ_space", "logit_space"):
        block = row.get(stage)
        if block is None:
            if not reason:
                raise IncompleteScreen(
                    f"{cid} missing {stage} without early_stop_reason"
                )
            continue
        if isinstance(block, Mapping) and block.get("skipped"):
            if not (reason or block.get("reason")):
                raise IncompleteScreen(
                    f"{cid} skipped {stage} without a reason"
                )
            continue
        if stage == "logit_space" and isinstance(block, Mapping):
            if block.get("skipped"):
                continue
            # Load-bearing: a measured logit block must not be argmax-only.
            if "kl_nats" not in block or "top_k_agreement" not in block:
                raise ArgmaxAloneParityRefuse(f"{cid} logit_space")
            if block.get("argmax_is_not_parity") is not True:
                raise ArgmaxAloneParityRefuse(
                    f"{cid} logit_space did not flag argmax_is_not_parity"
                )


def evidence_tier_for(
    *,
    weight: Mapping[str, Any] | None,
    organ: Mapping[str, Any] | None,
    logit: Mapping[str, Any] | None,
    early: str | None,
) -> str:
    if early:
        return REFUTED
    stages = (weight, organ, logit)
    if any(s is None or (isinstance(s, Mapping) and s.get("skipped")) for s in stages):
        return REFUTED
    if weight and weight.get("failed"):
        return REFUTED
    if organ and organ.get("failed"):
        return REFUTED
    if logit and logit.get("failed"):
        return REFUTED
    return FITTED_HELDOUT


# ---------------------------------------------------------------------------
# Screen body.
# ---------------------------------------------------------------------------


def _identity(root: Path | None = None) -> dict[str, Any]:
    sealed = load_sealed()
    artifact = root if root is not None else resolve_artifact_root(sealed)
    geo = load_geometry(artifact)
    return {
        "resident_identity": sealed.get("resident_identity"),
        "artifact_root": str(artifact),
        "model_id": sealed.get("model_id"),
        "geometry": {
            "hidden_size": geo["hidden_size"],
            "intermediate_size": geo["intermediate_size"],
            "num_hidden_layers": geo["num_hidden_layers"],
        },
    }


def _layer_organs(
    index: Mapping[tuple[int, str], Mapping[str, Any]], layer: int
) -> dict[str, dict[str, Any]]:
    out = {}
    for organ in ("mlp.gate", "mlp.up", "mlp.down"):
        rec = index.get((int(layer), organ))
        if rec is None:
            raise ScreenRefuse(f"missing {organ} layer {layer}")
        out[organ] = dict(rec)
    return out


def _weight_space_one(
    packed: Mapping[str, Any],
    what: np.ndarray,
    fit: Mapping[str, Any],
    *,
    slices: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
    layer: int,
    organ: str,
) -> dict[str, Any]:
    W = packed["W"]
    total = relfro(what, W)
    mk: list[float] = []
    ct: list[float] = []
    per_slice: list[dict[str, Any]] = []
    for sl in slices.get((int(layer), str(organ))) or []:
        if sl["kind"] != "channel":
            continue
        a, b = int(sl["row0"]), int(sl["row1"])
        r = relfro(what[a:b], W[a:b])
        rec = {
            "region_id": sl["id"],
            "sensitivity": sl["sensitivity"],
            "relfro": float(r),
            "row0": a,
            "row1": b,
        }
        per_slice.append(rec)
        if sl["sensitivity"] == "MUST_KEEP":
            mk.append(r)
        elif sl["sensitivity"] == "COULD_TAKE":
            ct.append(r)
    conc = error_concentrates_in_sensitive(must_keep_relfro=mk, could_take_relfro=ct)
    return {
        "relfro": float(total),
        "failed": weight_fails(total),
        "fit": _py(fit),
        "slices": _py(per_slice),
        "sensitivity": _py(conc),
        "shape": [int(W.shape[0]), int(W.shape[1])],
    }


def _screen_layers() -> list[int]:
    """Corpus representatives, not layer 0 as 'the MLP'."""
    return [3, 31, 38, 63]


def run_screen(
    *,
    layers: Sequence[int] | None = None,
    organ_hold_cap: int | None = ORGAN_HOLD_ROW_CAP,
    logit_hold_rows: int = LOGIT_HOLD_ROWS,
    skip_organ: bool = False,
    skip_logit: bool = False,
) -> dict[str, Any]:
    identity = _identity()
    root = Path(identity["artifact_root"])
    index = organ_index(root)
    cmap = load_capability_map()
    alloc = cmap["allocation"]
    slices = mlp_sensitivity_slices(alloc)
    layers_u = [int(x) for x in (layers if layers is not None else _screen_layers())]
    rng = np.random.default_rng(RNG_SEED)

    corpus_dir: Path | None = None
    corpus_error: str | None = None
    try:
        corpus_dir = resolve_corpus_dir()
    except CorpusAbsent as exc:
        corpus_error = str(exc)

    # Heterogeneous runs for every tensor we will touch. Unmeasured → G=64.
    hetero_runs_by_tensor: dict[tuple[int, str], list[dict[str, Any]]] = {}
    hetero_all_runs: list[dict[str, Any]] = []
    for layer in layers_u:
        recs = _layer_organs(index, layer)
        for organ, rec in recs.items():
            path = Path(rec["segment_path"])
            header = parse_hgrafv01_header(path)
            n_rows, n_cols = int(header["shape"][0]), int(header["shape"][1])
            runs = allocate_group_runs(
                layer=layer,
                organ=organ,
                n_rows=n_rows,
                n_cols=n_cols,
                coarse_group=1024,
                slices=slices,
            )
            hetero_runs_by_tensor[(layer, organ)] = runs
            # Only mixed tensors (some run not G=64) enter the byte bill.
            if any(int(r["group_size"]) != INCUMBENT_GROUP for r in runs):
                hetero_all_runs.extend(runs)

    hetero_bytes = aux_bytes_heterogeneous(hetero_all_runs)

    u8_bytes = u8_aux_bytes()
    g256_removed = bytes_eliminated_at_group(256)
    g1024_removed = bytes_eliminated_at_group(1024)

    lever_specs: list[dict[str, Any]] = [
        {
            "id": "group_size_1024",
            "kind": "group_size",
            "group_size": 1024,
            "bytes_removed": g1024_removed,
            "bytes_added": 0,
            "reusable_family": True,
        },
        {
            "id": "group_size_256",
            "kind": "group_size",
            "group_size": 256,
            "bytes_removed": g256_removed,
            "bytes_added": 0,
            "reusable_family": True,
        },
        {
            "id": "quantize_aux_u8",
            "kind": "u8",
            "group_size": INCUMBENT_GROUP,
            "bytes_removed": u8_bytes["bytes_removed"],
            "bytes_added": {"metadata": u8_bytes["bytes_added_metadata"]},
            "reusable_family": True,
        },
        {
            "id": HETERO_ID,
            "kind": "heterogeneous",
            "group_size": 1024,
            "bytes_removed": hetero_bytes["bytes_removed"],
            "bytes_added": {"metadata": hetero_bytes["bytes_added_metadata"]},
            "reusable_family": True,
        },
    ]

    hold_cache: dict[int, dict[str, Any]] = {}

    def _hold(layer: int) -> dict[str, Any]:
        if layer not in hold_cache:
            if corpus_dir is None:
                raise CorpusAbsent(corpus_error or "corpus absent")
            hold_cache[layer] = load_hold_xy(
                layer, corpus_dir, max_rows=organ_hold_cap, rng=rng
            )
        return hold_cache[layer]

    # Per-lever accumulators. Layer-outer so each packed tensor is loaded
    # once, refit once per lever, then dropped.
    acc: dict[str, dict[str, Any]] = {
        str(spec["id"]): {
            "spec": spec,
            "weight_rows": [],
            "mk": [],
            "ct": [],
            "organ_per": [],
            "Y_hat_logit": None,
            "Y_inc_logit": None,
            "n_hold_available_63": None,
        }
        for spec in lever_specs
    }

    for layer in layers_u:
        print(f"  screening layer {layer}", flush=True)
        packed = {
            organ: load_affine_q2(Path(index[(int(layer), organ)]["segment_path"]))
            for organ in ("mlp.gate", "mlp.up", "mlp.down")
        }
        hold = None
        if corpus_dir is not None and not skip_organ:
            hold = _hold(layer)
        for spec in lever_specs:
            cid = str(spec["id"])
            kind = str(spec["kind"])
            gsz = spec.get("group_size")
            Ws: dict[str, np.ndarray] = {}
            per_organ: dict[str, Any] = {}
            for organ in ("mlp.gate", "mlp.up", "mlp.down"):
                runs = (
                    hetero_runs_by_tensor[(layer, organ)]
                    if kind == "heterogeneous"
                    else None
                )
                what, fit = apply_lever_to_packed(
                    packed[organ],
                    kind=kind,
                    group_size=None if kind == "u8" else int(gsz or INCUMBENT_GROUP),
                    runs=runs,
                )
                one = _weight_space_one(
                    packed[organ],
                    what,
                    fit,
                    slices=slices,
                    layer=layer,
                    organ=organ,
                )
                Ws[organ] = what
                per_organ[organ] = one
                for sl in one.get("slices") or []:
                    if sl.get("sensitivity") == "MUST_KEEP":
                        acc[cid]["mk"].append(float(sl["relfro"]))
                    elif sl.get("sensitivity") == "COULD_TAKE":
                        acc[cid]["ct"].append(float(sl["relfro"]))
            mean_l = float(
                sum(per_organ[o]["relfro"] for o in ("mlp.gate", "mlp.up", "mlp.down"))
                / 3.0
            )
            acc[cid]["weight_rows"].append(
                {
                    "layer": int(layer),
                    "relfro_mean": mean_l,
                    "relfro_gate": per_organ["mlp.gate"]["relfro"],
                    "relfro_up": per_organ["mlp.up"]["relfro"],
                    "relfro_down": per_organ["mlp.down"]["relfro"],
                    "failed": weight_fails(mean_l),
                    "organs": {
                        k: {
                            "relfro": v["relfro"],
                            "failed": v["failed"],
                            "slices": v["slices"],
                            "sensitivity": v["sensitivity"],
                            "fit": v["fit"],
                            "shape": v["shape"],
                        }
                        for k, v in per_organ.items()
                    },
                }
            )
            if hold is not None:
                Y_hat = swiglu_from_weights(
                    hold["X"], Ws["mlp.gate"], Ws["mlp.up"], Ws["mlp.down"]
                )
                rf = relfro(Y_hat, hold["Y"])
                co = cosine(Y_hat, hold["Y"])
                acc[cid]["organ_per"].append(
                    {
                        "layer": int(layer),
                        "relfro": float(rf),
                        "cosine": float(co),
                        "failed": organ_fails(cosine_v=co, relfro_v=rf),
                        "n_hold_used": int(hold["n_hold_used"]),
                        "n_hold_available": int(hold["n_hold_available"]),
                        "n_hold_prompts": int(hold["n_hold_prompts"]),
                        "split": "hold",
                        "split_unit": "prompt_id",
                    }
                )
                if int(layer) == 63:
                    n_take = min(int(logit_hold_rows), int(Y_hat.shape[0]))
                    acc[cid]["Y_hat_logit"] = np.array(Y_hat[:n_take], copy=True)
                    acc[cid]["Y_inc_logit"] = np.array(hold["Y"][:n_take], copy=True)
                    acc[cid]["n_hold_available_63"] = int(hold["n_hold_available"])
                del Y_hat
            del Ws
        del packed

    levers_out: list[dict[str, Any]] = []
    for spec in lever_specs:
        cid = str(spec["id"])
        kind = str(spec["kind"])
        bucket = acc[cid]
        weight_rows = list(bucket["weight_rows"])
        w_mean = float(
            sum(r["relfro_mean"] for r in weight_rows) / max(len(weight_rows), 1)
        )
        conc = error_concentrates_in_sensitive(
            must_keep_relfro=bucket["mk"], could_take_relfro=bucket["ct"]
        )
        weight_block = {
            "relfro_mean": w_mean,
            "bar": WEIGHT_RELFRO_BAR,
            "failed": weight_fails(w_mean),
            "early_stop_bar": WEIGHT_RELFRO_EARLY_STOP,
            "n_layers": len(weight_rows),
            "per_layer": weight_rows,
            "sensitivity": conc,
            "note": (
                "Rel-fro of reconstructed W vs incumbent affine-Q2 G=64. "
                "Group-size levers are a real LS refit at the new G, not an "
                "analytic bound. u8 keeps the 2-bit codes and requants aux."
            ),
        }
        early: str | None = None
        catastrophic = w_mean > WEIGHT_RELFRO_EARLY_STOP
        if catastrophic:
            early = (
                f"weight-space mean rel-fro {w_mean:.4f} exceeds early-stop bar "
                f"{WEIGHT_RELFRO_EARLY_STOP}; organ-space and logit-space still "
                "ran on already-built reconstructions"
            )

        if skip_organ:
            organ_block = {"skipped": True, "reason": "skip_organ requested"}
        elif corpus_dir is None:
            organ_block = {
                "skipped": True,
                "reason": corpus_error or "teacher corpus absent",
            }
            if not early:
                early = organ_block["reason"]
        else:
            per = list(bucket["organ_per"])
            cos_mean = float(sum(p["cosine"] for p in per) / max(len(per), 1))
            rf_mean = float(sum(p["relfro"] for p in per) / max(len(per), 1))
            organ_block = {
                "skipped": False,
                "relfro_mean": rf_mean,
                "cosine_mean": cos_mean,
                "cosine_bar": ORGAN_COSINE_BAR,
                "relfro_bar": ORGAN_RELFRO_BAR,
                "failed": organ_fails(cosine_v=cos_mean, relfro_v=rf_mean) if per else True,
                "per_layer": per,
                "corpus_dir": str(corpus_dir),
                "note": (
                    "Held-out by prompt_id on the real teacher corpus. Y is "
                    "incumbent F(X)=down(silu(gate(X))*up(X)); Y_hat is the "
                    "same organ after the auxiliary requant."
                ),
            }
        skip_this_logit = bool(skip_logit)
        # Y_hat for L63 is already in hand from organ-space; skipping
        # logit-space here would throw away a cheaper-than-already-paid
        # measurement. Early-stop only when those outputs were never built.
        if skip_this_logit:
            logit_block = {
                "skipped": True,
                "reason": early or "skip_logit requested",
            }
        elif corpus_dir is None or bucket["Y_hat_logit"] is None:
            logit_block = {
                "skipped": True,
                "reason": corpus_error or "no last-layer organ outputs for logit-space",
            }
            if not early:
                early = logit_block["reason"]
        else:
            try:
                parity = mean_logit_parity(
                    lm_head_batch(bucket["Y_inc_logit"]),
                    lm_head_batch(bucket["Y_hat_logit"]),
                    k=TOPK,
                )
                logit_block = {
                    "skipped": False,
                    "kind": "lm_head_on_last_layer_organ_output",
                    "not_full_stack_generate": True,
                    "failed": logit_fails(parity),
                    "kl_bar": LOGIT_KL_BAR,
                    "top_k_bar": TOPK_AGREE_BAR,
                    "layer": 63,
                    "split": "hold",
                    "split_unit": "prompt_id",
                    "n_hold_available": bucket["n_hold_available_63"],
                    **parity,
                    "note": (
                        "LM head is the sealed HQ30UQ4 table. Inputs are "
                        "last-layer organ outputs on prompt-split hold rows, "
                        "not the residual stream. Full-stack generate identity "
                        "is UNMEASURED (no GPU lease). Argmax is not parity."
                    ),
                }
            except (OSError, CatalogAbsent, ScreenRefuse, MemoryError) as exc:
                logit_block = {
                    "skipped": True,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                if not early:
                    early = logit_block["reason"]

        if (
            organ_block
            and not organ_block.get("skipped")
            and logit_block
            and not logit_block.get("skipped")
        ):
            # All three stages measured; the bars are the verdict.
            early = None

        economics = score_lever(
            lever_id=cid,
            bytes_removed=int(spec["bytes_removed"]),
            bytes_added=spec["bytes_added"],
            reusable_family=bool(spec["reusable_family"]),
            status="OPEN",
        )
        tier = evidence_tier_for(
            weight=weight_block, organ=organ_block, logit=logit_block, early=early
        )
        row = {
            "id": cid,
            "kind": kind,
            "evidence_tier": tier,
            "evidence_tier_was": PROSPECTIVE_ECONOMIC,
            "status": "OPEN" if tier == FITTED_HELDOUT else REFUTED,
            "capability": (
                "MEASURED_ON_HELDOUT" if tier == FITTED_HELDOUT else "FAILED_HELDOUT"
            ),
            "generate_gate": "UNMEASURED",
            "early_stop_reason": early,
            "weight_space": _py(weight_block),
            "organ_space": _py(organ_block),
            "logit_space": _py(logit_block),
            "economics": _py(economics),
            "bytes_removed": int(spec["bytes_removed"]),
            "bytes_added": _py(
                spec["bytes_added"]
                if isinstance(spec["bytes_added"], Mapping)
                else {
                    "generator": 0,
                    "embeddings": 0,
                    "residuals": 0,
                    "metadata": int(spec["bytes_added"]),
                    "state": 0,
                }
            ),
            "consuming_primitive": "FusedDecodeCompute",
            "dense_rematerialization": DIRECT_CONSUME,
            "sensitivity_concentration": _py(conc),
        }
        if kind == "heterogeneous":
            row["heterogeneous"] = _py(
                {
                    "coarse_group": 1024,
                    "incumbent_group": INCUMBENT_GROUP,
                    "allocator": "CAPABILITY_INFORMATION_MAP.allocation",
                    "unmeasured_keeps_incumbent": True,
                    "byte_model": hetero_bytes,
                    "runs": hetero_all_runs,
                    "note": (
                        "First use of the sensitivity map as an allocator: "
                        "G=1024 only on MLP slices the map marked supported, "
                        "G=64 on must-keep and on every unmeasured tensor. "
                        "Variable-G metadata is billed in bytes_added.metadata."
                    ),
                }
            )
        assert_complete_lever(row)
        levers_out.append(row)

    overlaps = overlap_relations()
    assert_overlap_vocab(overlaps)

    # Byte-model pins, so a closer cannot swap in a ratio.
    curve = group_size_byte_curve(
        n_parameters=N_PARAMS,
        n_tensors=N_TENSORS,
        header_bytes_per_tensor=HEADER_PER_TENSOR,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
    )
    by_g = {int(r["group_size"]): r for r in curve}

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "identity": identity,
        "purpose": (
            "Capability screen for the three OPEN auxiliary byte levers on "
            "the 1.07 GB of MLP scale/bias/header, plus a heterogeneous "
            "allocator that is the first use of the sensitivity map as a "
            "byte-placement rule rather than a byte claim."
        ),
        "what_this_does_not_prove": [
            "generate identity (no GPU lease, no resident generate gate)",
            "full-stack logits under a perturbation of every layer at once",
            "that a FITTED_HELDOUT lever is QUALIFIED on the resident",
            "physical EBPW of a different packing",
        ],
        "bars": {
            "weight_relfro": WEIGHT_RELFRO_BAR,
            "weight_relfro_early_stop": WEIGHT_RELFRO_EARLY_STOP,
            "organ_cosine": ORGAN_COSINE_BAR,
            "organ_relfro": ORGAN_RELFRO_BAR,
            "logit_kl_nats": LOGIT_KL_BAR,
            "top_k": TOPK,
            "top_k_agreement": TOPK_AGREE_BAR,
            "argmax_is_not_parity": True,
        },
        "byte_model": {
            "formula": "aux(G) = 4*n_params/G + 58176",
            "n_params": N_PARAMS,
            "n_tensors": N_TENSORS,
            "header_bytes": HEADER_BYTES,
            "incumbent_group": INCUMBENT_GROUP,
            "incumbent_aux_bytes": AUXILIARY_BYTES_TARGET,
            "group_size_256": {
                "aux_bytes": aux_bytes_at_group(256),
                "bytes_eliminated": g256_removed,
                "curve_row": by_g.get(256),
            },
            "group_size_1024": {
                "aux_bytes": aux_bytes_at_group(1024),
                "bytes_eliminated": g1024_removed,
                "curve_row": by_g.get(1024),
            },
            "quantize_aux_u8": u8_bytes,
            "heterogeneous": hetero_bytes,
            "source": AUX_REL,
            "scored_by": "tools/future/executable_economics.py",
        },
        "mlp_byte_census_where": {
            "source": CENSUS_REL,
            "mlp_active_bytes": 5_347_795_776,
            "auxiliary_bytes": AUXILIARY_BYTES_TARGET,
            "code_bytes": 4_278_190_080,
            "organs": ["mlp.gate", "mlp.up", "mlp.down"],
            "note": (
                "The 1.07 GB is the sum of 192 HGRAVF01 scale+bias+header "
                "parts on mlp.gate/up/down, reconciled by "
                "mlp_auxiliary_information.accounting."
            ),
        },
        "layers_screened": layers_u,
        "corpus": {
            "dir": None if corpus_dir is None else str(corpus_dir),
            "error": corpus_error,
            "split_unit": "prompt_id",
            "source": CORPUS_REL,
        },
        "overlap_relations": overlaps,
        "overlap_vocab": list(OVERLAP_VOCAB),
        "levers": levers_out,
        "n_fitted_heldout": sum(
            1 for r in levers_out if r["evidence_tier"] == FITTED_HELDOUT
        ),
        "n_refuted": sum(1 for r in levers_out if r["evidence_tier"] == REFUTED),
        "argmax_alone_is_not_parity": True,
        "sources": [AUX_REL, MAP_REL, CENSUS_REL, CORPUS_REL, BUDGET_REL, PATH71_REL],
    }


def build(**kwargs: Any) -> Path:
    doc = run_screen(**kwargs)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("This module")[0].strip())
    parser.add_argument("--build", action="store_true", help=f"write receipts/future/{RECEIPT}")
    parser.add_argument("--record", action="store_true", help="alias of --build")
    parser.add_argument(
        "--layers",
        type=str,
        default="",
        help="comma-separated layers (default: 3,31,38,63)",
    )
    parser.add_argument(
        "--organ-hold-cap",
        type=int,
        default=ORGAN_HOLD_ROW_CAP,
        help="max hold rows per layer for organ-space",
    )
    parser.add_argument(
        "--logit-hold-rows",
        type=int,
        default=LOGIT_HOLD_ROWS,
        help="hold rows for last-layer LM-head KL",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    layers = None
    if args.layers.strip():
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    kwargs = {
        "layers": layers,
        "organ_hold_cap": args.organ_hold_cap,
        "logit_hold_rows": args.logit_hold_rows,
    }
    if args.build or args.record:
        path = build(**kwargs)
        print(f"wrote {path}")
        doc = json.loads(path.read_text())
    else:
        doc = run_screen(**kwargs)
        print(json.dumps({k: doc[k] for k in ("n_fitted_heldout", "n_refuted") if k in doc}))
    for row in doc.get("levers") or []:
        eco = row.get("economics") or {}
        print(
            f"  {row['id']:32s}  {row['evidence_tier']:16s}  "
            f"W={((row.get('weight_space') or {}).get('relfro_mean'))}  "
            f"econ_ms_saved={eco.get('predicted_ms_saved')}  "
            f"{row.get('early_stop_reason') or ''}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

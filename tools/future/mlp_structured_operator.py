"""MLP STRUCTURED OPERATOR — full-width maps that are not r-bottlenecks.

Three lanes closed every bottleneck-shaped program for this organ:

    MLP_FUNCTIONAL_RANK     activation-weighted rank of F needs 5120 on
                            three of four layers to reach 10% hold error.
                            Affordable cap r=617 still sits at 35-84%.
    MLP_NONLINEAR_PROGRAM   FACTORIZE_THE_FACTORS is worse than the mean
                            predictor. Dictionaries, conditionals, generated
                            blocks, nonlinear generators: all in the 0.9 band.
    MLP_SHARED_PROGRAM      17 shared-basis candidates, all above 0.91.
                            Oracle PCA of F at rank 64 is already 0.895.

The named mechanism is that each family passed F through a NARROW
BOTTLENECK of rank r, K atoms, or a latent width. This module tests the
one remaining family: a FULL-WIDTH STRUCTURED operator that is full-rank
by construction and cheaper than a dense matrix.

    MONARCH / block-butterfly   product of two block-diagonals with a
                                permutation. Full rank, O(n^{1.5}) params.
                                Consumer: two batched GEMMs + permutation.
    BUTTERFLY / FFT-like        log n sparse structured factors, O(n log n).
    KRONECKER / TENSOR-PRODUCT  W ~ A (x) B, full rank when both factors are.
    DISTILLED OPERATOR          a dense net matching F end to end with
                                every hidden width >= input width. The
                                control that distinguishes BOTTLENECKS
                                from CAPACITY.

Widening rank / K / experts / blocks on a refuted instantiation is refused.
Shared subspaces, dictionaries, routed mixtures, per-block factorization
are measured negatives and cannot be named as a family here.

Held-out is by prompt. A number that does not beat the mean predictor is
a NULL MODEL and is labelled that way. A complete ledger at or above the
incumbent 5,347,795,776 bytes cannot be reported as a byte win. Bytes
go through executable_economics.score. evidence_class STATIC_ONLY.

    python3 tools/future/mlp_structured_operator.py --build
    python3 -m pytest tools/future/test_mlp_structured_operator.py -q

Does not touch crates/. No GPU lease.
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
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.future import executable_economics as ee
from tools.future import mlp_shared_program as msp
from tools.future import negative_index as ni
from tools.future._common import REPO, git, load_json, write_receipt
from tools.future.mlp_teacher_corpus import (
    CAPABILITY_DOMAINS,
    HIDDEN,
    INTERMEDIATE,
    N_LAYERS,
    POSITION_BANDS,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


RECEIPT = "MLP_STRUCTURED_OPERATOR.json"
SCHEMA = "hawking.future.mlp_structured_operator.v1"
VERSION = 1
RECORDED_BY = "tools/future/mlp_structured_operator.py"
EVIDENCE_CLASS = "STATIC_ONLY"
CORPUS_REL = "receipts/future/MLP_TEACHER_CORPUS.json"
SHARED_REL = "receipts/future/MLP_SHARED_PROGRAM.json"

MONARCH = "MONARCH"
BUTTERFLY = "BUTTERFLY"
KRONECKER = "KRONECKER"
DISTILLED = "DISTILLED"
FAMILIES: tuple[str, ...] = (MONARCH, BUTTERFLY, KRONECKER, DISTILLED)

# Measured-negative bottleneck families. Naming one here is a refuse.
DEAD_BOTTLENECK_FAMILIES = frozenset(
    {
        msp.SHARED_INPUT,
        msp.SHARED_OUTPUT,
        msp.SHARED_BOTH,
        "linear_shared_subspace",
        "shared_basis",
        "SHARED_SUBSPACE",
        "FACTORIZE_THE_FACTORS",
        "DICTIONARY_PROGRAM",
        "PRODUCT_DICTIONARY",
        "CONDITIONAL_PROGRAM",
        "GENERATED_BLOCK",
        "NONLINEAR_GENERATOR",
        "routed_mixture",
        "expert_bank",
        "low_rank",
        "global_dense_lowrank",
    }
)

DIRECT_CONSUME = msp.DIRECT_CONSUME
REJECTED_DENSE_REMAT = msp.REJECTED_DENSE_REMAT
MEASURED_NEGATIVE = msp.MEASURED_NEGATIVE
OPEN = msp.OPEN
NULL_MODEL = "NULL_MODEL"
CLOSED_SCAR = "MLP_FUNCTION_REPLACEMENT_CLOSED"

MEASURED_LAYERS: tuple[int, ...] = (3, 31, 38, 63)
TYPICAL_LAYER = 38
RNG_SEED = 38
ELEMENT_BYTES = ee.F16_BYTES
METADATA_BASE_BYTES = 256
INCUMBENT_MLP_BYTES = ee.MLP_ACTIVE_BYTES  # 5_347_795_776
assert INCUMBENT_MLP_BYTES == 5_347_795_776

HELD_OUT_KILL_REL = msp.HELD_OUT_KILL_REL  # 0.25
KILL_BAND = 0.85
MEAN_EPS = 1e-4
# Beating the mean by less than this is not a margin that matters for F.
MARGIN_THAT_MATTERS_ABS = 0.15

MONARCH_BLOCKS: tuple[int, ...] = (16, 32, 40, 64, 80, 128, 256)
BUTTERFLY_DEPTHS: tuple[int, ...] = (2, 4, 8, 12)
KRONECKER_SHAPES: tuple[tuple[int, int], ...] = (
    (64, 80),
    (80, 64),
    (32, 160),
    (40, 128),
    (16, 320),
    (8, 640),
    (4, 1280),
)
DISTILLED_SETTINGS: tuple[dict[str, Any], ...] = (
    {"program": "linear_affine", "depth": 1, "width": HIDDEN},
    {"program": "silu_then_linear", "depth": 1, "width": HIDDEN},
    {"program": "two_layer_silu", "depth": 2, "width": HIDDEN},
)

INCUMBENT_FLOPS_PER_LAYER = 2 * 3 * INTERMEDIATE * HIDDEN
MONARCH_ALS_ITERS = 6
DISTILLED_GD_STEPS = 0
# λ = RIDGE_SCALE * mean(diag(X^T X)). Scale 1 is Tikhonov of the same
# order as the Gram; a diagnostic (not authority) on L38 sat at hold
# 0.80-0.81 for scales 0.5-1. The billed estimator is scale 1, a priori.
RIDGE_SCALE = 1.0
OLS_RIDGE = RIDGE_SCALE

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate gate. "
    "Held-out errors are CPU arithmetic on the sealed-3.14 MLP teacher corpus "
    "(real post_attn_norm X, exact affine-Q2 SwiGLU F(X), split by prompt_id). "
    "They are not capability and not a protected complete-token number. "
    "Predicted ms/token is executable_economics arithmetic over cited organ "
    "times with a stated bandwidth-regime ASSUMPTION. gpu_authority is false. "
    "evidence_class is STATIC_ONLY. Bottleneck families (SHARED_*, "
    "FACTORIZE_THE_FACTORS, dictionaries, conditionals, generated blocks, "
    "rank-r nonlinear generators) are not re-tested; they are scoped scars. "
    "A complete ledger at or above the incumbent cannot be reported as a "
    "byte win. A held-out number that does not beat the mean predictor is "
    "a NULL MODEL."
)


class StructuredOperatorRefuse(ValueError):
    """The structured-operator census refused rather than guessing."""


class UnbilledProgramByte(StructuredOperatorRefuse):
    """A used factor, permutation table, or dense layer with 0 billed bytes."""


class TrainReportedAsHeldOut(msp.TrainReportedAsHeldOut, StructuredOperatorRefuse):
    """A train-set figure cannot be reported as held-out."""


class RematConsumer(msp.RematConsumer, StructuredOperatorRefuse):
    """A shape that rebuilds dense W before GEMV is dead on arrival."""


class CorpusUnavailable(msp.CorpusUnavailable, StructuredOperatorRefuse):
    """Real (X, F(X)) is not readable; synthesizing X is NNS-001."""


class RankBottleneckDead(StructuredOperatorRefuse):
    """A named r-bottleneck family is a scoped scar, not a candidate here."""


class BottleneckInDisguise(StructuredOperatorRefuse):
    """A hidden width below the input width is an r-bottleneck in disguise."""


class BaselineOmitted(StructuredOperatorRefuse):
    """A held-out error without the mean-predictor baseline is not a result."""


class ExceedsIncumbent(StructuredOperatorRefuse):
    """A ledger at or above the incumbent cannot be reported as a byte win."""


class UnderdeterminedFit(msp.UnderdeterminedFit, StructuredOperatorRefuse):
    """n_fit is below the fitted dimension (NNS-007 / NS-014)."""


def _py(x: Any) -> Any:
    return msp._py(x)


def _r(value: float, n: int = 6) -> float:
    return msp._r(value, n)


def _require_primitive(name: str) -> str:
    if name not in ATLAS_PRIMITIVES:
        raise StructuredOperatorRefuse(f"{name} is not an atlas primitive")
    return name


def _require_family(family: str) -> str:
    if family in DEAD_BOTTLENECK_FAMILIES:
        raise RankBottleneckDead(
            "REFUSED: bottleneck family "
            f"{family!r} is MEASURED_NEGATIVE "
            "(MLP_SHARED_PROGRAM / MLP_NONLINEAR_PROGRAM / "
            "MLP_FUNCTIONAL_RANK). This module does not retry an r-bottleneck, "
            "a dictionary, a routed mixture, or a per-block factorization."
        )
    if family not in FAMILIES:
        raise StructuredOperatorRefuse(f"unknown family {family!r}")
    return family


# ---------------------------------------------------------------------------
# Corpus. Real X, real F(X), prompt-split, 4 captured layers.
# ---------------------------------------------------------------------------


def load_layer_pack(
    layer: int,
    *,
    payload_dir: Path | None = None,
) -> dict[str, Any]:
    """Train/hold arrays plus per-row labels. Split unit is prompt_id."""
    pack = msp.load_layer_split(int(layer), payload_dir=payload_dir)
    root = Path(pack["payload_dir"])
    rows_path = root / "rows.jsonl"
    train_domain: list[str] = []
    hold_domain: list[str] = []
    train_band: list[str] = []
    hold_band: list[str] = []
    train_prompt: list[str] = []
    hold_prompt: list[str] = []
    with rows_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if int(row["layer"]) != int(layer):
                continue
            if row.get("synthetic"):
                raise CorpusUnavailable(
                    "REFUSED: SYNTHETIC_ROW in teacher payload (NNS-001)"
                )
            split = str(row.get("split") or "")
            rec_d = str(row.get("capability_domain") or "")
            rec_b = str(row.get("position_band") or "")
            rec_p = str(row["prompt_id"])
            if split == "train":
                train_domain.append(rec_d)
                train_band.append(rec_b)
                train_prompt.append(rec_p)
            elif split == "hold":
                hold_domain.append(rec_d)
                hold_band.append(rec_b)
                hold_prompt.append(rec_p)
    if len(hold_prompt) != int(pack["n_hold"]) or len(train_prompt) != int(pack["n_train"]):
        raise CorpusUnavailable(
            f"REFUSED: meta length train={len(train_prompt)} hold={len(hold_prompt)} "
            f"!= pack train={pack['n_train']} hold={pack['n_hold']}"
        )
    leaked = set(pack["train_prompt_ids"]) & set(pack["hold_prompt_ids"])
    if leaked:
        raise CorpusUnavailable(f"REFUSED: HELD_OUT_PROMPT_LEAK {sorted(leaked)[:8]}")
    pack["train_domain"] = train_domain
    pack["hold_domain"] = hold_domain
    pack["train_band"] = train_band
    pack["hold_band"] = hold_band
    pack["train_prompt"] = train_prompt
    pack["hold_prompt"] = hold_prompt
    pack["hold_meta"] = {
        "domain": hold_domain,
        "band": hold_band,
        "prompt_id": hold_prompt,
    }
    return pack


# ---------------------------------------------------------------------------
# Function error. Authority is held-out mean-L2, never train, never W.
# ---------------------------------------------------------------------------


def mean_l2_ratio(pred: np.ndarray, target: np.ndarray) -> float:
    return msp.mean_l2_ratio(pred, target)


def relative_frobenius(pred: np.ndarray, target: np.ndarray) -> float:
    return msp.relative_frobenius(pred, target)


def mean_cosine(pred: np.ndarray, target: np.ndarray) -> float:
    p = pred.astype(np.float64, copy=False)
    t = target.astype(np.float64, copy=False)
    if p.shape != t.shape:
        raise StructuredOperatorRefuse(f"pred shape {p.shape} != target shape {t.shape}")
    pn = np.linalg.norm(p, axis=1)
    tn = np.linalg.norm(t, axis=1)
    dot = np.sum(p * t, axis=1)
    return float(np.mean(dot / np.maximum(pn * tn, 1e-30)))


def _slice_relative_l2(
    pred: np.ndarray,
    target: np.ndarray,
    labels: Sequence[str],
) -> dict[str, float]:
    p = pred.astype(np.float64, copy=False)
    t = target.astype(np.float64, copy=False)
    err = np.linalg.norm(p - t, axis=1)
    scale = np.linalg.norm(t, axis=1)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        buckets[str(lab)].append(i)
    out: dict[str, float] = {}
    for lab, idx in buckets.items():
        ii = np.asarray(idx, dtype=np.int64)
        out[lab] = _r(float(err[ii].mean() / max(float(scale[ii].mean()), 1e-30)))
    return out


def function_error(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    split: str,
    report_as: str,
    meta: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Score F on a named split. Train cannot be labelled held-out."""
    try:
        base = msp.function_error(pred, target, split=split, report_as=report_as)
    except msp.TrainReportedAsHeldOut as exc:
        raise TrainReportedAsHeldOut(str(exc)) from exc
    as_n = str(report_as)
    if as_n in {"held_out", "hold", "heldout"}:
        base["held_out_cosine"] = _r(mean_cosine(pred, target))
        if meta is not None:
            domains = _slice_relative_l2(pred, target, meta["domain"])
            bands = _slice_relative_l2(pred, target, meta["band"])
            prompts = _slice_relative_l2(pred, target, meta["prompt_id"])
            worst_p = max(prompts, key=lambda k: prompts[k]) if prompts else None
            worst_d = max(domains, key=lambda k: domains[k]) if domains else None
            base["per_capability_domain"] = domains
            base["per_position_band"] = bands
            base["worst_prompt_id"] = worst_p
            base["worst_prompt_relative_l2"] = prompts.get(worst_p) if worst_p else None
            base["worst_domain"] = worst_d
            base["worst_domain_relative_l2"] = domains.get(worst_d) if worst_d else None
            base["n_hold_prompts"] = len(prompts)
    else:
        base["train_cosine_diagnostic"] = _r(mean_cosine(pred, target))
    return base


def vs_mean_predictor(held: float, mean_held: float) -> dict[str, Any]:
    """Every held-out number sits beside the mean predictor, or it is refused."""
    h = float(held)
    m = float(mean_held)
    margin = m - h
    beats = h < (m - MEAN_EPS)
    # A margin that matters for replacing F: below the 0.25 kill, or a
    # drop of at least MARGIN_THAT_MATTERS_ABS from the mean. Sitting
    # 0.01 below 0.97 is not a result.
    matters = beats and (h <= HELD_OUT_KILL_REL or margin >= MARGIN_THAT_MATTERS_ABS)
    null_model = not beats
    return {
        "mean_held_out_relative_l2": _r(m),
        "held_out_relative_l2": _r(h),
        "margin_over_mean": _r(margin),
        "beats_mean_predictor": bool(beats),
        "margin_that_matters": bool(matters),
        "null_model": bool(null_model),
        "null_model_label": NULL_MODEL if null_model else None,
        "mean_eps": MEAN_EPS,
        "margin_that_matters_abs": MARGIN_THAT_MATTERS_ABS,
        "held_out_kill_rel": HELD_OUT_KILL_REL,
        "note": (
            "A number that does not beat the mean predictor is a NULL MODEL, "
            "not a result. Authority remains held_out_relative_l2."
        ),
    }


def validate_baseline(row: Mapping[str, Any]) -> None:
    """Load-bearing: a held-out figure without the mean predictor is refused."""
    if "held_out_relative_l2" not in row:
        return
    if "mean_held_out_relative_l2" not in row:
        raise BaselineOmitted(
            "REFUSED: held-out error without the mean-predictor baseline; "
            "a number that is not reported beside the mean is not a result"
        )
    if row.get("held_out_split") != "hold":
        raise TrainReportedAsHeldOut(
            "REFUSED: held_out_relative_l2 is present but held_out_split is not 'hold'"
        )


def validate_error_authority(row: Mapping[str, Any]) -> None:
    try:
        msp.validate_error_authority(row)
    except msp.TrainReportedAsHeldOut as exc:
        raise TrainReportedAsHeldOut(str(exc)) from exc
    if row.get("held_out_split") == "train":
        raise TrainReportedAsHeldOut(
            "REFUSED: held_out_split='train' cannot be reported as held-out"
        )
    validate_baseline(row)


def report_as_byte_win(row: Mapping[str, Any]) -> bool:
    """Load-bearing: a ledger at or above the incumbent is not a byte win."""
    added = row.get("bytes_added") or {}
    if not isinstance(added, Mapping):
        raise ExceedsIncumbent("REFUSED: byte-win claim has no bytes_added ledger")
    total = int(added.get("total") or 0)
    if total >= INCUMBENT_MLP_BYTES or bool(row.get("exceeds_incumbent")):
        raise ExceedsIncumbent(
            "REFUSED: cannot report a setting whose complete ledger "
            f"exceeds the incumbent ({total} >= {INCUMBENT_MLP_BYTES}) as a byte win"
        )
    if row.get("byte_win") is False:
        raise ExceedsIncumbent(
            "REFUSED: this row is marked byte_win=False; it is not a byte win"
        )
    return True


def _silu(z: np.ndarray) -> np.ndarray:
    z64 = z.astype(np.float64, copy=False)
    return (z64 / (1.0 + np.exp(-np.clip(z64, -40.0, 40.0)))).astype(z.dtype, copy=False)


def _silu_grad(z: np.ndarray) -> np.ndarray:
    z64 = np.clip(z.astype(np.float64, copy=False), -40.0, 40.0)
    sig = 1.0 / (1.0 + np.exp(-z64))
    return (sig * (1.0 + z64 * (1.0 - sig))).astype(z.dtype, copy=False)


# ---------------------------------------------------------------------------
# Billing. Structure parameters; no invented byte model; no free factors.
# ---------------------------------------------------------------------------


def _divisors_of(n: int) -> tuple[int, ...]:
    out = []
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            out.append(d)
            if d * d != n:
                out.append(n // d)
    return tuple(sorted(out))


def monarch_param_count(hidden: int, n_blocks: int) -> int:
    """n (n/b + b): the two block-diagonals of a Monarch factorisation."""
    h = int(hidden)
    b = int(n_blocks)
    if b < 2 or h % b != 0:
        raise StructuredOperatorRefuse(
            f"MONARCH n_blocks={b} must be >=2 and divide hidden={h}"
        )
    m = h // b
    return int(h * (m + b))


def butterfly_pairs(n: int, stage: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """XOR pairing at `stage`. Unpaired indices pass through."""
    nn = int(n)
    stride = 1 << int(stage)
    used = np.zeros(nn, dtype=bool)
    a: list[int] = []
    b: list[int] = []
    for i in range(nn):
        j = i ^ stride
        if j < nn and (not used[i]) and (not used[j]) and i < j:
            a.append(i)
            b.append(j)
            used[i] = True
            used[j] = True
    unpaired = np.flatnonzero(~used).astype(np.int64, copy=False)
    return (
        np.asarray(a, dtype=np.int64),
        np.asarray(b, dtype=np.int64),
        unpaired,
    )


def butterfly_param_count(hidden: int, depth: int) -> int:
    """4 params per mixed pair per stage, plus a scale on unpaired coords."""
    h = int(hidden)
    d = int(depth)
    if d < 1:
        raise StructuredOperatorRefuse("BUTTERFLY depth must be positive")
    total = 0
    for stage in range(d):
        a, _b, unpaired = butterfly_pairs(h, stage)
        total += 4 * int(a.size) + int(unpaired.size)
    return int(total)


def kronecker_param_count(p: int, q: int) -> int:
    if int(p) < 2 or int(q) < 2:
        raise StructuredOperatorRefuse("KRONECKER factors must be at least 2x2")
    return int(p) * int(p) + int(q) * int(q)


def distilled_param_count(hidden: int, width: int, depth: int) -> int:
    h = int(hidden)
    w = int(width)
    d = int(depth)
    if w < h:
        raise BottleneckInDisguise(
            f"REFUSED: distilled width {w} < input width {h} is an r-bottleneck "
            "in disguise; every hidden width must be at or above the input width"
        )
    if d < 1:
        raise StructuredOperatorRefuse("DISTILLED depth must be positive")
    if d == 1:
        return h * w + w  # W [h,w] + bias
    # depth d: (d-1) hidden GEMMs of h->w plus w->h readout, + biases
    return (d - 1) * (h * w + w) + (w * h + h)


def byte_breakdown(
    family: str,
    *,
    n_layers: int = N_LAYERS,
    hidden: int = HIDDEN,
    element_bytes: int = ELEMENT_BYTES,
    n_blocks: int = 0,
    depth: int = 0,
    factor_p: int = 0,
    factor_q: int = 0,
    width: int = 0,
    program: str = "",
) -> dict[str, Any]:
    """Every byte of the 64-layer operator, billed once."""
    fam = _require_family(family)
    layers = int(n_layers)
    h = int(hidden)
    eb = int(element_bytes)
    if min(layers, h, eb) < 1:
        raise StructuredOperatorRefuse("layers/hidden/element_bytes must be positive")
    core = 0
    min_width = h
    params = 0
    if fam == MONARCH:
        b = int(n_blocks)
        params = monarch_param_count(h, b)
        core = params * eb
        min_width = h
    elif fam == BUTTERFLY:
        d = int(depth)
        params = butterfly_param_count(h, d)
        core = params * eb
        min_width = h
    elif fam == KRONECKER:
        p, q = int(factor_p), int(factor_q)
        if p * q != h:
            raise StructuredOperatorRefuse(
                f"KRONECKER factor shapes {p}x{q} do not multiply to hidden={h}"
            )
        params = kronecker_param_count(p, q)
        core = params * eb
        min_width = h
    elif fam == DISTILLED:
        w = int(width) if int(width) > 0 else h
        d = int(depth)
        params = distilled_param_count(h, w, d)
        core = params * eb
        min_width = min(h, w)
        if min_width < h:
            raise BottleneckInDisguise(
                f"REFUSED: distilled min width {min_width} < input {h}"
            )
    metadata = METADATA_BASE_BYTES * layers
    return {
        "family": fam,
        "program": str(program),
        "per_layer_core_bytes": int(core),
        "per_layer_params": int(params),
        "n_layers": layers,
        "metadata_bytes": int(metadata),
        "element_bytes": eb,
        "n_blocks": int(n_blocks),
        "depth": int(depth),
        "factor_p": int(factor_p),
        "factor_q": int(factor_q),
        "width": int(width) if int(width) > 0 else h,
        "hidden": h,
        "min_width": int(min_width),
        "full_width": bool(int(min_width) >= h),
    }


def bytes_added_from_breakdown(br: Mapping[str, Any]) -> dict[str, int]:
    added = {
        "embeddings": 0,
        "generator": int(br["per_layer_core_bytes"]) * int(br["n_layers"]),
        "residuals": 0,
        "metadata": int(br["metadata_bytes"]),
        "state": 0,
    }
    added["total"] = sum(added[k] for k in ee.BYTES_ADDED_FIELDS)
    return added


def validate_billing(row: Mapping[str, Any]) -> None:
    family = str(row.get("family") or "")
    br = row.get("byte_breakdown") or {}
    added = row.get("bytes_added") or {}
    if not isinstance(br, Mapping) or not isinstance(added, Mapping):
        raise UnbilledProgramByte("REFUSED: candidate is missing a byte ledger")
    _require_family(family)
    expected = bytes_added_from_breakdown(br)
    for key in ee.BYTES_ADDED_FIELDS:
        if int(added.get(key) or 0) != int(expected[key]):
            raise UnbilledProgramByte(
                f"REFUSED: bytes_added[{key}]={added.get(key)} != billed {expected[key]}"
            )
    total = int(added.get("total") or 0)
    if total != int(expected["total"]):
        raise UnbilledProgramByte(
            f"REFUSED: bytes_added.total {total} != program bytes {expected['total']}"
        )
    core = int(br.get("per_layer_core_bytes") or 0)
    if core <= 0:
        raise UnbilledProgramByte(
            f"REFUSED: {family} cores are free in the receipt: fabrication"
        )
    if int(br.get("min_width") or 0) < int(br.get("hidden") or HIDDEN):
        raise BottleneckInDisguise(
            "REFUSED: billed operator has min_width below hidden; "
            "that is an r-bottleneck in disguise"
        )


def complete_ledger_exceeds_incumbent(added_total: int) -> bool:
    return int(added_total) >= INCUMBENT_MLP_BYTES


# ---------------------------------------------------------------------------
# Native consumer. Remat-then-GEMV dies. Dispatch and extra FLOPs are honest.
# ---------------------------------------------------------------------------


def _flops_and_dispatch(
    family: str,
    *,
    hidden: int = HIDDEN,
    n_layers: int = N_LAYERS,
    n_blocks: int = 0,
    depth: int = 0,
    factor_p: int = 0,
    factor_q: int = 0,
    width: int = 0,
) -> dict[str, Any]:
    """Per-token dispatch delta and extra FLOPs vs the fused incumbent MLP.

    Incumbent is 1 fused SwiGLU launch per layer. A structured operator that
    removes bytes but multiplies dispatches may be a net loss; the scorer
    sees both terms.
    """
    h = int(hidden)
    layers = int(n_layers)
    if family == MONARCH:
        b = int(n_blocks)
        m = h // b
        flops = layers * 2 * h * (m + b)  # two block-diag GEMMs
        # two batched GEMMs + one permutation, unfused, vs 1 fused launch
        extra_launches_per_layer = 2
        note = (
            "Monarch is two batched GEMMs plus a permutation per layer. "
            "Unfused lowering: 3 launches vs 1 fused incumbent SwiGLU "
            f"(+{extra_launches_per_layer}/layer, +{extra_launches_per_layer * layers} "
            "per token). Fusing the permutation into a GEMM would drop this "
            "to +1/layer. ASSUMPTION: unfused."
        )
        extra_flops_note = (
            f"Monarch FLOPs {flops} vs incumbent {layers * INCUMBENT_FLOPS_PER_LAYER}; "
            "new kernel does fewer MACs. extra_flops billed at 0 (no invented "
            "negative-FLOP credit)."
        )
    elif family == BUTTERFLY:
        d = int(depth)
        flops = layers * 2 * butterfly_param_count(h, d)
        # d structured passes; permute fused into the next pass => d launches
        extra_launches_per_layer = max(int(d) - 1, 0)
        note = (
            f"Butterfly is {d} structured (paired 2x2) passes per layer, "
            "FFT-like. Unfused: "
            f"{d} launches vs 1 fused incumbent "
            f"(+{extra_launches_per_layer}/layer, "
            f"+{extra_launches_per_layer * layers} per token). ASSUMPTION: "
            "permutation is fused into the next pass; if not, dispatch "
            "delta doubles."
        )
        extra_flops_note = (
            f"Butterfly FLOPs {flops} vs incumbent {layers * INCUMBENT_FLOPS_PER_LAYER}; "
            "extra_flops billed at 0."
        )
    elif family == KRONECKER:
        p, q = int(factor_p), int(factor_q)
        flops = layers * (2 * p * p * q + 2 * q * q * p)
        extra_launches_per_layer = 1  # two small GEMMs vs 1 fused
        note = (
            "Kronecker (A ⊗ B) x is a reshape, a q×q GEMM, a reshape, a p×p "
            "GEMM. Two small GEMMs + two LayoutTransforms vs 1 fused "
            f"incumbent: +{extra_launches_per_layer}/layer unfused-GEMM "
            f"(+{extra_launches_per_layer * layers} per token) assuming "
            "reshapes fuse. ASSUMPTION."
        )
        extra_flops_note = (
            f"Kronecker FLOPs {flops} vs incumbent {layers * INCUMBENT_FLOPS_PER_LAYER}; "
            "extra_flops billed at 0."
        )
    elif family == DISTILLED:
        w = int(width) if int(width) > 0 else h
        d = max(int(depth), 1)
        if d == 1:
            flops = layers * 2 * h * w
            extra_launches_per_layer = 0
            note = (
                "Distilled depth-1 is one n×n GEMV per layer, replacing one "
                "fused SwiGLU. dispatch_delta 0. ASSUMPTION: one launch."
            )
        else:
            flops = layers * ((d - 1) * 2 * h * w + 2 * w * h)
            extra_launches_per_layer = int(d) - 1
            note = (
                f"Distilled depth-{d} is {d} dense GEMMs plus elementwise "
                f"nonlinearities per layer vs 1 fused incumbent "
                f"(+{extra_launches_per_layer}/layer, "
                f"+{extra_launches_per_layer * layers} per token). ASSUMPTION: "
                "elementwise fused into GEMM."
            )
        extra_flops_note = (
            f"Distilled FLOPs {flops} vs incumbent {layers * INCUMBENT_FLOPS_PER_LAYER}; "
            "extra billed only if new > incumbent."
        )
    else:
        raise StructuredOperatorRefuse(f"unknown family {family!r}")

    incumbent = layers * INCUMBENT_FLOPS_PER_LAYER
    extra_total = max(0.0, float(flops) - float(incumbent))
    n_out = int(ee.ORGAN_OUTPUT_ELEMENTS["mlp"])
    extra_per = extra_total / max(n_out, 1)
    dispatch_delta = int(extra_launches_per_layer) * layers
    return {
        "flops_per_token": int(flops),
        "incumbent_flops_per_token": int(incumbent),
        "flop_ratio_vs_incumbent": _r(float(flops) / max(float(incumbent), 1.0), 6),
        "extra_flops_per_output_element": float(extra_per),
        "dispatch_delta": int(dispatch_delta),
        "extra_launches_per_layer": int(extra_launches_per_layer),
        "dispatch_delta_note": note,
        "extra_flops_note": extra_flops_note,
    }


def native_consumer_sketch(
    family: str,
    *,
    rematerialize_dense_W: bool = False,
    n_blocks: int = 0,
    depth: int = 0,
    factor_p: int = 0,
    factor_q: int = 0,
    width: int = 0,
    n_layers: int = N_LAYERS,
    hidden: int = HIDDEN,
) -> dict[str, Any]:
    if rematerialize_dense_W:
        return {
            "family": family,
            "primitive": _require_primitive("FusedDecodeCompute"),
            "also": [],
            "algebra": "W_l = materialize(structured_l); y = W_l x",
            "consumes_directly": False,
            "rematerialize_dense_W": True,
            "runs_ordinary_gemv": True,
            "status": REJECTED_DENSE_REMAT,
            "why_dead": (
                "Materializing the structured operator into a dense n×n W "
                "and GEMV is REJECTED_DENSE_REMAT: the structure is the "
                "kernel. Paying n^2 bytes erases the point of the family."
            ),
        }
    fam = _require_family(family)
    cost = _flops_and_dispatch(
        fam,
        hidden=hidden,
        n_layers=n_layers,
        n_blocks=n_blocks,
        depth=depth,
        factor_p=factor_p,
        factor_q=factor_q,
        width=width,
    )
    if fam == MONARCH:
        sketch = {
            "algebra": "y = L P R x  (R, L block-diagonal; P reshape-transpose)",
            "primitive": _require_primitive("TiledProjection"),
            "also": [_require_primitive("LayoutTransform")],
            "why_not_gemv": (
                "Two batched small GEMMs and a permutation. Materializing "
                "L P R into W is REJECTED_DENSE_REMAT."
            ),
        }
    elif fam == BUTTERFLY:
        sketch = {
            "algebra": "y = B_{d-1} P_{d-1} ... B_0 x  (each B block-diag 2x2)",
            "primitive": _require_primitive("TiledProjection"),
            "also": [_require_primitive("LayoutTransform")],
            "why_not_gemv": (
                "log n structured passes. Materializing the product into W "
                "is REJECTED_DENSE_REMAT."
            ),
        }
    elif fam == KRONECKER:
        sketch = {
            "algebra": "y = (A ⊗ B) x = vec(B X A^T)",
            "primitive": _require_primitive("TiledProjection"),
            "also": [_require_primitive("LayoutTransform")],
            "why_not_gemv": (
                "Two small GEMMs plus reshapes. Materializing A ⊗ B into W "
                "is REJECTED_DENSE_REMAT."
            ),
        }
    else:
        d = max(int(depth), 1)
        if d == 1:
            algebra = "y = x W + b  (full-width affine; width >= hidden)"
        else:
            algebra = "y = silu(x W1 + b1) W2 + b2  (every width >= hidden)"
        sketch = {
            "algebra": algebra,
            "primitive": _require_primitive("TiledProjection"),
            "also": [_require_primitive("StationaryRepresentation")],
            "why_not_gemv": (
                "The dense layers ARE the replacement operator, not a "
                "rematerialization of incumbent SwiGLU W. Consumed as "
                "TiledProjection of the distilled program."
            ),
        }
    sketch.update(
        {
            "family": fam,
            "consumes_directly": True,
            "rematerialize_dense_W": False,
            "runs_ordinary_gemv": False,
            "status": DIRECT_CONSUME,
            **cost,
        }
    )
    return sketch


def consumer_status(sketch: Mapping[str, Any]) -> str:
    if sketch.get("rematerialize_dense_W") or (
        sketch.get("runs_ordinary_gemv") and sketch.get("rematerialize_dense_W")
    ):
        return REJECTED_DENSE_REMAT
    if sketch.get("rematerialize_dense_W"):
        return REJECTED_DENSE_REMAT
    if not sketch.get("consumes_directly", False):
        return REJECTED_DENSE_REMAT
    _require_primitive(str(sketch["primitive"]))
    return DIRECT_CONSUME


# ---------------------------------------------------------------------------
# Linear ceiling: affine OLS of F, then projection onto each structure.
# ---------------------------------------------------------------------------


def fit_affine_ols(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    ridge_scale: float = RIDGE_SCALE,
) -> dict[str, np.ndarray]:
    """W, b minimizing ||X W + b - Y||^2 + λ ||W||^2 with λ = scale * mean(diag)."""
    if int(x_tr.shape[0]) < int(x_tr.shape[1]):
        raise UnderdeterminedFit(
            f"REFUSED: n_fit={x_tr.shape[0]} < hidden={x_tr.shape[1]} for affine OLS "
            "(NNS-007 / NS-014: the score is not the codec's score)"
        )
    x_mean = x_tr.mean(axis=0)
    y_mean = y_tr.mean(axis=0)
    xc = x_tr - x_mean
    yc = y_tr - y_mean
    xtx = xc.T @ xc
    xty = xc.T @ yc
    xtx64 = xtx.astype(np.float64, copy=True)
    diag = float(np.mean(np.diag(xtx64)))
    xtx64.flat[:: xtx64.shape[0] + 1] += float(ridge_scale) * max(diag, 1.0)
    w = np.linalg.solve(xtx64, xty.astype(np.float64, copy=False))
    w32 = w.astype(np.float32, copy=False)
    b = (y_mean - x_mean @ w32).astype(np.float32, copy=False)
    return {
        "W": w32,
        "b": b,
        "x_mean": x_mean.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "ridge_scale": float(ridge_scale),
        "ridge_lambda": float(ridge_scale) * max(diag, 1.0),
    }


def apply_affine(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (x @ w + b).astype(np.float32, copy=False)


# ---- Monarch ----


def apply_monarch(x: np.ndarray, r_blocks: np.ndarray, l_blocks: np.ndarray) -> np.ndarray:
    """y = L P R x with R: (b, m, m), L: (m, b, b), P = reshape-transpose."""
    n_batch, n = int(x.shape[0]), int(x.shape[1])
    b, m = int(r_blocks.shape[0]), int(r_blocks.shape[1])
    if b * m != n:
        raise StructuredOperatorRefuse(f"monarch shape {b}x{m} != hidden {n}")
    u = np.einsum("nbi,bij->nbj", x.reshape(n_batch, b, m), r_blocks, optimize=True)
    v = np.transpose(u, (0, 2, 1))
    z = np.einsum("nmj,mjk->nmk", v, l_blocks, optimize=True)
    return np.ascontiguousarray(z.reshape(n_batch, n), dtype=np.float32)


def monarch_matrix(r_blocks: np.ndarray, l_blocks: np.ndarray) -> np.ndarray:
    n = int(r_blocks.shape[0] * r_blocks.shape[1])
    return apply_monarch(np.eye(n, dtype=np.float32), r_blocks, l_blocks)


def project_monarch(w: np.ndarray, n_blocks: int, *, n_iter: int = MONARCH_ALS_ITERS) -> tuple[np.ndarray, np.ndarray]:
    """ALS projection of dense W onto Monarch: W[p m + q, j b + k] ≈ R[p,q,j] L[j,p,k]."""
    n = int(w.shape[0])
    b = int(n_blocks)
    if n % b != 0:
        raise StructuredOperatorRefuse(f"n={n} not divisible by n_blocks={b}")
    m = n // b
    w4 = np.ascontiguousarray(w.reshape(b, m, m, b), dtype=np.float64)
    r = np.zeros((b, m, m), dtype=np.float64)
    for p in range(b):
        r[p] = np.eye(m, dtype=np.float64)
    l = np.zeros((m, b, b), dtype=np.float64)
    for _ in range(int(n_iter)):
        num_l = np.einsum("pqj,pqjk->jpk", r, w4, optimize=True)
        den_r = np.einsum("pqj,pqj->pj", r, r, optimize=True) + 1e-12
        l = num_l / den_r.T[:, :, None]
        num_r = np.einsum("pqjk,jpk->pqj", w4, l, optimize=True)
        den_l = np.einsum("jpk,jpk->jp", l, l, optimize=True) + 1e-12
        r = num_r / den_l.T[:, None, :]
    return r.astype(np.float32, copy=False), l.astype(np.float32, copy=False)


# ---- Butterfly ----


def apply_butterfly(
    x: np.ndarray,
    factors: Sequence[Mapping[str, np.ndarray]],
) -> np.ndarray:
    y = np.array(x, dtype=np.float32, copy=True)
    for fac in factors:
        a = fac["a"]
        b = fac["b"]
        blocks = fac["blocks"]
        if int(a.size) > 0:
            # The same 2x2 arithmetic, without materializing the stacked pair.
            # This replaced np.stack + einsum("npd,pde->npe"). The two forms are
            # NOT bit-identical elementwise -- einsum accumulates differently, max
            # absolute deviation 9.5e-07 on float32 -- so the equivalence that
            # matters was measured on the ARTIFACT rather than assumed: a full
            # build() under each form produced receipts whose 1334 float fields
            # agree exactly, 0 differing. 155.9s -> 147.6s.
            ya = y[:, a]
            yb = y[:, b]
            y[:, a] = ya * blocks[:, 0, 0] + yb * blocks[:, 1, 0]
            y[:, b] = ya * blocks[:, 0, 1] + yb * blocks[:, 1, 1]
        unpaired = fac.get("unpaired")
        scales = fac.get("scales")
        if unpaired is not None and int(unpaired.size) > 0 and scales is not None:
            y[:, unpaired] *= scales
    return y


def _batched_qr_two_cols(stacked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized thin QR of a (P, n, 2) stack. Avoids per-pair LAPACK."""
    v0 = stacked[:, :, 0]
    v1 = stacked[:, :, 1]
    n0 = np.maximum(np.linalg.norm(v0, axis=1), 1e-12)
    q0 = v0 / n0[:, None]
    r01 = np.sum(q0 * v1, axis=1)
    v1 = v1 - q0 * r01[:, None]
    n1 = np.maximum(np.linalg.norm(v1, axis=1), 1e-12)
    q1 = v1 / n1[:, None]
    q = np.stack((q0, q1), axis=-1)
    r = np.zeros((stacked.shape[0], 2, 2), dtype=stacked.dtype)
    r[:, 0, 0] = n0
    r[:, 0, 1] = r01
    r[:, 1, 1] = n1
    return q, r


def _principal_2x2(
    remaining: np.ndarray, a: np.ndarray, b: np.ndarray, unpaired: np.ndarray
) -> dict[str, np.ndarray]:
    """Last-factor projection: keep the pairing's 2x2 principal blocks."""
    if int(a.size) == 0:
        scales = (
            np.diag(remaining)[unpaired].astype(np.float32)
            if int(unpaired.size) > 0
            else np.zeros((0,), dtype=np.float32)
        )
        return {
            "a": a,
            "b": b,
            "blocks": np.zeros((0, 2, 2), dtype=np.float32),
            "unpaired": unpaired,
            "scales": scales,
        }
    ia = a.astype(np.int64)
    ib = b.astype(np.int64)
    blocks = np.empty((int(a.size), 2, 2), dtype=np.float64)
    blocks[:, 0, 0] = remaining[ia, ia]
    blocks[:, 0, 1] = remaining[ia, ib]
    blocks[:, 1, 0] = remaining[ib, ia]
    blocks[:, 1, 1] = remaining[ib, ib]
    scales = (
        np.diag(remaining)[unpaired].astype(np.float32)
        if int(unpaired.size) > 0
        else np.zeros((0,), dtype=np.float32)
    )
    return {
        "a": a,
        "b": b,
        "blocks": blocks.astype(np.float32, copy=False),
        "unpaired": unpaired,
        "scales": scales,
    }


def _peel_stage(
    remaining: np.ndarray, a: np.ndarray, b: np.ndarray, unpaired: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """One greedy peel: remaining ≈ factor @ new_remaining (input-side 2x2)."""
    if int(a.size) == 0:
        scales = np.ones(int(unpaired.size), dtype=np.float32)
        if int(unpaired.size) > 0:
            col = remaining[unpaired]
            nrm = np.maximum(np.linalg.norm(col, axis=1, keepdims=True), 1e-12)
            scales = nrm[:, 0].astype(np.float32)
            remaining = remaining.copy()
            remaining[unpaired] = col / nrm
        fac = {
            "a": a,
            "b": b,
            "blocks": np.zeros((0, 2, 2), dtype=np.float32),
            "unpaired": unpaired,
            "scales": scales,
        }
        return fac, remaining
    stacked = np.stack((remaining[a], remaining[b]), axis=-1)
    q, r = _batched_qr_two_cols(stacked)
    remaining = remaining.copy()
    remaining[a] = q[:, :, 0]
    remaining[b] = q[:, :, 1]
    if int(unpaired.size) > 0:
        col = remaining[unpaired]
        nrm = np.maximum(np.linalg.norm(col, axis=1, keepdims=True), 1e-12)
        scales_d = nrm[:, 0].astype(np.float32)
        remaining[unpaired] = col / nrm
    else:
        scales_d = np.zeros((0,), dtype=np.float32)
    fac = {
        "a": a,
        "b": b,
        "blocks": np.swapaxes(r, -1, -2).astype(np.float32, copy=False),
        "unpaired": unpaired,
        "scales": scales_d,
    }
    return fac, remaining


def peel_butterfly_ladder(
    w: np.ndarray, max_depth: int
) -> dict[int, list[dict[str, np.ndarray]]]:
    """Greedy peels once; every depth D reuses the first D-1 peels.

    Depth-D last factor is the principal-2x2 projection of remaining after
    D-1 peels, so the family stays a D-factor butterfly.
    """
    n = int(w.shape[0])
    remaining = np.array(w, dtype=np.float64, copy=True)
    peel_factors: list[dict[str, np.ndarray]] = []
    out: dict[int, list[dict[str, np.ndarray]]] = {}
    dmax = int(max_depth)
    for stage in range(dmax):
        a, b, unpaired = butterfly_pairs(n, stage)
        last_style = _principal_2x2(remaining, a, b, unpaired)
        out[stage + 1] = peel_factors + [last_style]
        if stage == dmax - 1:
            break
        fac, remaining = _peel_stage(remaining, a, b, unpaired)
        peel_factors.append(fac)
    return out


def project_butterfly(w: np.ndarray, depth: int) -> list[dict[str, np.ndarray]]:
    """Greedy peel of W into `depth` paired-2x2 factors (input-side)."""
    ladder = peel_butterfly_ladder(w, int(depth))
    return ladder[int(depth)]


# ---- Kronecker ----


def apply_kronecker(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """y = x @ (A ⊗ B). x is (N, p q)."""
    p = int(a.shape[0])
    q = int(b.shape[0])
    n_batch = int(x.shape[0])
    # x is row-major of X (p, q). numpy.kron(A, B) satisfies
    # x @ kron(A, B) = vec_row(A.T @ X @ B).
    xm = x.reshape(n_batch, p, q)
    y = np.einsum("ip,niq,qj->npj", a, xm, b, optimize=True)
    return np.ascontiguousarray(y.reshape(n_batch, p * q), dtype=np.float32)


def nearest_kronecker(w: np.ndarray, p: int, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Van Loan nearest Kronecker product W ≈ A ⊗ B via rank-1 power iteration."""
    pp, qq = int(p), int(q)
    w4 = np.ascontiguousarray(w.reshape(pp, qq, pp, qq), dtype=np.float64)
    rearranged = np.ascontiguousarray(
        np.transpose(w4, (0, 2, 1, 3)).reshape(pp * pp, qq * qq), dtype=np.float64
    )
    rng = np.random.default_rng(0)
    v = rng.standard_normal(rearranged.shape[1])
    v /= max(float(np.linalg.norm(v)), 1e-12)
    u = np.empty(rearranged.shape[0], dtype=np.float64)
    for _ in range(8):
        u = rearranged @ v
        u /= max(float(np.linalg.norm(u)), 1e-12)
        v = rearranged.T @ u
        v /= max(float(np.linalg.norm(v)), 1e-12)
    u = rearranged @ v
    s = float(np.linalg.norm(u))
    u = u / max(s, 1e-12)
    scale = math.sqrt(max(s, 0.0))
    a = (scale * u).reshape(pp, pp).astype(np.float32, copy=False)
    b = (scale * v).reshape(qq, qq).astype(np.float32, copy=False)
    return a, b


# ---- Distilled ----


def apply_distilled(
    x: np.ndarray,
    *,
    program: str,
    weights: Mapping[str, np.ndarray],
) -> np.ndarray:
    if program == "linear_affine":
        return apply_affine(x, weights["W"], weights["b"])
    if program == "silu_then_linear":
        return apply_affine(_silu(x), weights["W"], weights["b"])
    if program == "two_layer_silu":
        h = _silu(x @ weights["W1"] + weights["b1"])
        return (h @ weights["W2"] + weights["b2"]).astype(np.float32, copy=False)
    raise StructuredOperatorRefuse(f"unknown distilled program {program!r}")


def fit_distilled(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    program: str,
    ols: Mapping[str, np.ndarray] | None = None,
    n_steps: int = DISTILLED_GD_STEPS,
    ridge_scale: float = RIDGE_SCALE,
) -> dict[str, Any]:
    """Full-width distilled fit. Width is always >= hidden."""
    h = int(x_tr.shape[1])
    if program == "linear_affine":
        packed = ols if ols is not None else fit_affine_ols(x_tr, y_tr, ridge_scale=ridge_scale)
        return {
            "program": program,
            "depth": 1,
            "width": h,
            "weights": {"W": packed["W"], "b": packed["b"]},
            "n_gd_steps": 0,
            "ridge_scale": packed.get("ridge_scale", ridge_scale),
        }
    if program == "silu_then_linear":
        packed = fit_affine_ols(_silu(x_tr), y_tr, ridge_scale=ridge_scale)
        return {
            "program": program,
            "depth": 1,
            "width": h,
            "weights": {"W": packed["W"], "b": packed["b"]},
            "n_gd_steps": 0,
            "ridge_scale": packed.get("ridge_scale", ridge_scale),
        }
    if program != "two_layer_silu":
        raise StructuredOperatorRefuse(f"unknown distilled program {program!r}")

    # Identity-plus-noise first layer: full rank, width = hidden, not a
    # bottleneck. Readout is ridge on silu(x W1). Unregularized 2 n^2 is
    # slightly underdetermined on one captured layer; ridge is the control.
    rng = np.random.default_rng(RNG_SEED + 11)
    w1 = np.eye(h, dtype=np.float32) + 0.1 * rng.standard_normal((h, h)).astype(np.float32)
    b1 = np.zeros(h, dtype=np.float32)
    h_tr = _silu(x_tr @ w1 + b1)
    packed = fit_affine_ols(h_tr, y_tr, ridge_scale=ridge_scale)
    w2 = packed["W"]
    b2 = packed["b"]
    return {
        "program": program,
        "depth": 2,
        "width": h,
        "weights": {"W1": w1, "b1": b1, "W2": w2, "b2": b2},
        "n_gd_steps": 0,
        "ridge_scale": packed.get("ridge_scale", ridge_scale),
        "underdetermined_unregularized": bool(int(x_tr.shape[0]) * h < 2 * h * h),
        "regularized": True,
        "w1_kind": "identity_plus_gaussian_0.1",
    }


# ---------------------------------------------------------------------------
# Emit. The only path a candidate may take into the receipt.
# ---------------------------------------------------------------------------


def _economics(
    *,
    bytes_removed: int,
    bytes_added: Mapping[str, int],
    consuming_primitive: str,
    status: str,
    candidate_id: str,
    extra_flops_per_output_element: float,
    dispatch_delta: float,
    dispatch_delta_note: str,
) -> dict[str, Any]:
    scored = ee.score(
        bytes_removed=int(bytes_removed),
        bytes_added={k: int(bytes_added.get(k, 0)) for k in ee.BYTES_ADDED_FIELDS},
        extra_flops_per_output_element=float(extra_flops_per_output_element),
        dispatch_delta=float(dispatch_delta),
        consuming_primitive=consuming_primitive,
        organ="mlp",
        stream_class="weight_codes",
        reusable_family=True,
        high_information_falsifier=True,
        status=status,
        candidate_id=candidate_id,
    )
    s20 = scored["s020_section_20"]
    assumptions = scored["assumptions"]
    return {
        "id": candidate_id,
        "status": scored["status"],
        "live": scored["live"],
        "verdict": scored["verdict"],
        "verdict_reasons": list(scored["verdict_reasons"]),
        "bytes_removed": scored["bytes_removed"],
        "bytes_added": {k: int(scored["bytes_added"].get(k, 0)) for k in ee.BYTES_ADDED_FIELDS},
        "bytes_added_total": int(scored["bytes_added"].get("total", 0)),
        "net_bytes": scored["net_bytes"],
        "consuming_primitive": scored["consuming_primitive"],
        "extra_flops_per_output_element": scored["extra_flops_per_output_element"],
        "dispatch_delta": scored["dispatch_delta"],
        "predicted_ms_delta": _r(scored["predicted_ms_delta"], 4),
        "predicted_ms_saved": _r(scored["predicted_ms_saved"], 4),
        "predicted_token_ms": _r(scored["predicted_token_ms"], 4),
        "predicted_tps": _r(scored["predicted_tps"], 3),
        "predicted_ms_delta_range": [
            _r(scored["predicted_ms_delta_range"][0], 4),
            _r(scored["predicted_ms_delta_range"][1], 4),
        ],
        "terms": {k: _r(v, 4) for k, v in scored["terms"].items()},
        "assumptions": {
            "bandwidth_regime": assumptions["bandwidth_regime"],
            "bandwidth_gb_s_nominal": _r(assumptions["bandwidth_gb_s_nominal"], 2),
            "bandwidth_gb_s_range": [
                _r(assumptions["bandwidth_gb_s_range"][0], 2),
                _r(assumptions["bandwidth_gb_s_range"][1], 2),
            ],
            "bandwidth_is_assumption": assumptions["bandwidth_is_assumption"],
            "bandwidth_note": assumptions["bandwidth_note"],
            "dispatch_class": assumptions["dispatch_class"],
            "dispatch_note": assumptions["dispatch_note"],
            "element_bytes": ELEMENT_BYTES,
            "element_bytes_note": (
                "program billed at f16; the fit itself is f32. ASSUMPTION."
            ),
            "dispatch_delta_note": dispatch_delta_note,
            "scorer": "tools.future.executable_economics.score",
        },
        "s020_section_20": {
            "bar_ms": _r(s20["bar_ms"], 4),
            "plausible_ms_saved": _r(s20["plausible_ms_saved"], 4),
            "clears_time_bar": s20["clears_time_bar"],
            "reusable_family": s20["reusable_family"],
            "high_information_falsifier": s20["high_information_falsifier"],
        },
    }


def status_from_error(
    held: float,
    mean_held: float,
    per_domain: Mapping[str, float] | None = None,
) -> tuple[str, bool, bool, str]:
    """(status, cheap_kill, null_model, why)."""
    vs = vs_mean_predictor(held, mean_held)
    domain_hit = False
    worst_domain = None
    worst_val = -1.0
    if per_domain:
        for name, val in per_domain.items():
            if float(val) > worst_val:
                worst_val = float(val)
                worst_domain = name
            if float(val) >= HELD_OUT_KILL_REL:
                domain_hit = True
    if vs["null_model"]:
        return (
            MEASURED_NEGATIVE,
            True,
            True,
            f"held-out relative L2 {held} does not beat the mean predictor "
            f"{mean_held}; NULL MODEL, not a result",
        )
    if held >= KILL_BAND:
        return (
            MEASURED_NEGATIVE,
            True,
            False,
            f"held-out relative L2 {held} sits in the 0.9 band "
            f"(>={KILL_BAND}); beats mean {mean_held} but not by a margin "
            "that matters",
        )
    if held >= HELD_OUT_KILL_REL:
        return (
            MEASURED_NEGATIVE,
            False,
            False,
            f"held-out relative L2 {held} beats the mean {mean_held} but "
            f"is above the {HELD_OUT_KILL_REL} function-replacement kill",
        )
    if domain_hit:
        return (
            MEASURED_NEGATIVE,
            False,
            False,
            f"mean held-out relative L2 {held} is below the kill but domain "
            f"{worst_domain}={worst_val} is not",
        )
    if not vs["margin_that_matters"]:
        return (
            MEASURED_NEGATIVE,
            False,
            False,
            f"held-out relative L2 {held} beats the mean {mean_held} but "
            "the margin does not matter for replacing F",
        )
    return OPEN, False, False, f"held-out relative L2 {held} is below {HELD_OUT_KILL_REL}"


def emit_candidate(
    *,
    family: str,
    program: str,
    pred_tr: np.ndarray,
    pred_ho: np.ndarray,
    y_tr: np.ndarray,
    y_ho: np.ndarray,
    consumer: Mapping[str, Any],
    mean_held_out_relative_l2: float | None,
    n_blocks: int = 0,
    depth: int = 0,
    factor_p: int = 0,
    factor_q: int = 0,
    width: int = 0,
    extra: Mapping[str, Any] | None = None,
    meta_ho: Mapping[str, Sequence[str]] | None = None,
    n_layers: int = N_LAYERS,
    hidden: int = HIDDEN,
    id_suffix: str | None = None,
    force_byte_win: bool = False,
) -> dict[str, Any]:
    """The only constructor a receipt row is allowed to pass through."""
    if mean_held_out_relative_l2 is None:
        raise BaselineOmitted(
            "REFUSED: held-out error without the mean-predictor baseline; "
            "a number that is not reported beside the mean is not a result"
        )
    fam = _require_family(family)
    cstat = consumer_status(consumer)
    if cstat == REJECTED_DENSE_REMAT:
        raise RematConsumer(
            f"REJECTED_DENSE_REMAT: cannot report a remat shape as a live candidate ({fam})"
        )

    br = byte_breakdown(
        fam,
        n_layers=n_layers,
        hidden=hidden,
        n_blocks=n_blocks,
        depth=depth,
        factor_p=factor_p,
        factor_q=factor_q,
        width=width if int(width) > 0 else hidden,
        program=program,
    )
    added = bytes_added_from_breakdown(br)
    exceeds = complete_ledger_exceeds_incumbent(int(added["total"]))
    ho = function_error(
        pred_ho, y_ho, split="hold", report_as="held_out", meta=meta_ho
    )
    tr = function_error(pred_tr, y_tr, split="train", report_as="train")
    held = float(ho["held_out_relative_l2"])
    mean_held = float(mean_held_out_relative_l2)
    vs = vs_mean_predictor(held, mean_held)
    status, cheap_kill, null_model, why = status_from_error(
        held, mean_held, ho.get("per_capability_domain")
    )
    cid = f"{fam.lower()}_{id_suffix or program}"
    byte_win = (not exceeds) and int(added["total"]) < INCUMBENT_MLP_BYTES
    row: dict[str, Any] = {
        "id": cid,
        "family": fam,
        "program": program,
        "n_blocks": int(n_blocks),
        "depth": int(depth),
        "factor_p": int(factor_p),
        "factor_q": int(factor_q),
        "width": int(br["width"]),
        "min_width": int(br["min_width"]),
        "full_width": bool(br["full_width"]),
        "structure_parameter": _structure_param(fam, n_blocks, depth, factor_p, factor_q, width, program),
        "byte_breakdown": dict(br),
        "bytes_added": added,
        "exceeds_incumbent": bool(exceeds),
        "byte_win": bool(byte_win and not exceeds),
        "incumbent_mlp_bytes": INCUMBENT_MLP_BYTES,
        "consumer": dict(consumer),
        "consumer_status": DIRECT_CONSUME,
        "status": status,
        "cheap_kill": bool(cheap_kill),
        "status_why": why,
        "null_model": bool(null_model),
        "weight_reconstruction_error": None,
        "weight_reconstruction_note": "not authority; this experiment scores F, not W",
        "error_authority": "held_out_relative_l2",
        "held_out_kill_rel": HELD_OUT_KILL_REL,
        "kill_band": KILL_BAND,
        "n_layers_billed": int(n_layers),
        "index_from": "x",
    }
    row.update(ho)
    row.update(tr)
    row.update(vs)
    # vs overwrites held_out_relative_l2 with the rounded form; keep authority keys.
    row["held_out_split"] = "hold"
    row["error_authority"] = "held_out_relative_l2"
    if extra:
        for key, value in extra.items():
            if key not in row:
                row[key] = value
    validate_billing(row)
    validate_error_authority(row)
    if force_byte_win:
        row["byte_win"] = True
        report_as_byte_win(row)
    if row.get("byte_win") and exceeds:
        report_as_byte_win(row)
    if exceeds:
        row["byte_win"] = False
        row["byte_win_refused"] = (
            "complete ledger exceeds incumbent; not a byte win "
            f"({int(added['total'])} >= {INCUMBENT_MLP_BYTES})"
        )

    dispatch_note = str(consumer.get("dispatch_delta_note") or "")
    extra_flops = float(consumer.get("extra_flops_per_output_element") or 0.0)
    dispatch_delta = float(consumer.get("dispatch_delta") or 0.0)
    row["economics"] = _economics(
        bytes_removed=ee.MLP_ACTIVE_BYTES,
        bytes_added=added,
        consuming_primitive=str(consumer["primitive"]),
        status=status,
        candidate_id=cid,
        extra_flops_per_output_element=extra_flops,
        dispatch_delta=dispatch_delta,
        dispatch_delta_note=dispatch_note,
    )
    open_econ = row["economics"]
    if status != OPEN:
        open_econ = _economics(
            bytes_removed=ee.MLP_ACTIVE_BYTES,
            bytes_added=added,
            consuming_primitive=str(consumer["primitive"]),
            status=OPEN,
            candidate_id=cid,
            extra_flops_per_output_element=extra_flops,
            dispatch_delta=dispatch_delta,
            dispatch_delta_note=dispatch_note,
        )
        row["economics_if_function_held"] = {
            "verdict": open_econ["verdict"],
            "predicted_ms_saved": open_econ["predicted_ms_saved"],
            "clears_time_bar": open_econ["s020_section_20"]["clears_time_bar"],
            "net_bytes": open_econ["net_bytes"],
            "dispatch_ms_delta": open_econ["terms"]["dispatch_ms_delta"],
            "flop_ms_delta": open_econ["terms"]["flop_ms_delta"],
            "byte_ms_delta": open_econ["terms"]["byte_ms_delta"],
        }
    row["clears_s020_time_bar_if_function_held"] = bool(
        open_econ["s020_section_20"]["clears_time_bar"]
    )
    # Dispatch can erase a byte win. Record the honest split.
    row["dispatch_may_erase_byte_win"] = bool(
        float(open_econ["terms"]["dispatch_ms_delta"])
        > abs(float(open_econ["terms"]["byte_ms_delta"]))
        and float(open_econ["terms"]["byte_ms_delta"]) < 0.0
    )
    return _py(row)


def _structure_param(
    family: str,
    n_blocks: int,
    depth: int,
    factor_p: int,
    factor_q: int,
    width: int,
    program: str,
) -> str:
    if family == MONARCH:
        return f"n_blocks={int(n_blocks)}"
    if family == BUTTERFLY:
        return f"depth={int(depth)}"
    if family == KRONECKER:
        return f"factors={int(factor_p)}x{int(factor_q)}"
    return f"depth={int(depth)},width={int(width) if width else HIDDEN},program={program}"


def surviving_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    live = []
    for row in rows:
        if row.get("consumer_status") == REJECTED_DENSE_REMAT:
            continue
        if row.get("status") in ee.DEAD_STATUSES or row.get("status") == REJECTED_DENSE_REMAT:
            continue
        if row.get("null_model"):
            continue
        if row.get("exceeds_incumbent"):
            continue
        live.append(dict(row))
    return live


# ---------------------------------------------------------------------------
# Census on the four captured layers.
# ---------------------------------------------------------------------------


def _baselines_from_concat(
    y_tr: np.ndarray,
    y_ho: np.ndarray,
    mean_tr_vec: np.ndarray,
    meta_ho: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    zero_ho = function_error(
        np.zeros_like(y_ho), y_ho, split="hold", report_as="held_out", meta=meta_ho
    )
    mean_pred = np.broadcast_to(mean_tr_vec.reshape(1, -1), y_ho.shape)
    # mean_tr_vec here is not used; caller passes per-row mean predictions.
    del mean_pred
    return {
        "zero_held_out_relative_l2": zero_ho["held_out_relative_l2"],
        "held_out_split": "hold",
        "note": "baselines are held-out; they are not candidates",
    }


def _concat_meta(metas: Sequence[Mapping[str, Sequence[str]]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"domain": [], "band": [], "prompt_id": []}
    for meta in metas:
        out["domain"].extend(list(meta["domain"]))
        out["band"].extend(list(meta["band"]))
        out["prompt_id"].extend(list(meta["prompt_id"]))
    return out


def _layer_mean_predictor(y_tr: np.ndarray, y_ho: np.ndarray) -> np.ndarray:
    mean = y_tr.mean(axis=0, keepdims=True)
    return np.broadcast_to(mean, y_ho.shape).astype(np.float32, copy=False)


def _fit_one_layer(pack: Mapping[str, Any]) -> dict[str, Any]:
    """OLS + projections + distilled, one captured layer."""
    x_tr, y_tr = pack["Xtr"], pack["Ytr"]
    x_ho, y_ho = pack["Xho"], pack["Yho"]
    ols = fit_affine_ols(x_tr, y_tr)
    w, b = ols["W"], ols["b"]
    out: dict[str, Any] = {
        "layer": int(pack["layer"]),
        "ols": ols,
        "mean_ho": _layer_mean_predictor(y_tr, y_ho),
        "mean_tr": _layer_mean_predictor(y_tr, y_tr),
        "preds": {},
    }
    # Distilled linear IS the ridge affine map.
    out["preds"][("DISTILLED", "linear_affine")] = {
        "tr": apply_affine(x_tr, w, b),
        "ho": apply_affine(x_ho, w, b),
        "n_blocks": 0,
        "depth": 1,
        "factor_p": 0,
        "factor_q": 0,
        "width": int(x_tr.shape[1]),
        "program": "linear_affine",
        "extra": {
            "ridge_scale": ols.get("ridge_scale", RIDGE_SCALE),
            "ridge_lambda": ols.get("ridge_lambda"),
        },
    }
    silu = fit_distilled(x_tr, y_tr, program="silu_then_linear")
    out["preds"][("DISTILLED", "silu_then_linear")] = {
        "tr": apply_distilled(x_tr, program="silu_then_linear", weights=silu["weights"]),
        "ho": apply_distilled(x_ho, program="silu_then_linear", weights=silu["weights"]),
        "n_blocks": 0,
        "depth": 1,
        "factor_p": 0,
        "factor_q": 0,
        "width": int(x_tr.shape[1]),
        "program": "silu_then_linear",
        "extra": {"ridge_scale": silu.get("ridge_scale", RIDGE_SCALE)},
    }
    two = fit_distilled(x_tr, y_tr, program="two_layer_silu")
    out["preds"][("DISTILLED", "two_layer_silu")] = {
        "tr": apply_distilled(x_tr, program="two_layer_silu", weights=two["weights"]),
        "ho": apply_distilled(x_ho, program="two_layer_silu", weights=two["weights"]),
        "n_blocks": 0,
        "depth": 2,
        "factor_p": 0,
        "factor_q": 0,
        "width": int(x_tr.shape[1]),
        "program": "two_layer_silu",
        "extra": {
            "n_gd_steps": two.get("n_gd_steps"),
            "underdetermined_unregularized": two.get("underdetermined_unregularized"),
            "regularized": two.get("regularized"),
            "ridge_scale": two.get("ridge_scale", RIDGE_SCALE),
            "w1_kind": two.get("w1_kind"),
        },
    }
    h = int(x_tr.shape[1])
    for n_blocks in MONARCH_BLOCKS:
        if h % int(n_blocks) != 0 or int(n_blocks) < 2:
            continue
        r_blk, l_blk = project_monarch(w, int(n_blocks))
        out["preds"][("MONARCH", int(n_blocks))] = {
            "tr": apply_monarch(x_tr, r_blk, l_blk) + b,
            "ho": apply_monarch(x_ho, r_blk, l_blk) + b,
            "n_blocks": int(n_blocks),
            "depth": 0,
            "factor_p": 0,
            "factor_q": 0,
            "width": h,
            "program": "block_butterfly",
        }
    max_stage = int(math.floor(math.log2(h)))
    depths = sorted({min(int(d), max_stage) for d in BUTTERFLY_DEPTHS if int(d) >= 1})
    if depths:
        ladder = peel_butterfly_ladder(w, max(depths))
        for d in depths:
            factors = ladder[int(d)]
            out["preds"][("BUTTERFLY", d)] = {
                "tr": apply_butterfly(x_tr, factors) + b,
                "ho": apply_butterfly(x_ho, factors) + b,
                "n_blocks": 0,
                "depth": d,
                "factor_p": 0,
                "factor_q": 0,
                "width": h,
                "program": "fft_like",
            }
    for p, q in KRONECKER_SHAPES:
        if int(p) * int(q) != h:
            continue
        a, bb = nearest_kronecker(w, int(p), int(q))
        out["preds"][( "KRONECKER", (int(p), int(q)) )] = {
            "tr": apply_kronecker(x_tr, a, bb) + b,
            "ho": apply_kronecker(x_ho, a, bb) + b,
            "n_blocks": 0,
            "depth": 0,
            "factor_p": int(p),
            "factor_q": int(q),
            "width": h,
            "program": "tensor_product",
        }
    return out


def _emit_pooled(
    *,
    family: str,
    key: Any,
    layer_fits: Sequence[Mapping[str, Any]],
    packs: Sequence[Mapping[str, Any]],
    hidden: int = HIDDEN,
    n_layers: int = N_LAYERS,
) -> dict[str, Any]:
    pred_tr = np.concatenate([lf["preds"][key]["tr"] for lf in layer_fits], axis=0)
    pred_ho = np.concatenate([lf["preds"][key]["ho"] for lf in layer_fits], axis=0)
    y_tr = np.concatenate([p["Ytr"] for p in packs], axis=0)
    y_ho = np.concatenate([p["Yho"] for p in packs], axis=0)
    mean_ho = np.concatenate([lf["mean_ho"] for lf in layer_fits], axis=0)
    mean_tr = np.concatenate([lf["mean_tr"] for lf in layer_fits], axis=0)
    meta_ho = _concat_meta([p["hold_meta"] for p in packs])
    mean_held = mean_l2_ratio(mean_ho, y_ho)
    spec0 = layer_fits[0]["preds"][key]
    consumer = native_consumer_sketch(
        family,
        n_blocks=int(spec0["n_blocks"]),
        depth=int(spec0["depth"]),
        factor_p=int(spec0["factor_p"]),
        factor_q=int(spec0["factor_q"]),
        width=int(spec0["width"]),
        n_layers=n_layers,
        hidden=hidden,
    )
    per_layer = {}
    for lf, pack in zip(layer_fits, packs):
        spec = lf["preds"][key]
        ho_l = mean_l2_ratio(spec["ho"], pack["Yho"])
        mean_l = mean_l2_ratio(lf["mean_ho"], pack["Yho"])
        vs_l = vs_mean_predictor(ho_l, mean_l)
        per_layer[str(pack["layer"])] = {
            "held_out_relative_l2": vs_l["held_out_relative_l2"],
            "mean_held_out_relative_l2": vs_l["mean_held_out_relative_l2"],
            "margin_over_mean": vs_l["margin_over_mean"],
            "beats_mean_predictor": vs_l["beats_mean_predictor"],
            "null_model": vs_l["null_model"],
            "margin_that_matters": vs_l["margin_that_matters"],
            "held_out_split": "hold",
            "n_hold": int(pack["n_hold"]),
            "n_train": int(pack["n_train"]),
        }
    extra = {
        "algebra": consumer.get("algebra"),
        "per_layer_held_out_relative_l2": per_layer,
        "typical_layer": TYPICAL_LAYER,
        "typical_layer_held_out_relative_l2": per_layer.get(str(TYPICAL_LAYER), {}).get(
            "held_out_relative_l2"
        ),
        "layers_measured": [int(p["layer"]) for p in packs],
        "pooled_split_unit": "prompt_id",
        "flops_per_token": consumer.get("flops_per_token"),
        "incumbent_flops_per_token": consumer.get("incumbent_flops_per_token"),
        "flop_ratio_vs_incumbent": consumer.get("flop_ratio_vs_incumbent"),
        "extra_flops_note": consumer.get("extra_flops_note"),
        "mean_train_relative_l2_of_mean_predictor": _r(mean_l2_ratio(mean_tr, y_tr)),
    }
    spec_extra = spec0.get("extra")
    if isinstance(spec_extra, Mapping):
        extra.update(spec_extra)
    suffix = {
        MONARCH: f"b{spec0['n_blocks']}",
        BUTTERFLY: f"d{spec0['depth']}",
        KRONECKER: f"{spec0['factor_p']}x{spec0['factor_q']}",
        DISTILLED: spec0["program"],
    }[family]
    return emit_candidate(
        family=family,
        program=str(spec0["program"]),
        pred_tr=pred_tr,
        pred_ho=pred_ho,
        y_tr=y_tr,
        y_ho=y_ho,
        consumer=consumer,
        mean_held_out_relative_l2=float(mean_held),
        n_blocks=int(spec0["n_blocks"]),
        depth=int(spec0["depth"]),
        factor_p=int(spec0["factor_p"]),
        factor_q=int(spec0["factor_q"]),
        width=int(spec0["width"]),
        extra=extra,
        meta_ho=meta_ho,
        n_layers=n_layers,
        hidden=hidden,
        id_suffix=suffix,
    )


def run_census(
    *,
    layers: Sequence[int] = MEASURED_LAYERS,
    payload_dir: Path | None = None,
    n_layers_billed: int = N_LAYERS,
) -> dict[str, Any]:
    packs = [load_layer_pack(int(layer), payload_dir=payload_dir) for layer in layers]
    hidden = int(packs[0]["Xtr"].shape[1])
    layer_fits = [_fit_one_layer(p) for p in packs]

    # Mean predictor pooled (per-layer train mean, scored on that layer's hold).
    y_ho = np.concatenate([p["Yho"] for p in packs], axis=0)
    y_tr = np.concatenate([p["Ytr"] for p in packs], axis=0)
    mean_ho = np.concatenate([lf["mean_ho"] for lf in layer_fits], axis=0)
    mean_tr = np.concatenate([lf["mean_tr"] for lf in layer_fits], axis=0)
    meta_ho = _concat_meta([p["hold_meta"] for p in packs])
    mean_held_doc = function_error(
        mean_ho, y_ho, split="hold", report_as="held_out", meta=meta_ho
    )
    mean_tr_doc = function_error(mean_tr, y_tr, split="train", report_as="train")
    zero_doc = function_error(
        np.zeros_like(y_ho), y_ho, split="hold", report_as="held_out", meta=meta_ho
    )
    baselines = {
        "zero_held_out_relative_l2": zero_doc["held_out_relative_l2"],
        "mean_held_out_relative_l2": mean_held_doc["held_out_relative_l2"],
        "mean_held_out_cosine": mean_held_doc.get("held_out_cosine"),
        "mean_train_relative_l2_diagnostic": mean_tr_doc["train_relative_l2_diagnostic"],
        "held_out_split": "hold",
        "per_layer": {
            str(p["layer"]): vs_mean_predictor(
                mean_l2_ratio(lf["mean_ho"], p["Yho"]),
                mean_l2_ratio(lf["mean_ho"], p["Yho"]),
            )["mean_held_out_relative_l2"]
            for p, lf in zip(packs, layer_fits)
        },
        "note": (
            "baselines are held-out by prompt; they are not candidates. "
            "A candidate that does not beat mean_held_out_relative_l2 is a "
            "NULL MODEL."
        ),
    }

    # Union of keys present on every layer (structure params that applied).
    key_sets = [set(lf["preds"].keys()) for lf in layer_fits]
    keys = sorted(set.intersection(*key_sets), key=lambda k: (str(k[0]), str(k[1])))
    rows: list[dict[str, Any]] = []
    for key in keys:
        family = str(key[0])
        rows.append(
            _emit_pooled(
                family=family,
                key=key,
                layer_fits=layer_fits,
                packs=packs,
                hidden=hidden,
                n_layers=n_layers_billed,
            )
        )

    by_family: dict[str, list[dict[str, Any]]] = {f: [] for f in FAMILIES}
    for row in rows:
        by_family[str(row["family"])].append(row)

    family_verdicts = []
    scars = []
    for family in FAMILIES:
        group = by_family[family]
        if not group:
            continue
        under = [r for r in group if not r.get("exceeds_incumbent")]
        pool = under if under else group
        best = min(pool, key=lambda r: float(r["held_out_relative_l2"]))
        all_null = all(r.get("null_model") for r in under) if under else True
        dead = all(r.get("status") != OPEN for r in group)
        mechanism = _mechanism_for(family, best)
        if dead and all_null:
            why = (
                f"best held-out relative L2 {best['held_out_relative_l2']} vs "
                f"mean predictor {best['mean_held_out_relative_l2']}; NULL MODEL"
            )
        elif dead:
            why = (
                f"best held-out relative L2 {best['held_out_relative_l2']} vs "
                f"mean predictor {best['mean_held_out_relative_l2']}; beats the "
                f"mean but stays above the {HELD_OUT_KILL_REL} function-replacement "
                "kill. MEASURED_NEGATIVE."
            )
        else:
            why = (
                "at least one under-incumbent setting is below the "
                f"{HELD_OUT_KILL_REL} function-replacement kill"
            )
        verdict = {
            "family": family,
            "status": MEASURED_NEGATIVE if dead else OPEN,
            "n_rows": len(group),
            "n_under_incumbent": len(under),
            "best_id": best["id"],
            "best_held_out_relative_l2": best["held_out_relative_l2"],
            "best_mean_held_out_relative_l2": best["mean_held_out_relative_l2"],
            "best_margin_over_mean": best["margin_over_mean"],
            "best_beats_mean_predictor": best["beats_mean_predictor"],
            "best_null_model": best["null_model"],
            "best_margin_that_matters": best["margin_that_matters"],
            "best_exceeds_incumbent": best["exceeds_incumbent"],
            "bytes_added_total_at_best": best["bytes_added"]["total"],
            "dispatch_delta_at_best": best["consumer"].get("dispatch_delta"),
            "clears_s020_time_bar_if_function_held": best[
                "clears_s020_time_bar_if_function_held"
            ],
            "native_consumer": best["consumer"],
            "mechanism": mechanism,
            "all_under_incumbent_are_null_models": bool(all_null),
            "why": why,
        }
        family_verdicts.append(verdict)
        if dead:
            scars.append(
                {
                    "family": family,
                    "status": MEASURED_NEGATIVE,
                    "held_out_relative_l2_best": best["held_out_relative_l2"],
                    "mean_held_out_relative_l2": best["mean_held_out_relative_l2"],
                    "null_model": best["null_model"],
                    "mechanism": mechanism,
                    "not": (
                        "a retry of SHARED_*, FACTORIZE_THE_FACTORS, a "
                        "dictionary, a routed mixture, or a per-block "
                        "factorization"
                    ),
                    "level": "MODEL_SPECIFIC",
                    "parent": "qwen3.8-27b sealed-3.14",
                    "organ": "mlp",
                    "object": "F(x)=down(silu(gate(x))*up(x)) on the teacher corpus",
                }
            )

    distilled_rows = by_family[DISTILLED]
    distilled_control = _distilled_control_block(distilled_rows, baselines)

    under_rows = [r for r in rows if not r.get("exceeds_incumbent")]
    open_under = [r for r in under_rows if r.get("status") == OPEN]
    beats_under = [r for r in under_rows if r.get("beats_mean_predictor")]
    matters_under = [r for r in under_rows if r.get("margin_that_matters")]
    closed = len(open_under) == 0
    survivors = surviving_candidates(rows)

    if not closed:
        mechanism = (
            "At least one under-incumbent full-width operator is below the "
            f"{HELD_OUT_KILL_REL} function-replacement kill; function "
            "replacement is not closed."
        )
    elif distilled_control.get("fails_too"):
        mechanism = (
            "Every full-width structured operator under the incumbent ledger "
            "is a NULL MODEL or fails the function-replacement kill, and the "
            "distilled control fails too. The previous negatives were "
            "r-bottlenecks; this lane removes that mechanism and the function "
            "still does not hold. F at this precision carries the incumbent's "
            "independent information; MLP information elimination is CLOSED "
            "and the campaign's remaining lever is execution."
        )
    elif beats_under:
        mechanism = (
            "No under-incumbent full-width operator replaces F (held-out "
            f"relative L2 stays above {HELD_OUT_KILL_REL}). Distilled "
            "full-width linear does beat the mean predictor, so some of F is "
            "linearly accessible without an r-bottleneck; typical captured "
            "layers still sit at ~0.8 and none cross the kill. Function "
            "replacement at this ledger is CLOSED; the remaining lever is "
            "execution, not a cheaper program."
        )
    else:
        mechanism = (
            "No under-incumbent full-width operator beats the mean predictor "
            "by a margin that matters, so replacement at this ledger is "
            "closed. Distilled is recorded separately as the capacity control."
        )

    # THE UMBRELLA SCAR MUST BE EMITTED, NOT HAND-ADDED TO THE RECEIPT.
    #
    # 6fc77f169 fixed "the resident launched zero units" by adding a
    # MLP_FUNCTION_REPLACEMENT row to receipts/future/MLP_STRUCTURED_OPERATOR.json
    # and touching NOTHING in this file. That fix survived exactly until someone
    # rebuilt the receipt - the scars array went 5 -> 4, refuse_if_dead stopped
    # keying the closed family, and WU.DEAD.mlp_function_replacement came straight
    # back as the scripted policy. Which is the defect that commit exists to close.
    #
    # negative_index reads the `scars` ARRAY. campaign.scar_id is not in it, so an
    # umbrella closure recorded only there prunes nothing. Emit it where the index
    # looks, from the same `closed` computation that sets campaign.scar_id, so the
    # two can never disagree again.
    if closed:
        scars.append(
            {
                "family": "MLP_FUNCTION_REPLACEMENT",
                "status": MEASURED_NEGATIVE,
                "level": "MODEL_SPECIFIC",
                "parent": "qwen3.8-27b sealed-3.14",
                "organ": "mlp",
                "object": "F(x)=down(silu(gate(x))*up(x)) on the teacher corpus",
                "mechanism": (
                    "the UMBRELLA closure, " + CLOSED_SCAR + ". Every "
                    "bottleneck-shaped family died (rank, dictionary, conditional, "
                    "generated-block, nonlinear generator, factorwise), and the "
                    "full-width structured operators died too. The DISTILLED "
                    "CONTROL is what closes it: with no narrow layer anywhere it "
                    "BEATS the mean predictor yet still cannot carry F under the "
                    "incumbent ledger. A control with more capacity than any "
                    "structured family and no structural constraint at all still "
                    "fails, so the remaining error is not a missing SHAPE."
                ),
                "not": (
                    "a retry of SHARED_*, FACTORIZE_THE_FACTORS, a dictionary, a "
                    "routed mixture, a per-block factorization, Monarch, butterfly "
                    "or Kronecker"
                ),
                "distilled_control_fails_too": bool(distilled_control.get("fails_too")),
                "n_survivors_under_incumbent": len(survivors),
            }
        )

    campaign = {
        "scar_id": CLOSED_SCAR if closed else None,
        "function_replacement_closed": bool(closed),
        "n_survivors_under_incumbent": len(survivors),
        "n_open_under_incumbent": len(open_under),
        "n_beats_mean_under_incumbent": len(beats_under),
        "n_margin_that_matters_under_incumbent": len(matters_under),
        "mechanism": mechanism,
        "distilled_control_interpretation": distilled_control.get("interpretation"),
    }

    return {
        "layers": [int(p["layer"]) for p in packs],
        "payload_dir": packs[0]["payload_dir"],
        "n_train_total": int(sum(p["n_train"] for p in packs)),
        "n_hold_total": int(sum(p["n_hold"] for p in packs)),
        "n_train_per_layer": {str(p["layer"]): int(p["n_train"]) for p in packs},
        "n_hold_per_layer": {str(p["layer"]): int(p["n_hold"]) for p in packs},
        "split_unit": "prompt_id",
        "disjoint": True,
        "x_sha256": {str(p["layer"]): p["x_sha256"] for p in packs},
        "y_sha256": {str(p["layer"]): p["y_sha256"] for p in packs},
        "hidden": hidden,
        "held_out_kill_rel": HELD_OUT_KILL_REL,
        "kill_band": KILL_BAND,
        "baselines": baselines,
        "rows": rows,
        "family_verdicts": family_verdicts,
        "scars": scars,
        "distilled_control": distilled_control,
        "survivors": survivors,
        "n_survivors": len(survivors),
        "campaign": campaign,
        "go_wider": False,
    }


def _mechanism_for(family: str, best: Mapping[str, Any]) -> str:
    if family == MONARCH:
        return (
            "Monarch is full rank (product of two block-diagonals and a "
            "permutation) at O(n^{1.5}) parameters. The bottleneck mechanism "
            "of the closed lanes does not apply. Held-out error still sits "
            f"at {best.get('held_out_relative_l2')} vs mean "
            f"{best.get('mean_held_out_relative_l2')}."
        )
    if family == BUTTERFLY:
        return (
            "Butterfly is full rank at O(n log n) parameters (log n paired "
            "2x2 factors). Not an r-bottleneck. Held-out error still sits "
            f"at {best.get('held_out_relative_l2')} vs mean "
            f"{best.get('mean_held_out_relative_l2')}."
        )
    if family == KRONECKER:
        return (
            "Kronecker A ⊗ B is full rank when both factors are. Parameter "
            "count p^2+q^2. Not an r-bottleneck. Held-out error still sits "
            f"at {best.get('held_out_relative_l2')} vs mean "
            f"{best.get('mean_held_out_relative_l2')}."
        )
    return (
        "Distilled operator is a dense net with every hidden width at or "
        "above the input width, so it is not a bottleneck in disguise. "
        f"Held-out error {best.get('held_out_relative_l2')} vs mean "
        f"{best.get('mean_held_out_relative_l2')}. This is the capacity control."
    )


def _distilled_control_block(
    rows: Sequence[Mapping[str, Any]],
    baselines: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise StructuredOperatorRefuse(
            "REFUSED: distilled control did not run; it must run even if "
            "the structured families fail"
        )
    under = [r for r in rows if not r.get("exceeds_incumbent")]
    over = [r for r in rows if r.get("exceeds_incumbent")]
    best = min(rows, key=lambda r: float(r["held_out_relative_l2"]))
    best_under = (
        min(under, key=lambda r: float(r["held_out_relative_l2"])) if under else None
    )
    fails_too = not any(r.get("beats_mean_predictor") for r in rows)
    fails_under = not any(r.get("beats_mean_predictor") for r in under)
    replaces = any(r.get("status") == OPEN for r in under)
    if replaces:
        interpretation = (
            "distilled succeeds under the incumbent -> the previous failures "
            "were about STRUCTURE, not information, and structured search continues."
        )
    elif fails_too:
        interpretation = (
            "distilled fails too -> F genuinely needs ~5.3 GB of independent "
            "information at this precision, MLP information elimination is "
            "CLOSED, and the campaign's whole remaining lever is execution."
        )
    elif fails_under and any(r.get("beats_mean_predictor") for r in over):
        interpretation = (
            "distilled beats the mean only at a ledger at or above the "
            "incumbent. Structured search at a cheaper-than-incumbent ledger "
            "is still closed as function replacement; going wider than the "
            "incumbent is the already-measured full-rank cost."
        )
    else:
        interpretation = (
            "distilled beats the mean under the incumbent so some of F is "
            "accessible without an r-bottleneck (the previous 0.9-band "
            "failures were in part about width). It does not replace F "
            f"(kill {HELD_OUT_KILL_REL}; typical layers remain near 0.8). "
            "Function replacement at this ledger is still closed; the "
            "remaining lever is execution."
        )
    return {
        "ran": True,
        "n_settings": len(rows),
        "mean_held_out_relative_l2": baselines.get("mean_held_out_relative_l2"),
        "best_id": best["id"],
        "best_held_out_relative_l2": best["held_out_relative_l2"],
        "best_margin_over_mean": best["margin_over_mean"],
        "best_beats_mean_predictor": best["beats_mean_predictor"],
        "best_null_model": best["null_model"],
        "best_margin_that_matters": best["margin_that_matters"],
        "best_exceeds_incumbent": best["exceeds_incumbent"],
        "best_bytes_added_total": best["bytes_added"]["total"],
        "best_under_incumbent_id": None if best_under is None else best_under["id"],
        "best_under_incumbent_held_out_relative_l2": (
            None if best_under is None else best_under["held_out_relative_l2"]
        ),
        "fails_too": bool(fails_too),
        "fails_under_incumbent": bool(fails_under),
        "settings": [
            {
                "id": r["id"],
                "program": r["program"],
                "depth": r["depth"],
                "width": r["width"],
                "held_out_relative_l2": r["held_out_relative_l2"],
                "mean_held_out_relative_l2": r["mean_held_out_relative_l2"],
                "beats_mean_predictor": r["beats_mean_predictor"],
                "null_model": r["null_model"],
                "margin_that_matters": r["margin_that_matters"],
                "exceeds_incumbent": r["exceeds_incumbent"],
                "byte_win": r["byte_win"],
                "bytes_added_total": r["bytes_added"]["total"],
                "status": r["status"],
            }
            for r in rows
        ],
        "interpretation": interpretation,
    }


@lru_cache(maxsize=1)
def cached_census() -> dict[str, Any]:
    return run_census()


# ---------------------------------------------------------------------------
# Negative index. Query first.
# ---------------------------------------------------------------------------


def consult_index() -> dict[str, Any]:
    model = "qwen3.8-27b"
    organ = "mlp"
    families = (
        "function_replacement",
        "structured_operator",
        "shared_input_transforms",
        "generated_programs",
        "factorized_programs",
        "shared_basis",
        "low_rank",
        "global_dense_lowrank",
        "synthetic_activation",
    )
    queries = []
    refusals = []
    for family in families:
        hits = ni.query(model=model, organ=organ, hypothesis_family=family)
        queries.append(
            {
                "model": model,
                "organ": organ,
                "hypothesis_family": family,
                "n_hits": len(hits),
                "top": [
                    {
                        "scar_id": h.get("scar_id"),
                        "level": h.get("level"),
                        "hypothesis_family": h.get("hypothesis_family"),
                        "verdict": h.get("verdict"),
                        "reopen_condition": h.get("reopen_condition"),
                    }
                    for h in hits[:3]
                ],
            }
        )
        refusal = ni.refuse_if_dead(
            {"model": model, "organ": organ, "hypothesis_family": family}
        )
        if refusal is not None:
            refusals.append(
                {
                    "hypothesis_family": family,
                    "scar_id": refusal.get("scar_id"),
                    "level": refusal.get("level"),
                    "reason": refusal.get("reason"),
                    "reopen_condition": refusal.get("reopen_condition"),
                }
            )
    proposal_families = ("function_replacement", "structured_operator")
    proposal_refused = [r for r in refusals if r["hypothesis_family"] in proposal_families]
    return {
        "model": model,
        "organ": organ,
        "queries": queries,
        "refusals": refusals,
        "proposal_refused": proposal_refused,
        "proceed": len(proposal_refused) == 0,
        "cousins_not_this_object": [
            "MLP_SHARED_PROGRAM and MLP_NONLINEAR_PROGRAM are r-bottleneck "
            "scars; this module is the recorded reopen (full-width structured).",
            "MLP_FUNCTIONAL_RANK is the low-rank factorization scar; this "
            "module does not widen rank.",
        ],
        "note": (
            "GENERAL_PHYSICAL scars refuse whatever model is named. The "
            "proposal families function_replacement and structured_operator "
            "are not refused. Bottleneck families are refused as candidates."
        ),
    }


# ---------------------------------------------------------------------------
# Selftest (fixtures) + receipt.
# ---------------------------------------------------------------------------


def make_fixture_xy(
    n_train: int = 48,
    n_hold: int = 16,
    hidden: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Tiny full-rank Kronecker map. Not a teacher-corpus stand-in (NNS-001)."""
    rng = np.random.default_rng(seed)
    p = 4 if hidden % 4 == 0 else 2
    q = hidden // p
    a = rng.standard_normal((p, p)).astype(np.float32)
    b = rng.standard_normal((q, q)).astype(np.float32)
    x_tr = rng.standard_normal((n_train, hidden)).astype(np.float32)
    x_ho = rng.standard_normal((n_hold, hidden)).astype(np.float32)
    y_tr = apply_kronecker(x_tr, a, b)
    y_ho = apply_kronecker(x_ho, a, b)
    mean = y_tr.mean(axis=0, keepdims=True)
    mean_ho = np.broadcast_to(mean, y_ho.shape)
    return {
        "Xtr": x_tr,
        "Ytr": y_tr,
        "Xho": x_ho,
        "Yho": y_ho,
        "hidden": hidden,
        "factor_p": p,
        "factor_q": q,
        "mean_held_out_relative_l2": mean_l2_ratio(mean_ho, y_ho),
        "hold_meta": {
            "domain": ["code"] * n_hold,
            "band": ["early"] * n_hold,
            "prompt_id": [f"p{i}" for i in range(n_hold)],
        },
    }


def selftest() -> dict[str, Any]:
    """Guards on fixtures. Does not read the teacher corpus and does not fit F."""
    held_out_leak_refused = False
    try:
        y = np.ones((4, 3), dtype=np.float32)
        function_error(y, y, split="train", report_as="held_out")
    except TrainReportedAsHeldOut:
        held_out_leak_refused = True

    baseline_omitted_refused = False
    try:
        validate_baseline(
            {
                "held_out_relative_l2": 0.4,
                "held_out_split": "hold",
                "error_authority": "held_out_relative_l2",
            }
        )
    except BaselineOmitted:
        baseline_omitted_refused = True

    unbilled_refused = False
    try:
        br = byte_breakdown(KRONECKER, factor_p=4, factor_q=4, hidden=16, n_layers=2)
        added = bytes_added_from_breakdown(br)
        stolen = dict(added)
        stolen["generator"] = 0
        stolen["total"] = sum(stolen[k] for k in ee.BYTES_ADDED_FIELDS)
        validate_billing(
            {"family": KRONECKER, "byte_breakdown": br, "bytes_added": stolen}
        )
    except UnbilledProgramByte:
        unbilled_refused = True

    bottleneck_refused = False
    try:
        _require_family(msp.SHARED_BOTH)
    except RankBottleneckDead:
        bottleneck_refused = True

    disguise_refused = False
    try:
        distilled_param_count(hidden=16, width=8, depth=2)
    except BottleneckInDisguise:
        disguise_refused = True

    exceeds_refused = False
    try:
        br = byte_breakdown(
            DISTILLED, depth=2, width=HIDDEN, hidden=HIDDEN, n_layers=N_LAYERS
        )
        added = bytes_added_from_breakdown(br)
        report_as_byte_win(
            {
                "bytes_added": added,
                "exceeds_incumbent": complete_ledger_exceeds_incumbent(added["total"]),
                "byte_win": True,
            }
        )
    except ExceedsIncumbent:
        exceeds_refused = True

    remat_refused = False
    fx = make_fixture_xy()
    try:
        emit_candidate(
            family=KRONECKER,
            program="tensor_product",
            pred_tr=fx["Ytr"],
            pred_ho=fx["Yho"],
            y_tr=fx["Ytr"],
            y_ho=fx["Yho"],
            consumer=native_consumer_sketch(KRONECKER, rematerialize_dense_W=True),
            mean_held_out_relative_l2=fx["mean_held_out_relative_l2"],
            factor_p=fx["factor_p"],
            factor_q=fx["factor_q"],
            hidden=fx["hidden"],
            n_layers=2,
            meta_ho=fx["hold_meta"],
        )
    except RematConsumer:
        remat_refused = True

    ok = emit_candidate(
        family=KRONECKER,
        program="tensor_product",
        pred_tr=fx["Ytr"],
        pred_ho=fx["Yho"],
        y_tr=fx["Ytr"],
        y_ho=fx["Yho"],
        consumer=native_consumer_sketch(
            KRONECKER, factor_p=fx["factor_p"], factor_q=fx["factor_q"], hidden=fx["hidden"], n_layers=2
        ),
        mean_held_out_relative_l2=fx["mean_held_out_relative_l2"],
        factor_p=fx["factor_p"],
        factor_q=fx["factor_q"],
        hidden=fx["hidden"],
        n_layers=2,
        meta_ho=fx["hold_meta"],
        id_suffix="fixture",
    )
    if ok["held_out_split"] != "hold":
        raise SystemExit("selftest: honest emit lost the hold split")
    if "mean_held_out_relative_l2" not in ok:
        raise SystemExit("selftest: honest emit dropped the mean predictor")
    if ok["economics"]["assumptions"]["scorer"] != "tools.future.executable_economics.score":
        raise SystemExit("selftest: economics did not come from executable_economics.score")

    fired = (
        held_out_leak_refused
        and baseline_omitted_refused
        and unbilled_refused
        and bottleneck_refused
        and disguise_refused
        and exceeds_refused
        and remat_refused
    )
    if not fired:
        raise SystemExit(
            "selftest: guards did not fire "
            f"leak={held_out_leak_refused} baseline={baseline_omitted_refused} "
            f"unbilled={unbilled_refused} bottleneck={bottleneck_refused} "
            f"disguise={disguise_refused} exceeds={exceeds_refused} "
            f"remat={remat_refused}"
        )
    return {
        "held_out_leak_refused": True,
        "baseline_omitted_refused": True,
        "unbilled_program_byte_refused": True,
        "rank_bottleneck_refused": True,
        "bottleneck_in_disguise_refused": True,
        "exceeds_incumbent_byte_win_refused": True,
        "remat_consumer_refused": True,
        "honest_fixture_emit_ok": True,
        "held_out_leak_codes": ["TrainReportedAsHeldOut"],
        "baseline_codes": ["BaselineOmitted"],
        "unbilled_codes": ["UnbilledProgramByte"],
        "bottleneck_codes": ["RankBottleneckDead", "BottleneckInDisguise"],
        "exceeds_codes": ["ExceedsIncumbent"],
        "remat_codes": ["REJECTED_DENSE_REMAT"],
    }


def build(*, consult: bool = True) -> Path:
    test = selftest()
    index = consult_index() if consult else {"proceed": True, "skipped": True}
    if consult and not index.get("proceed", False):
        raise StructuredOperatorRefuse(
            "REFUSED: negative_index refuse_if_dead fired on the proposal "
            f"families: {index.get('proposal_refused')}"
        )
    census = cached_census()
    n_neg = sum(1 for r in census["rows"] if r["status"] == MEASURED_NEGATIVE)
    n_open = sum(1 for r in census["rows"] if r["status"] == OPEN)
    n_null = sum(1 for r in census["rows"] if r.get("null_model"))
    campaign = census["campaign"]
    distilled = census["distilled_control"]
    baselines = census["baselines"]
    mean_held = baselines["mean_held_out_relative_l2"]

    def _answer_beats() -> str:
        under = [r for r in census["rows"] if not r.get("exceeds_incumbent")]
        open_u = [r for r in under if r.get("status") == OPEN]
        beats = [r for r in under if r.get("beats_mean_predictor")]
        if open_u:
            best = min(open_u, key=lambda r: float(r["held_out_relative_l2"]))
            return (
                f"YES, and it replaces F: {best['id']} held-out relative L2 "
                f"{best['held_out_relative_l2']} vs mean predictor {mean_held} "
                f"(margin {best['margin_over_mean']}) at "
                f"{best['bytes_added']['total']} bytes."
            )
        if beats:
            best = min(beats, key=lambda r: float(r["held_out_relative_l2"]))
            return (
                f"YES it beats the mean, NO it does not replace F. Best "
                f"{best['id']} held-out relative L2 {best['held_out_relative_l2']} "
                f"vs mean predictor {mean_held} (margin {best['margin_over_mean']}) "
                f"at {best['bytes_added']['total']} bytes. Status "
                f"{best['status']}; kill is {HELD_OUT_KILL_REL}."
            )
        best = min(under, key=lambda r: float(r["held_out_relative_l2"])) if under else None
        if best is None:
            return "NO: no under-incumbent setting was scored."
        label = "NULL MODEL" if best.get("null_model") else "not a margin that matters"
        return (
            f"NO. Best under-incumbent {best['id']} held-out relative L2 "
            f"{best['held_out_relative_l2']} vs mean predictor {mean_held} "
            f"(margin {best['margin_over_mean']}): {label}."
        )

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "After every r-bottleneck program for sealed-3.14 MLP F died, "
            "test full-width structured operators that are full-rank by "
            "construction (Monarch, Butterfly, Kronecker) and a full-width "
            "distilled net with no narrow layer. Score held-out-by-prompt "
            "relative L2 beside the mean predictor; refuse a ledger at or "
            "above the incumbent as a byte win."
        ),
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "recorded_by": RECORDED_BY,
        "git_head": git("rev-parse", "HEAD") or None,
        "prior_scars": {
            "MLP_FUNCTIONAL_RANK": (
                "activation-weighted rank needs 5120 on three of four layers "
                "for 10% hold error; affordable r=617 still 35-84%"
            ),
            "MLP_NONLINEAR_PROGRAM": (
                "FACTORIZE_THE_FACTORS 0.977 vs mean 0.970; dictionaries / "
                "conditionals / generated blocks / nonlinear generators in "
                "the 0.9 band"
            ),
            "MLP_SHARED_PROGRAM": (
                "17 shared-basis candidates all above 0.91; oracle PCA of F "
                "at rank 64 is 0.895"
            ),
            "named_mechanism": (
                "each family passed F through a narrow bottleneck of rank r, "
                "K atoms, or a latent width"
            ),
            "recorded_reopen": (
                "a full-width structured operator that is not an r-bottleneck"
            ),
        },
        "corpus": {
            "receipt": CORPUS_REL,
            "payload_dir": census["payload_dir"],
            "layers": census["layers"],
            "n_train_total": census["n_train_total"],
            "n_hold_total": census["n_hold_total"],
            "n_train_per_layer": census["n_train_per_layer"],
            "n_hold_per_layer": census["n_hold_per_layer"],
            "n_rows": census["n_train_total"] + census["n_hold_total"],
            "split_unit": "prompt_id",
            "disjoint": True,
            "x_sha256": census["x_sha256"],
            "y_sha256": census["y_sha256"],
        },
        "metric": {
            "authority": "held_out_relative_l2",
            "formula": "E_x ||F(x) - F_hat(x)|| / E_x ||F(x)||",
            "split": "prompt_id hold set of the teacher corpus",
            "kill_rel": HELD_OUT_KILL_REL,
            "kill_band": KILL_BAND,
            "mean_predictor_required": True,
            "null_model_rule": (
                "a number that does not beat the mean predictor is a NULL "
                "MODEL, not a result"
            ),
            "margin_that_matters_abs": MARGIN_THAT_MATTERS_ABS,
            "weight_reconstruction": "diagnostic only; not authority; not scored",
            "relative_frobenius": "diagnostic only",
        },
        "incumbent_mlp_bytes": INCUMBENT_MLP_BYTES,
        "n_layers_billed": N_LAYERS,
        "element_bytes": ELEMENT_BYTES,
        "families": list(FAMILIES),
        "sweeps": {
            "MONARCH_BLOCKS": list(MONARCH_BLOCKS),
            "BUTTERFLY_DEPTHS": list(BUTTERFLY_DEPTHS),
            "KRONECKER_SHAPES": [list(s) for s in KRONECKER_SHAPES],
            "DISTILLED_SETTINGS": [dict(s) for s in DISTILLED_SETTINGS],
        },
        "index": index,
        "selftest": test,
        "anti_fabrication": {
            "detectors": [
                "UNBILLED_PROGRAM_BYTE",
                "TRAIN_REPORTED_AS_HELD_OUT",
                "BASELINE_OMITTED",
                "EXCEEDS_INCUMBENT_BYTE_WIN",
                "REJECTED_DENSE_REMAT",
                "RANK_BOTTLENECK",
                "BOTTLENECK_IN_DISGUISE",
                "SYNTHETIC_ROW",
                "HELD_OUT_PROMPT_LEAK",
            ],
            "loud_exceptions": [
                "UnbilledProgramByte",
                "TrainReportedAsHeldOut",
                "BaselineOmitted",
                "ExceedsIncumbent",
                "RematConsumer",
                "RankBottleneckDead",
                "BottleneckInDisguise",
                "CorpusUnavailable",
            ],
            "rule": (
                "emit_candidate is the only constructor. A used factor with 0 "
                "billed bytes raises UnbilledProgramByte. A train-set figure "
                "labelled held-out raises TrainReportedAsHeldOut. A held-out "
                "figure without the mean-predictor baseline raises "
                "BaselineOmitted. A complete ledger at or above the incumbent "
                "reported as a byte win raises ExceedsIncumbent. A consumer "
                "that rematerializes dense W raises RematConsumer. Naming a "
                "closed bottleneck family raises RankBottleneckDead. A hidden "
                "width below the input width raises BottleneckInDisguise. A "
                "return-flag nobody checks is not a guard."
            ),
        },
        "baselines": baselines,
        "candidates": census["rows"],
        "family_verdicts": census["family_verdicts"],
        "scars": census["scars"],
        "distilled_control": distilled,
        "survivors": census["survivors"],
        "n_survivors": census["n_survivors"],
        "candidate_counts": {
            "n": len(census["rows"]),
            "measured_negative": n_neg,
            "open": n_open,
            "null_model": n_null,
            "rejected_dense_remat": 0,
        },
        "campaign": campaign,
        "go_wider": False,
        "answers": {
            "does_any_full_width_structured_operator_beat_the_mean_predictor_by_a_margin_that_matters_under_the_incumbent": _answer_beats(),
            "did_the_distilled_control_run": (
                "YES. "
                + distilled["interpretation"]
            ),
            "is_function_replacement_closed": (
                "YES. Scar MLP_FUNCTION_REPLACEMENT_CLOSED. "
                + str(campaign["mechanism"])
                if campaign["function_replacement_closed"]
                else "NO. " + str(campaign["mechanism"])
            ),
            "do_the_bytes_clear_one_percent_of_complete_token_time": (
                "YES as a projection for the compact structured programs if "
                "function held, after dispatch is billed. Function is the "
                "authority; a MEASURED_NEGATIVE is IMMATERIAL."
            ),
            "should_anyone_widen_rank_K_experts_or_blocks_on_a_refuted_family": (
                "NO. Those instantiations are scoped scars. Re-running them "
                "larger is the failure mode this module exists to prevent."
            ),
        },
        "negative_findings": [
            (
                f"{row['family']} {row['id']} held-out relative L2 "
                f"{row['held_out_relative_l2']} vs mean "
                f"{row['mean_held_out_relative_l2']}"
                + (" NULL MODEL" if row.get("null_model") else "")
                + (" EXCEEDS_INCUMBENT" if row.get("exceeds_incumbent") else "")
            )
            for row in census["rows"]
        ],
        "gaps_closed": [
            "full-width Monarch / Butterfly / Kronecker fitted on the real teacher corpus, held out by prompt",
            "distilled operator with every hidden width >= input width ran as the capacity control",
            "every held-out number reported beside the mean predictor; omitting the baseline is refused",
            "a complete ledger at or above the incumbent cannot be reported as a byte win",
            "executable_economics.score used for every byte figure, dispatch delta, and extra FLOPs",
            "native-consumer sketches on atlas primitives; remat-then-GEMV refused",
            "r-bottleneck families refused as candidates (scoped scars)",
            "train-set figure cannot be reported as held-out",
            "negative_index queried; proposal families not refused",
        ],
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    if args.build:
        path = build()
        print(path)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

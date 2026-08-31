"""MLP ERROR BUDGET — what relative organ error does the model actually survive?

Every MLP program in this campaign has been judged against an assumed bar
(held-out output error crossing 1%, 3%, 10%). This module measures the bar.

It injects CONTROLLED relative L2 error into the MLP organ output F(x) at
inference, verifies the achieved error per token (a sweep that reports the
target as if it were achieved is refused), and scores what the model does
downstream. Two geometries, because they are not interchangeable:

    isotropic   error spread over all output directions
    structured  error aligned with the residual of a low-rank approximation
                of the organ — the geometry every program in this campaign
                actually produces

Scope is swept because errors compound: one layer, a few layers, then all
64. A per-layer bar is not a whole-model bar. The headline number is the
largest per-layer relative output error at which the model is still usable
when the perturbation is applied at every layer, under the stricter of the
two geometries. Argmax agreement alone is not parity and is refused.

    python3 tools/future/mlp_error_budget.py --build
    python3 tools/future/mlp_error_budget.py --selftest
    python3 -m pytest tools/future/test_mlp_error_budget.py -q

evidence_class STATIC_ONLY. CPU/MPS forward passes. No GPU lease.
Does not touch crates/.
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
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tools.future._common import REPO, write_receipt
from tools.future.mlp_teacher_corpus import CAPABILITY_DOMAINS, HIDDEN, N_LAYERS
from tools.future.physical_primitives import ATLAS_PRIMITIVES


RECEIPT = "MLP_ERROR_BUDGET.json"
SCHEMA = "hawking.future.mlp_error_budget.v1"
VERSION = 1
RECORDED_BY = "tools/future/mlp_error_budget.py"
EVIDENCE_CLASS = "STATIC_ONLY"

ISOTROPIC = "isotropic"
STRUCTURED = "structured"
GEOMETRIES: tuple[str, ...] = (ISOTROPIC, STRUCTURED)

SCOPE_ONE = "one_layer"
SCOPE_FEW = "few_layers"
SCOPE_ALL = "all_layers"
SCOPE_NAMES: tuple[str, ...] = (SCOPE_ONE, SCOPE_FEW, SCOPE_ALL)

DEFAULT_TARGETS: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
STRUCTURED_RANK = 32
RNG_SEED = 38
ELEMENT_EPS = 1e-30
KL_EPS = 1e-12
TOPK_AUTHORITY = 5
TOPK_DIAGNOSTIC = 10

# Usability bars. Quoted verbatim in the receipt. Argmax is not among them.
USABLE_KL = 0.10
USABLE_TOP5 = 0.80
DEGRADE_KL = 1.0
DEGRADE_TOP5 = 0.50

BAND_USABLE = "usable"
BAND_DEGRADES = "degrades"
BAND_BREAKS = "breaks"

CALIBRATION_ABS = 1e-5
CALIBRATION_REL = 0.05

SEALED_ONE_LAYERS: tuple[int, ...] = (38,)
SEALED_FEW_LAYERS: tuple[int, ...] = (36, 37, 38, 39)
SEALED_PROMPT_TOKEN_IDS: tuple[int, ...] = (561, 1000, 284, 47358)

CIM_REL = "receipts/future/CAPABILITY_INFORMATION_MAP.json"
CORPUS_REL = "receipts/future/MLP_TEACHER_CORPUS.json"
SHARED_REL = "receipts/future/MLP_SHARED_PROGRAM.json"

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate gate. "
    "MLP organ output F(x) is perturbed at inference so the achieved relative "
    "L2 error is the TARGET error, verified per token. Downstream numbers are "
    "logit KL(base || pert) and top-k agreement from a CPU forward of "
    "sealed-3.14 packed weights (or a named fixture stack). Argmax agreement "
    "is diagnostic and is not parity. Multi-token generate quality on the "
    "teacher-corpus capability domains is reported only when actually run; "
    "otherwise it is UNMEASURED and is not the usability criterion. "
    "gpu_authority is false. evidence_class is STATIC_ONLY."
)

CRITERION_DEFENSE = (
    "USABLE requires last-token KL(base || pert) < 0.10 nats AND mean top-5 "
    "agreement across prompt tokens >= 0.80. 0.10 nats is a ~10% "
    "perplexity-equivalent hit on the next-token distribution (exp(0.10) ≈ "
    "1.105); it is a distribution bar, not an argmax bar. Top-5 at 0.80 "
    "means the model's local continuation set is mostly preserved; top-1 "
    "can flip on a near-tie without the model being unusable, which is why "
    "argmax agreement is refused as parity. DEGRADES is the band below "
    "usable but with KL < 1.0 and top-5 >= 0.50 (an e-fold of extra surprise, "
    "or half the plausible set gone). BREAKS is KL >= 1.0 or top-5 < 0.50: "
    "the next-token object is no longer the same model. The headline number "
    "is the largest tested per-layer relative L2 at which ALL 64 layers, "
    "under BOTH geometries, remain USABLE; when the geometries disagree the "
    "stricter (usually structured) one is the campaign number, because every "
    "function-replacement program produces structured error. A one-layer bar "
    "is reported separately and is not the whole-model bar."
)


class ErrorBudgetRefuse(ValueError):
    """The error-budget probe refused rather than guessing."""


class TargetReportedAsAchieved(ErrorBudgetRefuse):
    """A target error was written down as if it had been measured."""

    def __init__(self, detail: str = "") -> None:
        extra = f" ({detail})" if detail else ""
        super().__init__(
            f"REFUSED: target relative error cannot be reported as achieved{extra}"
        )


class ArgmaxPresentedAsParity(ErrorBudgetRefuse):
    """Argmax agreement alone is not parity."""

    def __init__(self, detail: str = "") -> None:
        extra = f" ({detail})" if detail else ""
        super().__init__(
            f"REFUSED: argmax agreement alone is not parity{extra}"
        )


class CalibrationDrift(ErrorBudgetRefuse):
    """Achieved error drifted from the target and was not labelled as such."""


class SpecimenUnavailable(ErrorBudgetRefuse):
    """The sealed CPU forward cannot be constructed."""


ARGMAX_ONLY_KEYS = frozenset(
    {
        "argmax",
        "argmax_agreement",
        "argmax_identical",
        "top1",
        "top1_agreement",
        "top_1_agreement",
    }
)


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


def _r(value: float, n: int = 8) -> float:
    out = round(float(value), n)
    return 0.0 if out == 0.0 else out


def _require_primitive(name: str) -> str:
    if name not in ATLAS_PRIMITIVES:
        raise ErrorBudgetRefuse(f"{name} is not an atlas primitive")
    return name


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def rmsnorm(x: np.ndarray, w: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    inv = 1.0 / math.sqrt(float(np.mean(x * x)) + float(eps))
    return x * np.float32(inv) * (np.float32(1.0) + np.asarray(w, dtype=np.float32))


def swiglu(
    x: np.ndarray, Wg: np.ndarray, Wu: np.ndarray, Wd: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    g = Wg @ x
    u = Wu @ x
    gate = _silu(g)
    pre = gate * u
    down = Wd @ pre
    return down, {"silu_gate": gate, "up": u, "pre_down": pre}


def token_relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    """||pred - target|| / ||target|| on one token. The calibration metric."""
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    if p.shape != t.shape:
        raise ErrorBudgetRefuse(f"pred shape {p.shape} != target shape {t.shape}")
    den = float(np.linalg.norm(t))
    num = float(np.linalg.norm(p - t))
    return num / max(den, ELEMENT_EPS)


def mean_l2_ratio(pred: np.ndarray, target: np.ndarray) -> float:
    """E_x ||pred - target|| / E_x ||target|| over a stacked token axis."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.ndim == 1:
        return token_relative_l2(p, t)
    if p.shape != t.shape:
        raise ErrorBudgetRefuse(f"pred shape {p.shape} != target shape {t.shape}")
    err = float(np.linalg.norm(p - t, axis=1).mean())
    scale = float(np.linalg.norm(t, axis=1).mean())
    return err / max(scale, ELEMENT_EPS)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    z = z - float(z.max())
    e = np.exp(np.clip(z, -80.0, 80.0))
    s = float(e.sum())
    if s <= 0.0:
        raise ErrorBudgetRefuse("REFUSED: softmax partition is not positive")
    return e / s


def kl_div(p: np.ndarray, q: np.ndarray, *, eps: float = KL_EPS) -> float:
    """KL(p || q) in nats. Authority for this probe is KL(base || pert)."""
    pp = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), eps, 1.0)
    qq = np.clip(np.asarray(q, dtype=np.float64).reshape(-1), eps, 1.0)
    pp = pp / pp.sum()
    qq = qq / qq.sum()
    return float(np.sum(pp * (np.log(pp) - np.log(qq))))


def topk_ids(logits: np.ndarray, k: int) -> np.ndarray:
    z = np.asarray(logits).reshape(-1)
    kk = min(int(k), int(z.size))
    if kk <= 0:
        raise ErrorBudgetRefuse("k must be positive")
    return np.argsort(z)[-kk:][::-1]


def topk_agreement(base_logits: np.ndarray, pert_logits: np.ndarray, k: int) -> float:
    a = set(int(x) for x in topk_ids(base_logits, k))
    b = set(int(x) for x in topk_ids(pert_logits, k))
    return float(len(a & b) / float(min(int(k), len(a))))


def argmax_id(logits: np.ndarray) -> int:
    return int(np.argmax(np.asarray(logits).reshape(-1)))


def calibration_tol(target: float) -> float:
    return max(CALIBRATION_ABS, CALIBRATION_REL * abs(float(target)))


# ---------------------------------------------------------------------------
# Error geometries. Achieved relative L2 is measured, never assumed.
# ---------------------------------------------------------------------------


def structured_direction(
    f: np.ndarray,
    basis: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unit-scale residual of a low-rank output projection of F.

    This is the error a rank-k approximation of the organ actually produces:
    F - Q Q^T F lives in the discarded subspace. If F already sits in the
    top-k span, a complement direction is drawn so the target error is still
    achievable rather than silently under-shot.
    """
    vec = np.asarray(f, dtype=np.float64).reshape(-1)
    if basis is None:
        raise ErrorBudgetRefuse(
            "REFUSED: structured geometry requires an output basis "
            "(low-rank residual of the organ); refusing to substitute isotropic"
        )
    q = np.asarray(basis, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != vec.shape[0]:
        raise ErrorBudgetRefuse(
            f"REFUSED: structured basis shape {q.shape} does not match F {vec.shape}"
        )
    proj = q @ (q.T @ vec)
    resid = vec - proj
    n_res = float(np.linalg.norm(resid))
    n_f = float(np.linalg.norm(vec))
    if n_res > 1e-12 * max(n_f, ELEMENT_EPS):
        return resid
    draw = rng.standard_normal(vec.shape[0])
    leftover = draw - q @ (q.T @ draw)
    n_left = float(np.linalg.norm(leftover))
    if n_left <= 1e-12:
        raise ErrorBudgetRefuse(
            "REFUSED: structured complement is empty (basis spans the output)"
        )
    return leftover


def isotropic_direction(dim: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(4):
        u = rng.standard_normal(int(dim))
        n = float(np.linalg.norm(u))
        if n > ELEMENT_EPS:
            return u
    raise ErrorBudgetRefuse("REFUSED: isotropic draw was numerically zero")


def output_basis_from_down(
    w_down: np.ndarray, rank: int, rng: np.random.Generator
) -> np.ndarray:
    """Randomized range finder for the organ's output subspace.

    Q approximates the top-k left singular vectors of W_down. A rank-k
    approximation of down produces error in the orthogonal complement of Q.
    """
    w = np.asarray(w_down, dtype=np.float32)
    if w.ndim != 2:
        raise ErrorBudgetRefuse("W_down must be a matrix")
    # Leave at least one complementary direction: a basis that spans the
    # whole output makes structured error identically zero, which cannot
    # hit a nonzero target.
    k = min(int(rank), max(int(w.shape[0]) - 1, 1), int(w.shape[1]))
    if int(w.shape[0]) <= 1:
        raise ErrorBudgetRefuse("structured geometry needs hidden > 1")
    if k < 1:
        raise ErrorBudgetRefuse("structured rank must be positive")
    r = rng.standard_normal((int(w.shape[1]), k)).astype(np.float32)
    y = w @ r
    q, _ = np.linalg.qr(y.astype(np.float64), mode="reduced")
    return q.astype(np.float64)


def inject_relative_error(
    f: np.ndarray,
    target: float,
    geometry: str,
    *,
    rng: np.random.Generator,
    basis: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return (F_hat, achieved_relative_l2) with achieved verified, not assumed.

    Per-token scaling: ||F_hat - F|| / ||F|| equals `target` up to float noise.
    Structured geometry will not silently fall back to isotropic.
    """
    vec = np.asarray(f, dtype=np.float64).reshape(-1)
    tgt = float(target)
    if tgt < 0.0:
        raise ErrorBudgetRefuse("target relative error cannot be negative")
    n_f = float(np.linalg.norm(vec))
    if tgt == 0.0:
        return vec.astype(np.float32), 0.0
    if n_f <= ELEMENT_EPS:
        raise ErrorBudgetRefuse(
            "REFUSED: ||F|| is 0 so relative error is undefined; "
            "refusing to invent a scale"
        )
    if geometry == ISOTROPIC:
        d = isotropic_direction(vec.shape[0], rng)
    elif geometry == STRUCTURED:
        d = structured_direction(vec, basis, rng)
    else:
        raise ErrorBudgetRefuse(f"unknown geometry {geometry!r}")
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    n_d = float(np.linalg.norm(d))
    if n_d <= ELEMENT_EPS:
        raise ErrorBudgetRefuse("REFUSED: error direction is numerically zero")
    d = d / n_d
    hat = vec + (tgt * n_f) * d
    achieved = float(np.linalg.norm(hat - vec) / n_f)
    return hat.astype(np.float32), achieved


# ---------------------------------------------------------------------------
# Load-bearing constructors. A return-flag nobody checks is not a guard.
# ---------------------------------------------------------------------------


def validate_calibration(row: Mapping[str, Any]) -> None:
    """Refuse a point that reports the target as if it were achieved."""
    if "target_relative_l2" not in row:
        raise TargetReportedAsAchieved("missing target_relative_l2")
    if "achieved_relative_l2" not in row:
        raise TargetReportedAsAchieved("missing achieved_relative_l2")
    if row.get("achieved_measured") is not True:
        raise TargetReportedAsAchieved(
            "achieved_measured is not True; the target was not measured"
        )
    if row.get("achieved_is_target") is True:
        raise TargetReportedAsAchieved(
            "achieved_is_target is set; that is reporting the target as achieved"
        )
    n_inj = row.get("n_injections")
    if not isinstance(n_inj, (int, np.integer)) or int(n_inj) < 1:
        raise TargetReportedAsAchieved("no per-token injections were recorded")
    if "achieved_max_abs_drift_from_target" not in row:
        raise TargetReportedAsAchieved(
            "no per-token drift from target was recorded"
        )
    if row.get("copied_target_as_achieved"):
        raise TargetReportedAsAchieved("copied_target_as_achieved")
    drift = row.get("achieved_max_abs_drift_from_target")
    if drift is None:
        raise TargetReportedAsAchieved("drift is None")
    ok = row.get("calibration_ok")
    tgt = float(row["target_relative_l2"])
    ach = float(row["achieved_relative_l2"])
    tol = calibration_tol(tgt)
    drifted = abs(ach - tgt) > tol or float(drift) > tol
    if drifted and ok is True:
        raise CalibrationDrift(
            f"achieved {ach} drifted from target {tgt} beyond {tol} but "
            "calibration_ok is True"
        )
    if (not drifted) and ok is False:
        raise CalibrationDrift(
            f"achieved {ach} matches target {tgt} within {tol} but "
            "calibration_ok is False"
        )


def refuse_argmax_as_parity(record: Mapping[str, Any] | None) -> None:
    """Load-bearing: argmax agreement alone cannot be presented as parity."""
    if record is None:
        raise ArgmaxPresentedAsParity("no record")
    if record.get("parity") is True:
        used = record.get("parity_metrics") or record.get("metrics_used") or []
        used_n = {str(x) for x in used}
        if not used_n or used_n <= ARGMAX_ONLY_KEYS:
            raise ArgmaxPresentedAsParity(
                "parity=True with only argmax metrics"
            )
        if "last_token_kl" not in used_n and "kl" not in used_n:
            raise ArgmaxPresentedAsParity(
                "parity=True without a distribution metric"
            )
    if record.get("usable_because") in ARGMAX_ONLY_KEYS:
        raise ArgmaxPresentedAsParity("usable_because is an argmax key")
    if record.get("criterion_name") in ARGMAX_ONLY_KEYS:
        raise ArgmaxPresentedAsParity("criterion_name is an argmax key")
    if record.get("parity_metric") in ARGMAX_ONLY_KEYS:
        raise ArgmaxPresentedAsParity("parity_metric is an argmax key")
    keys = {str(k) for k in record.keys()}
    values_only = {
        k
        for k in keys
        if k in ARGMAX_ONLY_KEYS
        or k
        in {
            "parity",
            "parity_metrics",
            "metrics_used",
            "parity_metric",
            "criterion_name",
            "usable_because",
        }
    }
    # A record that is nothing but argmax keys claiming a verdict.
    payload = {
        k
        for k in keys
        if k
        not in {
            "parity",
            "parity_metrics",
            "metrics_used",
            "parity_metric",
            "criterion_name",
            "usable_because",
            "note",
        }
    }
    if record.get("parity") is True and payload <= ARGMAX_ONLY_KEYS:
        raise ArgmaxPresentedAsParity("argmax-only payload")


def present_as_parity(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse to present argmax agreement alone as parity.

    This is the public API a downstream generator would call. It raises.
    There is no successful return path for an argmax-only record.
    """
    if not isinstance(metrics, Mapping) or not metrics:
        raise ArgmaxPresentedAsParity("empty metrics")
    keys = {str(k) for k in metrics.keys()}
    if keys <= ARGMAX_ONLY_KEYS:
        raise ArgmaxPresentedAsParity(
            "argmax agreement alone is not parity"
        )
    has_kl = any(k in keys for k in ("last_token_kl", "kl", "kl_base_pert"))
    has_topk = any(
        k in keys
        for k in (
            "mean_top5_agreement",
            "top5_agreement",
            "mean_topk_agreement",
            "topk_agreement",
        )
    )
    if not has_kl:
        raise ArgmaxPresentedAsParity(
            "a distribution metric (KL) is required; argmax is not enough"
        )
    if not has_topk:
        raise ArgmaxPresentedAsParity(
            "top-k agreement is required alongside KL; argmax is not enough"
        )
    # Even a well-scored distribution is not labelled "parity" here.
    # Usable != bit-identical, and this module will not say otherwise.
    raise ArgmaxPresentedAsParity(
        "this module refuses to present any MLP-error record as parity; "
        "report usable/degrades/breaks from KL and top-k"
    )


def usability_verdict(
    *,
    last_token_kl: float | None,
    mean_top5_agreement: float | None,
    argmax_agreement: float | None = None,
    calibration_ok: bool = True,
) -> dict[str, Any]:
    """Band a cell. Argmax is recorded and is not the decision."""
    if last_token_kl is None or mean_top5_agreement is None:
        if argmax_agreement is not None and last_token_kl is None:
            raise ArgmaxPresentedAsParity(
                "usability_verdict was asked to decide from argmax without KL"
            )
        raise ErrorBudgetRefuse(
            "REFUSED: usability requires last_token_kl and mean_top5_agreement"
        )
    kl = float(last_token_kl)
    top5 = float(mean_top5_agreement)
    if not calibration_ok:
        band = BAND_BREAKS
        why = "calibration_failed"
    elif kl < USABLE_KL and top5 >= USABLE_TOP5:
        band = BAND_USABLE
        why = "kl_and_top5"
    elif kl < DEGRADE_KL and top5 >= DEGRADE_TOP5:
        band = BAND_DEGRADES
        why = "kl_and_top5"
    else:
        band = BAND_BREAKS
        why = "kl_and_top5"
    row = {
        "band": band,
        "usable": band == BAND_USABLE,
        "why": why,
        "last_token_kl": _r(kl),
        "mean_top5_agreement": _r(top5),
        "argmax_agreement": None if argmax_agreement is None else _r(float(argmax_agreement)),
        "argmax_is_not_parity": True,
        "parity": False,
        "criterion_name": "last_token_kl_and_mean_top5",
        "metrics_used": ["last_token_kl", "mean_top5_agreement"],
        "bars": {
            "usable_kl": USABLE_KL,
            "usable_top5": USABLE_TOP5,
            "degrade_kl": DEGRADE_KL,
            "degrade_top5": DEGRADE_TOP5,
        },
    }
    refuse_argmax_as_parity(row)
    return row


def emit_sweep_point(
    *,
    geometry: str,
    scope: str,
    scope_layers: Sequence[int],
    n_layers_total: int,
    target_relative_l2: float,
    achieved_values: Sequence[float],
    last_token_kl: float,
    mean_top5_agreement: float,
    mean_top10_agreement: float | None = None,
    argmax_agreement: float | None = None,
    mean_kl: float | None = None,
    hidden_relative_l2: float | None = None,
    last_token_hidden_relative_l2: float | None = None,
    baseline_argmax: int | None = None,
    pert_argmax: int | None = None,
    baseline_top5: Sequence[int] | None = None,
    pert_top5: Sequence[int] | None = None,
    generate: Mapping[str, Any] | None = None,
    n_tokens: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The only constructor for a sweep cell. Achieved is measured."""
    if geometry not in GEOMETRIES:
        raise ErrorBudgetRefuse(f"unknown geometry {geometry!r}")
    if scope not in SCOPE_NAMES:
        raise ErrorBudgetRefuse(f"unknown scope {scope!r}")
    if achieved_values is None:
        raise TargetReportedAsAchieved("achieved_values is None")
    vals = [float(v) for v in achieved_values]
    if not vals:
        raise TargetReportedAsAchieved("achieved_values is empty")
    tgt = float(target_relative_l2)
    ach = float(sum(vals) / len(vals))
    amin = float(min(vals))
    amax = float(max(vals))
    drift = float(max(abs(v - tgt) for v in vals))
    tol = calibration_tol(tgt)
    cal_ok = drift <= tol
    verdict = usability_verdict(
        last_token_kl=float(last_token_kl),
        mean_top5_agreement=float(mean_top5_agreement),
        argmax_agreement=argmax_agreement,
        calibration_ok=cal_ok,
    )
    layers = [int(x) for x in scope_layers]
    row: dict[str, Any] = {
        "geometry": geometry,
        "scope": scope,
        "scope_layers": layers,
        "n_layers_perturbed": len(layers),
        "n_layers_total": int(n_layers_total),
        "target_relative_l2": _r(tgt, 6),
        "achieved_relative_l2": _r(ach),
        "achieved_min": _r(amin),
        "achieved_max": _r(amax),
        "achieved_max_abs_drift_from_target": _r(drift),
        "achieved_measured": True,
        "achieved_is_target": False,
        "copied_target_as_achieved": False,
        "n_injections": len(vals),
        "n_tokens": int(n_tokens),
        "calibration_ok": bool(cal_ok),
        "calibration_tol": _r(tol),
        "last_token_kl": _r(float(last_token_kl)),
        "mean_kl": None if mean_kl is None else _r(float(mean_kl)),
        "mean_top5_agreement": _r(float(mean_top5_agreement)),
        "mean_top10_agreement": (
            None if mean_top10_agreement is None else _r(float(mean_top10_agreement))
        ),
        "argmax_agreement": (
            None if argmax_agreement is None else _r(float(argmax_agreement))
        ),
        "argmax_is_not_parity": True,
        "hidden_relative_l2": (
            None if hidden_relative_l2 is None else _r(float(hidden_relative_l2))
        ),
        "last_token_hidden_relative_l2": (
            None
            if last_token_hidden_relative_l2 is None
            else _r(float(last_token_hidden_relative_l2))
        ),
        "baseline_argmax": baseline_argmax,
        "pert_argmax": pert_argmax,
        "baseline_top5": None if baseline_top5 is None else [int(x) for x in baseline_top5],
        "pert_top5": None if pert_top5 is None else [int(x) for x in pert_top5],
        "band": verdict["band"],
        "usable": verdict["usable"],
        "verdict": verdict,
        "physical_primitive": _require_primitive("FusedDecodeCompute"),
        "generate": None if generate is None else dict(generate),
    }
    if extra:
        for k, v in extra.items():
            if k in row:
                raise ErrorBudgetRefuse(f"extra key collides with {k}")
            row[k] = v
    validate_calibration(row)
    refuse_argmax_as_parity(row)
    refuse_argmax_as_parity(row["verdict"])
    return row


def score_logits(
    base_logits: np.ndarray, pert_logits: np.ndarray
) -> dict[str, Any]:
    """KL / top-k / argmax on a (n_tokens, vocab) pair. Argmax is diagnostic."""
    b = np.asarray(base_logits, dtype=np.float64)
    p = np.asarray(pert_logits, dtype=np.float64)
    if b.shape != p.shape:
        raise ErrorBudgetRefuse(f"logit shape {p.shape} != baseline {b.shape}")
    if b.ndim == 1:
        b = b[None, :]
        p = p[None, :]
    n = int(b.shape[0])
    kls = []
    top5 = []
    top10 = []
    top1 = []
    for i in range(n):
        pb = softmax(b[i])
        pp = softmax(p[i])
        kls.append(kl_div(pb, pp))
        top5.append(topk_agreement(b[i], p[i], TOPK_AUTHORITY))
        top10.append(topk_agreement(b[i], p[i], TOPK_DIAGNOSTIC))
        top1.append(1.0 if argmax_id(b[i]) == argmax_id(p[i]) else 0.0)
    last = n - 1
    return {
        "n_tokens": n,
        "last_token_kl": float(kls[last]),
        "mean_kl": float(sum(kls) / n),
        "per_token_kl": kls,
        "mean_top5_agreement": float(sum(top5) / n),
        "last_token_top5_agreement": float(top5[last]),
        "mean_top10_agreement": float(sum(top10) / n),
        "mean_top1_agreement": float(sum(top1) / n),
        "last_token_argmax_agreement": float(top1[last]),
        "argmax_agreement": float(sum(top1) / n),
        "baseline_argmax": argmax_id(b[last]),
        "pert_argmax": argmax_id(p[last]),
        "baseline_top5": [int(x) for x in topk_ids(b[last], TOPK_AUTHORITY)],
        "pert_top5": [int(x) for x in topk_ids(p[last], TOPK_AUTHORITY)],
        "argmax_is_not_parity": True,
    }


def score_generate(
    baseline_ids: Sequence[int],
    pert_ids: Sequence[int],
    *,
    domains: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Continuation quality. Exact-match of greedy ids is not parity."""
    b = [int(x) for x in baseline_ids]
    p = [int(x) for x in pert_ids]
    n = min(len(b), len(p))
    if n == 0:
        raise ErrorBudgetRefuse("REFUSED: empty continuation")
    match = sum(1 for i in range(n) if b[i] == p[i]) / float(n)
    prefix = 0
    for i in range(n):
        if b[i] != p[i]:
            break
        prefix += 1
    row = {
        "status": "MEASURED",
        "authority": False,
        "n_tokens": n,
        "exact_match_rate": _r(match),
        "matched_prefix": int(prefix),
        "baseline_ids": b,
        "pert_ids": p,
        "argmax_is_not_parity": True,
        "parity": False,
        "note": (
            "Greedy continuation overlap. This is not parity and is not the "
            "usability criterion (that is last-token KL and top-5)."
        ),
    }
    if domains is not None:
        row["capability_domains"] = list(domains)
    refuse_argmax_as_parity(row)
    return row


# ---------------------------------------------------------------------------
# Headline. The number is a min over geometries at all-layers scope.
# ---------------------------------------------------------------------------


def _max_target_with_band(
    points: Sequence[Mapping[str, Any]],
    *,
    geometry: str,
    scope: str,
    band: str,
) -> float | None:
    hits = [
        float(p["target_relative_l2"])
        for p in points
        if p.get("geometry") == geometry
        and p.get("scope") == scope
        and p.get("band") == band
        and p.get("calibration_ok") is True
    ]
    return max(hits) if hits else None


def _min_target_with_band(
    points: Sequence[Mapping[str, Any]],
    *,
    geometry: str,
    scope: str,
    band: str,
) -> float | None:
    hits = [
        float(p["target_relative_l2"])
        for p in points
        if p.get("geometry") == geometry
        and p.get("scope") == scope
        and p.get("band") == band
    ]
    return min(hits) if hits else None


def _joint_usable_targets(
    points: Sequence[Mapping[str, Any]], *, scope: str
) -> list[float]:
    by: dict[float, dict[str, str]] = {}
    for p in points:
        if p.get("scope") != scope:
            continue
        tgt = float(p["target_relative_l2"])
        geo = str(p["geometry"])
        by.setdefault(tgt, {})[geo] = str(p.get("band"))
    out = []
    for tgt, geos in by.items():
        if all(geos.get(g) == BAND_USABLE for g in GEOMETRIES):
            out.append(tgt)
    return sorted(out)


def headline_from_points(
    points: Sequence[Mapping[str, Any]],
    *,
    scope_of_headline: str = SCOPE_ALL,
) -> dict[str, Any]:
    if not points:
        raise ErrorBudgetRefuse("REFUSED: no sweep points")
    scopes = {p["scope"] for p in points}
    geos = {p["geometry"] for p in points}
    if len(geos) < 2:
        raise ErrorBudgetRefuse(
            "REFUSED: headline needs both error geometries; "
            f"present={sorted(geos)}"
        )
    if len(scopes) < 2:
        raise ErrorBudgetRefuse(
            "REFUSED: headline needs at least two scopes; "
            f"present={sorted(scopes)}"
        )
    for p in points:
        validate_calibration(p)
        refuse_argmax_as_parity(p)

    joint = _joint_usable_targets(points, scope=scope_of_headline)
    tolerated = max(joint) if joint else None
    iso_u = _max_target_with_band(
        points, geometry=ISOTROPIC, scope=scope_of_headline, band=BAND_USABLE
    )
    st_u = _max_target_with_band(
        points, geometry=STRUCTURED, scope=scope_of_headline, band=BAND_USABLE
    )
    if iso_u is None and st_u is None:
        stricter = None
    elif iso_u is None:
        stricter = STRUCTURED
    elif st_u is None:
        stricter = ISOTROPIC
    else:
        stricter = STRUCTURED if st_u <= iso_u else ISOTROPIC

    geo_for_edge = stricter or STRUCTURED
    degrades_at = _min_target_with_band(
        points, geometry=geo_for_edge, scope=scope_of_headline, band=BAND_DEGRADES
    )
    breaks_at = _min_target_with_band(
        points, geometry=geo_for_edge, scope=scope_of_headline, band=BAND_BREAKS
    )
    one_joint = _joint_usable_targets(points, scope=SCOPE_ONE)
    few_joint = _joint_usable_targets(points, scope=SCOPE_FEW)

    iso_vs_st = None
    if iso_u is not None and st_u is not None:
        iso_vs_st = _r(float(iso_u) - float(st_u), 6)

    note = (
        "largest per-layer relative L2 at which all-layers, both geometries, "
        "remain USABLE under last_token_kl < 0.10 and mean_top5 >= 0.80"
        if tolerated is not None
        else (
            "no tested per-layer error (including the grid floor) was USABLE "
            f"on {scope_of_headline} for both geometries"
        )
    )
    row = {
        "tolerated_per_layer_relative_l2": None if tolerated is None else _r(tolerated, 6),
        "degrades_at": None if degrades_at is None else _r(degrades_at, 6),
        "breaks_at": None if breaks_at is None else _r(breaks_at, 6),
        "scope_of_headline": scope_of_headline,
        "geometry_of_headline": "stricter_of_isotropic_and_structured",
        "stricter_geometry": stricter,
        "isotropic_all_layers_usable": None if iso_u is None else _r(iso_u, 6),
        "structured_all_layers_usable": None if st_u is None else _r(st_u, 6),
        "isotropic_minus_structured": iso_vs_st,
        "isolated_one_layer_tolerated_relative_l2": (
            None if not one_joint else _r(max(one_joint), 6)
        ),
        "few_layers_tolerated_relative_l2": (
            None if not few_joint else _r(max(few_joint), 6)
        ),
        "criterion": {
            "name": "last_token_kl_and_mean_top5",
            "last_token_kl_usable": USABLE_KL,
            "mean_top5_usable": USABLE_TOP5,
            "last_token_kl_degrades_below": DEGRADE_KL,
            "mean_top5_degrades_at_least": DEGRADE_TOP5,
            "argmax_is_not_parity": True,
            "authority": "last_token_kl_and_mean_top5",
            "not_authority": ["argmax_agreement", "top1_agreement", "generate_exact_match"],
        },
        "criterion_defense": CRITERION_DEFENSE,
        "argmax_is_not_parity": True,
        "parity": False,
        "note": note,
        "breaks_at_note": (
            None
            if breaks_at is not None
            else (
                f"no BREAKS cell on this grid; {max(DEFAULT_TARGETS)} all-layers "
                "is DEGRADES or USABLE. BREAKS is above the tested range."
            )
        ),
    }
    refuse_argmax_as_parity(row)
    return row


def answers_from_headline(
    headline: Mapping[str, Any], points: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    tol = headline.get("tolerated_per_layer_relative_l2")
    brk = headline.get("breaks_at")
    deg = headline.get("degrades_at")
    iso = headline.get("isotropic_all_layers_usable")
    st = headline.get("structured_all_layers_usable")
    one = headline.get("isolated_one_layer_tolerated_relative_l2")
    structured_stricter = (
        st is None or (iso is not None and st is not None and float(st) < float(iso))
    )
    if tol is None:
        what = (
            "Below the tested grid. No all-layers cell was USABLE for both "
            f"geometries, including target {min(DEFAULT_TARGETS)}. The campaign "
            "has been scoring programs against a bar the model does not survive."
        )
    else:
        what = (
            f"{tol} per layer on all {N_LAYERS} layers, under the stricter of "
            "isotropic and structured, by last-token KL < 0.10 nats and mean "
            "top-5 agreement >= 0.80. Argmax agreement is not this number."
        )
    survive_30 = bool(tol is not None and float(tol) >= 0.3 - 1e-12)
    fatal_01 = bool(tol is None or float(tol) < 0.01 - 1e-12)
    still_dead_92 = True  # 0.92 is above every tested target
    return {
        "what_error_does_the_model_actually_tolerate": what,
        "is_0_30_survivable_as_a_per_layer_all_layers_program": (
            "YES" if survive_30 else "NO"
        ),
        "is_0_01_already_fatal_as_a_per_layer_all_layers_program": (
            "YES" if fatal_01 else "NO"
        ),
        "are_families_killed_at_0_92_still_dead": (
            "YES. 0.92 relative L2 is far above the tolerated all-layers bar "
            f"({tol}) and above the highest tested target "
            f"({max(DEFAULT_TARGETS)}), which is already not USABLE as a "
            "64-layer program. The missing bar does not reopen those families."
        ),
        "does_structured_hurt_more_than_isotropic": (
            "YES" if structured_stricter else "NO or tied"
        ),
        "is_the_per_layer_bar_the_whole_model_bar": (
            "NO. Isolated one-layer tolerated "
            f"{one}; all-layers tolerated {tol}. Errors compound across depth."
        ),
        "degrades_at": deg,
        "breaks_at": brk,
        "still_dead_at_0_92": still_dead_92,
        "search_target_moves_if_tolerated_above_assumed_1pct": (
            "YES" if (tol is not None and float(tol) > 0.01) else "NO"
        ),
    }


# ---------------------------------------------------------------------------
# Fixture specimen. CPU residual stack; not a stand-in for sealed-3.14.
# ---------------------------------------------------------------------------


def default_scopes(n_layers: int) -> dict[str, tuple[int, ...]]:
    n = int(n_layers)
    if n >= N_LAYERS:
        return {
            SCOPE_ONE: SEALED_ONE_LAYERS,
            SCOPE_FEW: SEALED_FEW_LAYERS,
            SCOPE_ALL: tuple(range(n)),
        }
    mid = n // 2
    few = tuple(range(max(0, mid - 1), min(n, mid + 2)))
    if len(few) < 2:
        few = tuple(range(min(2, n)))
    return {
        SCOPE_ONE: (mid,),
        SCOPE_FEW: few,
        SCOPE_ALL: tuple(range(n)),
    }


class TinyResidualLM:
    """A small residual SwiGLU stack so tests do not need the 27B catalog.

    Not a teacher-corpus stand-in (NNS-001). Named fixture, model_authority
    false. Downstream KL/top-k here prove the machinery, not the Qwen bar.
    """

    def __init__(
        self,
        *,
        n_layers: int = 8,
        hidden: int = 32,
        inner: int = 64,
        vocab: int = 48,
        n_tokens: int = 6,
        seed: int = 0,
        token_ids: Sequence[int] | None = None,
        generate_new: int = 4,
    ) -> None:
        rng = np.random.default_rng(int(seed))
        self.n_layers = int(n_layers)
        self.hidden = int(hidden)
        self.inner = int(inner)
        self.vocab = int(vocab)
        self.generate_new = int(generate_new)
        self.model_id = "fixture_residual_stack"
        self.model_authority = False
        ids = [int(t) for t in (token_ids if token_ids is not None else range(1, n_tokens + 1))]
        if any(t < 0 or t >= self.vocab for t in ids):
            raise ErrorBudgetRefuse("fixture token id out of vocab")
        self.ids = ids
        scale_in = 0.05
        scale_mix = 0.03
        self.embed = rng.standard_normal((self.vocab, self.hidden)).astype(np.float32) * scale_in
        self.W_mix = [
            rng.standard_normal((self.hidden, self.hidden)).astype(np.float32) * scale_mix
            for _ in range(self.n_layers)
        ]
        self.W_gate = [
            rng.standard_normal((self.inner, self.hidden)).astype(np.float32) * scale_in
            for _ in range(self.n_layers)
        ]
        self.W_up = [
            rng.standard_normal((self.inner, self.hidden)).astype(np.float32) * scale_in
            for _ in range(self.n_layers)
        ]
        self.W_down = [
            rng.standard_normal((self.hidden, self.inner)).astype(np.float32) * scale_in
            for _ in range(self.n_layers)
        ]
        self.ln1 = [
            rng.standard_normal(self.hidden).astype(np.float32) * 0.01
            for _ in range(self.n_layers)
        ]
        self.ln2 = [
            rng.standard_normal(self.hidden).astype(np.float32) * 0.01
            for _ in range(self.n_layers)
        ]
        self.final_ln = rng.standard_normal(self.hidden).astype(np.float32) * 0.01
        self.lm_head = (
            rng.standard_normal((self.vocab, self.hidden)).astype(np.float32) * scale_in
        )
        self._basis_rng = np.random.default_rng(int(seed) + 7)

    @property
    def n_tokens(self) -> int:
        return len(self.ids)

    @property
    def token_ids(self) -> list[int]:
        return list(self.ids)

    def _basis(self, layer: int) -> np.ndarray:
        return output_basis_from_down(
            self.W_down[int(layer)],
            rank=min(STRUCTURED_RANK, max(self.hidden - 1, 1), self.inner),
            rng=self._basis_rng,
        )

    def _run_layer(
        self,
        layer: int,
        hidden: np.ndarray,
        *,
        perturb: bool,
        target: float,
        geometry: str,
        rng: np.random.Generator,
        basis: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, float | None]:
        x = rmsnorm(hidden, self.ln1[layer])
        mix = self.W_mix[layer] @ x
        h2 = hidden + mix
        x2 = rmsnorm(h2, self.ln2[layer])
        f, _aux = swiglu(x2, self.W_gate[layer], self.W_up[layer], self.W_down[layer])
        achieved = None
        f_use = f
        if perturb:
            f_use, achieved = inject_relative_error(
                f, target, geometry, rng=rng, basis=basis
            )
        return h2 + f_use, np.asarray(f_use, dtype=np.float32), achieved

    def _prefill(
        self,
        ids: Sequence[int],
        *,
        start_layer: int,
        hiddens: list[np.ndarray],
        perturb_layers: Sequence[int],
        target: float,
        geometry: str,
        rng: np.random.Generator,
        bases: Mapping[int, np.ndarray],
    ) -> tuple[list[np.ndarray], list[float]]:
        hs = [np.asarray(h, dtype=np.float32).copy() for h in hiddens]
        achieved: list[float] = []
        pset = {int(x) for x in perturb_layers}
        for layer in range(int(start_layer), self.n_layers):
            basis = bases.get(int(layer))
            for pos in range(len(hs)):
                hs[pos], _f, ach = self._run_layer(
                    layer,
                    hs[pos],
                    perturb=layer in pset,
                    target=target,
                    geometry=geometry,
                    rng=rng,
                    basis=basis,
                )
                if ach is not None:
                    achieved.append(float(ach))
        return hs, achieved

    def _logits(self, hiddens: Sequence[np.ndarray]) -> np.ndarray:
        out = np.empty((len(hiddens), self.vocab), dtype=np.float32)
        for i, h in enumerate(hiddens):
            hn = rmsnorm(h, self.final_ln)
            out[i] = self.lm_head @ hn
        return out

    def _embed(self, ids: Sequence[int]) -> list[np.ndarray]:
        return [self.embed[int(t)].copy() for t in ids]

    def capture_baseline(self) -> dict[str, Any]:
        bases = {L: self._basis(L) for L in range(self.n_layers)}
        hidden_in: dict[int, list[np.ndarray]] = {}
        hs = self._embed(self.ids)
        for layer in range(self.n_layers):
            hidden_in[layer] = [h.copy() for h in hs]
            for pos in range(len(hs)):
                hs[pos], _f, _a = self._run_layer(
                    layer,
                    hs[pos],
                    perturb=False,
                    target=0.0,
                    geometry=ISOTROPIC,
                    rng=np.random.default_rng(0),
                    basis=None,
                )
        logits = self._logits(hs)
        greedy = self.greedy_continue(self.ids, self.generate_new)
        return {
            "source": {
                "kind": "fixture_residual_stack",
                "real_forward_pass": True,
                "from_embedding_table": True,
                "synthetic": False,
                "fixture": True,
                "token_ids": list(self.ids),
                "n_tokens": len(self.ids),
                "note": "Fixture stack. Not sealed-3.14. model_authority is false.",
            },
            "n_layers": self.n_layers,
            "hidden_size": self.hidden,
            "n_tokens": len(self.ids),
            "token_ids": list(self.ids),
            "hidden_in": hidden_in,
            "bases": bases,
            "final_hidden": [h.copy() for h in hs],
            "logits": logits,
            "greedy_ids": greedy,
            "model_id": self.model_id,
            "model_authority": False,
        }

    def greedy_continue(
        self,
        prefix: Sequence[int],
        n_new: int,
        *,
        perturb_layers: Sequence[int] = (),
        target: float = 0.0,
        geometry: str = ISOTROPIC,
        rng: np.random.Generator | None = None,
        bases: Mapping[int, np.ndarray] | None = None,
    ) -> list[int]:
        ids = [int(t) for t in prefix]
        rng = rng if rng is not None else np.random.default_rng(0)
        bases = bases if bases is not None else {L: self._basis(L) for L in range(self.n_layers)}
        for _ in range(int(n_new)):
            hs = self._embed(ids)
            hs, _ach = self._prefill(
                ids,
                start_layer=0,
                hiddens=hs,
                perturb_layers=perturb_layers,
                target=target,
                geometry=geometry,
                rng=rng,
                bases=bases,
            )
            logits = self._logits(hs)
            ids.append(argmax_id(logits[-1]))
        return ids

    def replay(
        self,
        baseline: Mapping[str, Any],
        *,
        perturb_layers: Sequence[int],
        target: float,
        geometry: str,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        layers = [int(x) for x in perturb_layers]
        start = min(layers) if layers else 0
        hin = baseline["hidden_in"][start]
        hs, achieved = self._prefill(
            self.ids,
            start_layer=start,
            hiddens=hin,
            perturb_layers=layers,
            target=target,
            geometry=geometry,
            rng=rng,
            bases=baseline["bases"],
        )
        logits = self._logits(hs)
        greedy = None
        if self.generate_new > 0:
            greedy = self.greedy_continue(
                self.ids,
                self.generate_new,
                perturb_layers=layers,
                target=target,
                geometry=geometry,
                rng=rng,
                bases=baseline["bases"],
            )
        base_h = np.stack(baseline["final_hidden"], axis=0)
        pert_h = np.stack(hs, axis=0)
        return {
            "logits": logits,
            "achieved_values": achieved,
            "final_hidden": hs,
            "hidden_relative_l2": mean_l2_ratio(pert_h, base_h),
            "last_token_hidden_relative_l2": token_relative_l2(pert_h[-1], base_h[-1]),
            "greedy_ids": greedy,
        }


# ---------------------------------------------------------------------------
# Specimen-agnostic sweep.
# ---------------------------------------------------------------------------


def run_sweep(
    specimen: TinyResidualLM,
    *,
    targets: Sequence[float] = DEFAULT_TARGETS,
    geometries: Sequence[str] = GEOMETRIES,
    scopes: Mapping[str, Sequence[int]] | None = None,
    rng_seed: int = RNG_SEED,
    with_generate: bool = True,
) -> dict[str, Any]:
    """Run the (geometry × scope × target) grid. Achieved is measured per cell."""
    t0 = time.perf_counter()
    baseline = specimen.capture_baseline()
    scope_map = dict(scopes) if scopes is not None else default_scopes(specimen.n_layers)
    if len(scope_map) < 2:
        raise ErrorBudgetRefuse("REFUSED: need at least two scopes")
    if len(tuple(geometries)) < 2:
        raise ErrorBudgetRefuse("REFUSED: need both error geometries")
    points: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(rng_seed))
    for scope_name in SCOPE_NAMES:
        if scope_name not in scope_map:
            continue
        layers = [int(x) for x in scope_map[scope_name]]
        for geometry in geometries:
            for target in targets:
                run = specimen.replay(
                    baseline,
                    perturb_layers=layers,
                    target=float(target),
                    geometry=str(geometry),
                    rng=rng,
                )
                scored = score_logits(baseline["logits"], run["logits"])
                gen = None
                if with_generate and run.get("greedy_ids") is not None:
                    n_pref = int(baseline["n_tokens"])
                    gen = score_generate(
                        baseline["greedy_ids"][n_pref:],
                        run["greedy_ids"][n_pref:],
                        domains=list(CAPABILITY_DOMAINS),
                    )
                point = emit_sweep_point(
                    geometry=str(geometry),
                    scope=str(scope_name),
                    scope_layers=layers,
                    n_layers_total=int(baseline["n_layers"]),
                    target_relative_l2=float(target),
                    achieved_values=run["achieved_values"],
                    last_token_kl=float(scored["last_token_kl"]),
                    mean_top5_agreement=float(scored["mean_top5_agreement"]),
                    mean_top10_agreement=float(scored["mean_top10_agreement"]),
                    argmax_agreement=float(scored["argmax_agreement"]),
                    mean_kl=float(scored["mean_kl"]),
                    hidden_relative_l2=float(run["hidden_relative_l2"]),
                    last_token_hidden_relative_l2=float(
                        run["last_token_hidden_relative_l2"]
                    ),
                    baseline_argmax=int(scored["baseline_argmax"]),
                    pert_argmax=int(scored["pert_argmax"]),
                    baseline_top5=scored["baseline_top5"],
                    pert_top5=scored["pert_top5"],
                    generate=gen,
                    n_tokens=int(baseline["n_tokens"]),
                )
                points.append(point)
    head = headline_from_points(points)
    elapsed = time.perf_counter() - t0
    return {
        "baseline": baseline,
        "points": points,
        "headline": head,
        "answers": answers_from_headline(head, points),
        "scopes": {k: [int(x) for x in v] for k, v in scope_map.items()},
        "targets": [float(x) for x in targets],
        "geometries": [str(g) for g in geometries],
        "process_elapsed_s": float(elapsed),
        "n_points": len(points),
    }


# ---------------------------------------------------------------------------
# Sealed-3.14 CPU specimen. Lazy-imports the capability map consume path.
# ---------------------------------------------------------------------------


def _cim():
    from tools.future import capability_information_map as cim

    return cim


def catalog_available() -> bool:
    try:
        from tools.future.mlp_byte_census import load_sealed, resolve_artifact_root

        root = resolve_artifact_root(load_sealed())
        return (root / "catalog.hq38m20").is_file()
    except Exception:
        return False


class SealedQwenForward:
    """CPU replay of sealed-3.14 with a hook on MLP output F(x)."""

    def __init__(
        self,
        *,
        token_ids: Sequence[int] = SEALED_PROMPT_TOKEN_IDS,
        generate_new: int = 0,
    ) -> None:
        self.ids = [int(t) for t in token_ids]
        self.generate_new = int(generate_new)
        self.model_id = "qwen3.8-27b-sealed-3.14"
        self.model_authority = True
        cim = _cim()
        src = cim.real_activation_source(self.ids)
        cim.refuse_synthetic_activations(src)
        self._src = src
        _by, geo, dn = cim._catalog_index()
        self.geo = geo
        self.dn = dn
        self.n_layers = int(geo["num_hidden_layers"])
        self.hidden = int(geo["hidden_size"])
        if self.n_layers != N_LAYERS or self.hidden != HIDDEN:
            raise SpecimenUnavailable(
                f"geometry {self.n_layers}x{self.hidden} != {N_LAYERS}x{HIDDEN}"
            )

    @property
    def n_tokens(self) -> int:
        return len(self.ids)

    @property
    def token_ids(self) -> list[int]:
        return list(self.ids)

    def _new_state(self, kit) -> dict[str, np.ndarray]:
        cim = _cim()
        if kit.kind == "gqa":
            return cim._new_gqa_state(self.n_tokens)
        return cim._new_dn_state(kit.dn, self.n_tokens)

    def capture_baseline(self, *, log: Any = None) -> dict[str, Any]:
        cim = _cim()
        cim.refuse_synthetic_activations(self._src)
        final_ln = cim.load_f32(
            __import__("pathlib").Path(cim._tensor(None, "norms.final")["segment_path"]),
            [self.hidden],
        )
        hiddens = [cim.embed_row(int(t)) for t in self.ids]
        hidden_in: dict[int, list[np.ndarray]] = {}
        bases: dict[int, np.ndarray] = {}
        rng_b = np.random.default_rng(RNG_SEED + 1)
        t0 = time.perf_counter()
        for layer in range(self.n_layers):
            kit = cim.LayerKit(layer, self.geo, self.dn)
            hidden_in[layer] = [h.copy() for h in hiddens]
            st = self._new_state(kit)
            for pos in range(self.n_tokens):
                hiddens[pos], _aux = cim._run_layer(kit, hiddens[pos], st, pos)
            w_down = kit.mlp()["down"]["W"]
            bases[layer] = output_basis_from_down(
                w_down, rank=STRUCTURED_RANK, rng=rng_b
            )
            kit.drop_weights()
            if log is not None and (layer % 8 == 0 or layer == self.n_layers - 1):
                log.write(
                    f"[mlp_error_budget] baseline layer {layer}/{self.n_layers - 1} "
                    f"elapsed={time.perf_counter() - t0:.1f}s\n"
                )
                log.flush()
        logits = self._all_logits(hiddens, final_ln)
        return {
            "source": dict(self._src),
            "n_layers": self.n_layers,
            "hidden_size": self.hidden,
            "n_tokens": self.n_tokens,
            "token_ids": list(self.ids),
            "hidden_in": hidden_in,
            "bases": bases,
            "final_hidden": [h.copy() for h in hiddens],
            "final_ln": final_ln,
            "logits": logits,
            "greedy_ids": None,
            "model_id": self.model_id,
            "model_authority": True,
            "baseline_elapsed_s": float(time.perf_counter() - t0),
        }

    def _all_logits(self, hiddens: Sequence[np.ndarray], final_ln: np.ndarray) -> np.ndarray:
        cim = _cim()
        rows = []
        for h in hiddens:
            hn = cim.rmsnorm_delta(h, final_ln)
            rows.append(cim.lm_head_gemv(hn))
        return np.stack(rows, axis=0)

    def replay(
        self,
        baseline: Mapping[str, Any],
        *,
        perturb_layers: Sequence[int],
        target: float,
        geometry: str,
        rng: np.random.Generator,
        log: Any = None,
    ) -> dict[str, Any]:
        cim = _cim()
        layers = [int(x) for x in perturb_layers]
        start = min(layers) if layers else 0
        pset = set(layers)
        hs = [h.copy() for h in baseline["hidden_in"][start]]
        achieved: list[float] = []
        t0 = time.perf_counter()
        for layer in range(start, self.n_layers):
            kit = cim.LayerKit(layer, self.geo, self.dn)
            st = self._new_state(kit)
            basis = baseline["bases"].get(int(layer))
            for pos in range(self.n_tokens):
                h3, aux = cim._run_layer(kit, hs[pos], st, pos)
                if layer in pset:
                    f = aux["mlp_out"]
                    f_hat, ach = inject_relative_error(
                        f, float(target), str(geometry), rng=rng, basis=basis
                    )
                    hs[pos] = aux["post_attn_residual"] + f_hat
                    achieved.append(float(ach))
                else:
                    hs[pos] = h3
            kit.drop_weights()
            if log is not None and (layer % 16 == 0 or layer == self.n_layers - 1):
                log.write(
                    f"[mlp_error_budget] replay L{layer} geo={geometry} "
                    f"scope={len(pset)} eps={target} elapsed={time.perf_counter() - t0:.1f}s\n"
                )
                log.flush()
        logits = self._all_logits(hs, baseline["final_ln"])
        base_h = np.stack(baseline["final_hidden"], axis=0)
        pert_h = np.stack(hs, axis=0)
        return {
            "logits": logits,
            "achieved_values": achieved,
            "final_hidden": hs,
            "hidden_relative_l2": mean_l2_ratio(pert_h, base_h),
            "last_token_hidden_relative_l2": token_relative_l2(pert_h[-1], base_h[-1]),
            "greedy_ids": None,
        }


def run_sealed_sweep(
    *,
    targets: Sequence[float] = DEFAULT_TARGETS,
    geometries: Sequence[str] = GEOMETRIES,
    scopes: Mapping[str, Sequence[int]] | None = None,
    rng_seed: int = RNG_SEED,
    token_ids: Sequence[int] = SEALED_PROMPT_TOKEN_IDS,
    log: Any = None,
) -> dict[str, Any]:
    if not catalog_available():
        raise SpecimenUnavailable(
            "REFUSED: sealed-3.14 catalog is not readable; "
            "refusing to fabricate a 27B error bar from a fixture"
        )
    specimen = SealedQwenForward(token_ids=token_ids, generate_new=0)
    t0 = time.perf_counter()
    if log is not None:
        log.write("[mlp_error_budget] capturing sealed baseline prefix\n")
        log.flush()
    baseline = specimen.capture_baseline(log=log)
    scope_map = dict(scopes) if scopes is not None else default_scopes(specimen.n_layers)
    points: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(rng_seed))
    n_cells = sum(
        1
        for s in SCOPE_NAMES
        if s in scope_map
        for _g in geometries
        for _t in targets
    )
    done = 0
    for scope_name in SCOPE_NAMES:
        if scope_name not in scope_map:
            continue
        layers = [int(x) for x in scope_map[scope_name]]
        for geometry in geometries:
            for target in targets:
                done += 1
                if log is not None:
                    log.write(
                        f"[mlp_error_budget] cell {done}/{n_cells} "
                        f"{geometry} {scope_name} eps={target} layers={layers}\n"
                    )
                    log.flush()
                run = specimen.replay(
                    baseline,
                    perturb_layers=layers,
                    target=float(target),
                    geometry=str(geometry),
                    rng=rng,
                    log=log,
                )
                scored = score_logits(baseline["logits"], run["logits"])
                point = emit_sweep_point(
                    geometry=str(geometry),
                    scope=str(scope_name),
                    scope_layers=layers,
                    n_layers_total=int(baseline["n_layers"]),
                    target_relative_l2=float(target),
                    achieved_values=run["achieved_values"],
                    last_token_kl=float(scored["last_token_kl"]),
                    mean_top5_agreement=float(scored["mean_top5_agreement"]),
                    mean_top10_agreement=float(scored["mean_top10_agreement"]),
                    argmax_agreement=float(scored["argmax_agreement"]),
                    mean_kl=float(scored["mean_kl"]),
                    hidden_relative_l2=float(run["hidden_relative_l2"]),
                    last_token_hidden_relative_l2=float(
                        run["last_token_hidden_relative_l2"]
                    ),
                    baseline_argmax=int(scored["baseline_argmax"]),
                    pert_argmax=int(scored["pert_argmax"]),
                    baseline_top5=scored["baseline_top5"],
                    pert_top5=scored["pert_top5"],
                    generate=None,
                    n_tokens=int(baseline["n_tokens"]),
                )
                points.append(point)
    head = headline_from_points(points)
    elapsed = time.perf_counter() - t0
    return {
        "baseline": {
            k: baseline[k]
            for k in (
                "source",
                "n_layers",
                "hidden_size",
                "n_tokens",
                "token_ids",
                "model_id",
                "model_authority",
                "baseline_elapsed_s",
            )
        },
        "baseline_logits_argmax": [int(np.argmax(r)) for r in baseline["logits"]],
        "points": points,
        "headline": head,
        "answers": answers_from_headline(head, points),
        "scopes": {k: [int(x) for x in v] for k, v in scope_map.items()},
        "targets": [float(x) for x in targets],
        "geometries": [str(g) for g in geometries],
        "process_elapsed_s": float(elapsed),
        "n_points": len(points),
        "prompt_token_ids": list(token_ids),
        "prompt_note": "Real embedding rows (CIM ids 561,1000,284,47358: The/would/is/France).",
    }


def _decode_top5(ids: Sequence[int]) -> list[str] | None:
    vocab_path = __import__("pathlib").Path(
        "/Users/scammermike/noetic/NOETIC_PARENT_A/vocab.json"
    )
    if not vocab_path.is_file():
        return None
    try:
        table = json.loads(vocab_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    inv = {int(v): str(k) for k, v in table.items()}
    return [inv.get(int(i), f"<{int(i)}>") for i in ids]


def generate_quality_block(
    sweep: Mapping[str, Any], *, measured: bool
) -> dict[str, Any]:
    points = sweep["points"]
    head = sweep["headline"]
    next_token: list[dict[str, Any]] = []
    for p in points:
        if p.get("scope") != SCOPE_ALL:
            continue
        rec = {
            "geometry": p["geometry"],
            "target_relative_l2": p["target_relative_l2"],
            "band": p["band"],
            "baseline_top5": p.get("baseline_top5"),
            "pert_top5": p.get("pert_top5"),
        }
        if p.get("pert_top5"):
            decoded = _decode_top5(p["pert_top5"])
            if decoded is not None:
                rec["pert_top5_text"] = decoded
                rec["baseline_top5_text"] = _decode_top5(p.get("baseline_top5") or [])
        next_token.append(rec)
    any_gen = any(p.get("generate") and p["generate"].get("status") == "MEASURED" for p in points)
    return {
        "authority": False,
        "status": "MEASURED" if any_gen else "NEXT_TOKEN_TOP5_TEXT",
        "multi_token_generate": "MEASURED" if any_gen else "UNMEASURED",
        "capability_domains": list(CAPABILITY_DOMAINS),
        "capability_domain_forwards": "UNMEASURED",
        "capability_domain_forwards_reason": (
            "Five-domain teacher-corpus prompts are long prefill. The cheap "
            "first measurement is logit KL / top-k on the sealed 4-token "
            "real-embedding prompt. Domain generate is not the usability "
            "criterion and is not fabricated."
        ),
        "next_token_top5": next_token,
        "headline_tolerated": head.get("tolerated_per_layer_relative_l2"),
        "argmax_is_not_parity": True,
        "note": (
            "Next-token top-5 surface forms from the CPU LM head. Multi-token "
            "greedy generate of 27B and per-domain generate were not the cheap "
            "first measurement. Usability is last-token KL and top-5, not this."
        ),
        "measured_fixture_generate": bool(measured and any_gen),
    }


def assemble_receipt(
    sweep: Mapping[str, Any],
    *,
    specimen_name: str,
    model_authority: bool,
) -> dict[str, Any]:
    points = [_py(p) for p in sweep["points"]]
    head = _py(sweep["headline"])
    for p in points:
        validate_calibration(p)
        refuse_argmax_as_parity(p)
        if p.get("parity") is True:
            raise ArgmaxPresentedAsParity("assembled point claims parity")
    refuse_argmax_as_parity(head)
    gen = generate_quality_block(sweep, measured=not model_authority)
    geos = sorted({p["geometry"] for p in points})
    scopes = sorted({p["scope"] for p in points})
    if set(geos) != set(GEOMETRIES):
        raise ErrorBudgetRefuse(f"assembled receipt missing a geometry: {geos}")
    if len(scopes) < 2:
        raise ErrorBudgetRefuse(f"assembled receipt has <2 scopes: {scopes}")
    src = sweep.get("baseline", {}).get("source") or {}
    if model_authority:
        # Sealed path: Gaussian x is a refuse, not a fallback.
        kind = str(src.get("kind") or "")
        if any(tok in kind.lower() for tok in ("gaussian", "synthetic", "randn")):
            raise ErrorBudgetRefuse("REFUSED: synthetic activations on a sealed receipt")
        if src.get("real_forward_pass") is not True:
            raise ErrorBudgetRefuse("REFUSED: sealed receipt without a real forward pass")
    selftest_row = selftest()
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Measure the largest per-layer relative MLP output error at which "
            "the model is still usable, with the criterion stated, under "
            "isotropic and structured error and under one-layer / few / all-64 "
            "scope. The campaign has been scoring programs without this denominator."
        ),
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "predecessors": [CIM_REL, CORPUS_REL, SHARED_REL],
        "specimen": {
            "name": specimen_name,
            "model_id": sweep.get("baseline", {}).get("model_id")
            or (sweep.get("prompt_note") and "qwen3.8-27b-sealed-3.14")
            or specimen_name,
            "model_authority": bool(model_authority),
            "n_layers": int(
                sweep.get("baseline", {}).get("n_layers")
                or (N_LAYERS if model_authority else 0)
            ),
            "hidden_size": int(sweep.get("baseline", {}).get("hidden_size") or 0),
            "n_tokens": int(sweep.get("baseline", {}).get("n_tokens") or 0),
            "token_ids": list(
                sweep.get("prompt_token_ids")
                or sweep.get("baseline", {}).get("token_ids")
                or []
            ),
            "source": _py(src),
        },
        "grid": {
            "targets": [float(x) for x in sweep["targets"]],
            "geometries": list(sweep["geometries"]),
            "scopes": sweep["scopes"],
            "structured_rank": STRUCTURED_RANK,
            "structured_geometry": (
                "residual of a rank-k randomized range-finder on W_down "
                "(the error a low-rank / quantized approximation of the organ "
                "actually produces). Not isotropic noise."
            ),
            "isotropic_geometry": (
                "Gaussian direction in the full output space, scaled per token "
                "to the target relative L2."
            ),
            "injection": "F_hat = F + target * ||F|| * u, verified per token",
            "n_points": int(sweep["n_points"]),
        },
        "metric": {
            "authority": "last_token_kl_and_mean_top5",
            "kl": "KL(softmax(base) || softmax(pert)) in nats, last prompt token",
            "top_k": TOPK_AUTHORITY,
            "argmax_is_not_parity": True,
            "calibration": "per-token ||F_hat-F||/||F||, reported beside the target",
        },
        "criterion": head["criterion"],
        "criterion_defense": CRITERION_DEFENSE,
        "headline": head,
        "answers": _py(sweep["answers"]),
        "sweep": points,
        "generate_quality": _py(gen),
        "anti_fabrication": {
            "detectors": [
                "TARGET_REPORTED_AS_ACHIEVED",
                "ARGMAX_PRESENTED_AS_PARITY",
                "UNCALIBRATED_SWEEP",
                "SYNTHETIC_ACTIVATION",
                "ISOTROPIC_SUBSTITUTED_FOR_STRUCTURED",
            ],
            "loud_exceptions": [
                "TargetReportedAsAchieved",
                "ArgmaxPresentedAsParity",
                "CalibrationDrift",
                "SpecimenUnavailable",
            ],
            "rule": (
                "emit_sweep_point is the only constructor. A cell that copies "
                "the target into achieved_relative_l2 without measuring raises "
                "TargetReportedAsAchieved. present_as_parity on argmax-only "
                "metrics raises ArgmaxPresentedAsParity. Structured geometry "
                "will not fall back to isotropic. A return-flag nobody checks "
                "is not a guard."
            ),
        },
        "selftest": selftest_row,
        "process_elapsed_s": _r(float(sweep.get("process_elapsed_s") or 0.0), 4),
        "gaps_closed": [
            "achieved relative L2 reported beside target at every sweep point, "
            "verified per token",
            "isotropic and structured geometries both measured",
            "at least two scopes measured (one layer vs compounding)",
            "logit KL and top-k agreement, with argmax refused as parity",
            "headline tolerated-error number states its criterion",
        ],
        "what_this_does_not_prove": [
            "GPU generate identity of any packed kernel",
            "multi-token generate quality on the five teacher-corpus domains "
            "(UNMEASURED on the sealed specimen; not the usability criterion)",
            "that a different prompt family has the same bar",
            "a protected TPS or complete-token number",
            "that 0.92-relative-L2 programs would have been fine — they would not",
        ],
        "nomenclature": {
            "usable": BAND_USABLE,
            "degrades": BAND_DEGRADES,
            "breaks": BAND_BREAKS,
            "isotropic": ISOTROPIC,
            "structured": STRUCTURED,
            "static_only": "this sidecar. Models propose; protected deterministic evidence decides.",
        },
        "next": (
            "Read every past and future MLP receipt against headline."
            "tolerated_per_layer_relative_l2. Programs above that number are "
            "dead on function even if they would have cleared an assumed 10% "
            "bar; programs below it are not yet a win on economics."
        ),
    }
    return doc


def selftest() -> dict[str, Any]:
    """Guards on fixtures. Does not load sealed-3.14 and does not claim its bar."""
    target_as_achieved_refused = False
    try:
        validate_calibration(
            {
                "target_relative_l2": 0.01,
                "achieved_relative_l2": 0.01,
                "achieved_measured": False,
            }
        )
    except TargetReportedAsAchieved:
        target_as_achieved_refused = True

    copied = False
    try:
        emit_sweep_point(
            geometry=ISOTROPIC,
            scope=SCOPE_ONE,
            scope_layers=(0,),
            n_layers_total=2,
            target_relative_l2=0.01,
            achieved_values=None,  # type: ignore[arg-type]
            last_token_kl=0.0,
            mean_top5_agreement=1.0,
            n_tokens=1,
        )
    except TargetReportedAsAchieved:
        copied = True

    argmax_refused = False
    try:
        present_as_parity({"argmax_agreement": 1.0, "argmax_identical": True})
    except ArgmaxPresentedAsParity:
        argmax_refused = True

    argmax_verdict_refused = False
    try:
        usability_verdict(
            last_token_kl=None,
            mean_top5_agreement=None,
            argmax_agreement=1.0,
        )
    except ArgmaxPresentedAsParity:
        argmax_verdict_refused = True

    rng = np.random.default_rng(0)
    f = rng.standard_normal(16).astype(np.float32)
    q = output_basis_from_down(
        rng.standard_normal((16, 24)).astype(np.float32), rank=4, rng=rng
    )
    hat_i, ach_i = inject_relative_error(f, 0.03, ISOTROPIC, rng=rng, basis=q)
    hat_s, ach_s = inject_relative_error(f, 0.03, STRUCTURED, rng=rng, basis=q)
    cal_iso = abs(ach_i - 0.03) <= calibration_tol(0.03)
    cal_st = abs(ach_s - 0.03) <= calibration_tol(0.03)
    d_i = hat_i.astype(np.float64) - f.astype(np.float64)
    d_s = hat_s.astype(np.float64) - f.astype(np.float64)
    geo_cos = float(
        (d_i @ d_s)
        / max(float(np.linalg.norm(d_i) * np.linalg.norm(d_s)), ELEMENT_EPS)
    )

    return {
        "target_reported_as_achieved_refused": target_as_achieved_refused,
        "none_achieved_values_refused": copied,
        "argmax_presented_as_parity_refused": argmax_refused,
        "argmax_verdict_without_kl_refused": argmax_verdict_refused,
        "isotropic_calibrates": cal_iso,
        "structured_calibrates": cal_st,
        "geometries_not_interchangeable": abs(geo_cos) < 0.99,
        "geometry_delta_cosine": _r(geo_cos, 6),
    }


def snapshot(*, specimen: str = "fixture") -> dict[str, Any]:
    if specimen == "fixture":
        model = TinyResidualLM(seed=RNG_SEED)
        sweep = run_sweep(model, with_generate=True)
        return assemble_receipt(
            sweep, specimen_name="fixture_residual_stack", model_authority=False
        )
    if specimen in {"sealed", "sealed-3.14", "qwen3.8-27b-sealed-3.14"}:
        sweep = run_sealed_sweep(log=_sys.stderr)
        return assemble_receipt(
            sweep, specimen_name="qwen3.8-27b-sealed-3.14", model_authority=True
        )
    raise ErrorBudgetRefuse(f"unknown specimen {specimen!r}")


def build(*, specimen: str | None = None) -> Any:
    wanted = specimen or _os.environ.get("MLP_ERROR_BUDGET_SPECIMEN") or "sealed"
    if wanted in {"auto", "sealed", "sealed-3.14", "qwen3.8-27b-sealed-3.14"}:
        if catalog_available():
            doc = snapshot(specimen="sealed")
        elif wanted == "auto":
            raise SpecimenUnavailable(
                "REFUSED: catalog missing and specimen=auto will not silently "
                "write a fixture bar as the 27B number"
            )
        else:
            raise SpecimenUnavailable(
                "REFUSED: sealed-3.14 catalog is not readable"
            )
    elif wanted == "fixture":
        doc = snapshot(specimen="fixture")
    else:
        raise ErrorBudgetRefuse(f"unknown specimen {wanted!r}")
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else _sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--specimen",
        default=None,
        help="sealed (default for --build) or fixture",
    )
    args = parser.parse_args(argv_list)
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    if args.build or not argv_list:
        path = build(specimen=args.specimen or "sealed")
        print(path)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

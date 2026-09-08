"""DELTANET MULTI-STEP AUTHORITY — no recurrence claim survives one step.

DeltaNet is 2,961,659,904 bytes and 8.227 ms (cited; not re-measured): 30% of
the token and the second-largest organ. A recurrent state machine that is 99%
accurate for one step can diverge completely by step 64. One-step error is the
wrong statistic. This module is the instrument that every future DeltaNet
claim has to clear before it may be called FITTED_HELDOUT.

    python3 tools/future/deltanet_multistep.py --build
    python3 -m pytest tools/future/test_deltanet_multistep.py -q

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
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tools.future import deltanet_state_function as dsf
from tools.future import executable_economics as ee
from tools.future._common import REPO, git, load_json, write_receipt
from tools.future.physical_primitives import ATLAS_PRIMITIVES


RECEIPT = "DELTANET_MULTISTEP.json"
SCHEMA = "hawking.future.deltanet_multistep.v1"
VERSION = 1
RECORDED_BY = "tools/future/deltanet_multistep.py"
EVIDENCE_CLASS = "STATIC_ONLY"

PREDECESSOR_STATE = "receipts/future/DELTANET_STATE_FUNCTION.json"
PREDECESSOR_REPR = "receipts/future/DELTANET_REPRESENTATION.json"
PREDECESSOR_QKVZ = "receipts/future/DELTANET_QKVZ_PRECISION.json"
ECONOMICS_REL = "receipts/future/EXECUTABLE_ECONOMICS.json"
BUDGET_REL = "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json"
CAP_MAP_REL = "receipts/future/CAPABILITY_INFORMATION_MAP.json"
CORPUS_REL = "receipts/future/MLP_TEACHER_CORPUS.json"

DELTANET_ACTIVE_TARGET = dsf.DELTANET_ACTIVE_TARGET
QKVZ_ACTIVE_TARGET = dsf.QKVZ_ACTIVE_TARGET
TOKEN_ACTIVE_TARGET = dsf.TOKEN_ACTIVE_TARGET
REC_STATE_RESIDENT = dsf.REC_STATE_RESIDENT
CITED_DN_MS = ee.DN_MS  # 8.227, cited organ time, not a measurement of this lane
CITED_TOKEN_MS = ee.CITED_TOKEN_MS  # 28.722
CITED_RESIDENT_TPS = 34.82  # ladder rung; never stored under the key "tps"
CAP_MAP_SHARE_OF_TOKEN = 0.0028028380503877935  # 0.28%; residual budget, not a win
CAP_MAP_BYTES = 27_688_960

REQUIRED_HORIZONS: tuple[int, ...] = (1, 4, 16, 64, 128, 256)
PLUS_HORIZONS: tuple[int, ...] = (512, 1024)
ALL_NAMED_HORIZONS: tuple[int, ...] = REQUIRED_HORIZONS + PLUS_HORIZONS

SERIES_KEYS: tuple[str, ...] = ("state_error", "output_error", "logit_effect")
STATE_ERROR, OUTPUT_ERROR, LOGIT_EFFECT = SERIES_KEYS

# STATIC operator bars. Not a generate identity gate. Same family as
# DELTANET_QKVZ_PRECISION.state_relfro_bar.
STATE_BAR = 0.01
OUTPUT_BAR = 0.01
LOGIT_BAR = 0.01
COSINE_BAR = 0.990
PLATEAU_GROWTH_BAR = 2.0
FLAT_ZERO_ABS = 1.0e-8  # float32 SVD residual of a rank-deficient S; not a real error
RMS_EPS = 1.0e-6
RNG_SEED = 38

DN_PROBE_LAYER = 38  # linear_attention (38 % 4 != 3); teacher-corpus representative
PROBE_HEADS = 8
PROBE_DIM = dsf.KEY_HEAD_DIM  # 128
PROBE_VOCAB = 256
HIDDEN = dsf.HIDDEN
F32_BYTES = 4

IDENTITY = "identity_control"
TRUNCATED_STATE = "truncated_state_rank16"
LOWER_RANK_TRANSITION = "lower_rank_transition_rank8"
GENERATED_TRANSITION = "generated_transition_coefficients"

FITTED_HELDOUT = "FITTED_HELDOUT"
MEASURED_NEGATIVE = "MEASURED_NEGATIVE"
INCOMPLETE = "INCOMPLETE"
CONTROL = "CONTROL"
NOT_LANDED = "NOT_LANDED"
UNMEASURED = "UNMEASURED"
OPEN = "OPEN"

RUN = "RUN"
SKIPPED_FOR_COST = "SKIPPED_FOR_COST"
SKIPPED_INSUFFICIENT_SEQUENCE = "SKIPPED_INSUFFICIENT_SEQUENCE"

PLATEAU = "PLATEAU"
COMPOUNDING = "COMPOUNDING"
FLAT_ZERO = "FLAT_ZERO"
ONSET_THEN_PLATEAU = "ONSET_THEN_PLATEAU"

DIRECT_CONSUME = dsf.DIRECT_CONSUME
REJECTED_DENSE_REMAT = dsf.REJECTED_DENSE_REMAT

LANDING_RECEIPT_RELS: tuple[str, ...] = (
    "receipts/future/DELTANET_GENERATED_TRANSITION.json",
    "receipts/future/GENERATED_TRANSITION.json",
    "receipts/future/T1DNGEN.json",
    "receipts/future/DELTANET_GENERATED_TRANSITION_FIT.json",
)
LANDING_SCRATCH: tuple[Path, ...] = (
    REPO / "workspace" / "ops" / "local" / "scratch" / "t1dngen",
    Path("/Users/scammermike/Downloads/hawking/workspace/ops/local/scratch/t1dngen"),
)


# ---------------------------------------------------------------------------
# THE SIX FAMILIES S020 SEC 16 NAMES, AND WHICH OF THEM THIS AUTHORITY HAS JUDGED.
#
# G053 asks what recurrent update the model actually NEEDS, judged on state
# trajectory and logits rather than on reconstructing the stored matrices. This
# module is that judge. Two of the six have curves; the rest do not, and the
# honest report is which, not a paragraph saying "partially".
#
# The last row is the one worth reading twice. A fused update+consume changes
# WHERE arithmetic happens, not WHAT the recurrence computes, so its trajectory
# is identical by construction - the judge would return FLAT_ZERO exactly like
# the identity control, and a reader would see a pass. Applying a function judge
# to an execution change manufactures evidence. It is scored on bytes, launches
# and bit-identity instead, and this table says so rather than leaving the row
# blank for someone to fill in with a green curve.
# ---------------------------------------------------------------------------

NEEDS_A_FITTED_PROGRAM = "NEEDS_A_FITTED_PROGRAM"
NOT_A_TRAJECTORY_QUESTION = "NOT_A_TRAJECTORY_QUESTION"

S020_FAMILIES: tuple[dict[str, Any], ...] = (
    {"family": "smaller_state", "candidate_id": TRUNCATED_STATE, "judged": True,
     "how": "state, output and logit curves over 7 horizons"},
    {"family": "structured_transitions", "candidate_id": LOWER_RANK_TRANSITION, "judged": True,
     "how": "state, output and logit curves over 7 horizons"},
    {"family": "generated_coefficients", "candidate_id": GENERATED_TRANSITION, "judged": False,
     "blocked_by": NEEDS_A_FITTED_PROGRAM,
     "unblocks_when": "a fitted T1/T2-class program lands at one of LANDING_RECEIPT_RELS",
     "how": "a byte claim without a program is not a curve"},
    {"family": "learned_recurrence", "candidate_id": None, "judged": False,
     "blocked_by": NEEDS_A_FITTED_PROGRAM,
     "unblocks_when": "a fitted recurrence program exists on disk to run through this judge",
     "how": "no program, no trajectory"},
    {"family": "conditional_recurrence", "candidate_id": None, "judged": False,
     "blocked_by": NEEDS_A_FITTED_PROGRAM,
     "unblocks_when": "a fitted router plus per-regime recurrence exists on disk",
     "how": "no program, no trajectory"},
    {"family": "fused_update_and_consume", "candidate_id": None, "judged": False,
     "blocked_by": NOT_A_TRAJECTORY_QUESTION,
     "unblocks_when": "never through this judge",
     "how": "execution change, bit-identical recurrence; this judge would return "
            "FLAT_ZERO like the identity control and that would not be evidence. "
            "Score it on bytes, launches and bit-identity."},
)


def families_report() -> dict[str, Any]:
    """Which of the six are judged, which are blocked, and by what."""
    judged = [f for f in S020_FAMILIES if f["judged"]]
    return {
        "n_families": len(S020_FAMILIES),
        "n_judged_on_trajectory": len(judged),
        "judged": [f["family"] for f in judged],
        "blocked": {
            f["family"]: {"blocked_by": f["blocked_by"], "unblocks_when": f["unblocks_when"]}
            for f in S020_FAMILIES if not f["judged"]
        },
        "families": [dict(f) for f in S020_FAMILIES],
    }


PAYLOAD_CANDIDATES: tuple[Path, ...] = (
    REPO / "workspace" / "ops" / "local" / "scratch" / "mlp_teacher_corpus",
    Path("/Users/scammermike/Downloads/hawking/workspace/ops/local/scratch/mlp_teacher_corpus"),
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate gate. "
    "The 2,961,659,904 / 8.227 ms / 34.82 TPS figures are cited from "
    "DELTANET_REPRESENTATION and RESIDENT_71TPS_CAUSAL_BUDGET; they are not "
    "re-measured here. Multi-step errors are CPU arithmetic of the incumbent "
    "gated-delta operator rolled from a shared S=0 on held-out-by-prompt "
    "teacher-corpus residual-stream rows (L38 post_attn_norm) mapped through "
    "a seeded STATIC coefficient map. That map is not W_qkvz. The trained "
    "q/k/v trajectory is UNMEASURED. Argmax agreement is reported and is "
    "not parity. A one-step-only number is not admissible evidence for a "
    "recurrent claim. evidence_class is STATIC_ONLY. gpu_authority is false."
)

FITTED_HELDOUT_RULE = (
    "A DeltaNet candidate may be called FITTED_HELDOUT only if every clause "
    "holds. (1) The divergence curve reports THREE separate series — "
    "state_error, output_error, logit_effect — at every named horizon; a "
    "single collapsed 'error' is refused, not scored. (2) Required horizons "
    "{1, 4, 16, 64, 128, 256} are each either RUN or explicitly SKIPPED "
    "(SKIPPED_FOR_COST or SKIPPED_INSUFFICIENT_SEQUENCE); a horizon that is "
    "silently omitted is a refuse. FITTED_HELDOUT additionally requires every "
    "required horizon RUN — a skip of 64 or longer is INCOMPLETE, not a fit. "
    "(3) A one-step-only number is not admissible evidence for a recurrent "
    "claim: demand_fitted_heldout raises OneStepOnlyRefuse rather than "
    "returning a verdict. (4) Argmax agreement is not parity: logit_effect "
    "must carry relative_l2 (and cosine) independently, and a candidate that "
    "preserves argmax while drifting in logit space has not been validated. "
    "(5) The curve shape on every series is PLATEAU (or FLAT_ZERO) from "
    "horizon 1. COMPOUNDING cannot support a promotion. ONSET_THEN_PLATEAU "
    "(exact until the recurrent rank fills, then a residual that sits) is "
    "the same refuse: the one-step and even 16-step number was silent about "
    "the function. (6) At every RUN required horizon, including the "
    "longest: state relative L2 <= 0.01, output relative L2 <= 0.01, logit "
    "relative L2 <= 0.01. Cosine of each series is reported; clearing argmax "
    "without clearing logit relative L2 is a fail. (7) Held-out means "
    "held-out PROMPTS. A train-set number reported as held-out is refused. "
    "(8) Executable economics (bytes_removed AND bytes_added, extra FLOPs, "
    "dispatch delta, consuming primitive) scored by "
    "tools/future/executable_economics.py; a compression ratio is not a "
    "candidate. (9) This is STATIC operator evidence, not a generate "
    "identity gate and not capability. The capability-information-map 0.28% "
    "token licence is a residual budget to allocate, never a byte win of "
    "this lane."
)


StepFn = Callable[[np.ndarray, Mapping[str, np.ndarray]], tuple[np.ndarray, np.ndarray]]


class DeltaNetMultistepRefuse(ValueError):
    """The multi-step authority refused rather than guessing."""


class OneStepOnlyRefuse(DeltaNetMultistepRefuse):
    """A one-step number is not admissible evidence for a recurrent claim."""

    def __init__(self, detail: str = "") -> None:
        extra = f" ({detail})" if detail else ""
        super().__init__(
            "REFUSED: one-step-only number is not admissible evidence for a "
            f"recurrent DeltaNet claim{extra}. Roll at least the required "
            f"horizons {list(REQUIRED_HORIZONS)} or name each skip. "
            "demand_fitted_heldout does not return a verdict on one step."
        )


class CollapsedSeriesRefuse(DeltaNetMultistepRefuse):
    """State, output and logit errors must stay three separate series."""

    def __init__(self, detail: str = "") -> None:
        extra = f": {detail}" if detail else ""
        super().__init__(
            "REFUSED: state_error, output_error and logit_effect must be "
            f"three separate series, never collapsed into one number{extra}"
        )


class ArgmaxIsNotParity(DeltaNetMultistepRefuse):
    """Argmax agreement without a logit-space error is not parity."""

    def __init__(self, detail: str = "") -> None:
        extra = f": {detail}" if detail else ""
        super().__init__(
            "REFUSED: argmax agreement is not parity. logit_effect must "
            f"report relative_l2 independently{extra}"
        )


class IncompleteHorizonsRefuse(DeltaNetMultistepRefuse):
    """FITTED_HELDOUT is not available when a required horizon was skipped."""

    def __init__(self, missing: Sequence[int], *, detail: str = "") -> None:
        self.missing = tuple(int(x) for x in missing)
        extra = f" ({detail})" if detail else ""
        super().__init__(
            "REFUSED: FITTED_HELDOUT requires every required horizon RUN; "
            f"missing/skipped {list(self.missing)}{extra}"
        )


class SilentOmissionRefuse(DeltaNetMultistepRefuse):
    """A named horizon that is neither RUN nor SKIPPED was dropped."""


class TrainReportedAsHeldOut(DeltaNetMultistepRefuse):
    """A train-set figure cannot be reported as held-out."""


class MissingEconomics(DeltaNetMultistepRefuse):
    """A candidate with no bytes_removed and bytes_added is not a candidate."""

    def __init__(self, cand_id: str, *, missing: Sequence[str]) -> None:
        self.cand_id = cand_id
        self.missing = tuple(missing)
        super().__init__(
            f"REFUSED: candidate {cand_id!r} has no executable economics "
            f"(missing {list(self.missing)}; a compression ratio is not a candidate)."
        )


class CorpusUnavailable(DeltaNetMultistepRefuse):
    """Real residual-stream rows are not readable; synthesizing X is NNS-001."""


# ---------------------------------------------------------------------------
# Small numeric helpers.
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


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """||a-b|| / ||b||. den=0 → 0 if also num=0 else +inf."""
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
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


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def gated_rmsnorm(
    h: np.ndarray,
    z: np.ndarray | None,
    weight: np.ndarray | None = None,
    *,
    eps: float = RMS_EPS,
) -> np.ndarray:
    """Organ output of the mixer: RMSNorm(h) * weight * silu(z). z=None → h."""
    if z is None:
        return np.asarray(h, dtype=np.float32)
    inv = 1.0 / np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + eps)
    w = 1.0 if weight is None else weight
    return (h * inv * w * _silu(z)).astype(np.float32, copy=False)


def truncate_state(state: np.ndarray, rank: int) -> np.ndarray:
    """Per-head truncated SVD of S. Native rank-r store; not a dense remat."""
    h, d_k, d_v = state.shape
    r = min(int(rank), d_k, d_v)
    if r <= 0:
        raise DeltaNetMultistepRefuse(f"truncate rank {rank} is not positive")
    if r >= min(d_k, d_v):
        return np.array(state, copy=True)
    out = np.empty_like(state)
    for i in range(h):
        u, s, vt = np.linalg.svd(state[i].astype(np.float64, copy=False), full_matrices=False)
        out[i] = ((u[:, :r] * s[:r]) @ vt[:r]).astype(state.dtype, copy=False)
    return out


def project_rows(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """x @ P @ P.T with P orthonormal of shape (dim, rank)."""
    return (x @ basis) @ basis.T


# ---------------------------------------------------------------------------
# Series guards. Load-bearing: collapsing the three errors is a refuse.
# ---------------------------------------------------------------------------


def require_separate_series(record: Mapping[str, Any]) -> None:
    """Refuse a horizon record that collapsed the three errors into one."""
    if not isinstance(record, Mapping):
        raise CollapsedSeriesRefuse("horizon record is not a mapping")
    if "error" in record and not all(k in record for k in SERIES_KEYS):
        raise CollapsedSeriesRefuse(
            "found a collapsed 'error' field without the three series"
        )
    missing = [k for k in SERIES_KEYS if k not in record]
    if missing:
        raise CollapsedSeriesRefuse(f"missing {missing}")
    for key in SERIES_KEYS:
        val = record[key]
        if isinstance(val, (int, float, bool)) or val is None:
            raise CollapsedSeriesRefuse(
                f"{key} collapsed to {type(val).__name__}; want a mapping "
                "with relative_l2 (and, for logit_effect, argmax_agreement)"
            )
        if not isinstance(val, Mapping):
            raise CollapsedSeriesRefuse(f"{key} is {type(val).__name__}, not a mapping")
        if "relative_l2" not in val:
            raise CollapsedSeriesRefuse(f"{key} is missing relative_l2")
    require_logit_effect(record[LOGIT_EFFECT])


def require_logit_effect(row: Mapping[str, Any]) -> None:
    """Argmax agreement without logit relative_l2 is not parity."""
    if not isinstance(row, Mapping):
        raise ArgmaxIsNotParity("logit_effect is not a mapping")
    if "relative_l2" not in row:
        raise ArgmaxIsNotParity(
            "argmax_agreement="
            + repr(row.get("argmax_agreement"))
            + " with no relative_l2"
        )
    if "argmax_agreement" not in row:
        raise ArgmaxIsNotParity("logit_effect is missing argmax_agreement")
    if "argmax_is_not_parity" in row and row["argmax_is_not_parity"] is False:
        raise ArgmaxIsNotParity(
            "argmax_is_not_parity must stay True: argmax agreement is not parity"
        )


def require_named_horizons(
    *,
    required: Sequence[int] = REQUIRED_HORIZONS,
    plus: Sequence[int] = PLUS_HORIZONS,
    run: Sequence[int],
    skipped: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Every named horizon is RUN or SKIPPED. Silent omission is a refuse."""
    run_set = {int(h) for h in run}
    skipped_by = {int(row["horizon"]): row for row in skipped}
    overlap = run_set & set(skipped_by)
    if overlap:
        raise SilentOmissionRefuse(
            f"horizon(s) {sorted(overlap)} are both RUN and SKIPPED"
        )
    named: list[dict[str, Any]] = []
    for h in list(required) + list(plus):
        h = int(h)
        if h in run_set:
            named.append({"horizon": h, "status": RUN})
        elif h in skipped_by:
            row = dict(skipped_by[h])
            reason = str(row.get("reason") or SKIPPED_FOR_COST)
            if reason not in {SKIPPED_FOR_COST, SKIPPED_INSUFFICIENT_SEQUENCE}:
                raise SilentOmissionRefuse(
                    f"horizon {h} skip reason {reason!r} is not a named skip"
                )
            named.append(
                {
                    "horizon": h,
                    "status": "SKIPPED",
                    "reason": reason,
                    **{
                        k: v
                        for k, v in row.items()
                        if k not in {"horizon", "status", "reason"}
                    },
                }
            )
        else:
            raise SilentOmissionRefuse(
                f"horizon {h} is neither RUN nor SKIPPED (silent omission)"
            )
    return named


def classify_shape(values_by_horizon: Mapping[int, float]) -> str:
    """Plateau vs compounding vs delayed onset. Flat zero is named as such.

    A family that is exact until the recurrent rank fills and then sits at a
    residual is ONSET_THEN_PLATEAU, not PLATEAU: the one-step (and even
    16-step) number was silent. Only a PLATEAU that is quiet from horizon 1
    can support a promotion, and only if it also clears the absolute bars.
    """
    if not values_by_horizon:
        raise DeltaNetMultistepRefuse("cannot classify an empty divergence series")
    hs = sorted(int(h) for h in values_by_horizon)
    series = [float(values_by_horizon[h]) for h in hs]
    if any(math.isnan(v) or math.isinf(v) for v in series):
        return COMPOUNDING
    if all(v <= FLAT_ZERO_ABS for v in series):
        return FLAT_ZERO
    e1 = max(series[0], FLAT_ZERO_ABS)
    growth = series[-1] / e1
    late = series[max(1, len(series) // 2) :]
    late_hi = max(late)
    late_lo = min(late)
    late_stable = (late_hi / max(late_lo, FLAT_ZERO_ABS)) <= PLATEAU_GROWTH_BAR
    started_quiet = series[0] <= FLAT_ZERO_ABS
    became_noisy = any(v > FLAT_ZERO_ABS for v in series[1:])
    if started_quiet and became_noisy:
        return ONSET_THEN_PLATEAU if late_stable else COMPOUNDING
    if growth > PLATEAU_GROWTH_BAR:
        return ONSET_THEN_PLATEAU if late_stable else COMPOUNDING
    if len(series) >= 3:
        for later in series[1:]:
            if later > PLATEAU_GROWTH_BAR * e1 and later > series[0]:
                return ONSET_THEN_PLATEAU if late_stable else COMPOUNDING
    return PLATEAU


def shape_supports_promotion(shape: str) -> bool:
    """Only a plateau that is quiet from step 1 can promote. Delayed onset cannot."""
    return shape in {PLATEAU, FLAT_ZERO}


def overall_shape(shapes: Mapping[str, str]) -> str:
    """Worst series names the curve. Compounding outranks delayed onset outranks plateau."""
    vals = [str(s) for s in shapes.values()]
    if not vals or any(s == INCOMPLETE for s in vals):
        if any(s == COMPOUNDING for s in vals):
            return COMPOUNDING
        if not vals:
            return INCOMPLETE
    if any(s == COMPOUNDING for s in vals):
        return COMPOUNDING
    if any(s == ONSET_THEN_PLATEAU for s in vals):
        return ONSET_THEN_PLATEAU
    if any(s == PLATEAU for s in vals):
        return PLATEAU
    if vals and all(s == FLAT_ZERO for s in vals):
        return FLAT_ZERO
    return INCOMPLETE


# ---------------------------------------------------------------------------
# Verdict. One-step raises; it does not return a fake no.
# ---------------------------------------------------------------------------


def _run_horizons(curve: Mapping[str, Any]) -> list[int]:
    raw = curve.get("horizons_run")
    if raw is None:
        recs = curve.get("per_horizon") or []
        raw = [r.get("horizon") for r in recs]
    return sorted({int(h) for h in raw if h is not None})


def demand_fitted_heldout(
    curve: Mapping[str, Any],
    *,
    required: Sequence[int] = REQUIRED_HORIZONS,
    bars: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return FITTED_HELDOUT or MEASURED_NEGATIVE, or raise.

    A one-step-only curve raises OneStepOnlyRefuse rather than reporting a
    verdict. Missing required horizons raise IncompleteHorizonsRefuse.
    Collapsed series raise CollapsedSeriesRefuse. Those are not 'no' verdicts;
    they are missing evidence.
    """
    recs = list(curve.get("per_horizon") or [])
    for rec in recs:
        require_separate_series(rec)
    run = _run_horizons(curve)
    req = [int(h) for h in required]
    if not run:
        raise OneStepOnlyRefuse("no horizon was RUN")
    if run == [1] or (max(run) == 1 and set(run) <= {1}):
        raise OneStepOnlyRefuse(f"horizons_run={run}")
    missing = [h for h in req if h not in run]
    if missing:
        raise IncompleteHorizonsRefuse(missing, detail=f"horizons_run={run}")

    bars = dict(bars or {})
    state_bar = float(bars.get(STATE_ERROR, STATE_BAR))
    output_bar = float(bars.get(OUTPUT_ERROR, OUTPUT_BAR))
    logit_bar = float(bars.get(LOGIT_EFFECT, LOGIT_BAR))

    series_vals: dict[str, dict[int, float]] = {k: {} for k in SERIES_KEYS}
    argmax_at: dict[int, float] = {}
    for rec in recs:
        h = int(rec["horizon"])
        if h not in req:
            continue
        series_vals[STATE_ERROR][h] = float(rec[STATE_ERROR]["relative_l2"])
        series_vals[OUTPUT_ERROR][h] = float(rec[OUTPUT_ERROR]["relative_l2"])
        series_vals[LOGIT_EFFECT][h] = float(rec[LOGIT_EFFECT]["relative_l2"])
        argmax_at[h] = float(rec[LOGIT_EFFECT]["argmax_agreement"])

    shapes = {k: classify_shape(series_vals[k]) for k in SERIES_KEYS}
    overall = overall_shape(shapes)
    reasons: list[str] = []
    if not all(shape_supports_promotion(s) for s in shapes.values()):
        reasons.append(f"shape:{overall}")
    for rec in recs:
        h = int(rec["horizon"])
        if h not in req:
            continue
        if float(rec[STATE_ERROR]["relative_l2"]) > state_bar:
            reasons.append(f"state_error@{h}>{state_bar}")
        if float(rec[OUTPUT_ERROR]["relative_l2"]) > output_bar:
            reasons.append(f"output_error@{h}>{output_bar}")
        logit = rec[LOGIT_EFFECT]
        if float(logit["relative_l2"]) > logit_bar:
            reasons.append(f"logit_effect@{h}>{logit_bar}")
            if float(logit["argmax_agreement"]) >= 1.0 - 1e-12:
                reasons.append(
                    f"argmax_survived@{h}_but_logit_relative_l2_failed (argmax is not parity)"
                )

    status = FITTED_HELDOUT if not reasons else MEASURED_NEGATIVE
    return {
        "status": status,
        "reasons": reasons,
        "shape": {"by_series": shapes, "overall": overall},
        "bars": {
            STATE_ERROR: state_bar,
            OUTPUT_ERROR: output_bar,
            LOGIT_EFFECT: logit_bar,
            "cosine_bar": COSINE_BAR,
            "plateau_growth_bar": PLATEAU_GROWTH_BAR,
        },
        "horizons_run": run,
        "argmax_agreement_by_horizon": {str(h): argmax_at[h] for h in sorted(argmax_at)},
        "argmax_is_not_parity": True,
        "one_step_only_admissible": False,
        "rule": "see acceptance.fitted_heldout_rule",
    }


# ---------------------------------------------------------------------------
# Coefficients. Real residual-stream rows, or an explicit fixture.
# ---------------------------------------------------------------------------


def resolve_payload_dir() -> Path | None:
    for path in PAYLOAD_CANDIDATES:
        if (path / "rows.jsonl").is_file() and (path / f"L{DN_PROBE_LAYER:02d}_x.f32").is_file():
            return path
    return None


def _corpus_split(payload: Path) -> dict[str, Any]:
    cap_path = payload / "CAPTURE.json"
    if cap_path.is_file():
        cap = json.loads(cap_path.read_text())
        split = cap.get("split") or {}
        if isinstance(split, dict) and split.get("hold_prompt_ids"):
            return split
    receipt = REPO / CORPUS_REL
    if receipt.is_file():
        doc = load_json(receipt)
        split = ((doc.get("capture") or {}).get("split")) or doc.get("split") or {}
        if isinstance(split, dict):
            return split
    raise CorpusUnavailable("REFUSED: teacher-corpus split is not readable")


def load_held_out_hidden_sequences(
    *,
    layer: int = DN_PROBE_LAYER,
    min_tokens: int = 256,
    max_sequences: int = 2,
    payload: Path | None = None,
) -> list[dict[str, Any]]:
    """Real post_attn_norm rows, held-out by prompt_id. Train prompts refused."""
    root = payload if payload is not None else resolve_payload_dir()
    if root is None:
        raise CorpusUnavailable(
            "REFUSED: mlp_teacher_corpus payload is not readable; "
            "refusing to synthesise X (NNS-001)"
        )
    split = _corpus_split(root)
    hold_ids = set(str(x) for x in (split.get("hold_prompt_ids") or []))
    train_ids = set(str(x) for x in (split.get("train_prompt_ids") or []))
    if not hold_ids:
        raise CorpusUnavailable("REFUSED: hold_prompt_ids missing from corpus split")
    rows_path = root / "rows.jsonl"
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    with rows_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if int(rec.get("layer", -1)) != int(layer):
                continue
            pid = str(rec["prompt_id"])
            declared = str(rec.get("split") or "")
            if pid in train_ids or declared == "train":
                continue
            if pid not in hold_ids:
                continue
            if rec.get("synthetic") is True:
                raise CorpusUnavailable(
                    f"REFUSED: synthetic row {rec.get('row_id')} in hold set (NNS-001)"
                )
            by_prompt.setdefault(pid, []).append(rec)
    sequences: list[dict[str, Any]] = []
    x_path = root / f"L{int(layer):02d}_x.f32"
    n_rows = x_path.stat().st_size // (HIDDEN * F32_BYTES)
    mm = np.memmap(x_path, dtype="<f4", mode="r", shape=(n_rows, HIDDEN))
    ranked = sorted(by_prompt.items(), key=lambda kv: -len(kv[1]))
    for pid, recs in ranked:
        recs = sorted(recs, key=lambda r: int(r["token_position"]))
        if len(recs) < int(min_tokens):
            continue
        if any(str(r.get("split")) == "train" for r in recs):
            raise TrainReportedAsHeldOut(
                f"REFUSED: prompt {pid} is in the hold id set but a row is split=train"
            )
        idx = [int(r["x_row_index"]) for r in recs]
        x = np.asarray(mm[idx], dtype=np.float32)
        sequences.append(
            {
                "prompt_id": pid,
                "layer": int(layer),
                "split": "hold",
                "n_tokens": int(x.shape[0]),
                "seq_len_declared": int(recs[0].get("seq_len") or x.shape[0]),
                "capability_domain": recs[0].get("capability_domain"),
                "x": x,
                "x_path": str(x_path),
                "held_out_unit": "prompt_id",
            }
        )
        if len(sequences) >= int(max_sequences):
            break
    if not sequences:
        raise CorpusUnavailable(
            f"REFUSED: no held-out prompt at L{layer} has >= {min_tokens} tokens"
        )
    return sequences


def coefficients_from_hidden(
    x: np.ndarray,
    *,
    n_heads: int = PROBE_HEADS,
    dim: int = PROBE_DIM,
    seed: int = RNG_SEED,
) -> dict[str, np.ndarray]:
    """Seeded STATIC map of a real residual-stream trajectory to (q,k,v,z,decay,beta).

    Not W_qkvz. Shader L2 recipe on q,k. decay in (0.85, 0.99), beta = sigmoid.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != HIDDEN:
        raise DeltaNetMultistepRefuse(
            f"hidden X shape {x.shape} is not (T, {HIDDEN})"
        )
    t = int(x.shape[0])
    rng = np.random.default_rng(int(seed))
    scale = 1.0 / math.sqrt(HIDDEN)
    wq = rng.normal(size=(HIDDEN, n_heads * dim)).astype(np.float32) * scale
    wk = rng.normal(size=(HIDDEN, n_heads * dim)).astype(np.float32) * scale
    wv = rng.normal(size=(HIDDEN, n_heads * dim)).astype(np.float32) * scale
    wz = rng.normal(size=(HIDDEN, n_heads * dim)).astype(np.float32) * scale
    wd = rng.normal(size=(HIDDEN, n_heads)).astype(np.float32) * scale
    wb = rng.normal(size=(HIDDEN, n_heads)).astype(np.float32) * scale
    q = (x @ wq).reshape(t, n_heads, dim)
    k = (x @ wk).reshape(t, n_heads, dim)
    v = (x @ wv).reshape(t, n_heads, dim)
    z = (x @ wz).reshape(t, n_heads, dim)
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + RMS_EPS) / math.sqrt(dim)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + RMS_EPS)
    decay_raw = x @ wd
    beta_raw = x @ wb
    decay = (0.85 + 0.14 / (1.0 + np.exp(-np.clip(decay_raw, -60.0, 60.0)))).astype(
        np.float32
    )
    beta = (1.0 / (1.0 + np.exp(-np.clip(beta_raw, -60.0, 60.0)))).astype(np.float32)
    return {
        "q": q.astype(np.float32, copy=False),
        "k": k.astype(np.float32, copy=False),
        "v": v.astype(np.float32, copy=False),
        "z": z.astype(np.float32, copy=False),
        "decay": decay,
        "beta": beta,
    }


def make_fixture_hidden(
    n_tokens: int,
    *,
    hidden: int = HIDDEN,
    seed: int = RNG_SEED,
) -> np.ndarray:
    """Deterministic AR stream for unit tests. Not held-out and not real X."""
    rng = np.random.default_rng(int(seed))
    n = int(n_tokens)
    x = rng.normal(size=(n, hidden)).astype(np.float32) * (1.0 / math.sqrt(hidden))
    for t in range(1, n):
        x[t] = 0.92 * x[t - 1] + 0.08 * x[t]
    return x


def logit_matrix(
    *,
    n_heads: int,
    dim: int,
    vocab: int = PROBE_VOCAB,
    seed: int = RNG_SEED,
) -> np.ndarray:
    """STATIC proxy readout. Not the production LM head."""
    rng = np.random.default_rng(int(seed) + 7)
    w = rng.normal(size=(int(vocab), int(n_heads) * int(dim))).astype(np.float32)
    w /= np.sqrt((w * w).sum(-1, keepdims=True) + RMS_EPS)
    return w


# ---------------------------------------------------------------------------
# Recurrence steps.
# ---------------------------------------------------------------------------


def reference_step(
    state: np.ndarray, coeffs: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Incumbent vi kernel: S := (I - beta k k^T)(decay S) + beta k v^T; h := S^T q."""
    return dsf.gated_delta_step(
        state,
        coeffs["q"],
        coeffs["k"],
        coeffs["v"],
        coeffs["decay"],
        coeffs["beta"],
    )


def identity_step(
    state: np.ndarray, coeffs: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    return reference_step(state, coeffs)


def make_truncated_state_step(rank: int) -> StepFn:
    r = int(rank)

    def _step(
        state: np.ndarray, coeffs: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        state2, _h = reference_step(state, coeffs)
        state3 = truncate_state(state2, r)
        h3 = np.einsum("hkv,hk->hv", state3, coeffs["q"])
        return state3, h3

    return _step


def make_lower_rank_transition_step(basis: np.ndarray) -> StepFn:
    p = np.asarray(basis, dtype=np.float32)

    def _step(
        state: np.ndarray, coeffs: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        k2 = project_rows(coeffs["k"], p)
        return dsf.gated_delta_step(
            state,
            coeffs["q"],
            k2,
            coeffs["v"],
            coeffs["decay"],
            coeffs["beta"],
        )

    return _step


def orthonormal_basis(dim: int, rank: int, *, seed: int = RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 11)
    raw = rng.normal(size=(int(dim), int(rank))).astype(np.float64)
    q, _r = np.linalg.qr(raw)
    return q.astype(np.float32)


# ---------------------------------------------------------------------------
# Rollout.
# ---------------------------------------------------------------------------


def _coeffs_at(bundle: Mapping[str, np.ndarray], t: int) -> dict[str, np.ndarray]:
    return {
        "q": bundle["q"][t],
        "k": bundle["k"][t],
        "v": bundle["v"][t],
        "z": bundle["z"][t] if "z" in bundle else None,  # type: ignore[dict-item]
        "decay": bundle["decay"][t],
        "beta": bundle["beta"][t],
    }


def measure_horizon(
    *,
    horizon: int,
    state_c: np.ndarray,
    state_r: np.ndarray,
    h_c: np.ndarray,
    h_r: np.ndarray,
    z: np.ndarray | None,
    w_logit: np.ndarray,
    logit_offset: float = 0.0,
) -> dict[str, Any]:
    """Three separate errors. logit_offset is a test injection, default 0."""
    out_c = gated_rmsnorm(h_c, z)
    out_r = gated_rmsnorm(h_r, z)
    logits_c = w_logit @ out_c.reshape(-1)
    logits_r = w_logit @ out_r.reshape(-1)
    if logit_offset != 0.0:
        logits_c = logits_c + float(logit_offset)
    rec = {
        "horizon": int(horizon),
        STATE_ERROR: {
            "relative_l2": relative_l2(state_c, state_r),
            "cosine": cosine(state_c, state_r),
        },
        OUTPUT_ERROR: {
            "relative_l2": relative_l2(out_c, out_r),
            "cosine": cosine(out_c, out_r),
            "readout_h_relative_l2": relative_l2(h_c, h_r),
            "readout_h_cosine": cosine(h_c, h_r),
        },
        LOGIT_EFFECT: {
            "relative_l2": relative_l2(logits_c, logits_r),
            "cosine": cosine(logits_c, logits_r),
            "argmax_agreement": float(
                np.argmax(logits_c) == np.argmax(logits_r)
            ),
            "argmax_is_not_parity": True,
            "proxy": "STATIC orthonormal vocab readout of gated mixer output; not lm_head",
        },
    }
    require_separate_series(rec)
    return rec


def skip_record(horizon: int, *, reason: str, detail: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {"horizon": int(horizon), "reason": str(reason)}
    if detail:
        row["detail"] = detail
    return row


def roll_curve(
    *,
    candidate_step: StepFn,
    coeffs: Mapping[str, np.ndarray],
    w_logit: np.ndarray,
    required: Sequence[int] = REQUIRED_HORIZONS,
    plus: Sequence[int] = PLUS_HORIZONS,
    skip_for_cost: Sequence[int] = (),
    logit_offset: float = 0.0,
    n_heads: int | None = None,
    dim: int | None = None,
) -> dict[str, Any]:
    """Roll reference and candidate from the same S=0. Name every horizon."""
    t_max = int(np.asarray(coeffs["q"]).shape[0])
    heads = int(n_heads if n_heads is not None else coeffs["q"].shape[1])
    d = int(dim if dim is not None else coeffs["q"].shape[2])
    skip_cost = {int(h) for h in skip_for_cost}
    named = list(required) + list(plus)
    snapshots = [int(h) for h in named if h not in skip_cost]
    s_r = np.zeros((heads, d, d), dtype=np.float32)
    s_c = np.zeros((heads, d, d), dtype=np.float32)
    per_horizon: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    want = {h for h in snapshots}
    h_r = h_c = z_t = None
    for t in range(t_max):
        c = _coeffs_at(coeffs, t)
        z_t = c["z"]
        s_r, h_r = reference_step(s_r, c)
        s_c, h_c = candidate_step(s_c, c)
        step = t + 1
        if step in want:
            per_horizon.append(
                measure_horizon(
                    horizon=step,
                    state_c=s_c,
                    state_r=s_r,
                    h_c=h_c,
                    h_r=h_r,
                    z=z_t,
                    w_logit=w_logit,
                    logit_offset=logit_offset,
                )
            )
            want.remove(step)
        if not want:
            break
    run = [int(r["horizon"]) for r in per_horizon]
    for h in named:
        h = int(h)
        if h in run:
            continue
        if h in skip_cost:
            skipped.append(
                skip_record(h, reason=SKIPPED_FOR_COST, detail="caller named this horizon as cost-skipped")
            )
        elif h > t_max:
            skipped.append(
                skip_record(
                    h,
                    reason=SKIPPED_INSUFFICIENT_SEQUENCE,
                    detail=f"sequence length {t_max} < horizon {h}",
                )
            )
        else:
            skipped.append(
                skip_record(
                    h,
                    reason=SKIPPED_INSUFFICIENT_SEQUENCE,
                    detail=f"horizon {h} was not reached (rolled {max(run) if run else 0} of {t_max})",
                )
            )
    named_rows = require_named_horizons(
        required=required, plus=plus, run=run, skipped=skipped
    )
    series: dict[str, list[dict[str, Any]]] = {k: [] for k in SERIES_KEYS}
    for rec in per_horizon:
        h = int(rec["horizon"])
        series[STATE_ERROR].append(
            {"horizon": h, **{k: rec[STATE_ERROR][k] for k in rec[STATE_ERROR]}}
        )
        series[OUTPUT_ERROR].append(
            {"horizon": h, **{k: rec[OUTPUT_ERROR][k] for k in rec[OUTPUT_ERROR]}}
        )
        series[LOGIT_EFFECT].append(
            {"horizon": h, **{k: rec[LOGIT_EFFECT][k] for k in rec[LOGIT_EFFECT]}}
        )
    shapes = {}
    for key in SERIES_KEYS:
        vals = {int(row["horizon"]): float(row["relative_l2"]) for row in series[key]}
        shapes[key] = classify_shape(vals) if vals else INCOMPLETE
    overall = overall_shape(shapes)
    return {
        "horizons_required": [int(h) for h in required],
        "horizons_plus": [int(h) for h in plus],
        "horizons_run": run,
        "horizons_skipped": skipped,
        "horizons_named": named_rows,
        "n_tokens_available": t_max,
        "n_heads": heads,
        "dim": d,
        "per_horizon": per_horizon,
        "series": series,
        "shape": {"by_series": shapes, "overall": overall},
        "state_init": "S=0",
        "argmax_is_not_parity": True,
        "one_step_only_admissible": False,
    }


def aggregate_curves(
    curves: Sequence[Mapping[str, Any]],
    *,
    required: Sequence[int] | None = None,
    plus: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Worst-prompt (max) relative L2 is the authority; mean is descriptive."""
    if not curves:
        raise DeltaNetMultistepRefuse("no curves to aggregate")
    req = [int(h) for h in (required if required is not None else curves[0]["horizons_required"])]
    plus_h = [int(h) for h in (plus if plus is not None else curves[0]["horizons_plus"])]
    by_h: dict[int, dict[str, list[float]]] = {}
    argmax_h: dict[int, list[float]] = {}
    run_union: set[int] = set()
    skipped_union: dict[int, dict[str, Any]] = {}
    for curve in curves:
        run_union.update(int(h) for h in curve["horizons_run"])
        for row in curve["horizons_skipped"]:
            skipped_union.setdefault(int(row["horizon"]), dict(row))
        for rec in curve["per_horizon"]:
            h = int(rec["horizon"])
            slot = by_h.setdefault(
                h, {k: [] for k in SERIES_KEYS}
            )
            for k in SERIES_KEYS:
                slot[k].append(float(rec[k]["relative_l2"]))
            argmax_h.setdefault(h, []).append(float(rec[LOGIT_EFFECT]["argmax_agreement"]))
    per_horizon: list[dict[str, Any]] = []
    for h in sorted(by_h):
        rec = {"horizon": h}
        for k in SERIES_KEYS:
            vals = by_h[h][k]
            rec[k] = {
                "relative_l2": float(max(vals)),
                "relative_l2_mean": float(sum(vals) / len(vals)),
                "relative_l2_max": float(max(vals)),
                "n_sequences": len(vals),
            }
        rec[LOGIT_EFFECT]["argmax_agreement"] = float(
            sum(argmax_h[h]) / len(argmax_h[h])
        )
        rec[LOGIT_EFFECT]["argmax_agreement_min"] = float(min(argmax_h[h]))
        rec[LOGIT_EFFECT]["argmax_is_not_parity"] = True
        require_separate_series(rec)
        per_horizon.append(rec)
    run = [int(r["horizon"]) for r in per_horizon]
    # A horizon that ran on at least one sequence is RUN. Skip only if none ran.
    skipped = [skipped_union[h] for h in sorted(skipped_union) if h not in run_union]
    named = require_named_horizons(
        required=req, plus=plus_h, run=run, skipped=skipped
    )
    series: dict[str, list[dict[str, Any]]] = {k: [] for k in SERIES_KEYS}
    for rec in per_horizon:
        h = int(rec["horizon"])
        for k in SERIES_KEYS:
            series[k].append({"horizon": h, **rec[k]})
    shapes = {
        k: classify_shape({int(r["horizon"]): float(r["relative_l2"]) for r in series[k]})
        for k in SERIES_KEYS
        if series[k]
    }
    overall = overall_shape(shapes)
    return {
        "n_sequences": len(list(curves)),
        "horizons_required": list(req),
        "horizons_plus": list(plus_h),
        "horizons_run": run,
        "horizons_skipped": skipped,
        "horizons_named": named,
        "per_horizon": per_horizon,
        "series": series,
        "shape": {"by_series": shapes, "overall": overall},
        "authority": "max_relative_l2_across_held_out_prompts",
        "argmax_is_not_parity": True,
        "one_step_only_admissible": False,
    }


# ---------------------------------------------------------------------------
# Economics. A ratio is not a candidate.
# ---------------------------------------------------------------------------


def score_candidate_economics(
    *,
    cand_id: str,
    bytes_removed: Any,
    bytes_added: Any,
    extra_flops_per_output_element: float = 0.0,
    dispatch_delta: float = 0.0,
    consuming_primitive: str,
    stream_class: str,
    status: str | None = None,
    reusable_family: bool = False,
    high_information_falsifier: bool = True,
) -> dict[str, Any]:
    if bytes_removed is None or bytes_added is None:
        missing = [k for k, v in (("bytes_removed", bytes_removed), ("bytes_added", bytes_added)) if v is None]
        raise MissingEconomics(cand_id, missing=missing)
    if consuming_primitive not in ATLAS_PRIMITIVES:
        raise DeltaNetMultistepRefuse(
            f"{cand_id}: {consuming_primitive} is not an atlas primitive"
        )
    removed = bytes_removed
    added = bytes_added
    if isinstance(bytes_removed, Mapping):
        if "total" not in bytes_removed:
            raise MissingEconomics(cand_id, missing=("bytes_removed.total",))
        removed = int(bytes_removed["total"])
    if isinstance(bytes_added, Mapping) and "total" in bytes_added:
        added = {
            k: int(v)
            for k, v in bytes_added.items()
            if k != "total"
        }
    row = ee.score(
        bytes_removed=int(removed),
        bytes_added=added,
        extra_flops_per_output_element=float(extra_flops_per_output_element),
        dispatch_delta=float(dispatch_delta),
        consuming_primitive=consuming_primitive,
        stream_class=stream_class,
        organ="deltanet",
        reusable_family=bool(reusable_family),
        high_information_falsifier=bool(high_information_falsifier),
        candidate_id=cand_id,
        status=status,
    )
    # Lane MATERIAL bar is 1 ms or 5% of model bytes or family or falsifier.
    net = int(row["net_bytes"])
    saved = -net if net < 0 else 0
    five_pct_bytes = int(TOKEN_ACTIVE_TARGET * 0.05)
    ms_saved = float(row["predicted_ms_saved"])
    lane_material = bool(
        ms_saved >= 1.0
        or saved >= five_pct_bytes
        or reusable_family
        or high_information_falsifier
    )
    row["lane_material"] = {
        "bar": (
            "MATERIAL means >= 1 ms of cited token time, or >= 5% of model "
            "bytes, or a reusable family, or a decisive falsifier. Cited "
            f"token time is {CITED_TOKEN_MS} ms; 5% of {TOKEN_ACTIVE_TARGET} "
            f"bytes is {five_pct_bytes}. This is not a hardware measurement."
        ),
        "clears": lane_material,
        "one_ms": ms_saved >= 1.0,
        "five_percent_bytes": saved >= five_pct_bytes,
        "reusable_family": bool(reusable_family),
        "decisive_falsifier": bool(high_information_falsifier),
    }
    return row


def generated_transition_claimed_economics() -> dict[str, Any]:
    """The t1dngen claim as recorded by DELTANET_STATE_FUNCTION. Not a fit."""
    removed = dsf.removed(catalog_weights=QKVZ_ACTIVE_TARGET)
    added = dsf.added(
        generator=2_924_624,
        embeddings=49_152,
        residuals=1_572_864,
        metadata=1_920,
    )
    return {
        "id": GENERATED_TRANSITION,
        "bytes_removed": removed,
        "bytes_added": added,
        "net_bytes": removed["total"] - added["total"],
        "dispatch_delta": 48,
        "consuming_primitive": "TiledProjection",
        "source": PREDECESSOR_STATE,
        "note": (
            "Claimed skinny program T2 T1 (5120→256→16384) plus per-layer f16 "
            "diagonals. ~2.14 GB removed, ~4.5 MB added. Capability UNMEASURED "
            "until a multi-step curve exists. One-step of this claim is not evidence."
        ),
    }


# ---------------------------------------------------------------------------
# Discovery. t1dngen may or may not have landed by the time we run.
# ---------------------------------------------------------------------------


def _read_rel(rel: str) -> tuple[dict[str, Any] | None, str]:
    path = REPO / rel
    if path.is_file():
        try:
            return load_json(path), "disk"
        except (OSError, json.JSONDecodeError):
            return None, "unreadable"
    blob = git("show", f"HEAD:{rel}")
    if blob:
        try:
            return json.loads(blob), "git:HEAD"
        except json.JSONDecodeError:
            return None, "git-unreadable"
    return None, "missing"


def discover_landed_candidates() -> dict[str, Any]:
    """Look for a sibling generated-transition receipt or scratch payload."""
    hits: list[dict[str, Any]] = []
    for rel in LANDING_RECEIPT_RELS:
        doc, src = _read_rel(rel)
        hits.append(
            {
                "rel": rel,
                "source": src,
                "landed": doc is not None,
                "has_program": bool(
                    doc
                    and (
                        doc.get("program")
                        or doc.get("fitted")
                        or doc.get("T1")
                        or (doc.get("candidates") and any(
                            (c or {}).get("id") == GENERATED_TRANSITION
                            and (c or {}).get("fitted")
                            for c in (doc.get("candidates") or [])
                        ))
                    )
                ),
            }
        )
    scratch: list[dict[str, Any]] = []
    for path in LANDING_SCRATCH:
        scratch.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "is_dir": path.is_dir() if path.exists() else False,
            }
        )
    landed = any(h["landed"] and h["has_program"] for h in hits) or any(
        s["exists"] for s in scratch
    )
    return {
        "landed": landed,
        "status": "LANDED" if landed else NOT_LANDED,
        "receipts_probed": hits,
        "scratch_probed": scratch,
        "candidate_id": GENERATED_TRANSITION,
        "note": (
            "A generated-transition candidate is only evaluable when a fitted "
            "program (T1/T2 or equivalent) is on disk. A byte claim without a "
            "program is not a curve."
        ),
    }


# ---------------------------------------------------------------------------
# Evaluate one candidate on one or more sequences.
# ---------------------------------------------------------------------------


def evaluate_candidate(
    *,
    cand_id: str,
    step: StepFn,
    sequences: Sequence[Mapping[str, Any]],
    coeff_seed: int = RNG_SEED,
    n_heads: int = PROBE_HEADS,
    dim: int = PROBE_DIM,
    vocab: int = PROBE_VOCAB,
    required: Sequence[int] = REQUIRED_HORIZONS,
    plus: Sequence[int] = PLUS_HORIZONS,
    skip_for_cost: Sequence[int] = (),
    logit_offset: float = 0.0,
    economics: Mapping[str, Any],
    report_as: str = "held_out",
) -> dict[str, Any]:
    if report_as == "held_out":
        for seq in sequences:
            split = str(seq.get("split") or "")
            if split == "train":
                raise TrainReportedAsHeldOut(
                    f"REFUSED: candidate {cand_id} reports split=train as held_out"
                )
    w_logit = logit_matrix(n_heads=n_heads, dim=dim, vocab=vocab, seed=coeff_seed)
    curves: list[dict[str, Any]] = []
    seq_meta: list[dict[str, Any]] = []
    for seq in sequences:
        x = np.asarray(seq["x"], dtype=np.float32)
        bundle = coefficients_from_hidden(
            x, n_heads=n_heads, dim=dim, seed=coeff_seed
        )
        curve = roll_curve(
            candidate_step=step,
            coeffs=bundle,
            w_logit=w_logit,
            required=required,
            plus=plus,
            skip_for_cost=skip_for_cost,
            logit_offset=logit_offset,
            n_heads=n_heads,
            dim=dim,
        )
        curves.append(curve)
        seq_meta.append(
            {
                "prompt_id": seq.get("prompt_id"),
                "layer": seq.get("layer"),
                "split": seq.get("split"),
                "n_tokens": int(x.shape[0]),
                "capability_domain": seq.get("capability_domain"),
                "horizons_run": curve["horizons_run"],
                "shape": curve["shape"],
            }
        )
    agg = aggregate_curves(curves, required=required, plus=plus)
    econ = score_candidate_economics(
        cand_id=cand_id,
        bytes_removed=economics.get("bytes_removed"),
        bytes_added=economics.get("bytes_added"),
        extra_flops_per_output_element=float(
            economics.get("extra_flops_per_output_element") or 0.0
        ),
        dispatch_delta=float(economics.get("dispatch_delta") or 0.0),
        consuming_primitive=str(economics["consuming_primitive"]),
        # Each candidate must SAY which stream it takes bytes out of. Defaulting
        # to the organ average is how the aux-u8 overcredit stayed invisible, and
        # DeltaNet is where it would hide best: the recurrent state S is resident
        # activation read and written every token, while the qkvz payload and the
        # transition matrices are weight codes. Billing state bytes at the weight
        # rate, or the reverse, is a whole-organ error.
        stream_class=str(economics["stream_class"]),
        status=economics.get("status"),
        reusable_family=bool(economics.get("reusable_family")),
        high_information_falsifier=bool(
            economics.get("high_information_falsifier", True)
        ),
    )
    try:
        verdict = demand_fitted_heldout(agg, required=required)
        verdict_error = None
    except OneStepOnlyRefuse as exc:
        verdict = None
        verdict_error = {"class": "OneStepOnlyRefuse", "message": str(exc)}
    except IncompleteHorizonsRefuse as exc:
        verdict = None
        verdict_error = {
            "class": "IncompleteHorizonsRefuse",
            "message": str(exc),
            "missing": list(exc.missing),
        }
    except (CollapsedSeriesRefuse, ArgmaxIsNotParity) as exc:
        verdict = None
        verdict_error = {"class": type(exc).__name__, "message": str(exc)}

    status: str
    if verdict_error and verdict_error["class"] == "OneStepOnlyRefuse":
        raise OneStepOnlyRefuse(verdict_error["message"])
    if verdict is None:
        status = INCOMPLETE
    elif cand_id == IDENTITY:
        status = CONTROL
    else:
        status = str(verdict["status"])

    return {
        "id": cand_id,
        "report_as": report_as,
        "held_out_unit": "prompt_id",
        "n_sequences": len(sequences),
        "sequences": seq_meta,
        "curve": agg,
        "per_sequence_shapes": [s["shape"] for s in seq_meta],
        "economics": _py(econ),
        "dense_rematerialization": economics.get(
            "dense_rematerialization", DIRECT_CONSUME
        ),
        "verdict": verdict,
        "verdict_error": verdict_error,
        "status": status,
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "argmax_is_not_parity": True,
        "one_step_only_admissible": False,
        "coefficients": (
            "seeded STATIC map of real residual-stream X through the shader "
            "L2 recipe; not W_qkvz. Trained q/k/v trajectory UNMEASURED."
        ),
    }


def cheap_control_specs(*, dim: int = PROBE_DIM, seed: int = RNG_SEED) -> list[dict[str, Any]]:
    """Identity, truncated state, lower-rank transition. Proven on real numbers."""
    rec_uv_r16 = (
        dsf.N_DN_LAYERS * dsf.VALUE_HEADS * 2 * dim * 16 * F32_BYTES
    )  # U,V of a rank-16 store of S
    householder_r8 = int(
        dsf.structured_state_bytes(kind="orthogonal_householder", rank=8)["resident_bytes"]
    )
    return [
        {
            "id": IDENTITY,
            "name": "identity control (candidate = reference gated-delta)",
            "step": identity_step,
            "rank": None,
            "economics": {
                "bytes_removed": 0,
                "bytes_added": 0,
                "extra_flops_per_output_element": 0.0,
                "dispatch_delta": 0.0,
                "consuming_primitive": "LocalStateMachine",
                # All three take their bytes out of REC_STATE_RESIDENT: the
                # recurrent state S, read and written every token. That is
                # resident ACTIVATION, not weight codes, and billing it at the
                # weight rate would credit a whole-organ error.
                "stream_class": "activation",
                "status": "EXISTING_LEVER",
                "reusable_family": False,
                "high_information_falsifier": True,
                "dense_rematerialization": DIRECT_CONSUME,
            },
            "note": "Must be FLAT_ZERO. A non-zero identity curve is a harness bug.",
        },
        {
            "id": TRUNCATED_STATE,
            "name": "truncate S to rank 16 after every incumbent write, readout from the store",
            "step": make_truncated_state_step(16),
            "rank": 16,
            "economics": {
                "bytes_removed": dsf.removed(state=REC_STATE_RESIDENT),
                "bytes_added": dsf.added(state=rec_uv_r16, metadata=1_920),
                "extra_flops_per_output_element": 0.0,
                "dispatch_delta": 0.0,
                "consuming_primitive": "LocalStateMachine",
                # All three take their bytes out of REC_STATE_RESIDENT: the
                # recurrent state S, read and written every token. That is
                # resident ACTIVATION, not weight codes, and billing it at the
                # weight rate would credit a whole-organ error.
                "stream_class": "activation",
                "status": OPEN,
                "reusable_family": True,
                "high_information_falsifier": True,
                "dense_rematerialization": DIRECT_CONSUME,
            },
            "note": (
                "A rank-16 store of 128x128 is not closed under the rank-1 write "
                "(DELTANET_STATE_FUNCTION). Expanding U,V to dense S then running "
                "vi_simd is REJECTED_DENSE_REMAT; this control consumes the truncated "
                "S natively. The curve is the question."
            ),
        },
        {
            "id": LOWER_RANK_TRANSITION,
            "name": "project k onto a fixed rank-8 orthonormal basis before the incumbent write",
            "step": make_lower_rank_transition_step(
                orthonormal_basis(dim, 8, seed=seed)
            ),
            "rank": 8,
            "economics": {
                "bytes_removed": dsf.removed(state=REC_STATE_RESIDENT),
                "bytes_added": dsf.added(state=householder_r8, metadata=1_920),
                "extra_flops_per_output_element": 0.0,
                "dispatch_delta": 0.0,
                "consuming_primitive": "LocalStateMachine",
                # All three take their bytes out of REC_STATE_RESIDENT: the
                # recurrent state S, read and written every token. That is
                # resident ACTIVATION, not weight codes, and billing it at the
                # weight rate would credit a whole-organ error.
                "stream_class": "activation",
                "status": OPEN,
                "reusable_family": True,
                "high_information_falsifier": True,
                "dense_rematerialization": DIRECT_CONSUME,
            },
            "note": (
                "Lower-rank transition: the write key is forced into 8 directions. "
                "This is a different mixer, not a packing of W_qkvz. W is unchanged "
                "in the economics so the byte model is honest about S, not a fake "
                "qkvz win."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Snapshot / build.
# ---------------------------------------------------------------------------


def _capability_residual_budget() -> dict[str, Any]:
    return {
        "share_of_token": CAP_MAP_SHARE_OF_TOKEN,
        "bytes": CAP_MAP_BYTES,
        "source": CAP_MAP_REL,
        "use": (
            "ALLOCATE a residual bit budget. Not a byte win of this lane. "
            "Sensitivity is not uniform; at a strict bar it only licences "
            "0.28% of the token."
        ),
        "deltanet_slice_bytes": 15_728_640,
        "not_a_win": True,
    }


def _cited_budget() -> dict[str, Any]:
    return {
        "cited_resident_tps": CITED_RESIDENT_TPS,
        "cited_token_ms": CITED_TOKEN_MS,
        "cited_deltanet_ms": CITED_DN_MS,
        "cited_deltanet_bytes": DELTANET_ACTIVE_TARGET,
        "cited_token_bytes": TOKEN_ACTIVE_TARGET,
        "mlp_bytes": ee.MLP_ACTIVE_BYTES,
        "mlp_ms": ee.MLP_MS,
        "source": BUDGET_REL,
        "note": (
            "Cited, not re-measured. Keys are cited_* so this sidecar cannot "
            "assert a hardware field (tps, token_ns, ...)."
        ),
    }


def snapshot(
    *,
    sequences: Sequence[Mapping[str, Any]] | None = None,
    n_heads: int = PROBE_HEADS,
    dim: int = PROBE_DIM,
    required: Sequence[int] = REQUIRED_HORIZONS,
    plus: Sequence[int] = PLUS_HORIZONS,
    skip_for_cost: Sequence[int] = (),
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    landed = discover_landed_candidates()
    corpus_error = None
    seqs: list[dict[str, Any]]
    input_kind: str
    if sequences is not None:
        seqs = list(sequences)
        input_kind = str(seqs[0].get("split") or "supplied")
    else:
        try:
            seqs = load_held_out_hidden_sequences(
                layer=DN_PROBE_LAYER, min_tokens=256, max_sequences=2
            )
            input_kind = "held_out_prompt_teacher_corpus_L38_post_attn_norm"
        except CorpusUnavailable as exc:
            corpus_error = str(exc)
            x = make_fixture_hidden(256, seed=seed)
            seqs = [
                {
                    "prompt_id": "FIXTURE_NOT_HELD_OUT",
                    "layer": None,
                    "split": "fixture",
                    "n_tokens": int(x.shape[0]),
                    "x": x,
                    "capability_domain": None,
                }
            ]
            input_kind = "fixture_ar_stream_NOT_held_out"
    report_as = "held_out" if input_kind.startswith("held_out") else "fixture"
    controls = []
    for spec in cheap_control_specs(dim=dim, seed=seed):
        row = evaluate_candidate(
            cand_id=spec["id"],
            step=spec["step"],
            sequences=seqs,
            coeff_seed=seed,
            n_heads=n_heads,
            dim=dim,
            required=required,
            plus=plus,
            skip_for_cost=skip_for_cost,
            economics=spec["economics"],
            report_as=report_as,
        )
        row["name"] = spec["name"]
        row["note"] = spec["note"]
        row["rank"] = spec["rank"]
        controls.append(row)

    gen_claim = generated_transition_claimed_economics()
    gen_econ = score_candidate_economics(
        cand_id=GENERATED_TRANSITION,
        bytes_removed=gen_claim["bytes_removed"],
        bytes_added=gen_claim["bytes_added"],
        extra_flops_per_output_element=0.0,
        dispatch_delta=float(gen_claim["dispatch_delta"]),
        consuming_primitive=str(gen_claim["consuming_primitive"]),
        # A generated transition replaces stored transition coefficients: weights.
        stream_class=str(gen_claim.get("stream_class") or "weight_codes"),
        status=OPEN,
        reusable_family=True,
        high_information_falsifier=True,
    )
    generated = {
        "id": GENERATED_TRANSITION,
        "landed": landed["landed"],
        "status": NOT_LANDED if not landed["landed"] else UNMEASURED,
        "discovery": landed,
        "claimed_economics": _py(gen_claim),
        "economics": _py(gen_econ),
        "curve": None,
        "note": (
            "Sibling t1dngen had not landed a fitted program when this receipt "
            "was written. The byte claim is scored so it is comparable the "
            "moment a program arrives; the curve is NOT_LANDED, not a one-step "
            "proxy. Re-run this module against the fitted program; do not "
            "promote on a one-step number from that lane."
            if not landed["landed"]
            else (
                "A generated-transition artifact is on disk but this lane did "
                "not load a native step function for it. Curve left UNMEASURED "
                "rather than faked."
            )
        ),
        "one_step_only_admissible": False,
        "argmax_is_not_parity": True,
    }

    return {
        "input": {
            "kind": input_kind,
            "report_as": report_as,
            "layer": DN_PROBE_LAYER,
            "n_sequences": len(seqs),
            "prompt_ids": [s.get("prompt_id") for s in seqs],
            "n_tokens": [int(np.asarray(s["x"]).shape[0]) for s in seqs],
            "corpus_error": corpus_error,
            "n_heads": n_heads,
            "dim": dim,
            "coefficients": (
                "seeded STATIC map of X, shader L2 on q/k; not W_qkvz"
            ),
            "x_is_post_attn_norm": True,
            "x_is_mixer_input": False,
            "trained_qkv_trajectory": UNMEASURED,
        },
        "landed_candidates": landed,
        "controls": controls,
        "generated_transition": generated,
        "cited": _cited_budget(),
        "capability_residual_budget": _capability_residual_budget(),
    }


def build(
    *,
    sequences: Sequence[Mapping[str, Any]] | None = None,
    n_heads: int = PROBE_HEADS,
    dim: int = PROBE_DIM,
) -> Path:
    snap = snapshot(sequences=sequences, n_heads=n_heads, dim=dim)
    controls = snap["controls"]
    findings: list[str] = []
    for row in controls:
        shape = (row.get("curve") or {}).get("shape") or {}
        findings.append(
            f"{row['id']}: status={row['status']} shape={shape.get('overall')} "
            f"run={row['curve']['horizons_run']}"
        )
    gen = snap["generated_transition"]
    findings.append(
        f"{GENERATED_TRANSITION}: {gen['status']} (landed={gen['landed']})"
    )
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Build the multi-step evaluation authority for DeltaNet candidates "
            "and apply it. A recurrent claim validated at one step is not a "
            "claim. State, output and logit errors are three series. The "
            "divergence curve's shape is the finding."
        ),
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "predecessor": [PREDECESSOR_STATE, PREDECESSOR_REPR, PREDECESSOR_QKVZ],
        "acceptance": {
            "fitted_heldout_rule": FITTED_HELDOUT_RULE,
            "required_horizons": list(REQUIRED_HORIZONS),
            "plus_horizons": list(PLUS_HORIZONS),
            "series": list(SERIES_KEYS),
            "state_bar": STATE_BAR,
            "output_bar": OUTPUT_BAR,
            "logit_bar": LOGIT_BAR,
            "cosine_bar": COSINE_BAR,
            "plateau_growth_bar": PLATEAU_GROWTH_BAR,
            "one_step_only_admissible": False,
            "argmax_is_not_parity": True,
            "held_out_unit": "prompt_id",
            "silent_horizon_omission": "REFUSED",
            "collapsed_series": "REFUSED",
        },
        "what_this_does_not_prove": [
            "capability or generate identity of any candidate",
            "that the STATIC coefficient map is W_qkvz @ x",
            "that post_attn_norm X is the mixer input",
            "physical EBPW or a protected TPS number",
            "that t1dngen's byte claim holds: it had not landed a program, or the curve is UNMEASURED",
        ],
        "cited": snap["cited"],
        "capability_residual_budget": snap["capability_residual_budget"],
        "input": {
            k: v
            for k, v in snap["input"].items()
            if k != "x"
        },
        "landed_candidates": snap["landed_candidates"],
        "controls": _py(controls),
        "generated_transition": _py(gen),
        "findings": findings,
        "answers": {
            "is_one_step_admissible": {
                "answer": "NO. demand_fitted_heldout raises OneStepOnlyRefuse.",
                "one_step_only_admissible": False,
            },
            "is_argmax_parity": {
                "answer": "NO. logit_effect.relative_l2 is required; argmax agreement is not parity.",
                "argmax_is_not_parity": True,
            },
            "what_shape_can_promote": {
                "answer": (
                    "Only PLATEAU (or FLAT_ZERO of a real candidate that also "
                    "clears the absolute bars on all three series at every "
                    "required horizon). COMPOUNDING cannot support FITTED_HELDOUT. "
                    "ONSET_THEN_PLATEAU cannot either: a family that is exact "
                    "until rank fills and then sits is the failure mode one-step "
                    "validation exists to miss."
                ),
                "promotable": [PLATEAU, FLAT_ZERO],
                "not_promotable": [COMPOUNDING, ONSET_THEN_PLATEAU],
            },
            "did_t1dngen_land": {
                "answer": "YES" if gen["landed"] else "NO",
                "status": gen["status"],
            },
            "do_cheap_controls_plateau": {
                "answer": {
                    row["id"]: (row.get("curve") or {}).get("shape", {}).get("overall")
                    for row in controls
                },
                "note": (
                    "Identity must be FLAT_ZERO (harness check). Truncated rank-16 "
                    "is the load-bearing negative: relative L2 is 0 at horizons "
                    "1, 4 and 16 (rank(S) has not yet exceeded 16) and then sits "
                    "near 0.1 — a one-step or 16-step number would have promoted "
                    "it. Lower-rank transition is wrong from step 1 and plateaus "
                    "there (a different mixer, not a diverging approximation)."
                ),
            },
        },
        "nomenclature": {
            "fitted_heldout": FITTED_HELDOUT,
            "measured_negative": MEASURED_NEGATIVE,
            "incomplete": INCOMPLETE,
            "control": CONTROL,
            "not_landed": NOT_LANDED,
            "plateau": PLATEAU,
            "compounding": COMPOUNDING,
            "onset_then_plateau": ONSET_THEN_PLATEAU,
            "flat_zero": FLAT_ZERO,
            "run": RUN,
            "skipped_for_cost": SKIPPED_FOR_COST,
            "skipped_insufficient_sequence": SKIPPED_INSUFFICIENT_SEQUENCE,
            "series": list(SERIES_KEYS),
            "static_only": "this sidecar. Models propose; protected deterministic evidence decides.",
        },
        "s020_families": families_report(),
        "gaps_closed": [
            "the six S020 families are enumerated with judged/blocked status, so "
            "an unjudged family cannot read as judged by omission",
            "state, output and logit errors are three series; collapsing them raises",
            "a one-step-only number raises rather than returning a verdict",
            "every named horizon is RUN or SKIPPED; silent omission raises",
            "argmax agreement is reported and is not treated as parity",
            "held-out is by prompt_id; train reported as held-out raises",
            "every candidate is scored by executable_economics (bytes_removed AND bytes_added)",
            "t1dngen is probed on disk; absence is NOT_LANDED, not a fake curve",
            "FITTED_HELDOUT rule is in the receipt, not tribal knowledge",
        ],
        "negative_findings": [
            "one-step accuracy is not a recurrent claim",
            "a candidate that drifts in logit space while preserving argmax is not validated",
            "the capability-information-map 0.28% is a residual budget, not a DeltaNet byte win",
            "ordinary entropy coding of MLP codes is a different organ and is not this school",
            "COMPOSITE_MLP_SIMPLE_LINEAR_LOW_RANK is a different organ and is already refuted",
        ],
        "recovered_implementation": {
            "gated_delta": "S := (I - beta k k^T)(decay S) + beta k v^T; h := S^T q",
            "kernel": "qwen38_gated_delta_decode_vi_simd",
            "authority": RECORDED_BY,
            "horizons": list(ALL_NAMED_HORIZONS),
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


selftest = build


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else _sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv_list)
    if args.build or args.selftest or not argv_list:
        out = build()
        print(out)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(_sys.argv[1:]))


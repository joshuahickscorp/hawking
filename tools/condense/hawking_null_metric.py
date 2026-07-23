#!/usr/bin/env python3
"""The null-corrected promotion metric, for every parent and every representation.

Raw activation cosine promoted candidates that lose to predicting a constant.  A GLM-5.2
block output has a constant-mean raw cosine near 0.898, so a family scoring 0.87 was
reported as "87 percent fidelity" while being worse than storing one vector.  Every number
this module returns is measured against a null fitted on the fit split, so that failure
mode cannot recur silently.

The contract:

* ``centered_cosine`` is the promotion cosine.  The mean is removed using the FIT split
  mean, never the held-out mean, because subtracting a statistic of the evaluation set is
  how a null gets smuggled into the candidate.
* ``skill`` is ``1 - SSE(candidate)/SSE(mean-null)``, which is zero for a candidate no
  better than the constant and one for an exact reproduction.  It is signed: a candidate
  worse than the constant reports a negative number rather than a small positive cosine.
* ``skill_lower`` is a bootstrap lower confidence bound over held-out positions.  A
  candidate promotes on the bound, not the point estimate.
* ``raw_cosine`` is reported and is diagnostic only.

    selftest
"""
from __future__ import annotations

import json
import numpy as np

SCHEMA = "hawking.null_corrected_metric.v1"
# Promotion thresholds, frozen here rather than at each call site so a campaign cannot
# quietly relax them.  Skill must be positive with a positive lower bound; the centered
# gate is what "meaningfully better than a constant" means in cosine terms.
GATE_SKILL_LOWER = 0.0
GATE_CENTERED_COSINE = 0.5
BOOTSTRAP_RESAMPLES = 512


def _flat(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    return array.reshape(-1, array.shape[-1])


def fit_null(y_fit: np.ndarray) -> dict:
    """Everything the held-out score is allowed to know about the target distribution."""
    flat = _flat(y_fit)
    return {"mean": flat.mean(axis=0), "count": int(flat.shape[0])}


def _sse_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    difference = a - b
    return np.einsum("ij,ij->i", difference, difference)


def _bootstrap_skill_lower(sse_candidate: np.ndarray, sse_null: np.ndarray,
                           *, resamples: int, alpha: float, seed: int) -> float:
    """Percentile lower bound on the ratio of sums, resampling held-out rows.

    The statistic is a ratio of sums rather than a mean of ratios, because a row whose
    null SSE is near zero would otherwise dominate.
    """
    generator = np.random.default_rng(seed)
    count = sse_candidate.shape[0]
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        pick = generator.integers(0, count, count)
        draws[index] = 1.0 - sse_candidate[pick].sum() / max(sse_null[pick].sum(), 1e-30)
    return float(np.quantile(draws, alpha))


def score(y_true: np.ndarray, y_pred: np.ndarray, null: dict, *,
          alpha: float = 0.05, seed: int = 0,
          resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Score one candidate against a null that was fitted elsewhere."""
    truth = _flat(y_true)
    prediction = _flat(y_pred)
    if truth.shape != prediction.shape:
        raise ValueError(f"shape mismatch: {truth.shape} vs {prediction.shape}")
    mean = np.asarray(null["mean"], dtype=np.float64)

    centred_truth = truth - mean
    centred_prediction = prediction - mean
    denominator = max(float(np.linalg.norm(centred_truth))
                      * float(np.linalg.norm(centred_prediction)), 1e-30)
    centered_cosine = float(np.tensordot(centred_truth, centred_prediction, axes=2)
                            / denominator)

    raw_denominator = max(float(np.linalg.norm(truth)) * float(np.linalg.norm(prediction)),
                          1e-30)
    raw_cosine = float(np.tensordot(truth, prediction, axes=2) / raw_denominator)

    sse_candidate = _sse_per_row(prediction, truth)
    sse_null = _sse_per_row(np.broadcast_to(mean, truth.shape), truth)
    total_null = max(float(sse_null.sum()), 1e-30)
    skill = float(1.0 - sse_candidate.sum() / total_null)
    lower = _bootstrap_skill_lower(sse_candidate, sse_null, resamples=resamples,
                                   alpha=alpha, seed=seed)

    truth_norm = max(float(np.linalg.norm(truth)), 1e-30)
    relative_l2 = float(np.linalg.norm(prediction - truth) / truth_norm)
    rmse = float(np.sqrt(sse_candidate.sum() / truth.size))
    normalized_rmse = float(rmse / max(float(truth.std()), 1e-30))

    per_row_skill = 1.0 - sse_candidate / np.maximum(sse_null, 1e-30)
    return {
        "centered_cosine": centered_cosine,
        "raw_cosine": raw_cosine,
        "skill": skill,
        "skill_lower": lower,
        "alpha": alpha,
        "relative_l2": relative_l2,
        "rmse": rmse,
        "normalized_rmse": normalized_rmse,
        "positions": int(truth.shape[0]),
        "per_position_skill": {
            "p05": float(np.quantile(per_row_skill, 0.05)),
            "median": float(np.median(per_row_skill)),
            "p95": float(np.quantile(per_row_skill, 0.95)),
            "fraction_beating_null": float((per_row_skill > 0).mean()),
        },
        "passes": bool(lower > GATE_SKILL_LOWER
                       and centered_cosine >= GATE_CENTERED_COSINE),
        "gate": {"skill_lower_above": GATE_SKILL_LOWER,
                 "centered_cosine_at_least": GATE_CENTERED_COSINE},
        "schema": SCHEMA,
    }


def constant_null_raw_cosine(y_true: np.ndarray, null: dict) -> float:
    """What the broken metric would have said about predicting the fit-split constant."""
    truth = _flat(y_true)
    mean = np.broadcast_to(np.asarray(null["mean"], dtype=np.float64), truth.shape)
    return float(np.tensordot(truth, mean, axes=2)
                 / max(float(np.linalg.norm(truth)) * float(np.linalg.norm(mean)), 1e-30))


def selftest() -> int:
    generator = np.random.default_rng(0)
    count, width = 4096, 64
    # A target with a large constant component: the case where raw cosine lies.
    offset = generator.standard_normal(width) * 5.0
    signal = generator.standard_normal((count, width))
    y = offset + signal
    y_fit = offset + generator.standard_normal((count, width))
    null = fit_null(y_fit)

    # Predicting the fit-split constant must score exactly zero skill and a high raw
    # cosine.  That gap is the entire reason this module exists.
    constant = np.broadcast_to(null["mean"], y.shape)
    flat = score(y, constant, null)
    assert abs(flat["skill"]) < 1e-9, flat["skill"]
    assert flat["raw_cosine"] > 0.9, flat["raw_cosine"]
    assert abs(flat["centered_cosine"]) < 0.2, flat["centered_cosine"]
    assert not flat["passes"]

    # A candidate strictly worse than the constant must report negative skill, where raw
    # cosine would still report a large positive number.
    worse = constant + generator.standard_normal(y.shape) * 3.0
    bad = score(y, worse, null)
    assert bad["skill"] < 0, bad["skill"]
    assert bad["raw_cosine"] > 0.8, bad["raw_cosine"]
    assert not bad["passes"]

    # A candidate that recovers most of the varying part must pass on the lower bound.
    good = offset + signal * 0.95 + generator.standard_normal(y.shape) * 0.15
    strong = score(y, good, null)
    assert strong["skill"] > 0.8, strong["skill"]
    assert strong["skill_lower"] > 0.8, strong["skill_lower"]
    assert strong["centered_cosine"] > 0.9, strong["centered_cosine"]
    assert strong["passes"]
    assert strong["skill_lower"] <= strong["skill"] + 1e-6

    # An exact reproduction is skill 1.
    exact = score(y, y, null)
    assert abs(exact["skill"] - 1.0) < 1e-9
    assert abs(exact["centered_cosine"] - 1.0) < 1e-9

    # The null must be fitted, not read off the target.  A constant taken from the
    # held-out set beats the fit-split constant, so a candidate allowed to peek would earn
    # positive skill for having learned nothing.  That is the smuggling this forbids.
    peeked = np.broadcast_to(fit_null(y)["mean"], y.shape)
    assert score(y, peeked, null)["skill"] > 0.0, "peeking at the target is not detectable"

    print(json.dumps({
        "selftest": "PASS",
        "constant_null_raw_cosine_example": round(flat["raw_cosine"], 4),
        "constant_null_skill": round(flat["skill"], 6),
        "gate": {"skill_lower_above": GATE_SKILL_LOWER,
                 "centered_cosine_at_least": GATE_CENTERED_COSINE},
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())

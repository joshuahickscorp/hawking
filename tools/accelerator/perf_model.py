"""A learned performance model. FRONT E (G047, steer S015 §113).

The bandit receipt named this as the unbuilt half: "NOT UCB, not Bayesian, not the
learned performance model §113 asks for". This is that model, and it is deliberately
the SMALLEST thing that can answer whether the idea pays on this landscape.

WHAT A PERFORMANCE MODEL IS FOR, and why this landscape makes it hard: the forge
already measured the top cluster FLAT -- the best 9 of 15 variants span 0.65% against
~4.5% arm noise. A model that predicts within 0.65% there is not accurate, it is
merely predicting a constant, and so is `predict the mean`. THE ONLY THING WORTH
PREDICTING ON A FLAT LANDSCAPE IS THE CLIFF: the forge's re-verdict found the whole
20.19% spread was driven by ONE configuration, tg64_ept1, the lowest-occupancy point.

So the model is graded on TWO questions kept separate, because a single average score
would let the flat region's easiness hide a failure on the only case that matters:

  1. On the FLAT points, does it beat `predict the training mean`? Expected: NO.
  2. On the CLIFF point, held out and never seen, does it predict SLOWER THAN EVERY
     FLAT POINT? That is the actionable question -- avoidance of a bad configuration
     without paying to measure it.

Ridge on 4 features, closed form, no dependencies beyond numpy. 15 points does not
support more, and a model with more parameters than the landscape has structure would
be fitting the noise the forge spent 200 reps per candidate measuring.
"""
from __future__ import annotations

import math


def features(threadgroup: int, ept: int) -> list[float]:
    """Four features chosen from the mechanism, not from a search.

    per_tg = threadgroup * ept is ELEMENTS PER THREADGROUP, and its RECIPROCAL is the
    feature that can express the occupancy cliff: tg64_ept1 has the unique minimum
    per_tg of 64, so a linear term in 1/per_tg is the only way a model trained on the
    flat region could extrapolate to it at all. Including it is a bet that the cliff is
    an occupancy effect; if the model still fails, that bet is what failed.
    """
    per_tg = threadgroup * ept
    return [1.0, math.log2(threadgroup), float(ept), 1.0 / per_tg]


def fit(rows: list[tuple[int, int, float]], *, ridge: float = 1e-6) -> list[float]:
    """rows are (threadgroup, ept, milliseconds). Closed-form ridge."""
    import numpy as np
    X = np.array([features(t, e) for t, e, _ in rows], dtype=np.float64)
    y = np.array([ms for _, _, ms in rows], dtype=np.float64)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    return list(np.linalg.solve(A, X.T @ y))


def predict(w: list[float], threadgroup: int, ept: int) -> float:
    return float(sum(a * b for a, b in zip(w, features(threadgroup, ept))))


def leave_one_out(rows: list[tuple[int, int, float]]) -> list[dict]:
    """Every point predicted by a model that never saw it, beside the two baselines a
    model has to beat to have any content at all."""
    import numpy as np
    out = []
    for i, (t, e, ms) in enumerate(rows):
        train = rows[:i] + rows[i + 1:]
        w = fit(train)
        ys = np.array([r[2] for r in train])
        out.append({"threadgroup": t, "ept": e, "actual_ms": ms,
                    "model_ms": predict(w, t, e),
                    "mean_baseline_ms": float(ys.mean()),
                    "median_baseline_ms": float(np.median(ys))})
    for r in out:
        for k in ("model", "mean_baseline", "median_baseline"):
            r[f"{k}_abs_err_pct"] = abs(r[f"{k}_ms"] - r["actual_ms"]) / r["actual_ms"] * 100
    return out


def grade(loo: list[dict], *, cliff_threshold_pct: float = 5.0,
          noise_pct: float = 4.5) -> dict:
    """Grade the two questions SEPARATELY. Averaging them would let 14 easy points
    drown the one that decides whether the model is worth anything."""
    import numpy as np
    fastest = min(r["actual_ms"] for r in loo)
    cliff = [r for r in loo
             if (r["actual_ms"] - fastest) / fastest * 100 >= cliff_threshold_pct]
    flat = [r for r in loo if r not in cliff]
    med = lambda rs, k: float(np.median([r[f"{k}_abs_err_pct"] for r in rs])) if rs else None
    flat_model, flat_mean = med(flat, "model"), med(flat, "mean_baseline")
    # the actionable test: does a model that never saw the cliff still rank it last?
    cliff_ranked_last = None
    if cliff:
        c = cliff[0]
        flat_preds = [r["model_ms"] for r in flat]
        cliff_ranked_last = bool(c["model_ms"] > max(flat_preds))
    return {
        "n_flat": len(flat), "n_cliff": len(cliff),
        "flat_median_abs_err_pct": flat_model,
        "flat_mean_baseline_err_pct": flat_mean,
        "flat_model_beats_baseline": (flat_model < flat_mean) if flat else None,
        "flat_both_inside_arm_noise": (flat_model is not None and
                                       flat_model < noise_pct and flat_mean < noise_pct),
        "cliff_abs_err_pct": (cliff[0]["model_abs_err_pct"] if cliff else None),
        "cliff_predicted_slower_than_every_flat_point": cliff_ranked_last,
        "means": "a model that beats the mean on the flat points by less than arm noise "
                 "has NOT been shown to predict anything; the cliff row is the one that "
                 "decides whether the model can avoid a bad configuration unmeasured",
    }

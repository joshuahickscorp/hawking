#!/usr/bin/env python3.12
"""Adversarial check on this campaign's own headline claim about the rate-distortion wall.

THE CLAIM UNDER ATTACK. `qwen_function_aware_probe` reported that the function-aware codec measures
rel_error 0.6435 at 0.625 index bpw against a "floor" of sqrt(2^-2R) = 0.6484, and this campaign
used that to seal Lane A as exhausted.

THE PROBLEM WITH IT. sqrt(2^-2R) is the Gaussian rate-distortion function, and among all sources of
a given variance the Gaussian is the WORST case - it maximizes differential entropy, so its D(R) is
an UPPER bound on the distortion any same-variance source requires. Matching it proves the codec is
as good as if the weights were Gaussian. It does NOT prove no codec can do better. Weight
distributions are famously heavy-tailed, and a heavy-tailed source has strictly lower differential
entropy at the same variance, hence a strictly lower achievable distortion at the same rate.

THE CORRECT FLOOR is the Shannon lower bound, which uses the source's ACTUAL differential entropy:

    D(R) >= (1 / (2 pi e)) * 2^(2h(X)) * 2^(-2R)

with h(X) the differential entropy per dimension in bits. For a Gaussian, 2^(2h) = 2 pi e sigma^2
and the bound collapses to sigma^2 2^(-2R), recovering the Gaussian expression exactly. For anything
more concentrated than Gaussian it is lower - possibly far lower.

So the honest question is not "did we hit the Gaussian number" but:

    gap_decades = log10( measured_mse / shannon_lower_bound_mse )

If that gap is near zero, Lane A really is exhausted and the campaign's pivot to structural and
source-changing methods is justified. If it is large, THERE IS HEADROOM LEFT IN POST-HOC CODING and
this campaign closed a lane it had no right to close.

h(X) is estimated two independent ways so a single estimator cannot carry the conclusion:
  * Kozachenko-Leonenko k-nearest-neighbour estimator on the real d-dimensional chunk vectors,
    which is what the codec actually quantizes (it captures intra-chunk dependence).
  * a per-scalar histogram estimate, as a sanity cross-check on the marginal only.

The KL estimator is the load-bearing one: the vector bound is what governs a d-dimensional VQ.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import qwen_function_aware_codec as FAC  # noqa: E402
from qwen_real_forward import SafetensorsIndexReader  # noqa: E402

SCHEMA = "hawking.gravity.shannon_bound.v1"
_LOG2E = 1.0 / math.log(2.0)


def differential_entropy_knn(v: np.ndarray, k: int = 3, max_n: int = 20000,
                             seed: int = 0) -> float:
    """Kozachenko-Leonenko differential entropy of rows of v, in BITS PER DIMENSION.

        h_nats = psi(n) - psi(k) + log(V_d) + (d/n) * sum_i log(eps_i)

    where eps_i is the distance to the i-th point's k-th neighbour and V_d is the volume of the
    unit d-ball. Subsampled to `max_n` points because the estimator is O(n^2) in memory here; the
    subsample is deterministic.
    """
    from scipy.special import digamma  # type: ignore

    rng = np.random.default_rng(seed)
    n_all, d = v.shape
    if n_all > max_n:
        v = v[rng.choice(n_all, max_n, replace=False)]
    n = v.shape[0]
    # pairwise distances in blocks; keep the k-th smallest nonzero per row
    eps = np.empty(n, dtype=np.float64)
    step = max(1, int(2_000_000 // max(1, n)))
    v64 = v.astype(np.float64)
    sq = (v64 * v64).sum(1)
    for i in range(0, n, step):
        blk = v64[i:i + step]
        d2 = sq[i:i + step, None] - 2.0 * (blk @ v64.T) + sq[None, :]
        np.maximum(d2, 0.0, out=d2)
        part = np.partition(d2, k, axis=1)[:, :k + 1]
        part.sort(axis=1)
        eps[i:i + step] = np.sqrt(part[:, k])           # k-th neighbour, excluding self at index 0
    eps = np.maximum(eps, 1e-30)
    log_vd = (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1.0)
    h_nats = float(digamma(n) - digamma(k) + log_vd + (d / n) * np.log(eps).sum())
    return h_nats * _LOG2E / d                          # bits per dimension


def differential_entropy_hist(x: np.ndarray, bins: int = 4096) -> float:
    """Per-scalar differential entropy in bits, by histogram. Cross-check only."""
    a = np.asarray(x, np.float64).ravel()
    lo, hi = np.percentile(a, [0.01, 99.99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")
    hist, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    w = edges[1] - edges[0]
    p = hist[hist > 0] * w
    return float(-(p * np.log2(hist[hist > 0])).sum() * 1.0) if p.size else float("nan")


def shannon_lower_bound_mse(h_bits_per_dim: float, rate_bits_per_dim: float) -> float:
    """D(R) >= (1/(2 pi e)) 2^(2h) 2^(-2R), h and R both in bits per dimension."""
    return (1.0 / (2.0 * math.pi * math.e)) * (2.0 ** (2.0 * h_bits_per_dim)) * \
           (2.0 ** (-2.0 * rate_bits_per_dim))


def analyse(mats: list[np.ndarray], *, dim: int, k: int, stages: int, seed: int = 0,
            iters: int = 6) -> dict[str, Any]:
    """Measured codec MSE vs the Gaussian value vs the true Shannon lower bound, same rate."""
    rate_per_vector = stages * math.log2(k)
    rate_per_dim = rate_per_vector / dim

    # what the codec actually achieves, in the normalized space it actually codes
    books = FAC.fit(mats, dim=dim, k=k, stages=stages, seed=seed, row_scale=True, iters=iters)
    mses, var = [], []
    chunks = []
    for m in mats:
        rec = FAC.apply_refit(books, m, dim=dim)
        u, s = FAC.normalize_rows(m)
        ur, _ = FAC.normalize_rows(rec, s)                # compare in the coded (unit-RMS) space
        mses.append(float(np.mean((u - ur) ** 2)))
        var.append(float(np.var(u)))
        chunks.append(u.reshape(-1, dim))
    measured_mse = float(np.mean(mses))
    sigma2 = float(np.mean(var))

    pool = np.concatenate(chunks, 0)
    h_knn = differential_entropy_knn(pool, seed=seed)
    h_hist = differential_entropy_hist(pool)
    h_gauss = 0.5 * math.log2(2.0 * math.pi * math.e * sigma2)

    slb = shannon_lower_bound_mse(h_knn, rate_per_dim)
    gauss = sigma2 * (2.0 ** (-2.0 * rate_per_dim))
    return {
        "rate_bits_per_dim": rate_per_dim, "dim": dim, "k": k, "stages": stages,
        "sigma2_coded_space": round(sigma2, 6),
        "h_bits_per_dim": {"knn_kozachenko_leonenko": round(h_knn, 6),
                           "histogram_marginal": round(h_hist, 6),
                           "gaussian_at_same_variance": round(h_gauss, 6),
                           "non_gaussianity_bits": round(h_gauss - h_knn, 6)},
        "mse": {"measured": round(measured_mse, 8),
                "gaussian_worst_case_D_R": round(gauss, 8),
                "shannon_lower_bound": round(slb, 8)},
        "rel_error": {"measured": round(math.sqrt(measured_mse / max(sigma2, 1e-30)), 6),
                      "gaussian_worst_case": round(math.sqrt(gauss / max(sigma2, 1e-30)), 6),
                      "shannon_lower_bound": round(math.sqrt(slb / max(sigma2, 1e-30)), 6)},
        "headroom": {
            "gap_to_shannon_decades": round(math.log10(max(measured_mse, 1e-30) /
                                                       max(slb, 1e-30)), 4),
            "max_rel_error_improvement_factor": round(
                math.sqrt(max(measured_mse, 1e-30) / max(slb, 1e-30)), 4)},
    }


def verdict(a: dict[str, Any]) -> dict[str, Any]:
    """Is Lane A actually exhausted, or did this campaign close it too early?"""
    gap = a["headroom"]["gap_to_shannon_decades"]
    if gap < 0.15:
        v = "LANE_A_EXHAUSTED"
        why = ("measured MSE is within 0.15 decades of the true Shannon lower bound for the "
               "measured source entropy; there is no meaningful post-hoc headroom left")
    elif gap < 0.5:
        v = "LANE_A_NEARLY_EXHAUSTED"
        why = ("some headroom exists but under a factor of ~1.8 in rel_error; it cannot bridge the "
               "gap from 0.64 to a survivable reconstruction")
    else:
        v = "LANE_A_NOT_CLOSED"
        why = ("the source is materially non-Gaussian and a better post-hoc codec is provably "
               "possible; sealing Lane A on the Gaussian number was PREMATURE")
    return {"verdict": v, "why": why, "gap_to_shannon_decades": gap}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Shannon lower bound vs measured codec MSE.")
    ap.add_argument("--source", default="models/qwen3-235b-a22b")
    ap.add_argument("--layers", default="0,46,93")
    ap.add_argument("--organs", default="gate,down")
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    specs = {"gate": (8, 1024, 2), "up": (8, 1024, 2), "down": (16, 1024, 1)}
    suffix = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
    r = SafetensorsIndexReader(args.source)
    if not r.source_present():
        raise SystemExit("source shards absent")
    cells = []
    for L in (int(x) for x in args.layers.split(",")):
        for organ in args.organs.split(","):
            mats = [r.bf16(f"model.layers.{L}.mlp.experts.{e}.{suffix[organ]}.weight"
                           ).astype(np.float32) for e in range(args.experts)]
            dim, k, st = specs[organ]
            a = analyse(mats, dim=dim, k=k, stages=st)
            a["cell"] = {"layer": L, "organ": organ, "n_experts": args.experts}
            a["verdict"] = verdict(a)
            cells.append(a)
            del mats
    gaps = [c["headroom"]["gap_to_shannon_decades"] for c in cells]
    agg = {"mean_gap_decades": round(float(np.mean(gaps)), 4),
           "max_gap_decades": round(float(np.max(gaps)), 4)}
    agg.update(verdict({"headroom": {"gap_to_shannon_decades": agg["mean_gap_decades"]}}))
    out = {"schema": SCHEMA, "claim_under_attack":
           "that matching sqrt(2^-2R) proves post-hoc coding is exhausted",
           "aggregate": agg, "cells": cells}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(agg, indent=2, sort_keys=True))
    return 0


def demo() -> None:
    """Self-check: a Gaussian source must reproduce the Gaussian bound; a spiky one must beat it."""
    rng = np.random.default_rng(0)
    g = rng.standard_normal((512, 256)).astype(np.float32)
    a = analyse([g], dim=8, k=256, stages=1, iters=6)
    hk = a["h_bits_per_dim"]["knn_kozachenko_leonenko"]
    hg = a["h_bits_per_dim"]["gaussian_at_same_variance"]
    assert abs(hk - hg) < 0.15, (hk, hg)          # KL estimator recovers the Gaussian entropy
    # a heavy-tailed source must measure LOWER differential entropy at the same variance
    t = (rng.standard_normal((512, 256)) * rng.standard_gamma(0.4, (512, 256))).astype(np.float32)
    b = analyse([t], dim=8, k=256, stages=1, iters=6)
    assert (b["h_bits_per_dim"]["gaussian_at_same_variance"]
            - b["h_bits_per_dim"]["knn_kozachenko_leonenko"]) > \
           (hg - hk), "heavy tails must show more non-Gaussianity than a Gaussian does"
    assert b["headroom"]["gap_to_shannon_decades"] > a["headroom"]["gap_to_shannon_decades"]
    print(json.dumps({"ok": True, "gaussian_gap": a["headroom"]["gap_to_shannon_decades"],
                      "heavytail_gap": b["headroom"]["gap_to_shannon_decades"],
                      "gaussian_h_err": round(abs(hk - hg), 4)}, indent=2))


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        raise SystemExit(main())

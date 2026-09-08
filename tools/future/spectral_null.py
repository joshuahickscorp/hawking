"""A participation ratio means nothing until you know what noise scores.

The anatomy tool ships SHARING_LIVE_BELOW = 0.95 against ratio = participation/n.
Centred independent rows score EXACTLY (n-1)/n -- 0.750 at n=4, 0.950 at n=20,
0.992 at n=128 -- because centring removes one degree of freedom and nothing
else is shared. So 0.95 IS the noise floor at n=20, and the shipped rule reports
"sharing is LIVE" on pure noise for every n <= 20.

Two corrections follow:

  1. Normalise by (n-1), not n. Then independent rows score 1.0000 at every n and
     the threshold stops depending on how many experts a body happens to have.

  2. Even that is not enough, because ROW-NORM HETEROGENEITY depresses the ratio
     without any shared direction. Qwen3-30B-A3B's layer-0 experts score 0.9796,
     which reads as a 1.27% deficit from independence -- but a Gaussian null with
     the SAME per-row norms scores 0.9797. The deficit is entirely norm spread.
     The experts are exactly as independent as random vectors of matching size.

The honest statistic is therefore the deficit against a norm-matched null, not a
comparison to a constant.
"""
from __future__ import annotations

import mlx.core as mx

# Feature-block width for the Gram accumulation. A single command buffer over a
# (512, 3.3M) matmul exceeds Metal's watchdog and raises kIOGPUCommandBufferCallbackErrorTimeout.
GRAM_BLOCK = 1 << 22


def centred_gram(X: "mx.array", block: int = GRAM_BLOCK) -> "mx.array":
    """(n,n) Gram of column-centred rows, accumulated in feature blocks."""
    mu = mx.mean(X, axis=0, keepdims=True)
    n, D = X.shape
    G = mx.zeros((n, n), dtype=mx.float32)
    for s in range(0, D, block):
        B = X[:, s:s + block] - mu[:, s:s + block]
        G = G + (B @ B.T)
        mx.eval(G)                      # bound each command buffer
    return G / max(D, 1)


def participation(G: "mx.array") -> float:
    """exp of the spectral entropy of a Gram: the effective number of directions."""
    ev = mx.maximum(mx.sort(mx.linalg.eigvalsh(G.astype(mx.float32), stream=mx.cpu))[::-1], 0.0)
    tot = float(mx.sum(ev, stream=mx.cpu))
    p = ev / max(tot, 1e-30)
    ent = float(-mx.sum(mx.where(p > 0, p * mx.log(mx.maximum(p, 1e-30)), 0.0), stream=mx.cpu))
    return float(mx.exp(mx.array(ent)).item())


def norm_matched_null(X: "mx.array", seed: int = 5) -> "mx.array":
    """Gaussian rows carrying the SAME per-row norms as X.

    Matching norms is the point. An unmatched Gaussian null scores 0.9921 where
    the real experts score 0.9796, which would license a 1.27% "shared structure"
    claim that is purely an artefact of unequal row magnitudes.
    """
    norms = mx.sqrt(mx.sum(X * X, axis=1, keepdims=True))
    R = mx.random.normal(X.shape, key=mx.random.key(seed))
    R = R / mx.sqrt(mx.sum(R * R, axis=1, keepdims=True)) * norms
    mx.eval(R)
    return R


def row_norms(X: "mx.array") -> "mx.array":
    """Per-row norms -- the only thing the null needs from X."""
    r = mx.sqrt(mx.sum(X * X, axis=1, keepdims=True))
    mx.eval(r)
    return r


def null_from_norms(norms: "mx.array", d: int, seed: int = 5) -> float:
    """Participation of a Gaussian null carrying `norms`, built WITHOUT X.

    Holding the real stack and its null at once doubles peak memory for no
    reason: the null needs the norms and the shape, nothing else. On GLM-4.5-Air
    that is the difference between 9.5 GB and 5.5 GB of expectation, which is
    the difference between the campaign guard refusing and permitting the run
    while two Grok lanes hold the rest of the machine.
    """
    n = int(norms.shape[0])
    R = mx.random.normal((n, d), key=mx.random.key(seed))
    R = R / mx.sqrt(mx.sum(R * R, axis=1, keepdims=True)) * norms
    mx.eval(R)
    return participation(centred_gram(R))


def calibrated(X: "mx.array", seed: int = 5) -> dict:
    """Participation of X, of its norm-matched null, and the deficit between them.

    Convenience for callers that can afford both at once. A caller under memory
    pressure should use row_norms(), free X, then null_from_norms(), and assemble
    the same dict with deficit().
    """
    n = int(X.shape[0])
    if n < 3:
        raise ValueError(f"participation needs at least 3 rows to be meaningful, got {n}")
    real = participation(centred_gram(X))
    null = participation(centred_gram(norm_matched_null(X, seed)))
    return {"n": n, "d": int(X.shape[1]),
            "participation": round(real, 3),
            "ratio_n": round(real / n, 4),                 # what the old rule compared to 0.95
            "ratio_n_minus_1": round(real / (n - 1), 4),   # 1.0 for independent rows at any n
            "null_participation": round(null, 3),
            "null_ratio_n": round(null / n, 4),
            "deficit_pct": round(100 * (null - real) / max(null, 1e-30), 3),
            "noise_floor_ratio_n": round((n - 1) / n, 4)}


def deficit(real: float, null: float, n: int, d: int) -> dict:
    """Assemble the same result dict from separately measured halves."""
    return {"n": n, "d": d,
            "participation": round(real, 3),
            "ratio_n": round(real / n, 4),
            "ratio_n_minus_1": round(real / (n - 1), 4),
            "null_participation": round(null, 3),
            "null_ratio_n": round(null / n, 4),
            "deficit_pct": round(100 * (null - real) / max(null, 1e-30), 3),
            "noise_floor_ratio_n": round((n - 1) / n, 4)}


def verdict(c: dict, live_deficit_pct: float = 2.0) -> str:
    """Sharing is LIVE only if the real spectrum falls MEASURABLY below its own null."""
    if c["deficit_pct"] >= live_deficit_pct:
        return (f"sharing is LIVE: {c['deficit_pct']:.2f}% below a norm-matched null "
                f"(n={c['n']}) -- a shared basis may pay")
    return (f"sharing is DEAD: {c['deficit_pct']:.2f}% from a norm-matched null "
            f"(n={c['n']}, noise floor for the old p/n rule is {c['noise_floor_ratio_n']}) "
            f"-- these rows are as independent as random vectors of the same norms")


def _selfcheck() -> None:
    # 1. Centred independent rows score EXACTLY n-1. This is the whole argument.
    for n in (4, 8, 20, 64):
        R = mx.random.normal((n, 1 << 16), key=mx.random.key(n))
        mx.eval(R)
        c = calibrated(R)
        assert abs(c["ratio_n"] - (n - 1) / n) < 5e-3, (n, c)
        assert abs(c["ratio_n_minus_1"] - 1.0) < 5e-3, (n, c)
        assert abs(c["deficit_pct"]) < 1.0, (n, c)
        assert "DEAD" in verdict(c), (n, verdict(c))

    # 2. The OLD rule calls pure noise LIVE at n=20. That is the defect.
    R = mx.random.normal((20, 1 << 16), key=mx.random.key(20)); mx.eval(R)
    assert calibrated(R)["ratio_n"] <= 0.9505, calibrated(R)   # <= the shipped 0.95 threshold

    # 3. Genuinely shared structure must be DETECTED, or this is only a null check.
    #    Rows built from a rank-3 basis share almost everything.
    B = mx.random.normal((3, 1 << 16), key=mx.random.key(1))
    C = mx.random.normal((24, 3), key=mx.random.key(2))
    X = C @ B
    mx.eval(X)
    c = calibrated(X)
    assert c["deficit_pct"] > 50.0, c
    assert "LIVE" in verdict(c), verdict(c)

    # 4. Norm heterogeneity ALONE must not read as sharing.
    R = mx.random.normal((32, 1 << 16), key=mx.random.key(9))
    scale = mx.exp(mx.random.normal((32, 1), key=mx.random.key(10)) * 2.0)
    X = R * scale
    mx.eval(X)
    c = calibrated(X)
    assert abs(c["deficit_pct"]) < 2.0, ("norm spread misread as sharing", c)
    assert c["ratio_n"] < 0.95, ("...and the OLD rule would have called this LIVE", c)
    # 5. The two-phase path must give the SAME answer as the one-shot path,
    #    or the memory saving is bought with a different measurement.
    for Y in (mx.random.normal((32, 1 << 15), key=mx.random.key(21)), C @ B):
        mx.eval(Y)
        one = calibrated(Y, seed=5)
        nm = row_norms(Y)
        two = deficit(participation(centred_gram(Y)),
                      null_from_norms(nm, int(Y.shape[1]), seed=5),
                      int(Y.shape[0]), int(Y.shape[1]))
        assert one == two, ("two-phase diverged from one-shot", one, two)

    print("selfcheck OK -- noise floor is (n-1)/n; rank-3 basis detected at "
          f"{calibrated(C @ B)['deficit_pct']:.1f}% deficit; two-phase == one-shot")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()

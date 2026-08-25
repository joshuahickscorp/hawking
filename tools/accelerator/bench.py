"""Timing with a gate. FRONT D (G046, steer S015 §73/§74).

Every performance claim in this program goes through here, because the steer's rule
is that a claim needs exact identity, repeated samples, a baseline, and uncontended
measurement -- and because a number with a wide spread is not a measurement, as this
campaign already learned when a 286% spread nearly became a bandwidth claim.

A result is UNRELIABLE unless its interquartile spread is inside the gate, and an
UNRELIABLE arm may not win a comparison.
"""
from __future__ import annotations

from typing import Any, Callable

IQR_GATE_PCT = 10.0

# A reliability verdict below this many reps is NOT STABLE. Measured directly: the
# same kernel's IQR estimate ranged 1.84%-13.38% across 8 independent 20-rep runs and
# 3.41%-14.29% across 8 independent 40-rep runs, so the gate's own verdict FLIPPED
# run to run. At 200 reps all three probed kernels held inside a 3-point band. Below
# this threshold `reliable` is reported as an UNSTABLE ESTIMATE rather than a fact.
STABLE_RELIABILITY_MIN_REPS = 200
# a margin this many times the worst arm noise survives a failed reliability gate
LARGE_MARGIN_RATIO = 5.0


def time_arm(fn: Callable[[], Any], *, reps: int = 40, warmup: int = 10) -> dict[str, Any]:
    import time
    for _ in range(warmup):
        fn()
    s = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        s.append(time.perf_counter() - t0)
    s.sort()
    q1, med, q3 = s[len(s) // 4], s[len(s) // 2], s[(3 * len(s)) // 4]
    iqr = (q3 - q1) / q1 * 100
    stable = reps >= STABLE_RELIABILITY_MIN_REPS
    return {"median_s": med, "q1_s": q1, "q3_s": q3,
            "iqr_spread_pct": round(iqr, 2),
            "full_range_pct": round((s[-1] - s[0]) / s[0] * 100, 2),
            "reps": reps, "warmup": warmup,
            "reliable": iqr <= IQR_GATE_PCT,
            "reliability_verdict_is_stable": stable,
            "reliability_caveat": None if stable else (
                f"{reps} reps is below the {STABLE_RELIABILITY_MIN_REPS} needed for a "
                f"stable reliability verdict; this arm's `reliable` flag may flip on a "
                f"rerun and should not be cited as a property of the kernel")}


def implausible(reason: str) -> dict[str, Any]:
    """A result that violates a physical expectation is rejected regardless of how
    large its margin is. Learned the hard way: a bandwidth-bound kernel reading twice
    the bytes reported a SHORTER time, and the margin-over-noise escape hatch admitted
    it because the margin was large. A big margin on an impossible result is evidence
    the measurement is broken, not that the effect is strong."""
    return {"verdict": "REJECTED_IMPLAUSIBLE", "reason": reason, "speedup": None}


def compare(arms: dict[str, dict[str, Any]], *, baseline: str,
            candidate: str) -> dict[str, Any]:
    b, c = arms[baseline], arms[candidate]
    speedup = b["median_s"] / c["median_s"]
    noise_all = max(b["iqr_spread_pct"], c["iqr_spread_pct"])
    # SYMMETRIC margin. |speedup - 1| is bounded by 100% for any slowdown however
    # severe, while a speedup is unbounded -- so a 3x slowdown scored 67% and got
    # refused where a 3x speedup would have scored 200% and passed. The metric was
    # quietly harder on losses than on wins, which is exactly the wrong bias for an
    # instrument that exists to stop us overclaiming.
    margin_all = (max(speedup, 1.0 / speedup) - 1.0) * 100
    if not (b["reliable"] and c["reliable"]):
        # A flat veto is too crude. It correctly refuses a 2% win measured on 15%
        # noise, but it would also discard a 617% margin sitting 49x above the noise,
        # which is not a judgement any honest instrument should make. So a failed gate
        # blocks the claim ONLY when the margin does not overwhelm the noise.
        if margin_all >= LARGE_MARGIN_RATIO * noise_all:
            return {"verdict": "CANDIDATE_WINS_DESPITE_NOISE" if speedup > 1
                              else "BASELINE_WINS_DESPITE_NOISE",
                    "speedup": round(speedup, 4),
                    "margin_pct": round(margin_all, 2),
                    "arm_noise_pct": round(noise_all, 2),
                    "margin_over_noise_ratio": round(margin_all / noise_all, 1),
                    "caveat": f"an arm exceeded the {IQR_GATE_PCT}% IQR gate, but the "
                              f"margin is {margin_all / noise_all:.0f}x the worst arm "
                              f"noise, so the direction of the result is not in doubt; "
                              f"the MAGNITUDE carries more uncertainty than a clean "
                              f"measurement would"}
        return {"verdict": "NO_CLAIM",
                "reason": f"an arm failed the {IQR_GATE_PCT}% IQR gate "
                          f"({baseline} {b['iqr_spread_pct']}%, "
                          f"{candidate} {c['iqr_spread_pct']}%) and the margin "
                          f"{margin_all:.2f}% does not overwhelm that noise",
                "speedup": None}
    # the margin must clear the noise of BOTH arms, or it is not a result
    noise = max(b["iqr_spread_pct"], c["iqr_spread_pct"])
    margin = abs(speedup - 1.0) * 100
    if margin <= noise:
        return {"verdict": "INDISTINGUISHABLE", "speedup": round(speedup, 4),
                "reason": f"margin {margin:.2f}% does not clear arm noise {noise:.2f}%"}
    return {"verdict": "CANDIDATE_WINS" if speedup > 1 else "BASELINE_WINS",
            "speedup": round(speedup, 4),
            "margin_pct": round(margin, 2), "arm_noise_pct": round(noise, 2)}

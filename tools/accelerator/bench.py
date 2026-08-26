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


# Fewer than this many samples at or above p95 and the figure is an order
# statistic rather than a percentile estimate. 40 reps gives 3; 200 gives 11.
TAIL_SAMPLES_FOR_A_STABLE_P95 = 5


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
    # S032 §16: latency is first class, and a median alone hides the tail a caller
    # actually waits on. p95 is reported with the SAMPLE COUNT BEHIND IT, because
    # at the default 40 reps only THREE samples sit at or above p95 -- order
    # statistics, not an estimate of the 95th percentile. A tail figure whose
    # resolution is invisible is how a three-sample tail gets quoted as a
    # distribution.
    def _pct(p: float) -> float:
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]
    p95, p99 = _pct(0.95), _pct(0.99)
    tail_samples = max(1, len(s) - int(round(0.95 * (len(s) - 1))))
    # AN ARM FASTER THAN THE CLOCK IS NOT AN ARM WITH ZERO SPREAD. perf_counter can
    # return the same value twice for a cheap enough callable, and dividing by that
    # sample raised ZeroDivisionError in roughly one run of four -- an intermittent
    # crash where the honest answer is that the arm was never measured. Reported as
    # UNMEASURABLE with infinite spread so it can never pass the reliability gate,
    # because a silent 0.00% would read as the most reliable arm ever timed.
    below_resolution = q1 <= 0.0 or s[0] <= 0.0
    iqr = float("inf") if below_resolution else (q3 - q1) / q1 * 100
    rng = float("inf") if below_resolution else (s[-1] - s[0]) / s[0] * 100
    stable = reps >= STABLE_RELIABILITY_MIN_REPS
    return {"median_s": med, "q1_s": q1, "q3_s": q3,
            "p50_s": med, "p95_s": p95, "p99_s": p99,
            "p95_over_p50": None if med <= 0 else round(p95 / med, 4),
            "samples_at_or_above_p95": tail_samples,
            "tail_resolution": (
                "p95 here is one sample of %d; it is an ORDER STATISTIC, not an "
                "estimate of the 95th percentile" % len(s))
                if tail_samples < TAIL_SAMPLES_FOR_A_STABLE_P95 else None,
            "below_timer_resolution": below_resolution,
            "iqr_spread_pct": round(iqr, 2),
            "full_range_pct": round(rng, 2),
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


# --- QUIESCENCE -------------------------------------------------------------
#
# The steer asks for UNCONTENDED measurement and until now this program checked
# for it with an ad-hoc `pgrep modellake` in a shell command, whose silence was
# then written into a receipt as "the machine is quiet". That is the wrong shape
# of check twice over: it was never a recorded FIELD, so no receipt could be
# audited for it; and it MATCHES NAMES, so it can only ever find the one
# contender this program had already met.
#
# It failed exactly that way. TOKEN_GRAPH_REDUCTION_TIMED recorded "no lake fill
# running (pgrep modellake = 0)" while fileproviderd and mds_stores had been
# pegged near 100% for five hours and an unrelated 31 GiB MLX model boot later
# joined them. A quiescence check that names one process family and is read as
# machine-wide reports WHAT IT LOOKED FOR, NOT WHAT IS THERE.
#
# So this ENUMERATES. There is no name list and there is deliberately no way to
# pass one: the whole defect was a filter, and an instrument whose blind spot is
# configurable is an instrument someone will configure blind.

# RSS matters more than CPU for the claims this program makes. The 2026-08-22 law
# is that METAL WORKING SET is the admission gate on this machine, not free RAM,
# so a large-RSS neighbour competes with a resident model in the one dimension
# that decides. CPU is reported too because it is real, not because it is equal.
QUIET_CPU_PCT = 20.0
QUIET_RSS_GIB = 2.0


def machine_quiescence(*, cpu_pct: float = QUIET_CPU_PCT,
                       rss_gib: float = QUIET_RSS_GIB) -> dict[str, Any]:
    """Every process over either threshold, found by ENUMERATION not by name.

    Returns quiet=False with the contenders named. The caller's own process is
    reported like any other rather than excluded: a benchmark harness that is
    itself burning CPU is a fact about the measurement, and hiding it would be
    the same self-exemption the name filter made by accident.
    """
    import subprocess
    r = subprocess.run(["ps", "-Ao", "pid,pcpu,rss,comm"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        # An enumeration that FAILED must never read as an enumeration that found
        # nothing -- that is the 0-of-0-cases shape this program has sealed four
        # times. quiet is None, not True.
        return {"quiet": None, "method": "enumerate",
                "refused": f"ps exited {r.returncode}", "contenders": []}
    out = []
    for line in r.stdout.splitlines()[1:]:
        f = line.split(None, 3)
        if len(f) < 4:
            continue
        try:
            pid, pc, rss = int(f[0]), float(f[1]), int(f[2])
        except ValueError:
            continue
        gib = rss / (1024 * 1024)
        if pc >= cpu_pct or gib >= rss_gib:
            out.append({"pid": pid, "cpu_pct": pc, "rss_gib": round(gib, 2),
                        "comm": f[3].strip()})
    out.sort(key=lambda d: (-d["rss_gib"], -d["cpu_pct"]))
    return {
        "quiet": not out,
        "method": "enumerate",
        "no_name_filter": True,
        "thresholds": {"cpu_pct": cpu_pct, "rss_gib": rss_gib},
        "contenders": out,
        "n_contenders": len(out),
        "total_cpu_pct": round(sum(d["cpu_pct"] for d in out), 1),
        "max_rss_gib": max((d["rss_gib"] for d in out), default=0.0),
        "rss_is_the_gate_not_cpu": (
            "Metal working set is this machine's admission gate (2026-08-22), so a "
            "large-RSS neighbour contends with a resident model in the dimension "
            "that decides; CPU is reported because it is real, not because it is "
            "equally predictive."),
    }


def name_filter_quiescence(names: tuple[str, ...]) -> dict[str, Any]:
    """The BROKEN check, kept executable as the control for the one above.

    This is what `pgrep modellake` did. It is retained ONLY so a test can watch
    it report quiet while the enumerating check names real contenders -- a
    refutation nobody has watched is indistinguishable from a rule nobody needed.
    Never call this to decide whether a measurement is admissible.
    """
    import subprocess
    hits = []
    for n in names:
        r = subprocess.run(["pgrep", "-f", n], capture_output=True, text=True)
        hits += [int(x) for x in r.stdout.split()]
    return {"quiet": not hits, "method": "match_names", "names": list(names),
            "hits": hits,
            "why_this_is_kept": "executable demonstration that a name filter "
                                "reports what it looked for, not what is there"}


def bench_block(*, machine: str, note: str | None = None,
                before: dict[str, Any] | None = None,
                after: dict[str, Any] | None = None) -> dict[str, Any]:
    """The S032 §3 machine-state block a performance receipt must carry.

    The state is DERIVED from the samples, never asserted alongside them. Pass
    the quiescence samples taken around the measurement; if none were taken the
    state is UNKNOWN, which is the steer's rule verbatim: "If quiescence is
    unknown: BENCH_STATE = UNKNOWN, not quiet."

    A sample whose enumeration FAILED (quiet is None) is also UNKNOWN, not quiet
    -- a `ps` that exited non-zero found nothing because it could not look.
    """
    import time as _t
    samples = [s for s in (before, after) if isinstance(s, dict)]
    if not samples:
        state, worst = "UNKNOWN", None
    elif any(s.get("quiet") is None for s in samples):
        state, worst = "UNKNOWN", max(samples, key=lambda s: s.get("n_contenders") or 0)
    elif all(s.get("quiet") is True for s in samples):
        # BOTH ENDS OR IT IS NOT QUIESCED. A single quiet sample says the machine
        # was quiet at one instant, not across the window -- contention that
        # started after a quiet `before` and ended before there was an `after` to
        # see it leaves no trace. Evidence of noise is evidence; absence of
        # observed noise on ONE side is not evidence of quiet, and this is the
        # same asymmetry the enumeration failure above already respects.
        state, worst = (("QUIESCED", samples[0]) if before is not None and after is not None
                        else ("UNKNOWN", samples[0]))
    else:
        state, worst = "CONTENDED", max(samples, key=lambda s: s.get("max_rss_gib") or 0.0)
    return {
        "state": state,
        "recorded_at": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime()),
        "machine": machine,
        "quiescence": worst,
        "samples": {"before": before, "after": after},
        "note": note,
        "rule": "S032 §3 -- if quiescence is unknown the state is UNKNOWN, not quiet",
    }

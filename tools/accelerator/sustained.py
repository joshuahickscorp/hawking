"""Sustained qualification. FRONT G (G049, steer S015 §29).

A microbenchmark measures a machine that has just been asked to do one thing. A
production profile runs for hours. §29 requires those to be distinguished, and this
is what makes the difference measurable: hold the champion kernel under continuous
load and watch whether it holds its rate.

The pass criterion is deliberately about DEGRADATION and VARIABILITY, not peak. A
kernel that starts fast and decays has not qualified for a sustained profile no
matter how good its first sample looked.
"""
from __future__ import annotations

import time
from typing import Any, Callable

MAX_DEGRADATION_PCT = 5.0      # last window vs first window
MAX_WINDOW_SPREAD_PCT = 15.0   # across all windows


def run(work: Callable[[], Any], *, seconds: float, window_s: float = 30.0,
        warmup_s: float = 10.0) -> dict[str, Any]:
    t_end = time.perf_counter() + warmup_s
    while time.perf_counter() < t_end:
        work()

    windows: list[dict[str, Any]] = []
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        w0 = time.perf_counter()
        n = 0
        while time.perf_counter() - w0 < window_s:
            work()
            n += 1
        dur = time.perf_counter() - w0
        windows.append({"index": len(windows), "iterations": n,
                        "seconds": round(dur, 3),
                        "rate_hz": round(n / dur, 3),
                        "elapsed_s": round(time.perf_counter() - start, 1)})

    rates = [w["rate_hz"] for w in windows]
    first, last = rates[0], rates[-1]
    degradation = (first - last) / first * 100
    spread = (max(rates) - min(rates)) / min(rates) * 100
    passed = degradation <= MAX_DEGRADATION_PCT and spread <= MAX_WINDOW_SPREAD_PCT
    return {
        "windows": windows, "window_count": len(windows),
        "total_seconds": round(sum(w["seconds"] for w in windows), 1),
        "first_window_hz": first, "last_window_hz": last,
        "best_hz": max(rates), "worst_hz": min(rates),
        "degradation_pct": round(degradation, 2),
        "window_spread_pct": round(spread, 2),
        "gate": {"max_degradation_pct": MAX_DEGRADATION_PCT,
                 "max_window_spread_pct": MAX_WINDOW_SPREAD_PCT},
        "passed": passed,
        "verdict": ("SUSTAINED" if passed else
                    "NOT SUSTAINED — the rate did not hold under continuous load"),
    }

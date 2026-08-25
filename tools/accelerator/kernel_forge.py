"""Kernel Forge — generation, specialization, autotuning. FRONT E (G047, S015).

Per primitive the steer wants four things kept distinct: a ReferenceKernel (what is
correct), an AppleBaselineKernel (the strongest thing that already exists), an
AcceleratorChampion (what currently wins here), and ExperimentalCandidates. They are
separate fields so a champion can never be mistaken for a reference, and so a
champion that loses to the baseline is recorded as losing rather than quietly
promoted.

Search is by priors, not brute force. The prior is that threadgroup size wants to be
a small multiple of the 32-wide SIMD group and that elements-per-thread trades
occupancy against per-thread work; the grid is generated from those two knobs rather
than from an exhaustive sweep.

EVERY CANDIDATE IS VERIFIED BEFORE IT IS TIMED. An incorrect candidate is rejected
outright and never enters the ranking, because the fastest way to run a kernel is to
compute the wrong answer.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "receipts/headless/KERNEL_FORGE_CACHE.json"

# Priors, not a sweep. 32 is the SIMD width; below one full group the machine idles,
# and past 1024 Metal refuses. Elements-per-thread beyond 8 stops paying on a
# memory-bound chain.
RELIABILITY_GATE_PCT = 10.0

# The prior stops at 64 and that exclusion is now MEASURED rather than assumed.
# ELEMENTS PER THREADGROUP (threadgroup x elems_per_thread) is the controlling
# variable: 32 -> 0.956 ms, 64 -> 0.595, 128 -> 0.512, >=256 -> 0.511-0.522, and
# tg32_ept2 (per_tg 64) lands at 0.5954 against tg64_ept1's 0.5955 -- two different
# configurations with the SAME per_tg agreeing to 0.02%. So the cliff is an occupancy
# threshold below ~128 elements per threadgroup, and the prior correctly refuses to
# spend reps there. See ACCELERATOR_PERF_MODEL.json.
THREADGROUP_PRIOR = (64, 128, 256, 512, 1024)
ELEMS_PER_THREAD_PRIOR = (1, 2, 4)


@dataclass
class Candidate:
    threadgroup: int
    elems_per_thread: int
    source: str
    correct: bool | None = None
    max_abs_err: float | None = None
    timing: dict[str, Any] | None = None
    rejected_because: str | None = None

    @property
    def name(self) -> str:
        return f"tg{self.threadgroup}_ept{self.elems_per_thread}"


@dataclass
class Primitive:
    """The four roles the steer names, kept in separate fields on purpose."""
    name: str
    reference_kernel: str
    apple_baseline_kernel: str
    champion: Candidate | None = None
    candidates: list[Candidate] = field(default_factory=list)
    baseline_timing: dict[str, Any] | None = None

    def champion_beats_baseline(self) -> bool | None:
        if not (self.champion and self.champion.timing and self.baseline_timing):
            return None
        return self.champion.timing["median_s"] < self.baseline_timing["median_s"]


def fused_chain_source(threadgroup: int, elems_per_thread: int, n: int) -> str:
    """mul -> relu -> silu, one pass, with a specialization on how much each thread
    does. The body is generated, not hand-written per variant."""
    body = [f"uint base = thread_position_in_grid.x * {elems_per_thread}u;"]
    for k in range(elems_per_thread):
        body += [
            f"if (base + {k}u < {n}u) {{",
            f"    float v = a[base + {k}u] * b[base + {k}u];",
            f"    v = max(v, 0.0f);",
            f"    out[base + {k}u] = v / (1.0f + exp(-v));",
            f"}}",
        ]
    return "\n    ".join(body)


def generate(n: int) -> list[Candidate]:
    return [Candidate(tg, ept, fused_chain_source(tg, ept, n))
            for tg in THREADGROUP_PRIOR for ept in ELEMS_PER_THREAD_PRIOR]


def forge(primitive: Primitive, n: int, *, verify: Callable[[Candidate], tuple[bool, float]],
          time_it: Callable[[Candidate], dict[str, Any]],
          tolerance: float = 1e-5) -> Primitive:
    for c in generate(n):
        try:
            ok, err = verify(c)
        except Exception as e:                     # a candidate that will not compile
            c.correct = False
            c.rejected_because = f"{type(e).__name__}: {str(e)[:120]}"
            primitive.candidates.append(c)
            continue
        c.correct, c.max_abs_err = ok, err
        if not ok:
            c.rejected_because = f"max_abs_err {err:.3e} exceeds {tolerance:.0e}"
            primitive.candidates.append(c)
            continue
        c.timing = time_it(c)
        if not c.timing.get("reliable"):
            c.rejected_because = (f"IQR spread {c.timing['iqr_spread_pct']}% failed the "
                                  f"reliability gate; an unreliable arm cannot be champion")
        primitive.candidates.append(c)

    eligible = [c for c in primitive.candidates
                if c.correct and c.timing and c.timing.get("reliable")]
    primitive.champion = min(eligible, key=lambda c: c.timing["median_s"]) if eligible else None
    return primitive


def successive_halving(candidates: list[Any], *, measure, rounds: int = 3,
                       keep: float = 0.5, base_reps: int = 5) -> dict[str, Any]:
    """Bandit search: spend few reps on many candidates, then more on the survivors.

    The forge's first search timed all fifteen variants equally, which is the brute
    force the obligation says not to do. Successive halving gives every candidate a
    cheap look, keeps the better half, and doubles the reps on those -- so measurement
    budget follows evidence instead of being spread flat.

    `measure(candidate, reps)` returns (median_time, iqr_pct). The caller owns
    correctness: an incorrect candidate must never reach here, because a bandit
    optimises whatever it is given and would happily converge on a fast wrong answer.

    THE FINAL ROUND APPLIES THE RELIABILITY GATE. The first version of this ranked on
    median alone and crowned a candidate that had failed the forge's own 10% IQR gate
    -- an unrepeatable kernel that happened to look fast. A bandit inherits every
    blind spot of its objective, so the objective has to carry the gate.
    """
    alive = list(candidates)
    reps = base_reps
    spent = 0
    history: list[dict[str, Any]] = []
    while len(alive) > 1 and rounds > 0:
        timed = []
        for c in alive:
            med, iqr = measure(c, reps)
            timed.append((med, iqr, c))
            spent += reps
        # An UNREPEATABLE candidate must never displace a repeatable one during
        # elimination. Gating only at the end let a fast jittery candidate knock out
        # the steady ones in round one and then get rejected itself, leaving NO
        # champion at all -- the gate fired correctly and the answer was still wrong.
        timed.sort(key=lambda p: (p[1] > RELIABILITY_GATE_PCT, p[0]))
        n_keep = max(1, int(len(timed) * keep))
        history.append({"round": len(history) + 1, "reps_each": reps,
                        "candidates": len(timed), "kept": n_keep,
                        "best_time_s": timed[0][0],
                        "eliminated": [str(c) for _, _, c in timed[n_keep:]]})
        alive = [c for _, _, c in timed[:n_keep]]
        reps *= 2
        rounds -= 1
    # Final confirmation at full budget, WITH the reliability gate. A survivor that
    # cannot repeat itself is barred from winning however fast its median looked.
    confirmed = []
    rejected = []
    for c in alive:
        med, iqr = measure(c, reps)
        spent += reps
        (confirmed if iqr <= RELIABILITY_GATE_PCT else rejected).append((med, iqr, c))
    confirmed.sort(key=lambda p: p[0])
    return {"champion": confirmed[0][2] if confirmed else None,
            "champion_time_s": confirmed[0][0] if confirmed else None,
            "champion_iqr_pct": confirmed[0][1] if confirmed else None,
            "rejected_for_unreliability": [
                {"candidate": str(c), "median_s": m, "iqr_pct": i} for m, i, c in rejected],
            "total_reps_spent": spent, "rounds": history,
            "survivors": [str(c) for _, _, c in confirmed]}


def exhaustive_cost(n_candidates: int, reps: int) -> int:
    """What flat measurement of the same space would have cost."""
    return n_candidates * reps


def cache_key(machine: dict[str, Any], primitive: str, representation: str,
              shape: tuple[int, ...], runtime: dict[str, Any]) -> str:
    """Identity-based invalidation (steer §27): retune only when something material
    changes, never on a timer."""
    return json.dumps({
        "soc": machine.get("soc"), "gpu_cores": machine.get("gpu_cores"),
        "os": machine.get("os_product"), "primitive": primitive,
        "representation": representation, "shape": list(shape),
        "mlx": runtime.get("mlx"),
    }, sort_keys=True)


def load_cache() -> dict[str, Any]:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            return {}
    return {}


def save_champion(key: str, payload: dict[str, Any]) -> None:
    c = load_cache()
    c[key] = payload | {"cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c, indent=1))

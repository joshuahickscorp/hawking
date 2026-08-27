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
import re
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

# The prior stops at 64 and that exclusion is MEASURED -- ON ONE PRIMITIVE, WHICH IS
# THE PART THAT MATTERS. ELEMENTS PER THREADGROUP (threadgroup x elems_per_thread) is
# the controlling variable, and that now rests on a real collapse rather than one
# pair: at per_tg 256 the four splits (32,8) (64,4) (128,2) (256,1) land within 1.9%
# while their THREAD COUNTS differ 8x, so threadgroup count is what tracks and thread
# count is not. See ACCELERATOR_PERF_MODEL.json and ACCELERATOR_CLIFF_TRANSFER.json.
#
# BUT THE CLIFF'S LOCATION IS PRIMITIVE-SPECIFIC AND THIS PRIOR IS CALIBRATED ON THE
# FUSED CHAIN. Measured at 2^24 f32, times relative to each primitive's own floor:
#   per_tg          32     64    128    256    512
#   write-only    2.94x  1.90x  1.37x  1.11x  1.02x   <- still 1.9x AT THE PRIOR'S FLOOR
#   fused chain   1.87x  1.17x  1.00x  1.00x  1.00x   <- what this prior was fit on
#   heavy (FMA)   1.15x  1.05x  1.05x  1.05x  1.00x   <- barely a cliff at all
# A cheaper primitive stays dispatch-bound to a LARGER per_tg, so tg=64 is a safe
# floor for the chain and lands inside the bad region for a write-only kernel; an
# expensive one is barely affected and the exclusion merely costs it a candidate.
# TREAT THIS TUPLE AS A CHAIN-SHAPED PRIOR, NOT A PROPERTY OF THE MACHINE.
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

    @property
    def per_tg(self) -> int:
        """Elements per threadgroup -- the variable that actually tracks time. Two
        candidates sharing it are the same physical point to within a couple of
        percent, whatever their threadgroup sizes."""
        return self.threadgroup * self.elems_per_thread


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


def collapse_groups(candidates: list[Candidate]) -> dict[int, list[str]]:
    """Group candidates by per_tg, because candidates sharing it are NOT independent
    samples of the landscape.

    The default grid is 15 candidates over 7 distinct per_tg values, so roughly half
    the search re-measures a point it already has. This does not dedupe them -- the
    collapse is close but not exact, and the exception is measured: on the READ-heavy
    chain an ept of 8 costs ~7.9% against the same per_tg reached with a bigger
    threadgroup, while the write-only kernel shows no such penalty. Reporting the
    redundancy beats silently discarding candidates that can differ.
    """
    out: dict[int, list[str]] = {}
    for c in candidates:
        out.setdefault(c.per_tg, []).append(c.name)
    return out


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


AWKWARD_SHAPES = (1, 2, 3, 7, 13, 17, 31, 33, 63, 65, 127, 129, 257)


def shape_sweep(cases, check, *, name: str = "sweep") -> dict[str, Any]:
    """Vary the SHAPE, because a single-shape check never varies the LAUNCH.

    The forge has verified every candidate before timing it since the first block, on
    the principle that the fastest way to run a kernel is to compute the wrong answer.
    That verification calls the op's OWN execute(), so it reuses the very launch it is
    meant to be testing -- which is how the tiled matmul stayed wrong at every shape
    where m was not a multiple of the tile, for many blocks, with every check passing.

    `check(case)` returns a relative error, or raises. A case that raises is recorded
    as a FAILURE, not skipped, because a shape the kernel refuses is a shape the caller
    needs told about.

    A SWEEP THAT RAN NOTHING IS NOT A PASSING SWEEP. The first version of this
    program's own shape fuzz reported ZERO FAILURES for four primitives it never
    executed -- one filter demanded every dimension be a multiple of 8 while the shape
    grid held exactly one such value -- so cases_run is returned and zero RAISES.
    See ACCELERATOR_SHAPE_FUZZ.json.
    """
    results, failures = [], []
    for case in cases:
        try:
            err = float(check(case))
            bad = not (err == err and err < 1e-4)      # NaN-safe
        except Exception as e:                          # noqa: BLE001
            err, bad = float("inf"), True
            failures.append({"case": list(case), "raised": f"{type(e).__name__}: {e}"[:120]})
        else:
            if bad:
                failures.append({"case": list(case), "rel_err": err})
        results.append({"case": list(case), "rel_err": err, "ok": not bad})
    if not results:
        raise ValueError(
            f"shape_sweep {name!r} ran ZERO cases. A sweep that executed nothing "
            f"cannot report zero failures -- check the shape filter, not the kernel.")
    return {"name": name, "cases_run": len(results), "failures": len(failures),
            "worst_rel_err": max(r["rel_err"] for r in results),
            "failing": failures[:20], "all_ok": not failures}


WIDTH_PRIOR = (32, 64, 128, 256, 512, 1024)


# MEASURED, and it is the ONLY factor that moved a barrier control's loudness.
# A 2x2x2x2 factorial at threadgroup 64 varied what the racing read FINDS
# (uninitialised garbage vs a prior valid value), whether an UPSTREAM barrier
# precedes the stripped one, whether the racing write's operand is hoisted into a
# register or reloaded from global memory, and the input values. Marginal mean
# p_fired: upstream ABSENT 1.000 vs PRESENT 0.084, against slot contents
# 0.574/0.510, load 0.505/0.579 and values 0.534/0.551. See
# ACCELERATOR_BARRIER_CONTROL_MECHANISM.json.
# CORRECTED ONE BLOCK LATER, and the correction is an ASYMMETRY. The LOUD case is
# a reliable prediction: with no upstream barrier every cell ever measured fires
# 1.00 -- all eight factorial cells, and every width and work count in the window
# sweep. The QUIET case IS NOT A NUMBER. The same kernel at width 256 measured
# p = 0.05 in one process and 0.65 in another; the factorial's 0.084 marginal was a
# sample from that distribution, and quoting it as an expectation was wrong.
# See ACCELERATOR_BARRIER_WINDOW.json.
# BOTH ENTRIES ARE None, AND THE LOUD ONE ONLY BECAME None AFTER IT WAS REFUTED.
# ACCELERATOR_BARRIER_WINDOW retracted the QUIET number and kept the loud one as
# "a reliable prediction, since with no upstream barrier EVERY CELL EVER MEASURED
# fires 1.00". ACCELERATOR_THE_REDUCTION_TAIL_IS_FREE measured a REAL SHIPPED
# kernel with no upstream barrier at p = 0.067 over 60 runs (intact wrong 0 of 60)
# and at 8 of 8 in another process on the same code. The asymmetry does not hold:
# the loud case is as unstable as the quiet one, and a number here gets cited.
CONTROL_LOUDNESS_PRIOR = {"no_upstream_barrier": None, "upstream_barrier": None}
QUIET_OBSERVED_RANGE = (0.01, 0.90)
LOUD_OBSERVED_RANGE = (0.067, 1.000)
_BARRIER_TOKENS = ("threadgroup_barrier", "simdgroup_barrier")


def barrier_control_prior(source: str, strip_index: int = 0) -> dict[str, Any]:
    """How much evidence a stripped-barrier control can carry, from the SOURCE.

    A control that strips a barrier is only as informative as the race it exposes,
    and the race window is set by HOW SYNCHRONIZED THE THREADGROUP ALREADY IS when
    it reaches the conflicting access. An upstream barrier resynchronizes it, so a
    barrier stripped just after another barrier exposes a window narrow enough that
    the defect mostly does not fire -- 0.084 against 1.000 with no upstream barrier.

    That is a PRIOR ON THE CONTROL, never a claim about the kernel: a quiet control
    means the sweep proved little, NOT that the barrier is unnecessary.

    It predicts the two shipped controls from structure. AirNorm's barrier publishes
    a row sum written straight after an unsynchronized per-thread scan, so nothing is
    upstream of it -- measured LOUD, 40 of 48. AirTopKSample's sits inside a tree
    reduce where the previous iteration's barrier is one compare-and-swap upstream --
    measured QUIET, 4 of 48. ACCELERATOR_QUIET_CONTROL.json reported that 10x gap
    with no mechanism; this is the mechanism.
    """
    hits = [m.start() for m in re.finditer("|".join(_BARRIER_TOKENS), source)]
    if not 0 <= strip_index < len(hits):
        raise ValueError(
            f"barrier_control_prior: strip_index {strip_index} is outside the "
            f"{len(hits)} barrier(s) in this source. Naming a barrier that is not "
            f"there would silently return the prior for a different one.")
    at = hits[strip_index]
    upstream = strip_index > 0
    since = source[hits[strip_index - 1]:at] if upstream else source[:at]
    work = len([ln for ln in since.splitlines()[1:] if ln.strip()
                and not ln.strip().startswith("//")])
    return {
        "barriers_in_source": len(hits), "stripped_at": strip_index,
        "has_upstream_barrier": upstream, "statements_since_upstream": work,
        # None IN BOTH CASES ON PURPOSE: there is no expectation to give, and a
        # number here would be cited as one. The loud case carried 1.000 until a
        # real shipped kernel measured 0.067.
        "expected_p_fired": None,
        "expected_p_fired_range": QUIET_OBSERVED_RANGE if upstream else LOUD_OBSERVED_RANGE,
        "verdict": "QUIET_CONTROL_WEAK_EVIDENCE" if upstream else "LOUD_CONTROL",
        # Measured: work between the two barriers widens the race window and drives
        # p back to 1.00 -- at width 1024 that takes ~16 fused multiply-adds, at 256
        # ~256, at 64 ~1024. So statements_since_upstream really is the knob, and it
        # is reported rather than folded into a single verdict because the amount of
        # work that makes a quiet control loud DEPENDS ON THE WIDTH.
        "quiet_only_while_the_gap_is_small": upstream,
        # THE MECHANISM THIS FUNCTION CANNOT SEE, and the one that refuted the loud
        # prior: a window is also closed when THE READING THREAD IS THE SLOWEST ONE.
        # In gravity_native's serial reduction tail the lanes divide GROUPS round
        # robin, so lane 0 -- the lane that reads every slot -- is in the group doing
        # the MOST work and arrives at its read after the lanes it reads from have
        # long since written. That is a property of the WORK DISTRIBUTION ACROSS
        # LANES, not of the barrier structure, and this function reads only the
        # source's barriers. It is reported rather than guessed at.
        "blind_to_lane_work_imbalance": True,
        "basis": "ACCELERATOR_BARRIER_WINDOW.json (work sweep + placement control); "
                 "ACCELERATOR_BARRIER_CONTROL_MECHANISM.json (factorial, tg=64); "
                 "ACCELERATOR_THE_REDUCTION_TAIL_IS_FREE.json (loud prior refuted "
                 "on a shipped kernel at p=0.067 of 60)",
    }


def swept_with_control(cases, check, broken_check, *, name: str = "sweep",
                       repeat: int = 1, control_priors: dict[str, Any] | None = None
                       ) -> dict[str, Any]:
    """A sweep that has never been watched fail cannot report zero failures.

    Runs the real check over `cases` AND a deliberately broken variant over the same
    cases. If the broken arm passes everywhere, the RESULT IS REFUSED -- not because
    the kernel is wrong but because the sweep proved nothing about it.

    `repeat` exists because a WIDTH-dependent defect is usually a RACE and therefore
    PROBABILISTIC, unlike a shape-dependent one which is deterministic. The top-k
    barrier fired in 16 of 30 runs at threadgroup 1024 and 0 of 30 at 256; a single
    run of that check is a coin flip. Detection power is set by `repeat`, not by the
    length of the case list: a race with per-run probability p is missed with
    (1-p)**repeat, so repeat=8 is blind to anything under ~13% per run.

    MEASURED, AND THE CONTROL HAS ITS OWN BLIND WIDTH: stripping one barrier from
    AirNorm is caught 8 of 8 runs at threadgroups 64 through 1024 and is EXACT at 32,
    where the whole threadgroup is ONE SIMDGROUP and lockstep makes the fence
    unnecessary. A control run only at 32 would have reported the broken kernel fine.
    See ACCELERATOR_WIDTH_SWEEP.json.
    """
    def run(fn, label, raises_are_failures=True):
        """A RAISE MEANS DIFFERENT THINGS IN THE TWO ARMS, and conflating them made a
        control report that it detected the defect at every width when it had in fact
        crashed at every width -- a TypeError in the probe's own call, counted as
        `inf` and therefore as a detection. Caught because that control's blind list
        disagreed with a shipped receipt, not by re-reading the loop.

        For the REAL arm a raise IS a failure: a shape the kernel refuses is a shape
        the caller needs told about (ACCELERATOR_SHAPE_FUZZ.json). For a CONTROL arm
        a raise means the control NEVER EXECUTED, so it proves nothing about the
        kernel and the sweep is refused rather than crediting it with a detection.
        """
        out = []
        for case in cases:
            errs = []
            for _ in range(repeat):
                try:
                    errs.append(float(fn(case)))
                except Exception as e:                       # noqa: BLE001
                    if not raises_are_failures:
                        raise ValueError(
                            f"sweep {name!r} REFUSED: control {label!r} RAISED at case "
                            f"{list(case)} ({type(e).__name__}: {e}). A control that "
                            f"crashes never ran the kernel, so counting it as a "
                            f"detection would credit the sweep with power it has not "
                            f"got.") from e
                    errs.append(float("inf"))
                    del e
            worst = max(errs)
            out.append({"case": list(case), "worst": worst, "repeat": repeat,
                        "wrong": sum(1 for x in errs if not (x < 1e-4)),
                        "ok": bool(worst == worst and worst < 1e-4)})
        if not out:
            raise ValueError(f"{label} {name!r} ran ZERO cases")
        return out

    real = run(check, "sweep")
    controls = broken_check if isinstance(broken_check, (list, tuple)) else \
        [("control", broken_check)]
    controls = [c if isinstance(c, (list, tuple)) else (f"control{i}", c)
                for i, c in enumerate(controls)]

    seen: list[dict[str, Any]] = []
    for label, fn in controls:
        ctl = run(fn, label, raises_are_failures=False)
        det = [c["case"] for c in ctl if not c["ok"]]
        if not det:
            raise ValueError(
                f"sweep {name!r} REFUSED: control {label!r} passed at every case, so a "
                f"clean result here would be evidence of nothing. Either the control "
                f"does not break what the check looks at, or the cases do not reach it.")
        seen.append({"label": label, "detected_at": det,
                     "blind_at": [c["case"] for c in ctl if c["ok"]],
                     "fired_runs": sum(c["wrong"] for c in ctl),
                     "total_runs": len(ctl) * repeat,
                     # A control blind for a KNOWN structural reason is interpretable;
                     # one blind for an unknown reason is not, and the two read the
                     # same in a blind list. See barrier_control_prior.
                     "prior": (control_priors or {}).get(label)})

    reached = {tuple(c) for s_ in seen for c in s_["detected_at"]}
    blind_everywhere = [list(c) for c in cases if tuple(c) not in reached]
    return {"name": name, "cases_run": len(real), "repeat": repeat,
            "executions": len(real) * repeat * (1 + len(controls)),
            "failures": sum(1 for r in real if not r["ok"]),
            "all_ok": all(r["ok"] for r in real),
            "controls": seen,
            # BACK-COMPAT, AND A TRAP IF READ ALONE: these describe the FIRST control
            # only. ONE control's blind list is not the SWEEP's blind list -- measured
            # in ACCELERATOR_QUIET_CONTROL.json, where a LOUD control is blind at 1 of
            # 6 widths and a QUIET one at 4 of 6 ON THE SAME MACHINE.
            "control_detected_at": seen[0]["detected_at"],
            "control_blind_at": seen[0]["blind_at"],
            "blind_at_every_control": blind_everywhere,
            "detection_floor_per_run": round(1 - 0.5 ** (1 / repeat), 4),
            # THE FLOOR BOUNDS `repeat`, NOT THE SWEEP. It is calibrated -- 40 blocks
            # of 8 on a real race with p = 0.1187 detected in 25, against a binomial
            # 0.6363, and the per-block hit distribution fits Binomial(8, p) at
            # chi2 = 0.150 on 2 dof. What it does NOT cover is a case where the defect
            # does not fire AT ALL: no repeat count rescues p = 0, and that is exactly
            # where the quiet control was blind.
            "floor_bounds_repeat_not_coverage": True}

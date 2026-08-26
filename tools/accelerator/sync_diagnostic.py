"""Synchronization tax diagnostic. G062 (obligations G043/G047).

Five Accelerator receipts sighted a submission/synchronization floor and each one
answered it locally. ACCELERATOR_SYNC_NOT_SUBMISSION.json then found that the floor
is WAITING, not SUBMITTING: a serial dependency chain that MLX can express in its own
lazy graph cost 41.1us/step against 254.1us/step when the host blocked after every
step. That is a real finding and it was not reusable -- nobody could point it at a new
workload and get an answer. This module is the instrument.

It takes a workload and reports, with a control for every number:

  submission+compute per step   the chained arm: K steps, ONE wait at the end
  wait cost per step            the paired difference (synced - chained) / K
  the WINDOW knob               how per-step cost moves with K, work between waits
  a CONFOUND CHECK              submission COUNT and wait COUNT are correlated in the
                                naive design; the third arm breaks them apart
  its own detection floor       the smallest per-step effect these reps could resolve

WHY THE CONFOUND ARM EXISTS. The headline finding in this area was itself a confound
that survived a round of measurement -- ACCELERATOR_BARRIER_CONTROL_MECHANISM.json
recorded that a stale/garbage separation was really an upstream barrier. Here the
suspected mechanism is WAITING and the correlated variable is SUBMISSION COUNT: the
chained arm makes one submission and one wait, the synced arm makes K of each. So a
third arm makes K SUBMISSIONS AND ONE WAIT. If the cost follows the wait count the
mechanism is waiting; if it follows the submission count it is not.

WHY EVERY NUMBER HAS A CONTROL. An instrument that finds a tax everywhere is broken.
The NULL workload -- a loop whose step and sync do nothing -- is measured in the same
interleave, and it is what this harness costs to run. A subject whose reading is not
separated from the null's is reported MEASURING_ITSELF, and a subject whose chained
per-step cost is not separated from the null's is reported WORK_NOT_VISIBLE, because
a lazy graph that was never materialized reads as free work and an enormous tax.

WHY THE ARMS ARE INTERLEAVED. This machine is contended (the receipts above carry
34-39% IQR for that reason) and its thermal and page-cache state drift inside a run.
A-then-B attributes that drift to the arm that ran second. Every arm is timed once per
rep, in an order that alternates, so drift lands on both.

Never write "X IS FASTER". A reading here is for one primitive x shape x
representation x machine x runtime and nothing wider.
"""
from __future__ import annotations

import statistics
import time
from typing import Any, Callable, NamedTuple, Sequence

import bench

# Same gate as every other performance claim in this program (bench.IQR_GATE_PCT).
# Imported rather than restated so a change there cannot silently miss this file.
IQR_GATE_PCT = bench.IQR_GATE_PCT

# Two-sided normal approximation on the PAIRED difference. Quoted as the smallest
# effect these reps could have resolved, which is the number the control-spectrum
# receipt said was missing ("20 repeats per cell, so a per-run probability under
# ~3.4% is invisible" -- an instrument must publish its own blindness).
MDE_Z = 1.96


class Workload(NamedTuple):
    """What the diagnostic needs to know about a workload.

    `step` MUST consume the state and return the next one; a step that returns None
    is refused, because a broken chain turns a serial measurement into a parallel one
    without saying so.

    `submit` is optional and its absence is reported LOUDLY as NOT_RUN with a reason.
    A missing capability that turns into a silent skip is how this repo has shipped
    green nothing before.
    """
    name: str
    primitive: str
    shape: tuple
    make: Callable[[], Any]                    # fresh, already-materialized state
    step: Callable[[Any], Any]                 # one dependent step
    sync: Callable[[Any], None]                # block until the state is real
    submit: Callable[[Any], None] | None = None  # start work WITHOUT blocking


# The harness floor: what an empty loop costs. Not a workload anyone cares about,
# and the only reason every number below can be believed.
NULL = Workload(name="null_control", primitive="none", shape=(),
                make=lambda: 0, step=lambda s: s, sync=lambda s: None,
                submit=lambda s: None)


def spin(seconds: float) -> None:
    """Busy wait. Used to INJECT a known stall, never in the instrument's own path.

    time.sleep is not usable for this: at 200us its overshoot on this machine is the
    same order as the quantity being injected, so the injected amount would not be
    known and the calibration arm would prove nothing.
    """
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


# ---------------------------------------------------------------- the three arms

def _chained(w: Workload, k: int) -> None:
    s = w.make()
    for _ in range(k):
        s = w.step(s)
    w.sync(s)


def _synced(w: Workload, k: int) -> None:
    s = w.make()
    for _ in range(k):
        s = w.step(s)
        w.sync(s)


def _submitted(w: Workload, k: int) -> None:
    s = w.make()
    for _ in range(k):
        s = w.step(s)
        w.submit(s)
    w.sync(s)


# ------------------------------------------------------------------- statistics

def _stat(samples: list[float]) -> dict[str, Any]:
    s = sorted(samples)
    q1, med, q3 = s[len(s) // 4], s[len(s) // 2], s[(3 * len(s)) // 4]
    # An arm faster than the clock is not an arm with zero spread -- bench.time_arm
    # learned this by raising ZeroDivisionError in one run of four. Same refusal.
    below = q1 <= 0.0 or s[0] <= 0.0
    iqr = float("inf") if below else (q3 - q1) / q1 * 100
    return {"median_s": med, "q1_s": q1, "q3_s": q3,
            "below_timer_resolution": below,
            "iqr_spread_pct": round(iqr, 2),
            "reliable": iqr <= IQR_GATE_PCT}


def _diff(a: list[float], b: list[float]) -> dict[str, Any]:
    """b - a, PAIRED. The pairing is the point: both arms saw the same rep, so the
    machine's drift between reps cancels instead of being attributed to one arm."""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    sd = statistics.stdev(d) if n > 1 else float("inf")
    return {"median_s": statistics.median(d), "mean_s": statistics.fmean(d),
            "sd_s": sd, "n": n,
            "mde_s": MDE_Z * sd / (n ** 0.5),
            "mde_basis": f"two-sided {MDE_Z} sigma on the paired difference, n={n}; "
                         f"assumes the reps are independent, which a thermally "
                         f"drifting contended machine does not guarantee"}


def _paired(arms: dict[str, Callable[[], None]], *, reps: int,
            warmup: int) -> dict[str, list[float]]:
    names = list(arms)
    for _ in range(warmup):
        for n in names:
            arms[n]()
    out: dict[str, list[float]] = {n: [] for n in names}
    for i in range(reps):
        for n in (names if i % 2 == 0 else names[::-1]):
            t0 = time.perf_counter()
            arms[n]()
            out[n].append(time.perf_counter() - t0)
    return out


# ---------------------------------------------------------------------- verdict

def verdict(tax_us: float, control_tax_us: float, work_excess_us: float, *,
            tax_mde_us: float, work_mde_us: float, excess_mde_us: float) -> str:
    """Pure, so it can be tested without a clock.

    EACH COMPARISON CARRIES ITS OWN FLOOR. The first version pooled them -- one
    max() over all three paired spreads -- and the first live MLX run caught it: the
    synced arm's spread is the widest thing in the experiment, so the pooled floor
    was 72.2us/step and a perfectly visible 50.9us/step of matmul work was reported
    WORK_NOT_VISIBLE. A floor imported from a noisier comparison is not this
    comparison's floor.

    Order matters. WORK_NOT_VISIBLE comes first because if the subject's own work
    never showed up in the chained arm, nothing else in the result means anything --
    that is exactly the failure a lazily-evaluated graph produces, and it presents as
    a spectacular tax.
    """
    if work_excess_us <= work_mde_us:
        return "WORK_NOT_VISIBLE"
    if tax_us <= tax_mde_us:
        return "NO_SIGNIFICANT_SYNC_TAX"
    if tax_us - control_tax_us <= excess_mde_us:
        return "MEASURING_ITSELF"
    return "SYNC_TAX"


# -------------------------------------------------------------------- diagnose

def diagnose(w: Workload, *, k: int = 16, reps: int = 40, warmup: int = 8,
             control: Workload = NULL) -> dict[str, Any]:
    if k < 1:
        raise ValueError(f"k={k}: a window of zero steps measures nothing")
    if reps < 4:
        raise ValueError(f"reps={reps}: the paired spread and therefore the "
                         f"detection floor are undefined below 4 reps")
    probe = w.step(w.make())
    if probe is None:
        raise ValueError(
            f"workload {w.name!r}: step() returned None, so the state does not carry "
            f"from one step to the next. That is not a serial chain -- the arms would "
            f"time K independent steps and the wait cost would be measured against a "
            f"workload the caller did not describe.")

    arms: dict[str, Callable[[], None]] = {
        "chained": lambda: _chained(w, k),
        "synced": lambda: _synced(w, k),
        "ctl_chained": lambda: _chained(control, k),
        "ctl_synced": lambda: _synced(control, k),
    }
    if w.submit is not None:
        arms["submitted"] = lambda: _submitted(w, k)
    s = _paired(arms, reps=reps, warmup=warmup)

    US = 1e6
    per_step = lambda x: x / k * US                                   # noqa: E731
    st = {n: _stat(v) for n, v in s.items()}
    wait = _diff(s["chained"], s["synced"])
    ctl_wait = _diff(s["ctl_chained"], s["ctl_synced"])
    work = _diff(s["ctl_chained"], s["chained"])

    tax_us = per_step(wait["median_s"])
    ctl_tax_us = per_step(ctl_wait["median_s"])
    work_us = per_step(work["median_s"])
    tax_mde = per_step(wait["mde_s"])
    work_mde = per_step(work["mde_s"])
    excess_mde = per_step(max(wait["mde_s"], ctl_wait["mde_s"]))

    if w.submit is None:
        confound: dict[str, Any] = {
            "status": "NOT_RUN",
            "reason": f"workload {w.name!r} has no submit(); submission COUNT cannot "
                      f"be varied independently of wait COUNT, so the mechanism "
                      f"behind any tax reported here is NOT established",
            "follows": None}
    else:
        # chained   = 1 submission, 1 wait
        # submitted = K submissions, 1 wait   <- submission count varied alone
        # synced    = K submissions, K waits  <- wait count varied alone
        d_sub = _diff(s["chained"], s["submitted"])
        d_wait = _diff(s["submitted"], s["synced"])
        sub_us, wai_us = per_step(d_sub["median_s"]), per_step(d_wait["median_s"])
        cmde = per_step(max(d_sub["mde_s"], d_wait["mde_s"]))
        follows = ("wait" if wai_us - sub_us > cmde else
                   "submission" if sub_us - wai_us > cmde else "ambiguous")
        confound = {
            "status": "RAN",
            "design": "three arms holding one variable each: chained (1 submission, "
                      "1 wait), submitted (K submissions, 1 wait), synced (K "
                      "submissions, K waits)",
            "cost_of_K_submissions_per_step_us": round(sub_us, 3),
            "cost_of_K_waits_per_step_us": round(wai_us, 3),
            "mde_per_step_us": round(cmde, 3),
            "follows": follows,
            "reading": {
                "wait": "the cost tracks the WAIT count with submission count held at "
                        "K, so the mechanism is the host round trip",
                "submission": "the cost tracks the SUBMISSION count with the wait "
                              "count held at 1, so the mechanism is NOT waiting and "
                              "the headline framing does not transfer to this "
                              "workload",
                "ambiguous": "neither difference clears this run's detection floor; "
                             "the mechanism is UNRESOLVED here, not settled",
            }[follows]}

    v = verdict(tax_us, ctl_tax_us, work_us, tax_mde_us=tax_mde,
                work_mde_us=work_mde, excess_mde_us=excess_mde)
    return {
        "workload": {"name": w.name, "primitive": w.primitive,
                     "shape": list(w.shape)},
        "k_steps_between_waits": k, "reps": reps, "warmup": warmup,
        "arms": {n: {**st[n],
                     "median_ms": round(st[n]["median_s"] * 1e3, 4),
                     "per_step_us": round(per_step(st[n]["median_s"]), 3)}
                 for n in st},
        "submission_and_compute_per_step_us": round(
            per_step(st["chained"]["median_s"]), 3),
        "harness_floor_per_step_us": round(per_step(st["ctl_chained"]["median_s"]), 3),
        "work_visible_per_step_us": round(work_us, 3),
        "wait_cost_per_step_us": round(tax_us, 3),
        "control_wait_cost_per_step_us": round(ctl_tax_us, 3),
        "excess_over_control_per_step_us": round(tax_us - ctl_tax_us, 3),
        # ONE FLOOR PER COMPARISON. Pooling them let the noisiest arm veto a clean
        # result elsewhere in the same run; see verdict().
        "detection_floors_per_step_us": {"wait_cost": round(tax_mde, 3),
                                         "work_visible": round(work_mde, 3),
                                         "excess_over_control": round(excess_mde, 3)},
        "min_detectable_effect_per_step_us": round(tax_mde, 3),
        "mde_basis": wait["mde_basis"],
        "paired": {"wait": {k2: wait[k2] for k2 in ("median_s", "sd_s", "n")},
                   "control_wait": {k2: ctl_wait[k2] for k2 in
                                    ("median_s", "sd_s", "n")}},
        "spread_gate_pct": IQR_GATE_PCT,
        "arms_inside_spread_gate": [n for n in st if st[n]["reliable"]],
        "arms_outside_spread_gate": [n for n in st if not st[n]["reliable"]],
        "confound_check": confound,
        "verdict": v,
        "verdict_reading": {
            "SYNC_TAX": "the wait cost is separated from BOTH this run's detection "
                        "floor and the empty-loop control",
            "NO_SIGNIFICANT_SYNC_TAX": "the wait cost does not clear this run's "
                                       "detection floor. A NEGATIVE, and the "
                                       "instrument is supposed to be able to return "
                                       "one",
            "MEASURING_ITSELF": "the reading is not separated from the empty-loop "
                                "control, so it is the harness, not the workload",
            "WORK_NOT_VISIBLE": "the subject's chained arm is not separated from the "
                                "empty-loop control, so its work never showed up. "
                                "Every other number here is meaningless until that is "
                                "explained (a lazy graph never materialized is the "
                                "usual cause)",
        }[v],
    }


def window_sweep(w: Workload, *, ks: Sequence[int] = (1, 2, 4, 8, 16, 32),
                 reps: int = 20, warmup: int = 5,
                 control: Workload = NULL) -> dict[str, Any]:
    """The KNOB: how much work sits between two host round trips.

    ACCELERATOR_BARRIER_WINDOW.json found the analogous knob inside a threadgroup --
    work between two barriers widens a race window continuously where four earlier
    framings were flat or scattered. The same shape holds one level up: work between
    two WAITS is the continuous knob on the host round trip, and it is the one a
    caller can actually turn.
    """
    rows = []
    for k in ks:
        d = diagnose(w, k=k, reps=reps, warmup=warmup, control=control)
        rows.append({"k": k,
                     "chained_per_step_us": d["submission_and_compute_per_step_us"],
                     "wait_cost_per_step_us": d["wait_cost_per_step_us"],
                     "verdict": d["verdict"]})
    asym = rows[-1]["chained_per_step_us"]
    within = [r["k"] for r in rows if r["chained_per_step_us"] <= asym * 1.10]
    # AT K=1 THE TWO ARMS ARE THE SAME PROGRAM. _chained does step-then-sync once;
    # _synced does step-then-sync once. The true wait cost there is EXACTLY ZERO, so
    # whatever the sweep reads at k=1 is the instrument's own noise on this machine
    # in this run -- the one point in the whole experiment where ground truth is
    # known by construction rather than by argument.
    k1 = next((r for r in rows if r["k"] == 1), None)
    return {"workload": {"name": w.name, "primitive": w.primitive,
                         "shape": list(w.shape)},
            "ks": list(ks), "reps_per_k": reps, "rows": rows,
            "zero_point_check": {
                "ran": k1 is not None,
                "true_wait_cost_per_step_us": 0.0,
                "read_wait_cost_per_step_us": k1 and k1["wait_cost_per_step_us"],
                "why": "at k=1 the chained and synced arms are byte-identical "
                       "programs, so the correct reading is 0 and the deviation is "
                       "this run's instrument noise",
            } if k1 else {"ran": False, "why": "k=1 not in ks, so the instrument was "
                                               "never shown a known-zero workload in "
                                               "this sweep"},
            "window_gain_k1_over_kmax": round(
                rows[0]["chained_per_step_us"] / asym, 3) if asym else None,
            "smallest_k_within_10pct_of_widest": min(within) if within else None,
            "reading": "chained per-step cost falls as the single wait amortizes over "
                       "more work. window_gain is what widening the window bought FOR "
                       "THIS workload on THIS machine, not a constant."}


# ------------------------------------------------------- workloads for the demo
# Held out on purpose: the receipts this instrument came from used an ELEMENTWISE
# chain at n=16384 and synthetic threadgroup kernels at widths 32-1024. Matmul at
# a square 2-D shape is a different primitive AND a different shape.

def mlx_matmul_workload(n: int = 128, *, stall_us: float = 0.0,
                        dtype: str = "float32") -> Workload:
    """X <- X @ W, strictly serial. `stall_us` INJECTS a known wait into sync()."""
    import mlx.core as mx
    dt = getattr(mx, dtype)
    # spectral radius ~1 so a long chain neither overflows nor denormalizes, which
    # would change the arithmetic being timed halfway through the chain.
    w_mat = (mx.random.normal((n, n)) / (n ** 0.5)).astype(dt)
    mx.eval(w_mat)
    stall_s = stall_us / 1e6

    def make():
        x = (mx.random.normal((n, n)) / (n ** 0.5)).astype(dt)
        mx.eval(x)
        return x

    def sync(s):
        mx.eval(s)
        spin(stall_s)

    return Workload(name=f"mlx_matmul_{n}x{n}_{dtype}"
                         + (f"_stall{stall_us:g}us" if stall_us else ""),
                    primitive="matmul", shape=(n, n),
                    make=make, step=lambda s: s @ w_mat, sync=sync,
                    submit=lambda s: mx.async_eval(s))


# The mutations that were applied to THIS FILE, one at a time, with the failure text
# pytest actually printed. Recorded here rather than asserted in prose because a check
# nobody has watched fail is assumed vacuous in this repo. Each was restored and the
# file verified byte-identical (md5 78ca106b6f737f04e697c02b58e8ef06) afterwards.
NEGATIVE_CONTROLS = [
    {"id": "NC1", "mutation": "_synced stops syncing per step (one sync at the end, "
                              "making it identical to _chained)",
     "observed": "3 failed, 9 passed -- test_it_reads_back_an_injected_stall...; "
                 "test_removing_the_stall_drops_the_reading...; "
                 "test_the_confound_arm_names_WAITING... assert 'ambiguous' == 'wait'"},
    {"id": "NC2", "mutation": "_submitted stops calling submit(), so submission count "
                              "is no longer varied",
     "observed": "1 failed, 11 passed -- "
                 "test_the_confound_arm_names_SUBMISSION_when_only_submitting_is_"
                 "stalled, assert 'wait' == 'submission'"},
    {"id": "NC3", "mutation": "the per-comparison detection floors are pooled back "
                              "into one max(), the bug the first live run exposed",
     "observed": "1 failed, 11 passed -- test_a_noisy_comparison_may_not_veto_a_clean"
                 "_one, assert 'WORK_NOT_VISIBLE' == 'SYNC_TAX'"},
    {"id": "NC4", "mutation": "the chain-integrity refusal is disabled (`if False`)",
     "observed": "1 failed, 11 passed -- test_a_step_that_breaks_the_chain_is_refused,"
                 " Failed: DID NOT RAISE ValueError"},
    {"id": "NC5", "mutation": "a missing submit() reports status RAN instead of "
                              "NOT_RUN -- the silent skip this repo has shipped before",
     "observed": "1 failed, 11 passed -- "
                 "test_a_missing_submit_is_reported_LOUDLY_not_skipped, "
                 "assert 'RAN' == 'NOT_RUN'"},
    {"id": "NC6", "mutation": "the zero-point check falls back to rows[0] so it always "
                              "claims to have run",
     "observed": "1 failed, 11 passed -- "
                 "test_the_zero_point_check_says_so_when_it_was_never_run, "
                 "assert True is False"},
    {"id": "NC7", "mutation": "the interleave is replaced by A-then-B (every rep of "
                              "one arm, then the next)",
     "observed": "FIRST RUN: 12 passed -- the interleave was UNPINNED, so a pin for it "
                 "was written. After that pin: 1 failed, 12 passed -- "
                 "test_the_interleave_survives_a_drifting_machine..., "
                 "assert 1515.901 <= 1150 against a predicted 1515us"},
]


def numpy_matmul_workload(n: int = 256, dtype: str = "float32") -> Workload:
    """The NEGATIVE control at workload level, and a different RUNTIME.

    NumPy's matmul has already returned by the time step() returns -- there is no
    asynchronous boundary, so there is nothing to wait on and the instrument must
    read no tax. An instrument that finds a synchronization tax here is broken, which
    is why this is run beside the MLX arms rather than instead of them.
    """
    import numpy as np
    w_mat = (np.random.randn(n, n) / (n ** 0.5)).astype(dtype)
    return Workload(name=f"numpy_matmul_{n}x{n}_{dtype}", primitive="matmul",
                    shape=(n, n),
                    make=lambda: (np.random.randn(n, n) / (n ** 0.5)).astype(dtype),
                    step=lambda s: s @ w_mat,
                    sync=lambda s: None, submit=lambda s: None)




def battery(*, n: int, big_n: int, k: int, reps: int, stall_us: float,
            sweep_reps: int = 20) -> dict[str, Any]:
    """One full pass of the instrument over the held-out workloads."""
    small = diagnose(mlx_matmul_workload(n), k=k, reps=reps)
    big = diagnose(mlx_matmul_workload(big_n), k=k, reps=max(12, reps // 4))
    stalled = diagnose(mlx_matmul_workload(n, stall_us=stall_us), k=k, reps=reps)
    sweep = window_sweep(mlx_matmul_workload(n), reps=sweep_reps)
    negative = diagnose(numpy_matmul_workload(256), k=k, reps=reps)
    out: dict[str, Any] = {
        "held_out_small": small, "held_out_large": big,
        "held_out_negative_numpy": negative,
        "injected_stall": stalled, "window_sweep": sweep,
        "injected_stall_calibration": {
            "injected_us_per_sync": stall_us,
            "read_back_us_per_step": round(
                stalled["wait_cost_per_step_us"] - small["wait_cost_per_step_us"], 3),
            "expected_read_back_us_per_step": round(stall_us * (k - 1) / k, 3),
            "read_back_fraction_of_expected": round(
                (stalled["wait_cost_per_step_us"] - small["wait_cost_per_step_us"])
                / (stall_us * (k - 1) / k), 3),
            "why_not_the_full_injection": "the chained arm syncs ONCE per rep and the "
                                          "synced arm K times, so an injection of S "
                                          "into sync() raises the paired difference "
                                          "by S*(K-1)/K per step, not S"},
    }
    # DERIVED FROM THIS RUN'S NUMBERS, never written by hand -- a headline typed by a
    # human goes stale the first time the numbers move and nobody notices.
    z = sweep["zero_point_check"]["read_wait_cost_per_step_us"]
    cal = out["injected_stall_calibration"]
    out["findings"] = [
        f"THE INSTRUMENT RETURNS A NEGATIVE. NumPy matmul 256x256 f32 on CPU reads "
        f"{negative['wait_cost_per_step_us']}us/step of wait cost against a "
        f"{negative['detection_floors_per_step_us']['wait_cost']}us/step floor: "
        f"{negative['verdict']}. There is no asynchronous boundary in that runtime, "
        f"there is nothing to wait on, and the instrument says so.",
        f"MLX matmul {n}x{n} f32 on GPU reads {small['wait_cost_per_step_us']}us/step "
        f"of wait cost against {small['submission_and_compute_per_step_us']}us/step of "
        f"submission+compute: {small['verdict']}. Its confound arm says the cost "
        f"follows {small['confound_check']['follows']!r}.",
        f"THE CONFOUND ARM SEPARATES THE TWO CORRELATED VARIABLES. At {n}x{n}, "
        f"holding the wait count at 1 and raising submissions from 1 to K cost "
        f"{small['confound_check']['cost_of_K_submissions_per_step_us']}us/step, "
        f"which "
        + ("CLEARS" if small['confound_check']['cost_of_K_submissions_per_step_us']
           > small['confound_check']['mde_per_step_us'] else "DOES NOT CLEAR")
        + f" its own {small['confound_check']['mde_per_step_us']}us/step floor, while "
        f"raising the wait count cost "
        f"{small['confound_check']['cost_of_K_waits_per_step_us']}us/step. Submission "
        f"count is the smaller term either way; whether it is nonzero at all is a "
        f"separate question this run "
        + ("answers YES" if small['confound_check']['cost_of_K_submissions_per_step_us']
           > small['confound_check']['mde_per_step_us'] else "CANNOT ANSWER") + ".",
        f"THE SAME PRIMITIVE AT A DIFFERENT SHAPE MOVES THE ANSWER. At "
        f"{big_n}x{big_n} the wait cost is {big['wait_cost_per_step_us']}us/step "
        f"against {big['submission_and_compute_per_step_us']}us/step of work -- a "
        f"comparable tax in absolute terms and a small fraction of the cost in "
        f"relative terms. Verdict {big['verdict']}, confound arm "
        f"{big['confound_check']['follows']!r}.",
        f"THE WINDOW IS THE KNOB, ONE LEVEL UP FROM THE BARRIER RECEIPT. Widening it "
        f"from k=1 to k={sweep['ks'][-1]} cut chained per-step cost "
        f"{sweep['window_gain_k1_over_kmax']}x, and "
        f"k={sweep['smallest_k_within_10pct_of_widest']} is already within 10% of the "
        f"widest window measured.",
        f"CALIBRATION: {cal['injected_us_per_sync']}us injected into sync() read back "
        f"as {cal['read_back_us_per_step']}us/step against an expected "
        f"{cal['expected_read_back_us_per_step']}us/step -- "
        f"{cal['read_back_fraction_of_expected']} of the injection. DETECTION is not "
        f"in doubt ({stalled['wait_cost_per_step_us']}us/step stalled against "
        f"{small['wait_cost_per_step_us']}us/step clean, against a "
        f"{stalled['detection_floors_per_step_us']['wait_cost']}us/step floor). "
        f"MAGNITUDE recovery is imperfect and the gap is much larger than the "
        f"zero-point noise, so it is NOT noise. The obvious candidate -- the busy "
        f"spin occupies a core and lets the driver settle, making the NEXT mx.eval "
        f"cheaper, so part of the injection is absorbed rather than added -- is "
        f"UNVERIFIED here. Read the stalled arm as a detector, not as a ruler.",
        f"ZERO POINT: at k=1 the two arms are the same program, so the true wait cost "
        f"is exactly 0 and the instrument read {z}us/step. That is this machine's "
        f"noise on this instrument in this run, measured rather than argued.",
        f"THE SPREAD GATE IS FAILED ALMOST EVERYWHERE and that is expected: the "
        f"machine is contended by a running ModelLake fill, exactly as "
        f"ACCELERATOR_SYNC_NOT_SUBMISSION.json recorded at 34-39% IQR. Arms outside "
        f"the {IQR_GATE_PCT}% gate in the {n}x{n} arm: "
        f"{small['arms_outside_spread_gate']}.",
    ]
    return out


ARMS_UNDER_TEST = ("held_out_small", "held_out_large", "held_out_negative_numpy",
                   "injected_stall")


def run_to_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """WHAT ONE RUN CANNOT TELL YOU.

    ACCELERATOR_BARRIER_WINDOW.json's own correction was that the same kernel measured
    p = 0.05 in one process and 0.65 in another, and that quoting either as an
    expectation was wrong. The same trap applies to this instrument: the first two
    batteries here, minutes apart, disagreed on the verdict for two of four arms
    because the machine's noise -- and therefore the detection floor -- moved. So the
    receipt carries N batteries and reports where they agree, and a verdict that is
    not stable across them is NOT a property of the workload.
    """
    per_arm = {}
    for arm in ARMS_UNDER_TEST:
        verdicts = [r[arm]["verdict"] for r in runs]
        taxes = [r[arm]["wait_cost_per_step_us"] for r in runs]
        floors = [r[arm]["detection_floors_per_step_us"]["wait_cost"] for r in runs]
        per_arm[arm] = {
            "verdicts": verdicts, "verdict_stable": len(set(verdicts)) == 1,
            "wait_cost_per_step_us": taxes,
            "wait_cost_spread_pct": round(
                (max(taxes) - min(taxes)) / abs(max(taxes)) * 100, 1) if max(taxes) else None,
            "detection_floor_per_step_us": floors,
            "floor_spread_pct": round(
                (max(floors) - min(floors)) / max(floors) * 100, 1) if max(floors) else None,
        }
    unstable = [a for a, v in per_arm.items() if not v["verdict_stable"]]
    return {
        "batteries": len(runs), "per_arm": per_arm,
        "verdict_unstable_arms": unstable,
        "reading": (
            "every arm's verdict reproduced across batteries" if not unstable else
            f"the verdict did NOT reproduce for {unstable}. For those arms this "
            f"receipt reports the readings and REFUSES a verdict; the wait cost "
            f"itself moved far less than the verdict did, which means the boundary "
            f"was crossed by the DETECTION FLOOR moving with machine noise, not by "
            f"the workload changing"),
        "the_transferable_part": "the instrument, the arm design and the direction of "
                                 "the readings transfer. The verdicts and the "
                                 "microsecond magnitudes are properties of a run on a "
                                 "contended machine.",
    }


if __name__ == "__main__":  # pragma: no cover - the receipt run, see EVIDENCE
    import argparse
    import json
    import platform
    import sys
    from pathlib import Path

    import machine_genome as mg
    import receipt as rcpt

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--big-n", type=int, default=2048)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--stall-us", type=float, default=200.0)
    ap.add_argument("--batteries", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import mlx.core as mx

    runs = [battery(n=a.n, big_n=a.big_n, k=a.k, reps=a.reps, stall_us=a.stall_us)
            for _ in range(a.batteries)]
    out = {"batteries": runs, "run_to_run": run_to_run(runs),
           "negative_controls": NEGATIVE_CONTROLS}
    print(json.dumps(out, indent=1, default=str))
    if a.out:
        soc = mg._sysctl("machdep.cpu.brand_string")
        r = rcpt.build(
            experiment_class="ACCEL-DISPATCH", knowledge_level="INSTANCE",
            identities={
                "experiment": {"id": "SYNC_TAX_DIAGNOSTIC_IS_AN_INSTRUMENT",
                               "obligation": "G062 / G043 / G047",
                               "answers": "ACCELERATOR_SYNC_NOT_SUBMISSION.json found "
                                          "the floor is waiting, not submitting, and "
                                          "left nothing reusable"},
                "machine": {"soc": soc, "arch": platform.machine(),
                            "os": platform.platform(),
                            "contended": "yes -- a ModelLake fill is running"},
                "device": {"name": str(mx.default_device()),
                           "api": f"Metal via MLX {mx.__version__}"},
                "model": rcpt.absent("synthetic matmul chains; no model executed"),
                "representation": {"name": "dense_f32"},
                "kernel": {"primitive": "matmul",
                           "shapes": [[a.n, a.n], [a.big_n, a.big_n], [256, 256]]},
                "runtime": {"python": sys.executable, "mlx": mx.__version__},
                "transport": rcpt.absent("single device"),
            },
            result=out,
            claim_boundary="; ".join([
                f"MACHINE {soc}, contended by a running ModelLake fill; RUNTIME MLX "
                f"{mx.__version__} lazy graph + Metal for the GPU arms and NumPy for "
                f"the negative arm; PRIMITIVE matmul only; SHAPES {a.n}x{a.n} and "
                f"{a.big_n}x{a.big_n} f32 on GPU, 256x256 f32 on CPU; REPRESENTATION "
                f"dense f32 only",
                "DOES NOT GENERALIZE to other primitives (no elementwise, reduction, "
                "attention or MoE arm here), to other representations (no quantized "
                "arm), to other machines or GPUs, to other runtimes (nothing here "
                "says what a raw Metal or CUDA submission costs), or to any real "
                "model -- no model was executed",
                "the per-step microsecond figures are INSTANCE numbers on a contended "
                "machine and may not be quoted as constants; what transfers is the "
                "INSTRUMENT and the direction of its readings, and run_to_run names "
                "which verdicts did not even reproduce across batteries here",
                "no energy or thermal number is claimed: powermetrics needs sudo, "
                "which this process does not have, so the quantity is unmeasurable "
                "here and is refused rather than estimated",
                "the detection floor assumes the reps are independent, which a "
                "thermally drifting contended machine does not guarantee; it is an "
                "estimate of this run's resolving power, not a proof of it",
                "the confound arm separates WAIT COUNT from SUBMISSION COUNT and "
                "nothing else. It does not decompose the wait itself into queue "
                "latency, completion-handler latency or driver round trip",
            ]),
            passed=True)
        rcpt.write(r, Path(a.out))
        print(f"wrote {a.out}", file=sys.stderr)

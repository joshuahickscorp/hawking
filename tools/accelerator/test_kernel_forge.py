"""Kernel Forge pins, including the bandit."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
import kernel_forge as kf  # noqa: E402


class Fake:
    def __init__(self, name, t, iqr=1.0):
        self.name, self.t, self.iqr = name, t, iqr
    def __str__(self):
        return self.name


def measure_of(cands):
    def m(c, reps):
        return c.t, c.iqr
    return m


def test_successive_halving_spends_less_than_flat_measurement():
    cands = [Fake(f"c{i}", 1.0 + i * 0.1) for i in range(16)]
    res = kf.successive_halving(cands, measure=measure_of(cands), rounds=3, base_reps=5)
    assert res["total_reps_spent"] < kf.exhaustive_cost(len(cands), 40)


def test_it_finds_the_clear_winner_when_the_landscape_is_not_flat():
    cands = [Fake("slow", 10.0), Fake("mid", 5.0), Fake("FAST", 1.0), Fake("slower", 20.0)]
    res = kf.successive_halving(cands, measure=measure_of(cands), rounds=3, base_reps=5)
    assert res["champion"].name == "FAST"


def test_an_unreliable_candidate_cannot_be_champion():
    """The bug that actually happened: the first bandit ranked on median alone and
    crowned a candidate that had failed the forge's own 10% IQR gate."""
    cands = [Fake("fast_but_jittery", 1.0, iqr=30.0), Fake("steady", 2.0, iqr=2.0)]
    res = kf.successive_halving(cands, measure=measure_of(cands), rounds=1, base_reps=5)
    assert res["champion"].name == "steady"
    # Check the BEHAVIOUR, not the mechanism. An earlier version of this test also
    # required the jittery candidate to appear in rejected_for_unreliability, which
    # pinned it to being rejected at the FINAL gate. The better fix sinks it during
    # elimination instead, so it never reaches that gate -- and the over-specific
    # assertion failed on a change that improved the code.
    eliminated_early = any("fast_but_jittery" in r["eliminated"] for r in res["rounds"])
    rejected_late = any(r["candidate"] == "fast_but_jittery"
                        for r in res["rejected_for_unreliability"])
    assert eliminated_early or rejected_late


def test_no_champion_when_every_survivor_is_unreliable():
    cands = [Fake("a", 1.0, iqr=40.0), Fake("b", 2.0, iqr=50.0)]
    res = kf.successive_halving(cands, measure=measure_of(cands), rounds=1, base_reps=5)
    assert res["champion"] is None
    assert len(res["rejected_for_unreliability"]) >= 1


def test_each_round_keeps_half_and_doubles_the_reps():
    cands = [Fake(f"c{i}", float(i)) for i in range(8)]
    res = kf.successive_halving(cands, measure=measure_of(cands), rounds=3, base_reps=5)
    reps = [r["reps_each"] for r in res["rounds"]]
    assert reps == [5, 10, 20]
    assert [r["kept"] for r in res["rounds"]] == [4, 2, 1]


def test_a_jittery_candidate_cannot_knock_out_steady_ones_during_elimination():
    """The failure this actually had: gating only at the END let a fast jittery
    candidate win round one, eliminate every steady candidate, and then be rejected
    itself -- leaving no champion at all. The gate fired correctly and the answer was
    still wrong."""
    cands = [Fake("jitter1", 0.1, iqr=40.0), Fake("jitter2", 0.2, iqr=40.0),
             Fake("steady", 5.0, iqr=1.0), Fake("steady2", 6.0, iqr=1.0)]
    res = kf.successive_halving(cands, measure=measure_of(cands), rounds=2, base_reps=5)
    assert res["champion"] is not None, "a reliable candidate existed and must survive"
    assert res["champion"].name == "steady"


def test_bench_flags_a_reliability_verdict_taken_from_too_few_reps():
    """Measured: the same kernel's IQR ranged 1.84-13.38% across eight independent
    20-rep runs, so the gate verdict flipped run to run. A verdict below the stable
    threshold must say so rather than presenting itself as a property of the kernel."""
    import bench
    calls = {"n": 0}
    def work():
        calls["n"] += 1
    few = bench.time_arm(work, reps=20, warmup=2)
    assert few["reliability_verdict_is_stable"] is False
    assert "may flip on a rerun" in few["reliability_caveat"]
    many = bench.time_arm(work, reps=bench.STABLE_RELIABILITY_MIN_REPS, warmup=2)
    assert many["reliability_verdict_is_stable"] is True
    assert many["reliability_caveat"] is None


def test_per_tg_is_the_variable_and_the_grid_repeats_itself():
    """Candidates sharing per_tg are the same physical point, so the default grid
    explores far fewer distinct configurations than it has candidates."""
    cands = kf.generate(1 << 20)
    groups = kf.collapse_groups(cands)
    assert len(cands) == 15 and len(groups) == 7
    assert sorted(groups) == [64, 128, 256, 512, 1024, 2048, 4096]
    # the repeats are real: per_tg 256 is reached three different ways
    assert sorted(groups[256]) == ["tg128_ept2", "tg256_ept1", "tg64_ept4"]


def test_the_threadgroup_prior_is_documented_as_primitive_specific():
    """The prior's floor was measured on the fused chain and does NOT transfer -- a
    write-only kernel is still 1.9x off its floor at per_tg 64. Pinning the caveat in
    the source is what stops the tuple being read as a property of the machine."""
    src = Path(kf.__file__).read_text()
    assert "PRIMITIVE-SPECIFIC" in src and "CHAIN-SHAPED PRIOR" in src


def test_a_sweep_that_ran_nothing_is_not_a_passing_sweep():
    """The first shape fuzz in this program reported ZERO FAILURES for four primitives
    it never executed -- one filter demanded every dimension be a multiple of 8 while
    the grid held exactly one such value. A coverage counter is part of the check."""
    with pytest.raises(ValueError, match="ZERO cases"):
        kf.shape_sweep([], lambda c: 0.0, name="empty")


def test_shape_sweep_records_a_raise_as_a_failure_not_a_skip():
    """A shape the kernel REFUSES is a shape the caller needs told about; skipping it
    would let a refusal read as a pass."""
    def check(case):
        if case[0] == 3:
            raise RuntimeError("threadgroup memory exceeded")
        return 0.0
    r = kf.shape_sweep([(1,), (3,), (7,)], check, name="raises")
    assert r["cases_run"] == 3 and r["failures"] == 1 and not r["all_ok"]
    assert "threadgroup memory" in r["failing"][0]["raised"]
    good = kf.shape_sweep([(1,), (7,)], check, name="clean")
    assert good["all_ok"] and good["cases_run"] == 2


def test_shape_sweep_would_have_caught_the_tiled_matmul_launch_bug():
    """The defect: correct whenever m is a multiple of the tile, wrong otherwise. A
    single-shape check at 64x64x64 passes it; the sweep does not."""
    def check(case):
        m, tile = case
        return 0.0 if m % tile == 0 else 3.84      # the measured 60x60x60 error
    assert kf.shape_sweep([(64, 16)], check, name="one")["all_ok"]
    swept = kf.shape_sweep([(m, 16) for m in kf.AWKWARD_SHAPES],
                                     check, name="many")
    assert not swept["all_ok"] and swept["failures"] == len(kf.AWKWARD_SHAPES)


def test_a_sweep_whose_control_never_fails_is_refused():
    """A probabilistic sweep reporting zero failures is exactly the shape of a check
    that cannot fail. If the broken arm passes everywhere, the clean result is
    evidence of nothing and the sweep says so instead of reporting green."""
    with pytest.raises(ValueError, match="passed at every case"):
        kf.swept_with_control([(64,), (256,)], lambda c: 0.0, lambda c: 0.0,
                              name="vacuous")


def test_the_control_may_be_blind_at_some_widths_and_that_is_reported():
    """Stripping one barrier from AirNorm is caught at threadgroups 64 through 1024
    and is EXACT at 32, where the whole threadgroup is one simdgroup. A control run
    only at 32 would have called the broken kernel fine, so the widths where the
    control is blind are part of the result."""
    def broken(case):
        return 0.0 if case[0] == 32 else 1.0      # the measured lockstep behaviour
    r = kf.swept_with_control([(w,) for w in kf.WIDTH_PRIOR],
                              lambda c: 0.0, broken, name="lockstep", repeat=3)
    assert r["all_ok"] and r["failures"] == 0
    assert r["control_blind_at"] == [[32]]
    assert len(r["control_detected_at"]) == 5
    assert r["executions"] == len(kf.WIDTH_PRIOR) * 3 * 2


def test_repeat_sets_the_detection_floor_not_the_case_list():
    """A width-dependent defect is usually a RACE and therefore probabilistic; the
    resolving power comes from repeats, and the floor is reported so nobody reads a
    clean sweep as proof that no race exists."""
    one = kf.swept_with_control([(1,)], lambda c: 0.0, lambda c: 1.0, repeat=1)
    many = kf.swept_with_control([(1,)], lambda c: 0.0, lambda c: 1.0, repeat=8)
    assert one["detection_floor_per_run"] > many["detection_floor_per_run"]
    assert many["detection_floor_per_run"] < 0.10


# ---------------------------------------------------------------------------
# Two controls, and the raise that looked like a detection.
# ACCELERATOR_QUIET_CONTROL.json.
# ---------------------------------------------------------------------------

def test_a_control_that_raises_refuses_the_sweep():
    """A control that crashes never ran the kernel. Counting its exception as `inf`
    -- the rule that is CORRECT for the real arm -- credited a control with detecting
    the defect at every width when a TypeError in the probe was all it detected."""
    with pytest.raises(ValueError, match="RAISED at case"):
        kf.swept_with_control([(64,), (256,)], lambda c: 0.0,
                              [("crashes", lambda c: 1 / 0)], name="raiser")


def test_the_real_arm_still_counts_a_raise_as_a_failure():
    """The asymmetry is deliberate: a shape the kernel refuses is a shape the caller
    needs told about, while a control that refuses proves nothing."""
    def half_raises(case):
        if case[0] == 256:
            raise RuntimeError("kernel refuses this width")
        return 0.0
    r = kf.swept_with_control([(64,), (256,)], half_raises,
                              [("ctl", lambda c: 9.9)], name="mixed")
    assert r["failures"] == 1 and not r["all_ok"]


def test_two_controls_report_separate_blind_lists():
    """ONE control's blind list is not the SWEEP's. Measured on real kernels: a barrier
    on a TOTAL dependency is caught at 5 of 6 widths (40 of 48 runs) while one on an
    INCIDENTAL dependency is caught at 2 of 6 (4 of 48) -- same repeat, same widths,
    same machine."""
    cases = [(32,), (64,), (256,)]
    loud = lambda c: 9.9                                     # fires everywhere
    quiet = lambda c: 9.9 if c[0] == 256 else 0.0            # fires at one width
    r = kf.swept_with_control(cases, lambda c: 0.0,
                              [("loud", loud), ("quiet", quiet)], name="two")
    by = {c["label"]: c for c in r["controls"]}
    assert by["loud"]["blind_at"] == []
    assert [x[0] for x in by["quiet"]["blind_at"]] == [32, 64]
    # the sweep is NOT refused for a control being quiet -- only for one that is blind
    # everywhere, which proves nothing at all.
    assert r["all_ok"]


def test_blind_at_every_control_is_what_the_sweep_actually_missed():
    """A case no control reached is a case the sweep says nothing about, however many
    controls ran."""
    cases = [(32,), (64,), (256,)]
    a = lambda c: 9.9 if c[0] == 64 else 0.0
    b = lambda c: 9.9 if c[0] == 256 else 0.0
    r = kf.swept_with_control(cases, lambda c: 0.0, [("a", a), ("b", b)], name="union")
    assert [x[0] for x in r["blind_at_every_control"]] == [32]
    # and the back-compat field describes the FIRST control alone, which is the trap
    assert [x[0] for x in r["control_blind_at"]] == [32, 256]


def test_the_floor_bounds_repeat_and_says_so():
    """1 - 0.5**(1/repeat) is calibrated -- 40 blocks of 8 on a real race at p=0.1187
    detected in 25 against a binomial 0.6363 -- but no repeat count rescues p = 0."""
    r = kf.swept_with_control([(64,)], lambda c: 0.0, [("c", lambda c: 9.9)],
                              name="floor", repeat=8)
    assert r["detection_floor_per_run"] == 0.083
    assert r["floor_bounds_repeat_not_coverage"] is True


# --- barrier_control_prior: the mechanism behind the measured 10x -------------
# ACCELERATOR_QUIET_CONTROL.json measured a LOUD control firing 40 of 48 and a
# QUIET one 4 of 48 and could not say why. A factorial found the cause is the
# UPSTREAM BARRIER (1.000 vs 0.084), not what the racing read finds (0.574 vs
# 0.510). These pin BOTH directions, because a prior that only ever says LOUD is
# as useless as one that only ever says QUIET.

_NORM_SHAPED = """
    float ss = 0.0f;
    for (uint c = lid; c < cols; c += tg) ss += x[base + c] * x[base + c];
    if (lane == 0u) red[warp] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0u; i < lanes; ++i) tot += red[i];
"""
_TOPK_SHAPED = """
    cv[lid] = bv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span = tg / 2u; span > 0u; span >>= 1u) {
        if (lid < span) cv[lid] = max(cv[lid], cv[lid + span]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


def test_a_barrier_with_NOTHING_upstream_is_a_LOUD_control():
    """AirNorm's row sum: a long unsynchronized scan, then the write, then the
    barrier. The threadgroup arrives with full accumulated skew."""
    p = kf.barrier_control_prior(_NORM_SHAPED, 0)
    assert p["has_upstream_barrier"] is False
    assert p["verdict"] == "LOUD_CONTROL"
    # The loud case IS a reliable prediction: every cell ever measured with no
    # upstream barrier fires 1.00 -- all eight factorial cells and every width and
    # work count in the window sweep.
    # REFUTED 2026-08-26: the loud case carried 1.000 until gravity_native's serial
    # reduction tail -- no upstream barrier, so LOUD by this prior -- measured 0.067
    # over 60 runs with the intact kernel wrong 0 of 60. No number is offered now.
    assert p["expected_p_fired"] is None
    assert p["expected_p_fired_range"] == kf.LOUD_OBSERVED_RANGE
    assert p["blind_to_lane_work_imbalance"] is True


def test_a_barrier_JUST_AFTER_ANOTHER_is_a_QUIET_control_and_says_so():
    """Top-k's tree reduce: the previous iteration's barrier resynchronized the
    group one compare-and-swap ago, so the race window is narrow."""
    p = kf.barrier_control_prior(_TOPK_SHAPED, 1)
    assert p["has_upstream_barrier"] is True
    # NO POINT ESTIMATE, and that is the correction: the same kernel measured
    # p = 0.05 and 0.65 at width 256 in two processes, so the factorial's 0.084 was
    # a sample from an unstable distribution and a number here would be cited as an
    # expectation it cannot support.
    assert p["expected_p_fired"] is None
    assert p["expected_p_fired_range"] == (0.01, 0.90)
    # the WEAKNESS must be in the verdict, not left for a reader to infer from a
    # small number: a quiet control means the SWEEP proved little, never that the
    # barrier is unnecessary.
    assert p["verdict"] == "QUIET_CONTROL_WEAK_EVIDENCE"


def test_the_SAME_source_gives_OPPOSITE_verdicts_for_different_barriers():
    """The prior is a property of a POSITION, not of a kernel -- top-k's first
    barrier has nothing upstream while its loop barriers do."""
    first = kf.barrier_control_prior(_TOPK_SHAPED, 0)
    later = kf.barrier_control_prior(_TOPK_SHAPED, 1)
    assert first["verdict"] == "LOUD_CONTROL"
    assert later["verdict"] == "QUIET_CONTROL_WEAK_EVIDENCE"


def test_naming_a_barrier_that_is_not_there_RAISES():
    """Silently returning the prior for a different barrier would be worse than
    refusing: the caller would act on a number about the wrong position."""
    import pytest
    with pytest.raises(ValueError, match="outside"):
        kf.barrier_control_prior(_NORM_SHAPED, 7)


def test_the_prior_travels_with_the_control_it_describes():
    """A control blind for a KNOWN structural reason is interpretable; one blind
    for an unknown reason is not, and the two read identically in a blind list."""
    prior = kf.barrier_control_prior(_TOPK_SHAPED, 1)
    r = kf.swept_with_control([(64,), (1024,)], lambda c: 0.0,
                              [("quiet", lambda c: 9.9 if c[0] == 1024 else 0.0)],
                              name="p", repeat=1, control_priors={"quiet": prior})
    assert r["controls"][0]["prior"]["verdict"] == "QUIET_CONTROL_WEAK_EVIDENCE"
    assert r["controls"][0]["blind_at"] == [[64]]


def test_a_control_with_no_prior_supplied_reports_None_not_a_guess():
    r = kf.swept_with_control([(64,)], lambda c: 0.0, [("c", lambda c: 9.9)],
                              name="p", repeat=1)
    assert r["controls"][0]["prior"] is None


def test_an_arm_faster_than_the_clock_is_UNMEASURABLE_not_perfectly_stable():
    """Found as an intermittent ZeroDivisionError at bench.py:42, roughly one run in
    four: perf_counter can return the same value twice for a cheap callable, and the
    spread was computed by dividing by that sample. Zero spread would have read as the
    most reliable arm ever timed, so the arm is refused instead."""
    import bench
    r = bench.time_arm(lambda: None, reps=8, warmup=1)
    if r["below_timer_resolution"]:
        assert r["iqr_spread_pct"] == float("inf")
        assert r["reliable"] is False          # cannot pass a gate it never entered
    else:                                       # the clock did resolve it: normal path
        assert r["iqr_spread_pct"] >= 0.0



def test_the_quiet_prior_holds_only_while_the_gap_is_SMALL():
    """Work between the two barriers widens the race window and drives p back to
    1.00 -- ~16 fused multiply-adds at width 1024, ~256 at 256, ~1024 at 64. A
    quiet control is quiet because the gap is short, not because the barrier is
    unimportant, so the gap size travels with the verdict."""
    p = kf.barrier_control_prior(_TOPK_SHAPED, 1)
    assert p["quiet_only_while_the_gap_is_small"] is True
    assert p["statements_since_upstream"] >= 1
    loud = kf.barrier_control_prior(_NORM_SHAPED, 0)
    assert loud["quiet_only_while_the_gap_is_small"] is False

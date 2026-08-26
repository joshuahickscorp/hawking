"""Pins for the synchronization tax diagnostic (G062).

These run under BOTH interpreters on this machine on purpose. The default python3
is 3.14.6 and has no mlx; a suite that importorskip'd mlx would print a skip count
here and a summary reader would take it for coverage. So every pin below drives the
instrument with a SYNTHETIC workload whose stall is injected by a busy spin and is
therefore KNOWN, and the mlx demonstration is a live run recorded in
receipts/headless/ACCELERATOR_SYNC_DIAGNOSTIC.json -- physical measurement, which
outranks any test.

The bands are wide because the machine is contended. They are not wide enough to
pass if the instrument stops separating wait from work: the stalled arm injects
1000us per sync against a 50us step, so a diagnostic that confused the two would
miss the band by an order of magnitude, not a few percent.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
import sync_diagnostic as sd  # noqa: E402

K = 8
REPS = 8


def fake(*, step_us=0.0, sync_us=0.0, submit_us=None, chain=True):
    """A workload whose every cost is injected and therefore known."""
    def step(s):
        sd.spin(step_us / 1e6)
        return (s + 1) if chain else None

    def sync(s):
        sd.spin(sync_us / 1e6)

    submit = None
    if submit_us is not None:
        def submit(s):                                            # noqa: F811
            sd.spin(submit_us / 1e6)

    return sd.Workload(name="fake", primitive="spin", shape=(1,),
                       make=lambda: 0, step=step, sync=sync, submit=submit)


def test_it_reads_back_an_injected_stall_at_roughly_the_injected_size():
    """1000us injected into sync(), 50us into step(). The wait cost per step should
    come back near 1000*(K-1)/K = 875us -- the chained arm syncs once per rep and the
    synced arm K times, so the paired difference carries K-1 of the injections."""
    d = sd.diagnose(fake(step_us=50, sync_us=1000, submit_us=0), k=K, reps=REPS,
                    warmup=1)
    assert d["verdict"] == "SYNC_TAX", d
    assert 700 <= d["wait_cost_per_step_us"] <= 1200, d["wait_cost_per_step_us"]
    # the instrument must publish its own blindness, and it must be finite
    assert 0 < d["min_detectable_effect_per_step_us"] < 700, d
    assert set(d["detection_floors_per_step_us"]) == {
        "wait_cost", "work_visible", "excess_over_control"}
    # every arm timed in the SAME interleave, not A-then-B
    assert set(d["arms"]) == {"chained", "synced", "ctl_chained", "ctl_synced",
                              "submitted"}
    assert all(a["reps"] == REPS for a in d["paired"].values() if "reps" in a)


def test_removing_the_stall_drops_the_reading_by_orders_of_magnitude():
    """The other half of the same control. If both arms read the same tax the
    instrument is measuring itself, not the workload.

    The clean arm's verdict is deliberately NOT pinned. Its sync() is still a real
    Python call, and the instrument resolves the ~0.06us/step that K of them cost
    against a ~0.01us floor -- so it says SYNC_TAX, correctly, about a tax four orders
    of magnitude below the injected one. A verdict is a yes/no; the number is what
    carries the size, and the number is what this pins."""
    stalled = sd.diagnose(fake(step_us=50, sync_us=1000, submit_us=0), k=K,
                          reps=REPS, warmup=1)
    clean = sd.diagnose(fake(step_us=50, sync_us=0, submit_us=0), k=K, reps=REPS,
                        warmup=1)
    assert clean["wait_cost_per_step_us"] * 100 < stalled["wait_cost_per_step_us"], (
        clean["wait_cost_per_step_us"], stalled["wait_cost_per_step_us"])
    assert clean["wait_cost_per_step_us"] < 5.0, clean["wait_cost_per_step_us"]
    assert stalled["wait_cost_per_step_us"] > 500.0, stalled["wait_cost_per_step_us"]


def test_a_workload_indistinguishable_from_the_empty_loop_never_reads_SYNC_TAX():
    """The vacuity a lazy graph produces: the chained arm costs the same as an empty
    loop, so the work never showed up and any tax number is meaningless.

    The subject here IS the empty loop, under another name. The verdict must be one
    of the three that report no finding; which one it lands on depends on where the
    run's noise falls, and SYNC_TAX is the one it may never be.

    Measured while writing this pin: a subject whose step merely calls spin(0) is NOT
    this case -- the instrument resolved its extra 0.057us/step of Python call
    overhead against a 0.012us floor and correctly said SYNC_TAX. That is the
    instrument working, and it is why this pin uses the empty loop itself."""
    same = sd.NULL._replace(name="a_workload_that_is_really_the_empty_loop")
    d = sd.diagnose(same, k=K, reps=32, warmup=2)
    assert d["verdict"] in ("WORK_NOT_VISIBLE", "NO_SIGNIFICANT_SYNC_TAX",
                            "MEASURING_ITSELF"), d


def test_the_verdict_rule_itself():
    """Pure, so the rule is pinned without a clock in the loop."""
    floors = {"tax_mde_us": 10, "work_mde_us": 10, "excess_mde_us": 10}
    assert sd.verdict(900, 1, 100, **floors) == "SYNC_TAX"
    assert sd.verdict(3, 1, 100, **floors) == "NO_SIGNIFICANT_SYNC_TAX"
    assert sd.verdict(900, 895, 100, **floors) == "MEASURING_ITSELF"
    # work invisible outranks everything else, however large the apparent tax
    assert sd.verdict(9000, 1, 2, **floors) == "WORK_NOT_VISIBLE"


def test_a_noisy_comparison_may_not_veto_a_clean_one():
    """THE BUG THE FIRST LIVE MLX RUN FOUND. The floors were pooled with a max(), so
    the synced arm's spread -- the widest in the experiment -- set the floor for the
    WORK check too, and 50.9us/step of visible matmul work was reported
    WORK_NOT_VISIBLE against a 72.2us/step floor imported from a different question."""
    assert sd.verdict(183.973, 0.023, 50.884, tax_mde_us=72.206, work_mde_us=3.0,
                      excess_mde_us=72.206) == "SYNC_TAX"
    # pooling the floors is what produced the wrong answer
    assert sd.verdict(183.973, 0.023, 50.884, tax_mde_us=72.206, work_mde_us=72.206,
                      excess_mde_us=72.206) == "WORK_NOT_VISIBLE"


def test_the_confound_arm_names_WAITING_when_only_waiting_is_stalled():
    d = sd.diagnose(fake(step_us=20, sync_us=1000, submit_us=0), k=K, reps=REPS,
                    warmup=1)
    c = d["confound_check"]
    assert c["status"] == "RAN"
    assert c["follows"] == "wait", c
    assert c["cost_of_K_waits_per_step_us"] > 10 * max(
        1.0, c["cost_of_K_submissions_per_step_us"]), c


def test_the_confound_arm_names_SUBMISSION_when_only_submitting_is_stalled():
    """An instrument that answers 'wait' whatever it is shown is not an instrument.
    Here submitting costs 1000us and waiting is free, and the answer must flip."""
    d = sd.diagnose(fake(step_us=20, sync_us=0, submit_us=1000), k=K, reps=REPS,
                    warmup=1)
    assert d["confound_check"]["follows"] == "submission", d["confound_check"]


def test_a_missing_submit_is_reported_LOUDLY_not_skipped():
    d = sd.diagnose(fake(step_us=20, sync_us=200), k=K, reps=REPS, warmup=1)
    c = d["confound_check"]
    assert c["status"] == "NOT_RUN"
    assert c["follows"] is None
    assert "NOT established" in c["reason"], c


def test_a_step_that_breaks_the_chain_is_refused():
    with pytest.raises(ValueError, match="serial chain"):
        sd.diagnose(fake(step_us=10, chain=False), k=K, reps=REPS, warmup=1)


def test_it_refuses_a_window_or_a_rep_count_that_cannot_carry_a_result():
    with pytest.raises(ValueError, match="window of zero"):
        sd.diagnose(fake(), k=0, reps=REPS)
    with pytest.raises(ValueError, match="detection floor"):
        sd.diagnose(fake(), k=K, reps=3)


def test_the_window_knob_moves_per_step_cost_and_the_sweep_reports_the_gain():
    s = sd.window_sweep(fake(step_us=20, sync_us=500, submit_us=0),
                        ks=(1, 2, 4, 8), reps=6, warmup=1)
    assert [r["k"] for r in s["rows"]] == [1, 2, 4, 8]
    assert s["rows"][-1]["chained_per_step_us"] < s["rows"][0]["chained_per_step_us"]
    assert s["window_gain_k1_over_kmax"] > 2.0, s
    assert s["smallest_k_within_10pct_of_widest"] is not None
    # the one point where ground truth is known by construction: at k=1 the two arms
    # ARE the same program, so the correct wait cost is exactly zero
    z = s["zero_point_check"]
    assert z["ran"] and z["true_wait_cost_per_step_us"] == 0.0
    assert abs(z["read_wait_cost_per_step_us"]) < 50.0, z


def test_the_interleave_survives_a_drifting_machine_where_A_then_B_would_not():
    """WRITTEN BECAUSE IT WAS MISSING. Replacing the interleave with A-then-B -- every
    rep of one arm, then every rep of the next -- left all twelve other pins green, so
    the module's central design decision was unpinned.

    Here every call to step() is more expensive than the last, which is what a heating
    or increasingly contended machine looks like. Under A-then-B the later arm eats all
    of the drift and the reading inflates well past the injected 1000us/sync; paired,
    the two arms sit next to each other in time and the drift mostly cancels."""
    calls = [0]

    def drifting_step(s):
        calls[0] += 1
        sd.spin((20 + 10 * calls[0]) / 1e6)     # 20us, +10us per call
        return s + 1

    w = sd.Workload(name="drifting", primitive="spin", shape=(1,), make=lambda: 0,
                    step=drifting_step, sync=lambda s: sd.spin(1000 / 1e6),
                    submit=lambda s: None)
    d = sd.diagnose(w, k=K, reps=REPS, warmup=1)
    assert 700 <= d["wait_cost_per_step_us"] <= 1150, (
        d["wait_cost_per_step_us"],
        "the reading moved with the machine's drift, not with the injected stall")


def test_the_zero_point_check_says_so_when_it_was_never_run():
    s = sd.window_sweep(fake(step_us=20, sync_us=200, submit_us=0), ks=(4, 8),
                        reps=5, warmup=1)
    assert s["zero_point_check"]["ran"] is False
    assert "never shown a known-zero" in s["zero_point_check"]["why"]

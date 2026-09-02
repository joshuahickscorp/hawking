"""A loop that cannot produce work must stop, not spin.

The run this guards against is real and measured: 330 consecutive iterations
over 4.4 hours, each emitting a byte-identical 1667-character reply every ~48
seconds, `degenerated=True` on every one, `n_accepted=0` on every one. The loop
computed the degeneracy flag and did nothing with it.

The spiral is self-reinforcing, which is why it never escaped on its own: no
work accepted means the kernel does not change, an unchanged kernel rebuilds a
byte-identical pack, and greedy decoding returns the same reply, which accepts
no work. Nothing inside that cycle can perturb itself, so the perturbation has
to come from the loop's own bookkeeping.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.future import hcli_sovereign as sov


def _streak(accepted_per_turn):
    """Replay the loop's own streak bookkeeping over a sequence of turns."""
    unproductive = 0
    trace = []
    for accepted in accepted_per_turn:
        forced_terse = unproductive >= sov.UNPRODUCTIVE_TERSE_AFTER
        stop = unproductive >= sov.UNPRODUCTIVE_STOP_AFTER
        trace.append({"terse": forced_terse, "stop": stop, "streak": unproductive})
        if stop:
            break
        unproductive = 0 if accepted > 0 else unproductive + 1
    return trace


def test_thresholds_are_ordered_and_finite():
    """Terse must be tried BEFORE giving up, or the cheap fix never runs."""
    assert 0 < sov.UNPRODUCTIVE_TERSE_AFTER < sov.UNPRODUCTIVE_STOP_AFTER
    assert sov.UNPRODUCTIVE_STOP_AFTER < 50, "a bound this loose is not a bound"


def test_a_productive_turn_clears_the_streak():
    trace = _streak([0, 0, 1, 0, 0])
    assert trace[2]["streak"] == 2
    assert trace[3]["streak"] == 0, "one accepted item resets the count"
    assert not any(t["stop"] for t in trace)


def test_the_pack_is_shortened_before_the_loop_gives_up():
    trace = _streak([0] * 20)
    first_terse = next(i for i, t in enumerate(trace) if t["terse"])
    first_stop = next(i for i, t in enumerate(trace) if t["stop"])
    assert first_terse < first_stop, "terse must be tried first"
    assert first_terse == sov.UNPRODUCTIVE_TERSE_AFTER


def test_a_loop_that_produces_nothing_stops():
    """The 330-iteration, 4.4-hour case. It must not reach double digits."""
    trace = _streak([0] * 400)
    assert any(t["stop"] for t in trace), "it span forever before this guard"
    assert len(trace) <= sov.UNPRODUCTIVE_STOP_AFTER + 1, len(trace)


def test_a_parsed_reply_that_accepts_nothing_still_counts_as_unproductive():
    """The stuck run parsed cleanly every turn. Parsing is not producing."""
    trace = _streak([0] * 12)
    assert any(t["stop"] for t in trace)


def test_alternating_work_never_trips_the_stop():
    """A slow but real loop must not be killed for being slow."""
    trace = _streak([0, 0, 1] * 30)
    assert not any(t["stop"] for t in trace)
    assert not any(t["terse"] for t in trace)


def test_context_pack_accepts_the_terse_flag_the_guard_sets():
    """The guard is worthless if the escape hatch it reaches for is not real."""
    kernel = {
        "objective": "o", "measured_state": {}, "hypotheses": [], "scars": [],
        "live_hypotheses": [], "iterations": [], "observations": [],
        "tried_params": [], "frontier": [], "executable_work_types": [],
    }
    long_pack = sov.context_pack(kernel)
    short_pack = sov.context_pack(kernel, terse=True)
    assert isinstance(short_pack, str) and short_pack
    assert len(short_pack) <= len(long_pack)


# --- Generalization: GAP 1 (accepted != executed), GAP 2 (frontier must move,
# not just the reply), the deterministic stuck-state detector, and the
# escalation ladder built on top of the same streak. --------------------


def test_escalation_ladder_thresholds_are_ordered_and_finite():
    """Every rung must be tried before the one above it, or the cheaper fix
    never runs before the loop gives up."""
    assert 0 < sov.UNPRODUCTIVE_EMPHASIZE_DELTA_AFTER < sov.UNPRODUCTIVE_TERSE_AFTER
    assert sov.UNPRODUCTIVE_TERSE_AFTER < sov.UNPRODUCTIVE_ROTATE_EVIDENCE_AFTER
    assert (sov.UNPRODUCTIVE_ROTATE_EVIDENCE_AFTER
            < sov.UNPRODUCTIVE_DIAGNOSE_REJECTIONS_AFTER)
    assert sov.UNPRODUCTIVE_DIAGNOSE_REJECTIONS_AFTER < sov.UNPRODUCTIVE_STOP_AFTER
    assert sov.UNPRODUCTIVE_STOP_AFTER < 50, "a bound this loose is not a bound"


def test_escalation_level_never_reports_the_unwired_L5():
    """L5 (escalate provider) has no safe mechanism owned by this file - the
    only lever this loop has over the provider is the resident BODY
    subprocess, which the operator has forbidden this loop from restarting on
    its own. No streak value may map to it."""
    levels = {sov.escalation_level(s) for s in range(0, 40)}
    assert 5 not in levels
    assert levels == {0, 1, 2, 3, 4, 6}


def test_accepted_but_not_run_is_not_progress():
    """GAP 1, live and measured: turns 38/41/43-46/48-50 of the real run all
    logged 'PERTURB {...} -> DID NOT RUN' with n_accepted=1, and the OLD guard
    reset the streak to 0 on every one because it only checked n_accepted."""
    accepted_not_run = [{"type": "PERTURB", "ran": False,
                          "params": {"tensor": "up", "layer": 1},
                          "result": {}}]
    signals = sov.progress_signals(
        prev_it=None, reply="same reply", results=accepted_not_run,
        n_accepted=1, futs={"f": accepted_not_run[0]},
        mission_state_changed=False, frontier_changed=False,
    )
    assert signals["accepted_work"] is True
    assert signals["work_actually_ran"] is False
    assert signals["productive"] is False, (
        "accepted-but-never-executed must NOT reset the streak"
    )


def test_reply_changed_alone_is_not_progress():
    """GAP 2: a model can emit different prose and accomplish nothing."""
    prev_it = {"output_hash": sov._digest("old reply")}
    signals = sov.progress_signals(
        prev_it=prev_it, reply="a brand new reply, never seen before",
        results=[], n_accepted=0, futs={},
        mission_state_changed=False, frontier_changed=False,
    )
    assert signals["reply_changed"] is True
    assert signals["productive"] is False


def test_a_real_state_transition_is_the_only_thing_that_counts():
    base = dict(prev_it=None, reply="x", results=[], n_accepted=0, futs={},
                mission_state_changed=False, frontier_changed=False)
    cases = [
        (dict(results=[{"type": "PERTURB", "ran": True,
                         "result": {"damage": 0.1}}], n_accepted=1), True),
        (dict(mission_state_changed=True), True),
        (dict(frontier_changed=True), True),
        (dict(results=[{"type": "READ_RECEIPT", "ran": True,
                         "result": {"keys": ["a"]}}]), True),
        (dict(results=[{"type": "PERTURB", "ran": True, "result": {}}]), True),
        (dict(futs={"f": object()}), False),  # launched, nothing else moved
    ]
    for kwargs, expect in cases:
        call = dict(base, **kwargs)
        assert sov.progress_signals(**call)["productive"] is expect, kwargs


def test_deterministic_stuck_fires_immediately_not_after_a_streak():
    """The check that would have caught the 4.4-hour incident on turn 2."""
    prev_it = {"pack_fingerprint": "abc", "output_hash": "xyz",
               "validation": {"n_accepted": 0}}
    assert sov.deterministic_stuck(prev_it, "abc", "xyz", 0) is True
    assert sov.deterministic_stuck(prev_it, "different", "xyz", 0) is False
    assert sov.deterministic_stuck(prev_it, "abc", "different", 0) is False
    assert sov.deterministic_stuck(prev_it, "abc", "xyz", 1) is False, (
        "work accepted this turn - not stuck"
    )
    prev_accepted = dict(prev_it, validation={"n_accepted": 1})
    assert sov.deterministic_stuck(prev_accepted, "abc", "xyz", 0) is False, (
        "previous turn accepted work - this is not a REPEAT of a stuck state"
    )
    assert sov.deterministic_stuck(None, "abc", "xyz", 0) is False, (
        "no prior turn to compare against"
    )


def test_pack_fingerprint_ignores_only_the_turn_counter():
    """context_pack embeds 'TURN N.' specifically so consecutive packs are
    never byte-identical under greedy decoding. Two packs differing ONLY
    there are still the same reasoning input to the resident."""
    p1 = "some pack text TURN 5. more text"
    p2 = "some pack text TURN 6. more text"
    assert sov._pack_fingerprint(p1) == sov._pack_fingerprint(p2)
    p3 = "some OTHER pack text TURN 6. more text"
    assert sov._pack_fingerprint(p1) != sov._pack_fingerprint(p3)


def test_zero_accept_surfaces_the_rejection_reason_not_a_silent_line():
    """Zero-accept is a first-class signal, never a silent success."""
    rejected = [{"work": {"type": "PERTURB"}, "why": "layer or fraction out of range"}]
    assert sov._results_summary([], [], rejected) == [
        "REJECTED: layer or fraction out of range"
    ]
    assert sov._results_summary([], [], []) == [
        "no work was accepted from that turn: the resident selected none"
    ]


def test_context_pack_wires_the_ladders_L1_and_L4_escape_hatches():
    """Same law as the terse test above: the ladder's rungs are worthless if
    the pack content they reach for is not real."""
    kernel = {
        "objective": "o", "measured_state": {}, "hypotheses": [], "scars": [],
        "live_hypotheses": [], "iterations": [], "observations": [],
        "tried_params": [], "frontier": [], "executable_work_types": [],
    }
    plain = sov.context_pack(kernel)
    focused = sov.context_pack(kernel, emphasize_delta=True)
    diagnosed = sov.context_pack(kernel, rejection_digest="WHY: x (x3)")
    assert focused != plain and "FOCUS" in focused
    assert diagnosed != plain and "WHY: x (x3)" in diagnosed


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all green")


# --- "tried" must mean RAN, not "accepted" ---------------------------------
# A verifier refuted the first generalization on exactly this: tried_params is
# in _MISSION_STATE_KEYS, so appending to it flips mission_state_changed, which
# alone satisfies the productive OR-chain and resets the streak. The loop
# appended a param for EVERY result, including ones whose result was
# "DID NOT RUN". So a loop accepting work it never executed still looked
# productive - GAP 1 reintroduced one level up.


def _tried_after(results):
    """Replay the kernel's tried_params bookkeeping over one turn's results."""
    kernel = {}
    for r in results:
        if not r.get("ran"):
            continue
        pp = r.get("params") or {}
        if isinstance(pp, dict) and pp:
            kernel.setdefault("tried_params", []).append(
                "%s/L%s/%s/%s" % (pp.get("tensor"), pp.get("layer"),
                                  pp.get("side"), pp.get("fraction"))
            )
    return kernel.get("tried_params", [])


def test_a_param_that_did_not_run_is_not_recorded_as_tried():
    """The exact live shape: PERTURB accepted, result DID NOT RUN."""
    results = [{"ran": False, "params": {"tensor": "up", "layer": 1,
                                         "side": "rows", "fraction": 0.1}}]
    assert _tried_after(results) == [], (
        "a perturbation that never executed was recorded as tried, which both "
        "lies to the context pack and resets the no-progress streak"
    )


def test_a_param_that_ran_is_recorded():
    results = [{"ran": True, "params": {"tensor": "gate", "layer": 0,
                                        "side": "rows", "fraction": 0.5}}]
    assert _tried_after(results) == ["gate/L0/rows/0.5"]


def test_a_mixed_turn_records_only_the_executed_half():
    results = [
        {"ran": False, "params": {"tensor": "up", "layer": 1, "side": "rows", "fraction": 0.1}},
        {"ran": True, "params": {"tensor": "gate", "layer": 0, "side": "rows", "fraction": 0.5}},
    ]
    assert _tried_after(results) == ["gate/L0/rows/0.5"]


def test_tried_params_is_still_a_mission_state_key():
    """The fix is at the producer, not by removing the signal. If someone drops
    tried_params from the state keys instead, real progress stops counting."""
    assert "tried_params" in sov._MISSION_STATE_KEYS


def test_the_source_actually_guards_on_ran():
    """The helper above mirrors the loop; this pins the loop itself, so the
    mirror cannot drift away from the code it claims to represent."""
    import inspect

    source = inspect.getsource(sov.run)
    assert 'if not r.get("ran"):' in source, (
        "the tried_params append no longer skips results that did not run"
    )

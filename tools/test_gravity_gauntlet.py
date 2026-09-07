"""Tests for tools/gravity_allocator.py:gauntlet -- the target-seeking search
that replaces odyssey_patient_runner's one-sample --gravity build.

Scope note: this worktree has no BF16 tensors / activation-capture on disk
(workspace/campaign/records/runs/... does not exist here), so the real
magnitude-aware adequacy check (tools/gravity_doctor_gate.py:axes/gate,
already fixed for the cosine-scale-invariance scar -- see
tools/test_gravity_adequacy_magnitude.py) cannot be exercised end-to-end in
this lane. What IS real and exercised here: the EBPW measurement path (the
sealed MIX_REPORT via tools/future/complete_ebpw.py), the PROVEN_UNABLE
bound (independently reproduces complete_ebpw.bar_reachability's own
floor), and the control flow (three distinct outcomes, localize-driven
proposals, adequacy-gated TARGET_HIT). The localize-steering and
verified-adequacy tests below use injected synthetic functions -- labelled
as such -- to prove the WIRING, not to claim any specimen result.
"""
from __future__ import annotations

import pytest

from tools import gravity_allocator as ga
from tools.future import complete_ebpw as ce


# --------------------------------------------------------------------------
# PROVEN_UNABLE: the real specimen, the real bound.
# --------------------------------------------------------------------------

def test_gauntlet_proven_unable_matches_bar_reachability_floor():
    """On the real sealed incumbent, sub-1 EBPW is not reachable by MLP
    density alone -- complete_ebpw.py's own auditor (bar_reachability)
    already knows this. gauntlet() must reach the same conclusion, from the
    same real bytes, without spending a single attempt."""
    br = ce.bar_reachability(1.0)
    assert br["reachable_by_mlp_density_alone"] is False, (
        "if this specimen's own bound flips reachable, the fixture this "
        "test depends on changed -- re-derive, don't just widen the assert"
    )
    golden_floor = br["with_the_entire_mlp_at_zero_bytes"]["complete_ebpw"]

    r = ga.gauntlet(target=1.0, max_attempts=8)
    assert r["outcome"] == ga.PROVEN_UNABLE
    assert r["is_pass"] is False
    assert r["attempts"] == 0, "the bound must fire before any candidate is measured"
    assert r["bound"]["floor_ebpw"] == pytest.approx(golden_floor, abs=1e-6)


def test_floor_bound_is_load_bearing():
    """Mutation check: a candidate ladder whose densest rung is still above
    the floor must not be mistaken for reachability. Directly assert the
    floor exceeds target even at the ladder's cheapest rung (0.5 bpw MLP),
    so skipping the PROVEN_UNABLE short-circuit would just waste six
    measurements arriving at BUDGET_EXHAUSTED instead -- never a pass."""
    floor = ga.non_searchable_floor_bpw()
    cheapest_rung_candidate = ga.mlp_mix_candidate(min(ga.MLP_BPW_LADDER))
    cheapest_ebpw = ce.cost(cheapest_rung_candidate)["complete_ebpw"]
    assert floor > 1.0
    assert cheapest_ebpw > 1.0, (
        "even the densest MLP rung this ladder can express stays above the "
        "target -- PROVEN_UNABLE via the floor is the honest outcome, not "
        "a coincidence of attempt count"
    )


# --------------------------------------------------------------------------
# Measurement wiring: cross-check against the existing sealed instrument.
# --------------------------------------------------------------------------

def test_gauntlet_measurement_matches_bar_reachability_sensitivity_table():
    """gauntlet's default measure_fn bills each ladder rung through
    complete_ebpw.cost() on a real rebilled candidate; bar_reachability
    computes the same rung through its own simplified formula. They should
    agree to within the ~58KB of per-tensor header bytes bar_reachability's
    formula does not model (a few 1e-5 bpw) -- not exactly, which would
    suggest one of them is silently reusing the other's number."""
    br = ce.bar_reachability(2.5)
    golden = {row["mlp_bpw"]: row["complete_ebpw"] for row in br["mlp_density_sensitivity"]}

    r = ga.gauntlet(target=2.5, max_attempts=8)
    assert r["outcome"] == ga.BUDGET_EXHAUSTED  # no adequacy_fn -> never a pass
    assert r["is_pass"] is False
    assert len(r["history"]) == len(ga.MLP_BPW_LADDER)
    for row in r["history"]:
        expected = golden[row["proposal"]]
        assert row["ebpw"] == pytest.approx(expected, abs=5e-3), row


# --------------------------------------------------------------------------
# TARGET_HIT requires POSITIVELY verified adequacy -- never a silent pass.
# --------------------------------------------------------------------------

def test_gauntlet_unverified_adequacy_never_yields_target_hit():
    """Several ladder rungs measure under a loose target (2.5), but with no
    adequacy_fn supplied, adequacy is UNKNOWN (healthy=None) on every one.
    UNKNOWN must never be read as a pass."""
    r = ga.gauntlet(target=2.5, max_attempts=8)
    assert r["outcome"] == ga.BUDGET_EXHAUSTED
    assert r["is_pass"] is False
    under = [h for h in r["history"] if h["under_target"]]
    assert under, "fixture is meaningless if nothing ever measured under target"
    assert all(h["adequacy"]["healthy"] is None for h in under)


def test_gauntlet_hits_target_only_with_verified_adequacy():
    """Positive control: supplying an adequacy_fn that actually returns
    healthy=True lets the same real ladder/measurement reach TARGET_HIT.
    Proves the gate in the previous test is a real gate, not a check that
    only ever fails (which would prove nothing)."""
    def adequacy_verified_healthy(candidate):
        return {"healthy": True, "evidence": "SYNTHETIC (test double)"}

    r = ga.gauntlet(target=2.5, max_attempts=8, adequacy_fn=adequacy_verified_healthy)
    assert r["outcome"] == ga.TARGET_HIT
    assert r["is_pass"] is True
    assert r["ebpw"] <= 2.5
    assert r["attempts"] == 5  # first rung (1.0 bpw MLP) that measures under 2.5


# --------------------------------------------------------------------------
# localize_gravity_failure steers the next proposal -- not blind sampling.
# --------------------------------------------------------------------------

def test_gauntlet_calls_localize_gravity_failure_on_every_miss():
    """Every non-hitting attempt must produce a localization record (using
    the SAME function odyssey_ctl uses for capability failures), even when
    per_organ_sensitivity is unavailable -- in which case it is honestly
    None/most_likely_component='unknown', never fabricated."""
    r = ga.gauntlet(target=2.5, max_attempts=3)
    assert r["outcome"] == ga.BUDGET_EXHAUSTED
    for h in r["history"]:
        assert "localization" in h
        # delta_hits was never supplied (UNKNOWN in this lane) -> no verdict
        assert h["localization"] is None


def test_gauntlet_localization_actually_changes_the_next_proposal():
    """Inject real delta_hits + per_organ_sensitivity (synthetic values,
    labelled as such) and a propose_fn that reads localize_gravity_failure's
    most_likely_component. Two different sensitivity maps must steer the
    SAME miss toward two DIFFERENT second proposals -- proof the loop
    consumes localization rather than sampling a fixed sequence regardless
    of it."""
    from tools.odyssey_ctl import localize_gravity_failure

    def make_propose(sensitivity):
        seen = []

        def propose(attempt, history, localization):
            if attempt == 1:
                seen.append("gate_proj")   # first probe, arbitrary
                return 2.5
            # steered: protect (skip past) whatever localization flagged,
            # by returning a value distinguishable per branch
            comp = (localization or {}).get("most_likely_component")
            seen.append(comp)
            return 1.5 if comp == "gate_proj" else 1.0

        propose.seen = seen
        return propose

    sens_gate_worst = {
        "gate_proj": {"round8": {"delta_hits": -12}},
        "down_proj": {"round8": {"delta_hits": -1}},
    }
    sens_down_worst = {
        "gate_proj": {"round8": {"delta_hits": -1}},
        "down_proj": {"round8": {"delta_hits": -12}},
    }

    def measure_stub(_candidate):
        return 5.0  # never under target -> forces a second attempt

    def localize_with_fake_delta_hits(_delta_hits, per_organ_sensitivity, threshold):
        # the real function, called with a REAL (if synthetic) delta_hits
        # signal, so most_likely_component is genuinely derived, not chosen.
        return localize_gravity_failure(-3, per_organ_sensitivity, threshold)

    p1 = make_propose(sens_gate_worst)
    r1 = ga.gauntlet(
        target=1.0, max_attempts=2, propose_fn=p1, measure_fn=measure_stub,
        localize_fn=localize_with_fake_delta_hits,
        per_organ_sensitivity=sens_gate_worst,
    )
    p2 = make_propose(sens_down_worst)
    r2 = ga.gauntlet(
        target=1.0, max_attempts=2, propose_fn=p2, measure_fn=measure_stub,
        localize_fn=localize_with_fake_delta_hits,
        per_organ_sensitivity=sens_down_worst,
    )

    assert p1.seen[1] == "gate_proj", "worse-sensitivity organ must be localized"
    assert p2.seen[1] == "down_proj"
    assert p1.seen[1] != p2.seen[1], (
        "same miss, different sensitivity data, must steer to a different "
        "next proposal -- otherwise localization is being ignored"
    )
    assert r1["history"][1]["proposal"] != r2["history"][1]["proposal"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

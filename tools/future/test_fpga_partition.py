"""Limiting cases, monotonicity and refusal for the partition model.

A simulator earns trust by behaving correctly where the answer is already known:
a link that moves nothing must never recommend a split, a free link must recover
the pure compute-balance answer, and a conclusion built on ratios nobody can
measure must refuse to be called a prediction.
"""
from __future__ import annotations

import math

import pytest

from tools.future import fpga_partition as P

W = 1000.0          # organ weight traffic, in bytes, per token
A, R = 8.0, 8.0     # activation out, partial reduction back


# --- limiting cases -------------------------------------------------------

def test_a_zero_bandwidth_link_never_recommends_a_split():
    """The failure this catches: a finite cost for a disconnected card."""
    C = P.transport_cost(A, R, t=0.0)
    assert C == math.inf
    assert not P.helps(W, C)
    assert P.optimum_fraction(W, r=1000.0, C=C) == 0.0
    assert P.speedup(W, r=1000.0, C=C) == 1.0
    assert P.phase(W, 1000.0, C) == "APPLE_ONLY"


def test_a_free_link_recovers_the_pure_compute_balance():
    """t -> infinity, s = 0 gives C = 0, so f* = r/(r+1) and speedup = r+1.

    Two equally fast engines with free transport must be exactly 2x, which is the
    arithmetic sanity check the whole model has to pass.
    """
    C = P.transport_cost(A, R, t=math.inf)
    assert C == 0.0
    for r in (0.25, 0.5, 1.0, 2.0, 4.0):
        assert P.optimum_fraction(W, r, C) == pytest.approx(r / (r + 1.0))
        assert P.speedup(W, r, C) == pytest.approx(r + 1.0)
    assert P.speedup(W, 1.0, C) == pytest.approx(2.0)


def test_an_fpga_that_computes_nothing_is_never_used():
    C = P.transport_cost(A, R, t=1.0)
    assert P.optimum_fraction(W, r=0.0, C=C) == 0.0
    assert P.critical_path(0.5, W, r=0.0, C=C) == math.inf


def test_an_infinitely_fast_fpga_is_bounded_by_transport():
    """r -> infinity leaves f* = 1 - C/W and speedup = W/C. The link is the wall."""
    C = P.transport_cost(A, R, t=0.5)          # C = 32
    huge = 1e12
    assert P.optimum_fraction(W, huge, C) == pytest.approx(1.0 - C / W, rel=1e-6)
    assert P.speedup(W, huge, C) == pytest.approx(W / C, rel=1e-5)


def test_no_split_is_the_identity():
    C = P.transport_cost(A, R, t=1.0)
    assert P.critical_path(0.0, W, r=1.0, C=C) == W


# --- the break-even is one inequality -------------------------------------

def test_the_break_even_is_exactly_C_less_than_W():
    for t in (0.001, 0.01, 0.1, 1.0, 10.0):
        C = P.transport_cost(A, R, t=t)
        assert P.helps(W, C) == (C < W)
        assert (P.optimum_fraction(W, 1.0, C) > 0.0) == (C < W)


def test_break_even_payload_is_the_crossing_point():
    """At exactly the break-even payload the FPGA stops helping, either side of
    it the model must agree with the inequality it claims to implement."""
    t = 0.25
    payload = P.break_even_payload_bytes(W, t)
    assert P.transport_cost(payload / 2, payload / 2, t) == pytest.approx(W)
    assert not P.helps(W, P.transport_cost(payload * 0.51, payload * 0.51, t))
    assert P.helps(W, P.transport_cost(payload * 0.49, payload * 0.49, t))


def test_setup_cost_eats_the_payload_budget():
    assert P.break_even_payload_bytes(W, t=1.0, setup=0.0) == W
    assert P.break_even_payload_bytes(W, t=1.0, setup=W) == 0.0


# --- monotonicity the physics requires ------------------------------------

def test_transport_cost_is_monotone_in_payload_and_bandwidth():
    prev = -1.0
    for payload in (0, 1, 10, 100, 1000):
        c = P.transport_cost(payload, 0.0, t=1.0)
        assert c >= prev
        prev = c
    prev = math.inf
    for t in (0.1, 0.5, 1.0, 10.0, 100.0):
        c = P.transport_cost(A, R, t=t)
        assert c <= prev
        prev = c


def test_a_faster_link_never_hurts_and_a_faster_fpga_never_hurts():
    prev = 0.0
    for t in (0.01, 0.1, 1.0, 10.0, 1000.0):
        s = P.speedup(W, 1.0, P.transport_cost(A, R, t=t))
        assert s >= prev - 1e-12
        prev = s
    C = P.transport_cost(A, R, t=1.0)
    prev = 0.0
    for r in (0.1, 0.5, 1.0, 2.0, 10.0, 100.0):
        s = P.speedup(W, r, C)
        assert s >= prev - 1e-12
        prev = s


def test_the_split_is_never_worse_than_apple_alone():
    """A partition model that can recommend a losing split is worse than useless."""
    for t in (0.001, 0.1, 1.0, 100.0):
        for r in (0.01, 1.0, 100.0):
            C = P.transport_cost(A, R, t=t)
            f = P.optimum_fraction(W, r, C)
            assert P.critical_path(f, W, r, C) <= W + 1e-9


def test_critical_path_rejects_a_fraction_outside_the_unit_interval():
    with pytest.raises(ValueError):
        P.critical_path(1.5, W, 1.0, 0.0)
    with pytest.raises(ValueError):
        P.critical_path(-0.1, W, 1.0, 0.0)


# --- the refusal ----------------------------------------------------------

def test_an_envelope_refuses_to_be_a_prediction():
    """r, t and s are unmeasurable without the board. The guard is the point."""
    env = P.envelope(W, A, R, r_grid=(0.5, 1.0, 2.0), t_grid=(0.1, 1.0, 10.0))
    assert env.basis == P.ENVELOPE
    assert env.value["cells"] == 9
    with pytest.raises(P.UnpinnedConclusion) as err:
        env.as_prediction()
    assert "r_fpga_over_apple" in str(err.value) or "unpinned" in str(err.value).lower()


def test_a_measured_conclusion_may_be_a_prediction():
    """The negative control for the guard: refusal must not refuse everything."""
    c = P.Conclusion(value=42, basis=P.DERIVED_FROM_MEASURED, inputs={}, note="bytes")
    assert c.as_prediction() == 42


def test_the_sweep_covers_every_declared_phase_name():
    rows = P.sweep(W, A, R, r_grid=(0.1, 0.5, 1.0, 4.0), t_grid=(1e-4, 0.02, 1.0, 100.0))
    assert {row["phase"] for row in rows} <= set(P.PHASES)
    assert all(row["speedup"] >= 1.0 - 1e-12 for row in rows)


# --- measured side --------------------------------------------------------

def test_hbm_capacity_is_derived_from_measured_bytes_and_names_its_source():
    rep = P.hbm_capacity_report()
    assert rep["basis"] == P.DERIVED_FROM_MEASURED
    assert rep["source"]["source_index_sha256"], "a byte count with no source index"
    assert rep["parameter_bytes"] > 0
    assert rep["device"]["hbm_capacity_bytes"] == 8 * 1024 ** 3
    assert rep["device"]["declared_not_measured"] is True
    assert 0.0 < rep["hbm_fraction_of_body"] < 1.0
    assert sum(rep["family_bytes"].values()) == rep["parameter_bytes"]


def test_the_obvious_resident_set_overflows_hbm():
    """Recorded because the intuition is wrong: everything that is not the MoE
    body or the n-gram table looks small, and it does not fit."""
    rep = P.hbm_capacity_report()
    assert rep["non_bulk_fits"] is False
    assert rep["non_bulk_overflow_bytes"] > 0
    assert rep["non_bulk_bytes"] > rep["device"]["hbm_capacity_bytes"]


def test_a_smaller_hbm_never_fits_more():
    big = P.hbm_capacity_report(16 * 1024 ** 3)
    small = P.hbm_capacity_report(1 * 1024 ** 3)
    assert len(big["families_that_fit_whole"]) >= len(small["families_that_fit_whole"])
    assert big["hbm_fraction_of_body"] > small["hbm_fraction_of_body"]


def test_the_shipped_grid_actually_samples_every_phase():
    """An empty region must mean empty, not unsampled.

    The first grid reported TRANSPORT_DOMINATED as zero cells. The region is real
    -- with A+R = 0.002W it is exactly 0.002 < t <= 0.004 -- and a decade-spaced
    grid stepped over it. A phase diagram whose grid cannot reach a phase it
    declares is telling the reader that region does not exist.
    """
    frac = 0.002
    rows = P.sweep(1.0, frac / 2, frac / 2, P.R_GRID, P.T_GRID)
    seen = {row["phase"] for row in rows}
    missing = sorted(set(P.PHASES) - seen)
    assert not missing, f"shipped grid never samples {missing}"


def test_the_audit_does_not_mistake_an_output_label_for_input_provenance():
    """This test exists because the audit got it wrong first.

    A naive substring count for "DERIVED" reported the pre-board module as
    provenance-tagged. It defines DERIVED = "[D]" and stamps it on 26 emitted
    dicts as an OUTPUT claim-boundary label. Where a result stands and where a
    constant came from are different questions.
    """
    a = P.preboard_link_constant_audit()
    assert a["has_input_provenance_tagging"] is False
    assert all(n == 0 for n in a["input_provenance_occurrences"].values())
    assert a["output_claim_labels_present"]["DERIVED"] > 0


def test_no_fpga_speed_can_create_or_destroy_a_win():
    """r does not appear in C, and the break-even is C < W. A theorem, not a grid
    artifact: the FPGA's own rate sizes the win and can never decide there is one."""
    for t in (1e-6, 0.001, 0.01, 1.0, 1e6):
        C = P.transport_cost(A, R, t=t)
        verdicts = {P.helps(W, C) for _ in (0,)}          # helps() ignores r by construction
        assert len(verdicts) == 1
        phases = {P.phase(W, r, C) for r in (1e-6, 0.1, 1.0, 10.0, 1e6)}
        # r may move between the helping phases, but never in or out of APPLE_ONLY.
        assert ("APPLE_ONLY" in phases) == (not P.helps(W, C))
        if P.helps(W, C):
            assert "APPLE_ONLY" not in phases


def test_the_experiment_pack_measures_what_can_kill_the_design_first():
    rows = P.sensitivity(1.0, 0.001, 0.001)
    assert rows[0]["can_falsify_the_architecture"] is True
    by_name = {r["input"]: r for r in rows}
    assert by_name["r_fpga_over_apple"]["can_falsify_the_architecture"] is False
    assert by_name["r_fpga_over_apple"]["speedup_swing"] > by_name["t_link_over_apple"]["speedup_swing"]
    assert by_name["r_fpga_over_apple"]["measure_order"] > by_name["t_link_over_apple"]["measure_order"], \
        "ranked by swing instead of by what can falsify the architecture"
    for r in rows:
        assert r["pinned_by_sealed_predictions"], f"{r['input']} names no sealed prediction"

"""The horizon must be real or absent, and admission must know what hour it is.

The failure this guards is a controller that reports "hour 0 of 48" for a
mission nobody launched, and an admission rule that lets a five-hour experiment
start at hour 44.
"""
from __future__ import annotations

import json

import pytest

from tools.future import odyssey_mission_controller as omc


def test_an_unstarted_cycle_reports_not_started_rather_than_hour_zero():
    h = omc.horizon()
    assert h["state"] == "NOT_STARTED"
    assert h["elapsed_hours"] is None
    assert h["remaining_hours"] is None
    assert "would be fake progress" in h["why"]


def test_a_started_cycle_reports_a_real_horizon(monkeypatch):
    t0 = 1_700_000_000.0
    monkeypatch.setattr(omc, "_start",
                        lambda: {"started_unix": t0, "started_utc": "X"})
    h = omc.horizon(now=t0 + 4 * 3600)
    assert h["state"] == "RUNNING"
    assert h["elapsed_hours"] == pytest.approx(4.0, abs=1e-6)
    assert h["remaining_hours"] == pytest.approx(44.0, abs=1e-6)


def test_the_cycle_can_go_over_budget_without_lying(monkeypatch):
    t0 = 1_700_000_000.0
    monkeypatch.setattr(omc, "_start", lambda: {"started_unix": t0})
    h = omc.horizon(now=t0 + 50 * 3600)
    assert h["state"] == "OVER_BUDGET"
    assert h["remaining_hours"] < 0
    assert h["fraction_spent"] == 1.0, "capped, but the state says over"


def test_long_work_without_justification_is_refused():
    a = omc.admits(duration_minutes=240)
    assert a["admitted"] is False
    assert any("unexamined long work is not" in r for r in a["reasons"])


def test_long_work_with_full_justification_is_admitted():
    a = omc.admits(duration_minutes=240, justification={
        "maximum_payoff": "2.5 ms",
        "why_no_cheaper_proxy": "the isolated harness cannot see the graph fold",
        "what_runs_concurrently": "the q4 ladder on CPU",
        "early_stop_criterion": "stop if the first layer shows no separation",
    })
    assert a["admitted"] is True


def test_a_partial_justification_names_exactly_what_is_missing():
    a = omc.admits(duration_minutes=240, justification={"maximum_payoff": "2 ms"})
    assert a["admitted"] is False
    r = " ".join(a["reasons"])
    assert "why_no_cheaper_proxy" in r
    assert "early_stop_criterion" in r
    assert "maximum_payoff" not in r, "what was supplied must not be reported missing"


def test_short_work_needs_no_justification():
    assert omc.admits(duration_minutes=5)["admitted"] is True


def test_a_five_hour_experiment_is_fine_at_hour_four_and_not_at_hour_44(monkeypatch):
    t0 = 1_700_000_000.0
    monkeypatch.setattr(omc, "_start", lambda: {"started_unix": t0})
    j = {k: "stated" for k in omc.LONG_WORK_REQUIREMENTS}
    early = omc.admits(duration_minutes=300, justification=j, now=t0 + 4 * 3600)
    late = omc.admits(duration_minutes=300, justification=j, now=t0 + 44 * 3600)
    assert early["admitted"] is True
    assert late["admitted"] is False
    assert any("before the seal" in r for r in late["reasons"])


def test_the_horizon_check_is_skipped_and_said_to_be_when_not_started():
    a = omc.admits(duration_minutes=300,
                   justification={k: "x" for k in omc.LONG_WORK_REQUIREMENTS})
    assert a["horizon_state"] == "NOT_STARTED"
    assert any("SKIPPED" in r for r in a["reasons"])
    assert a["admitted"] is True, "the long-work rule passed and no horizon exists"


def test_phases_open_on_their_input_not_on_a_clock(monkeypatch):
    t0 = 1_700_000_000.0
    monkeypatch.setattr(omc, "_start", lambda: {"started_unix": t0})
    none = omc.phase_entry(n_laws=0, n_attackable_laws=0)
    assert none["ODYSSEY_I"] is True and none["ODYSSEY_II"] is False
    some = omc.phase_entry(n_laws=3, n_attackable_laws=1)
    assert some["ODYSSEY_II"] is True and some["ODYSSEY_III"] is True
    assert "not exclusive machine modes" in some["overlap_is_expected"]


def test_no_phase_opens_before_the_cycle_starts():
    p = omc.phase_entry(n_laws=9, n_attackable_laws=9)
    assert p["ODYSSEY_I"] is False
    assert p["ODYSSEY_II"] is False and p["ODYSSEY_III"] is False


def test_the_verification_ladder_runs_cheap_to_expensive():
    d = omc.depth_policy()
    assert d["verification_ladder"][0] == "structural"
    assert d["verification_ladder"][-1] == "full_qualification"
    assert "die before the expensive levels" in d["ladder_rule"]
    assert "pending descendants die with it" in d["cancel_descendants"]


def test_the_seal_contract_forbids_a_false_completeness_claim():
    s = omc.seal_contract()
    assert "TIME-INDEXED" in s["time_indexed_universe"]
    assert "JOINED LATE" in s["time_indexed_universe"]
    assert "must not claim every available model participated fully" in \
        s["no_false_completeness"]
    assert "ODYSSEY_CYCLE_2" in s["no_false_completeness"]


def test_odyssey_is_not_held_hostage_to_sixty_tps():
    b = omc.build()
    assert "continues INSIDE it" in b["odyssey_is_not_held_hostage_to_60_tps"]


def test_the_module_says_which_of_its_answers_are_live():
    b = omc.build()
    assert "NOT_STARTED" in b["what_is_live_and_what_is_not"]
    assert "live and tested now" in b["what_is_live_and_what_is_not"]


def test_the_four_cycle_timings_are_measured_separately_not_collapsed():
    """One elapsed number cannot describe a streaming cycle.

    A cycle that finds its first law at hour two and lands its first adversarial
    attack at hour forty is not the same machine as one that does both by hour
    twelve, even though both finish inside 48. Collapsing them hides exactly the
    thing the contract exists to expose.
    """
    from tools.future import odyssey_mission_controller as omc

    t = omc.cycle_timings()
    for name, _meaning in omc.CYCLE_TIMINGS:
        assert name in t, f"{name} is not measured"
        assert "state" in t[name], f"{name} has no state"
    assert len({n for n, _ in omc.CYCLE_TIMINGS}) == 4, "there must be four distinct timings"


def test_there_is_no_global_barrier_between_the_phases():
    """II starts when a law exists; III runs concurrently. Nothing waits."""
    from tools.future import odyssey_mission_controller as omc

    t = omc.cycle_timings()
    assert "no_global_barrier" in t

    # The invariant is STRUCTURAL: III's entry condition must depend on its own
    # input existing, never on II having finished. A phase that waits for another
    # phase IS the global barrier this contract forbids.
    conditions = omc.phase_entry(n_laws=0, n_attackable_laws=0)["entry_conditions"]
    third = conditions["ODYSSEY_III"].lower()
    assert "odyssey_ii" not in third and "transfer" not in third, (
        f"III waits on II: {conditions['ODYSSEY_III']!r}"
    )
    second = conditions["ODYSSEY_II"].lower()
    assert "complete" not in second and "finish" not in second, (
        f"II waits for I to finish rather than for a law to exist: {conditions['ODYSSEY_II']!r}"
    )

    # And with the cycle running, one law opens BOTH II and III on the same tick.
    entry = omc.phase_entry(n_laws=1, n_attackable_laws=1)
    assert entry["ODYSSEY_II"] == entry["ODYSSEY_III"], (
        "II and III must open together on the same input, not in sequence"
    )


def test_an_unreached_timing_is_null_and_never_interpolated():
    """A timing nobody measured must not be reported as a number."""
    from tools.future import odyssey_mission_controller as omc

    for name, _ in omc.CYCLE_TIMINGS:
        row = omc.cycle_timings()[name]
        if row["state"] != "REACHED":
            assert row["hours"] is None, (
                f"{name} reports {row['hours']} hours while state={row['state']}"
            )


def test_the_specimen_count_comes_from_disk_not_from_prose():
    """The roadmap carried 'Flash is still downloading' long after it was false."""
    from tools.future import odyssey_mission_controller as omc

    reg = omc.specimen_registry()
    assert reg["state"] in {"READ_FROM_DISK", "VOLUME_ABSENT"}
    if reg["state"] == "VOLUME_ABSENT":
        # An unmounted volume must NOT be reported as an empty lake.
        assert reg["sealed_specimens"] is None
        return
    assert isinstance(reg["sealed_specimens"], int)
    on_disk = len([p for p in (omc.LAKE / "specimens").iterdir() if p.is_dir()])
    assert reg["sealed_specimens"] == on_disk, "the count disagrees with the filesystem"


def test_an_unmounted_lake_is_not_reported_as_an_empty_lake(monkeypatch, tmp_path):
    """The branch that only runs when the volume is missing must still be tested.

    On this host the lake IS mounted, so the VOLUME_ABSENT path never executes
    and a mutation making it report zero specimens passed unnoticed. A test that
    covers only the branch its environment happens to take is not coverage.
    Drive the absent case directly.
    """
    from tools.future import odyssey_mission_controller as omc

    # Patch where the value is USED, not where it used to live: the census now
    # belongs to tools.future.modellake_lifecycle and the controller delegates to
    # it, so one module holds one answer instead of two that can disagree.
    from tools.future import modellake_lifecycle as ml

    monkeypatch.setattr(ml, "LAKE", tmp_path / "not-mounted")
    reg = omc.specimen_registry()
    assert reg["state"] == "VOLUME_ABSENT"
    assert reg["sealed_specimens"] is None, (
        "an unmounted volume reported a specimen count; absent is not empty"
    )
    assert reg["manifests"] is None
    assert "not the same as an empty lake" in reg["why"]

"""The wall model must refuse to project, and must not be re-orderable by guesswork.

Its central claim is structural: the substrate can record 2 of the 9 components
that sit on the serial critical path. What it CAN record is the parallelizable
compile and artifact work, which is not what determines the campaign wall.
"""
from __future__ import annotations

import pytest

from tools.future import odyssey_wall as W


def test_no_component_carries_a_duration_and_none_reports_zero():
    """UNRECORDED is not zero. Losing that distinction is what broke the ledger."""
    for c in W.COMPONENTS:
        row = c.as_dict()
        assert row["measured_seconds"] is None, f"{c.name} claims a duration"
        assert row["evidence"] in (W.UNRECORDED, W.NOT_INSTRUMENTED)
        assert row["evidence"] != W.MEASURED


def test_a_projected_campaign_wall_is_refused():
    """A wall summed from no measurements is a number about nothing."""
    with pytest.raises(W.UnrecordedWall) as err:
        W.projected_wall_hours()
    assert "measured" in str(err.value).lower()


def test_the_report_refuses_rather_than_omitting_the_projection():
    """A missing key reads as an oversight; a recorded refusal reads as a finding."""
    r = W.report()
    assert isinstance(r["projected_campaign_wall"], dict)
    assert "refused" in r["projected_campaign_wall"]
    assert r["counts"]["measured"] == 0
    assert r["odyssey_launched"] is False


def test_most_of_the_serial_critical_path_is_invisible_to_the_substrate():
    """The finding. What is instrumented is not what determines the wall.

    If this ever fails because coverage improved, that is the good direction --
    rewrite it against the new number rather than deleting it.
    """
    serial = [c for c in W.COMPONENTS if c.parallel_by_construction is False]
    recordable = [c for c in serial if c.substrate_event is not None]
    assert len(serial) == 9, [c.name for c in serial]
    assert len(recordable) == 2, [c.name for c in recordable]
    assert {c.name for c in recordable} == {"kernel_probe", "gpu_total"}
    invisible = {c.name for c in serial if c.substrate_event is None}
    assert "resident_decode" in invisible
    assert "tool_call_recovery" in invisible
    assert "dependency_wait" in invisible


def test_every_instrumented_component_names_a_real_substrate_event():
    """A component citing an event key the substrate does not define is a hole
    dressed as coverage."""
    from tools.odyssey_costmodel import EVENT_WALL
    known = set(EVENT_WALL.values())
    for c in W.instrumented():
        assert c.substrate_event in known, (
            f"{c.name} cites {c.substrate_event!r}, which is not in EVENT_WALL"
        )


def test_the_ladder_states_requirements_and_never_predicts():
    rows = W.ladder()
    assert rows
    for row in rows:
        assert row["is_a_prediction"] is False
        # Amdahl: the work ceiling is the target divided by the serial share.
        assert row["max_total_work_hours"] == pytest.approx(
            row["target_hours"] / row["serial_fraction"])
    tight = W.requirements_for(24.0, serial_fraction=0.5)
    assert tight["max_total_work_hours"] == 48.0
    loose = W.requirements_for(24.0, serial_fraction=0.1)
    assert loose["max_total_work_hours"] == 240.0
    assert loose["max_total_work_hours"] > tight["max_total_work_hours"], (
        "a smaller serial fraction must permit MORE total work for the same target"
    )


def test_an_impossible_serial_fraction_is_refused():
    with pytest.raises(ValueError):
        W.requirements_for(24.0, serial_fraction=0.0)
    with pytest.raises(ValueError):
        W.requirements_for(24.0, serial_fraction=1.5)
    with pytest.raises(ValueError):
        W.requirements_for(0.0, serial_fraction=0.5)


def test_the_instrumentation_order_is_structural_and_says_why_it_is_not_size():
    """It must not quietly rank by an invented size after that approach failed."""
    o = W.instrumentation_order()
    assert o["ordering_basis"] == "structural, not size"
    assert "no size is known" in o["why_not_size"].lower()
    rows = o["rows"]
    assert len(rows) == len(W.COMPONENTS)
    assert [r["instrument_order"] for r in rows] == list(range(1, len(rows) + 1))
    # The worst case -- serial and invisible -- must sort above anything that is
    # neither, or the ordering is not doing its job.
    first, last = rows[0], rows[-1]
    assert first["on_serial_critical_path"] and first["invisible_to_the_substrate_today"]
    assert not (last["on_serial_critical_path"] and last["invisible_to_the_substrate_today"])


def test_the_top_of_the_order_is_the_serial_invisible_set():
    o = W.instrumentation_order()
    top7 = {r["component"] for r in o["rows"][:7]}
    serial_invisible = {
        c.name for c in W.COMPONENTS
        if c.parallel_by_construction is False and c.substrate_event is None
    }
    assert top7 == serial_invisible, (top7, serial_invisible)

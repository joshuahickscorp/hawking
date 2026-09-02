"""The delta receipt must be DERIVED, never hand-edited."""

from __future__ import annotations

import json

from tools.roadmap import saturation


def test_receipt_regenerates_identically():
    """A hand-edited score is exactly what this campaign removed. Guard it."""
    assert saturation.RECEIPT.exists(), "delta receipt has never been emitted"
    fresh = saturation.build()
    stored = json.loads(saturation.RECEIPT.read_text())
    for doc in (fresh, stored):
        # derived_from and work are git-derived: they move with every commit by
        # design and can never be stable. What must not drift is the substance
        # the auditor produced -- statuses, blockers, remaining gaps.
        doc.pop("derived_from", None)
        doc.pop("work", None)
    assert fresh == stored, "committed receipt is stale; regenerate, do not hand-edit"


def test_every_status_count_comes_from_the_auditor_graph():
    graph = json.loads(saturation.GRAPH.read_text())
    doc = saturation.build()
    assert doc["final"]["by_status"] == graph["counts"]["gates_by_status"]


def test_hardware_blocked_gates_all_carry_a_wake_condition():
    doc = saturation.build()
    blocked = doc["hardware_blocked"]
    assert blocked, "expected hardware-blocked gates"
    assert all(row["wake_condition"] for row in blocked)


def test_externally_blocked_gates_all_name_their_blocker():
    doc = saturation.build()
    ext = doc["externally_blocked"]
    assert ext, "expected externally-blocked gates"
    assert all(row["blocker"] for row in ext)


def test_movable_percentage_is_computed_not_asserted():
    doc = saturation.build()
    final = doc["final"]
    expected = round(
        100.0 * final["movable_scaffolded_or_better"] / final["movable_gates"], 1
    )
    assert final["movable_scaffolded_or_better_pct"] == expected

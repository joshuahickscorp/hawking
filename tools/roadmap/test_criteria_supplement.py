"""Six gates named after six numbers shared one criterion that mentions none.

The U50 ladder's APPENDIX O span, 8989-8997, is a promotion POLICY -- five
general conditions and a note that a negative result still pays tuition. Sound
policy, and it contains no throughput figure, so nothing in it distinguishes
40 tok/s from 90 tok/s. These tests hold the authored replacements to the one
property the shared span could not have: a rung must be refusable at its own
number and no sibling's.
"""
from __future__ import annotations

import json
import re

import pytest

from tools.roadmap.auditor import REPO

SUPPLEMENT = REPO / "civilization" / "GATE_CRITERIA_SUPPLEMENT.json"
LADDER = ["U50_34_TO_40", "U50_40_TO_50", "U50_50_TO_60",
          "U50_60_TO_70", "U50_70_TO_80", "U50_80_TO_90"]


@pytest.fixture(scope="module")
def gates():
    return json.loads(SUPPLEMENT.read_text())["gates"]


def test_every_entry_obeys_the_supplement_law(gates):
    """An entry is legitimate only when it records why APPENDIX O does not define
    the gate, and when it was authored."""
    for gid, entry in gates.items():
        assert entry.get("span_states"), f"{gid} does not say what its span contains"
        assert entry.get("authored_at_commit"), f"{gid} records no authoring commit"
        assert entry.get("criterion", "").strip(), f"{gid} states no criterion"
        assert entry.get("falsifiers"), f"{gid} names no falsifier; a clause with none is prose"


def test_each_ladder_rung_names_its_own_number_and_not_a_sibling_target(gates):
    """The property the shared span could never have."""
    targets = {gid: int(gid.split("_TO_")[1]) for gid in LADDER}
    for gid in LADDER:
        crit = gates[gid]["criterion"]
        mine = targets[gid]
        assert f"least {mine} tok/s" in crit, f"{gid} does not state its own target"
        others = {t for g, t in targets.items() if g != gid} - {int(gid.split("_")[1])}
        for other in others:
            assert f"least {other} tok/s" not in crit, (
                f"{gid} states a sibling's target {other}; the rungs are not distinguishable"
            )


def test_the_rungs_are_strictly_increasing_and_cover_the_ladder(gates):
    pairs = [(int(g.split("_")[1]), int(g.split("_TO_")[1])) for g in LADDER]
    assert pairs == sorted(pairs), pairs
    for (lo_a, hi_a), (lo_b, hi_b) in zip(pairs, pairs[1:]):
        assert hi_a == lo_b, f"gap in the ladder between {hi_a} and {lo_b}"


def test_no_ladder_rung_can_be_satisfied_by_a_simulation(gates):
    """Directive 5H. A simulation predicting 90 tok/s is not a 90 tok/s result,
    and this repo now contains a simulator that produces exactly such numbers."""
    for gid in LADDER:
        entry = gates[gid]
        assert entry["requires_hardware"] is True
        assert entry["evidence_tier_ceiling"] == "HARDWARE_MEASURED"
        assert entry.get("wake_condition") == "U50_PRESENT"
        forbidden = " ".join(entry["may_not_be_satisfied_by"]).lower()
        assert "simulation" in forbidden and "envelope" in forbidden, gid


def test_every_rung_demands_a_paired_control_and_provenance(gates):
    """An unpaired absolute reading is uninterpretable on a machine recorded to
    inflate decode throughput several-fold in an interactive session."""
    for gid in LADDER:
        crit = gates[gid]["criterion"]
        assert "PAIRED" in crit and "control" in crit, gid
        assert "measurement_provenance" in crit, gid
        assert "ACCEPTED means" in crit, f"{gid} does not distinguish accepted from raw decode"


def test_the_unestablished_baseline_is_recorded_not_assumed(gates):
    """The ladder reads as if ~34 tok/s were the standing reference. The highest
    accepted_tps in receipts/ is 25.29, and it has no provenance."""
    for gid in LADDER:
        note = gates[gid].get("baseline_is_not_established", "")
        assert "25.29" in note, f"{gid} does not record that the ladder floor is unestablished"


def test_the_ladder_gates_are_all_still_blocked_on_hardware():
    """Authoring a criterion must not make a rung reachable."""
    graph = json.loads((REPO / "civilization" / "CAPABILITY_GRAPH.json").read_text())
    for gid in LADDER:
        gate = graph["gates"][gid]
        assert gate["status"] == "BLOCKED_HARDWARE", (gid, gate["status"])
        assert gate["accepted"]["value"] is False, gid

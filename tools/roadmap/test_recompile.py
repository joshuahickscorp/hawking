"""The recompiled roadmap and its machine-readable state must agree.

Two documents that disagree about how much work is left are worse than one,
because a reader trusts whichever is nearer to hand. These tests exist so a
divergence fails loudly instead of being discovered by someone planning against
the wrong number.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools.roadmap import recompile

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs" / "roadmap"
STATE = REPO / "civilization" / "ROADMAP_STATE.json"
LINEAGE = REPO / "docs" / "roadmap-lineage"

PARTS = (
    "PART_I_VERIFIED_TODAY.md",
    "PART_II_ACTION_PLAN.md",
    "PART_III_CONSTITUTION_AND_RESEARCH.md",
    "APPENDIX_LINEAGE.md",
)


def _state() -> dict:
    return json.loads(STATE.read_text())


def test_all_four_parts_exist_and_are_not_empty():
    for name in PARTS:
        p = DOCS / name
        assert p.is_file(), f"{name} missing: the recompilation is not four parts"
        assert len(p.read_text().strip()) > 200, f"{name} is a stub"


def test_part_ii_census_matches_the_state_file():
    """The headline number must be the same in both places."""
    text = (DOCS / "PART_II_ACTION_PLAN.md").read_text()
    state = _state()
    m = re.search(r"SOFTWARE_CONNECTION_REMAINING\s+(\d+)", text)
    assert m, "PART II no longer prints the census"
    assert int(m.group(1)) == state["software_connection_remaining_count"], (
        "PART II and ROADMAP_STATE disagree about software connections remaining"
    )


def test_every_gate_lands_in_exactly_one_bucket():
    """No gate may be counted twice or vanish between buckets."""
    state = _state()
    buckets = (
        "integrated_capabilities", "active_actions", "experiment_required",
        "long_run_required", "hardware_required", "unknown_research",
    )
    seen: list[str] = []
    for b in buckets:
        seen.extend(state[b])
    graph = json.loads((REPO / "civilization" / "CAPABILITY_GRAPH.json").read_text())
    assert sorted(seen) == sorted(graph["gates"]), "a gate is double-counted or missing"
    assert len(seen) == len(set(seen)), "a gate appears in two buckets"


def test_built_is_not_silently_equated_with_verified():
    """The count gap must stay visible, not be reconciled away.

    completed_capabilities (status BUILT) and integrated_capabilities (nothing
    left blocking) differ by exactly the gates that are wired and acceptance
    receipted while NO test cites them. That gap is the only interesting thing
    in this file; a future edit that squares the counts would hide it.
    """
    state = _state()
    gap = len(state["completed_capabilities"]) - len(state["integrated_capabilities"])
    assert gap == len(state["built_but_no_verifier"]), (
        "the BUILT/integrated gap no longer equals built_but_no_verifier"
    )


def test_no_gate_claims_a_physical_measurement_it_does_not_have():
    """simulated != measured, enforced rather than promised."""
    state = _state()
    levels = state["evidence_levels"]
    assert set(levels) == {"STATIC"}, (
        f"a gate claims a non-STATIC evidence tier: {levels}. "
        "HARDWARE_MEASURED requires a board that is not present."
    )


def test_the_preserved_lineage_still_matches_its_recorded_hash():
    """Old authority must remain byte-identical to what was preserved."""
    import hashlib

    record = (LINEAGE / "PRESERVATION.md").read_text()
    m = re.search(r"sha256\s+([0-9a-f]{64})", record)
    assert m, "PRESERVATION.md no longer records a hash"
    copy = LINEAGE / "H-ROADMAP.superseded-2026-09-02.md"
    assert copy.is_file(), "the preserved roadmap copy is gone"
    got = hashlib.sha256(copy.read_bytes()).hexdigest()
    assert got == m.group(1), "the preserved lineage copy was modified"


def test_blocker_class_is_derived_from_evidence_not_assigned():
    """A gate with no caller is a software connection; with one, it is not.

    This is the rule the whole plan rests on, so it is asserted directly rather
    than trusted. Flipping either branch changes the campaign's headline number.
    """
    unwired = {
        "id": "X", "status": "SCAFFOLDED", "code_refs": [{"file": "a.py"}],
        "tests": [{"file": "t.py"}], "wired": {"value": False}, "accepted": {"value": False},
    }
    assert recompile.blocker_class(unwired)[0] == "SOFTWARE_CONNECTION_REMAINING"

    wired_unaccepted = dict(unwired, wired={"value": True})
    assert recompile.blocker_class(wired_unaccepted)[0] == "EXPERIMENTATION_REQUIRED"

    hardware = dict(unwired, status="BLOCKED_HARDWARE", wake_condition="U50_PRESENT")
    assert recompile.blocker_class(hardware)[0] == "PHYSICAL_HARDWARE_REQUIRED"

    done = dict(unwired, wired={"value": True}, accepted={"value": True})
    assert recompile.blocker_class(done)[0] == ""


def test_net_future_burden_means_the_same_thing_in_both_documents():
    """One name, one definition.

    ROADMAP_STATE computed net_future_burden as software+experiment while
    COMPRESSION.md computed software+experiment+long_run: 31 against 41, the
    same phrase carrying two answers. Whichever is chosen, both must use it.
    """
    state = _state()
    expected = (
        state["software_connection_remaining_count"]
        + len(state["experiment_required"])
        + len(state["long_run_required"])
    )
    assert state["net_future_burden"] == expected, "state's own arithmetic disagrees"

    text = (DOCS / "COMPRESSION.md").read_text()
    m = re.search(r"NET FUTURE BURDEN\s+(\d+)", text)
    assert m, "COMPRESSION.md no longer prints a net future burden"
    assert int(m.group(1)) == state["net_future_burden"], (
        "COMPRESSION.md and ROADMAP_STATE disagree about net future burden"
    )

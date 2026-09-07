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
    # DERIVED from the classifier's own bucket map, not retyped. A hardcoded list
    # silently drops a gate the moment a new blocker class appears -- VERIFIER_MISSING
    # was added and its gate vanished from the partition while every bucket still
    # looked internally consistent.
    from tools.roadmap.blockers import CLASSES
    from tools.roadmap.recompile import bucket_names
    buckets = ("integrated_capabilities",) + tuple(bucket_names())
    missing_class = [c for c in CLASSES if c not in bucket_names.map]
    assert not missing_class, f"blocker classes with no state bucket: {missing_class}"
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


def test_the_superseded_lineage_identity_remains_compacted():
    """Historical identity survives without a duplicate active roadmap copy."""
    record = (LINEAGE / "PRESERVATION.md").read_text()
    assert re.search(r"sha256\s+[0-9a-f]{64}", record)
    assert "Git history" in record


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
    # Every non-hardware bucket, so a new blocker class cannot silently vanish
    # from the burden the way DEFERRED_PROGRAM and EXTERNAL_ENVIRONMENT did when
    # the taxonomy expanded underneath a hardcoded sum.
    expected = state["ACTIVE_NONHARDWARE_BURDEN"]
    assert state["net_future_burden"] == expected, "state's own arithmetic disagrees"

    text = (DOCS / "COMPRESSION.md").read_text()
    m = re.search(r"NET FUTURE BURDEN\s+(\d+)", text)
    assert m, "COMPRESSION.md no longer prints a net future burden"
    assert int(m.group(1)) == state["net_future_burden"], (
        "COMPRESSION.md and ROADMAP_STATE disagree about net future burden"
    )


# --------------------------------------------------------------------------- boot ROM

def _canonical_roadmap() -> Path | None:
    """The operator's roadmap. Canonical name is H-ROADMAP.md in ~/Downloads.

    Operator, 2026-09-05: "the roadmap should be h-roadmap.md", and separately
    "it's updated so frequently there isn't one true roadmap except for the one
    I point you to". So: no pinned hash, no verification, one designated name.

    A browser re-download lands as `H-ROADMAP(10).md` rather than overwriting, so
    a suffixed variant NEWER than H-ROADMAP.md is preferred and the caller is told
    -- otherwise a fresh roadmap would be silently ignored in favour of a stale
    canonical copy. That is the failure this whole area already had once: the
    previous constant pointed at H-ROADMAP-REVISED.md, which does not exist, and
    the reader called pytest.skip().
    """
    downloads = Path.home() / "Downloads"
    if not downloads.is_dir():
        return None
    canonical = downloads / "H-ROADMAP.md"
    variants = sorted(
        (f for f in downloads.glob("H-ROADMAP*.md") if f != canonical),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if canonical.is_file():
        if variants and variants[0].stat().st_mtime > canonical.stat().st_mtime:
            return variants[0]
        return canonical
    return variants[0] if variants else None


ROADMAP = _canonical_roadmap()


def _roadmap() -> str:
    # A skip here used to hide the fact that the pinned filename did not exist.
    # If the operator's Downloads holds no roadmap at all, that is worth failing
    # on, not skipping past.
    assert ROADMAP is not None and ROADMAP.is_file(), (
        "no H-ROADMAP*.md in ~/Downloads; the operator designates that directory "
        "as roadmap authority"
    )
    return ROADMAP.read_text()



def test_no_headline_claims_verification_the_evidence_does_not_support():
    """VERIFIED_* must not name a gate with no independent verifier.

    This is the vocabulary defect the axes exist to prevent: a gate that is wired
    and acceptance-receipted but has NO test was previously called
    VERIFIED_INTEGRATED.
    """
    state = _state()
    for gid, status in state["derived_status"].items():
        if state["axes"][gid]["verification_state"] == "NONE":
            assert not status.startswith("VERIFIED"), (
                f"{gid} has no verifier but its status reads {status}"
            )


def test_the_three_headline_counts_cannot_be_mistaken_for_one_another():
    state = _state()
    total = state["TOTAL_UNRESOLVED_GATES"]
    nonhw = state["ACTIVE_NONHARDWARE_BURDEN"]
    soft = state["software_connection_remaining_count"]
    assert soft <= nonhw <= total, f"{soft} <= {nonhw} <= {total} does not hold"
    hardware = state["blocker_census"].get("PHYSICAL_HARDWARE_REQUIRED", 0)
    assert total - hardware == nonhw, "non-hardware burden is not total minus hardware"
    for key in ("TOTAL_UNRESOLVED_GATES", "ACTIVE_NONHARDWARE_BURDEN",
                "SOFTWARE_CONNECTION_REMAINING"):
        assert key in state["counts_explained"], f"{key} is not explained anywhere"


def test_absent_code_is_not_filed_as_unknown_research():
    """UNKNOWN_RESEARCH means the ANSWER is unknown, not that code is unwritten.

    Filing missing implementation as research excuses it from ever being built.
    """
    state = _state()
    census = state["blocker_census"]
    for gid in state.get("unknown_research", []):
        axes = state["axes"][gid]
        assert axes["implementation_state"] != "ABSENT", (
            f"{gid} has no implementation but is filed as unknown research"
        )
    assert census.get("UNKNOWN_RESEARCH", 0) == len(state.get("unknown_research", [])), (
        "the census and the bucket disagree about unknown research"
    )


def test_the_hot_frontier_names_owner_lane_and_verifier_for_every_row():
    """A frontier row nobody can act on is decoration."""
    state = _state()
    for row in state["hot_frontier"]:
        for field in ("gate", "owner", "resource_lane", "verifier", "stop_condition",
                      "depends_on", "unlocks_transitive", "blocker_class"):
            assert field in row, f"{row.get('gate')} is missing {field}"
        assert row["owner"] != "unassigned", f"{row['gate']} has no owner"
        assert row["actionable_now"], "a blocked gate is occupying the hot frontier"




# RETIRED 2026-09-06, operator instruction, fourth of its family. It asserted the
# roadmap contains the literal header "IF YOU ARE CHATGPT / CLAUDE / GROK / HCLI"
# and a fixed "READ IN THIS ORDER" chain. The operator's new canonical
# ~/Downloads/H-ROADMAP.md does not carry that header, and pinning the SHAPE of a
# document the operator rewrites daily is the same assumption already retired with
# the section-order, fingerprint and freshness checks.
#
# The INTENT was good and is preserved as intent: a fresh reader should be oriented,
# and machine state should outrank the document. That belongs in the roadmap itself,
# which the operator owns, not in a test that fails whenever they edit it.

def test_no_gate_is_wired_solely_by_its_own_module_calling_itself():
    """A module using its own function is not evidence anything reaches it.

    The auditor accepts any non-test call of the implementing symbol, including
    one made INSIDE the implementing module. Taken alone that is "registration is
    not wiring" in a new costume: a producer that calls its own helper would read
    WIRED with nothing in the production path reaching it.

    Swept at the time of writing: zero gates depended on a self-call, so nothing
    is grandfathered here. This keeps it that way, because the auditor cannot
    currently tell an internal caller from an external one.
    """
    graph = json.loads((REPO / "civilization" / "CAPABILITY_GRAPH.json").read_text())
    offenders = []
    for gid, gate in sorted(graph["gates"].items()):
        callers = gate.get("runtime_caller") or []
        if not callers:
            continue
        implementing = {
            (r.get("file") if isinstance(r, dict) else r)
            for r in (gate.get("code_refs") or [])
        }
        external = [c for c in callers if c.get("file") not in implementing]
        if not external:
            offenders.append(f"{gid} (only caller: {callers[0].get('file')})")
    assert not offenders, (
        "these gates are WIRED only by their own module calling itself:\n  "
        + "\n  ".join(offenders)
    )


def test_the_regeneration_steps_are_in_the_only_correct_order():
    """Order is load-bearing and was wrong twice by hand.

    --build must precede recompile, because recompile renders the state FROM the
    graph: reversed, the state renders from the previous graph and the frontier
    reports a closed gate as the top priority. emit_revised must be last, because
    it fingerprints the state, so anything rewriting the state after it makes the
    emitted roadmap instantly stale against its own detector.
    """
    from tools.roadmap import regenerate
    labels = [label for label, _ in regenerate.STEPS]
    assert labels.index("graph") < labels.index("documents+state")
    assert labels[-1] == "revised roadmap"
    assert labels.index("saturation") < labels.index("revised roadmap")
    for _label, args in regenerate.STEPS:
        assert args[0] == "-m", args


# RETIRED 2026-09-05 by operator instruction, recorded rather than silently dropped:
#   test_the_document_leads_with_the_kernel_not_with_history
#   test_the_roadmap_carries_a_fingerprint_and_can_detect_its_own_staleness
#   test_the_freshness_check_agrees_with_the_committed_artifacts
#
# All three verified the STRUCTURE and FRESHNESS of the generated boot-ROM
# document emitted by emit_revised.py. The operator designated ~/Downloads as
# roadmap authority and removed roadmap verification outright: "it's updated so
# frequently there isn't one true roadmap except for the one I point you to."
# A fingerprint, a staleness check and a fixed section order all assume a single
# pinned document, which is the assumption that was retired.
#
# What REMAINS asserted: everything about the CATALOG and the recompiler that
# does not depend on the roadmap being a fixed artifact. Those 14 tests still run.

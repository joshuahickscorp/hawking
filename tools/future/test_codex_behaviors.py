"""Tests for the Codex optimization-metabolism WorkUnit species.

Negative controls are watched failing: no citation, self-promotion, lease
acquisition, and a blocked GPU lane emitting SLEEPING rather than FAILED.
"""
from __future__ import annotations

import json

import pytest

from hcli.workunit import WorkUnit
from tools.future import codex_behaviors as cb
from tools.future import workunit_species as ws
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


def _handoff():
    doc, src = cb.load_handoff()
    if doc is None:
        pytest.skip(f"{cb.HANDOFF_REL} not visible (looked at checkout roots); loaded_from={src}")
    return doc, src


def _base_kwargs(**patch):
    spec = dict(cb._species_specs()[0])
    spec.update(patch)
    return spec


def test_build_and_selftest_emit_sealed_receipt():
    out = cb.selftest()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "CODEX_WORKUNIT_SPECIES.json"
    assert doc["schema"] == "hawking.future.codex_behaviors.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["work_units"]
    assert doc["species"]
    _assert_no_hardware_claims(doc)


def test_thirty_species_match_closed_vocabulary():
    handoff, _src = _handoff()
    species = cb.catalog(handoff=handoff)
    ids = [item["id"] for item in species]
    assert ids == list(cb.SPECIES_IDS)
    assert len(ids) == len(cb.SPECIES_IDS)
    assert len(set(ids)) == len(ids)
    assert "Era VI" not in json.dumps(species)
    assert "Odyssey IV" not in json.dumps(species)


def test_every_species_has_full_field_set():
    handoff, _src = _handoff()
    for item in cb.catalog(handoff=handoff):
        for field in cb.CODEX_UNIT_FIELDS:
            assert field in item, f"{item['id']} missing {field}"
        assert item["may_promote"] is False
        assert item["may_modify_verifier"] is False
        assert item["may_acquire_lease"] is False
        assert item["resources"]["gpu_authority"] is False
        assert item["measurement_class"] == "STATIC_ONLY"
        assert item["bench_state"] == "UNKNOWN"
        assert item["verification_level"] in cb.VERIFICATION_LEVELS
        assert item["era"] in ws.ERAS
        assert item["era"] != "VI"
        if item["odyssey"] is not None:
            assert item["odyssey"] in ws.ODYSSEYS
        if item["grounding"] == cb.GROUNDING_GROUNDED:
            assert item["citation"]
            assert item["citation"]["field_path"]
            assert item["citation"]["fragment"]
        else:
            assert item["grounding"] == cb.GROUNDING_UNGROUNDED
            assert item["would_ground"]
            assert item["citation"] is None


def test_grounded_citations_resolve_in_handoff():
    handoff, src = _handoff()
    # Cope with either checkout: record which path we took.
    assert src
    grounded = [s for s in cb.catalog(handoff=handoff) if s["grounding"] == cb.GROUNDING_GROUNDED]
    assert grounded, "at least some species must be grounded in the training trace"
    for item in grounded:
        proven = cb.prove_citation(handoff, item["citation"])
        assert proven["fragment"] == item["citation"]["fragment"]


def test_ungrounded_species_are_named_not_invented():
    handoff, _src = _handoff()
    ungrounded = [s for s in cb.catalog(handoff=handoff) if s["grounding"] == cb.GROUNDING_UNGROUNDED]
    ids = {s["id"] for s in ungrounded}
    assert "UPDATE_SCOREBOARD" in ids
    assert "ATTACK_LAW" in ids
    for item in ungrounded:
        assert "would_ground" in item and item["would_ground"]
        assert item["citation"] is None


def test_extends_parent_vocabulary_does_not_fork():
    handoff, _src = _handoff()
    parent = set(ws.SPECIES_IDS)
    child = cb.catalog(handoff=handoff)
    assert len(parent) == len(ws.SPECIES_IDS)
    for item in child:
        ext = item.get("extends_parent")
        if ext:
            assert ext in parent, f"{item['id']} extends unknown parent {ext}"
        assert item["id"] not in parent
    assert set(cb.SPECIES_IDS).isdisjoint(parent)


def test_emitter_refuses_species_with_no_handoff_citation():
    """NEGATIVE CONTROL: a wish-list species without a citation is refused."""
    handoff, _src = _handoff()
    with pytest.raises(cb.UngroundedSpeciesError, match="cites no observed"):
        cb.define_codex_species(handoff=handoff, **_base_kwargs(citation=None, grounding=cb.GROUNDING_GROUNDED))
    with pytest.raises(cb.UngroundedSpeciesError, match="cites no observed"):
        cb.define_codex_species(
            handoff=handoff,
            **_base_kwargs(citation={"field_path": "", "fragment": ""}),
        )


def test_emitter_refuses_invented_citation():
    """NEGATIVE CONTROL: a fragment that is not in the handoff is not a citation."""
    handoff, _src = _handoff()
    with pytest.raises(cb.CitationError, match="not found"):
        cb.define_codex_species(
            handoff=handoff,
            **_base_kwargs(
                citation={
                    "field_path": "atomic_change.name",
                    "fragment": "this-fragment-is-not-in-the-handoff",
                }
            ),
        )


def test_constructor_refuses_self_promotion():
    """NEGATIVE CONTROL: self-promotion authority is refused."""
    handoff, _src = _handoff()
    with pytest.raises(ws.SpeciesAuthorityError, match="self-promotion"):
        cb.define_codex_species(handoff=handoff, **_base_kwargs(may_promote=True))
    with pytest.raises(ws.SpeciesAuthorityError, match="forbidden authority"):
        cb.define_codex_species(
            handoff=handoff,
            **_base_kwargs(bounded_authority=("self_promotion", "read_receipts")),
        )


def test_constructor_refuses_lease_acquisition():
    """NEGATIVE CONTROL: a species cannot take a GPU lease."""
    handoff, _src = _handoff()
    with pytest.raises(ws.SpeciesAuthorityError, match="GPU lease"):
        cb.define_codex_species(handoff=handoff, **_base_kwargs(may_acquire_lease=True))
    with pytest.raises(ws.SpeciesAuthorityError, match="GPU lease"):
        cb.define_codex_species(
            handoff=handoff,
            **_base_kwargs(bounded_authority=("acquire_gpu_lease", "read_receipts")),
        )


def test_authority_and_citation_guards_are_watched_failing():
    handoff, _src = _handoff()
    results = cb.prove_refusals(handoff)
    assert len(results) >= 8
    assert all(row["refused"] is True for row in results)
    trials = {row["trial"] for row in results}
    assert "no_citation" in trials
    assert "self_promotion_flag" in trials
    assert "acquire_lease_flag" in trials
    assert "invented_citation" in trials


def test_blocked_lane_emits_sleeping_not_failed():
    """NEGATIVE CONTROL: a blocked GPU species is SLEEPING with a wake condition, never FAILED."""
    handoff, _src = _handoff()
    blockers = cb.blockers_from_handoff(handoff)
    assert blockers, "handoff lists exact_physical_blockers; empty would make this test tautological"
    species = cb.catalog_by_id(handoff=handoff)
    gpu = cb.emit_species_unit(species["PROFILE_GPU"], cycle=0, dependencies=[], blockers=blockers)
    assert gpu["status"] == cb.STATUS_SLEEPING
    assert gpu["classification"] == cb.CLASS_SLEEPING
    assert gpu.get("wake_condition")
    assert "SLEEPING until" in gpu["wake_condition"]
    assert gpu["status"] != "failed"
    assert gpu["status"] != "skipped"
    assert str(gpu["status"]).lower() not in {"failed", "skipped"}
    # Static lane on the same blocker set stays runnable.
    static = cb.emit_species_unit(species["FIND_TALLEST_COST"], cycle=0, dependencies=[], blockers=blockers)
    assert static["status"] != cb.STATUS_SLEEPING
    assert static["status"] == "pending"
    # Sleeping is derived from blockers, not hard-coded: no blockers => not SLEEPING.
    awake = cb.emit_species_unit(species["PROFILE_GPU"], cycle=0, dependencies=[], blockers=[])
    assert awake["status"] != cb.STATUS_SLEEPING
    assert awake["status"] == "pending"


def test_loop_orders_profile_then_tallest_then_generate():
    handoff, _src = _handoff()
    units = cb.emit_cycle(handoff=handoff, cycle=0)
    by_species = {row["species"]: row for row in units}
    assert "PROFILE_COMPLETE_TOKEN" in by_species
    assert "FIND_TALLEST_COST" in by_species
    assert "GENERATE_KERNEL_CANDIDATE" in by_species
    assert "PROTECTED_AB" in by_species
    assert all(row.get("species") != cb.REFILL_SPECIES for row in units)
    tallest = by_species["FIND_TALLEST_COST"]
    profile_ids = {by_species[s]["id"] for s in cb.CYCLE_WAVES[0]}
    assert set(tallest["dependencies"]) == profile_ids
    gen = by_species["GENERATE_KERNEL_CANDIDATE"]
    assert by_species["SEARCH_ARCHITECTURE_LAWS"]["id"] in gen["dependencies"]


def test_initial_cycle_does_not_include_reprofile():
    handoff, _src = _handoff()
    units = cb.emit_cycle(handoff=handoff, cycle=0)
    assert all(row.get("species") != cb.REFILL_SPECIES for row in units)


def test_win_enqueues_reprofile_after_win():
    handoff, _src = _handoff()
    units = cb.emit_cycle(handoff=handoff, cycle=0)
    winner = next(row for row in units if row.get("species") == "PROTECTED_AB")
    refill = cb.enqueue_after_win(
        units,
        winner_id=winner["id"],
        outcome="PHYSICAL_WIN_MODEL_LOCAL",
        handoff=handoff,
    )
    assert refill
    assert refill[0]["species"] == cb.REFILL_SPECIES
    assert winner["id"] in refill[0]["dependencies"]
    profiles = [row for row in refill if str(row.get("species") or "").startswith("PROFILE_")]
    assert profiles
    assert all(refill[0]["id"] in (row.get("dependencies") or []) for row in profiles)


def test_non_win_does_not_enqueue_reprofile():
    handoff, _src = _handoff()
    units = cb.emit_cycle(handoff=handoff, cycle=0)
    with pytest.raises(cb.CodexSpeciesError, match="not a physical win"):
        cb.enqueue_after_win(
            units,
            winner_id="codex.metabolism.0.PROTECTED_AB",
            outcome="IMPLEMENTED_UNMEASURED",
            handoff=handoff,
        )


def test_emitted_units_match_hcli_shape_and_codex_fields():
    handoff, _src = _handoff()
    units = cb.emit_cycle(handoff=handoff, cycle=0)
    assert units
    for row in units:
        ws.validate_emitted_unit(row)
        cb._validate_codex_fields(row)
        roundtrip = WorkUnit.from_dict(row)
        assert roundtrip.id == row["id"]
        assert roundtrip.verifier == row["verifier"]
        assert roundtrip.resource_class == row["resource_class"]
        assert row["measurement_class"] == "STATIC_ONLY"
        assert row["bench_state"] == "UNKNOWN"
        assert row["gpu_authority"] is False
        assert row.get("may_promote") in (None, False)


def test_receipt_records_loaded_from_without_asserting_absence():
    """Sparse checkout: missing-here is not evidence the file does not exist."""
    doc, src = cb.load_handoff()
    # Either path is legal. Record which one we took.
    if doc is None:
        assert src in {"missing", "cached"} or src.startswith("git:")
    else:
        assert src
        assert "recurring_hcli_workunit_species" in doc


def test_counts_are_derived_from_data():
    handoff, _src = _handoff()
    species = cb.catalog(handoff=handoff)
    units = cb.emit_cycle(handoff=handoff, cycle=0)
    sleeping = [row for row in units if row["status"] == cb.STATUS_SLEEPING]
    blockers = cb.blockers_from_handoff(handoff)
    assert len(species) == len(cb.SPECIES_IDS)
    assert len(units) == sum(len(wave) for wave in cb.CYCLE_WAVES)
    assert len(sleeping) == sum(
        1
        for row in units
        if (row.get("resources") or {}).get("lane") in cb.lanes_blocked_by(blockers)
    )
    loose = handoff.get("recurring_hcli_workunit_species") or []
    assert list(loose) == list(cb.LOOSE_HANDOFF_SPECIES)


def test_no_hardware_field_is_numeric_in_emitted_docs():
    handoff, _src = _handoff()
    units = cb.emit_cycle(handoff=handoff, cycle=0)
    blob = json.dumps(units)
    for key in HARDWARE_FIELDS:
        # Presence of the *name* as a required-field citation is allowed;
        # a numeric value under that key is not.
        for row in units:
            val = row.get(key)
            assert not isinstance(val, (int, float)), f"{row['id']} smuggled {key}={val!r}"
    _assert_no_hardware_claims({"work_units": units, "species": cb.catalog(handoff=handoff)})
    del blob


def test_nine_loose_labels_are_covered_by_precise_species():
    handoff, _src = _handoff()
    mapping = cb._loose_mapping(cb.catalog(handoff=handoff))
    for label in cb.LOOSE_HANDOFF_SPECIES:
        assert mapping.get(label), f"loose label {label!r} has no precise species"


def test_fail_closed_when_handoff_missing_and_species_must_be_grounded():
    previous = cb._HANDOFF
    previous_src = cb._HANDOFF_SRC
    try:
        cb.inject_handoff(None, "missing")
        with pytest.raises(FileNotFoundError, match="CODEX_ACCELERATOR_HANDOFF"):
            cb.catalog()
    finally:
        if previous is cb._UNSET:
            cb._HANDOFF = cb._UNSET
            cb._HANDOFF_SRC = previous_src
        else:
            cb.inject_handoff(previous, previous_src or "restored")
        cb.load_handoff(force=True)

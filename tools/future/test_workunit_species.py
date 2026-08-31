"""Tests for the future HCLI WorkUnit species catalog and starting queue."""
from __future__ import annotations

import json

import pytest

from hcli.workunit import WorkUnit
from tools.future import workunit_species as ws
from tools.future._common import RECEIPTS


def _base_species_kwargs(**patch):
    spec = dict(ws._species_specs()[0])
    spec.update(patch)
    return spec


def test_build_and_selftest_emit_sealed_receipt():
    out = ws.selftest()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "HCLI_FUTURE_WORKUNITS.json"
    assert doc["schema"] == "hawking.future.workunit_species.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["work_units"]
    assert doc["species"]


def test_ten_species_declared_through_constructor():
    species = ws.catalog()
    ids = [item["id"] for item in species]
    assert ids == list(ws.SPECIES_IDS)
    assert len(ids) == 10
    assert len(set(ids)) == 10
    required = (
        "evidence_parents",
        "bounded_authority",
        "resource_class",
        "verifier",
        "budget",
        "stop_condition",
    )
    for item in species:
        for field in required:
            assert item[field], f"{item['id']} missing {field}"
        assert item["may_promote"] is False
        assert item["may_modify_verifier"] is False
        assert item["may_choose_singularity"] is False
        assert item["may_destructive_mutate"] is False
        assert item["era"] in ws.ERAS
        assert item["era"] != "VI"
        if item["odyssey"] is not None:
            assert item["odyssey"] in ws.ODYSSEYS
            assert item["odyssey"] != "IV"
        assert "MUTATION" not in item["bounded_authority"]
        assert not (set(item["bounded_authority"]) & ws.FORBIDDEN_AUTHORITY)


def test_constructor_rejects_self_promotion_authority():
    with pytest.raises(ws.SpeciesAuthorityError, match="forbidden authority"):
        ws.define_species(**_base_species_kwargs(bounded_authority=("self_promotion",)))
    with pytest.raises(ws.SpeciesAuthorityError, match="self-promotion"):
        ws.define_species(**_base_species_kwargs(may_promote=True))


def test_constructor_rejects_verifier_modification_authority():
    with pytest.raises(ws.SpeciesAuthorityError, match="forbidden authority"):
        ws.define_species(**_base_species_kwargs(bounded_authority=("weaken_verifier",)))
    with pytest.raises(ws.SpeciesAuthorityError, match="verifier-modification"):
        ws.define_species(**_base_species_kwargs(may_modify_verifier=True))
    with pytest.raises(ws.SpeciesAuthorityError, match="weaken verification"):
        ws.define_species(**_base_species_kwargs(verifier="self"))


def test_constructor_rejects_singularity_and_destructive_mutation():
    with pytest.raises(ws.SpeciesAuthorityError, match="Singularity"):
        ws.define_species(**_base_species_kwargs(may_choose_singularity=True))
    with pytest.raises(ws.SpeciesAuthorityError, match="destructive"):
        ws.define_species(**_base_species_kwargs(may_destructive_mutate=True))
    with pytest.raises(ws.SpeciesAuthorityError, match="forbidden authority"):
        ws.define_species(**_base_species_kwargs(bounded_authority=("choose_singularity",)))
    with pytest.raises(ws.SpeciesAuthorityError, match="forbidden authority"):
        ws.define_species(**_base_species_kwargs(bounded_authority=("destructive_mutation",)))


def test_authority_guard_is_watched_failing():
    # A guard nobody has watched fail is not a guard.
    results = ws._prove_authority_refusal()
    assert len(results) >= 4
    assert all(row["refused"] is True for row in results)
    trials = {row["trial"] for row in results}
    assert "self_promotion" in trials
    assert "weaken_verifier" in trials


def test_unknown_authority_is_refused():
    with pytest.raises(ws.SpeciesAuthorityError, match="unknown authority"):
        ws.define_species(
            **_base_species_kwargs(bounded_authority=("quietly_install_resident",))
        )


def test_starting_queue_nonempty_and_references_live_candidate_ids():
    units = ws.build_starting_queue()
    assert units, "starting queue must not be empty"
    ids = ws.queue_identity_sets(units)
    assert ids["candidate_ids"]
    missing_ready = [cid for cid in ws.CORE_READY_QWEN27 if cid not in ids["candidate_ids"]]
    missing_blocked = [cid for cid in ws.CORE_BLOCKED_FLASH if cid not in ids["candidate_ids"]]
    assert not missing_ready, missing_ready
    assert not missing_blocked, missing_blocked
    for eid in (
        "qwen27-move-or-recompute-boundary",
        "flash-semantic-transport-hwir",
        "ane-regular-island-probe",
    ):
        assert eid in ids["experiment_ids"]


def test_blocked_flash_units_carry_blockers():
    units = ws.build_starting_queue()
    blocked = [
        row
        for row in units
        if row.get("candidate_id") in ws.CORE_BLOCKED_FLASH
    ]
    assert len(blocked) == len(ws.CORE_BLOCKED_FLASH)
    for row in blocked:
        assert row["status"] == "blocked"
        assert row.get("blocked_reason"), row["id"]
        assert row["verifier"]
        assert row["resource_class"] == "GPU_EXCLUSIVE"
        assert row.get("species") == "accelerator_candidate_qualification"


def test_emitted_units_match_recovered_hcli_shape():
    units = ws.build_starting_queue()
    for row in units:
        ws.validate_emitted_unit(row)
        roundtrip = WorkUnit.from_dict(row)
        assert roundtrip.id == row["id"]
        assert roundtrip.verifier == row["verifier"]
        assert roundtrip.resource_class == row["resource_class"]
        assert "claim_boundary" in row


def test_live_work_units_are_consumable_not_placeholders():
    qual, qual_src = ws.load_headless(ws.QUAL_REL)
    units = ws.build_starting_queue()
    physical = [row for row in units if row.get("candidate_id") in ws.CORE_READY_QWEN27]
    assert physical
    for row in physical:
        assert row["id"] == f"accelerator.physical.{row['candidate_id']}"
        assert row["role"] == "accelerator_physical_qualification"
        assert row["provider"] == "accelerator_physical_queue"
        assert row["effect_class"] == "REVERSIBLE"
        if qual is not None:
            assert row.get("diagnostic_command") or row.get("protected_command")
    assert qual_src != ""


def test_every_species_has_a_queue_unit():
    units = ws.build_starting_queue()
    present = {row.get("species") for row in units}
    missing = [sid for sid in ws.SPECIES_IDS if sid not in present]
    assert not missing, missing


def test_fpga_species_is_simulation_not_a_civilization():
    fpga = next(item for item in ws.catalog() if item["id"] == "fpga_simulation")
    assert fpga["resource_class"] == "COMPILE"
    assert "simulate_fpga_without_board" in fpga["bounded_authority"]
    assert "backend" not in fpga["title"].lower() or "not" in fpga["description"].lower()
    assert "not an FPGA backend" in fpga["description"] or "not its own" in fpga["description"]
    assert fpga["era"] != "VI"


def test_green_machine_does_not_invent_joules():
    green = next(item for item in ws.catalog() if item["id"] == "green_machine_measurement")
    assert "record_unknown_metrics" in green["bounded_authority"]
    assert "UNKNOWN" in green["stop_condition"]
    units = [row for row in ws.build_starting_queue() if row.get("species") == "green_machine_measurement"]
    assert units
    blob = json.dumps(units)
    assert "joules_per_token" not in blob or "UNKNOWN" in blob


def test_no_species_uses_era_vi_or_odyssey_iv():
    for item in ws.catalog():
        assert item["era"] in {"I", "II", "III", "IV", "V"}
        assert "VI" not in str(item["era"])
        assert item.get("odyssey") in {None, "I", "II", "III"}


def test_queue_units_cannot_express_self_promotion():
    for row in ws.build_starting_queue():
        assert not row.get("may_promote")
        assert not row.get("may_modify_verifier")
        assert row.get("effect_class") in ws.ALLOWED_EFFECT


def test_static_only_physical_candidates_are_not_dispatchable():
    units = ws.build_starting_queue()
    static = [
        row
        for row in units
        if row.get("candidate_status") == "STATIC_ONLY" and row.get("candidate_id")
    ]
    assert static
    for row in static:
        assert row["status"] != "pending", row["id"]
        assert row["status"] == "blocked"
        assert row.get("blocked_reason")

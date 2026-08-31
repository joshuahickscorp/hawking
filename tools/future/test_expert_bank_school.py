import json

import pytest

from tools.future import expert_bank_school as ebs
from tools.future._common import RECEIPTS, HardwareClaimError, write_receipt


def test_build_emits_sealed_receipt():
    out = ebs.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "EXPERT_BANK_SCHOOL.json"
    assert doc["schema"] == "hawking.future.expert_bank_school.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert "Static sidecar" in doc["claim_boundary"]
    assert doc["fit_policy"] == "NOT_FIT"
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["counts"]["storage"] == 13
    assert doc["counts"]["compute"] == 7
    assert doc["counts"]["refusal_controls_fired"] == 3


def test_selftest_aliases_build():
    assert ebs.selftest().name == "EXPERT_BANK_SCHOOL.json"


def test_storage_kinds_complete_and_admitted():
    rows = ebs.generate_storage()
    assert [c["kind"] for c in rows] == list(ebs.REQUIRED_STORAGE_KINDS)
    for c in rows:
        for field in ebs.STORAGE_FIELDS:
            assert field in c, f"{c['id']} missing {field}"
        assert c["axis"] == "STORAGE"
        assert c["forbids_dense_rematerialization"] is True
        assert c["status"] == "HYPOTHESIS_UNFITTED"
        assert c["evidence_class"] == "STATIC_ONLY"
        assert c["native_execution_concept"]
        assert c["cheapest_falsifier"]
        assert c["scar_distance"]


def test_compute_kinds_complete_and_name_repeated_work():
    rows = ebs.generate_compute()
    assert [c["kind"] for c in rows] == list(ebs.REQUIRED_COMPUTE_KINDS)
    for c in rows:
        for field in ebs.COMPUTE_FIELDS:
            assert field in c, f"{c['id']} missing {field}"
        assert c["axis"] == "COMPUTE"
        assert c["repeated_computation"]
        assert c["why_currently_repeated"]
        assert c["forbids_dense_rematerialization"] is True


def test_emitted_candidates_are_not_the_dead_families():
    blobs = []
    for c in ebs.generate():
        blobs.append(ebs._blob(c))
        assert c["kind"] not in {
            "raw_global_expert_similarity",
            "trivial_shared_basis",
            "unchanged_archetype",
        }
        assert ebs.match_scar(c) is None
    joined = " ".join(blobs)
    assert "raw global expert similarity" not in joined
    assert "trivial shared basis" not in joined
    assert "unchanged archetype" not in joined


def test_negative_control_refuses_raw_global_similarity_but_emits_distinct():
    """Guard nobody has watched fail is not a guard.

    The generator must refuse the recorded-dead 'raw global expert
    similarity' family AND still emit a structurally distinct candidate,
    so the refusal is targeted, not a blanket ban on sharing.
    """
    with pytest.raises(ebs.DeadHypothesisError) as ei:
        ebs.admit_candidate(
            {
                "id": "PROBE-RAW-GLOBAL-SIMILARITY",
                "mechanism": "raw global expert similarity",
            },
            require_schema=False,
        )
    err = ei.value
    assert err.scar["id"] == "SCAR-RAW-GLOBAL-SIMILARITY"
    assert err.scar["family"] == ebs.DEAD_FAMILY_RAW
    assert "REFUSED" in str(err)
    assert "SCAR-RAW-GLOBAL-SIMILARITY" in str(err)

    live = next(
        c for c in ebs.generate_storage() if c["kind"] == "common_left_subspaces"
    )
    admitted = ebs.admit_candidate(live)
    assert admitted["id"] == "STORE-COMMON-LEFT-SUBSPACE"
    assert admitted["axis"] == "STORAGE"
    assert ebs.match_scar(admitted) is None


def test_refusal_is_targeted_not_blanket():
    with pytest.raises(ebs.DeadHypothesisError) as ei_basis:
        ebs.admit_candidate(
            {"id": "P-BASIS", "mechanism": "trivial shared basis"},
            require_schema=False,
        )
    assert ei_basis.value.scar["family"] == ebs.DEAD_FAMILY_BASIS

    with pytest.raises(ebs.DeadHypothesisError) as ei_arch:
        ebs.admit_candidate(
            {"id": "P-ARCH", "mechanism": "unchanged archetype"},
            require_schema=False,
        )
    assert ei_arch.value.scar["family"] == ebs.DEAD_FAMILY_ARCHETYPE

    # Structured cousins that share surface words must still be admitted.
    route = next(
        c
        for c in ebs.generate_storage()
        if c["kind"] == "route_conditioned_archetypes"
    )
    ebs.admit_candidate(route)
    shared_in = next(
        c
        for c in ebs.generate_storage()
        if c["kind"] == "shared_input_transforms"
    )
    ebs.admit_candidate(shared_in)
    compute = next(
        c
        for c in ebs.generate_compute()
        if c["kind"] == "shared_xb_then_skinny"
    )
    ebs.admit_candidate(compute)


def test_family_tag_also_fires_refusal():
    with pytest.raises(ebs.DeadHypothesisError) as ei:
        ebs.admit_candidate(
            {
                "id": "P-FAM",
                "mechanism": "a structurally novel packing of orthogonal experts",
                "family": ebs.DEAD_FAMILY_RAW,
            },
            require_schema=False,
        )
    assert ei.value.scar["family"] == ebs.DEAD_FAMILY_RAW


def test_refusal_controls_in_receipt_actually_fired():
    rows = ebs.assert_guards_fire()
    assert len(rows) == 3
    assert {r["probe_id"] for r in rows} == {
        "PROBE-RAW-GLOBAL-SIMILARITY",
        "PROBE-TRIVIAL-SHARED-BASIS",
        "PROBE-UNCHANGED-ARCHETYPE",
    }
    assert all(r["refused"] is True for r in rows)
    assert {r["scar_id"] for r in rows} == {
        "SCAR-RAW-GLOBAL-SIMILARITY",
        "SCAR-TRIVIAL-SHARED-BASIS",
        "SCAR-UNCHANGED-ARCHETYPE",
    }


def test_extra_index_scar_is_unioned_not_instead_of_builtin():
    extra = [
        {
            "id": "SCAR-PLANTED",
            "family": "INDEX",
            "title": "planted dead lever",
            "phrases": ("planted dead lever",),
        }
    ]
    with pytest.raises(ebs.DeadHypothesisError) as ei:
        ebs.admit_candidate(
            {"id": "P-PLANT", "mechanism": "planted dead lever"},
            extra_scars=extra,
            require_schema=False,
        )
    assert ei.value.scar["id"] == "SCAR-PLANTED"
    # Builtin still fires when extras are present.
    with pytest.raises(ebs.DeadHypothesisError) as ei2:
        ebs.admit_candidate(
            {"id": "P-RAW", "mechanism": "raw global expert similarity"},
            extra_scars=extra,
            require_schema=False,
        )
    assert ei2.value.scar["id"] == "SCAR-RAW-GLOBAL-SIMILARITY"
    live = ebs.generate_storage(extra_scars=extra)[0]
    assert live["id"] == "STORE-COMMON-LEFT-SUBSPACE"


def test_live_candidate_missing_fields_is_schema_error_not_scar():
    with pytest.raises(ebs.CandidateSchemaError):
        ebs.admit_candidate(
            {
                "id": "STORE-INCOMPLETE",
                "axis": "STORAGE",
                "kind": "common_left_subspaces",
                "mechanism": "common left subspaces incomplete probe",
            }
        )


def test_schema_requires_unfitted_static_only():
    live = dict(ebs.STORAGE_CANDIDATES[0])
    live["status"] = "FITTED"
    with pytest.raises(ebs.CandidateSchemaError):
        ebs.admit_candidate(live)
    live = dict(ebs.STORAGE_CANDIDATES[0])
    live["evidence_class"] = "DIAGNOSTIC_RELATIVE"
    with pytest.raises(ebs.CandidateSchemaError):
        ebs.admit_candidate(live)
    live = dict(ebs.STORAGE_CANDIDATES[0])
    live["forbids_dense_rematerialization"] = False
    with pytest.raises(ebs.CandidateSchemaError):
        ebs.admit_candidate(live)


def test_receipt_does_not_claim_hardware_numbers():
    # write_receipt raises HardwareClaimError if a hardware field is numeric.
    # The school receipt must be writable, which we already did in test_build.
    # Pin the inverse: a planted tps would be refused by the common seal.
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "_EXPERT_BANK_SCHOOL_HARDWARE_PROBE.json",
            {"schema": "probe", "version": 1, "tps": 12.0},
            "tools/future/test_expert_bank_school.py",
        )


def test_qwen80_negative_is_recovered_when_on_disk():
    rec = ebs._recover_qwen80()
    assert rec["on_disk"] is True
    assert rec["layer"] == 10
    assert rec["n_experts"] == 96
    assert rec["gate_pairwise_cosine_mean"] < 0.01
    assert rec["up_subspace_overlap_top32"] < 0.05


def test_consult_does_not_import_sibling_and_records_index_absence():
    sources = ebs.consult_scar_sources()
    # The sibling index lane may or may not have landed; the module must cope with
    # both and say which path it took.
    assert isinstance(sources["index_present"], bool)
    assert "NEGATIVE_SCIENCE_INDEX.json" in sources["index_path"]
    named = {r["path"]: r["on_disk"] for r in sources["named_flash_tools"]}
    # Environment-coupled: this file is uncommitted, so it is invisible from a
    # sparse lane worktree and visible from the primary one. Its presence is a
    # fact about the checkout, not about this module -- assert the module COPES
    # either way rather than pinning the environment it was written in.
    assert isinstance(named["tools/flash_expert_bank_profile.py"], bool)
    # Environment-coupled: uncommitted files are invisible from a sparse lane
    # worktree and visible from the primary one. Assert the module copes, not
    # the checkout it was written in.
    assert isinstance(named["tools/flash_doctor_bank_screen.py"], bool)


def test_generate_is_deterministic():
    a = ebs.generate()
    b = ebs.generate()
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert [c["id"] for c in a] == sorted(c["id"] for c in a)


def test_killed_hypotheses_cover_the_three_named_families():
    families = {s["family"] for s in ebs.BUILTIN_SCARS}
    assert families == {
        ebs.DEAD_FAMILY_RAW,
        ebs.DEAD_FAMILY_BASIS,
        ebs.DEAD_FAMILY_ARCHETYPE,
    }
    titles = {s["title"] for s in ebs.BUILTIN_SCARS}
    assert "raw global expert similarity" in titles
    assert "trivial shared basis" in titles
    assert "unchanged archetype" in titles

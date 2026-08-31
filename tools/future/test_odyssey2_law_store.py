"""Tests for the Odyssey II scoped law store.

The negative control is the load-bearing test: promote() and
transfer_candidates() must RAISE, not set a flag.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from tools.future import odyssey2_law_store as ols
from tools.future import status_causality as sc
from tools.future._common import RECEIPTS, HardwareClaimError


def _law(**kwargs) -> ols.Law:
    defaults = dict(
        law_id="LAW-TEST",
        statement="a model-local observation used by tests",
        source_model="Qwen3.8-27B",
        source_device="UNKNOWN",
        architecture_family="dense_hybrid_transformer",
        organ_class="mlp",
        backend="Metal",
        evidence_strength="DIAGNOSTIC_RELATIVE",
        evidence_refs=("receipts/headless/ODYSSEY_TRANSFER_PROVEN.json",),
        scope="MODEL_LOCAL",
        transfer_candidates=(),
        transfer_confidence={
            "value": 0.45,
            "basis": "test fixture",
        },
        counterexample_requirement="a measurement that fails the statement",
        expected_saved_experiments=None,
        actual_saved_experiments=None,
        time_to_first_useful_executable_ns=None,
    )
    defaults.update(kwargs)
    return ols.validate_law(ols.Law(**defaults))


def _two_model_evidence(**kwargs) -> dict:
    ev = {
        "models": ["Qwen3.8-27B", "qwen3.8-27b-abliterated-twin"],
        "architecture_families": ["dense_hybrid_transformer"],
        "backends": ["Metal"],
        "machines": ["APPLE_GPU_0"],
        "evidence_strength": "DIAGNOSTIC_RELATIVE",
        "evidence_refs": ["receipts/headless/ODYSSEY_TRANSFER_PROVEN.json"],
        "counterexample_discharged": False,
    }
    ev.update(kwargs)
    return ev


def test_law_record_has_exactly_the_contract_fields():
    law = _law()
    d = law.to_dict()
    assert tuple(d) == ols.LAW_FIELDS
    for field in ols.LAW_FIELDS:
        assert field in d
    assert d["time_to_first_useful_executable_ns"] is None


def test_lattice_is_sequential_and_complete():
    assert ols.SCOPES == (
        "MODEL_LOCAL",
        "ARCHITECTURE_FAMILY",
        "BACKEND_FAMILY",
        "MACHINE_LOCAL",
        "GENERIC_CANDIDATE",
        "GENERIC_VERIFIED",
    )
    assert ols.EVIDENCE_STRENGTHS == (
        "ANECDOTE",
        "STATIC",
        "DIAGNOSTIC_RELATIVE",
        "PROTECTED_ABSOLUTE",
        "REPRODUCED",
    )


def test_promote_architecture_family_with_two_models():
    law = _law()
    out = ols.promote(law, "ARCHITECTURE_FAMILY", _two_model_evidence())
    assert out.scope == "ARCHITECTURE_FAMILY"
    assert law.scope == "MODEL_LOCAL"
    assert out.law_id == law.law_id


def test_promote_backend_family_with_two_backends():
    law = _law(scope="ARCHITECTURE_FAMILY")
    out = ols.promote(
        law,
        "BACKEND_FAMILY",
        _two_model_evidence(backends=["Metal", "CUDA"]),
    )
    assert out.scope == "BACKEND_FAMILY"


def test_promote_generic_verified_with_protected_two_families_and_discharged_counterexample():
    law = _law(
        scope="GENERIC_CANDIDATE",
        evidence_strength="PROTECTED_ABSOLUTE",
        transfer_confidence={"value": 0.75, "basis": "fixture"},
    )
    out = ols.promote(
        law,
        "GENERIC_VERIFIED",
        {
            "models": ["Qwen3.8-27B", "tiiuae/Falcon-H1-7B-Instruct"],
            "architecture_families": ["dense_hybrid_transformer", "falcon_h1"],
            "backends": ["Metal", "CUDA"],
            "evidence_strength": "PROTECTED_ABSOLUTE",
            "evidence_refs": ["receipts/headless/MATCHED_BITS_FALCON_H1.json"],
            "counterexample_discharged": True,
        },
    )
    assert out.scope == "GENERIC_VERIFIED"


def test_promote_refuses_model_local_to_generic_verified_on_single_model_evidence():
    """NEGATIVE CONTROL: single-model evidence cannot jump to GENERIC_VERIFIED."""
    law = _law()
    evidence = {
        "models": ["Qwen3.8-27B"],
        "architecture_families": ["dense_hybrid_transformer"],
        "backends": ["Metal"],
        "evidence_strength": "DIAGNOSTIC_RELATIVE",
        "evidence_refs": ["receipts/headless/ODYSSEY_TRANSFER_PROVEN.json"],
        "counterexample_discharged": False,
    }
    with pytest.raises(ols.ScopeViolation) as ei:
        ols.promote(law, "GENERIC_VERIFIED", evidence)
    err = ei.value
    assert isinstance(err, ols.ScopeViolation)
    assert err.from_scope == "MODEL_LOCAL"
    assert err.to_scope == "GENERIC_VERIFIED"
    assert err.reason == "level_skip"
    assert law.scope == "MODEL_LOCAL"


def test_promote_refuses_level_skip_even_with_rich_evidence():
    """NEGATIVE CONTROL: skipping ARCHITECTURE_FAMILY is refused even if
    the evidence would have been enough for a later level."""
    law = _law()
    evidence = _two_model_evidence(
        backends=["Metal", "CUDA"],
        architecture_families=["dense_hybrid_transformer", "falcon_h1"],
        evidence_strength="PROTECTED_ABSOLUTE",
        counterexample_discharged=True,
    )
    with pytest.raises(ols.ScopeViolation) as ei:
        ols.promote(law, "BACKEND_FAMILY", evidence)
    assert ei.value.reason == "level_skip"
    assert "refusing level skip" in str(ei.value)
    with pytest.raises(ols.ScopeViolation) as ei2:
        ols.promote(law, "GENERIC_CANDIDATE", evidence)
    assert ei2.value.reason == "level_skip"


def test_promote_refuses_architecture_family_on_one_model():
    law = _law()
    with pytest.raises(ols.ScopeViolation) as ei:
        ols.promote(
            law,
            "ARCHITECTURE_FAMILY",
            _two_model_evidence(models=["Qwen3.8-27B"]),
        )
    assert ei.value.reason == "need_two_models_in_family"
    assert ei.value.to_scope == "ARCHITECTURE_FAMILY"


def test_promote_refuses_generic_verified_without_protected_evidence():
    law = _law(
        scope="GENERIC_CANDIDATE",
        transfer_confidence={"value": 0.75, "basis": "fixture"},
    )
    with pytest.raises(ols.ScopeViolation) as ei:
        ols.promote(
            law,
            "GENERIC_VERIFIED",
            {
                "models": ["Qwen3.8-27B", "Falcon-H1-7B"],
                "architecture_families": ["dense_hybrid_transformer", "falcon_h1"],
                "evidence_strength": "DIAGNOSTIC_RELATIVE",
                "counterexample_discharged": True,
            },
        )
    assert ei.value.reason == "need_protected_or_reproduced"


def test_promote_refuses_generic_verified_without_discharged_counterexample():
    law = _law(
        scope="GENERIC_CANDIDATE",
        evidence_strength="REPRODUCED",
        transfer_confidence={"value": 0.75, "basis": "fixture"},
    )
    with pytest.raises(ols.ScopeViolation) as ei:
        ols.promote(
            law,
            "GENERIC_VERIFIED",
            {
                "models": ["Qwen3.8-27B", "Falcon-H1-7B"],
                "architecture_families": ["dense_hybrid_transformer", "falcon_h1"],
                "evidence_strength": "REPRODUCED",
                "counterexample_discharged": False,
            },
        )
    assert ei.value.reason == "counterexample_not_discharged"


def test_promote_does_not_clamp_unknown_or_narrowing():
    law = _law()
    with pytest.raises(ols.ScopeViolation):
        ols.promote(law, "NOT_A_SCOPE", _two_model_evidence())
    with pytest.raises(ols.ScopeViolation) as ei:
        ols.promote(
            _law(scope="ARCHITECTURE_FAMILY"),
            "MODEL_LOCAL",
            _two_model_evidence(),
        )
    assert ei.value.reason == "not_a_widening"


def test_transfer_refuses_negative_atlas_lever():
    """NEGATIVE CONTROL: a lever the atlas already killed must raise."""
    law = _law(
        law_id="LAW-INTER-EXPERT-REDUNDANCY",
        statement=(
            "delta coding / shared low-rank bases / cluster-mean subtraction "
            "across experts (inter_expert_redundancy)"
        ),
        organ_class="moe_expert",
        source_model="Qwen3.8-27B",
    )
    with pytest.raises(ols.NegativeTransferError) as ei:
        ols.transfer_candidates(law, "Flash")
    err = ei.value
    assert isinstance(err, ols.NegativeTransferError)
    assert err.atlas_key == "inter_expert_redundancy"
    assert err.target == "Flash"
    assert "refusing transfer" in str(err)


def test_transfer_refuses_expert_merging_atlas_entry():
    law = _law(
        law_id="LAW-EXPERT-MERGING-OMITTED-FROM-SURVIVORS",
        statement=(
            "reconstruct an omitted MoE expert from a learned combination of "
            "surviving experts"
        ),
        organ_class="moe_expert",
    )
    with pytest.raises(ols.NegativeTransferError) as ei:
        ols.transfer_candidates(law, "Qwen27")
    assert ei.value.atlas_key == "expert_merging_omitted_from_survivors"


def test_transfer_proposes_flash_from_qwen27_for_a_live_law():
    law = _law(
        law_id="LAW-XFER-QWEN27-PACKED_LOW_BIT_GEMV",
        statement="packed low-bit GEMV labelled DIRECT_TRANSFER on Qwen27",
        source_model="Qwen3.8-27B",
        organ_class="packed low-bit GEMV",
        transfer_confidence={
            "value": 0.70,
            "basis": "DIRECT_TRANSFER in QWEN38_ACCELERATOR_TRANSFER_MAP; claim_boundary requires re-earn of parity",
        },
    )
    proposals = ols.transfer_candidates(law, "Flash")
    assert proposals, "a live Qwen27 law must propose Flash"
    assert proposals[0]["target_school"] == "Flash"
    assert proposals[0]["target_model"] == "Qwen/Qwen3.8-Flash-Next"
    assert 0.0 <= proposals[0]["confidence"] <= 1.0
    assert proposals[0]["counterexample_requirement"]


def test_transfer_proposes_qwen27_from_flash():
    law = _law(
        law_id="LAW-XFER-FLASH-COMMAND_BUFFER_SCHEDULING_TELEMETRY",
        statement="command-buffer scheduling/telemetry labelled DIRECT_TRANSFER on Flash",
        source_model="Qwen/Qwen3.8-Flash-Next",
        architecture_family="qwen4_exp",
        organ_class="command-buffer scheduling/telemetry",
        evidence_strength="STATIC",
        transfer_confidence={
            "value": 0.70,
            "basis": "DIRECT_TRANSFER in QWEN38_ACCELERATOR_TRANSFER_MAP; claim_boundary requires re-earn of parity",
        },
    )
    proposals = ols.transfer_candidates(law, "Qwen27")
    assert proposals
    assert proposals[0]["target_school"] == "Qwen27"
    assert proposals[0]["target_model"] == "Qwen3.8-27B"


def test_transfer_to_self_is_empty_not_an_error():
    law = _law(source_model="Qwen3.8-27B")
    assert ols.transfer_candidates(law, "Qwen27") == []


def test_schools_cover_flash_and_qwen27():
    assert "Flash" in ols.SCHOOLS
    assert "Qwen27" in ols.SCHOOLS
    assert ols.school_of_model("Qwen3.8-27B") == "Qwen27"
    assert ols.school_of_model("Qwen/Qwen3.8-Flash-Next") == "Flash"
    assert ols.school_of_model("qwen3.8-27b-abliterated") == "Qwen27"


def test_seed_store_does_not_invent_and_covers_the_school():
    laws, report = ols.seed_store()
    assert laws, "seed produced no laws; a listed receipt failed to load"
    ids = [law.law_id for law in laws]
    assert len(ids) == len(set(ids))
    for law in laws:
        ols.validate_law(law)
        assert law.time_to_first_useful_executable_ns is None
        assert law.scope in ols.SCOPES
        assert law.evidence_strength in ols.EVIDENCE_STRENGTHS
        assert 0.0 <= law.transfer_confidence["value"] <= 1.0
        for ref in law.evidence_refs:
            assert ref.endswith(".json"), ref
    presence = ols.school_presence(laws)
    assert presence["both_schools_are_sources"], presence
    assert presence["both_schools_are_targets"], presence
    # No GENERIC_VERIFIED from seed: sidecar has no protected evidence.
    assert all(law.scope != "GENERIC_VERIFIED" for law in laws)
    assert report["seed_receipts_found"]["receipts/headless/ODYSSEY_TRANSFER_PROVEN.json"]
    assert report["seed_receipts_found"]["receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json"]
    assert report["seed_receipts_found"]["tools/foundry/NEGATIVE_TRANSFER_ATLAS.json"]
    # The atlas is uncommitted, so it is invisible from a sparse lane worktree and
    # visible from the primary one. Its presence is an environment fact, not a
    # property of the law store -- assert the store COPES either way rather than
    # pinning the environment the module happened to be written in.
    atlas_key = "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json"
    assert isinstance(report["seed_receipts_found"][atlas_key], bool)
    assert any("ACCELERATOR_ARCHITECTURE_ATLAS" in n for n in report["notes"]), (
        "the store must say which vocabulary it seeded from, present or absent"
    )


def test_accounting_tracks_expected_versus_actual():
    laws, _ = ols.seed_store()
    by_id = {law.law_id: law for law in laws}
    cold = by_id["LAW-COLD-CONTROL-BEAT-TRANSFER-SEED"]
    assert cold.actual_saved_experiments == -8
    assert cold.expected_saved_experiments is None
    doctor = by_id["LAW-QWEN-FAILURE-NEVER-PRUNES"]
    assert doctor.expected_saved_experiments == 219
    assert doctor.actual_saved_experiments == 129
    summary = ols.accounting_summary(laws)
    assert summary["null_time_to_first_useful_executable_ns"] is True
    assert summary["sum_actual_saved_experiments"] is not None


def test_fitted_affine_is_generic_candidate_not_verified():
    laws, _ = ols.seed_store()
    affine = {law.law_id: law for law in laws}["LAW-FITTED-AFFINE-BEATS-RTN"]
    assert affine.scope == "GENERIC_CANDIDATE"
    assert affine.evidence_strength != "PROTECTED_ABSOLUTE"
    # Climbing the last step without protected evidence still refuses.
    with pytest.raises(ols.ScopeViolation) as ei:
        ols.promote(
            affine,
            "GENERIC_VERIFIED",
            {
                "models": ["qwen3.8-27b-abliterated", "Qwen/Qwen3-30B-A3B", "tiiuae/Falcon-H1-7B-Instruct"],
                "architecture_families": [
                    "dense_hybrid_transformer",
                    "qwen3_moe",
                    "falcon_h1",
                ],
                "evidence_strength": "DIAGNOSTIC_RELATIVE",
                "counterexample_discharged": True,
            },
        )
    assert ei.value.reason == "need_protected_or_reproduced"


def test_mlp_floor_stays_model_local_because_it_was_refuted_inside_family():
    laws, _ = ols.seed_store()
    floor = {law.law_id: law for law in laws}["LAW-MLP-FLOOR-2.25"]
    assert floor.scope == "MODEL_LOCAL"
    assert "already refuted" in floor.counterexample_requirement


def test_build_emits_sealed_receipt():
    out = ols.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "ODYSSEY2_LAW_STORE.json"
    assert doc["schema"] == "hawking.future.odyssey2_law_store.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["laws"]
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["school_presence"]["both_schools_are_sources"]
    assert doc["school_presence"]["both_schools_are_targets"]
    for law in doc["laws"]:
        assert set(law) == set(ols.LAW_FIELDS)
        assert law["time_to_first_useful_executable_ns"] is None
        assert "tps" not in law
        assert "bandwidth_gbps" not in law
        assert "gpu_ns" not in law


def test_selftest_aliases_build():
    assert ols.selftest is ols.build or callable(ols.selftest)
    out = ols.selftest()
    assert out.name == "ODYSSEY2_LAW_STORE.json"


def test_write_receipt_still_refuses_a_hardware_number():
    # Guard that we did not weaken _common: a tps in the document must raise.
    with pytest.raises(HardwareClaimError):
        from tools.future._common import write_receipt

        write_receipt(
            "SHOULD_NOT_EXIST.json",
            {"schema": "test", "version": 1, "tps": 12.0},
            "tools/future/test_odyssey2_law_store.py",
        )


def test_failed_atlas_index_excludes_live_layer_zero():
    failed = ols.failed_atlas_entries()
    assert "inter_expert_redundancy" in failed
    assert "expert_merging_omitted_from_survivors" in failed
    assert "layer_zero_is_a_different_source" not in failed


# ---------------------------------------------------------------------------
# G007 consumer: build records the five causality fields.
# ---------------------------------------------------------------------------


def test_build_records_the_five_causality_fields():
    """A coverage number no test defends will drift back to zero."""
    out = ols.build()
    doc = json.loads(out.read_text())
    assert ols.records_five_fields(doc)
    src = pathlib.Path(ols.__file__).read_text()
    assert "sc.emit(" in src
    assert doc["schools"]["Flash"]["physical_status"] == "metadata_only_weights_not_present"
    assert "SCHOOLS" in doc["probe_performed"] or "physical_status" in doc["probe_performed"]
    assert doc["direct_observation"] != "WEIGHTS_NOT_PRESENT"
    assert "physical_status=" in doc["direct_observation"]
    assert doc["causality_verdict"] in {sc.SUPPORTED, sc.OVERREACHING, sc.UNTESTED}


def test_unsupplied_observation_records_untested_not_a_restatement():
    result = {"schools": {"Flash": {"physical_status": "metadata_only_weights_not_present"}}}
    rec = ols.record_law_store_causality(
        result, probe_performed="", direct_observation=""
    )
    assert rec["verdict"] == sc.UNTESTED
    assert rec["direct_observation"] in ("", None)
    assert rec["direct_observation"] != "WEIGHTS_NOT_PRESENT"
    assert rec["direct_observation"] != "metadata_only_weights_not_present"
    assert result["schools"]["Flash"]["physical_status"] == "metadata_only_weights_not_present"
    assert rec["interpretation"] != rec["direct_observation"]


def test_overreaching_does_not_override_flash_physical_status(monkeypatch):
    def overreach(status, **kwargs):
        return {
            "probe_performed": kwargs.get("probe_performed") or "p",
            "direct_observation": kwargs.get("direct_observation") or "o",
            "interpretation": kwargs.get("interpretation") or status,
            "confidence": {
                "level": "LOW",
                "about": "a",
                "would_raise": "b",
                "would_lower": "c",
            },
            "alternatives": [
                {
                    "hypothetical": "h",
                    "consistent_with_observation": True,
                    "consistent_with_claim": False,
                }
            ],
            "verdict": sc.OVERREACHING,
            "falsifier": "f",
            "probe_kind": sc.PROBE_METADATA,
            "claim_kind": sc.CLAIM_OBJECT_ABSENCE,
        }

    monkeypatch.setattr(ols.sc, "emit", overreach)
    out = ols.build()
    doc = json.loads(out.read_text())
    assert doc["schools"]["Flash"]["physical_status"] == "metadata_only_weights_not_present"
    assert doc["causality_verdict"] == sc.OVERREACHING
    assert ols.SCHOOLS["Flash"]["physical_status"] == "metadata_only_weights_not_present"


def test_coverage_receipt_names_odyssey2_law_store_as_recording():
    path = RECEIPTS / "STATUS_CAUSALITY_COVERAGE.json"
    doc = json.loads(path.read_text())
    assert "odyssey2_law_store" in doc["recording_five_fields"]
    assert doc["n_gates"] == 18

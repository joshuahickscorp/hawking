"""Tests for the Qwen27 accelerator profile schema.

The load-bearing guard is the negative control: a profile missing
active_byte_model must be REJECTED with that field named.
"""
from __future__ import annotations

import copy
import json

import pytest

from tools.future import qwen27_profile_schema as qs
from tools.future._common import HARDWARE_FIELDS, RECEIPTS


def _complete() -> dict:
    return qs.control_profile()


def test_build_emits_sealed_receipt():
    out = qs.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "QWEN27_ACCELERATOR_PROFILE_SCHEMA.json"
    assert doc["schema"] == "hawking.future.qwen27_profile.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert doc["control_profile_validation"]["status"] == "ACCEPTED"
    assert doc["negative_control"]["fires"] is True


def test_selftest_emits_sealed_receipt_and_keeps_the_guard():
    out = qs.selftest()
    doc = json.loads(out.read_text())
    assert doc["seal_sha256"]
    assert doc["negative_control"]["fires"] is True


def test_complete_profile_is_accepted():
    profile = _complete()
    result = qs.validate_profile(profile)
    assert result["status"] == "ACCEPTED"
    assert result["missing_fields"] == []
    qs.accept_profile(profile)  # must not raise


def test_missing_active_byte_model_is_rejected_with_field_named():
    """Negative control: the refusal must actually fire, and name the field."""
    profile = _complete()
    assert "active_byte_model" in profile
    del profile["active_byte_model"]

    result = qs.validate_profile(profile)
    assert result["status"] == "REJECTED"
    assert "active_byte_model" in result["missing_fields"]
    assert "active_byte_model" in result["named_refusal"]
    assert result["named_refusal"].startswith("REJECTED:")

    with pytest.raises(qs.ProfileRejectedError) as caught:
        qs.accept_profile(profile)
    assert "active_byte_model" in str(caught.value)
    assert caught.value.missing_fields[0] == "active_byte_model" or (
        "active_byte_model" in caught.value.missing_fields
    )


def test_complete_and_incomplete_are_not_comparable():
    complete = _complete()
    incomplete = copy.deepcopy(complete)
    del incomplete["active_byte_model"]
    assert qs.profiles_comparable(complete, complete) is True
    assert qs.profiles_comparable(complete, incomplete) is False
    assert qs.profiles_comparable(incomplete, incomplete) is False


def test_empty_active_byte_model_is_rejected_naming_subfields():
    profile = _complete()
    profile["active_byte_model"] = {}
    result = qs.validate_profile(profile)
    assert result["status"] == "REJECTED"
    named = result["missing_fields"]
    assert any(f.startswith("active_byte_model.") for f in named)
    assert "active_byte_model.actual_read_bytes_per_token" in named
    assert "active_byte_model.activations_included" in named


def test_missing_nested_actual_read_bytes_is_rejected():
    profile = _complete()
    del profile["active_byte_model"]["actual_read_bytes_per_token"]
    result = qs.validate_profile(profile)
    assert result["status"] == "REJECTED"
    assert "active_byte_model.actual_read_bytes_per_token" in result["missing_fields"]
    with pytest.raises(qs.ProfileRejectedError) as caught:
        qs.accept_profile(profile)
    assert "active_byte_model.actual_read_bytes_per_token" in str(caught.value)


def test_every_required_section_absence_is_rejected_by_name():
    complete = _complete()
    for section in qs.REQUIRED_SECTIONS:
        profile = copy.deepcopy(complete)
        del profile[section]
        result = qs.validate_profile(profile)
        assert result["status"] == "REJECTED", section
        assert section in result["missing_fields"], section


def test_unknown_is_accepted_in_measurement_slots_but_absent_is_not():
    profile = _complete()
    profile["active_byte_model"]["actual_read_bytes_per_token"] = "UNKNOWN"
    profile["active_byte_model"]["transient_bytes_per_token"] = None
    assert qs.validate_profile(profile)["status"] == "ACCEPTED"
    del profile["resident_resource_model"]["resident_bytes"]
    result = qs.validate_profile(profile)
    assert result["status"] == "REJECTED"
    assert "resident_resource_model.resident_bytes" in result["missing_fields"]


def test_invalid_claim_class_and_role_are_rejected():
    profile = _complete()
    profile["identity"]["claim_class"] = "PRETTY_SURE"
    result = qs.validate_profile(profile)
    assert result["status"] == "REJECTED"
    assert "identity.claim_class" in result["missing_fields"]

    profile = _complete()
    profile["identity"]["role"] = "WINNER"
    result = qs.validate_profile(profile)
    assert result["status"] == "REJECTED"
    assert "identity.role" in result["missing_fields"]


def test_transfer_law_tagging_covers_every_section_with_legal_scope():
    profile = _complete()
    tags = profile["transfer_law_tagging"]["by_section"]
    for section in qs.REQUIRED_SECTIONS:
        assert section in tags
        assert tags[section]["scope"] in qs.TRANSFER_SCOPES
        for extra in tags[section]["secondary_scopes"]:
            assert extra in qs.TRANSFER_SCOPES
    tags["active_byte_model"]["scope"] = "GALAXY_LOCAL"
    result = qs.validate_profile(profile)
    assert result["status"] == "REJECTED"
    assert "transfer_law_tagging.by_section.active_byte_model.scope" in result["missing_fields"]


def test_organs_and_metrics_must_be_the_budget_set():
    profile = _complete()
    profile["token_ns_receipt_decomposition"]["organs"] = ["mlp"]
    result = qs.validate_profile(profile)
    assert result["status"] == "REJECTED"
    assert "token_ns_receipt_decomposition.organs.attention" in result["missing_fields"]

    profile = _complete()
    profile["token_ns_receipt_decomposition"]["system_ledger_fields"] = ["fallback_count"]
    result = qs.validate_profile(profile)
    assert result["status"] == "REJECTED"
    assert (
        "token_ns_receipt_decomposition.system_ledger_fields.actual_read_bytes_per_token"
        in result["missing_fields"]
    )


def test_token_ns_budget_projection_names_satisfied_and_unsatisfied_sections():
    budget = {
        "schema": "hawking.accelerator.qwen27_token_ns_budget.v1",
        "status": "PLANNED_UNTIL_NATIVE_PROTECTED_EXECUTION",
        "model": "qwen3.8-27b-sealed-3.14",
        "baseline": {
            "profile": "hcli/hawking-native.sealed-3.14.json",
            "representation": "native-packed sealed control",
        },
        "organs": [{"organ": name, "source_weight_bytes_per_token": None} for name in qs.REQUIRED_ORGANS],
        "system_ledger": {name: None for name in qs.REQUIRED_METRICS},
        "measurement_protocol": {flag: True for flag in qs.MEASUREMENT_PROTOCOL_FLAGS},
        "lifecycle_buckets": {name: None for name in qs.LIFECYCLE_BUCKETS},
        "source_byte_denominator": {
            "active_weight_bytes_per_token": 9878901136,
            "complete_ebpw": 3.139300850311054,
            "regions": [{"kernel": "qwen_affine_q2_group32_matvec_geo_tpr64_tg128"}],
            "claim_boundary": (
                "catalog-derived weight traffic only; activations, KV, and recurrent state are not included"
            ),
        },
        "promotion_allowed": False,
    }
    assessment = qs.assess_token_ns_budget(budget)
    assert assessment["status"] == "ASSESSED"
    assert "token_ns_receipt_decomposition" in assessment["satisfied_sections"]
    assert "cold_warm_lifecycle" in assessment["satisfied_sections"]
    for section in (
        "identity",
        "active_byte_model",
        "dispatch_topology",
        "layout_search_space",
        "candidate_geometry",
        "representation_census",
        "resident_resource_model",
        "transfer_law_tagging",
        "capability_workload_design",
        "regression_fixtures",
    ):
        assert section in assessment["unsatisfied_sections"], section
    assert budget["schema"] == assessment["schema"]
    assert any("activations" in f for f in assessment["findings"])

    # A budget is not a profile. Passing it to validate_profile must REJECT.
    as_profile = qs.validate_profile(budget)
    assert as_profile["status"] == "REJECTED"
    assert "active_byte_model" in as_profile["missing_fields"]


def test_absent_budget_is_reported_not_invented():
    assessment = qs.assess_token_ns_budget(None)
    assert assessment["status"] == "RECEIPT_ABSENT"
    assert assessment["satisfied_sections"] == []
    assert assessment["unsatisfied_sections"] == list(qs.REQUIRED_SECTIONS)


def test_recovered_budget_shape_reports_the_known_gaps():
    _src, atlas = qs.load_authority(qs.ATLAS_REL)
    shape = qs.recovered_budget_shape(atlas)
    assessment = qs.assess_token_ns_budget(shape)
    assert assessment["schema"] == "hawking.accelerator.qwen27_token_ns_budget.v1"
    assert "token_ns_receipt_decomposition" in assessment["satisfied_sections"]
    assert "cold_warm_lifecycle" in assessment["satisfied_sections"]
    assert "active_byte_model" in assessment["unsatisfied_sections"]
    assert "layout_search_space" in assessment["unsatisfied_sections"]
    assert "transfer_law_tagging" in assessment["unsatisfied_sections"]
    assert shape["control_observation_metrics_copied"] is False


def test_incumbent_is_control_not_target_and_avoids_hardware_field_keys():
    doc = qs.build_document()
    control = doc["incumbent_control"]
    assert control["role"] == "CONTROL"
    assert control["not_a_target"] is True
    assert control["not_a_ceiling"] is True
    assert control["historical_physical_ebpw"]["approx_label"] == "~3.14"
    assert control["historical_accepted_tokens_per_second_record"]["approx_label"] == "~25"
    assert control["historical_accepted_tokens_per_second_record"]["do_not_promote"] is True

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                    raise AssertionError(f"hardware field {key}={value!r} in incumbent/schema receipt")
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(doc)
    # The ~25 record must not be stored under accepted_tps / tps.
    record = control["historical_accepted_tokens_per_second_record"]
    assert "accepted_tps" not in record
    assert "tps" not in record
    assert "recorded_complete_tokens_per_second" in record


def test_nomenclature_preserves_five_eras_and_three_odysseys():
    doc = qs.build_document()
    nom = doc["nomenclature"]
    assert len(nom["eras"]) == 5
    assert "VI" not in "".join(nom["eras"])
    assert nom["no_era_vi"] is True
    assert len(nom["odysseys"]) == 3
    assert nom["no_odyssey_iv"] is True
    assert "civilization" in nom["fpga"].lower() or "not its own" in nom["fpga"].lower()
    assert "never promotes" in nom["diagnostic_relative"].lower()


def test_no_partial_credit_status():
    profile = _complete()
    del profile["layout_search_space"]
    del profile["regression_fixtures"]
    result = qs.validate_profile(profile)
    assert result["status"] == "REJECTED"
    assert result["status"] != "PARTIAL"
    assert "layout_search_space" in result["missing_fields"]
    assert "regression_fixtures" in result["missing_fields"]

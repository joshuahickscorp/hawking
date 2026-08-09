"""CPU-only regression coverage for the Qwen80 HCLI-adapter readiness gate."""
from __future__ import annotations

from copy import deepcopy

from lab.operators import ascension_qwen80_hcli_adapter_readiness as readiness
from lab.receipts import seal


def _sha(character: str) -> str:
    return character * 64


def _current() -> dict[str, object]:
    return {
        "model_id": readiness.MODEL_ID,
        "model_key": readiness.MODEL_KEY,
        "source_repository": readiness.SOURCE_REPOSITORY,
        "source_revision": readiness.SOURCE_REVISION,
        "manifest_seal_sha256": readiness.MANIFEST_SEAL,
        "admission_receipt_seal_sha256": readiness.ADMISSION_RECEIPT_SEAL,
        "runtime_receipt_seal_sha256": _sha("a"),
        "runtime_executable_sha256": _sha("b"),
    }


def _reseal(document: dict[str, object]) -> dict[str, object]:
    body = deepcopy(document)
    body.pop("seal_sha256", None)
    return seal(body)


def _decoder(current: dict[str, object]) -> dict[str, object]:
    return {
        "schema": readiness.DECODER_SCHEMA,
        "status": readiness.DECODER_STATUS,
        "complete_decoder_readiness_earned": True,
        "real_gravity_server_launch_precondition_satisfied": True,
        "input_schema_valid": True,
        "source_artifact_binding_valid": True,
        "exact_48_layer_schedule_valid": True,
        "missing_operator_classes_or_layers": [],
        "source_artifact_binding": {
            key: current[key]
            for key in (
                "model_id",
                "model_key",
                "source_repository",
                "source_revision",
                "manifest_seal_sha256",
                "admission_receipt_seal_sha256",
            )
        },
    }


def _state_contract() -> dict[str, object]:
    return {
        "schema": readiness.STATE_CONTRACT_SCHEMA,
        "status": readiness.STATE_CONTRACT_STATUS,
        "complete_decoder_readiness_earned": False,
        "source_archaeology": {
            "source_repository": readiness.SOURCE_REPOSITORY,
            "source_revision": readiness.SOURCE_REVISION,
            "layer_count": 48,
            "deltanet_layers": 36,
            "gqa_layers": 12,
        },
        "execution_boundary": {
            "not_runtime": True,
            "no_model_token_execution": True,
            "no_hcli_execution": True,
            "no_tps_or_tg_measurement": True,
        },
        "fixture_contract_checks": {
            "exact_schedule_checked": True,
            "state_slot_aliasing_checked": True,
            "causal_position_and_update_order_checked": True,
            "restart_identity_checked": True,
            "rollback_identity_checked": True,
            "cross_session_leakage_rejected": True,
            "lm_head_reserved_tail_token_rejected": True,
        },
    }


def _tokenizer_sampler() -> dict[str, object]:
    return {
        "schema": readiness.TOKENIZER_SCHEMA,
        "status": readiness.TOKENIZER_STATUS,
        "component_only": True,
        "source_binding": {
            "source_repository": readiness.SOURCE_REPOSITORY,
            "source_revision_from_pre_admitted_source_audit": readiness.SOURCE_REVISION,
            "tokenizer_addressable_vocab_size": readiness.TOKENIZER_VOCAB,
            "lm_head_vocab_size": readiness.LM_HEAD_VOCAB,
            "reserved_lm_head_tail_rows": readiness.RESERVED_TAIL_ROWS,
        },
        "sampler_fixture": {
            "reserved_tail_mask_cutoff": readiness.TOKENIZER_VOCAB,
            "all_selected_reserved_fixture_logits_are_negative_infinity": True,
            "sampled_id_is_tokenizer_addressable": True,
            "sampled_feedback_token_id": readiness.TOKENIZER_VOCAB - 1,
        },
        "rejection_tests": {
            "reserved_prompt_token_rejected": True,
            "sample_before_tail_mask_rejected": True,
            "wrong_tail_mask_cutoff_rejected": True,
            "reserved_tail_feedback_rejected": True,
        },
    }


def _full_runtime(current: dict[str, object]) -> dict[str, object]:
    return seal(
        {
            "schema": readiness.FULL_RUNTIME_SCHEMA,
            "status": readiness.FULL_RUNTIME_STATUS,
            "current_receipt": current,
            "source_bound": True,
            "artifact_bound": True,
            "full_runtime": True,
            "complete_token_path": True,
            "full_48_layer_token_executed": True,
            "all_36_deltanet_layers_executed": True,
            "all_12_gqa_layers_executed": True,
            "final_norm_lm_head_tail_mask_sampler_executed": True,
            "fixture_only": False,
            "component_only": False,
            "synthetic_input": False,
            "fallback_used": False,
            "shadow_model_used": False,
            "raw_bf16_or_mps_fallback_used": False,
            "hcli_execution_performed": False,
            "tps_or_tg_measurement_performed": False,
        }
    )


def _session_kv(current: dict[str, object]) -> dict[str, object]:
    return seal(
        {
            "schema": readiness.SESSION_KV_SCHEMA,
            "status": readiness.SESSION_KV_STATUS,
            "current_receipt": current,
            "source_bound": True,
            "artifact_bound": True,
            "complete_token_path": True,
            "real_device_resident_state": True,
            "all_36_deltanet_state_slots_bound": True,
            "all_12_gqa_kv_slots_bound": True,
            "current_position_kv_append_then_causal_read_verified": True,
            "no_cross_session_state_or_kv_leakage": True,
            "restart_passed": True,
            "rollback_passed": True,
            "fixture_only": False,
            "component_only": False,
            "synthetic_input": False,
            "fallback_used": False,
            "observed_session_ids": ["q80-session-a", "q80-session-b"],
            "deltanet_state_bytes": 19_759_104,
            "gqa_key_cache_bytes": 49_152,
            "gqa_value_cache_bytes": 49_152,
        }
    )


def _telemetry(current: dict[str, object]) -> dict[str, object]:
    return seal(
        {
            "schema": readiness.TELEMETRY_SCHEMA,
            "status": readiness.TELEMETRY_STATUS,
            "current_receipt": current,
            "port_available_checked": True,
            "no_existing_listener": True,
            "telemetry_schema_validated": True,
            "session_metrics_bound_to_session_id": True,
            "state_kv_metrics_bound_to_current_receipt": True,
            "server_started": False,
            "hcli_request_executed": False,
            "tps_or_tg_measurement_performed": False,
            "proposed_host": "127.0.0.1",
            "proposed_port": 18_480,
            "telemetry_fields": sorted(readiness.TELEMETRY_FIELDS),
        }
    )


def _qualified_bundle() -> dict[str, object]:
    current = _current()
    return {
        "schema": readiness.INPUT_SCHEMA,
        "current_receipt": current,
        "decoder_readiness": _decoder(current),
        "state_kv_contract": _state_contract(),
        "tokenizer_template_sampler": _tokenizer_sampler(),
        "full_runtime_receipt": _full_runtime(current),
        "session_kv_receipt": _session_kv(current),
        "telemetry_receipt": _telemetry(current),
    }


def _condition(report: dict[str, object], name: str) -> dict[str, object]:
    return next(item for item in report["conditions"] if item["name"] == name)  # type: ignore[index]


def test_missing_full_runtime_stays_not_ready_no_server() -> None:
    bundle = _qualified_bundle()
    del bundle["full_runtime_receipt"]
    report = readiness.assess_adapter_readiness(bundle)
    assert report["status"] == readiness.NOT_READY_STATUS
    assert report["hcli_adapter_launch_precondition_satisfied"] is False
    assert _condition(report, "future_exact_full_runtime_token_no_fallback")["satisfied"] is False
    assert report["claim_boundary"]["hcli_or_tps_earned_by_this_result"] is False  # type: ignore[index]


def test_tail_masking_evidence_cannot_be_weakened() -> None:
    bundle = _qualified_bundle()
    sampler = bundle["tokenizer_template_sampler"]["sampler_fixture"]  # type: ignore[index]
    sampler["all_selected_reserved_fixture_logits_are_negative_infinity"] = False
    report = readiness.assess_adapter_readiness(bundle)
    assert report["status"] == readiness.NOT_READY_STATUS
    assert _condition(report, "tokenizer_template_tail_mask_component_contract")["satisfied"] is False


def test_duplicate_logical_session_identity_is_rejected() -> None:
    bundle = _qualified_bundle()
    session = deepcopy(bundle["session_kv_receipt"])
    session["observed_session_ids"] = ["q80-session-a", "q80-session-a"]
    bundle["session_kv_receipt"] = _reseal(session)
    report = readiness.assess_adapter_readiness(bundle)
    assert report["status"] == readiness.NOT_READY_STATUS
    condition = _condition(report, "future_session_kv_restart_rollback")
    assert condition["satisfied"] is False
    assert any("distinct" in item for item in condition["blockers"])


def test_missing_rollback_is_rejected_even_with_restart() -> None:
    bundle = _qualified_bundle()
    session = deepcopy(bundle["session_kv_receipt"])
    session["rollback_passed"] = False
    bundle["session_kv_receipt"] = _reseal(session)
    report = readiness.assess_adapter_readiness(bundle)
    assert report["status"] == readiness.NOT_READY_STATUS
    condition = _condition(report, "future_session_kv_restart_rollback")
    assert any("rollback_passed" in item for item in condition["blockers"])


def test_mismatched_current_receipt_is_rejected() -> None:
    bundle = _qualified_bundle()
    runtime = deepcopy(bundle["full_runtime_receipt"])
    runtime["current_receipt"]["runtime_receipt_seal_sha256"] = _sha("c")
    bundle["full_runtime_receipt"] = _reseal(runtime)
    report = readiness.assess_adapter_readiness(bundle)
    assert report["status"] == readiness.NOT_READY_STATUS
    condition = _condition(report, "future_exact_full_runtime_token_no_fallback")
    assert any("does not match selected current" in item for item in condition["blockers"])


def test_fixture_or_component_runtime_receipt_cannot_be_promoted() -> None:
    bundle = _qualified_bundle()
    runtime = deepcopy(bundle["full_runtime_receipt"])
    runtime["fixture_only"] = True
    runtime["component_only"] = True
    bundle["full_runtime_receipt"] = _reseal(runtime)
    report = readiness.assess_adapter_readiness(bundle)
    assert report["status"] == readiness.NOT_READY_STATUS
    condition = _condition(report, "future_exact_full_runtime_token_no_fallback")
    assert any("fixture_only" in item for item in condition["blockers"])
    assert any("component_only" in item for item in condition["blockers"])


def test_fully_qualified_synthetic_contract_is_prelaunch_ready_only() -> None:
    report = readiness.assess_adapter_readiness(_qualified_bundle())
    assert report["status"] == readiness.READY_STATUS
    assert report["hcli_adapter_launch_precondition_satisfied"] is True
    assert report["server_may_be_started_by_separate_controlled_launcher"] is True
    assert report["blockers"] == []
    assert report["claim_boundary"]["this_validator_started_no_server"] is True  # type: ignore[index]
    assert report["claim_boundary"]["this_validator_executed_no_hcli_request"] is True  # type: ignore[index]
    assert report["claim_boundary"]["this_validator_measured_no_tps_or_tg"] is True  # type: ignore[index]

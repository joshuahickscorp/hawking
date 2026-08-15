"""CPU-only tests for the one-resident-Q80-server activation gate."""
from __future__ import annotations

from copy import deepcopy

from lab.operators import ascension_qwen80_resident_server_activation_gate as gate
from lab.receipts import seal


def _sha(character: str) -> str:
    return character * 64


def _current() -> dict[str, object]:
    return {
        "model_id": gate.MODEL_ID,
        "model_key": gate.MODEL_KEY,
        "source_repository": gate.SOURCE_REPOSITORY,
        "source_revision": gate.SOURCE_REVISION,
        "manifest_seal_sha256": gate.MANIFEST_SEAL,
        "admission_receipt_seal_sha256": gate.ADMISSION_RECEIPT_SEAL,
        "runtime_receipt_seal_sha256": _sha("a"),
        "runtime_executable_sha256": _sha("b"),
    }


def _reseal(document: dict[str, object]) -> dict[str, object]:
    body = deepcopy(document)
    body.pop("seal_sha256", None)
    return seal(body)


def _decoder(current: dict[str, object]) -> dict[str, object]:
    return seal(
        {
            "schema": gate.DECODER_SCHEMA,
            "status": gate.DECODER_STATUS,
            "complete_decoder_readiness_earned": True,
            "real_gravity_server_launch_precondition_satisfied": True,
            "input_schema_valid": True,
            "source_artifact_binding_valid": True,
            "exact_48_layer_schedule_valid": True,
            "missing_operator_classes_or_layers": [],
            "source_artifact_binding": {key: current[key] for key in gate.SOURCE_IDENTITY_KEYS},
        }
    )


def _runtime(current: dict[str, object]) -> dict[str, object]:
    return seal(
        {
            "schema": gate.FULL_RUNTIME_SCHEMA,
            "status": gate.FULL_RUNTIME_STATUS,
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
            "schema": gate.SESSION_KV_SCHEMA,
            "status": gate.SESSION_KV_STATUS,
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
            "observed_session_ids": ["alpha", "beta"],
            "deltanet_state_bytes": 1_048_576,
            "gqa_key_cache_bytes": 2_097_152,
            "gqa_value_cache_bytes": 2_097_152,
        }
    )


def _terminal(current: dict[str, object]) -> dict[str, object]:
    return seal(
        {
            "schema": gate.TERMINAL_SCHEMA,
            "status": gate.TERMINAL_STATUS,
            "current_receipt": current,
            "source_bound": True,
            "artifact_bound": True,
            "post_48_hidden_device_parity_passed": True,
            "final_rmsnorm_device_parity_passed": True,
            "lm_head_all_rows_device_parity_passed": True,
            "reserved_tail_mask_applied_before_sample": True,
            "deterministic_sample_and_feedback_executed": True,
            "sampled_token_is_tokenizer_addressable": True,
            "fixture_only": False,
            "synthetic_input": False,
            "fallback_used": False,
            "shadow_model_used": False,
        }
    )


def _hcli_prelaunch(current: dict[str, object]) -> dict[str, object]:
    return seal(
        {
            "schema": gate.HCLI_PRELAUNCH_SCHEMA,
            "status": gate.HCLI_PRELAUNCH_STATUS,
            "current_receipt": current,
            "port_available_checked": True,
            "no_existing_listener": True,
            "telemetry_schema_validated": True,
            "session_metrics_bound_to_session_id": True,
            "state_kv_metrics_bound_to_current_receipt": True,
            "hcli_transport_contract_validated": True,
            "logical_session_multiplexing_preflight": True,
            "server_started": False,
            "hcli_request_executed": False,
            "tps_or_tg_measurement_performed": False,
            "proposed_host": gate.DEFAULT_HOST,
            "proposed_port": gate.DEFAULT_PORT,
        }
    )


def _memory(current: dict[str, object]) -> dict[str, object]:
    return seal(
        {
            "schema": gate.MEMORY_SCHEMA,
            "status": gate.MEMORY_STATUS,
            "current_receipt": current,
            "measured_on_host": True,
            "memory_envelope_healthy": True,
            "one_q80_process_envelope": True,
            "co_resident_envelope_accounted_for": True,
            "resident_q80_model_processes": 1,
            "resident_q80_rss_bytes": 34 * 1024**3,
            "available_memory_bytes": 46 * 1024**3,
            "minimum_required_available_bytes": 20 * 1024**3,
            "swap_used_bytes": 0,
        }
    )


def _topology() -> dict[str, object]:
    return {
        "model_key": gate.MODEL_KEY,
        "bind_host": gate.DEFAULT_HOST,
        "bind_port": gate.DEFAULT_PORT,
        "desired_q80_model_processes": 1,
        "existing_q80_model_processes": 0,
        "server_process_starts_per_activation": 1,
        "duplicate_q80_processes_prohibited": True,
        "listener_absent_prelaunch": True,
        "single_resident_process_many_logical_sessions": True,
        "logical_session_state_isolated": True,
        "maximum_logical_sessions": 32,
    }


def _rollback(current: dict[str, object]) -> dict[str, object]:
    return seal(
        {
            "schema": gate.ROLLBACK_SCHEMA,
            "status": gate.ROLLBACK_STATUS,
            "current_receipt": current,
            "automatic_launch_only_after_all_conditions_pass": True,
            "launch_exactly_one_q80_process": True,
            "record_child_pid_before_health": True,
            "rollback_on_health_identity_mismatch": True,
            "rollback_on_session_kv_leak": True,
            "rollback_on_memory_envelope_breach": True,
            "release_loopback_port_after_child_exit": True,
            "terminal_rollback_receipt_written_last": True,
            "automatic_retry_same_activation_prohibited": True,
            "server_started": False,
            "rollback_executed": False,
            "hcli_request_executed": False,
        }
    )


def _evidence() -> dict[str, object]:
    current = _current()
    return {
        "schema": gate.INPUT_SCHEMA,
        "current_receipt": current,
        "decoder_readiness": _decoder(current),
        "full_runtime_receipt": _runtime(current),
        "session_kv_receipt": _session_kv(current),
        "terminal_receipt": _terminal(current),
        "hcli_prelaunch_receipt": _hcli_prelaunch(current),
        "memory_envelope_receipt": _memory(current),
        "launch_topology": _topology(),
        "rollback_prelaunch_receipt": _rollback(current),
    }


def _condition(report: dict[str, object], name: str) -> dict[str, object]:
    return next(condition for condition in report["conditions"] if condition["name"] == name)  # type: ignore[index,return-value]


def test_current_component_evidence_is_hard_refused_and_starts_nothing() -> None:
    report = gate.assess_activation(gate.current_component_evidence())
    assert report["status"] == gate.REFUSED_STATUS
    assert report["automatic_launch_eligible"] is False
    assert report["server_may_be_started_by_separate_controlled_launcher"] is False
    assert report["current_component_evidence_hard_refused"] is True
    assert _condition(report, "sealed_truthful_complete_decoder_readiness")["satisfied"] is False
    assert report["claim_boundary"]["gate_started_no_server"] is True
    assert report["claim_boundary"]["gate_bound_no_port"] is True


def test_complete_sealed_prelaunch_evidence_is_eligible_for_one_server_only() -> None:
    report = gate.assess_activation(_evidence())
    assert report["status"] == gate.ELIGIBLE_STATUS
    assert report["automatic_launch_eligible"] is True
    assert report["blockers"] == []
    assert report["target_topology"]["endpoint"] == {
        "host": "127.0.0.1",
        "port": 18_480,
    }
    assert report["automatic_launch_contract"]["processes_to_start"] == 1
    assert report["automatic_launch_contract"]["logical_sessions"] == "many_inside_the_single_resident_process"
    assert report["automatic_launch_contract"]["gate_starts_no_process"] is True
    assert report["rollback_contract"]["automatic_retry_same_activation_prohibited"] is True


def test_duplicate_q80_model_process_is_refused_instead_of_cloned() -> None:
    evidence = _evidence()
    evidence["launch_topology"]["existing_q80_model_processes"] = 1  # type: ignore[index]
    report = gate.assess_activation(evidence)
    assert report["automatic_launch_eligible"] is False
    topology = _condition(report, "one_q80_process_many_logical_sessions_topology")
    assert topology["satisfied"] is False
    assert any("existing_q80_model_processes" in blocker for blocker in topology["blockers"])
    assert report["claim_boundary"]["gate_started_no_server"] is True


def test_qwen30_port_reuse_and_hcli_port_drift_are_refused() -> None:
    evidence = _evidence()
    evidence["launch_topology"]["bind_port"] = gate.QWEN30_CONVENTIONAL_PORT  # type: ignore[index]
    hcli = deepcopy(evidence["hcli_prelaunch_receipt"])
    hcli["proposed_port"] = gate.QWEN30_CONVENTIONAL_PORT
    evidence["hcli_prelaunch_receipt"] = _reseal(hcli)
    report = gate.assess_activation(evidence)
    assert report["automatic_launch_eligible"] is False
    assert _condition(report, "sealed_hcli_prelaunch_on_q80_loopback_port")["satisfied"] is False
    topology = _condition(report, "one_q80_process_many_logical_sessions_topology")
    assert any("must not reuse Q30" in blocker for blocker in topology["blockers"])


def test_runtime_fallback_or_terminal_tail_order_blocks_activation() -> None:
    evidence = _evidence()
    runtime = deepcopy(evidence["full_runtime_receipt"])
    runtime["fallback_used"] = True
    evidence["full_runtime_receipt"] = _reseal(runtime)
    terminal = deepcopy(evidence["terminal_receipt"])
    terminal["reserved_tail_mask_applied_before_sample"] = False
    evidence["terminal_receipt"] = _reseal(terminal)
    report = gate.assess_activation(evidence)
    assert report["automatic_launch_eligible"] is False
    assert _condition(report, "sealed_exact_runtime_no_fallback")["satisfied"] is False
    assert _condition(report, "sealed_terminal_head_tail_mask_sampler")["satisfied"] is False


def test_session_kv_or_memory_envelope_failure_blocks_activation() -> None:
    evidence = _evidence()
    session = deepcopy(evidence["session_kv_receipt"])
    session["no_cross_session_state_or_kv_leakage"] = False
    evidence["session_kv_receipt"] = _reseal(session)
    memory = deepcopy(evidence["memory_envelope_receipt"])
    memory["swap_used_bytes"] = 4096
    evidence["memory_envelope_receipt"] = _reseal(memory)
    report = gate.assess_activation(evidence)
    assert report["automatic_launch_eligible"] is False
    assert _condition(report, "sealed_session_kv_state_and_rollback")["satisfied"] is False
    assert _condition(report, "sealed_measured_healthy_memory_envelope")["satisfied"] is False


def test_unsealed_or_identity_drifted_decoder_is_refused() -> None:
    evidence = _evidence()
    decoder = deepcopy(evidence["decoder_readiness"])
    decoder.pop("seal_sha256")
    decoder["source_artifact_binding"]["manifest_seal_sha256"] = _sha("c")
    evidence["decoder_readiness"] = decoder
    report = gate.assess_activation(evidence)
    condition = _condition(report, "sealed_truthful_complete_decoder_readiness")
    assert condition["satisfied"] is False
    assert any("invalid sealed receipt" in blocker for blocker in condition["blockers"])
    assert any("does not match current receipt" in blocker for blocker in condition["blockers"])

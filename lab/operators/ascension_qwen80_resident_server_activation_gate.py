#!/usr/bin/env python3
"""Fail-closed activation gate for one resident Qwen80 Gravity server.

This is deliberately a CPU-only pre-launch evaluator. It reads already
produced evidence and can write a new eligibility report, but it never opens a
model artifact, probes or binds a port, starts a process, or sends an HCLI
request. The controlled launcher that eventually consumes a positive result
must still make the one process start and write a terminal/rollback receipt.

The topology is intentionally non-negotiable:

* one resident Q80 model process on one loopback endpoint;
* many logical sessions inside that process;
* no duplicate Q80 model process to manufacture apparent parallelism.

No Q80-specific bind convention exists in the current repository; Q30 uses
``127.0.0.1:18430``, so the Q80 default is the distinct
``127.0.0.1:18480``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.receipts import SealIntegrityError, verify


INPUT_SCHEMA = "hawking.ascension.qwen80_resident_server_activation_input.v1"
RESULT_SCHEMA = "hawking.ascension.qwen80_resident_server_activation_result.v1"
REFUSED_STATUS = "REFUSED_QWEN80_ONE_RESIDENT_SERVER_ACTIVATION_NOT_READY_NO_SERVER"
ELIGIBLE_STATUS = "ELIGIBLE_QWEN80_ONE_RESIDENT_SERVER_AUTOMATIC_LAUNCH_PRECONDITION_ONLY"

MODEL_ID = "Qwen3-Coder-Next-80B"
MODEL_KEY = "qwen80"
SOURCE_REPOSITORY = "Qwen/Qwen3-Coder-Next"
SOURCE_REVISION = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb"
MANIFEST_SEAL = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b"
ADMISSION_RECEIPT_SEAL = "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18_480
QWEN30_CONVENTIONAL_PORT = 18_430

DECODER_SCHEMA = "hawking.ascension.qwen80_complete_decoder_readiness_result.v1"
DECODER_STATUS = "EARNED_QWEN80_COMPLETE_DECODER_READINESS_CONTRACT_ONLY"
FULL_RUNTIME_SCHEMA = "hawking.ascension.qwen80_full_hybrid_runtime_token_receipt.v1"
FULL_RUNTIME_STATUS = "EARNED_QWEN80_EXACT_FULL_HYBRID_RUNTIME_TOKEN_NO_FALLBACK"
SESSION_KV_SCHEMA = "hawking.ascension.qwen80_session_kv_restart_rollback_receipt.v1"
SESSION_KV_STATUS = "EARNED_QWEN80_SESSION_KV_RESTART_ROLLBACK_PARITY"
TERMINAL_SCHEMA = "hawking.ascension.qwen80_terminal_head_full_token_receipt.v1"
TERMINAL_STATUS = "EARNED_QWEN80_FINAL_NORM_LM_HEAD_TAIL_MASK_SAMPLE_FEEDBACK_NO_FALLBACK"
HCLI_PRELAUNCH_SCHEMA = "hawking.ascension.qwen80_server_telemetry_preflight.v1"
HCLI_PRELAUNCH_STATUS = "EARNED_QWEN80_SERVER_TELEMETRY_PRELAUNCH_CONTRACT"
MEMORY_SCHEMA = "hawking.ascension.qwen80_resident_memory_envelope_receipt.v1"
MEMORY_STATUS = "EARNED_QWEN80_RESIDENT_MEMORY_ENVELOPE_HEALTHY"
ROLLBACK_SCHEMA = "hawking.ascension.qwen80_resident_server_rollback_prelaunch.v1"
ROLLBACK_STATUS = "EARNED_QWEN80_ONE_RESIDENT_SERVER_ROLLBACK_PRELAUNCH"

CURRENT_IDENTITY_KEYS = (
    "model_id",
    "model_key",
    "source_repository",
    "source_revision",
    "manifest_seal_sha256",
    "admission_receipt_seal_sha256",
    "runtime_receipt_seal_sha256",
    "runtime_executable_sha256",
)
SOURCE_IDENTITY_KEYS = CURRENT_IDENTITY_KEYS[:6]


class ResidentActivationGateError(ValueError):
    """The input/output path or evidence grammar is unsafe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _mapping(value: object, label: str, errors: list[str]) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    errors.append(f"{label}: missing object")
    return None


def _expect(document: Mapping[str, Any], key: str, expected: object, label: str, errors: list[str]) -> None:
    if document.get(key) != expected:
        errors.append(f"{label}.{key}: expected {expected!r}, observed {document.get(key)!r}")


def _expect_true(document: Mapping[str, Any], key: str, label: str, errors: list[str]) -> None:
    if document.get(key) is not True:
        errors.append(f"{label}.{key}: must be true")


def _expect_false(document: Mapping[str, Any], key: str, label: str, errors: list[str]) -> None:
    if document.get(key) is not False:
        errors.append(f"{label}.{key}: must be false")


def _expect_sha256(document: Mapping[str, Any], key: str, label: str, errors: list[str]) -> None:
    if not _is_sha256(document.get(key)):
        errors.append(f"{label}.{key}: must be a lowercase SHA-256")


def _sealed(document: Mapping[str, Any], label: str, errors: list[str]) -> None:
    try:
        verify(document, label=label)
    except SealIntegrityError as exc:
        errors.append(f"{label}: invalid sealed receipt: {exc}")


def _validate_current_identity(value: object) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    current = _mapping(value, "current_receipt", errors)
    if current is None:
        return None, errors
    for key, expected in (
        ("model_id", MODEL_ID),
        ("model_key", MODEL_KEY),
        ("source_repository", SOURCE_REPOSITORY),
        ("source_revision", SOURCE_REVISION),
        ("manifest_seal_sha256", MANIFEST_SEAL),
        ("admission_receipt_seal_sha256", ADMISSION_RECEIPT_SEAL),
    ):
        _expect(current, key, expected, "current_receipt", errors)
    for key in ("runtime_receipt_seal_sha256", "runtime_executable_sha256"):
        _expect_sha256(current, key, "current_receipt", errors)
    return current, errors


def _validate_current_binding(
    document: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if document is None:
        errors.append(f"{label}: missing receipt")
        return errors
    binding = _mapping(document.get("current_receipt"), f"{label}.current_receipt", errors)
    if binding is None or current is None:
        return errors
    for key in CURRENT_IDENTITY_KEYS:
        if binding.get(key) != current.get(key):
            errors.append(f"{label}.current_receipt.{key}: does not match selected runtime identity")
    return errors


def _validate_decoder(value: object, current: Mapping[str, Any] | None) -> list[str]:
    errors: list[str] = []
    document = _mapping(value, "decoder_readiness", errors)
    if document is None:
        return errors
    _sealed(document, "decoder_readiness", errors)
    _expect(document, "schema", DECODER_SCHEMA, "decoder_readiness", errors)
    _expect(document, "status", DECODER_STATUS, "decoder_readiness", errors)
    for field in (
        "complete_decoder_readiness_earned",
        "real_gravity_server_launch_precondition_satisfied",
        "input_schema_valid",
        "source_artifact_binding_valid",
        "exact_48_layer_schedule_valid",
    ):
        _expect_true(document, field, "decoder_readiness", errors)
    if document.get("missing_operator_classes_or_layers") != []:
        errors.append("decoder_readiness.missing_operator_classes_or_layers: must be empty")
    binding = _mapping(document.get("source_artifact_binding"), "decoder_readiness.source_artifact_binding", errors)
    if binding is not None:
        for key in SOURCE_IDENTITY_KEYS:
            if current is not None and binding.get(key) != current.get(key):
                errors.append(f"decoder_readiness source binding {key}: does not match current receipt")
    return errors


def _validate_runtime(value: object, current: Mapping[str, Any] | None) -> list[str]:
    errors: list[str] = []
    document = _mapping(value, "full_runtime_receipt", errors)
    if document is None:
        return errors
    _sealed(document, "full_runtime_receipt", errors)
    _expect(document, "schema", FULL_RUNTIME_SCHEMA, "full_runtime_receipt", errors)
    _expect(document, "status", FULL_RUNTIME_STATUS, "full_runtime_receipt", errors)
    errors.extend(_validate_current_binding(document, current, "full_runtime_receipt"))
    for field in (
        "source_bound",
        "artifact_bound",
        "full_runtime",
        "complete_token_path",
        "full_48_layer_token_executed",
        "all_36_deltanet_layers_executed",
        "all_12_gqa_layers_executed",
        "final_norm_lm_head_tail_mask_sampler_executed",
    ):
        _expect_true(document, field, "full_runtime_receipt", errors)
    for field in (
        "fixture_only",
        "component_only",
        "synthetic_input",
        "fallback_used",
        "shadow_model_used",
        "raw_bf16_or_mps_fallback_used",
        "hcli_execution_performed",
        "tps_or_tg_measurement_performed",
    ):
        _expect_false(document, field, "full_runtime_receipt", errors)
    return errors


def _validate_session_kv(value: object, current: Mapping[str, Any] | None) -> list[str]:
    errors: list[str] = []
    document = _mapping(value, "session_kv_receipt", errors)
    if document is None:
        return errors
    _sealed(document, "session_kv_receipt", errors)
    _expect(document, "schema", SESSION_KV_SCHEMA, "session_kv_receipt", errors)
    _expect(document, "status", SESSION_KV_STATUS, "session_kv_receipt", errors)
    errors.extend(_validate_current_binding(document, current, "session_kv_receipt"))
    for field in (
        "source_bound",
        "artifact_bound",
        "complete_token_path",
        "real_device_resident_state",
        "all_36_deltanet_state_slots_bound",
        "all_12_gqa_kv_slots_bound",
        "current_position_kv_append_then_causal_read_verified",
        "no_cross_session_state_or_kv_leakage",
        "restart_passed",
        "rollback_passed",
    ):
        _expect_true(document, field, "session_kv_receipt", errors)
    for field in ("fixture_only", "component_only", "synthetic_input", "fallback_used"):
        _expect_false(document, field, "session_kv_receipt", errors)
    sessions = document.get("observed_session_ids")
    if (
        not isinstance(sessions, list)
        or len(sessions) < 2
        or not all(isinstance(session, str) and session for session in sessions)
        or len(set(sessions)) != len(sessions)
    ):
        errors.append("session_kv_receipt.observed_session_ids: requires at least two unique logical sessions")
    for field in ("deltanet_state_bytes", "gqa_key_cache_bytes", "gqa_value_cache_bytes"):
        measured = document.get(field)
        if not isinstance(measured, int) or measured <= 0:
            errors.append(f"session_kv_receipt.{field}: must be a positive measured byte count")
    return errors


def _validate_terminal(value: object, current: Mapping[str, Any] | None) -> list[str]:
    errors: list[str] = []
    document = _mapping(value, "terminal_receipt", errors)
    if document is None:
        return errors
    _sealed(document, "terminal_receipt", errors)
    _expect(document, "schema", TERMINAL_SCHEMA, "terminal_receipt", errors)
    _expect(document, "status", TERMINAL_STATUS, "terminal_receipt", errors)
    errors.extend(_validate_current_binding(document, current, "terminal_receipt"))
    for field in (
        "source_bound",
        "artifact_bound",
        "post_48_hidden_device_parity_passed",
        "final_rmsnorm_device_parity_passed",
        "lm_head_all_rows_device_parity_passed",
        "reserved_tail_mask_applied_before_sample",
        "deterministic_sample_and_feedback_executed",
        "sampled_token_is_tokenizer_addressable",
    ):
        _expect_true(document, field, "terminal_receipt", errors)
    for field in ("fixture_only", "synthetic_input", "fallback_used", "shadow_model_used"):
        _expect_false(document, field, "terminal_receipt", errors)
    return errors


def _validate_hcli_prelaunch(value: object, current: Mapping[str, Any] | None) -> list[str]:
    errors: list[str] = []
    document = _mapping(value, "hcli_prelaunch_receipt", errors)
    if document is None:
        return errors
    _sealed(document, "hcli_prelaunch_receipt", errors)
    _expect(document, "schema", HCLI_PRELAUNCH_SCHEMA, "hcli_prelaunch_receipt", errors)
    _expect(document, "status", HCLI_PRELAUNCH_STATUS, "hcli_prelaunch_receipt", errors)
    errors.extend(_validate_current_binding(document, current, "hcli_prelaunch_receipt"))
    for field in (
        "port_available_checked",
        "no_existing_listener",
        "telemetry_schema_validated",
        "session_metrics_bound_to_session_id",
        "state_kv_metrics_bound_to_current_receipt",
        "hcli_transport_contract_validated",
        "logical_session_multiplexing_preflight",
    ):
        _expect_true(document, field, "hcli_prelaunch_receipt", errors)
    for field in ("server_started", "hcli_request_executed", "tps_or_tg_measurement_performed"):
        _expect_false(document, field, "hcli_prelaunch_receipt", errors)
    _expect(document, "proposed_host", DEFAULT_HOST, "hcli_prelaunch_receipt", errors)
    _expect(document, "proposed_port", DEFAULT_PORT, "hcli_prelaunch_receipt", errors)
    return errors


def _validate_memory(value: object, current: Mapping[str, Any] | None) -> list[str]:
    errors: list[str] = []
    document = _mapping(value, "memory_envelope_receipt", errors)
    if document is None:
        return errors
    _sealed(document, "memory_envelope_receipt", errors)
    _expect(document, "schema", MEMORY_SCHEMA, "memory_envelope_receipt", errors)
    _expect(document, "status", MEMORY_STATUS, "memory_envelope_receipt", errors)
    errors.extend(_validate_current_binding(document, current, "memory_envelope_receipt"))
    for field in (
        "measured_on_host",
        "memory_envelope_healthy",
        "one_q80_process_envelope",
        "co_resident_envelope_accounted_for",
    ):
        _expect_true(document, field, "memory_envelope_receipt", errors)
    _expect(document, "resident_q80_model_processes", 1, "memory_envelope_receipt", errors)
    for field in ("resident_q80_rss_bytes", "available_memory_bytes", "minimum_required_available_bytes"):
        measured = document.get(field)
        if not isinstance(measured, int) or measured <= 0:
            errors.append(f"memory_envelope_receipt.{field}: must be a positive measured byte count")
    available = document.get("available_memory_bytes")
    minimum = document.get("minimum_required_available_bytes")
    if isinstance(available, int) and isinstance(minimum, int) and available < minimum:
        errors.append("memory_envelope_receipt: available memory is below its measured launch floor")
    swap = document.get("swap_used_bytes")
    if not isinstance(swap, int) or swap != 0:
        errors.append("memory_envelope_receipt.swap_used_bytes: must be measured zero for resident activation")
    return errors


def _validate_topology(value: object) -> list[str]:
    errors: list[str] = []
    document = _mapping(value, "launch_topology", errors)
    if document is None:
        return errors
    _expect(document, "model_key", MODEL_KEY, "launch_topology", errors)
    _expect(document, "bind_host", DEFAULT_HOST, "launch_topology", errors)
    _expect(document, "bind_port", DEFAULT_PORT, "launch_topology", errors)
    _expect(document, "desired_q80_model_processes", 1, "launch_topology", errors)
    _expect(document, "existing_q80_model_processes", 0, "launch_topology", errors)
    _expect(document, "server_process_starts_per_activation", 1, "launch_topology", errors)
    for field in (
        "duplicate_q80_processes_prohibited",
        "listener_absent_prelaunch",
        "single_resident_process_many_logical_sessions",
        "logical_session_state_isolated",
    ):
        _expect_true(document, field, "launch_topology", errors)
    max_sessions = document.get("maximum_logical_sessions")
    if not isinstance(max_sessions, int) or max_sessions < 2:
        errors.append("launch_topology.maximum_logical_sessions: must support at least two logical sessions")
    if document.get("bind_port") == QWEN30_CONVENTIONAL_PORT:
        errors.append("launch_topology.bind_port: Q80 must not reuse Q30's 127.0.0.1:18430 port")
    return errors


def _validate_rollback(value: object, current: Mapping[str, Any] | None) -> list[str]:
    errors: list[str] = []
    document = _mapping(value, "rollback_prelaunch_receipt", errors)
    if document is None:
        return errors
    _sealed(document, "rollback_prelaunch_receipt", errors)
    _expect(document, "schema", ROLLBACK_SCHEMA, "rollback_prelaunch_receipt", errors)
    _expect(document, "status", ROLLBACK_STATUS, "rollback_prelaunch_receipt", errors)
    errors.extend(_validate_current_binding(document, current, "rollback_prelaunch_receipt"))
    for field in (
        "automatic_launch_only_after_all_conditions_pass",
        "launch_exactly_one_q80_process",
        "record_child_pid_before_health",
        "rollback_on_health_identity_mismatch",
        "rollback_on_session_kv_leak",
        "rollback_on_memory_envelope_breach",
        "release_loopback_port_after_child_exit",
        "terminal_rollback_receipt_written_last",
        "automatic_retry_same_activation_prohibited",
    ):
        _expect_true(document, field, "rollback_prelaunch_receipt", errors)
    for field in ("server_started", "rollback_executed", "hcli_request_executed"):
        _expect_false(document, field, "rollback_prelaunch_receipt", errors)
    return errors


def _condition(name: str, errors: list[str], *, kind: str) -> dict[str, Any]:
    return {"name": name, "kind": kind, "satisfied": not errors, "blockers": errors}


def _automatic_launch_contract() -> dict[str, Any]:
    return {
        "only_when_activation_eligible": True,
        "processes_to_start": 1,
        "model_key": MODEL_KEY,
        "endpoint": {"host": DEFAULT_HOST, "port": DEFAULT_PORT},
        "logical_sessions": "many_inside_the_single_resident_process",
        "duplicate_model_process_start_prohibited": True,
        "gate_starts_no_process": True,
        "future_controlled_launcher_must_record_child_pid_before_health": True,
    }


def _rollback_contract() -> dict[str, Any]:
    return {
        "trigger_conditions": [
            "health_or_identity_mismatch",
            "session_kv_namespace_leak",
            "memory_envelope_breach",
            "unexpected_duplicate_q80_process",
        ],
        "actions": [
            "terminate_only_the_child_pid_recorded_by_the_controlled_launcher",
            "wait_for_child_reap",
            "release_127_0_0_1_18480_after_child_exit",
            "write_terminal_rollback_receipt_last",
        ],
        "automatic_retry_same_activation_prohibited": True,
        "gate_executes_no_rollback_itself": True,
    }


def assess_activation(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate evidence only; this function has no process or socket path."""
    input_errors: list[str] = []
    _expect(evidence, "schema", INPUT_SCHEMA, "evidence", input_errors)
    current, current_errors = _validate_current_identity(evidence.get("current_receipt"))
    conditions = [
        _condition("input_schema", input_errors, kind="input"),
        _condition("current_runtime_identity", current_errors, kind="identity"),
        _condition(
            "sealed_truthful_complete_decoder_readiness",
            _validate_decoder(evidence.get("decoder_readiness"), current),
            kind="decoder-readiness",
        ),
        _condition(
            "sealed_exact_runtime_no_fallback",
            _validate_runtime(evidence.get("full_runtime_receipt"), current),
            kind="runtime",
        ),
        _condition(
            "sealed_session_kv_state_and_rollback",
            _validate_session_kv(evidence.get("session_kv_receipt"), current),
            kind="state-kv",
        ),
        _condition(
            "sealed_terminal_head_tail_mask_sampler",
            _validate_terminal(evidence.get("terminal_receipt"), current),
            kind="terminal",
        ),
        _condition(
            "sealed_hcli_prelaunch_on_q80_loopback_port",
            _validate_hcli_prelaunch(evidence.get("hcli_prelaunch_receipt"), current),
            kind="hcli-prelaunch",
        ),
        _condition(
            "sealed_measured_healthy_memory_envelope",
            _validate_memory(evidence.get("memory_envelope_receipt"), current),
            kind="memory",
        ),
        _condition(
            "one_q80_process_many_logical_sessions_topology",
            _validate_topology(evidence.get("launch_topology")),
            kind="topology",
        ),
        _condition(
            "sealed_automatic_launch_and_rollback_preflight",
            _validate_rollback(evidence.get("rollback_prelaunch_receipt"), current),
            kind="rollback",
        ),
    ]
    blockers = [
        f"{condition['name']}: {blocker}"
        for condition in conditions
        for blocker in condition["blockers"]
    ]
    eligible = not blockers
    report: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": ELIGIBLE_STATUS if eligible else REFUSED_STATUS,
        "recorded_at": _utc_now(),
        "automatic_launch_eligible": eligible,
        "server_may_be_started_by_separate_controlled_launcher": eligible,
        "conditions": conditions,
        "blockers": blockers,
        "target_topology": {
            "resident_q80_model_processes": 1,
            "logical_sessions": "many",
            "endpoint": {"host": DEFAULT_HOST, "port": DEFAULT_PORT},
            "qwen30_port_reuse_refused": QWEN30_CONVENTIONAL_PORT,
        },
        "automatic_launch_contract": _automatic_launch_contract(),
        "rollback_contract": _rollback_contract(),
        "current_component_evidence_hard_refused": True,
        "claim_boundary": {
            "gate_started_no_server": True,
            "gate_bound_no_port": True,
            "gate_opened_no_model_artifact": True,
            "gate_executed_no_model_token": True,
            "gate_executed_no_hcli_request": True,
            "gate_measured_no_tps_or_tg": True,
            "eligibility_is_not_server_start_or_hcli_or_tps_evidence": True,
        },
    }
    report["unsealed_preimage_sha256"] = _sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return report


def current_component_evidence() -> dict[str, Any]:
    """The available component-only state, intentionally insufficient to launch."""
    return {
        "schema": INPUT_SCHEMA,
        "current_receipt": {
            "model_id": MODEL_ID,
            "model_key": MODEL_KEY,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "manifest_seal_sha256": MANIFEST_SEAL,
            "admission_receipt_seal_sha256": ADMISSION_RECEIPT_SEAL,
            "runtime_receipt_seal_sha256": "0" * 64,
            "runtime_executable_sha256": "1" * 64,
        },
        "decoder_readiness": {
            "schema": DECODER_SCHEMA,
            "status": "INCOMPLETE_QWEN80_COMPLETE_DECODER_READINESS_NO_DECODER_TOKEN_HCLI_OR_TPS_CLAIM",
            "complete_decoder_readiness_earned": False,
            "real_gravity_server_launch_precondition_satisfied": False,
        },
        "launch_topology": {
            "model_key": MODEL_KEY,
            "bind_host": DEFAULT_HOST,
            "bind_port": DEFAULT_PORT,
            "desired_q80_model_processes": 1,
            "existing_q80_model_processes": 0,
            "server_process_starts_per_activation": 1,
            "duplicate_q80_processes_prohibited": True,
            "listener_absent_prelaunch": True,
            "single_resident_process_many_logical_sessions": True,
            "logical_session_state_isolated": True,
            "maximum_logical_sessions": 2,
        },
    }


def _regular_absolute_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ResidentActivationGateError(f"{label} must be an absolute path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ResidentActivationGateError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ResidentActivationGateError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _read_input(path: Path) -> dict[str, Any]:
    clean = _regular_absolute_file(path, "--input")
    try:
        value = json.loads(clean.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResidentActivationGateError(f"cannot read input JSON {clean}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ResidentActivationGateError("input JSON root must be an object")
    return dict(value)


def _write_new_report(path: Path, report: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise ResidentActivationGateError("--out must be an absolute path")
    if not path.parent.is_dir():
        raise ResidentActivationGateError("--out parent directory must already exist")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing activation report: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(report), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing activation report: {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="absolute JSON evidence bundle")
    parser.add_argument("--current-component-evidence", action="store_true")
    parser.add_argument("--out", type=Path, required=True, help="new absolute JSON report path")
    arguments = parser.parse_args()
    if bool(arguments.input) == bool(arguments.current_component_evidence):
        parser.error("supply exactly one of --input or --current-component-evidence")
    return arguments


def main() -> int:
    arguments = _parse_args()
    evidence = current_component_evidence() if arguments.current_component_evidence else _read_input(arguments.input)
    report = assess_activation(evidence)
    _write_new_report(arguments.out, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["automatic_launch_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

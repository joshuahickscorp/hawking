#!/usr/bin/env python3
"""Fail-closed Qwen80 real-HCLI-adapter readiness contract.

This is a CPU-only evidence validator, not a server launcher.  It consumes a
caller-supplied JSON bundle containing the current Qwen80 decoder-readiness,
state/KV-contract, tokenizer/template/sampler evidence, and the *future*
sealed full-runtime/session-KV/telemetry receipts required before an adapter
may be started.  It never opens a model artifact, starts a process, binds a
port, contacts an HCLI endpoint, or measures TPS/TG.

Current component and fixture reports are deliberately useful only as
non-promotable prerequisites.  Until all future exact receipts are present,
the result remains ``NOT_READY_NO_SERVER`` and explicitly makes no HCLI or
throughput claim.

Example (read-only evaluation plus a new report):

    python -m lab.operators.ascension_qwen80_hcli_adapter_readiness \
      --input /absolute/path/QWEN80_HCLI_ADAPTER_EVIDENCE.json \
      --out /absolute/path/QWEN80_HCLI_ADAPTER_READINESS.json
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


INPUT_SCHEMA = "hawking.ascension.qwen80_hcli_adapter_readiness_input.v1"
RESULT_SCHEMA = "hawking.ascension.qwen80_hcli_adapter_readiness_result.v1"
NOT_READY_STATUS = "NOT_READY_NO_SERVER_QWEN80_REAL_HCLI_ADAPTER"
READY_STATUS = "READY_FOR_CONTROLLED_QWEN80_REAL_GRAVITY_HCLI_SERVER_LAUNCH_NO_HCLI_OR_TPS_CLAIM"

DECODER_SCHEMA = "hawking.ascension.qwen80_complete_decoder_readiness_result.v1"
DECODER_STATUS = "EARNED_QWEN80_COMPLETE_DECODER_READINESS_CONTRACT_ONLY"
STATE_CONTRACT_SCHEMA = "hawking.ascension.qwen80_decode_state_contract.v1"
STATE_CONTRACT_STATUS = "NOT_RUNTIME_NO_TOKEN_NO_HCLI_NO_TPS_QWEN80_HYBRID_DECODE_STATE_KV_CONTRACT"
TOKENIZER_SCHEMA = "hawking.ascension.qwen80_tokenizer_sampler_handoff_contract.v1"
TOKENIZER_STATUS = "EARNED_SOURCE_BOUND_TOKENIZER_TEMPLATE_SAMPLER_HANDOFF_COMPONENT_NOT_RUNTIME_OR_TOKEN"
FULL_RUNTIME_SCHEMA = "hawking.ascension.qwen80_full_hybrid_runtime_token_receipt.v1"
FULL_RUNTIME_STATUS = "EARNED_QWEN80_EXACT_FULL_HYBRID_RUNTIME_TOKEN_NO_FALLBACK"
SESSION_KV_SCHEMA = "hawking.ascension.qwen80_session_kv_restart_rollback_receipt.v1"
SESSION_KV_STATUS = "EARNED_QWEN80_SESSION_KV_RESTART_ROLLBACK_PARITY"
TELEMETRY_SCHEMA = "hawking.ascension.qwen80_server_telemetry_preflight.v1"
TELEMETRY_STATUS = "EARNED_QWEN80_SERVER_TELEMETRY_PRELAUNCH_CONTRACT"

MODEL_ID = "Qwen3-Coder-Next-80B"
MODEL_KEY = "qwen80"
SOURCE_REPOSITORY = "Qwen/Qwen3-Coder-Next"
SOURCE_REVISION = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb"
MANIFEST_SEAL = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b"
ADMISSION_RECEIPT_SEAL = "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628"
TOKENIZER_VOCAB = 151_669
LM_HEAD_VOCAB = 151_936
RESERVED_TAIL_ROWS = LM_HEAD_VOCAB - TOKENIZER_VOCAB

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
TELEMETRY_FIELDS = {
    "current_receipt",
    "server_binding",
    "active_logical_sessions",
    "session_id",
    "decode_position",
    "kv_bytes",
    "deltanet_state_bytes",
    "restart_rollback_generation",
    "adapter_error_classification",
}


class HcliAdapterReadinessError(ValueError):
    """The evidence bundle cannot safely make an adapter-ready claim."""


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
        if not _is_sha256(current.get(key)):
            errors.append(f"current_receipt.{key}: must be a lowercase SHA-256")
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
            errors.append(
                f"{label}.current_receipt.{key}: does not match selected current Qwen80 receipt"
            )
    return errors


def _validate_decoder_contract(value: object, current: Mapping[str, Any] | None) -> list[str]:
    errors: list[str] = []
    document = _mapping(value, "decoder_readiness", errors)
    if document is None:
        return errors
    _expect(document, "schema", DECODER_SCHEMA, "decoder_readiness", errors)
    _expect(document, "status", DECODER_STATUS, "decoder_readiness", errors)
    _expect_true(document, "complete_decoder_readiness_earned", "decoder_readiness", errors)
    _expect_true(
        document, "real_gravity_server_launch_precondition_satisfied", "decoder_readiness", errors
    )
    _expect_true(document, "input_schema_valid", "decoder_readiness", errors)
    _expect_true(document, "source_artifact_binding_valid", "decoder_readiness", errors)
    _expect_true(document, "exact_48_layer_schedule_valid", "decoder_readiness", errors)
    if document.get("missing_operator_classes_or_layers") != []:
        errors.append("decoder_readiness: missing operator coverage remains")
    binding = _mapping(document.get("source_artifact_binding"), "decoder_readiness.source_artifact_binding", errors)
    if binding is not None:
        for key in (
            "model_id",
            "model_key",
            "source_repository",
            "source_revision",
            "manifest_seal_sha256",
            "admission_receipt_seal_sha256",
        ):
            if current is not None and binding.get(key) != current.get(key):
                errors.append(f"decoder_readiness source binding {key} does not match current receipt")
    return errors


def _validate_state_contract(value: object) -> list[str]:
    """Validate the non-promotable state/KV archaeology contract is present."""
    errors: list[str] = []
    document = _mapping(value, "state_kv_contract", errors)
    if document is None:
        return errors
    _expect(document, "schema", STATE_CONTRACT_SCHEMA, "state_kv_contract", errors)
    _expect(document, "status", STATE_CONTRACT_STATUS, "state_kv_contract", errors)
    _expect_false(document, "complete_decoder_readiness_earned", "state_kv_contract", errors)
    source = _mapping(document.get("source_archaeology"), "state_kv_contract.source_archaeology", errors)
    if source is not None:
        for key, expected in (
            ("source_repository", SOURCE_REPOSITORY),
            ("source_revision", SOURCE_REVISION),
            ("layer_count", 48),
            ("deltanet_layers", 36),
            ("gqa_layers", 12),
        ):
            _expect(source, key, expected, "state_kv_contract.source_archaeology", errors)
    boundary = _mapping(document.get("execution_boundary"), "state_kv_contract.execution_boundary", errors)
    if boundary is not None:
        for key in (
            "not_runtime",
            "no_model_token_execution",
            "no_hcli_execution",
            "no_tps_or_tg_measurement",
        ):
            _expect_true(boundary, key, "state_kv_contract.execution_boundary", errors)
    checks = _mapping(document.get("fixture_contract_checks"), "state_kv_contract.fixture_contract_checks", errors)
    if checks is not None:
        for key in (
            "exact_schedule_checked",
            "state_slot_aliasing_checked",
            "causal_position_and_update_order_checked",
            "restart_identity_checked",
            "rollback_identity_checked",
            "cross_session_leakage_rejected",
            "lm_head_reserved_tail_token_rejected",
        ):
            _expect_true(checks, key, "state_kv_contract.fixture_contract_checks", errors)
    return errors


def _validate_tokenizer_sampler(value: object) -> list[str]:
    """Require the source-bound component evidence, without promoting it."""
    errors: list[str] = []
    document = _mapping(value, "tokenizer_template_sampler", errors)
    if document is None:
        return errors
    _expect(document, "schema", TOKENIZER_SCHEMA, "tokenizer_template_sampler", errors)
    _expect(document, "status", TOKENIZER_STATUS, "tokenizer_template_sampler", errors)
    _expect_true(document, "component_only", "tokenizer_template_sampler", errors)
    source = _mapping(document.get("source_binding"), "tokenizer_template_sampler.source_binding", errors)
    if source is not None:
        for key, expected in (
            ("source_repository", SOURCE_REPOSITORY),
            ("source_revision_from_pre_admitted_source_audit", SOURCE_REVISION),
            ("tokenizer_addressable_vocab_size", TOKENIZER_VOCAB),
            ("lm_head_vocab_size", LM_HEAD_VOCAB),
            ("reserved_lm_head_tail_rows", RESERVED_TAIL_ROWS),
        ):
            _expect(source, key, expected, "tokenizer_template_sampler.source_binding", errors)
    sampler = _mapping(document.get("sampler_fixture"), "tokenizer_template_sampler.sampler_fixture", errors)
    if sampler is not None:
        _expect(sampler, "reserved_tail_mask_cutoff", TOKENIZER_VOCAB, "tokenizer_template_sampler.sampler_fixture", errors)
        _expect_true(
            sampler,
            "all_selected_reserved_fixture_logits_are_negative_infinity",
            "tokenizer_template_sampler.sampler_fixture",
            errors,
        )
        _expect_true(
            sampler,
            "sampled_id_is_tokenizer_addressable",
            "tokenizer_template_sampler.sampler_fixture",
            errors,
        )
        sampled = sampler.get("sampled_feedback_token_id")
        if not isinstance(sampled, int) or not 0 <= sampled < TOKENIZER_VOCAB:
            errors.append("tokenizer_template_sampler: sampled fixture feedback is outside tokenizer namespace")
    rejections = _mapping(document.get("rejection_tests"), "tokenizer_template_sampler.rejection_tests", errors)
    if rejections is not None:
        for key in (
            "reserved_prompt_token_rejected",
            "sample_before_tail_mask_rejected",
            "wrong_tail_mask_cutoff_rejected",
            "reserved_tail_feedback_rejected",
        ):
            _expect_true(rejections, key, "tokenizer_template_sampler.rejection_tests", errors)
    return errors


def _validate_future_full_runtime(value: object, current: Mapping[str, Any] | None) -> list[str]:
    errors: list[str] = []
    document = _mapping(value, "full_runtime_receipt", errors)
    if document is None:
        return errors
    _sealed(document, "full_runtime_receipt", errors)
    _expect(document, "schema", FULL_RUNTIME_SCHEMA, "full_runtime_receipt", errors)
    _expect(document, "status", FULL_RUNTIME_STATUS, "full_runtime_receipt", errors)
    errors.extend(_validate_current_binding(document, current, "full_runtime_receipt"))
    for key in (
        "source_bound",
        "artifact_bound",
        "full_runtime",
        "complete_token_path",
        "full_48_layer_token_executed",
        "all_36_deltanet_layers_executed",
        "all_12_gqa_layers_executed",
        "final_norm_lm_head_tail_mask_sampler_executed",
    ):
        _expect_true(document, key, "full_runtime_receipt", errors)
    for key in (
        "fixture_only",
        "component_only",
        "synthetic_input",
        "fallback_used",
        "shadow_model_used",
        "raw_bf16_or_mps_fallback_used",
        "hcli_execution_performed",
        "tps_or_tg_measurement_performed",
    ):
        _expect_false(document, key, "full_runtime_receipt", errors)
    return errors


def _validate_future_session_kv(value: object, current: Mapping[str, Any] | None) -> list[str]:
    errors: list[str] = []
    document = _mapping(value, "session_kv_receipt", errors)
    if document is None:
        return errors
    _sealed(document, "session_kv_receipt", errors)
    _expect(document, "schema", SESSION_KV_SCHEMA, "session_kv_receipt", errors)
    _expect(document, "status", SESSION_KV_STATUS, "session_kv_receipt", errors)
    errors.extend(_validate_current_binding(document, current, "session_kv_receipt"))
    for key in (
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
        _expect_true(document, key, "session_kv_receipt", errors)
    for key in ("fixture_only", "component_only", "synthetic_input", "fallback_used"):
        _expect_false(document, key, "session_kv_receipt", errors)
    sessions = document.get("observed_session_ids")
    if (
        not isinstance(sessions, list)
        or len(sessions) < 2
        or not all(isinstance(session, str) and session for session in sessions)
        or len(set(sessions)) != len(sessions)
    ):
        errors.append("session_kv_receipt.observed_session_ids: requires at least two distinct non-empty sessions")
    for field in ("deltanet_state_bytes", "gqa_key_cache_bytes", "gqa_value_cache_bytes"):
        value = document.get(field)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"session_kv_receipt.{field}: must be a positive actual state byte count")
    return errors


def _validate_future_telemetry(value: object, current: Mapping[str, Any] | None) -> list[str]:
    errors: list[str] = []
    document = _mapping(value, "telemetry_receipt", errors)
    if document is None:
        return errors
    _sealed(document, "telemetry_receipt", errors)
    _expect(document, "schema", TELEMETRY_SCHEMA, "telemetry_receipt", errors)
    _expect(document, "status", TELEMETRY_STATUS, "telemetry_receipt", errors)
    errors.extend(_validate_current_binding(document, current, "telemetry_receipt"))
    for key in (
        "port_available_checked",
        "no_existing_listener",
        "telemetry_schema_validated",
        "session_metrics_bound_to_session_id",
        "state_kv_metrics_bound_to_current_receipt",
    ):
        _expect_true(document, key, "telemetry_receipt", errors)
    for key in ("server_started", "hcli_request_executed", "tps_or_tg_measurement_performed"):
        _expect_false(document, key, "telemetry_receipt", errors)
    _expect(document, "proposed_host", "127.0.0.1", "telemetry_receipt", errors)
    port = document.get("proposed_port")
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        errors.append("telemetry_receipt.proposed_port: must be an unprivileged TCP port")
    fields = document.get("telemetry_fields")
    if not isinstance(fields, list) or not TELEMETRY_FIELDS.issubset(set(fields)):
        errors.append("telemetry_receipt.telemetry_fields: required per-session state/KV fields are incomplete")
    return errors


def _condition(name: str, errors: list[str], *, kind: str) -> dict[str, Any]:
    return {"name": name, "kind": kind, "satisfied": not errors, "blockers": errors}


def _rawls_server_port_handoff() -> list[dict[str, Any]]:
    return [
        {
            "condition": "single resident Q80 process only",
            "required_before_start": "One Q80 Gravity process owns one chosen loopback port; create many logical sessions inside it rather than cloning model processes.",
        },
        {
            "condition": "port ownership and health identity",
            "required_before_start": "Bind only 127.0.0.1 on the telemetry-preflight port after a fresh no-listener check; health/context must expose the exact current runtime receipt, executable SHA, manifest seal, and admission receipt seal.",
        },
        {
            "condition": "session/KV namespace",
            "required_before_start": "Every HCLI session ID must select distinct DeltaNet/KV state namespaces, report position and state/KV bytes, and reject collisions or a stale restart/rollback generation.",
        },
        {
            "condition": "HCLI and performance boundary",
            "required_before_start": "Do not treat startup, health, transport smoke, or concurrent logical sessions as HCLI coherence or TPS. Schedule any real HCLI/TPS work only under the separate clean measurement policy.",
        },
    ]


def assess_adapter_readiness(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return a read-only, fail-closed Q80 HCLI server-launch assessment.

    The function deliberately accepts synthetic test documents so the grammar is
    testable, but a ``READY`` result remains only a pre-launch condition.  It
    cannot start a process or prove that an HCLI request succeeded.
    """
    input_errors: list[str] = []
    _expect(evidence, "schema", INPUT_SCHEMA, "evidence", input_errors)
    current, current_errors = _validate_current_identity(evidence.get("current_receipt"))
    conditions = [
        _condition("input_schema", input_errors, kind="input"),
        _condition("current_receipt", current_errors, kind="identity"),
        _condition(
            "complete_decoder_readiness_contract",
            _validate_decoder_contract(evidence.get("decoder_readiness"), current),
            kind="current-contract",
        ),
        _condition(
            "state_kv_archaeology_contract",
            _validate_state_contract(evidence.get("state_kv_contract")),
            kind="current-contract",
        ),
        _condition(
            "tokenizer_template_tail_mask_component_contract",
            _validate_tokenizer_sampler(evidence.get("tokenizer_template_sampler")),
            kind="current-component-prerequisite",
        ),
        _condition(
            "future_exact_full_runtime_token_no_fallback",
            _validate_future_full_runtime(evidence.get("full_runtime_receipt"), current),
            kind="future-runtime-receipt",
        ),
        _condition(
            "future_session_kv_restart_rollback",
            _validate_future_session_kv(evidence.get("session_kv_receipt"), current),
            kind="future-runtime-receipt",
        ),
        _condition(
            "future_server_telemetry_and_port_preflight",
            _validate_future_telemetry(evidence.get("telemetry_receipt"), current),
            kind="future-runtime-receipt",
        ),
    ]
    blockers = [
        f"{condition['name']}: {blocker}"
        for condition in conditions
        for blocker in condition["blockers"]
    ]
    ready = not blockers
    report: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": READY_STATUS if ready else NOT_READY_STATUS,
        "recorded_at": _utc_now(),
        "hcli_adapter_launch_precondition_satisfied": ready,
        "server_may_be_started_by_separate_controlled_launcher": ready,
        "conditions": conditions,
        "blockers": blockers,
        "non_promotable_current_evidence": {
            "state_kv_contract_is_not_runtime": True,
            "tokenizer_template_sampler_is_component_only": True,
            "component_or_fixture_receipts_are_rejected_for_full_runtime_and_session_kv_requirements": True,
        },
        "rawls_server_port_handoff": _rawls_server_port_handoff(),
        "claim_boundary": {
            "this_validator_started_no_server": True,
            "this_validator_bound_no_port": True,
            "this_validator_executed_no_model_token": True,
            "this_validator_executed_no_hcli_request": True,
            "this_validator_measured_no_tps_or_tg": True,
            "hcli_or_tps_earned_by_this_result": False,
            "ready_means_prelaunch_contract_only": True,
        },
    }
    report["unsealed_preimage_sha256"] = _sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return report


def _regular_absolute_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise HcliAdapterReadinessError(f"{label} must be an absolute path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HcliAdapterReadinessError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise HcliAdapterReadinessError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _read_input(path: Path) -> dict[str, Any]:
    clean = _regular_absolute_file(path, "--input")
    try:
        value = json.loads(clean.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HcliAdapterReadinessError(f"cannot read input JSON {clean}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise HcliAdapterReadinessError("input JSON root must be an object")
    return dict(value)


def _write_new_report(path: Path, report: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise HcliAdapterReadinessError("--out must be an absolute path")
    parent = path.parent
    if not parent.is_dir():
        raise HcliAdapterReadinessError("--out parent directory must already exist")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing readiness report: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(report), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="absolute evidence-bundle JSON path")
    parser.add_argument("--out", type=Path, required=True, help="new absolute readiness-report JSON path")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        evidence = _read_input(arguments.input)
        report = assess_adapter_readiness(evidence)
        _write_new_report(arguments.out, report)
    except (HcliAdapterReadinessError, FileExistsError) as exc:
        print(
            json.dumps(
                {"status": NOT_READY_STATUS, "detail": str(exc), "hcli_or_tps_earned": False},
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "readiness_report": str(arguments.out),
                "hcli_or_tps_earned": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())

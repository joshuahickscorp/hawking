#!/usr/bin/env python3.12
"""Durable, fail-closed controller state for the GLM-5.2 streaming campaign.

This module is deliberately control-plane only.  It does not download, execute, or
delete model data.  It provides the integrity and restart substrate that those actions
must enter through:

* the exact Part VI campaign FSM;
* one-use, idempotent transition claims;
* a strict, hash-chained JSONL event log;
* an atomic self-sealed checkpoint anchored to both durable logs;
* exact single-tail crash recovery and split-brain rejection;
* an exclusive ``flock`` controller lease; and
* ``GLM52_WINDOW_LEDGER.jsonl`` records with monotonic source/tensor coverage.

The chains are unkeyed integrity records, not signatures.  A checkpoint anchor detects
tail truncation and forks relative to the last durable checkpoint; a party able to
rewrite every record and checkpoint can still forge an unkeyed history.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import hmac
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TextIO

from glm52_common import (
    Glm52Error,
    atomic_json,
    canonical,
    read_sealed_json,
    seal,
    utc_now,
    verify_sealed,
)


EVENT_SCHEMA = "hawking.glm52.controller_event.v2"
CHECKPOINT_SCHEMA = "hawking.glm52.controller_checkpoint.v2"
LEASE_SCHEMA = "hawking.glm52.controller_lease.v1"
TELEGRAM_RECEIPT_SCHEMA = "hawking.glm52.telegram_delivery_receipt.v3"
WINDOW_EVENT_SCHEMA = "hawking.glm52.window_event.v1"
WINDOW_RECORD_SCHEMA = "hawking.glm52.window_record.v1"
EXPECTED_CONTRACT_SCHEMA = "hawking.glm52.expected_campaign_contract.v2"
TERMINAL_EVIDENCE_SCHEMA = "hawking.glm52.state_terminal_evidence.v2"
TRANSITION_INTENT_SCHEMA = "hawking.glm52.state_transition_intent.v1"
CONTROLLER_ANCHOR_SCHEMA = "hawking.glm52.controller_anchor.v1"
CAMPAIGN_STATUS_SCHEMA = "hawking.glm52.canonical_campaign_status.v1"
STOP_CONDITION_EVIDENCE_SCHEMA = "hawking.glm52.stop_condition_evidence.v1"
GENESIS_HASH = "0" * 64
WINDOW_LEDGER_FILENAME = "GLM52_WINDOW_LEDGER.jsonl"
OFFICIAL_WEIGHT_SHARD_COUNT = 282
OFFICIAL_WEIGHT_LOGICAL_BYTES = 1_506_667_387_408
OFFICIAL_ASSEMBLY_REQUIRED_ARTIFACTS = frozenset(
    {
        "official_source_manifest",
        "logical_weight_ledger",
        "streaming_schedule",
        "tensor_coverage_ledger",
    }
)
OFFICIAL_COMPLETE_REQUIRED_ARTIFACTS = frozenset(
    {
        "glm52_gravity_final",
        "gravity_completeness_audit_final",
        "byte_auction",
        "terminal_outcome",
        "rollback_transfer",
    }
)
MANDATORY_COMPLETE_STOP_CONDITIONS: tuple[str, ...] = (
    "kimi_final_evidence_verified",
    "kimi_raw_source_safely_released",
    "official_glm52_immutable_revision_sealed",
    "bf16_source_manifest_complete",
    "exact_logical_weight_ledger_sealed",
    "gravity_pre_audit_complete",
    "external_baseline_matrix_complete",
    "xet_autotune_complete",
    "dependency_streaming_schedule_sealed",
    "adapter_twin_green",
    "bf16_reference_forward_validated",
    "corpus_integrity_green",
    "oracle_causal_pilot_complete",
    "full_candidate_program_frozen",
    "every_official_bf16_shard_fetched_verified",
    "every_source_tensor_terminal",
    "complete_sub_one_artifact_assembled",
    "half_bpw_exact_tested_f0_f1",
    "earned_rate_fidelity_evaluated",
    "doctor_native_student_compared",
    "full_compact_model_executed",
    "capability_result_sealed",
    "direct_runtime_boundary_reported",
    "all_temporary_bf16_windows_evicted",
    "source_byte_amplification_reported",
    "telegram_phone_status_complete",
    "gravity_post_audit_complete",
    "terminal_outcome_sealed",
    "final_reports_committed_pushed",
    "rollback_next_parent_transfer_complete",
)


# Exact order and spelling required by the campaign goal, Part VI section 6.1.
STATES: tuple[str, ...] = (
    "PRECHECK",
    "CLOSE_KIMI",
    "RELEASE_KIMI_SOURCE",
    "ADMIT_GLM_SOURCE",
    "BUILD_MANIFEST",
    "BUILD_DEPENDENCY_GRAPH",
    "AUTOTUNE_XET",
    "BUILD_ADAPTER",
    "BUILD_REFERENCE",
    "BUILD_CORPUS",
    "PILOT_ORACLES",
    "FREEZE_PROGRAM",
    "FETCH_WINDOW",
    "VERIFY_WINDOW",
    "CAPTURE_TEACHER",
    "FIT_CANDIDATES",
    "PACK_CANDIDATES",
    "RUN_WINDOW_FORWARD",
    "SEAL_WINDOW",
    "EVICT_WINDOW",
    "ASSEMBLE_ARTIFACT",
    "VERIFY_ARTIFACT",
    "RUN_FULL_COMPACT",
    "RUN_DOCTOR_REFINEMENT",
    "RUN_RATE_DESCENT",
    "SEAL_GLM_RESULT",
    "FINAL_GRAVITY_AUDIT",
    "COMPLETE",
    "BLOCKED",
)


# BLOCKED is handled specially: every non-terminal state may enter it, and may leave it
# only for the exact state saved in ``blocked_from`` with a resolution receipt.
_FORWARD_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "PRECHECK": ("CLOSE_KIMI",),
    "CLOSE_KIMI": ("RELEASE_KIMI_SOURCE",),
    "RELEASE_KIMI_SOURCE": ("ADMIT_GLM_SOURCE",),
    "ADMIT_GLM_SOURCE": ("BUILD_MANIFEST",),
    "BUILD_MANIFEST": ("BUILD_DEPENDENCY_GRAPH",),
    "BUILD_DEPENDENCY_GRAPH": ("AUTOTUNE_XET",),
    "AUTOTUNE_XET": ("BUILD_ADAPTER",),
    "BUILD_ADAPTER": ("BUILD_REFERENCE",),
    "BUILD_REFERENCE": ("BUILD_CORPUS",),
    "BUILD_CORPUS": ("PILOT_ORACLES",),
    "PILOT_ORACLES": ("FREEZE_PROGRAM",),
    "FREEZE_PROGRAM": ("FETCH_WINDOW",),
    "FETCH_WINDOW": ("VERIFY_WINDOW",),
    "VERIFY_WINDOW": ("CAPTURE_TEACHER",),
    "CAPTURE_TEACHER": ("FIT_CANDIDATES",),
    "FIT_CANDIDATES": ("PACK_CANDIDATES",),
    "PACK_CANDIDATES": ("RUN_WINDOW_FORWARD",),
    "RUN_WINDOW_FORWARD": ("SEAL_WINDOW",),
    "SEAL_WINDOW": ("EVICT_WINDOW",),
    # The sole source-stream loop: fetch another declared dependency window, or assemble.
    "EVICT_WINDOW": ("FETCH_WINDOW", "ASSEMBLE_ARTIFACT"),
    "ASSEMBLE_ARTIFACT": ("VERIFY_ARTIFACT",),
    "VERIFY_ARTIFACT": ("RUN_FULL_COMPACT",),
    # Doctor may be unnecessary.  A treated artifact is re-run at full compact fidelity.
    "RUN_FULL_COMPACT": ("RUN_DOCTOR_REFINEMENT", "RUN_RATE_DESCENT"),
    "RUN_DOCTOR_REFINEMENT": ("RUN_FULL_COMPACT", "RUN_RATE_DESCENT"),
    # Rate descent can schedule the next full compact candidate or close the earned floor.
    "RUN_RATE_DESCENT": ("RUN_FULL_COMPACT", "SEAL_GLM_RESULT"),
    "SEAL_GLM_RESULT": ("FINAL_GRAVITY_AUDIT",),
    "FINAL_GRAVITY_AUDIT": ("COMPLETE",),
    "COMPLETE": (),
    "BLOCKED": (),
}

WINDOW_STATES = frozenset(
    {
        "FETCH_WINDOW",
        "VERIFY_WINDOW",
        "CAPTURE_TEACHER",
        "FIT_CANDIDATES",
        "PACK_CANDIDATES",
        "RUN_WINDOW_FORWARD",
        "SEAL_WINDOW",
        "EVICT_WINDOW",
    }
)

WINDOW_PHASES: tuple[str, ...] = (
    "PLANNED",
    "FETCHING",
    "FETCHED",
    "VERIFIED",
    "TEACHER_CAPTURED",
    "CANDIDATES_FIT",
    "CANDIDATES_PACKED",
    "FORWARD_COMPLETE",
    "SEALED",
    "EVICTED",
)
_WINDOW_NEXT = {left: right for left, right in zip(WINDOW_PHASES, WINDOW_PHASES[1:])}

SOURCE_COVERAGE_STATES: tuple[str, ...] = (
    "DECLARED",
    "FETCHING",
    "FETCHED",
    "HASH_VERIFIED",
    "CONSUMED",
    "EVICTED",
)
_SOURCE_NEXT = {
    left: right for left, right in zip(SOURCE_COVERAGE_STATES, SOURCE_COVERAGE_STATES[1:])
}

TENSOR_PROGRESS_STATES: tuple[str, ...] = (
    "DECLARED",
    "SOURCE_VERIFIED",
    "TEACHER_EVIDENCED",
    "CANDIDATE_PACKED",
    "FORWARD_VERIFIED",
)
TENSOR_TERMINAL_STATES: tuple[str, ...] = (
    "PACKED_IN_CORE_ARTIFACT",
    "PACKED_IN_OPTIONAL_MTP_PACK",
    "PROTECTED_SOURCE_NATIVE_WITH_BILLED_BYTES",
    "INTENTIONALLY_OMITTED_WITH_CAPABILITY_JUSTIFICATION",
    "NON_MODEL_FILE",
)
_TENSOR_NEXT = {
    left: right for left, right in zip(TENSOR_PROGRESS_STATES, TENSOR_PROGRESS_STATES[1:])
}

_CLAIM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{7,255}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_STATUS_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,127}$")
_RATE_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
_UTC_Z_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
MAX_TELEGRAM_MESSAGE_CHARS = 4096
CONTROLLER_HEARTBEAT_MAX_AGE_SECONDS = 120

_EVIDENCE_VALIDATOR_SOURCES: dict[str, str] = {
    "sealed_exact_v1": (
        "require strict JSON, valid self-seal, exact frozen seal, exact schema, allowed status"
    ),
    "campaign_artifact_v1": (
        "require campaign/revision/contract identity, non-empty evidence, evidence hash, producer HMAC"
    ),
    "stop_condition_v1": (
        "require stop-condition/campaign/revision/contract identity, PASS, non-empty evidence, "
        "evidence hash, producer HMAC"
    ),
    "phone_status_v2": (
        "require GREEN phone v2, live controller lease, in-sync applied operator request, "
        "currently fresh heartbeat, exact pre-transition checkpoint anchor, and producer HMAC"
    ),
    "frozen_schedule_v2": (
        "require post-Xet frozen schedule, exact contract window dependency/acquisition membership, "
        "plan/preliminary/result seals, selected-profile hash, campaign identity, evidence hash, "
        "and producer HMAC"
    ),
    "xet_autotune_result_v1": (
        "reconstruct the exact raw live-Xet result from its authenticated original seal, validate "
        "it against the frozen plan with the live executor validator, and require campaign identity, "
        "producing controller anchor, non-empty evidence, evidence hash, and producer HMAC"
    ),
    "future_artifact_blocked_v1": (
        "fail closed until this artifact schema has a dedicated scientific validator and cross-file bindings"
    ),
    "stop_condition_blocked_v1": (
        "fail closed until this stop condition is derived from grounded campaign evidence"
    ),
}
EVIDENCE_VALIDATOR_SOURCE_SHA256: dict[str, str] = {
    name: hashlib.sha256(source.encode("utf-8")).hexdigest()
    for name, source in _EVIDENCE_VALIDATOR_SOURCES.items()
}


# Telegram event names are controller policy, never caller-controlled strings.
TRANSITION_EVENT_KINDS: dict[str, str] = {
    "PRECHECK": "campaign_precheck",
    "CLOSE_KIMI": "kimi_close",
    "RELEASE_KIMI_SOURCE": "kimi_source_release",
    "ADMIT_GLM_SOURCE": "glm_source_admission",
    "BUILD_MANIFEST": "official_manifest_sealed",
    "BUILD_DEPENDENCY_GRAPH": "dependency_graph_sealed",
    "AUTOTUNE_XET": "xet_autotune_start",
    "BUILD_ADAPTER": "xet_autotune_result",
    "BUILD_REFERENCE": "adapter_twin_result",
    "BUILD_CORPUS": "bf16_reference_result",
    "PILOT_ORACLES": "corpus_integrity_result",
    "FREEZE_PROGRAM": "oracle_pilot_result",
    "FETCH_WINDOW": "source_stream_start",
    "VERIFY_WINDOW": "window_source_verified",
    "CAPTURE_TEACHER": "teacher_capture_start",
    "FIT_CANDIDATES": "candidate_fit_start",
    "PACK_CANDIDATES": "candidate_pack_start",
    "RUN_WINDOW_FORWARD": "window_forward_start",
    "SEAL_WINDOW": "window_completed",
    "EVICT_WINDOW": "window_eviction_start",
    "ASSEMBLE_ARTIFACT": "compact_artifact_assembly",
    "VERIFY_ARTIFACT": "compact_artifact_verification",
    "RUN_FULL_COMPACT": "full_compact_run",
    "RUN_DOCTOR_REFINEMENT": "doctor_diagnosis_treatment",
    "RUN_RATE_DESCENT": "rate_descent",
    "SEAL_GLM_RESULT": "glm_result_sealed",
    "FINAL_GRAVITY_AUDIT": "final_gravity_audit",
    "COMPLETE": "campaign_final",
    "BLOCKED": "campaign_fault",
}


class StateError(Glm52Error):
    """A controller, checkpoint, event-chain, or coverage invariant failed."""


class TelegramAuthConfig:
    """In-memory Telegram receipt authenticator; the HMAC key is never serialized."""

    __slots__ = ("_hmac_key", "expected_chat_identity_digest")

    def __init__(self, *, hmac_key: bytes, expected_chat_identity_digest: str):
        if not isinstance(hmac_key, bytes) or len(hmac_key) < 32:
            raise StateError("Telegram HMAC key must be at least 32 bytes")
        if not _is_sha256(expected_chat_identity_digest):
            raise StateError("expected Telegram chat identity digest must be a sha256")
        self._hmac_key = bytes(hmac_key)
        self.expected_chat_identity_digest = expected_chat_identity_digest

    def __repr__(self) -> str:
        return (
            "TelegramAuthConfig(hmac_key=<redacted>, "
            f"expected_chat_identity_digest={self.expected_chat_identity_digest!r})"
        )

    def authenticate(self, body: Mapping[str, Any]) -> str:
        return hmac.new(self._hmac_key, canonical(dict(body)), hashlib.sha256).hexdigest()

    def verify(self, body: Mapping[str, Any], recorded: Any) -> bool:
        return _is_sha256(recorded) and hmac.compare_digest(
            self.authenticate(body), str(recorded)
        )


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HEX64_RE.fullmatch(value) is not None


def _parse_utc_z(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or _UTC_Z_RE.fullmatch(value) is None:
        raise StateError(f"{label} must be an RFC3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise StateError(f"{label} is not a valid RFC3339 UTC timestamp") from exc
    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset is None or offset.total_seconds() != 0:
        raise StateError(f"{label} must be UTC")
    return parsed


def _heartbeat_freshness(heartbeat: Any) -> tuple[bool, str, str | None]:
    if not isinstance(heartbeat, dict):
        return False, "checkpoint heartbeat absent", None
    heartbeat_at = heartbeat.get("at")
    try:
        observed = _parse_utc_z(heartbeat_at, "checkpoint heartbeat.at")
        now = _parse_utc_z(utc_now(), "controller status current time")
    except StateError as exc:
        return False, str(exc), heartbeat_at if isinstance(heartbeat_at, str) else None
    age_seconds = (now - observed).total_seconds()
    if age_seconds < 0:
        return False, "checkpoint heartbeat is in the future", heartbeat_at
    if age_seconds > CONTROLLER_HEARTBEAT_MAX_AGE_SECONDS:
        return False, "checkpoint heartbeat exceeded the maximum age", heartbeat_at
    return True, "checkpoint heartbeat is fresh", heartbeat_at


def _clone(value: Any) -> Any:
    """JSON-domain copy that also rejects NaN and non-serializable payloads."""
    return json.loads(canonical(value).decode("utf-8"))


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StateError(f"{label} must be a JSON object")
    try:
        return _clone(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StateError(f"{label} is not canonical JSON: {exc}") from exc


def _require_claim(claim_id: str) -> str:
    if not isinstance(claim_id, str) or _CLAIM_RE.fullmatch(claim_id) is None:
        raise StateError(
            "claim_id must be 8-256 characters from [A-Za-z0-9._:/@+-], starting alphanumeric"
        )
    return claim_id


def _require_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StateError(f"{label} must be a non-empty, trimmed string")
    return value


def _validate_identity(campaign_id: str, source_revision: str, controller_epoch: str) -> None:
    _require_name(campaign_id, "campaign_id")
    _require_name(controller_epoch, "controller_epoch")
    if not isinstance(source_revision, str) or _HEX40_RE.fullmatch(source_revision) is None:
        raise StateError("source_revision must be the immutable 40-hex Git revision")


def telegram_chat_identity_digest(chat_id: str | int) -> str:
    """Hash the stable Bot API chat id without persisting the raw identity."""
    if isinstance(chat_id, bool) or not isinstance(chat_id, (str, int)) or not str(chat_id):
        raise StateError("Telegram chat id must be a non-empty string or integer")
    return _sha256(
        {
            "schema": "hawking.glm52.telegram_chat_identity.v1",
            "chat_id": str(chat_id),
        }
    )


def seal_producer_authenticated_evidence(
    body: Mapping[str, Any], *, auth: TelegramAuthConfig
) -> dict[str, Any]:
    """Add the campaign producer HMAC and self-seal used by future gate artifacts."""
    value = _require_object(dict(body), "producer evidence body")
    if "seal_sha256" in value or "producer_hmac_sha256" in value:
        raise StateError("producer evidence body must not contain generated authentication fields")
    if not isinstance(auth, TelegramAuthConfig):
        raise StateError("producer evidence requires TelegramAuthConfig")
    signature = auth.authenticate({
        "schema": "hawking.glm52.evidence_producer_auth.v1",
        "artifact": value,
    })
    return seal({**value, "producer_hmac_sha256": signature})


def _validate_contract_shard(shard: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(shard, dict) or set(shard) != {
        "path",
        "logical_bytes",
        "content_hash",
        "content_hash_kind",
    }:
        raise StateError(f"{label} has an invalid source-shard identity schema")
    value = _clone(shard)
    _require_name(value.get("path"), f"{label}.path")
    size = value.get("logical_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise StateError(f"{label}.logical_bytes must be positive")
    if not _is_sha256(value.get("content_hash")):
        raise StateError(f"{label}.content_hash must be a sha256/Xet hash")
    if value.get("content_hash_kind") not in {"sha256", "xet"}:
        raise StateError(f"{label}.content_hash_kind invalid")
    return value


def _validate_evidence_policy(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "path", "expected_seal_sha256", "expected_schema", "allowed_statuses",
        "validator_id", "validator_source_sha256", "require_producer_hmac",
    }:
        raise StateError(f"{label} policy schema invalid")
    policy = _clone(value)
    path = _require_name(policy.get("path"), f"{label}.path")
    candidate = Path(path)
    if candidate.is_absolute() or path in {".", ".."} or ".." in candidate.parts:
        raise StateError(f"{label}.path must be a contained relative path")
    expected_seal = policy.get("expected_seal_sha256")
    if expected_seal is not None and not _is_sha256(expected_seal):
        raise StateError(f"{label}.expected_seal_sha256 invalid")
    _require_name(policy.get("expected_schema"), f"{label}.expected_schema")
    statuses = policy.get("allowed_statuses")
    if not isinstance(statuses, list) or not statuses or len(statuses) != len(set(statuses)):
        raise StateError(f"{label}.allowed_statuses must be a non-empty unique list")
    for status_name in statuses:
        _require_name(status_name, f"{label}.allowed_statuses entry")
    validator_id = policy.get("validator_id")
    if validator_id not in EVIDENCE_VALIDATOR_SOURCE_SHA256:
        raise StateError(f"{label}.validator_id is not registered")
    if policy.get("validator_source_sha256") != \
            EVIDENCE_VALIDATOR_SOURCE_SHA256[validator_id]:
        raise StateError(f"{label}.validator_source_sha256 mismatch")
    if not isinstance(policy.get("require_producer_hmac"), bool):
        raise StateError(f"{label}.require_producer_hmac must be boolean")
    if expected_seal is None and validator_id not in {
        "campaign_artifact_v1", "stop_condition_v1", "phone_status_v2",
        "frozen_schedule_v2", "xet_autotune_result_v1",
        "future_artifact_blocked_v1", "stop_condition_blocked_v1",
    }:
        raise StateError(f"{label} has no frozen seal or semantic validator")
    if expected_seal is None and validator_id in {
        "campaign_artifact_v1", "stop_condition_v1", "frozen_schedule_v2",
        "xet_autotune_result_v1", "phone_status_v2",
        "future_artifact_blocked_v1", "stop_condition_blocked_v1",
    } and policy["require_producer_hmac"] is not True:
        raise StateError(f"{label} future evidence must require producer HMAC")
    if validator_id == "sealed_exact_v1" and expected_seal is None:
        raise StateError(f"{label} sealed_exact_v1 requires a frozen seal")
    return policy


def _validate_state_gate(state: str, gate: Any) -> dict[str, Any]:
    if state not in STATES:
        raise StateError(f"expected contract has a gate for unknown state: {state}")
    required_keys = {
        "require_source_complete",
        "require_tensor_complete",
        "require_final_source_eviction",
        "require_telegram_delivery",
        "require_phone_status",
        "required_phone_status_path",
        "required_artifacts",
        "required_checklist",
    }
    if not isinstance(gate, dict) or set(gate) != required_keys:
        raise StateError(f"state gate {state} fields must be exactly {sorted(required_keys)}")
    value = _clone(gate)
    for key in (
        "require_source_complete",
        "require_tensor_complete",
        "require_final_source_eviction",
        "require_telegram_delivery",
        "require_phone_status",
    ):
        if not isinstance(value[key], bool):
            raise StateError(f"state gate {state}.{key} must be boolean")
    phone_path = value["required_phone_status_path"]
    if value["require_phone_status"]:
        _require_name(phone_path, f"state gate {state}.required_phone_status_path")
    elif phone_path is not None:
        raise StateError(f"state gate {state} has a phone path but phone status is not required")
    artifacts = value["required_artifacts"]
    if not isinstance(artifacts, dict):
        raise StateError(f"state gate {state}.required_artifacts must be an object")
    for name, spec in artifacts.items():
        _require_name(name, f"state gate {state} artifact label")
        _validate_evidence_policy(spec, label=f"state gate {state} artifact {name}")
    checklist = value["required_checklist"]
    if not isinstance(checklist, dict):
        raise StateError(f"state gate {state}.required_checklist must be an object")
    for item, spec in checklist.items():
        _require_name(item, f"state gate {state} checklist item")
        _validate_evidence_policy(spec, label=f"state gate {state} checklist {item}")
    return value


def _validate_expected_contract(value: Any) -> dict[str, Any]:
    contract = _require_object(value, "expected campaign contract")
    try:
        verify_sealed(contract, label="expected campaign contract")
    except Glm52Error as exc:
        raise StateError(str(exc)) from exc
    required_keys = {
        "schema",
        "campaign_id",
        "source_revision",
        "expected_chat_identity_digest",
        "source",
        "tensors",
        "window_schedule",
        "state_gates",
        "created_at",
        "seal_sha256",
    }
    if set(contract) != required_keys:
        raise StateError("expected campaign contract has missing or unknown fields")
    if contract.get("schema") != EXPECTED_CONTRACT_SCHEMA:
        raise StateError("expected campaign contract schema mismatch")
    _validate_identity(contract["campaign_id"], contract["source_revision"], "contract-validation")
    if not _is_sha256(contract.get("expected_chat_identity_digest")):
        raise StateError("expected campaign contract chat identity digest invalid")
    source = contract.get("source")
    if not isinstance(source, dict) or set(source) != {
        "profile",
        "expected_shard_count",
        "expected_logical_bytes",
        "shards",
    }:
        raise StateError("expected campaign contract source schema invalid")
    if source.get("profile") not in {"OFFICIAL_GLM52_BF16", "SYNTHETIC_TEST_ONLY"}:
        raise StateError("expected campaign contract source profile invalid")
    shards = [
        _validate_contract_shard(item, label=f"expected source shard[{index}]")
        for index, item in enumerate(source["shards"])
    ] if isinstance(source.get("shards"), list) else []
    if not shards:
        raise StateError("expected campaign contract must contain source shards")
    paths = [item["path"] for item in shards]
    if len(paths) != len(set(paths)):
        raise StateError("expected campaign contract contains duplicate source shard paths")
    logical_bytes = sum(item["logical_bytes"] for item in shards)
    if source.get("expected_shard_count") != len(shards):
        raise StateError("expected source shard count does not equal its authoritative manifest")
    if source.get("expected_logical_bytes") != logical_bytes:
        raise StateError("expected source bytes do not equal its authoritative manifest")
    if source["profile"] == "OFFICIAL_GLM52_BF16" and (
        len(shards) != OFFICIAL_WEIGHT_SHARD_COUNT
        or logical_bytes != OFFICIAL_WEIGHT_LOGICAL_BYTES
    ):
        raise StateError(
            "official GLM-5.2 BF16 contract must contain exactly 282 weight shards and "
            "1,506,667,387,408 logical bytes"
        )
    if source["profile"] == "OFFICIAL_GLM52_BF16":
        expected_official_paths = {
            f"model-{index:05d}-of-00282.safetensors"
            for index in range(1, OFFICIAL_WEIGHT_SHARD_COUNT + 1)
        }
        if set(paths) != expected_official_paths:
            raise StateError("official GLM-5.2 BF16 contract shard filenames are incomplete")
    tensors = contract.get("tensors")
    if not isinstance(tensors, dict) or set(tensors) != {"expected_tensor_count", "names"}:
        raise StateError("expected campaign contract tensor schema invalid")
    tensor_names = tensors.get("names")
    if not isinstance(tensor_names, list) or not tensor_names \
            or len(tensor_names) != len(set(tensor_names)) \
            or any(not isinstance(name, str) or not name for name in tensor_names):
        raise StateError("expected tensor names must be a non-empty unique list")
    if tensors.get("expected_tensor_count") != len(tensor_names):
        raise StateError("expected tensor count does not equal its authoritative list")
    schedule = contract.get("window_schedule")
    if not isinstance(schedule, list) or not schedule:
        raise StateError("expected campaign contract requires a non-empty window schedule")
    schedule_keys = {
        "schedule_index",
        "window_id",
        "source_shards",
        "carry_in_shards",
        "new_fetch_shards",
        "refetch_shards",
        "carry_out_shards",
        "evict_shards",
        "tensor_set",
    }
    expected_paths = set(paths)
    seen_paths: set[str] = set()
    new_fetch_counts: Counter[str] = Counter()
    tensor_owners: dict[str, str] = {}
    previous_carry: set[str] = set()
    last_disposition: dict[str, str] = {}
    seen_window_ids: set[str] = set()
    for index, item in enumerate(schedule):
        if not isinstance(item, dict) or set(item) != schedule_keys:
            raise StateError(f"expected window schedule entry {index} schema invalid")
        if item.get("schedule_index") != index:
            raise StateError("window schedule indices must be contiguous from zero")
        window_id = _require_name(item.get("window_id"), f"window_schedule[{index}].window_id")
        if window_id in seen_window_ids:
            raise StateError(f"window schedule repeats window_id: {window_id}")
        seen_window_ids.add(window_id)
        sets: dict[str, set[str]] = {}
        for key in (
            "source_shards",
            "carry_in_shards",
            "new_fetch_shards",
            "refetch_shards",
            "carry_out_shards",
            "evict_shards",
        ):
            raw = item.get(key)
            if not isinstance(raw, list) or len(raw) != len(set(raw)) \
                    or any(not isinstance(path, str) or not path for path in raw):
                raise StateError(f"window_schedule[{index}].{key} must be a unique path list")
            sets[key] = set(raw)
            if not sets[key].issubset(expected_paths):
                raise StateError(f"window_schedule[{index}].{key} contains an unexpected shard")
        if sets["carry_in_shards"] != previous_carry:
            raise StateError(f"window_schedule[{index}] carry_in != prior carry_out")
        fetched_here = sets["new_fetch_shards"] | sets["refetch_shards"]
        if sets["new_fetch_shards"] & sets["refetch_shards"] \
                or sets["carry_in_shards"] & fetched_here:
            raise StateError(f"window_schedule[{index}] source acquisition sets overlap")
        if sets["source_shards"] != sets["carry_in_shards"] | fetched_here:
            raise StateError(f"window_schedule[{index}] source_shards partition invalid")
        if sets["carry_out_shards"] & sets["evict_shards"] \
                or sets["source_shards"] != sets["carry_out_shards"] | sets["evict_shards"]:
            raise StateError(f"window_schedule[{index}] carry_out/evict partition invalid")
        for path in sets["new_fetch_shards"]:
            if path in seen_paths:
                raise StateError(f"shard is marked new-fetch more than once: {path}")
            new_fetch_counts[path] += 1
        for path in sets["refetch_shards"]:
            if path not in seen_paths or last_disposition.get(path) != "EVICTED":
                raise StateError(f"refetch is not of a previously evicted shard: {path}")
        window_tensors = item.get("tensor_set")
        if not isinstance(window_tensors, list) \
                or len(window_tensors) != len(set(window_tensors)):
            raise StateError(f"window_schedule[{index}].tensor_set must be unique")
        for tensor in window_tensors:
            if tensor not in tensor_names:
                raise StateError(f"window schedule contains unexpected tensor: {tensor}")
            if tensor in tensor_owners:
                raise StateError(f"tensor assigned to multiple scheduled windows: {tensor}")
            tensor_owners[tensor] = window_id
        seen_paths.update(sets["source_shards"])
        for path in sets["carry_out_shards"]:
            last_disposition[path] = "CARRIED"
        for path in sets["evict_shards"]:
            last_disposition[path] = "EVICTED"
        previous_carry = sets["carry_out_shards"]
    if previous_carry:
        raise StateError("final scheduled window must carry out no source shards")
    if set(new_fetch_counts) != expected_paths or any(count != 1 for count in new_fetch_counts.values()):
        raise StateError("every expected source shard must appear as new-fetch exactly once")
    if set(tensor_owners) != set(tensor_names):
        raise StateError("window schedule must own every expected tensor exactly once")
    gates = contract.get("state_gates")
    if not isinstance(gates, dict):
        raise StateError("expected campaign contract state_gates must be an object")
    validated_gates = {state: _validate_state_gate(state, gate) for state, gate in gates.items()}
    for required_state in ("ASSEMBLE_ARTIFACT", "COMPLETE"):
        if required_state not in validated_gates:
            raise StateError(f"expected campaign contract lacks required gate {required_state}")
    assembly = validated_gates["ASSEMBLE_ARTIFACT"]
    if not all(assembly[key] for key in (
        "require_source_complete", "require_tensor_complete", "require_final_source_eviction",
        "require_telegram_delivery",
    )):
        raise StateError("ASSEMBLE_ARTIFACT gate must require source/tensor/eviction/Telegram")
    complete = validated_gates["COMPLETE"]
    if not all(complete[key] for key in (
        "require_source_complete", "require_tensor_complete", "require_final_source_eviction",
        "require_telegram_delivery", "require_phone_status",
    )) or not complete["required_artifacts"] or not complete["required_checklist"]:
        raise StateError("COMPLETE gate lacks mandatory campaign closure requirements")
    if source["profile"] == "OFFICIAL_GLM52_BF16":
        if not OFFICIAL_ASSEMBLY_REQUIRED_ARTIFACTS.issubset(
            assembly["required_artifacts"]
        ):
            raise StateError("official ASSEMBLE gate lacks mandatory source/schedule artifacts")
        for name in OFFICIAL_ASSEMBLY_REQUIRED_ARTIFACTS:
            policy = assembly["required_artifacts"][name]
            if policy["expected_seal_sha256"] is None and not (
                name == "streaming_schedule"
                and policy["validator_id"] == "frozen_schedule_v2"
                and policy["require_producer_hmac"] is True
            ):
                raise StateError(
                    f"official ASSEMBLE artifact requires a frozen expected seal: {name}"
                )
        if not OFFICIAL_COMPLETE_REQUIRED_ARTIFACTS.issubset(
            complete["required_artifacts"]
        ):
            raise StateError("official COMPLETE gate lacks mandatory final artifacts")
        if not set(MANDATORY_COMPLETE_STOP_CONDITIONS).issubset(
            complete["required_checklist"]
        ):
            raise StateError("official COMPLETE gate lacks the mandatory 30 stop conditions")
    _require_name(contract.get("created_at"), "expected campaign contract created_at")
    return contract


def make_expected_campaign_contract(
    *,
    campaign_id: str,
    source_revision: str,
    expected_chat_identity_digest: str,
    source_shards: Sequence[Mapping[str, Any]],
    expected_tensors: Sequence[str],
    window_schedule: Sequence[Mapping[str, Any]],
    state_gates: Mapping[str, Mapping[str, Any]],
    source_profile: str = "OFFICIAL_GLM52_BF16",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build and validate the frozen authoritative campaign-completeness contract."""
    shards = [_clone(dict(item)) for item in source_shards]
    tensors = _clone(list(expected_tensors))
    contract = seal(
        {
            "schema": EXPECTED_CONTRACT_SCHEMA,
            "campaign_id": campaign_id,
            "source_revision": source_revision,
            "expected_chat_identity_digest": expected_chat_identity_digest,
            "source": {
                "profile": source_profile,
                "expected_shard_count": len(shards),
                "expected_logical_bytes": sum(int(item["logical_bytes"]) for item in shards),
                "shards": shards,
            },
            "tensors": {"expected_tensor_count": len(tensors), "names": tensors},
            "window_schedule": _clone(list(window_schedule)),
            "state_gates": _clone(dict(state_gates)),
            "created_at": created_at or utc_now(),
        }
    )
    return _validate_expected_contract(contract)


def make_state_terminal_evidence(
    contract: Mapping[str, Any],
    state: str,
    *,
    artifact_root: str | os.PathLike[str],
    controller_anchor: Mapping[str, Any],
    evidence_auth: TelegramAuthConfig | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Snapshot a gated state's evidence from the trusted filesystem.

    Callers do not supply hashes or statuses.  The controller opens each path named by
    the frozen contract and records what was actually parsed and verified.
    """
    validated = _validate_expected_contract(contract)
    anchor = _validate_controller_anchor(controller_anchor)
    store = TrustedArtifactStore(artifact_root)
    gate = validated["state_gates"].get(state)
    if gate is None:
        raise StateError(f"state {state} has no terminal evidence gate")
    artifacts = {
        name: _snapshot_policy_evidence(
            store,
            policy,
            label=f"{state} artifact {name}",
            contract=validated,
            evidence_auth=evidence_auth,
        )
        for name, policy in gate["required_artifacts"].items()
    }
    checklist = {
        item: _snapshot_policy_evidence(
            store,
            policy,
            label=f"{state} checklist {item}",
            contract=validated,
            stop_condition=item,
            evidence_auth=evidence_auth,
        )
        for item, policy in gate["required_checklist"].items()
    }
    phone = None
    if gate["require_phone_status"]:
        phone_policy = {
            "path": gate["required_phone_status_path"],
            "expected_seal_sha256": None,
            "expected_schema": "hawking.glm52.phone_status.v2",
            "allowed_statuses": ["GREEN"],
            "validator_id": "phone_status_v2",
            "validator_source_sha256": EVIDENCE_VALIDATOR_SOURCE_SHA256[
                "phone_status_v2"
            ],
            "require_producer_hmac": True,
        }
        phone = _snapshot_policy_evidence(
            store,
            phone_policy,
            label=f"{state} phone status",
            contract=validated,
            phone_anchor=anchor,
            evidence_auth=evidence_auth,
        )
    evidence = seal(
        {
            "schema": TERMINAL_EVIDENCE_SCHEMA,
            "state": state,
            "expected_contract_sha256": validated["seal_sha256"],
            "controller_anchor_sha256": anchor["anchor_sha256"],
            "artifact_seals": artifacts,
            "checklist": checklist,
            "phone_status": phone,
            "created_at": created_at or utc_now(),
        }
    )
    _validate_terminal_evidence(
        validated,
        state,
        evidence,
        artifact_store=store,
        controller_anchor=anchor,
        evidence_auth=evidence_auth,
    )
    return evidence


def _validate_xet_autotune_result_artifact(
    store: "TrustedArtifactStore",
    artifact: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    label: str,
) -> None:
    """Rebuild and validate the exact raw Xet result wrapped for campaign use."""
    plan_policy_raw = contract.get("state_gates", {}).get("AUTOTUNE_XET", {}).get(
        "required_artifacts", {}
    ).get("xet_autotune_plan")
    if not isinstance(plan_policy_raw, dict):
        raise StateError(f"{label} contract lacks the frozen Xet autotune plan")
    plan_policy = _validate_evidence_policy(
        plan_policy_raw,
        label=f"{label} frozen Xet plan",
    )
    if plan_policy["expected_seal_sha256"] is None:
        raise StateError(f"{label} frozen Xet plan has no expected seal")
    plan, _ = store.read_sealed(
        plan_policy["path"],
        label=f"{label} frozen Xet plan",
    )
    if plan.get("seal_sha256") != plan_policy["expected_seal_sha256"] \
            or plan.get("schema") != plan_policy["expected_schema"] \
            or plan.get("status") not in plan_policy["allowed_statuses"]:
        raise StateError(f"{label} frozen Xet plan differs from the expected contract")

    semantic_evidence = artifact.get("evidence")
    raw_seal = (
        semantic_evidence.get("raw_xet_autotune_result_seal_sha256")
        if isinstance(semantic_evidence, dict)
        else None
    )
    if not _is_sha256(raw_seal):
        raise StateError(f"{label} lacks the original raw Xet result seal")
    if not isinstance(semantic_evidence, dict) or not _is_sha256(
        semantic_evidence.get("controller_anchor_sha256")
    ):
        raise StateError(f"{label} lacks its producing controller anchor")
    try:
        import glm52_xet_live as xet_live

        raw_fields = set(xet_live.AUTOTUNE_RESULT_FIELDS)
        if "seal_sha256" not in raw_fields \
                or any(field not in artifact for field in raw_fields - {"seal_sha256"}):
            raise StateError(f"{label} lacks raw live-Xet result fields")
        raw_result = {
            field: _clone(artifact[field])
            for field in raw_fields
            if field != "seal_sha256"
        }
        raw_result["seal_sha256"] = raw_seal
        xet_live.validate_autotune_result(raw_result, plan=plan)
    except StateError:
        raise
    except (Glm52Error, ImportError, KeyError, TypeError, ValueError) as exc:
        raise StateError(f"{label} raw live-Xet result validation failed: {exc}") from exc


def _snapshot_policy_evidence(
    store: "TrustedArtifactStore",
    policy: Mapping[str, Any],
    *,
    label: str,
    contract: Mapping[str, Any],
    stop_condition: str | None = None,
    phone_anchor: Mapping[str, Any] | None = None,
    evidence_auth: TelegramAuthConfig | None = None,
) -> dict[str, Any]:
    normalized = _validate_evidence_policy(policy, label=label)
    artifact, file_sha256 = store.read_sealed(normalized["path"], label=label)
    if artifact.get("schema") != normalized["expected_schema"]:
        raise StateError(f"{label} schema mismatch")
    if artifact.get("status") not in normalized["allowed_statuses"]:
        raise StateError(f"{label} status is not allowed by the frozen contract")
    if normalized["expected_seal_sha256"] is not None \
            and artifact.get("seal_sha256") != normalized["expected_seal_sha256"]:
        raise StateError(f"{label} seal differs from the frozen contract")
    validator_id = normalized["validator_id"]
    if validator_id == "future_artifact_blocked_v1":
        raise StateError(
            f"{label} has no registered schema-specific scientific validator"
        )
    if validator_id == "stop_condition_blocked_v1":
        raise StateError(
            f"{label} is not yet derived from grounded campaign evidence"
        )
    if validator_id in {
        "campaign_artifact_v1", "stop_condition_v1", "xet_autotune_result_v1",
        "frozen_schedule_v2",
    }:
        for key, expected in (
            ("campaign_id", contract["campaign_id"]),
            ("source_revision", contract["source_revision"]),
            ("expected_contract_sha256", contract["seal_sha256"]),
        ):
            if artifact.get(key) != expected:
                raise StateError(f"{label} {key} identity mismatch")
        semantic_evidence = artifact.get("evidence")
        if not isinstance(semantic_evidence, dict) or not semantic_evidence:
            raise StateError(f"{label} semantic evidence is empty")
        if artifact.get("evidence_sha256") != _sha256(semantic_evidence):
            raise StateError(f"{label} semantic evidence hash mismatch")
    if validator_id == "xet_autotune_result_v1":
        _validate_xet_autotune_result_artifact(
            store,
            artifact,
            contract=contract,
            label=label,
        )
    if validator_id == "frozen_schedule_v2":
        if artifact.get("repo") not in (None, "zai-org/GLM-5.2") \
                or artifact.get("revision") != contract["source_revision"]:
            raise StateError(f"{label} official source identity mismatch")
        binding = artifact.get("autotune_binding")
        profile = artifact.get("selected_profile")
        if not isinstance(binding, dict) or not isinstance(profile, dict) or not profile:
            raise StateError(f"{label} lacks post-autotune binding/profile")
        for key in (
            "xet_autotune_result_seal_sha256", "xet_autotune_plan_seal_sha256",
            "preliminary_schedule_seal_sha256", "selected_profile_sha256",
        ):
            if not _is_sha256(binding.get(key)):
                raise StateError(f"{label} autotune binding {key} invalid")
        if binding["selected_profile_sha256"] != _sha256(profile):
            raise StateError(f"{label} selected profile hash mismatch")
        expected_plan = contract["state_gates"].get("AUTOTUNE_XET", {}).get(
            "required_artifacts", {}
        ).get("xet_autotune_plan", {}).get("expected_seal_sha256")
        expected_preliminary = contract["state_gates"].get(
            "ASSEMBLE_ARTIFACT", {}
        ).get("required_artifacts", {}).get(
            "preliminary_streaming_schedule", {}
        ).get("expected_seal_sha256")
        if binding["xet_autotune_plan_seal_sha256"] != expected_plan \
                or binding["preliminary_schedule_seal_sha256"] != expected_preliminary:
            raise StateError(f"{label} plan/preliminary schedule binding mismatch")
        windows = artifact.get("windows")
        if not isinstance(windows, list) or len(windows) != len(contract["window_schedule"]):
            raise StateError(f"{label} window inventory mismatch")
        normalized_windows: list[dict[str, Any]] = []
        for index, window in enumerate(windows):
            if not isinstance(window, dict):
                raise StateError(f"{label} window {index} invalid")
            normalized_windows.append({
                "schedule_index": window.get("schedule_index", index),
                "window_id": window.get("window_id"),
                "source_shards": window.get("source_shards"),
                "carry_in_shards": window.get("carry_in_shards"),
                "new_fetch_shards": window.get("new_fetch_shards"),
                "refetch_shards": window.get("refetch_shards"),
                "carry_out_shards": window.get("carry_out_shards"),
                "evict_shards": window.get(
                    "evict_shards", window.get("evict_after_seal_shards")
                ),
                "tensor_set": window.get("tensor_set"),
            })
        if normalized_windows != contract["window_schedule"]:
            raise StateError(f"{label} changed frozen window dependency membership")
    if normalized["require_producer_hmac"]:
        if not isinstance(evidence_auth, TelegramAuthConfig):
            raise StateError(f"{label} requires configured producer authentication")
        signature = artifact.get("producer_hmac_sha256")
        producer_body = {
            key: item for key, item in artifact.items()
            if key not in {"seal_sha256", "producer_hmac_sha256"}
        }
        if not evidence_auth.verify(
            {
                "schema": "hawking.glm52.evidence_producer_auth.v1",
                "artifact": producer_body,
            },
            signature,
        ):
            raise StateError(f"{label} producer HMAC authentication failed")
    if stop_condition is not None:
        if artifact.get("stop_condition") != stop_condition:
            raise StateError(f"{label} stop-condition identity mismatch")
        for key, expected in (
            ("campaign_id", contract["campaign_id"]),
            ("source_revision", contract["source_revision"]),
            ("expected_contract_sha256", contract["seal_sha256"]),
        ):
            if artifact.get(key) != expected:
                raise StateError(f"{label} {key} identity mismatch")
    if phone_anchor is not None:
        _validate_phone_artifact(artifact, contract=contract, anchor=phone_anchor, label=label)
    return {
        "path": normalized["path"],
        "file_sha256": file_sha256,
        "seal_sha256": artifact["seal_sha256"],
        "schema": artifact["schema"],
        "status": artifact["status"],
    }


def _validate_phone_artifact(
    artifact: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    anchor: Mapping[str, Any],
    label: str,
) -> None:
    if artifact.get("overall_status") != "GREEN" or artifact.get("status") != "GREEN":
        raise StateError(f"{label} is not GREEN")
    for key, expected in (
        ("campaign_id", contract["campaign_id"]),
        ("source_revision", contract["source_revision"]),
        ("controller_epoch", anchor["controller_epoch"]),
        ("expected_contract_sha256", contract["seal_sha256"]),
    ):
        if artifact.get(key) != expected:
            raise StateError(f"{label} {key} identity mismatch")
    if not _is_sha256(artifact.get("cli_config_sha256")):
        raise StateError(f"{label} cli_config_sha256 invalid")
    controller = artifact.get("controller")
    if not isinstance(controller, dict) or controller.get("durable_state_ok") is not True \
            or controller.get("live_worker_lease_ok") is not True \
            or controller.get("heartbeat_fresh_ok") is not True \
            or controller.get("heartbeat_max_age_seconds") != \
            CONTROLLER_HEARTBEAT_MAX_AGE_SECONDS:
        raise StateError(f"{label} controller is not durably green")
    heartbeat_fresh, heartbeat_reason, heartbeat_at = _heartbeat_freshness(
        {"at": controller.get("heartbeat_at")}
    )
    if not heartbeat_fresh or controller.get("heartbeat_at") != heartbeat_at:
        raise StateError(f"{label} controller heartbeat is not currently fresh: {heartbeat_reason}")
    operator = artifact.get("operator_control")
    applied = operator.get("applied") if isinstance(operator, dict) else None
    if not isinstance(operator, dict) or not isinstance(applied, dict) \
            or operator.get("application_state") != "IN_SYNC" \
            or operator.get("requested_sequence") != applied.get("applied_request_sequence") \
            or operator.get("requested_action") != applied.get("applied_action"):
        raise StateError(f"{label} operator control is not applied/in sync")
    if artifact.get("checkpoint_anchor") != anchor["checkpoint"]:
        raise StateError(f"{label} checkpoint anchor mismatch")


def _validate_terminal_evidence(
    contract: Mapping[str, Any],
    state: str,
    evidence: Any,
    *,
    artifact_store: "TrustedArtifactStore",
    controller_anchor: Mapping[str, Any],
    evidence_auth: TelegramAuthConfig | None = None,
) -> dict[str, Any]:
    gates = contract["state_gates"]
    if state not in gates:
        if evidence is not None:
            raise StateError(f"state {state} does not accept terminal_evidence")
        return {}
    value = _require_object(evidence, f"{state} terminal_evidence")
    try:
        verify_sealed(value, label=f"{state} terminal_evidence")
    except Glm52Error as exc:
        raise StateError(str(exc)) from exc
    if set(value) != {
        "schema", "state", "expected_contract_sha256", "controller_anchor_sha256",
        "artifact_seals", "checklist", "phone_status", "created_at", "seal_sha256",
    }:
        raise StateError(f"{state} terminal evidence schema fields invalid")
    if value.get("schema") != TERMINAL_EVIDENCE_SCHEMA or value.get("state") != state \
            or value.get("expected_contract_sha256") != contract["seal_sha256"]:
        raise StateError(f"{state} terminal evidence identity mismatch")
    anchor = _validate_controller_anchor(controller_anchor)
    if value.get("controller_anchor_sha256") != anchor["anchor_sha256"]:
        raise StateError(f"{state} terminal evidence controller anchor mismatch")
    gate = gates[state]
    artifacts = value.get("artifact_seals")
    if not isinstance(artifacts, dict) or set(artifacts) != set(gate["required_artifacts"]):
        raise StateError(f"{state} terminal artifact inventory is incomplete or unexpected")
    for name, expected in gate["required_artifacts"].items():
        actual = artifacts[name]
        if not isinstance(actual, dict) or set(actual) != {
            "path", "file_sha256", "seal_sha256", "schema", "status",
        }:
            raise StateError(f"{state} terminal artifact {name} evidence schema invalid")
        grounded = _snapshot_policy_evidence(
            artifact_store,
            expected,
            label=f"{state} artifact {name}",
            contract=contract,
            evidence_auth=evidence_auth,
        )
        if actual != grounded:
            raise StateError(f"{state} terminal artifact {name} does not match trusted bytes")
    if "frozen_streaming_schedule" in artifacts and "xet_autotune_result" in artifacts:
        schedule_path = gate["required_artifacts"]["frozen_streaming_schedule"]["path"]
        schedule_document, _ = artifact_store.read_sealed(
            schedule_path, label=f"{state} frozen streaming schedule cross-binding"
        )
        binding = schedule_document.get("autotune_binding")
        if not isinstance(binding, dict) or binding.get(
            "xet_autotune_result_seal_sha256"
        ) != artifacts["xet_autotune_result"]["seal_sha256"]:
            raise StateError(f"{state} frozen schedule does not bind exact Xet result")
        result_path = gate["required_artifacts"]["xet_autotune_result"]["path"]
        result_document, _ = artifact_store.read_sealed(
            result_path, label=f"{state} Xet result cross-binding"
        )
        selections = result_document.get("selections")
        expected_profile = (
            {
                "acquisition": selections.get("acquisition"),
                "steady": selections.get("steady"),
            }
            if isinstance(selections, dict)
            else None
        )
        if expected_profile is None or expected_profile["acquisition"] is None \
                or expected_profile["steady"] is None \
                or expected_profile != schedule_document.get("selected_profile"):
            raise StateError(f"{state} Xet result/frozen schedule selected profile mismatch")
    checklist = value.get("checklist")
    if not isinstance(checklist, dict) or set(checklist) != set(gate["required_checklist"]):
        raise StateError(f"{state} stop-condition checklist is incomplete or unexpected")
    for item, result in checklist.items():
        if not isinstance(result, dict):
            raise StateError(f"{state} stop condition is not evidenced PASS: {item}")
        grounded = _snapshot_policy_evidence(
            artifact_store,
            gate["required_checklist"][item],
            label=f"{state} checklist {item}",
            contract=contract,
            stop_condition=item,
            evidence_auth=evidence_auth,
        )
        if result != grounded:
            raise StateError(f"{state} stop condition does not match trusted bytes: {item}")
    phone = value.get("phone_status")
    if gate["require_phone_status"]:
        phone_policy = {
            "path": gate["required_phone_status_path"],
            "expected_seal_sha256": None,
            "expected_schema": "hawking.glm52.phone_status.v2",
            "allowed_statuses": ["GREEN"],
            "validator_id": "phone_status_v2",
            "validator_source_sha256": EVIDENCE_VALIDATOR_SOURCE_SHA256[
                "phone_status_v2"
            ],
            "require_producer_hmac": True,
        }
        grounded = _snapshot_policy_evidence(
            artifact_store,
            phone_policy,
            label=f"{state} phone status",
            contract=contract,
            phone_anchor=anchor,
            evidence_auth=evidence_auth,
        )
        if phone != grounded:
            raise StateError(f"{state} requires sealed phone status evidence")
    elif phone is not None:
        raise StateError(f"{state} does not permit phone status evidence")
    _require_name(value.get("created_at"), f"{state} terminal evidence created_at")
    return value


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        result = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise StateError(f"invalid JSON in {label}: {exc}") from exc
    if not isinstance(result, dict):
        raise StateError(f"{label} is not a JSON object")
    return result


class TrustedArtifactStore:
    """Read evidence beneath one explicit root without following symlinks.

    Each path component is opened relative to an already-open directory with
    ``O_NOFOLLOW``.  The final regular file is read from that descriptor exactly once,
    so validation cannot accidentally hash one path and parse another.
    """

    def __init__(self, root: str | os.PathLike[str], *, max_bytes: int = 256 * 1024 * 1024):
        self.root = Path(root)
        try:
            root_stat = self.root.lstat()
        except OSError as exc:
            raise StateError(f"trusted artifact root is unavailable: {exc}") from exc
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise StateError("trusted artifact root must be a real directory, not a symlink")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise StateError("trusted artifact read limit must be positive")
        self.max_bytes = max_bytes

    @staticmethod
    def _parts(relative_path: Any) -> tuple[str, ...]:
        path = _require_name(relative_path, "trusted artifact relative path")
        candidate = Path(path)
        if candidate.is_absolute() or path in {".", ".."}:
            raise StateError("trusted artifact path must be relative to its root")
        parts = candidate.parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise StateError("trusted artifact path escapes or ambiguously names its root")
        return tuple(parts)

    def read_bytes(self, relative_path: str) -> bytes:
        parts = self._parts(relative_path)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow
        descriptor = os.open(self.root, directory_flags)
        try:
            for component in parts[:-1]:
                child = os.open(component, directory_flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            file_descriptor = os.open(parts[-1], os.O_RDONLY | nofollow, dir_fd=descriptor)
            try:
                metadata = os.fstat(file_descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise StateError(f"trusted evidence is not a regular file: {relative_path}")
                if metadata.st_size > self.max_bytes:
                    raise StateError(f"trusted evidence exceeds read limit: {relative_path}")
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = os.read(file_descriptor, min(1024 * 1024, self.max_bytes + 1 - total))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > self.max_bytes:
                        raise StateError(f"trusted evidence exceeds read limit: {relative_path}")
                return b"".join(chunks)
            finally:
                os.close(file_descriptor)
        except OSError as exc:
            raise StateError(f"cannot securely open trusted evidence {relative_path}: {exc}") from exc
        finally:
            os.close(descriptor)

    def read_sealed(self, relative_path: str, *, label: str) -> tuple[dict[str, Any], str]:
        raw = self.read_bytes(relative_path)
        value = _strict_json(raw, label=label)
        try:
            verify_sealed(value, label=label)
        except Glm52Error as exc:
            raise StateError(str(exc)) from exc
        return value, hashlib.sha256(raw).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class HashChainLog:
    """Strict append-only JSONL chain.  Writers must already hold the controller lease."""

    _KEYS = frozenset(
        {"schema", "seq", "kind", "claim_id", "at", "prev_hash", "payload", "chain_sha256"}
    )

    def __init__(self, path: str | os.PathLike[str], *, schema: str):
        self.path = Path(path)
        self.schema = schema

    def verified_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            raise StateError(f"cannot read event log {self.path}: {exc}") from exc
        if not raw:
            return []
        if not raw.endswith(b"\n"):
            raise StateError(f"unsealed/torn JSONL tail in {self.path}")
        events: list[dict[str, Any]] = []
        previous = GENESIS_HASH
        claims: set[str] = set()
        for seq, line in enumerate(raw.splitlines()):
            if not line:
                raise StateError(f"blank JSONL record at seq {seq} in {self.path}")
            event = _strict_json(line, label=f"{self.path}:seq={seq}")
            if set(event) != self._KEYS:
                raise StateError(f"unexpected/missing event fields at seq {seq} in {self.path}")
            if event.get("schema") != self.schema:
                raise StateError(f"event schema mismatch at seq {seq} in {self.path}")
            if event.get("seq") != seq or isinstance(event.get("seq"), bool):
                raise StateError(f"event sequence break at expected seq {seq} in {self.path}")
            claim_id = event.get("claim_id")
            _require_claim(claim_id)
            if claim_id in claims:
                raise StateError(f"one-use claim repeated in {self.path}: {claim_id}")
            claims.add(claim_id)
            _require_name(event.get("kind"), f"event[{seq}].kind")
            _require_name(event.get("at"), f"event[{seq}].at")
            _require_object(event.get("payload"), f"event[{seq}].payload")
            if event.get("prev_hash") != previous:
                raise StateError(f"event prev_hash break at seq {seq} in {self.path}")
            recorded = event.get("chain_sha256")
            unsigned = {key: value for key, value in event.items() if key != "chain_sha256"}
            expected = _sha256(unsigned)
            if not _is_sha256(recorded) or recorded != expected:
                raise StateError(f"event chain hash mismatch at seq {seq} in {self.path}")
            previous = recorded
            events.append(event)
        return events

    def verify_chain(self) -> tuple[bool, list[str]]:
        try:
            self.verified_events()
        except StateError as exc:
            return False, [str(exc)]
        return True, []

    def head_hash(self, events: Sequence[Mapping[str, Any]] | None = None) -> str:
        records = list(events) if events is not None else self.verified_events()
        return str(records[-1]["chain_sha256"]) if records else GENESIS_HASH

    def find_claim(self, claim_id: str) -> dict[str, Any] | None:
        _require_claim(claim_id)
        for event in self.verified_events():
            if event["claim_id"] == claim_id:
                return event
        return None

    def append(self, kind: str, claim_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        _require_name(kind, "kind")
        _require_claim(claim_id)
        payload_copy = _require_object(dict(payload), "event payload")
        events = self.verified_events()
        if any(event["claim_id"] == claim_id for event in events):
            raise StateError(f"one-use claim already consumed: {claim_id}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, Any] = {
            "schema": self.schema,
            "seq": len(events),
            "kind": kind,
            "claim_id": claim_id,
            "at": utc_now(),
            "prev_hash": self.head_hash(events),
            "payload": payload_copy,
        }
        body["chain_sha256"] = _sha256(body)
        encoded = canonical(body) + b"\n"
        existed = self.path.exists()
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise StateError(f"short append to {self.path}")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if not existed:
            _fsync_directory(self.path.parent)
        verified = self.verified_events()
        if not verified or verified[-1]["chain_sha256"] != body["chain_sha256"]:
            raise StateError(f"post-append verification failed for {self.path}")
        return body


_PROCESS_LEASES: set[str] = set()
_PROCESS_LEASES_LOCK = threading.Lock()


class SingletonLease:
    """Exclusive non-blocking controller lease with a sealed diagnostic owner record."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        campaign_id: str,
        controller_epoch: str,
        owner: str = "glm52-controller",
    ):
        self.path = Path(path)
        self.campaign_id = campaign_id
        self.controller_epoch = controller_epoch
        self.owner = owner
        self._handle: TextIO | None = None
        self._registry_key: str | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def assert_held(self) -> None:
        if not self.held:
            raise StateError("controller mutation refused: singleton lease is not held")

    def acquire(self) -> "SingletonLease":
        if self.held:
            raise StateError("singleton lease already held by this handle")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = str(self.path.resolve())
        with _PROCESS_LEASES_LOCK:
            if key in _PROCESS_LEASES:
                raise StateError(f"already-running: singleton lease held in this process: {self.path}")
            _PROCESS_LEASES.add(key)
        handle: TextIO | None = None
        try:
            handle = self.path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StateError(f"already-running: singleton lease held: {self.path}") from exc
            stamp = seal(
                {
                    "schema": LEASE_SCHEMA,
                    "campaign_id": self.campaign_id,
                    "controller_epoch": self.controller_epoch,
                    "owner": self.owner,
                    "pid": os.getpid(),
                    "acquired_at": utc_now(),
                }
            )
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(stamp, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            _fsync_directory(self.path.parent)
            self._handle = handle
            self._registry_key = key
            return self
        except BaseException:
            if handle is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
                handle.close()
            with _PROCESS_LEASES_LOCK:
                _PROCESS_LEASES.discard(key)
            raise

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None
            if self._registry_key is not None:
                with _PROCESS_LEASES_LOCK:
                    _PROCESS_LEASES.discard(self._registry_key)
                self._registry_key = None

    def probe(self) -> dict[str, Any]:
        """Observe live flock ownership without mutating the owner record."""
        if not self.path.exists():
            return {
                "lock_state": "ABSENT",
                "live_lock_held": False,
                "held_by_this_handle": self.held,
                "owner_record_ok": False,
                "owner": None,
                "owner_pid": None,
                "owner_pid_alive": False,
                "controller_epoch": None,
            }
        live_lock = self.held
        descriptor: int | None = None
        if not self.held:
            try:
                descriptor = os.open(self.path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    live_lock = True
                else:
                    live_lock = False
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                live_lock = False
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        owner_record: dict[str, Any] | None = None
        try:
            raw = self.path.read_bytes()
            owner_record = _strict_json(raw, label="controller lease owner record")
            verify_sealed(owner_record, label="controller lease owner record")
            if owner_record.get("schema") != LEASE_SCHEMA \
                    or owner_record.get("campaign_id") != self.campaign_id:
                owner_record = None
        except (OSError, StateError, Glm52Error):
            owner_record = None
        owner_pid = owner_record.get("pid") if owner_record is not None else None
        owner_pid_alive = False
        if not isinstance(owner_pid, bool) and isinstance(owner_pid, int) and owner_pid > 0:
            try:
                os.kill(owner_pid, 0)
                owner_pid_alive = True
            except PermissionError:
                owner_pid_alive = True
            except ProcessLookupError:
                owner_pid_alive = False
        return {
            "lock_state": (
                "HELD_BY_THIS_HANDLE" if self.held
                else "HELD_BY_OTHER_PROCESS" if live_lock
                else "UNLOCKED"
            ),
            "live_lock_held": live_lock,
            "held_by_this_handle": self.held,
            "owner_record_ok": owner_record is not None,
            "owner": owner_record.get("owner") if owner_record is not None else None,
            "owner_pid": owner_pid,
            "owner_pid_alive": owner_pid_alive,
            "controller_epoch": (
                owner_record.get("controller_epoch") if owner_record is not None else None
            ),
        }

    def __enter__(self) -> "SingletonLease":
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _controller_anchor_hash(value: Mapping[str, Any]) -> str:
    return _sha256({key: item for key, item in value.items() if key != "anchor_sha256"})


def _validate_controller_anchor(value: Any) -> dict[str, Any]:
    anchor = _require_object(value, "controller anchor")
    required = {
        "schema", "campaign_id", "source_revision", "controller_epoch",
        "expected_contract_sha256", "from_state", "checkpoint", "anchor_sha256",
    }
    if set(anchor) != required or anchor.get("schema") != CONTROLLER_ANCHOR_SCHEMA:
        raise StateError("controller anchor schema mismatch")
    _require_name(anchor.get("campaign_id"), "controller anchor campaign_id")
    if not isinstance(anchor.get("source_revision"), str) \
            or _HEX40_RE.fullmatch(anchor["source_revision"]) is None:
        raise StateError("controller anchor source_revision invalid")
    _require_name(anchor.get("controller_epoch"), "controller anchor controller_epoch")
    if not _is_sha256(anchor.get("expected_contract_sha256")):
        raise StateError("controller anchor expected contract hash invalid")
    if anchor.get("from_state") is not None and anchor.get("from_state") not in STATES:
        raise StateError("controller anchor from_state invalid")
    checkpoint = anchor.get("checkpoint")
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "event_count", "event_head_hash", "window_event_count", "window_event_head_hash",
        "checkpoint_seal_sha256",
    }:
        raise StateError("controller checkpoint anchor schema invalid")
    for key in ("event_count", "window_event_count"):
        if isinstance(checkpoint[key], bool) or not isinstance(checkpoint[key], int) \
                or checkpoint[key] < 0:
            raise StateError(f"controller checkpoint anchor {key} invalid")
    for key in ("event_head_hash", "window_event_head_hash"):
        if not _is_sha256(checkpoint[key]):
            raise StateError(f"controller checkpoint anchor {key} invalid")
    checkpoint_seal = checkpoint["checkpoint_seal_sha256"]
    if checkpoint_seal is not None and not _is_sha256(checkpoint_seal):
        raise StateError("controller checkpoint anchor seal invalid")
    if anchor.get("anchor_sha256") != _controller_anchor_hash(anchor):
        raise StateError("controller anchor hash mismatch")
    return anchor


def _campaign_status_hash(status: Mapping[str, Any]) -> str:
    return _sha256({"schema": CAMPAIGN_STATUS_SCHEMA, "status": dict(status)})


def _validate_campaign_status(value: Any) -> dict[str, Any]:
    status = _require_object(value, "canonical campaign status")
    required = {
        "state", "source_coverage_percent", "shards", "network_bytes",
        "throughput_bytes_per_second", "eta_seconds", "current", "candidate_rates",
        "best_metrics", "resources", "process",
    }
    if set(status) != required or status.get("state") not in STATES:
        raise StateError("canonical campaign status fields/state invalid")
    coverage = status.get("source_coverage_percent")
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) \
            or not math.isfinite(float(coverage)) or not (0 <= float(coverage) <= 100):
        raise StateError("canonical campaign status source coverage invalid")
    shards = status.get("shards")
    if not isinstance(shards, dict) or set(shards) != {"fetched", "verified", "evicted", "total"}:
        raise StateError("canonical campaign status shard schema invalid")
    for key, amount in shards.items():
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise StateError(f"canonical campaign status shards.{key} invalid")
    if not (shards["evicted"] <= shards["verified"] <= shards["fetched"] <= shards["total"]):
        raise StateError("canonical campaign status shard counts inconsistent")
    for key in ("network_bytes",):
        amount = status.get(key)
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise StateError(f"canonical campaign status {key} invalid")
    throughput = status.get("throughput_bytes_per_second")
    if isinstance(throughput, bool) or not isinstance(throughput, (int, float)) \
            or not math.isfinite(float(throughput)) or float(throughput) < 0:
        raise StateError("canonical campaign status throughput invalid")
    eta = status.get("eta_seconds")
    if eta is not None and (isinstance(eta, bool) or not isinstance(eta, int) or eta < 0):
        raise StateError("canonical campaign status eta invalid")
    current = status.get("current")
    if not isinstance(current, dict) or set(current) != {"window", "layer"}:
        raise StateError("canonical campaign status current schema invalid")
    if current["window"] is not None:
        _require_name(current["window"], "canonical campaign status current.window")
    if current["layer"] is not None and (
        isinstance(current["layer"], bool) or not isinstance(current["layer"], int)
        or current["layer"] < 0
    ):
        raise StateError("canonical campaign status current.layer invalid")
    if not isinstance(status.get("candidate_rates"), list) \
            or len(status["candidate_rates"]) > 20 \
            or any(not isinstance(item, str) or _RATE_RE.fullmatch(item) is None
                   for item in status["candidate_rates"]):
        raise StateError("canonical campaign status candidate rates invalid")
    metrics = status.get("best_metrics")
    if not isinstance(metrics, dict) or len(metrics) > 20:
        raise StateError("canonical campaign status best metrics invalid")
    for name, metric in metrics.items():
        if not isinstance(name, str) or _SAFE_STATUS_NAME_RE.fullmatch(name) is None:
            raise StateError("canonical campaign status metric name invalid")
        if metric is not None and not isinstance(metric, (bool, int, float)):
            raise StateError("canonical campaign status metric value invalid")
        if isinstance(metric, float) and not math.isfinite(metric):
            raise StateError("canonical campaign status metric value is non-finite")
    resources = status.get("resources")
    if not isinstance(resources, dict) or set(resources) != {
        "disk_free_bytes", "ram_available_bytes", "swap_used_bytes",
    }:
        raise StateError("canonical campaign status resources schema invalid")
    for key, amount in resources.items():
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise StateError(f"canonical campaign status resources.{key} invalid")
    process = status.get("process")
    if not isinstance(process, dict) or set(process) != {"pid", "lease_held", "lease_owner"} \
            or isinstance(process["pid"], bool) or not isinstance(process["pid"], int) \
            or process["pid"] <= 0 or process["lease_held"] is not True:
        raise StateError("canonical campaign status process/lease invalid")
    lease_owner = _require_name(
        process.get("lease_owner"), "canonical campaign status lease owner"
    )
    if _SAFE_STATUS_NAME_RE.fullmatch(lease_owner) is None:
        raise StateError("canonical campaign status lease owner is unsafe")
    return status


def render_campaign_status_message(
    event_kind: str,
    dedupe_key: str,
    status: Mapping[str, Any],
    controller_anchor: Mapping[str, Any],
    *,
    claim_id: str,
    from_state: str | None,
    to_state: str,
) -> str:
    """Render the sole canonical state-transition notification text."""
    _require_name(event_kind, "Telegram event kind")
    _require_claim(claim_id)
    if to_state not in STATES or event_kind != TRANSITION_EVENT_KINDS[to_state]:
        raise StateError("Telegram event kind is not canonical for the target state")
    if not _is_sha256(dedupe_key):
        raise StateError("Telegram dedupe_key must be a sha256")
    value = _validate_campaign_status(status)
    anchor = _validate_controller_anchor(controller_anchor)
    if value["state"] != to_state or anchor["from_state"] != from_state:
        raise StateError("Telegram campaign status/controller anchor state mismatch")
    shards = value["shards"]
    current = value["current"]
    resources = value["resources"]
    process = value["process"]
    rates = ",".join(value["candidate_rates"]) or "none"
    metrics = json.dumps(value["best_metrics"], sort_keys=True, separators=(",", ":"))
    rendered = "\n".join([
        "GLM-5.2 Gravity",
        f"event={event_kind}",
        f"claim_id={claim_id}",
        f"dedupe_key={dedupe_key}",
        f"controller_anchor_sha256={anchor['anchor_sha256']}",
        f"from_state={from_state if from_state is not None else 'GENESIS'} to_state={to_state}",
        f"state={value['state']}",
        f"source_coverage_percent={float(value['source_coverage_percent']):.6f}",
        (
            f"shards fetched={shards['fetched']} verified={shards['verified']} "
            f"evicted={shards['evicted']} total={shards['total']}"
        ),
        (
            f"network_bytes={value['network_bytes']} "
            f"throughput_bytes_per_second={float(value['throughput_bytes_per_second']):.6f} "
            f"eta_seconds={value['eta_seconds'] if value['eta_seconds'] is not None else 'UNAVAILABLE'}"
        ),
        (
            f"current_window={current['window'] if current['window'] is not None else 'UNAVAILABLE'} "
            f"current_layer={current['layer'] if current['layer'] is not None else 'UNAVAILABLE'}"
        ),
        f"candidate_rates={rates}",
        f"best_metrics={metrics}",
        (
            f"disk_free_bytes={resources['disk_free_bytes']} "
            f"ram_available_bytes={resources['ram_available_bytes']} "
            f"swap_used_bytes={resources['swap_used_bytes']}"
        ),
        (
            f"pid={process['pid']} lease_held={str(process['lease_held']).lower()} "
            f"lease_owner={process['lease_owner']}"
        ),
    ])
    if len(rendered) > MAX_TELEGRAM_MESSAGE_CHARS:
        raise StateError("canonical campaign status exceeds Telegram message limit")
    return rendered


def _validate_transition_intent_shape(
    value: Any, auth: TelegramAuthConfig | None = None
) -> dict[str, Any]:
    intent = _require_object(value, "prepared transition intent")
    required = {
        "schema", "campaign_id", "source_revision", "controller_epoch",
        "expected_contract_sha256", "event_kind", "from_state", "to_state", "claim_id",
        "requested_payload", "state_payload", "request_sha256", "dedupe_key",
        "controller_anchor", "canonical_status", "canonical_status_sha256",
        "rendered_message", "rendered_message_sha256", "prepared_at", "seal_sha256",
        "controller_hmac_sha256",
    }
    if set(intent) != required or intent.get("schema") != TRANSITION_INTENT_SCHEMA:
        raise StateError("prepared transition intent schema/fields invalid")
    _validate_identity(
        intent.get("campaign_id"), intent.get("source_revision"), intent.get("controller_epoch")
    )
    if not _is_sha256(intent.get("expected_contract_sha256")):
        raise StateError("prepared transition expected contract hash invalid")
    to_state = intent.get("to_state")
    from_state = intent.get("from_state")
    if to_state not in STATES or (from_state is not None and from_state not in STATES):
        raise StateError("prepared transition state invalid")
    if intent.get("event_kind") != TRANSITION_EVENT_KINDS[to_state]:
        raise StateError("prepared transition event kind is not canonical")
    _require_claim(intent.get("claim_id"))
    _require_object(intent.get("requested_payload"), "prepared requested_payload")
    _require_object(intent.get("state_payload"), "prepared state_payload")
    for key in (
        "request_sha256", "dedupe_key", "canonical_status_sha256",
        "rendered_message_sha256", "seal_sha256",
    ):
        if not _is_sha256(intent.get(key)):
            raise StateError(f"prepared transition {key} invalid")
    anchor = _validate_controller_anchor(intent.get("controller_anchor"))
    if anchor["from_state"] != from_state:
        raise StateError("prepared transition/controller anchor state mismatch")
    status = _validate_campaign_status(intent.get("canonical_status"))
    if status["state"] != to_state \
            or intent["canonical_status_sha256"] != _campaign_status_hash(status):
        raise StateError("prepared transition canonical status mismatch")
    rendered = intent.get("rendered_message")
    if not isinstance(rendered, str) or not rendered or intent["rendered_message_sha256"] != \
            hashlib.sha256(rendered.encode("utf-8")).hexdigest():
        raise StateError("prepared transition rendered message hash invalid")
    expected_rendered = render_campaign_status_message(
        intent["event_kind"],
        intent["dedupe_key"],
        status,
        anchor,
        claim_id=intent["claim_id"],
        from_state=from_state,
        to_state=to_state,
    )
    if rendered != expected_rendered:
        raise StateError("prepared transition message is not the canonical rendering")
    _require_name(intent.get("prepared_at"), "prepared transition timestamp")
    sealed_intent = {
        key: item for key, item in intent.items() if key != "controller_hmac_sha256"
    }
    try:
        verify_sealed(sealed_intent, label="prepared transition intent")
    except Glm52Error as exc:
        raise StateError(str(exc)) from exc
    if not _is_sha256(intent.get("controller_hmac_sha256")):
        raise StateError("prepared transition controller HMAC invalid")
    if auth is not None:
        auth_body = {
            "schema": "hawking.glm52.state_transition_intent_auth.v1",
            "intent": sealed_intent,
        }
        if not auth.verify(auth_body, intent["controller_hmac_sha256"]):
            raise StateError("prepared transition controller HMAC authentication failed")
    return intent


def validate_transition_intent(
    value: Mapping[str, Any], auth: TelegramAuthConfig | None = None
) -> dict[str, Any]:
    """Public, side-effect-free validator for Telegram/outbox integrations."""
    return _validate_transition_intent_shape(value, auth)


def make_telegram_delivery_receipt(
    transition_intent: Mapping[str, Any],
    *,
    auth: TelegramAuthConfig,
    bot_api_response: Mapping[str, Any],
    http_status: int,
    delivered_at: str | None = None,
) -> dict[str, Any]:
    """Authenticate a Bot API response against one complete prepared transition."""
    intent = _validate_transition_intent_shape(transition_intent, auth)
    if not isinstance(auth, TelegramAuthConfig):
        raise StateError("TelegramAuthConfig is required")
    if isinstance(http_status, bool) or http_status != 200:
        raise StateError("Telegram Bot API HTTP status must be 200")
    response = _require_object(dict(bot_api_response), "Telegram Bot API response")
    if response.get("ok") is not True or not isinstance(response.get("result"), dict):
        raise StateError("Telegram Bot API response is not a validated success")
    result = response["result"]
    message_id = result.get("message_id")
    if isinstance(message_id, bool) or not isinstance(message_id, int) or message_id <= 0:
        raise StateError("Telegram Bot API result.message_id must be a positive integer")
    if result.get("text") != intent["rendered_message"]:
        raise StateError("Telegram Bot API response did not echo the exact prepared message")
    chat = result.get("chat")
    if not isinstance(chat, dict) or "id" not in chat:
        raise StateError("Telegram Bot API result.chat.id is missing")
    chat_digest = telegram_chat_identity_digest(chat["id"])
    if chat_digest != auth.expected_chat_identity_digest:
        raise StateError("Telegram Bot API response chat identity does not match configured chat")
    body = {
        "schema": TELEGRAM_RECEIPT_SCHEMA,
        "status": "DELIVERED",
        "algorithm": "HMAC-SHA256",
        "event_kind": intent["event_kind"],
        "claim_id": intent["claim_id"],
        "from_state": intent["from_state"],
        "to_state": intent["to_state"],
        "dedupe_key": intent["dedupe_key"],
        "canonical_status": intent["canonical_status"],
        "canonical_status_sha256": intent["canonical_status_sha256"],
        "rendered_message": intent["rendered_message"],
        "rendered_message_sha256": intent["rendered_message_sha256"],
        "controller_anchor": intent["controller_anchor"],
        "controller_anchor_sha256": intent["controller_anchor"]["anchor_sha256"],
        "transition_intent": intent,
        "transition_intent_sha256": intent["seal_sha256"],
        "message_id": message_id,
        "delivered_at": delivered_at or utc_now(),
        "bot_api_http_status": http_status,
        "bot_api_response_sha256": _sha256(response),
        "chat_identity_digest": chat_digest,
        "response_validated": True,
    }
    return {**body, "hmac_sha256": auth.authenticate(body)}


def _validate_delivery_receipt(
    receipt: Any, transition_intent: Mapping[str, Any], auth: TelegramAuthConfig
) -> dict[str, Any]:
    intent = _validate_transition_intent_shape(transition_intent, auth)
    value = _require_object(receipt, "telegram_delivery_receipt")
    bot_required = {
        "schema", "status", "algorithm", "event_kind", "claim_id", "from_state",
        "to_state", "dedupe_key", "canonical_status", "canonical_status_sha256",
        "rendered_message", "rendered_message_sha256", "controller_anchor",
        "controller_anchor_sha256", "transition_intent", "transition_intent_sha256", "message_id",
        "delivered_at", "bot_api_http_status", "bot_api_response_sha256",
        "chat_identity_digest", "response_validated", "hmac_sha256",
    }
    if value.get("schema") != TELEGRAM_RECEIPT_SCHEMA or set(value) != bot_required:
        raise StateError("Telegram delivery receipt schema/fields invalid")
    if value.get("algorithm") != "HMAC-SHA256" or value.get("status") != "DELIVERED":
        raise StateError("state transition refused: Telegram delivery is not successful")
    bindings = {
        "event_kind": intent["event_kind"],
        "claim_id": intent["claim_id"],
        "from_state": intent["from_state"],
        "to_state": intent["to_state"],
        "dedupe_key": intent["dedupe_key"],
        "canonical_status": intent["canonical_status"],
        "canonical_status_sha256": intent["canonical_status_sha256"],
        "rendered_message": intent["rendered_message"],
        "rendered_message_sha256": intent["rendered_message_sha256"],
        "controller_anchor": intent["controller_anchor"],
        "controller_anchor_sha256": intent["controller_anchor"]["anchor_sha256"],
        "transition_intent": intent,
        "transition_intent_sha256": intent["seal_sha256"],
    }
    if any(value.get(key) != expected for key, expected in bindings.items()):
        raise StateError("Telegram delivery receipt does not bind the prepared transition")
    if value["canonical_status_sha256"] != _campaign_status_hash(value["canonical_status"]):
        raise StateError("Telegram delivery receipt campaign-status hash mismatch")
    if value["rendered_message_sha256"] != hashlib.sha256(
        value["rendered_message"].encode("utf-8")
    ).hexdigest():
        raise StateError("Telegram delivery receipt rendered-message hash mismatch")
    _validate_controller_anchor(value["controller_anchor"])
    if value.get("bot_api_http_status") != 200 or value.get("response_validated") is not True \
            or not _is_sha256(value.get("bot_api_response_sha256")):
        raise StateError("Telegram delivery receipt lacks validated successful Bot API response")
    if value.get("chat_identity_digest") != auth.expected_chat_identity_digest:
        raise StateError("Telegram delivery receipt chat identity mismatch")
    message_id = value.get("message_id")
    if isinstance(message_id, bool) or not isinstance(message_id, int) or message_id <= 0:
        raise StateError("Telegram delivery receipt message_id invalid")
    _require_name(value.get("delivered_at"), "Telegram delivery proof time")
    body = {key: item for key, item in value.items() if key != "hmac_sha256"}
    if not auth.verify(body, value.get("hmac_sha256")):
        raise StateError("Telegram delivery receipt HMAC authentication failed")
    return value


def validate_telegram_delivery_receipt(
    receipt: Mapping[str, Any],
    transition_intent: Mapping[str, Any],
    auth: TelegramAuthConfig,
) -> dict[str, Any]:
    """Public exact-binding validator used by durable delivery adapters."""
    return _validate_delivery_receipt(receipt, transition_intent, auth)


def _validate_disk_sample(value: Any, label: str, *, permit_none: bool = False) -> None:
    if value is None and permit_none:
        return
    if not isinstance(value, dict) or not value:
        raise StateError(f"{label} must be a non-empty object")
    for key, amount in value.items():
        _require_name(key, f"{label} key")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise StateError(f"{label}.{key} must be a non-negative integer byte count")


def _validate_artifact_refs(value: Any, label: str) -> None:
    if not isinstance(value, list):
        raise StateError(f"{label} must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise StateError(f"{label}[{index}] must be an object")
        _require_name(item.get("path"), f"{label}[{index}].path")
        if not _is_sha256(item.get("sha256")):
            raise StateError(f"{label}[{index}].sha256 must be a sha256")


def _coverage_move(previous: str, current: str, next_map: Mapping[str, str], *, label: str) -> None:
    if current == previous:
        return
    if next_map.get(previous) != current:
        raise StateError(f"illegal backward/skipped {label} transition: {previous} -> {current}")


def _tensor_coverage_move(previous: str, current: str, *, label: str) -> None:
    if current == previous:
        return
    if previous == "FORWARD_VERIFIED" and current in TENSOR_TERMINAL_STATES:
        return
    if _TENSOR_NEXT.get(previous) == current:
        return
    raise StateError(f"illegal backward/skipped {label} transition: {previous} -> {current}")


class WindowLedger:
    """Hash-chained window records with monotonic coverage and exact tensor ownership."""

    _RECORD_KEYS = frozenset(
        {
            "schema",
            "campaign_id",
            "source_revision",
            "expected_contract_sha256",
            "window_id",
            "schedule_index",
            "phase",
            "source_shards",
            "carry_in_shards",
            "new_fetch_shards",
            "refetch_shards",
            "carry_out_shards",
            "evict_shards",
            "tensor_set",
            "layer_organ_dependencies",
            "download_start",
            "download_end",
            "bytes_transferred",
            "transfer_accounting",
            "hash_verification",
            "teacher_evidence_produced",
            "candidate_payloads_produced",
            "metrics",
            "compact_shard_hashes",
            "source_eviction",
            "disk_before",
            "disk_after",
            "retry_count",
            "source_coverage",
            "tensor_coverage",
            "terminal_coverage_evidence",
            "updated_at",
            "seal_sha256",
        }
    )
    _MUTABLE_PATCH_KEYS = frozenset(
        {
            "download_start",
            "download_end",
            "bytes_transferred",
            "transfer_accounting",
            "hash_verification",
            "teacher_evidence_produced",
            "candidate_payloads_produced",
            "metrics",
            "compact_shard_hashes",
            "source_eviction",
            "disk_after",
            "terminal_coverage_evidence",
        }
    )

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        campaign_id: str,
        source_revision: str,
        expected_contract: Mapping[str, Any],
        lease_guard: Callable[[], None] | None = None,
    ):
        self.path = Path(path)
        self.campaign_id = campaign_id
        self.source_revision = source_revision
        self.expected_contract = _validate_expected_contract(expected_contract)
        if self.expected_contract["campaign_id"] != campaign_id \
                or self.expected_contract["source_revision"] != source_revision:
            raise StateError("window ledger expected contract identity mismatch")
        self.expected_contract_sha256 = self.expected_contract["seal_sha256"]
        self._expected_shards = {
            item["path"]: item for item in self.expected_contract["source"]["shards"]
        }
        self._expected_tensors = set(self.expected_contract["tensors"]["names"])
        self._lease_guard = lease_guard
        self.log = HashChainLog(self.path, schema=WINDOW_EVENT_SCHEMA)

    def _require_lease(self) -> None:
        if self._lease_guard is not None:
            self._lease_guard()

    def _refuse_ungrounded_official_mutation(self, operation: str) -> None:
        """Keep caller-asserted phase data test-only until grounded APIs land.

        The legacy mutation surface validates sequencing and shape, but it does
        not itself open source/artifact files, measure network/disk state, or
        prove eviction.  It is useful for synthetic controller tests and must
        never create official campaign completeness evidence.
        """
        if self.expected_contract["source"]["profile"] == "OFFICIAL_GLM52_BF16":
            raise StateError(
                f"official {operation} refused: generic WindowLedger mutations are "
                "caller-asserted; filesystem-executor-v1 grounding is required"
            )

    @staticmethod
    def _source_paths(record: Mapping[str, Any]) -> list[str]:
        return [str(item["path"]) for item in record["source_shards"]]

    def _validate_record(self, record: dict[str, Any]) -> None:
        if set(record) != self._RECORD_KEYS:
            missing = sorted(self._RECORD_KEYS - set(record))
            extra = sorted(set(record) - self._RECORD_KEYS)
            raise StateError(f"window record fields invalid; missing={missing}, extra={extra}")
        try:
            verify_sealed(record, label=f"window {record.get('window_id')}")
        except Glm52Error as exc:
            raise StateError(str(exc)) from exc
        if record.get("schema") != WINDOW_RECORD_SCHEMA:
            raise StateError("window record schema mismatch")
        if record.get("campaign_id") != self.campaign_id:
            raise StateError("window record campaign identity mismatch")
        if record.get("source_revision") != self.source_revision:
            raise StateError("window record source revision mismatch")
        if record.get("expected_contract_sha256") != self.expected_contract_sha256:
            raise StateError("window record expected-contract identity mismatch")
        window_id = _require_name(record.get("window_id"), "window_id")
        schedule_index = record.get("schedule_index")
        if isinstance(schedule_index, bool) or not isinstance(schedule_index, int) \
                or not 0 <= schedule_index < len(self.expected_contract["window_schedule"]):
            raise StateError("window schedule_index invalid")
        scheduled = self.expected_contract["window_schedule"][schedule_index]
        if scheduled["window_id"] != window_id:
            raise StateError("window_id does not match authoritative schedule index")
        if record.get("phase") not in WINDOW_PHASES:
            raise StateError(f"unknown window phase: {record.get('phase')}")
        shards = record.get("source_shards")
        if not isinstance(shards, list) or not shards:
            raise StateError("source_shards must be a non-empty list")
        source_paths: list[str] = []
        for index, shard in enumerate(shards):
            identity = _validate_contract_shard(shard, label=f"source_shards[{index}]")
            source_paths.append(identity["path"])
            if self._expected_shards.get(identity["path"]) != identity:
                raise StateError(f"source shard identity differs from expected contract: {identity['path']}")
        if len(source_paths) != len(set(source_paths)):
            raise StateError("duplicate source shard in one window")
        if set(source_paths) != set(scheduled["source_shards"]):
            raise StateError("window source_shards differ from authoritative schedule")
        source_path_set = set(source_paths)
        for key in (
            "carry_in_shards", "new_fetch_shards", "refetch_shards", "carry_out_shards",
            "evict_shards",
        ):
            value = record.get(key)
            if not isinstance(value, list) or len(value) != len(set(value)) \
                    or any(not isinstance(path, str) or not path for path in value):
                raise StateError(f"{key} must be a unique shard-path list")
            if set(value) != set(scheduled[key]) or not set(value).issubset(source_path_set):
                raise StateError(f"{key} differs from authoritative schedule")
        tensors = record.get("tensor_set")
        if not isinstance(tensors, list):
            raise StateError("tensor_set must be a list")
        if any(not isinstance(name, str) or not name for name in tensors):
            raise StateError("every tensor_set entry must be a non-empty string")
        if len(tensors) != len(set(tensors)):
            raise StateError("duplicate tensor in one window")
        if set(tensors) != set(scheduled["tensor_set"]):
            raise StateError("window tensor_set differs from authoritative schedule")
        if not isinstance(record.get("layer_organ_dependencies"), list):
            raise StateError("layer_organ_dependencies must be a list")
        for time_key in ("download_start", "download_end"):
            value = record.get(time_key)
            if value is not None:
                _require_name(value, time_key)
        transferred = record.get("bytes_transferred")
        if isinstance(transferred, bool) or not isinstance(transferred, int) or transferred < 0:
            raise StateError("bytes_transferred must be a non-negative integer")
        transfer = record.get("transfer_accounting")
        if not isinstance(transfer, dict) or set(transfer) != {
            "new_fetch_network_bytes", "refetch_network_bytes", "protocol_overhead_bytes"
        }:
            raise StateError("transfer_accounting schema invalid")
        for key, amount in transfer.items():
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                raise StateError(f"transfer_accounting.{key} must be non-negative integer bytes")
        if sum(transfer.values()) != transferred:
            raise StateError("bytes_transferred must equal exact transfer_accounting components")
        if not record["new_fetch_shards"] and transfer["new_fetch_network_bytes"] != 0:
            raise StateError("new-fetch network bytes recorded without new-fetch shards")
        if not record["refetch_shards"] and transfer["refetch_network_bytes"] != 0:
            raise StateError("refetch network bytes recorded without refetch shards")
        retry = record.get("retry_count")
        if isinstance(retry, bool) or not isinstance(retry, int) or retry < 0:
            raise StateError("retry_count must be a non-negative integer")
        for object_key in (
            "hash_verification",
            "metrics",
            "compact_shard_hashes",
            "source_eviction",
            "terminal_coverage_evidence",
        ):
            if not isinstance(record.get(object_key), dict):
                raise StateError(f"{object_key} must be an object")
        _validate_artifact_refs(record.get("teacher_evidence_produced"), "teacher_evidence_produced")
        _validate_artifact_refs(record.get("candidate_payloads_produced"), "candidate_payloads_produced")
        _validate_disk_sample(record.get("disk_before"), "disk_before")
        _validate_disk_sample(record.get("disk_after"), "disk_after", permit_none=True)
        source_coverage = record.get("source_coverage")
        if not isinstance(source_coverage, dict) or set(source_coverage) != set(source_paths):
            raise StateError("source_coverage keys must exactly equal source shard paths")
        if any(value not in SOURCE_COVERAGE_STATES for value in source_coverage.values()):
            raise StateError("source_coverage contains an unknown state")
        tensor_coverage = record.get("tensor_coverage")
        if not isinstance(tensor_coverage, dict) or set(tensor_coverage) != set(tensors):
            raise StateError("tensor_coverage keys must exactly equal tensor_set")
        valid_tensor_states = set(TENSOR_PROGRESS_STATES) | set(TENSOR_TERMINAL_STATES)
        if any(value not in valid_tensor_states for value in tensor_coverage.values()):
            raise StateError("tensor_coverage contains an unknown state")
        for path, digest in record["compact_shard_hashes"].items():
            _require_name(path, "compact_shard_hashes path")
            if not _is_sha256(digest):
                raise StateError(f"compact shard hash is not sha256: {path}")
        _require_name(record.get("updated_at"), "updated_at")
        self._validate_phase_gates(record)

    def _validate_phase_gates(self, record: Mapping[str, Any]) -> None:
        phase_index = WINDOW_PHASES.index(str(record["phase"]))
        source = record["source_coverage"]
        tensors = record["tensor_coverage"]
        source_paths = self._source_paths(record)
        if phase_index >= WINDOW_PHASES.index("FETCHING") and not record.get("download_start"):
            raise StateError("FETCHING or later requires download_start")
        if phase_index >= WINDOW_PHASES.index("FETCHED"):
            if not record.get("download_end"):
                raise StateError("FETCHED or later requires download_end")
            if any(SOURCE_COVERAGE_STATES.index(value) < SOURCE_COVERAGE_STATES.index("FETCHED")
                   for value in source.values()):
                raise StateError("FETCHED or later requires every source shard fetched")
        if phase_index >= WINDOW_PHASES.index("VERIFIED"):
            verification = record["hash_verification"]
            if verification.get("status") != "VERIFIED":
                raise StateError("VERIFIED or later requires successful hash verification")
            if set(verification.get("verified_shards", [])) != set(source_paths):
                raise StateError("hash verification must cover every source shard exactly")
            if any(SOURCE_COVERAGE_STATES.index(value) < SOURCE_COVERAGE_STATES.index("HASH_VERIFIED")
                   for value in source.values()):
                raise StateError("VERIFIED or later requires every source shard hash-verified")
            if any(value == "DECLARED" for value in tensors.values()):
                raise StateError("VERIFIED or later requires source-verified tensors")
        if phase_index >= WINDOW_PHASES.index("TEACHER_CAPTURED"):
            if not record["teacher_evidence_produced"]:
                raise StateError("TEACHER_CAPTURED or later requires teacher evidence")
            allowed = set(TENSOR_PROGRESS_STATES[2:]) | set(TENSOR_TERMINAL_STATES)
            if any(value not in allowed for value in tensors.values()):
                raise StateError("TEACHER_CAPTURED or later requires teacher-evidenced tensors")
        if phase_index >= WINDOW_PHASES.index("CANDIDATES_PACKED"):
            if not record["candidate_payloads_produced"]:
                raise StateError("CANDIDATES_PACKED or later requires candidate payloads")
            allowed = set(TENSOR_PROGRESS_STATES[3:]) | set(TENSOR_TERMINAL_STATES)
            if any(value not in allowed for value in tensors.values()):
                raise StateError("CANDIDATES_PACKED or later requires packed tensor candidates")
        if phase_index >= WINDOW_PHASES.index("FORWARD_COMPLETE"):
            allowed = {"FORWARD_VERIFIED"} | set(TENSOR_TERMINAL_STATES)
            if any(value not in allowed for value in tensors.values()):
                raise StateError("FORWARD_COMPLETE or later requires forward-verified tensors")
            if not record["metrics"]:
                raise StateError("FORWARD_COMPLETE or later requires metrics")
        if phase_index >= WINDOW_PHASES.index("SEALED"):
            if any(value not in TENSOR_TERMINAL_STATES for value in tensors.values()):
                raise StateError("SEALED requires exactly one terminal state for every tensor")
            if any(value not in {"CONSUMED", "EVICTED"} for value in source.values()):
                raise StateError("SEALED requires every source shard consumed")
            evidence = record["terminal_coverage_evidence"]
            if set(evidence) != set(record["tensor_set"]):
                raise StateError("terminal coverage evidence must cover every tensor exactly")
            for tensor, disposition in tensors.items():
                item = evidence[tensor]
                if not isinstance(item, dict) or item.get("disposition") != disposition:
                    raise StateError(f"terminal evidence disposition mismatch for {tensor}")
                if not _is_sha256(item.get("evidence_sha256")):
                    raise StateError(f"terminal evidence hash missing for {tensor}")
                if disposition == "PROTECTED_SOURCE_NATIVE_WITH_BILLED_BYTES":
                    billed = item.get("billed_bytes")
                    if isinstance(billed, bool) or not isinstance(billed, int) or billed <= 0:
                        raise StateError(f"protected native tensor lacks billed_bytes: {tensor}")
                if disposition == "INTENTIONALLY_OMITTED_WITH_CAPABILITY_JUSTIFICATION":
                    _require_name(item.get("capability_justification"),
                                  f"capability justification for {tensor}")
            packed = {
                "PACKED_IN_CORE_ARTIFACT",
                "PACKED_IN_OPTIONAL_MTP_PACK",
            }
            if any(value in packed for value in tensors.values()) and not record["compact_shard_hashes"]:
                raise StateError("packed tensors require at least one compact shard hash")
        if phase_index >= WINDOW_PHASES.index("EVICTED"):
            carry_out = set(record["carry_out_shards"])
            evicted = set(record["evict_shards"])
            if any(source[path] != "CONSUMED" for path in carry_out):
                raise StateError("EVICTED window must retain carried shards as CONSUMED")
            if any(source[path] != "EVICTED" for path in evicted):
                raise StateError("EVICTED window must evict every non-carried shard")
            eviction = record["source_eviction"]
            if eviction.get("status") != "EVICTED":
                raise StateError("EVICTED requires a source eviction receipt")
            if set(eviction.get("evicted_shards", [])) != evicted:
                raise StateError("source eviction receipt must cover exactly non-carried shards")
            if not _is_sha256(eviction.get("receipt_sha256")):
                raise StateError("source eviction receipt_sha256 is required")
            _validate_disk_sample(record.get("disk_after"), "disk_after")

    @staticmethod
    def _record_hash(record: Mapping[str, Any]) -> str:
        return str(record["seal_sha256"])

    def _replay(self) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
        records: dict[str, dict[str, Any]] = {}
        tensor_owner: dict[str, str] = {}
        events = self.log.verified_events()
        for event in events:
            payload = event["payload"]
            record = payload.get("record")
            if not isinstance(record, dict):
                raise StateError(f"window event seq {event['seq']} lacks record snapshot")
            self._validate_record(record)
            window_id = record["window_id"]
            request_sha = payload.get("request_sha256")
            if not _is_sha256(request_sha):
                raise StateError(f"window event seq {event['seq']} request hash invalid")
            if event["kind"] == "WINDOW_DECLARED":
                if window_id in records:
                    raise StateError(f"window declared twice: {window_id}")
                if record["schedule_index"] != len(records):
                    raise StateError("windows must be declared in authoritative schedule order")
                previous_window: dict[str, Any] | None = None
                if record["schedule_index"] > 0:
                    previous_id = self.expected_contract["window_schedule"][
                        record["schedule_index"] - 1
                    ]["window_id"]
                    previous_window = records.get(previous_id)
                    if previous_window is None or previous_window["phase"] != "EVICTED":
                        raise StateError("next window declaration requires prior window EVICTED")
                    if set(record["carry_in_shards"]) != set(previous_window["carry_out_shards"]):
                        raise StateError("declared carry_in does not equal prior carry_out")
                expected_previous = (
                    self._record_hash(previous_window) if previous_window is not None else None
                )
                if payload.get("previous_record_sha256") != expected_previous:
                    raise StateError("window declaration previous-record anchor mismatch")
                if record["phase"] != "PLANNED" or record["retry_count"] != 0:
                    raise StateError("declared window must begin PLANNED with retry_count=0")
                carry_in = set(record["carry_in_shards"])
                for path, status in record["source_coverage"].items():
                    expected_status = "HASH_VERIFIED" if path in carry_in else "DECLARED"
                    if status != expected_status:
                        raise StateError(
                            "declared source coverage must start carry-in HASH_VERIFIED and fetches DECLARED"
                        )
                if any(value != "DECLARED" for value in record["tensor_coverage"].values()):
                    raise StateError("declared tensor coverage must begin DECLARED")
                for tensor in record["tensor_set"]:
                    if tensor in tensor_owner:
                        raise StateError(
                            f"tensor {tensor} owned by multiple windows: "
                            f"{tensor_owner[tensor]} and {window_id}"
                        )
                    tensor_owner[tensor] = window_id
                expected_request = _sha256(
                    {
                        "kind": "WINDOW_DECLARED",
                        "campaign_id": self.campaign_id,
                        "source_revision": self.source_revision,
                        "claim_id": event["claim_id"],
                        "record": {key: value for key, value in record.items() if key not in {
                            "updated_at", "seal_sha256"
                        }},
                    }
                )
            elif event["kind"] in {"WINDOW_ADVANCED", "WINDOW_RETRY"}:
                if window_id not in records:
                    raise StateError(f"window event before declaration: {window_id}")
                previous = records[window_id]
                if previous["schedule_index"] != max(
                    item["schedule_index"] for item in records.values()
                ):
                    raise StateError("a prior window cannot mutate after a later declaration")
                if payload.get("previous_record_sha256") != self._record_hash(previous):
                    raise StateError(f"window record chain break: {window_id}")
                for immutable in (
                    "campaign_id",
                    "source_revision",
                    "expected_contract_sha256",
                    "window_id",
                    "schedule_index",
                    "source_shards",
                    "carry_in_shards",
                    "new_fetch_shards",
                    "refetch_shards",
                    "carry_out_shards",
                    "evict_shards",
                    "tensor_set",
                    "layer_organ_dependencies",
                    "disk_before",
                ):
                    if record[immutable] != previous[immutable]:
                        raise StateError(f"immutable window field changed: {window_id}.{immutable}")
                for shard, old_state in previous["source_coverage"].items():
                    _coverage_move(
                        old_state,
                        record["source_coverage"][shard],
                        _SOURCE_NEXT,
                        label=f"source {window_id}:{shard}",
                    )
                for tensor, old_state in previous["tensor_coverage"].items():
                    _tensor_coverage_move(
                        old_state,
                        record["tensor_coverage"][tensor],
                        label=f"tensor {window_id}:{tensor}",
                    )
                if event["kind"] == "WINDOW_ADVANCED":
                    if _WINDOW_NEXT.get(previous["phase"]) != record["phase"]:
                        raise StateError(
                            f"illegal backward/skipped window transition: "
                            f"{previous['phase']} -> {record['phase']}"
                        )
                    if record["retry_count"] != previous["retry_count"]:
                        raise StateError("WINDOW_ADVANCED cannot alter retry_count")
                    if WINDOW_PHASES.index(previous["phase"]) >= WINDOW_PHASES.index("FETCHED") \
                            and (
                                record["bytes_transferred"] != previous["bytes_transferred"]
                                or record["transfer_accounting"] != previous["transfer_accounting"]
                            ):
                        raise StateError("transfer accounting is immutable after FETCHED")
                else:
                    if record["phase"] != previous["phase"]:
                        raise StateError("WINDOW_RETRY cannot alter phase")
                    if record["retry_count"] != previous["retry_count"] + 1:
                        raise StateError("WINDOW_RETRY must increment retry_count exactly once")
                    if record["source_coverage"] != previous["source_coverage"] \
                            or record["tensor_coverage"] != previous["tensor_coverage"]:
                        raise StateError("WINDOW_RETRY cannot alter coverage")
                request_body = payload.get("request_body")
                if not isinstance(request_body, dict):
                    raise StateError("window mutation lacks request_body")
                if request_body.get("kind") != event["kind"]:
                    raise StateError("window request kind mismatch")
                if request_body.get("claim_id") != event["claim_id"]:
                    raise StateError("window request claim mismatch")
                if request_body.get("campaign_id") != self.campaign_id \
                        or request_body.get("source_revision") != self.source_revision \
                        or request_body.get("window_id") != window_id:
                    raise StateError("window request identity fields mismatch")
                # Prove that the recorded snapshot is exactly the requested mutation of
                # its predecessor, rather than merely a legal but different snapshot.
                expected_body = {
                    key: copy.deepcopy(value)
                    for key, value in previous.items()
                    if key != "seal_sha256"
                }
                if event["kind"] == "WINDOW_ADVANCED":
                    patch = request_body.get("patch")
                    source_delta = request_body.get("source_coverage")
                    tensor_delta = request_body.get("tensor_coverage")
                    if not isinstance(patch, dict) or not isinstance(source_delta, dict) \
                            or not isinstance(tensor_delta, dict):
                        raise StateError("window advance request deltas are malformed")
                    if request_body.get("to_phase") != record["phase"]:
                        raise StateError("window advance request phase mismatch")
                    expected_body.update(copy.deepcopy(patch))
                    expected_body["phase"] = request_body["to_phase"]
                    expected_body["source_coverage"].update(copy.deepcopy(source_delta))
                    expected_body["tensor_coverage"].update(copy.deepcopy(tensor_delta))
                else:
                    retry_metrics = request_body.get("metrics")
                    if not isinstance(retry_metrics, dict):
                        raise StateError("window retry metrics are malformed")
                    expected_body["retry_count"] += 1
                    expected_body["metrics"] = {
                        **expected_body["metrics"],
                        **copy.deepcopy(retry_metrics),
                    }
                expected_body["updated_at"] = record["updated_at"]
                if seal(expected_body) != record:
                    raise StateError("window record differs from its claimed mutation")
                expected_request = _sha256(request_body)
            else:
                raise StateError(f"unknown window event kind: {event['kind']}")
            if request_sha != expected_request:
                raise StateError(f"window request identity mismatch at seq {event['seq']}")
            records[window_id] = record
        return records, tensor_owner, events

    def records(self) -> dict[str, dict[str, Any]]:
        records, _, _ = self._replay()
        return copy.deepcopy(records)

    def record(self, window_id: str) -> dict[str, Any]:
        _require_name(window_id, "window_id")
        records = self.records()
        if window_id not in records:
            raise StateError(f"unknown window: {window_id}")
        return records[window_id]

    def _existing_claim(self, claim_id: str, expected_request: str) -> dict[str, Any] | None:
        event = self.log.find_claim(claim_id)
        if event is None:
            return None
        if event["payload"].get("request_sha256") != expected_request:
            raise StateError(f"claim reused for a different window operation: {claim_id}")
        return copy.deepcopy(event["payload"]["record"])

    def declare_window(
        self,
        *,
        window_id: str,
        schedule_index: int,
        source_shards: Sequence[Mapping[str, Any]],
        carry_in_shards: Sequence[str],
        new_fetch_shards: Sequence[str],
        refetch_shards: Sequence[str],
        carry_out_shards: Sequence[str],
        evict_shards: Sequence[str],
        tensor_set: Sequence[str],
        layer_organ_dependencies: Sequence[Any],
        disk_before: Mapping[str, int],
        claim_id: str,
    ) -> dict[str, Any]:
        self._require_lease()
        self._refuse_ungrounded_official_mutation("window declaration")
        _require_claim(claim_id)
        _require_name(window_id, "window_id")
        shards = _clone(list(source_shards))
        tensors = _clone(list(tensor_set))
        dependencies = _clone(list(layer_organ_dependencies))
        before = _clone(dict(disk_before))
        carry_in = _clone(list(carry_in_shards))
        new_fetch = _clone(list(new_fetch_shards))
        refetch = _clone(list(refetch_shards))
        carry_out = _clone(list(carry_out_shards))
        evict = _clone(list(evict_shards))
        if not shards or any(not isinstance(item, dict) or "path" not in item for item in shards):
            raise StateError("source_shards must contain non-empty shard identity objects")
        if any(not isinstance(item, str) or not item for item in tensors):
            raise StateError("tensor_set must contain only non-empty tensor names")
        base = {
            "schema": WINDOW_RECORD_SCHEMA,
            "campaign_id": self.campaign_id,
            "source_revision": self.source_revision,
            "expected_contract_sha256": self.expected_contract_sha256,
            "window_id": window_id,
            "schedule_index": schedule_index,
            "phase": "PLANNED",
            "source_shards": shards,
            "carry_in_shards": carry_in,
            "new_fetch_shards": new_fetch,
            "refetch_shards": refetch,
            "carry_out_shards": carry_out,
            "evict_shards": evict,
            "tensor_set": tensors,
            "layer_organ_dependencies": dependencies,
            "download_start": None,
            "download_end": None,
            "bytes_transferred": 0,
            "transfer_accounting": {
                "new_fetch_network_bytes": 0,
                "refetch_network_bytes": 0,
                "protocol_overhead_bytes": 0,
            },
            "hash_verification": {"status": "PENDING", "verified_shards": []},
            "teacher_evidence_produced": [],
            "candidate_payloads_produced": [],
            "metrics": {},
            "compact_shard_hashes": {},
            "source_eviction": {"status": "PENDING", "evicted_shards": [],
                                "receipt_sha256": None},
            "disk_before": before,
            "disk_after": None,
            "retry_count": 0,
            "source_coverage": {
                item["path"]: ("HASH_VERIFIED" if item["path"] in set(carry_in) else "DECLARED")
                for item in shards
            },
            "tensor_coverage": {name: "DECLARED" for name in tensors},
            "terminal_coverage_evidence": {},
        }
        request_base = {
            "kind": "WINDOW_DECLARED",
            "campaign_id": self.campaign_id,
            "source_revision": self.source_revision,
            "claim_id": claim_id,
            "record": base,
        }
        request_sha = _sha256(request_base)
        existing = self._existing_claim(claim_id, request_sha)
        if existing is not None:
            return existing
        records, tensor_owner, _ = self._replay()
        if window_id in records:
            raise StateError(f"window already declared: {window_id}")
        if schedule_index != len(records):
            raise StateError("window declaration is out of authoritative schedule order")
        previous_record = None
        if schedule_index > 0:
            previous_id = self.expected_contract["window_schedule"][schedule_index - 1]["window_id"]
            previous_record = records.get(previous_id)
            if previous_record is None or previous_record["phase"] != "EVICTED":
                raise StateError("next window cannot be declared before prior window eviction")
            if set(carry_in) != set(previous_record["carry_out_shards"]):
                raise StateError("carry_in must exactly equal prior window carry_out")
        overlap = sorted(set(tensors) & set(tensor_owner))
        if overlap:
            raise StateError(f"tensor ownership overlap with existing window: {overlap[0]}")
        record = seal({**base, "updated_at": utc_now()})
        self._validate_record(record)
        self.log.append(
            "WINDOW_DECLARED",
            claim_id,
            {
                "previous_record_sha256": (
                    self._record_hash(previous_record) if previous_record is not None else None
                ),
                "request_sha256": request_sha,
                "record": record,
            },
        )
        self._replay()
        return copy.deepcopy(record)

    def advance(
        self,
        window_id: str,
        to_phase: str,
        *,
        claim_id: str,
        patch: Mapping[str, Any] | None = None,
        source_coverage: Mapping[str, str] | None = None,
        tensor_coverage: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        self._require_lease()
        self._refuse_ungrounded_official_mutation("window advance")
        _require_claim(claim_id)
        _require_name(window_id, "window_id")
        if to_phase not in WINDOW_PHASES:
            raise StateError(f"unknown window phase: {to_phase}")
        patch_copy = _require_object(dict(patch or {}), "window patch")
        forbidden = set(patch_copy) - self._MUTABLE_PATCH_KEYS
        if forbidden:
            raise StateError(f"window patch attempts immutable/unknown fields: {sorted(forbidden)}")
        source_delta = _require_object(dict(source_coverage or {}), "source coverage delta")
        tensor_delta = _require_object(dict(tensor_coverage or {}), "tensor coverage delta")
        request_body = {
            "kind": "WINDOW_ADVANCED",
            "campaign_id": self.campaign_id,
            "source_revision": self.source_revision,
            "window_id": window_id,
            "to_phase": to_phase,
            "claim_id": claim_id,
            "patch": patch_copy,
            "source_coverage": source_delta,
            "tensor_coverage": tensor_delta,
        }
        request_sha = _sha256(request_body)
        existing = self._existing_claim(claim_id, request_sha)
        if existing is not None:
            return existing
        current = self.record(window_id)
        all_records = self.records()
        if current["schedule_index"] != max(
            record["schedule_index"] for record in all_records.values()
        ):
            raise StateError("a prior window cannot advance after a later declaration")
        if _WINDOW_NEXT.get(current["phase"]) != to_phase:
            raise StateError(f"illegal backward/skipped window transition: {current['phase']} -> {to_phase}")
        if not set(source_delta).issubset(current["source_coverage"]):
            raise StateError("source coverage delta contains an undeclared shard")
        if not set(tensor_delta).issubset(current["tensor_coverage"]):
            raise StateError("tensor coverage delta contains an undeclared tensor")
        body = {key: copy.deepcopy(value) for key, value in current.items() if key != "seal_sha256"}
        body.update(patch_copy)
        body["phase"] = to_phase
        body["source_coverage"].update(source_delta)
        body["tensor_coverage"].update(tensor_delta)
        body["updated_at"] = utc_now()
        record = seal(body)
        self._validate_record(record)
        # Compare coverage before append as well as during full replay.
        for shard, old in current["source_coverage"].items():
            _coverage_move(old, record["source_coverage"][shard], _SOURCE_NEXT,
                           label=f"source {window_id}:{shard}")
        for tensor, old in current["tensor_coverage"].items():
            _tensor_coverage_move(old, record["tensor_coverage"][tensor],
                                  label=f"tensor {window_id}:{tensor}")
        self.log.append(
            "WINDOW_ADVANCED",
            claim_id,
            {
                "previous_record_sha256": self._record_hash(current),
                "request_body": request_body,
                "request_sha256": request_sha,
                "record": record,
            },
        )
        self._replay()
        return copy.deepcopy(record)

    def record_retry(
        self,
        window_id: str,
        *,
        claim_id: str,
        metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_lease()
        self._refuse_ungrounded_official_mutation("window retry")
        _require_claim(claim_id)
        metric_copy = _require_object(dict(metrics or {}), "retry metrics")
        request_body = {
            "kind": "WINDOW_RETRY",
            "campaign_id": self.campaign_id,
            "source_revision": self.source_revision,
            "window_id": window_id,
            "claim_id": claim_id,
            "metrics": metric_copy,
        }
        request_sha = _sha256(request_body)
        existing = self._existing_claim(claim_id, request_sha)
        if existing is not None:
            return existing
        current = self.record(window_id)
        all_records = self.records()
        if current["schedule_index"] != max(
            record["schedule_index"] for record in all_records.values()
        ):
            raise StateError("a prior window cannot retry after a later declaration")
        if current["phase"] == "EVICTED":
            raise StateError("an EVICTED window is immutable")
        body = {key: copy.deepcopy(value) for key, value in current.items() if key != "seal_sha256"}
        body["retry_count"] += 1
        body["metrics"] = {**body["metrics"], **metric_copy}
        body["updated_at"] = utc_now()
        record = seal(body)
        self._validate_record(record)
        self.log.append(
            "WINDOW_RETRY",
            claim_id,
            {
                "previous_record_sha256": self._record_hash(current),
                "request_body": request_body,
                "request_sha256": request_sha,
                "record": record,
            },
        )
        self._replay()
        return copy.deepcopy(record)

    def summary(self) -> dict[str, Any]:
        records, tensor_owner, events = self._replay()
        phase_counts = Counter(record["phase"] for record in records.values())
        terminal_counts = Counter(
            status
            for record in records.values()
            for status in record["tensor_coverage"].values()
            if status in TENSOR_TERMINAL_STATES
        )
        ordered = sorted(records.values(), key=lambda record: record["schedule_index"])
        new_fetch_counts = Counter(
            path for record in ordered for path in record["new_fetch_shards"]
        )
        refetch_events = [
            (record["window_id"], path)
            for record in ordered
            for path in record["refetch_shards"]
        ]
        total_network_bytes = sum(record["bytes_transferred"] for record in ordered)
        new_fetch_network_bytes = sum(
            record["transfer_accounting"]["new_fetch_network_bytes"] for record in ordered
        )
        refetch_network_bytes = sum(
            record["transfer_accounting"]["refetch_network_bytes"] for record in ordered
        )
        protocol_overhead_bytes = sum(
            record["transfer_accounting"]["protocol_overhead_bytes"] for record in ordered
        )
        expected_bytes = self.expected_contract["source"]["expected_logical_bytes"]
        new_fetch_logical_bytes = sum(
            self._expected_shards[path]["logical_bytes"] for path in new_fetch_counts
        )
        refetch_logical_bytes = sum(
            self._expected_shards[path]["logical_bytes"] for _, path in refetch_events
        )
        return {
            "window_count": len(records),
            "window_event_count": len(events),
            "phase_counts": dict(sorted(phase_counts.items())),
            "owned_tensor_count": len(tensor_owner),
            "terminal_tensor_count": sum(terminal_counts.values()),
            "terminal_counts": dict(sorted(terminal_counts.items())),
            "source_accounting": {
                "expected_shard_count": self.expected_contract["source"]["expected_shard_count"],
                "expected_logical_bytes": expected_bytes,
                "unique_new_fetch_shard_count": len(new_fetch_counts),
                "new_fetch_logical_bytes": new_fetch_logical_bytes,
                "each_new_fetch_once_so_far": all(count == 1 for count in new_fetch_counts.values()),
                "refetch_event_count": len(refetch_events),
                "refetch_logical_bytes": refetch_logical_bytes,
                "new_fetch_network_bytes": new_fetch_network_bytes,
                "refetch_network_bytes": refetch_network_bytes,
                "protocol_overhead_bytes": protocol_overhead_bytes,
                "total_network_bytes": total_network_bytes,
                "transfer_amplification": {
                    "numerator_network_bytes": total_network_bytes,
                    "denominator_expected_logical_bytes": expected_bytes,
                    "ratio": total_network_bytes / expected_bytes,
                },
            },
        }

    def resume_plan(self) -> dict[str, Any]:
        """Return the earliest incomplete declared dependency without mutating anything."""
        records, _, events = self._replay()
        declaration_order = [
            event["payload"]["record"]["window_id"]
            for event in events
            if event["kind"] == "WINDOW_DECLARED"
        ]
        incomplete = next(
            (window_id for window_id in declaration_order if records[window_id]["phase"] != "EVICTED"),
            None,
        )
        sealed = [
            window_id
            for window_id in declaration_order
            if records[window_id]["phase"] in {"SEALED", "EVICTED"}
        ]
        if incomplete is None:
            next_schedule = (
                self.expected_contract["window_schedule"][len(declaration_order)]
                if len(declaration_order) < len(self.expected_contract["window_schedule"])
                else None
            )
            return {
                "campaign_windows_complete": (
                    len(declaration_order) == len(self.expected_contract["window_schedule"])
                    and bool(declaration_order)
                ),
                "earliest_incomplete_window_id": (
                    next_schedule["window_id"] if next_schedule is not None else None
                ),
                "resume_from_phase": "UNDECLARED" if next_schedule is not None else None,
                "next_controller_state": (
                    "FREEZE_PROGRAM" if not declaration_order and next_schedule is not None
                    else "EVICT_WINDOW" if next_schedule is not None else None
                ),
                "next_action": "DECLARE_WINDOW" if next_schedule is not None else None,
                "last_sealed_window_id": sealed[-1] if sealed else None,
                "reuse_verified_source_shards": [],
                "reuse_teacher_evidence": [],
                "partial_output_policy":
                    "DELETE_ONLY_UNSEALED_EXACT_PATHS_AFTER_HASH_AND_OWNERSHIP_VALIDATION",
            }
        record = records[incomplete]
        next_state = {
            "PLANNED": "FETCH_WINDOW",
            "FETCHING": "FETCH_WINDOW",
            "FETCHED": "VERIFY_WINDOW",
            "VERIFIED": "CAPTURE_TEACHER",
            "TEACHER_CAPTURED": "FIT_CANDIDATES",
            "CANDIDATES_FIT": "PACK_CANDIDATES",
            "CANDIDATES_PACKED": "RUN_WINDOW_FORWARD",
            "FORWARD_COMPLETE": "SEAL_WINDOW",
            "SEALED": "EVICT_WINDOW",
        }[record["phase"]]
        reusable_source = sorted(
            path
            for path, status in record["source_coverage"].items()
            if status in {"HASH_VERIFIED", "CONSUMED"}
        )
        return {
            "campaign_windows_complete": False,
            "earliest_incomplete_window_id": incomplete,
            "resume_from_phase": record["phase"],
            "next_controller_state": next_state,
            "next_action": "RESUME_WINDOW_PHASE",
            "last_sealed_window_id": sealed[-1] if sealed else None,
            "reuse_verified_source_shards": reusable_source,
            "reuse_teacher_evidence": copy.deepcopy(record["teacher_evidence_produced"]),
            "partial_output_policy":
                "DELETE_ONLY_UNSEALED_EXACT_PATHS_AFTER_HASH_AND_OWNERSHIP_VALIDATION",
        }

    def assert_complete_source_coverage(self) -> dict[str, Any]:
        records, _, _ = self._replay()
        schedule = self.expected_contract["window_schedule"]
        if len(records) != len(schedule):
            raise StateError(
                f"source schedule incomplete: declared={len(records)} expected={len(schedule)}"
            )
        ordered = [records[item["window_id"]] for item in schedule]
        if any(record["phase"] != "EVICTED" for record in ordered):
            raise StateError("source completeness requires every scheduled window EVICTED")
        new_fetch_counts = Counter(
            path for record in ordered for path in record["new_fetch_shards"]
        )
        expected_paths = set(self._expected_shards)
        if set(new_fetch_counts) != expected_paths or any(
            count != 1 for count in new_fetch_counts.values()
        ):
            raise StateError("every expected source shard must be new-fetched exactly once")
        new_fetch_bytes = sum(
            self._expected_shards[path]["logical_bytes"] for path in new_fetch_counts
        )
        expected_bytes = self.expected_contract["source"]["expected_logical_bytes"]
        if new_fetch_bytes != expected_bytes:
            raise StateError("new-fetch logical byte total differs from expected source bytes")
        last_status: dict[str, str] = {}
        for record in ordered:
            for path, status in record["source_coverage"].items():
                last_status[path] = status
        non_evicted = sorted(path for path in expected_paths if last_status.get(path) != "EVICTED")
        if non_evicted or ordered[-1]["carry_out_shards"]:
            raise StateError(f"expected source shards not eventually evicted: {non_evicted[:3]}")
        accounting = self.summary()["source_accounting"]
        return {
            "status": "COMPLETE",
            "expected_shard_count": len(expected_paths),
            "expected_logical_bytes": expected_bytes,
            "new_fetch_shard_count": len(new_fetch_counts),
            "new_fetch_logical_bytes": new_fetch_bytes,
            "all_eventually_evicted": True,
            "refetch_event_count": accounting["refetch_event_count"],
            "refetch_logical_bytes": accounting["refetch_logical_bytes"],
            "transfer_amplification": accounting["transfer_amplification"],
        }

    def assert_complete_tensor_coverage(
        self, expected_tensors: Iterable[str] | None = None
    ) -> dict[str, Any]:
        expected = (
            list(self.expected_contract["tensors"]["names"])
            if expected_tensors is None
            else list(expected_tensors)
        )
        if len(expected) != len(set(expected)) or any(not isinstance(item, str) or not item for item in expected):
            raise StateError("expected_tensors must contain unique non-empty names")
        if set(expected) != self._expected_tensors:
            raise StateError("tensor completeness may only use the authoritative expected contract")
        records, owners, _ = self._replay()
        missing = sorted(set(expected) - set(owners))
        unexpected = sorted(set(owners) - set(expected))
        nonterminal = sorted(
            tensor
            for tensor, owner in owners.items()
            if records[owner]["tensor_coverage"][tensor] not in TENSOR_TERMINAL_STATES
        )
        if missing or unexpected or nonterminal:
            raise StateError(
                f"incomplete tensor coverage: missing={missing[:3]}, unexpected={unexpected[:3]}, "
                f"nonterminal={nonterminal[:3]}"
            )
        return {
            "status": "COMPLETE",
            "expected_tensor_count": len(expected),
            "terminal_tensor_count": len(expected),
            "terminal_counts": self.summary()["terminal_counts"],
        }


class Controller:
    """Lease-guarded GLM controller with exact checkpoint/log reconciliation."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        artifact_root: str | os.PathLike[str],
        campaign_id: str,
        source_revision: str,
        expected_contract: Mapping[str, Any],
        telegram_auth: TelegramAuthConfig,
        allow_synthetic_contract: bool = False,
        controller_epoch: str = "glm52-controller-v2",
    ):
        _validate_identity(campaign_id, source_revision, controller_epoch)
        self.root = Path(root)
        self.artifact_store = TrustedArtifactStore(artifact_root)
        self.artifact_root = self.artifact_store.root
        self.campaign_id = campaign_id
        self.source_revision = source_revision
        self.controller_epoch = controller_epoch
        self.expected_contract = _validate_expected_contract(expected_contract)
        if self.expected_contract["campaign_id"] != campaign_id \
                or self.expected_contract["source_revision"] != source_revision:
            raise StateError("controller expected contract identity mismatch")
        if self.expected_contract["source"]["profile"] == "SYNTHETIC_TEST_ONLY" \
                and allow_synthetic_contract is not True:
            raise StateError("synthetic expected contract requires explicit test-only opt-in")
        if not isinstance(telegram_auth, TelegramAuthConfig):
            raise StateError("controller requires TelegramAuthConfig")
        if telegram_auth.expected_chat_identity_digest != \
                self.expected_contract["expected_chat_identity_digest"]:
            raise StateError("Telegram auth chat identity differs from expected contract")
        self.telegram_auth = telegram_auth
        self.expected_contract_sha256 = self.expected_contract["seal_sha256"]
        self.events = HashChainLog(self.root / "GLM52_CONTROLLER_EVENTS.jsonl", schema=EVENT_SCHEMA)
        self.checkpoint_path = self.root / "GLM52_CONTROLLER_CHECKPOINT.json"
        self.lease = SingletonLease(
            self.root / "GLM52_CONTROLLER.lease",
            campaign_id=campaign_id,
            controller_epoch=controller_epoch,
        )
        self.window_ledger = WindowLedger(
            self.root / WINDOW_LEDGER_FILENAME,
            campaign_id=campaign_id,
            source_revision=source_revision,
            expected_contract=self.expected_contract,
            lease_guard=self.lease.assert_held,
        )

    def acquire(self) -> "Controller":
        self.lease.acquire()
        return self

    def close(self) -> None:
        self.lease.close()

    def __enter__(self) -> "Controller":
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _request_identity(self, to_state: str, claim_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema": "hawking.glm52.state_transition_request.v2",
            "campaign_id": self.campaign_id,
            "source_revision": self.source_revision,
            "controller_epoch": self.controller_epoch,
            "expected_contract_sha256": self.expected_contract_sha256,
            "claim_id": claim_id,
            "to_state": to_state,
            "state_payload": _clone(dict(payload)),
        }

    def telegram_dedupe_key(
        self, to_state: str, *, claim_id: str, payload: Mapping[str, Any] | None = None
    ) -> str:
        """Return the dedupe identity from a fully validated prepared transition."""
        return self.prepare_transition(
            to_state, claim_id=claim_id, payload=payload
        )["dedupe_key"]

    def _controller_anchor(self, checkpoint: Mapping[str, Any] | None) -> dict[str, Any]:
        if checkpoint is None:
            from_state = None
            checkpoint_anchor = {
                "event_count": 0,
                "event_head_hash": GENESIS_HASH,
                "window_event_count": 0,
                "window_event_head_hash": GENESIS_HASH,
                "checkpoint_seal_sha256": None,
            }
        else:
            from_state = checkpoint["state"]
            checkpoint_anchor = {
                "event_count": checkpoint["event_count"],
                "event_head_hash": checkpoint["event_head_hash"],
                "window_event_count": checkpoint["window_event_count"],
                "window_event_head_hash": checkpoint["window_event_head_hash"],
                "checkpoint_seal_sha256": checkpoint["seal_sha256"],
            }
        body = {
            "schema": CONTROLLER_ANCHOR_SCHEMA,
            "campaign_id": self.campaign_id,
            "source_revision": self.source_revision,
            "controller_epoch": self.controller_epoch,
            "expected_contract_sha256": self.expected_contract_sha256,
            "from_state": from_state,
            "checkpoint": checkpoint_anchor,
        }
        return {**body, "anchor_sha256": _controller_anchor_hash(body)}

    @staticmethod
    def _telemetry_value(
        telemetry: Mapping[str, Any], key: str, expected_type: type, default: Any
    ) -> Any:
        value = telemetry.get(key, default)
        if expected_type is int:
            return value if not isinstance(value, bool) and isinstance(value, int) and value >= 0 else default
        if expected_type is float:
            return value if not isinstance(value, bool) and isinstance(value, (int, float)) \
                and math.isfinite(float(value)) and float(value) >= 0 else default
        return value if isinstance(value, expected_type) else default

    def _sample_resources(self) -> dict[str, int]:
        """Take a fresh OS-backed resource sample; never synthesize missing values."""
        try:
            disk_free = int(shutil.disk_usage(self.artifact_root).free)
        except OSError as exc:
            raise StateError(f"cannot obtain truthful disk resource sample: {exc}") from exc
        ram_available: int | None = None
        swap_used: int | None = None
        proc_meminfo = Path("/proc/meminfo")
        if proc_meminfo.is_file():
            try:
                values: dict[str, int] = {}
                for line in proc_meminfo.read_text(encoding="ascii").splitlines():
                    name, separator, remainder = line.partition(":")
                    fields = remainder.strip().split()
                    if separator and fields and fields[0].isdigit():
                        values[name] = int(fields[0]) * 1024
                if "SwapTotal" in values and "SwapFree" in values:
                    swap_used = max(0, values["SwapTotal"] - values["SwapFree"])
                ram_available = values.get("MemAvailable")
            except (OSError, UnicodeError, ValueError):
                swap_used = None
        elif sys.platform == "darwin":
            try:
                vm_result = subprocess.run(
                    ["/usr/bin/vm_stat"],
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=5,
                )
                page_match = re.search(r"page size of ([0-9]+) bytes", vm_result.stdout)
                page_counts: dict[str, int] = {}
                for line in vm_result.stdout.splitlines():
                    name, separator, raw_value = line.partition(":")
                    cleaned = raw_value.strip().rstrip(".")
                    if separator and cleaned.isdigit():
                        page_counts[name] = int(cleaned)
                if vm_result.returncode == 0 and page_match is not None:
                    available_pages = sum(
                        page_counts.get(name, 0)
                        for name in ("Pages free", "Pages inactive", "Pages speculative")
                    )
                    ram_available = available_pages * int(page_match.group(1))
                result = subprocess.run(
                    ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=5,
                )
                matched = re.search(r"used\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMG])", result.stdout)
                if result.returncode == 0 and matched is not None:
                    multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3}[matched.group(2)]
                    swap_used = int(float(matched.group(1)) * multiplier)
            except (OSError, subprocess.SubprocessError, ValueError):
                swap_used = None
        if ram_available is None or swap_used is None:
            raise StateError("cannot obtain truthful RAM/swap resource sample")
        return {
            "disk_free_bytes": disk_free,
            "ram_available_bytes": ram_available,
            "swap_used_bytes": swap_used,
        }

    def _canonical_campaign_status(
        self,
        to_state: str,
        requested_payload: Mapping[str, Any],
        checkpoint: Mapping[str, Any] | None,
        *,
        process_pid: int | None = None,
        resource_snapshot: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Derive notification status from the verified checkpoint/window ledger only."""
        records = self.window_ledger.records()
        fetched: set[str] = set()
        verified: set[str] = set()
        latest: dict[str, str] = {}
        for scheduled in self.expected_contract["window_schedule"]:
            record = records.get(scheduled["window_id"])
            if record is None:
                continue
            for path, source_state in record["source_coverage"].items():
                latest[path] = source_state
                if SOURCE_COVERAGE_STATES.index(source_state) >= SOURCE_COVERAGE_STATES.index("FETCHED"):
                    fetched.add(path)
                if SOURCE_COVERAGE_STATES.index(source_state) >= SOURCE_COVERAGE_STATES.index("HASH_VERIFIED"):
                    verified.add(path)
        total = self.expected_contract["source"]["expected_shard_count"]
        window_summary = self.window_ledger.summary()
        durations = 0.0
        for record in records.values():
            start = record.get("download_start")
            end = record.get("download_end")
            if isinstance(start, str) and isinstance(end, str):
                try:
                    started = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    ended = datetime.fromisoformat(end.replace("Z", "+00:00"))
                    durations += max(0.0, (ended - started).total_seconds())
                except ValueError:
                    raise StateError("window ledger contains an invalid download timestamp") from None
        network_bytes = window_summary["source_accounting"]["total_network_bytes"]
        throughput = (network_bytes / durations) if durations > 0 else 0.0
        expected_bytes = self.expected_contract["source"]["expected_logical_bytes"]
        verified_bytes = sum(
            shard["logical_bytes"]
            for path, shard in {
                item["path"]: item for item in self.expected_contract["source"]["shards"]
            }.items()
            if path in verified
        )
        eta = int(max(0, expected_bytes - verified_bytes) / throughput) if throughput > 0 else None
        best_metrics: dict[str, Any] = {}
        if records:
            latest_record = max(records.values(), key=lambda record: record["schedule_index"])
            for name, metric in sorted(latest_record.get("metrics", {}).items()):
                if len(best_metrics) >= 20:
                    break
                if isinstance(name, str) and _SAFE_STATUS_NAME_RE.fullmatch(name) \
                        and (metric is None or isinstance(metric, (bool, int, float))) \
                        and not (isinstance(metric, float) and not math.isfinite(metric)):
                    best_metrics[name] = metric
        active_window = None
        if to_state in WINDOW_STATES:
            requested_window = requested_payload.get("window_id")
            active_window = requested_window if isinstance(requested_window, str) else (
                checkpoint.get("active_window_id") if checkpoint is not None else None
            )
        current_layer = None
        pid = os.getpid() if process_pid is None else process_pid
        resources = (
            self._sample_resources()
            if resource_snapshot is None
            else _require_object(resource_snapshot, "prepared resource snapshot")
        )
        status = {
            "state": to_state,
            "source_coverage_percent": (100.0 * len(verified) / total) if total else 0.0,
            "shards": {
                "fetched": len(fetched),
                "verified": len(verified),
                "evicted": sum(value == "EVICTED" for value in latest.values()),
                "total": total,
            },
            "network_bytes": network_bytes,
            "throughput_bytes_per_second": throughput,
            "eta_seconds": eta,
            "current": {"window": active_window, "layer": current_layer},
            "candidate_rates": [],
            "best_metrics": _clone(best_metrics),
            "resources": resources,
            "process": {
                "pid": pid,
                "lease_held": True,
                "lease_owner": self.lease.owner,
            },
        }
        return _validate_campaign_status(status)

    def _validate_transition_preconditions(
        self,
        current: str | None,
        to_state: str,
        state_payload: Mapping[str, Any],
        checkpoint: Mapping[str, Any] | None,
    ) -> None:
        if current is None:
            if to_state != "PRECHECK":
                raise StateError("controller genesis must enter PRECHECK")
            return
        if current == "COMPLETE":
            raise StateError("COMPLETE is terminal")
        if to_state == "BLOCKED":
            if current == "BLOCKED":
                raise StateError("controller is already BLOCKED")
            _require_name(state_payload.get("reason"), "BLOCKED reason")
        elif current == "BLOCKED":
            if checkpoint is None or to_state != checkpoint.get("blocked_from"):
                raise StateError(
                    f"BLOCKED may resume only to {checkpoint.get('blocked_from') if checkpoint else None}, "
                    f"not {to_state}"
                )
            if not _is_sha256(state_payload.get("resolution_receipt_sha256")):
                raise StateError("BLOCKED recovery requires resolution_receipt_sha256")
        elif to_state not in _FORWARD_TRANSITIONS.get(current, ()):
            raise StateError(
                f"illegal controller transition {current} -> {to_state}; "
                f"allowed={_FORWARD_TRANSITIONS.get(current, ()) + ('BLOCKED',)}"
            )
        if to_state in WINDOW_STATES:
            window_id = state_payload.get("window_id")
            _require_name(window_id, f"{to_state}.window_id")
            assert checkpoint is not None
            if to_state == "FETCH_WINDOW" and current == "BLOCKED" \
                    and window_id != checkpoint.get("active_window_id"):
                raise StateError("BLOCKED window recovery cannot change active_window_id")
            if to_state != "FETCH_WINDOW" and window_id != checkpoint.get("active_window_id"):
                raise StateError("window_id differs from the active dependency window")
            records = self.window_ledger.records()
            if window_id not in records:
                raise StateError(f"controller window state references undeclared window: {window_id}")
            required_phase = {
                "FETCH_WINDOW": "PLANNED",
                "VERIFY_WINDOW": "FETCHED",
                "CAPTURE_TEACHER": "VERIFIED",
                "FIT_CANDIDATES": "TEACHER_CAPTURED",
                "PACK_CANDIDATES": "CANDIDATES_FIT",
                "RUN_WINDOW_FORWARD": "CANDIDATES_PACKED",
                "SEAL_WINDOW": "FORWARD_COMPLETE",
                "EVICT_WINDOW": "SEALED",
            }[to_state]
            if current != "BLOCKED" and records[window_id]["phase"] != required_phase:
                raise StateError(
                    f"entering {to_state} requires window phase {required_phase}, "
                    f"not {records[window_id]['phase']}"
                )
            if current == "EVICT_WINDOW" and to_state == "FETCH_WINDOW":
                previous_window = checkpoint.get("active_window_id")
                if previous_window not in records or records[previous_window]["phase"] != "EVICTED":
                    raise StateError("next source fetch requires the active prior window EVICTED")
        if current == "EVICT_WINDOW" and to_state == "ASSEMBLE_ARTIFACT":
            records = self.window_ledger.records()
            if not records or any(record["phase"] != "EVICTED" for record in records.values()):
                raise StateError("artifact assembly requires every declared window EVICTED")

    def prepare_transition(
        self,
        to_state: str,
        *,
        claim_id: str,
        payload: Mapping[str, Any] | None = None,
        prepared_at: str | None = None,
    ) -> dict[str, Any]:
        """Validate a transition completely before a Telegram send is permitted."""
        self.lease.assert_held()
        if to_state not in STATES:
            raise StateError(f"unknown state: {to_state}")
        _require_claim(claim_id)
        requested_payload = _require_object(dict(payload or {}), "transition requested payload")
        if "terminal_evidence" in requested_payload:
            raise StateError("terminal_evidence is controller-generated and cannot be caller supplied")
        event_history = self.events.verified_events()
        existing = self.events.find_claim(claim_id)
        if existing is not None:
            recorded = existing["payload"].get("transition_intent")
            if existing["kind"] != "STATE_TRANSITION" or not isinstance(recorded, dict) \
                    or recorded.get("to_state") != to_state \
                    or recorded.get("requested_payload") != requested_payload:
                raise StateError(f"claim reused for a different controller operation: {claim_id}")
            return _validate_transition_intent_shape(recorded, self.telegram_auth)
        if not event_history:
            if self.window_ledger.log.verified_events() or self.checkpoint_path.exists():
                raise StateError("boot refused: durable non-genesis history exists")
            checkpoint: dict[str, Any] | None = None
            current = None
        else:
            checkpoint = self.resume()
            current = checkpoint["state"]
        self._claim_in_other_log(claim_id, target="controller")
        self._validate_transition_preconditions(current, to_state, requested_payload, checkpoint)
        anchor = self._controller_anchor(checkpoint)
        prepared_timestamp = prepared_at or (
            checkpoint["written_at"] if checkpoint is not None else self.expected_contract["created_at"]
        )
        state_payload = _clone(requested_payload)
        if to_state in self.expected_contract["state_gates"]:
            state_payload["terminal_evidence"] = make_state_terminal_evidence(
                self.expected_contract,
                to_state,
                artifact_root=self.artifact_root,
                controller_anchor=anchor,
                evidence_auth=self.telegram_auth,
                created_at=prepared_timestamp,
            )
        _validate_terminal_evidence(
            self.expected_contract,
            to_state,
            state_payload.get("terminal_evidence"),
            artifact_store=self.artifact_store,
            controller_anchor=anchor,
            evidence_auth=self.telegram_auth,
        )
        gate = self.expected_contract["state_gates"].get(to_state)
        if gate is not None:
            if gate["require_source_complete"] or gate["require_final_source_eviction"]:
                self.window_ledger.assert_complete_source_coverage()
            if gate["require_tensor_complete"]:
                self.window_ledger.assert_complete_tensor_coverage()
        request_sha = _sha256(self._request_identity(to_state, claim_id, state_payload))
        status = self._canonical_campaign_status(to_state, requested_payload, checkpoint)
        status_sha = _campaign_status_hash(status)
        event_kind = TRANSITION_EVENT_KINDS[to_state]
        dedupe_key = _sha256({
            "schema": "hawking.glm52.transition_notification_identity.v2",
            "campaign_id": self.campaign_id,
            "source_revision": self.source_revision,
            "controller_epoch": self.controller_epoch,
            "expected_contract_sha256": self.expected_contract_sha256,
            "event_kind": event_kind,
            "claim_id": claim_id,
            "from_state": current,
            "to_state": to_state,
            "requested_payload_sha256": _sha256(requested_payload),
            "controller_anchor_sha256": anchor["anchor_sha256"],
            "canonical_status_sha256": status_sha,
        })
        rendered = render_campaign_status_message(
            event_kind,
            dedupe_key,
            status,
            anchor,
            claim_id=claim_id,
            from_state=current,
            to_state=to_state,
        )
        sealed_intent = seal({
            "schema": TRANSITION_INTENT_SCHEMA,
            "campaign_id": self.campaign_id,
            "source_revision": self.source_revision,
            "controller_epoch": self.controller_epoch,
            "expected_contract_sha256": self.expected_contract_sha256,
            "event_kind": event_kind,
            "from_state": current,
            "to_state": to_state,
            "claim_id": claim_id,
            "requested_payload": requested_payload,
            "state_payload": state_payload,
            "request_sha256": request_sha,
            "dedupe_key": dedupe_key,
            "controller_anchor": anchor,
            "canonical_status": status,
            "canonical_status_sha256": status_sha,
            "rendered_message": rendered,
            "rendered_message_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "prepared_at": prepared_timestamp,
        })
        intent = {
            **sealed_intent,
            "controller_hmac_sha256": self.telegram_auth.authenticate({
                "schema": "hawking.glm52.state_transition_intent_auth.v1",
                "intent": sealed_intent,
            }),
        }
        return _validate_transition_intent_shape(intent, self.telegram_auth)

    def _replay_controller(self, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        state: str | None = None
        blocked_from: str | None = None
        active_window_id: str | None = None
        transition_count = 0
        heartbeat: dict[str, Any] | None = None
        telegram: dict[str, Any] | None = None
        telegram_response_hashes: set[str] = set()
        telegram_message_ids: set[tuple[str, int]] = set()
        window_events = self.window_ledger.log.verified_events()
        for event in events:
            payload = event["payload"]
            if payload.get("campaign_id") != self.campaign_id \
                    or payload.get("source_revision") != self.source_revision \
                    or payload.get("controller_epoch") != self.controller_epoch \
                    or payload.get("expected_contract_sha256") != self.expected_contract_sha256:
                raise StateError(f"controller identity mismatch at event seq {event['seq']}")
            kind = event["kind"]
            if kind == "STATE_TRANSITION":
                from_state = payload.get("from_state")
                to_state = payload.get("to_state")
                state_payload = payload.get("state_payload")
                if not isinstance(state_payload, dict):
                    raise StateError("state transition payload is not an object")
                if from_state != state:
                    raise StateError(
                        f"state chain split at seq {event['seq']}: expected from={state}, got {from_state}"
                    )
                if state is None:
                    if to_state != "PRECHECK":
                        raise StateError("controller genesis must enter PRECHECK")
                elif to_state == "BLOCKED":
                    if state in {"COMPLETE", "BLOCKED"}:
                        raise StateError(f"cannot block from {state}")
                    reason = state_payload.get("reason")
                    _require_name(reason, "BLOCKED reason")
                    blocked_from = state
                elif state == "BLOCKED":
                    if to_state != blocked_from:
                        raise StateError(
                            f"BLOCKED may resume only to {blocked_from}, not {to_state}"
                        )
                    if not _is_sha256(state_payload.get("resolution_receipt_sha256")):
                        raise StateError("BLOCKED recovery requires resolution_receipt_sha256")
                    blocked_from = None
                elif to_state not in _FORWARD_TRANSITIONS.get(state, ()):
                    raise StateError(f"illegal controller transition: {state} -> {to_state}")
                if to_state not in STATES:
                    raise StateError(f"unknown controller state in event: {to_state}")
                intent = _validate_transition_intent_shape(
                    payload.get("transition_intent"), self.telegram_auth
                )
                if intent["campaign_id"] != self.campaign_id \
                        or intent["source_revision"] != self.source_revision \
                        or intent["controller_epoch"] != self.controller_epoch \
                        or intent["expected_contract_sha256"] != self.expected_contract_sha256 \
                        or intent["from_state"] != from_state or intent["to_state"] != to_state \
                        or intent["claim_id"] != event["claim_id"] \
                        or intent["state_payload"] != state_payload:
                    raise StateError(f"prepared transition identity mismatch at seq {event['seq']}")
                request = self._request_identity(to_state, event["claim_id"], state_payload)
                request_sha = _sha256(request)
                if payload.get("request_sha256") != request_sha \
                        or intent["request_sha256"] != request_sha:
                    raise StateError(f"transition request identity mismatch at seq {event['seq']}")
                anchor = _validate_controller_anchor(intent["controller_anchor"])
                checkpoint_anchor = anchor["checkpoint"]
                if anchor["campaign_id"] != self.campaign_id \
                        or anchor["source_revision"] != self.source_revision \
                        or anchor["controller_epoch"] != self.controller_epoch \
                        or anchor["expected_contract_sha256"] != self.expected_contract_sha256 \
                        or anchor["from_state"] != from_state:
                    raise StateError(f"controller anchor identity mismatch at seq {event['seq']}")
                if checkpoint_anchor["event_count"] != event["seq"] \
                        or checkpoint_anchor["event_head_hash"] != event["prev_hash"]:
                    raise StateError(f"controller anchor event prefix mismatch at seq {event['seq']}")
                window_count = checkpoint_anchor["window_event_count"]
                if window_count > len(window_events) or self._anchor_at(
                    window_events, window_count
                ) != checkpoint_anchor["window_event_head_hash"]:
                    raise StateError(f"controller anchor window prefix mismatch at seq {event['seq']}")
                if event["seq"] == 0:
                    if checkpoint_anchor["checkpoint_seal_sha256"] is not None:
                        raise StateError("genesis transition cannot anchor a prior checkpoint seal")
                elif not _is_sha256(checkpoint_anchor["checkpoint_seal_sha256"]):
                    raise StateError("non-genesis transition lacks prior checkpoint seal anchor")
                receipt = _validate_delivery_receipt(
                    payload.get("telegram_delivery"), intent, self.telegram_auth
                )
                response_hash = receipt["bot_api_response_sha256"]
                message_identity = (
                    receipt["chat_identity_digest"], receipt["message_id"]
                )
                if response_hash in telegram_response_hashes or message_identity in telegram_message_ids:
                    raise StateError("Telegram Bot API response/message reused across transitions")
                telegram_response_hashes.add(response_hash)
                telegram_message_ids.add(message_identity)
                _validate_terminal_evidence(
                    self.expected_contract,
                    to_state,
                    state_payload.get("terminal_evidence"),
                    artifact_store=self.artifact_store,
                    controller_anchor=anchor,
                    evidence_auth=self.telegram_auth,
                )
                event_heartbeat = payload.get("heartbeat")
                if not isinstance(event_heartbeat, dict) or event_heartbeat.get("state") != to_state:
                    raise StateError("every state transition must produce a state-bound heartbeat")
                _require_name(event_heartbeat.get("at"), "transition heartbeat time")
                commit_pid = event_heartbeat.get("pid")
                if isinstance(commit_pid, bool) or not isinstance(commit_pid, int) or commit_pid <= 0:
                    raise StateError("transition commit heartbeat pid invalid")
                if event_heartbeat.get("notification_pid") != \
                        intent["canonical_status"]["process"]["pid"]:
                    raise StateError("transition notification pid binding mismatch")
                if to_state in WINDOW_STATES:
                    window_id = state_payload.get("window_id")
                    _require_name(window_id, f"{to_state}.window_id")
                    if to_state == "FETCH_WINDOW":
                        active_window_id = window_id
                    elif active_window_id != window_id:
                        raise StateError(
                            f"window identity changed inside pipeline: {active_window_id} -> {window_id}"
                        )
                elif state == "EVICT_WINDOW" and to_state == "ASSEMBLE_ARTIFACT":
                    active_window_id = None
                state = to_state
                transition_count += 1
                heartbeat = {**event_heartbeat, "event_seq": event["seq"]}
                telegram = {
                    "status": "DELIVERED",
                    "dedupe_key": intent["dedupe_key"],
                    "event_kind": intent["event_kind"],
                    "canonical_status_sha256": intent["canonical_status_sha256"],
                    "rendered_message_sha256": intent["rendered_message_sha256"],
                    "controller_anchor_sha256": anchor["anchor_sha256"],
                    "receipt_hmac_sha256": receipt["hmac_sha256"],
                    "delivery_proof_sha256": response_hash,
                    "delivery_proof_kind": "BOT_API_RESPONSE",
                    "event_seq": event["seq"],
                }
            elif kind == "HEARTBEAT":
                if state is None or payload.get("state") != state:
                    raise StateError(f"heartbeat state mismatch at seq {event['seq']}")
                heartbeat_value = payload.get("heartbeat")
                if not isinstance(heartbeat_value, dict):
                    raise StateError("heartbeat event lacks heartbeat object")
                _require_name(heartbeat_value.get("at"), "heartbeat time")
                telemetry = heartbeat_value.get("telemetry")
                if not isinstance(telemetry, dict):
                    raise StateError("heartbeat telemetry is not an object")
                expected_request_sha = _sha256(
                    {
                        "kind": "HEARTBEAT",
                        "campaign_id": self.campaign_id,
                        "source_revision": self.source_revision,
                        "controller_epoch": self.controller_epoch,
                        "expected_contract_sha256": self.expected_contract_sha256,
                        "claim_id": event["claim_id"],
                        "state": state,
                        "telemetry": telemetry,
                    }
                )
                if payload.get("request_sha256") != expected_request_sha:
                    raise StateError(f"heartbeat request identity mismatch at seq {event['seq']}")
                heartbeat = {**heartbeat_value, "state": state, "event_seq": event["seq"]}
            else:
                raise StateError(f"unknown controller event kind: {kind}")
        return {
            "state": state,
            "blocked_from": blocked_from,
            "active_window_id": active_window_id,
            "state_transition_count": transition_count,
            "heartbeat": heartbeat,
            "telegram": telegram,
        }

    def _checkpoint_document(
        self,
        events: Sequence[Mapping[str, Any]],
        window_events: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        replay = self._replay_controller(events)
        records = self.window_ledger.records()
        if replay["state"] in WINDOW_STATES:
            active = replay["active_window_id"]
            if active not in records:
                raise StateError("checkpoint references an undeclared active window")
            allowed_phases = {
                "FETCH_WINDOW": {"PLANNED", "FETCHING", "FETCHED"},
                "VERIFY_WINDOW": {"FETCHED", "VERIFIED"},
                "CAPTURE_TEACHER": {"VERIFIED", "TEACHER_CAPTURED"},
                "FIT_CANDIDATES": {"TEACHER_CAPTURED", "CANDIDATES_FIT"},
                "PACK_CANDIDATES": {"CANDIDATES_FIT", "CANDIDATES_PACKED"},
                "RUN_WINDOW_FORWARD": {"CANDIDATES_PACKED", "FORWARD_COMPLETE"},
                "SEAL_WINDOW": {"FORWARD_COMPLETE", "SEALED"},
                "EVICT_WINDOW": {"SEALED", "EVICTED"},
            }[replay["state"]]
            if records[active]["phase"] not in allowed_phases:
                raise StateError(
                    f"controller/window checkpoint mismatch: {replay['state']} with "
                    f"phase {records[active]['phase']}"
                )
        effective_state = (
            replay["blocked_from"] if replay["state"] == "BLOCKED" else replay["state"]
        )
        campaign_completeness = None
        if effective_state is not None and STATES.index(effective_state) >= STATES.index(
            "ASSEMBLE_ARTIFACT"
        ):
            source_complete = self.window_ledger.assert_complete_source_coverage()
            tensor_complete = self.window_ledger.assert_complete_tensor_coverage()
            campaign_completeness = {
                "expected_contract_sha256": self.expected_contract_sha256,
                "source": source_complete,
                "tensors": tensor_complete,
            }
        window_summary = self.window_ledger.summary()
        window_resume_plan = self.window_ledger.resume_plan()
        last_event = events[-1] if events else None
        last_window = window_events[-1] if window_events else None
        return seal(
            {
                "schema": CHECKPOINT_SCHEMA,
                "campaign_id": self.campaign_id,
                "source_revision": self.source_revision,
                "controller_epoch": self.controller_epoch,
                "expected_contract_sha256": self.expected_contract_sha256,
                **replay,
                "event_count": len(events),
                "event_head_hash": self.events.head_hash(events),
                "window_event_count": len(window_events),
                "window_event_head_hash": self.window_ledger.log.head_hash(window_events),
                "last_claim_id": last_event["claim_id"] if last_event else None,
                "last_window_claim_id": last_window["claim_id"] if last_window else None,
                "window_summary": window_summary,
                "window_resume_plan": window_resume_plan,
                "campaign_completeness": campaign_completeness,
                "written_at": utc_now(),
            }
        )

    def _write_checkpoint(self) -> dict[str, Any]:
        self.lease.assert_held()
        events = self.events.verified_events()
        _, _, window_events = self.window_ledger._replay()
        checkpoint = self._checkpoint_document(events, window_events)
        atomic_json(self.checkpoint_path, checkpoint)
        try:
            restored = read_sealed_json(self.checkpoint_path)
        except Glm52Error as exc:
            raise StateError(f"checkpoint post-write validation failed: {exc}") from exc
        if restored != checkpoint:
            raise StateError("checkpoint post-write content mismatch")
        return checkpoint

    def _read_checkpoint(self) -> dict[str, Any]:
        try:
            checkpoint = read_sealed_json(self.checkpoint_path)
        except Glm52Error as exc:
            raise StateError(str(exc)) from exc
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise StateError("checkpoint schema mismatch")
        for key, expected in (
            ("campaign_id", self.campaign_id),
            ("source_revision", self.source_revision),
            ("controller_epoch", self.controller_epoch),
            ("expected_contract_sha256", self.expected_contract_sha256),
        ):
            if checkpoint.get(key) != expected:
                raise StateError(f"checkpoint {key} mismatch (split-brain identity)")
        return checkpoint

    @staticmethod
    def _anchor_at(events: Sequence[Mapping[str, Any]], count: int) -> str:
        if count == 0:
            return GENESIS_HASH
        return str(events[count - 1]["chain_sha256"])

    def resume(self, *, recover_single_tail: bool = True) -> dict[str, Any]:
        """Verify exact durable state, optionally sealing one deterministic crash tail.

        A single valid event ahead of either checkpoint anchor is the only recoverable
        write-order gap.  More than one orphan event, tails in both logs, a checkpoint
        ahead of a log, or any prefix/hash mismatch is rejected as ambiguous split-brain.
        """
        self.lease.assert_held()
        events = self.events.verified_events()
        _, _, window_events = self.window_ledger._replay()
        if not self.checkpoint_path.exists():
            if recover_single_tail and len(events) == 1 and not window_events:
                replay = self._replay_controller(events)
                if replay["state"] != "PRECHECK":
                    raise StateError("checkpoint absent and genesis is not PRECHECK")
                return self._write_checkpoint()
            raise StateError("no checkpoint to resume (or ambiguous uncheckpointed history)")
        checkpoint = self._read_checkpoint()
        event_count = checkpoint.get("event_count")
        window_count = checkpoint.get("window_event_count")
        if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 0:
            raise StateError("checkpoint event_count invalid")
        if isinstance(window_count, bool) or not isinstance(window_count, int) or window_count < 0:
            raise StateError("checkpoint window_event_count invalid")
        if event_count > len(events) or window_count > len(window_events):
            raise StateError("checkpoint is ahead of a log (tail truncation/split-brain)")
        if self._anchor_at(events, event_count) != checkpoint.get("event_head_hash"):
            raise StateError("controller log prefix differs from checkpoint (fork/split-brain)")
        if self._anchor_at(window_events, window_count) != checkpoint.get("window_event_head_hash"):
            raise StateError("window log prefix differs from checkpoint (fork/split-brain)")
        event_tail = len(events) - event_count
        window_tail = len(window_events) - window_count
        if event_tail == 0 and window_tail == 0:
            expected = self._checkpoint_document(events, window_events)
            for key in (
                "state",
                "blocked_from",
                "active_window_id",
                "state_transition_count",
                "heartbeat",
                "telegram",
                "last_claim_id",
                "last_window_claim_id",
                "window_summary",
                "window_resume_plan",
                "campaign_completeness",
            ):
                if checkpoint.get(key) != expected.get(key):
                    raise StateError(f"checkpoint replay mismatch for {key} (split-brain)")
            return checkpoint
        if not recover_single_tail:
            raise StateError("log/checkpoint tail mismatch and recovery disabled")
        if (event_tail, window_tail) not in {(1, 0), (0, 1)}:
            raise StateError(
                f"ambiguous uncheckpointed tails: controller={event_tail}, window={window_tail}"
            )
        # Full replay above has already proved the one tail is sequenced and semantically valid.
        return self._write_checkpoint()

    def _claim_in_other_log(self, claim_id: str, *, target: str) -> None:
        other = self.window_ledger.log if target == "controller" else self.events
        if other.find_claim(claim_id) is not None:
            raise StateError(f"one-use claim already consumed by the other durable log: {claim_id}")

    def boot(
        self,
        *,
        transition_intent: Mapping[str, Any] | None = None,
        claim_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        telegram_delivery: Mapping[str, Any],
    ) -> dict[str, Any]:
        if transition_intent is None:
            embedded = telegram_delivery.get("transition_intent") \
                if isinstance(telegram_delivery, Mapping) else None
            if embedded is not None:
                transition_intent = embedded
            elif claim_id is None:
                raise StateError("boot requires transition_intent or claim_id")
            else:
                transition_intent = self.prepare_transition(
                    "PRECHECK", claim_id=claim_id, payload=payload
                )
        intent = _validate_transition_intent_shape(transition_intent, self.telegram_auth)
        if claim_id is not None and intent["claim_id"] != claim_id:
            raise StateError("boot claim differs from prepared intent")
        if payload is not None and intent["requested_payload"] != _clone(dict(payload)):
            raise StateError("boot payload differs from prepared intent")
        if intent["from_state"] is not None or intent["to_state"] != "PRECHECK":
            raise StateError("boot requires a prepared genesis -> PRECHECK intent")
        return self.commit_transition(intent, telegram_delivery=telegram_delivery)

    def _append_transition(
        self,
        transition_intent: Mapping[str, Any],
        telegram_delivery: Mapping[str, Any],
    ) -> dict[str, Any]:
        intent = _validate_transition_intent_shape(transition_intent, self.telegram_auth)
        receipt = _validate_delivery_receipt(telegram_delivery, intent, self.telegram_auth)
        heartbeat = {
            "at": utc_now(),
            "state": intent["to_state"],
            "pid": os.getpid(),
            "notification_pid": intent["canonical_status"]["process"]["pid"],
        }
        self.events.append(
            "STATE_TRANSITION",
            intent["claim_id"],
            {
                "campaign_id": self.campaign_id,
                "source_revision": self.source_revision,
                "controller_epoch": self.controller_epoch,
                "expected_contract_sha256": self.expected_contract_sha256,
                "from_state": intent["from_state"],
                "to_state": intent["to_state"],
                "state_payload": intent["state_payload"],
                "request_sha256": intent["request_sha256"],
                "transition_intent": intent,
                "heartbeat": heartbeat,
                "telegram_delivery": receipt,
            },
        )
        return self._write_checkpoint()

    def commit_transition(
        self,
        transition_intent: Mapping[str, Any],
        *,
        telegram_delivery: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically append a still-current prepared transition after exact delivery."""
        self.lease.assert_held()
        intent = _validate_transition_intent_shape(transition_intent, self.telegram_auth)
        for key, expected in (
            ("campaign_id", self.campaign_id),
            ("source_revision", self.source_revision),
            ("controller_epoch", self.controller_epoch),
            ("expected_contract_sha256", self.expected_contract_sha256),
        ):
            if intent.get(key) != expected:
                raise StateError(f"prepared transition {key} differs from this controller")
        existing = self.events.find_claim(intent["claim_id"])
        if existing is not None:
            if existing["kind"] != "STATE_TRANSITION" \
                    or existing["payload"].get("transition_intent", {}).get("seal_sha256") != \
                    intent["seal_sha256"]:
                raise StateError(
                    f"claim reused for a different controller operation: {intent['claim_id']}"
                )
            return self.resume()
        event_history = self.events.verified_events()
        if not event_history:
            if self.window_ledger.log.verified_events() or self.checkpoint_path.exists():
                raise StateError("genesis commit refused: unexpected durable history exists")
            checkpoint: dict[str, Any] | None = None
            current = None
        else:
            checkpoint = self.resume()
            current = checkpoint["state"]
        if intent["from_state"] != current:
            raise StateError("prepared transition is stale: controller state changed")
        current_anchor = self._controller_anchor(checkpoint)
        if intent["controller_anchor"] != current_anchor:
            raise StateError("prepared transition is stale: durable controller anchor changed")
        self._claim_in_other_log(intent["claim_id"], target="controller")
        requested_payload = intent["requested_payload"]
        self._validate_transition_preconditions(current, intent["to_state"], requested_payload, checkpoint)
        expected_state_payload = _clone(requested_payload)
        gate = self.expected_contract["state_gates"].get(intent["to_state"])
        if gate is not None:
            evidence = intent["state_payload"].get("terminal_evidence")
            expected_state_payload["terminal_evidence"] = evidence
            _validate_terminal_evidence(
                self.expected_contract,
                intent["to_state"],
                evidence,
                artifact_store=self.artifact_store,
                controller_anchor=current_anchor,
                evidence_auth=self.telegram_auth,
            )
            if gate["require_source_complete"] or gate["require_final_source_eviction"]:
                self.window_ledger.assert_complete_source_coverage()
            if gate["require_tensor_complete"]:
                self.window_ledger.assert_complete_tensor_coverage()
        else:
            _validate_terminal_evidence(
                self.expected_contract,
                intent["to_state"],
                intent["state_payload"].get("terminal_evidence"),
                artifact_store=self.artifact_store,
                controller_anchor=current_anchor,
                evidence_auth=self.telegram_auth,
            )
        if intent["state_payload"] != expected_state_payload:
            raise StateError("prepared transition final payload contains caller-controlled evidence")
        expected_request_sha = _sha256(
            self._request_identity(intent["to_state"], intent["claim_id"], expected_state_payload)
        )
        if intent["request_sha256"] != expected_request_sha:
            raise StateError("prepared transition request hash mismatch")
        process_pid = intent["canonical_status"]["process"]["pid"]
        expected_status = self._canonical_campaign_status(
            intent["to_state"],
            requested_payload,
            checkpoint,
            process_pid=process_pid,
            resource_snapshot=intent["canonical_status"]["resources"],
        )
        if intent["canonical_status"] != expected_status \
                or intent["canonical_status_sha256"] != _campaign_status_hash(expected_status):
            raise StateError("prepared transition campaign status is not controller-derived")
        expected_dedupe = _sha256({
            "schema": "hawking.glm52.transition_notification_identity.v2",
            "campaign_id": self.campaign_id,
            "source_revision": self.source_revision,
            "controller_epoch": self.controller_epoch,
            "expected_contract_sha256": self.expected_contract_sha256,
            "event_kind": intent["event_kind"],
            "claim_id": intent["claim_id"],
            "from_state": current,
            "to_state": intent["to_state"],
            "requested_payload_sha256": _sha256(requested_payload),
            "controller_anchor_sha256": current_anchor["anchor_sha256"],
            "canonical_status_sha256": intent["canonical_status_sha256"],
        })
        if intent["dedupe_key"] != expected_dedupe:
            raise StateError("prepared transition dedupe identity mismatch")
        _validate_delivery_receipt(telegram_delivery, intent, self.telegram_auth)
        return self._append_transition(intent, telegram_delivery)

    def transition(
        self,
        to_state: str,
        *,
        transition_intent: Mapping[str, Any] | None = None,
        claim_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        telegram_delivery: Mapping[str, Any],
    ) -> dict[str, Any]:
        if transition_intent is None:
            embedded = telegram_delivery.get("transition_intent") \
                if isinstance(telegram_delivery, Mapping) else None
            if embedded is not None:
                transition_intent = embedded
            elif claim_id is None:
                raise StateError("transition requires transition_intent or claim_id")
            else:
                transition_intent = self.prepare_transition(
                    to_state, claim_id=claim_id, payload=payload
                )
        intent = _validate_transition_intent_shape(transition_intent, self.telegram_auth)
        if claim_id is not None and intent["claim_id"] != claim_id:
            raise StateError("transition claim differs from prepared intent")
        if payload is not None and intent["requested_payload"] != _clone(dict(payload)):
            raise StateError("transition payload differs from prepared intent")
        if intent["to_state"] != to_state:
            raise StateError("transition target differs from prepared intent")
        return self.commit_transition(intent, telegram_delivery=telegram_delivery)

    def heartbeat(
        self,
        *,
        claim_id: str,
        telemetry: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.lease.assert_held()
        _require_claim(claim_id)
        telemetry_copy = _require_object(dict(telemetry or {}), "heartbeat telemetry")
        checkpoint = self.resume()
        existing = self.events.find_claim(claim_id)
        request_state = (
            existing["payload"].get("state")
            if existing is not None and existing["kind"] == "HEARTBEAT"
            else checkpoint["state"]
        )
        request_sha = _sha256(
            {
                "kind": "HEARTBEAT",
                "campaign_id": self.campaign_id,
                "source_revision": self.source_revision,
                "controller_epoch": self.controller_epoch,
                "expected_contract_sha256": self.expected_contract_sha256,
                "claim_id": claim_id,
                "state": request_state,
                "telemetry": telemetry_copy,
            }
        )
        if existing is not None:
            if existing["kind"] != "HEARTBEAT" \
                    or existing["payload"].get("request_sha256") != request_sha:
                raise StateError(f"claim reused for a different heartbeat: {claim_id}")
            return checkpoint
        self._claim_in_other_log(claim_id, target="controller")
        heartbeat = {"at": utc_now(), "pid": os.getpid(), "telemetry": telemetry_copy}
        self.events.append(
            "HEARTBEAT",
            claim_id,
            {
                "campaign_id": self.campaign_id,
                "source_revision": self.source_revision,
                "controller_epoch": self.controller_epoch,
                "expected_contract_sha256": self.expected_contract_sha256,
                "state": checkpoint["state"],
                "heartbeat": heartbeat,
                "request_sha256": request_sha,
            },
        )
        return self._write_checkpoint()

    def declare_window(self, **kwargs: Any) -> dict[str, Any]:
        self.lease.assert_held()
        claim_id = kwargs.get("claim_id")
        _require_claim(claim_id)
        checkpoint = self.resume()
        self._claim_in_other_log(claim_id, target="window")
        schedule_index = kwargs.get("schedule_index")
        if isinstance(schedule_index, bool) or not isinstance(schedule_index, int):
            raise StateError("window declaration requires integer schedule_index")
        if schedule_index == 0:
            if checkpoint["state"] != "FREEZE_PROGRAM":
                raise StateError("first window may be declared only in FREEZE_PROGRAM")
        else:
            if checkpoint["state"] != "EVICT_WINDOW":
                raise StateError("subsequent windows may be declared only in EVICT_WINDOW")
            active = checkpoint.get("active_window_id")
            records = self.window_ledger.records()
            if active not in records or records[active]["phase"] != "EVICTED":
                raise StateError("subsequent window declaration requires active prior window EVICTED")
        result = self.window_ledger.declare_window(**kwargs)
        self._write_checkpoint()
        return result

    def advance_window(self, window_id: str, to_phase: str, **kwargs: Any) -> dict[str, Any]:
        self.lease.assert_held()
        claim_id = kwargs.get("claim_id")
        _require_claim(claim_id)
        checkpoint = self.resume()
        self._claim_in_other_log(claim_id, target="window")
        required_state = {
            "FETCHING": "FETCH_WINDOW",
            "FETCHED": "FETCH_WINDOW",
            "VERIFIED": "VERIFY_WINDOW",
            "TEACHER_CAPTURED": "CAPTURE_TEACHER",
            "CANDIDATES_FIT": "FIT_CANDIDATES",
            "CANDIDATES_PACKED": "PACK_CANDIDATES",
            "FORWARD_COMPLETE": "RUN_WINDOW_FORWARD",
            "SEALED": "SEAL_WINDOW",
            "EVICTED": "EVICT_WINDOW",
        }.get(to_phase)
        if required_state is None:
            raise StateError(f"Controller cannot advance a window to phase {to_phase}")
        if checkpoint["state"] != required_state:
            raise StateError(
                f"window phase {to_phase} requires controller state {required_state}, "
                f"not {checkpoint['state']}"
            )
        if checkpoint.get("active_window_id") != window_id:
            raise StateError("window mutation does not target the active dependency window")
        result = self.window_ledger.advance(window_id, to_phase, **kwargs)
        self._write_checkpoint()
        return result

    def record_window_retry(self, window_id: str, **kwargs: Any) -> dict[str, Any]:
        self.lease.assert_held()
        claim_id = kwargs.get("claim_id")
        _require_claim(claim_id)
        checkpoint = self.resume()
        self._claim_in_other_log(claim_id, target="window")
        if checkpoint.get("active_window_id") != window_id:
            raise StateError("retry does not target the active dependency window")
        result = self.window_ledger.record_retry(window_id, **kwargs)
        self._write_checkpoint()
        return result

    def status(self) -> dict[str, Any]:
        events_ok, event_reasons = self.events.verify_chain()
        windows_ok, window_reasons = self.window_ledger.log.verify_chain()
        checkpoint_seal_ok = False
        checkpoint_anchor_ok = False
        checkpoint_replay_ok = False
        checkpoint_state = None
        checkpoint_heartbeat: dict[str, Any] | None = None
        checkpoint_reasons: list[str] = []
        if self.checkpoint_path.exists():
            try:
                checkpoint = self._read_checkpoint()
                checkpoint_seal_ok = True
                if not events_ok or not windows_ok:
                    raise StateError("cannot reconcile checkpoint against a corrupt durable log")
                events = self.events.verified_events()
                _, _, window_events = self.window_ledger._replay()
                if checkpoint.get("event_count") != len(events) \
                        or checkpoint.get("window_event_count") != len(window_events):
                    raise StateError("checkpoint/log event counts differ (stale or truncated checkpoint)")
                if checkpoint.get("event_head_hash") != self.events.head_hash(events) \
                        or checkpoint.get("window_event_head_hash") != \
                        self.window_ledger.log.head_hash(window_events):
                    raise StateError("checkpoint/log head anchors differ")
                checkpoint_anchor_ok = True
                expected = self._checkpoint_document(events, window_events)
                for key in (
                    "state", "blocked_from", "active_window_id", "state_transition_count",
                    "heartbeat", "telegram", "last_claim_id", "last_window_claim_id",
                    "window_summary", "window_resume_plan", "campaign_completeness",
                ):
                    if checkpoint.get(key) != expected.get(key):
                        raise StateError(f"checkpoint replay differs for {key}")
                # Optimistic read consistency: never report green if either log moved while
                # status was composing its replay result.
                events_after = self.events.verified_events()
                _, _, window_after = self.window_ledger._replay()
                if self.events.head_hash(events_after) != self.events.head_hash(events) \
                        or len(events_after) != len(events) \
                        or self.window_ledger.log.head_hash(window_after) != \
                        self.window_ledger.log.head_hash(window_events) \
                        or len(window_after) != len(window_events):
                    raise StateError("durable logs changed during status reconciliation")
                checkpoint_replay_ok = True
                checkpoint_state = checkpoint.get("state")
                checkpoint_heartbeat = checkpoint.get("heartbeat")
            except StateError as exc:
                checkpoint_reasons.append(str(exc))
        else:
            checkpoint_reasons.append("checkpoint absent")
        durable_state_ok = (
            events_ok and windows_ok and checkpoint_seal_ok
            and checkpoint_anchor_ok and checkpoint_replay_ok
        )
        if durable_state_ok:
            heartbeat_fresh_ok, heartbeat_reason, heartbeat_at = _heartbeat_freshness(
                checkpoint_heartbeat
            )
        else:
            heartbeat_fresh_ok = False
            heartbeat_reason = "durable checkpoint unavailable for heartbeat validation"
            heartbeat_at = None
        lease_observation = self.lease.probe()
        live_worker_lease_ok = bool(
            lease_observation["live_lock_held"]
            and lease_observation["owner_record_ok"]
            and lease_observation["owner"] == self.lease.owner
            and lease_observation["owner_pid_alive"]
            and lease_observation["controller_epoch"] == self.controller_epoch
            and heartbeat_fresh_ok
        )
        return {
            "campaign_id": self.campaign_id,
            "source_revision": self.source_revision,
            "state": checkpoint_state if durable_state_ok else None,
            "lease_held": self.lease.held,
            "lease_observation": lease_observation,
            "live_worker_lease_ok": live_worker_lease_ok,
            "heartbeat_at": heartbeat_at,
            "heartbeat_fresh_ok": heartbeat_fresh_ok,
            "heartbeat_freshness_reason": heartbeat_reason,
            "heartbeat_max_age_seconds": CONTROLLER_HEARTBEAT_MAX_AGE_SECONDS,
            "controller_event_chain_ok": events_ok,
            "controller_event_chain_reasons": event_reasons,
            "window_event_chain_ok": windows_ok,
            "window_event_chain_reasons": window_reasons,
            "checkpoint_seal_ok": checkpoint_seal_ok,
            "checkpoint_anchor_ok": checkpoint_anchor_ok,
            "checkpoint_replay_ok": checkpoint_replay_ok,
            "checkpoint_reasons": checkpoint_reasons,
            "durable_state_ok": durable_state_ok,
        }


def selfcheck() -> dict[str, Any]:
    """Tiny offline smoke check; exhaustive adversarial cases live in pytest."""
    import tempfile

    revision = "b4734de4facf877f85769a911abafc5283eab3d9"
    campaign_id = "glm52-selfcheck"
    chat_id = "offline-selfcheck-chat"
    chat_digest = telegram_chat_identity_digest(chat_id)
    auth = TelegramAuthConfig(
        hmac_key=b"glm52-offline-selfcheck-key-material-32-bytes",
        expected_chat_identity_digest=chat_digest,
    )
    shard = {
        "path": "model-00001-of-00001.safetensors",
        "logical_bytes": 1024,
        "content_hash": "a" * 64,
        "content_hash_kind": "xet",
    }
    gates = {
        "ASSEMBLE_ARTIFACT": {
            "require_source_complete": True,
            "require_tensor_complete": True,
            "require_final_source_eviction": True,
            "require_telegram_delivery": True,
            "require_phone_status": False,
            "required_phone_status_path": None,
            "required_artifacts": {},
            "required_checklist": {},
        },
        "COMPLETE": {
            "require_source_complete": True,
            "require_tensor_complete": True,
            "require_final_source_eviction": True,
            "require_telegram_delivery": True,
            "require_phone_status": True,
            "required_phone_status_path": "GLM52_PHONE_STATUS.json",
            "required_artifacts": {
                "final": {
                    "path": "GLM52_GRAVITY_FINAL.json",
                    "expected_seal_sha256": None,
                    "expected_schema": "hawking.glm52.gravity_final.v1",
                    "allowed_statuses": ["PASS"],
                    "validator_id": "campaign_artifact_v1",
                    "validator_source_sha256": EVIDENCE_VALIDATOR_SOURCE_SHA256[
                        "campaign_artifact_v1"
                    ],
                    "require_producer_hmac": True,
                }
            },
            "required_checklist": {
                "all_stop_conditions": {
                    "path": "evidence/all_stop_conditions.json",
                    "expected_seal_sha256": None,
                    "expected_schema": STOP_CONDITION_EVIDENCE_SCHEMA,
                    "allowed_statuses": ["PASS"],
                    "validator_id": "stop_condition_v1",
                    "validator_source_sha256": EVIDENCE_VALIDATOR_SOURCE_SHA256[
                        "stop_condition_v1"
                    ],
                    "require_producer_hmac": True,
                }
            },
        },
    }
    contract = make_expected_campaign_contract(
        campaign_id=campaign_id,
        source_revision=revision,
        expected_chat_identity_digest=chat_digest,
        source_shards=[shard],
        expected_tensors=["selfcheck.weight"],
        window_schedule=[{
            "schedule_index": 0,
            "window_id": "selfcheck-window",
            "source_shards": [shard["path"]],
            "carry_in_shards": [],
            "new_fetch_shards": [shard["path"]],
            "refetch_shards": [],
            "carry_out_shards": [],
            "evict_shards": [shard["path"]],
            "tensor_set": ["selfcheck.weight"],
        }],
        state_gates=gates,
        source_profile="SYNTHETIC_TEST_ONLY",
        created_at="2026-07-21T00:00:00Z",
    )
    with tempfile.TemporaryDirectory() as directory:
        controller = Controller(
            Path(directory) / "state",
            artifact_root=directory,
            campaign_id=campaign_id,
            source_revision=revision,
            expected_contract=contract,
            telegram_auth=auth,
            allow_synthetic_contract=True,
        )
        with controller:
            claim = "selfcheck:boot:0001"
            intent = controller.prepare_transition("PRECHECK", claim_id=claim)
            controller.boot(
                transition_intent=intent,
                telegram_delivery=make_telegram_delivery_receipt(
                    intent,
                    auth=auth,
                    bot_api_response={
                        "ok": True,
                        "result": {
                            "message_id": 1,
                            "chat": {"id": chat_id},
                            "text": intent["rendered_message"],
                        },
                    },
                    http_status=200,
                ),
            )
            resumed = controller.resume()
            if resumed["state"] != "PRECHECK":
                raise StateError("selfcheck resume state mismatch")
            ok, reasons = controller.events.verify_chain()
            if not ok:
                raise StateError(f"selfcheck event chain failed: {reasons}")
        return {
            "status": "PASS",
            "exact_state_count": len(STATES),
            "event_chain_ok": True,
            "telegram_receipts_hmac_authenticated": True,
            "authoritative_expected_contract": True,
            "exact_resume": True,
            "no_model_io": True,
        }


if __name__ == "__main__":
    print(json.dumps(selfcheck(), indent=2, sort_keys=True))

#!/usr/bin/env python3.12
"""Fail-closed, offline notification journal for the GLM-5.2 campaign.

Only a live, lease-holding :class:`glm52_state.Controller` may prepare a
notification.  Facts, status, evidence, and controller anchors are derived
inside that controller from its verified durable ledgers; this module accepts
none of them from a caller.  It performs no network I/O.

The journal has three deliberately separate Ed25519 roles:

* the journal owns only the producer signer used for intents and journal heads;
* a sender owns the delivery-receipt signer and gives the journal its verifier;
* an operator owns the reconciliation signer and gives the journal its verifier.

Every on-disk head is producer-signed and is also advanced, one entry at a
time, into the controller event chain/checkpoint.  This makes deletion, prefix
rollback, alternate tails, and path/inode replacement fail closed.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from glm52_common import Glm52Error, canonical, seal, utc_now, verify_sealed
from glm52_state import (
    CONTROLLER_ANCHOR_SCHEMA,
    Controller,
    GENESIS_HASH,
    NOTIFICATION_HEAD_BINDING_SCHEMA,
    OFFICIAL_WEIGHT_LOGICAL_BYTES,
    OFFICIAL_WEIGHT_SHARD_COUNT,
    StateError,
    telegram_chat_identity_digest,
)


CAMPAIGN_ID = "glm52-bf16-xet-gravity"
OFFICIAL_REPO = "zai-org/GLM-5.2"
OFFICIAL_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
OFFICIAL_WEIGHT_SHARDS = OFFICIAL_WEIGHT_SHARD_COUNT
OFFICIAL_WINDOWS = 20
OFFICIAL_TENSORS = 59_585

STATUS_SCHEMA = "hawking.glm52.notification_status.v2"
FACTS_SCHEMA = "hawking.glm52.notification_facts.v2"
INTENT_SCHEMA = "hawking.glm52.notification_intent.v2"
ENTRY_SCHEMA = "hawking.glm52.notification_journal_entry.v2"
HEAD_SCHEMA = "hawking.glm52.notification_journal_head.v2"
BOT_RECEIPT_SCHEMA = "hawking.glm52.notification_bot_delivery_receipt.v2"
RECONCILIATION_CHALLENGE_SCHEMA = (
    "hawking.glm52.notification_reconciliation_challenge.v2"
)
RECONCILIATION_SCHEMA = "hawking.glm52.notification_reconciliation.v2"
OUTBOX_AUDIT_SCHEMA = "hawking.glm52.notification_outbox_audit.v1"

INTENT_AUTH_SCHEMA = "hawking.glm52.notification_intent_auth.v2"
ENTRY_AUTH_SCHEMA = "hawking.glm52.notification_journal_entry_auth.v2"
HEAD_AUTH_SCHEMA = "hawking.glm52.notification_journal_head_auth.v2"
BINDING_AUTH_SCHEMA = "hawking.glm52.notification_head_binding_auth.v1"
BOT_RECEIPT_AUTH_SCHEMA = "hawking.glm52.notification_bot_receipt_auth.v2"
RECONCILIATION_AUTH_SCHEMA = "hawking.glm52.notification_reconciliation_auth.v2"

MAX_JOURNAL_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_MESSAGE_CHARS = 4096

KIMI_RELEASE = "kimi_release"
GLM_ADMISSION = "glm_admission"
XET_AUTOTUNE_RESULT = "xet_autotune_result"
SOURCE_STREAM_STARTED = "source_stream_started"
SOURCE_WINDOW_VERIFIED = "source_window_verified"
SOURCE_WINDOW_COMPLETED = "source_window_completed"
SOURCE_WINDOW_EVICTED = "source_window_evicted"
SOURCE_COVERAGE_5_PERCENT = "source_coverage_5_percent"
CANDIDATE_PROMOTION = "candidate_promotion"
CANDIDATE_RETIREMENT = "candidate_retirement"
DOCTOR_DIAGNOSIS = "doctor_diagnosis"
DOCTOR_TREATMENT = "doctor_treatment"
COMPLETE_COMPACT_ARTIFACT = "complete_compact_artifact"
RATE_DESCENT = "rate_descent"
FULL_COMPACT_RUN = "full_compact_run"
FAULT = "fault"
RESUME = "resume"
FINAL_RESULT = "final_result"

EVENT_KINDS: tuple[str, ...] = (
    KIMI_RELEASE,
    GLM_ADMISSION,
    XET_AUTOTUNE_RESULT,
    SOURCE_STREAM_STARTED,
    SOURCE_WINDOW_VERIFIED,
    SOURCE_WINDOW_COMPLETED,
    SOURCE_WINDOW_EVICTED,
    SOURCE_COVERAGE_5_PERCENT,
    CANDIDATE_PROMOTION,
    CANDIDATE_RETIREMENT,
    DOCTOR_DIAGNOSIS,
    DOCTOR_TREATMENT,
    COMPLETE_COMPACT_ARTIFACT,
    RATE_DESCENT,
    FULL_COMPACT_RUN,
    FAULT,
    RESUME,
    FINAL_RESULT,
)

PREPARED = "PREPARED"
SEND_STARTED = "SEND_STARTED"
AMBIGUOUS_BLOCKED = "AMBIGUOUS_BLOCKED"
REPLAY_AUTHORIZED = "REPLAY_AUTHORIZED"
COMMITTED = "COMMITTED"
LIFECYCLE_KINDS = frozenset(
    {PREPARED, SEND_STARTED, AMBIGUOUS_BLOCKED, REPLAY_AUTHORIZED, COMMITTED}
)

_SUPPORTED_STATES: dict[str, frozenset[str]] = {
    KIMI_RELEASE: frozenset({"RELEASE_KIMI_SOURCE"}),
    GLM_ADMISSION: frozenset({"ADMIT_GLM_SOURCE"}),
    XET_AUTOTUNE_RESULT: frozenset({"BUILD_ADAPTER"}),
    SOURCE_STREAM_STARTED: frozenset({"FETCH_WINDOW"}),
    SOURCE_WINDOW_VERIFIED: frozenset({"VERIFY_WINDOW", "CAPTURE_TEACHER"}),
    SOURCE_WINDOW_COMPLETED: frozenset({"SEAL_WINDOW", "EVICT_WINDOW"}),
    SOURCE_WINDOW_EVICTED: frozenset({"EVICT_WINDOW"}),
    SOURCE_COVERAGE_5_PERCENT: frozenset({
        "VERIFY_WINDOW", "CAPTURE_TEACHER", "FIT_CANDIDATES",
        "PACK_CANDIDATES", "RUN_WINDOW_FORWARD", "SEAL_WINDOW", "EVICT_WINDOW",
    }),
    FINAL_RESULT: frozenset({"COMPLETE"}),
}
_UNSUPPORTED = frozenset(EVENT_KINDS) - frozenset(_SUPPORTED_STATES)
_PRIMARY_OUTCOMES = frozenset(
    {"OUTCOME_A", "OUTCOME_B", "OUTCOME_C", "OUTCOME_D", "OUTCOME_E", "OUTCOME_F"}
)
_HALF_BIT_CLASSIFICATIONS = frozenset({
    "HALF_BIT_F1_REACHABLE",
    "HALF_BIT_F2_PASS_ONLY",
    "HALF_BIT_PHYSICAL_ONLY",
    "HALF_BIT_CAPABILITY_PASS",
    "HALF_BIT_CLOSED_IN_TESTED_REGION",
})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SIG_RE = re.compile(r"^[0-9a-f]{128}$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_SAFE_TEXT_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,1024}$")


class NotificationError(Glm52Error):
    """A notification provenance, durability, or lifecycle gate failed."""


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NotificationError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise NotificationError(f"{label} fields invalid")


def _uint(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NotificationError(f"{label} must be a non-negative integer")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or float(value) < 0:
        raise NotificationError(f"{label} must be a finite non-negative number")
    return float(value)


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise NotificationError(f"{label} must be a strict UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise NotificationError(f"{label} is not a real UTC timestamp") from None
    if parsed.tzinfo != timezone.utc:
        raise NotificationError(f"{label} is not UTC")
    return parsed


def _safe_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_TEXT_RE.fullmatch(value) is None:
        raise NotificationError(f"{label} must be bounded printable text")
    return value


def telegram_bot_identity_digest(bot_id: int | str) -> str:
    if isinstance(bot_id, bool) or not isinstance(bot_id, (int, str)):
        raise NotificationError("Telegram bot id invalid")
    rendered = str(bot_id)
    if not rendered or not rendered.lstrip("-").isdigit():
        raise NotificationError("Telegram bot id invalid")
    return hashlib.sha256(
        b"hawking.glm52.telegram-bot-identity.v1\0" + rendered.encode("ascii")
    ).hexdigest()


def _public_bytes(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _private_from_bytes(value: bytes | None) -> Ed25519PrivateKey:
    if value is None:
        return Ed25519PrivateKey.generate()
    if not isinstance(value, bytes) or len(value) != 32:
        raise NotificationError("Ed25519 private key must be exactly 32 bytes")
    try:
        return Ed25519PrivateKey.from_private_bytes(value)
    except ValueError:
        raise NotificationError("Ed25519 private key invalid") from None


@dataclass(frozen=True)
class NotificationVerifier:
    """Public-only verifier bound to one notification role."""

    role: str
    public_key: bytes

    def __post_init__(self) -> None:
        if self.role not in {"producer", "receipt", "reconciliation"}:
            raise NotificationError("unknown notification signing role")
        if not isinstance(self.public_key, bytes) or len(self.public_key) != 32:
            raise NotificationError("Ed25519 public key must be exactly 32 bytes")

    @property
    def key_id(self) -> str:
        return hashlib.sha256(self.public_key).hexdigest()

    @property
    def public_key_hex(self) -> str:
        return self.public_key.hex()

    def verify(self, schema: str, body: Mapping[str, Any], signature: Any) -> None:
        if not isinstance(signature, str) or _SIG_RE.fullmatch(signature) is None:
            raise NotificationError(f"{self.role} Ed25519 signature invalid")
        envelope = {"schema": schema, self.role: copy.deepcopy(dict(body))}
        # Binding verification is shared with glm52_state and intentionally uses
        # the fixed field name expected there.
        if schema == BINDING_AUTH_SCHEMA:
            envelope = {"schema": schema, "binding": copy.deepcopy(dict(body))}
        try:
            Ed25519PublicKey.from_public_bytes(self.public_key).verify(
                bytes.fromhex(signature), canonical(envelope)
            )
        except (InvalidSignature, ValueError):
            raise NotificationError(f"{self.role} Ed25519 signature invalid") from None


class _RoleSigner:
    _role: str

    def __init__(self, private_key: bytes | None = None) -> None:
        self.__private_key = _private_from_bytes(private_key)

    @property
    def verifier(self) -> NotificationVerifier:
        return NotificationVerifier(
            self._role, _public_bytes(self.__private_key.public_key())
        )

    @property
    def key_id(self) -> str:
        return self.verifier.key_id

    def _sign(self, schema: str, body: Mapping[str, Any]) -> str:
        field = "binding" if schema == BINDING_AUTH_SCHEMA else self._role
        envelope = {"schema": schema, field: copy.deepcopy(dict(body))}
        return self.__private_key.sign(canonical(envelope)).hex()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key_id={self.key_id!r})"


class NotificationProducerSigner(_RoleSigner):
    _role = "producer"


class NotificationReceiptSigner(_RoleSigner):
    """Sender-held key and exact Telegram response validator."""

    _role = "receipt"

    def __init__(
        self,
        private_key: bytes | None = None,
        *,
        expected_chat_identity_digest: str,
        expected_bot_identity_digest: str,
        producer_verifier: NotificationVerifier,
    ) -> None:
        super().__init__(private_key)
        if not _is_sha256(expected_chat_identity_digest) \
                or not _is_sha256(expected_bot_identity_digest):
            raise NotificationError("expected Telegram identity digests invalid")
        if producer_verifier.role != "producer":
            raise NotificationError("receipt signer requires the producer verifier")
        self.expected_chat_identity_digest = expected_chat_identity_digest
        self.expected_bot_identity_digest = expected_bot_identity_digest
        self.producer_verifier = producer_verifier

    def make_delivery_receipt(
        self,
        intent_value: Mapping[str, Any],
        send_started_entry: Mapping[str, Any],
        *,
        bot_api_response: Mapping[str, Any],
        http_status: int,
        delivered_at: str | None = None,
    ) -> dict[str, Any]:
        intent = validate_notification_intent(intent_value, self.producer_verifier)
        entry = validate_journal_entry_signature(send_started_entry, self.producer_verifier)
        if entry["kind"] != SEND_STARTED or entry["intent_sha256"] != intent["seal_sha256"]:
            raise NotificationError("receipt is not bound to the exact SEND_STARTED entry")
        if isinstance(http_status, bool) or not isinstance(http_status, int) \
                or http_status < 200 or http_status >= 300:
            raise NotificationError("Telegram delivery did not return a 2xx HTTP status")
        try:
            response_bytes = canonical(bot_api_response)
        except (TypeError, ValueError):
            raise NotificationError("Telegram response is not canonical JSON") from None
        if len(response_bytes) > MAX_RESPONSE_BYTES:
            raise NotificationError("Telegram response exceeded the safe limit")
        response = _object(dict(bot_api_response), "Telegram Bot response")
        if response.get("ok") is not True:
            raise NotificationError("Telegram Bot response is not successful")
        result = _object(response.get("result"), "Telegram Bot response result")
        if result.get("text") != intent["rendered_message"]:
            raise NotificationError("Telegram did not echo the exact prepared message")
        message_id = _uint(result.get("message_id"), "Telegram message_id")
        if message_id == 0:
            raise NotificationError("Telegram message_id must be positive")
        chat = _object(result.get("chat"), "Telegram response chat")
        chat_digest = telegram_chat_identity_digest(chat.get("id"))
        if chat_digest != self.expected_chat_identity_digest:
            raise NotificationError("Telegram response came from a different configured chat")
        sender = _object(result.get("from"), "Telegram response sender")
        if sender.get("is_bot") is not True:
            raise NotificationError("Telegram response sender is not a bot")
        bot_digest = telegram_bot_identity_digest(sender.get("id"))
        if bot_digest != self.expected_bot_identity_digest:
            raise NotificationError("Telegram response came from a different bot")
        delivered = delivered_at or utc_now()
        delivered_dt = _parse_utc(delivered, "receipt delivered_at")
        prepared_dt = _parse_utc(intent["prepared_at"], "intent prepared_at")
        started_dt = _parse_utc(entry["recorded_at"], "SEND_STARTED recorded_at")
        if not prepared_dt <= started_dt <= delivered_dt:
            raise NotificationError("receipt timestamps do not follow prepare/send/deliver order")
        body = {
            "schema": BOT_RECEIPT_SCHEMA,
            "status": "DELIVERED",
            "intent_sha256": intent["seal_sha256"],
            "dedupe_key": intent["dedupe_key"],
            "attempt": entry["attempt"],
            "send_started_seq": entry["seq"],
            "send_started_chain_sha256": entry["chain_sha256"],
            "send_started_signature": entry["producer_signature"],
            "message_id": message_id,
            "chat_identity_digest": chat_digest,
            "bot_identity_digest": bot_digest,
            "rendered_message_sha256": intent["rendered_message_sha256"],
            "bot_api_response_sha256": hashlib.sha256(response_bytes).hexdigest(),
            "http_status": http_status,
            "delivered_at": delivered,
            "receipt_key_id": self.key_id,
        }
        signature = self._sign(BOT_RECEIPT_AUTH_SCHEMA, body)
        return seal({**body, "receipt_signature": signature})


class NotificationReconciliationSigner(_RoleSigner):
    """Operator-held key; it never enters the journal object."""

    _role = "reconciliation"

    def authorize(
        self,
        challenge_value: Mapping[str, Any],
        *,
        authorization_id: str,
        reason: str,
        authorized_at: str | None = None,
    ) -> dict[str, Any]:
        challenge = validate_reconciliation_challenge(challenge_value)
        authorization_id = _safe_text(authorization_id, "authorization_id")
        reason = _safe_text(reason, "reconciliation reason")
        timestamp = authorized_at or utc_now()
        if _parse_utc(timestamp, "authorized_at") < _parse_utc(
            challenge["issued_at"], "challenge issued_at"
        ):
            raise NotificationError("reconciliation authorization predates its challenge")
        body = {
            "schema": RECONCILIATION_SCHEMA,
            "status": "REPLAY_AUTHORIZED",
            "authorization_id": authorization_id,
            "challenge_sha256": challenge["seal_sha256"],
            "intent_sha256": challenge["intent_sha256"],
            "dedupe_key": challenge["dedupe_key"],
            "ambiguous_seq": challenge["ambiguous_seq"],
            "ambiguous_chain_sha256": challenge["ambiguous_chain_sha256"],
            "send_started_seq": challenge["send_started_seq"],
            "send_started_chain_sha256": challenge["send_started_chain_sha256"],
            "authorized_attempt": challenge["authorized_attempt"],
            "journal_binding_sha256": challenge["journal_binding_sha256"],
            "reason": reason,
            "authorized_at": timestamp,
            "reconciliation_key_id": self.key_id,
        }
        return seal({
            **body,
            "reconciliation_signature": self._sign(RECONCILIATION_AUTH_SCHEMA, body),
        })


def _validate_snapshot(value: Any, controller: Controller) -> dict[str, Any]:
    snapshot = _object(copy.deepcopy(value), "controller notification snapshot")
    try:
        verify_sealed(snapshot, label="controller notification snapshot")
    except Glm52Error as exc:
        raise NotificationError(str(exc)) from exc
    required = {
        "schema", "campaign_id", "source_revision", "controller_epoch",
        "expected_contract_sha256", "state", "controller_anchor", "source",
        "windows", "tensors", "current_window", "best_metrics", "candidate_rates",
        "resources", "process", "evidence_seals", "terminal_evidence", "final_result",
        "captured_at", "seal_sha256",
    }
    _exact(snapshot, required, "controller notification snapshot")
    if snapshot["schema"] != "hawking.glm52.notification_snapshot.v1":
        raise NotificationError("controller notification snapshot schema invalid")
    for key, expected in (
        ("campaign_id", controller.campaign_id),
        ("source_revision", controller.source_revision),
        ("controller_epoch", controller.controller_epoch),
        ("expected_contract_sha256", controller.expected_contract_sha256),
    ):
        if snapshot.get(key) != expected:
            raise NotificationError(f"controller notification snapshot {key} mismatch")
    if snapshot.get("state") == "BLOCKED":
        raise NotificationError("BLOCKED controller state cannot prepare notifications")
    _parse_utc(snapshot.get("captured_at"), "snapshot captured_at")
    if snapshot.get("process", {}).get("lease_held") is not True:
        raise NotificationError("notification snapshot does not hold the controller lease")
    if not isinstance(snapshot.get("evidence_seals"), dict) \
            or not snapshot["evidence_seals"] \
            or any(not isinstance(name, str) or not _is_sha256(digest)
                   for name, digest in snapshot["evidence_seals"].items()):
        raise NotificationError("notification snapshot evidence seals invalid")
    return snapshot


def _coverage(expected: int, verified: int) -> dict[str, Any]:
    _uint(expected, "expected source bytes")
    _uint(verified, "verified source bytes")
    if expected <= 0 or verified > expected:
        raise NotificationError("verified source coverage exceeds its exact denominator")
    millionths = verified * 100_000_000 // expected
    return {
        "verified_logical_bytes": verified,
        "total_logical_bytes": expected,
        "percent_millionths": millionths,
        "percent": f"{millionths / 1_000_000:.6f}",
    }


def _canonical_status(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    source = _object(snapshot.get("source"), "snapshot source")
    windows = _object(snapshot.get("windows"), "snapshot windows")
    tensors = _object(snapshot.get("tensors"), "snapshot tensors")
    current_value = snapshot.get("current_window")
    current: dict[str, Any] | None = None
    if current_value is not None:
        current_snapshot = _object(current_value, "snapshot current window")
        record = _object(current_snapshot.get("record"), "snapshot current window record")
        schedule = _object(current_snapshot.get("schedule"), "snapshot current window schedule")
        current = {
            "window_id": record.get("window_id"),
            "schedule_index": record.get("schedule_index"),
            "phase": record.get("phase"),
            "organs": copy.deepcopy(current_snapshot.get("organs")),
            "layers": copy.deepcopy(current_snapshot.get("layers")),
            "window_record_seal_sha256": record.get("seal_sha256"),
            "scheduled_source_shards_sha256": _sha256(schedule.get("source_shards")),
            "scheduled_tensors_sha256": _sha256(schedule.get("tensor_set")),
        }
    status = {
        "schema": STATUS_SCHEMA,
        "state": snapshot.get("state"),
        "source_coverage": _coverage(
            source.get("expected_logical_bytes"), source.get("verified_logical_bytes")
        ),
        "counts": {
            "shards_fetched": source.get("fetched_shard_count"),
            "shards_verified": source.get("verified_shard_count"),
            "shards_evicted": source.get("evicted_shard_count"),
            "shards_total": source.get("expected_shard_count"),
            "windows_verified": windows.get("verified"),
            "windows_completed": windows.get("completed"),
            "windows_evicted": windows.get("evicted"),
            "windows_total": windows.get("total"),
            "tensors_terminal": tensors.get("terminal"),
            "tensors_total": tensors.get("total"),
        },
        "network": {
            "bytes": source.get("network_bytes"),
            "throughput_bytes_per_second": source.get("throughput_bytes_per_second"),
            "eta_seconds": source.get("eta_seconds"),
        },
        "current": current,
        "candidate_rates": copy.deepcopy(snapshot.get("candidate_rates")),
        "best_metrics": copy.deepcopy(snapshot.get("best_metrics")),
        "resources": copy.deepcopy(snapshot.get("resources")),
        "process": copy.deepcopy(snapshot.get("process")),
        "evidence_seals": copy.deepcopy(snapshot.get("evidence_seals")),
        "controller_snapshot_sha256": snapshot.get("seal_sha256"),
        "captured_at": snapshot.get("captured_at"),
    }
    return validate_canonical_status(status)


def _validate_controller_anchor(
    value: Any,
    *,
    identity: Mapping[str, str],
    expected_state: str,
) -> dict[str, Any]:
    anchor = _object(copy.deepcopy(value), "notification controller anchor")
    _exact(anchor, {
        "schema", "campaign_id", "source_revision", "controller_epoch",
        "expected_contract_sha256", "from_state", "checkpoint", "anchor_sha256",
    }, "notification controller anchor")
    if anchor.get("schema") != CONTROLLER_ANCHOR_SCHEMA:
        raise NotificationError("notification controller anchor schema invalid")
    for key, expected in identity.items():
        if anchor.get(key) != expected:
            raise NotificationError(f"notification controller anchor {key} differs")
    if anchor.get("from_state") != expected_state:
        raise NotificationError("notification controller anchor state differs from status")
    checkpoint = _object(anchor.get("checkpoint"), "notification checkpoint anchor")
    _exact(checkpoint, {
        "event_count", "event_head_hash", "window_event_count", "window_event_head_hash",
        "checkpoint_seal_sha256",
    }, "notification checkpoint anchor")
    for key in ("event_count", "window_event_count"):
        _uint(checkpoint.get(key), f"notification checkpoint {key}")
    for key in ("event_head_hash", "window_event_head_hash"):
        if not _is_sha256(checkpoint.get(key)):
            raise NotificationError(f"notification checkpoint {key} invalid")
    if not _is_sha256(checkpoint.get("checkpoint_seal_sha256")):
        raise NotificationError("notification checkpoint seal invalid")
    unsigned = {key: item for key, item in anchor.items() if key != "anchor_sha256"}
    if anchor.get("anchor_sha256") != _sha256(unsigned):
        raise NotificationError("notification controller anchor hash mismatch")
    return anchor


def validate_canonical_status(value: Any) -> dict[str, Any]:
    status = _object(copy.deepcopy(value), "canonical notification status")
    _exact(status, {
        "schema", "state", "source_coverage", "counts", "network", "current",
        "candidate_rates", "best_metrics", "resources", "process",
        "evidence_seals", "controller_snapshot_sha256", "captured_at",
    }, "canonical notification status")
    if status.get("schema") != STATUS_SCHEMA or not isinstance(status.get("state"), str) \
            or status.get("state") == "BLOCKED":
        raise NotificationError("canonical notification status schema/state invalid")
    coverage = _object(status.get("source_coverage"), "source coverage")
    _exact(coverage, {
        "verified_logical_bytes", "total_logical_bytes", "percent_millionths", "percent"
    }, "source coverage")
    expected_coverage = _coverage(
        coverage.get("total_logical_bytes"), coverage.get("verified_logical_bytes")
    )
    if coverage != expected_coverage:
        raise NotificationError("source coverage is not exact to its denominator")
    counts = _object(status.get("counts"), "notification counts")
    count_keys = {
        "shards_fetched", "shards_verified", "shards_evicted", "shards_total",
        "windows_verified", "windows_completed", "windows_evicted", "windows_total",
        "tensors_terminal", "tensors_total",
    }
    _exact(counts, count_keys, "notification counts")
    for key in count_keys:
        _uint(counts.get(key), f"notification counts.{key}")
    if not counts["shards_verified"] <= counts["shards_fetched"] <= counts["shards_total"] \
            or counts["shards_evicted"] > counts["shards_fetched"]:
        raise NotificationError("source shard counts are inconsistent")
    if not counts["windows_evicted"] <= counts["windows_completed"] \
            <= counts["windows_verified"] <= counts["windows_total"]:
        raise NotificationError("window counts are inconsistent")
    if counts["tensors_terminal"] > counts["tensors_total"]:
        raise NotificationError("tensor counts are inconsistent")
    network = _object(status.get("network"), "notification network")
    _exact(network, {"bytes", "throughput_bytes_per_second", "eta_seconds"},
           "notification network")
    _uint(network.get("bytes"), "network bytes")
    _finite(network.get("throughput_bytes_per_second"), "network throughput")
    if network.get("eta_seconds") is not None:
        _uint(network["eta_seconds"], "network eta_seconds")
    current = status.get("current")
    if current is not None:
        current = _object(current, "notification current window")
        _exact(current, {
            "window_id", "schedule_index", "phase", "organs", "layers",
            "window_record_seal_sha256", "scheduled_source_shards_sha256",
            "scheduled_tensors_sha256",
        }, "notification current window")
        if not isinstance(current.get("window_id"), str) or not current["window_id"] \
                or not isinstance(current.get("phase"), str) or not current["phase"]:
            raise NotificationError("current window identity/phase invalid")
        _uint(current.get("schedule_index"), "current schedule_index")
        if not isinstance(current.get("organs"), list) \
                or current["organs"] != sorted(set(current["organs"])) \
                or any(not isinstance(item, str) or not item for item in current["organs"]):
            raise NotificationError("current organ list invalid")
        if not isinstance(current.get("layers"), list) \
                or current["layers"] != sorted(set(current["layers"])):
            raise NotificationError("current layer list invalid")
        for layer in current["layers"]:
            _uint(layer, "current layer")
        for key in (
            "window_record_seal_sha256", "scheduled_source_shards_sha256",
            "scheduled_tensors_sha256",
        ):
            if not _is_sha256(current.get(key)):
                raise NotificationError(f"current {key} invalid")
    rates = status.get("candidate_rates")
    if not isinstance(rates, list) or any(
        not isinstance(rate, str) or re.fullmatch(r"^(?:0|[1-9]\d*)(?:\.\d+)?$", rate) is None
        for rate in rates
    ):
        raise NotificationError("candidate rates invalid")
    if not isinstance(status.get("best_metrics"), dict):
        raise NotificationError("best metrics must be an object")
    resources = _object(status.get("resources"), "notification resources")
    _exact(resources, {"disk_free_bytes", "ram_available_bytes", "swap_used_bytes"},
           "notification resources")
    for key in resources:
        _uint(resources[key], f"resources.{key}")
    process = _object(status.get("process"), "notification process")
    _exact(process, {"pid", "lease_held", "lease_owner"}, "notification process")
    if _uint(process.get("pid"), "process pid") == 0 \
            or process.get("lease_held") is not True \
            or not isinstance(process.get("lease_owner"), str) \
            or not process["lease_owner"]:
        raise NotificationError("notification process lacks the live held controller lease")
    seals = status.get("evidence_seals")
    if not isinstance(seals, dict) or not seals or any(
        not isinstance(name, str) or not name or not _is_sha256(digest)
        for name, digest in seals.items()
    ):
        raise NotificationError("canonical status requires non-empty evidence seals")
    if not _is_sha256(status.get("controller_snapshot_sha256")):
        raise NotificationError("controller snapshot seal invalid")
    _parse_utc(status.get("captured_at"), "status captured_at")
    return status


def _terminal_seal(snapshot: Mapping[str, Any], *, required: bool) -> str | None:
    evidence = snapshot.get("terminal_evidence")
    if evidence is None:
        if required:
            raise NotificationError("milestone lacks validated controller terminal evidence")
        return None
    evidence = _object(evidence, "terminal evidence")
    if not _is_sha256(evidence.get("seal_sha256")):
        raise NotificationError("terminal evidence seal invalid")
    return evidence["seal_sha256"]


def _window_subject(snapshot: Mapping[str, Any], *, phases: set[str]) -> dict[str, Any]:
    current = _object(snapshot.get("current_window"), "current grounded window")
    record = _object(current.get("record"), "current grounded window record")
    schedule = _object(current.get("schedule"), "current frozen schedule record")
    if record.get("phase") not in phases:
        raise NotificationError(
            f"window milestone is premature for durable phase {record.get('phase')!r}"
        )
    if record.get("window_id") != schedule.get("window_id") \
            or record.get("schedule_index") != schedule.get("schedule_index"):
        raise NotificationError("window record differs from the frozen schedule")
    return {
        "window_id": record["window_id"],
        "schedule_index": record["schedule_index"],
        "phase": record["phase"],
        "window_record_seal_sha256": record["seal_sha256"],
        "source_shards_sha256": _sha256(schedule.get("source_shards")),
        "tensor_set_sha256": _sha256(schedule.get("tensor_set")),
        "organs": copy.deepcopy(current.get("organs")),
        "layers": copy.deepcopy(current.get("layers")),
        "download_start": record.get("download_start"),
        "download_end": record.get("download_end"),
    }


def _derive_event_facts(
    controller: Controller,
    event_kind: str,
    snapshot: Mapping[str, Any],
    *,
    coverage_threshold: int | None = None,
) -> dict[str, Any]:
    if event_kind not in EVENT_KINDS:
        raise NotificationError("unknown notification event kind")
    if event_kind in _UNSUPPORTED:
        raise NotificationError(
            f"{event_kind} has no dedicated grounded evidence schema and is refused"
        )
    state = snapshot.get("state")
    if state not in _SUPPORTED_STATES[event_kind]:
        raise NotificationError(
            f"{event_kind} is unsupported in verified controller state {state}"
        )
    terminal_required = (
        controller.expected_contract["source"]["profile"] == "OFFICIAL_GLM52_BF16"
        and event_kind in {KIMI_RELEASE, GLM_ADMISSION, XET_AUTOTUNE_RESULT, FINAL_RESULT}
    )
    subject: dict[str, Any]
    if event_kind == KIMI_RELEASE:
        subject = {
            "release_status": "SAFE_RELEASE_VERIFIED",
            "terminal_evidence_seal_sha256": _terminal_seal(
                snapshot, required=terminal_required
            ),
        }
    elif event_kind == GLM_ADMISSION:
        subject = {
            "repo": OFFICIAL_REPO,
            "revision": controller.source_revision,
            "admission_status": "PASS",
            "terminal_evidence_seal_sha256": _terminal_seal(
                snapshot, required=terminal_required
            ),
        }
    elif event_kind == XET_AUTOTUNE_RESULT:
        subject = {
            "result_status": "PASS",
            "terminal_evidence_seal_sha256": _terminal_seal(
                snapshot, required=terminal_required
            ),
        }
    elif event_kind == SOURCE_STREAM_STARTED:
        subject = _window_subject(snapshot, phases={"FETCHING", "FETCHED"})
        if subject["download_start"] is None:
            raise NotificationError("source stream lacks its durable download start")
    elif event_kind == SOURCE_WINDOW_VERIFIED:
        subject = _window_subject(snapshot, phases={
            "VERIFIED", "TEACHER_CAPTURED", "CANDIDATES_FIT", "CANDIDATES_PACKED",
            "FORWARD_COMPLETE", "SEALED", "EVICTED",
        })
    elif event_kind == SOURCE_WINDOW_COMPLETED:
        subject = _window_subject(snapshot, phases={"SEALED", "EVICTED"})
    elif event_kind == SOURCE_WINDOW_EVICTED:
        subject = _window_subject(snapshot, phases={"EVICTED"})
    elif event_kind == SOURCE_COVERAGE_5_PERCENT:
        if coverage_threshold is None or coverage_threshold not in range(5, 101, 5):
            raise NotificationError("coverage notification requires an exact 5% crossing")
        source = _object(snapshot.get("source"), "snapshot source")
        expected = _uint(source.get("expected_logical_bytes"), "expected source bytes")
        verified = _uint(source.get("verified_logical_bytes"), "verified source bytes")
        if verified * 100 < expected * coverage_threshold:
            raise NotificationError("coverage threshold has not been reached")
        subject = {
            "threshold_percent": coverage_threshold,
            "verified_logical_bytes_at_prepare": verified,
            "total_logical_bytes": expected,
            "verified_paths_sha256": source.get("verified_paths_sha256"),
        }
    elif event_kind == FINAL_RESULT:
        final = _object(snapshot.get("final_result"), "controller final result")
        if final.get("result_status") != "PASS" \
                or final.get("outcome") not in _PRIMARY_OUTCOMES \
                or final.get("half_bit_classification") not in _HALF_BIT_CLASSIFICATIONS \
                or not _is_sha256(final.get("final_report_seal_sha256")):
            raise NotificationError("final notification requires an authenticated PASS result")
        subject = {
            **copy.deepcopy(final),
            "terminal_evidence_seal_sha256": _terminal_seal(snapshot, required=True),
        }
    else:  # pragma: no cover - guarded by the supported map above
        raise NotificationError("unsupported notification event kind")
    facts = {
        "schema": FACTS_SCHEMA,
        "event_kind": event_kind,
        "controller_state": state,
        "controller_snapshot_sha256": snapshot["seal_sha256"],
        "subject": subject,
        "evidence_seals": copy.deepcopy(snapshot["evidence_seals"]),
    }
    return validate_event_facts(facts, expected_event_kind=event_kind)


def validate_event_facts(
    value: Any, *, expected_event_kind: str | None = None
) -> dict[str, Any]:
    facts = _object(copy.deepcopy(value), "notification event facts")
    _exact(facts, {
        "schema", "event_kind", "controller_state", "controller_snapshot_sha256",
        "subject", "evidence_seals",
    }, "notification event facts")
    if facts.get("schema") != FACTS_SCHEMA or facts.get("event_kind") not in _SUPPORTED_STATES:
        raise NotificationError("notification event facts schema/kind invalid")
    if expected_event_kind is not None and facts["event_kind"] != expected_event_kind:
        raise NotificationError("notification facts event kind differs")
    if facts.get("controller_state") not in _SUPPORTED_STATES[facts["event_kind"]]:
        raise NotificationError("notification facts are premature for controller state")
    if not _is_sha256(facts.get("controller_snapshot_sha256")) \
            or not isinstance(facts.get("subject"), dict) or not facts["subject"]:
        raise NotificationError("notification facts provenance/subject invalid")
    seals = facts.get("evidence_seals")
    if not isinstance(seals, dict) or not seals or any(
        not isinstance(name, str) or not _is_sha256(digest)
        for name, digest in seals.items()
    ):
        raise NotificationError("notification facts evidence seals invalid")
    subject = facts["subject"]
    event_kind = facts["event_kind"]
    if event_kind == KIMI_RELEASE:
        _exact(subject, {"release_status", "terminal_evidence_seal_sha256"},
               "Kimi release facts")
        if subject.get("release_status") != "SAFE_RELEASE_VERIFIED":
            raise NotificationError("Kimi release facts are not safely verified")
    elif event_kind == GLM_ADMISSION:
        _exact(subject, {
            "repo", "revision", "admission_status", "terminal_evidence_seal_sha256"
        }, "GLM admission facts")
        if subject.get("repo") != OFFICIAL_REPO \
                or not isinstance(subject.get("revision"), str) \
                or _HEX40_RE.fullmatch(subject["revision"]) is None \
                or subject.get("admission_status") != "PASS":
            raise NotificationError("GLM admission facts are not exact PASS facts")
    elif event_kind == XET_AUTOTUNE_RESULT:
        _exact(subject, {"result_status", "terminal_evidence_seal_sha256"},
               "Xet autotune facts")
        if subject.get("result_status") != "PASS":
            raise NotificationError("Xet autotune facts are not PASS")
    elif event_kind in {
        SOURCE_STREAM_STARTED, SOURCE_WINDOW_VERIFIED,
        SOURCE_WINDOW_COMPLETED, SOURCE_WINDOW_EVICTED,
    }:
        _exact(subject, {
            "window_id", "schedule_index", "phase", "window_record_seal_sha256",
            "source_shards_sha256", "tensor_set_sha256", "organs", "layers",
            "download_start", "download_end",
        }, "window milestone facts")
        if not isinstance(subject.get("window_id"), str) or not subject["window_id"]:
            raise NotificationError("window milestone identity invalid")
        _uint(subject.get("schedule_index"), "window milestone schedule_index")
        if not isinstance(subject.get("phase"), str) or not subject["phase"]:
            raise NotificationError("window milestone phase invalid")
        for key in (
            "window_record_seal_sha256", "source_shards_sha256", "tensor_set_sha256"
        ):
            if not _is_sha256(subject.get(key)):
                raise NotificationError(f"window milestone {key} invalid")
        if not isinstance(subject.get("organs"), list) \
                or subject["organs"] != sorted(set(subject["organs"])) \
                or any(not isinstance(item, str) or not item for item in subject["organs"]):
            raise NotificationError("window milestone organs invalid")
        if not isinstance(subject.get("layers"), list) \
                or subject["layers"] != sorted(set(subject["layers"])):
            raise NotificationError("window milestone layers invalid")
        for layer in subject["layers"]:
            _uint(layer, "window milestone layer")
        started = subject.get("download_start")
        ended = subject.get("download_end")
        if started is not None:
            started_dt = _parse_utc(started, "window download_start")
        else:
            started_dt = None
        if ended is not None:
            ended_dt = _parse_utc(ended, "window download_end")
            if started_dt is None or ended_dt < started_dt:
                raise NotificationError("window download times are inconsistent")
        allowed_phases = {
            SOURCE_STREAM_STARTED: {"FETCHING", "FETCHED"},
            SOURCE_WINDOW_VERIFIED: {
                "VERIFIED", "TEACHER_CAPTURED", "CANDIDATES_FIT", "CANDIDATES_PACKED",
                "FORWARD_COMPLETE", "SEALED", "EVICTED",
            },
            SOURCE_WINDOW_COMPLETED: {"SEALED", "EVICTED"},
            SOURCE_WINDOW_EVICTED: {"EVICTED"},
        }[event_kind]
        if subject["phase"] not in allowed_phases:
            raise NotificationError("window milestone durable phase is premature")
        if event_kind == SOURCE_STREAM_STARTED and started_dt is None:
            raise NotificationError("source stream milestone lacks download_start")
        if seals.get("window_record") != subject["window_record_seal_sha256"]:
            raise NotificationError("window milestone seal is absent from grounded evidence")
    elif event_kind == SOURCE_COVERAGE_5_PERCENT:
        _exact(subject, {
            "threshold_percent", "verified_logical_bytes_at_prepare",
            "total_logical_bytes", "verified_paths_sha256",
        }, "coverage milestone facts")
        threshold = _uint(subject.get("threshold_percent"), "coverage threshold")
        verified = _uint(
            subject.get("verified_logical_bytes_at_prepare"), "coverage verified bytes"
        )
        total = _uint(subject.get("total_logical_bytes"), "coverage total bytes")
        if threshold not in range(5, 101, 5) or total == 0 \
                or verified > total or verified * 100 < total * threshold \
                or not _is_sha256(subject.get("verified_paths_sha256")):
            raise NotificationError("coverage milestone facts are premature or invalid")
    elif event_kind == FINAL_RESULT:
        _exact(subject, {
            "outcome", "half_bit_classification", "final_report_seal_sha256",
            "result_status", "terminal_evidence_seal_sha256",
        }, "final result facts")
        if subject.get("result_status") != "PASS" \
                or subject.get("outcome") not in _PRIMARY_OUTCOMES \
                or subject.get("half_bit_classification") not in _HALF_BIT_CLASSIFICATIONS \
                or not _is_sha256(subject.get("final_report_seal_sha256")):
            raise NotificationError("final result facts are not an authenticated PASS")
        if seals.get("terminal_outcome") != subject["final_report_seal_sha256"]:
            raise NotificationError("final result report seal is absent from grounded evidence")
    terminal_seal = subject.get("terminal_evidence_seal_sha256")
    if terminal_seal is not None and (
        not _is_sha256(terminal_seal) or seals.get("terminal_evidence") != terminal_seal
    ):
        raise NotificationError("milestone terminal evidence is absent from grounded seals")
    return facts


def _render_message(
    event_kind: str,
    dedupe_key: str,
    facts: Mapping[str, Any],
    status: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> str:
    coverage = status["source_coverage"]
    counts = status["counts"]
    network = status["network"]
    current = status["current"]
    current_text = "none" if current is None else (
        f"{current['window_id']}:{current['phase']}:"
        f"layers={','.join(map(str, current['layers'])) or 'none'}:"
        f"organs={','.join(current['organs']) or 'none'}"
    )
    message = "\n".join((
        f"GLM-5.2 milestone={event_kind}",
        f"state={status['state']} dedupe={dedupe_key}",
        f"source_coverage={coverage['percent']}% "
        f"({coverage['verified_logical_bytes']}/{coverage['total_logical_bytes']})",
        f"shards fetched={counts['shards_fetched']} verified={counts['shards_verified']} "
        f"evicted={counts['shards_evicted']} total={counts['shards_total']}",
        f"windows verified={counts['windows_verified']} completed={counts['windows_completed']} "
        f"evicted={counts['windows_evicted']} total={counts['windows_total']}",
        f"tensors terminal={counts['tensors_terminal']} total={counts['tensors_total']}",
        f"network_bytes={network['bytes']} "
        f"throughput_bytes_per_second={network['throughput_bytes_per_second']} "
        f"eta_seconds={network['eta_seconds'] if network['eta_seconds'] is not None else 'unknown'}",
        f"current={current_text}",
        f"candidate_rates={','.join(status['candidate_rates']) or 'none'} "
        f"best_metrics={json.dumps(status['best_metrics'], sort_keys=True, separators=(',', ':'))}",
        f"resources={json.dumps(status['resources'], sort_keys=True, separators=(',', ':'))}",
        f"process=pid:{status['process']['pid']},lease_held:true,"
        f"lease_owner:{status['process']['lease_owner']}",
        f"evidence_seals={json.dumps(status['evidence_seals'], sort_keys=True, separators=(',', ':'))}",
        f"facts_sha256={_sha256(facts)} controller_anchor={anchor['anchor_sha256']}",
    ))
    if len(message) > MAX_MESSAGE_CHARS:
        raise NotificationError("rendered notification exceeds Telegram's safe message limit")
    return message


def _make_intent(
    controller: Controller,
    event_kind: str,
    snapshot: Mapping[str, Any],
    producer: NotificationProducerSigner,
    *,
    coverage_threshold: int | None = None,
    prepared_at: str | None = None,
) -> dict[str, Any]:
    snapshot = _validate_snapshot(snapshot, controller)
    status = _canonical_status(snapshot)
    facts = _derive_event_facts(
        controller, event_kind, snapshot, coverage_threshold=coverage_threshold
    )
    anchor = _validate_controller_anchor(
        snapshot["controller_anchor"],
        identity={
            "campaign_id": controller.campaign_id,
            "source_revision": controller.source_revision,
            "controller_epoch": controller.controller_epoch,
            "expected_contract_sha256": controller.expected_contract_sha256,
        },
        expected_state=status["state"],
    )
    subject_identity = {
        "schema": "hawking.glm52.notification_event_identity.v2",
        "campaign_id": controller.campaign_id,
        "source_revision": controller.source_revision,
        "controller_epoch": controller.controller_epoch,
        "expected_contract_sha256": controller.expected_contract_sha256,
        "event_kind": event_kind,
        "subject": facts["subject"],
    }
    event_identity_sha256 = _sha256(subject_identity)
    dedupe_key = _sha256({
        "schema": "hawking.glm52.notification_dedupe.v2",
        "event_identity_sha256": event_identity_sha256,
    })
    timestamp = prepared_at or snapshot["captured_at"]
    if _parse_utc(timestamp, "intent prepared_at") < _parse_utc(
        snapshot["captured_at"], "snapshot captured_at"
    ):
        raise NotificationError("intent prepared_at predates its controller snapshot")
    rendered = _render_message(event_kind, dedupe_key, facts, status, anchor)
    body = {
        "schema": INTENT_SCHEMA,
        "campaign_id": controller.campaign_id,
        "source_revision": controller.source_revision,
        "controller_epoch": controller.controller_epoch,
        "expected_contract_sha256": controller.expected_contract_sha256,
        "event_kind": event_kind,
        "event_identity_sha256": event_identity_sha256,
        "dedupe_key": dedupe_key,
        "facts": facts,
        "canonical_status": status,
        "controller_anchor": anchor,
        "controller_snapshot_sha256": snapshot["seal_sha256"],
        "rendered_message": rendered,
        "rendered_message_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "prepared_at": timestamp,
        "producer_key_id": producer.key_id,
    }
    return seal({**body, "producer_signature": producer._sign(INTENT_AUTH_SCHEMA, body)})


def validate_notification_intent(
    value: Any,
    producer_verifier: NotificationVerifier,
    *,
    expected_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if producer_verifier.role != "producer":
        raise NotificationError("intent validation requires producer verifier")
    intent = _object(copy.deepcopy(value), "notification intent")
    required = {
        "schema", "campaign_id", "source_revision", "controller_epoch",
        "expected_contract_sha256", "event_kind", "event_identity_sha256", "dedupe_key",
        "facts", "canonical_status", "controller_anchor", "controller_snapshot_sha256",
        "rendered_message", "rendered_message_sha256", "prepared_at", "producer_key_id",
        "producer_signature", "seal_sha256",
    }
    _exact(intent, required, "notification intent")
    try:
        verify_sealed(intent, label="notification intent")
    except Glm52Error as exc:
        raise NotificationError(str(exc)) from exc
    if intent.get("schema") != INTENT_SCHEMA or intent.get("event_kind") not in _SUPPORTED_STATES:
        raise NotificationError("notification intent schema/kind invalid")
    if expected_identity is not None:
        for key, expected in expected_identity.items():
            if intent.get(key) != expected:
                raise NotificationError(f"notification intent {key} differs from controller")
    if intent.get("producer_key_id") != producer_verifier.key_id:
        raise NotificationError("notification intent producer key differs")
    facts = validate_event_facts(intent.get("facts"), expected_event_kind=intent["event_kind"])
    status = validate_canonical_status(intent.get("canonical_status"))
    if facts["controller_state"] != status["state"] \
            or facts["controller_snapshot_sha256"] != status["controller_snapshot_sha256"] \
            or intent.get("controller_snapshot_sha256") != status["controller_snapshot_sha256"]:
        raise NotificationError("notification intent snapshot/status/facts binding differs")
    _validate_controller_anchor(
        intent.get("controller_anchor"),
        identity={
            "campaign_id": intent["campaign_id"],
            "source_revision": intent["source_revision"],
            "controller_epoch": intent["controller_epoch"],
            "expected_contract_sha256": intent["expected_contract_sha256"],
        },
        expected_state=status["state"],
    )
    subject_identity = {
        "schema": "hawking.glm52.notification_event_identity.v2",
        "campaign_id": intent["campaign_id"],
        "source_revision": intent["source_revision"],
        "controller_epoch": intent["controller_epoch"],
        "expected_contract_sha256": intent["expected_contract_sha256"],
        "event_kind": intent["event_kind"],
        "subject": facts["subject"],
    }
    if intent.get("event_identity_sha256") != _sha256(subject_identity) \
            or intent.get("dedupe_key") != _sha256({
                "schema": "hawking.glm52.notification_dedupe.v2",
                "event_identity_sha256": intent.get("event_identity_sha256"),
            }):
        raise NotificationError("notification event identity/dedupe mismatch")
    rendered = intent.get("rendered_message")
    if not isinstance(rendered, str) or not rendered or len(rendered) > MAX_MESSAGE_CHARS \
            or intent.get("rendered_message_sha256") != hashlib.sha256(
                rendered.encode("utf-8")
            ).hexdigest():
        raise NotificationError("notification rendered message hash invalid")
    expected_rendered = _render_message(
        intent["event_kind"],
        intent["dedupe_key"],
        facts,
        status,
        intent["controller_anchor"],
    )
    if rendered != expected_rendered:
        raise NotificationError("notification rendered message differs from grounded fields")
    _parse_utc(intent.get("prepared_at"), "intent prepared_at")
    body = {key: item for key, item in intent.items() if key not in {
        "producer_signature", "seal_sha256"
    }}
    producer_verifier.verify(INTENT_AUTH_SCHEMA, body, intent.get("producer_signature"))
    return intent


def _entry_chain_body(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item) for key, item in entry.items()
        if key not in {"chain_sha256", "producer_signature", "seal_sha256"}
    }


def _make_journal_entry(
    producer: NotificationProducerSigner,
    *,
    seq: int,
    kind: str,
    recorded_at: str,
    intent_sha256: str,
    dedupe_key: str,
    attempt: int,
    payload: Mapping[str, Any],
    previous_chain_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema": ENTRY_SCHEMA,
        "seq": seq,
        "kind": kind,
        "recorded_at": recorded_at,
        "intent_sha256": intent_sha256,
        "dedupe_key": dedupe_key,
        "attempt": attempt,
        "payload": copy.deepcopy(dict(payload)),
        "previous_chain_sha256": previous_chain_sha256,
        "producer_key_id": producer.key_id,
    }
    chained = {**body, "chain_sha256": _sha256(body)}
    return seal({
        **chained,
        "producer_signature": producer._sign(ENTRY_AUTH_SCHEMA, chained),
    })


def validate_journal_entry_signature(
    value: Any, producer_verifier: NotificationVerifier
) -> dict[str, Any]:
    if producer_verifier.role != "producer":
        raise NotificationError("journal entry validation requires producer verifier")
    entry = _object(copy.deepcopy(value), "notification journal entry")
    _exact(entry, {
        "schema", "seq", "kind", "recorded_at", "intent_sha256", "dedupe_key",
        "attempt", "payload", "previous_chain_sha256", "producer_key_id",
        "chain_sha256", "producer_signature", "seal_sha256",
    }, "notification journal entry")
    try:
        verify_sealed(entry, label="notification journal entry")
    except Glm52Error as exc:
        raise NotificationError(str(exc)) from exc
    if entry.get("schema") != ENTRY_SCHEMA or entry.get("kind") not in LIFECYCLE_KINDS:
        raise NotificationError("notification journal entry schema/kind invalid")
    _uint(entry.get("seq"), "journal seq")
    _uint(entry.get("attempt"), "journal attempt")
    _parse_utc(entry.get("recorded_at"), "journal recorded_at")
    if not _is_sha256(entry.get("intent_sha256")) \
            or not _is_sha256(entry.get("dedupe_key")) \
            or not _is_sha256(entry.get("previous_chain_sha256")) \
            or not isinstance(entry.get("payload"), dict):
        raise NotificationError("notification journal entry hashes/payload invalid")
    if entry.get("producer_key_id") != producer_verifier.key_id:
        raise NotificationError("notification journal entry producer key differs")
    chain_body = _entry_chain_body(entry)
    if entry.get("chain_sha256") != _sha256(chain_body):
        raise NotificationError("notification journal hash chain mismatch")
    signed = {key: item for key, item in entry.items() if key not in {
        "producer_signature", "seal_sha256"
    }}
    producer_verifier.verify(ENTRY_AUTH_SCHEMA, signed, entry.get("producer_signature"))
    return entry


def validate_bot_delivery_receipt(
    value: Any,
    intent_value: Mapping[str, Any],
    send_started_entry: Mapping[str, Any],
    *,
    producer_verifier: NotificationVerifier,
    receipt_verifier: NotificationVerifier,
    expected_chat_identity_digest: str,
    expected_bot_identity_digest: str,
) -> dict[str, Any]:
    if receipt_verifier.role != "receipt":
        raise NotificationError("delivery receipt requires the receipt verifier")
    intent = validate_notification_intent(intent_value, producer_verifier)
    started = validate_journal_entry_signature(send_started_entry, producer_verifier)
    receipt = _object(copy.deepcopy(value), "notification delivery receipt")
    _exact(receipt, {
        "schema", "status", "intent_sha256", "dedupe_key", "attempt",
        "send_started_seq", "send_started_chain_sha256", "send_started_signature",
        "message_id", "chat_identity_digest", "bot_identity_digest",
        "rendered_message_sha256", "bot_api_response_sha256", "http_status",
        "delivered_at", "receipt_key_id", "receipt_signature", "seal_sha256",
    }, "notification delivery receipt")
    try:
        verify_sealed(receipt, label="notification delivery receipt")
    except Glm52Error as exc:
        raise NotificationError(str(exc)) from exc
    if receipt.get("schema") != BOT_RECEIPT_SCHEMA or receipt.get("status") != "DELIVERED":
        raise NotificationError("notification delivery receipt schema/status invalid")
    expected = {
        "intent_sha256": intent["seal_sha256"],
        "dedupe_key": intent["dedupe_key"],
        "attempt": started["attempt"],
        "send_started_seq": started["seq"],
        "send_started_chain_sha256": started["chain_sha256"],
        "send_started_signature": started["producer_signature"],
        "chat_identity_digest": expected_chat_identity_digest,
        "bot_identity_digest": expected_bot_identity_digest,
        "rendered_message_sha256": intent["rendered_message_sha256"],
        "receipt_key_id": receipt_verifier.key_id,
    }
    if any(receipt.get(key) != expected_value for key, expected_value in expected.items()):
        raise NotificationError("delivery receipt differs from its exact send attempt/identity")
    if _uint(receipt.get("message_id"), "receipt message_id") == 0 \
            or not _is_sha256(receipt.get("bot_api_response_sha256")):
        raise NotificationError("delivery receipt message/response identity invalid")
    status = receipt.get("http_status")
    if isinstance(status, bool) or not isinstance(status, int) or not 200 <= status < 300:
        raise NotificationError("delivery receipt HTTP status invalid")
    delivered = _parse_utc(receipt.get("delivered_at"), "receipt delivered_at")
    if not _parse_utc(intent["prepared_at"], "intent prepared_at") \
            <= _parse_utc(started["recorded_at"], "SEND_STARTED recorded_at") <= delivered:
        raise NotificationError("receipt timestamps do not follow prepare/send/deliver order")
    body = {key: item for key, item in receipt.items() if key not in {
        "receipt_signature", "seal_sha256"
    }}
    receipt_verifier.verify(
        BOT_RECEIPT_AUTH_SCHEMA, body, receipt.get("receipt_signature")
    )
    return receipt


def validate_reconciliation_challenge(value: Any) -> dict[str, Any]:
    challenge = _object(copy.deepcopy(value), "reconciliation challenge")
    _exact(challenge, {
        "schema", "intent_sha256", "dedupe_key", "ambiguous_seq",
        "ambiguous_chain_sha256", "send_started_seq", "send_started_chain_sha256",
        "authorized_attempt", "journal_binding_sha256", "nonce", "issued_at",
        "seal_sha256",
    }, "reconciliation challenge")
    try:
        verify_sealed(challenge, label="reconciliation challenge")
    except Glm52Error as exc:
        raise NotificationError(str(exc)) from exc
    if challenge.get("schema") != RECONCILIATION_CHALLENGE_SCHEMA:
        raise NotificationError("reconciliation challenge schema invalid")
    for key in (
        "intent_sha256", "dedupe_key", "ambiguous_chain_sha256",
        "send_started_chain_sha256", "journal_binding_sha256",
    ):
        if not _is_sha256(challenge.get(key)):
            raise NotificationError(f"reconciliation challenge {key} invalid")
    for key in ("ambiguous_seq", "send_started_seq", "authorized_attempt"):
        _uint(challenge.get(key), f"reconciliation challenge {key}")
    if challenge["authorized_attempt"] == 0 \
            or not isinstance(challenge.get("nonce"), str) \
            or re.fullmatch(r"^[0-9a-f]{64}$", challenge["nonce"]) is None:
        raise NotificationError("reconciliation challenge attempt/nonce invalid")
    _parse_utc(challenge.get("issued_at"), "challenge issued_at")
    return challenge


def validate_reconciliation_authorization(
    value: Any,
    challenge_value: Mapping[str, Any],
    *,
    verifier: NotificationVerifier,
) -> dict[str, Any]:
    if verifier.role != "reconciliation":
        raise NotificationError("reconciliation requires its public verifier")
    challenge = validate_reconciliation_challenge(challenge_value)
    authorization = _object(copy.deepcopy(value), "reconciliation authorization")
    _exact(authorization, {
        "schema", "status", "authorization_id", "challenge_sha256", "intent_sha256",
        "dedupe_key", "ambiguous_seq", "ambiguous_chain_sha256", "send_started_seq",
        "send_started_chain_sha256", "authorized_attempt", "journal_binding_sha256",
        "reason", "authorized_at", "reconciliation_key_id",
        "reconciliation_signature", "seal_sha256",
    }, "reconciliation authorization")
    try:
        verify_sealed(authorization, label="reconciliation authorization")
    except Glm52Error as exc:
        raise NotificationError(str(exc)) from exc
    if authorization.get("schema") != RECONCILIATION_SCHEMA \
            or authorization.get("status") != "REPLAY_AUTHORIZED":
        raise NotificationError("reconciliation authorization schema/status invalid")
    mapping = {
        "challenge_sha256": "seal_sha256",
        "intent_sha256": "intent_sha256",
        "dedupe_key": "dedupe_key",
        "ambiguous_seq": "ambiguous_seq",
        "ambiguous_chain_sha256": "ambiguous_chain_sha256",
        "send_started_seq": "send_started_seq",
        "send_started_chain_sha256": "send_started_chain_sha256",
        "authorized_attempt": "authorized_attempt",
        "journal_binding_sha256": "journal_binding_sha256",
    }
    if any(authorization.get(auth_key) != challenge.get(challenge_key)
           for auth_key, challenge_key in mapping.items()):
        raise NotificationError("reconciliation authorization differs from its exact challenge")
    _safe_text(authorization.get("authorization_id"), "authorization_id")
    _safe_text(authorization.get("reason"), "reconciliation reason")
    if authorization.get("reconciliation_key_id") != verifier.key_id:
        raise NotificationError("reconciliation authorization key differs")
    if _parse_utc(authorization.get("authorized_at"), "authorized_at") < _parse_utc(
        challenge["issued_at"], "challenge issued_at"
    ):
        raise NotificationError("reconciliation authorization predates challenge")
    body = {key: item for key, item in authorization.items() if key not in {
        "reconciliation_signature", "seal_sha256"
    }}
    verifier.verify(
        RECONCILIATION_AUTH_SCHEMA,
        body,
        authorization.get("reconciliation_signature"),
    )
    return authorization


@dataclass(frozen=True)
class ReplaySnapshot:
    entries: tuple[dict[str, Any], ...]
    intents: Mapping[str, dict[str, Any]]
    lifecycle: Mapping[str, str]
    attempts: Mapping[str, int]
    latest_send_started: Mapping[str, dict[str, Any]]
    latest_ambiguous: Mapping[str, dict[str, Any]]
    committed_event_identities: frozenset[str]
    prepared_coverage_thresholds: frozenset[int]
    latest_delivered_status: dict[str, Any] | None
    head: dict[str, Any]
    binding: dict[str, Any]


_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}


def _thread_lock(path: Path) -> threading.RLock:
    key = os.fspath(path)
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


def _file_identity(info: os.stat_result) -> dict[str, int]:
    return {"device": int(info.st_dev), "inode": int(info.st_ino)}


def _same_identity(info: os.stat_result, identity: Mapping[str, int]) -> bool:
    return int(info.st_dev) == identity.get("device") \
        and int(info.st_ino) == identity.get("inode")


def _validate_secure_regular(info: os.stat_result, label: str) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise NotificationError(f"{label} must be a regular file")
    if info.st_nlink != 1:
        raise NotificationError(f"{label} must have exactly one hard link")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise NotificationError(f"{label} is not owned by this user")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise NotificationError(f"{label} permissions are not private")


def _normalized_absolute_path(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    if not os.path.isabs(raw):
        raise NotificationError("notification journal path must be absolute")
    if any(part in {".", ".."} for part in Path(raw).parts[1:]):
        raise NotificationError("notification journal path contains a traversal component")
    path = Path(os.path.normpath(raw))
    if not path.name or path.name in {".", ".."}:
        raise NotificationError("notification journal filename invalid")
    return path


def _open_directory_components(path: Path) -> int:
    if not path.is_absolute():
        raise NotificationError("secure directory path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    fd = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, flags | nofollow, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode):
            raise NotificationError("notification parent is not a directory")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise NotificationError("notification parent is not owned by this user")
        if stat.S_IMODE(info.st_mode) & 0o022:
            raise NotificationError("notification parent is group/world writable")
        return fd
    except BaseException:
        os.close(fd)
        raise


class NotificationJournal:
    """Controller-bound notification outbox and delivery lifecycle journal."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        controller: Controller,
        producer_signer: NotificationProducerSigner,
        receipt_verifier: NotificationVerifier,
        reconciliation_verifier: NotificationVerifier,
        expected_bot_identity_digest: str,
        clock: Callable[[], str] = utc_now,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.path = _normalized_absolute_path(path)
        self.head_path = self.path.with_name(self.path.name + ".head.json")
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        if not isinstance(controller, Controller):
            raise NotificationError("notification journal requires a live Controller")
        controller.lease.assert_held()
        if not isinstance(producer_signer, NotificationProducerSigner):
            raise NotificationError("notification journal requires its producer signer")
        if receipt_verifier.role != "receipt" \
                or reconciliation_verifier.role != "reconciliation":
            raise NotificationError("notification journal received a signer in the wrong role")
        key_ids = {
            producer_signer.key_id,
            receipt_verifier.key_id,
            reconciliation_verifier.key_id,
        }
        if len(key_ids) != 3:
            raise NotificationError("notification signing roles require distinct Ed25519 keys")
        if not _is_sha256(expected_bot_identity_digest):
            raise NotificationError("expected Telegram bot identity digest invalid")
        self.controller = controller
        self.producer_signer = producer_signer
        self.producer_verifier = producer_signer.verifier
        self.receipt_verifier = receipt_verifier
        self.reconciliation_verifier = reconciliation_verifier
        self.expected_chat_identity_digest = (
            controller.expected_contract["expected_chat_identity_digest"]
        )
        self.expected_bot_identity_digest = expected_bot_identity_digest
        self._clock = clock
        self._fault_injector = fault_injector
        self._mutex = _thread_lock(self.path)
        self._path_sha256 = hashlib.sha256(
            b"hawking.glm52.notification-journal-path.v1\0" + os.fsencode(self.path)
        ).hexdigest()
        self._pinned: dict[str, dict[str, int]] = {}
        try:
            with self._mutex:
                anchored = controller.notification_journal_binding()
                if anchored is None:
                    parent_fd = _open_directory_components(self.path.parent)
                    try:
                        existing = [
                            self._lstat_name(parent_fd, name)
                            for name in (
                                self.path.name, self.head_path.name, self.lock_path.name
                            )
                        ]
                    finally:
                        os.close(parent_fd)
                    if all(item is None for item in existing):
                        self._create_genesis()
                    elif all(item is not None for item in existing):
                        self._recover_unanchored_genesis()
                    else:
                        raise NotificationError(
                            "unanchored notification storage is partial or pre-existing"
                        )
                else:
                    self._adopt_anchored_identity(anchored)
                    with self._locked_files() as (parent_fd, journal_fd):
                        self._load_locked(parent_fd, journal_fd)
        except StateError as exc:
            raise NotificationError(str(exc)) from exc

    def __repr__(self) -> str:
        return (
            f"NotificationJournal(path={str(self.path)!r}, "
            f"producer_key_id={self.producer_verifier.key_id!r})"
        )

    @property
    def identity(self) -> dict[str, str]:
        return {
            "campaign_id": self.controller.campaign_id,
            "source_revision": self.controller.source_revision,
            "controller_epoch": self.controller.controller_epoch,
            "expected_contract_sha256": self.controller.expected_contract_sha256,
        }

    def _fault(self, stage: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(stage)

    def _now_after(self, *floors: str) -> str:
        value = self._clock()
        parsed = _parse_utc(value, "journal clock")
        for floor in floors:
            if parsed < _parse_utc(floor, "journal time floor"):
                raise NotificationError("journal clock regressed behind durable UTC time")
        return value

    @staticmethod
    def _lstat_name(parent_fd: int, name: str) -> os.stat_result | None:
        try:
            return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None

    def _assert_parent_identity(self) -> None:
        parent_fd = _open_directory_components(self.path.parent)
        try:
            if not _same_identity(os.fstat(parent_fd), self._pinned["parent_identity"]):
                raise NotificationError("notification parent identity changed")
        finally:
            os.close(parent_fd)

    def _adopt_anchored_identity(self, binding: Mapping[str, Any]) -> None:
        immutable = {
            "campaign_id": self.controller.campaign_id,
            "source_revision": self.controller.source_revision,
            "controller_epoch": self.controller.controller_epoch,
            "expected_contract_sha256": self.controller.expected_contract_sha256,
            "producer_public_key": self.producer_verifier.public_key_hex,
            "producer_key_id": self.producer_verifier.key_id,
            "receipt_key_id": self.receipt_verifier.key_id,
            "reconciliation_key_id": self.reconciliation_verifier.key_id,
            "expected_chat_identity_digest": self.expected_chat_identity_digest,
            "expected_bot_identity_digest": self.expected_bot_identity_digest,
            "path_sha256": self._path_sha256,
        }
        if any(binding.get(key) != expected for key, expected in immutable.items()):
            raise NotificationError("notification journal configuration differs from controller anchor")
        self._pinned = {
            "parent_identity": copy.deepcopy(binding["parent_identity"]),
            "journal_identity": copy.deepcopy(binding["journal_identity"]),
            "lock_identity": copy.deepcopy(binding["lock_identity"]),
        }

    def _create_genesis(self) -> None:
        parent_fd = _open_directory_components(self.path.parent)
        lock_fd: int | None = None
        journal_fd: int | None = None
        try:
            for name in (self.path.name, self.head_path.name, self.lock_path.name):
                if self._lstat_name(parent_fd, name) is not None:
                    raise NotificationError(
                        "unanchored notification storage is partial or pre-existing"
                    )
            flags = (
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NONBLOCK
                | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            )
            lock_fd = os.open(self.lock_path.name, flags, 0o600, dir_fd=parent_fd)
            os.fchmod(lock_fd, 0o600)
            _validate_secure_regular(os.fstat(lock_fd), "notification lock")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            journal_fd = os.open(self.path.name, flags, 0o600, dir_fd=parent_fd)
            os.fchmod(journal_fd, 0o600)
            _validate_secure_regular(os.fstat(journal_fd), "notification journal")
            os.fsync(journal_fd)
            os.fsync(parent_fd)
            self._pinned = {
                "parent_identity": _file_identity(os.fstat(parent_fd)),
                "journal_identity": _file_identity(os.fstat(journal_fd)),
                "lock_identity": _file_identity(os.fstat(lock_fd)),
            }
            observed_at = self._now_after()
            head = self._make_head((), None, observed_at=observed_at)
            self._write_head(parent_fd, head, expected_head_identity=None)
            binding = self._binding_from_head(head)
            self._fault("after_head_fsync_before_controller_anchor")
            self.controller.anchor_notification_journal_head(binding)
            self._fault("after_controller_anchor")
        except FileExistsError:
            raise NotificationError("notification storage creation raced another owner") from None
        except OSError as exc:
            raise NotificationError(f"cannot create secure notification storage: {exc}") from None
        finally:
            if journal_fd is not None:
                os.close(journal_fd)
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
            os.close(parent_fd)

    def _recover_unanchored_genesis(self) -> None:
        """Recover only a fully signed genesis whose controller anchor write crashed."""
        parent_fd = _open_directory_components(self.path.parent)
        try:
            flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) \
                | getattr(os, "O_CLOEXEC", 0)
            identities: dict[str, dict[str, int]] = {
                "parent_identity": _file_identity(os.fstat(parent_fd))
            }
            for name, target in (
                (self.path.name, "journal_identity"),
                (self.lock_path.name, "lock_identity"),
            ):
                fd = os.open(name, flags, dir_fd=parent_fd)
                try:
                    info = os.fstat(fd)
                    _validate_secure_regular(info, f"notification {target}")
                    identities[target] = _file_identity(info)
                finally:
                    os.close(fd)
            self._pinned = identities
        except (FileNotFoundError, OSError) as exc:
            raise NotificationError(f"cannot recover signed notification genesis: {exc}") from None
        finally:
            os.close(parent_fd)
        with self._locked_files() as (locked_parent, journal_fd):
            snapshot = self._load_locked(locked_parent, journal_fd)
            if snapshot.binding["entry_count"] != 0:
                raise NotificationError("unanchored notification history is not exact genesis")

    @contextmanager
    def _locked_files(self) -> Iterator[tuple[int, int]]:
        parent_fd: int | None = None
        lock_fd: int | None = None
        journal_fd: int | None = None
        try:
            parent_fd = _open_directory_components(self.path.parent)
            if not _same_identity(os.fstat(parent_fd), self._pinned["parent_identity"]):
                raise NotificationError("notification parent identity changed")
            flags = (
                os.O_RDWR | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            lock_fd = os.open(self.lock_path.name, flags, dir_fd=parent_fd)
            lock_info = os.fstat(lock_fd)
            _validate_secure_regular(lock_info, "notification lock")
            if not _same_identity(lock_info, self._pinned["lock_identity"]):
                raise NotificationError("notification lock identity changed")
            name_lock = self._lstat_name(parent_fd, self.lock_path.name)
            if name_lock is None or not _same_identity(name_lock, self._pinned["lock_identity"]):
                raise NotificationError("notification lock name changed")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            journal_fd = os.open(self.path.name, flags | os.O_APPEND, dir_fd=parent_fd)
            journal_info = os.fstat(journal_fd)
            _validate_secure_regular(journal_info, "notification journal")
            if not _same_identity(journal_info, self._pinned["journal_identity"]):
                raise NotificationError("notification journal identity changed")
            name_journal = self._lstat_name(parent_fd, self.path.name)
            if name_journal is None or not _same_identity(
                name_journal, self._pinned["journal_identity"]
            ):
                raise NotificationError("notification journal name changed")
            yield parent_fd, journal_fd
            for name, identity, label in (
                (self.path.name, self._pinned["journal_identity"], "journal"),
                (self.lock_path.name, self._pinned["lock_identity"], "lock"),
            ):
                named = self._lstat_name(parent_fd, name)
                if named is None or not _same_identity(named, identity):
                    raise NotificationError(f"notification {label} name changed during operation")
            self._assert_parent_identity()
        except FileNotFoundError:
            raise NotificationError("controller-anchored notification storage was deleted") from None
        except OSError as exc:
            raise NotificationError(f"cannot safely access notification storage: {exc}") from None
        finally:
            if journal_fd is not None:
                os.close(journal_fd)
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(lock_fd)
            if parent_fd is not None:
                os.close(parent_fd)

    @staticmethod
    def _read_exact_fd(fd: int, size: int) -> bytes:
        chunks: list[bytes] = []
        offset = 0
        while offset < size:
            chunk = os.pread(fd, min(1024 * 1024, size - offset), offset)
            if not chunk:
                raise NotificationError("notification file changed while reading")
            chunks.append(chunk)
            offset += len(chunk)
        return b"".join(chunks)

    def _read_entries(self, journal_fd: int) -> list[dict[str, Any]]:
        before = os.fstat(journal_fd)
        _validate_secure_regular(before, "notification journal")
        if before.st_size > MAX_JOURNAL_BYTES:
            raise NotificationError("notification journal exceeded its safe size")
        raw = self._read_exact_fd(journal_fd, before.st_size)
        after = os.fstat(journal_fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise NotificationError("notification journal changed while reading")
        if raw and not raw.endswith(b"\n"):
            raise NotificationError("notification journal has a torn/truncated tail")
        entries: list[dict[str, Any]] = []
        for index, line in enumerate(raw.splitlines(), 1):
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise NotificationError(f"notification journal line {index} is invalid JSON") from None
            if not isinstance(value, dict) or canonical(value) != line:
                raise NotificationError(
                    f"notification journal line {index} is not exact canonical JSON"
                )
            entries.append(validate_journal_entry_signature(value, self.producer_verifier))
        return entries

    def _read_head(self, parent_fd: int) -> tuple[dict[str, Any], os.stat_result]:
        flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) \
            | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(self.head_path.name, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            raise NotificationError("controller-anchored notification journal lacks its head") from None
        try:
            before = os.fstat(fd)
            _validate_secure_regular(before, "notification journal head")
            if before.st_size > 1024 * 1024:
                raise NotificationError("notification journal head exceeded its safe size")
            raw = self._read_exact_fd(fd, before.st_size)
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
            ):
                raise NotificationError("notification journal head changed while reading")
            named = self._lstat_name(parent_fd, self.head_path.name)
            if named is None or not _same_identity(named, _file_identity(after)):
                raise NotificationError("notification journal head name changed while reading")
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise NotificationError("notification journal head is invalid JSON") from None
            if not isinstance(value, dict) or canonical(value) + b"\n" != raw:
                raise NotificationError("notification journal head is not exact canonical JSON")
            return self._validate_head(value), after
        finally:
            os.close(fd)

    def _head_common(self) -> dict[str, Any]:
        return {
            **self.identity,
            "producer_public_key": self.producer_verifier.public_key_hex,
            "producer_key_id": self.producer_verifier.key_id,
            "receipt_key_id": self.receipt_verifier.key_id,
            "reconciliation_key_id": self.reconciliation_verifier.key_id,
            "expected_chat_identity_digest": self.expected_chat_identity_digest,
            "expected_bot_identity_digest": self.expected_bot_identity_digest,
            "path_sha256": self._path_sha256,
            "parent_identity": copy.deepcopy(self._pinned["parent_identity"]),
            "journal_identity": copy.deepcopy(self._pinned["journal_identity"]),
            "lock_identity": copy.deepcopy(self._pinned["lock_identity"]),
        }

    def _make_head(
        self,
        entries: Sequence[Mapping[str, Any]],
        previous_binding_sha256: str | None,
        *,
        observed_at: str,
    ) -> dict[str, Any]:
        _parse_utc(observed_at, "notification head observed_at")
        last = entries[-1] if entries else None
        body = {
            "schema": HEAD_SCHEMA,
            **self._head_common(),
            "entry_count": len(entries),
            "head_chain_sha256": last["chain_sha256"] if last else GENESIS_HASH,
            "head_entry_signature": last["producer_signature"] if last else None,
            "previous_binding_sha256": previous_binding_sha256,
            "observed_at": observed_at,
        }
        return seal({**body, "producer_signature": self.producer_signer._sign(HEAD_AUTH_SCHEMA, body)})

    def _validate_head(self, value: Any) -> dict[str, Any]:
        head = _object(copy.deepcopy(value), "notification journal head")
        required = {
            "schema", *self._head_common().keys(), "entry_count", "head_chain_sha256",
            "head_entry_signature", "previous_binding_sha256", "observed_at",
            "producer_signature", "seal_sha256",
        }
        _exact(head, set(required), "notification journal head")
        try:
            verify_sealed(head, label="notification journal head")
        except Glm52Error as exc:
            raise NotificationError(str(exc)) from exc
        if head.get("schema") != HEAD_SCHEMA:
            raise NotificationError("notification journal head schema invalid")
        common = self._head_common()
        if any(head.get(key) != expected for key, expected in common.items()):
            raise NotificationError("notification journal head identity/configuration differs")
        count = _uint(head.get("entry_count"), "notification head entry_count")
        if not _is_sha256(head.get("head_chain_sha256")):
            raise NotificationError("notification head chain hash invalid")
        if count == 0:
            if head["head_chain_sha256"] != GENESIS_HASH \
                    or head.get("head_entry_signature") is not None \
                    or head.get("previous_binding_sha256") is not None:
                raise NotificationError("notification genesis head is inconsistent")
        elif not isinstance(head.get("head_entry_signature"), str) \
                or _SIG_RE.fullmatch(head["head_entry_signature"]) is None \
                or not _is_sha256(head.get("previous_binding_sha256")):
            raise NotificationError("notification non-genesis head is inconsistent")
        _parse_utc(head.get("observed_at"), "notification head observed_at")
        body = {key: item for key, item in head.items() if key not in {
            "producer_signature", "seal_sha256"
        }}
        self.producer_verifier.verify(
            HEAD_AUTH_SCHEMA, body, head.get("producer_signature")
        )
        return head

    def _binding_from_head(self, head_value: Mapping[str, Any]) -> dict[str, Any]:
        head = self._validate_head(head_value)
        body = {
            "schema": NOTIFICATION_HEAD_BINDING_SCHEMA,
            **self._head_common(),
            "entry_count": head["entry_count"],
            "head_chain_sha256": head["head_chain_sha256"],
            "head_entry_signature": head["head_entry_signature"],
            "head_document_sha256": hashlib.sha256(canonical(head)).hexdigest(),
            "head_signature": head["producer_signature"],
            "previous_binding_sha256": head["previous_binding_sha256"],
            "observed_at": head["observed_at"],
        }
        return seal({
            **body,
            "binding_signature": self.producer_signer._sign(BINDING_AUTH_SCHEMA, body),
        })

    def _write_head(
        self,
        parent_fd: int,
        head_value: Mapping[str, Any],
        *,
        expected_head_identity: Mapping[str, int] | None,
    ) -> os.stat_result:
        head = self._validate_head(head_value)
        existing = self._lstat_name(parent_fd, self.head_path.name)
        if expected_head_identity is None:
            if existing is not None:
                raise NotificationError("notification genesis head path unexpectedly exists")
        elif existing is None or not _same_identity(existing, expected_head_identity):
            raise NotificationError("notification journal head changed before replacement")
        temporary = f".{self.head_path.name}.{os.getpid()}.{secrets.token_hex(16)}.tmp"
        flags = (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NONBLOCK
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        )
        fd: int | None = None
        raw = canonical(head) + b"\n"
        try:
            fd = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
            os.fchmod(fd, 0o600)
            _validate_secure_regular(os.fstat(fd), "temporary notification head")
            offset = 0
            while offset < len(raw):
                written = os.write(fd, raw[offset:])
                if written <= 0:
                    raise NotificationError("short write of notification journal head")
                offset += written
            os.fsync(fd)
            os.close(fd)
            fd = None
            os.rename(
                temporary,
                self.head_path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
            restored, info = self._read_head(parent_fd)
            if restored != head:
                raise NotificationError("notification head post-write content mismatch")
            return info
        finally:
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass

    def _controller_bindings(self) -> dict[int, dict[str, Any]]:
        bindings: dict[int, dict[str, Any]] = {}
        for event in self.controller.events.verified_events():
            if event.get("kind") != "NOTIFICATION_HEAD":
                continue
            payload = event.get("payload")
            binding = payload.get("notification_binding") if isinstance(payload, dict) else None
            if isinstance(binding, dict):
                bindings[binding["entry_count"]] = binding
        return bindings

    def _validate_intent_anchor_history(
        self,
        intent: Mapping[str, Any],
        *,
        controller_events: Sequence[Mapping[str, Any]],
        window_events: Sequence[Mapping[str, Any]],
        previous_anchor: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        anchor = _validate_controller_anchor(
            intent["controller_anchor"],
            identity=self.identity,
            expected_state=intent["canonical_status"]["state"],
        )
        checkpoint = anchor["checkpoint"]
        event_count = checkpoint["event_count"]
        window_count = checkpoint["window_event_count"]
        if event_count > len(controller_events) or window_count > len(window_events):
            raise NotificationError("notification anchor is ahead of verified controller logs")
        expected_event_head = (
            controller_events[event_count - 1]["chain_sha256"]
            if event_count else GENESIS_HASH
        )
        expected_window_head = (
            window_events[window_count - 1]["chain_sha256"]
            if window_count else GENESIS_HASH
        )
        if checkpoint["event_head_hash"] != expected_event_head \
                or checkpoint["window_event_head_hash"] != expected_window_head:
            raise NotificationError("notification anchor forks verified controller log prefixes")
        if event_count == 0:
            raise NotificationError("notification anchor cannot precede controller genesis")
        try:
            replayed_state = self.controller._replay_controller(
                controller_events[:event_count]
            )["state"]
        except StateError as exc:
            raise NotificationError(str(exc)) from exc
        if replayed_state != anchor["from_state"]:
            raise NotificationError("notification anchor state differs from its log prefix")
        if previous_anchor is not None:
            prior = previous_anchor["checkpoint"]
            for count_key, head_key in (
                ("event_count", "event_head_hash"),
                ("window_event_count", "window_event_head_hash"),
            ):
                if checkpoint[count_key] < prior[count_key]:
                    raise NotificationError("notification controller anchor regressed")
                if checkpoint[count_key] == prior[count_key] \
                        and checkpoint[head_key] != prior[head_key]:
                    raise NotificationError(
                        "notification controller anchor partially forked an unchanged log"
                    )
        return anchor

    @staticmethod
    def _validate_status_monotonic(
        previous: Mapping[str, Any] | None, current: Mapping[str, Any]
    ) -> None:
        if previous is None:
            return
        if current["source_coverage"]["verified_logical_bytes"] \
                < previous["source_coverage"]["verified_logical_bytes"]:
            raise NotificationError("notification verified source coverage regressed")
        for key in (
            "shards_fetched", "shards_verified", "windows_verified",
            "windows_completed", "windows_evicted", "tensors_terminal",
        ):
            if current["counts"][key] < previous["counts"][key]:
                raise NotificationError(f"notification durable count regressed: {key}")
        if current["network"]["bytes"] < previous["network"]["bytes"]:
            raise NotificationError("notification network byte accounting regressed")

    def _replay_entries(
        self,
        entries: Sequence[Mapping[str, Any]],
        head: Mapping[str, Any],
        binding: Mapping[str, Any],
    ) -> ReplaySnapshot:
        if head["entry_count"] != len(entries):
            raise NotificationError("notification journal/head entry counts differ")
        expected_head_chain = entries[-1]["chain_sha256"] if entries else GENESIS_HASH
        expected_head_signature = entries[-1]["producer_signature"] if entries else None
        if head["head_chain_sha256"] != expected_head_chain \
                or head["head_entry_signature"] != expected_head_signature:
            raise NotificationError("notification journal/head fork or clean-tail truncation")
        intents: dict[str, dict[str, Any]] = {}
        lifecycle: dict[str, str] = {}
        attempts: dict[str, int] = {}
        latest_started: dict[str, dict[str, Any]] = {}
        latest_ambiguous: dict[str, dict[str, Any]] = {}
        prepared_entries: dict[str, dict[str, Any]] = {}
        event_ids: set[str] = set()
        committed_ids: set[str] = set()
        coverage: set[int] = set()
        latest_status: dict[str, Any] | None = None
        delivery_response_hashes: set[str] = set()
        delivery_message_ids: set[tuple[str, int]] = set()
        previous_chain = GENESIS_HASH
        previous_time: datetime | None = None
        previous_anchor: dict[str, Any] | None = None
        previous_status: dict[str, Any] | None = None
        controller_bindings = self._controller_bindings()
        controller_events = self.controller.events.verified_events()
        _, _, window_events = self.controller.window_ledger._replay()
        for index, raw_entry in enumerate(entries, 1):
            entry = validate_journal_entry_signature(raw_entry, self.producer_verifier)
            if entry["seq"] != index or entry["previous_chain_sha256"] != previous_chain:
                raise NotificationError("notification journal sequence/hash chain is discontinuous")
            recorded_time = _parse_utc(entry["recorded_at"], "journal recorded_at")
            if previous_time is not None and recorded_time < previous_time:
                raise NotificationError("notification journal UTC time regressed")
            previous_time = recorded_time
            previous_chain = entry["chain_sha256"]
            intent_hash = entry["intent_sha256"]
            payload = entry["payload"]
            if entry["kind"] == PREPARED:
                if entry["attempt"] != 0 or set(payload) != {"intent"}:
                    raise NotificationError("PREPARED journal entry payload/attempt invalid")
                intent = validate_notification_intent(
                    payload["intent"], self.producer_verifier, expected_identity=self.identity
                )
                if intent["seal_sha256"] != intent_hash \
                        or intent["dedupe_key"] != entry["dedupe_key"]:
                    raise NotificationError("PREPARED entry differs from its intent")
                if intent_hash in lifecycle or intent["event_identity_sha256"] in event_ids:
                    raise NotificationError("duplicate notification intent/event identity")
                if _parse_utc(intent["prepared_at"], "intent prepared_at") > recorded_time:
                    raise NotificationError("PREPARED journal entry predates its intent")
                previous_anchor = self._validate_intent_anchor_history(
                    intent,
                    controller_events=controller_events,
                    window_events=window_events,
                    previous_anchor=previous_anchor,
                )
                self._validate_status_monotonic(
                    previous_status, intent["canonical_status"]
                )
                previous_status = intent["canonical_status"]
                intents[intent_hash] = intent
                lifecycle[intent_hash] = PREPARED
                attempts[intent_hash] = 0
                prepared_entries[intent_hash] = entry
                event_ids.add(intent["event_identity_sha256"])
                if intent["event_kind"] == SOURCE_COVERAGE_5_PERCENT:
                    coverage.add(intent["facts"]["subject"]["threshold_percent"])
                continue
            if intent_hash not in intents or entry["dedupe_key"] != intents[intent_hash]["dedupe_key"]:
                raise NotificationError("lifecycle entry references an unknown notification intent")
            current = lifecycle[intent_hash]
            if entry["kind"] == SEND_STARTED:
                if current not in {PREPARED, REPLAY_AUTHORIZED} \
                        or set(payload) != {"prepared_seq", "prepared_chain_sha256"}:
                    raise NotificationError("SEND_STARTED lifecycle transition invalid")
                prepared = prepared_entries[intent_hash]
                if payload != {
                    "prepared_seq": prepared["seq"],
                    "prepared_chain_sha256": prepared["chain_sha256"],
                }:
                    raise NotificationError("SEND_STARTED differs from its PREPARED entry")
                expected_attempt = 1 if current == PREPARED else attempts[intent_hash]
                if entry["attempt"] != expected_attempt:
                    raise NotificationError("SEND_STARTED attempt differs from authorization")
                lifecycle[intent_hash] = SEND_STARTED
                attempts[intent_hash] = entry["attempt"]
                latest_started[intent_hash] = entry
            elif entry["kind"] == AMBIGUOUS_BLOCKED:
                if current != SEND_STARTED or entry["attempt"] != attempts[intent_hash] \
                        or set(payload) != {"send_started_seq", "send_started_chain_sha256", "reason"}:
                    raise NotificationError("AMBIGUOUS_BLOCKED lifecycle transition invalid")
                started = latest_started[intent_hash]
                if payload["send_started_seq"] != started["seq"] \
                        or payload["send_started_chain_sha256"] != started["chain_sha256"]:
                    raise NotificationError("ambiguity record differs from exact send start")
                _safe_text(payload["reason"], "ambiguity reason")
                lifecycle[intent_hash] = AMBIGUOUS_BLOCKED
                latest_ambiguous[intent_hash] = entry
            elif entry["kind"] == REPLAY_AUTHORIZED:
                if current != AMBIGUOUS_BLOCKED or set(payload) != {
                    "challenge", "authorization"
                }:
                    raise NotificationError("REPLAY_AUTHORIZED lifecycle transition invalid")
                challenge = validate_reconciliation_challenge(payload["challenge"])
                authorization = validate_reconciliation_authorization(
                    payload["authorization"], challenge,
                    verifier=self.reconciliation_verifier,
                )
                started = latest_started[intent_hash]
                ambiguous = latest_ambiguous[intent_hash]
                anchored_at_ambiguity = controller_bindings.get(ambiguous["seq"])
                expected_binding = (
                    anchored_at_ambiguity.get("seal_sha256")
                    if anchored_at_ambiguity is not None else None
                )
                expected = {
                    "intent_sha256": intent_hash,
                    "dedupe_key": intents[intent_hash]["dedupe_key"],
                    "ambiguous_seq": ambiguous["seq"],
                    "ambiguous_chain_sha256": ambiguous["chain_sha256"],
                    "send_started_seq": started["seq"],
                    "send_started_chain_sha256": started["chain_sha256"],
                    "authorized_attempt": attempts[intent_hash] + 1,
                    "journal_binding_sha256": expected_binding,
                }
                if expected_binding is None or any(
                    challenge.get(key) != value for key, value in expected.items()
                ) or entry["attempt"] != challenge["authorized_attempt"]:
                    raise NotificationError("reconciliation does not bind the exact ambiguous head")
                if _parse_utc(authorization["authorized_at"], "authorized_at") > recorded_time:
                    raise NotificationError("REPLAY_AUTHORIZED entry predates its authorization")
                lifecycle[intent_hash] = REPLAY_AUTHORIZED
                attempts[intent_hash] = entry["attempt"]
            elif entry["kind"] == COMMITTED:
                if current not in {SEND_STARTED, AMBIGUOUS_BLOCKED} \
                        or set(payload) != {"receipt"} \
                        or entry["attempt"] != attempts[intent_hash]:
                    raise NotificationError("COMMITTED lifecycle transition invalid")
                started = latest_started[intent_hash]
                receipt = validate_bot_delivery_receipt(
                    payload["receipt"], intents[intent_hash], started,
                    producer_verifier=self.producer_verifier,
                    receipt_verifier=self.receipt_verifier,
                    expected_chat_identity_digest=self.expected_chat_identity_digest,
                    expected_bot_identity_digest=self.expected_bot_identity_digest,
                )
                message_identity = (
                    receipt["chat_identity_digest"], receipt["message_id"]
                )
                if receipt["bot_api_response_sha256"] in delivery_response_hashes \
                        or message_identity in delivery_message_ids:
                    raise NotificationError(
                        "Telegram delivery response/message identity was reused"
                    )
                delivery_response_hashes.add(receipt["bot_api_response_sha256"])
                delivery_message_ids.add(message_identity)
                if _parse_utc(receipt["delivered_at"], "receipt delivered_at") > recorded_time:
                    raise NotificationError("COMMITTED journal entry predates delivery")
                lifecycle[intent_hash] = COMMITTED
                committed_ids.add(intents[intent_hash]["event_identity_sha256"])
                latest_status = copy.deepcopy(intents[intent_hash]["canonical_status"])
            else:  # pragma: no cover
                raise NotificationError("unknown notification journal lifecycle")
        return ReplaySnapshot(
            entries=tuple(copy.deepcopy(list(entries))),
            intents=copy.deepcopy(intents),
            lifecycle=copy.deepcopy(lifecycle),
            attempts=copy.deepcopy(attempts),
            latest_send_started=copy.deepcopy(latest_started),
            latest_ambiguous=copy.deepcopy(latest_ambiguous),
            committed_event_identities=frozenset(committed_ids),
            prepared_coverage_thresholds=frozenset(coverage),
            latest_delivered_status=latest_status,
            head=copy.deepcopy(dict(head)),
            binding=copy.deepcopy(dict(binding)),
        )

    def _load_locked(self, parent_fd: int, journal_fd: int) -> ReplaySnapshot:
        """Read, reconcile one authenticated crash tail, and compare controller anchor."""
        entries = self._read_entries(journal_fd)
        head, head_info = self._read_head(parent_fd)
        if len(entries) == head["entry_count"] + 1:
            # The sole recoverable journal/head gap: one fully producer-signed,
            # hash-linked entry after a durable old head and exact controller anchor.
            old_binding = self._binding_from_head(head)
            try:
                controller_binding = self.controller.notification_journal_binding()
            except StateError as exc:
                raise NotificationError(str(exc)) from exc
            if controller_binding != old_binding:
                raise NotificationError(
                    "notification journal tail cannot recover from a non-current head"
                )
            tail = entries[-1]
            if tail["seq"] != len(entries) \
                    or tail["previous_chain_sha256"] != head["head_chain_sha256"]:
                raise NotificationError("notification journal tail is not an exact head advance")
            observed = self._now_after(head["observed_at"], tail["recorded_at"])
            head = self._make_head(
                entries, old_binding["seal_sha256"], observed_at=observed
            )
            new_binding = self._binding_from_head(head)
            self._replay_entries(entries, head, new_binding)
            head_info = self._write_head(
                parent_fd,
                head,
                expected_head_identity=_file_identity(head_info),
            )
            try:
                self.controller.anchor_notification_journal_head(new_binding)
            except StateError as exc:
                raise NotificationError(str(exc)) from exc
        elif len(entries) != head["entry_count"]:
            raise NotificationError(
                "notification journal/head clean-tail truncation or ambiguous gap"
            )

        disk_binding = self._binding_from_head(head)
        semantic_replay = self._replay_entries(entries, head, disk_binding)
        try:
            controller_binding = self.controller.notification_journal_binding()
        except StateError as exc:
            raise NotificationError(str(exc)) from exc
        if controller_binding is None:
            if disk_binding["entry_count"] != 0 \
                    or disk_binding["previous_binding_sha256"] is not None:
                raise NotificationError("non-genesis notification history lacks controller anchor")
            try:
                self.controller.anchor_notification_journal_head(disk_binding)
            except StateError as exc:
                raise NotificationError(str(exc)) from exc
            controller_binding = self.controller.notification_journal_binding()
        elif disk_binding != controller_binding:
            recoverable = (
                disk_binding["entry_count"] == controller_binding["entry_count"] + 1
                and disk_binding["previous_binding_sha256"]
                == controller_binding["seal_sha256"]
            )
            if recoverable and controller_binding["entry_count"] > 0:
                prior = entries[controller_binding["entry_count"] - 1]
                recoverable = (
                    prior["chain_sha256"] == controller_binding["head_chain_sha256"]
                    and prior["producer_signature"]
                    == controller_binding["head_entry_signature"]
                )
            elif recoverable:
                recoverable = controller_binding["head_chain_sha256"] == GENESIS_HASH
            if not recoverable:
                raise NotificationError(
                    "notification disk head was deleted, rolled back, replaced, or forked"
                )
            try:
                self.controller.anchor_notification_journal_head(disk_binding)
            except StateError as exc:
                raise NotificationError(str(exc)) from exc
            controller_binding = self.controller.notification_journal_binding()
        if controller_binding != disk_binding:
            raise NotificationError("notification controller/disk binding reconciliation failed")
        return semantic_replay

    def _append_locked(
        self,
        parent_fd: int,
        journal_fd: int,
        replay: ReplaySnapshot,
        *,
        kind: str,
        intent: Mapping[str, Any],
        attempt: int,
        payload: Mapping[str, Any],
        time_floors: Sequence[str] = (),
    ) -> dict[str, Any]:
        if kind not in LIFECYCLE_KINDS:
            raise NotificationError("unknown journal lifecycle kind")
        intent = validate_notification_intent(
            intent, self.producer_verifier, expected_identity=self.identity
        )
        floors = list(time_floors)
        if replay.entries:
            floors.append(replay.entries[-1]["recorded_at"])
        floors.append(intent["prepared_at"])
        recorded_at = self._now_after(*floors)
        entry = _make_journal_entry(
            self.producer_signer,
            seq=len(replay.entries) + 1,
            kind=kind,
            recorded_at=recorded_at,
            intent_sha256=intent["seal_sha256"],
            dedupe_key=intent["dedupe_key"],
            attempt=attempt,
            payload=payload,
            previous_chain_sha256=(
                replay.entries[-1]["chain_sha256"] if replay.entries else GENESIS_HASH
            ),
        )
        before = os.fstat(journal_fd)
        if not _same_identity(before, self._pinned["journal_identity"]):
            raise NotificationError("notification journal identity changed before append")
        raw = canonical(entry) + b"\n"
        offset = 0
        while offset < len(raw):
            written = os.write(journal_fd, raw[offset:])
            if written <= 0:
                raise NotificationError("short write to notification journal")
            offset += written
        os.fsync(journal_fd)
        after = os.fstat(journal_fd)
        if not _same_identity(after, self._pinned["journal_identity"]) \
                or after.st_size != before.st_size + len(raw):
            raise NotificationError("notification journal append identity/length mismatch")
        restored_entries = self._read_entries(journal_fd)
        if len(restored_entries) != len(replay.entries) + 1 \
                or restored_entries[-1] != entry:
            raise NotificationError("notification journal append post-write mismatch")
        self._fault("after_journal_fsync")
        old_head, old_head_info = self._read_head(parent_fd)
        if old_head != replay.head:
            raise NotificationError("notification journal head changed before append commit")
        new_head = self._make_head(
            restored_entries,
            replay.binding["seal_sha256"],
            observed_at=self._now_after(old_head["observed_at"], entry["recorded_at"]),
        )
        binding = self._binding_from_head(new_head)
        self._replay_entries(restored_entries, new_head, binding)
        self._write_head(
            parent_fd,
            new_head,
            expected_head_identity=_file_identity(old_head_info),
        )
        self._fault("after_head_fsync_before_controller_anchor")
        try:
            self.controller.anchor_notification_journal_head(binding)
        except StateError as exc:
            raise NotificationError(str(exc)) from exc
        self._fault("after_controller_anchor")
        return entry

    def replay(self) -> ReplaySnapshot:
        self.controller.lease.assert_held()
        with self._mutex:
            with self._locked_files() as (parent_fd, journal_fd):
                return self._load_locked(parent_fd, journal_fd)

    @staticmethod
    def _recorded_intent(
        replay: ReplaySnapshot, intent_value: Mapping[str, Any],
        producer_verifier: NotificationVerifier,
    ) -> tuple[str, dict[str, Any]]:
        intent = validate_notification_intent(intent_value, producer_verifier)
        intent_hash = intent["seal_sha256"]
        recorded = replay.intents.get(intent_hash)
        if recorded is None or recorded != intent:
            raise NotificationError("notification intent is not the exact journaled intent")
        return intent_hash, intent

    def prepare(self, *, event_kind: str) -> dict[str, Any]:
        """Prepare one milestone solely from the held controller's verified snapshot."""
        if event_kind == SOURCE_COVERAGE_5_PERCENT:
            raise NotificationError(
                "coverage milestones may only be prepared through exact crossing detection"
            )
        self.controller.lease.assert_held()
        with self._mutex:
            with self._locked_files() as (parent_fd, journal_fd):
                replay = self._load_locked(parent_fd, journal_fd)
                try:
                    snapshot = self.controller.notification_snapshot()
                except StateError as exc:
                    raise NotificationError(str(exc)) from exc
                intent = _make_intent(
                    self.controller, event_kind, snapshot, self.producer_signer
                )
                if any(
                    existing["event_identity_sha256"] == intent["event_identity_sha256"]
                    for existing in replay.intents.values()
                ):
                    raise NotificationError("duplicate notification event identity")
                self._append_locked(
                    parent_fd,
                    journal_fd,
                    replay,
                    kind=PREPARED,
                    intent=intent,
                    attempt=0,
                    payload={"intent": intent},
                )
                return intent

    def prepare_coverage_crossings(self) -> list[dict[str, Any]]:
        """Prepare each newly crossed exact 5% threshold once, from live ledgers."""
        self.controller.lease.assert_held()
        prepared: list[dict[str, Any]] = []
        with self._mutex:
            with self._locked_files() as (parent_fd, journal_fd):
                replay = self._load_locked(parent_fd, journal_fd)
                if not any(
                    intent["event_kind"] == SOURCE_STREAM_STARTED
                    and replay.lifecycle[intent_hash] == COMMITTED
                    for intent_hash, intent in replay.intents.items()
                ):
                    raise NotificationError(
                        "coverage notifications require a committed source-stream start"
                    )
                try:
                    snapshot = self.controller.notification_snapshot()
                except StateError as exc:
                    raise NotificationError(str(exc)) from exc
                source = _object(snapshot.get("source"), "snapshot source")
                expected = _uint(source.get("expected_logical_bytes"), "expected source bytes")
                verified = _uint(source.get("verified_logical_bytes"), "verified source bytes")
                crossed = [
                    threshold for threshold in range(5, 101, 5)
                    if verified * 100 >= expected * threshold
                    and threshold not in replay.prepared_coverage_thresholds
                ]
                for threshold in crossed:
                    # Each PREPARED append advances the controller anchor; take a
                    # fresh verified snapshot for the next crossing.
                    try:
                        fresh = self.controller.notification_snapshot()
                    except StateError as exc:
                        raise NotificationError(str(exc)) from exc
                    intent = _make_intent(
                        self.controller,
                        SOURCE_COVERAGE_5_PERCENT,
                        fresh,
                        self.producer_signer,
                        coverage_threshold=threshold,
                    )
                    self._append_locked(
                        parent_fd,
                        journal_fd,
                        replay,
                        kind=PREPARED,
                        intent=intent,
                        attempt=0,
                        payload={"intent": intent},
                    )
                    prepared.append(intent)
                    replay = self._load_locked(parent_fd, journal_fd)
        return prepared

    def start_delivery(self, intent_value: Mapping[str, Any]) -> dict[str, Any]:
        self.controller.lease.assert_held()
        with self._mutex:
            with self._locked_files() as (parent_fd, journal_fd):
                replay = self._load_locked(parent_fd, journal_fd)
                intent_hash, intent = self._recorded_intent(
                    replay, intent_value, self.producer_verifier
                )
                current = replay.lifecycle[intent_hash]
                if current not in {PREPARED, REPLAY_AUTHORIZED}:
                    if current in {SEND_STARTED, AMBIGUOUS_BLOCKED}:
                        raise NotificationError(
                            "delivery is ambiguous and requires exact reconciliation"
                        )
                    raise NotificationError("notification is already committed")
                attempt = 1 if current == PREPARED else replay.attempts[intent_hash]
                prepared = next(
                    entry for entry in replay.entries
                    if entry["kind"] == PREPARED and entry["intent_sha256"] == intent_hash
                )
                return self._append_locked(
                    parent_fd,
                    journal_fd,
                    replay,
                    kind=SEND_STARTED,
                    intent=intent,
                    attempt=attempt,
                    payload={
                        "prepared_seq": prepared["seq"],
                        "prepared_chain_sha256": prepared["chain_sha256"],
                    },
                )

    def mark_ambiguous(
        self, intent_value: Mapping[str, Any], *, reason: str
    ) -> dict[str, Any]:
        reason = _safe_text(reason, "ambiguity reason")
        self.controller.lease.assert_held()
        with self._mutex:
            with self._locked_files() as (parent_fd, journal_fd):
                replay = self._load_locked(parent_fd, journal_fd)
                intent_hash, intent = self._recorded_intent(
                    replay, intent_value, self.producer_verifier
                )
                if replay.lifecycle[intent_hash] != SEND_STARTED:
                    raise NotificationError("only SEND_STARTED delivery may become ambiguous")
                started = replay.latest_send_started[intent_hash]
                return self._append_locked(
                    parent_fd,
                    journal_fd,
                    replay,
                    kind=AMBIGUOUS_BLOCKED,
                    intent=intent,
                    attempt=started["attempt"],
                    payload={
                        "send_started_seq": started["seq"],
                        "send_started_chain_sha256": started["chain_sha256"],
                        "reason": reason,
                    },
                )

    def make_reconciliation_challenge(
        self, intent_value: Mapping[str, Any]
    ) -> dict[str, Any]:
        self.controller.lease.assert_held()
        with self._mutex:
            with self._locked_files() as (parent_fd, journal_fd):
                replay = self._load_locked(parent_fd, journal_fd)
                intent_hash, intent = self._recorded_intent(
                    replay, intent_value, self.producer_verifier
                )
                if replay.lifecycle[intent_hash] != AMBIGUOUS_BLOCKED:
                    raise NotificationError("reconciliation requires an ambiguous delivery")
                started = replay.latest_send_started[intent_hash]
                ambiguous = replay.latest_ambiguous[intent_hash]
                return seal({
                    "schema": RECONCILIATION_CHALLENGE_SCHEMA,
                    "intent_sha256": intent_hash,
                    "dedupe_key": intent["dedupe_key"],
                    "ambiguous_seq": ambiguous["seq"],
                    "ambiguous_chain_sha256": ambiguous["chain_sha256"],
                    "send_started_seq": started["seq"],
                    "send_started_chain_sha256": started["chain_sha256"],
                    "authorized_attempt": started["attempt"] + 1,
                    "journal_binding_sha256": replay.binding["seal_sha256"],
                    "nonce": secrets.token_hex(32),
                    "issued_at": self._now_after(
                        ambiguous["recorded_at"], replay.head["observed_at"]
                    ),
                })

    def authorize_replay(
        self,
        intent_value: Mapping[str, Any],
        challenge_value: Mapping[str, Any],
        authorization_value: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.controller.lease.assert_held()
        challenge = validate_reconciliation_challenge(challenge_value)
        authorization = validate_reconciliation_authorization(
            authorization_value, challenge, verifier=self.reconciliation_verifier
        )
        with self._mutex:
            with self._locked_files() as (parent_fd, journal_fd):
                replay = self._load_locked(parent_fd, journal_fd)
                intent_hash, intent = self._recorded_intent(
                    replay, intent_value, self.producer_verifier
                )
                if replay.lifecycle[intent_hash] != AMBIGUOUS_BLOCKED:
                    raise NotificationError("reconciliation is stale or already consumed")
                started = replay.latest_send_started[intent_hash]
                ambiguous = replay.latest_ambiguous[intent_hash]
                expected = {
                    "intent_sha256": intent_hash,
                    "dedupe_key": intent["dedupe_key"],
                    "ambiguous_seq": ambiguous["seq"],
                    "ambiguous_chain_sha256": ambiguous["chain_sha256"],
                    "send_started_seq": started["seq"],
                    "send_started_chain_sha256": started["chain_sha256"],
                    "authorized_attempt": started["attempt"] + 1,
                    "journal_binding_sha256": replay.binding["seal_sha256"],
                }
                if any(challenge.get(key) != value for key, value in expected.items()):
                    raise NotificationError("reconciliation challenge is stale or for another head")
                return self._append_locked(
                    parent_fd,
                    journal_fd,
                    replay,
                    kind=REPLAY_AUTHORIZED,
                    intent=intent,
                    attempt=challenge["authorized_attempt"],
                    payload={"challenge": challenge, "authorization": authorization},
                    time_floors=(authorization["authorized_at"],),
                )

    def commit(
        self, intent_value: Mapping[str, Any], receipt_value: Mapping[str, Any]
    ) -> dict[str, Any]:
        self.controller.lease.assert_held()
        with self._mutex:
            with self._locked_files() as (parent_fd, journal_fd):
                replay = self._load_locked(parent_fd, journal_fd)
                intent_hash, intent = self._recorded_intent(
                    replay, intent_value, self.producer_verifier
                )
                current = replay.lifecycle[intent_hash]
                if current == COMMITTED:
                    raise NotificationError("notification is already committed")
                if current not in {SEND_STARTED, AMBIGUOUS_BLOCKED}:
                    raise NotificationError("delivery must have a durable SEND_STARTED entry")
                started = replay.latest_send_started[intent_hash]
                receipt = validate_bot_delivery_receipt(
                    receipt_value,
                    intent,
                    started,
                    producer_verifier=self.producer_verifier,
                    receipt_verifier=self.receipt_verifier,
                    expected_chat_identity_digest=self.expected_chat_identity_digest,
                    expected_bot_identity_digest=self.expected_bot_identity_digest,
                )
                return self._append_locked(
                    parent_fd,
                    journal_fd,
                    replay,
                    kind=COMMITTED,
                    intent=intent,
                    attempt=started["attempt"],
                    payload={"receipt": receipt},
                    time_floors=(receipt["delivered_at"],),
                )

    def pending(self) -> list[dict[str, Any]]:
        replay = self.replay()
        result: list[dict[str, Any]] = []
        for intent_hash, intent in replay.intents.items():
            lifecycle = replay.lifecycle[intent_hash]
            if lifecycle == COMMITTED:
                continue
            result.append({
                "intent": copy.deepcopy(intent),
                "lifecycle": lifecycle,
                "attempt": replay.attempts[intent_hash],
                "safe_to_send": lifecycle in {PREPARED, REPLAY_AUTHORIZED},
                "ambiguous": lifecycle in {SEND_STARTED, AMBIGUOUS_BLOCKED},
            })
        return result


def make_bot_delivery_receipt(
    intent_value: Mapping[str, Any],
    send_started_entry: Mapping[str, Any],
    *,
    signer: NotificationReceiptSigner,
    bot_api_response: Mapping[str, Any],
    http_status: int,
    delivered_at: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper; no network I/O occurs here."""
    if not isinstance(signer, NotificationReceiptSigner):
        raise NotificationError("delivery receipt requires the separate sender signer")
    return signer.make_delivery_receipt(
        intent_value,
        send_started_entry,
        bot_api_response=bot_api_response,
        http_status=http_status,
        delivered_at=delivered_at,
    )


class _ReadOnlyNotificationOutboxView:
    """Duck-typed reuse of journal validators without any signing/write methods."""

    _lstat_name = staticmethod(NotificationJournal._lstat_name)
    _read_exact_fd = staticmethod(NotificationJournal._read_exact_fd)
    _read_entries = NotificationJournal._read_entries
    _read_head = NotificationJournal._read_head
    _head_common = NotificationJournal._head_common
    _validate_head = NotificationJournal._validate_head
    _controller_bindings = NotificationJournal._controller_bindings
    _validate_intent_anchor_history = NotificationJournal._validate_intent_anchor_history
    _validate_status_monotonic = staticmethod(NotificationJournal._validate_status_monotonic)
    _replay_entries = NotificationJournal._replay_entries

    def __init__(
        self,
        path: Path,
        *,
        controller: Controller,
        producer_verifier: NotificationVerifier,
        receipt_verifier: NotificationVerifier,
        reconciliation_verifier: NotificationVerifier,
        expected_bot_identity_digest: str,
        binding: Mapping[str, Any],
    ) -> None:
        self.path = path
        self.head_path = path.with_name(path.name + ".head.json")
        self.lock_path = path.with_name(path.name + ".lock")
        self.controller = controller
        self.producer_verifier = producer_verifier
        self.receipt_verifier = receipt_verifier
        self.reconciliation_verifier = reconciliation_verifier
        self.expected_chat_identity_digest = (
            controller.expected_contract["expected_chat_identity_digest"]
        )
        self.expected_bot_identity_digest = expected_bot_identity_digest
        self._path_sha256 = hashlib.sha256(
            b"hawking.glm52.notification-journal-path.v1\0" + os.fsencode(path)
        ).hexdigest()
        expected = {
            **self.identity,
            "producer_public_key": producer_verifier.public_key_hex,
            "producer_key_id": producer_verifier.key_id,
            "receipt_key_id": receipt_verifier.key_id,
            "reconciliation_key_id": reconciliation_verifier.key_id,
            "expected_chat_identity_digest": self.expected_chat_identity_digest,
            "expected_bot_identity_digest": expected_bot_identity_digest,
            "path_sha256": self._path_sha256,
        }
        if any(binding.get(key) != value for key, value in expected.items()):
            raise NotificationError(
                "notification audit configuration differs from controller binding"
            )
        self._pinned = {
            "parent_identity": copy.deepcopy(binding["parent_identity"]),
            "journal_identity": copy.deepcopy(binding["journal_identity"]),
            "lock_identity": copy.deepcopy(binding["lock_identity"]),
        }

    @property
    def identity(self) -> dict[str, str]:
        return {
            "campaign_id": self.controller.campaign_id,
            "source_revision": self.controller.source_revision,
            "controller_epoch": self.controller.controller_epoch,
            "expected_contract_sha256": self.controller.expected_contract_sha256,
        }

    def _assert_parent_identity(self) -> None:
        NotificationJournal._assert_parent_identity(self)

    @contextmanager
    def locked_files(self) -> Iterator[tuple[int, int]]:
        """Take a shared advisory lock with read-only file descriptors."""
        parent_fd: int | None = None
        lock_fd: int | None = None
        journal_fd: int | None = None
        try:
            parent_fd = _open_directory_components(self.path.parent)
            if not _same_identity(os.fstat(parent_fd), self._pinned["parent_identity"]):
                raise NotificationError("notification parent identity changed")
            flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) \
                | getattr(os, "O_CLOEXEC", 0)
            lock_fd = os.open(self.lock_path.name, flags, dir_fd=parent_fd)
            lock_info = os.fstat(lock_fd)
            _validate_secure_regular(lock_info, "notification lock")
            if not _same_identity(lock_info, self._pinned["lock_identity"]):
                raise NotificationError("notification lock identity changed")
            named_lock = self._lstat_name(parent_fd, self.lock_path.name)
            if named_lock is None or not _same_identity(
                named_lock, self._pinned["lock_identity"]
            ):
                raise NotificationError("notification lock name changed")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise NotificationError(
                    "notification outbox is busy; read-only audit refused to wait"
                ) from None
            journal_fd = os.open(self.path.name, flags, dir_fd=parent_fd)
            journal_info = os.fstat(journal_fd)
            _validate_secure_regular(journal_info, "notification journal")
            if not _same_identity(journal_info, self._pinned["journal_identity"]):
                raise NotificationError("notification journal identity changed")
            named_journal = self._lstat_name(parent_fd, self.path.name)
            if named_journal is None or not _same_identity(
                named_journal, self._pinned["journal_identity"]
            ):
                raise NotificationError("notification journal name changed")
            yield parent_fd, journal_fd
            for name, identity, label in (
                (self.path.name, self._pinned["journal_identity"], "journal"),
                (self.lock_path.name, self._pinned["lock_identity"], "lock"),
            ):
                named = self._lstat_name(parent_fd, name)
                if named is None or not _same_identity(named, identity):
                    raise NotificationError(
                        f"notification {label} name changed during audit"
                    )
            self._assert_parent_identity()
        except FileNotFoundError:
            raise NotificationError(
                "controller-anchored notification storage was deleted"
            ) from None
        except OSError as exc:
            raise NotificationError(
                f"cannot safely audit notification storage: {exc}"
            ) from None
        finally:
            if journal_fd is not None:
                os.close(journal_fd)
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(lock_fd)
            if parent_fd is not None:
                os.close(parent_fd)


def _head_matches_controller_binding(
    head: Mapping[str, Any], binding: Mapping[str, Any]
) -> bool:
    return all((
        binding.get("entry_count") == head.get("entry_count"),
        binding.get("head_chain_sha256") == head.get("head_chain_sha256"),
        binding.get("head_entry_signature") == head.get("head_entry_signature"),
        binding.get("head_document_sha256")
        == hashlib.sha256(canonical(head)).hexdigest(),
        binding.get("head_signature") == head.get("producer_signature"),
        binding.get("previous_binding_sha256") == head.get("previous_binding_sha256"),
        binding.get("observed_at") == head.get("observed_at"),
    ))


def audit_notification_outbox(
    path: str | os.PathLike[str],
    *,
    controller: Controller,
    producer_verifier: NotificationVerifier,
    receipt_verifier: NotificationVerifier,
    reconciliation_verifier: NotificationVerifier,
    expected_bot_identity_digest: str,
) -> dict[str, Any]:
    """Strictly read and semantically replay one controller-bound outbox.

    The API owns no signing key and performs no repair, append, head replacement,
    controller anchoring, checkpoint recovery, or network operation.  Even the
    otherwise recoverable one-entry crash gaps are audit failures.
    """
    if not isinstance(controller, Controller):
        raise NotificationError("notification audit requires a live Controller")
    try:
        controller.lease.assert_held()
    except StateError as exc:
        raise NotificationError(str(exc)) from exc
    verifiers = (producer_verifier, receipt_verifier, reconciliation_verifier)
    if any(not isinstance(item, NotificationVerifier) for item in verifiers) \
            or tuple(item.role for item in verifiers) != (
                "producer", "receipt", "reconciliation"
            ):
        raise NotificationError("notification audit verifier roles are invalid")
    if len({item.key_id for item in verifiers}) != 3:
        raise NotificationError("notification audit verifier keys must be distinct")
    if not _is_sha256(expected_bot_identity_digest):
        raise NotificationError("notification audit bot identity digest invalid")
    normalized = _normalized_absolute_path(path)
    mutex = _thread_lock(normalized)
    if not mutex.acquire(blocking=False):
        raise NotificationError(
            "notification outbox is busy; read-only audit refused to wait"
        )
    try:
        try:
            checkpoint = controller.resume(recover_single_tail=False)
        except StateError as exc:
            raise NotificationError(str(exc)) from exc
        binding = checkpoint.get("notification_journal")
        if not isinstance(binding, dict):
            raise NotificationError("controller has no anchored notification outbox")
        view = _ReadOnlyNotificationOutboxView(
            normalized,
            controller=controller,
            producer_verifier=producer_verifier,
            receipt_verifier=receipt_verifier,
            reconciliation_verifier=reconciliation_verifier,
            expected_bot_identity_digest=expected_bot_identity_digest,
            binding=binding,
        )
        with view.locked_files() as (parent_fd, journal_fd):
            entries = view._read_entries(journal_fd)
            head, _ = view._read_head(parent_fd)
            if len(entries) != head["entry_count"]:
                raise NotificationError(
                    "read-only notification audit refuses every journal/head crash-tail gap"
                )
            if not _head_matches_controller_binding(head, binding):
                raise NotificationError(
                    "notification disk head differs from the controller binding"
                )
            replay = view._replay_entries(entries, head, binding)
            try:
                checkpoint_after = controller.resume(recover_single_tail=False)
            except StateError as exc:
                raise NotificationError(str(exc)) from exc
            if checkpoint_after.get("notification_journal") != binding:
                raise NotificationError(
                    "notification controller binding changed during read-only audit"
                )
    finally:
        mutex.release()

    ordered_intent_hashes = [
        entry["intent_sha256"] for entry in replay.entries if entry["kind"] == PREPARED
    ]
    intent_summaries = [
        {
            "intent_sha256": intent_hash,
            "event_kind": replay.intents[intent_hash]["event_kind"],
            "event_identity_sha256": replay.intents[intent_hash][
                "event_identity_sha256"
            ],
            "dedupe_key": replay.intents[intent_hash]["dedupe_key"],
            "lifecycle": replay.lifecycle[intent_hash],
            "attempt": replay.attempts[intent_hash],
        }
        for intent_hash in ordered_intent_hashes
    ]
    lifecycle_counts = {
        kind: sum(item["lifecycle"] == kind for item in intent_summaries)
        for kind in (PREPARED, SEND_STARTED, AMBIGUOUS_BLOCKED, REPLAY_AUTHORIZED, COMMITTED)
    }
    unresolved = [item for item in intent_summaries if item["lifecycle"] != COMMITTED]
    latest_status = copy.deepcopy(replay.latest_delivered_status)
    body = {
        "schema": OUTBOX_AUDIT_SCHEMA,
        "status": "PASS",
        "replay_status": "VERIFIED",
        **view.identity,
        "path_sha256": view._path_sha256,
        "producer_key_id": producer_verifier.key_id,
        "receipt_key_id": receipt_verifier.key_id,
        "reconciliation_key_id": reconciliation_verifier.key_id,
        "expected_chat_identity_digest": view.expected_chat_identity_digest,
        "expected_bot_identity_digest": expected_bot_identity_digest,
        "entry_count": len(replay.entries),
        "head_chain_sha256": head["head_chain_sha256"],
        "head_document_sha256": hashlib.sha256(canonical(head)).hexdigest(),
        "controller_binding_sha256": binding["seal_sha256"],
        "intent_count": len(intent_summaries),
        "committed_intent_count": lifecycle_counts[COMMITTED],
        "unresolved_intent_count": len(unresolved),
        "safe_to_send_intent_count": sum(
            item["lifecycle"] in {PREPARED, REPLAY_AUTHORIZED}
            for item in unresolved
        ),
        "ambiguous_intent_count": sum(
            item["lifecycle"] in {SEND_STARTED, AMBIGUOUS_BLOCKED}
            for item in unresolved
        ),
        "lifecycle_counts": lifecycle_counts,
        "intents": intent_summaries,
        "prepared_coverage_thresholds": sorted(
            replay.prepared_coverage_thresholds
        ),
        "latest_delivered_status": latest_status,
        "latest_delivered_status_sha256": (
            _sha256(latest_status) if latest_status is not None else None
        ),
    }
    return {**body, "audit_sha256": _sha256(body)}

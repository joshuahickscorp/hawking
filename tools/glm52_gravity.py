#!/usr/bin/env python3.12
"""Offline phone/control surface for the durable GLM-5.2 campaign controller.

The four commands required by the campaign contract are intentionally control-plane
only.  They never contact Telegram, launch a worker, download a shard, or remove a
file.  Mutating commands append an authenticated operator request to a hash-chained,
fsync'd journal.  A worker may honor that request only at its documented safe point.

Configuration is sealed, secret-free JSON.  The receipt HMAC key and private-chat
identity are loaded only from the GLM-specific macOS Keychain provider; they are
kept in memory and never written to the journal or phone-status artifacts.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


_CONDENSE = Path(__file__).resolve().parent / "condense"
if str(_CONDENSE) not in sys.path:
    sys.path.insert(0, str(_CONDENSE))

import glm52_state as state  # noqa: E402
import glm52_evidence_auth as evidence_module  # noqa: E402
import glm52_grounding_auth as grounding_module  # noqa: E402
import glm52_telegram as telegram_module  # noqa: E402
from glm52_common import (  # noqa: E402
    Glm52Error,
    atomic_json,
    atomic_text,
    canonical,
    read_sealed_json,
    seal,
    utc_now,
    verify_sealed,
)


CONFIG_SCHEMA = "hawking.glm52.controller_cli_config.v4"
CONTROL_EVENT_SCHEMA = "hawking.glm52.operator_control_event.v1"
CONTROL_PAYLOAD_SCHEMA = "hawking.glm52.operator_control_request.v1"
CONTROL_APPLIED_EVENT_SCHEMA = "hawking.glm52.operator_control_applied_event.v1"
CONTROL_APPLIED_PAYLOAD_SCHEMA = "hawking.glm52.operator_control_applied.v1"
PHONE_STATUS_SCHEMA = "hawking.glm52.phone_status.v2"

ACTION_RUN = "RUN"
ACTION_PAUSE = "PAUSE_AFTER_WINDOW"
ACTION_STOP = "STOP"
CONTROL_ACTIONS = frozenset({ACTION_RUN, ACTION_PAUSE, ACTION_STOP})
WORKER_SAFE_POINTS = frozenset({
    "BEFORE_WINDOW_FETCH",
    "BETWEEN_RANGE_READS",
    "BEFORE_HEAVY_COMPUTE",
    "AFTER_WINDOW_EVICTION",
    "BETWEEN_WINDOWS",
})
PAUSE_SAFE_POINTS = frozenset({
    "BEFORE_WINDOW_FETCH",
    "AFTER_WINDOW_EVICTION",
    "BETWEEN_WINDOWS",
})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OFFICIAL_SOURCE_PROFILE = "OFFICIAL_GLM52_BF16"
SYNTHETIC_SOURCE_PROFILE = "SYNTHETIC_TEST_ONLY"
DEFAULT_CONTROLLER_EPOCH = "glm52-controller-v2"


class CliError(Glm52Error):
    """The CLI configuration, controller state, or operator journal is unsafe."""


@dataclass(frozen=True)
class Runtime:
    config_path: Path
    config_sha256: str
    controller_root: Path
    artifact_root: Path
    expected_contract_path: Path
    phone_status_directory: Path
    campaign_id: str
    source_revision: str
    controller_epoch: str
    allow_synthetic_contract: bool
    expected_contract: dict[str, Any]
    telegram_auth: state.TelegramAuthConfig
    evidence_auth: state.EvidenceAuthConfig
    grounding_auth: grounding_module.ProducerAuthenticator
    contract_phone_status_path: str

    def __post_init__(self) -> None:
        _validate_authentication_role_separation(
            telegram_auth=self.telegram_auth,
            evidence_auth=self.evidence_auth,
            grounding_auth=self.grounding_auth,
        )
        if self.evidence_auth.campaign_id != self.campaign_id \
                or self.evidence_auth.source_revision != self.source_revision:
            raise CliError("evidence authenticator campaign identity mismatch")

    @property
    def phone_json_path(self) -> Path:
        return self.phone_status_directory / "GLM52_PHONE_STATUS.json"

    @property
    def phone_markdown_path(self) -> Path:
        return self.phone_status_directory / "GLM52_PHONE_STATUS.md"

    @property
    def control_log_path(self) -> Path:
        return self.controller_root / "GLM52_OPERATOR_CONTROL.jsonl"

    @property
    def control_lock_path(self) -> Path:
        return self.controller_root / "GLM52_OPERATOR_CONTROL.lock"

    @property
    def control_applied_log_path(self) -> Path:
        return self.controller_root / "GLM52_OPERATOR_APPLIED.jsonl"

    @property
    def phone_lock_path(self) -> Path:
        return self.phone_status_directory / ".GLM52_PHONE_STATUS.lock"

    def controller(self) -> state.Controller:
        return state.Controller(
            self.controller_root,
            artifact_root=self.artifact_root,
            campaign_id=self.campaign_id,
            source_revision=self.source_revision,
            expected_contract=self.expected_contract,
            telegram_auth=self.telegram_auth,
            evidence_auth=self.evidence_auth,
            allow_synthetic_contract=self.allow_synthetic_contract,
            controller_epoch=self.controller_epoch,
        )


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CliError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CliError(f"JSON root is not an object: {path}")
    return value


def _resolve_path(config_path: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise CliError(f"{label} must be a non-empty path string")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve(strict=False)


def _normalized_absolute_path(value: str | os.PathLike[str], label: str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise CliError(f"{label} must be a normalized absolute path") from exc
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise CliError(f"{label} must be a non-empty absolute path")
    candidate = Path(raw)
    if not candidate.is_absolute() or raw.startswith("//") \
            or os.path.normpath(raw) != raw:
        raise CliError(f"{label} must be a normalized absolute path")
    return candidate


def load_expected_contract(
    expected_contract_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read one sealed contract without following a symlink or hard-link alias."""

    candidate = _normalized_absolute_path(
        expected_contract_path, "expected_contract_path"
    )
    try:
        value, _file_sha256 = state.TrustedArtifactStore(
            os.fspath(candidate.parent)
        ).read_sealed(candidate.name, label="expected campaign contract")
        return state._validate_expected_contract(value)
    except state.StateError as exc:
        raise CliError(str(exc)) from exc


def _contract_phone_output_path(
    contract: Mapping[str, Any], artifact_root: Path
) -> tuple[str, Path]:
    complete_gate = (contract.get("state_gates") or {}).get("COMPLETE")
    relative = (
        complete_gate.get("required_phone_status_path")
        if isinstance(complete_gate, dict)
        else None
    )
    if not isinstance(relative, str) or not relative or relative != relative.strip():
        raise CliError("expected campaign contract has no COMPLETE phone-status path")
    path = Path(relative)
    if path.is_absolute() or relative in {".", ".."} \
            or any(part in {"", ".", ".."} for part in path.parts):
        raise CliError("COMPLETE.required_phone_status_path must remain under artifact_root")
    output = artifact_root.joinpath(*path.parts)
    if output.name != "GLM52_PHONE_STATUS.json":
        raise CliError(
            "COMPLETE.required_phone_status_path must name GLM52_PHONE_STATUS.json"
        )
    return relative, output


def _validate_telegram_provider(config: Mapping[str, Any]) -> str:
    telegram_config = config.get("telegram")
    if not isinstance(telegram_config, dict) or set(telegram_config) != {
        "expected_chat_identity_digest",
        "credential_provider",
        "private_chat_id_service",
        "receipt_hmac_key_service",
    }:
        raise CliError("config.telegram has missing or unknown fields")
    if telegram_config.get("credential_provider") != "macOS Keychain" \
            or telegram_config.get("private_chat_id_service") != telegram_module.CHAT_SERVICE \
            or telegram_config.get("receipt_hmac_key_service") != telegram_module.HMAC_SERVICE:
        raise CliError("config.telegram does not name the fixed GLM macOS Keychain services")
    digest = telegram_config.get("expected_chat_identity_digest")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise CliError("expected_chat_identity_digest must be a sha256")
    return digest


def _validate_evidence_provider(config: Mapping[str, Any]) -> None:
    evidence_config = config.get("evidence_auth")
    if not isinstance(evidence_config, dict) or set(evidence_config) != {
        "credential_provider", "producer_hmac_key_service",
    }:
        raise CliError("config.evidence_auth has missing or unknown fields")
    if evidence_config.get("credential_provider") != "macOS Keychain" \
            or evidence_config.get("producer_hmac_key_service") != \
            evidence_module.EVIDENCE_HMAC_SERVICE:
        raise CliError("config.evidence_auth does not name the fixed evidence Keychain service")


def _validate_grounding_provider(config: Mapping[str, Any]) -> None:
    grounding_config = config.get("grounding_auth")
    if not isinstance(grounding_config, dict) or set(grounding_config) != {
        "credential_provider", "producer_hmac_key_service", "keychain_account",
    }:
        raise CliError("config.grounding_auth has missing or unknown fields")
    if grounding_config.get("credential_provider") != "macOS Keychain" \
            or grounding_config.get("producer_hmac_key_service") != \
            grounding_module.GROUNDING_HMAC_SERVICE \
            or grounding_config.get("keychain_account") != \
            grounding_module.KEYCHAIN_ACCOUNT:
        raise CliError(
            "config.grounding_auth does not name the fixed grounding Keychain "
            "service/account"
        )


def _validate_authentication_role_separation(
    *,
    telegram_auth: state.TelegramAuthConfig,
    evidence_auth: state.EvidenceAuthConfig,
    grounding_auth: grounding_module.ProducerAuthenticator,
) -> None:
    if not isinstance(telegram_auth, state.TelegramAuthConfig):
        raise CliError("TelegramAuthConfig is required")
    if not isinstance(evidence_auth, state.EvidenceAuthConfig):
        raise CliError("EvidenceAuthConfig is required")
    if not isinstance(grounding_auth, grounding_module.ProducerAuthenticator):
        raise CliError("grounding ProducerAuthenticator is required")
    # These equality fingerprints are domain-separated and remain in memory.  No key
    # or fingerprint is written to the sealed controller configuration.
    identities = {
        telegram_auth._key_material_identity(),
        evidence_auth._key_material_identity(),
        grounding_auth._key_material_identity(),
    }
    if len(identities) != 3:
        raise CliError(
            "Telegram receipt, scientific evidence, and filesystem grounding "
            "producer keys must be pairwise distinct"
        )


def load_runtime(
    config_path: str | os.PathLike[str], *,
    keychain: telegram_module.Keychain | None = None,
    evidence_keychain: evidence_module.Keychain | None = None,
    grounding_keychain: grounding_module.Keychain | None = None,
) -> Runtime:
    path = Path(config_path).resolve(strict=False)
    config = _strict_json(path)
    try:
        verify_sealed(config, label=str(path))
    except Glm52Error as exc:
        raise CliError(str(exc)) from exc
    required = {
        "schema",
        "campaign_id",
        "source_revision",
        "controller_epoch",
        "controller_root",
        "artifact_root",
        "expected_contract_path",
        "phone_status_directory",
        "allow_synthetic_contract",
        "telegram",
        "evidence_auth",
        "grounding_auth",
        "seal_sha256",
    }
    if set(config) != required:
        raise CliError("CLI config has missing or unknown fields")
    if config.get("schema") != CONFIG_SCHEMA:
        raise CliError("CLI config schema mismatch")
    if not isinstance(config.get("allow_synthetic_contract"), bool):
        raise CliError("allow_synthetic_contract must be boolean")
    for key in ("campaign_id", "controller_epoch"):
        value = config.get(key)
        if not isinstance(value, str) or not value or value != value.strip():
            raise CliError(f"{key} must be a non-empty, trimmed string")
    revision = config.get("source_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise CliError("source_revision must be an immutable 40-hex revision")
    chat_digest = _validate_telegram_provider(config)
    _validate_evidence_provider(config)
    _validate_grounding_provider(config)
    controller_root = _resolve_path(path, config["controller_root"], "controller_root")
    artifact_root = _resolve_path(path, config["artifact_root"], "artifact_root")
    contract_path = _resolve_path(
        path, config["expected_contract_path"], "expected_contract_path"
    )
    contract = load_expected_contract(contract_path)
    if contract.get("campaign_id") != config.get("campaign_id") \
            or contract.get("source_revision") != config.get("source_revision"):
        raise CliError("CLI config identity differs from expected campaign contract")
    source_profile = contract.get("source", {}).get("profile")
    if source_profile == OFFICIAL_SOURCE_PROFILE \
            and config["allow_synthetic_contract"] is not False:
        raise CliError("official contract forbids synthetic-contract authorization")
    if source_profile == SYNTHETIC_SOURCE_PROFILE \
            and config["allow_synthetic_contract"] is not True:
        raise CliError("synthetic contract requires explicit synthetic authorization")
    if contract.get("expected_chat_identity_digest") != chat_digest:
        raise CliError("CLI Telegram chat digest differs from expected campaign contract")
    contract_phone_path, required_phone_path = _contract_phone_output_path(
        contract, artifact_root
    )
    phone_directory = _resolve_path(
        path, config["phone_status_directory"], "phone_status_directory"
    )
    if required_phone_path != phone_directory / "GLM52_PHONE_STATUS.json":
        raise CliError(
            "configured phone-status output does not match the COMPLETE contract path"
        )
    try:
        auth = telegram_module.load_telegram_auth(
            keychain or telegram_module.MacOSKeychain()
        )
        evidence_auth = evidence_module.load_evidence_auth(
            evidence_keychain or keychain or evidence_module.MacOSKeychain(),
            campaign_id=config["campaign_id"],
            source_revision=config["source_revision"],
        )
        grounding_auth = grounding_module.load_grounding_auth(
            grounding_keychain or keychain or grounding_module.MacOSKeychain()
        )
        if auth.expected_chat_identity_digest != chat_digest:
            raise CliError(
                "Keychain private-chat identity differs from CLI/contract binding"
            )
        # Constructor validation is authoritative for controller identity and contract shape.
        runtime = Runtime(
            config_path=path,
            config_sha256=str(config["seal_sha256"]),
            controller_root=controller_root,
            artifact_root=artifact_root,
            expected_contract_path=contract_path,
            phone_status_directory=phone_directory,
            campaign_id=config["campaign_id"],
            source_revision=config["source_revision"],
            controller_epoch=config["controller_epoch"],
            allow_synthetic_contract=config["allow_synthetic_contract"],
            expected_contract=contract,
            telegram_auth=auth,
            evidence_auth=evidence_auth,
            grounding_auth=grounding_auth,
            contract_phone_status_path=contract_phone_path,
        )
        runtime.controller()
    except (
        state.StateError,
        telegram_module.TelegramSecurityError,
        evidence_module.EvidenceSecurityError,
        grounding_module.GroundingSecurityError,
    ) as exc:
        raise CliError(str(exc)) from exc
    return runtime


def make_config(
    *,
    campaign_id: str,
    source_revision: str,
    controller_root: str | os.PathLike[str],
    artifact_root: str | os.PathLike[str],
    expected_contract_path: str | os.PathLike[str],
    phone_status_directory: str | os.PathLike[str],
    expected_chat_identity_digest: str,
    allow_synthetic_contract: bool = False,
    controller_epoch: str = DEFAULT_CONTROLLER_EPOCH,
) -> dict[str, Any]:
    """Create a sealed, secret-free CLI configuration document."""
    return seal(
        {
            "schema": CONFIG_SCHEMA,
            "campaign_id": campaign_id,
            "source_revision": source_revision,
            "controller_epoch": controller_epoch,
            "controller_root": os.fspath(controller_root),
            "artifact_root": os.fspath(artifact_root),
            "expected_contract_path": os.fspath(expected_contract_path),
            "phone_status_directory": os.fspath(phone_status_directory),
            "allow_synthetic_contract": allow_synthetic_contract,
            "telegram": {
                "expected_chat_identity_digest": expected_chat_identity_digest,
                "credential_provider": "macOS Keychain",
                "private_chat_id_service": telegram_module.CHAT_SERVICE,
                "receipt_hmac_key_service": telegram_module.HMAC_SERVICE,
            },
            "evidence_auth": {
                "credential_provider": "macOS Keychain",
                "producer_hmac_key_service": evidence_module.EVIDENCE_HMAC_SERVICE,
            },
            "grounding_auth": {
                "credential_provider": "macOS Keychain",
                "producer_hmac_key_service": grounding_module.GROUNDING_HMAC_SERVICE,
                "keychain_account": grounding_module.KEYCHAIN_ACCOUNT,
            },
        }
    )


def make_config_from_contract(
    *,
    expected_contract_path: str | os.PathLike[str],
    controller_root: str | os.PathLike[str],
    artifact_root: str | os.PathLike[str],
    allow_synthetic_contract: bool = False,
    controller_epoch: str = DEFAULT_CONTROLLER_EPOCH,
) -> dict[str, Any]:
    """Derive every controller identity field from exact sealed contract bytes.

    The caller selects only filesystem placement and the controller epoch.  Campaign
    identity, source revision, Telegram chat identity, source profile, and the phone
    output location all come from the authoritative contract.
    """

    contract_path = _normalized_absolute_path(
        expected_contract_path, "expected_contract_path"
    )
    controller_path = _normalized_absolute_path(controller_root, "controller_root")
    artifact_path = _normalized_absolute_path(artifact_root, "artifact_root")
    if not isinstance(allow_synthetic_contract, bool):
        raise CliError("allow_synthetic_contract must be boolean")
    contract = load_expected_contract(contract_path)
    source_profile = contract["source"]["profile"]
    if source_profile == OFFICIAL_SOURCE_PROFILE:
        if allow_synthetic_contract:
            raise CliError("official contract forbids synthetic-contract authorization")
    elif source_profile == SYNTHETIC_SOURCE_PROFILE:
        if not allow_synthetic_contract:
            raise CliError("synthetic contract requires explicit synthetic authorization")
    else:  # Defensive: the authoritative contract validator currently rejects this.
        raise CliError("expected campaign contract source profile is unsupported")
    _contract_phone_path, phone_output = _contract_phone_output_path(
        contract, artifact_path
    )
    return make_config(
        campaign_id=contract["campaign_id"],
        source_revision=contract["source_revision"],
        controller_root=controller_path,
        artifact_root=artifact_path,
        expected_contract_path=contract_path,
        phone_status_directory=phone_output.parent,
        expected_chat_identity_digest=contract["expected_chat_identity_digest"],
        allow_synthetic_contract=allow_synthetic_contract,
        controller_epoch=controller_epoch,
    )


@contextmanager
def _control_lock(path: Path, *, exclusive: bool) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _checkpoint_anchor(controller: state.Controller) -> dict[str, Any]:
    status_before = controller.status()
    if not status_before.get("durable_state_ok"):
        reasons = status_before.get("checkpoint_reasons") or []
        raise CliError(f"controller durable state is not green: {reasons}")
    try:
        checkpoint_before = read_sealed_json(controller.checkpoint_path)
    except Glm52Error as exc:
        raise CliError(str(exc)) from exc
    status_after = controller.status()
    try:
        checkpoint_after = read_sealed_json(controller.checkpoint_path)
    except Glm52Error as exc:
        raise CliError(str(exc)) from exc
    if status_before != status_after or checkpoint_before != checkpoint_after:
        raise CliError("controller state changed while the operator request was anchored")
    return {
        "state": checkpoint_before.get("state"),
        "blocked_from": checkpoint_before.get("blocked_from"),
        "active_window_id": checkpoint_before.get("active_window_id"),
        "event_count": checkpoint_before.get("event_count"),
        "event_head_hash": checkpoint_before.get("event_head_hash"),
        "window_event_count": checkpoint_before.get("window_event_count"),
        "window_event_head_hash": checkpoint_before.get("window_event_head_hash"),
        "checkpoint_seal_sha256": checkpoint_before.get("seal_sha256"),
    }


class OperatorControlJournal:
    """Authenticated, serialized operator requests; latest valid event is authoritative."""

    _PAYLOAD_KEYS = {
        "schema",
        "campaign_id",
        "source_revision",
        "controller_epoch",
        "expected_contract_sha256",
        "sequence",
        "action",
        "previous_action",
        "requested_at",
        "controller_anchor",
        "claim_id",
        "auth_hmac_sha256",
    }
    _ANCHOR_KEYS = {
        "state",
        "blocked_from",
        "active_window_id",
        "event_count",
        "event_head_hash",
        "window_event_count",
        "window_event_head_hash",
        "checkpoint_seal_sha256",
    }
    _LEGAL = {
        ACTION_RUN: frozenset({ACTION_PAUSE, ACTION_STOP}),
        ACTION_PAUSE: frozenset({ACTION_RUN, ACTION_STOP}),
        ACTION_STOP: frozenset({ACTION_RUN}),
    }

    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self.log = state.HashChainLog(runtime.control_log_path, schema=CONTROL_EVENT_SCHEMA)

    def _verify_payload(
        self, payload: Any, *, event_seq: int, expected_previous: str
    ) -> dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) != self._PAYLOAD_KEYS:
            raise CliError(f"operator request {event_seq} schema invalid")
        value = json.loads(canonical(payload).decode("utf-8"))
        if value.get("schema") != CONTROL_PAYLOAD_SCHEMA:
            raise CliError(f"operator request {event_seq} payload schema mismatch")
        if value.get("campaign_id") != self.runtime.campaign_id \
                or value.get("source_revision") != self.runtime.source_revision \
                or value.get("controller_epoch") != self.runtime.controller_epoch \
                or value.get("expected_contract_sha256") != \
                self.runtime.expected_contract.get("seal_sha256"):
            raise CliError(f"operator request {event_seq} campaign identity mismatch")
        if value.get("sequence") != event_seq or isinstance(value.get("sequence"), bool):
            raise CliError(f"operator request {event_seq} sequence mismatch")
        action = value.get("action")
        if action not in CONTROL_ACTIONS or value.get("previous_action") != expected_previous:
            raise CliError(f"operator request {event_seq} action chain mismatch")
        if action == expected_previous or action not in self._LEGAL[expected_previous]:
            raise CliError(f"operator request {event_seq} contains an illegal/redundant transition")
        if not isinstance(value.get("requested_at"), str) or not value["requested_at"]:
            raise CliError(f"operator request {event_seq} requested_at invalid")
        if value.get("claim_id") != f"operator:{event_seq:08d}:{str(action).lower()}":
            raise CliError(f"operator request {event_seq} claim identity mismatch")
        anchor = value.get("controller_anchor")
        if not isinstance(anchor, dict) or set(anchor) != self._ANCHOR_KEYS:
            raise CliError(f"operator request {event_seq} controller anchor invalid")
        if anchor.get("state") not in state.STATES:
            raise CliError(f"operator request {event_seq} controller state invalid")
        for key in ("event_count", "window_event_count"):
            if isinstance(anchor.get(key), bool) or not isinstance(anchor.get(key), int) \
                    or anchor[key] < 0:
                raise CliError(f"operator request {event_seq} {key} invalid")
        for key in ("event_head_hash", "window_event_head_hash", "checkpoint_seal_sha256"):
            if not isinstance(anchor.get(key), str) or _SHA256_RE.fullmatch(anchor[key]) is None:
                raise CliError(f"operator request {event_seq} {key} invalid")
        signature = value.pop("auth_hmac_sha256")
        if not self.runtime.telegram_auth.verify(value, signature):
            raise CliError(f"operator request {event_seq} HMAC authentication failed")
        value["auth_hmac_sha256"] = signature
        return value

    def verified_requests(self) -> list[dict[str, Any]]:
        try:
            events = self.log.verified_events()
        except state.StateError as exc:
            raise CliError(str(exc)) from exc
        current = ACTION_RUN
        requests: list[dict[str, Any]] = []
        for index, event in enumerate(events):
            if event.get("kind") != "OPERATOR_COMMAND":
                raise CliError(f"operator event {index} kind invalid")
            payload = self._verify_payload(
                event.get("payload"), event_seq=index, expected_previous=current
            )
            if event.get("claim_id") != payload["claim_id"]:
                raise CliError(f"operator event {index} outer/inner claim mismatch")
            current = payload["action"]
            requests.append({
                "sequence": index,
                "action": current,
                "previous_action": payload["previous_action"],
                "requested_at": payload["requested_at"],
                "controller_anchor": payload["controller_anchor"],
                "claim_id": payload["claim_id"],
                "event_chain_sha256": event["chain_sha256"],
            })
        return requests

    def replay(self) -> dict[str, Any]:
        requests = self.verified_requests()
        last = requests[-1] if requests else None
        current = last["action"] if last else ACTION_RUN
        return {
            "journal_ok": True,
            "event_count": len(requests),
            "head_hash": (
                last["event_chain_sha256"] if last else state.GENESIS_HASH
            ),
            "effective_action": current,
            "requested_action": current,
            "requested_sequence": last["sequence"] if last else -1,
            "last_request": last,
        }

    @staticmethod
    def _validate_for_controller_state(action: str, controller_state: str) -> None:
        if controller_state == "COMPLETE":
            raise CliError("operator control is closed after COMPLETE")
        if action == ACTION_PAUSE and controller_state not in state.WINDOW_STATES | {
            "FREEZE_PROGRAM"
        }:
            raise CliError(
                "pause-after-window is legal only after FREEZE_PROGRAM enters the window campaign"
            )
        if action == ACTION_RUN and controller_state == "BLOCKED":
            raise CliError("operator resume cannot bypass the BLOCKED resolution transition")

    def issue(self, action: str, controller: state.Controller) -> dict[str, Any]:
        if action not in CONTROL_ACTIONS:
            raise CliError(f"unknown operator action: {action}")
        with _control_lock(self.runtime.control_lock_path, exclusive=True):
            anchor = _checkpoint_anchor(controller)
            snapshot = self.replay()
            current = snapshot["effective_action"]
            self._validate_for_controller_state(action, str(anchor["state"]))
            if action == current:
                return {**snapshot, "changed": False, "idempotent": True}
            if action not in self._LEGAL[current]:
                raise CliError(f"illegal operator transition: {current} -> {action}")
            sequence = int(snapshot["event_count"])
            claim_id = f"operator:{sequence:08d}:{action.lower()}"
            payload: dict[str, Any] = {
                "schema": CONTROL_PAYLOAD_SCHEMA,
                "campaign_id": self.runtime.campaign_id,
                "source_revision": self.runtime.source_revision,
                "controller_epoch": self.runtime.controller_epoch,
                "expected_contract_sha256": self.runtime.expected_contract["seal_sha256"],
                "sequence": sequence,
                "action": action,
                "previous_action": current,
                "requested_at": utc_now(),
                "controller_anchor": anchor,
                "claim_id": claim_id,
            }
            payload["auth_hmac_sha256"] = self.runtime.telegram_auth.authenticate(payload)
            try:
                self.log.append("OPERATOR_COMMAND", claim_id, payload)
            except state.StateError as exc:
                raise CliError(str(exc)) from exc
            updated = self.replay()
            return {**updated, "changed": True, "idempotent": False}


class WorkerControlAcknowledgements:
    """Worker-only acknowledgements proving when a requested action took effect.

    Merely appending an operator request never claims that a worker observed it.
    The long-running worker must hold the authoritative controller lease and call
    :meth:`poll` at every declared safe point.  STOP/RUN apply at any safe point;
    PAUSE_AFTER_WINDOW applies only before a new fetch or after/between windows.
    """

    _PAYLOAD_KEYS = {
        "schema",
        "campaign_id",
        "source_revision",
        "controller_epoch",
        "expected_contract_sha256",
        "applied_sequence",
        "request_sequence",
        "action",
        "safe_point",
        "applied_at",
        "request_event_chain_sha256",
        "controller_anchor",
        "previous_applied_request_sequence",
        "previous_applied_action",
        "claim_id",
        "auth_hmac_sha256",
    }

    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self.control = OperatorControlJournal(runtime)
        self.log = state.HashChainLog(
            runtime.control_applied_log_path,
            schema=CONTROL_APPLIED_EVENT_SCHEMA,
        )

    def _verify_payload(
        self,
        payload: Any,
        *,
        applied_sequence: int,
        previous_request_sequence: int,
        previous_action: str,
        requests: Mapping[int, Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) != self._PAYLOAD_KEYS:
            raise CliError(f"worker acknowledgement {applied_sequence} schema invalid")
        value = json.loads(canonical(payload).decode("utf-8"))
        if value.get("schema") != CONTROL_APPLIED_PAYLOAD_SCHEMA:
            raise CliError(
                f"worker acknowledgement {applied_sequence} payload schema mismatch"
            )
        if value.get("campaign_id") != self.runtime.campaign_id \
                or value.get("source_revision") != self.runtime.source_revision \
                or value.get("controller_epoch") != self.runtime.controller_epoch \
                or value.get("expected_contract_sha256") != \
                self.runtime.expected_contract.get("seal_sha256"):
            raise CliError(
                f"worker acknowledgement {applied_sequence} campaign identity mismatch"
            )
        if value.get("applied_sequence") != applied_sequence \
                or isinstance(value.get("applied_sequence"), bool):
            raise CliError(f"worker acknowledgement {applied_sequence} sequence mismatch")
        request_sequence = value.get("request_sequence")
        if isinstance(request_sequence, bool) or not isinstance(request_sequence, int) \
                or request_sequence <= previous_request_sequence:
            raise CliError(
                f"worker acknowledgement {applied_sequence} request sequence is stale"
            )
        request = requests.get(request_sequence)
        if request is None:
            raise CliError(
                f"worker acknowledgement {applied_sequence} references no operator request"
            )
        action = value.get("action")
        if action != request.get("action") or action not in CONTROL_ACTIONS:
            raise CliError(
                f"worker acknowledgement {applied_sequence} action/request mismatch"
            )
        if value.get("request_event_chain_sha256") != request.get(
            "event_chain_sha256"
        ):
            raise CliError(
                f"worker acknowledgement {applied_sequence} request hash mismatch"
            )
        if value.get("previous_applied_request_sequence") != previous_request_sequence \
                or value.get("previous_applied_action") != previous_action:
            raise CliError(
                f"worker acknowledgement {applied_sequence} application chain mismatch"
            )
        safe_point = value.get("safe_point")
        if safe_point not in WORKER_SAFE_POINTS:
            raise CliError(
                f"worker acknowledgement {applied_sequence} safe point invalid"
            )
        if action == ACTION_PAUSE and safe_point not in PAUSE_SAFE_POINTS:
            raise CliError(
                f"worker acknowledgement {applied_sequence} applied pause mid-window"
            )
        if not isinstance(value.get("applied_at"), str) or not value["applied_at"]:
            raise CliError(
                f"worker acknowledgement {applied_sequence} applied_at invalid"
            )
        expected_claim = (
            f"operator-applied:{applied_sequence:08d}:"
            f"{request_sequence:08d}:{str(action).lower()}"
        )
        if value.get("claim_id") != expected_claim:
            raise CliError(
                f"worker acknowledgement {applied_sequence} claim identity mismatch"
            )
        anchor = value.get("controller_anchor")
        if not isinstance(anchor, dict) \
                or set(anchor) != OperatorControlJournal._ANCHOR_KEYS:
            raise CliError(
                f"worker acknowledgement {applied_sequence} controller anchor invalid"
            )
        signature = value.pop("auth_hmac_sha256")
        if not self.runtime.telegram_auth.verify(value, signature):
            raise CliError(
                f"worker acknowledgement {applied_sequence} HMAC authentication failed"
            )
        value["auth_hmac_sha256"] = signature
        return value

    def replay(self) -> dict[str, Any]:
        requests_list = self.control.verified_requests()
        requests = {int(row["sequence"]): row for row in requests_list}
        try:
            events = self.log.verified_events()
        except state.StateError as exc:
            raise CliError(str(exc)) from exc
        previous_request_sequence = -1
        previous_action = ACTION_RUN
        last: dict[str, Any] | None = None
        for index, event in enumerate(events):
            if event.get("kind") != "OPERATOR_ACTION_APPLIED":
                raise CliError(f"worker acknowledgement event {index} kind invalid")
            payload = self._verify_payload(
                event.get("payload"),
                applied_sequence=index,
                previous_request_sequence=previous_request_sequence,
                previous_action=previous_action,
                requests=requests,
            )
            if event.get("claim_id") != payload["claim_id"]:
                raise CliError(
                    f"worker acknowledgement {index} outer/inner claim mismatch"
                )
            previous_request_sequence = int(payload["request_sequence"])
            previous_action = str(payload["action"])
            last = {
                "applied_sequence": index,
                "request_sequence": previous_request_sequence,
                "action": previous_action,
                "safe_point": payload["safe_point"],
                "applied_at": payload["applied_at"],
                "controller_anchor": payload["controller_anchor"],
                "request_event_chain_sha256": payload[
                    "request_event_chain_sha256"
                ],
                "claim_id": payload["claim_id"],
                "event_chain_sha256": event["chain_sha256"],
            }
        return {
            "journal_ok": True,
            "event_count": len(events),
            "head_hash": self.log.head_hash(events),
            "applied_action": previous_action,
            "applied_request_sequence": previous_request_sequence,
            "last_applied": last,
        }

    @staticmethod
    def _directive(action: str) -> str:
        return {
            ACTION_RUN: "CONTINUE",
            ACTION_PAUSE: "PAUSE",
            ACTION_STOP: "STOP",
        }[action]

    def poll(self, controller: state.Controller, *, safe_point: str) -> dict[str, Any]:
        """Apply the newest request at an allowed worker safe point.

        The caller must be the live worker holding the controller lease.  A pause
        observed mid-window remains REQUESTED until the after-window boundary;
        STOP never waits for that boundary.
        """
        if safe_point not in WORKER_SAFE_POINTS:
            raise CliError(f"unknown worker control safe point: {safe_point}")
        controller.lease.assert_held()
        with _control_lock(self.runtime.control_lock_path, exclusive=True):
            anchor = _checkpoint_anchor(controller)
            control = self.control.replay()
            applied = self.replay()
            request = control["last_request"]
            if request is None \
                    or int(request["sequence"]) <= int(
                        applied["applied_request_sequence"]
                    ):
                return {
                    **applied,
                    "application_state": "IN_SYNC",
                    "worker_directive": self._directive(applied["applied_action"]),
                    "changed": False,
                }
            requested_action = str(request["action"])
            if requested_action == ACTION_PAUSE and safe_point not in PAUSE_SAFE_POINTS:
                return {
                    **applied,
                    "application_state": "REQUESTED_PENDING_SAFE_POINT",
                    "requested_action": requested_action,
                    "requested_sequence": request["sequence"],
                    "worker_directive": "CONTINUE_CURRENT_WINDOW",
                    "changed": False,
                }
            applied_sequence = int(applied["event_count"])
            request_sequence = int(request["sequence"])
            claim_id = (
                f"operator-applied:{applied_sequence:08d}:"
                f"{request_sequence:08d}:{requested_action.lower()}"
            )
            payload: dict[str, Any] = {
                "schema": CONTROL_APPLIED_PAYLOAD_SCHEMA,
                "campaign_id": self.runtime.campaign_id,
                "source_revision": self.runtime.source_revision,
                "controller_epoch": self.runtime.controller_epoch,
                "expected_contract_sha256": self.runtime.expected_contract[
                    "seal_sha256"
                ],
                "applied_sequence": applied_sequence,
                "request_sequence": request_sequence,
                "action": requested_action,
                "safe_point": safe_point,
                "applied_at": utc_now(),
                "request_event_chain_sha256": request["event_chain_sha256"],
                "controller_anchor": anchor,
                "previous_applied_request_sequence": applied[
                    "applied_request_sequence"
                ],
                "previous_applied_action": applied["applied_action"],
                "claim_id": claim_id,
            }
            payload["auth_hmac_sha256"] = self.runtime.telegram_auth.authenticate(
                payload
            )
            try:
                self.log.append("OPERATOR_ACTION_APPLIED", claim_id, payload)
            except state.StateError as exc:
                raise CliError(str(exc)) from exc
            updated = self.replay()
            return {
                **updated,
                "application_state": "IN_SYNC",
                "worker_directive": self._directive(updated["applied_action"]),
                "changed": True,
            }


def _controller_phone_snapshot(controller: state.Controller) -> tuple[dict[str, Any], Any]:
    first = controller.status()
    checkpoint: dict[str, Any] | None = None
    if first.get("durable_state_ok"):
        try:
            checkpoint = read_sealed_json(controller.checkpoint_path)
        except Glm52Error as exc:
            first = {
                **first,
                "durable_state_ok": False,
                "state": None,
                "checkpoint_replay_ok": False,
                "checkpoint_reasons": [str(exc)],
            }
    second = controller.status()
    if first != second:
        raise CliError("controller changed while phone status was being sampled")
    if checkpoint is not None:
        try:
            if checkpoint != read_sealed_json(controller.checkpoint_path):
                raise CliError("checkpoint changed while phone status was being sampled")
        except Glm52Error as exc:
            raise CliError(str(exc)) from exc
    return first, checkpoint


def build_phone_status(runtime: Runtime) -> dict[str, Any]:
    controller = runtime.controller()
    controller_status, checkpoint = _controller_phone_snapshot(controller)
    journal = OperatorControlJournal(runtime)
    try:
        with _control_lock(runtime.control_lock_path, exclusive=False):
            control = journal.replay()
            applied = WorkerControlAcknowledgements(runtime).replay()
        control_error = None
    except CliError as exc:
        control = {
            "journal_ok": False,
            "event_count": None,
            "head_hash": None,
            "effective_action": ACTION_STOP,
            "requested_action": ACTION_STOP,
            "requested_sequence": None,
            "last_request": None,
        }
        applied = {
            "journal_ok": False,
            "event_count": None,
            "head_hash": None,
            "applied_action": ACTION_STOP,
            "applied_request_sequence": None,
            "last_applied": None,
        }
        control_error = str(exc)
    in_sync = bool(
        control.get("journal_ok")
        and applied.get("journal_ok")
        and control.get("requested_sequence")
        == applied.get("applied_request_sequence")
        and control.get("requested_action") == applied.get("applied_action")
    )
    control = {
        **control,
        "application_state": "IN_SYNC" if in_sync else "REQUESTED_NOT_APPLIED",
        "applied": applied,
    }
    # Re-sample the controller to prevent a green snapshot composed across a state change.
    controller_after, checkpoint_after = _controller_phone_snapshot(controller)
    if controller_status != controller_after or checkpoint != checkpoint_after:
        raise CliError("controller changed while phone and operator status were composed")
    structurally_green = bool(
        controller_status.get("durable_state_ok")
        and controller_status.get("live_worker_lease_ok")
        and controller_status.get("heartbeat_fresh_ok")
        and control.get("journal_ok")
        and applied.get("journal_ok")
    )
    overall_status = (
        "GREEN" if structurally_green and in_sync
        else "AMBER_CONTROL_PENDING" if structurally_green
        else "RED"
    )
    document = {
        "schema": PHONE_STATUS_SCHEMA,
        "campaign_id": runtime.campaign_id,
        "source_revision": runtime.source_revision,
        "controller_epoch": runtime.controller_epoch,
        "expected_contract_sha256": runtime.expected_contract["seal_sha256"],
        "contract_phone_status_path": runtime.contract_phone_status_path,
        "cli_config_sha256": runtime.config_sha256,
        "status": overall_status,
        "overall_status": overall_status,
        "controller": controller_status,
        "operator_control": control,
        "operator_control_error": control_error,
        "checkpoint_anchor": {
            "event_count": checkpoint.get("event_count"),
            "event_head_hash": checkpoint.get("event_head_hash"),
            "window_event_count": checkpoint.get("window_event_count"),
            "window_event_head_hash": checkpoint.get("window_event_head_hash"),
            "checkpoint_seal_sha256": checkpoint.get("seal_sha256"),
        } if checkpoint else None,
        "heartbeat": checkpoint.get("heartbeat") if checkpoint else None,
        "telegram": checkpoint.get("telegram") if checkpoint else None,
        "window_summary": checkpoint.get("window_summary") if checkpoint else None,
        "window_resume_plan": checkpoint.get("window_resume_plan") if checkpoint else None,
        "campaign_completeness": checkpoint.get("campaign_completeness") if checkpoint else None,
    }
    try:
        return state.seal_producer_authenticated_evidence(
            document, auth=runtime.evidence_auth
        )
    except state.StateError as exc:
        raise CliError(f"phone-status producer authentication failed: {exc}") from exc


def _render_phone_markdown(document: Mapping[str, Any]) -> str:
    controller = document["controller"]
    control = document["operator_control"]
    completeness = document.get("campaign_completeness") or {}
    source = completeness.get("source") or {}
    tensors = completeness.get("tensors") or {}
    checkpoint_anchor = document.get("checkpoint_anchor") or {}
    lines = [
        "# GLM-5.2 Phone Status",
        "",
        f"- Overall: `{document['overall_status']}`",
        f"- Controller state: `{controller.get('state') or 'UNTRUSTED'}`",
        f"- Durable state: `{'PASS' if controller.get('durable_state_ok') else 'FAIL'}`",
        f"- Live worker lease: `{'PASS' if controller.get('live_worker_lease_ok') else 'FAIL'}`",
        f"- Heartbeat fresh: `{'PASS' if controller.get('heartbeat_fresh_ok') else 'FAIL'}`",
        f"- Heartbeat at: `{controller.get('heartbeat_at') or 'NONE'}`",
        f"- Operator request: `{control.get('effective_action')}`",
        f"- Operator applied: `{(control.get('applied') or {}).get('applied_action')}`",
        f"- Control application: `{control.get('application_state')}`",
        f"- Source coverage complete: `{source.get('status') == 'COMPLETE'}`",
        f"- Tensor coverage complete: `{tensors.get('status') == 'COMPLETE'}`",
        f"- Active/next window: `{(document.get('window_resume_plan') or {}).get('earliest_incomplete_window_id') or 'NONE'}`",
        f"- Controller events: `{checkpoint_anchor.get('event_count', 'N/A')}`",
        f"- Window events: `{checkpoint_anchor.get('window_event_count', 'N/A')}`",
        f"- Control events: `{control.get('event_count')}`",
        f"- Status seal: `{document['seal_sha256']}`",
        "",
        "This surface distinguishes requested from worker-applied control; the worker "
        "must acknowledge requests at a sealed safe point.",
        "",
    ]
    return "\n".join(lines)


def _validate_existing_phone_status(
    runtime: Runtime, document: Mapping[str, Any], markdown: str
) -> None:
    """Validate the authoritative JSON commit record.

    Markdown is a deterministic projection, not an independent authority.  A
    crash after the atomic JSON replacement but before the Markdown replacement
    is therefore repaired on the next call instead of being misclassified as
    tampering.  A malformed or foreign JSON commit record still fails closed.
    """
    json_exists = runtime.phone_json_path.exists()
    if json_exists:
        try:
            previous = read_sealed_json(runtime.phone_json_path)
        except Glm52Error as exc:
            raise CliError(f"refusing to overwrite invalid phone status: {exc}") from exc
        producer_body = {
            key: item
            for key, item in previous.items()
            if key not in {"seal_sha256", "producer_hmac_sha256"}
        }
        if not runtime.evidence_auth.verify(
            {
                "schema": "hawking.glm52.evidence_producer_auth.v1",
                "artifact": producer_body,
            },
            previous.get("producer_hmac_sha256"),
        ):
            raise CliError(
                "refusing to overwrite phone status with invalid producer authentication"
            )
        if previous.get("schema") != PHONE_STATUS_SCHEMA \
                or previous.get("campaign_id") != runtime.campaign_id \
                or previous.get("source_revision") != runtime.source_revision \
                or previous.get("expected_contract_sha256") != \
                runtime.expected_contract["seal_sha256"] \
                or previous.get("cli_config_sha256") != runtime.config_sha256:
            raise CliError("refusing to overwrite phone status for a different identity")


def _write_phone_status_locked(
    runtime: Runtime, document: dict[str, Any], markdown: str
) -> None:
    _validate_existing_phone_status(runtime, document, markdown)
    # The sealed JSON replacement is the transaction commit.  Markdown follows
    # as a rebuildable view; recovery safely rewrites it from JSON if a crash
    # separates these two file replacements.
    atomic_json(runtime.phone_json_path, document)
    atomic_text(runtime.phone_markdown_path, markdown)
    try:
        written = read_sealed_json(runtime.phone_json_path)
    except Glm52Error as exc:
        raise CliError(str(exc)) from exc
    if written != document or runtime.phone_markdown_path.read_text(encoding="utf-8") != markdown:
        raise CliError("phone-status post-write verification failed")


def write_phone_status(runtime: Runtime) -> dict[str, Any]:
    with _control_lock(runtime.phone_lock_path, exclusive=True):
        document = build_phone_status(runtime)
        markdown = _render_phone_markdown(document)
        _write_phone_status_locked(runtime, document, markdown)
    return document


def run_command(runtime: Runtime, command: str) -> dict[str, Any]:
    if command == "status":
        return write_phone_status(runtime)
    action = {
        "pause-after-window": ACTION_PAUSE,
        "resume": ACTION_RUN,
        "stop": ACTION_STOP,
    }.get(command)
    if action is None:
        raise CliError(f"unknown command: {command}")
    controller = runtime.controller()
    # Serialize phone preflight, durable intent, and derived-status replacement.  This
    # prevents a known-invalid existing phone artifact from causing a hidden partial
    # command after the operator journal has already advanced.
    with _control_lock(runtime.phone_lock_path, exclusive=True):
        before = build_phone_status(runtime)
        _validate_existing_phone_status(runtime, before, _render_phone_markdown(before))
        result = OperatorControlJournal(runtime).issue(action, controller)
        phone = build_phone_status(runtime)
        _write_phone_status_locked(runtime, phone, _render_phone_markdown(phone))
    return {
        "schema": "hawking.glm52.operator_command_result.v1",
        "command": command,
        "changed": result["changed"],
        "idempotent": result["idempotent"],
        "effective_action": result["effective_action"],
        "requested_action": result["requested_action"],
        "requested_sequence": result["requested_sequence"],
        "applied_action": phone["operator_control"]["applied"]["applied_action"],
        "applied_request_sequence": phone["operator_control"]["applied"][
            "applied_request_sequence"
        ],
        "application_state": phone["operator_control"]["application_state"],
        "control_event_count": result["event_count"],
        "control_head_hash": result["head_hash"],
        "phone_status": phone,
    }


def _default_config_path(environ: Mapping[str, str]) -> Path:
    configured = environ.get("GLM52_CONTROLLER_CONFIG")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "GLM52_CONTROLLER_CONFIG.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="sealed CLI configuration (or set GLM52_CONTROLLER_CONFIG)",
    )
    parser.add_argument(
        "command",
        choices=("status", "pause-after-window", "resume", "stop"),
    )
    return parser


def main(
    argv: Sequence[str] | None = None, *,
    environ: Mapping[str, str] | None = None,
    keychain: telegram_module.Keychain | None = None,
    evidence_keychain: evidence_module.Keychain | None = None,
    grounding_keychain: grounding_module.Keychain | None = None,
) -> int:
    env = os.environ if environ is None else environ
    args = build_parser().parse_args(argv)
    config_path = args.config or _default_config_path(env)
    try:
        runtime = load_runtime(
            config_path,
            keychain=keychain,
            evidence_keychain=evidence_keychain,
            grounding_keychain=grounding_keychain,
        )
        result = run_command(runtime, args.command)
    except (CliError, state.StateError, Glm52Error, OSError) as exc:
        print(
            json.dumps(
                {"status": "ERROR", "error": type(exc).__name__, "message": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.command == "status" and result.get("overall_status") != "GREEN":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

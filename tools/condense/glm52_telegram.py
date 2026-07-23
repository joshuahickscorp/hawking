#!/usr/bin/env python3.12
"""Secure Telegram credentials and delivery for the GLM-5.2 campaign.

Secrets live in three GLM-specific macOS Keychain items.  They are never
accepted through command-line arguments or environment variables, and public
status values contain only booleans and domain-separated SHA-256 digests.

The sender is intentionally explicit: callers supply a complete, controller-
authenticated transition intent.  Telegram must echo its exact canonical text,
including the dedupe key, before ``glm52_state`` will authenticate a delivery
receipt.  Network and Keychain implementations are injectable so the entire
security contract can be tested without touching the real Keychain or Telegram.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import getpass
import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
import threading
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from glm52_common import Glm52Error, canonical, utc_now
from glm52_evidence_auth import (
    _get_generic_password_native,
    _set_generic_password_native,
)
from glm52_state import (
    StateError,
    TELEGRAM_RECEIPT_SCHEMA,
    TelegramAuthConfig,
    make_telegram_delivery_receipt,
    telegram_chat_identity_digest,
    validate_telegram_delivery_receipt as _state_validate_telegram_delivery_receipt,
    validate_transition_intent as _state_validate_transition_intent,
)


KEYCHAIN_ACCOUNT = "hawking-glm52-gravity"
TOKEN_SERVICE = "com.hawking.glm52.gravity.telegram.bot-token"
CHAT_SERVICE = "com.hawking.glm52.gravity.telegram.private-chat-id"
HMAC_SERVICE = "com.hawking.glm52.gravity.telegram.receipt-hmac-key"
KEYCHAIN_SERVICES = (TOKEN_SERVICE, CHAT_SERVICE, HMAC_SERVICE)

TOKEN_RE = re.compile(r"^[0-9]{6,16}:[A-Za-z0-9_-]{20,}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,127}$")
CLAIM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{7,255}$")
RATE_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_METHODS = frozenset({"getMe", "getUpdates", "sendMessage"})
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_MESSAGE_CHARS = 4096
HMAC_KEY_BYTES = 32
DELIVERY_LEDGER_SCHEMA = "hawking.glm52.telegram_delivery_ledger_entry.v1"
DELIVERY_LEDGER_HEAD_SCHEMA = "hawking.glm52.telegram_delivery_ledger_head.v1"
DELIVERY_BINDING_SCHEMA = "hawking.glm52.telegram_delivery_binding.v1"
LEDGER_GENESIS_HASH = "0" * 64
MAX_LEDGER_BYTES = 64 * 1024 * 1024
NORMAL_DELIVERY_LIFECYCLE = (
    "OUTBOX_PREPARED",
    "SEND_STARTED",
    "DELIVERY_RECEIPT",
)
LEDGER_ENTRY_KINDS = frozenset({
    *NORMAL_DELIVERY_LIFECYCLE,
    "AMBIGUOUS_BLOCKED",
    "DUPLICATE_RETRY_AUTHORIZED",
})
RECONCILIATION_AUTH_SCHEMA = "hawking.glm52.telegram_reconciliation_authorization.v1"
AMBIGUITY_BLOCK_SCHEMA = "hawking.glm52.telegram_ambiguity_block.v1"

STATUS_KEYS = frozenset({
    "state",
    "source_coverage_percent",
    "shards",
    "network_bytes",
    "throughput_bytes_per_second",
    "eta_seconds",
    "current",
    "candidate_rates",
    "best_metrics",
    "resources",
    "process",
})
SHARD_KEYS = frozenset({"fetched", "verified", "evicted", "total"})
CURRENT_KEYS = frozenset({"window", "layer"})
RESOURCE_KEYS = frozenset({"disk_free_bytes", "ram_available_bytes", "swap_used_bytes"})
PROCESS_KEYS = frozenset({"pid", "lease_held", "lease_owner"})
_DELIVERY_BINDING_KEYS = frozenset({
    "schema",
    "transition_intent",
    "transition_intent_sha256",
    "event_kind",
    "claim_id",
    "from_state",
    "to_state",
    "dedupe_key",
    "canonical_status",
    "canonical_status_sha256",
    "rendered_message",
    "rendered_message_sha256",
    "rendered_message_utf8_bytes",
    "controller_anchor",
    "controller_anchor_sha256",
})
_LEDGER_ENTRY_KEYS = frozenset({
    "schema",
    "seq",
    "kind",
    "recorded_at",
    "dedupe_key",
    "binding",
    "receipt",
    "reconciliation",
    "prev_chain_sha256",
    "chain_sha256",
    "hmac_sha256",
})
_LEDGER_HEAD_KEYS = frozenset({
    "schema",
    "entry_count",
    "head_chain_sha256",
    "head_entry_hmac_sha256",
    "hmac_sha256",
})
_RECONCILIATION_AUTH_KEYS = frozenset({
    "schema",
    "status",
    "action",
    "authorization_claim_id",
    "transition_intent_sha256",
    "dedupe_key",
    "canonical_status_sha256",
    "rendered_message_sha256",
    "controller_anchor_sha256",
    "ambiguous_entry_seq",
    "ambiguous_entry_chain_sha256",
    "send_started_seq",
    "send_started_chain_sha256",
    "attempt",
    "reason",
    "authorized_at",
    "seal_sha256",
    "hmac_sha256",
})
_AMBIGUITY_BLOCK_KEYS = frozenset({
    "schema",
    "status",
    "attempt",
    "send_started_seq",
    "send_started_chain_sha256",
    "reason",
})

_LIFECYCLE_NEXT_KINDS: dict[str | None, frozenset[str]] = {
    None: frozenset({"OUTBOX_PREPARED"}),
    "OUTBOX_PREPARED": frozenset({"SEND_STARTED"}),
    "SEND_STARTED": frozenset({"AMBIGUOUS_BLOCKED", "DELIVERY_RECEIPT"}),
    "AMBIGUOUS_BLOCKED": frozenset({
        "DUPLICATE_RETRY_AUTHORIZED", "DELIVERY_RECEIPT",
    }),
    "DUPLICATE_RETRY_AUTHORIZED": frozenset({"SEND_STARTED"}),
    "DELIVERY_RECEIPT": frozenset(),
}

_LEDGER_LOCKS_GUARD = threading.Lock()
_LEDGER_LOCKS: dict[str, threading.Lock] = {}


class TelegramSecurityError(Glm52Error):
    """A secret, identity, Bot API, or message-binding gate failed."""


class Keychain(Protocol):
    def get(self, service: str) -> str | None:
        """Return one credential or ``None`` without exposing it elsewhere."""

    def set(self, service: str, value: str) -> None:
        """Store one credential without placing it in process arguments."""


@dataclass(frozen=True)
class TelegramHTTPResponse:
    status: int
    body: Mapping[str, Any]


class TelegramTransport(Protocol):
    def call(
        self, token: str, method: str, payload: Mapping[str, Any]
    ) -> TelegramHTTPResponse:
        """Call one Bot API method; implementations must redact the token."""


class MacOSKeychain:
    """In-process Security.framework adapter for Telegram credentials.

    ``security add-generic-password -w`` does not consume the password from
    stdin when no argument follows ``-w``; it can silently store an empty
    password.  Passing the value after ``-w`` would expose it in argv.  Both
    reads and writes therefore use Security.framework in this process, so a
    credential never crosses an argv, environment, stdin, or subprocess
    boundary.
    """

    def __init__(
        self,
        *,
        native_reader: Callable[[str, str], str | None] | None = None,
        native_writer: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self._native_reader = native_reader or _get_generic_password_native
        self._native_writer = native_writer or _set_generic_password_native

    @staticmethod
    def _validate_service(service: str) -> str:
        if service not in KEYCHAIN_SERVICES:
            raise TelegramSecurityError("unrecognized GLM Telegram Keychain service")
        return service

    def get(self, service: str) -> str | None:
        service = self._validate_service(service)
        try:
            value = self._native_reader(service, KEYCHAIN_ACCOUNT)
        except Exception:
            raise TelegramSecurityError("macOS Keychain read failed") from None
        if value is None:
            return None
        if not value or "\n" in value or "\r" in value:
            raise TelegramSecurityError("macOS Keychain returned an invalid credential")
        return value

    def set(self, service: str, value: str) -> None:
        service = self._validate_service(service)
        _validate_keychain_value(value)
        try:
            self._native_writer(service, KEYCHAIN_ACCOUNT, value)
        except Exception:
            raise TelegramSecurityError("macOS Keychain write failed") from None


class UrllibTelegramTransport:
    """Small Bot API transport with token-redacted failures."""

    def __init__(
        self,
        *,
        opener: Callable[..., Any] = urllib.request.urlopen,
        timeout_seconds: int = 20,
    ) -> None:
        self._opener = opener
        self._timeout = timeout_seconds

    def call(
        self, token: str, method: str, payload: Mapping[str, Any]
    ) -> TelegramHTTPResponse:
        _validate_token(token)
        if method not in ALLOWED_METHODS:
            raise TelegramSecurityError("unrecognized Telegram Bot API method")
        encoded = urllib.parse.urlencode(dict(payload)).encode("utf-8")
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/{method}",
            data=encoded,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "hawking-glm52-gravity/1",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._timeout) as response:
                status = int(response.getcode())
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except Exception:
            raise TelegramSecurityError(f"Telegram {method} transport failed") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise TelegramSecurityError(f"Telegram {method} response exceeded the safe limit")
        try:
            body = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise TelegramSecurityError(f"Telegram {method} returned invalid JSON") from None
        if not isinstance(body, dict):
            raise TelegramSecurityError(f"Telegram {method} returned a non-object response")
        return TelegramHTTPResponse(status=status, body=body)


@dataclass(frozen=True, repr=False)
class _SenderCredentials:
    token: str
    chat_id: str
    auth: TelegramAuthConfig

    def __repr__(self) -> str:
        return "_SenderCredentials(token=<redacted>, chat_id=<redacted>, auth=<redacted>)"


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


class _RedactingArgumentParser(argparse.ArgumentParser):
    """Reject bad CLI input without reflecting a mistakenly pasted secret."""

    def error(self, _message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: invalid command-line arguments\n")


def _validate_keychain_value(value: Any) -> str:
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value \
            or "\x00" in value:
        raise TelegramSecurityError("credential must be a non-empty single-line value")
    return value


def _validate_token(token: Any) -> str:
    if not isinstance(token, str) or TOKEN_RE.fullmatch(token) is None:
        raise TelegramSecurityError("Telegram bot credential shape is invalid")
    return token


def _parse_private_chat_id(value: Any) -> str:
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise TelegramSecurityError("private chat credential is invalid")
    number = int(value)
    if number <= 0 or str(number) != value:
        raise TelegramSecurityError("private chat credential is invalid")
    return value


def _encode_hmac_key(value: bytes) -> str:
    if not isinstance(value, bytes) or len(value) < HMAC_KEY_BYTES:
        raise TelegramSecurityError("receipt authentication key is too short")
    return base64.urlsafe_b64encode(value).decode("ascii")


def _decode_hmac_key(value: Any) -> bytes:
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        raise TelegramSecurityError("receipt authentication credential is invalid")
    try:
        decoded = base64.b64decode(value.encode("ascii"), altchars=b"-_", validate=True)
    except (ValueError, UnicodeEncodeError):
        raise TelegramSecurityError("receipt authentication credential is invalid") from None
    if len(decoded) < HMAC_KEY_BYTES:
        raise TelegramSecurityError("receipt authentication credential is invalid")
    return decoded


def _secret_digest(kind: str, value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(
        canonical({
            "schema": "hawking.glm52.telegram_secret_identity.v1",
            "kind": kind,
            "secret_sha256": hashlib.sha256(raw).hexdigest(),
        })
    ).hexdigest()


def _notification_bot_identity_digest(token: str) -> str:
    """Return the notification verifier's public bot-ID identity from a valid token."""
    valid = _validate_token(token)
    bot_id, separator, _secret = valid.partition(":")
    if separator != ":" or not bot_id.isascii() or not bot_id.isdigit():
        raise TelegramSecurityError("Telegram bot credential shape is invalid")
    return hashlib.sha256(
        b"hawking.glm52.telegram-bot-identity.v1\0" + bot_id.encode("ascii")
    ).hexdigest()


def _bot_identity_digest(identity: Mapping[str, Any]) -> str:
    bot_id = identity.get("id")
    username = identity.get("username")
    if isinstance(bot_id, bool) or not isinstance(bot_id, int) or bot_id <= 0 \
            or identity.get("is_bot") is not True \
            or not isinstance(username, str) or not username:
        raise TelegramSecurityError("Telegram getMe identity is invalid")
    return hashlib.sha256(
        canonical({
            "schema": "hawking.glm52.telegram_bot_identity.v1",
            "bot_id": bot_id,
            "username": username,
        })
    ).hexdigest()


def _keychain_get(keychain: Keychain, service: str) -> str | None:
    if service not in KEYCHAIN_SERVICES:
        raise TelegramSecurityError("unrecognized GLM Telegram Keychain service")
    try:
        return keychain.get(service)
    except Exception:
        raise TelegramSecurityError("Telegram Keychain read failed") from None


def _keychain_set(keychain: Keychain, service: str, value: str) -> None:
    if service not in KEYCHAIN_SERVICES:
        raise TelegramSecurityError("unrecognized GLM Telegram Keychain service")
    try:
        keychain.set(service, value)
    except Exception:
        raise TelegramSecurityError("Telegram Keychain write failed") from None


def _transport_call(
    transport: TelegramTransport,
    token: str,
    method: str,
    payload: Mapping[str, Any],
) -> TelegramHTTPResponse:
    if method not in ALLOWED_METHODS:
        raise TelegramSecurityError("unrecognized Telegram Bot API method")
    try:
        response = transport.call(token, method, payload)
    except Exception:
        raise TelegramSecurityError(f"Telegram {method} transport failed") from None
    if not isinstance(response, TelegramHTTPResponse):
        raise TelegramSecurityError(f"Telegram {method} transport returned an invalid response")
    return response


def _require_ok(response: TelegramHTTPResponse, method: str) -> Any:
    if not isinstance(response, TelegramHTTPResponse) or response.status != 200:
        raise TelegramSecurityError(f"Telegram {method} did not return HTTP 200")
    body = response.body
    if not isinstance(body, Mapping) or body.get("ok") is not True or "result" not in body:
        raise TelegramSecurityError(f"Telegram {method} did not return a validated success")
    return body["result"]


def configure_token(
    keychain: Keychain,
    transport: TelegramTransport,
    *,
    hidden_prompt: Callable[[str], str] = getpass.getpass,
) -> dict[str, bool | str]:
    """Prompt without echo, validate with getMe, then store the token."""
    try:
        entered = hidden_prompt("Paste the rotated BotFather token (input hidden): ")
    except Exception:
        raise TelegramSecurityError("hidden Telegram credential input failed") from None
    token = _validate_token(entered.strip() if isinstance(entered, str) else entered)
    response = _transport_call(transport, token, "getMe", {})
    identity = _require_ok(response, "getMe")
    if not isinstance(identity, Mapping):
        raise TelegramSecurityError("Telegram getMe identity is invalid")
    identity_digest = _bot_identity_digest(identity)
    _keychain_set(keychain, TOKEN_SERVICE, token)
    return {
        "token_configured": True,
        "bot_identity_digest": identity_digest,
    }


def discover_private_chat(
    keychain: Keychain,
    transport: TelegramTransport,
) -> dict[str, bool | str]:
    """Store a chat only when getUpdates identifies one unambiguous human private chat."""
    token = _keychain_get(keychain, TOKEN_SERVICE)
    if token is None:
        raise TelegramSecurityError("Telegram bot credential is not configured")
    token = _validate_token(token)
    response = _transport_call(
        transport,
        token,
        "getUpdates",
        {"timeout": 0, "allowed_updates": '["message"]'},
    )
    updates = _require_ok(response, "getUpdates")
    if not isinstance(updates, list):
        raise TelegramSecurityError("Telegram getUpdates result is invalid")
    chat_ids: set[str] = set()
    for update in updates:
        message = update.get("message") if isinstance(update, Mapping) else None
        chat = message.get("chat") if isinstance(message, Mapping) else None
        sender = message.get("from") if isinstance(message, Mapping) else None
        if not isinstance(chat, Mapping) or chat.get("type") != "private":
            continue
        if not isinstance(sender, Mapping) or sender.get("is_bot") is not False:
            continue
        chat_id = chat.get("id")
        sender_id = sender.get("id")
        if isinstance(chat_id, bool) or not isinstance(chat_id, int) or chat_id <= 0 \
                or sender_id != chat_id:
            continue
        chat_ids.add(str(chat_id))
    if not chat_ids:
        raise TelegramSecurityError(
            "no safe private chat was found; send this bot one direct message and retry"
        )
    if len(chat_ids) != 1:
        raise TelegramSecurityError("multiple private chats were found; automatic selection is unsafe")
    chat_id = next(iter(chat_ids))
    _keychain_set(keychain, CHAT_SERVICE, chat_id)
    return {
        "private_chat_configured": True,
        "chat_identity_digest": telegram_chat_identity_digest(chat_id),
    }


def configure_hmac_key(
    keychain: Keychain,
    *,
    random_bytes: Callable[[int], bytes] = secrets.token_bytes,
) -> dict[str, bool | str]:
    """Generate and store a receipt key; never accept key material from the CLI or env."""
    existing = _keychain_get(keychain, HMAC_SERVICE)
    if existing is not None:
        decoded = _decode_hmac_key(existing)
        return {
            "hmac_key_configured": True,
            "hmac_key_identity_digest": _secret_digest("receipt_hmac_key", decoded),
        }
    try:
        generated = random_bytes(HMAC_KEY_BYTES)
    except Exception:
        raise TelegramSecurityError("secure receipt key generation failed") from None
    if not isinstance(generated, bytes) or len(generated) != HMAC_KEY_BYTES:
        raise TelegramSecurityError("secure receipt key generator returned an invalid key")
    _keychain_set(keychain, HMAC_SERVICE, _encode_hmac_key(generated))
    return {
        "hmac_key_configured": True,
        "hmac_key_identity_digest": _secret_digest("receipt_hmac_key", generated),
    }


def credential_status(keychain: Keychain) -> dict[str, bool | str]:
    """Return only booleans and digests; malformed credentials are not exposed."""
    result: dict[str, bool | str] = {}
    token = _keychain_get(keychain, TOKEN_SERVICE)
    try:
        valid_token = _validate_token(token) if token is not None else None
    except TelegramSecurityError:
        valid_token = None
    result["token_configured"] = valid_token is not None
    if valid_token is not None:
        result["token_identity_digest"] = _secret_digest("bot_token", valid_token)
        result["notification_bot_identity_digest"] = (
            _notification_bot_identity_digest(valid_token)
        )

    chat = _keychain_get(keychain, CHAT_SERVICE)
    try:
        valid_chat = _parse_private_chat_id(chat) if chat is not None else None
    except TelegramSecurityError:
        valid_chat = None
    result["private_chat_configured"] = valid_chat is not None
    if valid_chat is not None:
        result["chat_identity_digest"] = telegram_chat_identity_digest(valid_chat)

    encoded_key = _keychain_get(keychain, HMAC_SERVICE)
    try:
        valid_key = _decode_hmac_key(encoded_key) if encoded_key is not None else None
    except TelegramSecurityError:
        valid_key = None
    result["hmac_key_configured"] = valid_key is not None
    if valid_key is not None:
        result["hmac_key_identity_digest"] = _secret_digest("receipt_hmac_key", valid_key)
    result["ready"] = bool(valid_token and valid_chat and valid_key)
    return result


def load_telegram_auth(keychain: Keychain) -> TelegramAuthConfig:
    """Load only chat digest + HMAC into the non-serializable state authenticator."""
    chat = _keychain_get(keychain, CHAT_SERVICE)
    encoded_key = _keychain_get(keychain, HMAC_SERVICE)
    if chat is None or encoded_key is None:
        raise TelegramSecurityError("Telegram receipt authentication is not configured")
    try:
        chat_id = _parse_private_chat_id(chat)
        hmac_key = _decode_hmac_key(encoded_key)
        return TelegramAuthConfig(
            hmac_key=hmac_key,
            expected_chat_identity_digest=telegram_chat_identity_digest(chat_id),
        )
    except (TelegramSecurityError, StateError):
        raise TelegramSecurityError("Telegram receipt authentication is invalid") from None


def _load_sender_credentials(keychain: Keychain) -> _SenderCredentials:
    token = _keychain_get(keychain, TOKEN_SERVICE)
    chat = _keychain_get(keychain, CHAT_SERVICE)
    if token is None or chat is None:
        raise TelegramSecurityError("Telegram sender credentials are not configured")
    try:
        valid_token = _validate_token(token)
        chat_id = _parse_private_chat_id(chat)
        auth = load_telegram_auth(keychain)
    except TelegramSecurityError:
        raise TelegramSecurityError("Telegram sender credentials are invalid") from None
    return _SenderCredentials(valid_token, chat_id, auth)


def _nonnegative_integer(value: Any, label: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TelegramSecurityError(f"campaign status {label} is invalid")
    return value


def _finite_number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or float(value) < minimum:
        raise TelegramSecurityError(f"campaign status {label} is invalid")
    return float(value)


def _safe_name(value: Any, label: str, *, permit_none: bool = False) -> str | None:
    if value is None and permit_none:
        return None
    if not isinstance(value, str) or SAFE_NAME_RE.fullmatch(value) is None:
        raise TelegramSecurityError(f"campaign status {label} is invalid")
    return value


def validate_campaign_status(status: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize every field required by the campaign brief."""
    if not isinstance(status, Mapping) or set(status) != STATUS_KEYS:
        raise TelegramSecurityError("campaign status fields are incomplete or unknown")
    state = _safe_name(status["state"], "state")
    coverage = _finite_number(status["source_coverage_percent"], "source coverage")
    if coverage > 100:
        raise TelegramSecurityError("campaign status source coverage is invalid")

    shards = status["shards"]
    if not isinstance(shards, Mapping) or set(shards) != SHARD_KEYS:
        raise TelegramSecurityError("campaign status shard fields are incomplete")
    normalized_shards = {
        key: _nonnegative_integer(shards[key], f"shards.{key}")
        for key in sorted(SHARD_KEYS)
    }
    if not (
        normalized_shards["evicted"] <= normalized_shards["verified"]
        <= normalized_shards["fetched"] <= normalized_shards["total"]
    ):
        raise TelegramSecurityError("campaign status shard counts are inconsistent")

    current = status["current"]
    if not isinstance(current, Mapping) or set(current) != CURRENT_KEYS:
        raise TelegramSecurityError("campaign status current fields are incomplete")
    window = _safe_name(current["window"], "current.window", permit_none=True)
    layer = current["layer"]
    if layer is not None:
        layer = _nonnegative_integer(layer, "current.layer")

    candidate_rates = status["candidate_rates"]
    if not isinstance(candidate_rates, list) or len(candidate_rates) > 20 \
            or any(not isinstance(rate, str) or RATE_RE.fullmatch(rate) is None for rate in candidate_rates):
        raise TelegramSecurityError("campaign status candidate rates are invalid")

    best_metrics = status["best_metrics"]
    if not isinstance(best_metrics, Mapping) or len(best_metrics) > 20:
        raise TelegramSecurityError("campaign status best metrics are invalid")
    metric_names = list(best_metrics)
    if any(
        not isinstance(key, str) or SAFE_NAME_RE.fullmatch(key) is None
        for key in metric_names
    ):
        raise TelegramSecurityError("campaign status metric name is invalid")
    normalized_metrics: dict[str, int | float | bool | None] = {}
    for key in sorted(metric_names):
        value = best_metrics[key]
        if value is None or isinstance(value, bool):
            normalized_metrics[key] = value
        elif isinstance(value, int):
            normalized_metrics[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            normalized_metrics[key] = value
        else:
            raise TelegramSecurityError("campaign status metric value is invalid")

    resources = status["resources"]
    if not isinstance(resources, Mapping) or set(resources) != RESOURCE_KEYS:
        raise TelegramSecurityError("campaign status resource fields are incomplete")
    normalized_resources = {
        key: _nonnegative_integer(resources[key], f"resources.{key}")
        for key in sorted(RESOURCE_KEYS)
    }

    process = status["process"]
    if not isinstance(process, Mapping) or set(process) != PROCESS_KEYS \
            or not isinstance(process["lease_held"], bool):
        raise TelegramSecurityError("campaign status process/lease fields are invalid")
    normalized_process = {
        "pid": _nonnegative_integer(process["pid"], "process.pid", positive=True),
        "lease_held": process["lease_held"],
        "lease_owner": _safe_name(process["lease_owner"], "process.lease_owner"),
    }

    eta = status["eta_seconds"]
    if eta is not None:
        eta = _nonnegative_integer(eta, "eta_seconds")
    return {
        "state": state,
        "source_coverage_percent": coverage,
        "shards": normalized_shards,
        "network_bytes": _nonnegative_integer(status["network_bytes"], "network_bytes"),
        "throughput_bytes_per_second": _finite_number(
            status["throughput_bytes_per_second"], "throughput"
        ),
        "eta_seconds": eta,
        "current": {"window": window, "layer": layer},
        "candidate_rates": list(candidate_rates),
        "best_metrics": normalized_metrics,
        "resources": normalized_resources,
        "process": normalized_process,
    }


def compose_message(event: str, dedupe_key: str, status: Mapping[str, Any]) -> str:
    event = _safe_name(event, "event")
    if not isinstance(dedupe_key, str) or SHA256_RE.fullmatch(dedupe_key) is None:
        raise TelegramSecurityError("Telegram event dedupe key is invalid")
    value = validate_campaign_status(status)
    shards = value["shards"]
    current = value["current"]
    resources = value["resources"]
    process = value["process"]
    rates = ",".join(value["candidate_rates"]) or "none"
    metrics = json.dumps(value["best_metrics"], sort_keys=True, separators=(",", ":"))
    text = "\n".join([
        "GLM-5.2 Gravity",
        f"event={event}",
        f"dedupe_key={dedupe_key}",
        f"state={value['state']}",
        f"source_coverage_percent={value['source_coverage_percent']:.6f}",
        (
            f"shards fetched={shards['fetched']} verified={shards['verified']} "
            f"evicted={shards['evicted']} total={shards['total']}"
        ),
        (
            f"network_bytes={value['network_bytes']} "
            f"throughput_bytes_per_second={value['throughput_bytes_per_second']:.6f} "
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
    if len(text) > MAX_MESSAGE_CHARS:
        raise TelegramSecurityError("complete campaign status exceeds Telegram message limit")
    return text


def _canonical_object(value: Any, label: str, *, nonempty: bool = False) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TelegramSecurityError(f"{label} must be a canonical JSON object")
    try:
        cloned = json.loads(canonical(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError):
        raise TelegramSecurityError(f"{label} must be a canonical JSON object") from None
    if not isinstance(cloned, dict) or (nonempty and not cloned):
        raise TelegramSecurityError(f"{label} must be a non-empty canonical JSON object")
    return cloned


def make_delivery_binding(
    transition_intent: Mapping[str, Any],
    auth: TelegramAuthConfig,
) -> dict[str, Any]:
    """Extract the complete sealed controller intent before any outbox or network I/O."""
    if not isinstance(auth, TelegramAuthConfig):
        raise TelegramSecurityError("Telegram receipt authenticator is required")
    try:
        intent = _state_validate_transition_intent(transition_intent, auth)
    except StateError:
        raise TelegramSecurityError("prepared controller transition intent is invalid") from None
    intent = _canonical_object(intent, "prepared controller transition intent", nonempty=True)
    rendered = intent["rendered_message"]
    return {
        "schema": DELIVERY_BINDING_SCHEMA,
        "transition_intent": intent,
        "transition_intent_sha256": intent["seal_sha256"],
        "event_kind": intent["event_kind"],
        "claim_id": intent["claim_id"],
        "from_state": intent["from_state"],
        "to_state": intent["to_state"],
        "dedupe_key": intent["dedupe_key"],
        "canonical_status": intent["canonical_status"],
        "canonical_status_sha256": intent["canonical_status_sha256"],
        "rendered_message": rendered,
        "rendered_message_sha256": intent["rendered_message_sha256"],
        "rendered_message_utf8_bytes": len(rendered.encode("utf-8")),
        "controller_anchor": intent["controller_anchor"],
        "controller_anchor_sha256": intent["controller_anchor"]["anchor_sha256"],
    }


def _validate_delivery_binding(value: Any, auth: TelegramAuthConfig) -> dict[str, Any]:
    binding = _canonical_object(value, "Telegram delivery binding", nonempty=True)
    if set(binding) != _DELIVERY_BINDING_KEYS \
            or binding.get("schema") != DELIVERY_BINDING_SCHEMA:
        raise TelegramSecurityError("Telegram delivery binding fields are invalid")
    expected = make_delivery_binding(binding.get("transition_intent"), auth)
    if canonical(binding) != canonical(expected):
        raise TelegramSecurityError("Telegram delivery binding digest or message is invalid")
    return binding


def make_reconciliation_authorization(
    transition_intent: Mapping[str, Any],
    *,
    ambiguity_entry: Mapping[str, Any],
    auth: TelegramAuthConfig,
    action: str,
    authorization_claim_id: str,
    reason: str,
    authorized_at: str | None = None,
) -> dict[str, Any]:
    """Authorize one retry of one exact, durably blocked send attempt."""
    binding = make_delivery_binding(transition_intent, auth)
    if action != "AUTHORIZE_DUPLICATE_RETRY":
        raise TelegramSecurityError("Telegram reconciliation action is invalid")
    blocked = _validate_standalone_ambiguity_entry(ambiguity_entry, binding, auth)
    block = blocked["reconciliation"]
    if not isinstance(authorization_claim_id, str) \
            or CLAIM_RE.fullmatch(authorization_claim_id) is None:
        raise TelegramSecurityError("Telegram reconciliation claim id is invalid")
    if not isinstance(reason, str) or not reason or reason != reason.strip() \
            or len(reason) > 512 or "\n" in reason or "\r" in reason:
        raise TelegramSecurityError("Telegram reconciliation reason is invalid")
    timestamp = authorized_at or utc_now()
    if not isinstance(timestamp, str) or not timestamp:
        raise TelegramSecurityError("Telegram reconciliation timestamp is invalid")
    body = {
        "schema": RECONCILIATION_AUTH_SCHEMA,
        "status": "AUTHORIZED",
        "action": action,
        "authorization_claim_id": authorization_claim_id,
        "transition_intent_sha256": binding["transition_intent_sha256"],
        "dedupe_key": binding["dedupe_key"],
        "canonical_status_sha256": binding["canonical_status_sha256"],
        "rendered_message_sha256": binding["rendered_message_sha256"],
        "controller_anchor_sha256": binding["controller_anchor_sha256"],
        "ambiguous_entry_seq": blocked["seq"],
        "ambiguous_entry_chain_sha256": blocked["chain_sha256"],
        "send_started_seq": block["send_started_seq"],
        "send_started_chain_sha256": block["send_started_chain_sha256"],
        "attempt": block["attempt"],
        "reason": reason,
        "authorized_at": timestamp,
    }
    sealed = {**body, "seal_sha256": hashlib.sha256(canonical(body)).hexdigest()}
    return {**sealed, "hmac_sha256": auth.authenticate(sealed)}


def _validate_reconciliation_authorization(
    value: Any,
    binding: Mapping[str, Any],
    auth: TelegramAuthConfig,
    *,
    ambiguity_entry: Mapping[str, Any],
) -> dict[str, Any]:
    authorization = _canonical_object(
        value, "Telegram reconciliation authorization", nonempty=True
    )
    if set(authorization) != _RECONCILIATION_AUTH_KEYS \
            or authorization.get("schema") != RECONCILIATION_AUTH_SCHEMA \
            or authorization.get("status") != "AUTHORIZED" \
            or authorization.get("action") != "AUTHORIZE_DUPLICATE_RETRY":
        raise TelegramSecurityError("Telegram reconciliation authorization fields are invalid")
    blocked = _validate_standalone_ambiguity_entry(ambiguity_entry, binding, auth)
    block = blocked["reconciliation"]
    expected_bindings = {
        "transition_intent_sha256": binding["transition_intent_sha256"],
        "dedupe_key": binding["dedupe_key"],
        "canonical_status_sha256": binding["canonical_status_sha256"],
        "rendered_message_sha256": binding["rendered_message_sha256"],
        "controller_anchor_sha256": binding["controller_anchor_sha256"],
        "ambiguous_entry_seq": blocked["seq"],
        "ambiguous_entry_chain_sha256": blocked["chain_sha256"],
        "send_started_seq": block["send_started_seq"],
        "send_started_chain_sha256": block["send_started_chain_sha256"],
        "attempt": block["attempt"],
    }
    if any(authorization.get(key) != expected for key, expected in expected_bindings.items()) \
            or isinstance(authorization.get("ambiguous_entry_seq"), bool) \
            or not isinstance(authorization.get("ambiguous_entry_seq"), int) \
            or authorization["ambiguous_entry_seq"] < 0 \
            or isinstance(authorization.get("send_started_seq"), bool) \
            or not isinstance(authorization.get("send_started_seq"), int) \
            or authorization["send_started_seq"] < 0 \
            or isinstance(authorization.get("attempt"), bool) \
            or not isinstance(authorization.get("attempt"), int) \
            or authorization["attempt"] <= 0 \
            or not _is_sha256(authorization.get("ambiguous_entry_chain_sha256")) \
            or not _is_sha256(authorization.get("send_started_chain_sha256")) \
            or not isinstance(authorization.get("authorization_claim_id"), str) \
            or CLAIM_RE.fullmatch(authorization["authorization_claim_id"]) is None \
            or not isinstance(authorization.get("reason"), str) \
            or not authorization["reason"] or authorization["reason"] != authorization["reason"].strip() \
            or len(authorization["reason"]) > 512 \
            or "\n" in authorization["reason"] or "\r" in authorization["reason"] \
            or not isinstance(authorization.get("authorized_at"), str) \
            or not authorization["authorized_at"]:
        raise TelegramSecurityError("Telegram reconciliation authorization binding is invalid")
    unsigned = {
        key: item for key, item in authorization.items()
        if key not in {"seal_sha256", "hmac_sha256"}
    }
    if authorization.get("seal_sha256") != hashlib.sha256(canonical(unsigned)).hexdigest():
        raise TelegramSecurityError("Telegram reconciliation authorization seal is invalid")
    signed = {**unsigned, "seal_sha256": authorization["seal_sha256"]}
    if not auth.verify(signed, authorization.get("hmac_sha256")):
        raise TelegramSecurityError("Telegram reconciliation authorization HMAC is invalid")
    return authorization


def _ambiguity_block(
    attempt: int,
    reason: str,
    *,
    send_started_seq: int,
    send_started_chain_sha256: str,
) -> dict[str, Any]:
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0 \
            or isinstance(send_started_seq, bool) or not isinstance(send_started_seq, int) \
            or send_started_seq < 0 or not _is_sha256(send_started_chain_sha256) \
            or reason not in {
                "PROCESS_RECOVERY_FOUND_SEND_STARTED",
                "SENDER_FAILED_AFTER_SEND_STARTED",
            }:
        raise TelegramSecurityError("Telegram ambiguity block evidence is invalid")
    return {
        "schema": AMBIGUITY_BLOCK_SCHEMA,
        "status": "BLOCKED_AMBIGUOUS_SEND",
        "attempt": attempt,
        "send_started_seq": send_started_seq,
        "send_started_chain_sha256": send_started_chain_sha256,
        "reason": reason,
    }


def _validate_ambiguity_block(value: Any) -> dict[str, Any]:
    block = _canonical_object(value, "Telegram ambiguity block", nonempty=True)
    if set(block) != _AMBIGUITY_BLOCK_KEYS:
        raise TelegramSecurityError("Telegram ambiguity block fields are invalid")
    return _ambiguity_block(
        block.get("attempt"),
        block.get("reason"),
        send_started_seq=block.get("send_started_seq"),
        send_started_chain_sha256=block.get("send_started_chain_sha256"),
    )


def _validate_standalone_ambiguity_entry(
    value: Any,
    binding: Mapping[str, Any],
    auth: TelegramAuthConfig,
) -> dict[str, Any]:
    """Authenticate block evidence before it is embedded in a retry authorization."""
    entry = _canonical_object(value, "Telegram ambiguity ledger entry", nonempty=True)
    if set(entry) != _LEDGER_ENTRY_KEYS \
            or entry.get("schema") != DELIVERY_LEDGER_SCHEMA \
            or isinstance(entry.get("seq"), bool) or not isinstance(entry.get("seq"), int) \
            or entry["seq"] < 0 or entry.get("kind") != "AMBIGUOUS_BLOCKED" \
            or not isinstance(entry.get("recorded_at"), str) or not entry["recorded_at"] \
            or entry.get("dedupe_key") != binding["dedupe_key"] \
            or entry.get("receipt") is not None \
            or not _is_sha256(entry.get("prev_chain_sha256")):
        raise TelegramSecurityError("Telegram ambiguity ledger entry fields are invalid")
    validated_binding = _validate_delivery_binding(entry.get("binding"), auth)
    if canonical(validated_binding) != canonical(binding):
        raise TelegramSecurityError("Telegram ambiguity ledger entry binding is invalid")
    block = _validate_ambiguity_block(entry.get("reconciliation"))
    unsigned = {
        key: item for key, item in entry.items()
        if key not in {"chain_sha256", "hmac_sha256"}
    }
    expected_chain = hashlib.sha256(canonical(unsigned)).hexdigest()
    signed = {**unsigned, "chain_sha256": expected_chain}
    if entry.get("chain_sha256") != expected_chain \
            or not auth.verify(signed, entry.get("hmac_sha256")):
        raise TelegramSecurityError("Telegram ambiguity ledger entry authentication failed")
    return {**entry, "binding": validated_binding, "reconciliation": block}


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite constant")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise TelegramSecurityError(f"{label} is not strict canonical JSON") from None
    if not isinstance(value, dict):
        raise TelegramSecurityError(f"{label} is not a JSON object")
    return value


def _process_ledger_lock(path: Path) -> threading.Lock:
    key = os.path.abspath(os.fspath(path))
    with _LEDGER_LOCKS_GUARD:
        lock = _LEDGER_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LEDGER_LOCKS[key] = lock
        return lock


class TelegramDeliveryLedger:
    """Durable HMAC-authenticated outbox and exact receipt journal.

    Each append is fsynced before the authenticated head sidecar advances.  A
    valid ledger ahead of its head is safe crash recovery; a head ahead of the
    ledger is clean-tail truncation and is refused.  A SEND_STARTED record
    without a receipt is deliberately ambiguous and can be resent only after
    an HMAC authorization binds the exact durable ambiguity and prior attempt.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        fault_injector: Callable[[str], None] | None = None,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_absolute() or self.path.anchor != os.sep \
                or any(part in {"", ".", ".."} for part in self.path.parts[1:]) \
                or self.path.name in {"", ".", ".."}:
            raise TelegramSecurityError("Telegram delivery ledger path must be absolute and normalized")
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise TelegramSecurityError("platform lacks required no-symlink filesystem primitives")
        self._fault_injector = fault_injector
        self._clock = clock
        self._lock_name = self.path.name + ".lock"
        self._head_name = self.path.name + ".head"
        self._process_lock = _process_ledger_lock(self.path)

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self._lock_name)

    @property
    def head_path(self) -> Path:
        return self.path.with_name(self._head_name)

    def _fault(self, phase: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(phase)

    def _open_parent_fd(self) -> int:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        descriptor = -1
        try:
            descriptor = os.open(os.sep, flags)
            for component in self.path.parent.parts[1:]:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
                metadata = os.fstat(next_descriptor)
                if not stat.S_ISDIR(metadata.st_mode):
                    os.close(next_descriptor)
                    raise TelegramSecurityError("Telegram ledger parent is not a directory")
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor
        except TelegramSecurityError:
            if descriptor >= 0:
                os.close(descriptor)
            raise
        except OSError:
            if descriptor >= 0:
                os.close(descriptor)
            raise TelegramSecurityError("Telegram ledger parent path is missing or unsafe") from None

    @staticmethod
    def _open_regular_leaf(
        parent_fd: int,
        name: str,
        flags: int,
        mode: int = 0o600,
    ) -> int:
        try:
            descriptor = os.open(
                name,
                flags | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                mode,
                dir_fd=parent_fd,
            )
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                os.close(descriptor)
                raise TelegramSecurityError("Telegram ledger leaf is not a unique regular file")
            return descriptor
        except TelegramSecurityError:
            raise
        except OSError:
            raise TelegramSecurityError("Telegram ledger leaf is missing or unsafe") from None

    @contextmanager
    def _locked_files(self) -> Iterator[tuple[int, int]]:
        self._process_lock.acquire()
        parent_fd = lock_fd = ledger_fd = -1
        try:
            parent_fd = self._open_parent_fd()
            lock_fd = self._open_regular_leaf(
                parent_fd,
                self._lock_name,
                os.O_RDWR | os.O_CREAT,
            )
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            except OSError:
                raise TelegramSecurityError("Telegram ledger concurrency lock failed") from None
            ledger_fd = self._open_regular_leaf(
                parent_fd,
                self.path.name,
                os.O_RDWR | os.O_APPEND | os.O_CREAT,
            )
            try:
                fcntl.flock(ledger_fd, fcntl.LOCK_EX)
                os.fsync(parent_fd)
            except OSError:
                raise TelegramSecurityError("Telegram ledger durable open failed") from None
            yield parent_fd, ledger_fd
        finally:
            if ledger_fd >= 0:
                try:
                    fcntl.flock(ledger_fd, fcntl.LOCK_UN)
                finally:
                    os.close(ledger_fd)
            if lock_fd >= 0:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
            if parent_fd >= 0:
                os.close(parent_fd)
            self._process_lock.release()

    @staticmethod
    def _read_fd(descriptor: int, maximum: int) -> bytes:
        try:
            metadata = os.fstat(descriptor)
            if metadata.st_size > maximum:
                raise TelegramSecurityError("Telegram delivery ledger exceeds its safe size limit")
            os.lseek(descriptor, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            remaining = maximum + 1
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > maximum:
                raise TelegramSecurityError("Telegram delivery ledger exceeds its safe size limit")
            return raw
        except TelegramSecurityError:
            raise
        except OSError:
            raise TelegramSecurityError("Telegram delivery ledger read failed") from None

    def _read_head(self, parent_fd: int, auth: TelegramAuthConfig) -> dict[str, Any] | None:
        try:
            descriptor = self._open_regular_leaf(parent_fd, self._head_name, os.O_RDONLY)
        except TelegramSecurityError:
            try:
                os.stat(self._head_name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return None
            except OSError:
                pass
            raise
        try:
            raw = self._read_fd(descriptor, 16 * 1024)
        finally:
            os.close(descriptor)
        if not raw or not raw.endswith(b"\n") or raw.count(b"\n") != 1:
            raise TelegramSecurityError("Telegram delivery ledger head is torn")
        head = _strict_json_object(raw[:-1], "Telegram delivery ledger head")
        if canonical(head) != raw[:-1]:
            raise TelegramSecurityError("Telegram delivery ledger head is not canonical JSON")
        if set(head) != _LEDGER_HEAD_KEYS or head.get("schema") != DELIVERY_LEDGER_HEAD_SCHEMA \
                or isinstance(head.get("entry_count"), bool) \
                or not isinstance(head.get("entry_count"), int) or head["entry_count"] < 0 \
                or not _is_sha256(head.get("head_chain_sha256")) \
                or not _is_sha256(head.get("head_entry_hmac_sha256")):
            raise TelegramSecurityError("Telegram delivery ledger head fields are invalid")
        signed = {key: value for key, value in head.items() if key != "hmac_sha256"}
        if not auth.verify(signed, head.get("hmac_sha256")):
            raise TelegramSecurityError("Telegram delivery ledger head HMAC is invalid")
        return head

    @staticmethod
    def _head_value(entries: Sequence[Mapping[str, Any]], auth: TelegramAuthConfig) -> dict[str, Any]:
        if entries:
            chain = entries[-1]["chain_sha256"]
            entry_hmac = entries[-1]["hmac_sha256"]
        else:
            chain = LEDGER_GENESIS_HASH
            entry_hmac = LEDGER_GENESIS_HASH
        body = {
            "schema": DELIVERY_LEDGER_HEAD_SCHEMA,
            "entry_count": len(entries),
            "head_chain_sha256": chain,
            "head_entry_hmac_sha256": entry_hmac,
        }
        return {**body, "hmac_sha256": auth.authenticate(body)}

    def _write_head(
        self,
        parent_fd: int,
        entries: Sequence[Mapping[str, Any]],
        auth: TelegramAuthConfig,
    ) -> None:
        try:
            metadata = os.stat(self._head_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            metadata = None
        except OSError:
            raise TelegramSecurityError("Telegram delivery ledger head inspection failed") from None
        if metadata is not None and (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1):
            raise TelegramSecurityError("Telegram delivery ledger head is not a unique regular file")
        value = self._head_value(entries, auth)
        encoded = canonical(value) + b"\n"
        temporary = f".{self._head_name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(8)}.tmp"
        descriptor = -1
        try:
            descriptor = self._open_regular_leaf(
                parent_fd,
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            )
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, self._head_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.fsync(parent_fd)
        except TelegramSecurityError:
            raise
        except OSError:
            raise TelegramSecurityError("Telegram delivery ledger head durable write failed") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass

    @staticmethod
    def _validate_receipt(
        receipt: Any,
        binding: Mapping[str, Any],
        auth: TelegramAuthConfig,
    ) -> dict[str, Any]:
        value = _canonical_object(receipt, "Telegram signed delivery receipt", nonempty=True)
        try:
            validated = _state_validate_telegram_delivery_receipt(
                value,
                binding["transition_intent"],
                auth,
            )
        except StateError:
            raise TelegramSecurityError(
                "Telegram signed delivery receipt binding/HMAC is invalid"
            ) from None
        if validated.get("schema") != TELEGRAM_RECEIPT_SCHEMA \
                or validated.get("status") != "DELIVERED" \
                or validated.get("response_validated") is not True \
                or validated.get("bot_api_http_status") != 200:
            raise TelegramSecurityError(
                "Telegram outbox requires a validated successful Bot API receipt"
            )
        return _canonical_object(validated, "Telegram signed delivery receipt", nonempty=True)

    def _verified_entries(
        self,
        parent_fd: int,
        ledger_fd: int,
        auth: TelegramAuthConfig,
    ) -> list[dict[str, Any]]:
        raw = self._read_fd(ledger_fd, MAX_LEDGER_BYTES)
        torn_tail = bool(raw and not raw.endswith(b"\n"))
        if torn_tail:
            final_newline = raw.rfind(b"\n")
            complete_raw = raw[:final_newline + 1] if final_newline >= 0 else b""
        else:
            complete_raw = raw
        entries: list[dict[str, Any]] = []
        previous = LEDGER_GENESIS_HASH
        lifecycles: dict[str, list[dict[str, Any]]] = {}
        bound_intents: dict[str, bytes] = {}
        authorization_claims: set[str] = set()
        for seq, line in enumerate(complete_raw.splitlines()):
            if not line:
                raise TelegramSecurityError("Telegram delivery ledger contains a blank record")
            entry = _strict_json_object(line, "Telegram delivery ledger entry")
            if canonical(entry) != line:
                raise TelegramSecurityError("Telegram delivery ledger entry is not canonical JSON")
            if set(entry) != _LEDGER_ENTRY_KEYS or entry.get("schema") != DELIVERY_LEDGER_SCHEMA \
                    or isinstance(entry.get("seq"), bool) or entry.get("seq") != seq \
                    or entry.get("kind") not in LEDGER_ENTRY_KINDS \
                    or not isinstance(entry.get("recorded_at"), str) or not entry["recorded_at"] \
                    or not _is_sha256(entry.get("dedupe_key")) \
                    or entry.get("prev_chain_sha256") != previous:
                raise TelegramSecurityError("Telegram delivery ledger entry fields or sequence are invalid")
            binding = _validate_delivery_binding(entry.get("binding"), auth)
            if entry["dedupe_key"] != binding["dedupe_key"]:
                raise TelegramSecurityError("Telegram delivery ledger dedupe binding is invalid")
            unsigned = {
                key: item for key, item in entry.items()
                if key not in {"chain_sha256", "hmac_sha256"}
            }
            expected_chain = hashlib.sha256(canonical(unsigned)).hexdigest()
            if entry.get("chain_sha256") != expected_chain:
                raise TelegramSecurityError("Telegram delivery ledger hash chain is invalid")
            signed = {**unsigned, "chain_sha256": expected_chain}
            if not auth.verify(signed, entry.get("hmac_sha256")):
                raise TelegramSecurityError("Telegram delivery ledger entry HMAC is invalid")

            dedupe_key = entry["dedupe_key"]
            encoded_binding = canonical(binding)
            prior_binding = bound_intents.setdefault(dedupe_key, encoded_binding)
            if prior_binding != encoded_binding:
                raise TelegramSecurityError(
                    "Telegram dedupe key is rebound to a different delivery intent"
                )
            history = lifecycles.setdefault(dedupe_key, [])
            prior_kind = history[-1]["kind"] if history else None
            kind = entry["kind"]
            if kind not in _LIFECYCLE_NEXT_KINDS[prior_kind]:
                raise TelegramSecurityError("Telegram delivery ledger outbox lifecycle is invalid")
            receipt = entry.get("receipt")
            reconciliation = entry.get("reconciliation")
            if kind in {"OUTBOX_PREPARED", "SEND_STARTED"}:
                if receipt is not None or reconciliation is not None:
                    raise TelegramSecurityError("ordinary Telegram outbox entry has extra evidence")
            elif kind == "AMBIGUOUS_BLOCKED":
                if receipt is not None:
                    raise TelegramSecurityError("ambiguous Telegram outbox may not claim delivery")
                block = _validate_ambiguity_block(reconciliation)
                started = history[-1]
                expected_block = _ambiguity_block(
                    sum(item["kind"] == "SEND_STARTED" for item in history),
                    block["reason"],
                    send_started_seq=started["seq"],
                    send_started_chain_sha256=started["chain_sha256"],
                )
                if canonical(block) != canonical(expected_block):
                    raise TelegramSecurityError(
                        "Telegram ambiguity block does not identify the prior send attempt"
                    )
            elif kind == "DUPLICATE_RETRY_AUTHORIZED":
                if receipt is not None:
                    raise TelegramSecurityError("Telegram retry authorization may not claim delivery")
                authorization = _validate_reconciliation_authorization(
                    reconciliation,
                    binding,
                    auth,
                    ambiguity_entry=history[-1],
                )
                claim_id = authorization["authorization_claim_id"]
                if claim_id in authorization_claims:
                    raise TelegramSecurityError(
                        "Telegram retry authorization claim was already consumed"
                    )
                authorization_claims.add(claim_id)
            elif kind == "DELIVERY_RECEIPT":
                if reconciliation is not None:
                    raise TelegramSecurityError(
                        "validated Bot API receipt may not carry operator delivery evidence"
                    )
                self._validate_receipt(receipt, binding, auth)
            previous = expected_chain
            validated_entry = {**entry, "binding": binding}
            entries.append(validated_entry)
            history.append(validated_entry)

        head = self._read_head(parent_fd, auth)
        expected_head = self._head_value(entries, auth)
        if torn_tail:
            if head is None or head != expected_head:
                raise TelegramSecurityError(
                    "Telegram delivery ledger torn tail lacks an authenticated recovery head"
                )
            try:
                os.ftruncate(ledger_fd, len(complete_raw))
                os.fsync(ledger_fd)
                os.fsync(parent_fd)
            except OSError:
                raise TelegramSecurityError(
                    "Telegram delivery ledger torn-tail recovery failed"
                ) from None
            self._fault("after_torn_tail_recovery_fsync")
            return entries
        if head is None:
            if entries:
                raise TelegramSecurityError(
                    "Telegram delivery ledger lacks its authenticated durable head"
                )
            # Establish a durable authenticated genesis before the first append,
            # so a later missing head can never be mistaken for first-write recovery.
            self._write_head(parent_fd, entries, auth)
        elif head["entry_count"] > len(entries):
            raise TelegramSecurityError("Telegram delivery ledger was clean-tail truncated")
        elif head["entry_count"] == len(entries):
            if head != expected_head:
                raise TelegramSecurityError("Telegram delivery ledger head does not match its chain")
        else:
            # The entry append was fsynced before a crash prevented head advancement.
            self._write_head(parent_fd, entries, auth)
        return entries

    def _append_entry(
        self,
        parent_fd: int,
        ledger_fd: int,
        entries: Sequence[Mapping[str, Any]],
        *,
        kind: str,
        binding: Mapping[str, Any],
        receipt: Mapping[str, Any] | None,
        reconciliation: Mapping[str, Any] | None,
        auth: TelegramAuthConfig,
    ) -> list[dict[str, Any]]:
        if kind not in LEDGER_ENTRY_KINDS:
            raise TelegramSecurityError("Telegram delivery ledger entry kind is invalid")
        validated_binding = _validate_delivery_binding(binding, auth)
        matching = [
            entry for entry in entries
            if entry["dedupe_key"] == validated_binding["dedupe_key"]
        ]
        if any(
            canonical(entry["binding"]) != canonical(validated_binding)
            for entry in matching
        ):
            raise TelegramSecurityError(
                "Telegram dedupe key is already bound to a different delivery intent"
            )
        prior_kind = matching[-1]["kind"] if matching else None
        if kind not in _LIFECYCLE_NEXT_KINDS[prior_kind]:
            raise TelegramSecurityError("Telegram delivery outbox lifecycle is invalid")
        validated_receipt: dict[str, Any] | None = None
        validated_reconciliation: dict[str, Any] | None = None
        if kind in {"OUTBOX_PREPARED", "SEND_STARTED"}:
            if receipt is not None or reconciliation is not None:
                raise TelegramSecurityError("ordinary Telegram outbox entry has extra evidence")
        elif kind == "AMBIGUOUS_BLOCKED":
            if receipt is not None:
                raise TelegramSecurityError("ambiguous Telegram outbox may not claim delivery")
            validated_reconciliation = _validate_ambiguity_block(reconciliation)
            started = matching[-1]
            expected_block = _ambiguity_block(
                sum(entry["kind"] == "SEND_STARTED" for entry in matching),
                validated_reconciliation["reason"],
                send_started_seq=started["seq"],
                send_started_chain_sha256=started["chain_sha256"],
            )
            if canonical(validated_reconciliation) != canonical(expected_block):
                raise TelegramSecurityError(
                    "Telegram ambiguity block does not identify the prior send attempt"
                )
        elif kind == "DUPLICATE_RETRY_AUTHORIZED":
            if receipt is not None:
                raise TelegramSecurityError("Telegram retry authorization may not claim delivery")
            validated_reconciliation = _validate_reconciliation_authorization(
                reconciliation,
                validated_binding,
                auth,
                ambiguity_entry=matching[-1],
            )
            claim_id = validated_reconciliation["authorization_claim_id"]
            if any(
                entry["kind"] == "DUPLICATE_RETRY_AUTHORIZED"
                and entry["reconciliation"]["authorization_claim_id"] == claim_id
                for entry in entries
            ):
                raise TelegramSecurityError(
                    "Telegram retry authorization claim was already consumed"
                )
        elif kind == "DELIVERY_RECEIPT":
            validated_receipt = self._validate_receipt(receipt, validated_binding, auth)
            if reconciliation is not None:
                raise TelegramSecurityError(
                    "validated Bot API receipt may not carry operator delivery evidence"
                )
        recorded_at = self._clock()
        if not isinstance(recorded_at, str) or not recorded_at:
            raise TelegramSecurityError("Telegram delivery ledger clock is invalid")
        unsigned: dict[str, Any] = {
            "schema": DELIVERY_LEDGER_SCHEMA,
            "seq": len(entries),
            "kind": kind,
            "recorded_at": recorded_at,
            "dedupe_key": validated_binding["dedupe_key"],
            "binding": validated_binding,
            "receipt": validated_receipt,
            "reconciliation": validated_reconciliation,
            "prev_chain_sha256": (
                entries[-1]["chain_sha256"] if entries else LEDGER_GENESIS_HASH
            ),
        }
        chain_sha256 = hashlib.sha256(canonical(unsigned)).hexdigest()
        signed = {**unsigned, "chain_sha256": chain_sha256}
        entry = {**signed, "hmac_sha256": auth.authenticate(signed)}
        encoded = canonical(entry) + b"\n"
        try:
            if os.fstat(ledger_fd).st_size + len(encoded) > MAX_LEDGER_BYTES:
                raise TelegramSecurityError(
                    "Telegram delivery ledger exceeds its safe size limit"
                )
            view = memoryview(encoded)
            while view:
                written = os.write(ledger_fd, view)
                if written <= 0:
                    raise OSError("short append")
                view = view[written:]
            os.fsync(ledger_fd)
        except OSError:
            raise TelegramSecurityError("Telegram delivery ledger durable append failed") from None
        updated = [*entries, entry]
        phase = kind.lower()
        self._fault(f"after_{phase}_ledger_fsync_before_head")
        self._write_head(parent_fd, updated, auth)
        self._fault(f"after_{phase}_fsync")
        verified = self._verified_entries(parent_fd, ledger_fd, auth)
        if verified[-1] != entry:
            raise TelegramSecurityError("Telegram delivery ledger post-append verification failed")
        return verified

    def _append_ambiguity_block(
        self,
        parent_fd: int,
        ledger_fd: int,
        entries: Sequence[Mapping[str, Any]],
        binding: Mapping[str, Any],
        auth: TelegramAuthConfig,
        *,
        reason: str,
    ) -> list[dict[str, Any]]:
        matching = [
            entry for entry in entries
            if entry["dedupe_key"] == binding["dedupe_key"]
        ]
        if not matching or matching[-1]["kind"] != "SEND_STARTED":
            raise TelegramSecurityError(
                "Telegram ambiguity block requires the current SEND_STARTED attempt"
            )
        started = matching[-1]
        evidence = _ambiguity_block(
            sum(entry["kind"] == "SEND_STARTED" for entry in matching),
            reason,
            send_started_seq=started["seq"],
            send_started_chain_sha256=started["chain_sha256"],
        )
        return self._append_entry(
            parent_fd,
            ledger_fd,
            entries,
            kind="AMBIGUOUS_BLOCKED",
            binding=binding,
            receipt=None,
            reconciliation=evidence,
            auth=auth,
        )

    def verified_entries(self, auth: TelegramAuthConfig) -> list[dict[str, Any]]:
        if not isinstance(auth, TelegramAuthConfig):
            raise TelegramSecurityError("Telegram receipt authenticator is required")
        with self._locked_files() as (parent_fd, ledger_fd):
            return self._verified_entries(parent_fd, ledger_fd, auth)

    def deliver_or_replay(
        self,
        binding: Mapping[str, Any],
        *,
        auth: TelegramAuthConfig,
        sender: Callable[[], Mapping[str, Any]],
        duplicate_retry_authorization: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(auth, TelegramAuthConfig):
            raise TelegramSecurityError("Telegram receipt authenticator is required")
        validated_binding = _validate_delivery_binding(binding, auth)
        with self._locked_files() as (parent_fd, ledger_fd):
            entries = self._verified_entries(parent_fd, ledger_fd, auth)
            matching = [
                entry for entry in entries
                if entry["dedupe_key"] == validated_binding["dedupe_key"]
            ]
            if matching and any(
                canonical(entry["binding"]) != canonical(validated_binding)
                for entry in matching
            ):
                raise TelegramSecurityError(
                    "Telegram dedupe key is already bound to a different delivery intent"
                )
            latest_kind = matching[-1]["kind"] if matching else None
            if latest_kind == "DELIVERY_RECEIPT":
                return self._validate_receipt(matching[-1]["receipt"], validated_binding, auth)
            if latest_kind == "SEND_STARTED":
                self._append_ambiguity_block(
                    parent_fd,
                    ledger_fd,
                    entries,
                    validated_binding,
                    auth,
                    reason="PROCESS_RECOVERY_FOUND_SEND_STARTED",
                )
                raise TelegramSecurityError(
                    "Telegram delivery outcome is ambiguous and durably blocked; "
                    "a bound HMAC duplicate-retry authorization is required"
                )
            if latest_kind == "AMBIGUOUS_BLOCKED":
                if duplicate_retry_authorization is None:
                    raise TelegramSecurityError(
                        "Telegram delivery outcome is ambiguous and durably blocked; "
                        "a bound HMAC duplicate-retry authorization is required"
                    )
                entries = self._append_entry(
                    parent_fd,
                    ledger_fd,
                    entries,
                    kind="DUPLICATE_RETRY_AUTHORIZED",
                    binding=validated_binding,
                    receipt=None,
                    reconciliation=duplicate_retry_authorization,
                    auth=auth,
                )
                latest_kind = "DUPLICATE_RETRY_AUTHORIZED"
            elif latest_kind == "DUPLICATE_RETRY_AUTHORIZED":
                durable_authorization = matching[-1]["reconciliation"]
                if duplicate_retry_authorization is not None and canonical(
                    _canonical_object(
                        duplicate_retry_authorization,
                        "Telegram reconciliation authorization",
                        nonempty=True,
                    )
                ) != canonical(durable_authorization):
                    raise TelegramSecurityError(
                        "Telegram retry authorization differs from the durable authorization"
                    )
            elif duplicate_retry_authorization is not None:
                raise TelegramSecurityError(
                    "Telegram duplicate-retry authorization has no current ambiguity block"
                )

            if latest_kind is None:
                entries = self._append_entry(
                    parent_fd,
                    ledger_fd,
                    entries,
                    kind="OUTBOX_PREPARED",
                    binding=validated_binding,
                    receipt=None,
                    reconciliation=None,
                    auth=auth,
                )
            elif latest_kind not in {"OUTBOX_PREPARED", "DUPLICATE_RETRY_AUTHORIZED"}:
                raise TelegramSecurityError("Telegram delivery outbox lifecycle is invalid")
            entries = self._append_entry(
                parent_fd,
                ledger_fd,
                entries,
                kind="SEND_STARTED",
                binding=validated_binding,
                receipt=None,
                reconciliation=None,
                auth=auth,
            )
            self._fault("before_network_send")
            try:
                receipt = sender()
                validated_receipt = self._validate_receipt(receipt, validated_binding, auth)
            except Exception:
                self._append_ambiguity_block(
                    parent_fd,
                    ledger_fd,
                    entries,
                    validated_binding,
                    auth,
                    reason="SENDER_FAILED_AFTER_SEND_STARTED",
                )
                raise TelegramSecurityError(
                    "Telegram sender failed after SEND_STARTED; outcome is durably blocked"
                ) from None
            self._fault("after_send_success_before_receipt_append")
            self._append_entry(
                parent_fd,
                ledger_fd,
                entries,
                kind="DELIVERY_RECEIPT",
                binding=validated_binding,
                receipt=validated_receipt,
                reconciliation=None,
                auth=auth,
            )
            return validated_receipt

    def reconcile_ambiguous(
        self,
        transition_intent: Mapping[str, Any],
        receipt: Mapping[str, Any],
        *,
        auth: TelegramAuthConfig,
    ) -> dict[str, Any]:
        """Close an ambiguous send only with its exact retained Bot API receipt."""
        binding = make_delivery_binding(transition_intent, auth)
        validated_receipt = self._validate_receipt(receipt, binding, auth)
        with self._locked_files() as (parent_fd, ledger_fd):
            entries = self._verified_entries(parent_fd, ledger_fd, auth)
            matching = [
                entry for entry in entries
                if entry["dedupe_key"] == binding["dedupe_key"]
            ]
            if matching and any(
                canonical(entry["binding"]) != canonical(binding)
                for entry in matching
            ):
                raise TelegramSecurityError(
                    "Telegram dedupe key is already bound to a different delivery intent"
                )
            latest_kind = matching[-1]["kind"] if matching else None
            if latest_kind == "DELIVERY_RECEIPT":
                stored = self._validate_receipt(matching[-1]["receipt"], binding, auth)
                if canonical(stored) != canonical(validated_receipt):
                    raise TelegramSecurityError(
                        "Telegram reconciliation receipt differs from the durable receipt"
                    )
                return stored
            if latest_kind == "SEND_STARTED":
                entries = self._append_ambiguity_block(
                    parent_fd,
                    ledger_fd,
                    entries,
                    binding,
                    auth,
                    reason="PROCESS_RECOVERY_FOUND_SEND_STARTED",
                )
                latest_kind = "AMBIGUOUS_BLOCKED"
            if latest_kind != "AMBIGUOUS_BLOCKED":
                raise TelegramSecurityError(
                    "Telegram reconciliation requires a durably blocked ambiguous send"
                )
            self._append_entry(
                parent_fd,
                ledger_fd,
                entries,
                kind="DELIVERY_RECEIPT",
                binding=binding,
                receipt=validated_receipt,
                reconciliation=None,
                auth=auth,
            )
            return validated_receipt


def send_campaign_status(
    transition_intent: Mapping[str, Any],
    *,
    ledger: TelegramDeliveryLedger,
    keychain: Keychain,
    transport: TelegramTransport,
    duplicate_retry_authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Durably deliver or replay one exact prepared controller transition."""
    if not isinstance(ledger, TelegramDeliveryLedger):
        raise TelegramSecurityError("Telegram delivery ledger is required")
    credentials = _load_sender_credentials(keychain)
    binding = make_delivery_binding(transition_intent, credentials.auth)
    text = binding["rendered_message"]
    if len(text) > MAX_MESSAGE_CHARS:
        raise TelegramSecurityError("prepared campaign status exceeds Telegram message limit")

    def sender() -> dict[str, Any]:
        response = _transport_call(
            transport,
            credentials.token,
            "sendMessage",
            {
                "chat_id": credentials.chat_id,
                "text": text,
                "disable_web_page_preview": "true",
            },
        )
        result = _require_ok(response, "sendMessage")
        if not isinstance(result, Mapping) or result.get("text") != text:
            raise TelegramSecurityError("Telegram sendMessage did not echo the exact prepared status")
        chat = result.get("chat")
        if not isinstance(chat, Mapping) or chat.get("type") != "private":
            raise TelegramSecurityError("Telegram sendMessage did not target a private chat")
        try:
            return make_telegram_delivery_receipt(
                binding["transition_intent"],
                auth=credentials.auth,
                bot_api_response=response.body,
                http_status=response.status,
            )
        except StateError:
            raise TelegramSecurityError(
                "Telegram sendMessage response failed prepared-intent receipt validation"
            ) from None

    return ledger.deliver_or_replay(
        binding,
        auth=credentials.auth,
        sender=sender,
        duplicate_retry_authorization=duplicate_retry_authorization,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _RedactingArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="show only configured booleans and credential digests")
    subparsers.add_parser("configure-token", help="hidden prompt; validates getMe before storage")
    subparsers.add_parser("discover-private-chat", help="store one unambiguous private chat")
    subparsers.add_parser("configure-hmac-key", help="generate a receipt HMAC key in Keychain")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    keychain: Keychain | None = None,
    transport: TelegramTransport | None = None,
    hidden_prompt: Callable[[str], str] = getpass.getpass,
    random_bytes: Callable[[int], bytes] = secrets.token_bytes,
) -> int:
    args = build_parser().parse_args(argv)
    store = keychain or MacOSKeychain()
    network = transport or UrllibTelegramTransport()
    try:
        if args.command == "status":
            result = credential_status(store)
        elif args.command == "configure-token":
            result = configure_token(store, network, hidden_prompt=hidden_prompt)
        elif args.command == "discover-private-chat":
            result = discover_private_chat(store, network)
        elif args.command == "configure-hmac-key":
            result = configure_hmac_key(store, random_bytes=random_bytes)
        else:  # pragma: no cover - argparse owns the command choices
            raise AssertionError("unreachable command")
    except TelegramSecurityError as exc:
        print(f"GLM52_TELEGRAM_REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

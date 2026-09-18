"""Short-lived, object-scoped Hawking share capabilities.

The token in ``/s/<token>`` is the capability.  Hawking stores only its
SHA-256 digest, so the raw bearer value is never written to the workspace.
The bridge deliberately exposes a Goal/conversation snapshot rather than the
daemon API, provider credentials, prompts, or an arbitrary tool surface.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import os
import re
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .persist import atomic_write_json

SHARE_SCHEMA = "hawking.share_capability.v1"
SHARE_EVENT_SCHEMA = "hawking.share_event.v1"
TUNNEL_SCHEMA = "hawking.tunnel.session.v1"
TUNNEL_EVENT_SCHEMA = "hawking.tunnel.event.v1"
TUNNEL_FRAME_SCHEMA = "hawking.tunnel.frame.v1"
DEFAULT_TTL_SECONDS = 60 * 60
MAX_TTL_SECONDS = 7 * 24 * 60 * 60
TUNNEL_INVITATION_TTL_SECONDS = 5 * 60
TUNNEL_MAX_INVITATION_TTL_SECONDS = 60 * 60
KNOWN_PERMISSIONS = {"read", "comment", "steer", "prompt", "build"}
TUNNEL_OPERATION_PERMISSIONS = {
    "list_chats": "read",
    "list_models": "read",
    "stream_prompt": "prompt",
    "scoped_build": "build",
    "retrieve_result": "read",
}


def tunnel_operation_permission(operation: str) -> str:
    """Return the permission required for a tunnel operation.

    Unknown operations fail closed with an explicit :class:`ShareError`
    (HTTP 400) instead of silently defaulting to a permissive scope, so a
    caller can never reach a tunnel handler without a declared permission.
    """
    if not isinstance(operation, str):
        raise ShareError("tunnel operation must be a string", status=400)
    try:
        return TUNNEL_OPERATION_PERMISSIONS[operation]
    except KeyError:
        raise ShareError(
            f"unknown tunnel operation: {operation!r}", status=400
        ) from None


def tunnel_operation_allowed(record: Mapping[str, Any], operation: str) -> bool:
    """Return whether a share record grants the permission for ``operation``.

    The tunnel session record carries the capability's granted permissions in
    ``permissions``.  A tunnel handler must consult this predicate before
    dispatching an operation so that a read-only capability can never reach a
    ``prompt`` or ``build`` handler.  Unknown operations fail closed (the
    permission lookup raises :class:`ShareError`), and a record whose
    ``permissions`` field is missing, malformed, or empty is treated as
    read-only rather than as an unrestricted capability.
    """
    required = tunnel_operation_permission(operation)
    if not isinstance(record, Mapping):
        return False
    granted = _permissions(record.get("permissions"))
    if required not in granted:
        return False
    if required == "read":
        return True
    expires_at = record.get("expires_at")
    if expires_at is None:
        return True
    try:
        deadline = float(expires_at)
    except (TypeError, ValueError):
        return False
    return time.time() < deadline
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{10,64}$")
GOAL_RE = re.compile(r"^GOAL-[A-Za-z0-9_-]{1,100}$", re.IGNORECASE)
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DEVICE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
TUNNEL_SECRET_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(openrouter_api_key\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)

_LOCK = threading.RLock()


class ShareError(RuntimeError):
    """An explicit share failure that maps cleanly to an HTTP status."""

    def __init__(self, message: str, status: int = 404):
        super().__init__(message)
        self.status = int(status)


def _root(workspace: str | os.PathLike[str]) -> Path:
    path = Path(workspace).expanduser().resolve() / ".hawking" / "shares"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _path(workspace: str | os.PathLike[str], token: str) -> Path:
    return _root(workspace) / f"{_token_hash(token)}.json"


def _clean_text(value: Any, limit: int = 1600) -> str:
    text = str(value or "")
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(
            lambda match: (
                f"{match.group(1)}[redacted]" if match.lastindex else "[redacted]"
            ),
            text,
        )
    return text[:limit]


def _clean_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return "[omitted]"
    if isinstance(value, str):
        return _clean_text(value, 1200)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_clean_value(item, depth=depth + 1) for item in value[:24]]
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in list(value.items())[:40]:
            key_text = str(key)
            lowered = key_text.casefold()
            if any(marker in lowered for marker in ("authorization", "api_key", "secret", "cookie", "environment", "system_prompt")):
                continue
            result[key_text[:100]] = _clean_value(item, depth=depth + 1)
        return result
    return _clean_text(value, 500)


def _permissions(value: Iterable[Any] | str | None) -> list[str]:
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = list(value or [])
    result: list[str] = []
    for item in values:
        name = str(item or "").strip().casefold()
        if name in KNOWN_PERMISSIONS and name not in result:
            result.append(name)
    if "read" not in result:
        result.insert(0, "read")
    return result


def _load(workspace: str | os.PathLike[str], token: str) -> Dict[str, Any]:
    token = str(token or "").strip()
    if not TOKEN_RE.fullmatch(token):
        raise ShareError("share not found", 404)
    path = _path(workspace, token)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ShareError("share not found", 404) from exc
    if not isinstance(record, dict) or not hmac.compare_digest(
        str(record.get("capability_hash") or ""), _token_hash(token)
    ):
        raise ShareError("share not found", 404)
    if record.get("revoked"):
        raise ShareError("share revoked", 410)
    try:
        expires_at = float(record.get("expires_at") or 0.0)
    except (TypeError, ValueError):
        expires_at = 0.0
    if expires_at <= time.time():
        raise ShareError("share expired", 410)
    return record


def _event(record: Dict[str, Any], kind: str, **fields: Any) -> None:
    events = record.setdefault("events", [])
    if not isinstance(events, list):
        events = []
        record["events"] = events
    events.append({
        "schema": SHARE_EVENT_SCHEMA,
        "kind": str(kind),
        "at": time.time(),
        **{str(key): _clean_value(value) for key, value in fields.items()},
    })
    record["events"] = events[-48:]


def _save(workspace: str | os.PathLike[str], token: str, record: Mapping[str, Any]) -> None:
    atomic_write_json(_path(workspace, token), dict(record))


def _base(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _tunnel_state(record: Dict[str, Any]) -> Dict[str, Any]:
    state = record.setdefault("tunnel", {})
    if not isinstance(state, dict):
        raise ShareError("tunnel state is malformed", 500)
    state.setdefault("schema", TUNNEL_SCHEMA)
    state.setdefault("next_sequence", 0)
    state.setdefault("invitations", {})
    state.setdefault("connections", {})
    state.setdefault("events", [])
    if not isinstance(state["invitations"], dict) or not isinstance(
        state["connections"], dict
    ):
        raise ShareError("tunnel state is malformed", 500)
    return state


def _tunnel_event(record: Dict[str, Any], kind: str, **fields: Any) -> Dict[str, Any]:
    state = _tunnel_state(record)
    sequence = int(state.get("next_sequence") or 0) + 1
    event = {
        "schema": TUNNEL_EVENT_SCHEMA,
        "sequence": sequence,
        "kind": str(kind),
        "at": time.time(),
        **{str(key): _clean_value(value) for key, value in fields.items()},
    }
    state["next_sequence"] = sequence
    events = list(state.get("events") or [])
    events.append(event)
    state["events"] = events[-256:]
    _event(record, kind, sequence=sequence, **fields)
    return event


def _tunnel_secret_hash(secret: str) -> str:
    return _token_hash(secret)


def _tunnel_aead(secret: str) -> Any:
    """Return the optional AEAD primitive, failing closed if unavailable."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        raise ShareError(
            "encrypted Tunnel relay requires the cryptography package", 503
        ) from exc
    value = str(secret or "").strip()
    if not TUNNEL_SECRET_RE.fullmatch(value):
        raise ShareError("tunnel connection is not valid", 403)
    key = hashlib.sha256(
        value.encode("utf-8") + b"\0hawking-tunnel-frame-v1"
    ).digest()
    return AESGCM(key)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: Any) -> bytes:
    raw = str(value or "").strip()
    if not raw or len(raw) > 160000:
        raise ShareError("Tunnel frame encoding is invalid", 400)
    try:
        return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    except (ValueError, TypeError) as exc:
        raise ShareError("Tunnel frame encoding is invalid", 400) from exc


def encode_tunnel_frame(
    reconnect_secret: str,
    *,
    request_id: str,
    sequence: int,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Encode a compact authenticated frame for direct or relay transport."""
    request = str(request_id or "").strip()
    if not request or len(request) > 160:
        raise ShareError("request_id is required", 400)
    try:
        number = max(0, int(sequence))
    except (TypeError, ValueError) as exc:
        raise ShareError("sequence must be an integer", 400) from exc
    if not isinstance(payload, Mapping):
        raise ShareError("Tunnel frame payload must be an object", 400)
    aad = json.dumps({
        "schema": TUNNEL_FRAME_SCHEMA,
        "request_id": request,
        "sequence": number,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    plaintext = json.dumps(
        _clean_value(dict(payload)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    nonce = secrets.token_bytes(12)
    ciphertext = _tunnel_aead(reconnect_secret).encrypt(nonce, plaintext, aad)
    return {
        "schema": TUNNEL_FRAME_SCHEMA,
        "encoding": "base64url",
        "aead": "AES-256-GCM",
        "request_id": request,
        "sequence": number,
        "nonce": _b64(nonce),
        "ciphertext": _b64(ciphertext),
    }


def decode_tunnel_frame(
    reconnect_secret: str, frame: Mapping[str, Any],
) -> Dict[str, Any]:
    """Verify and decode one Tunnel frame without accepting plaintext fallbacks."""
    if not isinstance(frame, Mapping) or frame.get("schema") != TUNNEL_FRAME_SCHEMA:
        raise ShareError("Tunnel frame schema is invalid", 400)
    request = str(frame.get("request_id") or "").strip()
    if not request or len(request) > 160:
        raise ShareError("request_id is required", 400)
    try:
        number = max(0, int(frame.get("sequence")))
    except (TypeError, ValueError) as exc:
        raise ShareError("sequence must be an integer", 400) from exc
    if frame.get("aead") != "AES-256-GCM" or frame.get("encoding") != "base64url":
        raise ShareError("Tunnel frame encryption is invalid", 400)
    aad = json.dumps({
        "schema": TUNNEL_FRAME_SCHEMA,
        "request_id": request,
        "sequence": number,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        plaintext = _tunnel_aead(reconnect_secret).decrypt(
            _unb64(frame.get("nonce")),
            _unb64(frame.get("ciphertext")),
            aad,
        )
        value = json.loads(plaintext.decode("utf-8"))
    except ShareError:
        raise
    except Exception as exc:
        raise ShareError("Tunnel frame authentication failed", 403) from exc
    if not isinstance(value, dict):
        raise ShareError("Tunnel frame payload is invalid", 400)
    return value


def _tunnel_connection(
    record: Dict[str, Any], secret: str,
) -> Tuple[str, Dict[str, Any]]:
    value = str(secret or "").strip()
    if not TUNNEL_SECRET_RE.fullmatch(value):
        raise ShareError("tunnel connection is not valid", 403)
    state = _tunnel_state(record)
    digest = _tunnel_secret_hash(value)
    for connection_id, connection in state["connections"].items():
        if isinstance(connection, Mapping) and hmac.compare_digest(
            str(connection.get("secret_hash") or ""), digest
        ):
            return str(connection_id), dict(connection)
    raise ShareError("tunnel connection is not valid", 403)


def _tunnel_transport(
    invitation: Mapping[str, Any], preference: Any,
) -> str:
    choice = str(preference or "auto").strip().casefold()
    if choice not in {"auto", "direct", "relay"}:
        raise ShareError("transport_preference must be auto, direct, or relay", 400)
    direct = bool(invitation.get("direct_available"))
    relay = bool(invitation.get("relay_available"))
    if choice == "relay":
        if not relay:
            raise ShareError("encrypted relay is unavailable", 503)
        return "encrypted_relay"
    if direct:
        return "direct_p2p"
    if relay:
        return "encrypted_relay"
    raise ShareError("no Tunnel transport is available", 503)


def _tunnel_request_payload(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ShareError("tunnel request payload must be an object", 400)
    return dict(_clean_value(value))


def _tunnel_model_rows() -> list[Dict[str, Any]]:
    # The Tunnel must expose Hawking's admitted Web roster, not an arbitrary
    # provider catalogue.  Provider discovery remains behind the existing
    # H-Web search owner and is never copied into a pairing record.
    from .auto_mode import REMOTE_AUTO_ROSTER

    rows = [
        {"id": model, "object": "model", "owned_by": "hawking", "scope": "cloud"}
        for model in REMOTE_AUTO_ROSTER
    ]
    rows.append({
        "id": "KIMI_P0_OPERATIONAL",
        "object": "model",
        "owned_by": "hawking",
        "scope": "local",
        "review_only": True,
    })
    return rows


def create_share(
    workspace: str | os.PathLike[str],
    *,
    target_type: str,
    target_id: str,
    permissions: Iterable[Any] | str | None = None,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    max_uses: int = 0,
    local_base: str = "http://hawking.localhost:8014",
    public_base: Optional[str] = None,
) -> Dict[str, Any]:
    kind = str(target_type or "").strip().casefold()
    target = str(target_id or "").strip()
    if kind not in {"goal", "conversation"}:
        raise ShareError("target_type must be goal or conversation", 400)
    if (kind == "goal" and not GOAL_RE.fullmatch(target)) or (
        kind == "conversation" and not SESSION_RE.fullmatch(target)
    ):
        raise ShareError("target_id is not a valid Hawking object", 400)
    try:
        ttl = min(MAX_TTL_SECONDS, max(60.0, float(ttl_seconds)))
    except (TypeError, ValueError) as exc:
        raise ShareError("ttl_seconds must be numeric", 400) from exc
    try:
        uses = max(0, int(max_uses))
    except (TypeError, ValueError) as exc:
        raise ShareError("max_uses must be an integer", 400) from exc
    token = secrets.token_urlsafe(9)
    if not TOKEN_RE.fullmatch(token):
        raise ShareError("could not create share capability", 500)
    now = time.time()
    record: Dict[str, Any] = {
        "schema": SHARE_SCHEMA,
        "capability_hash": _token_hash(token),
        "target_type": kind,
        "target_id": target,
        "permissions": _permissions(permissions),
        "created_at": now,
        "expires_at": now + ttl,
        "max_uses": uses,
        "uses": 0,
        "revoked": False,
        "idempotency": {},
        "events": [],
    }
    _event(record, "share.created", target=target, target_type=kind, permissions=record["permissions"])
    with _LOCK:
        _save(workspace, token, record)
    local = f"{_base(local_base)}/s/{token}"
    configured_public = _base(public_base or os.environ.get("HAWKING_PUBLIC_SHARE_BASE"))
    return {
        "kind": "hawking.share.created",
        "share_id": token,
        "scope": kind,
        "target_id": target,
        "permissions": list(record["permissions"]),
        "expires_at": record["expires_at"],
        "local_url": local,
        "external_url": f"{configured_public}/s/{token}" if configured_public else None,
        "public_bridge": "configured" if configured_public else "not_configured",
        "claim_boundary": "capability grants only the listed actions on this object",
    }


def create_tunnel_invitation(
    workspace: str | os.PathLike[str],
    token: str,
    *,
    capability: str,
    ttl_seconds: float = TUNNEL_INVITATION_TTL_SECONDS,
    direct_available: bool = True,
    relay_available: bool = True,
) -> Dict[str, Any]:
    """Mint one short-lived device invitation for an existing share.

    Only the invitation digest is durable.  The returned value is intended for
    the host's QR/browser handoff and is never placed in a receipt, event, or
    workspace file.  This is the pairing half of Tunnel; the existing share
    remains the host delegation and the only object scope.
    """
    if not hmac.compare_digest(str(capability or ""), str(token or "")):
        raise ShareError("share capability is invalid", 403)
    try:
        ttl = min(
            TUNNEL_MAX_INVITATION_TTL_SECONDS,
            max(30.0, float(ttl_seconds)),
        )
    except (TypeError, ValueError) as exc:
        raise ShareError("ttl_seconds must be numeric", 400) from exc
    if not direct_available and not relay_available:
        raise ShareError("at least one Tunnel transport is required", 503)
    invitation = secrets.token_urlsafe(18)
    pairing_id = f"pair-{uuid.uuid4().hex[:20]}"
    now = time.time()
    with _LOCK:
        record = _load(workspace, token)
        state = _tunnel_state(record)
        state["invitations"][pairing_id] = {
            "invitation_hash": _token_hash(invitation),
            "created_at": now,
            "expires_at": now + ttl,
            "used": False,
            "direct_available": bool(direct_available),
            "relay_available": bool(relay_available),
        }
        _tunnel_event(
            record,
            "tunnel.invitation_created",
            pairing_id=pairing_id,
            expires_at=now + ttl,
            direct_available=bool(direct_available),
            relay_available=bool(relay_available),
        )
        _save(workspace, token, record)
    return {
        "kind": "hawking.tunnel.invitation",
        "schema": "hawking.tunnel.invitation.v1",
        "share_id": str(token),
        "pairing_id": pairing_id,
        "invitation": invitation,
        "expires_at": now + ttl,
        "pair_url": f"/s/{token}/tunnel/pair",
        "qr_payload": json.dumps({
            "kind": "hawking.tunnel.pair",
            "pair_url": f"/s/{token}/tunnel/pair",
            "pairing_id": pairing_id,
            "invitation": invitation,
        }, ensure_ascii=False, separators=(",", ":")),
        "transport_options": {
            "direct_p2p": bool(direct_available),
            "encrypted_relay": bool(relay_available),
        },
        "claim_boundary": (
            "one-use device enrollment for this shared object; no local admin "
            "cookie, daemon credential, or workspace enumeration"
        ),
    }


def pair_tunnel(
    workspace: str | os.PathLike[str],
    token: str,
    *,
    invitation: str,
    device_id: str,
    requested_permissions: Iterable[Any] | str | None = None,
    session_permissions: Iterable[Any] | str | None = None,
    transport_preference: str = "auto",
) -> Dict[str, Any]:
    """Enroll a device and return a one-time reconnect secret.

    The granted capability is the intersection of the host share, the device
    request, and the session authority.  The raw reconnect secret is returned
    exactly once; only its digest is stored.
    """
    invite_value = str(invitation or "").strip()
    device = str(device_id or "").strip()
    if not TUNNEL_SECRET_RE.fullmatch(invite_value):
        raise ShareError("Tunnel invitation is invalid", 403)
    if not DEVICE_RE.fullmatch(device):
        raise ShareError("device_id is invalid", 400)
    invite_digest = _token_hash(invite_value)
    now = time.time()
    with _LOCK:
        record = _load(workspace, token)
        state = _tunnel_state(record)
        pairing_id = ""
        invitation_record: Optional[Mapping[str, Any]] = None
        for candidate_id, candidate in state["invitations"].items():
            if not isinstance(candidate, Mapping):
                continue
            if hmac.compare_digest(
                str(candidate.get("invitation_hash") or ""), invite_digest
            ):
                pairing_id = str(candidate_id)
                invitation_record = candidate
                break
        if not pairing_id or invitation_record is None:
            raise ShareError("Tunnel invitation is invalid", 403)
        if invitation_record.get("used"):
            raise ShareError("Tunnel invitation has already been used", 410)
        try:
            expires_at = float(invitation_record.get("expires_at") or 0.0)
        except (TypeError, ValueError):
            expires_at = 0.0
        if expires_at <= now:
            raise ShareError("Tunnel invitation has expired", 410)
        host_permissions = set(_permissions(record.get("permissions")))
        device_permissions = set(
            _permissions(
                requested_permissions
                if requested_permissions is not None else host_permissions
            )
        )
        session_grant = set(
            _permissions(
                session_permissions
                if session_permissions is not None else host_permissions
            )
        )
        granted = sorted(host_permissions & device_permissions & session_grant)
        if not granted:
            raise ShareError("Tunnel capability intersection is empty", 403)
        transport = _tunnel_transport(invitation_record, transport_preference)
        connection_id = f"conn-{uuid.uuid4().hex[:20]}"
        reconnect_secret = secrets.token_urlsafe(24)
        # Do not advertise a relay path unless this runtime can actually
        # authenticate its application frames.  The external rendezvous/
        # packet transport is still outside this adapter; its wire contract
        # is represented by encode/decode_tunnel_frame below.
        _tunnel_aead(reconnect_secret)
        state["connections"][connection_id] = {
            "secret_hash": _tunnel_secret_hash(reconnect_secret),
            "device_id_hash": _token_hash(device),
            "pairing_id": pairing_id,
            "granted_permissions": granted,
            "session_permissions": sorted(session_grant),
            "transport": transport,
            "wire_security": "AES-256-GCM",
            "connected": True,
            "created_at": now,
            "last_seen_at": now,
            "reconnects": 0,
            "cursor": 0,
            "request_idempotency": {},
        }
        invitation_record = dict(invitation_record)
        invitation_record["used"] = True
        state["invitations"][pairing_id] = invitation_record
        event = _tunnel_event(
            record,
            "tunnel.paired",
            connection_id=connection_id,
            pairing_id=pairing_id,
            device_id_hash=_token_hash(device),
            granted_permissions=granted,
            transport=transport,
        )
        connection = state["connections"][connection_id]
        connection["cursor"] = int(event.get("sequence") or 0)
        _save(workspace, token, record)
    return {
        "kind": "hawking.tunnel.paired",
        "schema": TUNNEL_SCHEMA,
        "share_id": str(token),
        "connection_id": connection_id,
        "reconnect_secret": reconnect_secret,
        "pairing_id": pairing_id,
        "granted_permissions": granted,
        "operations": sorted(
            operation for operation, permission in TUNNEL_OPERATION_PERMISSIONS.items()
            if permission in granted
        ),
        "transport": transport,
        "wire_security": "AES-256-GCM",
        "cursor": int(event.get("sequence") or 0),
        "claim_boundary": (
            "scoped Tunnel capability only; reconnect_secret is a bearer for "
            "this connection and is never a local admin credential"
        ),
    }


def _tunnel_connection_view(
    token: str, connection_id: str, connection: Mapping[str, Any],
) -> Dict[str, Any]:
    granted = list(connection.get("granted_permissions") or [])
    return {
        "schema": TUNNEL_SCHEMA,
        "share_id": str(token),
        "connection_id": connection_id,
        "transport": connection.get("transport"),
        "wire_security": connection.get("wire_security") or "AES-256-GCM",
        "connected": bool(connection.get("connected")),
        "granted_permissions": granted,
        "operations": sorted(
            operation for operation, permission in TUNNEL_OPERATION_PERMISSIONS.items()
            if permission in granted
        ),
        "cursor": int(connection.get("cursor") or 0),
        "reconnects": int(connection.get("reconnects") or 0),
    }


def reconnect_tunnel(
    workspace: str | os.PathLike[str],
    token: str,
    *,
    reconnect_secret: str,
    after_sequence: int = 0,
) -> Dict[str, Any]:
    """Resume a disconnected Tunnel without re-enrolling the device."""
    try:
        after = max(0, int(after_sequence))
    except (TypeError, ValueError) as exc:
        raise ShareError("after_sequence must be an integer", 400) from exc
    with _LOCK:
        record = _load(workspace, token)
        state = _tunnel_state(record)
        connection_id, connection = _tunnel_connection(record, reconnect_secret)
        connection["connected"] = True
        connection["last_seen_at"] = time.time()
        connection["reconnects"] = int(connection.get("reconnects") or 0) + 1
        state["connections"][connection_id] = connection
        _tunnel_event(record, "tunnel.reconnected", connection_id=connection_id)
        events = [
            dict(event) for event in list(state.get("events") or [])
            if isinstance(event, Mapping) and int(event.get("sequence") or 0) > after
        ]
        connection["cursor"] = int(state.get("next_sequence") or 0)
        _save(workspace, token, record)
        return {
            "kind": "hawking.tunnel.reconnected",
            "connection": _tunnel_connection_view(token, connection_id, connection),
            "events": events,
            "through_sequence": int(state.get("next_sequence") or 0),
            "gap": bool(events and int(events[0].get("sequence") or 0) > after + 1),
        }


def disconnect_tunnel(
    workspace: str | os.PathLike[str],
    token: str,
    *,
    reconnect_secret: str,
    request_id: str,
) -> Dict[str, Any]:
    """Close one Tunnel connection; repeated request IDs are idempotent."""
    request = str(request_id or "").strip()
    if not request or len(request) > 160:
        raise ShareError("request_id is required", 400)
    with _LOCK:
        record = _load(workspace, token)
        state = _tunnel_state(record)
        connection_id, connection = _tunnel_connection(record, reconnect_secret)
        idem = connection.setdefault("request_idempotency", {})
        if not isinstance(idem, dict):
            raise ShareError("tunnel connection is malformed", 500)
        prior = idem.get(request)
        if isinstance(prior, Mapping):
            return {**dict(prior.get("response") or {}), "duplicate": True}
        connection["connected"] = False
        connection["last_seen_at"] = time.time()
        event = _tunnel_event(
            record, "tunnel.disconnected", connection_id=connection_id,
        )
        connection["cursor"] = int(event.get("sequence") or 0)
        response = {
            "kind": "hawking.tunnel.disconnected",
            "connection_id": connection_id,
            "connected": False,
            "cursor": connection["cursor"],
        }
        idem[request] = {"response": response, "at": time.time()}
        if len(idem) > 64:
            for key in list(idem)[:-64]:
                idem.pop(key, None)
        state["connections"][connection_id] = connection
        _save(workspace, token, record)
        return response


def request_tunnel(
    workspace: str | os.PathLike[str],
    token: str,
    *,
    reconnect_secret: str,
    request_id: str,
    operation: str,
    payload: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Admit one scoped Tunnel request and publish a replayable result.

    Read operations are fulfilled from the existing Goal/Session owners.  A
    prompt or Build request is admitted as a bounded H-Web dispatch envelope;
    the serving owner remains responsible for executing that envelope, so the
    Tunnel never grows a second model or mutation executor.
    """
    request = str(request_id or "").strip()
    op = str(operation or "").strip().casefold()
    if not request or len(request) > 160:
        raise ShareError("request_id is required", 400)
    if op not in TUNNEL_OPERATION_PERMISSIONS:
        raise ShareError("unsupported Tunnel operation", 400)
    body = _tunnel_request_payload(payload)
    with _LOCK:
        record = _load(workspace, token)
        state = _tunnel_state(record)
        connection_id, connection = _tunnel_connection(record, reconnect_secret)
        if not connection.get("connected"):
            raise ShareError("Tunnel connection is disconnected", 409)
        required = TUNNEL_OPERATION_PERMISSIONS[op]
        if required not in set(connection.get("granted_permissions") or []):
            raise ShareError(f"Tunnel does not permit {op}", 403)
        idem = connection.setdefault("request_idempotency", {})
        if not isinstance(idem, dict):
            raise ShareError("tunnel connection is malformed", 500)
        prior = idem.get(request)
        if isinstance(prior, Mapping):
            if prior.get("operation") != op:
                raise ShareError("request_id was used for another operation", 409)
            return {**dict(prior.get("response") or {}), "duplicate": True}
        target_type = str(record.get("target_type") or "")
        target_id = str(record.get("target_id") or "")
        if op == "list_chats":
            conversations = []
            if target_type == "conversation":
                conversations = [{
                    "session_id": target_id,
                    "scope": "shared_object",
                }]
            response: Dict[str, Any] = {
                "kind": "hawking.tunnel.chats",
                "request_id": request,
                "conversations": conversations,
            }
        elif op == "list_models":
            response = {
                "kind": "hawking.tunnel.models",
                "request_id": request,
                "models": _tunnel_model_rows(),
            }
        elif op == "retrieve_result":
            if target_type == "goal":
                result = _goal_snapshot(workspace, target_id)
            else:
                result = _conversation_snapshot(workspace, target_id)
            response = {
                "kind": "hawking.tunnel.result",
                "request_id": request,
                "result": result,
            }
        else:
            message = str(body.get("message") or "").strip()
            if not message or len(message) > 8000:
                raise ShareError("Tunnel prompt/build message is required and bounded", 400)
            if op == "scoped_build" and target_type != "goal":
                raise ShareError("scoped_build requires a Goal share", 403)
            response = {
                "kind": "hawking.tunnel.dispatch",
                "request_id": request,
                "operation": op,
                "accepted": True,
                "execution_state": "ADMITTED_TO_HWEB",
                "dispatch": {
                    "owner": "hawkingd",
                    "route": "/hawking/web/chat",
                    "target_type": target_type,
                    "target_id": target_id,
                    "message": message,
                    "builder_scope": op == "scoped_build",
                },
                "claim_boundary": (
                    "request admitted to the canonical H-Web owner; this "
                    "Tunnel function does not execute a provider or mutate a repo"
                ),
            }
        event = _tunnel_event(
            record,
            "tunnel.request_completed",
            connection_id=connection_id,
            request_id=request,
            operation=op,
        )
        response["cursor"] = int(event.get("sequence") or 0)
        idem[request] = {"operation": op, "response": response, "at": time.time()}
        if len(idem) > 64:
            for key in list(idem)[:-64]:
                idem.pop(key, None)
        connection["last_seen_at"] = time.time()
        connection["cursor"] = int(event.get("sequence") or 0)
        state["connections"][connection_id] = connection
        _save(workspace, token, record)
        return response


def tunnel_action(
    workspace: str | os.PathLike[str],
    token: str,
    *,
    action: str,
    body: Mapping[str, Any],
) -> Dict[str, Any]:
    """Dispatch the narrow public `/s/<share>/tunnel/*` protocol."""
    action = str(action or "").strip().casefold()
    values = dict(body or {})
    if action == "pair":
        return pair_tunnel(
            workspace,
            token,
            invitation=str(values.get("invitation") or ""),
            device_id=str(values.get("device_id") or ""),
            requested_permissions=values.get("requested_permissions"),
            session_permissions=values.get("session_permissions"),
            transport_preference=str(values.get("transport_preference") or "auto"),
        )
    if action == "reconnect":
        return reconnect_tunnel(
            workspace,
            token,
            reconnect_secret=str(values.get("reconnect_secret") or ""),
            after_sequence=values.get("after_sequence", 0),
        )
    if action == "disconnect":
        return disconnect_tunnel(
            workspace,
            token,
            reconnect_secret=str(values.get("reconnect_secret") or ""),
            request_id=str(values.get("request_id") or ""),
        )
    if action == "request":
        return request_tunnel(
            workspace,
            token,
            reconnect_secret=str(values.get("reconnect_secret") or ""),
            request_id=str(values.get("request_id") or ""),
            operation=str(values.get("operation") or ""),
            payload=values.get("payload"),
        )
    raise ShareError("unsupported Tunnel action", 400)


def _goal_snapshot(workspace: str | os.PathLike[str], goal_id: str) -> Dict[str, Any]:
    from .goal_surface import load_goal, load_result
    from .workunit_owner import WorkunitOwner

    goal = load_goal(workspace, goal_id)
    if not goal:
        raise ShareError("shared Goal not found", 404)
    wid = str(goal.get("workunit_id") or goal_id)
    try:
        workunit = WorkunitOwner(workspace).status(wid)
    except Exception as exc:  # the Goal remains readable if its owner is gone
        workunit = {"workunit_id": wid, "state": "UNKNOWN", "error": type(exc).__name__}
    allowed_goal = {
        key: goal.get(key)
        for key in (
            "goal_id", "objective", "status", "phase", "goal_mode",
            "created_at", "updated_at", "resident_assignment", "workunit_id",
            "parent_goal_ids", "auto_plan", "budget_plan",
        )
        if goal.get(key) is not None
    }
    allowed_worker_keys = (
        "workunit_id", "goal_id", "state", "current_phase", "worker_model",
        "worker_id", "worker_status", "worker_kill_reason", "last_turn",
        "last_outcome", "worker_cost_usd", "parent_worker_id",
        "parent_workunit_id", "child_worker_ids", "qualification", "resource_state",
        "exact_next_action", "checkpoint_path", "worker_packet_path",
    )
    worker_view = {
        key: workunit.get(key)
        for key in allowed_worker_keys
        if workunit.get(key) is not None
    }
    result = load_result(workspace, goal_id)
    result_view = None
    if isinstance(result, Mapping):
        result_view = {
            key: result.get(key)
            for key in ("status", "summary", "completion", "output_excerpt", "evidence")
            if result.get(key) is not None
        }
    evidence = []
    for item in list(workunit.get("evidence") or [])[-12:]:
        if not isinstance(item, Mapping):
            continue
        evidence.append({
            key: item.get(key)
            for key in ("kind", "at", "turn", "tool", "name", "path", "status", "signal", "outcome", "note", "summary")
            if item.get(key) is not None
        })
    return _clean_value({
        "kind": "hawking.goal.share",
        "goal_id": goal_id,
        "objective": allowed_goal.get("objective", ""),
        "status": allowed_goal.get("status") or worker_view.get("state"),
        "phase": allowed_goal.get("phase") or worker_view.get("current_phase"),
        "goal": allowed_goal,
        "workunit": worker_view,
        "workers": [{
            "worker_id": worker_view.get("worker_id"),
            "model": worker_view.get("worker_model"),
            "status": worker_view.get("worker_status") or worker_view.get("state"),
            "cost_usd": worker_view.get("worker_cost_usd", 0.0),
        }],
        "recent_events": evidence,
        "result": result_view,
    })


def _conversation_snapshot(workspace: str | os.PathLike[str], session_id: str) -> Dict[str, Any]:
    from .session import SessionStore

    session = SessionStore(str(workspace)).load(session_id)
    if session is None:
        raise ShareError("shared conversation not found", 404)
    messages = []
    for item in list(getattr(session, "messages", []) or [])[-40:]:
        if not isinstance(item, Mapping) or item.get("role") not in {"user", "assistant"}:
            continue
        messages.append({"role": item.get("role"), "content": _clean_text(item.get("content"), 4000)})
    ui = getattr(session, "ui", {}) or {}
    comments = list(ui.get("external_comments") or []) if isinstance(ui, Mapping) else []
    steering = list(ui.get("external_steering") or []) if isinstance(ui, Mapping) else []
    return _clean_value({
        "kind": "hawking.conversation.share",
        "session_id": session_id,
        "messages": messages,
        "goal_ids": list((getattr(session, "ui", {}) or {}).get("goal_ids") or [])[:32],
        "external_comments": comments[-12:],
        "external_steering": steering[-12:],
    })


def snapshot_share(workspace: str | os.PathLike[str], token: str) -> Dict[str, Any]:
    with _LOCK:
        record = _load(workspace, token)
        if record.get("target_type") == "goal":
            snapshot = _goal_snapshot(workspace, str(record.get("target_id") or ""))
        else:
            snapshot = _conversation_snapshot(workspace, str(record.get("target_id") or ""))
        allowed = list(record.get("permissions") or [])
        snapshot.update({
            "share_id": str(token),
            "expires_at": record.get("expires_at"),
            "allowed_actions": allowed,
            "manifest": f"/s/{token}/manifest.json",
        })
        _event(record, "share.read", target=record.get("target_id"))
        _save(workspace, token, record)
        return snapshot


def manifest_share(workspace: str | os.PathLike[str], token: str) -> Dict[str, Any]:
    with _LOCK:
        record = _load(workspace, token)
    actions: Dict[str, Dict[str, str]] = {}
    for action in ("comment", "steer"):
        if action in set(record.get("permissions") or []):
            actions[action] = {"method": "POST", "path": f"/s/{token}/{action}"}
    return {
        "kind": "hawking.share.manifest",
        "share_id": token,
        "read": f"/s/{token}.json",
        "actions": actions,
        "tunnel": {
            "pair": {"method": "POST", "path": f"/s/{token}/tunnel/pair"},
            "reconnect": {"method": "POST", "path": f"/s/{token}/tunnel/reconnect"},
            "request": {"method": "POST", "path": f"/s/{token}/tunnel/request"},
            "disconnect": {"method": "POST", "path": f"/s/{token}/tunnel/disconnect"},
            "host_invite": {"method": "POST", "path": "/hawking/web/tunnel"},
        },
        "expires_at": record.get("expires_at"),
        "claim_boundary": (
            "only this shared object; Tunnel pairing still requires a one-use "
            "invitation and never grants daemon, provider, filesystem, or Goal enumeration"
        ),
    }


def html_share(workspace: str | os.PathLike[str], token: str) -> bytes:
    snapshot = snapshot_share(workspace, token)
    title = html.escape(str(snapshot.get("goal_id") or snapshot.get("session_id") or "Hawking share"))
    payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    token_json = json.dumps(str(token))
    return f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>hawking · {title}</title>
<style>html,body{{margin:0;background:#000;color:#eee;font:14px/1.55 system-ui,sans-serif}}main{{width:min(720px,calc(100% - 40px));margin:48px auto}}h1{{font-size:14px;font-weight:500}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;color:#bbb;border-top:1px solid #333;padding-top:18px}}textarea{{width:100%;box-sizing:border-box;background:#080808;color:#eee;border:1px solid #444;border-radius:6px;padding:9px;margin-top:8px}}button{{margin:8px 6px 0 0;background:transparent;color:#eee;border:1px solid #555;border-radius:6px;padding:7px 10px;cursor:pointer}}button:hover{{border-color:#eee}}small{{color:#888}}</style></head>
<body><main><h1>hawking · shared object</h1><small id=\"meta\"></small><pre id=\"snapshot\"></pre><section id=\"actions\" hidden><textarea id=\"message\" rows=\"4\" placeholder=\"Comment or steer Hawking…\"></textarea><button data-action=\"comment\">Comment</button><button data-action=\"steer\">Steer</button><small id=\"result\"></small></section></main>
<script>const share={payload};const capability={token_json};const meta=document.getElementById('meta');const snapshot=document.getElementById('snapshot');const actions=document.getElementById('actions');const message=document.getElementById('message');const result=document.getElementById('result');meta.textContent=`${{share.status||'shared'}} · expires ${{new Date(share.expires_at*1000).toLocaleString()}} · ${{(share.allowed_actions||[]).join(', ')}}`;snapshot.textContent=JSON.stringify(share,null,2);const allowed=new Set(share.allowed_actions||[]);if(allowed.has('comment')||allowed.has('steer'))actions.hidden=false;document.querySelectorAll('[data-action]').forEach(button=>{{const action=button.dataset.action;button.hidden=!allowed.has(action);button.onclick=async()=>{{const text=message.value.trim();if(!text)return;const response=await fetch(location.pathname+'/'+action,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{capability,idempotency_key:crypto.randomUUID(),message:text}})}});const data=await response.json();result.textContent=response.ok?'accepted':(data.error&&data.error.message)||'action failed';if(response.ok)message.value='';}}}});</script></body></html>""".encode("utf-8")


def act_share(
    workspace: str | os.PathLike[str],
    token: str,
    *,
    action: str,
    capability: str,
    message: str,
    idempotency_key: str,
) -> Dict[str, Any]:
    action = str(action or "").strip().casefold()
    if action not in {"comment", "steer"}:
        raise ShareError("unsupported share action", 400)
    if not hmac.compare_digest(str(capability or ""), str(token or "")):
        raise ShareError("share capability is invalid", 403)
    note = _clean_text(message, 2000).strip()
    if not note:
        raise ShareError("message is required", 400)
    idem = str(idempotency_key or "").strip()
    if not idem or len(idem) > 160:
        raise ShareError("idempotency_key is required", 400)
    with _LOCK:
        record = _load(workspace, token)
        permissions = set(record.get("permissions") or [])
        if action not in permissions:
            raise ShareError(f"share does not permit {action}", 403)
        saved = record.setdefault("idempotency", {})
        prior = saved.get(idem) if isinstance(saved, Mapping) else None
        if isinstance(prior, Mapping):
            if prior.get("action") != action:
                raise ShareError("idempotency key was used for another action", 409)
            return {**dict(prior.get("response") or {}), "duplicate": True}
        max_uses = int(record.get("max_uses") or 0)
        if max_uses and int(record.get("uses") or 0) >= max_uses:
            raise ShareError("share use limit reached", 410)
        target_type = str(record.get("target_type") or "")
        target_id = str(record.get("target_id") or "")
        now = time.time()
        if target_type == "goal":
            from .goal_surface import load_goal, update_goal
            goal = load_goal(workspace, target_id)
            if not goal:
                raise ShareError("shared Goal not found", 404)
            if action == "comment":
                comments = list(goal.get("external_comments") or [])
                comments.append({"source": "share", "at": now, "message": note})
                updated = update_goal(workspace, target_id, external_comments=comments[-32:]) or goal
                response = {"kind": "hawking.share.comment", "share_id": token, "goal_id": target_id, "status": updated.get("status"), "accepted": True}
            else:
                from .workunit_owner import WorkunitOwner
                wid = str(goal.get("workunit_id") or target_id)
                owner = WorkunitOwner(workspace)
                record_owner = owner.load(wid)
                record_owner.exact_next_action = note
                record_owner.current_phase = "STEERED"
                record_owner.evidence.append({"at": now, "kind": "external_share_steer", "note": note})
                owner.save(record_owner)
                updated = update_goal(workspace, target_id, phase="STEERED", last_operator_action="share_steer") or goal
                response = {"kind": "hawking.share.steer", "share_id": token, "goal_id": target_id, "status": updated.get("status"), "accepted": True}
        else:
            # A conversation share is still a Hawking-owned session object.
            # Persist external annotations on that session so a general chat
            # share is useful beyond a read-only snapshot.  Steering is kept
            # as a durable session note; it does not silently become a Goal or
            # grant the external caller any broader authority.
            from .session import SessionStore
            session = SessionStore(str(workspace)).load(target_id)
            if session is None:
                raise ShareError("shared conversation not found", 404)
            ui = dict(getattr(session, "ui", {}) or {})
            if action == "comment":
                comments = list(ui.get("external_comments") or [])
                comments.append({"source": "share", "at": now, "message": note})
                ui["external_comments"] = comments[-32:]
            else:
                steering = list(ui.get("external_steering") or [])
                steering.append({"source": "share", "at": now, "message": note})
                ui["external_steering"] = steering[-32:]
                session.steering = [str(item) for item in list(getattr(session, "steering", []) or [])[-63:]] + [note]
            session.ui = ui
            SessionStore(str(workspace)).save(session)
            response = {
                "kind": f"hawking.share.{action}",
                "share_id": token,
                "session_id": target_id,
                "accepted": True,
                "scope": "conversation",
            }
        _event(record, f"share.{action}", target=target_id)
        record["uses"] = int(record.get("uses") or 0) + 1
        saved[idem] = {"action": action, "response": response, "at": now}
        if len(saved) > 64:
            for key in list(saved)[:-64]:
                saved.pop(key, None)
        _save(workspace, token, record)
        return response


def revoke_share(workspace: str | os.PathLike[str], token: str) -> Dict[str, Any]:
    with _LOCK:
        record = _load(workspace, token)
        record["revoked"] = True
        state = _tunnel_state(record)
        for connection in state["connections"].values():
            if isinstance(connection, dict):
                connection["connected"] = False
                connection["revoked_at"] = time.time()
        _tunnel_event(record, "tunnel.revoked", target=record.get("target_id"))
        _event(record, "share.revoked", target=record.get("target_id"))
        _save(workspace, token, record)
    return {"kind": "hawking.share.revoked", "share_id": token, "revoked": True}


def parse_share_get(path: str) -> Tuple[str, str]:
    value = str(path or "").strip().rstrip("/")
    if not value.startswith("/s/"):
        raise ShareError("share not found", 404)
    tail = value[3:]
    if tail.endswith("/manifest.json"):
        return tail[:-14], "manifest"
    if tail.endswith(".json"):
        return tail[:-5], "json"
    return tail, "html"


def parse_share_post(path: str) -> Tuple[str, str]:
    value = str(path or "").strip().rstrip("/")
    if not value.startswith("/s/"):
        raise ShareError("share not found", 404)
    tail = value[3:]
    try:
        token, action = tail.split("/", 1)
    except ValueError as exc:
        raise ShareError("share action is required", 400) from exc
    return token, action


__all__ = [
    "DEFAULT_TTL_SECONDS", "ShareError", "TUNNEL_SCHEMA",
    "act_share", "create_share", "create_tunnel_invitation",
    "decode_tunnel_frame", "disconnect_tunnel", "encode_tunnel_frame",
    "html_share", "manifest_share", "pair_tunnel",
    "parse_share_get", "parse_share_post", "reconnect_tunnel",
    "request_tunnel", "revoke_share", "snapshot_share", "tunnel_action",
]
def tunnel_implementation_ready(health: dict) -> bool:
    """Return True only when the bridge is connected and not degraded."""
    return bool(health.get("connected")) and not bool(health.get("degraded"))


def with_tunnel_implementation_ready(health: dict) -> dict:
    """Return a copy of ``health`` with the tunnel readiness flag attached."""
    result = dict(health)
    result["tunnel_implementation_ready"] = tunnel_implementation_ready(health)
    return result
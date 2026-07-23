#!/usr/bin/env python3.12
"""Independent Keychain-backed producer authentication for GLM-5.2 evidence.

This key is deliberately distinct from the Telegram receipt key.  It is never
accepted through argv or the environment, and status output exposes only a
domain-separated identity digest.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import secrets
import subprocess
from typing import Any, Callable, Mapping, Protocol, Sequence

from glm52_common import Glm52Error
from glm52_state import EvidenceAuthConfig, StateError


KEYCHAIN_ACCOUNT = "hawking-glm52-gravity"
EVIDENCE_HMAC_SERVICE = "com.hawking.glm52.gravity.evidence.producer-hmac-key"
HMAC_KEY_BYTES = 32


class EvidenceSecurityError(Glm52Error):
    """Evidence signing credentials are missing, malformed, or unsafe."""


class Keychain(Protocol):
    def get(self, service: str) -> str | None: ...

    def set(self, service: str, value: str) -> None: ...


class MacOSKeychain:
    """Single-service macOS Keychain adapter with native secret writes.

    ``security add-generic-password -w`` does *not* read a password from stdin
    when ``-w`` has no argument; on macOS it silently stores an empty password.
    Supplying the value after ``-w`` would expose it through the process argv.
    Writes therefore go through Security.framework in this process, while the
    read-only CLI remains useful for retrieving an already protected item.
    """

    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        native_reader: Callable[[str, str], str | None] | None = None,
        native_writer: Callable[[str, str, str], None] | None = None,
    ) -> None:
        # ``runner`` remains injectable for API compatibility with earlier
        # tests/callers, but credentials never cross a subprocess boundary.
        self._runner = runner
        self._native_reader = native_reader or _get_generic_password_native
        self._native_writer = native_writer or _set_generic_password_native

    @staticmethod
    def _service(service: str) -> str:
        if service != EVIDENCE_HMAC_SERVICE:
            raise EvidenceSecurityError("unrecognized GLM evidence Keychain service")
        return service

    def get(self, service: str) -> str | None:
        service = self._service(service)
        try:
            value = self._native_reader(service, KEYCHAIN_ACCOUNT)
        except Exception:
            raise EvidenceSecurityError("macOS Keychain read failed") from None
        if value is None:
            return None
        if not value or "\n" in value or "\r" in value:
            raise EvidenceSecurityError("macOS Keychain returned an invalid credential")
        return value

    def set(self, service: str, value: str) -> None:
        service = self._service(service)
        if not isinstance(value, str) or not value or any(
            marker in value for marker in ("\n", "\r", "\x00")
        ):
            raise EvidenceSecurityError("evidence credential must be one non-empty line")
        try:
            self._native_writer(service, KEYCHAIN_ACCOUNT, value)
        except Exception:
            raise EvidenceSecurityError("macOS Keychain write failed") from None


def _set_generic_password_native(service: str, account: str, value: str) -> None:
    """Add or replace one generic password through Security.framework.

    The password bytes remain in this process.  They are never placed in an
    environment variable, filesystem path, subprocess stdin, or argv.
    """

    security = ctypes.CDLL(
        "/System/Library/Frameworks/Security.framework/Security"
    )
    core_foundation = ctypes.CDLL(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )
    uint32 = ctypes.c_uint32
    void_p = ctypes.c_void_p

    find = security.SecKeychainFindGenericPassword
    find.argtypes = [
        void_p,
        uint32,
        ctypes.c_char_p,
        uint32,
        ctypes.c_char_p,
        ctypes.POINTER(uint32),
        ctypes.POINTER(void_p),
        ctypes.POINTER(void_p),
    ]
    find.restype = ctypes.c_int32
    add = security.SecKeychainAddGenericPassword
    add.argtypes = [
        void_p,
        uint32,
        ctypes.c_char_p,
        uint32,
        ctypes.c_char_p,
        uint32,
        void_p,
        ctypes.POINTER(void_p),
    ]
    add.restype = ctypes.c_int32
    modify = security.SecKeychainItemModifyAttributesAndData
    modify.argtypes = [void_p, void_p, uint32, void_p]
    modify.restype = ctypes.c_int32
    free_content = security.SecKeychainItemFreeContent
    free_content.argtypes = [void_p, void_p]
    free_content.restype = ctypes.c_int32
    release = core_foundation.CFRelease
    release.argtypes = [void_p]
    release.restype = None

    service_bytes = service.encode("utf-8")
    account_bytes = account.encode("utf-8")
    secret_bytes = value.encode("utf-8")
    secret_buffer = ctypes.create_string_buffer(secret_bytes)
    secret_pointer = ctypes.cast(secret_buffer, void_p)
    password_length = uint32()
    password_data = void_p()
    item = void_p()
    status = find(
        None,
        len(service_bytes),
        service_bytes,
        len(account_bytes),
        account_bytes,
        ctypes.byref(password_length),
        ctypes.byref(password_data),
        ctypes.byref(item),
    )
    try:
        if status == 0:
            if password_data.value:
                free_content(None, password_data)
                password_data = void_p()
            status = modify(item, None, len(secret_bytes), secret_pointer)
        elif status == -25300:  # errSecItemNotFound
            status = add(
                None,
                len(service_bytes),
                service_bytes,
                len(account_bytes),
                account_bytes,
                len(secret_bytes),
                secret_pointer,
                None,
            )
        if status != 0:
            raise EvidenceSecurityError("macOS Security.framework rejected the credential")
    finally:
        if password_data.value:
            free_content(None, password_data)
        if item.value:
            release(item)


def _get_generic_password_native(service: str, account: str) -> str | None:
    """Read one generic password in-process and immediately release its buffer."""

    security = ctypes.CDLL(
        "/System/Library/Frameworks/Security.framework/Security"
    )
    uint32 = ctypes.c_uint32
    void_p = ctypes.c_void_p
    find = security.SecKeychainFindGenericPassword
    find.argtypes = [
        void_p,
        uint32,
        ctypes.c_char_p,
        uint32,
        ctypes.c_char_p,
        ctypes.POINTER(uint32),
        ctypes.POINTER(void_p),
        ctypes.POINTER(void_p),
    ]
    find.restype = ctypes.c_int32
    free_content = security.SecKeychainItemFreeContent
    free_content.argtypes = [void_p, void_p]
    free_content.restype = ctypes.c_int32

    service_bytes = service.encode("utf-8")
    account_bytes = account.encode("utf-8")
    password_length = uint32()
    password_data = void_p()
    status = find(
        None,
        len(service_bytes),
        service_bytes,
        len(account_bytes),
        account_bytes,
        ctypes.byref(password_length),
        ctypes.byref(password_data),
        None,
    )
    if status == -25300:  # errSecItemNotFound
        return None
    if status != 0 or not password_data.value:
        raise EvidenceSecurityError("macOS Security.framework could not read the credential")
    try:
        raw = ctypes.string_at(password_data, password_length.value)
    finally:
        free_content(None, password_data)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise EvidenceSecurityError("macOS Keychain credential is not UTF-8") from None


def _encode(key: bytes) -> str:
    if not isinstance(key, bytes) or len(key) != HMAC_KEY_BYTES:
        raise EvidenceSecurityError("evidence authentication key must be exactly 32 bytes")
    return base64.urlsafe_b64encode(key).decode("ascii")


def _decode(value: Any) -> bytes:
    if not isinstance(value, str) or not value or any(
        marker in value for marker in ("\n", "\r", "\x00")
    ):
        raise EvidenceSecurityError("evidence authentication credential is invalid")
    try:
        key = base64.b64decode(value, altchars=b"-_", validate=True)
    except (ValueError, TypeError):
        raise EvidenceSecurityError("evidence authentication credential is invalid") from None
    if len(key) != HMAC_KEY_BYTES or _encode(key) != value:
        raise EvidenceSecurityError("evidence authentication credential is invalid")
    return key


def _identity_digest(key: bytes) -> str:
    return hashlib.sha256(
        b"hawking.glm52.evidence-key-public-identity.v1\0" + key
    ).hexdigest()


def configure_hmac_key(
    keychain: Keychain,
    *,
    random_bytes: Callable[[int], bytes] = secrets.token_bytes,
) -> dict[str, Any]:
    """Create the independent evidence key once; an existing valid key is retained."""
    existing = keychain.get(EVIDENCE_HMAC_SERVICE)
    if existing is not None:
        key = _decode(existing)
        return {
            "status": "ALREADY_CONFIGURED",
            "configured": True,
            "key_identity_digest": _identity_digest(key),
        }
    try:
        key = random_bytes(HMAC_KEY_BYTES)
    except Exception:
        raise EvidenceSecurityError("secure evidence-key generation failed") from None
    encoded = _encode(key)
    keychain.set(EVIDENCE_HMAC_SERVICE, encoded)
    if keychain.get(EVIDENCE_HMAC_SERVICE) != encoded:
        raise EvidenceSecurityError("evidence Keychain post-write verification failed")
    return {
        "status": "CONFIGURED",
        "configured": True,
        "key_identity_digest": _identity_digest(key),
    }


def credential_status(keychain: Keychain) -> dict[str, Any]:
    encoded = keychain.get(EVIDENCE_HMAC_SERVICE)
    if encoded is None:
        return {"configured": False, "ready": False}
    try:
        key = _decode(encoded)
    except EvidenceSecurityError:
        return {"configured": False, "ready": False}
    return {
        "configured": True,
        "ready": True,
        "key_identity_digest": _identity_digest(key),
    }


def load_evidence_auth(
    keychain: Keychain,
    *,
    campaign_id: str,
    source_revision: str,
) -> EvidenceAuthConfig:
    encoded = keychain.get(EVIDENCE_HMAC_SERVICE)
    if encoded is None:
        raise EvidenceSecurityError("scientific evidence authentication is not configured")
    try:
        return EvidenceAuthConfig(
            hmac_key=_decode(encoded),
            campaign_id=campaign_id,
            source_revision=source_revision,
        )
    except (EvidenceSecurityError, StateError):
        raise EvidenceSecurityError("scientific evidence authentication is invalid") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configure-hmac-key", "status"))
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    keychain: Keychain | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    provider = keychain or MacOSKeychain()
    try:
        result = (
            configure_hmac_key(provider)
            if args.command == "configure-hmac-key"
            else credential_status(provider)
        )
    except EvidenceSecurityError as exc:
        print(
            json.dumps(
                {"status": "ERROR", "error": type(exc).__name__, "message": str(exc)},
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ready", result.get("configured", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())

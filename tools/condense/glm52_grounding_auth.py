#!/usr/bin/env python3.12
"""Independent Keychain credential for GLM-5.2 filesystem observations."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import sys
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
for import_root in (REPOSITORY_ROOT, HERE):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from glm52_common import Glm52Error
from glm52_evidence_auth import (
    KEYCHAIN_ACCOUNT,
    _get_generic_password_native,
    _set_generic_password_native,
)
from glm52_grounding import GroundingError, ProducerAuthenticator


GROUNDING_HMAC_SERVICE = "com.hawking.glm52.gravity.grounding.producer-hmac-key"
HMAC_KEY_BYTES = 32


class GroundingSecurityError(Glm52Error):
    """The observation-producer credential is missing, malformed, or unsafe."""


class Keychain(Protocol):
    def get(self, service: str) -> str | None: ...
    def set(self, service: str, value: str) -> None: ...


class MacOSKeychain:
    """Native, in-process access to the one grounding Keychain service."""

    def __init__(
        self,
        *,
        native_reader: Callable[[str, str], str | None] = _get_generic_password_native,
        native_writer: Callable[[str, str, str], None] = _set_generic_password_native,
    ) -> None:
        self._reader = native_reader
        self._writer = native_writer

    @staticmethod
    def _service(service: str) -> str:
        if service != GROUNDING_HMAC_SERVICE:
            raise GroundingSecurityError("unrecognized GLM grounding Keychain service")
        return service

    def get(self, service: str) -> str | None:
        service = self._service(service)
        try:
            value = self._reader(service, KEYCHAIN_ACCOUNT)
        except Exception:
            raise GroundingSecurityError("macOS Keychain grounding read failed") from None
        if value is not None and (not value or any(mark in value for mark in ("\n", "\r", "\x00"))):
            raise GroundingSecurityError("macOS Keychain returned invalid grounding data")
        return value

    def set(self, service: str, value: str) -> None:
        service = self._service(service)
        if not isinstance(value, str) or not value \
                or any(mark in value for mark in ("\n", "\r", "\x00")):
            raise GroundingSecurityError("grounding credential must be one non-empty line")
        try:
            self._writer(service, KEYCHAIN_ACCOUNT, value)
        except Exception:
            raise GroundingSecurityError("macOS Keychain grounding write failed") from None


def _encode(key: bytes) -> str:
    if not isinstance(key, bytes) or len(key) != HMAC_KEY_BYTES:
        raise GroundingSecurityError("grounding key must be exactly 32 bytes")
    return base64.urlsafe_b64encode(key).decode("ascii")


def _decode(value: Any) -> bytes:
    if not isinstance(value, str) or not value \
            or any(mark in value for mark in ("\n", "\r", "\x00")):
        raise GroundingSecurityError("grounding credential is invalid")
    try:
        key = base64.b64decode(value, altchars=b"-_", validate=True)
    except (TypeError, ValueError):
        raise GroundingSecurityError("grounding credential is invalid") from None
    if len(key) != HMAC_KEY_BYTES or _encode(key) != value:
        raise GroundingSecurityError("grounding credential is invalid")
    return key


def _public_identity(key: bytes) -> str:
    return hashlib.sha256(
        b"hawking.glm52.grounding-key-public-identity.v1\0" + key
    ).hexdigest()


def configure_hmac_key(
    keychain: Keychain,
    *,
    random_bytes: Callable[[int], bytes] = secrets.token_bytes,
) -> dict[str, Any]:
    existing = keychain.get(GROUNDING_HMAC_SERVICE)
    if existing is not None:
        key = _decode(existing)
        return {
            "status": "ALREADY_CONFIGURED",
            "configured": True,
            "key_identity_digest": _public_identity(key),
        }
    try:
        key = random_bytes(HMAC_KEY_BYTES)
    except Exception:
        raise GroundingSecurityError("secure grounding-key generation failed") from None
    encoded = _encode(key)
    keychain.set(GROUNDING_HMAC_SERVICE, encoded)
    if keychain.get(GROUNDING_HMAC_SERVICE) != encoded:
        raise GroundingSecurityError("grounding Keychain post-write verification failed")
    return {
        "status": "CONFIGURED",
        "configured": True,
        "key_identity_digest": _public_identity(key),
    }


def credential_status(keychain: Keychain) -> dict[str, Any]:
    encoded = keychain.get(GROUNDING_HMAC_SERVICE)
    if encoded is None:
        return {"configured": False, "ready": False}
    try:
        key = _decode(encoded)
    except GroundingSecurityError:
        return {"configured": False, "ready": False}
    return {
        "configured": True,
        "ready": True,
        "key_identity_digest": _public_identity(key),
    }


def load_grounding_auth(keychain: Keychain) -> ProducerAuthenticator:
    encoded = keychain.get(GROUNDING_HMAC_SERVICE)
    if encoded is None:
        raise GroundingSecurityError("filesystem observation authentication is not configured")
    try:
        return ProducerAuthenticator(_decode(encoded))
    except (GroundingSecurityError, GroundingError):
        raise GroundingSecurityError("filesystem observation authentication is invalid") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configure-hmac-key", "status"))
    return parser


def main(argv: Sequence[str] | None = None, *, keychain: Keychain | None = None) -> int:
    args = build_parser().parse_args(argv)
    provider = keychain or MacOSKeychain()
    try:
        result = (
            configure_hmac_key(provider)
            if args.command == "configure-hmac-key"
            else credential_status(provider)
        )
    except GroundingSecurityError as exc:
        print(json.dumps({
            "status": "ERROR", "error": type(exc).__name__, "message": str(exc),
        }, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ready", result.get("configured", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())

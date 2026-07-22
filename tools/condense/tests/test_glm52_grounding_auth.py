#!/usr/bin/env python3.12
from __future__ import annotations

import base64
import pathlib
import sys

import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_grounding_auth as auth  # noqa: E402
import glm52_state as state  # noqa: E402


KEY = bytes(range(32))


class FakeKeychain:
    def __init__(self, values=None, *, discard=False):
        self.values = dict(values or {})
        self.discard = discard

    def get(self, service):
        return self.values.get(service)

    def set(self, service, value):
        if not self.discard:
            self.values[service] = value


def test_configures_idempotently_and_loads_redacted_authenticator() -> None:
    keychain = FakeKeychain()
    first = auth.configure_hmac_key(keychain, random_bytes=lambda _size: KEY)
    second = auth.configure_hmac_key(
        keychain,
        random_bytes=lambda _size: (_ for _ in ()).throw(AssertionError()),
    )
    assert first["status"] == "CONFIGURED"
    assert second["status"] == "ALREADY_CONFIGURED"
    assert first["key_identity_digest"] == second["key_identity_digest"]
    loaded = auth.load_grounding_auth(keychain)
    assert "redacted" in repr(loaded)
    assert KEY not in repr(loaded).encode()
    assert loaded._key_material_identity() == state.TelegramAuthConfig(
        hmac_key=KEY,
        expected_chat_identity_digest="a" * 64,
    )._key_material_identity()


@pytest.mark.parametrize("value", ("", "bad!!", base64.urlsafe_b64encode(b"x").decode()))
def test_malformed_credentials_fail_closed(value: str) -> None:
    keychain = FakeKeychain({auth.GROUNDING_HMAC_SERVICE: value})
    assert auth.credential_status(keychain) == {"configured": False, "ready": False}
    with pytest.raises(auth.GroundingSecurityError, match="invalid"):
        auth.load_grounding_auth(keychain)


def test_postwrite_persistence_is_mandatory() -> None:
    with pytest.raises(auth.GroundingSecurityError, match="post-write"):
        auth.configure_hmac_key(
            FakeKeychain(discard=True), random_bytes=lambda _size: KEY
        )


def test_native_adapter_never_uses_subprocess_or_wrong_service() -> None:
    calls = []
    encoded = base64.urlsafe_b64encode(KEY).decode()

    def reader(service, account):
        calls.append(("read", service, account))
        return encoded

    def writer(service, account, value):
        calls.append(("write", service, account, value))

    provider = auth.MacOSKeychain(native_reader=reader, native_writer=writer)
    assert provider.get(auth.GROUNDING_HMAC_SERVICE) == encoded
    provider.set(auth.GROUNDING_HMAC_SERVICE, encoded)
    assert [item[0] for item in calls] == ["read", "write"]
    with pytest.raises(auth.GroundingSecurityError, match="unrecognized"):
        provider.get("com.example.wrong")

#!/usr/bin/env python3.12
from __future__ import annotations

import base64
import pathlib
import subprocess
import sys

import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_evidence_auth as evidence  # noqa: E402


CAMPAIGN = "glm52-evidence-auth-test"
REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
KEY = bytes(range(32))


class FakeKeychain:
    def __init__(self, values=None, *, discard_writes=False):
        self.values = dict(values or {})
        self.discard_writes = discard_writes
        self.set_calls = []

    def get(self, service):
        return self.values.get(service)

    def set(self, service, value):
        self.set_calls.append((service, value))
        if not self.discard_writes:
            self.values[service] = value


def test_configure_is_idempotent_secret_free_and_loads_bound_auth() -> None:
    keychain = FakeKeychain()
    first = evidence.configure_hmac_key(keychain, random_bytes=lambda size: KEY)
    second = evidence.configure_hmac_key(
        keychain,
        random_bytes=lambda _size: (_ for _ in ()).throw(AssertionError()),
    )
    assert first["status"] == "CONFIGURED"
    assert second["status"] == "ALREADY_CONFIGURED"
    assert first["key_identity_digest"] == second["key_identity_digest"]
    assert len(keychain.set_calls) == 1
    assert KEY not in repr(first).encode()
    assert evidence.credential_status(keychain)["ready"] is True
    auth = evidence.load_evidence_auth(
        keychain,
        campaign_id=CAMPAIGN,
        source_revision=REVISION,
    )
    assert auth.campaign_id == CAMPAIGN
    assert auth.source_revision == REVISION
    assert "redacted" in repr(auth)


@pytest.mark.parametrize("encoded", ("invalid!!!", "", base64.urlsafe_b64encode(b"x").decode()))
def test_malformed_keychain_values_fail_closed(encoded: str) -> None:
    keychain = FakeKeychain({evidence.EVIDENCE_HMAC_SERVICE: encoded})
    assert evidence.credential_status(keychain) == {"configured": False, "ready": False}
    with pytest.raises(evidence.EvidenceSecurityError, match="invalid"):
        evidence.load_evidence_auth(
            keychain,
            campaign_id=CAMPAIGN,
            source_revision=REVISION,
        )


def test_configure_requires_verified_keychain_persistence() -> None:
    keychain = FakeKeychain(discard_writes=True)
    with pytest.raises(evidence.EvidenceSecurityError, match="post-write"):
        evidence.configure_hmac_key(keychain, random_bytes=lambda size: KEY)


def test_macos_keychain_write_keeps_secret_out_of_subprocesses() -> None:
    subprocess_calls = []
    native_calls = []

    def runner(arguments, **kwargs):
        subprocess_calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    def writer(service, account, value):
        native_calls.append((service, account, value))

    provider = evidence.MacOSKeychain(
        runner=runner,
        native_reader=lambda _service, _account: None,
        native_writer=writer,
    )
    encoded = base64.urlsafe_b64encode(KEY).decode("ascii")
    provider.set(evidence.EVIDENCE_HMAC_SERVICE, encoded)
    assert subprocess_calls == []
    assert native_calls == [(
        evidence.EVIDENCE_HMAC_SERVICE,
        evidence.KEYCHAIN_ACCOUNT,
        encoded,
    )]


def test_macos_keychain_read_keeps_secret_out_of_subprocesses() -> None:
    subprocess_calls = []
    native_calls = []
    encoded = base64.urlsafe_b64encode(KEY).decode("ascii")

    def runner(arguments, **kwargs):
        subprocess_calls.append((arguments, kwargs))
        raise AssertionError("credential read must not spawn a process")

    def reader(service, account):
        native_calls.append((service, account))
        return encoded

    provider = evidence.MacOSKeychain(
        runner=runner,
        native_reader=reader,
        native_writer=lambda *_args: None,
    )
    assert provider.get(evidence.EVIDENCE_HMAC_SERVICE) == encoded
    assert subprocess_calls == []
    assert native_calls == [(
        evidence.EVIDENCE_HMAC_SERVICE,
        evidence.KEYCHAIN_ACCOUNT,
    )]


def test_wrong_service_is_rejected_before_keychain_access() -> None:
    provider = evidence.MacOSKeychain(
        runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError())
    )
    with pytest.raises(evidence.EvidenceSecurityError, match="unrecognized"):
        provider.get("com.example.wrong")

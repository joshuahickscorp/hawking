from __future__ import annotations

import copy
import hashlib
import os
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))

import appendix_physical_counter_authority as authority  # noqa: E402


def test_source_sealed_registry_and_operator_custody_are_valid() -> None:
    registry = authority.load_default_registry()
    assert authority.validate_registry(
        registry, verify_files=True, require_default=True,
    ) == []
    assert registry["private_key_embedded"] is False
    custody = authority.signing_key_custody_status(registry=registry)
    assert custody["signing_key_available"] is True
    assert custody["matches_pinned_registry"] is True
    assert custody["private_key_bytes_read"] is False
    assert custody["private_key_outside_repository"] is True
    assert custody["private_key_mode"] == "0o600"


def test_registry_tamper_and_substituted_allowed_signers_fail() -> None:
    registry = authority.load_default_registry()
    tampered = copy.deepcopy(registry)
    tampered["signer_identity"] = "attacker"
    tampered = authority._stamp(tampered, "registry_sha256")
    errors = authority.validate_registry(tampered, verify_files=True, require_default=True)
    assert any("pinned production identity" in error for error in errors)
    assert any("differs from the source-sealed trust root" in error for error in errors)

    substituted = copy.deepcopy(registry)
    substituted["allowed_signers_sha256"] = "f" * 64
    substituted["public_key_blob_sha256"] = "e" * 64
    substituted = authority._stamp(substituted, "registry_sha256")
    errors = authority.validate_registry(
        substituted, verify_files=False, require_default=False,
    )
    assert any("compiled operator trust anchor" in error for error in errors)


def test_live_host_uuid_is_measured_and_normalized_before_hashing() -> None:
    raw_uuid = "A1B2C3D4-E5F6-47A8-9B0C-1234567890AB"

    def runner(argv, **kwargs):  # type: ignore[no-untyped-def]
        assert argv == [str(authority.IOREG), "-rd1", "-c", "IOPlatformExpertDevice"]
        assert kwargs["shell"] is False
        return SimpleNamespace(
            returncode=0,
            stdout=f'    "IOPlatformUUID" = "{raw_uuid}"\n',
            stderr="",
        )

    digest, detail = authority.live_host_hardware_uuid_sha256(runner=runner)
    assert digest == hashlib.sha256(raw_uuid.lower().encode("ascii")).hexdigest()
    assert "measured" in detail


def test_live_host_uuid_probe_fails_closed_on_missing_or_ambiguous_output() -> None:
    def runner(_argv, **_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(returncode=0, stdout="no platform UUID", stderr="")

    digest, detail = authority.live_host_hardware_uuid_sha256(runner=runner)
    assert digest is None
    assert "did not expose" in detail


def test_receipt_builder_uses_supplied_live_host_hash_and_canonical_claims(
    tmp_path: pathlib.Path,
) -> None:
    binary = tmp_path / "normalizer"
    binary.write_bytes(b"normalizer")
    receipt = authority.build_receipt(
        receipt_kind="capability", subject="normalizer", binary=binary,
        command_abi_sha256="a" * 64,
        claims=["z", "a", "a"], release_build_sha256="b" * 64,
        valid_seconds=60, now_unix_ns=10,
        host_hardware_uuid_sha256="c" * 64,
    )
    assert receipt["host_hardware_uuid_sha256"] == "c" * 64
    assert receipt["claims"] == ["a", "z"]
    assert authority.validate_receipt(receipt) == []


def test_signer_refuses_private_key_that_does_not_match_pinned_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "binary"
    binary.write_bytes(b"binary")
    receipt = authority.build_receipt(
        receipt_kind="capability", subject="normalizer", binary=binary,
        command_abi_sha256="a" * 64, claims=["claim"],
        release_build_sha256="b" * 64, valid_seconds=60,
        now_unix_ns=10, host_hardware_uuid_sha256="c" * 64,
    )
    wrong_key = tmp_path / "wrong-key"
    wrong_key.write_bytes(b"not a real private key")
    wrong_key.chmod(0o600)
    monkeypatch.setattr(
        authority, "_derived_public_key",
        lambda _path: "ssh-ed25519 AAAAdefinitely-not-the-pinned-key",
    )
    with pytest.raises(authority.AuthorityError, match="does not match"):
        authority.sign_receipt(
            receipt, private_key=wrong_key,
            detached_signature_output=tmp_path / "receipt.sig",
            envelope_output=tmp_path / "envelope.json",
        )
    assert not (tmp_path / "receipt.sig").exists()
    assert not (tmp_path / "envelope.json").exists()


def test_result_signer_rejects_tampered_draft_before_private_key_access(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = {
        "schema": authority.RESULT_ATTESTATION_SCHEMA,
        "kind": "device",
        "result_attestation_sha256": "f" * 64,
    }
    key_accessed = False

    def forbidden_key_access(_path: pathlib.Path) -> str:
        nonlocal key_accessed
        key_accessed = True
        raise AssertionError("invalid drafts must not reach the operator key")

    monkeypatch.setattr(authority, "_derived_public_key", forbidden_key_access)
    with pytest.raises(authority.AuthorityError, match="hash is invalid"):
        authority.sign_result_attestation(
            draft, private_key=tmp_path / "operator-key",
            detached_signature_output=tmp_path / "result.sig",
            envelope_output=tmp_path / "result.json",
        )
    assert key_accessed is False
    assert not (tmp_path / "result.sig").exists()
    assert not (tmp_path / "result.json").exists()


def test_status_distinguishes_registry_validity_and_signing_key_availability() -> None:
    value = authority.status()
    assert value["registry_valid"] is True
    assert value["signing_key_available"] is True
    assert value["private_key_read"] is False
    assert value["signing_key_custody"]["matches_pinned_registry"] is True
    assert authority.dry_run("sign")["would_read_private_key"] is False


def test_immutable_authority_output_rejects_writable_identical_file(
    tmp_path: pathlib.Path,
) -> None:
    output = tmp_path / "receipt.sig"
    output.write_bytes(b"same")
    output.chmod(0o666)
    with pytest.raises(authority.AuthorityError, match="single-link mode"):
        authority._atomic_bytes(output, b"same")
    assert output.read_bytes() == b"same"
    assert output.stat().st_mode & 0o222


def test_immutable_authority_output_rejects_short_write_and_parent_swap(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    short = tmp_path / "short.sig"
    monkeypatch.setattr(authority.os, "write", lambda *_args, **_kwargs: 0)
    with pytest.raises(OSError, match="short write"):
        authority._atomic_bytes(short, b"signature")
    assert not short.exists()
    assert list(tmp_path.glob(".short.sig.*.tmp")) == []

    monkeypatch.undo()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    original_open = authority._open_dir_nofollow

    def swapped(path: pathlib.Path, *, create: bool) -> int:
        if not create:
            return os.open(
                replacement,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        return original_open(path, create=create)

    monkeypatch.setattr(authority, "_open_dir_nofollow", swapped)
    swapped_output = tmp_path / "swapped.sig"
    with pytest.raises(authority.AuthorityError, match="parent was replaced"):
        authority._atomic_bytes(swapped_output, b"signature")
    assert not swapped_output.exists()

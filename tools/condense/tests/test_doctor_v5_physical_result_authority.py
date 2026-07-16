from __future__ import annotations

import copy
import json
import os
import pathlib
import stat
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))

import appendix_physical_counter_authority as authority_root  # noqa: E402
import doctor_v5_physical_ab_controller as controller  # noqa: E402
import doctor_v5_physical_result_authority as result_authority  # noqa: E402
import physical_counter_attestation  # noqa: E402


VERIFIED = lambda _envelope, _payload: (True, "verified")


def _packet() -> dict:
    return result_authority._stamp({
        "schema": result_authority.PACKET_SCHEMA,
        "plan_sha256": "a" * 64,
        "source_manifest": {"manifest_sha256": "b" * 64},
        "release_boundary": {
            "attestation_sha256": "c" * 64,
            "observed_at_unix_ns": 10,
        },
        "facet_receipts": {
            facet: {
                "receipt_sha256": result_authority.canonical_sha256(
                    f"physical-facet:{facet}"
                ),
            }
            for facet in result_authority.FACETS
        },
        "post120_handoff": {},
        "post120_qualification": {},
        "appendix_physical_packet": {},
        "runtime_defaults_changed": False,
        "activation_requested": False,
        "component_speedups_multiplied": False,
    }, "packet_sha256")


def _signed(tmp_path: pathlib.Path, packet: dict, *, issued: int = 20) -> dict:
    attestation = result_authority.build_result_attestation(
        packet, issued_at_unix_ns=issued, valid_seconds=1,
    )
    signature = tmp_path / "doctor-result.sshsig"
    signature.write_bytes(b"synthetic-test-signature-not-production-evidence")
    registry = authority_root.load_default_registry()
    return result_authority._stamp({
        "schema": result_authority.ENVELOPE_SCHEMA,
        "attestation": attestation,
        "signer_identity": result_authority.SIGNER_IDENTITY,
        "signature_namespace": result_authority.SSHSIG_NAMESPACE,
        "allowed_signers": authority_root.allowed_signers_identity(registry),
        "detached_signature": physical_counter_attestation.file_identity(signature),
    }, "envelope_sha256")


def _sealed(tmp_path: pathlib.Path) -> tuple[dict, dict]:
    packet = _packet()
    signed = _signed(tmp_path, packet)
    sealed = result_authority.build_sealed_evidence(
        packet=packet, signed_result_attestation=signed,
        sealed_at_unix_ns=30, verify_files=False,
        signature_verifier=VERIFIED,
    )
    return packet, sealed


def _restamp_envelope(sealed: dict) -> dict:
    sealed["signed_result_attestation"] = result_authority._stamp(
        sealed["signed_result_attestation"], "envelope_sha256",
    )
    return result_authority._stamp(sealed, "sealed_evidence_sha256")


def test_attestation_binds_exact_packet_boundary_source_plan_and_ten_facets() -> None:
    packet = _packet()
    draft = result_authority.build_result_attestation(
        packet, issued_at_unix_ns=20, valid_seconds=1,
    )
    assert draft["packet_schema"] == result_authority.PACKET_SCHEMA
    assert draft["packet_sha256"] == packet["packet_sha256"]
    assert draft["packet_canonical_sha256"] == result_authority.canonical_sha256(packet)
    assert draft["plan_sha256"] == packet["plan_sha256"]
    assert draft["source_manifest_sha256"] == packet["source_manifest"]["manifest_sha256"]
    assert draft["release_boundary_attestation_sha256"] \
        == packet["release_boundary"]["attestation_sha256"]
    assert tuple(draft["facet_receipt_sha256"]) == result_authority.FACETS
    assert len(set(draft["facet_receipt_sha256"].values())) == 10
    assert result_authority.validate_result_attestation(
        draft, packet=packet, now_unix_ns=30,
    ) == []


def test_raw_self_hashed_packet_is_rejected_and_controller_scores_zero() -> None:
    packet = _packet()
    errors = result_authority.validate_sealed_evidence(
        packet, verify_files=False, now_unix_ns=30,
        signature_verifier=VERIFIED,
    )
    assert any("raw/self-hashed" in error for error in errors)
    card = controller.build_scorecard(packet, verify_files=False)
    assert card["physical_rating"] == "0/10"
    assert all(not row["green"] for row in card["facets"])
    assert all(
        any("operator-signed sealed" in error for error in row["errors"])
        for row in card["facets"]
    )


def test_valid_seal_unwraps_only_after_signature_verification(
    tmp_path: pathlib.Path,
) -> None:
    packet, sealed = _sealed(tmp_path)
    core, errors = result_authority.validate_and_unwrap(
        sealed, verify_files=False, now_unix_ns=40,
        signature_verifier=VERIFIED,
    )
    assert errors == []
    assert core == packet and core is not packet
    forged_core, forged_errors = result_authority.validate_and_unwrap(
        sealed, verify_files=False, now_unix_ns=40,
        signature_verifier=lambda _envelope, _payload: (False, "forged"),
    )
    assert forged_core is None
    assert any("SSHSIG verification failed" in error for error in forged_errors)


def test_wrong_signer_namespace_trust_root_and_detached_signature_fail_closed(
    tmp_path: pathlib.Path,
) -> None:
    _packet_value, sealed = _sealed(tmp_path)
    attacks: list[tuple[str, dict]] = []

    wrong_signer = copy.deepcopy(sealed)
    wrong_signer["signed_result_attestation"]["signer_identity"] = "attacker"
    attacks.append(("signer", _restamp_envelope(wrong_signer)))

    wrong_namespace = copy.deepcopy(sealed)
    wrong_namespace["signed_result_attestation"]["signature_namespace"] = "wrong-namespace"
    attacks.append(("namespace", _restamp_envelope(wrong_namespace)))

    substituted = copy.deepcopy(sealed)
    substituted["signed_result_attestation"]["allowed_signers"] = copy.deepcopy(
        substituted["signed_result_attestation"]["allowed_signers"]
    )
    substituted["signed_result_attestation"]["allowed_signers"]["sha256"] = "f" * 64
    attacks.append(("trust root", _restamp_envelope(substituted)))

    for expected, attacked in attacks:
        errors = result_authority.validate_sealed_evidence(
            attacked, verify_files=False, now_unix_ns=40,
            signature_verifier=VERIFIED,
        )
        assert any(expected in error for error in errors), errors

    detached = pathlib.Path(
        sealed["signed_result_attestation"]["detached_signature"]["path"]
    )
    detached.write_bytes(b"tampered-after-envelope")
    errors = result_authority.validate_sealed_evidence(
        sealed, verify_files=True, now_unix_ns=40,
        signature_verifier=VERIFIED,
    )
    assert any("detached signature changed" in error for error in errors)


def test_expired_envelope_and_core_envelope_tampering_are_rejected(
    tmp_path: pathlib.Path,
) -> None:
    _packet_value, sealed = _sealed(tmp_path)
    errors = result_authority.validate_sealed_evidence(
        sealed, verify_files=False, now_unix_ns=1_000_000_021,
        signature_verifier=VERIFIED,
    )
    assert any("expired" in error for error in errors)

    core_tamper = copy.deepcopy(sealed)
    core_tamper["core_packet"]["post120_handoff"] = {"tampered": True}
    core_tamper = result_authority._stamp(core_tamper, "sealed_evidence_sha256")
    errors = result_authority.validate_sealed_evidence(
        core_tamper, verify_files=False, now_unix_ns=40,
        signature_verifier=VERIFIED,
    )
    assert any(
        "packet self-hash" in error or "packet_canonical_sha256" in error
        for error in errors
    )

    envelope_tamper = copy.deepcopy(sealed)
    envelope_tamper["signed_result_attestation"]["signer_identity"] = "attacker"
    envelope_tamper = result_authority._stamp(
        envelope_tamper, "sealed_evidence_sha256",
    )
    errors = result_authority.validate_sealed_evidence(
        envelope_tamper, verify_files=False, now_unix_ns=40,
        signature_verifier=VERIFIED,
    )
    assert any("result_envelope.envelope_sha256 mismatch" in error for error in errors)


def test_malformed_nonfinite_unhashable_and_duplicate_json_fail_closed(
    tmp_path: pathlib.Path,
) -> None:
    _packet_value, sealed = _sealed(tmp_path)
    malformed = copy.deepcopy(sealed)
    malformed["signed_result_attestation"]["attestation"][
        "facet_receipt_sha256"
    ][result_authority.FACETS[0]] = {"not": "a hash"}
    malformed["signed_result_attestation"] = result_authority._stamp(
        malformed["signed_result_attestation"], "envelope_sha256",
    )
    malformed = result_authority._stamp(malformed, "sealed_evidence_sha256")
    errors = result_authority.validate_sealed_evidence(
        malformed, verify_files=False, now_unix_ns=40,
        signature_verifier=VERIFIED,
    )
    assert any("ten distinct exact facet hashes" in error for error in errors)

    nonfinite = copy.deepcopy(sealed)
    nonfinite["core_packet"]["post120_handoff"] = {"value": float("nan")}
    errors = result_authority.validate_sealed_evidence(
        nonfinite, verify_files=False, now_unix_ns=40,
        signature_verifier=VERIFIED,
    )
    assert any("not canonical finite JSON" in error for error in errors)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    with pytest.raises(result_authority.ResultAuthorityError, match="duplicate JSON"):
        result_authority._safe_json(duplicate)


def test_exact_facet_substitution_breaks_signed_aggregate_binding(
    tmp_path: pathlib.Path,
) -> None:
    _packet_value, sealed = _sealed(tmp_path)
    attacked = copy.deepcopy(sealed)
    facet = result_authority.FACETS[4]
    attacked["core_packet"]["facet_receipts"][facet] = {
        "receipt_sha256": "f" * 64,
    }
    attacked["core_packet"] = result_authority._stamp(
        attacked["core_packet"], "packet_sha256",
    )
    attacked = result_authority._stamp(attacked, "sealed_evidence_sha256")
    errors = result_authority.validate_sealed_evidence(
        attacked, verify_files=False, now_unix_ns=40,
        signature_verifier=VERIFIED,
    )
    assert any(
        "facet_receipt_sha256" in error or "packet_canonical_sha256" in error
        for error in errors
    )


def test_signer_revalidates_draft_before_operator_key_access(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _packet()
    draft = result_authority.build_result_attestation(
        packet, issued_at_unix_ns=20, valid_seconds=1,
    )
    draft["facet_receipt_sha256"][result_authority.FACETS[0]] = "f" * 64
    draft = result_authority._stamp(draft, "result_attestation_sha256")
    key_accessed = False

    def forbidden(_path: pathlib.Path) -> str:
        nonlocal key_accessed
        key_accessed = True
        raise AssertionError("invalid draft reached operator key")

    monkeypatch.setattr(authority_root, "_derived_public_key", forbidden)
    with pytest.raises(result_authority.ResultAuthorityError, match="refused invalid"):
        result_authority.sign_result_attestation(
            draft, packet=packet, private_key=tmp_path / "operator-key",
            detached_signature_output=tmp_path / "result.sig",
            envelope_output=tmp_path / "result.json", now_unix_ns=30,
        )
    assert key_accessed is False
    assert not (tmp_path / "result.sig").exists()
    assert not (tmp_path / "result.json").exists()


def test_immutable_outputs_reject_writable_existing_short_write_and_parent_swap(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    writable = tmp_path / "writable.json"
    writable.write_bytes(b"same")
    writable.chmod(0o600)
    with pytest.raises(result_authority.ResultAuthorityError, match="single-link mode"):
        result_authority._atomic_bytes(writable, b"same")
    assert stat.S_IMODE(writable.stat().st_mode) == 0o600

    original_write = authority_root.os.write
    monkeypatch.setattr(authority_root.os, "write", lambda *_args, **_kwargs: 0)
    short = tmp_path / "short.json"
    with pytest.raises(OSError, match="short write"):
        result_authority._atomic_bytes(short, b"sealed")
    assert not short.exists()
    assert not list(tmp_path.glob(".short.json.*.tmp"))

    monkeypatch.setattr(authority_root.os, "write", original_write)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    original_open = authority_root._open_dir_nofollow

    def swapped(path: pathlib.Path, *, create: bool) -> int:
        if not create:
            return os.open(
                replacement,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        return original_open(path, create=create)

    monkeypatch.setattr(authority_root, "_open_dir_nofollow", swapped)
    swapped_output = tmp_path / "swapped.json"
    with pytest.raises(result_authority.ResultAuthorityError, match="parent was replaced"):
        result_authority._atomic_bytes(swapped_output, b"sealed")
    assert not swapped_output.exists()


def test_full_draft_sign_seal_verify_cli_uses_exact_result_namespace(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _packet()
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    draft_path = tmp_path / "draft.json"
    assert result_authority.main([
        "--draft-result", str(packet_path), "--output", str(draft_path),
        "--valid-seconds", "60",
    ]) == 0
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    assert draft["packet_canonical_sha256"] == result_authority.canonical_sha256(packet)
    assert stat.S_IMODE(draft_path.stat().st_mode) == 0o444

    key = tmp_path / "operator-key"
    key.write_bytes(b"operator-key-remains-outside-repository")
    key.chmod(0o600)
    allowed_parts = authority_root.DEFAULT_ALLOWED_SIGNERS.read_text(
        encoding="utf-8"
    ).split()
    pinned_public = f"{allowed_parts[1]} {allowed_parts[2]}"
    monkeypatch.setattr(authority_root, "_derived_public_key", lambda _path: pinned_public)

    def fake_signer(argv, **kwargs):  # type: ignore[no-untyped-def]
        assert argv[:3] == [str(result_authority.SSH_KEYGEN), "-Y", "sign"]
        assert argv[argv.index("-n") + 1] == result_authority.SSHSIG_NAMESPACE
        assert kwargs["shell"] is False
        message = pathlib.Path(argv[-1])
        assert message.read_bytes() == result_authority.canonical_bytes(draft)
        message.with_suffix(message.suffix + ".sig").write_bytes(b"fake-test-sshsig")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(result_authority.subprocess, "run", fake_signer)
    monkeypatch.setattr(
        authority_root, "verify_sshsig_envelope",
        lambda *_args, **_kwargs: (True, "verified"),
    )
    signature_path = tmp_path / "result.sshsig"
    envelope_path = tmp_path / "envelope.json"
    assert result_authority.main([
        "--sign-result", str(draft_path), "--packet", str(packet_path),
        "--private-key", str(key), "--signature-output", str(signature_path),
        "--envelope-output", str(envelope_path),
    ]) == 0
    assert stat.S_IMODE(signature_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(envelope_path.stat().st_mode) == 0o444

    sealed_path = tmp_path / "sealed.json"
    assert result_authority.main([
        "--seal-result", str(envelope_path), "--packet", str(packet_path),
        "--output", str(sealed_path),
    ]) == 0
    assert stat.S_IMODE(sealed_path.stat().st_mode) == 0o444
    assert result_authority.main(["--verify", str(sealed_path)]) == 0


def test_controller_passes_only_verified_unwrapped_core_to_deep_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _packet()
    observed: list[dict] = []

    def unwrap(_value, **_kwargs):  # type: ignore[no-untyped-def]
        return copy.deepcopy(core), []

    def deep(value, **_kwargs):  # type: ignore[no-untyped-def]
        observed.append(value)
        return [], {facet: [] for facet in controller.FACETS}

    monkeypatch.setattr(controller.result_authority, "validate_and_unwrap", unwrap)
    monkeypatch.setattr(controller, "_validate_core_packet", deep)
    global_errors, facet_errors = controller.validate_packet(
        {"sealed": True}, plan=controller.build_plan(), verify_files=False,
    )
    assert global_errors == []
    assert all(not errors for errors in facet_errors.values())
    assert observed == [core]

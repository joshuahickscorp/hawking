#!/usr/bin/env python3.12
"""Operator seal for the aggregate Doctor V5 physical A/B result.

This module is deliberately not a physical executor.  It exposes only a
cheap, default-off draft -> operator SSHSIG sign -> seal -> verify workflow.
The verifier owns the signer, trust root, and result-only namespace in source;
an evidence envelope cannot select any of them.

The signed attestation binds the complete canonical core packet, its exact
schema and self-hash, the plan/source/release-boundary hashes, and one distinct
receipt hash for each of the ten reviewed physical facets.  A raw packet is
therefore never equivalent to operator-sealed evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from typing import Any

import appendix_physical_counter_authority as authority_root
import physical_counter_attestation


ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKET_SCHEMA = "hawking.doctor_v5_physical_ab_evidence.v1"
ATTESTATION_SCHEMA = "hawking.doctor_v5_physical_ab_result_attestation.v1"
ENVELOPE_SCHEMA = "hawking.doctor_v5_physical_ab_signed_result_envelope.v1"
SEALED_EVIDENCE_SCHEMA = "hawking.doctor_v5_physical_ab_operator_sealed_evidence.v1"
SSHSIG_NAMESPACE = "hawking-doctor-v5-physical-result-v1"
SIGNER_IDENTITY = authority_root.SIGNER_IDENTITY
DEFAULT_PRIVATE_KEY = authority_root.DEFAULT_OPERATOR_PRIVATE_KEY
SSH_KEYGEN = authority_root.SSH_KEYGEN
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_JSON_BYTES = 512 * 1024 * 1024
DEFAULT_VALIDITY_SECONDS = 24 * 60 * 60
MAX_VALIDITY_SECONDS = 7 * 24 * 60 * 60

FACETS = (
    "release_authority",
    "thread_profiles",
    "block_parallel",
    "ordered_overlap",
    "bounded_reuse",
    "ram_swap_recovery",
    "native_io_pgo",
    "disk_lifecycle",
    "full_stack_parity_ab",
    "post120_appendix_bindings",
)

CORE_PACKET_FIELDS = {
    "schema", "plan_sha256", "source_manifest", "release_boundary",
    "facet_receipts", "post120_handoff", "post120_qualification",
    "appendix_physical_packet", "runtime_defaults_changed",
    "activation_requested", "component_speedups_multiplied", "packet_sha256",
}

SignatureVerifier = Callable[[dict[str, Any], bytes], tuple[bool, str]]


class ResultAuthorityError(ValueError):
    """The aggregate result or its immutable operator output is untrusted."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _stamp(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    stamped = copy.deepcopy(dict(value))
    stamped.pop(field, None)
    stamped[field] = canonical_sha256(stamped)
    return stamped


def _hex(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _hash_errors(value: Any, field: str, *, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} must be an object"]
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop(field, None)
    if not _hex(claimed):
        return [f"{label}.{field} is invalid"]
    try:
        observed = canonical_sha256(unstamped)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return [f"{label} is not canonical finite JSON"]
    return [] if claimed == observed else [f"{label}.{field} mismatch"]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _safe_json(path: pathlib.Path) -> Any:
    try:
        raw = authority_root._safe_file_bytes(path, maximum=MAX_JSON_BYTES)
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except FileNotFoundError:
        raise
    except (
        authority_root.AuthorityError, OSError, UnicodeError,
        json.JSONDecodeError, ValueError,
    ) as exc:
        raise ResultAuthorityError(f"unsafe or invalid JSON {path}: {exc}") from exc


def _atomic_bytes(path: pathlib.Path, raw: bytes) -> None:
    """Use the source-reviewed retained-dirfd immutable installer."""
    try:
        authority_root._atomic_bytes(path, raw, mode=0o444)
    except authority_root.AuthorityError as exc:
        raise ResultAuthorityError(str(exc)) from exc


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _atomic_bytes(path, raw)


def _core_bindings(packet: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Extract only unambiguous, exact aggregate bindings from one core."""
    errors: list[str] = []
    if not isinstance(packet, dict) or set(packet) != CORE_PACKET_FIELDS:
        return None, ["core physical packet fields are incomplete or unexpected"]
    if packet.get("schema") != PACKET_SCHEMA:
        errors.append("core physical packet schema is invalid")
    unstamped = copy.deepcopy(packet)
    claimed_packet_sha256 = unstamped.pop("packet_sha256", None)
    try:
        observed_packet_sha256 = canonical_sha256(unstamped)
    except (TypeError, ValueError, OverflowError, RecursionError):
        observed_packet_sha256 = None
    if not _hex(claimed_packet_sha256) \
            or claimed_packet_sha256 != observed_packet_sha256:
        errors.append("core physical packet self-hash is invalid")
    plan_sha256 = packet.get("plan_sha256")
    source = packet.get("source_manifest")
    boundary = packet.get("release_boundary")
    source_sha256 = source.get("manifest_sha256") if isinstance(source, dict) else None
    boundary_sha256 = boundary.get("attestation_sha256") \
        if isinstance(boundary, dict) else None
    for label, value in (
        ("plan_sha256", plan_sha256),
        ("source_manifest_sha256", source_sha256),
        ("release_boundary_attestation_sha256", boundary_sha256),
    ):
        if not _hex(value):
            errors.append(f"core physical packet {label} is invalid")
    receipts = packet.get("facet_receipts")
    facet_hashes: dict[str, str] = {}
    if not isinstance(receipts, dict) or set(receipts) != set(FACETS) \
            or len(receipts) != len(FACETS):
        errors.append("core physical packet does not contain the exact ten-facet map")
    else:
        for facet in FACETS:
            receipt = receipts.get(facet)
            digest = receipt.get("receipt_sha256") if isinstance(receipt, dict) else None
            if not _hex(digest):
                errors.append(f"core physical packet facet {facet} receipt hash is invalid")
            else:
                facet_hashes[facet] = digest
        if len(set(facet_hashes.values())) != len(FACETS):
            errors.append("core physical packet reuses a facet receipt hash")
    if packet.get("runtime_defaults_changed") is not False \
            or packet.get("activation_requested") is not False \
            or packet.get("component_speedups_multiplied") is not False:
        errors.append("core physical packet weakened default-off or speedup-isolation policy")
    if errors:
        return None, errors
    return {
        "packet_schema": PACKET_SCHEMA,
        "packet_sha256": claimed_packet_sha256,
        "packet_canonical_sha256": canonical_sha256(packet),
        "plan_sha256": plan_sha256,
        "source_manifest_sha256": source_sha256,
        "release_boundary_attestation_sha256": boundary_sha256,
        "facet_receipt_sha256": facet_hashes,
    }, []


def build_result_attestation(
    packet: Mapping[str, Any], *, issued_at_unix_ns: int | None = None,
    valid_seconds: int = DEFAULT_VALIDITY_SECONDS,
) -> dict[str, Any]:
    bindings, errors = _core_bindings(packet)
    if errors or bindings is None:
        raise ResultAuthorityError("cannot draft aggregate result: " + "; ".join(errors))
    if isinstance(valid_seconds, bool) or not isinstance(valid_seconds, int) \
            or not 1 <= valid_seconds <= MAX_VALIDITY_SECONDS:
        raise ResultAuthorityError(
            f"result validity must be 1..{MAX_VALIDITY_SECONDS} seconds"
        )
    issued = time.time_ns() if issued_at_unix_ns is None else issued_at_unix_ns
    if isinstance(issued, bool) or not isinstance(issued, int) or issued <= 0:
        raise ResultAuthorityError("result issue time is invalid")
    boundary = packet["release_boundary"]
    observed = boundary.get("observed_at_unix_ns") if isinstance(boundary, Mapping) else None
    if not isinstance(observed, int) or isinstance(observed, bool) or issued < observed:
        raise ResultAuthorityError("result issue time predates its physical release boundary")
    return _stamp({
        "schema": ATTESTATION_SCHEMA,
        **bindings,
        "issued_at_unix_ns": issued,
        "expires_at_unix_ns": issued + valid_seconds * 1_000_000_000,
        "runtime_defaults_changed": False,
        "activation_requested": False,
    }, "result_attestation_sha256")


def validate_result_attestation(
    value: Any, *, packet: Mapping[str, Any], now_unix_ns: int | None = None,
) -> list[str]:
    expected_fields = {
        "schema", "packet_schema", "packet_sha256", "packet_canonical_sha256",
        "plan_sha256", "source_manifest_sha256",
        "release_boundary_attestation_sha256", "facet_receipt_sha256",
        "issued_at_unix_ns", "expires_at_unix_ns", "runtime_defaults_changed",
        "activation_requested", "result_attestation_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        return ["aggregate result attestation fields are incomplete or unexpected"]
    errors = _hash_errors(value, "result_attestation_sha256", label="result_attestation")
    bindings, binding_errors = _core_bindings(packet)
    errors.extend(binding_errors)
    if value.get("schema") != ATTESTATION_SCHEMA:
        errors.append("aggregate result attestation schema is invalid")
    if isinstance(bindings, dict):
        for field, expected in bindings.items():
            if value.get(field) != expected:
                errors.append(f"aggregate result attestation {field} differs from the core packet")
    issued = value.get("issued_at_unix_ns")
    expires = value.get("expires_at_unix_ns")
    now = time.time_ns() if now_unix_ns is None else now_unix_ns
    valid_now = not isinstance(now, bool) and isinstance(now, int) and now > 0
    if not valid_now:
        errors.append("aggregate result verification time is invalid")
    if isinstance(issued, bool) or not isinstance(issued, int) or issued <= 0 \
            or isinstance(expires, bool) or not isinstance(expires, int) \
            or expires <= issued \
            or expires - issued > MAX_VALIDITY_SECONDS * 1_000_000_000:
        errors.append("aggregate result attestation validity interval is invalid")
    elif valid_now and now < issued:
        errors.append("aggregate result attestation is not yet valid")
    elif valid_now and now > expires:
        errors.append("aggregate result attestation is expired")
    boundary = packet.get("release_boundary")
    observed = boundary.get("observed_at_unix_ns") if isinstance(boundary, Mapping) else None
    if isinstance(issued, int) and not isinstance(issued, bool) \
            and isinstance(observed, int) and not isinstance(observed, bool) \
            and issued < observed:
        errors.append("aggregate result attestation predates the release boundary")
    facets = value.get("facet_receipt_sha256")
    facet_values_valid = isinstance(facets, dict) and all(
        _hex(facets.get(facet)) for facet in FACETS
    )
    if not isinstance(facets, dict) or set(facets) != set(FACETS) \
            or len(facets) != len(FACETS) \
            or not facet_values_valid \
            or len(set(facets.values())) != len(FACETS):
        errors.append("aggregate result attestation lacks ten distinct exact facet hashes")
    if value.get("runtime_defaults_changed") is not False \
            or value.get("activation_requested") is not False:
        errors.append("aggregate result attestation weakens default-off policy")
    return list(dict.fromkeys(errors))


def _sshsig_verify(envelope: dict[str, Any], payload: bytes) -> tuple[bool, str]:
    return authority_root.verify_sshsig_envelope(
        envelope, payload, namespace=SSHSIG_NAMESPACE,
    )


def _identity_errors(value: Any, *, verify_file: bool) -> list[str]:
    expected = {"path", "sha256", "size_bytes"}
    if not isinstance(value, dict) or set(value) != expected:
        return ["aggregate result detached signature identity is malformed"]
    if not isinstance(value.get("path"), str) or not pathlib.Path(value["path"]).is_absolute() \
            or not _hex(value.get("sha256")) \
            or isinstance(value.get("size_bytes"), bool) \
            or not isinstance(value.get("size_bytes"), int) or value["size_bytes"] <= 0:
        return ["aggregate result detached signature identity is invalid"]
    if verify_file:
        try:
            actual = physical_counter_attestation.file_identity(pathlib.Path(value["path"]))
        except (OSError, ValueError) as exc:
            return [f"aggregate result detached signature cannot be verified: {exc}"]
        if actual != value:
            return ["aggregate result detached signature changed after signing"]
    return []


def validate_signed_result(
    envelope: Any, *, packet: Mapping[str, Any], verify_files: bool = True,
    now_unix_ns: int | None = None,
    signature_verifier: SignatureVerifier = _sshsig_verify,
) -> list[str]:
    expected_fields = {
        "schema", "attestation", "signer_identity", "signature_namespace",
        "allowed_signers", "detached_signature", "envelope_sha256",
    }
    if not isinstance(envelope, dict) or set(envelope) != expected_fields:
        return ["signed aggregate result envelope is malformed"]
    errors = _hash_errors(envelope, "envelope_sha256", label="result_envelope")
    try:
        registry = authority_root.load_default_registry()
        errors.extend(authority_root.validate_registry(
            registry, verify_files=verify_files, require_default=True,
        ))
    except (OSError, authority_root.AuthorityError) as exc:
        errors.append(f"source-pinned aggregate signer registry is invalid: {exc}")
        registry = None
    if envelope.get("schema") != ENVELOPE_SCHEMA:
        errors.append("signed aggregate result envelope schema is invalid")
    if envelope.get("signer_identity") != SIGNER_IDENTITY:
        errors.append("signed aggregate result envelope signer is not source-pinned")
    if envelope.get("signature_namespace") != SSHSIG_NAMESPACE:
        errors.append("signed aggregate result envelope namespace is invalid")
    try:
        pinned = authority_root.allowed_signers_identity(registry) \
            if isinstance(registry, dict) else None
    except (OSError, authority_root.AuthorityError) as exc:
        errors.append(f"source-pinned allowed-signers identity is unavailable: {exc}")
        pinned = None
    if envelope.get("allowed_signers") != pinned:
        errors.append("signed aggregate result attempted to substitute its trust root")
    errors.extend(_identity_errors(
        envelope.get("detached_signature"), verify_file=verify_files,
    ))
    attestation = envelope.get("attestation")
    errors.extend(validate_result_attestation(
        attestation, packet=packet, now_unix_ns=now_unix_ns,
    ))
    if not errors:
        try:
            ok, detail = signature_verifier(envelope, canonical_bytes(attestation))
        except Exception as exc:  # verifier failure is evidence failure, never acceptance
            ok, detail = False, str(exc)
        if not ok:
            errors.append(
                "aggregate result SSHSIG verification failed"
                + (f": {detail}" if detail else "")
            )
    return list(dict.fromkeys(errors))


def build_sealed_evidence(
    *, packet: Mapping[str, Any], signed_result_attestation: Mapping[str, Any],
    sealed_at_unix_ns: int | None = None, verify_files: bool = True,
    signature_verifier: SignatureVerifier = _sshsig_verify,
) -> dict[str, Any]:
    sealed_at = time.time_ns() if sealed_at_unix_ns is None else sealed_at_unix_ns
    errors = validate_signed_result(
        signed_result_attestation, packet=packet, verify_files=verify_files,
        now_unix_ns=sealed_at, signature_verifier=signature_verifier,
    )
    if errors:
        raise ResultAuthorityError("cannot seal invalid aggregate result: " + "; ".join(errors))
    attestation = signed_result_attestation["attestation"]
    if isinstance(sealed_at, bool) or not isinstance(sealed_at, int) \
            or not attestation["issued_at_unix_ns"] <= sealed_at \
            <= attestation["expires_at_unix_ns"]:
        raise ResultAuthorityError("aggregate result seal time is outside signed validity")
    return _stamp({
        "schema": SEALED_EVIDENCE_SCHEMA,
        "core_packet": copy.deepcopy(dict(packet)),
        "signed_result_attestation": copy.deepcopy(dict(signed_result_attestation)),
        "sealed_at_unix_ns": sealed_at,
        "runtime_defaults_changed": False,
        "activation_requested": False,
    }, "sealed_evidence_sha256")


def validate_sealed_evidence(
    value: Any, *, verify_files: bool = True, now_unix_ns: int | None = None,
    signature_verifier: SignatureVerifier = _sshsig_verify,
) -> list[str]:
    expected_fields = {
        "schema", "core_packet", "signed_result_attestation", "sealed_at_unix_ns",
        "runtime_defaults_changed", "activation_requested", "sealed_evidence_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        return [
            "operator-sealed Doctor physical evidence is required; "
            "raw/self-hashed packets are rejected"
        ]
    errors = _hash_errors(value, "sealed_evidence_sha256", label="sealed_evidence")
    if value.get("schema") != SEALED_EVIDENCE_SCHEMA:
        errors.append("operator-sealed Doctor physical evidence schema is invalid")
    if value.get("runtime_defaults_changed") is not False \
            or value.get("activation_requested") is not False:
        errors.append("operator-sealed Doctor physical evidence weakens default-off policy")
    packet = value.get("core_packet")
    signed = value.get("signed_result_attestation")
    if not isinstance(packet, dict) or not isinstance(signed, dict):
        errors.append("operator-sealed Doctor physical evidence has no exact core/signature")
        return list(dict.fromkeys(errors))
    now = time.time_ns() if now_unix_ns is None else now_unix_ns
    errors.extend(validate_signed_result(
        signed, packet=packet, verify_files=verify_files, now_unix_ns=now,
        signature_verifier=signature_verifier,
    ))
    sealed_at = value.get("sealed_at_unix_ns")
    attestation = signed.get("attestation")
    if isinstance(attestation, dict):
        issued = attestation.get("issued_at_unix_ns")
        expires = attestation.get("expires_at_unix_ns")
        if isinstance(sealed_at, bool) or not isinstance(sealed_at, int) \
                or not isinstance(issued, int) or isinstance(issued, bool) \
                or not isinstance(expires, int) or isinstance(expires, bool) \
                or not issued <= sealed_at <= expires:
            errors.append("operator seal time is outside the signed validity interval")
    else:
        errors.append("operator-sealed Doctor physical attestation is malformed")
    return list(dict.fromkeys(errors))


def validate_and_unwrap(
    value: Any, *, verify_files: bool = True, now_unix_ns: int | None = None,
    signature_verifier: SignatureVerifier = _sshsig_verify,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = validate_sealed_evidence(
        value, verify_files=verify_files, now_unix_ns=now_unix_ns,
        signature_verifier=signature_verifier,
    )
    if errors:
        return None, errors
    return copy.deepcopy(value["core_packet"]), []


def sign_result_attestation(
    attestation: Mapping[str, Any], *, packet: Mapping[str, Any],
    private_key: pathlib.Path, detached_signature_output: pathlib.Path,
    envelope_output: pathlib.Path, now_unix_ns: int | None = None,
) -> dict[str, Any]:
    """Sign only a dynamically revalidated Doctor aggregate draft."""
    errors = validate_result_attestation(
        attestation, packet=packet, now_unix_ns=now_unix_ns,
    )
    if errors:
        raise ResultAuthorityError(
            "operator signer refused invalid aggregate draft: " + "; ".join(errors)
        )
    try:
        registry = authority_root.load_default_registry()
        registry_errors = authority_root.validate_registry(
            registry, verify_files=True, require_default=True,
        )
    except (OSError, authority_root.AuthorityError) as exc:
        raise ResultAuthorityError(f"source-pinned signer registry is invalid: {exc}") from exc
    if registry_errors:
        raise ResultAuthorityError(
            "source-pinned signer registry is invalid: " + "; ".join(registry_errors)
        )
    try:
        private_stat = private_key.stat(follow_symlinks=False)
        private_resolved = private_key.resolve(strict=True)
    except OSError as exc:
        raise ResultAuthorityError(f"operator private key is unavailable: {exc}") from exc
    if private_key.is_symlink() or not stat.S_ISREG(private_stat.st_mode) \
            or private_stat.st_mode & 0o077:
        raise ResultAuthorityError(
            "operator private key must be a non-symlink mode-0600 regular file"
        )
    try:
        private_resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ResultAuthorityError("operator private key must remain outside the repository")
    try:
        public_key = authority_root._derived_public_key(private_key)
    except authority_root.AuthorityError as exc:
        raise ResultAuthorityError(str(exc)) from exc
    if hashlib.sha256(public_key.encode("ascii")).hexdigest() \
            != authority_root.PINNED_PUBLIC_KEY_BLOB_SHA256:
        raise ResultAuthorityError("operator key does not match the source-pinned signer")
    payload = canonical_bytes(attestation)
    with tempfile.TemporaryDirectory(prefix="hawking-doctor-result-sign-") as directory:
        message_path = pathlib.Path(directory) / "result.canonical.json"
        message_path.write_bytes(payload)
        process = subprocess.run(
            [
                str(SSH_KEYGEN), "-Y", "sign", "-f", str(private_key),
                "-n", SSHSIG_NAMESPACE, str(message_path),
            ],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30, check=False, shell=False,
        )
        generated = message_path.with_suffix(message_path.suffix + ".sig")
        if process.returncode != 0 or not generated.is_file():
            detail = (process.stderr or process.stdout or "")[-500:]
            raise ResultAuthorityError(
                f"aggregate result SSHSIG signing failed ({process.returncode}): {detail}"
            )
        signature_raw = generated.read_bytes()
    _atomic_bytes(detached_signature_output, signature_raw)
    envelope = _stamp({
        "schema": ENVELOPE_SCHEMA,
        "attestation": copy.deepcopy(dict(attestation)),
        "signer_identity": SIGNER_IDENTITY,
        "signature_namespace": SSHSIG_NAMESPACE,
        "allowed_signers": authority_root.allowed_signers_identity(registry),
        "detached_signature": physical_counter_attestation.file_identity(
            detached_signature_output,
        ),
    }, "envelope_sha256")
    _atomic_json(envelope_output, envelope)
    return envelope


def draft_result_file(
    packet_path: pathlib.Path, output: pathlib.Path, *, valid_seconds: int,
) -> dict[str, Any]:
    packet = _safe_json(packet_path)
    attestation = build_result_attestation(packet, valid_seconds=valid_seconds)
    _atomic_json(output, attestation)
    return attestation


def sign_result_files(
    *, packet_path: pathlib.Path, draft_path: pathlib.Path,
    private_key: pathlib.Path, detached_signature_output: pathlib.Path,
    envelope_output: pathlib.Path,
) -> dict[str, Any]:
    packet = _safe_json(packet_path)
    draft = _safe_json(draft_path)
    return sign_result_attestation(
        draft, packet=packet, private_key=private_key,
        detached_signature_output=detached_signature_output,
        envelope_output=envelope_output,
    )


def seal_result_files(
    *, packet_path: pathlib.Path, envelope_path: pathlib.Path,
    output: pathlib.Path,
) -> dict[str, Any]:
    packet = _safe_json(packet_path)
    envelope = _safe_json(envelope_path)
    sealed = build_sealed_evidence(
        packet=packet, signed_result_attestation=envelope,
    )
    _atomic_json(output, sealed)
    return sealed


def status() -> dict[str, Any]:
    try:
        registry = authority_root.load_default_registry()
        registry_errors = authority_root.validate_registry(
            registry, verify_files=True, require_default=True,
        )
    except (OSError, authority_root.AuthorityError) as exc:
        registry_errors = [str(exc)]
    return _stamp({
        "schema": "hawking.doctor_v5_physical_result_authority_status.v1",
        "workflow": ["draft", "operator-sshsig-sign", "seal", "verify"],
        "physical_execution_capability": False,
        "default_off": True,
        "signature_namespace": SSHSIG_NAMESPACE,
        "signer_identity": SIGNER_IDENTITY,
        "source_pinned_registry_valid": not registry_errors,
        "registry_errors": registry_errors,
        "required_facets": list(FACETS),
        "raw_packet_accepted": False,
        "runtime_defaults_changed": False,
    }, "status_sha256")


def _selftest() -> int:
    value = status()
    assert value["physical_execution_capability"] is False
    assert value["raw_packet_accepted"] is False
    assert tuple(value["required_facets"]) == FACETS
    print("doctor_v5_physical_result_authority.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--draft-result", type=pathlib.Path, metavar="PACKET")
    actions.add_argument("--sign-result", type=pathlib.Path, metavar="DRAFT")
    actions.add_argument("--seal-result", type=pathlib.Path, metavar="ENVELOPE")
    actions.add_argument("--verify", type=pathlib.Path, metavar="SEALED")
    actions.add_argument("--selftest", action="store_true")
    parser.add_argument("--packet", type=pathlib.Path)
    parser.add_argument("--private-key", type=pathlib.Path, default=DEFAULT_PRIVATE_KEY)
    parser.add_argument("--signature-output", type=pathlib.Path)
    parser.add_argument("--envelope-output", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--valid-seconds", type=int, default=DEFAULT_VALIDITY_SECONDS)
    args = parser.parse_args(argv)
    try:
        if args.status:
            print(json.dumps(status(), indent=2, sort_keys=True))
            return 0
        if args.selftest:
            return _selftest()
        if args.draft_result is not None:
            if args.output is None:
                parser.error("--draft-result requires --output")
            draft_result_file(
                args.draft_result, args.output, valid_seconds=args.valid_seconds,
            )
            return 0
        if args.sign_result is not None:
            if args.packet is None or args.signature_output is None \
                    or args.envelope_output is None:
                parser.error(
                    "--sign-result requires --packet, --signature-output, and --envelope-output"
                )
            sign_result_files(
                packet_path=args.packet, draft_path=args.sign_result,
                private_key=args.private_key,
                detached_signature_output=args.signature_output,
                envelope_output=args.envelope_output,
            )
            return 0
        if args.seal_result is not None:
            if args.packet is None or args.output is None:
                parser.error("--seal-result requires --packet and --output")
            seal_result_files(
                packet_path=args.packet, envelope_path=args.seal_result,
                output=args.output,
            )
            return 0
        if args.verify is not None:
            errors = validate_sealed_evidence(_safe_json(args.verify))
            print(json.dumps({"valid": not errors, "errors": errors}, indent=2, sort_keys=True))
            return 0 if not errors else 1
    except (OSError, ResultAuthorityError, ValueError) as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 75
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

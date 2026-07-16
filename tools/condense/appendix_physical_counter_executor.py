#!/usr/bin/env python3.12
"""Fail-closed activation path for Appendix physical-counter evidence.

This is deliberately separate from the read-only counter collector contract.
``status`` and ``dry-run`` never start a collector or probe.  ``execute`` is a
release-only surface: it requires the already-open canonical heavy-lease file
descriptor, the final owner-free Doctor boundary, exact source/build/corpus
parents, green resource admission, and SSHSIG-verified authority receipts.

The Rust probes do not currently expose a native start barrier.  A tiny child
mode therefore waits on an inherited pipe and then uses ``execve`` (not a
shell) to replace itself with the exact probe argv.  This preserves the PID
that xctrace attaches to while proving that both counter streams started before
the first probe instruction can execute.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import os
import pathlib
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import appendix_contract
import appendix_device_runner
import appendix_physical_counter_collector as collector
import appendix_physical_counter_authority as authority_root
import appendix_process_joule_collector as process_joule
import appendix_xctrace_export_adapter as xctrace_adapter
import appendix_physical_release_packet as release_packet
import physical_counter_attestation
import ram_scheduler
import spec_receipt_contract
import spec_reentry_scaffold
import spec_tq_runner
import tq_receipt_contract
import tq_runtime_matrix


ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports" / "appendix" / "physical_release"
OBSERVER = ROOT / "reports" / "condense" / "doctor_v5_ultra" / "post_120b" / "observer_state.json"
HEAVY_LOCK = ROOT / "reports" / "cron" / "studio_heavy.lock"
PROCESS_JOULE_CONTRACT = pathlib.Path(process_joule.__file__).resolve()
FULL_XCODE_XCTRACE = pathlib.Path(
    "/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace"
)
SSH_KEYGEN = pathlib.Path("/usr/bin/ssh-keygen")

REQUEST_SCHEMA = "hawking.appendix_physical_counter_execution_request.v1"
STATUS_SCHEMA = "hawking.appendix_physical_counter_executor_status.v1"
DRY_RUN_SCHEMA = "hawking.appendix_physical_counter_executor_dry_run.v1"
CAPABILITY_SCHEMA = "hawking.appendix_physical_counter_execution_capability.v1"
AUTHORITY_REQUIREMENTS_SCHEMA = "hawking.appendix_counter_authority_requirements.v1"
AUTHORITY_SCHEMA = authority_root.AUTHORITY_SCHEMA
ENVELOPE_SCHEMA = authority_root.ENVELOPE_SCHEMA
EXECUTION_SCHEMA = "hawking.appendix_physical_counter_execution_receipt.v1"
SSHSIG_NAMESPACE = authority_root.SSHSIG_NAMESPACE
RESULT_SSHSIG_NAMESPACE = authority_root.RESULT_SSHSIG_NAMESPACE
RESULT_ATTESTATION_SCHEMA = authority_root.RESULT_ATTESTATION_SCHEMA
RESULT_ENVELOPE_SCHEMA = authority_root.RESULT_ENVELOPE_SCHEMA
SEALED_EVIDENCE_SCHEMA = authority_root.SEALED_EVIDENCE_SCHEMA
XCTRACE_EXPORT_EVIDENCE_SCHEMA = "hawking.appendix_xctrace_export_evidence.v1"
XCTRACE_AUTHORITY_KEYS = (
    "xctrace_capability", "xctrace_privilege", "xctrace_attribution",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXIT_BLOCKED = 75
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_CAPTURE_BYTES = physical_counter_attestation.MAX_CAPTURE_BYTES

PROCESS_JOULE_ABI = authority_root.PROCESS_JOULE_ABI
XCTRACE_ABI = authority_root.XCTRACE_ABI
NORMALIZER_ABI = authority_root.NORMALIZER_ABI
ABI_HASHES = authority_root.COMMAND_ABI_HASHES
AUTHORITY_SPECS = authority_root.AUTHORITY_SPECS


class AdmissionBlocked(RuntimeError):
    """A release-only precondition is absent; callers map this to EX_TEMPFAIL."""


class EvidenceError(ValueError):
    """A purported authority or physical artifact is malformed or inconsistent."""


def canonical_sha256(value: Any) -> str:
    return appendix_contract.canonical_sha256(value)


def _stamp(value: dict[str, Any], field: str) -> dict[str, Any]:
    stamped = copy.deepcopy(value)
    stamped.pop(field, None)
    stamped[field] = canonical_sha256(stamped)
    return stamped


def _hash_errors(value: Any, field: str, *, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} must be an object"]
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop(field, None)
    if not isinstance(claimed, str) or HEX64.fullmatch(claimed) is None:
        return [f"{label}.{field} is invalid"]
    return [] if claimed == canonical_sha256(unstamped) else [f"{label}.{field} mismatch"]


def _load_json(path: pathlib.Path) -> Any:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.absolute(), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
                or before.st_size <= 0 or before.st_size > MAX_REQUEST_BYTES:
            raise EvidenceError(f"unsafe or oversized JSON input: {path}")
        chunks: list[bytes] = []
        observed = 0
        while observed <= MAX_REQUEST_BYTES:
            chunk = os.read(descriptor, min(1024 * 1024, MAX_REQUEST_BYTES + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns, row.st_nlink,
        )
        if observed > MAX_REQUEST_BYTES or identity(before) != identity(after) or observed != after.st_size:
            raise EvidenceError(f"JSON input changed while reading: {path}")
        return json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid JSON input {path}: {exc}") from exc
    finally:
        os.close(descriptor)


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise EvidenceError(f"refusing to replace immutable output: {path}")
    finally:
        temporary.unlink(missing_ok=True)


class SafeOutputDirectory:
    """A release output directory held by inode, not by mutable pathname."""

    def __init__(
        self, requested: pathlib.Path, *, report_fd: int, directory_fd: int,
        directory_identity: tuple[int, int],
    ) -> None:
        self.requested = requested
        self.report_fd = report_fd
        self.directory_fd = directory_fd
        self.directory_identity = directory_identity
        self.closed = False

    @staticmethod
    def _open_report_root() -> int:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
            | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        current = os.open(ROOT, flags)
        try:
            for component in REPORT_ROOT.relative_to(ROOT).parts:
                try:
                    child = os.open(component, flags, dir_fd=current)
                except FileNotFoundError:
                    os.mkdir(component, mode=0o700, dir_fd=current)
                    child = os.open(component, flags, dir_fd=current)
                os.close(current)
                current = child
            return current
        except BaseException:
            os.close(current)
            raise

    @classmethod
    def create(cls, requested: pathlib.Path) -> "SafeOutputDirectory":
        if requested.parent != REPORT_ROOT or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", requested.name,
        ) is None:
            raise EvidenceError("physical output must be one safe direct child of REPORT_ROOT")
        report_fd = cls._open_report_root()
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
            | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            os.mkdir(requested.name, mode=0o700, dir_fd=report_fd)
            directory_fd = os.open(requested.name, flags, dir_fd=report_fd)
        except BaseException:
            os.close(report_fd)
            raise
        metadata = os.fstat(directory_fd)
        return cls(
            requested, report_fd=report_fd, directory_fd=directory_fd,
            directory_identity=(metadata.st_dev, metadata.st_ino),
        )

    def operational_path(self, name: str) -> pathlib.Path:
        if pathlib.PurePath(name).name != name or name in {"", ".", ".."}:
            raise EvidenceError("unsafe physical output leaf name")
        return pathlib.Path(f"/dev/fd/{self.directory_fd}") / name

    def published_path(self, name: str) -> pathlib.Path:
        return self.requested / name

    def assert_attached(self) -> None:
        current_report = self._open_report_root()
        try:
            if (os.fstat(current_report).st_dev, os.fstat(current_report).st_ino) != (
                os.fstat(self.report_fd).st_dev, os.fstat(self.report_fd).st_ino,
            ):
                raise EvidenceError("physical REPORT_ROOT was replaced during execution")
        finally:
            os.close(current_report)
        try:
            linked = os.stat(self.requested.name, dir_fd=self.report_fd, follow_symlinks=False)
        except OSError as exc:
            raise EvidenceError("physical output directory was detached during execution") from exc
        if not stat.S_ISDIR(linked.st_mode) or (linked.st_dev, linked.st_ino) != self.directory_identity:
            raise EvidenceError("physical output directory pathname was replaced during execution")

    def published_file_identity(self, name: str) -> dict[str, Any]:
        self.assert_attached()
        operational = self.operational_path(name)
        identity = physical_counter_attestation.file_identity(operational)
        identity["path"] = str(self.published_path(name))
        # Re-read through the public path only after proving its full directory
        # inode chain still maps to our held descriptor.
        observed = physical_counter_attestation.file_identity(self.published_path(name))
        if observed != identity:
            raise EvidenceError(f"published output identity changed for {name}")
        return identity

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        os.close(self.directory_fd)
        os.close(self.report_fd)

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass


def _identity_errors(value: Any, *, label: str, verify_file: bool) -> list[str]:
    errors = physical_counter_attestation._artifact_errors(
        value, label=label, verify_files=verify_file,
    )
    return errors


SignatureVerifier = Callable[[dict[str, Any], bytes], tuple[bool, str]]


def _sshsig_verify(envelope: dict[str, Any], payload: bytes) -> tuple[bool, str]:
    return authority_root.verify_sshsig_envelope(
        envelope, payload, namespace=SSHSIG_NAMESPACE,
    )


def _result_sshsig_verify(envelope: dict[str, Any], payload: bytes) -> tuple[bool, str]:
    return authority_root.verify_sshsig_envelope(
        envelope, payload, namespace=RESULT_SSHSIG_NAMESPACE,
    )


def validate_signed_authority(
    envelope: Any, *, expected_subject: str, expected_kind: str,
    required_claims: Iterable[str], expected_release_build_sha256: str,
    now_unix_ns: int, registry: Mapping[str, Any],
    expected_abi_sha256: str | None = None,
    verify_files: bool = True, signature_verifier: SignatureVerifier = _sshsig_verify,
) -> list[str]:
    """Verify integrity, expiry, immutable programs, and a detached SSHSIG."""
    expected_envelope_fields = {
        "schema", "receipt", "signer_identity", "signature_namespace",
        "allowed_signers", "detached_signature", "envelope_sha256",
    }
    if not isinstance(envelope, dict) or set(envelope) != expected_envelope_fields:
        return [f"{expected_subject}/{expected_kind} signed envelope is malformed"]
    errors = _hash_errors(envelope, "envelope_sha256", label="authority_envelope")
    errors.extend(authority_root.validate_registry(
        registry, verify_files=verify_files, require_default=True,
    ))
    if envelope.get("schema") != ENVELOPE_SCHEMA:
        errors.append("authority envelope schema is invalid")
    if envelope.get("signature_namespace") != registry.get("sshsig_namespace"):
        errors.append("authority envelope SSHSIG namespace is invalid")
    if envelope.get("signer_identity") != registry.get("signer_identity"):
        errors.append("authority envelope signer is not the independently pinned identity")
    try:
        expected_allowed_signers = authority_root.allowed_signers_identity(dict(registry))
    except (OSError, authority_root.AuthorityError) as exc:
        errors.append(f"pinned allowed-signers identity cannot be established: {exc}")
        expected_allowed_signers = None
    if envelope.get("allowed_signers") != expected_allowed_signers:
        errors.append("authority envelope attempted to select a non-pinned trust root")
    errors.extend(_identity_errors(
        envelope.get("detached_signature"), label="authority detached_signature", verify_file=verify_files,
    ))
    receipt = envelope.get("receipt")
    expected_receipt_fields = {
        "schema", "receipt_kind", "subject", "host_hardware_uuid_sha256",
        "binary", "command_abi_sha256", "claims", "issued_at_unix_ns",
        "expires_at_unix_ns", "release_build_sha256", "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_receipt_fields:
        errors.append("authority receipt is malformed")
        return errors
    errors.extend(_hash_errors(receipt, "receipt_sha256", label="authority_receipt"))
    if receipt.get("schema") != AUTHORITY_SCHEMA \
            or receipt.get("subject") != expected_subject \
            or receipt.get("receipt_kind") != expected_kind:
        errors.append("authority receipt schema/subject/kind is invalid")
    host = receipt.get("host_hardware_uuid_sha256")
    if not isinstance(host, str) or HEX64.fullmatch(host) is None:
        errors.append("authority receipt host identity is invalid")
    if receipt.get("release_build_sha256") != expected_release_build_sha256:
        errors.append("authority receipt is not bound to the exact release build")
    issued, expires = receipt.get("issued_at_unix_ns"), receipt.get("expires_at_unix_ns")
    if (
        isinstance(issued, bool) or not isinstance(issued, int) or issued <= 0
        or isinstance(expires, bool) or not isinstance(expires, int) or expires <= issued
        or not issued <= now_unix_ns <= expires
    ):
        errors.append("authority receipt is not currently valid")
    claims = receipt.get("claims")
    required = set(required_claims)
    if not isinstance(claims, list) or len(set(claims)) != len(claims) \
            or not required.issubset(set(claims)):
        errors.append("authority receipt claims are incomplete or duplicated")
    errors.extend(_identity_errors(
        receipt.get("binary"), label=f"authority {expected_subject} binary", verify_file=verify_files,
    ))
    abi = receipt.get("command_abi_sha256")
    if not isinstance(abi, str) or HEX64.fullmatch(abi) is None:
        errors.append("authority command ABI hash is invalid")
    if expected_abi_sha256 is not None and abi != expected_abi_sha256:
        errors.append("authority command ABI/parser receipt is for a different argv contract")
    if not errors:
        payload = appendix_contract.canonical_bytes(receipt)
        ok, detail = signature_verifier(envelope, payload)
        if not ok:
            errors.append("authority SSHSIG verification failed" + (f": {detail}" if detail else ""))
    return errors


def _required_abi(subject: str) -> str | None:
    return ABI_HASHES.get(subject)


def _process_provenance_claim_errors(
    provenance: Any, receipts: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if not isinstance(provenance, dict):
        return ["direct-process-joule library provenance is malformed"]
    try:
        expected_claims = {
            "dyld_shared_cache_uuid": provenance["dyld_shared_cache_uuid"],
            "os_build": provenance["os_build"],
            "machine": provenance["machine"],
            "proc_libversion_major": str(provenance["proc_libversion_major"]),
            "proc_libversion_minor": str(provenance["proc_libversion_minor"]),
            "resource_header_sha256": provenance["resource_header"]["sha256"],
            "libproc_header_sha256": provenance["libproc_header"]["sha256"],
            "struct_layout_sha256": provenance["struct_layout_sha256"],
            "library_provenance_sha256": provenance["provenance_sha256"],
        }
    except (KeyError, TypeError, ValueError):
        return ["direct-process-joule library provenance fields are incomplete"]
    claims = receipts.get("process_joule_capability", {}).get("claims", [])
    errors: list[str] = []
    for key, expected in expected_claims.items():
        values = [
            claim.split("=", 1)[1] for claim in claims
            if isinstance(claim, str) and claim.startswith(f"{key}=")
        ]
        if values != [expected]:
            errors.append(f"process-joule capability receipt lacks exact live {key}")
    return errors


def validate_authorities(
    authorities: Any, *, release_build_sha256: str, now_unix_ns: int,
    registry: Mapping[str, Any], expected_host_hardware_uuid_sha256: str,
    verify_files: bool = True, signature_verifier: SignatureVerifier = _sshsig_verify,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if not isinstance(authorities, dict) or set(authorities) != set(AUTHORITY_SPECS):
        return ["authority set is incomplete or unexpected"], {}
    errors: list[str] = []
    receipts: dict[str, dict[str, Any]] = {}
    hosts: set[str] = set()
    for key, (subject, kind, claims) in AUTHORITY_SPECS.items():
        envelope = authorities[key]
        # Capability, privilege, and attribution must all name the same exact
        # parser contract.  A privilege receipt for a different invocation is
        # not authority for the command we are about to launch.
        abi = _required_abi(subject)
        errors.extend(f"{key}: {error}" for error in validate_signed_authority(
            envelope, expected_subject=subject, expected_kind=kind,
            required_claims=claims, expected_release_build_sha256=release_build_sha256,
            now_unix_ns=now_unix_ns, registry=registry, expected_abi_sha256=abi,
            verify_files=verify_files, signature_verifier=signature_verifier,
        ))
        if isinstance(envelope, dict) and isinstance(envelope.get("receipt"), dict):
            receipts[key] = envelope["receipt"]
            host = envelope["receipt"].get("host_hardware_uuid_sha256")
            if isinstance(host, str):
                hosts.add(host)
    if len(hosts) != 1:
        errors.append("authority receipts are not bound to one exact host")
    if hosts != {expected_host_hardware_uuid_sha256}:
        errors.append("authority receipts do not match the live measured host UUID")
    for subject in ("process_joule", "xctrace", "normalizer"):
        binaries = {
            canonical_sha256(receipt.get("binary"))
            for key, receipt in receipts.items() if key.startswith(subject)
        }
        if len(binaries) != 1:
            errors.append(f"{subject} authority receipts bind different binaries")
    if receipts.get("xctrace_capability", {}).get("binary", {}).get("path") != str(FULL_XCODE_XCTRACE):
        errors.append("xctrace receipt does not bind the full-Xcode binary")
    if verify_files:
        try:
            profile_raw, _profile_identity = xctrace_adapter._read_json(
                xctrace_adapter.DEFAULT_PROFILE,
            )
            profile = xctrace_adapter.validate_profile(
                profile_raw, production=True, now_unix_ns=now_unix_ns,
            )
        except (OSError, UnicodeError, ValueError, xctrace_adapter.XctraceAdapterError) as exc:
            profile = None
            errors.append(f"signed production xctrace export profile is unavailable: {exc}")
        if isinstance(profile, dict) and profile.get("xctrace", {}).get("binary") \
                != receipts.get("xctrace_capability", {}).get("binary"):
            errors.append("xctrace authority differs from the operator-signed export profile")
    try:
        expected_normalizer = authority_root.trusted_normalizer_identity()["file"]
    except (OSError, ValueError, authority_root.AuthorityError) as exc:
        expected_normalizer = None
        errors.append(f"trusted normalizer identity cannot be established: {exc}")
    if receipts.get("normalizer_capability", {}).get("binary") != expected_normalizer:
        errors.append("normalizer receipt does not bind the pinned trusted normalizer bytes")
    for subject in ("process_joule", "xctrace", "normalizer"):
        path_raw = receipts.get(f"{subject}_capability", {}).get("binary", {}).get("path")
        if verify_files and (
            not isinstance(path_raw, str) or not os.access(path_raw, os.X_OK)
        ):
            errors.append(f"{subject} signed binary is not executable")
    for subject in ("xctrace",):
        claims = receipts.get(f"{subject}_capability", {}).get("claims", [])
        exit_codes = [
            claim.split("=", 1)[1] for claim in claims
            if isinstance(claim, str) and claim.startswith("graceful_exit_code=")
        ]
        if not exit_codes or any(
            not value.lstrip("-").isdigit() or not -255 <= int(value) <= 255
            for value in exit_codes
        ):
            errors.append(f"{subject} capability receipt lacks exact graceful exit-code ABI")
    try:
        process_provenance = process_joule.library_provenance()
    except (OSError, ValueError, process_joule.ProcessJouleError) as exc:
        process_provenance = None
        errors.append(f"direct-process-joule library provenance is unavailable: {exc}")
    if isinstance(process_provenance, dict):
        errors.extend(_process_provenance_claim_errors(process_provenance, receipts))
    device_claims = receipts.get("device_identity", {}).get("claims", [])
    for key in ("registry_id", "name", "architecture", "os_build", "driver_build"):
        values = [
            claim.split("=", 1)[1] for claim in device_claims
            if isinstance(claim, str) and claim.startswith(f"{key}=")
        ]
        if len(values) != 1 or not values[0]:
            errors.append(f"device identity receipt lacks one exact {key} value")
    lease_claims = receipts.get("inherited_lease", {}).get("claims", [])
    for key in (
        "lock_device", "lock_inode", "parent_pid", "observer_state_sha256",
        "release_boundary_attestation_sha256",
    ):
        values = [
            claim.split("=", 1)[1] for claim in lease_claims
            if isinstance(claim, str) and claim.startswith(f"{key}=")
        ]
        if len(values) != 1 or not values[0]:
            errors.append(f"inherited lease receipt lacks one exact {key} value")
    return errors, receipts


def _claim_value(receipt: Mapping[str, Any], key: str) -> str | None:
    values = [
        claim.split("=", 1)[1]
        for claim in receipt.get("claims", [])
        if isinstance(claim, str) and claim.startswith(f"{key}=")
    ]
    return values[0] if len(values) == 1 and values[0] else None


def _xctrace_export_evidence_errors_inner(
    value: Any, *, request: Mapping[str, Any], execution_receipt: Mapping[str, Any],
    core_evidence: Mapping[str, Any], verify_files: bool,
    signature_verifier: SignatureVerifier = xctrace_adapter._verify_profile_signature,
) -> list[str]:
    """Validate the exact adapter handoff, reopening every source when requested."""
    expected = {
        "schema", "adapter_schema", "adapter_contract_sha256", "kind",
        "probe_pid", "run_nonce", "probe_argv_sha256", "metal_registry_id",
        "xctrace_binary", "xctrace_authority_chain_sha256",
        "profile_identity", "profile_sha256", "raw_bundle_identity",
        "raw_bundle_sha256", "trace_identity", "toc_identity",
        "export_identities", "canonical_capture_identity", "capture_sha256",
        "adapter_receipt_identity", "adapter_receipt_sha256", "lease",
        "output_directory", "file_backed_validation",
        "physical_evidence_eligible", "xctrace_export_evidence_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return ["xctrace_export_evidence fields are incomplete or unexpected"]
    errors = _hash_errors(
        value, "xctrace_export_evidence_sha256", label="xctrace_export_evidence",
    )
    if value.get("schema") != XCTRACE_EXPORT_EVIDENCE_SCHEMA \
            or value.get("adapter_schema") != xctrace_adapter.SCHEMA \
            or value.get("adapter_contract_sha256") != xctrace_adapter.CONTRACT_SHA256 \
            or value.get("physical_evidence_eligible") is not True:
        errors.append("xctrace export evidence schema/adapter/eligibility differs")
    for field in (
        "run_nonce", "probe_argv_sha256", "xctrace_authority_chain_sha256",
        "profile_sha256", "raw_bundle_sha256", "capture_sha256",
        "adapter_receipt_sha256",
    ):
        if not isinstance(value.get(field), str) or HEX64.fullmatch(value[field]) is None:
            errors.append(f"xctrace export evidence {field} is invalid")
    probe_pid = value.get("probe_pid")
    if isinstance(probe_pid, bool) or not isinstance(probe_pid, int) or probe_pid <= 0:
        errors.append("xctrace export evidence probe PID is invalid")
    if not isinstance(value.get("metal_registry_id"), str) \
            or not value["metal_registry_id"]:
        errors.append("xctrace export evidence Metal registry ID is invalid")

    authority_chain = request.get("authorities")
    xctrace_authorities = {
        key: authority_chain.get(key) if isinstance(authority_chain, Mapping) else None
        for key in XCTRACE_AUTHORITY_KEYS
    }
    xctrace_receipt = xctrace_authorities.get("xctrace_capability")
    xctrace_receipt = xctrace_receipt.get("receipt") \
        if isinstance(xctrace_receipt, Mapping) else None
    device_envelope = authority_chain.get("device_identity") \
        if isinstance(authority_chain, Mapping) else None
    device_receipt = device_envelope.get("receipt") \
        if isinstance(device_envelope, Mapping) else {}
    lease_envelope = authority_chain.get("inherited_lease") \
        if isinstance(authority_chain, Mapping) else None
    lease_receipt = lease_envelope.get("receipt") \
        if isinstance(lease_envelope, Mapping) else {}
    expected_lease = {
        "inherited": True,
        "device": int(_claim_value(lease_receipt, "lock_device") or -1),
        "inode": int(_claim_value(lease_receipt, "lock_inode") or -1),
    }
    expected_authority_sha = canonical_sha256(xctrace_authorities)
    if value.get("xctrace_binary") != (
        xctrace_receipt.get("binary") if isinstance(xctrace_receipt, Mapping) else None
    ) or value.get("xctrace_binary", {}).get("path") != str(FULL_XCODE_XCTRACE):
        errors.append("xctrace export evidence differs from signed full-Xcode authority")
    if value.get("xctrace_authority_chain_sha256") != expected_authority_sha:
        errors.append("xctrace export evidence does not bind all signed xctrace authorities")
    if value.get("lease") != expected_lease:
        errors.append("xctrace export evidence differs from signed inherited lease")
    if value.get("metal_registry_id") != _claim_value(device_receipt, "registry_id"):
        errors.append("xctrace export evidence differs from signed Metal registry ID")

    bundle = core_evidence.get("raw_bundle")
    execution = core_evidence.get("execution_authority")
    if value.get("kind") != request.get("kind") \
            or value.get("kind") != execution_receipt.get("kind"):
        errors.append("xctrace export evidence kind differs from the physical execution")
    if value.get("probe_pid") != execution_receipt.get("probe_pid") \
            or value.get("run_nonce") != execution_receipt.get("run_nonce"):
        errors.append("xctrace export evidence PID/nonce differs from execution receipt")
    if not isinstance(execution, Mapping) \
            or value.get("run_nonce") != execution.get("run_nonce") \
            or value.get("probe_argv_sha256") != execution.get("argv_sha256") \
            or value.get("probe_argv_sha256") != execution_receipt.get("probe_argv_sha256"):
        errors.append("xctrace export evidence differs from exact probe argv authority")
    bundle_sha = bundle.get("raw_bundle_sha256") if isinstance(bundle, Mapping) else None
    if value.get("raw_bundle_sha256") != bundle_sha:
        errors.append("xctrace export evidence differs from exact raw bundle")

    identity_fields = (
        "xctrace_binary", "profile_identity", "raw_bundle_identity",
        "toc_identity", "canonical_capture_identity", "adapter_receipt_identity",
    )
    for field in identity_fields:
        errors.extend(_identity_errors(
            value.get(field), label=f"xctrace export evidence {field}",
            verify_file=verify_files,
        ))
    exports = value.get("export_identities")
    if not isinstance(exports, dict) or set(exports) != set(xctrace_adapter.REQUIRED_TABLES):
        errors.append("xctrace export evidence does not bind the exact four exports")
        exports = {}
    for table in xctrace_adapter.REQUIRED_TABLES:
        errors.extend(_identity_errors(
            exports.get(table), label=f"xctrace export evidence {table} export",
            verify_file=verify_files,
        ))
    trace = value.get("trace_identity")
    if not isinstance(trace, dict) or set(trace) != {
        "schema", "path", "total_size_bytes", "files", "tree_sha256",
    } or trace.get("schema") != "hawking.xctrace_trace_tree_identity.v1" \
            or not isinstance(trace.get("path"), str) \
            or not pathlib.Path(trace.get("path", "")).is_absolute() \
            or not isinstance(trace.get("files"), list) or not trace["files"] \
            or not isinstance(trace.get("tree_sha256"), str) \
            or HEX64.fullmatch(trace["tree_sha256"]) is None:
        errors.append("xctrace export evidence trace-tree identity is malformed")

    output_directory = value.get("output_directory")
    if not isinstance(output_directory, dict) or set(output_directory) != {
        "held", "path", "device", "inode",
    } or output_directory.get("held") is not True \
            or output_directory.get("path") != request.get("output_directory"):
        errors.append("xctrace export evidence stable output-directory binding differs")
        output_directory = {}
    else:
        output_root = pathlib.Path(output_directory["path"])
        bound_paths = [
            value.get("raw_bundle_identity", {}).get("path"),
            trace.get("path") if isinstance(trace, Mapping) else None,
            value.get("toc_identity", {}).get("path"),
            value.get("canonical_capture_identity", {}).get("path"),
            value.get("adapter_receipt_identity", {}).get("path"),
            *(row.get("path") for row in exports.values() if isinstance(row, Mapping)),
        ]
        if any(
            not isinstance(path, str) or pathlib.Path(path).parent != output_root
            for path in bound_paths
        ):
            errors.append("xctrace export provenance is not published under the stable output path")
    if value.get("profile_identity", {}).get("path") \
            != str(xctrace_adapter.DEFAULT_PROFILE.absolute()):
        errors.append("xctrace export evidence profile is not the pinned production profile")
    validation = value.get("file_backed_validation")
    expected_validation = {
        "receipt_sha256": value.get("adapter_receipt_sha256"),
        "capture_sha256": value.get("capture_sha256"),
        "all_provenance_files_reopened": True,
        "physical_evidence_eligible": True,
    }
    if validation != expected_validation:
        errors.append("xctrace export evidence file-backed validation receipt differs")

    if verify_files and not errors:
        try:
            trace_path = pathlib.Path(trace["path"])
            if xctrace_adapter.trace_tree_identity(
                trace_path, require_immutable=True,
            ) != trace:
                errors.append("xctrace export evidence trace tree differs after publication")
            observed = xctrace_adapter.validate_receipt(
                receipt_path=pathlib.Path(value["adapter_receipt_identity"]["path"]),
                capture_path=pathlib.Path(value["canonical_capture_identity"]["path"]),
                kind=value["kind"],
                raw_bundle_path=pathlib.Path(value["raw_bundle_identity"]["path"]),
                profile_path=pathlib.Path(value["profile_identity"]["path"]),
                trace_path=trace_path,
                toc_path=pathlib.Path(value["toc_identity"]["path"]),
                export_paths={
                    table: pathlib.Path(exports[table]["path"])
                    for table in xctrace_adapter.REQUIRED_TABLES
                },
                probe_pid=value["probe_pid"], run_nonce=value["run_nonce"],
                probe_argv_sha256=value["probe_argv_sha256"],
                metal_registry_id=value["metal_registry_id"],
                expected_lease=value["lease"],
                expected_output_directory=output_directory,
                signature_verifier=signature_verifier,
            )
            if observed != validation:
                errors.append("xctrace export evidence no longer recomputes from disk")
        except (
            OSError, UnicodeError, ValueError, KeyError, TypeError,
            xctrace_adapter.XctraceAdapterError,
        ) as exc:
            errors.append(f"xctrace export evidence file-backed verification failed: {exc}")
    return list(dict.fromkeys(errors))


def _xctrace_export_evidence_errors(
    value: Any, *, request: Mapping[str, Any], execution_receipt: Mapping[str, Any],
    core_evidence: Mapping[str, Any], verify_files: bool,
    signature_verifier: SignatureVerifier = xctrace_adapter._verify_profile_signature,
) -> list[str]:
    """No-throw facade for adversarial sealed-evidence inputs."""
    try:
        return _xctrace_export_evidence_errors_inner(
            value, request=request, execution_receipt=execution_receipt,
            core_evidence=core_evidence, verify_files=verify_files,
            signature_verifier=signature_verifier,
        )
    except (
        AttributeError, KeyError, OSError, TypeError, ValueError, OverflowError,
        RecursionError, xctrace_adapter.XctraceAdapterError,
    ) as exc:
        return [f"xctrace_export_evidence is malformed: {exc}"]


def build_xctrace_export_evidence(
    *, request: Mapping[str, Any], bundle: Mapping[str, Any],
    execution_authority: Mapping[str, Any], probe_pid: int,
    metal_registry_id: str, capture: Mapping[str, Any], receipt: Mapping[str, Any],
    capture_path: pathlib.Path, receipt_path: pathlib.Path,
    file_backed_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the adapter receipt and every stable published provenance path."""
    authorities = request["authorities"]
    xctrace_authorities = {key: authorities[key] for key in XCTRACE_AUTHORITY_KEYS}
    value = _stamp({
        "schema": XCTRACE_EXPORT_EVIDENCE_SCHEMA,
        "adapter_schema": xctrace_adapter.SCHEMA,
        "adapter_contract_sha256": xctrace_adapter.CONTRACT_SHA256,
        "kind": request["kind"], "probe_pid": probe_pid,
        "run_nonce": execution_authority["run_nonce"],
        "probe_argv_sha256": execution_authority["argv_sha256"],
        "metal_registry_id": metal_registry_id,
        "xctrace_binary": authorities["xctrace_capability"]["receipt"]["binary"],
        "xctrace_authority_chain_sha256": canonical_sha256(xctrace_authorities),
        "profile_identity": receipt["profile_identity"],
        "profile_sha256": receipt["profile_sha256"],
        "raw_bundle_identity": receipt["raw_bundle_identity"],
        "raw_bundle_sha256": bundle["raw_bundle_sha256"],
        "trace_identity": receipt["trace_identity"],
        "toc_identity": receipt["toc_identity"],
        "export_identities": receipt["export_identities"],
        "canonical_capture_identity": physical_counter_attestation.file_identity(
            capture_path,
        ),
        "capture_sha256": capture["capture_sha256"],
        "adapter_receipt_identity": physical_counter_attestation.file_identity(
            receipt_path,
        ),
        "adapter_receipt_sha256": receipt["receipt_sha256"],
        "lease": receipt["lease"],
        "output_directory": receipt["xctrace_runtime"]["output_directory"],
        "file_backed_validation": dict(file_backed_validation),
        "physical_evidence_eligible": True,
    }, "xctrace_export_evidence_sha256")
    # The production executor has already invoked the file-backed adapter
    # validator.  This structural replay catches assembly mistakes without
    # performing a third export parse here; sealed validation reopens it all.
    provisional_execution = {
        "kind": request["kind"], "probe_pid": probe_pid,
        "run_nonce": execution_authority["run_nonce"],
        "probe_argv_sha256": execution_authority["argv_sha256"],
    }
    core = {
        "raw_bundle": bundle, "execution_authority": execution_authority,
    }
    errors = _xctrace_export_evidence_errors(
        value, request=request, execution_receipt=provisional_execution,
        core_evidence=core, verify_files=False,
    )
    if errors:
        raise EvidenceError("xctrace export evidence assembly failed: " + "; ".join(errors))
    return value


def _core_evidence_errors(
    evidence: Any, *, request: Mapping[str, Any],
    execution_receipt: Mapping[str, Any], verify_files: bool,
    xctrace_signature_verifier: SignatureVerifier = (
        xctrace_adapter._verify_profile_signature
    ),
) -> list[str]:
    if not isinstance(evidence, dict):
        return ["core evidence must be an object"]
    expected_by_kind = {
        "device": {
            "cell_id", "raw_bundle", "receipt", "execution_authority",
            "counter_payload", "counter_attestation", "xctrace_export_evidence",
        },
        "spec": {
            "runtime_path", "raw_bundle", "parity_receipt", "curve_receipt",
            "execution_authority", "counter_payload", "counter_attestation",
            "xctrace_export_evidence",
        },
    }
    kind = request.get("kind")
    expected = expected_by_kind.get(kind)
    if expected is None:
        return ["core evidence request kind is invalid"]
    errors: list[str] = []
    if set(evidence) != expected:
        errors.append(f"{kind} core evidence fields are incomplete or unexpected")
    errors.extend(_xctrace_export_evidence_errors(
        evidence.get("xctrace_export_evidence"), request=request,
        execution_receipt=execution_receipt, core_evidence=evidence,
        verify_files=verify_files, signature_verifier=xctrace_signature_verifier,
    ))
    return errors


def _execution_receipt_errors(
    receipt: Any, *, request: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> list[str]:
    expected = {
        "schema", "request_sha256", "request_file", "authority_chain_sha256",
        "kind", "run_nonce", "probe_pid", "probe_argv_sha256",
        "external_trace_started_before_barrier", "capture_readiness",
        "execution_started_at_unix_ns", "execution_ended_at_unix_ns",
        "raw_bundle_sha256", "counter_payload_sha256",
        "counter_attestation_sha256", "xctrace_export_evidence_sha256",
        "core_evidence_sha256", "evidence_file",
        "process_joule_provenance_file",
        "prelaunch_release_cas", "final_release_cas", "runtime_defaults_changed",
        "execution_receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        return ["execution receipt fields are incomplete or unexpected"]
    errors = _hash_errors(receipt, "execution_receipt_sha256", label="execution_receipt")
    if receipt.get("schema") != EXECUTION_SCHEMA or receipt.get("kind") not in {"device", "spec"}:
        errors.append("execution receipt schema/kind is invalid")
    if not isinstance(receipt.get("run_nonce"), str) \
            or re.fullmatch(r"[0-9a-f]{64}", receipt.get("run_nonce", "")) is None:
        errors.append("execution receipt run nonce is invalid")
    if receipt.get("external_trace_started_before_barrier") is not True \
            or receipt.get("runtime_defaults_changed") is not False:
        errors.append("execution receipt weakens barrier/default invariants")
    started = receipt.get("execution_started_at_unix_ns")
    ended = receipt.get("execution_ended_at_unix_ns")
    if isinstance(started, bool) or not isinstance(started, int) or started <= 0 \
            or isinstance(ended, bool) or not isinstance(ended, int) or ended < started:
        errors.append("execution receipt time interval is invalid")
    for field in (
        "request_sha256", "authority_chain_sha256", "probe_argv_sha256",
        "raw_bundle_sha256", "counter_payload_sha256", "counter_attestation_sha256",
        "xctrace_export_evidence_sha256", "core_evidence_sha256",
    ):
        if not isinstance(receipt.get(field), str) or HEX64.fullmatch(receipt[field]) is None:
            errors.append(f"execution receipt {field} is invalid")
    for field in ("request_file", "evidence_file", "process_joule_provenance_file"):
        errors.extend(_identity_errors(
            receipt.get(field), label=f"execution receipt {field}", verify_file=False,
        ))
    for phase in ("prelaunch_release_cas", "final_release_cas"):
        cas = receipt.get(phase)
        if not isinstance(cas, dict) or _hash_errors(cas, "cas_sha256", label=phase):
            errors.append(f"execution receipt {phase} is invalid")
    if request is not None:
        if receipt.get("request_sha256") != request.get("request_sha256"):
            errors.append("execution receipt is not bound to the exact request")
        if receipt.get("kind") != request.get("kind"):
            errors.append("execution receipt kind differs from request")
        if receipt.get("authority_chain_sha256") != canonical_sha256(request.get("authorities")):
            errors.append("execution receipt does not bind the full signed authority chain")
    if evidence is not None:
        if receipt.get("core_evidence_sha256") != canonical_sha256(evidence):
            errors.append("execution receipt does not bind the exact core evidence")
        xctrace_evidence = evidence.get("xctrace_export_evidence") \
            if isinstance(evidence, Mapping) else None
        xctrace_sha = xctrace_evidence.get("xctrace_export_evidence_sha256") \
            if isinstance(xctrace_evidence, Mapping) else None
        if receipt.get("xctrace_export_evidence_sha256") != xctrace_sha:
            errors.append("execution receipt does not bind exact xctrace export evidence")
        authority = evidence.get("execution_authority") if isinstance(evidence, Mapping) else None
        if not isinstance(authority, Mapping) or authority.get("run_nonce") != receipt.get("run_nonce"):
            errors.append("execution receipt run nonce differs from core evidence")
    return errors


def build_result_attestation(
    *, request: Mapping[str, Any], execution_receipt: Mapping[str, Any],
    evidence: Mapping[str, Any], attested_at_unix_ns: int | None = None,
) -> dict[str, Any]:
    errors = _validate_request_structure(request, require_unused_output=False)
    errors.extend(_execution_receipt_errors(
        execution_receipt, request=request, evidence=evidence,
    ))
    if errors:
        raise EvidenceError("cannot draft result attestation: " + "; ".join(errors))
    release = request["release"]
    probe = request["workload"]["probe"]
    authorities = request["authorities"]
    host_values = {
        envelope.get("receipt", {}).get("host_hardware_uuid_sha256")
        for envelope in authorities.values() if isinstance(envelope, dict)
    }
    if len(host_values) != 1 or not isinstance(next(iter(host_values), None), str):
        raise EvidenceError("cannot draft result attestation from a non-uniform authority host")
    attested = time.time_ns() if attested_at_unix_ns is None else attested_at_unix_ns
    ended = execution_receipt["execution_ended_at_unix_ns"]
    if isinstance(attested, bool) or not isinstance(attested, int) or attested < ended:
        raise EvidenceError("result attestation time predates physical execution")
    return _stamp({
        "schema": RESULT_ATTESTATION_SCHEMA,
        "kind": request["kind"],
        "run_nonce": execution_receipt["run_nonce"],
        "request_sha256": request["request_sha256"],
        "execution_receipt_sha256": execution_receipt["execution_receipt_sha256"],
        "core_evidence_sha256": canonical_sha256(evidence),
        "authority_chain_sha256": canonical_sha256(authorities),
        "host_hardware_uuid_sha256": next(iter(host_values)),
        "release_boundary_attestation_sha256": release["boundary_attestation"][
            "attestation_sha256"
        ],
        "release_boundary_observation_sha256": release["boundary_observation"][
            "observation_sha256"
        ],
        "release_build_sha256": release["release_build"]["receipt_sha256"],
        "corpus_index_sha256": release["corpus_index"]["index_sha256"],
        "source_manifest_sha256": release["source_manifest"]["manifest_sha256"],
        "probe_binary_sha256": probe["sha256"],
        "workload_identity_sha256": canonical_sha256(request["workload"]),
        "execution_started_at_unix_ns": execution_receipt["execution_started_at_unix_ns"],
        "execution_ended_at_unix_ns": ended,
        "attested_at_unix_ns": attested,
        "runtime_defaults_changed": False,
    }, "result_attestation_sha256")


def validate_result_attestation(
    value: Any, *, request: Mapping[str, Any], execution_receipt: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[str]:
    try:
        expected = build_result_attestation(
            request=request, execution_receipt=execution_receipt, evidence=evidence,
            attested_at_unix_ns=value.get("attested_at_unix_ns") if isinstance(value, dict) else 0,
        )
    except (EvidenceError, KeyError, TypeError, ValueError) as exc:
        return [f"result attestation inputs are invalid: {exc}"]
    return [] if value == expected else ["result attestation differs from exact dynamic execution bindings"]


def validate_result_envelope(
    envelope: Any, *, request: Mapping[str, Any], execution_receipt: Mapping[str, Any],
    evidence: Mapping[str, Any], verify_files: bool = True,
    signature_verifier: SignatureVerifier = _result_sshsig_verify,
) -> list[str]:
    expected_fields = {
        "schema", "attestation", "signer_identity", "signature_namespace",
        "allowed_signers", "detached_signature", "envelope_sha256",
    }
    if not isinstance(envelope, dict) or set(envelope) != expected_fields:
        return ["signed result envelope is malformed"]
    errors = _hash_errors(envelope, "envelope_sha256", label="result_envelope")
    registry = request.get("release", {}).get("authority_registry")
    if not isinstance(registry, dict):
        errors.append("signed result envelope lacks the request trust root")
        return errors
    errors.extend(authority_root.validate_registry(
        registry, verify_files=verify_files, require_default=True,
    ))
    if envelope.get("schema") != RESULT_ENVELOPE_SCHEMA \
            or envelope.get("signer_identity") != registry.get("signer_identity") \
            or envelope.get("signature_namespace") != RESULT_SSHSIG_NAMESPACE:
        errors.append("signed result envelope schema/signer/namespace is invalid")
    try:
        pinned = authority_root.allowed_signers_identity(registry)
    except (OSError, authority_root.AuthorityError) as exc:
        errors.append(f"result signer identity cannot be established: {exc}")
        pinned = None
    if envelope.get("allowed_signers") != pinned:
        errors.append("signed result envelope attempted to select a non-pinned trust root")
    errors.extend(_identity_errors(
        envelope.get("detached_signature"), label="result detached_signature",
        verify_file=verify_files,
    ))
    attestation = envelope.get("attestation")
    errors.extend(validate_result_attestation(
        attestation, request=request, execution_receipt=execution_receipt,
        evidence=evidence,
    ))
    if not errors:
        ok, detail = signature_verifier(envelope, appendix_contract.canonical_bytes(attestation))
        if not ok:
            errors.append("result SSHSIG verification failed" + (f": {detail}" if detail else ""))
    return errors


def validate_sealed_evidence(
    value: Any, *, verify_files: bool = True,
    authority_signature_verifier: SignatureVerifier = _sshsig_verify,
    result_signature_verifier: SignatureVerifier = _result_sshsig_verify,
    xctrace_signature_verifier: SignatureVerifier = (
        xctrace_adapter._verify_profile_signature
    ),
) -> list[str]:
    expected = {
        "schema", "request", "execution_receipt", "signed_authority_chain",
        "core_evidence", "signed_result_attestation", "sealed_at_unix_ns",
        "default_mutation_requested", "sealed_evidence_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return ["sealed counter evidence fields are incomplete or unexpected"]
    errors = _hash_errors(value, "sealed_evidence_sha256", label="sealed_counter_evidence")
    if value.get("schema") != SEALED_EVIDENCE_SCHEMA \
            or value.get("default_mutation_requested") is not False:
        errors.append("sealed counter evidence schema/default policy is invalid")
    request = value.get("request")
    execution = value.get("execution_receipt")
    evidence = value.get("core_evidence")
    errors.extend(_validate_request_structure(request, require_unused_output=False))
    if not isinstance(request, dict) or not isinstance(execution, dict) \
            or not isinstance(evidence, dict):
        return errors
    if value.get("signed_authority_chain") != request.get("authorities"):
        errors.append("sealed evidence dropped or changed the request authority chain")
    errors.extend(_execution_receipt_errors(execution, request=request, evidence=evidence))
    errors.extend(_core_evidence_errors(
        evidence, request=request, execution_receipt=execution,
        verify_files=verify_files,
        xctrace_signature_verifier=xctrace_signature_verifier,
    ))
    execution_time = execution.get("execution_started_at_unix_ns")
    execution_end = execution.get("execution_ended_at_unix_ns")
    signed_result = value.get("signed_result_attestation")
    result_attestation = signed_result.get("attestation") \
        if isinstance(signed_result, Mapping) else None
    host = result_attestation.get("host_hardware_uuid_sha256") \
        if isinstance(result_attestation, Mapping) else None
    release = request.get("release")
    release_build = release.get("release_build") if isinstance(release, Mapping) else None
    registry = release.get("authority_registry") if isinstance(release, Mapping) else None
    release_build_sha256 = release_build.get("receipt_sha256") \
        if isinstance(release_build, Mapping) else None
    authority_context_valid = (
        isinstance(release_build_sha256, str)
        and HEX64.fullmatch(release_build_sha256) is not None
        and isinstance(registry, dict)
    )
    if not authority_context_valid:
        errors.append("sealed evidence lacks a valid release build and pinned authority registry")
    authority_receipts: dict[str, dict[str, Any]] = {}
    if isinstance(execution_time, int) and not isinstance(execution_time, bool) \
            and isinstance(host, str) and authority_context_valid:
        authority_errors, authority_receipts = validate_authorities(
            value.get("signed_authority_chain"),
            release_build_sha256=release_build_sha256,
            now_unix_ns=execution_time,
            registry=registry,
            expected_host_hardware_uuid_sha256=host,
            verify_files=verify_files,
            signature_verifier=authority_signature_verifier,
        )
        errors.extend(f"sealed authority chain: {error}" for error in authority_errors)
        if isinstance(execution_end, int) and not isinstance(execution_end, bool):
            end_errors, _ = validate_authorities(
                value.get("signed_authority_chain"),
                release_build_sha256=release_build_sha256,
                now_unix_ns=execution_end,
                registry=registry,
                expected_host_hardware_uuid_sha256=host,
                verify_files=verify_files,
                signature_verifier=authority_signature_verifier,
            )
            errors.extend(
                f"sealed authority chain at execution end: {error}" for error in end_errors
            )
    else:
        errors.append("sealed evidence lacks a recorded execution time/host for authority validation")
    workload = request.get("workload")
    probe = workload.get("probe") if isinstance(workload, Mapping) else None
    errors.extend(_identity_errors(
        probe, label="sealed request selected release probe", verify_file=verify_files,
    ))
    expected_probe_abi = ABI_HASHES.get(f"{request.get('kind')}-probe")
    if authority_receipts.get("device_identity", {}).get("binary") != probe:
        errors.append("sealed device identity receipt differs from the selected release probe")
    if authority_receipts.get("device_identity", {}).get("command_abi_sha256") \
            != expected_probe_abi:
        errors.append("sealed device identity receipt binds a different probe argv ABI")
    for key in ("process_joule_capability", "process_joule_attribution"):
        if authority_receipts.get(key, {}).get("binary") != probe:
            errors.append(f"sealed {key} receipt differs from the self-sampling release probe")
    errors.extend(validate_result_envelope(
        value.get("signed_result_attestation"), request=request,
        execution_receipt=execution, evidence=evidence, verify_files=verify_files,
        signature_verifier=result_signature_verifier,
    ))
    sealed_at = value.get("sealed_at_unix_ns")
    attested_at = result_attestation.get("attested_at_unix_ns") \
        if isinstance(result_attestation, Mapping) else None
    if isinstance(sealed_at, bool) or not isinstance(sealed_at, int) \
            or not isinstance(attested_at, int) or sealed_at < attested_at:
        errors.append("sealed evidence timestamp predates the operator attestation")
    return errors


def build_sealed_evidence(
    *, request: Mapping[str, Any], execution_receipt: Mapping[str, Any],
    evidence: Mapping[str, Any], signed_result_attestation: Mapping[str, Any],
    sealed_at_unix_ns: int | None = None,
    authority_signature_verifier: SignatureVerifier = _sshsig_verify,
    result_signature_verifier: SignatureVerifier = _result_sshsig_verify,
    xctrace_signature_verifier: SignatureVerifier = (
        xctrace_adapter._verify_profile_signature
    ),
    verify_files: bool = True,
) -> dict[str, Any]:
    result_attestation = signed_result_attestation.get("attestation") \
        if isinstance(signed_result_attestation, Mapping) else None
    attested_at = result_attestation.get("attested_at_unix_ns") \
        if isinstance(result_attestation, Mapping) else None
    sealed_at = time.time_ns() if sealed_at_unix_ns is None else sealed_at_unix_ns
    if not isinstance(attested_at, int) or isinstance(attested_at, bool) \
            or not isinstance(sealed_at, int) or isinstance(sealed_at, bool) \
            or sealed_at < attested_at:
        raise EvidenceError("sealed evidence time predates the signed result")
    value = _stamp({
        "schema": SEALED_EVIDENCE_SCHEMA,
        "request": dict(request),
        "execution_receipt": dict(execution_receipt),
        "signed_authority_chain": copy.deepcopy(request.get("authorities")),
        "core_evidence": dict(evidence),
        "signed_result_attestation": dict(signed_result_attestation),
        "sealed_at_unix_ns": sealed_at,
        "default_mutation_requested": False,
    }, "sealed_evidence_sha256")
    errors = validate_sealed_evidence(
        value, verify_files=verify_files,
        authority_signature_verifier=authority_signature_verifier,
        result_signature_verifier=result_signature_verifier,
        xctrace_signature_verifier=xctrace_signature_verifier,
    )
    if errors:
        raise EvidenceError("sealed evidence validation failed: " + "; ".join(errors))
    return value


def draft_result_files(
    *, request_path: pathlib.Path, execution_receipt_path: pathlib.Path,
    evidence_path: pathlib.Path, output_path: pathlib.Path,
) -> dict[str, Any]:
    """Draft the exact dynamic result attestation from immutable run files."""
    request = _load_json(request_path)
    execution = _load_json(execution_receipt_path)
    evidence = _load_json(evidence_path)
    if not all(isinstance(row, dict) for row in (request, execution, evidence)):
        raise EvidenceError("result draft inputs must all be JSON objects")
    if execution.get("request_file") != physical_counter_attestation.file_identity(request_path):
        raise EvidenceError("execution receipt request file differs from the drafting input")
    if execution.get("evidence_file") != physical_counter_attestation.file_identity(evidence_path):
        raise EvidenceError("execution receipt evidence file differs from the drafting input")
    value = build_result_attestation(
        request=request, execution_receipt=execution, evidence=evidence,
    )
    _atomic_json(output_path, value)
    return value


def sign_result_files(
    *, request_path: pathlib.Path, execution_receipt_path: pathlib.Path,
    evidence_path: pathlib.Path, draft_path: pathlib.Path,
    private_key: pathlib.Path, detached_signature_output: pathlib.Path,
    envelope_output: pathlib.Path,
) -> dict[str, Any]:
    """Validate a result draft against its run, then invoke the pinned signer."""
    request = _load_json(request_path)
    execution = _load_json(execution_receipt_path)
    evidence = _load_json(evidence_path)
    draft = _load_json(draft_path)
    if not all(isinstance(row, dict) for row in (request, execution, evidence, draft)):
        raise EvidenceError("result signing inputs must all be JSON objects")
    if execution.get("request_file") != physical_counter_attestation.file_identity(request_path):
        raise EvidenceError("execution receipt request file differs from the signing input")
    if execution.get("evidence_file") != physical_counter_attestation.file_identity(evidence_path):
        raise EvidenceError("execution receipt evidence file differs from the signing input")
    draft_errors = validate_result_attestation(
        draft, request=request, execution_receipt=execution, evidence=evidence,
    )
    if draft_errors:
        raise EvidenceError("operator refused invalid result draft: " + "; ".join(draft_errors))
    return authority_root.sign_result_attestation(
        draft, private_key=private_key,
        detached_signature_output=detached_signature_output,
        envelope_output=envelope_output,
    )


def seal_result_files(
    *, request_path: pathlib.Path, execution_receipt_path: pathlib.Path,
    evidence_path: pathlib.Path, signed_result_path: pathlib.Path,
    output_path: pathlib.Path,
    authority_signature_verifier: SignatureVerifier = _sshsig_verify,
    result_signature_verifier: SignatureVerifier = _result_sshsig_verify,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate the complete provenance chain, then publish one immutable item."""
    request = _load_json(request_path)
    execution = _load_json(execution_receipt_path)
    evidence = _load_json(evidence_path)
    signed_result = _load_json(signed_result_path)
    if not all(isinstance(row, dict) for row in (request, execution, evidence, signed_result)):
        raise EvidenceError("result seal inputs must all be JSON objects")
    if execution.get("request_file") != physical_counter_attestation.file_identity(request_path):
        raise EvidenceError("execution receipt request file differs from the sealing input")
    if execution.get("evidence_file") != physical_counter_attestation.file_identity(evidence_path):
        raise EvidenceError("execution receipt evidence file differs from the sealing input")
    value = build_sealed_evidence(
        request=request, execution_receipt=execution, evidence=evidence,
        signed_result_attestation=signed_result,
        authority_signature_verifier=authority_signature_verifier,
        result_signature_verifier=result_signature_verifier,
        verify_files=verify_files,
    )
    _atomic_json(output_path, value)
    return value


def execution_capability_contract() -> dict[str, Any]:
    """Stable interface for orchestrators; it grants no present admission.

    The collector/normalizer module intentionally keeps
    ``collection_cli_exposed=false``.  Callers must bind this separate contract
    instead of weakening that default-off invariant.
    """
    authority = authority_requirements()
    registry = authority_root.load_default_registry()
    return _stamp({
        "schema": CAPABILITY_SCHEMA,
        "surface": "separate-release-gated-executor",
        "execute_cli": [
            "python3.12", "tools/condense/appendix_physical_counter_executor.py",
            "--execute", "REQUEST", "--acknowledge-request-sha256", "SHA256",
        ],
        "result_workflow_cli": {
            "draft": [
                "--draft-result", "REQUEST", "--execution-receipt", "RECEIPT",
                "--evidence", "CORE", "--output", "DRAFT",
            ],
            "sign": [
                "--sign-result", "DRAFT", "--request", "REQUEST",
                "--execution-receipt", "RECEIPT", "--evidence", "CORE",
                "--private-key", "OPERATOR_KEY", "--signature-output", "SSHSIG",
                "--envelope-output", "SIGNED_RESULT",
            ],
            "seal": [
                "--seal-result", "SIGNED_RESULT", "--request", "REQUEST",
                "--execution-receipt", "RECEIPT", "--evidence", "CORE",
                "--output", "SEALED_EVIDENCE",
            ],
        },
        "request_schema": REQUEST_SCHEMA,
        "execution_receipt_schema": EXECUTION_SCHEMA,
        "blocked_exit_code": EXIT_BLOCKED,
        "collector_normalizer_collection_cli_exposed": False,
        "collector_config_sha256": collector.build_config()["config_sha256"],
        "execute_surface_exposed": True,
        "present_execution_admission": False,
        "requires_signed_authority_keys": sorted(AUTHORITY_SPECS),
        "authority_requirements_sha256": authority["requirements_sha256"],
        "authority_registry_sha256": registry["registry_sha256"],
        "allowed_signers_sha256": registry["allowed_signers_sha256"],
        "exact_command_abi_sha256": ABI_HASHES,
        "explicit_abi_boundaries": [
            "Rust probes have no native start barrier; executor wrapper must retain PID then execve",
            "release probes must bracket each operation-only PhaseRecorder interval with libproc RUSAGE_INFO_V6 snapshots",
            "powermetrics energy-impact is explicitly ineligible for joule evidence",
            "full-Xcode Metal System Trace attach/output argv must have a current signed parser receipt",
            "xctrace trace-package readiness and deterministic sealed export must be receipt-proven",
            "a separately pinned trusted attributed-sample normalizer must implement NORMALIZER_ABI",
        ],
        "invariants": [
            "one inherited canonical shared-heavy-lease descriptor",
            "Doctor final-ready and zero owners rechecked before Popen",
            "green RAM/swap/thermal/disk admission before Popen",
            "full-Xcode xctrace authority and signed live libproc provenance before Popen",
            "xctrace is live before the PID-preserving probe barrier",
            "exactly one direct device/spec probe and no shell",
            "trusted attributed-sample normalizer then existing v2 validator/attestation/finalizer",
            "post-run result draft, operator SSHSIG, and sealed core are exact dynamically validated stages",
            "runtime defaults remain unchanged",
        ],
    }, "capability_sha256")


def authority_requirements() -> dict[str, Any]:
    """Return the exact, non-authorizing receipt/envelope contract."""
    rows = []
    for key, (subject, kind, claims) in sorted(AUTHORITY_SPECS.items()):
        dynamic: list[str] = []
        if key == "xctrace_capability":
            dynamic.append("graceful_exit_code=<signed integer in -255..255>")
        if key == "process_joule_capability":
            dynamic.extend([
                "dyld_shared_cache_uuid=<live 128-bit lowercase hex>",
                "os_build=<live Darwin build>", "machine=<live architecture>",
                "proc_libversion_major=<live integer>",
                "proc_libversion_minor=<live integer>",
                "resource_header_sha256=<release SDK header bytes>",
                "libproc_header_sha256=<release SDK header bytes>",
                "struct_layout_sha256=<pinned rusage_info_v6 layout>",
                "library_provenance_sha256=<exact live provenance receipt>",
            ])
        if key == "device_identity":
            dynamic.extend([
                "registry_id=<nonempty>", "name=<nonempty>",
                "architecture=<nonempty>", "os_build=<nonempty>",
                "driver_build=<nonempty>",
            ])
        if key == "inherited_lease":
            dynamic.extend([
                "lock_device=<live canonical st_dev>", "lock_inode=<live canonical st_ino>",
                "parent_pid=<executor parent PID>",
                "observer_state_sha256=<current observer hash>",
                "release_boundary_attestation_sha256=<exact boundary hash>",
            ])
        rows.append({
            "key": key, "subject": subject, "receipt_kind": kind,
            "required_claims": sorted(claims), "required_dynamic_claims": dynamic,
            "command_abi_sha256": _required_abi(subject),
        })
    registry = authority_root.load_default_registry()
    return _stamp({
        "schema": AUTHORITY_REQUIREMENTS_SCHEMA,
        "receipt_schema": AUTHORITY_SCHEMA,
        "envelope_schema": ENVELOPE_SCHEMA,
        "signature": {
            "tool": str(SSH_KEYGEN), "format": "SSHSIG",
            "namespace": SSHSIG_NAMESPACE,
            "message": "canonical JSON bytes of the self-hashed receipt",
            "allowed_signers_and_signature_are_file-identity-bound": True,
        },
        "independently_pinned_registry_sha256": registry["registry_sha256"],
        "independently_pinned_allowed_signers_sha256": registry["allowed_signers_sha256"],
        "receipt_fields": [
            "schema", "receipt_kind", "subject", "host_hardware_uuid_sha256",
            "binary", "command_abi_sha256", "claims", "issued_at_unix_ns",
            "expires_at_unix_ns", "release_build_sha256", "receipt_sha256",
        ],
        "envelope_fields": [
            "schema", "receipt", "signer_identity", "signature_namespace",
            "allowed_signers", "detached_signature", "envelope_sha256",
        ],
        "authorities": rows,
        "default_off": True,
        "grants_execution_by_itself": False,
    }, "requirements_sha256")


def status(
    *, euid: int | None = None, final_ready: bool | None = None,
    heavy_owner_count: int | None = None, inherited_lease_fds: Sequence[int] | None = None,
    process_joule_present: bool | None = None, full_xcode_present: bool | None = None,
) -> dict[str, Any]:
    """Cheap status only: no capability command, trace, model, corpus, or GPU read."""
    if final_ready is None:
        try:
            observer = _load_json(OBSERVER)
        except (OSError, EvidenceError):
            observer = None
        final_ready = bool(isinstance(observer, dict) and observer.get("final_interpretation_ready") is True)
    if heavy_owner_count is None:
        heavy_owner_count = len(spec_reentry_scaffold.active_heavy_owners())
    if inherited_lease_fds is None:
        inherited_lease_fds = ram_scheduler.inherited_lease_fds()
    euid = os.geteuid() if euid is None else euid
    process_status = process_joule.status()
    pj_present = (
        process_status["direct_process_nanojoule_backend_available"] is True
        if process_joule_present is None else process_joule_present
    )
    xc_present = FULL_XCODE_XCTRACE.is_file() if full_xcode_present is None else full_xcode_present
    adapter_status = xctrace_adapter.status()
    blockers: list[str] = []
    if not final_ready:
        blockers.append("Doctor final_interpretation_ready is false")
    if heavy_owner_count:
        blockers.append(f"{heavy_owner_count} heavy owner(s) remain")
    if len(tuple(inherited_lease_fds)) != 1:
        blockers.append("one inherited shared-heavy-lease descriptor is absent")
    if not pj_present:
        blockers.append("libproc RUSAGE_INFO_V6 direct-process-joule backend is absent")
    if not xc_present:
        blockers.append("full-Xcode xctrace is absent")
    blockers.extend(
        f"xctrace export adapter: {row}" for row in adapter_status["blockers"]
    )
    blockers.append("a fully signed, hash-bound execution request has not been supplied")
    try:
        registry = authority_root.load_default_registry()
        registry_green = True
    except (OSError, authority_root.AuthorityError):
        registry = {}
        registry_green = False
        blockers.append("source-sealed authority registry is invalid")
    capability = execution_capability_contract() if registry_green else {"capability_sha256": None}
    return _stamp({
        "schema": STATUS_SCHEMA,
        "default_off": True,
        "execution_ready": False,
        "collection_started": False,
        "final_interpretation_ready": final_ready,
        "active_heavy_owner_count": heavy_owner_count,
        "inherited_shared_heavy_lease_count": len(tuple(inherited_lease_fds)),
        "process_joule_backend_present": pj_present,
        "process_joule_backend": process_joule.BACKEND_ID,
        "powermetrics_energy_impact_eligible": False,
        "full_xcode_xctrace_present": xc_present,
        "xctrace_export_adapter_contract_sha256": xctrace_adapter.CONTRACT_SHA256,
        "xctrace_production_export_ready": adapter_status["production_export_ready"],
        "execute_surface_exposed": True,
        "execution_capability": "release-gated-only",
        "execution_capability_contract_sha256": capability["capability_sha256"],
        "collector_normalizer_default_off_preserved": True,
        "authority_registry_valid": registry_green,
        "authority_registry_sha256": registry.get("registry_sha256"),
        "exact_command_abi_sha256": ABI_HASHES,
        "blockers": blockers,
    }, "status_sha256")


def dry_run(kind: str) -> dict[str, Any]:
    if kind not in {"device", "spec"}:
        raise ValueError("kind must be device or spec")
    capability = execution_capability_contract()
    return _stamp({
        "schema": DRY_RUN_SCHEMA,
        "kind": kind,
        "would_execute": False,
        "would_start_collectors": False,
        "would_run_probe": False,
        "would_mutate_runtime_default": False,
        "would_open_heavy_lease": False,
        "requires_inherited_heavy_lease": True,
        "execution_capability_contract_sha256": capability["capability_sha256"],
        "probe_barrier": "pid-preserving-python-child-then-execve",
        "collector_start": "xctrace Popen/readiness precedes probe barrier; energy self-samples in probe",
        "exact_command_abi_sha256": ABI_HASHES,
        "blockers": [
            "execute requires an immutable request and explicit --acknowledge-request-sha256",
            "all SSHSIG capability/privilege/attribution envelopes must verify",
            "full-Xcode xctrace and signed libproc/probe self-sampling authority are release-time requirements",
        ],
    }, "dry_run_sha256")


def _validate_request_structure(
    request: Any, *, require_unused_output: bool = True,
) -> list[str]:
    expected = {
        "schema", "kind", "release", "workload", "authorities", "output_directory",
        "scratch_reserve_gb", "additional_parent_receipt_sha256", "execution_requested",
        "runtime_default_mutation_requested", "request_sha256",
    }
    if not isinstance(request, dict) or set(request) != expected:
        return ["execution request fields are incomplete or unexpected"]
    errors = _hash_errors(request, "request_sha256", label="execution_request")
    if request.get("schema") != REQUEST_SCHEMA or request.get("kind") not in {"device", "spec"}:
        errors.append("execution request schema/kind is invalid")
    if request.get("execution_requested") is not True:
        errors.append("execution request does not explicitly request one physical run")
    if request.get("runtime_default_mutation_requested") is not False:
        errors.append("execution request asks to mutate a runtime default")
    scratch = request.get("scratch_reserve_gb")
    if isinstance(scratch, bool) or not isinstance(scratch, (int, float)) or not 1 <= scratch <= 50:
        errors.append("execution scratch reserve must be within 1..50 GB")
    parents = request.get("additional_parent_receipt_sha256")
    if not isinstance(parents, list) or len(set(parents)) != len(parents) or any(
        not isinstance(value, str) or HEX64.fullmatch(value) is None for value in parents
    ):
        errors.append("additional parent receipt hashes are invalid or duplicated")
    output = request.get("output_directory")
    if not isinstance(output, str) or not pathlib.Path(output).is_absolute():
        errors.append("output_directory must be absolute")
    else:
        path = pathlib.Path(output)
        if path.parent != REPORT_ROOT or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", path.name,
        ) is None:
            errors.append("output_directory escapes the unbound physical-release tree")
        if require_unused_output and path.exists():
            errors.append("output_directory must be unused")
    return errors


def _release_source_commit(release_build: Mapping[str, Any]) -> str:
    value = release_build.get("source_base_commit", release_build.get("source_commit"))
    return value if isinstance(value, str) else ""


def _release_errors(release: Any, *, observer: Any, verify_files: bool) -> list[str]:
    expected = {
        "boundary_attestation", "boundary_observation", "corpus_index",
        "corpus_verification", "corpus_prebuild_verification_receipt",
        "corpus_verification_receipt", "source_manifest", "release_build",
        "authority_registry",
    }
    if not isinstance(release, dict) or set(release) != expected:
        return ["release parent set is incomplete or unexpected"]
    boundary = release["boundary_attestation"]
    observation = release["boundary_observation"]
    corpus = release["corpus_index"]
    source = release["source_manifest"]
    build = release["release_build"]
    errors = release_packet.validate_release_boundary_attestation(
        boundary, observation=observation, observer=observer,
    )
    errors.extend(release_packet.validate_corpus_verification(
        release["corpus_verification"], receipt=release["corpus_verification_receipt"],
        index=corpus, boundary_attestation=boundary, boundary_observation=observation,
        parent_verification_receipt=release[
            "corpus_prebuild_verification_receipt"
        ],
    ))
    errors.extend(release_packet.validate_clean_source_manifest(source, verify_current=verify_files))
    errors.extend(release_packet.validate_release_build_receipt(
        build, source_manifest=source, release_boundary=boundary,
        verify_current=verify_files,
    ))
    errors.extend(authority_root.validate_registry(
        release["authority_registry"], verify_files=verify_files, require_default=True,
    ))
    return errors


def _live_release_cas(
    request: Mapping[str, Any], *, observer: Mapping[str, Any], phase: str,
    checked_at_unix_ns: int | None = None,
) -> dict[str, Any]:
    """Rehash every Doctor reference and every frozen corpus byte under lease."""
    if phase not in {"prelaunch", "final"}:
        raise ValueError("release CAS phase must be prelaunch or final")
    release = request["release"]
    errors = _release_errors(release, observer=observer, verify_files=True)
    reference_errors, verified = release_packet.verify_final_packet_references(observer)
    errors.extend(f"final reference: {error}" for error in reference_errors)
    observation = release["boundary_observation"]
    if isinstance(verified, dict):
        comparisons = {
            "final_packet_file_sha256": verified.get("final_packet_file_sha256"),
            "final_packet_canonical_sha256": verified.get("final_packet", {}).get("packet_sha256"),
            "verified_reference_count": verified.get("verified_reference_count"),
            "verified_references_sha256": verified.get("verified_references_sha256"),
        }
        for field, current in comparisons.items():
            if observation.get(field) != current:
                errors.append(f"release boundary {field} differs from complete live final reference set")
    checked = time.time_ns() if checked_at_unix_ns is None else checked_at_unix_ns
    try:
        corpus_receipt, corpus_attestation = release_packet.build_corpus_verification(
            release["corpus_index"],
            boundary_attestation=release["boundary_attestation"],
            boundary_observation=observation,
            verified_at_unix_ns=checked,
            verification_phase="post_release_build",
            parent_verification_receipt=release["corpus_prebuild_verification_receipt"],
        )
    except (OSError, release_packet.EvidenceError, ValueError) as exc:
        errors.append(f"complete live corpus verification failed: {exc}")
        corpus_receipt = {}
        corpus_attestation = {}
    try:
        observer_after = _load_json(OBSERVER)
    except (OSError, EvidenceError) as exc:
        errors.append(f"Doctor observer cannot be reread after corpus CAS: {exc}")
    else:
        if observer_after != observer:
            errors.append("Doctor observer changed while final references/corpus were revalidated")
        final_reference_errors, final_verified = release_packet.verify_final_packet_references(
            observer_after,
        )
        errors.extend(f"final reference post-corpus: {error}" for error in final_reference_errors)
        if isinstance(verified, dict) and final_verified != verified:
            errors.append("complete Doctor final reference set changed during corpus CAS")
    if spec_reentry_scaffold.active_heavy_owners():
        errors.append("heavy owners appeared during complete release CAS")
    if errors:
        raise AdmissionBlocked(f"{phase} release CAS failed: " + "; ".join(errors))
    assert isinstance(verified, dict)
    return _stamp({
        "schema": "hawking.appendix_counter_live_release_cas.v1",
        "phase": phase,
        "checked_at_unix_ns": checked,
        "observer_state_sha256": observer["state_sha256"],
        "final_packet_file_sha256": verified["final_packet_file_sha256"],
        "final_packet_canonical_sha256": verified["final_packet"]["packet_sha256"],
        "verified_reference_count": verified["verified_reference_count"],
        "verified_references_sha256": verified["verified_references_sha256"],
        "corpus_index_sha256": release["corpus_index"]["index_sha256"],
        "live_corpus_verification_receipt_sha256": corpus_receipt[
            "verification_receipt_sha256"
        ],
        "live_corpus_attestation_sha256": corpus_attestation["attestation_sha256"],
        "release_boundary_attestation_sha256": release["boundary_attestation"][
            "attestation_sha256"
        ],
        "exclusive_inherited_lease_proven": True,
        "active_heavy_owner_count": 0,
    }, "cas_sha256")


def _corpus_file_errors(binding: Any, *, corpus_index: Mapping[str, Any], label: str,
                        verify_file: bool) -> list[str]:
    errors = _identity_errors(binding, label=label, verify_file=verify_file)
    if not isinstance(binding, dict):
        return errors
    entries = corpus_index.get("entries")
    root_raw = corpus_index.get("root")
    if not isinstance(entries, list) or not isinstance(root_raw, str):
        return [*errors, f"{label} cannot be resolved in the corpus index"]
    try:
        path = pathlib.Path(binding["path"]).resolve()
        root = pathlib.Path(root_raw).resolve()
        relative = path.relative_to(root).as_posix()
    except (KeyError, OSError, ValueError):
        errors.append(f"{label} path is outside the frozen corpus root")
        return errors
    matches = [
        row for row in entries if isinstance(row, dict)
        and row.get("path") == relative and row.get("sha256") == binding.get("sha256")
        and row.get("size") == binding.get("size_bytes")
    ]
    if len(matches) != 1:
        errors.append(f"{label} is not one exact frozen corpus entry")
    return errors


def _workload_errors(request: Mapping[str, Any], *, verify_files: bool) -> list[str]:
    kind = request["kind"]
    workload = request.get("workload")
    corpus = request["release"]["corpus_index"]
    build = request["release"]["release_build"]
    if not isinstance(workload, dict):
        return ["workload must be an object"]
    expected_common = {"runtime_path", "probe", "label"}
    if kind == "device":
        expected = expected_common | {
            "cell_id", "artifact", "tensor", "residual_artifact", "residual_tensor",
            "warmups", "trials",
        }
    else:
        expected = expected_common | {
            "weights", "artifact", "prompts", "generated_tokens",
            "warmups_per_batch", "repeats_per_batch",
        }
    errors: list[str] = []
    if set(workload) != expected:
        errors.append("workload fields are incomplete or unexpected")
        return errors
    runtime = workload.get("runtime_path")
    if runtime not in collector.RUNTIME_PATHS:
        errors.append("workload runtime_path is invalid")
    label = workload.get("label")
    if not isinstance(label, str) or not label:
        errors.append("workload label is empty")
    expected_probe = build.get("probes", {}).get(kind)
    if workload.get("probe") != expected_probe:
        errors.append("workload probe differs from exact release-build binding")
    errors.extend(_identity_errors(workload.get("probe"), label="workload probe", verify_file=verify_files))
    if kind == "device":
        matrix = tq_runtime_matrix.build_matrix()
        cell = next((row for row in matrix.get("cells", []) if row.get("id") == workload.get("cell_id")), None)
        if not isinstance(cell, dict) or cell.get("state") != "deferred" \
                or cell.get("runtime_path") != runtime or cell.get("tensor_family") != workload.get("tensor"):
            errors.append("device workload does not identify one exact deferred matrix cell")
        for field in ("warmups", "trials"):
            value = workload.get(field)
            minimum = 3 if field == "warmups" else 10
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                errors.append(f"device workload {field} is below its exact minimum")
        errors.extend(_corpus_file_errors(
            workload.get("artifact"), corpus_index=corpus, label="device artifact",
            verify_file=verify_files,
        ))
        residual, residual_tensor = workload.get("residual_artifact"), workload.get("residual_tensor")
        if (residual is None) != (residual_tensor is None):
            errors.append("residual artifact and tensor must be supplied together")
        elif residual is not None:
            errors.extend(_corpus_file_errors(
                residual, corpus_index=corpus, label="device residual artifact",
                verify_file=verify_files,
            ))
            if residual == workload.get("artifact"):
                errors.append("residual artifact must be distinct from the base artifact")
    else:
        for field in ("weights", "artifact", "prompts"):
            errors.extend(_corpus_file_errors(
                workload.get(field), corpus_index=corpus, label=f"spec {field}",
                verify_file=verify_files,
            ))
        for field, minimum in (
            ("generated_tokens", 1), ("warmups_per_batch", 3), ("repeats_per_batch", 5),
        ):
            value = workload.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                errors.append(f"spec workload {field} is below its exact minimum")
        try:
            spec_tq_runner._cells(runtime, label)
        except (KeyError, StopIteration, TypeError, ValueError):
            errors.append("spec workload does not resolve exact parity/curve matrix cells")
    return errors


def _thermal_green() -> bool:
    try:
        process = subprocess.run(
            ["/usr/bin/pmset", "-g", "therm"], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
            check=False, shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return ram_scheduler.thermal_output_ok(
        process.returncode, (process.stdout or process.stderr).strip(),
    )


def _resource_errors(resource: Any, *, thermal_green: bool, scratch_reserve_gb: float) -> list[str]:
    if not isinstance(resource, dict):
        return ["resource snapshot is absent"]
    errors: list[str] = []
    if resource.get("ok") is not True or ram_scheduler.classify_resource_state(resource) != "green":
        errors.append("RAM/swap guard is not green")
    free, reserve = resource.get("disk_free_gb"), resource.get("disk_reserve_gb")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (free, reserve)) \
            or float(free) < float(reserve) + float(scratch_reserve_gb):
        errors.append("disk free space does not preserve reserve plus bounded scratch")
    if not thermal_green:
        errors.append("thermal admission is not explicitly green")
    return errors


def competing_exclusive_lock_is_held(lock_path: pathlib.Path = HEAVY_LOCK) -> tuple[bool, str]:
    """Attempt an independent nonblocking flock on a separately opened FD."""
    try:
        competitor = lock_path.open("a+")
    except OSError as exc:
        return False, f"competing heavy-lock descriptor cannot be opened: {exc}"
    try:
        try:
            fcntl.flock(competitor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True, "independent exclusive flock attempt was blocked"
        else:
            fcntl.flock(competitor.fileno(), fcntl.LOCK_UN)
            return False, "independent descriptor acquired the heavy lock"
    finally:
        competitor.close()


def _lease_proof_child(fd_text: str) -> int:
    """Child half of the exact-open-file-description flock proof."""
    try:
        fd = int(fd_text)
        os.fstat(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return EXIT_BLOCKED
    except (OSError, TypeError, ValueError):
        return 64
    return 0


def inherited_fd_owns_exclusive_flock(
    fd: int, *, lock_path: pathlib.Path = HEAVY_LOCK,
    runner: Callable[..., Any] = subprocess.run,
) -> tuple[bool, str]:
    """Prove the exact inherited open-file description owns the lock.

    A separately opened competitor must be blocked both before and after a
    fresh child process re-locks *only the inherited FD*.  An unrelated lock
    holder therefore cannot make an unlocked inherited descriptor look valid.
    """
    before, detail = competing_exclusive_lock_is_held(lock_path)
    if not before:
        return False, "pre-proof competitor was not blocked: " + detail
    try:
        process = runner(
            [
                sys.executable, str(pathlib.Path(__file__).resolve()),
                "--lease-proof-child", str(fd),
            ],
            cwd=ROOT, env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
            pass_fds=(fd,), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=10, check=False, shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"inherited-FD child proof failed: {exc}"
    if process.returncode != 0:
        return False, f"inherited-FD child could not re-lock exact descriptor ({process.returncode})"
    after, detail = competing_exclusive_lock_is_held(lock_path)
    if not after:
        return False, "post-proof competitor was not blocked: " + detail
    return True, "exact inherited open-file description re-locked in a child process"


def _validate_inherited_lease(
    receipt: Mapping[str, Any], *, env: Mapping[str, str],
    expected_observer_sha256: str, expected_boundary_sha256: str,
    lock_path: pathlib.Path = HEAVY_LOCK, expected_parent_pid: int | None = None,
) -> tuple[int | None, list[str]]:
    fds = ram_scheduler.inherited_lease_fds(env)
    if len(fds) != 1:
        return None, ["exactly one inherited shared-heavy-lease descriptor is required"]
    fd = fds[0]
    errors: list[str] = []
    try:
        inherited = os.fstat(fd)
        canonical = lock_path.stat()
    except OSError as exc:
        return None, [f"inherited shared-heavy lease cannot be inspected: {exc}"]
    if (inherited.st_dev, inherited.st_ino) != (canonical.st_dev, canonical.st_ino):
        errors.append("inherited descriptor is not the canonical heavy lock")
    locked, lock_detail = inherited_fd_owns_exclusive_flock(fd, lock_path=lock_path)
    if not locked:
        errors.append("inherited descriptor does not own an exclusive flock: " + lock_detail)
    claims = receipt.get("claims", [])
    parsed: dict[str, list[str]] = {}
    for claim in claims if isinstance(claims, list) else []:
        if isinstance(claim, str) and "=" in claim:
            key, value = claim.split("=", 1)
            parsed.setdefault(key, []).append(value)
    expected = {
        "lock_device": str(inherited.st_dev),
        "lock_inode": str(inherited.st_ino),
        "parent_pid": str(os.getppid() if expected_parent_pid is None else expected_parent_pid),
        "observer_state_sha256": expected_observer_sha256,
        "release_boundary_attestation_sha256": expected_boundary_sha256,
    }
    for key, value in expected.items():
        if parsed.get(key) != [value]:
            errors.append(f"inherited lease signed {key} does not match live execution")
    return fd, errors


@contextlib.contextmanager
def held_heavy_lease(
    lock_path: pathlib.Path = HEAVY_LOCK,
) -> Iterable[int]:
    """Production lease-holder primitive used by release orchestration."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        lock_path.absolute(),
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AdmissionBlocked("canonical heavy lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AdmissionBlocked("canonical shared-heavy lease is already held") from exc
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def lease_holder_execute(
    request_path: pathlib.Path, *, acknowledgement: str,
    runner: Callable[..., Any] = subprocess.run, lock_path: pathlib.Path = HEAVY_LOCK,
) -> int:
    """Hold the canonical flock in a parent while executing the exact child.

    The request's signed ``parent_pid`` must equal this lease-holder PID.  An
    in-process release orchestrator can enter :func:`held_heavy_lease`, draft
    and sign that dynamic receipt, then use the same child argv.  This CLI
    wrapper is the production non-shell transport for an already prepared
    request.
    """
    request = _load_json(request_path)
    errors = _validate_request_structure(request)
    if errors:
        raise AdmissionBlocked("invalid lease-holder request: " + "; ".join(errors))
    if acknowledgement != request.get("request_sha256"):
        raise AdmissionBlocked("lease-holder request acknowledgement is absent or wrong")
    with held_heavy_lease(lock_path) as lease_fd:
        env = dict(os.environ)
        env[ram_scheduler.HEAVY_LEASE_FD_ENV] = str(lease_fd)
        receipt = request.get("authorities", {}).get("inherited_lease", {}).get("receipt", {})
        observer = _load_json(OBSERVER)
        _fd, lease_errors = _validate_inherited_lease(
            receipt, env=env,
            expected_observer_sha256=observer.get("state_sha256", ""),
            expected_boundary_sha256=request.get("release", {}).get(
                "boundary_attestation", {},
            ).get("attestation_sha256", ""),
            lock_path=lock_path, expected_parent_pid=os.getpid(),
        )
        if lease_errors:
            raise AdmissionBlocked("lease-holder proof failed: " + "; ".join(lease_errors))
        process = runner(
            [
                sys.executable, str(pathlib.Path(__file__).resolve()),
                "--execute", str(request_path),
                "--acknowledge-request-sha256", acknowledgement,
            ],
            cwd=ROOT, env=env, pass_fds=(lease_fd,), stdin=subprocess.DEVNULL,
            stdout=None, stderr=None, check=False, shell=False,
        )
        return int(process.returncode)


def _probe_argv(
    request: Mapping[str, Any], raw_path: pathlib.Path,
    process_joule_provenance_path: pathlib.Path,
) -> list[str]:
    workload = request["workload"]
    commit = _release_source_commit(request["release"]["release_build"])
    probe = workload["probe"]["path"]
    if request["kind"] == "device":
        cell = next(row for row in tq_runtime_matrix.build_matrix()["cells"] if row["id"] == workload["cell_id"])
        argv = [
            probe, "--artifact", workload["artifact"]["path"], "--runtime-path", workload["runtime_path"],
            "--warmups", str(workload["warmups"]), "--trials", str(workload["trials"]),
            "--matrix-cell-id", cell["id"], "--matrix-model", cell["model"],
            "--matrix-cell-sha256", canonical_sha256(cell), "--matrix-tensor-family", cell["tensor_family"],
            "--source-commit", commit,
            "--process-joule-provenance", str(process_joule_provenance_path),
            "--output", str(raw_path), "--tensor", cell["tensor_family"],
        ]
        if workload["residual_artifact"] is not None:
            argv.extend([
                "--residual-artifact", workload["residual_artifact"]["path"],
                "--residual-tensor", workload["residual_tensor"],
            ])
        return argv
    parity, curve, _known = spec_tq_runner._cells(workload["runtime_path"], workload["label"])
    return [
        probe, "--weights", workload["weights"]["path"], "--artifact", workload["artifact"]["path"],
        "--prompts", workload["prompts"]["path"], "--runtime-path", workload["runtime_path"],
        "--generated-tokens", str(workload["generated_tokens"]), "--source-commit", commit,
        "--process-joule-provenance", str(process_joule_provenance_path),
        "--warmups-per-batch", str(workload["warmups_per_batch"]),
        "--repeats-per-batch", str(workload["repeats_per_batch"]),
        "--parity-cell-id", parity["id"], "--curve-cell-id", curve["id"],
        "--output", str(raw_path),
    ]


def _collector_argv(name: str, *, probe_pid: int, raw_path: pathlib.Path) -> list[str]:
    if name == "xctrace":
        return [
            str(FULL_XCODE_XCTRACE), "record", "--template", "Metal System Trace",
            "--attach", str(probe_pid), "--output", str(raw_path),
        ]
    raise ValueError(name)


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[bytes]
    stdout_handle: Any
    stderr_handle: Any

    @property
    def pid(self) -> int:
        return self.process.pid


class ProcessBackend:
    """Small injectable process surface; every production launch uses shell=False."""

    def spawn(
        self, name: str, argv: Sequence[str], *, env: Mapping[str, str],
        pass_fds: Sequence[int], stdout_path: pathlib.Path, stderr_path: pathlib.Path,
    ) -> ManagedProcess:
        stdout_handle = stdout_path.open("xb")
        stderr_handle = stderr_path.open("xb")
        try:
            process = subprocess.Popen(
                list(argv), cwd=ROOT, env=dict(env), pass_fds=tuple(pass_fds),
                stdin=subprocess.DEVNULL, stdout=stdout_handle, stderr=stderr_handle,
                shell=False, close_fds=True,
            )
        except BaseException:
            stdout_handle.close()
            stderr_handle.close()
            raise
        return ManagedProcess(name, process, stdout_handle, stderr_handle)

    def alive(self, process: ManagedProcess) -> bool:
        return process.process.poll() is None

    def wait(self, process: ManagedProcess, timeout: float) -> int:
        try:
            return process.process.wait(timeout=timeout)
        finally:
            process.stdout_handle.close()
            process.stderr_handle.close()

    def interrupt(self, process: ManagedProcess) -> None:
        if process.process.poll() is None:
            process.process.send_signal(signal.SIGINT)

    def terminate(self, process: ManagedProcess) -> None:
        if process.process.poll() is None:
            process.process.terminate()

    def kill(self, process: ManagedProcess) -> None:
        if process.process.poll() is None:
            process.process.kill()


@dataclass
class OrchestrationResult:
    probe_pid: int
    probe_argv: list[str]
    probe_started_at_unix_ns: int
    probe_started_at_continuous_ns: int
    probe_ended_at_unix_ns: int
    probe_ended_at_continuous_ns: int
    collector_argv: dict[str, list[str]]
    collector_returncodes: dict[str, int]
    readiness: dict[str, dict[str, int]]
    probe_returncode: int


def _capture_has_bytes(path: pathlib.Path) -> bool:
    try:
        if path.is_symlink():
            return False
        if path.is_file():
            return path.stat().st_size > 0
        if path.is_dir():
            return any(
                candidate.is_file() and not candidate.is_symlink() and candidate.stat().st_size > 0
                for candidate in path.rglob("*")
            )
    except OSError:
        return False
    return False


def _await_ready(
    backend: ProcessBackend, process: ManagedProcess, path: pathlib.Path,
    *, timeout_seconds: float = 30.0,
) -> dict[str, int]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not backend.alive(process):
            raise AdmissionBlocked(f"{process.name} exited before capture readiness")
        if _capture_has_bytes(path):
            return {
                "ready_at_unix_ns": time.time_ns(),
                "ready_at_continuous_ns": time.monotonic_ns(),
            }
        time.sleep(0.05)
    raise AdmissionBlocked(f"{process.name} did not prove an online capture stream")


def _stop_process(backend: ProcessBackend, process: ManagedProcess) -> int:
    backend.interrupt(process)
    try:
        return backend.wait(process, 30)
    except subprocess.TimeoutExpired:
        backend.terminate(process)
        try:
            backend.wait(process, 10)
        except subprocess.TimeoutExpired:
            backend.kill(process)
            backend.wait(process, 5)
        raise AdmissionBlocked(f"{process.name} did not stop cleanly after coverage")


def orchestrate_capture(
    *, backend: ProcessBackend, probe_argv: Sequence[str], probe_env: Mapping[str, str],
    lease_fd: int, output_paths: Mapping[str, pathlib.Path],
    output_dir_fd: int | None = None,
    probe_timeout_seconds: float, readiness_wait: Callable[..., dict[str, int]] = _await_ready,
) -> OrchestrationResult:
    """Run exactly one barriered probe with two already-live collectors.

    Both collector ``spawn`` calls occur before either readiness call.  Tests
    assert this ordering directly; production additionally requires non-empty
    raw output and a live process before the barrier byte is written.
    """
    read_fd, write_fd = os.pipe()
    probe: ManagedProcess | None = None
    collectors: dict[str, ManagedProcess] = {}
    barrier_released = False
    try:
        child_argv = [
            sys.executable, str(pathlib.Path(__file__).resolve()),
            "--barrier-child", str(read_fd), "--", *probe_argv,
        ]
        output_fds = (() if output_dir_fd is None else (output_dir_fd,))
        probe = backend.spawn(
            "probe", child_argv, env=probe_env,
            pass_fds=(lease_fd, *output_fds, read_fd),
            stdout_path=output_paths["probe_stdout"], stderr_path=output_paths["probe_stderr"],
        )
        os.close(read_fd)
        read_fd = -1
        collector_argv = {
            "xctrace": _collector_argv(
                "xctrace", probe_pid=probe.pid, raw_path=output_paths["xctrace_raw"],
            ),
        }
        # Energy snapshots bracket operation-only probe intervals; xctrace must
        # be live before the probe barrier is released.
        for name in ("xctrace",):
            collectors[name] = backend.spawn(
                name, collector_argv[name], env=probe_env,
                pass_fds=(lease_fd, *output_fds),
                stdout_path=output_paths[f"{name}_stdout"],
                stderr_path=output_paths[f"{name}_stderr"],
            )
        readiness = {
            "xctrace": readiness_wait(
                backend, collectors["xctrace"], output_paths["xctrace_raw"],
            ),
        }
        if any(not backend.alive(collectors[name]) for name in collectors):
            raise AdmissionBlocked("a collector died at the probe barrier")
        started_unix = time.time_ns()
        started_continuous = time.monotonic_ns()
        if os.write(write_fd, b"G") != 1:
            raise AdmissionBlocked("probe barrier release was incomplete")
        barrier_released = True
        os.close(write_fd)
        write_fd = -1
        probe_rc = backend.wait(probe, probe_timeout_seconds)
        ended_continuous = time.monotonic_ns()
        ended_unix = time.time_ns()
        if any(not backend.alive(collectors[name]) for name in collectors):
            raise AdmissionBlocked("a collector ended before probe coverage completed")
        collector_rc = {name: _stop_process(backend, process) for name, process in collectors.items()}
        return OrchestrationResult(
            probe_pid=probe.pid, probe_argv=list(probe_argv),
            probe_started_at_unix_ns=started_unix,
            probe_started_at_continuous_ns=started_continuous,
            probe_ended_at_unix_ns=ended_unix,
            probe_ended_at_continuous_ns=ended_continuous,
            collector_argv=collector_argv, collector_returncodes=collector_rc,
            readiness=readiness, probe_returncode=probe_rc,
        )
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
        if not barrier_released and probe is not None:
            backend.terminate(probe)
            try:
                backend.wait(probe, 5)
            except subprocess.TimeoutExpired:
                backend.kill(probe)
                backend.wait(probe, 5)
        for process in collectors.values():
            if backend.alive(process):
                try:
                    _stop_process(backend, process)
                except (AdmissionBlocked, subprocess.TimeoutExpired):
                    pass


def _copy_regular_capture(source: pathlib.Path, destination: pathlib.Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise EvidenceError(f"collector capture is not a regular file: {source}")
    before = source.stat(follow_symlinks=False)
    if before.st_nlink != 1:
        raise EvidenceError("collector capture must be a single-link file")
    if before.st_size <= 0 or before.st_size > MAX_CAPTURE_BYTES:
        raise EvidenceError("collector capture is empty or exceeds the bounded attestation limit")
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    after = source.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
    ):
        raise EvidenceError("collector capture changed while sealing")


def seal_capture(source: pathlib.Path, destination: pathlib.Path) -> dict[str, Any]:
    """Freeze one regular capture; trace packages use the signed adapter path."""
    if destination.exists():
        raise EvidenceError(f"sealed capture destination already exists: {destination}")
    if not source.is_file():
        raise EvidenceError(f"collector did not create a sealable capture: {source}")
    _copy_regular_capture(source, destination)
    destination.chmod(0o444)
    return physical_counter_attestation.file_identity(destination)


def freeze_trace_tree(source: pathlib.Path) -> dict[str, Any]:
    """Recursively freeze one safe raw .trace tree without archiving it."""
    try:
        before = xctrace_adapter.trace_tree_identity(source, require_immutable=False)
        relative_files = [row["relative_path"] for row in before["files"]]
        directories = sorted(
            (candidate for candidate in source.rglob("*") if candidate.is_dir()),
            key=lambda candidate: len(candidate.relative_to(source).parts),
            reverse=True,
        )
        for relative in relative_files:
            candidate = source / relative
            if candidate.is_symlink():
                raise EvidenceError("xctrace package changed to a symlink while freezing")
            candidate.chmod(0o444, follow_symlinks=False)
        for directory in directories:
            if directory.is_symlink():
                raise EvidenceError("xctrace package changed to a symlink while freezing")
            directory.chmod(0o555, follow_symlinks=False)
        source.chmod(0o555, follow_symlinks=False)
        after = xctrace_adapter.trace_tree_identity(source, require_immutable=True)
    except (OSError, ValueError, xctrace_adapter.XctraceAdapterError) as exc:
        if isinstance(exc, EvidenceError):
            raise
        raise EvidenceError(f"xctrace package cannot be frozen safely: {exc}") from exc
    if before != after:
        raise EvidenceError("xctrace package changed while recursively freezing")
    return after


def _sha_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bind_release_parents(
    receipt: dict[str, Any], *, release: Mapping[str, Any], extra: Sequence[str],
    parity_parent: str | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(receipt)
    bindings = result["bindings"]
    corpus_sha = release["corpus_index"]["index_sha256"]
    build_sha = release["release_build"]["receipt_sha256"]
    source_sha = release["source_manifest"]["manifest_sha256"]
    parents = [corpus_sha, build_sha, *extra]
    if parity_parent is not None:
        parents.append(parity_parent)
    bindings["parent_receipt_sha256"] = list(dict.fromkeys(parents))
    bindings["corpus_index_sha256"] = corpus_sha
    bindings["release_build_sha256"] = build_sha
    bindings["source_manifest_sha256"] = source_sha
    return appendix_contract.stamp_receipt(result)


def _normalizer_argv(
    binary: str, *, kind: str, bundle: pathlib.Path, process_joule_capture: pathlib.Path,
    xctrace: pathlib.Path, probe_pid: int, run_nonce: str,
    metal_registry_id: str, output: pathlib.Path,
) -> list[str]:
    return [
        binary, "--kind", kind, "--raw-bundle", str(bundle),
        "--process-joule", str(process_joule_capture), "--xctrace", str(xctrace),
        "--probe-pid", str(probe_pid), "--run-nonce", run_nonce,
        "--metal-registry-id", metal_registry_id,
        "--output", str(output),
    ]


def _counter_attestation(
    *, request: Mapping[str, Any], bundle: Mapping[str, Any],
    payload: Mapping[str, Any], attributed: Mapping[str, Any], receipts: Mapping[str, Mapping[str, Any]],
    normalized_wrapper_identity: Mapping[str, Any], sealed: Mapping[str, Mapping[str, Any]],
    normalizer_argv: Sequence[str], collector_argv: Mapping[str, Sequence[str]],
    request_file_identity: Mapping[str, Any],
) -> dict[str, Any]:
    authority = bundle["execution_authority"]
    required = collector.DEVICE_DOMAINS if request["kind"] == "device" else collector.SPEC_DOMAINS
    by_collector = {row["id"]: row for row in attributed["collectors"]}
    capture_start_u = min(row["capture_started_at_unix_ns"] for row in by_collector.values())
    capture_end_u = max(row["capture_ended_at_unix_ns"] for row in by_collector.values())
    capture_start_c = min(row["capture_started_at_continuous_ns"] for row in by_collector.values())
    capture_end_c = max(row["capture_ended_at_continuous_ns"] for row in by_collector.values())
    device_receipt = receipts["device_identity"]
    device_claims = {
        claim.split("=", 1)[0]: claim.split("=", 1)[1]
        for claim in device_receipt["claims"] if isinstance(claim, str) and "=" in claim
    }
    required_device_claims = ("registry_id", "name", "architecture", "os_build", "driver_build")
    if any(not device_claims.get(field) for field in required_device_claims):
        raise EvidenceError("signed device identity receipt lacks exact device values")
    domains = []
    for domain in required:
        collector_id = collector.DOMAIN_COLLECTOR[domain]
        source = by_collector[collector_id]
        domains.append({
            "domain": domain,
            "unit": physical_counter_attestation.DOMAIN_UNITS[domain],
            "source_kind": (
                "libproc-rusage-v6-direct-process-joule-self-sampled"
                if collector_id == "process_joule"
                else "xctrace-metal-system-trace-pid-phase-attributed"
            ),
            "collector": receipts[f"{collector_id}_capability"]["binary"],
            "collector_invocation_sha256": canonical_sha256(list(collector_argv[collector_id])),
            "raw_capture": sealed[collector_id],
            "capture_started_at_unix_ns": source["capture_started_at_unix_ns"],
            "capture_ended_at_unix_ns": source["capture_ended_at_unix_ns"],
            "capture_started_at_continuous_ns": source["capture_started_at_continuous_ns"],
            "capture_ended_at_continuous_ns": source["capture_ended_at_continuous_ns"],
            "sample_count": len(attributed["samples"]),
            "estimated": False,
        })
    value = physical_counter_attestation.stamp({
        "schema": physical_counter_attestation.SCHEMA,
        "raw_bundle_sha256": bundle["raw_bundle_sha256"],
        "artifact_sha256": bundle["raw_probe"]["artifact"]["sha256"],
        "execution_authority_sha256": canonical_sha256(authority),
        "counter_payload_sha256": canonical_sha256(payload),
        "capture_window": {
            "capture_started_at_unix_ns": capture_start_u,
            "workload_started_at_unix_ns": authority["started_at_unix_ns"],
            "workload_ended_at_unix_ns": authority["ended_at_unix_ns"],
            "capture_ended_at_unix_ns": capture_end_u,
            "capture_started_at_continuous_ns": capture_start_c,
            "workload_started_at_continuous_ns": authority["started_at_continuous_ns"],
            "workload_ended_at_continuous_ns": authority["ended_at_continuous_ns"],
            "capture_ended_at_continuous_ns": capture_end_c,
            "clock_source": "clock_gettime_wall_continuous_crosscheck",
        },
        "device": {
            "registry_id": device_claims["registry_id"],
            "name": device_claims["name"],
            "architecture": device_claims["architecture"],
            "os_build": device_claims["os_build"],
            "driver_build": device_claims["driver_build"],
            "hardware_uuid_sha256": device_receipt["host_hardware_uuid_sha256"],
            "probe_receipt": dict(request_file_identity),
        },
        "normalizer": {
            "program": receipts["normalizer_capability"]["binary"],
            "invocation_sha256": canonical_sha256(list(normalizer_argv)),
            "source_commit": _release_source_commit(request["release"]["release_build"]),
            "output": dict(normalized_wrapper_identity),
        },
        "domains": domains,
    })
    minimum = len(attributed["samples"])
    errors = physical_counter_attestation.validate(
        value, raw_bundle_sha256=bundle["raw_bundle_sha256"],
        artifact_sha256=bundle["raw_probe"]["artifact"]["sha256"],
        execution_authority=authority, counter_payload=dict(payload),
        required_domains=required, minimum_samples=minimum, verify_files=True,
    )
    if errors:
        raise EvidenceError("counter attestation failed: " + "; ".join(errors))
    return value


def _barrier_child(fd_text: str, argv: Sequence[str]) -> int:
    try:
        fd = int(fd_text)
    except ValueError:
        return 64
    if not argv or argv[0] == "--" or not pathlib.Path(argv[0]).is_absolute():
        return 64
    token = os.read(fd, 1)
    os.close(fd)
    if token != b"G":
        return EXIT_BLOCKED
    os.execve(argv[0], list(argv), dict(os.environ))
    return 70


def execute(
    request_path: pathlib.Path, *, acknowledgement: str,
    backend: ProcessBackend | None = None, euid: int | None = None,
    signature_verifier: SignatureVerifier = _sshsig_verify,
) -> dict[str, Any]:
    """Execute one release cell; all cheap blockers are checked before Popen."""
    request = _load_json(request_path)
    request_file_identity = physical_counter_attestation.file_identity(request_path)
    errors = _validate_request_structure(request)
    if errors:
        raise AdmissionBlocked("; ".join(errors))
    if acknowledgement != request["request_sha256"]:
        raise AdmissionBlocked("explicit request hash acknowledgement is absent or wrong")
    euid = os.geteuid() if euid is None else euid
    # These checks intentionally precede signature verification or any Popen.
    process_status = process_joule.status()
    if process_status["direct_process_nanojoule_backend_available"] is not True:
        raise AdmissionBlocked("libproc RUSAGE_INFO_V6 direct-process-joule backend is absent")
    if not FULL_XCODE_XCTRACE.is_file():
        raise AdmissionBlocked("full-Xcode xctrace is absent")
    try:
        observer = _load_json(OBSERVER)
    except (OSError, EvidenceError) as exc:
        raise AdmissionBlocked(f"Doctor observer cannot be verified: {exc}") from exc
    errors = _release_errors(request["release"], observer=observer, verify_files=True)
    errors.extend(_workload_errors(request, verify_files=True))
    if spec_reentry_scaffold.active_heavy_owners():
        errors.append("heavy owners remain before inherited-lease execution")
    now = time.time_ns()
    live_host_sha256, live_host_detail = authority_root.live_host_hardware_uuid_sha256()
    if live_host_sha256 is None:
        errors.append(live_host_detail)
    authority_errors, receipts = validate_authorities(
        request["authorities"], release_build_sha256=request["release"]["release_build"]["receipt_sha256"],
        now_unix_ns=now, registry=request["release"]["authority_registry"],
        expected_host_hardware_uuid_sha256=live_host_sha256 or "",
        verify_files=True, signature_verifier=signature_verifier,
    )
    errors.extend(authority_errors)
    try:
        process_provenance = process_joule.library_provenance()
    except (OSError, ValueError, process_joule.ProcessJouleError) as exc:
        process_provenance = None
        errors.append(f"direct-process-joule provenance cannot be frozen: {exc}")
    else:
        # Recheck the exact provenance that will be handed to the probe.  This
        # catches a library/SDK/OS change after authority validation but before
        # any output directory or process is created.
        errors.extend(_process_provenance_claim_errors(process_provenance, receipts))
    expected_probe_abi = ABI_HASHES[f"{request['kind']}-probe"]
    device_receipt = receipts.get("device_identity", {})
    device_claims = {
        claim.split("=", 1)[0]: claim.split("=", 1)[1]
        for claim in device_receipt.get("claims", [])
        if isinstance(claim, str) and "=" in claim
    }
    if device_receipt.get("binary") != request.get("workload", {}).get("probe"):
        errors.append("device identity receipt differs from the exact selected release probe")
    if device_receipt.get("command_abi_sha256") != expected_probe_abi:
        errors.append("device identity receipt binds a different probe argv ABI")
    for key in ("process_joule_capability", "process_joule_attribution"):
        if receipts.get(key, {}).get("binary") != request.get("workload", {}).get("probe"):
            errors.append(f"{key} receipt differs from the exact selected self-sampling release probe")
    lease_receipt = receipts.get("inherited_lease", {})
    lease_fd, lease_errors = _validate_inherited_lease(
        lease_receipt, env=os.environ,
        expected_observer_sha256=observer.get("state_sha256", ""),
        expected_boundary_sha256=request["release"]["boundary_attestation"].get(
            "attestation_sha256", "",
        ),
    )
    errors.extend(lease_errors)
    observation = request["release"]["boundary_observation"]
    snapshot = observation.get("owner_snapshot", {}) if isinstance(observation, dict) else {}
    if lease_fd is not None:
        current_stat = os.fstat(lease_fd)
        if (snapshot.get("lock_device"), snapshot.get("lock_inode")) != (current_stat.st_dev, current_stat.st_ino):
            errors.append("inherited lease differs from the exact release-boundary owner snapshot")
    resource_before = ram_scheduler.resource_snapshot(ROOT)
    thermal_before = _thermal_green()
    errors.extend(_resource_errors(
        resource_before, thermal_green=thermal_before,
        scratch_reserve_gb=float(request["scratch_reserve_gb"]),
    ))
    if errors:
        raise AdmissionBlocked("; ".join(errors))
    assert lease_fd is not None
    assert process_provenance is not None
    # Final CAS immediately before any output directory or process exists.
    if spec_reentry_scaffold.active_heavy_owners():
        raise AdmissionBlocked("a heavy owner appeared under the inherited lease")
    current_observer = _load_json(OBSERVER)
    if current_observer.get("state_sha256") != observer.get("state_sha256"):
        raise AdmissionBlocked("Doctor observer changed before collector launch")
    resource_cas = ram_scheduler.resource_snapshot(ROOT)
    if _resource_errors(
        resource_cas, thermal_green=_thermal_green(),
        scratch_reserve_gb=float(request["scratch_reserve_gb"]),
    ):
        raise AdmissionBlocked("resource/thermal CAS failed before collector launch")
    prelaunch_release_cas = _live_release_cas(
        request, observer=observer, phase="prelaunch",
    )

    output_tree = SafeOutputDirectory.create(pathlib.Path(request["output_directory"]))
    names = {
        "raw_probe": "raw_probe.json",
        "raw_bundle": "raw_bundle.json",
        "process_joule_provenance": "process_joule_provenance.json",
        "probe_stdout": "probe.stdout",
        "probe_stderr": "probe.stderr",
        "process_joule_raw": "process_joule.embedded.json",
        "xctrace_raw": "metal.trace",
        "xctrace_stdout": "xctrace.stdout",
        "xctrace_stderr": "xctrace.stderr",
        "process_joule_sealed": "process_joule.sealed.json",
        "xctrace_toc": "xctrace.toc.export",
        "xctrace_capture": "xctrace.canonical.json",
        "xctrace_adapter_receipt": "xctrace.adapter-receipt.json",
        "attributed": "attributed_samples.json",
        "counter_payload": "counter_payload.json",
        "normalized_wrapper": "normalized_counter_binding.json",
        "counter_attestation": "counter_attestation.json",
        "evidence": "evidence.json",
        "execution_receipt": "execution_receipt.json",
        "result_attestation_draft": "result_attestation_draft.json",
        "normalizer_stdout": "normalizer.stdout",
        "normalizer_stderr": "normalizer.stderr",
    }
    paths = {key: output_tree.operational_path(name) for key, name in names.items()}
    run_nonce = secrets.token_hex(32)
    env = dict(os.environ)
    env[ram_scheduler.HEAVY_LEASE_FD_ENV] = str(lease_fd)
    env["HAWKING_PHYSICAL_RUN_NONCE"] = run_nonce
    env["HAWKING_APPENDIX_DEVICE_ADMITTED" if request["kind"] == "device" else "HAWKING_APPENDIX_SPEC_ADMITTED"] = "1"
    _atomic_json(paths["process_joule_provenance"], process_provenance)
    process_provenance_identity = output_tree.published_file_identity(
        names["process_joule_provenance"],
    )
    output_tree.assert_attached()
    if physical_counter_attestation.file_identity(
        paths["process_joule_provenance"]
    ) != process_provenance_identity:
        raise AdmissionBlocked("process-joule provenance changed before probe launch")
    probe_argv = _probe_argv(
        request, paths["raw_probe"], paths["process_joule_provenance"],
    )
    backend = ProcessBackend() if backend is None else backend
    result = orchestrate_capture(
        backend=backend, probe_argv=probe_argv, probe_env=env, lease_fd=lease_fd,
        output_paths=paths, output_dir_fd=output_tree.directory_fd,
        probe_timeout_seconds=3600 if request["kind"] == "device" else 21600,
    )
    if result.probe_returncode != 0:
        raise EvidenceError(f"physical probe failed with exit code {result.probe_returncode}")
    accepted_codes = {
        "xctrace": {
            int(claim.split("=", 1)[1])
            for claim in receipts["xctrace_capability"]["claims"]
            if isinstance(claim, str) and claim.startswith("graceful_exit_code=")
        },
    }
    for name, returncode in result.collector_returncodes.items():
        if not accepted_codes[name] or returncode not in accepted_codes[name]:
            raise EvidenceError(f"{name} exit code {returncode} is not admitted by its signed ABI receipt")

    raw = _load_json(paths["raw_probe"])
    process_counter_errors = process_joule.probe_counter_block_errors(
        raw.get("process_energy_counters"), expected_pid=result.probe_pid,
    )
    if process_counter_errors:
        raise EvidenceError(
            "release probe direct-process-joule counters failed: "
            + "; ".join(process_counter_errors)
        )
    authority = {
        "probe_binary": request["workload"]["probe"],
        "argv_sha256": canonical_sha256(probe_argv),
        "run_nonce": run_nonce,
        "started_at_unix_ns": result.probe_started_at_unix_ns,
        "ended_at_unix_ns": result.probe_ended_at_unix_ns,
        "started_at_continuous_ns": result.probe_started_at_continuous_ns,
        "ended_at_continuous_ns": result.probe_ended_at_continuous_ns,
        "exit_code": result.probe_returncode,
        "raw_probe_sha256": canonical_sha256(raw),
        "stdout_sha256": _sha_file(paths["probe_stdout"]),
        "stderr_sha256": _sha_file(paths["probe_stderr"]),
    }
    resource_after = ram_scheduler.resource_snapshot(ROOT)
    thermal_after = _thermal_green()
    if request["kind"] == "device":
        bundle = appendix_device_runner.build_bundle(
            raw, resource_before=resource_before, resource_after=resource_after,
            thermal_before="nominal" if thermal_before else "serious",
            thermal_after="nominal" if thermal_after else "serious",
            execution_authority=authority,
        )
        bundle_errors = appendix_device_runner.validate_bundle(bundle)
    else:
        bundle = spec_tq_runner.build_bundle(
            raw, resource_before=resource_before, resource_after=resource_after,
            thermal_before="nominal" if thermal_before else "serious",
            thermal_after="nominal" if thermal_after else "serious",
            execution_authority=authority,
        )
        bundle_errors = spec_tq_runner.validate_bundle(bundle)
    if bundle_errors:
        raise EvidenceError("raw bundle failed: " + "; ".join(bundle_errors))
    _atomic_json(paths["raw_bundle"], bundle)
    raw_bundle_published = output_tree.published_path(names["raw_bundle"])
    raw_bundle_published.chmod(0o444)
    raw_bundle_identity = output_tree.published_file_identity(names["raw_bundle"])
    _atomic_json(paths["process_joule_raw"], raw["process_energy_counters"])
    seal_capture(paths["process_joule_raw"], paths["process_joule_sealed"])
    trace_published = output_tree.published_path(names["xctrace_raw"])
    trace_identity = freeze_trace_tree(trace_published)
    output_tree.assert_attached()
    if raw_bundle_identity != physical_counter_attestation.file_identity(
        raw_bundle_published,
    ):
        raise EvidenceError("raw bundle changed before xctrace export")
    xctrace_capture, xctrace_receipt = xctrace_adapter.run_export(
        kind=request["kind"], trace_path=trace_published,
        raw_bundle_path=raw_bundle_published,
        profile_path=xctrace_adapter.DEFAULT_PROFILE,
        probe_pid=result.probe_pid, run_nonce=run_nonce,
        probe_argv_sha256=authority["argv_sha256"],
        metal_registry_id=device_claims["registry_id"],
        output_dir_fd=output_tree.directory_fd,
        toc_output_name=names["xctrace_toc"],
        export_output_prefix="xctrace.export",
        capture_output_name=names["xctrace_capture"],
        receipt_output_name=names["xctrace_adapter_receipt"],
        lease_fd=lease_fd,
    )
    if xctrace_receipt.get("trace_identity") != trace_identity:
        raise EvidenceError("xctrace adapter receipt differs from frozen trace tree")
    xctrace_toc_path = pathlib.Path(xctrace_receipt["toc_identity"]["path"])
    xctrace_export_paths = {
        table: pathlib.Path(xctrace_receipt["export_identities"][table]["path"])
        for table in xctrace_adapter.REQUIRED_TABLES
    }
    xctrace_capture_path = output_tree.published_path(names["xctrace_capture"])
    xctrace_receipt_path = output_tree.published_path(names["xctrace_adapter_receipt"])
    xctrace_file_validation = xctrace_adapter.validate_receipt(
        receipt_path=xctrace_receipt_path, capture_path=xctrace_capture_path,
        kind=request["kind"], raw_bundle_path=raw_bundle_published,
        profile_path=xctrace_adapter.DEFAULT_PROFILE,
        trace_path=trace_published, toc_path=xctrace_toc_path,
        export_paths=xctrace_export_paths, probe_pid=result.probe_pid,
        run_nonce=run_nonce, probe_argv_sha256=authority["argv_sha256"],
        metal_registry_id=device_claims["registry_id"],
        expected_lease=xctrace_receipt["lease"],
        expected_output_directory=xctrace_receipt["xctrace_runtime"][
            "output_directory"
        ],
    )
    xctrace_export_evidence = build_xctrace_export_evidence(
        request=request, bundle=bundle, execution_authority=authority,
        probe_pid=result.probe_pid, metal_registry_id=device_claims["registry_id"],
        capture=xctrace_capture, receipt=xctrace_receipt,
        capture_path=xctrace_capture_path, receipt_path=xctrace_receipt_path,
        file_backed_validation=xctrace_file_validation,
    )
    sealed = {
        "process_joule": output_tree.published_file_identity(names["process_joule_sealed"]),
        "xctrace": output_tree.published_file_identity(names["xctrace_capture"]),
    }
    normalizer_binary = receipts["normalizer_capability"]["binary"]["path"]
    normalizer_argv = _normalizer_argv(
        normalizer_binary, kind=request["kind"], bundle=raw_bundle_published,
        process_joule_capture=output_tree.published_path(names["process_joule_sealed"]),
        xctrace=xctrace_capture_path, probe_pid=result.probe_pid,
        run_nonce=run_nonce, metal_registry_id=device_claims["registry_id"],
        output=paths["attributed"],
    )
    with paths["normalizer_stdout"].open("xb") as stdout, paths["normalizer_stderr"].open("xb") as stderr:
        normalizer = subprocess.run(
            normalizer_argv, cwd=ROOT, env=env,
            pass_fds=(lease_fd, output_tree.directory_fd),
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            timeout=600, check=False, shell=False,
        )
    if normalizer.returncode != 0:
        raise EvidenceError(f"trusted attributed-sample normalizer failed ({normalizer.returncode})")
    attributed = _load_json(paths["attributed"])
    lease_stat = os.fstat(lease_fd)
    expected_capture_sha256s = {
        collector_id: identity["sha256"] for collector_id, identity in sealed.items()
    }
    attributed_errors = collector.validate_attributed_samples(
        bundle, attributed, kind=request["kind"], expected_probe_pid=result.probe_pid,
        expected_capture_sha256s=expected_capture_sha256s,
        expected_metal_registry_id=device_claims["registry_id"],
        expected_lease={"device": lease_stat.st_dev, "inode": lease_stat.st_ino},
    )
    if attributed_errors:
        raise EvidenceError("attributed samples failed: " + "; ".join(attributed_errors))
    payload = collector.normalize_v2(
        bundle, attributed, kind=request["kind"], expected_probe_pid=result.probe_pid,
        expected_capture_sha256s=expected_capture_sha256s,
        expected_metal_registry_id=device_claims["registry_id"],
        expected_lease={"device": lease_stat.st_dev, "inode": lease_stat.st_ino},
    )
    _atomic_json(paths["counter_payload"], payload)
    raw_capture_sha256s = {
        domain: sealed[collector.DOMAIN_COLLECTOR[domain]]["sha256"]
        for domain in (collector.DEVICE_DOMAINS if request["kind"] == "device" else collector.SPEC_DOMAINS)
    }
    normalized_wrapper = {
        "schema": physical_counter_attestation.NORMALIZED_SCHEMA,
        "raw_bundle_sha256": bundle["raw_bundle_sha256"],
        "execution_authority_sha256": canonical_sha256(authority),
        "counter_payload": payload,
        "raw_capture_sha256s": raw_capture_sha256s,
    }
    _atomic_json(paths["normalized_wrapper"], normalized_wrapper)
    normalized_wrapper_identity = output_tree.published_file_identity(
        names["normalized_wrapper"],
    )
    attestation = _counter_attestation(
        request=request, bundle=bundle, payload=payload,
        attributed=attributed, receipts=receipts,
        normalized_wrapper_identity=normalized_wrapper_identity, sealed=sealed,
        normalizer_argv=normalizer_argv,
        collector_argv={**result.collector_argv, "process_joule": result.probe_argv},
        request_file_identity=request_file_identity,
    )
    _atomic_json(paths["counter_attestation"], attestation)
    extra = request["additional_parent_receipt_sha256"]
    if request["kind"] == "device":
        receipt = appendix_device_runner.finalize_receipt(
            bundle, payload, attestation, request["workload"]["cell_id"],
        )
        receipt = _bind_release_parents(receipt, release=request["release"], extra=extra)
        receipt_errors = tq_receipt_contract.validate_receipt(
            receipt, known_cell_ids={row["id"] for row in tq_runtime_matrix.build_matrix()["cells"]},
        )
        if receipt_errors:
            raise EvidenceError("release-bound device receipt failed: " + "; ".join(receipt_errors))
        evidence = {
            "cell_id": request["workload"]["cell_id"], "raw_bundle": bundle,
            "receipt": receipt, "execution_authority": authority,
            "counter_payload": payload, "counter_attestation": attestation,
            "xctrace_export_evidence": xctrace_export_evidence,
        }
    else:
        parity, curve = spec_tq_runner.finalize_receipts(
            bundle, payload, attestation, label=request["workload"]["label"],
        )
        parity = _bind_release_parents(parity, release=request["release"], extra=extra)
        curve = _bind_release_parents(
            curve, release=request["release"], extra=extra,
            parity_parent=parity["receipt_sha256"],
        )
        known = spec_tq_runner._cells(request["workload"]["runtime_path"], request["workload"]["label"])[2]
        for name, receipt in (("parity", parity), ("curve", curve)):
            receipt_errors = spec_receipt_contract.validate_receipt(receipt, known_cell_ids=known)
            if receipt_errors:
                raise EvidenceError(f"release-bound spec {name} receipt failed: " + "; ".join(receipt_errors))
        evidence = {
            "runtime_path": request["workload"]["runtime_path"], "raw_bundle": bundle,
            "parity_receipt": parity, "curve_receipt": curve,
            "execution_authority": authority, "counter_payload": payload,
            "counter_attestation": attestation,
            "xctrace_export_evidence": xctrace_export_evidence,
        }
    # Final fail-closed CAS before publishing a complete evidence item.  Raw
    # captures remain as honest partial evidence if this boundary changed.
    if physical_counter_attestation.file_identity(request_path) != request_file_identity:
        raise EvidenceError("execution request changed during the physical operation")
    final_observer = _load_json(OBSERVER)
    if final_observer.get("state_sha256") != observer.get("state_sha256"):
        raise EvidenceError("Doctor observer changed during the physical operation")
    if spec_reentry_scaffold.active_heavy_owners():
        raise EvidenceError("a heavy owner appeared during the physical operation")
    final_resource = ram_scheduler.resource_snapshot(ROOT)
    final_resource_errors = _resource_errors(
        final_resource, thermal_green=_thermal_green(),
        scratch_reserve_gb=float(request["scratch_reserve_gb"]),
    )
    if final_resource_errors:
        raise EvidenceError("final resource CAS failed: " + "; ".join(final_resource_errors))
    final_release_cas = _live_release_cas(
        request, observer=final_observer, phase="final",
    )
    if physical_counter_attestation.file_identity(
        paths["process_joule_provenance"]
    ) != process_provenance_identity:
        raise EvidenceError("process-joule provenance changed during physical execution")
    output_tree.assert_attached()
    _atomic_json(paths["evidence"], evidence)
    evidence_file_identity = output_tree.published_file_identity(names["evidence"])
    receipt = _stamp({
        "schema": EXECUTION_SCHEMA,
        "request_sha256": request["request_sha256"],
        "request_file": request_file_identity,
        "authority_chain_sha256": canonical_sha256(request["authorities"]),
        "kind": request["kind"], "run_nonce": run_nonce,
        "probe_pid": result.probe_pid, "probe_argv_sha256": canonical_sha256(probe_argv),
        "external_trace_started_before_barrier": True,
        "capture_readiness": result.readiness,
        "execution_started_at_unix_ns": result.probe_started_at_unix_ns,
        "execution_ended_at_unix_ns": result.probe_ended_at_unix_ns,
        "raw_bundle_sha256": bundle["raw_bundle_sha256"],
        "counter_payload_sha256": canonical_sha256(payload),
        "counter_attestation_sha256": attestation["attestation_sha256"],
        "xctrace_export_evidence_sha256": xctrace_export_evidence[
            "xctrace_export_evidence_sha256"
        ],
        "core_evidence_sha256": canonical_sha256(evidence),
        "evidence_file": evidence_file_identity,
        "process_joule_provenance_file": process_provenance_identity,
        "prelaunch_release_cas": prelaunch_release_cas,
        "final_release_cas": final_release_cas,
        "runtime_defaults_changed": False,
    }, "execution_receipt_sha256")
    receipt_errors = _execution_receipt_errors(receipt, request=request, evidence=evidence)
    if receipt_errors:
        raise EvidenceError("execution receipt failed: " + "; ".join(receipt_errors))
    _atomic_json(paths["execution_receipt"], receipt)
    result_draft = build_result_attestation(
        request=request, execution_receipt=receipt, evidence=evidence,
    )
    _atomic_json(paths["result_attestation_draft"], result_draft)
    output_tree.published_file_identity(names["execution_receipt"])
    output_tree.published_file_identity(names["result_attestation_draft"])
    output_tree.assert_attached()
    output_tree.close()
    return receipt


def _selftest() -> int:
    assert status(
        euid=501, final_ready=False, heavy_owner_count=4, inherited_lease_fds=(),
        process_joule_present=True, full_xcode_present=False,
    )["execution_ready"] is False
    assert dry_run("device")["would_execute"] is False
    assert set(AUTHORITY_SPECS) == {
        "process_joule_capability", "process_joule_attribution",
        "xctrace_capability", "xctrace_privilege", "xctrace_attribution",
        "normalizer_capability", "normalizer_attribution", "device_identity", "inherited_lease",
    }
    print("appendix_physical_counter_executor.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--lease-proof-child":
        return _lease_proof_child(argv[1])
    if len(argv) >= 4 and argv[0] == "--barrier-child" and argv[2] == "--":
        return _barrier_child(argv[1], argv[3:])
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true")
    group.add_argument("--capability", action="store_true")
    group.add_argument("--authority-requirements", action="store_true")
    group.add_argument("--dry-run", choices=("device", "spec"))
    group.add_argument("--execute", type=pathlib.Path, metavar="REQUEST")
    group.add_argument("--lease-holder-execute", type=pathlib.Path, metavar="REQUEST")
    group.add_argument("--draft-result", type=pathlib.Path, metavar="REQUEST")
    group.add_argument("--sign-result", type=pathlib.Path, metavar="DRAFT")
    group.add_argument("--seal-result", type=pathlib.Path, metavar="SIGNED_RESULT")
    group.add_argument("--selftest", action="store_true")
    parser.add_argument("--acknowledge-request-sha256")
    parser.add_argument("--request", type=pathlib.Path)
    parser.add_argument("--execution-receipt", type=pathlib.Path)
    parser.add_argument("--evidence", type=pathlib.Path)
    parser.add_argument("--private-key", type=pathlib.Path)
    parser.add_argument("--signature-output", type=pathlib.Path)
    parser.add_argument("--envelope-output", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.status:
        print(json.dumps(status(), indent=2, sort_keys=True))
        return 0
    if args.capability:
        print(json.dumps(execution_capability_contract(), indent=2, sort_keys=True))
        return 0
    if args.authority_requirements:
        print(json.dumps(authority_requirements(), indent=2, sort_keys=True))
        return 0
    if args.dry_run:
        print(json.dumps(dry_run(args.dry_run), indent=2, sort_keys=True))
        return 0
    try:
        if args.draft_result is not None:
            if args.execution_receipt is None or args.evidence is None or args.output is None:
                parser.error(
                    "--draft-result requires --execution-receipt, --evidence, and --output"
                )
            value = draft_result_files(
                request_path=args.draft_result,
                execution_receipt_path=args.execution_receipt,
                evidence_path=args.evidence, output_path=args.output,
            )
            print(json.dumps(value, indent=2, sort_keys=True))
            return 0
        if args.sign_result is not None:
            if any(row is None for row in (
                args.request, args.execution_receipt, args.evidence, args.private_key,
                args.signature_output, args.envelope_output,
            )):
                parser.error(
                    "--sign-result requires --request, --execution-receipt, --evidence, "
                    "--private-key, --signature-output, and --envelope-output"
                )
            value = sign_result_files(
                request_path=args.request,
                execution_receipt_path=args.execution_receipt,
                evidence_path=args.evidence, draft_path=args.sign_result,
                private_key=args.private_key,
                detached_signature_output=args.signature_output,
                envelope_output=args.envelope_output,
            )
            print(json.dumps(value, indent=2, sort_keys=True))
            return 0
        if args.seal_result is not None:
            if any(row is None for row in (
                args.request, args.execution_receipt, args.evidence, args.output,
            )):
                parser.error(
                    "--seal-result requires --request, --execution-receipt, --evidence, and --output"
                )
            value = seal_result_files(
                request_path=args.request,
                execution_receipt_path=args.execution_receipt,
                evidence_path=args.evidence, signed_result_path=args.seal_result,
                output_path=args.output,
            )
            print(json.dumps(value, indent=2, sort_keys=True))
            return 0
    except (EvidenceError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"appendix physical counter result workflow failed: {exc}", file=sys.stderr)
        return 1
    if args.lease_holder_execute is not None:
        if not args.acknowledge_request_sha256:
            parser.error("--lease-holder-execute requires --acknowledge-request-sha256")
        try:
            return lease_holder_execute(
                args.lease_holder_execute,
                acknowledgement=args.acknowledge_request_sha256,
            )
        except AdmissionBlocked as exc:
            print(f"appendix physical counter lease holder blocked: {exc}", file=sys.stderr)
            return EXIT_BLOCKED
    if not args.acknowledge_request_sha256:
        parser.error("--execute requires --acknowledge-request-sha256")
    try:
        value = execute(
            args.execute, acknowledgement=args.acknowledge_request_sha256,
        )
    except AdmissionBlocked as exc:
        print(f"appendix physical counter execution blocked: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    except (EvidenceError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"appendix physical counter evidence failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

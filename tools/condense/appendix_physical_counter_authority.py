#!/usr/bin/env python3.12
"""Pinned trust root and signing workflow for Appendix counter authority.

An authority envelope is never allowed to choose its own verifier.  This
module owns a source-sealed registry which pins one SSHSIG namespace, signer
identity, and allowed-signers byte hash.  The executor compares the request's
registry to that exact source file and compares every envelope's signer file to
the registry before invoking ``ssh-keygen -Y verify``.

Status, requirements, dry-run, and self-test are non-executing.  Receipt
drafting only measures host identity and hashes named binaries.  Signing is an
explicit operator action; the private key is read by ``ssh-keygen`` and is
never copied into a report or receipt.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

import condense_profiles

condense_profiles.install_archive_importer()

import appendix_contract
from condense_common import canonical_sha256, stamp_sha256 as _stamp
import appendix_physical_counter_normalizer as trusted_normalizer
import appendix_process_joule_collector as process_joule
import physical_counter_attestation


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_ALLOWED_SIGNERS = ROOT / "docs" / "plans" / "appendix_counter_authority_allowed_signers"
DEFAULT_REGISTRY = ROOT / "docs" / "plans" / "appendix_counter_authority_registry.json"
DEFAULT_OPERATOR_PRIVATE_KEY = pathlib.Path.home() / ".ssh" / "id_ed25519"
DEFAULT_OPERATOR_PUBLIC_KEY = pathlib.Path.home() / ".ssh" / "id_ed25519.pub"
SSH_KEYGEN = pathlib.Path("/usr/bin/ssh-keygen")
IOREG = pathlib.Path("/usr/sbin/ioreg")
SIGNER_IDENTITY = "hawking-appendix-release"
SSHSIG_NAMESPACE = "hawking-appendix-counter-authority-v1"
RESULT_SSHSIG_NAMESPACE = "hawking-appendix-counter-result-v1"
TRUSTED_NORMALIZER = pathlib.Path(trusted_normalizer.__file__).resolve()
TRUSTED_NORMALIZER_RELATIVE_PATH = TRUSTED_NORMALIZER.relative_to(ROOT.resolve()).as_posix()

# These values are the reviewed operator trust anchor, not values learned from
# a request or regenerated from the working tree.  The registry, its
# allowed-signers bytes, and the operator-held public key must all agree with
# these constants.  This prevents replacing both mutable registry files and
# then blessing an attacker-selected key during a fresh release preparation.
PINNED_REGISTRY_SHA256 = "b5d538250e92e244586aeb7fa2af86636f8111ae2ac2328b06a71a2f249612d2"
PINNED_ALLOWED_SIGNERS_SHA256 = "a70164edd312fc8d33ae77fcd292f57241790aed737dcfefa5f7615c36498620"
PINNED_PUBLIC_KEY_BLOB_SHA256 = "6db91ec62ce28aa4c6fe51f019f06b985e9e78bb3f8c345f61bd6186dd4d7dc8"

REGISTRY_SCHEMA = "hawking.appendix_counter_authority_registry.v1"
AUTHORITY_SCHEMA = "hawking.appendix_counter_authority_receipt.v1"
ENVELOPE_SCHEMA = "hawking.appendix_counter_signed_authority_envelope.v1"
RESULT_ATTESTATION_SCHEMA = "hawking.appendix_counter_result_attestation.v1"
RESULT_ENVELOPE_SCHEMA = "hawking.appendix_counter_signed_result_envelope.v1"
SEALED_EVIDENCE_SCHEMA = "hawking.appendix_counter_sealed_evidence.v1"
STATUS_SCHEMA = "hawking.appendix_counter_authority_status.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UUID = re.compile(
    r'"IOPlatformUUID"\s*=\s*"([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-'
    r'[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})"'
)


# The ABI and claim contracts live beside the compiled signer anchors so both
# the release executor and the offline aggregate gate consume one independent
# definition.  A sealed result cannot weaken these values by carrying its own
# alternative policy.
POWERMETRICS_ABI = (
    "/usr/bin/powermetrics", "--show-process-energy", "--samplers",
    "tasks,gpu_power", "-i", "100", "-f", "plist", "-o", "{raw_output}",
)
XCTRACE_ABI = (
    "/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace",
    "record", "--template", "Metal System Trace", "--attach", "{probe_pid}",
    "--output", "{raw_output}",
    "then", "python3.12", "-m", "tools.condense", "legacy",
    "appendix_xctrace_export_adapter",
    "--export", "--kind", "{kind}", "--trace", "{raw_output}",
    "--raw-bundle", "{raw_bundle}", "--profile", "{signed_profile}",
    "--probe-pid", "{probe_pid}", "--run-nonce", "{run_nonce}",
    "--probe-argv-sha256", "{probe_argv_sha256}",
    "--metal-registry-id", "{metal_registry_id}",
    "--output-dir-fd", "{held_output_dir_fd}",
    "--toc-output-name", "{toc_output_name}",
    "--export-output-prefix", "{export_output_prefix}",
    "--capture-output-name", "{canonical_capture_output_name}",
    "--receipt-output-name", "{adapter_receipt_output_name}",
)
NORMALIZER_ABI = (
    "{normalizer}", "--kind", "{kind}", "--raw-bundle", "{raw_bundle}",
    "--process-joule", "{process_joule_capture}", "--xctrace", "{xctrace_capture}",
    "--probe-pid", "{probe_pid}", "--run-nonce", "{run_nonce}",
    "--metal-registry-id", "{metal_registry_id}",
    "--output", "{attributed_samples}",
)
PROCESS_JOULE_ABI = (
    "release-probe-bracketing-self-sampled-operation-only-timing",
    "proc_pid_rusage", "RUSAGE_INFO_V6",
    process_joule.PROBE_COUNTERS_SCHEMA, process_joule.PHASE_RECORD_SCHEMA,
    process_joule.BACKEND_ID, process_joule.CONTRACT_SHA256,
    process_joule.BOUNDARY_PROTOCOL,
    "ri_energy_nj", "ri_penergy_nj", "ri_instructions", "ri_cycles",
    "pid+ri_uuid+ri_proc_start_abstime",
)
LEASE_HOLDER_ABI = (
    "python3.12", "-m", "tools.condense", "legacy",
    "appendix_physical_counter_executor",
    "--lease-holder-execute", "REQUEST", "--acknowledge-request-sha256", "SHA256",
)
COMMAND_ABI_HASHES = {
    "powermetrics-rejected-proxy": appendix_contract.canonical_sha256(list(POWERMETRICS_ABI)),
    "process-joule": appendix_contract.canonical_sha256(list(PROCESS_JOULE_ABI)),
    "xctrace": appendix_contract.canonical_sha256(list(XCTRACE_ABI)),
    "normalizer": appendix_contract.canonical_sha256(list(NORMALIZER_ABI)),
    "release-orchestrator": appendix_contract.canonical_sha256(list(LEASE_HOLDER_ABI)),
    "device-probe": appendix_contract.canonical_sha256([
        "probe", "--artifact", "{artifact}", "--runtime-path", "{runtime_path}",
        "--warmups", "{warmups}", "--trials", "{trials}", "--matrix-cell-id",
        "{cell_id}", "--matrix-model", "{model}", "--matrix-cell-sha256",
        "{cell_sha256}", "--matrix-tensor-family", "{tensor}", "--source-commit",
        "{commit}", "--process-joule-provenance", "{process_joule_provenance}",
        "--output", "{output}", "--tensor", "{tensor}",
        "[--residual-artifact {path} --residual-tensor {tensor}]",
    ]),
    "spec-probe": appendix_contract.canonical_sha256([
        "probe", "--weights", "{weights}", "--artifact", "{artifact}",
        "--prompts", "{prompts}", "--runtime-path", "{runtime_path}",
        "--generated-tokens", "{tokens}", "--source-commit", "{commit}",
        "--process-joule-provenance", "{process_joule_provenance}",
        "--warmups-per-batch", "{warmups}", "--repeats-per-batch", "{repeats}",
        "--parity-cell-id", "{parity_cell}", "--curve-cell-id", "{curve_cell}",
        "--output", "{output}",
    ]),
}

AUTHORITY_SPECS: dict[str, tuple[str, str, frozenset[str]]] = {
    "process_joule_capability": (
        "process-joule", "capability",
        frozenset({
            "proc_pid_rusage", "RUSAGE_INFO_V6", "ri_energy_nj", "ri_penergy_nj",
            "bracketing-self-sampling-operation-only-timing", "online-readiness",
            f"backend_id={process_joule.BACKEND_ID}",
            f"collector_contract_sha256={process_joule.CONTRACT_SHA256}",
            f"probe_counters_schema={process_joule.PROBE_COUNTERS_SCHEMA}",
            f"phase_record_schema={process_joule.PHASE_RECORD_SCHEMA}",
            f"boundary_protocol={process_joule.BOUNDARY_PROTOCOL}",
        }),
    ),
    "process_joule_attribution": (
        "process-joule", "attribution",
        frozenset({
            "probe-pid", "process-uuid", "process-start-abstime", "phase-marker",
            "interval-sha256", "wall+continuous-window",
            "counter-snapshots-bracket-operation-interval",
            "performance-interval-excludes-counter-reads",
            "monotone-no-wrap", "no-estimation", "no-apportionment",
            "sdk-header-hashes", "dyld-shared-cache-uuid", "os-build",
        }),
    ),
    "xctrace_capability": (
        "xctrace", "capability",
        frozenset({
            "full-xcode", "metal-system-trace", "attach-pid", "online-readiness",
            "operator-sshsig-export-profile", "four-table-positional-export",
            "full-row-recompute-receipt", "zero-synthetic-credit",
        }),
    ),
    "xctrace_privilege": (
        "xctrace", "privilege",
        frozenset({"noninteractive", "metal-trace-authorized", "no-gui-prompt"}),
    ),
    "xctrace_attribution": (
        "xctrace", "attribution",
        frozenset({
            "probe-pid", "metal-registry-id", "run-nonce", "phase-marker",
            "wall+continuous-window", "predeclared-interval-id",
            "os-signpost-begin+end", "command-buffer+encoder-label",
            "direct-counter-row-join", "no-overlap-only-attribution",
        }),
    ),
    "normalizer_capability": (
        "normalizer", "capability",
        frozenset({
            "attributed-samples-v2", "device+spec", "sealed-capture-input",
            "v2-lossless", "direct-process-joule-only",
            "xctrace-canonical-json-only", "xctrace-adapter-receipt-required",
            "unsupported-backend-fail-closed",
            f"normalizer_schema={trusted_normalizer.SCHEMA}",
            f"normalizer_contract_sha256={trusted_normalizer.CONTRACT_SHA256}",
            f"normalizer_relative_path={TRUSTED_NORMALIZER_RELATIVE_PATH}",
            f"process_joule_backend={process_joule.BACKEND_ID}",
            f"process_joule_contract_sha256={process_joule.CONTRACT_SHA256}",
        }),
    ),
    "normalizer_attribution": (
        "normalizer", "attribution",
        frozenset({
            "probe-pid", "run-nonce", "exact-phase-join", "source-sample-ids",
            "inherited-lease-device+inode", "sealed-capture-sha256",
            "metal-registry-id", "no-estimation", "no-apportionment",
        }),
    ),
    "device_identity": (
        "release-probe", "device-identity",
        frozenset({"registry-id", "hardware-uuid", "os-build", "driver-build"}),
    ),
    "inherited_lease": (
        "release-orchestrator", "inherited-lease",
        frozenset({"canonical-lock", "exclusive", "opened-before-executor", "owner-rechecked"}),
    ),
}


class AuthorityError(ValueError):
    """A registry, receipt, key, or immutable output is not trustworthy."""


def _module_source_identity(
    module_name: str, path: pathlib.Path,
) -> tuple[dict[str, Any], str | None]:
    """Identify retained bytes or their immutable compatibility archive."""
    if condense_profiles.normalize(module_name) in condense_profiles.EXECUTABLE_MODULES:
        record = condense_profiles.legacy_record(module_name)
        source = condense_profiles.archive_source(module_name)
        return {
            "path": f"git:{record['archive_commit']}:{record['path']}",
            "sha256": record["source_sha256"],
            "size_bytes": len(source),
        }, str(record["archive_commit"])
    if os.path.lexists(path):
        return physical_counter_attestation.file_identity(path), None
    record = condense_profiles.legacy_record(module_name)
    source = condense_profiles.archive_source(module_name)
    return {
        "path": f"git:{record['archive_commit']}:{record['path']}",
        "sha256": record["source_sha256"],
        "size_bytes": len(source),
    }, str(record["archive_commit"])




def trusted_normalizer_identity() -> dict[str, Any]:
    """Measure the reviewed normalizer bytes at their one pinned source path."""
    identity, normalizer_archive = _module_source_identity(
        "appendix_physical_counter_normalizer", TRUSTED_NORMALIZER,
    )
    if normalizer_archive is None and identity["path"] != str(TRUSTED_NORMALIZER):
        raise AuthorityError("trusted normalizer resolved outside its pinned source path")
    process_identity, process_archive = _module_source_identity(
        "appendix_process_joule_collector", pathlib.Path(process_joule.__file__),
    )
    return {
        "relative_path": TRUSTED_NORMALIZER_RELATIVE_PATH,
        "file": identity,
        "archive_commit": normalizer_archive,
        "schema": trusted_normalizer.SCHEMA,
        "contract_sha256": trusted_normalizer.CONTRACT_SHA256,
        "attributed_schema": trusted_normalizer.ATTRIBUTED_SCHEMA,
        "process_joule_collector": {
            "file": process_identity,
            "archive_commit": process_archive,
            "schema": process_joule.SCHEMA,
            "contract_sha256": process_joule.CONTRACT_SHA256,
            "backend_id": process_joule.BACKEND_ID,
        },
    }




def _hash_errors(value: Any, field: str, *, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} must be an object"]
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop(field, None)
    if not isinstance(claimed, str) or HEX64.fullmatch(claimed) is None:
        return [f"{label}.{field} is invalid"]
    return [] if claimed == canonical_sha256(unstamped) else [f"{label}.{field} mismatch"]


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_file_bytes(path: pathlib.Path, *, maximum: int = 1024 * 1024) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.absolute(), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
                or before.st_size <= 0 or before.st_size > maximum:
            raise AuthorityError(f"unsafe authority file: {path}")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(65536, maximum + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > maximum:
                raise AuthorityError(f"authority file exceeds bounded size: {path}")
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns, row.st_nlink,
        )
        if identity(before) != identity(after) or observed != after.st_size:
            raise AuthorityError(f"authority file changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(_safe_file_bytes(path, maximum=16 * 1024 * 1024).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"invalid authority JSON {path}: {exc}") from exc


def _lexical_absolute(path: pathlib.Path) -> pathlib.Path:
    """Normalize an output path without following a mutable symlink chain."""
    return pathlib.Path(os.path.abspath(os.fspath(path)))


def _open_dir_nofollow(path: pathlib.Path, *, create: bool) -> int:
    """Walk an absolute directory using retained, no-follow descriptors."""
    absolute = _lexical_absolute(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_immutable_at(
    directory_fd: int, name: str, *, expected: bytes, mode: int,
) -> tuple[int, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
                or stat.S_IMODE(before.st_mode) != mode:
            raise AuthorityError(
                f"authority output is not a single-link mode-{mode:04o} regular file: {name}"
            )
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > len(expected):
                break
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
            row.st_ctime_ns, row.st_nlink, stat.S_IMODE(row.st_mode),
        )
        if identity(before) != identity(after) or b"".join(chunks) != expected:
            raise AuthorityError(f"authority output changed or differs: {name}")
        return int(after.st_dev), int(after.st_ino)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: pathlib.Path, raw: bytes, *, mode: int = 0o444) -> None:
    """Install one immutable output through a retained parent dirfd.

    Byte equality is insufficient: a matching pre-existing writable file is
    rejected.  Parent replacement, symlink insertion, short writes, and a
    concurrent different writer all fail closed.
    """
    if not raw:
        raise AuthorityError(f"refusing to seal an empty authority output: {path}")
    target = _lexical_absolute(path)
    directory_fd = _open_dir_nofollow(target.parent, create=True)
    temporary = f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    installed_identity: tuple[int, int] | None = None
    created_target = False
    try:
        try:
            _read_immutable_at(directory_fd, target.name, expected=raw, mode=mode)
        except FileNotFoundError:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL \
                | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
            try:
                view = memoryview(raw)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short write while sealing authority output")
                    view = view[written:]
                os.fsync(descriptor)
                os.fchmod(descriptor, mode)
            finally:
                os.close(descriptor)
            try:
                os.link(
                    temporary, target.name, src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd, follow_symlinks=False,
                )
            except FileExistsError:
                _read_immutable_at(directory_fd, target.name, expected=raw, mode=mode)
            else:
                linked = os.stat(target.name, dir_fd=directory_fd, follow_symlinks=False)
                installed_identity = (int(linked.st_dev), int(linked.st_ino))
                created_target = True
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            _read_immutable_at(directory_fd, target.name, expected=raw, mode=mode)
        os.fsync(directory_fd)
        visible_fd = _open_dir_nofollow(target.parent, create=False)
        try:
            retained, visible = os.fstat(directory_fd), os.fstat(visible_fd)
            if (retained.st_dev, retained.st_ino) != (visible.st_dev, visible.st_ino):
                raise AuthorityError(f"authority output parent was replaced: {target.parent}")
        finally:
            os.close(visible_fd)
    except BaseException:
        if created_target and installed_identity is not None:
            try:
                current = os.stat(target.name, dir_fd=directory_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == installed_identity:
                    os.unlink(target.name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(directory_fd)


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _parse_allowed_signers(raw: bytes, *, signer_identity: str) -> tuple[str, str]:
    try:
        lines = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]
    except UnicodeError as exc:
        raise AuthorityError("allowed-signers file is not UTF-8") from exc
    if len(lines) != 1:
        raise AuthorityError("authority registry requires exactly one allowed-signers line")
    parts = lines[0].split()
    if len(parts) != 3 or parts[0] != signer_identity:
        raise AuthorityError("allowed-signers identity is not the exact pinned signer")
    algorithm, blob = parts[1], parts[2]
    if algorithm != "ssh-ed25519" or not blob.startswith("AAAA"):
        raise AuthorityError("authority registry requires one Ed25519 public key")
    return algorithm, blob


def build_registry(
    allowed_signers_path: pathlib.Path = DEFAULT_ALLOWED_SIGNERS,
    *, signer_identity: str = SIGNER_IDENTITY,
) -> dict[str, Any]:
    raw = _safe_file_bytes(allowed_signers_path)
    algorithm, blob = _parse_allowed_signers(raw, signer_identity=signer_identity)
    try:
        relative = allowed_signers_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise AuthorityError("allowed-signers trust root must live inside the source capsule") from exc
    return _stamp({
        "schema": REGISTRY_SCHEMA,
        "registry_version": 1,
        "generated_by": "root-release-authority-bootstrap",
        "signer_identity": signer_identity,
        "sshsig_namespace": SSHSIG_NAMESPACE,
        "allowed_signers_relative_path": relative,
        "allowed_signers_sha256": _sha_bytes(raw),
        "public_key_algorithm": algorithm,
        "public_key_blob_sha256": _sha_bytes(f"{algorithm} {blob}".encode("ascii")),
        "authority_receipt_schema": AUTHORITY_SCHEMA,
        "authority_envelope_schema": ENVELOPE_SCHEMA,
        "private_key_embedded": False,
        "default_off": True,
        "registry_grants_execution": False,
    }, "registry_sha256")


def validate_registry(
    value: Any, *, verify_files: bool = True, require_default: bool = True,
    registry_path: pathlib.Path = DEFAULT_REGISTRY,
) -> list[str]:
    expected = {
        "schema", "registry_version", "generated_by", "signer_identity",
        "sshsig_namespace", "allowed_signers_relative_path", "allowed_signers_sha256",
        "public_key_algorithm", "public_key_blob_sha256", "authority_receipt_schema",
        "authority_envelope_schema", "private_key_embedded", "default_off",
        "registry_grants_execution", "registry_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return ["authority registry fields are incomplete or unexpected"]
    errors = _hash_errors(value, "registry_sha256", label="authority_registry")
    if value.get("schema") != REGISTRY_SCHEMA or value.get("registry_version") != 1:
        errors.append("authority registry schema/version is invalid")
    if value.get("generated_by") != "root-release-authority-bootstrap":
        errors.append("authority registry was not generated by the root release bootstrap")
    if value.get("signer_identity") != SIGNER_IDENTITY \
            or value.get("sshsig_namespace") != SSHSIG_NAMESPACE:
        errors.append("authority registry signer/namespace is not the pinned production identity")
    if value.get("private_key_embedded") is not False or value.get("default_off") is not True \
            or value.get("registry_grants_execution") is not False:
        errors.append("authority registry weakens default-off/private-key isolation")
    if value.get("authority_receipt_schema") != AUTHORITY_SCHEMA \
            or value.get("authority_envelope_schema") != ENVELOPE_SCHEMA:
        errors.append("authority registry schema bindings are wrong")
    if value.get("registry_sha256") != PINNED_REGISTRY_SHA256:
        errors.append("authority registry differs from the compiled operator trust anchor")
    if value.get("allowed_signers_sha256") != PINNED_ALLOWED_SIGNERS_SHA256:
        errors.append("allowed-signers bytes differ from the compiled operator trust anchor")
    if value.get("public_key_blob_sha256") != PINNED_PUBLIC_KEY_BLOB_SHA256:
        errors.append("authority public key differs from the compiled operator trust anchor")
    relative = value.get("allowed_signers_relative_path")
    if not isinstance(relative, str) or pathlib.PurePosixPath(relative).is_absolute() \
            or ".." in pathlib.PurePosixPath(relative).parts:
        errors.append("authority registry allowed-signers path is unsafe")
        allowed = None
    else:
        allowed = ROOT / relative
    if verify_files and allowed is not None:
        try:
            raw = _safe_file_bytes(allowed)
            algorithm, blob = _parse_allowed_signers(raw, signer_identity=SIGNER_IDENTITY)
        except (OSError, AuthorityError) as exc:
            errors.append(f"pinned allowed-signers file cannot be verified: {exc}")
        else:
            if value.get("allowed_signers_sha256") != _sha_bytes(raw):
                errors.append("pinned allowed-signers byte hash mismatch")
            if value.get("public_key_algorithm") != algorithm \
                    or value.get("public_key_blob_sha256") != _sha_bytes(
                        f"{algorithm} {blob}".encode("ascii")
                    ):
                errors.append("pinned signer public-key hash mismatch")
        try:
            operator_public = _safe_file_bytes(DEFAULT_OPERATOR_PUBLIC_KEY)
            operator_parts = operator_public.decode("utf-8").strip().split()
            operator_hash = _sha_bytes(
                f"{operator_parts[0]} {operator_parts[1]}".encode("ascii")
            ) if len(operator_parts) >= 2 else None
        except (OSError, UnicodeError, AuthorityError) as exc:
            errors.append(f"out-of-repository operator public key cannot be verified: {exc}")
        else:
            if operator_hash != PINNED_PUBLIC_KEY_BLOB_SHA256:
                errors.append("out-of-repository operator public key differs from trust anchor")
    if require_default:
        try:
            default = _load_json(registry_path)
        except (OSError, AuthorityError) as exc:
            errors.append(f"source-sealed authority registry cannot be read: {exc}")
        else:
            if value != default:
                errors.append("request authority registry differs from the source-sealed trust root")
    return errors


def load_default_registry() -> dict[str, Any]:
    value = _load_json(DEFAULT_REGISTRY)
    errors = validate_registry(value, verify_files=True, require_default=False)
    if errors:
        raise AuthorityError("default authority registry is invalid: " + "; ".join(errors))
    return value


def allowed_signers_identity(registry: dict[str, Any]) -> dict[str, Any]:
    errors = validate_registry(registry, verify_files=True, require_default=True)
    if errors:
        raise AuthorityError("authority registry is invalid: " + "; ".join(errors))
    path = ROOT / registry["allowed_signers_relative_path"]
    return physical_counter_attestation.file_identity(path)


@contextlib.contextmanager
def pinned_verification_material(
    envelope: dict[str, Any], registry: dict[str, Any],
) -> Iterable[tuple[pathlib.Path, pathlib.Path]]:
    """Yield private stable copies of the pinned signer list and signature.

    ``ssh-keygen -Y verify`` is never given a pathname selected by the
    envelope.  Both inputs are opened with ``O_NOFOLLOW``, byte-hash checked,
    and copied into a mode-0700 temporary directory before the verifier is
    launched.  This closes the identity-check/pathname-use race.
    """
    registry_errors = validate_registry(registry, verify_files=True, require_default=True)
    if registry_errors:
        raise AuthorityError("authority registry is invalid: " + "; ".join(registry_errors))
    allowed_raw = _safe_file_bytes(DEFAULT_ALLOWED_SIGNERS)
    if _sha_bytes(allowed_raw) != PINNED_ALLOWED_SIGNERS_SHA256:
        raise AuthorityError("pinned allowed-signers bytes changed before verification")
    signature = envelope.get("detached_signature")
    if not isinstance(signature, dict) or set(signature) != {"path", "sha256", "size_bytes"}:
        raise AuthorityError("detached signature identity is malformed")
    try:
        signature_path = pathlib.Path(signature["path"])
    except TypeError as exc:
        raise AuthorityError("detached signature path is invalid") from exc
    signature_raw = _safe_file_bytes(signature_path, maximum=1024 * 1024)
    if _sha_bytes(signature_raw) != signature.get("sha256") \
            or len(signature_raw) != signature.get("size_bytes"):
        raise AuthorityError("detached signature changed before verification")
    with tempfile.TemporaryDirectory(prefix="hawking-sshsig-verify-") as directory:
        root = pathlib.Path(directory)
        root.chmod(0o700)
        allowed_copy = root / "allowed_signers"
        signature_copy = root / "signature.sshsig"
        for path, raw in ((allowed_copy, allowed_raw), (signature_copy, signature_raw)):
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o400,
            )
            try:
                view = memoryview(raw)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short write while pinning SSHSIG verification material")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        yield allowed_copy, signature_copy


def verify_sshsig_envelope(
    envelope: dict[str, Any], payload: bytes, *, namespace: str,
    registry: dict[str, Any] | None = None, runner: Any = subprocess.run,
) -> tuple[bool, str]:
    """Verify with stable pinned bytes; never trust envelope path selection."""
    try:
        registry = load_default_registry() if registry is None else registry
        with pinned_verification_material(envelope, registry) as (allowed, signature):
            process = runner(
                [
                    str(SSH_KEYGEN), "-Y", "verify", "-f", str(allowed),
                    "-I", SIGNER_IDENTITY, "-n", namespace, "-s", str(signature),
                ],
                cwd=ROOT, input=payload, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=15, check=False, shell=False,
            )
    except (OSError, subprocess.TimeoutExpired, AuthorityError) as exc:
        return False, f"SSHSIG verifier failed: {exc}"
    detail_raw = process.stdout or process.stderr
    detail = detail_raw.decode("utf-8", "replace").strip() \
        if isinstance(detail_raw, bytes) else str(detail_raw).strip()
    return process.returncode == 0, detail[-1000:]


def live_host_hardware_uuid_sha256(
    *, runner: Any = subprocess.run,
) -> tuple[str | None, str]:
    """Measure the actual IOPlatform UUID and return only its SHA-256."""
    argv = [str(IOREG), "-rd1", "-c", "IOPlatformExpertDevice"]
    try:
        process = runner(
            argv, cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=10, check=False, shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"live host UUID probe failed: {exc}"
    detail = process.stdout or process.stderr
    if process.returncode != 0:
        return None, f"live host UUID probe exited {process.returncode}: {detail[-500:]}"
    match = UUID.search(detail)
    if match is None:
        return None, "live host UUID probe did not expose one IOPlatformUUID"
    normalized = match.group(1).lower()
    return _sha_bytes(normalized.encode("ascii")), "IOPlatformUUID measured and hashed"


def validate_receipt(value: Any) -> list[str]:
    expected = {
        "schema", "receipt_kind", "subject", "host_hardware_uuid_sha256",
        "binary", "command_abi_sha256", "claims", "issued_at_unix_ns",
        "expires_at_unix_ns", "release_build_sha256", "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return ["authority receipt fields are incomplete or unexpected"]
    errors = _hash_errors(value, "receipt_sha256", label="authority_receipt")
    if value.get("schema") != AUTHORITY_SCHEMA:
        errors.append("authority receipt schema is invalid")
    for field in ("receipt_kind", "subject"):
        if not isinstance(value.get(field), str) or not value[field]:
            errors.append(f"authority receipt {field} is empty")
    for field in ("host_hardware_uuid_sha256", "command_abi_sha256", "release_build_sha256"):
        if not isinstance(value.get(field), str) or HEX64.fullmatch(value[field]) is None:
            errors.append(f"authority receipt {field} is invalid")
    errors.extend(physical_counter_attestation._artifact_errors(
        value.get("binary"), label="authority receipt binary", verify_files=True,
    ))
    claims = value.get("claims")
    if not isinstance(claims, list) or not claims or claims != sorted(set(claims)) \
            or any(not isinstance(claim, str) or not claim for claim in claims):
        errors.append("authority receipt claims must be nonempty, unique, and canonical")
    issued, expires = value.get("issued_at_unix_ns"), value.get("expires_at_unix_ns")
    if isinstance(issued, bool) or not isinstance(issued, int) or issued <= 0 \
            or isinstance(expires, bool) or not isinstance(expires, int) or expires <= issued:
        errors.append("authority receipt validity interval is invalid")
    return errors


def build_receipt(
    *, receipt_kind: str, subject: str, binary: pathlib.Path,
    command_abi_sha256: str, claims: Iterable[str], release_build_sha256: str,
    valid_seconds: int = 3600, now_unix_ns: int | None = None,
    host_hardware_uuid_sha256: str | None = None,
) -> dict[str, Any]:
    if host_hardware_uuid_sha256 is None:
        host_hardware_uuid_sha256, detail = live_host_hardware_uuid_sha256()
        if host_hardware_uuid_sha256 is None:
            raise AuthorityError(detail)
    if not isinstance(valid_seconds, int) or isinstance(valid_seconds, bool) \
            or not 60 <= valid_seconds <= 86_400:
        raise AuthorityError("authority receipt validity must be within 60..86400 seconds")
    issued = time.time_ns() if now_unix_ns is None else now_unix_ns
    receipt = _stamp({
        "schema": AUTHORITY_SCHEMA,
        "receipt_kind": receipt_kind,
        "subject": subject,
        "host_hardware_uuid_sha256": host_hardware_uuid_sha256,
        "binary": physical_counter_attestation.file_identity(binary),
        "command_abi_sha256": command_abi_sha256,
        "claims": sorted(set(claims)),
        "issued_at_unix_ns": issued,
        "expires_at_unix_ns": issued + valid_seconds * 1_000_000_000,
        "release_build_sha256": release_build_sha256,
    }, "receipt_sha256")
    errors = validate_receipt(receipt)
    if errors:
        raise AuthorityError("constructed authority receipt is invalid: " + "; ".join(errors))
    return receipt


def _derived_public_key(private_key: pathlib.Path) -> str:
    process = subprocess.run(
        [str(SSH_KEYGEN), "-y", "-f", str(private_key)], cwd=ROOT,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=15, check=False, shell=False,
    )
    if process.returncode != 0:
        raise AuthorityError(f"cannot derive signing public key: {process.stderr[-500:]}")
    parts = process.stdout.strip().split()
    if len(parts) != 2:
        raise AuthorityError("derived signing public key has an unexpected format")
    return " ".join(parts)


def signing_key_custody_status(
    *, private_key: pathlib.Path = DEFAULT_OPERATOR_PRIVATE_KEY,
    public_key: pathlib.Path = DEFAULT_OPERATOR_PUBLIC_KEY,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check custody without reading private-key bytes or invoking ssh-keygen."""
    registry = load_default_registry() if registry is None else registry
    problems: list[str] = []
    try:
        private_resolved = private_key.resolve(strict=True)
        private_stat = private_key.stat(follow_symlinks=False)
    except OSError as exc:
        private_resolved = private_key.absolute()
        private_stat = None
        problems.append(f"operator private key is unavailable: {exc}")
    else:
        if private_key.is_symlink() or not stat.S_ISREG(private_stat.st_mode):
            problems.append("operator private key is not a regular non-symlink file")
        if private_stat.st_mode & 0o077:
            problems.append("operator private key permissions are broader than 0600")
        try:
            private_resolved.relative_to(ROOT.resolve())
        except ValueError:
            pass
        else:
            problems.append("operator private key must remain outside the repository")
    try:
        public_raw = _safe_file_bytes(public_key)
        public_parts = public_raw.decode("utf-8").strip().split()
    except (OSError, UnicodeError, AuthorityError) as exc:
        problems.append(f"operator public key cannot be verified: {exc}")
        public_parts = []
    if len(public_parts) < 2:
        problems.append("operator public key has an unexpected format")
        public_hash = None
    else:
        public_hash = _sha_bytes(f"{public_parts[0]} {public_parts[1]}".encode("ascii"))
        if public_hash != registry.get("public_key_blob_sha256"):
            problems.append("operator public key differs from the source-pinned signer")
    return {
        "private_key_path": str(private_resolved),
        "private_key_bytes_read": False,
        "private_key_outside_repository": not any("outside" in row for row in problems),
        "private_key_mode": (
            oct(stat.S_IMODE(private_stat.st_mode)) if private_stat is not None else None
        ),
        "public_key_path": str(public_key.absolute()),
        "public_key_blob_sha256": public_hash,
        "matches_pinned_registry": public_hash == registry.get("public_key_blob_sha256"),
        "signing_key_available": not problems,
        "problems": problems,
    }


def sign_receipt(
    receipt: dict[str, Any], *, private_key: pathlib.Path,
    detached_signature_output: pathlib.Path, envelope_output: pathlib.Path,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = load_default_registry() if registry is None else registry
    registry_errors = validate_registry(registry, verify_files=True, require_default=True)
    receipt_errors = validate_receipt(receipt)
    if registry_errors or receipt_errors:
        raise AuthorityError("cannot sign invalid registry/receipt: " + "; ".join([
            *registry_errors, *receipt_errors,
        ]))
    try:
        private_stat = private_key.stat(follow_symlinks=False)
        private_resolved = private_key.resolve(strict=True)
    except OSError as exc:
        raise AuthorityError(f"operator private key is unavailable: {exc}") from exc
    if private_key.is_symlink() or not stat.S_ISREG(private_stat.st_mode) \
            or private_stat.st_mode & 0o077:
        raise AuthorityError("operator private key must be a non-symlink regular file with mode 0600")
    try:
        private_resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise AuthorityError("operator private key must remain outside the repository")
    public_key = _derived_public_key(private_key)
    if _sha_bytes(public_key.encode("ascii")) != registry["public_key_blob_sha256"]:
        raise AuthorityError("private key does not match the independently pinned signer")
    message = appendix_contract.canonical_bytes(receipt)
    with tempfile.TemporaryDirectory(prefix="hawking-authority-sign-") as directory:
        message_path = pathlib.Path(directory) / "receipt.canonical.json"
        message_path.write_bytes(message)
        process = subprocess.run(
            [
                str(SSH_KEYGEN), "-Y", "sign", "-f", str(private_key),
                "-n", registry["sshsig_namespace"], str(message_path),
            ],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30, check=False, shell=False,
        )
        generated_signature = message_path.with_suffix(message_path.suffix + ".sig")
        if process.returncode != 0 or not generated_signature.is_file():
            raise AuthorityError(
                f"SSHSIG signing failed ({process.returncode}): {(process.stderr or process.stdout)[-500:]}"
            )
        signature_bytes = generated_signature.read_bytes()
    _atomic_bytes(detached_signature_output, signature_bytes)
    envelope = _stamp({
        "schema": ENVELOPE_SCHEMA,
        "receipt": receipt,
        "signer_identity": registry["signer_identity"],
        "signature_namespace": registry["sshsig_namespace"],
        "allowed_signers": allowed_signers_identity(registry),
        "detached_signature": physical_counter_attestation.file_identity(
            detached_signature_output,
        ),
    }, "envelope_sha256")
    _atomic_json(envelope_output, envelope)
    return envelope


def sign_result_attestation(
    attestation: dict[str, Any], *, private_key: pathlib.Path,
    detached_signature_output: pathlib.Path, envelope_output: pathlib.Path,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Operator-sign one already self-hashed final counter result.

    The function is intentionally specific rather than a general signing
    oracle.  Structural/binding validation remains the executor's job before
    this function is called; here we enforce the schema/hash and pinned-key
    custody, then sign the exact canonical bytes in a result-only namespace.
    """
    registry = load_default_registry() if registry is None else registry
    registry_errors = validate_registry(registry, verify_files=True, require_default=True)
    if registry_errors:
        raise AuthorityError("cannot sign with an invalid registry: " + "; ".join(registry_errors))
    if not isinstance(attestation, dict) or attestation.get("schema") != RESULT_ATTESTATION_SCHEMA:
        raise AuthorityError("result attestation schema is invalid")
    hash_errors = _hash_errors(
        attestation, "result_attestation_sha256", label="result_attestation",
    )
    if hash_errors:
        raise AuthorityError("result attestation hash is invalid: " + "; ".join(hash_errors))
    try:
        private_stat = private_key.stat(follow_symlinks=False)
        private_resolved = private_key.resolve(strict=True)
    except OSError as exc:
        raise AuthorityError(f"operator private key is unavailable: {exc}") from exc
    if private_key.is_symlink() or not stat.S_ISREG(private_stat.st_mode) \
            or private_stat.st_mode & 0o077:
        raise AuthorityError("operator private key must be a non-symlink regular file with mode 0600")
    try:
        private_resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise AuthorityError("operator private key must remain outside the repository")
    public_key = _derived_public_key(private_key)
    if _sha_bytes(public_key.encode("ascii")) != PINNED_PUBLIC_KEY_BLOB_SHA256:
        raise AuthorityError("private key does not match the independently pinned signer")
    message = appendix_contract.canonical_bytes(attestation)
    with tempfile.TemporaryDirectory(prefix="hawking-result-sign-") as directory:
        message_path = pathlib.Path(directory) / "result.canonical.json"
        message_path.write_bytes(message)
        process = subprocess.run(
            [
                str(SSH_KEYGEN), "-Y", "sign", "-f", str(private_key),
                "-n", RESULT_SSHSIG_NAMESPACE, str(message_path),
            ],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30, check=False, shell=False,
        )
        generated_signature = message_path.with_suffix(message_path.suffix + ".sig")
        if process.returncode != 0 or not generated_signature.is_file():
            raise AuthorityError(
                f"result SSHSIG signing failed ({process.returncode}): "
                f"{(process.stderr or process.stdout)[-500:]}"
            )
        signature_bytes = generated_signature.read_bytes()
    _atomic_bytes(detached_signature_output, signature_bytes)
    envelope = _stamp({
        "schema": RESULT_ENVELOPE_SCHEMA,
        "attestation": attestation,
        "signer_identity": registry["signer_identity"],
        "signature_namespace": RESULT_SSHSIG_NAMESPACE,
        "allowed_signers": allowed_signers_identity(registry),
        "detached_signature": physical_counter_attestation.file_identity(
            detached_signature_output,
        ),
    }, "envelope_sha256")
    _atomic_json(envelope_output, envelope)
    return envelope


def status() -> dict[str, Any]:
    try:
        registry = load_default_registry()
        registry_errors: list[str] = []
    except (OSError, AuthorityError) as exc:
        registry = None
        registry_errors = [str(exc)]
    if isinstance(registry, dict):
        custody = signing_key_custody_status(registry=registry)
    else:
        custody = {
            "private_key_path": str(DEFAULT_OPERATOR_PRIVATE_KEY),
            "private_key_bytes_read": False,
            "private_key_outside_repository": True,
            "private_key_mode": None,
            "public_key_path": str(DEFAULT_OPERATOR_PUBLIC_KEY),
            "public_key_blob_sha256": None,
            "matches_pinned_registry": False,
            "signing_key_available": False,
            "problems": ["registry invalid; signing-key match not evaluated"],
        }
    try:
        normalizer_identity = trusted_normalizer_identity()
        normalizer_errors: list[str] = []
    except (OSError, ValueError, AuthorityError) as exc:
        normalizer_identity = None
        normalizer_errors = [f"trusted normalizer cannot be identified: {exc}"]
    try:
        process_status = process_joule.status()
    except (OSError, ValueError, process_joule.ProcessJouleError) as exc:
        process_errors = [f"collector working-tree status unavailable: {exc}"]
        try:
            provenance = process_joule.library_provenance()
        except (OSError, ValueError, process_joule.ProcessJouleError) as provenance_exc:
            provenance = None
            process_errors.append(f"libproc provenance unavailable: {provenance_exc}")
        record = condense_profiles.legacy_record("appendix_process_joule_collector")
        process_errors.extend([
            (
                "collector source is immutable in Git archive "
                f"{record['archive_commit']}:{record['path']}"
            ),
            (
                "probe-side bracketing proc_pid_rusage_v6 source is wired but lacks "
                "a fresh release-build/runtime receipt"
            ),
        ])
        process_status = {
            "direct_process_nanojoule_backend_available": provenance is not None,
            "blockers": process_errors,
        }
    return _stamp({
        "schema": STATUS_SCHEMA,
        "default_off": True,
        "signing_requested": False,
        "private_key_read": False,
        "live_host_uuid_measured": False,
        "registry_valid": not registry_errors,
        "registry_sha256": registry.get("registry_sha256") if isinstance(registry, dict) else None,
        "allowed_signers_sha256": (
            registry.get("allowed_signers_sha256") if isinstance(registry, dict) else None
        ),
        "signing_key_available": custody["signing_key_available"],
        "signing_key_custody": custody,
        "trusted_normalizer": normalizer_identity,
        "direct_process_nanojoule_backend_available": process_status[
            "direct_process_nanojoule_backend_available"
        ],
        "direct_process_joule_backend_admitted": False,
        "powermetrics_process_energy_semantics": "energy-impact proxy; not joules",
        "physical_evidence_claimed": False,
        "blockers": [
            *registry_errors, *custody["problems"], *normalizer_errors,
            *process_status["blockers"],
            "direct-process-joule backend and self-sampling probe are not yet signed/admitted",
        ],
    }, "status_sha256")


def dry_run(action: str) -> dict[str, Any]:
    if action not in {"registry", "receipt", "sign"}:
        raise ValueError("dry-run action is invalid")
    return {
        "schema": "hawking.appendix_counter_authority_dry_run.v1",
        "action": action,
        "would_write": False,
        "would_read_private_key": False,
        "would_start_collector_or_probe": False,
        "would_mutate_runtime_default": False,
        "registry_path": str(DEFAULT_REGISTRY),
    }


def _selftest() -> int:
    registry = load_default_registry()
    assert validate_registry(registry, verify_files=True, require_default=True) == []
    assert status()["registry_valid"] is True
    assert dry_run("sign")["would_read_private_key"] is False
    assert trusted_normalizer_identity()["contract_sha256"] == trusted_normalizer.CONTRACT_SHA256
    assert status()["direct_process_joule_backend_admitted"] is False
    print("appendix_physical_counter_authority.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    dry = sub.add_parser("dry-run")
    dry.add_argument("action", choices=("registry", "receipt", "sign"))
    registry_parser = sub.add_parser("registry")
    registry_parser.add_argument("--allowed-signers", type=pathlib.Path, required=True)
    registry_parser.add_argument("--signer-identity", default=SIGNER_IDENTITY)
    registry_parser.add_argument("--output", type=pathlib.Path, required=True)
    receipt_parser = sub.add_parser("receipt")
    receipt_parser.add_argument("--receipt-kind", required=True)
    receipt_parser.add_argument("--subject", required=True)
    receipt_parser.add_argument("--binary", type=pathlib.Path, required=True)
    receipt_parser.add_argument("--command-abi-sha256", required=True)
    receipt_parser.add_argument("--claim", action="append", required=True)
    receipt_parser.add_argument("--release-build-sha256", required=True)
    receipt_parser.add_argument("--valid-seconds", type=int, default=3600)
    receipt_parser.add_argument("--output", type=pathlib.Path, required=True)
    sign_parser = sub.add_parser("sign")
    sign_parser.add_argument("--receipt", type=pathlib.Path, required=True)
    sign_parser.add_argument("--private-key", type=pathlib.Path, required=True)
    sign_parser.add_argument("--signature-output", type=pathlib.Path, required=True)
    sign_parser.add_argument("--envelope-output", type=pathlib.Path, required=True)
    sub.add_parser("live-host")
    sub.add_parser("selftest")
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps(status(), indent=2, sort_keys=True))
        return 0
    if args.command == "dry-run":
        print(json.dumps(dry_run(args.action), indent=2, sort_keys=True))
        return 0
    try:
        if args.command == "registry":
            value = build_registry(args.allowed_signers, signer_identity=args.signer_identity)
            _atomic_json(args.output, value)
        elif args.command == "receipt":
            value = build_receipt(
                receipt_kind=args.receipt_kind, subject=args.subject, binary=args.binary,
                command_abi_sha256=args.command_abi_sha256, claims=args.claim,
                release_build_sha256=args.release_build_sha256,
                valid_seconds=args.valid_seconds,
            )
            _atomic_json(args.output, value)
        elif args.command == "sign":
            value = sign_receipt(
                _load_json(args.receipt), private_key=args.private_key,
                detached_signature_output=args.signature_output,
                envelope_output=args.envelope_output,
            )
        elif args.command == "live-host":
            digest, detail = live_host_hardware_uuid_sha256()
            value = {"ok": digest is not None, "host_hardware_uuid_sha256": digest, "detail": detail}
            print(json.dumps(value, indent=2, sort_keys=True))
            return 0 if digest is not None else 75
        else:
            return _selftest()
    except (AuthorityError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"appendix counter authority blocked: {exc}", file=sys.stderr)
        return 75
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

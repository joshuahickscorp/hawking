#!/usr/bin/env python3.12
"""Fail-closed, inherited-lease physical A/B executor for Doctor V5.

This module is the reviewed launch surface that the physical A/B controller
describes.  It never uses a shell, never acquires the shared heavy-work lease,
and confines every file it creates to the staged physical-A/B execution root.
An ``execute`` request is rejected with EX_TEMPFAIL (75) before input hashing or
output creation unless Doctor is final-ready, the machine is owner-free, the
direct-counter collection authority is open, and an inherited lock descriptor
is already held by the caller.

The counter collection activation surface is intentionally closed today.  The
status and dry-run commands are therefore safe during the live campaign and an
execute command cannot launch a benchmark.  Once that separately reviewed
authority exists, this executor has the complete no-shell orchestration path:
it revalidates observer, owners, lease, input identities, RAM, swap, thermal,
power, and disk before every arm; starts a hash-bound collector behind a ready
barrier; directly execs the hash-bound arm; and seals exact sidecars and a
controller-compatible facet receipt without changing Doctor state or defaults.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import pathlib
import re
import secrets
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import frontier_runtime as doctor_frontier_worker
import doctor_v5_physical_ab_controller as controller
import doctor_v5_physical_counter_barrier as counter_barrier
import frontier_runtime as spec_reentry_scaffold


ROOT = pathlib.Path(__file__).resolve().parents[2]
OBSERVER_PATH = controller.OBSERVER_PATH
HEAVY_LOCK = ROOT / "reports" / "cron" / "studio_heavy.lock"
HEAVY_LEASE_FD_ENV = "HAWKING_HEAVY_LEASE_FD"
EXECUTION_ROOT = controller.REPORT_ROOT / "executor_runs"

EXECUTOR_SCHEMA = "hawking.doctor_v5_physical_ab_executor.v1"
STATUS_SCHEMA = "hawking.doctor_v5_physical_ab_executor_status.v1"
DRY_RUN_SCHEMA = "hawking.doctor_v5_physical_ab_executor_dry_run.v1"
LAUNCH_SCHEMA = "hawking.doctor_v5_physical_ab_launch_contract.v1"
ARGV_SCHEMA = "hawking.doctor_v5_physical_ab_argv_manifest.v1"
INPUT_SCHEMA = "hawking.doctor_v5_physical_ab_input_manifest.v1"
COLLECTOR_ARGV_SCHEMA = "hawking.doctor_v5_physical_ab_collector_argv.v1"
COLLECTOR_AUTHORITY_SCHEMA = "hawking.doctor_v5_physical_ab_collector_authority.v1"
RELEASE_AUTHORITY_SCHEMA = "hawking.doctor_v5_physical_ab_launch_authority.v1"
COLLECTOR_READY_SCHEMA = "hawking.doctor_v5_physical_ab_collector_ready.v1"
COUNTER_ATTESTATION_SCHEMA = "hawking.doctor_v5_physical_ab_counter_attestation.v1"
ARM_SIDECAR_SCHEMA = "hawking.doctor_v5_physical_ab_arm_sidecars.v1"
EXECUTION_RECEIPT_SCHEMA = "hawking.doctor_v5_physical_ab_execution_receipt.v1"
SCIENTIFIC_RECEIPT_SCHEMA = "hawking.doctor_v5_physical_ab_scientific_receipt.v1"
MATRIX_SCHEMA = "hawking.doctor_v5_physical_ab_multi_launch_manifest.v1"
MULTI_RECEIPT_SCHEMA = controller.MULTI_EXECUTION_RECEIPT_SCHEMA

HEX64 = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER = re.compile(r"\{[A-Z][A-Z0-9_]*\}")
EX_TEMPFAIL = 75
EX_USAGE = 64
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_ARGV_ITEMS = 4096
MAX_ARG_BYTES = 64 * 1024
MAX_REPEATS = 100

ARM_PLACEHOLDERS = {
    "{INPUT_MANIFEST_PATH}",
    "{OUTPUT_PATH}",
    "{SCIENTIFIC_RECEIPT_PATH}",
    "{FACET_PAYLOAD_PATH}",
    "{RUN_NONCE}",
}
COLLECTOR_PLACEHOLDERS = {
    "{COLLECTOR_REQUEST_PATH}",
    "{COLLECTOR_READY_PATH}",
    "{COLLECTOR_STOP_PATH}",
    "{ARM_STARTED_PATH}",
    "{COUNTER_OUTPUT_PATH}",
    "{COUNTER_ATTESTATION_PATH}",
}
EXACT_COUNTER_BARRIER_ARGV = [
    "collect",
    "--request", "{COLLECTOR_REQUEST_PATH}",
    "--ready", "{COLLECTOR_READY_PATH}",
    "--arm-started", "{ARM_STARTED_PATH}",
    "--stop", "{COLLECTOR_STOP_PATH}",
    "--counter-output", "{COUNTER_OUTPUT_PATH}",
    "--counter-attestation", "{COUNTER_ATTESTATION_PATH}",
]
COUNTER_FIELDS = (
    "energy_j", "cpu_time_ns", "read_bytes", "write_bytes", "peak_rss_bytes",
)
BANNED_ENV_KEYS = {
    HEAVY_LEASE_FD_ENV,
    "HAWKING_PHYSICAL_AB_ADMITTED",
    "HAWKING_PHYSICAL_RUN_NONCE",
    "HAWKING_RUNTIME_DEFAULTS",
    "PYTHONPATH",
}
BANNED_ENV_PREFIXES = ("DYLD_", "LD_")

# Concrete facet programs must be reviewed and registered in controller source.
# The present registry is intentionally empty, so no generic executable plus a
# caller-authored manifest can turn structural scaffolding into physical proof.
PROGRAM_ADAPTER_REGISTRY = controller.PROGRAM_ADAPTER_REGISTRY


def _adapter_for_scope(
    *, facet: str, segment: Any, model: Any,
) -> dict[str, str] | None:
    """Select an adapter only from its exact signed segment/model domain."""
    if (segment, model) == ("sub-120b-doctor", "Doctor-V5"):
        return PROGRAM_ADAPTER_REGISTRY.get(facet)
    if isinstance(segment, str) and isinstance(model, str):
        return controller.POST120_PROGRAM_ADAPTER_REGISTRY.get(
            (segment, model, facet)
        )
    return None


class AdmissionBlocked(RuntimeError):
    """A transient release, lease, collector, or resource gate is closed."""


class ContractError(ValueError):
    """A frozen launch artifact is malformed, stale, or unsafe."""


class AnchoredDirectory:
    """An output directory addressed only through a retained directory FD.

    ``display_path`` is evidence metadata, never write authority.  All creates,
    reads, chmods, links, and child traversal use *at syscalls with O_NOFOLLOW.
    Renaming the directory or replacing any pathname ancestor therefore cannot
    redirect physical evidence into an attacker-controlled tree; the final
    pathname/inode recheck fails closed if the visible path moved.
    """

    def __init__(self, descriptor: int, display_path: pathlib.Path):
        self.fd = descriptor
        self.display_path = display_path.absolute()
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            os.close(descriptor)
            raise ContractError(f"anchored output is not a directory: {display_path}")
        self.identity = (int(info.st_dev), int(info.st_ino))

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "AnchoredDirectory":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    @staticmethod
    def _name(name: str) -> str:
        if not isinstance(name, str) or not name or name in {".", ".."} \
                or "/" in name or "\x00" in name:
            raise ContractError(f"unsafe anchored output component: {name!r}")
        return name

    def child(self, name: str, *, create: bool = False, mode: int = 0o700) -> "AnchoredDirectory":
        name = self._name(name)
        if create:
            try:
                os.mkdir(name, mode=mode, dir_fd=self.fd)
            except FileExistsError as exc:
                raise ContractError(f"anchored output child already exists: {name}") from exc
        flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(name, flags, dir_fd=self.fd)
        except OSError as exc:
            raise ContractError(f"cannot open anchored output child {name}: {exc}") from exc
        return AnchoredDirectory(descriptor, self.display_path / name)

    def exists(self, name: str) -> bool:
        name = self._name(name)
        try:
            os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    def write_exclusive(self, name: str, raw: bytes, *, immutable: bool = True) -> None:
        name = self._name(name)
        flags = (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL |
            getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(name, flags, 0o600, dir_fd=self.fd)
        try:
            written = 0
            while written < len(raw):
                count = os.write(descriptor, raw[written:])
                if count <= 0:
                    raise OSError("short anchored output write")
                written += count
            os.fsync(descriptor)
            if immutable:
                os.fchmod(descriptor, 0o400)
        finally:
            os.close(descriptor)

    def write_json(self, name: str, value: Any, *, immutable: bool = True) -> None:
        self.write_exclusive(name, _json_bytes(value), immutable=immutable)

    def open_file(
        self, name: str, flags: int, *, mode: int = 0o600,
    ) -> int:
        name = self._name(name)
        return os.open(
            name, flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            mode, dir_fd=self.fd,
        )

    def chmod(self, name: str, mode: int) -> None:
        descriptor = self.open_file(name, os.O_RDONLY)
        try:
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)

    def size(self, name: str) -> int:
        descriptor = self.open_file(name, os.O_RDONLY)
        try:
            return int(os.fstat(descriptor).st_size)
        finally:
            os.close(descriptor)

    def artifact(
        self, name: str, *, allow_empty: bool = False,
        maximum_bytes: int | None = None,
    ) -> dict[str, Any]:
        name = self._name(name)
        descriptor = self.open_file(name, os.O_RDONLY)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
                    or not allow_empty and before.st_size <= 0:
                raise ContractError(f"anchored artifact is unsafe: {name}")
            if maximum_bytes is not None and before.st_size > maximum_bytes:
                raise ContractError(f"anchored artifact exceeds its bound: {name}")
            digest = hashlib.sha256()
            observed = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                observed += len(chunk)
                if maximum_bytes is not None and observed > maximum_bytes:
                    raise ContractError(f"anchored artifact exceeds its bound: {name}")
                digest.update(chunk)
            after = os.fstat(descriptor)
            identity = lambda row: (
                row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
                row.st_ctime_ns, row.st_nlink,
            )
            if identity(before) != identity(after) or observed != after.st_size:
                raise ContractError(f"anchored artifact changed while hashing: {name}")
        finally:
            os.close(descriptor)
        return {
            "path": str(self.display_path / name),
            "sha256": digest.hexdigest(), "bytes": observed,
        }

    def load_json(self, name: str, *, maximum_bytes: int = MAX_JSON_BYTES) -> Any:
        name = self._name(name)
        descriptor = self.open_file(name, os.O_RDONLY)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
                    or before.st_size <= 0 or before.st_size > maximum_bytes:
                raise ContractError(f"anchored JSON is unsafe or oversized: {name}")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > maximum_bytes:
                    raise ContractError(f"anchored JSON exceeds its bound: {name}")
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                    before.st_ctime_ns, before.st_nlink) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                    after.st_ctime_ns, after.st_nlink) or total != after.st_size:
                raise ContractError(f"anchored JSON changed while reading: {name}")
        finally:
            os.close(descriptor)
        try:
            return json.loads(b"".join(chunks))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"invalid anchored JSON {name}: {exc}") from exc

    def revalidate_visible_path(self) -> None:
        try:
            other = _open_absolute_directory(self.display_path, create=False)
        except (OSError, ContractError) as exc:
            raise ContractError(
                f"anchored output pathname moved or disappeared: {self.display_path}: {exc}"
            ) from exc
        try:
            if other.identity != self.identity:
                raise ContractError(
                    f"anchored output pathname was replaced: {self.display_path}"
                )
        finally:
            other.close()


def _open_absolute_directory(path: pathlib.Path, *, create: bool) -> AnchoredDirectory:
    absolute = path.absolute()
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise ContractError("anchored directory path must be normalized and absolute")
    flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
        getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open("/", flags)
    current = pathlib.Path("/")
    try:
        for part in absolute.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            current /= part
    except BaseException:
        os.close(descriptor)
        raise
    return AnchoredDirectory(descriptor, absolute)


def _prepare_output_target(path: pathlib.Path) -> tuple[AnchoredDirectory, str]:
    if not path.is_absolute() or path.suffix != ".json":
        raise ContractError("output must be an absolute JSON path")
    root_lexical = EXECUTION_ROOT.absolute()
    try:
        relative = path.absolute().relative_to(root_lexical)
    except ValueError as exc:
        raise ContractError("output path escapes the staged physical-A/B execution root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ContractError("output path has an unsafe relative component")
    root = _open_absolute_directory(root_lexical, create=True)
    current = root
    try:
        for part in relative.parts[:-1]:
            if current.exists(part):
                child = current.child(part)
            else:
                child = current.child(part, create=True)
            if current is not root:
                current.close()
            current = child
        final_name = AnchoredDirectory._name(relative.parts[-1])
        if current.exists(final_name):
            raise ContractError("output already exists; result overwrite is forbidden")
        if current is root:
            root = None
        return current, final_name
    except BaseException:
        if current is not root:
            current.close()
        raise
    finally:
        if root is not None:
            root.close()


def canonical_sha256(value: Any) -> str:
    return controller.canonical_sha256(value)


def _stamp(value: dict[str, Any], field: str) -> dict[str, Any]:
    stamped = copy.deepcopy(value)
    stamped.pop(field, None)
    stamped[field] = canonical_sha256(stamped)
    return stamped


def _hex(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _integer(value: Any, *, minimum: int = 0, maximum: int | None = None) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool)
        and value >= minimum and (maximum is None or value <= maximum)
    )


def _artifact_identity(
    path: pathlib.Path, *, executable: bool = False, allow_empty: bool = False,
    maximum_bytes: int | None = None,
) -> dict[str, Any]:
    """Hash a stable regular file through a no-follow descriptor."""
    if path.is_symlink():
        raise ContractError(f"artifact path is a symlink: {path}")
    path = path.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ContractError(f"artifact is not a single-link regular file: {path}")
        if not allow_empty and before.st_size <= 0:
            raise ContractError(f"artifact is empty: {path}")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise ContractError(f"artifact exceeds its bounded size: {path}")
        if executable and not os.access(path, os.X_OK):
            raise ContractError(f"program is not executable: {path}")
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if maximum_bytes is not None and observed > maximum_bytes:
                raise ContractError(f"artifact exceeds its bounded size: {path}")
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
            row.st_ctime_ns, row.st_nlink,
        )
        if identity(before) != identity(after) or observed != after.st_size:
            raise ContractError(f"artifact changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": observed}


def _artifact_errors(
    value: Any, *, label: str, verify_files: bool, executable: bool = False,
    allow_empty: bool = False, maximum_bytes: int | None = None,
) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}:
        return [f"{label} must contain exactly path/sha256/bytes"]
    errors: list[str] = []
    path = value.get("path")
    if not isinstance(path, str) or not pathlib.Path(path).is_absolute():
        errors.append(f"{label}.path must be absolute")
    if not _hex(value.get("sha256")):
        errors.append(f"{label}.sha256 is invalid")
    size = value.get("bytes")
    if not _integer(size, minimum=0) or (not allow_empty and size == 0):
        errors.append(f"{label}.bytes is invalid")
    if maximum_bytes is not None and isinstance(size, int) and size > maximum_bytes:
        errors.append(f"{label}.bytes exceeds the limit")
    if verify_files and not errors:
        try:
            observed = _artifact_identity(
                pathlib.Path(path), executable=executable, allow_empty=allow_empty,
                maximum_bytes=maximum_bytes,
            )
            if observed != value:
                errors.append(f"{label} differs from the frozen artifact")
        except (OSError, ContractError) as exc:
            errors.append(f"{label} cannot be verified: {exc}")
    return errors


def _load_stable_json(path: pathlib.Path, *, maximum_bytes: int = MAX_JSON_BYTES) -> Any:
    if path.is_symlink():
        raise ContractError(f"JSON artifact is a symlink: {path}")
    resolved = path.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
                or before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ContractError(f"JSON artifact is unsafe or exceeds its bound: {path}")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ContractError(f"JSON artifact exceeds its bound: {path}")
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
            row.st_ctime_ns, row.st_nlink,
        )
        if identity(before) != identity(after) or observed != after.st_size:
            raise ContractError(f"JSON artifact changed while being read: {path}")
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        return json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON artifact {path}: {exc}") from exc


def _hash_errors(value: Any, field: str, *, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} must be an object"]
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop(field, None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        return [f"{label}.{field} mismatch"]
    return []


def _validate_environment(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} must be an exact string map"]
    errors: list[str] = []
    for key, item in value.items():
        if not isinstance(key, str) or not key or "\x00" in key \
                or not isinstance(item, str) or "\x00" in item:
            errors.append(f"{label} contains a non-string or NUL value")
            continue
        if key in BANNED_ENV_KEYS or key.startswith(BANNED_ENV_PREFIXES):
            errors.append(f"{label} attempts to set protected environment key {key}")
    return errors


def _placeholder_errors(argv: Any, *, expected: set[str], label: str) -> list[str]:
    if not isinstance(argv, list) or not 1 <= len(argv) <= MAX_ARGV_ITEMS:
        return [f"{label}.argv must be a bounded non-empty list"]
    errors: list[str] = []
    observed: list[str] = []
    total = 0
    for index, token in enumerate(argv):
        if not isinstance(token, str) or not token or "\x00" in token:
            errors.append(f"{label}.argv[{index}] is not a safe string")
            continue
        total += len(token.encode("utf-8"))
        observed.extend(PLACEHOLDER.findall(token))
    if total > MAX_ARG_BYTES:
        errors.append(f"{label}.argv exceeds the byte limit")
    if set(observed) != expected or len(observed) != len(expected):
        errors.append(f"{label}.argv placeholder coverage is not exact")
    if any(token not in expected for token in observed):
        errors.append(f"{label}.argv contains an unknown placeholder")
    return errors


def validate_argv_manifest(
    value: Any, *, role: str, program_sha256: str,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{role} argv manifest is absent"]
    expected = {
        "schema", "role", "program_sha256", "program_abi", "abi_reviewed",
        "direct_exec", "writes_confined_to_dynamic_paths", "argv",
        "placeholders", "environment", "cwd", "stdin", "shell",
        "mutates_live_doctor", "mutates_runtime_defaults", "deletes_sources",
        "manifest_sha256",
    }
    errors: list[str] = []
    if set(value) != expected:
        errors.append(f"{role} argv manifest fields are incomplete or unexpected")
    if value.get("schema") != ARGV_SCHEMA or value.get("role") != role \
            or value.get("program_sha256") != program_sha256:
        errors.append(f"{role} argv schema/role/program binding is invalid")
    if value.get("program_abi") != "hawking.doctor_v5_physical_ab_program.v1" \
            or value.get("abi_reviewed") is not True \
            or value.get("direct_exec") is not True \
            or value.get("writes_confined_to_dynamic_paths") is not True:
        errors.append(f"{role} concrete direct-exec program ABI is unavailable or unreviewed")
    errors.extend(_placeholder_errors(value.get("argv"), expected=ARM_PLACEHOLDERS, label=role))
    if value.get("placeholders") != sorted(ARM_PLACEHOLDERS):
        errors.append(f"{role} placeholder declaration differs from the exact ABI")
    errors.extend(_validate_environment(value.get("environment"), label=f"{role}.environment"))
    if value.get("cwd") != str(ROOT) or value.get("stdin") != "devnull" \
            or value.get("shell") is not False:
        errors.append(f"{role} execution must use workspace cwd, devnull stdin, and no shell")
    if value.get("mutates_live_doctor") is not False \
            or value.get("mutates_runtime_defaults") is not False \
            or value.get("deletes_sources") is not False:
        errors.append(f"{role} ABI permits a forbidden mutation")
    errors.extend(_hash_errors(value, "manifest_sha256", label=f"{role} argv manifest"))
    return errors


def validate_collector_argv(value: Any, *, program_sha256: str) -> list[str]:
    if not isinstance(value, dict):
        return ["collector argv manifest is absent"]
    expected = {
        "schema", "program_sha256", "collector_abi", "abi_reviewed", "argv",
        "placeholders", "environment", "cwd", "stdin", "shell",
        "inherits_shared_heavy_lease", "opens_heavy_lease",
        "launches_benchmark_with_shell", "manifest_sha256",
    }
    errors: list[str] = []
    if set(value) != expected:
        errors.append("collector argv manifest fields are incomplete or unexpected")
    if value.get("schema") != COLLECTOR_ARGV_SCHEMA \
            or value.get("program_sha256") != program_sha256:
        errors.append("collector argv schema/program binding is invalid")
    if value.get("collector_abi") != "hawking.doctor_v5_physical_counter_barrier.v1" \
            or value.get("abi_reviewed") is not True:
        errors.append("concrete collector barrier ABI is unavailable or unreviewed")
    errors.extend(_placeholder_errors(
        value.get("argv"), expected=COLLECTOR_PLACEHOLDERS, label="collector",
    ))
    if value.get("argv") != EXACT_COUNTER_BARRIER_ARGV:
        errors.append("collector argv is not the exact reviewed generic barrier ABI")
    if value.get("placeholders") != sorted(COLLECTOR_PLACEHOLDERS):
        errors.append("collector placeholder declaration differs from the exact ABI")
    errors.extend(_validate_environment(value.get("environment"), label="collector.environment"))
    if value.get("cwd") != str(ROOT) or value.get("stdin") != "devnull" \
            or value.get("shell") is not False:
        errors.append("collector must use workspace cwd, devnull stdin, and no shell")
    if value.get("inherits_shared_heavy_lease") is not True \
            or value.get("opens_heavy_lease") is not False \
            or value.get("launches_benchmark_with_shell") is not False:
        errors.append("collector lease/no-shell ABI is unsafe")
    errors.extend(_hash_errors(value, "manifest_sha256", label="collector argv manifest"))
    return errors


def validate_input_manifest(value: Any, *, verify_files: bool) -> list[str]:
    if not isinstance(value, dict):
        return ["input manifest is absent"]
    expected = {
        "schema", "artifacts", "frozen", "immutable", "parent_sources_retained",
        "rate_or_branch_dependent_work_reused", "manifest_sha256",
    }
    errors: list[str] = []
    if set(value) != expected or value.get("schema") != INPUT_SCHEMA:
        errors.append("input manifest fields/schema are incomplete or unexpected")
    if value.get("frozen") is not True or value.get("immutable") is not True \
            or value.get("parent_sources_retained") is not True \
            or value.get("rate_or_branch_dependent_work_reused") is not False:
        errors.append("input manifest weakens frozen source or reuse isolation")
    rows = value.get("artifacts")
    if not isinstance(rows, list) or not rows:
        errors.append("input manifest must contain at least one exact artifact")
        rows = []
    paths: list[Any] = []
    for index, row in enumerate(rows):
        errors.extend(_artifact_errors(
            row, label=f"input artifacts[{index}]", verify_files=verify_files,
        ))
        paths.append(row.get("path") if isinstance(row, dict) else None)
    if len(set(paths)) != len(paths):
        errors.append("input manifest reuses an artifact path")
    errors.extend(_hash_errors(value, "manifest_sha256", label="input manifest"))
    return errors


def validate_collector_authority(
    value: Any, *, plan: dict[str, Any], verify_files: bool,
) -> list[str]:
    if not isinstance(value, dict):
        return ["collector authority is absent"]
    expected = {
        "schema", "plan_sha256", "source_manifest_sha256", "collector_program",
        "collector_argv_manifest", "collector_config_sha256", "execution_abi_available",
        "directly_measured", "estimated", "process_attributed", "phase_attributed",
        "required_counter_fields", "counter_payload_schema", "attestation_schema",
        "inherited_shared_heavy_lease_required", "opens_heavy_lease", "shell",
        "authority_sha256",
    }
    errors: list[str] = []
    if set(value) != expected or value.get("schema") != COLLECTOR_AUTHORITY_SCHEMA:
        errors.append("collector authority fields/schema are incomplete or unexpected")
    if value.get("plan_sha256") != plan.get("plan_sha256") \
            or value.get("source_manifest_sha256") \
            != plan.get("source_manifest", {}).get("manifest_sha256"):
        errors.append("collector authority is not bound to the exact controller plan/source")
    errors.extend(_artifact_errors(
        value.get("collector_program"), label="collector program",
        verify_files=verify_files, executable=True,
    ))
    try:
        expected_barrier = _artifact_identity(
            pathlib.Path(counter_barrier.__file__), executable=True,
        )
    except (OSError, ContractError) as exc:
        errors.append(f"reviewed generic counter barrier is unavailable: {exc}")
    else:
        if value.get("collector_program") != expected_barrier:
            errors.append("collector authority does not bind the reviewed generic counter barrier")
    errors.extend(_artifact_errors(
        value.get("collector_argv_manifest"), label="collector argv artifact",
        verify_files=verify_files, maximum_bytes=MAX_JSON_BYTES,
    ))
    if value.get("collector_config_sha256") != counter_barrier.build_config()["config_sha256"]:
        errors.append("collector authority config differs from the current collector contract")
    if value.get("execution_abi_available") is not True \
            or value.get("directly_measured") is not True \
            or value.get("estimated") is not False \
            or value.get("process_attributed") is not True \
            or value.get("phase_attributed") is not True:
        errors.append("collector authority does not prove direct attributed counters")
    if value.get("required_counter_fields") != list(COUNTER_FIELDS) \
            or value.get("counter_payload_schema") != controller.COUNTER_SCHEMA \
            or value.get("attestation_schema") != COUNTER_ATTESTATION_SCHEMA:
        errors.append("collector output schemas/domains differ from the executor ABI")
    if value.get("inherited_shared_heavy_lease_required") is not True \
            or value.get("opens_heavy_lease") is not False \
            or value.get("shell") is not False:
        errors.append("collector authority weakens inherited-lease/no-shell policy")
    errors.extend(_hash_errors(value, "authority_sha256", label="collector authority"))
    return errors


def build_randomized_order(seed_sha256: str, repeats: int) -> list[str]:
    """Derive paired/interleaved order without depending on random-module versions."""
    if not _hex(seed_sha256) or not _integer(repeats, minimum=controller.MIN_REPEATS, maximum=MAX_REPEATS):
        raise ContractError("random seed/repeat count is invalid")
    repeat_order = sorted(
        range(repeats),
        key=lambda index: hashlib.sha256(
            f"{seed_sha256}:pair:{index}".encode("ascii")
        ).digest(),
    )
    order: list[str] = []
    first_roles: set[str] = set()
    for index in repeat_order:
        bit = hashlib.sha256(f"{seed_sha256}:arm:{index}".encode("ascii")).digest()[0] & 1
        roles = ("baseline", "candidate") if bit == 0 else ("candidate", "baseline")
        first_roles.add(roles[0])
        order.extend(f"{role}:{index}" for role in roles)
    if first_roles != {"baseline", "candidate"}:
        raise ContractError("random seed does not counterbalance first-arm order")
    return order


def _limits_errors(value: Any) -> list[str]:
    expected = {
        "timeout_seconds", "collector_ready_timeout_seconds", "max_stdout_bytes",
        "max_stderr_bytes", "max_output_bytes", "max_scientific_receipt_bytes",
        "max_facet_payload_bytes", "maximum_swap_used_bytes",
        "maximum_swap_growth_bytes", "minimum_disk_free_bytes",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return ["run limits are incomplete or unexpected"]
    errors: list[str] = []
    bounds = {
        "timeout_seconds": (1, 86_400),
        "collector_ready_timeout_seconds": (1, 300),
        "max_stdout_bytes": (1, 512 * 1024 * 1024),
        "max_stderr_bytes": (1, 512 * 1024 * 1024),
        "max_output_bytes": (1, MAX_JSON_BYTES),
        "max_scientific_receipt_bytes": (1, MAX_JSON_BYTES),
        "max_facet_payload_bytes": (1, MAX_JSON_BYTES),
        "maximum_swap_used_bytes": (0, 1 << 50),
        "maximum_swap_growth_bytes": (0, 1 << 50),
        "minimum_disk_free_bytes": (controller.MIN_TOTAL_DISK_ADMISSION_BYTES, 1 << 60),
    }
    for field, (minimum, maximum) in bounds.items():
        if not _integer(value.get(field), minimum=minimum, maximum=maximum):
            errors.append(f"run_limits.{field} is outside its exact safe bound")
    return errors


def validate_launch_contract(
    value: Any, *, plan: dict[str, Any], facet: str, verify_files: bool,
    execution_scope: dict[str, Any] | None = None,
) -> list[str]:
    if not isinstance(value, dict):
        return ["launch contract is absent"]
    expected = {
        "schema", "plan_sha256", "source_manifest_sha256", "executor_source_sha256",
        "facet", "baseline_program", "baseline_argv_manifest", "candidate_program",
        "candidate_argv_manifest", "input_manifest", "execution_scope",
        "collector_authority", "pairing", "run_limits", "output_policy",
        "mutation_policy", "contract_sha256",
    }
    errors: list[str] = []
    if set(value) != expected or value.get("schema") != LAUNCH_SCHEMA:
        errors.append("launch contract fields/schema are incomplete or unexpected")
    if facet not in controller.FACETS or value.get("facet") != facet:
        errors.append("launch facet is outside the exact controller allowlist")
    if value.get("plan_sha256") != plan.get("plan_sha256") \
            or value.get("source_manifest_sha256") \
            != plan.get("source_manifest", {}).get("manifest_sha256"):
        errors.append("launch contract is not bound to the exact controller plan/source")
    runner = plan.get("executor_manifest", {}).get("runner_source", {})
    if value.get("executor_source_sha256") != runner.get("sha256"):
        errors.append("launch contract is not bound to the reviewed executor source")
    segment = execution_scope.get("segment") if isinstance(execution_scope, dict) else None
    model = execution_scope.get("model") if isinstance(execution_scope, dict) else None
    adapter = _adapter_for_scope(facet=facet, segment=segment, model=model)
    if not isinstance(adapter, dict):
        errors.append(f"no reviewed concrete physical program adapter is registered for {facet}")
    else:
        expected_adapter_bindings = {
            "baseline_program_sha256": value.get("baseline_program", {}).get("sha256"),
            "baseline_argv_manifest_sha256": value.get("baseline_argv_manifest", {}).get("sha256"),
            "candidate_program_sha256": value.get("candidate_program", {}).get("sha256"),
            "candidate_argv_manifest_sha256": value.get("candidate_argv_manifest", {}).get("sha256"),
        }
        if set(adapter) != {
            "adapter_id", *expected_adapter_bindings, "launch_contract_sha256",
            "execution_scope_sha256", "scientific_receipt_schema",
            "scientific_validator",
        } or any(adapter.get(field) != expected_value for field, expected_value in expected_adapter_bindings.items()) \
                or adapter.get("launch_contract_sha256") != value.get("contract_sha256") \
                or adapter.get("execution_scope_sha256") \
                != value.get("execution_scope", {}).get("sha256") \
                or adapter.get("scientific_receipt_schema") != SCIENTIFIC_RECEIPT_SCHEMA \
                or adapter.get("scientific_validator") \
                != "doctor_v5_physical_ab_executor.validate_scientific_receipt.v1" \
                or not isinstance(adapter.get("adapter_id"), str) \
                or not adapter["adapter_id"]:
            errors.append(f"{facet} launch artifacts differ from its source-reviewed adapter")
    for field, executable in (
        ("baseline_program", True), ("baseline_argv_manifest", False),
        ("candidate_program", True), ("candidate_argv_manifest", False),
        ("input_manifest", False), ("execution_scope", False),
        ("collector_authority", False),
    ):
        errors.extend(_artifact_errors(
            value.get(field), label=f"launch.{field}", verify_files=verify_files,
            executable=executable, maximum_bytes=None if executable else MAX_JSON_BYTES,
        ))
    pairing = value.get("pairing")
    if not isinstance(pairing, dict) or set(pairing) != {
        "warmups_per_arm", "repeats_per_arm", "random_seed_sha256",
        "randomized_interleaved", "order", "order_sha256",
    }:
        errors.append("paired protocol is incomplete or unexpected")
    else:
        repeats = pairing.get("repeats_per_arm")
        seed = pairing.get("random_seed_sha256")
        if pairing.get("warmups_per_arm") != 1 \
                or not _integer(repeats, minimum=controller.MIN_REPEATS, maximum=MAX_REPEATS) \
                or pairing.get("randomized_interleaved") is not True:
            errors.append("paired protocol does not require one warmup and >=5 measured pairs")
        else:
            try:
                expected_order = build_randomized_order(seed, repeats)
            except ContractError as exc:
                errors.append(str(exc))
            else:
                if pairing.get("order") != expected_order \
                        or pairing.get("order_sha256") != canonical_sha256(expected_order):
                    errors.append("paired order is not the exact hash-derived randomized order")
    errors.extend(_limits_errors(value.get("run_limits")))
    if value.get("output_policy") != {
        "root": str(EXECUTION_ROOT),
        "exclusive_create": True,
        "immutable_sidecars": True,
        "atomic_final_receipt": True,
        "source_deletion_permitted": False,
    }:
        errors.append("output policy does not confine immutable exclusive writes")
    if value.get("mutation_policy") != {
        "live_doctor_mutation": False,
        "completed_evidence_mutation": False,
        "runtime_default_mutation": False,
        "result_overwrite": False,
        "source_deletion": False,
    }:
        errors.append("launch contract permits a forbidden mutation")
    errors.extend(_hash_errors(value, "contract_sha256", label="launch contract"))
    return errors


def validate_release_authority(
    value: Any, *, plan: dict[str, Any], collector_authority_sha256: str,
    verify_files: bool,
) -> list[str]:
    if not isinstance(value, dict):
        return ["launch release authority is absent"]
    expected = {
        "schema", "plan_sha256", "source_manifest_sha256", "observer_state_sha256",
        "final_interpretation_ready", "active_heavy_owner_count", "owner_inventory",
        "shared_heavy_lease", "collector_authority_sha256", "issued_at_unix_ns",
        "expires_at_unix_ns", "authority_sha256",
    }
    errors: list[str] = []
    if set(value) != expected or value.get("schema") != RELEASE_AUTHORITY_SCHEMA:
        errors.append("launch release authority fields/schema are incomplete or unexpected")
    if value.get("plan_sha256") != plan.get("plan_sha256") \
            or value.get("source_manifest_sha256") \
            != plan.get("source_manifest", {}).get("manifest_sha256"):
        errors.append("launch release authority is not bound to the controller plan/source")
    if not _hex(value.get("observer_state_sha256")) \
            or value.get("final_interpretation_ready") is not True \
            or value.get("active_heavy_owner_count") != 0:
        errors.append("launch release authority lacks final-ready owner-free evidence")
    errors.extend(_artifact_errors(
        value.get("owner_inventory"), label="release owner inventory",
        verify_files=verify_files, maximum_bytes=MAX_JSON_BYTES,
    ))
    lease = value.get("shared_heavy_lease")
    if not isinstance(lease, dict) or set(lease) != {
        "lock_file", "st_dev", "st_ino", "fd_env", "inherited_descriptor",
        "held", "owners_rechecked_under_lock", "acquired_at_unix_ns",
    }:
        errors.append("launch shared-heavy-lease authority is malformed")
    else:
        errors.extend(_artifact_errors(
            lease.get("lock_file"), label="release lease lock file",
            verify_files=verify_files, allow_empty=True,
        ))
        if not _integer(lease.get("st_dev")) or not _integer(lease.get("st_ino")) \
                or lease.get("fd_env") != HEAVY_LEASE_FD_ENV \
                or lease.get("inherited_descriptor") is not True \
                or lease.get("held") is not True \
                or lease.get("owners_rechecked_under_lock") is not True \
                or not _integer(lease.get("acquired_at_unix_ns"), minimum=1):
            errors.append("launch shared-heavy-lease identity/state is invalid")
    if value.get("collector_authority_sha256") != collector_authority_sha256:
        errors.append("release authority is not bound to the exact collector authority")
    issued, expires = value.get("issued_at_unix_ns"), value.get("expires_at_unix_ns")
    if not _integer(issued, minimum=1) or not _integer(expires, minimum=1) \
            or isinstance(issued, int) and isinstance(expires, int) and expires <= issued:
        errors.append("launch release authority interval is invalid")
    errors.extend(_hash_errors(value, "authority_sha256", label="launch release authority"))
    return errors


def _same_artifact(reference: Any, path: pathlib.Path, *, executable: bool = False) -> bool:
    try:
        return reference == _artifact_identity(path, executable=executable)
    except (OSError, ContractError):
        return False


def validate_launch_bundle(
    *, plan_path: pathlib.Path, contract_path: pathlib.Path,
    release_authority_path: pathlib.Path, baseline_program: pathlib.Path,
    baseline_argv_path: pathlib.Path, candidate_program: pathlib.Path,
    candidate_argv_path: pathlib.Path, input_manifest_path: pathlib.Path,
    execution_scope_path: pathlib.Path, collector_authority_path: pathlib.Path,
    facet: str, verify_files: bool = True,
) -> tuple[list[str], dict[str, Any] | None]:
    """Validate every frozen launch artifact without executing a process."""
    errors: list[str] = []
    try:
        plan = _load_stable_json(plan_path)
        expected_plan = controller.build_plan()
        if plan != expected_plan:
            errors.append("supplied controller plan is stale, altered, or not exact")
        contract = _load_stable_json(contract_path)
        release = _load_stable_json(release_authority_path)
        baseline_argv = _load_stable_json(baseline_argv_path)
        candidate_argv = _load_stable_json(candidate_argv_path)
        inputs = _load_stable_json(input_manifest_path)
        execution_scope = _load_stable_json(execution_scope_path)
        collector_authority = _load_stable_json(collector_authority_path)
        collector_argv_path_value = (
            collector_authority.get("collector_argv_manifest", {}).get("path")
            if isinstance(collector_authority, dict) else None
        )
        if not isinstance(collector_argv_path_value, str):
            raise ContractError("collector argv artifact path is absent")
        collector_argv_path = pathlib.Path(collector_argv_path_value)
        collector_argv = _load_stable_json(collector_argv_path)
    except (OSError, ContractError) as exc:
        return [str(exc)], None

    errors.extend(validate_launch_contract(
        contract, plan=plan, facet=facet, verify_files=verify_files,
        execution_scope=execution_scope,
    ))
    errors.extend(validate_collector_authority(
        collector_authority, plan=plan, verify_files=verify_files,
    ))
    collector_program_sha = (
        collector_authority.get("collector_program", {}).get("sha256")
        if isinstance(collector_authority, dict) else ""
    )
    errors.extend(validate_collector_argv(
        collector_argv, program_sha256=collector_program_sha,
    ))
    errors.extend(validate_argv_manifest(
        baseline_argv, role="baseline",
        program_sha256=contract.get("baseline_program", {}).get("sha256", ""),
    ))
    errors.extend(validate_argv_manifest(
        candidate_argv, role="candidate",
        program_sha256=contract.get("candidate_program", {}).get("sha256", ""),
    ))
    errors.extend(validate_input_manifest(inputs, verify_files=verify_files))
    errors.extend(controller._physical_execution_scope_errors(
        execution_scope, facet=facet, verify_files=verify_files,
    ))
    errors.extend(validate_release_authority(
        release, plan=plan,
        collector_authority_sha256=collector_authority.get("authority_sha256", ""),
        verify_files=verify_files,
    ))

    path_bindings = (
        ("baseline_program", baseline_program, True),
        ("baseline_argv_manifest", baseline_argv_path, False),
        ("candidate_program", candidate_program, True),
        ("candidate_argv_manifest", candidate_argv_path, False),
        ("input_manifest", input_manifest_path, False),
        ("execution_scope", execution_scope_path, False),
        ("collector_authority", collector_authority_path, False),
    )
    for field, path, executable in path_bindings:
        if not _same_artifact(contract.get(field), path, executable=executable):
            errors.append(f"CLI {field} differs from its exact launch-contract identity")
    if collector_authority.get("collector_argv_manifest") \
            != _artifact_identity(collector_argv_path):
        errors.append("collector argv content differs from collector authority")
    collector_program_path = pathlib.Path(
        collector_authority.get("collector_program", {}).get("path", "")
    )
    if not _same_artifact(
        collector_authority.get("collector_program"), collector_program_path,
        executable=True,
    ):
        errors.append("collector program differs from collector authority")
    if release.get("observer_state_sha256"):
        try:
            observer = _load_stable_json(OBSERVER_PATH)
        except (OSError, ContractError) as exc:
            errors.append(f"current Doctor observer cannot be verified: {exc}")
        else:
            errors.extend(controller._observer_errors(observer))
            if observer.get("state_sha256") != release.get("observer_state_sha256"):
                errors.append("release authority is stale relative to the Doctor observer")

    if errors:
        return list(dict.fromkeys(errors)), None
    segment = execution_scope.get("segment")
    model = execution_scope.get("model")
    adapter = _adapter_for_scope(facet=facet, segment=segment, model=model)
    return [], {
        "plan": plan, "contract": contract, "release_authority": release,
        "baseline_program": baseline_program.resolve(),
        "baseline_argv": baseline_argv,
        "candidate_program": candidate_program.resolve(),
        "candidate_argv": candidate_argv,
        "input_manifest_path": input_manifest_path.resolve(),
        "input_manifest": inputs,
        "execution_scope_path": execution_scope_path.resolve(),
        "execution_scope": execution_scope,
        "program_adapter": copy.deepcopy(adapter),
        "collector_authority_path": collector_authority_path.resolve(),
        "collector_authority": collector_authority,
        "collector_program": collector_program_path.resolve(),
        "collector_argv": collector_argv,
    }


def _collector_runtime_status() -> dict[str, Any]:
    """Cheap status from the exact generic external-arm counter barrier."""
    config = counter_barrier.build_config()
    status = counter_barrier.build_status()
    return {
        "config_sha256": config.get("config_sha256"),
        "execution_ready": status.get("execution_ready") is True,
        "blockers": list(status.get("blockers", [])),
    }


def _read_observer() -> Any:
    try:
        return _load_stable_json(OBSERVER_PATH, maximum_bytes=64 * 1024 * 1024)
    except (OSError, ContractError):
        return None


def build_status(
    *, observer: Any | None = None, owners: list[dict[str, Any]] | None = None,
    collector_status: dict[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a read-only admission view; never opens a lease or hashes inputs."""
    plan = controller.build_plan()
    controller.refresh_program_adapter_registries(plan=plan)
    observer = _read_observer() if observer is None else observer
    owner_rows = spec_reentry_scaffold.active_heavy_owners() if owners is None else owners
    counter = _collector_runtime_status() if collector_status is None else collector_status
    environment = os.environ if env is None else env
    blockers = controller._observer_errors(observer)
    if owner_rows:
        blockers.append(f"{len(owner_rows)} heavy owner(s) remain")
    if counter.get("execution_ready") is not True:
        blockers.extend(str(row) for row in counter.get("blockers", ["counter authority is closed"]))
    runner = plan.get("executor_manifest", {})
    if runner.get("trusted_executor_available") is not True:
        blockers.append("reviewed physical A/B executor source is not plan-bound")
    missing_adapters = [
        facet for facet in controller.FACETS
        if facet not in PROGRAM_ADAPTER_REGISTRY
    ]
    blockers.extend(
        f"{facet}: reviewed concrete baseline/candidate program adapter is absent"
        for facet in missing_adapters
    )
    blockers.extend(controller.PROGRAM_ADAPTER_REGISTRY_ERRORS)
    inherited_fd = environment.get(HEAVY_LEASE_FD_ENV)
    try:
        fd = int(inherited_fd) if inherited_fd is not None else -1
        os.fstat(fd)
    except (OSError, TypeError, ValueError):
        blockers.append("an inherited shared-heavy-lease descriptor is absent")
    blockers.append("a concrete hash-bound launch bundle is required")
    value = {
        "schema": STATUS_SCHEMA,
        "executor_config_sha256": build_config()["config_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "facet_allowlist": list(controller.FACETS),
        "final_interpretation_ready": (
            isinstance(observer, dict) and observer.get("final_interpretation_ready") is True
        ),
        "active_heavy_owner_count": len(owner_rows),
        "collector_execution_ready": counter.get("execution_ready") is True,
        "reviewed_program_adapter_facets": sorted(PROGRAM_ADAPTER_REGISTRY),
        "missing_program_adapter_facets": missing_adapters,
        "inherited_shared_heavy_lease_present": not any(
            "inherited shared-heavy-lease" in row for row in blockers
        ),
        "shared_heavy_lease_opened": False,
        "input_artifacts_hashed": False,
        "output_created": False,
        "benchmark_spawned": False,
        "execution_ready": not blockers,
        "blockers": list(dict.fromkeys(blockers)),
    }
    return _stamp(value, "status_sha256")


def build_config() -> dict[str, Any]:
    return _stamp({
        "schema": EXECUTOR_SCHEMA,
        "default_off": True,
        "facet_allowlist": list(controller.FACETS),
        "minimum_randomized_pairs": controller.MIN_REPEATS,
        "shell_permitted": False,
        "acquires_shared_heavy_lease": False,
        "inherited_shared_heavy_lease_required": True,
        "revalidates_before_every_arm": True,
        "direct_counter_authority_required": True,
        "direct_counter_barrier_config_sha256": counter_barrier.build_config()["config_sha256"],
        "program_adapter_registry_facets": sorted(PROGRAM_ADAPTER_REGISTRY),
        "immutable_receipts_required": True,
        "execution_root": str(EXECUTION_ROOT),
        "live_doctor_mutation": False,
        "completed_evidence_mutation": False,
        "runtime_default_mutation": False,
        "source_deletion": False,
        "status_and_dry_run_nonexecuting": True,
    }, "config_sha256")


def build_dry_run(*, facet: str | None = None) -> dict[str, Any]:
    if facet is not None and facet not in controller.FACETS:
        raise ContractError("dry-run facet is outside the controller allowlist")
    status = build_status()
    return _stamp({
        "schema": DRY_RUN_SCHEMA,
        "executor_config_sha256": build_config()["config_sha256"],
        "plan_sha256": status["plan_sha256"],
        "facet": facet,
        "would_execute": False,
        "commands": [],
        "would_open_heavy_lease": False,
        "would_hash_inputs": False,
        "would_create_output": False,
        "would_spawn_benchmark": False,
        "would_mutate_live_doctor": False,
        "would_mutate_runtime_defaults": False,
        "status_sha256": status["status_sha256"],
        "blockers": status["blockers"],
    }, "dry_run_sha256")


def _validate_inherited_lease(
    authority: dict[str, Any], *, env: Mapping[str, str] | None = None,
) -> tuple[int | None, list[str]]:
    """Prove the supplied descriptor owns the existing lock; never acquire it."""
    environment = os.environ if env is None else env
    lease = authority.get("shared_heavy_lease", {})
    errors: list[str] = []
    try:
        fd = int(environment.get(HEAVY_LEASE_FD_ENV, ""))
        inherited = os.fstat(fd)
    except (OSError, TypeError, ValueError):
        return None, ["inherited shared-heavy-lease descriptor is absent or closed"]
    lock_path = pathlib.Path(lease.get("lock_file", {}).get("path", ""))
    try:
        expected_path = HEAVY_LOCK.resolve(strict=True)
        actual_path = lock_path.resolve(strict=True)
        path_stat = os.stat(actual_path, follow_symlinks=False)
    except OSError as exc:
        return None, [f"shared-heavy-lease lock path is unavailable: {exc}"]
    if actual_path != expected_path:
        errors.append("release authority points at a different heavy-lock path")
    identity = (int(inherited.st_dev), int(inherited.st_ino))
    declared = (lease.get("st_dev"), lease.get("st_ino"))
    path_identity = (int(path_stat.st_dev), int(path_stat.st_ino))
    if identity != declared or identity != path_identity:
        errors.append("inherited lease descriptor identity differs from authority/path")
    if not stat.S_ISREG(inherited.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        errors.append("shared-heavy-lease descriptor/path is not regular")
    fresh_fd = -1
    fresh_acquired = False
    try:
        fresh_fd = os.open(actual_path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
        try:
            fcntl.flock(fresh_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fresh_acquired = True
        except BlockingIOError:
            fresh_acquired = False
        if fresh_acquired:
            fcntl.flock(fresh_fd, fcntl.LOCK_UN)
            errors.append("heavy lease is not already held by the inherited descriptor")
        else:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                errors.append("heavy lease belongs to a different open-file description")
    except OSError as exc:
        errors.append(f"heavy lease ownership probe failed: {exc}")
    finally:
        if fresh_fd >= 0:
            os.close(fresh_fd)
    return (fd if not errors else None), errors


def _resource_errors(snapshot: Any, *, limits: dict[str, Any]) -> list[str]:
    if not isinstance(snapshot, dict):
        return ["resource snapshot is absent"]
    errors = [str(row) for row in snapshot.get("errors", [])]
    if snapshot.get("probe_ok") is not True:
        errors.append("resource probe is not complete")
    if snapshot.get("pressure_level") != 1:
        errors.append("memory pressure is not normal")
    if snapshot.get("thermal_state") != 0:
        errors.append("thermal state is not nominal")
    if snapshot.get("power_source") != "AC Power":
        errors.append("AC power is not confirmed")
    swap = snapshot.get("swap_used_bytes")
    if not _integer(swap) or swap > limits.get("maximum_swap_used_bytes", -1):
        errors.append("swap use exceeds the frozen controlled-swap ceiling")
    disk = snapshot.get("disk_free_bytes")
    minimum = max(
        controller.MIN_TOTAL_DISK_ADMISSION_BYTES,
        limits.get("minimum_disk_free_bytes", controller.MIN_TOTAL_DISK_ADMISSION_BYTES),
    )
    if not _integer(disk, minimum=minimum):
        errors.append("free disk violates the authoritative reserve")
    return list(dict.fromkeys(errors))


def _swap_growth_errors(
    before: Any, after: Any, *, maximum_growth_bytes: Any,
) -> list[str]:
    if not isinstance(before, dict) or not isinstance(after, dict) \
            or not _integer(before.get("swap_used_bytes")) \
            or not _integer(after.get("swap_used_bytes")) \
            or not _integer(maximum_growth_bytes):
        return ["arm swap growth samples/bound are unavailable"]
    if after["swap_used_bytes"] - before["swap_used_bytes"] > maximum_growth_bytes:
        return ["arm swap growth exceeded the frozen recovery bound"]
    return []


def _arm_admission(
    bundle: dict[str, Any], *, env: Mapping[str, str] | None = None,
    owner_probe: Any = spec_reentry_scaffold.active_heavy_owners,
    resource_probe: Any = doctor_frontier_worker._resource_snapshot,
) -> dict[str, Any]:
    """Re-prove every mutable gate immediately before an arm."""
    authority = bundle["release_authority"]
    now = time.time_ns()
    if now >= authority["expires_at_unix_ns"]:
        raise AdmissionBlocked("launch release authority expired")
    try:
        observer = _load_stable_json(OBSERVER_PATH, maximum_bytes=64 * 1024 * 1024)
    except (OSError, ContractError) as exc:
        raise AdmissionBlocked(f"Doctor observer cannot be revalidated: {exc}") from exc
    observer_errors = controller._observer_errors(observer)
    if observer_errors or observer.get("state_sha256") != authority["observer_state_sha256"]:
        raise AdmissionBlocked(
            "Doctor final-ready authority changed: " + "; ".join(observer_errors)
        )
    fd, lease_errors = _validate_inherited_lease(authority, env=env)
    if lease_errors or fd is None:
        raise AdmissionBlocked("; ".join(lease_errors))
    owners = owner_probe()
    if owners:
        raise AdmissionBlocked(f"{len(owners)} heavy owner(s) appeared under the inherited lease")
    snapshot = resource_probe(ROOT)
    resource_errors = _resource_errors(snapshot, limits=bundle["contract"]["run_limits"])
    if resource_errors:
        raise AdmissionBlocked("; ".join(resource_errors))
    input_errors = validate_input_manifest(bundle["input_manifest"], verify_files=True)
    if input_errors:
        raise AdmissionBlocked("frozen inputs changed: " + "; ".join(input_errors))
    for role in ("baseline", "candidate"):
        if not _same_artifact(
            bundle["contract"][f"{role}_program"], bundle[f"{role}_program"],
            executable=True,
        ):
            raise AdmissionBlocked(f"{role} program identity changed before an arm")
    return {"observer": observer, "owners": owners, "resource": snapshot, "lease_fd": fd}


def _confined_output(path: pathlib.Path) -> pathlib.Path:
    if not path.is_absolute():
        raise ContractError("output path must be absolute")
    root = EXECUTION_ROOT.resolve(strict=False)
    candidate = path.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ContractError("output path escapes the staged physical-A/B execution root") from exc
    if candidate == root or candidate.suffix != ".json":
        raise ContractError("output must be a JSON file below the execution root")
    cursor = candidate.parent
    while cursor != root.parent and cursor.exists():
        if cursor.is_symlink():
            raise ContractError("output path contains a symlink")
        if cursor == root:
            break
        cursor = cursor.parent
    if candidate.exists() or candidate.is_symlink():
        raise ContractError("output already exists; result overwrite is forbidden")
    return candidate


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_exclusive(path: pathlib.Path, raw: bytes, *, immutable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600,
    )
    try:
        written = 0
        while written < len(raw):
            count = os.write(descriptor, raw[written:])
            if count <= 0:
                raise OSError("short write while sealing receipt")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if immutable:
        os.chmod(path, 0o400)


def _write_json_exclusive(path: pathlib.Path, value: Any, *, immutable: bool = True) -> None:
    _write_exclusive(path, _json_bytes(value), immutable=immutable)


def _atomic_final_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    _write_exclusive(temporary, _json_bytes(value), immutable=False)
    try:
        os.link(temporary, path)
        os.chmod(path, 0o400)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_final_json_at(directory: AnchoredDirectory, name: str, value: Any) -> None:
    """Seal one final receipt without resolving a mutable pathname ancestor."""
    name = AnchoredDirectory._name(name)
    temporary = f".{name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    directory.write_exclusive(temporary, _json_bytes(value), immutable=False)
    try:
        os.link(
            temporary, name, src_dir_fd=directory.fd, dst_dir_fd=directory.fd,
            follow_symlinks=False,
        )
        directory.chmod(name, 0o400)
        os.fsync(directory.fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory.fd)
        except FileNotFoundError:
            pass


def _anchored_preexec(directory_fd: int) -> Any:
    """Return the sole child setup action: fchdir to a retained safe inode."""
    def enter_directory() -> None:
        os.fchdir(directory_fd)
    return enter_directory


def _substitute(argv: list[str], replacements: dict[str, str]) -> list[str]:
    result: list[str] = []
    for token in argv:
        expanded = token
        for marker, replacement in replacements.items():
            expanded = expanded.replace(marker, replacement)
        if PLACEHOLDER.search(expanded):
            raise ContractError("unresolved command placeholder remains")
        result.append(expanded)
    return result


def _execution_environment(manifest: dict[str, Any], *, lease_fd: int, nonce: str) -> dict[str, str]:
    environment = dict(manifest["environment"])
    environment.update({
        HEAVY_LEASE_FD_ENV: str(lease_fd),
        "HAWKING_PHYSICAL_AB_ADMITTED": "1",
        "HAWKING_PHYSICAL_RUN_NONCE": nonce,
    })
    return environment


def _wait_bounded(
    process: subprocess.Popen[bytes], *, path_limits: Sequence[tuple[pathlib.Path, int]],
    timeout_seconds: int,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    while True:
        returncode = process.poll()
        if returncode is not None:
            return returncode
        if time.monotonic() >= deadline:
            process.kill()
            process.wait()
            raise AdmissionBlocked("child process exceeded its frozen timeout")
        for path, maximum_bytes in path_limits:
            try:
                if path.stat().st_size > maximum_bytes:
                    process.kill()
                    process.wait()
                    raise AdmissionBlocked("child output exceeded its frozen byte limit")
            except FileNotFoundError:
                pass
        time.sleep(0.05)


def _wait_collector_ready(
    process: subprocess.Popen[bytes], path: pathlib.Path, *, timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            return _load_stable_json(path, maximum_bytes=4 * 1024 * 1024)
        if process.poll() is not None:
            raise AdmissionBlocked("direct-counter collector exited before its ready barrier")
        time.sleep(0.05)
    process.kill()
    process.wait()
    raise AdmissionBlocked("direct-counter collector did not reach its ready barrier")


def _wait_bounded_at(
    process: subprocess.Popen[bytes], *, directory: AnchoredDirectory,
    names_and_limits: Sequence[tuple[str, int]], timeout_seconds: int,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    while True:
        returncode = process.poll()
        if returncode is not None:
            return returncode
        if time.monotonic() >= deadline:
            process.kill()
            process.wait()
            raise AdmissionBlocked("child process exceeded its frozen timeout")
        for name, maximum_bytes in names_and_limits:
            try:
                if directory.size(name) > maximum_bytes:
                    process.kill()
                    process.wait()
                    raise AdmissionBlocked("child output exceeded its frozen byte limit")
            except FileNotFoundError:
                pass
        time.sleep(0.05)


def _wait_collector_ready_at(
    process: subprocess.Popen[bytes], directory: AnchoredDirectory, name: str, *,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if directory.exists(name):
            return directory.load_json(name, maximum_bytes=4 * 1024 * 1024)
        if process.poll() is not None:
            raise AdmissionBlocked("direct-counter collector exited before its ready barrier")
        time.sleep(0.05)
    process.kill()
    process.wait()
    raise AdmissionBlocked("direct-counter collector did not reach its ready barrier")


def _validate_ready(
    value: Any, *, bundle: dict[str, Any], role: str, repeat: int,
    phase: str, nonce: str,
) -> list[str]:
    if not isinstance(value, dict):
        return ["collector ready receipt is absent"]
    expected = {
        "schema", "plan_sha256", "contract_sha256", "facet", "phase", "role",
        "repeat", "run_nonce", "collector_program_sha256", "benchmark_program_sha256",
        "directly_measured", "estimated", "ready_at_unix_ns", "ready_at_continuous_ns",
        "ready_sha256",
    }
    errors: list[str] = []
    program = bundle["contract"][f"{role}_program"]
    collector_program = bundle["collector_authority"]["collector_program"]
    if set(value) != expected or value.get("schema") != COLLECTOR_READY_SCHEMA:
        errors.append("collector ready fields/schema are incomplete or unexpected")
    comparisons = {
        "plan_sha256": bundle["plan"]["plan_sha256"],
        "contract_sha256": bundle["contract"]["contract_sha256"],
        "facet": bundle["contract"]["facet"], "phase": phase,
        "role": role, "repeat": repeat, "run_nonce": nonce,
        "collector_program_sha256": collector_program["sha256"],
        "benchmark_program_sha256": program["sha256"],
    }
    if any(value.get(field) != expected_value for field, expected_value in comparisons.items()):
        errors.append("collector ready receipt is not bound to the exact arm")
    if value.get("directly_measured") is not True or value.get("estimated") is not False:
        errors.append("collector ready receipt permits estimated counters")
    if not _integer(value.get("ready_at_unix_ns"), minimum=1) \
            or not _integer(value.get("ready_at_continuous_ns"), minimum=1):
        errors.append("collector ready time is invalid")
    errors.extend(_hash_errors(value, "ready_sha256", label="collector ready receipt"))
    return errors


def validate_counter_attestation(
    value: Any, *, bundle: dict[str, Any], role: str, repeat: int, phase: str,
    nonce: str, invocation_sha256: str, started_at_unix_ns: int,
    ended_at_unix_ns: int, started_at_continuous_ns: int,
    ended_at_continuous_ns: int, counter_payload: dict[str, Any],
    output: dict[str, Any], scientific: dict[str, Any], stdout: dict[str, Any],
    stderr: dict[str, Any], verify_files: bool,
    allowed_capture_root: pathlib.Path | None = None,
) -> list[str]:
    if not isinstance(value, dict):
        return ["counter attestation is absent"]
    expected = {
        "schema", "plan_sha256", "contract_sha256", "facet", "phase", "role",
        "repeat", "run_nonce", "collector_authority_sha256",
        "collector_program_sha256", "benchmark_program_sha256", "invocation_sha256",
        "execution_interval", "capture_interval", "directly_measured", "estimated",
        "domains", "counter_payload_sha256", "output_sha256", "scientific_sha256",
        "stdout_sha256", "stderr_sha256", "raw_captures", "attestation_sha256",
    }
    errors: list[str] = []
    if set(value) != expected or value.get("schema") != COUNTER_ATTESTATION_SCHEMA:
        errors.append("counter attestation fields/schema are incomplete or unexpected")
    comparisons = {
        "plan_sha256": bundle["plan"]["plan_sha256"],
        "contract_sha256": bundle["contract"]["contract_sha256"],
        "facet": bundle["contract"]["facet"], "phase": phase, "role": role,
        "repeat": repeat, "run_nonce": nonce,
        "collector_authority_sha256": bundle["collector_authority"]["authority_sha256"],
        "collector_program_sha256": bundle["collector_authority"]["collector_program"]["sha256"],
        "benchmark_program_sha256": bundle["contract"][f"{role}_program"]["sha256"],
        "invocation_sha256": invocation_sha256,
        "counter_payload_sha256": counter_payload.get("counter_payload_sha256"),
        "output_sha256": output.get("sha256"), "scientific_sha256": scientific.get("sha256"),
        "stdout_sha256": stdout.get("sha256"), "stderr_sha256": stderr.get("sha256"),
    }
    if any(value.get(field) != expected_value for field, expected_value in comparisons.items()):
        errors.append("counter attestation is not bound to exact arm artifacts")
    execution = value.get("execution_interval")
    if execution != {
        "started_at_unix_ns": started_at_unix_ns,
        "ended_at_unix_ns": ended_at_unix_ns,
        "started_at_continuous_ns": started_at_continuous_ns,
        "ended_at_continuous_ns": ended_at_continuous_ns,
    }:
        errors.append("counter attestation execution interval differs")
    capture = value.get("capture_interval")
    if not isinstance(capture, dict) or set(capture) != {
        "started_at_unix_ns", "ended_at_unix_ns", "started_at_continuous_ns",
        "ended_at_continuous_ns",
    }:
        errors.append("counter capture interval is incomplete")
    else:
        if not (
            capture.get("started_at_unix_ns", 1 << 80) <= started_at_unix_ns
            < ended_at_unix_ns <= capture.get("ended_at_unix_ns", -1)
            and capture.get("started_at_continuous_ns", 1 << 80) <= started_at_continuous_ns
            < ended_at_continuous_ns <= capture.get("ended_at_continuous_ns", -1)
        ):
            errors.append("direct-counter capture does not cover the exact arm interval")
    if value.get("directly_measured") is not True or value.get("estimated") is not False \
            or value.get("domains") != list(COUNTER_FIELDS):
        errors.append("counter attestation is estimated or lacks exact direct domains")
    raw = value.get("raw_captures")
    if not isinstance(raw, list) or not raw:
        errors.append("counter attestation lacks immutable raw captures")
        raw = []
    hashes = []
    for index, artifact in enumerate(raw):
        errors.extend(_artifact_errors(
            artifact, label=f"counter raw captures[{index}]", verify_files=verify_files,
            maximum_bytes=MAX_JSON_BYTES,
        ))
        hashes.append(artifact.get("sha256") if isinstance(artifact, dict) else None)
        if allowed_capture_root is not None and isinstance(artifact, dict) \
                and isinstance(artifact.get("path"), str):
            try:
                pathlib.Path(artifact["path"]).resolve(strict=True).relative_to(
                    allowed_capture_root.resolve(strict=True)
                )
            except (OSError, ValueError):
                errors.append(
                    f"counter raw captures[{index}] escapes the exclusive arm directory"
                )
    if len(set(hashes)) != len(hashes):
        errors.append("counter raw capture identity is reused")
    errors.extend(_hash_errors(value, "attestation_sha256", label="counter attestation"))
    return errors


def validate_scientific_receipt(
    value: Any, *, bundle: dict[str, Any], input_manifest: dict[str, Any],
    output: dict[str, Any], facet_payload: dict[str, Any],
) -> list[str]:
    if not isinstance(value, dict):
        return ["scientific receipt is absent or not JSON"]
    expected = {
        "schema", "plan_sha256", "facet", "adapter_id", "input_manifest_sha256",
        "output_sha256", "facet_payload_sha256", "exact_output", "skipped",
        "negative_evidence_preserved", "synthetic", "receipt_sha256",
    }
    errors: list[str] = []
    adapter = bundle.get("program_adapter", {})
    if set(value) != expected or value.get("schema") != SCIENTIFIC_RECEIPT_SCHEMA:
        errors.append("scientific receipt fields/schema are incomplete or unexpected")
    bindings = {
        "plan_sha256": bundle["plan"]["plan_sha256"],
        "facet": bundle["contract"]["facet"],
        "adapter_id": adapter.get("adapter_id"),
        "input_manifest_sha256": input_manifest["sha256"],
        "output_sha256": output["sha256"],
        "facet_payload_sha256": facet_payload["sha256"],
    }
    if any(value.get(field) != expected_value for field, expected_value in bindings.items()):
        errors.append("scientific receipt is not bound to exact input/output/payload/adapter")
    if value.get("exact_output") is not True or value.get("skipped") is not False \
            or value.get("negative_evidence_preserved") is not True \
            or value.get("synthetic") is not False:
        errors.append("scientific receipt is skipped, synthetic, inexact, or drops negative evidence")
    errors.extend(_hash_errors(value, "receipt_sha256", label="scientific receipt"))
    return errors


def _owner_snapshot_receipt(
    bundle: dict[str, Any], admission: dict[str, Any], *, phase: str,
    role: str, repeat: int, nonce: str, position: str,
) -> dict[str, Any]:
    lease_stat = os.fstat(admission["lease_fd"])
    owners = admission["owners"]
    return _stamp({
        "schema": "hawking.doctor_v5_physical_ab_owner_snapshot.v1",
        "plan_sha256": bundle["plan"]["plan_sha256"],
        "contract_sha256": bundle["contract"]["contract_sha256"],
        "facet": bundle["contract"]["facet"], "phase": phase,
        "role": role, "repeat": repeat, "run_nonce": nonce,
        "position": position, "observed_at_unix_ns": time.time_ns(),
        "ps_program": _artifact_identity(pathlib.Path(spec_reentry_scaffold.PS_PATH), executable=True),
        "shared_heavy_lease": {
            "path": str(HEAVY_LOCK.resolve(strict=True)),
            "st_dev": int(lease_stat.st_dev), "st_ino": int(lease_stat.st_ino),
            "inherited_descriptor": True, "held": True,
        },
        "owners": owners, "owner_count": len(owners),
        "probe_ok": not owners, "synthetic": False,
    }, "snapshot_sha256")


def _resource_guard_receipt(
    bundle: dict[str, Any], admission: dict[str, Any], *, phase: str,
    role: str, repeat: int, nonce: str, position: str,
) -> dict[str, Any]:
    limits = bundle["contract"]["run_limits"]
    snapshot = admission["resource"]
    errors = _resource_errors(snapshot, limits=limits)
    return _stamp({
        "schema": "hawking.doctor_v5_physical_ab_resource_guard.v1",
        "plan_sha256": bundle["plan"]["plan_sha256"],
        "contract_sha256": bundle["contract"]["contract_sha256"],
        "facet": bundle["contract"]["facet"], "phase": phase,
        "role": role, "repeat": repeat, "run_nonce": nonce,
        "position": position, "observed_at_unix_ns": time.time_ns(),
        "limits": copy.deepcopy(limits), "snapshot": copy.deepcopy(snapshot),
        "health_errors": errors, "healthy": not errors, "synthetic": False,
    }, "receipt_sha256")


def _run_arm_in_anchored_cwd(
    bundle: dict[str, Any], *, phase: str, role: str, repeat: int,
    arm_root: pathlib.Path,
) -> dict[str, Any]:
    """Execute one arm after the caller fchdir'd into its retained directory."""
    first = _arm_admission(bundle)
    lease_fd = first["lease_fd"]
    limits = bundle["contract"]["run_limits"]
    orchestration_group_sha256 = bundle["orchestration_group_sha256"]
    scope_sha256 = bundle["execution_scope"]["scope_sha256"]
    nonce = canonical_sha256({
        "orchestration_group_sha256": orchestration_group_sha256,
        "execution_scope_sha256": scope_sha256,
        "facet": bundle["contract"]["facet"], "phase": phase,
        "role": role, "repeat": repeat, "entropy": secrets.token_hex(32),
    })
    program = bundle["contract"][f"{role}_program"]
    output_path = arm_root / "output.json"
    scientific_path = arm_root / "scientific_receipt.json"
    payload_path = arm_root / "facet_payload.json"
    counter_path = arm_root / "counter_payload.json"
    attestation_path = arm_root / "counter_attestation.json"
    request_path = arm_root / "collector_request.json"
    ready_path = arm_root / "collector_ready.json"
    stop_path = arm_root / "collector_stop.json"
    arm_started_path = arm_root / "arm_started.json"
    stdout_path = arm_root / "stdout.bin"
    stderr_path = arm_root / "stderr.bin"
    collector_stdout_path = arm_root / "collector_stdout.bin"
    collector_stderr_path = arm_root / "collector_stderr.bin"

    argv_manifest = bundle[f"{role}_argv"]
    replacements = {
        "{INPUT_MANIFEST_PATH}": str(bundle["input_manifest_path"]),
        "{OUTPUT_PATH}": str(output_path),
        "{SCIENTIFIC_RECEIPT_PATH}": str(scientific_path),
        "{FACET_PAYLOAD_PATH}": str(payload_path),
        "{RUN_NONCE}": nonce,
    }
    argv = [str(bundle[f"{role}_program"]), *_substitute(argv_manifest["argv"], replacements)]
    environment = _execution_environment(argv_manifest, lease_fd=lease_fd, nonce=nonce)
    invocation = _stamp({
        "schema": "hawking.doctor_v5_physical_ab_invocation.v1",
        "plan_sha256": bundle["plan"]["plan_sha256"],
        "contract_sha256": bundle["contract"]["contract_sha256"],
        "execution_scope_sha256": scope_sha256,
        "orchestration_group_sha256": orchestration_group_sha256,
        "facet": bundle["contract"]["facet"], "phase": phase,
        "role": role, "repeat": repeat, "run_nonce": nonce,
        "program": bundle["contract"][f"{role}_program"],
        "argv": argv, "shell": False, "cwd": "retained-arm-directory-fd",
        "stdin": "devnull",
    }, "invocation_sha256")
    environment_manifest = _stamp({
        "schema": "hawking.doctor_v5_physical_ab_environment.v1",
        "run_nonce": nonce, "environment": environment,
        "ambient_environment_inherited": False,
    }, "environment_sha256")
    request = _stamp({
        "schema": "hawking.doctor_v5_physical_ab_collector_request.v1",
        "plan_sha256": bundle["plan"]["plan_sha256"],
        "contract_sha256": bundle["contract"]["contract_sha256"],
        "execution_scope_sha256": scope_sha256,
        "orchestration_group_sha256": orchestration_group_sha256,
        "facet": bundle["contract"]["facet"], "phase": phase,
        "role": role, "repeat": repeat, "run_nonce": nonce,
        "collector_authority_sha256": bundle["collector_authority"]["authority_sha256"],
        "benchmark_program": bundle["contract"][f"{role}_program"],
        "invocation_sha256": invocation["invocation_sha256"],
        "counter_output_path": str(counter_path),
        "counter_attestation_path": str(attestation_path),
        "ready_path": str(ready_path), "arm_started_path": str(arm_started_path),
        "stop_path": str(stop_path),
        "direct_counters_required": True, "estimated_counters_permitted": False,
    }, "request_sha256")
    _write_json_exclusive(request_path, request)
    collector_replacements = {
        "{COLLECTOR_REQUEST_PATH}": str(request_path),
        "{COLLECTOR_READY_PATH}": str(ready_path),
        "{COLLECTOR_STOP_PATH}": str(stop_path),
        "{ARM_STARTED_PATH}": str(arm_started_path),
        "{COUNTER_OUTPUT_PATH}": str(counter_path),
        "{COUNTER_ATTESTATION_PATH}": str(attestation_path),
    }
    collector_argv = [
        str(bundle["collector_program"]),
        *_substitute(bundle["collector_argv"]["argv"], collector_replacements),
    ]
    collector_env = _execution_environment(
        bundle["collector_argv"], lease_fd=lease_fd, nonce=nonce,
    )
    with collector_stdout_path.open("xb") as collector_stdout, \
            collector_stderr_path.open("xb") as collector_stderr:
        collector_process = subprocess.Popen(
            collector_argv, cwd=None, env=collector_env, stdin=subprocess.DEVNULL,
            stdout=collector_stdout, stderr=collector_stderr,
            pass_fds=(lease_fd,), shell=False,
        )
    try:
        ready = _wait_collector_ready(
            collector_process, ready_path,
            timeout_seconds=limits["collector_ready_timeout_seconds"],
        )
        ready_errors = _validate_ready(
            ready, bundle=bundle, role=role, repeat=repeat, phase=phase, nonce=nonce,
        )
        if ready_errors:
            raise AdmissionBlocked("; ".join(ready_errors))
        ready_identity = _artifact_identity(ready_path)
        os.chmod(ready_path, 0o400)
        # This is the immediate-before-arm recheck required by the contract.
        immediate = _arm_admission(bundle)
        if immediate["lease_fd"] != lease_fd:
            raise AdmissionBlocked("inherited lease descriptor changed at the arm barrier")
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            started_at_unix_ns = time.time_ns()
            started_at_continuous_ns = time.monotonic_ns()
            process = subprocess.Popen(
                argv, cwd=None, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, pass_fds=(lease_fd,), shell=False,
            )
        arm_started = _stamp({
            "schema": "hawking.doctor_v5_physical_ab_arm_started.v1",
            "run_nonce": nonce, "process_id": process.pid,
            "benchmark_program_sha256": program["sha256"],
            "invocation_sha256": invocation["invocation_sha256"],
            "started_at_unix_ns": started_at_unix_ns,
            "started_at_continuous_ns": started_at_continuous_ns,
        }, "arm_started_sha256")
        _write_json_exclusive(arm_started_path, arm_started)
        returncode = _wait_bounded(
            process, path_limits=(
                (stdout_path, limits["max_stdout_bytes"]),
                (stderr_path, limits["max_stderr_bytes"]),
            ),
            timeout_seconds=limits["timeout_seconds"],
        )
        ended_at_continuous_ns = time.monotonic_ns()
        ended_at_unix_ns = time.time_ns()
        stop = _stamp({
            "schema": "hawking.doctor_v5_physical_ab_collector_stop.v1",
            "run_nonce": nonce, "exit_code": returncode,
            "ended_at_unix_ns": ended_at_unix_ns,
            "ended_at_continuous_ns": ended_at_continuous_ns,
        }, "stop_sha256")
        _write_json_exclusive(stop_path, stop)
        collector_returncode = _wait_bounded(
            collector_process, path_limits=(
                (collector_stdout_path, limits["max_stdout_bytes"]),
                (collector_stderr_path, limits["max_stderr_bytes"]),
            ),
            timeout_seconds=limits["collector_ready_timeout_seconds"],
        )
    finally:
        if collector_process.poll() is None:
            collector_process.kill()
            collector_process.wait()
    if returncode != 0 or collector_returncode != 0:
        raise AdmissionBlocked(
            f"arm/collector failed with exit codes {returncode}/{collector_returncode}"
        )
    second = _arm_admission(bundle)
    swap_errors = _swap_growth_errors(
        first["resource"], second["resource"],
        maximum_growth_bytes=limits["maximum_swap_growth_bytes"],
    )
    if swap_errors:
        raise AdmissionBlocked("; ".join(swap_errors))

    invocation_path = arm_root / "invocation.json"
    environment_path = arm_root / "environment.json"
    owners_before_path = arm_root / "owners_before.json"
    owners_after_path = arm_root / "owners_after.json"
    resource_before_path = arm_root / "resource_before.json"
    resource_after_path = arm_root / "resource_after.json"
    owner_before = _owner_snapshot_receipt(
        bundle, first, phase=phase, role=role, repeat=repeat, nonce=nonce,
        position="before",
    )
    owner_after = _owner_snapshot_receipt(
        bundle, second, phase=phase, role=role, repeat=repeat, nonce=nonce,
        position="after",
    )
    resource_before = _resource_guard_receipt(
        bundle, first, phase=phase, role=role, repeat=repeat, nonce=nonce,
        position="before",
    )
    resource_after = _resource_guard_receipt(
        bundle, second, phase=phase, role=role, repeat=repeat, nonce=nonce,
        position="after",
    )
    for path, value in (
        (invocation_path, invocation), (environment_path, environment_manifest),
        (owners_before_path, owner_before), (owners_after_path, owner_after),
        (resource_before_path, resource_before), (resource_after_path, resource_after),
    ):
        _write_json_exclusive(path, value)
    for path in (
        stdout_path, stderr_path, collector_stdout_path, collector_stderr_path,
        output_path, scientific_path, payload_path, counter_path, attestation_path,
        ready_path,
    ):
        os.chmod(path, 0o400)

    output = _artifact_identity(output_path, maximum_bytes=limits["max_output_bytes"])
    scientific = _artifact_identity(
        scientific_path, maximum_bytes=limits["max_scientific_receipt_bytes"],
    )
    payload = _artifact_identity(payload_path, maximum_bytes=limits["max_facet_payload_bytes"])
    payload_value = _load_stable_json(
        payload_path, maximum_bytes=limits["max_facet_payload_bytes"],
    )
    scientific_value = _load_stable_json(
        scientific_path, maximum_bytes=limits["max_scientific_receipt_bytes"],
    )
    input_identity = _artifact_identity(bundle["input_manifest_path"])
    scientific_errors = validate_scientific_receipt(
        scientific_value, bundle=bundle, input_manifest=input_identity,
        output=output, facet_payload=payload,
    )
    if scientific_errors:
        raise AdmissionBlocked("; ".join(scientific_errors))
    stdout = _artifact_identity(stdout_path, allow_empty=True, maximum_bytes=limits["max_stdout_bytes"])
    stderr = _artifact_identity(stderr_path, allow_empty=True, maximum_bytes=limits["max_stderr_bytes"])
    counter_payload = _load_stable_json(counter_path, maximum_bytes=64 * 1024 * 1024)
    counter_errors = controller._counter_errors(
        counter_payload, facet=bundle["contract"]["facet"], nonce=nonce,
    )
    if counter_errors:
        raise AdmissionBlocked("; ".join(counter_errors))
    attestation = _load_stable_json(attestation_path, maximum_bytes=64 * 1024 * 1024)
    attestation_errors = validate_counter_attestation(
        attestation, bundle=bundle, role=role, repeat=repeat, phase=phase, nonce=nonce,
        invocation_sha256=invocation["invocation_sha256"],
        started_at_unix_ns=started_at_unix_ns, ended_at_unix_ns=ended_at_unix_ns,
        started_at_continuous_ns=started_at_continuous_ns,
        ended_at_continuous_ns=ended_at_continuous_ns,
        counter_payload=counter_payload, output=output, scientific=scientific,
        stdout=stdout, stderr=stderr, verify_files=True,
        allowed_capture_root=arm_root,
    )
    if attestation_errors:
        raise AdmissionBlocked("; ".join(attestation_errors))
    before_resource = first["resource"]
    after_resource = second["resource"]
    counter_attestation_ref = _artifact_identity(attestation_path)
    invocation_ref = _artifact_identity(invocation_path)
    environment_ref = _artifact_identity(environment_path)
    input_ref = input_identity
    owner_before_ref = _artifact_identity(owners_before_path)
    owner_after_ref = _artifact_identity(owners_after_path)
    resource_before_ref = _artifact_identity(resource_before_path)
    resource_after_ref = _artifact_identity(resource_after_path)
    sidecars = _stamp({
        "schema": ARM_SIDECAR_SCHEMA,
        "plan_sha256": bundle["plan"]["plan_sha256"],
        "contract_sha256": bundle["contract"]["contract_sha256"],
        "execution_scope_sha256": scope_sha256,
        "orchestration_group_sha256": orchestration_group_sha256,
        "facet": bundle["contract"]["facet"], "phase": phase,
        "role": role, "repeat": repeat, "run_nonce": nonce,
        "program_adapter": copy.deepcopy(bundle["program_adapter"]),
        "program": program, "benchmark_runner": _artifact_identity(pathlib.Path(__file__)),
        "invocation_manifest": invocation_ref, "environment_manifest": environment_ref,
        "input_manifest": input_ref, "output_manifest": output,
        "scientific_receipt": scientific,
        "owner_inventory_before": owner_before_ref,
        "owner_inventory_after": owner_after_ref,
        "resource_guard_before": resource_before_ref,
        "resource_guard_after": resource_after_ref,
        "stdout": stdout, "stderr": stderr,
        "collector_stdout": _artifact_identity(collector_stdout_path, allow_empty=True),
        "collector_stderr": _artifact_identity(collector_stderr_path, allow_empty=True),
        "collector_request": _artifact_identity(request_path),
        "collector_ready": ready_identity,
        "collector_stop": _artifact_identity(stop_path),
        "arm_started": _artifact_identity(arm_started_path),
        "counter_payload": _artifact_identity(counter_path),
        "counter_attestation": counter_attestation_ref,
        "facet_payload": payload,
        "resource_before": resource_before_ref, "resource_after": resource_after_ref,
        "direct_counter_validated": True, "shell_used": False,
        "ambient_environment_inherited": False,
        "live_doctor_mutated": False, "runtime_defaults_changed": False,
        "source_files_deleted": False, "synthetic": False,
    }, "sidecars_sha256")
    sidecars_path = arm_root / "arm_sidecars.json"
    _write_json_exclusive(sidecars_path, sidecars)
    sidecars_ref = _artifact_identity(sidecars_path)
    run = {
        "role": role, "repeat": repeat, "run_nonce": nonce,
        "boundary_attestation_sha256": bundle["release_authority"]["authority_sha256"],
        "program": program,
        "benchmark_runner": _artifact_identity(pathlib.Path(__file__)),
        "invocation_manifest": invocation_ref,
        "environment_manifest": environment_ref,
        "input_manifest": input_ref,
        "output_manifest": output, "scientific_receipt": scientific,
        "counter_attestation": counter_attestation_ref,
        "owner_inventory_before": owner_before_ref,
        "owner_inventory_after": owner_after_ref,
        "executor_arm_evidence": sidecars,
        "executor_arm_evidence_artifact": sidecars_ref,
        "invocation_sha256": invocation_ref["sha256"],
        "environment_sha256": environment_ref["sha256"],
        "started_at_unix_ns": started_at_unix_ns, "ended_at_unix_ns": ended_at_unix_ns,
        "exit_code": returncode, "skipped": False,
        "owner_count_before": len(first["owners"]), "owner_count_after": len(second["owners"]),
        "thermal_before": "nominal" if before_resource["thermal_state"] == 0 else "non-nominal",
        "thermal_after": "nominal" if after_resource["thermal_state"] == 0 else "non-nominal",
        "memory_pressure_before": "normal" if before_resource["pressure_level"] == 1 else "non-normal",
        "memory_pressure_after": "normal" if after_resource["pressure_level"] == 1 else "non-normal",
        "swap_before_mb": before_resource["swap_used_bytes"] / 1_000_000,
        "swap_after_mb": after_resource["swap_used_bytes"] / 1_000_000,
        "disk_free_before_bytes": before_resource["disk_free_bytes"],
        "disk_free_after_bytes": after_resource["disk_free_bytes"],
        "counter_payload": counter_payload,
        "counter_attestation_binding_sha256": canonical_sha256({
            "counter_payload_sha256": counter_payload["counter_payload_sha256"],
            "counter_attestation_sha256": counter_attestation_ref["sha256"],
            "run_nonce": nonce, "program_sha256": program["sha256"],
            "input_manifest_sha256": input_ref["sha256"],
            "output_manifest_sha256": output["sha256"],
        }),
        "exact_output_sha256": output["sha256"],
    }
    run = _stamp(run, "run_sha256")
    return {"run": run, "sidecars": sidecars, "payload_value": payload_value}


def _run_arm(
    bundle: dict[str, Any], *, phase: str, role: str, repeat: int,
    arm_parent: AnchoredDirectory, arm_name: str,
) -> dict[str, Any]:
    """Anchor all descendant pathname resolution to a retained arm dirfd."""
    with arm_parent.child(arm_name, create=True) as arm:
        previous = os.open(
            ".", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fchdir(arm.fd)
            result = _run_arm_in_anchored_cwd(
                bundle, phase=phase, role=role, repeat=repeat,
                arm_root=pathlib.Path("."),
            )
        finally:
            os.fchdir(previous)
            os.close(previous)
        arm.revalidate_visible_path()
        return result


def _final_payload(
    facet: str, measured: list[dict[str, Any]], *, plan: dict[str, Any],
) -> dict[str, Any]:
    payload_hashes = [row["sidecars"]["facet_payload"]["sha256"] for row in measured]
    if not payload_hashes or len(set(payload_hashes)) != 1:
        raise AdmissionBlocked("baseline/candidate facet payload bytes are not exact")
    payloads = [row["payload_value"] for row in measured]
    if not payloads or any(value != payloads[0] for value in payloads[1:]):
        raise AdmissionBlocked("baseline/candidate facet payloads are not semantically exact")
    payload = copy.deepcopy(payloads[0])
    if facet == "release_authority":
        payload = {
            "final_ready_rechecked_each_run": True,
            "zero_owners_rechecked_each_run": True,
            "shared_lease_continuous": True,
            "guard_sampled_each_run": True,
        }
    if facet == "full_stack_parity_ab":
        by_key = {(row["run"]["role"], row["run"]["repeat"]): row["run"] for row in measured}
        repeats = len(measured) // 2
        ratios = []
        for index in range(repeats):
            baseline = by_key[("baseline", index)]
            candidate = by_key[("candidate", index)]
            ratios.append(
                (baseline["ended_at_unix_ns"] - baseline["started_at_unix_ns"])
                / (candidate["ended_at_unix_ns"] - candidate["started_at_unix_ns"])
            )
        payload["paired_speedups"] = ratios
        payload["conservative_speedup"] = min(ratios)
        payload["component_speedups_multiplied"] = False
    errors = controller._domain_payload_errors(facet, payload, plan=plan)
    if errors:
        raise AdmissionBlocked("physical facet payload is not admissible: " + "; ".join(errors))
    return payload


def _build_release_boundary(
    bundle: dict[str, Any], *, execution_root: AnchoredDirectory,
    acquired_at_unix_ns: int,
) -> dict[str, Any]:
    final = _arm_admission(bundle)
    execution_root.write_json("boundary_owners.json", final["owners"])
    execution_root.write_json("boundary_ram_swap_guard.json", _stamp({
        "schema": "hawking.doctor_v5_physical_ab_ram_swap_guard.v1",
        "resource": final["resource"], "healthy": True,
    }, "receipt_sha256"))
    execution_root.write_json("boundary_disk_lifecycle.json", _stamp({
        "schema": "hawking.doctor_v5_physical_ab_disk_guard.v1",
        "disk_free_bytes": final["resource"]["disk_free_bytes"],
        "minimum_disk_reserve_bytes": controller.MIN_DISK_RESERVE_BYTES,
        "minimum_phase_aware_scratch_reserve_bytes": controller.MIN_SCRATCH_RESERVE_BYTES,
        "minimum_combined_admission_bytes": controller.MIN_TOTAL_DISK_ADMISSION_BYTES,
        "healthy": True,
    }, "receipt_sha256"))
    observed_at = time.time_ns()
    lease_authority = bundle["release_authority"]["shared_heavy_lease"]
    boundary = {
        "schema": controller.BOUNDARY_SCHEMA,
        "plan_sha256": bundle["plan"]["plan_sha256"],
        "observer_state_sha256": final["observer"]["state_sha256"],
        "final_interpretation_ready": True, "active_heavy_owner_count": 0,
        "owner_inventory": execution_root.artifact("boundary_owners.json"),
        "shared_heavy_lease": {
            "lock_file": _artifact_identity(HEAVY_LOCK, allow_empty=True),
            "held": True, "inherited_descriptor": True,
            "owners_rechecked_under_lock": True,
            "acquired_at_unix_ns": acquired_at_unix_ns,
            "released_at_unix_ns": observed_at,
        },
        "ram_swap_guard_receipt": execution_root.artifact("boundary_ram_swap_guard.json"),
        "disk_lifecycle_receipt": execution_root.artifact("boundary_disk_lifecycle.json"),
        "observed_at_unix_ns": observed_at,
    }
    if lease_authority["lock_file"] != boundary["shared_heavy_lease"]["lock_file"]:
        raise AdmissionBlocked("release-boundary lock identity changed during execution")
    return _stamp(boundary, "attestation_sha256")


def _execute_bundle_arms(
    bundle: dict[str, Any], *, work_root: AnchoredDirectory,
    orchestration_group_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime_bundle = dict(bundle)
    runtime_bundle["orchestration_group_sha256"] = orchestration_group_sha256
    warmups = [
        _run_arm(
            runtime_bundle, phase="warmup", role=role, repeat=0,
            arm_parent=work_root, arm_name=f"warmup-{index:02d}-{role}",
        )
        for index, role in enumerate(("baseline", "candidate"))
    ]
    measured: list[dict[str, Any]] = []
    for ordinal, label in enumerate(bundle["contract"]["pairing"]["order"]):
        role, raw_repeat = label.split(":", 1)
        measured.append(_run_arm(
            runtime_bundle, phase="measured", role=role, repeat=int(raw_repeat),
            arm_parent=work_root,
            arm_name=f"measured-{ordinal:03d}-{role}-{raw_repeat}",
        ))
    return warmups, measured


def _assemble_execution_receipt(
    bundle: dict[str, Any], *, work_root: AnchoredDirectory,
    warmups: list[dict[str, Any]], measured: list[dict[str, Any]],
    boundary: dict[str, Any], orchestration_group_sha256: str,
) -> dict[str, Any]:
    boundary_sha = boundary["attestation_sha256"]
    runs = []
    for row in measured:
        run = copy.deepcopy(row["run"])
        run["boundary_attestation_sha256"] = boundary_sha
        runs.append(_stamp(run, "run_sha256"))
    facet = bundle["contract"]["facet"]
    payload = _final_payload(facet, measured, plan=bundle["plan"])
    protocol = bundle["contract"]["pairing"]
    receipt = _stamp({
        "schema": controller.FACET_SCHEMA, "facet": facet, "status": "pass",
        "scope": "physical-owner-free-real-artifact", "structural_only": False,
        "plan_sha256": bundle["plan"]["plan_sha256"],
        "source_manifest_sha256": bundle["plan"]["source_manifest"]["manifest_sha256"],
        "boundary_attestation_sha256": boundary_sha,
        "paired_protocol": {
            "warmups_per_arm": protocol["warmups_per_arm"],
            "repeats_per_arm": protocol["repeats_per_arm"],
            "randomized_interleaved": True, "order": protocol["order"],
            "order_sha256": protocol["order_sha256"],
        },
        "runs": runs, "payload": payload, "runtime_defaults_changed": False,
        "source_files_deleted": False, "completed_evidence_mutated": False,
        "component_speedups_multiplied": False,
    }, "receipt_sha256")
    receipt_errors = controller._facet_errors(
        receipt, facet=facet, plan=bundle["plan"], boundary_sha=boundary_sha,
        verify_files=True,
        lease_interval=(
            boundary["shared_heavy_lease"]["acquired_at_unix_ns"],
            boundary["shared_heavy_lease"]["released_at_unix_ns"],
        ),
    )
    if receipt_errors:
        raise AdmissionBlocked(
            "executor-produced facet receipt failed: " + "; ".join(receipt_errors)
        )
    return _stamp({
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "plan_sha256": bundle["plan"]["plan_sha256"],
        "source_manifest_sha256": bundle["plan"]["source_manifest"]["manifest_sha256"],
        "executor_source_sha256": bundle["contract"]["executor_source_sha256"],
        "launch_contract_sha256": bundle["contract"]["contract_sha256"],
        "release_authority_sha256": bundle["release_authority"]["authority_sha256"],
        "collector_authority_sha256": bundle["collector_authority"]["authority_sha256"],
        "facet": facet, "release_boundary": boundary, "facet_receipt": receipt,
        "execution_scope": copy.deepcopy(bundle["execution_scope"]),
        "orchestration_group_sha256": orchestration_group_sha256,
        "warmup_run_sha256": [row["run"]["run_sha256"] for row in warmups],
        "measured_sidecars_sha256": [
            row["sidecars"]["sidecars_sha256"] for row in measured
        ],
        "sidecar_root": str(work_root.display_path), "shell_used": False,
        "shared_heavy_lease_acquired_by_executor": False,
        "live_doctor_mutated": False, "completed_evidence_mutated": False,
        "runtime_defaults_changed": False, "source_files_deleted": False,
    }, "execution_receipt_sha256")


def execute_bundle(bundle: dict[str, Any], *, output: pathlib.Path) -> dict[str, Any]:
    """Run one diagnostic facet; ten-facet evidence uses ``execute-all``."""
    fd, lease_errors = _validate_inherited_lease(bundle["release_authority"])
    if lease_errors or fd is None:
        raise AdmissionBlocked("; ".join(lease_errors))
    acquired_at = bundle["release_authority"]["shared_heavy_lease"]["acquired_at_unix_ns"]
    parent, output_name = _prepare_output_target(output)
    try:
        work_name = (
            f"{pathlib.Path(output_name).stem}."
            f"{bundle['contract']['contract_sha256'][:16]}.sidecars"
        )
        with parent.child(work_name, create=True) as work_root:
            group = canonical_sha256({
                "mode": "single-facet-diagnostic",
                "plan_sha256": bundle["plan"]["plan_sha256"],
                "release_authority_sha256": bundle["release_authority"]["authority_sha256"],
                "execution_scope_sha256": bundle["execution_scope"]["scope_sha256"],
                "facet": bundle["contract"]["facet"], "entropy": secrets.token_hex(32),
            })
            warmups, measured = _execute_bundle_arms(
                bundle, work_root=work_root, orchestration_group_sha256=group,
            )
            boundary = _build_release_boundary(
                bundle, execution_root=work_root, acquired_at_unix_ns=acquired_at,
            )
            result = _assemble_execution_receipt(
                bundle, work_root=work_root, warmups=warmups, measured=measured,
                boundary=boundary, orchestration_group_sha256=group,
            )
            work_root.revalidate_visible_path()
        parent.revalidate_visible_path()
        _atomic_final_json_at(parent, output_name, result)
        return result
    finally:
        parent.close()


def validate_matrix_manifest(
    value: Any, *, verify_files: bool = True,
) -> list[str]:
    expected = {
        "schema", "plan", "release_authority", "collector_authority", "entries",
        "exact_facet_order", "one_inherited_shared_heavy_lease", "one_common_boundary",
        "shell_permitted", "ambient_environment_inheritance_permitted",
        "runtime_defaults_changed", "activation_requested", "matrix_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return ["multi-launch matrix fields are incomplete or unexpected"]
    errors: list[str] = []
    if value.get("schema") != MATRIX_SCHEMA \
            or value.get("exact_facet_order") != list(controller.FACETS):
        errors.append("multi-launch matrix schema/facet order is invalid")
    for field in ("plan", "release_authority", "collector_authority"):
        errors.extend(_artifact_errors(
            value.get(field), label=f"matrix.{field}", verify_files=verify_files,
            maximum_bytes=MAX_JSON_BYTES,
        ))
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != len(controller.FACETS):
        errors.append("multi-launch matrix must contain exactly ten entries")
        entries = []
    entry_expected = {
        "facet", "launch_contract", "baseline_program", "baseline_argv_manifest",
        "candidate_program", "candidate_argv_manifest", "input_manifest",
        "execution_scope",
    }
    facets: list[Any] = []
    artifact_hashes: list[Any] = []
    for index, row in enumerate(entries):
        if not isinstance(row, dict) or set(row) != entry_expected:
            errors.append(f"multi-launch entry {index} fields are incomplete or unexpected")
            continue
        facets.append(row.get("facet"))
        for field, executable in (
            ("launch_contract", False), ("baseline_program", True),
            ("baseline_argv_manifest", False), ("candidate_program", True),
            ("candidate_argv_manifest", False), ("input_manifest", False),
            ("execution_scope", False),
        ):
            errors.extend(_artifact_errors(
                row.get(field), label=f"matrix.entries[{index}].{field}",
                verify_files=verify_files, executable=executable,
                maximum_bytes=None if executable else MAX_JSON_BYTES,
            ))
            if isinstance(row.get(field), dict):
                artifact_hashes.append((index, field, row[field].get("sha256")))
    if facets != list(controller.FACETS):
        errors.append("multi-launch entries are not the exact ten unique facets in plan order")
    # Programs may intentionally be shared, but contracts, argv manifests,
    # inputs, and scopes must be independently signed/hashed per facet.
    for field in (
        "launch_contract", "baseline_argv_manifest", "candidate_argv_manifest",
        "input_manifest", "execution_scope",
    ):
        hashes = [sha for _index, name, sha in artifact_hashes if name == field]
        if len(hashes) != len(set(hashes)):
            errors.append(f"multi-launch matrix reuses a {field} artifact across facets")
    if value.get("one_inherited_shared_heavy_lease") is not True \
            or value.get("one_common_boundary") is not True \
            or value.get("shell_permitted") is not False \
            or value.get("ambient_environment_inheritance_permitted") is not False \
            or value.get("runtime_defaults_changed") is not False \
            or value.get("activation_requested") is not False:
        errors.append("multi-launch matrix weakens common-lease/no-shell/default-off policy")
    errors.extend(_hash_errors(value, "matrix_sha256", label="multi-launch matrix"))
    return list(dict.fromkeys(errors))


def _artifact_path(value: Any, *, label: str) -> pathlib.Path:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ContractError(f"{label} artifact path is absent")
    return pathlib.Path(value["path"])


def _scope_population_identity(scope: dict[str, Any]) -> str:
    return canonical_sha256({
        key: copy.deepcopy(item) for key, item in scope.items()
        if key not in {"facet", "scope_sha256"}
    })


def load_matrix_bundles(
    matrix_path: pathlib.Path,
) -> tuple[list[str], dict[str, Any] | None, list[dict[str, Any]]]:
    try:
        matrix = _load_stable_json(matrix_path)
    except (OSError, ContractError) as exc:
        return [str(exc)], None, []
    errors = validate_matrix_manifest(matrix, verify_files=True)
    if errors:
        return errors, matrix, []
    bundles: list[dict[str, Any]] = []
    for row in matrix["entries"]:
        row_errors, bundle = validate_launch_bundle(
            plan_path=_artifact_path(matrix["plan"], label="matrix plan"),
            contract_path=_artifact_path(row["launch_contract"], label="launch contract"),
            release_authority_path=_artifact_path(
                matrix["release_authority"], label="release authority",
            ),
            baseline_program=_artifact_path(row["baseline_program"], label="baseline program"),
            baseline_argv_path=_artifact_path(
                row["baseline_argv_manifest"], label="baseline argv",
            ),
            candidate_program=_artifact_path(row["candidate_program"], label="candidate program"),
            candidate_argv_path=_artifact_path(
                row["candidate_argv_manifest"], label="candidate argv",
            ),
            input_manifest_path=_artifact_path(row["input_manifest"], label="input manifest"),
            execution_scope_path=_artifact_path(row["execution_scope"], label="execution scope"),
            collector_authority_path=_artifact_path(
                matrix["collector_authority"], label="collector authority",
            ),
            facet=row["facet"], verify_files=True,
        )
        errors.extend(f"{row['facet']}: {problem}" for problem in row_errors)
        if bundle is not None:
            bundles.append(bundle)
    if len(bundles) == len(controller.FACETS):
        plan_hashes = {row["plan"]["plan_sha256"] for row in bundles}
        release_hashes = {
            row["release_authority"]["authority_sha256"] for row in bundles
        }
        collector_hashes = {
            row["collector_authority"]["authority_sha256"] for row in bundles
        }
        scope_populations = {
            _scope_population_identity(row["execution_scope"]) for row in bundles
        }
        scope_facets = [row["execution_scope"].get("facet") for row in bundles]
        if len(plan_hashes) != 1 or len(release_hashes) != 1 \
                or len(collector_hashes) != 1 or len(scope_populations) != 1 \
                or scope_facets != list(controller.FACETS):
            errors.append(
                "all ten entries must share one plan/release/collector/scope population "
                "with exact per-facet scope bindings"
            )
    return list(dict.fromkeys(errors)), matrix, bundles


def validate_multi_execution_receipt(value: Any) -> list[str]:
    expected = {
        "schema", "matrix_sha256", "plan_sha256", "source_manifest_sha256",
        "release_authority_sha256", "collector_authority_sha256",
        "execution_scopes", "scope_population_sha256", "orchestration_group_sha256",
        "common_release_boundary",
        "facet_execution_receipts", "facet_execution_receipt_sha256",
        "run_nonces", "all_facets_completed", "one_inherited_shared_heavy_lease",
        "shell_used", "ambient_environment_inherited", "runtime_defaults_changed",
        "source_files_deleted", "multi_execution_receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return ["multi-execution receipt fields are incomplete or unexpected"]
    errors: list[str] = []
    if value.get("schema") != MULTI_RECEIPT_SCHEMA \
            or not _hex(value.get("matrix_sha256")) \
            or not _hex(value.get("plan_sha256")) \
            or not _hex(value.get("source_manifest_sha256")) \
            or not _hex(value.get("release_authority_sha256")) \
            or not _hex(value.get("collector_authority_sha256")) \
            or not _hex(value.get("orchestration_group_sha256")):
        errors.append("multi-execution receipt authority bindings are invalid")
    scopes = value.get("execution_scopes")
    if not isinstance(scopes, dict) or set(scopes) != set(controller.FACETS):
        errors.append("multi-execution receipt lacks exact per-facet execution scopes")
        scopes = scopes if isinstance(scopes, dict) else {}
    populations: list[str] = []
    for facet in controller.FACETS:
        scope = scopes.get(facet)
        errors.extend(controller._physical_execution_scope_errors(
            scope, facet=facet, verify_files=True,
        ))
        if isinstance(scope, dict):
            populations.append(_scope_population_identity(scope))
    if len(set(populations)) != 1 or not populations \
            or value.get("scope_population_sha256") != populations[0]:
        errors.append("multi-execution scope population differs across facets")
    boundary = value.get("common_release_boundary")
    boundary_sha = boundary.get("attestation_sha256") if isinstance(boundary, dict) else None
    rows = value.get("facet_execution_receipts")
    hashes = value.get("facet_execution_receipt_sha256")
    if not isinstance(rows, dict) or set(rows) != set(controller.FACETS) \
            or not isinstance(hashes, dict) or set(hashes) != set(controller.FACETS):
        errors.append("multi-execution receipt lacks the exact ten facet receipts")
        rows = rows if isinstance(rows, dict) else {}
        hashes = hashes if isinstance(hashes, dict) else {}
    seen_receipts: list[Any] = []
    seen_nonces: list[Any] = []
    for facet in controller.FACETS:
        row = rows.get(facet)
        if not isinstance(row, dict) or row.get("facet") != facet \
                or row.get("release_boundary") != boundary \
                or row.get("execution_scope") != scopes.get(facet) \
                or row.get("orchestration_group_sha256") \
                != value.get("orchestration_group_sha256"):
            errors.append(f"multi-execution {facet} receipt differs from common scope/boundary")
            continue
        if row.get("execution_receipt_sha256") != hashes.get(facet):
            errors.append(f"multi-execution {facet} hash map differs")
        seen_receipts.append(row.get("execution_receipt_sha256"))
        facet_receipt = row.get("facet_receipt", {})
        if isinstance(facet_receipt, dict):
            if facet_receipt.get("boundary_attestation_sha256") != boundary_sha:
                errors.append(f"multi-execution {facet} is not common-boundary bound")
            seen_nonces.extend(
                run.get("run_nonce") for run in facet_receipt.get("runs", [])
                if isinstance(run, dict)
            )
    if len(seen_receipts) != len(set(seen_receipts)):
        errors.append("multi-execution reuses a facet execution receipt")
    if value.get("run_nonces") != sorted(seen_nonces) \
            or len(seen_nonces) != len(set(seen_nonces)):
        errors.append("multi-execution run-nonce census is incomplete or reused")
    if value.get("all_facets_completed") is not True \
            or value.get("one_inherited_shared_heavy_lease") is not True \
            or value.get("shell_used") is not False \
            or value.get("ambient_environment_inherited") is not False \
            or value.get("runtime_defaults_changed") is not False \
            or value.get("source_files_deleted") is not False:
        errors.append("multi-execution receipt weakens completion/lease/lifecycle policy")
    errors.extend(_hash_errors(
        value, "multi_execution_receipt_sha256", label="multi-execution receipt",
    ))
    return list(dict.fromkeys(errors))


def execute_all_bundles(
    matrix: dict[str, Any], bundles: list[dict[str, Any]], *, output: pathlib.Path,
) -> dict[str, Any]:
    if len(bundles) != len(controller.FACETS):
        raise ContractError("execute-all requires ten validated bundles")
    first = bundles[0]
    fd, lease_errors = _validate_inherited_lease(first["release_authority"])
    if lease_errors or fd is None:
        raise AdmissionBlocked("; ".join(lease_errors))
    acquired_at = first["release_authority"]["shared_heavy_lease"]["acquired_at_unix_ns"]
    parent, output_name = _prepare_output_target(output)
    try:
        work_name = (
            f"{pathlib.Path(output_name).stem}.{matrix['matrix_sha256'][:16]}.sidecars"
        )
        with parent.child(work_name, create=True) as group_root:
            group = canonical_sha256({
                "matrix_sha256": matrix["matrix_sha256"],
                "plan_sha256": first["plan"]["plan_sha256"],
                "release_authority_sha256": first["release_authority"]["authority_sha256"],
                "scope_population_sha256": _scope_population_identity(
                    first["execution_scope"]
                ),
                "exact_facets": list(controller.FACETS), "entropy": secrets.token_hex(32),
            })
            executed: dict[str, tuple[dict[str, Any], AnchoredDirectory,
                                      list[dict[str, Any]], list[dict[str, Any]]]] = {}
            try:
                for bundle in bundles:
                    facet = bundle["contract"]["facet"]
                    facet_root = group_root.child(f"facet-{facet}", create=True)
                    warmups, measured = _execute_bundle_arms(
                        bundle, work_root=facet_root,
                        orchestration_group_sha256=group,
                    )
                    executed[facet] = (bundle, facet_root, warmups, measured)
                boundary = _build_release_boundary(
                    first, execution_root=group_root, acquired_at_unix_ns=acquired_at,
                )
                receipts: dict[str, dict[str, Any]] = {}
                receipt_refs: dict[str, dict[str, Any]] = {}
                for facet in controller.FACETS:
                    bundle, facet_root, warmups, measured = executed[facet]
                    result = _assemble_execution_receipt(
                        bundle, work_root=facet_root, warmups=warmups, measured=measured,
                        boundary=boundary, orchestration_group_sha256=group,
                    )
                    receipt_name = f"facet-{facet}.execution.json"
                    group_root.write_json(receipt_name, result)
                    receipts[facet] = result
                    receipt_refs[facet] = group_root.artifact(receipt_name)
                    facet_root.revalidate_visible_path()
                run_nonces = sorted(
                    run["run_nonce"]
                    for result in receipts.values()
                    for run in result["facet_receipt"]["runs"]
                )
                if len(run_nonces) != len(set(run_nonces)):
                    raise AdmissionBlocked("execute-all generated a duplicate run nonce")
                multi = _stamp({
                    "schema": MULTI_RECEIPT_SCHEMA,
                    "matrix_sha256": matrix["matrix_sha256"],
                    "plan_sha256": first["plan"]["plan_sha256"],
                    "source_manifest_sha256": first["plan"]["source_manifest"]["manifest_sha256"],
                    "release_authority_sha256": first["release_authority"]["authority_sha256"],
                    "collector_authority_sha256": first["collector_authority"]["authority_sha256"],
                    "execution_scopes": {
                        bundle["contract"]["facet"]: copy.deepcopy(bundle["execution_scope"])
                        for bundle in bundles
                    },
                    "scope_population_sha256": _scope_population_identity(
                        first["execution_scope"]
                    ),
                    "orchestration_group_sha256": group,
                    "common_release_boundary": boundary,
                    "facet_execution_receipts": receipts,
                    "facet_execution_receipt_sha256": {
                        facet: receipts[facet]["execution_receipt_sha256"]
                        for facet in controller.FACETS
                    },
                    "run_nonces": run_nonces, "all_facets_completed": True,
                    "one_inherited_shared_heavy_lease": True, "shell_used": False,
                    "ambient_environment_inherited": False,
                    "runtime_defaults_changed": False, "source_files_deleted": False,
                }, "multi_execution_receipt_sha256")
                problems = validate_multi_execution_receipt(multi)
                if problems:
                    raise AdmissionBlocked(
                        "executor-produced multi receipt failed: " + "; ".join(problems)
                    )
                # Receipt refs are retained as independently immutable files;
                # the inline map is the exact common-boundary transaction.
                group_root.write_json("facet_execution_receipt_artifacts.json", receipt_refs)
                group_root.revalidate_visible_path()
            finally:
                for _bundle, facet_root, _warmups, _measured in executed.values():
                    facet_root.close()
        parent.revalidate_visible_path()
        _atomic_final_json_at(parent, output_name, multi)
        return multi
    finally:
        parent.close()


def _bundle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--facet", required=True, choices=controller.FACETS)
    parser.add_argument("--plan", required=True, type=pathlib.Path)
    parser.add_argument("--launch-contract", required=True, type=pathlib.Path)
    parser.add_argument("--release-authority", required=True, type=pathlib.Path)
    parser.add_argument("--baseline-program", required=True, type=pathlib.Path)
    parser.add_argument("--baseline-argv-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-program", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-argv-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--input-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--execution-scope", required=True, type=pathlib.Path)
    parser.add_argument("--collector-authority", required=True, type=pathlib.Path)


def _hard_blockers_for_facet(status: dict[str, Any], facet: str) -> list[str]:
    """Keep global gates plus the selected facet's adapter gate only."""
    adapter_suffix = ": reviewed concrete baseline/candidate program adapter is absent"
    blockers = []
    for row in status.get("blockers", []):
        if row == "a concrete hash-bound launch bundle is required":
            continue
        if isinstance(row, str) and row.endswith(adapter_suffix) \
                and not row.startswith(f"{facet}:"):
            continue
        blockers.append(str(row))
    return blockers


def _load_bundle_from_args(args: argparse.Namespace) -> tuple[list[str], dict[str, Any] | None]:
    return validate_launch_bundle(
        plan_path=args.plan, contract_path=args.launch_contract,
        release_authority_path=args.release_authority,
        baseline_program=args.baseline_program,
        baseline_argv_path=args.baseline_argv_manifest,
        candidate_program=args.candidate_program,
        candidate_argv_path=args.candidate_argv_manifest,
        input_manifest_path=args.input_manifest,
        execution_scope_path=args.execution_scope,
        collector_authority_path=args.collector_authority,
        facet=args.facet, verify_files=True,
    )


def _selftest() -> int:
    config = build_config()
    assert config["default_off"] is True
    assert config["shell_permitted"] is False
    assert config["acquires_shared_heavy_lease"] is False
    seed = next(
        hashlib.sha256(f"selftest:{index}".encode()).hexdigest()
        for index in range(1000)
        if (lambda order: order != sorted(order))(
            build_randomized_order(hashlib.sha256(f"selftest:{index}".encode()).hexdigest(), 5)
        )
    )
    assert len(build_randomized_order(seed, 5)) == 10
    assert build_dry_run()["would_execute"] is False
    print("doctor_v5_physical_ab_executor.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    dry = subparsers.add_parser("dry-run")
    dry.add_argument("--facet", choices=controller.FACETS)
    validate = subparsers.add_parser("validate")
    _bundle_args(validate)
    execute = subparsers.add_parser("execute")
    _bundle_args(execute)
    execute.add_argument("--output", required=True, type=pathlib.Path)
    execute_all = subparsers.add_parser("execute-all")
    execute_all.add_argument("--matrix-manifest", required=True, type=pathlib.Path)
    execute_all.add_argument("--output", required=True, type=pathlib.Path)
    subparsers.add_parser("selftest")
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps(build_status(), indent=2, sort_keys=True))
        return 0
    if args.command == "dry-run":
        print(json.dumps(build_dry_run(facet=args.facet), indent=2, sort_keys=True))
        return 0
    if args.command == "selftest":
        return _selftest()
    if args.command in {"execute", "validate", "execute-all"}:
        # Do not inspect/hash caller artifacts (including potentially enormous
        # frozen inputs) or create output while the cheap release/collector
        # boundary is closed.
        status = build_status()
        hard_blockers = (
            _hard_blockers_for_facet(status, args.facet)
            if args.command != "execute-all"
            else [
                str(row) for row in status.get("blockers", [])
                if row != "a concrete hash-bound launch bundle is required"
            ]
        )
        if hard_blockers:
            print(json.dumps({
                "ok": False, "exit_code": EX_TEMPFAIL,
                "reason": "physical validation/execution admission is closed",
                "blockers": hard_blockers,
            }, indent=2, sort_keys=True), file=sys.stderr)
            return EX_TEMPFAIL
    if args.command == "execute-all":
        errors, matrix, bundles = load_matrix_bundles(args.matrix_manifest)
        if errors or matrix is None:
            print(json.dumps({"ok": False, "errors": errors}, indent=2, sort_keys=True))
            return 1
        try:
            result = execute_all_bundles(matrix, bundles, output=args.output)
        except (AdmissionBlocked, ContractError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return EX_TEMPFAIL if isinstance(exc, AdmissionBlocked) else 1
        print(json.dumps({
            "ok": True,
            "multi_execution_receipt_sha256": result["multi_execution_receipt_sha256"],
        }, indent=2, sort_keys=True))
        return 0
    errors, bundle = _load_bundle_from_args(args)
    if errors or bundle is None:
        print(json.dumps({"ok": False, "errors": errors}, indent=2, sort_keys=True))
        return 1
    if args.command == "validate":
        print(json.dumps({
            "ok": True, "plan_sha256": bundle["plan"]["plan_sha256"],
            "contract_sha256": bundle["contract"]["contract_sha256"],
            "facet": bundle["contract"]["facet"], "would_execute": False,
        }, indent=2, sort_keys=True))
        return 0
    try:
        result = execute_bundle(bundle, output=args.output)
    except (AdmissionBlocked, ContractError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return EX_TEMPFAIL if isinstance(exc, AdmissionBlocked) else 1
    print(json.dumps({
        "ok": True, "execution_receipt_sha256": result["execution_receipt_sha256"]
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

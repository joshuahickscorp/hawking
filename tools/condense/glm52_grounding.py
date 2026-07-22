#!/usr/bin/env python3.12
"""Fail-closed, read-only grounding observations for the GLM-5.2 campaign.

This module deliberately has no file-writing, deletion, network, or model-data
API.  Filesystem observations are rooted at an already-open directory and use
``*at`` operations with ``O_NOFOLLOW``.  A successful observation is stable
across pre/post descriptor checks and is producer-authenticated before it is
returned.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import stat
import subprocess
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.condense.glm52_common import (
    Glm52Error,
    canonical,
    seal,
    utc_now,
    verify_sealed,
)


FILE_OBSERVATION_SCHEMA = "hawking.glm52.grounded_file_observation.v1"
ABSENCE_OBSERVATION_SCHEMA = "hawking.glm52.grounded_absence_observation.v1"
RESOURCE_SAMPLE_SCHEMA = "hawking.glm52.grounded_resource_sample.v1"
AUTHENTICATION_DOMAIN = "hawking.glm52.grounding.producer-auth.v1"
_KEY_MATERIAL_IDENTITY_DOMAIN = b"hawking.glm52.auth-key-material-identity.v1\0"

_KNOWN_SCHEMAS = frozenset(
    {FILE_OBSERVATION_SCHEMA, ABSENCE_OBSERVATION_SCHEMA, RESOURCE_SAMPLE_SCHEMA}
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RFC3339_UTC_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z"
)
_VM_PAGE_SIZE_RE = re.compile(r"page size of\s+(\d+)\s+bytes", re.IGNORECASE)
_SWAP_VALUE_RE = re.compile(
    r"\b(total|used|free)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT])\b",
    re.IGNORECASE,
)
_MEMINFO_RE = re.compile(r"^([A-Za-z_()]+):\s+(\d+)\s+kB\s*$")

_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


class GroundingError(Glm52Error):
    """Raised when an observation cannot be grounded without ambiguity."""


class ResourceFloorError(GroundingError):
    """A fail-closed resource refusal with its authenticated sample attached."""

    def __init__(self, message: str, receipt: dict[str, Any]) -> None:
        super().__init__(message)
        self.receipt = receipt


class ProducerAuthenticator:
    """HMAC-SHA256 producer identity for deterministic observation receipts."""

    __slots__ = ("_key", "_key_identity_sha256")

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, bytes) or len(key) < 32:
            raise GroundingError("producer HMAC key must contain at least 32 bytes")
        self._key = key
        self._key_identity_sha256 = hashlib.sha256(key).hexdigest()

    def __repr__(self) -> str:
        return (
            "ProducerAuthenticator(key=<redacted>, "
            f"key_identity_sha256={self._key_identity_sha256!r})"
        )

    @property
    def key_identity_sha256(self) -> str:
        return self._key_identity_sha256

    def _key_material_identity(self) -> str:
        """Return an in-memory equality fingerprint for cross-role reuse checks.

        This deliberately uses the same private domain as the controller's other
        authenticators.  It is not a serialized producer identity and never exposes
        the key bytes.
        """
        return hashlib.sha256(_KEY_MATERIAL_IDENTITY_DOMAIN + self._key).hexdigest()

    def authenticate(self, body: Mapping[str, Any]) -> str:
        envelope = {"domain": AUTHENTICATION_DOMAIN, "observation": dict(body)}
        return hmac.new(self._key, canonical(envelope), hashlib.sha256).hexdigest()

    def verify(self, body: Mapping[str, Any], recorded: object) -> bool:
        return isinstance(recorded, str) and hmac.compare_digest(
            self.authenticate(body), recorded
        )


def _strict_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or _RFC3339_UTC_RE.fullmatch(value) is None:
        raise GroundingError(f"{label} must be a strict RFC3339 UTC-Z timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise GroundingError(f"{label} is not a valid UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise GroundingError(f"{label} must identify UTC")
    return parsed


def _authenticated_receipt(
    body: Mapping[str, Any], authenticator: ProducerAuthenticator
) -> dict[str, Any]:
    if "seal_sha256" in body or "producer_hmac_sha256" in body:
        raise GroundingError("receipt body contains a generated authentication field")
    unsigned = {
        **dict(body),
        "producer_key_identity_sha256": authenticator.key_identity_sha256,
    }
    return seal(
        {
            **unsigned,
            "producer_hmac_sha256": authenticator.authenticate(unsigned),
        }
    )


def verify_authenticated_observation(
    receipt: Mapping[str, Any],
    authenticator: ProducerAuthenticator,
    *,
    now: str | None = None,
    max_age_seconds: int | None = None,
) -> dict[str, Any]:
    """Verify the canonical seal, producer identity/HMAC, and optional freshness."""

    value = verify_sealed(dict(receipt), label="grounding observation")
    schema = value.get("schema")
    if schema not in _KNOWN_SCHEMAS:
        raise GroundingError(f"unknown grounding observation schema: {schema!r}")
    if value.get("producer_key_identity_sha256") != authenticator.key_identity_sha256:
        raise GroundingError("grounding observation producer identity mismatch")
    unsigned = {
        key: item
        for key, item in value.items()
        if key not in {"seal_sha256", "producer_hmac_sha256"}
    }
    if not authenticator.verify(unsigned, value.get("producer_hmac_sha256")):
        raise GroundingError("grounding observation producer HMAC mismatch")

    timestamp_field = (
        "sampled_at" if schema == RESOURCE_SAMPLE_SCHEMA else "observed_at"
    )
    observed = _strict_utc(value.get(timestamp_field), label=timestamp_field)
    if max_age_seconds is not None:
        if type(max_age_seconds) is not int or max_age_seconds < 0:
            raise GroundingError("max_age_seconds must be a non-negative integer")
        current = _strict_utc(now if now is not None else utc_now(), label="now")
        age = (current - observed).total_seconds()
        if age < 0:
            raise GroundingError("grounding observation timestamp is in the future")
        if age > max_age_seconds:
            raise GroundingError(
                f"grounding observation is stale: age={age:.6f}s "
                f"maximum={max_age_seconds}s"
            )
    return value


def _validate_expected(
    expected_size: object, expected_sha256: object
) -> tuple[int, str]:
    if type(expected_size) is not int or expected_size < 0:
        raise GroundingError("expected_size_bytes must be a non-negative integer")
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_RE.fullmatch(expected_sha256) is None
    ):
        raise GroundingError("expected_sha256 must be exactly 64 lowercase hex digits")
    return expected_size, expected_sha256


def _relative_parts(relative_path: object) -> tuple[str, tuple[str, ...]]:
    if not isinstance(relative_path, str) or not relative_path:
        raise GroundingError("relative path must be a non-empty string")
    if "\x00" in relative_path:
        raise GroundingError("relative path contains NUL")
    if relative_path.startswith("/"):
        raise GroundingError("absolute paths are forbidden")
    parts = tuple(relative_path.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise GroundingError("relative path must be normalized and may not traverse")
    normalized = "/".join(parts)
    if normalized != relative_path:
        raise GroundingError("relative path must use normalized POSIX separators")
    return normalized, parts


def _type_identity(st: os.stat_result) -> tuple[int, int, int]:
    return int(st.st_dev), int(st.st_ino), stat.S_IFMT(st.st_mode)


def _stable_file_fingerprint(st: os.stat_result) -> tuple[int, ...]:
    return (
        int(st.st_dev),
        int(st.st_ino),
        stat.S_IFMT(st.st_mode),
        int(st.st_size),
        int(st.st_nlink),
        int(st.st_mtime_ns),
        int(st.st_ctime_ns),
    )


def _stable_directory_fingerprint(st: os.stat_result) -> tuple[int, ...]:
    return (
        int(st.st_dev),
        int(st.st_ino),
        stat.S_IFMT(st.st_mode),
        int(st.st_size),
        int(st.st_nlink),
        int(st.st_mtime_ns),
        int(st.st_ctime_ns),
    )


def _safe_lstat(name: str, directory_fd: int) -> os.stat_result:
    return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)


def _normalized_absolute_root(root: str | os.PathLike[str]) -> str:
    root_path = os.fspath(root)
    if not isinstance(root_path, str):
        raise GroundingError("trusted filesystem root must be a text path")
    if not os.path.isabs(root_path):
        raise GroundingError("trusted filesystem root must be absolute")
    if root_path.startswith("//"):
        raise GroundingError("implementation-defined double-slash roots are forbidden")
    normalized = os.path.normpath(root_path)
    if normalized != root_path:
        raise GroundingError("trusted filesystem root must be normalized")
    return normalized


def _open_absolute_directory_chain(
    root_path: str,
) -> tuple[
    list[int],
    list[tuple[str, tuple[int, int, int]]],
    os.stat_result,
]:
    """Open every absolute-root component from ``/`` without following links."""

    if not _NOFOLLOW or not _DIRECTORY:
        raise GroundingError("O_NOFOLLOW and O_DIRECTORY are required")
    fds: list[int] = []
    links: list[tuple[str, tuple[int, int, int]]] = []
    try:
        filesystem_root_named = os.stat("/", follow_symlinks=False)
        filesystem_root_fd = os.open(
            "/", os.O_RDONLY | _CLOEXEC | _NOFOLLOW | _DIRECTORY
        )
        fds.append(filesystem_root_fd)
        filesystem_root_opened = os.fstat(filesystem_root_fd)
        if (
            not stat.S_ISDIR(filesystem_root_named.st_mode)
            or _type_identity(filesystem_root_named)
            != _type_identity(filesystem_root_opened)
        ):
            raise GroundingError("filesystem root changed while opening")

        components = tuple(component for component in root_path.split("/") if component)
        for component in components:
            try:
                named = _safe_lstat(component, fds[-1])
            except OSError as exc:
                raise GroundingError(
                    f"cannot inspect trusted root component {component!r}: {exc}"
                ) from exc
            if stat.S_ISLNK(named.st_mode):
                raise GroundingError(
                    f"symlink trusted root component is forbidden: {component!r}"
                )
            if not stat.S_ISDIR(named.st_mode):
                raise GroundingError(
                    f"trusted root component is not a directory: {component!r}"
                )
            try:
                child_fd = os.open(
                    component,
                    os.O_RDONLY | _CLOEXEC | _NOFOLLOW | _DIRECTORY,
                    dir_fd=fds[-1],
                )
            except OSError as exc:
                raise GroundingError(
                    f"cannot safely open trusted root component {component!r}: {exc}"
                ) from exc
            try:
                opened = os.fstat(child_fd)
            except OSError:
                os.close(child_fd)
                raise
            if _type_identity(named) != _type_identity(opened):
                os.close(child_fd)
                raise GroundingError(
                    f"trusted root component changed while opening: {component!r}"
                )
            links.append((component, _type_identity(opened)))
            fds.append(child_fd)
        root_stat = os.fstat(fds[-1])
        return fds, links, root_stat
    except BaseException:
        for fd in reversed(fds):
            os.close(fd)
        raise


def _verify_absolute_directory_chain(
    fds: Sequence[int],
    links: Sequence[tuple[str, tuple[int, int, int]]],
    root_stat: os.stat_result,
) -> None:
    if len(fds) != len(links) + 1:
        raise GroundingError("trusted root descriptor chain is malformed")
    for index, (component, expected_identity) in enumerate(links):
        try:
            named = _safe_lstat(component, fds[index])
            opened = os.fstat(fds[index + 1])
        except OSError as exc:
            raise GroundingError(
                f"trusted root component changed during observation: "
                f"{component!r}: {exc}"
            ) from exc
        if stat.S_ISLNK(named.st_mode):
            raise GroundingError(
                f"trusted root component became a symlink: {component!r}"
            )
        if (
            _type_identity(named) != expected_identity
            or _type_identity(opened) != expected_identity
        ):
            raise GroundingError(
                f"trusted root component identity changed: {component!r}"
            )
    if _type_identity(os.fstat(fds[-1])) != _type_identity(root_stat):
        raise GroundingError("trusted filesystem root descriptor identity changed")


class TrustedFilesystemObserver:
    """Descriptor-relative observer constrained to one stable directory root."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        root_id: str,
        authenticator: ProducerAuthenticator,
        clock: Callable[[], str] = utc_now,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if not _NOFOLLOW or not _DIRECTORY:
            raise GroundingError("O_NOFOLLOW and O_DIRECTORY are required")
        if not isinstance(root_id, str) or not root_id:
            raise GroundingError("root_id must be a non-empty string")
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise GroundingError("chunk_bytes must be a positive integer")
        self._root_path = _normalized_absolute_root(root)
        self._root_id = root_id
        self._authenticator = authenticator
        self._clock = clock
        self._chunk_bytes = chunk_bytes

    def _open_directory_chain(
        self, root_fd: int, components: Sequence[str]
    ) -> tuple[list[int], list[tuple[str, tuple[int, int, int]]]]:
        fds = [os.dup(root_fd)]
        links: list[tuple[str, tuple[int, int, int]]] = []
        try:
            for component in components:
                try:
                    named = _safe_lstat(component, fds[-1])
                except OSError as exc:
                    raise GroundingError(
                        f"cannot inspect directory component {component!r}: {exc}"
                    ) from exc
                if stat.S_ISLNK(named.st_mode):
                    raise GroundingError(
                        f"symlink directory component is forbidden: {component!r}"
                    )
                if not stat.S_ISDIR(named.st_mode):
                    raise GroundingError(
                        f"path component is not a directory: {component!r}"
                    )
                try:
                    child = os.open(
                        component,
                        os.O_RDONLY | _CLOEXEC | _NOFOLLOW | _DIRECTORY,
                        dir_fd=fds[-1],
                    )
                except OSError as exc:
                    raise GroundingError(
                        f"cannot safely open directory component {component!r}: {exc}"
                    ) from exc
                try:
                    opened = os.fstat(child)
                except OSError:
                    os.close(child)
                    raise
                if _type_identity(named) != _type_identity(opened):
                    os.close(child)
                    raise GroundingError(
                        f"directory component changed while opening: {component!r}"
                    )
                links.append((component, _type_identity(opened)))
                fds.append(child)
            return fds, links
        except BaseException:
            for fd in reversed(fds):
                os.close(fd)
            raise

    def _verify_directory_chain(
        self,
        fds: Sequence[int],
        links: Sequence[tuple[str, tuple[int, int, int]]],
    ) -> None:
        for index, (component, expected_identity) in enumerate(links):
            try:
                named = _safe_lstat(component, fds[index])
                opened = os.fstat(fds[index + 1])
            except OSError as exc:
                raise GroundingError(
                    "directory component changed during observation: "
                    f"{component!r}: {exc}"
                ) from exc
            if (
                _type_identity(named) != expected_identity
                or _type_identity(opened) != expected_identity
            ):
                raise GroundingError(
                    f"directory component identity changed: {component!r}"
                )

    def _timestamp(self, label: str) -> str:
        value = self._clock()
        _strict_utc(value, label=label)
        return value

    def _after_stream_before_post_fstat(self, relative_path: str, fd: int) -> None:
        """Test seam for proving that post-read mutation is rejected."""

    def _after_post_fstat_before_path_recheck(
        self, relative_path: str, fd: int
    ) -> None:
        """Test seam for proving that a name-to-inode swap is rejected."""

    def _after_absence_probe_before_recheck(
        self, relative_path: str, first_missing_component: str
    ) -> None:
        """Test seam for proving that absence races are rejected."""

    def observe_regular_file(
        self,
        relative_path: str,
        *,
        expected_size_bytes: int,
        expected_sha256: str,
    ) -> dict[str, Any]:
        """Hash and authenticate one contained, single-link regular file."""

        normalized, parts = _relative_parts(relative_path)
        expected_size, expected_hash = _validate_expected(
            expected_size_bytes, expected_sha256
        )
        root_fds, root_links, root_stat = _open_absolute_directory_chain(
            self._root_path
        )
        root_fd = root_fds[-1]
        fds: list[int] = []
        file_fd: int | None = None
        try:
            fds, links = self._open_directory_chain(root_fd, parts[:-1])
            parent_fd = fds[-1]
            leaf = parts[-1]
            try:
                named_pre = _safe_lstat(leaf, parent_fd)
            except OSError as exc:
                raise GroundingError(
                    f"cannot inspect grounded file {normalized!r}: {exc}"
                ) from exc
            if stat.S_ISLNK(named_pre.st_mode):
                raise GroundingError("grounded file may not be a symlink")
            if not stat.S_ISREG(named_pre.st_mode):
                raise GroundingError("grounded path is not a regular file")
            if int(named_pre.st_nlink) != 1:
                raise GroundingError("grounded file must have exactly one hard link")
            try:
                file_fd = os.open(
                    leaf,
                    os.O_RDONLY | _CLOEXEC | _NOFOLLOW | _NONBLOCK,
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                raise GroundingError(
                    f"cannot safely open grounded file: {exc}"
                ) from exc
            descriptor_pre = os.fstat(file_fd)
            if not stat.S_ISREG(descriptor_pre.st_mode):
                raise GroundingError("opened grounded object is not a regular file")
            if _stable_file_fingerprint(named_pre) != _stable_file_fingerprint(
                descriptor_pre
            ):
                raise GroundingError("grounded file changed while opening")
            if int(descriptor_pre.st_nlink) != 1:
                raise GroundingError("grounded file acquired an additional hard link")
            if int(descriptor_pre.st_size) != expected_size:
                raise GroundingError(
                    "grounded file size mismatch: "
                    f"observed={descriptor_pre.st_size} expected={expected_size}"
                )

            digest = hashlib.sha256()
            bytes_read = 0
            while True:
                chunk = os.read(file_fd, self._chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
                bytes_read += len(chunk)

            self._after_stream_before_post_fstat(normalized, file_fd)
            descriptor_post = os.fstat(file_fd)
            if _stable_file_fingerprint(descriptor_pre) != _stable_file_fingerprint(
                descriptor_post
            ):
                raise GroundingError("grounded file changed while it was being read")
            if bytes_read != expected_size:
                raise GroundingError(
                    "grounded read length mismatch: "
                    f"read={bytes_read} expected={expected_size}"
                )
            observed_hash = digest.hexdigest()
            if observed_hash != expected_hash:
                raise GroundingError(
                    "grounded file SHA-256 mismatch: "
                    f"observed={observed_hash} expected={expected_hash}"
                )

            self._after_post_fstat_before_path_recheck(normalized, file_fd)
            try:
                named_post = _safe_lstat(leaf, parent_fd)
            except OSError as exc:
                raise GroundingError(
                    f"grounded file name changed after reading: {exc}"
                ) from exc
            if _stable_file_fingerprint(named_post) != _stable_file_fingerprint(
                descriptor_post
            ):
                raise GroundingError(
                    "grounded file name now identifies different metadata"
                )
            self._verify_directory_chain(fds, links)
            _verify_absolute_directory_chain(root_fds, root_links, root_stat)

            body = {
                "schema": FILE_OBSERVATION_SCHEMA,
                "status": "PASS",
                "observation_kind": "contained_regular_file",
                "observed_at": self._timestamp("observed_at"),
                "root_id": self._root_id,
                "root_device": int(root_stat.st_dev),
                "root_inode": int(root_stat.st_ino),
                "relative_path": normalized,
                "expected_size_bytes": expected_size,
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "logical_bytes": int(descriptor_post.st_size),
                "allocated_bytes": int(descriptor_post.st_blocks) * 512,
                "device": int(descriptor_post.st_dev),
                "inode": int(descriptor_post.st_ino),
                "hard_link_count": int(descriptor_post.st_nlink),
            }
            return _authenticated_receipt(body, self._authenticator)
        finally:
            if file_fd is not None:
                os.close(file_fd)
            for fd in reversed(fds):
                os.close(fd)
            for fd in reversed(root_fds):
                os.close(fd)

    def observe_absence(self, relative_path: str) -> dict[str, Any]:
        """Authenticate stable absence without following any path component."""

        normalized, parts = _relative_parts(relative_path)
        root_fds, root_links, root_stat = _open_absolute_directory_chain(
            self._root_path
        )
        root_fd = root_fds[-1]
        fds: list[int] = []
        links: list[tuple[str, tuple[int, int, int]]] = []
        try:
            fds.append(os.dup(root_fd))
            for index, component in enumerate(parts):
                parent_fd = fds[-1]
                parent_pre = os.fstat(parent_fd)
                try:
                    named = _safe_lstat(component, parent_fd)
                except FileNotFoundError:
                    missing_prefix = "/".join(parts[: index + 1])
                    self._after_absence_probe_before_recheck(normalized, missing_prefix)
                    try:
                        _safe_lstat(component, parent_fd)
                    except FileNotFoundError:
                        pass
                    else:
                        raise GroundingError("absent path appeared during observation")
                    parent_post = os.fstat(parent_fd)
                    if _stable_directory_fingerprint(
                        parent_pre
                    ) != _stable_directory_fingerprint(parent_post):
                        raise GroundingError(
                            "parent directory changed during absence observation"
                        )
                    self._verify_directory_chain(fds, links)
                    _verify_absolute_directory_chain(
                        root_fds, root_links, root_stat
                    )
                    parent_path = "/".join(parts[:index]) or "."
                    body = {
                        "schema": ABSENCE_OBSERVATION_SCHEMA,
                        "status": "PASS",
                        "observation_kind": "contained_path_absence",
                        "observed_at": self._timestamp("observed_at"),
                        "root_id": self._root_id,
                        "root_device": int(root_stat.st_dev),
                        "root_inode": int(root_stat.st_ino),
                        "relative_path": normalized,
                        "absent": True,
                        "first_missing_component": missing_prefix,
                        "existing_parent": parent_path,
                        "parent_device": int(parent_post.st_dev),
                        "parent_inode": int(parent_post.st_ino),
                    }
                    return _authenticated_receipt(body, self._authenticator)
                except OSError as exc:
                    raise GroundingError(
                        f"cannot inspect path during absence observation: {exc}"
                    ) from exc

                if stat.S_ISLNK(named.st_mode):
                    raise GroundingError(
                        "symlink is forbidden in an absence observation"
                    )
                if index == len(parts) - 1:
                    raise GroundingError("cannot observe absence: path exists")
                if not stat.S_ISDIR(named.st_mode):
                    raise GroundingError(
                        "non-directory component blocks absence observation"
                    )
                try:
                    child = os.open(
                        component,
                        os.O_RDONLY | _CLOEXEC | _NOFOLLOW | _DIRECTORY,
                        dir_fd=parent_fd,
                    )
                except OSError as exc:
                    raise GroundingError(
                        f"cannot safely open absence path component: {exc}"
                    ) from exc
                try:
                    opened = os.fstat(child)
                except OSError:
                    os.close(child)
                    raise
                if _type_identity(named) != _type_identity(opened):
                    os.close(child)
                    raise GroundingError(
                        "absence path directory changed while opening"
                    )
                links.append((component, _type_identity(opened)))
                fds.append(child)
            raise AssertionError("validated path loop ended without a result")
        finally:
            for fd in reversed(fds):
                os.close(fd)
            for fd in reversed(root_fds):
                os.close(fd)


@dataclass(frozen=True, slots=True)
class ResourceReservePolicy:
    """Disk commitments and memory floors required before an operation starts."""

    emergency_floor_bytes: int = 5 * 1024**3
    largest_atomic_source_write_bytes: int = 0
    largest_compact_shard_write_bytes: int = 0
    next_checkpoint_write_bytes: int = 0
    xet_reconstruction_scratch_bytes: int = 0
    two_largest_official_source_shards_bytes: int = 0
    projected_remaining_compact_bytes: int = 0
    projected_teacher_evidence_bytes: int = 0
    active_scratch_bytes: int = 0
    current_best_artifact_bytes: int = 0
    rollback_capsule_bytes: int = 0
    minimum_available_ram_bytes: int = 0
    maximum_swap_used_bytes: int | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "maximum_swap_used_bytes" and value is None:
                continue
            if type(value) is not int or value < 0:
                raise GroundingError(
                    f"resource policy {field.name} must be a non-negative integer"
                )

    @property
    def operational_reserve_floor_bytes(self) -> int:
        return max(
            self.emergency_floor_bytes,
            self.largest_atomic_source_write_bytes,
            self.largest_compact_shard_write_bytes,
            self.next_checkpoint_write_bytes,
            self.xet_reconstruction_scratch_bytes,
            self.two_largest_official_source_shards_bytes,
        )

    @property
    def additional_reserved_bytes(self) -> int:
        return sum(
            (
                self.projected_remaining_compact_bytes,
                self.projected_teacher_evidence_bytes,
                self.active_scratch_bytes,
                self.current_best_artifact_bytes,
                self.rollback_capsule_bytes,
            )
        )

    @property
    def required_free_disk_bytes(self) -> int:
        return self.operational_reserve_floor_bytes + self.additional_reserved_bytes

    def as_dict(self) -> dict[str, int | None]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class DiskSample:
    total_bytes: int
    free_bytes: int
    used_bytes: int
    device: int

    def __post_init__(self) -> None:
        for name in ("total_bytes", "free_bytes", "used_bytes", "device"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise GroundingError(f"disk sample {name} must be non-negative integer")
        if self.free_bytes > self.total_bytes or self.used_bytes > self.total_bytes:
            raise GroundingError("disk sample values exceed total capacity")


@dataclass(frozen=True, slots=True)
class MemorySample:
    total_ram_bytes: int
    available_ram_bytes: int
    total_swap_bytes: int
    used_swap_bytes: int
    source: str

    def __post_init__(self) -> None:
        for name in (
            "total_ram_bytes",
            "available_ram_bytes",
            "total_swap_bytes",
            "used_swap_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise GroundingError(
                    f"memory sample {name} must be non-negative integer"
                )
        if self.available_ram_bytes > self.total_ram_bytes:
            raise GroundingError("available RAM exceeds total RAM")
        if self.used_swap_bytes > self.total_swap_bytes:
            raise GroundingError("used swap exceeds total swap")
        if not isinstance(self.source, str) or not self.source:
            raise GroundingError("memory sample source must be non-empty")


def _disk_sample_from_fd(root_fd: int) -> DiskSample:
    values = os.fstatvfs(root_fd)
    unit = int(values.f_frsize or values.f_bsize)
    total = int(values.f_blocks) * unit
    free_for_caller = int(values.f_bavail) * unit
    free_all = int(values.f_bfree) * unit
    used = total - free_all
    return DiskSample(
        total_bytes=total,
        free_bytes=free_for_caller,
        used_bytes=used,
        device=int(os.fstat(root_fd).st_dev),
    )


def _default_command_runner(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GroundingError(
            f"resource command failed {tuple(command)!r}: {exc}"
        ) from exc
    return completed.stdout


def parse_linux_meminfo(text: str) -> MemorySample:
    values: dict[str, int] = {}
    for line in text.splitlines():
        matched = _MEMINFO_RE.fullmatch(line.strip())
        if matched:
            values[matched.group(1)] = int(matched.group(2)) * 1024
    required = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    missing = sorted(required - values.keys())
    if missing:
        raise GroundingError(f"Linux meminfo is missing fields: {missing}")
    if values["SwapFree"] > values["SwapTotal"]:
        raise GroundingError("Linux SwapFree exceeds SwapTotal")
    return MemorySample(
        total_ram_bytes=values["MemTotal"],
        available_ram_bytes=values["MemAvailable"],
        total_swap_bytes=values["SwapTotal"],
        used_swap_bytes=values["SwapTotal"] - values["SwapFree"],
        source="linux:/proc/meminfo",
    )


def _unit_bytes(number: str, unit: str) -> int:
    powers = {"K": 1, "M": 2, "G": 3, "T": 4}
    try:
        result = Decimal(number) * (Decimal(1024) ** powers[unit.upper()])
    except (InvalidOperation, KeyError) as exc:
        raise GroundingError(f"invalid Darwin swap quantity: {number}{unit}") from exc
    if result != result.to_integral_value():
        raise GroundingError(
            "Darwin swap quantity is not an integral byte count: "
            f"{number}{unit}"
        )
    return int(result)


def parse_darwin_memory(
    *, hw_memsize: str, vm_stat: str, swapusage: str
) -> MemorySample:
    try:
        total_ram = int(hw_memsize.strip())
    except ValueError as exc:
        raise GroundingError("Darwin hw.memsize is not an integer") from exc
    page_match = _VM_PAGE_SIZE_RE.search(vm_stat)
    if page_match is None:
        raise GroundingError("Darwin vm_stat omitted its page size")
    page_size = int(page_match.group(1))
    page_values: dict[str, int] = {}
    for line in vm_stat.splitlines()[1:]:
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        digits = raw.strip().rstrip(".")
        if digits.isdigit():
            page_values[name.strip()] = int(digits)
    available_names = ("Pages free", "Pages inactive", "Pages speculative")
    missing_pages = [name for name in available_names if name not in page_values]
    if missing_pages:
        raise GroundingError(f"Darwin vm_stat is missing fields: {missing_pages}")
    available_ram = sum(page_values[name] for name in available_names) * page_size

    swap_values = {
        match.group(1).lower(): _unit_bytes(match.group(2), match.group(3))
        for match in _SWAP_VALUE_RE.finditer(swapusage)
    }
    if "total" not in swap_values or "used" not in swap_values:
        raise GroundingError("Darwin vm.swapusage omitted total or used")
    return MemorySample(
        total_ram_bytes=total_ram,
        available_ram_bytes=available_ram,
        total_swap_bytes=swap_values["total"],
        used_swap_bytes=swap_values["used"],
        source="darwin:sysctl+vm_stat",
    )


def _read_linux_meminfo() -> str:
    try:
        return Path("/proc/meminfo").read_text(encoding="ascii")
    except OSError as exc:
        raise GroundingError(f"cannot read live Linux memory counters: {exc}") from exc


def _live_memory_sample(platform_name: str) -> MemorySample:
    if platform_name == "linux":
        return parse_linux_meminfo(_read_linux_meminfo())
    if platform_name == "darwin":
        return parse_darwin_memory(
            hw_memsize=_default_command_runner(
                ("/usr/sbin/sysctl", "-n", "hw.memsize")
            ),
            vm_stat=_default_command_runner(("/usr/bin/vm_stat",)),
            swapusage=_default_command_runner(
                ("/usr/sbin/sysctl", "-n", "vm.swapusage")
            ),
        )
    raise GroundingError(
        "unsupported resource sampling platform "
        f"{platform_name!r}; Darwin/Linux required"
    )


def _sample_resources_with_providers(
    root: str | os.PathLike[str],
    *,
    root_id: str,
    policy: ResourceReservePolicy,
    authenticator: ProducerAuthenticator,
    clock: Callable[[], str],
    platform_name: str,
    disk_sampler: Callable[[int], DiskSample],
    memory_sampler: Callable[[], MemorySample],
    enforce: bool,
) -> dict[str, Any]:
    """Private deterministic harness; production must call ``sample_resources``."""

    if not isinstance(policy, ResourceReservePolicy):
        raise GroundingError("policy must be a ResourceReservePolicy")
    if not isinstance(root_id, str) or not root_id:
        raise GroundingError("root_id must be a non-empty string")
    if platform_name not in {"darwin", "linux"}:
        raise GroundingError("resource platform must be exactly 'darwin' or 'linux'")
    root_path = _normalized_absolute_root(root)
    root_fds, root_links, opened_root = _open_absolute_directory_chain(root_path)
    root_fd = root_fds[-1]
    try:
        memory = memory_sampler()
        # Disk is sampled last because it is the immediate launch/continuation gate.
        disk = disk_sampler(root_fd)
        _verify_absolute_directory_chain(
            root_fds, root_links, opened_root
        )
    finally:
        for fd in reversed(root_fds):
            os.close(fd)

    if not isinstance(disk, DiskSample):
        raise GroundingError("disk sampler must return DiskSample")
    if not isinstance(memory, MemorySample):
        raise GroundingError("memory sampler must return MemorySample")
    sampled_at = clock()
    _strict_utc(sampled_at, label="sampled_at")

    required_free = policy.required_free_disk_bytes
    usable_raw_window = disk.free_bytes - required_free
    disk_ok = disk.free_bytes >= required_free
    ram_ok = memory.available_ram_bytes >= policy.minimum_available_ram_bytes
    swap_ok = (
        policy.maximum_swap_used_bytes is None
        or memory.used_swap_bytes <= policy.maximum_swap_used_bytes
    )
    failures = []
    if not disk_ok:
        failures.append("disk_operational_reserve")
    if not ram_ok:
        failures.append("available_ram_floor")
    if not swap_ok:
        failures.append("swap_usage_ceiling")
    status = "PASS" if not failures else "REFUSED"

    body = {
        "schema": RESOURCE_SAMPLE_SCHEMA,
        "status": status,
        "observation_kind": "live_disk_ram_swap_resources",
        "sampled_at": sampled_at,
        "platform": platform_name,
        "root_id": root_id,
        "root_device": int(opened_root.st_dev),
        "root_inode": int(opened_root.st_ino),
        "disk_device": disk.device,
        "disk_total_bytes": disk.total_bytes,
        "disk_used_bytes": disk.used_bytes,
        "disk_free_bytes": disk.free_bytes,
        "total_ram_bytes": memory.total_ram_bytes,
        "available_ram_bytes": memory.available_ram_bytes,
        "total_swap_bytes": memory.total_swap_bytes,
        "used_swap_bytes": memory.used_swap_bytes,
        "memory_sample_source": memory.source,
        "resource_policy": policy.as_dict(),
        "operational_reserve_floor_bytes": policy.operational_reserve_floor_bytes,
        "additional_reserved_bytes": policy.additional_reserved_bytes,
        "required_free_disk_bytes": required_free,
        "usable_raw_window_bytes": usable_raw_window,
        "disk_operational_reserve_ok": disk_ok,
        "available_ram_floor_ok": ram_ok,
        "swap_usage_ceiling_ok": swap_ok,
        "refusal_reasons": failures,
    }
    receipt = _authenticated_receipt(body, authenticator)
    if failures and enforce:
        raise ResourceFloorError(
            "resource sample refused operation: " + ", ".join(failures), receipt
        )
    return receipt


def sample_resources(
    root: str | os.PathLike[str],
    *,
    root_id: str,
    policy: ResourceReservePolicy,
    authenticator: ProducerAuthenticator,
) -> dict[str, Any]:
    """Take live OS resource facts and fail closed below any declared floor.

    Unlike the private deterministic test harness, this production entry point
    exposes no provider, platform, command, timestamp, or enforcement override.
    """

    platform_name = os.uname().sysname.lower()
    return _sample_resources_with_providers(
        root,
        root_id=root_id,
        policy=policy,
        authenticator=authenticator,
        clock=utc_now,
        platform_name=platform_name,
        disk_sampler=_disk_sample_from_fd,
        memory_sampler=lambda: _live_memory_sample(platform_name),
        enforce=True,
    )

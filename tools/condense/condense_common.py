#!/usr/bin/env python3.12
"""Dependency-free primitives shared by condensation tooling."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import stat
import sys
import tempfile
from collections.abc import Mapping
from typing import Any


DEFAULT_MAX_JSON_BYTES = 16 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any, *, ensure_ascii: bool = False) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=ensure_ascii
    ).encode("utf-8")


def canonical_json(value: Any, *, ensure_ascii: bool = False) -> str:
    return canonical_json_bytes(value, ensure_ascii=ensure_ascii).decode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any, *, ensure_ascii: bool = False) -> str:
    return sha256_bytes(canonical_json_bytes(value, ensure_ascii=ensure_ascii))


def stamp_sha256(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    stamped = copy.deepcopy(value)
    stamped.pop(field, None)
    stamped[field] = canonical_sha256(stamped)
    return stamped


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def require_sha256(value: Any, *, label: str = "sha256") -> str:
    if not is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def read_bytes(
    path: str | os.PathLike[str],
    *,
    maximum: int = DEFAULT_MAX_JSON_BYTES,
    require_regular: bool = True,
) -> bytes:
    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    source = pathlib.Path(path)
    info = source.stat()
    if require_regular and not stat.S_ISREG(info.st_mode):
        raise ValueError(f"not a regular file: {source}")
    if info.st_size > maximum:
        raise ValueError(f"file exceeds {maximum} bytes: {source}")
    with source.open("rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError(f"file exceeds {maximum} bytes: {source}")
    return raw


def read_json(
    path: str | os.PathLike[str],
    *,
    maximum: int = DEFAULT_MAX_JSON_BYTES,
) -> Any:
    return json.loads(read_bytes(path, maximum=maximum).decode("utf-8"))


def read_json_optional(
    path: str | os.PathLike[str],
    *,
    maximum: int = DEFAULT_MAX_JSON_BYTES,
) -> Any | None:
    try:
        return read_json(path, maximum=maximum)
    except FileNotFoundError:
        return None


def fsync_directory(path: str | os.PathLike[str], *, strict: bool = False) -> None:
    directory = pathlib.Path(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        if strict:
            raise


def atomic_write_bytes(
    path: str | os.PathLike[str],
    value: bytes,
    *,
    mode: int | None = None,
) -> None:
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode is None:
        try:
            mode = stat.S_IMODE(destination.stat().st_mode)
        except FileNotFoundError:
            mode = 0o644
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    mode: int | None = None,
) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_bytes(path, raw, mode=mode)


def utc_now(*, timespec: str = "seconds") -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec=timespec)


def file_reference(
    path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    source = pathlib.Path(path)
    resolved = source.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"not a regular file: {source}")
    shown = resolved.relative_to(pathlib.Path(root).resolve()).as_posix() if root else str(resolved)
    return {
        "path": shown,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def file_reference_errors(
    value: Any,
    *,
    base: str | os.PathLike[str] | None = None,
    verify_file: bool = False,
) -> list[str]:
    if not isinstance(value, Mapping):
        return ["file reference must be an object"]
    errors: list[str] = []
    path_value = value.get("path")
    digest = value.get("sha256")
    size = value.get("size_bytes")
    if not isinstance(path_value, str) or not path_value:
        errors.append("file reference path must be a non-empty string")
    if not is_sha256(digest):
        errors.append("file reference sha256 must be a lowercase SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        errors.append("file reference size_bytes must be a non-negative integer")
    if verify_file and not errors:
        path = pathlib.Path(path_value)
        if not path.is_absolute():
            if base is None:
                errors.append("relative file reference requires a base")
                return errors
            path = pathlib.Path(base) / path
        try:
            current = file_reference(path)
        except (OSError, ValueError) as exc:
            errors.append(f"file reference cannot be read: {exc}")
        else:
            if current["sha256"] != digest:
                errors.append("file reference sha256 mismatch")
            if current["size_bytes"] != size:
                errors.append("file reference size_bytes mismatch")
    return errors


def process_alive(pid: int) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_identity(pid: int | None = None) -> dict[str, Any]:
    selected = os.getpid() if pid is None else pid
    identity: dict[str, Any] = {"pid": selected, "alive": process_alive(selected)}
    if selected == os.getpid():
        identity.update(
            {
                "ppid": os.getppid(),
                "executable": str(pathlib.Path(sys.executable).resolve()),
                "argv_sha256": canonical_sha256(sys.argv),
            }
        )
    return identity

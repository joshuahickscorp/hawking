#!/usr/bin/env python3.12
"""Immutable, bounded-memory source reader for unbound Doctor work graphs.

The production Qwen worker and the live Doctor queue do not import this module.
It provides the source primitive needed by future GPT-OSS and larger-model
workers: a file is opened read-only without following symlinks, every read is a
bounded ``pread``, and identity is checked before and after access.  Full-file
hashing streams fixed-size chunks and never returns the file as one byte string.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Iterator


SHA_RE = re.compile(r"[0-9a-f]{64}")
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
MAX_CHUNK_BYTES = 64 * 1024 * 1024


class StreamingSourceError(RuntimeError):
    """A source identity or bounded-read contract was violated."""


@dataclass(frozen=True)
class SourceIdentity:
    path: str
    bytes: int
    device: int
    inode: int
    mtime_ns: int

    @classmethod
    def from_stat(cls, path: Path, row: os.stat_result) -> "SourceIdentity":
        return cls(
            path=str(path), bytes=row.st_size, device=row.st_dev,
            inode=row.st_ino, mtime_ns=row.st_mtime_ns,
        )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev, left.st_ino, left.st_size, left.st_mtime_ns
    ) == (
        right.st_dev, right.st_ino, right.st_size, right.st_mtime_ns
    )


class ImmutableSourceReader:
    """Read one immutable regular file by bounded byte ranges.

    ``expected_sha256`` is an inherited content authority.  It is checked only
    by :meth:`hash_all`; range reads instead bind their own digest plus the
    inherited source digest.  This avoids repeatedly hashing multi-gigabyte
    shards while still making every downstream receipt source-addressable.
    """

    def __init__(
        self, path: Path, *, expected_bytes: int,
        expected_sha256: str, expected_device: int | None = None,
        expected_inode: int | None = None, expected_mtime_ns: int | None = None,
    ) -> None:
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) \
                or expected_bytes < 0:
            raise StreamingSourceError("expected source byte count is invalid")
        if not isinstance(expected_sha256, str) \
                or SHA_RE.fullmatch(expected_sha256) is None:
            raise StreamingSourceError("expected source SHA-256 is invalid")
        raw = Path(path)
        if raw.is_symlink():
            raise StreamingSourceError(f"symlinked source is forbidden: {raw}")
        self.path = raw.resolve(strict=True)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) \
            | getattr(os, "O_NOFOLLOW", 0)
        try:
            self._fd = os.open(self.path, flags)
        except OSError as exc:
            raise StreamingSourceError(f"cannot open immutable source {raw}: {exc}") from exc
        self.expected_sha256 = expected_sha256
        self._closed = False
        try:
            row = os.fstat(self._fd)
            if not stat.S_ISREG(row.st_mode) or row.st_size != expected_bytes:
                raise StreamingSourceError("source type/size differs from authority")
            for name, expected, observed in (
                ("device", expected_device, row.st_dev),
                ("inode", expected_inode, row.st_ino),
                ("mtime_ns", expected_mtime_ns, row.st_mtime_ns),
            ):
                if expected is not None and expected != observed:
                    raise StreamingSourceError(f"source {name} differs from authority")
            self._opened = row
            self.identity = SourceIdentity.from_stat(self.path, row)
        except BaseException:
            os.close(self._fd)
            self._closed = True
            raise

    def __enter__(self) -> "ImmutableSourceReader":
        return self

    def __exit__(self, _kind: object, _value: object, _trace: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            os.close(self._fd)
            self._closed = True

    def _check_open(self) -> os.stat_result:
        if self._closed:
            raise StreamingSourceError("source reader is closed")
        row = os.fstat(self._fd)
        if not _same_identity(self._opened, row):
            raise StreamingSourceError("source changed after it was opened")
        return row

    def read_exact(self, offset: int, size: int) -> bytes:
        row = self._check_open()
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (offset, size)) or offset < 0 or size < 0 \
                or size > MAX_CHUNK_BYTES or offset + size > row.st_size:
            raise StreamingSourceError(
                f"unsafe source byte range: {self.path}@{offset}+{size}"
            )
        value = os.pread(self._fd, size, offset)
        if len(value) != size:
            raise StreamingSourceError("source was truncated during bounded pread")
        self._check_open()
        return value

    def iter_range(
        self, offset: int, size: int, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> Iterator[bytes]:
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) \
                or not 1 <= chunk_bytes <= MAX_CHUNK_BYTES:
            raise StreamingSourceError("stream chunk size is outside the safety envelope")
        row = self._check_open()
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (offset, size)) or offset < 0 or size < 0 \
                or offset + size > row.st_size:
            raise StreamingSourceError("stream range is outside the source")
        cursor, remaining = offset, size
        while remaining:
            amount = min(chunk_bytes, remaining)
            yield self.read_exact(cursor, amount)
            cursor += amount
            remaining -= amount

    def hash_range(
        self, offset: int, size: int, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> dict[str, object]:
        digest = hashlib.sha256()
        total = 0
        for block in self.iter_range(offset, size, chunk_bytes=chunk_bytes):
            digest.update(block)
            total += len(block)
        return {
            "source": {
                "path": self.identity.path, "sha256": self.expected_sha256,
                "bytes": self.identity.bytes,
            },
            "absolute_byte_range": [offset, offset + size],
            "bytes": total, "range_sha256": digest.hexdigest(),
            "whole_file_materialized": False,
            "maximum_buffer_bytes": min(chunk_bytes, size) if size else 0,
        }

    def hash_all(self, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> dict[str, object]:
        receipt = self.hash_range(0, self.identity.bytes, chunk_bytes=chunk_bytes)
        if receipt["range_sha256"] != self.expected_sha256:
            raise StreamingSourceError("streamed full-file SHA-256 differs from authority")
        receipt["content_authority_verified"] = True
        return receipt


def open_manifest_source(row: dict[str, object]) -> ImmutableSourceReader:
    """Open a canonical local-source manifest row without relaxing identity."""
    if not isinstance(row, dict):
        raise StreamingSourceError("source manifest row is not an object")
    try:
        return ImmutableSourceReader(
            Path(str(row["path"])), expected_bytes=row["bytes"],
            expected_sha256=row["sha256"],
            expected_device=row.get("device"), expected_inode=row.get("inode"),
            expected_mtime_ns=row.get("mtime_ns"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StreamingSourceError(f"source manifest row is invalid: {exc}") from exc

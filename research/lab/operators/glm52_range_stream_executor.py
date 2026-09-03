"""Resumable direct-Xet executor for the sealed GLM-5.2 organ schedule.

Tensor bodies are framed directly from ordered hf_xet range streams into one
external window operator.  This authority never materializes a full source
shard.  The operator must return a sealed cold-handoff and exact-eviction
receipt before the next window can start.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import select
import signal
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from lab.layout import resolve_workspace_path
from lab.operators.glm52_common import canonical, read_sealed_json, seal, verify_sealed
from lab.operators.glm52_restream_contract import REPO_ID, REVISION, live_window_admission
from lab.operators.gravity_range_scheduler import plan_glm52_organ_windows
from ramanujan.restream_guard import (
    TRANSITION_SCHEMA,
    load_pinned_owner_public_key,
    start_claimed_owner_authorization,
    validate_bounded_restream,
)


RECEIPT_SCHEMA = "hawking.glm52.range_window_terminal_receipt.v1"
TERMINAL_SCHEMA = "hawking.glm52.range_restream_terminal_receipt.v1"
PROTOCOL = "hawking.glm52.window_stream.framed.v1"
MAX_OPERATOR_RECEIPT_BYTES = 1024 * 1024
MAX_FRAME_HEADER_BYTES = 1024 * 1024
DEFAULT_OPERATOR_TIMEOUT_SECONDS = 2 * 60 * 60
MIN_OPERATOR_TIMEOUT_SECONDS = 60
MAX_OPERATOR_TIMEOUT_SECONDS = 6 * 60 * 60
DEFAULT_RANGE_TIMEOUT_SECONDS = 15 * 60
MIN_RANGE_TIMEOUT_SECONDS = 30
MAX_RANGE_TIMEOUT_SECONDS = 60 * 60


@dataclass(frozen=True)
class FramedMessage:
    """One exact length-delimited operator message.

    The production executor writes a JSON header preceded by an eight-byte
    big-endian header length.  Range messages carry their raw body immediately
    after the header and *must* declare ``payload_bytes``.  Keeping this parser
    next to the writer gives a fixture operator the exact same wire contract
    without making the parent-restream executor itself an authority to process
    model data.
    """

    header: dict[str, Any]
    payload: bytes


class RangeExecutorError(RuntimeError):
    """A range identity, lifecycle, resource, or operator gate failed closed."""


def _sha256_list(values: list[str]) -> str:
    return hashlib.sha256(canonical(values)).hexdigest()


def range_source_identity_sha256(window: Mapping[str, Any]) -> str:
    """Bind exact official object identities and byte intervals for a window."""
    rows = window.get("ranges")
    if not isinstance(rows, list) or not rows:
        raise RangeExecutorError("window lacks rebuilt source ranges")
    identity = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise RangeExecutorError("window source range is not a mapping")
        identity.append({
            "range_id": row.get("range_id"),
            "shard": row.get("shard"),
            "xet_hash": row.get("xet_hash"),
            "file_bytes": row.get("file_bytes"),
            "start": row.get("start"),
            "end": row.get("end"),
            "payload_bytes": row.get("payload_bytes"),
        })
    return hashlib.sha256(canonical(identity)).hexdigest()


class _HashingChunks:
    """One-pass body hasher that preserves the official stream iterator."""

    def __init__(self, chunks: Any, *, before_chunk: Any = None) -> None:
        self._chunks = chunks
        self._before_chunk = before_chunk
        self._hash = hashlib.sha256()
        self.count = 0

    def __iter__(self):
        for chunk in self._chunks:
            if not isinstance(chunk, bytes) or not chunk:
                raise RangeExecutorError("Xet stream yielded a nonempty-bytes violation")
            if self._before_chunk is not None:
                self._before_chunk(len(chunk))
            self._hash.update(chunk)
            self.count += len(chunk)
            yield chunk

    @property
    def sha256(self) -> str:
        return self._hash.hexdigest()


class _RangeDeadlineChunks:
    """Bound a possibly stuck Xet iterator without buffering a whole range."""

    def __init__(self, chunks: Any, *, range_id: str, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise RangeExecutorError("range deadline must be positive")
        self._chunks = chunks
        self._range_id = range_id
        self._deadline = time.monotonic() + timeout_seconds
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=2)
        self._cancelled = threading.Event()
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _put(self, kind: str, value: Any) -> None:
        while not self._cancelled.is_set():
            try:
                self._queue.put((kind, value), timeout=0.05)
                return
            except queue.Full:
                continue

    def _produce(self) -> None:
        try:
            for chunk in self._chunks:
                if self._cancelled.is_set():
                    return
                self._put("chunk", chunk)
            self._put("done", None)
        except BaseException as exc:  # noqa: BLE001 - carry Xet producer failure to the bounded consumer.
            self._put("error", exc)

    def cancel(self) -> None:
        self._cancelled.set()
        try:
            self._chunks.cancel()
        except Exception:
            pass

    def __iter__(self):
        try:
            while True:
                remaining = self._deadline - time.monotonic()
                if remaining <= 0:
                    self.cancel()
                    raise RangeExecutorError(f"Xet range deadline expired: {self._range_id}")
                try:
                    kind, value = self._queue.get(timeout=remaining)
                except queue.Empty as exc:
                    self.cancel()
                    raise RangeExecutorError(f"Xet range deadline expired: {self._range_id}") from exc
                if kind == "done":
                    return
                if kind == "error":
                    raise value
                yield value
        finally:
            self.cancel()


def _call_with_deadline(
    call: Any, *, range_id: str, timeout_seconds: float, on_timeout: Any = None,
) -> Any:
    """Run range setup under the same deadline as body consumption.

    ``hf_xet`` creates the stream synchronously.  A network stall there used
    to occur before ``_RangeDeadlineChunks`` existed, leaving the executor
    subject only to the much broader window watchdog.  The worker is daemon
    only because a third-party native call cannot safely be force-killed from
    Python; the campaign itself fails closed and invokes the supplied session
    abort hook before returning control to its caller.
    """
    if timeout_seconds <= 0:
        raise RangeExecutorError("range deadline must be positive")
    result: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            result.put(("result", call()))
        except BaseException as exc:  # noqa: BLE001 - preserve Xet setup errors.
            result.put(("error", exc))

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    try:
        kind, value = result.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        if on_timeout is not None:
            try:
                on_timeout()
            except Exception:
                pass
        raise RangeExecutorError(f"Xet range setup deadline expired: {range_id}") from exc
    if kind == "error":
        raise value
    return value


@contextmanager
def _exclusive_claim_execution(claim_path: Path):
    """Hold one OS lock for the entire launch/resume campaign process."""
    import fcntl
    import stat

    lock_path = claim_path.with_suffix(".executor.lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        named = os.lstat(lock_path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) & 0o077
        ):
            raise RangeExecutorError("owner launch execution lock is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RangeExecutorError("owner launch campaign is already executing") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _operator_timeout_seconds() -> int:
    raw = os.environ.get("HAWKING_GLM52_OPERATOR_TIMEOUT_SECONDS", str(DEFAULT_OPERATOR_TIMEOUT_SECONDS))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RangeExecutorError("operator timeout must be an integer number of seconds") from exc
    if not MIN_OPERATOR_TIMEOUT_SECONDS <= value <= MAX_OPERATOR_TIMEOUT_SECONDS:
        raise RangeExecutorError(
            f"operator timeout must be in [{MIN_OPERATOR_TIMEOUT_SECONDS}, {MAX_OPERATOR_TIMEOUT_SECONDS}] seconds"
        )
    return value


def _range_timeout_seconds() -> int:
    raw = os.environ.get("HAWKING_GLM52_RANGE_TIMEOUT_SECONDS", str(DEFAULT_RANGE_TIMEOUT_SECONDS))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RangeExecutorError("range timeout must be an integer number of seconds") from exc
    if not MIN_RANGE_TIMEOUT_SECONDS <= value <= MAX_RANGE_TIMEOUT_SECONDS:
        raise RangeExecutorError(
            f"range timeout must be in [{MIN_RANGE_TIMEOUT_SECONDS}, {MAX_RANGE_TIMEOUT_SECONDS}] seconds"
        )
    return value


def assert_unified_floor(
    workspace_root: Path, *, protected_floor_bytes: int, additional_bytes: int = 0,
    observed_free_bytes: int | None = None, stage: str,
) -> dict[str, int]:
    """Fail before a bounded write when current free bytes would cross the floor."""
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (protected_floor_bytes, additional_bytes)):
        raise RangeExecutorError("unified-floor byte inputs must be non-negative integers")
    free = shutil.disk_usage(workspace_root).free if observed_free_bytes is None else observed_free_bytes
    if isinstance(free, bool) or not isinstance(free, int) or free < 0:
        raise RangeExecutorError("unified-floor free-byte sample is invalid")
    if free - additional_bytes < protected_floor_bytes:
        raise RangeExecutorError(
            f"unified filesystem floor crossed at {stage}: free={free} "
            f"next_bytes={additional_bytes} floor={protected_floor_bytes}"
        )
    return {
        "free_bytes": free,
        "additional_bytes": additional_bytes,
        "protected_floor_bytes": protected_floor_bytes,
    }


class _BoundedPipeCollector:
    """Continuously drain a child pipe without unbounded disk or memory growth."""

    def __init__(self, handle: BinaryIO, *, limit: int, keep_tail: bool = False) -> None:
        self.handle = handle
        self.limit = limit
        self.keep_tail = keep_tail
        self.data = bytearray()
        self.overflow = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while True:
            chunk = self.handle.read(64 * 1024)
            if not chunk:
                return
            if self.keep_tail:
                self.data.extend(chunk)
                if len(self.data) > self.limit:
                    del self.data[:-self.limit]
                    self.overflow = True
            elif len(self.data) < self.limit + 1:
                remaining = self.limit + 1 - len(self.data)
                self.data.extend(chunk[:remaining])
                self.overflow = self.overflow or len(chunk) > remaining or len(self.data) > self.limit
            else:
                self.overflow = True

    def start(self) -> None:
        self.thread.start()

    def finish(self) -> bytes:
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RangeExecutorError("operator output collector did not reach EOF")
        return bytes(self.data[: self.limit] if not self.keep_tail else self.data)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def rebuild_schedule_ranges(schedule: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rebuild all tensor ranges and bind them to the compact sealed schedule."""
    inputs = schedule.get("inputs")
    if not isinstance(inputs, Mapping):
        raise RangeExecutorError("schedule lacks sealed input bindings")
    # Schedules retain their historical root-relative bindings.  Translate
    # those logical locations only at the live read boundary.
    manifest_path = resolve_workspace_path(str(inputs.get("manifest_path", "")))
    graph_path = resolve_workspace_path(str(inputs.get("dependency_graph_path", "")))
    manifest = read_sealed_json(manifest_path)
    graph = read_sealed_json(graph_path)
    if manifest.get("seal_sha256") != inputs.get("manifest_seal_sha256"):
        raise RangeExecutorError("official manifest differs from schedule binding")
    if graph.get("seal_sha256") != inputs.get("dependency_graph_seal_sha256"):
        raise RangeExecutorError("dependency graph differs from schedule binding")
    candidate = plan_glm52_organ_windows(
        "GLM-5.2-parent-restream",
        manifest_path=manifest_path,
        graph_path=graph_path,
        artifact_bytes_per_window=2 * 1024**3,
        metadata_bytes_per_window=64 * 1024**2,
        scratch_multiplier=1.05,
    )
    if candidate["partition"]["window_partition_sha256"] != schedule["partition"]["window_partition_sha256"]:
        raise RangeExecutorError("rebuilt window partition differs from schedule")
    rows_by_window: dict[str, list[dict[str, Any]]] = {}
    for row in candidate["ranges"]:
        rows_by_window.setdefault(str(row["window_id"]), []).append(dict(row))
    manifest_files = {
        str(row["path"]): row for row in manifest.get("files", []) if isinstance(row, Mapping)
    }
    rebuilt: list[dict[str, Any]] = []
    for window in schedule["windows"]:
        window_id = str(window["window_id"])
        ranges = rows_by_window.get(window_id, [])
        ids = [str(row["range_id"]) for row in ranges]
        if len(ranges) != window["range_count"] or _sha256_list(ids) != window["ordered_range_ids_sha256"]:
            raise RangeExecutorError(f"rebuilt ranges differ for {window_id}")
        enriched = []
        for row in ranges:
            source = manifest_files.get(str(row["shard"]))
            if source is None:
                raise RangeExecutorError(f"range names a shard absent from manifest: {row['shard']}")
            xet_hash = source.get("xet_hash")
            file_bytes = source.get("logical_bytes")
            if not isinstance(xet_hash, str) or len(xet_hash) != 64 or not isinstance(file_bytes, int):
                raise RangeExecutorError(f"manifest lacks exact Xet identity for {row['shard']}")
            enriched.append({**row, "xet_hash": xet_hash, "file_bytes": file_bytes})
        accounting = window.get("incremental_accounting")
        if not isinstance(accounting, Mapping):
            raise RangeExecutorError(f"schedule window {window_id} lacks incremental accounting")
        rebuilt_source_bytes = sum(
            ((int(row["payload_bytes"]) + 64 * 1024 - 1) // (64 * 1024)) * (64 * 1024)
            for row in enriched
        )
        if rebuilt_source_bytes != accounting.get("source_range_rounded_bytes"):
            raise RangeExecutorError(
                f"rebuilt source-range accounting differs for {window_id}: "
                f"{rebuilt_source_bytes} != {accounting.get('source_range_rounded_bytes')}"
            )
        rebuilt.append({**dict(window), "ranges": enriched})
    return rebuilt


def validate_receipt_directory(receipt_dir: Path, windows: list[Mapping[str, Any]]) -> None:
    """Refuse ambiguous resume state before trusting any terminal receipt.

    A receipt name is part of the range-execution identity: one ordered window
    maps to exactly one regular JSON path.  Silently ignoring a second receipt
    (or following a symlink at the canonical name) would let a stale or forged
    side record survive a restart without being considered by the receipt
    chain.  The campaign owns this directory exclusively, so extra entries are
    never operationally necessary and must fail closed.
    """
    expected: set[str] = set()
    for window in windows:
        try:
            name = f"{int(window['execution_order']):03d}_{window['window_id']}.json"
        except (KeyError, TypeError, ValueError) as exc:
            raise RangeExecutorError("window lacks a canonical receipt identity") from exc
        if name in expected:
            raise RangeExecutorError(f"duplicate scheduled receipt identity: {name}")
        expected.add(name)

    try:
        entries = list(receipt_dir.iterdir())
    except OSError as exc:
        raise RangeExecutorError("receipt directory cannot be listed safely") from exc
    for entry in entries:
        if entry.name not in expected:
            raise RangeExecutorError(f"receipt directory has an unexpected entry: {entry.name}")
        if entry.is_symlink() or not entry.is_file():
            raise RangeExecutorError(f"receipt path is not a regular owned file: {entry.name}")


def write_frame(handle: BinaryIO, header: Mapping[str, Any], chunks: Any = ()) -> int:
    """Write one bounded metadata header followed by its declared raw payload."""
    encoded = canonical(dict(header))
    if len(encoded) > MAX_OPERATOR_RECEIPT_BYTES:
        raise RangeExecutorError("operator frame header exceeds bound")
    handle.write(struct.pack(">Q", len(encoded)))
    handle.write(encoded)
    count = 0
    for chunk in chunks:
        if not isinstance(chunk, bytes) or not chunk:
            raise RangeExecutorError("Xet stream yielded a nonempty-bytes violation")
        handle.write(chunk)
        count += len(chunk)
    return count


def _write_all_before_deadline(handle: BinaryIO, body: bytes, *, deadline: float, range_id: str) -> None:
    """Write a pipe body without permitting OS backpressure to outlive a range.

    The production child stdin is a POSIX pipe.  ``BufferedWriter.write`` can
    block indefinitely when a failed operator stops reading, so use readiness
    polling and ``os.write`` for this one authority path.  File-like fixture
    handles intentionally retain the ordinary bounded writer behaviour.
    """
    try:
        descriptor = handle.fileno()
    except (AttributeError, OSError, io.UnsupportedOperation):
        if time.monotonic() >= deadline:
            raise RangeExecutorError(f"operator pipe deadline expired: {range_id}")
        handle.write(body)
        if time.monotonic() >= deadline:
            raise RangeExecutorError(f"operator pipe deadline expired: {range_id}")
        return
    was_blocking = os.get_blocking(descriptor)
    os.set_blocking(descriptor, False)
    try:
        view = memoryview(body)
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RangeExecutorError(f"operator pipe deadline expired: {range_id}")
            _readable, writable, _failed = select.select([], [descriptor], [], remaining)
            if not writable:
                raise RangeExecutorError(f"operator pipe deadline expired: {range_id}")
            try:
                written = os.write(descriptor, view)
            except BlockingIOError:
                continue
            except BrokenPipeError as exc:
                raise RangeExecutorError(f"window operator closed stdin during {range_id}") from exc
            if written <= 0:
                raise RangeExecutorError(f"window operator made no write progress during {range_id}")
            view = view[written:]
    finally:
        os.set_blocking(descriptor, was_blocking)


def write_frame_before_deadline(
    handle: BinaryIO, header: Mapping[str, Any], chunks: Any = (), *, deadline: float, range_id: str,
) -> int:
    """Write an exact frame, bounding header and every body write by one deadline."""
    encoded = canonical(dict(header))
    if len(encoded) > MAX_OPERATOR_RECEIPT_BYTES:
        raise RangeExecutorError("operator frame header exceeds bound")
    _write_all_before_deadline(handle, struct.pack(">Q", len(encoded)), deadline=deadline, range_id=range_id)
    _write_all_before_deadline(handle, encoded, deadline=deadline, range_id=range_id)
    count = 0
    for chunk in chunks:
        if not isinstance(chunk, bytes) or not chunk:
            raise RangeExecutorError("Xet stream yielded a nonempty-bytes violation")
        _write_all_before_deadline(handle, chunk, deadline=deadline, range_id=range_id)
        count += len(chunk)
    return count


def _read_exact(handle: BinaryIO, count: int, *, label: str) -> bytes:
    """Read exactly ``count`` bytes or reject a truncated framed stream."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise RangeExecutorError(f"{label} byte count is invalid")
    pieces: list[bytes] = []
    remaining = count
    while remaining:
        piece = handle.read(remaining)
        if not isinstance(piece, bytes) or not piece:
            raise RangeExecutorError(f"truncated {label}")
        pieces.append(piece)
        remaining -= len(piece)
    return b"".join(pieces)


def read_frame(
    handle: BinaryIO,
    *,
    max_header_bytes: int = MAX_FRAME_HEADER_BYTES,
    max_payload_bytes: int = MAX_OPERATOR_RECEIPT_BYTES,
) -> FramedMessage:
    """Parse one exact framed message without accepting trailing ambiguity.

    The bounded fixture path intentionally sets a small payload maximum.  The
    live executor never calls this reader: model-sized range bodies continue to
    flow directly to an owner-approved external operator.  This keeps a test
    parser from accidentally becoming an unreviewed parent-restream path.
    """
    prefix = _read_exact(handle, 8, label="frame header length")
    header_bytes = int.from_bytes(prefix, "big")
    if header_bytes <= 0 or header_bytes > max_header_bytes:
        raise RangeExecutorError("framed header length is outside the declared bound")
    try:
        decoded = json.loads(_read_exact(handle, header_bytes, label="frame header"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RangeExecutorError("framed header is not a JSON object") from exc
    if not isinstance(decoded, dict):
        raise RangeExecutorError("framed header must be a JSON object")
    kind = decoded.get("kind")
    if not isinstance(kind, str) or not kind:
        raise RangeExecutorError("framed header requires a non-empty kind")
    declared = decoded.get("payload_bytes", 0)
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
        raise RangeExecutorError("framed payload_bytes must be a non-negative integer")
    if declared > max_payload_bytes:
        raise RangeExecutorError("framed payload exceeds the fixture bound")
    if kind == "RANGE" and "payload_bytes" not in decoded:
        raise RangeExecutorError("RANGE frame must declare payload_bytes")
    if kind != "RANGE" and declared != 0:
        raise RangeExecutorError("only RANGE frames may carry a payload")
    return FramedMessage(header=decoded, payload=_read_exact(handle, declared, label="frame payload"))


def validate_window_receipt(
    receipt: Mapping[str, Any], *, schedule: Mapping[str, Any], policy: Mapping[str, Any], window: Mapping[str, Any],
    operator_receipt_public_key_bytes: bytes, predecessor_receipt_seal_sha256: str | None,
    launch_nonce_sha256: str, lease_identity_sha256: str, operator_executable_sha256: str,
) -> dict[str, Any]:
    value = verify_sealed(dict(receipt), label="range window terminal receipt")
    required = {
        "schema", "status", "schedule_seal_sha256", "policy_seal_sha256", "window_id",
        "execution_order", "range_count", "ordered_range_ids_sha256", "payload_bytes",
        "source_range_identity_sha256", "payload_hash_chain_sha256",
        "predecessor_window_receipt_seal_sha256", "launch_nonce_sha256",
        "lease_identity_sha256", "operator_executable_sha256",
        "artifact_handoff", "eviction", "attestation", "seal_sha256",
    }
    if set(value) != required or value.get("schema") != RECEIPT_SCHEMA or value.get("status") != "PASS_SEALED_REMOTE_HASHED_EVICTED":
        raise RangeExecutorError("window operator returned an incomplete terminal receipt")
    expected = {
        "schedule_seal_sha256": schedule["seal_sha256"],
        "policy_seal_sha256": policy["seal_sha256"],
        "window_id": window["window_id"],
        "execution_order": window["execution_order"],
        "range_count": window["range_count"],
        "ordered_range_ids_sha256": window["ordered_range_ids_sha256"],
        "payload_bytes": sum(int(row["payload_bytes"]) for row in window["ranges"]),
        "source_range_identity_sha256": range_source_identity_sha256(window),
        "predecessor_window_receipt_seal_sha256": predecessor_receipt_seal_sha256,
        "launch_nonce_sha256": launch_nonce_sha256,
        "lease_identity_sha256": lease_identity_sha256,
        "operator_executable_sha256": operator_executable_sha256,
    }
    if any(value.get(key) != expected_item for key, expected_item in expected.items()):
        raise RangeExecutorError("window receipt identity or byte count differs from schedule")
    payload_chain = value.get("payload_hash_chain_sha256")
    if (
        not isinstance(payload_chain, str)
        or len(payload_chain) != 64
        or any(character not in "0123456789abcdef" for character in payload_chain)
    ):
        raise RangeExecutorError("window receipt lacks an exact payload hash chain")
    handoff = value.get("artifact_handoff")
    eviction = value.get("eviction")
    if not isinstance(handoff, Mapping) or not isinstance(eviction, Mapping):
        raise RangeExecutorError("window receipt lacks handoff/eviction mappings")
    if handoff.get("cold_upload_complete") is not True or handoff.get("remote_sha256_verified") is not True:
        raise RangeExecutorError("window artifact lacks verified cold remote handoff")
    remote_sha = handoff.get("remote_sha256")
    if not isinstance(remote_sha, str) or len(remote_sha) != 64:
        raise RangeExecutorError("window artifact lacks an exact remote SHA-256")
    booleans = (
        "source_ranges_retained_zero", "local_artifact_bytes_retained_zero", "temp_bytes_retained_zero",
        "exact_cache_items_purged", "exact_trash_items_purged", "free_byte_recovery_measured",
    )
    if any(eviction.get(name) is not True for name in booleans):
        raise RangeExecutorError("window receipt does not prove exact terminal eviction")
    before, after, delta = (eviction.get(name) for name in ("free_bytes_before_eviction", "free_bytes_after_eviction", "recovered_bytes"))
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (before, after, delta)):
        raise RangeExecutorError("window eviction free-byte evidence is invalid")
    if after - before != delta:
        raise RangeExecutorError("window eviction recovery arithmetic is not exact")
    attestation = value.get("attestation")
    if not isinstance(attestation, Mapping) or set(attestation) != {
        "algorithm", "public_key_sha256", "signature_ed25519_hex",
    }:
        raise RangeExecutorError("window receipt lacks an exact operator attestation")
    if len(operator_receipt_public_key_bytes) != 32:
        raise RangeExecutorError("approved operator receipt public key is invalid")
    key_hash = hashlib.sha256(operator_receipt_public_key_bytes).hexdigest()
    signature_hex = attestation.get("signature_ed25519_hex")
    if (
        attestation.get("algorithm") != "Ed25519"
        or attestation.get("public_key_sha256") != key_hash
        or not isinstance(signature_hex, str)
        or len(signature_hex) != 128
    ):
        raise RangeExecutorError("window receipt operator attestation identity differs from approval")
    signed_body = {key: item for key, item in value.items() if key not in {"seal_sha256", "attestation"}}
    try:
        Ed25519PublicKey.from_public_bytes(operator_receipt_public_key_bytes).verify(
            bytes.fromhex(signature_hex), canonical(signed_body)
        )
    except (ValueError, InvalidSignature) as exc:
        raise RangeExecutorError("window receipt operator attestation verification failed") from exc
    return value


def execute(
    schedule: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    operator_path: Path,
    receipt_dir: Path,
    workspace_root: Path,
    final_preflight: Mapping[str, Any],
    launch_claim_path: Path,
    range_executor_path: Path,
    operator_timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Execute or restart one campaign while holding its exclusive OS lock."""
    with _exclusive_claim_execution(launch_claim_path):
        return _execute_locked(
            schedule,
            policy,
            operator_path=operator_path,
            receipt_dir=receipt_dir,
            workspace_root=workspace_root,
            final_preflight=final_preflight,
            launch_claim_path=launch_claim_path,
            range_executor_path=range_executor_path,
            operator_timeout_seconds=operator_timeout_seconds,
        )


def _execute_locked(
    schedule: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    operator_path: Path,
    receipt_dir: Path,
    workspace_root: Path,
    final_preflight: Mapping[str, Any],
    launch_claim_path: Path,
    range_executor_path: Path,
    operator_timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Execute or resume the sealed window sequence after external authorization."""
    validate_bounded_restream(schedule, policy)
    try:
        preflight = verify_sealed(dict(final_preflight), label="green-light final preflight")
    except Exception as exc:  # noqa: BLE001
        raise RangeExecutorError(f"final preflight receipt is invalid: {exc}") from exc
    if (
        preflight.get("schema") != TRANSITION_SCHEMA
        or preflight.get("status") != "FINAL_PREFLIGHT"
        or preflight.get("production_authority") is not True
        or preflight.get("restream_started") is not False
        or preflight.get("exact_next_action") != "EXEC_RANGE_RESTREAM"
        or preflight.get("schedule_seal_sha256") != schedule.get("seal_sha256")
        or preflight.get("policy_seal_sha256") != policy.get("seal_sha256")
    ):
        raise RangeExecutorError("signed green-light FINAL_PREFLIGHT is absent or bound to different inputs")
    if schedule.get("repo") != REPO_ID or schedule.get("revision") != REVISION:
        raise RangeExecutorError("schedule is not bound to the pinned final parent")
    if os.environ.get("HAWKING_PARENT_RESTREAM_AUTHORIZED") != "YES":
        raise RangeExecutorError("owner parent-restream authorization is absent")
    lease = os.environ.get("HAWKING_CLEAN_GPU_LEASE_ID", "")
    if not lease:
        raise RangeExecutorError("clean GPU lease identity is absent")
    if not operator_path.is_file() or not os.access(operator_path, os.X_OK):
        raise RangeExecutorError("tested executable window operator is absent")
    owner_key = load_pinned_owner_public_key()
    if owner_key is None:
        raise RangeExecutorError("fixed OS owner trust anchor is absent or insecure")
    try:
        launch = start_claimed_owner_authorization(
            launch_claim_path,
            schedule=schedule,
            policy=policy,
            final_preflight=preflight,
            operator_path=operator_path,
            range_executor_path=range_executor_path,
            public_key_bytes=owner_key,
            allow_started_resume=True,
        )
    except Exception as exc:  # noqa: BLE001 - normalize authority failures.
        raise RangeExecutorError(f"signed single-use launch capability refused: {exc}") from exc
    lease_snapshot = launch.get("production_lease_receipt")
    approval_snapshot = launch.get("operator_approval_receipt")
    if not isinstance(lease_snapshot, Mapping) or not isinstance(approval_snapshot, Mapping):
        raise RangeExecutorError("started launch capability lacks lease/operator snapshots")
    if lease_snapshot.get("lease_id") != lease:
        raise RangeExecutorError("started launch capability differs from the live lease identity")
    try:
        operator_receipt_public_key = bytes.fromhex(str(approval_snapshot["receipt_public_key_ed25519_hex"]))
    except (KeyError, ValueError) as exc:
        raise RangeExecutorError("started launch capability lacks the approved receipt key") from exc
    launch_nonce_sha256 = str(launch.get("nonce_sha256", ""))
    lease_identity_sha256 = hashlib.sha256(lease.encode()).hexdigest()
    operator_executable_sha256 = hashlib.sha256(operator_path.read_bytes()).hexdigest()
    windows = rebuild_schedule_ranges(schedule)
    timeout_seconds = _operator_timeout_seconds() if operator_timeout_seconds is None else operator_timeout_seconds
    range_timeout_seconds = _range_timeout_seconds()
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not MIN_OPERATOR_TIMEOUT_SECONDS <= timeout_seconds <= MAX_OPERATOR_TIMEOUT_SECONDS:
        raise RangeExecutorError("operator timeout is outside the production bound")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    validate_receipt_directory(receipt_dir, windows)
    accepted: list[str] = []
    protected_floor_bytes = int(policy["policy"]["protected_filesystem_floor_bytes"])

    try:
        from hf_xet import XetFileInfo, XetSession
        from lab.operators.glm52_xet_live import _public_hub_stream_group, _runtime_config
    except ImportError as exc:
        raise RangeExecutorError("pinned official Xet runtime is unavailable") from exc
    runtime = _runtime_config()
    session = XetSession()
    group, public_auth = _public_hub_stream_group(session)
    try:
        for window in windows:
            receipt_path = receipt_dir / f"{int(window['execution_order']):03d}_{window['window_id']}.json"
            if receipt_path.is_file():
                receipt = validate_window_receipt(
                    read_sealed_json(receipt_path), schedule=schedule, policy=policy, window=window,
                    operator_receipt_public_key_bytes=operator_receipt_public_key,
                    predecessor_receipt_seal_sha256=accepted[-1] if accepted else None,
                    launch_nonce_sha256=launch_nonce_sha256,
                    lease_identity_sha256=lease_identity_sha256,
                    operator_executable_sha256=operator_executable_sha256,
                )
                accepted.append(receipt["seal_sha256"])
                continue
            if window["predecessor_window_id"] is not None and not accepted:
                raise RangeExecutorError("predecessor receipt is absent")
            free_bytes = shutil.disk_usage(workspace_root).free
            model_roots = (
                workspace_root / "workspace/ops/local/models",
                workspace_root / "models",  # historical location; never silently ignore it
            )
            if any(root.exists() and any(path.is_file() for path in root.rglob("*")) for root in model_roots):
                raise RangeExecutorError("one-active-model gate refuses resident model-lane files")
            admission = live_window_admission(
                schedule, policy, window_id=str(window["window_id"]), free_bytes=free_bytes
            )
            if admission["admitted"] is not True:
                raise RangeExecutorError(f"unified filesystem floor refuses {window['window_id']}")
            process = subprocess.Popen(
                [str(operator_path), "--protocol", PROTOCOL, "--window-id", str(window["window_id"])],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,
            )
            assert process.stdin is not None and process.stdout is not None and process.stderr is not None
            stdout_collector = _BoundedPipeCollector(process.stdout, limit=MAX_OPERATOR_RECEIPT_BYTES)
            stderr_collector = _BoundedPipeCollector(process.stderr, limit=2000, keep_tail=True)
            stdout_collector.start()
            stderr_collector.start()
            timed_out = threading.Event()
            range_timed_out = threading.Event()

            def expire_operator() -> None:
                if process.poll() is None:
                    timed_out.set()
                    _terminate_process_group(process)

            def expire_range() -> None:
                """Abort both endpoints when a per-range deadline is exhausted."""
                range_timed_out.set()
                _terminate_process_group(process)
                try:
                    session.sigint_abort()
                except Exception:
                    pass

            watchdog = threading.Timer(timeout_seconds, expire_operator)
            watchdog.daemon = True
            watchdog.start()
            payload_hashes: list[dict[str, str]] = []
            source_identity = range_source_identity_sha256(window)
            try:
                write_frame_before_deadline(process.stdin, {
                    "kind": "WINDOW", "protocol": PROTOCOL, "schedule_seal_sha256": schedule["seal_sha256"],
                    "policy_seal_sha256": policy["seal_sha256"], "lease_identity_sha256": hashlib.sha256(lease.encode()).hexdigest(),
                    "launch_nonce_sha256": launch_nonce_sha256,
                    "predecessor_window_receipt_seal_sha256": accepted[-1] if accepted else None,
                    "operator_executable_sha256": operator_executable_sha256,
                    "protected_floor_bytes": protected_floor_bytes,
                    "incremental_accounting": window["incremental_accounting"],
                    "window_id": window["window_id"], "execution_order": window["execution_order"],
                    "range_count": window["range_count"], "ordered_range_ids_sha256": window["ordered_range_ids_sha256"],
                    "source_range_identity_sha256": source_identity,
                }, deadline=time.monotonic() + range_timeout_seconds, range_id=f"{window['window_id']}:window-header")
                for row in window["ranges"]:
                    assert_unified_floor(
                        workspace_root, protected_floor_bytes=protected_floor_bytes,
                        stage=f"before range {row['range_id']}",
                    )
                    range_deadline = time.monotonic() + range_timeout_seconds
                    stream = _call_with_deadline(
                        lambda row=row: group.download_stream(
                            XetFileInfo(row["xet_hash"], row["file_bytes"]), start=row["start"], end=row["end"]
                        ),
                        range_id=str(row["range_id"]),
                        timeout_seconds=max(0.001, range_deadline - time.monotonic()),
                        on_timeout=expire_range,
                    )
                    deadline_stream = _RangeDeadlineChunks(
                        stream,
                        range_id=str(row["range_id"]),
                        timeout_seconds=max(0.001, range_deadline - time.monotonic()),
                    )
                    hashing_stream = _HashingChunks(
                        deadline_stream,
                        before_chunk=lambda size, range_id=row["range_id"]: assert_unified_floor(
                            workspace_root,
                            protected_floor_bytes=protected_floor_bytes,
                            additional_bytes=size,
                            stage=f"streaming range {range_id}",
                        ),
                    )
                    try:
                        count = write_frame_before_deadline(process.stdin, {
                            "kind": "RANGE", "range_id": row["range_id"], "shard": row["shard"],
                            "name": row["name"], "start": row["start"], "end": row["end"],
                            "payload_bytes": row["payload_bytes"], "xet_hash": row["xet_hash"],
                            "file_bytes": row["file_bytes"],
                        }, hashing_stream, deadline=range_deadline, range_id=str(row["range_id"]))
                    except BaseException:
                        deadline_stream.cancel()
                        raise
                    if count != row["payload_bytes"] or hashing_stream.count != count:
                        raise RangeExecutorError(f"Xet range length mismatch: {row['range_id']}")
                    assert_unified_floor(
                        workspace_root, protected_floor_bytes=protected_floor_bytes,
                        stage=f"after range {row['range_id']}",
                    )
                    payload_hashes.append({"range_id": str(row["range_id"]), "sha256": hashing_stream.sha256})
                payload_hash_chain = hashlib.sha256(canonical(payload_hashes)).hexdigest()
                assert_unified_floor(
                    workspace_root, protected_floor_bytes=protected_floor_bytes,
                    stage=f"before handoff {window['window_id']}",
                )
                write_frame_before_deadline(process.stdin, {
                    "kind": "END_WINDOW", "window_id": window["window_id"],
                    "source_range_identity_sha256": source_identity,
                    "payload_hash_chain_sha256": payload_hash_chain,
                }, deadline=time.monotonic() + range_timeout_seconds, range_id=f"{window['window_id']}:end-window")
                process.stdin.close()
                code = process.wait()
            except BaseException:
                _terminate_process_group(process)
                process.wait()
                raise
            finally:
                watchdog.cancel()
            output = stdout_collector.finish()
            error = stderr_collector.finish()
            if stdout_collector.overflow:
                raise RangeExecutorError(f"window operator stdout exceeded {MAX_OPERATOR_RECEIPT_BYTES} bytes")
            if timed_out.is_set():
                raise RangeExecutorError(f"window operator timed out after {timeout_seconds}s for {window['window_id']}")
            if range_timed_out.is_set():
                raise RangeExecutorError(f"Xet range deadline expired while executing {window['window_id']}")
            if code != 0 or len(output) > MAX_OPERATOR_RECEIPT_BYTES:
                raise RangeExecutorError(f"window operator failed for {window['window_id']}: {error[-2000:]!r}")
            try:
                receipt = validate_window_receipt(
                    json.loads(output), schedule=schedule, policy=policy, window=window,
                    operator_receipt_public_key_bytes=operator_receipt_public_key,
                    predecessor_receipt_seal_sha256=accepted[-1] if accepted else None,
                    launch_nonce_sha256=launch_nonce_sha256,
                    lease_identity_sha256=lease_identity_sha256,
                    operator_executable_sha256=operator_executable_sha256,
                )
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RangeExecutorError("window operator receipt is not bounded JSON") from exc
            if receipt.get("payload_hash_chain_sha256") != payload_hash_chain:
                raise RangeExecutorError("window operator payload hash chain differs from streamed bodies")
            from lab.operators.glm52_common import atomic_json
            atomic_json(receipt_path, receipt)
            accepted.append(receipt["seal_sha256"])
    finally:
        try:
            session.sigint_abort()
        except Exception:
            pass
    return seal({
        "schema": TERMINAL_SCHEMA,
        "status": "PASS_ALL_WINDOWS_REMOTE_HASHED_AND_EVICTED",
        "schedule_seal_sha256": schedule["seal_sha256"],
        "policy_seal_sha256": policy["seal_sha256"],
        "final_preflight_seal_sha256": preflight["seal_sha256"],
        "window_receipt_seals": accepted,
        "window_count": len(accepted),
        "runtime": runtime,
        "public_hub_auth": public_auth,
        "range_deadline_seconds": range_timeout_seconds,
    })

"""Fail-closed bounded header streaming for DeepSeek-V4-Flash.

This is deliberately *not* a downloader, decoder, packer, or runtime.  It is
the small physical gate immediately before those systems: a caller can seal a
plan for an exact, header-only HTTP byte range and then hand a locally captured
header to this module for bounded validation.  Tensor bodies are rejected.

The transport remains external on purpose.  That keeps this authority from
silently introducing an unbounded cache or a second model-body download path.
The future transport must deliver a requested range directly to a local header
capture, after which this module verifies the capture, checks the filesystem
floor before/during/after it, asserts declared source-retention locations are
empty, and atomically seals a receipt.

No result from this module establishes native FP4/FP8 decoding, Condense
packing, a Gravity artifact, a CPU oracle, a Metal forward, capability, or
throughput.  Those remain separate future execution gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from lab.receipts import SealIntegrityError, seal, verify


PLAN_SCHEMA = "hawking.gravity.deepseek_v4.bounded_stream_plan.v1"
PLAN_PREFLIGHT_SCHEMA = "hawking.gravity.deepseek_v4.bounded_stream_preflight.v1"
HEADER_RECEIPT_SCHEMA = "hawking.condense.deepseek_v4.header_range_execution.v1"

EXPECTED_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash"
MIN_FREE_FLOOR_BYTES = 15 * 1024**3
MAX_HEADER_CAPTURE_BYTES = 8 * 1024**2
DEFAULT_MAX_INFLIGHT_BYTES = 1024**2
MAX_RECEIPT_BYTES = 1024**2
DEFAULT_CHUNK_BYTES = 256 * 1024
MAX_RETENTION_DIRECTORY_ENTRIES = 4096

_SHA256_LENGTH = 64
_REVISION_LENGTH = 40


class DeepSeekV4StreamError(RuntimeError):
    """A bounded-stream plan, floor, or retention invariant failed closed."""


FreeBytesProvider = Callable[[Path], int]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeepSeekV4StreamError(f"{label} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise DeepSeekV4StreamError(f"{label} must be {qualifier}")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeepSeekV4StreamError(f"{label} must be a non-empty string")
    return value


def _hex(value: object, label: str, length: int) -> str:
    result = _nonempty_string(value, label)
    if len(result) != length:
        raise DeepSeekV4StreamError(f"{label} must be {length} hexadecimal characters")
    try:
        int(result, 16)
    except ValueError as exc:
        raise DeepSeekV4StreamError(f"{label} must be hexadecimal") from exc
    return result.lower()


def _absolute_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise DeepSeekV4StreamError(f"{label} must be an absolute path")
    # Do not call ``Path.resolve`` here: callers that supplied a terminal
    # symlink must remain visible to the later no-symlink safety checks.
    return Path(os.path.abspath(os.fspath(path)))


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _normalise_retention_paths(values: Sequence[str | Path]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise DeepSeekV4StreamError("source_retention_paths must be a sequence of paths")
    result: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        path = _absolute_path(value, f"source_retention_paths[{index}]")
        encoded = str(path)
        if encoded not in seen:
            seen.add(encoded)
            result.append(encoded)
    return result


def _normalise_header_range(value: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DeepSeekV4StreamError(f"header_ranges[{index}] must be an object")
    range_id = _nonempty_string(value.get("range_id"), f"header_ranges[{index}].range_id")
    shard = _nonempty_string(value.get("shard"), f"header_ranges[{index}].shard")
    if Path(shard).name != shard:
        raise DeepSeekV4StreamError(f"header_ranges[{index}].shard must be a filename, not a path")
    kind = value.get("kind", "safetensors_header")
    if kind != "safetensors_header":
        raise DeepSeekV4StreamError(
            f"header_ranges[{index}] requests {kind!r}; tensor/body ranges are not implemented"
        )
    start = _positive_int(value.get("start"), f"header_ranges[{index}].start", allow_zero=True)
    end = _positive_int(value.get("end"), f"header_ranges[{index}].end")
    if start != 0:
        raise DeepSeekV4StreamError(
            f"header_ranges[{index}] must start at byte zero to include the safetensors header length"
        )
    if end <= start:
        raise DeepSeekV4StreamError(f"header_ranges[{index}] has a non-positive interval")
    expected_bytes = end - start
    if expected_bytes > MAX_HEADER_CAPTURE_BYTES + 8:
        raise DeepSeekV4StreamError(
            f"header_ranges[{index}] exceeds the bounded header capture limit of "
            f"{MAX_HEADER_CAPTURE_BYTES + 8} bytes"
        )
    supplied = value.get("expected_bytes", expected_bytes)
    if _positive_int(supplied, f"header_ranges[{index}].expected_bytes") != expected_bytes:
        raise DeepSeekV4StreamError(
            f"header_ranges[{index}].expected_bytes must equal end - start"
        )
    expected_sha256 = value.get("expected_capture_sha256")
    if expected_sha256 is not None:
        expected_sha256 = _hex(
            expected_sha256,
            f"header_ranges[{index}].expected_capture_sha256",
            _SHA256_LENGTH,
        )
    return {
        "range_id": range_id,
        "shard": shard,
        "kind": "safetensors_header",
        "start": start,
        "end": end,
        "expected_bytes": expected_bytes,
        "expected_capture_sha256": expected_sha256,
    }


def _execution_boundary() -> dict[str, Any]:
    """Keep a header receipt from being mistaken for a model execution receipt."""
    return {
        "mode": "header_only",
        "source_body_bytes_consumed": 0,
        "source_body_persisted_by_executor": False,
        "native_fp4_decode": "not_implemented",
        "native_fp8_decode": "not_implemented",
        "router_execution": "not_executed",
        "mhc_execution": "not_executed",
        "attention_execution": "not_executed",
        "condense_packing": "not_implemented",
        "gravity_artifact": "not_created",
        "cpu_oracle": "not_executed",
        "metal_forward": "not_executed",
        "capability": "unknown",
        "throughput": "unknown",
    }


def build_plan(
    *,
    repository: str,
    revision: str,
    header_ranges: Sequence[Mapping[str, Any]],
    source_retention_paths: Sequence[str | Path] = (),
    protected_floor_bytes: int = MIN_FREE_FLOOR_BYTES,
    max_inflight_bytes: int = DEFAULT_MAX_INFLIGHT_BYTES,
) -> dict[str, Any]:
    """Create a sealed, non-executing plan for one or more header captures.

    Only a safetensors header beginning at byte zero is accepted.  Any plan for
    a tensor body is rejected before transport is considered.
    """
    if repository != EXPECTED_REPOSITORY:
        raise DeepSeekV4StreamError(
            f"repository must be the exact DeepSeek-V4-Flash source: {EXPECTED_REPOSITORY}"
        )
    revision = _hex(revision, "revision", _REVISION_LENGTH)
    if isinstance(header_ranges, (str, bytes)) or not header_ranges:
        raise DeepSeekV4StreamError("header_ranges must be a non-empty sequence")
    normalised = [_normalise_header_range(row, index=index) for index, row in enumerate(header_ranges)]
    ids = [row["range_id"] for row in normalised]
    if len(ids) != len(set(ids)):
        raise DeepSeekV4StreamError("header_ranges contains duplicate range_id values")
    protected_floor_bytes = _positive_int(protected_floor_bytes, "protected_floor_bytes")
    if protected_floor_bytes < MIN_FREE_FLOOR_BYTES:
        raise DeepSeekV4StreamError(
            f"protected_floor_bytes cannot be below the non-negotiable {MIN_FREE_FLOOR_BYTES}-byte floor"
        )
    max_inflight_bytes = _positive_int(max_inflight_bytes, "max_inflight_bytes")
    if max_inflight_bytes > MAX_HEADER_CAPTURE_BYTES + 8:
        raise DeepSeekV4StreamError("max_inflight_bytes exceeds the bounded header-only limit")
    if any(row["expected_bytes"] > max_inflight_bytes for row in normalised):
        raise DeepSeekV4StreamError(
            "max_inflight_bytes must cover every bounded header capture; body spilling is not allowed"
        )
    return seal(
        {
            "schema": PLAN_SCHEMA,
            "status": "REQUESTED_NOT_EXECUTED",
            "created_at": _utc_now(),
            "source": {
                "repository": repository,
                "revision": revision,
                "source_authority": "not_provided",
            },
            "modes": {
                "plan_only": True,
                "header_only": True,
                "tensor_body_streaming": False,
                "decoder_or_packer_execution": False,
            },
            "header_ranges": normalised,
            "storage_policy": {
                "protected_floor_bytes": protected_floor_bytes,
                "max_inflight_bytes": max_inflight_bytes,
                "floor_enforced_before_during_after_every_range": True,
                "no_source_body_accumulation": True,
                "source_retention_paths": _normalise_retention_paths(source_retention_paths),
            },
            "execution_boundary": _execution_boundary(),
        }
    )


def validate_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a plan seal and reapply every no-body storage invariant."""
    try:
        plan = verify(value, label="DeepSeek-V4 bounded stream plan")
    except SealIntegrityError as exc:
        raise DeepSeekV4StreamError(str(exc)) from exc
    if plan.get("schema") != PLAN_SCHEMA:
        raise DeepSeekV4StreamError("plan schema is not the bounded DeepSeek-V4 stream schema")
    if plan.get("status") != "REQUESTED_NOT_EXECUTED":
        raise DeepSeekV4StreamError("plan status must remain REQUESTED_NOT_EXECUTED")
    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise DeepSeekV4StreamError("plan lacks source binding")
    repository = source.get("repository")
    revision = source.get("revision")
    if repository != EXPECTED_REPOSITORY:
        raise DeepSeekV4StreamError("plan repository differs from DeepSeek-V4-Flash")
    _hex(revision, "plan revision", _REVISION_LENGTH)
    if source.get("source_authority") != "not_provided":
        raise DeepSeekV4StreamError("this scaffold cannot accept a source-authority claim")
    modes = plan.get("modes")
    if not isinstance(modes, Mapping) or modes != {
        "plan_only": True,
        "header_only": True,
        "tensor_body_streaming": False,
        "decoder_or_packer_execution": False,
    }:
        raise DeepSeekV4StreamError("plan mode boundary is invalid")
    rows = plan.get("header_ranges")
    if not isinstance(rows, list) or not rows:
        raise DeepSeekV4StreamError("plan lacks header_ranges")
    normalised = [_normalise_header_range(row, index=index) for index, row in enumerate(rows)]
    if normalised != rows:
        raise DeepSeekV4StreamError("plan header_ranges are not canonical")
    if len({row["range_id"] for row in normalised}) != len(normalised):
        raise DeepSeekV4StreamError("plan has duplicate range_id values")
    policy = plan.get("storage_policy")
    if not isinstance(policy, Mapping):
        raise DeepSeekV4StreamError("plan lacks storage_policy")
    floor = _positive_int(policy.get("protected_floor_bytes"), "plan protected_floor_bytes")
    if floor < MIN_FREE_FLOOR_BYTES:
        raise DeepSeekV4StreamError("plan lowers the non-negotiable 15 GiB floor")
    inflight = _positive_int(policy.get("max_inflight_bytes"), "plan max_inflight_bytes")
    if inflight > MAX_HEADER_CAPTURE_BYTES + 8 or any(row["expected_bytes"] > inflight for row in normalised):
        raise DeepSeekV4StreamError("plan permits an unbounded or underspecified header capture")
    if policy.get("floor_enforced_before_during_after_every_range") is not True:
        raise DeepSeekV4StreamError("plan does not require before/during/after floor enforcement")
    if policy.get("no_source_body_accumulation") is not True:
        raise DeepSeekV4StreamError("plan does not prohibit source-body accumulation")
    roots = policy.get("source_retention_paths")
    if not isinstance(roots, list) or _normalise_retention_paths(roots) != roots:
        raise DeepSeekV4StreamError("plan source_retention_paths are not canonical absolute paths")
    if plan.get("execution_boundary") != _execution_boundary():
        raise DeepSeekV4StreamError("plan execution boundary is invalid")
    return plan


def _read_free_bytes(workspace_root: Path, provider: FreeBytesProvider | None) -> int:
    if provider is None:
        try:
            value = shutil.disk_usage(workspace_root).free
        except OSError as exc:
            raise DeepSeekV4StreamError(f"cannot sample filesystem free bytes: {exc}") from exc
    else:
        value = provider(workspace_root)
    return _positive_int(value, "filesystem free bytes", allow_zero=True)


def assert_floor(
    workspace_root: str | Path,
    *,
    protected_floor_bytes: int,
    additional_bytes: int,
    stage: str,
    free_bytes_provider: FreeBytesProvider | None = None,
) -> dict[str, Any]:
    """Sample and enforce ``free - next_allocation >= 15 GiB``.

    ``additional_bytes`` is deliberately the full declared in-flight header
    budget even though this implementation retains it in memory.  That makes
    the preflight conservative against transport/cache behaviour outside this
    module.
    """
    root = _absolute_path(workspace_root, "workspace_root")
    floor = _positive_int(protected_floor_bytes, "protected_floor_bytes")
    if floor < MIN_FREE_FLOOR_BYTES:
        raise DeepSeekV4StreamError("protected floor cannot be below 15 GiB")
    additional = _positive_int(additional_bytes, "additional_bytes", allow_zero=True)
    free = _read_free_bytes(root, free_bytes_provider)
    remaining = free - additional
    if remaining < floor:
        raise DeepSeekV4StreamError(
            f"15 GiB filesystem floor crossed at {stage}: free={free} "
            f"next_bytes={additional} floor={floor}"
        )
    return {
        "stage": _nonempty_string(stage, "floor stage"),
        "observed_at": _utc_now(),
        "free_bytes": free,
        "additional_bytes": additional,
        "remaining_bytes": remaining,
        "protected_floor_bytes": floor,
        "status": "PASS",
    }


def assert_source_evicted(retention_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Assert only declared transport/cache roots contain no retained source.

    Nothing is deleted here.  A retained file, special node, or symlink fails
    the range before it can be represented as evicted.  Directory-only
    scaffolding is permitted because Xet may create empty session/cache
    directories even when its persistent chunk cache is set to zero.  The
    caller controls which dedicated Xet/HTTP cache and staging roots are
    declared, so the receipt's scope remains precise instead of falsely
    claiming a global cache audit.
    """
    paths = _normalise_retention_paths(list(retention_paths))
    if not paths:
        raise DeepSeekV4StreamError(
            "header execution requires at least one declared source_retention_path for eviction assertions"
        )
    result: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        try:
            node = os.lstat(path)
        except FileNotFoundError:
            result.append({"path": raw, "state": "ABSENT"})
            continue
        except OSError as exc:
            raise DeepSeekV4StreamError(f"cannot inspect source retention path {path}: {exc}") from exc
        if stat.S_ISLNK(node.st_mode):
            raise DeepSeekV4StreamError(f"source retention path must not be a symlink: {path}")
        if not stat.S_ISDIR(node.st_mode):
            raise DeepSeekV4StreamError(
                f"source retention path retains a non-directory source/cache object: {path}"
            )
        directories = 1
        stack = [path]
        while stack:
            directory = stack.pop()
            try:
                entries = list(os.scandir(directory))
            except OSError as exc:
                raise DeepSeekV4StreamError(
                    f"cannot inspect source retention directory {directory}: {exc}"
                ) from exc
            for entry in entries:
                try:
                    child = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise DeepSeekV4StreamError(
                        f"cannot inspect source retention entry {entry.path}: {exc}"
                    ) from exc
                if stat.S_ISLNK(child.st_mode):
                    raise DeepSeekV4StreamError(
                        f"source retention path contains a symlink; eviction is unproven: {entry.path}"
                    )
                if not stat.S_ISDIR(child.st_mode):
                    raise DeepSeekV4StreamError(
                        f"source retention path retains a file or special node; eviction is unproven: {entry.path}"
                    )
                directories += 1
                if directories > MAX_RETENTION_DIRECTORY_ENTRIES:
                    raise DeepSeekV4StreamError(
                        "source retention directory scaffolding exceeds the bounded inspection limit"
                    )
                stack.append(Path(entry.path))
        result.append(
            {
                "path": raw,
                "state": "EMPTY_DIRECTORY" if directories == 1 else "DIRECTORY_SCAFFOLDING_ONLY",
                "directory_count": directories,
                "retained_file_count": 0,
            }
        )
    return result


def _range_by_id(plan: Mapping[str, Any], range_id: str) -> dict[str, Any]:
    wanted = _nonempty_string(range_id, "range_id")
    rows = [row for row in plan["header_ranges"] if row["range_id"] == wanted]
    if len(rows) != 1:
        raise DeepSeekV4StreamError(f"plan does not contain exactly one header range {wanted!r}")
    return dict(rows[0])


def _secure_capture_size(path: Path, *, expected_bytes: int) -> int:
    try:
        node = os.lstat(path)
    except OSError as exc:
        raise DeepSeekV4StreamError(f"cannot stat header capture {path}: {exc}") from exc
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        raise DeepSeekV4StreamError("header capture must be a regular, non-symlink file")
    if node.st_size != expected_bytes:
        raise DeepSeekV4StreamError(
            f"header capture has {node.st_size} bytes; exact bounded range requires {expected_bytes}"
        )
    return int(node.st_size)


def _read_bounded_header_capture(
    path: Path,
    *,
    expected_bytes: int,
    chunk_bytes: int,
    workspace_root: Path,
    protected_floor_bytes: int,
    max_inflight_bytes: int,
    free_bytes_provider: FreeBytesProvider | None,
    floor_checks: list[dict[str, Any]],
) -> bytes:
    if expected_bytes > MAX_HEADER_CAPTURE_BYTES + 8:
        raise DeepSeekV4StreamError("requested header capture exceeds the absolute bounded limit")
    if chunk_bytes <= 0 or chunk_bytes > max_inflight_bytes:
        raise DeepSeekV4StreamError("chunk_bytes must be positive and at most max_inflight_bytes")
    _secure_capture_size(path, expected_bytes=expected_bytes)
    capture = bytearray()
    try:
        with path.open("rb", buffering=0) as handle:
            while len(capture) < expected_bytes:
                floor_checks.append(
                    assert_floor(
                        workspace_root,
                        protected_floor_bytes=protected_floor_bytes,
                        additional_bytes=max_inflight_bytes,
                        stage="during_range_before_chunk",
                        free_bytes_provider=free_bytes_provider,
                    )
                )
                chunk = handle.read(min(chunk_bytes, expected_bytes - len(capture)))
                if not chunk:
                    raise DeepSeekV4StreamError("header capture truncated during bounded read")
                if len(chunk) > chunk_bytes:
                    raise DeepSeekV4StreamError("header capture reader exceeded its bounded chunk size")
                capture.extend(chunk)
                if len(capture) > expected_bytes:
                    raise DeepSeekV4StreamError("header capture accumulated beyond its exact declared range")
                floor_checks.append(
                    assert_floor(
                        workspace_root,
                        protected_floor_bytes=protected_floor_bytes,
                        additional_bytes=max_inflight_bytes,
                        stage="during_range_after_chunk",
                        free_bytes_provider=free_bytes_provider,
                    )
                )
            trailing = handle.read(1)
            if trailing:
                raise DeepSeekV4StreamError("header capture has trailing tensor-body bytes")
    except OSError as exc:
        raise DeepSeekV4StreamError(f"cannot read header capture {path}: {exc}") from exc
    return bytes(capture)


def _inspect_safetensors_header(capture: bytes) -> dict[str, Any]:
    if len(capture) < 8:
        raise DeepSeekV4StreamError("header capture is shorter than its safetensors length prefix")
    header_bytes = struct.unpack("<Q", capture[:8])[0]
    if header_bytes <= 0 or header_bytes > MAX_HEADER_CAPTURE_BYTES:
        raise DeepSeekV4StreamError("safetensors header length exceeds the bounded header-only limit")
    if len(capture) != 8 + header_bytes:
        raise DeepSeekV4StreamError(
            "header capture must contain exactly the 8-byte prefix and declared JSON header, never tensor bytes"
        )
    try:
        decoded = json.loads(capture[8:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeepSeekV4StreamError(f"invalid safetensors header JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise DeepSeekV4StreamError("safetensors header JSON must be an object")
    descriptors = 0
    for name, descriptor in decoded.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(descriptor, Mapping):
            raise DeepSeekV4StreamError("safetensors header contains an invalid tensor descriptor")
        descriptors += 1
    if descriptors == 0:
        raise DeepSeekV4StreamError("safetensors header contains no tensor descriptors")
    return {
        "header_length_bytes": header_bytes,
        "capture_bytes": len(capture),
        "tensor_descriptor_count": descriptors,
        "metadata_present": "__metadata__" in decoded,
        "capture_sha256": _sha256(capture),
        "tensor_body_bytes_present": 0,
    }


def _atomic_json_once(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically create one receipt without silently overwriting evidence."""
    path = _absolute_path(path, "receipt_path")
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise DeepSeekV4StreamError(f"cannot inspect receipt path {path}: {exc}") from exc
    if existing is not None:
        raise DeepSeekV4StreamError(f"receipt path already exists; historical receipt replacement is refused: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    if len(payload) > MAX_RECEIPT_BYTES:
        raise DeepSeekV4StreamError("bounded header receipt exceeds the receipt size limit")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # The destination was checked above, and an externally-created receipt
        # between the check and replace is still a safety violation, not a value
        # we may overwrite.
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise DeepSeekV4StreamError(
                f"receipt path appeared during atomic creation; replacement is refused: {path}"
            ) from exc
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def preflight_plan(
    plan: Mapping[str, Any],
    *,
    workspace_root: str | Path,
    free_bytes_provider: FreeBytesProvider | None = None,
) -> dict[str, Any]:
    """Run only storage preflight; it never reads a capture or starts transport."""
    plan = validate_plan(plan)
    root = _absolute_path(workspace_root, "workspace_root")
    policy = plan["storage_policy"]
    checks = []
    for row in plan["header_ranges"]:
        checks.append(
            assert_floor(
                root,
                protected_floor_bytes=policy["protected_floor_bytes"],
                additional_bytes=policy["max_inflight_bytes"],
                stage=f"plan_only_before_{row['range_id']}",
                free_bytes_provider=free_bytes_provider,
            )
        )
    return seal(
        {
            "schema": PLAN_PREFLIGHT_SCHEMA,
            "status": "PLAN_ONLY_NOT_EXECUTED",
            "created_at": _utc_now(),
            "plan_seal_sha256": plan["seal_sha256"],
            "floor_checks": checks,
            "source_retention": {
                "state": "not_inspected_in_plan_only_mode",
                "declared_paths": plan["storage_policy"]["source_retention_paths"],
            },
            "execution_boundary": _execution_boundary(),
        }
    )


def execute_header_only(
    plan: Mapping[str, Any],
    *,
    range_id: str,
    header_capture_path: str | Path,
    receipt_path: str | Path,
    workspace_root: str | Path,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    free_bytes_provider: FreeBytesProvider | None = None,
) -> dict[str, Any]:
    """Validate one exact local header capture and atomically seal its receipt.

    This consumes at most the declared safetensors header capture.  It accepts
    no tensor body and performs no network action.  A caller must declare at
    least one dedicated source/cache retention root which is empty before and
    after the capture; the executor never deletes a cache to manufacture that
    condition.
    """
    plan = validate_plan(plan)
    row = _range_by_id(plan, range_id)
    policy = plan["storage_policy"]
    root = _absolute_path(workspace_root, "workspace_root")
    capture_path = _absolute_path(header_capture_path, "header_capture_path")
    output_path = _absolute_path(receipt_path, "receipt_path")
    chunk_bytes = _positive_int(chunk_bytes, "chunk_bytes")
    retention_paths = [Path(raw) for raw in policy["source_retention_paths"]]
    if not retention_paths:
        raise DeepSeekV4StreamError(
            "header execution is refused without declared source_retention_paths"
        )
    for retention_path in retention_paths:
        if _path_is_within(capture_path, retention_path):
            raise DeepSeekV4StreamError(
                "header capture must not reside in an asserted-empty source retention path"
            )
        if _path_is_within(output_path, retention_path):
            raise DeepSeekV4StreamError(
                "receipt must not reside in an asserted-empty source retention path"
            )

    floor_checks: list[dict[str, Any]] = [
        assert_floor(
            root,
            protected_floor_bytes=policy["protected_floor_bytes"],
            additional_bytes=policy["max_inflight_bytes"],
            stage="before_range",
            free_bytes_provider=free_bytes_provider,
        )
    ]
    eviction_before = assert_source_evicted(retention_paths)
    capture = _read_bounded_header_capture(
        capture_path,
        expected_bytes=row["expected_bytes"],
        chunk_bytes=chunk_bytes,
        workspace_root=root,
        protected_floor_bytes=policy["protected_floor_bytes"],
        max_inflight_bytes=policy["max_inflight_bytes"],
        free_bytes_provider=free_bytes_provider,
        floor_checks=floor_checks,
    )
    summary = _inspect_safetensors_header(capture)
    expected_sha256 = row["expected_capture_sha256"]
    if expected_sha256 is not None and summary["capture_sha256"] != expected_sha256:
        raise DeepSeekV4StreamError("header capture SHA-256 differs from the plan binding")
    # No capture bytes are emitted to a new file; only a digest and summary
    # enter the receipt.  The supplied capture remains a bounded metadata input.
    del capture
    floor_checks.append(
        assert_floor(
            root,
            protected_floor_bytes=policy["protected_floor_bytes"],
            additional_bytes=0,
            stage="after_range",
            free_bytes_provider=free_bytes_provider,
        )
    )
    eviction_after = assert_source_evicted(retention_paths)

    unsigned = {
        "schema": HEADER_RECEIPT_SCHEMA,
        "status": "HEADER_CAPTURE_SEALED_NOT_ADMITTED",
        "created_at": _utc_now(),
        "plan_seal_sha256": plan["seal_sha256"],
        "source": {
            "repository": plan["source"]["repository"],
            "revision": plan["source"]["revision"],
            "source_authority": "not_provided",
        },
        "range": row,
        "header_capture": {
            **summary,
            "capture_path": str(capture_path),
            "capture_sha256_verified_against_plan": expected_sha256 is not None,
            "source_body_download_or_transport": "outside_executor_not_claimed",
        },
        "floor_checks": floor_checks,
        "source_eviction_assertion": {
            "status": "PASS",
            "scope": "declared dedicated source/cache retention paths only",
            "before_range": eviction_before,
            "after_range": eviction_after,
            "source_range_files_retained_zero": True,
            "source_body_evicted_or_never_persisted_in_declared_paths": True,
        },
        "execution_boundary": _execution_boundary(),
        "architecture_admission_handoff": {
            "state": "not_admitted",
            "reason": "A header capture is structural input only; source-exact fixture evidence remains required.",
        },
    }
    # Account for the temporary file plus final directory entry before making
    # the atomic receipt write.  The range itself was already sampled after
    # consumption; this additionally prevents receipt creation from being the
    # action that crosses the protected floor.
    floor_checks.append(
        assert_floor(
            root,
            protected_floor_bytes=policy["protected_floor_bytes"],
            additional_bytes=2 * MAX_RECEIPT_BYTES,
            stage="before_atomic_receipt",
            free_bytes_provider=free_bytes_provider,
        )
    )
    receipt = seal(unsigned)
    _atomic_json_once(output_path, receipt)
    # Enforce (rather than merely project) the post-write floor.  It is not
    # resealed into the receipt because that would require a second receipt
    # write and defeat a single atomic terminal record.
    assert_floor(
        root,
        protected_floor_bytes=policy["protected_floor_bytes"],
        additional_bytes=0,
        stage="after_atomic_receipt",
        free_bytes_provider=free_bytes_provider,
    )
    return receipt


def validate_header_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the narrow contract of a successful bounded header receipt."""
    try:
        receipt = verify(value, label="DeepSeek-V4 bounded header receipt")
    except SealIntegrityError as exc:
        raise DeepSeekV4StreamError(str(exc)) from exc
    if receipt.get("schema") != HEADER_RECEIPT_SCHEMA:
        raise DeepSeekV4StreamError("unexpected header receipt schema")
    if receipt.get("status") != "HEADER_CAPTURE_SEALED_NOT_ADMITTED":
        raise DeepSeekV4StreamError("unexpected header receipt status")
    if receipt.get("execution_boundary") != _execution_boundary():
        raise DeepSeekV4StreamError("header receipt overclaims execution capability")
    eviction = receipt.get("source_eviction_assertion")
    if not isinstance(eviction, Mapping) or eviction.get("status") != "PASS":
        raise DeepSeekV4StreamError("header receipt lacks a passing source eviction assertion")
    floor_checks = receipt.get("floor_checks")
    if not isinstance(floor_checks, list):
        raise DeepSeekV4StreamError("header receipt lacks floor checks")
    stages = {row.get("stage") for row in floor_checks if isinstance(row, Mapping)}
    required = {"before_range", "during_range_before_chunk", "during_range_after_chunk", "after_range"}
    if not required <= stages:
        raise DeepSeekV4StreamError("header receipt lacks before/during/after range floor evidence")
    return receipt


def _read_plan_file(path: str | Path) -> dict[str, Any]:
    source = _absolute_path(path, "plan")
    try:
        loaded = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeepSeekV4StreamError(f"cannot read plan {source}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise DeepSeekV4StreamError("plan root must be a JSON object")
    return loaded


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="seal a non-executing header-only range plan")
    plan.add_argument("--repository", default=EXPECTED_REPOSITORY)
    plan.add_argument("--revision", required=True)
    plan.add_argument("--range-id", required=True)
    plan.add_argument("--shard", required=True)
    plan.add_argument("--range-end", type=int, required=True, help="exclusive byte end; start is fixed at 0")
    plan.add_argument("--expected-capture-sha256")
    plan.add_argument("--retention-path", action="append", default=[])
    plan.add_argument("--floor-bytes", type=int, default=MIN_FREE_FLOOR_BYTES)
    plan.add_argument("--max-inflight-bytes", type=int, default=DEFAULT_MAX_INFLIGHT_BYTES)
    plan.add_argument("--out", required=True)
    header = commands.add_parser("header", help="seal one bounded local header capture; never fetches")
    header.add_argument("--plan", required=True)
    header.add_argument("--range-id", required=True)
    header.add_argument("--header-capture", required=True)
    header.add_argument("--receipt", required=True)
    header.add_argument("--workspace-root", default=str(Path.cwd()))
    header.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    preflight = commands.add_parser("preflight", help="check planned 15 GiB floor without reading a capture")
    preflight.add_argument("--plan", required=True)
    preflight.add_argument("--workspace-root", default=str(Path.cwd()))
    preflight.add_argument("--out")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    try:
        if args.command == "plan":
            plan = build_plan(
                repository=args.repository,
                revision=args.revision,
                header_ranges=[
                    {
                        "range_id": args.range_id,
                        "shard": args.shard,
                        "start": 0,
                        "end": args.range_end,
                        "expected_capture_sha256": args.expected_capture_sha256,
                    }
                ],
                source_retention_paths=args.retention_path,
                protected_floor_bytes=args.floor_bytes,
                max_inflight_bytes=args.max_inflight_bytes,
            )
            _atomic_json_once(_absolute_path(args.out, "out"), plan)
            print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
            return 0
        plan = _read_plan_file(args.plan)
        if args.command == "preflight":
            receipt = preflight_plan(plan, workspace_root=args.workspace_root)
            if args.out:
                _atomic_json_once(_absolute_path(args.out, "out"), receipt)
            print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
            return 0
        receipt = execute_header_only(
            plan,
            range_id=args.range_id,
            header_capture_path=args.header_capture,
            receipt_path=args.receipt,
            workspace_root=args.workspace_root,
            chunk_bytes=args.chunk_bytes,
        )
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return 0
    except DeepSeekV4StreamError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())

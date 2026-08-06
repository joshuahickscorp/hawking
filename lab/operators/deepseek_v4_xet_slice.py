"""Bounded, zero-cache Xet tensor-range streaming for DeepSeek-V4-Flash.

This is the physical hand-off immediately after a sealed safetensors header:
it admits a *small, named* set of layer-4 tensor ranges, verifies the pinned
Hub/Xet control plane, streams one range at a time into bounded memory, hashes
it, and releases it.  It never materializes a safetensors shard or persists a
raw tensor body.

The module intentionally stops before decoding or Condense packing.  A later
native-codec consumer must receive these bytes in-process and emit its own
receipt; a range receipt by itself is not evidence of a codec, artifact,
runtime, capability, or throughput result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from lab.receipts import SealIntegrityError, seal, verify
from lab.operators import deepseek_v4_stream_executor as header_stream


PLAN_SCHEMA = "hawking.gravity.deepseek_v4.xet_slice_plan.v1"
RECEIPT_SCHEMA = "hawking.condense.deepseek_v4.xet_slice_execution.v1"

EXPECTED_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash"
EXPECTED_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
FIXTURE_SHARD = "model-00006-of-00046.safetensors"
FIXTURE_LFS_SHA256 = "51a65e6d9d0ccb70013e25ae70a50b177af8f97e59ac798c2d0ed5ebb169fe7a"
FIXTURE_FULL_SIZE_BYTES = 3_590_024_776
PINNED_XET_PACKAGES = {"huggingface_hub": "1.24.0", "hf_xet": "1.5.2"}

MAX_TARGETS = 8
MAX_ONE_TENSOR_BYTES = 16 * 1024**2
MAX_TOTAL_TENSOR_BYTES = 16 * 1024**2
_SHA256_LENGTH = 64


class DeepSeekV4XetSliceError(RuntimeError):
    """A bounded V4 Xet-slice invariant failed closed."""


MetadataProvider = Callable[[], Mapping[str, Any]]
StreamFactory = Callable[[Mapping[str, Any], Mapping[str, Any]], Iterable[bytes]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hex(value: object, label: str, length: int = _SHA256_LENGTH) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise DeepSeekV4XetSliceError(f"{label} must be {length} hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise DeepSeekV4XetSliceError(f"{label} must be hexadecimal") from exc
    return value.lower()


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeepSeekV4XetSliceError(f"{label} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise DeepSeekV4XetSliceError(f"{label} must be {'non-negative' if allow_zero else 'positive'}")
    return value


def _absolute_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise DeepSeekV4XetSliceError(f"{label} must be an absolute path")
    return Path(os.path.abspath(os.fspath(path)))


def _regular_file(path: Path, label: str) -> None:
    try:
        node = os.lstat(path)
    except OSError as exc:
        raise DeepSeekV4XetSliceError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        raise DeepSeekV4XetSliceError(f"{label} must be a regular non-symlink file")


def _bounded_header(path: Path) -> tuple[dict[str, Mapping[str, Any]], int, str]:
    """Open exactly the previously sealed small header capture."""

    _regular_file(path, "header_capture")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DeepSeekV4XetSliceError(f"cannot read header_capture: {exc}") from exc
    if len(raw) < 8 or len(raw) > header_stream.MAX_HEADER_CAPTURE_BYTES + 8:
        raise DeepSeekV4XetSliceError("header_capture is outside the bounded header-only limit")
    declared = int.from_bytes(raw[:8], "little", signed=False)
    if declared <= 0 or declared + 8 != len(raw):
        raise DeepSeekV4XetSliceError("header_capture does not contain exactly prefix + JSON header")
    try:
        value = json.loads(raw[8:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeepSeekV4XetSliceError(f"header_capture is not valid safetensors JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DeepSeekV4XetSliceError("header_capture safetensors JSON must be an object")
    tensors = {name: row for name, row in value.items() if name != "__metadata__"}
    if not tensors or any(not isinstance(name, str) or not isinstance(row, Mapping) for name, row in tensors.items()):
        raise DeepSeekV4XetSliceError("header_capture has invalid tensor descriptors")
    return tensors, len(raw), _sha256(raw)


def _shape_elements(value: object, label: str) -> int:
    if not isinstance(value, list) or not value:
        raise DeepSeekV4XetSliceError(f"{label} must be a non-empty shape list")
    count = 1
    for index, item in enumerate(value):
        dimension = _positive_int(item, f"{label}[{index}]")
        count *= dimension
        if count > FIXTURE_FULL_SIZE_BYTES:
            raise DeepSeekV4XetSliceError(f"{label} exceeds bounded fixture size")
    return count


def _descriptor_target(name: str, row: Mapping[str, Any], *, header_bytes: int) -> dict[str, Any]:
    if not name.startswith("layers.4."):
        raise DeepSeekV4XetSliceError("only bounded layer-4 fixture tensors are permitted")
    dtype = row.get("dtype")
    element_bytes = {"BF16": 2, "F32": 4, "I8": 1, "F8_E4M3": 1, "F8_E8M0": 1}.get(dtype)
    if element_bytes is None:
        raise DeepSeekV4XetSliceError(f"{name}: unsupported bounded fixture dtype {dtype!r}")
    elements = _shape_elements(row.get("shape"), f"{name}.shape")
    offsets = row.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in offsets)
        or offsets[0] > offsets[1]
    ):
        raise DeepSeekV4XetSliceError(f"{name}: invalid data_offsets")
    length = offsets[1] - offsets[0]
    expected = elements * element_bytes
    if length != expected:
        raise DeepSeekV4XetSliceError(
            f"{name}: descriptor length {length} differs from dtype/shape byte length {expected}"
        )
    if length <= 0 or length > MAX_ONE_TENSOR_BYTES:
        raise DeepSeekV4XetSliceError(f"{name}: tensor range exceeds bounded per-tensor limit")
    start, end = header_bytes + offsets[0], header_bytes + offsets[1]
    if end > FIXTURE_FULL_SIZE_BYTES:
        raise DeepSeekV4XetSliceError(f"{name}: tensor range exceeds pinned shard size")
    return {
        "name": name,
        "dtype": dtype,
        "shape": list(row["shape"]),
        "data_offsets": list(offsets),
        "start": start,
        "end": end,
        "length": length,
    }


def _execution_boundary() -> dict[str, Any]:
    return {
        "mode": "bounded_source_range_only",
        "source_body_bytes_persisted_by_executor": 0,
        "source_body_persisted": False,
        "native_fp4_decode": "not_executed",
        "native_fp8_decode": "not_executed",
        "router_execution": "not_executed",
        "condense_packing": "not_executed",
        "gravity_artifact": "not_created",
        "cpu_oracle": "not_executed",
        "metal_forward": "not_executed",
        "capability": "unknown",
        "throughput": "unknown",
    }


def build_plan(
    *,
    header_receipt: Mapping[str, Any],
    header_capture_path: str | Path,
    tensor_names: Sequence[str],
    source_retention_path: str | Path,
    protected_floor_bytes: int = header_stream.MIN_FREE_FLOOR_BYTES,
) -> dict[str, Any]:
    """Seal a first small body-range plan from a previously sealed header."""

    receipt = header_stream.validate_header_receipt(header_receipt)
    source = receipt.get("source")
    range_row = receipt.get("range")
    if not isinstance(source, Mapping) or not isinstance(range_row, Mapping):
        raise DeepSeekV4XetSliceError("header receipt lacks source/range binding")
    if source.get("repository") != EXPECTED_REPOSITORY or source.get("revision") != EXPECTED_REVISION:
        raise DeepSeekV4XetSliceError("header receipt is not bound to the pinned DeepSeek-V4-Flash source")
    if range_row.get("shard") != FIXTURE_SHARD:
        raise DeepSeekV4XetSliceError("header receipt does not cover the selected layer-4 fixture shard")
    header_path = _absolute_path(header_capture_path, "header_capture_path")
    tensors, header_bytes, header_sha256 = _bounded_header(header_path)
    captured = receipt.get("header_capture")
    if not isinstance(captured, Mapping) or captured.get("capture_sha256") != header_sha256:
        raise DeepSeekV4XetSliceError("header_capture digest does not match the sealed header receipt")
    if captured.get("capture_bytes") != header_bytes or range_row.get("expected_bytes") != header_bytes:
        raise DeepSeekV4XetSliceError("header_capture length does not match the sealed header receipt")
    if isinstance(tensor_names, (str, bytes)) or not tensor_names or len(tensor_names) > MAX_TARGETS:
        raise DeepSeekV4XetSliceError(f"tensor_names must contain 1..{MAX_TARGETS} names")
    names = [name for name in tensor_names if isinstance(name, str) and name]
    if len(names) != len(tensor_names) or len(names) != len(set(names)):
        raise DeepSeekV4XetSliceError("tensor_names must be distinct non-empty strings")
    targets = [_descriptor_target(name, tensors.get(name, {}), header_bytes=header_bytes) for name in names]
    total = sum(target["length"] for target in targets)
    if total > MAX_TOTAL_TENSOR_BYTES:
        raise DeepSeekV4XetSliceError("requested tensor ranges exceed the bounded total payload limit")
    floor = _positive_int(protected_floor_bytes, "protected_floor_bytes")
    if floor < header_stream.MIN_FREE_FLOOR_BYTES:
        raise DeepSeekV4XetSliceError("protected_floor_bytes cannot be below the non-negotiable 15 GiB")
    retention = _absolute_path(source_retention_path, "source_retention_path")
    return seal(
        {
            "schema": PLAN_SCHEMA,
            "status": "REQUESTED_NOT_EXECUTED",
            "created_at": _utc_now(),
            "source": {
                "repository": EXPECTED_REPOSITORY,
                "revision": EXPECTED_REVISION,
                "shard": FIXTURE_SHARD,
                "expected_lfs_sha256": FIXTURE_LFS_SHA256,
                "expected_file_bytes": FIXTURE_FULL_SIZE_BYTES,
                "header_receipt_seal_sha256": receipt["seal_sha256"],
                "header_capture_sha256": header_sha256,
                "header_capture_bytes": header_bytes,
            },
            "targets": targets,
            "storage_policy": {
                "protected_floor_bytes": floor,
                "max_one_tensor_bytes": MAX_ONE_TENSOR_BYTES,
                "max_total_tensor_bytes": MAX_TOTAL_TENSOR_BYTES,
                "source_retention_paths": [str(retention)],
                "no_raw_body_persistence": True,
                "xet_chunk_cache_size_bytes": 0,
            },
            "transport": {
                "kind": "hf_xet_direct_range_stream",
                "high_performance": True,
                "outer_concurrent_streams": 1,
                "end_semantics": "exclusive",
                "pinned_packages": PINNED_XET_PACKAGES,
            },
            "execution_boundary": _execution_boundary(),
        }
    )


def validate_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        plan = verify(value, label="DeepSeek-V4 Xet slice plan")
    except SealIntegrityError as exc:
        raise DeepSeekV4XetSliceError(str(exc)) from exc
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "REQUESTED_NOT_EXECUTED":
        raise DeepSeekV4XetSliceError("not a requested DeepSeek-V4 Xet slice plan")
    source = plan.get("source")
    if not isinstance(source, Mapping) or source != {
        "repository": EXPECTED_REPOSITORY,
        "revision": EXPECTED_REVISION,
        "shard": FIXTURE_SHARD,
        "expected_lfs_sha256": FIXTURE_LFS_SHA256,
        "expected_file_bytes": FIXTURE_FULL_SIZE_BYTES,
        "header_receipt_seal_sha256": source.get("header_receipt_seal_sha256") if isinstance(source, Mapping) else None,
        "header_capture_sha256": source.get("header_capture_sha256") if isinstance(source, Mapping) else None,
        "header_capture_bytes": source.get("header_capture_bytes") if isinstance(source, Mapping) else None,
    }:
        raise DeepSeekV4XetSliceError("plan source binding is not canonical")
    _hex(source["header_receipt_seal_sha256"], "header_receipt_seal_sha256")
    _hex(source["header_capture_sha256"], "header_capture_sha256")
    _positive_int(source["header_capture_bytes"], "header_capture_bytes")
    targets = plan.get("targets")
    if not isinstance(targets, list) or not 1 <= len(targets) <= MAX_TARGETS:
        raise DeepSeekV4XetSliceError("plan targets are outside bounded count")
    seen: set[str] = set()
    total = 0
    for target in targets:
        if not isinstance(target, Mapping):
            raise DeepSeekV4XetSliceError("plan target must be an object")
        name = target.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise DeepSeekV4XetSliceError("plan target names must be distinct non-empty strings")
        seen.add(name)
        canonical = _descriptor_target(name, target, header_bytes=source["header_capture_bytes"])
        if canonical != target:
            raise DeepSeekV4XetSliceError("plan target is not canonical to its descriptor")
        total += canonical["length"]
    if total > MAX_TOTAL_TENSOR_BYTES:
        raise DeepSeekV4XetSliceError("plan targets exceed total bounded payload")
    policy = plan.get("storage_policy")
    if not isinstance(policy, Mapping):
        raise DeepSeekV4XetSliceError("plan lacks storage_policy")
    if _positive_int(policy.get("protected_floor_bytes"), "protected_floor_bytes") < header_stream.MIN_FREE_FLOOR_BYTES:
        raise DeepSeekV4XetSliceError("plan lowers the non-negotiable 15 GiB floor")
    if policy.get("max_one_tensor_bytes") != MAX_ONE_TENSOR_BYTES or policy.get("max_total_tensor_bytes") != MAX_TOTAL_TENSOR_BYTES:
        raise DeepSeekV4XetSliceError("plan changes the bounded range ceilings")
    roots = policy.get("source_retention_paths")
    if not isinstance(roots, list) or len(roots) != 1 or str(_absolute_path(roots[0], "source_retention_path")) != roots[0]:
        raise DeepSeekV4XetSliceError("plan source retention path is invalid")
    if policy.get("no_raw_body_persistence") is not True or policy.get("xet_chunk_cache_size_bytes") != 0:
        raise DeepSeekV4XetSliceError("plan permits raw-body persistence or Xet chunk caching")
    if plan.get("transport") != {
        "kind": "hf_xet_direct_range_stream",
        "high_performance": True,
        "outer_concurrent_streams": 1,
        "end_semantics": "exclusive",
        "pinned_packages": PINNED_XET_PACKAGES,
    }:
        raise DeepSeekV4XetSliceError("plan transport binding is invalid")
    if plan.get("execution_boundary") != _execution_boundary():
        raise DeepSeekV4XetSliceError("plan execution boundary is invalid")
    return plan


def _configure_xet_environment(retention_root: Path) -> dict[str, str]:
    """Set all transfer cache knobs before importing Hub/Xet modules."""

    values = {
        "HF_HOME": str(retention_root),
        "HF_HUB_CACHE": str(retention_root / "hub-cache"),
        "HF_XET_CACHE": str(retention_root / "xet-cache"),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
        "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY": "false",
        "HF_XET_RECONSTRUCTION_USE_VECTORED_WRITE": "true",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_XET_LOG_DEST": "stderr",
        "HF_XET_LOG_FORMAT": "json",
    }
    os.environ.update(values)
    return values


def _live_metadata_and_stream_factory() -> tuple[Mapping[str, Any], StreamFactory, dict[str, Any]]:
    """Open one public read session only after environment hardening."""

    versions: dict[str, str] = {}
    for package, expected in PINNED_XET_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise DeepSeekV4XetSliceError(f"required package is unavailable: {package}") from exc
        if actual != expected:
            raise DeepSeekV4XetSliceError(f"{package} version drift: expected {expected}, got {actual}")
        versions[package] = actual
    try:
        from hf_xet import XetFileInfo, XetSession
        from huggingface_hub import hf_hub_url
        from huggingface_hub.file_download import get_hf_file_metadata
        from huggingface_hub.utils import build_hf_headers
        from huggingface_hub.utils._xet import XetTokenType, xet_connection_info_refresh_url, xet_headers_without_auth
    except ImportError as exc:
        raise DeepSeekV4XetSliceError("pinned hf_xet direct-stream APIs are unavailable") from exc
    metadata = get_hf_file_metadata(
        hf_hub_url(EXPECTED_REPOSITORY, FIXTURE_SHARD, revision=EXPECTED_REVISION), token=False
    )
    xet_data = getattr(metadata, "xet_file_data", None)
    if xet_data is None or not isinstance(getattr(xet_data, "file_hash", None), str):
        raise DeepSeekV4XetSliceError("pinned source metadata has no Xet file identity")
    remote = {
        "commit_hash": getattr(metadata, "commit_hash", None),
        "etag": getattr(metadata, "etag", None),
        "size": getattr(metadata, "size", None),
        "xet_hash": xet_data.file_hash,
    }
    headers = build_hf_headers(token=False, library_name="hawking-v4-slice", library_version="1")
    if any(key.lower() == "authorization" for key in headers):
        raise DeepSeekV4XetSliceError("public Xet stream unexpectedly contains an authorization header")
    refresh_url = xet_connection_info_refresh_url(
        token_type=XetTokenType.READ,
        repo_id=EXPECTED_REPOSITORY,
        repo_type="model",
        revision=EXPECTED_REVISION,
    )
    session = XetSession()
    group = session.new_download_stream_group(
        token_refresh_url=refresh_url,
        token_refresh_headers=headers,
        custom_headers=xet_headers_without_auth(headers),
    )

    def stream(target: Mapping[str, Any], checked_remote: Mapping[str, Any]) -> Iterable[bytes]:
        return group.download_stream(
            XetFileInfo(checked_remote["xet_hash"], checked_remote["size"]),
            start=target["start"],
            end=target["end"],
        )

    return remote, stream, {
        "packages": versions,
        "public_read_token_refresh": True,
        "authorization_header_present": False,
        "xet_file_hash": remote["xet_hash"],
    }


def _validate_remote_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("commit_hash") != EXPECTED_REVISION:
        raise DeepSeekV4XetSliceError("Hub metadata commit hash differs from the pinned source revision")
    if value.get("etag") != FIXTURE_LFS_SHA256:
        raise DeepSeekV4XetSliceError("Hub metadata ETag differs from the pinned full-shard LFS hash")
    if value.get("size") != FIXTURE_FULL_SIZE_BYTES:
        raise DeepSeekV4XetSliceError("Hub metadata size differs from the pinned shard size")
    return {
        "commit_hash": EXPECTED_REVISION,
        "etag": FIXTURE_LFS_SHA256,
        "size": FIXTURE_FULL_SIZE_BYTES,
        "xet_hash": _hex(value.get("xet_hash"), "metadata.xet_hash"),
    }


def _stream_in_memory(
    chunks: Iterable[bytes],
    *,
    target: Mapping[str, Any],
    workspace_root: Path,
    floor: int,
    free_bytes_provider: header_stream.FreeBytesProvider | None,
    floor_checks: list[dict[str, Any]],
) -> tuple[bytes, dict[str, Any]]:
    started = time.monotonic_ns()
    raw = bytearray()
    digest = hashlib.sha256()
    try:
        for chunk in chunks:
            if not isinstance(chunk, bytes) or not chunk:
                raise DeepSeekV4XetSliceError("Xet stream yielded a non-bytes or empty chunk")
            if len(raw) + len(chunk) > target["length"]:
                raise DeepSeekV4XetSliceError("Xet stream exceeded the exact sealed target length")
            floor_checks.append(
                header_stream.assert_floor(
                    workspace_root,
                    protected_floor_bytes=floor,
                    additional_bytes=target["length"],
                    stage=f"during_range_before_chunk:{target['name']}",
                    free_bytes_provider=free_bytes_provider,
                )
            )
            raw.extend(chunk)
            digest.update(chunk)
            floor_checks.append(
                header_stream.assert_floor(
                    workspace_root,
                    protected_floor_bytes=floor,
                    additional_bytes=target["length"],
                    stage=f"during_range_after_chunk:{target['name']}",
                    free_bytes_provider=free_bytes_provider,
                )
            )
    except BaseException:
        cancel = getattr(chunks, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass
        raise
    if len(raw) != target["length"]:
        raise DeepSeekV4XetSliceError(
            f"Xet stream length mismatch for {target['name']}: {len(raw)} != {target['length']}"
        )
    return bytes(raw), {
        "name": target["name"],
        "dtype": target["dtype"],
        "start": target["start"],
        "end": target["end"],
        "bytes": len(raw),
        "sha256": digest.hexdigest(),
        "elapsed_seconds": (time.monotonic_ns() - started) / 1_000_000_000,
    }


def execute_plan(
    plan: Mapping[str, Any],
    *,
    workspace_root: str | Path,
    metadata_provider: MetadataProvider | None = None,
    stream_factory: StreamFactory | None = None,
    free_bytes_provider: header_stream.FreeBytesProvider | None = None,
) -> dict[str, Any]:
    """Stream and hash each sealed range in memory; never decode or persist it."""

    plan = validate_plan(plan)
    root = _absolute_path(workspace_root, "workspace_root")
    policy = plan["storage_policy"]
    floor = policy["protected_floor_bytes"]
    retention = Path(policy["source_retention_paths"][0])
    floor_checks: list[dict[str, Any]] = [
        header_stream.assert_floor(
            root,
            protected_floor_bytes=floor,
            additional_bytes=max(target["length"] for target in plan["targets"]),
            stage="before_transport",
            free_bytes_provider=free_bytes_provider,
        )
    ]
    eviction_before = header_stream.assert_source_evicted([retention])
    environment = _configure_xet_environment(retention)
    runtime: dict[str, Any]
    if metadata_provider is None or stream_factory is None:
        if metadata_provider is not None or stream_factory is not None:
            raise DeepSeekV4XetSliceError("metadata_provider and stream_factory must be supplied together for tests")
        live_metadata, live_stream_factory, runtime = _live_metadata_and_stream_factory()
        metadata_provider = lambda: live_metadata
        stream_factory = live_stream_factory
    else:
        runtime = {"packages": "test-injected", "public_read_token_refresh": "not_claimed"}
    remote = _validate_remote_metadata(metadata_provider())
    ranges: list[dict[str, Any]] = []
    for target in plan["targets"]:
        floor_checks.append(
            header_stream.assert_floor(
                root,
                protected_floor_bytes=floor,
                additional_bytes=target["length"],
                stage=f"before_range:{target['name']}",
                free_bytes_provider=free_bytes_provider,
            )
        )
        raw, result = _stream_in_memory(
            stream_factory(target, remote),
            target=target,
            workspace_root=root,
            floor=floor,
            free_bytes_provider=free_bytes_provider,
            floor_checks=floor_checks,
        )
        # The narrow transport layer deliberately has no decoder/packer
        # callback.  Drop the only raw copy before any receipt is assembled.
        del raw
        floor_checks.append(
            header_stream.assert_floor(
                root,
                protected_floor_bytes=floor,
                additional_bytes=0,
                stage=f"after_range:{target['name']}",
                free_bytes_provider=free_bytes_provider,
            )
        )
        ranges.append(result)
    eviction_after = header_stream.assert_source_evicted([retention])
    receipt = seal(
        {
            "schema": RECEIPT_SCHEMA,
            "status": "RANGE_BYTES_SEALED_NOT_DECODED",
            "created_at": _utc_now(),
            "plan_seal_sha256": plan["seal_sha256"],
            "source": {
                **plan["source"],
                "verified_hub_metadata": remote,
                "sparse_ranges_do_not_rehash_full_shard": True,
            },
            "transport": {
                **plan["transport"],
                "runtime": runtime,
                "environment": environment,
            },
            "range_results": ranges,
            "payload_bytes": sum(row["bytes"] for row in ranges),
            "floor_checks": floor_checks,
            "source_eviction_assertion": {
                "status": "PASS",
                "scope": "declared dedicated HF_HOME/Xet cache root only",
                "before_transport": eviction_before,
                "after_transport": eviction_after,
                "source_range_files_retained_zero": True,
                "raw_tensor_body_persisted_by_executor": False,
            },
            "execution_boundary": _execution_boundary(),
        }
    )
    return receipt


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    source = _absolute_path(path, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeepSeekV4XetSliceError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise DeepSeekV4XetSliceError(f"{label} root must be an object")
    return value


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="seal an exact bounded V4 tensor-range plan")
    plan.add_argument("--header-receipt", required=True)
    plan.add_argument("--header-capture", required=True)
    plan.add_argument("--tensor", action="append", required=True)
    plan.add_argument("--source-retention-path", required=True)
    plan.add_argument("--floor-bytes", type=int, default=header_stream.MIN_FREE_FLOOR_BYTES)
    plan.add_argument("--out", required=True)
    run = commands.add_parser("stream", help="directly stream/hash planned V4 slices; never decode")
    run.add_argument("--plan", required=True)
    run.add_argument("--workspace-root", default=str(Path.cwd()))
    run.add_argument("--receipt", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    try:
        if args.command == "plan":
            plan = build_plan(
                header_receipt=_read_json(args.header_receipt, "header_receipt"),
                header_capture_path=args.header_capture,
                tensor_names=args.tensor,
                source_retention_path=args.source_retention_path,
                protected_floor_bytes=args.floor_bytes,
            )
            header_stream._atomic_json_once(_absolute_path(args.out, "out"), plan)
            print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
            return 0
        receipt = execute_plan(
            _read_json(args.plan, "plan"), workspace_root=args.workspace_root
        )
        header_stream._atomic_json_once(_absolute_path(args.receipt, "receipt"), receipt)
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return 0
    except DeepSeekV4XetSliceError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())

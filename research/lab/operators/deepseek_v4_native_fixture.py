#!/usr/bin/env python3.12
"""Run one bounded DeepSeek-V4-Flash native FP4/FP8 codec fixture over Xet.

The fixture is deliberately small and physical:

* one complete row (2,048 packed bytes plus 128 scale bytes) of the Layer 4
  routed FP4 expert tensor;
* one complete row (4,096 bytes plus 32 scale bytes) of the Layer 4 shared
  E4M3FN FP8 tensor.

That is exactly 6,304 source-body bytes.  Each range is read directly from an
``hf_xet`` ordered stream into memory, decoded immediately, summarized, and
discarded.  The harness never writes tensor body bytes.  It is a codec
mechanics/transport fixture, not source authority, a Condense artifact, a
forward, a capability result, or a throughput result.

Run it with the isolated current Hugging Face/Xet environment, for example::

    tools/condense/.venv/bin/python tools/condense/deepseek_v4_native_fixture.py run \\
      --header-capture /tmp/hawking-v4-probe.d4aXVU/model-00006.xet.header \\
      --workspace-root /Users/scammermike/Downloads/hawking \\
      --xet-root /tmp/hawking-v4-native-xet \\
      --receipt /tmp/hawking-v4-native-fixture.json

The Xet root is a declared retention boundary.  It may contain directory-only
session scaffolding, but any retained file, symlink, or special node fails the
run before a receipt can claim body eviction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lab.operators import deepseek_v4_stream_executor as stream_gate
from lab.receipts import SealIntegrityError, seal, verify
from tools.condense import deepseek_v4_native_codec as codec


FIXTURE_SCHEMA = "hawking.gravity.deepseek_v4.native_codec_xet_fixture.v1"
FIXTURE_STATUS = "BYTE_FIXTURE_DECODED_NOT_SOURCE_EXACT"
SHARD = "model-00006-of-00046.safetensors"
SHARD_SIZE_BYTES = 3_590_024_776
SHARD_ETAG_SHA256 = "51a65e6d9d0ccb70013e25ae70a50b177af8f97e59ac798c2d0ed5ebb169fe7a"
XET_FILE_HASH = "5f7c9f0ea087e246292ca37d5b532f33edd3421709b3f44df2ecaad8b17d603b"
HEADER_SHA256 = "6a227e0a48fb4a9bdc8ad8dd1842d587f949902972bad48ca8ff59f4a6584cc3"

# The maximum source-body buffer visible to this harness.  ``hf_xet`` does not
# expose its internal network/reconstruction allocation or wire-byte counters,
# so those are deliberately reported as not measured rather than folded into a
# misleading memory/transport claim.
MAX_INFLIGHT_BYTES = 1024 * 1024

_EXPECTED_DESCRIPTORS: dict[str, dict[str, Any]] = {
    "fp4_weight": {
        "name": "layers.4.ffn.experts.0.w1.weight",
        "dtype": "I8",
        "shape": [2048, 2048],
        "data_offsets": [368_625_752, 372_820_056],
    },
    "fp4_scale": {
        "name": "layers.4.ffn.experts.0.w1.scale",
        "dtype": "F8_E8M0",
        "shape": [2048, 128],
        "data_offsets": [26_788_440, 27_050_584],
    },
    "fp8_weight": {
        "name": "layers.4.ffn.shared_experts.w1.weight",
        "dtype": "F8_E4M3",
        "shape": [2048, 4096],
        "data_offsets": [343_459_928, 351_848_536],
    },
    "fp8_scale": {
        "name": "layers.4.ffn.shared_experts.w1.scale",
        "dtype": "F8_E8M0",
        "shape": [16, 32],
        "data_offsets": [228_115_032, 228_115_544],
    },
}


class NativeFixtureError(RuntimeError):
    """The fixture cannot establish its bounded physical/provenance contract."""


def _absolute(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        raise NativeFixtureError(f"{label} must be an absolute path")
    return Path(os.path.abspath(os.fspath(value)))


def _regular_non_symlink(path: Path, label: str) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise NativeFixtureError(f"cannot inspect {label} {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise NativeFixtureError(f"{label} must be a regular non-symlink file: {path}")


def _ensure_directory(path: Path, label: str) -> None:
    if path.exists():
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise NativeFixtureError(f"{label} must be a directory and not a symlink: {path}")
        return
    try:
        path.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise NativeFixtureError(f"cannot create {label} {path}: {exc}") from exc


def _read_header_capture(path: Path) -> tuple[bytes, dict[str, dict[str, Any]]]:
    _regular_non_symlink(path, "header capture")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise NativeFixtureError(f"cannot read header capture {path}: {exc}") from exc
    actual_hash = codec.sha256_hex(raw)
    if actual_hash != HEADER_SHA256:
        raise NativeFixtureError(
            f"header capture SHA-256 differs from the bounded Layer 4 header: {actual_hash}"
        )
    try:
        header = codec.parse_header_only(raw)
    except codec.DeepSeekV4NativeCodecError as exc:
        raise NativeFixtureError(str(exc)) from exc
    return raw, header


def _descriptors_from_header(header: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, expected in _EXPECTED_DESCRIPTORS.items():
        try:
            descriptor = codec.descriptor_from_header(header, expected["name"])
        except codec.DeepSeekV4NativeCodecError as exc:
            raise NativeFixtureError(str(exc)) from exc
        observed = {
            "name": descriptor["name"],
            "dtype": descriptor["dtype"],
            "shape": list(descriptor["shape"]),
            "data_offsets": list(descriptor["data_offsets"]),
        }
        if observed != expected:
            raise NativeFixtureError(
                f"header descriptor differs from selected Layer 4 fixture for {expected['name']!r}: "
                f"{observed!r}"
            )
        result[key] = descriptor
    return result


def fixture_ranges(header_bytes: int, descriptors: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, int | str]]:
    """Build the four exact one-row source ranges and recheck the 6,304B cap."""
    try:
        result = {
            key: codec.expected_source_range(descriptor, header_bytes=header_bytes, row_count=1)
            for key, descriptor in descriptors.items()
        }
    except codec.DeepSeekV4NativeCodecError as exc:
        raise NativeFixtureError(str(exc)) from exc
    intervals = sorted((int(row["file_start"]), int(row["file_stop"]), key) for key, row in result.items())
    for (_, previous_stop, previous_key), (next_start, _, next_key) in zip(intervals, intervals[1:]):
        if previous_stop > next_start:
            raise NativeFixtureError(f"fixture ranges overlap: {previous_key} / {next_key}")
    total = sum(int(row["byte_count"]) for row in result.values())
    if total != 6_304:
        raise NativeFixtureError(f"fixture body total must be 6,304 bytes, got {total}")
    if any(int(row["byte_count"]) > MAX_INFLIGHT_BYTES for row in result.values()):
        raise NativeFixtureError("a fixture range exceeds the bounded in-memory transport budget")
    return result


def _configure_xet_environment(root: Path) -> None:
    """Set zero-persistent-cache Xet configuration before its modules are imported."""
    _ensure_directory(root, "Xet retention root")
    os.environ["HF_HOME"] = str(root / "hf-home")
    os.environ["HF_XET_CACHE"] = str(root / "xet-cache")
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    os.environ["HF_XET_LOG_DEST"] = "stderr"
    os.environ["HF_XET_LOG_FORMAT"] = "json"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _load_xet_transport() -> dict[str, Any]:
    """Import only after the zero-cache environment is configured."""
    # A caller can deliberately make only NumPy visible through PYTHONPATH
    # while using the isolated Xet venv.  Put the active venv back ahead of
    # PYTHONPATH so an older globally installed hf_xet cannot silently replace
    # the range-stream API this fixture requires.
    active_site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if active_site.is_dir():
        active_site_text = str(active_site)
        if active_site_text in sys.path:
            sys.path.remove(active_site_text)
        sys.path.insert(0, active_site_text)
    try:
        from hf_xet import XetFileInfo, XetSession
        from huggingface_hub import hf_hub_url
        from huggingface_hub.file_download import get_hf_file_metadata
        from huggingface_hub.utils import build_hf_headers
        from huggingface_hub.utils._xet import (
            XetTokenType,
            xet_connection_info_refresh_url,
            xet_headers_without_auth,
        )
    except ImportError as exc:
        raise NativeFixtureError(
            "hf_xet/huggingface_hub with range streaming is required; run via "
            "tools/condense/.venv/bin/python after installing "
            "tools/condense/requirements-deepseek-v4.txt"
        ) from exc
    return {
        "XetFileInfo": XetFileInfo,
        "XetSession": XetSession,
        "hf_hub_url": hf_hub_url,
        "get_hf_file_metadata": get_hf_file_metadata,
        "build_hf_headers": build_hf_headers,
        "XetTokenType": XetTokenType,
        "xet_connection_info_refresh_url": xet_connection_info_refresh_url,
        "xet_headers_without_auth": xet_headers_without_auth,
    }


def _verified_xet_group(transport: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Resolve source metadata and reject any identity drift before body streaming."""
    url = transport["hf_hub_url"](codec.OFFICIAL_REPOSITORY, SHARD, revision=codec.OFFICIAL_REVISION)
    metadata = transport["get_hf_file_metadata"](url, token=False)
    xet_data = getattr(metadata, "xet_file_data", None)
    observed = {
        "repository": codec.OFFICIAL_REPOSITORY,
        "revision": codec.OFFICIAL_REVISION,
        "commit_hash": getattr(metadata, "commit_hash", None),
        "file_size_bytes": getattr(metadata, "size", None),
        "etag_sha256": getattr(metadata, "etag", None),
        "xet_file_hash": getattr(xet_data, "file_hash", None),
        "xet_refresh_route": getattr(xet_data, "refresh_route", None),
    }
    if observed["commit_hash"] != codec.OFFICIAL_REVISION:
        raise NativeFixtureError(f"Xet metadata commit differs from immutable revision: {observed['commit_hash']!r}")
    if observed["file_size_bytes"] != SHARD_SIZE_BYTES:
        raise NativeFixtureError("Xet metadata file size differs from selected shard")
    if observed["etag_sha256"] != SHARD_ETAG_SHA256:
        raise NativeFixtureError("Xet metadata ETag differs from selected shard SHA-256")
    if observed["xet_file_hash"] != XET_FILE_HASH or not observed["xet_refresh_route"]:
        raise NativeFixtureError("Xet metadata does not bind the selected source object")
    headers = transport["build_hf_headers"](token=False, library_name="hawking-native-fixture")
    refresh_url = transport["xet_connection_info_refresh_url"](
        token_type=transport["XetTokenType"].READ,
        repo_id=codec.OFFICIAL_REPOSITORY,
        repo_type="model",
        revision=codec.OFFICIAL_REVISION,
    )
    group = transport["XetSession"]().new_download_stream_group(
        token_refresh_url=refresh_url,
        token_refresh_headers=headers,
        custom_headers=transport["xet_headers_without_auth"](headers),
    )
    file_info = transport["XetFileInfo"](observed["xet_file_hash"], observed["file_size_bytes"])
    return group, file_info, observed


def _stream_exact_range(
    group: Any,
    file_info: Any,
    spec: Mapping[str, int | str],
    *,
    workspace_root: Path,
    protected_floor_bytes: int,
    floor_checks: list[dict[str, Any]],
) -> tuple[bytes, dict[str, Any]]:
    name = str(spec["tensor"])
    start, stop = int(spec["file_start"]), int(spec["file_stop"])
    expected = int(spec["byte_count"])
    floor_checks.append(
        stream_gate.assert_floor(
            workspace_root,
            protected_floor_bytes=protected_floor_bytes,
            additional_bytes=MAX_INFLIGHT_BYTES,
            stage=f"before_range:{name}",
        )
    )
    stream = group.download_stream(file_info, start=start, end=stop)
    body = bytearray()
    try:
        for chunk in stream:
            floor_checks.append(
                stream_gate.assert_floor(
                    workspace_root,
                    protected_floor_bytes=protected_floor_bytes,
                    additional_bytes=MAX_INFLIGHT_BYTES,
                    stage=f"during_range_before_chunk:{name}",
                )
            )
            if not isinstance(chunk, bytes) or not chunk:
                raise NativeFixtureError(f"Xet yielded an invalid empty/non-byte chunk for {name}")
            if len(chunk) > MAX_INFLIGHT_BYTES or len(body) + len(chunk) > expected:
                raise NativeFixtureError(f"Xet range exceeded exact bounded body size for {name}")
            body.extend(chunk)
            floor_checks.append(
                stream_gate.assert_floor(
                    workspace_root,
                    protected_floor_bytes=protected_floor_bytes,
                    additional_bytes=MAX_INFLIGHT_BYTES,
                    stage=f"during_range_after_chunk:{name}",
                )
            )
    finally:
        cancel = getattr(stream, "cancel", None)
        if callable(cancel):
            cancel()
    raw = bytes(body)
    if len(raw) != expected:
        raise NativeFixtureError(f"Xet returned {len(raw)} bytes for {name}, expected exactly {expected}")
    floor_checks.append(
        stream_gate.assert_floor(
            workspace_root,
            protected_floor_bytes=protected_floor_bytes,
            additional_bytes=MAX_INFLIGHT_BYTES,
            stage=f"after_range:{name}",
        )
    )
    return raw, {
        "tensor": name,
        "file_start": start,
        "file_stop": stop,
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "transport": "hf_xet_ordered_in_memory_range",
    }


def _decoded_summary(name: str, values: np.ndarray) -> dict[str, Any]:
    if values.dtype != np.float32 or values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise NativeFixtureError(f"{name} decode did not produce a finite non-empty float32 matrix")
    return {
        "tensor": name,
        "shape": list(values.shape),
        "dtype": "float32",
        "little_endian_f32_sha256": hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest(),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _write_new_sealed_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise NativeFixtureError(f"refusing to overwrite existing receipt: {path}")
    _ensure_directory(path.parent, "receipt parent")
    _regular_parent(path.parent)
    encoded = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise NativeFixtureError(f"cannot create receipt {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # The incomplete receipt is a known exact target created by this process.
        # Do not delete it silently; a caller can inspect it, and a retry must use
        # a new receipt path rather than making an overwrite ambiguous.
        raise


def _regular_parent(path: Path) -> None:
    """Refuse a symlink in the final receipt directory, not arbitrary ancestors."""
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise NativeFixtureError(f"cannot inspect receipt parent {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise NativeFixtureError(f"receipt parent must be a non-symlink directory: {path}")


def run_fixture(
    *,
    header_capture: str | Path,
    workspace_root: str | Path,
    xet_root: str | Path,
    receipt_path: str | Path,
    protected_floor_bytes: int = stream_gate.MIN_FREE_FLOOR_BYTES,
) -> dict[str, Any]:
    """Execute the four bounded in-memory Xet ranges and seal a non-claiming receipt."""
    header_path = _absolute(header_capture, "header_capture")
    workspace = _absolute(workspace_root, "workspace_root")
    retention_root = _absolute(xet_root, "xet_root")
    receipt = _absolute(receipt_path, "receipt_path")
    if protected_floor_bytes < stream_gate.MIN_FREE_FLOOR_BYTES:
        raise NativeFixtureError("protected_floor_bytes cannot be below the non-negotiable 15 GiB floor")
    if _path_within(receipt, retention_root):
        raise NativeFixtureError("receipt_path must be outside xet_root retention boundary")
    _ensure_directory(retention_root, "Xet retention root")
    _regular_parent(workspace)
    # No retained body/cache input is accepted.  This precondition also limits
    # the receipt's storage claim to this declared Xet root rather than the
    # entire user's unrelated Hugging Face cache.
    before_retention = stream_gate.assert_source_evicted([retention_root])
    floor_checks: list[dict[str, Any]] = [
        stream_gate.assert_floor(
            workspace,
            protected_floor_bytes=protected_floor_bytes,
            additional_bytes=MAX_INFLIGHT_BYTES,
            stage="before_header_and_metadata",
        )
    ]
    header_raw, header = _read_header_capture(header_path)
    descriptors = _descriptors_from_header(header)
    ranges = fixture_ranges(len(header_raw), descriptors)
    _configure_xet_environment(retention_root)
    transport = _load_xet_transport()
    group, file_info, source_identity = _verified_xet_group(transport)

    raw, fp4_weight_range = _stream_exact_range(
        group, file_info, ranges["fp4_weight"], workspace_root=workspace,
        protected_floor_bytes=protected_floor_bytes, floor_checks=floor_checks,
    )
    scale_raw, fp4_scale_range = _stream_exact_range(
        group, file_info, ranges["fp4_scale"], workspace_root=workspace,
        protected_floor_bytes=protected_floor_bytes, floor_checks=floor_checks,
    )
    try:
        fp4 = codec.decode_fp4_e2m1fn_x2_rows(
            raw, descriptors["fp4_weight"], scale_raw, descriptors["fp4_scale"], row_count=1
        )
    except codec.DeepSeekV4NativeCodecError as exc:
        raise NativeFixtureError(str(exc)) from exc
    finally:
        # Source range bytes have no receiver after decoder construction.
        del raw, scale_raw
    fp4_summary = _decoded_summary(descriptors["fp4_weight"]["name"], fp4)
    del fp4

    raw, fp8_weight_range = _stream_exact_range(
        group, file_info, ranges["fp8_weight"], workspace_root=workspace,
        protected_floor_bytes=protected_floor_bytes, floor_checks=floor_checks,
    )
    scale_raw, fp8_scale_range = _stream_exact_range(
        group, file_info, ranges["fp8_scale"], workspace_root=workspace,
        protected_floor_bytes=protected_floor_bytes, floor_checks=floor_checks,
    )
    try:
        fp8 = codec.decode_fp8_e4m3fn_rows(
            raw,
            descriptors["fp8_weight"],
            scale_raw,
            descriptors["fp8_scale"],
            row_count=1,
            scale_block_row_start=0,
        )
    except codec.DeepSeekV4NativeCodecError as exc:
        raise NativeFixtureError(str(exc)) from exc
    finally:
        del raw, scale_raw
    fp8_summary = _decoded_summary(descriptors["fp8_weight"]["name"], fp8)
    del fp8

    floor_checks.append(
        stream_gate.assert_floor(
            workspace,
            protected_floor_bytes=protected_floor_bytes,
            additional_bytes=MAX_INFLIGHT_BYTES,
            stage="after_decode_before_retention_assertion",
        )
    )
    after_retention = stream_gate.assert_source_evicted([retention_root])
    codec_status = codec.bounded_fixture_status(
        repository=codec.OFFICIAL_REPOSITORY,
        revision=codec.OFFICIAL_REVISION,
        header_capture_sha256=codec.sha256_hex(header_raw),
        source_authority_status=None,
    )
    document = seal(
        {
            "schema": FIXTURE_SCHEMA,
            "status": FIXTURE_STATUS,
            "source": source_identity,
            "header_capture": {
                "path": str(header_path),
                "bytes": len(header_raw),
                "sha256": codec.sha256_hex(header_raw),
                "body_bytes_present": 0,
            },
            "fixture": {
                "shard": SHARD,
                "range_count": 4,
                "source_body_bytes": 6_304,
                "harness_visible_max_inflight_bytes": MAX_INFLIGHT_BYTES,
                "source_body_persisted_by_harness": False,
                "range_transport": "hf_xet_ordered_in_memory_range",
                "xet_internal_network_bytes": "not_exposed_by_hf_xet_stream_api",
                "xet_internal_memory_bytes": "not_measured",
                "ranges": {
                    "fp4_weight": fp4_weight_range,
                    "fp4_scale": fp4_scale_range,
                    "fp8_weight": fp8_weight_range,
                    "fp8_scale": fp8_scale_range,
                },
                "decoded": {"fp4": fp4_summary, "fp8": fp8_summary},
            },
            "storage": {
                "protected_floor_bytes": protected_floor_bytes,
                "floor_checks": floor_checks,
                "declared_xet_retention_root": str(retention_root),
                "before_retention_assertion": before_retention,
                "after_retention_assertion": after_retention,
            },
            "codec_status": codec_status,
            "source_authority": "not_provided",
            "not_evidence_of": [
                "complete_shard_download",
                "source_exact_authority",
                "condense_artifact",
                "cpu_forward",
                "metal_forward",
                "capability",
                "throughput",
            ],
        }
    )
    try:
        verify(document, label="DeepSeek-V4 native codec fixture receipt")
    except SealIntegrityError as exc:
        raise NativeFixtureError(f"fixture receipt seal did not verify: {exc}") from exc
    _write_new_sealed_receipt(receipt, document)
    return document


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="execute the fixed 6,304-byte Xet native codec fixture")
    run.add_argument("--header-capture", required=True)
    run.add_argument("--workspace-root", required=True)
    run.add_argument("--xet-root", required=True)
    run.add_argument("--receipt", required=True)
    run.add_argument(
        "--protected-floor-bytes",
        type=int,
        default=stream_gate.MIN_FREE_FLOOR_BYTES,
        help="must be at least 15 GiB; default is exactly 15 GiB",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        receipt = run_fixture(
            header_capture=args.header_capture,
            workspace_root=args.workspace_root,
            xet_root=args.xet_root,
            receipt_path=args.receipt,
            protected_floor_bytes=args.protected_floor_bytes,
        )
    except (NativeFixtureError, stream_gate.DeepSeekV4StreamError, OSError, ValueError) as exc:
        print(json.dumps({"schema": FIXTURE_SCHEMA, "status": "NOT_EXECUTED", "reason": str(exc)}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

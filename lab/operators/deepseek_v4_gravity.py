#!/usr/bin/env python3.12
"""Bounded streamed Condense execution for a DeepSeek-V4 Gravity diagnostic.

This is deliberately a single internal Condense engine, not another public
model framework.  It constructs the smallest *full-width, layer-limited*
DeepSeek-V4 diagnostic window that can later be driven by the companion
runtime adapter:

    embed -> HC expansion -> complete layer 4 -> HC head -> logits

The window is intentionally not a full 43-layer DeepSeek-V4 forward.  Layer 4
normally receives the HC state emitted by layers 0--3, so the result is always
labelled ``diagnostic``.  It does, however, preserve every tensor needed by
that layer, including all 256 score-routed experts, so a later generated token
cannot silently fall back when its route changes.

The physical source contract is stricter than an ordinary downloader:

* direct ``hf_xet`` range streams only, with the pinned public revision;
* source ranges are bounded to 8 MiB and are never persisted as a parent
  safetensors file or cache;
* every range becomes an atomically written content-addressed Gravity chunk;
* the layer-limited diagnostic keeps one source range live in Python at a
  time; the full-model path uses a bounded shard-worker pool while retaining
  one range at a time per worker;
* exact chunk restart records are fsync'd after every sealed range;
* all three dependency-complete shards are re-hashed against their Hub LFS
  identities before the artifact manifest is sealed;
* each native FP4/FP8 tensor is decoded from a source-real row as a mechanics
  and repeatability check before sealing.

The resulting directory is a ``.gravity`` diagnostic artifact understood by
the V4 Python runtime adapter in this module.  It is intentionally separate
from the older Rust ``deepseek2`` adapter, which has a different architecture
and must not be relabelled as V4.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import importlib.metadata
import json
import math
import os
import platform
import shutil
import stat
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, OrderedDict
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, local
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from lab.receipts import SealIntegrityError, seal, verify
from tools.condense import deepseek_v4_native_codec as codec


REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash"
REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
MODEL_ID = f"{REPOSITORY}@{REVISION}"
ARTIFACT_SCHEMA = "hawking.gravity.deepseek_v4.diagnostic.v1"
FULL_ARTIFACT_SCHEMA = "hawking.gravity.deepseek_v4.full_stream.v1"
STATIC_EXPERT_RESIDENCY_SCHEMA = "hawking.gravity.deepseek_v4.static_expert_residency.v1"
JOURNAL_SCHEMA = "hawking.gravity.deepseek_v4.stream_journal.v1"
RANGE_SCHEMA = "hawking.gravity.deepseek_v4.stream_range.v1"
SOURCE_RECEIPT_SCHEMA = "hawking.gravity.deepseek_v4.source_window.v1"
MIN_FREE_FLOOR_BYTES = 15 * 1024**3
DEFAULT_RANGE_BYTES = 8 * 1024**2
MAX_RANGE_BYTES = 16 * 1024**2
MAX_HEADER_BYTES = 8 * 1024**2
HTTP_COALESCED_WINDOW_BYTES = MAX_RANGE_BYTES
# Hard ceiling for the canonical full-model transfer profile.  This is high
# enough to expose the local 10GbE path while still bounding per-worker
# buffers, file descriptors, and retry pressure against the Hub CDN.
MAX_PARALLEL_WORKERS = 32
# The live-suite packer only reads receipts produced by a bounded local HCLI
# exercise.  Keep its input window deliberately small so an arbitrary JSON
# file cannot turn a receipt-sealing step into an unbounded archival action.
MAX_HCLI_LIVE_EVIDENCE_FILES = 32
MAX_HCLI_LIVE_EVIDENCE_BYTES = 16 * 1024**2
MAX_HCLI_LIVE_ERROR_UTF8_BYTES = 4 * 1024
# The profile is intentionally bounded to the executable diagnostic window.
# These limits are not an attempted substitute for the 8K/512-token Base TPS
# benchmark, which this one-layer CPU adapter cannot satisfy.
MAX_DIAGNOSTIC_PROFILE_TRIALS = 4
MAX_DIAGNOSTIC_PROFILE_DECODE_FORWARDS = 4
MAX_DIAGNOSTIC_PROFILE_PROMPT_UTF8_BYTES = 64 * 1024
# Static residency analysis only ever reads sealed metadata and bounded
# receipts.  It must never become another runtime path or a way to archive
# unbounded diagnostic traces.
MAX_STATIC_RESIDENCY_RECEIPT_BYTES = 16 * 1024**2
STATIC_RESIDENCY_CONTEXTS = (2048, 8192, 32768)
PINNED_XET_PACKAGES = {"huggingface_hub": "1.24.0", "hf_xet": "1.5.2"}

# These are intentionally the *complete* three-shard dependency window.  The
# Hub index shows every byte in each named shard belongs to this diagnostic.
DIAGNOSTIC_SHARDS = (
    "model-00001-of-00046.safetensors",
    "model-00006-of-00046.safetensors",
    "model-00045-of-00046.safetensors",
)
FULL_SHARDS = tuple(f"model-{index:05d}-of-00046.safetensors" for index in range(1, 47))
# The pinned public index has 69,187 named tensors.  An older admission
# receipt counted 72,317 local entries before aliases were collapsed; the
# source index's named ``weight_map`` is the authoritative set for streaming.
FULL_EXPECTED_TENSOR_COUNT = 69187
METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "inference/config.json",
    "inference/model.py",
    "inference/convert.py",
    "inference/kernel.py",
)


class DeepSeekV4GravityError(RuntimeError):
    """The V4 streamed Condense execution cannot safely continue."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise DeepSeekV4GravityError(f"{label} must be an absolute path")
    return Path(os.path.abspath(os.fspath(path)))


def _positive(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeepSeekV4GravityError(f"{label} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise DeepSeekV4GravityError(
            f"{label} must be {'non-negative' if allow_zero else 'positive'}"
        )
    return value


def _regular_file(path: Path, label: str) -> None:
    try:
        node = os.lstat(path)
    except OSError as exc:
        raise DeepSeekV4GravityError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        raise DeepSeekV4GravityError(f"{label} must be a regular non-symlink file")


def _ensure_dir(path: Path, label: str) -> None:
    if path.exists():
        node = os.lstat(path)
        if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
            raise DeepSeekV4GravityError(f"{label} must be a non-symlink directory")
        return
    path.mkdir(parents=True, exist_ok=False)


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _floor_check(workspace: Path, floor: int, additional: int, stage: str) -> dict[str, Any]:
    free = _free_bytes(workspace)
    if free - additional < floor:
        raise DeepSeekV4GravityError(
            f"15 GiB storage floor would be crossed at {stage}: "
            f"free={free}, additional={additional}, floor={floor}"
        )
    return {
        "stage": stage,
        "free_bytes": free,
        "additional_bytes": additional,
        "protected_floor_bytes": floor,
        "status": "PASS",
    }


def _atomic_bytes(path: Path, raw: bytes, *, expected_sha256: str | None = None) -> str:
    """Write a result exactly once, or verify an already sealed identical result."""

    digest = _sha256(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        raise DeepSeekV4GravityError(f"output digest mismatch for {path.name}")
    if path.exists():
        _regular_file(path, str(path))
        observed = _sha256(path.read_bytes())
        if observed != digest:
            raise DeepSeekV4GravityError(f"refusing to overwrite different existing output {path}")
        return digest
    _ensure_dir(path.parent, f"parent for {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory = None
        if directory is not None:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    return _atomic_bytes(path, _canonical(value) + b"\n")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeepSeekV4GravityError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise DeepSeekV4GravityError(f"{label} must contain a JSON object")
    return value


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    """Durably append one restart record without rewriting the whole journal."""

    _ensure_dir(path.parent, f"parent for {path}")
    encoded = _canonical(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(descriptor, "ab", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    _regular_file(path, "stream range journal")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DeepSeekV4GravityError(f"invalid range journal line {number}: {exc}") from exc
        if not isinstance(row, dict):
            raise DeepSeekV4GravityError(f"range journal line {number} is not an object")
        rows.append(row)
    return rows


def _range_id(shard: str, name: str, start: int, end: int) -> str:
    return f"{shard}:{name}:{start}:{end}"


class XetTransport:
    """Pinned public Xet range transport with a dedicated zero-cache root.

    The normal diagnostic path uses the pinned ``hf_xet`` stream API.  The
    full-model path may use the same signed Xet bridge through exact HTTP
    ``Range`` requests when the unauthenticated Python session is throttled;
    both paths bind the same Hub revision, Xet object hash, and LFS SHA.
    """

    def __init__(self, retention_root: Path, *, direct_http: bool = False) -> None:
        self.retention_root = retention_root
        self.direct_http = direct_http
        self._http_local = local()
        _ensure_dir(retention_root, "Xet retention root")
        before = list(retention_root.iterdir())
        if before:
            raise DeepSeekV4GravityError(
                "declared Xet retention root must be empty before a new run; "
                "use the sealed artifact restart journal instead of retaining a transport cache"
            )
        environment = {
            "HF_HOME": str(retention_root / "hf-home"),
            "HF_HUB_CACHE": str(retention_root / "hub-cache"),
            "HF_XET_CACHE": str(retention_root / "xet-cache"),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
            # Keep the legacy spelling explicit for receipts and configure the
            # equivalent supported control as vectored SSD writes.  These
            # values must be visible *before* either Hub or hf_xet is imported.
            "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY": "false",
            "HF_XET_RECONSTRUCTION_USE_VECTORED_WRITE": "true",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_XET_LOG_DEST": "stderr",
            "HF_XET_LOG_FORMAT": "json",
        }
        os.environ.update(environment)
        configured_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        try:
            from huggingface_hub import get_token as _get_stored_token

            stored_token = _get_stored_token()
        except ImportError:
            stored_token = None
        versions: dict[str, str] = {}
        for package, expected in PINNED_XET_PACKAGES.items():
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError as exc:
                raise DeepSeekV4GravityError(f"missing required Xet package {package}") from exc
            if actual != expected:
                raise DeepSeekV4GravityError(
                    f"{package} version drift: expected {expected}, observed {actual}"
                )
            versions[package] = actual
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
            raise DeepSeekV4GravityError("pinned hf_xet range APIs are unavailable") from exc
        self._XetFileInfo = XetFileInfo
        self._XetSession = XetSession
        self._hf_hub_url = hf_hub_url
        self._get_metadata = get_hf_file_metadata
        token = configured_token or stored_token
        if token is not None and not isinstance(token, str):
            token = None
        self.authenticated = bool(token)
        self._token = token
        self._headers = build_hf_headers(
            token=token or False, library_name="hawking-v4-gravity", library_version="1"
        )
        if not self.authenticated and any(key.lower() == "authorization" for key in self._headers):
            raise DeepSeekV4GravityError("anonymous Xet transport unexpectedly includes authorization")
        self._XetTokenType = XetTokenType
        self._refresh_url = xet_connection_info_refresh_url
        self._no_auth_headers = xet_headers_without_auth
        self.environment = environment
        self.versions = versions

    def metadata(self, shard: str) -> dict[str, Any]:
        metadata = self._get_metadata(
            self._hf_hub_url(REPOSITORY, shard, revision=REVISION), token=self._token or False
        )
        xet = getattr(metadata, "xet_file_data", None)
        result = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "shard": shard,
            "commit_hash": getattr(metadata, "commit_hash", None),
            "etag_sha256": getattr(metadata, "etag", None),
            "file_size_bytes": getattr(metadata, "size", None),
            "xet_file_hash": getattr(xet, "file_hash", None),
            "xet_refresh_route": getattr(xet, "refresh_route", None),
            # ``location`` is the short-lived signed CDN URL returned by the
            # Hub metadata call.  Keeping it in-memory per shard avoids
            # repeating a Hub redirect/authentication round-trip for every
            # range request.
            "signed_location": getattr(metadata, "location", None),
        }
        if result["commit_hash"] != REVISION:
            raise DeepSeekV4GravityError(f"{shard}: Hub commit did not match pinned revision")
        if not isinstance(result["etag_sha256"], str) or len(result["etag_sha256"]) != 64:
            raise DeepSeekV4GravityError(f"{shard}: Hub metadata has no LFS SHA-256 identity")
        if not isinstance(result["file_size_bytes"], int) or result["file_size_bytes"] <= 8:
            raise DeepSeekV4GravityError(f"{shard}: Hub metadata has invalid file size")
        if not isinstance(result["xet_file_hash"], str) or len(result["xet_file_hash"]) != 64:
            raise DeepSeekV4GravityError(f"{shard}: Hub metadata has no Xet object identity")
        if not result["xet_refresh_route"]:
            raise DeepSeekV4GravityError(f"{shard}: Hub metadata has no public Xet refresh route")
        if not isinstance(result["signed_location"], str) or not result["signed_location"].startswith("https://"):
            raise DeepSeekV4GravityError(f"{shard}: Hub metadata has no signed CDN location")
        return result

    def group(self) -> Any:
        refresh = self._refresh_url(
            token_type=self._XetTokenType.READ,
            repo_id=REPOSITORY,
            repo_type="model",
            revision=REVISION,
        )
        return self._XetSession().new_download_stream_group(
            token_refresh_url=refresh,
            token_refresh_headers=self._headers,
            custom_headers=self._no_auth_headers(self._headers),
        )

    def read_range(
        self,
        group: Any,
        metadata: Mapping[str, Any],
        start: int,
        end: int,
        *,
        workspace: Path,
        floor: int,
        stage: str,
    ) -> tuple[bytes, list[dict[str, Any]]]:
        start = _positive(start, "range start", allow_zero=True)
        end = _positive(end, "range end")
        expected = end - start
        if end <= start or expected > MAX_RANGE_BYTES:
            raise DeepSeekV4GravityError(f"{stage}: source range is outside 1..{MAX_RANGE_BYTES} bytes")
        if end > int(metadata["file_size_bytes"]):
            raise DeepSeekV4GravityError(f"{stage}: source range escapes the sealed source object")
        if self.direct_http:
            return self._read_range_http(
                metadata,
                start,
                end,
                workspace=workspace,
                floor=floor,
                stage=stage,
            )
        checks = [_floor_check(workspace, floor, expected, f"{stage}:before")]
        file_info = self._XetFileInfo(metadata["xet_file_hash"], metadata["file_size_bytes"])
        stream = group.download_stream(file_info, start=start, end=end)
        raw = bytearray()
        try:
            for chunk in stream:
                if not isinstance(chunk, bytes) or not chunk:
                    raise DeepSeekV4GravityError(f"{stage}: Xet yielded an invalid chunk")
                if len(raw) + len(chunk) > expected:
                    raise DeepSeekV4GravityError(f"{stage}: Xet exceeded exact range length")
                checks.append(_floor_check(workspace, floor, expected, f"{stage}:during-before"))
                raw.extend(chunk)
                checks.append(_floor_check(workspace, floor, expected, f"{stage}:during-after"))
        finally:
            cancel = getattr(stream, "cancel", None)
            if callable(cancel):
                cancel()
        if len(raw) != expected:
            raise DeepSeekV4GravityError(
                f"{stage}: Xet returned {len(raw)} bytes, expected exactly {expected}"
            )
        checks.append(_floor_check(workspace, floor, 0, f"{stage}:after"))
        return bytes(raw), checks

    def _read_range_http(
        self,
        metadata: Mapping[str, Any],
        start: int,
        end: int,
        *,
        workspace: Path,
        floor: int,
        stage: str,
    ) -> tuple[bytes, list[dict[str, Any]]]:
        """Read one exact range from the signed public Xet bridge.

        Hugging Face resolves the pinned revision to a signed ``xet-bridge``
        URL.  Requiring a 206 response and an exact ``Content-Range`` keeps a
        proxy or redirect from silently returning the parent file.
        """

        expected = end - start
        checks = [_floor_check(workspace, floor, expected, f"{stage}:before")]
        file_size = int(metadata["file_size_bytes"])
        cached = getattr(self._http_local, "window", None)
        if (
            cached is not None
            and cached["shard"] == metadata["shard"]
            and int(cached["start"]) <= start
            and end <= int(cached["end"])
        ):
            offset = start - int(cached["start"])
            raw = cached["raw"][offset : offset + expected]
            checks.append(_floor_check(workspace, floor, 0, f"{stage}:cached"))
            return bytes(raw), checks
        fetch_start, fetch_end = start, end
        coalesced = False
        if expected <= HTTP_COALESCED_WINDOW_BYTES:
            candidate_start = (start // HTTP_COALESCED_WINDOW_BYTES) * HTTP_COALESCED_WINDOW_BYTES
            candidate_end = min(file_size, candidate_start + HTTP_COALESCED_WINDOW_BYTES)
            if candidate_start <= start and end <= candidate_end:
                fetch_start, fetch_end = candidate_start, candidate_end
                coalesced = fetch_start != start or fetch_end != end
        fetch_expected = fetch_end - fetch_start
        checks = [_floor_check(workspace, floor, fetch_expected, f"{stage}:before")]
        url = str(metadata["signed_location"])
        request_headers = {
            "Range": f"bytes={fetch_start}-{fetch_end - 1}",
            "User-Agent": "hawking-v4-gravity-xet-range/1",
        }
        if self._token:
            request_headers["Authorization"] = f"Bearer {self._token}"
        raw: bytearray | None = None
        retryable = {429, 500, 502, 503, 504}
        for attempt in range(8):
            try:
                status, response_headers, response_body = self._http_get(
                    url, request_headers, timeout=180
                )
                if status in retryable:
                    if attempt == 7:
                        raise DeepSeekV4GravityError(
                            f"{stage}: signed Xet bridge range failed after {attempt + 1} attempts: HTTP {status}"
                        )
                    retry_after = response_headers.get("retry-after")
                    try:
                        delay = float(retry_after) if retry_after is not None else min(8.0, 0.25 * 2.0**attempt)
                    except ValueError:
                        delay = min(8.0, 0.25 * 2.0**attempt)
                    delay = max(0.25, min(8.0, delay))
                    checks.append(_floor_check(workspace, floor, fetch_expected, f"{stage}:retry-{status}"))
                    time.sleep(delay)
                    continue
                content_range = response_headers.get("content-range", "")
                expected_range = f"bytes {fetch_start}-{fetch_end - 1}/{metadata['file_size_bytes']}"
                if status != 206 or content_range != expected_range:
                    raise DeepSeekV4GravityError(
                        f"{stage}: signed Xet bridge did not honor exact range "
                        f"(status={status}, content-range={content_range!r}, expected={expected_range!r})"
                    )
                raw = bytearray()
                cursor = 0
                while cursor < len(response_body):
                    chunk = response_body[cursor : cursor + 1024 * 1024]
                    cursor += len(chunk)
                    checks.append(_floor_check(workspace, floor, fetch_expected, f"{stage}:during-before"))
                    raw.extend(chunk)
                    if len(raw) > fetch_expected:
                        raise DeepSeekV4GravityError(f"{stage}: HTTP Xet range exceeded exact length")
                    checks.append(_floor_check(workspace, floor, fetch_expected, f"{stage}:during-after"))
                break
            except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
                raise DeepSeekV4GravityError(f"{stage}: signed Xet bridge range failed: {exc}") from exc
        if raw is None:
            raise DeepSeekV4GravityError(f"{stage}: signed Xet bridge returned no bytes")
        if len(raw) != fetch_expected:
            raise DeepSeekV4GravityError(f"{stage}: HTTP Xet returned {len(raw)} bytes, expected exactly {fetch_expected}")
        checks.append(_floor_check(workspace, floor, 0, f"{stage}:after"))
        if coalesced:
            self._http_local.window = {
                "shard": metadata["shard"],
                "start": fetch_start,
                "end": fetch_end,
                "raw": bytes(raw),
            }
            offset = start - fetch_start
            return bytes(raw[offset : offset + expected]), checks
        return bytes(raw), checks

    def _http_get(
        self, url: str, headers: Mapping[str, str], *, timeout: float
    ) -> tuple[int, dict[str, str], bytes]:
        """GET a signed CDN URL over a thread-local keep-alive connection."""

        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise DeepSeekV4GravityError("signed Xet location must be an HTTPS URL")
        connections = getattr(self._http_local, "connections", None)
        if connections is None:
            connections = {}
            self._http_local.connections = connections
        key = parsed.netloc
        connection = connections.get(key)
        if connection is None:
            connection = http.client.HTTPSConnection(parsed.netloc, timeout=timeout)
            connections[key] = connection
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        try:
            connection.request("GET", target, headers=dict(headers))
            response = connection.getresponse()
            status = int(response.status)
            response_headers = {key.lower(): value for key, value in response.getheaders()}
            body = response.read()
            if response.will_close:
                connection.close()
                connections.pop(key, None)
            return status, response_headers, body
        except Exception:
            connection.close()
            connections.pop(key, None)
            raise

    def assert_no_files(self) -> list[str]:
        retained = [str(path) for path in self.retention_root.rglob("*") if path.is_file()]
        if retained:
            raise DeepSeekV4GravityError(
                "zero-cache Xet retention assertion failed: " + ", ".join(retained[:4])
            )
        return retained


def _direct_metadata_file(path: str) -> tuple[bytes, dict[str, Any]]:
    """Fetch a small source-owned text asset without using a Hub cache."""

    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": "hawking-v4-gravity/1"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read()
            headers = {key.lower(): value for key, value in response.headers.items()}
            final_host = urllib.parse.urlparse(response.geturl()).netloc
            status = getattr(response, "status", 200)
    except (urllib.error.URLError, OSError) as exc:
        raise DeepSeekV4GravityError(f"cannot fetch source metadata {path}: {exc}") from exc
    if not raw:
        raise DeepSeekV4GravityError(f"source metadata {path} was empty")
    return raw, {
        "path": path,
        "url": url,
        "final_host": final_host,
        "http_status": status,
        "bytes": len(raw),
        "sha256": _sha256(raw),
        "commit_header": headers.get("x-repo-commit"),
        "etag": headers.get("etag"),
    }


def _header_from_xet(
    transport: XetTransport,
    group: Any,
    metadata: Mapping[str, Any],
    *,
    workspace: Path,
    floor: int,
) -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    first, checks = transport.read_range(
        group, metadata, 0, 8, workspace=workspace, floor=floor, stage=f"{metadata['shard']}:header-prefix"
    )
    length = struct.unpack("<Q", first)[0]
    if length <= 0 or length > MAX_HEADER_BYTES:
        raise DeepSeekV4GravityError(f"{metadata['shard']}: invalid safetensors header length {length}")
    raw, more = transport.read_range(
        group,
        metadata,
        0,
        8 + length,
        workspace=workspace,
        floor=floor,
        stage=f"{metadata['shard']}:header",
    )
    checks.extend(more)
    try:
        value = json.loads(raw[8:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeepSeekV4GravityError(f"{metadata['shard']}: invalid safetensors header JSON") from exc
    if not isinstance(value, dict):
        raise DeepSeekV4GravityError(f"{metadata['shard']}: safetensors header root is not an object")
    return raw, value, checks


def _normalise_descriptors(header: Mapping[str, Any], *, header_bytes: int, file_size: int) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for name, row in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(row, Mapping):
            raise DeepSeekV4GravityError("safetensors header has an invalid tensor descriptor")
        offsets = row.get("data_offsets")
        shape = row.get("shape")
        dtype = row.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in offsets)
            or offsets[0] >= offsets[1]
            or not isinstance(shape, list)
            or not shape
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
            or not isinstance(dtype, str)
        ):
            raise DeepSeekV4GravityError(f"invalid safetensors descriptor for {name}")
        values.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": list(offsets),
                "file_start": header_bytes + offsets[0],
                "file_end": header_bytes + offsets[1],
                "bytes": offsets[1] - offsets[0],
            }
        )
    values.sort(key=lambda row: (int(row["file_start"]), str(row["name"])))
    if not values:
        raise DeepSeekV4GravityError("safetensors header has no tensors")
    cursor = header_bytes
    for row in values:
        if row["file_start"] != cursor:
            raise DeepSeekV4GravityError(
                f"source tensor layout has a gap or overlap before {row['name']}; cannot exact-stream full shard"
            )
        cursor = int(row["file_end"])
    if cursor != file_size:
        raise DeepSeekV4GravityError(
            f"source tensor layout ends at {cursor}, not metadata file size {file_size}"
        )
    return values


def _segment_rows(descriptor: Mapping[str, Any], maximum: int) -> Iterator[tuple[int, int, int, int]]:
    """Yield body-relative chunks, aligned to whole tensor rows when possible."""

    total = int(descriptor["bytes"])
    shape = descriptor["shape"]
    if not isinstance(shape, list) or not shape:
        raise DeepSeekV4GravityError("descriptor shape is invalid")
    rows = int(shape[0]) if len(shape) >= 2 else 1
    row_bytes = total // rows
    if row_bytes <= 0 or row_bytes * rows != total:
        raise DeepSeekV4GravityError(f"{descriptor['name']}: tensor cannot be row-segmented")
    rows_per_range = max(1, maximum // row_bytes)
    for first_row in range(0, rows, rows_per_range):
        count = min(rows_per_range, rows - first_row)
        start = first_row * row_bytes
        end = start + count * row_bytes
        yield start, end, first_row, count


class SegmentedStore:
    """Read immutable logical tensors from content-addressed artifact chunks."""

    def __init__(self, artifact: Path, tensors: Mapping[str, Mapping[str, Any]]) -> None:
        self.artifact = artifact
        self.tensors = tensors

    def descriptor(self, name: str) -> Mapping[str, Any]:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise DeepSeekV4GravityError(f"artifact lacks required tensor {name}") from exc

    def read(self, name: str, start: int = 0, end: int | None = None) -> bytes:
        tensor = self.descriptor(name)
        total = int(tensor["bytes"])
        start = _positive(start, f"{name} read start", allow_zero=True)
        end = total if end is None else _positive(end, f"{name} read end")
        if end <= start or end > total:
            raise DeepSeekV4GravityError(f"artifact read escapes tensor {name}")
        chunks: list[bytes] = []
        cursor = start
        for segment in tensor["segments"]:
            seg_start = int(segment["tensor_start"])
            seg_end = int(segment["tensor_end"])
            if seg_end <= cursor or seg_start >= end:
                continue
            path = self.artifact / segment["chunk_relpath"]
            _regular_file(path, f"artifact chunk for {name}")
            raw = path.read_bytes()
            if len(raw) != int(segment["bytes"]) or _sha256(raw) != segment["sha256"]:
                raise DeepSeekV4GravityError(f"artifact chunk integrity failure for {name}")
            take_start = max(cursor, seg_start)
            take_end = min(end, seg_end)
            chunks.append(raw[take_start - seg_start : take_end - seg_start])
            cursor = take_end
            if cursor == end:
                break
        if cursor != end:
            raise DeepSeekV4GravityError(f"artifact has a segment gap while reading {name}")
        return b"".join(chunks)

    def rows(self, name: str, row_start: int, row_count: int) -> bytes:
        tensor = self.descriptor(name)
        shape = tensor["shape"]
        if not isinstance(shape, list) or not shape:
            raise DeepSeekV4GravityError(f"{name}: invalid tensor shape")
        rows = int(shape[0]) if len(shape) >= 2 else 1
        total = int(tensor["bytes"])
        row_bytes = total // rows
        if row_start < 0 or row_count <= 0 or row_start + row_count > rows:
            raise DeepSeekV4GravityError(f"{name}: requested rows escape artifact tensor")
        return self.read(name, row_start * row_bytes, (row_start + row_count) * row_bytes)


def _bf16_to_f32(raw: bytes) -> np.ndarray:
    if len(raw) % 2:
        raise DeepSeekV4GravityError("BF16 payload has an odd byte count")
    bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return bits.view("<f4")


def _native_validation(
    artifact: Path, tensors: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Decode a real complete source row from every native tensor pair.

    The diagnostic preserves native payloads for lazy execution, but this
    validation is still a physical decode step: it refuses malformed E2M1,
    E4M3, or E8M0 bytes before the manifest can be sealed.
    """

    store = SegmentedStore(artifact, tensors)
    results: list[dict[str, Any]] = []
    kinds: Counter[str] = Counter()
    for name, tensor in sorted(tensors.items()):
        dtype = tensor["dtype"]
        shape = tensor["shape"]
        if dtype == "I8" and name.endswith(".weight"):
            scale_name = name[: -len(".weight")] + ".scale"
            scale = store.descriptor(scale_name)
            if len(shape) != 2 or len(scale["shape"]) != 2:
                raise DeepSeekV4GravityError(f"{name}: native FP4 pair is not rank two")
            row = store.rows(name, 0, 1)
            scales = store.rows(scale_name, 0, 1)
            descriptor = {key: tensor[key] for key in ("name", "dtype", "shape", "data_offsets")}
            scale_descriptor = {key: scale[key] for key in ("name", "dtype", "shape", "data_offsets")}
            decoded = codec.decode_fp4_e2m1fn_x2_rows(row, descriptor, scales, scale_descriptor, row_count=1)
            repeated = codec.decode_fp4_e2m1fn_x2_rows(row, descriptor, scales, scale_descriptor, row_count=1)
            if not np.array_equal(decoded, repeated):
                raise DeepSeekV4GravityError(f"{name}: FP4 decode was not repeatable")
            results.append({"tensor": name, "codec": "native.fp4_e2m1fn_x2", "rows": 1, "f32_sha256": _sha256(decoded.astype("<f4", copy=False).tobytes())})
            kinds["fp4"] += 1
        elif dtype == "F8_E4M3" and name.endswith(".weight"):
            scale_name = name[: -len(".weight")] + ".scale"
            scale = store.descriptor(scale_name)
            if len(shape) != 2 or len(scale["shape"]) != 2:
                raise DeepSeekV4GravityError(f"{name}: native FP8 pair is not rank two")
            row = store.rows(name, 0, 1)
            scale_cols = int(scale["shape"][1])
            scales = store.read(scale_name, 0, scale_cols)
            descriptor = {key: tensor[key] for key in ("name", "dtype", "shape", "data_offsets")}
            scale_descriptor = {key: scale[key] for key in ("name", "dtype", "shape", "data_offsets")}
            decoded = codec.decode_fp8_e4m3fn_rows(
                row,
                descriptor,
                scales,
                scale_descriptor,
                row_count=1,
                scale_block_row_start=0,
            )
            repeated = codec.decode_fp8_e4m3fn_rows(
                row,
                descriptor,
                scales,
                scale_descriptor,
                row_count=1,
                scale_block_row_start=0,
            )
            if not np.array_equal(decoded, repeated):
                raise DeepSeekV4GravityError(f"{name}: FP8 decode was not repeatable")
            results.append({"tensor": name, "codec": "native.fp8_e4m3fn_e8m0", "rows": 1, "f32_sha256": _sha256(decoded.astype("<f4", copy=False).tobytes())})
            kinds["fp8"] += 1
        elif dtype == "BF16":
            first = _bf16_to_f32(store.read(name, 0, 2))
            if not np.isfinite(first).all():
                raise DeepSeekV4GravityError(f"{name}: BF16 source first value is not finite")
            kinds["bf16"] += 1
        elif dtype == "F32":
            raw = store.read(name, 0, 4)
            value = np.frombuffer(raw, dtype="<f4")
            if not np.isfinite(value).all():
                raise DeepSeekV4GravityError(f"{name}: F32 source first value is not finite")
            kinds["f32"] += 1
        elif dtype == "F8_E8M0":
            # Its matching native weight validates it; still reject a NaN scale
            # when an otherwise orphaned scale tensor is encountered.
            value = codec.decode_e8m0fnu(store.read(name, 0, 1))
            if not np.isfinite(value).all():
                raise DeepSeekV4GravityError(f"{name}: E8M0 source first value is not finite")
            kinds["e8m0"] += 1
        elif dtype == "I64":
            # Hash-routing tables are source integer IDs, not floating-point
            # weights.  Validate one complete value without pretending this is
            # a numeric dequantization pass.
            raw = store.read(name, 0, 8)
            if len(raw) != 8:
                raise DeepSeekV4GravityError(f"{name}: truncated I64 source value")
            np.frombuffer(raw, dtype="<i8")
            kinds["i64"] += 1
        else:
            raise DeepSeekV4GravityError(f"{name}: unsupported V4 diagnostic dtype {dtype!r}")
    return {
        "status": "PASS",
        "mode": "one_source_real_complete_row_per_native_weight_pair",
        "validated_native_tensors": len(results),
        "dtype_counts": dict(kinds),
        "results": results,
        "codec_authority": {
            "repository": codec.OFFICIAL_REPOSITORY,
            "revision": codec.OFFICIAL_REVISION,
            "convert_py_sha256": codec.OFFICIAL_CONVERT_SHA256,
            "kernel_py_sha256": codec.OFFICIAL_KERNEL_SHA256,
        },
    }


def _required_names(shard: str, descriptors: list[dict[str, Any]]) -> None:
    """Refuse a seemingly valid but wrong three-shard dependency window."""

    names = {row["name"] for row in descriptors}
    if shard == DIAGNOSTIC_SHARDS[0]:
        expected = {"embed.weight"}
    elif shard == DIAGNOSTIC_SHARDS[1]:
        expected = {name for name in names if name.startswith("layers.4.")}
        if len(expected) != len(names):
            raise DeepSeekV4GravityError("layer-4 dependency shard contains an unexpected non-layer-4 tensor")
        if len(expected) != 1576:
            raise DeepSeekV4GravityError(f"layer-4 dependency window has {len(expected)} tensors, expected 1576")
        for required in (
            "layers.4.hc_attn_fn",
            "layers.4.attn.wq_a.weight",
            "layers.4.attn.indexer.wq_b.weight",
            "layers.4.ffn.gate.weight",
            "layers.4.ffn.shared_experts.w1.weight",
            "layers.4.ffn.experts.255.w2.scale",
        ):
            if required not in names:
                raise DeepSeekV4GravityError(f"layer-4 dependency window lacks {required}")
        return
    elif shard == DIAGNOSTIC_SHARDS[2]:
        expected = {"hc_head_fn", "hc_head_base", "hc_head_scale", "norm.weight", "head.weight"}
    else:
        raise DeepSeekV4GravityError(f"unexpected diagnostic shard {shard}")
    if names != expected:
        raise DeepSeekV4GravityError(f"{shard}: source tensor set differs from diagnostic dependency contract")


def _artifact_paths(artifact: Path) -> dict[str, Path]:
    return {
        "chunks": artifact / "chunks",
        "metadata": artifact / "metadata",
        "journal": artifact / "stream-journal.json",
        "ranges": artifact / "stream-ranges.jsonl",
        "restart": artifact / "restart-receipt.json",
        "manifest": artifact / "manifest.json",
    }


def _initial_journal(
    artifact: Path,
    workspace: Path,
    floor: int,
    range_bytes: int,
    *,
    shard_names: Sequence[str] = DIAGNOSTIC_SHARDS,
    contract: str = "diagnostic",
) -> dict[str, Any]:
    value = {
        "schema": JOURNAL_SCHEMA,
        "status": "IN_PROGRESS",
        "created_at": _utc_now(),
        "artifact_directory": str(artifact),
        "workspace_root": str(workspace),
        "source": {"repository": REPOSITORY, "revision": REVISION},
        "storage_policy": {
            "protected_floor_bytes": floor,
            "max_source_range_bytes": range_bytes,
            "source_parent_cache": "forbidden",
            "one_active_source_range": True,
            "one_bounded_next_range": True,
        },
        "diagnostic_contract": {
            "kind": "full_width_layer_4_only",
            "shards": list(DIAGNOSTIC_SHARDS),
            "not_full_model": True,
        },
    }
    if contract == "full_model":
        value["diagnostic_contract"] = {
            "kind": "full_43_layer_source_stream",
            "shards": list(shard_names),
            "not_full_model": False,
            "runtime_ready": False,
        }
    return value


def _load_or_create_journal(
    paths: Mapping[str, Path],
    artifact: Path,
    workspace: Path,
    floor: int,
    range_bytes: int,
    *,
    shard_names: Sequence[str] = DIAGNOSTIC_SHARDS,
    contract: str = "diagnostic",
) -> dict[str, Any]:
    manifest = paths["manifest"]
    if manifest.exists():
        raise DeepSeekV4GravityError(
            f"artifact already sealed at {manifest}; use `inspect` or a new artifact directory"
        )
    journal = paths["journal"]
    expected = _initial_journal(
        artifact,
        workspace,
        floor,
        range_bytes,
        shard_names=shard_names,
        contract=contract,
    )
    if journal.exists():
        current = _read_json(journal, "stream journal")
        for key in ("schema", "artifact_directory", "workspace_root", "source", "storage_policy", "diagnostic_contract"):
            if current.get(key) != expected.get(key):
                raise DeepSeekV4GravityError(f"restart journal differs at {key}; refuse ambiguous resume")
        return current
    _atomic_json(journal, expected)
    return expected


def _completed_ranges(paths: Mapping[str, Path], artifact: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(paths["ranges"]):
        if row.get("schema") != RANGE_SCHEMA or row.get("status") != "SEALED":
            raise DeepSeekV4GravityError("stream range journal has an invalid record")
        range_id = row.get("range_id")
        chunk_relpath = row.get("chunk_relpath")
        if not isinstance(range_id, str) or not isinstance(chunk_relpath, str):
            raise DeepSeekV4GravityError("stream range journal lacks a range identity")
        chunk = artifact / chunk_relpath
        _regular_file(chunk, f"restarted chunk {range_id}")
        raw = chunk.read_bytes()
        if len(raw) != row.get("bytes") or _sha256(raw) != row.get("sha256"):
            raise DeepSeekV4GravityError(f"restart chunk did not verify for {range_id}")
        previous = completed.get(range_id)
        if previous is not None and previous != row:
            raise DeepSeekV4GravityError(f"range journal has conflicting restart records for {range_id}")
        completed[range_id] = row
    return completed


def _write_metadata_assets(paths: Mapping[str, Path]) -> dict[str, Any]:
    assets: dict[str, Any] = {}
    for remote_path in METADATA_FILES:
        target = paths["metadata"] / remote_path
        if target.exists():
            _regular_file(target, f"metadata asset {remote_path}")
            raw = target.read_bytes()
            source = {"path": remote_path, "bytes": len(raw), "sha256": _sha256(raw), "resumed": True}
        else:
            raw, source = _direct_metadata_file(remote_path)
            _atomic_bytes(target, raw, expected_sha256=source["sha256"])
        assets[remote_path] = source
    config = json.loads((paths["metadata"] / "config.json").read_text(encoding="utf-8"))
    inference = json.loads((paths["metadata"] / "inference/config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "deepseek_v4" or config.get("architectures") != ["DeepseekV4ForCausalLM"]:
        raise DeepSeekV4GravityError("pinned source config is not DeepSeekV4ForCausalLM")
    if inference.get("n_layers") != 43 or inference.get("hc_mult") != 4 or inference.get("expert_dtype") != "fp4":
        raise DeepSeekV4GravityError("pinned source inference grammar differs from the expected V4 contract")
    if assets["inference/convert.py"]["sha256"] != codec.OFFICIAL_CONVERT_SHA256:
        raise DeepSeekV4GravityError("pinned convert.py differs from native codec authority anchor")
    if assets["inference/kernel.py"]["sha256"] != codec.OFFICIAL_KERNEL_SHA256:
        raise DeepSeekV4GravityError("pinned kernel.py differs from native codec authority anchor")
    return {"assets": assets, "config": config, "inference_config": inference}


def _chunk_path(paths: Mapping[str, Path], digest: str) -> tuple[Path, str]:
    relative = Path("chunks") / digest[:2] / digest
    return paths["chunks"].parent / relative, str(relative)


def _stream_shards(
    *,
    artifact: Path,
    paths: Mapping[str, Path],
    transport: XetTransport,
    workspace: Path,
    floor: int,
    range_bytes: int,
    completed: dict[str, dict[str, Any]],
    shard_names: Sequence[str],
    require_diagnostic_names: bool,
    journal_lock: Lock | None = None,
    dependency_parallelism: int = 1,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stream exact tensor bodies for a diagnostic or the complete 46-shard model.

    This is the one source-body path for both modes.  A shard is the active
    dependency window; every source range is sealed as a content-addressed
    chunk before the next range is fetched, and no parent safetensors file is
    ever opened or retained.
    """

    source_windows: list[dict[str, Any]] = []
    tensors: dict[str, dict[str, Any]] = {}
    all_floor_checks: list[dict[str, Any]] = []
    for shard in shard_names:
        metadata = transport.metadata(shard)
        group = None if transport.direct_http else transport.group()
        header_raw, header, header_checks = _header_from_xet(
            transport, group, metadata, workspace=workspace, floor=floor
        )
        all_floor_checks.extend(header_checks)
        descriptors = _normalise_descriptors(
            header, header_bytes=len(header_raw), file_size=int(metadata["file_size_bytes"])
        )
        if require_diagnostic_names:
            _required_names(shard, descriptors)
        shard_hash = hashlib.sha256()
        shard_hash.update(header_raw)
        body_bytes = 0
        tensor_count = 0
        for descriptor in descriptors:
            name = str(descriptor["name"])
            segments: list[dict[str, Any]] = []
            for body_start, body_end, row_start, row_count in _segment_rows(descriptor, range_bytes):
                file_start = int(descriptor["file_start"]) + body_start
                file_end = int(descriptor["file_start"]) + body_end
                rid = _range_id(shard, name, file_start, file_end)
                row = completed.get(rid)
                if row is None:
                    raw, checks = transport.read_range(
                        group,
                        metadata,
                        file_start,
                        file_end,
                        workspace=workspace,
                        floor=floor,
                        stage=f"{shard}:{name}:{row_start}",
                    )
                    all_floor_checks.extend(checks)
                    digest = _sha256(raw)
                    target, relative = _chunk_path(paths, digest)
                    _atomic_bytes(target, raw, expected_sha256=digest)
                    row = {
                        "schema": RANGE_SCHEMA,
                        "status": "SEALED",
                        "sealed_at": _utc_now(),
                        "range_id": rid,
                        "source": {
                            "repository": REPOSITORY,
                            "revision": REVISION,
                            "shard": shard,
                            "source_file_start": file_start,
                            "source_file_end": file_end,
                        },
                        "tensor": name,
                        "tensor_start": body_start,
                        "tensor_end": body_end,
                        "row_start": row_start,
                        "row_count": row_count,
                        "bytes": len(raw),
                        "sha256": digest,
                        "chunk_relpath": relative,
                        "source_body_persisted_outside_artifact": False,
                    }
                    # Full-model workers share one append-only receipt.  The
                    # lock covers the durable append and the in-memory
                    # binding together so a restart record can never be
                    # observed half-written by another worker.
                    if journal_lock is None:
                        _append_jsonl(paths["ranges"], row)
                        completed[rid] = row
                    else:
                        with journal_lock:
                            existing = completed.get(rid)
                            if existing is None:
                                _append_jsonl(paths["ranges"], row)
                                completed[rid] = row
                            else:
                                row = existing
                    del raw
                elif int(row["bytes"]) != body_end - body_start:
                    raise DeepSeekV4GravityError(f"restart range size differs for {rid}")
                chunk = artifact / row["chunk_relpath"]
                raw_for_hash = chunk.read_bytes()
                if _sha256(raw_for_hash) != row["sha256"]:
                    raise DeepSeekV4GravityError(f"artifact chunk changed after seal for {rid}")
                shard_hash.update(raw_for_hash)
                del raw_for_hash
                segments.append(
                    {
                        "tensor_start": body_start,
                        "tensor_end": body_end,
                        "bytes": int(row["bytes"]),
                        "sha256": row["sha256"],
                        "chunk_relpath": row["chunk_relpath"],
                        "source_file_start": file_start,
                        "source_file_end": file_end,
                        "row_start": row_start,
                        "row_count": row_count,
                    }
                )
                body_bytes += body_end - body_start
            if sum(int(item["bytes"]) for item in segments) != int(descriptor["bytes"]):
                raise DeepSeekV4GravityError(f"incomplete tensor chunk mapping for {name}")
            if name in tensors:
                raise DeepSeekV4GravityError(f"duplicate streamed tensor name {name}")
            tensors[name] = {
                "name": name,
                "dtype": descriptor["dtype"],
                "shape": descriptor["shape"],
                "data_offsets": descriptor["data_offsets"],
                "bytes": descriptor["bytes"],
                "source_shard": shard,
                "source_file_start": descriptor["file_start"],
                "source_file_end": descriptor["file_end"],
                "segments": segments,
            }
            tensor_count += 1
        observed_sha = shard_hash.hexdigest()
        if observed_sha != metadata["etag_sha256"]:
            raise DeepSeekV4GravityError(
                f"{shard}: full streamed SHA-256 {observed_sha} differs from Hub LFS identity {metadata['etag_sha256']}"
            )
        source_windows.append(
            {
                "schema": SOURCE_RECEIPT_SCHEMA,
                "status": "VERIFIED_STREAMED_AND_EVICTED",
                # The signed CDN URL is an in-memory transport credential and
                # must never be sealed into a durable receipt.  Keep only the
                # fact that the optimized signed path was used.
                "source": {
                    key: value for key, value in metadata.items() if key != "signed_location"
                },
                "signed_cdn_location_used": True,
                "header_bytes": len(header_raw),
                "header_sha256": _sha256(header_raw),
                "tensor_count": tensor_count,
                "body_bytes": body_bytes,
                "streamed_full_file_sha256": observed_sha,
                "verified_against_hub_lfs_sha256": True,
                "source_parent_persisted": False,
                "one_active_dependency_window": dependency_parallelism == 1,
                "parallel_dependency_windows": dependency_parallelism,
            }
        )
        del header_raw, header, descriptors
    return tensors, source_windows, all_floor_checks


def _stream_shards_parallel(
    *,
    artifact: Path,
    paths: Mapping[str, Path],
    transport: XetTransport,
    workspace: Path,
    floor: int,
    range_bytes: int,
    completed: dict[str, dict[str, Any]],
    shard_names: Sequence[str],
    workers: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stream bounded shard workers into one durable, restartable journal.

    Each worker owns one dependency-complete shard and its Xet stream group.
    The only shared writer is the append-only range journal, guarded by a
    lock; content-addressed chunks remain independently atomic.  ``workers``
    is deliberately capped by the caller so the source transport and disk
    floor cannot turn into an unbounded downloader.
    """

    if workers < 1 or workers > MAX_PARALLEL_WORKERS:
        raise DeepSeekV4GravityError(
            f"parallel_workers must be between 1 and {MAX_PARALLEL_WORKERS}"
        )
    journal_lock = Lock()

    def run_one(shard: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        return _stream_shards(
            artifact=artifact,
            paths=paths,
            transport=transport,
            workspace=workspace,
            floor=floor,
            range_bytes=range_bytes,
            completed=completed,
            shard_names=(shard,),
            require_diagnostic_names=False,
            journal_lock=journal_lock,
            dependency_parallelism=workers,
        )

    tensors: dict[str, dict[str, Any]] = {}
    source_windows: list[dict[str, Any]] = []
    floor_checks: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v4-shard") as pool:
        futures = {pool.submit(run_one, shard): shard for shard in shard_names}
        try:
            for future in as_completed(futures):
                shard = futures[future]
                shard_tensors, shard_windows, shard_checks = future.result()
                overlap = set(tensors).intersection(shard_tensors)
                if overlap:
                    raise DeepSeekV4GravityError(
                        f"parallel shard workers produced duplicate tensors: {sorted(overlap)[:3]}"
                    )
                tensors.update(shard_tensors)
                source_windows.extend(shard_windows)
                floor_checks.extend(shard_checks)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    source_windows.sort(key=lambda row: str(row.get("source", {}).get("shard", "")))
    floor_checks.sort(key=lambda row: str(row.get("stage", "")))
    return tensors, source_windows, floor_checks


def build_diagnostic(
    *,
    artifact_dir: str | Path,
    workspace_root: str | Path,
    xet_root: str | Path,
    protected_floor_bytes: int = MIN_FREE_FLOOR_BYTES,
    range_bytes: int = DEFAULT_RANGE_BYTES,
) -> dict[str, Any]:
    """Stream, native-validate, and atomically seal a real V4 diagnostic artifact."""

    artifact = _absolute(artifact_dir, "artifact_dir")
    workspace = _absolute(workspace_root, "workspace_root")
    retention = _absolute(xet_root, "xet_root")
    floor = _positive(protected_floor_bytes, "protected_floor_bytes")
    if floor < MIN_FREE_FLOOR_BYTES:
        raise DeepSeekV4GravityError("protected_floor_bytes cannot be below the non-negotiable 15 GiB")
    size = _positive(range_bytes, "range_bytes")
    if size > MAX_RANGE_BYTES:
        raise DeepSeekV4GravityError(f"range_bytes cannot exceed {MAX_RANGE_BYTES}")
    _ensure_dir(workspace, "workspace_root")
    _floor_check(workspace, floor, size, "before-artifact-build")
    _ensure_dir(artifact, "artifact directory")
    if retention.resolve(strict=False) == artifact.resolve(strict=False):
        raise DeepSeekV4GravityError("xet_root must be separate from artifact_dir")
    # A sealed artifact is an immutable content-addressed result.  Treating a
    # second public `gravity execute` as a *full* local re-verification avoids
    # touching the original source window while proving every derived chunk,
    # journal binding, and runtime load is still intact.
    if (artifact / "manifest.json").is_file():
        return reverify_sealed_diagnostic(artifact)
    paths = _artifact_paths(artifact)
    _ensure_dir(paths["chunks"], "artifact chunk root")
    _ensure_dir(paths["metadata"], "artifact metadata root")
    journal = _load_or_create_journal(paths, artifact, workspace, floor, size)
    completed = _completed_ranges(paths, artifact)
    assets = _write_metadata_assets(paths)
    transport = XetTransport(retention)
    try:
        tensors, source_windows, all_floor_checks = _stream_shards(
            artifact=artifact,
            paths=paths,
            transport=transport,
            workspace=workspace,
            floor=floor,
            range_bytes=size,
            completed=completed,
            shard_names=DIAGNOSTIC_SHARDS,
            require_diagnostic_names=True,
        )
        if len(tensors) != 1582:
            raise DeepSeekV4GravityError(f"diagnostic has {len(tensors)} tensors, expected 1582")
        native_validation = _native_validation(artifact, tensors)
        no_xet_files = transport.assert_no_files()
        chunks = sorted(
            {
                segment["sha256"]
                for tensor in tensors.values()
                for segment in tensor["segments"]
            }
        )
        total_bytes = sum(int(tensor["bytes"]) for tensor in tensors.values())
        restart = seal(
            {
                "schema": "hawking.gravity.deepseek_v4.restart_receipt.v1",
                "status": "SEALED",
                "created_at": _utc_now(),
                "journal_sha256": _sha256(paths["journal"].read_bytes()),
                "range_journal_sha256": _sha256(paths["ranges"].read_bytes()),
                "range_count": len(completed),
                "source_windows": source_windows,
                "source_parent_retained": False,
                "xet_retention_files": no_xet_files,
                "resume_command": (
                    "tools/condense/.venv/bin/python tools/condense/deepseek_v4_gravity.py build "
                    f"--artifact-dir {artifact} --workspace-root {workspace} --xet-root <NEW_EMPTY_XET_ROOT>"
                ),
            }
        )
        _atomic_json(paths["restart"], restart)
        manifest = seal(
            {
                "schema": ARTIFACT_SCHEMA,
                "status": "DIAGNOSTIC_SEALED_LOADABLE_BY_V4_NUMPY_ADAPTER",
                "created_at": _utc_now(),
                "artifact": {
                    "kind": "deepseek_v4_full_width_layer_4_diagnostic",
                    "format": "gravity.content_addressed.chunk_directory.v1",
                    "total_tensor_bytes": total_bytes,
                    "content_addressed_chunk_count": len(chunks),
                    "content_addressed_chunk_sha256": _sha256(_canonical(chunks)),
                },
                "source": {
                    "repository": REPOSITORY,
                    "revision": REVISION,
                    "authority": "pinned_public_huggingface_hub_control_plane_plus_full_streamed_lfs_sha256",
                    "metadata_assets": assets["assets"],
                    "source_windows": source_windows,
                },
                "diagnostic_scope": {
                    "full_width": True,
                    "selected_layer": 4,
                    "contains_all_layer_4_routed_experts": 256,
                    "contains": [
                        "tokenizer_and_raw_prompt_template",
                        "embedding",
                        "hyper_connections_control_path",
                        "complete_sparse_compressed_attention_path",
                        "score_router",
                        "shared_expert",
                        "all_routed_expert_paths",
                        "head",
                        "generation_loop_adapter",
                    ],
                    "not_full_model": True,
                    "quality_claim": "not_inferable_from_layer_limited_diagnostic",
                    "missing_context": "layers_0_to_3_and_5_to_42_are_not_present",
                },
                "architecture": {
                    "model_type": assets["config"].get("model_type"),
                    "architectures": assets["config"].get("architectures"),
                    "inference_config": assets["inference_config"],
                    "layer_tensor_count": 1576,
                    "tensor_count": len(tensors),
                },
                "representation_and_kernel_grammar": {
                    "fp4": "E2M1FN x2 packed low-nibble then high-nibble along K; E8M0FNU scale per 32 logical K",
                    "fp8": "E4M3FN weights; E8M0FNU scale per 128 by 128 block",
                    "bf16": "IEEE bfloat16 little-endian source-preserving",
                    "fallback_runtime": "CPU NumPy diagnostic adapter; native source weight decode is exact, activation QAT/Metal parity remains unproven",
                    "official_convert_py_sha256": codec.OFFICIAL_CONVERT_SHA256,
                    "official_kernel_py_sha256": codec.OFFICIAL_KERNEL_SHA256,
                },
                "native_decode_validation": native_validation,
                "storage": {
                    "protected_floor_bytes": floor,
                    "floor_checks": all_floor_checks,
                    "source_parent_retained": False,
                    "xet_retention_files": no_xet_files,
                    "one_active_source_range": True,
                    "one_bounded_next_range": True,
                    "source_raw_bytes_retained_in_artifact": total_bytes,
                    "note": "Only the 5.316 GiB layer-limited diagnostic is retained as a derived Gravity artifact; the 159.6 GB parent is never materialized or retained.",
                },
                "runtime_adapter": {
                    "id": "hawking.gravity.deepseek_v4_numpy_diagnostic.v1",
                    "registration": "lab.operators.deepseek_v4_gravity:DeepSeekV4DiagnosticRuntime",
                    "load_command": "tools/condense/.venv/bin/python tools/condense/deepseek_v4_gravity.py inspect --artifact-dir <PATH>",
                    "device": "cpu",
                    "metal_dispatches": 0,
                    "capability_status": "diagnostic_cpu_only_not_tg_eligible",
                },
                "tensors": tensors,
                "restart_receipt": {
                    "path": "restart-receipt.json",
                    "seal_sha256": restart["seal_sha256"],
                },
            }
        )
        _atomic_json(paths["manifest"], manifest)
        return manifest
    finally:
        # A zero-sized / directory-only Xet root is intentionally left for
        # inspection.  Never recursively delete a caller-specified path.
        pass


def _write_full_index(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Capture the source-owned complete tensor index into the artifact metadata."""

    relative = "model.safetensors.index.json"
    target = paths["metadata"] / relative
    if target.exists():
        _regular_file(target, "full model tensor index")
        raw = target.read_bytes()
        source = {"path": relative, "bytes": len(raw), "sha256": _sha256(raw), "resumed": True}
    else:
        raw, source = _direct_metadata_file(relative)
        _atomic_bytes(target, raw, expected_sha256=source["sha256"])
    try:
        index = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeepSeekV4GravityError("source tensor index is not valid JSON") from exc
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise DeepSeekV4GravityError("source tensor index lacks a weight_map")
    weight_map = index["weight_map"]
    shards = {value for value in weight_map.values() if isinstance(value, str)}
    if len(weight_map) != FULL_EXPECTED_TENSOR_COUNT or shards != set(FULL_SHARDS):
        raise DeepSeekV4GravityError(
            "source tensor index does not describe the pinned 46-shard full model: "
            f"tensors={len(weight_map)}, shards={len(shards)}"
        )
    return {"index": index, "asset": source}


def _full_restart_receipt(
    *,
    paths: Mapping[str, Path],
    completed: Mapping[str, Mapping[str, Any]],
    source_windows: Sequence[Mapping[str, Any]],
    no_xet_files: Sequence[str],
    artifact: Path,
    workspace: Path,
    parallel_workers: int,
) -> dict[str, Any]:
    return seal(
        {
            "schema": "hawking.gravity.deepseek_v4.full_restart_receipt.v1",
            "status": "SEALED",
            "created_at": _utc_now(),
            "journal_sha256": _sha256(paths["journal"].read_bytes()),
            "range_journal_sha256": _sha256(paths["ranges"].read_bytes()),
            "range_count": len(completed),
            "source_window_count": len(source_windows),
            "source_windows": list(source_windows),
            "source_parent_retained": False,
            "parallel_workers": parallel_workers,
            "max_active_dependency_windows": parallel_workers,
            "xet_retention_files": list(no_xet_files),
            "resume_command": (
                "tools/condense/.venv/bin/python tools/condense/deepseek_v4_gravity.py build-full "
                f"--artifact-dir {artifact} --workspace-root {workspace} --xet-root <NEW_EMPTY_XET_ROOT>"
            ),
        }
    )


def build_full_model(
    *,
    artifact_dir: str | Path,
    workspace_root: str | Path,
    xet_root: str | Path,
    protected_floor_bytes: int = MIN_FREE_FLOOR_BYTES,
    range_bytes: int = DEFAULT_RANGE_BYTES,
    parallel_workers: int = 4,
) -> dict[str, Any]:
    """Stream and seal all 46 pinned model shards.

    The result is deliberately marked ``NOT_RUNTIME_READY``.  It proves the
    physical full-model artifact and restart contract, but the current local
    NumPy adapter remains the layer-4 diagnostic and must not be relabelled as
    a 43-layer inference engine.
    """

    artifact = _absolute(artifact_dir, "artifact_dir")
    workspace = _absolute(workspace_root, "workspace_root")
    retention = _absolute(xet_root, "xet_root")
    floor = _positive(protected_floor_bytes, "protected_floor_bytes")
    if floor < MIN_FREE_FLOOR_BYTES:
        raise DeepSeekV4GravityError("protected_floor_bytes cannot be below the non-negotiable 15 GiB")
    size = _positive(range_bytes, "range_bytes")
    if size > MAX_RANGE_BYTES:
        raise DeepSeekV4GravityError(f"range_bytes cannot exceed {MAX_RANGE_BYTES}")
    if parallel_workers < 1 or parallel_workers > MAX_PARALLEL_WORKERS:
        raise DeepSeekV4GravityError(
            f"parallel_workers must be between 1 and {MAX_PARALLEL_WORKERS}"
        )
    _ensure_dir(workspace, "workspace_root")
    _floor_check(workspace, floor, size, "before-full-model-build")
    _ensure_dir(artifact, "artifact directory")
    if retention.resolve(strict=False) == artifact.resolve(strict=False):
        raise DeepSeekV4GravityError("xet_root must be separate from artifact_dir")
    if (artifact / "manifest.json").is_file():
        return reverify_full_model(artifact)
    paths = _artifact_paths(artifact)
    _ensure_dir(paths["chunks"], "artifact chunk root")
    _ensure_dir(paths["metadata"], "artifact metadata root")
    _load_or_create_journal(
        paths,
        artifact,
        workspace,
        floor,
        size,
        shard_names=FULL_SHARDS,
        contract="full_model",
    )
    completed = _completed_ranges(paths, artifact)
    assets = _write_metadata_assets(paths)
    index_data = _write_full_index(paths)
    config = assets["config"]
    inference = assets["inference_config"]
    if config.get("num_hidden_layers") != 43 or inference.get("n_layers") != 43:
        raise DeepSeekV4GravityError("pinned full model config does not declare 43 layers")
    # The public unauthenticated hf_xet Python session can be heavily
    # throttled after sustained large-object traffic.  The signed Xet bridge
    # still exposes the same pinned object as exact HTTP ranges, so use that
    # path for the full model while keeping the diagnostic on hf_xet.
    transport = XetTransport(retention, direct_http=True)
    tensors, source_windows, floor_checks = _stream_shards_parallel(
        artifact=artifact,
        paths=paths,
        transport=transport,
        workspace=workspace,
        floor=floor,
        range_bytes=size,
        completed=completed,
        shard_names=FULL_SHARDS,
        workers=parallel_workers,
    )
    if len(tensors) != FULL_EXPECTED_TENSOR_COUNT:
        raise DeepSeekV4GravityError(
            f"full streamed model has {len(tensors)} tensors, expected {FULL_EXPECTED_TENSOR_COUNT}"
        )
    indexed_names = set(index_data["index"]["weight_map"])
    if set(tensors) != indexed_names:
        missing = sorted(indexed_names - set(tensors))[:5]
        extra = sorted(set(tensors) - indexed_names)[:5]
        raise DeepSeekV4GravityError(
            f"full streamed tensor set differs from source index; missing={missing}, extra={extra}"
        )
    native_validation = _native_validation(artifact, tensors)
    no_xet_files = transport.assert_no_files()
    chunks = sorted(
        {
            segment["sha256"]
            for tensor in tensors.values()
            for segment in tensor["segments"]
        }
    )
    total_bytes = sum(int(tensor["bytes"]) for tensor in tensors.values())
    restart = _full_restart_receipt(
        paths=paths,
        completed=completed,
        source_windows=source_windows,
        no_xet_files=no_xet_files,
        artifact=artifact,
        workspace=workspace,
        parallel_workers=parallel_workers,
    )
    _atomic_json(paths["restart"], restart)
    manifest = seal(
        {
            "schema": FULL_ARTIFACT_SCHEMA,
            "status": "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY",
            "created_at": _utc_now(),
            "artifact": {
                "kind": "deepseek_v4_full_43_layer_content_addressed_stream",
                "format": "gravity.content_addressed.chunk_directory.v1",
                "total_tensor_bytes": total_bytes,
                "source_index_total_size_bytes": index_data["index"].get("metadata", {}).get("total_size"),
                "content_addressed_chunk_count": len(chunks),
                "content_addressed_chunk_sha256": _sha256(_canonical(chunks)),
            },
            "source": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "authority": "pinned_public_huggingface_hub_control_plane_plus_full_streamed_lfs_sha256",
                "metadata_assets": {**assets["assets"], "model.safetensors.index.json": index_data["asset"]},
                "source_windows": source_windows,
                "source_shard_count": len(FULL_SHARDS),
                "source_parent_persisted": False,
            },
            "full_model_scope": {
                "full_width": True,
                "num_hidden_layers": 43,
                "contains_all_routed_experts_per_layer": 256,
                "contains_embedding_and_head": True,
                "tensor_count": len(tensors),
                "runtime_ready": False,
                "runtime_blocker": "full streamed artifact has no local 43-layer execution adapter yet",
                "quality_claim": "not_claimed_until_full_runtime_forward_is_executed",
            },
            "architecture": {
                "model_type": config.get("model_type"),
                "architectures": config.get("architectures"),
                "inference_config": inference,
                "layer_count": 43,
                "tensor_count": len(tensors),
            },
            "representation_and_kernel_grammar": {
                "fp4": "E2M1FN x2 packed low-nibble then high-nibble along K; E8M0FNU scale per 32 logical K",
                "fp8": "E4M3FN weights; E8M0FNU scale per 128 by 128 block",
                "bf16": "IEEE bfloat16 little-endian source-preserving",
                "native_validation": "every streamed native weight/scale pair has a source-row decode check",
                "official_convert_py_sha256": codec.OFFICIAL_CONVERT_SHA256,
                "official_kernel_py_sha256": codec.OFFICIAL_KERNEL_SHA256,
            },
            "native_decode_validation": native_validation,
            "storage": {
                "protected_floor_bytes": floor,
                "floor_checks": floor_checks,
                "source_parent_retained": False,
                "source_transport": "signed_huggingface_xet_bridge_http_exact_range",
                "authenticated": transport.authenticated,
                "xet_retention_files": no_xet_files,
                "one_active_source_range": parallel_workers == 1,
                "one_active_source_range_per_worker": True,
                "one_bounded_next_range": True,
                "parallel_source_shards": parallel_workers,
                "max_active_dependency_windows": parallel_workers,
                "source_raw_bytes_retained_in_artifact": total_bytes,
                "note": "The 159.6 GB source parent was never materialized; only verified content-addressed tensor chunks remain. Source shard parallelism is bounded and the 15 GiB floor remains enforced.",
            },
            "runtime_adapter": {
                "id": None,
                "registration": None,
                "device": None,
                "metal_dispatches": 0,
                "capability_status": "full_artifact_streamed_runtime_pending",
            },
            "tensors": tensors,
            "restart_receipt": {
                "path": "restart-receipt.json",
                "seal_sha256": restart["seal_sha256"],
            },
        }
    )
    _atomic_json(paths["manifest"], manifest)
    return manifest


def load_full_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    artifact = _absolute(artifact_dir, "artifact_dir")
    manifest = _read_json(artifact / "manifest.json", "V4 full model manifest")
    try:
        verified = verify(manifest, label="DeepSeek-V4 full model manifest")
    except SealIntegrityError as exc:
        raise DeepSeekV4GravityError(str(exc)) from exc
    if verified.get("schema") != FULL_ARTIFACT_SCHEMA:
        raise DeepSeekV4GravityError("artifact manifest is not a DeepSeek-V4 full stream")
    if verified.get("status") != "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY":
        raise DeepSeekV4GravityError("full stream is not sealed")
    return verified


def reverify_full_model(artifact_dir: str | Path) -> dict[str, Any]:
    """Full local chunk and journal re-verification for a sealed full stream."""

    started = time.perf_counter()
    artifact = _absolute(artifact_dir, "artifact_dir")
    manifest = load_full_manifest(artifact)
    tensors = manifest.get("tensors")
    if not isinstance(tensors, Mapping) or len(tensors) != FULL_EXPECTED_TENSOR_COUNT:
        raise DeepSeekV4GravityError("sealed full stream lacks the complete tensor mapping")
    expected_chunks: dict[str, tuple[str, int]] = {}
    for name, descriptor in tensors.items():
        if not isinstance(name, str) or not isinstance(descriptor, Mapping):
            raise DeepSeekV4GravityError("sealed full tensor mapping is invalid")
        segments = descriptor.get("segments")
        if not isinstance(segments, list) or not segments:
            raise DeepSeekV4GravityError(f"sealed full tensor {name} has no segments")
        for segment in segments:
            relative = segment.get("chunk_relpath")
            digest = segment.get("sha256")
            size = segment.get("bytes")
            if not isinstance(relative, str) or not isinstance(digest, str) or not isinstance(size, int):
                raise DeepSeekV4GravityError(f"sealed full tensor {name} has an incomplete segment")
            previous = expected_chunks.get(relative)
            if previous is not None and previous != (digest, size):
                raise DeepSeekV4GravityError(f"full stream chunk binding conflict for {relative}")
            expected_chunks[relative] = (digest, size)
    checked_bytes = 0
    for relative, (digest, size) in sorted(expected_chunks.items()):
        path = artifact / relative
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(artifact.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise DeepSeekV4GravityError(f"full stream chunk path escapes artifact: {relative}") from exc
        if path.stat().st_size != size:
            raise DeepSeekV4GravityError(f"full stream chunk byte size differs for {relative}")
        observed = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(DEFAULT_RANGE_BYTES):
                observed.update(block)
        if observed.hexdigest() != digest:
            raise DeepSeekV4GravityError(f"full stream chunk hash differs for {relative}")
        checked_bytes += size
    restart = _read_json(artifact / "restart-receipt.json", "full restart receipt")
    try:
        verify(restart, label="DeepSeek-V4 full restart receipt")
    except SealIntegrityError as exc:
        raise DeepSeekV4GravityError(str(exc)) from exc
    paths = _artifact_paths(artifact)
    if restart.get("journal_sha256") != _sha256(paths["journal"].read_bytes()):
        raise DeepSeekV4GravityError("full restart journal binding differs")
    if restart.get("range_journal_sha256") != _sha256(paths["ranges"].read_bytes()):
        raise DeepSeekV4GravityError("full restart range journal binding differs")
    return seal(
        {
            "schema": "hawking.gravity.deepseek_v4.full_reverify.v1",
            "status": "FULL_MODEL_STREAM_FULLY_REVERIFIED_RUNTIME_PENDING",
            "artifact_dir": str(artifact),
            "artifact_seal_sha256": manifest["seal_sha256"],
            "source_parent_retained": False,
            "source_transport": {"activated": False, "reason": "sealed full stream re-verification"},
            "full_chunk_verification": {
                "chunk_count": len(expected_chunks),
                "bytes_verified": checked_bytes,
                "sha256_verified": True,
            },
            "tensor_count": len(tensors),
            "runtime_adapter": manifest["runtime_adapter"],
            "resume_safe": True,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
    )


def inspect_full_stream(artifact_dir: str | Path) -> dict[str, Any]:
    """Read the sealed full-stream control plane without hashing 159 GB again."""

    artifact = _absolute(artifact_dir, "artifact_dir")
    manifest = load_full_manifest(artifact)
    restart = _read_json(artifact / "restart-receipt.json", "full restart receipt")
    try:
        verify(restart, label="DeepSeek-V4 full restart receipt")
    except SealIntegrityError as exc:
        raise DeepSeekV4GravityError(str(exc)) from exc
    paths = _artifact_paths(artifact)
    if restart.get("journal_sha256") != _sha256(paths["journal"].read_bytes()):
        raise DeepSeekV4GravityError("full restart journal binding differs")
    if restart.get("range_journal_sha256") != _sha256(paths["ranges"].read_bytes()):
        raise DeepSeekV4GravityError("full restart range journal binding differs")
    scope = manifest.get("full_model_scope")
    source = manifest.get("source")
    storage = manifest.get("storage")
    return {
        "schema": "hawking.gravity.deepseek_v4.full_inspect.v1",
        "status": "FULL_MODEL_STREAM_SEALED_RUNTIME_PENDING",
        "artifact_dir": str(artifact),
        "artifact_seal_sha256": manifest["seal_sha256"],
        "restart_seal_sha256": restart["seal_sha256"],
        "tensor_count": len(manifest.get("tensors", {})),
        "source_shard_count": len(source.get("source_windows", [])) if isinstance(source, Mapping) else None,
        "source_parent_retained": storage.get("source_parent_retained") if isinstance(storage, Mapping) else None,
        "runtime_adapter": manifest.get("runtime_adapter"),
        "full_model_scope": scope,
        "next_command": (
            "tools/condense/.venv/bin/python tools/condense/deepseek_v4_gravity.py "
            f"reverify-full --artifact-dir {artifact}"
        ),
    }


def _system_memory_bytes() -> int | None:
    """Return physical memory when the host exposes POSIX page counters."""

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(pages, int) or not isinstance(page_size, int) or pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def _dependency_probe(package: str) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}
    return {"installed": True, "version": version}


def full_runtime_blocker_report(
    artifact_dir: str | Path, out: str | Path
) -> dict[str, Any]:
    """Seal the exact, current boundary before a full V4 runtime claim.

    This is a *failure-policy evidence record*, not a status override.  It is
    emitted only when the full source stream has a deliberately absent adapter
    registration, so it cannot be used to relabel a diagnostic or a source
    directory as a runnable 43-layer model.
    """

    artifact = _absolute(artifact_dir, "artifact_dir")
    target = _absolute(out, "out")
    manifest = load_full_manifest(artifact)
    scope = manifest.get("full_model_scope")
    storage = manifest.get("storage")
    source = manifest.get("source")
    runtime = manifest.get("runtime_adapter")
    if not isinstance(scope, Mapping) or not isinstance(storage, Mapping):
        raise DeepSeekV4GravityError("sealed full stream lacks scope/storage evidence")
    if not isinstance(runtime, Mapping):
        raise DeepSeekV4GravityError("sealed full stream lacks runtime adapter evidence")
    if runtime.get("registration") is not None or runtime.get("id") is not None:
        raise DeepSeekV4GravityError(
            "full runtime blocker receipt is unavailable once a registered adapter exists"
        )
    floor = int(storage.get("protected_floor_bytes", MIN_FREE_FLOOR_BYTES))
    if floor < MIN_FREE_FLOOR_BYTES:
        raise DeepSeekV4GravityError("sealed full stream recorded an invalid storage floor")
    disk = shutil.disk_usage(artifact)
    total_tensor_bytes = int(manifest.get("artifact", {}).get("total_tensor_bytes", 0))
    if total_tensor_bytes <= 0:
        raise DeepSeekV4GravityError("sealed full stream lacks total source tensor bytes")
    metadata = artifact / "metadata"
    encoding = metadata / "encoding"
    package_probe = {
        package: _dependency_probe(package)
        for package in ("torch", "mlx", "mlx-lm", "transformers", "tilelang")
    }
    source_windows = source.get("source_windows", []) if isinstance(source, Mapping) else []
    report = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.full_runtime_blocker.v1",
            "status": "FULL_STREAMED_RUNTIME_NO_REGISTERED_43_LAYER_ADAPTER",
            "created_at": _utc_now(),
            "artifact": {
                "path": str(artifact),
                "manifest_seal_sha256": manifest["seal_sha256"],
                "repository": source.get("repository") if isinstance(source, Mapping) else None,
                "revision": source.get("revision") if isinstance(source, Mapping) else None,
                "tensor_count": len(manifest.get("tensors", {})),
                "source_shard_count": len(source_windows) if isinstance(source_windows, list) else None,
                "logical_tensor_bytes": total_tensor_bytes,
                "full_stream_status": manifest["status"],
                "source_parent_retained": storage.get("source_parent_retained"),
            },
            "milestones": {
                "source_streamed": True,
                "manifest_loadable": True,
                "full_chunk_reverify_required_before_promotion": True,
                "runtime_adapter_registered": False,
                "full_runtime_load": False,
                "full_first_forward": False,
                "full_first_token": False,
                "full_hcli_endpoint": False,
                "base_true_tps": False,
            },
            "first_missing_milestone": {
                "stage": "full_v4_adapter_registration_and_native_execution",
                "artifact_runtime_adapter": dict(runtime),
                "manifest_runtime_blocker": scope.get("runtime_blocker"),
                "public_cli_expected_refusal": (
                    "the DeepSeek-V4 full stream is sealed but runtime-pending; it cannot be served "
                    "until a registered 43-layer adapter is available"
                ),
            },
            "missing_execution_grammar": {
                "native_tensor_formats": manifest.get("representation_and_kernel_grammar"),
                "required_runtime_semantics": [
                    "43-layer Hyper-Connections forward",
                    "native FP4 E2M1FN x2 plus E8M0FNU scale decode/accumulate",
                    "native FP8 E4M3FN plus E8M0FNU scale decode/accumulate",
                    "MLA compression/indexer/sparse attention",
                    "Sinkhorn control path",
                    "router top-6 routed experts plus shared expert",
                    "source tokenizer and chat-format encoding",
                ],
                "encoding_assets_streamed": encoding.is_dir(),
                "encoding_asset_paths": (
                    sorted(str(path.relative_to(artifact)) for path in encoding.rglob("*") if path.is_file())
                    if encoding.is_dir()
                    else []
                ),
            },
            "engine_environment": {
                "python_executable": sys.executable,
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
                "physical_memory_bytes": _system_memory_bytes(),
                "packages": package_probe,
            },
            "storage_accounting": {
                "artifact_logical_tensor_bytes": total_tensor_bytes,
                "free_bytes": disk.free,
                "protected_floor_bytes": floor,
                "max_additional_successor_bytes_while_preserving_floor": max(0, disk.free - floor),
                "raw_parent_materialized": False,
                "raw_artifact_eviction_authorized": False,
                "eviction_rule": "retain the raw content-addressed source stream until a separately sealed and independently verified successor exists",
            },
            "restart_safety": {
                "resume_safe": True,
                "source_parent_retained": False,
                "sealed_manifest": "manifest.json",
                "sealed_restart_receipt": "restart-receipt.json",
            },
            "next_commands": {
                "reproduce_public_refusal": (
                    "workspace/ops/build/rust/debug/hawking gravity serve --artifact "
                    f"{artifact} --addr 127.0.0.1:8791"
                ),
                "reverify_before_any_promotion": (
                    "tools/condense/.venv/bin/python tools/condense/deepseek_v4_gravity.py "
                    f"reverify-full --artifact-dir {artifact}"
                ),
                "implementation_requirement": (
                    "register a 43-layer DeepSeek-V4 Gravity adapter with the listed native kernels/"
                    "semantics, then execute parity, first-forward, first-token, HCLI, and TPS gates"
                ),
            },
        }
    )
    _atomic_json(target, report)
    return report


def _static_residency_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeepSeekV4GravityError(f"{label} must be an object")
    return value


def _static_residency_integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DeepSeekV4GravityError(f"{label} must be an integer >= {minimum}")
    return value


def _static_residency_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DeepSeekV4GravityError(f"{label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise DeepSeekV4GravityError(f"{label} must be a SHA-256 digest") from exc
    return value


def _static_residency_descriptor(
    tensors: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    descriptor = _static_residency_mapping(tensors.get(name), f"full tensor {name}")
    if descriptor.get("name") != name:
        raise DeepSeekV4GravityError(f"full tensor descriptor name mismatch for {name}")
    _static_residency_integer(descriptor.get("bytes"), f"full tensor {name}.bytes", minimum=1)
    if not isinstance(descriptor.get("dtype"), str) or not descriptor["dtype"]:
        raise DeepSeekV4GravityError(f"full tensor {name} has no dtype")
    shape = descriptor.get("shape")
    if not isinstance(shape, list) or not shape:
        raise DeepSeekV4GravityError(f"full tensor {name} has no shape")
    for ordinal, dimension in enumerate(shape):
        _static_residency_integer(dimension, f"full tensor {name}.shape[{ordinal}]", minimum=1)
    return descriptor


def _static_residency_descriptor_rows(
    tensors: Mapping[str, Any], names: Iterable[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(names):
        descriptor = _static_residency_descriptor(tensors, name)
        rows.append(
            {
                "name": name,
                "bytes": int(descriptor["bytes"]),
                "dtype": descriptor["dtype"],
                "shape": list(descriptor["shape"]),
            }
        )
    return rows


def _static_residency_group(
    tensors: Mapping[str, Any], names: Iterable[str]
) -> dict[str, Any]:
    rows = _static_residency_descriptor_rows(tensors, names)
    return {
        "tensor_count": len(rows),
        "logical_source_bytes": sum(int(row["bytes"]) for row in rows),
        "manifest_descriptor_sha256": _sha256(_canonical(rows)),
    }


def _static_residency_source_config(
    artifact: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read source metadata only; do not create a model/runtime instance."""

    source = _static_residency_mapping(manifest.get("source"), "full manifest source")
    if source.get("repository") != REPOSITORY or source.get("revision") != REVISION:
        raise DeepSeekV4GravityError("full manifest does not bind the pinned DeepSeek-V4-Flash source")
    assets = _static_residency_mapping(source.get("metadata_assets"), "full manifest metadata assets")

    def asset(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        declared = _static_residency_mapping(assets.get(name), f"metadata asset {name}")
        digest = _static_residency_digest(declared.get("sha256"), f"metadata asset {name}.sha256")
        path = artifact / "metadata" / name
        _regular_file(path, f"full metadata asset {name}")
        raw = path.read_bytes()
        if _sha256(raw) != digest:
            raise DeepSeekV4GravityError(f"full metadata asset {name} differs from manifest source hash")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeepSeekV4GravityError(f"full metadata asset {name} is not JSON") from exc
        if not isinstance(value, dict):
            raise DeepSeekV4GravityError(f"full metadata asset {name} must be a JSON object")
        return value, {"path": str(path), "bytes": len(raw), "sha256": digest}

    config, config_binding = asset("config.json")
    inference, inference_binding = asset("inference/config.json")
    model_descriptor = _static_residency_mapping(assets.get("inference/model.py"), "model.py asset")
    model_digest = _static_residency_digest(model_descriptor.get("sha256"), "model.py source hash")
    model_path = artifact / "metadata" / "inference" / "model.py"
    _regular_file(model_path, "full inference model.py")
    if _sha256(model_path.read_bytes()) != model_digest:
        raise DeepSeekV4GravityError("full inference model.py differs from manifest source hash")
    return config, inference, {
        "config": config_binding,
        "inference_config": inference_binding,
        "inference_model_py_sha256": model_digest,
    }


def _static_residency_geometry(
    config: Mapping[str, Any], inference: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "hidden_size": _static_residency_integer(config.get("hidden_size"), "config.hidden_size", minimum=1),
        "num_hidden_layers": _static_residency_integer(
            config.get("num_hidden_layers"), "config.num_hidden_layers", minimum=1
        ),
        "n_routed_experts": _static_residency_integer(
            config.get("n_routed_experts"), "config.n_routed_experts", minimum=1
        ),
        "num_experts_per_tok": _static_residency_integer(
            config.get("num_experts_per_tok"), "config.num_experts_per_tok", minimum=1
        ),
        "n_shared_experts": _static_residency_integer(
            config.get("n_shared_experts"), "config.n_shared_experts", minimum=1
        ),
        "head_dim": _static_residency_integer(config.get("head_dim"), "config.head_dim", minimum=1),
        "sliding_window": _static_residency_integer(
            config.get("sliding_window"), "config.sliding_window", minimum=1
        ),
        "index_head_dim": _static_residency_integer(
            config.get("index_head_dim"), "config.index_head_dim", minimum=1
        ),
        "index_topk": _static_residency_integer(config.get("index_topk"), "config.index_topk", minimum=1),
        "hc_mult": _static_residency_integer(config.get("hc_mult"), "config.hc_mult", minimum=1),
        "vocab_size": _static_residency_integer(config.get("vocab_size"), "config.vocab_size", minimum=1),
        "num_hash_layers": _static_residency_integer(
            config.get("num_hash_layers"), "config.num_hash_layers", minimum=0
        ),
        "max_position_embeddings": _static_residency_integer(
            config.get("max_position_embeddings"), "config.max_position_embeddings", minimum=1
        ),
    }
    ratios = config.get("compress_ratios")
    if not isinstance(ratios, list) or len(ratios) < fields["num_hidden_layers"]:
        raise DeepSeekV4GravityError("config.compress_ratios does not cover every child-body layer")
    fields["compress_ratios"] = [
        _static_residency_integer(value, f"config.compress_ratios[{ordinal}]", minimum=0)
        for ordinal, value in enumerate(ratios[: fields["num_hidden_layers"]])
    ]
    if fields["num_experts_per_tok"] > fields["n_routed_experts"]:
        raise DeepSeekV4GravityError("source top-k exceeds its routed-expert count")
    if fields["n_shared_experts"] != 1:
        raise DeepSeekV4GravityError("static child-body analysis currently requires exactly one shared expert")
    if config.get("model_type") != "deepseek_v4" or config.get("expert_dtype") != "fp4":
        raise DeepSeekV4GravityError("config is not the expected DeepSeek-V4 FP4 MoE geometry")
    if inference.get("n_layers") != fields["num_hidden_layers"]:
        raise DeepSeekV4GravityError("inference config n_layers differs from full source config")
    for name in ("n_routed_experts", "n_activated_experts", "head_dim", "window_size", "hc_mult"):
        if name not in inference:
            raise DeepSeekV4GravityError(f"inference config lacks {name}")
    if int(inference["n_routed_experts"]) != fields["n_routed_experts"]:
        raise DeepSeekV4GravityError("inference config routed-expert count differs from full source config")
    if int(inference["n_activated_experts"]) != fields["num_experts_per_tok"]:
        raise DeepSeekV4GravityError("inference config activated-expert count differs from full source config")
    if int(inference["head_dim"]) != fields["head_dim"]:
        raise DeepSeekV4GravityError("inference config head dimension differs from full source config")
    if int(inference["window_size"]) != fields["sliding_window"]:
        raise DeepSeekV4GravityError("inference config window size differs from full source config")
    if int(inference["hc_mult"]) != fields["hc_mult"]:
        raise DeepSeekV4GravityError("inference config HC multiplicity differs from full source config")
    return fields


def _static_residency_kv_contract(geometry: Mapping[str, Any], *, layer: int) -> dict[str, Any]:
    """State-element contracts derived from source model.py, not an allocation probe."""

    ratio = int(geometry["compress_ratios"][layer])
    window = int(geometry["sliding_window"])
    head_dim = int(geometry["head_dim"])
    index_head_dim = int(geometry["index_head_dim"])
    index_topk = int(geometry["index_topk"])
    contexts: list[dict[str, Any]] = []
    for context in STATIC_RESIDENCY_CONTEXTS:
        main_slots = min(context, window) if ratio == 0 else window + context // ratio
        main_elements = main_slots * head_dim
        item: dict[str, Any] = {
            "context_tokens": context,
            "main_attention_kv_slots_per_batch": main_slots,
            "main_attention_kv_elements_per_batch": main_elements,
            "main_attention_kv_logical_bytes_if_bf16": main_elements * 2,
            "main_attention_kv_logical_bytes_if_fp32": main_elements * 4,
            "main_attention_decode_write_elements_per_token": head_dim,
            "main_attention_decode_write_logical_bytes_if_bf16": head_dim * 2,
        }
        if ratio:
            coff = 2 if ratio == 4 else 1
            compressor_elements_per_buffer = coff * ratio * coff * head_dim
            item["compressor"] = {
                "compress_ratio": ratio,
                "overlap": ratio == 4,
                "kv_state_and_score_state_dtype": "F32_EXPLICIT_IN_SOURCE",
                "elements_per_f32_buffer_per_batch": compressor_elements_per_buffer,
                "exact_bytes_per_f32_buffer_per_batch": compressor_elements_per_buffer * 4,
                "exact_combined_kv_and_score_state_bytes_per_batch": compressor_elements_per_buffer * 8,
            }
        if ratio == 4:
            index_slots = context // ratio
            index_elements = index_slots * index_head_dim
            selected = min(index_topk, index_slots)
            index_compressor_elements = 2 * ratio * 2 * index_head_dim
            item["indexer"] = {
                "index_kv_cache_slots_per_batch": index_slots,
                "index_kv_cache_elements_per_batch": index_elements,
                "index_kv_cache_logical_bytes_if_bf16": index_elements * 2,
                "index_kv_cache_logical_bytes_if_fp32": index_elements * 4,
                "index_compressor_kv_and_score_state_dtype": "F32_EXPLICIT_IN_SOURCE",
                "index_compressor_elements_per_f32_buffer_per_batch": index_compressor_elements,
                "index_compressor_exact_combined_state_bytes_per_batch": index_compressor_elements * 8,
                "selected_compressed_indices_per_decode_token": selected,
                "selected_compressed_indices_dtype": "I32_SOURCE_TOPK_CAST",
                "selected_compressed_indices_exact_bytes_per_decode_token": selected * 4,
            }
        contexts.append(item)
    return {
        "source_model_contract": "Attention.kv_cache plus optional Compressor/Indexer state; formula only",
        "compress_ratio": ratio,
        "main_attention_kv_storage_dtype": "RUNTIME_DEFAULT_NOT_PINNED_BY_SERIALIZED_ARTIFACT",
        "main_attention_kv_physical_resident_bytes": "NOT_MEASURED_NO_NATIVE_RUNTIME",
        "benchmark_context_contracts_batch_1": contexts,
    }


def _static_residency_layer(
    tensors: Mapping[str, Any], geometry: Mapping[str, Any], *, layer: int
) -> dict[str, Any]:
    prefix = f"layers.{layer}."
    layer_names = sorted(name for name in tensors if name.startswith(prefix))
    if not layer_names:
        raise DeepSeekV4GravityError(f"full stream lacks child-body layer {layer}")
    routed_prefix = prefix + "ffn.experts."
    shared_prefix = prefix + "ffn.shared_experts."
    gate_prefix = prefix + "ffn.gate."
    expected_expert_suffixes = {
        "w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale"
    }
    experts: dict[int, set[str]] = {}
    routed_names: list[str] = []
    shared_names: list[str] = []
    router_names: list[str] = []
    mhc_names: list[str] = []
    norm_names: list[str] = []
    attention_core_names: list[str] = []
    compressor_names: list[str] = []
    indexer_names: list[str] = []
    unclassified: list[str] = []
    for name in layer_names:
        rest = name[len(prefix) :]
        if name.startswith(routed_prefix):
            pieces = rest.split(".")
            if len(pieces) != 5 or pieces[:2] != ["ffn", "experts"] or not pieces[2].isdecimal():
                raise DeepSeekV4GravityError(f"invalid routed-expert tensor name {name}")
            expert = int(pieces[2])
            suffix = ".".join(pieces[3:])
            if expert < 0 or expert >= int(geometry["n_routed_experts"]) or suffix not in expected_expert_suffixes:
                raise DeepSeekV4GravityError(f"unexpected routed-expert tensor {name}")
            experts.setdefault(expert, set()).add(suffix)
            routed_names.append(name)
        elif name.startswith(shared_prefix):
            suffix = name[len(shared_prefix) :]
            if suffix not in expected_expert_suffixes:
                raise DeepSeekV4GravityError(f"unexpected shared-expert tensor {name}")
            shared_names.append(name)
        elif name.startswith(gate_prefix):
            router_names.append(name)
        elif rest in {"attn_norm.weight", "ffn_norm.weight"}:
            norm_names.append(name)
        elif rest.startswith("hc_attn_") or rest.startswith("hc_ffn_"):
            mhc_names.append(name)
        elif rest.startswith("attn.indexer."):
            indexer_names.append(name)
        elif rest.startswith("attn.compressor."):
            compressor_names.append(name)
        elif rest.startswith("attn."):
            attention_core_names.append(name)
        else:
            unclassified.append(name)
    if unclassified:
        raise DeepSeekV4GravityError(
            f"unclassified child-body layer {layer} tensors: {', '.join(unclassified[:4])}"
        )
    expected_ids = set(range(int(geometry["n_routed_experts"])))
    if set(experts) != expected_ids or any(suffixes != expected_expert_suffixes for suffixes in experts.values()):
        raise DeepSeekV4GravityError(f"layer {layer} does not contain all complete routed expert bundles")
    if {name[len(shared_prefix) :] for name in shared_names} != expected_expert_suffixes:
        raise DeepSeekV4GravityError(f"layer {layer} does not contain the complete shared expert")
    router_kind = "hash" if layer < int(geometry["num_hash_layers"]) else "score"
    expected_router_suffixes = {"weight", "tid2eid"} if router_kind == "hash" else {"weight", "bias"}
    if {name[len(gate_prefix) :] for name in router_names} != expected_router_suffixes:
        raise DeepSeekV4GravityError(f"layer {layer} router tensors do not match the declared {router_kind} route mode")
    expected_mhc_suffixes = {
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
    }
    if {name[len(prefix) :] for name in mhc_names} != expected_mhc_suffixes:
        raise DeepSeekV4GravityError(f"layer {layer} is missing an mHC control tensor")
    if {name[len(prefix) :] for name in norm_names} != {"attn_norm.weight", "ffn_norm.weight"}:
        raise DeepSeekV4GravityError(f"layer {layer} is missing an RMSNorm tensor")
    required_attention = {
        "attn.attn_sink",
        "attn.kv_norm.weight",
        "attn.q_norm.weight",
        "attn.wq_a.weight",
        "attn.wq_a.scale",
        "attn.wq_b.weight",
        "attn.wq_b.scale",
        "attn.wkv.weight",
        "attn.wkv.scale",
        "attn.wo_a.weight",
        "attn.wo_a.scale",
        "attn.wo_b.weight",
        "attn.wo_b.scale",
    }
    names_without_prefix = {name[len(prefix) :] for name in attention_core_names}
    if not required_attention.issubset(names_without_prefix):
        raise DeepSeekV4GravityError(f"layer {layer} lacks a complete base MLA attention tensor set")
    ratio = int(geometry["compress_ratios"][layer])
    if ratio == 0 and (compressor_names or indexer_names):
        raise DeepSeekV4GravityError(f"layer {layer} has compression tensors with a zero compression ratio")
    if ratio > 0 and not compressor_names:
        raise DeepSeekV4GravityError(f"layer {layer} is missing its declared attention compressor")
    if (ratio == 4) != bool(indexer_names):
        raise DeepSeekV4GravityError(f"layer {layer} indexer tensor presence disagrees with its compression ratio")

    routed_by_expert: dict[int, int] = {}
    for expert in sorted(experts):
        names = [f"{routed_prefix}{expert}.{suffix}" for suffix in sorted(expected_expert_suffixes)]
        routed_by_expert[expert] = int(_static_residency_group(tensors, names)["logical_source_bytes"])
    unique_expert_bytes = sorted(set(routed_by_expert.values()))
    topk = int(geometry["num_experts_per_tok"])
    exact_topk_bytes: int | None = None
    if len(unique_expert_bytes) == 1:
        exact_topk_bytes = unique_expert_bytes[0] * topk
    groups = {
        "routed_expert_bank": _static_residency_group(tensors, routed_names),
        "shared_expert": _static_residency_group(tensors, shared_names),
        "router": _static_residency_group(tensors, router_names),
        "mHC_control": _static_residency_group(tensors, mhc_names),
        "norm": _static_residency_group(tensors, norm_names),
        "attention_core": _static_residency_group(tensors, attention_core_names),
        "compressed_attention": _static_residency_group(tensors, compressor_names),
        "index_heads": _static_residency_group(tensors, indexer_names),
        "full_layer": _static_residency_group(tensors, layer_names),
    }
    attention_total = (
        int(groups["attention_core"]["logical_source_bytes"])
        + int(groups["compressed_attention"]["logical_source_bytes"])
        + int(groups["index_heads"]["logical_source_bytes"])
    )
    control_total = (
        int(groups["router"]["logical_source_bytes"])
        + int(groups["mHC_control"]["logical_source_bytes"])
        + int(groups["norm"]["logical_source_bytes"])
    )
    router_active_bytes = int(groups["router"]["logical_source_bytes"])
    hash_route_row_bytes: int | None = None
    if router_kind == "hash":
        route_table = _static_residency_descriptor(tensors, f"{gate_prefix}tid2eid")
        rows = _static_residency_integer(route_table["shape"][0], f"layer {layer} hash route rows", minimum=1)
        hash_route_row_bytes, remainder = divmod(int(route_table["bytes"]), rows)
        if remainder:
            raise DeepSeekV4GravityError(f"layer {layer} hash route table has non-row-aligned source bytes")
        # Hash routing looks up the current source-token row, so the complete
        # 6.2 MiB table is a static residency candidate but not a one-token
        # selected-byte requirement.
        router_active_bytes = int(_static_residency_descriptor(tensors, f"{gate_prefix}weight")["bytes"]) + hash_route_row_bytes
    control_active_total = (
        router_active_bytes
        + int(groups["mHC_control"]["logical_source_bytes"])
        + int(groups["norm"]["logical_source_bytes"])
    )
    active_weights = None
    if exact_topk_bytes is not None:
        active_weights = (
            exact_topk_bytes
            + int(groups["shared_expert"]["logical_source_bytes"])
            + attention_total
            + control_active_total
        )
    return {
        "layer": layer,
        "router_mode": router_kind,
        "compress_ratio": ratio,
        "tensor_counts": {
            "full_layer": len(layer_names),
            "routed_expert_tensors": len(routed_names),
            "shared_expert_tensors": len(shared_names),
        },
        "logical_source_tensor_groups": groups,
        "routed_expert_activation_contract": {
            "routed_expert_count": len(routed_by_expert),
            "top_k_selected_per_token": topk,
            "per_expert_bundle_logical_bytes_min": min(unique_expert_bytes),
            "per_expert_bundle_logical_bytes_max": max(unique_expert_bytes),
            "all_expert_bundle_bytes_equal": len(unique_expert_bytes) == 1,
            "per_expert_bundle_logical_bytes_when_uniform": (
                unique_expert_bytes[0] if len(unique_expert_bytes) == 1 else None
            ),
            "top_k_routed_expert_logical_bytes": exact_topk_bytes,
            "status": (
                "EXACT_STATIC_BYTES_FOR_ANY_VALID_TOP6_SET"
                if exact_topk_bytes is not None
                else "ROUTE_DEPENDENT_BYTES_REQUIRE_SELECTED_EXPERT_IDS"
            ),
        },
        "always_selected_shared_expert_logical_bytes": int(groups["shared_expert"]["logical_source_bytes"]),
        "attention_and_control_logical_bytes": {
            "attention_total": attention_total,
            "control_static_tensor_total": control_total,
            "control_selected_logical_bytes_per_decode_token": control_active_total,
            "router_static_tensor_bytes": int(groups["router"]["logical_source_bytes"]),
            "router_selected_logical_bytes_per_decode_token": router_active_bytes,
            "mHC": int(groups["mHC_control"]["logical_source_bytes"]),
            "norm": int(groups["norm"]["logical_source_bytes"]),
        },
        "static_selected_weight_logical_bytes_per_decode_token": active_weights,
        "static_selected_weight_interpretation": (
            "manifest-exact selected tensor bytes if all layer weights are fetched for one token; "
            "not a physical read counter, cache-hit measurement, or bandwidth claim"
        ),
        "kv_and_index_state_contract": _static_residency_kv_contract(geometry, layer=layer),
        "mhc_activation_state_contract": {
            "per_decode_token_shape": ["batch", 1, int(geometry["hc_mult"]), int(geometry["hidden_size"])],
            "elements_per_batch": int(geometry["hc_mult"]) * int(geometry["hidden_size"]),
            "logical_bytes_if_bf16": int(geometry["hc_mult"]) * int(geometry["hidden_size"]) * 2,
            "logical_bytes_if_fp32": int(geometry["hc_mult"]) * int(geometry["hidden_size"]) * 4,
            "physical_residency": "NOT_MEASURED_NO_NATIVE_RUNTIME",
        },
        "router_state_contract": {
            "score_vector_elements_per_token": int(geometry["n_routed_experts"]),
            "score_vector_bytes_f32": int(geometry["n_routed_experts"]) * 4,
            "selected_route_count": topk,
            "selected_route_ids_bytes_if_i64": topk * 8,
            "route_weights_bytes_f32": topk * 4,
            "hash_route_table_present": router_kind == "hash",
            "hash_route_table_selected_row_logical_source_bytes": hash_route_row_bytes,
            "physical_read_write_counters": "NOT_MEASURED_NO_NATIVE_RUNTIME",
        },
    }


def _static_residency_layer4_observation(
    receipt_path: str | Path | None,
    *,
    manifest: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Import only bounded, privacy-safe route aggregates from a sealed receipt."""

    if receipt_path is None:
        return {
            "availability": "NOT_PROVIDED",
            "expert_frequency": "NOT_MEASURED",
            "transition_observations": "NOT_MEASURED",
        }
    path = _absolute(receipt_path, "layer4_latent_route_receipt")
    _regular_file(path, "layer-4 latent route receipt")
    if path.stat().st_size > MAX_STATIC_RESIDENCY_RECEIPT_BYTES:
        raise DeepSeekV4GravityError("layer-4 latent route receipt exceeds the bounded input size")
    document = _read_json(path, "layer-4 latent route receipt")
    try:
        document = verify(document, label="layer-4 latent route receipt")
    except SealIntegrityError as exc:
        raise DeepSeekV4GravityError(str(exc)) from exc
    if document.get("schema") != "hawking.gravity.deepseek_v4.diagnostic_latent_route_receipt.v1":
        raise DeepSeekV4GravityError("layer-4 route receipt has an unexpected schema")
    if document.get("status") != "SEALED_BOUNDED_LAYER4_CPU_DIAGNOSTIC_LATENT_ROUTE_CAPTURE":
        raise DeepSeekV4GravityError("layer-4 route receipt has an unexpected status")
    artifact = _static_residency_mapping(document.get("artifact"), "latent receipt artifact")
    scope = _static_residency_mapping(artifact.get("diagnostic_scope"), "latent receipt diagnostic scope")
    if scope.get("selected_layer") != 4 or scope.get("not_full_model") is not True:
        raise DeepSeekV4GravityError("latent route receipt is not bound to the layer-4 diagnostic")
    source_binding = _static_residency_mapping(
        document.get("source_hash_binding"), "latent receipt source hash binding"
    )
    if source_binding.get("repository") != REPOSITORY or source_binding.get("revision") != REVISION:
        raise DeepSeekV4GravityError("latent route receipt is not bound to the pinned DeepSeek source")
    if source_binding.get("source_parent_retained") is not False:
        raise DeepSeekV4GravityError("latent route receipt has an invalid source retention claim")
    source = _static_residency_mapping(manifest.get("source"), "full manifest source")
    assets = _static_residency_mapping(source.get("metadata_assets"), "full manifest metadata assets")
    receipt_assets = _static_residency_mapping(
        source_binding.get("metadata_asset_sha256"), "latent receipt metadata hash binding"
    )
    for name, digest in receipt_assets.items():
        if not isinstance(name, str):
            raise DeepSeekV4GravityError("latent route receipt has an invalid metadata asset name")
        expected = _static_residency_digest(digest, f"latent receipt {name} source hash")
        full_asset = _static_residency_mapping(assets.get(name), f"full manifest metadata asset {name}")
        if full_asset.get("sha256") != expected:
            raise DeepSeekV4GravityError("latent route receipt source hashes do not match the full stream")
    capture = _static_residency_mapping(document.get("capture"), "latent receipt capture")
    limits = _static_residency_mapping(capture.get("collection_limits"), "latent receipt collection limits")
    if any(limits.get(field) is not False for field in ("raw_prompts_retained", "raw_completions_retained", "raw_hidden_states_retained")):
        raise DeepSeekV4GravityError("latent route receipt violates bounded privacy retention policy")
    aggregate = _static_residency_mapping(capture.get("route_aggregate"), "latent receipt route aggregate")
    forwards = _static_residency_integer(aggregate.get("total_source_forwards"), "latent route forward count", minimum=1)
    raw_frequency = _static_residency_mapping(aggregate.get("expert_frequency"), "latent expert frequency")
    frequency: dict[str, int] = {}
    for raw_expert, raw_count in raw_frequency.items():
        if not isinstance(raw_expert, str) or not raw_expert.isdecimal():
            raise DeepSeekV4GravityError("latent expert frequency contains an invalid expert ID")
        expert = int(raw_expert)
        if expert >= int(geometry["n_routed_experts"]):
            raise DeepSeekV4GravityError("latent expert frequency names an expert outside source geometry")
        frequency[str(expert)] = _static_residency_integer(
            raw_count, f"latent expert {expert} frequency", minimum=1
        )
    if not frequency or len(frequency) > int(geometry["n_routed_experts"]):
        raise DeepSeekV4GravityError("latent expert frequency is empty or exceeds source geometry")
    if sum(frequency.values()) != forwards * int(geometry["num_experts_per_tok"]):
        raise DeepSeekV4GravityError("latent expert frequency does not reconcile with forward/top-k count")
    route_sets = _static_residency_mapping(
        aggregate.get("route_set_frequency"), "latent route set frequency"
    )
    transitions = _static_residency_mapping(
        aggregate.get("route_set_transition_frequency"), "latent route transition frequency"
    )
    for raw_digest, raw_count in route_sets.items():
        _static_residency_digest(raw_digest, "latent route-set digest")
        _static_residency_integer(raw_count, "latent route-set frequency", minimum=1)
    for raw_transition, raw_count in transitions.items():
        if not isinstance(raw_transition, str) or raw_transition.count(":") != 1:
            raise DeepSeekV4GravityError("latent route transition is not a digest pair")
        left, right = raw_transition.split(":")
        _static_residency_digest(left, "latent route transition source digest")
        _static_residency_digest(right, "latent route transition destination digest")
        _static_residency_integer(raw_count, "latent route transition frequency", minimum=1)
    if aggregate.get("raw_route_sequence_retained") is not False:
        raise DeepSeekV4GravityError("latent route receipt retains an unsupported raw route sequence")
    membership = _static_residency_mapping(capture.get("membership_partition"), "latent membership partition")
    if membership.get("excluded_from") != ["fit", "calibration", "public_test", "hidden_test"]:
        raise DeepSeekV4GravityError("latent route receipt does not maintain the required disjoint membership")
    ranked = sorted(((int(expert), count) for expert, count in frequency.items()), key=lambda row: (-row[1], row[0]))
    return {
        "availability": "SEALED_BOUNDED_LAYER4_CPU_DIAGNOSTIC_ROUTE_OBSERVATIONS",
        "receipt": {
            "path": str(path),
            "file_sha256": _sha256(path.read_bytes()),
            "seal_sha256": document["seal_sha256"],
            "diagnostic_artifact_seal_sha256": artifact.get("seal_sha256"),
        },
        "scope": {
            "selected_layer": 4,
            "total_source_forwards": forwards,
            "source_top_k": int(geometry["num_experts_per_tok"]),
            "diagnostic_only": True,
            "not_a_full_model_route_distribution": True,
            "raw_prompts_completions_or_hidden_states_retained": False,
            "raw_route_sequence_retained": False,
        },
        "expert_frequency": {key: frequency[key] for key in sorted(frequency, key=lambda item: int(item))},
        "frequency_ranked_experts": [
            {"expert_id": expert, "selection_count": count} for expert, count in ranked
        ],
        "transition_observations": {
            "route_set_representation": "SHA256_DIGEST_ONLY",
            "distinct_route_set_count": len(route_sets),
            "total_route_set_observations": sum(int(count) for count in route_sets.values()),
            "distinct_adjacent_transition_count": len(transitions),
            "total_adjacent_transition_observations": sum(int(count) for count in transitions.values()),
            "transition_probability": "NOT_ESTIMATED_FOR_RUNTIME_PREFETCH",
        },
    }


def _static_residency_global_decode_contract(
    tensors: Mapping[str, Any], geometry: Mapping[str, Any]
) -> dict[str, Any]:
    global_names = [
        "embed.weight",
        "norm.weight",
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
        "head.weight",
    ]
    for name in global_names:
        _static_residency_descriptor(tensors, name)
    embedding = _static_residency_descriptor(tensors, "embed.weight")
    if embedding["shape"] != [int(geometry["vocab_size"]), int(geometry["hidden_size"])]:
        raise DeepSeekV4GravityError("embedding shape differs from source geometry")
    embedding_row_bytes, remainder = divmod(int(embedding["bytes"]), int(geometry["vocab_size"]))
    if remainder:
        raise DeepSeekV4GravityError("embedding source bytes do not divide into complete vocabulary rows")
    head = _static_residency_descriptor(tensors, "head.weight")
    if head["shape"] != [int(geometry["vocab_size"]), int(geometry["hidden_size"])]:
        raise DeepSeekV4GravityError("lm_head shape differs from source geometry")
    mtp_names = sorted(name for name in tensors if name.startswith("mtp."))
    if not mtp_names:
        raise DeepSeekV4GravityError("full stream lacks its recorded MTP tensors")
    permitted = set(global_names) | set(mtp_names)
    non_layer = {name for name in tensors if not name.startswith("layers.")}
    unexpected = sorted(non_layer.difference(permitted))
    if unexpected:
        raise DeepSeekV4GravityError(
            "full stream has an unaccounted non-child tensor: " + ", ".join(unexpected[:4])
        )
    return {
        "standard_transformer_decode": {
            "embedding": {
                "full_table_logical_source_bytes": int(embedding["bytes"]),
                "one_selected_token_row_logical_source_bytes": embedding_row_bytes,
            },
            "final_norm_and_hc_head": _static_residency_group(
                tensors, ("norm.weight", "hc_head_base", "hc_head_fn", "hc_head_scale")
            ),
            "lm_head": _static_residency_group(tensors, ("head.weight",)),
            "lm_head_execution_contract": (
                "dense vocabulary projection requires a native kernel before physical bytes, cache behavior, "
                "or throughput can be measured"
            ),
        },
        "mtp_tensors": {
            **_static_residency_group(tensors, mtp_names),
            "base_decode_inclusion": "EXCLUDED_FROM_STANDARD_TRANSFORMER_FORWARD_AND_BASE_TRUE_TPS",
            "reason": "The pinned source Transformer.forward iterates layers then emits its main head; MTP is separate.",
        },
    }


def _static_residency_output_parent(path: Path) -> Path:
    parent = path.parent
    while not parent.exists():
        if parent == parent.parent:
            raise DeepSeekV4GravityError("cannot find an existing parent for static residency output")
        parent = parent.parent
    if not parent.is_dir():
        raise DeepSeekV4GravityError("static residency output parent is not a directory")
    return parent


def static_expert_residency_report(
    full_artifact_dir: str | Path,
    *,
    out: str | Path,
    layer4_latent_route_receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Seal a manifest-only active-byte/residency contract for the child body.

    This deliberately does *not* instantiate the diagnostic runtime, read any
    tensor payload, submit a GPU command buffer, or measure throughput.  Its
    purpose is to provide exact source-layout byte contracts ahead of native
    routing/cache implementation.
    """

    artifact = _absolute(full_artifact_dir, "full_artifact_dir")
    target = _absolute(out, "out")
    try:
        target.resolve().relative_to(artifact.resolve())
    except ValueError:
        pass
    else:
        raise DeepSeekV4GravityError("static residency output must be external to the sealed full artifact")
    manifest = load_full_manifest(artifact)
    tensors = _static_residency_mapping(manifest.get("tensors"), "full manifest tensors")
    if len(tensors) != FULL_EXPECTED_TENSOR_COUNT:
        raise DeepSeekV4GravityError("static residency analysis requires the complete sealed full tensor map")
    config, inference, source_metadata = _static_residency_source_config(artifact, manifest)
    geometry = _static_residency_geometry(config, inference)
    layers = [
        _static_residency_layer(tensors, geometry, layer=layer)
        for layer in range(int(geometry["num_hidden_layers"]))
    ]
    global_decode = _static_residency_global_decode_contract(tensors, geometry)
    layer4_observation = _static_residency_layer4_observation(
        layer4_latent_route_receipt, manifest=manifest, geometry=geometry
    )
    full_tensor_bytes = sum(
        int(_static_residency_descriptor(tensors, name)["bytes"]) for name in sorted(tensors)
    )
    artifact_summary = _static_residency_mapping(manifest.get("artifact"), "full manifest artifact")
    if full_tensor_bytes != _static_residency_integer(
        artifact_summary.get("total_tensor_bytes"), "full manifest artifact.total_tensor_bytes", minimum=1
    ):
        raise DeepSeekV4GravityError("full tensor descriptor bytes do not reconcile with sealed artifact bytes")
    active_layer_weight_bytes = [layer["static_selected_weight_logical_bytes_per_decode_token"] for layer in layers]
    if any(value is None for value in active_layer_weight_bytes):
        raise DeepSeekV4GravityError("source has nonuniform expert bundles; selected top-k byte contract is not exact")
    total_active_body_weights = sum(int(value) for value in active_layer_weight_bytes if value is not None)
    shared_and_control_resident = sum(
        int(layer["always_selected_shared_expert_logical_bytes"])
        + int(layer["attention_and_control_logical_bytes"]["attention_total"])
        + int(layer["attention_and_control_logical_bytes"]["control_static_tensor_total"])
        for layer in layers
    )
    layer4_bundle = layers[4]["routed_expert_activation_contract"]["per_expert_bundle_logical_bytes_when_uniform"]
    assert isinstance(layer4_bundle, int)
    candidate_hot_banks: list[dict[str, Any]] = []
    if isinstance(layer4_observation.get("frequency_ranked_experts"), list):
        ranked = layer4_observation["frequency_ranked_experts"]
        for capacity in (6, 16, 32):
            selected = ranked[: min(capacity, len(ranked))]
            candidate_hot_banks.append(
                {
                    "layer": 4,
                    "expert_capacity": capacity,
                    "observed_expert_ids": [row["expert_id"] for row in selected],
                    "static_bundle_bytes": len(selected) * layer4_bundle,
                    "selection_source": "bounded_layer4_diagnostic_frequency_only",
                    "status": "CANDIDATE_ONLY_NO_CACHE_OR_PREFETCH_MEASUREMENT",
                }
            )
    storage = _static_residency_mapping(manifest.get("storage"), "full manifest storage")
    runtime_adapter = _static_residency_mapping(manifest.get("runtime_adapter"), "full manifest runtime adapter")
    source = _static_residency_mapping(manifest.get("source"), "full manifest source")
    report = seal(
        {
            "schema": STATIC_EXPERT_RESIDENCY_SCHEMA,
            "status": "SEALED_STATIC_FULL_STREAM_EXPERT_RESIDENCY_CONTRACT_RUNTIME_PENDING",
            "created_at": _utc_now(),
            "analysis_mode": {
                "manifest_and_metadata_only": True,
                "tensor_payload_bytes_read": 0,
                "runtime_forward_executed": False,
                "gpu_dispatches": 0,
                "command_buffers": 0,
                "base_true_tps": False,
                "training_or_distillation": False,
            },
            "source_binding": {
                "repository": source.get("repository"),
                "revision": source.get("revision"),
                "full_manifest_path": str(artifact / "manifest.json"),
                "full_manifest_seal_sha256": manifest["seal_sha256"],
                "full_stream_tensor_count": len(tensors),
                "full_stream_logical_tensor_bytes": full_tensor_bytes,
                "metadata": source_metadata,
            },
            "geometry": {
                **geometry,
                "benchmark_contexts": list(STATIC_RESIDENCY_CONTEXTS),
            },
            "per_layer": layers,
            "global_decode_contract": global_decode,
            "static_active_byte_summary": {
                "body_selected_weight_logical_bytes_per_decode_token": total_active_body_weights,
                "body_selected_weight_logical_bytes_interpretation": (
                    "sum of manifest-exact layer selections: one top-6 routed bundle, shared expert, "
                    "attention, router, mHC and norms for every main child-body layer; this is not a "
                    "measured physical byte count"
                ),
                "always_selected_shared_attention_and_control_logical_bytes": shared_and_control_resident,
                "dense_lm_head_logical_source_bytes_per_decode_token_without_residency": int(
                    global_decode["standard_transformer_decode"]["lm_head"]["logical_source_bytes"]
                ),
                "one_embedding_row_logical_source_bytes": int(
                    global_decode["standard_transformer_decode"]["embedding"]["one_selected_token_row_logical_source_bytes"]
                ),
                "physical_active_bytes_per_token": "NOT_MEASURED_NO_NATIVE_RUNTIME",
            },
            "static_packing_and_cache_candidates": {
                "per_layer_atomic_routed_expert_bundle": {
                    "layout": "{layer, expert}/[w1.weight,w1.scale,w3.weight,w3.scale,w2.weight,w2.scale]",
                    "bundle_bytes": layer4_bundle,
                    "top6_bundle_bytes_per_layer": layer4_bundle * int(geometry["num_experts_per_tok"]),
                    "status": "CANDIDATE_ONLY_SOURCE_CHUNKS_NOT_REPACKED",
                },
                "resident_shared_control_and_attention": {
                    "logical_source_bytes": shared_and_control_resident,
                    "policy": "shared experts plus fixed attention/router/mHC/norm tensors may be resident; routing remains dynamic",
                    "status": "CANDIDATE_ONLY_NOT_ALLOCATED_OR_MEASURED",
                },
                "frequency_ranked_layer4_hot_banks": candidate_hot_banks,
                "bounded_cold_tier": {
                    "unit": "one exact routed-expert bundle",
                    "logical_source_bytes_per_unit": layer4_bundle,
                    "eviction": "runtime-policy-required; no actual cache exists in the current full stream",
                    "status": "CANDIDATE_ONLY",
                },
                "route_predictive_and_layer_aware_prefetch": {
                    "layer4_transition_input": layer4_observation.get("transition_observations"),
                    "implementation": "NOT_IMPLEMENTED",
                    "accuracy": "NOT_MEASURED",
                    "cold_miss_latency": "NOT_MEASURED",
                    "promotion_requirement": "real 43-layer native runtime with cache event and physical-transfer counters",
                },
            },
            "bounded_layer4_route_observations": layer4_observation,
            "unavailable_until_native_runtime": {
                "per_layer_full_model_expert_frequency": "NOT_MEASURED",
                "route_transition_probabilities": "NOT_MEASURED",
                "hot_expert_cache_hit_rate": "NOT_MEASURED",
                "cold_expert_cache_miss_latency": "NOT_MEASURED",
                "next_layer_prefetch_accuracy": "NOT_MEASURED",
                "physical_bytes_read": "NOT_MEASURED",
                "physical_bytes_written": "NOT_MEASURED",
                "effective_bandwidth": "NOT_MEASURED",
                "gpu_duration": "NOT_MEASURED",
                "dispatches_command_buffers_or_waits": "NOT_MEASURED",
            },
            "current_runtime_boundary": {
                "full_runtime_adapter": dict(runtime_adapter),
                "source_parent_retained": storage.get("source_parent_retained"),
                "raw_content_addressed_stream_eviction_authorized": False,
                "claim": "This sealed static contract neither loads nor promotes a 43-layer runtime.",
            },
        }
    )
    parent = _static_residency_output_parent(target)
    _floor_check(parent, MIN_FREE_FLOOR_BYTES, MAX_STATIC_RESIDENCY_RECEIPT_BYTES, "static expert residency receipt")
    _atomic_json(target, report)
    return report


def load_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    artifact = _absolute(artifact_dir, "artifact_dir")
    manifest = _read_json(artifact / "manifest.json", "V4 diagnostic manifest")
    try:
        verified = verify(manifest, label="DeepSeek-V4 diagnostic manifest")
    except SealIntegrityError as exc:
        raise DeepSeekV4GravityError(str(exc)) from exc
    if verified.get("schema") != ARTIFACT_SCHEMA:
        raise DeepSeekV4GravityError("artifact manifest is not a DeepSeek-V4 diagnostic")
    if verified.get("status") != "DIAGNOSTIC_SEALED_LOADABLE_BY_V4_NUMPY_ADAPTER":
        raise DeepSeekV4GravityError("artifact is not sealed loadable diagnostic state")
    return verified


def inspect_artifact(artifact_dir: str | Path) -> dict[str, Any]:
    artifact = _absolute(artifact_dir, "artifact_dir")
    manifest = load_manifest(artifact)
    tensors = manifest.get("tensors")
    if not isinstance(tensors, Mapping):
        raise DeepSeekV4GravityError("sealed manifest lacks tensor mapping")
    store = SegmentedStore(artifact, tensors)
    checked = 0
    for name in ("embed.weight", "layers.4.attn.wq_a.weight", "layers.4.ffn.experts.0.w1.weight", "head.weight"):
        tensor = store.descriptor(name)
        raw = store.read(name, 0, min(int(tensor["bytes"]), 64))
        checked += len(raw)
    restart = _read_json(artifact / "restart-receipt.json", "restart receipt")
    try:
        verify(restart, label="DeepSeek-V4 restart receipt")
    except SealIntegrityError as exc:
        raise DeepSeekV4GravityError(str(exc)) from exc
    manifest_restart = manifest.get("restart_receipt")
    if not isinstance(manifest_restart, Mapping):
        raise DeepSeekV4GravityError("sealed manifest lacks restart receipt binding")
    if manifest_restart.get("path") != "restart-receipt.json":
        raise DeepSeekV4GravityError("sealed manifest restart receipt path is not canonical")
    if manifest_restart.get("seal_sha256") != restart.get("seal_sha256"):
        raise DeepSeekV4GravityError("restart receipt seal does not match sealed manifest binding")
    paths = _artifact_paths(artifact)
    journal_sha256 = _sha256(paths["journal"].read_bytes())
    ranges_sha256 = _sha256(paths["ranges"].read_bytes())
    if restart.get("journal_sha256") != journal_sha256:
        raise DeepSeekV4GravityError("restart receipt journal hash does not match artifact journal")
    if restart.get("range_journal_sha256") != ranges_sha256:
        raise DeepSeekV4GravityError("restart receipt range journal hash does not match artifact ranges")
    return {
        "schema": "hawking.gravity.deepseek_v4.inspect.v1",
        "status": "SAMPLED_DIAGNOSTIC_INTEGRITY_VERIFIED",
        "artifact_dir": str(artifact),
        "artifact_seal_sha256": manifest["seal_sha256"],
        "restart_seal_sha256": restart["seal_sha256"],
        "tensor_count": len(tensors),
        "sampled_chunk_bytes_verified": checked,
        "runtime_adapter": manifest["runtime_adapter"],
        "diagnostic_scope": manifest["diagnostic_scope"],
    }


def reverify_sealed_diagnostic(artifact_dir: str | Path) -> dict[str, Any]:
    """Full local integrity scan and adapter load for an existing sealed artifact.

    This is deliberately stronger than ``inspect``: it hashes every retained
    content-addressed source-derived chunk, verifies the manifest-to-restart
    journal binding, then constructs the runtime adapter.  It never opens the
    Xet transport or rehydrates any source parent bytes.
    """

    started = time.perf_counter()
    artifact = _absolute(artifact_dir, "artifact_dir")
    manifest = load_manifest(artifact)
    tensors = manifest.get("tensors")
    if not isinstance(tensors, Mapping):
        raise DeepSeekV4GravityError("sealed manifest lacks tensor mapping")
    # Reuse the sampled verifier for manifest/restart/journal binding before
    # scanning all content.  Its output is retained as a clearly scoped
    # subsidiary receipt, never mislabeled as the full verification itself.
    sampled = inspect_artifact(artifact)
    expected_chunks: dict[str, str] = {}
    expected_bytes: dict[str, int] = {}
    for name, descriptor in tensors.items():
        if not isinstance(name, str) or not isinstance(descriptor, Mapping):
            raise DeepSeekV4GravityError("sealed tensor mapping contains an invalid descriptor")
        segments = descriptor.get("segments")
        if not isinstance(segments, list) or not segments:
            raise DeepSeekV4GravityError(f"sealed tensor {name} has no chunk segments")
        for segment in segments:
            if not isinstance(segment, Mapping):
                raise DeepSeekV4GravityError(f"sealed tensor {name} has an invalid chunk segment")
            relative = segment.get("chunk_relpath")
            digest = segment.get("sha256")
            size = segment.get("bytes")
            if not isinstance(relative, str) or not isinstance(digest, str) or not isinstance(size, int):
                raise DeepSeekV4GravityError(f"sealed tensor {name} has an incomplete chunk segment")
            previous = expected_chunks.setdefault(relative, digest)
            if previous != digest:
                raise DeepSeekV4GravityError(f"chunk {relative} has conflicting digest bindings")
            previous_size = expected_bytes.setdefault(relative, size)
            if previous_size != size:
                raise DeepSeekV4GravityError(f"chunk {relative} has conflicting byte bindings")
    checked_bytes = 0
    for relative, digest in sorted(expected_chunks.items()):
        path = artifact / relative
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(artifact.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise DeepSeekV4GravityError(f"artifact chunk path escapes artifact directory: {relative}") from exc
        if path.stat().st_size != expected_bytes[relative]:
            raise DeepSeekV4GravityError(f"artifact chunk byte size differs for {relative}")
        observed = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(DEFAULT_RANGE_BYTES):
                observed.update(block)
        if observed.hexdigest() != digest:
            raise DeepSeekV4GravityError(f"artifact chunk hash differs for {relative}")
        checked_bytes += expected_bytes[relative]
    runtime = DeepSeekV4DiagnosticRuntime(artifact)
    return seal(
        {
            "schema": "hawking.gravity.deepseek_v4.execute_reverify.v1",
            "status": "DIAGNOSTIC_FULLY_REVERIFIED_RUNTIME_LOADED",
            "artifact_dir": str(artifact),
            "artifact_seal_sha256": manifest["seal_sha256"],
            "source_parent_retained": False,
            "source_transport": {
                "activated": False,
                "reason": "sealed artifact re-verification needs no source ranges",
            },
            "sampled_inspection": sampled,
            "full_chunk_verification": {
                "chunk_count": len(expected_chunks),
                "bytes_verified": checked_bytes,
                "sha256_verified": True,
            },
            "runtime_adapter_load": {
                "id": runtime.manifest["runtime_adapter"]["id"],
                "device": runtime.manifest["runtime_adapter"]["device"],
                "load_ms": runtime.load_ms,
                "status": "LOADED_FOR_RESTART_REVERIFY_ONLY",
            },
            "elapsed_ms": (time.perf_counter() - started) * 1_000.0,
            "resume_safe": True,
        }
    )


def tps_gate_report(
    artifact_dir: str | Path, benchmark_receipt: str | Path, out: str | Path
) -> dict[str, Any]:
    """Seal the explicit reason a diagnostic CPU measurement is not Base True TPS."""

    artifact = _absolute(artifact_dir, "artifact_dir")
    manifest = load_manifest(artifact)
    benchmark_path = _absolute(benchmark_receipt, "benchmark_receipt")
    benchmark = _read_json(benchmark_path, "HCLI benchmark receipt")
    aggregate = benchmark.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise DeepSeekV4GravityError("HCLI benchmark receipt lacks aggregate metrics")
    runtime = manifest.get("runtime_adapter")
    if not isinstance(runtime, Mapping):
        raise DeepSeekV4GravityError("sealed manifest lacks runtime adapter declaration")
    report = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.base_tps_gate.v1",
            "status": "BASE_TRUE_TPS_WITHHELD",
            "created_at": _utc_now(),
            "artifact": {
                "seal_sha256": manifest["seal_sha256"],
                "scope": manifest["diagnostic_scope"],
            },
            "observed_cpu_diagnostic_measurement": {
                "source": str(benchmark_path),
                "hcli_content_blake3": benchmark.get("content_blake3"),
                "complete_forward_tps": aggregate.get("complete_forward_tps"),
                "completed_decode_forwards": aggregate.get("completed_decode_forwards"),
                "decode_ms": aggregate.get("decode_ms"),
                "classification": "DIAGNOSTIC_CPU_ONLY_NOT_BASE_TRUE_TPS",
            },
            "required_base_true_tps_gates": {
                "source_tokenizer_authority": "observed for diagnostic tokenization",
                "source_cpu_authority": "NOT_PROVEN",
                "numeric_parity_v2_1": "NOT_PROVEN",
                "metal_dispatch": "NOT_PRESENT",
                "warmup_and_steady_state": "NOT_RUN_ON_ELIGIBLE_RUNTIME",
                "context_2k": "NOT_SUPPORTED_BY_DIAGNOSTIC",
                "generation_128_tokens": "NOT_SUPPORTED_BY_DIAGNOSTIC",
            },
            "physical_blockers": [
                "The sealed artifact contains one complete full-width layer-4 diagnostic, not the 43-layer source model.",
                "The only runtime adapter is NumPy CPU and declares metal_dispatches=0.",
                "The CPU adapter does not reproduce source activation/QAT behavior and has no independent numeric-parity receipt.",
                "The diagnostic endpoint accepts at most 128 source-token IDs, below the required 2K benchmark context.",
                "The diagnostic endpoint permits at most 4 output tokens, below the required 128-token steady-state benchmark.",
            ],
            "restart_command": (
                "hawking gravity execute --artifact-dir <NEW_OR_RESUMABLE_ARTIFACT_DIR> "
                "--xet-root <NEW_EMPTY_XET_ROOT> --workspace-root <HAWKING_WORKSPACE>"
            ),
        }
    )
    _atomic_json(_absolute(out, "out"), report)
    return report


def _hcli_live_is_hex_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _hcli_live_file(path: str | Path, label: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Read one small JSON evidence object and bind it to its byte hash."""

    evidence = _absolute(path, label)
    _regular_file(evidence, label)
    size = evidence.stat().st_size
    if size > MAX_HCLI_LIVE_EVIDENCE_BYTES:
        raise DeepSeekV4GravityError(
            f"{label} exceeds the {MAX_HCLI_LIVE_EVIDENCE_BYTES}-byte live-suite evidence cap"
        )
    digest = hashlib.sha256()
    with evidence.open("rb") as handle:
        while block := handle.read(DEFAULT_RANGE_BYTES):
            digest.update(block)
    document = _read_json(evidence, label)
    return evidence, document, {"path": str(evidence), "bytes": size, "sha256": digest.hexdigest()}


def _hcli_live_mapping_at(value: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any] | None:
    current: object = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


def _hcli_live_scalar_at(value: Mapping[str, Any], path: Sequence[str]) -> object | None:
    current: object = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _hcli_live_context_summary(context: Mapping[str, Any]) -> dict[str, Any]:
    """Return identity/capability fields, never an arbitrary endpoint payload."""

    keys = (
        "arch",
        "artifact_seal_sha256",
        "capability_status",
        "chat_template",
        "ctx_len_effective",
        "ctx_len_native",
        "free_slots",
        "max_batch",
        "max_output_tokens",
        "metal_dispatches",
        "model_id",
        "recurrent_state_bytes",
        "status",
        "tq_estimated",
        "tq_multiplier",
    )
    return {key: context[key] for key in keys if key in context}


def _hcli_live_health_summary(health: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if health is None:
        return None
    keys = ("http_status", "ready", "status", "url")
    return {key: health[key] for key in keys if key in health}


def _hcli_live_capability_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep declared capability flags without copying arbitrary text fields."""

    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(item, (bool, int, float)) or item is None:
            result[key] = item
    return result


def _hcli_live_artifact_claims(value: Mapping[str, Any]) -> list[dict[str, str]]:
    """Collect explicit artifact identities from a bounded JSON receipt.

    We intentionally only examine named artifact identity fields.  A generic
    ``seal_sha256`` can be the receipt's own seal and must not be confused
    with the diagnostic artifact it was generated against.
    """

    claims: list[dict[str, str]] = []
    stack: list[tuple[str, object]] = [("$", value)]
    nodes = 0
    while stack:
        location, current = stack.pop()
        nodes += 1
        if nodes > 250_000:
            raise DeepSeekV4GravityError("live-suite evidence JSON exceeds the structural walk cap")
        if isinstance(current, Mapping):
            for key, item in current.items():
                if not isinstance(key, str):
                    continue
                child = f"{location}.{key}"
                if key in {"artifact_seal_sha256", "manifest_seal_sha256"} and isinstance(item, str):
                    claims.append({"location": child, "seal_sha256": item})
                elif key == "artifact" and isinstance(item, Mapping):
                    seal_value = item.get("seal_sha256")
                    if isinstance(seal_value, str):
                        claims.append({"location": f"{child}.seal_sha256", "seal_sha256": seal_value})
                stack.append((child, item))
        elif isinstance(current, list):
            for index, item in enumerate(current):
                stack.append((f"{location}[{index}]", item))
    return sorted(claims, key=lambda item: (item["location"], item["seal_sha256"]))


def _hcli_live_prompt_hashes(value: Mapping[str, Any]) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Hash known HCLI prompt locations while keeping their text out of the receipt."""

    locations = (
        ("goal",),
        ("result", "goal"),
        ("result", "receipt", "goal"),
        ("agent", "goal"),
        ("result", "prompt"),
        ("result", "request", "prompt"),
        ("result", "params", "prompt"),
        ("result", "turn", "prompt"),
    )
    hashes: list[dict[str, Any]] = []
    texts: list[str] = []
    for path in locations:
        candidate = _hcli_live_scalar_at(value, path)
        location = ".".join(path)
        if isinstance(candidate, str):
            encoded = candidate.encode("utf-8")
            hashes.append(
                {
                    "location": location,
                    "sha256": _sha256(encoded),
                    "utf8_bytes": len(encoded),
                }
            )
            texts.append(candidate)
            continue
        if not isinstance(candidate, Mapping):
            continue
        text = candidate.get("text")
        record: dict[str, Any] = {"location": location}
        if isinstance(text, str):
            encoded = text.encode("utf-8")
            record.update({"sha256": _sha256(encoded), "utf8_bytes": len(encoded)})
            texts.append(text)
        declared_blake3 = candidate.get("blake3")
        if _hcli_live_is_hex_digest(declared_blake3):
            record["declared_blake3"] = declared_blake3
        if len(record) > 1:
            hashes.append(record)
    return hashes, tuple(texts)


def _hcli_live_error_summary(
    value: Mapping[str, Any], prompt_texts: Sequence[str]
) -> list[dict[str, Any]]:
    locations = (
        ("error",),
        ("reason",),
        ("failure",),
        ("result", "error"),
        ("result", "reason"),
        ("result", "failure"),
        ("result", "receipt", "error"),
        ("result", "receipt", "reason"),
        ("result", "receipt", "failure"),
    )
    errors: list[dict[str, Any]] = []
    for path in locations:
        message = _hcli_live_scalar_at(value, path)
        if not isinstance(message, str):
            continue
        raw = message.encode("utf-8")
        rendered = message
        prompt_material_redacted = False
        for prompt in prompt_texts:
            if prompt and prompt in rendered:
                prompt_material_redacted = True
                rendered = rendered.replace(prompt, f"<prompt_sha256:{_sha256(prompt.encode('utf-8'))}>")
        rendered_bytes = rendered.encode("utf-8")
        truncated = len(rendered_bytes) > MAX_HCLI_LIVE_ERROR_UTF8_BYTES
        if truncated:
            rendered = rendered_bytes[:MAX_HCLI_LIVE_ERROR_UTF8_BYTES].decode("utf-8", errors="ignore")
        errors.append(
            {
                "location": ".".join(path),
                "sha256": _sha256(raw),
                "utf8_bytes": len(raw),
                "text": rendered,
                "truncated": truncated,
                "prompt_material_redacted": prompt_material_redacted,
            }
        )
    return errors


def _hcli_live_status_summary(value: Mapping[str, Any]) -> list[dict[str, str]]:
    locations = (
        ("status",),
        ("result", "status"),
        ("result", "receipt", "status"),
        ("result", "turn", "status"),
    )
    statuses: list[dict[str, str]] = []
    for path in locations:
        status = _hcli_live_scalar_at(value, path)
        if isinstance(status, str):
            statuses.append({"location": ".".join(path), "value": status})
    return statuses


def _hcli_live_runtime_contexts(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    locations = (
        ("result", "runtime", "context"),
        ("runtime", "context"),
        ("runtime", "context_before"),
        ("runtime", "context_after"),
        ("result", "receipt", "runtime", "context"),
        ("result", "receipt", "runtime", "context_before"),
        ("result", "receipt", "runtime", "context_after"),
    )
    contexts: list[dict[str, Any]] = []
    for path in locations:
        context = _hcli_live_mapping_at(value, path)
        if context is not None:
            contexts.append({"location": ".".join(path), "context": _hcli_live_context_summary(context)})
    return contexts


def _hcli_live_capabilities(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    locations = (
        ("result", "backend", "capabilities"),
        ("host", "status", "capabilities"),
        ("result", "receipt", "host", "status", "capabilities"),
    )
    capabilities: list[dict[str, Any]] = []
    for path in locations:
        declared = _hcli_live_mapping_at(value, path)
        if declared is not None:
            capabilities.append(
                {"location": ".".join(path), "values": _hcli_live_capability_summary(declared)}
            )
    return capabilities


def _hcli_live_evidence_summary(
    path: Path,
    value: Mapping[str, Any],
    binding: Mapping[str, Any],
    expected_artifact_seal: str,
    *,
    kind: str,
) -> dict[str, Any]:
    claims = _hcli_live_artifact_claims(value)
    for claim in claims:
        observed = claim["seal_sha256"]
        if not _hcli_live_is_hex_digest(observed):
            raise DeepSeekV4GravityError(
                f"{kind} evidence has an invalid artifact identity at {claim['location']}"
            )
        if observed != expected_artifact_seal:
            raise DeepSeekV4GravityError(
                f"{kind} evidence artifact identity mismatch at {claim['location']}: "
                f"expected {expected_artifact_seal}, observed {observed}"
            )
    prompt_hashes, prompt_texts = _hcli_live_prompt_hashes(value)
    schema = value.get("schema")
    command = value.get("command")
    summary: dict[str, Any] = {
        "kind": kind,
        **binding,
        "schema": schema if isinstance(schema, str) else None,
        "command": command if isinstance(command, str) else None,
        "artifact_seal_claims": claims,
        "statuses": _hcli_live_status_summary(value),
        "runtime_contexts": _hcli_live_runtime_contexts(value),
        "capabilities": _hcli_live_capabilities(value),
        "errors": _hcli_live_error_summary(value, prompt_texts),
        "prompt_hashes": prompt_hashes,
        "prompt_text_disclosed": False,
    }
    # Keep the source path explicit in every evidence record, even if a
    # future binding helper uses a different record shape.
    summary["path"] = str(path)
    return summary


def hcli_live_suite_receipt(
    artifact_dir: str | Path,
    endpoint_capabilities: str | Path,
    evidence_files: Sequence[str | Path],
    out: str | Path,
) -> dict[str, Any]:
    """Seal a privacy-preserving aggregate of real, local HCLI evidence.

    The aggregate is intentionally an evidence index, not a promotion gate:
    it can demonstrate that HCLI was pointed at a sealed diagnostic endpoint,
    but it never turns that endpoint into a full V4, parity, Metal, or TPS
    result.  Original JSON evidence remains separately hashed and inspectable.
    """

    artifact = _absolute(artifact_dir, "artifact_dir")
    target = _absolute(out, "out")
    manifest = load_manifest(artifact)
    artifact_seal = manifest.get("seal_sha256")
    if not _hcli_live_is_hex_digest(artifact_seal):
        raise DeepSeekV4GravityError("sealed diagnostic manifest has an invalid artifact identity")
    if not evidence_files:
        raise DeepSeekV4GravityError("at least one HCLI live evidence file is required")
    if len(evidence_files) > MAX_HCLI_LIVE_EVIDENCE_FILES:
        raise DeepSeekV4GravityError(
            f"live-suite evidence exceeds the {MAX_HCLI_LIVE_EVIDENCE_FILES}-file cap"
        )

    endpoint_path, endpoint_document, endpoint_binding = _hcli_live_file(
        endpoint_capabilities, "endpoint_capabilities"
    )
    if endpoint_path == target:
        raise DeepSeekV4GravityError("out must not overwrite the endpoint capabilities evidence")
    if endpoint_document.get("schema") != "hcli.command.v1" or endpoint_document.get("command") != "capabilities":
        raise DeepSeekV4GravityError(
            "endpoint_capabilities must be an hcli.command.v1 capabilities result"
        )
    endpoint_context = _hcli_live_mapping_at(endpoint_document, ("result", "runtime", "context"))
    if endpoint_context is None:
        raise DeepSeekV4GravityError("endpoint_capabilities lacks result.runtime.context")
    endpoint_claim = endpoint_context.get("artifact_seal_sha256")
    if not _hcli_live_is_hex_digest(endpoint_claim):
        raise DeepSeekV4GravityError("endpoint_capabilities lacks a valid diagnostic artifact identity")
    if endpoint_claim != artifact_seal:
        raise DeepSeekV4GravityError(
            "endpoint capabilities artifact identity does not match the sealed diagnostic manifest"
        )
    endpoint_summary = _hcli_live_evidence_summary(
        endpoint_path,
        endpoint_document,
        endpoint_binding,
        artifact_seal,
        kind="endpoint_capabilities",
    )

    seen = {endpoint_path}
    evidence: list[dict[str, Any]] = []
    for index, candidate in enumerate(evidence_files, start=1):
        path, document, binding = _hcli_live_file(candidate, f"evidence[{index}]")
        if path == target:
            raise DeepSeekV4GravityError("out must not overwrite supplied HCLI evidence")
        if path in seen:
            raise DeepSeekV4GravityError(f"duplicate HCLI evidence path: {path}")
        seen.add(path)
        evidence.append(
            _hcli_live_evidence_summary(
                path, document, binding, artifact_seal, kind="hcli_live"
            )
        )

    report = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.hcli_live_suite.v1",
            "status": "HCLI_LIVE_SUITE_EVIDENCE_SEALED_DIAGNOSTIC_ONLY",
            "created_at": _utc_now(),
            "artifact": {
                "path": str(artifact),
                "seal_sha256": artifact_seal,
                "schema": manifest["schema"],
                "status": manifest["status"],
                "diagnostic_scope": manifest.get("diagnostic_scope"),
                "runtime_adapter": manifest.get("runtime_adapter"),
            },
            "endpoint": {
                "capability_file": endpoint_summary,
                "runtime_context": _hcli_live_context_summary(endpoint_context),
                "health": _hcli_live_health_summary(
                    _hcli_live_mapping_at(endpoint_document, ("result", "runtime", "health"))
                ),
            },
            "evidence": evidence,
            "prompt_disclosure": {
                "mode": "hash_only",
                "note": "Prompt text and completions are intentionally omitted; each supplied JSON file remains bound by SHA-256.",
            },
            "claim_boundary": {
                "hcli_endpoint_evidence": "bound_to_sealed_layer4_diagnostic",
                "full_43_layer_runtime": False,
                "numeric_parity": False,
                "metal_dispatch": False,
                "base_true_tps": False,
            },
        }
    )
    _atomic_json(target, report)
    return report


_COMPLETE_TOKEN_PROFILE_STAGES = (
    "tokenizer_template",
    "embedding",
    "mhc_state_control",
    "norm",
    "qkv",
    "compressed_sparse_attention",
    "index_heads_topk_index",
    "kv_state_read_write",
    "router_top6",
    "expert_gather",
    "gate_up",
    "activation",
    "down",
    "shared_expert",
    "route_combine",
    "residual",
    "lm_head",
    "topk_sampling",
    "endpoint_hcli_streaming",
    "runtime_bookkeeping",
)

_COMPLETE_TOKEN_PROFILE_STAGE_LABELS = {
    "tokenizer_template": "tokenizer/template",
    "embedding": "embedding",
    "mhc_state_control": "mHC state/control",
    "norm": "norm",
    "qkv": "QKV",
    "compressed_sparse_attention": "compressed/sparse attention",
    "index_heads_topk_index": "index heads/top-k index",
    "kv_state_read_write": "KV read/write",
    "router_top6": "router/top-6",
    "expert_gather": "expert gather",
    "gate_up": "gate/up",
    "activation": "activation",
    "down": "down",
    "shared_expert": "shared expert",
    "route_combine": "route combine",
    "residual": "residual",
    "lm_head": "lm_head",
    "topk_sampling": "top-k/sampling",
    "endpoint_hcli_streaming": "endpoint/HCLI streaming",
    "runtime_bookkeeping": "runtime bookkeeping",
}

_DIAGNOSTIC_CPU_FALLBACK = "cpu_numpy_activation_qat_and_metal_parity_not_implemented"

# Latent capture is deliberately bounded to a small, synthetic, hash-only
# membership.  These strings are source-controlled test prompts rather than
# donor data or an evaluation set.  They are never copied into a receipt: the
# capture output carries only per-member hashes, byte counts, and source-token
# hashes.  Keep this list to exactly the requested operational categories so
# it cannot quietly grow into a broad capability evaluation.
_DIAGNOSTIC_LATENT_TRACE_PROMPT_SUITE: tuple[tuple[str, str], ...] = (
    ("coding", "Rust: add two i32 values."),
    ("agent planning", "Plan: inspect, act, verify."),
    ("tool use", "Tool request: read file."),
    ("long-context retrieval", "Context: alpha, beta, gamma. Retrieve beta."),
    ("mathematical reasoning", "Compute two plus two."),
    ("repair", "Repair a failing test."),
    ("general conversation", "Hello. Reply briefly."),
)
_DIAGNOSTIC_LATENT_TRACE_CATEGORIES = tuple(
    category for category, _prompt in _DIAGNOSTIC_LATENT_TRACE_PROMPT_SUITE
)
MAX_DIAGNOSTIC_LATENT_TRACE_PROMPT_TOKENS = 24
MAX_DIAGNOSTIC_LATENT_TRACE_SHARDS_PER_PROMPT = 2
MAX_DIAGNOSTIC_LATENT_HCLI_RECEIPTS = 4
MAX_DIAGNOSTIC_LATENT_HCLI_RECEIPT_BYTES = 8 * 1024**2


def _latent_array_summary(values: np.ndarray) -> dict[str, Any]:
    """Return a bounded numeric witness without retaining an activation.

    The raw float payload is deliberately discarded before returning.  A
    digest gives later parity/transplant work a stable binding while the
    scalar moments, finite count, and shape are sufficient for this early
    diagnostic bridge record.
    """

    array = np.asarray(values)
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise DeepSeekV4GravityError("latent summary requires a numeric array")
    contiguous = np.ascontiguousarray(array)
    as_f32 = contiguous.astype("<f4", copy=False).reshape(-1)
    finite = np.isfinite(as_f32)
    finite_count = int(np.count_nonzero(finite))
    total = int(as_f32.size)
    summary: dict[str, Any] = {
        "shape": [int(value) for value in contiguous.shape],
        "dtype": str(contiguous.dtype),
        "element_count": total,
        "finite_count": finite_count,
        "nonfinite_count": total - finite_count,
        "raw_values_retained": False,
        "float32_le_sha256": _sha256(as_f32.tobytes()),
    }
    if finite_count == 0:
        summary.update(
            {
                "statistics_available": False,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "rms": None,
                "l2_norm": None,
                "p05": None,
                "p50": None,
                "p95": None,
            }
        )
        return summary
    observed = as_f32[finite].astype(np.float64, copy=False)
    summary.update(
        {
            "statistics_available": True,
            "min": float(np.min(observed)),
            "max": float(np.max(observed)),
            "mean": float(np.mean(observed)),
            "std": float(np.std(observed)),
            "rms": float(math.sqrt(float(np.mean(observed * observed)))),
            "l2_norm": float(np.linalg.norm(observed)),
            "p05": float(np.percentile(observed, 5)),
            "p50": float(np.percentile(observed, 50)),
            "p95": float(np.percentile(observed, 95)),
        }
    )
    return summary


def _latent_token_id_digest(token_id: int) -> str:
    if token_id < 0 or token_id > 0xFFFFFFFF:
        raise DeepSeekV4GravityError("latent trace token ID is outside uint32 range")
    return _sha256(struct.pack("<I", token_id))


def _latent_token_ids_digest(token_ids: Sequence[int]) -> str:
    packed = bytearray()
    for token_id in token_ids:
        packed.extend(struct.pack("<I", int(token_id)))
    return _sha256(bytes(packed))


# A freeze is intentionally a small, deterministic evidence bundle rather
# than another mutable campaign database.  These are the only files the
# command is allowed to create below its output directory.
_CHILD_BASELINE_FILENAMES = (
    "DSV4F_CHILD_BASELINE.json",
    "DSV4F_RUNTIME_PROFILE.json",
    "DSV4F_ROUTE_PROFILE.json",
    "DSV4F_LATENT_BRIDGE_CONTRACT.json",
    "DSV4F_TRANSPLANT_POINTS.json",
    "DSV4F_100TPS_SCOREBOARD.json",
    "DSV4F_KERNEL_REGISTRY.json",
    "DSV4F_ROOFLINE.json",
)
_CHILD_BASELINE_MAX_RECEIPT_BYTES = 32 * 1024**2
_CHILD_BASELINE_MAX_COMPONENT_TRACES = 256
_CANONICAL_COMPLETE_TOKEN_PROFILE_V3 = "complete-token-profile-receipt-v3.json"
_CANONICAL_HCLI_LIVE_SUITE_V2 = "hcli-live-suite-receipt-v2.json"
_CANONICAL_FP4_METAL_COMPONENT_PROBE = "fp4-metal-component-probe-receipt.json"
_CANONICAL_FULL_STREAM_READER_ADMISSION = "full-stream-reader-admission-receipt.json"
_CANONICAL_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP_V2 = (
    "DSV4F_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP-v2.json"
)
# The historical v2 sweep was emitted with a Rust insertion-order JSON seal
# and therefore fails ``lab.receipts.verify``.  v3 intentionally requires a
# separately named, fresh canonical reissue rather than accepting that stale
# byte sequence under a second verifier.
_CANONICAL_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP_CANONICAL_V1 = (
    "DSV4F_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP-CANONICAL-v1.json"
)
_CANONICAL_BOUNDED_LATENT_ROUTE_V2 = "bounded-latent-route-receipt-v2.json"
_CANONICAL_STATIC_EXPERT_RESIDENCY_V2 = "static-expert-residency-receipt-v2.json"
_CANONICAL_METAL_DEVICE_COPY_ROOFLINE_V1 = "metal-device-copy-roofline-receipt-v1.json"
_CANONICAL_ACT_QUANT_WQ_A_CPU_ORACLE_V2 = "DSV4F_ACT_QUANT_WQ_A_CPU_ORACLE-v2.json"

# v3 deliberately derives from, rather than rewrites, the already sealed v2
# bundle.  The parent schemas remain v1 because the first two bundles used the
# same stable public document schemas; the mandatory FP8 marker distinguishes
# the v2 parent from the pre-Metal v1 baseline.
_CHILD_BASELINE_V2_PARENT_SCHEMAS = {
    "DSV4F_CHILD_BASELINE.json": "hawking.gravity.deepseek_v4.child_baseline.v1",
    "DSV4F_RUNTIME_PROFILE.json": "hawking.gravity.deepseek_v4.runtime_profile.v1",
    "DSV4F_ROUTE_PROFILE.json": "hawking.gravity.deepseek_v4.route_profile.v1",
    "DSV4F_LATENT_BRIDGE_CONTRACT.json": "hawking.gravity.deepseek_v4.latent_bridge_contract.v1",
    "DSV4F_TRANSPLANT_POINTS.json": "hawking.gravity.deepseek_v4.transplant_points.v1",
    "DSV4F_100TPS_SCOREBOARD.json": "hawking.gravity.deepseek_v4.base_true_tps_scoreboard.v1",
    "DSV4F_KERNEL_REGISTRY.json": "hawking.gravity.deepseek_v4.kernel_registry.v1",
    "DSV4F_ROOFLINE.json": "hawking.gravity.deepseek_v4.roofline.v1",
}


def _child_baseline_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeepSeekV4GravityError(f"{label} must be an object")
    return value


def _child_baseline_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DeepSeekV4GravityError(f"{label} must be an integer >= {minimum}")
    return value


def _child_baseline_number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeepSeekV4GravityError(f"{label} must be a number")
    rendered = float(value)
    if not math.isfinite(rendered) or rendered < minimum:
        raise DeepSeekV4GravityError(f"{label} must be finite and >= {minimum}")
    return rendered


def _child_baseline_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise DeepSeekV4GravityError(f"{label} must be boolean")
    return value


def _child_baseline_digest(value: object, label: str) -> str:
    if not _hcli_live_is_hex_digest(value):
        raise DeepSeekV4GravityError(f"{label} must be a SHA-256 hex digest")
    return str(value)


def _child_baseline_receipt(
    path_value: str | Path,
    label: str,
    *,
    schema: str,
    status: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Read one bounded, sealed receipt and bind both bytes and its seal."""

    path = _absolute(path_value, label)
    _regular_file(path, label)
    size = path.stat().st_size
    if size > _CHILD_BASELINE_MAX_RECEIPT_BYTES:
        raise DeepSeekV4GravityError(
            f"{label} exceeds the {_CHILD_BASELINE_MAX_RECEIPT_BYTES}-byte baseline receipt cap"
        )
    document = _read_json(path, label)
    try:
        verified = verify(document, label=label)
    except SealIntegrityError as exc:
        raise DeepSeekV4GravityError(str(exc)) from exc
    if verified.get("schema") != schema:
        raise DeepSeekV4GravityError(f"{label} has unexpected schema")
    if verified.get("status") != status:
        raise DeepSeekV4GravityError(f"{label} has unexpected status")
    seal_sha256 = _child_baseline_digest(verified.get("seal_sha256"), f"{label}.seal_sha256")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(DEFAULT_RANGE_BYTES):
            digest.update(block)
    return path, verified, {
        "path": str(path),
        "bytes": size,
        "file_sha256": digest.hexdigest(),
        "seal_sha256": seal_sha256,
        "schema": schema,
        "status": status,
    }


def _child_baseline_identity(
    document: Mapping[str, Any],
    path: Sequence[str],
    expected: str,
    label: str,
) -> None:
    observed = _hcli_live_scalar_at(document, path)
    if observed != expected:
        raise DeepSeekV4GravityError(
            f"{label} does not bind the expected artifact seal at {'.'.join(path)}"
        )


def _child_baseline_source_identity(manifest: Mapping[str, Any], label: str) -> dict[str, str]:
    source = _child_baseline_mapping(manifest.get("source"), f"{label}.source")
    repository = source.get("repository")
    revision = source.get("revision")
    if repository != REPOSITORY or revision != REVISION:
        raise DeepSeekV4GravityError(f"{label} is not the pinned DeepSeek-V4-Flash source")
    return {"repository": REPOSITORY, "revision": REVISION}


def _child_baseline_config(
    artifact: Path, manifest: Mapping[str, Any], label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the source-bound config before deriving any bridge geometry."""

    source = _child_baseline_mapping(manifest.get("source"), f"{label}.source")
    assets = _child_baseline_mapping(source.get("metadata_assets"), f"{label}.source.metadata_assets")
    descriptor = _child_baseline_mapping(assets.get("config.json"), f"{label}.config descriptor")
    expected_sha256 = _child_baseline_digest(descriptor.get("sha256"), f"{label}.config sha256")
    config_path = artifact / "metadata" / "config.json"
    _regular_file(config_path, f"{label} config")
    raw = config_path.read_bytes()
    observed_sha256 = _sha256(raw)
    if observed_sha256 != expected_sha256:
        raise DeepSeekV4GravityError(f"{label} config does not match its sealed source asset hash")
    config = _read_json(config_path, f"{label} config")
    return config, {
        "path": str(config_path),
        "bytes": len(raw),
        "sha256": observed_sha256,
        "source_manifest_declared_sha256": expected_sha256,
    }


def _child_baseline_geometry(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only source-declared geometry used in the future bridge contract."""

    fields = {
        "hidden_size": _child_baseline_int(config.get("hidden_size"), "config.hidden_size", minimum=1),
        "num_hidden_layers": _child_baseline_int(
            config.get("num_hidden_layers"), "config.num_hidden_layers", minimum=1
        ),
        "num_attention_heads": _child_baseline_int(
            config.get("num_attention_heads"), "config.num_attention_heads", minimum=1
        ),
        "head_dim": _child_baseline_int(config.get("head_dim"), "config.head_dim", minimum=1),
        "q_lora_rank": _child_baseline_int(config.get("q_lora_rank"), "config.q_lora_rank", minimum=1),
        "o_lora_rank": _child_baseline_int(config.get("o_lora_rank"), "config.o_lora_rank", minimum=1),
        "n_routed_experts": _child_baseline_int(
            config.get("n_routed_experts"), "config.n_routed_experts", minimum=1
        ),
        "n_shared_experts": _child_baseline_int(
            config.get("n_shared_experts"), "config.n_shared_experts", minimum=1
        ),
        "num_experts_per_tok": _child_baseline_int(
            config.get("num_experts_per_tok"), "config.num_experts_per_tok", minimum=1
        ),
        "moe_intermediate_size": _child_baseline_int(
            config.get("moe_intermediate_size"), "config.moe_intermediate_size", minimum=1
        ),
        "vocab_size": _child_baseline_int(config.get("vocab_size"), "config.vocab_size", minimum=1),
        "hc_mult": _child_baseline_int(config.get("hc_mult"), "config.hc_mult", minimum=1),
        "hc_sinkhorn_iters": _child_baseline_int(
            config.get("hc_sinkhorn_iters"), "config.hc_sinkhorn_iters", minimum=1
        ),
        "index_n_heads": _child_baseline_int(
            config.get("index_n_heads"), "config.index_n_heads", minimum=1
        ),
        "index_head_dim": _child_baseline_int(
            config.get("index_head_dim"), "config.index_head_dim", minimum=1
        ),
        "index_topk": _child_baseline_int(config.get("index_topk"), "config.index_topk", minimum=1),
    }
    model_type = config.get("model_type")
    source_dtype = config.get("torch_dtype")
    expert_dtype = config.get("expert_dtype")
    if model_type != "deepseek_v4" or not isinstance(source_dtype, str) or not isinstance(expert_dtype, str):
        raise DeepSeekV4GravityError("source config lacks the expected V4 model/dtype declaration")
    fields.update(
        {
            "model_type": model_type,
            "source_torch_dtype": source_dtype,
            "expert_dtype": expert_dtype,
            "rms_norm_eps": config.get("rms_norm_eps"),
            "hidden_act": config.get("hidden_act"),
            "scoring_func": config.get("scoring_func"),
            "topk_method": config.get("topk_method"),
            "use_cache": config.get("use_cache"),
        }
    )
    return fields


def _child_baseline_output_directory(out_dir: str | Path, artifacts: Sequence[Path]) -> Path:
    target = _absolute(out_dir, "out_dir")
    for artifact in artifacts:
        try:
            target.resolve().relative_to(artifact.resolve())
        except ValueError:
            continue
        raise DeepSeekV4GravityError(
            "freeze-child-baseline out_dir must be external to sealed artifact histories"
        )
    if target.exists():
        node = os.lstat(target)
        if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
            raise DeepSeekV4GravityError("out_dir must be a non-symlink directory")
        present = {entry.name for entry in target.iterdir()}
        unexpected = present.difference(_CHILD_BASELINE_FILENAMES)
        if unexpected:
            raise DeepSeekV4GravityError(
                "out_dir already contains files outside the fixed child baseline bundle: "
                + ", ".join(sorted(unexpected))
            )
    return target


def _child_baseline_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current == current.parent:
            raise DeepSeekV4GravityError("unable to locate an existing output parent")
        current = current.parent
    return current


def _child_baseline_stage_summary(
    profile_aggregate: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, int]]:
    stages = _child_baseline_mapping(profile_aggregate.get("stages"), "complete token profile stages")
    required = tuple(dict.fromkeys(_COMPLETE_TOKEN_PROFILE_STAGES))
    missing = sorted(set(required).difference(stages))
    if missing:
        raise DeepSeekV4GravityError("complete token profile is missing required stages: " + ", ".join(missing))
    forwards = _child_baseline_int(
        profile_aggregate.get("real_diagnostic_forward_count"),
        "complete token profile forward count",
        minimum=1,
    )
    summary: dict[str, Any] = {}
    totals = {
        "bytes_read_estimate_total": 0,
        "bytes_written_estimate_total": 0,
        "fp_operations_estimate_total": 0,
        "integer_bit_operations_estimate_total": 0,
        "dispatches_total": 0,
        "command_buffers_total": 0,
        "waits_total": 0,
    }
    for name in required:
        stage = _child_baseline_mapping(stages.get(name), f"complete token profile stage {name}")
        counter_values = {
            field: _child_baseline_int(stage.get(field), f"{name}.{field}")
            for field in totals
        }
        totals = {field: totals[field] + counter_values[field] for field in totals}
        cpu_duration = _child_baseline_mapping(stage.get("cpu_duration_ms"), f"{name}.cpu_duration_ms")
        cpu_wall = _child_baseline_mapping(stage.get("cpu_wall_elapsed_ms"), f"{name}.cpu_wall_elapsed_ms")
        summary[name] = {
            "label": stage.get("label"),
            "execution_statuses": stage.get("execution_statuses"),
            "cpu_duration_ms": {
                key: _child_baseline_number(cpu_duration.get(key), f"{name}.cpu_duration_ms.{key}")
                for key in ("p50", "p95", "p99")
            },
            "cpu_wall_elapsed_ms": {
                key: _child_baseline_number(cpu_wall.get(key), f"{name}.cpu_wall_elapsed_ms.{key}")
                for key in ("p50", "p95", "p99")
            },
            "gpu_duration_ms": {
                "status": "NOT_AVAILABLE_CPU_NUMPY_DIAGNOSTIC",
                "p50": None,
                "p95": None,
                "p99": None,
            },
            "bytes_read_estimate_total": counter_values["bytes_read_estimate_total"],
            "bytes_written_estimate_total": counter_values["bytes_written_estimate_total"],
            "fp_operations_estimate_total": counter_values["fp_operations_estimate_total"],
            "integer_bit_operations_estimate_total": counter_values[
                "integer_bit_operations_estimate_total"
            ],
            "dispatches_total": counter_values["dispatches_total"],
            "command_buffers_total": counter_values["command_buffers_total"],
            "waits_total": counter_values["waits_total"],
            "occupancy": "NOT_AVAILABLE_CPU_NUMPY_DIAGNOSTIC",
            "effective_bandwidth": "NOT_AVAILABLE_HARDWARE_COUNTERS_NOT_COLLECTED",
            "fallback": "CPU_DIAGNOSTIC_NO_METAL_PARITY",
        }
    totals["real_diagnostic_forward_count"] = forwards
    return summary, totals


def _child_baseline_component_routes(
    trace_receipt: Mapping[str, Any], *, expected_route_count: int
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    result = _child_baseline_mapping(trace_receipt.get("result"), "component trace result")
    traces = result.get("trace")
    if not isinstance(traces, list) or not traces:
        raise DeepSeekV4GravityError("component trace receipt has no trace list")
    if len(traces) > _CHILD_BASELINE_MAX_COMPONENT_TRACES:
        raise DeepSeekV4GravityError("component trace receipt exceeds the bounded trace cap")
    routes: list[dict[str, Any]] = []
    execution = {
        "embedding": False,
        "attention": False,
        "hc_attention": False,
        "hc_ffn": False,
        "router": False,
        "routed_experts": False,
        "shared_expert": False,
        "head": False,
    }
    for ordinal, record in enumerate(traces, start=1):
        trace = _child_baseline_mapping(record, f"component trace {ordinal}")
        component = _child_baseline_mapping(
            trace.get("component_execution"), f"component trace {ordinal}.component_execution"
        )
        router = _child_baseline_mapping(component.get("router"), f"component trace {ordinal}.router")
        selected = router.get("selected_route_ids")
        if not isinstance(selected, list) or len(selected) != expected_route_count:
            raise DeepSeekV4GravityError("component trace does not contain the configured top-k route set")
        route_ids = [
            _child_baseline_int(value, f"component trace {ordinal}.route id", minimum=0)
            for value in selected
        ]
        if len(set(route_ids)) != len(route_ids):
            raise DeepSeekV4GravityError("component trace route set contains duplicate expert IDs")
        for name in execution:
            item = _child_baseline_mapping(component.get(name), f"component trace {ordinal}.{name}")
            execution[name] = execution[name] or _child_baseline_bool(
                item.get("executed"), f"component trace {ordinal}.{name}.executed"
            )
        routes.append(
            {
                "trace_ordinal": ordinal,
                "position": _child_baseline_int(trace.get("position"), f"component trace {ordinal}.position"),
                "selected_route_count": len(route_ids),
                "selected_route_ids": route_ids,
                "router_kind": router.get("kind"),
                "metal_dispatches": _child_baseline_int(
                    component.get("metal_dispatches"), f"component trace {ordinal}.metal_dispatches"
                ),
                "numeric_parity_v2_1": component.get("numeric_parity_v2_1"),
            }
        )
    return routes, execution


def _child_baseline_route_frequency(profile_receipt: Mapping[str, Any]) -> dict[str, int]:
    run = _child_baseline_mapping(profile_receipt.get("profile_run"), "complete token profile run")
    raw = _child_baseline_mapping(run.get("route_frequency"), "complete token profile route_frequency")
    if len(raw) > 256:
        raise DeepSeekV4GravityError("complete token profile route frequency exceeds the expert cap")
    frequency: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.isdecimal():
            raise DeepSeekV4GravityError("complete token profile route frequency key is invalid")
        expert_id = _child_baseline_int(int(key), "complete token profile route expert", minimum=0)
        frequency[str(expert_id)] = _child_baseline_int(
            value, f"complete token profile route frequency {key}", minimum=1
        )
    if not frequency:
        raise DeepSeekV4GravityError("complete token profile has no route-frequency evidence")
    return {key: frequency[key] for key in sorted(frequency, key=lambda item: int(item))}


def _child_baseline_reference_set(
    *,
    full_artifact: Path,
    full_manifest: Mapping[str, Any],
    diagnostic_artifact: Path,
    diagnostic_manifest: Mapping[str, Any],
    full_config: Mapping[str, Any],
    diagnostic_config: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Make one stable identity shared by every independently sealed file."""

    references: dict[str, Any] = {
        "full_stream_manifest": {
            "path": str(full_artifact / "manifest.json"),
            "seal_sha256": _child_baseline_digest(
                full_manifest.get("seal_sha256"), "full manifest seal"
            ),
            "schema": full_manifest.get("schema"),
            "status": full_manifest.get("status"),
        },
        "diagnostic_manifest": {
            "path": str(diagnostic_artifact / "manifest.json"),
            "seal_sha256": _child_baseline_digest(
                diagnostic_manifest.get("seal_sha256"), "diagnostic manifest seal"
            ),
            "schema": diagnostic_manifest.get("schema"),
            "status": diagnostic_manifest.get("status"),
        },
        "full_source_config": dict(full_config),
        "diagnostic_source_config": dict(diagnostic_config),
        "receipts": {key: dict(value) for key, value in sorted(receipts.items())},
    }
    return references, _sha256(_canonical(references))


def _child_baseline_bridge_contracts(
    geometry: Mapping[str, Any],
    *,
    selected_layer: int,
    source_config_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Declare future *interfaces*, never donor weights or trained capability."""

    width = _child_baseline_int(geometry.get("hidden_size"), "bridge geometry.hidden_size", minimum=1)
    experts = _child_baseline_int(
        geometry.get("n_routed_experts"), "bridge geometry.n_routed_experts", minimum=1
    )
    topk = _child_baseline_int(
        geometry.get("num_experts_per_tok"), "bridge geometry.num_experts_per_tok", minimum=1
    )
    source_dtype = geometry.get("source_torch_dtype")
    if not isinstance(source_dtype, str):
        raise DeepSeekV4GravityError("bridge geometry lacks source dtype")
    common = {
        "input_tensor_state": {
            "name": "per_token_hidden_state",
            "shape_contract": ["batch", "sequence", width],
            "source_dtype": source_dtype,
            "current_capture_status": "NOT_EMITTED_BY_CURRENT_DIAGNOSTIC_TRACE",
        },
        "output_tensor_state": {
            "name": "reversible_residual_adapter_output",
            "shape_contract": ["batch", "sequence", width],
            "source_dtype": source_dtype,
            "current_capture_status": "NOT_IMPLEMENTED",
        },
        "normalization": {
            "source_config_rms_norm_eps": geometry.get("rms_norm_eps"),
            "runtime_calibration": "NOT_MEASURED_NO_43_LAYER_RUNTIME",
        },
        "layer_location": {
            "full_source_layers": [0, _child_baseline_int(geometry.get("num_hidden_layers"), "bridge layer count", minimum=1) - 1],
            "current_component_trace_layer": selected_layer,
        },
        "token_alignment": "one source-token position to one adapter position; no donor alignment has been trained",
        "loss_target": "NOT_DEFINED_NO_DONOR_TRAINING_IN_THIS_LANE",
        "verification_method": "requires sealed 43-layer runtime trace, tokenizer alignment, bounded parity checks, and reversible adapter reload",
        "runtime_cost_ceiling": "NOT_MEASURED; must be budgeted against BASE_TRUE_TPS after native runtime registration",
        "reversible_adapter_format": "NOT_IMPLEMENTED; future adapter must be separately content-addressed, sealed, and removable",
        "direct_weight_transplant": False,
    }
    return {
        "contract_version": "DSV4F_LATENT_BRIDGE_CONTRACT.v1",
        "source_geometry_binding": dict(source_config_binding),
        "policy": {
            "donor_weights_present": False,
            "donor_training_performed": False,
            "inheritance_claim": "NONE",
            "direct_weight_transplant": False,
            "note": "Kimi and GLM labels name future interface consumers only; this child baseline contains no donor data or training result.",
        },
        "bridges": [
            {
                "bridge": "KIMI_STRATEGIC_BRIDGE",
                "future_target_functions": [
                    "planning",
                    "tool_policy",
                    "long_horizon_decomposition",
                    "critique",
                    "context_management",
                ],
                **common,
            },
            {
                "bridge": "GLM_MATH_BRIDGE",
                "future_target_functions": [
                    "method_selection",
                    "mathematical_decomposition",
                    "formalization",
                    "proof_repair",
                    "value_ranking",
                ],
                **common,
            },
            {
                "bridge": "DSV4F_CHILD_BODY",
                "current_body_functions": [
                    "runtime",
                    "routing",
                    "compressed_attention",
                    "mHC",
                    "expert_execution",
                    "HCLI_effects",
                ],
                "routing_output_state": {
                    "name": "top_k_expert_ids",
                    "shape_contract": ["batch", "sequence", topk],
                    "expert_id_range": [0, experts - 1],
                    "current_capture_status": "LAYER4_DIAGNOSTIC_ROUTE_IDS_ONLY",
                },
                **common,
            },
        ],
    }


def _child_baseline_transplant_points(
    geometry: Mapping[str, Any],
    *,
    selected_layer: int,
    stage_summary: Mapping[str, Any],
    component_execution: Mapping[str, bool],
    route_observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose named insertion points without pretending raw activations exist."""

    width = _child_baseline_int(geometry.get("hidden_size"), "transplant geometry.hidden_size", minimum=1)
    dtype = geometry.get("source_torch_dtype")
    if not isinstance(dtype, str):
        raise DeepSeekV4GravityError("transplant geometry lacks source dtype")
    route_count = len(route_observations)
    point_specs = (
        ("pre_norm_hidden_state", "norm", "not_individually_emitted"),
        ("post_attention_hidden_state", "compressed_sparse_attention", "not_individually_emitted"),
        ("pre_router_hidden_state", "router_top6", "not_individually_emitted"),
        ("router_logits", "router_top6", "not_emitted"),
        ("selected_expert_ids", "router_top6", "captured_as_bounded_route_ids"),
        ("route_probabilities_and_margins", "router_top6", "not_emitted"),
        ("post_moe_hidden_state", "route_combine", "not_individually_emitted"),
        ("mhc_state", "mhc_state_control", "not_emitted"),
        ("attention_index_state", "index_heads_topk_index", "not_emitted"),
        ("final_hidden_state", "lm_head", "not_emitted"),
        ("lm_head_logits", "lm_head", "hash_only_in_component_trace"),
        ("hcli_tool_action_decision", "endpoint_hcli_streaming", "not_emitted"),
    )
    points: list[dict[str, Any]] = []
    for name, stage, capture in point_specs:
        stage_data = _child_baseline_mapping(stage_summary.get(stage), f"transplant stage {stage}")
        points.append(
            {
                "name": name,
                "source_layer_location": {
                    "full_model": "all 43 layers require a future registered adapter",
                    "current_diagnostic_layer": selected_layer,
                },
                "shape_contract": ["batch", "sequence", width],
                "source_dtype": dtype,
                "source_execution_stage": stage,
                "stage_execution_statuses": stage_data.get("execution_statuses"),
                "component_path_observed": component_execution.get(
                    {
                        "compressed_sparse_attention": "attention",
                        "router_top6": "router",
                        "mhc_state_control": "hc_attention",
                        "lm_head": "head",
                    }.get(stage, "embedding"),
                    False,
                ),
                "current_capture_status": capture,
                "bounded_observation_count": route_count if name == "selected_expert_ids" else 0,
                "direct_weight_transplant": False,
                "future_adapter_requirement": "separately sealed reversible adapter or policy head; no direct donor weight replacement",
            }
        )
    return points


def freeze_child_baseline(
    *,
    full_artifact_dir: str | Path,
    diagnostic_artifact_dir: str | Path,
    complete_token_profile: str | Path,
    hcli_live_suite: str | Path,
    full_runtime_blocker: str | Path,
    fp8_metal_component_probe: str | Path,
    component_trace: str | Path,
    base_tps_gate: str | Path,
    full_reverify: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Freeze the exact DSV4F child-body baseline without promoting it.

    All inputs are independently sealed.  The generated files are deterministic
    for a fixed input set, so a repeat call can only reproduce the same bytes;
    it cannot quietly overwrite a prior baseline with a changed claim.
    """

    full_artifact = _absolute(full_artifact_dir, "full_artifact_dir")
    diagnostic_artifact = _absolute(diagnostic_artifact_dir, "diagnostic_artifact_dir")
    full_manifest = load_full_manifest(full_artifact)
    diagnostic_manifest = load_manifest(diagnostic_artifact)
    source_identity = _child_baseline_source_identity(full_manifest, "full manifest")
    if _child_baseline_source_identity(diagnostic_manifest, "diagnostic manifest") != source_identity:
        raise DeepSeekV4GravityError("full and diagnostic manifests do not name the same pinned source")
    full_config, full_config_binding = _child_baseline_config(full_artifact, full_manifest, "full manifest")
    diagnostic_config, diagnostic_config_binding = _child_baseline_config(
        diagnostic_artifact, diagnostic_manifest, "diagnostic manifest"
    )
    if full_config_binding["sha256"] != diagnostic_config_binding["sha256"]:
        raise DeepSeekV4GravityError("full and diagnostic artifacts do not bind the same source config")
    geometry = _child_baseline_geometry(full_config)
    if _child_baseline_geometry(diagnostic_config) != geometry:
        raise DeepSeekV4GravityError("full and diagnostic source geometry differs")
    diagnostic_scope = _child_baseline_mapping(
        diagnostic_manifest.get("diagnostic_scope"), "diagnostic manifest scope"
    )
    selected_layer = _child_baseline_int(
        diagnostic_scope.get("selected_layer"), "diagnostic scope selected_layer", minimum=0
    )
    if selected_layer >= geometry["num_hidden_layers"]:
        raise DeepSeekV4GravityError("diagnostic selected layer is outside source geometry")

    profile_path = _absolute(complete_token_profile, "complete_token_profile")
    if profile_path.name != _CANONICAL_COMPLETE_TOKEN_PROFILE_V3:
        raise DeepSeekV4GravityError(
            "complete_token_profile must be the canonical complete-token profile v3 filename"
        )
    _, profile, profile_binding = _child_baseline_receipt(
        profile_path,
        "complete_token_profile",
        schema="hawking.gravity.deepseek_v4.complete_token_profile_receipt.v1",
        status="SEALED_REAL_LAYER4_CPU_DIAGNOSTIC_PROFILE_NOT_BASE_TRUE_TPS",
    )
    hcli_path = _absolute(hcli_live_suite, "hcli_live_suite")
    if hcli_path.name != _CANONICAL_HCLI_LIVE_SUITE_V2:
        raise DeepSeekV4GravityError("hcli_live_suite must be the canonical HCLI live suite v2 filename")
    _, hcli, hcli_binding = _child_baseline_receipt(
        hcli_path,
        "hcli_live_suite",
        schema="hawking.gravity.deepseek_v4.hcli_live_suite.v1",
        status="HCLI_LIVE_SUITE_EVIDENCE_SEALED_DIAGNOSTIC_ONLY",
    )
    _, blocker, blocker_binding = _child_baseline_receipt(
        full_runtime_blocker,
        "full_runtime_blocker",
        schema="hawking.gravity.deepseek_v4.full_runtime_blocker.v1",
        status="FULL_STREAMED_RUNTIME_NO_REGISTERED_43_LAYER_ADAPTER",
    )
    _, fp8_probe, fp8_probe_binding = _child_baseline_receipt(
        fp8_metal_component_probe,
        "fp8_metal_component_probe",
        schema="hawking.gravity.deepseek_v4.fp8_e4m3fn_e8m0_metal_component_probe.v1",
        status="PASS_REAL_METAL_COMPONENT_PARITY_NOT_FULL_RUNTIME",
    )
    _, trace, trace_binding = _child_baseline_receipt(
        component_trace,
        "component_trace",
        schema="hawking.gravity.deepseek_v4.diagnostic_generation.v1",
        status="FIRST_TOKEN_GENERATED_DIAGNOSTIC",
    )
    _, tps_gate, tps_gate_binding = _child_baseline_receipt(
        base_tps_gate,
        "base_tps_gate",
        schema="hawking.gravity.deepseek_v4.base_tps_gate.v1",
        status="BASE_TRUE_TPS_WITHHELD",
    )
    _, reverify, reverify_binding = _child_baseline_receipt(
        full_reverify,
        "full_reverify",
        schema="hawking.gravity.deepseek_v4.full_reverify.v1",
        status="FULL_MODEL_STREAM_FULLY_REVERIFIED_RUNTIME_PENDING",
    )

    diagnostic_seal = _child_baseline_digest(diagnostic_manifest.get("seal_sha256"), "diagnostic manifest seal")
    full_seal = _child_baseline_digest(full_manifest.get("seal_sha256"), "full manifest seal")
    _child_baseline_identity(profile, ("artifact", "seal_sha256"), diagnostic_seal, "complete_token_profile")
    _child_baseline_identity(hcli, ("artifact", "seal_sha256"), diagnostic_seal, "hcli_live_suite")
    _child_baseline_identity(trace, ("artifact_seal_sha256",), diagnostic_seal, "component_trace")
    _child_baseline_identity(tps_gate, ("artifact", "seal_sha256"), diagnostic_seal, "base_tps_gate")
    _child_baseline_identity(
        blocker, ("artifact", "manifest_seal_sha256"), full_seal, "full_runtime_blocker"
    )
    _child_baseline_identity(
        fp8_probe, ("artifact", "manifest_seal_sha256"), full_seal, "fp8_metal_component_probe"
    )
    _child_baseline_identity(reverify, ("artifact_seal_sha256",), full_seal, "full_reverify")

    profile_claims = _child_baseline_mapping(profile.get("claim_boundary"), "complete token profile claim boundary")
    if profile_claims.get("base_true_tps") is not False or profile_claims.get("metal_dispatch") is not False:
        raise DeepSeekV4GravityError("complete token profile is not an honest CPU diagnostic receipt")
    profile_aggregate = _child_baseline_mapping(profile.get("aggregate"), "complete token profile aggregate")
    stage_summary, profile_totals = _child_baseline_stage_summary(profile_aggregate)
    timing_accounting = _child_baseline_mapping(
        profile_aggregate.get("timing_accounting"), "complete token profile timing accounting"
    )
    other_share = _child_baseline_number(
        timing_accounting.get("other_share_percent"), "complete token profile other share"
    )
    if other_share > 2.0:
        raise DeepSeekV4GravityError("complete token profile has an unexplained other bucket above 2%")
    gpu_accounting = _child_baseline_mapping(
        profile_aggregate.get("gpu_dispatch_accounting"), "complete token profile GPU accounting"
    )
    if any(
        _child_baseline_int(gpu_accounting.get(field), f"complete token profile GPU {field}") != 0
        for field in ("command_buffers_total", "dispatches_total", "waits_total")
    ):
        raise DeepSeekV4GravityError("CPU diagnostic profile unexpectedly declares GPU work")
    route_frequency = _child_baseline_route_frequency(profile)

    hcli_artifact = _child_baseline_mapping(hcli.get("artifact"), "HCLI live suite artifact")
    _child_baseline_identity(hcli_artifact, ("seal_sha256",), diagnostic_seal, "HCLI live suite artifact")
    hcli_endpoint = _child_baseline_mapping(hcli.get("endpoint"), "HCLI live suite endpoint")
    hcli_context = _child_baseline_mapping(
        hcli_endpoint.get("runtime_context"), "HCLI live suite runtime context"
    )
    if _child_baseline_int(hcli_context.get("metal_dispatches"), "HCLI endpoint metal dispatches") != 0:
        raise DeepSeekV4GravityError("HCLI diagnostic endpoint unexpectedly declares Metal dispatches")
    hcli_claims = _child_baseline_mapping(hcli.get("claim_boundary"), "HCLI live suite claim boundary")
    if hcli_claims.get("full_43_layer_runtime") is not False or hcli_claims.get("base_true_tps") is not False:
        raise DeepSeekV4GravityError("HCLI live suite claim boundary is not diagnostic-only")

    blocker_artifact = _child_baseline_mapping(blocker.get("artifact"), "full runtime blocker artifact")
    if blocker_artifact.get("repository") != REPOSITORY or blocker_artifact.get("revision") != REVISION:
        raise DeepSeekV4GravityError("full runtime blocker does not bind the pinned source")
    blocker_storage = _child_baseline_mapping(
        blocker.get("storage_accounting"), "full runtime blocker storage accounting"
    )
    if _child_baseline_bool(blocker_storage.get("raw_artifact_eviction_authorized"), "raw eviction"):
        raise DeepSeekV4GravityError("full runtime blocker illegally authorizes raw artifact eviction")
    fp8_source = _child_baseline_mapping(fp8_probe.get("source"), "FP8 Metal probe source")
    if fp8_source.get("repository") != REPOSITORY or fp8_source.get("revision") != REVISION:
        raise DeepSeekV4GravityError("FP8 Metal probe does not bind the pinned source")
    fp8_scope = _child_baseline_mapping(fp8_probe.get("scope"), "FP8 Metal probe scope")
    if not all(
        _child_baseline_bool(fp8_scope.get(field), f"FP8 Metal probe scope.{field}")
        for field in (
            "not_a_full_model_load",
            "not_a_generation_or_TPS_claim",
            "not_a_registered_43_layer_runtime_adapter",
        )
    ):
        raise DeepSeekV4GravityError("FP8 Metal probe scope is not safely component-only")
    fp8_parity = _child_baseline_mapping(fp8_probe.get("parity"), "FP8 Metal probe parity")
    if fp8_parity.get("status") != "PASS":
        raise DeepSeekV4GravityError("FP8 Metal probe has no parity pass")
    fp8_metal = _child_baseline_mapping(fp8_probe.get("metal"), "FP8 Metal probe Metal evidence")
    if _child_baseline_bool(fp8_metal.get("fallback"), "FP8 Metal probe fallback"):
        raise DeepSeekV4GravityError("FP8 Metal probe reports fallback")
    if _child_baseline_int(fp8_metal.get("gpu_dispatches"), "FP8 Metal probe GPU dispatches", minimum=1) < 1:
        raise DeepSeekV4GravityError("FP8 Metal probe did not record a GPU dispatch")
    fp8_weight = _child_baseline_mapping(fp8_source.get("weight"), "FP8 Metal probe weight")
    fp8_scale = _child_baseline_mapping(fp8_source.get("scale"), "FP8 Metal probe scale")
    fp8_probe_summary = {
        "receipt": fp8_probe_binding,
        "scope": {
            "component": fp8_scope.get("component"),
            "not_a_full_model_load": True,
            "not_a_generation_or_TPS_claim": True,
            "not_a_registered_43_layer_runtime_adapter": True,
        },
        "source_component": {
            "weight_name": fp8_weight.get("name"),
            "weight_dtype": fp8_weight.get("dtype"),
            "weight_shape": fp8_weight.get("shape"),
            "scale_name": fp8_scale.get("name"),
            "scale_dtype": fp8_scale.get("dtype"),
            "scale_shape": fp8_scale.get("shape"),
        },
        "parity": {
            "status": fp8_parity.get("status"),
            "max_abs_error": fp8_parity.get("max_abs_error"),
            "max_relative_error": fp8_parity.get("max_relative_error"),
        },
        "metal": {
            "device": fp8_metal.get("device"),
            "kernel": fp8_metal.get("kernel"),
            "gpu_dispatches": fp8_metal.get("gpu_dispatches"),
            "command_buffers": fp8_metal.get("command_buffers"),
            "compute_encoders": fp8_metal.get("compute_encoders"),
            "fallback": False,
            "gpu_duration_us": _child_baseline_mapping(
                fp8_metal.get("timing"), "FP8 Metal probe timing"
            ).get("gpu_duration_us"),
        },
    }
    full_verification = _child_baseline_mapping(
        reverify.get("full_chunk_verification"), "full reverify chunk verification"
    )
    if not _child_baseline_bool(full_verification.get("sha256_verified"), "full reverify sha256_verified"):
        raise DeepSeekV4GravityError("full reverify receipt does not attest SHA-256 chunk verification")

    observed_routes, component_execution = _child_baseline_component_routes(
        trace, expected_route_count=geometry["num_experts_per_tok"]
    )
    trace_result = _child_baseline_mapping(trace.get("result"), "component trace result")
    trace_stats = _child_baseline_mapping(trace_result.get("stats"), "component trace stats")
    observed_measurement = _child_baseline_mapping(
        tps_gate.get("observed_cpu_diagnostic_measurement"), "base TPS gate observed measurement"
    )
    if observed_measurement.get("classification") != "DIAGNOSTIC_CPU_ONLY_NOT_BASE_TRUE_TPS":
        raise DeepSeekV4GravityError("base TPS gate does not classify the CPU diagnostic correctly")

    receipt_bindings = {
        "base_tps_gate": tps_gate_binding,
        "component_trace": trace_binding,
        "complete_token_profile_v3": profile_binding,
        "fp8_metal_component_probe": fp8_probe_binding,
        "full_reverify": reverify_binding,
        "full_runtime_blocker": blocker_binding,
        "hcli_live_suite_v2": hcli_binding,
    }
    references, bundle_id = _child_baseline_reference_set(
        full_artifact=full_artifact,
        full_manifest=full_manifest,
        diagnostic_artifact=diagnostic_artifact,
        diagnostic_manifest=diagnostic_manifest,
        full_config=full_config_binding,
        diagnostic_config=diagnostic_config_binding,
        receipts=receipt_bindings,
    )
    common = {
        "baseline_bundle_id": bundle_id,
        "source": source_identity,
        "evidence_bindings": references,
        "claim_boundary": {
            "full_43_layer_runtime": False,
            "numeric_parity_v2_1": False,
            "source_cpu_parity": False,
            "full_43_layer_metal_dispatch": False,
            "base_true_tps_eligible_metal_dispatch": False,
            "component_only_fp8_metal_probe": True,
            "base_true_tps": False,
            "kimi_or_glm_donor_weights_present": False,
            "kimi_or_glm_training_performed": False,
            "direct_weight_transplant": False,
        },
    }

    full_runtime = _child_baseline_mapping(full_manifest.get("runtime_adapter"), "full manifest runtime adapter")
    diagnostic_runtime = _child_baseline_mapping(
        diagnostic_manifest.get("runtime_adapter"), "diagnostic manifest runtime adapter"
    )
    hcli_evidence = hcli.get("evidence")
    if not isinstance(hcli_evidence, list):
        raise DeepSeekV4GravityError("HCLI live suite lacks evidence list")
    profile_wall = _child_baseline_mapping(
        profile_aggregate.get("complete_token_wall_elapsed_ms"), "complete token profile wall timing"
    )
    profile_cpu = _child_baseline_mapping(
        profile_aggregate.get("complete_token_cpu_duration_ms"), "complete token profile CPU timing"
    )
    p50_wall_ms = _child_baseline_number(profile_wall.get("p50"), "complete token profile p50 wall")
    p99_wall_ms = _child_baseline_number(profile_wall.get("p99"), "complete token profile p99 wall")
    p50_cpu_ms = _child_baseline_number(profile_cpu.get("p50"), "complete token profile p50 CPU")

    route_transitions = [
        {
            "from_trace_ordinal": left["trace_ordinal"],
            "to_trace_ordinal": right["trace_ordinal"],
            "shared_expert_count": len(
                set(left["selected_route_ids"]).intersection(right["selected_route_ids"])
            ),
            "classification": "RAW_ADJACENT_DIAGNOSTIC_OBSERVATION_NOT_A_ROUTE_PROBABILITY",
        }
        for left, right in zip(observed_routes, observed_routes[1:])
    ]
    bridge_contract = _child_baseline_bridge_contracts(
        geometry,
        selected_layer=selected_layer,
        source_config_binding=full_config_binding,
    )
    transplant_points = _child_baseline_transplant_points(
        geometry,
        selected_layer=selected_layer,
        stage_summary=stage_summary,
        component_execution=component_execution,
        route_observations=observed_routes,
    )
    if any(
        expert >= geometry["n_routed_experts"]
        for route in observed_routes
        for expert in route["selected_route_ids"]
    ):
        raise DeepSeekV4GravityError("component trace selects an expert outside source geometry")
    if any(int(expert) >= geometry["n_routed_experts"] for expert in route_frequency):
        raise DeepSeekV4GravityError("profile route frequency names an expert outside source geometry")

    documents: dict[str, dict[str, Any]] = {}
    documents["DSV4F_CHILD_BASELINE.json"] = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.child_baseline.v1",
            "status": "DSV4F_CHILD_BASELINE_FROZEN_FULL_STREAM_RUNTIME_PENDING",
            **common,
            "artifacts": {
                "full_stream": {
                    "path": str(full_artifact),
                    "manifest_seal_sha256": full_seal,
                    "status": full_manifest.get("status"),
                    "tensor_count": full_manifest.get("artifact", {}).get("tensor_count")
                    if isinstance(full_manifest.get("artifact"), Mapping)
                    else None,
                    "runtime_adapter": dict(full_runtime),
                },
                "loadable_diagnostic": {
                    "path": str(diagnostic_artifact),
                    "manifest_seal_sha256": diagnostic_seal,
                    "status": diagnostic_manifest.get("status"),
                    "selected_layer": selected_layer,
                    "runtime_adapter": dict(diagnostic_runtime),
                },
            },
            "source_geometry": geometry,
            "frozen_metrics": {
                "base_true_tps": {"value": None, "status": "WITHHELD_NOT_ELIGIBLE"},
                "diagnostic_cpu_complete_forward_tps": observed_measurement.get("complete_forward_tps"),
                "diagnostic_complete_token_wall_p50_ms": p50_wall_ms,
                "diagnostic_complete_token_wall_p99_ms": p99_wall_ms,
                "diagnostic_gpu_dispatches": 0,
                "component_only_fp8_metal_gpu_dispatches": fp8_probe_summary["metal"]["gpu_dispatches"],
                "hcli_live_evidence_count": len(hcli_evidence),
                "route_trace_count": len(observed_routes),
            },
            "component_only_metal_milestone": fp8_probe_summary,
            "storage": {
                "source_parent_materialized": blocker_storage.get("raw_parent_materialized"),
                "source_parent_retained": blocker_artifact.get("source_parent_retained"),
                "raw_content_addressed_full_stream_eviction_authorized": blocker_storage.get(
                    "raw_artifact_eviction_authorized"
                ),
                "protected_floor_bytes": blocker_storage.get("protected_floor_bytes"),
                "eviction_rule": blocker_storage.get("eviction_rule"),
            },
            "freeze_policy": {
                "future_comparisons": [
                    "BASE_DSV4F",
                    "DSV4F_PLUS_KIMI",
                    "DSV4F_PLUS_GLM",
                    "DSV4F_PLUS_KIMI_PLUS_GLM",
                    "FINAL_GRAVITY_RECOMPOSED",
                ],
                "future_inheritance_status": "NOT_STARTED_NOT_CLAIMED",
                "source_windows": "parent source shards were streamed and evicted; retained Gravity chunks are the rollback point",
            },
        }
    )
    documents["DSV4F_RUNTIME_PROFILE.json"] = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.runtime_profile.v1",
            "status": "DSV4F_DIAGNOSTIC_CPU_RUNTIME_PROFILE_FROZEN_FULL_RUNTIME_PENDING",
            **common,
            "measurement_scope": {
                "profile_receipt": profile_binding,
                "real_diagnostic_forward_count": profile_totals["real_diagnostic_forward_count"],
                "context_limit_source_token_ids": hcli_context.get("ctx_len_effective"),
                "output_limit_tokens": hcli_context.get("max_output_tokens"),
                "full_43_layer_runtime": "NOT_REGISTERED",
                "base_true_tps_eligible": False,
            },
            "complete_token_timing": {
                "cpu_duration_ms": {
                    key: _child_baseline_number(profile_cpu.get(key), f"profile CPU {key}")
                    for key in ("p50", "p95", "p99")
                },
                "wall_elapsed_ms": {
                    key: _child_baseline_number(profile_wall.get(key), f"profile wall {key}")
                    for key in ("p50", "p95", "p99")
                },
                "unexplained_other_share_percent": other_share,
                "other_bucket_status": timing_accounting.get("status"),
            },
            "stages": stage_summary,
            "token_profile_totals": profile_totals,
            "gpu_and_hardware_counter_boundary": {
                "gpu_duration_ms": "NOT_AVAILABLE_CPU_NUMPY_DIAGNOSTIC",
                "dispatches_total": gpu_accounting.get("dispatches_total"),
                "command_buffers_total": gpu_accounting.get("command_buffers_total"),
                "waits_total": gpu_accounting.get("waits_total"),
                "occupancy": "NOT_AVAILABLE_CPU_NUMPY_DIAGNOSTIC",
                "effective_bandwidth": "NOT_AVAILABLE_HARDWARE_COUNTERS_NOT_COLLECTED",
            },
            "component_only_metal_probe": fp8_probe_summary,
            "hcli_streaming": {
                "profile_stage_status": _child_baseline_mapping(
                    profile_aggregate.get("endpoint_hcli_streaming"), "profile endpoint HCLI streaming"
                ).get("status"),
                "live_endpoint_context": _hcli_live_context_summary(hcli_context),
                "live_suite_receipt": hcli_binding,
                "complete_token_hcli_timing": "NOT_MEASURED_BY_THE_INPROCESS_PROFILE",
            },
            "fallback": {
                "component_trace_declared": trace_stats.get("fallback"),
                "base_true_tps_fallback_count": "NOT_MEASURED_NO_ELIGIBLE_BASE_RUN",
            },
        }
    )
    documents["DSV4F_ROUTE_PROFILE.json"] = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.route_profile.v1",
            "status": "DSV4F_LAYER4_ROUTE_PROFILE_FROZEN_NOT_FULL_ROUTE_RESIDENCY_PROFILE",
            **common,
            "scope": {
                "selected_layer": selected_layer,
                "source_routed_experts": geometry["n_routed_experts"],
                "source_shared_experts": geometry["n_shared_experts"],
                "source_experts_per_token": geometry["num_experts_per_tok"],
                "profile_forward_count": profile_totals["real_diagnostic_forward_count"],
                "component_trace_count": len(observed_routes),
            },
            "profile_route_frequency": {
                "counts": route_frequency,
                "total_selected_experts": sum(route_frequency.values()),
                "interpretation": "bounded layer-4 CPU diagnostic observations only; not a full-model expert frequency distribution",
            },
            "component_trace_route_sets": observed_routes,
            "observed_adjacent_route_transitions": route_transitions,
            "component_execution": component_execution,
            "unavailable_full_runtime_residency_metrics": {
                "expert_frequency_per_layer": "NOT_MEASURED",
                "route_transition_probabilities": "NOT_MEASURED",
                "route_margins": "NOT_EMITTED",
                "route_probabilities": "NOT_EMITTED",
                "hot_cold_expert_cache_hit_rate": "NOT_MEASURED",
                "cold_miss_latency": "NOT_MEASURED",
                "next_layer_prefetch_accuracy": "NOT_MEASURED",
                "active_expert_bytes_per_token": "NOT_MEASURED",
                "shared_control_bytes_per_token": "NOT_MEASURED",
                "kv_state_bytes_per_token": "NOT_MEASURED",
            },
        }
    )
    documents["DSV4F_LATENT_BRIDGE_CONTRACT.json"] = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.latent_bridge_contract.v1",
            "status": "DSV4F_FUTURE_BRIDGE_INTERFACES_DECLARED_NO_DONOR_INHERITANCE",
            **common,
            "contract": bridge_contract,
            "available_bounded_evidence": {
                "component_trace_receipt": trace_binding,
                "route_trace_count": len(observed_routes),
                "raw_hidden_states": "NOT_RETAINED",
                "small_trace_shards": "NOT_YET_CAPTURED_ON_A_43_LAYER_RUNTIME",
            },
        }
    )
    documents["DSV4F_TRANSPLANT_POINTS.json"] = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.transplant_points.v1",
            "status": "DSV4F_TRANSPLANT_POINTS_FROZEN_SOURCE_BOUND_NO_WEIGHT_GRAFT",
            **common,
            "source_geometry": geometry,
            "points": transplant_points,
            "capture_policy": {
                "raw_hidden_state_retention": "DISALLOWED_BY_CURRENT_BASELINE_NO_UNBOUNDED_RAW_STATES",
                "future_capture": "bounded sufficient statistics and qualified trace shards only after a real 43-layer runtime exists",
                "verification": "source hash, token alignment, layer location, and reversible adapter reload are required",
            },
        }
    )
    documents["DSV4F_100TPS_SCOREBOARD.json"] = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.base_true_tps_scoreboard.v1",
            "status": "BASE_TRUE_TPS_NOT_REACHED_NOT_ELIGIBLE_ON_CURRENT_RUNTIME",
            **common,
            "frozen_benchmark_policy": {
                "batch": 1,
                "primary_context_tokens": 8192,
                "diagnostic_context_tokens": 2048,
                "production_support_context_tokens": 32768,
                "warmup_generated_tokens_minimum": 64,
                "measurement_generated_tokens_minimum": 512,
                "clean_trials_minimum": 5,
                "target_median_base_true_tps": 100.0,
                "target_median_latency_ms_per_token_maximum": 10.0,
                "target_p99_latency_ms_per_token_maximum": 13.0,
                "target_fallback_count": 0,
                "target_real_gpu_dispatches_minimum": 1,
                "policy_status": "REQUIREMENT_NOT_A_MEASUREMENT",
            },
            "metrics": {
                "BASE_TRUE_TPS": {"value": None, "status": "WITHHELD"},
                "ACCELERATED_ACCEPTED_TPS": {"value": None, "status": "NOT_MEASURED"},
                "PREFILL_TPS": {"value": None, "status": "NOT_MEASURED"},
                "TTFT": {"value": None, "status": "NOT_MEASURED_ON_ELIGIBLE_RUNTIME"},
                "HCLI_TOOL_AUGMENTED_THROUGHPUT": {"value": None, "status": "NOT_MEASURED"},
                "diagnostic_cpu_complete_forward_tps": observed_measurement.get("complete_forward_tps"),
                "diagnostic_cpu_decode_ms": observed_measurement.get("decode_ms"),
                "diagnostic_gpu_dispatches": 0,
                "component_only_fp8_metal_gpu_dispatches": fp8_probe_summary["metal"]["gpu_dispatches"],
            },
            "gate_receipt": tps_gate_binding,
            "component_only_metal_probe": fp8_probe_summary,
            "withheld_because": tps_gate.get("physical_blockers"),
            "first_required_change_before_a_real_base_run": _child_baseline_mapping(
                blocker.get("first_missing_milestone"), "full runtime blocker first missing milestone"
            ).get("stage"),
        }
    )
    kernel_candidates = (
        "native_fp4_expert_matvec",
        "native_fp8_control_matvec",
        "fused_decode_scale_dot",
        "split_k",
        "multi_row_simdgroup",
        "packed_vector_loads",
        "threadgroup_codebook_caching",
        "qkv_projection_waves",
        "expert_gate_up_waves",
        "device_activation",
        "down_waves",
        "shared_expert_concurrency",
        "route_weighted_combine",
        "mhc_fusion",
        "compressed_attention_index_fusion",
        "lm_head_topk_sampling_fusion",
    )
    documents["DSV4F_KERNEL_REGISTRY.json"] = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.kernel_registry.v1",
            "status": "DSV4F_KERNEL_REGISTRY_FROZEN_NO_NATIVE_43_LAYER_KERNEL_REGISTERED",
            **common,
            "source_execution_grammar": _child_baseline_mapping(
                blocker.get("missing_execution_grammar"), "full runtime blocker execution grammar"
            ),
            "current_runtime": {
                "full_runtime_adapter": dict(full_runtime),
                "diagnostic_runtime_adapter": dict(diagnostic_runtime),
                "diagnostic_component_execution": component_execution,
                "diagnostic_runtime_metal_dispatches": 0,
                "numeric_parity_v2_1": "NOT_PROVEN",
            },
            "component_only_fp8_metal_probe": fp8_probe_summary,
            "kernel_sweep": [
                {
                    "candidate": candidate,
                    "same_model_before_after_win": "NOT_MEASURED",
                    "parity": "NOT_PROVEN",
                    "geometry_sweep": "NOT_RUN",
                    "promotion_status": "NOT_PROMOTED_NO_NATIVE_43_LAYER_RUNTIME",
                }
                for candidate in kernel_candidates
            ],
            "command_topology": {
                "diagnostic_gpu_command_buffers": 0,
                "diagnostic_gpu_dispatches": 0,
                "diagnostic_gpu_waits": 0,
                "native_token_graph": "NOT_IMPLEMENTED",
                "persistent_causal_loop": "NOT_EVALUATED",
            },
        }
    )
    bytes_per_forward = {
        "bytes_read_estimate": profile_totals["bytes_read_estimate_total"]
        / profile_totals["real_diagnostic_forward_count"],
        "bytes_written_estimate": profile_totals["bytes_written_estimate_total"]
        / profile_totals["real_diagnostic_forward_count"],
        "fp_operations_estimate": profile_totals["fp_operations_estimate_total"]
        / profile_totals["real_diagnostic_forward_count"],
        "integer_bit_operations_estimate": profile_totals["integer_bit_operations_estimate_total"]
        / profile_totals["real_diagnostic_forward_count"],
    }
    if p50_wall_ms <= 0.0:
        raise DeepSeekV4GravityError("complete token profile p50 wall time must be positive")
    documents["DSV4F_ROOFLINE.json"] = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.roofline.v1",
            "status": "DSV4F_DIAGNOSTIC_LOGICAL_ROOFLINE_FROZEN_FULL_GPU_ROOFLINE_UNAVAILABLE",
            **common,
            "diagnostic_logical_profile": {
                "per_real_diagnostic_forward": bytes_per_forward,
                "p50_wall_elapsed_ms": p50_wall_ms,
                "p99_wall_elapsed_ms": p99_wall_ms,
                "p50_cpu_duration_ms": p50_cpu_ms,
                "derived_logical_bytes_per_second_at_p50": (
                    (bytes_per_forward["bytes_read_estimate"] + bytes_per_forward["bytes_written_estimate"])
                    * 1000.0
                    / p50_wall_ms
                ),
                "derived_logical_fp_operations_per_second_at_p50": (
                    bytes_per_forward["fp_operations_estimate"] * 1000.0 / p50_wall_ms
                ),
                "interpretation": "logical source-tensor traffic and operation estimates from the CPU diagnostic; not a hardware bandwidth or GPU roofline measurement",
            },
            "full_geometry_roofline": {
                "status": "NOT_MEASURED_NO_REGISTERED_43_LAYER_NATIVE_RUNTIME",
                "active_bytes_per_token": None,
                "operations_per_token": None,
                "dependency_depth": None,
                "synchronization_cost": None,
                "real_gpu_dispatches": 0,
            },
            "component_only_metal_probe": fp8_probe_summary,
            "exact_current_blocker": _child_baseline_mapping(
                blocker.get("first_missing_milestone"), "full runtime blocker first missing milestone"
            ),
            "smallest_required_change_before_100_tps_can_be_tested": {
                "change": "register a real 43-layer DeepSeek-V4 adapter with source-native FP4/FP8, mHC, MLA/index attention, router, shared expert, tokenizer, and Metal execution",
                "claim": "REQUIRED_TO_MEASURE_NOT_A_CLAIM_THAT_100_TPS_WILL_BE_REACHED",
            },
        }
    )

    target = _child_baseline_output_directory(out_dir, (full_artifact, diagnostic_artifact))
    _floor_check(
        _child_baseline_existing_parent(target),
        MIN_FREE_FLOOR_BYTES,
        8 * 1024**2,
        "freeze-child-baseline output bundle",
    )
    for filename, document in documents.items():
        destination = target / filename
        if not destination.exists():
            continue
        _regular_file(destination, f"existing baseline output {filename}")
        if destination.read_bytes() != _canonical(document) + b"\n":
            raise DeepSeekV4GravityError(
                f"refusing to overwrite a different existing frozen baseline output: {filename}"
            )
    for filename in _CHILD_BASELINE_FILENAMES:
        _atomic_json(target / filename, documents[filename])
    return seal(
        {
            "schema": "hawking.gravity.deepseek_v4.child_baseline_freeze.v1",
            "status": "DSV4F_CHILD_BASELINE_BUNDLE_SEALED",
            "baseline_bundle_id": bundle_id,
            "out_dir": str(target),
            "files": [
                {"name": filename, "seal_sha256": documents[filename]["seal_sha256"]}
                for filename in _CHILD_BASELINE_FILENAMES
            ],
            "full_stream_manifest_seal_sha256": full_seal,
            "diagnostic_manifest_seal_sha256": diagnostic_seal,
        }
    )


def _child_baseline_v3_read_sealed(
    path_value: str | Path,
    label: str,
    *,
    schema: str | None = None,
    status: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Read a bounded sealed v3 input without inventing a success claim.

    v3 consumes several independently-produced receipts.  Keeping this reader
    separate from ``_child_baseline_receipt`` lets a deliberately optional
    future source-forward extension remain sealed and source-bound without
    pretending that its schema has already been approved by this freezer.
    """

    path = _absolute(path_value, label)
    _regular_file(path, label)
    size = path.stat().st_size
    if size > _CHILD_BASELINE_MAX_RECEIPT_BYTES:
        raise DeepSeekV4GravityError(
            f"{label} exceeds the {_CHILD_BASELINE_MAX_RECEIPT_BYTES}-byte baseline receipt cap"
        )
    document = _read_json(path, label)
    try:
        verified = verify(document, label=label)
    except SealIntegrityError as exc:
        raise DeepSeekV4GravityError(str(exc)) from exc
    actual_schema = verified.get("schema")
    actual_status = verified.get("status")
    if not isinstance(actual_schema, str) or not actual_schema:
        raise DeepSeekV4GravityError(f"{label} lacks a non-empty schema")
    if not isinstance(actual_status, str) or not actual_status:
        raise DeepSeekV4GravityError(f"{label} lacks a non-empty status")
    if schema is not None and actual_schema != schema:
        raise DeepSeekV4GravityError(f"{label} has unexpected schema")
    if status is not None and actual_status != status:
        raise DeepSeekV4GravityError(f"{label} has unexpected status")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(DEFAULT_RANGE_BYTES):
            digest.update(block)
    return path, verified, {
        "path": str(path),
        "bytes": size,
        "file_sha256": digest.hexdigest(),
        "seal_sha256": _child_baseline_digest(verified.get("seal_sha256"), f"{label}.seal_sha256"),
        "schema": actual_schema,
        "status": actual_status,
    }


def _child_baseline_v3_exact_seal(
    document: Mapping[str, Any], path: Sequence[str], expected: str, label: str
) -> None:
    observed = _hcli_live_scalar_at(document, path)
    if observed != expected:
        raise DeepSeekV4GravityError(
            f"{label} does not bind the expected full-stream seal at {'.'.join(path)}"
        )


def _child_baseline_v3_parent_bundle(
    parent_baseline_dir: str | Path,
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, Any]]:
    """Verify the sealed v2 parent as immutable input to the v3 revision."""

    parent = _absolute(parent_baseline_dir, "prior_baseline_v2_dir")
    if not parent.exists():
        raise DeepSeekV4GravityError("prior_baseline_v2_dir does not exist")
    node = os.lstat(parent)
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
        raise DeepSeekV4GravityError("prior_baseline_v2_dir must be a non-symlink directory")
    expected_names = set(_CHILD_BASELINE_FILENAMES)
    actual_names = {entry.name for entry in parent.iterdir()}
    if actual_names != expected_names:
        missing = sorted(expected_names.difference(actual_names))
        unexpected = sorted(actual_names.difference(expected_names))
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise DeepSeekV4GravityError(
            "prior_baseline_v2_dir must contain exactly the eight frozen child files: "
            + "; ".join(details)
        )

    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    bundle_ids: set[str] = set()
    evidence_fingerprints: set[str] = set()
    for filename in _CHILD_BASELINE_FILENAMES:
        _, document, binding = _child_baseline_v3_read_sealed(
            parent / filename,
            f"prior baseline {filename}",
            schema=_CHILD_BASELINE_V2_PARENT_SCHEMAS[filename],
        )
        bundle_ids.add(
            _child_baseline_digest(
                document.get("baseline_bundle_id"), f"prior baseline {filename}.baseline_bundle_id"
            )
        )
        claim_boundary = _child_baseline_mapping(
            document.get("claim_boundary"), f"prior baseline {filename}.claim_boundary"
        )
        for field in (
            "full_43_layer_runtime",
            "source_cpu_parity",
            "numeric_parity_v2_1",
            "base_true_tps",
        ):
            if _child_baseline_bool(claim_boundary.get(field), f"prior baseline {filename}.{field}"):
                raise DeepSeekV4GravityError(
                    f"prior baseline {filename} illegally promotes {field}"
                )
        evidence = _child_baseline_mapping(
            document.get("evidence_bindings"), f"prior baseline {filename}.evidence_bindings"
        )
        evidence_fingerprints.add(_sha256(_canonical(evidence)))
        documents[filename] = document
        bindings[filename] = binding
    if len(bundle_ids) != 1:
        raise DeepSeekV4GravityError("prior baseline files do not share one baseline bundle ID")
    if len(evidence_fingerprints) != 1:
        raise DeepSeekV4GravityError("prior baseline files do not share one evidence binding set")

    child = documents["DSV4F_CHILD_BASELINE.json"]
    child_claims = _child_baseline_mapping(child.get("claim_boundary"), "prior child baseline claim boundary")
    if child_claims.get("component_only_fp8_metal_probe") is not True:
        raise DeepSeekV4GravityError(
            "prior baseline is not the Metal-aware v2 child baseline required by v3"
        )
    evidence = _child_baseline_mapping(child.get("evidence_bindings"), "prior child baseline evidence")
    full_manifest = _child_baseline_mapping(
        evidence.get("full_stream_manifest"), "prior baseline full stream manifest"
    )
    diagnostic_manifest = _child_baseline_mapping(
        evidence.get("diagnostic_manifest"), "prior baseline diagnostic manifest"
    )
    full_seal = _child_baseline_digest(
        full_manifest.get("seal_sha256"), "prior baseline full stream manifest seal"
    )
    diagnostic_seal = _child_baseline_digest(
        diagnostic_manifest.get("seal_sha256"), "prior baseline diagnostic manifest seal"
    )
    receipts = _child_baseline_mapping(evidence.get("receipts"), "prior baseline receipts")
    if "fp8_metal_component_probe" not in receipts:
        raise DeepSeekV4GravityError("prior v2 baseline does not bind its FP8 Metal component probe")
    return parent, documents, {
        "baseline_bundle_id": next(iter(bundle_ids)),
        "evidence_bindings_sha256": next(iter(evidence_fingerprints)),
        "files": {name: bindings[name] for name in _CHILD_BASELINE_FILENAMES},
        "full_stream_manifest_seal_sha256": full_seal,
        "diagnostic_manifest_seal_sha256": diagnostic_seal,
        "parent_evidence_bindings": copy.deepcopy(evidence),
    }


def _child_baseline_v3_require_false(
    document: Mapping[str, Any], path: Sequence[str], label: str
) -> None:
    value = _hcli_live_scalar_at(document, path)
    if value is not False:
        raise DeepSeekV4GravityError(f"{label} must be explicitly false at {'.'.join(path)}")


def _child_baseline_v3_output_directory(
    out_dir: str | Path,
    *,
    parent_baseline: Path,
) -> Path:
    target = _child_baseline_output_directory(out_dir, ())
    try:
        target.resolve().relative_to(parent_baseline.resolve())
    except ValueError:
        return target
    raise DeepSeekV4GravityError("v3 out_dir must not be inside the immutable v2 baseline directory")


def _child_baseline_v3_named_receipt(
    path_value: str | Path,
    label: str,
    *,
    filename: str,
    schema: str,
    status: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _absolute(path_value, label)
    if path.name != filename:
        raise DeepSeekV4GravityError(f"{label} must be the canonical {filename} receipt")
    _, document, binding = _child_baseline_v3_read_sealed(
        path, label, schema=schema, status=status
    )
    return document, binding


def _child_baseline_v3_fp4_probe(
    path_value: str | Path, *, full_seal: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, binding = _child_baseline_v3_named_receipt(
        path_value,
        "fp4_metal_component_probe",
        filename=_CANONICAL_FP4_METAL_COMPONENT_PROBE,
        schema="hawking.gravity.deepseek_v4.fp4_e2m1fn_x2_e8m0_metal_component_probe.v1",
        status="PASS_REAL_METAL_COMPONENT_PARITY_NOT_FULL_RUNTIME",
    )
    _child_baseline_v3_exact_seal(
        document, ("artifact", "manifest_seal_sha256"), full_seal, "fp4_metal_component_probe"
    )
    source = _child_baseline_mapping(document.get("source"), "FP4 Metal probe source")
    if source.get("repository") != REPOSITORY or source.get("revision") != REVISION:
        raise DeepSeekV4GravityError("FP4 Metal probe is not bound to the pinned DeepSeek source")
    scope = _child_baseline_mapping(document.get("scope"), "FP4 Metal probe scope")
    for field in (
        "not_a_full_model_load",
        "not_a_generation_or_TPS_claim",
        "not_a_registered_43_layer_runtime_adapter",
        "not_an_MoE_route_or_expert_selection_claim",
    ):
        if _child_baseline_bool(scope.get(field), f"FP4 Metal probe scope.{field}") is not True:
            raise DeepSeekV4GravityError("FP4 Metal probe scope is not component-only")
    parity = _child_baseline_mapping(document.get("parity"), "FP4 Metal probe parity")
    if parity.get("status") != "PASS":
        raise DeepSeekV4GravityError("FP4 Metal probe does not have a parity pass")
    metal = _child_baseline_mapping(document.get("metal"), "FP4 Metal probe Metal evidence")
    if _child_baseline_bool(metal.get("fallback"), "FP4 Metal probe fallback"):
        raise DeepSeekV4GravityError("FP4 Metal probe reports fallback")
    if _child_baseline_int(metal.get("gpu_dispatches"), "FP4 Metal GPU dispatches", minimum=1) < 1:
        raise DeepSeekV4GravityError("FP4 Metal probe did not record a GPU dispatch")
    weight = _child_baseline_mapping(source.get("weight"), "FP4 Metal probe weight")
    scale = _child_baseline_mapping(source.get("scale"), "FP4 Metal probe scale")
    timing = _child_baseline_mapping(metal.get("timing"), "FP4 Metal probe timing")
    return document, {
        "receipt": binding,
        "parity_scope": "RAW_WEIGHT_COMPONENT_ONLY_NOT_SOURCE_FORWARD_PARITY",
        "source_component": {
            "weight_name": weight.get("name"),
            "weight_dtype": weight.get("dtype"),
            "weight_shape": weight.get("shape"),
            "scale_name": scale.get("name"),
            "scale_dtype": scale.get("dtype"),
            "scale_shape": scale.get("shape"),
        },
        "parity": {
            "status": parity.get("status"),
            "max_abs_error": parity.get("max_abs_error"),
            "max_relative_error": parity.get("max_relative_error"),
        },
        "metal": {
            "device": metal.get("device"),
            "kernel": metal.get("kernel"),
            "gpu_dispatches": metal.get("gpu_dispatches"),
            "command_buffers": metal.get("command_buffers"),
            "compute_encoders": metal.get("compute_encoders"),
            "gpu_duration_us": timing.get("gpu_duration_us"),
            "fallback": False,
        },
    }


def _child_baseline_v3_reader_admission(
    path_value: str | Path, *, full_seal: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, binding = _child_baseline_v3_named_receipt(
        path_value,
        "full_stream_reader_admission",
        filename=_CANONICAL_FULL_STREAM_READER_ADMISSION,
        schema="hawking.gravity.deepseek_v4.full_stream_reader_admission.v1",
        status="PASS_FULL_STREAM_READER_ADMISSION_NOT_FORWARD_OR_RUNTIME",
    )
    _child_baseline_v3_exact_seal(
        document, ("artifact", "manifest_seal_sha256"), full_seal, "full_stream_reader_admission"
    )
    artifact = _child_baseline_mapping(document.get("artifact"), "full stream reader artifact")
    source = _child_baseline_mapping(artifact.get("source"), "full stream reader source")
    if source.get("repository") != REPOSITORY or source.get("revision") != REVISION:
        raise DeepSeekV4GravityError("full stream reader admission is not pinned to DeepSeek-V4-Flash")
    execution = _child_baseline_mapping(document.get("execution_boundary"), "reader execution boundary")
    for field in (
        "base_true_tps_measured",
        "engine_created",
        "hcli_endpoint_started",
        "public_cli_serve_admission_changed",
        "reader_only",
    ):
        expected = field == "reader_only"
        if _child_baseline_bool(execution.get(field), f"reader execution boundary.{field}") is not expected:
            raise DeepSeekV4GravityError("full stream reader admission crosses its reader-only boundary")
    for field in ("forward_tokens", "gpu_dispatches", "metal_allocations"):
        if _child_baseline_int(execution.get(field), f"reader execution boundary.{field}") != 0:
            raise DeepSeekV4GravityError("full stream reader admission unexpectedly executed runtime work")
    validity = _child_baseline_mapping(document.get("admission_validity"), "reader admission validity")
    for field in (
        "manifest_seal_verified",
        "all_named_tensor_source_index_bindings_verified",
        "all_referenced_chunk_paths_regular_non_symlink_and_exact_tree_verified",
        "all_tensor_segment_contiguity_and_source_offset_mappings_verified",
        "restart_receipt_and_journal_bindings_verified",
        "schema_status_pinned_source_verified",
    ):
        if _child_baseline_bool(validity.get(field), f"reader admission validity.{field}") is not True:
            raise DeepSeekV4GravityError("full stream reader admission lacks required integrity evidence")
    contracts = document.get("validated_native_scale_contracts")
    if not isinstance(contracts, list) or len(contracts) < 2:
        raise DeepSeekV4GravityError("full stream reader admission lacks native scale contracts")
    kinds = {entry.get("kind") for entry in contracts if isinstance(entry, Mapping)}
    if not {
        "native.fp8_e4m3fn_e8m0",
        "native.fp4_e2m1fn_x2_e8m0",
    }.issubset(kinds):
        raise DeepSeekV4GravityError("full stream reader admission lacks FP4/FP8 scale contracts")
    reads = document.get("verified_reads")
    if not isinstance(reads, list) or not reads:
        raise DeepSeekV4GravityError("full stream reader admission lacks verified reads")
    for ordinal, read in enumerate(reads, start=1):
        item = _child_baseline_mapping(read, f"reader verified read {ordinal}")
        if _child_baseline_bool(
            item.get("touched_chunks_sha256_verified_before_return"),
            f"reader verified read {ordinal}.chunk verification",
        ) is not True:
            raise DeepSeekV4GravityError("full stream reader admission returned an unverified chunk")
    return document, {
        "receipt": binding,
        "admission_scope": "CONTENT_ADDRESSED_READER_ONLY_NOT_FORWARD_OR_RUNTIME",
        "verified_read_count": len(reads),
        "native_fp4_pair_count": validity.get("native_fp4_pair_count"),
        "native_fp8_pair_count": validity.get("native_fp8_pair_count"),
        "native_scale_pair_count": validity.get("native_scale_pair_count"),
        "all_chunk_sha256_bytes_verified": validity.get("all_chunk_sha256_bytes_verified"),
        "verified_read_chunk_sha256_before_return": True,
        "pinned_codec_assets": copy.deepcopy(validity.get("pinned_codec_assets")),
    }


def _child_baseline_v3_simdgroup_sweep(
    path_value: str | Path, *, full_seal: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _absolute(path_value, "raw_weight_simdgroup_splitk_sweep")
    if path.name != _CANONICAL_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP_CANONICAL_V1:
        raise DeepSeekV4GravityError(
            "raw_weight_simdgroup_splitk_sweep must be the fresh canonical reissue "
            f"{_CANONICAL_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP_CANONICAL_V1}; the historical "
            f"{_CANONICAL_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP_V2} receipt is not accepted"
        )
    _, document, binding = _child_baseline_v3_read_sealed(
        path,
        "raw_weight_simdgroup_splitk_sweep",
        schema="hawking.gravity.deepseek_v4.raw_weight_simdgroup_splitk_sweep.v1",
        status="PASS_REAL_M3_METAL_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP_NOT_SOURCE_FORWARD_OR_RUNTIME",
    )
    _child_baseline_v3_exact_seal(
        document, ("artifact_binding", "manifest_seal_sha256"), full_seal, "raw SIMDgroup sweep"
    )
    scope = _child_baseline_mapping(document.get("scope"), "raw SIMDgroup sweep scope")
    for field in (
        "raw_weight_component_only",
        "not_source_forward_parity",
        "not_a_full_model_load",
        "not_a_full_43_layer_runtime_adapter",
        "not_a_token_or_generation",
        "not_a_BASE_TRUE_TPS_measurement",
        "not_a_runtime_kernel_promotion",
        "same_sealed_full_gravity_artifact_before_and_after",
        "same_deterministic_input_and_raw_weight_cpu_reference_before_and_after",
    ):
        if _child_baseline_bool(scope.get(field), f"raw SIMDgroup scope.{field}") is not True:
            raise DeepSeekV4GravityError("raw SIMDgroup sweep has an unsafe scope claim")
    metal = _child_baseline_mapping(document.get("metal"), "raw SIMDgroup Metal evidence")
    if _child_baseline_bool(metal.get("fallback"), "raw SIMDgroup fallback"):
        raise DeepSeekV4GravityError("raw SIMDgroup sweep reports fallback")
    if _child_baseline_int(metal.get("aggregate_real_gpu_dispatches"), "raw SIMDgroup dispatches", minimum=1) < 1:
        raise DeepSeekV4GravityError("raw SIMDgroup sweep has no real GPU dispatches")
    before_after = _child_baseline_mapping(document.get("before_after"), "raw SIMDgroup before/after")
    winners: dict[str, Any] = {}
    for family in ("fp4_routed_expert", "fp8_control"):
        result = _child_baseline_mapping(before_after.get(family), f"raw SIMDgroup {family}")
        if result.get("p50_outcome") != "CANDIDATE_GPU_P50_WIN_NOT_PROMOTED":
            raise DeepSeekV4GravityError("raw SIMDgroup result must remain unpromoted")
        if _child_baseline_bool(
            result.get("same_raw_weight_input_and_cpu_reference"),
            f"raw SIMDgroup {family}.same_reference",
        ) is not True:
            raise DeepSeekV4GravityError("raw SIMDgroup before/after did not preserve its CPU reference")
        winners[family] = {
            "authority_serial_winner_gpu_p50_us": result.get("authority_serial_winner_gpu_p50_us"),
            "candidate_parallel_winner_gpu_p50_us": result.get("candidate_parallel_winner_gpu_p50_us"),
            "p50_speedup_authority_divided_by_candidate": result.get(
                "p50_speedup_authority_divided_by_candidate"
            ),
            "promotion": result.get("promotion"),
        }
    return document, {
        "receipt": binding,
        "parity_scope": "RAW_WEIGHT_COMPONENT_ONLY_NOT_SOURCE_FORWARD_PARITY",
        "device": metal.get("device"),
        "aggregate_real_gpu_dispatches": metal.get("aggregate_real_gpu_dispatches"),
        "aggregate_command_buffers": metal.get("aggregate_command_buffers"),
        "aggregate_cpu_visible_waits": metal.get("aggregate_cpu_visible_waits"),
        "fallback": False,
        "unpromoted_winners": winners,
    }


def _child_baseline_v3_latent_routes(
    path_value: str | Path, *, diagnostic_seal: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, binding = _child_baseline_v3_named_receipt(
        path_value,
        "bounded_latent_route_receipt",
        filename=_CANONICAL_BOUNDED_LATENT_ROUTE_V2,
        schema="hawking.gravity.deepseek_v4.diagnostic_latent_route_receipt.v1",
        status="SEALED_BOUNDED_LAYER4_CPU_DIAGNOSTIC_LATENT_ROUTE_CAPTURE",
    )
    _child_baseline_v3_exact_seal(
        document,
        ("source_hash_binding", "artifact_seal_sha256"),
        diagnostic_seal,
        "bounded latent route receipt",
    )
    artifact = _child_baseline_mapping(document.get("artifact"), "bounded latent route artifact")
    scope = _child_baseline_mapping(artifact.get("diagnostic_scope"), "bounded latent route scope")
    if _child_baseline_int(scope.get("selected_layer"), "bounded latent selected layer", minimum=0) != 4:
        raise DeepSeekV4GravityError("bounded latent route receipt is not the layer-4 diagnostic")
    if _child_baseline_bool(scope.get("not_full_model"), "bounded latent not_full_model") is not True:
        raise DeepSeekV4GravityError("bounded latent route receipt crosses its diagnostic scope")
    source = _child_baseline_mapping(document.get("source_hash_binding"), "bounded latent source binding")
    if source.get("repository") != REPOSITORY or source.get("revision") != REVISION:
        raise DeepSeekV4GravityError("bounded latent route receipt is not bound to the pinned source")
    if _child_baseline_bool(source.get("source_parent_retained"), "bounded latent source parent"):
        raise DeepSeekV4GravityError("bounded latent route receipt illegally retains a source parent")
    claims = _child_baseline_mapping(document.get("claim_boundary"), "bounded latent claim boundary")
    for field in (
        "base_true_tps",
        "donor_data_or_distillation",
        "full_43_layer_runtime",
        "hcli_tool_augmented_throughput",
        "metal_dispatch",
        "numeric_parity_v2_1",
        "source_cpu_parity",
    ):
        _child_baseline_v3_require_false(document, ("claim_boundary", field), f"bounded latent {field}")
    if _child_baseline_bool(
        claims.get("real_source_derived_layer4_forwards"),
        "bounded latent real layer-4 forwards",
    ) is not True:
        raise DeepSeekV4GravityError("bounded latent route receipt lacks real diagnostic forwards")
    capture = _child_baseline_mapping(document.get("capture"), "bounded latent capture")
    limits = _child_baseline_mapping(capture.get("collection_limits"), "bounded latent limits")
    for field in ("raw_completions_retained", "raw_hidden_states_retained", "raw_prompts_retained"):
        _child_baseline_v3_require_false(capture, ("collection_limits", field), f"bounded latent {field}")
    membership = _child_baseline_mapping(capture.get("membership_partition"), "bounded latent membership")
    for field in ("disjoint_prompt_hashes", "disjoint_source_token_sequences"):
        if _child_baseline_bool(membership.get(field), f"bounded latent membership.{field}") is not True:
            raise DeepSeekV4GravityError("bounded latent route membership is not disjoint")
    excluded = membership.get("excluded_from")
    if not isinstance(excluded, list) or not {"fit", "calibration", "public_test", "hidden_test"}.issubset(
        set(excluded)
    ):
        raise DeepSeekV4GravityError("bounded latent route receipt does not exclude evaluation memberships")
    aggregate = _child_baseline_mapping(capture.get("route_aggregate"), "bounded latent route aggregate")
    forwards = _child_baseline_int(
        aggregate.get("total_source_forwards"), "bounded latent forward count", minimum=1
    )
    raw_frequency = _child_baseline_mapping(
        aggregate.get("expert_frequency"), "bounded latent expert frequency"
    )
    frequency: dict[str, int] = {}
    for raw_expert, raw_count in raw_frequency.items():
        if not isinstance(raw_expert, str) or not raw_expert.isdecimal():
            raise DeepSeekV4GravityError("bounded latent frequency contains an invalid expert ID")
        expert = _child_baseline_int(int(raw_expert), "bounded latent expert ID", minimum=0)
        if expert >= 256:
            raise DeepSeekV4GravityError("bounded latent frequency names an out-of-range expert")
        frequency[str(expert)] = _child_baseline_int(
            raw_count, f"bounded latent expert {expert} frequency", minimum=1
        )
    if not frequency or sum(frequency.values()) != forwards * 6:
        raise DeepSeekV4GravityError("bounded latent route frequency does not reconcile with top-6 forwards")
    if _child_baseline_bool(
        aggregate.get("raw_route_sequence_retained"), "bounded latent raw route sequence"
    ):
        raise DeepSeekV4GravityError("bounded latent route receipt retains raw route sequences")
    shards = capture.get("trace_shards")
    if not isinstance(shards, list):
        raise DeepSeekV4GravityError("bounded latent route receipt lacks trace-shard metadata")
    max_shards = _child_baseline_int(limits.get("categories"), "bounded latent categories", minimum=1) * _child_baseline_int(
        limits.get("max_trace_shards_per_category"), "bounded latent trace cap", minimum=1
    )
    if len(shards) > max_shards:
        raise DeepSeekV4GravityError("bounded latent route receipt exceeds its trace-shard cap")
    ranked = sorted(
        ((int(expert), count) for expert, count in frequency.items()), key=lambda item: (-item[1], item[0])
    )
    return document, {
        "receipt": binding,
        "scope": {
            "selected_layer": 4,
            "diagnostic_only": True,
            "full_43_layer_runtime": False,
            "source_cpu_parity": False,
            "numeric_parity_v2_1": False,
            "raw_prompts_completions_hidden_states_retained": False,
            "raw_route_sequence_retained": False,
        },
        "collection": {
            "categories": limits.get("categories"),
            "trace_shard_count": len(shards),
            "max_trace_shards_per_category": limits.get("max_trace_shards_per_category"),
            "diagnostic_context_limit_source_tokens": limits.get(
                "diagnostic_context_limit_source_tokens"
            ),
            "membership_partition": membership.get("name"),
        },
        "route_aggregate": {
            "total_source_forwards": forwards,
            "distinct_expert_count": aggregate.get("distinct_expert_count"),
            "expert_frequency": {key: frequency[key] for key in sorted(frequency, key=int)},
            "top_frequency_ranked_experts": [
                {"expert_id": expert, "selection_count": count} for expert, count in ranked[:32]
            ],
            "distinct_route_set_count": len(
                _child_baseline_mapping(
                    aggregate.get("route_set_frequency"), "bounded latent route-set frequency"
                )
            ),
            "distinct_adjacent_transition_count": len(
                _child_baseline_mapping(
                    aggregate.get("route_set_transition_frequency"),
                    "bounded latent route-transition frequency",
                )
            ),
        },
    }


def _child_baseline_v3_static_residency(
    path_value: str | Path,
    *,
    full_seal: str,
    diagnostic_seal: str,
    latent_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, binding = _child_baseline_v3_named_receipt(
        path_value,
        "static_expert_residency",
        filename=_CANONICAL_STATIC_EXPERT_RESIDENCY_V2,
        schema=STATIC_EXPERT_RESIDENCY_SCHEMA,
        status="SEALED_STATIC_FULL_STREAM_EXPERT_RESIDENCY_CONTRACT_RUNTIME_PENDING",
    )
    _child_baseline_v3_exact_seal(
        document, ("source_binding", "full_manifest_seal_sha256"), full_seal, "static expert residency"
    )
    source = _child_baseline_mapping(document.get("source_binding"), "static residency source binding")
    if source.get("repository") != REPOSITORY or source.get("revision") != REVISION:
        raise DeepSeekV4GravityError("static residency receipt is not pinned to DeepSeek-V4-Flash")
    mode = _child_baseline_mapping(document.get("analysis_mode"), "static residency analysis mode")
    for field in ("base_true_tps", "runtime_forward_executed", "training_or_distillation"):
        _child_baseline_v3_require_false(document, ("analysis_mode", field), f"static residency {field}")
    for field in ("command_buffers", "gpu_dispatches", "tensor_payload_bytes_read"):
        if _child_baseline_int(mode.get(field), f"static residency {field}") != 0:
            raise DeepSeekV4GravityError("static residency receipt unexpectedly executed runtime work")
    if _child_baseline_bool(mode.get("manifest_and_metadata_only"), "static residency metadata only") is not True:
        raise DeepSeekV4GravityError("static residency receipt is not manifest-only")
    current = _child_baseline_mapping(document.get("current_runtime_boundary"), "static residency boundary")
    adapter = _child_baseline_mapping(current.get("full_runtime_adapter"), "static residency adapter")
    if adapter.get("id") is not None or _child_baseline_int(
        adapter.get("metal_dispatches"), "static residency adapter dispatches"
    ) != 0:
        raise DeepSeekV4GravityError("static residency receipt illegally registers a full runtime")
    if _child_baseline_bool(
        current.get("raw_content_addressed_stream_eviction_authorized"),
        "static residency raw stream eviction",
    ):
        raise DeepSeekV4GravityError("static residency receipt illegally authorizes raw stream eviction")
    observation = _child_baseline_mapping(
        document.get("bounded_layer4_route_observations"), "static residency layer-4 observation"
    )
    receipt = _child_baseline_mapping(observation.get("receipt"), "static residency latent receipt")
    if receipt.get("seal_sha256") != latent_binding.get("seal_sha256"):
        raise DeepSeekV4GravityError("static residency does not bind the supplied bounded latent receipt")
    if receipt.get("diagnostic_artifact_seal_sha256") != diagnostic_seal:
        raise DeepSeekV4GravityError("static residency binds a different diagnostic artifact")
    observation_scope = _child_baseline_mapping(observation.get("scope"), "static residency observation scope")
    if _child_baseline_int(observation_scope.get("selected_layer"), "static residency selected layer") != 4:
        raise DeepSeekV4GravityError("static residency observation is not layer 4")
    if _child_baseline_int(observation_scope.get("source_top_k"), "static residency top-k", minimum=1) != 6:
        raise DeepSeekV4GravityError("static residency observation is not source top-6")
    summary = _child_baseline_mapping(
        document.get("static_active_byte_summary"), "static residency active-byte summary"
    )
    body_bytes = _child_baseline_int(
        summary.get("body_selected_weight_logical_bytes_per_decode_token"),
        "static residency body logical bytes",
        minimum=1,
    )
    if summary.get("physical_active_bytes_per_token") != "NOT_MEASURED_NO_NATIVE_RUNTIME":
        raise DeepSeekV4GravityError("static residency illegally claims physical active bytes")
    packing = _child_baseline_mapping(
        document.get("static_packing_and_cache_candidates"), "static residency cache candidates"
    )
    unavailable = _child_baseline_mapping(
        document.get("unavailable_until_native_runtime"), "static residency unavailable metrics"
    )
    return document, {
        "receipt": binding,
        "scope": "STATIC_SOURCE_LAYOUT_CONTRACT_NOT_PHYSICAL_RUNTIME_MEASUREMENT",
        "geometry": copy.deepcopy(document.get("geometry")),
        "static_active_byte_summary": {
            "body_selected_weight_logical_bytes_per_decode_token": body_bytes,
            "always_selected_shared_attention_and_control_logical_bytes": summary.get(
                "always_selected_shared_attention_and_control_logical_bytes"
            ),
            "dense_lm_head_logical_source_bytes_per_decode_token_without_residency": summary.get(
                "dense_lm_head_logical_source_bytes_per_decode_token_without_residency"
            ),
            "physical_active_bytes_per_token": "NOT_MEASURED_NO_NATIVE_RUNTIME",
            "interpretation": summary.get("body_selected_weight_logical_bytes_interpretation"),
        },
        "bounded_layer4_route_observations": {
            "latent_receipt_seal_sha256": receipt.get("seal_sha256"),
            "total_source_forwards": observation_scope.get("total_source_forwards"),
            "selected_layer": 4,
            "source_top_k": 6,
            "frequency_ranked_experts": copy.deepcopy(observation.get("frequency_ranked_experts")),
            "transition_observations": copy.deepcopy(observation.get("transition_observations")),
        },
        "packing_and_cache_candidates": copy.deepcopy(packing),
        "unavailable_until_native_runtime": copy.deepcopy(unavailable),
    }


def _child_baseline_v3_device_copy_roofline(
    path_value: str | Path,
    *,
    full_seal: str,
    residency_binding: Mapping[str, Any],
    residency_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, binding = _child_baseline_v3_named_receipt(
        path_value,
        "metal_device_copy_roofline",
        filename=_CANONICAL_METAL_DEVICE_COPY_ROOFLINE_V1,
        schema="hawking.gravity.deepseek_v4.metal_device_copy_roofline.v1",
        status="PASS_REAL_M3_METAL_DEVICE_COPY_CEILING_NOT_MODEL_KERNEL_OR_TPS",
    )
    scope = _child_baseline_mapping(document.get("scope"), "device-copy roofline scope")
    for field in (
        "base_true_tps_measured",
        "deepseek_v4_weights_opened",
        "gravity_artifact_opened",
        "hcli_endpoint_started",
    ):
        _child_baseline_v3_require_false(document, ("scope", field), f"device-copy roofline {field}")
    for field in ("deepseek_v4_forward_tokens", "gpu_compute_dispatches", "gpu_model_kernels"):
        if _child_baseline_int(scope.get(field), f"device-copy roofline {field}") != 0:
            raise DeepSeekV4GravityError("device-copy roofline unexpectedly executed a model kernel")
    if _child_baseline_bool(scope.get("device_copy_only"), "device-copy-only") is not True:
        raise DeepSeekV4GravityError("device-copy roofline scope is not copy-only")
    metal = _child_baseline_mapping(document.get("metal"), "device-copy Metal evidence")
    if _child_baseline_bool(metal.get("fallback"), "device-copy roofline fallback"):
        raise DeepSeekV4GravityError("device-copy roofline reports fallback")
    if _child_baseline_int(
        metal.get("measured_gpu_compute_dispatches"), "device-copy roofline compute dispatches"
    ) != 0:
        raise DeepSeekV4GravityError("device-copy roofline illegally records compute dispatches")
    sizes = document.get("sizes")
    if not isinstance(sizes, list) or not sizes:
        raise DeepSeekV4GravityError("device-copy roofline has no measured size trials")
    for ordinal, size in enumerate(sizes, start=1):
        item = _child_baseline_mapping(size, f"device-copy size {ordinal}")
        if _child_baseline_int(item.get("measured_trials"), f"device-copy size {ordinal} trials", minimum=5) < 5:
            raise DeepSeekV4GravityError("device-copy roofline has too few clean trials")
        topology = _child_baseline_mapping(
            item.get("measured_topology"), f"device-copy size {ordinal} topology"
        )
        if _child_baseline_int(
            topology.get("gpu_compute_dispatches"), f"device-copy size {ordinal} compute dispatches"
        ) != 0:
            raise DeepSeekV4GravityError("device-copy roofline contains compute dispatches")
        if _child_baseline_bool(
            topology.get("accounting_reconciled"), f"device-copy size {ordinal} accounting"
        ) is not True:
            raise DeepSeekV4GravityError("device-copy roofline does not reconcile its topology")
    comparator = _child_baseline_mapping(
        document.get("static_dsv4f_source_layout_comparator"), "device-copy static comparator"
    )
    if comparator.get("full_manifest_seal_sha256") != full_seal:
        raise DeepSeekV4GravityError("device-copy comparator binds a different full stream")
    if comparator.get("receipt_seal_sha256") != residency_binding.get("seal_sha256"):
        raise DeepSeekV4GravityError("device-copy comparator does not bind the supplied residency receipt")
    static_bytes = _child_baseline_int(
        comparator.get("body_selected_weight_logical_bytes_per_decode_token"),
        "device-copy comparator static body bytes",
        minimum=1,
    )
    residency_bytes = _child_baseline_int(
        _child_baseline_mapping(
            residency_summary.get("static_active_byte_summary"), "residency summary active bytes"
        ).get("body_selected_weight_logical_bytes_per_decode_token"),
        "residency summary body bytes",
        minimum=1,
    )
    if static_bytes != residency_bytes:
        raise DeepSeekV4GravityError("device-copy comparator and residency contract disagree on static bytes")
    if comparator.get("physical_active_bytes_per_token") != "NOT_MEASURED_NO_NATIVE_RUNTIME":
        raise DeepSeekV4GravityError("device-copy comparator illegally claims physical active bytes")
    metrics = _child_baseline_mapping(comparator.get("comparator"), "device-copy comparator metrics")
    for field in (
        "best_median_payload_copy_gib_per_s",
        "best_median_read_plus_write_copy_traffic_gib_per_s",
        "ideal_body_only_ms_per_token_if_every_static_logical_byte_used_best_payload_copy_rate",
        "static_body_requirement_fraction_of_best_payload_copy_ceiling_at_100_tps",
    ):
        _child_baseline_number(metrics.get(field), f"device-copy comparator {field}", minimum=0.0)
    byte_reduction_factor = _child_baseline_number(
        metrics.get("static_body_requirement_fraction_of_best_payload_copy_ceiling_at_100_tps"),
        "device-copy comparator byte-side requirement fraction",
        minimum=0.0,
    )
    conditional_inference = {
        "condition": (
            "IF every static logical selected body byte is physically fetched for every decode token "
            "at the best measured private-buffer payload-copy ceiling"
        ),
        "static_body_logical_bytes_per_decode_token": static_bytes,
        "best_payload_copy_ceiling_gib_per_s": metrics.get("best_median_payload_copy_gib_per_s"),
        "conditional_ideal_body_only_ms_per_token": metrics.get(
            "ideal_body_only_ms_per_token_if_every_static_logical_byte_used_best_payload_copy_rate"
        ),
        "conditional_100_tps_byte_side_requirement_fraction": byte_reduction_factor,
        "smallest_viable_representation_direction": (
            f">= {byte_reduction_factor:.6g}x active-byte reduction, OR proven reuse/cache behavior "
            "that keeps physical bytes below the static logical selected-byte count by the same factor"
        ),
        "not_measured_runtime_traffic": True,
        "not_a_model_roofline_or_100_tps_claim": True,
        "sufficiency": "NOT_SUFFICIENT; compute, dependency depth, synchronization, cache misses, and full-runtime parity remain unmeasured",
    }
    return document, {
        "receipt": binding,
        "scope": "DEVICE_COPY_CEILING_ONLY_NOT_MODEL_KERNEL_OR_TPS",
        "device": metal.get("device"),
        "measured_sizes_mib": [
            _child_baseline_mapping(size, "device-copy size").get("copy_size_mib") for size in sizes
        ],
        "best_static_layout_comparator": copy.deepcopy(metrics),
        "conditional_byte_side_inference": conditional_inference,
        "physical_active_bytes_per_token": "NOT_MEASURED_NO_NATIVE_RUNTIME",
        "strict_interpretation": comparator.get("strict_interpretation"),
    }


def _child_baseline_v3_act_quant_oracle(
    path_value: str | Path,
    *,
    full_seal: str,
    reader_summary: Mapping[str, Any],
    residency_document: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, binding = _child_baseline_v3_named_receipt(
        path_value,
        "act_quant_wq_a_cpu_oracle",
        filename=_CANONICAL_ACT_QUANT_WQ_A_CPU_ORACLE_V2,
        schema="hawking.gravity.deepseek_v4.act_quant_fp8_wq_a_cpu_algorithm_oracle.v1",
        status="PASS_SOURCE_DERIVED_CPU_ALGORITHM_ORACLE_NOT_INDEPENDENT_SOURCE_RUNTIME_PARITY",
    )
    _child_baseline_v3_exact_seal(
        document, ("artifact", "manifest_seal_sha256"), full_seal, "act-quant CPU oracle"
    )
    artifact = _child_baseline_mapping(document.get("artifact"), "act-quant oracle artifact")
    source = _child_baseline_mapping(artifact.get("source"), "act-quant oracle source")
    if source.get("repository") != REPOSITORY or source.get("revision") != REVISION:
        raise DeepSeekV4GravityError("act-quant CPU oracle is not pinned to DeepSeek-V4-Flash")
    boundary = _child_baseline_mapping(document.get("execution_boundary"), "act-quant oracle boundary")
    for field in (
        "base_true_tps_measured",
        "full_model_forward",
        "full_model_loaded",
        "hcli_endpoint_started",
        "independently_source_runtime_parity",
        "source_runtime_executed",
    ):
        _child_baseline_v3_require_false(document, ("execution_boundary", field), f"act-quant {field}")
    for field in (
        "command_buffers",
        "cpu_visible_waits",
        "generated_tokens",
        "gpu_dispatches",
        "metal_allocations",
    ):
        if _child_baseline_int(boundary.get(field), f"act-quant {field}") != 0:
            raise DeepSeekV4GravityError("act-quant CPU oracle unexpectedly records runtime dispatch work")
    for field in ("not_independently_source_runtime_parity", "source_derived_algorithm_oracle"):
        if _child_baseline_bool(boundary.get(field), f"act-quant {field}") is not True:
            raise DeepSeekV4GravityError("act-quant CPU oracle crosses its algorithm-oracle boundary")
    input_record = _child_baseline_mapping(document.get("input"), "act-quant oracle input")
    if _child_baseline_bool(
        input_record.get("captured_from_model_forward"), "act-quant input captured from forward"
    ):
        raise DeepSeekV4GravityError("act-quant CPU oracle input was captured from a model forward")
    quant = _child_baseline_mapping(document.get("act_quant"), "act-quant oracle algorithm")
    if _child_baseline_bool(quant.get("source_derived"), "act-quant source-derived") is not True:
        raise DeepSeekV4GravityError("act-quant CPU oracle is not source-derived")
    if _child_baseline_int(quant.get("block_size"), "act-quant block size", minimum=1) != 128:
        raise DeepSeekV4GravityError("act-quant CPU oracle has an unexpected FP8 block size")
    assets = _child_baseline_mapping(
        document.get("source_algorithm_bindings"), "act-quant source algorithm bindings"
    ).get("official_assets_verified_by_admitted_full_stream_and_exact_anchor")
    assets = _child_baseline_mapping(assets, "act-quant official asset bindings")
    reader_assets = _child_baseline_mapping(
        reader_summary.get("pinned_codec_assets"), "reader pinned codec assets"
    )
    for short_name, reader_name in (
        ("inference/kernel.py", "inference_kernel_py_sha256"),
        ("inference/model.py", "inference_model_py_sha256"),
    ):
        if assets.get(short_name) != reader_assets.get(reader_name):
            raise DeepSeekV4GravityError("act-quant CPU oracle does not match admitted reader codec assets")
    residency_source = _child_baseline_mapping(
        residency_document.get("source_binding"), "residency source binding for act-quant oracle"
    )
    residency_metadata = _child_baseline_mapping(
        residency_source.get("metadata"), "residency metadata for act-quant oracle"
    )
    for short_name, metadata_name in (
        ("config.json", "config"),
        ("inference/config.json", "inference_config"),
    ):
        metadata = _child_baseline_mapping(
            residency_metadata.get(metadata_name), f"residency metadata {metadata_name}"
        )
        if assets.get(short_name) != metadata.get("sha256"):
            raise DeepSeekV4GravityError("act-quant CPU oracle does not match static residency metadata")
    gemv = _child_baseline_mapping(document.get("cpu_fp8_gemv"), "act-quant CPU FP8 GEMV")
    return document, {
        "receipt": binding,
        "scope": "SOURCE_DERIVED_CPU_ALGORITHM_ORACLE_NOT_INDEPENDENT_SOURCE_RUNTIME_PARITY",
        "operator": gemv.get("operator"),
        "shape": gemv.get("shape"),
        "act_quant": {
            "block_size": quant.get("block_size"),
            "activation_dtype": quant.get("activation_dtype"),
            "scale_dtype": quant.get("scale_dtype"),
            "scale_format": quant.get("scale_format"),
            "source_derived": True,
        },
        "output_fp32_le_sha256": gemv.get("output_fp32_le_sha256"),
        "full_runtime_or_source_forward_parity": False,
    }


def _child_baseline_v3_optional_source_forward_extension(
    path_value: str | Path | None,
    *,
    full_seal: str,
    act_quant_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a future GPU source-linear/prefix receipt without promoting it.

    There is intentionally no schema whitelist yet: it is an extension hook,
    not a backdoor to declare full source-forward parity before the explicit
    source-linear and layer-0-prefix contract has been reviewed.
    """

    if path_value is None:
        return {
            "status": "NOT_BOUND_OPTIONAL_SOURCE_FORWARD_EXTENSION_PENDING",
            "changes_any_full_runtime_or_BASE_TRUE_TPS_gate": False,
        }
    _, document, binding = _child_baseline_v3_read_sealed(
        path_value, "source_forward_extension_receipt"
    )
    if not str(binding["schema"]).startswith("hawking.gravity.deepseek_v4."):
        raise DeepSeekV4GravityError("source-forward extension must use a DeepSeek-V4 receipt schema")
    bound_paths = (
        ("artifact", "manifest_seal_sha256"),
        ("artifact", "full_stream_manifest_seal_sha256"),
        ("source_binding", "full_manifest_seal_sha256"),
        ("full_stream_manifest_seal_sha256",),
    )
    if not any(_hcli_live_scalar_at(document, candidate) == full_seal for candidate in bound_paths):
        raise DeepSeekV4GravityError(
            "source-forward extension must bind the same full-stream manifest seal"
        )
    model_linear_schema = "hawking.gravity.deepseek_v4.model_linear_fp8_act_quant_metal_component_parity.v1"
    if binding["schema"] == model_linear_schema:
        if binding["status"] != "PASS_REAL_METAL_MODEL_LINEAR_COMPONENT_PARITY_NOT_FULL_RUNTIME":
            raise DeepSeekV4GravityError("model-linear extension has an unexpected status")
        source = _child_baseline_mapping(document.get("source"), "model-linear extension source")
        if source.get("repository") != REPOSITORY or source.get("revision") != REVISION:
            raise DeepSeekV4GravityError("model-linear extension is not bound to the pinned source")
        scope = _child_baseline_mapping(document.get("scope"), "model-linear extension scope")
        for field in (
            "model_linear_component_only",
            "not_attention",
            "not_base_true_tps_measurement",
            "not_embedding",
            "not_full_model_forward",
            "not_full_model_load",
            "not_hcli_endpoint",
            "not_mhc",
            "not_registered_43_layer_runtime_adapter",
            "not_router_or_expert_execution",
            "not_token_execution_or_generation",
        ):
            if _child_baseline_bool(scope.get(field), f"model-linear extension scope.{field}") is not True:
                raise DeepSeekV4GravityError("model-linear extension crosses its component-only boundary")
        oracle = _child_baseline_mapping(
            document.get("canonical_cpu_oracle_v2"), "model-linear canonical CPU oracle"
        )
        if oracle.get("receipt_seal_sha256") != act_quant_binding.get("seal_sha256"):
            raise DeepSeekV4GravityError(
                "model-linear extension does not bind the supplied canonical act-quant CPU oracle v2"
            )
        if _child_baseline_bool(
            oracle.get("direct_cpu_oracle_recomputed_and_matches_sealed_v2"),
            "model-linear canonical CPU oracle recomputation",
        ) is not True:
            raise DeepSeekV4GravityError("model-linear extension lacks canonical CPU oracle recomputation")
        input_record = _child_baseline_mapping(document.get("input"), "model-linear extension input")
        if _child_baseline_bool(
            input_record.get("captured_from_model_forward"), "model-linear captured forward input"
        ):
            raise DeepSeekV4GravityError("model-linear extension input must not come from a model forward")
        metal = _child_baseline_mapping(document.get("metal"), "model-linear Metal evidence")
        if _child_baseline_bool(metal.get("fallback"), "model-linear extension fallback"):
            raise DeepSeekV4GravityError("model-linear extension reports fallback")
        if _child_baseline_int(metal.get("gpu_dispatches"), "model-linear extension GPU dispatches", minimum=1) < 1:
            raise DeepSeekV4GravityError("model-linear extension has no GPU dispatches")
        act_quant = _child_baseline_mapping(document.get("gpu_act_quant"), "model-linear GPU act-quant")
        projection = _child_baseline_mapping(
            document.get("gpu_fp8_weighted_projection"), "model-linear GPU FP8 projection"
        )
        if _child_baseline_bool(act_quant.get("fallback"), "model-linear act-quant fallback"):
            raise DeepSeekV4GravityError("model-linear GPU act-quant reports fallback")
        if _child_baseline_bool(projection.get("fallback"), "model-linear FP8 projection fallback"):
            raise DeepSeekV4GravityError("model-linear GPU FP8 projection reports fallback")
        selected_parity = _child_baseline_mapping(
            projection.get("selected_cpu_oracle_parity"), "model-linear selected CPU oracle parity"
        )
        if selected_parity.get("status") != "PASS":
            raise DeepSeekV4GravityError("model-linear extension does not pass its canonical CPU oracle parity")
        if _child_baseline_bool(
            projection.get("selected_output_bf16_hash_matches_cpu_oracle"),
            "model-linear selected BF16 hash parity",
        ) is not True:
            raise DeepSeekV4GravityError("model-linear extension does not match canonical BF16 output")
        return {
            "status": "BOUND_SOURCE_LINEAR_COMPONENT_CHECKPOINT_NOT_FULL_SOURCE_FORWARD",
            "receipt": binding,
            "component": scope.get("component"),
            "canonical_act_quant_cpu_oracle_v2_seal_sha256": oracle.get("receipt_seal_sha256"),
            "selected_kernel": projection.get("selected_kernel"),
            "gpu_dispatches": metal.get("gpu_dispatches"),
            "source_linear_component_parity": True,
            "full_source_forward_parity": False,
            "changes_any_full_runtime_or_BASE_TRUE_TPS_gate": False,
        }
    return {
        "status": "BOUND_UNPROMOTED_PENDING_EXPLICIT_SOURCE_FORWARD_CONTRACT",
        "receipt": binding,
        "changes_any_full_runtime_or_BASE_TRUE_TPS_gate": False,
        "full_source_forward_parity": False,
    }


def freeze_child_baseline_v3(
    *,
    prior_baseline_v2_dir: str | Path,
    fp4_metal_component_probe: str | Path,
    full_stream_reader_admission: str | Path,
    raw_weight_simdgroup_splitk_sweep: str | Path,
    bounded_latent_route_receipt: str | Path,
    static_expert_residency: str | Path,
    metal_device_copy_roofline: str | Path,
    act_quant_wq_a_cpu_oracle: str | Path,
    out_dir: str | Path,
    source_forward_extension_receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze a v3 child-body evidence revision without changing v1/v2.

    The new inputs advance *admission*, raw-weight component mechanics, a
    bounded layer-4 latent/route capture, static residency accounting, and a
    device-copy ceiling.  None is a 43-layer source forward, so this function
    intentionally cannot turn any full-runtime, parity, or BASE_TRUE_TPS gate
    green.  A later GPU source-linear/layer-0-prefix receipt may be bound by
    the optional extension argument, but is unpromoted until its exact
    contract is approved.
    """

    parent_dir, parent_documents, parent = _child_baseline_v3_parent_bundle(
        prior_baseline_v2_dir
    )
    full_seal = _child_baseline_digest(
        parent.get("full_stream_manifest_seal_sha256"), "v3 parent full stream seal"
    )
    diagnostic_seal = _child_baseline_digest(
        parent.get("diagnostic_manifest_seal_sha256"), "v3 parent diagnostic seal"
    )
    fp4_document, fp4_summary = _child_baseline_v3_fp4_probe(
        fp4_metal_component_probe, full_seal=full_seal
    )
    reader_document, reader_summary = _child_baseline_v3_reader_admission(
        full_stream_reader_admission, full_seal=full_seal
    )
    simd_document, simd_summary = _child_baseline_v3_simdgroup_sweep(
        raw_weight_simdgroup_splitk_sweep, full_seal=full_seal
    )
    latent_document, latent_summary = _child_baseline_v3_latent_routes(
        bounded_latent_route_receipt, diagnostic_seal=diagnostic_seal
    )
    residency_document, residency_summary = _child_baseline_v3_static_residency(
        static_expert_residency,
        full_seal=full_seal,
        diagnostic_seal=diagnostic_seal,
        latent_binding=latent_summary["receipt"],
    )
    roofline_document, roofline_summary = _child_baseline_v3_device_copy_roofline(
        metal_device_copy_roofline,
        full_seal=full_seal,
        residency_binding=residency_summary["receipt"],
        residency_summary=residency_summary,
    )
    act_quant_document, act_quant_summary = _child_baseline_v3_act_quant_oracle(
        act_quant_wq_a_cpu_oracle,
        full_seal=full_seal,
        reader_summary=reader_summary,
        residency_document=residency_document,
    )
    optional_extension = _child_baseline_v3_optional_source_forward_extension(
        source_forward_extension_receipt,
        full_seal=full_seal,
        act_quant_binding=act_quant_summary["receipt"],
    )

    # The values are deliberately referenced below only through summaries and
    # receipt seals.  Binding the locals makes accidental omission obvious to
    # static reviewers and preserves the checked reader/sweep/oracle inputs.
    _ = (fp4_document, reader_document, simd_document, latent_document, roofline_document, act_quant_document)

    advanced_receipts: dict[str, Any] = {
        "fp4_metal_component_probe": fp4_summary["receipt"],
        "full_stream_reader_admission": reader_summary["receipt"],
        "raw_weight_simdgroup_splitk_sweep_v2": simd_summary["receipt"],
        "bounded_latent_route_v2": latent_summary["receipt"],
        "static_expert_residency_v2": residency_summary["receipt"],
        "metal_device_copy_roofline_v1": roofline_summary["receipt"],
        "act_quant_wq_a_cpu_oracle_v2": act_quant_summary["receipt"],
    }
    if "receipt" in optional_extension:
        advanced_receipts["optional_source_forward_extension"] = optional_extension["receipt"]
    references = copy.deepcopy(parent["parent_evidence_bindings"])
    references["v3_extension"] = {
        "revision": "DSV4F_CHILD_BASELINE_V3",
        "prior_baseline_v2": {
            "path": str(parent_dir),
            "baseline_bundle_id": parent["baseline_bundle_id"],
            "evidence_bindings_sha256": parent["evidence_bindings_sha256"],
            "files": copy.deepcopy(parent["files"]),
        },
        "advanced_receipts": advanced_receipts,
        "optional_source_forward_extension": copy.deepcopy(optional_extension),
    }
    bundle_id = _sha256(_canonical(references))
    claim_boundary = {
        "full_43_layer_runtime": False,
        "source_cpu_parity": False,
        "numeric_parity_v2_1": False,
        "full_43_layer_metal_dispatch": False,
        "base_true_tps_eligible_metal_dispatch": False,
        "base_true_tps": False,
        "raw_weight_component_parity": True,
        "raw_weight_component_parity_is_not_source_forward_parity": True,
        "source_forward_parity": False,
        "source_forward_parity_gate": "NOT_PROVEN_NO_FULL_SOURCE_FORWARD",
        "component_only_fp8_metal_probe": True,
        "component_only_fp4_metal_probe": True,
        "source_derived_act_quant_cpu_oracle": True,
        "device_copy_ceiling_is_not_model_roofline": True,
        "optional_source_forward_extension_bound": "receipt" in optional_extension,
        "kimi_or_glm_donor_weights_present": False,
        "kimi_or_glm_training_performed": False,
        "direct_weight_transplant": False,
    }
    statuses = {
        "DSV4F_CHILD_BASELINE.json": "DSV4F_CHILD_BASELINE_V3_FROZEN_FULL_STREAM_RUNTIME_PENDING",
        "DSV4F_RUNTIME_PROFILE.json": "DSV4F_DIAGNOSTIC_CPU_RUNTIME_PROFILE_V3_FROZEN_FULL_RUNTIME_PENDING",
        "DSV4F_ROUTE_PROFILE.json": "DSV4F_LAYER4_ROUTE_PROFILE_V3_FROZEN_STATIC_RESIDENCY_NOT_FULL_RUNTIME",
        "DSV4F_LATENT_BRIDGE_CONTRACT.json": "DSV4F_FUTURE_BRIDGE_INTERFACES_V3_DECLARED_NO_DONOR_INHERITANCE",
        "DSV4F_TRANSPLANT_POINTS.json": "DSV4F_TRANSPLANT_POINTS_V3_FROZEN_SOURCE_BOUND_NO_WEIGHT_GRAFT",
        "DSV4F_100TPS_SCOREBOARD.json": "BASE_TRUE_TPS_NOT_REACHED_NOT_ELIGIBLE_ON_CURRENT_RUNTIME",
        "DSV4F_KERNEL_REGISTRY.json": "DSV4F_KERNEL_REGISTRY_V3_RAW_COMPONENT_EVIDENCE_NO_NATIVE_43_LAYER_KERNEL",
        "DSV4F_ROOFLINE.json": "DSV4F_STATIC_LAYOUT_AND_DEVICE_COPY_COMPARATOR_V3_NOT_MODEL_ROOFLINE",
    }

    documents: dict[str, dict[str, Any]] = {}
    for filename in _CHILD_BASELINE_FILENAMES:
        document = copy.deepcopy(parent_documents[filename])
        document.pop("seal_sha256", None)
        parent_schema = _CHILD_BASELINE_V2_PARENT_SCHEMAS[filename]
        if not parent_schema.endswith(".v1"):
            raise DeepSeekV4GravityError("v3 parent schema mapping must end in v1")
        document["schema"] = parent_schema[:-1] + "3"
        document["status"] = statuses[filename]
        document["baseline_bundle_id"] = bundle_id
        document["baseline_revision"] = {
            "revision": "v3",
            "prior_revision": "v2",
            "prior_baseline_bundle_id": parent["baseline_bundle_id"],
            "mode": "ADDITIVE_SEALED_EVIDENCE_REVISION",
            "v1_v2_mutated": False,
        }
        document["evidence_bindings"] = copy.deepcopy(references)
        document["claim_boundary"] = copy.deepcopy(claim_boundary)
        documents[filename] = document

    child = documents["DSV4F_CHILD_BASELINE.json"]
    child["v3_full_stream_readiness"] = {
        "reader_admission": reader_summary,
        "raw_weight_component_parity": {
            "fp8_parent_v2_receipt": _child_baseline_mapping(
                parent["parent_evidence_bindings"], "parent v2 evidence"
            )["receipts"]["fp8_metal_component_probe"],
            "fp4": fp4_summary,
            "simdgroup_splitk_sweep": simd_summary,
        },
        "act_quant_cpu_oracle": act_quant_summary,
        "source_forward_parity": {
            "status": "NOT_PROVEN",
            "distinction": "raw-weight component parity and a source-derived CPU algorithm oracle do not execute source activations through a full source forward",
            "optional_extension": optional_extension,
        },
    }
    frozen_metrics = _child_baseline_mapping(child.get("frozen_metrics"), "v3 child frozen metrics")
    frozen_metrics["component_only_fp4_metal_gpu_dispatches"] = fp4_summary["metal"]["gpu_dispatches"]
    frozen_metrics["source_forward_parity"] = "NOT_PROVEN"
    frozen_metrics["base_true_tps"] = {"value": None, "status": "WITHHELD_NOT_ELIGIBLE"}

    runtime = documents["DSV4F_RUNTIME_PROFILE.json"]
    runtime["v3_runtime_admission_and_component_boundary"] = {
        "reader_admission": reader_summary,
        "fp4_raw_weight_component": fp4_summary,
        "raw_simdgroup_splitk": simd_summary,
        "act_quant_cpu_oracle": act_quant_summary,
        "source_forward_parity": "NOT_PROVEN; no source-forward activation sequence or registered 43-layer runtime",
        "full_43_layer_runtime": "NOT_REGISTERED",
    }
    runtime["gpu_and_hardware_counter_boundary"]["v3_raw_component_dispatches_not_token_dispatches"] = {
        "fp4_component_gpu_dispatches": fp4_summary["metal"]["gpu_dispatches"],
        "raw_sweep_gpu_dispatches": simd_summary["aggregate_real_gpu_dispatches"],
        "interpretation": "component dispatch counts are not complete-token dispatch counts",
    }

    route = documents["DSV4F_ROUTE_PROFILE.json"]
    route["v3_bounded_latent_route_capture"] = latent_summary
    route["v3_static_expert_residency"] = residency_summary
    route["unavailable_full_runtime_residency_metrics"].update(
        {
            "physical_active_bytes_per_token": "NOT_MEASURED_NO_NATIVE_RUNTIME",
            "static_source_layout_contract": "BOUND_V2_NOT_A_PHYSICAL_CACHE_MEASUREMENT",
            "layer4_route_capture": "BOUND_V2_DIAGNOSTIC_ONLY_NOT_FULL_MODEL_DISTRIBUTION",
        }
    )

    bridge = documents["DSV4F_LATENT_BRIDGE_CONTRACT.json"]
    bridge["available_bounded_evidence"] = {
        "component_trace_receipt": _child_baseline_mapping(
            parent["parent_evidence_bindings"], "parent v2 evidence"
        )["receipts"]["component_trace"],
        "bounded_latent_route_capture_v2": latent_summary,
        "static_residency_contract_v2": {
            "receipt": residency_summary["receipt"],
            "scope": residency_summary["scope"],
        },
        "raw_hidden_states": "NOT_RETAINED",
        "small_trace_shards": "BOUNDED_LAYER4_CPU_DIAGNOSTIC_ONLY_NOT_A_43_LAYER_RUNTIME_TRACE",
        "donor_data_or_training": "NOT_PRESENT",
    }

    transplant = documents["DSV4F_TRANSPLANT_POINTS.json"]
    capture_status = {
        "pre_norm_hidden_state": "BOUNDED_LAYER4_SUMMARY_ONLY",
        "post_attention_hidden_state": "BOUNDED_LAYER4_SUMMARY_ONLY",
        "pre_router_hidden_state": "BOUNDED_LAYER4_SUMMARY_ONLY",
        "router_logits": "BOUNDED_LAYER4_SUMMARY_ONLY",
        "selected_expert_ids": "BOUNDED_LAYER4_TOP6_AGGREGATE_ONLY",
        "route_probabilities_and_margins": "NOT_RETAINED_BY_BOUNDARY",
        "post_moe_hidden_state": "BOUNDED_LAYER4_SUMMARY_ONLY",
        "mhc_state": "BOUNDED_LAYER4_SUMMARY_ONLY",
        "attention_index_state": "BOUNDED_LAYER4_SUMMARY_ONLY",
        "final_hidden_state": "BOUNDED_LAYER4_SUMMARY_ONLY",
        "lm_head_logits": "BOUNDED_LAYER4_SUMMARY_OR_HASH_ONLY",
        "hcli_tool_action_decision": "BOUND_HCLI_METADATA_ONLY",
    }
    points = transplant.get("points")
    if not isinstance(points, list):
        raise DeepSeekV4GravityError("v2 transplant points are malformed")
    for point in points:
        item = _child_baseline_mapping(point, "v2 transplant point")
        name = item.get("name")
        item["v3_bounded_layer4_capture_status"] = capture_status.get(
            name, "NOT_CAPTURED_BY_V3_BOUNDARY"
        )
        item["v3_capture_receipt_seal_sha256"] = latent_summary["receipt"]["seal_sha256"]
    transplant["v3_capture_boundary"] = {
        "bounded_latent_route_capture": latent_summary["receipt"],
        "static_expert_residency": residency_summary["receipt"],
        "full_43_layer_source_forward": "NOT_CAPTURED_NOT_REGISTERED",
        "raw_hidden_state_retention": "DISALLOWED",
    }

    scoreboard = documents["DSV4F_100TPS_SCOREBOARD.json"]
    metrics = _child_baseline_mapping(scoreboard.get("metrics"), "v3 TPS scoreboard metrics")
    metrics["BASE_TRUE_TPS"] = {"value": None, "status": "WITHHELD_NOT_ELIGIBLE"}
    metrics["raw_weight_component_parity"] = {
        "fp4": "PASS_COMPONENT_ONLY",
        "fp8": "PASS_COMPONENT_ONLY_PARENT_V2",
        "source_forward_parity": "NOT_PROVEN",
    }
    metrics["device_copy_ceiling"] = "NOT_A_MODEL_TPS_MEASUREMENT"
    scoreboard["v3_component_and_roofline_evidence"] = {
        "fp4_raw_weight_component": fp4_summary,
        "raw_simdgroup_splitk": simd_summary,
        "device_copy_comparator": roofline_summary,
        "static_residency": {
            "receipt": residency_summary["receipt"],
            "physical_active_bytes_per_token": "NOT_MEASURED_NO_NATIVE_RUNTIME",
        },
        "BASE_TRUE_TPS": "WITHHELD; no eligible 8K/512-token 43-layer native run",
    }

    kernels = documents["DSV4F_KERNEL_REGISTRY.json"]
    kernels["v3_raw_component_kernel_evidence"] = {
        "fp4_authority_probe": fp4_summary,
        "raw_weight_simdgroup_splitk_sweep": simd_summary,
        "act_quant_cpu_oracle": act_quant_summary,
        "promotion_boundary": "NO_COMPONENT_RESULT_IS_PROMOTED_TO_A_FULL_43_LAYER_RUNTIME_KERNEL",
        "source_forward_parity": "NOT_PROVEN",
    }
    kernels["command_topology"]["v3_raw_component_sweep"] = {
        "command_buffers": simd_summary["aggregate_command_buffers"],
        "cpu_visible_waits": simd_summary["aggregate_cpu_visible_waits"],
        "gpu_dispatches": simd_summary["aggregate_real_gpu_dispatches"],
        "interpretation": "raw component sweep topology only; not token graph topology",
    }

    roofline = documents["DSV4F_ROOFLINE.json"]
    roofline["v3_static_layout_and_device_copy_comparator"] = {
        "static_expert_residency": residency_summary,
        "device_copy_roofline": roofline_summary,
        "interpretation": "static logical source-layout bytes compared to a copy-only ceiling; this is neither measured model traffic nor a model roofline",
        "conditional_representation_direction": roofline_summary["conditional_byte_side_inference"],
    }
    full_geometry = _child_baseline_mapping(roofline.get("full_geometry_roofline"), "v3 full geometry roofline")
    full_geometry.update(
        {
            "status": "NOT_MEASURED_NO_REGISTERED_43_LAYER_NATIVE_RUNTIME",
            "active_bytes_per_token": None,
            "physical_active_bytes_per_token": "NOT_MEASURED_NO_NATIVE_RUNTIME",
            "static_layout_logical_selected_weight_bytes_per_decode_token": residency_summary[
                "static_active_byte_summary"
            ]["body_selected_weight_logical_bytes_per_decode_token"],
            "device_copy_comparator_bound": True,
        }
    )

    for filename, document in tuple(documents.items()):
        documents[filename] = seal(document)
    target = _child_baseline_v3_output_directory(out_dir, parent_baseline=parent_dir)
    _floor_check(
        _child_baseline_existing_parent(target),
        MIN_FREE_FLOOR_BYTES,
        8 * 1024**2,
        "freeze-child-baseline-v3 output bundle",
    )
    for filename, document in documents.items():
        destination = target / filename
        if not destination.exists():
            continue
        _regular_file(destination, f"existing v3 baseline output {filename}")
        if destination.read_bytes() != _canonical(document) + b"\n":
            raise DeepSeekV4GravityError(
                f"refusing to overwrite a different existing frozen v3 baseline output: {filename}"
            )
    for filename in _CHILD_BASELINE_FILENAMES:
        _atomic_json(target / filename, documents[filename])
    return seal(
        {
            "schema": "hawking.gravity.deepseek_v4.child_baseline_freeze.v3",
            "status": "DSV4F_CHILD_BASELINE_V3_BUNDLE_SEALED",
            "baseline_bundle_id": bundle_id,
            "prior_baseline_v2_bundle_id": parent["baseline_bundle_id"],
            "out_dir": str(target),
            "files": [
                {"name": filename, "seal_sha256": documents[filename]["seal_sha256"]}
                for filename in _CHILD_BASELINE_FILENAMES
            ],
            "full_stream_manifest_seal_sha256": full_seal,
            "diagnostic_manifest_seal_sha256": diagnostic_seal,
            "full_43_layer_runtime": False,
            "base_true_tps": False,
        }
    )


class _DiagnosticTokenProfiler:
    """Bounded, CPU-only accounting for one real diagnostic forward.

    The counters below are deliberately logical tensor/operation estimates,
    not hardware performance-counter claims.  The one-layer NumPy adapter has
    no GPU command queue, so every GPU/dispatch field is reported explicitly
    as unavailable or zero instead of inferred from CPU work.
    """

    def __init__(
        self,
        *,
        phase: str,
        token_ordinal: int,
        position: int,
        tokenizer_allocation: Mapping[str, Any] | None = None,
    ) -> None:
        self.phase = phase
        self.token_ordinal = token_ordinal
        self.position = position
        self._stages: dict[str, dict[str, Any]] = {
            name: {
                "stage": name,
                "label": _COMPLETE_TOKEN_PROFILE_STAGE_LABELS[name],
                "execution_status": "not_reached",
                "cpu_duration_ms": 0.0,
                "cpu_wall_elapsed_ms": 0.0,
                "gpu_duration_ms": 0.0,
                "gpu_duration_status": "not_available_cpu_numpy_diagnostic",
                "occupancy": None,
                "occupancy_status": "not_available_cpu_numpy_diagnostic",
                "effective_bandwidth_bytes_per_second": None,
                "effective_bandwidth_status": "not_available_cpu_numpy_diagnostic",
                "bytes_read_estimate": 0,
                "bytes_written_estimate": 0,
                "fp_operations_estimate": 0,
                "integer_bit_operations_estimate": 0,
                "dispatches": 0,
                "command_buffers": 0,
                "waits": 0,
                "fallback": _DIAGNOSTIC_CPU_FALLBACK,
                "notes": [],
            }
            for name in _COMPLETE_TOKEN_PROFILE_STAGES
        }
        self._external_wall_ms = 0.0
        self._external_cpu_ms = 0.0
        endpoint = self._stages["endpoint_hcli_streaming"]
        endpoint["execution_status"] = "unavailable_in_inprocess_profile"
        endpoint["notes"].append(
            "The profile invokes the runtime directly; no HTTP response or HCLI stream fragment exists to time."
        )
        if tokenizer_allocation is not None:
            self.add_manual(
                "tokenizer_template",
                cpu_duration_ms=float(tokenizer_allocation.get("cpu_duration_ms", 0.0)),
                cpu_wall_elapsed_ms=float(tokenizer_allocation.get("cpu_wall_elapsed_ms", 0.0)),
                bytes_read_estimate=int(tokenizer_allocation.get("bytes_read_estimate", 0)),
                bytes_written_estimate=int(tokenizer_allocation.get("bytes_written_estimate", 0)),
                execution_status=str(
                    tokenizer_allocation.get(
                        "execution_status", "executed_source_tokenizer_raw_prompt"
                    )
                ),
                note=str(
                    tokenizer_allocation.get(
                        "note",
                        "Tokenization is apportioned across input-token forwards; no Jinja chat template is executed.",
                    )
                ),
                include_in_total=True,
            )
        else:
            tokenizer = self._stages["tokenizer_template"]
            tokenizer["execution_status"] = "not_applicable_decode_forward"
            tokenizer["notes"].append(
                "This forward consumes an already sampled token ID; no prompt tokenization or chat-template rendering occurs."
            )

    def _entry(self, stage: str) -> dict[str, Any]:
        try:
            return self._stages[stage]
        except KeyError as exc:
            raise DeepSeekV4GravityError(f"unknown complete-token profile stage {stage}") from exc

    @contextmanager
    def measure(self, stage: str, *, note: str | None = None) -> Iterator[None]:
        entry = self._entry(stage)
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            yield
        finally:
            entry["execution_status"] = "executed"
            entry["cpu_duration_ms"] += (time.process_time() - cpu_started) * 1_000.0
            entry["cpu_wall_elapsed_ms"] += (time.perf_counter() - wall_started) * 1_000.0
            if note is not None and note not in entry["notes"]:
                entry["notes"].append(note)

    def add_manual(
        self,
        stage: str,
        *,
        cpu_duration_ms: float,
        cpu_wall_elapsed_ms: float,
        bytes_read_estimate: int = 0,
        bytes_written_estimate: int = 0,
        fp_operations_estimate: int = 0,
        integer_bit_operations_estimate: int = 0,
        execution_status: str = "executed",
        note: str | None = None,
        include_in_total: bool = False,
    ) -> None:
        entry = self._entry(stage)
        entry["execution_status"] = execution_status
        entry["cpu_duration_ms"] += max(0.0, cpu_duration_ms)
        entry["cpu_wall_elapsed_ms"] += max(0.0, cpu_wall_elapsed_ms)
        entry["bytes_read_estimate"] += max(0, bytes_read_estimate)
        entry["bytes_written_estimate"] += max(0, bytes_written_estimate)
        entry["fp_operations_estimate"] += max(0, fp_operations_estimate)
        entry["integer_bit_operations_estimate"] += max(0, integer_bit_operations_estimate)
        if note is not None and note not in entry["notes"]:
            entry["notes"].append(note)
        if include_in_total:
            self._external_wall_ms += max(0.0, cpu_wall_elapsed_ms)
            self._external_cpu_ms += max(0.0, cpu_duration_ms)

    def add_estimate(
        self,
        stage: str,
        *,
        bytes_read_estimate: int = 0,
        bytes_written_estimate: int = 0,
        fp_operations_estimate: int = 0,
        integer_bit_operations_estimate: int = 0,
        note: str | None = None,
    ) -> None:
        entry = self._entry(stage)
        entry["bytes_read_estimate"] += max(0, int(bytes_read_estimate))
        entry["bytes_written_estimate"] += max(0, int(bytes_written_estimate))
        entry["fp_operations_estimate"] += max(0, int(fp_operations_estimate))
        entry["integer_bit_operations_estimate"] += max(0, int(integer_bit_operations_estimate))
        if note is not None and note not in entry["notes"]:
            entry["notes"].append(note)

    def add_metadata(self, stage: str, **metadata: Any) -> None:
        entry = self._entry(stage)
        entry.setdefault("metadata", {}).update(metadata)

    def finish(self, *, forward_wall_elapsed_ms: float, forward_cpu_duration_ms: float) -> dict[str, Any]:
        """Close accounting without creating an unexplained timing bucket."""

        observed_wall = max(0.0, forward_wall_elapsed_ms) + self._external_wall_ms
        observed_cpu = max(0.0, forward_cpu_duration_ms) + self._external_cpu_ms
        non_bookkeeping_wall = sum(
            float(entry["cpu_wall_elapsed_ms"])
            for name, entry in self._stages.items()
            if name != "runtime_bookkeeping"
        )
        non_bookkeeping_cpu = sum(
            float(entry["cpu_duration_ms"])
            for name, entry in self._stages.items()
            if name != "runtime_bookkeeping"
        )
        # ``runtime_bookkeeping`` may already hold a measured trace-hashing
        # interval.  Reconcile only the remaining gap, never that measured
        # interval again, so stage totals remain additive rather than double
        # counting trace construction.
        measured_bookkeeping = self._stages["runtime_bookkeeping"]
        measured_bookkeeping_wall = float(measured_bookkeeping["cpu_wall_elapsed_ms"])
        measured_bookkeeping_cpu = float(measured_bookkeeping["cpu_duration_ms"])
        before_reconciliation_wall = non_bookkeeping_wall + measured_bookkeeping_wall
        before_reconciliation_cpu = non_bookkeeping_cpu + measured_bookkeeping_cpu
        wall_residual = max(0.0, observed_wall - before_reconciliation_wall)
        cpu_residual = max(0.0, observed_cpu - before_reconciliation_cpu)
        self.add_manual(
            "runtime_bookkeeping",
            cpu_duration_ms=cpu_residual,
            cpu_wall_elapsed_ms=wall_residual,
            execution_status="executed_timing_reconciliation",
            note="Explicit reconciliation for Python control flow, trace construction, and profiler boundaries.",
        )
        return {
            "schema": "hawking.gravity.deepseek_v4.diagnostic_complete_token.v1",
            "phase": self.phase,
            "token_ordinal": self.token_ordinal,
            "runtime_position": self.position,
            "device": "cpu_numpy_diagnostic",
            "diagnostic_only": True,
            "stage_metrics": [self._stages[name] for name in _COMPLETE_TOKEN_PROFILE_STAGES],
            "timing_accounting": {
                "observed_complete_token_wall_elapsed_ms": observed_wall,
                "observed_complete_token_cpu_duration_ms": observed_cpu,
                "stage_wall_elapsed_before_reconciliation_ms": before_reconciliation_wall,
                "stage_cpu_duration_before_reconciliation_ms": before_reconciliation_cpu,
                "runtime_bookkeeping_measured_wall_elapsed_ms": measured_bookkeeping_wall,
                "runtime_bookkeeping_measured_cpu_duration_ms": measured_bookkeeping_cpu,
                "runtime_bookkeeping_wall_elapsed_ms": wall_residual,
                "runtime_bookkeeping_cpu_duration_ms": cpu_residual,
                "timer_overlap_wall_elapsed_ms": max(0.0, before_reconciliation_wall - observed_wall),
                "timer_overlap_cpu_duration_ms": max(0.0, before_reconciliation_cpu - observed_cpu),
                "unexplained_other_wall_elapsed_ms": 0.0,
                "unexplained_other_cpu_duration_ms": 0.0,
                "status": "PASS_ALL_TIME_EXPLICITLY_NAMED",
            },
        }


def _profile_scope(
    profiler: _DiagnosticTokenProfiler | None, stage: str, *, note: str | None = None
) -> Any:
    return profiler.measure(stage, note=note) if profiler is not None else nullcontext()


def _rmsnorm(values: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if values.ndim != 1 or weight.ndim != 1 or values.shape != weight.shape:
        raise DeepSeekV4GravityError("RMSNorm input/weight geometry mismatch")
    return (values * np.reciprocal(np.sqrt(np.mean(values * values, dtype=np.float64) + eps)) * weight).astype(
        np.float32, copy=False
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp = np.exp(values[~positive])
    result[~positive] = exp / (1.0 + exp)
    return result


def _softplus(values: np.ndarray) -> np.ndarray:
    return (np.maximum(values, 0.0) + np.log1p(np.exp(-np.abs(values)))).astype(np.float32)


def _hadamard(values: np.ndarray) -> np.ndarray:
    """Deterministic normalized Walsh-Hadamard fallback used by the index path."""

    width = values.shape[-1]
    if width <= 0 or width & (width - 1):
        raise DeepSeekV4GravityError("Hadamard diagnostic fallback requires a power-of-two width")
    out = values.astype(np.float32, copy=True)
    flat = out.reshape(-1, width)
    step = 1
    while step < width:
        blocks = flat.reshape(-1, width // (step * 2), step * 2)
        left = blocks[:, :, :step].copy()
        right = blocks[:, :, step:].copy()
        blocks[:, :, :step] = left + right
        blocks[:, :, step:] = left - right
        step *= 2
    return out * np.float32(width ** -0.5)


def _rope(values: np.ndarray, position: int, rope_dim: int, theta: float, *, inverse: bool = False) -> np.ndarray:
    """Apply the source's interleaved RoPE convention to a final feature span."""

    if rope_dim <= 0 or rope_dim % 2 or values.shape[-1] < rope_dim:
        raise DeepSeekV4GravityError("invalid RoPE geometry")
    out = values.astype(np.float32, copy=True)
    pairs = out[..., -rope_dim:].reshape(*out.shape[:-1], rope_dim // 2, 2)
    frequencies = 1.0 / (theta ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim))
    phase = np.float32(position) * frequencies
    cosine, sine = np.cos(phase), np.sin(phase)
    if inverse:
        sine = -sine
    left, right = pairs[..., 0].copy(), pairs[..., 1].copy()
    pairs[..., 0] = left * cosine - right * sine
    pairs[..., 1] = left * sine + right * cosine
    return out


class _CompressorState:
    """CPU diagnostic state for V4's ratio-four gated KV compression."""

    def __init__(self, runtime: "DeepSeekV4DiagnosticRuntime", prefix: str, head_dim: int, *, rotate: bool) -> None:
        self.runtime = runtime
        self.prefix = prefix
        self.head_dim = head_dim
        self.rotate = rotate
        self.ratio = 4
        self.coff = 2
        self.kv_state = np.zeros((self.coff * self.ratio, self.coff * head_dim), dtype=np.float32)
        self.score_state = np.full_like(self.kv_state, -np.inf)
        self.cache: list[np.ndarray] = []

    def step(self, values: np.ndarray, position: int) -> np.ndarray | None:
        runtime = self.runtime
        kv = runtime._matvec(f"{self.prefix}.wkv.weight", values)
        score = runtime._matvec(f"{self.prefix}.wgate.weight", values)
        ape = runtime._matrix(f"{self.prefix}.ape")
        if ape.shape != (self.ratio, self.coff * self.head_dim):
            raise DeepSeekV4GravityError(f"{self.prefix}: compressor APE geometry changed")
        slot = position % self.ratio
        offset = self.ratio
        self.kv_state[offset + slot] = kv
        self.score_state[offset + slot] = score + ape[slot]
        if (position + 1) % self.ratio:
            return None
        # This follows the decode phase of the official overlap compressor.
        kv_window = np.concatenate((self.kv_state[: self.ratio, : self.head_dim], self.kv_state[self.ratio :, self.head_dim :]), axis=0)
        score_window = np.concatenate((self.score_state[: self.ratio, : self.head_dim], self.score_state[self.ratio :, self.head_dim :]), axis=0)
        maximum = np.max(score_window, axis=0, keepdims=True)
        weights = np.exp(score_window - maximum)
        weights /= np.sum(weights, axis=0, keepdims=True)
        compressed = np.sum(kv_window * weights, axis=0)
        self.kv_state[: self.ratio] = self.kv_state[self.ratio :]
        self.score_state[: self.ratio] = self.score_state[self.ratio :]
        norm = runtime._vector(f"{self.prefix}.norm.weight")
        compressed = _rmsnorm(compressed, norm, runtime.norm_eps)
        compressed = _rope(compressed, position, runtime.rope_head_dim, runtime.compress_rope_theta)
        if self.rotate:
            compressed = _hadamard(compressed)
        self.cache.append(compressed)
        return compressed


def _component_execution_trace(
    *,
    token_id: int,
    hc_mult: int,
    hc_iters: int,
    attention: Mapping[str, Any],
    moe: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe the components physically reached by one diagnostic forward.

    This is deliberately execution evidence, not a parity claim.  It is
    embedded in every generated-token trace and therefore covered by the
    generation receipt's seal without changing a sealed artifact manifest.
    """

    routes_raw = moe.get("selected_route_ids")
    if not isinstance(routes_raw, list) or not all(isinstance(route, int) for route in routes_raw):
        raise DeepSeekV4GravityError("diagnostic MoE execution did not report concrete route IDs")
    routes = list(routes_raw)
    if not bool(moe.get("shared_expert_executed")) or int(moe.get("routed_expert_count", -1)) != len(routes):
        raise DeepSeekV4GravityError("diagnostic MoE execution trace is incomplete")
    if not bool(attention.get("executed")):
        raise DeepSeekV4GravityError("diagnostic attention execution trace is incomplete")
    return {
        "schema": "hawking.gravity.deepseek_v4.diagnostic_component_execution.v1",
        "diagnostic_only": True,
        "embedding": {
            "executed": True,
            "tensor": "embed.weight",
            "source_token_id": int(token_id),
        },
        "hc_attention": {
            "executed": True,
            "hc_mult": int(hc_mult),
            "sinkhorn_iterations": int(hc_iters),
            "control_tensors": [
                "layers.4.hc_attn_fn",
                "layers.4.hc_attn_scale",
                "layers.4.hc_attn_base",
            ],
        },
        "attention": dict(attention),
        "router": {
            "executed": True,
            "kind": "score_router",
            "selected_route_ids": routes,
            "selected_route_count": len(routes),
        },
        "shared_expert": {
            "executed": True,
            "prefix": "layers.4.ffn.shared_experts",
            "weight_representation": "native_fp8_e4m3fn_e8m0",
        },
        "routed_experts": {
            "executed": True,
            "selected_route_ids": routes,
            "selected_route_count": len(routes),
            "weight_representation": "native_fp4_e2m1fn_x2_e8m0",
        },
        "hc_ffn": {
            "executed": True,
            "hc_mult": int(hc_mult),
            "sinkhorn_iterations": int(hc_iters),
            "control_tensors": [
                "layers.4.hc_ffn_fn",
                "layers.4.hc_ffn_scale",
                "layers.4.hc_ffn_base",
            ],
        },
        "head": {
            "executed": True,
            "tensors": ["hc_head_fn", "hc_head_scale", "hc_head_base", "norm.weight", "head.weight"],
        },
        "decoder_mode": {
            "native_weight_decode": "fp4_e2m1fn_x2_and_fp8_e4m3fn_e8m0_to_f32",
            "linear_execution": "cpu_numpy_float32_matvec",
        },
        "activation_qat": False,
        "numeric_parity_v2_1": "not_proven",
        "metal_dispatches": 0,
    }


class DeepSeekV4DiagnosticRuntime:
    """Load and execute the sealed, full-width layer-4 diagnostic artifact.

    This is a real source-derived V4 component execution path, but it is not a
    Numeric Parity V2.1 authority or a Metal runtime.  The source's native
    *weight* formats are decoded exactly through ``deepseek_v4_native_codec``;
    activation QAT and TileLang kernels are intentionally represented as an
    explicit CPU fallback rather than emulated as a parity pass.
    """

    def __init__(self, artifact_dir: str | Path) -> None:
        started = time.perf_counter()
        self.artifact = _absolute(artifact_dir, "artifact_dir")
        self.manifest = load_manifest(self.artifact)
        raw_tensors = self.manifest.get("tensors")
        if not isinstance(raw_tensors, Mapping):
            raise DeepSeekV4GravityError("sealed artifact lacks a tensor map")
        self.store = SegmentedStore(self.artifact, raw_tensors)
        architecture = self.manifest.get("architecture")
        if not isinstance(architecture, Mapping):
            raise DeepSeekV4GravityError("artifact lacks architecture binding")
        config = architecture.get("inference_config")
        if not isinstance(config, Mapping):
            raise DeepSeekV4GravityError("artifact lacks source inference configuration")
        self.config = dict(config)
        self.dim = int(config["dim"])
        self.vocab_size = int(config["vocab_size"])
        self.head_dim = int(config["head_dim"])
        self.n_heads = int(config["n_heads"])
        self.rope_head_dim = int(config["rope_head_dim"])
        self.q_rank = int(config["q_lora_rank"])
        self.o_groups = int(config["o_groups"])
        self.o_rank = int(config["o_lora_rank"])
        self.n_experts = int(config["n_routed_experts"])
        self.topk = int(config["n_activated_experts"])
        self.moe_dim = int(config["moe_inter_dim"])
        self.route_scale = float(config["route_scale"])
        self.swiglu_limit = float(config["swiglu_limit"])
        self.window_size = int(config["window_size"])
        self.hc_mult = int(config["hc_mult"])
        self.hc_iters = int(config["hc_sinkhorn_iters"])
        self.norm_eps = float(config.get("norm_eps", 1e-6))
        self.hc_eps = float(config.get("hc_eps", 1e-6))
        self.rope_theta = float(config.get("rope_theta", 10000.0))
        self.compress_rope_theta = float(config.get("compress_rope_theta", self.rope_theta))
        if self.dim != 4096 or self.hc_mult != 4 or self.topk != 6 or self.n_experts != 256:
            raise DeepSeekV4GravityError("artifact source geometry is not the pinned V4 diagnostic geometry")
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise DeepSeekV4GravityError(
                "tokenizers==0.22.2 is required to preserve source tokenizer IDs"
            ) from exc
        self.tokenizer = Tokenizer.from_file(str(self.artifact / "metadata" / "tokenizer.json"))
        tokenizer_config = json.loads((self.artifact / "metadata" / "tokenizer_config.json").read_text(encoding="utf-8"))
        eos = tokenizer_config.get("eos_token")
        self.eos_id = self.tokenizer.token_to_id(eos.get("content")) if isinstance(eos, Mapping) else None
        source_template = tokenizer_config.get("chat_template")
        self.source_chat_template = source_template if isinstance(source_template, str) and source_template.strip() else None
        self.chat_template_status = (
            "SOURCE_TEMPLATE_PRESENT_BUT_NOT_EXECUTED_BY_DIAGNOSTIC"
            if self.source_chat_template is not None
            else "SOURCE_TOKENIZER_CONFIG_HAS_NO_CHAT_TEMPLATE_ROLE_TAG_FALLBACK"
        )
        self._core: dict[str, np.ndarray] = {}
        self._expert_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._expert_cache_bytes = 0
        self._expert_cache_limit = 6 * 1024**3
        self._lock = Lock()
        self.load_ms = (time.perf_counter() - started) * 1_000.0
        self.reset()

    def reset(self) -> None:
        self.position = 0
        self.window_kv = np.zeros((self.window_size, self.head_dim), dtype=np.float32)
        self.main_compressor = _CompressorState(self, "layers.4.attn.compressor", self.head_dim, rotate=False)
        self.index_compressor = _CompressorState(self, "layers.4.attn.indexer.compressor", int(self.config["index_head_dim"]), rotate=True)
        self.last_routes: list[int] = []
        self.last_attention_execution: dict[str, Any] | None = None
        self.last_moe_execution: dict[str, Any] | None = None

    def render_chat_messages(self, messages: list[Any]) -> tuple[str, str]:
        """Render messages without inventing an unavailable source chat template."""

        if self.source_chat_template is not None:
            raise DeepSeekV4GravityError(
                "source chat template is present but this diagnostic does not implement its Jinja renderer"
            )
        rows: list[str] = []
        for item in messages:
            if not isinstance(item, Mapping):
                continue
            role = item.get("role")
            content = item.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            if role not in {"system", "user", "assistant", "tool"}:
                raise DeepSeekV4GravityError(f"unsupported chat role {role!r}")
            rows.append(f"{role}: {content}")
        if not rows:
            raise DeepSeekV4GravityError("chat request contains no text messages")
        return "\n".join(rows), self.chat_template_status

    def _cache(self, name: str, value: np.ndarray) -> np.ndarray:
        if ".ffn.experts." not in name:
            self._core[name] = value
            return value
        self._expert_cache[name] = value
        self._expert_cache.move_to_end(name)
        self._expert_cache_bytes += value.nbytes
        while self._expert_cache and self._expert_cache_bytes > self._expert_cache_limit:
            _, evicted = self._expert_cache.popitem(last=False)
            self._expert_cache_bytes -= evicted.nbytes
        return value

    def _matrix(self, name: str) -> np.ndarray:
        cached = self._core.get(name)
        if cached is not None:
            return cached
        cached = self._expert_cache.get(name)
        if cached is not None:
            self._expert_cache.move_to_end(name)
            return cached
        descriptor = self.store.descriptor(name)
        shape = tuple(int(value) for value in descriptor["shape"])
        dtype = descriptor["dtype"]
        raw = self.store.read(name)
        if dtype == "BF16":
            values = _bf16_to_f32(raw)
        elif dtype == "F32":
            values = np.frombuffer(raw, dtype="<f4")
        elif dtype == "F8_E4M3":
            scale_name = name[: -len(".weight")] + ".scale"
            scale = self.store.descriptor(scale_name)
            decoded = codec.decode_fp8_e4m3fn_rows(
                raw,
                {key: descriptor[key] for key in ("name", "dtype", "shape", "data_offsets")},
                self.store.read(scale_name),
                {key: scale[key] for key in ("name", "dtype", "shape", "data_offsets")},
                row_count=shape[0],
                scale_block_row_start=0,
            )
            return self._cache(name, decoded)
        elif dtype == "I8":
            scale_name = name[: -len(".weight")] + ".scale"
            scale = self.store.descriptor(scale_name)
            decoded = codec.decode_fp4_e2m1fn_x2_rows(
                raw,
                {key: descriptor[key] for key in ("name", "dtype", "shape", "data_offsets")},
                self.store.read(scale_name),
                {key: scale[key] for key in ("name", "dtype", "shape", "data_offsets")},
                row_count=shape[0],
            )
            return self._cache(name, decoded)
        else:
            raise DeepSeekV4GravityError(f"runtime cannot materialize {name} dtype {dtype}")
        if int(np.prod(shape, dtype=np.int64)) != values.size:
            raise DeepSeekV4GravityError(f"runtime shape mismatch for {name}")
        return self._cache(name, values.reshape(shape).astype(np.float32, copy=False))

    def _vector(self, name: str) -> np.ndarray:
        value = self._matrix(name)
        if value.ndim != 1:
            raise DeepSeekV4GravityError(f"{name} is not a vector")
        return value

    def _matvec(self, name: str, vector: np.ndarray) -> np.ndarray:
        matrix = self._matrix(name)
        if matrix.ndim != 2 or matrix.shape[1] != vector.size:
            raise DeepSeekV4GravityError(
                f"{name}: matvec geometry {matrix.shape} against {vector.shape} is invalid"
            )
        return (matrix @ vector).astype(np.float32, copy=False)

    def _profile_tensor_bytes(self, name: str) -> int:
        """Logical source bytes touched by one named tensor materialization.

        This is intentionally not a cache-miss counter.  The diagnostic store
        exposes immutable source-format tensor bytes, while NumPy executes on
        decoded float32 arrays that may be cached between tokens.
        """

        descriptor = self.store.descriptor(name)
        total = int(descriptor["bytes"])
        dtype = descriptor.get("dtype")
        if dtype in {"I8", "F8_E4M3"} and name.endswith(".weight"):
            scale_name = name[: -len(".weight")] + ".scale"
            total += int(self.store.descriptor(scale_name)["bytes"])
        return total

    def _profile_matvec_estimate(self, *names: str) -> dict[str, int]:
        bytes_read = 0
        bytes_written = 0
        fp_operations = 0
        for name in names:
            descriptor = self.store.descriptor(name)
            shape = descriptor.get("shape")
            if not isinstance(shape, list) or len(shape) != 2:
                raise DeepSeekV4GravityError(f"{name}: profile expected a rank-two matvec tensor")
            rows, columns = int(shape[0]), int(shape[1])
            bytes_read += self._profile_tensor_bytes(name) + columns * 4
            bytes_written += rows * 4
            # Dense y = W @ x: one multiply and one add per matrix element.
            fp_operations += 2 * rows * columns
        return {
            "bytes_read_estimate": bytes_read,
            "bytes_written_estimate": bytes_written,
            "fp_operations_estimate": fp_operations,
        }

    @staticmethod
    def _profile_rmsnorm_estimate(values: np.ndarray) -> dict[str, int]:
        count = int(values.size)
        # Count the square, reduction/add, reciprocal-square-root and two
        # elementwise multiplies as a transparent approximate scalar model.
        return {
            "bytes_read_estimate": count * 8,
            "bytes_written_estimate": count * 4,
            "fp_operations_estimate": 5 * count + 2,
        }

    @staticmethod
    def _profile_array_bytes(values: np.ndarray) -> int:
        return int(values.size) * int(values.dtype.itemsize)

    def _embedding(self, token_id: int) -> np.ndarray:
        if token_id < 0 or token_id >= self.vocab_size:
            raise DeepSeekV4GravityError(f"token ID {token_id} is outside source vocabulary")
        raw = self.store.rows("embed.weight", token_id, 1)
        result = _bf16_to_f32(raw)
        if result.size != self.dim:
            raise DeepSeekV4GravityError("embedding row geometry differs from source config")
        return result

    def _hc_split(self, mixes: np.ndarray, scale: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        expected = (2 + self.hc_mult) * self.hc_mult
        if mixes.size != expected or scale.size != 3 or base.size != expected:
            raise DeepSeekV4GravityError("Hyper-Connections control geometry mismatch")
        pre = _sigmoid(mixes[: self.hc_mult] * scale[0] + base[: self.hc_mult]) + self.hc_eps
        post = 2.0 * _sigmoid(
            mixes[self.hc_mult : 2 * self.hc_mult] * scale[1]
            + base[self.hc_mult : 2 * self.hc_mult]
        )
        comb = mixes[2 * self.hc_mult :].reshape(self.hc_mult, self.hc_mult)
        comb = comb * scale[2] + base[2 * self.hc_mult :].reshape(self.hc_mult, self.hc_mult)
        comb = np.exp(comb - np.max(comb, axis=1, keepdims=True))
        comb = comb / np.sum(comb, axis=1, keepdims=True) + self.hc_eps
        for _ in range(self.hc_iters):
            comb = comb / (np.sum(comb, axis=1, keepdims=True) + self.hc_eps)
            comb = comb / (np.sum(comb, axis=0, keepdims=True) + self.hc_eps)
        return pre.astype(np.float32), post.astype(np.float32), comb.astype(np.float32)

    def _hc_pre(self, hidden: np.ndarray, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if hidden.shape != (self.hc_mult, self.dim):
            raise DeepSeekV4GravityError("HC state geometry mismatch")
        flat = hidden.reshape(-1)
        rsqrt = 1.0 / math.sqrt(float(np.mean(flat * flat, dtype=np.float64)) + self.norm_eps)
        mixes = self._matvec(f"layers.4.{prefix}_fn", flat) * rsqrt
        pre, post, comb = self._hc_split(
            mixes,
            self._vector(f"layers.4.{prefix}_scale"),
            self._vector(f"layers.4.{prefix}_base"),
        )
        return np.sum(pre[:, None] * hidden, axis=0), post, comb

    def _hc_post(self, update: np.ndarray, residual: np.ndarray, post: np.ndarray, comb: np.ndarray) -> np.ndarray:
        return (post[:, None] * update[None, :] + np.einsum("ij,id->jd", comb, residual)).astype(np.float32)

    def _index_positions(self, values: np.ndarray, qr: np.ndarray, position: int) -> list[int]:
        if not self.index_compressor.cache:
            self.index_compressor.step(values, position)
            return []
        q = self._matvec("layers.4.attn.indexer.wq_b.weight", qr).reshape(self.n_heads, int(self.config["index_head_dim"]))
        q = _rope(q, position, self.rope_head_dim, self.compress_rope_theta)
        q = _hadamard(q)
        self.index_compressor.step(values, position)
        cache = np.asarray(self.index_compressor.cache, dtype=np.float32)
        if not cache.size:
            return []
        weights = self._matvec("layers.4.attn.indexer.weights_proj.weight", values)
        scores = np.sum(np.maximum(q @ cache.T, 0.0) * (weights[:, None] * (self.head_dim ** -0.5) * (self.n_heads ** -0.5)), axis=0)
        count = min(int(self.config["index_topk"]), len(scores))
        return np.argsort(-scores, kind="stable")[:count].astype(int).tolist()

    def _attention(
        self,
        values: np.ndarray,
        position: int,
        *,
        profiler: _DiagnosticTokenProfiler | None = None,
    ) -> np.ndarray:
        with _profile_scope(profiler, "qkv", note="Source-derived CPU Q/KV projections."):
            qr_unorm = self._matvec("layers.4.attn.wq_a.weight", values)
        if profiler is not None:
            profiler.add_estimate("qkv", **self._profile_matvec_estimate("layers.4.attn.wq_a.weight"))
        with _profile_scope(profiler, "norm", note="Attention Q RMSNorm."):
            qr = _rmsnorm(qr_unorm, self._vector("layers.4.attn.q_norm.weight"), self.norm_eps)
        if profiler is not None:
            profiler.add_estimate("norm", **self._profile_rmsnorm_estimate(qr_unorm))
        with _profile_scope(profiler, "qkv"):
            query = self._matvec("layers.4.attn.wq_b.weight", qr).reshape(self.n_heads, self.head_dim)
            query /= np.sqrt(
                np.mean(query * query, axis=1, keepdims=True, dtype=np.float64) + self.norm_eps
            ).astype(np.float32)
            query = _rope(query, position, self.rope_head_dim, self.compress_rope_theta)
        if profiler is not None:
            qkv_cost = self._profile_matvec_estimate("layers.4.attn.wq_b.weight")
            qkv_cost["fp_operations_estimate"] += 5 * int(query.size)
            profiler.add_estimate("qkv", **qkv_cost)
        with _profile_scope(profiler, "qkv"):
            kv_unorm = self._matvec("layers.4.attn.wkv.weight", values)
        if profiler is not None:
            profiler.add_estimate("qkv", **self._profile_matvec_estimate("layers.4.attn.wkv.weight"))
        with _profile_scope(profiler, "norm", note="Attention KV RMSNorm."):
            kv = _rmsnorm(kv_unorm, self._vector("layers.4.attn.kv_norm.weight"), self.norm_eps)
        if profiler is not None:
            profiler.add_estimate("norm", **self._profile_rmsnorm_estimate(kv_unorm))
        with _profile_scope(profiler, "qkv"):
            kv = _rope(kv, position, self.rope_head_dim, self.compress_rope_theta)

        with _profile_scope(
            profiler,
            "kv_state_read_write",
            note="Window KV update and main compressor state transition are CPU fallback work.",
        ):
            self.window_kv[position % self.window_size] = kv
            main_cache_before = len(self.main_compressor.cache)
            main_compressed = self.main_compressor.step(values, position)
            main_cache_after = len(self.main_compressor.cache)
        if profiler is not None:
            kv_cost = self._profile_matvec_estimate(
                "layers.4.attn.compressor.wkv.weight",
                "layers.4.attn.compressor.wgate.weight",
            )
            kv_cost["bytes_read_estimate"] += (
                self._profile_array_bytes(self.window_kv)
                + self._profile_array_bytes(self.main_compressor.kv_state)
                + self._profile_array_bytes(self.main_compressor.score_state)
            )
            kv_cost["bytes_written_estimate"] += (
                self._profile_array_bytes(kv)
                + self._profile_array_bytes(self.main_compressor.kv_state)
                + self._profile_array_bytes(self.main_compressor.score_state)
            )
            profiler.add_estimate("kv_state_read_write", **kv_cost)
            profiler.add_metadata(
                "kv_state_read_write",
                main_compressor_emitted=main_compressed is not None,
                main_compressor_cache_after=main_cache_after,
            )

        index_cache_before = len(self.index_compressor.cache)
        with _profile_scope(
            profiler,
            "index_heads_topk_index",
            note="Index projection, compressor update, and source top-k index selection.",
        ):
            compressed_indices = self._index_positions(values, qr, position)
        index_cache_after = len(self.index_compressor.cache)
        if profiler is not None:
            index_cost = self._profile_matvec_estimate(
                "layers.4.attn.indexer.compressor.wkv.weight",
                "layers.4.attn.indexer.compressor.wgate.weight",
            )
            if index_cache_before > 0:
                active_index = self._profile_matvec_estimate(
                    "layers.4.attn.indexer.wq_b.weight",
                    "layers.4.attn.indexer.weights_proj.weight",
                )
                for key, value in active_index.items():
                    index_cost[key] += value
            index_cost["integer_bit_operations_estimate"] = (
                int(index_cost.get("integer_bit_operations_estimate", 0))
                + max(0, len(self.index_compressor.cache) - 1)
            )
            profiler.add_estimate("index_heads_topk_index", **index_cost)
            profiler.add_metadata(
                "index_heads_topk_index",
                index_query_executed=index_cache_before > 0,
                compressed_index_count=len(compressed_indices),
            )

        with _profile_scope(
            profiler,
            "compressed_sparse_attention",
            note="Sparse key scoring, normalized weighted read, inverse RoPE, and output projections.",
        ):
            window_start = max(0, position - self.window_size + 1)
            window_keys = [
                self.window_kv[item % self.window_size] for item in range(window_start, position + 1)
            ]
            compressed_key_indices = [
                int(item) for item in compressed_indices if 0 <= int(item) < len(self.main_compressor.cache)
            ]
            keys = list(window_keys)
            keys.extend(self.main_compressor.cache[item] for item in compressed_key_indices)
            if not keys:
                raise DeepSeekV4GravityError("V4 attention has no visible keys")
            key_matrix = np.asarray(keys, dtype=np.float32)
            scores = query @ key_matrix.T * np.float32(self.head_dim ** -0.5)
            sink = self._vector("layers.4.attn.attn_sink")
            maximum = np.maximum(np.max(scores, axis=1), sink)
            weights = np.exp(scores - maximum[:, None])
            denominator = np.sum(weights, axis=1) + np.exp(sink - maximum)
            output = (weights / denominator[:, None]) @ key_matrix
            output = _rope(output, position, self.rope_head_dim, self.compress_rope_theta, inverse=True)
            grouped = output.reshape(self.o_groups, -1)
            wo_a = self._matrix("layers.4.attn.wo_a.weight").reshape(self.o_groups, self.o_rank, -1)
            projected = np.einsum("gri,gi->gr", wo_a, grouped, optimize=True).reshape(-1)
            result = self._matvec("layers.4.attn.wo_b.weight", projected)
        if profiler is not None:
            attention_cost = self._profile_matvec_estimate("layers.4.attn.wo_b.weight")
            attention_cost["bytes_read_estimate"] += (
                self._profile_tensor_bytes("layers.4.attn.wo_a.weight")
                + self._profile_array_bytes(query)
                + self._profile_array_bytes(key_matrix) * 2
            )
            attention_cost["bytes_written_estimate"] += (
                self._profile_array_bytes(scores)
                + self._profile_array_bytes(weights)
                + self._profile_array_bytes(output)
                + self._profile_array_bytes(projected)
            )
            attention_cost["fp_operations_estimate"] += (
                4 * self.n_heads * len(keys) * self.head_dim
                + 2 * int(np.prod(wo_a.shape, dtype=np.int64))
                + 8 * int(scores.size)
            )
            profiler.add_estimate("compressed_sparse_attention", **attention_cost)
            profiler.add_metadata(
                "compressed_sparse_attention",
                sparse_key_count=len(keys),
                compressed_key_count=len(compressed_key_indices),
            )
        self.last_attention_execution = {
            "executed": True,
            "kind": "sparse_compressed_attention_cpu_fallback",
            "position": int(position),
            "window_key_count": len(window_keys),
            "main_compressor_cache_before": main_cache_before,
            "main_compressor_emitted": main_compressed is not None,
            "main_compressor_cache_after": main_cache_after,
            "index_compressor_cache_before": index_cache_before,
            "index_compressor_emitted": index_cache_after > index_cache_before,
            "index_compressor_cache_after": index_cache_after,
            "index_query_executed": index_cache_before > 0,
            "compressed_index_count": len(compressed_indices),
            "compressed_key_count": len(compressed_key_indices),
            "compressed_key_indices": compressed_key_indices,
            "attention_key_count": len(keys),
        }
        return result

    def _expert(
        self,
        prefix: str,
        values: np.ndarray,
        *,
        route_weight: float | None = None,
        profiler: _DiagnosticTokenProfiler | None = None,
    ) -> np.ndarray:
        if profiler is None:
            gate = self._matvec(f"{prefix}.w1.weight", values)
            up = self._matvec(f"{prefix}.w3.weight", values)
            if self.swiglu_limit > 0:
                up = np.clip(up, -self.swiglu_limit, self.swiglu_limit)
                gate = np.minimum(gate, self.swiglu_limit)
            activation = gate / (1.0 + np.exp(-gate)) * up
            if route_weight is not None:
                activation *= np.float32(route_weight)
            return self._matvec(f"{prefix}.w2.weight", activation)

        w1_name = f"{prefix}.w1.weight"
        w3_name = f"{prefix}.w3.weight"
        w2_name = f"{prefix}.w2.weight"
        with _profile_scope(
            profiler,
            "expert_gather",
            note="Logical active-expert materialization/cache lookup; bytes are source-format logical bytes, not hardware cache misses.",
        ):
            w1 = self._matrix(w1_name)
            w3 = self._matrix(w3_name)
            w2 = self._matrix(w2_name)
        profiler.add_estimate(
            "expert_gather",
            bytes_read_estimate=(
                self._profile_tensor_bytes(w1_name)
                + self._profile_tensor_bytes(w3_name)
                + self._profile_tensor_bytes(w2_name)
            ),
            bytes_written_estimate=(self._profile_array_bytes(w1) + self._profile_array_bytes(w3) + self._profile_array_bytes(w2)),
        )
        with _profile_scope(profiler, "gate_up", note="Dense expert gate/up CPU matvecs."):
            gate = (w1 @ values).astype(np.float32, copy=False)
            up = (w3 @ values).astype(np.float32, copy=False)
        profiler.add_estimate("gate_up", **self._profile_matvec_estimate(w1_name, w3_name))
        with _profile_scope(profiler, "activation", note="SwiGLU and configured activation clipping."):
            if self.swiglu_limit > 0:
                up = np.clip(up, -self.swiglu_limit, self.swiglu_limit)
                gate = np.minimum(gate, self.swiglu_limit)
            activation = gate / (1.0 + np.exp(-gate)) * up
        profiler.add_estimate(
            "activation",
            bytes_read_estimate=self._profile_array_bytes(gate) + self._profile_array_bytes(up),
            bytes_written_estimate=self._profile_array_bytes(activation),
            fp_operations_estimate=7 * int(activation.size),
        )
        if route_weight is not None:
            with _profile_scope(profiler, "route_combine", note="Route weight multiplication before the expert down projection."):
                activation *= np.float32(route_weight)
            profiler.add_estimate(
                "route_combine",
                bytes_read_estimate=self._profile_array_bytes(activation),
                bytes_written_estimate=self._profile_array_bytes(activation),
                fp_operations_estimate=int(activation.size),
            )
        with _profile_scope(profiler, "down", note="Dense expert down CPU matvec."):
            output = (w2 @ activation).astype(np.float32, copy=False)
        profiler.add_estimate("down", **self._profile_matvec_estimate(w2_name))
        return output

    def _moe(
        self,
        values: np.ndarray,
        *,
        profiler: _DiagnosticTokenProfiler | None = None,
        latent_capture: dict[str, Any] | None = None,
    ) -> np.ndarray:
        with _profile_scope(
            profiler,
            "router_top6",
            note="Source score router, stable top-6 selection, and route-weight normalization.",
        ):
            logits = self._matvec("layers.4.ffn.gate.weight", values)
            original = np.sqrt(_softplus(logits))
            selected_scores = original + self._vector("layers.4.ffn.gate.bias")
            routes = np.argsort(-selected_scores, kind="stable")[: self.topk].astype(int)
            weights = original[routes]
            weights /= np.sum(weights)
            weights *= self.route_scale
        if latent_capture is not None:
            route_probabilities = weights / np.sum(weights)
            sorted_scores = selected_scores[routes]
            selected_set = set(int(route) for route in routes.tolist())
            rejected_scores = np.asarray(
                [
                    float(score)
                    for index, score in enumerate(selected_scores.tolist())
                    if index not in selected_set
                ],
                dtype=np.float32,
            )
            latent_capture.update(
                {
                    "router_logits": _latent_array_summary(logits),
                    "router_selection_scores": _latent_array_summary(selected_scores),
                    "top6": {
                        "selected_expert_ids": [int(route) for route in routes.tolist()],
                        "selected_probabilities": [
                            float(probability) for probability in route_probabilities.tolist()
                        ],
                        "applied_route_weights": [float(weight) for weight in weights.tolist()],
                        "probability_sum": float(np.sum(route_probabilities, dtype=np.float64)),
                        "route_scale": float(self.route_scale),
                        "top1_top2_score_margin": (
                            float(sorted_scores[0] - sorted_scores[1])
                            if len(sorted_scores) >= 2
                            else None
                        ),
                        "selection_cutoff_margin": (
                            float(sorted_scores[-1] - np.max(rejected_scores))
                            if rejected_scores.size
                            else None
                        ),
                    },
                }
            )
        if profiler is not None:
            router_cost = self._profile_matvec_estimate("layers.4.ffn.gate.weight")
            router_cost["bytes_read_estimate"] += self._profile_array_bytes(logits) * 2
            router_cost["bytes_written_estimate"] += self._profile_array_bytes(selected_scores)
            router_cost["fp_operations_estimate"] += 6 * int(logits.size) + 2 * self.topk
            router_cost["integer_bit_operations_estimate"] = (
                int(router_cost.get("integer_bit_operations_estimate", 0))
                + int(logits.size) * int(math.ceil(math.log2(max(2, logits.size))))
            )
            profiler.add_estimate("router_top6", **router_cost)
            profiler.add_metadata("router_top6", selected_route_ids=routes.tolist(), selected_route_count=len(routes))
            profiler.add_manual(
                "shared_expert",
                cpu_duration_ms=0.0,
                cpu_wall_elapsed_ms=0.0,
                execution_status="executed_compute_accounted_by_gate_up_activation_down",
                note="Shared expert executed; its dense work is attributed to expert gather, gate/up, activation, and down.",
            )
        self.last_routes = routes.tolist()
        output = self._expert("layers.4.ffn.shared_experts", values, profiler=profiler)
        for route, weight in zip(routes.tolist(), weights.tolist()):
            expert_output = self._expert(
                f"layers.4.ffn.experts.{route}", values, route_weight=weight, profiler=profiler
            )
            with _profile_scope(profiler, "route_combine", note="Route-weighted expert output accumulation."):
                output += expert_output
            if profiler is not None:
                profiler.add_estimate(
                    "route_combine",
                    bytes_read_estimate=self._profile_array_bytes(output) + self._profile_array_bytes(expert_output),
                    bytes_written_estimate=self._profile_array_bytes(output),
                    fp_operations_estimate=int(output.size),
                )
        self.last_moe_execution = {
            "selected_route_ids": self.last_routes.copy(),
            "routed_expert_count": len(self.last_routes),
            "shared_expert_executed": True,
        }
        return output.astype(np.float32, copy=False)

    def _head_logits(self, hidden: np.ndarray) -> np.ndarray:
        fn = self._matrix("hc_head_fn")
        base = self._vector("hc_head_base")
        scale = self._vector("hc_head_scale")
        flat = hidden.reshape(-1)
        rsqrt = 1.0 / math.sqrt(float(np.mean(flat * flat, dtype=np.float64)) + self.norm_eps)
        mixes = fn @ flat * rsqrt
        weights = _sigmoid(mixes * scale[0] + base) + self.hc_eps
        merged = np.sum(weights[:, None] * hidden, axis=0)
        merged = _rmsnorm(merged, self._vector("norm.weight"), self.norm_eps)
        descriptor = self.store.descriptor("head.weight")
        shape = tuple(int(value) for value in descriptor["shape"])
        if shape != (self.vocab_size, self.dim):
            raise DeepSeekV4GravityError("head geometry differs from source config")
        row_bytes = self.dim * 2
        rows_per_block = max(1, DEFAULT_RANGE_BYTES // row_bytes)
        logits = np.empty(self.vocab_size, dtype=np.float32)
        for first in range(0, self.vocab_size, rows_per_block):
            count = min(rows_per_block, self.vocab_size - first)
            raw = self.store.rows("head.weight", first, count)
            matrix = _bf16_to_f32(raw).reshape(count, self.dim)
            logits[first : first + count] = matrix @ merged
        return logits

    def forward_token(self, token_id: int) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        embedding = self._embedding(token_id)
        hidden = np.repeat(embedding[None, :], self.hc_mult, axis=0)
        residual = hidden
        values, post, comb = self._hc_pre(hidden, "hc_attn")
        attention = self._attention(_rmsnorm(values, self._vector("layers.4.attn_norm.weight"), self.norm_eps), self.position)
        hidden = self._hc_post(attention, residual, post, comb)
        residual = hidden
        values, post, comb = self._hc_pre(hidden, "hc_ffn")
        moe = self._moe(_rmsnorm(values, self._vector("layers.4.ffn_norm.weight"), self.norm_eps))
        hidden = self._hc_post(moe, residual, post, comb)
        logits = self._head_logits(hidden)
        if self.last_attention_execution is None or self.last_moe_execution is None:
            raise DeepSeekV4GravityError("diagnostic forward completed without component execution evidence")
        trace = {
            "position": self.position,
            "token_id": token_id,
            "routes": self.last_routes,
            "component_execution": _component_execution_trace(
                token_id=token_id,
                hc_mult=self.hc_mult,
                hc_iters=self.hc_iters,
                attention=self.last_attention_execution,
                moe=self.last_moe_execution,
            ),
            "elapsed_ms": (time.perf_counter() - started) * 1_000.0,
            "logits_sha256": _sha256(logits.astype("<f4", copy=False).tobytes()),
        }
        self.position += 1
        return logits, trace

    def _latent_compressor_summary(self, compressor: _CompressorState) -> dict[str, Any]:
        """Summarize a bounded diagnostic compressor without serializing it."""

        cache = compressor.cache
        return {
            "cache_entry_count": len(cache),
            "kv_state": _latent_array_summary(compressor.kv_state),
            "score_state": _latent_array_summary(compressor.score_state),
            "newest_cache_entry": (
                _latent_array_summary(cache[-1]) if cache else None
            ),
            "raw_state_retained": False,
        }

    def _latent_attention_index_summary(self) -> dict[str, Any]:
        """Capture only state summaries and hashes for the sparse index path."""

        execution = self.last_attention_execution
        if not isinstance(execution, Mapping):
            raise DeepSeekV4GravityError("latent capture has no attention execution state")
        raw_indices = execution.get("compressed_key_indices")
        if not isinstance(raw_indices, list) or not all(
            isinstance(value, int) for value in raw_indices
        ):
            raise DeepSeekV4GravityError("latent capture attention index state is malformed")
        execution_fields = (
            "kind",
            "position",
            "window_key_count",
            "main_compressor_cache_before",
            "main_compressor_emitted",
            "main_compressor_cache_after",
            "index_compressor_cache_before",
            "index_compressor_emitted",
            "index_compressor_cache_after",
            "index_query_executed",
            "compressed_index_count",
            "compressed_key_count",
            "attention_key_count",
        )
        return {
            "execution": {
                key: execution.get(key)
                for key in execution_fields
                if key in execution
            },
            "selected_compressed_key_indices": {
                "count": len(raw_indices),
                "ordered_indices_sha256": _sha256(_canonical(raw_indices)),
                "raw_indices_retained": False,
            },
            "window_kv": _latent_array_summary(self.window_kv),
            "main_compressor": self._latent_compressor_summary(self.main_compressor),
            "index_compressor": self._latent_compressor_summary(self.index_compressor),
            "raw_attention_or_index_state_retained": False,
        }

    def _latent_capture_forward_token(self, token_id: int) -> dict[str, Any]:
        """Run one real diagnostic forward and retain only bounded statistics.

        This deliberately mirrors ``forward_token`` rather than using its
        component-only trace.  It exposes the future transplant observables
        while every intermediate is summarized and discarded before return.
        """

        position = self.position
        embedding = self._embedding(token_id)
        hidden = np.repeat(embedding[None, :], self.hc_mult, axis=0)
        attention_input_hidden = hidden
        attention_values, attention_post, attention_comb = self._hc_pre(
            attention_input_hidden, "hc_attn"
        )
        normalized_attention = _rmsnorm(
            attention_values,
            self._vector("layers.4.attn_norm.weight"),
            self.norm_eps,
        )
        attention_update = self._attention(normalized_attention, position)
        post_attention_hidden = self._hc_post(
            attention_update, attention_input_hidden, attention_post, attention_comb
        )

        ffn_input_hidden = post_attention_hidden
        moe_values, moe_post, moe_comb = self._hc_pre(ffn_input_hidden, "hc_ffn")
        pre_router_hidden = _rmsnorm(
            moe_values,
            self._vector("layers.4.ffn_norm.weight"),
            self.norm_eps,
        )
        router_capture: dict[str, Any] = {}
        moe_update = self._moe(pre_router_hidden, latent_capture=router_capture)
        post_moe_hidden = self._hc_post(moe_update, ffn_input_hidden, moe_post, moe_comb)
        logits = self._head_logits(post_moe_hidden)
        top6 = router_capture.get("top6")
        if not isinstance(top6, Mapping) or not isinstance(
            top6.get("selected_expert_ids"), list
        ):
            raise DeepSeekV4GravityError("latent capture MoE did not report a top-6 route")
        if len(top6["selected_expert_ids"]) != self.topk:
            raise DeepSeekV4GravityError("latent capture route count differs from source top-k")
        if self.last_attention_execution is None:
            raise DeepSeekV4GravityError("latent capture forward has no attention execution evidence")
        sampled_token_id = int(np.argmax(logits))
        trace = {
            "schema": "hawking.gravity.deepseek_v4.diagnostic_latent_trace_shard.v1",
            "position": int(position),
            "source_token_id_sha256": _latent_token_id_digest(token_id),
            "embedding": _latent_array_summary(embedding),
            "mhc_state": {
                "attention": {
                    "input_hidden": _latent_array_summary(attention_input_hidden),
                    "pre_norm_hidden": _latent_array_summary(attention_values),
                    "post_gate": _latent_array_summary(attention_post),
                    "combine": _latent_array_summary(attention_comb),
                },
                "ffn": {
                    "input_hidden": _latent_array_summary(ffn_input_hidden),
                    "pre_norm_hidden": _latent_array_summary(moe_values),
                    "post_gate": _latent_array_summary(moe_post),
                    "combine": _latent_array_summary(moe_comb),
                },
                "sinkhorn_iterations": int(self.hc_iters),
                "raw_mhc_state_retained": False,
            },
            "pre_norm_hidden": {
                "attention": _latent_array_summary(attention_values),
                "router": _latent_array_summary(moe_values),
            },
            "attention_input": _latent_array_summary(normalized_attention),
            "attention_index_state": self._latent_attention_index_summary(),
            "post_attention_hidden": _latent_array_summary(post_attention_hidden),
            "pre_router_hidden": _latent_array_summary(pre_router_hidden),
            "router": router_capture,
            "post_moe_hidden": _latent_array_summary(post_moe_hidden),
            "final_hidden_state": _latent_array_summary(post_moe_hidden),
            "lm_head_logits": _latent_array_summary(logits),
            "sampling": {
                "argmax_token_id_sha256": _latent_token_id_digest(sampled_token_id),
                "completion_text_disclosed": False,
            },
            "raw_activations_retained": False,
        }
        self.position += 1
        return trace

    def capture_latent_route_suite(self) -> dict[str, Any]:
        """Capture a fixed, disjoint, hash-only diagnostic bridge suite.

        No generations are decoded and no HCLI call is made here.  This keeps
        the result useful for future bridge calibration while not becoming an
        evaluation, donor, or raw-activation archive.
        """

        if len(_DIAGNOSTIC_LATENT_TRACE_PROMPT_SUITE) != 7 or len(
            set(_DIAGNOSTIC_LATENT_TRACE_CATEGORIES)
        ) != 7:
            raise DeepSeekV4GravityError("latent trace suite must contain exactly seven unique categories")
        members: list[dict[str, Any]] = []
        trace_shards: list[dict[str, Any]] = []
        route_frequency: Counter[int] = Counter()
        route_set_frequency: Counter[str] = Counter()
        transition_frequency: Counter[str] = Counter()
        seen_prompt_hashes: set[str] = set()
        seen_token_id_hashes: set[str] = set()
        with self._lock:
            for category, prompt in _DIAGNOSTIC_LATENT_TRACE_PROMPT_SUITE:
                prompt_bytes = prompt.encode("utf-8")
                prompt_sha256 = _sha256(prompt_bytes)
                if prompt_sha256 in seen_prompt_hashes:
                    raise DeepSeekV4GravityError("latent prompt membership is not disjoint")
                seen_prompt_hashes.add(prompt_sha256)
                self.reset()
                encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
                input_ids = [int(token_id) for token_id in encoded.ids]
                if not input_ids:
                    raise DeepSeekV4GravityError("latent prompt source tokenizer produced no IDs")
                if len(input_ids) > MAX_DIAGNOSTIC_LATENT_TRACE_PROMPT_TOKENS:
                    raise DeepSeekV4GravityError(
                        "latent prompt exceeds the bounded "
                        f"{MAX_DIAGNOSTIC_LATENT_TRACE_PROMPT_TOKENS}-token capture cap"
                    )
                token_ids_sha256 = _latent_token_ids_digest(input_ids)
                if token_ids_sha256 in seen_token_id_hashes:
                    raise DeepSeekV4GravityError("latent token membership is not disjoint")
                seen_token_id_hashes.add(token_ids_sha256)
                qualified_positions = {0, len(input_ids) - 1}
                previous_route_set_sha256: str | None = None
                category_route_sets: set[str] = set()
                category_shard_positions: list[int] = []
                for input_index, token_id in enumerate(input_ids):
                    trace = self._latent_capture_forward_token(token_id)
                    router = trace["router"]
                    top6 = router["top6"]
                    selected_routes = [int(value) for value in top6["selected_expert_ids"]]
                    if len(selected_routes) != self.topk:
                        raise DeepSeekV4GravityError("latent trace selected an unexpected route count")
                    route_frequency.update(selected_routes)
                    route_set_sha256 = _sha256(_canonical(selected_routes))
                    route_set_frequency[route_set_sha256] += 1
                    category_route_sets.add(route_set_sha256)
                    if previous_route_set_sha256 is not None:
                        transition_frequency[
                            f"{previous_route_set_sha256}:{route_set_sha256}"
                        ] += 1
                    previous_route_set_sha256 = route_set_sha256
                    if input_index in qualified_positions:
                        trace_shards.append(
                            {
                                "category": category,
                                "token_ordinal": input_index,
                                "trace": trace,
                            }
                        )
                        category_shard_positions.append(input_index)
                if len(category_shard_positions) > MAX_DIAGNOSTIC_LATENT_TRACE_SHARDS_PER_PROMPT:
                    raise DeepSeekV4GravityError("latent trace shard cap exceeded")
                members.append(
                    {
                        "category": category,
                        "prompt_sha256": prompt_sha256,
                        "prompt_utf8_bytes": len(prompt_bytes),
                        "source_token_count": len(input_ids),
                        "source_token_ids_sha256": token_ids_sha256,
                        "qualified_trace_shard_positions": category_shard_positions,
                        "qualified_trace_shard_count": len(category_shard_positions),
                        "distinct_route_set_count": len(category_route_sets),
                        "diagnostic_context_limited": category == "long-context retrieval",
                        "category_scope": (
                            "retrieval-shaped synthetic prompt only; this 128-token layer-4 diagnostic "
                            "does not establish long-context retrieval behavior"
                            if category == "long-context retrieval"
                            else "synthetic diagnostic bridge prompt only; not a capability evaluation"
                        ),
                        "prompt_text_disclosed": False,
                        "completion_text_disclosed": False,
                        "raw_activations_retained": False,
                    }
                )
        return {
            "schema": "hawking.gravity.deepseek_v4.diagnostic_latent_route_capture_run.v1",
            "status": "REAL_LAYER4_CPU_DIAGNOSTIC_BOUNDED_LATENT_ROUTE_CAPTURE",
            "categories": list(_DIAGNOSTIC_LATENT_TRACE_CATEGORIES),
            "members": members,
            "trace_shards": trace_shards,
            "route_aggregate": {
                "total_source_forwards": sum(route_frequency.values()) // self.topk,
                "expert_frequency": {
                    str(route): count for route, count in sorted(route_frequency.items())
                },
                "distinct_expert_count": len(route_frequency),
                "route_set_frequency": dict(sorted(route_set_frequency.items())),
                "route_set_transition_frequency": dict(sorted(transition_frequency.items())),
                "raw_route_sequence_retained": False,
            },
            "membership_partition": {
                "name": "diagnostic_transplant_capture_only",
                "disjoint_prompt_hashes": True,
                "disjoint_source_token_sequences": True,
                "excluded_from": ["fit", "calibration", "public_test", "hidden_test"],
            },
            "collection_limits": {
                "categories": 7,
                "diagnostic_context_limit_source_tokens": 128,
                "max_source_tokens_per_category": MAX_DIAGNOSTIC_LATENT_TRACE_PROMPT_TOKENS,
                "max_trace_shards_per_category": MAX_DIAGNOSTIC_LATENT_TRACE_SHARDS_PER_PROMPT,
                "raw_prompts_retained": False,
                "raw_completions_retained": False,
                "raw_hidden_states_retained": False,
            },
            "execution_policy": {
                "state_reset_between_categories": True,
                "decoded_weight_cache_retained_within_process": True,
                "decode_generation": "not_run",
                "hcli_tool_loop": "not_run_by_inprocess_capture",
            },
        }

    def generate(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt:
            raise DeepSeekV4GravityError("diagnostic runtime requires a non-empty prompt")
        if max_tokens <= 0 or max_tokens > 4:
            raise DeepSeekV4GravityError("diagnostic runtime supports 1..4 output tokens; it is not TPS/TG eligible")
        with self._lock:
            self.reset()
            encode_started = time.perf_counter()
            encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = list(encoded.ids)
            encode_ms = (time.perf_counter() - encode_started) * 1_000.0
            if not input_ids:
                raise DeepSeekV4GravityError("source tokenizer produced no IDs for the prompt")
            if len(input_ids) > 128:
                raise DeepSeekV4GravityError("diagnostic context is limited to 128 source-token IDs")
            prefill_started = time.perf_counter()
            traces: list[dict[str, Any]] = []
            logits: np.ndarray | None = None
            for token_id in input_ids:
                logits, trace = self.forward_token(token_id)
                traces.append(trace)
            assert logits is not None
            prefill_ms = (time.perf_counter() - prefill_started) * 1_000.0
            completion_ids: list[int] = []
            completion_pieces: list[str] = []
            decode_ms: list[float] = []
            for index in range(max_tokens):
                token_id = int(np.argmax(logits))
                if self.eos_id is not None and token_id == self.eos_id:
                    break
                completion_ids.append(token_id)
                completion_pieces.append(self.tokenizer.decode([token_id]))
                if index + 1 >= max_tokens:
                    break
                started = time.perf_counter()
                logits, trace = self.forward_token(token_id)
                traces.append(trace)
                decode_ms.append((time.perf_counter() - started) * 1_000.0)
            return {
                "trace_schema": "hawking.gravity.deepseek_v4.diagnostic_forward_trace.v2",
                "text": "".join(completion_pieces),
                "input_ids": input_ids,
                "completion_ids": completion_ids,
                "stats": {
                    "prompt_tokens": len(input_ids),
                    "completion_tokens": len(completion_ids),
                    "prefill_ms": prefill_ms,
                    "first_token_ms": prefill_ms,
                    "decode_ms": sum(decode_ms),
                    "decode_token_ms": decode_ms,
                    "completed_decode_forwards": len(decode_ms),
                    "dispatches_per_token": 0,
                    "waits_per_token": 0,
                    "device": "cpu_numpy_diagnostic",
                    "fallback": "cpu_numpy_activation_qat_and_metal_parity_not_implemented",
                    "chat_template": "raw_prompt",
                    "load_ms": self.load_ms,
                    "tokenize_ms": encode_ms,
                },
                "trace": traces,
            }

    def _profile_hc_pre_estimate(
        self, prefix: str, hidden: np.ndarray
    ) -> dict[str, int]:
        cost = self._profile_matvec_estimate(f"layers.4.{prefix}_fn")
        flat_count = int(hidden.size)
        control_count = (2 + self.hc_mult) * self.hc_mult
        cost["bytes_read_estimate"] += flat_count * 4 + control_count * 4
        cost["bytes_written_estimate"] += (self.dim + control_count) * 4
        cost["fp_operations_estimate"] += (
            5 * flat_count
            + 8 * control_count
            + 4 * self.hc_iters * self.hc_mult * self.hc_mult
        )
        return cost

    def _profile_hc_post_estimate(
        self, update: np.ndarray, residual: np.ndarray, comb: np.ndarray
    ) -> dict[str, int]:
        count = int(residual.size)
        return {
            "bytes_read_estimate": self._profile_array_bytes(update)
            + self._profile_array_bytes(residual)
            + self._profile_array_bytes(comb),
            "bytes_written_estimate": count * 4,
            "fp_operations_estimate": 3 * count + 2 * int(comb.size) * self.dim,
        }

    def forward_token_profiled(
        self,
        token_id: int,
        *,
        phase: str,
        token_ordinal: int,
        tokenizer_allocation: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], int]:
        """Run one real diagnostic forward with non-promotional stage accounting."""

        position = self.position
        profiler = _DiagnosticTokenProfiler(
            phase=phase,
            token_ordinal=token_ordinal,
            position=position,
            tokenizer_allocation=tokenizer_allocation,
        )
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        with _profile_scope(profiler, "embedding", note="One source BF16 embedding row."):
            embedding = self._embedding(token_id)
        profiler.add_estimate(
            "embedding",
            bytes_read_estimate=self.dim * 2,
            bytes_written_estimate=self._profile_array_bytes(embedding),
        )
        hidden = np.repeat(embedding[None, :], self.hc_mult, axis=0)
        profiler.add_estimate(
            "embedding",
            bytes_written_estimate=self._profile_array_bytes(hidden),
            note="Embedding replication initializes the diagnostic four-way mHC state.",
        )

        residual = hidden
        with _profile_scope(profiler, "mhc_state_control", note="Attention mHC control projection and Sinkhorn mixing."):
            values, post, comb = self._hc_pre(hidden, "hc_attn")
        profiler.add_estimate("mhc_state_control", **self._profile_hc_pre_estimate("hc_attn", hidden))
        with _profile_scope(profiler, "norm", note="Pre-attention RMSNorm."):
            normalized_attention = _rmsnorm(
                values, self._vector("layers.4.attn_norm.weight"), self.norm_eps
            )
        profiler.add_estimate("norm", **self._profile_rmsnorm_estimate(values))
        attention = self._attention(normalized_attention, position, profiler=profiler)
        with _profile_scope(profiler, "residual", note="Attention residual mHC post-mix."):
            hidden = self._hc_post(attention, residual, post, comb)
        profiler.add_estimate("residual", **self._profile_hc_post_estimate(attention, residual, comb))

        residual = hidden
        with _profile_scope(profiler, "mhc_state_control", note="FFN mHC control projection and Sinkhorn mixing."):
            values, post, comb = self._hc_pre(hidden, "hc_ffn")
        profiler.add_estimate("mhc_state_control", **self._profile_hc_pre_estimate("hc_ffn", hidden))
        with _profile_scope(profiler, "norm", note="Pre-MoE RMSNorm."):
            normalized_moe = _rmsnorm(
                values, self._vector("layers.4.ffn_norm.weight"), self.norm_eps
            )
        profiler.add_estimate("norm", **self._profile_rmsnorm_estimate(values))
        moe = self._moe(normalized_moe, profiler=profiler)
        with _profile_scope(profiler, "residual", note="MoE residual mHC post-mix."):
            hidden = self._hc_post(moe, residual, post, comb)
        profiler.add_estimate("residual", **self._profile_hc_post_estimate(moe, residual, comb))

        with _profile_scope(profiler, "lm_head", note="mHC head merge, final norm, and complete vocabulary logits."):
            logits = self._head_logits(hidden)
        head_cost = self._profile_matvec_estimate("hc_head_fn", "head.weight")
        head_cost["bytes_read_estimate"] += self._profile_array_bytes(hidden)
        head_cost["bytes_written_estimate"] += self._profile_array_bytes(logits)
        head_cost["fp_operations_estimate"] += 5 * int(hidden.size) + 6 * self.dim
        profiler.add_estimate("lm_head", **head_cost)

        with _profile_scope(profiler, "topk_sampling", note="Diagnostic greedy argmax only; no draft/speculative acceptance is used."):
            sampled_token_id = int(np.argmax(logits))
        profiler.add_estimate(
            "topk_sampling",
            bytes_read_estimate=self._profile_array_bytes(logits),
            bytes_written_estimate=4,
            integer_bit_operations_estimate=max(0, int(logits.size) - 1),
        )

        if self.last_attention_execution is None or self.last_moe_execution is None:
            raise DeepSeekV4GravityError("diagnostic profiled forward completed without component execution evidence")
        with _profile_scope(profiler, "runtime_bookkeeping", note="Execution trace hashing and component-evidence assembly."):
            trace = {
                "position": position,
                "token_id": token_id,
                "routes": self.last_routes,
                "component_execution": _component_execution_trace(
                    token_id=token_id,
                    hc_mult=self.hc_mult,
                    hc_iters=self.hc_iters,
                    attention=self.last_attention_execution,
                    moe=self.last_moe_execution,
                ),
                "logits_sha256": _sha256(logits.astype("<f4", copy=False).tobytes()),
            }
        self.position += 1
        profile = profiler.finish(
            forward_wall_elapsed_ms=(time.perf_counter() - wall_started) * 1_000.0,
            forward_cpu_duration_ms=(time.process_time() - cpu_started) * 1_000.0,
        )
        trace["elapsed_ms"] = profile["timing_accounting"]["observed_complete_token_wall_elapsed_ms"]
        trace["complete_token_profile_schema"] = profile["schema"]
        return logits, trace, profile, sampled_token_id

    def profile_prompt(
        self, prompt: str, *, trials: int, decode_forwards: int
    ) -> dict[str, Any]:
        """Execute a bounded, redacted complete-token CPU profile.

        It is intentionally not a TPS measurement: the diagnostic is only a
        layer-4 CPU fallback and cannot execute the required long-context or
        long-decode Base True benchmark.
        """

        if not isinstance(prompt, str) or not prompt:
            raise DeepSeekV4GravityError("diagnostic profile requires a non-empty prompt")
        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) > MAX_DIAGNOSTIC_PROFILE_PROMPT_UTF8_BYTES:
            raise DeepSeekV4GravityError(
                f"diagnostic profile prompt exceeds {MAX_DIAGNOSTIC_PROFILE_PROMPT_UTF8_BYTES} UTF-8 bytes"
            )
        if trials <= 0 or trials > MAX_DIAGNOSTIC_PROFILE_TRIALS:
            raise DeepSeekV4GravityError(
                f"diagnostic profile trials must be 1..{MAX_DIAGNOSTIC_PROFILE_TRIALS}"
            )
        if decode_forwards < 0 or decode_forwards > MAX_DIAGNOSTIC_PROFILE_DECODE_FORWARDS:
            raise DeepSeekV4GravityError(
                "diagnostic profile decode forwards must be "
                f"0..{MAX_DIAGNOSTIC_PROFILE_DECODE_FORWARDS}"
            )
        records: list[dict[str, Any]] = []
        route_frequency: Counter[int] = Counter()
        trial_summaries: list[dict[str, Any]] = []
        with self._lock:
            for trial in range(trials):
                self.reset()
                tokenizer_wall_started = time.perf_counter()
                tokenizer_cpu_started = time.process_time()
                encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
                input_ids = list(encoded.ids)
                tokenizer_wall_ms = (time.perf_counter() - tokenizer_wall_started) * 1_000.0
                tokenizer_cpu_ms = (time.process_time() - tokenizer_cpu_started) * 1_000.0
                if not input_ids:
                    raise DeepSeekV4GravityError("source tokenizer produced no IDs for profile prompt")
                if len(input_ids) > 128:
                    raise DeepSeekV4GravityError("diagnostic profile context is limited to 128 source-token IDs")
                if len(input_ids) + decode_forwards > 128:
                    raise DeepSeekV4GravityError(
                        "diagnostic profile input tokens plus decode forwards exceed the 128-token diagnostic context"
                    )
                per_input = len(input_ids)
                tokenizer_allocation = {
                    "cpu_duration_ms": tokenizer_cpu_ms / per_input,
                    "cpu_wall_elapsed_ms": tokenizer_wall_ms / per_input,
                    "bytes_read_estimate": len(prompt_bytes) // per_input,
                    "bytes_written_estimate": 4,
                    "execution_status": "executed_source_tokenizer_raw_prompt",
                    "note": (
                        "Source tokenizer timing apportioned across input forwards; "
                        f"chat template status={self.chat_template_status}."
                    ),
                }
                sampled_token_id: int | None = None
                input_forward_count = 0
                for input_index, token_id in enumerate(input_ids):
                    _, trace, profile, sampled_token_id = self.forward_token_profiled(
                        int(token_id),
                        phase="prefill",
                        token_ordinal=input_index,
                        tokenizer_allocation=tokenizer_allocation,
                    )
                    records.append(profile)
                    route_frequency.update(int(route) for route in trace["routes"])
                    input_forward_count += 1
                assert sampled_token_id is not None
                decode_forward_count = 0
                stop_reason = "decode_forward_limit"
                for decode_index in range(decode_forwards):
                    if self.eos_id is not None and sampled_token_id == self.eos_id:
                        stop_reason = "eos_before_requested_decode_forward"
                        break
                    _, trace, profile, sampled_token_id = self.forward_token_profiled(
                        sampled_token_id,
                        phase="decode",
                        token_ordinal=decode_index,
                    )
                    records.append(profile)
                    route_frequency.update(int(route) for route in trace["routes"])
                    decode_forward_count += 1
                trial_summaries.append(
                    {
                        "trial": trial + 1,
                        "prompt_text_disclosed": False,
                        "prompt_utf8_bytes": len(prompt_bytes),
                        "input_token_count": len(input_ids),
                        "input_forward_count": input_forward_count,
                        "decode_forward_count": decode_forward_count,
                        "stop_reason": stop_reason,
                    }
                )
        return {
            "schema": "hawking.gravity.deepseek_v4.diagnostic_complete_token_profile_run.v1",
            "status": "REAL_LAYER4_CPU_DIAGNOSTIC_FORWARDS_PROFILED",
            "trials": trial_summaries,
            "token_profiles": records,
            "route_frequency": {str(route): count for route, count in sorted(route_frequency.items())},
            "execution_policy": {
                "state_reset_between_trials": True,
                "decoded_weight_cache_retained_within_process": True,
                "prompt_text_disclosed": False,
                "endpoint_hcli_streaming": "unavailable_in_inprocess_profile",
            },
        }


def _profile_percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise DeepSeekV4GravityError("complete-token profile has no values to aggregate")
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _aggregate_complete_token_profile(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate only explicit stages from a bounded in-process profile."""

    if not records:
        raise DeepSeekV4GravityError("complete-token profile produced no real forwards")
    by_stage: dict[str, list[Mapping[str, Any]]] = {name: [] for name in _COMPLETE_TOKEN_PROFILE_STAGES}
    complete_wall: list[float] = []
    complete_cpu: list[float] = []
    endpoint_unavailable = 0
    for record in records:
        timing = record.get("timing_accounting")
        stages = record.get("stage_metrics")
        if not isinstance(timing, Mapping) or not isinstance(stages, list):
            raise DeepSeekV4GravityError("malformed complete-token profile record")
        if float(timing.get("unexplained_other_wall_elapsed_ms", -1.0)) != 0.0:
            raise DeepSeekV4GravityError("complete-token profile contains unexplained wall time")
        if float(timing.get("unexplained_other_cpu_duration_ms", -1.0)) != 0.0:
            raise DeepSeekV4GravityError("complete-token profile contains unexplained CPU time")
        complete_wall.append(float(timing["observed_complete_token_wall_elapsed_ms"]))
        complete_cpu.append(float(timing["observed_complete_token_cpu_duration_ms"]))
        seen: set[str] = set()
        for stage in stages:
            if not isinstance(stage, Mapping):
                raise DeepSeekV4GravityError("complete-token profile contains a non-object stage")
            name = stage.get("stage")
            if not isinstance(name, str) or name not in by_stage or name in seen:
                raise DeepSeekV4GravityError("complete-token profile stage coverage is invalid")
            seen.add(name)
            by_stage[name].append(stage)
            if name == "endpoint_hcli_streaming" and stage.get("execution_status") == "unavailable_in_inprocess_profile":
                endpoint_unavailable += 1
        if seen != set(_COMPLETE_TOKEN_PROFILE_STAGES):
            raise DeepSeekV4GravityError("complete-token profile omits one or more required stages")

    stage_aggregates: dict[str, Any] = {}
    for name, stage_rows in by_stage.items():
        stage_aggregates[name] = {
            "label": _COMPLETE_TOKEN_PROFILE_STAGE_LABELS[name],
            "cpu_duration_ms": _profile_percentiles(
                [float(row["cpu_duration_ms"]) for row in stage_rows]
            ),
            "cpu_wall_elapsed_ms": _profile_percentiles(
                [float(row["cpu_wall_elapsed_ms"]) for row in stage_rows]
            ),
            "bytes_read_estimate_total": sum(int(row["bytes_read_estimate"]) for row in stage_rows),
            "bytes_written_estimate_total": sum(int(row["bytes_written_estimate"]) for row in stage_rows),
            "fp_operations_estimate_total": sum(int(row["fp_operations_estimate"]) for row in stage_rows),
            "integer_bit_operations_estimate_total": sum(
                int(row["integer_bit_operations_estimate"]) for row in stage_rows
            ),
            "dispatches_total": sum(int(row["dispatches"]) for row in stage_rows),
            "command_buffers_total": sum(int(row["command_buffers"]) for row in stage_rows),
            "waits_total": sum(int(row["waits"]) for row in stage_rows),
            "execution_statuses": dict(
                sorted(Counter(str(row["execution_status"]) for row in stage_rows).items())
            ),
        }
    return {
        "real_diagnostic_forward_count": len(records),
        "complete_token_wall_elapsed_ms": _profile_percentiles(complete_wall),
        "complete_token_cpu_duration_ms": _profile_percentiles(complete_cpu),
        "stages": stage_aggregates,
        "gpu_dispatch_accounting": {
            "gpu_duration_ms_total": 0.0,
            "dispatches_total": 0,
            "command_buffers_total": 0,
            "waits_total": 0,
            "status": "CPU_NUMPY_DIAGNOSTIC_NO_GPU_COUNTERS_OR_DISPATCHES",
        },
        "timing_accounting": {
            "unexplained_other_wall_elapsed_ms_total": 0.0,
            "unexplained_other_cpu_duration_ms_total": 0.0,
            "other_share_percent": 0.0,
            "status": "PASS_NO_UNEXPLAINED_OTHER_BUCKET",
        },
        "endpoint_hcli_streaming": {
            "unavailable_in_inprocess_profile_records": endpoint_unavailable,
            "status": "NOT_MEASURED_NO_HTTP_OR_HCLI_STREAM_FRAGMENT",
        },
    }


def profile_diagnostic_complete_token(
    artifact_dir: str | Path,
    prompt: str,
    *,
    trials: int,
    decode_forwards: int,
    out: str | Path,
) -> dict[str, Any]:
    """Seal a bounded stage profile without touching a sealed artifact history."""

    artifact = _absolute(artifact_dir, "artifact_dir")
    target = _absolute(out, "out")
    try:
        target.resolve().relative_to(artifact.resolve())
    except ValueError:
        pass
    else:
        raise DeepSeekV4GravityError(
            "complete-token profile out must be external to the sealed artifact directory"
        )
    runtime = DeepSeekV4DiagnosticRuntime(artifact)
    run = runtime.profile_prompt(prompt, trials=trials, decode_forwards=decode_forwards)
    profiles = run.get("token_profiles")
    if not isinstance(profiles, list):
        raise DeepSeekV4GravityError("diagnostic profile returned no token profile list")
    aggregate = _aggregate_complete_token_profile(profiles)
    report = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.complete_token_profile_receipt.v1",
            "status": "SEALED_REAL_LAYER4_CPU_DIAGNOSTIC_PROFILE_NOT_BASE_TRUE_TPS",
            "created_at": _utc_now(),
            "artifact": {
                "path": str(artifact),
                "seal_sha256": runtime.manifest["seal_sha256"],
                "schema": runtime.manifest["schema"],
                "diagnostic_scope": runtime.manifest["diagnostic_scope"],
                "runtime_adapter": runtime.manifest["runtime_adapter"],
            },
            "profile_run": run,
            "aggregate": aggregate,
            "operation_estimate_method": {
                "matrix_vector": "two floating-point operations per dense matrix element (multiply plus add)",
                "bytes": "logical source tensor and float32 activation traffic estimate; not a hardware cache, DRAM, or Metal counter",
                "integer_bit_operations": "only source-visible sort/argmax/index estimates; NumPy implementation internals are excluded",
            },
            "claim_boundary": {
                "real_source_derived_layer4_forwards": True,
                "full_43_layer_runtime": False,
                "source_cpu_parity": False,
                "numeric_parity_v2_1": False,
                "metal_dispatch": False,
                "base_true_tps": False,
                "hcli_tool_augmented_throughput": False,
                "reason": (
                    "This is a direct in-process profile of the 128-token/4-output CPU diagnostic; "
                    "it has no HCLI stream fragment, GPU dispatch, 8K context, 64-token warmup, or 512-token measurement."
                ),
            },
            "source_parent_retained": False,
            "resume_safe": True,
        }
    )
    _atomic_json(target, report)
    return report


def _latent_source_hash_binding(runtime: DeepSeekV4DiagnosticRuntime) -> dict[str, Any]:
    """Bind a latent receipt to the immutable model and local operator source."""

    source = runtime.manifest.get("source")
    if not isinstance(source, Mapping):
        raise DeepSeekV4GravityError("latent capture artifact lacks source provenance")
    repository = source.get("repository")
    revision = source.get("revision")
    assets = source.get("metadata_assets")
    if repository != REPOSITORY or revision != REVISION or not isinstance(assets, Mapping):
        raise DeepSeekV4GravityError("latent capture artifact is not bound to the pinned V4 source")
    metadata_hashes: dict[str, str] = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "inference/config.json", "inference/model.py", "inference/kernel.py"):
        item = assets.get(name)
        if not isinstance(item, Mapping) or not _hcli_live_is_hex_digest(item.get("sha256")):
            raise DeepSeekV4GravityError(f"latent capture source metadata hash missing for {name}")
        metadata_hashes[name] = str(item["sha256"])
    windows = source.get("source_windows")
    if not isinstance(windows, list) or not windows:
        raise DeepSeekV4GravityError("latent capture source windows are missing")
    window_hashes: list[str] = []
    for index, window in enumerate(windows, start=1):
        if not isinstance(window, Mapping) or not _hcli_live_is_hex_digest(
            window.get("streamed_full_file_sha256")
        ):
            raise DeepSeekV4GravityError(f"latent capture source window {index} lacks a digest")
        window_hashes.append(str(window["streamed_full_file_sha256"]))
    operator_path = Path(__file__).resolve()
    _regular_file(operator_path, "latent capture operator source")
    return {
        "model_id": MODEL_ID,
        "repository": repository,
        "revision": revision,
        "artifact_seal_sha256": runtime.manifest["seal_sha256"],
        "metadata_asset_sha256": metadata_hashes,
        "metadata_asset_set_sha256": _sha256(_canonical(metadata_hashes)),
        "source_window_full_file_sha256": window_hashes,
        "source_parent_retained": False,
        "operator_source_sha256": _sha256(operator_path.read_bytes()),
    }


def _latent_hcli_tool_action_summary(
    receipt_paths: Sequence[str | Path],
    *,
    artifact_seal_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Bind optional HCLI action metadata without copying prompts or events.

    HCLI audit receipts can contain raw goals or rich host details.  This
    extractor is intentionally a whitelist: it emits only receipt hashes,
    artifact binding, event-chain integrity, and numeric tool/verification
    counters needed at the child-body bridge boundary.
    """

    if not receipt_paths:
        return {
            "availability": "not_supplied_to_inprocess_diagnostic_capture",
            "tool_action_decisions": "unavailable_without_a_bound_hcli_audit_receipt",
            "raw_hcli_events_retained": False,
        }
    if len(receipt_paths) > MAX_DIAGNOSTIC_LATENT_HCLI_RECEIPTS:
        raise DeepSeekV4GravityError(
            f"latent capture accepts at most {MAX_DIAGNOSTIC_LATENT_HCLI_RECEIPTS} HCLI audit receipts"
        )
    seen: set[Path] = set()
    records: list[dict[str, Any]] = []
    for ordinal, candidate in enumerate(receipt_paths, start=1):
        path = _absolute(candidate, f"hcli_tool_action_receipt[{ordinal}]")
        if path == output_path:
            raise DeepSeekV4GravityError("latent capture out must not overwrite an HCLI audit receipt")
        if path in seen:
            raise DeepSeekV4GravityError(f"duplicate HCLI audit receipt: {path}")
        seen.add(path)
        _regular_file(path, f"hcli_tool_action_receipt[{ordinal}]")
        raw = path.read_bytes()
        if len(raw) > MAX_DIAGNOSTIC_LATENT_HCLI_RECEIPT_BYTES:
            raise DeepSeekV4GravityError(
                "HCLI audit receipt exceeds the bounded latent capture input cap"
            )
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeepSeekV4GravityError(
                f"cannot decode HCLI audit receipt {ordinal}: {exc}"
            ) from exc
        if not isinstance(document, Mapping):
            raise DeepSeekV4GravityError("HCLI audit receipt must be a JSON object")
        runtime = document.get("runtime")
        if not isinstance(runtime, Mapping):
            raise DeepSeekV4GravityError("HCLI audit receipt has no runtime binding")
        artifact_claims = [
            value
            for key in ("context_before", "context_after", "context")
            if isinstance(runtime.get(key), Mapping)
            for value in [runtime[key].get("artifact_seal_sha256")]
            if isinstance(value, str)
        ]
        if not artifact_claims or any(value != artifact_seal_sha256 for value in artifact_claims):
            raise DeepSeekV4GravityError(
                "HCLI audit receipt is not bound to this sealed diagnostic artifact"
            )
        agent = document.get("agent")
        if not isinstance(agent, Mapping):
            agent = {}
        tool_activity = agent.get("tool_activity")
        if not isinstance(tool_activity, Mapping):
            tool_activity = {}
        verification = agent.get("verification")
        if not isinstance(verification, Mapping):
            verification = {}
        verdict = verification.get("last_verdict")
        if not isinstance(verdict, Mapping):
            verdict = {}
        raw_failures = verdict.get("failures")
        if not isinstance(raw_failures, list):
            raw_failures = []
        failure_code_frequency: Counter[str] = Counter()
        failure_category_frequency: Counter[str] = Counter()
        for failure in raw_failures:
            if not isinstance(failure, Mapping):
                continue
            code = failure.get("code")
            category = failure.get("category")
            if isinstance(code, str):
                failure_code_frequency[code] += 1
            if isinstance(category, str):
                failure_category_frequency[category] += 1
        event_chain = document.get("event_chain")
        if not isinstance(event_chain, Mapping):
            event_chain = {}
        driver = document.get("driver")
        if not isinstance(driver, Mapping):
            driver = {}
        tool_counts = {
            key: int(value)
            for key in (
                "parsed_model_tool_calls",
                "durable_tool_call_events",
                "dispatched_model_tool_calls",
            )
            for value in [tool_activity.get(key, 0)]
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        }
        if len(tool_counts) != 3:
            raise DeepSeekV4GravityError("HCLI audit receipt has malformed tool activity counters")
        checked_events = event_chain.get("checked_events")
        if not isinstance(checked_events, int) or isinstance(checked_events, bool) or checked_events < 0:
            raise DeepSeekV4GravityError("HCLI audit receipt has malformed event-chain counter")
        records.append(
            {
                "ordinal": ordinal,
                "receipt_sha256": _sha256(raw),
                "schema": document.get("schema") if isinstance(document.get("schema"), str) else None,
                "status": document.get("status") if isinstance(document.get("status"), str) else None,
                "artifact_seal_sha256": artifact_seal_sha256,
                "event_chain": {
                    "ok": bool(event_chain.get("ok")),
                    "checked_events": checked_events,
                    "chain_root_sha256": (
                        event_chain.get("chain_root")
                        if _hcli_live_is_hex_digest(event_chain.get("chain_root"))
                        else None
                    ),
                },
                "tool_activity": tool_counts,
                "verification": {
                    "last_verdict_status": (
                        verdict.get("status") if isinstance(verdict.get("status"), str) else None
                    ),
                    "last_verdict_class": (
                        verdict.get("class") if isinstance(verdict.get("class"), str) else None
                    ),
                    "oracle": verdict.get("oracle") if isinstance(verdict.get("oracle"), str) else None,
                    "failure_code_frequency": dict(sorted(failure_code_frequency.items())),
                    "failure_category_frequency": dict(sorted(failure_category_frequency.items())),
                    "verify_result_events": (
                        int(verification["verify_result_events"])
                        if isinstance(verification.get("verify_result_events"), int)
                        and not isinstance(verification.get("verify_result_events"), bool)
                        else None
                    ),
                },
                "effect_policy_declared": isinstance(driver.get("compute_profile"), Mapping)
                and isinstance(driver["compute_profile"].get("effect_policy"), str),
                "raw_goal_prompt_or_event_retained": False,
            }
        )
    return {
        "availability": "bound_hcli_audit_metadata_whitelist",
        "receipt_count": len(records),
        "records": records,
        "raw_hcli_events_retained": False,
    }


def capture_diagnostic_latent_routes(
    artifact_dir: str | Path,
    *,
    out: str | Path,
    hcli_tool_action_receipts: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Seal one bounded bridge-oriented latent/route diagnostic receipt."""

    artifact = _absolute(artifact_dir, "artifact_dir")
    target = _absolute(out, "out")
    try:
        target.resolve().relative_to(artifact.resolve())
    except ValueError:
        pass
    else:
        raise DeepSeekV4GravityError(
            "latent route capture out must be external to the sealed artifact directory"
        )
    runtime = DeepSeekV4DiagnosticRuntime(artifact)
    source_binding = _latent_source_hash_binding(runtime)
    run = runtime.capture_latent_route_suite()
    hcli_metadata = _latent_hcli_tool_action_summary(
        hcli_tool_action_receipts,
        artifact_seal_sha256=runtime.manifest["seal_sha256"],
        output_path=target,
    )
    report = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.diagnostic_latent_route_receipt.v1",
            "status": "SEALED_BOUNDED_LAYER4_CPU_DIAGNOSTIC_LATENT_ROUTE_CAPTURE",
            "created_at": _utc_now(),
            "artifact": {
                "path": str(artifact),
                "seal_sha256": runtime.manifest["seal_sha256"],
                "schema": runtime.manifest["schema"],
                "diagnostic_scope": runtime.manifest["diagnostic_scope"],
                "runtime_adapter": runtime.manifest["runtime_adapter"],
            },
            "source_hash_binding": source_binding,
            "capture": run,
            "hcli_tool_action_metadata": hcli_metadata,
            "claim_boundary": {
                "real_source_derived_layer4_forwards": True,
                "full_43_layer_runtime": False,
                "source_cpu_parity": False,
                "numeric_parity_v2_1": False,
                "metal_dispatch": False,
                "base_true_tps": False,
                "hcli_tool_augmented_throughput": False,
                "donor_data_or_distillation": False,
                "reason": (
                    "The capture executes only the sealed full-width layer-4 NumPy diagnostic, "
                    "with a 128-token context ceiling, no source activation QAT/TileLang parity, "
                    "and no 43-layer runtime. It stores bounded statistics and small qualified "
                    "trace shards, never raw prompts, completions, or hidden-state arrays."
                ),
            },
            "source_parent_retained": False,
            "resume_safe": True,
        }
    )
    _atomic_json(target, report)
    return report


def _runtime_context(runtime: DeepSeekV4DiagnosticRuntime) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID + ":layer4-diagnostic",
        "arch": "deepseek_v4_layer4_diagnostic",
        "ctx_len_native": 128,
        "ctx_len_effective": 128,
        "tq_multiplier": 1.0,
        "tq_estimated": False,
        "recurrent_state_bytes": None,
        "active_slots": 0,
        "free_slots": 1,
        "max_batch": 1,
        "max_output_tokens": 4,
        "artifact_seal_sha256": runtime.manifest["seal_sha256"],
        "capability_status": "diagnostic_cpu_only_not_tg_eligible",
        "metal_dispatches": 0,
        "chat_template": runtime.chat_template_status,
    }


def serve_diagnostic(artifact_dir: str | Path, addr: str) -> None:
    """Serve a sealed diagnostic using the same native endpoint HCLI consumes."""

    if ":" not in addr:
        raise DeepSeekV4GravityError("--addr must be HOST:PORT")
    host, port_text = addr.rsplit(":", 1)
    try:
        port = int(port_text)
    except ValueError as exc:
        raise DeepSeekV4GravityError("--addr port must be an integer") from exc
    runtime = DeepSeekV4DiagnosticRuntime(artifact_dir)

    class Handler(BaseHTTPRequestHandler):
        server_version = "HawkingDeepSeekV4Diagnostic/1"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def _json(self, status: int, value: Mapping[str, Any]) -> None:
            raw = _canonical(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._json(200, {"ready": True, "runtime": _runtime_context(runtime)})
            elif self.path == "/v1/hawking/context":
                self._json(200, _runtime_context(runtime))
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            length = self.headers.get("Content-Length")
            try:
                count = int(length or "0")
                if count <= 0 or count > 8 * 1024**2:
                    raise ValueError
                body = json.loads(self.rfile.read(count).decode("utf-8"))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                self._json(400, {"error": {"message": "invalid JSON request"}})
                return
            if self.path == "/v1/hawking/generate":
                self._native_generate(body)
            elif self.path == "/v1/chat/completions":
                self._chat_generate(body)
            else:
                self._json(404, {"error": {"message": "not found"}})

        def _run(self, body: Mapping[str, Any]) -> dict[str, Any]:
            prompt = body.get("prompt")
            chat_template = "raw_prompt"
            if not isinstance(prompt, str):
                messages = body.get("messages")
                if isinstance(messages, list):
                    prompt, chat_template = runtime.render_chat_messages(messages)
            if not isinstance(prompt, str):
                raise DeepSeekV4GravityError("request requires prompt or messages")
            max_tokens = body.get("max_tokens", 1)
            if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
                raise DeepSeekV4GravityError("max_tokens must be an integer")
            result = runtime.generate(prompt, max_tokens)
            result["stats"]["chat_template"] = chat_template
            return result

        def _native_generate(self, body: Mapping[str, Any]) -> None:
            try:
                result = self._run(body)
            except DeepSeekV4GravityError as exc:
                self._json(422, {"error": {"message": str(exc)}})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            # The endpoint emits a complete tokenizer-decoded response as one
            # SSE fragment.  HCLI accepts fragments and terminal stats without
            # treating this adapter's CPU batching choice as token timing.
            if result["text"]:
                frame = _canonical({"tok_index": 0, "text": result["text"]})
                self.wfile.write(b"data: " + frame + b"\n\n")
            stats = result["stats"]
            frame = _canonical(
                {
                    "stats": {
                        "prompt_tokens": stats["prompt_tokens"],
                        "completion_tokens": stats["completion_tokens"],
                        "decode_ms": stats["decode_ms"],
                        "completed_decode_forwards": stats["completed_decode_forwards"],
                        "dec_tps": (
                            stats["completed_decode_forwards"] * 1_000.0 / stats["decode_ms"]
                            if stats["completed_decode_forwards"] and stats["decode_ms"] > 0
                            else None
                        ),
                        "runtime_receipt": {
                            "artifact_seal_sha256": runtime.manifest["seal_sha256"],
                            "fallback": stats["fallback"],
                            "metal_dispatches": 0,
                        },
                    }
                }
            )
            self.wfile.write(b"data: " + frame + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _chat_generate(self, body: Mapping[str, Any]) -> None:
            try:
                result = self._run(body)
            except DeepSeekV4GravityError as exc:
                self._json(422, {"error": {"message": str(exc)}})
                return
            self._json(
                200,
                {
                    "id": "deepseek-v4-diagnostic",
                    "object": "chat.completion",
                    "model": MODEL_ID + ":layer4-diagnostic",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": result["text"]}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(result["input_ids"]), "completion_tokens": len(result["completion_ids"])},
                    "hawking_stats": result["stats"],
                },
            )

    server = ThreadingHTTPServer((host, port), Handler)
    print(json.dumps({"schema": "hawking.gravity.deepseek_v4.endpoint.v1", "status": "READY", "addr": addr, "runtime": _runtime_context(runtime)}, sort_keys=True), flush=True)
    server.serve_forever()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="stream and seal the complete layer-4 diagnostic")
    build.add_argument("--artifact-dir", required=True)
    build.add_argument("--workspace-root", required=True)
    build.add_argument("--xet-root", required=True)
    build.add_argument("--protected-floor-bytes", type=int, default=MIN_FREE_FLOOR_BYTES)
    build.add_argument("--range-bytes", type=int, default=DEFAULT_RANGE_BYTES)
    full = commands.add_parser("build-full", help="stream and seal all 46 source shards without claiming runtime readiness")
    full.add_argument("--artifact-dir", required=True)
    full.add_argument("--workspace-root", required=True)
    full.add_argument("--xet-root", required=True)
    full.add_argument("--protected-floor-bytes", type=int, default=MIN_FREE_FLOOR_BYTES)
    full.add_argument("--range-bytes", type=int, default=DEFAULT_RANGE_BYTES)
    full.add_argument(
        "--parallel-workers",
        type=int,
        default=4,
        help=f"bounded concurrent source-shard workers (1..{MAX_PARALLEL_WORKERS}; default 4)",
    )
    inspect = commands.add_parser("inspect", help="verify a sealed V4 diagnostic manifest")
    inspect.add_argument("--artifact-dir", required=True)
    inspect_full = commands.add_parser("inspect-full", help="inspect the sealed full-model stream control plane")
    inspect_full.add_argument("--artifact-dir", required=True)
    reverify_full = commands.add_parser("reverify-full", help="hash every chunk in the sealed full-model stream")
    reverify_full.add_argument("--artifact-dir", required=True)
    reverify_full.add_argument("--out", type=Path, help="optional path for the sealed reverify receipt")
    runtime_blocker = commands.add_parser(
        "full-runtime-blocker",
        help="seal the exact missing full 43-layer runtime milestone without relabelling the stream",
    )
    runtime_blocker.add_argument("--artifact-dir", required=True)
    runtime_blocker.add_argument("--out", type=Path, required=True)
    residency = commands.add_parser(
        "static-expert-residency",
        help=(
            "seal manifest-only per-layer active-byte, expert-residency, and state contracts; "
            "it does not load or benchmark the full runtime"
        ),
    )
    residency.add_argument("--full-artifact-dir", required=True)
    residency.add_argument(
        "--layer4-latent-route-receipt",
        help="optional sealed bounded layer-4 route receipt; only aggregate privacy-safe observations are imported",
    )
    residency.add_argument("--out", required=True)
    gate = commands.add_parser("tps-gate", help="seal why a CPU diagnostic measurement is not Base True TPS")
    gate.add_argument("--artifact-dir", required=True)
    gate.add_argument("--benchmark-receipt", required=True)
    gate.add_argument("--out", required=True)
    profile = commands.add_parser(
        "profile-complete-token",
        help="seal bounded per-stage CPU accounting for real layer-4 diagnostic forwards",
    )
    profile.add_argument("--artifact-dir", required=True)
    profile.add_argument("--prompt", required=True)
    profile.add_argument(
        "--trials",
        type=int,
        default=1,
        help=f"real diagnostic trials (1..{MAX_DIAGNOSTIC_PROFILE_TRIALS}; default 1)",
    )
    profile.add_argument(
        "--decode-forwards",
        type=int,
        default=1,
        help=(
            "additional sampled-token forwards ("
            f"0..{MAX_DIAGNOSTIC_PROFILE_DECODE_FORWARDS}; default 1)"
        ),
    )
    profile.add_argument("--out", required=True)
    latent_routes = commands.add_parser(
        "capture-latent-routes",
        help=(
            "seal a bounded hash-only layer-4 latent/route bridge capture; "
            "it is diagnostic-only and not an evaluation or distillation run"
        ),
    )
    latent_routes.add_argument("--artifact-dir", required=True)
    latent_routes.add_argument(
        "--hcli-tool-action-receipt",
        action="append",
        default=[],
        help=(
            "optional absolute sealed HCLI audit receipt; repeat up to "
            f"{MAX_DIAGNOSTIC_LATENT_HCLI_RECEIPTS} times for whitelisted tool/action metadata"
        ),
    )
    latent_routes.add_argument("--out", required=True)
    live_suite = commands.add_parser(
        "hcli-live-suite",
        help="seal bounded HCLI live evidence without disclosing prompt text or promoting diagnostic capability",
    )
    live_suite.add_argument("--artifact-dir", required=True)
    live_suite.add_argument("--endpoint-capabilities", required=True)
    live_suite.add_argument(
        "--evidence",
        action="append",
        required=True,
        help="absolute JSON evidence path; repeat for each HCLI result or receipt",
    )
    live_suite.add_argument("--out", required=True)
    freeze_baseline = commands.add_parser(
        "freeze-child-baseline",
        help="verify canonical DeepSeek-V4 evidence and freeze the fixed eight-file child baseline bundle",
    )
    freeze_baseline.add_argument("--full-artifact-dir", required=True)
    freeze_baseline.add_argument("--diagnostic-artifact-dir", required=True)
    freeze_baseline.add_argument(
        "--complete-token-profile",
        required=True,
        help=f"canonical v3 receipt ({_CANONICAL_COMPLETE_TOKEN_PROFILE_V3})",
    )
    freeze_baseline.add_argument(
        "--hcli-live-suite",
        required=True,
        help=f"canonical v2 receipt ({_CANONICAL_HCLI_LIVE_SUITE_V2})",
    )
    freeze_baseline.add_argument("--full-runtime-blocker", required=True)
    freeze_baseline.add_argument(
        "--fp8-metal-component-probe",
        required=True,
        help="sealed source-native FP8 Metal component parity receipt; it remains component-only evidence",
    )
    freeze_baseline.add_argument("--component-trace", required=True)
    freeze_baseline.add_argument("--base-tps-gate", required=True)
    freeze_baseline.add_argument("--full-reverify", required=True)
    freeze_baseline.add_argument("--out-dir", required=True)
    freeze_baseline_v3 = commands.add_parser(
        "freeze-child-baseline-v3",
        help=(
            "derive a new eight-file v3 child-baseline revision from immutable v2 evidence "
            "and sealed component/admission/residency/roofline receipts"
        ),
    )
    freeze_baseline_v3.add_argument(
        "--prior-baseline-v2-dir",
        required=True,
        help="immutable directory containing exactly the eight sealed v2 baseline files",
    )
    freeze_baseline_v3.add_argument(
        "--fp4-metal-component-probe",
        required=True,
        help=f"canonical {_CANONICAL_FP4_METAL_COMPONENT_PROBE} receipt",
    )
    freeze_baseline_v3.add_argument(
        "--full-stream-reader-admission",
        required=True,
        help=f"canonical {_CANONICAL_FULL_STREAM_READER_ADMISSION} receipt",
    )
    freeze_baseline_v3.add_argument(
        "--raw-weight-simdgroup-splitk-sweep",
        required=True,
        help=(
            "fresh canonical reissue required at "
            f"{_CANONICAL_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP_CANONICAL_V1}; "
            f"historical {_CANONICAL_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP_V2} is rejected"
        ),
    )
    freeze_baseline_v3.add_argument(
        "--bounded-latent-route-receipt",
        required=True,
        help=f"canonical {_CANONICAL_BOUNDED_LATENT_ROUTE_V2} receipt",
    )
    freeze_baseline_v3.add_argument(
        "--static-expert-residency",
        required=True,
        help=f"canonical {_CANONICAL_STATIC_EXPERT_RESIDENCY_V2} receipt",
    )
    freeze_baseline_v3.add_argument(
        "--metal-device-copy-roofline",
        required=True,
        help=f"canonical {_CANONICAL_METAL_DEVICE_COPY_ROOFLINE_V1} receipt",
    )
    freeze_baseline_v3.add_argument(
        "--act-quant-wq-a-cpu-oracle",
        required=True,
        help=f"canonical {_CANONICAL_ACT_QUANT_WQ_A_CPU_ORACLE_V2} receipt",
    )
    freeze_baseline_v3.add_argument(
        "--source-forward-extension-receipt",
        help=(
            "optional future GPU source-linear/layer-0-prefix sealed extension; "
            "it is bound but cannot promote v3 full-runtime or TPS claims"
        ),
    )
    freeze_baseline_v3.add_argument("--out-dir", required=True)
    generate = commands.add_parser("generate", help="run one real source-derived diagnostic generation")
    generate.add_argument("--artifact-dir", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-tokens", type=int, default=1)
    generate.add_argument("--trace-out", type=Path)
    serve = commands.add_parser("serve", help="start the local HCLI-compatible diagnostic endpoint")
    serve.add_argument("--artifact-dir", required=True)
    serve.add_argument("--addr", default="127.0.0.1:8787")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            result = build_diagnostic(
                artifact_dir=args.artifact_dir,
                workspace_root=args.workspace_root,
                xet_root=args.xet_root,
                protected_floor_bytes=args.protected_floor_bytes,
                range_bytes=args.range_bytes,
            )
        elif args.command == "build-full":
            result = build_full_model(
                artifact_dir=args.artifact_dir,
                workspace_root=args.workspace_root,
                xet_root=args.xet_root,
                protected_floor_bytes=args.protected_floor_bytes,
                range_bytes=args.range_bytes,
                parallel_workers=args.parallel_workers,
            )
        elif args.command == "inspect":
            result = inspect_artifact(args.artifact_dir)
        elif args.command == "inspect-full":
            result = inspect_full_stream(args.artifact_dir)
        elif args.command == "reverify-full":
            result = reverify_full_model(args.artifact_dir)
            if args.out is not None:
                _atomic_json(_absolute(args.out, "out"), result)
        elif args.command == "full-runtime-blocker":
            result = full_runtime_blocker_report(args.artifact_dir, args.out)
        elif args.command == "static-expert-residency":
            result = static_expert_residency_report(
                args.full_artifact_dir,
                out=args.out,
                layer4_latent_route_receipt=args.layer4_latent_route_receipt,
            )
        elif args.command == "tps-gate":
            result = tps_gate_report(args.artifact_dir, args.benchmark_receipt, args.out)
        elif args.command == "profile-complete-token":
            result = profile_diagnostic_complete_token(
                args.artifact_dir,
                args.prompt,
                trials=args.trials,
                decode_forwards=args.decode_forwards,
                out=args.out,
            )
        elif args.command == "capture-latent-routes":
            result = capture_diagnostic_latent_routes(
                args.artifact_dir,
                out=args.out,
                hcli_tool_action_receipts=args.hcli_tool_action_receipt,
            )
        elif args.command == "hcli-live-suite":
            result = hcli_live_suite_receipt(
                args.artifact_dir,
                args.endpoint_capabilities,
                args.evidence,
                args.out,
            )
        elif args.command == "freeze-child-baseline":
            result = freeze_child_baseline(
                full_artifact_dir=args.full_artifact_dir,
                diagnostic_artifact_dir=args.diagnostic_artifact_dir,
                complete_token_profile=args.complete_token_profile,
                hcli_live_suite=args.hcli_live_suite,
                full_runtime_blocker=args.full_runtime_blocker,
                fp8_metal_component_probe=args.fp8_metal_component_probe,
                component_trace=args.component_trace,
                base_tps_gate=args.base_tps_gate,
                full_reverify=args.full_reverify,
                out_dir=args.out_dir,
            )
        elif args.command == "freeze-child-baseline-v3":
            result = freeze_child_baseline_v3(
                prior_baseline_v2_dir=args.prior_baseline_v2_dir,
                fp4_metal_component_probe=args.fp4_metal_component_probe,
                full_stream_reader_admission=args.full_stream_reader_admission,
                raw_weight_simdgroup_splitk_sweep=args.raw_weight_simdgroup_splitk_sweep,
                bounded_latent_route_receipt=args.bounded_latent_route_receipt,
                static_expert_residency=args.static_expert_residency,
                metal_device_copy_roofline=args.metal_device_copy_roofline,
                act_quant_wq_a_cpu_oracle=args.act_quant_wq_a_cpu_oracle,
                source_forward_extension_receipt=args.source_forward_extension_receipt,
                out_dir=args.out_dir,
            )
        elif args.command == "generate":
            runtime = DeepSeekV4DiagnosticRuntime(args.artifact_dir)
            result = seal(
                {
                    "schema": "hawking.gravity.deepseek_v4.diagnostic_generation.v1",
                    "status": "FIRST_TOKEN_GENERATED_DIAGNOSTIC",
                    "artifact_seal_sha256": runtime.manifest["seal_sha256"],
                    "result": runtime.generate(args.prompt, args.max_tokens),
                }
            )
            if args.trace_out is not None:
                _atomic_json(_absolute(args.trace_out, "trace_out"), result)
        else:
            serve_diagnostic(args.artifact_dir, args.addr)
            return 0
    except (DeepSeekV4GravityError, OSError, ValueError) as exc:
        print(json.dumps({"schema": ARTIFACT_SCHEMA, "status": "NOT_COMPLETED", "reason": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

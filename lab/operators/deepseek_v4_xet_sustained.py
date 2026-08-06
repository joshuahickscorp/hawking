"""Sustained, bounded public-path confirmation for DeepSeek-V4 transfers.

The original Xet autotune matrix is immutable.  This operator deliberately
creates a successor receipt instead of rewriting that result: it compares the
frozen direct-presigned HTTP/1.1 profile against a ``curl --http2`` control,
then runs two long confirmations of the faster stable implementation.

Every transfer is the same sealed fixed corpus range.  Bodies are read through
a bounded pipe or a bounded Python range request, SHA-256 verified, subjected
to the native-format structural check, packed into a fsync'd test frame, and
evicted before the next body is admitted.  It is therefore a public-network
and bounded-pack throughput measurement, *not* a claim of full Condense or a
validated 43-layer runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import stat
import subprocess
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.layout import RECORDS_ROOT, evidence_dir
from lab.receipts import SealIntegrityError, seal, verify
from lab.operators import deepseek_v4_xet_autotune as autotune


FOLLOWUP_SCHEMA = "hawking.gravity.deepseek_v4.public_path_sustained_followup.v1"
FOLLOWUP_WINNER_SCHEMA = "hawking.gravity.deepseek_v4.public_path_sustained_winner.v1"
FOLLOWUP_PARTIAL_SCHEMA = "hawking.gravity.deepseek_v4.public_path_sustained_partial.v1"
CHILD_SCHEMA = "hawking.gravity.deepseek_v4.public_path_sustained_child.v1"
FOLLOWUP_NAME = "TG_XET_PUBLIC_PATH_SUSTAINED_FOLLOWUP.json"
FOLLOWUP_WINNER_NAME = "TG_XET_PUBLIC_PATH_SUSTAINED_WINNER.json"
FOLLOWUP_PARTIAL_NAME = "TG_XET_PUBLIC_PATH_SUSTAINED_PARTIAL.jsonl"
FOLLOWUP_LAUNCHER_NAME = "DEEPSEEK_V4_FAST_STREAM_SUSTAINED_RESUME.sh"
SUSTAINED_ROOFLINE_NAME = "TG_XET_PUBLIC_PATH_ROOFLINE_SUSTAINED.json"
ACTIVE_CONFIG_NAME = "TG_XET_REAL_STREAM_ACTIVE_CONFIG.json"
DEFAULT_EVIDENCE_ROOT = evidence_dir("tg")
DEFAULT_RUN_ROOT = RECORDS_ROOT / "runs" / "deepseek-v4"
DEFAULT_PROBE_ROUNDS = 32
# The sealed corpus is exactly 65 MiB per round.  1,700 rounds transfer
# 110,500 MiB (~107.9 GiB) and take about eleven minutes at the currently
# measured public-path rate, which is long enough to expose settling rather
# than merely reporting the original twelve-second burst.
DEFAULT_SUSTAINED_ROUNDS = 1700
CHECKPOINT_SECONDS = 30.0
CURL_BATCH_RANGES = 48
_CURL_MARKER = re.compile(r"TG_CURL_TRANSFER:(\d{3}):([0-9.]+)")
_URL_RE = re.compile(r"https://[^\s]+")
_GENERATION_RE = re.compile(r"^[A-Z0-9_]{1,48}$")


class DeepSeekV4XetSustainedError(RuntimeError):
    """A sustained public-path result cannot be trusted or promoted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise DeepSeekV4XetSustainedError(f"{label} must be an absolute path")
    return Path(os.path.abspath(os.fspath(path)))


def _ensure_dir(path: Path, label: str) -> None:
    try:
        autotune._ensure_dir(path, label)
    except autotune.DeepSeekV4XetAutotuneError as exc:
        raise DeepSeekV4XetSustainedError(str(exc)) from exc


def _regular_file(path: Path, label: str) -> None:
    try:
        autotune._regular_file(path, label)
    except autotune.DeepSeekV4XetAutotuneError as exc:
        raise DeepSeekV4XetSustainedError(str(exc)) from exc


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return autotune._read_json(path, label)
    except autotune.DeepSeekV4XetAutotuneError as exc:
        raise DeepSeekV4XetSustainedError(str(exc)) from exc


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    try:
        return autotune._atomic_json(path, value)
    except autotune.DeepSeekV4XetAutotuneError as exc:
        raise DeepSeekV4XetSustainedError(str(exc)) from exc


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    try:
        autotune._append_jsonl(path, value)
    except autotune.DeepSeekV4XetAutotuneError as exc:
        raise DeepSeekV4XetSustainedError(str(exc)) from exc


def _safe_error(value: str) -> str:
    """Never persist presigned query strings in a diagnostic receipt."""

    return _URL_RE.sub("https://<redacted-presigned-host>/…", value).strip()[-1000:]


def _generated_name(name: str, generation: str | None) -> str:
    """Return an immutable retry-generation name, never overwriting a run."""

    if generation is None or not generation:
        return name
    if _GENERATION_RE.fullmatch(generation) is None:
        raise DeepSeekV4XetSustainedError(
            "generation must contain only uppercase A-Z, digits, and underscores"
        )
    path = Path(name)
    return f"{path.stem}_{generation}{path.suffix}"


def _load_frozen_inputs(
    *, winner_path: Path, corpus_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    winner = _read_json(winner_path, "frozen public-path winner")
    try:
        winner = verify(winner, label="frozen public-path winner")
    except SealIntegrityError as exc:
        raise DeepSeekV4XetSustainedError(str(exc)) from exc
    if winner.get("schema") != autotune.WINNER_SCHEMA or winner.get("status") != "FROZEN":
        raise DeepSeekV4XetSustainedError("the input public-path winner is not frozen")
    profile = winner.get("profile")
    if not isinstance(profile, Mapping):
        raise DeepSeekV4XetSustainedError("the input public-path winner has no profile")
    if profile.get("transport") != "direct_presigned_range" or not profile.get("connection_reuse"):
        raise DeepSeekV4XetSustainedError(
            "sustained follow-up requires the frozen direct-presigned reuse profile"
        )
    try:
        corpus = autotune.validate_fixed_corpus(_read_json(corpus_path, "fixed corpus"))
    except autotune.DeepSeekV4XetAutotuneError as exc:
        raise DeepSeekV4XetSustainedError(str(exc)) from exc
    return winner, dict(profile), corpus


def _same_metadata(
    bindings: autotune._SourceBindings, corpus: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    try:
        return autotune._validate_live_metadata(bindings, corpus)
    except autotune.DeepSeekV4XetAutotuneError as exc:
        raise DeepSeekV4XetSustainedError(str(exc)) from exc


def _floor_check(workspace: Path, *, additional_bytes: int, stage: str) -> None:
    try:
        autotune._floor_check(workspace, additional_bytes=additional_bytes, stage=stage)
    except autotune.DeepSeekV4XetAutotuneError as exc:
        raise DeepSeekV4XetSustainedError(str(exc)) from exc


class _ProgressGuard:
    """Aggregate stage timing while stopping on storage, swap, or thermal risk."""

    def __init__(self, *, workspace: Path, before: Mapping[str, Any]) -> None:
        self.workspace = workspace
        self.before = dict(before)
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.total_bytes = 0
        self.total_ranges = 0
        self.stop_reason: str | None = None
        self.next_sample = self.started + CHECKPOINT_SECONDS
        self.samples: list[dict[str, Any]] = []

    def completed(self, bytes_count: int) -> None:
        with self.lock:
            if self.stop_reason is not None:
                raise DeepSeekV4XetSustainedError(
                    f"pressure guard already stopped this candidate: {self.stop_reason}"
                )
            self.total_bytes += bytes_count
            self.total_ranges += 1
            now = time.monotonic()
            if now < self.next_sample:
                return
            sample = autotune.host_snapshot(self.workspace)
            reason = autotune._host_pressure_violation(self.before, sample)
            elapsed = max(0.001, now - self.started)
            self.samples.append(
                {
                    "elapsed_seconds": elapsed,
                    "completed_ranges": self.total_ranges,
                    "verified_sealed_evicted_bytes": self.total_bytes,
                    "sealed_and_evicted_mib_per_second": self.total_bytes / elapsed / 1024**2,
                    "host": sample,
                    "pressure_violation": reason,
                }
            )
            self.next_sample = now + CHECKPOINT_SECONDS
            if reason is not None:
                self.stop_reason = reason
                raise DeepSeekV4XetSustainedError(
                    f"pressure guard stopped this candidate: {reason}"
                )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "interval_seconds": CHECKPOINT_SECONDS,
                "samples": list(self.samples),
                "stop_reason": self.stop_reason,
            }


def _empty_metrics() -> dict[str, Any]:
    return {
        "range_count": 0,
        "source_bytes": 0,
        "packed_bytes": 0,
        "retry_count": 0,
        "fetch_wall_seconds_sum": 0.0,
        "verify_seconds_sum": 0.0,
        "decode_seconds_sum": 0.0,
        "pack_seal_seconds_sum": 0.0,
        "evict_seconds_sum": 0.0,
        "hosts": set(),
        "status_checks": [],
        "curl_commands": 0,
    }


def _merge_metrics(into: dict[str, Any], other: Mapping[str, Any]) -> None:
    for key in (
        "range_count",
        "source_bytes",
        "packed_bytes",
        "retry_count",
        "fetch_wall_seconds_sum",
        "verify_seconds_sum",
        "decode_seconds_sum",
        "pack_seal_seconds_sum",
        "evict_seconds_sum",
        "curl_commands",
    ):
        into[key] += other[key]
    into["hosts"].update(other["hosts"])
    into["status_checks"].extend(other["status_checks"])


def _record_processed(
    metrics: dict[str, Any],
    raw: bytes,
    target: Mapping[str, Any],
    *,
    fetch_seconds: float,
    retries: int,
    host: str,
    frames_root: Path,
    workspace: Path,
    guard: _ProgressGuard,
) -> None:
    try:
        result = autotune._process_and_evict(
            raw, target, frames_root=frames_root, workspace=workspace
        )
    except autotune.DeepSeekV4XetAutotuneError as exc:
        raise DeepSeekV4XetSustainedError(str(exc)) from exc
    timing = result["timings"]
    metrics["range_count"] += 1
    metrics["source_bytes"] += int(result["source_bytes"])
    metrics["packed_bytes"] += int(result["packed_bytes"])
    metrics["retry_count"] += retries
    metrics["fetch_wall_seconds_sum"] += fetch_seconds
    metrics["verify_seconds_sum"] += float(timing["verify_seconds"])
    metrics["decode_seconds_sum"] += float(timing["decode_seconds"])
    metrics["pack_seal_seconds_sum"] += float(timing["pack_seal_seconds"])
    metrics["evict_seconds_sum"] += float(timing["evict_seconds"])
    metrics["hosts"].add(host)
    guard.completed(int(result["source_bytes"]))


def _python_work_stealing_worker(
    *,
    jobs_queue: queue.SimpleQueue[list[dict[str, Any]]],
    bindings: autotune._SourceBindings,
    metadata: Mapping[str, Mapping[str, Any]],
    profile: Mapping[str, Any],
    frames_root: Path,
    workspace: Path,
    guard: _ProgressGuard,
) -> dict[str, Any]:
    transport = autotune._RangeTransport(bindings, metadata, profile)
    metrics = _empty_metrics()
    while True:
        try:
            batch = jobs_queue.get_nowait()
        except queue.Empty:
            break
        for target in batch:
            began = time.monotonic()
            try:
                raw, fetch = transport.fetch(target)
            except autotune.DeepSeekV4XetAutotuneError as exc:
                raise DeepSeekV4XetSustainedError(str(exc)) from exc
            _record_processed(
                metrics,
                raw,
                target,
                fetch_seconds=time.monotonic() - began,
                retries=int(fetch.get("retries", 0)),
                host=str(fetch.get("host") or "unobserved"),
                frames_root=frames_root,
                workspace=workspace,
                guard=guard,
            )
    return metrics


def _read_exact(handle: Any, expected: int) -> bytes:
    chunks: list[bytes] = []
    remaining = expected
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    if len(raw) != expected:
        raise DeepSeekV4XetSustainedError(
            f"curl range stream ended at {len(raw)} bytes; expected exactly {expected}"
        )
    return raw


def _curl_quote(value: str) -> str:
    """Quote one curl-config value without giving it shell semantics."""

    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _curl_config(
    target: Mapping[str, Any], metadata: Mapping[str, Any], retry: Mapping[str, float | int]
) -> str:
    """Build one in-memory curl config section.

    The signed location is deliberately supplied on curl's stdin, never its
    argv, environment, ledger, or a temporary config file.  That keeps the
    short-lived credential out of process listings while retaining curl's
    connection cache across the ``next``-separated transfers in a batch.
    """

    exact_range = f"{int(target['start'])}-{int(target['end']) - 1}"
    return "\n".join(
        (
            "http2",
            "fail",
            "silent",
            "show-error",
            "location",
            "max-redirs = 4",
            f"connect-timeout = {int(float(retry['connect_timeout']))}",
            f"max-time = {int(float(retry['read_timeout']))}",
            f"retry = {max(0, int(retry['max_attempts']) - 1)}",
            f"retry-delay = {max(1, int(float(retry['base_delay'])) + 1)}",
            f"retry-max-time = {max(1, int(float(retry['max_duration'])))}",
            f"range = {_curl_quote(exact_range)}",
            # Write status records only to stderr.  Source bodies stay on
            # stdout and are split by their sealed expected sizes below.
            "write-out = \"%{stderr}TG_CURL_TRANSFER:%{http_code}:%{size_download}\\\\n\"",
            f"url = {_curl_quote(str(metadata['signed_location']))}",
        )
    )


def _curl_protocol_probe(
    target: Mapping[str, Any], metadata: Mapping[str, Any], retry: Mapping[str, float | int]
) -> dict[str, Any]:
    # This is intentionally a 1 KiB control plane check.  It proves the
    # negotiated transport before the body measurement without contributing to
    # the steady-state byte rate.
    probe_end = min(int(target["end"]), int(target["start"]) + 1024)
    probe_target = dict(target)
    probe_target["end"] = probe_end
    config = _curl_config(probe_target, metadata, retry).replace(
        "write-out = \"%{stderr}TG_CURL_TRANSFER:%{http_code}:%{size_download}\\\\n\"",
        "write-out = \"%{http_version}\"\noutput = \"/dev/null\"",
    )
    completed = subprocess.run(
        ["curl", "--config", "-"], input=config, capture_output=True, text=True,
        timeout=180.0, check=False,
    )
    if completed.returncode != 0:
        raise DeepSeekV4XetSustainedError(
            "curl HTTP/2 protocol probe failed: " + _safe_error(completed.stderr)
        )
    version = completed.stdout.strip()
    if version != "2":
        raise DeepSeekV4XetSustainedError(
            f"curl --http2 did not negotiate HTTP/2 on the signed source endpoint (got {version!r})"
        )
    return {"requested": "HTTP/2", "negotiated_http_version": version, "bytes": probe_end - int(target["start"])}


def _curl_work_stealing_worker(
    *,
    jobs_queue: queue.SimpleQueue[list[dict[str, Any]]],
    metadata: Mapping[str, Mapping[str, Any]],
    profile: Mapping[str, Any],
    frames_root: Path,
    workspace: Path,
    guard: _ProgressGuard,
) -> dict[str, Any]:
    retry = autotune._retry_settings(profile)
    metrics = _empty_metrics()
    while True:
        try:
            batch = jobs_queue.get_nowait()
        except queue.Empty:
            break
        batch_size = len(batch)
        if not batch_size:
            continue
        sections: list[str] = []
        for index, target in enumerate(batch):
            if index:
                sections.append("next")
            sections.append(_curl_config(target, metadata[str(target["shard"])], retry))
        config = "\n".join(sections) + "\n"
        began = time.monotonic()
        process = subprocess.Popen(
            ["curl", "--config", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise DeepSeekV4XetSustainedError("could not capture bounded curl streams")
        try:
            process.stdin.write(config.encode("utf-8"))
            process.stdin.close()
            for target in batch:
                raw = _read_exact(process.stdout, int(target["length"]))
                item_metadata = metadata[str(target["shard"])]
                host = urllib.parse.urlsplit(str(item_metadata["signed_location"])).netloc
                _record_processed(
                    metrics,
                    raw,
                    target,
                    fetch_seconds=0.0,
                    retries=0,
                    host=host,
                    frames_root=frames_root,
                    workspace=workspace,
                    guard=guard,
                )
            trailing = process.stdout.read(1)
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            returncode = process.wait(timeout=180.0)
        except BaseException:
            process.kill()
            process.wait(timeout=30.0)
            raise
        elapsed = time.monotonic() - began
        if trailing:
            raise DeepSeekV4XetSustainedError("curl wrote bytes beyond the sealed range sequence")
        if returncode != 0:
            raise DeepSeekV4XetSustainedError(
                f"curl HTTP/2 bounded batch failed with code {returncode}: {_safe_error(stderr)}"
            )
        statuses = _CURL_MARKER.findall(stderr)
        if len(statuses) != batch_size:
            raise DeepSeekV4XetSustainedError(
                f"curl status accounting is incomplete ({len(statuses)}/{batch_size} exact ranges)"
            )
        for target, (status, body_size) in zip(batch, statuses, strict=True):
            expected = int(target["length"])
            if int(status) != 206 or abs(float(body_size) - expected) > 0.5:
                raise DeepSeekV4XetSustainedError(
                    "curl did not report an exact HTTP 206 range of the sealed expected size"
                )
        # The transfer/processing overlap is intentional: curl writes through
        # a bounded OS pipe and blocks while the current body is being sealed.
        # Attribute the command wall time once, outside the fully overlapped
        # per-range critical path, for observability rather than scoring.
        metrics["fetch_wall_seconds_sum"] += elapsed
        metrics["status_checks"].extend({"http_status": int(status), "bytes": float(size)} for status, size in statuses)
        metrics["curl_commands"] += 1
    return metrics


def _balanced_rotating_batches(
    targets: Sequence[Mapping[str, Any]], *, rounds: int
) -> list[list[dict[str, Any]]]:
    """Create equal-byte work units with rotating source order.

    The first sustained implementation assigned one shard per worker.  That
    was fine for a short probe but causes small source windows to drain early.
    Here every batch contains six complete corpus rounds (48 exact ranges), so
    each worker receives the same 390 MiB logical work and steals the next
    batch when it finishes.  The per-round rotation spreads CDN hosts without
    changing a single byte, hash, or number of reads.
    """

    if not targets or rounds < 1 or CURL_BATCH_RANGES % len(targets):
        raise DeepSeekV4XetSustainedError("balanced scheduler needs a non-empty corpus that divides curl batch size")
    all_jobs: list[dict[str, Any]] = []
    ordered = [dict(target) for target in targets]
    for number in range(rounds):
        offset = number % len(ordered)
        all_jobs.extend(dict(target) for target in ordered[offset:] + ordered[:offset])
    return [all_jobs[index : index + CURL_BATCH_RANGES] for index in range(0, len(all_jobs), CURL_BATCH_RANGES)]


def _assert_zero_source_cache(scratch_root: Path) -> dict[str, Any]:
    source_files: list[str] = []
    frame_files: list[str] = []
    for node in scratch_root.rglob("*"):
        if not node.is_file():
            continue
        relative = str(node.relative_to(scratch_root))
        if relative.startswith("frames/"):
            frame_files.append(relative)
        if (
            relative.startswith("xet-cache/")
            or relative.startswith("hub-cache/models--")
            or node.suffix in {".safetensors", ".bin", ".pt"}
        ):
            source_files.append(relative)
    if source_files or frame_files:
        raise DeepSeekV4XetSustainedError(
            "zero-cache/seal-before-evict assertion failed: "
            + ", ".join((source_files + frame_files)[:8])
        )
    return {"source_body_persisted": False, "source_cache_files": [], "test_frames_remaining": []}


def _curl_version() -> dict[str, Any]:
    completed = subprocess.run(["curl", "--version"], capture_output=True, text=True, timeout=30.0, check=False)
    if completed.returncode != 0:
        raise DeepSeekV4XetSustainedError("curl is unavailable for independent HTTP/2 control")
    lines = completed.stdout.splitlines()
    if not lines or "HTTP2" not in completed.stdout.upper():
        raise DeepSeekV4XetSustainedError("installed curl lacks the HTTP/2 feature required for this control")
    return {"version_line": lines[0], "features_line": lines[1] if len(lines) > 1 else ""}


def run_child(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Run one configured candidate, only in a child with Xet env set first."""

    if spec.get("schema") != "hawking.gravity.deepseek_v4.public_path_sustained_child_spec.v1":
        raise DeepSeekV4XetSustainedError("sustained child spec schema is invalid")
    implementation = spec.get("implementation")
    if implementation not in {"python_http11_reuse", "curl_http2_reuse"}:
        raise DeepSeekV4XetSustainedError("sustained child implementation is invalid")
    profile = spec.get("profile")
    if not isinstance(profile, Mapping):
        raise DeepSeekV4XetSustainedError("sustained child profile is invalid")
    workspace = _absolute(str(spec["workspace"]), "workspace")
    scratch_root = _absolute(str(spec["scratch_root"]), "scratch root")
    _ensure_dir(workspace, "workspace")
    _ensure_dir(scratch_root, "sustained child scratch root")
    try:
        corpus = autotune.validate_fixed_corpus(spec.get("corpus", {}))
    except autotune.DeepSeekV4XetAutotuneError as exc:
        raise DeepSeekV4XetSustainedError(str(exc)) from exc
    rounds = int(spec.get("rounds", 0))
    if rounds < 1:
        raise DeepSeekV4XetSustainedError("sustained child rounds must be positive")
    _floor_check(workspace, additional_bytes=autotune.MAX_RANGE_BYTES * 8, stage="sustained-child-admission")
    # Prove the child received its Xet environment before importing either
    # Hugging Face module.  The subsequent binding construction is therefore
    # part of the measured control plane, not an import-order accident.
    runtime = autotune._child_runtime_config()
    control_started = time.monotonic()
    bindings = autotune._SourceBindings()
    metadata = _same_metadata(bindings, corpus)
    control_seconds = time.monotonic() - control_started
    curl = _curl_version() if implementation == "curl_http2_reuse" else None
    if implementation == "curl_http2_reuse":
        protocol = _curl_protocol_probe(corpus["ranges"][0], metadata[str(corpus["ranges"][0]["shard"])], autotune._retry_settings(profile))
    else:
        protocol = {"requested": "HTTP/1.1 direct-presigned Python control", "negotiated_http_version": "not-observed-by-http.client"}
    before = autotune.host_snapshot(workspace)
    if before.get("contention_observed") is True:
        raise DeepSeekV4XetSustainedError("host is contended before sustained candidate; defer confirmation")
    guard = _ProgressGuard(workspace=workspace, before=before)
    frames_root = scratch_root / "frames"
    _ensure_dir(frames_root, "sustained temporary frame root")
    targets = sorted((dict(row) for row in corpus["ranges"]), key=lambda row: str(row["shard"]))
    batches = _balanced_rotating_batches(targets, rounds=rounds)
    jobs_queue: queue.SimpleQueue[list[dict[str, Any]]] = queue.SimpleQueue()
    for batch in batches:
        jobs_queue.put(batch)
    started = time.monotonic()
    worker_results: list[dict[str, Any]] = []
    worker_count = min(len(targets), len(batches))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="tg-public-path") as executor:
        futures = []
        for index in range(worker_count):
            worker_frames = frames_root / f"window-{index:02d}"
            _ensure_dir(worker_frames, "bounded source-window frame root")
            common = {
                "jobs_queue": jobs_queue,
                "metadata": metadata,
                "profile": profile,
                "frames_root": worker_frames,
                "workspace": workspace,
                "guard": guard,
            }
            if implementation == "python_http11_reuse":
                futures.append(executor.submit(_python_work_stealing_worker, bindings=bindings, **common))
            else:
                futures.append(executor.submit(_curl_work_stealing_worker, **common))
        for future in as_completed(futures):
            worker_results.append(future.result())
    elapsed = time.monotonic() - started
    after = autotune.host_snapshot(workspace)
    violation = autotune._host_pressure_violation(before, after)
    if violation is not None:
        raise DeepSeekV4XetSustainedError(f"local pressure guard stopped candidate: {violation}")
    storage = _assert_zero_source_cache(scratch_root)
    metrics = _empty_metrics()
    for result in worker_results:
        _merge_metrics(metrics, result)
    expected_ranges = len(targets) * rounds
    expected_bytes = sum(int(row["length"]) for row in targets) * rounds
    if metrics["range_count"] != expected_ranges or metrics["source_bytes"] != expected_bytes:
        raise DeepSeekV4XetSustainedError("not every scheduled sealed range was verified and evicted")
    rate = metrics["source_bytes"] / elapsed if elapsed > 0 else 0.0
    result = {
        "schema": CHILD_SCHEMA,
        "status": "PASS",
        "created_at": _utc_now(),
        "implementation": implementation,
        "source": {"repository": autotune.REPOSITORY, "revision": autotune.REVISION},
        "corpus_seal_sha256": corpus["seal_sha256"],
        "profile": dict(profile),
        "control_plane": {
            "metadata_dns_tls_control_seconds": control_seconds,
            "metadata_identity_revalidated": True,
            "protocol_control": protocol,
            "curl": curl,
        },
        "warmup": {
            "performed_in_separate_candidate_process": True,
            "excluded_from_steady_state": True,
            "note": "caller records a one-round warmup; this child runs steady state only",
        },
        "steady_state": {
            "rounds": rounds,
            "range_count": metrics["range_count"],
            "verified_decoded_test_frame_packed_sealed_evicted_bytes": metrics["source_bytes"],
            "test_frame_packed_bytes": metrics["packed_bytes"],
            "elapsed_seconds": elapsed,
            "sealed_and_evicted_bytes_per_second": rate,
            "sealed_and_evicted_mib_per_second": rate / 1024**2,
            "sealed_and_evicted_bytes_per_hour": rate * 3600.0,
            "retry_count": metrics["retry_count"],
            "retry_accounting": (
                "exact Python retry count"
                if implementation == "python_http11_reuse"
                else "curl command failures are exact; internal successful curl retries are not separately observable"
            ),
            "remote_host_distribution": sorted(metrics["hosts"]),
            "http_status_checks": {
                "count": len(metrics["status_checks"]),
                "all_exact_206": all(int(row["http_status"]) == 206 for row in metrics["status_checks"]),
            },
            "stage_cpu_seconds_sum": {
                "fetch_wall_sum": metrics["fetch_wall_seconds_sum"],
                "verify": metrics["verify_seconds_sum"],
                "decode": metrics["decode_seconds_sum"],
                "pack_seal": metrics["pack_seal_seconds_sum"],
                "evict": metrics["evict_seconds_sum"],
            },
            "curl_process_commands": metrics["curl_commands"],
            "test_frame_claim_boundary": (
                "benchmark frame exercises bounded verify/decode-check/pack/seal/evict; "
                "it is not a full Condense or validated 43-layer Gravity representation"
            ),
        },
        "storage": {
            **storage,
            "protected_floor_bytes": autotune.MIN_FREE_FLOOR_BYTES,
            "max_outer_source_windows": worker_count,
            "body_flow": "balanced rotating work-stealing batches; one body per worker plus bounded OS pipe; seal-before-evict before next body admission",
        },
        "host": {"before": before, "after": after, "checkpoint_samples": guard.snapshot()},
    }
    return seal(result)


def _child_main(args: argparse.Namespace) -> int:
    try:
        spec = _read_json(_absolute(args.spec, "child spec"), "child spec")
        sys.stdout.write(json.dumps(run_child(spec), sort_keys=True, separators=(",", ":")) + "\n")
        return 0
    except (DeepSeekV4XetSustainedError, SealIntegrityError) as exc:
        sys.stderr.write(f"deepseek-v4-xet-sustained-child-error: {_safe_error(str(exc))}\n")
        return 2


def _run_child(
    *, implementation: str, profile: Mapping[str, Any], corpus: Mapping[str, Any], workspace: Path,
    scratch_root: Path, rounds: int, timeout: float,
) -> dict[str, Any]:
    _ensure_dir(scratch_root, "sustained candidate scratch root")
    spec = {
        "schema": "hawking.gravity.deepseek_v4.public_path_sustained_child_spec.v1",
        "implementation": implementation,
        "profile": dict(profile),
        "corpus": dict(corpus),
        "workspace": str(workspace),
        "scratch_root": str(scratch_root),
        "rounds": rounds,
    }
    spec_path = scratch_root / "child-spec.json"
    _atomic_json(spec_path, spec)
    environment = autotune.profile_environment(profile, scratch_root)
    argv = [sys.executable, "-m", "lab.operators.deepseek_v4_xet_sustained", "--child", "--spec", str(spec_path)]
    try:
        completed = subprocess.run(
            argv, cwd=str(autotune.REPO_ROOT), env=environment, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise DeepSeekV4XetSustainedError(f"{implementation} child timed out after {timeout} seconds") from exc
    if completed.returncode != 0:
        raise DeepSeekV4XetSustainedError(
            f"{implementation} child failed: {_safe_error(completed.stderr)}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DeepSeekV4XetSustainedError(f"{implementation} child did not return one JSON object") from exc
    if not isinstance(result, dict):
        raise DeepSeekV4XetSustainedError(f"{implementation} child result is not an object")
    try:
        result = verify(result, label="sustained public-path child")
    except SealIntegrityError as exc:
        raise DeepSeekV4XetSustainedError(str(exc)) from exc
    if result.get("schema") != CHILD_SCHEMA or result.get("status") != "PASS":
        raise DeepSeekV4XetSustainedError("sustained child result schema/status is invalid")
    return result


def _warmup(
    *, implementation: str, profile: Mapping[str, Any], corpus: Mapping[str, Any], workspace: Path,
    scratch_root: Path,
) -> dict[str, Any]:
    # A fresh process intentionally prevents DNS/TLS/TCP setup from leaking
    # into the steady result.  It still verifies, packs, seals, and evicts its
    # source windows, never treating an unverified prefetch as a warmup.
    result = _run_child(
        implementation=implementation, profile=profile, corpus=corpus, workspace=workspace,
        scratch_root=scratch_root, rounds=1, timeout=900.0,
    )
    steady = result["steady_state"]
    return {
        "range_count": steady["range_count"],
        "verified_decoded_test_frame_packed_sealed_evicted_bytes": steady[
            "verified_decoded_test_frame_packed_sealed_evicted_bytes"
        ],
        "elapsed_seconds": steady["elapsed_seconds"],
        "excluded_from_steady_state": True,
        "child_seal_sha256": result["seal_sha256"],
    }


def _record_partial(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    value = seal({"schema": FOLLOWUP_PARTIAL_SCHEMA, "recorded_at": _utc_now(), **dict(event)})
    _append_jsonl(path, value)
    return value


def _partial_complete(path: Path, candidate_key: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    _regular_file(path, "sustained partial ledger")
    complete: dict[str, Any] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = verify(json.loads(raw), label="sustained partial ledger")
        except (json.JSONDecodeError, SealIntegrityError) as exc:
            raise DeepSeekV4XetSustainedError(f"sustained partial ledger is malformed: {exc}") from exc
        if row.get("candidate_key") == candidate_key and row.get("event") == "CANDIDATE_COMPLETE":
            complete = row
    return complete


def _candidate(
    *, key: str, implementation: str, phase: str, profile: Mapping[str, Any], corpus: Mapping[str, Any],
    workspace: Path, children_root: Path, rounds: int, partial_path: Path,
) -> dict[str, Any]:
    cached = _partial_complete(partial_path, key)
    if cached is not None:
        return dict(cached["candidate"])
    _record_partial(partial_path, {"event": "CANDIDATE_STARTED", "candidate_key": key, "phase": phase, "implementation": implementation, "rounds": rounds})
    scratch = children_root / key
    _ensure_dir(scratch, "sustained candidate scratch")
    try:
        warmup = _warmup(
            implementation=implementation, profile=profile, corpus=corpus, workspace=workspace,
            scratch_root=scratch / "warmup",
        )
        trial = _run_child(
            implementation=implementation, profile=profile, corpus=corpus, workspace=workspace,
            scratch_root=scratch / "steady", rounds=rounds,
            timeout=max(1800.0, rounds * 4.0),
        )
        candidate = {
            "status": "PASS",
            "candidate_key": key,
            "phase": phase,
            "implementation": implementation,
            "profile": dict(profile),
            "rounds": rounds,
            "warmup": warmup,
            "trial": trial,
        }
    except DeepSeekV4XetSustainedError as exc:
        candidate = {
            "status": "FAIL",
            "candidate_key": key,
            "phase": phase,
            "implementation": implementation,
            "profile": dict(profile),
            "rounds": rounds,
            "error": _safe_error(str(exc)),
        }
    _record_partial(partial_path, {"event": "CANDIDATE_COMPLETE", "candidate_key": key, "candidate": candidate})
    return candidate


def _rank(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rankable: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "PASS":
            continue
        trial = row.get("trial")
        if not isinstance(trial, Mapping):
            continue
        steady = trial.get("steady_state")
        host = trial.get("host")
        if not isinstance(steady, Mapping) or not isinstance(host, Mapping):
            continue
        before, after = host.get("before"), host.get("after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            continue
        if before.get("contention_observed") is True or after.get("contention_observed") is True:
            continue
        if int(steady.get("retry_count", 1)) != 0 or not steady.get("http_status_checks", {}).get("all_exact_206", False):
            continue
        rate = steady.get("sealed_and_evicted_bytes_per_second")
        if not isinstance(rate, (int, float)) or rate <= 0:
            continue
        rankable.append(dict(row))
    return sorted(rankable, key=lambda row: -float(row["trial"]["steady_state"]["sealed_and_evicted_bytes_per_second"]))


def _acquire_lock(evidence: Path) -> Path:
    path = evidence / ".TG_XET_PUBLIC_PATH_SUSTAINED.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise DeepSeekV4XetSustainedError("another sustained public-path run already owns the lease") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "created_at": _utc_now()}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _release_lock(path: Path) -> None:
    try:
        _regular_file(path, "sustained public-path lease")
        path.unlink()
    except FileNotFoundError:
        return


def _launcher_text(*, root: Path, winner: Path, artifact: Path, sustained: bool) -> str:
    if sustained:
        executable = "$ROOT/tools/condense/deepseek_v4_xet_sustained.py"
        command = "resume-real"
    else:
        executable = "$ROOT/tools/condense/deepseek_v4_xet_autotune.py"
        command = "resume-real"
    return f"""#!/bin/zsh
set -euo pipefail
ROOT={root}
PYTHON=\"$ROOT/tools/condense/.venv/bin/python\"
WINNER={winner}
ARTIFACT={artifact}
exec \"$PYTHON\" \"{executable}\" {command} \\
  --winner \"$WINNER\" --artifact-dir \"$ARTIFACT\" --workspace \"$ROOT/workspace\"
"""


def _write_launcher(path: Path, *, winner: Path, artifact: Path, sustained: bool) -> dict[str, Any]:
    raw = _launcher_text(
        root=autotune.REPO_ROOT, winner=winner, artifact=artifact, sustained=sustained
    ).encode("utf-8")
    try:
        digest = autotune._atomic_bytes(path, raw)
    except autotune.DeepSeekV4XetAutotuneError as exc:
        raise DeepSeekV4XetSustainedError(str(exc)) from exc
    os.chmod(path, 0o755)
    return {"path": str(path), "sha256": digest, "mode": "0755"}


def freeze_successor_artifacts(
    *, evidence_root: str | Path, run_root: str | Path, winner_path: str | Path,
    followup_path: str | Path, launcher_path: str | Path, generation: str | None,
) -> dict[str, Any]:
    """Write immutable roofline and active-config successors after promotion."""

    evidence = _absolute(evidence_root, "evidence root")
    runs = _absolute(run_root, "run root")
    winner_file = _absolute(winner_path, "sustained winner")
    followup_file = _absolute(followup_path, "sustained followup")
    launcher_file = _absolute(launcher_path, "sustained launcher")
    _ensure_dir(evidence, "evidence root")
    _ensure_dir(runs, "run root")
    _regular_file(launcher_file, "sustained launcher")
    winner = _read_json(winner_file, "sustained winner")
    followup = _read_json(followup_file, "sustained followup")
    try:
        winner = verify(winner, label="sustained winner")
        followup = verify(followup, label="sustained followup")
    except SealIntegrityError as exc:
        raise DeepSeekV4XetSustainedError(str(exc)) from exc
    if winner.get("schema") != FOLLOWUP_WINNER_SCHEMA or winner.get("status") != "FROZEN":
        raise DeepSeekV4XetSustainedError("successor artifacts require a frozen sustained winner")
    if followup.get("schema") != FOLLOWUP_SCHEMA or followup.get("status") != "COMPLETE_PROMOTED":
        raise DeepSeekV4XetSustainedError("successor artifacts require a promoted sustained followup")
    if winner.get("followup_seal_sha256") != followup.get("seal_sha256"):
        raise DeepSeekV4XetSustainedError("sustained winner/followup seal binding is invalid")
    confirmations = followup.get("sustained_confirmations")
    if not isinstance(confirmations, list) or len(confirmations) != 2:
        raise DeepSeekV4XetSustainedError("sustained followup lacks exactly two confirmation records")
    steady: list[Mapping[str, Any]] = []
    for row in confirmations:
        if not isinstance(row, Mapping) or row.get("status") != "PASS":
            raise DeepSeekV4XetSustainedError("sustained confirmation did not pass")
        trial = row.get("trial")
        if not isinstance(trial, Mapping) or not isinstance(trial.get("steady_state"), Mapping):
            raise DeepSeekV4XetSustainedError("sustained confirmation lacks steady-state metrics")
        steady.append(trial["steady_state"])
    rates = [float(row["sealed_and_evicted_mib_per_second"]) for row in steady]
    mean_mib = sum(rates) / len(rates)
    stable_spread = max(rates) / min(rates) - 1.0
    profile = winner.get("profile")
    if not isinstance(profile, Mapping):
        raise DeepSeekV4XetSustainedError("sustained winner profile is invalid")
    host = confirmations[0]["trial"].get("host")
    after = host.get("after") if isinstance(host, Mapping) else None
    launcher_digest = autotune._sha256(launcher_file.read_bytes())
    roofline = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.public_path_roofline_sustained.v1",
            "status": "MEASURED_FROZEN",
            "created_at": _utc_now(),
            "endpoint": "DEEPSEEK_V4_MAX_PUBLIC_STREAM_ACTIVE",
            "sustained_winner_seal_sha256": winner["seal_sha256"],
            "sustained_followup_seal_sha256": followup["seal_sha256"],
            "source": followup["fixed_corpus"],
            "configuration": {
                "implementation": winner.get("implementation"),
                "profile": dict(profile),
                "launcher": {"path": str(launcher_file), "sha256": launcher_digest},
            },
            "measured_public_path": {
                "fastest_stable_sealed_and_evicted_mib_per_second": mean_mib,
                "conservative_lower_confirmation_mib_per_second": min(rates),
                "fastest_stable_sealed_and_evicted_bytes_per_hour": mean_mib * 1024**2 * 3600.0,
                "confirmation_spread_fraction": stable_spread,
                "confirmation_count": 2,
                "retry_count_total": sum(int(row["retry_count"]) for row in steady),
                "binding": "public_network_or_remote_path_inferred_from_sustained_end_to_end_rate",
                "link_speed_is_not_wan_throughput": True,
                "historical_90_mib_per_second_comparison": "materially_beats_90_MiB_per_second",
            },
            "host_after": after,
            "claim_boundary": "This supersedes the prior short-run roofline for the same Mac/current public route; it is not a 10GbE WAN claim or full Condense-pack result.",
        }
    )
    active = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.real_stream_active_config.v1",
            "status": "CONFIGURATION_FROZEN_SOURCE_SUCCESSOR_BLOCKED",
            "created_at": _utc_now(),
            "endpoint": "DEEPSEEK_V4_MAX_PUBLIC_STREAM_ACTIVE",
            "sustained_winner_seal_sha256": winner["seal_sha256"],
            "sustained_roofline_seal_sha256": roofline["seal_sha256"],
            "launcher": {
                "path": str(launcher_file),
                "sha256": launcher_digest,
                "executable": bool(launcher_file.stat().st_mode & stat.S_IXUSR),
            },
            "source_window_policy": {
                "profile": dict(profile),
                "implementation": winner.get("implementation"),
                "source_cache_bytes": 0,
                "max_outer_source_windows": winner.get("real_stream_application", {}).get("outer_source_windows_maximum"),
                "seal_before_evict": True,
                "hard_free_floor_bytes": autotune.MIN_FREE_FLOOR_BYTES,
            },
            "handoff_state": "a sealed full source artifact exists but cannot be re-downloaded or evicted until an independently verified successor exists",
        }
    )
    roofline_path = evidence / _generated_name(SUSTAINED_ROOFLINE_NAME, generation)
    active_path = evidence / _generated_name(ACTIVE_CONFIG_NAME, generation)
    _atomic_json(roofline_path, roofline)
    _atomic_json(active_path, active)
    return {
        "roofline": roofline,
        "roofline_path": str(roofline_path),
        "active_config": active,
        "active_config_path": str(active_path),
    }


def resume_real_stream(
    *, winner_path: str | Path, artifact_dir: str | Path, workspace: str | Path
) -> dict[str, Any]:
    """Hand a promoted sustained transport to the guarded real-stream gate.

    This intentionally does not manufacture new source activity.  It has the
    same successor/eviction guard as the base launcher, but preserves the
    extended transport receipt that selected HTTP/2 or Python keep-alive.
    """

    winner_file = _absolute(winner_path, "sustained winner path")
    artifact = _absolute(artifact_dir, "artifact directory")
    workspace_path = _absolute(workspace, "workspace")
    winner = _read_json(winner_file, "sustained public-path winner")
    try:
        winner = verify(winner, label="sustained public-path winner")
    except SealIntegrityError as exc:
        raise DeepSeekV4XetSustainedError(str(exc)) from exc
    if winner.get("schema") != FOLLOWUP_WINNER_SCHEMA or winner.get("status") != "FROZEN":
        raise DeepSeekV4XetSustainedError("there is no frozen sustained public-path winner")
    profile = winner.get("profile")
    implementation = winner.get("implementation")
    if not isinstance(profile, Mapping) or implementation not in {"python_http11_reuse", "curl_http2_reuse"}:
        raise DeepSeekV4XetSustainedError("sustained winner lacks a safe transport configuration")
    progress = DEFAULT_RUN_ROOT / autotune.PROGRESS_NAME
    manifest_path = artifact / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path, "full stream manifest")
        if manifest.get("status") == "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY":
            event = {
                "event": "SUSTAINED_PUBLIC_PATH_HANDOFF_NOT_REDIRECTED",
                "status": "SEALED_SOURCE_WINDOWS_RETAINED_PENDING_INDEPENDENT_SUCCESSOR",
                "reason": (
                    "the only complete V4 source artifact is already sealed; re-downloading it would violate "
                    "no-parent-accumulation and cannot be presented as fresh work"
                ),
                "sustained_winner_seal_sha256": winner["seal_sha256"],
                "implementation": implementation,
                "profile": dict(profile),
                "artifact_manifest_seal_sha256": manifest.get("seal_sha256"),
            }
            autotune._record_progress(progress, event)
            return event
    _floor_check(workspace_path, additional_bytes=autotune.MAX_RANGE_BYTES * 8, stage="sustained-real-stream-preflight")
    event = {
        "event": "SUSTAINED_PUBLIC_PATH_REAL_STREAM_PRECHECK",
        "status": "READY_FOR_CONFIGURED_SOURCE_WINDOWS",
        "sustained_winner_seal_sha256": winner["seal_sha256"],
        "implementation": implementation,
        "profile": dict(profile),
        "artifact": str(artifact),
        "note": "A route-aware Gravity processor and independent successor remain required before source body work.",
    }
    autotune._record_progress(progress, event)
    return event


def run_followup(
    *, workspace: str | Path, evidence_root: str | Path, run_root: str | Path,
    frozen_winner_path: str | Path, corpus_path: str | Path,
    probe_rounds: int = DEFAULT_PROBE_ROUNDS, sustained_rounds: int = DEFAULT_SUSTAINED_ROUNDS,
    generation: str | None = None,
) -> dict[str, Any]:
    """Compare independent transports, then promote only two long confirmations."""

    if probe_rounds < 1 or sustained_rounds < 1:
        raise DeepSeekV4XetSustainedError("probe and sustained rounds must both be positive")
    workspace_path = _absolute(workspace, "workspace")
    evidence = _absolute(evidence_root, "evidence root")
    runs = _absolute(run_root, "run root")
    frozen_path = _absolute(frozen_winner_path, "frozen winner path")
    corpus_file = _absolute(corpus_path, "fixed corpus path")
    _ensure_dir(workspace_path, "workspace")
    _ensure_dir(evidence, "evidence root")
    _ensure_dir(runs, "run root")
    _floor_check(workspace_path, additional_bytes=autotune.MAX_RANGE_BYTES * 8, stage="sustained-followup-admission")
    generation_name = generation or None
    followup_path = evidence / _generated_name(FOLLOWUP_NAME, generation_name)
    winner_path = evidence / _generated_name(FOLLOWUP_WINNER_NAME, generation_name)
    partial_path = evidence / _generated_name(FOLLOWUP_PARTIAL_NAME, generation_name)
    if followup_path.exists() or winner_path.exists():
        if followup_path.exists() and winner_path.exists():
            return {"status": "ALREADY_COMPLETE", "followup": _read_json(followup_path, "followup"), "winner": _read_json(winner_path, "followup winner")}
        raise DeepSeekV4XetSustainedError("incomplete immutable sustained output set needs manual inspection")
    frozen, profile, corpus = _load_frozen_inputs(winner_path=frozen_path, corpus_path=corpus_file)
    children = evidence / ("sustained-children" + (f"-{generation_name}" if generation_name else ""))
    _ensure_dir(children, "sustained child root")
    bytes_per_round = sum(int(row["length"]) for row in corpus["ranges"])
    profile = dict(profile)
    profile["id"] = f"{profile['id']}_SUSTAINED_DYNAMIC_WORK_STEALING"
    profile["scheduler_shape"] = "dynamic_work_stealing"
    candidates = (
        ("python_http11_reuse", "python-direct"),
        ("curl_http2_reuse", "curl-http2"),
    )
    probes: list[dict[str, Any]] = []
    for implementation, label in candidates:
        probes.append(_candidate(
            key=f"probe-{label}-r{probe_rounds}", implementation=implementation, phase="independent_transport_probe",
            profile=profile, corpus=corpus, workspace=workspace_path, children_root=children,
            rounds=probe_rounds, partial_path=partial_path,
        ))
    ranked = _rank(probes)
    selected = ranked[0] if ranked else None
    confirmations: list[dict[str, Any]] = []
    if selected is not None:
        for ordinal in (1, 2):
            confirmations.append(_candidate(
                key=f"sustained-{selected['implementation']}-r{sustained_rounds}-pass{ordinal}",
                implementation=str(selected["implementation"]), phase="two_sustained_confirmations",
                profile=profile, corpus=corpus, workspace=workspace_path, children_root=children,
                rounds=sustained_rounds, partial_path=partial_path,
            ))
    promoted: dict[str, Any] | None = None
    ranked_confirmations = _rank(confirmations)
    if len(ranked_confirmations) == 2:
        rates = [float(row["trial"]["steady_state"]["sealed_and_evicted_bytes_per_second"]) for row in ranked_confirmations]
        if max(rates) / min(rates) <= 1.10:
            promoted = dict(selected)
    status = "COMPLETE_PROMOTED" if promoted is not None else "COMPLETE_NO_STABLE_PROMOTION"
    conclusion = "two_sustained_confirmations_passed" if promoted is not None else "no_candidate_completed_two_stable_sustained_confirmations"
    followup = seal({
        "schema": FOLLOWUP_SCHEMA,
        "status": status,
        "created_at": _utc_now(),
        "endpoint": "DEEPSEEK_V4_MAX_PUBLIC_STREAM_ACTIVE",
        "generation": generation_name,
        "base_frozen_winner": {"path": str(frozen_path), "seal_sha256": frozen["seal_sha256"]},
        "fixed_corpus": {"path": str(corpus_file), "seal_sha256": corpus["seal_sha256"], "bytes_per_round": bytes_per_round, "same_exact_ranges_and_hashes": True},
        "strategy": {
            "independent_controls": ["Python direct-presigned HTTP/1.1 keep-alive", "curl --http2 direct-presigned keep-alive"],
            "scheduler": "six-round balanced rotating work-stealing batches across at most eight bounded workers",
            "probe_rounds": probe_rounds,
            "sustained_rounds": sustained_rounds,
            "planned_sustained_bytes_per_confirmation": bytes_per_round * sustained_rounds,
            "promotion_requires_two_sustained_confirmations_within_10_percent": True,
            "safety": "stop on 15 GiB floor, swap growth/ceiling, or thermal warning; no source cache or retained test frames",
        },
        "probes": probes,
        "selected_after_probe": selected,
        "sustained_confirmations": confirmations,
        "selection": {"promoted": promoted, "conclusion": conclusion},
        "claim_boundary": {
            "is_full_condense_or_validated_gravity_pack": False,
            "is_10gbe_wan_claim": False,
            "meaning": "measured current public route ceiling using exact verified source ranges and bounded test-frame lifecycle",
        },
    })
    _atomic_json(followup_path, followup)
    winner = seal({
        "schema": FOLLOWUP_WINNER_SCHEMA,
        "status": "FROZEN" if promoted is not None else "NO_WINNER_PROMOTED",
        "created_at": _utc_now(),
        "followup_path": str(followup_path),
        "followup_seal_sha256": followup["seal_sha256"],
        "base_frozen_winner_seal_sha256": frozen["seal_sha256"],
        "profile": profile if promoted is not None else None,
        "implementation": promoted.get("implementation") if promoted is not None else None,
        "confirmation_records": confirmations if promoted is not None else [],
        "real_stream_application": {
            "outer_source_windows_maximum": len(corpus["ranges"]),
            "source_cache_bytes": 0,
            "xet_environment_set_before_import": True,
            "protected_floor_bytes": autotune.MIN_FREE_FLOOR_BYTES,
            "use_only_after_route_aware_gravity_processor_and_independent_successor_exist": True,
        },
    })
    _atomic_json(winner_path, winner)
    launcher = _write_launcher(
        runs / _generated_name(FOLLOWUP_LAUNCHER_NAME, generation_name),
        winner=winner_path if winner.get("status") == "FROZEN" else frozen_path,
        artifact=runs / "full-43-layer-stream.gravity",
        sustained=winner.get("status") == "FROZEN",
    )
    return {"status": status, "followup": followup, "winner": winner, "launcher": launcher}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--spec", type=Path)
    subcommands = parser.add_subparsers(dest="command")
    run = subcommands.add_parser("run", help="run independent transport probe and sustained confirmations")
    run.add_argument("--workspace", type=Path, default=autotune.REPO_ROOT / "workspace")
    run.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    run.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    run.add_argument("--winner", type=Path, default=DEFAULT_EVIDENCE_ROOT / autotune.WINNER_NAME)
    run.add_argument("--corpus", type=Path, default=DEFAULT_EVIDENCE_ROOT / autotune.CORPUS_NAME)
    run.add_argument("--probe-rounds", type=int, default=DEFAULT_PROBE_ROUNDS)
    run.add_argument("--sustained-rounds", type=int, default=DEFAULT_SUSTAINED_ROUNDS)
    run.add_argument("--generation", type=str, help="immutable retry-generation suffix after an aborted attempt")
    resume = subcommands.add_parser("resume-real", help="safely hand a frozen sustained winner to the real stream")
    resume.add_argument("--winner", type=Path, default=DEFAULT_EVIDENCE_ROOT / FOLLOWUP_WINNER_NAME)
    resume.add_argument("--artifact-dir", type=Path, required=True)
    resume.add_argument("--workspace", type=Path, default=autotune.REPO_ROOT / "workspace")
    freeze = subcommands.add_parser("freeze-successors", help="emit immutable sustained roofline and active-config receipts")
    freeze.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    freeze.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    freeze.add_argument("--winner", type=Path, required=True)
    freeze.add_argument("--followup", type=Path, required=True)
    freeze.add_argument("--launcher", type=Path, required=True)
    freeze.add_argument("--generation", type=str)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.child:
        if args.spec is None:
            raise SystemExit("--spec is required with --child")
        return _child_main(args)
    try:
        if args.command == "run":
            evidence = _absolute(args.evidence_root, "evidence root")
            _ensure_dir(evidence, "evidence root")
            lease = _acquire_lock(evidence)
            try:
                result = run_followup(
                    workspace=args.workspace, evidence_root=evidence, run_root=args.run_root,
                    frozen_winner_path=args.winner, corpus_path=args.corpus,
                    probe_rounds=args.probe_rounds, sustained_rounds=args.sustained_rounds,
                    generation=args.generation,
                )
            finally:
                _release_lock(lease)
        elif args.command == "resume-real":
            result = resume_real_stream(
                winner_path=args.winner, artifact_dir=args.artifact_dir, workspace=args.workspace
            )
        elif args.command == "freeze-successors":
            result = freeze_successor_artifacts(
                evidence_root=args.evidence_root,
                run_root=args.run_root,
                winner_path=args.winner,
                followup_path=args.followup,
                launcher_path=args.launcher,
                generation=args.generation,
            )
        else:
            raise DeepSeekV4XetSustainedError("a command is required")
    except (DeepSeekV4XetSustainedError, SealIntegrityError) as exc:
        sys.stderr.write(f"deepseek-v4-xet-sustained-error: {_safe_error(str(exc))}\n")
        return 2
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

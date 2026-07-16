#!/usr/bin/env python3.12
"""Strict sidecar contract for physical counter evidence.

This module does not collect counters and cannot authorize a run.  It closes the
boundary between a heavy, lease-gated probe and separately collected hardware
traces: every accepted number must be bound to the exact raw bundle, execution
window, device census, collector/normalizer programs, and immutable capture
files.  Callers cannot satisfy the contract with source-name strings alone.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import re
import stat
from typing import Any, Iterable


SCHEMA = "hawking.physical_counter_attestation.v1"
NORMALIZED_SCHEMA = "hawking.physical_counter_normalized.v1"
PHASE_MARKERS_SCHEMA = "hawking.physical_phase_markers.v1"
PHASE_INTERVAL_SCHEMA = "hawking.physical_phase_interval.v1"
PHASE_PAIR_SCHEMA = "hawking.physical_phase_pair.v1"
PHASE_INTERVAL_IDENTITY_SCHEMA = "hawking.physical_phase_interval_identity.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
DOMAIN_UNITS = {
    "energy": "joule",
    "gpu_time": "nanosecond",
    "physical_bytes": "byte",
    "occupancy": "percent",
    "bandwidth": "byte_per_second",
}
MAX_CAPTURE_BYTES = 512 * 1024 * 1024
MAX_NORMALIZED_BYTES = 64 * 1024 * 1024


def validate_phase_markers(
    value: Any,
    *,
    run_nonce: str | None = None,
    workload_started_at_unix_ns: int | None = None,
    workload_ended_at_unix_ns: int | None = None,
    workload_elapsed_continuous_ns: int | None = None,
) -> list[str]:
    """Validate self-hashed, dual-clock probe phase attribution.

    On macOS the probe converts ``mach_absolute_time`` into nanoseconds, the
    same monotonic epoch used by Python's runner clock.  The runner additionally
    binds the manifest to its unguessable nonce and encloses both clock
    intervals inside independently sampled execution authority.
    """
    if not isinstance(value, dict):
        return ["phase_markers must be an object"]
    expected = {
        "schema", "run_nonce", "clock_source",
        "probe_started_wall_unix_ns", "probe_ended_wall_unix_ns",
        "probe_started_continuous_ns", "probe_ended_continuous_ns",
        "intervals", "pairs",
        "phase_markers_sha256",
    }
    errors: list[str] = []
    if set(value) != expected:
        errors.append("phase_markers fields are incomplete or unexpected")
    if value.get("schema") != PHASE_MARKERS_SCHEMA:
        errors.append(f"phase_markers.schema must be {PHASE_MARKERS_SCHEMA}")
    nonce = value.get("run_nonce")
    if not isinstance(nonce, str) or not HEX64.fullmatch(nonce):
        errors.append("phase_markers.run_nonce must be a 256-bit lowercase hex nonce")
    if run_nonce is not None and nonce != run_nonce:
        errors.append("phase_markers are not bound to execution authority run_nonce")
    if value.get("clock_source") != "mach_absolute_time_plus_system_time_unix_epoch":
        errors.append("phase_markers clock source is invalid")
    wall_start = value.get("probe_started_wall_unix_ns")
    wall_end = value.get("probe_ended_wall_unix_ns")
    continuous_start = value.get("probe_started_continuous_ns")
    continuous_end = value.get("probe_ended_continuous_ns")
    if (
        isinstance(wall_start, bool) or not isinstance(wall_start, int) or wall_start <= 0
        or isinstance(wall_end, bool) or not isinstance(wall_end, int) or wall_end <= wall_start
        or isinstance(continuous_start, bool) or not isinstance(continuous_start, int)
        or continuous_start <= 0
        or isinstance(continuous_end, bool) or not isinstance(continuous_end, int)
        or continuous_end <= continuous_start
    ):
        errors.append("phase_markers probe interval is invalid")
    else:
        if abs((wall_end - wall_start) - (continuous_end - continuous_start)) > 1_000_000_000:
            errors.append("phase_markers wall/monotonic probe durations diverge")
        if (
            workload_started_at_unix_ns is not None
            and workload_ended_at_unix_ns is not None
            and not (
                workload_started_at_unix_ns <= wall_start < wall_end
                <= workload_ended_at_unix_ns
            )
        ):
            errors.append("phase_markers probe interval is outside execution authority")
        if (
            workload_elapsed_continuous_ns is not None
            and continuous_end - continuous_start
            > workload_elapsed_continuous_ns + 1_000_000_000
        ):
            errors.append("phase_markers monotonic duration exceeds execution authority")

    legacy_interval_fields = {
        "schema", "run_nonce", "sequence", "phase", "role", "batch",
        "iteration", "wall_started_unix_ns", "wall_ended_unix_ns",
        "continuous_started_ns", "continuous_ended_ns", "elapsed_ns",
        "interval_sha256",
    }
    interval_fields = legacy_interval_fields | {"interval_id", "signpost_id"}
    intervals = value.get("intervals")
    if not isinstance(intervals, list) or not intervals:
        errors.append("phase_markers.intervals must be non-empty")
        intervals = []
    by_hash: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(intervals):
        label = f"phase_markers.intervals[{index}]"
        if not isinstance(row, dict) or frozenset(row) not in {
            frozenset(legacy_interval_fields), frozenset(interval_fields),
        }:
            errors.append(f"{label} is incomplete or unexpected")
            continue
        if row.get("schema") != PHASE_INTERVAL_SCHEMA:
            errors.append(f"{label}.schema is invalid")
        if row.get("run_nonce") != nonce:
            errors.append(f"{label} is not bound to the root nonce")
        if row.get("sequence") != index:
            errors.append(f"{label}.sequence is not contiguous")
        if row.get("phase") not in {"parity", "warmup", "trial"}:
            errors.append(f"{label}.phase is invalid")
        if not isinstance(row.get("role"), str) or not row["role"]:
            errors.append(f"{label}.role is invalid")
        batch = row.get("batch")
        if batch is not None and (
            isinstance(batch, bool) or not isinstance(batch, int) or not 1 <= batch <= 8
        ):
            errors.append(f"{label}.batch is invalid")
        iteration = row.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            errors.append(f"{label}.iteration is invalid")
        if "interval_id" in row:
            identity = {
                "schema": PHASE_INTERVAL_IDENTITY_SCHEMA,
                "run_nonce": row.get("run_nonce"),
                "sequence": row.get("sequence"),
                "phase": row.get("phase"),
                "role": row.get("role"),
                "batch": row.get("batch"),
                "iteration": row.get("iteration"),
            }
            if row.get("interval_id") != canonical_sha256(identity):
                errors.append(f"{label}.interval_id does not match its predeclared identity")
            expected_signpost_id = int(row["interval_id"][:16], 16)
            if expected_signpost_id in {0, (1 << 64) - 1}:
                expected_signpost_id = 1
            if row.get("signpost_id") != expected_signpost_id:
                errors.append(f"{label}.signpost_id does not match its stable interval ID")
        mono_start = row.get("continuous_started_ns")
        mono_end = row.get("continuous_ended_ns")
        row_elapsed = row.get("elapsed_ns")
        row_wall_start = row.get("wall_started_unix_ns")
        row_wall_end = row.get("wall_ended_unix_ns")
        timing_valid = (
            isinstance(mono_start, int) and not isinstance(mono_start, bool) and mono_start >= 0
            and isinstance(mono_end, int) and not isinstance(mono_end, bool) and mono_end > mono_start
            and isinstance(row_elapsed, int) and not isinstance(row_elapsed, bool)
            and row_elapsed == mono_end - mono_start
            and isinstance(row_wall_start, int) and not isinstance(row_wall_start, bool)
            and isinstance(row_wall_end, int) and not isinstance(row_wall_end, bool)
            and row_wall_end > row_wall_start
        )
        if not timing_valid:
            errors.append(f"{label} timing is invalid")
        elif (
            isinstance(wall_start, int) and isinstance(wall_end, int)
            and isinstance(continuous_start, int) and isinstance(continuous_end, int)
        ):
            if not (
                wall_start <= row_wall_start < row_wall_end <= wall_end
                and continuous_start <= mono_start < mono_end <= continuous_end
            ):
                errors.append(f"{label} is outside the probe interval")
            if abs((row_wall_end - row_wall_start) - row_elapsed) > 1_000_000_000:
                errors.append(f"{label} wall/monotonic durations diverge")
        unstamped = copy.deepcopy(row)
        claimed = unstamped.pop("interval_sha256", None)
        if not isinstance(claimed, str) or not HEX64.fullmatch(claimed):
            errors.append(f"{label}.interval_sha256 is invalid")
        elif claimed != canonical_sha256(unstamped):
            errors.append(f"{label}.interval_sha256 mismatch")
        elif claimed in by_hash:
            errors.append(f"{label}.interval_sha256 is reused")
        else:
            by_hash[claimed] = row

    pair_common_fields = {
        "schema", "run_nonce", "phase", "batch", "iteration", "first_role",
        "baseline_interval_sha256", "phase_marker_sha256",
    }
    pairs = value.get("pairs")
    if not isinstance(pairs, list):
        errors.append("phase_markers.pairs must be a list")
        pairs = []
    pair_hashes: set[str] = set()
    consumed_intervals: set[str] = set()
    for index, row in enumerate(pairs):
        label = f"phase_markers.pairs[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{label} is incomplete or unexpected")
            continue
        role_fields = set(row) - pair_common_fields
        if role_fields in (
            {"candidate_interval_sha256"},
            {"candidate_interval_sha256", "baseline_interval_id", "candidate_interval_id"},
        ):
            comparison_role = "candidate"
            comparison_field = "candidate_interval_sha256"
            comparison_id_field = "candidate_interval_id"
        elif role_fields in (
            {"verifier_interval_sha256"},
            {"verifier_interval_sha256", "baseline_interval_id", "verifier_interval_id"},
        ):
            comparison_role = "verifier"
            comparison_field = "verifier_interval_sha256"
            comparison_id_field = "verifier_interval_id"
        else:
            errors.append(f"{label} must bind exactly one candidate/verifier interval")
            continue
        if row.get("schema") != PHASE_PAIR_SCHEMA or row.get("run_nonce") != nonce:
            errors.append(f"{label} schema/nonce binding is invalid")
        if row.get("phase") not in {"parity", "warmup", "trial"}:
            errors.append(f"{label}.phase is invalid")
        if row.get("first_role") not in {"baseline", comparison_role}:
            errors.append(f"{label}.first_role is invalid")
        baseline_hash = row.get("baseline_interval_sha256")
        comparison_hash = row.get(comparison_field)
        baseline = by_hash.get(baseline_hash)
        comparison = by_hash.get(comparison_hash)
        if baseline is None or comparison is None or baseline_hash == comparison_hash:
            errors.append(f"{label} interval references are invalid")
        else:
            if baseline_hash in consumed_intervals or comparison_hash in consumed_intervals:
                errors.append(f"{label} reuses an interval from another pair")
            consumed_intervals.update((baseline_hash, comparison_hash))
            expected_identity = (row.get("phase"), row.get("batch"), row.get("iteration"))
            if (
                (baseline.get("phase"), baseline.get("batch"), baseline.get("iteration"))
                != expected_identity
                or (comparison.get("phase"), comparison.get("batch"), comparison.get("iteration"))
                != expected_identity
                or baseline.get("role") != "baseline"
                or comparison.get("role") != comparison_role
            ):
                errors.append(f"{label} interval identities do not match the pair")
            if "baseline_interval_id" in row and (
                row.get("baseline_interval_id") != baseline.get("interval_id")
                or row.get(comparison_id_field) != comparison.get("interval_id")
            ):
                errors.append(f"{label} stable interval IDs do not match referenced intervals")
            first = baseline if baseline.get("sequence", 0) < comparison.get("sequence", 0) else comparison
            if first.get("role") != row.get("first_role"):
                errors.append(f"{label}.first_role disagrees with interval order")
        unstamped = copy.deepcopy(row)
        claimed = unstamped.pop("phase_marker_sha256", None)
        if not isinstance(claimed, str) or not HEX64.fullmatch(claimed):
            errors.append(f"{label}.phase_marker_sha256 is invalid")
        elif claimed != canonical_sha256(unstamped):
            errors.append(f"{label}.phase_marker_sha256 mismatch")
        elif claimed in pair_hashes:
            errors.append(f"{label}.phase_marker_sha256 is reused")
        else:
            pair_hashes.add(claimed)

    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("phase_markers_sha256", None)
    if not isinstance(claimed, str) or not HEX64.fullmatch(claimed):
        errors.append("phase_markers.phase_markers_sha256 is invalid")
    elif claimed != canonical_sha256(unstamped):
        errors.append("phase_markers.phase_markers_sha256 mismatch")
    return errors


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def stamp(attestation: dict[str, Any]) -> dict[str, Any]:
    stamped = copy.deepcopy(attestation)
    stamped.pop("attestation_sha256", None)
    stamped["attestation_sha256"] = canonical_sha256(stamped)
    return stamped


def file_identity(path: pathlib.Path) -> dict[str, Any]:
    """Return a verified regular-file identity without following a final symlink."""
    resolved = path.absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"counter artifact is not a single-link regular file: {resolved}")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns,
            row.st_nlink,
        )
        if identity(before) != identity(after) or size != after.st_size:
            raise ValueError(f"counter artifact changed while hashing: {resolved}")
        return {"path": str(resolved), "sha256": digest.hexdigest(), "size_bytes": size}
    finally:
        os.close(descriptor)


def _artifact_errors(
    row: Any,
    *,
    label: str,
    verify_files: bool,
    maximum_bytes: int = MAX_CAPTURE_BYTES,
) -> list[str]:
    if not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes"}:
        return [f"{label} must contain exactly path/sha256/size_bytes"]
    path_text = row.get("path")
    if not isinstance(path_text, str) or not path_text or not pathlib.Path(path_text).is_absolute():
        return [f"{label}.path must be absolute"]
    if not isinstance(row.get("sha256"), str) or not HEX64.fullmatch(row["sha256"]):
        return [f"{label}.sha256 is invalid"]
    size = row.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0 or size > maximum_bytes:
        return [f"{label}.size_bytes is invalid or exceeds its bounded limit"]
    if not verify_files:
        return []
    try:
        observed = file_identity(pathlib.Path(path_text))
    except (OSError, ValueError) as exc:
        return [f"{label} cannot be verified: {exc}"]
    return [] if observed == row else [f"{label} differs from the immutable file"]


def validate_execution_authority(
    value: Any, *, raw_probe_sha256: str, verify_files: bool = True,
) -> list[str]:
    if not isinstance(value, dict):
        return ["execution_authority is missing"]
    expected = {
        "probe_binary", "argv_sha256", "run_nonce", "started_at_unix_ns",
        "ended_at_unix_ns", "started_at_continuous_ns", "ended_at_continuous_ns",
        "exit_code", "raw_probe_sha256", "stdout_sha256", "stderr_sha256",
    }
    errors: list[str] = []
    if set(value) != expected:
        errors.append("execution_authority fields are incomplete or unexpected")
    errors.extend(_artifact_errors(
        value.get("probe_binary"), label="execution_authority.probe_binary",
        verify_files=verify_files,
    ))
    for field in ("argv_sha256", "stdout_sha256", "stderr_sha256"):
        if not isinstance(value.get(field), str) or not HEX64.fullmatch(value[field]):
            errors.append(f"execution_authority.{field} is invalid")
    nonce = value.get("run_nonce")
    if not isinstance(nonce, str) or not HEX64.fullmatch(nonce):
        errors.append("execution_authority.run_nonce must be a 256-bit lowercase hex nonce")
    started, ended = value.get("started_at_unix_ns"), value.get("ended_at_unix_ns")
    continuous_started = value.get("started_at_continuous_ns")
    continuous_ended = value.get("ended_at_continuous_ns")
    if (
        isinstance(started, bool) or not isinstance(started, int) or started <= 0
        or isinstance(ended, bool) or not isinstance(ended, int) or ended <= started
    ):
        errors.append("execution_authority workload interval is invalid")
    if (
        isinstance(continuous_started, bool) or not isinstance(continuous_started, int)
        or continuous_started <= 0 or isinstance(continuous_ended, bool)
        or not isinstance(continuous_ended, int) or continuous_ended <= continuous_started
    ):
        errors.append("execution_authority continuous-clock interval is invalid")
    elif isinstance(started, int) and isinstance(ended, int) and abs(
        (ended - started) - (continuous_ended - continuous_started)
    ) > 1_000_000_000:
        errors.append("execution_authority wall/continuous durations diverge")
    if value.get("exit_code") != 0:
        errors.append("execution_authority must bind a successful probe")
    if value.get("raw_probe_sha256") != raw_probe_sha256:
        errors.append("execution_authority is not bound to the raw probe")
    return errors


def _load_normalized(path: pathlib.Path) -> Any:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.absolute(), flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_size > MAX_NORMALIZED_BYTES
        ):
            raise ValueError("normalized counter output is unsafe or exceeds its bounded limit")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, MAX_NORMALIZED_BYTES + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > MAX_NORMALIZED_BYTES:
                raise ValueError("normalized counter output exceeds its bounded limit")
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns,
            row.st_nlink,
        )
        if identity(before) != identity(after) or observed != after.st_size:
            raise ValueError("normalized counter output changed while reading")
        return json.loads(b"".join(chunks).decode("utf-8"))
    finally:
        os.close(descriptor)


def validate(
    attestation: Any,
    *,
    raw_bundle_sha256: str,
    artifact_sha256: str,
    execution_authority: dict[str, Any],
    counter_payload: dict[str, Any],
    required_domains: Iterable[str],
    minimum_samples: int,
    verify_files: bool = True,
) -> list[str]:
    """Validate an immutable physical-counter attestation and its sidecars."""
    if not isinstance(attestation, dict):
        return ["counter_attestation must be an object"]
    errors: list[str] = []
    required = tuple(required_domains)
    if not required or len(set(required)) != len(required) or any(
        domain not in DOMAIN_UNITS for domain in required
    ):
        return ["validator required_domains are invalid"]
    if isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int) or minimum_samples <= 0:
        return ["validator minimum_samples must be positive"]
    expected_fields = {
        "schema", "raw_bundle_sha256", "artifact_sha256",
        "execution_authority_sha256", "counter_payload_sha256", "capture_window",
        "device", "normalizer", "domains", "attestation_sha256",
    }
    if set(attestation) != expected_fields:
        errors.append("counter attestation fields are incomplete or unexpected")
    if attestation.get("schema") != SCHEMA:
        errors.append(f"counter attestation schema must be {SCHEMA}")
    if attestation.get("raw_bundle_sha256") != raw_bundle_sha256:
        errors.append("counter attestation is not bound to the raw bundle")
    if attestation.get("artifact_sha256") != artifact_sha256:
        errors.append("counter attestation is not bound to the target artifact")
    if attestation.get("execution_authority_sha256") != canonical_sha256(execution_authority):
        errors.append("counter attestation is not bound to execution authority")
    if attestation.get("counter_payload_sha256") != canonical_sha256(counter_payload):
        errors.append("counter attestation is not bound to normalized counter values")

    unstamped = copy.deepcopy(attestation)
    claimed = unstamped.pop("attestation_sha256", None)
    if claimed != canonical_sha256(unstamped):
        errors.append("counter attestation self-hash mismatch")

    start = execution_authority.get("started_at_unix_ns")
    end = execution_authority.get("ended_at_unix_ns")
    window = attestation.get("capture_window")
    expected_window_fields = {
        "capture_started_at_unix_ns", "workload_started_at_unix_ns",
        "workload_ended_at_unix_ns", "capture_ended_at_unix_ns",
        "capture_started_at_continuous_ns", "workload_started_at_continuous_ns",
        "workload_ended_at_continuous_ns", "capture_ended_at_continuous_ns",
        "clock_source",
    }
    if not isinstance(window, dict) or set(window) != expected_window_fields:
        errors.append("counter capture_window is incomplete or unexpected")
    else:
        capture_start = window.get("capture_started_at_unix_ns")
        capture_end = window.get("capture_ended_at_unix_ns")
        continuous_capture_start = window.get("capture_started_at_continuous_ns")
        continuous_capture_end = window.get("capture_ended_at_continuous_ns")
        continuous_start = execution_authority.get("started_at_continuous_ns")
        continuous_end = execution_authority.get("ended_at_continuous_ns")
        if (
            window.get("workload_started_at_unix_ns") != start
            or window.get("workload_ended_at_unix_ns") != end
            or isinstance(capture_start, bool) or not isinstance(capture_start, int)
            or isinstance(capture_end, bool) or not isinstance(capture_end, int)
            or not isinstance(start, int) or not isinstance(end, int)
            or capture_start > start or capture_end < end or capture_start >= capture_end
        ):
            errors.append("counter capture window does not cover the exact workload interval")
        if (
            window.get("workload_started_at_continuous_ns") != continuous_start
            or window.get("workload_ended_at_continuous_ns") != continuous_end
            or isinstance(continuous_capture_start, bool)
            or not isinstance(continuous_capture_start, int)
            or isinstance(continuous_capture_end, bool)
            or not isinstance(continuous_capture_end, int)
            or not isinstance(continuous_start, int) or not isinstance(continuous_end, int)
            or continuous_capture_start > continuous_start
            or continuous_capture_end < continuous_end
            or continuous_capture_start >= continuous_capture_end
        ):
            errors.append("counter continuous-clock window does not cover the exact workload interval")
        if window.get("clock_source") != "clock_gettime_wall_continuous_crosscheck":
            errors.append("counter capture window lacks the required wall/continuous clock crosscheck")

    device = attestation.get("device")
    if not isinstance(device, dict) or set(device) != {
        "registry_id", "name", "architecture", "os_build", "driver_build",
        "hardware_uuid_sha256", "probe_receipt",
    }:
        errors.append("counter device binding is incomplete or unexpected")
    else:
        for field in ("registry_id", "name", "architecture", "os_build", "driver_build"):
            if not isinstance(device.get(field), str) or not device[field].strip():
                errors.append(f"counter device.{field} is empty")
        if not isinstance(device.get("hardware_uuid_sha256"), str) or not HEX64.fullmatch(
            device["hardware_uuid_sha256"]
        ):
            errors.append("counter device.hardware_uuid_sha256 is invalid")
        errors.extend(_artifact_errors(
            device.get("probe_receipt"), label="counter device.probe_receipt",
            verify_files=verify_files,
        ))

    normalizer = attestation.get("normalizer")
    if not isinstance(normalizer, dict) or set(normalizer) != {
        "program", "invocation_sha256", "source_commit", "output",
    }:
        errors.append("counter normalizer binding is incomplete or unexpected")
    else:
        errors.extend(_artifact_errors(
            normalizer.get("program"), label="counter normalizer.program",
            verify_files=verify_files,
        ))
        errors.extend(_artifact_errors(
            normalizer.get("output"), label="counter normalizer.output",
            verify_files=verify_files, maximum_bytes=MAX_NORMALIZED_BYTES,
        ))
        if not isinstance(normalizer.get("invocation_sha256"), str) or not HEX64.fullmatch(
            normalizer["invocation_sha256"]
        ):
            errors.append("counter normalizer.invocation_sha256 is invalid")
        if not isinstance(normalizer.get("source_commit"), str) or not COMMIT.fullmatch(
            normalizer["source_commit"]
        ):
            errors.append("counter normalizer.source_commit is invalid")

    rows = attestation.get("domains")
    by_domain: dict[str, dict[str, Any]] = {}
    raw_capture_hashes: dict[str, str] = {}
    domain_fields = {
        "domain", "unit", "source_kind", "collector", "collector_invocation_sha256",
        "raw_capture", "capture_started_at_unix_ns", "capture_ended_at_unix_ns",
        "capture_started_at_continuous_ns", "capture_ended_at_continuous_ns",
        "sample_count", "estimated",
    }
    if not isinstance(rows, list) or len(rows) != len(required):
        errors.append("counter domains do not have exact required coverage")
        rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != domain_fields:
            errors.append(f"counter domains[{index}] is incomplete or unexpected")
            continue
        domain = row.get("domain")
        if domain not in required or domain in by_domain:
            errors.append(f"counter domains[{index}] is unknown or duplicated")
            continue
        by_domain[domain] = row
        if row.get("unit") != DOMAIN_UNITS[domain]:
            errors.append(f"counter domain {domain} has the wrong unit")
        if not isinstance(row.get("source_kind"), str) or not row["source_kind"].strip():
            errors.append(f"counter domain {domain} source_kind is empty")
        if row.get("estimated") is not False:
            errors.append(f"counter domain {domain} must be directly measured, not estimated")
        if (
            isinstance(row.get("sample_count"), bool)
            or not isinstance(row.get("sample_count"), int)
            or row["sample_count"] < minimum_samples
        ):
            errors.append(f"counter domain {domain} has insufficient sample coverage")
        if not isinstance(row.get("collector_invocation_sha256"), str) or not HEX64.fullmatch(
            row["collector_invocation_sha256"]
        ):
            errors.append(f"counter domain {domain} collector invocation is invalid")
        errors.extend(_artifact_errors(
            row.get("collector"), label=f"counter domain {domain}.collector",
            verify_files=verify_files,
        ))
        errors.extend(_artifact_errors(
            row.get("raw_capture"), label=f"counter domain {domain}.raw_capture",
            verify_files=verify_files,
        ))
        capture_start = row.get("capture_started_at_unix_ns")
        capture_end = row.get("capture_ended_at_unix_ns")
        continuous_capture_start = row.get("capture_started_at_continuous_ns")
        continuous_capture_end = row.get("capture_ended_at_continuous_ns")
        continuous_start = execution_authority.get("started_at_continuous_ns")
        continuous_end = execution_authority.get("ended_at_continuous_ns")
        if (
            isinstance(capture_start, bool) or not isinstance(capture_start, int)
            or isinstance(capture_end, bool) or not isinstance(capture_end, int)
            or not isinstance(start, int) or not isinstance(end, int)
            or capture_start > start or capture_end < end or capture_start >= capture_end
        ):
            errors.append(f"counter domain {domain} does not cover the workload interval")
        if (
            isinstance(continuous_capture_start, bool)
            or not isinstance(continuous_capture_start, int)
            or isinstance(continuous_capture_end, bool)
            or not isinstance(continuous_capture_end, int)
            or not isinstance(continuous_start, int) or not isinstance(continuous_end, int)
            or continuous_capture_start > continuous_start
            or continuous_capture_end < continuous_end
            or continuous_capture_start >= continuous_capture_end
        ):
            errors.append(f"counter domain {domain} does not cover the continuous workload interval")
        capture = row.get("raw_capture")
        if isinstance(capture, dict) and isinstance(capture.get("sha256"), str):
            raw_capture_hashes[domain] = capture["sha256"]
    if set(by_domain) != set(required):
        errors.append("counter domains do not match the exact required set")

    if verify_files and isinstance(normalizer, dict) and isinstance(normalizer.get("output"), dict):
        try:
            normalized = _load_normalized(pathlib.Path(normalizer["output"]["path"]))
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"normalized counter output cannot be read: {exc}")
        else:
            expected_normalized = {
                "schema": NORMALIZED_SCHEMA,
                "raw_bundle_sha256": raw_bundle_sha256,
                "execution_authority_sha256": canonical_sha256(execution_authority),
                "counter_payload": counter_payload,
                "raw_capture_sha256s": raw_capture_hashes,
            }
            if normalized != expected_normalized:
                errors.append("normalized counter output differs from bound values/captures")
    return errors


def requirements(required_domains: Iterable[str], *, minimum_samples: int) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "required_domains": list(required_domains),
        "minimum_samples_per_domain": minimum_samples,
        "invariants": [
            "exact raw-bundle, artifact, execution-authority, and counter-value hashes",
            "capture window and every domain cover the exact workload interval",
            "immutable file-verified device, collector, raw-capture, normalizer, and normalized-output identities",
            "direct measurements only; estimated=false",
            "normalized values bind every raw capture hash",
        ],
    }

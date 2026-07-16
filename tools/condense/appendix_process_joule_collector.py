#!/usr/bin/env python3.12
"""Pinned libproc/RUSAGE_INFO_V6 direct-process energy counter contract.

Darwin's ``proc_pid_rusage(pid, RUSAGE_INFO_V6, ...)`` exposes cumulative
``ri_energy_nj`` and ``ri_penergy_nj`` counters for one process.  This module
defines and validates the exact ABI, shared-cache/library provenance, process
identity, monotone counter snapshots, and bracketing self-sampling protocol
needed before a delta may be described as direct process joules.

The CLI is status/self-test only.  It cannot launch a probe or claim physical
evidence.  Production collection additionally needs each release probe to take
one snapshot immediately before and one immediately after every exact measured
closure.  The measured wall/continuous interval excludes both counter reads;
the counter snapshots bracket that interval so instrumentation cannot improve
or degrade the performance comparison itself.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import json
import os
import pathlib
import platform
import re
import sys
import time
from typing import Any, Mapping

import appendix_contract
import physical_counter_attestation


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = "hawking.appendix_libproc_process_joule_collector.v1"
SNAPSHOT_SCHEMA = "hawking.libproc_rusage_v6_snapshot.v1"
PHASE_RECORD_SCHEMA = "hawking.libproc_process_joule_phase_record.v1"
BACKEND_ID = "darwin-libproc-proc_pid_rusage-v6-ri_energy_nj-v1"
PROBE_COUNTERS_SCHEMA = "hawking.probe_process_energy_counters.v1"
CAPTURE_SCHEMA = PROBE_COUNTERS_SCHEMA
BOUNDARY_PROTOCOL = "probe-self-sampled-bracketing-phase-interval-v2"
RUSAGE_INFO_V6 = 6
LIBPROC_INSTALL_NAME = "/usr/lib/libproc.dylib"
RESOURCE_HEADER = pathlib.Path(
    "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/sys/resource.h"
)
LIBPROC_HEADER = pathlib.Path(
    "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/libproc.h"
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UUID32 = re.compile(r"^[0-9a-f]{32}$")


RUSAGE_V6_U64_FIELDS = (
    "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
    "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
    "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
    "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
    "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
    "ri_diskio_byteswritten", "ri_cpu_time_qos_default",
    "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background",
    "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
    "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
    "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
    "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
    "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
    "ri_runnable_time", "ri_flags", "ri_user_ptime", "ri_system_ptime",
    "ri_pinstructions", "ri_pcycles", "ri_energy_nj", "ri_penergy_nj",
    "ri_secure_time_in_system", "ri_secure_ptime_in_system", "ri_neural_footprint",
    "ri_lifetime_max_neural_footprint", "ri_interval_max_neural_footprint",
    "ri_conclave_footprint", "ri_page_wait_time_mach", "ri_page_cache_hits",
)
MONOTONE_FIELDS = (
    "ri_user_time", "ri_system_time", "ri_instructions", "ri_cycles",
    "ri_pinstructions", "ri_pcycles", "ri_energy_nj", "ri_penergy_nj",
)


class RUsageInfoV6(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        *((name, ctypes.c_uint64) for name in RUSAGE_V6_U64_FIELDS),
        ("ri_reserved", ctypes.c_uint64 * 6),
    ]


def canonical_sha256(value: Any) -> str:
    return appendix_contract.canonical_sha256(value)


def _stamp(value: dict[str, Any], field: str) -> dict[str, Any]:
    stamped = copy.deepcopy(value)
    stamped.pop(field, None)
    stamped[field] = canonical_sha256(stamped)
    return stamped


def contract() -> dict[str, Any]:
    offsets = {
        name: getattr(RUsageInfoV6, name).offset
        for name, _ctype in RUsageInfoV6._fields_
    }
    return _stamp({
        "schema": SCHEMA,
        "backend_id": BACKEND_ID,
        "capture_schema": CAPTURE_SCHEMA,
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "phase_record_schema": PHASE_RECORD_SCHEMA,
        "library_install_name": LIBPROC_INSTALL_NAME,
        "function": "proc_pid_rusage",
        "flavor": RUSAGE_INFO_V6,
        "struct": "rusage_info_v6",
        "struct_size_bytes": ctypes.sizeof(RUsageInfoV6),
        "field_offsets": offsets,
        "energy_field": "ri_energy_nj",
        "secondary_energy_field": "ri_penergy_nj",
        "energy_input_unit": "nanojoule",
        "energy_output_unit": "joule",
        "monotone_fields": list(MONOTONE_FIELDS),
        "identity_fields": ["pid", "ri_uuid", "ri_proc_start_abstime"],
        "phase_boundary_protocol": BOUNDARY_PROTOCOL,
        "requirements": [
            "same PID+UUID+process-start-abstime before and after",
            "release probe self-samples immediately before and after each measured closure",
            "the before read ends no later than operation start and the after read starts no earlier than operation end",
            "the exact PhaseRecorder performance interval excludes both counter-read windows",
            "all cumulative counters are monotone with no wrap",
            "energy delta is positive and converted exactly from nJ to J",
            "SDK header hashes, struct layout, libproc version, shared-cache UUID, and OS build are receipt-bound",
        ],
        "powermetrics_energy_impact_accepted": False,
        "estimated_or_apportioned_values_accepted": False,
        "launch_or_probe_capability": False,
        "physical_evidence_claimed": False,
    }, "contract_sha256")


CONTRACT_SHA256 = contract()["contract_sha256"]


class ProcessJouleError(ValueError):
    """A direct-process counter or its provenance is invalid."""


def _load_libproc() -> ctypes.CDLL:
    try:
        library = ctypes.CDLL(LIBPROC_INSTALL_NAME, use_errno=True)
    except OSError as exc:
        raise ProcessJouleError(f"libproc cannot be loaded: {exc}") from exc
    library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    library.proc_pid_rusage.restype = ctypes.c_int
    library.proc_libversion.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    library.proc_libversion.restype = ctypes.c_int
    library._dyld_get_shared_cache_uuid.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    library._dyld_get_shared_cache_uuid.restype = ctypes.c_bool
    return library


def library_provenance(library: Any | None = None) -> dict[str, Any]:
    library = _load_libproc() if library is None else library
    major, minor = ctypes.c_int(), ctypes.c_int()
    if library.proc_libversion(ctypes.byref(major), ctypes.byref(minor)) != 0:
        raise ProcessJouleError("proc_libversion failed")
    cache_uuid = (ctypes.c_uint8 * 16)()
    if library._dyld_get_shared_cache_uuid(cache_uuid) is not True:
        raise ProcessJouleError("dyld shared-cache UUID is unavailable")
    try:
        resource_identity = physical_counter_attestation.file_identity(RESOURCE_HEADER)
        libproc_identity = physical_counter_attestation.file_identity(LIBPROC_HEADER)
    except (OSError, ValueError) as exc:
        raise ProcessJouleError(f"SDK ABI headers cannot be identified: {exc}") from exc
    return _stamp({
        "schema": "hawking.libproc_rusage_v6_library_provenance.v1",
        "library_install_name": LIBPROC_INSTALL_NAME,
        "proc_libversion_major": major.value,
        "proc_libversion_minor": minor.value,
        "dyld_shared_cache_uuid": bytes(cache_uuid).hex(),
        "os_build": platform.version(),
        "machine": platform.machine(),
        "resource_header": resource_identity,
        "libproc_header": libproc_identity,
        "collector_contract_sha256": CONTRACT_SHA256,
        "struct_layout_sha256": canonical_sha256(contract()["field_offsets"]),
    }, "provenance_sha256")


def read_snapshot(pid: int, *, library: Any | None = None) -> dict[str, Any]:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ProcessJouleError("PID must be positive")
    library = _load_libproc() if library is None else library
    value = RUsageInfoV6()
    started_u, started_c = time.time_ns(), time.monotonic_ns()
    result = library.proc_pid_rusage(pid, RUSAGE_INFO_V6, ctypes.byref(value))
    ended_c, ended_u = time.monotonic_ns(), time.time_ns()
    if result != 0:
        number = ctypes.get_errno()
        raise ProcessJouleError(
            f"proc_pid_rusage failed for PID {pid}: [{number}] {os.strerror(number or errno.EIO)}"
        )
    counters = {name: int(getattr(value, name)) for name in RUSAGE_V6_U64_FIELDS}
    return _stamp({
        "schema": SNAPSHOT_SCHEMA,
        "backend_id": BACKEND_ID,
        "pid": pid,
        "ri_uuid": bytes(value.ri_uuid).hex(),
        "read_started_at_unix_ns": started_u,
        "read_ended_at_unix_ns": ended_u,
        "read_started_at_continuous_ns": started_c,
        "read_ended_at_continuous_ns": ended_c,
        "counters": counters,
    }, "snapshot_sha256")


def snapshot_errors(value: Any, *, expected_pid: int | None = None) -> list[str]:
    expected = {
        "schema", "backend_id", "pid", "ri_uuid", "read_started_at_unix_ns",
        "read_ended_at_unix_ns", "read_started_at_continuous_ns",
        "read_ended_at_continuous_ns", "counters", "snapshot_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return ["libproc snapshot fields are incomplete or unexpected"]
    errors: list[str] = []
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("snapshot_sha256", None)
    if claimed != canonical_sha256(unstamped):
        errors.append("libproc snapshot self-hash mismatch")
    if value.get("schema") != SNAPSHOT_SCHEMA or value.get("backend_id") != BACKEND_ID:
        errors.append("libproc snapshot schema/backend is invalid")
    pid = value.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 \
            or expected_pid is not None and pid != expected_pid:
        errors.append("libproc snapshot PID is invalid or differs")
    if not isinstance(value.get("ri_uuid"), str) or UUID32.fullmatch(value["ri_uuid"]) is None:
        errors.append("libproc snapshot process UUID is invalid")
    for suffix in ("unix_ns", "continuous_ns"):
        start, end = value.get(f"read_started_at_{suffix}"), value.get(f"read_ended_at_{suffix}")
        if isinstance(start, bool) or not isinstance(start, int) \
                or isinstance(end, bool) or not isinstance(end, int) or end < start:
            errors.append(f"libproc snapshot {suffix} read window is invalid")
    counters = value.get("counters")
    if not isinstance(counters, dict) or set(counters) != set(RUSAGE_V6_U64_FIELDS) \
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0
                   for item in counters.values()):
        errors.append("libproc snapshot counters are incomplete or invalid")
    return errors


def phase_record(
    *, before: Mapping[str, Any], after: Mapping[str, Any],
    phase_marker_sha256: str, interval_sha256: str,
    interval_started_at_unix_ns: int, interval_ended_at_unix_ns: int,
    interval_started_at_continuous_ns: int, interval_ended_at_continuous_ns: int,
) -> dict[str, Any]:
    errors = [*snapshot_errors(before), *snapshot_errors(after)]
    if errors:
        raise ProcessJouleError("; ".join(errors))
    before_counters, after_counters = before["counters"], after["counters"]
    if before["pid"] != after["pid"] or before["ri_uuid"] != after["ri_uuid"] \
            or before_counters["ri_proc_start_abstime"] != after_counters["ri_proc_start_abstime"]:
        raise ProcessJouleError("PID identity changed between direct energy snapshots")
    for field in MONOTONE_FIELDS:
        if after_counters[field] < before_counters[field]:
            raise ProcessJouleError(f"direct process counter wrapped or decreased: {field}")
    delta = after_counters["ri_energy_nj"] - before_counters["ri_energy_nj"]
    if delta <= 0:
        raise ProcessJouleError("direct process energy delta is not positive")
    if not (
        before["read_started_at_unix_ns"]
        <= before["read_ended_at_unix_ns"] <= interval_started_at_unix_ns
        < interval_ended_at_unix_ns <= after["read_started_at_unix_ns"]
        <= after["read_ended_at_unix_ns"]
    ) or not (
        before["read_started_at_continuous_ns"]
        <= before["read_ended_at_continuous_ns"]
        <= interval_started_at_continuous_ns < interval_ended_at_continuous_ns
        <= after["read_started_at_continuous_ns"]
        <= after["read_ended_at_continuous_ns"]
    ):
        raise ProcessJouleError(
            "self-sampled counter reads do not bracket the exact operation interval"
        )
    for token in (phase_marker_sha256, interval_sha256):
        if not isinstance(token, str) or HEX64.fullmatch(token) is None:
            raise ProcessJouleError("phase/interval hash is invalid")
    return _stamp({
        "schema": PHASE_RECORD_SCHEMA,
        "backend_id": BACKEND_ID,
        "boundary_protocol": BOUNDARY_PROTOCOL,
        "source_sample_id": canonical_sha256({
            "before": before["snapshot_sha256"], "after": after["snapshot_sha256"],
            "phase": phase_marker_sha256,
        }),
        "phase_marker_sha256": phase_marker_sha256,
        "interval_sha256": interval_sha256,
        "process_id": before["pid"],
        "process_uuid": before["ri_uuid"],
        "process_start_abstime": before_counters["ri_proc_start_abstime"],
        "interval_started_at_unix_ns": interval_started_at_unix_ns,
        "interval_ended_at_unix_ns": interval_ended_at_unix_ns,
        "interval_started_at_continuous_ns": interval_started_at_continuous_ns,
        "interval_ended_at_continuous_ns": interval_ended_at_continuous_ns,
        "self_sampled_by_release_probe": True,
        "before": dict(before),
        "after": dict(after),
        "energy_nj_delta": delta,
        "energy_j": delta / 1_000_000_000,
        "quantity": "energy",
        "unit": "joule",
        "measurement_scope": "exact-probe-process",
        "attribution": "direct-counter",
        "estimated": False,
        "apportioned": False,
    }, "phase_record_sha256")


def phase_record_errors(value: Any, *, expected_pid: int) -> list[str]:
    if not isinstance(value, dict) or value.get("schema") != PHASE_RECORD_SCHEMA:
        return ["libproc phase record schema is invalid"]
    try:
        rebuilt = phase_record(
            before=value["before"], after=value["after"],
            phase_marker_sha256=value["phase_marker_sha256"],
            interval_sha256=value["interval_sha256"],
            interval_started_at_unix_ns=value["interval_started_at_unix_ns"],
            interval_ended_at_unix_ns=value["interval_ended_at_unix_ns"],
            interval_started_at_continuous_ns=value["interval_started_at_continuous_ns"],
            interval_ended_at_continuous_ns=value["interval_ended_at_continuous_ns"],
        )
    except (KeyError, TypeError, ProcessJouleError) as exc:
        return [f"libproc phase record cannot be reconstructed: {exc}"]
    errors = [] if value == rebuilt else ["libproc phase record differs from direct counter reconstruction"]
    if value.get("process_id") != expected_pid:
        errors.append("libproc phase record PID differs")
    return errors


def build_probe_counter_block(
    *, records: list[dict[str, Any]], library: Mapping[str, Any], probe_pid: int,
) -> dict[str, Any]:
    """Build the exact block emitted by a self-sampling release probe."""
    if not records:
        raise ProcessJouleError("probe process-energy block has no interval records")
    errors: list[str] = []
    markers: set[str] = set()
    intervals: set[str] = set()
    sources: set[str] = set()
    for ordinal, record in enumerate(records):
        errors.extend(
            f"record {ordinal}: {error}"
            for error in phase_record_errors(record, expected_pid=probe_pid)
        )
        if isinstance(record, dict):
            marker = record.get("phase_marker_sha256")
            interval = record.get("interval_sha256")
            source = record.get("source_sample_id")
            if marker in markers or interval in intervals or source in sources:
                errors.append(f"record {ordinal}: phase/interval/source identity is reused")
            if isinstance(marker, str):
                markers.add(marker)
            if isinstance(interval, str):
                intervals.add(interval)
            if isinstance(source, str):
                sources.add(source)
    expected_library_fields = {
        "schema", "library_install_name", "proc_libversion_major",
        "proc_libversion_minor", "dyld_shared_cache_uuid", "os_build", "machine",
        "resource_header", "libproc_header", "collector_contract_sha256",
        "struct_layout_sha256", "provenance_sha256",
    }
    if not isinstance(library, Mapping) or set(library) != expected_library_fields:
        errors.append("libproc library provenance fields are incomplete or unexpected")
    else:
        unstamped = dict(library)
        claimed = unstamped.pop("provenance_sha256", None)
        if claimed != canonical_sha256(unstamped):
            errors.append("libproc library provenance hash mismatch")
        if library.get("library_install_name") != LIBPROC_INSTALL_NAME \
                or library.get("collector_contract_sha256") != CONTRACT_SHA256 \
                or library.get("struct_layout_sha256") != canonical_sha256(
                    contract()["field_offsets"],
                ):
            errors.append("libproc library provenance differs from the pinned ABI")
        if not isinstance(library.get("dyld_shared_cache_uuid"), str) \
                or UUID32.fullmatch(library["dyld_shared_cache_uuid"]) is None:
            errors.append("libproc dyld shared-cache UUID is invalid")
    if errors:
        raise ProcessJouleError("; ".join(errors))
    return _stamp({
        "schema": PROBE_COUNTERS_SCHEMA,
        "backend_id": BACKEND_ID,
        "collector_contract_sha256": CONTRACT_SHA256,
        "probe_pid": probe_pid,
        "library_provenance": dict(library),
        "records": copy.deepcopy(records),
    }, "counters_sha256")


def probe_counter_block_errors(value: Any, *, expected_pid: int) -> list[str]:
    expected = {
        "schema", "backend_id", "collector_contract_sha256", "probe_pid",
        "library_provenance", "records", "counters_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return ["probe process-energy counter block fields are incomplete or unexpected"]
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("counters_sha256", None)
    errors = [] if claimed == canonical_sha256(unstamped) \
        else ["probe process-energy counter block hash mismatch"]
    if value.get("schema") != PROBE_COUNTERS_SCHEMA or value.get("backend_id") != BACKEND_ID \
            or value.get("collector_contract_sha256") != CONTRACT_SHA256:
        errors.append("probe process-energy counter block schema/backend/contract is invalid")
    if value.get("probe_pid") != expected_pid:
        errors.append("probe process-energy counter block PID differs")
    try:
        rebuilt = build_probe_counter_block(
            records=value.get("records"), library=value.get("library_provenance"),
            probe_pid=expected_pid,
        )
    except (TypeError, ProcessJouleError) as exc:
        errors.append(f"probe process-energy counter block cannot be reconstructed: {exc}")
    else:
        if rebuilt != value:
            errors.append("probe process-energy counter block differs from reconstructed direct counters")
    return errors


def status() -> dict[str, Any]:
    blockers: list[str] = []
    try:
        provenance = library_provenance()
    except (OSError, ValueError, ProcessJouleError) as exc:
        provenance = None
        blockers.append(f"libproc provenance unavailable: {exc}")
    blockers.append(
        "probe-side bracketing proc_pid_rusage_v6 source is wired but lacks a fresh release-build/runtime receipt"
    )
    return _stamp({
        "schema": SCHEMA,
        "contract_sha256": CONTRACT_SHA256,
        "collector_binary": physical_counter_attestation.file_identity(pathlib.Path(__file__)),
        "library_provenance": provenance,
        "direct_process_nanojoule_backend_available": provenance is not None,
        "phase_exact_collection_ready": False,
        "default_off": True,
        "collection_started": False,
        "physical_evidence_claimed": False,
        "blockers": blockers,
    }, "status_sha256")


def _selftest() -> int:
    value = contract()
    assert value["contract_sha256"] == CONTRACT_SHA256
    assert value["energy_field"] == "ri_energy_nj"
    assert ctypes.sizeof(RUsageInfoV6) == 16 + (len(RUSAGE_V6_U64_FIELDS) + 6) * 8
    print("appendix_process_joule_collector.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true")
    group.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    print(json.dumps(status(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3.12
"""Trusted, default-off normalizer for directly attributed physical counters.

The production Appendix path must never relabel ``powermetrics`` energy-impact
scores, machine-wide power, or interval apportionment as process joules.  This
program therefore accepts only two explicit, self-hashed adapter exports: one
whose source records already are direct joule counters for the exact probe PID,
and one whose source records are direct Metal counters for that same PID and
device registry ID.  Unknown schemas/backends and estimated or apportioned
records fail closed.

This module does not collect counters and grants no execution authority.  Its
executable bytes, schema contract, invocation, inherited lease, raw captures,
probe PID/run nonce, phase markers, and device ID must all be independently
bound by the release executor and signed authority receipts.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import pathlib
import stat
import sys
from typing import Any, Mapping

import appendix_contract
import appendix_process_joule_collector as process_joule
import physical_counter_attestation
import ram_scheduler


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = "hawking.appendix_physical_counter_normalizer.v1"
ATTRIBUTED_SCHEMA = "hawking.physical_counter_attributed_samples.v2"
DIRECT_JOULE_CAPTURE_SCHEMA = process_joule.PROBE_COUNTERS_SCHEMA
METAL_CAPTURE_SCHEMA = "hawking.direct_metal_process_counter_capture.v1"
DIRECT_JOULE_BACKEND = process_joule.BACKEND_ID
METAL_BACKEND = "xctrace-metal-system-trace-direct-process-v1"
UNSUPPORTED_POWERMETRICS_BACKEND = "powermetrics-energy-impact-proxy"
HEX64 = __import__("re").compile(r"^[0-9a-f]{64}$")
MAX_INPUT_BYTES = 512 * 1024 * 1024


def canonical_sha256(value: Any) -> str:
    return appendix_contract.canonical_sha256(value)


def _stamp(value: dict[str, Any], field: str) -> dict[str, Any]:
    stamped = copy.deepcopy(value)
    stamped.pop(field, None)
    stamped[field] = canonical_sha256(stamped)
    return stamped


def contract() -> dict[str, Any]:
    """Return the immutable semantic contract; it is not admission authority."""
    return _stamp({
        "schema": SCHEMA,
        "output_schema": ATTRIBUTED_SCHEMA,
        "input_schemas": {
            "energy": DIRECT_JOULE_CAPTURE_SCHEMA,
            "metal": METAL_CAPTURE_SCHEMA,
        },
        "accepted_backends": {
            "energy": [DIRECT_JOULE_BACKEND],
            "metal": [METAL_BACKEND],
        },
        "rejected_energy_sources": [
            UNSUPPORTED_POWERMETRICS_BACKEND,
            "powermetrics-process-energy-impact-number",
            "machine-wide-energy-apportioned-by-time",
            "run-minus-baseline-estimate",
        ],
        "energy_semantics": {
            "quantity": "energy",
            "unit": "joule",
            "scope": "exact-probe-process",
            "attribution": "direct-counter",
            "estimated": False,
            "apportioned": False,
        },
        "direct_process_joule_collector": {
            "schema": process_joule.SCHEMA,
            "contract_sha256": process_joule.CONTRACT_SHA256,
            "probe_counter_schema": process_joule.PROBE_COUNTERS_SCHEMA,
            "phase_record_schema": process_joule.PHASE_RECORD_SCHEMA,
            "sampling": process_joule.BOUNDARY_PROTOCOL,
        },
        "metal_semantics": {
            "scope": "exact-probe-process+exact-metal-registry-id",
            "attribution": "direct-counter",
            "estimated": False,
            "apportioned": False,
        },
        "bindings": [
            "normalizer executable file identity",
            "inherited exclusive heavy-lease device+inode",
            "raw bundle and exact execution argv hash",
            "probe PID and 256-bit run nonce",
            "phase-marker and wall+continuous interval",
            "sealed raw-capture SHA-256",
            "exact Metal registry ID",
            "unique source-record IDs for every emitted domain",
        ],
        "unknown_backend_policy": "fail-closed",
        "legacy_v1_output_permitted": False,
        "collection_or_probe_permitted": False,
        "physical_evidence_claimed": False,
    }, "contract_sha256")


CONTRACT_SHA256 = contract()["contract_sha256"]


class NormalizerError(ValueError):
    """An input cannot prove direct, exact-process physical attribution."""


def _finite(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value)) and (value > 0 if positive else value >= 0)
    )


def _safe_json(path: pathlib.Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.absolute(), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
                or before.st_size <= 0 or before.st_size > MAX_INPUT_BYTES:
            raise NormalizerError(f"unsafe or empty normalizer input: {path}")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, MAX_INPUT_BYTES + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > MAX_INPUT_BYTES:
                raise NormalizerError(f"normalizer input exceeds bounded size: {path}")
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
            row.st_ctime_ns, row.st_nlink,
        )
        if identity(before) != identity(after) or observed != after.st_size:
            raise NormalizerError(f"normalizer input changed while reading: {path}")
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise NormalizerError(f"normalizer input is not canonical JSON: {path}: {exc}") from exc
    finally:
        os.close(descriptor)
    if not isinstance(value, dict):
        raise NormalizerError(f"normalizer input must be an object: {path}")
    return value


def _immutable_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path.absolute(), flags, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short normalizer output write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _capture_errors(
    value: Any, *, schema: str, backend: str, probe_pid: int, run_nonce: str,
    probe_argv_sha256: str, metal_registry_id: str | None,
) -> list[str]:
    if schema == DIRECT_JOULE_CAPTURE_SCHEMA:
        return process_joule.probe_counter_block_errors(value, expected_pid=probe_pid)
    expected = {
        "schema", "backend_id", "probe_pid", "run_nonce", "probe_argv_sha256",
        "metal_registry_id", "capture_started_at_unix_ns",
        "capture_ended_at_unix_ns", "capture_started_at_continuous_ns",
        "capture_ended_at_continuous_ns", "records", "capture_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return [f"{schema} capture fields are incomplete or unexpected"]
    errors: list[str] = []
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("capture_sha256", None)
    if claimed != canonical_sha256(unstamped):
        errors.append(f"{schema} capture self-hash mismatch")
    if value.get("schema") != schema or value.get("backend_id") != backend:
        errors.append(f"unsupported {schema} schema/backend")
    if value.get("probe_pid") != probe_pid or value.get("run_nonce") != run_nonce:
        errors.append(f"{schema} capture differs from exact probe PID/run nonce")
    if value.get("probe_argv_sha256") != probe_argv_sha256:
        errors.append(f"{schema} capture differs from exact probe argv")
    if schema == DIRECT_JOULE_CAPTURE_SCHEMA:
        if value.get("metal_registry_id") is not None:
            errors.append("direct-joule capture unexpectedly selects a Metal device")
    elif not isinstance(metal_registry_id, str) or not metal_registry_id \
            or value.get("metal_registry_id") != metal_registry_id:
        errors.append("Metal capture differs from exact registry ID")
    bounds = [
        value.get("capture_started_at_unix_ns"), value.get("capture_ended_at_unix_ns"),
        value.get("capture_started_at_continuous_ns"),
        value.get("capture_ended_at_continuous_ns"),
    ]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in bounds) \
            or not bounds[0] < bounds[1] or not bounds[2] < bounds[3]:
        errors.append(f"{schema} capture window is invalid")
    if not isinstance(value.get("records"), list) or not value["records"]:
        errors.append(f"{schema} capture has no records")
    return errors


def _record_errors(
    record: Any, *, source: str, target: Mapping[str, Any], ordinal: int,
    probe_pid: int, run_nonce: str,
) -> list[str]:
    if source == "energy":
        errors = process_joule.phase_record_errors(record, expected_pid=probe_pid)
        if not isinstance(record, dict):
            return errors
        interval = target["interval"]
        comparisons = {
            "phase_marker_sha256": target["marker"],
            "interval_sha256": target["interval_sha256"],
            "process_id": probe_pid,
            "interval_started_at_unix_ns": interval["wall_started_unix_ns"],
            "interval_ended_at_unix_ns": interval["wall_ended_unix_ns"],
            "interval_started_at_continuous_ns": interval["continuous_started_ns"],
            "interval_ended_at_continuous_ns": interval["continuous_ended_ns"],
        }
        if any(record.get(field) != expected for field, expected in comparisons.items()):
            errors.append(f"energy record {ordinal} is not an exact PID/phase/clock join")
        if record.get("measurement_scope") != "exact-probe-process" \
                or record.get("attribution") != "direct-counter" \
                or record.get("estimated") is not False \
                or record.get("apportioned") is not False \
                or record.get("quantity") != "energy" \
                or record.get("unit") != "joule" \
                or not _finite(record.get("energy_j"), positive=True):
            errors.append(f"energy record {ordinal} is not a positive direct joule quantity")
        return errors
    common = {
        "source_sample_id", "phase_marker_sha256", "interval_sha256", "process_id",
        "run_nonce", "interval_started_at_unix_ns", "interval_ended_at_unix_ns",
        "interval_started_at_continuous_ns", "interval_ended_at_continuous_ns",
        "measurement_scope", "attribution", "estimated", "apportioned",
    }
    expected = common | {
        "gpu_time_ns", "physical_bytes", "occupancy_percent",
        "bandwidth_bytes_per_second",
    }
    if not isinstance(record, dict) or set(record) != expected:
        return [f"{source} record {ordinal} fields are incomplete or unexpected"]
    errors: list[str] = []
    interval = target["interval"]
    comparisons = {
        "phase_marker_sha256": target["marker"],
        "interval_sha256": target["interval_sha256"],
        "process_id": probe_pid,
        "run_nonce": run_nonce,
        "interval_started_at_unix_ns": interval["wall_started_unix_ns"],
        "interval_ended_at_unix_ns": interval["wall_ended_unix_ns"],
        "interval_started_at_continuous_ns": interval["continuous_started_ns"],
        "interval_ended_at_continuous_ns": interval["continuous_ended_ns"],
    }
    if any(record.get(field) != expected_value for field, expected_value in comparisons.items()):
        errors.append(f"{source} record {ordinal} is not an exact PID/phase/clock join")
    if not isinstance(record.get("source_sample_id"), str) or not record["source_sample_id"]:
        errors.append(f"{source} record {ordinal} source ID is empty")
    required_scope = "exact-probe-process+exact-metal-registry-id"
    if record.get("measurement_scope") != required_scope \
            or record.get("attribution") != "direct-counter" \
            or record.get("estimated") is not False \
            or record.get("apportioned") is not False:
        errors.append(f"{source} record {ordinal} is estimated, apportioned, or not process-direct")
    for field in ("gpu_time_ns", "physical_bytes", "bandwidth_bytes_per_second"):
        if not _finite(record.get(field), positive=True):
            errors.append(f"Metal record {ordinal} {field} is invalid")
    occupancy = record.get("occupancy_percent")
    if not _finite(occupancy) or float(occupancy) > 100:
        errors.append(f"Metal record {ordinal} occupancy is invalid")
    return errors


def _lease_identity(fd: int) -> dict[str, Any]:
    if isinstance(fd, bool) or not isinstance(fd, int) or fd < 3:
        raise NormalizerError("one inherited heavy-lease descriptor is required")
    try:
        row = os.fstat(fd)
        # Re-acquiring on the inherited open-file description is non-mutating.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise NormalizerError(f"inherited heavy lease is invalid or unlocked: {exc}") from exc
    if not stat.S_ISREG(row.st_mode):
        raise NormalizerError("inherited heavy lease is not a regular lock file")
    return {"inherited": True, "device": row.st_dev, "inode": row.st_ino}


def normalize(
    *, kind: str, bundle_path: pathlib.Path, energy_path: pathlib.Path,
    metal_path: pathlib.Path, probe_pid: int, run_nonce: str,
    metal_registry_id: str, lease_fd: int,
) -> dict[str, Any]:
    """Validate exact adapter exports and construct attributed-samples v2."""
    if kind not in {"device", "spec"}:
        raise NormalizerError("kind must be device or spec")
    if isinstance(probe_pid, bool) or not isinstance(probe_pid, int) or probe_pid <= 0:
        raise NormalizerError("probe PID must be positive")
    if not isinstance(run_nonce, str) or HEX64.fullmatch(run_nonce) is None:
        raise NormalizerError("run nonce must be 256-bit lowercase hex")
    bundle = _safe_json(bundle_path)
    energy = _safe_json(energy_path)
    metal = _safe_json(metal_path)
    authority = bundle.get("execution_authority", {})
    raw = bundle.get("raw_probe", {})
    if authority.get("run_nonce") != run_nonce:
        raise NormalizerError("raw bundle differs from exact run nonce")
    argv_sha = authority.get("argv_sha256")
    if not isinstance(argv_sha, str) or HEX64.fullmatch(argv_sha) is None:
        raise NormalizerError("raw bundle lacks exact probe argv hash")
    # Imported lazily so the validator can import this module's constants
    # without creating an import cycle.
    import appendix_physical_counter_collector as collector

    targets, errors = collector._phase_targets(bundle, kind)
    if raw.get("process_energy_counters") != energy:
        errors.append("direct process-energy sidecar differs from the release-probe embedded counter block")
    errors.extend(_capture_errors(
        energy, schema=DIRECT_JOULE_CAPTURE_SCHEMA, backend=DIRECT_JOULE_BACKEND,
        probe_pid=probe_pid, run_nonce=run_nonce, probe_argv_sha256=argv_sha,
        metal_registry_id=None,
    ))
    errors.extend(_capture_errors(
        metal, schema=METAL_CAPTURE_SCHEMA, backend=METAL_BACKEND,
        probe_pid=probe_pid, run_nonce=run_nonce, probe_argv_sha256=argv_sha,
        metal_registry_id=metal_registry_id,
    ))
    for capture, label in ((energy, "energy"), (metal, "metal")):
        records = capture.get("records", []) if isinstance(capture, dict) else []
        if len(records) != len(targets):
            errors.append(f"{label} records do not cover every exact trial marker")
            continue
        identifiers: set[str] = set()
        for ordinal, (record, target) in enumerate(zip(records, targets)):
            errors.extend(_record_errors(
                record, source=label, target=target, ordinal=ordinal,
                probe_pid=probe_pid, run_nonce=run_nonce,
            ))
            if isinstance(record, dict) and isinstance(record.get("source_sample_id"), str):
                identifiers.add(record["source_sample_id"])
        if len(identifiers) != len(records):
            errors.append(f"{label} source sample IDs are reused")
    if errors:
        raise NormalizerError("; ".join(errors))

    lease = _lease_identity(lease_fd)
    energy_identity = physical_counter_attestation.file_identity(energy_path)
    metal_identity = physical_counter_attestation.file_identity(metal_path)
    binary_identity = physical_counter_attestation.file_identity(pathlib.Path(__file__))
    required = collector.DEVICE_DOMAINS if kind == "device" else collector.SPEC_DOMAINS
    collectors = []
    for collector_id, capture, identity, backend in (
        ("process_joule", energy, energy_identity, DIRECT_JOULE_BACKEND),
        ("xctrace", metal, metal_identity, METAL_BACKEND),
    ):
        collectors.append({
            "id": collector_id,
            "backend_id": backend,
            "raw_capture_sha256": identity["sha256"],
            "available": True,
            "privilege_verified": True,
            "process_attributed": True,
            "phase_attributed": True,
            "directly_measured": True,
            "estimated": False,
            "apportioned": False,
            "domains": [d for d in required if collector.DOMAIN_COLLECTOR[d] == collector_id],
            "capture_started_at_unix_ns": (
                authority["started_at_unix_ns"]
                if collector_id == "process_joule" else capture["capture_started_at_unix_ns"]
            ),
            "capture_ended_at_unix_ns": (
                authority["ended_at_unix_ns"]
                if collector_id == "process_joule" else capture["capture_ended_at_unix_ns"]
            ),
            "capture_started_at_continuous_ns": (
                authority["started_at_continuous_ns"]
                if collector_id == "process_joule" else capture["capture_started_at_continuous_ns"]
            ),
            "capture_ended_at_continuous_ns": (
                authority["ended_at_continuous_ns"]
                if collector_id == "process_joule" else capture["capture_ended_at_continuous_ns"]
            ),
        })
    samples = []
    for ordinal, target in enumerate(targets):
        energy_record = energy["records"][ordinal]
        metal_record = metal["records"][ordinal]
        sample = {
            "ordinal": ordinal,
            "batch": target["batch"],
            "repeat": target["iteration"] if kind == "spec" else None,
            "phase_marker_sha256": target["marker"],
            "interval_sha256": target["interval_sha256"],
            "interval_started_at_unix_ns": target["interval"]["wall_started_unix_ns"],
            "interval_ended_at_unix_ns": target["interval"]["wall_ended_unix_ns"],
            "interval_started_at_continuous_ns": target["interval"]["continuous_started_ns"],
            "interval_ended_at_continuous_ns": target["interval"]["continuous_ended_ns"],
            "run_nonce": run_nonce,
            "process_id": probe_pid,
            "energy_j": energy_record["energy_j"],
            "gpu_time_ns": metal_record["gpu_time_ns"],
            "physical_bytes": metal_record["physical_bytes"],
            "occupancy_percent": metal_record["occupancy_percent"] if kind == "device" else None,
            "bandwidth_bytes_per_second": (
                metal_record["bandwidth_bytes_per_second"] if kind == "device" else None
            ),
            "energy_provenance": {
                "backend_id": DIRECT_JOULE_BACKEND,
                "quantity": "energy",
                "unit": "joule",
                "scope": "exact-probe-process",
                "attribution": "direct-counter",
                "estimated": False,
                "apportioned": False,
                "source_process_id": probe_pid,
            },
            "source_sample_ids": {
                domain: [
                    energy_record["source_sample_id"]
                    if domain == "energy" else metal_record["source_sample_id"]
                ] for domain in required
            },
        }
        samples.append(sample)
    manifest = _stamp({
        "schema": ATTRIBUTED_SCHEMA,
        "kind": kind,
        "normalizer": {
            "schema": SCHEMA,
            "contract_sha256": CONTRACT_SHA256,
            "binary": binary_identity,
        },
        "lease": lease,
        "probe_pid": probe_pid,
        "probe_argv_sha256": argv_sha,
        "metal_registry_id": metal_registry_id,
        "raw_bundle_sha256": bundle["raw_bundle_sha256"],
        "artifact_sha256": raw["artifact"]["sha256"],
        "runtime_path": raw["runtime_path"],
        "run_nonce": run_nonce,
        "phase_markers_sha256": raw["phase_markers"]["phase_markers_sha256"],
        "collectors": collectors,
        "samples": samples,
    }, "manifest_sha256")
    return manifest


def status() -> dict[str, Any]:
    process_status = process_joule.status()
    return _stamp({
        "schema": SCHEMA,
        "contract_sha256": CONTRACT_SHA256,
        "binary": physical_counter_attestation.file_identity(pathlib.Path(__file__)),
        "default_off": True,
        "collection_or_probe_started": False,
        "physical_evidence_claimed": False,
        "direct_process_nanojoule_backend_available": process_status[
            "direct_process_nanojoule_backend_available"
        ],
        "production_process_joule_backend_admitted": False,
        "blockers": [
            "powermetrics process energy is an energy-impact proxy, not direct joules",
            *process_status["blockers"],
            "libproc collector/ABI and self-sampling release probe lack final operator-signed receipts",
        ],
    }, "status_sha256")


def _selftest() -> int:
    value = contract()
    assert value["contract_sha256"] == CONTRACT_SHA256
    assert value["energy_semantics"]["estimated"] is False
    assert UNSUPPORTED_POWERMETRICS_BACKEND in value["rejected_energy_sources"]
    assert status()["production_process_joule_backend_admitted"] is False
    print("appendix_physical_counter_normalizer.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true")
    group.add_argument("--selftest", action="store_true")
    group.add_argument("--kind", choices=("device", "spec"))
    parser.add_argument("--raw-bundle", type=pathlib.Path)
    parser.add_argument("--process-joule", type=pathlib.Path)
    parser.add_argument("--xctrace", type=pathlib.Path)
    parser.add_argument("--probe-pid", type=int)
    parser.add_argument("--run-nonce")
    parser.add_argument("--metal-registry-id")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.status:
        print(json.dumps(status(), indent=2, sort_keys=True))
        return 0
    if args.selftest:
        return _selftest()
    required = {
        "raw_bundle": args.raw_bundle, "process_joule": args.process_joule,
        "xctrace": args.xctrace, "probe_pid": args.probe_pid,
        "run_nonce": args.run_nonce, "metal_registry_id": args.metal_registry_id,
        "output": args.output,
    }
    if any(value is None for value in required.values()):
        parser.error("normalization requires all capture, PID, nonce, device, and output arguments")
    lease_raw = os.environ.get(ram_scheduler.HEAVY_LEASE_FD_ENV)
    try:
        lease_fd = int(lease_raw) if lease_raw is not None else -1
        value = normalize(
            kind=args.kind, bundle_path=args.raw_bundle, energy_path=args.process_joule,
            metal_path=args.xctrace, probe_pid=args.probe_pid,
            run_nonce=args.run_nonce, metal_registry_id=args.metal_registry_id,
            lease_fd=lease_fd,
        )
        _immutable_json(args.output, value)
    except (NormalizerError, OSError, ValueError) as exc:
        print(f"physical counter normalizer blocked: {exc}", file=sys.stderr)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

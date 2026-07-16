from __future__ import annotations

import json
import pathlib
import copy
from typing import Any, Iterable

import appendix_contract
import physical_counter_attestation


def _stamp(value: dict[str, Any], field: str) -> dict[str, Any]:
    stamped = copy.deepcopy(value)
    stamped.pop(field, None)
    stamped[field] = physical_counter_attestation.canonical_sha256(stamped)
    return stamped


def phase_markers(
    *,
    pairs: list[dict[str, Any]],
    singles: list[dict[str, Any]] | None = None,
    run_nonce: str = "2" * 64,
) -> dict[str, Any]:
    """Build deterministic dual-clock marker fixtures from pair identities."""
    intervals: list[dict[str, Any]] = []
    stamped_pairs: list[dict[str, Any]] = []

    def interval(phase: str, role: str, batch: int | None, iteration: int) -> str:
        sequence = len(intervals)
        continuous_start = 2_000_000_000 + sequence * 10_000_000
        wall_start = 11_000_000_000 + sequence * 10_000_000
        row = _stamp({
            "schema": physical_counter_attestation.PHASE_INTERVAL_SCHEMA,
            "run_nonce": run_nonce,
            "sequence": sequence,
            "phase": phase,
            "role": role,
            "batch": batch,
            "iteration": iteration,
            "wall_started_unix_ns": wall_start,
            "wall_ended_unix_ns": wall_start + 1_000_000,
            "continuous_started_ns": continuous_start,
            "continuous_ended_ns": continuous_start + 1_000_000,
            "elapsed_ns": 1_000_000,
        }, "interval_sha256")
        intervals.append(row)
        return row["interval_sha256"]

    for row in singles or []:
        row["interval_sha256"] = interval(
            row["phase"], row["role"], row.get("batch"), row["iteration"],
        )
    for pair in pairs:
        phase = pair["phase"]
        batch = pair.get("batch")
        iteration = pair["iteration"]
        comparison_role = pair["comparison_role"]
        first_role = pair["first_role"]
        if first_role == "baseline":
            baseline_hash = interval(phase, "baseline", batch, iteration)
            comparison_hash = interval(phase, comparison_role, batch, iteration)
        else:
            comparison_hash = interval(phase, comparison_role, batch, iteration)
            baseline_hash = interval(phase, "baseline", batch, iteration)
        stamped = _stamp({
            "schema": physical_counter_attestation.PHASE_PAIR_SCHEMA,
            "run_nonce": run_nonce,
            "phase": phase,
            "batch": batch,
            "iteration": iteration,
            "first_role": first_role,
            "baseline_interval_sha256": baseline_hash,
            f"{comparison_role}_interval_sha256": comparison_hash,
        }, "phase_marker_sha256")
        stamped_pairs.append(stamped)
        pair["phase_marker_sha256"] = stamped["phase_marker_sha256"]
    return _stamp({
        "schema": physical_counter_attestation.PHASE_MARKERS_SCHEMA,
        "run_nonce": run_nonce,
        "clock_source": "mach_absolute_time_plus_system_time_unix_epoch",
        "probe_started_wall_unix_ns": 10_500_000_000,
        "probe_ended_wall_unix_ns": 19_500_000_000,
        "probe_started_continuous_ns": 1_500_000_000,
        "probe_ended_continuous_ns": 10_500_000_000,
        "intervals": intervals,
        "pairs": stamped_pairs,
    }, "phase_markers_sha256")


def execution_authority(tmp_path: pathlib.Path, raw: dict[str, Any]) -> dict[str, Any]:
    probe = tmp_path / "probe-bin"
    probe.write_bytes(b"reviewed release probe")
    raw_sha = appendix_contract.canonical_sha256(raw)
    return {
        "probe_binary": physical_counter_attestation.file_identity(probe),
        "argv_sha256": "1" * 64,
        "run_nonce": "2" * 64,
        "started_at_unix_ns": 10_000_000_000,
        "ended_at_unix_ns": 20_000_000_000,
        "started_at_continuous_ns": 1_000_000_000,
        "ended_at_continuous_ns": 11_000_000_000,
        "exit_code": 0,
        "raw_probe_sha256": raw_sha,
        "stdout_sha256": "3" * 64,
        "stderr_sha256": "4" * 64,
    }


def attestation(
    tmp_path: pathlib.Path,
    *,
    bundle: dict[str, Any],
    counter_payload: dict[str, Any],
    domains: Iterable[str],
    sample_count: int,
) -> dict[str, Any]:
    domain_names = tuple(domains)
    collector = tmp_path / "counter-collector"
    collector.write_bytes(b"reviewed physical counter collector")
    normalizer_program = tmp_path / "counter-normalizer"
    normalizer_program.write_bytes(b"reviewed counter normalizer")
    device_probe = tmp_path / "device-probe.json"
    device_probe.write_text(
        json.dumps({"registry_id": "test:1", "driver": "test-driver"}),
        encoding="utf-8",
    )
    raw_capture_hashes: dict[str, str] = {}
    rows = []
    for index, domain in enumerate(domain_names):
        capture = tmp_path / f"{domain}.capture"
        capture.write_bytes(f"physical {domain} capture {index}".encode("utf-8"))
        capture_identity = physical_counter_attestation.file_identity(capture)
        raw_capture_hashes[domain] = capture_identity["sha256"]
        rows.append({
            "domain": domain,
            "unit": physical_counter_attestation.DOMAIN_UNITS[domain],
            "source_kind": f"reviewed-{domain}-collector",
            "collector": physical_counter_attestation.file_identity(collector),
            "collector_invocation_sha256": f"{index + 5:x}" * 64,
            "raw_capture": capture_identity,
            "capture_started_at_unix_ns": 9_000_000_000,
            "capture_ended_at_unix_ns": 21_000_000_000,
            "capture_started_at_continuous_ns": 0,
            "capture_ended_at_continuous_ns": 12_000_000_000,
            "sample_count": sample_count,
            "estimated": False,
        })
    authority = bundle["execution_authority"]
    normalized = {
        "schema": physical_counter_attestation.NORMALIZED_SCHEMA,
        "raw_bundle_sha256": bundle["raw_bundle_sha256"],
        "execution_authority_sha256": physical_counter_attestation.canonical_sha256(authority),
        "counter_payload": counter_payload,
        "raw_capture_sha256s": raw_capture_hashes,
    }
    normalized_path = tmp_path / "normalized-counters.json"
    normalized_path.write_text(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")), encoding="utf-8",
    )
    return physical_counter_attestation.stamp({
        "schema": physical_counter_attestation.SCHEMA,
        "raw_bundle_sha256": bundle["raw_bundle_sha256"],
        "artifact_sha256": bundle["raw_probe"]["artifact"]["sha256"],
        "execution_authority_sha256": physical_counter_attestation.canonical_sha256(authority),
        "counter_payload_sha256": physical_counter_attestation.canonical_sha256(counter_payload),
        "capture_window": {
            "capture_started_at_unix_ns": 9_000_000_000,
            "workload_started_at_unix_ns": authority["started_at_unix_ns"],
            "workload_ended_at_unix_ns": authority["ended_at_unix_ns"],
            "capture_ended_at_unix_ns": 21_000_000_000,
            "capture_started_at_continuous_ns": 0,
            "workload_started_at_continuous_ns": authority["started_at_continuous_ns"],
            "workload_ended_at_continuous_ns": authority["ended_at_continuous_ns"],
            "capture_ended_at_continuous_ns": 12_000_000_000,
            "clock_source": "clock_gettime_wall_continuous_crosscheck",
        },
        "device": {
            "registry_id": "test:1",
            "name": "Test Metal GPU",
            "architecture": "apple-test",
            "os_build": "TEST1",
            "driver_build": "test-driver-1",
            "hardware_uuid_sha256": "a" * 64,
            "probe_receipt": physical_counter_attestation.file_identity(device_probe),
        },
        "normalizer": {
            "program": physical_counter_attestation.file_identity(normalizer_program),
            "invocation_sha256": "b" * 64,
            "source_commit": "c" * 40,
            "output": physical_counter_attestation.file_identity(normalized_path),
        },
        "domains": rows,
    })

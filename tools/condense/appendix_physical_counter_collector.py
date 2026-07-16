#!/usr/bin/env python3.12
"""Default-off Appendix physical-counter collection and v2 normalization contract.

This module deliberately has no collection command.  It describes later
probe-local libproc energy sampling plus the external xctrace launch, reports prerequisites,
and normalizes already-attributed samples.  A sample is rejected unless every
source is available, privilege-verified, process-attributed, phase-attributed,
and covers the exact probe wall and continuous-clock interval.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
from typing import Any

import appendix_contract
import appendix_physical_counter_normalizer as trusted_normalizer
import appendix_process_joule_collector as process_joule
import physical_counter_attestation
import spec_reentry_scaffold


ROOT = pathlib.Path(__file__).resolve().parents[2]
OBSERVER_PATH = (
    ROOT / "reports" / "condense" / "doctor_v5_ultra" /
    "post_120b" / "observer_state.json"
)
SCHEMA = "hawking.appendix_physical_counter_collector.v1"
STATUS_SCHEMA = "hawking.appendix_physical_counter_collector_status.v1"
ATTRIBUTED_SCHEMA = trusted_normalizer.ATTRIBUTED_SCHEMA
PHASE_SCHEMA = "hawking.physical_phase_markers.v1"
DEVICE_COUNTER_SCHEMA = "hawking.tq_runtime_physical_counters.v2"
SPEC_COUNTER_SCHEMA = "hawking.spec_tq_physical_counters.v2"
RUNTIME_PATHS = ("stored", "compact", "hashed", "computed")
DEVICE_DOMAINS = ("energy", "gpu_time", "physical_bytes", "occupancy", "bandwidth")
SPEC_DOMAINS = ("energy", "gpu_time", "physical_bytes")
DOMAIN_COLLECTOR = {
    "energy": "process_joule",
    "gpu_time": "xctrace",
    "physical_bytes": "xctrace",
    "occupancy": "xctrace",
    "bandwidth": "xctrace",
}


def canonical_sha256(value: Any) -> str:
    return appendix_contract.canonical_sha256(value)


def _stamp(value: dict[str, Any], field: str) -> dict[str, Any]:
    stamped = copy.deepcopy(value)
    stamped.pop(field, None)
    stamped[field] = canonical_sha256(stamped)
    return stamped


def build_config() -> dict[str, Any]:
    """Return the inert future-launch and normalization contract."""
    return _stamp({
        "schema": SCHEMA,
        "default_off": True,
        "collection_cli_exposed": False,
        "runtime_default_mutation": False,
        "collectors_start_concurrently": False,
        "collectors": [
            {
                "id": "process_joule",
                "binary": "release probe self-sampling proc_pid_rusage/RUSAGE_INFO_V6",
                "domains": ["energy"],
                "reported_process_metric": "ri_energy_nj cumulative direct-process counter",
                "direct_process_joule_backend_admitted": False,
                "requires_privilege_receipt": False,
                "required_attribution": (
                    "probe-pid+process-uuid+process-start+phase-marker+operation-bracketing-before/after"
                ),
                "raw_capture_must_be_immutable": True,
            },
            {
                "id": "powermetrics",
                "binary": "/usr/bin/powermetrics",
                "domains": [],
                "reported_process_metric": "energy-impact proxy (not joules)",
                "physical_evidence_eligible": False,
                "reason": "estimated proxy cannot satisfy direct-process-joule contract",
            },
            {
                "id": "xctrace",
                "binary": "full-Xcode xctrace selected by capability probe",
                "domains": ["gpu_time", "physical_bytes", "occupancy", "bandwidth"],
                "requires_privilege_receipt": True,
                "required_template": "Metal System Trace",
                "command-line-tools-stub_is_capable": False,
                "required_attribution": "probe-pid+run-nonce+phase-marker+Metal-registry-id",
                "raw_capture_must_be_immutable": True,
            },
        ],
        "launch_order": [
            "consume an already-held inherited shared-heavy-lease descriptor",
            "revalidate final-ready release attestation and zero heavy owners under the lease",
            "verify collector binary/capability/privilege receipts and an unused output directory",
            "start xctrace before releasing the probe barrier; libproc snapshots bracket every operation-only phase interval",
            "run the exact hash-bound probe and retain every phase-marker ID",
            "stop collectors only after both probe wall and continuous intervals are covered",
            "seal raw captures read-only, normalize v2, then create a separate counter attestation",
        ],
        "admission_hooks": {
            "final_interpretation_ready_attestation": True,
            "zero_heavy_owners_rechecked_under_lease": True,
            "inherited_shared_heavy_lease_required": True,
            "lease_opened_by_this_module_now": False,
            "ram_swap_guard_healthy": True,
        },
        "authoritative_outputs": {
            "device": DEVICE_COUNTER_SCHEMA,
            "spec": SPEC_COUNTER_SCHEMA,
            "counter_attestation_is_separate": True,
            "legacy_v1_projection_emitted": False,
        },
        "trusted_normalizer": {
            "schema": trusted_normalizer.SCHEMA,
            "contract_sha256": trusted_normalizer.CONTRACT_SHA256,
            "attributed_schema": trusted_normalizer.ATTRIBUTED_SCHEMA,
            "unsupported_backend_policy": "fail-closed",
        },
    }, "config_sha256")


def _default_binary_paths() -> dict[str, str | None]:
    """Prefer a real Xcode tool over the inert /usr/bin compatibility stub."""
    xcode_trace = pathlib.Path(
        "/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace"
    )
    return {
        "process_joule": str(pathlib.Path(process_joule.__file__).resolve()),
        "xctrace": str(xcode_trace) if xcode_trace.is_file() else shutil.which("xctrace"),
    }


def _runtime_capability(collector_id: str, path: str | None) -> tuple[bool, str]:
    """Cheaply prove the installed command can expose the required collector.

    Merely finding ``/usr/bin/xctrace`` is not sufficient on a Command Line
    Tools-only host: that file is a dispatcher which exits until full Xcode is
    selected.  The template listing is read-only and does not start a trace.
    """
    if not path:
        return False, "binary unavailable"
    if collector_id == "process_joule":
        value = process_joule.status()
        return (
            value["direct_process_nanojoule_backend_available"] is True,
            "; ".join(value["blockers"]) or "libproc RUSAGE_INFO_V6 is available",
        )
    try:
        process = subprocess.run(
            [path, "list", "templates"], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"capability probe failed: {exc}"
    detail = (process.stdout or process.stderr).strip()
    if process.returncode != 0:
        return False, detail[-1000:] or f"xctrace exited {process.returncode}"
    if "Metal System Trace" not in process.stdout:
        return False, "full-Xcode xctrace lacks the required Metal System Trace template"
    return True, "Metal System Trace template is available"


def status(
    *,
    euid: int | None = None,
    binary_paths: dict[str, str | None] | None = None,
    final_ready: bool | None = None,
    active_heavy_owner_count: int | None = None,
    capability_receipts: dict[str, bool] | None = None,
    capability_checks: dict[str, tuple[bool, str] | bool] | None = None,
) -> dict[str, Any]:
    """Read-only readiness view.  It never opens the heavy lease."""
    euid = os.geteuid() if euid is None else euid
    if final_ready is None:
        try:
            with OBSERVER_PATH.open("r", encoding="utf-8") as handle:
                observer = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError):
            observer = None
        final_ready = bool(
            isinstance(observer, dict)
            and observer.get("final_interpretation_ready") is True
        )
    if active_heavy_owner_count is None:
        active_heavy_owner_count = len(spec_reentry_scaffold.active_heavy_owners())
    explicit_paths = binary_paths is not None
    paths = _default_binary_paths() if binary_paths is None else binary_paths
    receipts = {} if capability_receipts is None else capability_receipts
    rows = []
    for collector_id in ("process_joule", "xctrace"):
        path = paths.get(collector_id)
        supplied = None if capability_checks is None else capability_checks.get(collector_id)
        if isinstance(supplied, tuple):
            runtime_capable, capability_detail = supplied
        elif isinstance(supplied, bool):
            runtime_capable, capability_detail = supplied, "caller-supplied capability result"
        elif explicit_paths:
            # Deterministic test/dry-run callers may provide a synthetic path
            # set.  Production status, which omits binary_paths, always probes.
            runtime_capable, capability_detail = bool(path), "synthetic path-only status input"
        else:
            runtime_capable, capability_detail = _runtime_capability(collector_id, path)
        privilege = bool(path)
        rows.append({
            "id": collector_id,
            "path": path,
            "available": bool(path),
            "runtime_capable": runtime_capable is True,
            "capability_probe_detail": capability_detail,
            "privilege_available": privilege,
            "capability_receipt_verified": receipts.get(collector_id) is True,
            "attribution_verified": receipts.get(collector_id) is True,
            "direct_process_joule_capable": (
                runtime_capable is True if collector_id == "process_joule" else None
            ),
        })
    blockers = []
    if not final_ready:
        blockers.append("Doctor final_interpretation_ready is not attested")
    if active_heavy_owner_count:
        blockers.append("heavy owners remain")
    for row in rows:
        if not row["available"]:
            blockers.append(f"{row['id']} is unavailable")
        elif not row["runtime_capable"]:
            blockers.append(
                f"{row['id']} runtime capability is unavailable: "
                f"{row['capability_probe_detail']}"
            )
        if not row["privilege_available"]:
            blockers.append(f"{row['id']} privilege is unavailable")
        if not row["capability_receipt_verified"]:
            blockers.append(f"{row['id']} capability/attribution receipt is absent")
        if row["id"] == "process_joule" and row["runtime_capable"] is not True:
            blockers.append("direct-process-joule backend is not structurally capable")
    blockers.append("powermetrics energy-impact proxy is explicitly ineligible for joule evidence")
    blockers.append("collection activation surface is intentionally not exposed")
    return {
        "schema": STATUS_SCHEMA,
        "config_sha256": build_config()["config_sha256"],
        "collectors": rows,
        "final_interpretation_ready": final_ready,
        "active_heavy_owner_count": active_heavy_owner_count,
        "shared_heavy_lease_opened": False,
        "execution_ready": False,
        "blockers": blockers,
    }


def _finite(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value)) and (value > 0 if positive else value >= 0)
    )


def _phase_targets(bundle: dict[str, Any], kind: str) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    raw = bundle.get("raw_probe") if isinstance(bundle, dict) else None
    phase = raw.get("phase_markers") if isinstance(raw, dict) else None
    if not isinstance(phase, dict) or phase.get("schema") != PHASE_SCHEMA:
        return [], ["raw bundle lacks physical phase markers"]
    run_nonce = phase.get("run_nonce")
    authority = bundle.get("execution_authority", {})
    if run_nonce != authority.get("run_nonce"):
        errors.append("phase-marker run nonce differs from execution authority")
    continuous_start = authority.get("started_at_continuous_ns")
    continuous_end = authority.get("ended_at_continuous_ns")
    continuous_elapsed = (
        continuous_end - continuous_start
        if isinstance(continuous_start, int) and not isinstance(continuous_start, bool)
        and isinstance(continuous_end, int) and not isinstance(continuous_end, bool)
        else None
    )
    errors.extend(physical_counter_attestation.validate_phase_markers(
        phase,
        run_nonce=authority.get("run_nonce"),
        workload_started_at_unix_ns=authority.get("started_at_unix_ns"),
        workload_ended_at_unix_ns=authority.get("ended_at_unix_ns"),
        workload_elapsed_continuous_ns=continuous_elapsed,
    ))
    claimed = phase.get("phase_markers_sha256")
    unstamped = copy.deepcopy(phase)
    unstamped.pop("phase_markers_sha256", None)
    if claimed != canonical_sha256(unstamped):
        errors.append("phase marker manifest hash mismatch")
    if phase.get("clock_source") != "mach_absolute_time_plus_system_time_unix_epoch":
        errors.append("phase markers lack the absolute wall+continuous macOS clock")
    for phase_field, authority_field in (
        ("probe_started_wall_unix_ns", "started_at_unix_ns"),
        ("probe_ended_wall_unix_ns", "ended_at_unix_ns"),
        ("probe_started_continuous_ns", "started_at_continuous_ns"),
        ("probe_ended_continuous_ns", "ended_at_continuous_ns"),
    ):
        value = phase.get(phase_field)
        authority_value = authority.get(authority_field)
        is_start = "started" in phase_field
        if (
            isinstance(value, bool) or not isinstance(value, int)
            or isinstance(authority_value, bool) or not isinstance(authority_value, int)
            or (value < authority_value if is_start else value > authority_value)
        ):
            errors.append(f"phase {phase_field} is outside execution authority")
    phase_intervals = phase.get("intervals")
    intervals = {
        row.get("interval_sha256"): row for row in (
            phase_intervals if isinstance(phase_intervals, list) else []
        )
        if isinstance(row, dict)
    }
    targets = []
    phase_pairs = phase.get("pairs")
    for pair in phase_pairs if isinstance(phase_pairs, list) else []:
        if not isinstance(pair, dict) or pair.get("phase") != "trial":
            continue
        unstamped_pair = copy.deepcopy(pair)
        pair_hash = unstamped_pair.pop("phase_marker_sha256", None)
        if pair_hash != canonical_sha256(unstamped_pair):
            errors.append("phase pair self-hash mismatch")
        interval_key = "candidate_interval_sha256" if kind == "device" else "verifier_interval_sha256"
        interval = intervals.get(pair.get(interval_key))
        target = {
            "marker": pair.get("phase_marker_sha256"),
            "interval_sha256": pair.get(interval_key),
            "batch": pair.get("batch"),
            "iteration": pair.get("iteration"),
            "interval": interval,
        }
        targets.append(target)
    targets.sort(key=lambda row: ((row["batch"] or 0), row["iteration"] if isinstance(row["iteration"], int) else -1))
    expected = raw.get("benchmark", {}).get("trials") if kind == "device" else None
    if kind == "device" and (not isinstance(expected, int) or len(targets) != expected):
        errors.append("device trial phase-marker coverage is not exact")
    if kind == "device":
        raw_marker_index = raw.get("benchmark", {}).get("trial_phase_marker_sha256")
        if raw_marker_index != [row["marker"] for row in targets]:
            errors.append("device phase pairs differ from the raw trial marker index")
    if kind == "spec":
        grouped = [(row["batch"], row["iteration"]) for row in targets]
        if not grouped or sorted({row[0] for row in grouped}) != list(range(1, 9)):
            errors.append("spec trial phase markers do not cover B=1..8")
        protocol_rows = raw.get("measurement_protocol", {}).get("batches")
        protocol_by_batch = {
            row.get("b"): row for row in (
                protocol_rows if isinstance(protocol_rows, list) else []
            ) if isinstance(row, dict)
        }
        for batch in range(1, 9):
            expected_markers = [
                row["marker"] for row in targets if row["batch"] == batch
            ]
            observed_markers = [
                row.get("phase_marker_sha256")
                for row in protocol_by_batch.get(batch, {}).get("repeats", [])
                if isinstance(row, dict)
            ]
            if observed_markers != expected_markers:
                errors.append(f"spec B={batch} phase pairs differ from the raw repeat marker index")
    if any(not isinstance(row["marker"], str) or len(row["marker"]) != 64 for row in targets):
        errors.append("phase marker ID is invalid")
    if len({row["marker"] for row in targets}) != len(targets):
        errors.append("phase marker ID is reused")
    if any(not isinstance(row["interval"], dict) for row in targets):
        errors.append("phase pair does not resolve to its candidate/verifier interval")
    for row in targets:
        interval = row["interval"]
        if not isinstance(interval, dict):
            continue
        unstamped_interval = copy.deepcopy(interval)
        interval_hash = unstamped_interval.pop("interval_sha256", None)
        if interval_hash != canonical_sha256(unstamped_interval):
            errors.append("phase interval self-hash mismatch")
        for start_field, end_field in (
            ("wall_started_unix_ns", "wall_ended_unix_ns"),
            ("continuous_started_ns", "continuous_ended_ns"),
        ):
            start, end = interval.get(start_field), interval.get(end_field)
            if (
                isinstance(start, bool) or not isinstance(start, int)
                or isinstance(end, bool) or not isinstance(end, int) or end <= start
            ):
                errors.append("phase interval wall/continuous bounds are invalid")
    return targets, errors


def validate_attributed_samples(
    bundle: dict[str, Any], manifest: Any, *, kind: str,
    expected_probe_pid: int | None = None,
    expected_capture_sha256s: dict[str, str] | None = None,
    expected_metal_registry_id: str | None = None,
    expected_lease: dict[str, int] | None = None,
) -> list[str]:
    if kind not in {"device", "spec"}:
        return ["kind must be device or spec"]
    if not isinstance(manifest, dict):
        return ["attributed sample manifest must be an object"]
    errors: list[str] = []
    expected_fields = {
        "schema", "kind", "normalizer", "lease", "probe_pid",
        "probe_argv_sha256", "metal_registry_id", "raw_bundle_sha256",
        "artifact_sha256", "runtime_path", "run_nonce",
        "phase_markers_sha256", "collectors", "samples", "manifest_sha256",
    }
    if set(manifest) != expected_fields:
        errors.append("attributed sample manifest fields are incomplete or unexpected")
    if manifest.get("schema") != ATTRIBUTED_SCHEMA or manifest.get("kind") != kind:
        errors.append("attributed sample schema/kind is invalid")
    unstamped = copy.deepcopy(manifest)
    claimed = unstamped.pop("manifest_sha256", None)
    if claimed != canonical_sha256(unstamped):
        errors.append("attributed sample manifest hash mismatch")
    raw = bundle.get("raw_probe", {})
    authority = bundle.get("execution_authority", {})
    phase = raw.get("phase_markers", {})
    normalizer = manifest.get("normalizer")
    expected_normalizer = {
        "schema", "contract_sha256", "binary",
    }
    if not isinstance(normalizer, dict) or set(normalizer) != expected_normalizer:
        errors.append("trusted normalizer identity is incomplete or unexpected")
    else:
        if normalizer.get("schema") != trusted_normalizer.SCHEMA \
                or normalizer.get("contract_sha256") != trusted_normalizer.CONTRACT_SHA256:
            errors.append("trusted normalizer schema/contract hash differs")
        try:
            current_normalizer = physical_counter_attestation.file_identity(
                pathlib.Path(trusted_normalizer.__file__),
            )
        except (OSError, ValueError) as exc:
            errors.append(f"trusted normalizer executable cannot be identified: {exc}")
        else:
            if normalizer.get("binary") != current_normalizer:
                errors.append("trusted normalizer executable identity differs")
    lease = manifest.get("lease")
    if not isinstance(lease, dict) or set(lease) != {"inherited", "device", "inode"} \
            or lease.get("inherited") is not True \
            or any(isinstance(lease.get(field), bool) or not isinstance(lease.get(field), int)
                   or lease[field] <= 0 for field in ("device", "inode")):
        errors.append("attributed samples lack an exact inherited lease identity")
    if expected_lease is not None and lease != {
        "inherited": True,
        "device": expected_lease.get("device"),
        "inode": expected_lease.get("inode"),
    }:
        errors.append("attributed samples inherited lease differs from executor authority")
    probe_pid = manifest.get("probe_pid")
    if isinstance(probe_pid, bool) or not isinstance(probe_pid, int) or probe_pid <= 0:
        errors.append("attributed samples probe PID is invalid")
    if expected_probe_pid is not None and probe_pid != expected_probe_pid:
        errors.append("attributed samples probe PID differs from the launched process")
    if manifest.get("probe_argv_sha256") != authority.get("argv_sha256"):
        errors.append("attributed samples differ from the exact probe argv")
    registry_id = manifest.get("metal_registry_id")
    if not isinstance(registry_id, str) or not registry_id:
        errors.append("attributed samples Metal registry ID is empty")
    if expected_metal_registry_id is not None and registry_id != expected_metal_registry_id:
        errors.append("attributed samples Metal registry ID differs from signed device authority")
    for field, expected in (
        ("raw_bundle_sha256", bundle.get("raw_bundle_sha256")),
        ("artifact_sha256", raw.get("artifact", {}).get("sha256")),
        ("run_nonce", authority.get("run_nonce")),
        ("phase_markers_sha256", phase.get("phase_markers_sha256")),
    ):
        if manifest.get(field) != expected:
            errors.append(f"attributed samples {field} binding differs")
    if manifest.get("runtime_path") != raw.get("runtime_path") or raw.get("runtime_path") not in RUNTIME_PATHS:
        errors.append("attributed samples runtime path differs or is invalid")

    required = DEVICE_DOMAINS if kind == "device" else SPEC_DOMAINS
    collectors = manifest.get("collectors")
    by_id = {row.get("id"): row for row in collectors or [] if isinstance(row, dict)}
    if set(by_id) != {"process_joule", "xctrace"}:
        errors.append("both exact collectors are required")
    start_u, end_u = authority.get("started_at_unix_ns"), authority.get("ended_at_unix_ns")
    start_c, end_c = authority.get("started_at_continuous_ns"), authority.get("ended_at_continuous_ns")

    def covers(row: dict[str, Any]) -> bool:
        values = (
            start_u, end_u, start_c, end_c,
            row.get("capture_started_at_unix_ns"), row.get("capture_ended_at_unix_ns"),
            row.get("capture_started_at_continuous_ns"), row.get("capture_ended_at_continuous_ns"),
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            return False
        return bool(
            row["capture_started_at_unix_ns"] <= start_u < end_u
            <= row["capture_ended_at_unix_ns"]
            and row["capture_started_at_continuous_ns"] <= start_c < end_c
            <= row["capture_ended_at_continuous_ns"]
        )

    for collector_id, row in by_id.items():
        expected_collector_fields = {
            "id", "backend_id", "raw_capture_sha256", "available",
            "privilege_verified", "process_attributed", "phase_attributed",
            "directly_measured", "estimated", "apportioned", "domains",
            "capture_started_at_unix_ns", "capture_ended_at_unix_ns",
            "capture_started_at_continuous_ns", "capture_ended_at_continuous_ns",
        }
        if set(row) != expected_collector_fields:
            errors.append(f"{collector_id} collector fields are incomplete or unexpected")
        expected_backend = (
            trusted_normalizer.DIRECT_JOULE_BACKEND
            if collector_id == "process_joule" else trusted_normalizer.METAL_BACKEND
        )
        if row.get("backend_id") != expected_backend:
            errors.append(f"{collector_id} backend is unsupported")
        capture_sha = row.get("raw_capture_sha256")
        if not isinstance(capture_sha, str) or len(capture_sha) != 64:
            errors.append(f"{collector_id} raw capture hash is invalid")
        if expected_capture_sha256s is not None \
                and capture_sha != expected_capture_sha256s.get(collector_id):
            errors.append(f"{collector_id} raw capture differs from executor-sealed input")
        if row.get("available") is not True:
            errors.append(f"{collector_id} source is unavailable")
        if row.get("privilege_verified") is not True:
            errors.append(f"{collector_id} source is unprivileged")
        if row.get("process_attributed") is not True or row.get("phase_attributed") is not True:
            errors.append(f"{collector_id} source is unattributable")
        if row.get("directly_measured") is not True or row.get("estimated") is not False \
                or row.get("apportioned") is not False:
            errors.append(f"{collector_id} source is estimated, apportioned, or indirect")
        if set(row.get("domains", [])) != {d for d in required if DOMAIN_COLLECTOR[d] == collector_id}:
            errors.append(f"{collector_id} domain coverage is not exact")
        if not covers(row):
            errors.append(f"{collector_id} capture does not cover the exact wall+continuous probe interval")

    targets, target_errors = _phase_targets(bundle, kind)
    errors.extend(target_errors)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != len(targets):
        errors.append("attributed samples do not cover every exact trial marker")
        samples = []
    seen_source_ids: dict[str, set[str]] = {domain: set() for domain in required}
    for ordinal, (sample, target) in enumerate(zip(samples, targets)):
        if not isinstance(sample, dict):
            errors.append(f"sample {ordinal} is malformed")
            continue
        if (
            sample.get("ordinal") != ordinal
            or sample.get("phase_marker_sha256") != target["marker"]
            or sample.get("interval_sha256") != target["interval_sha256"]
            or sample.get("run_nonce") != authority.get("run_nonce")
        ):
            errors.append(f"sample {ordinal} is not bound to its exact phase marker/run")
        if kind == "spec" and (
            sample.get("batch") != target["batch"] or sample.get("repeat") != target["iteration"]
        ):
            errors.append(f"sample {ordinal} spec batch/repeat binding differs")
        interval = target["interval"]
        if isinstance(interval, dict) and any(
            sample.get(sample_field) != interval.get(marker_field)
            for sample_field, marker_field in (
                ("interval_started_at_unix_ns", "wall_started_unix_ns"),
                ("interval_ended_at_unix_ns", "wall_ended_unix_ns"),
                ("interval_started_at_continuous_ns", "continuous_started_ns"),
                ("interval_ended_at_continuous_ns", "continuous_ended_ns"),
            )
        ):
            errors.append(f"sample {ordinal} does not exactly join its wall+continuous phase interval")
        if not isinstance(sample.get("process_id"), int) or sample["process_id"] <= 0 \
                or sample.get("process_id") != probe_pid:
            errors.append(f"sample {ordinal} lacks process attribution")
        provenance = sample.get("energy_provenance")
        expected_provenance = {
            "backend_id": trusted_normalizer.DIRECT_JOULE_BACKEND,
            "quantity": "energy",
            "unit": "joule",
            "scope": "exact-probe-process",
            "attribution": "direct-counter",
            "estimated": False,
            "apportioned": False,
            "source_process_id": probe_pid,
        }
        if provenance != expected_provenance:
            errors.append(f"sample {ordinal} energy is not a direct, non-apportioned process-joule counter")
        sources = sample.get("source_sample_ids")
        if not isinstance(sources, dict) or set(sources) != set(required) or any(
            not isinstance(sources[d], list) or not sources[d]
            or any(not isinstance(source_id, str) or not source_id for source_id in sources[d])
            for d in required
        ):
            errors.append(f"sample {ordinal} lacks exact per-domain source IDs")
        elif isinstance(sources, dict):
            for domain in required:
                duplicates = set(sources[domain]) & seen_source_ids[domain]
                if duplicates:
                    errors.append(f"sample {ordinal} reuses {domain} source record IDs")
                seen_source_ids[domain].update(sources[domain])
        for field in ("energy_j", "gpu_time_ns", "physical_bytes"):
            if not _finite(sample.get(field), positive=True):
                errors.append(f"sample {ordinal} {field} is invalid")
        if kind == "device":
            if not _finite(sample.get("occupancy_percent")) or sample.get("occupancy_percent", 101) > 100:
                errors.append(f"sample {ordinal} occupancy is invalid")
            if not _finite(sample.get("bandwidth_bytes_per_second"), positive=True):
                errors.append(f"sample {ordinal} bandwidth is invalid")
    return errors


def normalize_v2(
    bundle: dict[str, Any], manifest: dict[str, Any], *, kind: str,
    expected_probe_pid: int | None = None,
    expected_capture_sha256s: dict[str, str] | None = None,
    expected_metal_registry_id: str | None = None,
    expected_lease: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Create the authoritative v2 payload; never project to legacy v1."""
    errors = validate_attributed_samples(
        bundle, manifest, kind=kind, expected_probe_pid=expected_probe_pid,
        expected_capture_sha256s=expected_capture_sha256s,
        expected_metal_registry_id=expected_metal_registry_id,
        expected_lease=expected_lease,
    )
    if errors:
        raise ValueError("; ".join(errors))
    raw = bundle["raw_probe"]
    rows = manifest["samples"]
    common = {
        "schema": DEVICE_COUNTER_SCHEMA if kind == "device" else SPEC_COUNTER_SCHEMA,
        "raw_bundle_sha256": bundle["raw_bundle_sha256"],
        "artifact_sha256": raw["artifact"]["sha256"],
        "runtime_path": raw["runtime_path"],
        "phase_markers_sha256": raw["phase_markers"]["phase_markers_sha256"],
    }
    if kind == "device":
        trials = [{
            "index": index,
            "phase_marker_sha256": row["phase_marker_sha256"],
            "energy_j": row["energy_j"],
            "gpu_time_ns": row["gpu_time_ns"],
            "physical_bytes": row["physical_bytes"],
            "occupancy_percent": row["occupancy_percent"],
            "bandwidth_bytes_per_second": row["bandwidth_bytes_per_second"],
        } for index, row in enumerate(rows)]
        return {
            **common,
            "tensor": raw["tensor"]["name"],
            "trials": trials,
            "summary": {
                "energy_j_total": sum(float(row["energy_j"]) for row in trials),
                "gpu_time_ns_total": sum(int(row["gpu_time_ns"]) for row in trials),
                "physical_bytes_total": sum(int(row["physical_bytes"]) for row in trials),
                "occupancy_percent_mean": statistics.fmean(float(row["occupancy_percent"]) for row in trials),
                "bandwidth_bytes_per_second_mean": statistics.fmean(float(row["bandwidth_bytes_per_second"]) for row in trials),
            },
        }
    batches = []
    for batch in range(1, 9):
        batch_rows = [row for row in rows if row["batch"] == batch]
        batches.append({
            "b": batch,
            "repeats": [{
                "repeat": row["repeat"],
                "phase_marker_sha256": row["phase_marker_sha256"],
                "energy_j": row["energy_j"],
                "gpu_time_ns": row["gpu_time_ns"],
                "physical_bytes": row["physical_bytes"],
            } for row in batch_rows],
        })
    return {**common, "batches": batches}


def dry_run(kind: str) -> dict[str, Any]:
    return {
        "schema": "hawking.appendix_physical_counter_dry_run.v1",
        "kind": kind,
        "config_sha256": build_config()["config_sha256"],
        "would_execute": False,
        "would_open_heavy_lease": False,
        "authoritative_schema": DEVICE_COUNTER_SCHEMA if kind == "device" else SPEC_COUNTER_SCHEMA,
        "blockers": ["an admitted release orchestrator and verified attributed captures are required"],
    }


def _selftest() -> int:
    config = build_config()
    assert config["default_off"] is True
    assert config["collection_cli_exposed"] is False
    assert status(euid=501)["execution_ready"] is False
    assert dry_run("device")["would_execute"] is False
    print("appendix_physical_counter_collector.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true")
    group.add_argument("--dry-run", choices=("device", "spec"))
    group.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    payload = status() if args.status else dry_run(args.dry_run)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3.12
"""Seal and audit the default-off Doctor V5 single-device sprint stack.

This tool is deliberately outside the live queue import graph.  It never edits a
campaign plan, runtime spec, registry, result, or runtime default.  It records
which acceleration capabilities exist, binds their source bytes, reports the
live release boundary, and refuses to call staged code production-ready without
an owner-free full-stack receipt accepted by ``doctor_v5_single_device_benchmark``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import doctor_v5_single_device_benchmark as benchmark
import doctor_v5_local_observer as local_observer


SCHEMA = "hawking.doctor_v5_single_device_sprint_audit.v1"
VERSION = "2026-07-14.3"
STAGE_ROOT = ROOT / "reports/condense/doctor_v5_ultra/staged_acceleration/single_device_v1"
DEFAULT_PACKET = STAGE_ROOT / "single_device_sprint_audit.json"
QUEUE_STATE = ROOT / "reports/condense/doctor_v5_ultra/queue_state.json"
OBSERVER_STATE = ROOT / "reports/condense/doctor_v5_ultra/post_120b/observer_state.json"
PRODUCTION_ETA = (
    ROOT / "reports/condense/doctor_v5_ultra/staged_acceleration/production_calibrated_eta.json"
)
AGGRESSIVE_OVERLAY = (
    ROOT / "reports/condense/doctor_v5_ultra/staged_acceleration/aggressive_v2/"
    "aggressive_admission_overlay.json"
)
MAX_JSON_BYTES = 64 * 1024 * 1024


COMPONENT_SOURCES: dict[str, tuple[str, ...]] = {
    "qualified-thread-profile": (
        "vendor/strand-quant/tools/thread_profile_contract.py",
        "tools/condense/doctor_v5_aggressive_admission_policy.py",
    ),
    "ordered-read-rht-encode-write": (
        "vendor/strand-quant/src/ordered_pipeline.rs",
        "vendor/strand-quant/src/bin/gate-quantize-model-ordered-pipeline.rs",
    ),
    "cross-shard-finalize-prepare-window": (
        "tools/condense/doctor_v5_qwen_shard_window.py",
    ),
    "shared-rate-branch-preprocessing": (
        "tools/condense/doctor_v5_shared_preprocess_cache.py",
    ),
    "elastic-phase-admission": (
        "tools/condense/doctor_v5_elastic_phase_scheduler.py",
        "tools/condense/doctor_v5_local_observer.py",
        "tools/condense/doctor_v5_inert_phase_launcher.py",
        "tools/condense/doctor_v5_fixture_phase_validator.py",
    ),
    "native-pgo-io-profile": (
        "vendor/strand-quant/Cargo.toml",
        "vendor/strand-quant/src/lib.rs",
        "vendor/strand-quant/src/bin/quantize-model.rs",
        "vendor/strand-quant/tools/native_build.py",
        "vendor/strand-quant/src/native_io.rs",
        "vendor/strand-quant/src/safetensor_io.rs",
    ),
    "host-sprint-isolation": (
        "tools/condense/doctor_v5_host_sprint_plan.py",
        "tools/condense/doctor_v5_local_observer.py",
    ),
    "controlled-swap-shock-absorber": (
        "tools/condense/doctor_v5_aggressive_admission_policy.py",
    ),
    "phase-aware-remaining-scratch": (
        "tools/condense/doctor_v5_remaining_scratch_ledger.py",
        "tools/condense/doctor_v5_remaining_scratch_gate_adapter.py",
        "docs/plans/DOCTOR_V5_REMAINING_SCRATCH_LEDGER.md",
    ),
    "metal-rht-preprocessing": (
        "vendor/strand-quant/src/metal_rht_probe.rs",
        "vendor/strand-quant/src/bin/probe-metal-rht.rs",
        "vendor/strand-quant/tools/native_probe.py",
    ),
}

CHEAP_TEST_SOURCES = (
    "tools/condense/tests/test_doctor_v5_physical_ab_controller.py",
    "tools/condense/tests/test_doctor_v5_physical_ab_executor.py",
    "tools/condense/tests/test_doctor_v5_single_device_benchmark.py",
    "tools/condense/tests/test_doctor_v5_single_device_sprint_audit.py",
    "tools/condense/tests/test_doctor_v5_elastic_host_sprint.py",
    "tools/condense/tests/test_doctor_v5_host_sprint_gate.py",
    "tools/condense/tests/test_doctor_v5_shared_preprocess_cache_suite.py",
    "tools/condense/tests/test_doctor_v5_production_eta.py",
    "tools/condense/tests/test_doctor_v5_acceleration_eta.py",
    "tools/condense/tests/test_doctor_v5_blocked_cell_recovery.py",
    "tools/condense/tests/test_doctor_v5_resource_stop_recovery_stage.py",
    "tools/condense/tests/test_doctor_v5_remaining_scratch_ledger.py",
    "tools/condense/tests/test_doctor_v5_remaining_scratch_gate_adapter.py",
)

CONTROL_SOURCES = (
    "tools/condense/doctor_v5_acceleration_eta.py",
    "tools/condense/doctor_v5_blocked_cell_recovery.py",
    "tools/condense/doctor_v5_resource_stop_recovery_stage.py",
    "docs/plans/DOCTOR_V5_3BPW_RESOURCE_STOP_RECOVERY.md",
    "tools/condense/doctor_v5_production_eta.py",
    "tools/condense/doctor_v5_physical_ab_controller.py",
    "tools/condense/doctor_v5_physical_ab_executor.py",
    "tools/condense/doctor_v5_physical_counter_barrier.py",
    "tools/condense/doctor_v5_remaining_scratch_ledger.py",
    "tools/condense/doctor_v5_single_device_benchmark.py",
    "tools/condense/doctor_v5_single_device_sprint_audit.py",
)

# One source of truth prevents the sprint audit and Appendix owner gate from
# silently disagreeing about a competing heavy process.
HEAVY_COMMAND_PATTERNS = local_observer.HEAVY_COMMAND_PATTERNS
THRESHOLD_NUMERIC_FIELDS = (
    "sub_120b_baseline_days_range",
    "sub_120b_required_additional_speedup_by_endpoint",
    "sub_120b_required_additional_speedup_for_entire_range",
    "unchanged_120b_plus_appendix_increment_days_by_endpoint",
    "full_campaign_required_sub_120b_speedup_by_endpoint_if_other_segments_unchanged",
)


class SprintAuditError(RuntimeError):
    """A packet, source binding, or release boundary is invalid."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: child for key, child in value.items() if key != field}


def _self_hash_matches(value: Any, field: str) -> bool:
    if not isinstance(value, dict) or not benchmark._valid_sha256(value.get(field)):
        return False
    try:
        return value[field] == _hash_value(_without(value, field))
    except (TypeError, ValueError):
        return False


def _read_json_bound(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode) \
                or before.st_size > MAX_JSON_BYTES:
            raise SprintAuditError(f"unsafe JSON artifact: {path}")
        raw = path.read_bytes()
        after = path.lstat()
        identity = lambda info: (info.st_dev, info.st_ino, info.st_size,
                                 info.st_mtime_ns)
        if identity(before) != identity(after) or len(raw) != after.st_size:
            raise SprintAuditError(f"JSON artifact changed while reading: {path}")
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SprintAuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SprintAuditError(f"JSON root is not an object: {path}")
    try:
        display = str(path.resolve(strict=True).relative_to(ROOT.resolve()))
    except ValueError:
        display = str(path.resolve(strict=True))
    return value, {"path": display, "sha256": hashlib.sha256(raw).hexdigest(),
                   "bytes": len(raw)}


def _read_json(path: Path) -> dict[str, Any]:
    return _read_json_bound(path)[0]


def _file_reference(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise SprintAuditError(f"component source is not a regular file: {relative}")
        resolved.relative_to(ROOT.resolve())
        digest = hashlib.sha256()
        size = 0
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk); size += len(chunk)
        after = path.lstat()
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) \
                or size != after.st_size:
            raise SprintAuditError(f"component source changed while hashing: {relative}")
    except (OSError, ValueError) as exc:
        raise SprintAuditError(f"cannot bind component source {relative}: {exc}") from exc
    return {"path": relative, "sha256": digest.hexdigest(), "bytes": size}


def _active_heavy_owners() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,lstart=,command="],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise SprintAuditError(f"cannot establish heavy-owner inventory: {exc}") from exc
    owners = []
    excluded = {os.getpid(), os.getppid()}
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 7)
        if len(fields) < 8:
            continue
        try:
            pid, ppid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        command = fields[7]
        if pid in excluded or not any(pattern.search(command)
                                      for pattern in HEAVY_COMMAND_PATTERNS):
            continue
        owners.append({"pid": pid, "ppid": ppid,
                       "process_started": " ".join(fields[2:7]),
                       "command_sha256": hashlib.sha256(command.encode()).hexdigest()})
    return sorted(owners, key=lambda row: row["pid"])


def _qualification_status() -> dict[str, Any]:
    if not AGGRESSIVE_OVERLAY.is_file():
        return {"status": "absent", "qualification_sha256": None,
                "overlay_artifact": None}
    overlay, overlay_reference = _read_json_bound(AGGRESSIVE_OVERLAY)
    qualification = overlay.get("thread_profile_qualification")
    return {
        "status": qualification.get("status") if isinstance(qualification, dict)
        else "absent",
        "qualification_sha256": (
            qualification.get("qualification_sha256")
            if isinstance(qualification, dict) else None
        ),
        "overlay_sha256": overlay.get("overlay_sha256"),
        "overlay_artifact": overlay_reference,
    }


def _production_eta_authority() \
        -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build a fresh v2 authority without rewriting the staged snapshot."""
    stored, stored_reference = _read_json_bound(PRODUCTION_ETA)
    try:
        current = benchmark.eta_contract.build()
        errors = benchmark.eta_contract.validate(current, verify_freshness=True)
    except (OSError, KeyError, TypeError, ValueError,
            benchmark.eta_contract.ProductionEtaError) as exc:
        raise SprintAuditError(
            f"cannot build current production ETA authority: {exc}"
        ) from exc
    if errors:
        raise SprintAuditError(
            "current production ETA authority is invalid: " + "; ".join(errors)
        )
    metadata = {
        "authority_source": "read-only-v2-build-from-hash-bound-live-inputs",
        "stored_schema": stored.get("schema"),
        "stored_document_sha256": stored.get("document_sha256"),
        "stored_snapshot_matches_current": (
            stored.get("schema") == benchmark.eta_contract.SCHEMA
            and stored.get("document_sha256") == current.get("document_sha256")
        ),
    }
    return current, stored_reference, metadata


def build_packet() -> dict[str, Any]:
    queue, queue_reference = _read_json_bound(QUEUE_STATE)
    observer, observer_reference = _read_json_bound(OBSERVER_STATE)
    eta, eta_reference, eta_metadata = _production_eta_authority()
    threshold = benchmark.build_threshold(production_eta=eta, target_days=7.0)
    components = {
        name: {"source_artifacts": [_file_reference(path) for path in paths],
               "implementation_present": True,
               "production_speedup_credit": False}
        for name, paths in sorted(COMPONENT_SOURCES.items())
    }
    tests = [_file_reference(path) for path in CHEAP_TEST_SOURCES]
    controls = [_file_reference(path) for path in CONTROL_SOURCES]
    owners = _active_heavy_owners()
    qualification = _qualification_status()
    benchmark_receipt = STAGE_ROOT / "production_full_stack_benchmark.json"
    benchmark_authority = STAGE_ROOT / "production_benchmark_authority.json"
    benchmark_errors: list[str]
    if benchmark_receipt.is_file() and benchmark_authority.is_file():
        benchmark_errors = benchmark.validate_receipt(
            _read_json(benchmark_receipt), require_production=True,
            production_authority=_read_json(benchmark_authority),
            production_eta_sha256=eta.get("document_sha256"),
        )
    elif benchmark_receipt.is_file():
        benchmark_errors = ["frozen external production benchmark authority is absent"]
    else:
        benchmark_errors = ["owner-free production full-stack benchmark is absent"]
    blockers = list(benchmark_errors)
    if owners:
        blockers.append("one or more heavy owners are active")
    if queue.get("status") not in {"complete", "completed", "terminal"}:
        blockers.append("Doctor queue is not at a terminal quiescent checkpoint")
    if observer.get("final_interpretation_ready") is not True:
        blockers.append("Doctor final interpretation gate is not ready")
    if qualification.get("status") != "qualified":
        blockers.append("exact 8/12/16/20 production thread profile is not qualified")
    if eta.get("eta_blocked") is True:
        blockers.append("current v2 production ETA is blocked and exposes no numeric threshold")
    if eta_metadata["stored_snapshot_matches_current"] is not True:
        blockers.append("stored production ETA snapshot is stale relative to current v2 authority")
    blockers.extend([
        "trusted physical benchmark runner/independent campaign attestation is absent",
        "owner-free trusted observer promotion receipt is absent",
        "live queue generation does not consume phase-aware remaining-scratch receipts",
        "native/PGO full-stack production training and A/B receipt is pending",
        "physical Metal parity/performance receipt is pending",
        "GPT-OSS 120B and Appendix require their own segment-specific receipts",
    ])
    packet: dict[str, Any] = {
        "schema": SCHEMA, "version": VERSION, "created_at": _now(),
        "mode": "unbound-default-off-audit-only",
        "snapshot_only": True,
        "historical_packet_cannot_activate": True,
        "current_release_readiness_requires_rebuild": True,
        "activation_permitted": False, "runtime_defaults_changed": False,
        "live_source_bound_files_mutated": False,
        "completed_evidence_mutated": False, "parent_sources_deleted": False,
        "components": components, "cheap_test_sources": tests,
        "control_sources": controls,
        "required_component_names": list(benchmark.REQUIRED_COMPONENTS),
        "optional_component_names": list(benchmark.OPTIONAL_COMPONENTS),
        "all_required_implementations_present": all(
            name in components for name in benchmark.REQUIRED_COMPONENTS
        ),
        "qualification": qualification,
        "release_boundary": {
            "queue_state": queue_reference,
            "queue_status": queue.get("status"),
            "queue_plan_sha256": queue.get("plan_sha256"),
            "observer_state": observer_reference,
            "final_interpretation_ready": observer.get("final_interpretation_ready"),
            "gpt_oss_120b_execution_ready": observer.get(
                "gpt_oss_120b_execution_ready"
            ),
            "active_heavy_owner_count": len(owners),
            "active_heavy_owners": owners,
            "owner_free": not owners,
        },
        "eta_claim_boundary": {
            "production_eta": eta_reference,
            "production_eta_authority": eta,
            **eta_metadata,
            "seven_day_threshold": threshold,
            "benchmark_receipt_path": str(benchmark_receipt.relative_to(ROOT)),
            "benchmark_authority_path": str(benchmark_authority.relative_to(ROOT)),
            "benchmark_validation_errors": benchmark_errors,
            "component_or_synthetic_speedups_multiplied": False,
            "unmeasured_120b_or_appendix_speedup_applied": False,
        },
        "production_ready": False,
        "blockers": sorted(set(blockers)),
    }
    packet["audit_sha256"] = _hash_value(packet)
    return packet


def validate_packet(packet: Any) -> list[str]:
    if not isinstance(packet, dict) or packet.get("schema") != SCHEMA \
            or packet.get("version") != VERSION:
        return ["single-device audit schema/version mismatch"]
    errors: list[str] = []
    if not _self_hash_matches(packet, "audit_sha256"):
        errors.append("single-device audit hash mismatch")
    if packet.get("mode") != "unbound-default-off-audit-only" \
            or packet.get("snapshot_only") is not True \
            or packet.get("historical_packet_cannot_activate") is not True \
            or packet.get("current_release_readiness_requires_rebuild") is not True \
            or packet.get("activation_permitted") is not False \
            or packet.get("runtime_defaults_changed") is not False \
            or packet.get("live_source_bound_files_mutated") is not False \
            or packet.get("completed_evidence_mutated") is not False \
            or packet.get("parent_sources_deleted") is not False:
        errors.append("single-device audit weakened isolation/lifecycle boundaries")
    components = packet.get("components")
    if not isinstance(components, dict) \
            or not set(benchmark.REQUIRED_COMPONENTS).issubset(components):
        errors.append("single-device audit component inventory is incomplete")
        components = {}
    for name, row in components.items():
        if name not in COMPONENT_SOURCES or not isinstance(row, dict) \
                or row.get("implementation_present") is not True \
                or row.get("production_speedup_credit") is not False:
            errors.append(f"invalid component capability row: {name}")
            continue
        expected = []
        try:
            expected = [_file_reference(path) for path in COMPONENT_SOURCES[name]]
        except SprintAuditError as exc:
            errors.append(str(exc))
        if row.get("source_artifacts") != expected:
            errors.append(f"component source binding changed: {name}")
    try:
        expected_tests = [_file_reference(path) for path in CHEAP_TEST_SOURCES]
    except SprintAuditError as exc:
        expected_tests = []
        errors.append(str(exc))
    if packet.get("cheap_test_sources") != expected_tests:
        errors.append("single-device cheap-test source binding changed")
    try:
        expected_controls = [_file_reference(path) for path in CONTROL_SOURCES]
    except SprintAuditError as exc:
        expected_controls = []
        errors.append(str(exc))
    if packet.get("control_sources") != expected_controls:
        errors.append("single-device control source binding changed")
    if packet.get("required_component_names") != list(benchmark.REQUIRED_COMPONENTS) \
            or packet.get("optional_component_names") \
            != list(benchmark.OPTIONAL_COMPONENTS) \
            or packet.get("all_required_implementations_present") is not True:
        errors.append("single-device component claim inventory differs")
    eta = packet.get("eta_claim_boundary")
    if not isinstance(eta, dict) \
            or eta.get("component_or_synthetic_speedups_multiplied") is not False \
            or eta.get("unmeasured_120b_or_appendix_speedup_applied") is not False \
            or eta.get("benchmark_receipt_path") \
            != str((STAGE_ROOT / "production_full_stack_benchmark.json").relative_to(ROOT)) \
            or eta.get("benchmark_authority_path") \
            != str((STAGE_ROOT / "production_benchmark_authority.json").relative_to(ROOT)):
        errors.append("single-device ETA claim boundary is weakened")
    threshold = eta.get("seven_day_threshold") if isinstance(eta, dict) else None
    authority = eta.get("production_eta_authority") \
        if isinstance(eta, dict) else None
    authority_errors = benchmark.eta_contract.validate(
        authority, verify_freshness=True
    ) if isinstance(authority, dict) else ["authority is absent"]
    if authority_errors:
        errors.append("single-device production ETA authority is invalid")
    try:
        stored, current_reference = _read_json_bound(PRODUCTION_ETA)
    except SprintAuditError as exc:
        stored, current_reference = {}, None
        errors.append(str(exc))
    stored_matches = (
        stored.get("schema") == benchmark.eta_contract.SCHEMA
        and isinstance(authority, dict)
        and stored.get("document_sha256") == authority.get("document_sha256")
    )
    if not isinstance(eta, dict) \
            or eta.get("production_eta") != current_reference \
            or eta.get("authority_source") \
            != "read-only-v2-build-from-hash-bound-live-inputs" \
            or eta.get("stored_schema") != stored.get("schema") \
            or eta.get("stored_document_sha256") != stored.get("document_sha256") \
            or eta.get("stored_snapshot_matches_current") is not stored_matches:
        errors.append("single-device production ETA source authority differs")
    expected_threshold = None
    if not authority_errors:
        try:
            expected_threshold = benchmark.build_threshold(
                production_eta=authority, target_days=7.0
            )
        except benchmark.SprintBenchmarkError:
            expected_threshold = None
    if not isinstance(threshold, dict) \
            or threshold != expected_threshold \
            or threshold.get("schema") != benchmark.THRESHOLD_SCHEMA \
            or not _self_hash_matches(threshold, "threshold_sha256") \
            or threshold.get("eta_scope") != "sub-120b-only" \
            or threshold.get("target_contract") \
            != benchmark.STRICT_TARGET_CONTRACT \
            or threshold.get("strict_inequality") \
            != benchmark.STRICT_SPEEDUP_RELATION \
            or threshold.get("threshold_equality_is_sufficient") is not False \
            or threshold.get("unmeasured_segment_speedup_applied") is not False \
            or threshold.get("component_speedups_multiplied") is not False:
        errors.append("single-device seven-day threshold authority is invalid")
    if isinstance(authority, dict) and authority.get("eta_blocked") is True \
            and (not isinstance(threshold, dict)
                 or threshold.get("available") is not False
                 or any(threshold.get(field) is not None
                        for field in THRESHOLD_NUMERIC_FIELDS)
                 or threshold.get("gpt_oss_120b_threshold_available") is not False
                 or threshold.get("appendix_threshold_available") is not False):
        errors.append("blocked production ETA exposes a numeric threshold")
    release = packet.get("release_boundary")
    release_owners = release.get("active_heavy_owners") \
        if isinstance(release, dict) else None
    if not isinstance(release, dict) or not isinstance(release_owners, list) \
            or release.get("active_heavy_owner_count") != len(release_owners) \
            or release.get("owner_free") \
            is not (release.get("active_heavy_owner_count") == 0):
        errors.append("single-device release-boundary owner inventory is inconsistent")
    elif any(not isinstance(row, dict)
             or isinstance(row.get("pid"), bool)
             or not isinstance(row.get("pid"), int) or row.get("pid") <= 0
             or isinstance(row.get("ppid"), bool)
             or not isinstance(row.get("ppid"), int) or row.get("ppid") < 0
             or not isinstance(row.get("process_started"), str)
             or not benchmark._valid_sha256(row.get("command_sha256"))
             for row in release_owners) \
            or [row["pid"] for row in release_owners] \
            != sorted({row["pid"] for row in release_owners}):
        errors.append("single-device release-boundary owner rows are invalid")
    blockers = packet.get("blockers")
    if packet.get("production_ready") is not False \
            or not isinstance(blockers, list) or not blockers \
            or any(not isinstance(value, str) or not value for value in blockers) \
            or blockers != sorted(set(blockers)):
        errors.append("staged audit overclaims production readiness")
    return errors


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True,
                         ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    stage = sub.add_parser("stage")
    stage.add_argument("--output", type=Path, default=DEFAULT_PACKET)
    verify = sub.add_parser("verify")
    verify.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            packet = _read_json(args.packet)
        else:
            packet = build_packet()
            if args.command == "stage":
                output = args.output.resolve()
                try:
                    output.relative_to(STAGE_ROOT.resolve())
                except ValueError as exc:
                    raise SprintAuditError(
                        "single-device packet must remain in its inert staging root"
                    ) from exc
                _atomic_json(output, packet)
        errors = validate_packet(packet)
        print(json.dumps({"ok": not errors, "errors": errors,
                          "audit_sha256": packet.get("audit_sha256"),
                          "production_ready": packet.get("production_ready"),
                          "blockers": packet.get("blockers", [])},
                         indent=2, sort_keys=True))
        return 0 if not errors else 2
    except (OSError, KeyError, TypeError, ValueError, SprintAuditError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

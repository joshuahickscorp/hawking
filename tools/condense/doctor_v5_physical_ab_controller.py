#!/usr/bin/env python3.12
"""Default-off physical A/B qualification controller for Doctor V5.

This is a trust contract, not an executor.  It has no run/activate/promote
command and never acquires the heavy lease, opens a model, invokes Metal/Cargo,
or mutates Doctor state.  Later, an external lease-holding runner can produce
the exact immutable receipts described here.  Until then every physical facet
scores zero, regardless of how complete the structural implementation is.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import sys
from typing import Any

import appendix_physical_evidence_gate
import doctor_v5_post120_acceleration_scaffold as post120
import doctor_v5_physical_adapter_registry as adapter_registry
import doctor_v5_physical_counter_barrier as counter_barrier
import doctor_v5_physical_result_authority as result_authority
import spec_reentry_scaffold


ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORT_ROOT = (
    ROOT / "reports" / "condense" / "doctor_v5_ultra" /
    "staged_acceleration" / "physical_ab_v1"
)
PLAN_PATH = REPORT_ROOT / "physical_ab_plan.json"
STATUS_PATH = REPORT_ROOT / "physical_ab_status.json"
PACKET_PATH = REPORT_ROOT / "physical_ab_evidence.json"
OBSERVER_PATH = (
    ROOT / "reports" / "condense" / "doctor_v5_ultra" /
    "post_120b" / "observer_state.json"
)
APPENDIX_PACKET_PATH = (
    ROOT / "reports" / "appendix" / "physical_release" /
    "physical_evidence_packet.json"
)
POST120_HANDOFF_PATH = post120.DEFAULT_HANDOFF

PLAN_SCHEMA = "hawking.doctor_v5_physical_ab_plan.v1"
PACKET_SCHEMA = "hawking.doctor_v5_physical_ab_evidence.v1"
FACET_SCHEMA = "hawking.doctor_v5_physical_ab_facet.v1"
BOUNDARY_SCHEMA = "hawking.doctor_v5_physical_ab_release_boundary.v1"
COUNTER_SCHEMA = "hawking.doctor_v5_physical_ab_counters.v1"
POST120_SCHEMA = "hawking.doctor_v5_post120_physical_qualification.v1"
SCORECARD_SCHEMA = "hawking.doctor_v5_physical_10of10_scorecard.v1"
STATUS_SCHEMA = "hawking.doctor_v5_physical_ab_status.v1"
EXECUTION_RECEIPT_SCHEMA = "hawking.doctor_v5_physical_ab_execution_receipt.v1"
EXECUTION_SCOPE_SCHEMA = "hawking.doctor_v5_physical_ab_execution_scope.v1"
SOURCE_UNIT_MANIFEST_SCHEMA = "hawking.doctor_v5_physical_source_unit_manifest.v1"
MULTI_EXECUTION_RECEIPT_SCHEMA = "hawking.doctor_v5_physical_ab_multi_execution_receipt.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

TIERS = ("3B", "7B", "14B", "32B", "72B")
RATES = ("0.1", "0.25", "0.33", "0.5", "0.55", "0.8", "1", "2", "3", "4")
BRANCHES = ("codec_control", "doctor_conditional", "doctor_full", "doctor_static")
THREADS = (8, 12, 16, 20)
MIN_REPEATS = 5
# Exact authoritative Doctor production policy uses decimal byte quantities:
# retain 150 GB free disk and a separate 64 GB phase-aware scratch reserve.
# Do not transpose these two gates or silently convert them to GiB.
MIN_DISK_RESERVE_BYTES = 150_000_000_000
MIN_SCRATCH_RESERVE_BYTES = 64_000_000_000
MIN_TOTAL_DISK_ADMISSION_BYTES = MIN_DISK_RESERVE_BYTES + MIN_SCRATCH_RESERVE_BYTES

# A physical program is trusted only after exact baseline/candidate executable,
# argv-manifest, launch-contract, and execution-scope hashes are sealed by the
# source-reviewed signed registry contract.  The in-memory maps remain empty
# while that default-off release artifact is missing, stale, partial, or
# unsigned; a generic caller manifest can never populate them.
PROGRAM_ADAPTER_REGISTRY: dict[str, dict[str, str]] = {}
POST120_PROGRAM_ADAPTER_REGISTRY: dict[tuple[str, str, str], dict[str, str]] = {}
PROGRAM_ADAPTER_REGISTRY_ERRORS: list[str] = [
    "signed physical program-adapter registry has not been loaded"
]

# Exactly ten objective gates.  One valid, physical, exact receipt contributes
# one point.  Structural code, plans, synthetic canaries, estimates, and prose
# contribute zero points.
FACETS = (
    "release_authority",
    "thread_profiles",
    "block_parallel",
    "ordered_overlap",
    "bounded_reuse",
    "ram_swap_recovery",
    "native_io_pgo",
    "disk_lifecycle",
    "full_stack_parity_ab",
    "post120_appendix_bindings",
)

SOURCE_PATHS = (
    "tools/condense/doctor_v5_physical_ab_controller.py",
    "tools/condense/doctor_v5_physical_ab_executor.py",
    "tools/condense/doctor_v5_physical_counter_barrier.py",
    "tools/condense/doctor_v5_physical_adapter_registry.py",
    "tools/condense/doctor_v5_physical_result_authority.py",
    "tools/condense/appendix_physical_counter_authority.py",
    "tools/condense/appendix_contract.py",
    "tools/condense/doctor_v5_single_device_benchmark.py",
    "tools/condense/doctor_v5_single_device_sprint_audit.py",
    "tools/condense/doctor_v5_block_parallel_config_matrix.py",
    "tools/condense/doctor_v5_elastic_phase_scheduler.py",
    "tools/condense/doctor_v5_aggressive_admission_policy.py",
    "tools/condense/doctor_v5_shared_preprocess_cache.py",
    "tools/condense/doctor_v5_qwen_shard_window.py",
    "tools/condense/doctor_v5_remaining_scratch_ledger.py",
    "tools/condense/doctor_v5_remaining_scratch_gate_adapter.py",
    "tools/condense/doctor_v5_post120_acceleration_scaffold.py",
    "tools/condense/appendix_physical_evidence_gate.py",
    "tools/condense/appendix_physical_counter_collector.py",
    "tools/condense/physical_counter_attestation.py",
    "tools/condense/doctor_frontier_worker.py",
    "vendor/strand-quant/tools/thread_profile_contract.py",
    "vendor/strand-quant/tools/native_build.py",
    "vendor/strand-quant/src/ordered_pipeline.rs",
    "vendor/strand-quant/src/native_io.rs",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _stamp(value: dict[str, Any], field: str) -> dict[str, Any]:
    stamped = copy.deepcopy(value)
    stamped.pop(field, None)
    stamped[field] = canonical_sha256(stamped)
    return stamped


def _hex(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _finite(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value)) and (value > 0 if positive else value >= 0)
    )


def _file_identity(path: pathlib.Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(fd)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
        if identity(before) != identity(after) or size != after.st_size:
            raise ValueError(f"file changed while hashing: {path}")
    finally:
        os.close(fd)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": size}


def _artifact_errors(value: Any, *, label: str, verify_files: bool) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}:
        return [f"{label} must contain exactly path/sha256/bytes"]
    errors: list[str] = []
    path = value.get("path")
    if not isinstance(path, str) or not pathlib.Path(path).is_absolute():
        errors.append(f"{label}.path must be absolute")
    if not _hex(value.get("sha256")):
        errors.append(f"{label}.sha256 is invalid")
    size = value.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        errors.append(f"{label}.bytes is invalid")
    if verify_files and not errors:
        try:
            if _file_identity(pathlib.Path(path)) != value:
                errors.append(f"{label} differs from immutable file")
        except (OSError, ValueError) as exc:
            errors.append(f"{label} cannot be verified: {exc}")
    return errors


def _source_manifest() -> dict[str, Any]:
    entries = []
    for relative in SOURCE_PATHS:
        identity = _file_identity(ROOT / relative)
        identity["path"] = relative
        entries.append(identity)
    return _stamp({
        "entries": entries,
        "paths": list(SOURCE_PATHS),
    }, "manifest_sha256")


def _cell_id(prefix: str, *parts: str | int) -> str:
    label = "/".join([prefix, *(str(part) for part in parts)])
    return f"{prefix}-{canonical_sha256(label)[:16]}"


def _executor_manifest() -> dict[str, Any]:
    """Bind the reviewed inherited-lease executor without admitting it now."""
    intended = ROOT / "tools" / "condense" / "doctor_v5_physical_ab_executor.py"
    available = intended.is_file() and not intended.is_symlink()
    runner_source = _file_identity(intended) if available else None
    if isinstance(runner_source, dict):
        runner_source["path"] = str(intended.relative_to(ROOT))
    rows = []
    for facet in FACETS:
        rows.append({
            "facet": facet,
            "implemented": available,
            "currently_admitted": False,
            "argv_contract": [
                "python3.12", str(intended), "execute",
                "--facet", facet,
                "--plan", str(PLAN_PATH),
                "--launch-contract", "<HASH_BOUND_LAUNCH_CONTRACT.json>",
                "--release-authority", "<OWNER_FREE_RELEASE_AUTHORITY.json>",
                "--baseline-program", "<HASH_BOUND_BASELINE_EXECUTABLE>",
                "--baseline-argv-manifest", "<FROZEN_BASELINE_ARGV.json>",
                "--candidate-program", "<HASH_BOUND_CANDIDATE_EXECUTABLE>",
                "--candidate-argv-manifest", "<FROZEN_CANDIDATE_ARGV.json>",
                "--input-manifest", "<FROZEN_REAL_ARTIFACT_INPUTS.json>",
                "--execution-scope", "<FROZEN_SEGMENT_MODEL_TIER_RATE_BRANCH_SOURCE_SCOPE.json>",
                "--collector-authority", "<DIRECT_COUNTER_COLLECTOR_AUTHORITY.json>",
                "--output", f"<IMMUTABLE_{facet.upper()}_RECEIPT.json>",
            ],
            "receipt_schema": "hawking.doctor_v5_physical_ab_execution_receipt.v1",
            "embedded_facet_receipt_schema": FACET_SCHEMA,
            "minimum_paired_repeats": MIN_REPEATS,
            "preconditions": [
                "Doctor final_interpretation_ready=true",
                "zero heavy owners rechecked under inherited shared lease",
                "normal memory pressure, nominal thermal state, healthy swap guard",
                "source, executable, input, output, owner, counter, and attestation paths frozen",
            ],
        })
    return {
        "trusted_executor_available": available,
        "intended_runner_path": str(intended),
        "intended_runner_exists": available,
        "runner_source": runner_source,
        "program_adapter_registry_contract": {
            "policy": adapter_registry.build_policy(),
            "default_envelope_path": str(adapter_registry.DEFAULT_ENVELOPE),
            "release_time_exact_artifact_binding": True,
            "controller_source_edit_required": False,
            "unsigned_or_partial_registry_accepted": False,
            "registry_grants_execution": False,
        },
        "execute_subcommand_exposed_by_controller": False,
        "commands_executable": False,
        "commands_executable_after_all_admission_gates": available,
        "commands": rows,
        "multi_facet_orchestrator": {
            "implemented": available,
            "currently_admitted": False,
            "requires_exact_facets": list(FACETS),
            "holds_one_inherited_shared_heavy_lease": True,
            "mints_one_common_release_boundary_after_all_arms": True,
            "manifest_schema": "hawking.doctor_v5_physical_ab_multi_launch_manifest.v1",
            "receipt_schema": MULTI_EXECUTION_RECEIPT_SCHEMA,
            "argv_contract": [
                "python3.12", str(intended), "execute-all",
                "--matrix-manifest", "<HASH_BOUND_10_FACET_MATRIX.json>",
                "--output", "<IMMUTABLE_COMMON_BOUNDARY_RECEIPT.json>",
            ],
        },
        "current_admission_blocker": (
            "the runner is default-off; Doctor final-ready, zero owners, inherited shared lease, "
            "direct-counter authority, exact launch bundle, and per-arm resource gates must all pass"
        ),
    }


def build_plan() -> dict[str, Any]:
    thread_cells = [
        {
            "id": _cell_id("thread", tier, rate, branch, threads),
            "tier": tier, "rate": rate, "branch": branch, "threads": threads,
            "physical_receipt_required": True,
        }
        for tier in TIERS for rate in RATES for branch in BRANCHES for threads in THREADS
    ]
    block_cells = [
        {
            "id": _cell_id("block", tier, rate, branch),
            "tier": tier, "rate": rate, "branch": branch,
            "serial_and_parallel_exact_receipts_required": True,
        }
        for tier in TIERS for rate in RATES for branch in BRANCHES
    ]
    plan = {
        "schema": PLAN_SCHEMA,
        "mode": "unbound-default-off-validation-only",
        "execution_capability": False,
        "activation_capability": False,
        "runtime_default_mutation_permitted": False,
        "heavy_lease_opened_by_controller": False,
        "source_manifest": _source_manifest(),
        "tiers": list(TIERS), "rates": list(RATES), "branches": list(BRANCHES),
        "thread_candidates": list(THREADS),
        "minimum_randomized_paired_repeats": MIN_REPEATS,
        "thread_profile_cells": thread_cells,
        "block_parallel_cells": block_cells,
        "counts": {
            "thread_profile_cells": len(thread_cells),
            "thread_profile_selections": len(TIERS) * len(RATES) * len(BRANCHES),
            "block_parallel_cells": len(block_cells),
            "facets": len(FACETS),
        },
        "facet_order": list(FACETS),
        "resource_contract": {
            "minimum_disk_reserve_bytes": MIN_DISK_RESERVE_BYTES,
            "minimum_phase_aware_scratch_reserve_bytes": MIN_SCRATCH_RESERVE_BYTES,
            "minimum_combined_disk_admission_bytes": MIN_TOTAL_DISK_ADMISSION_BYTES,
            "memory_pressure_required": "normal",
            "thermal_required": "nominal",
            "oom_permitted": False,
            "swap_growth_must_be_bounded_and_recovered": True,
        },
        "overlap_contract": {
            "phases": ["read", "rht", "encode", "write", "attest"],
            "serial_finalizers": 1,
            "maximum_prepared_shards": 1,
            "20_thread_encoder_exclusive": True,
            "stacked_admission_requires_isolated_and_aggregate_receipts": True,
        },
        "reuse_contract": {
            "maximum_cache_unit_bytes": 8 * 1024**3,
            "minimum_disk_reserve_bytes": MIN_TOTAL_DISK_ADMISSION_BYTES,
            "includes_phase_aware_scratch_reserve": True,
            "rate_or_branch_dependent_work_reused": False,
            "parent_source_deletion_permitted": False,
            "exact_serial_oracle_required": True,
        },
        "claim_contract": {
            "component_speedups_multiplied": False,
            "only_full_stack_paired_speedup_may_feed_eta": True,
            "sub120_receipt_applied_to_120b_or_appendix": False,
            "structural_or_synthetic_points": 0,
            "physical_score_denominator": 10,
        },
        "external_bindings": {
            "post120_handoff_path": str(POST120_HANDOFF_PATH),
            "appendix_physical_packet_path": str(APPENDIX_PACKET_PATH),
            "appendix_counter_executor_path": str(
                ROOT / "tools" / "condense" / "appendix_physical_counter_executor.py"
            ),
        },
        "executor_manifest": _executor_manifest(),
    }
    return _stamp(plan, "plan_sha256")


def refresh_program_adapter_registries(
    *, plan: dict[str, Any] | None = None,
    envelope_path: pathlib.Path = adapter_registry.DEFAULT_ENVELOPE,
) -> list[str]:
    """Load only a current, signed, exact release registry.

    The plan is deliberately independent of the mutable release envelope, so
    launch contracts can bind the stable source-reviewed plan before exact
    binaries exist.  Swapping or deleting the envelope clears both maps.
    """
    exact_plan = build_plan() if plan is None else plan
    sub120, post120_entries, errors = adapter_registry.load_registries(
        plan_sha256=exact_plan["plan_sha256"],
        source_manifest_sha256=exact_plan["source_manifest"]["manifest_sha256"],
        envelope_path=envelope_path,
    )
    PROGRAM_ADAPTER_REGISTRY.clear()
    POST120_PROGRAM_ADAPTER_REGISTRY.clear()
    if not errors:
        PROGRAM_ADAPTER_REGISTRY.update(sub120)
        POST120_PROGRAM_ADAPTER_REGISTRY.update(post120_entries)
    PROGRAM_ADAPTER_REGISTRY_ERRORS[:] = list(errors)
    return list(errors)


def _boundary_errors(value: Any, *, plan: dict[str, Any]) -> list[str]:
    if not isinstance(value, dict):
        return ["release boundary is missing"]
    expected = {
        "schema", "plan_sha256", "observer_state_sha256",
        "final_interpretation_ready", "active_heavy_owner_count",
        "owner_inventory", "shared_heavy_lease", "ram_swap_guard_receipt",
        "disk_lifecycle_receipt", "observed_at_unix_ns", "attestation_sha256",
    }
    errors: list[str] = []
    if set(value) != expected:
        errors.append("release boundary fields are incomplete or unexpected")
    if value.get("schema") != BOUNDARY_SCHEMA or value.get("plan_sha256") != plan["plan_sha256"]:
        errors.append("release boundary schema/plan binding is invalid")
    if value.get("final_interpretation_ready") is not True or not _hex(value.get("observer_state_sha256")):
        errors.append("release boundary lacks final-ready observer authority")
    if value.get("active_heavy_owner_count") != 0:
        errors.append("release boundary is not owner-free")
    for field in ("owner_inventory", "ram_swap_guard_receipt", "disk_lifecycle_receipt"):
        errors.extend(_artifact_errors(value.get(field), label=f"release_boundary.{field}", verify_files=False))
    lease = value.get("shared_heavy_lease")
    if not isinstance(lease, dict) or set(lease) != {
        "lock_file", "held", "inherited_descriptor", "owners_rechecked_under_lock",
        "acquired_at_unix_ns", "released_at_unix_ns",
    }:
        errors.append("release boundary shared-heavy-lease evidence is malformed")
    else:
        errors.extend(_artifact_errors(lease.get("lock_file"), label="shared lease lock", verify_files=False))
        if lease.get("held") is not True or lease.get("inherited_descriptor") is not True \
                or lease.get("owners_rechecked_under_lock") is not True:
            errors.append("shared heavy lease was not inherited/held/rechecked")
        start, end = lease.get("acquired_at_unix_ns"), lease.get("released_at_unix_ns")
        if not isinstance(start, int) or isinstance(start, bool) or start <= 0 \
                or not isinstance(end, int) or isinstance(end, bool) or end <= start:
            errors.append("shared heavy lease interval is invalid")
    observed = value.get("observed_at_unix_ns")
    if not isinstance(observed, int) or isinstance(observed, bool) or observed <= 0:
        errors.append("release boundary observation time is invalid")
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("attestation_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append("release boundary attestation hash mismatch")
    return errors


def _counter_errors(value: Any, *, facet: str, nonce: str) -> list[str]:
    if not isinstance(value, dict):
        return ["physical counter payload is missing"]
    expected = {
        "schema", "facet", "run_nonce", "energy_j", "cpu_time_ns",
        "read_bytes", "write_bytes", "peak_rss_bytes", "sample_count",
        "directly_measured", "estimated", "counter_payload_sha256",
    }
    errors: list[str] = []
    if set(value) != expected:
        errors.append("physical counter payload fields are incomplete or unexpected")
    if value.get("schema") != COUNTER_SCHEMA or value.get("facet") != facet \
            or value.get("run_nonce") != nonce:
        errors.append("physical counters are not bound to facet/run nonce")
    for field in ("energy_j", "cpu_time_ns", "peak_rss_bytes"):
        if not _finite(value.get(field), positive=True):
            errors.append(f"physical counter {field} is invalid")
    for field in ("read_bytes", "write_bytes"):
        if not _finite(value.get(field)):
            errors.append(f"physical counter {field} is invalid")
    count = value.get("sample_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        errors.append("physical counter sample_count is invalid")
    if value.get("directly_measured") is not True or value.get("estimated") is not False:
        errors.append("physical counters must be direct, not estimated")
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("counter_payload_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append("physical counter payload hash mismatch")
    return errors


def _load_bound_json(
    artifact: Any, *, label: str, verify_files: bool,
) -> tuple[Any | None, list[str]]:
    errors = _artifact_errors(artifact, label=label, verify_files=verify_files)
    if errors or not verify_files or not isinstance(artifact, dict):
        return None, errors
    try:
        value = _load_optional(pathlib.Path(artifact["path"]))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return None, [f"{label} cannot be loaded safely: {exc}"]
    if value is None:
        errors.append(f"{label} is absent")
    return value, errors


def _owner_snapshot_errors(
    value: Any, *, plan: dict[str, Any], contract_sha256: str, facet: str,
    role: str, repeat: int, nonce: str, position: str, verify_files: bool,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{role}:{repeat} {position} owner snapshot is absent"]
    expected = {
        "schema", "plan_sha256", "contract_sha256", "facet", "phase", "role",
        "repeat", "run_nonce", "position", "observed_at_unix_ns", "ps_program",
        "shared_heavy_lease", "owners", "owner_count", "probe_ok", "synthetic",
        "snapshot_sha256",
    }
    errors: list[str] = []
    if set(value) != expected \
            or value.get("schema") != "hawking.doctor_v5_physical_ab_owner_snapshot.v1":
        errors.append(f"{role}:{repeat} {position} owner snapshot schema is invalid")
    comparisons = {
        "plan_sha256": plan["plan_sha256"], "contract_sha256": contract_sha256,
        "facet": facet, "phase": "measured", "role": role, "repeat": repeat,
        "run_nonce": nonce, "position": position,
    }
    if any(value.get(field) != expected_value for field, expected_value in comparisons.items()):
        errors.append(f"{role}:{repeat} {position} owner snapshot binding differs")
    if not isinstance(value.get("observed_at_unix_ns"), int) \
            or isinstance(value.get("observed_at_unix_ns"), bool) \
            or value.get("observed_at_unix_ns", 0) <= 0:
        errors.append(f"{role}:{repeat} {position} owner observation time is invalid")
    errors.extend(_artifact_errors(
        value.get("ps_program"), label=f"{role}:{repeat} {position} ps program",
        verify_files=verify_files,
    ))
    lease = value.get("shared_heavy_lease")
    if not isinstance(lease, dict) or set(lease) != {
        "path", "st_dev", "st_ino", "inherited_descriptor", "held",
    } or lease.get("path") != str(
        ROOT / "reports" / "cron" / "studio_heavy.lock"
    ) or lease.get("inherited_descriptor") is not True or lease.get("held") is not True \
            or not isinstance(lease.get("st_dev"), int) \
            or not isinstance(lease.get("st_ino"), int):
        errors.append(f"{role}:{repeat} {position} owner snapshot lease is invalid")
    if value.get("owners") != [] or value.get("owner_count") != 0 \
            or value.get("probe_ok") is not True or value.get("synthetic") is not False:
        errors.append(f"{role}:{repeat} {position} owner snapshot is opaque/synthetic/nonzero")
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("snapshot_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append(f"{role}:{repeat} {position} owner snapshot hash mismatch")
    return errors


def _resource_guard_errors(
    value: Any, *, plan: dict[str, Any], contract_sha256: str, facet: str,
    role: str, repeat: int, nonce: str, position: str,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{role}:{repeat} {position} resource guard is absent"]
    expected = {
        "schema", "plan_sha256", "contract_sha256", "facet", "phase", "role",
        "repeat", "run_nonce", "position", "observed_at_unix_ns", "limits",
        "snapshot", "health_errors", "healthy", "synthetic", "receipt_sha256",
    }
    errors: list[str] = []
    if set(value) != expected \
            or value.get("schema") != "hawking.doctor_v5_physical_ab_resource_guard.v1":
        errors.append(f"{role}:{repeat} {position} resource guard schema is invalid")
    comparisons = {
        "plan_sha256": plan["plan_sha256"], "contract_sha256": contract_sha256,
        "facet": facet, "phase": "measured", "role": role, "repeat": repeat,
        "run_nonce": nonce, "position": position,
    }
    if any(value.get(field) != expected_value for field, expected_value in comparisons.items()):
        errors.append(f"{role}:{repeat} {position} resource guard binding differs")
    limits, snapshot = value.get("limits"), value.get("snapshot")
    if not isinstance(limits, dict) or not isinstance(snapshot, dict):
        errors.append(f"{role}:{repeat} {position} resource guard payload is incomplete")
    else:
        if limits.get("minimum_disk_free_bytes", 0) < MIN_TOTAL_DISK_ADMISSION_BYTES:
            errors.append(f"{role}:{repeat} {position} resource guard weakens disk+scratch")
        swap_ceiling = limits.get("maximum_swap_used_bytes")
        swap = snapshot.get("swap_used_bytes")
        if not _finite(swap) or not _finite(swap_ceiling) or swap > swap_ceiling:
            errors.append(f"{role}:{repeat} {position} resource guard swap is invalid")
        if snapshot.get("probe_ok") is not True or snapshot.get("pressure_level") != 1 \
                or snapshot.get("thermal_state") != 0 \
                or snapshot.get("power_source") != "AC Power" \
                or not isinstance(snapshot.get("disk_free_bytes"), int) \
                or snapshot.get("disk_free_bytes", 0) < MIN_TOTAL_DISK_ADMISSION_BYTES:
            errors.append(f"{role}:{repeat} {position} resource sample is not green")
    if value.get("health_errors") != [] or value.get("healthy") is not True \
            or value.get("synthetic") is not False:
        errors.append(f"{role}:{repeat} {position} resource guard is opaque/synthetic/red")
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("receipt_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append(f"{role}:{repeat} {position} resource guard hash mismatch")
    return errors


def _scientific_receipt_errors(
    value: Any, *, plan: dict[str, Any], facet: str, adapter: Any,
    input_manifest: Any, output_manifest: Any, facet_payload: Any,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{facet} scientific receipt is absent or not JSON"]
    expected = {
        "schema", "plan_sha256", "facet", "adapter_id", "input_manifest_sha256",
        "output_sha256", "facet_payload_sha256", "exact_output", "skipped",
        "negative_evidence_preserved", "synthetic", "receipt_sha256",
    }
    errors: list[str] = []
    if set(value) != expected \
            or value.get("schema") \
            != "hawking.doctor_v5_physical_ab_scientific_receipt.v1":
        errors.append(f"{facet} scientific receipt schema is invalid")
    bindings = {
        "plan_sha256": plan["plan_sha256"], "facet": facet,
        "adapter_id": adapter.get("adapter_id") if isinstance(adapter, dict) else None,
        "input_manifest_sha256": input_manifest.get("sha256")
        if isinstance(input_manifest, dict) else None,
        "output_sha256": output_manifest.get("sha256")
        if isinstance(output_manifest, dict) else None,
        "facet_payload_sha256": facet_payload.get("sha256")
        if isinstance(facet_payload, dict) else None,
    }
    if any(value.get(field) != expected_value for field, expected_value in bindings.items()):
        errors.append(f"{facet} scientific receipt artifact bindings differ")
    if value.get("exact_output") is not True or value.get("skipped") is not False \
            or value.get("negative_evidence_preserved") is not True \
            or value.get("synthetic") is not False:
        errors.append(f"{facet} scientific receipt is skipped/synthetic/inexact/incomplete")
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("receipt_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append(f"{facet} scientific receipt hash mismatch")
    return errors


def _arm_evidence_errors(
    value: Any, artifact: Any, *, run: dict[str, Any], plan: dict[str, Any],
    facet: str, role: str, repeat: int, nonce: str, verify_files: bool,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{role}:{repeat} executor arm evidence is absent"]
    expected = {
        "schema", "plan_sha256", "contract_sha256", "facet", "phase", "role",
        "repeat", "run_nonce", "execution_scope_sha256",
        "orchestration_group_sha256", "program_adapter", "program", "benchmark_runner",
        "invocation_manifest", "environment_manifest", "input_manifest",
        "output_manifest", "scientific_receipt", "owner_inventory_before",
        "owner_inventory_after", "resource_guard_before", "resource_guard_after",
        "stdout", "stderr", "collector_stdout", "collector_stderr",
        "collector_request", "collector_ready", "collector_stop", "arm_started",
        "counter_payload", "counter_attestation", "facet_payload", "resource_before",
        "resource_after", "direct_counter_validated", "shell_used",
        "ambient_environment_inherited",
        "live_doctor_mutated", "runtime_defaults_changed", "source_files_deleted",
        "synthetic", "sidecars_sha256",
    }
    errors: list[str] = []
    if set(value) != expected \
            or value.get("schema") != "hawking.doctor_v5_physical_ab_arm_sidecars.v1":
        errors.append(f"{role}:{repeat} executor arm evidence schema is invalid")
    if value.get("plan_sha256") != plan["plan_sha256"] \
            or not _hex(value.get("contract_sha256")) \
            or value.get("facet") != facet or value.get("phase") != "measured" \
            or value.get("role") != role or value.get("repeat") != repeat \
            or value.get("run_nonce") != nonce:
        errors.append(f"{role}:{repeat} executor arm identity binding differs")
    if not _hex(value.get("execution_scope_sha256")) \
            or not _hex(value.get("orchestration_group_sha256")):
        errors.append(f"{role}:{repeat} execution scope/group binding is absent")
    # The top-level ten-point Doctor scorecard is the sub-120B population.
    # Post-120B adapters are accepted only by the segment-specific execution
    # receipt validators below; allowing any same-facet adapter here would let
    # a correctly signed but wrong-scope run leak across evidence domains.
    expected_adapter = PROGRAM_ADAPTER_REGISTRY.get(facet)
    adapter = value.get("program_adapter")
    if not isinstance(adapter, dict) or adapter != expected_adapter \
            or adapter.get("execution_scope_sha256") \
            != value.get("execution_scope_sha256"):
        errors.append(
            f"{role}:{repeat} lacks a source-reviewed concrete sub-120B {facet} adapter"
        )
    for field in (
        "program", "benchmark_runner", "invocation_manifest", "environment_manifest",
        "input_manifest", "output_manifest", "scientific_receipt",
        "owner_inventory_before", "owner_inventory_after", "counter_attestation",
    ):
        if value.get(field) != run.get(field):
            errors.append(f"{role}:{repeat} arm evidence {field} differs from run")
    for field in (
        "stdout", "stderr", "collector_stdout", "collector_stderr", "collector_request",
        "collector_ready", "collector_stop", "arm_started", "counter_payload",
        "facet_payload", "resource_guard_before", "resource_guard_after",
        "resource_before", "resource_after",
    ):
        errors.extend(_artifact_errors(
            value.get(field), label=f"{role}:{repeat} evidence.{field}",
            verify_files=verify_files,
        ))
    if value.get("resource_before") != value.get("resource_guard_before") \
            or value.get("resource_after") != value.get("resource_guard_after"):
        errors.append(f"{role}:{repeat} resource guard aliases differ")
    if value.get("direct_counter_validated") is not True \
            or value.get("shell_used") is not False \
            or value.get("ambient_environment_inherited") is not False \
            or value.get("live_doctor_mutated") is not False \
            or value.get("runtime_defaults_changed") is not False \
            or value.get("source_files_deleted") is not False \
            or value.get("synthetic") is not False:
        errors.append(f"{role}:{repeat} arm evidence weakens physical/no-mutation policy")
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("sidecars_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append(f"{role}:{repeat} arm evidence hash mismatch")
    bound, bound_errors = _load_bound_json(
        artifact, label=f"{role}:{repeat} arm evidence artifact", verify_files=verify_files,
    )
    errors.extend(bound_errors)
    if verify_files and bound != value:
        errors.append(f"{role}:{repeat} inline arm evidence differs from immutable artifact")
    owner_values: dict[str, Any] = {}
    resource_values: dict[str, Any] = {}
    for position in ("before", "after"):
        owner, owner_errors = _load_bound_json(
            value.get(f"owner_inventory_{position}"),
            label=f"{role}:{repeat} {position} owner receipt", verify_files=verify_files,
        )
        errors.extend(owner_errors)
        if owner is not None:
            errors.extend(_owner_snapshot_errors(
                owner, plan=plan, contract_sha256=value.get("contract_sha256", ""),
                facet=facet, role=role, repeat=repeat, nonce=nonce,
                position=position, verify_files=verify_files,
            ))
            owner_values[position] = owner
        resource, resource_errors = _load_bound_json(
            value.get(f"resource_guard_{position}"),
            label=f"{role}:{repeat} {position} resource receipt", verify_files=verify_files,
        )
        errors.extend(resource_errors)
        if resource is not None:
            errors.extend(_resource_guard_errors(
                resource, plan=plan, contract_sha256=value.get("contract_sha256", ""),
                facet=facet, role=role, repeat=repeat, nonce=nonce, position=position,
            ))
            resource_values[position] = resource
    if set(resource_values) == {"before", "after"}:
        before = resource_values["before"]
        after = resource_values["after"]
        before_swap = before.get("snapshot", {}).get("swap_used_bytes")
        after_swap = after.get("snapshot", {}).get("swap_used_bytes")
        growth_limit = before.get("limits", {}).get("maximum_swap_growth_bytes")
        if not _finite(before_swap) or not _finite(after_swap) or not _finite(growth_limit) \
                or after_swap - before_swap > growth_limit:
            errors.append(f"{role}:{repeat} measured swap growth exceeds its frozen bound")
    if verify_files:
        science, science_errors = _load_bound_json(
            value.get("scientific_receipt"),
            label=f"{role}:{repeat} scientific receipt", verify_files=True,
        )
        errors.extend(science_errors)
        if science is not None:
            errors.extend(_scientific_receipt_errors(
                science, plan=plan, facet=facet, adapter=adapter,
                input_manifest=value.get("input_manifest"),
                output_manifest=value.get("output_manifest"),
                facet_payload=value.get("facet_payload"),
            ))
        attestation, attestation_errors = _load_bound_json(
            value.get("counter_attestation"),
            label=f"{role}:{repeat} counter attestation", verify_files=True,
        )
        errors.extend(attestation_errors)
        if isinstance(attestation, dict):
            attestation_bindings = {
                "plan_sha256": plan["plan_sha256"],
                "contract_sha256": value.get("contract_sha256"),
                "counter_payload_sha256": run.get("counter_payload", {}).get(
                    "counter_payload_sha256"
                ),
                "output_sha256": run.get("output_manifest", {}).get("sha256"),
                "scientific_sha256": run.get("scientific_receipt", {}).get("sha256"),
                "stdout_sha256": value.get("stdout", {}).get("sha256"),
                "stderr_sha256": value.get("stderr", {}).get("sha256"),
            }
            if any(
                attestation.get(field) != expected_value
                for field, expected_value in attestation_bindings.items()
            ):
                errors.append(f"{role}:{repeat} counter attestation artifact bindings differ")
            errors.extend(counter_barrier.validate_backend_result(
                run.get("counter_payload"), attestation, facet=facet, role=role,
                repeat=repeat, run_nonce=nonce,
                program_sha256=run.get("program", {}).get("sha256", ""),
                started_at_unix_ns=run.get("started_at_unix_ns", 0),
                ended_at_unix_ns=run.get("ended_at_unix_ns", 0),
                started_at_continuous_ns=attestation.get("execution_interval", {}).get(
                    "started_at_continuous_ns", 0
                ),
                ended_at_continuous_ns=attestation.get("execution_interval", {}).get(
                    "ended_at_continuous_ns", 0
                ),
            ))
    return errors


def _run_errors(
    value: Any, *, facet: str, role: str, repeat: int,
    boundary_sha: str, verify_files: bool, plan: dict[str, Any],
    lease_interval: tuple[int, int] | None = None,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{role}:{repeat} run is missing"]
    expected = {
        "role", "repeat", "run_nonce", "boundary_attestation_sha256",
        "program", "benchmark_runner", "invocation_manifest",
        "environment_manifest", "input_manifest", "output_manifest", "scientific_receipt",
        "counter_attestation", "owner_inventory_before", "owner_inventory_after",
        "invocation_sha256", "environment_sha256", "started_at_unix_ns",
        "ended_at_unix_ns", "exit_code", "skipped", "owner_count_before",
        "owner_count_after", "thermal_before", "thermal_after",
        "memory_pressure_before", "memory_pressure_after", "swap_before_mb",
        "swap_after_mb", "disk_free_before_bytes", "disk_free_after_bytes",
        "counter_payload", "counter_attestation_binding_sha256",
        "exact_output_sha256", "executor_arm_evidence",
        "executor_arm_evidence_artifact", "run_sha256",
    }
    errors: list[str] = []
    if set(value) != expected:
        errors.append(f"{role}:{repeat} run fields are incomplete or unexpected")
    nonce = value.get("run_nonce")
    if value.get("role") != role or value.get("repeat") != repeat \
            or not _hex(nonce) or value.get("boundary_attestation_sha256") != boundary_sha:
        errors.append(f"{role}:{repeat} run identity/boundary binding is invalid")
    for field in (
        "program", "benchmark_runner", "invocation_manifest", "environment_manifest",
        "input_manifest", "output_manifest", "scientific_receipt",
        "counter_attestation", "owner_inventory_before", "owner_inventory_after",
    ):
        errors.extend(_artifact_errors(
            value.get(field), label=f"{role}:{repeat}.{field}", verify_files=verify_files,
        ))
    for field in ("invocation_sha256", "environment_sha256", "exact_output_sha256"):
        if not _hex(value.get(field)):
            errors.append(f"{role}:{repeat}.{field} is invalid")
    if isinstance(value.get("invocation_manifest"), dict) \
            and value.get("invocation_sha256") != value["invocation_manifest"].get("sha256"):
        errors.append(f"{role}:{repeat} invocation hash differs from immutable manifest")
    if isinstance(value.get("environment_manifest"), dict) \
            and value.get("environment_sha256") != value["environment_manifest"].get("sha256"):
        errors.append(f"{role}:{repeat} environment hash differs from immutable manifest")
    if isinstance(value.get("output_manifest"), dict) \
            and value.get("exact_output_sha256") != value["output_manifest"].get("sha256"):
        errors.append(f"{role}:{repeat} exact output hash differs from output artifact")
    start, end = value.get("started_at_unix_ns"), value.get("ended_at_unix_ns")
    if not isinstance(start, int) or isinstance(start, bool) or start <= 0 \
            or not isinstance(end, int) or isinstance(end, bool) or end <= start:
        errors.append(f"{role}:{repeat} execution interval is invalid")
    elif lease_interval is not None and not (
        lease_interval[0] <= start < end <= lease_interval[1]
    ):
        errors.append(f"{role}:{repeat} execution escaped the shared-lease interval")
    if value.get("exit_code") != 0 or value.get("skipped") is not False:
        errors.append(f"{role}:{repeat} was failed or skipped")
    if value.get("owner_count_before") != 0 or value.get("owner_count_after") != 0:
        errors.append(f"{role}:{repeat} was not owner-free")
    if value.get("thermal_before") != "nominal" or value.get("thermal_after") != "nominal" \
            or value.get("memory_pressure_before") != "normal" \
            or value.get("memory_pressure_after") != "normal":
        errors.append(f"{role}:{repeat} thermal/pressure envelope is not nominal")
    for field in ("swap_before_mb", "swap_after_mb"):
        if not _finite(value.get(field)):
            errors.append(f"{role}:{repeat}.{field} is invalid")
    for field in ("disk_free_before_bytes", "disk_free_after_bytes"):
        number = value.get(field)
        if not isinstance(number, int) or isinstance(number, bool) \
                or number < MIN_TOTAL_DISK_ADMISSION_BYTES:
            errors.append(
                f"{role}:{repeat}.{field} violates the disk reserve plus phase-scratch admission"
            )
    errors.extend(_counter_errors(value.get("counter_payload"), facet=facet, nonce=nonce))
    counter = value.get("counter_payload") if isinstance(value.get("counter_payload"), dict) else {}
    attestation = value.get("counter_attestation") if isinstance(value.get("counter_attestation"), dict) else {}
    expected_counter_binding = canonical_sha256({
        "counter_payload_sha256": counter.get("counter_payload_sha256"),
        "counter_attestation_sha256": attestation.get("sha256"),
        "run_nonce": nonce,
        "program_sha256": value.get("program", {}).get("sha256")
        if isinstance(value.get("program"), dict) else None,
        "input_manifest_sha256": value.get("input_manifest", {}).get("sha256")
        if isinstance(value.get("input_manifest"), dict) else None,
        "output_manifest_sha256": value.get("output_manifest", {}).get("sha256")
        if isinstance(value.get("output_manifest"), dict) else None,
    })
    if value.get("counter_attestation_binding_sha256") != expected_counter_binding:
        errors.append(f"{role}:{repeat} counter attestation is not bound to run artifacts")
    errors.extend(_arm_evidence_errors(
        value.get("executor_arm_evidence"), value.get("executor_arm_evidence_artifact"),
        run=value, plan=plan, facet=facet, role=role, repeat=repeat, nonce=nonce,
        verify_files=verify_files,
    ))
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("run_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append(f"{role}:{repeat} run hash mismatch")
    return errors


def _domain_payload_errors(facet: str, payload: Any, *, plan: dict[str, Any]) -> list[str]:
    if not isinstance(payload, dict):
        return [f"{facet} payload is missing"]
    errors: list[str] = []
    if facet == "release_authority":
        if payload != {
            "final_ready_rechecked_each_run": True,
            "zero_owners_rechecked_each_run": True,
            "shared_lease_continuous": True,
            "guard_sampled_each_run": True,
        }:
            errors.append("release-authority payload is incomplete")
    elif facet == "thread_profiles":
        rows = payload.get("measurements")
        selections = payload.get("selections")
        expected = {row["id"] for row in plan["thread_profile_cells"]}
        observed = {row.get("cell_id") for row in rows if isinstance(row, dict)} if isinstance(rows, list) else set()
        if observed != expected or len(rows or []) != len(expected):
            errors.append("thread profiles lack exact 8/12/16/20 tier/rate/branch coverage")
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) or row.get("exact_output") is not True
            or not _finite(row.get("wall_seconds"), positive=True)
            or not _hex(row.get("receipt_sha256")) for row in rows
        ):
            errors.append("thread-profile physical measurements are invalid")
        selection_keys = {
            (row.get("tier"), row.get("rate"), row.get("branch"))
            for row in selections if isinstance(row, dict)
        } if isinstance(selections, list) else set()
        expected_keys = {(tier, rate, branch) for tier in TIERS for rate in RATES for branch in BRANCHES}
        if selection_keys != expected_keys or len(selections or []) != len(expected_keys) \
                or any(row.get("threads") not in THREADS or row.get("nearest_fallback") is not False
                       for row in selections or [] if isinstance(row, dict)):
            errors.append("thread-profile selections are incomplete or use fallback")
    elif facet == "block_parallel":
        rows = payload.get("cells")
        expected = {row["id"] for row in plan["block_parallel_cells"]}
        observed = {row.get("cell_id") for row in rows if isinstance(row, dict)} if isinstance(rows, list) else set()
        if observed != expected or len(rows or []) != len(expected) or any(
            row.get("serial_output_sha256") != row.get("parallel_output_sha256")
            or row.get("exact") is not True or row.get("skipped") != 0
            for row in rows or [] if isinstance(row, dict)
        ):
            errors.append("block-parallel exact physical matrix is incomplete")
    elif facet == "ordered_overlap":
        if payload.get("phase_order") != plan["overlap_contract"]["phases"] \
                or payload.get("isolated_receipts_green") is not True \
                or payload.get("aggregate_receipt_green") is not True \
                or payload.get("serial_finalizers") != 1 \
                or payload.get("maximum_prepared_shards") != 1 \
                or payload.get("20_thread_encoder_exclusive") is not True \
                or payload.get("full_stack_speedup_only") is not True:
            errors.append("ordered-overlap physical envelope is incomplete")
    elif facet == "bounded_reuse":
        if not isinstance(payload.get("maximum_cache_unit_bytes"), int) \
                or payload["maximum_cache_unit_bytes"] > 8 * 1024**3 \
                or payload.get("minimum_disk_reserve_bytes", 0) \
                < MIN_TOTAL_DISK_ADMISSION_BYTES \
                or payload.get("rate_or_branch_dependent_work_reused") is not False \
                or payload.get("all_serial_oracles_exact") is not True \
                or payload.get("unique_outputs") is not True \
                or payload.get("refcounts_exact_before_gc") is not True \
                or payload.get("parent_sources_retained") is not True:
            errors.append("bounded-reuse physical lifecycle/parity is incomplete")
    elif facet == "ram_swap_recovery":
        states = payload.get("observed_states")
        if states != ["green", "soft", "hard", "emergency", "hard", "soft", "green"] \
                or payload.get("oom_events") != 0 or payload.get("exact_after_recovery") is not True \
                or payload.get("phase_aware_scratch_reserve_bytes", 0) < MIN_SCRATCH_RESERVE_BYTES \
                or payload.get("cooldowns_and_green_streaks_observed") is not True \
                or payload.get("swap_growth_recovered") is not True:
            errors.append("RAM/swap shock and recovery receipt is incomplete")
    elif facet == "native_io_pgo":
        required = {
            "instrumented_build_receipt_sha256", "training_receipt_sha256",
            "merge_receipt_sha256", "use_build_receipt_sha256",
            "llvm_profdata_identity_sha256", "training_corpus_sha256",
        }
        if any(not _hex(payload.get(field)) for field in required) \
                or payload.get("mmap_input_exercised") is not True \
                or payload.get("preallocated_output_exercised") is not True \
                or payload.get("pgo_generated_merged_used") is not True \
                or payload.get("exact_output_and_receipt_parity") is not True \
                or payload.get("source_identity_stable") is not True:
            errors.append("native I/O and PGO physical chain is incomplete")
    elif facet == "disk_lifecycle":
        if payload.get("minimum_disk_free_bytes", 0) < MIN_TOTAL_DISK_ADMISSION_BYTES \
                or payload.get("exclusive_partial_creation") is not True \
                or payload.get("atomic_finalization") is not True \
                or payload.get("source_identity_before_after_exact") is not True \
                or payload.get("parent_source_deleted") is not False \
                or payload.get("gc_only_ephemeral_refcount_zero") is not True \
                or payload.get("crash_resume_and_rollback_exact") is not True:
            errors.append("disk/lifecycle physical receipt is incomplete")
    elif facet == "full_stack_parity_ab":
        ratios = payload.get("paired_speedups")
        if not isinstance(ratios, list) or len(ratios) < MIN_REPEATS \
                or any(not _finite(row, positive=True) for row in ratios) \
                or payload.get("conservative_speedup") != min(ratios or [0]) \
                or payload.get("all_required_components_exercised") is not True \
                or payload.get("all_outputs_exact") is not True \
                or payload.get("all_scientific_receipts_exact") is not True \
                or payload.get("component_speedups_multiplied") is not False \
                or payload.get("eta_segment") != "sub-120b-doctor":
            errors.append("full-stack paired physical A/B receipt is incomplete")
    elif facet == "post120_appendix_bindings":
        if not _hex(payload.get("post120_qualification_sha256")) \
                or not _hex(payload.get("appendix_gate_sha256")) \
                or payload.get("sub120_receipts_reused") is not False \
                or payload.get("segment_specific_physical_receipts") is not True:
            errors.append("post-120B/Appendix segment bindings are incomplete")
    else:
        errors.append(f"unknown physical facet: {facet}")
    return errors


def _facet_errors(
    value: Any, *, facet: str, plan: dict[str, Any], boundary_sha: str,
    verify_files: bool, lease_interval: tuple[int, int] | None = None,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{facet} physical receipt is absent"]
    expected = {
        "schema", "facet", "status", "scope", "structural_only",
        "plan_sha256", "source_manifest_sha256", "boundary_attestation_sha256",
        "paired_protocol", "runs", "payload", "runtime_defaults_changed",
        "source_files_deleted", "completed_evidence_mutated",
        "component_speedups_multiplied", "receipt_sha256",
    }
    errors: list[str] = []
    if set(value) != expected:
        errors.append(f"{facet} receipt fields are incomplete or unexpected")
    if value.get("schema") != FACET_SCHEMA or value.get("facet") != facet \
            or value.get("status") != "pass" \
            or value.get("scope") != "physical-owner-free-real-artifact" \
            or value.get("structural_only") is not False:
        errors.append(f"{facet} is not a passing physical real-artifact receipt")
    if value.get("plan_sha256") != plan["plan_sha256"] \
            or value.get("source_manifest_sha256") != plan["source_manifest"]["manifest_sha256"] \
            or value.get("boundary_attestation_sha256") != boundary_sha:
        errors.append(f"{facet} plan/source/release binding differs")
    protocol = value.get("paired_protocol")
    if not isinstance(protocol, dict) or set(protocol) != {
        "warmups_per_arm", "repeats_per_arm", "randomized_interleaved",
        "order", "order_sha256",
    }:
        errors.append(f"{facet} paired protocol is malformed")
        repeats = 0
        order = []
    else:
        repeats = protocol.get("repeats_per_arm")
        order = protocol.get("order")
        expected_labels = {
            f"{role}:{index}" for role in ("baseline", "candidate")
            for index in range(repeats if isinstance(repeats, int) else 0)
        }
        if not isinstance(protocol.get("warmups_per_arm"), int) \
                or protocol["warmups_per_arm"] < 1 \
                or not isinstance(repeats, int) or isinstance(repeats, bool) \
                or repeats < MIN_REPEATS \
                or protocol.get("randomized_interleaved") is not True \
                or not isinstance(order, list) or set(order) != expected_labels \
                or len(order) != len(expected_labels) or order == sorted(order) \
                or protocol.get("order_sha256") != canonical_sha256(order):
            errors.append(f"{facet} paired order is incomplete or not randomized")
    runs = value.get("runs")
    by_key = {
        (row.get("role"), row.get("repeat")): row
        for row in runs if isinstance(row, dict)
    } if isinstance(runs, list) else {}
    if not isinstance(runs, list) or len(runs) != repeats * 2 or len(by_key) != repeats * 2:
        errors.append(f"{facet} paired run coverage is incomplete")
    for index in range(repeats if isinstance(repeats, int) else 0):
        baseline = by_key.get(("baseline", index))
        candidate = by_key.get(("candidate", index))
        errors.extend(_run_errors(
            baseline, facet=facet, role="baseline", repeat=index,
            boundary_sha=boundary_sha, verify_files=verify_files,
            plan=plan,
            lease_interval=lease_interval,
        ))
        errors.extend(_run_errors(
            candidate, facet=facet, role="candidate", repeat=index,
            boundary_sha=boundary_sha, verify_files=verify_files,
            plan=plan,
            lease_interval=lease_interval,
        ))
        if isinstance(baseline, dict) and isinstance(candidate, dict):
            if baseline.get("input_manifest") != candidate.get("input_manifest") \
                    or baseline.get("exact_output_sha256") != candidate.get("exact_output_sha256") \
                    or baseline.get("scientific_receipt", {}).get("sha256") \
                    != candidate.get("scientific_receipt", {}).get("sha256"):
                errors.append(f"{facet} pair {index} input/output/scientific parity differs")
    if isinstance(runs, list) and len(runs) == repeats * 2 and isinstance(order, list):
        observed_order = [
            f"{row.get('role')}:{row.get('repeat')}"
            for row in sorted(
                (row for row in runs if isinstance(row, dict)),
                key=lambda row: row.get("started_at_unix_ns", -1),
            )
        ]
        if observed_order != order:
            errors.append(f"{facet} recorded run order differs from execution timestamps")
    if facet == "full_stack_parity_ab" and all(
        isinstance(by_key.get((role, index)), dict)
        for index in range(repeats if isinstance(repeats, int) else 0)
        for role in ("baseline", "candidate")
    ):
        derived = []
        for index in range(repeats):
            baseline = by_key[("baseline", index)]
            candidate = by_key[("candidate", index)]
            base_ns = baseline.get("ended_at_unix_ns", 0) - baseline.get("started_at_unix_ns", 0)
            candidate_ns = candidate.get("ended_at_unix_ns", 0) - candidate.get("started_at_unix_ns", 0)
            if base_ns <= 0 or candidate_ns <= 0:
                derived = []
                break
            derived.append(base_ns / candidate_ns)
        claimed_ratios = value.get("payload", {}).get("paired_speedups")
        if len(derived) != repeats or not isinstance(claimed_ratios, list) \
                or len(claimed_ratios) != repeats \
                or any(not math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
                       for left, right in zip(derived, claimed_ratios)):
            errors.append("full-stack paired speedups are not derived from bound run intervals")
    errors.extend(_domain_payload_errors(facet, value.get("payload"), plan=plan))
    if value.get("runtime_defaults_changed") is not False \
            or value.get("source_files_deleted") is not False \
            or value.get("completed_evidence_mutated") is not False \
            or value.get("component_speedups_multiplied") is not False:
        errors.append(f"{facet} weakened lifecycle or speedup isolation")
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("receipt_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append(f"{facet} receipt hash mismatch")
    return errors


def _source_unit_manifest_errors(
    artifact: Any, *, segment: str, model: str, tier: str,
    source_units: int, verify_files: bool,
) -> list[str]:
    """Validate the exact, hash-bound source-unit population for a run scope."""
    value, errors = _load_bound_json(
        artifact, label=f"{segment}/{model} source-unit manifest",
        verify_files=verify_files,
    )
    if not verify_files or not isinstance(value, dict):
        return errors
    expected = {
        "schema", "segment", "model", "tier", "units", "manifest_sha256",
    }
    if set(value) != expected or value.get("schema") != SOURCE_UNIT_MANIFEST_SCHEMA \
            or value.get("segment") != segment or value.get("model") != model \
            or value.get("tier") != tier:
        errors.append(f"{segment}/{model} source-unit manifest scope is invalid")
    units = value.get("units")
    if not isinstance(units, list) or len(units) != source_units:
        errors.append(f"{segment}/{model} source-unit manifest count is not exact")
        units = units if isinstance(units, list) else []
    ids: list[Any] = []
    hashes: list[Any] = []
    for index, row in enumerate(units):
        if not isinstance(row, dict) or set(row) != {"source_unit_id", "source_sha256"} \
                or not isinstance(row.get("source_unit_id"), str) \
                or not row.get("source_unit_id") or not _hex(row.get("source_sha256")):
            errors.append(f"{segment}/{model} source unit {index} is malformed")
            continue
        ids.append(row["source_unit_id"])
        hashes.append(row["source_sha256"])
    if len(set(ids)) != len(ids):
        errors.append(f"{segment}/{model} source-unit identifiers are reused")
    if len(set(zip(ids, hashes))) != len(ids):
        errors.append(f"{segment}/{model} source-unit identities are reused")
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("manifest_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append(f"{segment}/{model} source-unit manifest hash mismatch")
    return errors


def _physical_execution_scope_errors(
    value: Any, *, facet: str, verify_files: bool,
    segment: str | None = None, model: str | None = None,
    tier: str | None = None, parameter_scope: str | None = None,
    source_units: int | None = None, source_unit_manifest: Any | None = None,
    rates: list[str] | None = None, branches: list[str] | None = None,
    cells: int | None = None, jobs: int | None = None, skips: int = 0,
) -> list[str]:
    """Validate scope carried by the execution receipt, not by its wrapper."""
    if not isinstance(value, dict):
        return [f"{segment or 'unknown'}/{model or 'unknown'}/{facet} execution scope is absent"]
    expected = {
        "schema", "segment", "model", "tier", "parameter_scope", "facet",
        "source_units", "source_unit_manifest", "rates", "branches", "cells",
        "jobs", "skips", "scope_sha256",
    }
    errors: list[str] = []
    if set(value) != expected or value.get("schema") != EXECUTION_SCOPE_SCHEMA:
        errors.append(f"{segment or 'unknown'}/{model or 'unknown'}/{facet} execution scope schema is invalid")
    for field in ("segment", "model", "tier", "parameter_scope"):
        if not isinstance(value.get(field), str) or not value.get(field):
            errors.append(f"execution scope {field} is absent")
    comparisons = {
        "facet": facet, "segment": segment, "model": model, "tier": tier,
        "parameter_scope": parameter_scope, "source_units": source_units,
        "source_unit_manifest": source_unit_manifest, "rates": rates,
        "branches": branches, "cells": cells, "jobs": jobs, "skips": skips,
    }
    for field, expected_value in comparisons.items():
        if expected_value is not None and value.get(field) != expected_value:
            errors.append(f"execution scope {field} differs from its exact wrapper scope")
    observed_units = value.get("source_units")
    observed_rates = value.get("rates")
    observed_branches = value.get("branches")
    observed_cells = value.get("cells")
    observed_jobs = value.get("jobs")
    if not isinstance(observed_units, int) or isinstance(observed_units, bool) \
            or observed_units <= 0:
        errors.append("execution scope source-unit count is invalid")
    if not isinstance(observed_rates, list) or not observed_rates \
            or len(set(observed_rates)) != len(observed_rates) \
            or any(not isinstance(row, str) or not row for row in observed_rates):
        errors.append("execution scope rate population is invalid")
    if not isinstance(observed_branches, list) or not observed_branches \
            or len(set(observed_branches)) != len(observed_branches) \
            or any(not isinstance(row, str) or not row for row in observed_branches):
        errors.append("execution scope branch population is invalid")
    if isinstance(observed_rates, list) and isinstance(observed_branches, list) \
            and observed_cells != len(observed_rates) * len(observed_branches):
        errors.append("execution scope cell count is not rate x branch exact")
    if isinstance(observed_units, int) and isinstance(observed_cells, int) \
            and observed_jobs != observed_units * observed_cells:
        errors.append("execution scope job count is not source-unit x cell exact")
    if value.get("skips") != 0:
        errors.append("execution scope permits skipped work")
    if isinstance(observed_units, int) and observed_units > 0:
        errors.extend(_source_unit_manifest_errors(
            value.get("source_unit_manifest"),
            segment=value.get("segment", ""), model=value.get("model", ""),
            tier=value.get("tier", ""), source_units=observed_units,
            verify_files=verify_files,
        ))
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("scope_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append("execution scope hash mismatch")
    return list(dict.fromkeys(errors))


def _execution_receipt_scope_errors(
    value: Any, *, facet: str, segment: str, model: str, tier: str,
    parameter_scope: str, source_units: int, source_unit_manifest: Any,
    rates: list[str], branches: list[str], cells: int, jobs: int, skips: int,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{segment}/{model}/{facet} execution receipt is absent"]
    expected = {
        "schema", "plan_sha256", "source_manifest_sha256", "executor_source_sha256",
        "launch_contract_sha256", "release_authority_sha256",
        "collector_authority_sha256", "facet", "release_boundary", "facet_receipt",
        "execution_scope", "orchestration_group_sha256",
        "warmup_run_sha256", "measured_sidecars_sha256", "sidecar_root", "shell_used",
        "shared_heavy_lease_acquired_by_executor", "live_doctor_mutated",
        "completed_evidence_mutated", "runtime_defaults_changed", "source_files_deleted",
        "execution_receipt_sha256",
    }
    errors: list[str] = []
    adapter = POST120_PROGRAM_ADAPTER_REGISTRY.get((segment, model, facet))
    if not isinstance(adapter, dict):
        errors.append(
            f"{segment}/{model}/{facet} lacks a source-reviewed segment-specific adapter"
        )
    if set(value) != expected \
            or value.get("schema") != EXECUTION_RECEIPT_SCHEMA \
            or value.get("facet") != facet:
        errors.append(f"{segment}/{model}/{facet} execution receipt schema/scope is invalid")
    scope = value.get("execution_scope")
    errors.extend(_physical_execution_scope_errors(
        scope, facet=facet, segment=segment, model=model, tier=tier,
        parameter_scope=parameter_scope, source_units=source_units,
        source_unit_manifest=source_unit_manifest, rates=rates, branches=branches,
        cells=cells, jobs=jobs, skips=skips, verify_files=True,
    ))
    if not _hex(value.get("orchestration_group_sha256")):
        errors.append(f"{segment}/{model}/{facet} orchestration group binding is invalid")
    if isinstance(adapter, dict):
        required_adapter_fields = {
            "adapter_id", "baseline_program_sha256", "baseline_argv_manifest_sha256",
            "candidate_program_sha256", "candidate_argv_manifest_sha256",
            "launch_contract_sha256", "execution_scope_sha256",
            "scientific_receipt_schema", "scientific_validator",
        }
        if set(adapter) != required_adapter_fields \
                or adapter.get("launch_contract_sha256") \
                != value.get("launch_contract_sha256") \
                or adapter.get("execution_scope_sha256") \
                != (scope.get("scope_sha256") if isinstance(scope, dict) else None):
            errors.append(f"{segment}/{model}/{facet} differs from its source-reviewed adapter")
    if value.get("shell_used") is not False \
            or value.get("shared_heavy_lease_acquired_by_executor") is not False \
            or value.get("live_doctor_mutated") is not False \
            or value.get("completed_evidence_mutated") is not False \
            or value.get("runtime_defaults_changed") is not False \
            or value.get("source_files_deleted") is not False:
        errors.append(f"{segment}/{model}/{facet} execution receipt weakens isolation")
    receipt = value.get("facet_receipt")
    if not isinstance(receipt, dict) or receipt.get("facet") != facet \
            or receipt.get("scope") != "physical-owner-free-real-artifact" \
            or receipt.get("structural_only") is not False:
        errors.append(f"{segment}/{model}/{facet} embedded facet receipt is not physical")
    elif receipt.get("receipt_sha256") != canonical_sha256({
        key: item for key, item in receipt.items() if key != "receipt_sha256"
    }):
        errors.append(f"{segment}/{model}/{facet} embedded facet receipt hash mismatch")
    sidecars = value.get("measured_sidecars_sha256")
    if not isinstance(sidecars, list) or len(sidecars) < MIN_REPEATS * 2 \
            or any(not _hex(row) for row in sidecars) or len(set(sidecars)) != len(sidecars):
        errors.append(f"{segment}/{model}/{facet} measured sidecar coverage is invalid")
    if isinstance(receipt, dict) and isinstance(receipt.get("runs"), list):
        embedded = [
            row.get("executor_arm_evidence", {}).get("sidecars_sha256")
            for row in receipt["runs"] if isinstance(row, dict)
        ]
        if embedded != sidecars:
            errors.append(
                f"{segment}/{model}/{facet} execution receipt sidecars differ from embedded runs"
            )
        for index, run in enumerate(receipt["runs"]):
            if not isinstance(run, dict):
                errors.append(f"{segment}/{model}/{facet} run {index} is malformed")
                continue
            unstamped_run = copy.deepcopy(run)
            claimed_run = unstamped_run.pop("run_sha256", None)
            evidence = run.get("executor_arm_evidence")
            if not _hex(claimed_run) or claimed_run != canonical_sha256(unstamped_run) \
                    or not isinstance(evidence, dict) \
                    or evidence.get("schema") \
                    != "hawking.doctor_v5_physical_ab_arm_sidecars.v1" \
                    or evidence.get("synthetic") is not False \
                    or evidence.get("direct_counter_validated") is not True:
                errors.append(
                    f"{segment}/{model}/{facet} run {index} lacks exact executor evidence"
                )
            elif isinstance(adapter, dict):
                role = run.get("role")
                expected_program = adapter.get(f"{role}_program_sha256")
                if run.get("program", {}).get("sha256") != expected_program \
                        or evidence.get("program_adapter") != adapter \
                        or evidence.get("execution_scope_sha256") \
                        != (scope.get("scope_sha256") if isinstance(scope, dict) else None) \
                        or evidence.get("orchestration_group_sha256") \
                        != value.get("orchestration_group_sha256"):
                    errors.append(
                        f"{segment}/{model}/{facet} run {index} is not adapter-bound"
                    )
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("execution_receipt_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append(f"{segment}/{model}/{facet} execution receipt hash mismatch")
    return errors


def _execution_identity(
    value: dict[str, Any], *, artifact: Any, segment: str, model: str, facet: str,
) -> dict[str, Any]:
    """Extract every identity domain that must never cross a segment/model."""
    receipt = value.get("facet_receipt", {})
    runs = receipt.get("runs", []) if isinstance(receipt, dict) else []
    return {
        "segment": segment, "model": model, "facet": facet,
        "artifact_path": artifact.get("path") if isinstance(artifact, dict) else None,
        "artifact_sha256": artifact.get("sha256") if isinstance(artifact, dict) else None,
        "execution_receipt_sha256": value.get("execution_receipt_sha256"),
        "orchestration_group_sha256": value.get("orchestration_group_sha256"),
        "execution_scope_sha256": value.get("execution_scope", {}).get("scope_sha256"),
        "launch_contract_sha256": value.get("launch_contract_sha256"),
        "boundary_attestation_sha256": value.get("release_boundary", {}).get(
            "attestation_sha256"
        ),
        "run_nonces": [row.get("run_nonce") for row in runs if isinstance(row, dict)],
        "sidecars": list(value.get("measured_sidecars_sha256", []))
        if isinstance(value.get("measured_sidecars_sha256"), list) else [],
    }


def _global_execution_reuse_errors(rows: list[dict[str, Any]]) -> list[str]:
    """Reject receipt/run reuse across GPT-OSS and every higher-tier group."""
    errors: list[str] = []
    for field in (
        "artifact_path", "artifact_sha256", "execution_receipt_sha256",
        "execution_scope_sha256", "launch_contract_sha256",
    ):
        values = [row.get(field) for row in rows]
        if any(not isinstance(item, str) or not item for item in values) \
                or len(values) != len(set(values)):
            errors.append(f"post-120B physical executions reuse or omit {field}")
    nonces = [nonce for row in rows for nonce in row.get("run_nonces", [])]
    if any(not _hex(nonce) for nonce in nonces) or len(nonces) != len(set(nonces)):
        errors.append("post-120B physical executions reuse or omit run nonces")
    sidecars = [sidecar for row in rows for sidecar in row.get("sidecars", [])]
    if any(not _hex(sidecar) for sidecar in sidecars) \
            or len(sidecars) != len(set(sidecars)):
        errors.append("post-120B physical executions reuse or omit measured sidecars")
    group_scopes: dict[str, set[tuple[Any, Any, Any]]] = {}
    group_facets: dict[str, set[Any]] = {}
    for row in rows:
        group = row.get("orchestration_group_sha256")
        if not _hex(group):
            errors.append("post-120B execution lacks an orchestration-group identity")
            continue
        group_scopes.setdefault(group, set()).add((
            row.get("segment"), row.get("model"),
            row.get("boundary_attestation_sha256"),
        ))
        group_facets.setdefault(group, set()).add(row.get("facet"))
    if any(len(scopes) != 1 for scopes in group_scopes.values()):
        errors.append("an orchestration group is reused across segment/model/boundary scopes")
    if any(len(facets) != len(set(facets)) for facets in group_facets.values()):
        errors.append("an orchestration group reuses a facet")
    return list(dict.fromkeys(errors))


def _post120_segment_receipt_errors(
    artifact: Any, *, facet: str, verify_files: bool,
    execution_identities: list[dict[str, Any]] | None = None,
) -> list[str]:
    value, errors = _load_bound_json(
        artifact, label=f"post120.{facet}", verify_files=verify_files,
    )
    if not verify_files:
        return [*errors, f"post120.{facet} exact segment scope was not file-verified"]
    if not isinstance(value, dict):
        return errors
    expected = {
        "schema", "segment", "model", "tier", "facet", "source_units", "rates",
        "branches", "cells", "jobs", "skips", "source_unit_manifest",
        "status", "physical", "synthetic",
        "sub120_receipts_reused", "execution_receipt", "receipt_sha256",
    }
    if set(value) != expected \
            or value.get("schema") != "hawking.doctor_v5_post120_segment_facet_receipt.v1" \
            or value.get("segment") != "gpt-oss-120b" \
            or value.get("model") != "GPT-OSS" or value.get("tier") != "120B" \
            or value.get("facet") != facet:
        errors.append(f"post120.{facet} segment/tier/facet scope is invalid")
    if value.get("source_units") != 615 or value.get("rates") != list(RATES) \
            or value.get("branches") != list(BRANCHES) or value.get("cells") != 40 \
            or value.get("jobs") != 24600 or value.get("skips") != 0:
        errors.append(f"post120.{facet} exact 615x10x4 scope is incomplete")
    if value.get("status") != "physical-exact-qualified" \
            or value.get("physical") is not True or value.get("synthetic") is not False \
            or value.get("sub120_receipts_reused") is not False:
        errors.append(f"post120.{facet} is structural/synthetic/reused")
    execution, execution_errors = _load_bound_json(
        value.get("execution_receipt"), label=f"post120.{facet}.execution",
        verify_files=True,
    )
    errors.extend(execution_errors)
    errors.extend(_execution_receipt_scope_errors(
        execution, facet=facet, segment="gpt-oss-120b", model="GPT-OSS",
        tier="120B", parameter_scope="exactly-120B", source_units=615,
        source_unit_manifest=value.get("source_unit_manifest"), rates=list(RATES),
        branches=list(BRANCHES), cells=40, jobs=24600, skips=0,
    ))
    if isinstance(execution, dict) and execution_identities is not None:
        execution_identities.append(_execution_identity(
            execution, artifact=value.get("execution_receipt"),
            segment="gpt-oss-120b", model="GPT-OSS", facet=facet,
        ))
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("receipt_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append(f"post120.{facet} receipt hash mismatch")
    return errors


def _higher_tier_receipt_errors(
    artifact: Any, *, model: str, verify_files: bool,
    execution_identities: list[dict[str, Any]] | None = None,
) -> list[str]:
    value, errors = _load_bound_json(
        artifact, label=f"higher-tier.{model}", verify_files=verify_files,
    )
    if not verify_files:
        return [*errors, f"higher-tier.{model} exact scope was not file-verified"]
    if not isinstance(value, dict):
        return errors
    expected = {
        "schema", "segment", "model", "tier", "parameter_scope", "source_units",
        "source_unit_manifest", "rates", "branches", "cells", "jobs", "skips",
        "facets", "facet_execution_receipts", "status", "physical", "synthetic",
        "sub120_receipts_reused", "receipt_sha256",
    }
    if set(value) != expected \
            or value.get("schema") != "hawking.doctor_v5_higher_tier_physical_receipt.v1" \
            or value.get("segment") != "post-120b-higher-tier" \
            or value.get("model") != model \
            or not isinstance(value.get("tier"), str) or not value.get("tier") \
            or value.get("parameter_scope") != "strictly-greater-than-120B":
        errors.append(f"higher-tier.{model} model/tier scope is invalid")
    source_units = value.get("source_units")
    if not isinstance(source_units, int) or isinstance(source_units, bool) \
            or source_units <= 0 or value.get("cells") != 40 \
            or value.get("jobs") != source_units * 40 if isinstance(source_units, int) else True \
            or value.get("skips") != 0:
        errors.append(f"higher-tier.{model} source-unit/cell/job scope is invalid")
    if value.get("rates") != list(RATES) or value.get("branches") != list(BRANCHES) \
            or value.get("facets") != list(FACETS):
        errors.append(f"higher-tier.{model} rate/branch/facet scope is incomplete")
    if value.get("status") != "physical-exact-qualified" \
            or value.get("physical") is not True or value.get("synthetic") is not False \
            or value.get("sub120_receipts_reused") is not False:
        errors.append(f"higher-tier.{model} is structural/synthetic/reused")
    rows = value.get("facet_execution_receipts")
    if not isinstance(rows, dict) or set(rows) != set(FACETS):
        errors.append(f"higher-tier.{model} facet execution map is incomplete")
        rows = rows if isinstance(rows, dict) else {}
    hashes: list[Any] = []
    for facet in FACETS:
        execution, execution_errors = _load_bound_json(
            rows.get(facet), label=f"higher-tier.{model}.{facet}", verify_files=True,
        )
        errors.extend(execution_errors)
        errors.extend(_execution_receipt_scope_errors(
            execution, facet=facet, segment="post-120b-higher-tier", model=model,
            tier=value.get("tier", ""), parameter_scope="strictly-greater-than-120B",
            source_units=source_units if isinstance(source_units, int) else -1,
            source_unit_manifest=value.get("source_unit_manifest"), rates=list(RATES),
            branches=list(BRANCHES), cells=40,
            jobs=value.get("jobs") if isinstance(value.get("jobs"), int) else -1,
            skips=0,
        ))
        if isinstance(execution, dict) and execution_identities is not None:
            execution_identities.append(_execution_identity(
                execution, artifact=rows.get(facet),
                segment="post-120b-higher-tier", model=model, facet=facet,
            ))
        if isinstance(rows.get(facet), dict):
            hashes.append(rows[facet].get("sha256"))
    if len(set(hashes)) != len(FACETS):
        errors.append(f"higher-tier.{model} reuses facet execution receipts")
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("receipt_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append(f"higher-tier.{model} receipt hash mismatch")
    return errors


def _post120_errors(value: Any, *, handoff: Any, verify_files: bool) -> list[str]:
    if not isinstance(value, dict):
        return ["post-120B physical qualification is absent"]
    expected = {
        "schema", "post120_handoff_sha256", "status", "gptoss_coverage",
        "segment_facet_receipts", "higher_tier_receipts",
        "sub120_receipts_reused", "runtime_defaults_changed",
        "qualification_sha256",
    }
    errors: list[str] = []
    if set(value) != expected or value.get("schema") != POST120_SCHEMA \
            or value.get("status") != "physical-exact-qualified":
        errors.append("post-120B qualification schema/status is invalid")
    handoff_sha = handoff.get("handoff_sha256") if isinstance(handoff, dict) else None
    if value.get("post120_handoff_sha256") != handoff_sha:
        errors.append("post-120B qualification is not bound to structural handoff")
    coverage = value.get("gptoss_coverage")
    if coverage != {
        "source_units": 615, "rates": 10, "branches": 4,
        "cells": 40, "jobs": 24600, "skips": 0, "exact": True,
    }:
        errors.append("post-120B GPT-OSS physical coverage is incomplete")
    execution_identities: list[dict[str, Any]] = []
    rows = value.get("segment_facet_receipts")
    required = FACETS[:-1]
    if not isinstance(rows, dict) or set(rows) != set(required):
        errors.append("post-120B segment facet receipt map is incomplete")
    else:
        hashes = []
        for facet in required:
            errors.extend(_post120_segment_receipt_errors(
                rows.get(facet), facet=facet, verify_files=verify_files,
                execution_identities=execution_identities,
            ))
            if isinstance(rows.get(facet), dict):
                hashes.append(rows[facet].get("sha256"))
        if len(set(hashes)) != len(required):
            errors.append("post-120B facet receipts are reused")
    horizons = value.get("higher_tier_receipts")
    expected_horizons = {"DeepSeek-V4-Flash", "Kimi-K2.6", "DeepSeek-V4-Pro"}
    if not isinstance(horizons, dict) or set(horizons) != expected_horizons:
        errors.append("higher-tier physical receipt coverage is incomplete")
    else:
        for name, artifact in horizons.items():
            errors.extend(_higher_tier_receipt_errors(
                artifact, model=name, verify_files=verify_files,
                execution_identities=execution_identities,
            ))
    if verify_files:
        errors.extend(_global_execution_reuse_errors(execution_identities))
    if value.get("sub120_receipts_reused") is not False \
            or value.get("runtime_defaults_changed") is not False:
        errors.append("post-120B qualification reused sub120 evidence or changed defaults")
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("qualification_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append("post-120B qualification hash mismatch")
    return errors


def _validate_core_packet(
    packet: Any, *, plan: dict[str, Any] | None = None,
    verify_files: bool = True,
) -> tuple[list[str], dict[str, list[str]]]:
    plan = build_plan() if plan is None else plan
    global_errors: list[str] = []
    facet_errors: dict[str, list[str]] = {facet: [] for facet in FACETS}
    if not isinstance(packet, dict):
        return ["physical A/B packet is absent"], {
            facet: [f"{facet} exact physical receipt is absent"] for facet in FACETS
        }
    expected = {
        "schema", "plan_sha256", "source_manifest", "release_boundary",
        "facet_receipts", "post120_handoff", "post120_qualification",
        "appendix_physical_packet", "runtime_defaults_changed",
        "activation_requested", "component_speedups_multiplied", "packet_sha256",
    }
    if set(packet) != expected or packet.get("schema") != PACKET_SCHEMA \
            or packet.get("plan_sha256") != plan["plan_sha256"]:
        global_errors.append("physical A/B packet schema/plan binding is invalid")
    unstamped = copy.deepcopy(packet)
    claimed = unstamped.pop("packet_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        global_errors.append("physical A/B packet hash mismatch")
    if packet.get("source_manifest") != plan["source_manifest"]:
        global_errors.append("physical A/B source manifest differs from plan")
    boundary = packet.get("release_boundary")
    boundary_errors = _boundary_errors(boundary, plan=plan)
    global_errors.extend(boundary_errors)
    boundary_sha = boundary.get("attestation_sha256", "") if isinstance(boundary, dict) else ""
    lease = boundary.get("shared_heavy_lease", {}) if isinstance(boundary, dict) else {}
    lease_interval = (
        (lease.get("acquired_at_unix_ns"), lease.get("released_at_unix_ns"))
        if isinstance(lease, dict)
        and isinstance(lease.get("acquired_at_unix_ns"), int)
        and isinstance(lease.get("released_at_unix_ns"), int)
        else None
    )
    receipts = packet.get("facet_receipts")
    if not isinstance(receipts, dict) or set(receipts) != set(FACETS):
        global_errors.append("physical facet receipt map is incomplete")
        receipts = receipts if isinstance(receipts, dict) else {}
    for facet in FACETS:
        facet_errors[facet].extend(boundary_errors)
        facet_errors[facet].extend(_facet_errors(
            receipts.get(facet), facet=facet, plan=plan,
            boundary_sha=boundary_sha, verify_files=verify_files,
            lease_interval=lease_interval,
        ))
    handoff = packet.get("post120_handoff")
    handoff_errors = post120.validate_handoff_packet(
        handoff, verify_files=verify_files,
    ) if isinstance(handoff, dict) else ["post-120B structural handoff is absent"]
    facet_errors["post120_appendix_bindings"].extend(handoff_errors)
    post_errors = _post120_errors(
        packet.get("post120_qualification"), handoff=handoff,
        verify_files=verify_files,
    )
    facet_errors["post120_appendix_bindings"].extend(post_errors)
    appendix = packet.get("appendix_physical_packet")
    appendix_errors = appendix_physical_evidence_gate.validate_gate(
        appendix, verify_counter_files=verify_files,
    )
    facet_errors["post120_appendix_bindings"].extend(appendix_errors)
    segment_receipt = receipts.get("post120_appendix_bindings")
    segment_payload = segment_receipt.get("payload", {}) if isinstance(segment_receipt, dict) else {}
    if isinstance(packet.get("post120_qualification"), dict) \
            and segment_payload.get("post120_qualification_sha256") \
            != packet["post120_qualification"].get("qualification_sha256"):
        facet_errors["post120_appendix_bindings"].append(
            "segment receipt does not bind post-120B qualification"
        )
    if isinstance(appendix, dict) and segment_payload.get("appendix_gate_sha256") \
            != appendix.get("gate_sha256"):
        facet_errors["post120_appendix_bindings"].append(
            "segment receipt does not bind Appendix aggregate gate"
        )
    if packet.get("runtime_defaults_changed") is not False \
            or packet.get("activation_requested") is not False \
            or packet.get("component_speedups_multiplied") is not False:
        global_errors.append("physical packet weakened default-off or speedup isolation")
    return global_errors, facet_errors


def validate_packet(
    packet: Any, *, plan: dict[str, Any] | None = None,
    verify_files: bool = True,
) -> tuple[list[str], dict[str, list[str]]]:
    """Accept only an operator-signed, unexpired aggregate evidence seal.

    The legacy raw/self-hashed packet remains useful as the core generated by
    the physical runner, but it is never scoreable on its own.  Only the
    source-pinned result authority may unwrap that core for the existing deep
    facet validator.
    """
    plan = build_plan() if plan is None else plan
    core, seal_errors = result_authority.validate_and_unwrap(
        packet, verify_files=verify_files,
    )
    if seal_errors or core is None:
        errors = [
            "operator-signed sealed aggregate Doctor evidence is required",
            *(f"operator seal: {error}" for error in seal_errors),
        ]
        return list(dict.fromkeys(errors)), {
            facet: [f"{facet} is not admitted from raw or unverified aggregate evidence"]
            for facet in FACETS
        }
    return _validate_core_packet(core, plan=plan, verify_files=verify_files)


def build_scorecard(
    packet: Any | None, *, plan: dict[str, Any] | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    plan = build_plan() if plan is None else plan
    global_errors, errors = validate_packet(
        packet, plan=plan, verify_files=verify_files,
    )
    rows = []
    for facet in FACETS:
        facet_problems = [*global_errors, *errors[facet]]
        # Deduplicate without hiding exact failure text.
        facet_problems = list(dict.fromkeys(facet_problems))
        green = not facet_problems
        rows.append({
            "facet": facet,
            "green": green,
            "physical_points": 1 if green else 0,
            "structural_points": 0,
            "errors": facet_problems,
        })
    score = sum(row["physical_points"] for row in rows)
    admitted_core = packet.get("core_packet") \
        if not global_errors and isinstance(packet, dict) else None
    card = {
        "schema": SCORECARD_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "packet_sha256": admitted_core.get("packet_sha256")
        if isinstance(admitted_core, dict) else None,
        "facets": rows,
        "physical_score": score,
        "physical_score_denominator": 10,
        "physical_rating": f"{score}/10",
        "all_facets_green": score == 10,
        "structural_scaffolding_physical_score": 0,
        "component_speedups_multiplied": False,
        "eta_promotion_authorized": False,
        "runtime_default_activation_authorized": False,
    }
    return _stamp(card, "scorecard_sha256")


def _observer_errors(observer: Any) -> list[str]:
    if not isinstance(observer, dict):
        return ["Doctor observer state is absent"]
    errors = []
    unstamped = copy.deepcopy(observer)
    claimed = unstamped.pop("state_sha256", None)
    if observer.get("schema") != "hawking.doctor_v5_post_120b_observer_state.v1" \
            or not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append("Doctor observer schema/self-hash is invalid")
    if observer.get("final_interpretation_ready") is not True:
        errors.append("Doctor final_interpretation_ready is false")
    return errors


def build_status(
    *, observer: Any, active_owners: list[dict[str, Any]],
    packet: Any | None, verify_files: bool = True,
) -> dict[str, Any]:
    plan = build_plan()
    boundary_errors = _observer_errors(observer)
    if active_owners:
        boundary_errors.append(f"{len(active_owners)} heavy owner(s) remain")
    release_open = not boundary_errors
    evaluated = release_open and packet is not None
    card = build_scorecard(
        packet if evaluated else None, plan=plan,
        verify_files=verify_files if evaluated else False,
    )
    blockers = list(boundary_errors)
    if not release_open:
        blockers.append("physical packet is not loaded before final-ready zero-owner release")
    elif packet is None:
        blockers.append("physical A/B evidence packet is absent")
    if plan["executor_manifest"]["trusted_executor_available"] is not True:
        blockers.append(
            "trusted physical executor is absent; only non-executable command contracts exist"
        )
    blockers.extend(
        f"{facet}: source-reviewed concrete physical program adapter is absent"
        for facet in FACETS if facet not in PROGRAM_ADAPTER_REGISTRY
    )
    blockers.extend(PROGRAM_ADAPTER_REGISTRY_ERRORS)
    blockers.extend(
        f"{row['facet']}: {error}"
        for row in card["facets"] for error in row["errors"]
        if evaluated
    )
    status = {
        "schema": STATUS_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "mode": "status-only-default-off",
        "release_boundary_open": release_open,
        "physical_packet_evaluated": evaluated,
        "execution_ready": False,
        "execution_capability": False,
        "activation_ready": False,
        "runtime_defaults_changed": False,
        "scorecard": card,
        "blockers": list(dict.fromkeys(blockers)),
    }
    return _stamp(status, "status_sha256")


def build_dry_run(status: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = build_plan()
    return _stamp({
        "schema": "hawking.doctor_v5_physical_ab_dry_run.v1",
        "plan_sha256": plan["plan_sha256"],
        "status_sha256": status.get("status_sha256") if isinstance(status, dict) else None,
        "would_execute": False,
        "commands": [],
        "heavy_lease_acquired": False,
        "models_opened": False,
        "gpu_used": False,
        "cargo_invoked": False,
        "live_doctor_mutated": False,
        "runtime_defaults_changed": False,
        "future_physical_work": {
            "thread_profile_cells": plan["counts"]["thread_profile_cells"],
            "block_parallel_cells": plan["counts"]["block_parallel_cells"],
            "randomized_paired_repeats_per_facet": MIN_REPEATS,
            "facets": list(FACETS),
        },
        "future_command_contracts": plan["executor_manifest"]["commands"],
        "reason": "validation-only controller exposes no physical launch surface",
    }, "dry_run_sha256")


def _load_optional(path: pathlib.Path) -> Any | None:
    try:
        # The result-authority reader holds an O_NOFOLLOW descriptor, checks a
        # stable single-link identity, and rejects duplicate/non-finite JSON.
        # Use the same parser for observer state and for the sealed aggregate
        # so no alternate pathname or ambiguous JSON semantics reach scoring.
        return result_authority._safe_json(path)
    except FileNotFoundError:
        return None


def current_status() -> dict[str, Any]:
    refresh_program_adapter_registries()
    observer = _load_optional(OBSERVER_PATH)
    owners = spec_reentry_scaffold.active_heavy_owners()
    packet = None
    # Avoid even loading a future large physical packet while Doctor owns the
    # machine.  The validator rechecks the same release boundary inside it.
    if not owners and not _observer_errors(observer):
        packet = _load_optional(PACKET_PATH)
    return build_status(observer=observer, active_owners=owners, packet=packet)


def _atomic_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _selftest() -> int:
    plan = build_plan()
    assert plan["counts"]["thread_profile_cells"] == 800
    assert plan["counts"]["block_parallel_cells"] == 200
    card = build_scorecard(None, plan=plan, verify_files=False)
    assert card["physical_rating"] == "0/10"
    assert not card["all_facets_green"]
    assert build_dry_run()["commands"] == []
    print("doctor_v5_physical_ab_controller.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", type=pathlib.Path)
    parser.add_argument("--write-plan", type=pathlib.Path)
    parser.add_argument("--write-status", type=pathlib.Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.plan or args.write_plan is not None:
        value = build_plan()
        if args.write_plan is not None:
            _atomic_json(args.write_plan, value)
        else:
            print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.status or args.write_status is not None:
        value = current_status()
        if args.write_status is not None:
            _atomic_json(args.write_status, value)
        else:
            print(json.dumps(value, indent=2, sort_keys=True))
        return 0 if value["scorecard"]["all_facets_green"] else 75
    if args.dry_run:
        print(json.dumps(build_dry_run(current_status()), indent=2, sort_keys=True))
        return 75
    if args.validate is not None:
        packet = _load_optional(args.validate)
        card = build_scorecard(packet, verify_files=True)
        print(json.dumps(card, indent=2, sort_keys=True))
        return 0 if card["all_facets_green"] else 1
    parser.error("choose --plan, --status, --dry-run, --validate, --write-plan, --write-status, or --selftest")
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

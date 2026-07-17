#!/usr/bin/env python3.12
"""Fail-closed acceleration contracts for GPT-OSS 120B and later tiers.

This module is deliberately separate from the live Doctor queue.  It binds the
existing exact GPT-OSS 10-rate x 4-branch graph, and it can bind an arbitrary
validated >120B source/admission manifest, to the same single-host acceleration
ideology.  It never launches a worker, reads model payload bytes, writes a live
runtime spec, registers an adapter, or changes a runtime default.

The plans are aggressive *qualification* plans.  Every optimization remains
unselected until physical, exact-output evidence exists for the exact model,
rate, branch, executable, host, and source generation.  Estimates cannot open
the execution gate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import doctor_v5_gptoss_mxfp4 as mxfp4
import doctor_v5_gptoss_parallel_scaffold as parallel
import doctor_v5_gptoss_reuse_fanout as fanout
import doctor_v5_higher_tier_scaffold as higher
import doctor_v5_shared_preprocess_cache as shared_cache


REQUIREMENTS_SCHEMA = "hawking.doctor_v5_post120_acceleration_requirements.v1"
HORIZON_SCHEMA = "hawking.doctor_v5_post120_horizon_acceleration.v1"
GPTOSS_PLAN_SCHEMA = "hawking.doctor_v5_gptoss_acceleration_plan.v1"
HIGHER_PLAN_SCHEMA = "hawking.doctor_v5_higher_tier_acceleration_plan.v1"
HANDOFF_SCHEMA = "hawking.doctor_v5_post120_acceleration_handoff.v1"
PHYSICAL_PLAN_SCHEMA = "hawking.doctor_v5_physical_ab_plan.v1"
DEFAULT_ROOT = ROOT / "reports/condense/doctor_v5_unbound/post120_acceleration"
DEFAULT_REQUIREMENTS = DEFAULT_ROOT / "requirements.json"
DEFAULT_HORIZONS = DEFAULT_ROOT / "named_horizons.json"
DEFAULT_GPTOSS_PLAN = DEFAULT_ROOT / "gptoss_120b_acceleration_plan.json"
DEFAULT_HANDOFF = DEFAULT_ROOT / "handoff.json"
# Keep post-120B preparation strictly outside the live Doctor tree.  This
# source-rebuilt plan is only an unbound qualification parent; the physical
# controller must still issue and verify its live release plan at the
# final-ready, zero-owner boundary.
DEFAULT_PHYSICAL_AB_PLAN = DEFAULT_ROOT / "physical_ab_plan.json"
DEFAULT_WORK_PLAN = parallel.DEFAULT_WORK_PLAN
DEFAULT_PENDING_WIRING = parallel.DEFAULT_PENDING_WIRING
DEFAULT_FANOUT_PLAN = fanout.DEFAULT_FANOUT_PLAN
SHA_RE = re.compile(r"[0-9a-f]{64}")
MAX_JSON_BYTES = 512 * 1024 * 1024
THREAD_CANDIDATES = (8, 12, 16, 20)
QUEUE_DEPTH_CANDIDATES = (2, 3, 4, 6)
BRANCHES = tuple(row[0] for row in parallel.BRANCHES)
RATES = tuple(parallel.RATES)
PHASES = ("read", "rht", "encode", "write", "attest")
FACETS = (
    "rate_thread_profiles",
    "block_parallelism",
    "ordered_phase_overlap",
    "bounded_preprocess_reuse",
    "ram_lane_packing",
    "controlled_swap",
    "disk_lifecycle_gc",
    "native_pgo",
    "metal_preprocess",
    "exact_quality_receipts",
    "rollback_cas",
)
PHYSICAL_AB_FACETS = (
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
HORIZONS = (
    ("DeepSeek-V4-Flash", "284B", "local_or_range_streamed"),
    ("Kimi-K2.6", "1.1T", "range_streamed"),
    ("DeepSeek-V4-Pro", "1.6T", "remote_range_streamed"),
)
HANDOFF_SEMANTIC_FIELDS = {
    "gptoss_work_plan": "work_plan_sha256",
    "gptoss_pending_wiring": "pending_wiring_sha256",
    "gptoss_reuse_fanout": "fanout_plan_sha256",
    "shared_preprocess_requirements": "requirements_sha256",
    "shared_preprocess_manifest": "manifest_sha256",
    "shared_preprocess_plan": "cache_plan_sha256",
    "tokenizer_binding": "tokenizer_binding_sha256",
    "tokenizer_gate": "gate_sha256",
    "higher_tier_requirements": "requirements_sha256",
    "acceleration_requirements": "requirements_sha256",
    "named_horizons": "horizon_scaffold_sha256",
    "gptoss_acceleration_plan": "acceleration_plan_sha256",
    "physical_ab_plan": "plan_sha256",
}
DEFAULT_TOKENIZER_GATE = parallel.DEFAULT_OUTPUT_ROOT / "tokenizer_gate.json"
DEFAULT_TOKENIZER_BINDING = parallel.DEFAULT_OUTPUT_ROOT / "tokenizer_binding.json"
DEFAULT_SHARED_REQUIREMENTS = shared_cache.DEFAULT_REQUIREMENTS
DEFAULT_SHARED_MANIFEST = shared_cache.DEFAULT_GPTOSS_MANIFEST
DEFAULT_SHARED_PLAN = shared_cache.DEFAULT_GPTOSS_PLAN
DEFAULT_HIGHER_REQUIREMENTS = higher.DEFAULT_REQUIREMENTS


class AccelerationScaffoldError(RuntimeError):
    """An unbound acceleration authority is incomplete or inconsistent."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(doc: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in doc.items() if key != field}


def _with_hash(doc: dict[str, Any], field: str) -> dict[str, Any]:
    if field in doc:
        raise AccelerationScaffoldError(f"refusing to replace {field}")
    doc[field] = _hash_value(doc)
    return doc


def _read_bytes(path: Path, maximum: int = MAX_JSON_BYTES) -> bytes:
    path = Path(path).resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise AccelerationScaffoldError(f"cannot open {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise AccelerationScaffoldError(f"invalid or oversized file: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise AccelerationScaffoldError(f"short read: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise AccelerationScaffoldError(f"file grew during read: {path}")
        after = os.fstat(fd)
    finally:
        os.close(fd)
    current = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) \
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns
            ) or stat.S_ISLNK(current.st_mode):
        raise AccelerationScaffoldError(f"file identity changed during read: {path}")
    return b"".join(chunks)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_bytes(path).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AccelerationScaffoldError(f"cannot decode JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AccelerationScaffoldError(f"JSON root is not an object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    raw = _read_bytes(path)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mxfp4._atomic_json(Path(path), doc)


def _tool_bindings() -> list[dict[str, Any]]:
    paths = (
        HERE / "doctor_v5_post120_acceleration_scaffold.py",
        HERE / "doctor_v5_block_parallel_config_matrix.py",
        HERE / "doctor_v5_elastic_phase_scheduler.py",
        HERE / "doctor_v5_inert_phase_launcher.py",
        HERE / "doctor_v5_aggressive_admission_policy.py",
        HERE / "doctor_v5_remaining_scratch_ledger.py",
        HERE / "doctor_v5_higher_tier_scaffold.py",
        HERE / "doctor_v5_higher_tier_authority.py",
        HERE / "doctor_v5_physical_ab_controller.py",
        HERE / "doctor_v5_physical_ab_executor.py",
        HERE / "doctor_v5_physical_counter_barrier.py",
        ROOT / "vendor/strand-quant/src/ordered_pipeline.rs",
        ROOT / "vendor/strand-quant/src/native_io.rs",
        ROOT / "vendor/strand-quant/src/metal_rht_probe.rs",
    )
    return [_artifact(path) for path in paths]


def _matrix_cells(prefix: str, *, adapter_ids: bool) -> list[dict[str, Any]]:
    authority = {branch: (command, adapter) for branch, command, adapter
                 in parallel.BRANCHES}
    cells: list[dict[str, Any]] = []
    for rate in RATES:
        for branch in BRANCHES:
            command, adapter = authority[branch]
            cells.append({
                "cell_template_id": f"{prefix}/rate={rate}/branch={branch}",
                "rate_id": rate, "branch": branch, "command": command,
                "adapter_id": adapter if adapter_ids else None,
                "status": "pending-unbound-qualification",
                "execution_permitted": False,
            })
    return sorted(cells, key=lambda row: row["cell_template_id"])


def _profiles(scope_sha256: str) -> list[dict[str, Any]]:
    return [
        {
            "rate_id": rate,
            "profile_authority_sha256": _hash_value({
                "scope_sha256": scope_sha256, "rate_id": rate,
                "thread_candidates": THREAD_CANDIDATES, "phases": PHASES,
            }),
            "thread_candidates": list(THREAD_CANDIDATES),
            "selected_threads": None,
            "selected_phase_threads": None,
            "selected_lane_cap": None,
            "calibration": {
                "status": "missing-physical-exact-output-receipts",
                "minimum_repetitions_per_candidate": 3,
                "warmup_separate_from_measurement": True,
                "exact_artifact_and_attestation_parity_required": True,
                "same_source_rate_branch_and_host_required": True,
                "winner_must_be_measured_not_estimated": True,
            },
        }
        for rate in RATES
    ]


def _facets(scope_sha256: str) -> dict[str, Any]:
    common = {
        "status": "unqualified-default-off",
        "scope_sha256": scope_sha256,
        "physical_receipt_required": True,
        "simulation_can_activate": False,
        "quality_or_negative_outcome_claimed": False,
    }
    return {
        "rate_thread_profiles": {
            **common, "candidates": list(THREAD_CANDIDATES),
            "per_rate_and_phase_selection": True,
            "cross_rate_transfer_permitted": False,
        },
        "block_parallelism": {
            **common, "partition_authority": "source-unit-and-tensor-block-boundaries",
            "candidate_workers": [1, 2, 3, 4],
            "canonical_merge_order": "source-unit-then-tensor-then-block",
            "bit_exact_archive_and_attestation_parity_required": True,
            "duplicate_or_missing_block_action": "reject-generation",
        },
        "ordered_phase_overlap": {
            **common, "phase_order": list(PHASES),
            "queue_depth_candidates": list(QUEUE_DEPTH_CANDIDATES),
            "overlap_scope": "different-source-units-only",
            "per_unit_phase_order_may_change": False,
            "bounded_buffers_and_backpressure_required": True,
            "writer_fsync_before_attestation_publish": True,
        },
        "bounded_preprocess_reuse": {
            **common, "maximum_consumers_per_source_traversal": 40,
            "scope": "same-source-unit-exact-canonical-values",
            "source_range_receipt_required": True,
            "scientific_evidence_reuse_permitted": False,
            "output_artifact_reuse_permitted": False,
            "reuse_equivalence_parity_required": True,
        },
        "ram_lane_packing": {
            **common, "per_phase_measured_high_water_required": True,
            "first_fit_decreasing_is_provisional_only": True,
            "generation_bound_resource_claim_required": True,
            "claim_must_reach_and_be_acknowledged_by_target_before_heavy_work": True,
            "soft_pause_and_hard_stop_receipts_required": True,
            "oom_is_terminal_qualification_failure": True,
        },
        "controlled_swap": {
            **common, "owner_free_non_ratcheting_baseline_required": True,
            "swap_growth_and_growth_rate_both_guarded": True,
            "deliberate_unbounded_swap_permitted": False,
            "new_launches_stop_before_active_generation_termination": True,
            "recovery_requires_normal_pressure_thermal_and_nonrising_swap": True,
        },
        "disk_lifecycle_gc": {
            **common, "minimum_free_disk_reserve_bytes": 50_000_000_000,
            "phase_aware_remaining_scratch_required": True,
            "disk_admission_before_every_launch": True,
            "hash_fsync_reporter_seal_successor_binding_before_ephemeral_gc": True,
            "source_or_parent_deletion_permitted": False,
            "whole_artifact_bytes_amortized_across_rates": False,
        },
        "native_pgo": {
            **common, "native_io_default_enabled": False,
            "pgo_default_enabled": False,
            "cpu_toolchain_binary_and_profile_corpus_hashes_required": True,
            "profile_training_and_measurement_sets_must_be_disjoint": True,
            "exact_output_and_attestation_parity_required": True,
            "per_rate_regression_floor_required": True,
        },
        "metal_preprocess": {
            **common, "default_enabled": False,
            "scope": "preprocess-rht-only-until-separate-runtime-parity",
            "same_artifact_cpu_reference_required": True,
            "stored_compact_hashed_computed_path_parity_required": True,
            "device_identity_kernel_hash_gpu_time_energy_and_bytes_required": True,
            "zero_skip_required": True,
            "fallback": "cpu-source-bound-path",
        },
        "exact_quality_receipts": {
            **common, "exact_ten_rate_four_branch_coverage_required": True,
            "tokenizer_prompt_corpus_chat_template_and_model_hashes_required": True,
            "same_examples_and_scoring_contract_required": True,
            "zero_skip_and_zero_synthetic_outcome_required": True,
            "branch_dependency_receipt_chain_required": True,
            "optimization_equivalence_and_scientific_quality_are_separate_gates": True,
        },
        "rollback_cas": {
            **common, "immutable_generation_and_parent_hashes_required": True,
            "start_and_completion_compare_and_swap_required": True,
            "descendant_process_and_resource_claim_proof_required": True,
            "sealed_pre_activation_rollback_point_required": True,
            "stale_partial_or_forged_receipt_action": "fail-closed",
            "live_mutation_from_this_plan_permitted": False,
        },
    }


def _promotion_gate() -> dict[str, Any]:
    return {
        "currently_permitted": False,
        "all_required": True,
        "facet_receipts": {facet: "missing" for facet in FACETS},
        "additional_requirements": [
            "source_manifest_and_parent_hashes_revalidated",
            "exact_10x4_matrix_preflighted",
            "all_adapters_reviewed_for_exact_architecture",
            "owner_free_baselines_and_rollback_point_sealed",
            "disk_and_lifecycle_admission_passed",
            "observer_structural_readiness_passed",
            "quiescent_generation_compare_and_swap_succeeded",
            "physical_ab_all_10_facets_green_for_the_exact_segment",
        ],
        "estimates_or_simulations_can_satisfy_gate": False,
    }


def _job_space_root(unit_ids: Iterable[str], cell_ids: Iterable[str]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for unit_id in sorted(unit_ids):
        for cell_id in sorted(cell_ids):
            raw = _canonical([unit_id, cell_id])
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
            count += 1
    return count, digest.hexdigest()


def build_requirements_packet(*, created_at: str | None = None) -> dict[str, Any]:
    scope = _hash_value({"rates": RATES, "branches": BRANCHES,
                         "facets": FACETS, "version": 1})
    doc: dict[str, Any] = {
        "schema": REQUIREMENTS_SCHEMA, "created_at": created_at or _now(),
        "status": "unbound-qualification-contract-only",
        "scope": {
            "gptoss_120b_exact_matrix": {"rates": 10, "branches": 4, "cells": 40},
            "logical_parameters_strictly_greater_than": 120_000_000_000,
            "named_horizons_are_identifiers_not_parameter_authorities": True,
        },
        "profile_template": _profiles(scope),
        "facets": _facets(scope),
        "controller_sources": _tool_bindings(),
        "promotion_gate": _promotion_gate(),
        "runtime_defaults_changed": False,
        "live_queue_or_registry_mutation_permitted": False,
        "source_or_parent_deletion_permitted": False,
        "execution_permitted": False,
        "quality_claims_permitted": False,
    }
    return _with_hash(doc, "requirements_sha256")


def validate_requirements(doc: Any) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != REQUIREMENTS_SCHEMA:
        return ["requirements schema mismatch"]
    errors: list[str] = []
    if doc.get("requirements_sha256") != _hash_value(_without(doc, "requirements_sha256")):
        errors.append("requirements hash mismatch")
    if doc.get("status") != "unbound-qualification-contract-only" \
            or doc.get("execution_permitted") is not False \
            or doc.get("quality_claims_permitted") is not False \
            or doc.get("runtime_defaults_changed") is not False \
            or doc.get("live_queue_or_registry_mutation_permitted") is not False \
            or doc.get("source_or_parent_deletion_permitted") is not False:
        errors.append("requirements cross the unbound boundary")
    facets = doc.get("facets")
    if not isinstance(facets, dict) or set(facets) != set(FACETS):
        errors.append("requirements facet inventory differs")
    elif any(row.get("status") != "unqualified-default-off"
             or row.get("physical_receipt_required") is not True
             or row.get("simulation_can_activate") is not False
             for row in facets.values() if isinstance(row, dict)):
        errors.append("one or more acceleration facets is not fail closed")
    profiles = doc.get("profile_template")
    if not isinstance(profiles, list) or [row.get("rate_id") for row in profiles] != list(RATES) \
            or any(row.get("thread_candidates") != list(THREAD_CANDIDATES)
                   or row.get("selected_threads") is not None for row in profiles):
        errors.append("per-rate thread profile lattice differs")
    gate = doc.get("promotion_gate")
    if not isinstance(gate, dict) or gate.get("currently_permitted") is not False \
            or gate.get("facet_receipts") != {facet: "missing" for facet in FACETS} \
            or gate.get("estimates_or_simulations_can_satisfy_gate") is not False:
        errors.append("requirements promotion gate is not closed")
    try:
        bindings = doc["controller_sources"]
        if not isinstance(bindings, list) or len(bindings) != len(_tool_bindings()):
            raise AccelerationScaffoldError("controller source set differs")
        for expected, observed in zip(_tool_bindings(), bindings):
            if expected != observed:
                raise AccelerationScaffoldError("controller source identity differs")
    except (KeyError, TypeError, OSError, AccelerationScaffoldError) as exc:
        errors.append(f"controller source verification failed: {exc}")
    try:
        expected = build_requirements_packet(created_at=doc.get("created_at"))
        if doc != expected:
            errors.append("requirements differ from the canonical acceleration policy")
    except (OSError, TypeError, AccelerationScaffoldError) as exc:
        errors.append(f"canonical requirements cannot be rebuilt: {exc}")
    return errors


def build_named_horizon_scaffold(*, requirements: dict[str, Any] | None = None,
                                 created_at: str | None = None) -> dict[str, Any]:
    requirements = requirements or build_requirements_packet(created_at=created_at)
    errors = validate_requirements(requirements)
    if errors:
        raise AccelerationScaffoldError("invalid acceleration requirements: "
                                        + "; ".join(errors))
    models: list[dict[str, Any]] = []
    for label, nominal, transport in HORIZONS:
        horizon_id = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
        cells = _matrix_cells(horizon_id, adapter_ids=False)
        models.append({
            "horizon_id": horizon_id, "label": label,
            "nominal_parameter_display_only": nominal,
            "logical_parameter_authority": None,
            "architecture_adapter": None, "source_manifest_sha256": None,
            "admission_plan_sha256": None, "transport_intent": transport,
            "cells": cells,
            "profiles": _profiles(_hash_value({"horizon_id": horizon_id,
                                                "requirements": requirements[
                                                    "requirements_sha256"]})),
            "status": "awaiting-exact-architecture-source-and-admission-authority",
            "execution_permitted": False,
        })
    doc: dict[str, Any] = {
        "schema": HORIZON_SCHEMA, "created_at": created_at or _now(),
        "status": "named-horizons-unbound-only",
        "requirements_sha256": requirements["requirements_sha256"],
        "models": models,
        "matrix": {"models": len(models), "rates_per_model": 10,
                   "branches_per_rate": 4, "cells_per_model": 40,
                   "total_cell_templates": sum(len(row["cells"]) for row in models)},
        "runtime_defaults_changed": False, "execution_permitted": False,
        "quality_claims_permitted": False,
    }
    return _with_hash(doc, "horizon_scaffold_sha256")


def validate_named_horizon_scaffold(doc: Any) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != HORIZON_SCHEMA:
        return ["horizon scaffold schema mismatch"]
    errors: list[str] = []
    if doc.get("horizon_scaffold_sha256") != _hash_value(
            _without(doc, "horizon_scaffold_sha256")):
        errors.append("horizon scaffold hash mismatch")
    if doc.get("status") != "named-horizons-unbound-only" \
            or doc.get("execution_permitted") is not False \
            or doc.get("quality_claims_permitted") is not False \
            or doc.get("runtime_defaults_changed") is not False:
        errors.append("named horizon scaffold is not fail closed")
    models = doc.get("models")
    if not isinstance(models, list) or len(models) != len(HORIZONS):
        errors.append("named horizon inventory differs")
        models = []
    expected_labels = [row[0] for row in HORIZONS]
    if [row.get("label") for row in models] != expected_labels:
        errors.append("named horizon order/identity differs")
    for model in models:
        cells = model.get("cells") if isinstance(model, dict) else None
        coverage = {(row.get("rate_id"), row.get("branch")) for row in cells
                    if isinstance(row, dict)} if isinstance(cells, list) else set()
        if not isinstance(cells, list) or len(cells) != 40 \
                or coverage != {(rate, branch) for rate in RATES for branch in BRANCHES} \
                or any(row.get("adapter_id") is not None
                       or row.get("execution_permitted") is not False for row in cells) \
                or model.get("logical_parameter_authority") is not None \
                or model.get("source_manifest_sha256") is not None \
                or model.get("admission_plan_sha256") is not None \
                or model.get("execution_permitted") is not False:
            errors.append(f"named horizon is prematurely bound: {model.get('label')}")
    for model, (label, nominal, transport) in zip(models, HORIZONS):
        horizon_id = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
        expected_cells = _matrix_cells(horizon_id, adapter_ids=False)
        expected_profiles = _profiles(_hash_value({
            "horizon_id": horizon_id,
            "requirements": doc.get("requirements_sha256"),
        }))
        expected_model = {
            "horizon_id": horizon_id, "label": label,
            "nominal_parameter_display_only": nominal,
            "logical_parameter_authority": None,
            "architecture_adapter": None, "source_manifest_sha256": None,
            "admission_plan_sha256": None, "transport_intent": transport,
            "cells": expected_cells, "profiles": expected_profiles,
            "status": "awaiting-exact-architecture-source-and-admission-authority",
            "execution_permitted": False,
        }
        if model != expected_model:
            errors.append(f"named horizon policy differs: {label}")
    if doc.get("matrix") != {"models": 3, "rates_per_model": 10,
                             "branches_per_rate": 4, "cells_per_model": 40,
                             "total_cell_templates": 120}:
        errors.append("named horizon matrix differs")
    return errors


def build_gptoss_acceleration_plan(
    work_plan: dict[str, Any], pending_wiring: dict[str, Any],
    fanout_plan: dict[str, Any], *, created_at: str | None = None,
) -> dict[str, Any]:
    errors = parallel.validate_work_plan(work_plan)
    errors += parallel.validate_pending_wiring(pending_wiring)
    errors += fanout.validate_fanout_plan(fanout_plan, work_plan, pending_wiring)
    if errors:
        raise AccelerationScaffoldError("invalid GPT-OSS parent: " + "; ".join(errors))
    if pending_wiring["work_plan"]["work_plan_sha256"] != work_plan["work_plan_sha256"] \
            or fanout_plan["work_plan_sha256"] != work_plan["work_plan_sha256"] \
            or fanout_plan["pending_wiring_sha256"] != pending_wiring[
                "pending_wiring_sha256"
            ]:
        raise AccelerationScaffoldError("GPT-OSS parent hashes do not close")
    cells = [{
        "cell_id": row["cell_id"], "cell_identity_sha256": row["cell_identity_sha256"],
        "cell_spec_sha256": row["cell_spec_sha256"], "rate_id": row["rate_id"],
        "branch": row["branch"], "command": row["command"],
        "adapter_id": row["adapter_id"], "execution_permitted": False,
    } for row in pending_wiring["cell_bindings"]]
    cells.sort(key=lambda row: row["cell_id"])
    source_ids = [row["unit_id"] for row in work_plan["source_units"]]
    job_count, job_root = _job_space_root(source_ids, [row["cell_id"] for row in cells])
    scope = _hash_value({
        "work_plan_sha256": work_plan["work_plan_sha256"],
        "pending_wiring_sha256": pending_wiring["pending_wiring_sha256"],
        "fanout_plan_sha256": fanout_plan["fanout_plan_sha256"],
        "job_space_sha256": job_root,
    })
    doc: dict[str, Any] = {
        "schema": GPTOSS_PLAN_SCHEMA, "created_at": created_at or _now(),
        "status": "unbound-physical-qualification-required",
        "model": work_plan["model"],
        "parents": {
            "work_plan_sha256": work_plan["work_plan_sha256"],
            "pending_wiring_sha256": pending_wiring["pending_wiring_sha256"],
            "fanout_plan_sha256": fanout_plan["fanout_plan_sha256"],
        },
        "matrix": {"rates": 10, "branches": 4, "cells": 40,
                   "source_units": len(source_ids), "isolated_jobs": job_count,
                   "isolated_job_space_sha256": job_root,
                   "job_space_framing": "u64be-length-plus-canonical-json-pair"},
        "cells": cells, "profiles": _profiles(scope), "facets": _facets(scope),
        "controller_sources": _tool_bindings(), "promotion_gate": _promotion_gate(),
        "activation": {
            "selected_profile_count": 0, "qualified_facet_count": 0,
            "runtime_specs_written": False, "registry_entries_written": False,
            "queue_mutated": False, "runtime_defaults_changed": False,
            "execution_permitted": False,
        },
        "source_or_parent_deletion_permitted": False,
        "quality_claims_permitted": False,
    }
    return _with_hash(doc, "acceleration_plan_sha256")


def validate_gptoss_acceleration_plan(
    doc: Any, work_plan: dict[str, Any], pending_wiring: dict[str, Any],
    fanout_plan: dict[str, Any],
) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != GPTOSS_PLAN_SCHEMA:
        return ["GPT-OSS acceleration plan schema mismatch"]
    errors: list[str] = []
    if doc.get("acceleration_plan_sha256") != _hash_value(
            _without(doc, "acceleration_plan_sha256")):
        errors.append("GPT-OSS acceleration plan hash mismatch")
    expected_parents = {
        "work_plan_sha256": work_plan.get("work_plan_sha256"),
        "pending_wiring_sha256": pending_wiring.get("pending_wiring_sha256"),
        "fanout_plan_sha256": fanout_plan.get("fanout_plan_sha256"),
    }
    if doc.get("parents") != expected_parents:
        errors.append("GPT-OSS acceleration parent binding differs")
    if doc.get("status") != "unbound-physical-qualification-required" \
            or doc.get("quality_claims_permitted") is not False \
            or doc.get("source_or_parent_deletion_permitted") is not False:
        errors.append("GPT-OSS acceleration plan crosses the claim boundary")
    activation = doc.get("activation")
    if not isinstance(activation, dict) or activation != {
            "selected_profile_count": 0, "qualified_facet_count": 0,
            "runtime_specs_written": False, "registry_entries_written": False,
            "queue_mutated": False, "runtime_defaults_changed": False,
            "execution_permitted": False,
            }:
        errors.append("GPT-OSS acceleration activation is not closed")
    cells = doc.get("cells")
    coverage = {(row.get("rate_id"), row.get("branch")) for row in cells
                if isinstance(row, dict)} if isinstance(cells, list) else set()
    expected_coverage = {(rate, branch) for rate in RATES for branch in BRANCHES}
    if not isinstance(cells, list) or len(cells) != 40 or coverage != expected_coverage \
            or any(row.get("execution_permitted") is not False for row in cells):
        errors.append("GPT-OSS acceleration matrix is not exact and pending")
    expected_cells = {(row["cell_id"], row["cell_identity_sha256"],
                       row["cell_spec_sha256"], row["rate_id"], row["branch"],
                       row["command"], row["adapter_id"])
                      for row in pending_wiring.get("cell_bindings", [])}
    observed_cells = {(row.get("cell_id"), row.get("cell_identity_sha256"),
                       row.get("cell_spec_sha256"), row.get("rate_id"),
                       row.get("branch"), row.get("command"), row.get("adapter_id"))
                      for row in cells if isinstance(row, dict)} \
        if isinstance(cells, list) else set()
    if observed_cells != expected_cells:
        errors.append("GPT-OSS cell authority differs from pending wiring")
    expected_cell_rows = [{
        "cell_id": row["cell_id"], "cell_identity_sha256": row["cell_identity_sha256"],
        "cell_spec_sha256": row["cell_spec_sha256"], "rate_id": row["rate_id"],
        "branch": row["branch"], "command": row["command"],
        "adapter_id": row["adapter_id"], "execution_permitted": False,
    } for row in pending_wiring.get("cell_bindings", [])]
    expected_cell_rows.sort(key=lambda row: row["cell_id"])
    if cells != expected_cell_rows:
        errors.append("GPT-OSS exact cell policy differs")
    source_ids = [row["unit_id"] for row in work_plan.get("source_units", [])]
    count, root = _job_space_root(source_ids, [row[0] for row in expected_cells])
    matrix = doc.get("matrix")
    if not isinstance(matrix, dict) or matrix.get("rates") != 10 \
            or matrix.get("branches") != 4 or matrix.get("cells") != 40 \
            or matrix.get("source_units") != len(source_ids) \
            or matrix.get("isolated_jobs") != count \
            or matrix.get("isolated_job_space_sha256") != root \
            or count != fanout.EXPECTED_JOBS:
        errors.append("GPT-OSS isolated job-space authority differs")
    scope = _hash_value({
        "work_plan_sha256": work_plan.get("work_plan_sha256"),
        "pending_wiring_sha256": pending_wiring.get("pending_wiring_sha256"),
        "fanout_plan_sha256": fanout_plan.get("fanout_plan_sha256"),
        "job_space_sha256": root,
    })
    errors += _validate_common_plan(doc, scope)
    return errors


def _higher_cells(
    manifest: dict[str, Any], *, sealed_manifest_sha256: str,
    manifest_attestation_sha256: str,
) -> list[dict[str, Any]]:
    model = manifest["model"]
    prefix = re.sub(r"[^a-z0-9]+", "-", model["label"].lower()).strip("-")
    cells = _matrix_cells(prefix, adapter_ids=False)
    for row in cells:
        row["source_manifest_sha256"] = manifest["manifest_sha256"]
        row["sealed_manifest_sha256"] = sealed_manifest_sha256
        row["manifest_attestation_sha256"] = manifest_attestation_sha256
        row["architecture_adapter_required"] = True
        row["adapter_id"] = manifest["architecture_adapter"]["adapter_id"]
        row["architecture_adapter_binding_sha256"] = manifest[
            "architecture_adapter"
        ]["binding_sha256"]
        row["tokenizer_binding_sha256"] = manifest["tokenizer_binding"][
            "binding_sha256"
        ]
        row["parameter_authority_sha256"] = model["parameter_authority_sha256"]
        row["lifecycle_binding_sha256"] = manifest["lifecycle_manifest"][
            "binding_sha256"
        ]
        row["transport_binding_sha256"] = manifest["transport_manifest"][
            "binding_sha256"
        ]
    return cells


def build_higher_tier_acceleration_plan(
    manifest: dict[str, Any], admission_plan: dict[str, Any], *,
    created_at: str | None = None,
) -> dict[str, Any]:
    core, errors = higher.unwrap_exact_source_manifest(manifest, verify_files=True)
    if core is not None:
        errors += higher.validate_admission_plan(admission_plan, core)
    if errors:
        raise AccelerationScaffoldError("invalid higher-tier parent: " + "; ".join(errors))
    assert core is not None
    sealed_manifest_sha256 = manifest["sealed_manifest_sha256"]
    manifest_attestation_sha256 = manifest["signed_manifest_attestation"][
        "attestation"
    ]["attestation_sha256"]
    cells = _higher_cells(
        core, sealed_manifest_sha256=sealed_manifest_sha256,
        manifest_attestation_sha256=manifest_attestation_sha256,
    )
    unit_ids = [row["unit_id"] for row in core["work_units"]]
    count, root = _job_space_root(unit_ids, [row["cell_template_id"] for row in cells])
    scope = _hash_value({
        "source_manifest_sha256": core["manifest_sha256"],
        "sealed_manifest_sha256": sealed_manifest_sha256,
        "manifest_attestation_sha256": manifest_attestation_sha256,
        "admission_plan_sha256": admission_plan["admission_plan_sha256"],
        "parameter_authority_sha256": core["model"]["parameter_authority_sha256"],
        "architecture_adapter_binding_sha256": core[
            "architecture_adapter"
        ]["binding_sha256"],
        "tokenizer_binding_sha256": core["tokenizer_binding"]["binding_sha256"],
        "lifecycle_binding_sha256": core["lifecycle_manifest"]["binding_sha256"],
        "transport_binding_sha256": core["transport_manifest"]["binding_sha256"],
        "job_space_sha256": root,
    })
    doc: dict[str, Any] = {
        "schema": HIGHER_PLAN_SCHEMA, "created_at": created_at or _now(),
        "status": "unbound-physical-qualification-required",
        "model": core["model"],
        "parents": {
            "source_manifest_sha256": core["manifest_sha256"],
            "sealed_manifest_sha256": sealed_manifest_sha256,
            "manifest_attestation_sha256": manifest_attestation_sha256,
            "admission_plan_sha256": admission_plan["admission_plan_sha256"],
            "parameter_authority_sha256": core["model"]["parameter_authority_sha256"],
            "architecture_adapter_binding_sha256": core[
                "architecture_adapter"
            ]["binding_sha256"],
            "tokenizer_binding_sha256": core["tokenizer_binding"]["binding_sha256"],
            "lifecycle_binding_sha256": core["lifecycle_manifest"]["binding_sha256"],
            "transport_binding_sha256": core["transport_manifest"]["binding_sha256"],
        },
        "matrix": {"rates": 10, "branches": 4, "cells": 40,
                   "source_units": len(unit_ids), "isolated_jobs": count,
                   "isolated_job_space_sha256": root,
                   "job_space_framing": "u64be-length-plus-canonical-json-pair"},
        "cells": cells, "estimated_waves": admission_plan["waves"],
        "profiles": _profiles(scope), "facets": _facets(scope),
        "controller_sources": _tool_bindings(), "promotion_gate": _promotion_gate(),
        "activation": {
            "architecture_adapter_bound": True,
            "parameter_authority_bound": True,
            "tokenizer_binding_bound": True,
            "lifecycle_manifest_bound": True,
            "transport_manifest_bound": True,
            "operator_manifest_authority_bound": True,
            "architecture_adapter_physically_qualified": False,
            "selected_profile_count": 0, "qualified_facet_count": 0,
            "runtime_specs_written": False, "registry_entries_written": False,
            "queue_mutated": False, "runtime_defaults_changed": False,
            "execution_permitted": False,
        },
        "source_or_parent_deletion_permitted": False,
        "quality_claims_permitted": False,
    }
    return _with_hash(doc, "acceleration_plan_sha256")


def validate_higher_tier_acceleration_plan(
    doc: Any, manifest: dict[str, Any], admission_plan: dict[str, Any],
) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != HIGHER_PLAN_SCHEMA:
        return ["higher-tier acceleration plan schema mismatch"]
    core, parent_errors = higher.unwrap_exact_source_manifest(
        manifest, verify_files=True,
    )
    if core is not None:
        parent_errors.extend(higher.validate_admission_plan(admission_plan, core))
    if parent_errors:
        return list(dict.fromkeys(parent_errors))
    assert core is not None
    sealed_manifest_sha256 = manifest.get("sealed_manifest_sha256")
    manifest_attestation_sha256 = manifest.get(
        "signed_manifest_attestation", {}
    ).get("attestation", {}).get("attestation_sha256")
    errors: list[str] = []
    if doc.get("acceleration_plan_sha256") != _hash_value(
            _without(doc, "acceleration_plan_sha256")):
        errors.append("higher-tier acceleration plan hash mismatch")
    if doc.get("parents") != {
            "source_manifest_sha256": core.get("manifest_sha256"),
            "sealed_manifest_sha256": sealed_manifest_sha256,
            "manifest_attestation_sha256": manifest_attestation_sha256,
            "admission_plan_sha256": admission_plan.get("admission_plan_sha256"),
            "parameter_authority_sha256": core.get("model", {}).get(
                "parameter_authority_sha256"
            ),
            "architecture_adapter_binding_sha256": core.get(
                "architecture_adapter", {}
            ).get("binding_sha256"),
            "tokenizer_binding_sha256": core.get("tokenizer_binding", {}).get(
                "binding_sha256"
            ),
            "lifecycle_binding_sha256": core.get("lifecycle_manifest", {}).get(
                "binding_sha256"
            ),
            "transport_binding_sha256": core.get("transport_manifest", {}).get(
                "binding_sha256"
            ),
            }:
        errors.append("higher-tier acceleration parent binding differs")
    if doc.get("status") != "unbound-physical-qualification-required" \
            or doc.get("quality_claims_permitted") is not False \
            or doc.get("source_or_parent_deletion_permitted") is not False:
        errors.append("higher-tier acceleration plan crosses the claim boundary")
    activation = doc.get("activation")
    if not isinstance(activation, dict) or activation.get("execution_permitted") is not False \
            or activation.get("architecture_adapter_bound") is not True \
            or activation.get("parameter_authority_bound") is not True \
            or activation.get("tokenizer_binding_bound") is not True \
            or activation.get("lifecycle_manifest_bound") is not True \
            or activation.get("transport_manifest_bound") is not True \
            or activation.get("operator_manifest_authority_bound") is not True \
            or activation.get("architecture_adapter_physically_qualified") is not False \
            or activation.get("queue_mutated") is not False \
            or activation.get("runtime_defaults_changed") is not False:
        errors.append("higher-tier acceleration activation is not closed")
    cells = doc.get("cells")
    coverage = {(row.get("rate_id"), row.get("branch")) for row in cells
                if isinstance(row, dict)} if isinstance(cells, list) else set()
    if not isinstance(cells, list) or len(cells) != 40 \
            or coverage != {(rate, branch) for rate in RATES for branch in BRANCHES} \
            or any(row.get("adapter_id") != core.get("architecture_adapter", {}).get(
                        "adapter_id"
                    )
                   or row.get("execution_permitted") is not False
                   or row.get("source_manifest_sha256") != core.get("manifest_sha256")
                   or row.get("sealed_manifest_sha256") != sealed_manifest_sha256
                   or row.get("manifest_attestation_sha256")
                   != manifest_attestation_sha256
                   or row.get("architecture_adapter_binding_sha256")
                   != core.get("architecture_adapter", {}).get("binding_sha256")
                   or row.get("tokenizer_binding_sha256")
                   != core.get("tokenizer_binding", {}).get("binding_sha256")
                   or row.get("parameter_authority_sha256")
                   != core.get("model", {}).get("parameter_authority_sha256")
                   or row.get("lifecycle_binding_sha256")
                   != core.get("lifecycle_manifest", {}).get("binding_sha256")
                   or row.get("transport_binding_sha256")
                   != core.get("transport_manifest", {}).get("binding_sha256")
                   for row in cells):
        errors.append("higher-tier 10x4 matrix is not exact, source-bound, and pending")
    if cells != _higher_cells(
        core, sealed_manifest_sha256=sealed_manifest_sha256,
        manifest_attestation_sha256=manifest_attestation_sha256,
    ):
        errors.append("higher-tier 10x4 cell policy differs")
    unit_ids = [row["unit_id"] for row in core.get("work_units", [])]
    cell_ids = [row["cell_template_id"] for row in _higher_cells(
        core, sealed_manifest_sha256=sealed_manifest_sha256,
        manifest_attestation_sha256=manifest_attestation_sha256,
    )]
    count, root = _job_space_root(unit_ids, cell_ids)
    matrix = doc.get("matrix")
    if not isinstance(matrix, dict) or matrix.get("rates") != 10 \
            or matrix.get("branches") != 4 or matrix.get("cells") != 40 \
            or matrix.get("source_units") != len(unit_ids) \
            or matrix.get("isolated_jobs") != count \
            or matrix.get("isolated_job_space_sha256") != root:
        errors.append("higher-tier isolated job-space authority differs")
    if doc.get("estimated_waves") != admission_plan.get("waves"):
        errors.append("higher-tier estimated admission waves differ")
    scope = _hash_value({
        "source_manifest_sha256": core.get("manifest_sha256"),
        "sealed_manifest_sha256": sealed_manifest_sha256,
        "manifest_attestation_sha256": manifest_attestation_sha256,
        "admission_plan_sha256": admission_plan.get("admission_plan_sha256"),
        "parameter_authority_sha256": core.get("model", {}).get(
            "parameter_authority_sha256"
        ),
        "architecture_adapter_binding_sha256": core.get(
            "architecture_adapter", {}
        ).get("binding_sha256"),
        "tokenizer_binding_sha256": core.get("tokenizer_binding", {}).get(
            "binding_sha256"
        ),
        "lifecycle_binding_sha256": core.get("lifecycle_manifest", {}).get(
            "binding_sha256"
        ),
        "transport_binding_sha256": core.get("transport_manifest", {}).get(
            "binding_sha256"
        ),
        "job_space_sha256": root,
    })
    errors += _validate_common_plan(doc, scope)
    return errors


def _validate_common_plan(doc: dict[str, Any], scope_sha256: str) -> list[str]:
    errors: list[str] = []
    profiles = doc.get("profiles")
    if not isinstance(profiles, list) or [row.get("rate_id") for row in profiles] != list(RATES) \
            or any(row.get("thread_candidates") != list(THREAD_CANDIDATES)
                   or row.get("selected_threads") is not None
                   or row.get("selected_phase_threads") is not None
                   or row.get("selected_lane_cap") is not None for row in profiles):
        errors.append("acceleration profiles are selected or incomplete")
    if profiles != _profiles(scope_sha256):
        errors.append("acceleration profile policy differs")
    facets = doc.get("facets")
    if not isinstance(facets, dict) or set(facets) != set(FACETS) \
            or any(not isinstance(row, dict)
                   or row.get("status") != "unqualified-default-off"
                   or row.get("physical_receipt_required") is not True
                   or row.get("simulation_can_activate") is not False
                   for row in facets.values()):
        errors.append("acceleration facet set is incomplete or prequalified")
    if facets != _facets(scope_sha256):
        errors.append("acceleration facet policy differs")
    gate = doc.get("promotion_gate")
    if not isinstance(gate, dict) or gate.get("currently_permitted") is not False \
            or gate.get("facet_receipts") != {facet: "missing" for facet in FACETS} \
            or gate.get("estimates_or_simulations_can_satisfy_gate") is not False:
        errors.append("acceleration promotion gate is open")
    if gate != _promotion_gate():
        errors.append("acceleration promotion policy differs")
    try:
        if doc.get("controller_sources") != _tool_bindings():
            raise AccelerationScaffoldError("controller source hashes differ")
    except (OSError, AccelerationScaffoldError) as exc:
        errors.append(f"acceleration controller verification failed: {exc}")
    return errors


def _validate_simple_requirements(
    doc: Any, *, schema: str, status: str, hash_field: str,
    require_source_deletion_field: bool = True,
) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != schema:
        return [f"{schema} schema mismatch"]
    errors: list[str] = []
    if doc.get(hash_field) != _hash_value(_without(doc, hash_field)):
        errors.append(f"{schema} hash mismatch")
    if doc.get("status") != status or doc.get("execution_permitted") is not False \
            or doc.get("quality_claims_permitted") is not False \
            or (require_source_deletion_field
                and doc.get("source_deletion_permitted") is not False):
        errors.append(f"{schema} boundary is invalid")
    return errors


def _validate_tokenizer_gate(doc: Any, *, verify_files: bool) -> list[str]:
    schema = "hawking.doctor_v5_gptoss_tokenizer_gate.v1"
    if not isinstance(doc, dict) or doc.get("schema") != schema \
            or doc.get("status") != "pass":
        return ["tokenizer gate schema/status invalid"]
    errors: list[str] = []
    if doc.get("gate_sha256") != _hash_value(_without(doc, "gate_sha256")):
        errors.append("tokenizer gate hash mismatch")
    if doc.get("checks") != {
            "dual_path_token_id_parity": True,
            "token_id_roundtrip_idempotence": True,
            "chat_template_reference_vector_present": True,
            "source_and_revision_bound": True,
            }:
        errors.append("tokenizer gate checks are incomplete")
    if doc.get("promotion_reviewed") is not False \
            or doc.get("quality_evaluation_permitted") is not False \
            or doc.get("source_deletion_permitted") is not False:
        errors.append("tokenizer gate overclaims promotion")
    if verify_files:
        for row in doc.get("files", []):
            try:
                if _artifact(Path(row["path"])) != row:
                    errors.append(f"tokenizer asset differs: {row.get('path')}")
            except (KeyError, TypeError, OSError, AccelerationScaffoldError):
                errors.append("tokenizer asset cannot be verified")
    return errors


def _validate_physical_ab_plan(doc: Any) -> list[str]:
    """Bind the exact physical controller lazily, avoiding its import cycle.

    ``doctor_v5_physical_ab_controller`` imports this module for the post-120B
    packet validator.  Importing it at module initialization would therefore
    create a cycle; by the time handoff validation runs, both modules are fully
    initialized and the controller can rebuild its deterministic source-bound
    plan safely.
    """
    if not isinstance(doc, dict) or doc.get("schema") != PHYSICAL_PLAN_SCHEMA:
        return ["physical A/B plan schema mismatch"]
    errors: list[str] = []
    if doc.get("plan_sha256") != _hash_value(_without(doc, "plan_sha256")):
        errors.append("physical A/B plan hash mismatch")
    try:
        physical = importlib.import_module("doctor_v5_physical_ab_controller")
        rebuilt = physical.build_plan()
    except (ImportError, OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"physical A/B plan cannot be rebuilt: {exc}")
        return errors
    if doc != rebuilt:
        errors.append("physical A/B plan differs from the current source-bound controller")
    if (
        doc.get("mode") != "unbound-default-off-validation-only"
        or doc.get("execution_capability") is not False
        or doc.get("activation_capability") is not False
        or doc.get("runtime_default_mutation_permitted") is not False
        or doc.get("heavy_lease_opened_by_controller") is not False
    ):
        errors.append("physical A/B controller crosses its default-off boundary")
    if doc.get("facet_order") != list(PHYSICAL_AB_FACETS) or doc.get("counts") != {
        "thread_profile_cells": 800,
        "thread_profile_selections": 200,
        "block_parallel_cells": 200,
        "facets": len(PHYSICAL_AB_FACETS),
    }:
        errors.append("physical A/B exact 10-facet/800-thread/200-block scope differs")
    return errors


def _handoff_inputs(
    *, work_plan_path: Path = DEFAULT_WORK_PLAN,
    pending_wiring_path: Path = DEFAULT_PENDING_WIRING,
    fanout_plan_path: Path = DEFAULT_FANOUT_PLAN,
    shared_requirements_path: Path = DEFAULT_SHARED_REQUIREMENTS,
    shared_manifest_path: Path = DEFAULT_SHARED_MANIFEST,
    shared_plan_path: Path = DEFAULT_SHARED_PLAN,
    tokenizer_binding_path: Path = DEFAULT_TOKENIZER_BINDING,
    tokenizer_gate_path: Path = DEFAULT_TOKENIZER_GATE,
    higher_requirements_path: Path = DEFAULT_HIGHER_REQUIREMENTS,
    acceleration_requirements_path: Path = DEFAULT_REQUIREMENTS,
    horizons_path: Path = DEFAULT_HORIZONS,
    gptoss_acceleration_path: Path = DEFAULT_GPTOSS_PLAN,
    physical_ab_plan_path: Path = DEFAULT_PHYSICAL_AB_PLAN,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    paths = {
        "gptoss_work_plan": Path(work_plan_path),
        "gptoss_pending_wiring": Path(pending_wiring_path),
        "gptoss_reuse_fanout": Path(fanout_plan_path),
        "shared_preprocess_requirements": Path(shared_requirements_path),
        "shared_preprocess_manifest": Path(shared_manifest_path),
        "shared_preprocess_plan": Path(shared_plan_path),
        "tokenizer_binding": Path(tokenizer_binding_path),
        "tokenizer_gate": Path(tokenizer_gate_path),
        "higher_tier_requirements": Path(higher_requirements_path),
        "acceleration_requirements": Path(acceleration_requirements_path),
        "named_horizons": Path(horizons_path),
        "gptoss_acceleration_plan": Path(gptoss_acceleration_path),
        "physical_ab_plan": Path(physical_ab_plan_path),
    }
    return ({name: _read_json(path) for name, path in paths.items()}, paths)


def _validate_handoff_inputs(inputs: dict[str, dict[str, Any]], *,
                             verify_files: bool) -> list[str]:
    work = inputs["gptoss_work_plan"]
    pending = inputs["gptoss_pending_wiring"]
    reuse = inputs["gptoss_reuse_fanout"]
    shared_requirements = inputs["shared_preprocess_requirements"]
    shared_manifest = inputs["shared_preprocess_manifest"]
    shared_plan = inputs["shared_preprocess_plan"]
    tokenizer_binding = inputs["tokenizer_binding"]
    tokenizer_gate = inputs["tokenizer_gate"]
    higher_requirements = inputs["higher_tier_requirements"]
    accel_requirements = inputs["acceleration_requirements"]
    horizons = inputs["named_horizons"]
    accel_plan = inputs["gptoss_acceleration_plan"]
    physical_plan = inputs["physical_ab_plan"]
    errors = parallel.validate_work_plan(work)
    errors += parallel.validate_pending_wiring(pending)
    errors += fanout.validate_fanout_plan(reuse, work, pending)
    errors += _validate_simple_requirements(
        shared_requirements, schema=shared_cache.REQUIREMENTS_SCHEMA,
        status="unbound-default-off-no-live-import", hash_field="requirements_sha256",
    )
    try:
        if shared_requirements != shared_cache.build_requirements_packet(
                created_at=shared_requirements.get("created_at")):
            errors.append("shared-preprocess requirements source bindings are stale")
    except (OSError, KeyError, TypeError, shared_cache.SharedCacheError) as exc:
        errors.append(f"shared-preprocess requirements cannot be rebuilt: {exc}")
    errors += shared_cache.validate_consumer_manifest(shared_manifest)
    errors += shared_cache.validate_cache_plan(shared_plan, shared_manifest)
    errors += parallel.validate_tokenizer_binding(tokenizer_binding,
                                                  verify_files=verify_files)
    errors += _validate_tokenizer_gate(tokenizer_gate, verify_files=verify_files)
    errors += _validate_simple_requirements(
        higher_requirements, schema=higher.REQUIREMENTS_SCHEMA,
        status="generic-scaffold-only-no-model-admitted",
        hash_field="requirements_sha256", require_source_deletion_field=False,
    )
    try:
        # The higher-tier builder has no timestamp argument. Rebuild its body,
        # transplant the recorded timestamp, and recompute the semantic seal.
        rebuilt = higher.build_requirements_packet()
        rebuilt["created_at"] = higher_requirements.get("created_at")
        rebuilt["requirements_sha256"] = _hash_value(
            _without(rebuilt, "requirements_sha256")
        )
        if higher_requirements != rebuilt:
            errors.append("higher-tier requirements source bindings are stale")
    except (OSError, KeyError, TypeError, higher.HigherTierError) as exc:
        errors.append(f"higher-tier requirements cannot be rebuilt: {exc}")
    errors += validate_requirements(accel_requirements)
    errors += validate_named_horizon_scaffold(horizons)
    errors += validate_gptoss_acceleration_plan(accel_plan, work, pending, reuse)
    errors += _validate_physical_ab_plan(physical_plan)
    if pending.get("work_plan", {}).get("work_plan_sha256") \
            != work.get("work_plan_sha256") \
            or reuse.get("work_plan_sha256") != work.get("work_plan_sha256") \
            or shared_manifest.get("parent_bindings", {}).get("work_plan_sha256") \
            != work.get("work_plan_sha256") \
            or shared_manifest.get("parent_bindings", {}).get("fanout_plan_sha256") \
            != reuse.get("fanout_plan_sha256"):
        errors.append("GPT-OSS work/wiring/reuse/cache parent chain does not close")
    try:
        if shared_manifest.get("parent_bindings", {}).get(
                "shared_cache_builder_sha256") != _artifact(
                    Path(shared_cache.__file__)
                )["sha256"]:
            errors.append("shared-preprocess manifest builder binding is stale")
    except (OSError, AccelerationScaffoldError) as exc:
        errors.append(f"shared-preprocess builder cannot be verified: {exc}")
    if horizons.get("requirements_sha256") != accel_requirements.get(
            "requirements_sha256"):
        errors.append("named horizons bind a different acceleration requirements packet")
    if tokenizer_binding.get("model_source_manifest_sha256") \
            != tokenizer_gate.get("model_source_manifest_sha256") \
            or tokenizer_binding.get("model_source_manifest_sha256") \
            != work.get("inventory", {}).get("source_manifest_sha256"):
        errors.append("tokenizer/model/work-plan source authority differs")
    return errors


def build_handoff_packet(*, created_at: str | None = None,
                         verify_files: bool = True, **paths: Path) -> dict[str, Any]:
    inputs, input_paths = _handoff_inputs(**paths)
    errors = _validate_handoff_inputs(inputs, verify_files=verify_files)
    if errors:
        raise AccelerationScaffoldError("handoff inputs invalid: " + "; ".join(errors))
    bindings = {
        name: {**_artifact(input_paths[name]),
               "schema": inputs[name].get("schema"),
               "semantic_hash_field": HANDOFF_SEMANTIC_FIELDS[name],
               "semantic_sha256": inputs[name][HANDOFF_SEMANTIC_FIELDS[name]]}
        for name in input_paths
    }
    cache_blocked = inputs["shared_preprocess_plan"].get("coverage", {}).get(
        "consumers_without_qualified_derived_preprocess"
    )
    doc: dict[str, Any] = {
        "schema": HANDOFF_SCHEMA, "created_at": created_at or _now(),
        "status": "sealed-unbound-handoff-not-executable",
        "bindings": bindings,
        "coverage": {
            "gptoss_source_units": parallel.EXPECTED_SOURCE_UNITS,
            "gptoss_isolated_jobs": fanout.EXPECTED_JOBS,
            "gptoss_rates": 10, "gptoss_branches": 4,
            "gptoss_pending_cells": 40,
            "named_higher_tier_horizons": len(HORIZONS),
            "named_horizon_cell_templates": len(HORIZONS) * 40,
            "aggressive_facets": list(FACETS),
            "physical_ab_facets": list(PHYSICAL_AB_FACETS),
            "physical_thread_profile_cells": 800,
            "physical_block_parallel_cells": 200,
        },
        "readiness": _handoff_readiness(cache_blocked, inputs["physical_ab_plan"]),
        "promotion_gate": _promotion_gate(),
        "claim_boundary": _handoff_claim_boundary(),
    }
    return _with_hash(doc, "handoff_sha256")


def validate_handoff_packet(doc: Any, *, verify_files: bool = True) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != HANDOFF_SCHEMA:
        return ["post-120B handoff schema mismatch"]
    errors: list[str] = []
    if doc.get("handoff_sha256") != _hash_value(_without(doc, "handoff_sha256")):
        errors.append("post-120B handoff hash mismatch")
    boundary = doc.get("claim_boundary")
    if doc.get("status") != "sealed-unbound-handoff-not-executable" \
            or boundary != _handoff_claim_boundary():
        errors.append("post-120B handoff crosses its claim boundary")
    bindings = doc.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != {
            "gptoss_work_plan", "gptoss_pending_wiring", "gptoss_reuse_fanout",
            "shared_preprocess_requirements", "shared_preprocess_manifest",
            "shared_preprocess_plan", "tokenizer_binding", "tokenizer_gate",
            "higher_tier_requirements", "acceleration_requirements",
            "named_horizons", "gptoss_acceleration_plan", "physical_ab_plan",
            }:
        errors.append("post-120B handoff binding inventory differs")
        return errors
    try:
        paths = {name: Path(row["path"]) for name, row in bindings.items()}
        inputs = {name: _read_json(path) for name, path in paths.items()}
        for name, row in bindings.items():
            observed = _artifact(paths[name])
            if any(observed[field] != row.get(field) for field in ("path", "sha256", "bytes")) \
                    or row.get("semantic_hash_field") \
                    != HANDOFF_SEMANTIC_FIELDS[name] \
                    or inputs[name].get(row.get("semantic_hash_field")) \
                    != row.get("semantic_sha256") \
                    or not isinstance(row.get("semantic_sha256"), str) \
                    or SHA_RE.fullmatch(row["semantic_sha256"]) is None \
                    or inputs[name].get("schema") != row.get("schema"):
                errors.append(f"handoff artifact binding differs: {name}")
        errors += _validate_handoff_inputs(inputs, verify_files=verify_files)
        cache_blocked = inputs["shared_preprocess_plan"].get("coverage", {}).get(
            "consumers_without_qualified_derived_preprocess"
        )
        if doc.get("readiness") != _handoff_readiness(
                cache_blocked, inputs["physical_ab_plan"]):
            errors.append("post-120B handoff readiness differs from bound inputs")
    except (OSError, KeyError, TypeError, AccelerationScaffoldError) as exc:
        errors.append(f"post-120B handoff input verification failed: {exc}")
    if doc.get("coverage") != {
            "gptoss_source_units": parallel.EXPECTED_SOURCE_UNITS,
            "gptoss_isolated_jobs": fanout.EXPECTED_JOBS,
            "gptoss_rates": 10, "gptoss_branches": 4,
            "gptoss_pending_cells": 40,
            "named_higher_tier_horizons": len(HORIZONS),
            "named_horizon_cell_templates": len(HORIZONS) * 40,
            "aggressive_facets": list(FACETS),
            "physical_ab_facets": list(PHYSICAL_AB_FACETS),
            "physical_thread_profile_cells": 800,
            "physical_block_parallel_cells": 200,
            }:
        errors.append("post-120B handoff coverage differs")
    if doc.get("promotion_gate") != _promotion_gate():
        errors.append("post-120B handoff promotion gate differs")
    return errors


def _handoff_readiness(
    cache_blocked: Any, physical_plan: dict[str, Any],
) -> dict[str, Any]:
    registry = physical_plan.get("executor_manifest", {}).get(
        "program_adapter_registry", {}
    )
    return {
        "exact_120b_work_plan_structurally_valid": True,
        "exact_40_cell_pending_wiring_structurally_valid": True,
        "exact_source_rate_branch_fanout_structurally_valid": True,
        "shared_preprocess_cache_structurally_valid": True,
        "shared_derived_preprocess_consumers_still_unqualified": cache_blocked,
        "tokenizer_dual_path_gate_passed_but_not_promotion_reviewed": True,
        "higher_tier_generic_manifest_and_admission_contracts_present": True,
        "higher_tier_exact_source_manifests_present": 0,
        "physical_acceleration_facets_qualified": 0,
        "physical_ab_plan_structurally_valid": True,
        "physical_ab_facets_total": len(PHYSICAL_AB_FACETS),
        "physical_ab_facets_qualified": 0,
        "physical_ab_program_adapters_registered": (
            len(registry) if isinstance(registry, dict) else 0
        ),
        "physical_ab_execution_permitted": False,
        "reviewed_120b_live_adapters": 0,
    }


def _handoff_claim_boundary() -> dict[str, Any]:
    return {
        "intermediate_or_structural_evidence_is_final_quality": False,
        "unsupported_or_negative_outcomes_synthesized": False,
        "live_queue_worker_registry_plan_or_runtime_specs_mutated": False,
        "runtime_defaults_changed": False,
        "source_or_parent_deletion_permitted": False,
        "execution_permitted": False,
        "quality_claims_permitted": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    req = sub.add_parser("requirements")
    req.add_argument("--output", type=Path, default=DEFAULT_REQUIREMENTS)
    horizons = sub.add_parser("horizons")
    horizons.add_argument("--output", type=Path, default=DEFAULT_HORIZONS)
    gptoss = sub.add_parser("build-gptoss")
    gptoss.add_argument("--work-plan", type=Path, default=DEFAULT_WORK_PLAN)
    gptoss.add_argument("--pending-wiring", type=Path, default=DEFAULT_PENDING_WIRING)
    gptoss.add_argument("--fanout-plan", type=Path, default=DEFAULT_FANOUT_PLAN)
    gptoss.add_argument("--output", type=Path, default=DEFAULT_GPTOSS_PLAN)
    verify_gptoss = sub.add_parser("verify-gptoss")
    verify_gptoss.add_argument("--work-plan", type=Path, default=DEFAULT_WORK_PLAN)
    verify_gptoss.add_argument("--pending-wiring", type=Path,
                              default=DEFAULT_PENDING_WIRING)
    verify_gptoss.add_argument("--fanout-plan", type=Path,
                              default=DEFAULT_FANOUT_PLAN)
    verify_gptoss.add_argument("--plan", type=Path, default=DEFAULT_GPTOSS_PLAN)
    higher_cmd = sub.add_parser("build-higher")
    higher_cmd.add_argument("--manifest", type=Path, required=True)
    higher_cmd.add_argument("--admission-plan", type=Path, required=True)
    higher_cmd.add_argument("--output", type=Path, required=True)
    verify_higher = sub.add_parser("verify-higher")
    verify_higher.add_argument("--manifest", type=Path, required=True)
    verify_higher.add_argument("--admission-plan", type=Path, required=True)
    verify_higher.add_argument("--plan", type=Path, required=True)
    handoff_cmd = sub.add_parser("handoff")
    handoff_cmd.add_argument("--output", type=Path, default=DEFAULT_HANDOFF)
    verify_handoff = sub.add_parser("verify-handoff")
    verify_handoff.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    args = parser.parse_args(argv)
    try:
        if args.command == "requirements":
            doc = build_requirements_packet()
            errors = validate_requirements(doc)
            output = args.output
        elif args.command == "horizons":
            requirements = _read_json(DEFAULT_REQUIREMENTS) \
                if DEFAULT_REQUIREMENTS.exists() else build_requirements_packet()
            doc = build_named_horizon_scaffold(requirements=requirements)
            errors = validate_named_horizon_scaffold(doc)
            output = args.output
        elif args.command in {"build-gptoss", "verify-gptoss"}:
            work = _read_json(args.work_plan)
            pending = _read_json(args.pending_wiring)
            fanout_plan = _read_json(args.fanout_plan)
            doc = build_gptoss_acceleration_plan(work, pending, fanout_plan) \
                if args.command == "build-gptoss" else _read_json(args.plan)
            errors = validate_gptoss_acceleration_plan(doc, work, pending, fanout_plan)
            output = args.output if args.command == "build-gptoss" else None
        elif args.command in {"build-higher", "verify-higher"}:
            manifest = _read_json(args.manifest)
            admission = _read_json(args.admission_plan)
            doc = build_higher_tier_acceleration_plan(manifest, admission) \
                if args.command == "build-higher" else _read_json(args.plan)
            errors = validate_higher_tier_acceleration_plan(doc, manifest, admission)
            output = args.output if args.command == "build-higher" else None
        elif args.command == "handoff":
            doc = build_handoff_packet()
            errors = validate_handoff_packet(doc)
            output = args.output
        else:
            doc = _read_json(args.handoff)
            errors = validate_handoff_packet(doc)
            output = None
        if errors:
            print(json.dumps({"status": "invalid", "errors": errors},
                             indent=2, sort_keys=True))
            return 2
        if output is not None:
            _write_json(output, doc)
        print(json.dumps({
            "status": "ok", "execution_permitted": False,
            "output": str(output.resolve()) if output is not None else None,
            "schema": doc["schema"],
            "sha256": doc.get("handoff_sha256")
                      or doc.get("horizon_scaffold_sha256")
                      or doc.get("acceleration_plan_sha256")
                      or doc.get("requirements_sha256"),
        }, indent=2, sort_keys=True))
        return 0
    except (AccelerationScaffoldError, higher.HigherTierError,
            parallel.ParallelScaffoldError, fanout.FanoutError, OSError,
            KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

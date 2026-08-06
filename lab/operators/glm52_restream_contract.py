"""Build the fail-closed, range-windowed GLM-5.2 parent-restream contract.

This module is deliberately an offline contract builder.  It proves exact
range partitioning and conservative incremental byte accounting, but it does
not authorize source-body acquisition or claim a live allocation/capability
result.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from lab.layout import resolve_workspace_path
from lab.operators.glm52_common import canonical, read_sealed_json, seal
from lab.operators.gravity_range_scheduler import (
    DEFAULT_ENVELOPE_BYTES,
    RANGE_ALIGNMENT_BYTES,
    plan_glm52_organ_windows,
)
from ramanujan.restream_guard import ACCOUNTING_COMPONENTS, validate_bounded_restream


REPO_ID = "zai-org/GLM-5.2"
REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
PROTECTED_FILESYSTEM_FLOOR_BYTES = 200_005_889_556
DEFAULT_ROLLBACK_SCRATCH_BYTES = 16 * 1024**3
DEFAULT_ARTIFACT_BYTES = 2 * 1024**3
DEFAULT_METADATA_BYTES = 64 * 1024**2


class RestreamContractError(RuntimeError):
    """The proposed offline restream contract is incomplete or unsafe."""


def _round_up(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RestreamContractError("byte counts must be non-negative integers")
    return ((value + RANGE_ALIGNMENT_BYTES - 1) // RANGE_ALIGNMENT_BYTES) * RANGE_ALIGNMENT_BYTES


def _sha256_list(values: list[str]) -> str:
    return hashlib.sha256(canonical(values)).hexdigest()


def build_contract(
    *,
    manifest_path: str | Path,
    graph_path: str | Path,
    artifact_bytes_per_window: int = DEFAULT_ARTIFACT_BYTES,
    metadata_bytes_per_window: int = DEFAULT_METADATA_BYTES,
    rollback_scratch_bytes: int = DEFAULT_ROLLBACK_SCRATCH_BYTES,
    envelope_bytes: int = DEFAULT_ENVELOPE_BYTES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a sealed schedule and matching policy without reading model bodies.

    Unlike the retained-artifact diagnostic, this schedule explicitly requires
    a sealed-and-evicted receipt for N before N+1 can materialize.  Therefore
    only N's bounded artifact family is charged in each window.  A dedicated
    16-GiB rollback/scratch reserve is charged even when the planner's 5%
    source scratch estimate is smaller.
    """
    manifest = read_sealed_json(resolve_workspace_path(manifest_path))
    graph = read_sealed_json(resolve_workspace_path(graph_path))
    if manifest.get("repo") != REPO_ID or manifest.get("revision") != REVISION:
        raise RestreamContractError("manifest is not the pinned final parent")
    if graph.get("repo") != REPO_ID or graph.get("revision") != REVISION:
        raise RestreamContractError("dependency graph is not the pinned final parent")

    candidate = plan_glm52_organ_windows(
        "GLM-5.2-parent-restream",
        manifest_path=manifest_path,
        graph_path=graph_path,
        artifact_bytes_per_window=artifact_bytes_per_window,
        metadata_bytes_per_window=metadata_bytes_per_window,
        scratch_multiplier=1.05,
        envelope_bytes=envelope_bytes,
    )
    ranges_by_window: dict[str, list[str]] = {}
    for row in candidate["ranges"]:
        ranges_by_window.setdefault(str(row["window_id"]), []).append(str(row["range_id"]))

    windows: list[dict[str, Any]] = []
    peak = 0
    for execution_order, row in enumerate(candidate["windows"]):
        window_id = str(row["window_id"])
        scratch = max(_round_up(int(row["scratch_bytes"])), _round_up(rollback_scratch_bytes))
        accounting = {
            "source_range_rounded_bytes": int(row["range_rounded_bytes"]),
            "source_scratch_bytes": scratch,
            "retained_artifact_bytes": int(row["artifact_rounded_bytes"]),
            "teacher_evidence_bytes": 0,
            "metadata_bytes": int(row["metadata_rounded_bytes"]),
            "carry_bytes": int(row["carry_rounded_bytes"]),
            "prefetch_bytes": int(row["prefetch_rounded_bytes"]),
        }
        if tuple(accounting) != ACCOUNTING_COMPONENTS:
            raise AssertionError("accounting component order drifted")
        resident = sum(accounting.values())
        peak = max(peak, resident)
        range_ids = ranges_by_window[window_id]
        windows.append(
            {
                "window_id": window_id,
                "execution_order": execution_order,
                "range_count": len(range_ids),
                "ordered_range_ids_sha256": _sha256_list(range_ids),
                "predecessor_window_id": windows[-1]["window_id"] if windows else None,
                "predecessor_gate": (
                    "GENESIS" if not windows else "SEALED_ARTIFACT_HASH_AND_EXACT_SOURCE_ARTIFACT_TEMP_EVICTION_RECEIPT"
                ),
                "incremental_accounting": {**accounting, "resident_incremental_bytes": resident},
            }
        )
    if peak > envelope_bytes:
        raise RestreamContractError(f"seal/evict schedule peak {peak} exceeds {envelope_bytes}")

    schedule = seal(
        {
            "schema": "hawking.glm52.streaming_range_schedule.v1",
            "status": "OFFLINE_ADMITTED_LIVE_ALLOCATION_AND_CAPABILITY_REQUIRED",
            "authoritative": False,
            "live_execution_authorized": False,
            "repo": REPO_ID,
            "revision": REVISION,
            "active_model_limit": 1,
            "inputs": {
                "manifest_path": str(manifest_path),
                "manifest_seal_sha256": manifest["seal_sha256"],
                "dependency_graph_path": str(graph_path),
                "dependency_graph_seal_sha256": graph["seal_sha256"],
            },
            "partition": {
                **candidate["partition"],
                "physical_range_fetch_implemented": True,
                "source_ranges_rebuilt_from_sealed_inputs_at_execution": True,
            },
            "incremental_accounting_contract": {
                "alignment_bytes": RANGE_ALIGNMENT_BYTES,
                "components": list(ACCOUNTING_COMPONENTS),
                "envelope_bytes": envelope_bytes,
                "peak_incremental_bytes": peak,
                "logical_source_bytes_are_not_allocated_byte_authority": True,
            },
            "lifecycle": {
                "order": "N-1 seal/hash/evict -> N stream/process/one-pass-pack -> N+1 bounded prefetch",
                "artifact_retention": "CURRENT_WINDOW_ONLY",
                "next_window_refused_until_predecessor_receipt_is_sealed": True,
                "receipt_requires_exact_source_artifact_temp_cache_eviction": True,
                "one_heavy_lease": True,
                "one_pass_multi_artifact_pack": True,
                "rollback_scratch_reserve_bytes": _round_up(rollback_scratch_bytes),
            },
            "source_accounting": {
                "logical_payload_bytes": candidate["totals"]["range_payload_bytes"],
                "rounded_payload_bytes": candidate["totals"]["range_rounded_bytes"],
                "range_count": candidate["partition"]["range_count"],
                "window_count": candidate["partition"]["window_count"],
                "exact_once": candidate["exact_once_range_accounting"],
            },
            "windows": windows,
            "required_live_gates": [
                "OWNER_PARENT_RESTREAM_AUTHORIZATION",
                "CLEAN_GPU_LEASE",
                "FRESH_FREE_BYTES_MINUS_WINDOW_PEAK_AT_OR_ABOVE_PROTECTED_FLOOR",
                "PINNED_XET_RUNTIME",
                "TESTED_WINDOW_OPERATOR",
                "LIVE_ALLOCATION_AND_CAPABILITY_RECEIPT",
            ],
        }
    )
    policy = seal(
        {
            "schema": "hawking.glm52.resource_reserve_policy_90gb.v1",
            "status": "SEALED_POLICY_LIVE_SAMPLE_REQUIRED",
            "authoritative": False,
            "input_seals": {
                "streaming_schedule": schedule["seal_sha256"],
                "official_manifest": manifest["seal_sha256"],
                "dependency_graph": graph["seal_sha256"],
            },
            "policy": {
                "incremental_storage_ceiling_bytes": envelope_bytes,
                "active_model_limit": 1,
                "protected_filesystem_floor_bytes": PROTECTED_FILESYSTEM_FLOOR_BYTES,
                "recompute_free_bytes_before_every_window": True,
                "admission_law": "free_bytes - resident_incremental_bytes >= protected_filesystem_floor_bytes",
                "incremental_accounting_components": list(ACCOUNTING_COMPONENTS),
                "rollback_scratch_reserve_bytes": _round_up(rollback_scratch_bytes),
                "all_model_source_artifact_temp_cache_bytes_evicted_after_terminal_receipt": True,
            },
        }
    )
    validated = validate_bounded_restream(schedule, policy)
    if validated != {"peak_incremental_bytes": peak, "window_count": len(windows)}:
        raise AssertionError("guard result differs from constructed schedule")
    return schedule, policy


def live_window_admission(
    schedule: Mapping[str, Any], policy: Mapping[str, Any], *, window_id: str, free_bytes: int
) -> dict[str, Any]:
    """Apply the unified-floor law to one freshly sampled window admission."""
    validate_bounded_restream(schedule, policy)
    rows = [row for row in schedule["windows"] if row.get("window_id") == window_id]
    if len(rows) != 1:
        raise RestreamContractError(f"unknown or duplicate window_id: {window_id!r}")
    if isinstance(free_bytes, bool) or not isinstance(free_bytes, int) or free_bytes < 0:
        raise RestreamContractError("free_bytes must be a non-negative integer")
    required = int(rows[0]["incremental_accounting"]["resident_incremental_bytes"])
    floor = int(policy["policy"]["protected_filesystem_floor_bytes"])
    residual = free_bytes - floor
    return {
        "window_id": window_id,
        "free_bytes": free_bytes,
        "protected_filesystem_floor_bytes": floor,
        "residual_above_floor_bytes": residual,
        "resident_incremental_bytes": required,
        "admitted": residual >= required,
    }

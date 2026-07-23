#!/usr/bin/env python3.12
"""Freeze the conservative live resource policy for the GLM-5.2 campaign.

The policy is derived only from already sealed admission artifacts.  It is an
upper-bound reservation, not a live resource observation: the worker must also
sample disk/RAM/swap immediately before every fetch, seal, and eviction.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
for import_root in (REPOSITORY_ROOT, HERE):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from glm52_common import Glm52Error, atomic_json, read_sealed_json, seal, verify_sealed
from glm52_grounding import ResourceReservePolicy


SCHEMA = "hawking.glm52.resource_reserve_policy.v1"
STATUS = "FROZEN_CONSERVATIVE_PRELIVE_POLICY"
OFFICIAL_REPO = "zai-org/GLM-5.2"
OFFICIAL_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
FROZEN_CREATED_AT = "2026-07-21T22:07:16Z"
EXACT_INPUTS = {
    "streaming_schedule": (
        "GLM52_STREAMING_SCHEDULE_PRE_AUTOTUNE.json",
        "hawking.glm52.streaming_schedule.v1",
        "PRELIMINARY_DEPENDENCY_COMPLETE_PENDING_XET_AUTOTUNE",
        "ca765d5197d25a888ebc1e31ca5d03cae4492235d3470dd6c93c53f98682af89",
    ),
    "official_manifest": (
        "GLM52_OFFICIAL_MANIFEST.json",
        "hawking.glm52.official_manifest.v1",
        "PASS_CONTROL_PLANE_AND_HEADERS_BODY_PENDING",
        "ef4e3cbf6b4b6ddc52aa7278d657a9f7f63254fa72d35591cc8f6c5c706c4791",
    ),
    "logical_weight_ledger": (
        "GLM52_LOGICAL_WEIGHT_LEDGER.json",
        "hawking.glm52.logical_weight_ledger.v1",
        "PASS_HEADER_DERIVED",
        "d479791a7f3c279f0d1cc56a6203e214e4e2610da04010ba020347603dacb0db",
    ),
    "kimi_byte_auction": (
        "KIMI_K26_FINAL_BYTE_AUCTION.json",
        "hawking.kimi_k26.final_byte_auction.v1",
        "PASS",
        "d8b8f02b2e19c4bee39cf8a5ceb91f05e1878ee18e889c2bcf5f88ccc1b42719",
    ),
}
MINIMUM_AVAILABLE_RAM_BYTES = 16 * 1024**3
MAXIMUM_SWAP_USED_BYTES = 8 * 1024**3
NEXT_CHECKPOINT_WRITE_BYTES = 1 * 1024**3
SERIALIZED_OR_PARTIAL_PREFETCH = "SERIALIZED_OR_PARTIAL_PREFETCH"
LIVE_ALLOCATED_MEASUREMENT_REQUIRED = "LIVE_ALLOCATED_MEASUREMENT_REQUIRED"
PROVISIONAL_CONTROL_LIMIT_STATUS = (
    "PREREGISTERED_PROVISIONAL_NOT_DERIVED_FROM_SEALED_INPUTS"
)
EXPECTED_LARGEST_ADJACENT_UNION_BYTES = 236_190_533_120
EXPECTED_LARGEST_ADJACENT_TRANSITION = ("W017", "W018")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class ResourcePolicyError(Glm52Error):
    """A frozen resource-policy input or derivation is invalid."""


def _created_at(value: Any) -> str:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise ResourcePolicyError("created_at must be explicit UTC YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise ResourcePolicyError("created_at is not a real UTC timestamp") from None
    return value


def _load_inputs(root: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for name, (filename, schema, status, expected_seal) in EXACT_INPUTS.items():
        try:
            value = read_sealed_json(root / filename)
        except (OSError, Glm52Error) as exc:
            raise ResourcePolicyError(f"cannot verify frozen resource input {filename}: {exc}") \
                from exc
        if value.get("schema") != schema or value.get("status") != status \
                or value.get("seal_sha256") != expected_seal:
            raise ResourcePolicyError(f"frozen resource input identity changed: {filename}")
        if name != "kimi_byte_auction" and (
            value.get("repo") != OFFICIAL_REPO
            or value.get("revision") != OFFICIAL_REVISION
        ):
            raise ResourcePolicyError(f"frozen resource input source changed: {filename}")
        values[name] = value
    return values


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResourcePolicyError(f"{label} must be a non-negative integer")
    return value


def _weight_size_index(manifest: Mapping[str, Any]) -> dict[str, int]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ResourcePolicyError("official manifest file inventory is absent")
    result: dict[str, int] = {}
    for index, item in enumerate(files):
        if not isinstance(item, Mapping) or item.get("is_weight") is not True:
            continue
        path = item.get("path")
        if not isinstance(path, str) or not path or path in result:
            raise ResourcePolicyError(
                f"official manifest weight path is invalid at index {index}"
            )
        result[path] = _nonnegative_int(
            item.get("logical_bytes"), f"official manifest {path}.logical_bytes"
        )
    if len(result) != 282:
        raise ResourcePolicyError("official manifest must contain exactly 282 weight shards")
    return result


def _path_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or len(value) != len(set(value)) \
            or any(not isinstance(item, str) or not item for item in value):
        raise ResourcePolicyError(f"{label} must be a unique non-empty path list")
    return list(value)


def _prefetch_control(
    windows: Sequence[Mapping[str, Any]],
    shard_bytes: Mapping[str, int],
    *,
    maximum_materialized_raw_allocated_bytes: int,
) -> dict[str, Any]:
    """Bind static logical proxies while refusing to treat them as allocation facts."""

    transitions: list[dict[str, Any]] = []
    for index, (active, following) in enumerate(zip(windows, windows[1:])):
        if not isinstance(active, Mapping) or not isinstance(following, Mapping):
            raise ResourcePolicyError("streaming schedule window is malformed")
        active_id = active.get("window_id")
        following_id = following.get("window_id")
        if not isinstance(active_id, str) or not isinstance(following_id, str):
            raise ResourcePolicyError("streaming schedule window identity is malformed")
        active_paths = _path_list(
            active.get("source_shards"), f"window {active_id}.source_shards"
        )
        following_paths = _path_list(
            following.get("source_shards"), f"window {following_id}.source_shards"
        )
        new_fetch_paths = _path_list(
            following.get("new_fetch_shards"),
            f"window {following_id}.new_fetch_shards",
        )
        unknown = (
            set(active_paths) | set(following_paths) | set(new_fetch_paths)
        ) - set(shard_bytes)
        if unknown:
            raise ResourcePolicyError(
                f"streaming schedule transition references unknown shards: {sorted(unknown)}"
            )
        active_bytes = sum(shard_bytes[path] for path in active_paths)
        requested_prefetch_bytes = sum(shard_bytes[path] for path in new_fetch_paths)
        adjacent_union_bytes = sum(
            shard_bytes[path] for path in set(active_paths) | set(following_paths)
        )
        if active.get("resident_logical_bytes") != active_bytes \
                or following.get("new_fetch_logical_bytes") != requested_prefetch_bytes:
            raise ResourcePolicyError(
                f"streaming schedule transition byte accounting changed at {active_id}"
            )
        proxy_ceiling = max(
            0, maximum_materialized_raw_allocated_bytes - active_bytes
        )
        deficit = max(
            0, adjacent_union_bytes - maximum_materialized_raw_allocated_bytes
        )
        transitions.append({
            "transition_index": index,
            "active_window_id": active_id,
            "prefetch_window_id": following_id,
            "active_resident_remote_logical_bytes": active_bytes,
            "requested_prefetch_remote_logical_bytes": requested_prefetch_bytes,
            "adjacent_union_remote_logical_bytes": adjacent_union_bytes,
            "preregistered_additional_prefetch_logical_proxy_ceiling_bytes":
                proxy_ceiling,
            "full_prefetch_remote_logical_deficit_bytes": deficit,
            "preregistered_mode": (
                SERIALIZED_OR_PARTIAL_PREFETCH
                if deficit
                else LIVE_ALLOCATED_MEASUREMENT_REQUIRED
            ),
        })
    if not transitions:
        raise ResourcePolicyError("resource policy requires adjacent window transitions")
    largest = max(
        transitions,
        key=lambda item: (
            item["adjacent_union_remote_logical_bytes"],
            item["transition_index"],
        ),
    )
    largest_identity = (
        largest["active_window_id"], largest["prefetch_window_id"]
    )
    if largest["adjacent_union_remote_logical_bytes"] != \
            EXPECTED_LARGEST_ADJACENT_UNION_BYTES \
            or largest_identity != EXPECTED_LARGEST_ADJACENT_TRANSITION:
        raise ResourcePolicyError(
            "sealed schedule largest adjacent active/prefetch union changed"
        )
    deficit = max(
        0,
        int(largest["adjacent_union_remote_logical_bytes"])
        - maximum_materialized_raw_allocated_bytes,
    )
    return {
        "mode": SERIALIZED_OR_PARTIAL_PREFETCH,
        "full_two_complete_window_pipeline_preregistered": False,
        "maximum_materialized_raw_allocated_bytes":
            maximum_materialized_raw_allocated_bytes,
        "maximum_additional_prefetch_allocated_bytes_formula": (
            "max(0, min(maximum_materialized_raw_allocated_bytes "
            "- live_materialized_raw_allocated_bytes_before_prefetch, "
            "live_disk_free_bytes - required_free_disk_bytes))"
        ),
        "remote_logical_bytes_are_allocated_upper_bounds": False,
        "live_allocated_byte_measurement_required_before_materialized_xet_body_acquisition":
            True,
        "when_live_full_prefetch_exceeds_ceiling": SERIALIZED_OR_PARTIAL_PREFETCH,
        "largest_adjacent_transition": list(largest_identity),
        "largest_adjacent_active_plus_prefetch_union_remote_logical_bytes":
            largest["adjacent_union_remote_logical_bytes"],
        "largest_adjacent_full_prefetch_deficit_bytes": deficit,
        "maximum_additional_prefetch_logical_proxy_ceiling_bytes": max(
            item["preregistered_additional_prefetch_logical_proxy_ceiling_bytes"]
            for item in transitions
        ),
        "transition_count": len(transitions),
        "transitions": transitions,
    }


def derive_policy(root: str | Path, *, created_at: str) -> dict[str, Any]:
    """Derive one deterministic policy from exact sealed admission evidence."""

    base = Path(root).resolve()
    timestamp = _created_at(created_at)
    if timestamp != FROZEN_CREATED_AT:
        raise ResourcePolicyError(
            f"created_at must equal the frozen policy timestamp {FROZEN_CREATED_AT}"
        )
    inputs = _load_inputs(base)
    schedule = inputs["streaming_schedule"]
    manifest = inputs["official_manifest"]
    ledger = inputs["logical_weight_ledger"]
    auction = inputs["kimi_byte_auction"]
    planner = schedule.get("planner_inputs")
    windows = schedule.get("windows")
    rates = ledger.get("rate_budgets")
    current_best = auction.get("current_best")
    if not isinstance(planner, Mapping) or not isinstance(windows, list) or not windows \
            or not isinstance(rates, Mapping) or not isinstance(current_best, Mapping):
        raise ResourcePolicyError("resource-policy input structure is incomplete")
    if schedule.get("window_count") != 20 or len(windows) != 20:
        raise ResourcePolicyError("resource policy requires the exact 20-window schedule")
    try:
        max_window_bytes = max(
            _nonnegative_int(
                item["resident_logical_bytes"], "window resident_logical_bytes"
            )
            for item in windows
        )
        largest_source = _nonnegative_int(
            manifest["largest_weight_shard"]["bytes"], "largest weight shard bytes"
        )
        hard_floor = _nonnegative_int(planner["hard_floor_bytes"], "hard floor")
        two_largest = _nonnegative_int(
            planner["largest_two_source_shards_bytes"], "two largest source shards"
        )
        projected_compact = _nonnegative_int(
            planner[
                "projected_three_complete_artifacts_bytes_0_98_plus_0_75_plus_0_50"
            ],
            "projected compact artifacts",
        )
        projected_evidence = _nonnegative_int(
            planner["projected_evidence_bytes"], "projected evidence"
        )
        active_scratch = _nonnegative_int(
            planner["active_scratch_bytes"], "active scratch"
        )
        initial_free = _nonnegative_int(
            planner["free_disk_bytes"], "preregistered free disk"
        )
        largest_compact = _nonnegative_int(
            rates["planned_0_98_bpw"]["maximum_complete_physical_bytes"],
            "largest compact artifact",
        )
        rollback_bytes = _nonnegative_int(
            current_best["complete_physical_bytes"], "rollback capsule"
        )
    except (KeyError, TypeError, ValueError):
        raise ResourcePolicyError("resource-policy numeric input is malformed") from None
    if any(value < 0 for value in (
        max_window_bytes, largest_source, hard_floor, two_largest,
        projected_compact, projected_evidence, active_scratch, initial_free,
        largest_compact, rollback_bytes,
    )):
        raise ResourcePolicyError("resource-policy numeric input is negative")
    if current_best.get("candidate") != "P1_DUAL_PATH_RECOVERY_R16X2" \
            or rollback_bytes != 5_001_815:
        raise ResourcePolicyError("Kimi rollback reservation is not the sealed winner")
    if planner.get("two_simultaneous_complete_raw_windows") is not True \
            or int(planner.get("operational_reserve_bytes", -1)) != two_largest:
        raise ResourcePolicyError("streaming schedule reserve premise changed")

    policy = ResourceReservePolicy(
        emergency_floor_bytes=hard_floor,
        largest_atomic_source_write_bytes=largest_source,
        largest_compact_shard_write_bytes=largest_compact,
        next_checkpoint_write_bytes=NEXT_CHECKPOINT_WRITE_BYTES,
        xet_reconstruction_scratch_bytes=two_largest,
        two_largest_official_source_shards_bytes=two_largest,
        projected_remaining_compact_bytes=projected_compact,
        projected_teacher_evidence_bytes=projected_evidence,
        active_scratch_bytes=active_scratch,
        current_best_artifact_bytes=largest_compact,
        rollback_capsule_bytes=rollback_bytes,
        minimum_available_ram_bytes=MINIMUM_AVAILABLE_RAM_BYTES,
        maximum_swap_used_bytes=MAXIMUM_SWAP_USED_BYTES,
    )
    usable = initial_free - policy.required_free_disk_bytes
    if usable < 0:
        raise ResourcePolicyError(
            "conservative policy exceeds preregistered free disk capacity"
        )
    shard_size_index = _weight_size_index(manifest)
    ordered_shard_sizes = sorted(shard_size_index.values(), reverse=True)
    if largest_source != ordered_shard_sizes[0] \
            or two_largest != sum(ordered_shard_sizes[:2]):
        raise ResourcePolicyError(
            "manifest and schedule largest-shard reservations disagree"
        )
    prefetch_control = _prefetch_control(
        windows,
        shard_size_index,
        maximum_materialized_raw_allocated_bytes=usable,
    )
    required_window_headroom = prefetch_control[
        "largest_adjacent_active_plus_prefetch_union_remote_logical_bytes"
    ]
    window_headroom_deficit = prefetch_control[
        "largest_adjacent_full_prefetch_deficit_bytes"
    ]
    input_seals = {
        name: value["seal_sha256"] for name, value in sorted(inputs.items())
    }
    return seal({
        "schema": SCHEMA,
        "status": STATUS,
        "repo": OFFICIAL_REPO,
        "revision": OFFICIAL_REVISION,
        "created_at": timestamp,
        "input_seals": input_seals,
        "policy": policy.as_dict(),
        "derived": {
            "operational_reserve_floor_bytes": policy.operational_reserve_floor_bytes,
            "additional_reserved_bytes": policy.additional_reserved_bytes,
            "required_free_disk_bytes": policy.required_free_disk_bytes,
            "preregistered_free_disk_bytes": initial_free,
            "preregistered_usable_raw_window_bytes": usable,
            "maximum_scheduled_resident_window_bytes": max_window_bytes,
            "required_window_plus_prefetch_headroom_bytes": required_window_headroom,
            "full_window_plus_prefetch_headroom_deficit_bytes":
                window_headroom_deficit,
        },
        "prefetch_control": prefetch_control,
        "provisional_control_limits": {
            "status": PROVISIONAL_CONTROL_LIMIT_STATUS,
            "minimum_available_ram_bytes": MINIMUM_AVAILABLE_RAM_BYTES,
            "maximum_swap_used_bytes": MAXIMUM_SWAP_USED_BYTES,
            "next_checkpoint_write_bytes": NEXT_CHECKPOINT_WRITE_BYTES,
            "derived_from_sealed_input_evidence": False,
            "live_measurement_required": True,
        },
        "activation": {
            "authority": "EXPECTED_CONTRACT_V3_EXACT_SEAL_PLUS_LIVE_RESOURCE_SAMPLE",
            "live_xet_result_required": True,
            "final_schedule_must_bind_this_policy_seal": True,
            "worker_must_hold_controller_lease": True,
            "eviction_must_remain_disabled_until_all_are_true": True,
            "live_allocated_byte_measurement_required_before_materialized_xet_body_acquisition":
                True,
            "remote_logical_bytes_authorize_materialized_body_acquisition": False,
        },
        "notes": [
            "The largest legal 0.98-BPW artifact is reserved both as an atomic write and as the preserved current best.",
            "The full 0.98+0.75+0.50 artifact ladder and teacher evidence remain additionally reserved.",
            "This static bound never substitutes for a fresh authenticated disk/RAM/swap observation.",
            "Remote logical shard bytes are scheduling proxies, not allocated-byte upper bounds.",
            "The RAM, swap, and checkpoint limits are preregistered provisional controls, not derived evidence.",
        ],
    })


def validate_policy(
    value: Mapping[str, Any],
    *,
    root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResourcePolicyError("resource reserve policy must be an object")
    candidate = dict(value)
    try:
        verify_sealed(candidate, label="GLM-5.2 resource reserve policy")
    except Glm52Error as exc:
        raise ResourcePolicyError(str(exc)) from exc
    expected_keys = {
        "schema", "status", "repo", "revision", "created_at", "input_seals",
        "policy", "derived", "prefetch_control", "provisional_control_limits",
        "activation", "notes", "seal_sha256",
    }
    if set(candidate) != expected_keys or candidate.get("schema") != SCHEMA \
            or candidate.get("status") != STATUS or candidate.get("repo") != OFFICIAL_REPO \
            or candidate.get("revision") != OFFICIAL_REVISION:
        raise ResourcePolicyError("resource reserve policy identity/fields are invalid")
    _created_at(candidate.get("created_at"))
    if candidate.get("input_seals") != {
        name: spec[3] for name, spec in sorted(EXACT_INPUTS.items())
    }:
        raise ResourcePolicyError("resource reserve policy input seals are not exact")
    expected = derive_policy(root, created_at=str(candidate["created_at"]))
    if candidate != expected:
        raise ResourcePolicyError(
            "resource reserve policy differs from its exact deterministic derivation"
        )
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--created-at")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = args.output or args.root / "GLM52_RESOURCE_RESERVE_POLICY.json"
        if args.command == "build":
            value = derive_policy(args.root, created_at=args.created_at)
            atomic_json(output, value)
            restored = validate_policy(read_sealed_json(output), root=args.root)
        else:
            restored = validate_policy(read_sealed_json(output), root=args.root)
    except (OSError, Glm52Error, TypeError) as exc:
        print(json.dumps({"status": "ERROR", "message": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "status": "PASS",
        "path": str(output.resolve()),
        "seal_sha256": restored["seal_sha256"],
        "required_free_disk_bytes": restored["derived"]["required_free_disk_bytes"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

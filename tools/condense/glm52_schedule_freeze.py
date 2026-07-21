#!/usr/bin/env python3.12
"""Pure, fail-closed post-autotune schedule freeze for GLM-5.2.

The preliminary schedule already freezes dependency membership.  Live Xet
autotuning is allowed to choose acquisition and steady transfer profiles, but
it is not allowed to rewrite a window's source or tensor ownership.  This
module joins those two independently sealed inputs under the expected campaign
contract and returns a producer-authenticated final schedule.

The implementation is intentionally side-effect free: it accepts in-memory
JSON objects, performs no network or model-body reads, and never writes the
production schedule path.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_state as state  # noqa: E402
import glm52_xet_live as xet_live  # noqa: E402
from glm52_common import (  # noqa: E402
    Glm52Error,
    REPO_ROOT,
    canonical,
    verify_sealed,
)


FINAL_SCHEMA = "hawking.glm52.streaming_schedule.v2"
FINAL_STATUS = "FROZEN_AFTER_XET_AUTOTUNE"
PRELIMINARY_SCHEMA = "hawking.glm52.streaming_schedule.v1"
PRELIMINARY_STATUS = "PRELIMINARY_DEPENDENCY_COMPLETE_PENDING_XET_AUTOTUNE"
RAW_RESULT_SEAL_EVIDENCE_KEY = "raw_xet_autotune_result_seal_sha256"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRELIMINARY_FIELDS = frozenset({
    "schema",
    "status",
    "repo",
    "revision",
    "planner_inputs",
    "pipeline",
    "window_count",
    "maximum_resident_shards_in_one_window",
    "maximum_simultaneous_shards_active_plus_prefetch_upper_bound",
    "source_shards_scheduled",
    "planned_refetches",
    "windows",
    "freeze_boundary",
    "seal_sha256",
})
_PRELIMINARY_WINDOW_FIELDS = frozenset({
    "window_id",
    "organ_ids",
    "source_shards",
    "source_shard_count",
    "carry_in_shards",
    "new_fetch_shards",
    "refetch_shards",
    "carry_out_shards",
    "evict_after_seal_shards",
    "new_fetch_logical_bytes",
    "resident_logical_bytes",
})
_PLANNER_INPUT_FIELDS = frozenset({
    "free_disk_bytes",
    "hard_floor_bytes",
    "largest_two_source_shards_bytes",
    "operational_reserve_bytes",
    "projected_three_complete_artifacts_bytes_0_98_plus_0_75_plus_0_50",
    "projected_evidence_bytes",
    "active_scratch_bytes",
    "usable_raw_window_bytes",
    "safety_fraction",
    "p99_allocated_proxy_uses_remote_logical_shard_bytes",
    "two_simultaneous_complete_raw_windows",
    "disk_limited_shards_per_window",
    "preliminary_target_shards_per_window",
})
_PIPELINE = {
    "n_minus_1": "verification/sealing/eviction; only carry-out bodies remain",
    "n": "active BF16 teacher/fit/pack/forward",
    "n_plus_1": "prefetch/reconstruction after measured admission",
    "fourth_window": "DISALLOWED_PENDING_MEASUREMENT",
}
_FREEZE_BOUNDARY = (
    "Window size/concurrency may change only after GLM52_XET_AUTOTUNE; tensor "
    "dependencies and one-fetch accounting remain immutable."
)
_RESULT_WRAPPER_FIELDS = frozenset({
    "campaign_id",
    "source_revision",
    "expected_contract_sha256",
    "evidence",
    "evidence_sha256",
    "producer_hmac_sha256",
})
_FINAL_FIELDS = frozenset({
    "schema",
    "status",
    "repo",
    "revision",
    "campaign_id",
    "source_revision",
    "expected_contract_sha256",
    "window_count",
    "source_shards_scheduled",
    "planned_refetches",
    "maximum_resident_shards_in_one_window",
    "maximum_simultaneous_shards_active_plus_prefetch_upper_bound",
    "selected_profile",
    "autotune_binding",
    "dependency_freeze",
    "windows",
    "evidence",
    "evidence_sha256",
    "producer_hmac_sha256",
    "seal_sha256",
})


class ScheduleFreezeError(Glm52Error):
    """A frozen input, authentication, or dependency invariant failed."""


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _clone(value: Any) -> Any:
    try:
        return json.loads(canonical(value).decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ScheduleFreezeError(f"value is not canonical JSON: {exc}") from exc


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ScheduleFreezeError(
            f"{label} fields differ: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) \
            or any(not isinstance(item, str) or not item or item != item.strip() for item in value) \
            or len(value) != len(set(value)):
        raise ScheduleFreezeError(f"{label} must be a unique non-empty string list")
    return list(value)


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScheduleFreezeError(f"{label} must be a nonnegative integer")
    return value


def _validated_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = state._validate_expected_contract(dict(contract))
    except (state.StateError, TypeError, ValueError) as exc:
        raise ScheduleFreezeError(f"expected campaign contract is invalid: {exc}") from exc
    if value["source_revision"] != xet_live.REVISION:
        raise ScheduleFreezeError("campaign contract source revision differs from live Xet")
    return value


def _expected_artifact_policy(
    contract: Mapping[str, Any],
    *,
    gate: str,
    label: str,
    expected_path: str,
) -> dict[str, Any]:
    raw = contract.get("state_gates", {}).get(gate, {}).get(
        "required_artifacts", {}
    ).get(label)
    if not isinstance(raw, dict):
        raise ScheduleFreezeError(
            f"expected campaign contract lacks {gate}.{label} artifact policy"
        )
    policy = dict(raw)
    if policy.get("path") != expected_path or not _is_sha256(
        policy.get("expected_seal_sha256")
    ):
        raise ScheduleFreezeError(
            f"expected campaign contract does not freeze exact {expected_path} bytes"
        )
    return policy


def _validate_plan(
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    root: Path,
    rebuild_plan: bool,
) -> dict[str, Any]:
    policy = _expected_artifact_policy(
        contract,
        gate="AUTOTUNE_XET",
        label="xet_autotune_plan",
        expected_path="GLM52_XET_AUTOTUNE_PLAN.json",
    )
    try:
        value = xet_live.validate_live_plan(
            plan,
            root=root,
            rebuild=rebuild_plan,
        )
    except (Glm52Error, KeyError, TypeError, ValueError) as exc:
        raise ScheduleFreezeError(f"Xet autotune plan validation failed: {exc}") from exc
    if value.get("seal_sha256") != policy["expected_seal_sha256"] \
            or value.get("schema") != policy.get("expected_schema") \
            or value.get("status") not in policy.get("allowed_statuses", []):
        raise ScheduleFreezeError("Xet autotune plan differs from the expected contract")
    return value


def _validate_preliminary_schedule(
    preliminary: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    policy = _expected_artifact_policy(
        contract,
        gate="ASSEMBLE_ARTIFACT",
        label="preliminary_streaming_schedule",
        expected_path="GLM52_STREAMING_SCHEDULE_PRE_AUTOTUNE.json",
    )
    try:
        value = verify_sealed(dict(preliminary), label="pre-autotune streaming schedule")
    except Glm52Error as exc:
        raise ScheduleFreezeError(str(exc)) from exc
    _exact_fields(value, _PRELIMINARY_FIELDS, "pre-autotune streaming schedule")
    if value.get("schema") != PRELIMINARY_SCHEMA \
            or value.get("status") != PRELIMINARY_STATUS \
            or value.get("repo") != xet_live.REPO_ID \
            or value.get("revision") != contract["source_revision"]:
        raise ScheduleFreezeError("pre-autotune schedule schema/status/identity mismatch")
    if value["seal_sha256"] != policy["expected_seal_sha256"] \
            or value["schema"] != policy.get("expected_schema") \
            or value["status"] not in policy.get("allowed_statuses", []):
        raise ScheduleFreezeError("pre-autotune schedule differs from the expected contract")
    if value.get("pipeline") != _PIPELINE or value.get("freeze_boundary") != _FREEZE_BOUNDARY:
        raise ScheduleFreezeError("pre-autotune schedule weakens its immutable freeze boundary")
    planner = value.get("planner_inputs")
    if not isinstance(planner, dict) or set(planner) != _PLANNER_INPUT_FIELDS:
        raise ScheduleFreezeError("pre-autotune planner input schema mismatch")
    for key, item in planner.items():
        if key == "safety_fraction":
            if isinstance(item, bool) or not isinstance(item, (int, float)) \
                    or not 0 < float(item) <= 1:
                raise ScheduleFreezeError("pre-autotune safety_fraction is invalid")
        elif key == "two_simultaneous_complete_raw_windows":
            if item is not True:
                raise ScheduleFreezeError("pre-autotune two-window bound is not frozen true")
        else:
            _nonnegative_int(item, f"planner_inputs.{key}")

    expected_windows = contract["window_schedule"]
    windows = value.get("windows")
    if not isinstance(windows, list) or len(windows) != len(expected_windows) \
            or value.get("window_count") != len(expected_windows):
        raise ScheduleFreezeError("pre-autotune window inventory differs from the contract")
    shard_bytes = {
        row["path"]: row["logical_bytes"] for row in contract["source"]["shards"]
    }
    seen_organs: set[str] = set()
    new_fetches: list[str] = []
    refetches = 0
    maximum_resident = 0
    for index, (window, expected) in enumerate(zip(windows, expected_windows)):
        if not isinstance(window, dict):
            raise ScheduleFreezeError(f"pre-autotune window {index} is not an object")
        _exact_fields(window, _PRELIMINARY_WINDOW_FIELDS, f"pre-autotune window {index}")
        organs = _string_list(window.get("organ_ids"), f"window {index}.organ_ids")
        if not organs or seen_organs.intersection(organs):
            raise ScheduleFreezeError("pre-autotune organ ownership is empty or duplicated")
        seen_organs.update(organs)
        normalized = {
            "schedule_index": index,
            "window_id": window.get("window_id"),
            "source_shards": _string_list(
                window.get("source_shards"), f"window {index}.source_shards"
            ),
            "carry_in_shards": _string_list(
                window.get("carry_in_shards"), f"window {index}.carry_in_shards"
            ),
            "new_fetch_shards": _string_list(
                window.get("new_fetch_shards"), f"window {index}.new_fetch_shards"
            ),
            "refetch_shards": _string_list(
                window.get("refetch_shards"), f"window {index}.refetch_shards"
            ),
            "carry_out_shards": _string_list(
                window.get("carry_out_shards"), f"window {index}.carry_out_shards"
            ),
            "evict_shards": _string_list(
                window.get("evict_after_seal_shards"),
                f"window {index}.evict_after_seal_shards",
            ),
            "tensor_set": _clone(expected["tensor_set"]),
        }
        if normalized != expected:
            raise ScheduleFreezeError(
                f"pre-autotune window {index} changed frozen dependency membership"
            )
        if any(path not in shard_bytes for path in normalized["source_shards"]):
            raise ScheduleFreezeError(f"pre-autotune window {index} references an unknown shard")
        source_count = len(normalized["source_shards"])
        if window.get("source_shard_count") != source_count \
                or window.get("resident_logical_bytes") != sum(
                    shard_bytes[path] for path in normalized["source_shards"]
                ) \
                or window.get("new_fetch_logical_bytes") != sum(
                    shard_bytes[path] for path in normalized["new_fetch_shards"]
                ):
            raise ScheduleFreezeError(
                f"pre-autotune window {index} source byte/count accounting mismatch"
            )
        new_fetches.extend(normalized["new_fetch_shards"])
        refetches += len(normalized["refetch_shards"])
        maximum_resident = max(maximum_resident, source_count)
    expected_paths = set(shard_bytes)
    if len(new_fetches) != len(set(new_fetches)) or set(new_fetches) != expected_paths:
        raise ScheduleFreezeError("pre-autotune schedule does not new-fetch every shard once")
    if value.get("source_shards_scheduled") != len(expected_paths) \
            or value.get("planned_refetches") != refetches \
            or value.get("maximum_resident_shards_in_one_window") != maximum_resident \
            or value.get(
                "maximum_simultaneous_shards_active_plus_prefetch_upper_bound"
            ) != 2 * maximum_resident \
            or planner.get("preliminary_target_shards_per_window") != maximum_resident \
            or planner.get("disk_limited_shards_per_window", -1) < maximum_resident:
        raise ScheduleFreezeError("pre-autotune schedule summary accounting mismatch")
    return value


def _validate_attested_result(
    result: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    auth: state.TelegramAuthConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(auth, state.TelegramAuthConfig):
        raise ScheduleFreezeError("producer authentication is required")
    if auth.expected_chat_identity_digest != contract["expected_chat_identity_digest"]:
        raise ScheduleFreezeError("producer authentication chat identity differs from contract")
    try:
        outer = verify_sealed(dict(result), label="producer-attested Xet result")
    except Glm52Error as exc:
        raise ScheduleFreezeError(str(exc)) from exc
    raw_fields = frozenset(xet_live.AUTOTUNE_RESULT_FIELDS)
    expected_fields = (raw_fields - {"seal_sha256"}) | _RESULT_WRAPPER_FIELDS | {
        "seal_sha256"
    }
    _exact_fields(outer, expected_fields, "producer-attested Xet result")
    for key, expected in (
        ("campaign_id", contract["campaign_id"]),
        ("source_revision", contract["source_revision"]),
        ("expected_contract_sha256", contract["seal_sha256"]),
    ):
        if outer.get(key) != expected:
            raise ScheduleFreezeError(f"producer-attested Xet result {key} mismatch")
    evidence = outer.get("evidence")
    if not isinstance(evidence, dict) or not evidence \
            or outer.get("evidence_sha256") != _sha256(evidence):
        raise ScheduleFreezeError("producer-attested Xet result evidence hash mismatch")
    raw_seal = evidence.get(RAW_RESULT_SEAL_EVIDENCE_KEY)
    if not _is_sha256(raw_seal):
        raise ScheduleFreezeError("producer-attested Xet result lacks its original raw seal")
    if evidence.get("xet_autotune_plan_seal_sha256") != plan["seal_sha256"]:
        raise ScheduleFreezeError("producer-attested Xet result evidence plan seal mismatch")
    if not _is_sha256(evidence.get("controller_anchor_sha256")):
        raise ScheduleFreezeError(
            "producer-attested Xet result lacks a valid controller anchor"
        )
    producer_body = {
        key: _clone(item) for key, item in outer.items()
        if key not in {"seal_sha256", "producer_hmac_sha256"}
    }
    if not auth.verify(
        {
            "schema": "hawking.glm52.evidence_producer_auth.v1",
            "artifact": producer_body,
        },
        outer.get("producer_hmac_sha256"),
    ):
        raise ScheduleFreezeError("producer-attested Xet result HMAC authentication failed")
    raw_body = {
        key: _clone(outer[key]) for key in raw_fields if key != "seal_sha256"
    }
    if _sha256(raw_body) != raw_seal:
        raise ScheduleFreezeError("producer-attested Xet result raw seal reconstruction failed")
    raw_result = {**raw_body, "seal_sha256": raw_seal}
    try:
        validated = xet_live.validate_autotune_result(raw_result, plan=plan)
    except (Glm52Error, KeyError, TypeError, ValueError) as exc:
        raise ScheduleFreezeError(f"raw live-Xet result validation failed: {exc}") from exc
    return outer, validated


def attest_xet_autotune_result(
    raw_result: Mapping[str, Any],
    xet_autotune_plan: Mapping[str, Any],
    expected_campaign_contract: Mapping[str, Any],
    *,
    auth: state.TelegramAuthConfig,
    controller_anchor_sha256: str,
    root: Path = REPO_ROOT,
    rebuild_plan: bool = True,
) -> dict[str, Any]:
    """Validate and producer-attest one exact raw live-Xet result.

    The controller anchor is mandatory and is covered by both the evidence hash
    and producer HMAC.  Offline tests use an explicit synthetic digest rather
    than weakening the production artifact schema.
    """
    contract = _validated_contract(expected_campaign_contract)
    plan = _validate_plan(
        xet_autotune_plan,
        contract,
        root=Path(root),
        rebuild_plan=rebuild_plan,
    )
    if not isinstance(auth, state.TelegramAuthConfig) \
            or auth.expected_chat_identity_digest != contract["expected_chat_identity_digest"]:
        raise ScheduleFreezeError("producer authentication differs from the campaign contract")
    if not _is_sha256(controller_anchor_sha256):
        raise ScheduleFreezeError("controller_anchor_sha256 must be 64 lowercase hex")
    try:
        validated = xet_live.validate_autotune_result(raw_result, plan=plan)
    except (Glm52Error, KeyError, TypeError, ValueError) as exc:
        raise ScheduleFreezeError(f"raw live-Xet result validation failed: {exc}") from exc
    evidence = {
        "producer": "glm52-live-worker",
        RAW_RESULT_SEAL_EVIDENCE_KEY: validated["seal_sha256"],
        "xet_autotune_plan_seal_sha256": plan["seal_sha256"],
    }
    evidence["controller_anchor_sha256"] = controller_anchor_sha256
    body = {
        key: _clone(item) for key, item in validated.items() if key != "seal_sha256"
    }
    body.update({
        "campaign_id": contract["campaign_id"],
        "source_revision": contract["source_revision"],
        "expected_contract_sha256": contract["seal_sha256"],
        "evidence": evidence,
        "evidence_sha256": _sha256(evidence),
    })
    try:
        attested = state.seal_producer_authenticated_evidence(body, auth=auth)
    except state.StateError as exc:
        raise ScheduleFreezeError(f"Xet result producer authentication failed: {exc}") from exc
    _validate_attested_result(
        attested,
        plan=plan,
        contract=contract,
        auth=auth,
    )
    return attested


def _final_windows(
    preliminary: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for pre, expected in zip(preliminary["windows"], contract["window_schedule"]):
        result.append({
            "schedule_index": expected["schedule_index"],
            "window_id": expected["window_id"],
            "organ_ids": _clone(pre["organ_ids"]),
            "source_shards": _clone(expected["source_shards"]),
            "source_shard_count": pre["source_shard_count"],
            "carry_in_shards": _clone(expected["carry_in_shards"]),
            "new_fetch_shards": _clone(expected["new_fetch_shards"]),
            "refetch_shards": _clone(expected["refetch_shards"]),
            "carry_out_shards": _clone(expected["carry_out_shards"]),
            "evict_shards": _clone(expected["evict_shards"]),
            "tensor_set": _clone(expected["tensor_set"]),
            "new_fetch_logical_bytes": pre["new_fetch_logical_bytes"],
            "resident_logical_bytes": pre["resident_logical_bytes"],
        })
    return result


def freeze_schedule(
    preliminary_schedule: Mapping[str, Any],
    xet_autotune_plan: Mapping[str, Any],
    producer_attested_xet_result: Mapping[str, Any],
    expected_campaign_contract: Mapping[str, Any],
    *,
    auth: state.TelegramAuthConfig,
    root: Path = REPO_ROOT,
    rebuild_plan: bool = True,
) -> dict[str, Any]:
    """Return, but never write, the authenticated post-autotune schedule."""
    contract = _validated_contract(expected_campaign_contract)
    plan = _validate_plan(
        xet_autotune_plan,
        contract,
        root=Path(root),
        rebuild_plan=rebuild_plan,
    )
    preliminary = _validate_preliminary_schedule(preliminary_schedule, contract)
    attested, raw_result = _validate_attested_result(
        producer_attested_xet_result,
        plan=plan,
        contract=contract,
        auth=auth,
    )
    selected_profile = {
        "acquisition": _clone(raw_result["selections"]["acquisition"]),
        "steady": _clone(raw_result["selections"]["steady"]),
    }
    windows = _final_windows(preliminary, contract)
    window_schedule_sha256 = _sha256(contract["window_schedule"])
    evidence = {
        "producer": "glm52-schedule-freezer",
        "xet_autotune_attested_seal_sha256": attested["seal_sha256"],
        "xet_autotune_raw_result_seal_sha256": attested["evidence"][
            RAW_RESULT_SEAL_EVIDENCE_KEY
        ],
        "xet_autotune_plan_seal_sha256": plan["seal_sha256"],
        "preliminary_schedule_seal_sha256": preliminary["seal_sha256"],
        "expected_contract_sha256": contract["seal_sha256"],
        "window_schedule_sha256": window_schedule_sha256,
        "network_access": False,
        "model_body_bytes_read": 0,
        "production_artifact_written": False,
    }
    body = {
        "schema": FINAL_SCHEMA,
        "status": FINAL_STATUS,
        "repo": xet_live.REPO_ID,
        "revision": contract["source_revision"],
        "campaign_id": contract["campaign_id"],
        "source_revision": contract["source_revision"],
        "expected_contract_sha256": contract["seal_sha256"],
        "window_count": len(windows),
        "source_shards_scheduled": preliminary["source_shards_scheduled"],
        "planned_refetches": preliminary["planned_refetches"],
        "maximum_resident_shards_in_one_window": preliminary[
            "maximum_resident_shards_in_one_window"
        ],
        "maximum_simultaneous_shards_active_plus_prefetch_upper_bound": preliminary[
            "maximum_simultaneous_shards_active_plus_prefetch_upper_bound"
        ],
        "selected_profile": selected_profile,
        "autotune_binding": {
            "xet_autotune_result_seal_sha256": attested["seal_sha256"],
            "xet_autotune_plan_seal_sha256": plan["seal_sha256"],
            "preliminary_schedule_seal_sha256": preliminary["seal_sha256"],
            "selected_profile_sha256": _sha256(selected_profile),
        },
        "dependency_freeze": {
            "status": "IMMUTABLE_FROM_PRE_AUTOTUNE_EXPECTED_CONTRACT",
            "window_schedule_sha256": window_schedule_sha256,
            "window_count": len(windows),
            "source_dependency_membership_changed": False,
            "tensor_ownership_changed": False,
        },
        "windows": windows,
        "evidence": evidence,
        "evidence_sha256": _sha256(evidence),
    }
    try:
        final = state.seal_producer_authenticated_evidence(body, auth=auth)
    except state.StateError as exc:
        raise ScheduleFreezeError(f"final schedule producer authentication failed: {exc}") from exc
    _exact_fields(final, _FINAL_FIELDS, "frozen streaming schedule")
    return final


def validate_frozen_schedule(
    schedule: Mapping[str, Any],
    preliminary_schedule: Mapping[str, Any],
    xet_autotune_plan: Mapping[str, Any],
    producer_attested_xet_result: Mapping[str, Any],
    expected_campaign_contract: Mapping[str, Any],
    *,
    auth: state.TelegramAuthConfig,
    root: Path = REPO_ROOT,
    rebuild_plan: bool = True,
) -> dict[str, Any]:
    """Validate a final schedule by rebuilding its entire deterministic body."""
    try:
        value = verify_sealed(dict(schedule), label="frozen streaming schedule")
    except Glm52Error as exc:
        raise ScheduleFreezeError(str(exc)) from exc
    _exact_fields(value, _FINAL_FIELDS, "frozen streaming schedule")
    expected = freeze_schedule(
        preliminary_schedule,
        xet_autotune_plan,
        producer_attested_xet_result,
        expected_campaign_contract,
        auth=auth,
        root=root,
        rebuild_plan=rebuild_plan,
    )
    if canonical(value) != canonical(expected):
        raise ScheduleFreezeError(
            "frozen streaming schedule differs from the authenticated deterministic freeze"
        )
    return value


__all__ = [
    "FINAL_SCHEMA",
    "FINAL_STATUS",
    "RAW_RESULT_SEAL_EVIDENCE_KEY",
    "ScheduleFreezeError",
    "attest_xet_autotune_result",
    "freeze_schedule",
    "validate_frozen_schedule",
]

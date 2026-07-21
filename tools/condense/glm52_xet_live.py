#!/usr/bin/env python3.12
"""Authority-gated, body-file-free live Xet trials for GLM-5.2.

The offline planner in :mod:`glm52_xet_autotune` is the source of truth for
every byte range and trial knob.  This module adds the execution boundary but
does not weaken it:

* a caller must present a sealed, independently verified live capability;
* every trial gets a fresh subprocess so ``HF_XET_*`` is read before import;
* the child uses ``hf_xet.XetSession`` range streams and hashes bytes in memory;
* public-Hub token refresh headers are constructed in the child and never
  appear in a spec, result, log summary, argv, or environment variable;
* an independent monotonic network counter enforces a predeclared byte cap;
* missing Xet retry or amplification measurements remain ``null`` and make a
  trial ineligible rather than being silently converted to zero; and
* this module contains no cache deletion or garbage-collection operation.

The default Darwin sampler and network counter are intentionally conservative.
Tests inject strict fakes and never open a socket or consume model bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from glm52_common import (  # noqa: E402
    Glm52Error,
    REPO_ROOT,
    canonical,
    seal,
    sha256_file,
    verify_sealed,
)
import glm52_xet_autotune as autotune  # noqa: E402


REPO_ID = autotune.REPO_ID
REVISION = autotune.REVISION
PINNED_VERSIONS = dict(autotune.PINNED_VERSIONS)

TRIAL_SPEC_SCHEMA = "hawking.glm52.xet_live_trial_spec.v1"
TRIAL_RESULT_SCHEMA = "hawking.glm52.xet_live_trial_result.v1"
CHILD_RESULT_SCHEMA = "hawking.glm52.xet_live_child_result.v1"
CAPABILITY_SCHEMA = "hawking.glm52.xet_live_capability.v1"
LARGEST_EVIDENCE_SCHEMA = "hawking.glm52.xet_largest_shard_validation.v1"
AUTOTUNE_RESULT_SCHEMA = "hawking.glm52.xet_autotune_result.v1"
RESOURCE_SAMPLE_SCHEMA = "hawking.glm52.darwin_resource_sample.v1"
NETWORK_COUNTER_SCHEMA = "hawking.glm52.darwin_network_counter.v1"
CHILD_PROTOCOL = "hawking.glm52.xet_live_child_protocol.v1"

EXECUTE_ENV = "HAWKING_GLM52_XET_EXECUTE"
CHILD_ENV = "HAWKING_GLM52_XET_CHILD"
SPEC_SEAL_ENV = "HAWKING_GLM52_XET_SPEC_SEAL"
RESERVED_XET_ENV = {
    "HF_XET_LOG_DEST": "stderr",
    "HF_XET_LOG_FORMAT": "json",
}
REQUIRED_CHILD_ENV = {
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
ALLOWED_PLAN_ENV = frozenset({
    "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS",
    "HF_XET_FIXED_DOWNLOAD_CONCURRENCY",
    "HF_XET_HIGH_PERFORMANCE",
    "HF_XET_CHUNK_CACHE_SIZE_BYTES",
})
RUNTIME_CONFIG_FIELDS = tuple(autotune._RUNTIME_FIELDS) + ("log.dest", "log.format")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_TRIAL_RE = re.compile(r"^[A-Z0-9][A-Z0-9_]{0,63}$")

SPEC_FIELDS = frozenset({
    "schema", "status", "repo", "revision", "plan_seal_sha256",
    "capability_seal_sha256", "trial", "targets", "network_budget",
    "execution", "public_hub_auth", "credentials_serialized", "seal_sha256",
})
RESULT_FIELDS = frozenset({
    "schema", "status", "repo", "revision", "plan_seal_sha256",
    "spec_seal_sha256", "capability_seal_sha256", "trial_binding",
    "runtime", "public_hub_auth", "range_results", "stream_measurement",
    "network_accounting", "xet_log_evidence", "resource_observations",
    "body_persistence", "process", "failure", "seal_sha256",
})
AUTOTUNE_RESULT_FIELDS = frozenset({
    "schema", "status", "repo", "revision", "bindings", "coverage",
    "selections", "selected_profile", "largest_shard_validations",
    "network_budget", "claim_boundary", "seal_sha256",
})
CAPABILITY_FIELDS = frozenset({
    "schema", "status", "repo", "revision", "plan_seal_sha256", "trial_id",
    "allowed_kind", "max_network_bytes", "controller", "expires_unix_ns",
    "credentials_serialized", "seal_sha256",
})
BASE_RESOURCE_FIELDS = frozenset({
    "schema", "sampled_monotonic_ns", "pid", "swap_used_bytes", "swapouts",
    "thermal_warning", "free_disk_bytes", "available_ram_bytes", "cpu_percent",
    "process_rss_bytes", "disk_write_bytes_per_second",
})
OPTIONAL_XET_SAMPLE_FIELDS = frozenset({
    "reconstruction_latency_seconds", "retry_rate", "temporary_amplification_ratio",
})


class LiveCapabilityVerifier(Protocol):
    """Executor-owned verification against live controller/key material."""

    def verify_live_capability(
        self,
        capability: Mapping[str, Any],
        *,
        plan: Mapping[str, Any],
        spec: Mapping[str, Any],
    ) -> bool:
        """Return exactly ``True`` only while the capability remains live."""


class ResourceSampler(Protocol):
    """Strict resource sampler used by the supervising parent process."""

    def sample(self, pid: int) -> Mapping[str, Any]:
        """Return one exact resource sample or raise; never invent a value."""


class NetworkBudgetCounter(Protocol):
    """Monotonic counter whose deltas conservatively bound trial network I/O."""

    def snapshot(self) -> int:
        """Return a nonnegative cumulative byte count."""

    def evidence(self) -> Mapping[str, Any]:
        """Describe counter method and scope without secret material."""


class CommandRunner(Protocol):
    def __call__(self, argv: Sequence[str]) -> str:
        """Run a read-only command and return stdout, raising on failure."""


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Glm52Error(f"{label} must be an integer >= {minimum}")
    return value


def _require_number(
    value: Any,
    label: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Glm52Error(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum \
            or (maximum is not None and result > maximum):
        raise Glm52Error(f"{label} is outside its allowed finite range")
    return result


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise Glm52Error(
            f"{label} fields differ: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )


def _validate_plan_environment(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise Glm52Error("trial environment must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if key not in ALLOWED_PLAN_ENV:
            raise Glm52Error(f"trial contains an unapproved environment knob: {key!r}")
        if not isinstance(item, str) or not item or "\x00" in item:
            raise Glm52Error(f"trial environment value is invalid: {key!r}")
        result[str(key)] = item
    return result


def validate_live_plan(
    plan: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    rebuild: bool = True,
) -> dict[str, Any]:
    """Delegate immutable-plan verification to the planner, then check live shape."""
    value = autotune.verify_plan(plan, root=root, rebuild=rebuild)
    matrix = value.get("trial_matrix")
    if not isinstance(matrix, list) or len(matrix) != 12:
        raise Glm52Error("live execution requires exactly 12 sealed planner trials")
    ids = [row.get("trial_id") for row in matrix if isinstance(row, Mapping)]
    if len(ids) != 12 or len(set(ids)) != 12 \
            or any(not isinstance(item, str) or SAFE_TRIAL_RE.fullmatch(item) is None for item in ids):
        raise Glm52Error("sealed trial IDs are incomplete, duplicated, or unsafe")
    required = {f"FILES_{value:02d}" for value in autotune.REQUIRED_FILE_SETTINGS}
    if not required <= set(ids):
        raise Glm52Error("sealed trial matrix lacks a required file-concurrency setting")
    for row in matrix:
        _validate_plan_environment(row.get("environment"))
    return value


def _target_from_range(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "range_id_sha256": row.get("range_id_sha256"),
        "path": row.get("path"),
        "xet_hash": row.get("xet_hash"),
        "lfs_sha256": row.get("lfs_sha256"),
        "file_bytes": row.get("file_bytes"),
        "start": row.get("start"),
        "end": row.get("end"),
        "length": row.get("length"),
    }
    if any(not _is_sha256(fields[key]) for key in (
        "range_id_sha256", "xet_hash", "lfs_sha256",
    )):
        raise Glm52Error("sealed target contains an invalid content identity")
    if not isinstance(fields["path"], str) or not fields["path"]:
        raise Glm52Error("sealed target path is invalid")
    file_bytes = _require_int(fields["file_bytes"], "target.file_bytes", minimum=1)
    start = _require_int(fields["start"], "target.start")
    end = _require_int(fields["end"], "target.end", minimum=1)
    length = _require_int(fields["length"], "target.length", minimum=1)
    if end - start != length or end > file_bytes:
        raise Glm52Error("sealed target byte interval is invalid")
    return fields


def _matrix_trial(plan: Mapping[str, Any], trial_id: str) -> dict[str, Any]:
    matches = [
        dict(row) for row in plan.get("trial_matrix", [])
        if isinstance(row, Mapping) and row.get("trial_id") == trial_id
    ]
    if len(matches) != 1:
        raise Glm52Error(f"trial ID is not unique in sealed plan: {trial_id!r}")
    return matches[0]


def _budget_block(
    plan: Mapping[str, Any],
    *,
    consumed_bytes: int,
    trial_cap_bytes: int,
    planned_payload_bytes: int,
) -> dict[str, int]:
    consumed = _require_int(consumed_bytes, "campaign_consumed_bytes")
    cap = _require_int(trial_cap_bytes, "trial_network_cap_bytes", minimum=1)
    payload = _require_int(planned_payload_bytes, "planned_payload_bytes", minimum=1)
    hard = _require_int(
        plan.get("network_budget", {}).get("hard_cap_bytes"),
        "plan.network_budget.hard_cap_bytes",
        minimum=1,
    )
    if consumed > hard or cap > hard - consumed:
        raise Glm52Error("trial cap does not fit the remaining campaign network budget")
    if payload > cap:
        raise Glm52Error("planned payload does not fit the trial network cap")
    return {
        "campaign_hard_cap_bytes": hard,
        "campaign_consumed_before_bytes": consumed,
        "campaign_remaining_before_bytes": hard - consumed,
        "trial_network_cap_bytes": cap,
        "planned_payload_bytes": payload,
    }


def _execution_block(timeout_seconds: float, sample_interval_seconds: float) -> dict[str, Any]:
    timeout = _require_number(timeout_seconds, "timeout_seconds", minimum=1.0)
    interval = _require_number(
        sample_interval_seconds,
        "sample_interval_seconds",
        minimum=0.05,
        maximum=timeout,
    )
    return {
        "fresh_subprocess_required": True,
        "child_protocol": CHILD_PROTOCOL,
        "timeout_seconds": timeout,
        "sample_interval_seconds": interval,
        "stream_api": "hf_xet.XetSession.new_download_stream_group.download_stream",
        "body_sink": "IN_MEMORY_SHA256_ONLY",
        "body_file_writes_allowed": False,
        "destructive_gc_allowed": False,
    }


def build_trial_spec(
    plan: Mapping[str, Any],
    trial_id: str,
    *,
    capability_seal_sha256: str,
    campaign_consumed_bytes: int,
    trial_network_cap_bytes: int,
    timeout_seconds: float = 900.0,
    sample_interval_seconds: float = 1.0,
    root: Path = REPO_ROOT,
    rebuild_plan: bool = True,
) -> dict[str, Any]:
    """Project one sealed planner row into an exact live child specification."""
    value = validate_live_plan(plan, root=root, rebuild=rebuild_plan)
    if not _is_sha256(capability_seal_sha256):
        raise Glm52Error("capability seal is invalid")
    trial = _matrix_trial(value, trial_id)
    ranges = {
        row["range_id_sha256"]: row
        for row in value["range_strategy"]["body_ranges"]
    }
    ordered_ids = trial.get("ordered_range_ids")
    if not isinstance(ordered_ids, list) or len(ordered_ids) != trial.get("range_count") \
            or any(item not in ranges for item in ordered_ids):
        raise Glm52Error("trial ordered range IDs do not resolve exactly")
    targets = [_target_from_range(ranges[item]) for item in ordered_ids]
    planned = sum(item["length"] for item in targets)
    if planned != trial.get("planned_payload_bytes"):
        raise Glm52Error("trial planned payload differs from its exact range projection")
    environment = _validate_plan_environment(trial.get("environment"))
    projected_trial = {**trial, "environment": environment}
    return seal({
        "schema": TRIAL_SPEC_SCHEMA,
        "status": "READY_FOR_AUTHORIZED_LIVE_EXECUTION",
        "repo": REPO_ID,
        "revision": REVISION,
        "plan_seal_sha256": value["seal_sha256"],
        "capability_seal_sha256": capability_seal_sha256,
        "trial": projected_trial,
        "targets": targets,
        "network_budget": _budget_block(
            value,
            consumed_bytes=campaign_consumed_bytes,
            trial_cap_bytes=trial_network_cap_bytes,
            planned_payload_bytes=planned,
        ),
        "execution": _execution_block(timeout_seconds, sample_interval_seconds),
        "public_hub_auth": {
            "repository_visibility": "PUBLIC",
            "hub_token_mode": "EXPLICIT_TOKEN_FALSE",
            "xet_access_token_source": "HUB_READ_TOKEN_REFRESH_ROUTE_IN_CHILD",
            "authorization_header_serialized": False,
            "refresh_headers_serialized": False,
        },
        "credentials_serialized": False,
    })


def build_largest_validation_spec(
    plan: Mapping[str, Any],
    *,
    lane: str,
    selected_trial_id: str,
    capability_seal_sha256: str,
    campaign_consumed_bytes: int,
    trial_network_cap_bytes: int,
    timeout_seconds: float = 1800.0,
    sample_interval_seconds: float = 1.0,
    root: Path = REPO_ROOT,
    rebuild_plan: bool = True,
) -> dict[str, Any]:
    """Build one of the two mandatory full largest-shard validation specs."""
    value = validate_live_plan(plan, root=root, rebuild=rebuild_plan)
    if lane not in {"acquisition", "steady"}:
        raise Glm52Error("largest-shard validation lane must be acquisition or steady")
    if not _is_sha256(capability_seal_sha256):
        raise Glm52Error("capability seal is invalid")
    selected = _matrix_trial(value, selected_trial_id)
    largest = value.get("largest_shard_validation")
    if not isinstance(largest, Mapping):
        raise Glm52Error("plan largest-shard validation identity is absent")
    size = _require_int(largest.get("bytes"), "largest_shard.bytes", minimum=1)
    identity = {
        "schema": "hawking.glm52.xet_largest_range_identity.v1",
        "path": largest.get("path"),
        "xet_hash": largest.get("xet_hash"),
        "lfs_sha256": largest.get("lfs_sha256"),
        "start": 0,
        "end": size,
        "length": size,
    }
    target = _target_from_range({
        "range_id_sha256": _canonical_sha256(identity),
        "path": identity["path"],
        "xet_hash": identity["xet_hash"],
        "lfs_sha256": identity["lfs_sha256"],
        "file_bytes": size,
        "start": 0,
        "end": size,
        "length": size,
    })
    pass_index = 0 if lane == "acquisition" else 1
    trial = {
        "ordinal": 12 + pass_index,
        "trial_id": f"LARGEST_{lane.upper()}",
        "kind": "FULL_LARGEST_SHARD_VALIDATION",
        "validation_lane": lane,
        "validation_pass": largest.get("passes", [None, None])[pass_index],
        "selected_profile_trial_id": selected_trial_id,
        "caller_concurrent_shard_streams": 1,
        "environment": _validate_plan_environment(selected.get("environment")),
        "range_count": 1,
        "ordered_range_ids": [target["range_id_sha256"]],
        "ordered_range_ids_sha256": _canonical_sha256([target["range_id_sha256"]]),
        "planned_payload_bytes": size,
        "selected_plan_trial_sha256": _canonical_sha256(selected),
    }
    return seal({
        "schema": TRIAL_SPEC_SCHEMA,
        "status": "READY_FOR_AUTHORIZED_LIVE_EXECUTION",
        "repo": REPO_ID,
        "revision": REVISION,
        "plan_seal_sha256": value["seal_sha256"],
        "capability_seal_sha256": capability_seal_sha256,
        "trial": trial,
        "targets": [target],
        "network_budget": _budget_block(
            value,
            consumed_bytes=campaign_consumed_bytes,
            trial_cap_bytes=trial_network_cap_bytes,
            planned_payload_bytes=size,
        ),
        "execution": _execution_block(timeout_seconds, sample_interval_seconds),
        "public_hub_auth": {
            "repository_visibility": "PUBLIC",
            "hub_token_mode": "EXPLICIT_TOKEN_FALSE",
            "xet_access_token_source": "HUB_READ_TOKEN_REFRESH_ROUTE_IN_CHILD",
            "authorization_header_serialized": False,
            "refresh_headers_serialized": False,
        },
        "credentials_serialized": False,
    })


def validate_trial_spec(spec: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    value = verify_sealed(dict(spec), label="live Xet trial spec")
    _exact_fields(value, SPEC_FIELDS, "live Xet trial spec")
    if value.get("schema") != TRIAL_SPEC_SCHEMA \
            or value.get("status") != "READY_FOR_AUTHORIZED_LIVE_EXECUTION":
        raise Glm52Error("live Xet trial spec schema/status mismatch")
    if value.get("repo") != REPO_ID or value.get("revision") != REVISION \
            or value.get("plan_seal_sha256") != plan.get("seal_sha256"):
        raise Glm52Error("live Xet trial spec immutable identity mismatch")
    if not _is_sha256(value.get("capability_seal_sha256")) \
            or value.get("credentials_serialized") is not False:
        raise Glm52Error("live Xet trial spec capability/credential boundary is invalid")
    trial = value.get("trial")
    if not isinstance(trial, Mapping):
        raise Glm52Error("live Xet trial projection is absent")
    trial_id = trial.get("trial_id")
    kind = trial.get("kind")
    if not isinstance(trial_id, str) or SAFE_TRIAL_RE.fullmatch(trial_id) is None:
        raise Glm52Error("live Xet trial ID is unsafe")
    targets = value.get("targets")
    if not isinstance(targets, list) or not targets:
        raise Glm52Error("live Xet trial has no targets")
    normalized = [_target_from_range(item) for item in targets if isinstance(item, Mapping)]
    if len(normalized) != len(targets) or normalized != targets:
        raise Glm52Error("live Xet targets are not canonical exact projections")
    ordered = [item["range_id_sha256"] for item in targets]
    if ordered != trial.get("ordered_range_ids") \
            or _canonical_sha256(ordered) != trial.get("ordered_range_ids_sha256") \
            or len(targets) != trial.get("range_count") \
            or sum(item["length"] for item in targets) != trial.get("planned_payload_bytes"):
        raise Glm52Error("live Xet target order/count/bytes differ from trial projection")
    if kind == "BOUNDED_XET_BODY_RANGE":
        expected = _matrix_trial(plan, trial_id)
        if dict(trial) != expected:
            raise Glm52Error("live Xet trial projection differs from sealed matrix")
        plan_ranges = {
            item["range_id_sha256"]: _target_from_range(item)
            for item in plan["range_strategy"]["body_ranges"]
        }
        if any(plan_ranges.get(item["range_id_sha256"]) != item for item in targets):
            raise Glm52Error("live Xet target differs from exact sealed range identity")
    elif kind == "FULL_LARGEST_SHARD_VALIDATION":
        lane = trial.get("validation_lane")
        if lane not in {"acquisition", "steady"} or len(targets) != 1:
            raise Glm52Error("largest-shard validation projection is invalid")
        largest = plan.get("largest_shard_validation", {})
        target = targets[0]
        if target.get("path") != largest.get("path") \
                or target.get("xet_hash") != largest.get("xet_hash") \
                or target.get("lfs_sha256") != largest.get("lfs_sha256") \
                or target.get("start") != 0 or target.get("end") != largest.get("bytes"):
            raise Glm52Error("largest-shard validation target differs from sealed plan")
        selected = _matrix_trial(plan, str(trial.get("selected_profile_trial_id")))
        if trial.get("environment") != selected.get("environment") \
                or trial.get("selected_plan_trial_sha256") != _canonical_sha256(selected):
            raise Glm52Error("largest-shard validation profile projection is invalid")
    else:
        raise Glm52Error("unsupported live Xet trial kind")
    _validate_plan_environment(trial.get("environment"))
    budget = value.get("network_budget")
    if not isinstance(budget, Mapping) or set(budget) != {
        "campaign_hard_cap_bytes", "campaign_consumed_before_bytes",
        "campaign_remaining_before_bytes", "trial_network_cap_bytes",
        "planned_payload_bytes",
    }:
        raise Glm52Error("live Xet budget projection is invalid")
    expected_budget = _budget_block(
        plan,
        consumed_bytes=budget.get("campaign_consumed_before_bytes"),
        trial_cap_bytes=budget.get("trial_network_cap_bytes"),
        planned_payload_bytes=trial.get("planned_payload_bytes"),
    )
    if dict(budget) != expected_budget:
        raise Glm52Error("live Xet budget arithmetic changed")
    execution = value.get("execution")
    if not isinstance(execution, Mapping) \
            or execution.get("fresh_subprocess_required") is not True \
            or execution.get("body_file_writes_allowed") is not False \
            or execution.get("destructive_gc_allowed") is not False \
            or execution.get("child_protocol") != CHILD_PROTOCOL:
        raise Glm52Error("live Xet execution boundary is invalid")
    auth = value.get("public_hub_auth")
    if not isinstance(auth, Mapping) \
            or auth.get("hub_token_mode") != "EXPLICIT_TOKEN_FALSE" \
            or auth.get("authorization_header_serialized") is not False \
            or auth.get("refresh_headers_serialized") is not False:
        raise Glm52Error("live Xet public-Hub auth boundary is invalid")
    return value


def validate_live_capability(
    capability: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    spec: Mapping[str, Any],
    verifier: LiveCapabilityVerifier | None,
    now_unix_ns: int | None = None,
) -> dict[str, Any]:
    value = verify_sealed(dict(capability), label="live Xet capability")
    _exact_fields(value, CAPABILITY_FIELDS, "live Xet capability")
    if value.get("schema") != CAPABILITY_SCHEMA or value.get("status") != "AUTHORIZED":
        raise Glm52Error("live Xet capability schema/status mismatch")
    trial = spec["trial"]
    if value.get("repo") != REPO_ID or value.get("revision") != REVISION \
            or value.get("plan_seal_sha256") != plan.get("seal_sha256") \
            or value.get("trial_id") != trial.get("trial_id") \
            or value.get("allowed_kind") != trial.get("kind"):
        raise Glm52Error("live Xet capability identity/action mismatch")
    if value.get("seal_sha256") != spec.get("capability_seal_sha256"):
        raise Glm52Error("trial spec does not bind the supplied capability")
    if _require_int(value.get("max_network_bytes"), "capability.max_network_bytes", minimum=1) \
            < spec["network_budget"]["trial_network_cap_bytes"]:
        raise Glm52Error("capability network allowance is smaller than the trial cap")
    expires = _require_int(value.get("expires_unix_ns"), "capability.expires_unix_ns", minimum=1)
    now = time.time_ns() if now_unix_ns is None else _require_int(now_unix_ns, "now_unix_ns")
    if expires <= now:
        raise Glm52Error("live Xet capability is expired")
    controller = value.get("controller")
    if not isinstance(controller, Mapping) or set(controller) != {
        "controller_epoch", "checkpoint_seal_sha256", "lease_identity_sha256",
        "telegram_receipt_seal_sha256",
    } or not isinstance(controller.get("controller_epoch"), str) \
            or not controller.get("controller_epoch") \
            or any(not _is_sha256(controller.get(key)) for key in (
                "checkpoint_seal_sha256", "lease_identity_sha256",
                "telegram_receipt_seal_sha256",
            )):
        raise Glm52Error("live Xet capability controller evidence is incomplete")
    if value.get("credentials_serialized") is not False:
        raise Glm52Error("live Xet capability must not serialize credentials")
    if verifier is None:
        raise Glm52Error("independent live capability verifier is required")
    try:
        accepted = verifier.verify_live_capability(value, plan=plan, spec=spec)
    except Exception:
        raise Glm52Error("independent live capability verification failed") from None
    if accepted is not True:
        raise Glm52Error("independent live capability verification refused execution")
    return value


def child_environment(spec: Mapping[str, Any], base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Create a clean child environment; all Xet knobs come from the sealed row."""
    environment = {
        key: value for key, value in dict(os.environ if base is None else base).items()
        if not key.startswith("HF_XET_")
        and key not in {CHILD_ENV, SPEC_SEAL_ENV, EXECUTE_ENV,
                        "HF_HUB_DISABLE_IMPLICIT_TOKEN", "HF_HUB_DISABLE_TELEMETRY"}
    }
    environment.update(_validate_plan_environment(spec["trial"]["environment"]))
    environment.update(RESERVED_XET_ENV)
    environment.update(REQUIRED_CHILD_ENV)
    environment[CHILD_ENV] = "1"
    environment[SPEC_SEAL_ENV] = str(spec["seal_sha256"])
    return environment


def _validate_child_environment(spec: Mapping[str, Any]) -> None:
    if os.environ.get(CHILD_ENV) != "1" \
            or os.environ.get(SPEC_SEAL_ENV) != spec.get("seal_sha256"):
        raise Glm52Error("direct child execution is refused")
    expected = {
        **_validate_plan_environment(spec["trial"]["environment"]),
        **RESERVED_XET_ENV,
    }
    actual_xet = {key: value for key, value in os.environ.items() if key.startswith("HF_XET_")}
    if actual_xet != expected:
        raise Glm52Error("child HF_XET environment differs from the sealed trial")
    if any(os.environ.get(key) != value for key, value in REQUIRED_CHILD_ENV.items()):
        raise Glm52Error("child public-Hub isolation environment is incomplete")


def _runtime_config() -> dict[str, Any]:
    try:
        from hf_xet import XetConfig
    except ImportError as exc:
        raise Glm52Error("pinned hf_xet runtime is unavailable") from exc
    versions: dict[str, str] = {}
    for package, expected in PINNED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise Glm52Error(f"required package is unavailable: {package}") from exc
        if actual != expected:
            raise Glm52Error(f"{package} version drift: expected {expected}, got {actual}")
        versions[package] = actual
    config = dict(XetConfig().items())
    effective = {key: config.get(key) for key in RUNTIME_CONFIG_FIELDS}
    if set(effective) != set(RUNTIME_CONFIG_FIELDS) or any(
        effective[key] is None for key in RUNTIME_CONFIG_FIELDS
    ):
        raise Glm52Error("effective Xet config evidence is incomplete")
    if effective["log.dest"] != "stderr" or effective["log.format"] != "json":
        raise Glm52Error("Xet JSON stderr logging did not bind before import")
    return {"versions": versions, "effective_xet_config": effective}


def _public_hub_stream_group(session: Any) -> tuple[Any, dict[str, Any]]:
    """Create a public-repo stream group without exporting any header value."""
    try:
        from huggingface_hub.utils import build_hf_headers
        from huggingface_hub.utils._xet import (
            XetTokenType,
            xet_connection_info_refresh_url,
            xet_headers_without_auth,
        )
    except ImportError as exc:
        raise Glm52Error("pinned huggingface_hub Xet helpers are unavailable") from exc
    headers = build_hf_headers(
        token=False,
        library_name="hawking-glm52-xet-live",
        library_version="1",
    )
    if any(key.lower() == "authorization" for key in headers):
        raise Glm52Error("public Hub headers unexpectedly contain authorization")
    refresh_url = xet_connection_info_refresh_url(
        token_type=XetTokenType.READ,
        repo_id=REPO_ID,
        repo_type="model",
        revision=REVISION,
    )
    custom = xet_headers_without_auth(headers)
    group = session.new_download_stream_group(
        token_refresh_url=refresh_url,
        token_refresh_headers=headers,
        custom_headers=custom,
    )
    evidence = {
        "mode": "PUBLIC_HUB_READ_TOKEN_REFRESH",
        "refresh_url_sha256": hashlib.sha256(refresh_url.encode("utf-8")).hexdigest(),
        "refresh_header_names": sorted(key.lower() for key in headers),
        "custom_header_names": sorted(key.lower() for key in custom),
        "authorization_header_present": False,
        "header_values_serialized": False,
        "xet_access_token_serialized": False,
    }
    return group, evidence


def _stream_one(group: Any, target: Mapping[str, Any], clock_ns: Callable[[], int]) -> dict[str, Any]:
    from hf_xet import XetFileInfo

    started = clock_ns()
    digest = hashlib.sha256()
    count = 0
    stream = group.download_stream(
        XetFileInfo(target["xet_hash"], target["file_bytes"]),
        start=target["start"],
        end=target["end"],
    )
    try:
        for chunk in stream:
            if not isinstance(chunk, bytes) or not chunk:
                raise Glm52Error("hf_xet range stream yielded a nonempty-bytes violation")
            count += len(chunk)
            if count > target["length"]:
                raise Glm52Error("hf_xet range stream exceeded its sealed length")
            digest.update(chunk)
    except BaseException:
        try:
            stream.cancel()
        except Exception:
            pass
        raise
    finished = clock_ns()
    if count != target["length"]:
        raise Glm52Error(
            f"hf_xet range stream length mismatch for {target['range_id_sha256']}: "
            f"{count} != {target['length']}"
        )
    return {
        "range_id_sha256": target["range_id_sha256"],
        "path": target["path"],
        "start": target["start"],
        "end": target["end"],
        "bytes": count,
        "sha256": digest.hexdigest(),
        "elapsed_seconds": (finished - started) / 1_000_000_000,
    }


def stream_spec_in_memory(
    spec: Mapping[str, Any],
    *,
    session_factory: Callable[[], Any] | None = None,
    group_factory: Callable[[Any], tuple[Any, Mapping[str, Any]]] | None = None,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> dict[str, Any]:
    """Execute exact targets in memory; dependency hooks exist only for offline tests."""
    started = clock_ns()
    runtime = _runtime_config()
    if session_factory is None:
        from hf_xet import XetSession
        session_factory = XetSession
    session = session_factory()
    group, auth = (group_factory or _public_hub_stream_group)(session)
    workers = _require_int(
        spec["trial"].get("caller_concurrent_shard_streams"),
        "caller_concurrent_shard_streams",
        minimum=1,
    )
    if workers > len(spec["targets"]):
        workers = len(spec["targets"])
    by_id: dict[str, dict[str, Any]] = {}
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="glm52-xet") as pool:
            futures = {
                pool.submit(_stream_one, group, target, clock_ns): target["range_id_sha256"]
                for target in spec["targets"]
            }
            for future in as_completed(futures):
                result = future.result()
                by_id[result["range_id_sha256"]] = result
    except BaseException:
        try:
            session.sigint_abort()
        except Exception:
            pass
        raise
    ordered = [by_id[target["range_id_sha256"]] for target in spec["targets"]]
    finished = clock_ns()
    payload = sum(item["bytes"] for item in ordered)
    if payload != spec["trial"]["planned_payload_bytes"]:
        raise Glm52Error("streamed payload differs from sealed planned payload")
    return seal({
        "schema": CHILD_RESULT_SCHEMA,
        "status": "PASS_STREAMED_IN_MEMORY",
        "spec_seal_sha256": spec["seal_sha256"],
        "plan_seal_sha256": spec["plan_seal_sha256"],
        "trial_id": spec["trial"]["trial_id"],
        "kind": spec["trial"]["kind"],
        "pid": os.getpid(),
        "runtime": runtime,
        "public_hub_auth": dict(auth),
        "range_results": ordered,
        "payload_bytes": payload,
        "started_monotonic_ns": started,
        "finished_monotonic_ns": finished,
        "elapsed_seconds": (finished - started) / 1_000_000_000,
        "python_body_file_writes": 0,
        "error": None,
    })


def _flatten_numbers(value: Any, prefix: str = "") -> Iterable[tuple[str, float]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{str(key).lower()}" if prefix else str(key).lower()
            yield from _flatten_numbers(item, name)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _flatten_numbers(item, f"{prefix}[{index}]")
    elif not isinstance(value, bool) and isinstance(value, (int, float)) \
            and math.isfinite(float(value)):
        yield prefix, float(value)


def parse_xet_json_logs(lines: Iterable[str]) -> dict[str, Any]:
    """Summarize explicit JSON metrics without retaining raw potentially sensitive logs."""
    digest = hashlib.sha256()
    total = 0
    parsed = 0
    invalid = 0
    retry_related = 0
    configuration_related = 0
    retry_rates: list[float] = []
    retry_counts: list[int] = []
    request_counts: list[int] = []
    latencies: list[float] = []
    amplifications: list[float] = []
    for raw in lines:
        if not isinstance(raw, str):
            raise Glm52Error("Xet log line must be text")
        total += 1
        encoded = raw.encode("utf-8", errors="replace")
        digest.update(encoded)
        if not raw.endswith("\n"):
            digest.update(b"\n")
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if not isinstance(record, Mapping):
            invalid += 1
            continue
        parsed += 1
        lowered = json.dumps(record, sort_keys=True, ensure_ascii=False).lower()
        retry_related += int("retry" in lowered)
        configuration_related += int("config" in lowered)
        for key, number in _flatten_numbers(record):
            leaf = re.sub(r".*[.\]]", "", key)
            if leaf in {"retry_rate", "retry_ratio"} and 0.0 <= number <= 1.0:
                retry_rates.append(number)
            elif leaf in {"retry_count", "retries", "retry_attempts"} \
                    and number >= 0 and number.is_integer():
                retry_counts.append(int(number))
            elif leaf in {"request_count", "requests", "requests_total"} \
                    and number > 0 and number.is_integer():
                request_counts.append(int(number))
            elif leaf in {"reconstruction_latency_seconds", "reconstruction_seconds"} \
                    and number >= 0:
                latencies.append(number)
            elif leaf in {"reconstruction_latency_ms", "reconstruction_milliseconds"} \
                    and number >= 0:
                latencies.append(number / 1000.0)
            elif leaf in {"temporary_amplification_ratio", "temp_amplification_ratio"} \
                    and number >= 0:
                amplifications.append(number)
    retry_rate: float | None = max(retry_rates) if retry_rates else None
    retry_method: str | None = "EXPLICIT_JSON_RATE" if retry_rates else None
    if retry_rate is None and retry_counts and request_counts:
        retry_rate = max(retry_counts) / max(request_counts)
        if retry_rate <= 1.0:
            retry_method = "EXPLICIT_JSON_COUNTER_RATIO"
        else:
            retry_rate = None
    return {
        "log_stream_sha256": digest.hexdigest(),
        "line_count": total,
        "parsed_json_records": parsed,
        "unparseable_lines": invalid,
        "retry_related_json_events": retry_related,
        "configuration_related_json_events": configuration_related,
        "retry_rate_available": retry_rate is not None,
        "retry_rate": retry_rate,
        "retry_rate_method": retry_method,
        "explicit_retry_count": max(retry_counts) if retry_counts else None,
        "explicit_request_count": max(request_counts) if request_counts else None,
        "reconstruction_latency_available": bool(latencies),
        "maximum_reconstruction_latency_seconds": max(latencies) if latencies else None,
        "temporary_amplification_available": bool(amplifications),
        "maximum_temporary_amplification_ratio": max(amplifications) if amplifications else None,
        "missing_metrics_are_zero": False,
        "raw_logs_serialized": False,
    }


def _run_read_only(argv: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(argv), capture_output=True, text=True, check=False, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Glm52Error(f"resource command failed to start: {argv[0]}: {exc}") from exc
    if completed.returncode != 0:
        raise Glm52Error(
            f"resource command failed ({completed.returncode}): {argv[0]}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


_BYTE_UNITS = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_swapusage(text: str) -> int:
    match = re.search(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMGT])\b", text)
    if match is None:
        raise Glm52Error("cannot parse Darwin vm.swapusage")
    return int(float(match.group(1)) * _BYTE_UNITS[match.group(2)])


def parse_vm_stat(text: str) -> tuple[int, int]:
    page = re.search(r"page size of\s+([0-9]+) bytes", text)
    if page is None:
        raise Glm52Error("cannot parse Darwin vm_stat page size")
    values: dict[str, int] = {}
    for name, raw in re.findall(r"^([^:\n]+):\s+([0-9]+)\.\s*$", text, re.MULTILINE):
        values[name.strip()] = int(raw)
    required = {"Pages free", "Pages inactive", "Pages speculative", "Swapouts"}
    if not required <= set(values):
        raise Glm52Error("Darwin vm_stat lacks required counters")
    available_pages = sum(values[key] for key in (
        "Pages free", "Pages inactive", "Pages speculative",
    ))
    return available_pages * int(page.group(1)), values["Swapouts"]


def parse_thermal_warning(text: str) -> bool:
    if "No thermal warning level has been recorded" in text:
        return False
    lowered = text.lower()
    if "thermal warning" in lowered or "thermal level" in lowered:
        return True
    raise Glm52Error("cannot classify Darwin thermal state")


def parse_ps_cpu_rss(text: str) -> tuple[float, int]:
    fields = text.split()
    if len(fields) != 2:
        raise Glm52Error("cannot parse Darwin ps CPU/RSS sample")
    try:
        cpu = float(fields[0])
        rss = int(fields[1]) * 1024
    except ValueError as exc:
        raise Glm52Error("Darwin ps CPU/RSS sample is nonnumeric") from exc
    _require_number(cpu, "process CPU percent")
    _require_int(rss, "process RSS bytes")
    return cpu, rss


def parse_iostat_total_bytes(text: str) -> int:
    candidates: list[list[str]] = []
    for line in text.splitlines():
        fields = line.split()
        if fields and len(fields) % 3 == 0:
            try:
                [float(item) for item in fields]
            except ValueError:
                continue
            candidates.append(fields)
    if not candidates:
        raise Glm52Error("cannot parse Darwin iostat totals")
    fields = candidates[-1]
    megabytes = sum(float(fields[index]) for index in range(2, len(fields), 3))
    if not math.isfinite(megabytes) or megabytes < 0:
        raise Glm52Error("Darwin iostat total is invalid")
    return int(megabytes * 1024**2)


def parse_netstat_link_bytes(text: str) -> int:
    total = 0
    rows = 0
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 10 or not fields[2].startswith("<Link#"):
            continue
        name = fields[0]
        if name == "lo0" or name.endswith("*"):
            continue
        # Darwin may omit the link-layer address (notably for ``lo0``), so
        # parse the stable seven trailing counters rather than fixed columns:
        # Ipkts Ierrs Ibytes Opkts Oerrs Obytes Coll.
        try:
            incoming = int(fields[-5])
            outgoing = int(fields[-2])
        except (ValueError, IndexError) as exc:
            raise Glm52Error("cannot parse Darwin netstat link counters") from exc
        if incoming < 0 or outgoing < 0:
            raise Glm52Error("Darwin netstat link counter is negative")
        rows += 1
        total += incoming + outgoing
    if rows == 0:
        raise Glm52Error("Darwin netstat has no eligible non-loopback link row")
    return total


class DarwinResourceSampler:
    """Strict read-only macOS sampler; any unavailable field aborts the trial."""

    def __init__(
        self,
        scratch_root: Path,
        *,
        command_runner: CommandRunner = _run_read_only,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleeper: Callable[[float], None] = time.sleep,
        disk_warmup_seconds: float = 0.2,
    ) -> None:
        if sys.platform != "darwin":
            raise Glm52Error("DarwinResourceSampler is available only on macOS")
        self.scratch_root = scratch_root
        self.command_runner = command_runner
        self.clock_ns = clock_ns
        self.sleeper = sleeper
        self.disk_warmup_seconds = _require_number(
            disk_warmup_seconds, "disk_warmup_seconds", minimum=0.01, maximum=5.0
        )
        self._disk_total: int | None = None
        self._disk_ns: int | None = None

    def _disk_rate(self) -> float:
        current = parse_iostat_total_bytes(
            self.command_runner(("/usr/sbin/iostat", "-Id", "-c", "1"))
        )
        now = self.clock_ns()
        if self._disk_total is None:
            self._disk_total, self._disk_ns = current, now
            self.sleeper(self.disk_warmup_seconds)
            current = parse_iostat_total_bytes(
                self.command_runner(("/usr/sbin/iostat", "-Id", "-c", "1"))
            )
            now = self.clock_ns()
        assert self._disk_total is not None and self._disk_ns is not None
        if current < self._disk_total or now <= self._disk_ns:
            raise Glm52Error("Darwin disk counter regressed or clock did not advance")
        rate = (current - self._disk_total) / ((now - self._disk_ns) / 1_000_000_000)
        self._disk_total, self._disk_ns = current, now
        return rate

    def sample(self, pid: int) -> Mapping[str, Any]:
        process_id = _require_int(pid, "sample pid", minimum=1)
        vm = self.command_runner(("/usr/bin/vm_stat",))
        available, swapouts = parse_vm_stat(vm)
        swap_used = parse_swapusage(
            self.command_runner(("/usr/sbin/sysctl", "-n", "vm.swapusage"))
        )
        thermal = parse_thermal_warning(
            self.command_runner(("/usr/bin/pmset", "-g", "therm"))
        )
        cpu, rss = parse_ps_cpu_rss(
            self.command_runner(("/bin/ps", "-o", "%cpu=,rss=", "-p", str(process_id)))
        )
        try:
            free_disk = shutil.disk_usage(self.scratch_root).free
        except OSError as exc:
            raise Glm52Error(f"cannot sample free disk for {self.scratch_root}: {exc}") from exc
        return {
            "schema": RESOURCE_SAMPLE_SCHEMA,
            "sampled_monotonic_ns": self.clock_ns(),
            "pid": process_id,
            "swap_used_bytes": swap_used,
            "swapouts": swapouts,
            "thermal_warning": thermal,
            "free_disk_bytes": free_disk,
            "available_ram_bytes": available,
            "cpu_percent": cpu,
            "process_rss_bytes": rss,
            "disk_write_bytes_per_second": self._disk_rate(),
        }


class DarwinHostNetworkCounter:
    """Conservative host-wide non-loopback counter used as an upper bound."""

    def __init__(self, *, command_runner: CommandRunner = _run_read_only) -> None:
        if sys.platform != "darwin":
            raise Glm52Error("DarwinHostNetworkCounter is available only on macOS")
        self.command_runner = command_runner

    def snapshot(self) -> int:
        return parse_netstat_link_bytes(
            self.command_runner(("/usr/sbin/netstat", "-ib"))
        )

    def evidence(self) -> Mapping[str, Any]:
        return {
            "schema": NETWORK_COUNTER_SCHEMA,
            "method": "DARWIN_NETSTAT_LINK_BYTES_CONSERVATIVE",
            "scope": "HOST_ALL_NON_LOOPBACK_LINK_INTERFACES",
            "counts_unrelated_host_traffic": True,
            "monotonicity_required": True,
            "credentials_serialized": False,
        }


def _validate_resource_sample(value: Mapping[str, Any], *, pid: int) -> dict[str, Any]:
    allowed = BASE_RESOURCE_FIELDS | OPTIONAL_XET_SAMPLE_FIELDS
    if not BASE_RESOURCE_FIELDS <= set(value) or not set(value) <= allowed:
        raise Glm52Error("resource sampler returned incomplete or unknown fields")
    result = dict(value)
    if result.get("schema") != RESOURCE_SAMPLE_SCHEMA or result.get("pid") != pid:
        raise Glm52Error("resource sample schema/PID mismatch")
    for key in (
        "sampled_monotonic_ns", "swap_used_bytes", "swapouts", "free_disk_bytes",
        "available_ram_bytes", "process_rss_bytes",
    ):
        _require_int(result.get(key), f"resource sample {key}")
    if not isinstance(result.get("thermal_warning"), bool):
        raise Glm52Error("resource sample thermal_warning must be boolean")
    _require_number(result.get("cpu_percent"), "resource sample cpu_percent")
    _require_number(
        result.get("disk_write_bytes_per_second"),
        "resource sample disk_write_bytes_per_second",
    )
    for key in OPTIONAL_XET_SAMPLE_FIELDS & set(result):
        _require_number(
            result[key], key, maximum=1.0 if key == "retry_rate" else None
        )
    return result


class _LogAccumulator:
    """Bounded in-memory log capture; overflow fails the evidence gate."""

    def __init__(self, *, maximum_bytes: int = 16 * 1024**2) -> None:
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._maximum_bytes = _require_int(maximum_bytes, "maximum log bytes", minimum=1)
        self._bytes = 0
        self._overflow = False

    def add(self, line: str) -> None:
        with self._lock:
            self._bytes += len(line.encode("utf-8", errors="replace"))
            if self._bytes > self._maximum_bytes:
                self._overflow = True
                return
            self._lines.append(line)

    def summarize(self) -> dict[str, Any]:
        with self._lock:
            if self._overflow:
                raise Glm52Error("Xet JSON log capture exceeded its bounded evidence budget")
            return parse_xet_json_logs(tuple(self._lines))


def _line_pump(handle: Any, output: queue.Queue[str] | None, logs: _LogAccumulator | None) -> None:
    try:
        for line in handle:
            if output is not None:
                output.put(line)
            if logs is not None:
                logs.add(line)
    finally:
        if output is not None:
            output.put("")


def _queue_line(lines: queue.Queue[str], deadline: float) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise Glm52Error("live Xet child protocol timed out")
    try:
        line = lines.get(timeout=remaining)
    except queue.Empty as exc:
        raise Glm52Error("live Xet child protocol timed out") from exc
    if line == "":
        raise Glm52Error("live Xet child closed stdout before protocol completion")
    return line


def _cancel_child(process: subprocess.Popen[str], *, grace_seconds: float = 5.0) -> None:
    """Cancel one isolated child process group; never touch files or caches."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired as exc:
        raise Glm52Error("live Xet child resisted SIGINT and SIGKILL") from exc


def _metric(
    sample: Mapping[str, Any],
    logs: Mapping[str, Any],
    child: Mapping[str, Any],
    key: str,
) -> float | None:
    if key in sample:
        return float(sample[key])
    if key == "retry_rate":
        value = logs.get("retry_rate")
    elif key == "temporary_amplification_ratio":
        value = logs.get("maximum_temporary_amplification_ratio")
    elif key == "reconstruction_latency_seconds":
        explicit = logs.get("maximum_reconstruction_latency_seconds")
        if explicit is not None:
            value = explicit
        else:
            ranges = child.get("range_results", [])
            value = max(
                (item.get("elapsed_seconds") for item in ranges if isinstance(item, Mapping)),
                default=None,
            )
    else:
        raise AssertionError(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _planner_observation(
    sample: Mapping[str, Any],
    *,
    network_counter: int,
    logs: Mapping[str, Any],
    child: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "swap_used_bytes": sample["swap_used_bytes"],
        "swapouts": sample["swapouts"],
        "thermal_warning": sample["thermal_warning"],
        "free_disk_bytes": sample["free_disk_bytes"],
        "available_ram_bytes": sample["available_ram_bytes"],
        "cpu_percent": sample["cpu_percent"],
        "disk_write_bytes_per_second": sample["disk_write_bytes_per_second"],
        "reconstruction_latency_seconds": _metric(
            sample, logs, child, "reconstruction_latency_seconds"
        ),
        "retry_rate": _metric(sample, logs, child, "retry_rate"),
        "temporary_amplification_ratio": _metric(
            sample, logs, child, "temporary_amplification_ratio"
        ),
        "actual_network_bytes": network_counter,
    }


def _validate_child_result(value: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    child = verify_sealed(dict(value), label="live Xet child result")
    expected = {
        "schema", "status", "spec_seal_sha256", "plan_seal_sha256", "trial_id",
        "kind", "pid", "runtime", "public_hub_auth", "range_results",
        "payload_bytes", "started_monotonic_ns", "finished_monotonic_ns",
        "elapsed_seconds", "python_body_file_writes", "error", "seal_sha256",
    }
    if set(child) != expected or child.get("schema") != CHILD_RESULT_SCHEMA \
            or child.get("status") != "PASS_STREAMED_IN_MEMORY" \
            or child.get("spec_seal_sha256") != spec.get("seal_sha256") \
            or child.get("plan_seal_sha256") != spec.get("plan_seal_sha256") \
            or child.get("trial_id") != spec["trial"]["trial_id"] \
            or child.get("kind") != spec["trial"]["kind"] \
            or child.get("python_body_file_writes") != 0 \
            or child.get("error") is not None:
        raise Glm52Error("live Xet child result identity/boundary mismatch")
    rows = child.get("range_results")
    if not isinstance(rows, list) or len(rows) != len(spec["targets"]):
        raise Glm52Error("live Xet child range coverage is incomplete")
    for target, row in zip(spec["targets"], rows):
        if not isinstance(row, Mapping) or set(row) != {
            "range_id_sha256", "path", "start", "end", "bytes", "sha256",
            "elapsed_seconds",
        } or row.get("range_id_sha256") != target["range_id_sha256"] \
                or row.get("path") != target["path"] \
                or row.get("start") != target["start"] \
                or row.get("end") != target["end"] \
                or row.get("bytes") != target["length"] \
                or not _is_sha256(row.get("sha256")):
            raise Glm52Error("live Xet child range result differs from sealed target")
        _require_number(row.get("elapsed_seconds"), "range elapsed_seconds")
    if child.get("payload_bytes") != sum(item["length"] for item in spec["targets"]):
        raise Glm52Error("live Xet child payload count mismatch")
    _require_number(child.get("elapsed_seconds"), "child elapsed_seconds", minimum=1e-12)
    return child


def _build_trial_result(
    spec: Mapping[str, Any],
    child: Mapping[str, Any],
    *,
    raw_samples: Sequence[Mapping[str, Any]],
    network_samples: Sequence[int],
    network_evidence: Mapping[str, Any],
    log_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if len(raw_samples) < 3 or len(raw_samples) != len(network_samples):
        raise Glm52Error("live trial requires before, runtime, and after samples")
    if any(right < left for left, right in zip(network_samples, network_samples[1:])):
        raise Glm52Error("network counter regressed during live Xet trial")
    actual = network_samples[-1] - network_samples[0]
    cap = spec["network_budget"]["trial_network_cap_bytes"]
    if actual <= 0 or actual > cap:
        raise Glm52Error("live Xet trial network delta is zero or above its cap")
    observations = [
        _planner_observation(
            sample, network_counter=counter, logs=log_evidence, child=child
        )
        for sample, counter in zip(raw_samples, network_samples)
    ]
    metrics_complete = all(
        observation[key] is not None
        for observation in observations
        for key in (
            "reconstruction_latency_seconds", "retry_rate",
            "temporary_amplification_ratio",
        )
    )
    elapsed = _require_number(child["elapsed_seconds"], "child elapsed_seconds", minimum=1e-12)
    peak_rss = max(int(sample["process_rss_bytes"]) for sample in raw_samples)
    effective = child["runtime"]["effective_xet_config"]
    transfer = effective.get("client.ac_initial_download_concurrency")
    _require_int(transfer, "effective initial download concurrency", minimum=1)
    result = {
        "schema": TRIAL_RESULT_SCHEMA,
        "status": "PASS_COMPLETE_MEASURED" if metrics_complete else "INCOMPLETE_REQUIRED_XET_METRICS",
        "repo": REPO_ID,
        "revision": REVISION,
        "plan_seal_sha256": spec["plan_seal_sha256"],
        "spec_seal_sha256": spec["seal_sha256"],
        "capability_seal_sha256": spec["capability_seal_sha256"],
        "trial_binding": {
            "ordinal": spec["trial"]["ordinal"],
            "trial_id": spec["trial"]["trial_id"],
            "kind": spec["trial"]["kind"],
            "caller_concurrent_shard_streams": spec["trial"][
                "caller_concurrent_shard_streams"
            ],
            "ordered_range_ids_sha256": spec["trial"]["ordered_range_ids_sha256"],
            "planned_payload_bytes": spec["trial"]["planned_payload_bytes"],
            "plan_trial_sha256": _canonical_sha256(spec["trial"]),
            "validation_context": (
                {
                    "lane": spec["trial"]["validation_lane"],
                    "selected_trial_id": spec["trial"]["selected_profile_trial_id"],
                    "validation_pass": spec["trial"]["validation_pass"],
                    "selected_plan_trial_sha256": spec["trial"][
                        "selected_plan_trial_sha256"
                    ],
                }
                if spec["trial"]["kind"] == "FULL_LARGEST_SHARD_VALIDATION"
                else None
            ),
        },
        "runtime": child["runtime"],
        "public_hub_auth": child["public_hub_auth"],
        "range_results": child["range_results"],
        "stream_measurement": {
            "payload_bytes": child["payload_bytes"],
            "elapsed_seconds": elapsed,
            "throughput_bytes_per_second": child["payload_bytes"] / elapsed,
            "throughput_interval_method": "SINGLE_TRIAL_POINT_MEASUREMENT",
            "peak_process_rss_bytes": peak_rss,
            "effective_transfer_concurrency": transfer,
        },
        "network_accounting": {
            "counter_evidence": dict(network_evidence),
            "baseline_bytes": network_samples[0],
            "final_bytes": network_samples[-1],
            "actual_network_bytes": actual,
            "trial_network_cap_bytes": cap,
            "counter_monotonic": True,
            "cap_enforced_during_execution": True,
        },
        "xet_log_evidence": dict(log_evidence),
        "resource_observations": {
            "before": observations[0],
            "samples": observations[1:-1],
            "after": observations[-1],
            "heavy_lane_regressions": [],
            "complete_source_views": 0,
            "required_metrics_complete": metrics_complete,
            "missing_metrics": sorted({
                key
                for observation in observations
                for key in (
                    "reconstruction_latency_seconds", "retry_rate",
                    "temporary_amplification_ratio",
                )
                if observation[key] is None
            }),
        },
        "body_persistence": {
            "python_body_file_writes": 0,
            "python_body_file_paths": [],
            "streamed_bytes_retained_after_hash": False,
            "destination_file_api_used": False,
            "xet_configured_chunk_cache_policy": spec["trial"].get(
                "chunk_cache_policy", "NOT_APPLICABLE_FULL_VALIDATION"
            ),
            "destructive_gc_performed": False,
        },
        "process": {
            "pid": child["pid"],
            "fresh_subprocess": True,
            "child_protocol": CHILD_PROTOCOL,
            "child_result_seal_sha256": child["seal_sha256"],
        },
        "failure": None,
    }
    return seal(result)


def execute_trial(
    plan: Mapping[str, Any],
    spec: Mapping[str, Any],
    capability: Mapping[str, Any],
    *,
    capability_verifier: LiveCapabilityVerifier | None,
    resource_sampler: ResourceSampler | None,
    network_counter: NetworkBudgetCounter | None,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run one fresh, supervised child process after every fail-closed gate."""
    if os.environ.get(EXECUTE_ENV) != "1":
        raise Glm52Error(f"live Xet execution requires {EXECUTE_ENV}=1")
    value = validate_live_plan(plan, root=root, rebuild=True)
    trial_spec = validate_trial_spec(spec, value)
    validate_live_capability(
        capability,
        plan=value,
        spec=trial_spec,
        verifier=capability_verifier,
    )
    if resource_sampler is None or network_counter is None:
        raise Glm52Error("live Xet execution requires strict resource and network instruments")

    timeout = float(trial_spec["execution"]["timeout_seconds"])
    interval = float(trial_spec["execution"]["sample_interval_seconds"])
    deadline = time.monotonic() + timeout
    argv = [sys.executable, str(Path(__file__).resolve()), "_child"]
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=child_environment(trial_spec),
            start_new_session=True,
        )
    except OSError as exc:
        raise Glm52Error(f"cannot start isolated live Xet child: {exc}") from exc
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    stdout_lines: queue.Queue[str] = queue.Queue()
    logs = _LogAccumulator()
    stdout_thread = threading.Thread(
        target=_line_pump, args=(process.stdout, stdout_lines, None), daemon=True
    )
    stderr_thread = threading.Thread(
        target=_line_pump, args=(process.stderr, None, logs), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    raw_samples: list[dict[str, Any]] = []
    network_samples: list[int] = []
    try:
        process.stdin.write(json.dumps(trial_spec, sort_keys=True, separators=(",", ":")) + "\n")
        process.stdin.flush()
        ready = json.loads(_queue_line(stdout_lines, deadline))
        if ready != {
            "protocol": CHILD_PROTOCOL,
            "status": "READY",
            "spec_seal_sha256": trial_spec["seal_sha256"],
        }:
            raise Glm52Error("live Xet child readiness handshake is invalid")
        raw_samples.append(_validate_resource_sample(resource_sampler.sample(process.pid), pid=process.pid))
        network_samples.append(_require_int(network_counter.snapshot(), "network baseline"))
        process.stdin.write(json.dumps({
            "command": "GO", "spec_seal_sha256": trial_spec["seal_sha256"],
        }, sort_keys=True, separators=(",", ":")) + "\n")
        process.stdin.flush()

        result_message: dict[str, Any] | None = None
        while result_message is None:
            if time.monotonic() >= deadline:
                raise Glm52Error("live Xet child exceeded its sealed timeout")
            try:
                line = stdout_lines.get(timeout=min(interval, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                line = None
            if line == "":
                raise Glm52Error("live Xet child exited before returning a result")
            if line is not None:
                message = json.loads(line)
                if not isinstance(message, dict) or message.get("protocol") != CHILD_PROTOCOL \
                        or message.get("status") != "RESULT" \
                        or set(message) != {"protocol", "status", "result"}:
                    raise Glm52Error("live Xet child returned an invalid protocol message")
                result_message = message
            raw_samples.append(
                _validate_resource_sample(resource_sampler.sample(process.pid), pid=process.pid)
            )
            network_samples.append(_require_int(network_counter.snapshot(), "network counter"))
            if network_samples[-1] < network_samples[0]:
                raise Glm52Error("network counter regressed during live Xet trial")
            if network_samples[-1] - network_samples[0] \
                    > trial_spec["network_budget"]["trial_network_cap_bytes"]:
                raise Glm52Error("live Xet trial crossed its network cap")
        child = _validate_child_result(result_message["result"], trial_spec)
        # The child remains alive until ACK, making the final process sample strict.
        if len(raw_samples) < 2:
            raw_samples.append(
                _validate_resource_sample(resource_sampler.sample(process.pid), pid=process.pid)
            )
            network_samples.append(_require_int(network_counter.snapshot(), "network final"))
        process.stdin.write(json.dumps({
            "command": "ACK", "spec_seal_sha256": trial_spec["seal_sha256"],
        }, sort_keys=True, separators=(",", ":")) + "\n")
        process.stdin.flush()
        try:
            returncode = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise Glm52Error("live Xet child did not exit after ACK") from exc
        if returncode != 0:
            raise Glm52Error(f"live Xet child exited with status {returncode}")
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        if len(raw_samples) < 3:
            # Duplicate neither values nor claims: obtain an additional real sample while
            # the protocol is live in normal operation.  A too-fast fake child is refused.
            raise Glm52Error("live Xet trial produced no runtime resource sample")
        log_evidence = logs.summarize()
        result = _build_trial_result(
            trial_spec,
            child,
            raw_samples=raw_samples,
            network_samples=network_samples,
            network_evidence=network_counter.evidence(),
            log_evidence=log_evidence,
        )
        return validate_trial_result(result, plan=value)
    except BaseException:
        _cancel_child(process)
        raise
    finally:
        try:
            process.stdin.close()
        except Exception:
            pass


def validate_trial_result(result: Mapping[str, Any], *, plan: Mapping[str, Any]) -> dict[str, Any]:
    value = verify_sealed(dict(result), label="live Xet trial result")
    _exact_fields(value, RESULT_FIELDS, "live Xet trial result")
    if value.get("schema") != TRIAL_RESULT_SCHEMA \
            or value.get("status") not in {
                "PASS_COMPLETE_MEASURED", "INCOMPLETE_REQUIRED_XET_METRICS",
            }:
        raise Glm52Error("live Xet trial result schema/status mismatch")
    if value.get("repo") != REPO_ID or value.get("revision") != REVISION \
            or value.get("plan_seal_sha256") != plan.get("seal_sha256") \
            or not _is_sha256(value.get("spec_seal_sha256")) \
            or not _is_sha256(value.get("capability_seal_sha256")):
        raise Glm52Error("live Xet trial result immutable binding is invalid")
    binding = value.get("trial_binding")
    if not isinstance(binding, Mapping) or set(binding) != {
        "ordinal", "trial_id", "kind", "caller_concurrent_shard_streams",
        "ordered_range_ids_sha256", "planned_payload_bytes", "plan_trial_sha256",
        "validation_context",
    }:
        raise Glm52Error("live Xet result trial binding shape is invalid")
    kind = binding.get("kind")
    if kind == "BOUNDED_XET_BODY_RANGE":
        expected = _matrix_trial(plan, str(binding.get("trial_id")))
        if binding.get("ordinal") != expected.get("ordinal") \
                or binding.get("caller_concurrent_shard_streams") != expected.get(
                    "caller_concurrent_shard_streams"
                ) or binding.get("ordered_range_ids_sha256") != expected.get(
                    "ordered_range_ids_sha256"
                ) or binding.get("planned_payload_bytes") != expected.get("planned_payload_bytes") \
                or binding.get("plan_trial_sha256") != _canonical_sha256(expected) \
                or binding.get("validation_context") is not None:
            raise Glm52Error("live Xet result differs from sealed matrix trial")
        expected_ids = expected["ordered_range_ids"]
    elif kind == "FULL_LARGEST_SHARD_VALIDATION":
        context = binding.get("validation_context")
        if not isinstance(context, Mapping) or set(context) != {
            "lane", "selected_trial_id", "validation_pass", "selected_plan_trial_sha256",
        } or context.get("lane") not in {"acquisition", "steady"}:
            raise Glm52Error("largest-shard result validation context is incomplete")
        selected = _matrix_trial(plan, str(context.get("selected_trial_id")))
        pass_index = 0 if context["lane"] == "acquisition" else 1
        if context.get("selected_plan_trial_sha256") != _canonical_sha256(selected) \
                or context.get("validation_pass") != plan["largest_shard_validation"][
                    "passes"
                ][pass_index]:
            raise Glm52Error("largest-shard result selected-profile binding is invalid")
        expected_ids = [item.get("range_id_sha256") for item in value.get("range_results", [])]
    else:
        raise Glm52Error("live Xet result kind is unsupported")
    rows = value.get("range_results")
    if not isinstance(rows, list) or [row.get("range_id_sha256") for row in rows] != expected_ids:
        raise Glm52Error("live Xet result exact range coverage changed")
    if any(
        not isinstance(row, Mapping) or not _is_sha256(row.get("sha256"))
        or row.get("bytes") != row.get("end") - row.get("start")
        for row in rows
    ):
        raise Glm52Error("live Xet result range hash/byte evidence is invalid")
    stream = value.get("stream_measurement")
    if not isinstance(stream, Mapping) or set(stream) != {
        "payload_bytes", "elapsed_seconds", "throughput_bytes_per_second",
        "throughput_interval_method", "peak_process_rss_bytes",
        "effective_transfer_concurrency",
    } or stream.get("payload_bytes") != binding.get("planned_payload_bytes"):
        raise Glm52Error("live Xet stream measurement shape/bytes are invalid")
    for key in ("elapsed_seconds", "throughput_bytes_per_second"):
        _require_number(stream.get(key), f"stream.{key}", minimum=1e-12)
    _require_int(stream.get("peak_process_rss_bytes"), "stream.peak_process_rss_bytes")
    _require_int(
        stream.get("effective_transfer_concurrency"),
        "stream.effective_transfer_concurrency",
        minimum=1,
    )
    network = value.get("network_accounting")
    if not isinstance(network, Mapping) or set(network) != {
        "counter_evidence", "baseline_bytes", "final_bytes", "actual_network_bytes",
        "trial_network_cap_bytes", "counter_monotonic", "cap_enforced_during_execution",
    }:
        raise Glm52Error("live Xet network accounting shape is invalid")
    baseline = _require_int(network.get("baseline_bytes"), "network.baseline_bytes")
    final = _require_int(network.get("final_bytes"), "network.final_bytes")
    actual = _require_int(network.get("actual_network_bytes"), "network.actual_network_bytes", minimum=1)
    cap = _require_int(network.get("trial_network_cap_bytes"), "network.trial_network_cap_bytes", minimum=1)
    if final - baseline != actual or actual > cap \
            or network.get("counter_monotonic") is not True \
            or network.get("cap_enforced_during_execution") is not True:
        raise Glm52Error("live Xet network accounting arithmetic/gates failed")
    resource = value.get("resource_observations")
    if not isinstance(resource, Mapping) or set(resource) != {
        "before", "samples", "after", "heavy_lane_regressions",
        "complete_source_views", "required_metrics_complete", "missing_metrics",
    } or not isinstance(resource.get("samples"), list):
        raise Glm52Error("live Xet resource observations shape is invalid")
    missing = resource.get("missing_metrics")
    if not isinstance(missing, list) or len(missing) != len(set(missing)):
        raise Glm52Error("live Xet missing-metric evidence is invalid")
    complete = not missing
    if resource.get("required_metrics_complete") is not complete \
            or (value["status"] == "PASS_COMPLETE_MEASURED") is not complete:
        raise Glm52Error("live Xet status does not match required metric availability")
    body = value.get("body_persistence")
    if not isinstance(body, Mapping) or body.get("python_body_file_writes") != 0 \
            or body.get("python_body_file_paths") != [] \
            or body.get("streamed_bytes_retained_after_hash") is not False \
            or body.get("destination_file_api_used") is not False \
            or body.get("destructive_gc_performed") is not False:
        raise Glm52Error("live Xet result crossed the body persistence boundary")
    if value.get("failure") is not None:
        raise Glm52Error("successful/incomplete live Xet result contains a failure")
    return value


def build_largest_validation_evidence(
    plan: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    lane: str,
    selected_trial_id: str,
) -> dict[str, Any]:
    value = validate_trial_result(result, plan=plan)
    if lane not in {"acquisition", "steady"} \
            or value["status"] != "PASS_COMPLETE_MEASURED" \
            or value["trial_binding"]["kind"] != "FULL_LARGEST_SHARD_VALIDATION":
        raise Glm52Error("largest validation result/lane mismatch")
    context = value["trial_binding"]["validation_context"]
    if context.get("lane") != lane or context.get("selected_trial_id") != selected_trial_id:
        raise Glm52Error("largest validation evidence selection differs from executed profile")
    largest = plan["largest_shard_validation"]
    rows = value["range_results"]
    if len(rows) != 1 or rows[0]["path"] != largest["path"] \
            or rows[0]["bytes"] != largest["bytes"] \
            or rows[0]["sha256"] != largest["lfs_sha256"]:
        raise Glm52Error("full largest-shard validation hash/bytes mismatch")
    pass_index = 0 if lane == "acquisition" else 1
    return seal({
        "schema": LARGEST_EVIDENCE_SCHEMA,
        "status": "PASS_FULL_HASH_IN_MEMORY",
        "repo": REPO_ID,
        "revision": REVISION,
        "plan_seal_sha256": plan["seal_sha256"],
        "lane": lane,
        "selected_trial_id": selected_trial_id,
        "validation_pass": largest["passes"][pass_index],
        "identity": {
            "path": largest["path"],
            "bytes": largest["bytes"],
            "xet_hash": largest["xet_hash"],
            "expected_lfs_sha256": largest["lfs_sha256"],
        },
        "measurement": {
            "streamed_bytes": rows[0]["bytes"],
            "observed_sha256": rows[0]["sha256"],
            "actual_network_bytes": value["network_accounting"]["actual_network_bytes"],
            "trial_result_seal_sha256": value["seal_sha256"],
        },
        "body_persistence": value["body_persistence"],
        "capability_seal_sha256": value["capability_seal_sha256"],
    })


def validate_largest_validation_evidence(
    evidence: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    value = verify_sealed(dict(evidence), label="largest-shard validation evidence")
    fields = {
        "schema", "status", "repo", "revision", "plan_seal_sha256", "lane",
        "selected_trial_id", "validation_pass", "identity", "measurement",
        "body_persistence", "capability_seal_sha256", "seal_sha256",
    }
    if set(value) != fields or value.get("schema") != LARGEST_EVIDENCE_SCHEMA \
            or value.get("status") != "PASS_FULL_HASH_IN_MEMORY" \
            or value.get("repo") != REPO_ID or value.get("revision") != REVISION \
            or value.get("plan_seal_sha256") != plan.get("seal_sha256"):
        raise Glm52Error("largest-shard validation evidence shape/identity mismatch")
    lane = value.get("lane")
    if lane not in {"acquisition", "steady"}:
        raise Glm52Error("largest-shard validation lane is invalid")
    largest = plan["largest_shard_validation"]
    pass_index = 0 if lane == "acquisition" else 1
    if value.get("validation_pass") != largest["passes"][pass_index] \
            or value.get("identity") != {
                "path": largest["path"],
                "bytes": largest["bytes"],
                "xet_hash": largest["xet_hash"],
                "expected_lfs_sha256": largest["lfs_sha256"],
            }:
        raise Glm52Error("largest-shard validation identity/pass changed")
    measurement = value.get("measurement")
    if not isinstance(measurement, Mapping) or set(measurement) != {
        "streamed_bytes", "observed_sha256", "actual_network_bytes",
        "trial_result_seal_sha256",
    } or measurement.get("streamed_bytes") != largest["bytes"] \
            or measurement.get("observed_sha256") != largest["lfs_sha256"] \
            or not _is_sha256(measurement.get("trial_result_seal_sha256")):
        raise Glm52Error("largest-shard validation measurement failed")
    _require_int(measurement.get("actual_network_bytes"), "validation actual network", minimum=1)
    if not _is_sha256(value.get("capability_seal_sha256")):
        raise Glm52Error("largest-shard validation capability binding is invalid")
    body = value.get("body_persistence")
    if not isinstance(body, Mapping) or body.get("python_body_file_writes") != 0 \
            or body.get("destructive_gc_performed") is not False:
        raise Glm52Error("largest-shard validation persisted body data or ran GC")
    return value


def _candidate_from_result(
    result: Mapping[str, Any],
    verdict: Mapping[str, Any],
) -> dict[str, Any]:
    measured = verdict.get("measured")
    if verdict.get("status") != "PASS" or not isinstance(measured, Mapping):
        raise Glm52Error(f"trial is not resource-safe: {result['trial_binding']['trial_id']}")
    stream = result["stream_measurement"]
    binding = result["trial_binding"]
    return {
        "trial_id": binding["trial_id"],
        "eligible_lanes": ["acquisition", "steady"],
        "resource_verdict": "PASS",
        "caller_concurrent_shard_streams": binding["caller_concurrent_shard_streams"],
        "throughput_bytes_per_second": stream["throughput_bytes_per_second"],
        "peak_rss_bytes": stream["peak_process_rss_bytes"],
        "effective_transfer_concurrency": stream["effective_transfer_concurrency"],
        **{key: measured[key] for key in autotune.SELECTABLE_TRIAL_MEASUREMENTS},
        "sustained_heavy_lane_regression": measured["sustained_heavy_lane_regression"],
    }


def assemble_autotune_result(
    plan: Mapping[str, Any],
    trial_results: Sequence[Mapping[str, Any]],
    largest_validations: Sequence[Mapping[str, Any]],
    *,
    required_free_bytes: int,
    required_available_ram_bytes: int,
    root: Path = REPO_ROOT,
    rebuild_plan: bool = True,
) -> dict[str, Any]:
    """Assemble all 12 trials and two full-hash passes into the selectable result."""
    value = validate_live_plan(plan, root=root, rebuild=rebuild_plan)
    free_floor = _require_int(required_free_bytes, "required_free_bytes")
    ram_floor = _require_int(required_available_ram_bytes, "required_available_ram_bytes")
    expected_ids = [row["trial_id"] for row in value["trial_matrix"]]
    if len(trial_results) != len(expected_ids):
        raise Glm52Error("autotune result requires exactly 12 trial results")
    validated: dict[str, dict[str, Any]] = {}
    range_hashes: dict[str, str] = {}
    evaluations: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for raw in trial_results:
        result = validate_trial_result(raw, plan=value)
        trial_id = result["trial_binding"]["trial_id"]
        if trial_id in validated:
            raise Glm52Error(f"duplicate live Xet trial result: {trial_id}")
        if result["status"] != "PASS_COMPLETE_MEASURED":
            raise Glm52Error(f"live Xet trial has incomplete required metrics: {trial_id}")
        validated[trial_id] = result
        for row in result["range_results"]:
            previous = range_hashes.setdefault(row["range_id_sha256"], row["sha256"])
            if previous != row["sha256"]:
                raise Glm52Error("same sealed range produced different SHA-256 bytes")
        resource = result["resource_observations"]
        verdict = autotune.evaluate_resource_trial(
            resource["before"],
            resource["samples"],
            resource["after"],
            required_free_bytes=free_floor,
            required_available_ram_bytes=ram_floor,
            trial_network_cap_bytes=result["network_accounting"]["trial_network_cap_bytes"],
            heavy_lane_regressions=resource["heavy_lane_regressions"],
            complete_source_views=resource["complete_source_views"],
        )
        candidate = _candidate_from_result(result, verdict)
        candidates.append(candidate)
        evaluations.append({
            "trial_id": trial_id,
            "trial_result_seal_sha256": result["seal_sha256"],
            "resource_verdict": verdict,
            "selection_candidate_sha256": _canonical_sha256(candidate),
        })
    if set(validated) != set(expected_ids):
        raise Glm52Error(
            f"autotune trial coverage differs: missing={sorted(set(expected_ids)-set(validated))} "
            f"unknown={sorted(set(validated)-set(expected_ids))}"
        )
    evaluations.sort(key=lambda item: expected_ids.index(item["trial_id"]))
    acquisition = autotune.select_profile(candidates, lane="acquisition")
    steady = autotune.select_profile(candidates, lane="steady")

    if len(largest_validations) != 2:
        raise Glm52Error("autotune result requires two full largest-shard validations")
    validations: dict[str, dict[str, Any]] = {}
    for raw in largest_validations:
        evidence = validate_largest_validation_evidence(raw, plan=value)
        lane = evidence["lane"]
        if lane in validations:
            raise Glm52Error("duplicate largest-shard validation lane")
        expected_selected = acquisition["trial_id"] if lane == "acquisition" else steady["trial_id"]
        if evidence["selected_trial_id"] != expected_selected:
            raise Glm52Error("largest-shard validation did not use the selected lane profile")
        validations[lane] = evidence
    if set(validations) != {"acquisition", "steady"}:
        raise Glm52Error("largest-shard validation lane coverage is incomplete")

    trial_payload = sum(
        result["stream_measurement"]["payload_bytes"] for result in validated.values()
    )
    validation_payload = sum(
        evidence["measurement"]["streamed_bytes"] for evidence in validations.values()
    )
    if trial_payload != value["network_budget"]["bounded_range_payload_bytes"] \
            or validation_payload != value["network_budget"]["largest_shard_validation_bytes"] \
            or trial_payload + validation_payload != value["network_budget"]["planned_maximum_bytes"]:
        raise Glm52Error("autotune payload accounting differs from the sealed plan")
    actual_network = sum(
        result["network_accounting"]["actual_network_bytes"]
        for result in validated.values()
    ) + sum(
        evidence["measurement"]["actual_network_bytes"]
        for evidence in validations.values()
    )
    hard_cap = value["network_budget"]["hard_cap_bytes"]
    if actual_network > hard_cap:
        raise Glm52Error("cumulative autotune network usage exceeds the sealed hard cap")

    toolchain = value["toolchain_binding"]
    input_refs = value["inputs"]
    source_refs = [
        dict(item) for item in input_refs
        if item.get("path") in {
            "GLM52_OFFICIAL_MANIFEST.json",
            "GLM52_SOURCE_FORMAT_LEDGER.json",
            "GLM52_SHARD_DEPENDENCY_GRAPH.json",
            "GLM52_SOURCE_ADMISSION.json",
        }
    ]
    selected_profile = {
        "acquisition": acquisition,
        "steady": steady,
    }
    result = seal({
        "schema": AUTOTUNE_RESULT_SCHEMA,
        "status": "PASS_LIVE_XET_AUTOTUNE_COMPLETE_SCHEDULE_REFREEZE_REQUIRED",
        "repo": REPO_ID,
        "revision": REVISION,
        "bindings": {
            "plan_seal_sha256": value["seal_sha256"],
            "plan_toolchain_binding_sha256": _canonical_sha256(toolchain),
            "plan_input_refs_sha256": _canonical_sha256(input_refs),
            "source_refs": source_refs,
            "live_executor_sha256": sha256_file(Path(__file__).resolve()),
        },
        "coverage": {
            "trial_ids_in_plan_order": expected_ids,
            "trial_results": evaluations,
            "required_file_settings": list(autotune.REQUIRED_FILE_SETTINGS),
            "required_file_settings_measured": sorted(
                int(item.removeprefix("FILES_"))
                for item in validated if item.startswith("FILES_")
            ),
            "fixed_profiles_measured": ["FIXED_16", "FIXED_32", "FIXED_64"],
            "high_performance_measured": True,
            "cache_profiles_measured": ["CACHE_1G_COLD", "CACHE_1G_REPLAY"],
            "unique_sealed_ranges_hashed": len(range_hashes),
            "repeated_range_sha256_consistent": True,
        },
        "selections": {
            "acquisition": acquisition,
            "steady": steady,
        },
        "selected_profile": selected_profile,
        "largest_shard_validations": [
            {
                "lane": lane,
                "evidence_seal_sha256": validations[lane]["seal_sha256"],
                "selected_trial_id": validations[lane]["selected_trial_id"],
                "observed_sha256": validations[lane]["measurement"]["observed_sha256"],
            }
            for lane in ("acquisition", "steady")
        ],
        "network_budget": {
            "planned_range_payload_bytes": trial_payload,
            "planned_full_validation_payload_bytes": validation_payload,
            "planned_total_payload_bytes": trial_payload + validation_payload,
            "actual_network_bytes": actual_network,
            "hard_cap_bytes": hard_cap,
            "remaining_bytes": hard_cap - actual_network,
            "protocol_overhead_and_retries_included": True,
        },
        "claim_boundary": {
            "xet_autotune_complete": True,
            "all_12_trials_measured": True,
            "two_largest_shard_full_hash_passes": True,
            "model_body_files_created_by_executor": 0,
            "full_model_downloaded": False,
            "model_capability_claimed": False,
            "streaming_schedule_refreeze_required": True,
        },
    })
    return validate_autotune_result(result, plan=value)


def validate_autotune_result(
    result: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate sealed overall shape and the controller-facing profile binding.

    This validates the self-contained overall artifact.  Rebuilding scientific
    measurements still requires the 12 trial artifacts and two full-shard
    evidence artifacts passed to :func:`assemble_autotune_result`.
    """
    value = verify_sealed(dict(result), label="live Xet autotune result")
    _exact_fields(value, AUTOTUNE_RESULT_FIELDS, "live Xet autotune result")
    if value.get("schema") != AUTOTUNE_RESULT_SCHEMA \
            or value.get("status") != (
                "PASS_LIVE_XET_AUTOTUNE_COMPLETE_SCHEDULE_REFREEZE_REQUIRED"
            ) or value.get("repo") != REPO_ID or value.get("revision") != REVISION:
        raise Glm52Error("live Xet autotune result schema/status/identity mismatch")
    selections = value.get("selections")
    selected_profile = value.get("selected_profile")
    if not isinstance(selections, Mapping) or set(selections) != {"acquisition", "steady"} \
            or not isinstance(selected_profile, Mapping) \
            or set(selected_profile) != {"acquisition", "steady"} \
            or any(not isinstance(selections.get(lane), Mapping) or not selections[lane]
                   for lane in ("acquisition", "steady")) \
            or dict(selected_profile) != {
                "acquisition": selections["acquisition"],
                "steady": selections["steady"],
            }:
        raise Glm52Error("live Xet selected_profile differs from canonical lane selections")
    expected_ids = [row["trial_id"] for row in plan.get("trial_matrix", [])]
    if len(expected_ids) != 12 or len(set(expected_ids)) != 12:
        raise Glm52Error("live Xet plan no longer has the exact 12-trial inventory")
    plan_trials = {row["trial_id"]: row for row in plan["trial_matrix"]}
    for lane in ("acquisition", "steady"):
        selection = selections[lane]
        if selection.get("lane") != lane or selection.get("status") != "SELECTED" \
                or selection.get("trial_id") not in plan_trials \
                or not isinstance(selection.get("selected_trial"), Mapping) \
                or selection["selected_trial"].get("trial_id") != selection["trial_id"] \
                or selection.get("selected_caller_concurrent_shard_streams") != plan_trials[
                    selection["trial_id"]
                ].get("caller_concurrent_shard_streams") \
                or selection.get("post_autotune_schedule_refreeze_required") is not True:
            raise Glm52Error(f"live Xet {lane} selection is invalid or not plan-bound")
    coverage = value.get("coverage")
    coverage_fields = {
        "trial_ids_in_plan_order", "trial_results", "required_file_settings",
        "required_file_settings_measured", "fixed_profiles_measured",
        "high_performance_measured", "cache_profiles_measured",
        "unique_sealed_ranges_hashed", "repeated_range_sha256_consistent",
    }
    if not isinstance(coverage, Mapping) or set(coverage) != coverage_fields \
            or coverage.get("trial_ids_in_plan_order") != expected_ids \
            or coverage.get("required_file_settings") != list(autotune.REQUIRED_FILE_SETTINGS) \
            or coverage.get("required_file_settings_measured") != list(
                autotune.REQUIRED_FILE_SETTINGS
            ) or coverage.get("fixed_profiles_measured") != [
                "FIXED_16", "FIXED_32", "FIXED_64"
            ] or coverage.get("high_performance_measured") is not True \
            or coverage.get("cache_profiles_measured") != [
                "CACHE_1G_COLD", "CACHE_1G_REPLAY"
            ] or coverage.get("unique_sealed_ranges_hashed") != len(
                plan.get("range_strategy", {}).get("body_ranges", [])
            ) or coverage.get("repeated_range_sha256_consistent") is not True:
        raise Glm52Error("live Xet autotune trial/profile coverage differs from the plan")
    trial_refs = coverage.get("trial_results")
    if not isinstance(trial_refs, list) or len(trial_refs) != 12 \
            or [item.get("trial_id") for item in trial_refs if isinstance(item, Mapping)] != expected_ids:
        raise Glm52Error("live Xet autotune result references are not exact plan order")
    for item in trial_refs:
        if not isinstance(item, Mapping) or set(item) != {
            "trial_id", "trial_result_seal_sha256", "resource_verdict",
            "selection_candidate_sha256",
        } or not _is_sha256(item.get("trial_result_seal_sha256")) \
                or not _is_sha256(item.get("selection_candidate_sha256")) \
                or not isinstance(item.get("resource_verdict"), Mapping) \
                or item["resource_verdict"].get("status") != "PASS" \
                or not isinstance(item["resource_verdict"].get("measured"), Mapping):
            raise Glm52Error("live Xet trial reference/resource verdict is invalid")
    largest_refs = value.get("largest_shard_validations")
    largest = plan.get("largest_shard_validation", {})
    if not isinstance(largest_refs, list) or len(largest_refs) != 2 \
            or [item.get("lane") for item in largest_refs if isinstance(item, Mapping)] != [
                "acquisition", "steady"
            ]:
        raise Glm52Error("live Xet largest-shard validation lane coverage is invalid")
    for item in largest_refs:
        lane = item["lane"]
        if set(item) != {
            "lane", "evidence_seal_sha256", "selected_trial_id", "observed_sha256",
        } or not _is_sha256(item.get("evidence_seal_sha256")) \
                or item.get("selected_trial_id") != selections[lane].get("trial_id") \
                or item.get("observed_sha256") != largest.get("lfs_sha256"):
            raise Glm52Error("live Xet largest-shard evidence/profile/hash binding is invalid")
    bindings = value.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "plan_seal_sha256", "plan_toolchain_binding_sha256",
        "plan_input_refs_sha256", "source_refs", "live_executor_sha256",
    } or bindings.get("plan_seal_sha256") != plan.get("seal_sha256") \
            or bindings.get("plan_toolchain_binding_sha256") != _canonical_sha256(
                plan.get("toolchain_binding")
            ) or bindings.get("plan_input_refs_sha256") != _canonical_sha256(
                plan.get("inputs")
            ) or bindings.get("live_executor_sha256") != sha256_file(Path(__file__).resolve()):
        raise Glm52Error("live Xet autotune result plan/tool/input binding mismatch")
    source_refs = [
        dict(item) for item in plan.get("inputs", [])
        if isinstance(item, Mapping) and item.get("path") in {
            "GLM52_OFFICIAL_MANIFEST.json",
            "GLM52_SOURCE_FORMAT_LEDGER.json",
            "GLM52_SHARD_DEPENDENCY_GRAPH.json",
            "GLM52_SOURCE_ADMISSION.json",
        }
    ]
    if bindings.get("source_refs") != source_refs:
        raise Glm52Error("live Xet autotune result source seals differ from the plan")
    network = value.get("network_budget")
    if not isinstance(network, Mapping) or set(network) != {
        "planned_range_payload_bytes", "planned_full_validation_payload_bytes",
        "planned_total_payload_bytes", "actual_network_bytes", "hard_cap_bytes",
        "remaining_bytes", "protocol_overhead_and_retries_included",
    }:
        raise Glm52Error("live Xet autotune result network accounting shape is invalid")
    planned_range = _require_int(
        network.get("planned_range_payload_bytes"), "overall planned range bytes", minimum=1
    )
    planned_validation = _require_int(
        network.get("planned_full_validation_payload_bytes"),
        "overall planned validation bytes",
        minimum=1,
    )
    planned_total = _require_int(
        network.get("planned_total_payload_bytes"), "overall planned total bytes", minimum=1
    )
    actual = _require_int(network.get("actual_network_bytes"), "overall actual network bytes", minimum=1)
    hard = _require_int(network.get("hard_cap_bytes"), "overall hard cap bytes", minimum=1)
    remaining = _require_int(network.get("remaining_bytes"), "overall remaining bytes")
    plan_budget = plan.get("network_budget", {})
    if planned_range + planned_validation != planned_total \
            or planned_range != plan_budget.get("bounded_range_payload_bytes") \
            or planned_validation != plan_budget.get("largest_shard_validation_bytes") \
            or planned_total != plan_budget.get("planned_maximum_bytes") \
            or hard != plan_budget.get("hard_cap_bytes") \
            or actual > hard or remaining != hard - actual \
            or network.get("protocol_overhead_and_retries_included") is not True:
        raise Glm52Error("live Xet autotune result network accounting arithmetic failed")
    claims = value.get("claim_boundary")
    if not isinstance(claims, Mapping) \
            or claims.get("xet_autotune_complete") is not True \
            or claims.get("all_12_trials_measured") is not True \
            or claims.get("two_largest_shard_full_hash_passes") is not True \
            or claims.get("model_body_files_created_by_executor") != 0 \
            or claims.get("full_model_downloaded") is not False \
            or claims.get("model_capability_claimed") is not False \
            or claims.get("streaming_schedule_refreeze_required") is not True:
        raise Glm52Error("live Xet autotune result claim boundary is invalid")
    return value


def _child_main() -> int:
    if os.environ.get(CHILD_ENV) != "1":
        raise Glm52Error("private live Xet child command refused")
    raw = sys.stdin.readline()
    if not raw:
        raise Glm52Error("live Xet child received no spec")
    try:
        spec = verify_sealed(json.loads(raw), label="child trial spec")
    except (json.JSONDecodeError, TypeError) as exc:
        raise Glm52Error("live Xet child spec is invalid JSON") from exc
    # Full plan validation happens in the trusted parent before spawn.  The
    # child rechecks its self-contained shape, sealed targets, and environment.
    _exact_fields(spec, SPEC_FIELDS, "child live Xet spec")
    _validate_child_environment(spec)
    print(json.dumps({
        "protocol": CHILD_PROTOCOL,
        "status": "READY",
        "spec_seal_sha256": spec["seal_sha256"],
    }, sort_keys=True, separators=(",", ":")), flush=True)
    command = json.loads(sys.stdin.readline())
    if command != {"command": "GO", "spec_seal_sha256": spec["seal_sha256"]}:
        raise Glm52Error("live Xet child GO command is invalid")
    result = stream_spec_in_memory(spec)
    print(json.dumps({
        "protocol": CHILD_PROTOCOL,
        "status": "RESULT",
        "result": result,
    }, sort_keys=True, separators=(",", ":")), flush=True)
    acknowledgement = json.loads(sys.stdin.readline())
    if acknowledgement != {"command": "ACK", "spec_seal_sha256": spec["seal_sha256"]}:
        raise Glm52Error("live Xet child ACK command is invalid")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    child = subparsers.add_parser("_child", help=argparse.SUPPRESS)
    child.set_defaults(handler=lambda _args: _child_main())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Glm52Error as exc:
        print(json.dumps({"status": "REFUSED", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

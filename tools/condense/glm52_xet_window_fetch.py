#!/usr/bin/env python3.12
"""Authorized, selected-profile-bound Xet materialization for one GLM-5.2 window.

Importing this module performs no network access and reads no model body.  A live
fetch is possible only through :func:`materialize_window`, after all of these have
been validated together:

* the expected-contract v3 and producer-authenticated frozen schedule;
* the exact offline Xet plan and producer-attested live autotune result;
* the exact frozen resource-reserve policy;
* a producer-authenticated fetch intent derived from the scheduled new/refetch set;
* an independent live capability verifier; and
* a fresh authenticated disk/RAM/swap sample for the one anchored destination root.

The adapter streams each full shard to an exclusive same-directory partial, hashes
while writing, fsyncs and verifies the descriptor, then publishes with a no-replace
rename only after every shard is complete.  Failures never unlink or overwrite any
path.  Unpublished partials remain for explicit forensic recovery/quarantine.

This module is intentionally not imported or dispatched by ``glm52_worker``.  Its
existence does not enable scientific dispatch.
"""
from __future__ import annotations

import copy
import ctypes
import errno
import fcntl
import hashlib
import hmac
import os
import re
import stat
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_grounding as grounding  # noqa: E402
import glm52_schedule_freeze as schedule_freeze  # noqa: E402
import glm52_state as state  # noqa: E402
import glm52_xet_live as xet_live  # noqa: E402
from glm52_common import Glm52Error, canonical, seal, utc_now, verify_sealed  # noqa: E402


FETCH_INTENT_SCHEMA = "hawking.glm52.xet_window_fetch_intent.v1"
FETCH_RECEIPT_SCHEMA = "hawking.glm52.xet_window_fetch_receipt.v1"
RESOURCE_POLICY_SCHEMA = "hawking.glm52.resource_reserve_policy.v1"
RESOURCE_POLICY_STATUS = "FROZEN_CONSERVATIVE_PRELIVE_POLICY"
RESOURCE_MAX_AGE_SECONDS = 120
MAX_CALLER_CONCURRENCY = 48
MAX_STREAM_CHUNK_BYTES = 256 * 1024**2
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


class WindowFetchError(Glm52Error):
    """A fetch was refused or failed without deleting any partial/published path."""

    def __init__(
        self,
        message: str,
        *,
        retained_partials: Sequence[str] = (),
        published_paths: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.retained_partials = tuple(retained_partials)
        self.published_paths = tuple(published_paths)


@dataclass(frozen=True, slots=True)
class FetchArtifacts:
    """The exact immutable artifact set required to derive one fetch intent."""

    expected_contract: Mapping[str, Any]
    preliminary_schedule: Mapping[str, Any]
    xet_autotune_plan: Mapping[str, Any]
    producer_attested_xet_result: Mapping[str, Any]
    frozen_schedule: Mapping[str, Any]
    resource_policy: Mapping[str, Any]


class FetchCapabilityVerifier(Protocol):
    """Executor-owned check against live lease/controller state and one-use intent."""

    def verify_window_fetch_capability(
        self,
        intent: Mapping[str, Any],
        *,
        frozen_schedule: Mapping[str, Any],
        resource_policy: Mapping[str, Any],
    ) -> bool:
        """Return exactly ``True`` only while this exact intent remains executable."""


class StreamProvider(Protocol):
    """One prepared stream group configured from the selected sealed plan row."""

    def open_stream(self, target: Mapping[str, Any]) -> Iterable[bytes]:
        """Yield the exact full-file byte stream for ``target``."""

    def abort(self) -> None:
        """Abort every active stream in this provider."""


class ResourceSampler(Protocol):
    """Trusted resource facts plus an inline descriptor-relative disk counter."""

    def authenticated_sample(
        self,
        root: str | os.PathLike[str],
        *,
        root_id: str,
        policy: grounding.ResourceReservePolicy,
        authenticator: grounding.ProducerAuthenticator,
    ) -> Mapping[str, Any]:
        """Return a fresh authenticated resource observation or raise."""

    def free_disk_bytes(self, root_fd: int) -> int:
        """Return immediately available bytes for the filesystem containing root_fd."""

    def allocation_unit_bytes(self, root_fd: int) -> int:
        """Return a positive conservative allocation unit for inline reservations."""


class GroundedResourceSampler:
    """Production sampler backed by authenticated grounding and ``fstatvfs``."""

    def authenticated_sample(
        self,
        root: str | os.PathLike[str],
        *,
        root_id: str,
        policy: grounding.ResourceReservePolicy,
        authenticator: grounding.ProducerAuthenticator,
    ) -> Mapping[str, Any]:
        return grounding.sample_resources(
            root,
            root_id=root_id,
            policy=policy,
            authenticator=authenticator,
        )

    def free_disk_bytes(self, root_fd: int) -> int:
        sample = os.fstatvfs(root_fd)
        return int(sample.f_bavail) * int(sample.f_frsize)

    def allocation_unit_bytes(self, root_fd: int) -> int:
        sample = os.fstatvfs(root_fd)
        unit = int(sample.f_frsize or sample.f_bsize)
        if unit <= 0:
            raise WindowFetchError("filesystem reported no positive allocation unit")
        return unit


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _clone(value: Any) -> Any:
    return copy.deepcopy(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise WindowFetchError(
            f"{label} fields differ: missing={sorted(expected - set(value))} "
            f"unknown={sorted(set(value) - expected)}"
        )


def _relative_path(value: object, label: str) -> tuple[str, tuple[str, ...]]:
    try:
        return grounding._relative_parts(value)
    except grounding.GroundingError as exc:
        raise WindowFetchError(f"{label}: {exc}") from exc


def _producer_verify(
    value: Mapping[str, Any],
    *,
    auth: state.EvidenceAuthConfig,
    label: str,
) -> dict[str, Any]:
    try:
        candidate = verify_sealed(dict(value), label=label)
    except Glm52Error as exc:
        raise WindowFetchError(str(exc)) from exc
    if not isinstance(auth, state.EvidenceAuthConfig):
        raise WindowFetchError(f"{label} requires the live evidence authenticator")
    if candidate.get("campaign_id") != auth.campaign_id \
            or candidate.get("source_revision") != auth.source_revision:
        raise WindowFetchError(f"{label} authenticator identity mismatch")
    unsigned = {
        key: _clone(item)
        for key, item in candidate.items()
        if key not in {"seal_sha256", "producer_hmac_sha256"}
    }
    if not auth.verify(
        {
            "schema": "hawking.glm52.evidence_producer_auth.v1",
            "artifact": unsigned,
        },
        candidate.get("producer_hmac_sha256"),
    ):
        raise WindowFetchError(f"{label} producer HMAC mismatch")
    return candidate


@dataclass(frozen=True, slots=True)
class _BindingContext:
    contract: dict[str, Any]
    plan: dict[str, Any]
    schedule: dict[str, Any]
    resource_artifact: dict[str, Any]
    resource_policy: grounding.ResourceReservePolicy


def _validate_resource_policy(
    value: Mapping[str, Any], *, source_revision: str
) -> tuple[dict[str, Any], grounding.ResourceReservePolicy]:
    try:
        policy = verify_sealed(dict(value), label="GLM52 resource reserve policy")
    except Glm52Error as exc:
        raise WindowFetchError(str(exc)) from exc
    expected_seal = state.OFFICIAL_RESOURCE_POLICY_SEAL_SHA256
    if policy.get("schema") != RESOURCE_POLICY_SCHEMA \
            or policy.get("status") != RESOURCE_POLICY_STATUS \
            or policy.get("seal_sha256") != expected_seal \
            or policy.get("repo") != xet_live.REPO_ID \
            or policy.get("revision") != source_revision:
        raise WindowFetchError("resource policy schema/status/seal/source binding mismatch")
    raw = policy.get("policy")
    if not isinstance(raw, dict):
        raise WindowFetchError("resource policy body is absent")
    try:
        reserve = grounding.ResourceReservePolicy(**raw)
    except (TypeError, grounding.GroundingError) as exc:
        raise WindowFetchError(f"resource policy body is invalid: {exc}") from exc
    derived = policy.get("derived")
    if not isinstance(derived, dict) \
            or derived.get("operational_reserve_floor_bytes") != \
            reserve.operational_reserve_floor_bytes \
            or derived.get("additional_reserved_bytes") != reserve.additional_reserved_bytes \
            or derived.get("required_free_disk_bytes") != reserve.required_free_disk_bytes \
            or reserve.required_free_disk_bytes != \
            state.OFFICIAL_RESOURCE_POLICY_REQUIRED_FREE_DISK_BYTES:
        raise WindowFetchError("resource policy derived reserve arithmetic mismatch")
    activation = policy.get("activation")
    prefetch = policy.get("prefetch_control")
    if not isinstance(activation, dict) \
            or activation.get(
                "live_allocated_byte_measurement_required_before_materialized_xet_body_acquisition"
            ) is not True \
            or activation.get("worker_must_hold_controller_lease") is not True \
            or activation.get("remote_logical_bytes_authorize_materialized_body_acquisition") \
            is not False \
            or not isinstance(prefetch, dict) \
            or prefetch.get("mode") != "SERIALIZED_OR_PARTIAL_PREFETCH" \
            or prefetch.get("full_two_complete_window_pipeline_preregistered") is not False:
        raise WindowFetchError("resource policy weakens live materialization controls")
    return policy, reserve


def _validate_binding_bundle(
    artifacts: FetchArtifacts,
    *,
    auth: state.EvidenceAuthConfig,
    root: Path,
    rebuild_plan: bool,
) -> _BindingContext:
    if not isinstance(artifacts, FetchArtifacts):
        raise WindowFetchError("FetchArtifacts is required")
    try:
        contract = state._validate_expected_contract(
            _clone(dict(artifacts.expected_contract))
        )
    except (state.StateError, TypeError, ValueError) as exc:
        raise WindowFetchError(f"expected campaign contract is invalid: {exc}") from exc
    if not isinstance(auth, state.EvidenceAuthConfig) \
            or auth.campaign_id != contract["campaign_id"] \
            or auth.source_revision != contract["source_revision"]:
        raise WindowFetchError("evidence authenticator differs from expected contract")
    try:
        schedule = schedule_freeze.validate_frozen_schedule(
            artifacts.frozen_schedule,
            artifacts.preliminary_schedule,
            artifacts.xet_autotune_plan,
            artifacts.producer_attested_xet_result,
            contract,
            auth=auth,
            root=root,
            rebuild_plan=rebuild_plan,
        )
        plan = xet_live.validate_live_plan(
            artifacts.xet_autotune_plan,
            root=root,
            rebuild=rebuild_plan,
        )
    except (Glm52Error, KeyError, TypeError, ValueError) as exc:
        raise WindowFetchError(f"frozen Xet/schedule binding failed: {exc}") from exc
    resource_artifact, reserve = _validate_resource_policy(
        artifacts.resource_policy,
        source_revision=contract["source_revision"],
    )
    expected_resource_binding = {
        "path": state.OFFICIAL_RESOURCE_POLICY_PATH,
        "seal_sha256": resource_artifact["seal_sha256"],
        "required_free_disk_bytes": reserve.required_free_disk_bytes,
        "expected_contract_sha256": contract["seal_sha256"],
    }
    if schedule.get("resource_policy_binding") != expected_resource_binding:
        raise WindowFetchError("frozen schedule resource-policy binding is not exact")
    return _BindingContext(
        contract=contract,
        plan=plan,
        schedule=schedule,
        resource_artifact=resource_artifact,
        resource_policy=reserve,
    )


def _window_and_targets(
    context: _BindingContext, schedule_index: int, lane: str
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if type(schedule_index) is not int or schedule_index < 0:
        raise WindowFetchError("schedule_index must be a nonnegative integer")
    if lane not in {"acquisition", "steady"}:
        raise WindowFetchError("fetch lane must be acquisition or steady")
    windows = context.schedule.get("windows")
    if not isinstance(windows, list) or schedule_index >= len(windows):
        raise WindowFetchError("schedule_index is outside the frozen schedule")
    window = dict(windows[schedule_index])
    if window.get("schedule_index") != schedule_index:
        raise WindowFetchError("frozen window index is not canonical")
    new_fetch = window.get("new_fetch_shards")
    refetch = window.get("refetch_shards")
    if not isinstance(new_fetch, list) or not isinstance(refetch, list):
        raise WindowFetchError("frozen window fetch membership is malformed")
    ordered_paths = [*new_fetch, *refetch]
    if not ordered_paths or len(ordered_paths) != len(set(ordered_paths)):
        raise WindowFetchError("frozen window fetch set is empty or duplicated")
    sources = {
        row["path"]: row for row in context.contract["source"]["shards"]
    }
    if any(path not in sources for path in ordered_paths):
        raise WindowFetchError("frozen window fetch set contains an unknown source shard")
    targets = []
    new_set = set(new_fetch)
    for path in ordered_paths:
        normalized, _parts = _relative_path(path, "scheduled source path")
        source = sources[normalized]
        targets.append(
            {
                "path": normalized,
                "role": "NEW_FETCH" if normalized in new_set else "REFETCH",
                "logical_bytes": source["logical_bytes"],
                "xet_hash": source["xet_hash"],
                "lfs_sha256": source["lfs_sha256"],
            }
        )
    selected = context.schedule.get("selected_profile", {}).get(lane)
    if not isinstance(selected, dict):
        raise WindowFetchError("frozen schedule lacks the selected fetch lane")
    trial_id = selected.get("trial_id")
    matches = [
        dict(row) for row in context.plan["trial_matrix"]
        if row.get("trial_id") == trial_id
    ]
    if len(matches) != 1:
        raise WindowFetchError("selected profile does not resolve to one sealed plan row")
    trial = matches[0]
    concurrency = selected.get("selected_caller_concurrent_shard_streams")
    if type(concurrency) is not int or not 1 <= concurrency <= MAX_CALLER_CONCURRENCY \
            or concurrency != trial.get("caller_concurrent_shard_streams"):
        raise WindowFetchError("selected caller concurrency differs from sealed plan")
    return window, targets, selected, trial


def _open_root(
    root: str | os.PathLike[str],
) -> tuple[str, list[int], list[tuple[str, tuple[int, int, int]]], os.stat_result]:
    try:
        root_path = grounding._normalized_absolute_root(root)
        fds, links, root_stat = grounding._open_absolute_directory_chain(root_path)
    except grounding.GroundingError as exc:
        raise WindowFetchError(str(exc)) from exc
    return root_path, fds, links, root_stat


def _root_id(contract: Mapping[str, Any]) -> str:
    return (
        f"glm52-source:{contract['campaign_id']}:"
        f"{contract['seal_sha256']}"
    )


def _target_intent_rows(targets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_clone(dict(target)) for target in targets]


def build_window_fetch_intent(
    artifacts: FetchArtifacts,
    *,
    schedule_index: int,
    lane: str,
    source_root: str | os.PathLike[str],
    controller_anchor_sha256: str,
    authority_nonce_sha256: str,
    auth: state.EvidenceAuthConfig,
    root: Path = xet_live.REPO_ROOT,
    rebuild_plan: bool = True,
) -> dict[str, Any]:
    """Derive and authenticate an exact fetch intent without opening a body stream."""
    if not _is_sha256(controller_anchor_sha256) \
            or not _is_sha256(authority_nonce_sha256):
        raise WindowFetchError("fetch authority anchor/nonce must be 64 lowercase hex")
    context = _validate_binding_bundle(
        artifacts,
        auth=auth,
        root=Path(root),
        rebuild_plan=rebuild_plan,
    )
    window, targets, selected, trial = _window_and_targets(
        context, schedule_index, lane
    )
    root_path, root_fds, root_links, root_stat = _open_root(source_root)
    try:
        grounding._verify_absolute_directory_chain(root_fds, root_links, root_stat)
        root_identity = {
            "normalized_path_sha256": hashlib.sha256(
                root_path.encode("utf-8")
            ).hexdigest(),
            "device": int(root_stat.st_dev),
            "inode": int(root_stat.st_ino),
        }
    except grounding.GroundingError as exc:
        raise WindowFetchError(str(exc)) from exc
    finally:
        for fd in reversed(root_fds):
            os.close(fd)
    total = sum(int(target["logical_bytes"]) for target in targets)
    body = {
        "schema": FETCH_INTENT_SCHEMA,
        "status": "PREPARED_EXPLICIT_ADAPTER_CALL_ONLY",
        "campaign_id": context.contract["campaign_id"],
        "source_revision": context.contract["source_revision"],
        "expected_contract_sha256": context.contract["seal_sha256"],
        "xet_autotune_plan_seal_sha256": context.plan["seal_sha256"],
        "producer_attested_xet_result_seal_sha256": artifacts.producer_attested_xet_result[
            "seal_sha256"
        ],
        "frozen_schedule_seal_sha256": context.schedule["seal_sha256"],
        "resource_policy_seal_sha256": context.resource_artifact["seal_sha256"],
        "controller_anchor_sha256": controller_anchor_sha256,
        "authority_nonce_sha256": authority_nonce_sha256,
        "schedule_index": schedule_index,
        "window_id": window["window_id"],
        "lane": lane,
        "selected_profile_sha256": _sha(selected),
        "selected_trial_id": trial["trial_id"],
        "selected_plan_trial_sha256": _sha(trial),
        "caller_concurrent_shard_streams": trial[
            "caller_concurrent_shard_streams"
        ],
        "source_root": root_identity,
        "authoritative_target_view_count": 1,
        "targets": _target_intent_rows(targets),
        "maximum_streamed_body_bytes": total,
        "prepared_at": utc_now(),
        "credentials_serialized": False,
        "worker_dispatch_enabled": False,
    }
    try:
        return state.seal_producer_authenticated_evidence(body, auth=auth)
    except state.StateError as exc:
        raise WindowFetchError(f"cannot authenticate fetch intent: {exc}") from exc


_INTENT_FIELDS = {
    "schema", "status", "campaign_id", "source_revision",
    "expected_contract_sha256", "xet_autotune_plan_seal_sha256",
    "producer_attested_xet_result_seal_sha256", "frozen_schedule_seal_sha256",
    "resource_policy_seal_sha256", "controller_anchor_sha256",
    "authority_nonce_sha256", "schedule_index", "window_id", "lane",
    "selected_profile_sha256", "selected_trial_id", "selected_plan_trial_sha256",
    "caller_concurrent_shard_streams", "source_root",
    "authoritative_target_view_count", "targets", "maximum_streamed_body_bytes",
    "prepared_at", "credentials_serialized", "worker_dispatch_enabled",
    "producer_hmac_sha256", "seal_sha256",
}


def _validate_intent(
    intent: Mapping[str, Any],
    context: _BindingContext,
    *,
    artifacts: FetchArtifacts,
    auth: state.EvidenceAuthConfig,
    root_path: str,
    root_stat: os.stat_result,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    value = _producer_verify(intent, auth=auth, label="Xet window fetch intent")
    _exact_keys(value, _INTENT_FIELDS, "Xet window fetch intent")
    if value.get("schema") != FETCH_INTENT_SCHEMA \
            or value.get("status") != "PREPARED_EXPLICIT_ADAPTER_CALL_ONLY" \
            or value.get("campaign_id") != context.contract["campaign_id"] \
            or value.get("source_revision") != context.contract["source_revision"] \
            or value.get("expected_contract_sha256") != context.contract["seal_sha256"] \
            or value.get("xet_autotune_plan_seal_sha256") != context.plan["seal_sha256"] \
            or value.get("producer_attested_xet_result_seal_sha256") != \
            artifacts.producer_attested_xet_result.get("seal_sha256") \
            or value.get("frozen_schedule_seal_sha256") != context.schedule["seal_sha256"] \
            or value.get("resource_policy_seal_sha256") != \
            context.resource_artifact["seal_sha256"]:
        raise WindowFetchError("fetch intent immutable artifact binding mismatch")
    if not _is_sha256(value.get("controller_anchor_sha256")) \
            or not _is_sha256(value.get("authority_nonce_sha256")) \
            or value.get("credentials_serialized") is not False \
            or value.get("worker_dispatch_enabled") is not False \
            or value.get("authoritative_target_view_count") != 1:
        raise WindowFetchError("fetch intent authority/claim boundary is invalid")
    window, targets, selected, trial = _window_and_targets(
        context,
        value.get("schedule_index"),
        value.get("lane"),
    )
    expected_root = {
        "normalized_path_sha256": hashlib.sha256(root_path.encode("utf-8")).hexdigest(),
        "device": int(root_stat.st_dev),
        "inode": int(root_stat.st_ino),
    }
    expected_total = sum(int(target["logical_bytes"]) for target in targets)
    if value.get("window_id") != window["window_id"] \
            or value.get("selected_profile_sha256") != _sha(selected) \
            or value.get("selected_trial_id") != trial["trial_id"] \
            or value.get("selected_plan_trial_sha256") != _sha(trial) \
            or value.get("caller_concurrent_shard_streams") != \
            trial["caller_concurrent_shard_streams"] \
            or value.get("source_root") != expected_root \
            or value.get("targets") != _target_intent_rows(targets) \
            or value.get("maximum_streamed_body_bytes") != expected_total:
        raise WindowFetchError("fetch intent differs from the scheduled authoritative view")
    if not isinstance(value.get("prepared_at"), str) or not value["prepared_at"]:
        raise WindowFetchError("fetch intent prepared_at is absent")
    return value, targets, trial


def _selected_xet_config(
    plan: Mapping[str, Any], trial: Mapping[str, Any]
) -> dict[str, Any]:
    """Translate the sealed alias row into an explicit per-session XetConfig."""
    compatibility = plan.get("runtime_compatibility")
    if not isinstance(compatibility, Mapping):
        raise WindowFetchError("Xet plan runtime compatibility is absent")
    environment = trial.get("environment")
    if not isinstance(environment, Mapping):
        raise WindowFetchError("selected trial environment is absent")
    trial_id = trial.get("trial_id")
    if environment.get("HF_XET_HIGH_PERFORMANCE") == "1":
        base = compatibility.get("effective_high_performance")
    elif "HF_XET_CHUNK_CACHE_SIZE_BYTES" in environment:
        base = compatibility.get("effective_cache_1g")
    else:
        base = compatibility.get("effective_default")
    if not isinstance(base, Mapping) or not base:
        raise WindowFetchError("selected Xet effective configuration is absent")
    result = dict(base)
    if "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS" in environment:
        try:
            result["data.max_concurrent_file_downloads"] = int(
                environment["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"]
            )
        except (TypeError, ValueError) as exc:
            raise WindowFetchError("selected file concurrency is invalid") from exc
    if "HF_XET_FIXED_DOWNLOAD_CONCURRENCY" in environment:
        try:
            fixed = int(environment["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"])
        except (TypeError, ValueError) as exc:
            raise WindowFetchError("selected fixed Xet concurrency is invalid") from exc
        for key in (
            "client.ac_min_download_concurrency",
            "client.ac_initial_download_concurrency",
            "client.ac_max_download_concurrency",
        ):
            result[key] = fixed
    allowed = {
        "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS",
        "HF_XET_FIXED_DOWNLOAD_CONCURRENCY",
        "HF_XET_HIGH_PERFORMANCE",
        "HF_XET_CHUNK_CACHE_SIZE_BYTES",
    }
    if set(environment) - allowed:
        raise WindowFetchError("selected trial contains an unsupported Xet setting")
    if not isinstance(trial_id, str) or not trial_id:
        raise WindowFetchError("selected trial ID is invalid")
    return result


class HfXetStreamProvider:
    """Lazy real Xet provider using an explicit per-session config, not global env."""

    def __init__(self, config_values: Mapping[str, Any]) -> None:
        # The imports and token-refresh group creation occur only after all authority
        # and resource gates.  Importing this module alone remains body/network free.
        try:
            from hf_xet import XetConfig, XetFileInfo, XetSession
            from huggingface_hub.utils import build_hf_headers
            from huggingface_hub.utils._xet import (
                XetTokenType,
                xet_connection_info_refresh_url,
                xet_headers_without_auth,
            )
        except ImportError as exc:
            raise WindowFetchError("pinned Xet runtime is unavailable") from exc
        config = XetConfig().with_config(dict(config_values))
        effective = dict(config.items())
        if any(effective.get(key) != value for key, value in config_values.items()):
            raise WindowFetchError("explicit Xet session configuration did not bind")
        headers = build_hf_headers(
            token=False,
            library_name="hawking-glm52-window-fetch",
            library_version="1",
        )
        if any(key.lower() == "authorization" for key in headers):
            raise WindowFetchError("public Xet headers unexpectedly contain authorization")
        refresh_url = xet_connection_info_refresh_url(
            token_type=XetTokenType.READ,
            repo_id=xet_live.REPO_ID,
            repo_type="model",
            revision=xet_live.REVISION,
        )
        self._session = XetSession(config)
        self._group = self._session.new_download_stream_group(
            token_refresh_url=refresh_url,
            token_refresh_headers=headers,
            custom_headers=xet_headers_without_auth(headers),
        )
        self._file_info = XetFileInfo

    def open_stream(self, target: Mapping[str, Any]) -> Iterable[bytes]:
        return self._group.download_stream(
            self._file_info(target["xet_hash"], target["logical_bytes"]),
            start=0,
            end=target["logical_bytes"],
        )

    def abort(self) -> None:
        self._session.sigint_abort()


def _verify_resource_receipt(
    receipt: Mapping[str, Any],
    *,
    context: _BindingContext,
    root_id: str,
    root_stat: os.stat_result,
    authenticator: grounding.ProducerAuthenticator,
) -> dict[str, Any]:
    try:
        value = grounding.verify_authenticated_observation(
            receipt,
            authenticator,
            max_age_seconds=RESOURCE_MAX_AGE_SECONDS,
        )
    except (grounding.GroundingError, Glm52Error) as exc:
        raise WindowFetchError(f"resource observation is invalid: {exc}") from exc
    if value.get("schema") != grounding.RESOURCE_SAMPLE_SCHEMA \
            or value.get("status") != "PASS" \
            or value.get("observation_kind") != "live_disk_ram_swap_resources" \
            or value.get("root_id") != root_id \
            or value.get("root_device") != int(root_stat.st_dev) \
            or value.get("root_inode") != int(root_stat.st_ino) \
            or value.get("resource_policy") != context.resource_policy.as_dict() \
            or value.get("required_free_disk_bytes") != \
            context.resource_policy.required_free_disk_bytes \
            or value.get("disk_operational_reserve_ok") is not True \
            or value.get("available_ram_floor_ok") is not True \
            or value.get("swap_usage_ceiling_ok") is not True \
            or value.get("refusal_reasons") != []:
        raise WindowFetchError("resource observation does not authorize this source root")
    return value


def _open_relative_parent(
    root_fd: int, parts: Sequence[str]
) -> tuple[list[int], list[tuple[str, tuple[int, int, int]]]]:
    if not _NOFOLLOW or not _DIRECTORY or not _CLOEXEC:
        raise WindowFetchError("O_NOFOLLOW/O_DIRECTORY/O_CLOEXEC are required")
    fds = [os.dup(root_fd)]
    links: list[tuple[str, tuple[int, int, int]]] = []
    flags = os.O_RDONLY | _NOFOLLOW | _DIRECTORY | _CLOEXEC
    try:
        for component in parts:
            named = os.stat(component, dir_fd=fds[-1], follow_symlinks=False)
            if stat.S_ISLNK(named.st_mode) or not stat.S_ISDIR(named.st_mode):
                raise WindowFetchError(
                    f"scheduled source parent is not a real directory: {component!r}"
                )
            child = os.open(component, flags, dir_fd=fds[-1])
            opened = os.fstat(child)
            if grounding._type_identity(named) != grounding._type_identity(opened):
                os.close(child)
                raise WindowFetchError("scheduled source parent changed while opening")
            links.append((component, grounding._type_identity(opened)))
            fds.append(child)
        return fds, links
    except BaseException:
        for fd in reversed(fds):
            os.close(fd)
        raise


def _verify_relative_parent(
    fds: Sequence[int], links: Sequence[tuple[str, tuple[int, int, int]]]
) -> None:
    if len(fds) != len(links) + 1:
        raise WindowFetchError("scheduled source parent descriptor chain is malformed")
    for index, (component, expected) in enumerate(links):
        named = os.stat(component, dir_fd=fds[index], follow_symlinks=False)
        opened = os.fstat(fds[index + 1])
        if stat.S_ISLNK(named.st_mode) \
                or grounding._type_identity(named) != expected \
                or grounding._type_identity(opened) != expected:
            raise WindowFetchError("scheduled source parent identity changed")


def _entry_absent(parent_fd: int, name: str, *, label: str) -> None:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise WindowFetchError(f"cannot inspect {label}: {exc}") from exc
    raise WindowFetchError(f"{label} already exists; overwrite is forbidden")


def _safe_file_stat(metadata: os.stat_result, *, label: str) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise WindowFetchError(f"{label} is not a regular file")
    if int(metadata.st_nlink) != 1:
        raise WindowFetchError(f"{label} must have exactly one hard link")
    if hasattr(os, "geteuid") and int(metadata.st_uid) != int(os.geteuid()):
        raise WindowFetchError(f"{label} is not owned by the effective user")


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        stat.S_IFMT(metadata.st_mode),
    )


@dataclass(slots=True)
class _OpenTarget:
    target: dict[str, Any]
    parent_fds: list[int]
    parent_links: list[tuple[str, tuple[int, int, int]]]
    leaf: str
    partial_name: str
    partial_path: str
    fd: int
    ready: dict[str, Any] | None = None

    @property
    def parent_fd(self) -> int:
        return self.parent_fds[-1]

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        for descriptor in reversed(self.parent_fds):
            os.close(descriptor)
        self.parent_fds.clear()


def _partial_name(leaf: str, intent_seal: str) -> str:
    name = f".{leaf}.glm52-partial-{intent_seal[:20]}"
    if not name or "/" in name or name in {".", ".."}:
        raise WindowFetchError("derived partial name is unsafe")
    return name


def _prepare_targets(
    root_fd: int,
    targets: Sequence[Mapping[str, Any]],
    *,
    intent_seal: str,
    prepared: list[_OpenTarget] | None = None,
) -> list[_OpenTarget]:
    # The caller supplies this list so a failure after one O_EXCL creation still
    # reports every retained partial.  No cleanup path ever unlinks those names.
    result = prepared if prepared is not None else []
    if result:
        raise WindowFetchError("prepared target output must start empty")
    # First inspect every destination and partial without creating anything.
    inspections: list[tuple[dict[str, Any], list[int], list[tuple[str, tuple[int, int, int]]], str, str]] = []
    try:
        for raw in targets:
            target = dict(raw)
            _normalized, parts = _relative_path(target["path"], "fetch target path")
            parent_fds, parent_links = _open_relative_parent(root_fd, parts[:-1])
            leaf = parts[-1]
            partial = _partial_name(leaf, intent_seal)
            # Transfer descriptor ownership to inspections immediately.  Later
            # inspection failures must not leak the just-opened parent chain.
            inspections.append((target, parent_fds, parent_links, leaf, partial))
            name_max = int(os.fpathconf(parent_fds[-1], "PC_NAME_MAX"))
            if len(partial.encode("utf-8")) > name_max:
                raise WindowFetchError("derived partial name exceeds filesystem NAME_MAX")
            _entry_absent(parent_fds[-1], leaf, label=f"destination {target['path']}")
            _entry_absent(
                parent_fds[-1], partial, label=f"retained partial for {target['path']}"
            )
        for target, parent_fds, parent_links, leaf, partial in inspections:
            descriptor = os.open(
                partial,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC,
                0o600,
                dir_fd=parent_fds[-1],
            )
            _normalized, parts = _relative_path(target["path"], "fetch target path")
            item = _OpenTarget(
                target=target,
                parent_fds=parent_fds,
                parent_links=parent_links,
                leaf=leaf,
                partial_name=partial,
                partial_path="/".join((*parts[:-1], partial)),
                fd=descriptor,
            )
            # Take ownership before any fallible post-open inspection so both the
            # descriptor and retained name remain attributable on every failure.
            result.append(item)
            opened = os.fstat(descriptor)
            named = os.stat(partial, dir_fd=parent_fds[-1], follow_symlinks=False)
            _safe_file_stat(opened, label=f"partial for {target['path']}")
            if _file_identity(named) != _file_identity(opened):
                raise WindowFetchError("partial name changed while creating")
            os.fsync(parent_fds[-1])
        return result
    except BaseException:
        owned = {id(item.parent_fds) for item in result}
        for item in result:
            item.close()
        for _target, parent_fds, _links, _leaf, _partial in inspections:
            if id(parent_fds) not in owned:
                for descriptor in reversed(parent_fds):
                    os.close(descriptor)
        raise


class _InlineBudget:
    def __init__(
        self,
        *,
        maximum_body_bytes: int,
        required_free_disk_bytes: int,
        root_fd: int,
        sampler: ResourceSampler,
        abort_event: threading.Event,
    ) -> None:
        self.maximum_body_bytes = maximum_body_bytes
        self.required_free_disk_bytes = required_free_disk_bytes
        self.root_fd = root_fd
        self.sampler = sampler
        self.abort_event = abort_event
        self.streamed_bytes = 0
        self.inflight_allocated_bound = 0
        self.lock = threading.Lock()
        unit = sampler.allocation_unit_bytes(root_fd)
        if type(unit) is not int or unit <= 0:
            raise WindowFetchError("resource sampler returned an invalid allocation unit")
        self.allocation_unit = unit

    def before_write(self, length: int) -> int:
        if type(length) is not int or not 0 < length <= MAX_STREAM_CHUNK_BYTES:
            raise WindowFetchError("stream chunk size is outside the finite adapter bound")
        bound = ((length + self.allocation_unit - 1) // self.allocation_unit) \
            * self.allocation_unit
        with self.lock:
            if self.abort_event.is_set():
                raise WindowFetchError("parallel fetch was already aborted")
            if self.streamed_bytes + length > self.maximum_body_bytes:
                raise WindowFetchError("inline streamed-body cap would be exceeded")
            free = self.sampler.free_disk_bytes(self.root_fd)
            if type(free) is not int or free < 0:
                raise WindowFetchError("resource sampler returned invalid free disk bytes")
            if free - self.inflight_allocated_bound - bound \
                    < self.required_free_disk_bytes:
                raise WindowFetchError("inline disk-floor reservation refused a body write")
            self.streamed_bytes += length
            self.inflight_allocated_bound += bound
            return bound

    def after_write(self, bound: int) -> None:
        with self.lock:
            self.inflight_allocated_bound -= bound
            if self.inflight_allocated_bound < 0:
                raise WindowFetchError("inline allocation reservation underflow")
            free = self.sampler.free_disk_bytes(self.root_fd)
            if type(free) is not int or free < 0:
                self.abort_event.set()
                raise WindowFetchError("resource sampler returned invalid free disk bytes")
            if free < self.required_free_disk_bytes:
                self.abort_event.set()
                raise WindowFetchError("live disk floor fell below the frozen reserve")


def _write_all(descriptor: int, chunk: bytes) -> None:
    view = memoryview(chunk)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise WindowFetchError("short write to anchored shard partial")
        view = view[written:]


def _stream_target(
    item: _OpenTarget,
    *,
    provider: StreamProvider,
    budget: _InlineBudget,
    abort_event: threading.Event,
) -> dict[str, Any]:
    target = item.target
    digest = hashlib.sha256()
    count = 0
    stream: Iterable[bytes] | None = None
    try:
        # A provider is an I/O mechanism, not an authority source.  Give it a
        # detached view so it cannot rewrite the validated target used by local
        # byte-count and digest enforcement.
        stream = provider.open_stream(_clone(target))
        for chunk in stream:
            if abort_event.is_set():
                raise WindowFetchError("parallel fetch was aborted")
            if not isinstance(chunk, bytes) or not chunk:
                raise WindowFetchError("Xet stream yielded a nonempty-bytes violation")
            if count + len(chunk) > int(target["logical_bytes"]):
                raise WindowFetchError("Xet stream exceeded the exact shard size")
            allocation_bound = budget.before_write(len(chunk))
            try:
                _write_all(item.fd, chunk)
            finally:
                budget.after_write(allocation_bound)
            digest.update(chunk)
            count += len(chunk)
        if count != int(target["logical_bytes"]):
            raise WindowFetchError(
                f"Xet stream length mismatch for {target['path']}: "
                f"{count} != {target['logical_bytes']}"
            )
        observed = digest.hexdigest()
        if not hmac.compare_digest(observed, str(target["lfs_sha256"])):
            raise WindowFetchError(f"full LFS SHA-256 mismatch for {target['path']}")
        os.fsync(item.fd)
        descriptor_stat = os.fstat(item.fd)
        named_stat = os.stat(
            item.partial_name,
            dir_fd=item.parent_fd,
            follow_symlinks=False,
        )
        _safe_file_stat(descriptor_stat, label=f"completed partial for {target['path']}")
        if int(descriptor_stat.st_size) != count \
                or _file_identity(named_stat) != _file_identity(descriptor_stat):
            raise WindowFetchError("completed partial descriptor/name verification failed")
        os.fchmod(item.fd, 0o444)
        os.fsync(item.fd)
        descriptor_stat = os.fstat(item.fd)
        named_stat = os.stat(
            item.partial_name,
            dir_fd=item.parent_fd,
            follow_symlinks=False,
        )
        _safe_file_stat(descriptor_stat, label=f"sealed partial for {target['path']}")
        if _file_identity(named_stat) != _file_identity(descriptor_stat) \
                or int(descriptor_stat.st_size) != count \
                or stat.S_IMODE(descriptor_stat.st_mode) != 0o444 \
                or stat.S_IMODE(named_stat.st_mode) != 0o444:
            raise WindowFetchError("sealed partial identity/size/mode verification failed")
        ready = {
            "path": target["path"],
            "role": target["role"],
            "logical_bytes": count,
            "observed_lfs_sha256": observed,
            "partial_name_sha256": hashlib.sha256(
                item.partial_name.encode("utf-8")
            ).hexdigest(),
            "device": int(descriptor_stat.st_dev),
            "inode": int(descriptor_stat.st_ino),
            "allocated_bytes": int(descriptor_stat.st_blocks) * 512,
        }
        item.ready = ready
        return ready
    except BaseException:
        abort_event.set()
        cancel = getattr(stream, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass
        raise


def _rename_noreplace(parent_fd: int, source: str, destination: str) -> None:
    """Atomic same-directory no-replace publish on Darwin or Linux."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_b = os.fsencode(source)
    destination_b = os.fsencode(destination)
    if sys.platform == "darwin":
        function = getattr(libc, "renameatx_np", None)
        if function is None:
            raise WindowFetchError("Darwin renameatx_np is unavailable")
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(parent_fd, source_b, parent_fd, destination_b, 0x00000004)
    elif sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        if function is None:
            raise WindowFetchError("Linux renameat2 is unavailable")
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(parent_fd, source_b, parent_fd, destination_b, 0x00000001)
    else:
        raise WindowFetchError("atomic no-replace publish requires Darwin or Linux")
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise WindowFetchError("destination appeared before no-replace publish")
        raise WindowFetchError(
            f"atomic no-replace publish failed: {os.strerror(error)}"
        )


def _publish(item: _OpenTarget) -> dict[str, Any]:
    if item.ready is None:
        raise WindowFetchError("attempted to publish an incomplete shard")
    _verify_relative_parent(item.parent_fds, item.parent_links)
    descriptor_stat = os.fstat(item.fd)
    named_partial = os.stat(
        item.partial_name, dir_fd=item.parent_fd, follow_symlinks=False
    )
    _safe_file_stat(descriptor_stat, label=f"publish descriptor for {item.target['path']}")
    _safe_file_stat(named_partial, label=f"named partial for {item.target['path']}")
    if _file_identity(named_partial) != _file_identity(descriptor_stat) \
            or int(descriptor_stat.st_size) != int(item.target["logical_bytes"]) \
            or stat.S_IMODE(descriptor_stat.st_mode) != 0o444 \
            or stat.S_IMODE(named_partial.st_mode) != 0o444:
        raise WindowFetchError("partial changed before atomic publish")
    _entry_absent(item.parent_fd, item.leaf, label=f"destination {item.target['path']}")
    _rename_noreplace(item.parent_fd, item.partial_name, item.leaf)
    published = os.stat(item.leaf, dir_fd=item.parent_fd, follow_symlinks=False)
    _safe_file_stat(published, label=f"published shard {item.target['path']}")
    if _file_identity(published) != _file_identity(descriptor_stat) \
            or int(published.st_size) != int(item.target["logical_bytes"]):
        raise WindowFetchError("published shard inode/size differs from verified partial")
    try:
        os.stat(item.partial_name, dir_fd=item.parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise WindowFetchError("partial name remained after atomic rename")
    os.fsync(item.parent_fd)
    durable = os.stat(item.leaf, dir_fd=item.parent_fd, follow_symlinks=False)
    durable_descriptor = os.fstat(item.fd)
    _safe_file_stat(durable, label=f"durable published shard {item.target['path']}")
    _safe_file_stat(
        durable_descriptor,
        label=f"durable publish descriptor for {item.target['path']}",
    )
    if _file_identity(durable) != _file_identity(durable_descriptor) \
            or _file_identity(durable) != _file_identity(published) \
            or int(durable.st_size) != int(item.target["logical_bytes"]) \
            or stat.S_IMODE(durable.st_mode) != 0o444:
        raise WindowFetchError("durable published path/descriptor verification failed")
    return {
        **item.ready,
        "published_device": int(published.st_dev),
        "published_inode": int(published.st_ino),
        "published_hard_link_count": int(published.st_nlink),
        "published_mode": stat.S_IMODE(published.st_mode),
    }


def materialize_window(
    artifacts: FetchArtifacts,
    intent: Mapping[str, Any],
    *,
    source_root: str | os.PathLike[str],
    auth: state.EvidenceAuthConfig,
    grounding_authenticator: grounding.ProducerAuthenticator,
    capability_verifier: FetchCapabilityVerifier | None,
    stream_provider: StreamProvider | None = None,
    resource_sampler: ResourceSampler | None = None,
    root: Path = xet_live.REPO_ROOT,
    rebuild_plan: bool = True,
) -> dict[str, Any]:
    """Materialize exactly one scheduled new/refetch set; never delete on failure."""
    if capability_verifier is None:
        raise WindowFetchError("independent live fetch capability verifier is required")
    if not isinstance(grounding_authenticator, grounding.ProducerAuthenticator):
        raise WindowFetchError("grounding ProducerAuthenticator is required")
    context = _validate_binding_bundle(
        artifacts,
        auth=auth,
        root=Path(root),
        rebuild_plan=rebuild_plan,
    )
    root_path, root_fds, root_links, root_stat = _open_root(source_root)
    prepared: list[_OpenTarget] = []
    published_paths: list[str] = []
    provider = stream_provider
    sampler = resource_sampler or GroundedResourceSampler()
    abort_event = threading.Event()
    root_fd = root_fds[-1]
    lock_held = False
    try:
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_held = True
        except BlockingIOError as exc:
            raise WindowFetchError("source root is locked by another materializer") from exc
        value, targets, trial = _validate_intent(
            intent,
            context,
            artifacts=artifacts,
            auth=auth,
            root_path=root_path,
            root_stat=root_stat,
        )
        try:
            accepted = capability_verifier.verify_window_fetch_capability(
                _clone(value),
                frozen_schedule=_clone(context.schedule),
                resource_policy=_clone(context.resource_artifact),
            )
        except Exception:
            raise WindowFetchError("independent live fetch capability verification failed") from None
        if accepted is not True:
            raise WindowFetchError("independent live fetch capability verifier refused intent")
        source_id = _root_id(context.contract)
        before = sampler.authenticated_sample(
            root_path,
            root_id=source_id,
            policy=context.resource_policy,
            authenticator=grounding_authenticator,
        )
        before = _verify_resource_receipt(
            before,
            context=context,
            root_id=source_id,
            root_stat=root_stat,
            authenticator=grounding_authenticator,
        )
        total = int(value["maximum_streamed_body_bytes"])
        unit = sampler.allocation_unit_bytes(root_fd)
        if type(unit) is not int or unit <= 0:
            raise WindowFetchError("resource sampler returned an invalid allocation unit")
        conservative_window_bound = sum(
            ((int(target["logical_bytes"]) + unit - 1) // unit) * unit
            for target in targets
        ) + len(targets) * unit
        if int(before["disk_free_bytes"]) - conservative_window_bound \
                < context.resource_policy.required_free_disk_bytes:
            raise WindowFetchError(
                "window allocation bound would cross the frozen disk reserve"
            )
        grounding._verify_absolute_directory_chain(root_fds, root_links, root_stat)
        _prepare_targets(
            root_fd,
            targets,
            intent_seal=value["seal_sha256"],
            prepared=prepared,
        )
        if provider is None:
            provider = HfXetStreamProvider(
                _selected_xet_config(context.plan, trial)
            )
        budget = _InlineBudget(
            maximum_body_bytes=total,
            required_free_disk_bytes=context.resource_policy.required_free_disk_bytes,
            root_fd=root_fd,
            sampler=sampler,
            abort_event=abort_event,
        )
        workers = min(int(value["caller_concurrent_shard_streams"]), len(prepared))
        first_error: BaseException | None = None
        executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="glm52-window-xet",
        )
        try:
            futures = {
                executor.submit(
                    _stream_target,
                    item,
                    provider=provider,
                    budget=budget,
                    abort_event=abort_event,
                ): item
                for item in prepared
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    abort_event.set()
                    for other in futures:
                        other.cancel()
                    try:
                        provider.abort()
                    except Exception:
                        pass
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if first_error is not None:
            if isinstance(first_error, WindowFetchError):
                raise first_error
            raise WindowFetchError(f"Xet shard stream failed: {first_error}") from first_error
        if budget.streamed_bytes != total or any(item.ready is None for item in prepared):
            raise WindowFetchError("window body coverage is incomplete before publish")
        prepublish = sampler.authenticated_sample(
            root_path,
            root_id=source_id,
            policy=context.resource_policy,
            authenticator=grounding_authenticator,
        )
        prepublish = _verify_resource_receipt(
            prepublish,
            context=context,
            root_id=source_id,
            root_stat=root_stat,
            authenticator=grounding_authenticator,
        )
        grounding._verify_absolute_directory_chain(root_fds, root_links, root_stat)
        outcomes = []
        for item in prepared:
            outcome = _publish(item)
            outcomes.append(outcome)
            published_paths.append(str(item.target["path"]))
        for item in prepared:
            _verify_relative_parent(item.parent_fds, item.parent_links)
        grounding._verify_absolute_directory_chain(root_fds, root_links, root_stat)
        evidence = {
            "schema": FETCH_RECEIPT_SCHEMA,
            "status": "PASS_EXPLICIT_ADAPTER_MATERIALIZATION",
            "campaign_id": context.contract["campaign_id"],
            "source_revision": context.contract["source_revision"],
            "expected_contract_sha256": context.contract["seal_sha256"],
            "fetch_intent_seal_sha256": value["seal_sha256"],
            "frozen_schedule_seal_sha256": context.schedule["seal_sha256"],
            "resource_policy_seal_sha256": context.resource_artifact["seal_sha256"],
            "resource_before_seal_sha256": before["seal_sha256"],
            "resource_prepublish_seal_sha256": prepublish["seal_sha256"],
            "schedule_index": value["schedule_index"],
            "window_id": value["window_id"],
            "lane": value["lane"],
            "selected_trial_id": value["selected_trial_id"],
            "caller_concurrent_shard_streams": value[
                "caller_concurrent_shard_streams"
            ],
            "effective_worker_count": workers,
            "source_root": _clone(value["source_root"]),
            "authoritative_target_view_count": 1,
            "maximum_streamed_body_bytes": total,
            "streamed_body_bytes": budget.streamed_bytes,
            "targets": outcomes,
            "retained_partial_paths": [],
            "published_paths": published_paths,
            "completed_at": utc_now(),
            "worker_dispatch_enabled": False,
        }
        try:
            return state.seal_producer_authenticated_evidence(evidence, auth=auth)
        except state.StateError as exc:
            raise WindowFetchError(f"cannot authenticate fetch receipt: {exc}") from exc
    except BaseException as exc:
        abort_event.set()
        if provider is not None:
            try:
                provider.abort()
            except Exception:
                pass
        partials = [
            item.partial_path
            for item in prepared
            if item.target["path"] not in published_paths
        ]
        if isinstance(exc, WindowFetchError):
            raise WindowFetchError(
                str(exc),
                retained_partials=partials,
                published_paths=published_paths,
            ) from exc
        raise WindowFetchError(
            f"window materialization failed: {exc}",
            retained_partials=partials,
            published_paths=published_paths,
        ) from exc
    finally:
        for item in prepared:
            item.close()
        if lock_held:
            try:
                fcntl.flock(root_fd, fcntl.LOCK_UN)
            except OSError:
                pass
        for fd in reversed(root_fds):
            os.close(fd)


__all__ = [
    "FetchArtifacts",
    "FetchCapabilityVerifier",
    "GroundedResourceSampler",
    "HfXetStreamProvider",
    "ResourceSampler",
    "StreamProvider",
    "WindowFetchError",
    "build_window_fetch_intent",
    "materialize_window",
]

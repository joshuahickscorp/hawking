#!/usr/bin/env python3.12
"""Fail-closed persistent worker spine for the GLM-5.2 Gravity campaign.

This process is intentionally a control-plane scaffold.  It owns the existing
``glm52_state.Controller`` lease, replays the sealed checkpoint, emits
producer-authenticated heartbeats, and applies operator controls only at the safe
points declared by :mod:`glm52_gravity`.  It performs no network I/O, Xet body
read, Telegram send, deletion, or scientific phase execution.

Production ``run`` is unavailable until five producer-authenticated readiness
receipts ground credential rotation, the notification outbox, the final post-Xet
schedule, the enforced resource policy, and concrete execution adapters.  The
receipts are not substitutes for the referenced bytes: this module re-reads and
hashes every referenced artifact or adapter through no-follow trusted roots.  Even
then, this scaffold reports ``NO_SCIENTIFIC_DISPATCH`` until a real dispatcher is
implemented and bound to the Controller in a later reviewed change.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import plistlib
import re
import shlex
import signal
import stat
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
for entry in (HERE, TOOLS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import glm52_state as state  # noqa: E402
import glm52_gravity as gravity  # noqa: E402
import glm52_notifications as notifications  # noqa: E402
from glm52_common import (  # noqa: E402
    Glm52Error,
    canonical,
    seal,
    utc_now,
    verify_sealed,
)


WORKER_CONFIG_SCHEMA = "hawking.glm52.persistent_worker_config.v3"
READINESS_SCHEMA = "hawking.glm52.worker_readiness.v1"
PREFLIGHT_SCHEMA = "hawking.glm52.worker_preflight.v1"
STATUS_SCHEMA = "hawking.glm52.worker_status.v1"
HEARTBEAT_SCHEMA = "hawking.glm52.worker_heartbeat.v1"

OFFICIAL_CAMPAIGN_ID = "glm52-bf16-xet-gravity"
OFFICIAL_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
OFFICIAL_CONTRACT_SCHEMA = "hawking.glm52.expected_campaign_contract.v3"
# Deliberately unset until the official deterministic contract v3 is built and
# reviewed.  Production authorization is impossible while this remains ``None``.
OFFICIAL_CONTRACT_SEAL: str | None = None
OFFICIAL_CONFIG_PROFILE = "OFFICIAL_PRODUCTION"
SYNTHETIC_CONFIG_PROFILE = "SYNTHETIC_TEST_ONLY"
OFFICIAL_RESOURCE_POLICY_PATH = "GLM52_RESOURCE_RESERVE_POLICY.json"
OFFICIAL_RESOURCE_POLICY_SCHEMA = "hawking.glm52.resource_reserve_policy.v1"
OFFICIAL_RESOURCE_POLICY_STATUS = "FROZEN_CONSERVATIVE_PRELIVE_POLICY"
OFFICIAL_RESOURCE_POLICY_SEAL = (
    "b33692a9e43aa37c59325ffd1b317d0d069e788c20f4f32cc3840b32ec901cca"
)
OFFICIAL_REQUIRED_FREE_DISK_BYTES = 416_036_394_619
LEASE_BOUND_NOTIFICATION_OUTBOX_AUDIT_REQUIRED = (
    "LEASE_BOUND_NOTIFICATION_OUTBOX_AUDIT_REQUIRED: static readiness can bind only "
    "the outbox path, public verifier identities, Telegram bot identity, and reviewed "
    "module bytes; semantic replay requires the live worker's held controller lease"
)

TELEGRAM_ROTATION = "telegram_rotation"
NOTIFICATION_OUTBOX = "notification_outbox"
FINAL_XET_SCHEDULE = "final_xet_schedule"
RESOURCE_POLICY = "resource_policy"
EXECUTION_ADAPTERS = "execution_adapters"
READINESS_KINDS = (
    TELEGRAM_ROTATION,
    NOTIFICATION_OUTBOX,
    FINAL_XET_SCHEDULE,
    RESOURCE_POLICY,
    EXECUTION_ADAPTERS,
)

REQUIRED_EXECUTION_ROLES = frozenset({
    "FETCH_WINDOW",
    "VERIFY_WINDOW",
    "CAPTURE_TEACHER",
    "FIT_CANDIDATES",
    "PACK_CANDIDATES",
    "RUN_WINDOW_FORWARD",
    "SEAL_WINDOW",
    "EVICT_WINDOW",
})
EXECUTION_ADAPTER_INTERFACE = "hawking.glm52.execution_adapter.v1"
# A reviewed source/entry-point registry must be compiled into this module in the
# same change that introduces the real Controller-bound dispatcher.  An empty
# immutable registry makes every claimed production adapter ungroundable today.
PRODUCTION_EXECUTION_ADAPTER_REGISTRY: Mapping[str, Mapping[str, Any]] = (
    MappingProxyType({})
)
PRODUCTION_EXECUTION_ADAPTER_REGISTRY_SHA256 = (
    "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
)

CAFFEINATE = Path("/usr/bin/caffeinate")
PS = Path("/bin/ps")
CAFFEINATE_FLAGS = frozenset("dimsu")
PINNED_WORKSPACE_ROOT = Path("/Users/scammermike/Downloads/hawking")
PINNED_PYTHON = PINNED_WORKSPACE_ROOT / ".venv/glm52/bin/python"
PINNED_WORKER = PINNED_WORKSPACE_ROOT / "tools/condense/glm52_worker.py"
PINNED_WORKER_CONFIG = PINNED_WORKSPACE_ROOT / "GLM52_WORKER_CONFIG.json"
PINNED_LAUNCHD_PLIST = (
    PINNED_WORKSPACE_ROOT / "deploy/launchd/com.hawking.glm52.gravity.plist"
)
PINNED_STDOUT = Path(
    "/Users/scammermike/Library/Logs/com.hawking.glm52.gravity.stdout.log"
)
PINNED_STDERR = Path(
    "/Users/scammermike/Library/Logs/com.hawking.glm52.gravity.stderr.log"
)
PINNED_LAUNCHD_ARGUMENTS = (
    str(CAFFEINATE),
    "-dimsu",
    str(PINNED_PYTHON),
    str(PINNED_WORKER),
    "--config",
    str(PINNED_WORKER_CONFIG),
    "--under-caffeinate",
    "run",
)
PINNED_LAUNCHD_KEYS = frozenset({
    "Label", "ProgramArguments", "WorkingDirectory", "RunAtLoad", "KeepAlive",
    "ProcessType", "ThrottleInterval", "StandardOutPath", "StandardErrorPath", "Umask",
})
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0
MAX_HEARTBEAT_INTERVAL_SECONDS = 60.0

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_PREFLIGHT_BLOCKED = 78

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_SECONDS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$")

SESSION_LIMITATIONS = (
    "This plist is a per-user LaunchAgent, not a privileged LaunchDaemon. It is "
    "terminated at logout, starts after the next user login (including after a "
    "reboot), and requires the user's login Keychain to be available. RunAtLoad "
    "and KeepAlive recover crashes only while that launchd user session exists."
)
NO_SCIENTIFIC_DISPATCH_REASON = (
    "NO_SCIENTIFIC_DISPATCH: this scaffold validates adapter manifests but has no "
    "controller-bound dispatcher callable; production launch cannot be authorized"
)


class WorkerError(Glm52Error):
    """A worker configuration, readiness, lease, or heartbeat invariant failed."""


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(canonical(value))


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkerError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkerError(f"{label} root is not an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise WorkerError(
            f"{label} fields differ: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )


def _require_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or value != value.strip() \
            or _SAFE_NAME_RE.fullmatch(value) is None:
        raise WorkerError(f"{label} must be a safe non-empty name")
    return value


def _require_utc_seconds(value: Any, label: str) -> str:
    if not isinstance(value, str) or _UTC_SECONDS_RE.fullmatch(value) is None:
        raise WorkerError(f"{label} must be deterministic UTC YYYY-MM-DDTHH:MM:SSZ")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise WorkerError(f"{label} is not a real UTC timestamp") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise WorkerError(f"{label} is not canonical UTC")
    return value


def _normalized_absolute(value: Any, label: str, *, must_exist: bool = False) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise WorkerError(f"{label} must be an absolute path string")
    candidate = Path(value)
    if not candidate.is_absolute() or value.startswith("//") \
            or os.path.normpath(value) != value:
        raise WorkerError(f"{label} must be a normalized absolute path")
    if must_exist and not candidate.exists():
        raise WorkerError(f"{label} does not exist: {candidate}")
    return candidate


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise WorkerError(f"{label} must be a relative path")
    path = Path(value)
    if path.is_absolute() or value in {".", ".."} \
            or any(part in {"", ".", ".."} for part in path.parts):
        raise WorkerError(f"{label} must remain beneath its trusted root")
    return value


@dataclass(frozen=True)
class ReadinessSpec:
    path: str
    expected_file_sha256: str


@dataclass(frozen=True)
class ControllerConfigBinding:
    """Exact bytes and path authority loaded from one sealed controller config."""

    file_sha256: str
    seal_sha256: str
    campaign_id: str
    source_revision: str
    controller_root: Path
    artifact_root: Path
    expected_contract_path: Path


@dataclass(frozen=True)
class NotificationAuditReadiness:
    """Immutable, public-only inputs for the mandatory held-lease outbox audit."""

    campaign_id: str
    source_revision: str
    controller_epoch: str
    controller_config_seal_sha256: str
    controller_root: Path
    artifact_root: Path
    expected_contract_sha256: str
    workspace_root: Path
    notification_subject_canonical: bytes
    telegram_rotation_receipt_seal_sha256: str
    telegram_rotation_canonical: bytes


@dataclass(frozen=True)
class WorkerConfig:
    path: Path
    profile: str
    campaign_id: str
    source_revision: str
    controller_config_path: Path
    controller_config_file_sha256: str
    controller_config_seal_sha256: str
    controller_root: Path
    artifact_root: Path
    expected_contract_path: Path
    workspace_root: Path
    expected_contract_sha256: str
    expected_contract_created_at: str
    heartbeat_interval_seconds: float
    readiness: Mapping[str, ReadinessSpec]
    producer_hmac_sha256: str
    seal_sha256: str
    document: Mapping[str, Any]


def _resolved_controller_config_path(
    config_path: Path,
    raw: Any,
    label: str,
) -> Path:
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise WorkerError(f"controller config {label} must be a non-empty path string")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    try:
        return candidate.resolve(strict=False)
    except OSError as exc:
        raise WorkerError(f"cannot resolve controller config {label}: {exc}") from exc


def _read_controller_config_binding(path: Path) -> ControllerConfigBinding:
    """Securely read once and derive the exact authority-bound controller targets."""
    normalized = _normalized_absolute(os.fspath(path), "controller_config_path")
    try:
        raw = state.TrustedArtifactStore(str(normalized.parent)).read_bytes(
            normalized.name
        )
    except state.StateError as exc:
        raise WorkerError(f"cannot securely read controller config: {exc}") from exc
    value = _strict_json(raw, label="controller CLI config")
    try:
        verified = verify_sealed(value, label="controller CLI config")
    except Glm52Error as exc:
        raise WorkerError(str(exc)) from exc
    required = {
        "schema",
        "campaign_id",
        "source_revision",
        "controller_epoch",
        "controller_root",
        "artifact_root",
        "expected_contract_path",
        "phone_status_directory",
        "allow_synthetic_contract",
        "telegram",
        "evidence_auth",
        "grounding_auth",
        "seal_sha256",
    }
    _require_exact_keys(verified, required, "controller CLI config")
    if verified.get("schema") != gravity.CONFIG_SCHEMA:
        raise WorkerError("controller CLI config schema mismatch")
    campaign_id = _require_name(
        verified.get("campaign_id"), "controller CLI config campaign_id"
    )
    revision = verified.get("source_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise WorkerError("controller CLI config source_revision is not immutable 40-hex")
    return ControllerConfigBinding(
        file_sha256=_sha256_bytes(raw),
        seal_sha256=str(verified["seal_sha256"]),
        campaign_id=campaign_id,
        source_revision=revision,
        controller_root=_resolved_controller_config_path(
            normalized, verified.get("controller_root"), "controller_root"
        ),
        artifact_root=_resolved_controller_config_path(
            normalized, verified.get("artifact_root"), "artifact_root"
        ),
        expected_contract_path=_resolved_controller_config_path(
            normalized,
            verified.get("expected_contract_path"),
            "expected_contract_path",
        ),
    )


def make_worker_config(
    *,
    profile: str,
    campaign_id: str,
    source_revision: str,
    controller_config_path: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str],
    expected_contract_sha256: str,
    expected_contract_created_at: str,
    readiness: Mapping[str, Mapping[str, str]],
    evidence_auth: state.EvidenceAuthConfig,
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Build an evidence-authority-authenticated config without writing it."""
    normalized_controller_config_path = _normalized_absolute(
        os.fspath(controller_config_path), "controller_config_path"
    )
    controller_binding = _read_controller_config_binding(
        normalized_controller_config_path
    )
    if controller_binding.campaign_id != campaign_id \
            or controller_binding.source_revision != source_revision:
        raise WorkerError(
            "controller config identity differs from requested worker identity"
        )
    body = {
        "schema": WORKER_CONFIG_SCHEMA,
        "profile": profile,
        "campaign_id": campaign_id,
        "source_revision": source_revision,
        "controller_config_path": os.fspath(normalized_controller_config_path),
        "controller_config_file_sha256": controller_binding.file_sha256,
        "controller_config_seal_sha256": controller_binding.seal_sha256,
        "controller_root": os.fspath(controller_binding.controller_root),
        "artifact_root": os.fspath(controller_binding.artifact_root),
        "expected_contract_path": os.fspath(
            controller_binding.expected_contract_path
        ),
        "workspace_root": os.fspath(workspace_root),
        "expected_contract_sha256": expected_contract_sha256,
        "expected_contract_created_at": expected_contract_created_at,
        "heartbeat_interval_seconds": heartbeat_interval_seconds,
        "readiness": {key: dict(value) for key, value in readiness.items()},
    }
    if not isinstance(evidence_auth, state.EvidenceAuthConfig) \
            or evidence_auth.campaign_id != campaign_id \
            or evidence_auth.source_revision != source_revision:
        raise WorkerError("worker config evidence authority identity mismatch")
    value = state.seal_producer_authenticated_evidence(body, auth=evidence_auth)
    _parse_worker_config_document(value, Path("/nonexistent/GLM52_WORKER_CONFIG.json"))
    return value


def _parse_worker_config_document(value: Mapping[str, Any], path: Path) -> WorkerConfig:
    try:
        verified = verify_sealed(dict(value), label="GLM52 worker config")
    except Glm52Error as exc:
        raise WorkerError(str(exc)) from exc
    _require_exact_keys(verified, {
        "schema",
        "profile",
        "campaign_id",
        "source_revision",
        "controller_config_path",
        "controller_config_file_sha256",
        "controller_config_seal_sha256",
        "controller_root",
        "artifact_root",
        "expected_contract_path",
        "workspace_root",
        "expected_contract_sha256",
        "expected_contract_created_at",
        "heartbeat_interval_seconds",
        "readiness",
        "producer_hmac_sha256",
        "seal_sha256",
    }, "worker config")
    if verified.get("schema") != WORKER_CONFIG_SCHEMA:
        raise WorkerError("worker config schema mismatch")
    profile = verified.get("profile")
    if profile not in {OFFICIAL_CONFIG_PROFILE, SYNTHETIC_CONFIG_PROFILE}:
        raise WorkerError("worker config profile is invalid")
    campaign_id = _require_name(verified.get("campaign_id"), "worker campaign_id")
    source_revision = verified.get("source_revision")
    if not isinstance(source_revision, str) or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise WorkerError("worker source_revision must be immutable 40-hex")
    producer_hmac = verified.get("producer_hmac_sha256")
    if not _is_sha256(producer_hmac):
        raise WorkerError("worker config producer HMAC is invalid")
    controller_config_path = _normalized_absolute(
        verified.get("controller_config_path"), "controller_config_path"
    )
    controller_config_file_sha256 = verified.get("controller_config_file_sha256")
    if not _is_sha256(controller_config_file_sha256):
        raise WorkerError("controller_config_file_sha256 must be a sha256")
    controller_config_seal_sha256 = verified.get("controller_config_seal_sha256")
    if not _is_sha256(controller_config_seal_sha256):
        raise WorkerError("controller_config_seal_sha256 must be a sha256")
    controller_root = _normalized_absolute(
        verified.get("controller_root"), "controller_root"
    )
    artifact_root = _normalized_absolute(
        verified.get("artifact_root"), "artifact_root"
    )
    expected_contract_path = _normalized_absolute(
        verified.get("expected_contract_path"), "expected_contract_path"
    )
    workspace_root = _normalized_absolute(
        verified.get("workspace_root"), "workspace_root"
    )
    contract_sha = verified.get("expected_contract_sha256")
    if not _is_sha256(contract_sha):
        raise WorkerError("expected_contract_sha256 must be a sha256")
    created_at = _require_utc_seconds(
        verified.get("expected_contract_created_at"),
        "expected_contract_created_at",
    )
    interval = verified.get("heartbeat_interval_seconds")
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) \
            or not 0.1 <= float(interval) <= MAX_HEARTBEAT_INTERVAL_SECONDS:
        raise WorkerError(
            f"heartbeat_interval_seconds must be in [0.1, {MAX_HEARTBEAT_INTERVAL_SECONDS}]"
        )
    raw_readiness = verified.get("readiness")
    if not isinstance(raw_readiness, dict) or set(raw_readiness) != set(READINESS_KINDS):
        raise WorkerError("worker config readiness must name every exact readiness kind")
    parsed: dict[str, ReadinessSpec] = {}
    for kind in READINESS_KINDS:
        spec = raw_readiness[kind]
        if not isinstance(spec, dict):
            raise WorkerError(f"readiness.{kind} must be an object")
        _require_exact_keys(spec, {"path", "expected_file_sha256"}, f"readiness.{kind}")
        relative = _relative_path(spec.get("path"), f"readiness.{kind}.path")
        expected_hash = spec.get("expected_file_sha256")
        if not _is_sha256(expected_hash):
            raise WorkerError(f"readiness.{kind}.expected_file_sha256 must be a sha256")
        parsed[kind] = ReadinessSpec(relative, str(expected_hash))
    return WorkerConfig(
        path=path,
        profile=str(profile),
        campaign_id=campaign_id,
        source_revision=source_revision,
        controller_config_path=controller_config_path,
        controller_config_file_sha256=str(controller_config_file_sha256),
        controller_config_seal_sha256=str(controller_config_seal_sha256),
        controller_root=controller_root,
        artifact_root=artifact_root,
        expected_contract_path=expected_contract_path,
        workspace_root=workspace_root,
        expected_contract_sha256=str(contract_sha),
        expected_contract_created_at=created_at,
        heartbeat_interval_seconds=float(interval),
        readiness=parsed,
        producer_hmac_sha256=str(producer_hmac),
        seal_sha256=str(verified["seal_sha256"]),
        document=json.loads(canonical(verified).decode("utf-8")),
    )


def load_worker_config(path: str | os.PathLike[str]) -> WorkerConfig:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(os.path.abspath(Path.cwd() / candidate))
    else:
        candidate = Path(os.path.abspath(candidate))
    try:
        store = state.TrustedArtifactStore(str(candidate.parent))
        raw = store.read_bytes(candidate.name)
    except state.StateError as exc:
        raise WorkerError(f"cannot securely read worker config: {exc}") from exc
    return _parse_worker_config_document(
        _strict_json(raw, label="GLM52 worker config"), candidate
    )


def make_readiness_receipt(
    kind: str,
    subject: Mapping[str, Any],
    *,
    runtime: gravity.Runtime,
    created_at: str,
) -> dict[str, Any]:
    """Build one authenticated readiness receipt; callers remain responsible for writing it."""
    if kind not in READINESS_KINDS:
        raise WorkerError(f"unknown readiness kind: {kind}")
    _require_utc_seconds(created_at, "readiness created_at")
    body = {
        "schema": READINESS_SCHEMA,
        "campaign_id": runtime.campaign_id,
        "source_revision": runtime.source_revision,
        "controller_epoch": runtime.controller_epoch,
        "expected_contract_sha256": runtime.expected_contract["seal_sha256"],
        "kind": kind,
        "status": "PASS",
        "created_at": created_at,
        "subject": json.loads(canonical(dict(subject)).decode("utf-8")),
    }
    try:
        return state.seal_producer_authenticated_evidence(
            body, auth=runtime.evidence_auth
        )
    except state.StateError as exc:
        raise WorkerError(str(exc)) from exc


def _validate_producer_authenticated_artifact(
    value: Mapping[str, Any],
    *,
    auth: state.EvidenceAuthConfig,
    label: str,
) -> dict[str, Any]:
    try:
        result = verify_sealed(dict(value), label=label)
    except Glm52Error as exc:
        raise WorkerError(str(exc)) from exc
    signature = result.get("producer_hmac_sha256")
    if not _is_sha256(signature):
        raise WorkerError(f"{label} lacks producer authentication")
    artifact = {
        key: item
        for key, item in result.items()
        if key not in {"seal_sha256", "producer_hmac_sha256"}
    }
    if not auth.verify({
        "schema": "hawking.glm52.evidence_producer_auth.v1",
        "artifact": artifact,
    }, signature):
        raise WorkerError(f"{label} producer HMAC failed")
    return result


def _validate_worker_config_authority(
    config: WorkerConfig,
    runtime: gravity.Runtime,
) -> dict[str, Any]:
    result = _validate_producer_authenticated_artifact(
        config.document,
        auth=runtime.evidence_auth,
        label="GLM52 worker config",
    )
    if result.get("schema") != WORKER_CONFIG_SCHEMA \
            or result.get("profile") != config.profile \
            or result.get("campaign_id") != config.campaign_id \
            or result.get("source_revision") != config.source_revision \
            or result.get("controller_config_path") != \
            os.fspath(config.controller_config_path) \
            or result.get("controller_config_file_sha256") != \
            config.controller_config_file_sha256 \
            or result.get("controller_config_seal_sha256") != \
            config.controller_config_seal_sha256 \
            or result.get("controller_root") != os.fspath(config.controller_root) \
            or result.get("artifact_root") != os.fspath(config.artifact_root) \
            or result.get("expected_contract_path") != \
            os.fspath(config.expected_contract_path) \
            or result.get("workspace_root") != os.fspath(config.workspace_root) \
            or result.get("expected_contract_sha256") != config.expected_contract_sha256 \
            or result.get("seal_sha256") != config.seal_sha256:
        raise WorkerError("worker config evidence authority binding mismatch")
    observed = _read_controller_config_binding(config.controller_config_path)
    expected_controller_binding = {
        "file_sha256": config.controller_config_file_sha256,
        "seal_sha256": config.controller_config_seal_sha256,
        "campaign_id": config.campaign_id,
        "source_revision": config.source_revision,
        "controller_root": config.controller_root,
        "artifact_root": config.artifact_root,
        "expected_contract_path": config.expected_contract_path,
    }
    observed_controller_binding = {
        "file_sha256": observed.file_sha256,
        "seal_sha256": observed.seal_sha256,
        "campaign_id": observed.campaign_id,
        "source_revision": observed.source_revision,
        "controller_root": observed.controller_root,
        "artifact_root": observed.artifact_root,
        "expected_contract_path": observed.expected_contract_path,
    }
    if observed_controller_binding != expected_controller_binding:
        mismatches = sorted(
            key for key, expected in expected_controller_binding.items()
            if observed_controller_binding.get(key) != expected
        )
        raise WorkerError(
            "controller config bytes/targets differ from authenticated worker "
            f"config: {mismatches}"
        )
    runtime_binding = {
        "config_path": Path(os.path.abspath(runtime.config_path)),
        "config_seal_sha256": runtime.config_sha256,
        "controller_root": Path(os.path.abspath(runtime.controller_root)),
        "artifact_root": Path(os.path.abspath(runtime.artifact_root)),
        "expected_contract_path": Path(
            os.path.abspath(runtime.expected_contract_path)
        ),
    }
    expected_runtime_binding = {
        "config_path": config.controller_config_path,
        "config_seal_sha256": config.controller_config_seal_sha256,
        "controller_root": config.controller_root,
        "artifact_root": config.artifact_root,
        "expected_contract_path": config.expected_contract_path,
    }
    if runtime_binding != expected_runtime_binding:
        mismatches = sorted(
            key for key, expected in expected_runtime_binding.items()
            if runtime_binding.get(key) != expected
        )
        raise WorkerError(
            "loaded runtime differs from authenticated controller config binding: "
            f"{mismatches}"
        )
    return {
        "profile": config.profile,
        "campaign_id": config.campaign_id,
        "source_revision": config.source_revision,
        "controller_config_file_sha256": config.controller_config_file_sha256,
        "controller_config_seal_sha256": config.controller_config_seal_sha256,
        "controller_root": os.fspath(config.controller_root),
        "artifact_root": os.fspath(config.artifact_root),
        "expected_contract_path": os.fspath(config.expected_contract_path),
        "producer_hmac_sha256": config.producer_hmac_sha256,
        "seal_sha256": config.seal_sha256,
    }


def _validate_receipt_envelope(
    raw: bytes,
    *,
    expected_kind: str,
    expected_file_sha256: str,
    runtime: gravity.Runtime,
) -> dict[str, Any]:
    if _sha256_bytes(raw) != expected_file_sha256:
        raise WorkerError(f"{expected_kind} readiness bytes differ from worker config")
    value = _strict_json(raw, label=f"{expected_kind} readiness receipt")
    result = _validate_producer_authenticated_artifact(
        value,
        auth=runtime.evidence_auth,
        label=f"{expected_kind} readiness receipt",
    )
    _require_exact_keys(result, {
        "schema",
        "campaign_id",
        "source_revision",
        "controller_epoch",
        "expected_contract_sha256",
        "kind",
        "status",
        "created_at",
        "subject",
        "producer_hmac_sha256",
        "seal_sha256",
    }, f"{expected_kind} readiness receipt")
    if result.get("schema") != READINESS_SCHEMA \
            or result.get("kind") != expected_kind \
            or result.get("status") != "PASS":
        raise WorkerError(f"{expected_kind} readiness status/schema mismatch")
    for key, expected in (
        ("campaign_id", runtime.campaign_id),
        ("source_revision", runtime.source_revision),
        ("controller_epoch", runtime.controller_epoch),
        ("expected_contract_sha256", runtime.expected_contract["seal_sha256"]),
    ):
        if result.get(key) != expected:
            raise WorkerError(f"{expected_kind} readiness {key} mismatch")
    _require_utc_seconds(result.get("created_at"), f"{expected_kind}.created_at")
    if not isinstance(result.get("subject"), dict):
        raise WorkerError(f"{expected_kind} readiness subject must be an object")
    return result


def _validate_telegram_rotation(
    subject: Mapping[str, Any],
    runtime: gravity.Runtime,
    credential_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_exact_keys(subject, {
        "expected_chat_identity_digest",
        "token_identity_digest",
        "notification_bot_identity_digest",
        "telegram_key_identity_sha256",
        "evidence_key_identity_sha256",
        "rotation_completed_at",
        "historical_credential_revoked",
    }, "telegram rotation subject")
    telegram_key_id = runtime.telegram_auth._key_material_identity()
    evidence_key_id = runtime.evidence_auth._key_material_identity()
    if not isinstance(credential_status, Mapping) \
            or credential_status.get("ready") is not True \
            or credential_status.get("token_configured") is not True \
            or credential_status.get("private_chat_configured") is not True \
            or credential_status.get("hmac_key_configured") is not True:
        raise WorkerError("complete Telegram sender credentials are not configured")
    token_identity = credential_status.get("token_identity_digest")
    if not _is_sha256(token_identity) \
            or subject.get("token_identity_digest") != token_identity:
        raise WorkerError("rotation receipt does not bind the loaded Bot token identity")
    bot_identity = credential_status.get("notification_bot_identity_digest")
    if not _is_sha256(bot_identity) \
            or subject.get("notification_bot_identity_digest") != bot_identity:
        raise WorkerError(
            "rotation receipt does not bind the loaded notification Bot identity"
        )
    if credential_status.get("chat_identity_digest") != \
            runtime.telegram_auth.expected_chat_identity_digest:
        raise WorkerError("Telegram credential status chat identity mismatch")
    if subject.get("expected_chat_identity_digest") != \
            runtime.telegram_auth.expected_chat_identity_digest:
        raise WorkerError("rotation receipt chat identity differs from loaded Keychain identity")
    if subject.get("telegram_key_identity_sha256") != telegram_key_id \
            or subject.get("evidence_key_identity_sha256") != evidence_key_id:
        raise WorkerError("rotation receipt key identities differ from loaded Keychain keys")
    if telegram_key_id == evidence_key_id:
        raise WorkerError("Telegram and evidence keys are not independent")
    if subject.get("historical_credential_revoked") is not True:
        raise WorkerError("rotation receipt does not attest historical credential revocation")
    _require_utc_seconds(subject.get("rotation_completed_at"), "rotation_completed_at")
    return {
        "chat_identity_digest": runtime.telegram_auth.expected_chat_identity_digest,
        "token_identity_digest": token_identity,
        "notification_bot_identity_digest": bot_identity,
        "authentication_roles_pairwise_distinct": True,
        "historical_credential_revoked": True,
    }


def _notification_verifiers(
    subject: Mapping[str, Any],
) -> tuple[
    notifications.NotificationVerifier,
    notifications.NotificationVerifier,
    notifications.NotificationVerifier,
]:
    values: list[notifications.NotificationVerifier] = []
    for role in ("producer", "receipt", "reconciliation"):
        field = f"{role}_public_key"
        encoded = subject.get(field)
        if not isinstance(encoded, str) or re.fullmatch(r"[0-9a-f]{64}", encoded) is None:
            raise WorkerError(f"notification {role} public key is invalid")
        try:
            values.append(notifications.NotificationVerifier(role, bytes.fromhex(encoded)))
        except (ValueError, notifications.NotificationError) as exc:
            raise WorkerError(f"notification {role} public key is invalid: {exc}") from exc
    if len({item.key_id for item in values}) != 3:
        raise WorkerError("notification signing roles do not use three distinct public keys")
    return values[0], values[1], values[2]


def _validate_notification_outbox_static(
    subject: Mapping[str, Any],
    *,
    workspace_store: state.TrustedArtifactStore,
    runtime: gravity.Runtime,
    telegram_rotation_receipt_seal_sha256: str,
    telegram_rotation: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact_keys(subject, {
        "journal_path",
        "notification_module_path",
        "notification_module_sha256",
        "producer_public_key",
        "receipt_public_key",
        "reconciliation_public_key",
        "expected_chat_identity_digest",
        "expected_bot_identity_digest",
        "telegram_rotation_readiness_seal_sha256",
        "network_sender_embedded_in_worker",
    }, "notification outbox subject")
    if subject.get("network_sender_embedded_in_worker") is not False:
        raise WorkerError("notification readiness embeds or claims a worker network sender")
    if subject.get("expected_chat_identity_digest") != \
            runtime.telegram_auth.expected_chat_identity_digest \
            or subject.get("expected_chat_identity_digest") != \
            telegram_rotation.get("chat_identity_digest") \
            or subject.get("expected_bot_identity_digest") != \
            telegram_rotation.get("notification_bot_identity_digest") \
            or subject.get("telegram_rotation_readiness_seal_sha256") != \
            telegram_rotation_receipt_seal_sha256:
        raise WorkerError(
            "notification outbox does not bind the current rotated Telegram identity receipt"
        )
    journal_path = _relative_path(subject.get("journal_path"), "notification journal_path")
    module_path = _relative_path(
        subject.get("notification_module_path"), "notification module_path"
    )
    if module_path != "tools/condense/glm52_notifications.py":
        raise WorkerError(
            "notification readiness does not bind exact tools/condense/glm52_notifications.py"
        )
    module = workspace_store.read_bytes(module_path)
    module_sha = subject.get("notification_module_sha256")
    if not _is_sha256(module_sha) or _sha256_bytes(module) != module_sha:
        raise WorkerError("notification module bytes differ from readiness receipt")
    producer, receipt, reconciliation = _notification_verifiers(subject)
    return {
        "journal_path": journal_path,
        "derived_head_path": f"{journal_path}.head.json",
        "module_sha256": module_sha,
        "producer_key_id": producer.key_id,
        "receipt_key_id": receipt.key_id,
        "reconciliation_key_id": reconciliation.key_id,
        "expected_chat_identity_digest": runtime.telegram_auth.expected_chat_identity_digest,
        "expected_bot_identity_digest": subject["expected_bot_identity_digest"],
        "telegram_rotation_readiness_seal_sha256":
            telegram_rotation_receipt_seal_sha256,
        "static_binding_verified": True,
        "lease_bound_semantic_audit_verified": False,
    }


def _audit_notification_outbox_under_lease(
    subject: Mapping[str, Any],
    *,
    workspace_store: state.TrustedArtifactStore,
    runtime: gravity.Runtime,
    controller: state.Controller,
    telegram_rotation_receipt_seal_sha256: str,
    telegram_rotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the public-key-only semantic audit while the exact controller lease is held."""
    static = _validate_notification_outbox_static(
        subject,
        workspace_store=workspace_store,
        runtime=runtime,
        telegram_rotation_receipt_seal_sha256=
            telegram_rotation_receipt_seal_sha256,
        telegram_rotation=telegram_rotation,
    )
    if Path(os.path.abspath(controller.root)) != Path(os.path.abspath(runtime.controller_root)) \
            or controller.campaign_id != runtime.campaign_id \
            or controller.source_revision != runtime.source_revision \
            or controller.controller_epoch != runtime.controller_epoch \
            or controller.expected_contract_sha256 != \
            runtime.expected_contract["seal_sha256"]:
        raise WorkerError("notification audit controller differs from the loaded runtime")
    producer, receipt, reconciliation = _notification_verifiers(subject)
    try:
        audit = notifications.audit_notification_outbox(
            runtime.controller_root / static["journal_path"],
            controller=controller,
            producer_verifier=producer,
            receipt_verifier=receipt,
            reconciliation_verifier=reconciliation,
            expected_bot_identity_digest=static["expected_bot_identity_digest"],
        )
    except notifications.NotificationError as exc:
        raise WorkerError(f"notification outbox semantic audit failed: {exc}") from exc
    if audit.get("status") != "PASS" \
            or audit.get("replay_status") != "VERIFIED" \
            or audit.get("ambiguous_intent_count") != 0:
        raise WorkerError(
            "notification outbox audit is not unambiguously safe for dispatch"
        )
    return {
        **static,
        "lease_bound_semantic_audit_verified": True,
        "audit_sha256": audit["audit_sha256"],
        "entry_count": audit["entry_count"],
        "intent_count": audit["intent_count"],
        "committed_intent_count": audit["committed_intent_count"],
        "unresolved_intent_count": audit["unresolved_intent_count"],
        "ambiguous_intent_count": 0,
        "controller_binding_sha256": audit["controller_binding_sha256"],
    }


def _freeze_notification_audit_readiness(
    *,
    workspace_root: Path,
    subject: Mapping[str, Any],
    runtime: gravity.Runtime,
    telegram_rotation_receipt_seal_sha256: str,
    telegram_rotation: Mapping[str, Any],
) -> NotificationAuditReadiness:
    """Validate then freeze every public input needed by the lease-bound audit."""
    normalized_workspace = _normalized_absolute(
        os.fspath(workspace_root), "notification audit workspace_root"
    )
    _validate_notification_outbox_static(
        subject,
        workspace_store=state.TrustedArtifactStore(str(normalized_workspace)),
        runtime=runtime,
        telegram_rotation_receipt_seal_sha256=
            telegram_rotation_receipt_seal_sha256,
        telegram_rotation=telegram_rotation,
    )
    if not _is_sha256(telegram_rotation_receipt_seal_sha256):
        raise WorkerError("notification audit Telegram rotation receipt seal is invalid")
    return NotificationAuditReadiness(
        campaign_id=runtime.campaign_id,
        source_revision=runtime.source_revision,
        controller_epoch=runtime.controller_epoch,
        controller_config_seal_sha256=runtime.config_sha256,
        controller_root=Path(os.path.abspath(runtime.controller_root)),
        artifact_root=Path(os.path.abspath(runtime.artifact_root)),
        expected_contract_sha256=runtime.expected_contract["seal_sha256"],
        workspace_root=normalized_workspace,
        notification_subject_canonical=canonical(dict(subject)),
        telegram_rotation_receipt_seal_sha256=
            telegram_rotation_receipt_seal_sha256,
        telegram_rotation_canonical=canonical(dict(telegram_rotation)),
    )


def _load_notification_audit_readiness(
    config: WorkerConfig,
    runtime: gravity.Runtime,
    *,
    telegram_credential_status: Mapping[str, Any],
) -> NotificationAuditReadiness:
    """Reopen the exact authenticated receipts and freeze the held-lease audit input."""
    artifact_store = state.TrustedArtifactStore(str(runtime.artifact_root))
    rotation_spec = config.readiness[TELEGRAM_ROTATION]
    rotation_receipt = _validate_receipt_envelope(
        artifact_store.read_bytes(rotation_spec.path),
        expected_kind=TELEGRAM_ROTATION,
        expected_file_sha256=rotation_spec.expected_file_sha256,
        runtime=runtime,
    )
    rotation = _validate_telegram_rotation(
        rotation_receipt["subject"], runtime, telegram_credential_status
    )
    notification_spec = config.readiness[NOTIFICATION_OUTBOX]
    notification_receipt = _validate_receipt_envelope(
        artifact_store.read_bytes(notification_spec.path),
        expected_kind=NOTIFICATION_OUTBOX,
        expected_file_sha256=notification_spec.expected_file_sha256,
        runtime=runtime,
    )
    return _freeze_notification_audit_readiness(
        workspace_root=config.workspace_root,
        subject=notification_receipt["subject"],
        runtime=runtime,
        telegram_rotation_receipt_seal_sha256=rotation_receipt["seal_sha256"],
        telegram_rotation=rotation,
    )


def _audit_frozen_notification_readiness_under_lease(
    readiness: NotificationAuditReadiness,
    *,
    runtime: gravity.Runtime,
    controller: state.Controller,
) -> dict[str, Any]:
    """Revalidate immutable readiness and run semantic replay under the held lease."""
    if not isinstance(readiness, NotificationAuditReadiness):
        raise WorkerError("notification audit readiness has the wrong type")
    observed_identity = (
        runtime.campaign_id,
        runtime.source_revision,
        runtime.controller_epoch,
        runtime.config_sha256,
        Path(os.path.abspath(runtime.controller_root)),
        Path(os.path.abspath(runtime.artifact_root)),
        runtime.expected_contract["seal_sha256"],
    )
    expected_identity = (
        readiness.campaign_id,
        readiness.source_revision,
        readiness.controller_epoch,
        readiness.controller_config_seal_sha256,
        readiness.controller_root,
        readiness.artifact_root,
        readiness.expected_contract_sha256,
    )
    if observed_identity != expected_identity:
        raise WorkerError(
            "notification audit readiness differs from the loaded runtime identity"
        )
    subject = _strict_json(
        readiness.notification_subject_canonical,
        label="frozen notification readiness subject",
    )
    rotation = _strict_json(
        readiness.telegram_rotation_canonical,
        label="frozen Telegram rotation detail",
    )
    if canonical(subject) != readiness.notification_subject_canonical \
            or canonical(rotation) != readiness.telegram_rotation_canonical:
        raise WorkerError("notification audit readiness is not canonical")
    return _audit_notification_outbox_under_lease(
        subject,
        workspace_store=state.TrustedArtifactStore(str(readiness.workspace_root)),
        runtime=runtime,
        controller=controller,
        telegram_rotation_receipt_seal_sha256=
            readiness.telegram_rotation_receipt_seal_sha256,
        telegram_rotation=rotation,
    )


def _validate_runtime_authentication_roles(runtime: gravity.Runtime) -> dict[str, Any]:
    """Validate role types and equality in memory without serializing fingerprints."""
    if not isinstance(runtime.telegram_auth, state.TelegramAuthConfig) \
            or not isinstance(runtime.evidence_auth, state.EvidenceAuthConfig) \
            or not isinstance(
                runtime.grounding_auth,
                gravity.grounding_module.ProducerAuthenticator,
            ):
        raise WorkerError(
            "Telegram, evidence, and grounding authenticators have invalid role types"
        )
    identities = {
        runtime.telegram_auth._key_material_identity(),
        runtime.evidence_auth._key_material_identity(),
        runtime.grounding_auth._key_material_identity(),
    }
    if len(identities) != 3:
        raise WorkerError(
            "Telegram, evidence, and grounding keys must be pairwise distinct"
        )
    return {
        "telegram_auth_loaded": True,
        "evidence_auth_loaded": True,
        "grounding_auth_loaded": True,
        "pairwise_distinct": True,
        "identity_fingerprints_serialized": False,
    }


def _controller_committed_xet_binding(runtime: gravity.Runtime) -> dict[str, Any]:
    controller = runtime.controller()
    checkpoint, _status = _read_only_checkpoint_snapshot(controller)
    if checkpoint.get("recovery_required") is True:
        raise WorkerError(
            "controller AUTOTUNE commit has not reached a durable checkpoint"
        )
    events = controller.events.verified_events()
    commits = [
        event for event in events
        if event.get("kind") == "STATE_TRANSITION"
        and isinstance(event.get("payload"), dict)
        and event["payload"].get("to_state") == "BUILD_ADAPTER"
    ]
    if len(commits) != 1:
        raise WorkerError(
            "controller has not committed exactly one semantically validated AUTOTUNE result"
        )
    event = commits[0]
    payload = event["payload"]
    state_payload = payload.get("state_payload")
    terminal = state_payload.get("terminal_evidence") \
        if isinstance(state_payload, dict) else None
    artifacts = terminal.get("artifact_seals") if isinstance(terminal, dict) else None
    xet = artifacts.get("xet_autotune_result") if isinstance(artifacts, dict) else None
    if not isinstance(xet, dict) or set(xet) != {
        "path", "file_sha256", "seal_sha256", "schema", "status",
    }:
        raise WorkerError("controller AUTOTUNE commit lacks exact Xet artifact evidence")
    if terminal.get("seal_sha256") is None \
            or event.get("seq", -1) >= checkpoint.get("event_count", 0):
        raise WorkerError("controller AUTOTUNE commit is not checkpoint-anchored")
    return {
        "path": xet["path"],
        "file_sha256": xet["file_sha256"],
        "seal_sha256": xet["seal_sha256"],
        "schema": xet["schema"],
        "status": xet["status"],
        "controller_event_seq": event["seq"],
        "controller_event_chain_sha256": event["chain_sha256"],
        "terminal_evidence_seal_sha256": terminal["seal_sha256"],
        "checkpoint_seal_sha256": checkpoint.get("seal_sha256"),
    }


def _validate_final_schedule(
    subject: Mapping[str, Any],
    *,
    artifact_store: state.TrustedArtifactStore,
    runtime: gravity.Runtime,
) -> dict[str, Any]:
    _require_exact_keys(subject, {
        "schedule_path",
        "schedule_file_sha256",
        "schedule_seal_sha256",
        "window_schedule_sha256",
        "live_xet_result_path",
        "live_xet_result_file_sha256",
        "live_xet_result_seal_sha256",
        "controller_autotune_event_chain_sha256",
        "controller_autotune_terminal_evidence_seal_sha256",
        "post_autotune_final",
    }, "final Xet schedule subject")
    if subject.get("post_autotune_final") is not True:
        raise WorkerError("schedule readiness is not final after live Xet autotune")
    path = _relative_path(subject.get("schedule_path"), "final schedule path")
    raw = artifact_store.read_bytes(path)
    if not _is_sha256(subject.get("schedule_file_sha256")) \
            or _sha256_bytes(raw) != subject["schedule_file_sha256"]:
        raise WorkerError("final schedule bytes differ from readiness receipt")
    schedule = _validate_producer_authenticated_artifact(
        _strict_json(raw, label="final Xet schedule"),
        auth=runtime.evidence_auth,
        label="final Xet schedule",
    )
    if schedule.get("schema") != "hawking.glm52.streaming_schedule.v2" \
            or schedule.get("status") != "FROZEN_AFTER_XET_AUTOTUNE":
        raise WorkerError("final schedule is not the post-autotune v2 artifact")
    if schedule.get("campaign_id") != runtime.campaign_id \
            or schedule.get("source_revision") != runtime.source_revision \
            or schedule.get("expected_contract_sha256") != \
            runtime.expected_contract["seal_sha256"]:
        raise WorkerError("final schedule campaign identity mismatch")
    if schedule.get("seal_sha256") != subject.get("schedule_seal_sha256"):
        raise WorkerError("final schedule seal differs from readiness receipt")
    result_path = _relative_path(
        subject.get("live_xet_result_path"), "live Xet result path"
    )
    result_raw = artifact_store.read_bytes(result_path)
    if not _is_sha256(subject.get("live_xet_result_file_sha256")) \
            or _sha256_bytes(result_raw) != subject["live_xet_result_file_sha256"]:
        raise WorkerError("live Xet result bytes differ from readiness receipt")
    xet_policy = runtime.expected_contract.get("state_gates", {}).get(
        "BUILD_ADAPTER", {}
    ).get("required_artifacts", {}).get("xet_autotune_result")
    if not isinstance(xet_policy, dict) \
            or xet_policy.get("path") != result_path \
            or xet_policy.get("validator_id") != "xet_autotune_result_v1" \
            or xet_policy.get("require_producer_hmac") is not True:
        raise WorkerError("contract lacks exact semantic live-Xet result policy")
    try:
        semantic_snapshot = state._snapshot_policy_evidence(
            artifact_store,
            xet_policy,
            label="worker live-Xet result",
            contract=runtime.expected_contract,
            evidence_auth=runtime.evidence_auth,
        )
    except state.StateError as exc:
        raise WorkerError(f"live-Xet semantic validation failed: {exc}") from exc
    if semantic_snapshot.get("file_sha256") != subject["live_xet_result_file_sha256"] \
            or semantic_snapshot.get("seal_sha256") != \
            subject.get("live_xet_result_seal_sha256"):
        raise WorkerError("live-Xet semantic snapshot differs from readiness receipt")
    committed = _controller_committed_xet_binding(runtime)
    if committed["path"] != result_path \
            or committed["file_sha256"] != subject["live_xet_result_file_sha256"] \
            or committed["seal_sha256"] != subject.get("live_xet_result_seal_sha256") \
            or committed["schema"] != semantic_snapshot.get("schema") \
            or committed["status"] != semantic_snapshot.get("status") \
            or committed["controller_event_chain_sha256"] != \
            subject.get("controller_autotune_event_chain_sha256") \
            or committed["terminal_evidence_seal_sha256"] != \
            subject.get("controller_autotune_terminal_evidence_seal_sha256"):
        raise WorkerError("live-Xet result is not the exact controller-committed AUTOTUNE result")
    expected_schedule_hash = _sha256_json(runtime.expected_contract["window_schedule"])
    if subject.get("window_schedule_sha256") != expected_schedule_hash:
        raise WorkerError("readiness receipt does not bind the frozen contract window schedule")
    dependency = schedule.get("dependency_freeze")
    if not isinstance(dependency, dict) \
            or dependency.get("window_schedule_sha256") != expected_schedule_hash \
            or dependency.get("source_dependency_membership_changed") is not False \
            or dependency.get("tensor_ownership_changed") is not False:
        raise WorkerError("final schedule dependency freeze differs from the contract")
    binding = schedule.get("autotune_binding")
    xet_seal = subject.get("live_xet_result_seal_sha256")
    if not _is_sha256(xet_seal) or not isinstance(binding, dict) \
            or binding.get("xet_autotune_result_seal_sha256") != xet_seal:
        raise WorkerError("final schedule does not bind the exact live Xet result")
    resource_binding = schedule.get("resource_policy_binding")
    expected_resource_binding = {
        "path": OFFICIAL_RESOURCE_POLICY_PATH,
        "seal_sha256": OFFICIAL_RESOURCE_POLICY_SEAL,
        "required_free_disk_bytes": OFFICIAL_REQUIRED_FREE_DISK_BYTES,
        "expected_contract_sha256": runtime.expected_contract["seal_sha256"],
    }
    if resource_binding != expected_resource_binding:
        raise WorkerError(
            "final schedule does not exactly bind the frozen resource reserve policy"
        )
    windows = schedule.get("windows")
    expected_windows = runtime.expected_contract["window_schedule"]
    if not isinstance(windows, list) or len(windows) != len(expected_windows) \
            or schedule.get("window_count") != len(expected_windows):
        raise WorkerError("final schedule window count differs from contract")
    comparable = (
        "schedule_index", "window_id", "source_shards", "carry_in_shards",
        "new_fetch_shards", "refetch_shards", "carry_out_shards", "evict_shards",
        "tensor_set",
    )
    for index, (actual, expected) in enumerate(zip(windows, expected_windows)):
        if not isinstance(actual, dict) \
                or any(actual.get(key) != expected.get(key) for key in comparable):
            raise WorkerError(f"final schedule window {index} differs from contract")
    return {
        "schedule_file_sha256": subject["schedule_file_sha256"],
        "schedule_seal_sha256": subject["schedule_seal_sha256"],
        "window_schedule_sha256": expected_schedule_hash,
        "live_xet_result_seal_sha256": xet_seal,
        "live_xet_result_file_sha256": subject["live_xet_result_file_sha256"],
        "controller_autotune_event_chain_sha256":
            committed["controller_event_chain_sha256"],
        "controller_autotune_terminal_evidence_seal_sha256":
            committed["terminal_evidence_seal_sha256"],
        "resource_policy_seal_sha256": OFFICIAL_RESOURCE_POLICY_SEAL,
        "required_free_disk_bytes": OFFICIAL_REQUIRED_FREE_DISK_BYTES,
        "window_count": len(expected_windows),
    }


def _validate_resource_policy(
    subject: Mapping[str, Any],
    *,
    artifact_store: state.TrustedArtifactStore,
    runtime: gravity.Runtime,
) -> dict[str, Any]:
    _require_exact_keys(subject, {
        "policy_path",
        "policy_file_sha256",
        "policy_seal_sha256",
        "policy_schema",
        "resource_policy_digest",
        "enforced_at_every_adapter_boundary",
    }, "resource policy subject")
    if subject.get("enforced_at_every_adapter_boundary") is not True:
        raise WorkerError("resource policy is not required at every adapter boundary")
    path = _relative_path(subject.get("policy_path"), "resource policy path")
    raw = artifact_store.read_bytes(path)
    if not _is_sha256(subject.get("policy_file_sha256")) \
            or _sha256_bytes(raw) != subject["policy_file_sha256"]:
        raise WorkerError("resource policy bytes differ from readiness receipt")
    policy = _strict_json(raw, label="resource policy")
    try:
        verify_sealed(policy, label="resource policy")
    except Glm52Error as exc:
        raise WorkerError(str(exc)) from exc
    if path != OFFICIAL_RESOURCE_POLICY_PATH \
            or subject.get("policy_schema") != OFFICIAL_RESOURCE_POLICY_SCHEMA \
            or subject.get("policy_seal_sha256") != OFFICIAL_RESOURCE_POLICY_SEAL \
            or policy.get("schema") != OFFICIAL_RESOURCE_POLICY_SCHEMA \
            or policy.get("seal_sha256") != OFFICIAL_RESOURCE_POLICY_SEAL \
            or policy.get("status") != OFFICIAL_RESOURCE_POLICY_STATUS:
        raise WorkerError("resource policy schema/seal/status mismatch")
    if policy.get("revision") != runtime.source_revision:
        raise WorkerError("resource policy source revision mismatch")
    derived = policy.get("derived")
    if not isinstance(derived, dict) or derived.get("required_free_disk_bytes") != \
            OFFICIAL_REQUIRED_FREE_DISK_BYTES:
        raise WorkerError("resource policy required-free-disk floor mismatch")
    activation = policy.get("activation")
    prefetch = policy.get("prefetch_control")
    provisional = policy.get("provisional_control_limits")
    if not isinstance(activation, dict) \
            or activation.get(
                "live_allocated_byte_measurement_required_before_materialized_xet_body_acquisition"
            ) is not True \
            or activation.get(
                "remote_logical_bytes_authorize_materialized_body_acquisition"
            ) is not False \
            or not isinstance(prefetch, dict) \
            or prefetch.get("mode") != "SERIALIZED_OR_PARTIAL_PREFETCH" \
            or prefetch.get("full_two_complete_window_pipeline_preregistered") is not False \
            or prefetch.get(
                "largest_adjacent_active_plus_prefetch_union_remote_logical_bytes"
            ) != 236_190_533_120 \
            or prefetch.get("largest_adjacent_full_prefetch_deficit_bytes") != \
            10_764_360_827 \
            or not isinstance(provisional, dict) \
            or provisional.get("derived_from_sealed_input_evidence") is not False \
            or provisional.get("live_measurement_required") is not True:
        raise WorkerError("resource policy does not preserve its fail-closed prefetch boundary")
    digest = subject.get("resource_policy_digest")
    if digest != OFFICIAL_RESOURCE_POLICY_SEAL:
        raise WorkerError("resource policy digest is not the official frozen seal")
    contract_bindings = []
    for gate_name, gate in runtime.expected_contract.get("state_gates", {}).items():
        if not isinstance(gate, dict):
            continue
        for artifact_name, spec in gate.get("required_artifacts", {}).items():
            if isinstance(spec, dict) \
                    and spec.get("path") == OFFICIAL_RESOURCE_POLICY_PATH \
                    and spec.get("expected_seal_sha256") == OFFICIAL_RESOURCE_POLICY_SEAL \
                    and spec.get("expected_schema") == OFFICIAL_RESOURCE_POLICY_SCHEMA \
                    and spec.get("allowed_statuses") == [OFFICIAL_RESOURCE_POLICY_STATUS] \
                    and spec.get("validator_id") == "sealed_exact_v1" \
                    and spec.get("require_producer_hmac") is False:
                contract_bindings.append(f"{gate_name}.{artifact_name}")
    expected_bindings = {
        "AUTOTUNE_XET.resource_reserve_policy",
        "ASSEMBLE_ARTIFACT.resource_reserve_policy",
        "COMPLETE.resource_reserve_policy",
    }
    source_profile = runtime.expected_contract.get("source", {}).get("profile")
    if source_profile == "OFFICIAL_GLM52_BF16" \
            and set(contract_bindings) != expected_bindings:
        raise WorkerError(
            "official expected contract does not bind the resource policy at exact gates"
        )
    if not contract_bindings:
        raise WorkerError(
            "expected contract v3 does not bind the exact resource reserve policy seal"
        )
    return {
        "policy_file_sha256": subject["policy_file_sha256"],
        "policy_seal_sha256": OFFICIAL_RESOURCE_POLICY_SEAL,
        "resource_policy_digest": digest,
        "required_free_disk_bytes": OFFICIAL_REQUIRED_FREE_DISK_BYTES,
        "expected_contract_bindings": sorted(contract_bindings),
    }


def _call_leaf(node: ast.Call) -> str | None:
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _inspect_registered_adapter_source(
    raw: bytes,
    *,
    source_path: str,
    adapter_id: str,
    entry_points: Mapping[str, str],
) -> None:
    """Reject structurally inert adapters and import-time executable effects."""
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=source_path)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise WorkerError(
            f"execution adapter {adapter_id} is not parseable Python: {exc}"
        ) from exc
    definitions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations = [
                node.returns,
                *(argument.annotation for argument in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )),
                node.args.vararg.annotation if node.args.vararg is not None else None,
                node.args.kwarg.annotation if node.args.kwarg is not None else None,
            ]
            unsafe_annotation = any(
                isinstance(descendant, (
                    ast.Call,
                    ast.Await,
                    ast.Yield,
                    ast.YieldFrom,
                    ast.NamedExpr,
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                    ast.Lambda,
                ))
                for annotation in annotations if annotation is not None
                for descendant in ast.walk(annotation)
            )
            if node.decorator_list or node.args.defaults \
                    or any(item is not None for item in node.args.kw_defaults) \
                    or unsafe_annotation:
                raise WorkerError(
                    f"execution adapter {adapter_id} has import-time defaults/decorators/annotations"
                )
            definitions[node.name] = node
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            assigned = node.value
            try:
                ast.literal_eval(assigned) if assigned is not None else None
            except (ValueError, TypeError):
                raise WorkerError(
                    f"execution adapter {adapter_id} has executable top-level assignment"
                ) from None
            continue
        raise WorkerError(
            f"execution adapter {adapter_id} has executable top-level effects: "
            f"{type(node).__name__}"
        )
    for role, raw_name in entry_points.items():
        if not isinstance(raw_name, str) or not raw_name.isidentifier() \
                or raw_name.startswith("_") \
                or any(
                    marker in raw_name.lower()
                    for marker in ("test", "fake", "stub", "noop")
                ):
            raise WorkerError(
                f"execution adapter {adapter_id} role {role} names a non-production entry point"
            )
        name = _require_name(raw_name, f"execution adapter {adapter_id}.{role}")
        function = definitions.get(name)
        if function is None:
            raise WorkerError(f"execution adapter {adapter_id} entry point is absent: {name}")
        executable = list(function.body)
        if executable and isinstance(executable[0], ast.Expr) \
                and isinstance(executable[0].value, ast.Constant) \
                and isinstance(executable[0].value.value, str):
            executable = executable[1:]
        if not executable or any(isinstance(node, ast.Pass) for node in executable) \
                or isinstance(executable[0], ast.Raise) \
                or (len(executable) == 1 and isinstance(executable[0], ast.Return)):
            raise WorkerError(
                f"execution adapter {adapter_id} entry point is disabled/no-op: {name}"
            )
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        if not calls:
            raise WorkerError(
                f"execution adapter {adapter_id} entry point performs no grounded operation: {name}"
            )
        for call in calls:
            leaf = (_call_leaf(call) or "").lower()
            if "test_only" in leaf or any(
                marker in leaf for marker in ("fake", "stub", "noop")
            ):
                raise WorkerError(
                    f"execution adapter {adapter_id} entry point reaches private/test code: {leaf}"
                )


def _validate_execution_adapters(
    subject: Mapping[str, Any],
    *,
    workspace_store: state.TrustedArtifactStore,
    resource_policy_digest: str,
) -> dict[str, Any]:
    _require_exact_keys(subject, {
        "adapters",
        "registered_dispatch_digest",
        "resource_policy_digest",
    }, "execution adapters subject")
    if subject.get("resource_policy_digest") != resource_policy_digest:
        raise WorkerError("execution adapters bind a different resource policy")
    if not PRODUCTION_EXECUTION_ADAPTER_REGISTRY:
        raise WorkerError(
            "NO_REVIEWED_EXECUTION_ADAPTER_REGISTRY: compiled production registry is empty"
        )
    adapters = subject.get("adapters")
    if not isinstance(adapters, list) or not adapters:
        raise WorkerError("execution adapter inventory is empty")
    normalized: list[dict[str, Any]] = []
    roles: set[str] = set()
    ids: set[str] = set()
    for index, item in enumerate(adapters):
        if not isinstance(item, dict):
            raise WorkerError(f"execution adapter {index} is not an object")
        _require_exact_keys(item, {
            "adapter_id",
            "interface_version",
            "roles",
            "entry_points",
            "source_path",
            "source_sha256",
            "resource_policy_digest",
        }, f"execution adapter {index}")
        adapter_id = _require_name(item.get("adapter_id"), f"adapter[{index}].adapter_id")
        if adapter_id in ids:
            raise WorkerError(f"duplicate execution adapter id: {adapter_id}")
        ids.add(adapter_id)
        registered = PRODUCTION_EXECUTION_ADAPTER_REGISTRY.get(adapter_id)
        if not isinstance(registered, Mapping):
            raise WorkerError(f"execution adapter is absent from reviewed registry: {adapter_id}")
        if item.get("interface_version") != EXECUTION_ADAPTER_INTERFACE:
            raise WorkerError(f"execution adapter {adapter_id} interface is not executable v1")
        adapter_roles = item.get("roles")
        if not isinstance(adapter_roles, list) or not adapter_roles \
                or len(adapter_roles) != len(set(adapter_roles)) \
                or any(role not in REQUIRED_EXECUTION_ROLES for role in adapter_roles):
            raise WorkerError(f"execution adapter {adapter_id} roles are invalid")
        entry_points = item.get("entry_points")
        if not isinstance(entry_points, dict) or set(entry_points) != set(adapter_roles):
            raise WorkerError(
                f"execution adapter {adapter_id} entry points do not cover exact roles"
            )
        source_path = _relative_path(
            item.get("source_path"), f"adapter[{index}].source_path"
        )
        raw = workspace_store.read_bytes(source_path)
        source_sha = item.get("source_sha256")
        if not _is_sha256(source_sha) or _sha256_bytes(raw) != source_sha:
            raise WorkerError(f"execution adapter {adapter_id} source hash mismatch")
        if item.get("resource_policy_digest") != resource_policy_digest:
            raise WorkerError(f"execution adapter {adapter_id} policy binding mismatch")
        expected_registry_entry = {
            "interface_version": item["interface_version"],
            "roles": item["roles"],
            "entry_points": item["entry_points"],
            "source_path": item["source_path"],
            "source_sha256": item["source_sha256"],
        }
        if dict(registered) != expected_registry_entry:
            raise WorkerError(f"execution adapter differs from reviewed registry: {adapter_id}")
        _inspect_registered_adapter_source(
            raw,
            source_path=source_path,
            adapter_id=adapter_id,
            entry_points=entry_points,
        )
        roles.update(adapter_roles)
        normalized.append(json.loads(canonical(item).decode("utf-8")))
    missing = sorted(REQUIRED_EXECUTION_ROLES - roles)
    if missing:
        raise WorkerError(f"execution adapters lack roles: {missing}")
    dispatch_digest = _sha256_json(normalized)
    if subject.get("registered_dispatch_digest") != dispatch_digest:
        raise WorkerError("execution adapter dispatch digest mismatch")
    return {
        "adapter_count": len(normalized),
        "roles": sorted(roles),
        "registered_dispatch_digest": dispatch_digest,
        "resource_policy_digest": resource_policy_digest,
        "reviewed_registry_sha256": PRODUCTION_EXECUTION_ADAPTER_REGISTRY_SHA256,
    }


def _validate_all_readiness(
    config: WorkerConfig,
    runtime: gravity.Runtime,
    *,
    telegram_credential_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    artifact_store = state.TrustedArtifactStore(str(runtime.artifact_root))
    workspace_store = state.TrustedArtifactStore(str(config.workspace_root))
    receipts: dict[str, dict[str, Any]] = {}
    for kind in READINESS_KINDS:
        spec = config.readiness[kind]
        raw = artifact_store.read_bytes(spec.path)
        receipts[kind] = _validate_receipt_envelope(
            raw,
            expected_kind=kind,
            expected_file_sha256=spec.expected_file_sha256,
            runtime=runtime,
        )
    results: dict[str, Any] = {}
    results[TELEGRAM_ROTATION] = _validate_telegram_rotation(
        receipts[TELEGRAM_ROTATION]["subject"],
        runtime,
        telegram_credential_status,
    )
    results[NOTIFICATION_OUTBOX] = _validate_notification_outbox_static(
        receipts[NOTIFICATION_OUTBOX]["subject"],
        workspace_store=workspace_store,
        runtime=runtime,
        telegram_rotation_receipt_seal_sha256=receipts[TELEGRAM_ROTATION][
            "seal_sha256"
        ],
        telegram_rotation=results[TELEGRAM_ROTATION],
    )
    results[FINAL_XET_SCHEDULE] = _validate_final_schedule(
        receipts[FINAL_XET_SCHEDULE]["subject"],
        artifact_store=artifact_store,
        runtime=runtime,
    )
    results[RESOURCE_POLICY] = _validate_resource_policy(
        receipts[RESOURCE_POLICY]["subject"],
        artifact_store=artifact_store,
        runtime=runtime,
    )
    results[EXECUTION_ADAPTERS] = _validate_execution_adapters(
        receipts[EXECUTION_ADAPTERS]["subject"],
        workspace_store=workspace_store,
        resource_policy_digest=results[RESOURCE_POLICY]["resource_policy_digest"],
    )
    return results


def _inspect_contract_without_credentials(
    config: WorkerConfig,
) -> tuple[dict[str, Any] | None, str | None]:
    """Read controller config and expected contract without touching Keychain or network."""
    try:
        observed = _read_controller_config_binding(config.controller_config_path)
        expected = (
            config.controller_config_file_sha256,
            config.controller_config_seal_sha256,
            config.campaign_id,
            config.source_revision,
            config.controller_root,
            config.artifact_root,
            config.expected_contract_path,
        )
        actual = (
            observed.file_sha256,
            observed.seal_sha256,
            observed.campaign_id,
            observed.source_revision,
            observed.controller_root,
            observed.artifact_root,
            observed.expected_contract_path,
        )
        if actual != expected:
            raise WorkerError(
                "controller config bytes/targets differ from authenticated worker config"
            )
        contract_path = config.expected_contract_path
        contract_store = state.TrustedArtifactStore(str(contract_path.parent))
        contract_raw = contract_store.read_bytes(contract_path.name)
        contract = _strict_json(contract_raw, label="expected campaign contract")
        state._validate_expected_contract(contract)
        return contract, None
    except (WorkerError, state.StateError, Glm52Error, OSError) as exc:
        return None, str(exc)


def _read_only_checkpoint_snapshot(
    controller: state.Controller,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a replay-validated checkpoint or prove one exact recoverable crash tail.

    This never takes the mutation lease and never performs recovery.  The persistent
    worker will call :meth:`Controller.resume` after acquiring the lease; preflight
    merely distinguishes that controller-supported single-tail case from an ambiguous
    fork so a legitimate crash cannot permanently prevent its own recovery.
    """
    status_before = controller.status()
    try:
        store = state.TrustedArtifactStore(str(controller.root))
        if status_before.get("durable_state_ok"):
            checkpoint, _file_sha256 = store.read_sealed(
                controller.checkpoint_path.name,
                label="controller checkpoint",
            )
            status_after = controller.status()
            if status_before != status_after:
                raise WorkerError("controller changed while checkpoint preflight was sampled")
            if checkpoint.get("seal_sha256") is None \
                    or checkpoint.get("state") != status_before.get("state"):
                raise WorkerError("controller checkpoint/status identity mismatch")
            return checkpoint, status_before

        # The normal status path is red for any log/checkpoint count difference.  Re-read
        # both authenticated hash chains and accept only Controller.resume's exact one-tail
        # recovery envelope; no state is written here.
        for path in (controller.events.path, controller.window_ledger.log.path):
            if path.exists():
                metadata = os.lstat(path)
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) \
                        or metadata.st_nlink != 1:
                    raise WorkerError(f"durable log target is unsafe: {path.name}")
        events = controller.events.verified_events()
        _records, _owners, window_events = controller.window_ledger._replay()
        events_head = controller.events.head_hash(events)
        windows_head = controller.window_ledger.log.head_hash(window_events)
        if controller.checkpoint_path.exists():
            checkpoint, _file_sha256 = store.read_sealed(
                controller.checkpoint_path.name,
                label="controller checkpoint",
            )
            if checkpoint.get("schema") != state.CHECKPOINT_SCHEMA:
                raise WorkerError("controller checkpoint schema mismatch")
            for key, expected in (
                ("campaign_id", controller.campaign_id),
                ("source_revision", controller.source_revision),
                ("controller_epoch", controller.controller_epoch),
                ("expected_contract_sha256", controller.expected_contract_sha256),
            ):
                if checkpoint.get(key) != expected:
                    raise WorkerError(f"controller checkpoint {key} mismatch")
            event_count = checkpoint.get("event_count")
            window_count = checkpoint.get("window_event_count")
            if isinstance(event_count, bool) or not isinstance(event_count, int) \
                    or isinstance(window_count, bool) or not isinstance(window_count, int) \
                    or event_count < 0 or window_count < 0 \
                    or event_count > len(events) or window_count > len(window_events):
                raise WorkerError("checkpoint/log counts are not a recoverable prefix")
            if controller._anchor_at(events, event_count) != checkpoint.get("event_head_hash") \
                    or controller._anchor_at(window_events, window_count) != \
                    checkpoint.get("window_event_head_hash"):
                raise WorkerError("checkpoint/log prefix is a fork")
            tails = (len(events) - event_count, len(window_events) - window_count)
            if tails not in {(1, 0), (0, 1)}:
                raise WorkerError(
                    f"ambiguous uncheckpointed tails: controller={tails[0]}, window={tails[1]}"
                )
            previous_seal = checkpoint.get("seal_sha256")
        else:
            if len(events) != 1 or window_events:
                raise WorkerError("checkpoint absent without one valid genesis tail")
            if controller._replay_controller(events).get("state") != "PRECHECK":
                raise WorkerError("checkpoint-absent genesis is not PRECHECK")
            checkpoint = {}
            tails = (1, 0)
            previous_seal = None

        # Full semantic replay proves the tail is not merely hash-valid but legal relative
        # to the controller/window state machines.  Its generated timestamp is discarded.
        expected = controller._checkpoint_document(events, window_events)
        events_after = controller.events.verified_events()
        _r2, _o2, windows_after = controller.window_ledger._replay()
        if len(events_after) != len(events) \
                or controller.events.head_hash(events_after) != events_head \
                or len(windows_after) != len(window_events) \
                or controller.window_ledger.log.head_hash(windows_after) != windows_head:
            raise WorkerError("durable logs changed during crash-tail preflight")
        status_after = controller.status()
        if status_after != status_before:
            raise WorkerError("controller changed during crash-tail preflight")
        snapshot = {
            **expected,
            "seal_sha256": previous_seal,
            "recovery_required": True,
            "recoverable_tail": {
                "controller_events": tails[0],
                "window_events": tails[1],
            },
        }
        return snapshot, {
            **status_before,
            "single_tail_recoverable": True,
            "recovery_required": True,
        }
    except (state.StateError, OSError) as exc:
        reasons = status_before.get("checkpoint_reasons") or []
        raise WorkerError(
            f"controller checkpoint replay is not green or singly recoverable: "
            f"status={reasons}; inspection={exc}"
        ) from exc


def evaluate_preflight(
    config: WorkerConfig,
    *,
    runtime: gravity.Runtime | None,
    runtime_error: str | None = None,
    contract: Mapping[str, Any] | None = None,
    telegram_credential_status: Mapping[str, Any] | None = None,
    allow_test_profile: bool = False,
) -> dict[str, Any]:
    """Return every currently knowable blocker; never starts, downloads, or mutates."""
    checks: dict[str, dict[str, Any]] = {}
    blockers: list[dict[str, str]] = []

    def pass_check(code: str, detail: Mapping[str, Any] | None = None) -> None:
        checks[code] = {"status": "PASS", "detail": dict(detail or {})}

    def block(code: str, reason: str) -> None:
        checks[code] = {"status": "BLOCKED", "reason": reason}
        blockers.append({"code": code, "reason": reason})

    if runtime is None:
        block(
            "worker_config_authority",
            "requires the campaign-bound independent evidence HMAC authority",
        )
    else:
        try:
            pass_check(
                "worker_config_authority",
                _validate_worker_config_authority(config, runtime),
            )
        except (WorkerError, state.StateError) as exc:
            block("worker_config_authority", str(exc))

    try:
        if config.profile == OFFICIAL_CONFIG_PROFILE:
            if config.campaign_id != OFFICIAL_CAMPAIGN_ID \
                    or config.source_revision != OFFICIAL_REVISION:
                raise WorkerError("official worker config identity mismatch")
            if OFFICIAL_CONTRACT_SEAL is None:
                raise WorkerError(
                    "OFFICIAL_CONTRACT_SEAL_UNSET: reviewed official contract v3 does not exist"
                )
            if config.expected_contract_sha256 != OFFICIAL_CONTRACT_SEAL:
                raise WorkerError("worker config does not bind the compiled official contract seal")
            pass_check(
                "official_contract_authority",
                {"official_contract_seal_sha256": OFFICIAL_CONTRACT_SEAL},
            )
        elif config.profile == SYNTHETIC_CONFIG_PROFILE and allow_test_profile:
            pass_check(
                "official_contract_authority",
                {"profile": SYNTHETIC_CONFIG_PROFILE, "test_only": True},
            )
        else:
            raise WorkerError("synthetic worker config requires explicit test-only opt-in")
    except WorkerError as exc:
        block("official_contract_authority", str(exc))

    validated_contract: dict[str, Any] | None = None
    if contract is None:
        contract, contract_error = _inspect_contract_without_credentials(config)
    else:
        contract_error = None
    try:
        if contract_error:
            raise WorkerError(contract_error)
        validated_contract = state._validate_expected_contract(dict(contract or {}))
        if validated_contract.get("schema") != OFFICIAL_CONTRACT_SCHEMA:
            raise WorkerError("contract is not schema v3")
        if validated_contract.get("seal_sha256") != config.expected_contract_sha256:
            raise WorkerError("contract seal differs from worker config")
        profile = validated_contract.get("source", {}).get("profile")
        if config.profile == OFFICIAL_CONFIG_PROFILE \
                and profile != "OFFICIAL_GLM52_BF16":
            raise WorkerError("official worker config requires the official BF16 contract")
        if config.profile == SYNTHETIC_CONFIG_PROFILE \
                and (profile != "SYNTHETIC_TEST_ONLY" or not allow_test_profile):
            raise WorkerError("synthetic contract requires explicit test-only profile")
        if profile != "OFFICIAL_GLM52_BF16" and not allow_test_profile:
            raise WorkerError("production worker refuses a synthetic source contract")
        pass_check("contract_v3", {
            "seal_sha256": validated_contract["seal_sha256"],
            "source_profile": profile,
        })
    except (WorkerError, state.StateError, TypeError, ValueError) as exc:
        block("contract_v3", str(exc))

    try:
        if validated_contract is None:
            raise WorkerError("contract v3 is unavailable")
        created = _require_utc_seconds(
            validated_contract.get("created_at"), "contract created_at"
        )
        if created != config.expected_contract_created_at:
            raise WorkerError("contract created_at differs from the worker's frozen timestamp")
        pass_check("deterministic_created_at", {"created_at": created})
    except WorkerError as exc:
        block("deterministic_created_at", str(exc))

    if runtime is None:
        block(
            "runtime_credentials",
            runtime_error or "controller runtime and Keychain credentials are unavailable",
        )
        for code in (
            "rotated_telegram_identity",
            "authentication_role_separation",
            "notification_outbox_ready",
            "final_xet_schedule",
            "resource_policy_digest",
            "execution_adapters_grounded",
            "durable_checkpoint_replay",
        ):
            block(code, "requires a validated controller runtime")
    else:
        try:
            if runtime.campaign_id != OFFICIAL_CAMPAIGN_ID \
                    or runtime.source_revision != OFFICIAL_REVISION:
                if config.profile != SYNTHETIC_CONFIG_PROFILE or not allow_test_profile:
                    raise WorkerError("runtime is not the official GLM-5.2 campaign identity")
            if runtime.campaign_id != config.campaign_id \
                    or runtime.source_revision != config.source_revision:
                raise WorkerError("runtime identity differs from authenticated worker config")
            if runtime.expected_contract.get("seal_sha256") != \
                    config.expected_contract_sha256:
                raise WorkerError("runtime contract differs from worker config")
            pass_check("runtime_credentials", {
                "campaign_id": runtime.campaign_id,
                "source_revision": runtime.source_revision,
            })
        except WorkerError as exc:
            block("runtime_credentials", str(exc))

        try:
            pass_check(
                "authentication_role_separation",
                _validate_runtime_authentication_roles(runtime),
            )
        except (AttributeError, WorkerError) as exc:
            block("authentication_role_separation", str(exc))

        readiness: dict[str, Any] = {}
        readiness_error: str | None = None
        try:
            readiness = _validate_all_readiness(
                config,
                runtime,
                telegram_credential_status=telegram_credential_status,
            )
        except (WorkerError, state.StateError, OSError) as exc:
            readiness_error = str(exc)

        mapping = {
            TELEGRAM_ROTATION: "rotated_telegram_identity",
            NOTIFICATION_OUTBOX: "notification_outbox_ready",
            FINAL_XET_SCHEDULE: "final_xet_schedule",
            RESOURCE_POLICY: "resource_policy_digest",
            EXECUTION_ADAPTERS: "execution_adapters_grounded",
        }
        if readiness_error is not None:
            # Validate independently as far as possible so status reports exact receipt blockers.
            artifact_store: state.TrustedArtifactStore | None = None
            controller_store: state.TrustedArtifactStore | None = None
            workspace_store: state.TrustedArtifactStore | None = None
            validated_receipts: dict[str, dict[str, Any]] = {}
            try:
                artifact_store = state.TrustedArtifactStore(str(runtime.artifact_root))
                controller_store = state.TrustedArtifactStore(str(runtime.controller_root))
                workspace_store = state.TrustedArtifactStore(str(config.workspace_root))
            except state.StateError as exc:
                readiness_error = str(exc)
            resource_digest: str | None = None
            for kind in READINESS_KINDS:
                code = mapping[kind]
                try:
                    if artifact_store is None or controller_store is None \
                            or workspace_store is None:
                        raise WorkerError(readiness_error or "trusted roots unavailable")
                    spec = config.readiness[kind]
                    receipt = _validate_receipt_envelope(
                        artifact_store.read_bytes(spec.path),
                        expected_kind=kind,
                        expected_file_sha256=spec.expected_file_sha256,
                        runtime=runtime,
                    )
                    validated_receipts[kind] = receipt
                    if kind == TELEGRAM_ROTATION:
                        detail = _validate_telegram_rotation(
                            receipt["subject"],
                            runtime,
                            telegram_credential_status,
                        )
                    elif kind == NOTIFICATION_OUTBOX:
                        rotation_receipt = validated_receipts.get(TELEGRAM_ROTATION) or {}
                        rotation_detail = _validate_telegram_rotation(
                            rotation_receipt.get("subject", {}),
                            runtime,
                            telegram_credential_status,
                        )
                        detail = _validate_notification_outbox_static(
                            receipt["subject"],
                            workspace_store=workspace_store,
                            runtime=runtime,
                            telegram_rotation_receipt_seal_sha256=
                                rotation_receipt.get("seal_sha256", ""),
                            telegram_rotation=rotation_detail,
                        )
                    elif kind == FINAL_XET_SCHEDULE:
                        detail = _validate_final_schedule(
                            receipt["subject"],
                            artifact_store=artifact_store,
                            runtime=runtime,
                        )
                    elif kind == RESOURCE_POLICY:
                        detail = _validate_resource_policy(
                            receipt["subject"],
                            artifact_store=artifact_store,
                            runtime=runtime,
                        )
                        resource_digest = detail["resource_policy_digest"]
                    else:
                        if resource_digest is None:
                            resource_receipt = validated_receipts.get(RESOURCE_POLICY)
                            if resource_receipt is None:
                                raise WorkerError("resource policy readiness is unavailable")
                            resource_detail = _validate_resource_policy(
                                resource_receipt["subject"],
                                artifact_store=artifact_store,
                                runtime=runtime,
                            )
                            resource_digest = resource_detail["resource_policy_digest"]
                        detail = _validate_execution_adapters(
                            receipt["subject"],
                            workspace_store=workspace_store,
                            resource_policy_digest=resource_digest,
                        )
                    if kind == NOTIFICATION_OUTBOX:
                        block(code, LEASE_BOUND_NOTIFICATION_OUTBOX_AUDIT_REQUIRED)
                    else:
                        pass_check(code, detail)
                except (WorkerError, state.StateError, OSError) as exc:
                    block(code, str(exc))
        else:
            for kind, code in mapping.items():
                if kind == NOTIFICATION_OUTBOX:
                    block(code, LEASE_BOUND_NOTIFICATION_OUTBOX_AUDIT_REQUIRED)
                else:
                    pass_check(code, readiness[kind])

        try:
            controller = runtime.controller()
            checkpoint, controller_status = _read_only_checkpoint_snapshot(controller)
            if checkpoint.get("state") in {None, "COMPLETE"}:
                raise WorkerError("controller checkpoint is absent or already COMPLETE")
            pass_check("durable_checkpoint_replay", {
                "state": checkpoint["state"],
                "event_count": checkpoint["event_count"],
                "window_event_count": checkpoint["window_event_count"],
                "checkpoint_seal_sha256": checkpoint["seal_sha256"],
                "recovery_required": bool(checkpoint.get("recovery_required")),
                "recoverable_tail": checkpoint.get("recoverable_tail"),
            })
        except (WorkerError, state.StateError, OSError) as exc:
            block("durable_checkpoint_replay", str(exc))

    # Deliberately permanent for this scaffold.  Source/receipt validation establishes
    # prerequisites, but cannot substitute for a real dispatcher implementation wired to
    # Controller transitions and phase receipts.  Remove this blocker only in the same
    # reviewed change that adds and binds that callable.
    block("scientific_dispatch_bound", NO_SCIENTIFIC_DISPATCH_REASON)
    authorized = not blockers
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY" if authorized else "BLOCKED",
        "production_start_authorized": authorized,
        "worker_config_sha256": config.seal_sha256,
        "expected_contract_sha256": config.expected_contract_sha256,
        "checks": checks,
        "blockers": blockers,
        "side_effects": {
            "network_access": False,
            "xet_body_bytes_read": 0,
            "telegram_messages_sent": 0,
            "scientific_phases_executed": 0,
            "files_deleted": 0,
        },
        "session_limitations": SESSION_LIMITATIONS,
    }


def preflight(
    config: WorkerConfig,
    *,
    keychain: Any = None,
    evidence_keychain: Any = None,
    grounding_keychain: Any = None,
) -> tuple[dict[str, Any], gravity.Runtime | None]:
    contract, _contract_error = _inspect_contract_without_credentials(config)
    runtime: gravity.Runtime | None = None
    runtime_error: str | None = None
    telegram_credential_status: Mapping[str, Any] | None = None
    telegram_keychain = keychain or gravity.telegram_module.MacOSKeychain()
    try:
        runtime = gravity.load_runtime(
            config.controller_config_path,
            keychain=telegram_keychain,
            evidence_keychain=evidence_keychain,
            grounding_keychain=grounding_keychain,
        )
        telegram_credential_status = gravity.telegram_module.credential_status(
            telegram_keychain
        )
    except (gravity.CliError, state.StateError, Glm52Error, OSError) as exc:
        runtime_error = str(exc)
    return evaluate_preflight(
        config,
        runtime=runtime,
        runtime_error=runtime_error,
        contract=contract,
        telegram_credential_status=telegram_credential_status,
    ), runtime


def inspect_caffeinate_parent(
    *,
    parent_pid: int | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    expected_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Verify that this process is a direct child of /usr/bin/caffeinate -dimsu."""
    pid = os.getppid() if parent_pid is None else parent_pid
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return {"ok": False, "reason": "invalid caffeinate parent PID", "parent_pid": pid}
    try:
        common = {
            "text": True,
            "capture_output": True,
            "check": False,
            "timeout": 5,
        }
        comm_result = runner(
            [str(PS), "-p", str(pid), "-o", "comm="], **common
        )
        command_result = runner(
            [str(PS), "-p", str(pid), "-o", "command="], **common
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "reason": f"cannot inspect parent process: {exc}", "parent_pid": pid}
    if comm_result.returncode != 0 or command_result.returncode != 0:
        return {"ok": False, "reason": "parent process inspection failed", "parent_pid": pid}
    comm = comm_result.stdout.strip()
    command = command_result.stdout.strip()
    if not comm or not command:
        return {"ok": False, "reason": "parent process inspection was empty", "parent_pid": pid}
    if comm not in {str(CAFFEINATE), "caffeinate"}:
        return {"ok": False, "reason": "parent is not /usr/bin/caffeinate", "parent_pid": pid}
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if not tokens or tokens[0] != str(CAFFEINATE):
        return {"ok": False, "reason": "parent caffeinate path is not /usr/bin/caffeinate", "parent_pid": pid}
    flags: set[str] = set()
    for token in tokens[1:]:
        if token == "--":
            break
        if not token.startswith("-") or token == "-":
            break
        flags.update(token[1:])
    missing = sorted(CAFFEINATE_FLAGS - flags)
    if missing:
        return {
            "ok": False,
            "reason": f"caffeinate lacks required flags: {missing}",
            "parent_pid": pid,
        }
    if expected_argv is not None and tokens != list(expected_argv):
        return {
            "ok": False,
            "reason": "parent caffeinate argv differs from the exact approved invocation",
            "parent_pid": pid,
        }
    return {
        "ok": True,
        "parent_pid": pid,
        "binary": str(CAFFEINATE),
        "flags": "-dimsu",
    }


def caffeinate_exec_argv(config: WorkerConfig) -> list[str]:
    if config.profile == OFFICIAL_CONFIG_PROFILE:
        observed = {
            "workspace_root": Path(os.path.abspath(config.workspace_root)),
            "worker_config": Path(os.path.abspath(config.path)),
            "python": Path(os.path.abspath(sys.executable)),
            "worker": Path(os.path.abspath(__file__)),
        }
        expected = {
            "workspace_root": PINNED_WORKSPACE_ROOT,
            "worker_config": PINNED_WORKER_CONFIG,
            "python": PINNED_PYTHON,
            "worker": PINNED_WORKER,
        }
        if observed != expected:
            raise WorkerError(
                "official caffeinate invocation differs from compiled absolute path pins"
            )
        return list(PINNED_LAUNCHD_ARGUMENTS)
    return [
        str(CAFFEINATE),
        "-dimsu",
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config.path),
        "--under-caffeinate",
        "run",
    ]


class SignalLatch:
    """Signal handlers only latch intent; the worker consumes it at a safe boundary."""

    def __init__(self, *, wake: Callable[[], Any] | None = None) -> None:
        self._lock = threading.Lock()
        self._signal: int | None = None
        self._previous: dict[int, Any] = {}
        self._wake = wake

    @property
    def requested_signal(self) -> int | None:
        with self._lock:
            return self._signal

    def request(self, signum: int) -> None:
        changed = False
        with self._lock:
            if self._signal is None:
                self._signal = signum
                changed = True
        if changed and self._wake is not None:
            self._wake()

    def _handler(self, signum: int, _frame: Any) -> None:
        self.request(signum)

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            raise WorkerError("signal handlers must be installed by the worker main thread")
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handler)

    def restore(self) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous.clear()


def _validate_authenticated_heartbeat(
    heartbeat: Any,
    *,
    runtime: gravity.Runtime,
) -> dict[str, Any] | None:
    """Validate a prior worker heartbeat if the checkpoint contains one."""
    if heartbeat is None:
        return None
    if not isinstance(heartbeat, dict):
        raise WorkerError("checkpoint heartbeat is malformed")
    telemetry = heartbeat.get("telemetry")
    if not isinstance(telemetry, dict) or telemetry.get("schema") != HEARTBEAT_SCHEMA:
        # State-transition heartbeats predate the worker schema and remain valid controller
        # history; only records claiming to be worker heartbeats require this HMAC.
        return None
    required = {
        "schema",
        "campaign_id",
        "source_revision",
        "controller_epoch",
        "expected_contract_sha256",
        "worker_instance_id",
        "worker_pid",
        "safe_point",
        "lifecycle",
        "controller_event_count_before",
        "window_event_count",
        "checkpoint_seal_sha256_before",
        "control",
        "session_limitations_sha256",
        "producer_hmac_sha256",
    }
    _require_exact_keys(telemetry, required, "worker heartbeat telemetry")
    for key, expected in (
        ("campaign_id", runtime.campaign_id),
        ("source_revision", runtime.source_revision),
        ("controller_epoch", runtime.controller_epoch),
        ("expected_contract_sha256", runtime.expected_contract["seal_sha256"]),
    ):
        if telemetry.get(key) != expected:
            raise WorkerError(f"worker heartbeat {key} mismatch")
    if telemetry.get("safe_point") not in gravity.WORKER_SAFE_POINTS:
        raise WorkerError("worker heartbeat safe point is invalid")
    if telemetry.get("session_limitations_sha256") != _sha256_bytes(
        SESSION_LIMITATIONS.encode("utf-8")
    ):
        raise WorkerError("worker heartbeat session-limitations binding mismatch")
    signature = telemetry.get("producer_hmac_sha256")
    unsigned = {key: item for key, item in telemetry.items() if key != "producer_hmac_sha256"}
    if not runtime.evidence_auth.verify({
        "schema": "hawking.glm52.worker_heartbeat_auth.v1",
        "heartbeat": unsigned,
    }, signature):
        raise WorkerError("worker heartbeat producer HMAC failed")
    return dict(telemetry)


def _lease_target_is_safe(controller: state.Controller) -> None:
    path = controller.lease.path
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise WorkerError(f"cannot inspect controller lease target: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) \
            or metadata.st_nlink != 1:
        raise WorkerError("controller lease target is not one no-symlink regular file")


def _validate_held_lease(controller: state.Controller) -> dict[str, Any]:
    controller.lease.assert_held()
    handle = controller.lease._handle
    if handle is None:
        raise WorkerError("controller lease handle disappeared")
    descriptor = os.fstat(handle.fileno())
    named = os.lstat(controller.lease.path)
    if not stat.S_ISREG(descriptor.st_mode) or descriptor.st_nlink != 1 \
            or stat.S_ISLNK(named.st_mode) or not stat.S_ISREG(named.st_mode) \
            or named.st_nlink != 1 \
            or (descriptor.st_dev, descriptor.st_ino) != (named.st_dev, named.st_ino):
        raise WorkerError("held controller lease name/descriptor identity is unsafe")
    observation = controller.lease.probe()
    if observation.get("held_by_this_handle") is not True \
            or observation.get("owner_pid") != os.getpid() \
            or observation.get("controller_epoch") != controller.controller_epoch:
        raise WorkerError("held controller lease owner record is inconsistent")
    return observation


class PersistentWorker:
    """Long-lived lease/heartbeat/control owner; contains no scientific executor."""

    def __init__(
        self,
        runtime: gravity.Runtime,
        *,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        waiter: Callable[[float], bool] | None = None,
        signal_latch: SignalLatch | None = None,
        instance_id: str | None = None,
        notification_audit_readiness: NotificationAuditReadiness | None = None,
        test_only_notification_audit_bypass: bool = False,
    ) -> None:
        if isinstance(heartbeat_interval_seconds, bool) \
                or not 0.0 <= heartbeat_interval_seconds <= MAX_HEARTBEAT_INTERVAL_SECONDS:
            raise WorkerError("worker heartbeat interval is invalid")
        if not isinstance(test_only_notification_audit_bypass, bool):
            raise WorkerError("notification audit bypass flag must be boolean")
        source_profile = runtime.expected_contract.get("source", {}).get("profile")
        synthetic_test_runtime = (
            runtime.allow_synthetic_contract is True
            and source_profile == "SYNTHETIC_TEST_ONLY"
            and runtime.campaign_id != OFFICIAL_CAMPAIGN_ID
        )
        if test_only_notification_audit_bypass:
            if notification_audit_readiness is not None:
                raise WorkerError(
                    "test-only notification audit bypass cannot accompany readiness"
                )
            if not synthetic_test_runtime:
                raise WorkerError(
                    "notification audit bypass is restricted to synthetic test runtimes"
                )
        elif notification_audit_readiness is not None \
                and not isinstance(
                    notification_audit_readiness, NotificationAuditReadiness
                ):
            raise WorkerError("notification audit readiness has the wrong type")
        if not synthetic_test_runtime \
                and notification_audit_readiness is None:
            raise WorkerError(LEASE_BOUND_NOTIFICATION_OUTBOX_AUDIT_REQUIRED)
        self.runtime = runtime
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self._wake = threading.Event()
        self.waiter = waiter or self._wake.wait
        self.signal_latch = signal_latch or SignalLatch(wake=self._wake.set)
        self.instance_id = instance_id or (
            f"worker-{os.getpid()}-{os.urandom(12).hex()}"
        )
        self.notification_audit_readiness = notification_audit_readiness
        self.test_only_notification_audit_bypass = \
            test_only_notification_audit_bypass
        self.last_notification_audit: Mapping[str, Any] | None = None
        _require_name(self.instance_id, "worker instance id")
        self.controller = runtime.controller()
        self.controls = gravity.WorkerControlAcknowledgements(runtime)

    def _notification_audit_gate(self) -> dict[str, Any]:
        """Audit at a held-lease safe boundary before any heartbeat/dispatch seam."""
        _validate_held_lease(self.controller)
        if self.test_only_notification_audit_bypass:
            result = {
                "status": "TEST_ONLY_EXPLICIT_BYPASS",
                "lease_bound_semantic_audit_verified": False,
                "scientific_dispatch_permitted": False,
            }
        else:
            if self.notification_audit_readiness is None:
                raise WorkerError(LEASE_BOUND_NOTIFICATION_OUTBOX_AUDIT_REQUIRED)
            result = _audit_frozen_notification_readiness_under_lease(
                self.notification_audit_readiness,
                runtime=self.runtime,
                controller=self.controller,
            )
        self.last_notification_audit = MappingProxyType(dict(result))
        return dict(result)

    def _heartbeat(
        self,
        *,
        safe_point: str,
        lifecycle: str,
        control: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.controller.lease.assert_held()
        checkpoint = self.controller.resume()
        body = {
            "schema": HEARTBEAT_SCHEMA,
            "campaign_id": self.runtime.campaign_id,
            "source_revision": self.runtime.source_revision,
            "controller_epoch": self.runtime.controller_epoch,
            "expected_contract_sha256": self.runtime.expected_contract["seal_sha256"],
            "worker_instance_id": self.instance_id,
            "worker_pid": os.getpid(),
            "safe_point": safe_point,
            "lifecycle": lifecycle,
            "controller_event_count_before": checkpoint["event_count"],
            "window_event_count": checkpoint["window_event_count"],
            "checkpoint_seal_sha256_before": checkpoint["seal_sha256"],
            "control": {
                "worker_directive": control.get("worker_directive"),
                "applied_action": control.get("applied_action"),
                "applied_request_sequence": control.get("applied_request_sequence"),
                "application_state": control.get("application_state"),
            },
            "session_limitations_sha256": _sha256_bytes(
                SESSION_LIMITATIONS.encode("utf-8")
            ),
        }
        body["producer_hmac_sha256"] = self.runtime.evidence_auth.authenticate({
            "schema": "hawking.glm52.worker_heartbeat_auth.v1",
            "heartbeat": body,
        })
        claim_id = (
            f"worker-heartbeat:{checkpoint['event_count']:012d}:"
            f"{_sha256_bytes(self.instance_id.encode())[:12]}"
        )
        updated = self.controller.heartbeat(claim_id=claim_id, telemetry=body)
        _validate_authenticated_heartbeat(updated.get("heartbeat"), runtime=self.runtime)
        return updated

    def boundary(self, safe_point: str) -> tuple[str, dict[str, Any]]:
        """Poll controls and signals at one caller-declared safe point."""
        if safe_point not in gravity.WORKER_SAFE_POINTS:
            raise WorkerError(f"unknown worker safe point: {safe_point}")
        control = self.controls.poll(self.controller, safe_point=safe_point)
        signal_number = self.signal_latch.requested_signal
        directive = str(control["worker_directive"])
        if signal_number is not None:
            # Signal intent is never applied between boundary calls.  At this boundary it is
            # recorded as a clean safe stop without forging an operator acknowledgement.
            directive = "SIGNAL_STOP"
            control = {**control, "signal": signal_number, "worker_directive": directive}
        return directive, control

    def run(self, *, max_cycles: int | None = None, install_signals: bool = True) -> dict[str, Any]:
        """Run the persistent control loop.

        ``max_cycles`` is an in-process test seam.  Production CLI never supplies it.
        With no scientific dispatcher in this scaffold, every iteration is the
        ``BETWEEN_WINDOWS`` boundary and therefore cannot fabricate phase progress.
        """
        if max_cycles is not None and (
            isinstance(max_cycles, bool) or not isinstance(max_cycles, int) or max_cycles <= 0
        ):
            raise WorkerError("max_cycles test seam must be a positive integer")
        _lease_target_is_safe(self.controller)
        installed = False
        cycles = 0
        try:
            self.controller.acquire()
            _validate_held_lease(self.controller)
            # This is the second-stage production gate: static preflight can never
            # substitute for semantic replay.  It runs after lease acquisition but
            # before resume validation, signal installation, controls, or a heartbeat.
            # The loop repeats it at every BETWEEN_WINDOWS boundary, which is the only
            # location where a future reviewed dispatcher could be introduced.
            self._notification_audit_gate()
            checkpoint = self.controller.resume()
            _validate_authenticated_heartbeat(
                checkpoint.get("heartbeat"), runtime=self.runtime
            )
            if install_signals:
                self.signal_latch.install()
                installed = True
            while True:
                _validate_held_lease(self.controller)
                if cycles:
                    self._notification_audit_gate()
                directive, control = self.boundary("BETWEEN_WINDOWS")
                if directive in {"STOP", "SIGNAL_STOP"}:
                    checkpoint = self._heartbeat(
                        safe_point="BETWEEN_WINDOWS",
                        lifecycle="STOPPED_AT_SAFE_BOUNDARY",
                        control=control,
                    )
                    return {
                        "status": "STOPPED",
                        "reason": directive,
                        "cycles": cycles,
                        "checkpoint_seal_sha256": checkpoint["seal_sha256"],
                    }
                lifecycle = (
                    "PAUSED_AFTER_WINDOW"
                    if directive == "PAUSE"
                    else "CONTROL_PLANE_STANDBY_NO_SCIENTIFIC_DISPATCH"
                )
                checkpoint = self._heartbeat(
                    safe_point="BETWEEN_WINDOWS",
                    lifecycle=lifecycle,
                    control=control,
                )
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    return {
                        "status": "TEST_CYCLE_LIMIT",
                        "reason": "NO_PRODUCTION_LIMIT_CONFIGURED",
                        "cycles": cycles,
                        "checkpoint_seal_sha256": checkpoint["seal_sha256"],
                    }
                self.waiter(self.heartbeat_interval_seconds)
        finally:
            if installed:
                self.signal_latch.restore()
            self.controller.close()


def build_status(
    config: WorkerConfig,
    preflight_document: Mapping[str, Any],
    runtime: gravity.Runtime | None,
    *,
    under_caffeinate: bool = False,
) -> dict[str, Any]:
    caffeinate = inspect_caffeinate_parent() if under_caffeinate else {
        "ok": False,
        "reason": "status command is not the persistent caffeinated worker",
        "parent_pid": os.getppid(),
    }
    controller_status: dict[str, Any] | None = None
    control: dict[str, Any] | None = None
    applied: dict[str, Any] | None = None
    status_errors: list[str] = []
    if runtime is not None:
        try:
            controller_status = runtime.controller().status()
        except (state.StateError, OSError) as exc:
            status_errors.append(f"controller status: {exc}")
        try:
            control = gravity.OperatorControlJournal(runtime).replay()
            applied = gravity.WorkerControlAcknowledgements(runtime).replay()
        except (gravity.CliError, state.StateError, OSError) as exc:
            status_errors.append(f"operator control: {exc}")
    return {
        "schema": STATUS_SCHEMA,
        "status": (
            "READY_NOT_RUNNING"
            if preflight_document.get("production_start_authorized")
            else "BLOCKED"
        ),
        "worker_config_sha256": config.seal_sha256,
        "preflight": dict(preflight_document),
        "controller": controller_status,
        "operator_control": control,
        "worker_control_acknowledgements": applied,
        "caffeinate": caffeinate,
        "status_errors": status_errors,
        "session_limitations": SESSION_LIMITATIONS,
        "observed_at": utc_now(),
    }


def validate_launchd_plist(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Offline validation used by tests and deployment review; never loads the job."""
    candidate = Path(os.path.abspath(path))
    try:
        raw = state.TrustedArtifactStore(str(candidate.parent)).read_bytes(candidate.name)
        text = raw.decode("utf-8")
        value = plistlib.loads(raw)
    except (UnicodeDecodeError, OSError, state.StateError, plistlib.InvalidFileException) as exc:
        raise WorkerError(f"cannot read launchd plist: {exc}") from exc
    if not text.lstrip().startswith("<?xml"):
        raise WorkerError("launchd plist must be reviewable UTF-8 XML")
    xml_keys = re.findall(r"<key>\s*([^<]*?)\s*</key>", text)
    expected_xml_keys = set(PINNED_LAUNCHD_KEYS) | {"Crashed", "SuccessfulExit"}
    if len(xml_keys) != len(expected_xml_keys) or set(xml_keys) != expected_xml_keys:
        raise WorkerError("launchd plist XML contains duplicate or unapproved dictionary keys")
    if not isinstance(value, dict):
        raise WorkerError("launchd plist root is not a dictionary")
    if set(value) != PINNED_LAUNCHD_KEYS:
        raise WorkerError("launchd plist contains missing or unapproved keys")
    exact = {
        "Label": "com.hawking.glm52.gravity",
        "ProgramArguments": list(PINNED_LAUNCHD_ARGUMENTS),
        "WorkingDirectory": str(PINNED_WORKSPACE_ROOT),
        "RunAtLoad": True,
        "KeepAlive": {"Crashed": True, "SuccessfulExit": False},
        "ProcessType": "Background",
        "ThrottleInterval": 30,
        "StandardOutPath": str(PINNED_STDOUT),
        "StandardErrorPath": str(PINNED_STDERR),
        "Umask": 63,
    }

    def exact_typed(actual: Any, expected: Any) -> bool:
        if type(actual) is not type(expected):
            return False
        if isinstance(expected, dict):
            return set(actual) == set(expected) and all(
                exact_typed(actual[key], expected[key]) for key in expected
            )
        if isinstance(expected, list):
            return len(actual) == len(expected) and all(
                exact_typed(left, right) for left, right in zip(actual, expected)
            )
        return bool(actual == expected)

    if not exact_typed(value, exact):
        raise WorkerError(
            "launchd plist differs from the exact caffeinate/Python/worker/config allowlist"
        )
    if any("\x00" in item or "\n" in item or "\r" in item for item in value["ProgramArguments"]):
        raise WorkerError("launchd ProgramArguments contain control characters")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "GLM52_WORKER_CONFIG.json",
        help="sealed secret-free worker config",
    )
    parser.add_argument(
        "--under-caffeinate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("command", choices=("preflight", "status", "run"))
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    keychain: Any = None,
    evidence_keychain: Any = None,
    grounding_keychain: Any = None,
    execv: Callable[[str, Sequence[str]], Any] = os.execv,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_worker_config(args.config)
        telegram_keychain = keychain or gravity.telegram_module.MacOSKeychain()
        report, runtime = preflight(
            config,
            keychain=telegram_keychain,
            evidence_keychain=evidence_keychain,
            grounding_keychain=grounding_keychain,
        )
        if args.command == "preflight":
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
            return EXIT_OK if report["production_start_authorized"] else EXIT_PREFLIGHT_BLOCKED
        if args.command == "status":
            document = build_status(
                config, report, runtime, under_caffeinate=args.under_caffeinate
            )
            print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))
            return EXIT_OK if not document["status_errors"] else EXIT_ERROR
        if not report["production_start_authorized"] or runtime is None:
            print(json.dumps(report, sort_keys=True, allow_nan=False), file=sys.stderr)
            return EXIT_PREFLIGHT_BLOCKED
        if not args.under_caffeinate:
            if config.profile == OFFICIAL_CONFIG_PROFILE:
                validate_launchd_plist(PINNED_LAUNCHD_PLIST)
            argv_exec = caffeinate_exec_argv(config)
            execv(str(CAFFEINATE), argv_exec)
            raise WorkerError("caffeinate exec unexpectedly returned")
        if config.profile == OFFICIAL_CONFIG_PROFILE:
            validate_launchd_plist(PINNED_LAUNCHD_PLIST)
        expected_caffeinate_argv = caffeinate_exec_argv(config)
        caffeinate = inspect_caffeinate_parent(expected_argv=expected_caffeinate_argv)
        if not caffeinate["ok"]:
            raise WorkerError(f"caffeinate validation failed: {caffeinate['reason']}")
        notification_audit_readiness = _load_notification_audit_readiness(
            config,
            runtime,
            telegram_credential_status=
                gravity.telegram_module.credential_status(telegram_keychain),
        )
        result = PersistentWorker(
            runtime,
            heartbeat_interval_seconds=config.heartbeat_interval_seconds,
            notification_audit_readiness=notification_audit_readiness,
        ).run()
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return EXIT_OK
    except (
        WorkerError,
        gravity.CliError,
        state.StateError,
        Glm52Error,
        OSError,
    ) as exc:
        print(json.dumps({
            "status": "ERROR",
            "error": type(exc).__name__,
            "message": str(exc),
        }, sort_keys=True), file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())

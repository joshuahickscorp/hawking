#!/usr/bin/env python3.12
"""Descriptor-grounded source-window execution receipts for GLM-5.2.

This module is deliberately narrower than the campaign controller.  It derives
all window membership from the sealed expected-contract v3, observes files
through :mod:`glm52_grounding`, emits producer-authenticated phase receipts,
and owns the one destructive primitive needed by a streamed run: exact-manifest
source eviction.  It contains no network or download implementation.

The public live APIs accept paths and cryptographic expectations, never fact
providers.  Except for the explicitly named eviction-journal APIs, every
operation is read-only and returns an in-memory receipt.

These receipts are evidence, not execution authority.  Until the controller
persists their chain while holding its live lease and binds it to the frozen
post-autotune schedule, a shape-valid ``controller_anchor_sha256`` must never
authorize fetch, eviction, or campaign completeness by itself.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import hmac
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from glm52_common import Glm52Error, canonical, seal, utc_now, verify_sealed  # noqa: E402
from glm52_grounding import (  # noqa: E402
    ABSENCE_OBSERVATION_SCHEMA,
    FILE_OBSERVATION_SCHEMA,
    RESOURCE_SAMPLE_SCHEMA,
    GroundingError,
    ProducerAuthenticator,
    ResourceReservePolicy,
    TrustedFilesystemObserver,
    _normalized_absolute_root,
    _open_absolute_directory_chain,
    _relative_parts,
    _type_identity,
    _verify_absolute_directory_chain,
    verify_authenticated_observation,
)
from glm52_state import EvidenceAuthConfig, _validate_expected_contract  # noqa: E402


DECLARED_WINDOW_SCHEMA = "hawking.glm52.grounded_declared_window.v1"
TERMINAL_POLICY_SCHEMA = "hawking.glm52.window_terminal_policy.v1"
ARTIFACT_MANIFEST_SCHEMA = "hawking.glm52.window_artifact_manifest.v1"
TERMINAL_COVERAGE_SCHEMA = "hawking.glm52.window_terminal_coverage.v1"
EVICTION_EVENT_SCHEMA = "hawking.glm52.eviction_journal_event.v1"

PHASES = (
    "FETCH_INTENT",
    "FETCH_COMMITTED",
    "SOURCES_VERIFIED",
    "TEACHER_CAPTURED",
    "CANDIDATES_FIT",
    "CANDIDATES_PACKED",
    "FORWARD_COMPLETE",
    "WINDOW_SEALED",
    "EVICTION_COMMITTED",
)
PHASE_SCHEMAS = {
    phase: f"hawking.glm52.window_{phase.lower()}.receipt.v1" for phase in PHASES
}

ARTIFACT_PHASES = frozenset(
    {"TEACHER_CAPTURED", "CANDIDATES_FIT", "CANDIDATES_PACKED", "FORWARD_COMPLETE"}
)
ARTIFACT_KINDS = {
    "TEACHER_CAPTURED": frozenset({"TEACHER_EVIDENCE", "TEACHER_INDEX"}),
    "CANDIDATES_FIT": frozenset({"FIT_RESULT", "FIT_EVIDENCE"}),
    "CANDIDATES_PACKED": frozenset({"COMPACT_PAYLOAD", "PACKING_INDEX"}),
    "FORWARD_COMPLETE": frozenset({"FORWARD_METRICS", "FORWARD_EVIDENCE"}),
}
REQUIRED_ARTIFACT_KIND = {
    "TEACHER_CAPTURED": "TEACHER_EVIDENCE",
    "CANDIDATES_FIT": "FIT_RESULT",
    "CANDIDATES_PACKED": "COMPACT_PAYLOAD",
    "FORWARD_COMPLETE": "FORWARD_METRICS",
}

COMPACT_DISPOSITION = "COMPACT_PAYLOAD_LINEAGE"
PROTECTED_DISPOSITION = "PROTECTED_SOURCE_NATIVE_WITH_BILLED_BYTES"
OMITTED_DISPOSITION = "INTENTIONALLY_OMITTED_WITH_CAPABILITY_JUSTIFICATION"
TERMINAL_DISPOSITIONS = frozenset(
    {COMPACT_DISPOSITION, PROTECTED_DISPOSITION, OMITTED_DISPOSITION}
)

RESOURCE_MAX_AGE_SECONDS = 120
MAX_EVICTION_JOURNAL_BYTES = 64 * 1024 * 1024
NON_AUTHORITATIVE_STATUS = "NON_AUTHORITATIVE_EVIDENCE_ONLY"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)

_FILE_OBSERVATION_KEYS = frozenset(
    {
        "schema", "status", "observation_kind", "observed_at", "root_id",
        "root_device", "root_inode", "relative_path", "expected_size_bytes",
        "expected_sha256", "observed_sha256", "logical_bytes", "allocated_bytes",
        "device", "inode", "hard_link_count", "producer_key_identity_sha256",
        "producer_hmac_sha256", "seal_sha256",
    }
)
_ABSENCE_OBSERVATION_KEYS = frozenset(
    {
        "schema", "status", "observation_kind", "observed_at", "root_id",
        "root_device", "root_inode", "relative_path", "absent",
        "first_missing_component", "existing_parent", "parent_device",
        "parent_inode", "producer_key_identity_sha256", "producer_hmac_sha256",
        "seal_sha256",
    }
)
_RESOURCE_OBSERVATION_KEYS = frozenset(
    {
        "schema", "status", "observation_kind", "sampled_at", "platform",
        "root_id", "root_device", "root_inode", "disk_device", "disk_total_bytes",
        "disk_used_bytes", "disk_free_bytes", "total_ram_bytes",
        "available_ram_bytes", "total_swap_bytes", "used_swap_bytes",
        "memory_sample_source", "resource_policy", "operational_reserve_floor_bytes",
        "additional_reserved_bytes", "required_free_disk_bytes",
        "usable_raw_window_bytes", "disk_operational_reserve_ok",
        "available_ram_floor_ok", "swap_usage_ceiling_ok", "refusal_reasons",
        "producer_key_identity_sha256", "producer_hmac_sha256", "seal_sha256",
    }
)


class WindowExecutionError(Glm52Error):
    """A source-window receipt or exact-filesystem invariant failed."""


@dataclass(frozen=True, slots=True)
class WindowExecutionAuthenticators:
    """Independent authenticators for OS observations and scientific evidence."""

    grounding: ProducerAuthenticator
    evidence: EvidenceAuthConfig

    def __post_init__(self) -> None:
        if not isinstance(self.grounding, ProducerAuthenticator):
            raise WindowExecutionError("grounding ProducerAuthenticator is required")
        if not isinstance(self.evidence, EvidenceAuthConfig):
            raise WindowExecutionError("independent EvidenceAuthConfig is required")
        # Both classes intentionally keep key bytes private, but this boundary
        # is the one place that must prohibit cross-role key reuse.  Comparing
        # the in-memory material does not persist or expose it.
        if hmac.compare_digest(self.grounding._key, self.evidence._hmac_key):
            raise WindowExecutionError(
                "grounding and scientific evidence keys must be different"
            )


def _validate_auth_campaign(
    authenticator: WindowExecutionAuthenticators, contract: Mapping[str, Any]
) -> None:
    if not isinstance(authenticator, WindowExecutionAuthenticators):
        raise WindowExecutionError("WindowExecutionAuthenticators is required")
    if authenticator.evidence.campaign_id != contract["campaign_id"] \
            or authenticator.evidence.source_revision != contract["source_revision"]:
        raise WindowExecutionError("evidence authenticator campaign identity mismatch")


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise WindowExecutionError(f"{label} must be exactly 64 lowercase hex digits")
    return str(value)


def _require_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise WindowExecutionError(f"{label} must be a non-empty text value")
    return value


def _normalized_relative(value: object, label: str) -> str:
    try:
        normalized, _ = _relative_parts(value)
    except GroundingError as exc:
        raise WindowExecutionError(f"{label}: {exc}") from exc
    return normalized


def _contract(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        contract = _validate_expected_contract(copy.deepcopy(dict(value)))
    except (Glm52Error, TypeError, ValueError) as exc:
        raise WindowExecutionError(f"invalid expected-contract v3: {exc}") from exc
    # expected-contract v3 validates membership but historically accepted any
    # non-empty shard text.  A filesystem executor must additionally reject
    # absolute/traversing/non-normalized source names before deriving a window.
    for item in contract["source"]["shards"]:
        _normalized_relative(item["path"], "expected source shard path")
    return contract


def _source_index(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["path"]): copy.deepcopy(dict(item))
        for item in contract["source"]["shards"]
    }


def source_root_id(expected_contract: Mapping[str, Any]) -> str:
    value = _contract(expected_contract)
    return f"glm52-source:{value['campaign_id']}:{value['seal_sha256']}"


def artifact_root_id(expected_contract: Mapping[str, Any]) -> str:
    value = _contract(expected_contract)
    return f"glm52-artifacts:{value['campaign_id']}:{value['seal_sha256']}"


def derive_declared_window(
    expected_contract: Mapping[str, Any], schedule_index: int
) -> dict[str, Any]:
    """Return one sealed declaration derived only from its authoritative index.

    There is intentionally no ``window_id``, shard-list, tensor-list, or byte
    argument.  A caller can select an index, but cannot assert any schedule fact.
    """

    contract = _contract(expected_contract)
    if type(schedule_index) is not int or not 0 <= schedule_index < len(
        contract["window_schedule"]
    ):
        raise WindowExecutionError("schedule_index is outside the expected contract")
    scheduled = copy.deepcopy(contract["window_schedule"][schedule_index])
    sources = _source_index(contract)
    identities = [copy.deepcopy(sources[path]) for path in scheduled["source_shards"]]
    return seal(
        {
            "schema": DECLARED_WINDOW_SCHEMA,
            "status": "DECLARED",
            "campaign_id": contract["campaign_id"],
            "source_revision": contract["source_revision"],
            "expected_contract_sha256": contract["seal_sha256"],
            "schedule_index": schedule_index,
            "window_id": scheduled["window_id"],
            "source_shards": identities,
            "carry_in_shards": scheduled["carry_in_shards"],
            "new_fetch_shards": scheduled["new_fetch_shards"],
            "refetch_shards": scheduled["refetch_shards"],
            "carry_out_shards": scheduled["carry_out_shards"],
            "evict_shards": scheduled["evict_shards"],
            "tensor_set": scheduled["tensor_set"],
        }
    )


def validate_declared_window(
    value: Mapping[str, Any], expected_contract: Mapping[str, Any], schedule_index: int
) -> dict[str, Any]:
    expected = derive_declared_window(expected_contract, schedule_index)
    candidate = copy.deepcopy(dict(value))
    try:
        verify_sealed(candidate, label="declared window")
    except Glm52Error as exc:
        raise WindowExecutionError(str(exc)) from exc
    if candidate != expected:
        raise WindowExecutionError(
            "declared window differs from authoritative expected-contract schedule"
        )
    return candidate


def _authenticated(
    body: Mapping[str, Any], authenticator: WindowExecutionAuthenticators
) -> dict[str, Any]:
    if not isinstance(authenticator, WindowExecutionAuthenticators):
        raise WindowExecutionError("independent execution authenticators are required")
    if "producer_hmac_sha256" in body or "seal_sha256" in body:
        raise WindowExecutionError("body contains generated authentication fields")
    evidence_auth = authenticator.evidence
    if body.get("campaign_id") != evidence_auth.campaign_id \
            or body.get("source_revision") != evidence_auth.source_revision:
        raise WindowExecutionError("scientific receipt authenticator identity mismatch")
    unsigned = {
        **copy.deepcopy(dict(body)),
        "producer_key_identity_sha256": evidence_auth._key_material_identity(),
    }
    return seal(
        {
            **unsigned,
            "producer_hmac_sha256": evidence_auth.authenticate(
                {
                    "schema": "hawking.glm52.window_execution_producer_auth.v1",
                    "artifact": unsigned,
                }
            ),
        }
    )


def _verify_authenticated(
    value: Mapping[str, Any], authenticator: WindowExecutionAuthenticators, *, label: str
) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(value))
    try:
        verify_sealed(candidate, label=label)
    except Glm52Error as exc:
        raise WindowExecutionError(str(exc)) from exc
    if not isinstance(authenticator, WindowExecutionAuthenticators):
        raise WindowExecutionError("independent execution authenticators are required")
    evidence_auth = authenticator.evidence
    if candidate.get("campaign_id") != evidence_auth.campaign_id \
            or candidate.get("source_revision") != evidence_auth.source_revision:
        raise WindowExecutionError(f"{label} evidence-auth campaign identity mismatch")
    if candidate.get("producer_key_identity_sha256") != \
            evidence_auth._key_material_identity():
        raise WindowExecutionError(f"{label} producer identity mismatch")
    unsigned = {
        key: item
        for key, item in candidate.items()
        if key not in {"seal_sha256", "producer_hmac_sha256"}
    }
    if not evidence_auth.verify(
        {
            "schema": "hawking.glm52.window_execution_producer_auth.v1",
            "artifact": unsigned,
        },
        candidate.get("producer_hmac_sha256"),
    ):
        raise WindowExecutionError(f"{label} producer HMAC mismatch")
    return candidate


def make_terminal_policy(
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    *,
    entries: Sequence[Mapping[str, Any]],
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    """Pre-register exactly one permitted terminal route for each tensor."""

    contract = _contract(expected_contract)
    declared = derive_declared_window(contract, schedule_index)
    body = {
        "schema": TERMINAL_POLICY_SCHEMA,
        "status": "PREREGISTERED",
        "campaign_id": contract["campaign_id"],
        "source_revision": contract["source_revision"],
        "expected_contract_sha256": contract["seal_sha256"],
        "window_id": declared["window_id"],
        "schedule_index": schedule_index,
        "schedule_entry_sha256": declared["seal_sha256"],
        "entries": copy.deepcopy(list(entries)),
        "created_at": utc_now(),
    }
    policy = _authenticated(body, authenticator)
    return validate_terminal_policy(policy, contract, schedule_index, authenticator)


def validate_terminal_policy(
    value: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    contract = _contract(expected_contract)
    declared = derive_declared_window(contract, schedule_index)
    policy = _verify_authenticated(value, authenticator, label="terminal policy")
    required = {
        "schema",
        "status",
        "campaign_id",
        "source_revision",
        "expected_contract_sha256",
        "window_id",
        "schedule_index",
        "schedule_entry_sha256",
        "entries",
        "created_at",
        "producer_key_identity_sha256",
        "producer_hmac_sha256",
        "seal_sha256",
    }
    if set(policy) != required:
        raise WindowExecutionError("terminal policy fields are not exact")
    expected_identity = {
        "schema": TERMINAL_POLICY_SCHEMA,
        "status": "PREREGISTERED",
        "campaign_id": contract["campaign_id"],
        "source_revision": contract["source_revision"],
        "expected_contract_sha256": contract["seal_sha256"],
        "window_id": declared["window_id"],
        "schedule_index": schedule_index,
        "schedule_entry_sha256": declared["seal_sha256"],
    }
    for key, expected in expected_identity.items():
        if policy.get(key) != expected:
            raise WindowExecutionError(f"terminal policy {key} mismatch")
    _require_name(policy.get("created_at"), "terminal policy created_at")
    raw_entries = policy.get("entries")
    if not isinstance(raw_entries, list):
        raise WindowExecutionError("terminal policy entries must be a list")
    expected_tensors = list(declared["tensor_set"])
    if len(raw_entries) != len(expected_tensors):
        raise WindowExecutionError("terminal policy must cover every scheduled tensor once")
    seen: set[str] = set()
    source_paths = {item["path"] for item in declared["source_shards"]}
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise WindowExecutionError("terminal policy entry must be an object")
        tensor = _require_name(entry.get("tensor_name"), "terminal policy tensor_name")
        if tensor in seen or tensor not in expected_tensors:
            raise WindowExecutionError("terminal policy tensor coverage is not exact")
        seen.add(tensor)
        disposition = entry.get("disposition")
        if disposition not in TERMINAL_DISPOSITIONS:
            raise WindowExecutionError(
                "terminal policy disposition is unknown; NON_MODEL_FILE is forbidden"
            )
        if disposition == COMPACT_DISPOSITION:
            if set(entry) != {"tensor_name", "disposition", "payload_class"}:
                raise WindowExecutionError("compact terminal policy fields are not exact")
            if entry.get("payload_class") not in {"CORE", "OPTIONAL_MTP"}:
                raise WindowExecutionError("compact payload_class is invalid")
        elif disposition == PROTECTED_DISPOSITION:
            if set(entry) != {
                "tensor_name",
                "disposition",
                "native_source_path",
                "billed_bytes",
                "protection_justification",
            }:
                raise WindowExecutionError("protected terminal policy fields are not exact")
            if entry.get("native_source_path") not in source_paths:
                raise WindowExecutionError("protected tensor source is outside this window")
            billed = entry.get("billed_bytes")
            if type(billed) is not int or billed <= 0:
                raise WindowExecutionError("protected billed_bytes must be positive")
            _require_name(
                entry.get("protection_justification"), "protection_justification"
            )
        else:
            if set(entry) != {
                "tensor_name",
                "disposition",
                "capability_justification",
                "justification_evidence_sha256",
            }:
                raise WindowExecutionError("omission terminal policy fields are not exact")
            _require_name(
                entry.get("capability_justification"), "capability_justification"
            )
            _require_sha256(
                entry.get("justification_evidence_sha256"),
                "justification_evidence_sha256",
            )
    if seen != set(expected_tensors):
        raise WindowExecutionError("terminal policy has a tensor coverage gap")
    return policy


def make_artifact_manifest(
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    *,
    phase: str,
    artifact_directory: str,
    files: Sequence[Mapping[str, Any]],
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    """Create the expected exact-file manifest for one artifact phase."""

    contract = _contract(expected_contract)
    declared = derive_declared_window(contract, schedule_index)
    manifest = _authenticated(
        {
            "schema": ARTIFACT_MANIFEST_SCHEMA,
            "status": "EXPECTED",
            "campaign_id": contract["campaign_id"],
            "source_revision": contract["source_revision"],
            "expected_contract_sha256": contract["seal_sha256"],
            "window_id": declared["window_id"],
            "schedule_index": schedule_index,
            "schedule_entry_sha256": declared["seal_sha256"],
            "phase": phase,
            "artifact_directory": artifact_directory,
            "files": copy.deepcopy(list(files)),
            "created_at": utc_now(),
        },
        authenticator,
    )
    return validate_artifact_manifest(
        manifest, contract, schedule_index, phase, authenticator
    )


def validate_artifact_manifest(
    value: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    phase: str,
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    if phase not in ARTIFACT_PHASES:
        raise WindowExecutionError("artifact manifest phase is not artifact-producing")
    contract = _contract(expected_contract)
    declared = derive_declared_window(contract, schedule_index)
    manifest = _verify_authenticated(value, authenticator, label="artifact manifest")
    required = {
        "schema",
        "status",
        "campaign_id",
        "source_revision",
        "expected_contract_sha256",
        "window_id",
        "schedule_index",
        "schedule_entry_sha256",
        "phase",
        "artifact_directory",
        "files",
        "created_at",
        "producer_key_identity_sha256",
        "producer_hmac_sha256",
        "seal_sha256",
    }
    if set(manifest) != required:
        raise WindowExecutionError("artifact manifest fields are not exact")
    expected_identity = {
        "schema": ARTIFACT_MANIFEST_SCHEMA,
        "status": "EXPECTED",
        "campaign_id": contract["campaign_id"],
        "source_revision": contract["source_revision"],
        "expected_contract_sha256": contract["seal_sha256"],
        "window_id": declared["window_id"],
        "schedule_index": schedule_index,
        "schedule_entry_sha256": declared["seal_sha256"],
        "phase": phase,
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise WindowExecutionError(f"artifact manifest {key} mismatch")
    _normalized_relative(manifest.get("artifact_directory"), "artifact_directory")
    _require_name(manifest.get("created_at"), "artifact manifest created_at")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise WindowExecutionError("artifact manifest files must be non-empty")
    paths: set[str] = set()
    kinds: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {
            "relative_path",
            "logical_bytes",
            "sha256",
            "artifact_kind",
        }:
            raise WindowExecutionError("artifact manifest file fields are not exact")
        path = _normalized_relative(item.get("relative_path"), "artifact relative_path")
        if path in paths:
            raise WindowExecutionError("artifact manifest repeats a file path")
        paths.add(path)
        size = item.get("logical_bytes")
        if type(size) is not int or size < 0:
            raise WindowExecutionError("artifact logical_bytes must be non-negative")
        _require_sha256(item.get("sha256"), "artifact file sha256")
        kind = item.get("artifact_kind")
        if kind not in ARTIFACT_KINDS[phase]:
            raise WindowExecutionError(f"artifact kind is invalid for {phase}")
        kinds.add(str(kind))
    if REQUIRED_ARTIFACT_KIND[phase] not in kinds:
        raise WindowExecutionError(f"artifact manifest lacks {REQUIRED_ARTIFACT_KIND[phase]}")
    return manifest


def _open_relative_directory_chain(
    root_fd: int, relative_directory: str
) -> tuple[list[int], list[tuple[int, str, tuple[int, int, int]]]]:
    _, parts = _relative_parts(relative_directory)
    fds = [os.dup(root_fd)]
    links: list[tuple[int, str, tuple[int, int, int]]] = []
    try:
        for component in parts:
            parent_fd = fds[-1]
            named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(named.st_mode):
                raise WindowExecutionError("manifest directory may not contain a symlink")
            if not stat.S_ISDIR(named.st_mode):
                raise WindowExecutionError("manifest directory component is not a directory")
            child = os.open(
                component,
                os.O_RDONLY | _CLOEXEC | _NOFOLLOW | _DIRECTORY,
                dir_fd=parent_fd,
            )
            opened = os.fstat(child)
            if _type_identity(named) != _type_identity(opened):
                os.close(child)
                raise WindowExecutionError("manifest directory changed while opening")
            links.append((parent_fd, component, _type_identity(opened)))
            fds.append(child)
        return fds, links
    except BaseException:
        for fd in reversed(fds):
            os.close(fd)
        raise


def _verify_relative_directory_chain(
    fds: Sequence[int], links: Sequence[tuple[int, str, tuple[int, int, int]]]
) -> None:
    if len(fds) != len(links) + 1:
        raise WindowExecutionError("manifest directory descriptor chain malformed")
    for index, (_, component, expected) in enumerate(links):
        named = os.stat(component, dir_fd=fds[index], follow_symlinks=False)
        opened = os.fstat(fds[index + 1])
        if stat.S_ISLNK(named.st_mode) or _type_identity(named) != expected \
                or _type_identity(opened) != expected:
            raise WindowExecutionError("manifest directory identity changed")


def _walk_regular_files(
    directory_fd: int, prefix: str = ""
) -> dict[str, tuple[int, int, int, int, int, int]]:
    """Descriptor-relative exact regular-file inventory; symlinks fail closed."""

    result: dict[str, tuple[int, int, int, int, int, int]] = {}
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as exc:
        raise WindowExecutionError(f"cannot list artifact directory: {exc}") from exc
    for name in names:
        if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name:
            raise WindowExecutionError("artifact directory returned an invalid name")
        path = f"{prefix}/{name}" if prefix else name
        try:
            named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise WindowExecutionError(f"cannot inspect artifact entry {path!r}: {exc}") from exc
        if stat.S_ISLNK(named.st_mode):
            raise WindowExecutionError(f"artifact tree contains a symlink: {path}")
        if stat.S_ISDIR(named.st_mode):
            child = os.open(
                name,
                os.O_RDONLY | _CLOEXEC | _NOFOLLOW | _DIRECTORY,
                dir_fd=directory_fd,
            )
            try:
                opened = os.fstat(child)
                if _type_identity(named) != _type_identity(opened):
                    raise WindowExecutionError("artifact directory changed while opening")
                result.update(_walk_regular_files(child, path))
                named_post = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if _type_identity(named_post) != _type_identity(opened):
                    raise WindowExecutionError("artifact directory identity changed")
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(named.st_mode):
            raise WindowExecutionError(f"artifact tree contains a non-regular file: {path}")
        if int(named.st_nlink) != 1:
            raise WindowExecutionError(f"artifact file must have one hard link: {path}")
        result[path] = (
            int(named.st_dev),
            int(named.st_ino),
            int(named.st_size),
            int(named.st_nlink),
            int(named.st_mtime_ns),
            int(named.st_ctime_ns),
        )
    if sorted(os.listdir(directory_fd)) != names:
        raise WindowExecutionError("artifact directory membership changed during inventory")
    return result


def _inventory_artifact_tree(
    artifact_root: str | os.PathLike[str], artifact_directory: str
) -> dict[str, tuple[int, int, int, int, int, int]]:
    try:
        root_path = _normalized_absolute_root(artifact_root)
    except GroundingError as exc:
        raise WindowExecutionError(str(exc)) from exc
    try:
        root_fds, root_links, root_stat = _open_absolute_directory_chain(root_path)
    except GroundingError as exc:
        raise WindowExecutionError(str(exc)) from exc
    relative_fds: list[int] = []
    try:
        relative_fds, relative_links = _open_relative_directory_chain(
            root_fds[-1], artifact_directory
        )
        inventory = _walk_regular_files(relative_fds[-1])
        _verify_relative_directory_chain(relative_fds, relative_links)
        _verify_absolute_directory_chain(root_fds, root_links, root_stat)
        return inventory
    except OSError as exc:
        raise WindowExecutionError(f"artifact inventory failed: {exc}") from exc
    finally:
        for fd in reversed(relative_fds):
            os.close(fd)
        for fd in reversed(root_fds):
            os.close(fd)


def _ground_artifact_manifest(
    manifest: Mapping[str, Any],
    *,
    artifact_root: str | os.PathLike[str],
    authenticator: WindowExecutionAuthenticators,
) -> list[dict[str, Any]]:
    directory = str(manifest["artifact_directory"])
    first_inventory = _inventory_artifact_tree(artifact_root, directory)
    expected_paths = [str(item["relative_path"]) for item in manifest["files"]]
    if set(first_inventory) != set(expected_paths) or len(first_inventory) != len(expected_paths):
        missing = sorted(set(expected_paths) - set(first_inventory))
        extra = sorted(set(first_inventory) - set(expected_paths))
        raise WindowExecutionError(
            f"artifact directory differs from exact manifest; missing={missing}, extra={extra}"
        )
    observer = TrustedFilesystemObserver(
        artifact_root,
        root_id=artifact_root_id_from_manifest(manifest),
        authenticator=authenticator.grounding,
    )
    observations: list[dict[str, Any]] = []
    for item in manifest["files"]:
        full_path = f"{directory}/{item['relative_path']}"
        observations.append(
            observer.observe_regular_file(
                full_path,
                expected_size_bytes=int(item["logical_bytes"]),
                expected_sha256=str(item["sha256"]),
            )
        )
    second_inventory = _inventory_artifact_tree(artifact_root, directory)
    if second_inventory != first_inventory:
        raise WindowExecutionError("artifact tree changed while it was grounded")
    return observations


def artifact_root_id_from_manifest(manifest: Mapping[str, Any]) -> str:
    return (
        f"glm52-artifacts:{manifest['campaign_id']}:"
        f"{manifest['expected_contract_sha256']}"
    )


def _source_roles(declared: Mapping[str, Any]) -> list[dict[str, str]]:
    new_fetch = set(declared["new_fetch_shards"])
    refetch = set(declared["refetch_shards"])
    carry = set(declared["carry_in_shards"])
    result = []
    for source in declared["source_shards"]:
        path = str(source["path"])
        if path in new_fetch:
            role = "NEW_FETCH"
        elif path in refetch:
            role = "REFETCH"
        elif path in carry:
            role = "CARRY_IN_RESIDENT"
        else:  # the v3 validator makes this unreachable
            raise WindowExecutionError("scheduled source lacks an acquisition role")
        result.append({"path": path, "acquisition_role": role, "xet_hash": source["xet_hash"]})
    return result


def _ground_sources(
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    *,
    source_root: str | os.PathLike[str],
    authenticator: WindowExecutionAuthenticators,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    declared = derive_declared_window(expected_contract, schedule_index)
    observer = TrustedFilesystemObserver(
        source_root,
        root_id=source_root_id(expected_contract),
        authenticator=authenticator.grounding,
    )
    observations = []
    for source in declared["source_shards"]:
        observations.append(
            observer.observe_regular_file(
                str(source["path"]),
                expected_size_bytes=int(source["logical_bytes"]),
                expected_sha256=str(source["lfs_sha256"]),
            )
        )
    return observations, _source_roles(declared)


def _verify_grounded_sources(
    evidence: Mapping[str, Any],
    contract: Mapping[str, Any],
    schedule_index: int,
    authenticator: WindowExecutionAuthenticators,
    *,
    expected_root_identity: tuple[int, int] | None = None,
) -> None:
    if set(evidence) != {"source_observations", "xet_acquisition_identities"}:
        raise WindowExecutionError("source evidence fields are not exact")
    declared = derive_declared_window(contract, schedule_index)
    observations = evidence.get("source_observations")
    roles = evidence.get("xet_acquisition_identities")
    if not isinstance(observations, list) or len(observations) != len(declared["source_shards"]):
        raise WindowExecutionError("source observations do not cover the full window")
    if roles != _source_roles(declared):
        raise WindowExecutionError("Xet acquisition identities differ from the contract")
    roots: set[tuple[int, int]] = set()
    for source, raw in zip(declared["source_shards"], observations):
        try:
            observation = verify_authenticated_observation(raw, authenticator.grounding)
        except GroundingError as exc:
            raise WindowExecutionError(f"source observation invalid: {exc}") from exc
        if observation.get("schema") != FILE_OBSERVATION_SCHEMA \
                or observation.get("status") != "PASS" \
                or observation.get("observation_kind") != "contained_regular_file" \
                or set(observation) != _FILE_OBSERVATION_KEYS:
            raise WindowExecutionError("source observation is not a passing regular file")
        expected = {
            "root_id": source_root_id(contract),
            "relative_path": source["path"],
            "expected_size_bytes": source["logical_bytes"],
            "expected_sha256": source["lfs_sha256"],
            "observed_sha256": source["lfs_sha256"],
            "logical_bytes": source["logical_bytes"],
            "hard_link_count": 1,
        }
        for key, wanted in expected.items():
            if observation.get(key) != wanted:
                raise WindowExecutionError(f"source observation {source['path']} {key} mismatch")
        roots.add((int(observation["root_device"]), int(observation["root_inode"])))
    if len(roots) != 1:
        raise WindowExecutionError("source observations do not share one trusted root")
    if expected_root_identity is not None and next(iter(roots)) != expected_root_identity:
        raise WindowExecutionError("source observations differ from resource-sampled root")


def _validate_resource_receipt(
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    authenticator: WindowExecutionAuthenticators,
    expected_policy: ResourceReservePolicy,
    *,
    as_of: str | None = None,
) -> dict[str, Any]:
    if not isinstance(expected_policy, ResourceReservePolicy):
        raise WindowExecutionError("exact ResourceReservePolicy is required")
    try:
        value = verify_authenticated_observation(
            receipt,
            authenticator.grounding,
            now=as_of or utc_now(),
            max_age_seconds=RESOURCE_MAX_AGE_SECONDS,
        )
    except GroundingError as exc:
        raise WindowExecutionError(f"live resource policy receipt invalid: {exc}") from exc
    if value.get("schema") != RESOURCE_SAMPLE_SCHEMA or value.get("status") != "PASS":
        raise WindowExecutionError("live resource policy did not pass")
    if set(value) != _RESOURCE_OBSERVATION_KEYS \
            or value.get("observation_kind") != "live_disk_ram_swap_resources":
        raise WindowExecutionError("resource receipt semantic fields are not exact")
    if value.get("root_id") != source_root_id(contract):
        raise WindowExecutionError("resource receipt was sampled for a different source root")
    if value.get("resource_policy") != expected_policy.as_dict():
        raise WindowExecutionError("resource receipt policy differs from the required policy")
    if value.get("required_free_disk_bytes") != expected_policy.required_free_disk_bytes \
            or value.get("operational_reserve_floor_bytes") != \
            expected_policy.operational_reserve_floor_bytes \
            or value.get("additional_reserved_bytes") != \
            expected_policy.additional_reserved_bytes:
        raise WindowExecutionError("resource receipt derived floors differ from policy")
    for key in (
        "disk_operational_reserve_ok",
        "available_ram_floor_ok",
        "swap_usage_ceiling_ok",
    ):
        if value.get(key) is not True:
            raise WindowExecutionError(f"resource receipt has a failed policy predicate: {key}")
    return value


_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "phase",
        "status",
        "authority_status",
        "campaign_stop_condition_eligible",
        "destructive_authority_eligible",
        "campaign_id",
        "source_revision",
        "expected_contract_sha256",
        "window_id",
        "schedule_index",
        "schedule_entry_sha256",
        "controller_anchor_sha256",
        "previous_receipt_sha256",
        "terminal_policy_sha256",
        "artifact_manifest_sha256s",
        "source_root_identity",
        "artifact_root_identity",
        "evidence",
        "evidence_sha256",
        "created_at",
        "producer_key_identity_sha256",
        "producer_hmac_sha256",
        "seal_sha256",
    }
)


def _verify_receipt_envelope(
    value: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    authenticator: WindowExecutionAuthenticators,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _contract(expected_contract)
    receipt = _verify_authenticated(value, authenticator, label="window phase receipt")
    if set(receipt) != _RECEIPT_KEYS:
        raise WindowExecutionError("window phase receipt fields are not exact")
    phase = receipt.get("phase")
    if phase not in PHASES or receipt.get("schema") != PHASE_SCHEMAS[phase]:
        raise WindowExecutionError("window phase receipt schema/phase mismatch")
    if receipt.get("status") != "EVIDENCE_ONLY" \
            or receipt.get("authority_status") != NON_AUTHORITATIVE_STATUS \
            or receipt.get("campaign_stop_condition_eligible") is not False \
            or receipt.get("destructive_authority_eligible") is not False:
        raise WindowExecutionError(
            "window phase receipt falsely claims semantic or destructive authority"
        )
    schedule_index = receipt.get("schedule_index")
    declared = derive_declared_window(contract, schedule_index)
    expected_identity = {
        "campaign_id": contract["campaign_id"],
        "source_revision": contract["source_revision"],
        "expected_contract_sha256": contract["seal_sha256"],
        "window_id": declared["window_id"],
        "schedule_entry_sha256": declared["seal_sha256"],
    }
    for key, expected in expected_identity.items():
        if receipt.get(key) != expected:
            raise WindowExecutionError(f"window receipt {key} mismatch")
    _require_sha256(receipt.get("controller_anchor_sha256"), "controller anchor")
    previous = receipt.get("previous_receipt_sha256")
    if phase == "FETCH_INTENT":
        if previous is not None:
            raise WindowExecutionError("FETCH_INTENT may not have a previous receipt")
    else:
        _require_sha256(previous, "previous receipt")
    _require_sha256(receipt.get("terminal_policy_sha256"), "terminal policy seal")
    manifests = receipt.get("artifact_manifest_sha256s")
    if not isinstance(manifests, dict) or any(
        key not in ARTIFACT_PHASES or not _is_sha256(item)
        for key, item in manifests.items()
    ):
        raise WindowExecutionError("artifact manifest seal registry is invalid")
    for root_label in ("source_root_identity", "artifact_root_identity"):
        identity = receipt.get(root_label)
        if identity is not None and (
            not isinstance(identity, dict)
            or set(identity) != {"device", "inode"}
            or any(type(identity[key]) is not int or identity[key] < 0 for key in identity)
        ):
            raise WindowExecutionError(f"{root_label} is invalid")
    evidence = receipt.get("evidence")
    if not isinstance(evidence, dict) or receipt.get("evidence_sha256") != _sha(evidence):
        raise WindowExecutionError("window receipt evidence hash mismatch")
    _require_name(receipt.get("created_at"), "window receipt created_at")
    return receipt, contract


def _make_receipt(
    phase: str,
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    *,
    controller_anchor_sha256: str,
    previous_receipt: Mapping[str, Any] | None,
    terminal_policy_sha256: str,
    artifact_manifest_sha256s: Mapping[str, str],
    evidence: Mapping[str, Any],
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    contract = _contract(expected_contract)
    declared = derive_declared_window(contract, schedule_index)
    _require_sha256(controller_anchor_sha256, "controller anchor")
    _require_sha256(terminal_policy_sha256, "terminal policy seal")
    previous_sha = None
    previous = None
    if previous_receipt is not None:
        previous, _ = _verify_receipt_envelope(previous_receipt, contract, authenticator)
        previous_sha = previous["seal_sha256"]
    source_identity = copy.deepcopy(
        previous["source_root_identity"] if previous is not None else None
    )
    artifact_identity = copy.deepcopy(
        previous["artifact_root_identity"] if previous is not None else None
    )

    def observed_root(items: object, label: str) -> dict[str, int]:
        if not isinstance(items, list) or not items or not isinstance(items[0], dict):
            raise WindowExecutionError(f"{label} lacks a grounded root observation")
        return {
            "device": int(items[0]["root_device"]),
            "inode": int(items[0]["root_inode"]),
        }

    candidate_source = None
    candidate_artifact = None
    if phase == "FETCH_INTENT":
        resource = evidence.get("resource_policy_receipt")
        if not isinstance(resource, dict):
            raise WindowExecutionError("FETCH_INTENT lacks a resource root")
        candidate_source = {
            "device": int(resource["root_device"]),
            "inode": int(resource["root_inode"]),
        }
    elif phase in {"FETCH_COMMITTED", "SOURCES_VERIFIED"}:
        candidate_source = observed_root(evidence.get("source_observations"), phase)
    elif phase in ARTIFACT_PHASES:
        candidate_artifact = observed_root(evidence.get("file_observations"), phase)
    elif phase == "WINDOW_SEALED":
        candidate_source = observed_root(
            evidence.get("native_source_observations"), phase
        )
        candidate_artifact = observed_root(
            evidence.get("compact_file_observations"), phase
        )
    elif phase == "EVICTION_COMMITTED":
        candidate_source = observed_root(evidence.get("absence_observations"), phase)
    if source_identity is None:
        source_identity = candidate_source
    elif candidate_source is not None and candidate_source != source_identity:
        raise WindowExecutionError("source filesystem root changed across phase receipts")
    if artifact_identity is None:
        artifact_identity = candidate_artifact
    elif candidate_artifact is not None and candidate_artifact != artifact_identity:
        raise WindowExecutionError("artifact filesystem root changed across phase receipts")
    receipt = _authenticated(
        {
            "schema": PHASE_SCHEMAS[phase],
            "phase": phase,
            "status": "EVIDENCE_ONLY",
            "authority_status": NON_AUTHORITATIVE_STATUS,
            "campaign_stop_condition_eligible": False,
            "destructive_authority_eligible": False,
            "campaign_id": contract["campaign_id"],
            "source_revision": contract["source_revision"],
            "expected_contract_sha256": contract["seal_sha256"],
            "window_id": declared["window_id"],
            "schedule_index": schedule_index,
            "schedule_entry_sha256": declared["seal_sha256"],
            "controller_anchor_sha256": controller_anchor_sha256,
            "previous_receipt_sha256": previous_sha,
            "terminal_policy_sha256": terminal_policy_sha256,
            "artifact_manifest_sha256s": copy.deepcopy(dict(artifact_manifest_sha256s)),
            "source_root_identity": source_identity,
            "artifact_root_identity": artifact_identity,
            "evidence": copy.deepcopy(dict(evidence)),
            "evidence_sha256": _sha(evidence),
            "created_at": utc_now(),
        },
        authenticator,
    )
    return receipt


def _verify_artifact_observations(
    evidence: Mapping[str, Any],
    contract: Mapping[str, Any],
    schedule_index: int,
    phase: str,
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    if set(evidence) != {"artifact_manifest", "file_observations"}:
        raise WindowExecutionError("artifact phase evidence fields are not exact")
    manifest = validate_artifact_manifest(
        evidence["artifact_manifest"], contract, schedule_index, phase, authenticator
    )
    observations = evidence.get("file_observations")
    if not isinstance(observations, list) or len(observations) != len(manifest["files"]):
        raise WindowExecutionError("artifact observations do not exactly cover the manifest")
    roots: set[tuple[int, int]] = set()
    for item, raw in zip(manifest["files"], observations):
        try:
            observation = verify_authenticated_observation(raw, authenticator.grounding)
        except GroundingError as exc:
            raise WindowExecutionError(f"artifact observation invalid: {exc}") from exc
        expected = {
            "schema": FILE_OBSERVATION_SCHEMA,
            "status": "PASS",
            "observation_kind": "contained_regular_file",
            "root_id": artifact_root_id(contract),
            "relative_path": (
                f"{manifest['artifact_directory']}/{item['relative_path']}"
            ),
            "expected_size_bytes": item["logical_bytes"],
            "expected_sha256": item["sha256"],
            "observed_sha256": item["sha256"],
            "logical_bytes": item["logical_bytes"],
            "hard_link_count": 1,
        }
        for key, wanted in expected.items():
            if observation.get(key) != wanted:
                raise WindowExecutionError(
                    f"artifact observation {item['relative_path']} {key} mismatch"
                )
        if set(observation) != _FILE_OBSERVATION_KEYS:
            raise WindowExecutionError("artifact observation fields are not exact")
        roots.add((int(observation["root_device"]), int(observation["root_inode"])))
    if len(roots) != 1:
        raise WindowExecutionError("artifact observations do not share one trusted root")
    return manifest


def make_terminal_coverage_manifest(
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    *,
    terminal_policy: Mapping[str, Any],
    packed_manifest: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    contract = _contract(expected_contract)
    declared = derive_declared_window(contract, schedule_index)
    policy = validate_terminal_policy(
        terminal_policy, contract, schedule_index, authenticator
    )
    packed = validate_artifact_manifest(
        packed_manifest,
        contract,
        schedule_index,
        "CANDIDATES_PACKED",
        authenticator,
    )
    manifest = _authenticated(
        {
            "schema": TERMINAL_COVERAGE_SCHEMA,
            "status": "TERMINAL",
            "campaign_id": contract["campaign_id"],
            "source_revision": contract["source_revision"],
            "expected_contract_sha256": contract["seal_sha256"],
            "window_id": declared["window_id"],
            "schedule_index": schedule_index,
            "schedule_entry_sha256": declared["seal_sha256"],
            "terminal_policy_sha256": policy["seal_sha256"],
            "packed_manifest_sha256": packed["seal_sha256"],
            "entries": copy.deepcopy(list(entries)),
            "created_at": utc_now(),
        },
        authenticator,
    )
    return validate_terminal_coverage_manifest(
        manifest,
        contract,
        schedule_index,
        terminal_policy=policy,
        packed_manifest=packed,
        authenticator=authenticator,
    )


def validate_terminal_coverage_manifest(
    value: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    *,
    terminal_policy: Mapping[str, Any],
    packed_manifest: Mapping[str, Any],
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    contract = _contract(expected_contract)
    declared = derive_declared_window(contract, schedule_index)
    policy = validate_terminal_policy(
        terminal_policy, contract, schedule_index, authenticator
    )
    packed = validate_artifact_manifest(
        packed_manifest,
        contract,
        schedule_index,
        "CANDIDATES_PACKED",
        authenticator,
    )
    coverage = _verify_authenticated(value, authenticator, label="terminal coverage")
    required = {
        "schema",
        "status",
        "campaign_id",
        "source_revision",
        "expected_contract_sha256",
        "window_id",
        "schedule_index",
        "schedule_entry_sha256",
        "terminal_policy_sha256",
        "packed_manifest_sha256",
        "entries",
        "created_at",
        "producer_key_identity_sha256",
        "producer_hmac_sha256",
        "seal_sha256",
    }
    if set(coverage) != required:
        raise WindowExecutionError("terminal coverage fields are not exact")
    expected_identity = {
        "schema": TERMINAL_COVERAGE_SCHEMA,
        "status": "TERMINAL",
        "campaign_id": contract["campaign_id"],
        "source_revision": contract["source_revision"],
        "expected_contract_sha256": contract["seal_sha256"],
        "window_id": declared["window_id"],
        "schedule_index": schedule_index,
        "schedule_entry_sha256": declared["seal_sha256"],
        "terminal_policy_sha256": policy["seal_sha256"],
        "packed_manifest_sha256": packed["seal_sha256"],
    }
    for key, expected in expected_identity.items():
        if coverage.get(key) != expected:
            raise WindowExecutionError(f"terminal coverage {key} mismatch")
    _require_name(coverage.get("created_at"), "terminal coverage created_at")
    policy_by_tensor = {item["tensor_name"]: item for item in policy["entries"]}
    entries = coverage.get("entries")
    if not isinstance(entries, list) or len(entries) != len(declared["tensor_set"]):
        raise WindowExecutionError("terminal coverage must contain every tensor once")
    payload_files = {
        item["relative_path"]: item
        for item in packed["files"]
        if item["artifact_kind"] == "COMPACT_PAYLOAD"
    }
    source_sizes = {
        item["path"]: item["logical_bytes"] for item in declared["source_shards"]
    }
    seen: set[str] = set()
    intervals: dict[str, list[tuple[int, int, str]]] = {}
    billed_by_source: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise WindowExecutionError("terminal coverage entry must be an object")
        tensor = _require_name(entry.get("tensor_name"), "coverage tensor_name")
        if tensor in seen or tensor not in policy_by_tensor:
            raise WindowExecutionError("terminal coverage tensor set is not exact")
        seen.add(tensor)
        policy_entry = policy_by_tensor[tensor]
        disposition = entry.get("disposition")
        if disposition != policy_entry["disposition"]:
            raise WindowExecutionError("terminal disposition differs from preregistration")
        if disposition == COMPACT_DISPOSITION:
            if set(entry) != {
                "tensor_name",
                "disposition",
                "payload_class",
                "payload_relative_path",
                "byte_offset",
                "byte_length",
            }:
                raise WindowExecutionError("compact terminal lineage fields are not exact")
            if entry.get("payload_class") != policy_entry["payload_class"]:
                raise WindowExecutionError("compact payload class differs from policy")
            payload_path = entry.get("payload_relative_path")
            if payload_path not in payload_files:
                raise WindowExecutionError("compact lineage names no grounded compact payload")
            offset = entry.get("byte_offset")
            length = entry.get("byte_length")
            if type(offset) is not int or offset < 0 or type(length) is not int or length <= 0:
                raise WindowExecutionError("compact lineage byte range is invalid")
            end = offset + length
            if end > payload_files[payload_path]["logical_bytes"]:
                raise WindowExecutionError("compact lineage extends beyond payload file")
            intervals.setdefault(str(payload_path), []).append((offset, end, tensor))
        elif disposition == PROTECTED_DISPOSITION:
            if set(entry) != {
                "tensor_name",
                "disposition",
                "native_source_path",
                "billed_bytes",
                "protection_justification",
                "payload_relative_path",
                "byte_offset",
                "byte_length",
            }:
                raise WindowExecutionError("protected terminal lineage fields are not exact")
            for key in (
                "native_source_path",
                "billed_bytes",
                "protection_justification",
            ):
                if entry.get(key) != policy_entry.get(key):
                    raise WindowExecutionError(
                        f"protected terminal {key} differs from preregistration"
                    )
            path = str(entry["native_source_path"])
            billed_by_source[path] = billed_by_source.get(path, 0) + int(entry["billed_bytes"])
            payload_path = entry.get("payload_relative_path")
            if payload_path not in payload_files:
                raise WindowExecutionError(
                    "protected native lineage names no grounded compact payload"
                )
            offset = entry.get("byte_offset")
            length = entry.get("byte_length")
            if type(offset) is not int or offset < 0 or type(length) is not int \
                    or length <= 0 or length != entry["billed_bytes"]:
                raise WindowExecutionError(
                    "protected native byte range must exactly equal billed_bytes"
                )
            end = offset + length
            if end > payload_files[payload_path]["logical_bytes"]:
                raise WindowExecutionError("protected native lineage exceeds payload file")
            intervals.setdefault(str(payload_path), []).append((offset, end, tensor))
        elif disposition == OMITTED_DISPOSITION:
            if set(entry) != {
                "tensor_name",
                "disposition",
                "capability_justification",
                "justification_evidence_sha256",
            } or entry != policy_entry:
                raise WindowExecutionError(
                    "omission must exactly match its preregistered justification"
                )
        else:
            raise WindowExecutionError(
                "unknown terminal disposition; NON_MODEL_FILE is forbidden for tensors"
            )
    if seen != set(declared["tensor_set"]):
        raise WindowExecutionError("terminal tensor coverage has a gap")
    for payload_path, ranges in intervals.items():
        previous_end = -1
        for start, end, _ in sorted(ranges):
            if start < previous_end:
                raise WindowExecutionError(
                    f"compact tensor byte ranges overlap in {payload_path}"
                )
            previous_end = end
    for path, billed in billed_by_source.items():
        if path not in source_sizes or billed > source_sizes[path]:
            raise WindowExecutionError("protected billed bytes exceed their grounded source")
    return coverage


def make_fetch_intent_receipt(
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    *,
    controller_anchor_sha256: str,
    resource_policy_receipt: Mapping[str, Any],
    expected_resource_policy: ResourceReservePolicy,
    terminal_policy: Mapping[str, Any],
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    contract = _contract(expected_contract)
    declared = derive_declared_window(contract, schedule_index)
    resource = _validate_resource_receipt(
        resource_policy_receipt, contract, authenticator, expected_resource_policy
    )
    policy = validate_terminal_policy(
        terminal_policy, contract, schedule_index, authenticator
    )
    receipt = _make_receipt(
        "FETCH_INTENT",
        contract,
        schedule_index,
        controller_anchor_sha256=controller_anchor_sha256,
        previous_receipt=None,
        terminal_policy_sha256=policy["seal_sha256"],
        artifact_manifest_sha256s={},
        evidence={
            "declared_window": declared,
            "resource_policy_receipt": resource,
            "terminal_policy": policy,
        },
        authenticator=authenticator,
    )
    return validate_fetch_intent_receipt(receipt, contract, authenticator)


def make_fetch_committed_receipt(
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    *,
    controller_anchor_sha256: str,
    previous_receipt: Mapping[str, Any],
    source_root: str | os.PathLike[str],
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    contract = _contract(expected_contract)
    previous = validate_fetch_intent_receipt(previous_receipt, contract, authenticator)
    observations, xet = _ground_sources(
        contract,
        schedule_index,
        source_root=source_root,
        authenticator=authenticator,
    )
    receipt = _make_receipt(
        "FETCH_COMMITTED",
        contract,
        schedule_index,
        controller_anchor_sha256=controller_anchor_sha256,
        previous_receipt=previous,
        terminal_policy_sha256=previous["terminal_policy_sha256"],
        artifact_manifest_sha256s=previous["artifact_manifest_sha256s"],
        evidence={
            "source_observations": observations,
            "xet_acquisition_identities": xet,
        },
        authenticator=authenticator,
    )
    return validate_fetch_committed_receipt(
        receipt, contract, previous, authenticator
    )


def make_sources_verified_receipt(
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    *,
    controller_anchor_sha256: str,
    previous_receipt: Mapping[str, Any],
    source_root: str | os.PathLike[str],
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    contract = _contract(expected_contract)
    previous = _validate_unlinked_phase_for_construction(
        previous_receipt,
        contract,
        "FETCH_COMMITTED",
        authenticator,
    )
    observations, xet = _ground_sources(
        contract,
        schedule_index,
        source_root=source_root,
        authenticator=authenticator,
    )
    receipt = _make_receipt(
        "SOURCES_VERIFIED",
        contract,
        schedule_index,
        controller_anchor_sha256=controller_anchor_sha256,
        previous_receipt=previous,
        terminal_policy_sha256=previous["terminal_policy_sha256"],
        artifact_manifest_sha256s=previous["artifact_manifest_sha256s"],
        evidence={
            "source_observations": observations,
            "xet_acquisition_identities": xet,
        },
        authenticator=authenticator,
    )
    return validate_sources_verified_receipt(receipt, contract, previous, authenticator)


def _make_artifact_phase_receipt(
    phase: str,
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    *,
    controller_anchor_sha256: str,
    previous_receipt: Mapping[str, Any],
    artifact_root: str | os.PathLike[str],
    artifact_manifest: Mapping[str, Any],
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    contract = _contract(expected_contract)
    previous = _validate_unlinked_phase_for_construction(
        previous_receipt,
        contract,
        PHASES[PHASES.index(phase) - 1],
        authenticator,
    )
    manifest = validate_artifact_manifest(
        artifact_manifest, contract, schedule_index, phase, authenticator
    )
    observations = _ground_artifact_manifest(
        manifest, artifact_root=artifact_root, authenticator=authenticator
    )
    manifests = dict(previous["artifact_manifest_sha256s"])
    if phase in manifests:
        raise WindowExecutionError("artifact phase manifest was already registered")
    manifests[phase] = manifest["seal_sha256"]
    receipt = _make_receipt(
        phase,
        contract,
        schedule_index,
        controller_anchor_sha256=controller_anchor_sha256,
        previous_receipt=previous,
        terminal_policy_sha256=previous["terminal_policy_sha256"],
        artifact_manifest_sha256s=manifests,
        evidence={"artifact_manifest": manifest, "file_observations": observations},
        authenticator=authenticator,
    )
    return validate_phase_receipt(
        receipt,
        contract,
        authenticator=authenticator,
        previous_receipt=previous,
    )


def make_teacher_captured_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _make_artifact_phase_receipt("TEACHER_CAPTURED", *args, **kwargs)


def make_candidates_fit_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _make_artifact_phase_receipt("CANDIDATES_FIT", *args, **kwargs)


def make_candidates_packed_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _make_artifact_phase_receipt("CANDIDATES_PACKED", *args, **kwargs)


def make_forward_complete_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _make_artifact_phase_receipt("FORWARD_COMPLETE", *args, **kwargs)


def _source_root_identity_from_evidence(evidence: Mapping[str, Any]) -> tuple[int, int]:
    observations = evidence.get("source_observations")
    if not isinstance(observations, list) or not observations:
        raise WindowExecutionError("source evidence has no observations")
    first = observations[0]
    if not isinstance(first, dict):
        raise WindowExecutionError("source observation is malformed")
    return int(first["root_device"]), int(first["root_inode"])


def _validate_window_sealed_evidence(
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    authenticator: WindowExecutionAuthenticators,
) -> None:
    evidence = receipt["evidence"]
    if set(evidence) != {
        "terminal_policy",
        "packed_manifest",
        "terminal_coverage_manifest",
        "prerequisite_artifacts",
        "compact_file_observations",
        "native_source_observations",
        "xet_acquisition_identities",
    }:
        raise WindowExecutionError("WINDOW_SEALED evidence fields are not exact")
    schedule_index = int(receipt["schedule_index"])
    policy = validate_terminal_policy(
        evidence["terminal_policy"], contract, schedule_index, authenticator
    )
    if policy["seal_sha256"] != receipt["terminal_policy_sha256"]:
        raise WindowExecutionError("WINDOW_SEALED terminal policy binding mismatch")
    packed = validate_artifact_manifest(
        evidence["packed_manifest"],
        contract,
        schedule_index,
        "CANDIDATES_PACKED",
        authenticator,
    )
    if receipt["artifact_manifest_sha256s"].get("CANDIDATES_PACKED") != \
            packed["seal_sha256"]:
        raise WindowExecutionError("WINDOW_SEALED packed-manifest provenance mismatch")
    validate_terminal_coverage_manifest(
        evidence["terminal_coverage_manifest"],
        contract,
        schedule_index,
        terminal_policy=policy,
        packed_manifest=packed,
        authenticator=authenticator,
    )
    prerequisites = evidence.get("prerequisite_artifacts")
    if not isinstance(prerequisites, dict) \
            or set(prerequisites) != ARTIFACT_PHASES:
        raise WindowExecutionError(
            "WINDOW_SEALED must re-ground every prerequisite artifact phase"
        )
    for phase in ARTIFACT_PHASES:
        item = prerequisites[phase]
        manifest = _verify_artifact_observations(
            item, contract, schedule_index, phase, authenticator
        )
        if receipt["artifact_manifest_sha256s"].get(phase) != manifest["seal_sha256"]:
            raise WindowExecutionError(
                f"WINDOW_SEALED prerequisite manifest mismatch for {phase}"
            )
    if evidence["compact_file_observations"] != \
            prerequisites["CANDIDATES_PACKED"]["file_observations"]:
        raise WindowExecutionError(
            "WINDOW_SEALED compact observations differ from re-grounded prerequisite"
        )
    _verify_artifact_observations(
        {
            "artifact_manifest": packed,
            "file_observations": evidence["compact_file_observations"],
        },
        contract,
        schedule_index,
        "CANDIDATES_PACKED",
        authenticator,
    )
    _verify_grounded_sources(
        {
            "source_observations": evidence["native_source_observations"],
            "xet_acquisition_identities": evidence["xet_acquisition_identities"],
        },
        contract,
        schedule_index,
        authenticator,
    )


def _validate_phase_evidence(
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    authenticator: WindowExecutionAuthenticators,
    *,
    expected_resource_policy: ResourceReservePolicy | None = None,
) -> None:
    phase = str(receipt["phase"])
    index = int(receipt["schedule_index"])
    evidence = receipt["evidence"]

    def identity_from(items: object, label: str) -> dict[str, int]:
        if not isinstance(items, list) or not items or not isinstance(items[0], dict):
            raise WindowExecutionError(f"{label} lacks a filesystem observation")
        return {
            "device": int(items[0]["root_device"]),
            "inode": int(items[0]["root_inode"]),
        }
    if phase == "FETCH_INTENT":
        if set(evidence) != {
            "declared_window",
            "resource_policy_receipt",
            "terminal_policy",
        }:
            raise WindowExecutionError("FETCH_INTENT evidence fields are not exact")
        validate_declared_window(evidence["declared_window"], contract, index)
        policy = validate_terminal_policy(
            evidence["terminal_policy"], contract, index, authenticator
        )
        if policy["seal_sha256"] != receipt["terminal_policy_sha256"]:
            raise WindowExecutionError("FETCH_INTENT terminal policy binding mismatch")
        required_policy = expected_resource_policy
        if required_policy is None:
            raw_policy = evidence["resource_policy_receipt"].get("resource_policy")
            if not isinstance(raw_policy, dict):
                raise WindowExecutionError("FETCH_INTENT resource policy is malformed")
            try:
                required_policy = ResourceReservePolicy(**raw_policy)
            except (TypeError, GroundingError) as exc:
                raise WindowExecutionError("FETCH_INTENT resource policy is invalid") from exc
        resource = _validate_resource_receipt(
            evidence["resource_policy_receipt"],
            contract,
            authenticator,
            required_policy,
            as_of=str(receipt["created_at"]),
        )
        if receipt["source_root_identity"] != {
            "device": int(resource["root_device"]),
            "inode": int(resource["root_inode"]),
        } or receipt["artifact_root_identity"] is not None:
            raise WindowExecutionError("FETCH_INTENT filesystem root binding mismatch")
        if receipt["artifact_manifest_sha256s"]:
            raise WindowExecutionError("FETCH_INTENT artifact registry must be empty")
    elif phase in {"FETCH_COMMITTED", "SOURCES_VERIFIED"}:
        _verify_grounded_sources(evidence, contract, index, authenticator)
        if receipt["source_root_identity"] != identity_from(
            evidence["source_observations"], phase
        ):
            raise WindowExecutionError(f"{phase} source root binding mismatch")
    elif phase in ARTIFACT_PHASES:
        manifest = _verify_artifact_observations(
            evidence, contract, index, phase, authenticator
        )
        if receipt["artifact_manifest_sha256s"].get(phase) != manifest["seal_sha256"]:
            raise WindowExecutionError("artifact registry does not bind this phase manifest")
        if receipt["artifact_root_identity"] != identity_from(
            evidence["file_observations"], phase
        ):
            raise WindowExecutionError(f"{phase} artifact root binding mismatch")
    elif phase == "WINDOW_SEALED":
        _validate_window_sealed_evidence(receipt, contract, authenticator)
        if receipt["source_root_identity"] != identity_from(
            evidence["native_source_observations"], phase
        ) or receipt["artifact_root_identity"] != identity_from(
            evidence["compact_file_observations"], phase
        ):
            raise WindowExecutionError("WINDOW_SEALED filesystem root binding mismatch")
    elif phase == "EVICTION_COMMITTED":
        _validate_eviction_committed_evidence(receipt, contract, authenticator)
        if receipt["source_root_identity"] != identity_from(
            evidence["absence_observations"], phase
        ):
            raise WindowExecutionError("EVICTION_COMMITTED source root binding mismatch")
    else:  # pragma: no cover - envelope rejects this first
        raise WindowExecutionError(f"unsupported window phase: {phase}")


def _validate_phase_receipt_internal(
    value: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    *,
    authenticator: WindowExecutionAuthenticators,
    previous_receipt: Mapping[str, Any] | None,
    expected_resource_policy: ResourceReservePolicy | None = None,
    allow_unlinked_previous: bool = False,
) -> dict[str, Any]:
    """Internal validator; only construction code may inspect an orphan receipt."""

    receipt, contract = _verify_receipt_envelope(value, expected_contract, authenticator)
    phase_index = PHASES.index(receipt["phase"])
    if previous_receipt is None:
        if phase_index != 0 and not allow_unlinked_previous:
            raise WindowExecutionError("non-initial phase requires its previous receipt")
    else:
        previous, _ = _verify_receipt_envelope(
            previous_receipt, contract, authenticator
        )
        if phase_index == 0 or previous["phase"] != PHASES[phase_index - 1]:
            raise WindowExecutionError("window receipt predecessor phase is not exact")
        if receipt["previous_receipt_sha256"] != previous["seal_sha256"]:
            raise WindowExecutionError("window receipt previous-receipt hash mismatch")
        for key in (
            "campaign_id",
            "source_revision",
            "expected_contract_sha256",
            "window_id",
            "schedule_index",
            "schedule_entry_sha256",
            "controller_anchor_sha256",
            "terminal_policy_sha256",
            "source_root_identity",
        ):
            if receipt[key] != previous[key]:
                raise WindowExecutionError(f"window receipt chain changed immutable {key}")
        previous_artifact_root = previous["artifact_root_identity"]
        current_artifact_root = receipt["artifact_root_identity"]
        if previous_artifact_root is None:
            if receipt["phase"] != "TEACHER_CAPTURED" and current_artifact_root is not None:
                raise WindowExecutionError("artifact root appeared before teacher capture")
        elif current_artifact_root != previous_artifact_root:
            raise WindowExecutionError("artifact filesystem root changed across phase receipts")
        expected_manifests = dict(previous["artifact_manifest_sha256s"])
        if receipt["phase"] in ARTIFACT_PHASES:
            manifest = receipt["evidence"].get("artifact_manifest")
            if not isinstance(manifest, dict) or not _is_sha256(manifest.get("seal_sha256")):
                raise WindowExecutionError("artifact phase lacks a sealed manifest")
            if receipt["phase"] in expected_manifests:
                raise WindowExecutionError("artifact manifest registry replay")
            expected_manifests[receipt["phase"]] = manifest["seal_sha256"]
        if receipt["artifact_manifest_sha256s"] != expected_manifests:
            raise WindowExecutionError("artifact manifest registry is not monotonic")
        _validate_phase_evidence(previous, contract, authenticator)
    _validate_phase_evidence(
        receipt,
        contract,
        authenticator,
        expected_resource_policy=expected_resource_policy,
    )
    return receipt


def validate_phase_receipt(
    value: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    *,
    authenticator: WindowExecutionAuthenticators,
    previous_receipt: Mapping[str, Any] | None,
    expected_resource_policy: ResourceReservePolicy | None = None,
) -> dict[str, Any]:
    """Validate one receipt with its mandatory canonical predecessor."""

    return _validate_phase_receipt_internal(
        value,
        expected_contract,
        authenticator=authenticator,
        previous_receipt=previous_receipt,
        expected_resource_policy=expected_resource_policy,
        allow_unlinked_previous=False,
    )


def _validate_unlinked_phase_for_construction(
    value: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    expected_phase: str,
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    receipt = _validate_phase_receipt_internal(
        value,
        expected_contract,
        authenticator=authenticator,
        previous_receipt=None,
        allow_unlinked_previous=True,
    )
    if receipt["phase"] != expected_phase:
        raise WindowExecutionError(f"expected {expected_phase} receipt")
    return receipt


def validate_fetch_intent_receipt(
    value: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    authenticator: WindowExecutionAuthenticators,
    expected_resource_policy: ResourceReservePolicy | None = None,
) -> dict[str, Any]:
    receipt = validate_phase_receipt(
        value,
        expected_contract,
        authenticator=authenticator,
        previous_receipt=None,
        expected_resource_policy=expected_resource_policy,
    )
    if receipt["phase"] != "FETCH_INTENT":
        raise WindowExecutionError("expected FETCH_INTENT receipt")
    return receipt


def validate_fetch_committed_receipt(
    value: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    previous_receipt: Mapping[str, Any] | None,
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    receipt = validate_phase_receipt(
        value,
        expected_contract,
        authenticator=authenticator,
        previous_receipt=previous_receipt,
    )
    if receipt["phase"] != "FETCH_COMMITTED":
        raise WindowExecutionError("expected FETCH_COMMITTED receipt")
    return receipt


def validate_sources_verified_receipt(
    value: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    previous_receipt: Mapping[str, Any] | None,
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    receipt = validate_phase_receipt(
        value,
        expected_contract,
        authenticator=authenticator,
        previous_receipt=previous_receipt,
    )
    if receipt["phase"] != "SOURCES_VERIFIED":
        raise WindowExecutionError("expected SOURCES_VERIFIED receipt")
    return receipt


def _phase_validator(expected_phase: str):
    def validate(
        value: Mapping[str, Any],
        expected_contract: Mapping[str, Any],
        previous_receipt: Mapping[str, Any] | None,
        authenticator: WindowExecutionAuthenticators,
    ) -> dict[str, Any]:
        receipt = validate_phase_receipt(
            value,
            expected_contract,
            authenticator=authenticator,
            previous_receipt=previous_receipt,
        )
        if receipt["phase"] != expected_phase:
            raise WindowExecutionError(f"expected {expected_phase} receipt")
        return receipt

    return validate


validate_teacher_captured_receipt = _phase_validator("TEACHER_CAPTURED")
validate_candidates_fit_receipt = _phase_validator("CANDIDATES_FIT")
validate_candidates_packed_receipt = _phase_validator("CANDIDATES_PACKED")
validate_forward_complete_receipt = _phase_validator("FORWARD_COMPLETE")
validate_window_sealed_receipt = _phase_validator("WINDOW_SEALED")
validate_eviction_committed_receipt = _phase_validator("EVICTION_COMMITTED")


def validate_receipt_chain(
    receipts: Sequence[Mapping[str, Any]],
    expected_contract: Mapping[str, Any],
    *,
    authenticator: WindowExecutionAuthenticators,
    expected_resource_policy: ResourceReservePolicy,
) -> list[dict[str, Any]]:
    if not isinstance(receipts, Sequence) or len(receipts) != len(PHASES):
        raise WindowExecutionError("complete window receipt chain must contain all phases")
    return _validate_receipt_prefix(
        receipts,
        expected_contract,
        authenticator=authenticator,
        expected_resource_policy=expected_resource_policy,
        expected_phases=PHASES,
    )


def _validate_receipt_prefix(
    receipts: Sequence[Mapping[str, Any]],
    expected_contract: Mapping[str, Any],
    *,
    authenticator: WindowExecutionAuthenticators,
    expected_resource_policy: ResourceReservePolicy,
    expected_phases: Sequence[str],
) -> list[dict[str, Any]]:
    if not isinstance(receipts, Sequence) or len(receipts) != len(expected_phases):
        raise WindowExecutionError("window receipt prefix length is not exact")
    if tuple(expected_phases) != PHASES[: len(expected_phases)]:
        raise WindowExecutionError("window receipt prefix phases are not canonical")
    result = []
    previous = None
    for phase, raw in zip(expected_phases, receipts):
        validated = validate_phase_receipt(
            raw,
            expected_contract,
            authenticator=authenticator,
            previous_receipt=previous,
            expected_resource_policy=(
                expected_resource_policy if phase == "FETCH_INTENT" else None
            ),
        )
        if validated["phase"] != phase:
            raise WindowExecutionError("complete window receipt chain phase mismatch")
        result.append(validated)
        previous = validated
    return result


def make_window_sealed_receipt(
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    *,
    controller_anchor_sha256: str,
    previous_receipt: Mapping[str, Any],
    receipt_chain: Sequence[Mapping[str, Any]],
    expected_resource_policy: ResourceReservePolicy,
    terminal_policy: Mapping[str, Any],
    packed_manifest: Mapping[str, Any],
    terminal_coverage_manifest: Mapping[str, Any],
    source_root: str | os.PathLike[str],
    artifact_root: str | os.PathLike[str],
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    contract = _contract(expected_contract)
    prefix = _validate_receipt_prefix(
        receipt_chain,
        contract,
        authenticator=authenticator,
        expected_resource_policy=expected_resource_policy,
        expected_phases=PHASES[: PHASES.index("WINDOW_SEALED")],
    )
    previous = prefix[-1]
    if dict(previous_receipt) != previous:
        raise WindowExecutionError(
            "WINDOW_SEALED previous receipt is not the canonical full-chain head"
        )
    if previous["schedule_index"] != schedule_index:
        raise WindowExecutionError("WINDOW_SEALED schedule index differs from predecessor")
    if previous["controller_anchor_sha256"] != controller_anchor_sha256:
        raise WindowExecutionError("WINDOW_SEALED controller anchor differs from predecessor")
    policy = validate_terminal_policy(
        terminal_policy, contract, schedule_index, authenticator
    )
    if policy["seal_sha256"] != previous["terminal_policy_sha256"]:
        raise WindowExecutionError("WINDOW_SEALED terminal policy was not preregistered")
    packed = validate_artifact_manifest(
        packed_manifest,
        contract,
        schedule_index,
        "CANDIDATES_PACKED",
        authenticator,
    )
    if previous["artifact_manifest_sha256s"].get("CANDIDATES_PACKED") != packed["seal_sha256"]:
        raise WindowExecutionError("packed manifest differs from its phase commitment")
    coverage = validate_terminal_coverage_manifest(
        terminal_coverage_manifest,
        contract,
        schedule_index,
        terminal_policy=policy,
        packed_manifest=packed,
        authenticator=authenticator,
    )
    prerequisite_artifacts: dict[str, dict[str, Any]] = {}
    for item in prefix:
        phase = item["phase"]
        if phase not in ARTIFACT_PHASES:
            continue
        manifest = validate_artifact_manifest(
            item["evidence"]["artifact_manifest"],
            contract,
            schedule_index,
            phase,
            authenticator,
        )
        prerequisite_artifacts[phase] = {
            "artifact_manifest": manifest,
            "file_observations": _ground_artifact_manifest(
                manifest,
                artifact_root=artifact_root,
                authenticator=authenticator,
            ),
        }
    if set(prerequisite_artifacts) != ARTIFACT_PHASES:
        raise WindowExecutionError(
            "WINDOW_SEALED canonical chain lacks prerequisite artifact phases"
        )
    compact_observations = prerequisite_artifacts[
        "CANDIDATES_PACKED"
    ]["file_observations"]
    native_observations, xet = _ground_sources(
        contract,
        schedule_index,
        source_root=source_root,
        authenticator=authenticator,
    )
    receipt = _make_receipt(
        "WINDOW_SEALED",
        contract,
        schedule_index,
        controller_anchor_sha256=controller_anchor_sha256,
        previous_receipt=previous,
        terminal_policy_sha256=previous["terminal_policy_sha256"],
        artifact_manifest_sha256s=previous["artifact_manifest_sha256s"],
        evidence={
            "terminal_policy": policy,
            "packed_manifest": packed,
            "terminal_coverage_manifest": coverage,
            "prerequisite_artifacts": prerequisite_artifacts,
            "compact_file_observations": compact_observations,
            "native_source_observations": native_observations,
            "xet_acquisition_identities": xet,
        },
        authenticator=authenticator,
    )
    return validate_window_sealed_receipt(
        receipt, contract, previous, authenticator
    )


def _eviction_sources(
    contract: Mapping[str, Any], schedule_index: int
) -> list[dict[str, Any]]:
    declared = derive_declared_window(contract, schedule_index)
    identities = {item["path"]: item for item in declared["source_shards"]}
    targets = [copy.deepcopy(identities[path]) for path in declared["evict_shards"]]
    if {item["path"] for item in targets} & set(declared["carry_out_shards"]):
        raise WindowExecutionError("eviction target intersects carry-out sources")
    if schedule_index + 1 < len(contract["window_schedule"]):
        prefetch = set(contract["window_schedule"][schedule_index + 1]["source_shards"])
        if {item["path"] for item in targets} & prefetch:
            raise WindowExecutionError(
                "eviction target intersects the next window's protected prefetch manifest"
            )
    return targets


def _eviction_protected_paths(
    contract: Mapping[str, Any], schedule_index: int
) -> list[str]:
    declared = derive_declared_window(contract, schedule_index)
    protected = set(declared["carry_out_shards"])
    if schedule_index + 1 < len(contract["window_schedule"]):
        protected.update(contract["window_schedule"][schedule_index + 1]["source_shards"])
    return sorted(protected)


def _ground_eviction_sources(
    contract: Mapping[str, Any],
    schedule_index: int,
    *,
    source_root: str | os.PathLike[str],
    authenticator: WindowExecutionAuthenticators,
) -> list[dict[str, Any]]:
    observer = TrustedFilesystemObserver(
        source_root,
        root_id=source_root_id(contract),
        authenticator=authenticator.grounding,
    )
    return [
        observer.observe_regular_file(
            source["path"],
            expected_size_bytes=source["logical_bytes"],
            expected_sha256=source["lfs_sha256"],
        )
        for source in _eviction_sources(contract, schedule_index)
    ]


def _verify_eviction_source_observations(
    observations: object,
    contract: Mapping[str, Any],
    schedule_index: int,
    authenticator: WindowExecutionAuthenticators,
) -> list[dict[str, Any]]:
    targets = _eviction_sources(contract, schedule_index)
    if not isinstance(observations, list) or len(observations) != len(targets):
        raise WindowExecutionError("eviction observations do not exactly cover targets")
    result = []
    roots: set[tuple[int, int]] = set()
    for source, raw in zip(targets, observations):
        try:
            item = verify_authenticated_observation(raw, authenticator.grounding)
        except GroundingError as exc:
            raise WindowExecutionError(f"eviction source observation invalid: {exc}") from exc
        expected = {
            "schema": FILE_OBSERVATION_SCHEMA,
            "status": "PASS",
            "observation_kind": "contained_regular_file",
            "root_id": source_root_id(contract),
            "relative_path": source["path"],
            "expected_size_bytes": source["logical_bytes"],
            "expected_sha256": source["lfs_sha256"],
            "observed_sha256": source["lfs_sha256"],
            "logical_bytes": source["logical_bytes"],
            "hard_link_count": 1,
        }
        if set(item) != _FILE_OBSERVATION_KEYS:
            raise WindowExecutionError("eviction file observation fields are not exact")
        for key, wanted in expected.items():
            if item.get(key) != wanted:
                raise WindowExecutionError(f"eviction source {source['path']} {key} mismatch")
        roots.add((int(item["root_device"]), int(item["root_inode"])))
        result.append(item)
    if len(roots) != 1:
        raise WindowExecutionError("eviction source observations have differing roots")
    return result


_EVICTION_EVENT_KEYS = frozenset(
    {
        "schema",
        "event_kind",
        "seq",
        "campaign_id",
        "source_revision",
        "expected_contract_sha256",
        "window_id",
        "schedule_index",
        "schedule_entry_sha256",
        "controller_anchor_sha256",
        "previous_event_sha256",
        "payload",
        "payload_sha256",
        "created_at",
        "producer_key_identity_sha256",
        "producer_hmac_sha256",
        "seal_sha256",
    }
)


def _make_eviction_event(
    kind: str,
    contract: Mapping[str, Any],
    schedule_index: int,
    *,
    controller_anchor_sha256: str,
    previous_event_sha256: str | None,
    payload: Mapping[str, Any],
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    declared = derive_declared_window(contract, schedule_index)
    seq_by_kind = {
        "EVICTION_INTENT": 0,
        "EVICTION_QUARANTINED": 1,
        "EVICTION_COMMITTED": 2,
    }
    if kind not in seq_by_kind:
        raise WindowExecutionError("unknown eviction journal event kind")
    seq = seq_by_kind[kind]
    if seq == 0 and previous_event_sha256 is not None:
        raise WindowExecutionError("eviction intent cannot have a previous event")
    if seq > 0:
        _require_sha256(previous_event_sha256, "eviction previous event")
    return _authenticated(
        {
            "schema": EVICTION_EVENT_SCHEMA,
            "event_kind": kind,
            "seq": seq,
            "campaign_id": contract["campaign_id"],
            "source_revision": contract["source_revision"],
            "expected_contract_sha256": contract["seal_sha256"],
            "window_id": declared["window_id"],
            "schedule_index": schedule_index,
            "schedule_entry_sha256": declared["seal_sha256"],
            "controller_anchor_sha256": controller_anchor_sha256,
            "previous_event_sha256": previous_event_sha256,
            "payload": copy.deepcopy(dict(payload)),
            "payload_sha256": _sha(payload),
            "created_at": utc_now(),
        },
        authenticator,
    )


def _validate_eviction_event(
    value: Mapping[str, Any],
    contract: Mapping[str, Any],
    schedule_index: int,
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    event = _verify_authenticated(value, authenticator, label="eviction journal event")
    if set(event) != _EVICTION_EVENT_KEYS:
        raise WindowExecutionError("eviction journal event fields are not exact")
    kind = event.get("event_kind")
    seq = event.get("seq")
    if (kind, seq) not in {
        ("EVICTION_INTENT", 0),
        ("EVICTION_QUARANTINED", 1),
        ("EVICTION_COMMITTED", 2),
    }:
        raise WindowExecutionError("eviction journal event kind/sequence mismatch")
    declared = derive_declared_window(contract, schedule_index)
    expected = {
        "schema": EVICTION_EVENT_SCHEMA,
        "campaign_id": contract["campaign_id"],
        "source_revision": contract["source_revision"],
        "expected_contract_sha256": contract["seal_sha256"],
        "window_id": declared["window_id"],
        "schedule_index": schedule_index,
        "schedule_entry_sha256": declared["seal_sha256"],
    }
    for key, wanted in expected.items():
        if event.get(key) != wanted:
            raise WindowExecutionError(f"eviction event {key} mismatch")
    _require_sha256(event.get("controller_anchor_sha256"), "eviction controller anchor")
    if kind == "EVICTION_INTENT":
        if event.get("previous_event_sha256") is not None:
            raise WindowExecutionError("eviction intent previous event must be null")
    else:
        _require_sha256(event.get("previous_event_sha256"), "eviction previous event")
    if not isinstance(event.get("payload"), dict) \
            or event.get("payload_sha256") != _sha(event["payload"]):
        raise WindowExecutionError("eviction event payload hash mismatch")
    _require_name(event.get("created_at"), "eviction event created_at")
    return event


def _open_eviction_journal(
    journal_path: str | os.PathLike[str],
) -> tuple[
    list[int],
    int,
    int,
    str,
    list[tuple[str, tuple[int, int, int]]],
    os.stat_result,
]:
    raw = os.fspath(journal_path)
    if not isinstance(raw, str) or not os.path.isabs(raw) or raw.startswith("//"):
        raise WindowExecutionError("eviction journal path must be normalized absolute text")
    normalized = os.path.normpath(raw)
    if normalized != raw:
        raise WindowExecutionError("eviction journal path must be normalized")
    parent = os.path.dirname(raw)
    leaf = os.path.basename(raw)
    if not leaf or leaf in {".", ".."} or "/" in leaf or "\x00" in leaf:
        raise WindowExecutionError("eviction journal filename is invalid")
    try:
        parent_fds, parent_links, parent_root_stat = _open_absolute_directory_chain(parent)
    except GroundingError as exc:
        raise WindowExecutionError(str(exc)) from exc
    parent_fd = parent_fds[-1]
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | _CLOEXEC | _NOFOLLOW
    try:
        fd = os.open(leaf, flags, 0o600, dir_fd=parent_fd)
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or int(opened.st_nlink) != 1:
            os.close(fd)
            raise WindowExecutionError("eviction journal must be a single-link regular file")
        named = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if _type_identity(named) != _type_identity(opened):
            os.close(fd)
            raise WindowExecutionError("eviction journal identity changed while opening")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise WindowExecutionError(
                "eviction journal is locked by another executor"
            ) from exc
        return (
            parent_fds,
            parent_fd,
            fd,
            leaf,
            parent_links,
            parent_root_stat,
        )
    except BaseException:
        for item in reversed(parent_fds):
            os.close(item)
        raise


def _read_eviction_journal_fd(fd: int) -> list[dict[str, Any]]:
    size = int(os.fstat(fd).st_size)
    if size > MAX_EVICTION_JOURNAL_BYTES:
        raise WindowExecutionError(
            "eviction journal exceeds the bounded read limit"
        )
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    bytes_read = 0
    while True:
        chunk = os.read(
            fd,
            min(1024 * 1024, MAX_EVICTION_JOURNAL_BYTES + 1 - bytes_read),
        )
        if not chunk:
            break
        chunks.append(chunk)
        bytes_read += len(chunk)
        if bytes_read > MAX_EVICTION_JOURNAL_BYTES:
            raise WindowExecutionError(
                "eviction journal exceeds the bounded read limit"
            )
    data = b"".join(chunks)
    if not data:
        return []
    if not data.endswith(b"\n"):
        raise WindowExecutionError("eviction journal has a torn trailing record")
    events = []
    for number, raw in enumerate(data.splitlines(), 1):
        try:
            item = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WindowExecutionError(f"eviction journal line {number} is invalid") from exc
        if not isinstance(item, dict) or canonical(item) != raw:
            raise WindowExecutionError(f"eviction journal line {number} is not canonical")
        events.append(item)
    return events


def _append_eviction_event(
    fd: int,
    parent_fd: int,
    leaf: str,
    event: Mapping[str, Any],
    parent_fds: Sequence[int],
    parent_links: Sequence[tuple[str, tuple[int, int, int]]],
    parent_root_stat: os.stat_result,
) -> None:
    _verify_absolute_directory_chain(parent_fds, parent_links, parent_root_stat)
    opened = os.fstat(fd)
    named_pre = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    if _type_identity(named_pre) != _type_identity(opened) \
            or int(opened.st_nlink) != 1:
        raise WindowExecutionError("eviction journal identity changed before append")
    encoded = canonical(event) + b"\n"
    view = memoryview(encoded)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise WindowExecutionError("short write to eviction journal")
        view = view[written:]
    os.fsync(fd)
    os.fsync(parent_fd)
    named_post = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    if _type_identity(named_post) != _type_identity(opened) \
            or _type_identity(os.fstat(fd)) != _type_identity(opened):
        raise WindowExecutionError("eviction journal identity changed during append")
    _verify_absolute_directory_chain(parent_fds, parent_links, parent_root_stat)


def _load_validate_eviction_events(
    fd: int,
    contract: Mapping[str, Any],
    schedule_index: int,
    authenticator: WindowExecutionAuthenticators,
) -> list[dict[str, Any]]:
    raw_events = _read_eviction_journal_fd(fd)
    if len(raw_events) > 3:
        raise WindowExecutionError("eviction journal contains replayed extra events")
    events = [
        _validate_eviction_event(item, contract, schedule_index, authenticator)
        for item in raw_events
    ]
    if events:
        if events[0]["event_kind"] != "EVICTION_INTENT":
            raise WindowExecutionError("eviction journal must begin with EVICTION_INTENT")
    expected_kinds = ["EVICTION_INTENT", "EVICTION_QUARANTINED", "EVICTION_COMMITTED"]
    for index, event in enumerate(events):
        if event["event_kind"] != expected_kinds[index]:
            raise WindowExecutionError("eviction journal event order is invalid")
        if index and event["previous_event_sha256"] != events[index - 1]["seal_sha256"]:
            raise WindowExecutionError("eviction journal chain is broken")
    return events


def _validate_intent_payload(
    event: Mapping[str, Any],
    contract: Mapping[str, Any],
    schedule_index: int,
    authenticator: WindowExecutionAuthenticators,
    expected_resource_policy: ResourceReservePolicy | None = None,
) -> dict[str, Any]:
    if event["event_kind"] != "EVICTION_INTENT":
        raise WindowExecutionError("expected eviction intent event")
    payload = event["payload"]
    if set(payload) != {
        "previous_window_sealed_receipt_sha256",
        "resource_policy_receipt",
        "evict_sources",
        "protected_paths",
        "pre_unlink_observations",
        "logical_recovery_bytes",
        "allocated_recovery_bytes",
    }:
        raise WindowExecutionError("eviction intent payload fields are not exact")
    _require_sha256(
        payload.get("previous_window_sealed_receipt_sha256"),
        "eviction previous WINDOW_SEALED receipt",
    )
    if expected_resource_policy is None:
        raw_policy = payload["resource_policy_receipt"].get("resource_policy")
        if not isinstance(raw_policy, dict):
            raise WindowExecutionError("eviction intent resource policy is malformed")
        try:
            expected_resource_policy = ResourceReservePolicy(**raw_policy)
        except (TypeError, GroundingError) as exc:
            raise WindowExecutionError("eviction intent resource policy is invalid") from exc
    _validate_resource_receipt(
        payload["resource_policy_receipt"],
        contract,
        authenticator,
        expected_resource_policy,
        as_of=str(event["created_at"]),
    )
    targets = _eviction_sources(contract, schedule_index)
    if payload.get("evict_sources") != targets:
        raise WindowExecutionError("eviction intent targets differ from exact schedule manifest")
    if payload.get("protected_paths") != _eviction_protected_paths(contract, schedule_index):
        raise WindowExecutionError("eviction intent protected-path manifest mismatch")
    if set(payload["protected_paths"]) & {item["path"] for item in targets}:
        raise WindowExecutionError("eviction intent targets a protected path")
    observations = _verify_eviction_source_observations(
        payload.get("pre_unlink_observations"),
        contract,
        schedule_index,
        authenticator,
    )
    if payload.get("logical_recovery_bytes") != sum(
        int(item["logical_bytes"]) for item in observations
    ) or payload.get("allocated_recovery_bytes") != sum(
        int(item["allocated_bytes"]) for item in observations
    ):
        raise WindowExecutionError("eviction intent recovery accounting is not grounded")
    return payload


DESTRUCTIVE_AUTHORITY_BLOCK_REASON = (
    "production eviction is disabled: expected-contract v3 does not bind a "
    "controller-committed ResourceReservePolicy digest, complete semantic "
    "receipt-chain head, and frozen-schedule authority"
)


def prepare_eviction_intent(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Fail closed until state supplies immutable destructive authority."""

    raise WindowExecutionError(DESTRUCTIVE_AUTHORITY_BLOCK_REASON)


def reconcile_eviction(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Fail closed until state supplies immutable destructive authority."""

    raise WindowExecutionError(DESTRUCTIVE_AUTHORITY_BLOCK_REASON)


def execute_eviction(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Fail closed until state supplies immutable destructive authority."""

    raise WindowExecutionError(DESTRUCTIVE_AUTHORITY_BLOCK_REASON)


def _prepare_eviction_intent_test_only(
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    *,
    controller_anchor_sha256: str,
    previous_receipt: Mapping[str, Any],
    source_root: str | os.PathLike[str],
    journal_path: str | os.PathLike[str],
    resource_policy_receipt: Mapping[str, Any],
    expected_resource_policy: ResourceReservePolicy,
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    """TEST ONLY: append+fsync an intent without production authority."""

    contract = _contract(expected_contract)
    previous = _validate_unlinked_phase_for_construction(
        previous_receipt,
        contract,
        "WINDOW_SEALED",
        authenticator,
    )
    if previous["schedule_index"] != schedule_index \
            or previous["controller_anchor_sha256"] != controller_anchor_sha256:
        raise WindowExecutionError("eviction intent predecessor identity mismatch")
    resource = _validate_resource_receipt(
        resource_policy_receipt, contract, authenticator, expected_resource_policy
    )
    (
        parent_fds,
        parent_fd,
        fd,
        journal_leaf,
        parent_links,
        parent_root_stat,
    ) = _open_eviction_journal(journal_path)
    try:
        existing = _load_validate_eviction_events(fd, contract, schedule_index, authenticator)
        if existing:
            intent = existing[0]
            payload = _validate_intent_payload(
                intent,
                contract,
                schedule_index,
                authenticator,
                expected_resource_policy,
            )
            if intent["controller_anchor_sha256"] != controller_anchor_sha256 \
                    or payload["previous_window_sealed_receipt_sha256"] != \
                    previous["seal_sha256"]:
                raise WindowExecutionError("existing eviction intent belongs to another request")
            return intent
        observations = _ground_eviction_sources(
            contract,
            schedule_index,
            source_root=source_root,
            authenticator=authenticator,
        )
        intent = _make_eviction_event(
            "EVICTION_INTENT",
            contract,
            schedule_index,
            controller_anchor_sha256=controller_anchor_sha256,
            previous_event_sha256=None,
            payload={
                "previous_window_sealed_receipt_sha256": previous["seal_sha256"],
                "resource_policy_receipt": resource,
                "evict_sources": _eviction_sources(contract, schedule_index),
                "protected_paths": _eviction_protected_paths(contract, schedule_index),
                "pre_unlink_observations": observations,
                "logical_recovery_bytes": sum(
                    int(item["logical_bytes"]) for item in observations
                ),
                "allocated_recovery_bytes": sum(
                    int(item["allocated_bytes"]) for item in observations
                ),
            },
            authenticator=authenticator,
        )
        _validate_intent_payload(
            intent,
            contract,
            schedule_index,
            authenticator,
            expected_resource_policy,
        )
        _append_eviction_event(
            fd,
            parent_fd,
            journal_leaf,
            intent,
            parent_fds,
            parent_links,
            parent_root_stat,
        )
        return intent
    finally:
        os.close(fd)
        for item in reversed(parent_fds):
            os.close(item)


def _unlink_exact_observed_file(
    source_root: str | os.PathLike[str],
    relative_path: str,
    observation: Mapping[str, Any],
) -> None:
    """TEST ONLY: unlink a checked name in the private transaction harness.

    POSIX has no portable unlink-by-open-file-descriptor operation.  Another
    writer that controls this directory can replace ``leaf`` between the final
    ``stat`` and ``unlink`` calls.  The live APIs therefore stay fail-closed
    until the controller can prove exclusive source-root ownership (or supply
    a stronger platform primitive); these checks are defense in depth only.
    """

    try:
        root_path = _normalized_absolute_root(source_root)
        normalized, parts = _relative_parts(relative_path)
        root_fds, root_links, root_stat = _open_absolute_directory_chain(root_path)
    except GroundingError as exc:
        raise WindowExecutionError(str(exc)) from exc
    relative_fds: list[int] = []
    try:
        if parts[:-1]:
            relative_fds, relative_links = _open_relative_directory_chain(
                root_fds[-1], "/".join(parts[:-1])
            )
            parent_fd = relative_fds[-1]
        else:
            relative_fds = [os.dup(root_fds[-1])]
            relative_links = []
            parent_fd = relative_fds[-1]
        leaf = parts[-1]
        named = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(named.st_mode) or not stat.S_ISREG(named.st_mode):
            raise WindowExecutionError(f"eviction target is not a regular file: {normalized}")
        if int(named.st_nlink) != 1:
            raise WindowExecutionError(f"eviction target has multiple hard links: {normalized}")
        if int(named.st_dev) != int(observation["device"]) \
                or int(named.st_ino) != int(observation["inode"]) \
                or int(named.st_size) != int(observation["logical_bytes"]):
            raise WindowExecutionError(f"eviction target identity changed: {normalized}")
        os.unlink(leaf, dir_fd=parent_fd)
        os.fsync(parent_fd)
        try:
            os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise WindowExecutionError(f"eviction target still exists after unlink: {normalized}")
        _verify_relative_directory_chain(relative_fds, relative_links)
        _verify_absolute_directory_chain(root_fds, root_links, root_stat)
    except OSError as exc:
        raise WindowExecutionError(f"descriptor-relative unlink failed: {exc}") from exc
    finally:
        for fd in reversed(relative_fds):
            os.close(fd)
        for fd in reversed(root_fds):
            os.close(fd)


def _quarantine_path(source_path: str, intent_sha256: str) -> str:
    parent, _, leaf = source_path.rpartition("/")
    quarantine_leaf = f".{leaf}.glm52-evict-{intent_sha256[:24]}.quarantine"
    return f"{parent}/{quarantine_leaf}" if parent else quarantine_leaf


def _quarantine_entries(intent: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = intent["payload"]
    pre_by_path = {
        item["relative_path"]: item for item in payload["pre_unlink_observations"]
    }
    return [
        {
            "source_path": source["path"],
            "quarantine_path": _quarantine_path(source["path"], intent["seal_sha256"]),
            "device": int(pre_by_path[source["path"]]["device"]),
            "inode": int(pre_by_path[source["path"]]["inode"]),
            "logical_bytes": int(pre_by_path[source["path"]]["logical_bytes"]),
            "allocated_bytes": int(pre_by_path[source["path"]]["allocated_bytes"]),
        }
        for source in payload["evict_sources"]
    ]


def _validate_quarantine_event(
    event: Mapping[str, Any], intent: Mapping[str, Any]
) -> dict[str, Any]:
    if event["event_kind"] != "EVICTION_QUARANTINED" \
            or event["previous_event_sha256"] != intent["seal_sha256"]:
        raise WindowExecutionError("eviction quarantine event chain mismatch")
    payload = event["payload"]
    if set(payload) != {"quarantined_sources"} \
            or payload["quarantined_sources"] != _quarantine_entries(intent):
        raise WindowExecutionError("eviction quarantine manifest mismatch")
    return payload


def _move_exact_to_quarantine(
    source_root: str | os.PathLike[str],
    source_path: str,
    quarantine_path: str,
    expected: Mapping[str, Any],
) -> None:
    source_parent, _, source_leaf = source_path.rpartition("/")
    quarantine_parent, _, quarantine_leaf = quarantine_path.rpartition("/")
    if source_parent != quarantine_parent:
        raise WindowExecutionError("eviction quarantine must share the source directory")
    root_path = _normalized_absolute_root(source_root)
    root_fds, root_links, root_stat = _open_absolute_directory_chain(root_path)
    relative_fds: list[int] = []
    try:
        if source_parent:
            relative_fds, relative_links = _open_relative_directory_chain(
                root_fds[-1], source_parent
            )
            parent_fd = relative_fds[-1]
        else:
            relative_fds = [os.dup(root_fds[-1])]
            relative_links = []
            parent_fd = relative_fds[-1]

        def named_or_none(name: str) -> os.stat_result | None:
            try:
                return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return None

        source_stat = named_or_none(source_leaf)
        quarantine_stat = named_or_none(quarantine_leaf)
        if source_stat is not None and quarantine_stat is not None:
            raise WindowExecutionError("source and quarantine paths both exist")
        if source_stat is None and quarantine_stat is None:
            raise WindowExecutionError(
                "eviction target vanished without its deterministic quarantine"
            )
        wanted = (int(expected["device"]), int(expected["inode"]))
        if quarantine_stat is not None:
            if stat.S_ISLNK(quarantine_stat.st_mode) or not stat.S_ISREG(quarantine_stat.st_mode) \
                    or (int(quarantine_stat.st_dev), int(quarantine_stat.st_ino)) != wanted:
                raise WindowExecutionError("existing eviction quarantine identity mismatch")
            return
        assert source_stat is not None
        if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISREG(source_stat.st_mode) \
                or int(source_stat.st_nlink) != 1 \
                or (int(source_stat.st_dev), int(source_stat.st_ino)) != wanted:
            raise WindowExecutionError("eviction target identity changed before quarantine")
        os.rename(
            source_leaf,
            quarantine_leaf,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
        moved = os.stat(quarantine_leaf, dir_fd=parent_fd, follow_symlinks=False)
        if (int(moved.st_dev), int(moved.st_ino)) != wanted:
            # Best-effort rollback prevents deleting a replacement raced into the name.
            try:
                os.rename(
                    quarantine_leaf,
                    source_leaf,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                os.fsync(parent_fd)
            finally:
                raise WindowExecutionError("eviction target raced during quarantine rename")
        _verify_relative_directory_chain(relative_fds, relative_links)
        _verify_absolute_directory_chain(root_fds, root_links, root_stat)
    except (OSError, GroundingError) as exc:
        raise WindowExecutionError(f"eviction quarantine operation failed: {exc}") from exc
    finally:
        for fd in reversed(relative_fds):
            os.close(fd)
        for fd in reversed(root_fds):
            os.close(fd)


def _scan_evicted_inodes_absent(
    source_root: str | os.PathLike[str], entries: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    root_path = _normalized_absolute_root(source_root)
    root_fds, root_links, root_stat = _open_absolute_directory_chain(root_path)
    try:
        inventory = _walk_regular_files(root_fds[-1])
        _verify_absolute_directory_chain(root_fds, root_links, root_stat)
    finally:
        for fd in reversed(root_fds):
            os.close(fd)
    result = []
    for entry in entries:
        found = sorted(
            path
            for path, metadata in inventory.items()
            if (metadata[0], metadata[1]) == (entry["device"], entry["inode"])
        )
        if found:
            raise WindowExecutionError(
                f"evicted source inode still exists under an unlisted name: {found}"
            )
        result.append(
            {"device": entry["device"], "inode": entry["inode"], "found_paths": []}
        )
    return result


def _verify_absence_observations(
    observations: object,
    contract: Mapping[str, Any],
    schedule_index: int,
    authenticator: WindowExecutionAuthenticators,
) -> list[dict[str, Any]]:
    targets = _eviction_sources(contract, schedule_index)
    if not isinstance(observations, list) or len(observations) != len(targets):
        raise WindowExecutionError("absence observations do not exactly cover eviction")
    result = []
    roots: set[tuple[int, int]] = set()
    for source, raw in zip(targets, observations):
        try:
            item = verify_authenticated_observation(raw, authenticator.grounding)
        except GroundingError as exc:
            raise WindowExecutionError(f"eviction absence observation invalid: {exc}") from exc
        path = str(source["path"])
        parent = path.rpartition("/")[0] or "."
        expected = {
            "schema": ABSENCE_OBSERVATION_SCHEMA,
            "status": "PASS",
            "observation_kind": "contained_path_absence",
            "root_id": source_root_id(contract),
            "relative_path": path,
            "absent": True,
            "first_missing_component": path,
            "existing_parent": parent,
        }
        if set(item) != _ABSENCE_OBSERVATION_KEYS:
            raise WindowExecutionError("eviction absence fields are not exact")
        for key, wanted in expected.items():
            if item.get(key) != wanted:
                raise WindowExecutionError(f"eviction absence {path} {key} mismatch")
        roots.add((int(item["root_device"]), int(item["root_inode"])))
        result.append(item)
    if len(roots) != 1:
        raise WindowExecutionError("eviction absence observations have differing roots")
    return result


def _validate_eviction_committed_evidence(
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    authenticator: WindowExecutionAuthenticators,
) -> None:
    evidence = receipt["evidence"]
    if set(evidence) != {
        "resource_policy_receipt",
        "eviction_intent_event",
        "eviction_quarantined_event",
        "evicted_shards",
        "absence_observations",
        "evicted_inode_scan",
        "logical_recovery_bytes",
        "allocated_recovery_bytes",
        "reconciled_previously_absent_shards",
    }:
        raise WindowExecutionError("EVICTION_COMMITTED evidence fields are not exact")
    index = int(receipt["schedule_index"])
    intent = _validate_eviction_event(
        evidence["eviction_intent_event"], contract, index, authenticator
    )
    payload = _validate_intent_payload(intent, contract, index, authenticator)
    quarantine_event = _validate_eviction_event(
        evidence["eviction_quarantined_event"], contract, index, authenticator
    )
    _validate_quarantine_event(quarantine_event, intent)
    if intent["controller_anchor_sha256"] != receipt["controller_anchor_sha256"] \
            or payload["previous_window_sealed_receipt_sha256"] != \
            receipt["previous_receipt_sha256"]:
        raise WindowExecutionError("eviction intent is not bound to this window receipt")
    paths = [item["path"] for item in _eviction_sources(contract, index)]
    if evidence.get("evicted_shards") != paths:
        raise WindowExecutionError("evicted_shards differs from exact schedule manifest")
    expected_inode_scan = [
        {"device": item["device"], "inode": item["inode"], "found_paths": []}
        for item in _quarantine_entries(intent)
    ]
    if evidence.get("evicted_inode_scan") != expected_inode_scan:
        raise WindowExecutionError("evicted inode absence scan is not exact")
    absences = _verify_absence_observations(
        evidence.get("absence_observations"), contract, index, authenticator
    )
    expected_policy_raw = evidence["resource_policy_receipt"].get("resource_policy")
    if not isinstance(expected_policy_raw, dict):
        raise WindowExecutionError("eviction resource policy is malformed")
    try:
        expected_policy = ResourceReservePolicy(**expected_policy_raw)
    except (TypeError, GroundingError) as exc:
        raise WindowExecutionError("eviction resource policy is invalid") from exc
    resource = _validate_resource_receipt(
        evidence["resource_policy_receipt"],
        contract,
        authenticator,
        expected_policy,
        as_of=str(receipt["created_at"]),
    )
    root_identity = (int(resource["root_device"]), int(resource["root_inode"]))
    if any(
        (int(item["root_device"]), int(item["root_inode"])) != root_identity
        for item in absences
    ):
        raise WindowExecutionError("eviction absence root differs from resource root")
    if evidence.get("logical_recovery_bytes") != payload["logical_recovery_bytes"] \
            or evidence.get("allocated_recovery_bytes") != \
            payload["allocated_recovery_bytes"]:
        raise WindowExecutionError("eviction recovery accounting changed after intent")
    reconciled = evidence.get("reconciled_previously_absent_shards")
    if not isinstance(reconciled, list) or len(reconciled) != len(set(reconciled)) \
            or not set(reconciled).issubset(paths):
        raise WindowExecutionError("eviction partial-reconciliation list is invalid")


def _verify_current_eviction_absence(
    contract: Mapping[str, Any],
    schedule_index: int,
    *,
    source_root: str | os.PathLike[str],
    authenticator: WindowExecutionAuthenticators,
) -> list[dict[str, Any]]:
    observer = TrustedFilesystemObserver(
        source_root,
        root_id=source_root_id(contract),
        authenticator=authenticator.grounding,
    )
    result = [
        observer.observe_absence(str(item["path"]))
        for item in _eviction_sources(contract, schedule_index)
    ]
    return _verify_absence_observations(
        result, contract, schedule_index, authenticator
    )


def _reconcile_eviction_test_only(
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    *,
    controller_anchor_sha256: str,
    previous_receipt: Mapping[str, Any],
    source_root: str | os.PathLike[str],
    journal_path: str | os.PathLike[str],
    resource_policy_receipt: Mapping[str, Any],
    expected_resource_policy: ResourceReservePolicy,
    authenticator: WindowExecutionAuthenticators,
) -> dict[str, Any]:
    """TEST ONLY: exercise reconciliation; never call from a live worker."""

    contract = _contract(expected_contract)
    previous = _validate_unlinked_phase_for_construction(
        previous_receipt,
        contract,
        "WINDOW_SEALED",
        authenticator,
    )
    if previous["schedule_index"] != schedule_index \
            or previous["controller_anchor_sha256"] != controller_anchor_sha256:
        raise WindowExecutionError("eviction reconcile predecessor identity mismatch")
    resource = _validate_resource_receipt(
        resource_policy_receipt, contract, authenticator, expected_resource_policy
    )
    (
        parent_fds,
        parent_fd,
        fd,
        journal_leaf,
        parent_links,
        parent_root_stat,
    ) = _open_eviction_journal(journal_path)
    try:
        events = _load_validate_eviction_events(fd, contract, schedule_index, authenticator)
        if not events:
            raise WindowExecutionError("cannot reconcile eviction without durable intent")
        intent = events[0]
        payload = _validate_intent_payload(
            intent,
            contract,
            schedule_index,
            authenticator,
            expected_resource_policy,
        )
        if intent["controller_anchor_sha256"] != controller_anchor_sha256 \
                or payload["previous_window_sealed_receipt_sha256"] != \
                previous["seal_sha256"]:
            raise WindowExecutionError("durable eviction intent identity mismatch")
        quarantine_event = events[1] if len(events) >= 2 else None
        if quarantine_event is not None:
            _validate_quarantine_event(quarantine_event, intent)
        if len(events) == 3:
            commit_payload = events[2]["payload"]
            if set(commit_payload) != {"eviction_committed_receipt"}:
                raise WindowExecutionError("eviction commit event payload fields are not exact")
            committed = validate_eviction_committed_receipt(
                commit_payload["eviction_committed_receipt"],
                contract,
                previous,
                authenticator,
            )
            _verify_current_eviction_absence(
                contract,
                schedule_index,
                source_root=source_root,
                authenticator=authenticator,
            )
            _scan_evicted_inodes_absent(
                source_root, _quarantine_entries(intent)
            )
            return committed

        observer = TrustedFilesystemObserver(
            source_root,
            root_id=source_root_id(contract),
            authenticator=authenticator.grounding,
        )
        quarantine_entries = _quarantine_entries(intent)
        for entry in quarantine_entries:
            if quarantine_event is not None:
                try:
                    observer.observe_absence(str(entry["source_path"]))
                    observer.observe_absence(str(entry["quarantine_path"]))
                except GroundingError:
                    pass
                else:
                    # The durable quarantine event precedes unlink.  Both
                    # names absent is the recoverable crash state after unlink
                    # and before commit; the inode scan below must still pass.
                    continue
            _move_exact_to_quarantine(
                source_root,
                str(entry["source_path"]),
                str(entry["quarantine_path"]),
                entry,
            )
        if quarantine_event is None:
            quarantine_event = _make_eviction_event(
                "EVICTION_QUARANTINED",
                contract,
                schedule_index,
                controller_anchor_sha256=controller_anchor_sha256,
                previous_event_sha256=intent["seal_sha256"],
                payload={"quarantined_sources": quarantine_entries},
                authenticator=authenticator,
            )
            _validate_quarantine_event(quarantine_event, intent)
            _append_eviction_event(
                fd,
                parent_fd,
                journal_leaf,
                quarantine_event,
                parent_fds,
                parent_links,
                parent_root_stat,
            )
        previously_absent = []
        absence_observations = []
        source_by_path = {item["path"]: item for item in payload["evict_sources"]}
        for entry in quarantine_entries:
            path = str(entry["source_path"])
            quarantine_path = str(entry["quarantine_path"])
            source = source_by_path[path]
            try:
                current = observer.observe_regular_file(
                    quarantine_path,
                    expected_size_bytes=int(source["logical_bytes"]),
                    expected_sha256=str(source["lfs_sha256"]),
                )
            except GroundingError as file_error:
                try:
                    observer.observe_absence(quarantine_path)
                except GroundingError as absence_error:
                    raise WindowExecutionError(
                        f"eviction quarantine is neither the intended file nor absent: "
                        f"{quarantine_path}; "
                        f"file={file_error}; absence={absence_error}"
                    ) from file_error
                previously_absent.append(path)
            else:
                if int(current["device"]) != int(entry["device"]) \
                        or int(current["inode"]) != int(entry["inode"]):
                    raise WindowExecutionError(
                        f"eviction quarantine identity mismatch: {quarantine_path}"
                    )
                _unlink_exact_observed_file(source_root, quarantine_path, current)
            absence_observations.append(observer.observe_absence(path))
        absences = _verify_absence_observations(
            absence_observations, contract, schedule_index, authenticator
        )
        inode_scan = _scan_evicted_inodes_absent(source_root, quarantine_entries)
        receipt = _make_receipt(
            "EVICTION_COMMITTED",
            contract,
            schedule_index,
            controller_anchor_sha256=controller_anchor_sha256,
            previous_receipt=previous,
            terminal_policy_sha256=previous["terminal_policy_sha256"],
            artifact_manifest_sha256s=previous["artifact_manifest_sha256s"],
            evidence={
                "resource_policy_receipt": resource,
                "eviction_intent_event": intent,
                "eviction_quarantined_event": quarantine_event,
                "evicted_shards": [item["path"] for item in payload["evict_sources"]],
                "absence_observations": absences,
                "evicted_inode_scan": inode_scan,
                "logical_recovery_bytes": payload["logical_recovery_bytes"],
                "allocated_recovery_bytes": payload["allocated_recovery_bytes"],
                "reconciled_previously_absent_shards": previously_absent,
            },
            authenticator=authenticator,
        )
        committed = validate_eviction_committed_receipt(
            receipt, contract, previous, authenticator
        )
        commit_event = _make_eviction_event(
            "EVICTION_COMMITTED",
            contract,
            schedule_index,
            controller_anchor_sha256=controller_anchor_sha256,
            previous_event_sha256=quarantine_event["seal_sha256"],
            payload={"eviction_committed_receipt": committed},
            authenticator=authenticator,
        )
        _append_eviction_event(
            fd,
            parent_fd,
            journal_leaf,
            commit_event,
            parent_fds,
            parent_links,
            parent_root_stat,
        )
        return committed
    finally:
        os.close(fd)
        for item in reversed(parent_fds):
            os.close(item)


def _execute_eviction_test_only(
    expected_contract: Mapping[str, Any],
    schedule_index: int,
    **kwargs: Any,
) -> dict[str, Any]:
    """TEST ONLY: exercise the transaction engine without production authority."""

    _prepare_eviction_intent_test_only(expected_contract, schedule_index, **kwargs)
    return _reconcile_eviction_test_only(expected_contract, schedule_index, **kwargs)

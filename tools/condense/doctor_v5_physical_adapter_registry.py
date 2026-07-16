#!/usr/bin/env python3.12
"""Signed, default-off program-adapter registry for Doctor physical A/B work.

The physical controller must not be edited merely because a release build has
produced new executable bytes.  This module separates the source-reviewed
*policy* from release-time artifact identities:

* source code pins the only signer, SSHSIG namespace, schemas, facets, segment
  classes, validator ABI, and lifecycle restrictions;
* ``draft`` hashes every exact program/argv/scope/launch artifact and emits an
  inert registry receipt;
* ``sign`` is an explicit operator action using the out-of-repository key; and
* consumers accept adapters only from a complete, detached-signature-verified
  envelope whose current files still match every recorded identity.

Missing, partial, expired, stale, unsigned, or malformed registries yield no
adapters.  A valid registry still does not open the heavy lease, start a model,
or change a runtime default; it only supplies the concrete bindings needed by
the separately gated executor.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_ENVELOPE = (
    ROOT / "reports" / "condense" / "doctor_v5_ultra" /
    "staged_acceleration" / "physical_ab_v1" / "program_adapters" /
    "adapter_registry.envelope.json"
)
DEFAULT_ALLOWED_SIGNERS = (
    ROOT / "docs" / "plans" / "appendix_counter_authority_allowed_signers"
)
DEFAULT_PRIVATE_KEY = pathlib.Path.home() / ".ssh" / "id_ed25519"
SSH_KEYGEN = pathlib.Path("/usr/bin/ssh-keygen")
SIGNER_IDENTITY = "hawking-appendix-release"
SSHSIG_NAMESPACE = "hawking-doctor-physical-adapter-registry-v2"
SSH_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}

PINNED_ALLOWED_SIGNERS_SHA256 = (
    "a70164edd312fc8d33ae77fcd292f57241790aed737dcfefa5f7615c36498620"
)
PINNED_PUBLIC_KEY_BLOB_SHA256 = (
    "6db91ec62ce28aa4c6fe51f019f06b985e9e78bb3f8c345f61bd6186dd4d7dc8"
)

POLICY_SCHEMA = "hawking.doctor_v5_physical_adapter_policy.v2"
ENTRY_REQUEST_SCHEMA = "hawking.doctor_v5_physical_adapter_entry_request.v2"
ENTRY_SCHEMA = "hawking.doctor_v5_physical_adapter_entry.v2"
REGISTRY_SCHEMA = "hawking.doctor_v5_physical_adapter_registry.v2"
ENVELOPE_SCHEMA = "hawking.doctor_v5_physical_adapter_registry_envelope.v2"
SCIENTIFIC_RECEIPT_SCHEMA = "hawking.doctor_v5_physical_ab_scientific_receipt.v1"
SCIENTIFIC_VALIDATOR = (
    "doctor_v5_physical_ab_executor.validate_scientific_receipt.v1"
)
EXECUTION_SCOPE_SCHEMA = "hawking.doctor_v5_physical_ab_execution_scope.v1"
LAUNCH_CONTRACT_SCHEMA = "hawking.doctor_v5_physical_ab_launch_contract.v1"
ARGV_MANIFEST_SCHEMA = "hawking.doctor_v5_physical_ab_argv_manifest.v1"
VERIFICATION_SCHEMA = "hawking.doctor_v5_physical_adapter_verification.v2"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ADAPTER_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
TIER = re.compile(r"^([1-9][0-9]*)B$")
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_SIGNATURE_BYTES = 1024 * 1024
MAX_PRIVATE_KEY_BYTES = 1024 * 1024
MIN_VALID_SECONDS = 60
MAX_VALID_SECONDS = 86_400

# Programs may intentionally be shared between facets.  These documents may
# not: each is role/facet/scope-specific and reusing one would make an exact
# ten-facet registry ambiguous even though the detached signature remained
# cryptographically valid.
UNIQUE_PER_GROUP_ARTIFACTS = (
    "baseline_argv_manifest", "candidate_argv_manifest", "execution_scope",
    "launch_contract",
)
UNIQUE_GLOBAL_ARTIFACTS = ("execution_scope", "launch_contract")

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
SEGMENTS = (
    "sub-120b-doctor",
    "gpt-oss-120b",
    "post-120b-higher-tier",
)


class RegistryError(ValueError):
    """A registry artifact, signature, or exact adapter is untrustworthy."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _stamp(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop(field, None)
    result[field] = canonical_sha256(result)
    return result


def _hex(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _safe_bytes(path: pathlib.Path, *, maximum: int = MAX_JSON_BYTES) -> bytes:
    if path.is_symlink():
        raise RegistryError(f"adapter artifact is a symlink: {path}")
    path = path.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
                or before.st_size <= 0 or before.st_size > maximum:
            raise RegistryError(f"unsafe or oversized adapter artifact: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise RegistryError(f"adapter artifact exceeds its byte bound: {path}")
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
            row.st_ctime_ns, row.st_nlink,
        )
        if identity(before) != identity(after) or total != after.st_size:
            raise RegistryError(f"adapter artifact changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(_safe_bytes(path).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RegistryError(f"invalid adapter JSON {path}: {exc}") from exc


def file_identity(
    path: pathlib.Path, *, executable: bool = False, maximum: int = MAX_JSON_BYTES,
) -> dict[str, Any]:
    if path.is_symlink():
        raise RegistryError(f"adapter artifact is a symlink: {path}")
    resolved = path.resolve(strict=True)
    raw = _safe_bytes(resolved, maximum=maximum)
    if executable and not os.access(resolved, os.X_OK):
        raise RegistryError(f"adapter program is not executable: {path}")
    return {
        "path": str(resolved), "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _artifact_errors(
    value: Any, *, label: str, executable: bool = False, verify_files: bool = True,
    maximum: int = MAX_JSON_BYTES,
) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}:
        return [f"{label} identity is malformed"]
    errors: list[str] = []
    path = value.get("path")
    if not isinstance(path, str) or not pathlib.Path(path).is_absolute() \
            or not _hex(value.get("sha256")) \
            or isinstance(value.get("bytes"), bool) \
            or not isinstance(value.get("bytes"), int) or value["bytes"] <= 0 \
            or value["bytes"] > maximum:
        errors.append(f"{label} identity is malformed")
    if verify_files and not errors:
        try:
            observed = file_identity(
                pathlib.Path(path), executable=executable, maximum=maximum,
            )
        except (OSError, RegistryError) as exc:
            errors.append(f"{label} cannot be verified: {exc}")
        else:
            if observed != value:
                errors.append(f"{label} differs from the signed identity")
    return errors


def build_policy() -> dict[str, Any]:
    return _stamp({
        "schema": POLICY_SCHEMA,
        "policy_version": 2,
        "facets": list(FACETS),
        "segments": list(SEGMENTS),
        "sub120_scope": {
            "segment": "sub-120b-doctor", "model": "Doctor-V5",
            "tier": "3B-through-72B", "parameter_scope": "3B-through-72B",
        },
        "gptoss_scope": {
            "segment": "gpt-oss-120b", "model": "GPT-OSS", "tier": "120B",
            "parameter_scope": "exactly-120B",
        },
        "higher_scope": {
            "segment": "post-120b-higher-tier",
            "parameter_scope": "strictly-greater-than-120B",
        },
        "scientific_receipt_schema": SCIENTIFIC_RECEIPT_SCHEMA,
        "scientific_validator": SCIENTIFIC_VALIDATOR,
        "signature_format": "SSHSIG",
        "signature_namespace": SSHSIG_NAMESPACE,
        "signer_identity": SIGNER_IDENTITY,
        "allowed_signers_sha256": PINNED_ALLOWED_SIGNERS_SHA256,
        "public_key_blob_sha256": PINNED_PUBLIC_KEY_BLOB_SHA256,
        "shell_permitted": False,
        "ambient_environment_inheritance_permitted": False,
        "runtime_default_mutation_permitted": False,
        "source_deletion_permitted": False,
        "registry_grants_execution": False,
        "unsigned_registry_accepted": False,
        "registry_validity_seconds": {
            "minimum": MIN_VALID_SECONDS, "maximum": MAX_VALID_SECONDS,
        },
        "adapter_id_pattern": ADAPTER_ID.pattern,
        "exact_ten_facet_groups_required": True,
        "unique_adapter_ids_required": True,
        "unique_per_group_artifacts": list(UNIQUE_PER_GROUP_ARTIFACTS),
        "unique_global_artifacts": list(UNIQUE_GLOBAL_ARTIFACTS),
        "scope_launch_semantic_binding_required": True,
        "issuer_preflight_required": True,
        "independent_verification_required": True,
    }, "policy_sha256")


def _scope_errors(segment: Any, model: Any, tier: Any, parameter_scope: Any) -> list[str]:
    errors: list[str] = []
    if segment not in SEGMENTS \
            or not isinstance(model, str) or not 1 <= len(model) <= 128 \
            or "\x00" in model \
            or not isinstance(tier, str) or not 1 <= len(tier) <= 64 \
            or "\x00" in tier:
        return ["adapter segment/model/tier scope is invalid"]
    if segment == "sub-120b-doctor" and (
        model != "Doctor-V5" or tier != "3B-through-72B"
        or parameter_scope != "3B-through-72B"
    ):
        errors.append("sub-120B adapter scope differs from source-reviewed policy")
    if segment == "gpt-oss-120b" and (
        model != "GPT-OSS" or tier != "120B" or parameter_scope != "exactly-120B"
    ):
        errors.append("GPT-OSS adapter scope differs from exact 120B policy")
    if segment == "post-120b-higher-tier":
        match = TIER.fullmatch(tier)
        if parameter_scope != "strictly-greater-than-120B" \
                or match is None or int(match.group(1)) <= 120:
            errors.append("higher-tier adapter scope is not an exact tier greater than 120B")
    return errors


def _self_hash_errors(value: Any, field: str, *, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} is not a JSON object"]
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop(field, None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        return [f"{label} {field} mismatch"]
    return []


def _bound_document_errors(
    entry: dict[str, Any], *, plan_sha256: str | None,
    source_manifest_sha256: str | None,
) -> list[str]:
    """Verify semantic bindings inside every signed JSON identity.

    The executor remains the full ABI validator.  This independent issuer
    check prevents a signer from blessing a self-hash forgery, cross-facet
    scope, cross-plan launch, or argv manifest bound to different program
    bytes before the executor is ever eligible to run.
    """
    errors: list[str] = []
    try:
        scope = _load_json(pathlib.Path(entry["execution_scope"]["path"]))
        launch = _load_json(pathlib.Path(entry["launch_contract"]["path"]))
        baseline_argv = _load_json(pathlib.Path(entry["baseline_argv_manifest"]["path"]))
        candidate_argv = _load_json(pathlib.Path(entry["candidate_argv_manifest"]["path"]))
    except (KeyError, OSError, RegistryError) as exc:
        return [f"adapter bound JSON cannot be verified: {exc}"]

    errors.extend(_self_hash_errors(scope, "scope_sha256", label="execution scope"))
    scope_bindings = {
        "schema": EXECUTION_SCOPE_SCHEMA,
        "segment": entry["segment"], "model": entry["model"],
        "tier": entry["tier"], "parameter_scope": entry["parameter_scope"],
        "facet": entry["facet"],
    }
    if not isinstance(scope, dict) or any(
        scope.get(field) != expected for field, expected in scope_bindings.items()
    ):
        errors.append("execution scope differs from its exact adapter segment/model/tier/facet")

    errors.extend(_self_hash_errors(
        launch, "contract_sha256", label="launch contract",
    ))
    launch_bindings: dict[str, Any] = {
        "schema": LAUNCH_CONTRACT_SCHEMA,
        "facet": entry["facet"],
        "baseline_program": entry["baseline_program"],
        "baseline_argv_manifest": entry["baseline_argv_manifest"],
        "candidate_program": entry["candidate_program"],
        "candidate_argv_manifest": entry["candidate_argv_manifest"],
        "execution_scope": entry["execution_scope"],
    }
    if plan_sha256 is not None:
        launch_bindings["plan_sha256"] = plan_sha256
    if source_manifest_sha256 is not None:
        launch_bindings["source_manifest_sha256"] = source_manifest_sha256
    if not isinstance(launch, dict) or any(
        launch.get(field) != expected for field, expected in launch_bindings.items()
    ):
        errors.append("launch contract differs from exact plan/source/facet/artifact bindings")

    for role, document, program in (
        ("baseline", baseline_argv, entry["baseline_program"]),
        ("candidate", candidate_argv, entry["candidate_program"]),
    ):
        errors.extend(_self_hash_errors(
            document, "manifest_sha256", label=f"{role} argv manifest",
        ))
        if not isinstance(document, dict) \
                or document.get("schema") != ARGV_MANIFEST_SCHEMA \
                or document.get("role") != role \
                or document.get("program_sha256") != program["sha256"] \
                or document.get("direct_exec") is not True \
                or document.get("shell") is not False \
                or document.get("mutates_live_doctor") is not False \
                or document.get("mutates_runtime_defaults") is not False \
                or document.get("deletes_sources") is not False:
            errors.append(f"{role} argv manifest weakens exact no-shell program binding")
    return errors


def _entry_errors(
    value: Any, *, verify_files: bool, plan_sha256: str | None = None,
    source_manifest_sha256: str | None = None,
) -> list[str]:
    expected = {
        "schema", "adapter_id", "segment", "model", "tier", "parameter_scope",
        "facet", "baseline_program", "baseline_argv_manifest",
        "candidate_program", "candidate_argv_manifest", "execution_scope",
        "launch_contract", "execution_scope_sha256", "launch_contract_sha256",
        "scientific_receipt_schema", "scientific_validator", "policy_sha256",
        "entry_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return ["adapter entry fields are incomplete or unexpected"]
    errors = _scope_errors(
        value.get("segment"), value.get("model"), value.get("tier"),
        value.get("parameter_scope"),
    )
    if value.get("schema") != ENTRY_SCHEMA or value.get("facet") not in FACETS \
            or not isinstance(value.get("adapter_id"), str) \
            or ADAPTER_ID.fullmatch(value["adapter_id"]) is None:
        errors.append("adapter entry schema/id/facet is invalid")
    if value.get("policy_sha256") != build_policy()["policy_sha256"] \
            or value.get("scientific_receipt_schema") != SCIENTIFIC_RECEIPT_SCHEMA \
            or value.get("scientific_validator") != SCIENTIFIC_VALIDATOR:
        errors.append("adapter entry differs from source-reviewed validator policy")
    for field, executable in (
        ("baseline_program", True), ("baseline_argv_manifest", False),
        ("candidate_program", True), ("candidate_argv_manifest", False),
        ("execution_scope", False), ("launch_contract", False),
    ):
        errors.extend(_artifact_errors(
            value.get(field), label=f"adapter.{field}", executable=executable,
            verify_files=verify_files,
        ))
    if not _hex(value.get("execution_scope_sha256")) \
            or not _hex(value.get("launch_contract_sha256")):
        errors.append("adapter scope/launch self-hash binding is invalid")
    if verify_files and not errors:
        errors.extend(_bound_document_errors(
            value, plan_sha256=plan_sha256,
            source_manifest_sha256=source_manifest_sha256,
        ))
        try:
            scope = _load_json(pathlib.Path(value["execution_scope"]["path"]))
            launch = _load_json(pathlib.Path(value["launch_contract"]["path"]))
        except (OSError, RegistryError) as exc:
            errors.append(f"adapter scope/launch JSON cannot be verified: {exc}")
        else:
            if not isinstance(scope, dict) or scope.get("scope_sha256") \
                    != value.get("execution_scope_sha256") \
                    or not isinstance(launch, dict) or launch.get("contract_sha256") \
                    != value.get("launch_contract_sha256"):
                errors.append("adapter scope/launch self-hashes differ from signed bindings")
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("entry_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append("adapter entry hash mismatch")
    return errors


def build_entry(
    request: Any, *, plan_sha256: str | None = None,
    source_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    expected = {
        "schema", "adapter_id", "segment", "model", "tier", "parameter_scope",
        "facet", "baseline_program_path", "baseline_argv_manifest_path",
        "candidate_program_path", "candidate_argv_manifest_path",
        "execution_scope_path", "launch_contract_path",
    }
    if not isinstance(request, dict) or set(request) != expected \
            or request.get("schema") != ENTRY_REQUEST_SCHEMA:
        raise RegistryError("adapter entry request fields/schema are invalid")
    scope_errors = _scope_errors(
        request.get("segment"), request.get("model"), request.get("tier"),
        request.get("parameter_scope"),
    )
    if scope_errors or request.get("facet") not in FACETS:
        raise RegistryError("; ".join([*scope_errors, "adapter facet is invalid"]))
    paths = {
        field: pathlib.Path(request[f"{field}_path"])
        for field in (
            "baseline_program", "baseline_argv_manifest", "candidate_program",
            "candidate_argv_manifest", "execution_scope", "launch_contract",
        )
    }
    scope = _load_json(paths["execution_scope"])
    launch = _load_json(paths["launch_contract"])
    entry = _stamp({
        "schema": ENTRY_SCHEMA,
        "adapter_id": request["adapter_id"],
        "segment": request["segment"], "model": request["model"],
        "tier": request["tier"], "parameter_scope": request["parameter_scope"],
        "facet": request["facet"],
        "baseline_program": file_identity(paths["baseline_program"], executable=True),
        "baseline_argv_manifest": file_identity(paths["baseline_argv_manifest"]),
        "candidate_program": file_identity(paths["candidate_program"], executable=True),
        "candidate_argv_manifest": file_identity(paths["candidate_argv_manifest"]),
        "execution_scope": file_identity(paths["execution_scope"]),
        "launch_contract": file_identity(paths["launch_contract"]),
        "execution_scope_sha256": scope.get("scope_sha256")
        if isinstance(scope, dict) else None,
        "launch_contract_sha256": launch.get("contract_sha256")
        if isinstance(launch, dict) else None,
        "scientific_receipt_schema": SCIENTIFIC_RECEIPT_SCHEMA,
        "scientific_validator": SCIENTIFIC_VALIDATOR,
        "policy_sha256": build_policy()["policy_sha256"],
    }, "entry_sha256")
    errors = _entry_errors(
        entry, verify_files=True, plan_sha256=plan_sha256,
        source_manifest_sha256=source_manifest_sha256,
    )
    if errors:
        raise RegistryError("constructed adapter entry is invalid: " + "; ".join(errors))
    return entry


def build_registry(
    requests: list[Any], *, plan_sha256: str, source_manifest_sha256: str,
    issued_at_unix_ns: int | None = None, valid_seconds: int = 86_400,
) -> dict[str, Any]:
    if not _hex(plan_sha256) or not _hex(source_manifest_sha256):
        raise RegistryError("adapter registry plan/source binding is invalid")
    if isinstance(valid_seconds, bool) or not isinstance(valid_seconds, int) \
            or not MIN_VALID_SECONDS <= valid_seconds <= MAX_VALID_SECONDS:
        raise RegistryError("adapter registry validity must be 60..86400 seconds")
    issued = time.time_ns() if issued_at_unix_ns is None else issued_at_unix_ns
    if isinstance(issued, bool) or not isinstance(issued, int) or issued <= 0 \
            or issued > time.time_ns():
        raise RegistryError("adapter registry issue time is invalid")
    entries = [
        build_entry(
            row, plan_sha256=plan_sha256,
            source_manifest_sha256=source_manifest_sha256,
        )
        for row in requests
    ]
    keys = [(row["segment"], row["model"], row["facet"]) for row in entries]
    if len(set(keys)) != len(keys):
        raise RegistryError("adapter registry reuses a segment/model/facet key")
    adapter_ids = [row["adapter_id"] for row in entries]
    if len(set(adapter_ids)) != len(adapter_ids):
        raise RegistryError("adapter registry reuses an adapter id")
    sub120 = [row for row in entries if row["segment"] == "sub-120b-doctor"]
    if {row["facet"] for row in sub120} != set(FACETS):
        raise RegistryError("adapter registry lacks the exact ten-facet sub-120B population")
    groups: dict[tuple[str, str], set[str]] = {}
    for row in entries:
        groups.setdefault((row["segment"], row["model"]), set()).add(row["facet"])
    if any(facets != set(FACETS) for facets in groups.values()):
        raise RegistryError("every adapter segment/model group must contain exactly ten facets")
    for segment, model in groups:
        rows = [
            row for row in entries
            if (row["segment"], row["model"]) == (segment, model)
        ]
        for field in UNIQUE_PER_GROUP_ARTIFACTS:
            hashes = [row[field]["sha256"] for row in rows]
            if len(hashes) != len(set(hashes)):
                raise RegistryError(
                    f"adapter registry reuses {field} within {segment}/{model}"
                )
    for field in UNIQUE_GLOBAL_ARTIFACTS:
        hashes = [row[field]["sha256"] for row in entries]
        if len(hashes) != len(set(hashes)):
            raise RegistryError(f"adapter registry reuses {field} across groups")
    return _stamp({
        "schema": REGISTRY_SCHEMA,
        "policy": build_policy(),
        "plan_sha256": plan_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "issued_at_unix_ns": issued,
        "expires_at_unix_ns": issued + valid_seconds * 1_000_000_000,
        "entries": sorted(entries, key=lambda row: (
            row["segment"], row["model"], FACETS.index(row["facet"]),
        )),
        "complete_ten_facet_groups": [
            {"segment": segment, "model": model}
            for segment, model in sorted(groups)
        ],
        "shell_permitted": False,
        "ambient_environment_inheritance_permitted": False,
        "runtime_defaults_changed": False,
        "source_files_deleted": False,
        "activation_requested": False,
        "registry_grants_execution": False,
    }, "registry_sha256")


def validate_registry(
    value: Any, *, plan_sha256: str, source_manifest_sha256: str,
    verify_files: bool, now_unix_ns: int | None = None,
) -> list[str]:
    expected = {
        "schema", "policy", "plan_sha256", "source_manifest_sha256",
        "issued_at_unix_ns", "expires_at_unix_ns", "entries",
        "complete_ten_facet_groups", "shell_permitted",
        "ambient_environment_inheritance_permitted", "runtime_defaults_changed",
        "source_files_deleted", "activation_requested", "registry_grants_execution",
        "registry_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return ["adapter registry fields are incomplete or unexpected"]
    errors: list[str] = []
    if value.get("schema") != REGISTRY_SCHEMA or value.get("policy") != build_policy() \
            or value.get("plan_sha256") != plan_sha256 \
            or value.get("source_manifest_sha256") != source_manifest_sha256:
        errors.append("adapter registry policy/plan/source binding is invalid")
    issued, expires = value.get("issued_at_unix_ns"), value.get("expires_at_unix_ns")
    now = time.time_ns() if now_unix_ns is None else now_unix_ns
    if isinstance(now, bool) or not isinstance(now, int) or now <= 0:
        errors.append("adapter registry verification time is invalid")
        now = 0
    duration = expires - issued if isinstance(issued, int) \
        and not isinstance(issued, bool) and isinstance(expires, int) \
        and not isinstance(expires, bool) else None
    if isinstance(issued, bool) or not isinstance(issued, int) or issued <= 0 \
            or isinstance(expires, bool) or not isinstance(expires, int) \
            or expires <= issued or now < issued or now >= expires:
        errors.append("adapter registry validity interval is not current")
    if not isinstance(duration, int) \
            or not MIN_VALID_SECONDS * 1_000_000_000 \
            <= duration <= MAX_VALID_SECONDS * 1_000_000_000:
        errors.append("adapter registry validity duration exceeds source-reviewed bounds")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("adapter registry contains no entries")
        entries = []
    keys: list[tuple[Any, Any, Any]] = []
    groups: dict[tuple[Any, Any], set[Any]] = {}
    for row in entries:
        errors.extend(_entry_errors(
            row, verify_files=verify_files, plan_sha256=plan_sha256,
            source_manifest_sha256=source_manifest_sha256,
        ))
        if isinstance(row, dict):
            key = (row.get("segment"), row.get("model"), row.get("facet"))
            keys.append(key)
            groups.setdefault(key[:2], set()).add(key[2])
    if len(set(keys)) != len(keys):
        errors.append("adapter registry reuses a segment/model/facet key")
    adapter_ids = [
        row.get("adapter_id") for row in entries if isinstance(row, dict)
    ]
    if len(adapter_ids) != len(set(adapter_ids)):
        errors.append("adapter registry reuses an adapter id")
    if groups.get(("sub-120b-doctor", "Doctor-V5")) != set(FACETS) \
            or any(facets != set(FACETS) for facets in groups.values()):
        errors.append("adapter registry has a partial ten-facet group")
    claimed_groups = value.get("complete_ten_facet_groups")
    expected_groups = [
        {"segment": segment, "model": model} for segment, model in sorted(groups)
    ]
    if claimed_groups != expected_groups:
        errors.append("adapter registry complete-group census is not exact")
    if all(isinstance(row, dict) and row.get("facet") in FACETS for row in entries):
        expected_order = sorted(entries, key=lambda row: (
            row.get("segment"), row.get("model"), FACETS.index(row["facet"]),
        ))
        if entries != expected_order:
            errors.append("adapter registry entries are not in canonical facet order")
    for segment, model in groups:
        rows = [
            row for row in entries if isinstance(row, dict)
            and (row.get("segment"), row.get("model")) == (segment, model)
        ]
        for field in UNIQUE_PER_GROUP_ARTIFACTS:
            hashes = [
                row[field].get("sha256") for row in rows
                if isinstance(row.get(field), dict)
            ]
            if len(hashes) != len(rows) or len(hashes) != len(set(hashes)):
                errors.append(
                    f"adapter registry reuses or omits {field} within {segment}/{model}"
                )
    for field in UNIQUE_GLOBAL_ARTIFACTS:
        hashes = [
            row[field].get("sha256") for row in entries
            if isinstance(row, dict) and isinstance(row.get(field), dict)
        ]
        if len(hashes) != len(entries) or len(hashes) != len(set(hashes)):
            errors.append(f"adapter registry reuses or omits {field} across groups")
    if value.get("shell_permitted") is not False \
            or value.get("ambient_environment_inheritance_permitted") is not False \
            or value.get("runtime_defaults_changed") is not False \
            or value.get("source_files_deleted") is not False \
            or value.get("activation_requested") is not False \
            or value.get("registry_grants_execution") is not False:
        errors.append("adapter registry weakens no-shell/default-off lifecycle policy")
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("registry_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append("adapter registry hash mismatch")
    return list(dict.fromkeys(errors))


def _allowed_signers_bytes() -> bytes:
    raw = _safe_bytes(DEFAULT_ALLOWED_SIGNERS, maximum=1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != PINNED_ALLOWED_SIGNERS_SHA256:
        raise RegistryError("allowed-signers bytes differ from the compiled trust anchor")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeError as exc:
        raise RegistryError("allowed-signers trust anchor is not UTF-8") from exc
    lines = [line.split() for line in decoded.splitlines() if line.strip()]
    if len(lines) != 1 or len(lines[0]) != 3 or lines[0][0] != SIGNER_IDENTITY \
            or lines[0][1] != "ssh-ed25519" \
            or hashlib.sha256(f"{lines[0][1]} {lines[0][2]}".encode("ascii")).hexdigest() \
            != PINNED_PUBLIC_KEY_BLOB_SHA256:
        raise RegistryError("allowed-signers public key differs from the compiled trust anchor")
    return raw


def _verify_signature(registry: dict[str, Any], signature: dict[str, Any]) -> tuple[bool, str]:
    signature_errors = _artifact_errors(
        signature, label="adapter detached signature", verify_files=True,
        maximum=MAX_SIGNATURE_BYTES,
    )
    if signature_errors:
        return False, "; ".join(signature_errors)
    allowed = _allowed_signers_bytes()
    with tempfile.TemporaryDirectory(prefix="hawking-doctor-adapter-verify-") as directory:
        root = pathlib.Path(directory)
        allowed_path = root / "allowed_signers"
        allowed_path.write_bytes(allowed)
        signature_copy = root / "registry.sig"
        signature_copy.write_bytes(_safe_bytes(pathlib.Path(signature["path"]), maximum=1024 * 1024))
        process = subprocess.run(
            [
                str(SSH_KEYGEN), "-Y", "verify", "-f", str(allowed_path),
                "-I", SIGNER_IDENTITY, "-n", SSHSIG_NAMESPACE,
                "-s", str(signature_copy),
            ],
            cwd=ROOT, input=canonical_bytes(registry), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30, check=False, shell=False,
            env=SSH_ENV,
        )
    detail = (process.stderr or process.stdout).decode("utf-8", "replace")[-500:]
    return process.returncode == 0, detail


def validate_envelope(
    value: Any, *, plan_sha256: str, source_manifest_sha256: str,
    verify_files: bool = True, verify_signature: bool = True,
    now_unix_ns: int | None = None,
) -> list[str]:
    expected = {
        "schema", "registry", "signer_identity", "signature_namespace",
        "allowed_signers_sha256", "detached_signature", "envelope_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return ["adapter registry envelope fields are incomplete or unexpected"]
    errors = validate_registry(
        value.get("registry"), plan_sha256=plan_sha256,
        source_manifest_sha256=source_manifest_sha256,
        verify_files=verify_files, now_unix_ns=now_unix_ns,
    )
    if value.get("schema") != ENVELOPE_SCHEMA \
            or value.get("signer_identity") != SIGNER_IDENTITY \
            or value.get("signature_namespace") != SSHSIG_NAMESPACE \
            or value.get("allowed_signers_sha256") != PINNED_ALLOWED_SIGNERS_SHA256:
        errors.append("adapter registry envelope trust-root binding is invalid")
    errors.extend(_artifact_errors(
        value.get("detached_signature"), label="adapter detached signature",
        verify_files=verify_files, maximum=MAX_SIGNATURE_BYTES,
    ))
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop("envelope_sha256", None)
    if not _hex(claimed) or claimed != canonical_sha256(unstamped):
        errors.append("adapter registry envelope hash mismatch")
    if verify_signature and not errors:
        try:
            valid, detail = _verify_signature(value["registry"], value["detached_signature"])
        except (OSError, RegistryError, subprocess.SubprocessError) as exc:
            errors.append(f"adapter registry SSHSIG verification failed: {exc}")
        else:
            if not valid:
                errors.append("adapter registry SSHSIG verification failed" + (
                    f": {detail}" if detail else ""
                ))
    return list(dict.fromkeys(errors))


def normalized_registries(value: dict[str, Any]) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str, str], dict[str, str]]]:
    sub120: dict[str, dict[str, str]] = {}
    post120: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in value["registry"]["entries"]:
        normalized = {
            "adapter_id": row["adapter_id"],
            "baseline_program_sha256": row["baseline_program"]["sha256"],
            "baseline_argv_manifest_sha256": row["baseline_argv_manifest"]["sha256"],
            "candidate_program_sha256": row["candidate_program"]["sha256"],
            "candidate_argv_manifest_sha256": row["candidate_argv_manifest"]["sha256"],
            "launch_contract_sha256": row["launch_contract_sha256"],
            "execution_scope_sha256": row["execution_scope_sha256"],
            "scientific_receipt_schema": row["scientific_receipt_schema"],
            "scientific_validator": row["scientific_validator"],
        }
        key = (row["segment"], row["model"], row["facet"])
        post120[key] = normalized
        if key[:2] == ("sub-120b-doctor", "Doctor-V5"):
            sub120[row["facet"]] = normalized
    return sub120, post120


def load_registries(
    *, plan_sha256: str, source_manifest_sha256: str,
    envelope_path: pathlib.Path = DEFAULT_ENVELOPE,
) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str, str], dict[str, str]], list[str]]:
    try:
        envelope = _load_json(envelope_path)
    except FileNotFoundError:
        return {}, {}, ["signed physical program-adapter registry is absent"]
    except (OSError, RegistryError) as exc:
        return {}, {}, [f"signed physical program-adapter registry cannot be read: {exc}"]
    errors = validate_envelope(
        envelope, plan_sha256=plan_sha256,
        source_manifest_sha256=source_manifest_sha256,
        verify_files=True, verify_signature=True,
    )
    if errors:
        return {}, {}, errors
    sub120, post120 = normalized_registries(envelope)
    return sub120, post120, []


def _lexical_absolute(path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.path.abspath(os.fspath(path)))


def _open_dir_nofollow(path: pathlib.Path, *, create: bool) -> int:
    absolute = _lexical_absolute(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_immutable_at(
    directory_fd: int, name: str, *, expected: bytes, mode: int,
) -> tuple[int, int]:
    descriptor = os.open(
        name, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
                or stat.S_IMODE(before.st_mode) != mode:
            raise RegistryError(
                f"adapter output is not a single-link mode-{mode:04o} regular file: {name}"
            )
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > len(expected):
                break
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
            row.st_ctime_ns, row.st_nlink, stat.S_IMODE(row.st_mode),
        )
        if identity(before) != identity(after) or b"".join(chunks) != expected:
            raise RegistryError(f"adapter output changed or differs: {name}")
        return int(after.st_dev), int(after.st_ino)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: pathlib.Path, raw: bytes, *, mode: int = 0o444) -> None:
    """Seal one adapter artifact through a retained no-follow parent dirfd."""
    if not raw:
        raise RegistryError(f"refusing to seal an empty adapter output: {path}")
    target = _lexical_absolute(path)
    directory_fd = _open_dir_nofollow(target.parent, create=True)
    temporary = f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    installed_identity: tuple[int, int] | None = None
    created_target = False
    try:
        try:
            _read_immutable_at(directory_fd, target.name, expected=raw, mode=mode)
        except FileNotFoundError:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                0o600, dir_fd=directory_fd,
            )
            try:
                view = memoryview(raw)
                while view:
                    count = os.write(descriptor, view)
                    if count <= 0:
                        raise OSError("short write while sealing adapter registry")
                    view = view[count:]
                os.fsync(descriptor)
                os.fchmod(descriptor, mode)
            finally:
                os.close(descriptor)
            try:
                os.link(
                    temporary, target.name, src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd, follow_symlinks=False,
                )
            except FileExistsError:
                _read_immutable_at(directory_fd, target.name, expected=raw, mode=mode)
            else:
                linked = os.stat(target.name, dir_fd=directory_fd, follow_symlinks=False)
                installed_identity = (int(linked.st_dev), int(linked.st_ino))
                created_target = True
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            _read_immutable_at(directory_fd, target.name, expected=raw, mode=mode)
        os.fsync(directory_fd)
        visible_fd = _open_dir_nofollow(target.parent, create=False)
        try:
            retained, visible = os.fstat(directory_fd), os.fstat(visible_fd)
            if (retained.st_dev, retained.st_ino) != (visible.st_dev, visible.st_ino):
                raise RegistryError(f"adapter output parent was replaced: {target.parent}")
        finally:
            os.close(visible_fd)
    except BaseException:
        if created_target and installed_identity is not None:
            try:
                current = os.stat(target.name, dir_fd=directory_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == installed_identity:
                    os.unlink(target.name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(directory_fd)


def _current_controller_bindings() -> tuple[str, str]:
    """Return bindings from current source, never from the registry being signed."""
    import doctor_v5_physical_ab_controller as controller  # local: breaks import cycle

    plan = controller.build_plan()
    plan_sha256 = plan.get("plan_sha256")
    source_sha256 = plan.get("source_manifest", {}).get("manifest_sha256")
    if not _hex(plan_sha256) or not _hex(source_sha256):
        raise RegistryError("current controller plan/source bindings are unavailable")
    return plan_sha256, source_sha256


def _private_key_preflight(private_key: pathlib.Path) -> pathlib.Path:
    if private_key.is_symlink():
        raise RegistryError("adapter signing key must not be a symlink")
    resolved = private_key.resolve(strict=True)
    key_stat = resolved.stat(follow_symlinks=False)
    if not stat.S_ISREG(key_stat.st_mode) or key_stat.st_nlink != 1 \
            or key_stat.st_uid != os.geteuid() or key_stat.st_size <= 0 \
            or key_stat.st_size > MAX_PRIVATE_KEY_BYTES \
            or key_stat.st_mode & 0o077:
        raise RegistryError(
            "adapter signing key must be owner-held, single-link, bounded, regular, and mode 0600"
        )
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise RegistryError("adapter signing key must remain outside the repository")
    process = subprocess.run(
        [str(SSH_KEYGEN), "-y", "-f", str(resolved)],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=30, check=False, shell=False, env=SSH_ENV,
    )
    try:
        fields = process.stdout.decode("ascii", "strict").strip().split() \
            if process.returncode == 0 else []
    except UnicodeDecodeError:
        fields = []
    if len(fields) < 2 or fields[0] != "ssh-ed25519" \
            or hashlib.sha256(f"{fields[0]} {fields[1]}".encode("ascii")).hexdigest() \
            != PINNED_PUBLIC_KEY_BLOB_SHA256:
        raise RegistryError("adapter signing key does not match the compiled release signer")
    return resolved


def sign_registry(
    registry: dict[str, Any], *, private_key: pathlib.Path,
    signature_output: pathlib.Path, envelope_output: pathlib.Path,
) -> dict[str, Any]:
    # Source bindings always come from current controller source, never from
    # caller arguments or the untrusted registry.  Tests can replace the
    # read-only binding provider; production has no override surface.
    expected_plan_sha256, expected_source_manifest_sha256 = (
        _current_controller_bindings()
    )
    if not _hex(expected_plan_sha256) or not _hex(expected_source_manifest_sha256):
        raise RegistryError("adapter signing source bindings are invalid")
    signature_target = signature_output.parent.resolve(strict=False) / signature_output.name
    envelope_target = envelope_output.parent.resolve(strict=False) / envelope_output.name
    if signature_target == envelope_target:
        raise RegistryError("adapter signature and envelope outputs must be distinct")
    registry_errors = validate_registry(
        registry, plan_sha256=expected_plan_sha256,
        source_manifest_sha256=expected_source_manifest_sha256,
        verify_files=True,
    )
    if registry_errors:
        raise RegistryError(
            "refusing to sign invalid adapter registry: " + "; ".join(registry_errors)
        )
    allowed = _allowed_signers_bytes()
    resolved_key = _private_key_preflight(private_key)
    with tempfile.TemporaryDirectory(prefix="hawking-doctor-adapter-sign-") as directory:
        message = pathlib.Path(directory) / "registry.canonical.json"
        message.write_bytes(canonical_bytes(registry))
        process = subprocess.run(
            [
                str(SSH_KEYGEN), "-Y", "sign", "-f", str(resolved_key),
                "-n", SSHSIG_NAMESPACE, str(message),
            ],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, timeout=30, check=False, shell=False,
            env=SSH_ENV,
        )
        generated = pathlib.Path(str(message) + ".sig")
        if process.returncode != 0 or not generated.is_file():
            raise RegistryError("adapter SSHSIG signing failed: " + (
                process.stderr or process.stdout
            ).decode("utf-8", "replace")[-500:])
        signature_raw = generated.read_bytes()
    _atomic_bytes(signature_output, signature_raw)
    envelope = _stamp({
        "schema": ENVELOPE_SCHEMA,
        "registry": registry,
        "signer_identity": SIGNER_IDENTITY,
        "signature_namespace": SSHSIG_NAMESPACE,
        "allowed_signers_sha256": hashlib.sha256(allowed).hexdigest(),
        "detached_signature": file_identity(signature_output),
    }, "envelope_sha256")
    envelope_errors = validate_envelope(
        envelope, plan_sha256=expected_plan_sha256,
        source_manifest_sha256=expected_source_manifest_sha256,
        verify_files=True, verify_signature=True,
    )
    if envelope_errors:
        raise RegistryError(
            "signed adapter envelope failed independent verification: "
            + "; ".join(envelope_errors)
        )
    # Recompute after hashing/signing so a concurrent source edit cannot
    # produce an apparently current envelope from a stale preflight.
    current_after = _current_controller_bindings()
    if current_after != (expected_plan_sha256, expected_source_manifest_sha256):
        raise RegistryError("controller plan/source changed while signing adapter registry")
    _atomic_bytes(
        envelope_output,
        (json.dumps(envelope, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return envelope


def verify_envelope_path(
    envelope_path: pathlib.Path, *, plan_sha256: str,
    source_manifest_sha256: str, now_unix_ns: int | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    envelope: Any = None
    identity: dict[str, Any] | None = None
    if not _hex(plan_sha256) or not _hex(source_manifest_sha256):
        errors.append("adapter verification plan/source bindings are invalid")
    try:
        envelope = _load_json(envelope_path)
        identity = file_identity(envelope_path)
    except (OSError, RegistryError) as exc:
        errors.append(f"adapter registry envelope cannot be read: {exc}")
    if not errors:
        errors.extend(validate_envelope(
            envelope, plan_sha256=plan_sha256,
            source_manifest_sha256=source_manifest_sha256,
            verify_files=True, verify_signature=True,
            now_unix_ns=now_unix_ns,
        ))
    sub120: dict[str, dict[str, str]] = {}
    post120: dict[tuple[str, str, str], dict[str, str]] = {}
    if not errors:
        sub120, post120 = normalized_registries(envelope)
    groups = sorted({(segment, model) for segment, model, _facet in post120})
    return _stamp({
        "schema": VERIFICATION_SCHEMA,
        "envelope": identity,
        "plan_sha256": plan_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "signature_verified": not errors,
        "exact_ten_facet_sub120_verified": set(sub120) == set(FACETS),
        "verified_adapter_count": len(post120),
        "verified_groups": [
            {"segment": segment, "model": model} for segment, model in groups
        ],
        "default_off": True,
        "execution_granted": False,
        "physical_execution_claimed": False,
        "models_opened": False,
        "gpu_used": False,
        "runtime_defaults_changed": False,
        "errors": list(dict.fromkeys(errors)),
    }, "verification_sha256")


def status() -> dict[str, Any]:
    trust_errors: list[str] = []
    try:
        _allowed_signers_bytes()
    except (OSError, RegistryError, UnicodeError) as exc:
        trust_errors.append(str(exc))
    ssh_keygen_ready = (
        SSH_KEYGEN.is_file() and not SSH_KEYGEN.is_symlink()
        and os.access(SSH_KEYGEN, os.X_OK)
    )
    if not ssh_keygen_ready:
        trust_errors.append("pinned /usr/bin/ssh-keygen is unavailable")
    envelope_present = (
        DEFAULT_ENVELOPE.is_file() and not DEFAULT_ENVELOPE.is_symlink()
    )
    return _stamp({
        "schema": "hawking.doctor_v5_physical_adapter_registry_status.v2",
        "policy_sha256": build_policy()["policy_sha256"],
        "default_envelope": str(DEFAULT_ENVELOPE),
        "default_envelope_present": envelope_present,
        "issuer_ready_for_concrete_inputs": not trust_errors,
        "independent_verifier_available": True,
        "required_production_facets": list(FACETS),
        "production_descriptor_state": (
            "present-requires-verify-command" if envelope_present else "absent"
        ),
        "production_descriptors_verified": 0,
        "exact_ten_production_descriptors_verified": False,
        "trust_preflight_errors": trust_errors,
        "default_off": True,
        "execution_granted": False,
        "private_key_read": False,
        "models_opened": False,
        "gpu_used": False,
        "runtime_defaults_changed": False,
    }, "status_sha256")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("policy")
    draft = sub.add_parser("draft")
    draft.add_argument("--entry-requests", required=True, type=pathlib.Path)
    draft.add_argument("--plan-sha256", required=True)
    draft.add_argument("--source-manifest-sha256", required=True)
    draft.add_argument("--valid-seconds", type=int, default=86_400)
    draft.add_argument("--output", required=True, type=pathlib.Path)
    sign = sub.add_parser("sign")
    sign.add_argument("--registry", required=True, type=pathlib.Path)
    sign.add_argument("--private-key", type=pathlib.Path, default=DEFAULT_PRIVATE_KEY)
    sign.add_argument("--signature-output", required=True, type=pathlib.Path)
    sign.add_argument("--envelope-output", required=True, type=pathlib.Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--envelope", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            value = status()
        elif args.command == "policy":
            value = build_policy()
        elif args.command == "draft":
            requests = _load_json(args.entry_requests)
            if not isinstance(requests, list):
                raise RegistryError("entry-request artifact must be a JSON list")
            value = build_registry(
                requests, plan_sha256=args.plan_sha256,
                source_manifest_sha256=args.source_manifest_sha256,
                valid_seconds=args.valid_seconds,
            )
            _atomic_bytes(
                args.output,
                (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
        elif args.command == "sign":
            registry = _load_json(args.registry)
            value = sign_registry(
                registry, private_key=args.private_key,
                signature_output=args.signature_output,
                envelope_output=args.envelope_output,
            )
        else:
            plan_sha256, source_manifest_sha256 = _current_controller_bindings()
            value = verify_envelope_path(
                args.envelope, plan_sha256=plan_sha256,
                source_manifest_sha256=source_manifest_sha256,
            )
            if value["signature_verified"] is not True:
                print(json.dumps(value, indent=2, sort_keys=True))
                return 1
    except (OSError, RegistryError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

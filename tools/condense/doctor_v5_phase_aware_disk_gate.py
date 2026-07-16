#!/usr/bin/env python3.12
"""Fail-closed, successor-only Doctor V5 phase-aware disk admission.

The live queue does not import this module.  A reviewed successor may install
it around the predecessor ``_execution_resource_gate`` with
``install_phase_gate``.  The wrapper never grants RAM or swap credit.  It can
only remove the predecessor's *exact* disk-capacity blocker after a persisted
operational receipt has been atomically written outside the result tree,
re-read, and independently recomputed from stable source evidence.

The read-only core is deliberately two step:

* :func:`build_phase_receipt` constructs a self-hashed evidence receipt.
* :func:`evaluate_persisted_phase_receipt` re-reads a caller-persisted receipt,
  recomputes it, probes ``statvfs`` directly, and returns either a narrowly
  relaxed gate or the byte-for-byte predecessor gate.

No source, checkpoint, result, plan, registry, runtime specification, or
runtime default is ever changed by this module.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import datetime as dt
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from types import ModuleType
from typing import Any, Callable

import doctor_v5_remaining_scratch_ledger as ledger


ROOT = Path(__file__).resolve().parents[2]
PHASE_GATE_API_SCHEMA = "hawking.doctor_v5_phase_gate_api.v1"
PHASE_RECEIPT_SCHEMA = "hawking.doctor_v5_phase_aware_disk_receipt.v1"
PHASE_DIAGNOSTIC_SCHEMA = "hawking.doctor_v5_phase_gate_diagnostic.v1"
VERSION = "2026-07-15.1"
MAX_RECEIPT_AGE_SECONDS = 180.0
DISK_BLOCKER_PREFIX = "disk free is below "
PHASE_POLICY_KEY = "phase_aware_disk_gate"
PHASE_RECEIPT_DIRECTORY = "phase_receipts"
STAGED_ROOT_PARTS = (
    "reports", "condense", "doctor_v5_ultra", "staged_acceleration",
)
SHA_RE = re.compile(r"[0-9a-f]{64}")


class PhaseGateError(RuntimeError):
    """Evidence cannot safely authorize a disk-only admission reduction."""


@dataclass(frozen=True)
class PhaseGateBinding:
    plan_path: Path
    plan_file_sha256: str
    plan_sha256: str
    plan_cell_sha256: str
    cell_id: str
    cell_identity_sha256: str
    runtime_spec_path: Path
    runtime_spec_file_sha256: str
    program_spec_sha256: str
    execution_output_root: Path
    policy_root: Path
    disk_reserve_bytes: int
    declared_scratch_bytes: int
    frozen_projected_output_bytes: int
    module_sha256: str
    ledger_module_sha256: str
    ram_credit_bytes: int = 0

    @property
    def phase_receipt_path(self) -> Path:
        return (self.policy_root / PHASE_RECEIPT_DIRECTORY
                / f"{self.cell_id}.json")

    @property
    def worker_output_root(self) -> Path:
        return self.execution_output_root / "strand_ladder"

    @property
    def worker_request_path(self) -> Path:
        return self.worker_output_root / "request.json"

    @property
    def worker_checkpoint_path(self) -> Path:
        return self.worker_output_root / "checkpoint.json"


@dataclass(frozen=True)
class PhaseGateResult:
    gate: dict[str, Any]
    applied: bool
    reasons: tuple[str, ...]
    persisted_receipt_ref: dict[str, Any] | None = None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], *keys: str) -> dict[str, Any]:
    excluded = set(keys)
    return {name: child for name, child in value.items() if name not in excluded}


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _integer(value: Any, *, minimum: int = 0) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and value >= minimum)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _timestamp(value: dt.datetime | None = None) -> str:
    observed = value or _now()
    if not isinstance(observed, dt.datetime) or observed.tzinfo is None:
        raise PhaseGateError("receipt time is naive or invalid")
    return observed.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def _parse_timestamp(value: Any) -> dt.datetime:
    if not isinstance(value, str):
        raise PhaseGateError("phase receipt time is not a string")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseGateError("phase receipt time is invalid") from exc
    if parsed.tzinfo is None:
        raise PhaseGateError("phase receipt time has no timezone")
    return parsed.astimezone(dt.timezone.utc)


def _source_sha(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PhaseGateError(f"cannot hash module source {path}: {exc}") from exc
    return hashlib.sha256(raw).hexdigest()


def _absolute_confined(path: Path | str, root: Path) -> Path:
    if not isinstance(path, (Path, str)):
        raise PhaseGateError("bound path has an invalid type")
    try:
        return ledger._lexical_path(os.fspath(path), root, require_absolute=True)
    except (ledger.ScratchLedgerError, OSError, TypeError, ValueError) as exc:
        raise PhaseGateError(str(exc)) from exc


def _resolve_workspace_path(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PhaseGateError("document path binding is missing or invalid")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return _absolute_confined(candidate, root)


def _stable_json(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return ledger._read_json_stable(_absolute_confined(path, root), root)
    except (ledger.ScratchLedgerError, OSError, TypeError, ValueError) as exc:
        raise PhaseGateError(f"stable JSON evidence failed for {path}: {exc}") from exc


def _same_ref(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return first == second


def _relative(path: Path, root: Path) -> str:
    return str(_absolute_confined(path, root).relative_to(root))


def _policy_root_errors(policy_root: Path, workspace_root: Path) -> list[str]:
    errors: list[str] = []
    try:
        root = _absolute_confined(policy_root, workspace_root)
        relative = root.relative_to(workspace_root)
        if relative.parts[:len(STAGED_ROOT_PARTS)] != STAGED_ROOT_PARTS:
            errors.append("policy_root is outside the staged-acceleration tree")
        if "results" in relative.parts:
            errors.append("policy_root may not be inside a result tree")
        ledger._no_symlink_components(root, workspace_root)
        info = os.lstat(root)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            errors.append("policy_root is not a real directory")
    except (ledger.ScratchLedgerError, PhaseGateError, OSError, ValueError) as exc:
        errors.append(f"policy_root is unavailable: {exc}")
    return errors


def _no_symlink_existing_prefix(path: Path, root: Path) -> None:
    path = _absolute_confined(path, root)
    cursor = root
    for part in path.relative_to(root).parts:
        cursor /= part
        try:
            info = os.lstat(cursor)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(info.st_mode):
            raise PhaseGateError(f"symlink path component is forbidden: {cursor}")
        if cursor != path and not stat.S_ISDIR(info.st_mode):
            raise PhaseGateError(f"non-directory path component: {cursor}")


def _binding_errors(binding: PhaseGateBinding, workspace_root: Path, *,
                    require_worker_evidence: bool = True) -> list[str]:
    if not isinstance(binding, PhaseGateBinding):
        return ["phase-gate binding type is invalid"]
    errors: list[str] = []
    if any(not _valid_sha(value) for value in (
            binding.plan_file_sha256, binding.plan_sha256,
            binding.plan_cell_sha256, binding.cell_identity_sha256,
            binding.runtime_spec_file_sha256, binding.program_spec_sha256,
            binding.module_sha256, binding.ledger_module_sha256)):
        errors.append("phase-gate binding contains an invalid SHA-256")
    if not isinstance(binding.cell_id, str) or not binding.cell_id \
            or Path(binding.cell_id).name != binding.cell_id \
            or "\x00" in binding.cell_id:
        errors.append("phase-gate cell_id is not a safe basename")
    for name, value in (
            ("disk reserve", binding.disk_reserve_bytes),
            ("declared scratch", binding.declared_scratch_bytes),
            ("frozen output projection", binding.frozen_projected_output_bytes)):
        if not _integer(value):
            errors.append(f"phase-gate {name} is not a nonnegative integer")
    if binding.disk_reserve_bytes != ledger.DISK_RESERVE_BYTES:
        errors.append("phase-gate reserve is not exactly 150 decimal GB")
    if binding.declared_scratch_bytes < 1:
        errors.append("phase-gate declared scratch is not positive")
    if binding.ram_credit_bytes != 0:
        errors.append("phase-gate RAM credit must be exactly zero")
    try:
        plan_path = _absolute_confined(binding.plan_path, workspace_root)
        spec_path = _absolute_confined(binding.runtime_spec_path, workspace_root)
        request_path = _absolute_confined(binding.worker_request_path, workspace_root)
        checkpoint_path = _absolute_confined(
            binding.worker_checkpoint_path, workspace_root)
        execution_root = _absolute_confined(
            binding.execution_output_root, workspace_root)
        worker_root = _absolute_confined(binding.worker_output_root, workspace_root)
        policy_root = _absolute_confined(binding.policy_root, workspace_root)
        if worker_root != execution_root / "strand_ladder":
            errors.append("worker output is not the exact strand_ladder child")
        try:
            policy_root.relative_to(execution_root)
            errors.append("policy_root is inside the execution result tree")
        except ValueError:
            pass
        try:
            execution_root.relative_to(policy_root)
            errors.append("execution result tree is inside policy_root")
        except ValueError:
            pass
        for bound in (plan_path, spec_path, policy_root):
            ledger._no_symlink_components(bound, workspace_root)
        if require_worker_evidence:
            for bound in (request_path, checkpoint_path, execution_root, worker_root):
                ledger._no_symlink_components(bound, workspace_root)
        else:
            _no_symlink_existing_prefix(execution_root, workspace_root)
    except (ledger.ScratchLedgerError, PhaseGateError, OSError, ValueError) as exc:
        errors.append(f"phase-gate bound path is unsafe: {exc}")
    errors.extend(_policy_root_errors(binding.policy_root, workspace_root))
    return errors


def _expected_disk_blocker(original_gate: dict[str, Any]) -> str:
    required = original_gate.get("required_free_bytes")
    if not _integer(required):
        raise PhaseGateError("predecessor required_free_bytes is invalid")
    return f"disk free is below {required / 1e9:.3f} GB"


def _validate_predecessor_gate(original_gate: Any,
                               binding: PhaseGateBinding) -> str:
    if not isinstance(original_gate, dict) \
            or original_gate.get("schema") \
            != "hawking.doctor_v5_ultra_resource_gate.v1":
        raise PhaseGateError("predecessor gate schema is not exact")
    blockers = original_gate.get("blockers")
    if not isinstance(blockers, list) or any(not isinstance(row, str) for row in blockers) \
            or len(blockers) != len(set(blockers)):
        raise PhaseGateError("predecessor blockers are invalid or duplicate")
    expected = _expected_disk_blocker(original_gate)
    disk_rows = [row for row in blockers if row.startswith(DISK_BLOCKER_PREFIX)]
    if disk_rows != [expected]:
        raise PhaseGateError("exact predecessor disk blocker is absent or ambiguous")
    observed = original_gate.get("observed_current_output_bytes")
    if not _integer(observed):
        raise PhaseGateError("predecessor observed output is invalid")
    projected_remaining = max(0, binding.frozen_projected_output_bytes - observed)
    expected_required = (binding.disk_reserve_bytes
                         + binding.declared_scratch_bytes + projected_remaining)
    if original_gate.get("ok") is not False \
            or original_gate.get("capacity_ok") is not False \
            or original_gate.get("disk_reserve_bytes") != binding.disk_reserve_bytes \
            or original_gate.get("scratch_bytes") != binding.declared_scratch_bytes \
            or original_gate.get("projected_whole_output_bytes") \
            != binding.frozen_projected_output_bytes \
            or original_gate.get("projected_incremental_output_bytes") \
            != projected_remaining \
            or original_gate.get("required_free_bytes") != expected_required:
        raise PhaseGateError("predecessor disk accounting envelope changed")
    return expected


def _validate_plan_and_spec(binding: PhaseGateBinding,
                            execution_cell: dict[str, Any],
                            workspace_root: Path) -> dict[str, Any]:
    plan, plan_ref = _stable_json(binding.plan_path, workspace_root)
    if plan_ref["sha256"] != binding.plan_file_sha256 \
            or plan.get("plan_sha256") != binding.plan_sha256 \
            or plan.get("plan_sha256") != _hash_value(
                _without(plan, "plan_sha256")):
        raise PhaseGateError("campaign plan file/self-hash binding changed")
    cells = plan.get("cells")
    matches = [row for row in cells if isinstance(row, dict)
               and row.get("cell_id") == binding.cell_id] \
        if isinstance(cells, list) else []
    if len(matches) != 1:
        raise PhaseGateError("campaign plan has no unique target cell")
    cell = matches[0]
    if _hash_value(cell) != binding.plan_cell_sha256 \
            or cell != execution_cell \
            or cell.get("cell_identity_sha256") != binding.cell_identity_sha256 \
            or cell.get("projected_output_bytes") \
            != binding.frozen_projected_output_bytes \
            or cell.get("source_deletion_permitted") is not False:
        raise PhaseGateError("campaign plan/execution cell binding changed")
    admission = cell.get("admission")
    if not isinstance(admission, dict) \
            or admission.get("disk_reserve_bytes") != binding.disk_reserve_bytes \
            or admission.get("recommended_scratch_bytes") \
            != binding.declared_scratch_bytes:
        raise PhaseGateError("campaign cell resource envelope changed")
    if _resolve_workspace_path(cell.get("runtime_spec_path"), workspace_root) \
            != binding.runtime_spec_path:
        raise PhaseGateError("campaign cell runtime-spec path changed")

    spec, spec_ref = _stable_json(binding.runtime_spec_path, workspace_root)
    if spec_ref["sha256"] != binding.runtime_spec_file_sha256 \
            or spec.get("schema") != cell.get("runtime_spec_schema") \
            or spec.get("program_spec_sha256") != binding.program_spec_sha256 \
            or spec.get("source_deletion_permitted") is not False \
            or spec.get("adapter_id") != cell.get("adapter_id") \
            or spec.get("backend") != cell.get("backend") \
            or spec.get("model_family") != cell.get("model_family") \
            or spec.get("operation") != cell.get("command"):
        raise PhaseGateError("runtime specification identity changed")
    resources = spec.get("resources")
    if not isinstance(resources, dict) \
            or resources.get("disk_reserve_bytes") != binding.disk_reserve_bytes \
            or resources.get("scratch_budget_bytes") != binding.declared_scratch_bytes:
        raise PhaseGateError("runtime specification resource envelope changed")
    resource_hash = spec.get("resource_admission_sha256")
    if resource_hash is not None and resource_hash != _hash_value(resources):
        raise PhaseGateError("runtime resource-admission self-hash changed")
    campaign = spec.get("campaign_binding")
    if not isinstance(campaign, dict) \
            or campaign.get("cell_id") != binding.cell_id \
            or campaign.get("cell_identity_sha256") != binding.cell_identity_sha256 \
            or campaign.get("branch") != cell.get("branch") \
            or campaign.get("target_rate_id") != cell.get("rate_id") \
            or campaign.get("label") != cell.get("model_label"):
        raise PhaseGateError("runtime campaign binding changed")
    return {
        "plan": plan, "plan_ref": plan_ref, "cell": cell,
        "spec": spec, "spec_ref": spec_ref, "campaign": campaign,
    }


def _checkpoint_candidate(binding: PhaseGateBinding,
                          documents: dict[str, Any],
                          workspace_root: Path) -> dict[str, Any]:
    campaign = documents["campaign"]
    cell = documents["cell"]
    spec = documents["spec"]
    request, request_ref = _stable_json(binding.worker_request_path, workspace_root)
    checkpoint, checkpoint_ref = _stable_json(
        binding.worker_checkpoint_path, workspace_root)
    if request.get("campaign_binding") != campaign \
            or request.get("output_root") != str(binding.worker_output_root):
        raise PhaseGateError("worker request campaign/output binding changed")
    if request.get("label") != cell.get("model_label") \
            or request.get("model_family") != cell.get("model_family") \
            or request.get("codec") != spec.get("codec") \
            or request.get("evaluation") != spec.get("evaluation") \
            or request.get("doctor_hook") != spec.get("doctor_hook"):
        raise PhaseGateError("worker request differs from runtime specification")
    parameter = request.get("parameter_manifest")
    cell_parameter = cell.get("parameter_manifest")
    if not isinstance(parameter, dict) or not isinstance(cell_parameter, dict) \
            or _resolve_workspace_path(parameter.get("path"), workspace_root) \
            != _resolve_workspace_path(cell_parameter.get("path"), workspace_root) \
            or parameter.get("sha256") != cell_parameter.get("file_sha256"):
        raise PhaseGateError("worker request parameter-manifest binding changed")
    source = request.get("source")
    source_census = cell.get("source_census")
    if not isinstance(source, dict) or not isinstance(source_census, dict) \
            or _resolve_workspace_path(source.get("census_path"), workspace_root) \
            != _resolve_workspace_path(source_census.get("path"), workspace_root) \
            or source.get("census_sha256") != source_census.get("file_sha256") \
            or source.get("source_manifest_sha256") \
            != cell_parameter.get("source_manifest_sha256") \
            or _resolve_workspace_path(source.get("model_dir"), workspace_root) \
            != _resolve_workspace_path(cell.get("model_dir"), workspace_root):
        raise PhaseGateError("worker request source authority binding changed")
    resources = request.get("resources")
    if not isinstance(resources, dict) \
            or resources.get("disk_reserve_bytes") != binding.disk_reserve_bytes \
            or resources.get("scratch_budget_bytes") != binding.declared_scratch_bytes:
        raise PhaseGateError("worker request resource envelope changed")
    if checkpoint.get("request_sha256") != request_ref["sha256"]:
        raise PhaseGateError("worker checkpoint request binding changed")
    completed = checkpoint.get("completed_units")
    units = checkpoint.get("units")
    shards = source.get("shards") if isinstance(source, dict) else None
    if not isinstance(completed, list) or not isinstance(units, dict) \
            or not isinstance(shards, list) or not shards:
        raise PhaseGateError("worker completion/source evidence is malformed")
    if "bundle_manifest" not in completed:
        raise PhaseGateError("bundle_manifest is not checkpoint-complete")

    packed_rows: list[dict[str, Any]] = []
    for ordinal in range(len(shards)):
        encode = f"encode:{ordinal:05d}"
        attest = f"attest:{ordinal:05d}"
        if encode not in completed or attest not in completed:
            raise PhaseGateError("not every ordinal has completed encode+attest")
        encode_row = units.get(encode)
        attest_row = units.get(attest)
        packed = encode_row.get("artifact") if isinstance(encode_row, dict) else None
        archive = attest_row.get("archive") if isinstance(attest_row, dict) else None
        if not isinstance(packed, dict) or packed != archive \
                or set(packed) != {"path", "sha256", "bytes"} \
                or not _valid_sha(packed.get("sha256")) \
                or not _integer(packed.get("bytes"), minimum=1) \
                or _resolve_workspace_path(packed.get("path"), workspace_root) \
                != binding.worker_output_root \
                / f"bundle/shards/{ordinal:05d}.strand":
            raise PhaseGateError("encode/attest packed identity is not exact")
        packed_rows.append({"ordinal": ordinal, **packed})

    bundle_unit = units.get("bundle_manifest")
    bundle_artifact = (bundle_unit.get("artifact")
                       if isinstance(bundle_unit, dict) else None)
    manifest_path = binding.worker_output_root / "bundle/manifest.json"
    if not isinstance(bundle_artifact, dict) \
            or set(bundle_artifact) != {"path", "sha256", "bytes"} \
            or not _valid_sha(bundle_artifact.get("sha256")) \
            or not _integer(bundle_artifact.get("bytes"), minimum=1) \
            or _resolve_workspace_path(bundle_artifact.get("path"), workspace_root) \
            != manifest_path:
        raise PhaseGateError("bundle_manifest artifact identity is not exact")
    manifest, manifest_ref = _stable_json(manifest_path, workspace_root)
    if manifest_ref["sha256"] != bundle_artifact["sha256"] \
            or manifest_ref["bytes"] != bundle_artifact["bytes"] \
            or manifest.get("schema") \
            != "hawking.doctor_v5_strand_ladder_bundle.v1" \
            or manifest.get("campaign_binding") != campaign:
        raise PhaseGateError("bundle manifest content/hash binding changed")
    claims = manifest.get("claims")
    if not isinstance(claims, dict) \
            or claims.get("packed_archive_roundtrip_validated") is not True \
            or claims.get("source_deletion") is not False:
        raise PhaseGateError("bundle manifest durability/source policy is invalid")
    manifest_shards = manifest.get("shards")
    normalized_manifest: list[dict[str, Any]] = []
    if not isinstance(manifest_shards, list):
        raise PhaseGateError("bundle manifest shard inventory is invalid")
    for row in manifest_shards:
        packed = row.get("packed") if isinstance(row, dict) else None
        if not isinstance(row, dict) or not _integer(row.get("ordinal")) \
                or not isinstance(packed, dict) \
                or set(packed) != {"path", "sha256", "bytes"}:
            raise PhaseGateError("bundle manifest packed row is invalid")
        normalized_manifest.append({"ordinal": row["ordinal"], **packed})
    if normalized_manifest != packed_rows:
        raise PhaseGateError("bundle manifest and checkpoint packed inventories differ")
    observed_packed = sum(row["bytes"] for row in packed_rows)
    return {
        "request": request, "request_ref": request_ref,
        "checkpoint": checkpoint, "checkpoint_ref": checkpoint_ref,
        "manifest_ref": manifest_ref, "bundle_artifact": bundle_artifact,
        "packed_rows": packed_rows, "observed_packed_bytes": observed_packed,
    }


def _ledger_core(receipt: dict[str, Any]) -> dict[str, Any]:
    return _without(receipt, "observed_at", "receipt_sha256")


def _build_evidence(binding: PhaseGateBinding, original_gate: dict[str, Any],
                    execution_cell: dict[str, Any],
                    workspace_root: Path) -> dict[str, Any]:
    errors = _binding_errors(binding, workspace_root)
    if errors:
        raise PhaseGateError("; ".join(sorted(set(errors))))
    if _source_sha(Path(__file__).resolve()) != binding.module_sha256:
        raise PhaseGateError("phase-gate module source hash changed")
    if _source_sha(Path(ledger.__file__).resolve()) != binding.ledger_module_sha256:
        raise PhaseGateError("remaining-scratch ledger source hash changed")
    disk_blocker = _validate_predecessor_gate(original_gate, binding)
    documents = _validate_plan_and_spec(
        binding, execution_cell, workspace_root)
    candidate = _checkpoint_candidate(binding, documents, workspace_root)
    observed = candidate["observed_packed_bytes"]
    if original_gate.get("observed_current_output_bytes") != observed:
        raise PhaseGateError("predecessor output observation differs from attested shards")
    effective_projection = max(binding.frozen_projected_output_bytes, observed)
    try:
        scratch = ledger.build_ledger(
            binding.worker_request_path,
            projected_packed_output_bytes=effective_projection,
            workspace_root=workspace_root,
        )
    except (ledger.ScratchLedgerError, OSError, TypeError, ValueError) as exc:
        raise PhaseGateError(f"remaining-scratch ledger refused evidence: {exc}") from exc
    ledger_errors = ledger.validate_receipt(scratch)
    if ledger_errors:
        raise PhaseGateError("remaining-scratch ledger invalid: "
                             + "; ".join(ledger_errors))
    if scratch.get("request", {}).get("sha256") \
            != candidate["request_ref"]["sha256"] \
            or scratch.get("checkpoint", {}).get("sha256") \
            != candidate["checkpoint_ref"]["sha256"] \
            or scratch.get("disk_reserve_bytes") != binding.disk_reserve_bytes \
            or scratch.get("declared_total_scratch_bytes") \
            != binding.declared_scratch_bytes \
            or scratch.get("durable_attested_packed_bytes") != observed \
            or scratch.get("projected_whole_packed_output_bytes") \
            != effective_projection:
        raise PhaseGateError("remaining-scratch ledger byte/source binding differs")
    rows = scratch.get("ordinals")
    if not isinstance(rows, list) or not rows \
            or any(row.get("encode_completed") is not True
                   or row.get("attest_completed") is not True
                   or row.get("packed_archive_durable") is not True
                   for row in rows):
        raise PhaseGateError("ledger lacks complete durable encode+attest proof")
    manifest_relative = _relative(
        binding.worker_output_root / "bundle/manifest.json", workspace_root)
    observations = scratch.get("artifact_identity_observations")
    manifest_observation = next((row for row in observations
                                 if isinstance(row, dict)
                                 and row.get("path") == manifest_relative), None) \
        if isinstance(observations, list) else None
    if not isinstance(manifest_observation, dict) \
            or "bundle_manifest" not in manifest_observation.get(
                "checkpoint_units", []):
        raise PhaseGateError("ledger lacks stable bundle_manifest observation")

    # A second stable pass catches path swaps or checkpoint advancement between
    # the candidate derivation and the ledger traversal.
    _, plan_after = _stable_json(binding.plan_path, workspace_root)
    _, spec_after = _stable_json(binding.runtime_spec_path, workspace_root)
    _, request_after = _stable_json(binding.worker_request_path, workspace_root)
    _, checkpoint_after = _stable_json(
        binding.worker_checkpoint_path, workspace_root)
    _, manifest_after = _stable_json(
        binding.worker_output_root / "bundle/manifest.json", workspace_root)
    for name, before, after in (
            ("plan", documents["plan_ref"], plan_after),
            ("runtime spec", documents["spec_ref"], spec_after),
            ("worker request", candidate["request_ref"], request_after),
            ("worker checkpoint", candidate["checkpoint_ref"], checkpoint_after),
            ("bundle manifest", candidate["manifest_ref"], manifest_after)):
        if not _same_ref(before, after):
            raise PhaseGateError(f"{name} raced during phase-gate observation")

    required = scratch["required_free_bytes"]
    return {
        "disk_blocker": disk_blocker,
        "original_gate_sha256": _hash_value(original_gate),
        "plan_file": documents["plan_ref"],
        "plan_sha256": binding.plan_sha256,
        "plan_cell_sha256": binding.plan_cell_sha256,
        "cell_id": binding.cell_id,
        "cell_identity_sha256": binding.cell_identity_sha256,
        "runtime_spec_file": documents["spec_ref"],
        "program_spec_sha256": binding.program_spec_sha256,
        "worker_request": candidate["request_ref"],
        "worker_checkpoint": candidate["checkpoint_ref"],
        "bundle_manifest": candidate["manifest_ref"],
        "ledger_receipt_sha256": scratch["receipt_sha256"],
        "ledger_core_sha256": _hash_value(_ledger_core(scratch)),
        "ordinal_count": len(rows),
        "all_encode_attest_complete": True,
        "bundle_manifest_complete": True,
        "disk_reserve_bytes": binding.disk_reserve_bytes,
        "declared_total_scratch_bytes": binding.declared_scratch_bytes,
        "durable_materialized_bytes": scratch["durable_materialized_bytes"],
        "remaining_scratch_bytes": scratch["remaining_scratch_bytes"],
        "frozen_projected_packed_output_bytes": (
            binding.frozen_projected_output_bytes),
        "observed_durable_packed_bytes": observed,
        "effective_whole_packed_output_bytes": effective_projection,
        "projection_overrun_bytes": max(
            0, observed - binding.frozen_projected_output_bytes),
        "projected_remaining_packed_output_bytes": scratch[
            "projected_remaining_packed_output_bytes"],
        "required_free_bytes": required,
        "overrun_credit_requires_complete_durable_proof": True,
        "ram_credit_bytes": 0,
        "result_tree_write_permitted": False,
    }


def build_phase_receipt(original_gate: dict[str, Any],
                        binding: PhaseGateBinding, *,
                        execution_cell: dict[str, Any],
                        workspace_root: Path = ROOT,
                        now: dt.datetime | None = None) -> dict[str, Any]:
    """Build a receipt without writing it or mutating any campaign artifact."""
    root = Path(os.path.abspath(workspace_root))
    if not isinstance(execution_cell, dict):
        raise PhaseGateError("execution cell evidence is absent")
    evidence = _build_evidence(binding, original_gate, execution_cell, root)
    receipt: dict[str, Any] = {
        "schema": PHASE_RECEIPT_SCHEMA, "version": VERSION,
        "created_at": _timestamp(now),
        "mode": "successor-operational-disk-only",
        "evidence": evidence,
        "isolation": {
            "ram_credit_bytes": 0, "swap_policy_changed": False,
            "result_tree_write_permitted": False,
            "source_deletion_permitted": False,
            "runtime_defaults_changed": False,
        },
    }
    receipt["receipt_sha256"] = _hash_value(receipt)
    errors = validate_phase_receipt(receipt, binding)
    if errors:
        raise PhaseGateError("generated phase receipt is invalid: "
                             + "; ".join(errors))
    return receipt


def validate_phase_receipt(receipt: Any,
                           binding: PhaseGateBinding) -> list[str]:
    if not isinstance(receipt, dict) \
            or receipt.get("schema") != PHASE_RECEIPT_SCHEMA \
            or receipt.get("version") != VERSION:
        return ["phase receipt schema/version mismatch"]
    errors: list[str] = []
    if set(receipt) != {
            "schema", "version", "created_at", "mode", "evidence",
            "isolation", "receipt_sha256"}:
        errors.append("phase receipt keys are not exact")
    if receipt.get("receipt_sha256") != _hash_value(
            _without(receipt, "receipt_sha256")):
        errors.append("phase receipt self-hash mismatch")
    if receipt.get("mode") != "successor-operational-disk-only":
        errors.append("phase receipt mode changed")
    expected_isolation = {
        "ram_credit_bytes": 0, "swap_policy_changed": False,
        "result_tree_write_permitted": False,
        "source_deletion_permitted": False,
        "runtime_defaults_changed": False,
    }
    if receipt.get("isolation") != expected_isolation:
        errors.append("phase receipt isolation contract weakened")
    evidence = receipt.get("evidence")
    expected_evidence = {
        "disk_blocker", "original_gate_sha256", "plan_file", "plan_sha256",
        "plan_cell_sha256", "cell_id", "cell_identity_sha256",
        "runtime_spec_file", "program_spec_sha256", "worker_request",
        "worker_checkpoint", "bundle_manifest", "ledger_receipt_sha256",
        "ledger_core_sha256", "ordinal_count", "all_encode_attest_complete",
        "bundle_manifest_complete", "disk_reserve_bytes",
        "declared_total_scratch_bytes", "durable_materialized_bytes",
        "remaining_scratch_bytes", "frozen_projected_packed_output_bytes",
        "observed_durable_packed_bytes", "effective_whole_packed_output_bytes",
        "projection_overrun_bytes", "projected_remaining_packed_output_bytes",
        "required_free_bytes", "overrun_credit_requires_complete_durable_proof",
        "ram_credit_bytes", "result_tree_write_permitted",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_evidence:
        errors.append("phase receipt evidence keys are not exact")
        return errors
    if evidence.get("cell_id") != binding.cell_id \
            or evidence.get("cell_identity_sha256") \
            != binding.cell_identity_sha256 \
            or evidence.get("plan_sha256") != binding.plan_sha256 \
            or evidence.get("plan_cell_sha256") != binding.plan_cell_sha256 \
            or evidence.get("program_spec_sha256") != binding.program_spec_sha256:
        errors.append("phase receipt campaign binding changed")
    if any(not _valid_sha(evidence.get(field)) for field in (
            "original_gate_sha256", "ledger_receipt_sha256",
            "ledger_core_sha256")):
        errors.append("phase receipt evidence hash is invalid")
    byte_fields = (
        "disk_reserve_bytes", "declared_total_scratch_bytes",
        "durable_materialized_bytes", "remaining_scratch_bytes",
        "frozen_projected_packed_output_bytes", "observed_durable_packed_bytes",
        "effective_whole_packed_output_bytes", "projection_overrun_bytes",
        "projected_remaining_packed_output_bytes", "required_free_bytes",
        "ram_credit_bytes", "ordinal_count",
    )
    if any(not _integer(evidence.get(field)) for field in byte_fields):
        errors.append("phase receipt byte/count fields are invalid")
        return errors
    if evidence["disk_reserve_bytes"] != binding.disk_reserve_bytes \
            or evidence["declared_total_scratch_bytes"] \
            != binding.declared_scratch_bytes \
            or evidence["frozen_projected_packed_output_bytes"] \
            != binding.frozen_projected_output_bytes \
            or evidence["ram_credit_bytes"] != 0:
        errors.append("phase receipt frozen resource envelope changed")
    observed = evidence["observed_durable_packed_bytes"]
    frozen = evidence["frozen_projected_packed_output_bytes"]
    if evidence["effective_whole_packed_output_bytes"] != max(frozen, observed) \
            or evidence["projection_overrun_bytes"] != max(0, observed - frozen) \
            or evidence["remaining_scratch_bytes"] != max(
                0, evidence["declared_total_scratch_bytes"]
                - evidence["durable_materialized_bytes"]) \
            or evidence["projected_remaining_packed_output_bytes"] != max(
                0, evidence["effective_whole_packed_output_bytes"] - observed) \
            or evidence["required_free_bytes"] != (
                evidence["disk_reserve_bytes"]
                + evidence["remaining_scratch_bytes"]
                + evidence["projected_remaining_packed_output_bytes"]):
        errors.append("phase receipt accounting equation changed")
    if evidence.get("all_encode_attest_complete") is not True \
            or evidence.get("bundle_manifest_complete") is not True \
            or evidence.get("overrun_credit_requires_complete_durable_proof") \
            is not True \
            or evidence.get("result_tree_write_permitted") is not False:
        errors.append("phase receipt completion/overrun contract weakened")
    for field in ("plan_file", "runtime_spec_file", "worker_request",
                  "worker_checkpoint", "bundle_manifest"):
        ref = evidence.get(field)
        if not isinstance(ref, dict) or not _valid_sha(ref.get("sha256")):
            errors.append(f"phase receipt {field} reference is invalid")
    return errors


def persist_phase_receipt_atomic(receipt: dict[str, Any],
                                 binding: PhaseGateBinding, *,
                                 workspace_root: Path = ROOT) -> dict[str, Any]:
    """Persist only to ``policy_root/phase_receipts/<cell_id>.json``.

    This is an explicit caller action; :func:`build_phase_receipt` remains pure.
    No result/output directory is created or written.
    """
    root = Path(os.path.abspath(workspace_root))
    errors = validate_phase_receipt(receipt, binding)
    if errors:
        raise PhaseGateError("refusing to persist invalid phase receipt: "
                             + "; ".join(errors))
    binding_errors = _binding_errors(binding, root)
    if binding_errors:
        raise PhaseGateError("unsafe phase receipt binding: "
                             + "; ".join(binding_errors))
    policy_root = _absolute_confined(binding.policy_root, root)
    receipt_root = policy_root / PHASE_RECEIPT_DIRECTORY
    try:
        receipt_root.mkdir(mode=0o700, exist_ok=True)
        ledger._no_symlink_components(receipt_root, root)
        root_info = os.lstat(receipt_root)
        if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
            raise PhaseGateError("phase receipt root is not a real directory")
    except (ledger.ScratchLedgerError, OSError) as exc:
        raise PhaseGateError(f"cannot prepare phase receipt root: {exc}") from exc
    target = _absolute_confined(binding.phase_receipt_path, root)
    if target != receipt_root / f"{binding.cell_id}.json":
        raise PhaseGateError("phase receipt target is not the exact cell path")
    raw = json.dumps(receipt, indent=2, sort_keys=True,
                     ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    directory_flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                       | getattr(os, "O_NOFOLLOW", 0)
                       | getattr(os, "O_DIRECTORY", 0))
    descriptor = -1
    temporary = f".{binding.cell_id}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    temp_fd = -1
    try:
        descriptor = os.open(receipt_root, directory_flags)
        try:
            present = os.stat(target.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            present = None
        if present is not None and (stat.S_ISLNK(present.st_mode)
                                    or not stat.S_ISREG(present.st_mode)
                                    or present.st_nlink != 1):
            raise PhaseGateError("existing phase receipt target is unsafe")
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
        temp_fd = os.open(temporary, flags, 0o600, dir_fd=descriptor)
        offset = 0
        while offset < len(raw):
            written = os.write(temp_fd, raw[offset:])
            if written <= 0:
                raise OSError("short phase receipt write")
            offset += written
        os.fsync(temp_fd)
        os.close(temp_fd); temp_fd = -1
        os.replace(temporary, target.name,
                   src_dir_fd=descriptor, dst_dir_fd=descriptor)
        os.fsync(descriptor)
    except (OSError, PhaseGateError) as exc:
        if temp_fd >= 0:
            os.close(temp_fd)
        if descriptor >= 0:
            try:
                os.unlink(temporary, dir_fd=descriptor)
            except OSError:
                pass
        if isinstance(exc, PhaseGateError):
            raise
        raise PhaseGateError(f"atomic phase receipt write failed: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    persisted, ref = _stable_json(target, root)
    if persisted != receipt or ref["sha256"] != hashlib.sha256(raw).hexdigest():
        raise PhaseGateError("persisted phase receipt verification failed")
    return ref


def _receipt_core(receipt: dict[str, Any]) -> dict[str, Any]:
    return _without(receipt, "created_at", "receipt_sha256")


def _direct_free_bytes(workspace_root: Path,
                       statvfs_fn: Callable[[os.PathLike[str]], Any]) -> int:
    try:
        snapshot = statvfs_fn(workspace_root)
        available = snapshot.f_bavail
        fragment = snapshot.f_frsize
    except Exception as exc:
        raise PhaseGateError(f"direct statvfs probe failed: {exc}") from exc
    if not _integer(available) or not _integer(fragment, minimum=1):
        raise PhaseGateError("direct statvfs fields are invalid")
    free = available * fragment
    if not _integer(free):
        raise PhaseGateError("direct statvfs byte result is invalid")
    return free


def _relaxed_gate(original_gate: dict[str, Any], receipt: dict[str, Any],
                  receipt_ref: dict[str, Any], direct_free: int,
                  binding: PhaseGateBinding) -> dict[str, Any]:
    evidence = receipt["evidence"]
    gate = copy.deepcopy(original_gate)
    gate["blockers"] = [row for row in original_gate["blockers"]
                        if row != evidence["disk_blocker"]]
    gate["ok"] = not gate["blockers"]
    gate["capacity_ok"] = True
    gate["scratch_bytes"] = evidence["remaining_scratch_bytes"]
    gate["projected_incremental_output_bytes"] = evidence[
        "projected_remaining_packed_output_bytes"]
    gate["required_free_bytes"] = evidence["required_free_bytes"]
    resident = original_gate.get("resident_payload_bytes", 0)
    if _integer(resident):
        gate["available_total_capacity_bytes"] = direct_free + resident
        gate["required_total_capacity_bytes"] = evidence["required_free_bytes"] + resident
    gate["phase_aware_disk_gate"] = {
        "schema": PHASE_GATE_API_SCHEMA,
        "applied": True, "disk_only": True, "ram_credit_bytes": 0,
        "direct_statvfs_free_bytes": direct_free,
        "declared_total_scratch_bytes": binding.declared_scratch_bytes,
        "remaining_scratch_bytes": evidence["remaining_scratch_bytes"],
        "frozen_projected_output_bytes": binding.frozen_projected_output_bytes,
        "observed_durable_packed_bytes": evidence["observed_durable_packed_bytes"],
        "effective_whole_packed_output_bytes": evidence[
            "effective_whole_packed_output_bytes"],
        "required_free_bytes": evidence["required_free_bytes"],
        "receipt_path": receipt_ref["path"],
        "receipt_file_sha256": receipt_ref["sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "result_tree_written": False,
    }
    return gate


def evaluate_persisted_phase_receipt(
        original_gate: dict[str, Any], binding: PhaseGateBinding, *,
        execution_cell: dict[str, Any], persisted_receipt_path: Path,
        workspace_root: Path = ROOT, now: dt.datetime | None = None,
        _statvfs_fn: Callable[[os.PathLike[str]], Any] = os.statvfs) \
        -> PhaseGateResult:
    """Apply only a fresh, persisted, fully recomputed disk-only receipt.

    Every failure returns a deep copy equal to ``original_gate``.  The private
    ``_statvfs_fn`` seam exists solely for deterministic tests; installed
    production wrappers always use ``os.statvfs``.
    """
    fallback = copy.deepcopy(original_gate)
    reasons: list[str] = []
    root = Path(os.path.abspath(workspace_root))
    receipt_ref: dict[str, Any] | None = None
    try:
        expected_path = _absolute_confined(binding.phase_receipt_path, root)
        actual_path = _absolute_confined(persisted_receipt_path, root)
        if actual_path != expected_path:
            raise PhaseGateError("persisted receipt path is not the exact policy path")
        receipt, receipt_ref = _stable_json(actual_path, root)
        errors = validate_phase_receipt(receipt, binding)
        if errors:
            raise PhaseGateError("persisted phase receipt is invalid: "
                                 + "; ".join(errors))
        observed_now = now or _now()
        if not isinstance(observed_now, dt.datetime) or observed_now.tzinfo is None:
            raise PhaseGateError("current phase-gate time is naive or invalid")
        age = (observed_now.astimezone(dt.timezone.utc)
               - _parse_timestamp(receipt["created_at"])).total_seconds()
        if age < -5.0 or age > MAX_RECEIPT_AGE_SECONDS:
            raise PhaseGateError("persisted phase receipt is stale or future-dated")
        recomputed = build_phase_receipt(
            original_gate, binding, execution_cell=execution_cell,
            workspace_root=root, now=observed_now)
        if _receipt_core(receipt) != _receipt_core(recomputed):
            raise PhaseGateError("persisted phase receipt differs from fresh recomputation")
        direct_free = _direct_free_bytes(root, _statvfs_fn)
        required = receipt["evidence"]["required_free_bytes"]
        if direct_free < required:
            raise PhaseGateError("direct statvfs remains below phase-aware requirement")
        gate = _relaxed_gate(original_gate, receipt, receipt_ref,
                             direct_free, binding)
        return PhaseGateResult(gate=gate, applied=True, reasons=(),
                               persisted_receipt_ref=receipt_ref)
    except (PhaseGateError, ledger.ScratchLedgerError, OSError, TypeError,
            ValueError, KeyError) as exc:
        reasons.append(str(exc))
    return PhaseGateResult(gate=fallback, applied=False,
                           reasons=tuple(sorted(set(reasons))),
                           persisted_receipt_ref=receipt_ref)


def _bindings_from_policy(successor_policy: Any, policy_root: Path,
                          workspace_root: Path) -> tuple[PhaseGateBinding, ...]:
    if not isinstance(successor_policy, dict):
        raise PhaseGateError("successor policy is not an object")
    config = successor_policy.get(PHASE_POLICY_KEY)
    if not isinstance(config, dict) or set(config) != {
            "schema", "enabled", "module_sha256", "ledger_module_sha256",
            "ram_credit_bytes", "bindings"}:
        raise PhaseGateError("successor phase-gate policy keys are not exact")
    if config.get("schema") != PHASE_GATE_API_SCHEMA \
            or config.get("enabled") is not True \
            or config.get("ram_credit_bytes") != 0:
        raise PhaseGateError("successor phase-gate policy is disabled or grants RAM")
    rows = config.get("bindings")
    expected = {
        "plan_path", "plan_file_sha256", "plan_sha256", "plan_cell_sha256",
        "cell_id", "cell_identity_sha256", "runtime_spec_path",
        "runtime_spec_file_sha256", "program_spec_sha256",
        "execution_output_root", "disk_reserve_bytes",
        "declared_scratch_bytes", "frozen_projected_output_bytes",
    }
    if not isinstance(rows, list) or not rows:
        raise PhaseGateError("successor phase-gate bindings are empty")
    bindings: list[PhaseGateBinding] = []
    cell_ids: set[str] = set()
    for values in rows:
        if not isinstance(values, dict) or set(values) != expected:
            raise PhaseGateError("successor phase-gate binding row is not exact")
        binding = PhaseGateBinding(
            plan_path=_absolute_confined(values["plan_path"], workspace_root),
            plan_file_sha256=values["plan_file_sha256"],
            plan_sha256=values["plan_sha256"],
            plan_cell_sha256=values["plan_cell_sha256"],
            cell_id=values["cell_id"],
            cell_identity_sha256=values["cell_identity_sha256"],
            runtime_spec_path=_absolute_confined(
                values["runtime_spec_path"], workspace_root),
            runtime_spec_file_sha256=values["runtime_spec_file_sha256"],
            program_spec_sha256=values["program_spec_sha256"],
            execution_output_root=_absolute_confined(
                values["execution_output_root"], workspace_root),
            policy_root=_absolute_confined(policy_root, workspace_root),
            disk_reserve_bytes=values["disk_reserve_bytes"],
            declared_scratch_bytes=values["declared_scratch_bytes"],
            frozen_projected_output_bytes=values["frozen_projected_output_bytes"],
            module_sha256=config["module_sha256"],
            ledger_module_sha256=config["ledger_module_sha256"],
            ram_credit_bytes=config["ram_credit_bytes"],
        )
        if binding.cell_id in cell_ids:
            raise PhaseGateError("successor phase-gate cell binding is duplicate")
        cell_ids.add(binding.cell_id)
        errors = _binding_errors(
            binding, workspace_root, require_worker_evidence=False)
        if errors:
            raise PhaseGateError("invalid successor phase-gate binding: "
                                 + "; ".join(sorted(set(errors))))
        bindings.append(binding)
    if _source_sha(Path(__file__).resolve()) != config["module_sha256"] \
            or _source_sha(Path(ledger.__file__).resolve()) \
            != config["ledger_module_sha256"]:
        raise PhaseGateError("successor phase-gate module hash binding changed")
    return tuple(bindings)


def _nonpermissive_diagnostic(gate: dict[str, Any], reasons: tuple[str, ...]) \
        -> dict[str, Any]:
    result = copy.deepcopy(gate)
    # Only append diagnostics to the known flexible Doctor gate.  Admission
    # fields and blockers remain byte-for-byte equal to the predecessor values.
    if result.get("schema") == "hawking.doctor_v5_ultra_resource_gate.v1" \
            and "phase_aware_disk_gate_diagnostic" not in result:
        diagnostic: dict[str, Any] = {
            "schema": PHASE_DIAGNOSTIC_SCHEMA, "version": VERSION,
            "applied": False, "nonpermissive": True,
            "predecessor_admission_unchanged": True,
            "reasons": list(reasons) or ["phase gate was not applicable"],
        }
        diagnostic["diagnostic_sha256"] = _hash_value(diagnostic)
        result["phase_aware_disk_gate_diagnostic"] = diagnostic
    return result


def install_phase_gate(base_module: ModuleType, predecessor_gate: Callable[..., Any],
                       successor_policy: dict[str, Any], policy_root: Path) \
        -> Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]]:
    """Return an exact-signature successor wrapper around the predecessor gate.

    ``policy_root`` is authoritative for operational writes.  The only allowed
    target is ``policy_root/phase_receipts/<cell_id>.json``.
    """
    if not callable(predecessor_gate):
        raise PhaseGateError("predecessor gate is not callable")
    try:
        parameters = list(inspect.signature(predecessor_gate).parameters)
    except (TypeError, ValueError) as exc:
        raise PhaseGateError(f"cannot inspect predecessor signature: {exc}") from exc
    if parameters != ["plan", "state", "execution"]:
        raise PhaseGateError("predecessor gate signature is not (plan,state,execution)")
    module_root = getattr(base_module, "ROOT", None)
    if not isinstance(module_root, Path):
        raise PhaseGateError("base module ROOT is absent or invalid")
    workspace_root = Path(os.path.abspath(module_root))
    bindings = _bindings_from_policy(
        successor_policy, Path(policy_root), workspace_root)
    bindings_by_cell = {row.cell_id: row for row in bindings}

    @functools.wraps(predecessor_gate)
    def phase_gate(plan: dict[str, Any], state: dict[str, Any],
                   execution: dict[str, Any]) -> dict[str, Any]:
        predecessor = predecessor_gate(plan, state, execution)
        try:
            if not isinstance(execution, dict) \
                    or not isinstance(execution.get("cell"), dict):
                return predecessor
            binding = bindings_by_cell.get(execution["cell"].get("cell_id"))
            if binding is None:
                return predecessor
            if not isinstance(plan, dict) or plan.get("plan_sha256") \
                    != binding.plan_sha256:
                raise PhaseGateError("in-memory campaign plan binding changed")
            if _hash_value(execution["cell"]) != binding.plan_cell_sha256:
                raise PhaseGateError("in-memory execution cell binding changed")
            output_dir = execution.get("output_dir")
            if not isinstance(output_dir, Path) \
                    or _absolute_confined(output_dir, workspace_root) \
                    != binding.execution_output_root:
                raise PhaseGateError("execution output root binding changed")
            if execution.get("scratch_bytes") != binding.declared_scratch_bytes:
                raise PhaseGateError("execution scratch binding changed")
            runtime = execution.get("runtime")
            if not isinstance(runtime, dict) \
                    or runtime.get("path") != binding.runtime_spec_path \
                    or runtime.get("sha256") != binding.runtime_spec_file_sha256 \
                    or not isinstance(runtime.get("document"), dict) \
                    or runtime["document"].get("program_spec_sha256") \
                    != binding.program_spec_sha256:
                raise PhaseGateError("execution runtime-spec binding changed")
            receipt = build_phase_receipt(
                predecessor, binding, execution_cell=execution["cell"],
                workspace_root=workspace_root)
            persist_phase_receipt_atomic(
                receipt, binding, workspace_root=workspace_root)
            result = evaluate_persisted_phase_receipt(
                predecessor, binding, execution_cell=execution["cell"],
                persisted_receipt_path=binding.phase_receipt_path,
                workspace_root=workspace_root)
            if result.applied:
                return result.gate
            return _nonpermissive_diagnostic(predecessor, result.reasons)
        except (PhaseGateError, ledger.ScratchLedgerError, OSError, TypeError,
                ValueError, KeyError) as exc:
            return _nonpermissive_diagnostic(predecessor, (str(exc),))

    phase_gate.phase_gate_api_schema = PHASE_GATE_API_SCHEMA  # type: ignore[attr-defined]
    phase_gate.phase_gate_bindings = bindings  # type: ignore[attr-defined]
    return phase_gate

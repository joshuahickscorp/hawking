#!/usr/bin/env python3.12
"""Build the authoritative expected contract for the official GLM-5.2 campaign.

This module is deliberately offline.  It reads the already-sealed source, tensor,
dependency, schedule, admission, adapter, parity, corpus, and Xet-plan artifacts;
cross-checks their identities and coverage; and translates the preliminary schedule
into the exact schema consumed by :mod:`glm52_state`.

No official contract is emitted without an explicit domain-separated Telegram chat
identity digest and an explicit deterministic ``created_at``.  A raw chat id is never
accepted.  ``preflight`` is read-only and reports the missing digest as a hard blocker.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_contract as source_contract  # noqa: E402
import glm52_state as state  # noqa: E402
from glm52_common import (  # noqa: E402
    Glm52Error,
    REPO_ROOT,
    atomic_json,
    canonical,
    seal,
    verify_sealed,
)


PREFLIGHT_SCHEMA = "hawking.glm52.expected_campaign_contract_preflight.v1"
OUTPUT_FILENAME = "GLM52_EXPECTED_CAMPAIGN_CONTRACT.json"
OFFICIAL_REPO = source_contract.REPO_ID
OFFICIAL_REVISION = source_contract.REVISION
EXACT_SHARDS = source_contract.EXPECTED_SHARDS
EXACT_TENSORS = source_contract.EXPECTED_TENSORS
EXACT_WINDOWS = 20
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class CampaignContractError(Glm52Error):
    """A frozen input or expected-contract invariant failed."""


@dataclass(frozen=True)
class InputSpec:
    key: str
    filename: str
    schema: str
    status: str


INPUT_SPECS: tuple[InputSpec, ...] = (
    InputSpec(
        "handoff_precheck",
        "GLM52_HANDOFF_PRECHECK.json",
        "hawking.glm52.handoff_precheck.v1",
        "PASS_WITH_SECURITY_AND_ROLLBACK_EXCEPTIONS",
    ),
    InputSpec(
        "kimi_source_release",
        "KIMI_K26_SOURCE_RELEASE_FOR_GLM52.json",
        "hawking.kimi_k26.source_release_for_glm52.v1",
        "RECONCILED_ALREADY_RELEASED",
    ),
    InputSpec(
        "gravity_pre_audit",
        "GRAVITY_COMPLETENESS_AUDIT_GLM52_PRE.json",
        "hawking.gravity_completeness_audit.glm52_pre.v1",
        "PASS_FROZEN_PRE_CAMPAIGN_BASELINE",
    ),
    InputSpec(
        "external_baseline_matrix",
        "GRAVITY_EXTERNAL_BASELINE_MATRIX.json",
        "hawking.gravity_external_baseline_matrix.v1",
        "PASS_PRIMARY_SOURCE_COMPARISON",
    ),
    InputSpec(
        "official_manifest",
        "GLM52_OFFICIAL_MANIFEST.json",
        "hawking.glm52.official_manifest.v1",
        "PASS_CONTROL_PLANE_AND_HEADERS_BODY_PENDING",
    ),
    InputSpec(
        "source_format_ledger",
        "GLM52_SOURCE_FORMAT_LEDGER.json",
        "hawking.glm52.source_format_ledger.v1",
        "PASS_HEADER_DERIVED_BODY_PENDING",
    ),
    InputSpec(
        "architecture_contract",
        "GLM52_ARCHITECTURE_CONTRACT.json",
        "hawking.glm52.architecture_contract.v1",
        "PASS_CONFIG_INDEX_AND_HEADERS",
    ),
    InputSpec(
        "logical_weight_ledger",
        "GLM52_LOGICAL_WEIGHT_LEDGER.json",
        "hawking.glm52.logical_weight_ledger.v1",
        "PASS_HEADER_DERIVED",
    ),
    InputSpec(
        "dependency_graph",
        "GLM52_SHARD_DEPENDENCY_GRAPH.json",
        "hawking.glm52.shard_dependency_graph.v1",
        "PASS_HEADER_DERIVED",
    ),
    InputSpec(
        "streaming_schedule",
        "GLM52_STREAMING_SCHEDULE_PRE_AUTOTUNE.json",
        "hawking.glm52.streaming_schedule.v1",
        "PRELIMINARY_DEPENDENCY_COMPLETE_PENDING_XET_AUTOTUNE",
    ),
    InputSpec(
        "source_admission",
        "GLM52_SOURCE_ADMISSION.json",
        "hawking.glm52.source_admission.v1",
        "ADMITTED_CONTROL_PLANE_HEADERS_AND_PLAN_BODY_PENDING",
    ),
    InputSpec(
        "adapter_twin",
        "GLM52_ADAPTER_TWIN.json",
        "hawking.glm52.adapter_twin.v1",
        "PASS_SYNTHETIC_TWIN_AND_OFFICIAL_HEADER_TOKENIZER_SCHEMA",
    ),
    InputSpec(
        "reference_parity",
        "GLM52_REFERENCE_PARITY.json",
        "hawking.glm52.reference_parity.v1",
        "PASS_SYNTHETIC_MAIN_AND_MTP_SELF_CONSISTENCY_SOURCE_PARENT_PENDING",
    ),
    InputSpec(
        "corpus_integrity",
        "GLM52_CORPUS_INTEGRITY.json",
        "hawking.glm52.corpus_integrity.v2",
        "PASS",
    ),
    InputSpec(
        "xet_autotune_plan",
        "GLM52_XET_AUTOTUNE_PLAN.json",
        "hawking.glm52.xet_autotune_plan.v2",
        "PASS_OFFLINE_PLAN_BODY_NOT_READ",
    ),
)
INPUT_BY_KEY = {spec.key: spec for spec in INPUT_SPECS}
INPUT_BY_FILENAME = {spec.filename: spec for spec in INPUT_SPECS}
INPUT_BY_FILENAME["GLM52_STREAMING_SCHEDULE.json"] = INPUT_BY_KEY["streaming_schedule"]


# These names are also the artifact labels consumed by glm52_state's mandatory
# official ASSEMBLE gate.  Current input seals are filled in by _state_gates().
ASSEMBLE_ARTIFACT_PATHS: dict[str, str] = {
    "handoff_precheck": "GLM52_HANDOFF_PRECHECK.json",
    "kimi_source_release": "KIMI_K26_SOURCE_RELEASE_FOR_GLM52.json",
    "official_source_manifest": "GLM52_OFFICIAL_MANIFEST.json",
    "source_format_ledger": "GLM52_SOURCE_FORMAT_LEDGER.json",
    "architecture_contract": "GLM52_ARCHITECTURE_CONTRACT.json",
    "logical_weight_ledger": "GLM52_LOGICAL_WEIGHT_LEDGER.json",
    "tensor_coverage_ledger": "GLM52_SHARD_DEPENDENCY_GRAPH.json",
    "streaming_schedule": "GLM52_STREAMING_SCHEDULE.json",
    "preliminary_streaming_schedule": "GLM52_STREAMING_SCHEDULE_PRE_AUTOTUNE.json",
    "source_admission": "GLM52_SOURCE_ADMISSION.json",
    "gravity_pre_audit": "GRAVITY_COMPLETENESS_AUDIT_GLM52_PRE.json",
    "external_baseline_matrix": "GRAVITY_EXTERNAL_BASELINE_MATRIX.json",
    "xet_autotune_plan": "GLM52_XET_AUTOTUNE_PLAN.json",
    "xet_autotune_result": "GLM52_XET_AUTOTUNE.json",
    "adapter_twin": "GLM52_ADAPTER_TWIN.json",
    "reference_parity": "GLM52_REFERENCE_PARITY.json",
    "bf16_reference_forward": "GLM52_BF16_REFERENCE_FORWARD.json",
    "corpus_integrity": "GLM52_CORPUS_INTEGRITY.json",
    "oracle_bandwidth": "GLM52_ORACLE_BANDWIDTH.json",
    "causal_atlas": "GLM52_CAUSAL_ATLAS.json",
    "frozen_program": "GLM52_FROZEN_GRAVITY_PROGRAM.json",
    "doctor_program": "GLM52_DOCTOR_PROGRAM.json",
    "rate_program": "GLM52_RATE_PROGRAM.json",
}


COMPLETE_ARTIFACT_PATHS: dict[str, str] = {
    **ASSEMBLE_ARTIFACT_PATHS,
    "compact_artifact_manifest": "GLM52_COMPACT_ARTIFACT_MANIFEST.json",
    "source_to_compact_coverage": "GLM52_SOURCE_TO_COMPACT_COVERAGE.json",
    "full_compact_results": "GLM52_FULL_COMPACT_RESULTS.json",
    "capability_results": "GLM52_CAPABILITY_RESULTS.json",
    "source_eviction_report": "GLM52_SOURCE_EVICTION_REPORT.json",
    "gravity_completeness_audit_final": "GRAVITY_COMPLETENESS_AUDIT_GLM52_POST.json",
    "byte_auction": "GLM52_FINAL_BYTE_AUCTION.json",
    "glm52_gravity_final": "GLM52_GRAVITY_FINAL.json",
    "terminal_outcome": "GLM52_GRAVITY_FINAL.json",
    "rollback_transfer": "GLM52_ROLLBACK.json",
}


FUTURE_ARTIFACT_POLICIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "GLM52_STREAMING_SCHEDULE.json": (
        "hawking.glm52.streaming_schedule.v2", ("FROZEN_AFTER_XET_AUTOTUNE",)
    ),
    "GLM52_XET_AUTOTUNE.json": (
        "hawking.glm52.xet_autotune_result.v1",
        ("PASS_LIVE_XET_AUTOTUNE_COMPLETE_SCHEDULE_REFREEZE_REQUIRED",),
    ),
    "GLM52_BF16_REFERENCE_FORWARD.json": (
        "hawking.glm52.bf16_reference_forward.v1", ("PASS", "GREEN")
    ),
    "GLM52_ORACLE_BANDWIDTH.json": (
        "hawking.glm52.oracle_bandwidth.v1", ("PASS", "GREEN")
    ),
    "GLM52_CAUSAL_ATLAS.json": (
        "hawking.glm52.causal_atlas.v1", ("PASS", "GREEN")
    ),
    "GLM52_FROZEN_GRAVITY_PROGRAM.json": (
        "hawking.glm52.frozen_gravity_program.v1", ("FROZEN", "PASS")
    ),
    "GLM52_DOCTOR_PROGRAM.json": (
        "hawking.glm52.doctor_program.v1", ("FROZEN", "PASS")
    ),
    "GLM52_RATE_PROGRAM.json": (
        "hawking.glm52.rate_program.v1", ("FROZEN", "PASS")
    ),
    "GLM52_COMPACT_ARTIFACT_MANIFEST.json": (
        "hawking.glm52.compact_artifact_manifest.v1", ("SEALED", "PASS")
    ),
    "GLM52_SOURCE_TO_COMPACT_COVERAGE.json": (
        "hawking.glm52.source_to_compact_coverage.v1", ("COMPLETE", "PASS")
    ),
    "GLM52_FULL_COMPACT_RESULTS.json": (
        "hawking.glm52.full_compact_results.v1", ("COMPLETE", "PASS")
    ),
    "GLM52_CAPABILITY_RESULTS.json": (
        "hawking.glm52.capability_results.v1", ("COMPLETE", "PASS")
    ),
    "GLM52_SOURCE_EVICTION_REPORT.json": (
        "hawking.glm52.source_eviction_report.v1", ("COMPLETE", "PASS")
    ),
    "GLM52_PHONE_STATUS.json": (
        "hawking.glm52.phone_status.v2", ("GREEN",)
    ),
    "GRAVITY_COMPLETENESS_AUDIT_GLM52_POST.json": (
        "hawking.gravity_completeness_audit.glm52_post.v1", ("PASS", "COMPLETE")
    ),
    "GLM52_FINAL_BYTE_AUCTION.json": (
        "hawking.glm52.final_byte_auction.v1", ("COMPLETE", "PASS")
    ),
    "GLM52_GRAVITY_FINAL.json": (
        "hawking.glm52.gravity_final.v1", ("COMPLETE", "PASS")
    ),
    "GLM52_ROLLBACK.json": (
        "hawking.glm52.rollback.v1", ("COMPLETE", "PASS", "READY")
    ),
}


@dataclass(frozen=True)
class InputBundle:
    root: Path
    artifacts: dict[str, dict[str, Any]]
    file_bytes_sha256: dict[str, str]


@dataclass(frozen=True)
class DerivedContractInputs:
    source_shards: list[dict[str, Any]]
    tensor_names: list[str]
    window_schedule: list[dict[str, Any]]
    input_seals: dict[str, str]


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
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
        raise CampaignContractError(f"invalid strict JSON in {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignContractError(f"{label} is not a JSON object")
    return value


def _sha256_bytes(raw: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def load_inputs(root: str | os.PathLike[str] = REPO_ROOT) -> InputBundle:
    base = Path(root).resolve(strict=False)
    artifacts: dict[str, dict[str, Any]] = {}
    byte_hashes: dict[str, str] = {}
    for spec in INPUT_SPECS:
        path = base / spec.filename
        if path.is_symlink() or not path.is_file():
            raise CampaignContractError(f"required sealed input is missing/unsafe: {spec.filename}")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise CampaignContractError(f"cannot read required input {spec.filename}: {exc}") from exc
        value = _strict_json_bytes(raw, label=spec.filename)
        try:
            verify_sealed(value, label=spec.filename)
        except Glm52Error as exc:
            raise CampaignContractError(str(exc)) from exc
        if value.get("schema") != spec.schema or value.get("status") != spec.status:
            raise CampaignContractError(
                f"{spec.filename} schema/status differs from the frozen input contract"
            )
        if value.get("repo") not in (None, OFFICIAL_REPO) \
                or value.get("revision") not in (None, OFFICIAL_REVISION):
            raise CampaignContractError(f"{spec.filename} official identity mismatch")
        artifacts[spec.key] = value
        byte_hashes[spec.filename] = _sha256_bytes(raw)
    return InputBundle(base, artifacts, byte_hashes)


def assert_inputs_stable(bundle: InputBundle) -> None:
    """Reject an optimistic read if any frozen input changed after loading."""
    for spec in INPUT_SPECS:
        path = bundle.root / spec.filename
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise CampaignContractError(f"cannot re-read required input {spec.filename}: {exc}") from exc
        if _sha256_bytes(raw) != bundle.file_bytes_sha256[spec.filename]:
            raise CampaignContractError(f"frozen input changed during contract build: {spec.filename}")


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CampaignContractError(f"{label} must be a positive integer")
    return value


def _string_list(value: Any, label: str, *, unique: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise CampaignContractError(f"{label} must be a string list")
    if unique and len(value) != len(set(value)):
        raise CampaignContractError(f"{label} contains duplicates")
    return list(value)


def _validate_admission_bindings(artifacts: Mapping[str, Mapping[str, Any]]) -> None:
    admission = artifacts["source_admission"]
    expected_evidence = {
        "official_manifest_seal_sha256": artifacts["official_manifest"]["seal_sha256"],
        "source_format_ledger_seal_sha256": artifacts["source_format_ledger"]["seal_sha256"],
        "architecture_contract_seal_sha256": artifacts["architecture_contract"]["seal_sha256"],
        "logical_weight_ledger_seal_sha256": artifacts["logical_weight_ledger"]["seal_sha256"],
        "dependency_graph_seal_sha256": artifacts["dependency_graph"]["seal_sha256"],
        "streaming_schedule_seal_sha256": artifacts["streaming_schedule"]["seal_sha256"],
    }
    if admission.get("evidence") != expected_evidence:
        raise CampaignContractError("source admission does not bind the exact source ledgers")
    gates = admission.get("admission_gates")
    required_true = {
        "all_index_tensors_mapped_to_verified_headers",
        "current_xet_stack",
        "dependency_schedule_complete",
        "exact_logical_weight_denominator",
        "immutable_revision",
        "license_verified",
        "official_file_manifest_complete",
    }
    if not isinstance(gates, dict) or any(gates.get(key) is not True for key in required_true) \
            or gates.get("unknown_layouts") != 0 or gates.get("body_stream") is not False:
        raise CampaignContractError("source admission gates do not match the header-only freeze boundary")
    source = admission.get("source")
    if not isinstance(source, dict) \
            or source.get("weight_shards") != EXACT_SHARDS \
            or source.get("logical_weight_denominator") != 753_329_940_480 \
            or source.get("weight_container_bytes") != state.OFFICIAL_WEIGHT_LOGICAL_BYTES:
        raise CampaignContractError("source admission exact accounting mismatch")


def _validate_xet_plan_bindings(artifacts: Mapping[str, Mapping[str, Any]]) -> None:
    plan = artifacts["xet_autotune_plan"]
    expected_names = {
        "GLM52_OFFICIAL_MANIFEST.json",
        "GLM52_SOURCE_FORMAT_LEDGER.json",
        "GLM52_SHARD_DEPENDENCY_GRAPH.json",
        "GLM52_STREAMING_SCHEDULE_PRE_AUTOTUNE.json",
        "GLM52_SOURCE_ADMISSION.json",
        "GLM52_ADAPTER_TWIN.json",
        "GLM52_REFERENCE_PARITY.json",
        "GLM52_CORPUS_INTEGRITY.json",
    }
    rows = plan.get("inputs")
    if not isinstance(rows, list) or len(rows) != len(expected_names):
        raise CampaignContractError("Xet plan input inventory is incomplete")
    by_name = {
        row.get("path"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    if set(by_name) != expected_names:
        raise CampaignContractError("Xet plan input paths differ from the frozen inventory")
    for filename in expected_names:
        spec = INPUT_BY_FILENAME[filename]
        actual = artifacts[spec.key]
        if by_name[filename] != {
            "path": filename,
            "schema": spec.schema,
            "status": spec.status,
            "seal_sha256": actual["seal_sha256"],
        }:
            raise CampaignContractError(f"Xet plan has a stale or malformed binding: {filename}")
    boundary = plan.get("body_read_boundary")
    authority = plan.get("execution_authority")
    claims = plan.get("claims")
    if not isinstance(boundary, dict) or boundary.get("planner_network_access") is not False \
            or boundary.get("planner_model_body_bytes_read") != 0 \
            or boundary.get("planner_cli_live_execution_implemented") is not False \
            or boundary.get("separate_live_executor_implemented") is not True \
            or boundary.get("separate_live_executor_path") != \
            "tools/condense/glm52_xet_live.py" \
            or boundary.get("offline_plan_alone_authorizes_execution") is not False:
        raise CampaignContractError("Xet plan crossed its offline body-read boundary")
    if not isinstance(authority, dict) or authority.get("current_plan_authorizes_execution") is not False:
        raise CampaignContractError("offline Xet plan unexpectedly authorizes execution")
    if not isinstance(claims, dict) or claims.get("source_shards_fetched") != 0 \
            or claims.get("source_stream_started") is not False \
            or claims.get("xet_autotune_complete") is not False:
        raise CampaignContractError("Xet plan overclaims live execution")


def _validate_closure_inputs(artifacts: Mapping[str, Mapping[str, Any]]) -> None:
    handoff = artifacts["handoff_precheck"]
    decision = handoff.get("admission_decision")
    release_summary = handoff.get("kimi_source_release")
    process = handoff.get("process_state")
    if not isinstance(decision, dict) or decision.get(
        "safe_to_begin_local_glm_metadata_and_implementation"
    ) is not True or decision.get("safe_to_start_bf16_stream") is not False:
        raise CampaignContractError("handoff precheck does not preserve its pre-stream boundary")
    if not isinstance(release_summary, dict) \
            or release_summary.get("state") != "ALREADY_RELEASED_AND_RECONCILED" \
            or not isinstance(process, dict) or process.get("matching_process_count") != 0:
        raise CampaignContractError("handoff precheck does not prove the reconciled Kimi handoff")
    release = artifacts["kimi_source_release"]
    if release.get("closure", {}).get("terminal_outcome") != "OUTCOME_C" \
            or release.get("source", {}).get("exists_now") is not False \
            or release.get("live_absence", {}).get("source_absent") is not True \
            or release.get("live_absence", {}).get("queue_or_outbox_possible") is not False:
        raise CampaignContractError("Kimi source release evidence is incomplete")
    audit = artifacts["gravity_pre_audit"]
    if audit.get("scoring", {}).get("axis_count") != 21 \
            or audit.get("snapshot", {}).get("later_glm_artifacts_excluded_from_scores") is not True:
        raise CampaignContractError("Gravity pre-audit is not the frozen pre-campaign baseline")
    matrix = artifacts["external_baseline_matrix"]
    if not isinstance(matrix.get("methods"), list) or len(matrix["methods"]) != 13 \
            or "First sub-1-bit PTQ." not in matrix.get("claim_policy", {}).get("unsafe", []):
        raise CampaignContractError("external baseline matrix claim boundary is incomplete")


def validate_and_derive(
    artifacts: Mapping[str, Mapping[str, Any]],
) -> DerivedContractInputs:
    if set(artifacts) != set(INPUT_BY_KEY):
        missing = sorted(set(INPUT_BY_KEY) - set(artifacts))
        extra = sorted(set(artifacts) - set(INPUT_BY_KEY))
        raise CampaignContractError(f"official input inventory mismatch: missing={missing} extra={extra}")
    for key, spec in INPUT_BY_KEY.items():
        value = artifacts[key]
        try:
            verify_sealed(dict(value), label=spec.filename)
        except Glm52Error as exc:
            raise CampaignContractError(str(exc)) from exc
        if value.get("schema") != spec.schema or value.get("status") != spec.status:
            raise CampaignContractError(f"{spec.filename} schema/status mismatch")

    manifest = artifacts["official_manifest"]
    files = manifest.get("files")
    if not isinstance(files, list):
        raise CampaignContractError("official manifest files are absent")
    weight_rows = [row for row in files if isinstance(row, dict) and row.get("is_weight") is True]
    expected_paths = {
        f"model-{index:05d}-of-00282.safetensors"
        for index in range(1, EXACT_SHARDS + 1)
    }
    if manifest.get("repo") != OFFICIAL_REPO or manifest.get("revision") != OFFICIAL_REVISION \
            or manifest.get("license") != "MIT" or manifest.get("weight_shards") != EXACT_SHARDS \
            or len(weight_rows) != EXACT_SHARDS \
            or manifest.get("weight_container_logical_bytes") != state.OFFICIAL_WEIGHT_LOGICAL_BYTES:
        raise CampaignContractError("official manifest identity/count/byte contract mismatch")
    if {row.get("path") for row in weight_rows} != expected_paths:
        raise CampaignContractError("official manifest weight shard names are incomplete")
    source_shards: list[dict[str, Any]] = []
    manifest_by_path: dict[str, Mapping[str, Any]] = {}
    for row in sorted(weight_rows, key=lambda item: str(item.get("path"))):
        path = row.get("path")
        logical_bytes = _positive_int(row.get("logical_bytes"), f"manifest {path}.logical_bytes")
        xet_hash = row.get("xet_hash")
        if row.get("role") != "WEIGHT_SHARD" or row.get("referenced_by_index") is not True \
                or not isinstance(xet_hash, str) or _SHA256_RE.fullmatch(xet_hash) is None:
            raise CampaignContractError(f"manifest shard identity is incomplete: {path}")
        manifest_by_path[str(path)] = row
        source_shards.append({
            "path": path,
            "logical_bytes": logical_bytes,
            "content_hash": xet_hash,
            "content_hash_kind": "xet",
        })
    if sum(row["logical_bytes"] for row in source_shards) != state.OFFICIAL_WEIGHT_LOGICAL_BYTES:
        raise CampaignContractError("official manifest shard bytes do not sum exactly")

    source_format = artifacts["source_format_ledger"]
    format_rows = source_format.get("per_shard")
    if source_format.get("weight_shards") != EXACT_SHARDS \
            or source_format.get("tensor_count") != EXACT_TENSORS \
            or source_format.get("container_logical_bytes") != state.OFFICIAL_WEIGHT_LOGICAL_BYTES \
            or not isinstance(format_rows, list) or len(format_rows) != EXACT_SHARDS:
        raise CampaignContractError("source-format ledger exact totals mismatch")
    format_by_path = {
        row.get("path"): row for row in format_rows if isinstance(row, dict)
    }
    if set(format_by_path) != expected_paths:
        raise CampaignContractError("source-format shard inventory differs from manifest")
    for path in expected_paths:
        left, right = manifest_by_path[path], format_by_path[path]
        if right.get("file_bytes") != left.get("logical_bytes") \
                or right.get("xet_hash") != left.get("xet_hash") \
                or right.get("lfs_sha256") != left.get("lfs_sha256"):
            raise CampaignContractError(f"manifest/source-format identity differs: {path}")

    architecture = artifacts["architecture_contract"]
    logical = artifacts["logical_weight_ledger"]
    if architecture.get("architecture") != "GlmMoeDsaForCausalLM" \
            or architecture.get("weights", {}).get("tensor_count") != EXACT_TENSORS \
            or architecture.get("weights", {}).get("logical_elements") != 753_329_940_480 \
            or logical.get("tensor_count") != EXACT_TENSORS \
            or logical.get("logical_weight_denominator") != 753_329_940_480 \
            or logical.get("source_payload_bytes") != 1_506_659_919_872:
        raise CampaignContractError("architecture/logical tensor ledger mismatch")

    graph = artifacts["dependency_graph"]
    graph_shards = graph.get("shards")
    graph_tensors = graph.get("tensors")
    organs = graph.get("organs")
    if graph.get("shard_count") != EXACT_SHARDS or graph.get("tensor_count") != EXACT_TENSORS \
            or graph.get("organ_count") != 81 or not isinstance(graph_shards, list) \
            or not isinstance(graph_tensors, list) or not isinstance(organs, list) \
            or len(graph_shards) != EXACT_SHARDS or len(graph_tensors) != EXACT_TENSORS \
            or len(organs) != 81:
        raise CampaignContractError("dependency graph exact counts mismatch")
    graph_shard_by_path = {
        row.get("path"): row for row in graph_shards if isinstance(row, dict)
    }
    if set(graph_shard_by_path) != expected_paths:
        raise CampaignContractError("dependency graph shard inventory differs from manifest")
    for path in expected_paths:
        graph_row, manifest_row = graph_shard_by_path[path], manifest_by_path[path]
        if graph_row.get("logical_bytes") != manifest_row.get("logical_bytes") \
                or graph_row.get("lfs_sha256") != manifest_row.get("lfs_sha256"):
            raise CampaignContractError(f"dependency graph shard identity differs: {path}")
    tensor_names = _string_list(
        [row.get("name") for row in graph_tensors if isinstance(row, dict)],
        "dependency graph tensor names",
    )
    if len(tensor_names) != EXACT_TENSORS:
        raise CampaignContractError("dependency graph tensor rows are malformed")
    tensor_set = set(tensor_names)
    for row in graph_tensors:
        if row.get("shard") not in expected_paths:
            raise CampaignContractError(f"tensor references an unknown shard: {row.get('name')}")
    organ_by_id: dict[str, Mapping[str, Any]] = {}
    organ_tensor_counts: Counter[str] = Counter()
    for ordinal, organ in enumerate(organs):
        if not isinstance(organ, dict) or organ.get("execution_order") != ordinal:
            raise CampaignContractError("dependency graph organ order is not contiguous")
        organ_id = organ.get("organ_id")
        if not isinstance(organ_id, str) or not organ_id or organ_id in organ_by_id:
            raise CampaignContractError("dependency graph organ identity is invalid")
        names = _string_list(organ.get("tensor_names"), f"organ {organ_id}.tensor_names")
        if organ.get("tensor_count") != len(names):
            raise CampaignContractError(f"organ tensor count mismatch: {organ_id}")
        organ_by_id[organ_id] = organ
        organ_tensor_counts.update(names)
    if set(organ_tensor_counts) != tensor_set or set(organ_tensor_counts.values()) != {1}:
        raise CampaignContractError("dependency graph organs do not partition every tensor once")

    schedule = artifacts["streaming_schedule"]
    windows = schedule.get("windows")
    if schedule.get("window_count") != EXACT_WINDOWS or not isinstance(windows, list) \
            or len(windows) != EXACT_WINDOWS or schedule.get("source_shards_scheduled") != EXACT_SHARDS \
            or schedule.get("planned_refetches") != 0:
        raise CampaignContractError("streaming schedule exact counts mismatch")
    source_window_keys = {
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
    }
    scheduled_organs: Counter[str] = Counter()
    mapped_windows: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        if not isinstance(window, dict) or set(window) != source_window_keys:
            raise CampaignContractError(f"streaming schedule window {index} schema mismatch")
        if window.get("window_id") != f"W{index:03d}":
            raise CampaignContractError("streaming window ids must be contiguous W000..W019")
        organ_ids = _string_list(window.get("organ_ids"), f"window {index}.organ_ids")
        if any(organ_id not in organ_by_id for organ_id in organ_ids):
            raise CampaignContractError(f"window {index} references an unknown organ")
        scheduled_organs.update(organ_ids)
        owned_tensors = sorted(
            name
            for organ_id in organ_ids
            for name in organ_by_id[organ_id]["tensor_names"]
        )
        for key in (
            "source_shards",
            "carry_in_shards",
            "new_fetch_shards",
            "refetch_shards",
            "carry_out_shards",
            "evict_after_seal_shards",
        ):
            _string_list(window.get(key), f"window {index}.{key}")
        if window.get("source_shard_count") != len(window["source_shards"]):
            raise CampaignContractError(f"window {index} source_shard_count mismatch")
        mapped_windows.append({
            "schedule_index": index,
            "window_id": window["window_id"],
            "source_shards": list(window["source_shards"]),
            "carry_in_shards": list(window["carry_in_shards"]),
            "new_fetch_shards": list(window["new_fetch_shards"]),
            "refetch_shards": list(window["refetch_shards"]),
            "carry_out_shards": list(window["carry_out_shards"]),
            # The planner uses an operational name; the state machine uses the
            # terminal action name.  They are intentionally a lossless 1:1 map.
            "evict_shards": list(window["evict_after_seal_shards"]),
            "tensor_set": owned_tensors,
        })
    if set(scheduled_organs) != set(organ_by_id) or set(scheduled_organs.values()) != {1}:
        raise CampaignContractError("streaming schedule does not own every organ exactly once")

    _validate_admission_bindings(artifacts)
    _validate_xet_plan_bindings(artifacts)
    _validate_closure_inputs(artifacts)
    if artifacts["reference_parity"].get("claim_boundary", {}).get(
        "official_bf16_parent_forward"
    ) != "PENDING_FIRST_ADMITTED_SOURCE_WINDOW":
        raise CampaignContractError("reference parity claim boundary is not source-pending")
    corpus_scope = artifacts["corpus_integrity"].get("scope")
    if not isinstance(corpus_scope, dict) or corpus_scope.get("network_access_used") is not False \
            or corpus_scope.get("model_payload_downloaded") is not False \
            or corpus_scope.get("capability_claim_permitted") is not False:
        raise CampaignContractError("corpus integrity artifact crosses its offline claim boundary")

    return DerivedContractInputs(
        source_shards=source_shards,
        tensor_names=sorted(tensor_names),
        window_schedule=mapped_windows,
        input_seals={spec.filename: artifacts[spec.key]["seal_sha256"] for spec in INPUT_SPECS},
    )


def _validate_chat_digest(value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None \
            or len(set(value)) == 1:
        raise CampaignContractError(
            "rotated Telegram chat identity digest is required as 64 lowercase hex"
        )
    return value


def _validate_created_at(value: Any) -> str:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise CampaignContractError("created_at must be explicit UTC YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise CampaignContractError("created_at is not a valid UTC timestamp") from exc
    return value


def _artifact_specs(
    paths: Mapping[str, str], input_seals: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        input_spec = INPUT_BY_FILENAME.get(path)
        is_final_schedule = label in {"streaming_schedule", "frozen_streaming_schedule"} \
            and path == "GLM52_STREAMING_SCHEDULE.json"
        is_xet_result = path == "GLM52_XET_AUTOTUNE.json"
        is_preliminary_archive = label == "preliminary_streaming_schedule"
        if is_preliminary_archive:
            input_spec = INPUT_BY_KEY["streaming_schedule"]
            path_for_seal = "GLM52_STREAMING_SCHEDULE_PRE_AUTOTUNE.json"
        else:
            path_for_seal = path
        if input_spec is not None and not is_final_schedule:
            schema = input_spec.schema
            statuses = (input_spec.status,)
        elif path in FUTURE_ARTIFACT_POLICIES:
            schema, statuses = FUTURE_ARTIFACT_POLICIES[path]
        else:
            raise CampaignContractError(f"artifact path lacks a frozen validation policy: {path}")
        validator_id = (
            "frozen_schedule_v2" if is_final_schedule
            else "xet_autotune_result_v1" if is_xet_result
            else "sealed_exact_v1" if input_seals.get(path_for_seal) is not None
            else "future_artifact_blocked_v1"
        )
        result[label] = {
            "path": path,
            "expected_seal_sha256": (
                None if is_final_schedule else input_seals.get(path_for_seal)
            ),
            "expected_schema": schema,
            "allowed_statuses": list(statuses),
            "validator_id": validator_id,
            "validator_source_sha256": state.EVIDENCE_VALIDATOR_SOURCE_SHA256[validator_id],
            "require_producer_hmac": is_final_schedule or input_seals.get(path_for_seal) is None,
        }
    return result


def _checklist_specs(items: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {
        item: {
            "path": f"evidence/stop_conditions/{item}.json",
            "expected_seal_sha256": None,
            "expected_schema": state.STOP_CONDITION_EVIDENCE_SCHEMA,
            "allowed_statuses": ["PASS"],
            "validator_id": "stop_condition_blocked_v1",
            "validator_source_sha256": state.EVIDENCE_VALIDATOR_SOURCE_SHA256[
                "stop_condition_blocked_v1"
            ],
            "require_producer_hmac": True,
        }
        for item in items
    }


def _state_gates(input_seals: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    def gate(
        artifacts: Mapping[str, str],
        checklist: Sequence[str],
        *,
        source_complete: bool = False,
        tensor_complete: bool = False,
        final_eviction: bool = False,
    ) -> dict[str, Any]:
        return {
            "require_source_complete": source_complete,
            "require_tensor_complete": tensor_complete,
            "require_final_source_eviction": final_eviction,
            "require_telegram_delivery": True,
            "require_phone_status": False,
            "required_phone_status_path": None,
            "required_artifacts": _artifact_specs(artifacts, input_seals),
            "required_checklist": _checklist_specs(checklist),
        }

    return {
        "AUTOTUNE_XET": gate(
            {
                "official_source_manifest": "GLM52_OFFICIAL_MANIFEST.json",
                "source_format_ledger": "GLM52_SOURCE_FORMAT_LEDGER.json",
                "tensor_coverage_ledger": "GLM52_SHARD_DEPENDENCY_GRAPH.json",
                "source_admission": "GLM52_SOURCE_ADMISSION.json",
                "xet_autotune_plan": "GLM52_XET_AUTOTUNE_PLAN.json",
            },
            ("official_source_admitted", "offline_xet_plan_sealed"),
        ),
        "BUILD_ADAPTER": gate(
            {
                "xet_autotune_result": "GLM52_XET_AUTOTUNE.json",
                "frozen_streaming_schedule": "GLM52_STREAMING_SCHEDULE.json",
            },
            ("xet_autotune_complete", "xet_selected_profile_sealed"),
        ),
        "BUILD_REFERENCE": gate(
            {"adapter_twin": "GLM52_ADAPTER_TWIN.json"},
            ("adapter_twin_green",),
        ),
        "BUILD_CORPUS": gate(
            {
                "reference_parity": "GLM52_REFERENCE_PARITY.json",
                "bf16_reference_forward": "GLM52_BF16_REFERENCE_FORWARD.json",
            },
            ("synthetic_reference_parity_green", "bf16_reference_forward_validated"),
        ),
        "PILOT_ORACLES": gate(
            {"corpus_integrity": "GLM52_CORPUS_INTEGRITY.json"},
            ("corpus_integrity_green",),
        ),
        "FREEZE_PROGRAM": gate(
            {
                "oracle_bandwidth": "GLM52_ORACLE_BANDWIDTH.json",
                "causal_atlas": "GLM52_CAUSAL_ATLAS.json",
            },
            ("oracle_causal_pilot_complete",),
        ),
        "FETCH_WINDOW": gate(
            {
                "xet_autotune_result": "GLM52_XET_AUTOTUNE.json",
                "frozen_streaming_schedule": "GLM52_STREAMING_SCHEDULE.json",
                "adapter_twin": "GLM52_ADAPTER_TWIN.json",
                "reference_parity": "GLM52_REFERENCE_PARITY.json",
                "bf16_reference_forward": "GLM52_BF16_REFERENCE_FORWARD.json",
                "corpus_integrity": "GLM52_CORPUS_INTEGRITY.json",
                "oracle_bandwidth": "GLM52_ORACLE_BANDWIDTH.json",
                "causal_atlas": "GLM52_CAUSAL_ATLAS.json",
                "frozen_program": "GLM52_FROZEN_GRAVITY_PROGRAM.json",
            },
            (
                "xet_autotune_complete",
                "xet_selected_profile_sealed",
                "bf16_reference_forward_validated",
                "corpus_integrity_green",
                "oracle_causal_pilot_complete",
                "full_candidate_program_frozen",
            ),
        ),
        "ASSEMBLE_ARTIFACT": {
            "require_source_complete": True,
            "require_tensor_complete": True,
            "require_final_source_eviction": True,
            "require_telegram_delivery": True,
            "require_phone_status": False,
            "required_phone_status_path": None,
            "required_artifacts": _artifact_specs(ASSEMBLE_ARTIFACT_PATHS, input_seals),
            "required_checklist": _checklist_specs(
                state.MANDATORY_COMPLETE_STOP_CONDITIONS[:16]
            ),
        },
        "VERIFY_ARTIFACT": gate(
            {
                "compact_artifact_manifest": "GLM52_COMPACT_ARTIFACT_MANIFEST.json",
                "source_to_compact_coverage": "GLM52_SOURCE_TO_COMPACT_COVERAGE.json",
            },
            ("complete_sub_one_artifact_assembled",),
            source_complete=True,
            tensor_complete=True,
            final_eviction=True,
        ),
        "RUN_FULL_COMPACT": gate(
            {
                "compact_artifact_manifest": "GLM52_COMPACT_ARTIFACT_MANIFEST.json",
                "source_to_compact_coverage": "GLM52_SOURCE_TO_COMPACT_COVERAGE.json",
            },
            ("complete_sub_one_artifact_assembled",),
            source_complete=True,
            tensor_complete=True,
            final_eviction=True,
        ),
        "RUN_DOCTOR_REFINEMENT": gate(
            {
                "full_compact_results": "GLM52_FULL_COMPACT_RESULTS.json",
                "capability_results": "GLM52_CAPABILITY_RESULTS.json",
                "doctor_program": "GLM52_DOCTOR_PROGRAM.json",
            },
            ("full_compact_model_executed", "capability_result_sealed"),
        ),
        "RUN_RATE_DESCENT": gate(
            {
                "full_compact_results": "GLM52_FULL_COMPACT_RESULTS.json",
                "capability_results": "GLM52_CAPABILITY_RESULTS.json",
                "rate_program": "GLM52_RATE_PROGRAM.json",
            },
            ("full_compact_model_executed", "capability_result_sealed"),
        ),
        "SEAL_GLM_RESULT": gate(
            {
                "full_compact_results": "GLM52_FULL_COMPACT_RESULTS.json",
                "capability_results": "GLM52_CAPABILITY_RESULTS.json",
                "source_eviction_report": "GLM52_SOURCE_EVICTION_REPORT.json",
                "byte_auction": "GLM52_FINAL_BYTE_AUCTION.json",
            },
            (
                "half_bpw_exact_tested_f0_f1",
                "earned_rate_fidelity_evaluated",
                "doctor_native_student_compared",
                "direct_runtime_boundary_reported",
            ),
            source_complete=True,
            tensor_complete=True,
            final_eviction=True,
        ),
        "FINAL_GRAVITY_AUDIT": gate(
            {
                "glm52_gravity_final": "GLM52_GRAVITY_FINAL.json",
                "gravity_completeness_audit_final":
                    "GRAVITY_COMPLETENESS_AUDIT_GLM52_POST.json",
            },
            ("gravity_post_audit_complete", "terminal_outcome_sealed"),
            source_complete=True,
            tensor_complete=True,
            final_eviction=True,
        ),
        "COMPLETE": {
            "require_source_complete": True,
            "require_tensor_complete": True,
            "require_final_source_eviction": True,
            "require_telegram_delivery": True,
            "require_phone_status": True,
            "required_phone_status_path": "GLM52_PHONE_STATUS.json",
            "required_artifacts": _artifact_specs(COMPLETE_ARTIFACT_PATHS, input_seals),
            "required_checklist": _checklist_specs(state.MANDATORY_COMPLETE_STOP_CONDITIONS),
        },
    }


def build_contract_from_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    chat_identity_digest: str,
    created_at: str,
) -> dict[str, Any]:
    digest = _validate_chat_digest(chat_identity_digest)
    timestamp = _validate_created_at(created_at)
    derived = validate_and_derive(artifacts)
    try:
        return state.make_expected_campaign_contract(
            campaign_id="glm52-bf16-xet-gravity",
            source_revision=OFFICIAL_REVISION,
            expected_chat_identity_digest=digest,
            source_shards=derived.source_shards,
            expected_tensors=derived.tensor_names,
            window_schedule=derived.window_schedule,
            state_gates=_state_gates(derived.input_seals),
            source_profile="OFFICIAL_GLM52_BF16",
            created_at=timestamp,
        )
    except state.StateError as exc:
        raise CampaignContractError(str(exc)) from exc


def build_contract(
    root: str | os.PathLike[str] = REPO_ROOT,
    *,
    chat_identity_digest: str,
    created_at: str,
) -> dict[str, Any]:
    bundle = load_inputs(root)
    contract = build_contract_from_artifacts(
        bundle.artifacts,
        chat_identity_digest=chat_identity_digest,
        created_at=created_at,
    )
    assert_inputs_stable(bundle)
    return contract


def preflight(
    root: str | os.PathLike[str] = REPO_ROOT,
    *,
    chat_identity_digest: str | None,
    created_at: str | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    bundle: InputBundle | None = None
    derived: DerivedContractInputs | None = None
    try:
        bundle = load_inputs(root)
        derived = validate_and_derive(bundle.artifacts)
        assert_inputs_stable(bundle)
    except CampaignContractError as exc:
        blockers.append(f"FROZEN_INPUTS_INVALID:{exc}")
    try:
        _validate_chat_digest(chat_identity_digest)
    except CampaignContractError:
        blockers.append("ROTATED_CHAT_IDENTITY_DIGEST_REQUIRED_64_LOWERCASE_HEX")
    try:
        _validate_created_at(created_at)
    except CampaignContractError:
        blockers.append("DETERMINISTIC_CREATED_AT_REQUIRED_UTC_SECONDS")
    ready = not blockers
    return seal({
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY_TO_BUILD" if ready else "BLOCKED",
        "build_authorized": ready,
        "repo": OFFICIAL_REPO,
        "revision": OFFICIAL_REVISION,
        "chat_identity_digest_present": chat_identity_digest is not None,
        "created_at_present": created_at is not None,
        "expected_counts": {
            "weight_shards": EXACT_SHARDS,
            "tensor_names": EXACT_TENSORS,
            "windows": EXACT_WINDOWS,
            "weight_container_logical_bytes": state.OFFICIAL_WEIGHT_LOGICAL_BYTES,
        },
        "observed_counts": {
            "weight_shards": len(derived.source_shards) if derived else None,
            "tensor_names": len(derived.tensor_names) if derived else None,
            "windows": len(derived.window_schedule) if derived else None,
        },
        "input_seals": derived.input_seals if derived else {},
        "blockers": blockers,
        "output_created": False,
    })


def write_contract(path: str | os.PathLike[str], contract: Mapping[str, Any]) -> Path:
    target = Path(path)
    value = dict(contract)
    try:
        verify_sealed(value, label="expected campaign contract")
        state._validate_expected_contract(value)  # authoritative schema validation
    except (Glm52Error, state.StateError) as exc:
        raise CampaignContractError(str(exc)) from exc
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise CampaignContractError("refusing to replace unsafe expected-contract path")
        existing = _strict_json_bytes(target.read_bytes(), label=os.fspath(target))
        try:
            verify_sealed(existing, label=os.fspath(target))
        except Glm52Error as exc:
            raise CampaignContractError("refusing to overwrite an invalid existing contract") from exc
        if canonical(existing) != canonical(value):
            raise CampaignContractError("refusing to overwrite a different frozen contract")
        return target
    atomic_json(target, value)
    written = _strict_json_bytes(target.read_bytes(), label=os.fspath(target))
    if canonical(written) != canonical(value):
        raise CampaignContractError("expected-contract post-write verification failed")
    return target


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: invalid command-line arguments\n")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--chat-identity-digest")
    parser.add_argument("--created-at")


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight", help="read-only readiness report")
    _add_common(preflight_parser)
    build = subparsers.add_parser("build", help="write the frozen official contract")
    _add_common(build)
    build.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        report = preflight(
            args.root,
            chat_identity_digest=args.chat_identity_digest,
            created_at=args.created_at,
        )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if report["build_authorized"] else 2
    try:
        contract = build_contract(
            args.root,
            chat_identity_digest=args.chat_identity_digest,
            created_at=args.created_at,
        )
        output = args.output or Path(args.root) / OUTPUT_FILENAME
        write_contract(output, contract)
    except (CampaignContractError, OSError) as exc:
        print(json.dumps({
            "status": "ERROR",
            "error": type(exc).__name__,
            "message": str(exc),
            "output_created": False,
        }, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "BUILT",
        "path": os.fspath(output),
        "seal_sha256": contract["seal_sha256"],
        "source_shards": contract["source"]["expected_shard_count"],
        "tensor_names": contract["tensors"]["expected_tensor_count"],
        "windows": len(contract["window_schedule"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

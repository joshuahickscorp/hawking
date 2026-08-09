"""Guarded, receipt-last outer contract for a future Q30 streamed source oracle.

This module is intentionally not an executor.  It only joins the immutable
Q30 source/replay records into a fail-closed outer preflight and provides
metadata validators a *future*, separately reviewed runner must satisfy.  It
does not accept a source root, a safetensors path, or a child command; it has
no subprocess, server, HCLI, Metal, MPS, or lease-issuing surface.

The present result is always blocked.  A real layer-streamed source executor,
fresh safety observation, source lease, source capture, source eviction, and
fresh native lease are all future evidence.  The order is deliberately rigid:

    source-streamed capture -> durable source-vector retention -> eviction
    -> native lease -> native capture

The helper validators below are deliberately receipt-only.  They do not start
children; their sole job is to reject an unsafe or replayed future receipt
bundle before a different outer implementation could consume it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators import ascension_qwen30_layer_streamed_source_bf16_oracle_feasibility as feasibility
from lab.operators import ascension_qwen30_quality_repack_raw_final_logit_retention_contract as retention_contract
from lab.operators import ascension_qwen30_quality_repack_source_bf16_memory_lease_preflight as memory_preflight
from lab.operators import ascension_qwen30_quality_repack_source_oracle_three_way_contract as three_way_contract
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates"
    / "gate-up-residual-v1"
)
DEFAULT_SOURCE_CONTRACT = (
    DEFAULT_ROOT / "source-bf16-three-way-final-logit-contract/receipts"
    / "QWEN30_HQ30GR2_SOURCE_BF16_THREE_WAY_FINAL_LOGIT_CONTRACT_"
    "883c59eec0371ebb6d4a9935cdbdc6bcb486c03eebd5312db608a0415a34911f.json"
)
DEFAULT_MEMORY_PREFLIGHT = (
    DEFAULT_ROOT / "source-bf16-three-way-memory-preflight/receipts"
    / "QWEN30_HQ30GR2_SOURCE_BF16_MEMORY_PREFLIGHT_"
    "efdacf5952583dc03d2aee37b73a0af284f2d865a3e36b9739b16150efe3f726.json"
)
DEFAULT_RANGE_AUTHORITY = (
    DEFAULT_ROOT / "source-range-map-authority/authorities"
    / "QWEN30_STREAMED_ORACLE_METADATA_ONLY_RANGE_MAP_AUTHORITY_"
    "b2cff646eb4bb1d68355c01b18ae02e7cf42d120.json"
)
DEFAULT_SEMANTICS = (
    DEFAULT_ROOT / "source-range-map-authority/authorities"
    / "QWEN30_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_"
    "b2cff646eb4bb1d68355c01b18ae02e7cf42d120.json"
)
DEFAULT_RAW_RETENTION = (
    DEFAULT_ROOT / "raw-final-logit-retention-successor/receipts"
    / "QWEN30_HQ30GR2_RAW_FINAL_LOGIT_RETENTION_SUCCESSOR_"
    "07260bb96d09dab6ba7b0955c4f72da541404dfb5c38117dffe944173a9e8e34.json"
)
DEFAULT_CURRENT_TRACE = (
    DEFAULT_ROOT / "all-layer-current-trace-comparison/receipts"
    / "QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_COMPARISON_"
    "883c59eec0371ebb6d4a9935cdbdc6bcb486c03eebd5312db608a0415a34911f_"
    "98db412d42e87e938a89c759d426f69bd51e700a944701704a5c82949244298f.json"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "streamed-source-oracle-outer-controller/receipts"

SCHEMA = "hawking.ascension.qwen30_guarded_streamed_source_oracle_outer_controller.v1"
BLOCKED_STATUS = "BLOCKED_QWEN30_GUARDED_STREAMED_SOURCE_ORACLE_OUTER_NO_EXECUTOR_NOT_EXECUTED"

RANGE_AUTHORITY_SCHEMA = "hawking.ascension.qwen30_streamed_oracle_metadata_only_range_map_authority.v1"
RANGE_AUTHORITY_STATUS = "PREPARED_QWEN30_STREAMED_ORACLE_SOURCE_RANGE_MAP_AUTHORITY_NOT_EXECUTED"
SEMANTICS_SCHEMA = "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_semantics_attester.v1"
SEMANTICS_STATUS = "PREPARED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_NOT_EXECUTED"
FUTURE_EXECUTION_SEMANTICS_SCHEMA = (
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_execution_attestation.v1"
)
FUTURE_EXECUTION_SEMANTICS_STATUS = (
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_ATTESTED"
)

# These schemas intentionally align with the existing raw-six-vector sequence
# so a future implementation cannot create a parallel, incompatible lease
# family merely to bypass the source-then-evict ordering.
SOURCE_LEASE_SCHEMA = "hawking.ascension.qwen30_hq30gr2_source_bf16_teacher_quiet_lease.v1"
SOURCE_LEASE_STATUS = "GRANTED_QWEN30_HQ30GR2_SOURCE_BF16_TEACHER_RAW_LOGIT_CAPTURE_ONE_SHOT"
SOURCE_TERMINAL_SCHEMA = "hawking.ascension.qwen30_hq30gr2_source_bf16_teacher_raw_logit_capture.v1"
SOURCE_TERMINAL_STATUS = "CAPTURED_QWEN30_HQ30GR2_SOURCE_BF16_TWO_RAW_FINAL_LOGITS_TEACHER_ONLY"
SOURCE_EVICTION_SCHEMA = "hawking.ascension.qwen30_hq30gr2_source_bf16_teacher_eviction.v1"
SOURCE_EVICTION_STATUS = "EARNED_QWEN30_HQ30GR2_SOURCE_BF16_TEACHER_EVICTED_BEFORE_NATIVE_CAPTURE"
NATIVE_LEASE_SCHEMA = "hawking.ascension.qwen30_hq30gr2_raw_final_logit_retention_quiet_lease.v1"
NATIVE_LEASE_STATUS = "GRANTED_QWEN30_HQ30GR2_RAW_FINAL_LOGIT_RETENTION_ONE_SHOT"

MAX_METADATA_BYTES = 64 * 1024 * 1024
PREFIX_TOKENS = three_way_contract.PREFIX_TOKENS
FORCED_TOKEN = three_way_contract.FORCED_TOKEN
VOCAB_ROWS = three_way_contract.VOCAB_ROWS
F32_VECTOR_BYTES = retention_contract.F32LE_BYTES_PER_VECTOR


class GuardedStreamedSourceOuterError(RuntimeError):
    """A metadata binding cannot safely describe a future source capture."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _regular_json(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise GuardedStreamedSourceOuterError(f"{label} must be absolute")
    if path.suffix != ".json":
        raise GuardedStreamedSourceOuterError(f"{label} must be a .json metadata receipt")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise GuardedStreamedSourceOuterError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise GuardedStreamedSourceOuterError(f"{label} must be a regular non-symlink file")
    if metadata.st_size <= 0 or metadata.st_size > MAX_METADATA_BYTES:
        raise GuardedStreamedSourceOuterError(
            f"{label} must contain 1..={MAX_METADATA_BYTES} metadata bytes"
        )
    return path.resolve(strict=True)


def _sealed_json(path: Path, *, label: str) -> tuple[dict[str, Any], Path]:
    clean = _regular_json(path, label=label)
    try:
        document = verify(json.loads(clean.read_text(encoding="utf-8")), label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise GuardedStreamedSourceOuterError(f"{label} is absent or invalid: {exc}") from exc
    if not isinstance(document, Mapping):
        raise GuardedStreamedSourceOuterError(f"{label} is not an object")
    return dict(document), clean


def _metadata_json(path: Path, *, label: str) -> tuple[dict[str, Any], Path]:
    """Read a bounded JSON metadata document that deliberately is not sealed."""
    clean = _regular_json(path, label=label)
    try:
        document = json.loads(clean.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardedStreamedSourceOuterError(f"{label} is absent or invalid: {exc}") from exc
    if not isinstance(document, Mapping):
        raise GuardedStreamedSourceOuterError(f"{label} is not an object")
    return dict(document), clean


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GuardedStreamedSourceOuterError(f"{label} must be an object")
    return dict(value)


def _sequence(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise GuardedStreamedSourceOuterError(f"{label} must be an array")
    return list(value)


def _text(value: object, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise GuardedStreamedSourceOuterError(f"{label} must be a non-empty string")
    if sha256 and (len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
        raise GuardedStreamedSourceOuterError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GuardedStreamedSourceOuterError(f"{label} must be an integer >= {minimum}")
    return value


def _require_true(value: object, *, label: str) -> None:
    if value is not True:
        raise GuardedStreamedSourceOuterError(f"{label} must be true")


def _schema_status(document: Mapping[str, Any], *, schema: str, status: str, label: str) -> None:
    if document.get("schema") != schema or document.get("status") != status:
        raise GuardedStreamedSourceOuterError(f"{label} schema/status drifted")


def _verified_future_receipt(document: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Require a sealed future receipt before accepting any safety assertion."""
    try:
        checked = verify(document, label=label)
    except SealIntegrityError as exc:
        raise GuardedStreamedSourceOuterError(f"{label} seal is absent or invalid: {exc}") from exc
    return dict(checked)


def _evidence(path: Path, document: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if document is not None:
        result["seal_sha256"] = _text(document.get("seal_sha256"), label="evidence seal", sha256=True)
    return result


def _same_path(value: object, *, expected: Path, label: str) -> None:
    observed = Path(_text(value, label=label))
    if not observed.is_absolute() or observed.resolve() != expected:
        raise GuardedStreamedSourceOuterError(f"{label} names a different metadata receipt")


def _trace_from_source_contract(source: Mapping[str, Any]) -> dict[str, Any]:
    _schema_status(
        source,
        schema=three_way_contract.SCHEMA,
        status=three_way_contract.STATUS,
        label="source three-way contract",
    )
    exact = _mapping(source.get("exact_input"), label="source exact trace")
    prefix = _integer(exact.get("source_template_token_count"), label="source prefix token count")
    forced = _integer(exact.get("forced_identical_continuation_token_id"), label="source forced token")
    if prefix != PREFIX_TOKENS or forced != FORCED_TOKEN:
        raise GuardedStreamedSourceOuterError("source contract trace length/token drifted")
    _require_true(
        exact.get("source_must_execute_the_same_369_token_prefix_then_the_forced_token"),
        label="source exact prefix/forced order",
    )
    _require_true(
        exact.get("sampling_or_autoregressive_feedback_is_forbidden"),
        label="source no-sampling boundary",
    )
    return {
        "probe_id": _text(exact.get("probe_id"), label="source probe ID"),
        "source_template_token_count": prefix,
        "forced_identical_continuation_token_id": forced,
        "source_template_token_ids_u32le_sha256": _text(
            exact.get("source_template_token_ids_u32le_sha256"), label="source token SHA", sha256=True
        ),
    }


def _validate_memory_binding(*, source: Mapping[str, Any], source_path: Path, memory: Mapping[str, Any]) -> dict[str, int | str]:
    if memory.get("schema") != memory_preflight.SCHEMA or memory.get("status") not in {
        memory_preflight.READY_STATUS,
        memory_preflight.BLOCKED_STATUS,
    }:
        raise GuardedStreamedSourceOuterError("strict source memory preflight schema/status drifted")
    pointer = _mapping(memory.get("source_bf16_three_way_contract"), label="memory source contract pointer")
    _same_path(pointer.get("path"), expected=source_path, label="memory source contract path")
    if pointer.get("seal_sha256") != source.get("seal_sha256"):
        raise GuardedStreamedSourceOuterError("memory source contract seal drifted")
    snapshot = _mapping(memory.get("measured_system_snapshot"), label="memory preflight snapshot")
    swap = _mapping(snapshot.get("swap"), label="memory preflight swap")
    vm = _mapping(snapshot.get("vm_stat"), label="memory preflight vm")
    headroom = _mapping(memory.get("headroom_assessment"), label="memory preflight headroom")
    swap_used = _integer(swap.get("used_bytes"), label="memory preflight swap used")
    if _integer(headroom.get("measured_swap_used_bytes"), label="headroom swap used") != swap_used:
        raise GuardedStreamedSourceOuterError("memory preflight swap values disagree")
    reclaimable = _integer(vm.get("reclaimable_bytes"), label="memory preflight reclaimable")
    if _integer(headroom.get("measured_reclaimable_bytes"), label="headroom reclaimable") != reclaimable:
        raise GuardedStreamedSourceOuterError("memory preflight reclaimable values disagree")
    required = _integer(
        headroom.get("minimum_reclaimable_bytes_required_before_source_load"),
        label="memory preflight required reclaimable",
        minimum=1,
    )
    return {
        "status": _text(memory.get("status"), label="memory preflight status"),
        "reclaimable_bytes": reclaimable,
        "required_reclaimable_bytes": required,
        "swap_used_bytes": swap_used,
        "swapouts_pages": _integer(vm.get("swapouts_pages"), label="memory preflight swapouts"),
    }


def _validate_range_authority(*, document: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    content_sha = _text(document.get("authority_content_sha256"), label="range authority content SHA", sha256=True)
    authority = _mapping(document.get("authority"), label="range authority material")
    _schema_status(authority, schema=RANGE_AUTHORITY_SCHEMA, status=RANGE_AUTHORITY_STATUS, label="range authority")
    source = _mapping(authority.get("source"), label="range authority source")
    if _integer(source.get("source_tensor_count"), label="range authority tensor count", minimum=1) != 18_867:
        raise GuardedStreamedSourceOuterError("range authority source tensor count drifted")
    if _integer(source.get("source_shard_count"), label="range authority shard count", minimum=1) != 16:
        raise GuardedStreamedSourceOuterError("range authority source shard count drifted")
    index = _mapping(source.get("source_index"), label="range authority source index")
    if _integer(index.get("weight_map_tensor_count"), label="range authority index tensor count", minimum=1) != 18_867:
        raise GuardedStreamedSourceOuterError("range authority index tensor count drifted")
    _text(index.get("sha256"), label="range authority index SHA", sha256=True)
    scope = _mapping(authority.get("exact_streamed_oracle_scope"), label="range authority scope")
    if _integer(scope.get("source_template_token_count"), label="range authority prefix") != PREFIX_TOKENS:
        raise GuardedStreamedSourceOuterError("range authority prefix length drifted")
    if _integer(scope.get("forced_identical_continuation_token_id"), label="range authority forced token") != FORCED_TOKEN:
        raise GuardedStreamedSourceOuterError("range authority forced token drifted")
    _require_true(scope.get("sampling_or_autoregressive_feedback_forbidden"), label="range authority no sampling")
    if _integer(scope.get("row_tile_rows"), label="range authority row tile rows", minimum=1) != 128:
        raise GuardedStreamedSourceOuterError("range authority row tile bound drifted")
    boundary = _mapping(authority.get("metadata_access_boundary"), label="range authority metadata boundary")
    for key in (
        "source_model_instantiated",
        "gpu_or_metal_invoked",
        "server_started",
        "hcli_invoked",
        "lease_requested",
        "mmap_or_memory_map_used",
        "tensor_payload_hashes_collected",
        "whole_shard_payload_checksum_collected",
    ):
        if boundary.get(key) is not False:
            raise GuardedStreamedSourceOuterError(f"range authority {key} must remain false")
    if _integer(boundary.get("source_tensor_payload_bytes_read"), label="range authority payload bytes") != 0:
        raise GuardedStreamedSourceOuterError("range authority must not contain payload reads")
    tensors = _sequence(authority.get("tensors"), label="range authority tensors")
    if not tensors:
        raise GuardedStreamedSourceOuterError("range authority needs at least one declared tensor window")
    maximum_window_bytes = 0
    for index_value, row in enumerate(tensors):
        tensor = _mapping(row, label=f"range authority tensor {index_value}")
        if tensor.get("source_dtype") != "BF16":
            raise GuardedStreamedSourceOuterError("range authority source dtype must remain BF16")
        shape = _sequence(tensor.get("row_window_shape"), label=f"range authority tensor {index_value} window")
        elements = 1
        for dimension in shape:
            elements *= _integer(dimension, label="range authority row-window dimension", minimum=1)
        maximum_window_bytes = max(maximum_window_bytes, elements * 2)
    return authority, maximum_window_bytes


def _validate_semantics(
    *,
    document: Mapping[str, Any],
    path: Path,
    range_document: Mapping[str, Any],
    range_path: Path,
    source: Mapping[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    _schema_status(document, schema=SEMANTICS_SCHEMA, status=SEMANTICS_STATUS, label="streamed semantics")
    boundary = _mapping(document.get("execution_boundary"), label="streamed semantics execution boundary")
    for key in (
        "source_tensor_payload_opened",
        "source_safetensors_or_other_weight_path_accepted",
        "source_model_instantiated",
        "source_inference_executed",
        "gpu_or_metal_invoked",
        "server_started",
        "hcli_invoked",
        "lease_requested",
        "source_quality_or_coherence_claim_made",
        "tps_or_tg_claim_made",
    ):
        if boundary.get(key) is not False:
            raise GuardedStreamedSourceOuterError(f"streamed semantics {key} must remain false")
    source_binding = _mapping(document.get("pinned_source_binding"), label="streamed semantics source binding")
    authority = _mapping(range_document.get("authority"), label="range authority material")
    authority_source = _mapping(authority.get("source"), label="range authority source")
    if source_binding.get("source_model_id") != authority_source.get("model_id"):
        raise GuardedStreamedSourceOuterError("semantics source model ID differs from range authority")
    if source_binding.get("source_revision") != authority_source.get("source_revision"):
        raise GuardedStreamedSourceOuterError("semantics source revision differs from range authority")
    if source_binding.get("source_index_sha256") != _mapping(
        authority_source.get("source_index"), label="range authority index"
    ).get("sha256"):
        raise GuardedStreamedSourceOuterError("semantics source index differs from range authority")
    geometry = _mapping(source_binding.get("geometry"), label="streamed semantics geometry")
    expected_geometry = {
        "layers": 48,
        "hidden_size": 2048,
        "vocab_size": VOCAB_ROWS,
        "attention_heads": 32,
        "key_value_heads": 4,
        "head_dim": 128,
        "experts": 128,
        "top_k": 8,
        "moe_intermediate": 768,
        "source_tensor_count": 18_867,
        "source_shard_count": 16,
    }
    for key, expected in expected_geometry.items():
        if _integer(geometry.get(key), label=f"semantics geometry {key}") != expected:
            raise GuardedStreamedSourceOuterError(f"semantics geometry {key} drifted")
    consumed = _mapping(document.get("consumed_metadata_contracts"), label="streamed semantics consumed contracts")
    range_ref = _mapping(consumed.get("range_authority"), label="semantics range authority pointer")
    _same_path(range_ref.get("path"), expected=range_path, label="semantics range authority path")
    if range_ref.get("document_sha256") != _sha256_file(range_path):
        raise GuardedStreamedSourceOuterError("semantics range authority document hash drifted")
    if range_ref.get("authority_content_sha256") != range_document.get("authority_content_sha256"):
        raise GuardedStreamedSourceOuterError("semantics range authority content hash drifted")
    if range_ref.get("source_payload_read_by_this_attester") is not False:
        raise GuardedStreamedSourceOuterError("semantics attester may not have read source payloads")
    source_ref = _mapping(consumed.get("sealed_replay_contract"), label="semantics source contract pointer")
    _same_path(source_ref.get("path"), expected=source_path, label="semantics source contract path")
    if source_ref.get("document_sha256") != _sha256_file(source_path):
        raise GuardedStreamedSourceOuterError("semantics source contract document hash drifted")
    if source_ref.get("seal_sha256") != source.get("seal_sha256"):
        raise GuardedStreamedSourceOuterError("semantics source contract seal drifted")
    future = _mapping(document.get("future_exact_execution_attestation"), label="future execution semantics")
    if future.get("schema") != FUTURE_EXECUTION_SEMANTICS_SCHEMA or future.get(
        "status_only_after_real_separately_leased_source_execution"
    ) != FUTURE_EXECUTION_SEMANTICS_STATUS:
        raise GuardedStreamedSourceOuterError("future execution semantics schema/status drifted")
    _require_true(
        future.get("must_retain_and_hash_six_full_f32_endpoint_logit_vectors_before_any_three_way_scoring"),
        label="future six-vector retention rule",
    )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "future_execution_attestation_schema": FUTURE_EXECUTION_SEMANTICS_SCHEMA,
        "future_execution_attestation_status": FUTURE_EXECUTION_SEMANTICS_STATUS,
    }


def _validate_raw_retention(
    *, raw: Mapping[str, Any], raw_path: Path, memory: Mapping[str, Any], memory_path: Path, trace: Mapping[str, Any]
) -> dict[str, Any]:
    _schema_status(raw, schema=retention_contract.SCHEMA, status=retention_contract.STATUS, label="raw six-vector retention")
    memory_ref = _mapping(raw.get("strict_source_bf16_memory_preflight"), label="raw retention memory pointer")
    _same_path(memory_ref.get("path"), expected=memory_path, label="raw retention memory path")
    if memory_ref.get("seal_sha256") != memory.get("seal_sha256"):
        raise GuardedStreamedSourceOuterError("raw retention memory seal drifted")
    replay = _mapping(raw.get("replay_binding"), label="raw retention replay binding")
    raw_trace = _mapping(replay.get("exact_trace"), label="raw retention exact trace")
    _assert_trace(raw_trace, trace, label="raw retention")
    plan = _mapping(raw.get("six_vector_retention_contract"), label="raw six-vector contract")
    expected_plan = retention_contract.raw_vector_plan()
    for key, expected in expected_plan.items():
        if plan.get(key) != expected:
            raise GuardedStreamedSourceOuterError(f"raw six-vector contract {key} drifted")
    gate = _mapping(raw.get("source_memory_and_eviction_gate"), label="raw retention source gate")
    _require_true(gate.get("must_evict_source_weights_and_confirm_release_before_native_capture"), label="raw eviction rule")
    _require_true(gate.get("source_and_native_model_bodies_must_not_be_resident_concurrently"), label="raw no co-residency rule")
    return {
        "evidence": _evidence(raw_path, raw),
        "six_vector_plan": expected_plan,
        "raw_retention_current_memory_status": _text(
            gate.get("current_memory_preflight_status"), label="raw retention memory status"
        ),
    }


def _assert_trace(observed: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    for key in (
        "probe_id",
        "source_template_token_count",
        "forced_identical_continuation_token_id",
        "source_template_token_ids_u32le_sha256",
    ):
        if observed.get(key) != expected.get(key):
            raise GuardedStreamedSourceOuterError(f"{label} trace {key} drifted")


def _validate_current_trace(*, current: Mapping[str, Any], current_path: Path, raw: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    _schema_status(
        current,
        schema=three_way_contract.COMPARISON_SCHEMA,
        status=three_way_contract.COMPARISON_STATUS,
        label="current native trace",
    )
    binding = _mapping(current.get("binding"), label="current native trace binding")
    _assert_trace(binding, trace, label="current native")
    replay = _mapping(raw.get("replay_binding"), label="raw retention replay binding")
    comparison = _mapping(replay.get("candidate_local_comparison"), label="raw current trace pointer")
    _same_path(comparison.get("path"), expected=current_path, label="raw current trace path")
    if comparison.get("seal_sha256") != current.get("seal_sha256"):
        raise GuardedStreamedSourceOuterError("raw current trace seal drifted")
    boundary = _mapping(current.get("claim_boundary"), label="current native trace claim boundary")
    _require_true(
        boundary.get("does_not_claim_semantic_coherence_hcli_tps_tg_capability_or_tournament"),
        label="current trace no qualification claim",
    )
    return _evidence(current_path, current)


def build_current_preflight(
    *,
    source_contract_path: Path,
    memory_preflight_path: Path,
    range_authority_path: Path,
    semantics_path: Path,
    raw_retention_path: Path,
    current_trace_path: Path,
) -> dict[str, Any]:
    """Seal the current, deliberately blocked outer receipt without a child.

    The feasibility document is derived in memory from sealed metadata only.
    Its expected refusal is intentional: the present semantics record is a
    metadata-only attester, not the future exact execution attestation, and
    there is no streamed executor to receive a future lease.
    """
    source, source_path = _sealed_json(source_contract_path, label="source three-way contract")
    trace = _trace_from_source_contract(source)
    memory, memory_path = _sealed_json(memory_preflight_path, label="strict source memory preflight")
    memory_state = _validate_memory_binding(source=source, source_path=source_path, memory=memory)
    range_document, range_path = _metadata_json(range_authority_path, label="source range authority")
    authority, maximum_window_bytes = _validate_range_authority(document=range_document)
    semantics, semantics_path = _metadata_json(semantics_path, label="streamed operator semantics")
    semantics_evidence = _validate_semantics(
        document=semantics,
        path=semantics_path,
        range_document=range_document,
        range_path=range_path,
        source=source,
        source_path=source_path,
    )
    raw, raw_path = _sealed_json(raw_retention_path, label="raw six-vector retention")
    raw_binding = _validate_raw_retention(
        raw=raw, raw_path=raw_path, memory=memory, memory_path=memory_path, trace=trace
    )
    current, current_path = _sealed_json(current_trace_path, label="current all-layer native trace")
    current_evidence = _validate_current_trace(
        current=current, current_path=current_path, raw=raw, trace=trace
    )
    # No semantics execution attestation is supplied to the feasibility
    # preparer.  That is exactly the current truthful condition.
    derived = feasibility.build_feasibility(
        source_contract_path=source_path,
        whole_model_preflight_path=memory_path,
        semantics_attestation_path=None,
    )
    if derived.get("schema") != feasibility.SCHEMA or derived.get("status") != feasibility.REFUSED_STATUS:
        raise GuardedStreamedSourceOuterError("current derived streamed feasibility must remain refused")
    feasibility_state = _mapping(derived.get("feasibility"), label="derived streamed feasibility state")
    if feasibility_state.get("oracle_execution_authorized") is not False:
        raise GuardedStreamedSourceOuterError("derived streamed feasibility may not authorize an oracle")
    working_set = _mapping(derived.get("working_set"), label="derived streamed working set")
    streamed_floor = _integer(
        working_set.get("minimum_reclaimable_bytes_required_for_streamed_plan"),
        label="derived streamed safety floor",
        minimum=1,
    )
    if maximum_window_bytes <= 0:
        raise GuardedStreamedSourceOuterError("range authority did not produce a cache bound")
    return seal(
        {
            "schema": SCHEMA,
            "status": BLOCKED_STATUS,
            "recorded_at": _utc_now(),
            "source_bf16_three_way_contract": _evidence(source_path, source),
            "strict_source_bf16_memory_preflight": _evidence(memory_path, memory),
            "metadata_only_range_authority": {
                "path": str(range_path),
                "sha256": _sha256_file(range_path),
                "authority_content_sha256": range_document["authority_content_sha256"],
                "source_index_sha256": _mapping(authority.get("source"), label="range source")["source_index"]["sha256"],
                "maximum_declared_single_bf16_row_window_bytes": maximum_window_bytes,
            },
            "metadata_only_operator_semantics": semantics_evidence,
            "raw_final_logit_retention": raw_binding,
            "current_native_trace_reference_only": current_evidence,
            "exact_trace": trace,
            "derived_current_streamed_feasibility": {
                "schema": derived["schema"],
                "status": derived["status"],
                "seal_sha256": derived["seal_sha256"],
                "streamed_memory_arithmetic_fits": _mapping(
                    derived.get("memory_assessment"), label="derived streamed memory assessment"
                )["streamed_memory_arithmetic_fits"],
                "zero_swap_in_historical_snapshot": _mapping(
                    derived.get("memory_assessment"), label="derived streamed memory assessment"
                )["zero_swap_condition_met"],
                "semantic_equivalence_proven": feasibility_state[
                    "semantic_equivalence_proven_by_external_sealed_attestation"
                ],
                "oracle_execution_authorized": False,
            },
            "current_memory_snapshot_reference_only": memory_state,
            "future_source_launch_contract": {
                "actual_streamed_executor_present": False,
                "this_module_has_no_child_launch_surface": True,
                "future_executor_must_be_separate_and_receipt_last": True,
                "source_lease": {"schema": SOURCE_LEASE_SCHEMA, "status": SOURCE_LEASE_STATUS},
                "source_terminal": {"schema": SOURCE_TERMINAL_SCHEMA, "status": SOURCE_TERMINAL_STATUS},
                "source_eviction": {"schema": SOURCE_EVICTION_SCHEMA, "status": SOURCE_EVICTION_STATUS},
                "native_lease": {"schema": NATIVE_LEASE_SCHEMA, "status": NATIVE_LEASE_STATUS},
                "minimum_reclaimable_bytes_required_immediately_before_source_child": streamed_floor,
                "zero_swap_and_zero_swapouts_required_immediately_before_every_child": True,
                "maximum_source_reader_cached_windows": 1,
                "maximum_source_reader_cached_bytes": maximum_window_bytes,
                "source_payload_read_accounting_must_be_explicit": True,
                "source_then_durable_eviction_then_native_is_mandatory": True,
                "automatic_retry_or_receipt_replay_forbidden": True,
                "future_validation_is_metadata_only_and_never_launches_a_child": True,
            },
            "blockers": [
                "no actual layer-streamed Q30 source executor or receipt-last child protocol exists",
                "no exact execution semantics attestation has been earned",
                "no fresh zero-swap/zero-swapout source safety receipt and one-shot source lease exist",
                "no source capture/eviction terminal evidence exists",
                "no fresh native lease may be considered before source eviction",
            ],
            "claim_boundary": {
                "metadata_only_preflight": True,
                "no_child_launched": True,
                "does_not_open_source_tensor_payloads_or_load_a_source_model": True,
                "does_not_create_gpu_metal_or_mps_context": True,
                "does_not_touch_qwen30_server_hcli_or_watcher": True,
                "does_not_issue_a_source_or_native_lease": True,
                "does_not_claim_source_quality_coherence_hcli_tps_tg_or_tournament": True,
            },
        }
    )


def _one_shot_lifecycle(lease: Mapping[str, Any], *, label: str) -> str:
    lifecycle = _mapping(lease.get("one_shot_lifecycle"), label=f"{label} one-shot lifecycle")
    _require_true(lifecycle.get("fresh_for_this_exact_launch"), label=f"{label} freshness")
    if lifecycle.get("prior_terminal_receipt") is not None:
        raise GuardedStreamedSourceOuterError(f"{label} must not reuse a terminal receipt")
    if lifecycle.get("automatic_retry_allowed") is not False:
        raise GuardedStreamedSourceOuterError(f"{label} must forbid automatic retry")
    _require_true(lifecycle.get("new_capture_root"), label=f"{label} new capture root")
    _require_true(lifecycle.get("existing_output_reuse_forbidden"), label=f"{label} output reuse prohibition")
    _require_true(lifecycle.get("replay_or_relaunch_forbidden"), label=f"{label} replay/relaunch prohibition")
    return _text(lifecycle.get("exact_launch_nonce"), label=f"{label} exact launch nonce", sha256=True)


def validate_fresh_zero_swap_safety(
    *, lease: Mapping[str, Any], minimum_reclaimable_bytes: int, label: str
) -> dict[str, int]:
    """Validate future pre-child safety evidence without starting any child."""
    safety = _mapping(lease.get("fresh_pre_child_safety"), label=f"{label} fresh pre-child safety")
    _require_true(safety.get("observed_immediately_before_child"), label=f"{label} observation freshness")
    _require_true(safety.get("exclusive_clean_window"), label=f"{label} exclusive clean window")
    _require_true(safety.get("no_source_or_native_model_body_resident_before_child"), label=f"{label} no co-residency")
    if _integer(safety.get("swap_used_bytes"), label=f"{label} swap used") != 0:
        raise GuardedStreamedSourceOuterError(f"{label} must have zero swap before child")
    if _integer(safety.get("swapouts_pages_delta"), label=f"{label} swapouts delta") != 0:
        raise GuardedStreamedSourceOuterError(f"{label} must have zero new swapouts before child")
    measured = _integer(safety.get("reclaimable_bytes"), label=f"{label} reclaimable bytes")
    declared_minimum = _integer(
        safety.get("minimum_reclaimable_bytes_required"),
        label=f"{label} declared reclaimable floor",
        minimum=minimum_reclaimable_bytes,
    )
    if declared_minimum != minimum_reclaimable_bytes:
        raise GuardedStreamedSourceOuterError(f"{label} reclaimable floor differs from its bound contract")
    if measured < declared_minimum:
        raise GuardedStreamedSourceOuterError(f"{label} reclaimable memory is below its fresh safety floor")
    return {"reclaimable_bytes": measured, "minimum_reclaimable_bytes_required": declared_minimum}


def _validate_source_read_accounting(
    *, terminal: Mapping[str, Any], authority: Mapping[str, Any], maximum_window_bytes: int
) -> dict[str, int]:
    cache = _mapping(terminal.get("bounded_per_read_cache"), label="source terminal bounded cache")
    if _integer(cache.get("maximum_allowed_window_bytes"), label="source cache declared maximum", minimum=1) != maximum_window_bytes:
        raise GuardedStreamedSourceOuterError("source cache maximum differs from range authority")
    if _integer(cache.get("maximum_observed_window_bytes"), label="source cache observed maximum") > maximum_window_bytes:
        raise GuardedStreamedSourceOuterError("source cache exceeded its per-read range bound")
    if _integer(cache.get("maximum_cached_bytes"), label="source cache maximum bytes") > maximum_window_bytes:
        raise GuardedStreamedSourceOuterError("source cache retained more than one bounded read window")
    if _integer(cache.get("maximum_cached_windows"), label="source cache maximum windows", minimum=1) != 1:
        raise GuardedStreamedSourceOuterError("source cache must retain at most one window")
    _require_true(cache.get("eviction_on_each_read_completion"), label="source per-read eviction")
    for key in ("complete_source_shard_mapped_or_cached", "mmap_or_memory_map_used"):
        if cache.get(key) is not False:
            raise GuardedStreamedSourceOuterError(f"source cache {key} must remain false")
    accounting = _mapping(terminal.get("source_payload_read_accounting"), label="source payload read accounting")
    _require_true(accounting.get("all_source_payload_reads_accounted"), label="source payload accounting completeness")
    _require_true(accounting.get("source_tensor_payload_reads_executed"), label="source payload read execution")
    total_bytes = _integer(accounting.get("source_tensor_payload_bytes_read"), label="source payload bytes", minimum=1)
    total_calls = _integer(accounting.get("source_tensor_payload_read_calls"), label="source payload read calls", minimum=1)
    known_shards = {
        _text(_mapping(item, label="range authority shard").get("relative_path"), label="range shard path")
        for item in _sequence(authority.get("shards"), label="range authority shards")
    }
    rows = _sequence(accounting.get("per_shard"), label="source payload per-shard accounting")
    if not rows:
        raise GuardedStreamedSourceOuterError("source payload accounting must list accessed shards")
    observed_shards: set[str] = set()
    summed_bytes = 0
    summed_calls = 0
    for index, row in enumerate(rows):
        item = _mapping(row, label=f"source payload shard accounting {index}")
        shard = _text(item.get("relative_path"), label="source payload shard relative path")
        if shard not in known_shards or shard in observed_shards:
            raise GuardedStreamedSourceOuterError("source payload accounting names an unknown/repeated shard")
        observed_shards.add(shard)
        payload_bytes = _integer(item.get("payload_bytes_read"), label="source shard payload bytes")
        calls = _integer(item.get("read_calls"), label="source shard read calls")
        if payload_bytes > 0 and calls == 0:
            raise GuardedStreamedSourceOuterError("source payload bytes require at least one read call")
        if item.get("whole_shard_read_as_one_window") is not False or item.get("whole_shard_cached") is not False:
            raise GuardedStreamedSourceOuterError("source stream may not read/cache a whole shard as one window")
        summed_bytes += payload_bytes
        summed_calls += calls
    if summed_bytes != total_bytes or summed_calls != total_calls:
        raise GuardedStreamedSourceOuterError("source payload accounting totals do not match per-shard rows")
    return {"payload_bytes_read": total_bytes, "payload_read_calls": total_calls}


def validate_future_source_then_evict_then_native(
    *,
    source_lease: Mapping[str, Any],
    source_terminal: Mapping[str, Any],
    source_eviction: Mapping[str, Any],
    native_lease: Mapping[str, Any],
    authority: Mapping[str, Any],
    raw_retention: Mapping[str, Any],
    trace: Mapping[str, Any],
    source_minimum_reclaimable_bytes: int,
    maximum_window_bytes: int,
) -> dict[str, Any]:
    """Validate a hypothetical future sequence; this never launches a child."""
    source_lease = _verified_future_receipt(source_lease, label="source lease")
    source_terminal = _verified_future_receipt(source_terminal, label="source terminal")
    source_eviction = _verified_future_receipt(source_eviction, label="source eviction")
    native_lease = _verified_future_receipt(native_lease, label="native lease")
    _schema_status(source_lease, schema=SOURCE_LEASE_SCHEMA, status=SOURCE_LEASE_STATUS, label="source lease")
    source_nonce = _one_shot_lifecycle(source_lease, label="source lease")
    source_safety = validate_fresh_zero_swap_safety(
        lease=source_lease,
        minimum_reclaimable_bytes=source_minimum_reclaimable_bytes,
        label="source lease",
    )
    _schema_status(source_terminal, schema=SOURCE_TERMINAL_SCHEMA, status=SOURCE_TERMINAL_STATUS, label="source terminal")
    source_terminal_lease = _mapping(source_terminal.get("source_lease"), label="source terminal lease pointer")
    if source_terminal_lease.get("seal_sha256") != source_lease.get("seal_sha256"):
        raise GuardedStreamedSourceOuterError("source terminal does not bind the fresh source lease")
    source_execution = _mapping(source_terminal.get("streamed_execution"), label="source streamed execution")
    if source_execution.get("mode") != "layer_streamed_bf16_source_teacher":
        raise GuardedStreamedSourceOuterError("source terminal is not a layer-streamed BF16 teacher capture")
    _require_true(source_execution.get("outer_reaped_child_before_terminal_receipt"), label="source outer reaped child")
    _require_true(source_execution.get("receipt_written_after_payload_fsyncs"), label="source receipt-last payload fsync")
    _assert_trace(_mapping(source_terminal.get("exact_trace"), label="source terminal exact trace"), trace, label="source terminal")
    read_accounting = _validate_source_read_accounting(
        terminal=source_terminal, authority=authority, maximum_window_bytes=maximum_window_bytes
    )
    source_payloads = _mapping(source_terminal.get("source_payloads"), label="source terminal retained payloads")
    # This reuses the established six-vector geometry/name validation but only
    # looks at receipt metadata; no vector is opened here.
    outer_like_contract = {"six_vector_retention_contract": raw_retention["six_vector_retention_contract"]}
    from lab.operators import ascension_qwen30_quality_repack_raw_final_logit_retention_outer_controller as raw_outer

    raw_outer.validate_source_payload_retention(contract=outer_like_contract, source_payloads=source_payloads)
    _schema_status(source_eviction, schema=SOURCE_EVICTION_SCHEMA, status=SOURCE_EVICTION_STATUS, label="source eviction")
    eviction_pointer = _mapping(source_eviction.get("source_teacher_terminal"), label="source eviction terminal pointer")
    if eviction_pointer.get("seal_sha256") != source_terminal.get("seal_sha256"):
        raise GuardedStreamedSourceOuterError("source eviction does not bind the source terminal")
    eviction = _mapping(source_eviction.get("eviction"), label="source eviction facts")
    for key in (
        "source_weights_evicted",
        "source_backend_shutdown",
        "source_model_residency_released",
        "streamed_reader_cache_cleared",
        "source_payloads_durable_and_immutable",
        "swap_remained_zero",
        "pre_native_lease_process_tree_checked",
    ):
        _require_true(eviction.get(key), label=f"source eviction {key}")
    _schema_status(native_lease, schema=NATIVE_LEASE_SCHEMA, status=NATIVE_LEASE_STATUS, label="native lease")
    native_nonce = _one_shot_lifecycle(native_lease, label="native lease")
    if native_nonce == source_nonce:
        raise GuardedStreamedSourceOuterError("source/native leases may not reuse a launch nonce")
    native_safety = validate_fresh_zero_swap_safety(
        lease=native_lease,
        minimum_reclaimable_bytes=_integer(
            _mapping(native_lease.get("fresh_pre_child_safety"), label="native lease safety").get(
                "minimum_reclaimable_bytes_required"
            ),
            label="native lease declared floor",
            minimum=1,
        ),
        label="native lease",
    )
    native_pointer = _mapping(native_lease.get("source_eviction"), label="native lease source eviction pointer")
    if native_pointer.get("seal_sha256") != source_eviction.get("seal_sha256"):
        raise GuardedStreamedSourceOuterError("native lease was not issued after this source eviction")
    native_raw = _mapping(native_lease.get("raw_retention_contract"), label="native lease raw retention pointer")
    if native_raw.get("seal_sha256") != raw_retention.get("seal_sha256"):
        raise GuardedStreamedSourceOuterError("native lease raw retention binding drifted")
    return {
        "validated_order": ["source_streamed", "source_evicted", "native_lease"],
        "source_safety": source_safety,
        "native_safety": native_safety,
        "source_payload_read_accounting": read_accounting,
        "metadata_validation_only_no_child_launched": True,
    }


def _write_new_json(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise GuardedStreamedSourceOuterError("--out must be absolute")
    if path.exists():
        raise GuardedStreamedSourceOuterError("--out must name a new immutable receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    except FileExistsError as exc:
        raise GuardedStreamedSourceOuterError("--out must name a new immutable receipt") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-contract", type=Path, default=DEFAULT_SOURCE_CONTRACT)
    parser.add_argument("--memory-preflight", type=Path, default=DEFAULT_MEMORY_PREFLIGHT)
    parser.add_argument("--range-authority", type=Path, default=DEFAULT_RANGE_AUTHORITY)
    parser.add_argument("--semantics", type=Path, default=DEFAULT_SEMANTICS)
    parser.add_argument("--raw-retention", type=Path, default=DEFAULT_RAW_RETENTION)
    parser.add_argument("--current-trace", type=Path, default=DEFAULT_CURRENT_TRACE)
    parser.add_argument("--out", type=Path, required=True, help="new absolute blocked receipt path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_current_preflight(
            source_contract_path=args.source_contract,
            memory_preflight_path=args.memory_preflight,
            range_authority_path=args.range_authority,
            semantics_path=args.semantics,
            raw_retention_path=args.raw_retention,
            current_trace_path=args.current_trace,
        )
        _write_new_json(args.out, result)
    except GuardedStreamedSourceOuterError as exc:
        print(f"Q30 guarded streamed-source outer controller refused: {exc}")
        return 2
    print(json.dumps({"output": str(args.out.resolve()), "status": result["status"], "seal_sha256": result["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

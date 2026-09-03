"""Prepare, but never run, Qwen30's source-BF16 three-way logit oracle.

The existing HQ30GR2 all-layer capture retained SHA-256 witnesses and bounded
top-k values, not the full F32 logits needed for a numerical distance to the
source model.  This CPU-only preparer binds the exact current trace, source
identity, candidate/control evidence, output geometry, and the future
capture's strict acceptance predicate.  It does *not* load a source shard,
instantiate a source model, use Metal, start HCLI, or take a resource lease.

The future oracle is intentionally a teacher/reference measurement only.  It
is not a Qwen30 production runtime or a coherence/TPS/capability result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates"
    / "gate-up-residual-v1"
)
DEFAULT_COMPARISON = (
    DEFAULT_ROOT
    / "all-layer-current-trace-comparison/receipts"
    / "QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_COMPARISON_"
    "883c59eec0371ebb6d4a9935cdbdc6bcb486c03eebd5312db608a0415a34911f_"
    "98db412d42e87e938a89c759d426f69bd51e700a944701704a5c82949244298f.json"
)
DEFAULT_SOURCE_SNAPSHOT = DEFAULT_ROOT / "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_SOURCE_BINDING_SNAPSHOT.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "source-bf16-three-way-final-logit-contract/receipts"

SCHEMA = "hawking.ascension.qwen30_hq30gr2_source_bf16_three_way_final_logit_contract.v1"
STATUS = "PREPARED_SOURCE_BF16_THREE_WAY_FINAL_LOGIT_DISTANCE_CONTRACT_NOT_RUN"

COMPARISON_SCHEMA = "hawking.ascension.qwen30_hq30gr2_all_layer_current_trace_comparison.v1"
COMPARISON_STATUS = "EARNED_CANDIDATE_LOCAL_ALL_LAYER_DIVERGENCE_UNQUALIFIED_NON_PROMOTABLE"
SOURCE_SNAPSHOT_SCHEMA = "hawking.ascension.qwen30_quality_repack_source_snapshot.v1"
SOURCE_SNAPSHOT_STATUS = "EARNED_IMMUTABLE_SOURCE_AND_ROLLBACK_BINDING"
SOURCE_REVALIDATION_SCHEMA = "hawking.ascension.complete_binary_source_revalidation.v1"
SOURCE_REVALIDATION_STATUS = "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED"
SOURCE_AUDIT_SCHEMA = "hawking.ascension.qwen30_source_body_audit_candidate.v1"
SOURCE_AUDIT_STATUS = "CANDIDATE_SOURCE_BODY_VERIFIED"
INNER_SCHEMA = "hawking.ascension.qwen30_quality_repack_all_layer_current_trace_diagnostic.v1"
INNER_STATUS = "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_TYPED_HQ30GR2_ALL_LAYER_CURRENT_TRACE_UNQUALIFIED"
COMPILER_SCHEMA = "hawking.ascension.qwen30_quality_repack_current_hcli_compiler_trace.v1"
COMPILER_STATUS = "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_PRE_EXECUTION_HCLI_COMPILER_TRACE"

PROBE_ID = "literal_hawking"
PREFIX_TOKENS = 369
FORCED_TOKEN = 949
VOCAB_ROWS = 151_936
F32_BYTES = 4
ENDPOINTS = ("exact_prefix", "forced_shared_continuation")
MODELS = ("source_bf16", "scalar_control", "hq30gr2_candidate")


class SourceOracleContractError(RuntimeError):
    """The prospective oracle does not have complete, source-bound inputs."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sealed(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        checked = verify(raw, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise SourceOracleContractError(f"{label} is absent or invalid: {exc}") from exc
    if not isinstance(checked, Mapping):
        raise SourceOracleContractError(f"{label} is not an object")
    return dict(checked)


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceOracleContractError(f"{label} must be an object")
    return dict(value)


def _text(value: object, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise SourceOracleContractError(f"{label} must be a non-empty string")
    if sha256 and (len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
        raise SourceOracleContractError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SourceOracleContractError(f"{label} must be an integer >= {minimum}")
    return value


def _assert_schema_status(document: Mapping[str, Any], *, schema: str, status: str, label: str) -> None:
    if document.get("schema") != schema or document.get("status") != status:
        raise SourceOracleContractError(f"{label} schema/status does not match the required source-bound record")


def _evidence(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        stat_result = path.stat()
    except OSError as exc:
        raise SourceOracleContractError(f"cannot stat evidence {path}: {exc}") from exc
    return {
        "path": str(path.resolve()),
        "bytes": stat_result.st_size,
        "sha256": _sha256_file(path),
        "seal_sha256": _text(document.get("seal_sha256"), label="evidence seal", sha256=True),
    }


def _file_identity(path: Path) -> dict[str, int]:
    try:
        stat_result = path.stat()
    except OSError as exc:
        raise SourceOracleContractError(f"cannot stat source file {path}: {exc}") from exc
    return {
        "bytes": stat_result.st_size,
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "mtime_ns": stat_result.st_mtime_ns,
        "ctime_ns": stat_result.st_ctime_ns,
    }


def _under(root: Path, relative: str, *, label: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise SourceOracleContractError(f"{label} escapes its sealed root") from exc
    return candidate


def _u32le_sha256(token_ids: Sequence[int]) -> str:
    try:
        raw = b"".join(struct.pack("<I", token_id) for token_id in token_ids)
    except struct.error as exc:
        raise SourceOracleContractError(f"token IDs are not U32 values: {exc}") from exc
    return _sha256_bytes(raw)


def _raw_vector_requirements(*, vocab_rows: int) -> dict[str, Any]:
    bytes_per_vector = vocab_rows * F32_BYTES
    payloads = [f"{model}_{endpoint}_logits.f32le" for model in MODELS for endpoint in ENDPOINTS]
    return {
        "dtype": "f32le",
        "vocab_rows": vocab_rows,
        "bytes_per_full_logit_vector": bytes_per_vector,
        "required_payload_count": len(payloads),
        "required_total_payload_bytes": bytes_per_vector * len(payloads),
        "payload_filenames": payloads,
        "all_values_must_be_finite": True,
        "source_control_candidate_payloads_must_use_identical_token_sequence_and_forced_token": True,
    }


def _relative_l2(source: np.ndarray, observed: np.ndarray) -> float:
    """Future capture's preregistered metric, accumulated in F64."""
    if source.dtype != np.float32 or observed.dtype != np.float32 or source.shape != observed.shape:
        raise SourceOracleContractError("three-way logit metric requires equal F32 vectors")
    if not np.isfinite(source).all() or not np.isfinite(observed).all():
        raise SourceOracleContractError("three-way logit metric refuses non-finite values")
    denominator = float(np.linalg.norm(source.astype(np.float64)))
    if denominator == 0.0 or not math.isfinite(denominator):
        raise SourceOracleContractError("source final logits have zero/non-finite L2 norm")
    numerator = float(np.linalg.norm(observed.astype(np.float64) - source.astype(np.float64)))
    if not math.isfinite(numerator):
        raise SourceOracleContractError("three-way logit metric numerator is non-finite")
    return numerator / denominator


def evaluate_three_way_vectors(
    *, source: np.ndarray, control: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    """Pure future-result predicate; this does not execute a model."""
    control_error = _relative_l2(source, control)
    candidate_error = _relative_l2(source, candidate)
    return {
        "metric": "relative_l2_f64_accumulation",
        "source_to_control_relative_l2": control_error,
        "source_to_candidate_relative_l2": candidate_error,
        "candidate_strictly_improves_over_control": candidate_error < control_error,
        "acceptance": "PASS_ONLY_IF_CANDIDATE_STRICTLY_LESS_THAN_CONTROL",
    }


def _comparison_and_inner(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    comparison = _sealed(path, label="candidate-local all-layer comparison")
    _assert_schema_status(comparison, schema=COMPARISON_SCHEMA, status=COMPARISON_STATUS, label="candidate-local all-layer comparison")
    evidence = _object(comparison.get("evidence"), label="comparison evidence")
    inner_ref = _object(evidence.get("inner_all_layer_diagnostic"), label="comparison inner diagnostic reference")
    inner_path = Path(_text(inner_ref.get("path"), label="comparison inner diagnostic path"))
    inner = _sealed(inner_path, label="sealed all-layer inner diagnostic")
    _assert_schema_status(inner, schema=INNER_SCHEMA, status=INNER_STATUS, label="sealed all-layer inner diagnostic")
    if _sha256_file(inner_path) != _text(inner_ref.get("sha256"), label="comparison inner diagnostic SHA", sha256=True):
        raise SourceOracleContractError("comparison no longer binds the inner diagnostic bytes")
    if inner.get("seal_sha256") != inner_ref.get("seal_sha256"):
        raise SourceOracleContractError("comparison no longer binds the inner diagnostic seal")
    return comparison, inner_path, inner


def _exact_trace(inner: Mapping[str, Any], comparison: Mapping[str, Any]) -> tuple[Path, dict[str, Any], list[int]]:
    upstream = _object(inner.get("upstream_diagnostic_binding"), label="inner upstream diagnostic binding")
    compiler_ref = _object(upstream.get("compiler_trace_receipt"), label="inner compiler trace reference")
    compiler_path = Path(_text(compiler_ref.get("path"), label="compiler trace path"))
    compiler = _sealed(compiler_path, label="current compiler trace receipt")
    _assert_schema_status(compiler, schema=COMPILER_SCHEMA, status=COMPILER_STATUS, label="current compiler trace receipt")
    if compiler.get("seal_sha256") != compiler_ref.get("seal_sha256"):
        raise SourceOracleContractError("inner diagnostic compiler binding seal differs")
    if _sha256_file(compiler_path) != _text(compiler_ref.get("sha256"), label="inner compiler trace SHA", sha256=True):
        raise SourceOracleContractError("inner diagnostic compiler binding bytes differ")
    binding = _object(compiler.get("binding"), label="compiler trace binding")
    run_root = Path(_text(binding.get("run_root"), label="compiler trace run root"))
    traces = compiler.get("public_probe_compiler_traces")
    if not isinstance(traces, list):
        raise SourceOracleContractError("compiler trace has no public probe list")
    matches = [row for row in traces if isinstance(row, Mapping) and row.get("probe_id") == PROBE_ID]
    if len(matches) != 1:
        raise SourceOracleContractError("compiler trace does not have exactly one literal_hawking probe")
    annotated_relative = _text(matches[0].get("annotated_trace_path"), label="literal_hawking annotated trace path")
    annotated_path = _under(run_root, annotated_relative, label="literal_hawking annotated trace")
    if _sha256_file(annotated_path) != _text(matches[0].get("annotated_trace_sha256"), label="annotated trace SHA", sha256=True):
        raise SourceOracleContractError("literal_hawking annotated trace bytes differ from compiler receipt")
    try:
        annotated_raw = json.loads(annotated_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceOracleContractError(f"literal_hawking annotated trace is unreadable: {exc}") from exc
    annotated = _object(annotated_raw, label="literal_hawking annotated trace")
    annotations = _object(annotated.get("source_tokenizer_annotations"), label="source tokenizer annotations")
    source_prompt = _object(annotations.get("source_one_user_native_prompt"), label="source one-user token annotation")
    token_ids = source_prompt.get("token_ids")
    if not isinstance(token_ids, list) or len(token_ids) != PREFIX_TOKENS:
        raise SourceOracleContractError("literal_hawking source token annotation does not contain the exact 369 IDs")
    tokens = [_integer(token, label="literal_hawking source token", minimum=0) for token in token_ids]
    token_hash = _u32le_sha256(tokens)
    if token_hash != _text(source_prompt.get("token_ids_u32le_sha256"), label="annotated token SHA", sha256=True):
        raise SourceOracleContractError("literal_hawking annotated token IDs do not match their declared SHA")
    comparison_binding = _object(comparison.get("binding"), label="comparison binding")
    if token_hash != _text(comparison_binding.get("source_template_token_ids_u32le_sha256"), label="comparison token SHA", sha256=True):
        raise SourceOracleContractError("literal_hawking compiler trace does not match the all-layer comparison tokens")
    if _integer(comparison_binding.get("source_template_token_count"), label="comparison token count") != PREFIX_TOKENS:
        raise SourceOracleContractError("all-layer comparison does not bind 369 source-template tokens")
    if _integer(comparison_binding.get("forced_identical_continuation_token_id"), label="comparison forced token") != FORCED_TOKEN:
        raise SourceOracleContractError("all-layer comparison forced token is not the current sealed control token")
    return compiler_path, compiler, tokens


def _source_binding(path: Path, *, expected_candidate_seal: str, vocab_rows: int) -> dict[str, Any]:
    snapshot = _sealed(path, label="HQ30GR2 source binding snapshot")
    _assert_schema_status(snapshot, schema=SOURCE_SNAPSHOT_SCHEMA, status=SOURCE_SNAPSHOT_STATUS, label="HQ30GR2 source binding snapshot")
    binding = _object(snapshot.get("binding"), label="source snapshot binding")
    selected = binding.get("selected_organs")
    expected_organs = [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
    ]
    if selected != expected_organs:
        raise SourceOracleContractError("source snapshot does not bind exactly the HQ30GR2 L0/E0 gate/up candidate")
    revalidation_ref = _object(binding.get("immutable_source_revalidation"), label="source snapshot revalidation")
    revalidation_path = Path(_text(revalidation_ref.get("path"), label="source revalidation path"))
    revalidation = _sealed(revalidation_path, label="source shard revalidation")
    _assert_schema_status(revalidation, schema=SOURCE_REVALIDATION_SCHEMA, status=SOURCE_REVALIDATION_STATUS, label="source shard revalidation")
    if revalidation.get("seal_sha256") != revalidation_ref.get("seal_sha256"):
        raise SourceOracleContractError("source snapshot revalidation pointer seal differs")
    if _sha256_file(revalidation_path) != _text(revalidation_ref.get("document_sha256"), label="source revalidation document SHA", sha256=True):
        raise SourceOracleContractError("source snapshot revalidation pointer bytes differ")
    source_dir = Path(_text(revalidation.get("source_model_dir"), label="source model directory"))
    source_revision = _text(revalidation.get("source_revision"), label="source revision")
    observed_total = _integer(revalidation.get("observed_total_bytes"), label="source total bytes", minimum=1)
    shards = _object(revalidation.get("shards"), label="source revalidated shards")
    if len(shards) != _integer(revalidation.get("sealed_shard_count"), label="sealed source shard count", minimum=1):
        raise SourceOracleContractError("source shard count differs from revalidation")
    expected_total = 0
    for name, row_raw in sorted(shards.items()):
        if not isinstance(name, str) or not name.endswith(".safetensors"):
            raise SourceOracleContractError("source revalidation contains an invalid shard name")
        row = _object(row_raw, label=f"source shard {name}")
        expected_bytes = _integer(row.get("expected_bytes"), label=f"source shard {name} bytes", minimum=1)
        if row.get("expected_sha256") != row.get("observed_sha256"):
            raise SourceOracleContractError(f"source shard {name} was not fully revalidated")
        expected_identity = _object(row.get("file_identity"), label=f"source shard {name} identity")
        observed_identity = _file_identity(source_dir / name)
        if observed_identity != expected_identity:
            raise SourceOracleContractError(f"source shard {name} identity drifted after revalidation")
        expected_total += expected_bytes
    if expected_total != observed_total:
        raise SourceOracleContractError("source revalidation shard bytes do not sum to its observed total")
    index_path = Path(_text(revalidation.get("index_path"), label="source index path"))
    if _sha256_file(index_path) != _text(revalidation.get("index_sha256"), label="source index SHA", sha256=True):
        raise SourceOracleContractError("source safetensors index SHA differs from revalidation")
    audit_ref = _object(binding.get("source_audit"), label="source snapshot audit reference")
    audit_path = Path(_text(audit_ref.get("path"), label="source audit path"))
    audit = _sealed(audit_path, label="source audit")
    _assert_schema_status(audit, schema=SOURCE_AUDIT_SCHEMA, status=SOURCE_AUDIT_STATUS, label="source audit")
    if audit.get("seal_sha256") != audit_ref.get("seal_sha256"):
        raise SourceOracleContractError("source snapshot audit pointer seal differs")
    if _sha256_file(audit_path) != _text(audit_ref.get("document_sha256"), label="source audit document SHA", sha256=True):
        raise SourceOracleContractError("source snapshot audit pointer bytes differ")
    audit_source = _object(audit.get("source"), label="source audit source")
    if Path(_text(audit_source.get("model_dir"), label="source audit model directory")).resolve() != source_dir.resolve():
        raise SourceOracleContractError("source audit model directory differs from source revalidation")
    if audit_source.get("revision") != source_revision or audit_source.get("total_bytes") != observed_total:
        raise SourceOracleContractError("source audit revision or bytes differ from source revalidation")
    config_path = source_dir / "config.json"
    try:
        config_raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceOracleContractError(f"source config is unreadable: {exc}") from exc
    config = _object(config_raw, label="source config")
    required = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 48,
        "vocab_size": vocab_rows,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "hidden_size": 2048,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise SourceOracleContractError(f"source config {key} does not match the source-BF16 oracle contract")
    return {
        "source_snapshot": _evidence(path, snapshot),
        "source_revalidation": _evidence(revalidation_path, revalidation),
        "source_audit": _evidence(audit_path, audit),
        "source_model_dir": str(source_dir.resolve()),
        "source_revision": source_revision,
        "source_index": {"path": str(index_path.resolve()), "sha256": _sha256_file(index_path)},
        "source_config": {"path": str(config_path.resolve()), "sha256": _sha256_file(config_path), "required_fields": required},
        "source_shard_count": len(shards),
        "source_weight_bytes_exact": observed_total,
        "source_weight_gib": observed_total / (1024**3),
        "candidate_manifest_seal_sha256": expected_candidate_seal,
    }


def build_contract(*, comparison_path: Path, source_snapshot_path: Path, physical_memory_bytes: int) -> dict[str, Any]:
    """Create the static, no-model-load contract for one future oracle capture."""
    if physical_memory_bytes <= 0:
        raise SourceOracleContractError("physical memory must be positive")
    comparison, inner_path, inner = _comparison_and_inner(comparison_path)
    binding = _object(comparison.get("binding"), label="comparison binding")
    candidate_seal = _text(binding.get("candidate_manifest_seal_sha256"), label="candidate seal", sha256=True)
    compiler_path, compiler, tokens = _exact_trace(inner, comparison)
    source = _source_binding(source_snapshot_path, expected_candidate_seal=candidate_seal, vocab_rows=VOCAB_ROWS)
    raw = _raw_vector_requirements(vocab_rows=VOCAB_ROWS)
    existing_vectors = _object(_object(comparison.get("divergence"), label="comparison divergence").get("final_logit_vectors"), label="comparison final-logit witnesses")
    expected_hashes: dict[str, Any] = {}
    for endpoint in ENDPOINTS:
        row = _object(existing_vectors.get(endpoint), label=f"comparison {endpoint} logits")
        if row.get("vocab_rows") != VOCAB_ROWS:
            raise SourceOracleContractError(f"comparison {endpoint} vocab does not match source config")
        expected_hashes[endpoint] = {
            "control_full_f32le_sha256": _text(row.get("control_full_f32le_sha256"), label=f"control {endpoint} hash", sha256=True),
            "candidate_full_f32le_sha256": _text(row.get("candidate_full_f32le_sha256"), label=f"candidate {endpoint} hash", sha256=True),
        }
    source_bytes = source["source_weight_bytes_exact"]
    source_fraction = source_bytes / physical_memory_bytes
    contract = {
        "schema": SCHEMA,
        "status": STATUS,
        "recorded_at": _utc_now(),
        "evidence": {
            "candidate_local_comparison": _evidence(comparison_path, comparison),
            "all_layer_inner_diagnostic": _evidence(inner_path, inner),
            "compiler_trace_receipt": _evidence(compiler_path, compiler),
            **source,
        },
        "exact_input": {
            "probe_id": PROBE_ID,
            "source_template_token_count": PREFIX_TOKENS,
            "source_template_token_ids_u32le_sha256": _u32le_sha256(tokens),
            "literal_token_ids_are_bound_in_compiler_trace_but_not_duplicated_here": True,
            "forced_identical_continuation_token_id": FORCED_TOKEN,
            "source_must_execute_the_same_369_token_prefix_then_the_forced_token": True,
            "sampling_or_autoregressive_feedback_is_forbidden": True,
        },
        "future_capture": {
            "source_reference": {
                "required": "official sealed source BF16 Qwen3MoeForCausalLM only",
                "role": "teacher_oracle_only_not_a_production_qwen30_result",
                "full_token_forwards": PREFIX_TOKENS + 1,
                "must_return_full_f32_logits_after_prefix_and_forced_continuation": True,
            },
            "native_control_and_candidate": {
                "new_capture_required": True,
                "reason": "The completed 98db all-layer receipt retained full-logit hashes/top-k only, not raw F32 values; hashes cannot produce a numerical distance.",
                "full_token_forwards_per_path": PREFIX_TOKENS + 1,
                "control_and_candidate_raw_payload_hashes_must_match_existing_witnesses": expected_hashes,
                "typed_hq30gr2_interception_at_L0_E0_must_reoccur_and_be_witnessed": True,
            },
            "raw_output_payloads": raw,
            "metric": {
                "name": "relative_l2_f64_accumulation",
                "formula": "||observed - source||_2 / ||source||_2 over every one of 151936 F32 logits",
                "endpoints": list(ENDPOINTS),
                "acceptance": "PASS_ONLY_IF_SOURCE_TO_CANDIDATE_RELATIVE_L2_IS_STRICTLY_LESS_THAN_SOURCE_TO_CONTROL_RELATIVE_L2_AT_EACH_ENDPOINT",
                "no_top_k_only_substitution": True,
                "no_aggregate_only_substitution": True,
            },
        },
        "resource_and_capture_requirements": {
            "source_weights_static_lower_bound_bytes": source_bytes,
            "source_weights_static_lower_bound_gib": source["source_weight_gib"],
            "physical_memory_bytes_observed_at_contract_preflight": physical_memory_bytes,
            "source_weight_lower_bound_fraction_of_physical_memory": source_fraction,
            "additional_allocator_activations_kv_and_existing_model_residency_bytes": "NOT_YET_MEASURED_MUST_BE_DECLARED_AND_ACCEPTED_BY_FUTURE_OUTER_CAPTURE",
            "free_pages_or_file_cache_are_not_a_sufficient_residency_proof": True,
            "requires_fresh_exclusive_unified_memory_and_gpu_lease_if_source_backend_uses_mps_or_metal": True,
            "requires_qwen80_model_work_held_and_qwen30_server_idle_but_not_stopped": True,
            "requires_one_outer_controller_with_active_child_stdout_stderr_reaping_and_receipt_last": True,
            "requires_no_automatic_retry_and_source_weight_eviction_after_terminal_capture": True,
            "source_oracle_backend_not_selected_or_authorized_by_this_contract": True,
            "source_model_has_not_been_loaded_by_this_preflight": True,
        },
        "preflight": {
            "source_shards_stat_identity_checked_without_reading_payloads": True,
            "source_safetensors_index_and_config_checked": True,
            "exact_369_token_trace_resolved_and_hash_matched": True,
            "raw_control_candidate_full_logit_vectors_absent_from_existing_capture": True,
            "future_capture_required_before_metric_can_be_evaluated": True,
            "source_oracle_executor_not_implemented_or_run_by_this_preparer": True,
        },
        "classification": {
            "candidate_remains_non_promotable": True,
            "only_future_numerical_diagnostic_is_prepared": True,
            "even_a_metric_pass_would_not_establish_hcli_coherence_tps_capability_or_tournament_qualification": True,
        },
        "claim_boundary": {
            "cpu_only_metadata_and_integrity_preflight": True,
            "does_not_read_source_weight_payloads_or_load_a_source_model": True,
            "does_not_execute_control_or_candidate_runtime": True,
            "does_not_use_metal_or_take_a_gpu_lease": True,
            "does_not_touch_live_server_watcher_adapter_or_hcli": True,
            "does_not_emit_a_quality_oracle_result": True,
        },
    }
    return seal(contract)


def _physical_memory_bytes() -> int:
    # This is a metadata query only: it never opens a source shard nor creates
    # an accelerator context.  A future outer controller must still make its
    # own live residency decision rather than trusting this static contract.
    try:
        value = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        value = 0
    if value <= 0:
        raise SourceOracleContractError("cannot determine physical memory; pass --physical-memory-bytes explicitly")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--source-snapshot", type=Path, default=DEFAULT_SOURCE_SNAPSHOT)
    parser.add_argument("--physical-memory-bytes", type=int, default=_physical_memory_bytes())
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_contract(
            comparison_path=args.comparison,
            source_snapshot_path=args.source_snapshot,
            physical_memory_bytes=args.physical_memory_bytes,
        )
        if args.output is None:
            candidate_seal = result["evidence"]["candidate_manifest_seal_sha256"]
            output = DEFAULT_OUTPUT_ROOT / f"QWEN30_HQ30GR2_SOURCE_BF16_THREE_WAY_FINAL_LOGIT_CONTRACT_{candidate_seal}.json"
        else:
            output = args.output
        _atomic_json(output, result)
    except SourceOracleContractError as exc:
        print(f"Q30 source-BF16 three-way oracle contract refused: {exc}")
        return 2
    print(json.dumps({"output": str(output.resolve()), "status": result["status"], "seal_sha256": result["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

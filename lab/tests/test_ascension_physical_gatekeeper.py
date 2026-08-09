"""Focused fail-closed coverage for the non-V3 physical gatekeeper."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from lab.operators import ascension_physical_gatekeeper as gatekeeper
from lab.operators import ascension_physical_tournament as physical_tournament
from lab.receipts import seal, verify


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_sealed(path: Path, document: dict[str, Any]) -> dict[str, Any]:
    sealed = seal(document)
    _write_json(path, sealed)
    return sealed


def _document_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_and_revalidation(root: Path, spec: gatekeeper.ModelSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create tiny current files while preserving the exact real-model pins."""

    source_dir = root / "tiny-sources" / spec.key
    source_dir.mkdir(parents=True)
    index_path = source_dir / "model.safetensors.index.json"
    index_path.write_text('{"weight_map":{}}\n', encoding="utf-8")
    control = {
        "path": index_path.name,
        "bytes": index_path.stat().st_size,
        "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }
    weight_rows: list[dict[str, Any]] = []
    revalidated: dict[str, dict[str, Any]] = {}
    for ordinal in range(spec.shard_count):
        shard = source_dir / f"model-{ordinal + 1:05d}-of-{spec.shard_count:05d}.safetensors"
        shard.write_bytes(f"{spec.key}:{ordinal}".encode("utf-8"))
        digest = hashlib.sha256(shard.read_bytes()).hexdigest()
        file_identity = {
            "bytes": shard.stat().st_size,
            "device": shard.stat().st_dev,
            "inode": shard.stat().st_ino,
            "mtime_ns": shard.stat().st_mtime_ns,
            "ctime_ns": shard.stat().st_ctime_ns,
        }
        weight_rows.append({"path": shard.name, "bytes": shard.stat().st_size, "sha256": digest})
        revalidated[shard.name] = {
            "expected_bytes": shard.stat().st_size,
            "expected_sha256": digest,
            "observed_sha256": digest,
            "file_identity": file_identity,
        }
    content_identity = hashlib.sha256(f"identity:{spec.key}".encode("utf-8")).hexdigest()
    identity = _write_sealed(
        root / spec.key / "evolution" / "SOURCE_CONTENT_IDENTITY.json",
        {
            "schema": gatekeeper.SOURCE_IDENTITY_SCHEMA,
            "status": "IMMUTABLE_SOURCE_CONTENT_IDENTITY_BOUND",
            "content_identity_sha256": content_identity,
            "model": {
                "id": spec.model_id,
                "architecture": spec.architecture,
                "repository": spec.repository,
                "revision": spec.revision,
                "source_dir": str(source_dir),
            },
            "source_content": {
                "architecture": spec.architecture,
                "repository": spec.repository,
                "revision": spec.revision,
                "control_files": [control],
                "verified_weight_shards": weight_rows,
            },
            # This is historical provenance only.  The current full-shard
            # revalidation below is the authority for an admission receipt.
            "weight_body_audit_seal_sha256": "f" * 64,
        },
    )
    source_audit_path = root / spec.key / f"{spec.prefix}_SOURCE_BODY_AUDIT_CANDIDATE.json"
    source_audit = _write_sealed(
        source_audit_path,
        {"schema": "test.source_audit.v1", "status": "SOURCE_BODY_AUDIT_BOUND"},
    )
    revalidation_payload: dict[str, Any] = {
        "schema": gatekeeper.SOURCE_REVALIDATION_SCHEMA,
        "status": "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED",
        "source_repository": spec.repository,
        "source_revision": spec.revision,
        "source_model_dir": str(source_dir),
        "index_path": str(index_path),
        "index_sha256": control["sha256"],
        "source_audit_path": str(source_audit_path),
        "source_audit_document_sha256": _document_sha(source_audit_path),
        "source_audit_seal_sha256": source_audit["seal_sha256"],
        "sealed_shard_count": spec.shard_count,
        "sealed_shard_hashes_sha256": hashlib.sha256(
            f"shards:{spec.key}".encode("utf-8")
        ).hexdigest(),
        "weight_map_sha256": hashlib.sha256(
            f"weight-map:{spec.key}".encode("utf-8")
        ).hexdigest(),
        "shards": revalidated,
    }
    # The live Qwen30 revalidation receipt predates this optional audit-index
    # field.  Its immutable source/index binding remains complete without it.
    if spec.key != "qwen30":
        revalidation_payload["sealed_audit_index_sha256"] = control["sha256"]
    revalidation = _write_sealed(
        root / spec.key / "complete-gravity" / f"{spec.prefix}_CURRENT_SOURCE_SHARD_REVALIDATION.json",
        revalidation_payload,
    )
    return identity, revalidation


def _write_observations(root: Path, spec: gatekeeper.ModelSpec, *, heartbeat: int = 1, candidates: int = 1) -> None:
    _write_sealed(
        root / spec.key / "evolution" / f"{spec.prefix}_DUAL_GRAVITY_STATUS.json",
        {
            "schema": "hawking.ascension.dual_gravity_worker.v1",
            "status": "REAL_DETERMINISTIC_EVOLUTION_ADVANCING",
            "phase": "EVOLVING_PHYSICAL_CANDIDATE",
            "model": {"id": spec.model_id, "key": spec.key},
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "heartbeat": heartbeat,
            "population": {"candidate_count": candidates, "completed_candidate_count": candidates},
            "current_experiment": {"sequence": candidates, "candidate_id": f"{spec.key}-current-{candidates}"},
            "next_experiment": {"sequence": candidates + 1, "candidate_id": f"{spec.key}-next-{candidates}"},
            "last_material_progress_at": f"2026-08-08T00:00:0{candidates}Z",
            "complete_pack": {"progress": {"completed_tensors": candidates, "next_cursor": candidates}},
        },
    )
    _write_json(
        root / spec.key / "complete-gravity" / f"{spec.prefix}_COMPLETE_GRAVITY_STATUS.json",
        {"phase": "PACKING_COMPLETE_BINARY_GRAVITY", "heartbeat": heartbeat, "progress": {"completed_tensors": candidates, "next_cursor": candidates, "planned_tensors": 100}},
    )
    _write_json(
        root / spec.key / "complete-runtime" / f"{spec.prefix}_COMPLETE_RUNTIME_STATUS.json",
        {"phase": "WAITING_FOR_NATIVE_COMPLETE_TOKEN_RUNTIME", "heartbeat": heartbeat},
    )
    _write_json(
        root / spec.key / "tg3" / f"{spec.prefix}_TG3_ASCENT_STATUS.json",
        {"phase": "WAITING_FOR_NATIVE_COMPLETE_TOKEN_RUNTIME", "heartbeat": heartbeat},
    )


def _binding(identity: dict[str, Any], revalidation: dict[str, Any], spec: gatekeeper.ModelSpec, **extra: str) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "source_content_identity_sha256": identity["content_identity_sha256"],
        "source_revalidation_seal_sha256": revalidation["seal_sha256"],
        **extra,
    }


def _write_native_admitted_artifact(
    root: Path,
    spec: gatekeeper.ModelSpec,
    identity: dict[str, Any],
    revalidation: dict[str, Any],
    *,
    payload_bytes: int = 8,
) -> dict[str, Any]:
    """Build the public storage-only admission proof consumed by the gatekeeper."""

    paths = gatekeeper._paths(root, spec)
    source_dir = Path(identity["model"]["source_dir"])
    source_row = identity["source_content"]["verified_weight_shards"][0]
    artifact_path = paths["complete_root"] / "tensors" / f"{spec.key}-test.hqbin"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"x" * payload_bytes)
    source_weight_elements = 64
    ledger = {
        "all_required_weight_artifact_bytes": payload_bytes,
        "complete_physical_bpw": 8.0 * payload_bytes / source_weight_elements,
        "explicitly_excluded_separate_state": ["KV_cache_bytes"],
        "manifest_bytes_billed": 0,
        "passes_storage_threshold": True,
        "source_weight_elements": source_weight_elements,
        "tensor_payload_bytes": payload_bytes,
        "threshold_bpw": 1.5,
    }
    manifest = _write_sealed(
        paths["complete_root"] / f"{spec.prefix}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json",
        {
            "schema": f"hawking.ascension.{spec.key}_complete_binary_gravity.v1",
            "status": "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED",
            "source_body_audit_seal_sha256": revalidation["source_audit_seal_sha256"],
            "source_revalidation_receipt_path": str(paths["revalidation"]),
            "source_revalidation_receipt_seal_sha256": revalidation["seal_sha256"],
            "source": {
                "repository": spec.repository,
                "model_dir": str(source_dir),
                "tensor_count": 1,
            },
            "complete_physical_bpw_ledger": ledger,
            "tensors": [
                {
                    "tensor_name": "model.test.weight",
                    "elements": source_weight_elements,
                    "artifact_bytes": payload_bytes,
                    "artifact_path": str(artifact_path),
                    "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                    "shape": [source_weight_elements],
                    "source_dtype": "BF16",
                    "source_shard": source_row["path"],
                    "source_shard_sha256": source_row["sha256"],
                    "layout": {
                        "magic": "TESTHQ01",
                        "group_size": 64,
                        "scale_dtype": "float16",
                        "sign_bit_order": "little",
                        "version": 1,
                    },
                }
            ],
            "claim_boundary": {
                "complete_physical_tensor_coverage_is_true": True,
                "complete_bpw_pass_does_not_substitute_for_capability": True,
                "not_native_runtime_execution": True,
                "not_tg10_tg3_hcli_agent_os_or_manager_qualified": True,
                "raw_source_remains_authority_teacher_only": True,
            },
        },
    )
    manifest_path = paths["complete_root"] / f"{spec.prefix}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
    immutable_identity = {
        "path": str(paths["identity"]),
        "document_sha256": _document_sha(paths["identity"]),
        "seal_sha256": identity["seal_sha256"],
        "content_identity_sha256": identity["content_identity_sha256"],
        "repository": spec.repository,
        "revision": spec.revision,
        "source_dir": str(source_dir),
        "index_sha256": identity["source_content"]["control_files"][0]["sha256"],
        "historical_weight_body_audit_seal_sha256": identity["weight_body_audit_seal_sha256"],
    }
    current_revalidation = {
        "path": str(paths["revalidation"]),
        "document_sha256": _document_sha(paths["revalidation"]),
        "seal_sha256": revalidation["seal_sha256"],
        "repository": spec.repository,
        "revision": spec.revision,
        "source_model_dir": str(source_dir),
        "index_path": str(source_dir / "model.safetensors.index.json"),
        "index_sha256": revalidation["index_sha256"],
        "source_audit_path": revalidation["source_audit_path"],
        "source_audit_document_sha256": revalidation["source_audit_document_sha256"],
        "source_audit_seal_sha256": revalidation["source_audit_seal_sha256"],
        "sealed_shard_count": spec.shard_count,
        "sealed_shard_hashes_sha256": revalidation["sealed_shard_hashes_sha256"],
        "weight_map_sha256": revalidation["weight_map_sha256"],
    }
    complete_manifest = {
        "path": str(manifest_path),
        "document_sha256": _document_sha(manifest_path),
        "seal_sha256": manifest["seal_sha256"],
        "schema": manifest["schema"],
        "status": manifest["status"],
    }
    request = _write_sealed(
        paths["complete_root"]
        / "complete-admission"
        / "requests"
        / f"{spec.prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_REQUEST_{manifest['seal_sha256']}.json",
        {
            "schema": "hawking.ascension.qwen_complete_binary_gravity_admission_request.v1",
            "status": "SEALED_EXACT_COMPLETE_BINARY_ADMISSION_REQUEST",
            "request_version": 1,
            "model": {
                "key": spec.key,
                "id": spec.model_id,
                "repository": spec.repository,
                "revision": spec.revision,
                "native_core_model": spec.key,
            },
            "immutable_source_identity": immutable_identity,
            "current_source_revalidation": current_revalidation,
            "complete_manifest": complete_manifest,
            "native_admission": {
                "required_api": "hawking_core::model::qwen_complete_binary::admit_complete_binary_artifact",
                "rechecks_complete_catalog_payload_hash_layout_and_current_source_identity": True,
            },
            "claim_boundary": {
                "manifest_is_bound_by_exact_seal_and_raw_document_sha256": True,
                "source_content_identity_and_current_full_shard_revalidation_are_both_required": True,
                "raw_bf16_source_is_authority_teacher_not_tournament_participant": True,
                "not_native_decoder_runtime_capability_hcli_tps_tg_or_tournament_qualification": True,
            },
        },
    )
    request_path = (
        paths["complete_root"]
        / "complete-admission"
        / "requests"
        / f"{spec.prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_REQUEST_{manifest['seal_sha256']}.json"
    )
    native_loader_path = root / "native-loader" / spec.key
    native_loader_path.parent.mkdir(parents=True, exist_ok=True)
    native_loader_path.write_bytes(b"test native loader")
    artifact = _write_sealed(
        paths["artifact_admission"],
        {
            "schema": gatekeeper.ARTIFACT_ADMISSION_SCHEMA,
            "status": gatekeeper.ARTIFACT_ADMISSION_STATUS,
            "model": {
                "key": spec.key,
                "id": spec.model_id,
                "repository": spec.repository,
                "revision": spec.revision,
            },
            "admission_request_path": str(request_path),
            "admission_request_seal_sha256": request["seal_sha256"],
            "immutable_source_identity": immutable_identity,
            "current_source_revalidation": current_revalidation,
            "complete_manifest": complete_manifest,
            "native_loader": {
                "api": "hawking_core::model::qwen_complete_binary::admit_complete_binary_artifact",
                "executable_path": str(native_loader_path),
                "executable_sha256": hashlib.sha256(native_loader_path.read_bytes()).hexdigest(),
                "result_schema": "hawking.ascension.qwen_complete_binary_native_admission_result.v1",
                "result_status": "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED",
                "tensor_count": 1,
                "source_weight_elements": source_weight_elements,
                "tensor_payload_bytes": payload_bytes,
            },
            "claim_boundary": {
                "native_complete_catalog_payload_hash_layout_and_source_chain_admission_passed": True,
                "admission_does_not_implement_or_claim_a_native_qwen_decoder": True,
                "admission_does_not_claim_capability_hcli_tps_tg_or_tournament_qualification": True,
                "raw_bf16_source_remains_authority_teacher_only": True,
            },
        },
    )
    return artifact


def _write_qualified_receipts(root: Path, spec: gatekeeper.ModelSpec, identity: dict[str, Any], revalidation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    paths = gatekeeper._paths(root, spec)
    artifact = _write_native_admitted_artifact(root, spec, identity, revalidation)
    runtime = _write_sealed(
        paths["runtime"],
        {
            "schema": gatekeeper.RUNTIME_SCHEMA,
            "status": "PASS_EXACT_NATIVE_FULL_TOKEN_RUNTIME",
            "binding": _binding(
                identity,
                revalidation,
                spec,
                complete_artifact_admission_seal_sha256=artifact["seal_sha256"],
                runtime_executable_sha256="7" * 64,
            ),
            "runtime": {
                "native_exact_decoder": True,
                "full_token_execution": True,
                "all_layers_executed": True,
                "all_weight_tensors_bound": True,
                "tokenizer_bound": True,
                "prompt_template_bound": True,
                "model_alone": True,
                "no_fallback": True,
                "raw_bf16_teacher_not_runtime_participant": True,
                "measured_token_count": 42,
                "timing_scope": "complete_model_token_loop",
            },
        },
    )
    hcli = _write_sealed(
        paths["hcli"],
        {
            "schema": gatekeeper.HCLI_SCHEMA,
            "status": "PASS_MEASURED_HCLI",
            "binding": _binding(
                identity,
                revalidation,
                spec,
                complete_artifact_admission_seal_sha256=artifact["seal_sha256"],
                runtime_receipt_seal_sha256=runtime["seal_sha256"],
            ),
            "measurement": {
                "prompt_dependent_generation": True,
                "uses_exact_native_runtime": True,
                "model_alone": True,
                "no_fallback": True,
                "measured_request_count": 2,
                "completed_generated_tokens": 42,
            },
        },
    )
    kernel = _write_sealed(
        paths["kernel"],
        {
            "schema": gatekeeper.KERNEL_SCHEMA,
            "status": "PASS_CUSTOM_KERNEL_FULL_MODEL_OPERATIONAL",
            "binding": _binding(
                identity,
                revalidation,
                spec,
                complete_artifact_admission_seal_sha256=artifact["seal_sha256"],
                runtime_receipt_seal_sha256=runtime["seal_sha256"],
            ),
            "measurement": {
                "custom_kernel_used": True,
                "full_token_execution": True,
                "model_alone": True,
                "no_fallback": True,
                "measured_token_count": 42,
                "timing_scope": "complete_model_token_loop",
                "base_true_tokens_per_second": 101.0,
            },
        },
    )
    tg3 = _write_sealed(
        paths["tg3"],
        {
            "schema": gatekeeper.TG3_SCHEMA,
            "status": "PASS_TG3_FULL_MODEL_QUALIFICATION",
            "binding": _binding(
                identity,
                revalidation,
                spec,
                complete_artifact_admission_seal_sha256=artifact["seal_sha256"],
                runtime_receipt_seal_sha256=runtime["seal_sha256"],
                hcli_receipt_seal_sha256=hcli["seal_sha256"],
                kernel_receipt_seal_sha256=kernel["seal_sha256"],
            ),
            "measurement": {
                "tg3_completed": True,
                "full_token_execution": True,
                "model_alone": True,
                "no_fallback": True,
                "prompt_dependent_hcli_generation": True,
                "measured_token_count": 42,
                "timing_scope": "complete_model_token_loop",
                "base_true_tokens_per_second": 333.0,
            },
        },
    )
    capability = _write_sealed(
        paths["capability"],
        {
            "schema": gatekeeper.CAPABILITY_SCHEMA,
            "status": "PASS_CAPABILITY_EVALUATION",
            "binding": _binding(
                identity,
                revalidation,
                spec,
                complete_artifact_admission_seal_sha256=artifact["seal_sha256"],
                runtime_receipt_seal_sha256=runtime["seal_sha256"],
                hcli_receipt_seal_sha256=hcli["seal_sha256"],
                tg3_receipt_seal_sha256=tg3["seal_sha256"],
            ),
            "evaluation": {
                "complete_model_evaluation": True,
                "prompt_dependent_generation": True,
                "no_fallback": True,
                "frozen_hidden_task_catalog": True,
                "hidden_task_catalog_sha256": "b" * 64,
                "attempted_task_count": 4,
                "verified_passed_task_count": 3,
            },
        },
    )
    return {"artifact": artifact, "runtime": runtime, "hcli": hcli, "kernel": kernel, "tg3": tg3, "capability": capability}


def _write_tg10_operational_receipt(
    root: Path,
    spec: gatekeeper.ModelSpec,
    identity: dict[str, Any],
    revalidation: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
    *,
    malformed: bool = False,
) -> dict[str, Any]:
    """Write only the independently-bound operational TG10 receipt."""

    paths = gatekeeper._paths(root, spec)
    payload: dict[str, Any] = {
        "schema": gatekeeper.TG10_SCHEMA,
        "status": gatekeeper.TG10_STATUS,
        "binding": _binding(
            identity,
            revalidation,
            spec,
            complete_artifact_admission_seal_sha256=receipts["artifact"]["seal_sha256"],
            runtime_receipt_seal_sha256=receipts["runtime"]["seal_sha256"],
            hcli_receipt_seal_sha256=receipts["hcli"]["seal_sha256"],
            kernel_receipt_seal_sha256=receipts["kernel"]["seal_sha256"],
        ),
        "rung": "TG10",
        "required_threshold_base_true_tps": 100.0,
        "complete_bpw": 1.0,
        "complete_native_model": True,
        "real_metal": True,
        "autoregressive_generation": True,
        "hcli_pass": True,
        "fallback_count": 0,
        "median_base_true_tps": 101.0,
        "sustained_base_true_tps": 100.5,
        "measurement": {
            "full_token_execution": True,
            "model_alone": True,
            "no_fallback": True,
            "prompt_dependent_hcli_generation": True,
            "tg3_completed": False,
            "measured_token_count": 42,
            "timing_scope": "complete_model_token_loop",
            "base_true_tokens_per_second": 101.0,
        },
        "claim_boundary": {
            "only_sealed_after_actual_hcli_complete_token_measurement": True,
            "component_prefill_roofline_and_speculative_rates_are_rejected": True,
        },
    }
    if malformed:
        payload["measurement"] = {
            **payload["measurement"],
            "tg3_completed": True,
        }
    return _write_sealed(paths["tg10"], payload)


def _write_runtime_supersession(
    root: Path, spec: gatekeeper.ModelSpec, runtime: dict[str, Any]
) -> dict[str, Any]:
    """Publish the v1 runtime revocation while preserving the old PASS bytes."""

    paths = gatekeeper._paths(root, spec)
    historical_raw = paths["runtime"].read_bytes()
    archive = (
        paths["runtime"].parent
        / "runtime-receipt-history"
        / f"{spec.prefix}_EXACT_FULL_TOKEN_RUNTIME_RECEIPT_{runtime['seal_sha256']}.json"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(historical_raw)
    executable = runtime["binding"]["runtime_executable_sha256"]
    archive_sha = hashlib.sha256(historical_raw).hexdigest()
    return _write_sealed(
        paths["runtime_supersession"],
        {
            "schema": gatekeeper.RUNTIME_SUPERSESSION_SCHEMA,
            "status": "REVOKED_TEST_NATIVE_RUNTIME_DEFECT",
            "recorded_at": "2026-08-08T00:00:00Z",
            "binding": {
                "model_id": spec.model_id,
                "canonical_runtime_receipt_path": str(paths["runtime"]),
                "superseded_runtime_receipt_seal_sha256": runtime["seal_sha256"],
                "defective_runtime_executable_sha256": executable,
                "archived_runtime_receipt_path": str(archive),
                "archived_runtime_receipt_document_sha256": archive_sha,
            },
            "revoked_runtime": {
                "canonical_receipt_path": str(paths["runtime"]),
                "canonical_receipt_seal_sha256": runtime["seal_sha256"],
                "complete_manifest_seal_sha256": "8" * 64,
                "model_id": spec.model_id,
                "runtime_executable_sha256": executable,
            },
            "historical_pass_archive_path": str(archive),
            "historical_pass_archive_sha256": archive_sha,
            "defect": {"class": "TEST_NATIVE_RUNTIME_DEFECT"},
            "invalidates": {
                "canonical_native_runtime_pass": True,
                "all_old_full_token_prompt_and_profile_controls_bound_to_runtime_sha": True,
                "native_http_adapter_and_transport_handoff_bound_to_runtime_sha": True,
                "any_hcli_tps_tg_capability_or_tournament_consumer_of_that_sha": True,
            },
            "required_before_reissue": ["new executable", "fresh full-token evidence"],
            "consumer_contract": {
                "fail_closed_if_canonical_status_is_not_pass": True,
                "fail_closed_if_this_supersession_revokes_the_bound_receipt_seal_or_runtime_executable_sha256": True,
                "historical_archive_is_for_negative_science_only_not_a_gate_authority": True,
            },
            "claim_boundary": {"revocation_is_not_a_new_runtime_pass": True},
        },
    )


def _write_final_review_marker(root: Path, source: dict[str, tuple[dict[str, Any], dict[str, Any]]], receipts: dict[str, dict[str, dict[str, Any]]]) -> None:
    reviews: list[dict[str, Any]] = []
    for spec in gatekeeper.MODEL_SPECS:
        identity, revalidation = source[spec.key]
        received = receipts[spec.key]
        reviews.append(
            {
                "model_id": spec.model_id,
                "review_disposition": "PRESENT_TO_FINAL_REVIEW",
                "source_content_identity_sha256": identity["content_identity_sha256"],
                "source_revalidation_seal_sha256": revalidation["seal_sha256"],
                "complete_artifact_admission_seal_sha256": received["artifact"]["seal_sha256"],
                "runtime_receipt_seal_sha256": received["runtime"]["seal_sha256"],
                "hcli_receipt_seal_sha256": received["hcli"]["seal_sha256"],
                "kernel_receipt_seal_sha256": received["kernel"]["seal_sha256"],
                "tg3_receipt_seal_sha256": received["tg3"]["seal_sha256"],
                "capability_evaluation_receipt_seal_sha256": received["capability"]["seal_sha256"],
            }
        )
    _write_sealed(
        root / "lifecycle" / gatekeeper.FINAL_REVIEW_FILENAME,
        {
            "schema": gatekeeper.FINAL_REVIEW_SCHEMA,
            "status": "PROTECTED_FINAL_REVIEW_MARKER_PRESENT",
            "authority": "human_operator",
            "does_not_choose_winner": True,
            "fixed_candidate_order": [spec.model_id for spec in gatekeeper.MODEL_SPECS],
            "reviews": reviews,
        },
    )


def _write_final_manager_operations(
    root: Path,
    spec: gatekeeper.ModelSpec,
    identity: dict[str, Any],
    revalidation: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
    suite: dict[str, Any],
) -> dict[str, Any]:
    paths = gatekeeper._paths(root, spec)
    session_measurements = [
        {
            "logical_sessions": sessions,
            "raw_model_tps": 333.0,
            "hcli_tps": 320.0,
            "per_session_p99_ms": 10.0 * sessions,
            "verified_tasks_per_hour": 25.0 * sessions,
            "kv_state_bytes": 1024 * sessions,
            "context_compile_latency_ms": 2.0 * sessions,
            "tool_wait_ms": 1.0,
            "queue_wait_ms": 1.0,
            "weight_reuse_observed": True,
            "starvation_free": True,
        }
        for sessions in (1, 2, 4, 8)
    ]
    return _write_sealed(
        paths["manager_operations"],
        {
            "schema": gatekeeper.MANAGER_OPERATIONS_SCHEMA,
            "status": gatekeeper.MANAGER_OPERATIONS_STATUS,
            "binding": _binding(
                identity,
                revalidation,
                spec,
                complete_artifact_admission_seal_sha256=receipts["artifact"]["seal_sha256"],
                runtime_receipt_seal_sha256=receipts["runtime"]["seal_sha256"],
                hcli_receipt_seal_sha256=receipts["hcli"]["seal_sha256"],
                kernel_receipt_seal_sha256=receipts["kernel"]["seal_sha256"],
                tg3_receipt_seal_sha256=receipts["tg3"]["seal_sha256"],
                capability_evaluation_receipt_seal_sha256=receipts["capability"]["seal_sha256"],
                tournament_suite_preflight_seal_sha256=suite["seal_sha256"],
            ),
            "operations": {
                "uses_exact_native_runtime": True,
                "uses_admitted_gravity_artifact_only": True,
                "agent_os_live": True,
                "context_kv_passed": True,
                "restart_passed": True,
                "residency_fit_passed": True,
                "rollback_passed": True,
                "storage_rollback_passed": True,
                "tool_recovery_passed": True,
                "long_unattended_task_passed": True,
                "single_model_body_shared_across_sessions": True,
                "no_fallback": True,
                "fallback_count": 0,
                "tool_environment_sha256": suite["tool_environment_sha256"],
                "hcli_endpoint": {
                    "protocol": "openai_chat_completions_v1",
                    "host": "127.0.0.1",
                    "port": 18030 if spec.key == "qwen30" else 18080,
                    "health_path": "/healthz",
                    "chat_path": "/v1/chat/completions",
                    "model": spec.gravity_artifact_id,
                    "gravity_artifact_id": spec.gravity_artifact_id,
                },
                "session_measurements": session_measurements,
            },
        },
    )


def _model_row(gate: dict[str, Any], key: str) -> dict[str, Any]:
    return next(row for row in gate["models"] if row["key"] == key)


def test_gatekeeper_fails_closed_and_does_not_promote_heartbeat_only_activity(tmp_path: Path) -> None:
    root = tmp_path / "physical"
    for spec in gatekeeper.MODEL_SPECS:
        _identity_and_revalidation(root, spec)
        _write_observations(root, spec)

    first, workflow = gatekeeper.run_once(root)
    assert first["status"] == "ARMED_WAITING_FOR_QUALIFICATIONS"
    assert first["tournament_execution"]["winner_selection"] == "DISABLED"
    q30 = _model_row(first, "qwen30")
    assert q30["requirements"]["verified_raw_source_identity"]["state"] == "PASS"
    assert q30["requirements"]["current_source_revalidation"]["state"] == "PASS"
    assert q30["requirements"]["complete_admitted_artifact_at_most_1_5_bpw"]["state"] == "BLOCKED"
    assert first["liveness"]["qwen30"]["activity"] == "BASELINE_RECORDED_AWAITING_MATERIAL_DELTA"
    assert verify(workflow)["authority"]["winner"] is None

    # A rising heartbeat with the same durable work facts is explicitly not active progress.
    _write_observations(root, gatekeeper.MODEL_SPECS[0], heartbeat=2, candidates=1)
    second, _ = gatekeeper.run_once(root)
    assert second["liveness"]["qwen30"]["activity"] == "HEARTBEAT_ADVANCED_WITHOUT_MATERIAL_PROGRESS"

    _write_observations(root, gatekeeper.MODEL_SPECS[0], heartbeat=3, candidates=2)
    third, _ = gatekeeper.run_once(root)
    assert third["liveness"]["qwen30"]["activity"] == "ACTIVE_WITH_MATERIAL_PROGRESS"
    assert third["liveness"]["qwen30"]["heartbeat_is_not_material_progress"] is True


def test_unsealed_worker_status_is_operational_observation_not_qualification(tmp_path: Path) -> None:
    root = tmp_path / "physical"
    for spec in gatekeeper.MODEL_SPECS:
        _identity_and_revalidation(root, spec)
        _write_observations(root, spec)
    spec = gatekeeper.MODEL_SPECS[0]
    worker_path = gatekeeper._paths(root, spec)["worker"]
    raw_worker = verify(json.loads(worker_path.read_text(encoding="utf-8")))
    raw_worker.pop("seal_sha256")
    _write_json(worker_path, raw_worker)

    first, _ = gatekeeper.run_once(root)
    assert first["liveness"]["qwen30"]["worker_status"]["trust"] == "OBSERVATIONAL_UNSEALED_STATUS"
    assert first["liveness"]["qwen30"]["activity"] == "BASELINE_RECORDED_AWAITING_MATERIAL_DELTA"
    assert _model_row(first, "qwen30")["pre_final_review_qualification"] == "BLOCKED"

    raw_worker["heartbeat"] = 2
    raw_worker["population"]["completed_candidate_count"] = 2
    raw_worker["current_experiment"]["sequence"] = 2
    raw_worker["current_experiment"]["candidate_id"] = "qwen30-current-2"
    raw_worker["next_experiment"]["sequence"] = 3
    raw_worker["next_experiment"]["candidate_id"] = "qwen30-next-2"
    raw_worker["last_material_progress_at"] = "2026-08-08T00:00:02Z"
    _write_json(worker_path, raw_worker)
    second, _ = gatekeeper.run_once(root)
    assert second["liveness"]["qwen30"]["activity"] == "ACTIVE_WITH_MATERIAL_PROGRESS"
    assert second["liveness"]["qwen30"]["worker_status"]["sealed"] is False


def test_native_complete_binary_admission_passes_only_the_storage_artifact_subgate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "physical"
    source: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for spec in gatekeeper.MODEL_SPECS:
        source[spec.key] = _identity_and_revalidation(root, spec)
        _write_observations(root, spec)

    q30_identity, q30_revalidation = source["qwen30"]
    _write_native_admitted_artifact(root, gatekeeper.MODEL_SPECS[0], q30_identity, q30_revalidation)
    result, _ = gatekeeper.run_once(root)
    q30 = _model_row(result, "qwen30")

    artifact = q30["requirements"]["complete_admitted_artifact_at_most_1_5_bpw"]
    assert artifact["state"] == "PASS"
    assert artifact["details"]["physical_bpw"] == 1.0
    assert artifact["details"]["not_a_runtime_capability_or_manager_qualification"] is True
    assert q30["requirements"]["native_exact_full_token_runtime"]["state"] == "BLOCKED"
    assert q30["requirements"]["measured_hcli"]["state"] == "BLOCKED"
    assert q30["requirements"]["custom_kernel_operational_at_least_100_tps"]["state"] == "BLOCKED"
    assert q30["requirements"]["tg3_at_least_333_tps"]["state"] == "BLOCKED"
    assert q30["requirements"]["capability_and_evaluation_receipt"]["state"] == "BLOCKED"
    assert q30["pre_final_review_qualification"] == "BLOCKED"
    assert result["status"] == "ARMED_WAITING_FOR_QUALIFICATIONS"
    contract = q30["future_receipt_contract"]["complete_artifact_admission"]
    assert contract["schema"] == gatekeeper.ARTIFACT_ADMISSION_SCHEMA
    assert contract["status"] == gatekeeper.ARTIFACT_ADMISSION_STATUS


def test_gatekeeper_current_admission_pointer_selects_manifest_matching_versioned_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "physical"
    source: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for spec in gatekeeper.MODEL_SPECS:
        source[spec.key] = _identity_and_revalidation(root, spec)
        _write_observations(root, spec)

    spec = gatekeeper.MODEL_SPECS[0]
    identity, revalidation = source[spec.key]
    _write_native_admitted_artifact(root, spec, identity, revalidation)
    paths = gatekeeper._paths(root, spec)
    historical_raw = paths["artifact_admission"].read_bytes()
    historical = verify(json.loads(historical_raw))
    manifest = historical["complete_manifest"]
    versioned = (
        paths["complete_root"]
        / "complete-admission"
        / "receipts"
        / f"{spec.prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT_{manifest['seal_sha256']}.json"
    )
    versioned.parent.mkdir(parents=True, exist_ok=True)
    versioned.write_bytes(historical_raw)
    pointer = _write_sealed(
        paths["artifact_admission_current"],
        {
            "schema": gatekeeper.ARTIFACT_ADMISSION_CURRENT_POINTER_SCHEMA,
            "status": gatekeeper.ARTIFACT_ADMISSION_CURRENT_POINTER_STATUS,
            "pointer_version": 1,
            "model": historical["model"],
            "complete_manifest": manifest,
            "admission_request_path": historical["admission_request_path"],
            "admission_request_seal_sha256": historical["admission_request_seal_sha256"],
            "admission_receipt": {
                "path": str(versioned),
                "document_sha256": hashlib.sha256(historical_raw).hexdigest(),
                "seal_sha256": historical["seal_sha256"],
                "selection_source": "VERSIONED_CURRENT_MANIFEST",
            },
            "claim_boundary": {
                "pointer_selects_only_a_receipt_matching_the_current_complete_manifest": True,
                "historical_receipts_are_preserved_not_overwritten_or_resealed": True,
                "pointer_is_storage_artifact_admission_only_not_runtime_or_qualification": True,
            },
        },
    )
    assert pointer["seal_sha256"]
    # A legacy fixed-path artifact is deliberately unusable now. The selected
    # versioned receipt still admits the same current manifest, proving the
    # gate follows the pointer rather than silently falling back to history.
    _write_json(paths["artifact_admission"], {"historical": "not-current"})

    result, _ = gatekeeper.run_once(root)
    artifact = _model_row(result, spec.key)["requirements"][
        "complete_admitted_artifact_at_most_1_5_bpw"
    ]
    assert artifact["state"] == "PASS"
    assert artifact["details"]["admission_selection"] == "CURRENT_POINTER"
    assert artifact["details"]["selected_admission_receipt_path"] == str(versioned)


def test_native_complete_binary_admission_blocks_a_manifest_above_one_point_five_bpw(
    tmp_path: Path,
) -> None:
    root = tmp_path / "physical"
    for spec in gatekeeper.MODEL_SPECS:
        identity, revalidation = _identity_and_revalidation(root, spec)
        _write_observations(root, spec)
        if spec.key == "qwen30":
            _write_native_admitted_artifact(root, spec, identity, revalidation, payload_bytes=16)

    result, _ = gatekeeper.run_once(root)
    artifact = _model_row(result, "qwen30")["requirements"][
        "complete_admitted_artifact_at_most_1_5_bpw"
    ]
    assert artifact["state"] == "BLOCKED"
    assert any("exceeds the 1.5 BPW gate" in reason for reason in artifact["reasons"])
    assert result["status"] == "ARMED_WAITING_FOR_QUALIFICATIONS"


def test_runtime_supersession_revokes_the_old_pass_and_every_downstream_gate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "physical"
    source: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    receipts: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in gatekeeper.MODEL_SPECS:
        identity, revalidation = _identity_and_revalidation(root, spec)
        _write_observations(root, spec)
        source[spec.key] = (identity, revalidation)
        receipts[spec.key] = _write_qualified_receipts(root, spec, identity, revalidation)

    spec = gatekeeper.MODEL_SPECS[0]
    supersession = _write_runtime_supersession(root, spec, receipts[spec.key]["runtime"])
    archived_path = Path(supersession["historical_pass_archive_path"])
    assert verify(json.loads(archived_path.read_text(encoding="utf-8")))["seal_sha256"] == receipts[spec.key]["runtime"]["seal_sha256"]

    result, _ = gatekeeper.run_once(root)
    q30 = _model_row(result, spec.key)
    runtime = q30["requirements"]["native_exact_full_token_runtime"]
    assert runtime["state"] == "BLOCKED"
    assert runtime["details"]["runtime_supersession"]["state"] == "CURRENT_RUNTIME_REVOKED"
    assert any("runtime supersession state is CURRENT_RUNTIME_REVOKED" in reason for reason in runtime["reasons"])
    assert q30["requirements"]["measured_hcli"]["state"] == "BLOCKED"
    assert q30["requirements"]["custom_kernel_operational_at_least_100_tps"]["state"] == "BLOCKED"
    assert q30["requirements"]["tg3_at_least_333_tps"]["state"] == "BLOCKED"
    assert q30["requirements"]["capability_and_evaluation_receipt"]["state"] == "BLOCKED"
    assert q30["requirements"]["final_manager_operations_agent_os_session_restart_residency_rollback_storage"]["state"] == "BLOCKED"

    # A corrected runtime can become current only with a new executable digest;
    # stale HCLI/TG records still bind the revoked seal and remain blocked.
    paths = gatekeeper._paths(root, spec)
    corrected = verify(json.loads(paths["runtime"].read_text(encoding="utf-8")))
    corrected.pop("seal_sha256")
    corrected["binding"] = {
        **corrected["binding"],
        "runtime_executable_sha256": "9" * 64,
    }
    _write_sealed(paths["runtime"], corrected)
    after_correction, _ = gatekeeper.run_once(root)
    q30_after = _model_row(after_correction, spec.key)
    assert q30_after["requirements"]["native_exact_full_token_runtime"]["state"] == "PASS"
    assert q30_after["requirements"]["measured_hcli"]["state"] == "BLOCKED"
    assert any(
        "runtime_receipt_seal_sha256" in reason
        for reason in q30_after["requirements"]["measured_hcli"]["reasons"]
    )


def test_malformed_runtime_supersession_is_itself_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "physical"
    for spec in gatekeeper.MODEL_SPECS:
        identity, revalidation = _identity_and_revalidation(root, spec)
        _write_observations(root, spec)
        _write_qualified_receipts(root, spec, identity, revalidation)

    paths = gatekeeper._paths(root, gatekeeper.MODEL_SPECS[0])
    _write_json(paths["runtime_supersession"], {"schema": gatekeeper.RUNTIME_SUPERSESSION_SCHEMA})
    result, _ = gatekeeper.run_once(root)
    runtime = _model_row(result, "qwen30")["requirements"]["native_exact_full_token_runtime"]
    assert runtime["state"] == "BLOCKED"
    assert runtime["details"]["runtime_supersession"]["state"] == "SUPERSESSION_INVALID_FAIL_CLOSED"


def test_gatekeeper_requires_final_manager_operations_and_hands_off_exactly_once(
    tmp_path: Path,
) -> None:
    root = tmp_path / "physical"
    sources: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    receipts: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in gatekeeper.MODEL_SPECS:
        identity, revalidation = _identity_and_revalidation(root, spec)
        _write_observations(root, spec)
        sources[spec.key] = (identity, revalidation)
        receipts[spec.key] = _write_qualified_receipts(root, spec, identity, revalidation)

    blocked, _ = gatekeeper.run_once(root)
    assert blocked["status"] == "ARMED_WAITING_FOR_QUALIFICATIONS"
    assert blocked["protected_final_review_marker"]["state"] == "BLOCKED"

    # The frozen suite is a real, current hash of the existing HCLI catalog,
    # hidden-membership commitment, evaluator, and local tool contract.  The
    # old manual marker remains informational only: it cannot be used to skip
    # the final manager operations evidence.
    suite = physical_tournament.freeze_suite_preflight(root)
    for spec in gatekeeper.MODEL_SPECS:
        identity, revalidation = sources[spec.key]
        _write_final_manager_operations(
            root, spec, identity, revalidation, receipts[spec.key], suite
        )

    calls: list[dict[str, Any]] = []

    def fake_request_launch(root_arg: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        document = seal(
            {
                "schema": physical_tournament.LAUNCH_SCHEMA,
                "status": physical_tournament.LAUNCH_REQUESTED,
                "launch_id": "a" * 64,
                "physical_root": str(root_arg),
                "qualification_fingerprint": kwargs["qualification_fingerprint"],
                "winner_selection": "DISABLED",
                "winner": None,
            }
        )
        _write_json(root_arg / "lifecycle" / physical_tournament.LAUNCH_FILENAME, document)
        return document

    qualified, workflow = gatekeeper.run_once(root, request_launch=fake_request_launch)
    assert qualified["status"] == "QUALIFICATIONS_COMPLETE"
    assert qualified["qualifications_complete"] is True
    assert qualified["protected_final_review_marker"]["state"] == "BLOCKED"
    assert qualified["tournament_execution"]["winner_selection"] == "DISABLED"
    assert qualified["tournament_execution"]["winner"] is None
    assert len(calls) == 1
    verified_workflow = verify(workflow)
    assert verified_workflow["runtime_phase"] == "QUALIFICATIONS_COMPLETE"
    assert verified_workflow["authority"]["winner_selection"] == "DISABLED"

    # A gatekeeper restart sees the same sealed launch lineage and does not
    # request a second runner for the same qualification fingerprint.
    second, _ = gatekeeper.run_once(root, request_launch=fake_request_launch)
    assert second["status"] == "QUALIFICATIONS_COMPLETE"
    assert len(calls) == 1

    _write_sealed(
        root / "lifecycle" / physical_tournament.RUNNER_FILENAME,
        {
            "schema": physical_tournament.RUNNER_SCHEMA,
            "status": physical_tournament.RUNNING,
            "launch_id": "a" * 64,
            "qualification_fingerprint": qualified["qualification_fingerprint"],
            "winner_selection": "DISABLED",
            "winner": None,
        },
    )
    running, _ = gatekeeper.run_once(root, request_launch=fake_request_launch)
    assert running["status"] == "MANAGER_TOURNAMENT_RUNNING"
    assert running["tournament_execution"]["winner_selection"] == "DISABLED"
    assert running["tournament_execution"]["winner"] is None
    assert len(calls) == 1


def test_component_timing_cannot_be_substituted_for_the_100_tps_kernel_gate(tmp_path: Path) -> None:
    root = tmp_path / "physical"
    sources: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    receipts: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in gatekeeper.MODEL_SPECS:
        identity, revalidation = _identity_and_revalidation(root, spec)
        _write_observations(root, spec)
        sources[spec.key] = (identity, revalidation)
        receipts[spec.key] = _write_qualified_receipts(root, spec, identity, revalidation)
    _write_final_review_marker(root, sources, receipts)

    spec = gatekeeper.MODEL_SPECS[0]
    paths = gatekeeper._paths(root, spec)
    kernel = verify(json.loads(paths["kernel"].read_text(encoding="utf-8")))
    kernel["measurement"]["timing_scope"] = "component_matvec"
    _write_json(paths["kernel"], seal(kernel))
    result, _ = gatekeeper.run_once(root)
    q30 = _model_row(result, "qwen30")
    kernel_requirement = q30["requirements"]["custom_kernel_operational_at_least_100_tps"]
    assert result["status"] == "ARMED_WAITING_FOR_QUALIFICATIONS"
    assert kernel_requirement["state"] == "BLOCKED"
    assert any("complete_model_token_loop" in reason for reason in kernel_requirement["reasons"])


def test_qwen80_current_complete_manifest_fits_the_narrow_model_specific_envelope() -> None:
    """The real admitted Qwen80 catalog is accepted without widening receipts."""

    spec = next(item for item in gatekeeper.MODEL_SPECS if item.key == "qwen80")
    manifest = (
        gatekeeper.DEFAULT_PHYSICAL_ROOT
        / spec.key
        / "complete-gravity"
        / f"{spec.prefix}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
    )
    if not manifest.is_file():
        pytest.skip("live Qwen80 complete manifest is unavailable in this checkout")
    assert manifest.stat().st_size == gatekeeper.QWEN80_AUDITED_COMPLETE_MANIFEST_BYTES
    assert gatekeeper._complete_manifest_max_bytes(spec) == 78 * 1024 * 1024
    assert manifest.stat().st_size <= gatekeeper._complete_manifest_max_bytes(spec)
    assert gatekeeper._complete_manifest_max_bytes(gatekeeper.MODEL_SPECS[0]) == 64 * 1024 * 1024
    loaded = gatekeeper._load_sealed(
        manifest, max_bytes=gatekeeper._complete_manifest_max_bytes(spec)
    )
    assert loaded.sealed
    assert loaded.seal_sha256 == "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b"


def test_qwen80_complete_manifest_envelope_refuses_oversize_and_malformed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = next(item for item in gatekeeper.MODEL_SPECS if item.key == "qwen80")
    cap = gatekeeper._complete_manifest_max_bytes(spec)
    oversized = tmp_path / "oversized.json"
    oversized.write_text("{}", encoding="utf-8")
    original_identity = gatekeeper._regular_file_identity

    def oversized_identity(path: Path, *, label: str) -> dict[str, int]:
        if path == oversized:
            return {
                "bytes": cap + 1,
                "device": 1,
                "inode": 1,
                "mtime_ns": 1,
                "ctime_ns": 1,
            }
        return original_identity(path, label=label)

    monkeypatch.setattr(gatekeeper, "_regular_file_identity", oversized_identity)
    refused = gatekeeper._load_sealed(oversized, max_bytes=cap)
    assert not refused.sealed
    assert refused.errors == [f"receipt exceeds {cap} byte safety limit"]

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    malformed_loaded = gatekeeper._load_sealed(malformed, max_bytes=cap)
    assert not malformed_loaded.sealed
    assert any("cannot read JSON" in error for error in malformed_loaded.errors)


def test_operational_ascent_waits_when_only_one_valid_tg10_exists(tmp_path: Path) -> None:
    root = tmp_path / "physical"
    sources: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    receipts: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in gatekeeper.MODEL_SPECS:
        identity, revalidation = _identity_and_revalidation(root, spec)
        _write_observations(root, spec)
        sources[spec.key] = (identity, revalidation)
        receipts[spec.key] = _write_qualified_receipts(root, spec, identity, revalidation)
    q30 = gatekeeper.MODEL_SPECS[0]
    _write_tg10_operational_receipt(
        root, q30, *sources[q30.key], receipts[q30.key]
    )

    gate, _ = gatekeeper.run_once(root)
    overlay = verify(
        json.loads((root / "lifecycle" / gatekeeper.OPERATIONAL_ASCENT_FILENAME).read_text())
    )
    assert gate["operational_ascent"]["status"] == gatekeeper.OPERATIONAL_ASCENT_WAITING
    assert overlay["status"] == gatekeeper.OPERATIONAL_ASCENT_WAITING
    assert overlay["both_valid_tg10_receipts"] is False
    assert _model_row(gate, "qwen30")["requirements"][
        "tg10_operational_exact_model_100_tps"
    ]["state"] == "PASS"
    assert _model_row(gate, "qwen80")["requirements"][
        "tg10_operational_exact_model_100_tps"
    ]["state"] == "BLOCKED"


def test_operational_ascent_requires_both_valid_tg10s_but_cannot_launch_tournament(
    tmp_path: Path,
) -> None:
    root = tmp_path / "physical"
    sources: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    receipts: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in gatekeeper.MODEL_SPECS:
        identity, revalidation = _identity_and_revalidation(root, spec)
        _write_observations(root, spec)
        sources[spec.key] = (identity, revalidation)
        receipts[spec.key] = _write_qualified_receipts(root, spec, identity, revalidation)
        _write_tg10_operational_receipt(root, spec, identity, revalidation, receipts[spec.key])

    gate, _ = gatekeeper.run_once(root)
    overlay = verify(
        json.loads((root / "lifecycle" / gatekeeper.OPERATIONAL_ASCENT_FILENAME).read_text())
    )
    assert gate["operational_ascent"]["status"] == gatekeeper.OPERATIONAL_ASCENT_EARNED
    assert overlay["status"] == gatekeeper.OPERATIONAL_ASCENT_EARNED
    assert overlay["both_valid_tg10_receipts"] is True
    assert overlay["protected_tournament"]["launch_requested"] is False
    assert overlay["protected_tournament"]["tg3_remains_required"] is True
    assert gate["qualifications_complete"] is False
    assert gate["tournament_execution"]["status"] == "NOT_LAUNCHED"


def test_operational_ascent_rejects_a_malformed_tg10_receipt(tmp_path: Path) -> None:
    root = tmp_path / "physical"
    sources: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    receipts: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in gatekeeper.MODEL_SPECS:
        identity, revalidation = _identity_and_revalidation(root, spec)
        _write_observations(root, spec)
        sources[spec.key] = (identity, revalidation)
        receipts[spec.key] = _write_qualified_receipts(root, spec, identity, revalidation)
        _write_tg10_operational_receipt(
            root,
            spec,
            identity,
            revalidation,
            receipts[spec.key],
            malformed=spec.key == "qwen80",
        )

    gate, _ = gatekeeper.run_once(root)
    overlay = verify(
        json.loads((root / "lifecycle" / gatekeeper.OPERATIONAL_ASCENT_FILENAME).read_text())
    )
    q80_tg10 = _model_row(gate, "qwen80")["requirements"]["tg10_operational_exact_model_100_tps"]
    assert q80_tg10["state"] == "BLOCKED"
    assert any("tg3_completed must be false" in reason for reason in q80_tg10["reasons"])
    assert overlay["status"] == gatekeeper.OPERATIONAL_ASCENT_WAITING

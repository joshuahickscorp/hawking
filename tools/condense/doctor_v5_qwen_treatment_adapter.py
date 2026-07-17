#!/usr/bin/env python3.12
"""Executable Doctor-v5 treatment adapters for the Qwen2.5 Ultra ladder.

The three campaign branches share one reviewed implementation and the same
checkpointed, shard-at-a-time STRAND worker as the codec control.  Every branch
re-encodes the bound source with a frozen treatment recipe, attests the packed
artifact, measures the complete physical model payload, and (where admitted)
evaluates an ephemeral BF16 reconstruction.  Upstream cells are consumed as
hash-bound comparators; no dependency is silently treated as training truth.

The conditional branch is intentionally an executable negative research
protocol.  It measures a production-decodable sparse residual proxy while
recording that activation-conditioned dispatch is absent.  It therefore cannot
be mistaken for evidence of a conditional runtime that does not exist.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import ModuleType
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ABI_PATH = HERE / "doctor_v5_adapter_abi.py"
CONTROL_ADAPTER_PATH = HERE / "doctor_v5_strand_ladder_adapter.py"
WORKER_PATH = HERE / "doctor_v5_strand_ladder_worker.py"
BASE_HELPER_PATH = HERE / "doctor_v5_pass_b_worker.py"
PARAMETER_MODULE_PATH = HERE / "doctor_v5_parameter_manifest.py"
EVALUATOR_PATH = HERE / "doctor_v5_sharded_eval.py"
PASS_A_INDEX = ROOT / "reports/condense/doctor_v5_scale/index.json"
QUANTIZER = ROOT / "vendor/strand-quant/target/release/quantize-model"
ATTESTOR = ROOT / "vendor/strand-decode-kernel/target/release/attest-strand"
DECODER = ROOT / "vendor/strand-decode-kernel/target/release/archive-to-safetensors"
PARAMETER_MANIFEST_DIR = ROOT / "reports/condense/doctor_v5_pass_b/parameter_manifests"
ULTRA_RESULTS = ROOT / "reports/condense/doctor_v5_ultra/results"
MODEL_FAMILY = "qwen2.5-dense"
BACKEND = "apple-cpu-strand"
ADAPTER_VERSION = "1"
INTERNAL_DIR = "qwen_treatment"
MAX_JSON_BYTES = 64 * 1024 * 1024
SHA_RE = re.compile(r"[0-9a-f]{64}")

OPERATIONS: dict[str, dict[str, Any]] = {
    "doctor_static": {
        "adapter_id": "doctor-v5-static-repair",
        "schema": "hawking.doctor_v5_static_spec.v1",
        "dependencies": ("codec_control",),
    },
    "doctor_conditional": {
        "adapter_id": "doctor-v5-conditional-repair",
        "schema": "hawking.doctor_v5_conditional_spec.v1",
        "dependencies": ("codec_control", "doctor_static"),
    },
    "doctor_full": {
        "adapter_id": "doctor-v5-full-treatment",
        "schema": "hawking.doctor_v5_full_spec.v1",
        "dependencies": ("codec_control", "doctor_static", "doctor_conditional"),
    },
}


class TreatmentError(RuntimeError):
    pass


def _load_module(name: str, path: Path) -> ModuleType:
    if not path.is_file() or path.is_symlink():
        raise TreatmentError(f"required module is missing or symlinked: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TreatmentError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTROL = _load_module("doctor_v5_strand_ladder_control_helper", CONTROL_ADAPTER_PATH)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JSON_BYTES:
            raise TreatmentError(f"invalid JSON file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TreatmentError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TreatmentError(f"JSON root is not an object: {path}")
    return value


def _workspace_path(raw: Any, *, must_exist: bool = True) -> Path:
    return CONTROL._workspace_path(raw, must_exist=must_exist)


def _hash_file(path: Path) -> tuple[str, int]:
    return CONTROL._hash_file(path)


def _artifact(path: Path) -> dict[str, Any]:
    digest, size = _hash_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def _errors(value: Any, context: str) -> None:
    if not isinstance(value, list) or any(not isinstance(row, str) for row in value):
        raise TreatmentError(f"{context} validator returned invalid data")
    if value:
        raise TreatmentError(f"{context} validation failed: " + "; ".join(value))


def _operation(value: str) -> dict[str, Any]:
    if value not in OPERATIONS:
        raise TreatmentError(f"unsupported Qwen treatment operation: {value}")
    return OPERATIONS[value]


def _dependency_paths(cell_id: str) -> dict[str, str]:
    root = ULTRA_RESULTS / cell_id
    return {
        "request_path": str(root / "request.json"),
        "registry_path": str(root / "adapter_registry.json"),
        "checkpoint_path": str(root / "checkpoint.json"),
        "result_path": str(root / "result.json"),
        "execution_receipt_path": str(root / "execution_receipt.json"),
        "packed_gc_receipt_path": str(root / "packed_gc_receipt.json"),
    }


def _normalize_dependency_specs(operation: str,
                                rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    expected = OPERATIONS[operation]["dependencies"]
    if len(rows) != len(expected):
        raise TreatmentError(f"{operation} requires {len(expected)} exact dependencies")
    result: list[dict[str, Any]] = []
    for expected_branch, row in zip(expected, rows):
        if not isinstance(row, dict) or set(row) != {
                "branch", "cell_id", "cell_identity_sha256"}:
            raise TreatmentError("dependency builder row keys are invalid")
        branch, cell_id, identity = (row["branch"], row["cell_id"],
                                     row["cell_identity_sha256"])
        if branch != expected_branch or not isinstance(cell_id, str) or not cell_id \
                or not isinstance(identity, str) or SHA_RE.fullmatch(identity) is None:
            raise TreatmentError("dependency order/identity differs from the campaign DAG")
        result.append({"branch": branch, "cell_id": cell_id,
                       "cell_identity_sha256": identity,
                       **_dependency_paths(cell_id)})
    return result


def _codec(worker: ModuleType, operation: str, rate_id: str) -> tuple[dict[str, Any],
                                                                       dict[str, Any]]:
    geometry = worker._rate_geometry(rate_id)
    recipe = worker.treatment_recipe(operation, rate_id)
    codec = {
        "rate_id": rate_id, "artifact_mode": geometry["artifact_mode"],
        "symbol_bits": geometry["symbol_bits"], "vector_dim": geometry["vector_dim"],
        "block_len": geometry["block_len"], "tensor_scope": "all-2d",
        "quality": True, "rht_cols": True,
        "outlier_channel_pct": recipe["outlier_channel_pct"],
        "outlier_bits": recipe["outlier_bits"], "sdsq_sideinfo": True,
        "c2f_outl": recipe["c2f_outl"], "ragged_v2": True,
        "allow_over_ceiling_control": True,
        "learned_codebook": recipe["learned_codebook"],
        "adaptive_scales": geometry["adaptive_scales"],
    }
    return codec, recipe


def _build_inputs(label: str) -> list[dict[str, Any]]:
    parameter_path = PARAMETER_MANIFEST_DIR / f"{label}.json"
    parameter = _load_json(parameter_path)
    census_path = _workspace_path(parameter.get("census_path"))
    census = _load_json(census_path)
    model_dir = _workspace_path(parameter.get("model_dir"))
    if census.get("status") != "complete" or census.get("label") != label:
        raise TreatmentError("completed matching census is required")
    rows = [
        CONTROL._input_row("adapter_abi", ABI_PATH),
        CONTROL._input_row("adapter_source", Path(__file__)),
        CONTROL._input_row("control_adapter_helper", CONTROL_ADAPTER_PATH),
        CONTROL._input_row("worker", WORKER_PATH),
        CONTROL._input_row("base_helper", BASE_HELPER_PATH),
        CONTROL._input_row("parameter_module", PARAMETER_MODULE_PATH),
        CONTROL._input_row("evaluator", EVALUATOR_PATH),
        CONTROL._input_row("quantizer", QUANTIZER),
        CONTROL._input_row("attestor", ATTESTOR),
        CONTROL._input_row("decoder", DECODER),
        CONTROL._input_row("pass_a_index", PASS_A_INDEX),
        CONTROL._input_row("parameter_manifest", parameter_path),
        CONTROL._input_row("source_census", census_path),
    ]
    for row in census.get("source", {}).get("shards", []):
        ordinal, name = row.get("ordinal"), row.get("name")
        if not isinstance(ordinal, int) or not isinstance(name, str):
            raise TreatmentError("census shard identity is invalid")
        rows.append(CONTROL._input_row(
            f"source_shard:{ordinal:05d}", model_dir / name,
            known_sha=row.get("file_sha256"), known_bytes=row.get("bytes"),
        ))
    metadata_names = set(_load_module(
        "doctor_v5_qwen_treatment_worker_inputs", WORKER_PATH).METADATA_NAMES)
    for row in census.get("source", {}).get("auxiliary_files", []):
        name = row.get("name") if isinstance(row, dict) else None
        if name not in metadata_names:
            continue
        rows.append(CONTROL._input_row(
            f"model_metadata:{name}", model_dir / name,
            known_sha=row.get("sha256"), known_bytes=row.get("bytes"),
        ))
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise TreatmentError("treatment input inventory contains duplicate paths")
    return sorted(rows, key=lambda row: row["role"])


def build_spec(*, operation: str, label: str, rate_id: str, cell_id: str,
               cell_identity_sha256: str, program_spec_sha256: str,
               resource_admission_sha256: str, dependencies: list[dict[str, str]],
               evaluation_mode: str, disk_reserve_bytes: int,
               scratch_budget_bytes: int, threads: int, output_path: Path) -> dict[str, Any]:
    worker = _load_module("doctor_v5_qwen_treatment_worker_builder", WORKER_PATH)
    cfg = _operation(operation)
    if label not in worker.SUPPORTED_LABELS or rate_id not in worker.CANONICAL_RATES:
        raise TreatmentError("builder accepts reviewed Qwen2.5 labels/canonical rates only")
    if not isinstance(cell_id, str) or not cell_id:
        raise TreatmentError("cell_id is invalid")
    for name, value in (("cell_identity_sha256", cell_identity_sha256),
                        ("program_spec_sha256", program_spec_sha256),
                        ("resource_admission_sha256", resource_admission_sha256)):
        if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
            raise TreatmentError(f"{name} is invalid")
    if evaluation_mode == "auto":
        evaluation_mode = "resident" if label in worker.RESIDENT_LABELS else "deferred"
    if evaluation_mode not in {"resident", "deferred"} \
            or label not in worker.RESIDENT_LABELS and evaluation_mode != "deferred":
        raise TreatmentError("evaluation mode violates the dense reconstruction gate")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
           for v in (disk_reserve_bytes, scratch_budget_bytes, threads)) \
            or disk_reserve_bytes < 50_000_000_000 or not 1 <= threads <= 32:
        raise TreatmentError("resource values are invalid")
    codec, recipe = _codec(worker, operation, rate_id)
    target = worker.CANONICAL_RATES[rate_id]
    spec = {
        "schema": cfg["schema"], "label": label,
        "campaign_binding": {
            "cell_id": cell_id, "cell_identity_sha256": cell_identity_sha256,
            "branch": operation, "target_rate_id": rate_id,
            "target_rate_bpw": float(target), "label": label,
        },
        "adapter_id": cfg["adapter_id"], "operation": operation,
        "model_family": MODEL_FAMILY, "backend": BACKEND, "codec": codec,
        "evaluation": {"mode": evaluation_mode,
                       "retain_dense_reconstruction": False},
        "treatment": recipe,
        "dependencies": _normalize_dependency_specs(operation, dependencies),
        "resources": {"disk_reserve_bytes": disk_reserve_bytes,
                      "scratch_budget_bytes": scratch_budget_bytes, "threads": threads},
        "program_spec_sha256": program_spec_sha256,
        "resource_admission_sha256": resource_admission_sha256,
        "source_deletion_permitted": False, "quality_claims_permitted": False,
        "inputs": _build_inputs(label),
    }
    fake = {"model": {"label": label, "family": MODEL_FAMILY},
            "operation": operation, "backend": BACKEND,
            "program_spec_sha256": program_spec_sha256,
            "resource_admission_sha256": resource_admission_sha256}
    _validate_spec(spec, fake, worker)
    abi = _load_module("doctor_v5_qwen_treatment_abi_writer", ABI_PATH)
    output = _workspace_path(str(output_path), must_exist=False)
    abi.atomic_json(output, spec)
    return spec


def _validate_spec(spec: dict[str, Any], request: dict[str, Any], worker: ModuleType) -> None:
    expected = {
        "schema", "label", "campaign_binding", "adapter_id", "operation",
        "model_family", "backend", "codec", "evaluation", "treatment",
        "dependencies", "resources", "program_spec_sha256",
        "resource_admission_sha256", "source_deletion_permitted",
        "quality_claims_permitted", "inputs",
    }
    if set(spec) != expected:
        raise TreatmentError("typed treatment spec keys are invalid")
    operation = spec.get("operation")
    cfg = _operation(operation)
    model = request.get("model")
    label = model.get("label") if isinstance(model, dict) else None
    if spec.get("schema") != cfg["schema"] or spec.get("label") != label \
            or spec.get("adapter_id") != cfg["adapter_id"] \
            or spec.get("model_family") != MODEL_FAMILY or spec.get("backend") != BACKEND:
        raise TreatmentError("typed treatment identity differs from canonical request")
    if request.get("operation") != operation or request.get("backend") != BACKEND \
            or not isinstance(model, dict) or model.get("family") != MODEL_FAMILY:
        raise TreatmentError("canonical request selected the wrong treatment adapter")
    if spec.get("program_spec_sha256") != request.get("program_spec_sha256") \
            or spec.get("resource_admission_sha256") != request.get(
                "resource_admission_sha256"):
        raise TreatmentError("program/resource binding differs from canonical request")
    if spec.get("source_deletion_permitted") is not False \
            or spec.get("quality_claims_permitted") is not False:
        raise TreatmentError("treatment claim boundary is invalid")
    codec = spec.get("codec")
    if not isinstance(codec, dict) or codec.get("rate_id") not in worker.CANONICAL_RATES:
        raise TreatmentError("treatment codec rate is invalid")
    worker._validate_campaign(spec.get("campaign_binding"), label,
                              codec["rate_id"], operation)
    expected_codec, recipe = _codec(worker, operation, codec["rate_id"])
    if codec != expected_codec or spec.get("treatment") != recipe:
        raise TreatmentError("treatment codec/recipe differs from reviewed mapping")
    dependencies = spec.get("dependencies")
    if not isinstance(dependencies, list):
        raise TreatmentError("treatment dependencies are invalid")
    simple = [{key: row.get(key) for key in (
        "branch", "cell_id", "cell_identity_sha256")}
        for row in dependencies if isinstance(row, dict)]
    normalized = _normalize_dependency_specs(operation, simple)
    if normalized != dependencies:
        raise TreatmentError("treatment dependency paths/order are not canonical")


def _verify_dependencies(spec: dict[str, Any], abi: ModuleType) -> dict[str, Any]:
    observed: list[dict[str, Any]] = []
    operation = spec["operation"]
    expected_operations = {
        "codec_control": "condense_control", "doctor_static": "doctor_static",
        "doctor_conditional": "doctor_conditional", "doctor_full": "doctor_full",
    }
    for binding in spec["dependencies"]:
        paths = {name: _workspace_path(binding[name]) for name in (
            "request_path", "registry_path", "checkpoint_path", "result_path",
            "execution_receipt_path")}
        registry = _load_json(paths["registry_path"])
        request = _load_json(paths["request_path"])
        result = _load_json(paths["result_path"])
        checkpoint = _load_json(paths["checkpoint_path"])
        receipt = _load_json(paths["execution_receipt_path"])
        _errors(abi.validate_registry(registry, verify_files=True, base_dir=ROOT),
                f"dependency {binding['branch']} registry")
        _errors(abi.validate_request(request, registry, verify_files=False),
                f"dependency {binding['branch']} request")
        _errors(abi.validate_result(result, request, registry, verify_files=False),
                f"dependency {binding['branch']} result")
        _errors(abi.validate_checkpoint(checkpoint, request, registry, verify_files=False),
                f"dependency {binding['branch']} checkpoint")
        command = abi.resolve_command(request, registry, request_path=paths["request_path"])
        _errors(abi.validate_execution_receipt(
            receipt, request, result, registry, checkpoint=checkpoint,
            command_argv=command), f"dependency {binding['branch']} execution receipt")
        if request.get("operation") != expected_operations[binding["branch"]]:
            raise TreatmentError("dependency operation differs from campaign branch")
        metrics = result.get("metrics", {})
        cell = metrics.get("campaign_cell") if isinstance(metrics, dict) else None
        if not isinstance(cell, dict) or cell.get("cell_id") != binding["cell_id"] \
                or cell.get("cell_identity_sha256") != binding["cell_identity_sha256"] \
                or cell.get("branch") != binding["branch"] \
                or cell.get("model_label") != spec["label"] \
                or cell.get("rate_id") != spec["codec"]["rate_id"]:
            raise TreatmentError("dependency result campaign binding mismatch")
        artifacts = result.get("output_artifacts", [])
        manifest_rows = [row for row in artifacts if isinstance(row, dict)
                         and row.get("role") == "bundle_manifest"]
        if len(manifest_rows) != 1:
            raise TreatmentError("dependency result has no unique bundle manifest")
        manifest_path = _workspace_path(manifest_rows[0].get("path"))
        manifest_sha, manifest_bytes = _hash_file(manifest_path)
        if manifest_sha != manifest_rows[0].get("sha256") \
                or manifest_bytes != manifest_rows[0].get("bytes"):
            raise TreatmentError("dependency bundle manifest live identity mismatch")
        # Packed payloads are comparator evidence, not input weights.  Their exact
        # identities remain transitively bound by the completed result, but the
        # payload files themselves may already have been receipt-GC'd after their
        # own result was reporter-sealed and before their immediate successor was
        # admitted.  Dependency reads therefore require the
        # retained manifest/result, never predecessor payload residency.
        packed_rows = [row for row in artifacts if isinstance(row, dict)
                       and str(row.get("role", "")).startswith("bundle_shard:")
                       and str(row.get("path", "")).endswith(".strand")]
        live_count = 0
        for row in packed_rows:
            if not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str) \
                    or SHA_RE.fullmatch(row["sha256"]) is None \
                    or not isinstance(row.get("bytes"), int) or row["bytes"] <= 0:
                raise TreatmentError("dependency packed comparator identity is invalid")
            packed = _workspace_path(row["path"], must_exist=False)
            if packed.exists():
                if packed.is_symlink() or not packed.is_file() \
                        or packed.stat().st_size != row["bytes"]:
                    raise TreatmentError("dependency packed comparator live size changed")
                live_count += 1
        if live_count not in {0, len(packed_rows)}:
            raise TreatmentError("dependency packed payload is partially present; GC is not atomic")
        gc_receipt: dict[str, Any] | None = None
        if packed_rows and live_count == 0:
            gc_path = _workspace_path(binding["packed_gc_receipt_path"])
            gc_receipt = _validate_gc_receipt(
                gc_path, binding=binding, result=result, packed_rows=packed_rows,
                consumer_spec=spec)
        physical = metrics.get("physical_accounting") if isinstance(metrics, dict) else None
        observed.append({
            "branch": binding["branch"], "cell_id": binding["cell_id"],
            "cell_identity_sha256": binding["cell_identity_sha256"],
            "request_sha256": request["request_sha256"],
            "result_sha256": result["result_sha256"],
            "execution_receipt_sha256": receipt["receipt_sha256"],
            "bundle_manifest": {"path": str(manifest_path), "sha256": manifest_sha,
                                "bytes": manifest_bytes},
            "packed_shard_count": len(packed_rows),
            "packed_payload_residency_required": False,
            "packed_payload_state": "live" if live_count else "receipt_gc",
            "packed_gc_receipt": ({"path": binding["packed_gc_receipt_path"],
                                   "sha256": gc_receipt["receipt_sha256"]}
                                  if gc_receipt is not None else None),
            "physical_accounting": physical,
            "quality_observation": metrics.get("quality_observation"),
        })
    return {
        "schema": "hawking.doctor_v5_treatment_dependency_evidence.v1",
        "campaign_binding": spec["campaign_binding"],
        "operation": operation, "recipe_id": spec["treatment"]["recipe_id"],
        "dependencies": observed,
        "consumption": (
            "hash-bound comparator/control evidence; treatment is independently re-encoded "
            "from the common source census"
        ),
        "dependency_evidence_sha256": _sha_value(observed),
    }


def _validate_gc_receipt(path: Path, *, binding: dict[str, Any],
                         result: dict[str, Any],
                         packed_rows: list[dict[str, Any]],
                         consumer_spec: dict[str, Any]) -> dict[str, Any]:
    receipt = _load_json(path)
    expected = {
        "schema", "cell_id", "cell_identity_sha256", "result_sha256",
        "successor", "reporter_sync", "deleted_artifacts",
        "retained_evidence_roles", "parent_source_deleted", "completed_at",
        "receipt_sha256",
    }
    if set(receipt) != expected \
            or receipt.get("schema") != "hawking.doctor_v5_packed_gc_receipt.v2" \
            or receipt.get("cell_id") != binding["cell_id"] \
            or receipt.get("cell_identity_sha256") != binding["cell_identity_sha256"] \
            or receipt.get("result_sha256") != result.get("result_sha256") \
            or receipt.get("parent_source_deleted") is not False:
        raise TreatmentError("dependency packed GC receipt identity/policy is invalid")
    successor = receipt.get("successor")
    if not isinstance(successor, dict) or set(successor) != {
            "cell_id", "cell_identity_sha256", "runtime_spec_path",
            "runtime_spec_sha256", "runtime_spec_bytes", "program_spec_sha256"} \
            or not all(isinstance(successor.get(key), str) and successor[key]
                       for key in ("cell_id", "cell_identity_sha256",
                                   "runtime_spec_path", "runtime_spec_sha256",
                                   "program_spec_sha256")) \
            or isinstance(successor.get("runtime_spec_bytes"), bool) \
            or not isinstance(successor.get("runtime_spec_bytes"), int) \
            or successor["runtime_spec_bytes"] <= 0 \
            or any(SHA_RE.fullmatch(successor[key]) is None
                   for key in ("cell_identity_sha256", "runtime_spec_sha256",
                               "program_spec_sha256")):
        raise TreatmentError("dependency packed GC successor proof is invalid")
    successor_spec_path = _workspace_path(successor["runtime_spec_path"])
    successor_spec_sha, successor_spec_bytes = _hash_file(successor_spec_path)
    if successor_spec_sha != successor["runtime_spec_sha256"] \
            or successor_spec_bytes != successor["runtime_spec_bytes"]:
        raise TreatmentError("dependency packed GC successor runtime identity changed")
    successor_spec = _load_json(successor_spec_path)
    successor_binding = successor_spec.get("campaign_binding")
    expected_successor_branch = {
        "codec_control": "doctor_static",
        "doctor_static": "doctor_conditional",
        "doctor_conditional": "doctor_full",
    }.get(binding["branch"])
    consumer_binding = consumer_spec.get("campaign_binding")
    if not isinstance(consumer_binding, dict) \
            or not isinstance(successor_binding, dict) \
            or successor_binding.get("cell_id") != successor["cell_id"] \
            or successor_binding.get("cell_identity_sha256") \
            != successor["cell_identity_sha256"] \
            or successor_binding.get("branch") != expected_successor_branch \
            or successor_binding.get("label") != consumer_spec.get("label") \
            or successor_binding.get("target_rate_id") \
            != consumer_binding.get("target_rate_id") \
            or successor_spec.get("program_spec_sha256") \
            != successor["program_spec_sha256"]:
        raise TreatmentError("dependency packed GC successor program binding is invalid")
    reporter = receipt.get("reporter_sync")
    if not isinstance(reporter, dict) or set(reporter) != {
            "path", "sha256", "bytes"}:
        raise TreatmentError("dependency packed GC reporter sync is invalid")
    reporter_path = _workspace_path(reporter.get("path"))
    reporter_sha, reporter_bytes = _hash_file(reporter_path)
    if reporter_sha != reporter.get("sha256") or reporter_bytes != reporter.get("bytes"):
        raise TreatmentError("dependency packed GC reporter sync live identity mismatch")
    deleted = receipt.get("deleted_artifacts")
    canonical_rows = [{"role": row["role"], "path": row["path"],
                       "sha256": row["sha256"], "bytes": row["bytes"]}
                      for row in packed_rows]
    if deleted != canonical_rows:
        raise TreatmentError("dependency packed GC deletion list differs from exact allowlist")
    retained = receipt.get("retained_evidence_roles")
    required_retained = {"bundle_manifest", "worker_request", "worker_checkpoint",
                         "worker_receipt", "outer_result", "outer_execution_receipt"}
    if not isinstance(retained, list) or not required_retained.issubset(set(retained)):
        raise TreatmentError("dependency packed GC receipt omits retained evidence roles")
    if receipt.get("receipt_sha256") != _sha_value({
            key: value for key, value in receipt.items() if key != "receipt_sha256"}):
        raise TreatmentError("dependency packed GC receipt hash mismatch")
    return receipt


def _treatment_outcome(spec: dict[str, Any], internal_receipt: dict[str, Any],
                       dependency_evidence: dict[str, Any]) -> dict[str, Any]:
    physical = internal_receipt.get("bundle", {}).get("physical_accounting", {})
    quality = internal_receipt.get("quality_observation")
    operation = spec["operation"]
    if operation == "doctor_conditional":
        protocol_status = "negative_activation_conditioning_not_implemented"
        conclusion = (
            "Sparse residual proxy executed through the real packed decoder; no "
            "activation-conditioned dispatch exists, so no conditional-quality claim is made."
        )
    else:
        protocol_status = "candidate_measured"
        conclusion = (
            "Treatment candidate encoded and measured; improvement is unresolved unless "
            "the resident evaluation and comparator observations support it."
        )
    return {
        "recipe": spec["treatment"], "protocol_status": protocol_status,
        "conclusion": conclusion,
        "physical_target_met": physical.get("target_met"),
        "quality_observation": quality,
        "comparators": [{"branch": row["branch"],
                         "physical_accounting": row.get("physical_accounting"),
                         "quality_observation": row.get("quality_observation")}
                        for row in dependency_evidence["dependencies"]],
        "quality_claims_permitted": False,
        "activation_conditioned_runtime_claimed": False,
    }


def _execute(request_path: Path) -> dict[str, Any]:
    abi = _load_module("doctor_v5_qwen_treatment_abi", ABI_PATH)
    parameter_abi = _load_module("doctor_v5_qwen_treatment_parameters", PARAMETER_MODULE_PATH)
    worker = _load_module("doctor_v5_qwen_treatment_worker", WORKER_PATH)
    request_path = _workspace_path(str(request_path))
    request = abi.read_json(request_path)
    registry_path = _workspace_path(str(request_path.parent / "adapter_registry.json"))
    registry = abi.read_json(registry_path)
    _errors(abi.validate_registry(registry, verify_files=True, base_dir=ROOT), "registry")
    _errors(abi.validate_request(request, registry, verify_files=True), "request")
    operation = request.get("operation")
    cfg = _operation(operation)
    entries = [row for row in registry.get("entries", []) if isinstance(row, dict)
               and row.get("adapter_id") == cfg["adapter_id"]]
    if len(entries) != 1:
        raise TreatmentError("registry does not contain exactly one treatment adapter")
    entry = entries[0]
    if entry.get("adapter_version") != ADAPTER_VERSION \
            or entry.get("operations") != [operation] \
            or entry.get("model_families") != [MODEL_FAMILY] \
            or entry.get("backends") != [BACKEND]:
        raise TreatmentError("registry entry differs from reviewed treatment scope")
    source_path = _workspace_path(entry["source_path"])
    if source_path != Path(__file__).resolve() \
            or _hash_file(source_path)[0] != entry["source_sha256"]:
        raise TreatmentError("registry source binding differs from this adapter")
    paths = CONTROL._paths(request, request_path)
    pilot_path = _workspace_path(request["pilot_spec"]["path"])
    if _hash_file(pilot_path)[0] != request["pilot_spec"]["sha256"] \
            or request["pilot_spec"]["schema"] != cfg["schema"]:
        raise TreatmentError("typed treatment spec identity/schema mismatch")
    spec = _load_json(pilot_path)
    _validate_spec(spec, request, worker)
    if request["inputs"] != CONTROL._normalized_inputs(spec):
        raise TreatmentError("canonical inputs differ from typed treatment spec")

    parameter_path = _workspace_path(request["parameter_manifest"]["path"])
    if _hash_file(parameter_path)[0] != request["parameter_manifest"]["sha256"]:
        raise TreatmentError("parameter manifest identity mismatch")
    parameter = _load_json(parameter_path)
    _errors(parameter_abi.validate_manifest(parameter, verify_files=True),
            "parameter manifest")
    if parameter.get("census_report_sha256") != request["source_census_sha256"]:
        raise TreatmentError("request census binding differs from parameter authority")

    internal_root = paths["output_dir"] / INTERNAL_DIR
    if internal_root.exists() and (not internal_root.is_dir() or internal_root.is_symlink()):
        raise TreatmentError("internal treatment output is invalid")
    internal_root.mkdir(parents=True, exist_ok=True)
    dependency_evidence = _verify_dependencies(spec, abi)
    dependency_path = internal_root / "dependency_evidence.json"
    CONTROL._write_once_or_equal(abi, dependency_path, dependency_evidence)
    dependency_sha = _hash_file(dependency_path)[0]
    inner_spec = dict(spec)
    inner_spec["doctor_hook"] = {
        "method": worker.TREATMENT_METHOD, "operation": operation,
        "recipe_id": spec["treatment"]["recipe_id"],
        "protocol_class": spec["treatment"]["protocol_class"],
        "dependency_evidence": {"path": str(dependency_path),
                                "sha256": dependency_sha,
                                "count": len(spec["dependencies"])},
        "quality_selection_permitted": False,
    }
    inner = CONTROL._build_internal(request, inner_spec, parameter, internal_root)
    inner_path = internal_root / "request.json"
    CONTROL._write_once_or_equal(abi, inner_path, inner)
    checkpoint = CONTROL._outer_checkpoint(
        abi, request, registry, internal_root, "running")
    abi.atomic_json(paths["checkpoint"], checkpoint)
    try:
        internal_receipt = worker._execute(inner_path, preflight_only=False)
    except (worker.LadderError, worker.BASE.PassBError) as exc:
        checkpoint = CONTROL._outer_checkpoint(
            abi, request, registry, internal_root, "checkpointed-stop")
        abi.atomic_json(paths["checkpoint"], checkpoint)
        raise TreatmentError(f"shared ladder worker refused treatment: {exc}") from exc
    if not isinstance(internal_receipt, dict) \
            or internal_receipt.get("status") != "complete" \
            or internal_receipt.get("schema") != worker.RECEIPT_SCHEMA:
        raise TreatmentError("shared ladder worker returned no complete receipt")

    artifacts = CONTROL._retained_artifacts(internal_root)
    artifacts.append({"role": "dependency_evidence", **_artifact(dependency_path)})
    artifacts.sort(key=lambda row: row["role"])
    lifecycle = CONTROL._lifecycle(internal_root, artifacts)
    lifecycle["treatment_artifact_retained"] = True
    campaign = spec["campaign_binding"]
    metrics = {
        "campaign_cell": {"cell_id": campaign["cell_id"],
                          "cell_identity_sha256": campaign["cell_identity_sha256"],
                          "model_label": campaign["label"],
                          "rate_id": campaign["target_rate_id"],
                          "branch": campaign["branch"]},
        "lifecycle": lifecycle,
        "packed_artifact_inventory": CONTROL._packed_artifact_inventory(
            artifacts, branch=campaign["branch"]),
        "dependency_evidence": dependency_evidence,
        "treatment_outcome": _treatment_outcome(
            spec, internal_receipt, dependency_evidence),
        "parameter_accounting": internal_receipt.get("parameter_accounting"),
        "physical_accounting": internal_receipt.get("bundle", {}).get(
            "physical_accounting"),
        "quality_observation": internal_receipt.get("quality_observation"),
        "baseline_cache": internal_receipt.get("baseline_cache"),
        "claims": internal_receipt.get("claims"),
        "completed_replicates": 1,
        "replicate_scope": "preliminary_scale_mapping_not_dominance",
    }
    result = abi.build_result(
        request=request, registry=registry, status="complete",
        output_artifacts=artifacts, metrics=metrics,
        completed_at=internal_receipt.get("completed_at"),
    )
    checkpoint = CONTROL._outer_checkpoint(
        abi, request, registry, internal_root, "complete")
    command = abi.resolve_command(request, registry, request_path=request_path)
    receipt = abi.build_execution_receipt(
        request=request, result=result, registry=registry, checkpoint=checkpoint,
        command_argv=command,
        resource_observations=internal_receipt.get("resources", {}),
        phase_resume=internal_receipt.get("resume", {}),
        completed_at=internal_receipt.get("completed_at"),
    )
    CONTROL._write_once_or_equal(abi, paths["result"], result)
    abi.atomic_json(paths["checkpoint"], checkpoint)
    CONTROL._write_once_or_equal(abi, paths["execution_receipt"], receipt)
    _errors(abi.validate_result(result, request, registry, verify_files=True), "result")
    _errors(abi.validate_checkpoint(
        checkpoint, request, registry, verify_files=True), "checkpoint")
    _errors(abi.validate_execution_receipt(
        receipt, request, result, registry, checkpoint=checkpoint,
        command_argv=command), "execution receipt")
    return result


def _parse_dependency(raw: str) -> dict[str, str]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "dependency must be BRANCH:CELL_ID:CELL_IDENTITY_SHA256")
    return {"branch": parts[0], "cell_id": parts[1],
            "cell_identity_sha256": parts[2]}


def _selftest() -> None:
    worker = _load_module("doctor_v5_qwen_treatment_worker_selftest", WORKER_PATH)
    for operation, cfg in OPERATIONS.items():
        recipe = worker.treatment_recipe(operation, "0.5")
        assert recipe["activation_conditioned_runtime_claimed"] is False
        assert cfg["adapter_id"].startswith("doctor-v5-")
    conditional = worker.treatment_recipe("doctor_conditional", "0.5")
    assert conditional["protocol_class"] == "executable_negative_conditional_protocol"
    for path in (ABI_PATH, CONTROL_ADAPTER_PATH, WORKER_PATH, BASE_HELPER_PATH,
                 PARAMETER_MODULE_PATH, EVALUATOR_PATH, QUANTIZER, ATTESTOR, DECODER):
        CONTROL._tool(path)
    CONTROL._python()
    print(json.dumps({"status": "ok", "operations": OPERATIONS}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--request", required=True, type=Path)
    build = sub.add_parser("build-spec")
    build.add_argument("--operation", required=True, choices=tuple(OPERATIONS))
    build.add_argument("--label", required=True)
    build.add_argument("--rate-id", required=True)
    build.add_argument("--cell-id", required=True)
    build.add_argument("--cell-identity-sha256", required=True)
    build.add_argument("--program-spec-sha256", required=True)
    build.add_argument("--resource-admission-sha256", required=True)
    build.add_argument("--dependency", action="append", default=[],
                       type=_parse_dependency)
    build.add_argument("--evaluation-mode", choices=("auto", "resident", "deferred"),
                       default="auto")
    build.add_argument("--disk-reserve-bytes", type=int, default=50_000_000_000)
    build.add_argument("--scratch-budget-bytes", type=int, required=True)
    build.add_argument("--threads", type=int, default=8)
    build.add_argument("--output", required=True, type=Path)
    sub.add_parser("selftest")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = _execute(args.request)
            print(json.dumps({"status": result["status"],
                              "result_sha256": result["result_sha256"]}, sort_keys=True))
        elif args.command == "build-spec":
            spec = build_spec(
                operation=args.operation, label=args.label, rate_id=args.rate_id,
                cell_id=args.cell_id,
                cell_identity_sha256=args.cell_identity_sha256,
                program_spec_sha256=args.program_spec_sha256,
                resource_admission_sha256=args.resource_admission_sha256,
                dependencies=args.dependency, evaluation_mode=args.evaluation_mode,
                disk_reserve_bytes=args.disk_reserve_bytes,
                scratch_budget_bytes=args.scratch_budget_bytes, threads=args.threads,
                output_path=args.output,
            )
            print(json.dumps({"status": "written", "schema": spec["schema"],
                              "operation": spec["operation"],
                              "output": str(args.output)}, sort_keys=True))
        else:
            _selftest()
        return 0
    except (TreatmentError, CONTROL.AdapterError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3.12
"""Unbound, fail-closed GPT-OSS 120B parallel work and wiring contracts.

Nothing in this module launches a model, edits the Doctor queue, writes runtime
specs, or changes the adapter registry.  It converts the reviewed MXFP4
inventory into independent source-traversal and per-rate output units, validates
receipts and canonical reassembly, exposes a structural MoE archive lookup, and
builds an exact *pending-only* 10-rate x 4-branch promotion packet.

The source graph deliberately separates one bounded source traversal from ten
rate outputs.  A future worker may keep a small staging unit alive while rate
encoders run concurrently, but every output remains independently hashed and
receipted.  This removes repeated 65 GB source traversal without changing a
rate, branch, tensor, or quality gate.
"""
from __future__ import annotations

import argparse
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import doctor_v5_gptoss_mxfp4 as mxfp4


WORK_PLAN_SCHEMA = "hawking.doctor_v5_gptoss_parallel_work_plan.v1"
SOURCE_TRAVERSAL_RECEIPT_SCHEMA = (
    "hawking.doctor_v5_gptoss_source_traversal_receipt.v1"
)
OUTPUT_RECEIPT_SCHEMA = "hawking.doctor_v5_gptoss_parallel_output_receipt.v1"
MERGE_MANIFEST_SCHEMA = "hawking.doctor_v5_gptoss_parallel_merge_manifest.v1"
TOKENIZER_BINDING_SCHEMA = "hawking.doctor_v5_tokenizer_binding.v1"
PENDING_WIRING_SCHEMA = "hawking.doctor_v5_gptoss_pending_wiring.v1"
DEFAULT_INVENTORY = mxfp4.DEFAULT_INVENTORY
DEFAULT_CAMPAIGN_PLAN = ROOT / "reports/condense/doctor_v5_ultra/campaign_plan.json"
DEFAULT_ADAPTER_REGISTRY = ROOT / "reports/condense/doctor_v5_ultra/adapter_registry.json"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "reports/condense/doctor_v5_unbound/gptoss_120b_parallel"
)
DEFAULT_WORK_PLAN = DEFAULT_OUTPUT_ROOT / "work_plan.json"
DEFAULT_PENDING_WIRING = DEFAULT_OUTPUT_ROOT / "pending_wiring.json"
SHA_RE = re.compile(r"[0-9a-f]{64}")
MAX_JSON_BYTES = 128 * 1024 * 1024
LAYERS = 36
EXPERTS = 128
EXPERTS_PER_BATCH = 8
EXPECTED_SOURCE_UNITS = 615
EXPECTED_OUTPUT_UNITS = 6_150
BRANCHES = (
    ("codec_control", "condense_control", "doctor-v5-strand-ladder-gpt-oss-moe"),
    ("doctor_static", "doctor_static", "doctor-v5-gpt-oss-static-repair"),
    ("doctor_conditional", "doctor_conditional", "doctor-v5-gpt-oss-conditional-repair"),
    ("doctor_full", "doctor_full", "doctor-v5-gpt-oss-full-treatment"),
)
RATES = tuple(rate_id for rate_id, _rate in mxfp4.CANONICAL_RATES)
RATE_FRACTIONS = dict(mxfp4.CANONICAL_RATES)


class ParallelScaffoldError(RuntimeError):
    """An unbound work, receipt, or promotion contract is invalid."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) \
                or info.st_size > MAX_JSON_BYTES:
            raise ParallelScaffoldError(f"invalid JSON file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ParallelScaffoldError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ParallelScaffoldError(f"JSON root is not an object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    digest, size = mxfp4._hash_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def _with_hash(doc: dict[str, Any], field: str) -> dict[str, Any]:
    if field in doc:
        raise ParallelScaffoldError(f"refusing to replace existing hash field: {field}")
    doc[field] = _hash_value(doc)
    return doc


def _without(doc: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in doc.items() if key != field}


def _extent(
    tensor: dict[str, Any], source: dict[str, Any], *,
    start: int | None = None, end: int | None = None,
) -> dict[str, Any]:
    offsets = tensor["absolute_data_offsets"]
    tensor_start, tensor_end = offsets
    begin = tensor_start if start is None else start
    finish = tensor_end if end is None else end
    if begin < tensor_start or finish > tensor_end or finish <= begin:
        raise ParallelScaffoldError(f"invalid tensor subrange: {tensor['name']}")
    return {
        "tensor": tensor["name"], "dtype": tensor["dtype"],
        "shape": tensor["shape"], "source_shard": tensor["shard"],
        "source_shard_path": source["path"],
        "source_shard_sha256": source["file_sha256"],
        "absolute_byte_range": [begin, finish], "bytes": finish - begin,
    }


def _source_unit(
    unit_id: str, kind: str, extents: list[dict[str, Any]], *,
    logical_parameters: int, staging: dict[str, Any],
    coordinates: dict[str, Any],
) -> dict[str, Any]:
    if not extents or logical_parameters <= 0:
        raise ParallelScaffoldError(f"empty source work unit: {unit_id}")
    binding = {
        "unit_id": unit_id, "kind": kind, "coordinates": coordinates,
        "source_extents": extents,
    }
    return {
        **binding, "source_binding_sha256": _hash_value(binding),
        "logical_parameters": logical_parameters,
        "source_bytes": sum(row["bytes"] for row in extents),
        "staging": staging,
        "lifecycle": {
            "source_read_only": True, "source_deletion_permitted": False,
            "staging_worker_owned": True,
            "staging_gc_requires_all_ten_output_receipts": True,
            "staging_gc_requires_hash_before_delete": True,
        },
    }


def _tensor_elements(row: dict[str, Any]) -> int:
    return math.prod(row["shape"])


def _load_inventory(path: Path) -> dict[str, Any]:
    doc = _read_json(path)
    errors = mxfp4.validate_inventory(doc)
    if errors:
        raise ParallelScaffoldError("invalid MXFP4 inventory: " + "; ".join(errors))
    if doc.get("model", {}).get("label") != "120B" \
            or doc.get("parameter_accounting", {}).get(
                "logical_model_parameters"
            ) != 116_829_156_672:
        raise ParallelScaffoldError("inventory is not the reviewed GPT-OSS 120B authority")
    return doc


def build_work_plan(
    inventory_path: Path = DEFAULT_INVENTORY, *, created_at: str | None = None,
) -> dict[str, Any]:
    """Build the exact 615-source/6,150-output graph without reading a shard."""
    inventory_path = Path(inventory_path).resolve(strict=True)
    inventory = _load_inventory(inventory_path)
    tensor_rows = inventory["tensor_inventory"]["tensors"]
    tensors = {row["name"]: row for row in tensor_rows}
    sources = {
        row["name"].split("/")[-1]: row
        for row in inventory["source_binding"]["shards"]
    }
    if len(tensors) != 543 or len(sources) != 7:
        raise ParallelScaffoldError("reviewed tensor/shard cardinality changed")

    units: list[dict[str, Any]] = []
    assigned_full: set[str] = set()
    for layer in range(LAYERS):
        for expert_start in range(0, EXPERTS, EXPERTS_PER_BATCH):
            extents: list[dict[str, Any]] = []
            logical = 0
            for projection in ("mlp1", "mlp2"):
                for suffix in ("blocks", "scales"):
                    name = f"block.{layer}.mlp.{projection}_weight.{suffix}"
                    row = tensors[name]
                    start0, end0 = row["absolute_data_offsets"]
                    stride, remainder = divmod(end0 - start0, EXPERTS)
                    if remainder:
                        raise ParallelScaffoldError(f"non-integral expert extent: {name}")
                    begin = start0 + expert_start * stride
                    end = begin + EXPERTS_PER_BATCH * stride
                    extents.append(_extent(row, sources[row["shard"]], start=begin, end=end))
                    if suffix == "blocks":
                        logical += EXPERTS_PER_BATCH * stride * 2
            units.append(_source_unit(
                f"expert/layer={layer:03d}/experts={expert_start:03d}-{expert_start + 7:03d}",
                "expert_batch", extents, logical_parameters=logical,
                coordinates={"layer": layer, "expert_start": expert_start,
                             "expert_count": EXPERTS_PER_BATCH},
                staging={
                    "format": "safetensors", "dtype": "BF16",
                    "orientation": "out_features,in_features",
                    "maximum_materialized_bytes": logical * 2,
                    "whole_shard_materialized": False,
                },
            ))

        dense_names = [
            f"block.{layer}.attn.out.weight",
            f"block.{layer}.attn.qkv.weight",
            f"block.{layer}.mlp.gate.weight",
        ]
        dense_rows = [tensors[name] for name in dense_names]
        assigned_full.update(dense_names)
        units.append(_source_unit(
            f"dense/layer={layer:03d}", "dense_layer",
            [_extent(row, sources[row["shard"]]) for row in dense_rows],
            logical_parameters=sum(_tensor_elements(row) for row in dense_rows),
            coordinates={"layer": layer},
            staging={
                "format": "safetensors", "dtype": "BF16",
                "orientation": "source_declared", "whole_shard_materialized": False,
                "maximum_materialized_bytes": sum(row["bytes"] for row in dense_rows),
            },
        ))

    for kind, name in (("embedding", "embedding.weight"),
                       ("output_head", "unembedding.weight")):
        row = tensors[name]
        assigned_full.add(name)
        units.append(_source_unit(
            kind, kind, [_extent(row, sources[row["shard"]])],
            logical_parameters=_tensor_elements(row), coordinates={},
            staging={
                "format": "safetensors", "dtype": "BF16",
                "orientation": "source_declared", "whole_shard_materialized": False,
                "stream_copy_chunk_bytes": 8 * 1024 * 1024,
                "maximum_materialized_bytes": 8 * 1024 * 1024,
            },
        ))

    packed = {name for name in tensors if name.endswith((".blocks", ".scales"))}
    sidecar_names = sorted(set(tensors) - packed - assigned_full)
    if len(sidecar_names) != 289 \
            or any(tensors[name]["dtype"] != "BF16" for name in sidecar_names):
        raise ParallelScaffoldError("lossless sidecar tensor partition changed")
    sidecar_rows = [tensors[name] for name in sidecar_names]
    units.append(_source_unit(
        "lossless-sidecar", "lossless_sidecar",
        [_extent(row, sources[row["shard"]]) for row in sidecar_rows],
        logical_parameters=sum(_tensor_elements(row) for row in sidecar_rows),
        coordinates={},
        staging={
            "format": "canonical_lossless_passthrough", "dtype": "BF16",
            "orientation": "source_declared", "whole_shard_materialized": False,
            "stream_copy_chunk_bytes": 8 * 1024 * 1024,
            "maximum_materialized_bytes": 8 * 1024 * 1024,
        },
    ))
    units.sort(key=lambda row: row["unit_id"])

    output_units: list[dict[str, Any]] = []
    for source_unit in units:
        for rate_id in RATES:
            rate = RATE_FRACTIONS[rate_id]
            output_units.append({
                "output_unit_id": f"{source_unit['unit_id']}/rate={rate_id}",
                "source_unit_id": source_unit["unit_id"],
                "source_binding_sha256": source_unit["source_binding_sha256"],
                "rate_id": rate_id,
                "target_fraction": [rate.numerator, rate.denominator],
                "target_bpw": float(rate),
                "status": "pending_unbound_execution",
                "archive_receipt_required": True,
                "attestation_required": source_unit["kind"] != "lossless_sidecar",
                "quality_claims_permitted": False,
            })
    output_units.sort(key=lambda row: row["output_unit_id"])
    model_parameters = sum(row["logical_parameters"] for row in units)
    doc: dict[str, Any] = {
        "schema": WORK_PLAN_SCHEMA, "created_at": created_at or _now(),
        "status": "unbound-scaffold-only",
        "model": {
            "label": "120B", "hf_id": "openai/gpt-oss-120b",
            "family": "gpt-oss-moe", "logical_parameters": model_parameters,
            "layers": LAYERS, "experts": EXPERTS, "experts_per_batch": EXPERTS_PER_BATCH,
        },
        "inventory": {
            **_artifact(inventory_path),
            "inventory_sha256": inventory["inventory_sha256"],
            "tensor_inventory_sha256": inventory["tensor_inventory"]["sha256"],
            "source_manifest_sha256": inventory["source_binding"][
                "source_manifest_sha256"
            ],
        },
        "rates": [
            {"rate_id": rate_id, "target_fraction": [rate.numerator, rate.denominator],
             "target_bpw": float(rate)}
            for rate_id, rate in mxfp4.CANONICAL_RATES
        ],
        "parallelism": {
            "source_units_independent": True,
            "rate_outputs_independent_after_source_staging": True,
            "canonical_output_order": "output_unit_id_lexicographic",
            "source_unit_count": len(units), "output_unit_count": len(output_units),
            "expert_source_units": LAYERS * (EXPERTS // EXPERTS_PER_BATCH),
            "dense_source_units": LAYERS, "embedding_or_head_source_units": 2,
            "lossless_sidecar_source_units": 1,
            "whole_model_materialization": False, "whole_shard_materialization": False,
            "cross_host_distribution_permitted_only_with_identical_plan_sha256": True,
        },
        "lifecycle": {
            "source_files_never_worker_owned": True,
            "source_files_never_deleted": True,
            "staging_deleted_only_after_all_dependent_output_hashes": True,
            "rate_artifact_gc_requires_reporter_seal_and_successor_binding": True,
            "whole_artifact_bytes_count_every_rate_component": True,
        },
        "execution_gate": {
            "executable": False, "reviewed_for_live_campaign": False,
            "requires": [
                "bit_exact_encoder_canary", "reviewed_runtime_adapters",
                "tokenizer_binding", "gptoss_moe_numerical_runtime_parity",
                "measured_ram_admission", "disk_lifecycle_admission",
                "queue_quiescent_promotion_checkpoint",
            ],
            "absence_is_not_negative_evidence": True,
        },
        "source_units": units, "output_units": output_units,
        "quality_claims_permitted": False, "source_deletion_permitted": False,
    }
    _with_hash(doc, "work_plan_sha256")
    errors = validate_work_plan(doc)
    if errors:
        raise ParallelScaffoldError("generated work plan is invalid: " + "; ".join(errors))
    return doc


def validate_work_plan(doc: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict) or doc.get("schema") != WORK_PLAN_SCHEMA:
        return ["work-plan schema mismatch"]
    digest = doc.get("work_plan_sha256")
    if not isinstance(digest, str) or digest != _hash_value(_without(doc, "work_plan_sha256")):
        errors.append("work-plan hash mismatch")
    if doc.get("status") != "unbound-scaffold-only" \
            or doc.get("execution_gate", {}).get("executable") is not False \
            or doc.get("quality_claims_permitted") is not False \
            or doc.get("source_deletion_permitted") is not False:
        errors.append("work-plan fail-closed boundary is invalid")
    model = doc.get("model")
    if not isinstance(model, dict) or model.get("logical_parameters") != 116_829_156_672 \
            or model.get("layers") != LAYERS or model.get("experts") != EXPERTS:
        errors.append("work-plan model authority is invalid")
    rates = doc.get("rates")
    if not isinstance(rates, list) or [row.get("rate_id") for row in rates
                                       if isinstance(row, dict)] != list(RATES):
        errors.append("work-plan rate order differs from the canonical ten rates")
    source_units, output_units = doc.get("source_units"), doc.get("output_units")
    if not isinstance(source_units, list) or len(source_units) != EXPECTED_SOURCE_UNITS:
        errors.append(f"source-unit coverage is not {EXPECTED_SOURCE_UNITS}")
        source_units = []
    if not isinstance(output_units, list) or len(output_units) != EXPECTED_OUTPUT_UNITS:
        errors.append(f"output-unit coverage is not {EXPECTED_OUTPUT_UNITS}")
        output_units = []
    ids: set[str] = set()
    source_by_id: dict[str, dict[str, Any]] = {}
    tensor_segments: dict[str, list[tuple[int, int]]] = {}
    tensor_extents: dict[str, list[dict[str, Any]]] = {}
    kind_counts: dict[str, int] = {}
    logical = 0
    for unit in source_units:
        if not isinstance(unit, dict) or not isinstance(unit.get("unit_id"), str) \
                or unit["unit_id"] in ids:
            errors.append("source unit is invalid or duplicated")
            continue
        ids.add(unit["unit_id"])
        source_by_id[unit["unit_id"]] = unit
        kind = unit.get("kind")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        binding = {key: unit.get(key) for key in (
            "unit_id", "kind", "coordinates", "source_extents"
        )}
        if unit.get("source_binding_sha256") != _hash_value(binding):
            errors.append(f"source binding hash mismatch: {unit['unit_id']}")
        extents = unit.get("source_extents")
        if not isinstance(extents, list) or not extents:
            errors.append(f"source extents missing: {unit['unit_id']}")
            continue
        logical_value = unit.get("logical_parameters")
        if isinstance(logical_value, bool) or not isinstance(logical_value, int) \
                or logical_value <= 0:
            errors.append(f"logical parameter count invalid: {unit['unit_id']}")
        else:
            logical += logical_value
        for extent in extents:
            byte_range = extent.get("absolute_byte_range") if isinstance(extent, dict) else None
            if not isinstance(extent, dict) or not isinstance(extent.get("tensor"), str) \
                    or not isinstance(extent.get("source_shard_path"), str) \
                    or not isinstance(extent.get("source_shard_sha256"), str) \
                    or SHA_RE.fullmatch(extent["source_shard_sha256"]) is None \
                    or not isinstance(byte_range, list) or len(byte_range) != 2 \
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           for value in byte_range) or byte_range[0] < 0 \
                    or byte_range[1] <= byte_range[0] \
                    or extent.get("bytes") != byte_range[1] - byte_range[0]:
                errors.append(f"invalid source extent: {unit['unit_id']}")
                continue
            tensor_segments.setdefault(extent["tensor"], []).append(tuple(byte_range))
            tensor_extents.setdefault(extent["tensor"], []).append(extent)
        expected_logical = 0
        for extent in extents:
            if not isinstance(extent, dict) or not isinstance(extent.get("bytes"), int):
                continue
            if kind == "expert_batch":
                if extent.get("tensor", "").endswith(".blocks"):
                    expected_logical += extent["bytes"] * 2
            elif extent.get("dtype") == "BF16":
                expected_logical += extent["bytes"] // 2
        if isinstance(logical_value, int) and logical_value != expected_logical:
            errors.append(f"logical parameters do not match source bytes: {unit['unit_id']}")
        if unit.get("source_bytes") != sum(
                extent.get("bytes", 0) for extent in extents if isinstance(extent, dict)):
            errors.append(f"source byte total mismatch: {unit['unit_id']}")
    if logical != 116_829_156_672:
        errors.append("source-unit logical parameter coverage does not close")
    if len(tensor_segments) != 543:
        errors.append("source-unit tensor coverage is not exactly 543 tensors")
    expected_kind_counts = {
        "expert_batch": 576, "dense_layer": 36, "embedding": 1,
        "output_head": 1, "lossless_sidecar": 1,
    }
    if kind_counts != expected_kind_counts:
        errors.append("source-unit kind cardinalities differ from 576+36+1+1+1")
    inventory_binding = doc.get("inventory")
    try:
        if not isinstance(inventory_binding, dict):
            raise ParallelScaffoldError("inventory binding is missing")
        inventory_path = Path(inventory_binding["path"])
        observed_artifact = _artifact(inventory_path)
        if any(observed_artifact[field] != inventory_binding.get(field)
               for field in ("path", "sha256", "bytes")):
            raise ParallelScaffoldError("bound inventory file identity differs")
        inventory = _load_inventory(inventory_path)
        if inventory["inventory_sha256"] != inventory_binding.get("inventory_sha256") \
                or inventory["tensor_inventory"]["sha256"] != inventory_binding.get(
                    "tensor_inventory_sha256"
                ) or inventory["source_binding"]["source_manifest_sha256"] \
                != inventory_binding.get("source_manifest_sha256"):
            raise ParallelScaffoldError("bound inventory semantic identity differs")
        expected_tensors = {
            row["name"]: row for row in inventory["tensor_inventory"]["tensors"]
        }
        source_by_name = {
            row["name"].split("/")[-1]: row
            for row in inventory["source_binding"]["shards"]
        }
        if set(tensor_segments) != set(expected_tensors):
            raise ParallelScaffoldError("planned and inventory tensors differ")
        for tensor, expected in expected_tensors.items():
            ordered = sorted(tensor_segments[tensor])
            start, end = expected["absolute_data_offsets"]
            if not ordered or ordered[0][0] != start or ordered[-1][1] != end \
                    or any(left[1] != right[0]
                           for left, right in zip(ordered, ordered[1:])):
                errors.append(f"non-exact source tensor coverage: {tensor}")
            source = source_by_name[expected["shard"]]
            for extent in tensor_extents[tensor]:
                if extent.get("dtype") != expected["dtype"] \
                        or extent.get("shape") != expected["shape"] \
                        or extent.get("source_shard") != expected["shard"] \
                        or extent.get("source_shard_path") != source["path"] \
                        or extent.get("source_shard_sha256") != source["file_sha256"]:
                    errors.append(f"source tensor authority mismatch: {tensor}")
                    break
    except (OSError, KeyError, TypeError, ParallelScaffoldError,
            mxfp4.Mxfp4Error) as exc:
        errors.append(f"work-plan inventory verification failed: {exc}")
    expected_outputs = {
        f"{unit_id}/rate={rate_id}" for unit_id in ids for rate_id in RATES
    }
    observed_outputs: set[str] = set()
    for output in output_units:
        if not isinstance(output, dict) or not isinstance(output.get("output_unit_id"), str) \
                or output["output_unit_id"] in observed_outputs:
            errors.append("output unit is invalid or duplicated")
            continue
        observed_outputs.add(output["output_unit_id"])
        parent = source_by_id.get(output.get("source_unit_id"))
        rate_id = output.get("rate_id")
        if parent is None or rate_id not in RATES \
                or output["output_unit_id"] != f"{parent['unit_id']}/rate={rate_id}" \
                or output.get("source_binding_sha256") != parent["source_binding_sha256"] \
                or output.get("status") != "pending_unbound_execution" \
                or output.get("quality_claims_permitted") is not False:
            errors.append(f"output-unit binding is invalid: {output['output_unit_id']}")
    if observed_outputs != expected_outputs:
        errors.append("output-unit exact cross-product coverage differs")
    parallelism = doc.get("parallelism")
    if not isinstance(parallelism, dict) \
            or parallelism.get("source_unit_count") != EXPECTED_SOURCE_UNITS \
            or parallelism.get("output_unit_count") != EXPECTED_OUTPUT_UNITS:
        errors.append("parallelism cardinality contract is invalid")
    return errors


def build_output_receipt(
    plan: dict[str, Any], *, output_unit_id: str,
    source_traversal_receipt: dict[str, Any], archive: dict[str, Any],
    attestation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one structural output receipt; it makes no quality claim."""
    errors = validate_work_plan(plan)
    if errors:
        raise ParallelScaffoldError("cannot receipt invalid plan: " + "; ".join(errors))
    expected = {row["output_unit_id"]: row for row in plan["output_units"]}.get(output_unit_id)
    if expected is None:
        raise ParallelScaffoldError(f"output is absent from the plan: {output_unit_id}")
    traversal_errors = validate_source_traversal_receipt(plan, source_traversal_receipt)
    if traversal_errors:
        raise ParallelScaffoldError(
            "source traversal receipt is invalid: " + "; ".join(traversal_errors)
        )
    if source_traversal_receipt["source_unit_id"] != expected["source_unit_id"] \
            or source_traversal_receipt["source_binding_sha256"] \
            != expected["source_binding_sha256"]:
        raise ParallelScaffoldError("source traversal belongs to a different output parent")
    for role, artifact in (("archive", archive),):
        if not isinstance(artifact, dict) or not isinstance(artifact.get("sha256"), str) \
                or SHA_RE.fullmatch(artifact["sha256"]) is None \
                or isinstance(artifact.get("bytes"), bool) \
                or not isinstance(artifact.get("bytes"), int) or artifact["bytes"] < 0:
            raise ParallelScaffoldError(f"{role} artifact identity is invalid")
    if expected["attestation_required"]:
        if not isinstance(attestation, dict) \
                or not isinstance(attestation.get("root_sha256"), str) \
                or SHA_RE.fullmatch(attestation["root_sha256"]) is None:
            raise ParallelScaffoldError("required STR2 attestation root is missing")
    elif attestation is not None:
        raise ParallelScaffoldError("lossless sidecar must not fabricate STR2 attestation")
    doc: dict[str, Any] = {
        "schema": OUTPUT_RECEIPT_SCHEMA, "created_at": _now(), "status": "complete",
        "work_plan_sha256": plan["work_plan_sha256"],
        "output_unit_id": output_unit_id,
        "source_unit_id": expected["source_unit_id"],
        "source_binding_sha256": expected["source_binding_sha256"],
        "rate_id": expected["rate_id"],
        "source_traversal_receipt_sha256": source_traversal_receipt[
            "receipt_sha256"
        ],
        "staging_artifact": source_traversal_receipt["staging_artifact"],
        "archive": archive, "attestation": attestation,
        "lifecycle": {
            "source_files_deleted": False,
            "staging_deletion_now_permitted_for_this_output_only": True,
            "parent_staging_gc_still_requires_all_ten_output_receipts": True,
        },
        "claims": {"codec_complete": True, "quality": False, "deployable": False},
    }
    _with_hash(doc, "receipt_sha256")
    receipt_errors = validate_output_receipt(plan, doc)
    if receipt_errors:
        raise ParallelScaffoldError("generated receipt is invalid: " + "; ".join(receipt_errors))
    return doc


def build_source_traversal_receipt(
    plan: dict[str, Any], *, source_unit_id: str,
    staging_artifact: dict[str, Any], range_sha256: list[str],
) -> dict[str, Any]:
    """Bind every staged byte to its exact original shard range."""
    errors = validate_work_plan(plan)
    if errors:
        raise ParallelScaffoldError("cannot receipt invalid plan: " + "; ".join(errors))
    source = {row["unit_id"]: row for row in plan["source_units"]}.get(source_unit_id)
    if source is None:
        raise ParallelScaffoldError(f"source unit is absent from plan: {source_unit_id}")
    if not isinstance(staging_artifact, dict) \
            or not isinstance(staging_artifact.get("sha256"), str) \
            or SHA_RE.fullmatch(staging_artifact["sha256"]) is None \
            or isinstance(staging_artifact.get("bytes"), bool) \
            or not isinstance(staging_artifact.get("bytes"), int) \
            or staging_artifact["bytes"] < 0:
        raise ParallelScaffoldError("staging artifact identity is invalid")
    if not isinstance(range_sha256, list) \
            or len(range_sha256) != len(source["source_extents"]) \
            or any(not isinstance(digest, str) or SHA_RE.fullmatch(digest) is None
                   for digest in range_sha256):
        raise ParallelScaffoldError("source range digest coverage is invalid")
    range_receipts = [
        {
            "tensor": extent["tensor"],
            "source_shard_sha256": extent["source_shard_sha256"],
            "absolute_byte_range": extent["absolute_byte_range"],
            "bytes": extent["bytes"], "range_sha256": digest,
        }
        for extent, digest in zip(source["source_extents"], range_sha256, strict=True)
    ]
    doc: dict[str, Any] = {
        "schema": SOURCE_TRAVERSAL_RECEIPT_SCHEMA, "created_at": _now(),
        "status": "complete", "work_plan_sha256": plan["work_plan_sha256"],
        "source_unit_id": source_unit_id,
        "source_binding_sha256": source["source_binding_sha256"],
        "range_receipts": range_receipts,
        "range_receipts_sha256": _hash_value(range_receipts),
        "staging_artifact": staging_artifact,
        "memory_observation": {
            "whole_shard_materialized": False,
            "whole_model_materialized": False,
            "measured_peak_rss_bytes": None,
            "measured_peak_required_before_live_promotion": True,
        },
        "source_files_deleted": False,
    }
    _with_hash(doc, "receipt_sha256")
    receipt_errors = validate_source_traversal_receipt(plan, doc)
    if receipt_errors:
        raise ParallelScaffoldError(
            "generated traversal receipt invalid: " + "; ".join(receipt_errors)
        )
    return doc


def validate_source_traversal_receipt(
    plan: dict[str, Any], receipt: Any,
) -> list[str]:
    if not isinstance(receipt, dict) \
            or receipt.get("schema") != SOURCE_TRAVERSAL_RECEIPT_SCHEMA:
        return ["source traversal receipt schema mismatch"]
    errors: list[str] = []
    if receipt.get("receipt_sha256") != _hash_value(_without(receipt, "receipt_sha256")):
        errors.append("source traversal receipt hash mismatch")
    source = {row["unit_id"]: row for row in plan.get("source_units", [])}.get(
        receipt.get("source_unit_id")
    )
    if source is None:
        errors.append("source traversal unit is absent from plan")
        return errors
    if receipt.get("work_plan_sha256") != plan.get("work_plan_sha256") \
            or receipt.get("source_binding_sha256") != source["source_binding_sha256"]:
        errors.append("source traversal does not bind its exact plan unit")
    expected_ranges = [
        {
            "tensor": extent["tensor"],
            "source_shard_sha256": extent["source_shard_sha256"],
            "absolute_byte_range": extent["absolute_byte_range"],
            "bytes": extent["bytes"],
        }
        for extent in source["source_extents"]
    ]
    observed = receipt.get("range_receipts")
    if not isinstance(observed, list) or len(observed) != len(expected_ranges):
        errors.append("source traversal range coverage differs")
    else:
        for expected, row in zip(expected_ranges, observed, strict=True):
            if not isinstance(row, dict) or any(row.get(key) != value
                                                for key, value in expected.items()) \
                    or not isinstance(row.get("range_sha256"), str) \
                    or SHA_RE.fullmatch(row["range_sha256"]) is None:
                errors.append("source traversal range identity differs")
                break
        if receipt.get("range_receipts_sha256") != _hash_value(observed):
            errors.append("source traversal range receipt hash mismatch")
    artifact = receipt.get("staging_artifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("sha256"), str) \
            or SHA_RE.fullmatch(artifact["sha256"]) is None \
            or isinstance(artifact.get("bytes"), bool) \
            or not isinstance(artifact.get("bytes"), int):
        errors.append("source traversal staging artifact is invalid")
    memory = receipt.get("memory_observation")
    if not isinstance(memory, dict) or memory.get("whole_shard_materialized") is not False \
            or memory.get("whole_model_materialized") is not False \
            or memory.get("measured_peak_required_before_live_promotion") is not True:
        errors.append("source traversal memory boundary is invalid")
    if receipt.get("source_files_deleted") is not False:
        errors.append("source traversal deleted a parent source")
    return errors


def validate_output_receipt(plan: dict[str, Any], receipt: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(receipt, dict) or receipt.get("schema") != OUTPUT_RECEIPT_SCHEMA:
        return ["output receipt schema mismatch"]
    if receipt.get("receipt_sha256") != _hash_value(_without(receipt, "receipt_sha256")):
        errors.append("output receipt hash mismatch")
    expected = {row["output_unit_id"]: row for row in plan.get("output_units", [])}.get(
        receipt.get("output_unit_id")
    )
    if expected is None:
        errors.append("receipt output is not in the work plan")
    elif receipt.get("work_plan_sha256") != plan.get("work_plan_sha256") \
            or receipt.get("source_unit_id") != expected["source_unit_id"] \
            or receipt.get("source_binding_sha256") != expected["source_binding_sha256"] \
            or receipt.get("rate_id") != expected["rate_id"]:
        errors.append("receipt does not bind its exact planned output")
    traversal_sha = receipt.get("source_traversal_receipt_sha256")
    if not isinstance(traversal_sha, str) or SHA_RE.fullmatch(traversal_sha) is None:
        errors.append("receipt source traversal identity is invalid")
    for field in ("staging_artifact", "archive"):
        row = receipt.get(field)
        if not isinstance(row, dict) or not isinstance(row.get("sha256"), str) \
                or SHA_RE.fullmatch(row["sha256"]) is None \
                or isinstance(row.get("bytes"), bool) or not isinstance(row.get("bytes"), int):
            errors.append(f"receipt {field} identity is invalid")
    if expected is not None and expected["attestation_required"]:
        root = receipt.get("attestation", {}).get("root_sha256") \
            if isinstance(receipt.get("attestation"), dict) else None
        if not isinstance(root, str) or SHA_RE.fullmatch(root) is None:
            errors.append("receipt attestation root is missing")
    if receipt.get("lifecycle", {}).get("source_files_deleted") is not False \
            or receipt.get("claims") != {
                "codec_complete": True, "quality": False, "deployable": False
            }:
        errors.append("receipt lifecycle/claim boundary is invalid")
    return errors


def build_merge_manifest(
    plan: dict[str, Any], receipts: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Require exact 6,150-receipt coverage and create canonical rate manifests."""
    plan_errors = validate_work_plan(plan)
    if plan_errors:
        raise ParallelScaffoldError("cannot merge invalid plan: " + "; ".join(plan_errors))
    by_id: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        errors = validate_output_receipt(plan, receipt)
        if errors:
            raise ParallelScaffoldError("invalid output receipt: " + "; ".join(errors))
        output_id = receipt["output_unit_id"]
        if output_id in by_id:
            raise ParallelScaffoldError(f"duplicate output receipt: {output_id}")
        by_id[output_id] = receipt
    expected = {row["output_unit_id"] for row in plan["output_units"]}
    missing, extra = sorted(expected - set(by_id)), sorted(set(by_id) - expected)
    if missing or extra:
        raise ParallelScaffoldError(
            f"receipt coverage differs: missing={len(missing)} extra={len(extra)}"
        )
    rate_manifests = []
    for rate_id in RATES:
        components = [
            {
                "output_unit_id": output_id,
                "receipt_sha256": row["receipt_sha256"],
                "archive": row["archive"], "attestation": row["attestation"],
                "source_binding_sha256": row["source_binding_sha256"],
            }
            for output_id, row in sorted(by_id.items()) if row["rate_id"] == rate_id
        ]
        rate_manifests.append({
            "rate_id": rate_id, "component_count": len(components),
            "whole_artifact_component_bytes": sum(row["archive"]["bytes"]
                                                    for row in components),
            "components": components, "components_sha256": _hash_value(components),
        })
    doc: dict[str, Any] = {
        "schema": MERGE_MANIFEST_SCHEMA, "created_at": _now(),
        "status": "codec-structural-reassembly-complete-quality-deferred",
        "work_plan_sha256": plan["work_plan_sha256"],
        "source_manifest_sha256": plan["inventory"]["source_manifest_sha256"],
        "receipt_count": len(by_id), "rate_manifests": rate_manifests,
        "runtime_gate": {
            "gptoss_numerical_loader_parity": False,
            "tokenizer_bound_quality_evaluation": False,
            "campaign_execution_permitted": False,
        },
        "source_files_deleted": False, "quality_claims_permitted": False,
    }
    return _with_hash(doc, "merge_manifest_sha256")


class MoeArchiveIndex:
    """Resolve router-selected experts to canonical planned or merged archives."""

    def __init__(self, plan: dict[str, Any], merge: dict[str, Any] | None = None) -> None:
        errors = validate_work_plan(plan)
        if errors:
            raise ParallelScaffoldError("invalid plan for MoE index: " + "; ".join(errors))
        self.plan = plan
        self._units = {row["unit_id"]: row for row in plan["source_units"]}
        self._components: dict[str, dict[str, Any]] = {}
        if merge is not None:
            if merge.get("schema") != MERGE_MANIFEST_SCHEMA \
                    or merge.get("work_plan_sha256") != plan["work_plan_sha256"] \
                    or merge.get("merge_manifest_sha256") != _hash_value(
                        _without(merge, "merge_manifest_sha256")
                    ):
                raise ParallelScaffoldError("merge manifest is invalid for MoE index")
            for rate in merge["rate_manifests"]:
                for component in rate["components"]:
                    self._components[component["output_unit_id"]] = component

    def resolve_expert(self, *, rate_id: str, layer: int, expert: int) -> dict[str, Any]:
        if rate_id not in RATES or not 0 <= layer < LAYERS or not 0 <= expert < EXPERTS:
            raise ParallelScaffoldError("MoE route coordinate is outside the reviewed layout")
        start = expert // EXPERTS_PER_BATCH * EXPERTS_PER_BATCH
        source_id = f"expert/layer={layer:03d}/experts={start:03d}-{start + 7:03d}"
        output_id = f"{source_id}/rate={rate_id}"
        result: dict[str, Any] = {
            "rate_id": rate_id, "layer": layer, "expert": expert,
            "source_unit_id": source_id, "output_unit_id": output_id,
            "source_binding_sha256": self._units[source_id]["source_binding_sha256"],
            "tensors": {
                "gate_up": f"block.{layer}.mlp.mlp1_weight.expert.{expert:03d}",
                "down": f"block.{layer}.mlp.mlp2_weight.expert.{expert:03d}",
            },
            "canonical_orientation": "out_features,in_features",
            "numerical_execution_proven": False,
        }
        component = self._components.get(output_id)
        result["archive"] = component["archive"] if component else None
        result["attestation"] = component["attestation"] if component else None
        result["status"] = "artifact-bound" if component else "pending-output-receipt"
        return result


def build_tokenizer_binding(
    *, model_source_manifest_sha256: str, files: Iterable[Path],
    chat_template_sha256: str,
) -> dict[str, Any]:
    """Bind tokenizer files without assuming a tokenizer is present or compatible."""
    if SHA_RE.fullmatch(model_source_manifest_sha256) is None \
            or SHA_RE.fullmatch(chat_template_sha256) is None:
        raise ParallelScaffoldError("tokenizer source/chat-template hash is invalid")
    rows = []
    for path in files:
        raw = Path(path)
        if raw.is_symlink():
            raise ParallelScaffoldError(f"symlinked tokenizer file is forbidden: {raw}")
        rows.append(_artifact(raw.resolve(strict=True)))
    rows.sort(key=lambda row: row["path"])
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise ParallelScaffoldError("tokenizer binding needs unique concrete files")
    doc: dict[str, Any] = {
        "schema": TOKENIZER_BINDING_SCHEMA, "created_at": _now(),
        "model_source_manifest_sha256": model_source_manifest_sha256,
        "files": rows, "chat_template_sha256": chat_template_sha256,
        "token_id_semantics_reviewed": False,
        "quality_evaluation_permitted": False,
        "promotion_requires_roundtrip_and_reference_token_parity": True,
    }
    return _with_hash(doc, "tokenizer_binding_sha256")


def validate_tokenizer_binding(doc: Any, *, verify_files: bool = False) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != TOKENIZER_BINDING_SCHEMA:
        return ["tokenizer binding schema mismatch"]
    errors: list[str] = []
    if doc.get("tokenizer_binding_sha256") != _hash_value(
            _without(doc, "tokenizer_binding_sha256")):
        errors.append("tokenizer binding hash mismatch")
    if doc.get("quality_evaluation_permitted") is not False \
            or doc.get("token_id_semantics_reviewed") is not False:
        errors.append("unreviewed tokenizer binding is not fail closed")
    files = doc.get("files")
    if not isinstance(files, list) or not files:
        errors.append("tokenizer binding files are missing")
    elif verify_files:
        for row in files:
            try:
                observed = _artifact(Path(row["path"]))
                if observed != row:
                    errors.append(f"tokenizer file identity differs: {row.get('path')}")
            except (OSError, KeyError, TypeError, mxfp4.Mxfp4Error) as exc:
                errors.append(f"tokenizer file verification failed: {exc}")
    return errors


def build_pending_wiring_packet(
    work_plan: dict[str, Any], campaign_plan_path: Path = DEFAULT_CAMPAIGN_PLAN,
) -> dict[str, Any]:
    """Describe the exact 40 promotions while guaranteeing that none is active."""
    errors = validate_work_plan(work_plan)
    if errors:
        raise ParallelScaffoldError("cannot wire invalid work plan: " + "; ".join(errors))
    campaign_plan_path = Path(campaign_plan_path).resolve(strict=True)
    campaign = _read_json(campaign_plan_path)
    cells = sorted(
        [row for row in campaign.get("cells", []) if row.get("model_label") == "120B"],
        key=lambda row: row["cell_id"],
    )
    branch_map = {branch: (command, adapter) for branch, command, adapter in BRANCHES}
    registry = _read_json(DEFAULT_ADAPTER_REGISTRY)
    registered_ids = {
        row.get("adapter_id") for row in registry.get("entries", [])
        if isinstance(row, dict)
    }
    target_adapter_ids = {adapter for _branch, _command, adapter in BRANCHES}
    if registered_ids & target_adapter_ids:
        raise ParallelScaffoldError("a target GPT-OSS adapter is already registered")
    bindings = []
    for cell in cells:
        branch = cell.get("branch")
        if branch not in branch_map:
            raise ParallelScaffoldError(f"unknown 120B branch in campaign: {branch}")
        command, adapter = branch_map[branch]
        if cell.get("command") != command or cell.get("adapter_id") != adapter:
            raise ParallelScaffoldError(f"120B cell adapter identity changed: {cell['cell_id']}")
        bindings.append({
            "cell_id": cell["cell_id"],
            "cell_identity_sha256": cell["cell_identity_sha256"],
            "cell_spec_sha256": cell["cell_spec_sha256"],
            "rate_id": cell["rate_id"], "branch": branch,
            "command": command, "adapter_id": adapter,
            "runtime_spec_path": cell["runtime_spec_path"],
            "runtime_spec_schema": cell["runtime_spec_schema"],
            "dependencies": cell["dependencies"],
            "work_plan_sha256": work_plan["work_plan_sha256"],
            "status": "pending-not-written-not-registered",
            "execution_permitted": False,
        })
    existing_runtime_specs = [
        row["runtime_spec_path"] for row in bindings
        if (ROOT / row["runtime_spec_path"]).exists()
    ]
    if existing_runtime_specs:
        raise ParallelScaffoldError(
            f"expected pending runtime specs are already present: {len(existing_runtime_specs)}"
        )
    doc: dict[str, Any] = {
        "schema": PENDING_WIRING_SCHEMA, "created_at": _now(),
        "status": "pending-only-no-live-mutation",
        "campaign_plan": {**_artifact(campaign_plan_path),
                          "plan_sha256": campaign.get("plan_sha256")},
        "work_plan": {
            "path": str(DEFAULT_WORK_PLAN),
            "work_plan_sha256": work_plan["work_plan_sha256"],
            "source_unit_count": EXPECTED_SOURCE_UNITS,
            "output_unit_count": EXPECTED_OUTPUT_UNITS,
        },
        "matrix": {"rate_count": 10, "branch_count": 4, "cell_count": len(bindings)},
        "cell_bindings": bindings,
        "proposed_registry_entries": [
            {"adapter_id": adapter, "branch": branch,
             "status": "pending-review-no-registry-write"}
            for branch, _command, adapter in BRANCHES
        ],
        "observed_pending_state": {
            "adapter_registry": _artifact(DEFAULT_ADAPTER_REGISTRY),
            "target_registry_entries_present": 0,
            "target_runtime_specs_present": 0,
        },
        "promotion_gate": {
            "all_required": True, "currently_permitted": False,
            "requirements": [
                "sub_120b_terminal_quiescent_checkpoint",
                "all_four_adapters_reviewed",
                "exact_40_runtime_specs_validated",
                "adapter_preflights_pass",
                "tokenizer_reference_parity_pass",
                "gptoss_moe_numerical_runtime_parity_pass",
                "bit_exact_encoder_canary_pass",
                "disk_and_dynamic_ram_admission_pass",
                "rollback_point_sealed",
                "observer_structural_readiness_pass",
            ],
        },
        "live_registry_mutated": False, "live_runtime_specs_written": False,
        "live_queue_mutated": False, "quality_claims_permitted": False,
    }
    _with_hash(doc, "pending_wiring_sha256")
    packet_errors = validate_pending_wiring(doc)
    if packet_errors:
        raise ParallelScaffoldError("generated pending packet invalid: " + "; ".join(packet_errors))
    return doc


def validate_pending_wiring(doc: Any) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != PENDING_WIRING_SCHEMA:
        return ["pending-wiring schema mismatch"]
    errors: list[str] = []
    if doc.get("pending_wiring_sha256") != _hash_value(_without(doc, "pending_wiring_sha256")):
        errors.append("pending-wiring hash mismatch")
    if doc.get("status") != "pending-only-no-live-mutation" \
            or doc.get("live_registry_mutated") is not False \
            or doc.get("live_runtime_specs_written") is not False \
            or doc.get("live_queue_mutated") is not False \
            or doc.get("quality_claims_permitted") is not False:
        errors.append("pending packet crosses the live-mutation boundary")
    bindings = doc.get("cell_bindings")
    if not isinstance(bindings, list) or len(bindings) != 40:
        errors.append("pending packet does not bind exactly 40 cells")
        bindings = []
    coverage = {(row.get("rate_id"), row.get("branch")) for row in bindings
                if isinstance(row, dict)}
    if coverage != {(rate, branch) for rate in RATES
                    for branch, _command, _adapter in BRANCHES}:
        errors.append("pending packet is not the exact 10x4 matrix")
    if any(row.get("status") != "pending-not-written-not-registered"
           or row.get("execution_permitted") is not False for row in bindings):
        errors.append("a pending cell is incorrectly executable")
    branch_authority = {
        branch: (command, adapter) for branch, command, adapter in BRANCHES
    }
    for row in bindings:
        if not isinstance(row, dict):
            continue
        expected = branch_authority.get(row.get("branch"))
        if expected is None or (row.get("command"), row.get("adapter_id")) != expected \
                or not isinstance(row.get("cell_identity_sha256"), str) \
                or SHA_RE.fullmatch(row["cell_identity_sha256"]) is None \
                or not isinstance(row.get("cell_spec_sha256"), str) \
                or SHA_RE.fullmatch(row["cell_spec_sha256"]) is None \
                or row.get("work_plan_sha256") != doc.get("work_plan", {}).get(
                    "work_plan_sha256"
                ):
            errors.append(f"pending cell authority is invalid: {row.get('cell_id')}")
    matrix = doc.get("matrix")
    if matrix != {"rate_count": 10, "branch_count": 4, "cell_count": 40}:
        errors.append("pending matrix cardinality is invalid")
    proposed = doc.get("proposed_registry_entries")
    if not isinstance(proposed, list) or len(proposed) != 4 \
            or any(row.get("status") != "pending-review-no-registry-write"
                   for row in proposed if isinstance(row, dict)):
        errors.append("proposed registry entries are not strictly pending")
    if doc.get("promotion_gate", {}).get("currently_permitted") is not False:
        errors.append("pending promotion gate is not closed")
    campaign = doc.get("campaign_plan")
    try:
        if not isinstance(campaign, dict):
            raise ParallelScaffoldError("campaign binding is missing")
        observed = _artifact(Path(campaign["path"]))
        if any(observed[field] != campaign.get(field)
               for field in ("path", "sha256", "bytes")):
            raise ParallelScaffoldError("campaign plan file identity differs")
        live = _read_json(Path(campaign["path"]))
        if live.get("plan_sha256") != campaign.get("plan_sha256"):
            raise ParallelScaffoldError("campaign plan semantic identity differs")
    except (OSError, KeyError, TypeError, ParallelScaffoldError,
            mxfp4.Mxfp4Error) as exc:
        errors.append(f"pending campaign verification failed: {exc}")
    observed_state = doc.get("observed_pending_state")
    try:
        if not isinstance(observed_state, dict) \
                or observed_state.get("target_registry_entries_present") != 0 \
                or observed_state.get("target_runtime_specs_present") != 0:
            raise ParallelScaffoldError("pending observed-state counters are invalid")
        registry_binding = observed_state["adapter_registry"]
        observed_registry = _artifact(Path(registry_binding["path"]))
        if any(observed_registry[field] != registry_binding.get(field)
               for field in ("path", "sha256", "bytes")):
            raise ParallelScaffoldError("adapter registry identity changed")
        registry = _read_json(Path(registry_binding["path"]))
        registered = {
            row.get("adapter_id") for row in registry.get("entries", [])
            if isinstance(row, dict)
        }
        targets = {adapter for _branch, _command, adapter in BRANCHES}
        if registered & targets:
            raise ParallelScaffoldError("a target adapter is no longer pending")
        existing_specs = [
            row["runtime_spec_path"] for row in bindings
            if isinstance(row, dict) and isinstance(row.get("runtime_spec_path"), str)
            and (ROOT / row["runtime_spec_path"]).exists()
        ]
        if existing_specs:
            raise ParallelScaffoldError("one or more target runtime specs now exists")
    except (OSError, KeyError, TypeError, ParallelScaffoldError,
            mxfp4.Mxfp4Error) as exc:
        errors.append(f"pending live-state verification failed: {exc}")
    return errors


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mxfp4._atomic_json(path, doc)


def _selftest() -> None:
    plan = build_work_plan()
    packet = build_pending_wiring_packet(plan)
    if validate_work_plan(plan) or validate_pending_wiring(packet):
        raise ParallelScaffoldError("parallel scaffold selftest failed")
    index = MoeArchiveIndex(plan)
    route = index.resolve_expert(rate_id="0.5", layer=35, expert=127)
    if route["status"] != "pending-output-receipt" \
            or route["tensors"]["down"] != "block.35.mlp.mlp2_weight.expert.127":
        raise ParallelScaffoldError("MoE structural lookup selftest failed")
    print(json.dumps({
        "status": "ok", "work_plan_sha256": plan["work_plan_sha256"],
        "source_units": EXPECTED_SOURCE_UNITS, "output_units": EXPECTED_OUTPUT_UNITS,
        "pending_cells": 40, "live_mutation": False,
    }, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    build.add_argument("--campaign-plan", type=Path, default=DEFAULT_CAMPAIGN_PLAN)
    build.add_argument("--work-plan-output", type=Path, default=DEFAULT_WORK_PLAN)
    build.add_argument("--pending-wiring-output", type=Path, default=DEFAULT_PENDING_WIRING)
    verify = sub.add_parser("verify")
    verify.add_argument("--work-plan", type=Path, default=DEFAULT_WORK_PLAN)
    verify.add_argument("--pending-wiring", type=Path, default=DEFAULT_PENDING_WIRING)
    sub.add_parser("selftest")
    args = parser.parse_args(argv)
    try:
        if args.command == "selftest":
            _selftest()
            return 0
        if args.command == "build":
            plan = build_work_plan(args.inventory)
            packet = build_pending_wiring_packet(plan, args.campaign_plan)
            _write_json(args.work_plan_output, plan)
            _write_json(args.pending_wiring_output, packet)
            print(json.dumps({
                "status": "ok", "execution_permitted": False,
                "work_plan": str(args.work_plan_output.resolve()),
                "work_plan_sha256": plan["work_plan_sha256"],
                "pending_wiring": str(args.pending_wiring_output.resolve()),
                "pending_wiring_sha256": packet["pending_wiring_sha256"],
                "source_units": EXPECTED_SOURCE_UNITS,
                "output_units": EXPECTED_OUTPUT_UNITS, "pending_cells": 40,
            }, indent=2, sort_keys=True))
            return 0
        plan, packet = _read_json(args.work_plan), _read_json(args.pending_wiring)
        errors = validate_work_plan(plan) + validate_pending_wiring(packet)
        if packet.get("work_plan", {}).get("work_plan_sha256") \
                != plan.get("work_plan_sha256"):
            errors.append("pending packet and work plan hashes differ")
        print(json.dumps({"status": "ok" if not errors else "invalid",
                          "errors": errors}, indent=2, sort_keys=True))
        return 0 if not errors else 2
    except (ParallelScaffoldError, mxfp4.Mxfp4Error, OSError, KeyError,
            TypeError, ValueError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

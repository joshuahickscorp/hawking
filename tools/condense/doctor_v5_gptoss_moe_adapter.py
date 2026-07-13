#!/usr/bin/env python3.12
"""Fail-closed GPT-OSS 120B MXFP4 adapter contract for Doctor-v5.

The typed spec builder makes all ten Ultra codec cells addressable now.  The
preflight is executable and validates the real checkpoint plus logical parameter
authority without dense materialization.  ``run`` intentionally refuses until
the source-bound STR2 subarchive reassembly/runtime blockers reported by
``capabilities`` are implemented and reviewed; a contract must never masquerade
as a completed codec experiment.
"""
from __future__ import annotations

import argparse
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import doctor_v5_gptoss_mxfp4 as mxfp4


ADAPTER_ID = "doctor-v5-strand-ladder-gpt-oss-moe"
ADAPTER_VERSION = "0.1-contract"
OPERATION = "condense_control"
MODEL_FAMILY = "gpt-oss-moe"
BACKEND = "apple-cpu-strand"
SPEC_SCHEMA = "hawking.doctor_v5_strand_ladder_spec.v1"
PREFLIGHT_SCHEMA = "hawking.doctor_v5_gptoss_moe_preflight.v1"
CAPABILITY_SCHEMA = "hawking.doctor_v5_gptoss_moe_capabilities.v1"
DEFAULT_INVENTORY = mxfp4.DEFAULT_INVENTORY
DEFAULT_CENSUS = mxfp4.DEFAULT_CENSUS
QUANTIZER = ROOT / "vendor/strand-quant/target/release/quantize-model"
ATTESTOR = ROOT / "vendor/strand-decode-kernel/target/release/attest-strand"
DECODER = ROOT / "vendor/strand-decode-kernel/target/release/archive-to-safetensors"
SHA_RE = re.compile(r"[0-9a-f]{64}")
MAX_JSON_BYTES = 64 * 1024 * 1024
RATE_GEOMETRY: dict[str, tuple[int, int, int]] = {
    "4": (4, 1, 256), "3": (3, 1, 256), "2": (2, 1, 256),
    "1": (1, 1, 256), "0.8": (3, 5, 256),
    "0.55": (1, 3, 256), "0.5": (1, 4, 512),
    "0.33": (1, 8, 1024), "0.25": (1, 16, 2048),
    "0.1": (1, 32, 8192),
}
CANONICAL_RATES = dict(mxfp4.CANONICAL_RATES)
BLOCKERS = (
    {
        "id": "strand-reader-whole-file-u8-unsupported",
        "layer": "codec_input",
        "detail": (
            "vendor/strand-quant SafeTensors::open uses std::fs::read and to_f32 "
            "rejects U8; raw 10.5 GB GPT-OSS shards cannot be passed directly"
        ),
        "exit_criterion": (
            "adapter stages only validated BF16 2-D units and quantizer never opens a source shard"
        ),
    },
    {
        "id": "original-source-provenance-reassembly-missing",
        "layer": "artifact_format",
        "detail": (
            "STR2 binds the staging safetensors SHA, while campaign evidence also needs "
            "original shard/tensor/expert byte ranges and deterministic reassembly"
        ),
        "exit_criterion": (
            "reviewed manifest binds every archive tensor to source shard SHA, byte ranges, "
            "orientation, staging SHA, archive SHA, and archive attestation root"
        ),
    },
    {
        "id": "gptoss-moe-str2-loader-missing",
        "layer": "runtime",
        "detail": (
            "no Apple-Silicon loader maps per-expert STR2 archives into router-selected "
            "GPT-OSS fused gate/up/down execution"
        ),
        "exit_criterion": "round-trip and inference parity tests pass for routing and fused projections",
    },
    {
        "id": "gptoss-tokenizer-missing",
        "layer": "evaluation",
        "detail": "the completed local source contains config/dtypes/index/shards but no tokenizer",
        "exit_criterion": "tokenizer/chat-template files are downloaded, hashed, and source-bound",
    },
    {
        "id": "ten-artifact-disk-retention-infeasible",
        "layer": "lifecycle",
        "detail": (
            "ten nominal payload ceilings total 182,983,666,640 bytes before overhead, "
            "which exceeds space above the 150 GB reserve at current capacity"
        ),
        "exit_criterion": (
            "model-by-rate four-branch chain reporting and hash-before-GC retention receipts are live"
        ),
    },
)


class AdapterError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) \
                or info.st_size > MAX_JSON_BYTES:
            raise AdapterError(f"invalid JSON file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"JSON root is not an object: {path}")
    return value


def _workspace_path(raw: str | Path, *, must_exist: bool = True) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    try:
        resolved = path.resolve(strict=must_exist)
        resolved.relative_to(ROOT.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise AdapterError(f"path is missing or outside workspace: {raw}") from exc
    if must_exist and path.is_symlink():
        raise AdapterError(f"symlinked input forbidden: {raw}")
    return resolved


def _artifact(path: Path) -> dict[str, Any]:
    digest, size = mxfp4._hash_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def _known_artifact(path: Path, sha256: Any, size: Any) -> dict[str, Any]:
    if not isinstance(sha256, str) or SHA_RE.fullmatch(sha256) is None \
            or isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise AdapterError(f"invalid inherited source identity: {path}")
    return {"path": str(path.resolve(strict=True)), "sha256": sha256, "bytes": size}


def _input(role: str, artifact: dict[str, Any]) -> dict[str, Any]:
    return {"role": role, **artifact}


def _rate_geometry(rate_id: str) -> dict[str, Any]:
    if rate_id not in CANONICAL_RATES:
        raise AdapterError(f"unknown canonical rate: {rate_id}")
    bits, vector_dim, block_len = RATE_GEOMETRY[rate_id]
    target = CANONICAL_RATES[rate_id]
    nominal = Fraction(bits, vector_dim)
    if nominal > target:
        raise AdapterError(f"codec geometry exceeds target: {rate_id}")
    return {
        "rate_id": rate_id, "target_bpw": float(target),
        "target_fraction": [target.numerator, target.denominator],
        "symbol_bits": bits, "vector_dim": vector_dim, "block_len": block_len,
        "artifact_mode": "packed_scalar_control" if vector_dim == 1
                         else "packed_vector_control",
        "nominal_symbol_payload_bpw": float(nominal),
        "nominal_symbol_payload_fraction": [nominal.numerator, nominal.denominator],
        "whole_artifact_target_must_be_measured": True,
    }


def capabilities() -> dict[str, Any]:
    return {
        "schema": CAPABILITY_SCHEMA,
        "adapter_id": ADAPTER_ID, "adapter_version": ADAPTER_VERSION,
        "operation": OPERATION, "model_family": MODEL_FAMILY, "backend": BACKEND,
        "labels": ["120B"], "rates": list(CANONICAL_RATES),
        "logical_parameter_denominator": 116_829_156_672,
        "native_tensor_payload_bpw": 4.467981630796993,
        "implemented": {
            "typed_spec_builder": True,
            "header_only_checkpoint_validation": True,
            "logical_parameter_accounting": True,
            "bounded_memory_per_expert_mxfp4_to_bf16": True,
            "source_read_only_byte_range_receipts": True,
            "full_str2_campaign_execution": False,
            "quality_evaluation": False,
            "apple_silicon_moe_runtime": False,
        },
        "reviewed_for_live_campaign_execution": False,
        "blockers": list(BLOCKERS),
        "source_deletion_permitted": False, "quality_claims_permitted": False,
    }


def _load_inventory(path: Path) -> dict[str, Any]:
    path = _workspace_path(path)
    doc = _read_json(path)
    errors = mxfp4.validate_inventory(doc)
    if errors:
        raise AdapterError("MXFP4 inventory is invalid: " + "; ".join(errors))
    if doc.get("execution_readiness", {}).get("full_ten_rate_codec_execution") \
            != "blocked":
        raise AdapterError("contract adapter requires fail-closed inventory status")
    return doc


def _spec_inputs(inventory_path: Path, inventory: dict[str, Any]) -> list[dict[str, Any]]:
    binding = inventory["source_binding"]
    inputs = [
        _input("adapter_source", _artifact(Path(__file__).resolve())),
        _input("mxfp4_core", _artifact(Path(mxfp4.__file__).resolve())),
        _input("mxfp4_inventory", _artifact(inventory_path)),
        _input("source_census", _known_artifact(
            Path(binding["census_path"]), binding["census_file_sha256"],
            binding["census_file_bytes"],
        )),
        _input("quantizer", _artifact(QUANTIZER)),
        _input("attestor", _artifact(ATTESTOR)),
        _input("decoder", _artifact(DECODER)),
    ]
    for role in ("config", "dtypes", "index"):
        inputs.append(_input(f"model_metadata:{role}", binding[role]))
    for row in binding["shards"]:
        inputs.append(_input(
            f"source_shard:{row['ordinal']:05d}",
            _known_artifact(Path(row["path"]), row["file_sha256"], row["file_bytes"]),
        ))
    roles = [row["role"] for row in inputs]
    paths = [row["path"] for row in inputs]
    if len(roles) != len(set(roles)) or len(paths) != len(set(paths)):
        raise AdapterError("typed input inventory contains duplicates")
    return sorted(inputs, key=lambda row: row["role"])


def build_spec(*, rate_id: str, cell_id: str, cell_identity_sha256: str,
               program_spec_sha256: str, resource_admission_sha256: str,
               disk_reserve_bytes: int, scratch_budget_bytes: int, threads: int,
               inventory_path: Path = DEFAULT_INVENTORY) -> dict[str, Any]:
    for name, value in (
        ("cell_identity_sha256", cell_identity_sha256),
        ("program_spec_sha256", program_spec_sha256),
        ("resource_admission_sha256", resource_admission_sha256),
    ):
        if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
            raise AdapterError(f"{name} is invalid")
    if not isinstance(cell_id, str) or not cell_id:
        raise AdapterError("cell_id is invalid")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (disk_reserve_bytes, scratch_budget_bytes, threads)) \
            or disk_reserve_bytes < 150_000_000_000 \
            or scratch_budget_bytes < 12_000_000_000 \
            or not 1 <= threads <= 16:
        raise AdapterError("resource values violate the Ultra safety envelope")
    inventory_path = _workspace_path(inventory_path)
    inventory = _load_inventory(inventory_path)
    geometry = _rate_geometry(rate_id)
    vector = geometry["vector_dim"] > 1
    spec: dict[str, Any] = {
        "schema": SPEC_SCHEMA, "label": "120B",
        "campaign_binding": {
            "cell_id": cell_id, "cell_identity_sha256": cell_identity_sha256,
            "branch": "codec_control", "target_rate_id": rate_id,
            "target_rate_bpw": geometry["target_bpw"], "label": "120B",
        },
        "adapter_id": ADAPTER_ID, "operation": OPERATION,
        "model_family": MODEL_FAMILY, "backend": BACKEND,
        "codec": {
            **geometry, "tensor_scope": "all-2d",
            "source_encoding": "gpt-oss-mxfp4-e2m1-ue8",
            "staging_dtype": "BF16", "staging_orientation": "out,in",
            "experts_per_batch": 8, "quality": True, "rht_cols": True,
            "ragged_v2": True, "sdsq_sideinfo": True,
            "outlier_channel_pct": 0 if vector else 1,
            "outlier_bits": 8, "c2f_outl": not vector,
            "adaptive_scales": rate_id != "0.1",
            "learned_codebook": False, "allow_over_ceiling_control": True,
        },
        "evaluation": {
            "mode": "deferred", "reason": "tokenizer_and_gptoss_str2_runtime_missing",
            "retain_dense_reconstruction": False,
        },
        "doctor_hook": {"method": "none",
                        "dependent_cells_require_packed_base": True},
        "resources": {"disk_reserve_bytes": disk_reserve_bytes,
                      "scratch_budget_bytes": scratch_budget_bytes, "threads": threads},
        "logical_parameter_authority": {
            "inventory_path": str(inventory_path),
            "inventory_sha256": inventory["inventory_sha256"],
            "logical_model_parameters": 116_829_156_672,
            "serialized_u8_elements_are_not_parameters": True,
            "ue8_scales_are_side_information": True,
        },
        "execution_contract": {
            "status": "blocked_pending_reviewed_runtime",
            "preflight_executable": True, "codec_execution_executable": False,
            "unit_order": ["expert_batches", "dense_layer_tensors",
                           "embedding", "output_head", "lossless_sidecar",
                           "reassembly_manifest", "attestation"],
            "checkpoint_after_every_unit": True,
            "staging_deleted_only_after_archive_and_receipt_hash": True,
            "source_files_never_deleted": True,
            "blocker_ids": [row["id"] for row in BLOCKERS],
        },
        "program_spec_sha256": program_spec_sha256,
        "resource_admission_sha256": resource_admission_sha256,
        "source_deletion_permitted": False, "quality_claims_permitted": False,
        "inputs": _spec_inputs(inventory_path, inventory),
    }
    errors = validate_spec(spec)
    if errors:
        raise AdapterError("built invalid GPT-OSS spec: " + "; ".join(errors))
    return spec


def validate_spec(spec: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(spec, dict):
        return ["spec is not an object"]
    for key, expected in (
        ("schema", SPEC_SCHEMA), ("label", "120B"), ("adapter_id", ADAPTER_ID),
        ("operation", OPERATION), ("model_family", MODEL_FAMILY), ("backend", BACKEND),
    ):
        if spec.get(key) != expected:
            errors.append(f"spec identity mismatch: {key}")
    campaign = spec.get("campaign_binding")
    codec = spec.get("codec")
    if not isinstance(campaign, dict) or set(campaign) != {
            "cell_id", "cell_identity_sha256", "branch", "target_rate_id",
            "target_rate_bpw", "label"}:
        errors.append("campaign binding is invalid")
    elif campaign.get("branch") != "codec_control" or campaign.get("label") != "120B" \
            or not isinstance(campaign.get("cell_identity_sha256"), str) \
            or SHA_RE.fullmatch(campaign["cell_identity_sha256"]) is None:
        errors.append("campaign binding identity is invalid")
    if not isinstance(codec, dict) or codec.get("rate_id") not in CANONICAL_RATES:
        errors.append("codec rate is invalid")
    else:
        try:
            geometry = _rate_geometry(codec["rate_id"])
            for field in ("target_bpw", "symbol_bits", "vector_dim", "block_len",
                          "artifact_mode"):
                if codec.get(field) != geometry[field]:
                    errors.append(f"codec geometry mismatch: {field}")
            if isinstance(campaign, dict) and (
                    campaign.get("target_rate_id") != codec["rate_id"]
                    or campaign.get("target_rate_bpw") != geometry["target_bpw"]):
                errors.append("campaign/codec rate binding mismatch")
        except AdapterError as exc:
            errors.append(str(exc))
    for field in ("program_spec_sha256", "resource_admission_sha256"):
        if not isinstance(spec.get(field), str) or SHA_RE.fullmatch(spec[field]) is None:
            errors.append(f"{field} is invalid")
    if spec.get("source_deletion_permitted") is not False \
            or spec.get("quality_claims_permitted") is not False:
        errors.append("claim/source-deletion boundary is invalid")
    authority = spec.get("logical_parameter_authority")
    if not isinstance(authority, dict) \
            or authority.get("logical_model_parameters") != 116_829_156_672 \
            or authority.get("serialized_u8_elements_are_not_parameters") is not True \
            or authority.get("ue8_scales_are_side_information") is not True:
        errors.append("logical parameter authority is invalid")
    contract = spec.get("execution_contract")
    if not isinstance(contract, dict) or contract.get("codec_execution_executable") is not False \
            or contract.get("blocker_ids") != [row["id"] for row in BLOCKERS]:
        errors.append("fail-closed execution contract is invalid")
    inputs = spec.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        errors.append("typed input inventory is missing")
    else:
        roles: set[str] = set()
        for row in inputs:
            if not isinstance(row, dict) or set(row) != {"role", "path", "sha256", "bytes"} \
                    or not isinstance(row.get("role"), str) or row["role"] in roles \
                    or not isinstance(row.get("path"), str) \
                    or not isinstance(row.get("sha256"), str) \
                    or SHA_RE.fullmatch(row["sha256"]) is None \
                    or isinstance(row.get("bytes"), bool) or not isinstance(row.get("bytes"), int):
                errors.append("typed input row is invalid or duplicate")
                break
            roles.add(row["role"])
    return errors


def _write_spec(path: Path, spec: dict[str, Any]) -> None:
    path = _workspace_path(path, must_exist=False)
    mxfp4._atomic_json(path, spec)


def preflight(spec_path: Path, output: Path | None = None) -> dict[str, Any]:
    spec_path = _workspace_path(spec_path)
    spec = _read_json(spec_path)
    errors = validate_spec(spec)
    if errors:
        raise AdapterError("typed spec is invalid: " + "; ".join(errors))
    inputs = {row["role"]: row for row in spec["inputs"]}
    inventory_row = inputs.get("mxfp4_inventory")
    census_row = inputs.get("source_census")
    if not isinstance(inventory_row, dict) or not isinstance(census_row, dict):
        raise AdapterError("typed spec lacks inventory/census inputs")
    inventory_path = _workspace_path(inventory_row["path"])
    inventory = _load_inventory(inventory_path)
    if _artifact(inventory_path)["sha256"] != inventory_row["sha256"] \
            or inventory["inventory_sha256"] != spec[
                "logical_parameter_authority"
            ]["inventory_sha256"]:
        raise AdapterError("live MXFP4 inventory differs from typed spec")
    inspection = mxfp4.inspect_model(Path(inventory["model"]["model_dir"]),
                                     Path(census_row["path"]))
    live = mxfp4.build_inventory(inspection)
    if live["tensor_inventory"]["sha256"] != inventory["tensor_inventory"]["sha256"] \
            or live["parameter_accounting"]["logical_model_parameters"] \
            != 116_829_156_672:
        raise AdapterError("live GPT-OSS headers differ from bound logical inventory")
    resources = spec["resources"]
    usage = shutil.disk_usage(ROOT)
    target = CANONICAL_RATES[spec["codec"]["rate_id"]]
    maximum_payload = math.ceil(116_829_156_672 * target.numerator
                                / target.denominator / 8)
    available_above_reserve = usage.free - resources["disk_reserve_bytes"]
    receipt: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA, "created_at": _now(),
        "status": "preflight-complete-codec-blocked",
        "adapter_id": ADAPTER_ID,
        "campaign_binding": spec["campaign_binding"],
        "spec": {**_artifact(spec_path), "schema": SPEC_SCHEMA},
        "inventory": {**_artifact(inventory_path),
                      "inventory_sha256": inventory["inventory_sha256"]},
        "logical_model_parameters": 116_829_156_672,
        "native_tensor_payload_bpw": 4.467981630796993,
        "resources": {
            "disk_free_bytes": usage.free,
            "disk_reserve_bytes": resources["disk_reserve_bytes"],
            "available_above_reserve_bytes": available_above_reserve,
            "maximum_target_payload_bytes": maximum_payload,
            "single_cell_payload_fits_above_reserve": available_above_reserve > maximum_payload,
            "whole_model_or_shard_materialized": False,
        },
        "blockers": list(BLOCKERS),
        "quality_claims_permitted": False, "source_deletion_permitted": False,
    }
    receipt["receipt_sha256"] = _hash_value(receipt)
    if output is not None:
        output = _workspace_path(output, must_exist=False)
        mxfp4._atomic_json(output, receipt)
    return receipt


def _selftest() -> None:
    caps = capabilities()
    if caps["implemented"]["full_str2_campaign_execution"] is not False \
            or len(CANONICAL_RATES) != 10 \
            or any(_rate_geometry(rate_id)["nominal_symbol_payload_bpw"] > float(target)
                   for rate_id, target in CANONICAL_RATES.items()):
        raise AdapterError("adapter contract selftest failed")
    print(json.dumps({"status": "ok", "adapter_id": ADAPTER_ID,
                      "live_execution": False, "rate_count": 10}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("capabilities")
    sub.add_parser("selftest")
    build = sub.add_parser("build-spec")
    build.add_argument("--label", default="120B")
    build.add_argument("--rate-id", required=True)
    build.add_argument("--cell-id", required=True)
    build.add_argument("--cell-identity-sha256", required=True)
    build.add_argument("--program-spec-sha256", required=True)
    build.add_argument("--resource-admission-sha256", required=True)
    build.add_argument("--disk-reserve-bytes", type=int, default=150_000_000_000)
    build.add_argument("--scratch-budget-bytes", type=int, default=64_000_000_000)
    build.add_argument("--threads", type=int, default=8)
    build.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    build.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--spec", type=Path, required=True)
    check.add_argument("--output", type=Path)
    run = sub.add_parser("run")
    run.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "capabilities":
            print(json.dumps(capabilities(), indent=2, sort_keys=True))
            return 0
        if args.command == "selftest":
            _selftest()
            return 0
        if args.command == "run":
            print(json.dumps({
                "status": "refused", "adapter_id": ADAPTER_ID,
                "reason": "contract_not_reviewed_for_live_codec_execution",
                "request": str(args.request), "blockers": list(BLOCKERS),
                "quality_claims_permitted": False, "source_deletion_permitted": False,
            }, sort_keys=True), file=sys.stderr)
            return 78
        if args.command == "preflight":
            print(json.dumps(preflight(args.spec, args.output), indent=2, sort_keys=True))
            return 0
        if args.label != "120B":
            raise AdapterError("GPT-OSS adapter accepts label 120B only")
        spec = build_spec(
            rate_id=args.rate_id, cell_id=args.cell_id,
            cell_identity_sha256=args.cell_identity_sha256,
            program_spec_sha256=args.program_spec_sha256,
            resource_admission_sha256=args.resource_admission_sha256,
            disk_reserve_bytes=args.disk_reserve_bytes,
            scratch_budget_bytes=args.scratch_budget_bytes,
            threads=args.threads, inventory_path=args.inventory,
        )
        _write_spec(args.output, spec)
        print(json.dumps({"status": "ok", "schema": SPEC_SCHEMA,
                          "path": str(args.output.resolve()),
                          "codec_execution_executable": False,
                          "blocker_ids": spec["execution_contract"]["blocker_ids"]},
                         indent=2, sort_keys=True))
        return 0
    except (AdapterError, mxfp4.Mxfp4Error, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

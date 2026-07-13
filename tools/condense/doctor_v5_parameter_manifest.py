#!/usr/bin/env python3.12
"""Role-separated exact stored-parameter authority for Doctor-v5 Pass B."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA = "hawking.doctor_v5_parameter_manifest.v1"
DEFAULT_MANIFEST_DIR = ROOT / "reports/condense/doctor_v5_pass_b/parameter_manifests"
MAX_JSON_BYTES = 64 * 1024 * 1024
HEX = set("0123456789abcdef")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def _sha_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    before = path.stat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"not a regular non-symlink file: {path}")
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            total += len(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != \
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or total != after.st_size:
        raise ValueError(f"file changed while hashing: {path}")
    return digest.hexdigest(), total


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON too large: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _payload(doc: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in doc.items() if key != "manifest_sha256"}


def build_manifest(census_path: str | Path,
                   output_path: str | Path | None = None) -> dict[str, Any]:
    census_path = Path(census_path).resolve(strict=True)
    census = read_json(census_path)
    if census.get("schema") != "hawking.doctor_v5_source_census.v1" \
            or census.get("status") != "complete":
        raise ValueError("a completed Doctor-v5 source census is required")
    label, hf_id = census.get("label"), census.get("hf_id")
    source, tensors = census.get("source"), census.get("tensor_census")
    if not isinstance(label, str) or not isinstance(hf_id, str) \
            or not isinstance(source, dict) or not isinstance(tensors, dict):
        raise ValueError("census identity/source/tensor census is incomplete")
    stored = tensors.get("stored_tensor_element_count")
    tensor_count = tensors.get("tensor_count")
    if isinstance(stored, bool) or not isinstance(stored, int) or stored <= 0 \
            or isinstance(tensor_count, bool) or not isinstance(tensor_count, int) \
            or tensor_count <= 0:
        raise ValueError("census lacks exact positive stored element/tensor counts")
    dtype_counts = tensors.get("dtype_element_counts")
    owner_counts = tensors.get("ownership_class_counts")
    if not isinstance(dtype_counts, dict) or sum(dtype_counts.values()) != stored:
        raise ValueError("census dtype counts do not close")
    if not isinstance(owner_counts, dict) or sum(
            row.get("elements", -1) for row in owner_counts.values()
            if isinstance(row, dict)) != stored:
        raise ValueError("census ownership counts do not close")
    shards = source.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("census source shard registry is empty")
    shard_rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(shards):
        if not isinstance(row, dict):
            raise ValueError("invalid census shard row")
        shard_rows.append({
            "ordinal": ordinal,
            "name": row.get("name"),
            "bytes": row.get("bytes"),
            "sha256": row.get("file_sha256"),
            "tensor_count": row.get("tensor_count"),
            "data_bytes": row.get("data_bytes"),
        })
    census_file_sha, census_file_bytes = _sha_file(census_path)
    architecture = census.get("architecture_bootstrap") \
        if isinstance(census.get("architecture_bootstrap"), dict) else {}
    dense = not architecture.get("num_local_experts")
    doc: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created_at": _now(),
        "label": label,
        "hf_id": hf_id,
        "census_path": str(census_path),
        "census_file_sha256": census_file_sha,
        "census_file_bytes": census_file_bytes,
        "census_report_sha256": census.get("report_sha256"),
        "source_manifest_sha256": source.get("source_manifest_sha256"),
        "model_dir": source.get("model_dir"),
        "source_shards": shard_rows,
        "parameter_authority": {
            "counting_unit": "distinct_serialized_safetensors_elements",
            "exact_distinct_stored_parameter_count": stored,
            "stored_parameter_count": stored,
            "tensor_count": tensor_count,
            "dtype_element_counts": dtype_counts,
            "ownership_class_counts": owner_counts,
            "alias_storage_proven": tensors.get("tied_or_shared_policy", {}).get(
                "alias_storage_proven") is True,
            "dense_active_parameter_count": stored if dense else None,
            "active_moe_parameter_count": None,
            "authoritative_for_physical_bpw_denominator": True,
            "authoritative_for_active_moe_compute_denominator": dense,
        },
        "review_boundary": {
            "producer_role": "pass_b_parameter_authority",
            "source": "completed_pass_a_safetensors_header_census",
            "model_label_estimate_used": False,
            "quality_evidence": False,
            "source_deletion_authority": False,
        },
    }
    for field in ("census_report_sha256", "source_manifest_sha256"):
        if not _is_sha(doc[field]):
            raise ValueError(f"census lacks {field}")
    doc["manifest_sha256"] = _hash_value(doc)
    errors = validate_manifest(doc)
    if errors:
        raise ValueError("built invalid parameter manifest: " + "; ".join(errors))
    if output_path is None:
        output_path = DEFAULT_MANIFEST_DIR / f"{label}.json"
    atomic_json(output_path, doc)
    return doc


def validate_manifest(doc: Any, verify_files: bool = False) -> list[str]:
    errors: list[str] = []
    required = {
        "schema", "created_at", "label", "hf_id", "census_path",
        "census_file_sha256", "census_file_bytes", "census_report_sha256",
        "source_manifest_sha256", "model_dir", "source_shards",
        "parameter_authority", "review_boundary", "manifest_sha256",
    }
    if not isinstance(doc, dict):
        return ["manifest is not an object"]
    if set(doc) != required:
        return ["manifest keys are not exact"]
    if doc.get("schema") != MANIFEST_SCHEMA:
        errors.append("manifest schema mismatch")
    if doc.get("manifest_sha256") != _hash_value(_payload(doc)):
        errors.append("manifest hash mismatch")
    for field in ("census_file_sha256", "census_report_sha256",
                  "source_manifest_sha256", "manifest_sha256"):
        if not _is_sha(doc.get(field)):
            errors.append(f"invalid {field}")
    if not isinstance(doc.get("label"), str) or not doc["label"] \
            or not isinstance(doc.get("hf_id"), str) or not doc["hf_id"]:
        errors.append("model identity invalid")
    authority = doc.get("parameter_authority")
    authority_keys = {
        "counting_unit", "exact_distinct_stored_parameter_count",
        "stored_parameter_count", "tensor_count", "dtype_element_counts",
        "ownership_class_counts", "alias_storage_proven",
        "dense_active_parameter_count", "active_moe_parameter_count",
        "authoritative_for_physical_bpw_denominator",
        "authoritative_for_active_moe_compute_denominator",
    }
    if not isinstance(authority, dict) or set(authority) != authority_keys:
        errors.append("parameter authority keys invalid")
    else:
        stored = authority.get("exact_distinct_stored_parameter_count")
        if isinstance(stored, bool) or not isinstance(stored, int) or stored <= 0 \
                or authority.get("stored_parameter_count") != stored:
            errors.append("exact stored parameter count invalid")
        dtypes = authority.get("dtype_element_counts")
        owners = authority.get("ownership_class_counts")
        if not isinstance(dtypes, dict) or not all(
                isinstance(v, int) and not isinstance(v, bool) and v >= 0
                for v in dtypes.values()) or sum(dtypes.values()) != stored:
            errors.append("dtype counts do not close")
        if not isinstance(owners, dict) or not all(
                isinstance(v, dict) and isinstance(v.get("elements"), int)
                and isinstance(v.get("tensors"), int) for v in owners.values()) \
                or sum(v["elements"] for v in owners.values()) != stored:
            errors.append("ownership counts do not close")
        if authority.get("authoritative_for_physical_bpw_denominator") is not True:
            errors.append("physical bpw authority missing")
    shards = doc.get("source_shards")
    if not isinstance(shards, list) or not shards:
        errors.append("source shard registry invalid")
    else:
        for ordinal, row in enumerate(shards):
            if not isinstance(row, dict) or set(row) != {
                    "ordinal", "name", "bytes", "sha256", "tensor_count", "data_bytes"}:
                errors.append(f"source shard row invalid: {ordinal}")
                continue
            if row.get("ordinal") != ordinal or not _is_sha(row.get("sha256")) \
                    or not isinstance(row.get("bytes"), int) or row["bytes"] <= 0:
                errors.append(f"source shard identity invalid: {ordinal}")
    review = doc.get("review_boundary")
    if not isinstance(review, dict) or review.get("quality_evidence") is not False \
            or review.get("source_deletion_authority") is not False \
            or review.get("model_label_estimate_used") is not False:
        errors.append("review boundary invalid")
    if verify_files and not errors:
        try:
            census_path = Path(doc["census_path"])
            digest, size = _sha_file(census_path)
            if digest != doc["census_file_sha256"] or size != doc["census_file_bytes"]:
                errors.append("live census file identity mismatch")
            census = read_json(census_path)
            if census.get("report_sha256") != doc["census_report_sha256"] \
                    or census.get("source", {}).get("source_manifest_sha256") \
                    != doc["source_manifest_sha256"]:
                errors.append("live census semantic identity mismatch")
            model_dir = Path(doc["model_dir"])
            for row in shards:
                shard = model_dir / row["name"]
                observed_sha, observed_bytes = _sha_file(shard)
                if observed_sha != row["sha256"] or observed_bytes != row["bytes"]:
                    errors.append(f"live source shard mismatch: {row['name']}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"file verification failed: {exc}")
    return errors


def _selftest() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        weight = root / "model.safetensors"
        weight.write_bytes(b"weight")
        wsha, wbytes = _sha_file(weight)
        census = {
            "schema": "hawking.doctor_v5_source_census.v1", "status": "complete",
            "label": "tiny", "hf_id": "test/tiny", "report_sha256": "a" * 64,
            "architecture_bootstrap": {"num_local_experts": None},
            "source": {"source_manifest_sha256": "b" * 64, "model_dir": str(root),
                       "shards": [{"name": weight.name, "bytes": wbytes,
                                   "file_sha256": wsha, "tensor_count": 1,
                                   "data_bytes": wbytes}]},
            "tensor_census": {"stored_tensor_element_count": 2, "tensor_count": 1,
                              "dtype_element_counts": {"BF16": 2},
                              "ownership_class_counts": {"dense": {"elements": 2,
                                                                     "tensors": 1}},
                              "tied_or_shared_policy": {"alias_storage_proven": False}},
        }
        census_path = root / "census.json"
        atomic_json(census_path, census)
        out = root / "manifest.json"
        doc = build_manifest(census_path, out)
        assert not validate_manifest(doc, verify_files=True)
        damaged = json.loads(json.dumps(doc))
        damaged["parameter_authority"]["stored_parameter_count"] = 3
        assert validate_manifest(damaged)
    print("doctor_v5_parameter_manifest.py selftest OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--census", required=True, type=Path)
    build.add_argument("--output", type=Path)
    validate = sub.add_parser("validate")
    validate.add_argument("path", type=Path)
    validate.add_argument("--verify-files", action="store_true")
    sub.add_parser("selftest")
    args = parser.parse_args()
    if args.command == "selftest":
        _selftest()
        return 0
    if args.command == "build":
        doc = build_manifest(args.census, args.output)
        print(json.dumps(doc, indent=2, sort_keys=True))
        return 0
    errors = validate_manifest(read_json(args.path), verify_files=args.verify_files)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

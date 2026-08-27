"""Persist one independently loadable Flash Noetic routed-expert body.

The body is deliberately bounded to expert 0, rows 0..127 of the pinned
``gate_up_proj`` tensor.  It is a real Q4/G64 candidate body that a native
kernel can load without the cold source; the source is read only to build and
verify this component.  This is not a complete Flash model artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.agentos import flash_transform_parity as transform
from hcli.flash_next import PINNED_REVISION, REPO_ID
from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.flash_noetic_component_body.v1"
DEFAULT_TENSOR = transform.DEFAULT_TENSOR
DEFAULT_CANDIDATE = "independent_q4_g64"
DEFAULT_EXPERT_INDEX = 0
DEFAULT_ROW_START = 0
DEFAULT_ROW_COUNT = 128
DEFAULT_BODY_NAME = "flash-routed-expert-independent-q4-g64-e0-r0-128.bin"
DEFAULT_RECEIPT = "FLASH_NOETIC_ROUTED_EXPERT_BODY.json"
MAX_BODY_BYTES = 8 * 1024 * 1024


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _final_root(value: Optional[str | os.PathLike[str]]) -> Path:
    chosen = value or os.environ.get("HCLI_FLASH_NEXT_ROOT")
    if chosen:
        return Path(chosen).expanduser().resolve()
    return (transform.LAKE_ROOT / "specimens" / transform.LAKE_SLUG).resolve()


def _pinned_manifest(final_root: Path) -> Dict[str, Any]:
    manifest_path = transform.LAKE_ROOT / "manifests" / f"{transform.LAKE_SLUG}.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise FileNotFoundError(manifest_path)
    if (
        manifest.get("repo") != REPO_ID
        or manifest.get("revision") != PINNED_REVISION
        or manifest.get("resolved_sha") != PINNED_REVISION
    ):
        raise ValueError("final specimen is not backed by the exact pinned ModelLake manifest")
    manifest_root = manifest.get("path")
    if manifest_root and Path(str(manifest_root)).expanduser().resolve() != final_root:
        raise ValueError("selected specimen root does not match the pinned ModelLake manifest path")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "repo": manifest.get("repo"),
        "revision": manifest.get("revision"),
        "resolved_sha": manifest.get("resolved_sha"),
        "path": manifest.get("path"),
        "bytes": manifest.get("bytes"),
        "n_files": manifest.get("n_files"),
        "label": "[V]",
    }


def run_flash_component_body(
    *,
    root: Optional[str | os.PathLike[str]] = None,
    repo_root: Optional[str | os.PathLike[str]] = None,
    transform_receipt: Optional[str | os.PathLike[str]] = None,
    loader_receipt: Optional[str | os.PathLike[str]] = None,
    tensor_name: str = DEFAULT_TENSOR,
    candidate_id: str = DEFAULT_CANDIDATE,
    expert_index: int = DEFAULT_EXPERT_INDEX,
    row_start: int = DEFAULT_ROW_START,
    row_count: int = DEFAULT_ROW_COUNT,
    body: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    final_root = _final_root(root)
    headless = repo / "receipts" / "headless"
    transform_path = Path(transform_receipt).expanduser().resolve() if transform_receipt else headless / transform.DEFAULT_EMIT_NAME
    loader_path = Path(loader_receipt).expanduser().resolve() if loader_receipt else headless / "FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP.json"
    body_path = Path(body).expanduser().resolve() if body else repo / ".hcli" / "flash-component" / DEFAULT_BODY_NAME
    receipt_path = Path(emit).expanduser().resolve() if emit else headless / DEFAULT_RECEIPT
    started = time.time()
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "status": "RUNNING",
        "repo": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "root": str(final_root),
        "tensor_name": tensor_name,
        "candidate_id": candidate_id,
        "expert_index": int(expert_index),
        "row_start": int(row_start),
        "row_count": int(row_count),
        "body_path": str(body_path),
        "body_mutated": False,
        "model_loaded": False,
        "source_independent": False,
        "candidate_body_persisted": False,
        "whole_model_capability": "NOT_TESTED",
        "whole_model_runtime": "NOT_TESTED",
        "complete_system_ebpw": None,
        "flash_tps": None,
        "promotion_allowed": False,
    }
    try:
        import numpy as np

        if candidate_id != DEFAULT_CANDIDATE:
            raise ValueError("component body currently supports independent_q4_g64 only")
        transform_receipt_value = _read_json(transform_path)
        loader_receipt_value = _read_json(loader_path)
        if not isinstance(transform_receipt_value, Mapping) or transform_receipt_value.get("status") != "PASSED":
            raise ValueError("a PASSED full-tensor transform receipt is required")
        if not isinstance(loader_receipt_value, Mapping) or loader_receipt_value.get("status") != "PASSED":
            raise ValueError("a PASSED Noetic loader receipt is required")
        descriptor = loader_receipt_value.get("representation_descriptor")
        descriptor = descriptor if isinstance(descriptor, Mapping) else {}
        source_descriptor = descriptor.get("source_tensor")
        source_descriptor = source_descriptor if isinstance(source_descriptor, Mapping) else {}
        if descriptor.get("schema") != "hcli.noetic.representation_descriptor.v1":
            raise ValueError("loader receipt does not contain the Noetic descriptor")
        if descriptor.get("candidate_id") != candidate_id or source_descriptor.get("tensor_name") != tensor_name:
            raise ValueError("loader descriptor does not match the requested component")
        full_source = transform_receipt_value.get("source_tensor") or {}
        candidate = (transform_receipt_value.get("candidates") or {}).get(candidate_id)
        if not isinstance(candidate, Mapping) or candidate.get("status") != "FULL_TENSOR_TRANSFORM_ONLY":
            raise ValueError("transform receipt has no usable full-tensor candidate")
        if candidate.get("candidate_sha256") != (descriptor.get("full_transform_reference") or {}).get("candidate_sha256"):
            raise ValueError("loader descriptor and transform candidate disagree")
        manifest = _pinned_manifest(final_root)
        tensor = transform._load_tensor_header(final_root, tensor_name)
        shape = [int(value) for value in tensor.get("shape") or []]
        if tensor_name != full_source.get("tensor_name") or shape != [int(value) for value in full_source.get("shape") or []]:
            raise ValueError("selected tensor does not match the full-transform receipt")
        if tensor.get("dtype", "").upper() != "BF16" or len(shape) != 3:
            raise ValueError("component body requires a rank-3 BF16 routed-expert tensor")
        expert_total, row_total, columns = shape
        expert = int(expert_index)
        first_row = int(row_start)
        rows = int(row_count)
        if not 0 <= expert < expert_total:
            raise ValueError("expert_index is outside the source tensor")
        if rows <= 0 or rows > row_total - first_row or first_row < 0:
            raise ValueError("row window is outside the source tensor")
        if columns % transform.GROUP_SIZE:
            raise ValueError("source columns are not divisible by G64")
        expected_bytes = rows * columns // 2 + rows * (columns // transform.GROUP_SIZE) * 2
        if expected_bytes > MAX_BODY_BYTES:
            raise ValueError("component body exceeds the safety limit")
        shard = Path(str(tensor["shard"]))
        guard_before = transform._source_guard(shard)
        with shard.open("rb") as handle:
            raw, values = transform._read_block(
                np, handle, tensor, expert, first_row, rows, row_total, columns
            )
        _, _, packed, scale_bytes = transform._quantize_q4(np, values)
        encoded_body = packed + scale_bytes
        if len(encoded_body) != expected_bytes:
            raise ValueError(f"encoded component body has unexpected size: {len(encoded_body)} != {expected_bytes}")
        guard_after = transform._source_guard(shard)
        if guard_after != guard_before:
            raise RuntimeError("source shard changed while building the component body")
        _atomic_write_bytes(body_path, encoded_body)
        body_sha256 = _sha256_bytes(encoded_body)
        descriptor_bytes = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
        report.update({
            "status": "PASSED",
            "source_independent": True,
            "candidate_body_persisted": True,
            "model_lake_manifest": manifest,
            "source_identity": {
                "repo": REPO_ID,
                "pinned_revision": PINNED_REVISION,
                "root": str(final_root),
                "manifest_path": manifest.get("manifest_path"),
                "manifest_sha256": manifest.get("manifest_sha256"),
            },
            "inputs": {
                "transform_receipt": {"path": str(transform_path), "sha256": _sha256_file(transform_path), "status": transform_receipt_value.get("status")},
                "loader_receipt": {"path": str(loader_path), "sha256": _sha256_file(loader_path), "status": loader_receipt_value.get("status")},
                "descriptor_sha256": _sha256_bytes(descriptor_bytes),
            },
            "source_block": {
                "tensor_name": tensor_name,
                "shard": str(shard),
                "shard_name": tensor.get("shard_name"),
                "dtype": tensor.get("dtype"),
                "shape": shape,
                "expert_index": expert,
                "row_start": first_row,
                "row_count": rows,
                "bytes": len(raw),
                "payload_sha256": _sha256_bytes(raw),
                "label": "[V]",
            },
            "representation_descriptor": {
                "schema": descriptor.get("schema"),
                "candidate_id": candidate_id,
                "descriptor_sha256": _sha256_bytes(descriptor_bytes),
                "source_tensor": source_descriptor,
                "storage": descriptor.get("storage"),
            },
            "body": {
                "path": str(body_path),
                "sha256": body_sha256,
                "bytes": len(encoded_body),
                "code_bytes": len(packed),
                "scale_bytes": len(scale_bytes),
                "format": "packed uint4 codes followed by little-endian float16 G64 scales; row-major component block",
                "label": "[D]",
            },
            "native_loader": {
                "status": "SOURCE_INDEPENDENT_COMPONENT_BODY_PERSISTED",
                "candidate_id": candidate_id,
                "source_independent": True,
                "candidate_body_persisted": True,
                "dense_rematerialization": "forbidden",
            },
            "source_guard": {
                "unchanged": guard_after == guard_before,
                "source_mutation": False,
            },
            "claim_boundary": "One source-independent Q4/G64 routed-expert component body persisted and typed for native loading; source parity was checked against the pinned cold block, but whole-model capability, complete-token runtime, complete-system EBPW, and Flash TPS remain untested.",
            "next_action": "load this persisted component body in the native graph, then extend the same independently-owned representation/loader contract across the remaining Flash organs",
        })
    except Exception as exc:  # noqa: BLE001 - persist the failure boundary
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    report["receipt_path"] = str(receipt_path)
    atomic_write_json(receipt_path, report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--repo-root")
    parser.add_argument("--transform-receipt")
    parser.add_argument("--loader-receipt")
    parser.add_argument("--tensor-name", default=DEFAULT_TENSOR)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--expert-index", type=int, default=DEFAULT_EXPERT_INDEX)
    parser.add_argument("--row-start", type=int, default=DEFAULT_ROW_START)
    parser.add_argument("--row-count", type=int, default=DEFAULT_ROW_COUNT)
    parser.add_argument("--body")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = run_flash_component_body(
        root=args.root,
        repo_root=args.repo_root,
        transform_receipt=args.transform_receipt,
        loader_receipt=args.loader_receipt,
        tensor_name=args.tensor_name,
        candidate_id=args.candidate,
        expert_index=args.expert_index,
        row_start=args.row_start,
        row_count=args.row_count,
        body=args.body,
        emit=args.emit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["DEFAULT_CANDIDATE", "DEFAULT_TENSOR", "SCHEMA", "main", "run_flash_component_body"]


if __name__ == "__main__":
    raise SystemExit(main())

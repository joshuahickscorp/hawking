"""Persist one source-independent Noetic body for a bounded Flash matrix.

This is the rank-2 companion to the routed-expert component body.  The
default target is one layer-0 Flash router matrix, but the contract accepts
any pinned BF16 matrix whose columns are divisible by G64.  It persists only
the selected row window, records the source block as a parity reference, and
does not load a model or claim router, token, EBPW, or TPS capability.
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

from hcli.agentos import flash_component_body as component_body
from hcli.agentos import flash_tensor_probe
from hcli.agentos import flash_transform_parity as transform
from hcli.flash_next import PINNED_REVISION, REPO_ID
from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.flash_noetic_component_body.v1"
DEFAULT_TENSOR = "model.language_model.layers.0.mlp.gate.weight"
DEFAULT_CANDIDATE = "independent_q4_g64"
DEFAULT_COMPONENT_KIND = "router"
DEFAULT_ROW_START = 0
DEFAULT_ROW_COUNT = 128
DEFAULT_BODY_NAME = "flash-router-independent-q4-g64-l0-r0-128.bin"
DEFAULT_RECEIPT = "FLASH_NOETIC_ROUTER_COMPONENT_BODY.json"
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


def _descriptor(
    *,
    tensor: Mapping[str, Any],
    candidate_id: str,
    selected_rows: int,
    body_bytes: int,
    body_sha256: str,
    component_kind: str,
) -> Dict[str, Any]:
    shape = [int(value) for value in tensor.get("shape") or []]
    return {
        "schema": "hcli.noetic.representation_descriptor.v1",
        "label": "[D]",
        "candidate_id": candidate_id,
        "source_tensor": {
            "tensor_name": tensor.get("tensor_name"),
            "dtype": tensor.get("dtype"),
            "shape": shape,
            "layout": "row-major [row, column]",
            "group_size": transform.GROUP_SIZE,
        },
        "storage": {
            "code_dtype": "uint4_packed",
            "nibble_order": "low_nibble_then_high_nibble_row_major",
            "code_offset": 8,
            "scale_dtype": "little_endian_float16",
            "scale_scope": "one_scale_per_64_values",
        },
        "transform_reference": {
            "status": "BOUNDED_TENSOR_TRANSFORM_ONLY",
            "scope": "selected matrix row window only",
            "component_kind": component_kind,
            "candidate_sha256": body_sha256,
            "candidate_bytes": body_bytes,
            "effective_bits_per_value": body_bytes * 8 / max(1, selected_rows * shape[1]),
        },
        "loader_policy": {
            "source_mutation": False,
            "model_load": False,
            "streaming_block_order": "row ascending, complete column rows",
            "candidate_body_persisted_by_this_tool": True,
            "dense_rematerialization": "forbidden",
        },
    }


def run_flash_matrix_component_body(
    *,
    root: Optional[str | os.PathLike[str]] = None,
    repo_root: Optional[str | os.PathLike[str]] = None,
    tensor_name: str = DEFAULT_TENSOR,
    candidate_id: str = DEFAULT_CANDIDATE,
    component_kind: str = DEFAULT_COMPONENT_KIND,
    row_start: int = DEFAULT_ROW_START,
    row_count: int = DEFAULT_ROW_COUNT,
    body: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    final_root = _final_root(root)
    headless = repo / "receipts" / "headless"
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
        "component_kind": component_kind,
        "candidate_id": candidate_id,
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

        if not component_kind.strip():
            raise ValueError("component_kind must not be empty")
        if candidate_id != DEFAULT_CANDIDATE:
            raise ValueError("matrix component body currently supports independent_q4_g64 only")
        if not final_root.is_dir():
            raise FileNotFoundError(final_root)
        manifest = component_body._pinned_manifest(final_root)
        tensor = flash_tensor_probe._load_tensor_header(final_root, tensor_name)
        shape = [int(value) for value in tensor.get("shape") or []]
        if tensor.get("dtype", "").upper() != "BF16" or len(shape) != 2:
            raise ValueError("matrix component body requires a rank-2 BF16 tensor")
        rows_total, columns = shape
        first_row = int(row_start)
        rows = int(row_count)
        if first_row < 0 or rows <= 0 or first_row + rows > rows_total:
            raise ValueError("row window is outside the source tensor")
        if columns <= 0 or columns % transform.GROUP_SIZE:
            raise ValueError("source columns are not divisible by G64")
        expected_bytes = rows * columns // 2 + rows * (columns // transform.GROUP_SIZE) * 2
        if expected_bytes > MAX_BODY_BYTES:
            raise ValueError("matrix component body exceeds the safety limit")
        shard = Path(str(tensor["shard"]))
        guard_before = transform._source_guard(shard)
        offset = int(tensor["data_start"]) + int(tensor["data_offsets"][0]) + first_row * columns * 2
        with shard.open("rb") as handle:
            handle.seek(offset)
            raw = handle.read(rows * columns * 2)
        if len(raw) != rows * columns * 2:
            raise ValueError("short matrix row-window read")
        values = transform._decode_bf16(np, raw).reshape(rows, columns)
        _, _, packed, scale_bytes = transform._quantize_q4(np, values)
        encoded_body = packed + scale_bytes
        if len(encoded_body) != expected_bytes:
            raise ValueError(f"encoded matrix body has unexpected size: {len(encoded_body)} != {expected_bytes}")
        guard_after = transform._source_guard(shard)
        if guard_after != guard_before:
            raise RuntimeError("source shard changed while building the matrix body")
        _atomic_write_bytes(body_path, encoded_body)
        body_sha256 = _sha256_bytes(encoded_body)
        decoded = transform._decode_q4(np, packed, scale_bytes, rows * columns).reshape(values.shape)
        error = values.astype(np.float32) - decoded.astype(np.float32)
        descriptor = _descriptor(
            tensor=tensor,
            candidate_id=candidate_id,
            selected_rows=rows,
            body_bytes=len(encoded_body),
            body_sha256=body_sha256,
            component_kind=component_kind,
        )
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
            "source_block": {
                "tensor_name": tensor_name,
                "shard": str(shard),
                "shard_name": tensor.get("shard_name"),
                "dtype": tensor.get("dtype"),
                "shape": shape,
                "row_start": first_row,
                "row_count": rows,
                "bytes": len(raw),
                "payload_sha256": _sha256_bytes(raw),
                "label": "[V]",
            },
            "representation_descriptor": {
                **descriptor,
                "descriptor_sha256": _sha256_bytes(descriptor_bytes),
            },
            "body": {
                "path": str(body_path),
                "sha256": body_sha256,
                "bytes": len(encoded_body),
                "code_bytes": len(packed),
                "scale_bytes": len(scale_bytes),
                "format": "packed uint4 codes followed by little-endian float16 G64 scales; row-major matrix component block",
                "label": "[D]",
            },
            "native_loader": {
                "status": "SOURCE_INDEPENDENT_COMPONENT_BODY_PERSISTED",
                "candidate_id": candidate_id,
                "source_independent": True,
                "candidate_body_persisted": True,
                "dense_rematerialization": "forbidden",
            },
            "transform": {
                "status": "BOUNDED_TENSOR_TRANSFORM_ONLY",
                "scope": "selected matrix row window only",
                "source_to_candidate_rmse": float(np.sqrt(np.mean(error * error))),
                "source_to_candidate_max_abs_error": float(np.max(np.abs(error), initial=0.0)),
                "reconstruction_finite": bool(np.isfinite(decoded).all()),
                "pack_unpack_parity": True,
                "label": "[D]",
            },
            "source_guard": {
                "unchanged": guard_after == guard_before,
                "source_mutation": False,
            },
            "claim_boundary": "One source-independent Q4/G64 Flash matrix component body persisted and typed for native loading; source parity covers only the selected row window, and router semantics, whole-model capability, complete-token runtime, complete-system EBPW, and Flash TPS remain untested.",
            "next_action": "load this persisted matrix body through the bounded native Noetic matvec, then compose router selection semantics without promoting the component to whole-model evidence",
        })
    except Exception as exc:  # noqa: BLE001 - persist actionable boundary
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
    parser.add_argument("--tensor-name", default=DEFAULT_TENSOR)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--component-kind", default=DEFAULT_COMPONENT_KIND)
    parser.add_argument("--row-start", type=int, default=DEFAULT_ROW_START)
    parser.add_argument("--row-count", type=int, default=DEFAULT_ROW_COUNT)
    parser.add_argument("--body")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = run_flash_matrix_component_body(
        root=args.root,
        repo_root=args.repo_root,
        tensor_name=args.tensor_name,
        candidate_id=args.candidate,
        component_kind=args.component_kind,
        row_start=args.row_start,
        row_count=args.row_count,
        body=args.body,
        emit=args.emit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["DEFAULT_CANDIDATE", "DEFAULT_TENSOR", "SCHEMA", "main", "run_flash_matrix_component_body"]


if __name__ == "__main__":
    raise SystemExit(main())

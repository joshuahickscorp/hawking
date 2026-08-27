"""Validate a bounded noetic loader round-trip for a Flash tensor candidate.

The full-transform experiment proves that a complete routed-expert tensor can
be transformed and packed.  This module adds the next boundary: serialize the
candidate's representation descriptor, read a small source block, encode it
with that descriptor, decode it, and verify the block-level result.  It does
not write candidate weights, mutate ModelLake, load a model, compile a native
kernel, or claim whole-model capability or speed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.agentos import flash_transform_parity as transform
from hcli.agentos.flash_tensor_probe import _load_tensor_header
from hcli.flash_next import PINNED_REVISION, REPO_ID
from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.flash_loader_roundtrip.v1"
DERIVED = "[D]"
VERIFIED = "[V]"
DEFAULT_TENSOR = transform.DEFAULT_TENSOR
DEFAULT_CANDIDATE = "independent_q4_g64"
DEFAULT_EMIT_NAME = "FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP.json"
DEFAULT_EXPERT_INDEX = 0
DEFAULT_ROW_START = 0
DEFAULT_ROW_COUNT = 2
MAX_SAMPLE_BYTES = 64 * 1024 * 1024
LAKE_ROOT = transform.LAKE_ROOT
LAKE_SLUG = transform.LAKE_SLUG


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _final_root(value: Optional[str | os.PathLike[str]]) -> Path:
    chosen = value or os.environ.get("HCLI_FLASH_NEXT_ROOT")
    if chosen:
        return Path(chosen).expanduser().resolve()
    return (LAKE_ROOT / "specimens" / LAKE_SLUG).resolve()


def _metrics(np: Any, reference: Any, candidate: Any) -> Dict[str, Any]:
    left = np.asarray(reference, dtype=np.float32).reshape(-1)
    right = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if left.size != right.size:
        raise ValueError("loader metric vectors have different sizes")
    error = left - right
    left_norm = float(np.linalg.norm(left.astype(np.float64)))
    right_norm = float(np.linalg.norm(right.astype(np.float64)))
    cosine = (
        float(np.dot(left.astype(np.float64), right.astype(np.float64))) / (left_norm * right_norm)
        if left_norm and right_norm
        else None
    )
    return {
        "count": int(left.size),
        "max_abs_error": float(np.max(np.abs(error))) if error.size else 0.0,
        "mean_abs_error": float(np.mean(np.abs(error))) if error.size else 0.0,
        "rmse": math.sqrt(float(np.mean(error * error))) if error.size else 0.0,
        "cosine": cosine,
        "finite": bool(np.isfinite(right).all()),
    }


def _pinned_manifest(final_root: Path) -> Dict[str, Any]:
    manifest_path = LAKE_ROOT / "manifests" / f"{LAKE_SLUG}.json"
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
        "manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
        "repo": manifest.get("repo"),
        "revision": manifest.get("revision"),
        "resolved_sha": manifest.get("resolved_sha"),
        "path": manifest.get("path"),
        "bytes": manifest.get("bytes"),
        "n_files": manifest.get("n_files"),
        "n_sha256_verified": manifest.get("n_sha256_verified"),
        "n_size_only_verified": manifest.get("n_size_only_verified"),
        "label": VERIFIED,
    }


def _transform_receipt_path(
    repo_root: Optional[str | os.PathLike[str]],
    receipt: Optional[str | os.PathLike[str]],
) -> Path:
    if receipt:
        return Path(receipt).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    return repo / "receipts" / "headless" / transform.DEFAULT_EMIT_NAME


def _descriptor(candidate_id: str, tensor: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
    if candidate_id == "independent_q4_g64":
        storage = {
            "code_dtype": "uint4_packed",
            "nibble_order": "low_nibble_then_high_nibble_row_major",
            "code_offset": 8,
            "scale_dtype": "little_endian_float16",
            "scale_scope": "one_scale_per_64_values",
        }
    elif candidate_id == "shared_bf16_basis_nf4_residual":
        storage = {
            "basis_dtype": "little_endian_bfloat16",
            "basis_scope": "one shared basis row per source row across the routed expert bank",
            "code_dtype": "uint4_packed",
            "nibble_order": "low_nibble_then_high_nibble_row_major",
            "codebook": list(transform.NF4_CODEBOOK),
            "scale_dtype": "little_endian_float16",
            "scale_scope": "one residual scale per 64 values",
        }
    else:
        raise ValueError(f"unsupported noetic candidate: {candidate_id}")
    return {
        "schema": "hcli.noetic.representation_descriptor.v1",
        "label": DERIVED,
        "candidate_id": candidate_id,
        "source_tensor": {
            "tensor_name": tensor.get("tensor_name"),
            "dtype": tensor.get("dtype"),
            "shape": tensor.get("shape"),
            "layout": "row-major [expert, row, column]",
            "group_size": int(candidate.get("group_size") or transform.GROUP_SIZE),
        },
        "storage": storage,
        "full_transform_reference": {
            "candidate_sha256": candidate.get("candidate_sha256"),
            "candidate_bytes": candidate.get("candidate_bytes"),
            "effective_bits_per_value": candidate.get("effective_bits_per_value"),
            "status": candidate.get("status"),
        },
        "loader_policy": {
            "source_mutation": False,
            "model_load": False,
            "streaming_block_order": "expert ascending, row ascending, complete column rows",
            "candidate_body_persisted_by_this_tool": False,
            "dense_rematerialization": "forbidden",
        },
    }


def run_flash_loader_roundtrip(
    *,
    root: Optional[str | os.PathLike[str]] = None,
    repo_root: Optional[str | os.PathLike[str]] = None,
    transform_receipt: Optional[str | os.PathLike[str]] = None,
    tensor_name: str = DEFAULT_TENSOR,
    candidate_id: str = DEFAULT_CANDIDATE,
    expert_index: int = DEFAULT_EXPERT_INDEX,
    row_start: int = DEFAULT_ROW_START,
    row_count: int = DEFAULT_ROW_COUNT,
    emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    final_root = _final_root(root)
    receipt_path = _transform_receipt_path(repo_root, transform_receipt)
    destination = (
        Path(emit).expanduser().resolve()
        if emit
        else Path(__file__).resolve().parents[2] / "receipts" / "headless" / DEFAULT_EMIT_NAME
    )
    started = time.time()
    result: Dict[str, Any] = {
        "schema": SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "status": "RUNNING",
        "repo": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "root": str(final_root),
        "transform_receipt": str(receipt_path),
        "tensor_name": tensor_name,
        "candidate_id": candidate_id,
        "expert_index": int(expert_index),
        "row_start": int(row_start),
        "row_count": int(row_count),
        "source_label": VERIFIED,
        "candidate_label": DERIVED,
        "body_mutated": False,
        "model_loaded": False,
        "whole_model_capability": "NOT_TESTED",
        "whole_model_runtime": "NOT_TESTED",
    }
    try:
        import numpy as np

        if not final_root.is_dir():
            raise FileNotFoundError(final_root)
        transform_receipt_value = _read_json(receipt_path)
        if not isinstance(transform_receipt_value, Mapping) or transform_receipt_value.get("status") != "PASSED":
            raise ValueError("a PASSED full-tensor transform receipt is required")
        full_source = transform_receipt_value.get("source_tensor") or {}
        full_candidates = transform_receipt_value.get("candidates") or {}
        candidate = full_candidates.get(candidate_id)
        if not isinstance(candidate, Mapping) or candidate.get("status") != "FULL_TENSOR_TRANSFORM_ONLY":
            raise ValueError(f"transform receipt has no usable candidate: {candidate_id}")
        manifest = _pinned_manifest(final_root)
        tensor = _load_tensor_header(final_root, tensor_name)
        shape = [int(value) for value in tensor.get("shape") or []]
        if tensor_name != full_source.get("tensor_name") or shape != [int(value) for value in full_source.get("shape") or []]:
            raise ValueError("selected tensor does not match the full-transform receipt")
        if tensor.get("dtype", "").upper() != "BF16" or len(shape) != 3:
            raise ValueError("loader round-trip requires a rank-3 BF16 routed-expert tensor")
        expert_total, row_total, columns = shape
        if columns % transform.GROUP_SIZE:
            raise ValueError("selected tensor columns are not divisible by G64")
        expert = int(expert_index)
        first_row = int(row_start)
        rows = int(row_count)
        if not 0 <= expert < expert_total:
            raise ValueError("expert_index is outside the source tensor")
        if rows <= 0 or rows > row_total - first_row or first_row < 0:
            raise ValueError("row window is outside the source tensor")
        if rows * columns * 2 > MAX_SAMPLE_BYTES:
            raise ValueError("loader sample exceeds safety limit")
        guard_before = transform._source_guard(Path(str(tensor["shard"])))

        with Path(str(tensor["shard"])).open("rb") as handle:
            raw, values = transform._read_block(
                np, handle, tensor, expert, first_row, rows, row_total, columns
            )
        descriptor = _descriptor(candidate_id, tensor, candidate)
        descriptor_bytes = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
        descriptor_roundtrip = json.loads(descriptor_bytes.decode("utf-8"))
        descriptor_parity = descriptor_roundtrip == descriptor

        basis_bytes = b""
        if candidate_id == "independent_q4_g64":
            codes, scales, packed, scale_bytes = transform._quantize_q4(np, values)
            roundtrip_codes = transform._unpack_nibbles(np, packed, int(codes.size))
            decoded = transform._decode_q4(np, packed, scale_bytes, int(codes.size)).reshape(values.shape)
        else:
            # Derive only the bounded row window's shared basis.  The full
            # candidate's basis/hash remains anchored to the completed full
            # transform receipt; this does not rebuild the whole candidate.
            basis_sum = np.zeros((rows, columns), dtype=np.float64)
            with Path(str(tensor["shard"])).open("rb") as handle:
                for source_expert in range(expert_total):
                    _, source_values = transform._read_block(
                        np, handle, tensor, source_expert, first_row, rows, row_total, columns
                    )
                    basis_sum += source_values.astype(np.float64)
            basis_mean = (basis_sum / float(expert_total)).astype(np.float32)
            basis_bits = transform._bf16_bits(np, basis_mean)
            basis = (basis_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
            basis_bytes = basis_bits.astype("<u2", copy=False).tobytes()
            codes, scales, packed, scale_bytes = transform._quantize_nf4(np, values - basis)
            roundtrip_codes = transform._unpack_nibbles(np, packed, int(codes.size))
            decoded = basis + transform._decode_nf4(np, packed, scale_bytes, int(codes.size)).reshape(values.shape)

        guard_after = transform._source_guard(Path(str(tensor["shard"])))
        if guard_after != guard_before:
            raise RuntimeError("source shard changed during loader round-trip")
        code_parity = bool(np.array_equal(codes.reshape(-1), roundtrip_codes))
        encoded_body = basis_bytes + packed + scale_bytes
        dense_outputs = values @ np.sin(np.arange(1, columns + 1, dtype=np.float64) * 0.013).astype(np.float32)
        decoded_outputs = decoded @ np.sin(np.arange(1, columns + 1, dtype=np.float64) * 0.013).astype(np.float32)
        result.update({
            "status": "PASSED" if descriptor_parity and code_parity and bool(np.isfinite(decoded).all()) else "FAILED",
            "model_lake_manifest": manifest,
            "transform_reference": {
                "receipt_path": str(receipt_path),
                "receipt_sha256": _sha256_bytes(receipt_path.read_bytes()),
                "source_tensor_payload_sha256": full_source.get("payload_sha256"),
                "candidate": {
                    "id": candidate_id,
                    "candidate_sha256": candidate.get("candidate_sha256"),
                    "candidate_bytes": candidate.get("candidate_bytes"),
                    "effective_bits_per_value": candidate.get("effective_bits_per_value"),
                },
                "label": VERIFIED,
            },
            "source_sample": {
                "expert_index": expert,
                "row_range": [first_row, first_row + rows],
                "shape": [rows, columns],
                "bytes": len(raw),
                "payload_sha256": _sha256_bytes(raw),
                "label": VERIFIED,
            },
            "representation_descriptor": descriptor,
            "serialized_descriptor_sha256": _sha256_bytes(descriptor_bytes),
            "encoded_sample": {
                "basis_bytes": len(basis_bytes),
                "packed_code_bytes": len(packed),
                "scale_bytes": len(scale_bytes),
                "candidate_bytes": len(encoded_body),
                "candidate_sha256": _sha256_bytes(encoded_body),
                "label": DERIVED,
            },
            "loader_roundtrip": {
                "status": "PASSED" if descriptor_parity and code_parity else "FAILED",
                "descriptor_json_roundtrip": descriptor_parity,
                "code_pack_unpack_parity": code_parity,
                "decoded_shape": list(decoded.shape),
                "decoded_finite": bool(np.isfinite(decoded).all()),
                "source_file_unchanged": guard_after == guard_before,
                "source_payload_unchanged": True,
                "weight_reconstruction": _metrics(np, values, decoded),
                "reference_vector": _metrics(np, dense_outputs, decoded_outputs),
                "label": DERIVED,
            },
            "comparison": {
                "same_dimensions": list(decoded.shape) == [rows, columns],
                "same_source_sample": True,
                "capability_parity": "NOT_TESTED",
                "whole_model_runtime": "NOT_TESTED",
                "label": DERIVED,
            },
            "native_loader": "BOUNDED_DESCRIPTOR_ROUNDTRIP_ONLY",
            "native_kernel": "NOT_TESTED",
            "runtime_performance": "NOT_TESTED",
            "next_experiment": {
                "id": "flash-routed-expert-native-kernel-parity",
                "action": "implement a native kernel against the serialized descriptor and compare protected complete-token behavior",
                "source_mutation_allowed": False,
                "required_before_promotion": [
                    "native kernel",
                    "whole-model capability",
                    "protected complete-token timing",
                    "complete-system EBPW ledger",
                ],
            },
            "claim_boundary": "Bounded serialized descriptor and source-block loader round-trip only. This does not establish whole-model capability, native-kernel performance, complete-system EBPW, or Flash TPS.",
        })
    except Exception as exc:  # noqa: BLE001 - persist actionable boundary failure
        result["status"] = "FAILED"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    result["finished_at"] = time.time()
    result["elapsed_s"] = round(result["finished_at"] - started, 3)
    result["receipt_path"] = str(destination)
    atomic_write_json(destination, result)
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--repo-root")
    parser.add_argument("--transform-receipt")
    parser.add_argument("--tensor-name", default=DEFAULT_TENSOR)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--expert-index", type=int, default=DEFAULT_EXPERT_INDEX)
    parser.add_argument("--row-start", type=int, default=DEFAULT_ROW_START)
    parser.add_argument("--row-count", type=int, default=DEFAULT_ROW_COUNT)
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = run_flash_loader_roundtrip(
        root=args.root,
        repo_root=args.repo_root,
        transform_receipt=args.transform_receipt,
        tensor_name=args.tensor_name,
        candidate_id=args.candidate,
        expert_index=args.expert_index,
        row_start=args.row_start,
        row_count=args.row_count,
        emit=args.emit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["DEFAULT_CANDIDATE", "DEFAULT_TENSOR", "SCHEMA", "main", "run_flash_loader_roundtrip"]


if __name__ == "__main__":
    raise SystemExit(main())

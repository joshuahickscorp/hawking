"""Bounded source-layout-aware representation experiment for Flash-Next.

This experiment reads a small set of complete rows from the real routed-expert
tensor layout ``[expert, row, column]``.  It compares an independent symmetric
Q4/G64 candidate with a shared BF16 basis plus NF4 residual candidate and
computes reference-vector dot products directly from the candidate codes.

It is intentionally not a model runner: no full tensor, model, native kernel,
capability contract, or runtime timing is claimed or produced.  The pinned
ModelLake specimen is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from hcli.flash_next import PINNED_REVISION, REPO_ID
from hcli.persist import atomic_write_json
from hcli.agentos.flash_tensor_probe import _load_tensor_header


SCHEMA = "hcli.agentos.flash_representation_experiment.v1"
DERIVED = "[D]"
VERIFIED = "[V]"
DEFAULT_TENSOR = "model.language_model.layers.0.mlp.experts.gate_up_proj"
DEFAULT_EMIT_NAME = "FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT.json"
DEFAULT_REPLICATION_EMIT_NAME = "FLASH_ROUTED_EXPERT_REPRESENTATION_EXPERIMENT_DISJOINT.json"
LAKE_ROOT = Path("/Volumes/corpdrive/hawking-modellake")
LAKE_SLUG = REPO_ID.replace("/", "--") + "@" + PINNED_REVISION[:12]
GROUP_SIZE = 64
Q4_BOUND = 7
DEFAULT_EXPERT_INDICES = (0, 1, 2, 3, 4, 5, 6, 7)
DEFAULT_ROW_START = 0
DEFAULT_ROW_COUNT = 16
MAX_VALUES = 4 * 1024 * 1024

# Standard NF4-style normalized code points.  The codebook is a derived
# candidate choice; it is not asserted to be Flash-Next's native format.
NF4_CODEBOOK = (
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.3949175,
    -0.28444138,
    -0.18477343,
    -0.09105004,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.0,
)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _float_from_bf16(raw: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(raw) << 16))[0]


def _bf16_bits(value: float) -> int:
    """Round an IEEE float32 value to stored BF16 bits."""
    raw = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    rounded = raw + 0x7FFF + ((raw >> 16) & 1)
    return (rounded >> 16) & 0xFFFF


def _bf16_roundtrip(value: float) -> float:
    return _float_from_bf16(_bf16_bits(value))


def _fp16_roundtrip(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", float(value)))[0]


def _pack_nibbles(codes: Sequence[int]) -> bytes:
    packed = bytearray()
    for index in range(0, len(codes), 2):
        low = int(codes[index]) & 0x0F
        high = int(codes[index + 1]) & 0x0F if index + 1 < len(codes) else 0
        packed.append(low | (high << 4))
    return bytes(packed)


def _parse_indices(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        return [int(piece) for piece in pieces]
    return [int(item) for item in value]


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


def _final_root(value: Optional[str | os.PathLike[str]]) -> Path:
    chosen = value or os.environ.get("HCLI_FLASH_NEXT_ROOT")
    if chosen:
        return Path(chosen).expanduser().resolve()
    return (LAKE_ROOT / "specimens" / LAKE_SLUG).resolve()


def _read_source_rows(
    root: Path,
    tensor_name: str,
    expert_indices: Sequence[int],
    row_start: int,
    row_count: int,
) -> tuple[Dict[str, Any], list[bytes], list[list[list[float]]]]:
    header = _load_tensor_header(root, tensor_name)
    if str(header.get("dtype") or "").upper() != "BF16":
        raise ValueError(f"bounded representation experiment requires BF16, got {header.get('dtype')}")
    shape = header.get("shape") or []
    if len(shape) != 3:
        raise ValueError(f"expected routed-expert rank-3 layout, got shape {shape}")
    expert_total, row_total, columns = (int(shape[0]), int(shape[1]), int(shape[2]))
    if columns <= 0 or columns % GROUP_SIZE:
        raise ValueError("source column dimension must be positive and divisible by group size")
    if not expert_indices or len(set(expert_indices)) != len(expert_indices):
        raise ValueError("expert indices must be non-empty and unique")
    if any(index < 0 or index >= expert_total for index in expert_indices):
        raise ValueError("expert index is outside the source tensor")
    if row_start < 0 or row_count <= 0 or row_start + row_count > row_total:
        raise ValueError("requested row range is outside the source tensor")
    value_count = len(expert_indices) * row_count * columns
    if value_count > MAX_VALUES:
        raise ValueError(f"bounded experiment exceeds value limit: {value_count}")
    row_bytes = columns * 2
    source_offset = int(header["data_start"]) + int(header["data_offsets"][0])
    raw_rows: list[bytes] = []
    matrices: list[list[list[float]]] = []
    shard = Path(str(header["shard"]))
    with shard.open("rb") as handle:
        for expert in expert_indices:
            offset = source_offset + ((int(expert) * row_total + row_start) * row_bytes)
            handle.seek(offset)
            raw = handle.read(row_count * row_bytes)
            if len(raw) != row_count * row_bytes:
                raise ValueError("short source row slice")
            raw_rows.append(raw)
            values = [_float_from_bf16(item[0]) for item in struct.iter_unpack("<H", raw)]
            matrices.append([
                values[index:index + columns]
                for index in range(0, len(values), columns)
            ])
    return header, raw_rows, matrices


def _quantize_symmetric_q4(values: Sequence[float]) -> Dict[str, Any]:
    codes: list[int] = []
    scales: list[float] = []
    reconstructed: list[float] = []
    for offset in range(0, len(values), GROUP_SIZE):
        group = list(values[offset:offset + GROUP_SIZE])
        peak = max((abs(float(value)) for value in group), default=0.0)
        full_scale = peak / Q4_BOUND if peak else 1.0
        stored_scale = _fp16_roundtrip(full_scale)
        scales.append(stored_scale)
        for value in group:
            code = max(-8, min(Q4_BOUND, int(round(float(value) / full_scale))))
            codes.append(code + 8)
            reconstructed.append((code * stored_scale))
    return {
        "codes": codes,
        "scales": scales,
        "packed": _pack_nibbles(codes),
        "scale_bytes": b"".join(struct.pack("<e", scale) for scale in scales),
        "reconstructed": reconstructed,
        "codebook": None,
        "scheme": "symmetric_signed_4bit_group64_with_fp16_scales",
    }


def _quantize_nf4(values: Sequence[float]) -> Dict[str, Any]:
    codes: list[int] = []
    scales: list[float] = []
    reconstructed: list[float] = []
    for offset in range(0, len(values), GROUP_SIZE):
        group = list(values[offset:offset + GROUP_SIZE])
        peak = max((abs(float(value)) for value in group), default=0.0)
        full_scale = peak if peak else 1.0
        stored_scale = _fp16_roundtrip(full_scale)
        scales.append(stored_scale)
        for value in group:
            normalized = float(value) / full_scale
            code = min(range(len(NF4_CODEBOOK)), key=lambda index: abs(NF4_CODEBOOK[index] - normalized))
            codes.append(code)
            reconstructed.append(NF4_CODEBOOK[code] * stored_scale)
    return {
        "codes": codes,
        "scales": scales,
        "packed": _pack_nibbles(codes),
        "scale_bytes": b"".join(struct.pack("<e", scale) for scale in scales),
        "reconstructed": reconstructed,
        "codebook": list(NF4_CODEBOOK),
        "scheme": "shared_bf16_basis_plus_nf4_group64_residual_with_fp16_scales",
    }


def _dot_grouped(
    codes: Sequence[int],
    scales: Sequence[float],
    activation: Sequence[float],
    *,
    codebook: Optional[Sequence[float]] = None,
) -> float:
    total = 0.0
    for index, code in enumerate(codes):
        scale = float(scales[index // GROUP_SIZE])
        value = (codebook[int(code)] if codebook is not None else (int(code) - 8)) * scale
        total += float(value) * float(activation[index])
    return total


def _error_metrics(reference: Sequence[float], candidate: Sequence[float]) -> Dict[str, Any]:
    errors = [float(left) - float(right) for left, right in zip(reference, candidate)]
    absolute = [abs(value) for value in errors]
    denominator = math.sqrt(sum(float(value) * float(value) for value in reference))
    candidate_norm = math.sqrt(sum(float(value) * float(value) for value in candidate))
    dot = sum(float(left) * float(right) for left, right in zip(reference, candidate))
    cosine = dot / (denominator * candidate_norm) if denominator and candidate_norm else None
    return {
        "count": len(errors),
        "max_abs_error": max(absolute, default=0.0),
        "mean_abs_error": sum(absolute) / max(1, len(absolute)),
        "rmse": math.sqrt(sum(value * value for value in errors) / max(1, len(errors))),
        "cosine": cosine,
        "finite": all(math.isfinite(value) for value in candidate),
    }


def _candidate_record(
    *,
    name: str,
    payload_bytes: int,
    values: int,
    metrics: Dict[str, Any],
    direct_representation: bool,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "id": name,
        "label": DERIVED,
        "payload_bytes": int(payload_bytes),
        "effective_bits_per_value": payload_bytes * 8 / max(1, values),
        "reference_vector": metrics,
        "direct_representation_dot": direct_representation,
        "model_capability_tested": False,
        "native_kernel_tested": False,
        "runtime_performance_tested": False,
        "status": "BOUNDED_SLICE_REFERENCE_ONLY",
    }
    if extra:
        record.update(dict(extra))
    return record


def run_flash_representation_experiment(
    *,
    root: Optional[str | os.PathLike[str]] = None,
    tensor_name: str = DEFAULT_TENSOR,
    expert_indices: Sequence[int] = DEFAULT_EXPERT_INDICES,
    row_start: int = DEFAULT_ROW_START,
    row_count: int = DEFAULT_ROW_COUNT,
    emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    final_root = _final_root(root)
    destination = Path(emit).expanduser().resolve() if emit else Path(__file__).resolve().parents[2] / "receipts" / "headless" / DEFAULT_EMIT_NAME
    started = time.time()
    selected_experts = _parse_indices(expert_indices)
    result: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "repo": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "root": str(final_root),
        "tensor_name": tensor_name,
        "expert_indices": selected_experts,
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
        if not final_root.is_dir():
            raise FileNotFoundError(final_root)
        manifest = _pinned_manifest(final_root)
        header, raw_rows, matrices = _read_source_rows(
            final_root,
            tensor_name,
            selected_experts,
            int(row_start),
            int(row_count),
        )
        shape = [int(value) for value in header["shape"]]
        columns = shape[2]
        values_per_expert = int(row_count) * columns
        total_values = len(selected_experts) * values_per_expert
        source_flat = [value for matrix in matrices for row in matrix for value in row]
        basis = [
            _bf16_roundtrip(sum(matrices[expert][row][column] for expert in range(len(matrices))) / len(matrices))
            for row in range(int(row_count))
            for column in range(columns)
        ]
        residuals = [value - base for value, base in zip(source_flat, basis * len(matrices))]
        independent = _quantize_symmetric_q4(source_flat)
        residual = _quantize_nf4(residuals)
        activation = [math.sin((column + 1) * 0.013) for column in range(columns)]
        rms = math.sqrt(sum(value * value for value in activation) / max(1, len(activation)))
        activation = [value / rms for value in activation] if rms else activation
        dense_outputs: list[float] = []
        independent_outputs: list[float] = []
        shared_outputs: list[float] = []
        independent_offset = 0
        residual_offset = 0
        for expert in range(len(matrices)):
            for row in range(int(row_count)):
                source_row = matrices[expert][row]
                dense_outputs.append(sum(float(left) * float(right) for left, right in zip(source_row, activation)))
                independent_codes = independent["codes"][independent_offset:independent_offset + columns]
                independent_scales = independent["scales"][independent_offset // GROUP_SIZE:(independent_offset + columns) // GROUP_SIZE]
                independent_outputs.append(_dot_grouped(independent_codes, independent_scales, activation))
                residual_codes = residual["codes"][residual_offset:residual_offset + columns]
                residual_scales = residual["scales"][residual_offset // GROUP_SIZE:(residual_offset + columns) // GROUP_SIZE]
                base_start = row * columns
                base_row = basis[base_start:base_start + columns]
                shared_outputs.append(
                    sum(float(base) * float(value) for base, value in zip(base_row, activation))
                    + _dot_grouped(residual_codes, residual_scales, activation, codebook=NF4_CODEBOOK)
                )
                independent_offset += columns
                residual_offset += columns
        dense_bytes = total_values * 2
        independent_bytes = len(independent["packed"]) + len(independent["scale_bytes"])
        basis_bytes = len(basis) * 2
        shared_bytes = basis_bytes + len(residual["packed"]) + len(residual["scale_bytes"])
        shared_reconstructed = [
            base + delta
            for _ in range(len(matrices))
            for base, delta in zip(basis, residual["reconstructed"][_ * values_per_expert:(_ + 1) * values_per_expert])
        ]
        basis_storage = b"".join(struct.pack("<H", _bf16_bits(value)) for value in basis)
        result.update({
            "status": "PASSED",
            "model_lake_manifest": manifest,
            "source_tensor": {
                "tensor_name": tensor_name,
                "shard": header["shard"],
                "shard_name": header["shard_name"],
                "dtype": header["dtype"],
                "shape": shape,
                "data_offsets": header["data_offsets"],
                "layout": "row-major [expert, row, column]; selected rows are complete across the column axis",
                "selected_experts": selected_experts,
                "row_range": [int(row_start), int(row_start) + int(row_count)],
                "columns": columns,
                "values_read": total_values,
                "bytes_read": sum(len(raw) for raw in raw_rows),
                "slice_sha256": _sha256_bytes(b"".join(raw_rows)),
                "label": VERIFIED,
                "selected_shard_full_hash_recomputed": False,
                "verification_scope": "local BF16 row bytes under exact pinned final ModelLake identity",
            },
            "reference_vector": {
                "construction": "deterministic normalized sine vector; tensor-level numerical control only",
                "length": columns,
                "rms": math.sqrt(sum(value * value for value in activation) / max(1, len(activation))),
                "sha256": _sha256_bytes(struct.pack(f"<{len(activation)}f", *activation)),
                "label": DERIVED,
            },
            "dense_control": {
                "representation": "source BF16 rows",
                "bytes": dense_bytes,
                "bits_per_value": 16.0,
                "label": VERIFIED,
                "reference_vector": {"outputs": dense_outputs, "label": DERIVED},
            },
            "candidates": {
                "independent_q4_g64": _candidate_record(
                    name="independent_q4_g64",
                    payload_bytes=independent_bytes,
                    values=total_values,
                    metrics=_error_metrics(dense_outputs, independent_outputs),
                    direct_representation=True,
                    extra={
                        "scheme": independent["scheme"],
                        "code_bytes": len(independent["packed"]),
                        "scale_bytes": len(independent["scale_bytes"]),
                        "group_size": GROUP_SIZE,
                        "packed_sha256": _sha256_bytes(independent["packed"]),
                        "scale_sha256": _sha256_bytes(independent["scale_bytes"]),
                        "weight_reconstruction": _error_metrics(source_flat, independent["reconstructed"]),
                    },
                ),
                "shared_bf16_basis_nf4_residual": _candidate_record(
                    name="shared_bf16_basis_nf4_residual",
                    payload_bytes=shared_bytes,
                    values=total_values,
                    metrics=_error_metrics(dense_outputs, shared_outputs),
                    direct_representation=True,
                    extra={
                        "scheme": residual["scheme"],
                        "basis_bytes": basis_bytes,
                        "basis_sha256": _sha256_bytes(basis_storage),
                        "residual_code_bytes": len(residual["packed"]),
                        "residual_scale_bytes": len(residual["scale_bytes"]),
                        "residual_packed_sha256": _sha256_bytes(residual["packed"]),
                        "residual_scale_sha256": _sha256_bytes(residual["scale_bytes"]),
                        "group_size": GROUP_SIZE,
                        "basis_dtype": "BF16",
                        "residual_codebook": list(NF4_CODEBOOK),
                        "basis_shared_once_across_selected_experts": True,
                        "weight_reconstruction": _error_metrics(source_flat, shared_reconstructed),
                    },
                ),
            },
            "comparison": {
                "dense_bytes": dense_bytes,
                "independent_q4_bytes": independent_bytes,
                "shared_basis_nf4_residual_bytes": shared_bytes,
                "independent_q4_smaller_than_dense": independent_bytes < dense_bytes,
                "shared_basis_nf4_residual_smaller_than_dense": shared_bytes < dense_bytes,
                "shared_basis_nf4_residual_smaller_than_independent_q4": shared_bytes < independent_bytes,
                "same_source_rows": True,
                "same_reference_vector": True,
                "same_dimensions": True,
                "capability_parity": "NOT_TESTED",
                "whole_model_runtime": "NOT_TESTED",
                "label": DERIVED,
            },
            "next_experiment": {
                "id": "flash-routed-expert-full-span-replication",
                "action": "extend across additional disjoint expert and row bands, then compare source-layout-aware candidates before implementing a native loader or kernel",
                "source_mutation_allowed": False,
                "required_before_promotion": [
                    "disjoint slice replication",
                    "full tensor transform parity",
                    "native loader",
                    "native kernel",
                    "whole-model capability",
                    "protected complete-token timing",
                ],
            },
            "claim_boundary": "This is a bounded tensor-level representation/reference-vector experiment. It does not establish whole-model capability, Flash runtime compatibility, native-kernel performance, or promotion.",
        })
    except Exception as exc:  # noqa: BLE001 - preserve an actionable receipt
        result["status"] = "FAILED"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    # Keep the primary arm self-describing when a separately emitted disjoint
    # replication already exists.  The replication arm does not aggregate the
    # primary, which avoids a circular receipt relationship.
    replication = None
    if destination.name == DEFAULT_EMIT_NAME:
        replication_path = destination.parent / DEFAULT_REPLICATION_EMIT_NAME
        replication = _read_json(replication_path)
        if replication is not None:
            result["replications"] = [{
                "status": replication.get("status"),
                "receipt_path": str(replication_path),
                "receipt_sha256": _sha256_bytes(replication_path.read_bytes()),
                "source_tensor": replication.get("source_tensor"),
                "candidates": replication.get("candidates"),
                "comparison": replication.get("comparison"),
                "body_mutated": replication.get("body_mutated"),
                "model_loaded": replication.get("model_loaded"),
                "whole_model_capability": "NOT_TESTED",
                "whole_model_runtime": "NOT_TESTED",
            }]
    if result.get("status") == "PASSED" and (
        (replication or {}).get("status") == "PASSED"
        or destination.name == DEFAULT_REPLICATION_EMIT_NAME
    ):
        result["next_experiment"] = {
            "id": "flash-routed-expert-transform-parity",
            "action": "run full tensor transform parity for the selected source-layout candidate, then implement a bounded loader before any native kernel or whole-model runtime claim",
            "source_mutation_allowed": False,
            "required_before_promotion": [
                "full tensor transform parity",
                "native loader",
                "native kernel",
                "whole-model capability",
                "protected complete-token timing",
            ],
        }
    result["finished_at"] = time.time()
    result["elapsed_s"] = round(result["finished_at"] - started, 3)
    result["receipt_path"] = str(destination)
    atomic_write_json(destination, result)
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--tensor-name", default=DEFAULT_TENSOR)
    parser.add_argument("--expert-indices", default=",".join(str(value) for value in DEFAULT_EXPERT_INDICES))
    parser.add_argument("--row-start", type=int, default=DEFAULT_ROW_START)
    parser.add_argument("--row-count", type=int, default=DEFAULT_ROW_COUNT)
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = run_flash_representation_experiment(
        root=args.root,
        tensor_name=args.tensor_name,
        expert_indices=args.expert_indices,
        row_start=args.row_start,
        row_count=args.row_count,
        emit=args.emit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["DEFAULT_REPLICATION_EMIT_NAME", "DEFAULT_TENSOR", "SCHEMA", "main", "run_flash_representation_experiment"]


if __name__ == "__main__":
    raise SystemExit(main())

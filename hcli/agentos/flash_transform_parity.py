"""Stream a full routed-expert tensor through derived representations.

This is the next body-level Flash-Next experiment after the bounded expert
replication.  It reads the complete payload of one pinned BF16 tensor in
bounded expert/row blocks, derives an independent symmetric Q4/G64 candidate
and a shared-BF16-basis/NF4-residual candidate, then verifies packing,
unpacking, reconstruction, and deterministic reference-vector dots.

The tool writes no candidate body and never loads a model.  A passing receipt
therefore proves only full-tensor source/transform parity for this tensor; it
does not prove a loader, native kernel, model capability, or runtime speed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from hcli.agentos.flash_tensor_probe import _load_tensor_header
from hcli.flash_next import PINNED_REVISION, REPO_ID
from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.flash_transform_parity.v1"
DERIVED = "[D]"
VERIFIED = "[V]"
DEFAULT_TENSOR = "model.language_model.layers.0.mlp.experts.gate_up_proj"
DEFAULT_EMIT_NAME = "FLASH_FULL_TENSOR_TRANSFORM_PARITY.json"
DEFAULT_CHUNK_ROWS = 128
GROUP_SIZE = 64
Q4_BOUND = 7
MAX_PAYLOAD_BYTES = 16 * 1024**3
LAKE_ROOT = Path("/Volumes/corpdrive/hawking-modellake")
LAKE_SLUG = REPO_ID.replace("/", "--") + "@" + PINNED_REVISION[:12]
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


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment capability
        raise RuntimeError("flash-transform-parity requires numpy for bounded streaming") from exc
    return np


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


def _source_guard(path: Path) -> Tuple[int, int, int, int]:
    stat = path.stat()
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def _decode_bf16(np: Any, raw: bytes) -> Any:
    bits = np.frombuffer(raw, dtype="<u2")
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _bf16_bits(np: Any, values: Any) -> Any:
    raw = values.astype(np.float32, copy=False).view(np.uint32)
    rounded = raw + np.uint32(0x7FFF) + ((raw >> np.uint32(16)) & np.uint32(1))
    return ((rounded >> np.uint32(16)) & np.uint32(0xFFFF)).astype(np.uint16)


def _pack_nibbles(np: Any, codes: Any) -> bytes:
    flat = np.asarray(codes, dtype=np.uint8).reshape(-1)
    if flat.size % 2:
        flat = np.concatenate((flat, np.zeros(1, dtype=np.uint8)))
    packed = flat[0::2] | (flat[1::2] << np.uint8(4))
    return np.asarray(packed, dtype=np.uint8).tobytes()


def _unpack_nibbles(np: Any, packed: bytes, count: int) -> Any:
    raw = np.frombuffer(packed, dtype=np.uint8)
    codes = np.empty(raw.size * 2, dtype=np.uint8)
    codes[0::2] = raw & np.uint8(0x0F)
    codes[1::2] = raw >> np.uint8(4)
    return codes[: int(count)]


def _quantize_q4(np: Any, values: Any) -> Tuple[Any, Any, bytes, bytes]:
    grouped = np.asarray(values, dtype=np.float32).reshape(-1, GROUP_SIZE)
    peak = np.max(np.abs(grouped), axis=1)
    full_scale = np.where(peak > 0.0, peak / float(Q4_BOUND), 1.0).astype(np.float32)
    codes = np.rint(grouped / full_scale[:, None]).clip(-8, Q4_BOUND).astype(np.int8)
    stored_scales = full_scale.astype(np.float16).astype(np.float32)
    packed = _pack_nibbles(np, codes.astype(np.uint8) + np.uint8(8))
    scale_bytes = stored_scales.astype("<f2", copy=False).tobytes()
    return codes.astype(np.uint8) + np.uint8(8), stored_scales, packed, scale_bytes


def _quantize_nf4(np: Any, values: Any) -> Tuple[Any, Any, bytes, bytes]:
    grouped = np.asarray(values, dtype=np.float32).reshape(-1, GROUP_SIZE)
    peak = np.max(np.abs(grouped), axis=1)
    full_scale = np.where(peak > 0.0, peak, 1.0).astype(np.float32)
    normalized = grouped / full_scale[:, None]
    codebook = np.asarray(NF4_CODEBOOK, dtype=np.float32)
    boundaries = (codebook[:-1] + codebook[1:]) / np.float32(2.0)
    codes = np.searchsorted(boundaries, normalized, side="right").astype(np.uint8)
    stored_scales = full_scale.astype(np.float16).astype(np.float32)
    packed = _pack_nibbles(np, codes)
    scale_bytes = stored_scales.astype("<f2", copy=False).tobytes()
    return codes, stored_scales, packed, scale_bytes


def _decode_q4(np: Any, packed: bytes, scale_bytes: bytes, count: int) -> Any:
    codes = _unpack_nibbles(np, packed, count).astype(np.float32)
    scales = np.frombuffer(scale_bytes, dtype="<f2").astype(np.float32)
    return (codes - np.float32(8.0)) * np.repeat(scales, GROUP_SIZE)[: int(count)]


def _decode_nf4(np: Any, packed: bytes, scale_bytes: bytes, count: int) -> Any:
    codes = _unpack_nibbles(np, packed, count)
    scales = np.frombuffer(scale_bytes, dtype="<f2").astype(np.float32)
    return np.asarray(NF4_CODEBOOK, dtype=np.float32)[codes] * np.repeat(scales, GROUP_SIZE)[: int(count)]


def _read_block(
    np: Any,
    handle: Any,
    tensor: Mapping[str, Any],
    expert: int,
    row_start: int,
    row_count: int,
    row_total: int,
    columns: int,
) -> Tuple[bytes, Any]:
    row_bytes = int(columns) * 2
    offset = (
        int(tensor["data_start"])
        + int(tensor["data_offsets"][0])
        + ((int(expert) * int(row_total) + int(row_start)) * row_bytes)
    )
    count = int(row_count) * row_bytes
    handle.seek(offset)
    raw = handle.read(count)
    if len(raw) != count:
        raise ValueError(f"short tensor block read: {len(raw)} != {count}")
    values = _decode_bf16(np, raw).reshape(int(row_count), int(columns))
    return raw, values


@dataclass
class _Metrics:
    count: int = 0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0
    max_abs_error: float = 0.0
    reference_norm_squared: float = 0.0
    candidate_norm_squared: float = 0.0
    dot: float = 0.0
    finite: bool = True

    def update(self, np: Any, reference: Any, candidate: Any) -> None:
        left = np.asarray(reference, dtype=np.float32).reshape(-1)
        right = np.asarray(candidate, dtype=np.float32).reshape(-1)
        if left.size != right.size:
            raise ValueError("metric vectors have different sizes")
        error = left - right
        absolute = np.abs(error)
        self.count += int(error.size)
        self.sum_abs_error += float(np.sum(absolute, dtype=np.float64))
        self.sum_squared_error += float(np.sum(error * error, dtype=np.float64))
        self.max_abs_error = max(self.max_abs_error, float(np.max(absolute, initial=0.0)))
        self.reference_norm_squared += float(np.sum(left * left, dtype=np.float64))
        self.candidate_norm_squared += float(np.sum(right * right, dtype=np.float64))
        self.dot += float(np.sum(left * right, dtype=np.float64))
        self.finite = self.finite and bool(np.isfinite(right).all())

    def finish(self) -> Dict[str, Any]:
        ref_norm = math.sqrt(max(0.0, self.reference_norm_squared))
        candidate_norm = math.sqrt(max(0.0, self.candidate_norm_squared))
        cosine = self.dot / (ref_norm * candidate_norm) if ref_norm and candidate_norm else None
        return {
            "count": self.count,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.sum_abs_error / max(1, self.count),
            "rmse": math.sqrt(self.sum_squared_error / max(1, self.count)),
            "cosine": cosine,
            "finite": self.finite,
        }


@dataclass
class _Digest:
    packed: Any
    scales: Any
    combined: Any
    packed_bytes: int = 0
    scale_bytes: int = 0

    @classmethod
    def create(cls) -> "_Digest":
        return cls(hashlib.sha256(), hashlib.sha256(), hashlib.sha256())

    def update(self, packed: bytes, scales: bytes) -> None:
        self.packed.update(packed)
        self.scales.update(scales)
        self.combined.update(packed)
        self.combined.update(scales)
        self.packed_bytes += len(packed)
        self.scale_bytes += len(scales)

    def finish(self) -> Dict[str, Any]:
        return {
            "packed_sha256": self.packed.hexdigest(),
            "scale_sha256": self.scales.hexdigest(),
            "candidate_sha256": self.combined.hexdigest(),
            "code_bytes": self.packed_bytes,
            "scale_bytes": self.scale_bytes,
            "candidate_bytes": self.packed_bytes + self.scale_bytes,
        }


def _candidate_record(
    *,
    name: str,
    digest: _Digest,
    values: int,
    metrics: _Metrics,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "id": name,
        "label": DERIVED,
        **digest.finish(),
        "effective_bits_per_value": (digest.packed_bytes + digest.scale_bytes) * 8 / max(1, values),
        "weight_reconstruction": metrics.finish(),
        "model_capability_tested": False,
        "native_kernel_tested": False,
        "runtime_performance_tested": False,
        "status": "FULL_TENSOR_TRANSFORM_ONLY",
    }
    if extra:
        record.update(dict(extra))
    return record


def run_flash_transform_parity(
    *,
    root: Optional[str | os.PathLike[str]] = None,
    tensor_name: str = DEFAULT_TENSOR,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Run the full routed-expert tensor transform without loading a model."""
    final_root = _final_root(root)
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
        "tensor_name": tensor_name,
        "chunk_rows": int(chunk_rows),
        "source_label": VERIFIED,
        "candidate_label": DERIVED,
        "body_mutated": False,
        "model_loaded": False,
        "whole_model_capability": "NOT_TESTED",
        "whole_model_runtime": "NOT_TESTED",
    }
    try:
        np = _require_numpy()
        if not final_root.is_dir():
            raise FileNotFoundError(final_root)
        if int(chunk_rows) <= 0:
            raise ValueError("chunk_rows must be positive")
        manifest = _pinned_manifest(final_root)
        tensor = _load_tensor_header(final_root, tensor_name)
        if str(tensor.get("dtype") or "").upper() != "BF16":
            raise ValueError(f"full transform parity requires BF16, got {tensor.get('dtype')}")
        shape = [int(value) for value in tensor.get("shape") or []]
        if len(shape) != 3:
            raise ValueError(f"full transform parity requires rank-3 routed experts, got {shape}")
        expert_total, row_total, columns = shape
        if expert_total <= 0 or row_total <= 0 or columns <= 0 or columns % GROUP_SIZE:
            raise ValueError("source shape is not a positive [expert,row,column] G64 layout")
        expected_values = math.prod(shape)
        expected_payload = expected_values * 2
        actual_payload = int(tensor["data_offsets"][1]) - int(tensor["data_offsets"][0])
        if actual_payload != expected_payload:
            raise ValueError(f"tensor payload mismatch: {actual_payload} != {expected_payload}")
        if actual_payload > MAX_PAYLOAD_BYTES:
            raise ValueError(f"tensor payload exceeds safety limit: {actual_payload}")
        shard = Path(str(tensor["shard"]))
        guard_before = _source_guard(shard)

        # Pass one derives the shared basis and hashes the complete source
        # payload.  The basis is rounded to BF16 before residual formation, so
        # the second pass measures the exact stored candidate basis.
        basis_sum = np.zeros((row_total, columns), dtype=np.float64)
        source_digest = hashlib.sha256()
        source_bytes = 0
        pass_one_blocks = 0
        with shard.open("rb") as handle:
            for expert in range(expert_total):
                for row_start in range(0, row_total, int(chunk_rows)):
                    count = min(int(chunk_rows), row_total - row_start)
                    raw, values = _read_block(
                        np, handle, tensor, expert, row_start, count, row_total, columns
                    )
                    source_digest.update(raw)
                    source_bytes += len(raw)
                    basis_sum[row_start:row_start + count] += values.astype(np.float64)
                    pass_one_blocks += 1
        guard_after_pass_one = _source_guard(shard)
        if guard_after_pass_one != guard_before:
            raise RuntimeError("source shard changed during full transform parity")
        basis_mean = (basis_sum / float(expert_total)).astype(np.float32)
        basis_bits = _bf16_bits(np, basis_mean)
        basis = (basis_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
        basis_storage = basis_bits.astype("<u2", copy=False).tobytes()

        activation = np.sin(
            np.arange(1, columns + 1, dtype=np.float64) * 0.013,
        ).astype(np.float32)
        activation_rms = math.sqrt(float(np.mean(activation * activation)))
        if activation_rms:
            activation = activation / np.float32(activation_rms)
        reference_vector_sha256 = _sha256_bytes(activation.astype("<f4", copy=False).tobytes())

        independent_digest = _Digest.create()
        residual_digest = _Digest.create()
        independent_metrics = _Metrics()
        shared_metrics = _Metrics()
        independent_output_metrics = _Metrics()
        shared_output_metrics = _Metrics()
        source_output_metrics = _Metrics()
        source_bytes_second_pass = 0
        pass_two_blocks = 0
        pack_unpack_parity = True
        with shard.open("rb") as handle:
            for expert in range(expert_total):
                for row_start in range(0, row_total, int(chunk_rows)):
                    count = min(int(chunk_rows), row_total - row_start)
                    raw, values = _read_block(
                        np, handle, tensor, expert, row_start, count, row_total, columns
                    )
                    source_bytes_second_pass += len(raw)
                    basis_block = basis[row_start:row_start + count]
                    residual_values = values - basis_block
                    q4_codes, q4_scales, q4_packed, q4_scale_bytes = _quantize_q4(np, values)
                    nf4_codes, nf4_scales, nf4_packed, nf4_scale_bytes = _quantize_nf4(np, residual_values)
                    independent_digest.update(q4_packed, q4_scale_bytes)
                    residual_digest.update(nf4_packed, nf4_scale_bytes)
                    q4_roundtrip_codes = _unpack_nibbles(np, q4_packed, int(q4_codes.size))
                    nf4_roundtrip_codes = _unpack_nibbles(np, nf4_packed, int(nf4_codes.size))
                    pack_unpack_parity = pack_unpack_parity and bool(np.array_equal(q4_codes.reshape(-1), q4_roundtrip_codes))
                    pack_unpack_parity = pack_unpack_parity and bool(np.array_equal(nf4_codes.reshape(-1), nf4_roundtrip_codes))
                    q4_reconstructed = _decode_q4(np, q4_packed, q4_scale_bytes, int(q4_codes.size)).reshape(values.shape)
                    nf4_reconstructed = _decode_nf4(np, nf4_packed, nf4_scale_bytes, int(nf4_codes.size)).reshape(values.shape)
                    shared_reconstructed = basis_block + nf4_reconstructed
                    independent_metrics.update(np, values, q4_reconstructed)
                    shared_metrics.update(np, values, shared_reconstructed)
                    dense_outputs = values @ activation
                    q4_outputs = q4_reconstructed @ activation
                    shared_outputs = shared_reconstructed @ activation
                    source_output_metrics.update(np, dense_outputs, dense_outputs)
                    independent_output_metrics.update(np, dense_outputs, q4_outputs)
                    shared_output_metrics.update(np, dense_outputs, shared_outputs)
                    pass_two_blocks += 1
        guard_after_pass_two = _source_guard(shard)
        if guard_after_pass_two != guard_before:
            raise RuntimeError("source shard changed during full transform parity")

        source_complete = source_bytes == actual_payload == source_bytes_second_pass
        source_tensor = {
            "tensor_name": tensor_name,
            "shard": tensor["shard"],
            "shard_name": tensor["shard_name"],
            "dtype": tensor["dtype"],
            "shape": shape,
            "data_offsets": tensor["data_offsets"],
            "payload_bytes": actual_payload,
            "bytes_read_pass_one": source_bytes,
            "bytes_read_pass_two": source_bytes_second_pass,
            "payload_sha256": source_digest.hexdigest(),
            "layout": "row-major [expert, row, column]; complete tensor payload streamed in expert/row blocks",
            "label": VERIFIED,
            "selected_shard_full_hash_recomputed": False,
            "verification_scope": "complete BF16 tensor payload under exact pinned final ModelLake identity",
        }
        independent = _candidate_record(
            name="independent_q4_g64",
            digest=independent_digest,
            values=expected_values,
            metrics=independent_metrics,
            extra={
                "scheme": "symmetric_signed_4bit_group64_with_fp16_scales",
                "group_size": GROUP_SIZE,
                "source_layout_preserved": True,
                "pack_unpack_parity": pack_unpack_parity,
                "reference_vector": independent_output_metrics.finish(),
            },
        )
        basis_digest = hashlib.sha256(basis_storage).hexdigest()
        residual = _candidate_record(
            name="shared_bf16_basis_nf4_residual",
            digest=residual_digest,
            values=expected_values,
            metrics=shared_metrics,
            extra={
                "scheme": "shared_bf16_basis_plus_nf4_group64_residual_with_fp16_scales",
                "group_size": GROUP_SIZE,
                "basis_bytes": len(basis_storage),
                "basis_sha256": basis_digest,
                "basis_dtype": "BF16",
                "basis_shared_once_across_all_experts": True,
                "source_layout_preserved": True,
                "pack_unpack_parity": pack_unpack_parity,
                "reference_vector": shared_output_metrics.finish(),
                "residual_codebook": list(NF4_CODEBOOK),
            },
        )
        residual["candidate_bytes"] = int(residual["candidate_bytes"]) + len(basis_storage)
        residual["effective_bits_per_value"] = residual["candidate_bytes"] * 8 / max(1, expected_values)
        result.update({
            "status": "PASSED",
            "model_lake_manifest": manifest,
            "source_tensor": source_tensor,
            "reference_vector": {
                "construction": "deterministic normalized sine vector; tensor-level numerical control only",
                "length": columns,
                "rms": activation_rms,
                "sha256": reference_vector_sha256,
                "label": DERIVED,
            },
            "candidates": {
                "independent_q4_g64": independent,
                "shared_bf16_basis_nf4_residual": residual,
            },
            "comparison": {
                "dense_bytes": expected_values * 2,
                "independent_q4_bytes": independent["candidate_bytes"],
                "shared_basis_nf4_residual_bytes": residual["candidate_bytes"],
                "independent_q4_smaller_than_dense": independent["candidate_bytes"] < expected_values * 2,
                "shared_basis_nf4_residual_smaller_than_dense": residual["candidate_bytes"] < expected_values * 2,
                "shared_basis_nf4_residual_smaller_than_independent_q4": residual["candidate_bytes"] < independent["candidate_bytes"],
                "same_source_tensor": True,
                "same_dimensions": True,
                "full_payload_read": source_complete,
                "capability_parity": "NOT_TESTED",
                "whole_model_runtime": "NOT_TESTED",
                "label": DERIVED,
            },
            "transform_parity": {
                "status": "PASSED" if source_complete and pack_unpack_parity else "FAILED",
                "source_shape_payload_match": True,
                "complete_source_payload_read": source_complete,
                "source_file_unchanged_between_passes": guard_after_pass_two == guard_before,
                "pass_one_blocks": pass_one_blocks,
                "pass_two_blocks": pass_two_blocks,
                "pack_unpack_parity": pack_unpack_parity,
                "independent_reconstruction_finite": independent_metrics.finite,
                "shared_reconstruction_finite": shared_metrics.finite,
                "dense_reference_dots_finite": source_output_metrics.finite,
                "candidate_reference_dots_finite": independent_output_metrics.finite and shared_output_metrics.finite,
                "candidate_hashes_complete": independent["candidate_bytes"] > 0 and residual["candidate_bytes"] > 0,
                "label": DERIVED,
            },
            "whole_model_capability": "NOT_TESTED",
            "native_loader": "NOT_TESTED",
            "native_kernel": "NOT_TESTED",
            "runtime_performance": "NOT_TESTED",
            "next_experiment": {
                "id": "flash-routed-expert-bounded-loader-roundtrip",
                "action": "serialize the chosen source-layout candidate metadata and run a bounded loader encode/decode round-trip before implementing a native kernel",
                "source_mutation_allowed": False,
                "required_before_promotion": [
                    "bounded loader",
                    "native kernel",
                    "whole-model capability",
                    "protected complete-token timing",
                ],
            },
            "claim_boundary": "Full routed-expert tensor body transform parity only. This does not establish whole-model capability, native loader compatibility, native-kernel performance, or promotion.",
        })
        if result["transform_parity"]["status"] != "PASSED":
            result["status"] = "FAILED"
    except Exception as exc:  # noqa: BLE001 - persist actionable body evidence
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
    parser.add_argument("--tensor-name", default=DEFAULT_TENSOR)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = run_flash_transform_parity(
        root=args.root,
        tensor_name=args.tensor_name,
        chunk_rows=args.chunk_rows,
        emit=args.emit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["DEFAULT_TENSOR", "SCHEMA", "main", "run_flash_transform_parity"]


if __name__ == "__main__":
    raise SystemExit(main())

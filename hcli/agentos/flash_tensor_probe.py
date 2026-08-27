"""Bounded source-tensor and representation probe for acquired Flash-Next.

The probe reads a safetensors header and a small prefix of one verified final
tensor.  It performs a local dense-vs-packed-low-bit reconstruction comparison
on that prefix only.  It does not write the specimen, load a model, claim
capability, or claim runtime performance.
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


SCHEMA = "hcli.agentos.flash_tensor_probe.v1"
DERIVED = "[D]"
VERIFIED = "[V]"
DEFAULT_TENSOR = "model.language_model.layers.0.mlp.experts.gate_up_proj"
DEFAULT_EMIT_NAME = "FLASH_FIRST_TENSOR_PROBE.json"
LAKE_ROOT = Path("/Volumes/corpdrive/hawking-modellake")
LAKE_SLUG = REPO_ID.replace("/", "--") + "@" + PINNED_REVISION[:12]
MAX_SAMPLE_BYTES = 4 * 1024 * 1024
DEFAULT_SAMPLE_BYTES = 1 * 1024 * 1024
GROUP_SIZE = 64


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file_prefix(path: Path, count: int) -> Optional[str]:
    try:
        with path.open("rb") as handle:
            return _sha256_bytes(handle.read(max(0, int(count))))
    except OSError:
        return None


def _float_from_bf16(raw: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(raw) << 16))[0]


def _decode_bf16(data: bytes) -> list[float]:
    usable = len(data) - (len(data) % 2)
    return [_float_from_bf16(raw[0]) for raw in struct.iter_unpack("<H", data[:usable])]


def _quantize_packed(values: Sequence[float]) -> Dict[str, Any]:
    """Use a transparent symmetric 4-bit baseline, not an NF quality claim."""
    scales: list[float] = []
    codes: list[int] = []
    reconstructed: list[float] = []
    for offset in range(0, len(values), GROUP_SIZE):
        group = list(values[offset:offset + GROUP_SIZE])
        if not group:
            continue
        peak = max(abs(float(value)) for value in group)
        scale = peak / 7.0 if peak > 0.0 else 1.0
        scales.append(scale)
        for value in group:
            code = max(-8, min(7, int(round(float(value) / scale)))) if scale else 0
            codes.append(code)
            reconstructed.append(float(code) * scale)
    packed = bytearray()
    for index in range(0, len(codes), 2):
        low = int(codes[index]) + 8
        high = int(codes[index + 1]) + 8 if index + 1 < len(codes) else 8
        packed.append((low & 0x0F) | ((high & 0x0F) << 4))
    scale_bytes = b"".join(struct.pack("<e", float(scale)) for scale in scales)
    errors = [float(a) - float(b) for a, b in zip(values, reconstructed)]
    abs_errors = [abs(value) for value in errors]
    mse = sum(value * value for value in errors) / max(1, len(errors))
    peak = max((abs(float(value)) for value in values), default=0.0)
    return {
        "scheme": "symmetric_signed_4bit_group64_with_fp16_scales",
        "label": DERIVED,
        "status": "BOUNDED_SLICE_RECONSTRUCTION_ONLY",
        "group_size": GROUP_SIZE,
        "groups": len(scales),
        "code_bytes": len(packed),
        "scale_bytes": len(scale_bytes),
        "candidate_bytes": len(packed) + len(scale_bytes),
        "candidate_sha256": _sha256_bytes(bytes(packed) + scale_bytes),
        "effective_bits_per_value": (len(packed) + len(scale_bytes)) * 8 / max(1, len(values)),
        "max_abs_error": max(abs_errors, default=0.0),
        "mean_abs_error": sum(abs_errors) / max(1, len(abs_errors)),
        "rmse": math.sqrt(mse),
        "source_peak_abs": peak,
        "reconstruction_finite": all(math.isfinite(value) for value in reconstructed),
        "model_capability_tested": False,
        "runtime_performance_tested": False,
    }


def _final_root(value: Optional[str | os.PathLike[str]]) -> Path:
    chosen = value or os.environ.get("HCLI_FLASH_NEXT_ROOT")
    if chosen:
        return Path(chosen).expanduser().resolve()
    return (LAKE_ROOT / "specimens" / LAKE_SLUG).resolve()


def _load_tensor_header(root: Path, tensor_name: str) -> Dict[str, Any]:
    index_path = root / "model.safetensors.index.json"
    index = _read_json(index_path)
    if index is None:
        raise FileNotFoundError(index_path)
    weight_map = index.get("weight_map") if isinstance(index.get("weight_map"), Mapping) else {}
    shard_name = weight_map.get(tensor_name)
    if not shard_name:
        raise KeyError(f"tensor is absent from pinned index: {tensor_name}")
    shard = root / str(shard_name)
    try:
        shard.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("safetensors index points outside the selected specimen root") from exc
    with shard.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("safetensors shard has no header length")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes <= 0 or header_bytes > 64 * 1024 * 1024:
            raise ValueError(f"unsafe safetensors header length: {header_bytes}")
        header_raw = handle.read(header_bytes)
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors header: {exc}") from exc
    tensor = header.get(tensor_name) if isinstance(header, Mapping) else None
    if not isinstance(tensor, Mapping):
        raise KeyError(f"tensor is absent from local shard header: {tensor_name}")
    shape = tensor.get("shape")
    offsets = tensor.get("data_offsets")
    if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError("tensor header lacks shape/data_offsets")
    begin, end = int(offsets[0]), int(offsets[1])
    if begin < 0 or end < begin:
        raise ValueError("tensor data_offsets are invalid")
    file_size = shard.stat().st_size
    data_start = 8 + int(header_bytes)
    if data_start + end > file_size:
        raise ValueError("tensor data range exceeds local shard")
    expected_values = math.prod(int(value) for value in shape)
    if str(tensor.get("dtype") or "").upper() == "BF16" and expected_values * 2 != end - begin:
        raise ValueError("BF16 tensor shape does not match its declared payload bytes")
    return {
        "tensor_name": tensor_name,
        "shard": str(shard),
        "shard_name": str(shard_name),
        "shard_size": file_size,
        "header_bytes": int(header_bytes),
        "dtype": str(tensor.get("dtype") or "UNKNOWN"),
        "shape": [int(value) for value in shape],
        "data_offsets": [begin, end],
        "payload_bytes": end - begin,
        "data_start": data_start,
        "index_sha256": _sha256_file_prefix(root / "model.safetensors.index.json", (root / "model.safetensors.index.json").stat().st_size),
    }


def run_flash_tensor_probe(
    *,
    root: Optional[str | os.PathLike[str]] = None,
    tensor_name: str = DEFAULT_TENSOR,
    sample_bytes: int = DEFAULT_SAMPLE_BYTES,
    emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    final_root = _final_root(root)
    destination = Path(emit).expanduser().resolve() if emit else Path(__file__).resolve().parents[2] / "receipts" / "headless" / DEFAULT_EMIT_NAME
    started = time.time()
    result: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "repo": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "root": str(final_root),
        "tensor_name": tensor_name,
        "requested_sample_bytes": max(2, min(MAX_SAMPLE_BYTES, int(sample_bytes))),
        "source_label": VERIFIED,
        "candidate_label": DERIVED,
        "body_mutated": False,
        "model_loaded": False,
    }
    try:
        if not final_root.is_dir():
            raise FileNotFoundError(final_root)
        manifest_path = LAKE_ROOT / "manifests" / f"{LAKE_SLUG}.json"
        manifest = _read_json(manifest_path)
        final_identity = bool(
            isinstance(manifest, Mapping)
            and manifest.get("repo") == REPO_ID
            and manifest.get("resolved_sha") == PINNED_REVISION
            and manifest.get("revision") == PINNED_REVISION
        )
        if not final_identity:
            raise ValueError("final specimen is not backed by the pinned ModelLake manifest")
        manifest_root = manifest.get("path") if isinstance(manifest, Mapping) else None
        if manifest_root and Path(str(manifest_root)).expanduser().resolve() != final_root:
            raise ValueError("selected specimen root does not match the pinned ModelLake manifest path")
        tensor = _load_tensor_header(final_root, tensor_name)
        if tensor["dtype"].upper() != "BF16":
            raise ValueError(f"bounded probe currently requires BF16, got {tensor['dtype']}")
        count = min(result["requested_sample_bytes"], tensor["payload_bytes"])
        count -= count % 2
        with Path(tensor["shard"]).open("rb") as handle:
            handle.seek(tensor["data_start"] + tensor["data_offsets"][0])
            sample = handle.read(count)
        if len(sample) != count:
            raise ValueError(f"short tensor slice read: {len(sample)} != {count}")
        values = _decode_bf16(sample)
        dense_bytes = len(values) * 2
        candidate = _quantize_packed(values)
        result.update({
            "status": "PASSED",
            "final_identity": {
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256_file_prefix(manifest_path, manifest_path.stat().st_size),
                "resolved_sha": manifest.get("resolved_sha"),
                "n_files": manifest.get("n_files"),
                "bytes": manifest.get("bytes"),
                "label": VERIFIED,
            },
            "source_tensor": {
                **{key: value for key, value in tensor.items() if key != "data_start"},
                "label": VERIFIED,
                "slice_offset_bytes": 0,
                "slice_bytes": len(sample),
                "slice_values": len(values),
                "slice_sha256": _sha256_bytes(sample),
                "slice_finite": all(math.isfinite(value) for value in values),
                "slice_nonzero_values": sum(value != 0.0 for value in values),
                "verification_scope": "local safetensors header and bounded bytes under the exact pinned final ModelLake identity",
                "selected_shard_full_hash_recomputed": False,
            },
            "organ": {
                "id": "routed_experts",
                "mapping_basis": "tensor name model.language_model.layers.*.mlp.experts.*",
                "label": DERIVED,
            },
            "dense_vs_packed_low_bit": {
                "control": {
                    "representation": "source BF16 slice",
                    "bytes": dense_bytes,
                    "bits_per_value": 16.0,
                    "label": VERIFIED,
                },
                "candidate": candidate,
                "comparison": {
                    "candidate_bytes_over_control": candidate["candidate_bytes"] / max(1, dense_bytes),
                    "candidate_is_smaller": candidate["candidate_bytes"] < dense_bytes,
                    "same_values_compared": len(values),
                    "capability_parity": "NOT_TESTED",
                    "whole_model_runtime": "NOT_TESTED",
                    "label": DERIVED,
                },
            },
            "next_experiment": {
                "id": "flash-routed-expert-slice-to-organ",
                "action": "implement exact source-layout-aware shared-basis/NF-residual representation and matched reference-vector parity before any native kernel",
                "source_mutation_allowed": False,
                "required_before_promotion": ["full tensor transform parity", "native loader", "native kernel", "whole-model capability", "protected complete-token timing"],
            },
        })
    except Exception as exc:  # noqa: BLE001 - preserve bounded probe failure
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
    parser.add_argument("--sample-bytes", type=int, default=DEFAULT_SAMPLE_BYTES)
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = run_flash_tensor_probe(root=args.root, tensor_name=args.tensor_name, sample_bytes=args.sample_bytes, emit=args.emit)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["DEFAULT_TENSOR", "SCHEMA", "main", "run_flash_tensor_probe"]


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3.12
"""Bounded-memory GPT-OSS MXFP4 inventory and staging primitives.

This module is deliberately separate from the dense Qwen ladder worker.  The
GPT-OSS checkpoint stores two E2M1 values in every U8 ``.blocks`` byte and one
UE8 exponent per group of 32 logical weights.  Consequently, serialized U8
elements are not parameters and UE8 scale bytes are not model weights.

The audit path reads only safetensors headers.  The staging path reads one
expert slice at a time with ``pread`` and emits canonical BF16 ``[out, in]``
matrices without ever materializing a source shard, layer, or model densely.
Source files are read-only and are never deletion candidates.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import struct
import sys
from typing import Any, BinaryIO, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "scratch/staging/gpt-oss-120b.partial"
DEFAULT_CENSUS = ROOT / "reports/condense/doctor_v5_scale/120B/census.json"
DEFAULT_INVENTORY = (
    ROOT / "reports/condense/doctor_v5_ultra/gpt_oss_120b_mxfp4_inventory.json"
)
DEFAULT_LOGICAL_PARAMETER_MANIFEST = (
    ROOT / "reports/condense/doctor_v5_pass_b/parameter_manifests/120B.json"
)
INVENTORY_SCHEMA = "hawking.doctor_v5_gptoss_mxfp4_inventory.v1"
STAGING_RECEIPT_SCHEMA = "hawking.doctor_v5_gptoss_mxfp4_staging_receipt.v1"
PARAMETER_MANIFEST_SCHEMA = "hawking.doctor_v5_parameter_manifest.v1"
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
SHA_RE = re.compile(r"[0-9a-f]{64}")
FP4_VALUES = (
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
CANONICAL_RATES = (
    ("4", Fraction(4, 1)), ("3", Fraction(3, 1)),
    ("2", Fraction(2, 1)), ("1", Fraction(1, 1)),
    ("0.8", Fraction(4, 5)), ("0.55", Fraction(11, 20)),
    ("0.5", Fraction(1, 2)), ("0.33", Fraction(33, 100)),
    ("0.25", Fraction(1, 4)), ("0.1", Fraction(1, 10)),
)
EXPECTED_CONFIG = {
    "num_hidden_layers": 36,
    "num_experts": 128,
    "experts_per_token": 4,
    "vocab_size": 201088,
    "hidden_size": 2880,
    "intermediate_size": 2880,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
}
PROJECTION_SHAPES = {
    "mlp1": (128, 5760, 90, 16),
    "mlp2": (128, 2880, 90, 16),
}


class Mxfp4Error(RuntimeError):
    """The checkpoint, inventory, or staging request is unsafe or invalid."""


@dataclass(frozen=True)
class TensorMeta:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int
    shard_name: str
    shard_path: Path
    shard_header_sha256: str

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.data_end - self.data_start


@dataclass(frozen=True)
class ShardMeta:
    name: str
    path: Path
    file_bytes: int
    header_bytes: int
    header_sha256: str
    data_start: int
    data_bytes: int
    tensors: tuple[TensorMeta, ...]


@dataclass(frozen=True)
class Inspection:
    model_dir: Path
    weights_root: Path
    census_path: Path
    census: dict[str, Any]
    config: dict[str, Any]
    dtypes: dict[str, str]
    index: dict[str, Any]
    shards: tuple[ShardMeta, ...]
    tensors: dict[str, TensorMeta]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise Mxfp4Error(f"cannot open regular source file {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise Mxfp4Error(f"source is not a regular file: {path}")
        digest, total = hashlib.sha256(), 0
        while True:
            block = os.read(fd, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
        after = os.fstat(fd)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
        if identity(before) != identity(after) or total != after.st_size:
            raise Mxfp4Error(f"source changed while hashing: {path}")
        return digest.hexdigest(), total
    finally:
        os.close(fd)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) \
                or info.st_size > MAX_JSON_BYTES:
            raise Mxfp4Error(f"invalid JSON file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Mxfp4Error(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Mxfp4Error(f"JSON root is not an object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _safe_child(root: Path, child: str) -> Path:
    if not isinstance(child, str) or not child or Path(child).is_absolute() \
            or ".." in Path(child).parts:
        raise Mxfp4Error(f"unsafe relative source path: {child!r}")
    root = root.resolve(strict=True)
    candidate = root / child
    cursor = root
    for part in Path(child).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise Mxfp4Error(f"symlinked source is forbidden: {cursor}")
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Mxfp4Error(f"source path escapes model root: {child!r}") from exc
    return path


def _dtype_bytes(dtype: str) -> int:
    widths = {"U8": 1, "BF16": 2, "F16": 2, "F32": 4}
    try:
        return widths[dtype]
    except KeyError as exc:
        raise Mxfp4Error(f"unsupported safetensors dtype: {dtype}") from exc


def _read_header(path: Path, shard_name: str) -> ShardMeta:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise Mxfp4Error(f"cannot open shard {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise Mxfp4Error(f"shard is not regular: {path}")
        prefix = os.pread(fd, 8, 0)
        if len(prefix) != 8:
            raise Mxfp4Error(f"truncated safetensors length: {path}")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 0 or header_len > MAX_HEADER_BYTES \
                or 8 + header_len > before.st_size:
            raise Mxfp4Error(f"unsafe safetensors header length in {path}")
        header = os.pread(fd, header_len, 8)
        if len(header) != header_len:
            raise Mxfp4Error(f"truncated safetensors header: {path}")
        try:
            parsed = json.loads(header)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Mxfp4Error(f"invalid safetensors header {path}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise Mxfp4Error(f"safetensors header is not an object: {path}")
        data_start = 8 + header_len
        data_bytes = before.st_size - data_start
        tensors: list[TensorMeta] = []
        occupied: list[tuple[int, int, str]] = []
        for name, row in parsed.items():
            if name == "__metadata__":
                continue
            if not isinstance(name, str) or not name or not isinstance(row, dict) \
                    or set(row) != {"dtype", "shape", "data_offsets"}:
                raise Mxfp4Error(f"invalid tensor header entry in {path}: {name!r}")
            dtype, shape, offsets = row["dtype"], row["shape"], row["data_offsets"]
            if not isinstance(dtype, str) or not isinstance(shape, list) \
                    or not shape or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
                                        for x in shape) \
                    or not isinstance(offsets, list) or len(offsets) != 2 \
                    or any(isinstance(x, bool) or not isinstance(x, int) for x in offsets):
                raise Mxfp4Error(f"invalid tensor shape/offset in {path}: {name}")
            start, end = offsets
            expected = math.prod(shape) * _dtype_bytes(dtype)
            if start < 0 or end <= start or end > data_bytes or end - start != expected:
                raise Mxfp4Error(f"tensor byte extent is invalid in {path}: {name}")
            occupied.append((start, end, name))
            tensors.append(TensorMeta(
                name=name, dtype=dtype, shape=tuple(shape),
                data_start=data_start + start, data_end=data_start + end,
                shard_name=shard_name, shard_path=path,
                shard_header_sha256=hashlib.sha256(header).hexdigest(),
            ))
        occupied.sort()
        if any(a[1] > b[0] for a, b in zip(occupied, occupied[1:])):
            raise Mxfp4Error(f"overlapping tensor extents in {path}")
        after = os.fstat(fd)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
        if identity(before) != identity(after):
            raise Mxfp4Error(f"shard changed while reading its header: {path}")
        return ShardMeta(
            name=shard_name, path=path, file_bytes=before.st_size,
            header_bytes=header_len, header_sha256=hashlib.sha256(header).hexdigest(),
            data_start=data_start, data_bytes=data_bytes,
            tensors=tuple(sorted(tensors, key=lambda row: row.name)),
        )
    finally:
        os.close(fd)


def inspect_model(model_dir: Path = DEFAULT_MODEL_DIR,
                  census_path: Path = DEFAULT_CENSUS) -> Inspection:
    raw_model_dir, raw_census_path = Path(model_dir), Path(census_path)
    if raw_model_dir.is_symlink() or raw_census_path.is_symlink():
        raise Mxfp4Error("symlinked model/census inputs are forbidden")
    model_dir = raw_model_dir.resolve(strict=True)
    census_path = raw_census_path.resolve(strict=True)
    weights_candidate = model_dir if (model_dir / "model.safetensors.index.json").is_file() \
        else model_dir / "original"
    if weights_candidate.is_symlink():
        raise Mxfp4Error(f"symlinked weights root is forbidden: {weights_candidate}")
    weights_root = weights_candidate
    weights_root = weights_root.resolve(strict=True)
    config_path = _safe_child(weights_root, "config.json")
    dtypes_path = _safe_child(weights_root, "dtypes.json")
    index_path = _safe_child(weights_root, "model.safetensors.index.json")
    config, dtypes, index = (_read_json(config_path), _read_json(dtypes_path),
                             _read_json(index_path))
    census = _read_json(census_path)
    if census.get("schema") != "hawking.doctor_v5_source_census.v1" \
            or census.get("status") != "complete" or census.get("label") != "120B" \
            or census.get("hf_id") != "openai/gpt-oss-120b":
        raise Mxfp4Error("a completed matching GPT-OSS 120B census is required")
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise Mxfp4Error(f"config invariant differs: {key}={config.get(key)!r}")
    weight_map = index.get("weight_map")
    metadata = index.get("metadata")
    if not isinstance(weight_map, dict) or not weight_map \
            or not isinstance(metadata, dict):
        raise Mxfp4Error("safetensors index is incomplete")
    shard_names = sorted(set(weight_map.values()))
    if len(shard_names) != 7 or any(not isinstance(name, str) for name in shard_names):
        raise Mxfp4Error("GPT-OSS checkpoint must contain exactly seven indexed shards")
    census_rows = census.get("source", {}).get("shards")
    if not isinstance(census_rows, list) or len(census_rows) != len(shard_names):
        raise Mxfp4Error("census shard registry differs from the index")
    census_by_basename = {Path(row.get("name", "")).name: row for row in census_rows
                          if isinstance(row, dict)}
    shards: list[ShardMeta] = []
    tensors: dict[str, TensorMeta] = {}
    for shard_name in shard_names:
        path = _safe_child(weights_root, shard_name)
        shard = _read_header(path, shard_name)
        row = census_by_basename.get(shard_name)
        if not isinstance(row, dict):
            raise Mxfp4Error(f"shard is not bound by census: {shard_name}")
        info = path.stat()
        if shard.file_bytes != row.get("bytes") or shard.data_bytes != row.get("data_bytes") \
                or shard.header_bytes != row.get("header_bytes") \
                or shard.header_sha256 != row.get("header_sha256") \
                or info.st_dev != row.get("device") or info.st_ino != row.get("inode") \
                or info.st_size != row.get("bytes") or info.st_mtime_ns != row.get("mtime_ns"):
            raise Mxfp4Error(f"live shard identity differs from completed census: {shard_name}")
        for tensor in shard.tensors:
            if tensor.name in tensors:
                raise Mxfp4Error(f"duplicate tensor across shards: {tensor.name}")
            tensors[tensor.name] = tensor
        shards.append(shard)
    if set(weight_map) != set(tensors):
        raise Mxfp4Error("index and live shard tensor inventories differ")
    for name, shard_name in weight_map.items():
        if tensors[name].shard_name != shard_name:
            raise Mxfp4Error(f"index assigns tensor to wrong shard: {name}")
    if not isinstance(dtypes, dict) or set(dtypes) != set(tensors) \
            or any(not isinstance(value, str) for value in dtypes.values()):
        raise Mxfp4Error("dtypes manifest and live tensor inventory differ")
    if metadata.get("total_size") != sum(shard.data_bytes for shard in shards):
        raise Mxfp4Error("index total_size differs from live tensor payload")
    return Inspection(
        model_dir=model_dir, weights_root=weights_root, census_path=census_path,
        census=census, config=config, dtypes=dtypes, index=index,
        shards=tuple(shards), tensors=tensors,
    )


def _projection_pair(inspection: Inspection, layer: int,
                     projection: str) -> tuple[TensorMeta, TensorMeta]:
    if layer < 0 or layer >= EXPECTED_CONFIG["num_hidden_layers"]:
        raise Mxfp4Error(f"layer is outside [0,35]: {layer}")
    if projection not in PROJECTION_SHAPES:
        raise Mxfp4Error(f"unknown expert projection: {projection}")
    prefix = f"block.{layer}.mlp.{projection}_weight"
    try:
        blocks = inspection.tensors[f"{prefix}.blocks"]
        scales = inspection.tensors[f"{prefix}.scales"]
    except KeyError as exc:
        raise Mxfp4Error(f"missing MXFP4 pair for {prefix}") from exc
    expected = PROJECTION_SHAPES[projection]
    if blocks.dtype != "U8" or blocks.shape != expected \
            or scales.dtype != "U8" or scales.shape != expected[:-1] \
            or inspection.dtypes[blocks.name] != "FP4" \
            or inspection.dtypes[scales.name] != "UE8" \
            or blocks.shard_path != scales.shard_path:
        raise Mxfp4Error(f"invalid MXFP4 layout for {prefix}")
    if blocks.elements * 2 != scales.elements * 32:
        raise Mxfp4Error(f"MXFP4 block/scale ratio differs from 32: {prefix}")
    return blocks, scales


def _classify_bf16(name: str) -> str:
    if name == "embedding.weight":
        return "embedding_table"
    if name == "unembedding.weight":
        return "output_head"
    if re.fullmatch(r"block\.\d+\.mlp\.mlp[12]_bias", name):
        return "expert_bias"
    return "always_compute_dense"


def build_inventory(inspection: Inspection) -> dict[str, Any]:
    for layer in range(EXPECTED_CONFIG["num_hidden_layers"]):
        for projection in PROJECTION_SHAPES:
            _projection_pair(inspection, layer, projection)
    blocks = [row for name, row in inspection.tensors.items() if name.endswith(".blocks")]
    scales = [row for name, row in inspection.tensors.items() if name.endswith(".scales")]
    bf16 = [row for row in inspection.tensors.values() if row.dtype == "BF16"]
    if len(blocks) != 72 or len(scales) != 72 or len(bf16) != 399 \
            or len(inspection.tensors) != 543:
        raise Mxfp4Error("GPT-OSS tensor class counts differ from the reviewed layout")
    expected_u8 = {row.name for row in blocks + scales}
    for name, row in inspection.tensors.items():
        if name in expected_u8:
            if row.dtype != "U8":
                raise Mxfp4Error(f"packed expert tensor is not U8: {name}")
        elif row.dtype != "BF16" or inspection.dtypes[name] != "BF16":
            raise Mxfp4Error(f"unexpected non-BF16 tensor outside MXFP4 pairs: {name}")

    packed_block_bytes = sum(row.nbytes for row in blocks)
    scale_bytes = sum(row.nbytes for row in scales)
    expert_logical = sum(row.elements * 2 for row in blocks)
    bf16_logical = sum(row.elements for row in bf16)
    logical = expert_logical + bf16_logical
    tensor_payload = sum(row.data_bytes for row in inspection.shards)
    serialized = sum(row.file_bytes for row in inspection.shards)
    stored_elements = sum(row.elements for row in inspection.tensors.values())
    if (packed_block_bytes, scale_bytes, expert_logical, bf16_logical, logical,
            tensor_payload, stored_elements) != (
            57_330_892_800, 3_583_180_800, 114_661_785_600,
            2_167_371_072, 116_829_156_672, 65_248_815_744,
            63_081_444_672):
        raise Mxfp4Error("logical/physical accounting differs from reviewed GPT-OSS 120B totals")

    bf16_classes: dict[str, int] = {}
    for row in bf16:
        category = _classify_bf16(row.name)
        bf16_classes[category] = bf16_classes.get(category, 0) + row.elements
    selected_expert = expert_logical * EXPECTED_CONFIG["experts_per_token"] \
        // EXPECTED_CONFIG["num_experts"]
    selected_bias = bf16_classes["expert_bias"] * EXPECTED_CONFIG["experts_per_token"] \
        // EXPECTED_CONFIG["num_experts"]
    always = bf16_classes["always_compute_dense"]
    active_compute = (selected_expert + selected_bias + always
                      + EXPECTED_CONFIG["hidden_size"] + bf16_classes["output_head"])
    resident_equivalent = (selected_expert + selected_bias
                           + sum(bf16_classes.values())
                           - bf16_classes["expert_bias"])
    if active_compute != 5_132_852_352 or resident_equivalent != 5_711_982_912:
        raise Mxfp4Error("active MoE accounting no longer closes")

    tensor_rows = [{
        "name": row.name, "dtype": row.dtype, "shape": list(row.shape),
        "serialized_elements": row.elements, "bytes": row.nbytes,
        "shard": row.shard_name,
        "absolute_data_offsets": [row.data_start, row.data_end],
    } for row in sorted(inspection.tensors.values(), key=lambda value: value.name)]
    tensor_inventory_sha = _hash_value(tensor_rows)
    census_sha, census_bytes = _hash_file(inspection.census_path)
    config_path = inspection.weights_root / "config.json"
    dtypes_path = inspection.weights_root / "dtypes.json"
    index_path = inspection.weights_root / "model.safetensors.index.json"
    source = inspection.census["source"]
    rates = [{
        "rate_id": rate_id,
        "target_bpw": float(rate),
        "maximum_model_payload_bytes": math.ceil(logical * rate.numerator
                                                  / rate.denominator / 8),
    } for rate_id, rate in CANONICAL_RATES]
    source_rows = []
    census_by_basename = {Path(row["name"]).name: row for row in source["shards"]}
    for ordinal, shard in enumerate(inspection.shards):
        census_row = census_by_basename[shard.name]
        source_rows.append({
            "ordinal": ordinal, "name": str(shard.path.relative_to(inspection.model_dir)),
            "path": str(shard.path), "file_bytes": shard.file_bytes,
            "data_bytes": shard.data_bytes, "header_bytes": shard.header_bytes,
            "header_sha256": shard.header_sha256,
            "file_sha256": census_row["file_sha256"],
            "full_file_hash_authority": "completed_source_census",
            "live_identity_matches_census": True,
        })
    doc: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "created_at": _now(),
        "model": {"label": "120B", "hf_id": "openai/gpt-oss-120b",
                  "family": "gpt-oss-moe", "model_dir": str(inspection.model_dir),
                  "weights_root": str(inspection.weights_root)},
        "source_binding": {
            "census_path": str(inspection.census_path),
            "census_file_sha256": census_sha, "census_file_bytes": census_bytes,
            "census_report_sha256": inspection.census["report_sha256"],
            "source_manifest_sha256": source["source_manifest_sha256"],
            "config": _artifact(config_path), "dtypes": _artifact(dtypes_path),
            "index": _artifact(index_path), "shards": source_rows,
            "source_files_deleted": False,
        },
        "format": {
            "expert_weight_encoding": "MXFP4 E2M1; two logical values per U8 block byte",
            "scale_encoding": "UE8 exponent biased by 127",
            "scale_group_logical_weights": 32,
            "decode_equation": "weight = E2M1[nibble] * 2**(UE8-127)",
            "canonical_staging_orientation": "per-expert BF16 [out_features,in_features]",
            "reference_semantics": "transformers.integrations.mxfp4 FP4_VALUES and conversion",
        },
        "parameter_accounting": {
            "logical_model_parameters": logical,
            "logical_expert_weight_parameters": expert_logical,
            "logical_bf16_parameters": bf16_logical,
            "serialized_safetensors_elements_not_a_parameter_denominator": stored_elements,
            "packed_block_u8_elements": packed_block_bytes,
            "ue8_scale_elements_side_information_not_parameters": scale_bytes,
            "active_compute_parameter_equivalent": active_compute,
            "active_compute_convention": (
                "4/128 expert weights+biases, all always-compute dense tensors, one "
                "embedding row, and the full output head"
            ),
            "active_resident_table_equivalent": resident_equivalent,
            "active_resident_convention": (
                "4/128 expert weights+biases plus all stored non-expert BF16 tables"
            ),
            "bf16_logical_by_execution_class": bf16_classes,
            "authoritative_for_physical_bpw_denominator": True,
        },
        "physical_accounting": {
            "packed_e2m1_block_bytes": packed_block_bytes,
            "ue8_scale_side_information_bytes": scale_bytes,
            "bf16_tensor_bytes": bf16_logical * 2,
            "tensor_payload_bytes": tensor_payload,
            "serialized_shard_file_bytes": serialized,
            "native_expert_lane_bpw": (packed_block_bytes + scale_bytes) * 8
                                      / expert_logical,
            "native_tensor_payload_bpw_over_logical_parameters": tensor_payload * 8 / logical,
            "native_serialized_file_bpw_over_logical_parameters": serialized * 8 / logical,
            "canonical_target_payload_ceilings": rates,
            "four_bpw_transcode_required": tensor_payload > rates[0]["maximum_model_payload_bytes"],
        },
        "staging_contract": {
            "source_access": "read-only pread of exact tensor/expert byte ranges",
            "whole_shard_materialization": False,
            "whole_model_materialization": False,
            "recommended_experts_per_batch": 8,
            "expert_batch_units_per_rate": 36 * math.ceil(128 / 8),
            "dense_layer_units_per_rate": 36,
            "embedding_or_head_units_per_rate": 2,
            "expected_encode_units_per_rate": 36 * math.ceil(128 / 8) + 36 + 2,
            "expected_encode_units_all_ten_rates": 10 * (
                36 * math.ceil(128 / 8) + 36 + 2
            ),
            "maximum_recommended_expert_batch_bf16_bytes": 8 * 49_766_400,
            "staging_files_are_worker_owned_ephemera": True,
            "source_files_are_never_worker_owned": True,
            "required_reassembly_key": (
                "source shard sha256 + source tensor + expert ordinal + absolute byte ranges + "
                "canonical orientation + staging sha256 + STR2 sha256"
            ),
        },
        "tensor_inventory": {"tensor_count": len(tensor_rows),
                             "sha256": tensor_inventory_sha, "tensors": tensor_rows},
        "execution_readiness": {
            "header_and_logical_accounting": "ready",
            "bounded_memory_mxfp4_to_bf16_staging": "ready",
            "full_ten_rate_codec_execution": "blocked",
            "blockers": [
                "the current STRAND reader uses std::fs::read on a whole shard and rejects U8 tensors",
                "STR2 provenance currently binds a staging safetensors hash, not original MXFP4 byte ranges",
                "no reviewed subarchive reassembly manifest/loader maps per-expert archives back to GPT-OSS MoE execution",
                "no Apple-Silicon GPT-OSS STR2 MoE runtime/evaluator is present",
                "the local 120B source has no tokenizer files, so quality evaluation cannot run",
                "all ten target artifacts cannot be retained simultaneously under a 150 GB free-disk reserve",
            ],
            "safe_policy": "fail closed; no quality, deployability, or source-deletion claim",
        },
        "claims": {"quality": False, "deployable": False, "dominance": False,
                   "source_deletion": False},
    }
    doc["inventory_sha256"] = _hash_value(doc)
    return doc


def _artifact(path: Path) -> dict[str, Any]:
    digest, size = _hash_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def validate_inventory(doc: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["inventory is not an object"]
    digest = doc.get("inventory_sha256")
    payload = {key: value for key, value in doc.items() if key != "inventory_sha256"}
    if doc.get("schema") != INVENTORY_SCHEMA:
        errors.append("inventory schema mismatch")
    if not isinstance(digest, str) or digest != _hash_value(payload):
        errors.append("inventory hash mismatch")
    accounting = doc.get("parameter_accounting")
    if not isinstance(accounting, dict) \
            or accounting.get("logical_model_parameters") != 116_829_156_672 \
            or accounting.get("serialized_safetensors_elements_not_a_parameter_denominator") \
            != 63_081_444_672 \
            or accounting.get("ue8_scale_elements_side_information_not_parameters") \
            != 3_583_180_800:
        errors.append("logical parameter accounting mismatch")
    physical = doc.get("physical_accounting")
    if not isinstance(physical, dict) \
            or physical.get("tensor_payload_bytes") != 65_248_815_744:
        errors.append("physical payload accounting mismatch")
    tensors = doc.get("tensor_inventory")
    if not isinstance(tensors, dict) or tensors.get("tensor_count") != 543 \
            or not isinstance(tensors.get("tensors"), list) \
            or tensors.get("sha256") != _hash_value(tensors.get("tensors")):
        errors.append("tensor inventory mismatch")
    return errors


def build_logical_parameter_manifest(
    inspection: Inspection, *, inventory_path: Path = DEFAULT_INVENTORY,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build the generic Pass-B authority with GPT-OSS logical semantics.

    The resulting document intentionally uses the existing generic manifest
    schema so the Ultra orchestrator can consume it without treating MXFP4 U8
    storage elements as parameters.
    """
    raw_inventory_path = Path(inventory_path)
    if raw_inventory_path.is_symlink():
        raise Mxfp4Error("symlinked MXFP4 inventory is forbidden")
    inventory_path = raw_inventory_path.resolve(strict=True)
    inventory = _read_json(inventory_path)
    errors = validate_inventory(inventory)
    if errors:
        raise Mxfp4Error("MXFP4 inventory is invalid: " + "; ".join(errors))
    live = build_inventory(inspection)
    if live["tensor_inventory"]["sha256"] != inventory["tensor_inventory"]["sha256"] \
            or live["source_binding"]["source_manifest_sha256"] \
            != inventory["source_binding"]["source_manifest_sha256"]:
        raise Mxfp4Error("live source headers differ from the bound MXFP4 inventory")
    inventory_file_sha, inventory_file_bytes = _hash_file(inventory_path)
    census_sha, census_bytes = _hash_file(inspection.census_path)
    source_rows = []
    for ordinal, row in enumerate(inspection.census["source"]["shards"]):
        source_rows.append({
            "ordinal": ordinal, "name": row["name"], "bytes": row["bytes"],
            "sha256": row["file_sha256"], "tensor_count": row["tensor_count"],
            "data_bytes": row["data_bytes"],
        })
    authority = {
        "counting_unit": (
            "logical_model_weights; two E2M1 weights per MXFP4 block byte; "
            "UE8 scales excluded as side information"
        ),
        "exact_distinct_stored_parameter_count": 116_829_156_672,
        "stored_parameter_count": 116_829_156_672,
        "tensor_count": 543,
        "dtype_element_counts": {
            "MXFP4_E2M1_LOGICAL": 114_661_785_600,
            "BF16_LOGICAL": 2_167_371_072,
        },
        "ownership_class_counts": {
            "expert_weight": {"elements": 114_661_785_600, "tensors": 72},
            "expert_bias": {"elements": 39_813_120, "tensors": 72},
            "always_compute_dense": {"elements": 969_291_072, "tensors": 325},
            "embedding_table": {"elements": 579_133_440, "tensors": 1},
            "output_head": {"elements": 579_133_440, "tensors": 1},
        },
        "alias_storage_proven": False,
        "dense_active_parameter_count": None,
        "active_moe_parameter_count": 5_132_852_352,
        "authoritative_for_physical_bpw_denominator": True,
        "authoritative_for_active_moe_compute_denominator": True,
    }
    doc: dict[str, Any] = {
        "schema": PARAMETER_MANIFEST_SCHEMA,
        "created_at": _now(), "label": "120B", "hf_id": "openai/gpt-oss-120b",
        "census_path": str(inspection.census_path),
        "census_file_sha256": census_sha, "census_file_bytes": census_bytes,
        "census_report_sha256": inspection.census["report_sha256"],
        "source_manifest_sha256": inspection.census["source"]["source_manifest_sha256"],
        "model_dir": str(inspection.model_dir), "source_shards": source_rows,
        "parameter_authority": authority,
        "review_boundary": {
            "producer_role": "gptoss_mxfp4_logical_parameter_authority",
            "source": "completed_source_census_plus_header_validated_mxfp4_inventory",
            "inventory_path": str(inventory_path),
            "inventory_file_sha256": inventory_file_sha,
            "inventory_file_bytes": inventory_file_bytes,
            "inventory_sha256": inventory["inventory_sha256"],
            "tensor_inventory_sha256": inventory["tensor_inventory"]["sha256"],
            "serialized_safetensors_element_count": 63_081_444_672,
            "serialized_element_count_is_parameter_denominator": False,
            "ue8_scale_elements": 3_583_180_800,
            "ue8_scales_are_side_information": True,
            "active_moe_convention": (
                "4/128 expert weights+biases, always-compute dense tensors, one "
                "embedding row, full output head"
            ),
            "model_label_estimate_used": False, "quality_evidence": False,
            "source_deletion_authority": False,
        },
    }
    doc["manifest_sha256"] = _hash_value(doc)
    errors = validate_logical_parameter_manifest(doc)
    if errors:
        raise Mxfp4Error("built invalid logical parameter manifest: " + "; ".join(errors))
    if output_path is not None:
        _atomic_json(Path(output_path), doc)
    return doc


def validate_logical_parameter_manifest(
    doc: Any, *, verify_files: bool = False,
) -> list[str]:
    errors: list[str] = []
    required = {
        "schema", "created_at", "label", "hf_id", "census_path",
        "census_file_sha256", "census_file_bytes", "census_report_sha256",
        "source_manifest_sha256", "model_dir", "source_shards",
        "parameter_authority", "review_boundary", "manifest_sha256",
    }
    if not isinstance(doc, dict) or set(doc) != required:
        return ["logical parameter manifest keys are invalid"]
    if doc.get("schema") != PARAMETER_MANIFEST_SCHEMA or doc.get("label") != "120B" \
            or doc.get("hf_id") != "openai/gpt-oss-120b":
        errors.append("logical parameter manifest identity mismatch")
    payload = {key: value for key, value in doc.items() if key != "manifest_sha256"}
    if doc.get("manifest_sha256") != _hash_value(payload):
        errors.append("logical parameter manifest hash mismatch")
    for field in ("census_file_sha256", "census_report_sha256",
                  "source_manifest_sha256", "manifest_sha256"):
        if not isinstance(doc.get(field), str) or SHA_RE.fullmatch(doc[field]) is None:
            errors.append(f"invalid {field}")
    authority = doc.get("parameter_authority")
    expected_authority_keys = {
        "counting_unit", "exact_distinct_stored_parameter_count",
        "stored_parameter_count", "tensor_count", "dtype_element_counts",
        "ownership_class_counts", "alias_storage_proven",
        "dense_active_parameter_count", "active_moe_parameter_count",
        "authoritative_for_physical_bpw_denominator",
        "authoritative_for_active_moe_compute_denominator",
    }
    if not isinstance(authority, dict) or set(authority) != expected_authority_keys:
        errors.append("logical parameter authority keys are invalid")
    else:
        logical = authority.get("exact_distinct_stored_parameter_count")
        if logical != 116_829_156_672 or authority.get("stored_parameter_count") != logical:
            errors.append("logical denominator must be 116829156672, never serialized U8 elements")
        if logical == 63_081_444_672:
            errors.append("serialized safetensors element count is forbidden as denominator")
        if authority.get("dtype_element_counts") != {
                "MXFP4_E2M1_LOGICAL": 114_661_785_600,
                "BF16_LOGICAL": 2_167_371_072}:
            errors.append("logical dtype counts do not close")
        owners = authority.get("ownership_class_counts")
        if not isinstance(owners, dict) or sum(
                row.get("elements", -1) for row in owners.values()
                if isinstance(row, dict)) != 116_829_156_672:
            errors.append("logical ownership counts do not close")
        if authority.get("active_moe_parameter_count") != 5_132_852_352 \
                or authority.get("authoritative_for_physical_bpw_denominator") is not True \
                or authority.get("authoritative_for_active_moe_compute_denominator") is not True:
            errors.append("logical physical/active authority is incomplete")
    review = doc.get("review_boundary")
    if not isinstance(review, dict) \
            or review.get("serialized_safetensors_element_count") != 63_081_444_672 \
            or review.get("serialized_element_count_is_parameter_denominator") is not False \
            or review.get("ue8_scale_elements") != 3_583_180_800 \
            or review.get("ue8_scales_are_side_information") is not True \
            or review.get("quality_evidence") is not False \
            or review.get("source_deletion_authority") is not False \
            or not isinstance(review.get("inventory_file_sha256"), str) \
            or SHA_RE.fullmatch(review["inventory_file_sha256"]) is None:
        errors.append("MXFP4 review boundary is invalid")
    shards = doc.get("source_shards")
    if not isinstance(shards, list) or len(shards) != 7:
        errors.append("logical source shard registry must contain seven shards")
    else:
        for ordinal, row in enumerate(shards):
            if not isinstance(row, dict) or set(row) != {
                    "ordinal", "name", "bytes", "sha256", "tensor_count", "data_bytes"} \
                    or row.get("ordinal") != ordinal \
                    or not isinstance(row.get("sha256"), str) \
                    or SHA_RE.fullmatch(row["sha256"]) is None:
                errors.append(f"logical source shard row invalid: {ordinal}")
    if verify_files and not errors:
        try:
            census_path = Path(doc["census_path"])
            census_sha, census_bytes = _hash_file(census_path)
            if census_sha != doc["census_file_sha256"] \
                    or census_bytes != doc["census_file_bytes"]:
                errors.append("live census identity differs from logical manifest")
            inventory_path = Path(review["inventory_path"])
            inventory_sha, inventory_bytes = _hash_file(inventory_path)
            if inventory_sha != review["inventory_file_sha256"] \
                    or inventory_bytes != review["inventory_file_bytes"]:
                errors.append("live MXFP4 inventory file identity mismatch")
            inventory = _read_json(inventory_path)
            if inventory.get("inventory_sha256") != review["inventory_sha256"] \
                    or inventory.get("tensor_inventory", {}).get("sha256") \
                    != review["tensor_inventory_sha256"]:
                errors.append("live MXFP4 inventory semantic identity mismatch")
            inspection = inspect_model(Path(doc["model_dir"]), census_path)
            live = build_inventory(inspection)
            if live["tensor_inventory"]["sha256"] != review["tensor_inventory_sha256"]:
                errors.append("live GPT-OSS tensor headers differ from logical authority")
        except (OSError, Mxfp4Error, KeyError, TypeError) as exc:
            errors.append(f"logical manifest file verification failed: {exc}")
    return errors


def _pread_exact(path: Path, offset: int, size: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or offset < 0 or size < 0 \
                or offset + size > before.st_size:
            raise Mxfp4Error(f"unsafe source byte range: {path}@{offset}+{size}")
        value = os.pread(fd, size, offset)
        after = os.fstat(fd)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
        if len(value) != size or identity(before) != identity(after):
            raise Mxfp4Error(f"source changed or truncated during pread: {path}")
        return value
    finally:
        os.close(fd)


def decode_mxfp4_groups_bf16(blocks: Any, scales: Any) -> Any:
    """Decode ``[..., groups, packed_bytes]`` to BF16 bits.

    The returned NumPy ``uint16`` array has the same prefix and an interleaved
    last dimension of ``groups * packed_bytes * 2``.  It is byte-for-byte
    equivalent to the reviewed Transformers E2M1/UE8 conversion rounded to
    BF16, but keeps the canonical pre-transpose ``[out, in]`` orientation.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise Mxfp4Error("NumPy is required for MXFP4 staging") from exc
    blocks = np.asarray(blocks, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.uint8)
    if blocks.ndim < 2 or tuple(blocks.shape[:-1]) != tuple(scales.shape):
        raise Mxfp4Error("blocks/scales shapes do not satisfy MXFP4 pairing")
    lut = np.asarray(FP4_VALUES, dtype=np.float32)
    expanded = np.empty((*blocks.shape[:-1], blocks.shape[-1] * 2), dtype=np.float32)
    expanded[..., 0::2] = lut[blocks & np.uint8(0x0F)]
    expanded[..., 1::2] = lut[blocks >> np.uint8(4)]
    exponents = scales.astype(np.int32) - 127
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        np.ldexp(expanded, exponents[..., None], out=expanded)
    # PyTorch's reviewed BF16 reference conversion yields canonical quiet NaN
    # for ±0 multiplied by the reserved UE8 exponent 128 (serialized byte 255).
    # NumPy preserves zero, so match the reference explicitly before rounding.
    reserved = exponents[..., None] == 128
    reserved_zero = reserved & (expanded == 0)
    reserved_nonzero = reserved & ~reserved_zero
    if np.any(reserved_zero):
        expanded[reserved_zero] = np.float32("nan")
    if np.any(reserved_nonzero):
        expanded[reserved_nonzero] = np.copysign(
            np.float32("inf"), expanded[reserved_nonzero]
        )
    flat_shape = (*blocks.shape[:-2], blocks.shape[-2] * blocks.shape[-1] * 2)
    values = expanded.reshape(flat_shape)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2", copy=False)


def _tensor_expert_extent(tensor: TensorMeta, expert: int) -> tuple[int, int]:
    if expert < 0 or expert >= tensor.shape[0]:
        raise Mxfp4Error(f"expert ordinal outside tensor: {expert}")
    stride = tensor.nbytes // tensor.shape[0]
    if stride * tensor.shape[0] != tensor.nbytes:
        raise Mxfp4Error(f"tensor expert stride is not integral: {tensor.name}")
    return tensor.data_start + expert * stride, stride


def _staging_tensor_rows(layer: int, experts: Iterable[int]) \
        -> list[tuple[str, tuple[int, int], str, int]]:
    rows: list[tuple[str, tuple[int, int], str, int]] = []
    for expert in experts:
        for projection, shape in PROJECTION_SHAPES.items():
            out_features = shape[1]
            in_features = shape[2] * shape[3] * 2
            name = f"block.{layer}.mlp.{projection}_weight.expert.{expert:03d}"
            rows.append((name, (out_features, in_features), projection, expert))
    return rows


def _safetensors_header(rows: list[tuple[str, tuple[int, int], str, int]]) \
        -> tuple[bytes, dict[str, tuple[int, int]]]:
    header: dict[str, Any] = {}
    ranges: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, shape, _projection, _expert in rows:
        size = math.prod(shape) * 2
        header[name] = {"dtype": "BF16", "shape": list(shape),
                        "data_offsets": [cursor, cursor + size]}
        ranges[name] = (cursor, cursor + size)
        cursor += size
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    return encoded, ranges


def _write_projection_expert(handle: BinaryIO, blocks: TensorMeta, scales: TensorMeta,
                             expert: int, *, rows_per_decode: int = 128) \
        -> list[dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise Mxfp4Error("NumPy is required for MXFP4 staging") from exc
    block_offset, block_size = _tensor_expert_extent(blocks, expert)
    scale_offset, scale_size = _tensor_expert_extent(scales, expert)
    block_raw = _pread_exact(blocks.shard_path, block_offset, block_size)
    scale_raw = _pread_exact(scales.shard_path, scale_offset, scale_size)
    block_shape = blocks.shape[1:]
    scale_shape = scales.shape[1:]
    block_array = np.frombuffer(block_raw, dtype=np.uint8).reshape(block_shape)
    scale_array = np.frombuffer(scale_raw, dtype=np.uint8).reshape(scale_shape)
    written = 0
    for row0 in range(0, block_shape[0], rows_per_decode):
        row1 = min(row0 + rows_per_decode, block_shape[0])
        decoded = decode_mxfp4_groups_bf16(block_array[row0:row1],
                                            scale_array[row0:row1])
        data = decoded.tobytes(order="C")
        handle.write(data)
        written += len(data)
    expected = blocks.elements * 2 // blocks.shape[0] * 2
    if written != expected:
        raise Mxfp4Error(f"decoded byte count mismatch for {blocks.name} expert {expert}")
    return [
        {"role": "blocks", "tensor": blocks.name, "shard": blocks.shard_name,
         "source_shard_path": str(blocks.shard_path),
         "source_shard_sha256": None,
         "absolute_byte_range": [block_offset, block_offset + block_size],
         "bytes": block_size, "range_sha256": hashlib.sha256(block_raw).hexdigest()},
        {"role": "scales", "tensor": scales.name, "shard": scales.shard_name,
         "source_shard_path": str(scales.shard_path),
         "source_shard_sha256": None,
         "absolute_byte_range": [scale_offset, scale_offset + scale_size],
         "bytes": scale_size, "range_sha256": hashlib.sha256(scale_raw).hexdigest()},
    ]


def stage_expert_batch(inspection: Inspection, *, layer: int, expert_start: int,
                       expert_count: int, output: Path,
                       receipt_path: Path | None = None) -> dict[str, Any]:
    if expert_count <= 0 or expert_count > 16 or expert_start < 0 \
            or expert_start + expert_count > EXPECTED_CONFIG["num_experts"]:
        raise Mxfp4Error("expert batch must be a positive in-range slice of at most 16")
    raw_output = Path(output)
    if raw_output.is_symlink():
        raise Mxfp4Error(f"symlinked staging output is forbidden: {raw_output}")
    output = raw_output.resolve(strict=False)
    source_root = inspection.model_dir.resolve(strict=True)
    try:
        output.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise Mxfp4Error("worker-owned staging output must not be inside the source tree")
    if output.exists() or output.is_symlink():
        raise Mxfp4Error(f"refusing to overwrite staging artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    experts = list(range(expert_start, expert_start + expert_count))
    rows = _staging_tensor_rows(layer, experts)
    header, output_ranges = _safetensors_header(rows)
    pairs = {projection: _projection_pair(inspection, layer, projection)
             for projection in PROJECTION_SHAPES}
    census_by_basename = {
        Path(row["name"]).name: row for row in inspection.census["source"]["shards"]
    }
    tmp = output.with_name(f".{output.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    sources: list[dict[str, Any]] = []
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(tmp, flags, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(struct.pack("<Q", len(header)))
            handle.write(header)
            for name, shape, projection, expert in rows:
                before = handle.tell()
                blocks, scales = pairs[projection]
                extents = _write_projection_expert(handle, blocks, scales, expert)
                expected_start, expected_end = output_ranges[name]
                if before != 8 + len(header) + expected_start \
                        or handle.tell() != 8 + len(header) + expected_end:
                    raise Mxfp4Error(f"staging offset mismatch for {name}")
                for extent in extents:
                    extent["source_shard_sha256"] = census_by_basename[
                        extent["shard"]
                    ]["file_sha256"]
                sources.append({
                    "staging_tensor": name, "shape": list(shape),
                    "dtype": "BF16", "canonical_orientation": "out_features,in_features",
                    "layer": layer, "expert": expert, "projection": projection,
                    "source_extents": extents,
                })
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, output)
        _fsync_dir(output.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    artifact = _artifact(output)
    receipt: dict[str, Any] = {
        "schema": STAGING_RECEIPT_SCHEMA,
        "created_at": _now(), "status": "complete",
        "model": {"label": "120B", "family": "gpt-oss-moe"},
        "unit": {"layer": layer, "expert_start": expert_start,
                 "expert_count": expert_count, "tensor_count": len(rows)},
        "artifact": artifact,
        "logical_parameters": sum(math.prod(shape) for _name, shape, _p, _e in rows),
        "source_bindings": sources,
        "memory_contract": {
            "whole_shard_materialized": False, "whole_layer_materialized": False,
            "maximum_raw_source_slice_bytes": max(
                _tensor_expert_extent(pair[0], expert_start)[1]
                for pair in pairs.values()
            ),
            "maximum_simultaneous_block_plus_scale_bytes": max(
                _tensor_expert_extent(pair[0], expert_start)[1]
                + _tensor_expert_extent(pair[1], expert_start)[1]
                for pair in pairs.values()
            ),
            "decode_rows_per_chunk": 128,
            "conservative_python_process_peak_bytes": 256 * 1024 * 1024,
            "measured_process_rss_required_by_campaign_worker": True,
        },
        "lifecycle": {"worker_owned_ephemeral": True, "source_files_deleted": False,
                      "deletion_authority": "staging artifact only after downstream hash receipt"},
        "claims": {"quality": False, "deployable": False, "source_deletion": False},
    }
    receipt["receipt_sha256"] = _hash_value(receipt)
    if receipt_path is None:
        receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise Mxfp4Error(f"refusing to overwrite staging receipt: {receipt_path}")
    errors = validate_staging_receipt(receipt, verify_files=True)
    if errors:
        raise Mxfp4Error("generated staging receipt is invalid: " + "; ".join(errors))
    _atomic_json(receipt_path, receipt)
    return receipt


def validate_staging_receipt(doc: Any, *, verify_files: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict) or doc.get("schema") != STAGING_RECEIPT_SCHEMA:
        return ["staging receipt schema mismatch"]
    digest = doc.get("receipt_sha256")
    payload = {key: value for key, value in doc.items() if key != "receipt_sha256"}
    if not isinstance(digest, str) or digest != _hash_value(payload):
        errors.append("staging receipt hash mismatch")
    artifact = doc.get("artifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str) \
            or not isinstance(artifact.get("sha256"), str) \
            or not isinstance(artifact.get("bytes"), int):
        errors.append("staging artifact identity is invalid")
    bindings = doc.get("source_bindings")
    if not isinstance(bindings, list) or not bindings:
        errors.append("staging source bindings are missing")
    else:
        names: set[str] = set()
        for binding in bindings:
            if not isinstance(binding, dict) or not isinstance(
                    binding.get("staging_tensor"), str) \
                    or binding["staging_tensor"] in names:
                errors.append("staging tensor binding is invalid or duplicate")
                continue
            names.add(binding["staging_tensor"])
            extents = binding.get("source_extents")
            if not isinstance(extents, list) or len(extents) != 2:
                errors.append(f"source extents invalid: {binding['staging_tensor']}")
                continue
            for extent in extents:
                byte_range = extent.get("absolute_byte_range") \
                    if isinstance(extent, dict) else None
                if not isinstance(extent, dict) \
                        or extent.get("role") not in {"blocks", "scales"} \
                        or not isinstance(extent.get("source_shard_path"), str) \
                        or not isinstance(extent.get("source_shard_sha256"), str) \
                        or SHA_RE.fullmatch(extent["source_shard_sha256"]) is None \
                        or not isinstance(extent.get("range_sha256"), str) \
                        or SHA_RE.fullmatch(extent["range_sha256"]) is None \
                        or not isinstance(byte_range, list) or len(byte_range) != 2 \
                        or any(isinstance(value, bool) or not isinstance(value, int)
                               for value in byte_range) \
                        or byte_range[0] < 0 or byte_range[1] <= byte_range[0] \
                        or extent.get("bytes") != byte_range[1] - byte_range[0]:
                    errors.append(f"source range binding invalid: {binding['staging_tensor']}")
                    continue
                if verify_files:
                    try:
                        source_path = Path(extent["source_shard_path"])
                        if source_path.is_symlink():
                            raise Mxfp4Error("source range path is symlinked")
                        raw = _pread_exact(source_path, byte_range[0], extent["bytes"])
                        if hashlib.sha256(raw).hexdigest() != extent["range_sha256"]:
                            errors.append(
                                f"source range digest mismatch: {binding['staging_tensor']}"
                            )
                    except (OSError, Mxfp4Error) as exc:
                        errors.append(f"source range verification failed: {exc}")
    if verify_files and isinstance(artifact, dict) and isinstance(artifact.get("path"), str):
        try:
            observed_sha, observed_bytes = _hash_file(Path(artifact["path"]))
            if observed_sha != artifact.get("sha256") or observed_bytes != artifact.get("bytes"):
                errors.append("live staging artifact identity mismatch")
        except (OSError, Mxfp4Error) as exc:
            errors.append(f"staging artifact verification failed: {exc}")
    lifecycle = doc.get("lifecycle")
    if not isinstance(lifecycle, dict) or lifecycle.get("source_files_deleted") is not False:
        errors.append("staging source lifecycle boundary is invalid")
    return errors


def _selftest() -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise Mxfp4Error("NumPy is required for selftest") from exc
    blocks = np.asarray([[[0x10, 0x32]], [[0x98, 0xBA]]], dtype=np.uint8)
    scales = np.asarray([[127], [128]], dtype=np.uint8)
    decoded = decode_mxfp4_groups_bf16(blocks, scales)
    if decoded.shape != (2, 4) or decoded.dtype != np.dtype("<u2"):
        raise Mxfp4Error("synthetic decoder shape/dtype selftest failed")
    values = (decoded.astype(np.uint32) << np.uint32(16)).view(np.float32)
    expected = np.asarray([[0.0, 0.5, 1.0, 1.5], [-0.0, -1.0, -2.0, -3.0]],
                          dtype=np.float32)
    if not np.array_equal(values, expected) \
            or not np.array_equal(np.signbit(values), np.signbit(expected)):
        raise Mxfp4Error("synthetic E2M1/UE8 decoder selftest failed")
    print(json.dumps({"status": "ok", "schema": INVENTORY_SCHEMA,
                      "logical_parameters": 116_829_156_672}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    audit.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    audit.add_argument("--output", type=Path, default=DEFAULT_INVENTORY)
    verify = sub.add_parser("verify")
    verify.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    build_parameters = sub.add_parser("build-parameter-manifest")
    build_parameters.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    build_parameters.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    build_parameters.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    build_parameters.add_argument("--output", type=Path,
                                  default=DEFAULT_LOGICAL_PARAMETER_MANIFEST)
    validate_parameters = sub.add_parser("validate-parameter-manifest")
    validate_parameters.add_argument("--manifest", type=Path,
                                     default=DEFAULT_LOGICAL_PARAMETER_MANIFEST)
    validate_parameters.add_argument("--verify-files", action="store_true")
    stage = sub.add_parser("stage-experts")
    stage.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    stage.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    stage.add_argument("--layer", type=int, required=True)
    stage.add_argument("--expert-start", type=int, required=True)
    stage.add_argument("--expert-count", type=int, default=8)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--receipt", type=Path)
    sub.add_parser("selftest")
    args = parser.parse_args(argv)
    try:
        if args.command == "selftest":
            _selftest()
            return 0
        if args.command == "verify":
            errors = validate_inventory(_read_json(args.inventory))
            print(json.dumps({"status": "ok" if not errors else "invalid",
                              "errors": errors}, indent=2, sort_keys=True))
            return 0 if not errors else 2
        if args.command == "validate-parameter-manifest":
            errors = validate_logical_parameter_manifest(
                _read_json(args.manifest), verify_files=args.verify_files
            )
            print(json.dumps({"status": "ok" if not errors else "invalid",
                              "errors": errors}, indent=2, sort_keys=True))
            return 0 if not errors else 2
        inspection = inspect_model(args.model_dir, args.census)
        if args.command == "build-parameter-manifest":
            doc = build_logical_parameter_manifest(
                inspection, inventory_path=args.inventory, output_path=args.output
            )
            print(json.dumps({
                "status": "ok", "output": str(args.output.resolve()),
                "schema": doc["schema"], "manifest_sha256": doc["manifest_sha256"],
                "logical_parameters": doc["parameter_authority"][
                    "exact_distinct_stored_parameter_count"
                ], "active_moe_parameters": doc["parameter_authority"][
                    "active_moe_parameter_count"
                ],
            }, indent=2, sort_keys=True))
            return 0
        if args.command == "audit":
            doc = build_inventory(inspection)
            _atomic_json(args.output, doc)
            print(json.dumps({
                "status": "ok", "output": str(args.output.resolve()),
                "inventory_sha256": doc["inventory_sha256"],
                "logical_parameters": doc["parameter_accounting"]["logical_model_parameters"],
                "native_payload_bpw": doc["physical_accounting"][
                    "native_tensor_payload_bpw_over_logical_parameters"
                ], "full_codec_ready": False,
                "blockers": doc["execution_readiness"]["blockers"],
            }, indent=2, sort_keys=True))
            return 0
        receipt = stage_expert_batch(
            inspection, layer=args.layer, expert_start=args.expert_start,
            expert_count=args.expert_count, output=args.output,
            receipt_path=args.receipt,
        )
        print(json.dumps({"status": "ok", "artifact": receipt["artifact"],
                          "receipt_sha256": receipt["receipt_sha256"]},
                         indent=2, sort_keys=True))
        return 0
    except (Mxfp4Error, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

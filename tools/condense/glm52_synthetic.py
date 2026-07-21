#!/usr/bin/env python3.12
"""Deterministic architecture-preserving GLM-5.2 safetensors fixture.

The fixture is intentionally small in matrix dimensions, not in architectural
features.  It stores seven main layers (dense 0--2, sparse 3--6), the main
IndexShare pattern ``F,F,F,S,S,S,F``, and a separate physical layer 7 carrying
the sparse MTP block, its full indexer, and all four MTP-only tensors.  Every
sparse layer still has 256 routed experts, top-8 routing, and one shared expert.

``build_synthetic_fixture`` creates two independently indexed Hugging Face
checkpoint views:

* ``full``: two CORE shards plus a third MTP shard;
* ``main_only``: the same two immutable CORE shard payloads under a two-shard
  index, with no layer-7 names.

Both views are validated by :mod:`glm52_adapter` before the builder returns.
No network or teacher checkpoint access occurs.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from glm52_adapter import (
    BoundedSafetensorsReader,
    CORE,
    MTP,
    PROFILE_SYNTHETIC,
    SYNTHETIC_CONFIG_CONTRACT,
    SYNTHETIC_INDEXER_TYPES,
    VIEW_CORE,
    VIEW_FULL,
    Inventory,
    TensorSpec,
    expected_tensor_specs,
    schema_summary,
    validate_config,
    verify_checkpoint,
)


@dataclass(frozen=True)
class SyntheticFixture:
    root: Path
    full_dir: Path
    main_only_dir: Path
    config: Mapping[str, Any]
    index: Mapping[str, Any]
    main_only_index: Mapping[str, Any]
    metadata: Mapping[str, Any]
    full_inventory: Inventory
    main_only_inventory: Inventory

    def full_reader(self, *, max_tensor_bytes: int = 256 * 1024 * 1024) -> BoundedSafetensorsReader:
        return BoundedSafetensorsReader(self.full_inventory, max_tensor_bytes=max_tensor_bytes)

    def main_only_reader(
        self, *, max_tensor_bytes: int = 256 * 1024 * 1024
    ) -> BoundedSafetensorsReader:
        return BoundedSafetensorsReader(
            self.main_only_inventory, max_tensor_bytes=max_tensor_bytes
        )


def synthetic_config() -> dict[str, Any]:
    """Return a fresh copy of the one accepted synthetic architecture config."""
    config = json.loads(json.dumps(dict(SYNTHETIC_CONFIG_CONTRACT)))
    config["indexer_types"] = list(SYNTHETIC_INDEXER_TYPES)
    config["mlp_layer_types"] = ["dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse"]
    # Non-architectural fields make the fixture look like a normal local HF config.
    config.update({
        "eos_token_id": [60, 61, 62],
        "pad_token_id": 60,
        "initializer_range": 0.02,
        "rope_parameters": {"rope_theta": 10_000, "rope_type": "default"},
        "transformers_version": "synthetic-offline-fixture",
    })
    return config


def _float32_to_bf16_bytes(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype="<f4")
    bits = values.view(np.uint32)
    # Round-to-nearest-even, matching the conventional float32 -> BF16 cast.
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    words = ((bits + rounding) >> np.uint32(16)).astype("<u2")
    return words.tobytes(order="C")


def deterministic_values(spec: TensorSpec) -> np.ndarray:
    """Generate finite, reproducible values for one fixture tensor.

    Router rows are distinct positive binary fractions, so an all-positive
    hidden-state probe has 256 distinct scores without saturating sigmoid.
    Correction biases are also strictly increasing but small.  Other tensors
    use a name-keyed modular sequence with exactly representable binary
    fractions at initialization-like scale; this avoids random-library/version
    drift and prevents harmless BLAS-order differences from exploding across
    the seven-layer parity fixture.
    """
    size = spec.element_count
    if spec.organ == "router":
        rows = np.arange(1, spec.shape[0] + 1, dtype=np.float32)[:, None]
        return np.broadcast_to(rows / np.float32(256.0), spec.shape).copy()
    if spec.organ == "router_control":
        return np.arange(size, dtype=np.float32).reshape(spec.shape) / np.float32(65_536.0)

    digest = hashlib.sha256(spec.name.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "little")
    multiplier = np.uint64(2 * (seed % 113) + 1)
    offset = np.uint64((seed >> 9) % 251)
    sequence = np.arange(size, dtype=np.uint64)
    residues = ((sequence * multiplier + offset) % np.uint64(251)).astype(np.float32)
    centered = residues - np.float32(125.0)
    if spec.organ == "embeddings":
        values = centered / np.float32(256.0)
    elif spec.organ == "lm_head":
        values = centered / np.float32(512.0)
    else:
        values = centered / np.float32(8_192.0)
    if spec.organ in {"normalization", "mtp_normalization", "mtp_head_norm"}:
        values = np.float32(1.0) + values / np.float32(16.0)
    return values.reshape(spec.shape)


def tensor_payload(spec: TensorSpec) -> bytes:
    values = deterministic_values(spec)
    if spec.dtype == "BF16":
        payload = _float32_to_bf16_bytes(values)
    elif spec.dtype == "F32":
        payload = np.asarray(values, dtype="<f4").tobytes(order="C")
    else:
        raise ValueError(f"unsupported synthetic dtype: {spec.dtype}")
    if len(payload) != spec.byte_count:
        raise AssertionError(
            f"synthetic payload length drift for {spec.name}: {len(payload)} != {spec.byte_count}"
        )
    return payload


def _safetensors_bytes(specs: Mapping[str, TensorSpec]) -> tuple[bytes, int, str]:
    header: dict[str, Any] = {}
    payloads: list[bytes] = []
    cursor = 0
    for name in sorted(specs):
        spec = specs[name]
        payload = tensor_payload(spec)
        end = cursor + len(payload)
        header[name] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": [cursor, end],
        }
        payloads.append(payload)
        cursor = end
    encoded = json.dumps(
        header, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    body = struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads)
    return body, cursor, hashlib.sha256(body).hexdigest()


def _write_once(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"short write while creating {path}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_once(path: Path, value: Any) -> None:
    data = (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    _write_once(path, data)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partition(specs: Mapping[str, TensorSpec]) -> tuple[dict[str, TensorSpec], ...]:
    core_early: dict[str, TensorSpec] = {}
    core_late: dict[str, TensorSpec] = {}
    mtp: dict[str, TensorSpec] = {}
    for name, spec in specs.items():
        if spec.section == MTP:
            mtp[name] = spec
        elif spec.layer is None or spec.layer <= 2:
            core_early[name] = spec
        else:
            core_late[name] = spec
    if not core_early or not core_late or not mtp:
        raise AssertionError("synthetic three-way shard partition unexpectedly empty")
    return core_early, core_late, mtp


def _index_for_parts(
    parts: tuple[Mapping[str, TensorSpec], ...], shard_names: tuple[str, ...]
) -> dict[str, Any]:
    if len(parts) != len(shard_names):
        raise AssertionError("part/shard arity mismatch")
    weight_map: dict[str, str] = {}
    total_size = 0
    for part, shard in zip(parts, shard_names, strict=True):
        for name, spec in part.items():
            if name in weight_map:
                raise AssertionError(f"duplicate fixture tensor assignment: {name}")
            weight_map[name] = shard
            total_size += spec.byte_count
    return {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}


def build_synthetic_fixture(root: Path) -> SyntheticFixture:
    """Build and validate the deterministic full and main-only fixture views.

    ``root`` must not already exist.  Refusing replacement prevents a test or
    controller from silently overwriting evidence from a prior run.
    """
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(f"synthetic fixture root already exists: {root}")
    root.mkdir(parents=True)
    full_dir = root / "full"
    main_only_dir = root / "main_only"
    full_dir.mkdir()
    main_only_dir.mkdir()

    config = synthetic_config()
    geometry = validate_config(config, profile=PROFILE_SYNTHETIC)
    specs = expected_tensor_specs(geometry, view=VIEW_FULL)
    core_early, core_late, mtp = _partition(specs)

    full_shards = (
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    )
    main_shards = (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    shard_receipts: list[dict[str, Any]] = []
    for part, name in zip((core_early, core_late, mtp), full_shards, strict=True):
        body, payload_bytes, digest = _safetensors_bytes(part)
        path = full_dir / name
        _write_once(path, body)
        shard_receipts.append({
            "name": name,
            "section": MTP if part is mtp else CORE,
            "tensor_count": len(part),
            "payload_bytes": payload_bytes,
            "file_bytes": len(body),
            "sha256": digest,
        })

    index = _index_for_parts((core_early, core_late, mtp), full_shards)
    main_index = _index_for_parts((core_early, core_late), main_shards)
    for directory in (full_dir, main_only_dir):
        _write_json_once(directory / "config.json", config)
    _write_json_once(full_dir / "model.safetensors.index.json", index)
    _write_json_once(main_only_dir / "model.safetensors.index.json", main_index)

    # CORE shard files are immutable hard links: the filtered view introduces no
    # transformed payload and cannot accidentally acquire an MTP tensor.
    for source_name, destination_name in zip(full_shards[:2], main_shards, strict=True):
        os.link(full_dir / source_name, main_only_dir / destination_name)

    full_inventory = verify_checkpoint(
        full_dir, profile=PROFILE_SYNTHETIC, view=VIEW_FULL
    )
    main_inventory = verify_checkpoint(
        main_only_dir, profile=PROFILE_SYNTHETIC, view=VIEW_CORE
    )
    if full_inventory.core_names != main_inventory.core_names:
        raise AssertionError("filtered main-only view does not exactly equal full CORE view")
    if main_inventory.mtp_names:
        raise AssertionError("filtered main-only view contains MTP tensors")

    metadata = {
        "schema": "hawking.glm52.synthetic_fixture.v1",
        "deterministic": True,
        "network_access": False,
        "profile": PROFILE_SYNTHETIC,
        "architecture": schema_summary(geometry),
        "views": {
            "full": {
                "directory": "full",
                "tensor_count": full_inventory.tensor_count,
                "payload_bytes": full_inventory.payload_bytes,
                "shard_count": len(full_inventory.shards),
                "index_sha256": _sha256(full_dir / "model.safetensors.index.json"),
            },
            "main_only": {
                "directory": "main_only",
                "tensor_count": main_inventory.tensor_count,
                "payload_bytes": main_inventory.payload_bytes,
                "shard_count": len(main_inventory.shards),
                "index_sha256": _sha256(main_only_dir / "model.safetensors.index.json"),
                "is_exact_core_filter": True,
            },
        },
        "shards": shard_receipts,
        "routing_fixture": {
            "router_rows": "constant positive BF16 fractions 1/256..256/256",
            "all_positive_hidden_probe_is_tie_free": True,
            "correction_bias_is_strictly_increasing_f32": True,
        },
        "long_context_indexer_contract": {
            "maximum_positions": int(config["max_position_embeddings"]),
            "shape_probe_required": True,
            "capability_claimed": False,
        },
    }
    _write_json_once(root / "fixture_metadata.json", metadata)
    return SyntheticFixture(
        root=root,
        full_dir=full_dir,
        main_only_dir=main_only_dir,
        config=config,
        index=index,
        main_only_index=main_index,
        metadata=metadata,
        full_inventory=full_inventory,
        main_only_inventory=main_inventory,
    )


__all__ = [
    "SyntheticFixture",
    "build_synthetic_fixture",
    "deterministic_values",
    "synthetic_config",
    "tensor_payload",
]

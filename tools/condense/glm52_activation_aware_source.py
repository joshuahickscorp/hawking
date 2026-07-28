#!/usr/bin/env python3.12
"""Numpy GLM tensor source over activation-aware ``.aap`` shards.

Input-side tensors execute ``L @ (B.T @ x)`` and output-side tensors execute
``B @ (L @ x)``.  The full matrix is therefore never reconstructed on the
ordinary forward path.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from activation_aware_format import (
    BASIS_MAGIC,
    HEADER_BYTES,
    ActivationAwareFormatError,
    ActivationAwareShard,
    sha256_file,
)
from glm52_activation_aware_pack import deserialize_tensor_payload


INDEX_NAME = "model.activation_aware.index.json"


def _decode_pass_through(
    blob: bytes, dtype: str, shape: list[int]
) -> np.ndarray:
    if blob[:8] != b"GLM52PT0":
        raise ActivationAwareFormatError("not a pass-through payload")
    ndim, rows, cols = struct.unpack_from("<III", blob, 8)
    expected_shape = list(shape)
    if int(ndim) != len(expected_shape):
        raise ActivationAwareFormatError(
            f"pass-through ndim {ndim} != manifest shape {expected_shape}"
        )
    if expected_shape and int(rows) != expected_shape[0]:
        raise ActivationAwareFormatError(
            f"pass-through rows {rows} != manifest shape {expected_shape}"
        )
    if len(expected_shape) > 1 and int(cols) != expected_shape[1]:
        raise ActivationAwareFormatError(
            f"pass-through cols {cols} != manifest shape {expected_shape}"
        )
    raw = memoryview(blob)[HEADER_BYTES:]
    if dtype in ("BF16", "BFLOAT16"):
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        arr = (u16 << np.uint32(16)).view(np.float32)
    elif dtype in ("F16", "FLOAT16"):
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype in ("F32", "FLOAT32"):
        arr = np.frombuffer(raw, dtype="<f4")
    else:
        raise ActivationAwareFormatError(
            f"unsupported pass-through dtype {dtype!r}"
        )
    expected = int(np.prod(expected_shape, dtype=np.int64))
    if arr.size != expected:
        raise ActivationAwareFormatError(
            f"pass-through has {arr.size} values, expected {expected}"
        )
    return arr.reshape(expected_shape).astype(np.float32, copy=False)


def _decode_basis(blob: bytes) -> np.ndarray:
    if blob[:8] != BASIS_MAGIC:
        raise ActivationAwareFormatError("not a shared-basis payload")
    hidden, rank = struct.unpack_from("<II", blob, 8)
    expected_bytes = HEADER_BYTES + int(hidden) * int(rank) * 2
    if len(blob) != expected_bytes:
        raise ActivationAwareFormatError(
            f"basis has {len(blob)} bytes, expected {expected_bytes}"
        )
    return (
        np.frombuffer(blob, dtype="<f2", offset=HEADER_BYTES)
        .astype(np.float32)
        .reshape(int(hidden), int(rank))
    )


class ActivationAwareGlmSource:
    """``TensorSource``/``matvec``/``rows`` surface for the GLM reference."""

    def __init__(
        self,
        shard_dir: Path,
        *,
        index_json: Path | None = None,
        verify_hash: bool = True,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        index_json = Path(index_json or self.shard_dir / INDEX_NAME)
        self.manifest = json.loads(index_json.read_text())
        if self.manifest.get("schema") != "hawking.activation_aware.model_index.v1":
            raise ActivationAwareFormatError(
                f"{index_json.name}: unsupported schema {self.manifest.get('schema')!r}"
            )
        self._weight_map = dict(self.manifest["weight_map"])
        self._dtypes = dict(self.manifest["tensor_dtypes"])
        self._shard_hashes = dict(self.manifest.get("shard_sha256", {}))
        self._verify_hash = bool(verify_hash)
        self._verified_shards: set[str] = set()
        self._shards: dict[str, ActivationAwareShard] = {}
        self._basis_cache: dict[tuple[str, int], np.ndarray] = {}

    def _shard_for(self, name: str) -> tuple[str, ActivationAwareShard]:
        try:
            filename = self._weight_map[name]
        except KeyError:
            raise KeyError(f"no tensor named {name!r} in the assembled model") from None
        shard = self._shards.get(filename)
        if shard is None:
            path = self.shard_dir / filename
            if self._verify_hash and filename not in self._verified_shards:
                expected = self._shard_hashes.get(filename)
                if not expected:
                    raise ActivationAwareFormatError(
                        f"{filename}: manifest has no shard SHA-256"
                    )
                observed = sha256_file(path)
                if observed != expected:
                    raise ActivationAwareFormatError(
                        f"{filename}: SHA-256 mismatch: expected {expected}, got {observed}"
                    )
                self._verified_shards.add(filename)
            shard = ActivationAwareShard(path)
            self._shards[filename] = shard
        return filename, shard

    def _entry_blob(
        self, name: str
    ) -> tuple[str, ActivationAwareShard, dict[str, Any], bytes]:
        filename, shard = self._shard_for(name)
        entry = shard.tensor_entry(name)
        return filename, shard, entry, shard.read_tensor(name)

    def _basis(
        self, filename: str, shard: ActivationAwareShard, layer: int, rank: int
    ) -> np.ndarray:
        key = (filename, int(layer))
        basis = self._basis_cache.get(key)
        if basis is None:
            basis = _decode_basis(shard.read_basis(layer))
            self._basis_cache[key] = basis
        if rank > basis.shape[1]:
            raise ActivationAwareFormatError(
                f"{filename}: tensor rank {rank} exceeds basis rank {basis.shape[1]}"
            )
        return basis[:, :rank]

    def _decode(
        self, name: str
    ) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
        filename, shard, entry, blob = self._entry_blob(name)
        if entry.get("disposition") == "pass_through":
            dtype = self._dtypes.get(name)
            if not dtype:
                raise ActivationAwareFormatError(
                    f"{name}: assembled manifest has no source dtype"
                )
            return entry, _decode_pass_through(blob, dtype, entry["shape"]), None
        decoded = deserialize_tensor_payload(blob)
        if decoded["has_basis"]:
            basis = decoded["B"]
        else:
            basis = self._basis(
                filename,
                shard,
                int(decoded["basis_layer"]),
                int(decoded["rank"]),
            )
        if list(entry["shape"]) != [int(decoded["rows"]), int(decoded["cols"])]:
            raise ActivationAwareFormatError(
                f"{name}: payload shape {[decoded['rows'], decoded['cols']]} "
                f"!= index shape {entry['shape']}"
            )
        return decoded, decoded["L"], basis

    def contains(self, name: str) -> bool:
        return name in self._weight_map

    def tensor(self, name: str) -> np.ndarray:
        meta, left, basis = self._decode(name)
        if basis is None:
            assert left is not None
            return left
        assert left is not None
        if meta["side"] == "input":
            return np.ascontiguousarray(left @ basis.T, dtype=np.float32)
        return np.ascontiguousarray(basis @ left, dtype=np.float32)

    def matvec(self, name: str, x: np.ndarray) -> np.ndarray:
        meta, left, basis = self._decode(name)
        x = np.asarray(x, dtype=np.float32)
        if basis is None:
            assert left is not None
            return np.asarray(left @ x, dtype=np.float32)
        assert left is not None
        if meta["side"] == "input":
            return np.asarray(left @ (basis.T @ x), dtype=np.float32)
        return np.asarray(basis @ (left @ x), dtype=np.float32)

    def rows(self, name: str, ids: np.ndarray) -> np.ndarray:
        ids_arr = np.asarray(ids, dtype=np.int64)
        meta, left, basis = self._decode(name)
        if basis is None:
            assert left is not None
            return left[ids_arr]
        assert left is not None
        flat = ids_arr.ravel()
        if meta["side"] == "input":
            selected = left[flat] @ basis.T
        else:
            selected = basis[flat] @ left
        return np.asarray(
            selected.reshape(*ids_arr.shape, selected.shape[-1]),
            dtype=np.float32,
        )

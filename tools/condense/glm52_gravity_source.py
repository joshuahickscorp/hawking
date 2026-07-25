#!/usr/bin/env python3.12
"""An efficient `glm52_reference.py` source over real `.gravity` bytes.

`glm52_gravity_fixture.py`'s `GravityTensors` densifies every packed tensor by calling
`pq_execute` once per output *column*, one-hot at a time. That is fine at the fixture's
scale (hidden=64) and would not finish at the flagship's: `lm_head.weight` alone is
[154880, 6144], and a routed-expert layer calls three weight tensors per hit expert, up
to 8 times per layer across 78 layers -- 1,872 tensor reads for one token if each one is
densified first.

This does the same job `crates/hawking-core/src/gravity_llama.rs::GravityWeights` does in
Rust: read a packed payload once, then execute a REAL matvec against it
(`gravity_forge.pq_execute` with the actual `x`), never materializing a `[rows, cols]`
array nobody asked for. `.rows()` decodes only the requested embedding rows from their own
chunk codes, the same way `PqTensor::row` does. `glm52_reference.py`'s `_linear` helper
already knows to prefer `.matvec()` when a source offers one, so plugging this in changes
nothing about what is computed -- only how many bytes get touched to compute it.

Works over one shard (grading the tiny semantic fixture) or the full 282-shard assembled
model via its `model.gravity.index.json` weight map, so the SAME class is what proves the
fast path correct on the fixture and what the flagship oracle run uses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_pack as pack  # noqa: E402
import gravity_format as gravity  # noqa: E402
import gravity_forge as forge  # noqa: E402


def _decode_native(blob: bytes, codec: str, shape: list[int]) -> np.ndarray:
    dtype = codec.split(".", 1)[1]
    if dtype == "bf16":
        u16 = np.frombuffer(blob, dtype=np.uint16).astype(np.uint32) << 16
        arr = u16.view(np.float32)
    elif dtype == "f16":
        arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
    else:
        arr = np.frombuffer(blob, dtype=np.float32)
    return arr.reshape(shape).astype(np.float32, copy=False)


def _pq_rows(codes: dict[str, Any], ids: np.ndarray) -> np.ndarray:
    """Decode only `ids`' rows, straight from their own chunk codes.

    Mirrors `gravity_llama_reference.py::GravityWeights.row` and the Rust
    `PqTensor::row`: O(len(ids) * nchunk) work, independent of total rows --
    the property that makes an embedding lookup cheap instead of a 3.8 GB
    reconstruction of a table 99.998% of which is irrelevant to this call.
    """
    D, S, sub = codes["D"], codes["S"], codes["sub"]
    nchunk = codes["nchunk"]
    indices = codes["indices"]  # [rows*nchunk, S]
    codebooks = codes["codebooks"]  # S x [card, sub]
    ids = np.asarray(ids, dtype=np.int64).ravel()
    out = np.empty((len(ids), nchunk * D), dtype=np.float32)
    for i, r in enumerate(ids):
        base = int(r) * nchunk
        for c in range(nchunk):
            for s in range(S):
                code = int(indices[base + c, s])
                out[i, c * D + s * sub: c * D + (s + 1) * sub] = codebooks[s][code]
    return out


class GravityGlmSource:
    """`TensorSource` + `.matvec()` + `.rows()` over one or many `.gravity` shards.

    `verify_hash` is on by default: this class exists to BE ground truth, and a
    silently corrupt payload producing confident wrong logits is the one failure
    mode nothing downstream would catch.
    """

    def __init__(self, shard_dir: Path, *, index_json: Path | None = None,
                single_shard: str | None = None, verify_hash: bool = True) -> None:
        self.shard_dir = Path(shard_dir)
        self.verify_hash = verify_hash
        self._headers: dict[str, dict] = {}
        self._descriptors: dict[str, dict] = {}
        if single_shard is not None:
            self._weight_map: dict[str, str] | None = None
            self._single_shard = single_shard
        elif index_json is not None:
            self._weight_map = json.loads(Path(index_json).read_text())["weight_map"]
            self._single_shard = None
        else:
            raise ValueError("GravityGlmSource needs either single_shard or index_json")

    # -- location -----------------------------------------------------

    def _shard_for(self, name: str) -> str:
        if self._single_shard is not None:
            return self._single_shard
        try:
            return self._weight_map[name]
        except KeyError:
            raise KeyError(f"no tensor named {name!r} in the assembled model") from None

    def _header(self, shard: str) -> dict:
        h = self._headers.get(shard)
        if h is None:
            h = gravity.read_header(self.shard_dir / shard)
            self._headers[shard] = h
        return h

    def _descriptor(self, name: str) -> tuple[str, dict]:
        d = self._descriptors.get(name)
        if d is not None:
            return d
        shard = self._shard_for(name)
        header = self._header(shard)
        entry = next((t for t in header["tensors"] if t["name"] == name), None)
        if entry is None:
            raise KeyError(f"{shard}: no tensor named {name!r}")
        d = (shard, entry)
        self._descriptors[name] = d
        return d

    def _payload(self, name: str) -> bytes:
        shard, _ = self._descriptor(name)
        return gravity.read_tensor(self.shard_dir / shard, name, verify_hash=self.verify_hash)

    def contains(self, name: str) -> bool:
        try:
            self._descriptor(name)
            return True
        except KeyError:
            return False

    # -- TensorSource / _linear / embedding surface --------------------

    def tensor(self, name: str) -> np.ndarray:
        """Small/moderate weights only (norm affine, router gate, indexer
        projections) -- densifies via ONE vectorized batched matvec against
        an identity, not a python loop of single-column calls."""
        _shard, entry = self._descriptor(name)
        blob = self._payload(name)
        codec = entry["codec"]
        if codec.startswith("native."):
            return _decode_native(blob, codec, entry["shape"])
        artifact = pack.load_artifact(blob)
        cols = int(entry["shape"][1])
        dense = forge.pq_execute(artifact, np.eye(cols, dtype=np.float32))  # [rows, cols]
        return np.ascontiguousarray(dense, dtype=np.float32)

    def matvec(self, name: str, x: np.ndarray) -> np.ndarray:
        """`W @ x`, `x` shape `[cols, B]`. Never materializes `W`."""
        _shard, entry = self._descriptor(name)
        blob = self._payload(name)
        codec = entry["codec"]
        if codec.startswith("native."):
            w = _decode_native(blob, codec, entry["shape"])
            return (w @ x).astype(np.float32)
        artifact = pack.load_artifact(blob)
        return forge.pq_execute(artifact, x)

    def rows(self, name: str, ids: np.ndarray) -> np.ndarray:
        """`weight[ids]` for arbitrary-shaped `ids` (e.g. `[B, S]`), preserving
        `ids.shape + (hidden,)` the way numpy fancy indexing would.
        """
        ids_arr = np.asarray(ids, dtype=np.int64)
        _shard, entry = self._descriptor(name)
        blob = self._payload(name)
        codec = entry["codec"]
        if codec.startswith("native."):
            arr = _decode_native(blob, codec, entry["shape"])
            return arr[ids_arr]
        artifact = pack.load_artifact(blob)
        flat = _pq_rows(artifact.config["pq_codes"], ids_arr.ravel())
        return flat.reshape(*ids_arr.shape, flat.shape[-1])

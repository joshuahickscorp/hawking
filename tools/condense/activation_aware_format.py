#!/usr/bin/env python3.12
"""Reader and integrity checks for GLM-5.2 activation-aware ``.aap`` shards.

The activation-aware packer deliberately writes a streaming-native format:

    u64 index_bytes | JSON index | shared basis blobs | tensor payload blobs

Offsets in the JSON index are relative to the first byte after the JSON.  This
module is the single bounds-checked implementation of that contract used by
assembly and the numpy runtime.  It never reconstructs a full weight matrix.
"""
from __future__ import annotations


# --- archive path fixup (lane A1): resolve roots as if still in tools/condense/ ---
import sys as _sys_a1
from pathlib import Path as _Path_a1
_A1_HERE = _Path_a1(__file__).resolve().parent
_A1_CONDENSE = _A1_HERE.parent if _A1_HERE.name == "archive" else _A1_HERE
_A1_REPO = _A1_CONDENSE.parents[1]  # repo root (condense -> tools -> repo)
if str(_A1_CONDENSE) not in _sys_a1.path:
    _sys_a1.path.insert(0, str(_A1_CONDENSE))
# --- end archive path fixup ---
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


SCHEMA = "hawking.glm52.activation_aware_pack.v1"
MAX_INDEX_BYTES = 256 * 1024 * 1024
TENSOR_MAGIC = b"GLM52AAP"
PASS_THROUGH_MAGIC = b"GLM52PT0"
BASIS_MAGIC = b"GLM52BAS"
HEADER_BYTES = 64


class ActivationAwareFormatError(RuntimeError):
    """An activation-aware shard is malformed or internally inconsistent."""


def _read_exact(handle, n: int, what: str) -> bytes:
    data = handle.read(n)
    if len(data) != n:
        raise ActivationAwareFormatError(
            f"{what}: truncated (read {len(data)} bytes, expected {n})"
        )
    return data


def read_index(path: Path) -> tuple[dict[str, Any], int]:
    """Return the validated JSON index and absolute payload-body offset."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw_len = _read_exact(handle, 8, f"{path.name} index length")
        index_bytes = struct.unpack("<Q", raw_len)[0]
        if index_bytes == 0 or index_bytes > MAX_INDEX_BYTES:
            raise ActivationAwareFormatError(
                f"{path.name}: invalid index length {index_bytes}"
            )
        raw = _read_exact(handle, index_bytes, f"{path.name} index")
    try:
        index = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActivationAwareFormatError(
            f"{path.name}: invalid index JSON: {exc}"
        ) from exc
    if index.get("schema") != SCHEMA:
        raise ActivationAwareFormatError(
            f"{path.name}: schema {index.get('schema')!r}, expected {SCHEMA!r}"
        )
    if not isinstance(index.get("bases"), list) or not isinstance(
        index.get("tensors"), list
    ):
        raise ActivationAwareFormatError(
            f"{path.name}: index must contain bases and tensors arrays"
        )

    body_offset = 8 + index_bytes
    body_bytes = size - body_offset
    names: set[str] = set()
    basis_layers: set[int] = set()
    spans: list[tuple[int, int, str]] = []
    for kind, entries in (("basis", index["bases"]), ("tensor", index["tensors"])):
        for entry in entries:
            if not isinstance(entry, dict):
                raise ActivationAwareFormatError(
                    f"{path.name}: {kind} entry is not an object"
                )
            offset = int(entry.get("offset", -1))
            nbytes = int(entry.get("bytes", 0))
            label = (
                str(entry.get("basis_layer"))
                if kind == "basis"
                else str(entry.get("name"))
            )
            if offset < 0 or nbytes < HEADER_BYTES or offset + nbytes > body_bytes:
                raise ActivationAwareFormatError(
                    f"{path.name}: {kind} {label!r} span "
                    f"[{offset}, {offset + nbytes}) is outside {body_bytes}-byte body"
                )
            spans.append((offset, offset + nbytes, f"{kind}:{label}"))
            if kind == "tensor":
                if not label or label == "None" or label in names:
                    raise ActivationAwareFormatError(
                        f"{path.name}: missing or duplicate tensor name {label!r}"
                    )
                names.add(label)
            else:
                layer = int(entry.get("basis_layer", -1))
                if layer < 0 or layer in basis_layers:
                    raise ActivationAwareFormatError(
                        f"{path.name}: missing or duplicate basis layer {layer}"
                    )
                basis_layers.add(layer)

    spans.sort()
    if spans and spans[0][0] != 0:
        raise ActivationAwareFormatError(
            f"{path.name}: first payload begins at {spans[0][0]}, not zero"
        )
    for (_, previous_end, previous), (start, _, current) in zip(spans, spans[1:]):
        if start != previous_end:
            raise ActivationAwareFormatError(
                f"{path.name}: non-contiguous payloads {previous!r} and {current!r} "
                f"({previous_end} != {start})"
            )
    if spans and spans[-1][1] != body_bytes:
        raise ActivationAwareFormatError(
            f"{path.name}: indexed payloads end at {spans[-1][1]}, "
            f"physical body ends at {body_bytes}"
        )
    return index, body_offset


def read_span(path: Path, body_offset: int, entry: dict[str, Any]) -> bytes:
    """Read one bounds-checked basis or tensor span from an indexed shard."""
    offset = int(entry["offset"])
    nbytes = int(entry["bytes"])
    with Path(path).open("rb") as handle:
        handle.seek(body_offset + offset)
        return _read_exact(
            handle,
            nbytes,
            f"{Path(path).name}:{entry.get('name', entry.get('basis_layer'))}",
        )


def validate_payload_magics(
    path: Path, index: dict[str, Any], body_offset: int
) -> None:
    """Check every payload's type magic without reading its large body."""
    with Path(path).open("rb") as handle:
        for basis in index["bases"]:
            handle.seek(body_offset + int(basis["offset"]))
            magic = _read_exact(handle, 8, f"{Path(path).name} basis magic")
            if magic != BASIS_MAGIC:
                raise ActivationAwareFormatError(
                    f"{Path(path).name}: bad basis magic {magic!r}"
                )
        for tensor in index["tensors"]:
            handle.seek(body_offset + int(tensor["offset"]))
            magic = _read_exact(handle, 8, f"{Path(path).name} tensor magic")
            expected = (
                PASS_THROUGH_MAGIC
                if tensor.get("disposition") == "pass_through"
                else TENSOR_MAGIC
            )
            if magic != expected:
                raise ActivationAwareFormatError(
                    f"{Path(path).name}:{tensor.get('name')}: "
                    f"magic {magic!r}, expected {expected!r}"
                )


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


class ActivationAwareShard:
    """Lazy, seek-based view of one ``.aap`` shard."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.index, self.body_offset = read_index(self.path)
        self._tensors = {row["name"]: row for row in self.index["tensors"]}
        self._bases = {
            int(row["basis_layer"]): row for row in self.index["bases"]
        }

    def tensor_names(self) -> list[str]:
        return list(self._tensors)

    def tensor_entry(self, name: str) -> dict[str, Any]:
        try:
            return self._tensors[name]
        except KeyError:
            raise KeyError(f"{self.path.name}: no tensor named {name!r}") from None

    def basis_entry(self, layer: int) -> dict[str, Any]:
        try:
            return self._bases[int(layer)]
        except KeyError:
            raise KeyError(
                f"{self.path.name}: no shared basis for layer {layer}"
            ) from None

    def read_tensor(self, name: str) -> bytes:
        return read_span(self.path, self.body_offset, self.tensor_entry(name))

    def read_basis(self, layer: int) -> bytes:
        return read_span(self.path, self.body_offset, self.basis_entry(layer))

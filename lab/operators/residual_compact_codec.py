"""Compact sparse-residual encodings for the binary + outlier family.

The incumbent ``_residual_codec`` stores a uint32 index plus an fp16 value
per outlier (48 bits). This module keeps the *same* operation — a binary
base plus a sparse additive correction at the *same* top-|residual|
positions — and only changes how those positions and values are stored.

Index modes (measured, not assumed):

- ``rice``: sorted global indices as uint32 first + Rice-coded deltas
- ``group_local``: per-group counts + log2(group_size)-bit local indices
- ``bitmap``: group occupancy + per-occupied-group membership mask

Value modes:

- 16-bit: fp16, bit-identical reconstruction to the incumbent residual
- 2..8-bit: signed uniform with a single stored scale (absmax by default)
- 1-bit: sign * stored scale (mean_abs / rms / absmax); no zero code,
  because a selected outlier is never stored as zero

Selection is copied from ``_residual_codec`` and must stay identical.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from lab.operators.ascension_dual_gravity_worker import (
    GROUP_BINARY,
    CodecResult,
    DualGravityError,
    _binary_parts,
    _container,
    _pack_unsigned,
    _packed_byte_count,
    _parse_container,
    _unpack_unsigned,
)


MAGIC_RESIDUAL_COMPACT = b"HGRAVR02"
SCHEMA_RESIDUAL_COMPACT = "hawking.gravity.binary_outlier_residual.v2"
INDEX_MODES = ("rice", "group_local", "bitmap")
VALUE_SCALES = ("fp16", "absmax", "mean_abs", "rms")


def select_outlier_indices(residual: np.ndarray, outlier_ratio: float) -> tuple[np.ndarray, int]:
    """Identical selection to ``_residual_codec``: global top-k by |residual|."""

    if not 0.0 < outlier_ratio <= 0.1:
        raise DualGravityError("outlier residual ratio must be in (0, 0.1]")
    flat = np.ascontiguousarray(residual, dtype=np.float32).reshape(-1)
    count = max(1, int(math.ceil(flat.size * outlier_ratio)))
    indices = np.argpartition(np.abs(flat), -count)[-count:].astype("<u4")
    indices.sort()
    return indices, count


def default_value_scale(value_bits: int) -> str:
    if value_bits == 16:
        return "fp16"
    if value_bits == 1:
        return "rms"
    return "absmax"


class _BitWriter:
    """LSB-first bitstream, matching ``np.packbits(..., bitorder='little')``."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._acc = 0
        self._filled = 0

    def write_bit(self, bit: int) -> None:
        self._acc |= (int(bit) & 1) << self._filled
        self._filled += 1
        if self._filled == 8:
            self._buf.append(self._acc)
            self._acc = 0
            self._filled = 0

    def write_ones(self, count: int) -> None:
        n = int(count)
        if n < 0:
            raise DualGravityError("rice unary length must be non-negative")
        while n > 0:
            room = 8 - self._filled
            take = n if n < room else room
            self._acc |= ((1 << take) - 1) << self._filled
            self._filled += take
            n -= take
            if self._filled == 8:
                self._buf.append(self._acc)
                self._acc = 0
                self._filled = 0

    def write_lsbs(self, value: int, bits: int) -> None:
        v = int(value)
        for i in range(int(bits)):
            self.write_bit((v >> i) & 1)

    def tobytes(self) -> bytes:
        if self._filled:
            self._buf.append(self._acc)
            self._acc = 0
            self._filled = 0
        return bytes(self._buf)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self._data = np.frombuffer(payload, dtype=np.uint8)
        self._byte = 0
        self._bit = 0

    def read_bit(self) -> int:
        if self._byte >= self._data.size:
            raise DualGravityError("rice stream overran its payload")
        bit = int((int(self._data[self._byte]) >> self._bit) & 1)
        self._bit += 1
        if self._bit == 8:
            self._byte += 1
            self._bit = 0
        return bit

    def read_lsbs(self, bits: int) -> int:
        value = 0
        for i in range(int(bits)):
            value |= self.read_bit() << i
        return value

    def read_rice(self, k: int) -> int:
        q = 0
        while self.read_bit() == 1:
            q += 1
        remainder = self.read_lsbs(k) if k else 0
        return (q << int(k)) | remainder


def _best_rice_k(values: np.ndarray) -> int:
    vals = np.ascontiguousarray(values, dtype=np.uint64).reshape(-1)
    if vals.size == 0:
        return 0
    best_k = 0
    best_bits = 1 << 62
    n = int(vals.size)
    for k in range(0, 16):
        q = vals >> np.uint64(k)
        bits = int(q.sum()) + n * (1 + k)
        if bits < best_bits:
            best_k = k
            best_bits = bits
    return best_k


def _pack_rice(values: np.ndarray, k: int) -> bytes:
    vals = np.ascontiguousarray(values, dtype=np.uint64).reshape(-1)
    if vals.size == 0:
        return b""
    if k < 0 or k > 63:
        raise DualGravityError("rice k out of range")
    writer = _BitWriter()
    k_int = int(k)
    mask = (1 << k_int) - 1 if k_int else 0
    for raw in vals.tolist():
        value = int(raw)
        writer.write_ones(value >> k_int)
        writer.write_bit(0)
        if k_int:
            writer.write_lsbs(value & mask, k_int)
    return writer.tobytes()


def _unpack_rice(payload: bytes, count: int, k: int) -> np.ndarray:
    if count < 0:
        raise DualGravityError("rice count must be non-negative")
    if count == 0:
        return np.zeros(0, dtype=np.uint32)
    reader = _BitReader(payload)
    out = np.empty(count, dtype=np.uint32)
    k_int = int(k)
    for i in range(count):
        out[i] = reader.read_rice(k_int)
    return out


def _local_index_bits(group_size: int) -> int:
    if group_size <= 1:
        return 1
    return int(math.ceil(math.log2(group_size)))


def _quantize_residual_values(
    values: np.ndarray,
    *,
    value_bits: int,
    value_scale: str,
) -> tuple[np.ndarray, bytes, bytes, dict[str, Any]]:
    vals = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if value_bits == 16:
        if value_scale != "fp16":
            raise DualGravityError("16-bit residual values require value_scale='fp16'")
        stored = vals.astype("<f2")
        return stored.astype(np.float32), b"", stored.tobytes(), {
            "value_bits": 16,
            "value_scale": "fp16",
            "residual_scale_bytes": 0,
            "residual_bytes": int(stored.nbytes),
        }
    if value_bits < 1 or value_bits > 8:
        raise DualGravityError("compact residual value_bits must be 1..8 or 16")
    if value_scale not in ("absmax", "mean_abs", "rms"):
        raise DualGravityError("compact residual value_scale is not supported")

    abs_vals = np.abs(vals)
    if value_scale == "rms":
        stat = float(np.sqrt(np.mean(np.square(vals)))) if vals.size else 0.0
    elif value_scale == "mean_abs":
        stat = float(np.mean(abs_vals)) if vals.size else 0.0
    else:
        stat = float(np.max(abs_vals)) if vals.size else 0.0

    if value_bits == 1:
        # Selected outliers are never encoded as zero; one stored scale + a sign.
        scale = stat
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        scale16 = np.asarray([scale], dtype="<f2")
        scale_f = float(scale16[0])
        signs = (vals >= 0.0).astype(np.uint8)
        decoded = np.where(signs.astype(bool), scale_f, -scale_f).astype(np.float32)
        packed = _pack_unsigned(signs, 1)
        return decoded, scale16.tobytes(), packed, {
            "value_bits": 1,
            "value_scale": value_scale,
            "residual_scale_bytes": int(scale16.nbytes),
            "residual_bytes": len(packed),
            "codebook": "sign_times_stored_scale",
        }

    bound = (1 << (value_bits - 1)) - 1
    scale = stat / max(bound, 1)
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale16 = np.asarray([scale], dtype="<f2")
    scale_f = float(scale16[0])
    codes = np.rint(vals / scale_f).clip(-bound, bound).astype(np.int16)
    unsigned = (codes + bound).astype(np.uint8)
    packed = _pack_unsigned(unsigned, value_bits)
    decoded = (codes.astype(np.float32) * np.float32(scale_f))
    return decoded, scale16.tobytes(), packed, {
        "value_bits": int(value_bits),
        "value_scale": value_scale,
        "residual_scale_bytes": int(scale16.nbytes),
        "residual_bytes": len(packed),
        "codebook": "signed_uniform_stored_scale",
        "bound": int(bound),
    }


def _dequantize_residual_values(
    scale_bytes: bytes,
    value_bytes: bytes,
    *,
    count: int,
    value_bits: int,
    value_scale: str,
) -> np.ndarray:
    if value_bits == 16:
        if len(value_bytes) != count * np.dtype("<f2").itemsize:
            raise DualGravityError("fp16 residual byte ledger is invalid")
        return np.frombuffer(value_bytes, dtype="<f2", count=count).astype(np.float32)
    if value_bits == 1:
        if len(scale_bytes) != 2:
            raise DualGravityError("1-bit residual is missing its stored scale")
        scale = float(np.frombuffer(scale_bytes, dtype="<f2", count=1)[0])
        signs = _unpack_unsigned(value_bytes, count, 1)
        return np.where(signs.astype(bool), scale, -scale).astype(np.float32)
    bound = (1 << (value_bits - 1)) - 1
    if len(scale_bytes) != 2:
        raise DualGravityError("quantized residual is missing its stored scale")
    scale = float(np.frombuffer(scale_bytes, dtype="<f2", count=1)[0])
    unsigned = _unpack_unsigned(value_bytes, count, value_bits)
    codes = unsigned.astype(np.int16) - np.int16(bound)
    return codes.astype(np.float32) * np.float32(scale)


def _pack_group_local(indices: np.ndarray, *, elements: int, group_size: int) -> tuple[bytes, dict[str, Any]]:
    groups = math.ceil(elements / group_size)
    group_ids = (indices.astype(np.uint32) // np.uint32(group_size)).astype(np.int64)
    counts = np.bincount(group_ids, minlength=groups).astype(np.uint16)
    if counts.size != groups:
        raise DualGravityError("group-local count geometry is invalid")
    max_count = int(counts.max()) if counts.size else 0
    if max_count > group_size:
        raise DualGravityError("group-local count exceeds group size")
    count_bits = 4 if max_count <= 15 else 8
    if max_count > 255:
        raise DualGravityError("group-local count exceeds 8-bit packing")
    local_bits = _local_index_bits(group_size)
    locals_u8 = (indices.astype(np.uint32) % np.uint32(group_size)).astype(np.uint8)
    count_bytes = _pack_unsigned(counts.astype(np.uint8), count_bits)
    local_bytes = _pack_unsigned(locals_u8, local_bits)
    blob = count_bytes + local_bytes
    return blob, {
        "index_mode": "group_local",
        "count_bits": count_bits,
        "local_index_bits": local_bits,
        "count_bytes": len(count_bytes),
        "local_index_bytes": len(local_bytes),
        "index_bytes": len(blob),
        "occupied_groups": int((counts > 0).sum()),
        "max_group_count": max_count,
    }


def _unpack_group_local(
    blob: bytes,
    *,
    elements: int,
    group_size: int,
    count: int,
    count_bits: int,
    local_index_bits: int,
    count_bytes: int,
    local_index_bytes: int,
) -> np.ndarray:
    groups = math.ceil(elements / group_size)
    if count_bytes != _packed_byte_count(count=groups, bits=count_bits):
        raise DualGravityError("group-local count ledger is invalid")
    if local_index_bytes != _packed_byte_count(count=count, bits=local_index_bits):
        raise DualGravityError("group-local index ledger is invalid")
    if len(blob) != count_bytes + local_index_bytes:
        raise DualGravityError("group-local index bytes do not match the ledger")
    counts = _unpack_unsigned(blob[:count_bytes], groups, count_bits).astype(np.int64)
    if int(counts.sum()) != count:
        raise DualGravityError("group-local counts do not sum to the outlier count")
    locals_u = _unpack_unsigned(blob[count_bytes:], count, local_index_bits).astype(np.uint32)
    starts = np.repeat(np.arange(groups, dtype=np.uint32) * np.uint32(group_size), counts)
    indices = starts + locals_u
    if indices.size and int(indices[-1]) >= elements:
        raise DualGravityError("group-local index exceeds element count")
    return indices.astype("<u4", copy=False)


def _pack_bitmap(indices: np.ndarray, *, elements: int, group_size: int) -> tuple[bytes, dict[str, Any]]:
    groups = math.ceil(elements / group_size)
    padded = groups * group_size
    membership = np.zeros(padded, dtype=np.uint8)
    membership[indices.astype(np.int64)] = 1
    grouped = membership.reshape(groups, group_size)
    occupied = grouped.any(axis=1)
    occ_bytes = np.packbits(occupied.astype(np.uint8), bitorder="little").tobytes()
    mask_bytes = np.packbits(grouped[occupied].reshape(-1), bitorder="little").tobytes()
    blob = occ_bytes + mask_bytes
    return blob, {
        "index_mode": "bitmap",
        "occupancy_bytes": len(occ_bytes),
        "mask_bytes": len(mask_bytes),
        "index_bytes": len(blob),
        "occupied_groups": int(occupied.sum()),
    }


def _unpack_bitmap(
    blob: bytes,
    *,
    elements: int,
    group_size: int,
    count: int,
    occupancy_bytes: int,
    mask_bytes: int,
) -> np.ndarray:
    groups = math.ceil(elements / group_size)
    expected_occ = int(math.ceil(groups / 8))
    if occupancy_bytes != expected_occ or len(blob) != occupancy_bytes + mask_bytes:
        raise DualGravityError("bitmap residual index ledger is invalid")
    occupied = np.unpackbits(np.frombuffer(blob[:occupancy_bytes], dtype=np.uint8), bitorder="little")[:groups].astype(bool)
    n_occ = int(occupied.sum())
    expected_mask = int(math.ceil(n_occ * group_size / 8))
    if mask_bytes != expected_mask:
        raise DualGravityError("bitmap residual mask ledger is invalid")
    masks = np.unpackbits(np.frombuffer(blob[occupancy_bytes:], dtype=np.uint8), bitorder="little")[: n_occ * group_size]
    masks = masks.reshape(n_occ, group_size).astype(bool)
    group_ids = np.nonzero(occupied)[0].astype(np.int64)
    local = [np.nonzero(row)[0].astype(np.int64) for row in masks]
    if local:
        indices = np.concatenate([g * group_size + loc for g, loc in zip(group_ids, local)])
    else:
        indices = np.zeros(0, dtype=np.int64)
    indices = indices[indices < elements]
    if indices.size != count:
        raise DualGravityError("bitmap residual decoded a different outlier count")
    return indices.astype("<u4")


def _pack_rice_indices(indices: np.ndarray) -> tuple[bytes, dict[str, Any]]:
    idx = np.ascontiguousarray(indices, dtype="<u4").reshape(-1)
    if idx.size == 0:
        raise DualGravityError("rice residual requires at least one outlier")
    first = np.asarray([int(idx[0])], dtype="<u4")
    if idx.size == 1:
        blob = first.tobytes()
        return blob, {
            "index_mode": "rice",
            "rice_k": 0,
            "rice_bytes": 0,
            "first_index_bytes": 4,
            "index_bytes": len(blob),
        }
    diffs = np.diff(idx.astype(np.int64))
    if np.any(diffs <= 0):
        raise DualGravityError("rice residual requires strictly increasing indices")
    rice_k = _best_rice_k(diffs)
    rice_bytes = _pack_rice(diffs.astype(np.uint64), rice_k)
    blob = first.tobytes() + rice_bytes
    return blob, {
        "index_mode": "rice",
        "rice_k": int(rice_k),
        "rice_bytes": len(rice_bytes),
        "first_index_bytes": 4,
        "index_bytes": len(blob),
    }


def _unpack_rice_indices(blob: bytes, *, count: int, rice_k: int, rice_bytes: int, first_index_bytes: int) -> np.ndarray:
    if first_index_bytes != 4 or len(blob) != first_index_bytes + rice_bytes:
        raise DualGravityError("rice residual index ledger is invalid")
    first = int(np.frombuffer(blob[:4], dtype="<u4", count=1)[0])
    if count == 1:
        if rice_bytes != 0:
            raise DualGravityError("single-outlier rice stream must be empty")
        return np.asarray([first], dtype="<u4")
    diffs = _unpack_rice(blob[4:], count - 1, rice_k)
    indices = np.empty(count, dtype=np.uint64)
    indices[0] = first
    indices[1:] = diffs
    np.cumsum(indices, out=indices)
    return indices.astype("<u4")


def _rebuild_binary(scales: np.ndarray, sign_bytes: bytes, *, elements: int, group_size: int) -> np.ndarray:
    groups = int(scales.size)
    padded = groups * group_size
    bits = np.unpackbits(np.frombuffer(sign_bytes, dtype=np.uint8), bitorder="little")[:padded]
    signs = np.where(bits.reshape(groups, group_size) > 0, 1.0, -1.0).astype(np.float32)
    rebuilt = (signs * scales.astype(np.float32)[:, None]).reshape(-1)[:elements]
    return rebuilt


def encode_residual_compact(
    values: np.ndarray,
    *,
    outlier_ratio: float,
    group_size: int = GROUP_BINARY,
    index_mode: str = "rice",
    value_bits: int = 16,
    value_scale: str | None = None,
) -> CodecResult:
    """Binary base + sparse residual, packed with a compact index/value codec."""

    if index_mode not in INDEX_MODES:
        raise DualGravityError(f"unknown residual index_mode {index_mode!r}")
    if group_size <= 0:
        raise DualGravityError("residual group size must be positive")
    scale_name = default_value_scale(value_bits) if value_scale is None else value_scale
    if scale_name not in VALUE_SCALES:
        raise DualGravityError(f"unknown residual value_scale {scale_name!r}")

    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    _, scales, signs, base = _binary_parts(values, group_size=group_size)
    reconstructed = np.ascontiguousarray(base, dtype=np.float32).reshape(-1)
    residual = flat - reconstructed
    indices, count = select_outlier_indices(residual, outlier_ratio)
    decoded_values, scale_blob, value_blob, value_meta = _quantize_residual_values(
        residual[indices.astype(np.int64)],
        value_bits=value_bits,
        value_scale=scale_name,
    )

    if index_mode == "rice":
        index_blob, index_meta = _pack_rice_indices(indices)
    elif index_mode == "group_local":
        index_blob, index_meta = _pack_group_local(indices, elements=int(flat.size), group_size=group_size)
    else:
        index_blob, index_meta = _pack_bitmap(indices, elements=int(flat.size), group_size=group_size)

    header: dict[str, Any] = {
        "schema": SCHEMA_RESIDUAL_COMPACT,
        "representation": "binary_sign_scale_plus_compact_sparse_residual",
        "shape": [int(item) for item in values.shape],
        "elements": int(flat.size),
        "group_size": int(group_size),
        "groups": int(scales.size),
        "outlier_ratio_requested": float(outlier_ratio),
        "outlier_count": int(count),
        "scale_dtype": "float16",
        "scale_bytes": int(scales.nbytes),
        "sign_bytes": len(signs),
        "index_bytes": int(index_meta["index_bytes"]),
        "residual_scale_bytes": int(value_meta["residual_scale_bytes"]),
        "residual_bytes": int(value_meta["residual_bytes"]),
        "value_bits": int(value_bits),
        "value_scale": scale_name,
        "index_mode": index_mode,
        "selection": "global_top_k_abs_residual",
    }
    header.update(index_meta)
    header.update({k: v for k, v in value_meta.items() if k not in header})
    body = scales.tobytes() + signs + index_blob + scale_blob + value_blob
    payload = _container(MAGIC_RESIDUAL_COMPACT, header, body)
    return CodecResult(
        payload=payload,
        reconstruction=_decode_residual_compact(payload),
        metadata=header,
    )


def _decode_residual_compact(payload: bytes) -> np.ndarray:
    header, body = _parse_container(payload, expected_magic=MAGIC_RESIDUAL_COMPACT)
    try:
        shape = tuple(int(item) for item in header["shape"])
        elements = int(header["elements"])
        group_size = int(header["group_size"])
        groups = int(header["groups"])
        count = int(header["outlier_count"])
        scale_bytes = int(header["scale_bytes"])
        sign_bytes = int(header["sign_bytes"])
        index_bytes = int(header["index_bytes"])
        residual_scale_bytes = int(header["residual_scale_bytes"])
        residual_bytes = int(header["residual_bytes"])
        index_mode = str(header["index_mode"])
        value_bits = int(header["value_bits"])
        value_scale = str(header["value_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DualGravityError("compact residual header lacks required geometry") from exc
    if not shape or any(item <= 0 for item in shape) or math.prod(shape) != elements:
        raise DualGravityError("compact residual shape does not match element count")
    if group_size <= 0 or groups != math.ceil(elements / group_size):
        raise DualGravityError("compact residual group geometry is invalid")
    if count <= 0 or count > elements:
        raise DualGravityError("compact residual outlier count is invalid")
    expected_body = scale_bytes + sign_bytes + index_bytes + residual_scale_bytes + residual_bytes
    if len(body) != expected_body:
        raise DualGravityError("compact residual physical body bytes do not match its ledger")
    if scale_bytes != groups * np.dtype("<f2").itemsize:
        raise DualGravityError("compact residual binary scale ledger is invalid")

    cursor = 0
    scales = np.frombuffer(body[cursor : cursor + scale_bytes], dtype="<f2", count=groups)
    cursor += scale_bytes
    sign_blob = body[cursor : cursor + sign_bytes]
    cursor += sign_bytes
    index_blob = body[cursor : cursor + index_bytes]
    cursor += index_bytes
    scale_blob = body[cursor : cursor + residual_scale_bytes]
    cursor += residual_scale_bytes
    value_blob = body[cursor : cursor + residual_bytes]

    if index_mode == "rice":
        indices = _unpack_rice_indices(
            index_blob,
            count=count,
            rice_k=int(header.get("rice_k", 0)),
            rice_bytes=int(header.get("rice_bytes", 0)),
            first_index_bytes=int(header.get("first_index_bytes", 4)),
        )
    elif index_mode == "group_local":
        indices = _unpack_group_local(
            index_blob,
            elements=elements,
            group_size=group_size,
            count=count,
            count_bits=int(header["count_bits"]),
            local_index_bits=int(header["local_index_bits"]),
            count_bytes=int(header["count_bytes"]),
            local_index_bytes=int(header["local_index_bytes"]),
        )
    elif index_mode == "bitmap":
        indices = _unpack_bitmap(
            index_blob,
            elements=elements,
            group_size=group_size,
            count=count,
            occupancy_bytes=int(header["occupancy_bytes"]),
            mask_bytes=int(header["mask_bytes"]),
        )
    else:
        raise DualGravityError(f"unknown residual index_mode {index_mode!r}")

    if indices.size != count:
        raise DualGravityError("compact residual decoded a different outlier count")
    if indices.size and (int(indices[0]) < 0 or int(indices[-1]) >= elements):
        raise DualGravityError("compact residual index is out of range")

    decoded_values = _dequantize_residual_values(
        scale_blob,
        value_blob,
        count=count,
        value_bits=value_bits,
        value_scale=value_scale,
    )
    rebuilt = _rebuild_binary(scales, sign_blob, elements=elements, group_size=group_size)
    rebuilt[indices.astype(np.int64)] += decoded_values
    return np.ascontiguousarray(rebuilt.reshape(shape), dtype=np.float32)


def decode_residual_compact(payload: bytes) -> np.ndarray:
    return _decode_residual_compact(payload)

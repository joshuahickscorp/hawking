#!/usr/bin/env python3.12
"""Bounded native DeepSeek-V4-Flash FP4/FP8 decoding primitives.

This is intentionally a *pure byte-to-array* module.  It does not open model
files, download shards, retain source bodies, pack an artifact, or run a model.
The caller supplies a safetensors descriptor and exactly the requested body
bytes (normally an in-memory Xet range chunk).  Every layout assumption is
checked before a byte is decoded.

Authority boundary
------------------
The layout below is bound to the official, pinned ``inference/convert.py`` and
``inference/kernel.py`` for DeepSeek-V4-Flash:

* E2M1FN FP4 values are packed low-nibble then high-nibble along the last
  (input/K) dimension, two logical values per source byte.
* Routed-expert FP4 scales are E8M0FNU, one per 32 logical K values on each
  output row.
* E4M3FN FP8 weights use E8M0FNU scales over 128-by-128 output/K blocks.

Those source-code facts make this a *codec mechanics* implementation.  A call
to this module is not a source-authority proof: callers must bind fetched range
bytes to an independently verified immutable source receipt before reporting a
``source_exact`` result.  The functions fail closed rather than silently
guessing a different FP4, FP8, scale, or block layout.
"""
from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np


OFFICIAL_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash"
OFFICIAL_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
# SHA-256 values of the bounded, pinned official implementation files inspected
# when this contract was written.  They are anchors for review, not a substitute
# for a source-authority receipt.
OFFICIAL_CONVERT_SHA256 = "912acfc20bdd9ae4dbd5bde9dc7c8e61f6d27b6826d3ac2d052b2534c0881454"
OFFICIAL_KERNEL_SHA256 = "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2"

MAX_HEADER_BYTES = 8 * 1024**2
FP4_LOGICAL_BLOCK = 32
FP8_BLOCK_ROWS = 128
FP8_BLOCK_COLS = 128

# Exact literal in the official pinned inference/convert.py.  Keeping the
# signed-zero entries as zero matches its table lookup semantics.
FP4_E2M1FN_TABLE = np.asarray(
    (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ),
    dtype=np.float32,
)


class DeepSeekV4NativeCodecError(ValueError):
    """A supplied descriptor or bounded byte window cannot be decoded safely."""


def sha256_hex(raw: bytes) -> str:
    """Return a byte-exact digest without retaining or interpreting ``raw``."""
    if not isinstance(raw, bytes):
        raise DeepSeekV4NativeCodecError("raw input must be immutable bytes")
    return hashlib.sha256(raw).hexdigest()


def parse_header_only(capture: bytes) -> dict[str, dict[str, Any]]:
    """Parse an exact safetensors header capture and reject any tensor body.

    A header-only range is ``<u64 header length><JSON header>``.  Accepting
    trailing bytes here would make the control-plane helper accidentally become
    a body-retention path, so it is intentionally refused.
    """
    if not isinstance(capture, bytes):
        raise DeepSeekV4NativeCodecError("safetensors header capture must be bytes")
    if len(capture) < 8:
        raise DeepSeekV4NativeCodecError("safetensors header capture is truncated before length")
    header_length = struct.unpack_from("<Q", capture)[0]
    if header_length <= 0 or header_length > MAX_HEADER_BYTES:
        raise DeepSeekV4NativeCodecError("safetensors header length is outside bounded limit")
    expected = 8 + header_length
    if len(capture) != expected:
        raise DeepSeekV4NativeCodecError(
            "header-only capture must contain exactly its prefix and JSON header; "
            f"expected {expected} bytes, received {len(capture)}"
        )
    try:
        value = json.loads(capture[8:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeepSeekV4NativeCodecError("safetensors header JSON is invalid") from exc
    if not isinstance(value, dict):
        raise DeepSeekV4NativeCodecError("safetensors header root must be an object")
    return value


def descriptor_from_header(header: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return a canonical named descriptor suitable for a bounded decoder call."""
    if not isinstance(header, Mapping):
        raise DeepSeekV4NativeCodecError("safetensors header must be a mapping")
    if not isinstance(name, str) or not name:
        raise DeepSeekV4NativeCodecError("tensor name must be a non-empty string")
    value = header.get(name)
    if not isinstance(value, Mapping):
        raise DeepSeekV4NativeCodecError(f"header does not contain tensor {name!r}")
    descriptor = dict(value)
    descriptor["name"] = name
    _normalise_descriptor(descriptor)
    return descriptor


def expected_source_range(
    descriptor: Mapping[str, Any],
    *,
    header_bytes: int,
    row_start: int = 0,
    row_count: int | None = None,
) -> dict[str, int | str]:
    """Return the exact file interval for contiguous complete rows.

    This helper only plans a range.  It makes no network request and is useful
    for a transport that will feed the returned bytes directly to one of the
    decoder functions below.
    """
    canonical = _normalise_descriptor(descriptor)
    if header_bytes < 8:
        raise DeepSeekV4NativeCodecError("header_bytes must include safetensors u64 prefix")
    rows, columns = canonical["shape"]
    start, count = _normalise_row_window(rows, row_start, row_count)
    row_bytes = columns * _element_bytes(canonical["dtype"])
    data_start, data_stop = canonical["data_offsets"]
    file_start = header_bytes + data_start + start * row_bytes
    file_stop = file_start + count * row_bytes
    if file_stop > header_bytes + data_stop:
        raise DeepSeekV4NativeCodecError("requested rows would escape descriptor data extent")
    return {
        "tensor": canonical["name"],
        "file_start": file_start,
        "file_stop": file_stop,
        "byte_count": file_stop - file_start,
        "row_start": start,
        "row_count": count,
    }


def decode_e8m0fnu(raw: bytes | np.ndarray) -> np.ndarray:
    """Decode finite unsigned E8M0 scales exactly to ``float32``.

    The format is an exponent-only power of two.  For byte ``b`` in ``0..254``
    the exact value is ``2 ** (b - 127)``.  ``0xff`` is the sole NaN encoding
    of ``float8_e8m0fnu`` and is rejected: a scale tensor containing it cannot
    safely participate in a model forward.
    """
    values = _as_uint8(raw, "E8M0 scale bytes")
    if np.any(values == 0xFF):
        raise DeepSeekV4NativeCodecError("E8M0FNU scale contains the 0xff NaN encoding")
    exponent = values.astype(np.int16) - 127
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        decoded = np.ldexp(np.ones(values.shape, dtype=np.float32), exponent)
    if not np.isfinite(decoded).all() or np.any(decoded <= 0.0):
        raise DeepSeekV4NativeCodecError("E8M0FNU scale decode was not finite and positive")
    return decoded.astype(np.float32, copy=False)


def decode_e4m3fn(raw: bytes | np.ndarray) -> np.ndarray:
    """Decode finite E4M3FN bytes exactly to ``float32``.

    This is PyTorch's ``float8_e4m3fn`` layout used by the pinned source:
    exponent bias 7, subnormals at exponent zero, finite exponent-15 values
    through ``0x7e``/``0xfe`` (magnitude 448), and NaNs only at ``0x7f`` and
    ``0xff``.  NaNs are rejected rather than propagated into a Condense input.
    """
    bits = _as_uint8(raw, "E4M3FN weight bytes")
    exponent = ((bits >> 3) & 0x0F).astype(np.int16)
    mantissa = (bits & 0x07).astype(np.float32)
    nan_mask = (exponent == 0x0F) & (mantissa == 7.0)
    if np.any(nan_mask):
        raise DeepSeekV4NativeCodecError("E4M3FN weight contains a NaN encoding (0x7f or 0xff)")
    sign = np.where((bits & 0x80) != 0, np.float32(-1.0), np.float32(1.0))
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        normal = np.ldexp(np.float32(1.0) + mantissa / np.float32(8.0), exponent - 7)
    subnormal = mantissa * np.float32(2.0**-9)
    decoded = sign * np.where(exponent == 0, subnormal, normal)
    if not np.isfinite(decoded).all():
        raise DeepSeekV4NativeCodecError("E4M3FN weight decode was not finite")
    return decoded.astype(np.float32, copy=False)


def decode_fp4_e2m1fn_x2_rows(
    weight_bytes: bytes,
    weight_descriptor: Mapping[str, Any],
    scale_bytes: bytes,
    scale_descriptor: Mapping[str, Any],
    *,
    row_start: int = 0,
    row_count: int | None = None,
) -> np.ndarray:
    """Decode complete routed-expert rows from packed E2M1FN FP4 source bytes.

    ``weight_descriptor`` must be the original safetensors ``I8 [out, K/2]``
    descriptor.  ``weight_bytes`` must contain exactly the requested complete
    rows, not an arbitrary column segment.  Its matching scale descriptor is
    required to be ``F8_E8M0 [out, K/32]`` and ``scale_bytes`` must carry the
    matching scale rows.  The output is ``float32 [rows, K]``.
    """
    weight = _normalise_descriptor(weight_descriptor)
    scale = _normalise_descriptor(scale_descriptor)
    if weight["dtype"] != "I8":
        raise DeepSeekV4NativeCodecError(
            f"FP4 E2M1FN payload must be safetensors I8, got {weight['dtype']!r}"
        )
    if scale["dtype"] != "F8_E8M0":
        raise DeepSeekV4NativeCodecError(
            f"FP4 E2M1FN scale must be F8_E8M0, got {scale['dtype']!r}"
        )
    out_dim, packed_k = weight["shape"]
    logical_k = packed_k * 2
    if logical_k % FP4_LOGICAL_BLOCK:
        raise DeepSeekV4NativeCodecError("FP4 logical K must be divisible by 32")
    if scale["shape"] != (out_dim, logical_k // FP4_LOGICAL_BLOCK):
        raise DeepSeekV4NativeCodecError(
            "FP4 scale shape must be exactly [out, logical_K / 32]"
        )
    start, count = _normalise_row_window(out_dim, row_start, row_count)
    _require_exact_bytes(weight_bytes, count * packed_k, "FP4 packed weight row window")
    _require_exact_bytes(scale_bytes, count * (logical_k // FP4_LOGICAL_BLOCK), "FP4 scale row window")

    packed = np.frombuffer(weight_bytes, dtype=np.uint8).reshape(count, packed_k)
    nibbles = np.empty((count, logical_k), dtype=np.uint8)
    nibbles[:, 0::2] = packed & 0x0F
    nibbles[:, 1::2] = (packed >> 4) & 0x0F
    unit_values = FP4_E2M1FN_TABLE[nibbles]
    row_scales = decode_e8m0fnu(
        np.frombuffer(scale_bytes, dtype=np.uint8).reshape(count, logical_k // FP4_LOGICAL_BLOCK)
    )
    output = unit_values * np.repeat(row_scales, FP4_LOGICAL_BLOCK, axis=1)
    if not np.isfinite(output).all():
        raise DeepSeekV4NativeCodecError("FP4 E2M1FN dequantization was not finite")
    # ``start`` is deliberately validated even though the byte slice has already
    # been isolated by its caller; it prevents an out-of-range source claim.
    del start
    return output.astype(np.float32, copy=False)


def decode_fp8_e4m3fn_rows(
    weight_bytes: bytes,
    weight_descriptor: Mapping[str, Any],
    scale_bytes: bytes,
    scale_descriptor: Mapping[str, Any],
    *,
    row_start: int = 0,
    row_count: int | None = None,
    scale_block_row_start: int | None = None,
) -> np.ndarray:
    """Decode complete rows of a 128-by-128 E4M3FN/E8M0 tensor.

    A bounded caller may fetch one or more source weight rows but must also
    provide every intersecting scale-block row.  ``scale_block_row_start`` is
    explicit so that an accidental scale slice from the wrong 128-row group is
    rejected before arithmetic begins.  The output is ``float32 [rows, K]``.
    """
    weight = _normalise_descriptor(weight_descriptor)
    scale = _normalise_descriptor(scale_descriptor)
    if weight["dtype"] != "F8_E4M3":
        raise DeepSeekV4NativeCodecError(
            f"FP8 E4M3FN payload must be safetensors F8_E4M3, got {weight['dtype']!r}"
        )
    if scale["dtype"] != "F8_E8M0":
        raise DeepSeekV4NativeCodecError(
            f"FP8 E4M3FN scale must be F8_E8M0, got {scale['dtype']!r}"
        )
    out_dim, logical_k = weight["shape"]
    if out_dim % FP8_BLOCK_ROWS or logical_k % FP8_BLOCK_COLS:
        raise DeepSeekV4NativeCodecError("FP8 E4M3FN dimensions must be divisible by 128")
    expected_scale_shape = (out_dim // FP8_BLOCK_ROWS, logical_k // FP8_BLOCK_COLS)
    if scale["shape"] != expected_scale_shape:
        raise DeepSeekV4NativeCodecError(
            "FP8 scale shape must be exactly [out / 128, logical_K / 128]"
        )
    start, count = _normalise_row_window(out_dim, row_start, row_count)
    _require_exact_bytes(weight_bytes, count * logical_k, "FP8 E4M3FN weight row window")

    first_scale_row = start // FP8_BLOCK_ROWS
    final_scale_row = (start + count - 1) // FP8_BLOCK_ROWS
    required_scale_rows = final_scale_row - first_scale_row + 1
    if scale_block_row_start is None:
        # A full tensor call can safely infer its first scale row; a partial
        # range must name it to avoid accepting a plausible-but-wrong slice.
        if start != 0 or count != out_dim:
            raise DeepSeekV4NativeCodecError(
                "partial FP8 window requires explicit scale_block_row_start"
            )
        scale_block_row_start = 0
    if isinstance(scale_block_row_start, bool) or not isinstance(scale_block_row_start, int):
        raise DeepSeekV4NativeCodecError("scale_block_row_start must be an integer")
    if scale_block_row_start != first_scale_row:
        raise DeepSeekV4NativeCodecError(
            "FP8 scale_block_row_start does not cover the requested weight rows"
        )
    scale_cols = expected_scale_shape[1]
    _require_exact_bytes(scale_bytes, required_scale_rows * scale_cols, "FP8 E8M0 scale block-row window")

    unit_values = decode_e4m3fn(
        np.frombuffer(weight_bytes, dtype=np.uint8).reshape(count, logical_k)
    )
    block_scales = decode_e8m0fnu(
        np.frombuffer(scale_bytes, dtype=np.uint8).reshape(required_scale_rows, scale_cols)
    )
    source_rows = np.arange(start, start + count, dtype=np.int64)
    scale_row_index = source_rows // FP8_BLOCK_ROWS - first_scale_row
    per_row_scales = block_scales[scale_row_index]
    output = unit_values * np.repeat(per_row_scales, FP8_BLOCK_COLS, axis=1)
    if not np.isfinite(output).all():
        raise DeepSeekV4NativeCodecError("FP8 E4M3FN dequantization was not finite")
    return output.astype(np.float32, copy=False)


def bounded_fixture_status(
    *,
    repository: str,
    revision: str,
    header_capture_sha256: str,
    source_authority_status: str | None,
) -> dict[str, Any]:
    """Express the codec evidence boundary without inventing source authority.

    This module intentionally cannot report ``source_exact``.  A string that
    merely says an external authority passed is not evidence, and accepting it
    here would let a caller mint a false provenance upgrade.  A higher-level
    admission system must verify and join its own source receipt with this
    mechanics result before it publishes any source-exact statement.
    """
    if repository != OFFICIAL_REPOSITORY or revision != OFFICIAL_REVISION:
        raise DeepSeekV4NativeCodecError("fixture source is not the pinned DeepSeek-V4-Flash source")
    if not _is_sha256(header_capture_sha256):
        raise DeepSeekV4NativeCodecError("header_capture_sha256 must be a SHA-256 hex digest")
    return {
        "schema": "hawking.gravity.deepseek_v4.native_codec_fixture.v1",
        "repository": repository,
        "revision": revision,
        "official_codec_anchors": {
            "convert_py_sha256": OFFICIAL_CONVERT_SHA256,
            "kernel_py_sha256": OFFICIAL_KERNEL_SHA256,
        },
        "header_capture_sha256": header_capture_sha256.lower(),
        "codec_mechanics": "IMPLEMENTED_BOUNDED_BYTES_ONLY",
        "external_source_authority_claim": source_authority_status or "not_provided",
        "source_byte_fixture": "AWAITING_EXTERNAL_AUTHORITY_JOIN",
        "status": "NOT_SOURCE_EXACT",
        "not_evidence_of": [
            "complete_source_download",
            "condense_artifact",
            "cpu_model_forward",
            "metal_model_forward",
            "capability",
            "throughput",
        ],
    }


def _normalise_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DeepSeekV4NativeCodecError("tensor descriptor must be a mapping")
    name = value.get("name")
    dtype = value.get("dtype")
    shape = value.get("shape")
    offsets = value.get("data_offsets")
    if not isinstance(name, str) or not name:
        raise DeepSeekV4NativeCodecError("tensor descriptor requires a non-empty name")
    if dtype not in {"I8", "F8_E4M3", "F8_E8M0"}:
        raise DeepSeekV4NativeCodecError(f"unsupported native codec dtype {dtype!r}")
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise DeepSeekV4NativeCodecError("tensor descriptor shape must be a rank-2 sequence")
    dimensions: list[int] = []
    for index, dimension in enumerate(shape):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise DeepSeekV4NativeCodecError(f"tensor descriptor shape[{index}] must be positive integer")
        dimensions.append(dimension)
    if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
        raise DeepSeekV4NativeCodecError("tensor descriptor data_offsets must be [start, stop]")
    start, stop = offsets
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start
    ):
        raise DeepSeekV4NativeCodecError("tensor descriptor offsets must be increasing non-negative integers")
    expected_bytes = dimensions[0] * dimensions[1] * _element_bytes(dtype)
    if stop - start != expected_bytes:
        raise DeepSeekV4NativeCodecError(
            f"tensor descriptor byte extent {stop - start} does not equal its {dtype} shape extent {expected_bytes}"
        )
    return {
        "name": name,
        "dtype": dtype,
        "shape": tuple(dimensions),
        "data_offsets": (start, stop),
    }


def _element_bytes(dtype: str) -> int:
    if dtype in {"I8", "F8_E4M3", "F8_E8M0"}:
        return 1
    raise DeepSeekV4NativeCodecError(f"unsupported byte width for dtype {dtype!r}")


def _normalise_row_window(total_rows: int, row_start: int, row_count: int | None) -> tuple[int, int]:
    if isinstance(row_start, bool) or not isinstance(row_start, int) or row_start < 0:
        raise DeepSeekV4NativeCodecError("row_start must be a non-negative integer")
    if row_count is None:
        row_count = total_rows - row_start
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise DeepSeekV4NativeCodecError("row_count must be a positive integer")
    if row_start + row_count > total_rows:
        raise DeepSeekV4NativeCodecError("requested row window escapes tensor shape")
    return row_start, row_count


def _require_exact_bytes(raw: bytes, expected: int, label: str) -> None:
    if not isinstance(raw, bytes):
        raise DeepSeekV4NativeCodecError(f"{label} must be immutable bytes")
    if len(raw) != expected:
        raise DeepSeekV4NativeCodecError(
            f"{label} must contain exactly {expected} bytes, received {len(raw)}"
        )


def _as_uint8(raw: bytes | np.ndarray, label: str) -> np.ndarray:
    if isinstance(raw, bytes):
        return np.frombuffer(raw, dtype=np.uint8)
    if isinstance(raw, np.ndarray) and raw.dtype == np.uint8:
        return raw
    raise DeepSeekV4NativeCodecError(f"{label} must be bytes or a uint8 ndarray")


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True

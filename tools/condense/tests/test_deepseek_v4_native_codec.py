"""Focused bounded-byte contracts for the DeepSeek-V4 native codec fixture."""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from tools.condense import deepseek_v4_native_codec as codec


def _descriptor(name: str, dtype: str, shape: tuple[int, int], offset: int = 0) -> dict[str, object]:
    size = shape[0] * shape[1]
    return {
        "name": name,
        "dtype": dtype,
        "shape": list(shape),
        "data_offsets": [offset, offset + size],
    }


def _fp4_pair(rows: int = 1, packed_k: int = 16) -> tuple[dict[str, object], dict[str, object]]:
    logical_k = packed_k * 2
    weight = _descriptor("layers.4.ffn.experts.0.w1.weight", "I8", (rows, packed_k))
    scale = _descriptor("layers.4.ffn.experts.0.w1.scale", "F8_E8M0", (rows, logical_k // 32))
    return weight, scale


def _fp8_pair(rows: int = 128, logical_k: int = 128) -> tuple[dict[str, object], dict[str, object]]:
    weight = _descriptor("layers.4.attn.indexer.wq_b.weight", "F8_E4M3", (rows, logical_k))
    scale = _descriptor(
        "layers.4.attn.indexer.wq_b.scale", "F8_E8M0", (rows // 128, logical_k // 128)
    )
    return weight, scale


def test_fp4_low_then_high_nibble_and_e8m0_scale_are_exact() -> None:
    weight, scale = _fp4_pair()
    # Each byte gives two values: low nibble first, then high nibble.  Scale
    # 0x80 is 2 ** (128 - 127) = 2.0.
    packed = bytes([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2)
    decoded = codec.decode_fp4_e2m1fn_x2_rows(packed, weight, bytes([0x80]), scale)
    expected_unit = codec.FP4_E2M1FN_TABLE[
        np.asarray([n for byte in packed for n in (byte & 0x0F, byte >> 4)], dtype=np.uint8)
    ]
    assert decoded.shape == (1, 32)
    assert np.array_equal(decoded[0], expected_unit * np.float32(2.0))
    assert decoded[0, :4].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_fp4_requires_complete_matching_weight_and_scale_rows() -> None:
    weight, scale = _fp4_pair(rows=2)
    with pytest.raises(codec.DeepSeekV4NativeCodecError, match="exactly 16 bytes"):
        codec.decode_fp4_e2m1fn_x2_rows(b"\x00" * 15, weight, b"\x7f", scale, row_count=1)
    with pytest.raises(codec.DeepSeekV4NativeCodecError, match="exactly 1 bytes"):
        codec.decode_fp4_e2m1fn_x2_rows(b"\x00" * 16, weight, b"", scale, row_count=1)
    wrong_scale = _descriptor("layers.4.ffn.experts.0.w1.scale", "F8_E8M0", (2, 2))
    with pytest.raises(codec.DeepSeekV4NativeCodecError, match="scale shape"):
        codec.decode_fp4_e2m1fn_x2_rows(b"\x00" * 16, weight, b"\x7f", wrong_scale, row_count=1)


def test_e8m0fnu_all_finite_encodings_match_ml_dtypes_oracle() -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    raw = np.arange(255, dtype=np.uint8)
    expected = raw.view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
    actual = codec.decode_e8m0fnu(raw)
    assert np.array_equal(actual, expected)
    with pytest.raises(codec.DeepSeekV4NativeCodecError, match="0xff"):
        codec.decode_e8m0fnu(bytes([0xFF]))


def test_e4m3fn_all_finite_encodings_match_ml_dtypes_oracle() -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    raw = np.arange(256, dtype=np.uint8)
    finite = raw[(raw != 0x7F) & (raw != 0xFF)]
    expected = finite.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    actual = codec.decode_e4m3fn(finite)
    assert np.array_equal(actual, expected)
    for nan_code in (0x7F, 0xFF):
        with pytest.raises(codec.DeepSeekV4NativeCodecError, match="NaN"):
            codec.decode_e4m3fn(bytes([nan_code]))


def test_fp8_block_scale_maps_rows_and_columns_without_guessing() -> None:
    weight, scale = _fp8_pair()
    # 0x38 is exactly +1.0 E4M3FN.  0x80 E8M0FNU is exactly 2.0.
    decoded = codec.decode_fp8_e4m3fn_rows(b"\x38" * (128 * 128), weight, b"\x80", scale)
    assert decoded.shape == (128, 128)
    assert np.array_equal(decoded, np.full((128, 128), 2.0, dtype=np.float32))


def test_fp8_partial_rows_require_the_correct_explicit_scale_block_row() -> None:
    weight, scale = _fp8_pair(rows=256)
    # Rows 129 and 130 are in scale block row 1, not 0.
    source = b"\x38" * (2 * 128)
    decoded = codec.decode_fp8_e4m3fn_rows(
        source,
        weight,
        b"\x80",
        scale,
        row_start=129,
        row_count=2,
        scale_block_row_start=1,
    )
    assert np.array_equal(decoded, np.full((2, 128), 2.0, dtype=np.float32))
    with pytest.raises(codec.DeepSeekV4NativeCodecError, match="requires explicit"):
        codec.decode_fp8_e4m3fn_rows(
            source, weight, b"\x80", scale, row_start=129, row_count=2
        )
    with pytest.raises(codec.DeepSeekV4NativeCodecError, match="does not cover"):
        codec.decode_fp8_e4m3fn_rows(
            source,
            weight,
            b"\x80",
            scale,
            row_start=129,
            row_count=2,
            scale_block_row_start=0,
        )


def test_official_layer4_descriptor_shapes_plan_and_decode_one_bounded_row() -> None:
    """The real pinned header geometry must not silently regress to a generic layout."""
    header_bytes = 173_552
    fp4_weight = {
        "name": "layers.4.ffn.experts.0.w1.weight",
        "dtype": "I8",
        "shape": [2048, 2048],
        "data_offsets": [368_625_752, 372_820_056],
    }
    fp4_scale = {
        "name": "layers.4.ffn.experts.0.w1.scale",
        "dtype": "F8_E8M0",
        "shape": [2048, 128],
        "data_offsets": [26_788_440, 27_050_584],
    }
    fp8_weight = {
        "name": "layers.4.ffn.shared_experts.w1.weight",
        "dtype": "F8_E4M3",
        "shape": [2048, 4096],
        "data_offsets": [343_459_928, 351_848_536],
    }
    fp8_scale = {
        "name": "layers.4.ffn.shared_experts.w1.scale",
        "dtype": "F8_E8M0",
        "shape": [16, 32],
        "data_offsets": [228_115_032, 228_115_544],
    }
    assert codec.expected_source_range(fp4_weight, header_bytes=header_bytes, row_count=1) == {
        "tensor": fp4_weight["name"],
        "file_start": 368_799_304,
        "file_stop": 368_801_352,
        "byte_count": 2048,
        "row_start": 0,
        "row_count": 1,
    }
    assert codec.expected_source_range(fp8_scale, header_bytes=header_bytes, row_count=1) == {
        "tensor": fp8_scale["name"],
        "file_start": 228_288_584,
        "file_stop": 228_288_616,
        "byte_count": 32,
        "row_start": 0,
        "row_count": 1,
    }
    decoded_fp4 = codec.decode_fp4_e2m1fn_x2_rows(
        b"\x21" * 2048, fp4_weight, b"\x7f" * 128, fp4_scale, row_count=1
    )
    decoded_fp8 = codec.decode_fp8_e4m3fn_rows(
        b"\x38" * 4096,
        fp8_weight,
        b"\x7f" * 32,
        fp8_scale,
        row_count=1,
        scale_block_row_start=0,
    )
    assert decoded_fp4.shape == (1, 4096)
    assert decoded_fp8.shape == (1, 4096)
    assert np.array_equal(decoded_fp4[0, :4], np.asarray([0.5, 1.0, 0.5, 1.0], dtype=np.float32))
    assert np.array_equal(decoded_fp8, np.ones((1, 4096), dtype=np.float32))


def test_header_parser_refuses_body_and_binds_named_descriptor() -> None:
    entry = {
        "dtype": "I8",
        "shape": [1, 16],
        "data_offsets": [0, 16],
    }
    header = json.dumps({"layers.4.ffn.experts.0.w1.weight": entry}).encode("utf-8")
    capture = struct.pack("<Q", len(header)) + header
    parsed = codec.parse_header_only(capture)
    descriptor = codec.descriptor_from_header(parsed, "layers.4.ffn.experts.0.w1.weight")
    assert descriptor["name"] == "layers.4.ffn.experts.0.w1.weight"
    assert codec.expected_source_range(descriptor, header_bytes=len(capture), row_count=1) == {
        "tensor": "layers.4.ffn.experts.0.w1.weight",
        "file_start": len(capture),
        "file_stop": len(capture) + 16,
        "byte_count": 16,
        "row_start": 0,
        "row_count": 1,
    }
    with pytest.raises(codec.DeepSeekV4NativeCodecError, match="exactly"):
        codec.parse_header_only(capture + b"not-a-body")


def test_fixture_status_cannot_mint_source_exact_from_an_external_status_string() -> None:
    receipt = codec.bounded_fixture_status(
        repository=codec.OFFICIAL_REPOSITORY,
        revision=codec.OFFICIAL_REVISION,
        header_capture_sha256="a" * 64,
        source_authority_status=None,
    )
    assert receipt["status"] == "NOT_SOURCE_EXACT"
    assert receipt["source_byte_fixture"] == "AWAITING_EXTERNAL_AUTHORITY_JOIN"
    asserted = codec.bounded_fixture_status(
        repository=codec.OFFICIAL_REPOSITORY,
        revision=codec.OFFICIAL_REVISION,
        header_capture_sha256="b" * 64,
        source_authority_status="SOURCE_EXACT_VERIFIED",
    )
    assert asserted["status"] == "NOT_SOURCE_EXACT"
    assert asserted["source_byte_fixture"] == "AWAITING_EXTERNAL_AUTHORITY_JOIN"

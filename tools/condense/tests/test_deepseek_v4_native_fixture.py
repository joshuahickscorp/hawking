"""Contract tests for the fixed 6,304B Xet native codec fixture plan."""
from __future__ import annotations

import json
import struct

import pytest

from lab.operators import deepseek_v4_native_fixture as fixture


def _header() -> dict[str, object]:
    return {row["name"]: {key: value for key, value in row.items() if key != "name"}
            for row in fixture._EXPECTED_DESCRIPTORS.values()}


def test_real_layer4_descriptor_contract_builds_exact_bounded_ranges() -> None:
    descriptors = fixture._descriptors_from_header(_header())
    ranges = fixture.fixture_ranges(173_552, descriptors)
    assert sum(row["byte_count"] for row in ranges.values()) == 6_304
    assert ranges["fp4_weight"] == {
        "tensor": "layers.4.ffn.experts.0.w1.weight",
        "file_start": 368_799_304,
        "file_stop": 368_801_352,
        "byte_count": 2_048,
        "row_start": 0,
        "row_count": 1,
    }
    assert ranges["fp8_scale"] == {
        "tensor": "layers.4.ffn.shared_experts.w1.scale",
        "file_start": 228_288_584,
        "file_stop": 228_288_616,
        "byte_count": 32,
        "row_start": 0,
        "row_count": 1,
    }


def test_descriptor_drift_is_refused_before_xet_transport() -> None:
    header = _header()
    header["layers.4.ffn.experts.0.w1.scale"]["shape"] = [2048, 127]
    with pytest.raises(fixture.NativeFixtureError, match="extent|differs"):
        fixture._descriptors_from_header(header)


def test_header_capture_requires_exact_hash_and_no_body(tmp_path) -> None:
    header = json.dumps(_header(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    path = tmp_path / "header"
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(fixture.NativeFixtureError, match="SHA-256"):
        fixture._read_header_capture(path)


def test_receipt_cannot_be_placed_inside_retention_root(tmp_path) -> None:
    root = tmp_path / "xet"
    root.mkdir()
    assert fixture._path_within(root / "receipt.json", root)
    assert not fixture._path_within(tmp_path / "receipt.json", root)


def test_xet_metadata_constants_are_specific_and_not_placeholder_values() -> None:
    assert fixture.SHARD_SIZE_BYTES == 3_590_024_776
    assert len(fixture.SHARD_ETAG_SHA256) == 64
    assert len(fixture.XET_FILE_HASH) == 64
    assert fixture.FIXTURE_STATUS == "BYTE_FIXTURE_DECODED_NOT_SOURCE_EXACT"

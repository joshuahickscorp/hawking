"""Focused controls for bounded, zero-cache DeepSeek-V4 Xet tensor slices."""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from lab.operators import deepseek_v4_stream_executor as header_stream
from lab.operators import deepseek_v4_xet_slice as xet_slice


def _plenty(_root: Path) -> int:
    return header_stream.MIN_FREE_FLOOR_BYTES + 32 * 1024**2


def _write_header(path: Path) -> int:
    rows = {
        "layers.4.ffn.experts.0.w1.weight": {
            "dtype": "I8", "shape": [2, 2], "data_offsets": [0, 4]
        },
        "layers.4.ffn.experts.0.w1.scale": {
            "dtype": "F8_E8M0", "shape": [2, 1], "data_offsets": [4, 6]
        },
        "layers.4.attn.indexer.wq_b.weight": {
            "dtype": "F8_E4M3", "shape": [2, 2], "data_offsets": [6, 10]
        },
        "layers.4.attn.indexer.wq_b.scale": {
            "dtype": "F8_E8M0", "shape": [1, 1], "data_offsets": [10, 11]
        },
        "layers.4.ffn.gate.weight": {
            "dtype": "BF16", "shape": [1, 2], "data_offsets": [11, 15]
        },
    }
    header = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    return len(header) + 8


def _header_receipt(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    capture = tmp_path / "model-00006.header"
    size = _write_header(capture)
    digest = hashlib.sha256(capture.read_bytes()).hexdigest()
    plan = header_stream.build_plan(
        repository=header_stream.EXPECTED_REPOSITORY,
        revision=xet_slice.EXPECTED_REVISION,
        header_ranges=[
            {
                "range_id": "layer4-header",
                "shard": xet_slice.FIXTURE_SHARD,
                "start": 0,
                "end": size,
                "expected_capture_sha256": digest,
            }
        ],
        source_retention_paths=[tmp_path / "header-transport-cache"],
    )
    receipt = header_stream.execute_header_only(
        plan,
        range_id="layer4-header",
        header_capture_path=capture,
        receipt_path=tmp_path / "header-receipt.json",
        workspace_root=tmp_path,
        free_bytes_provider=_plenty,
    )
    return capture, receipt


def _remote() -> dict[str, object]:
    return {
        "commit_hash": xet_slice.EXPECTED_REVISION,
        "etag": xet_slice.FIXTURE_LFS_SHA256,
        "size": xet_slice.FIXTURE_FULL_SIZE_BYTES,
        "xet_hash": "a" * 64,
    }


def test_plan_derives_exact_file_ranges_from_sealed_header(tmp_path: Path) -> None:
    capture, header_receipt = _header_receipt(tmp_path)
    plan = xet_slice.build_plan(
        header_receipt=header_receipt,
        header_capture_path=capture,
        tensor_names=[
            "layers.4.ffn.experts.0.w1.weight",
            "layers.4.ffn.experts.0.w1.scale",
            "layers.4.attn.indexer.wq_b.weight",
            "layers.4.attn.indexer.wq_b.scale",
            "layers.4.ffn.gate.weight",
        ],
        source_retention_path=tmp_path / "slice-xet-home",
    )

    assert xet_slice.validate_plan(plan) == plan
    base = capture.stat().st_size
    assert plan["targets"][0]["start"] == base
    assert plan["targets"][0]["end"] == base + 4
    assert sum(target["length"] for target in plan["targets"]) == 15
    assert plan["transport"]["kind"] == "hf_xet_direct_range_stream"
    assert plan["execution_boundary"]["condense_packing"] == "not_executed"


def test_streams_exact_ranges_in_memory_then_seals_no_decode_receipt(tmp_path: Path) -> None:
    capture, header_receipt = _header_receipt(tmp_path)
    retention = tmp_path / "slice-xet-home"
    plan = xet_slice.build_plan(
        header_receipt=header_receipt,
        header_capture_path=capture,
        tensor_names=[
            "layers.4.ffn.experts.0.w1.weight",
            "layers.4.attn.indexer.wq_b.weight",
        ],
        source_retention_path=retention,
    )
    observed: list[tuple[str, int]] = []

    def stream(target: dict[str, object], _remote_value: dict[str, object]):
        payload = bytes(range(int(target["length"])))
        observed.append((str(target["name"]), len(payload)))
        return [payload[:1], payload[1:]]

    receipt = xet_slice.execute_plan(
        plan,
        workspace_root=tmp_path,
        metadata_provider=_remote,
        stream_factory=stream,
        free_bytes_provider=_plenty,
    )

    assert receipt["status"] == "RANGE_BYTES_SEALED_NOT_DECODED"
    assert receipt["payload_bytes"] == 8
    assert observed == [
        ("layers.4.ffn.experts.0.w1.weight", 4),
        ("layers.4.attn.indexer.wq_b.weight", 4),
    ]
    assert receipt["range_results"][0]["sha256"] == hashlib.sha256(bytes(range(4))).hexdigest()
    assert receipt["source_eviction_assertion"]["source_range_files_retained_zero"] is True
    assert receipt["execution_boundary"]["native_fp4_decode"] == "not_executed"
    assert not retention.exists()


def test_wrong_pinned_lfs_identity_refuses_before_opening_range_stream(tmp_path: Path) -> None:
    capture, header_receipt = _header_receipt(tmp_path)
    plan = xet_slice.build_plan(
        header_receipt=header_receipt,
        header_capture_path=capture,
        tensor_names=["layers.4.ffn.experts.0.w1.weight"],
        source_retention_path=tmp_path / "slice-xet-home",
    )
    remote = _remote()
    remote["etag"] = "b" * 64

    def must_not_stream(*_args: object):
        raise AssertionError("stream must not start on identity mismatch")

    with pytest.raises(xet_slice.DeepSeekV4XetSliceError, match="ETag"):
        xet_slice.execute_plan(
            plan,
            workspace_root=tmp_path,
            metadata_provider=lambda: remote,
            stream_factory=must_not_stream,
            free_bytes_provider=_plenty,
        )


def test_header_digest_substitution_and_large_body_plan_are_refused(tmp_path: Path) -> None:
    capture, header_receipt = _header_receipt(tmp_path)
    capture.write_bytes(capture.read_bytes() + b"x")
    with pytest.raises(xet_slice.DeepSeekV4XetSliceError, match="header_capture"):
        xet_slice.build_plan(
            header_receipt=header_receipt,
            header_capture_path=capture,
            tensor_names=["layers.4.ffn.experts.0.w1.weight"],
            source_retention_path=tmp_path / "slice-xet-home",
        )

"""Focused contract tests for bounded, header-only DeepSeek-V4 streaming."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from lab.operators import deepseek_v4_stream_executor as stream


REVISION = "a" * 40


def _header_capture(path: Path, *, extra: bytes = b"") -> int:
    header = json.dumps(
        {
            "layers.4.ffn.experts.0.w1.weight": {
                "dtype": "I8",
                "shape": [1],
                "data_offsets": [0, 1],
            },
            "__metadata__": {"format": "pt"},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + extra)
    return 8 + len(header)


def _plan(tmp_path: Path, *, capture_bytes: int, retention: bool = True) -> dict[str, object]:
    paths = [tmp_path / "transport-cache"] if retention else []
    return stream.build_plan(
        repository=stream.EXPECTED_REPOSITORY,
        revision=REVISION,
        header_ranges=[
            {
                "range_id": "layer4-header",
                "shard": "model-00006-of-00046.safetensors",
                "start": 0,
                "end": capture_bytes,
            }
        ],
        source_retention_paths=paths,
    )


def _plenty(_root: Path) -> int:
    return stream.MIN_FREE_FLOOR_BYTES + 20 * stream.MAX_RECEIPT_BYTES


def test_plan_is_sealed_header_only_and_nonexecuting(tmp_path: Path) -> None:
    capture = tmp_path / "header.capture"
    size = _header_capture(capture)
    plan = _plan(tmp_path, capture_bytes=size)

    assert plan["schema"] == stream.PLAN_SCHEMA
    assert plan["status"] == "REQUESTED_NOT_EXECUTED"
    assert plan["modes"]["tensor_body_streaming"] is False
    assert plan["execution_boundary"]["condense_packing"] == "not_implemented"
    assert stream.validate_plan(plan) == plan

    preflight = stream.preflight_plan(
        plan, workspace_root=tmp_path, free_bytes_provider=_plenty
    )
    assert preflight["status"] == "PLAN_ONLY_NOT_EXECUTED"
    assert preflight["execution_boundary"]["metal_forward"] == "not_executed"


def test_tensor_body_plan_is_refused_before_transport() -> None:
    with pytest.raises(stream.DeepSeekV4StreamError, match="tensor/body ranges are not implemented"):
        stream.build_plan(
            repository=stream.EXPECTED_REPOSITORY,
            revision=REVISION,
            header_ranges=[
                {
                    "range_id": "body",
                    "shard": "model-00006-of-00046.safetensors",
                    "kind": "tensor_body",
                    "start": 0,
                    "end": 128,
                }
            ],
        )


def test_header_capture_seals_atomic_receipt_with_floor_and_eviction_assertions(tmp_path: Path) -> None:
    capture = tmp_path / "header.capture"
    size = _header_capture(capture)
    plan = _plan(tmp_path, capture_bytes=size)
    receipt_path = tmp_path / "receipts" / "layer4-header.json"

    receipt = stream.execute_header_only(
        plan,
        range_id="layer4-header",
        header_capture_path=capture,
        receipt_path=receipt_path,
        workspace_root=tmp_path,
        chunk_bytes=17,
        free_bytes_provider=_plenty,
    )

    assert receipt_path.is_file()
    saved = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert stream.validate_header_receipt(saved) == receipt
    assert receipt["status"] == "HEADER_CAPTURE_SEALED_NOT_ADMITTED"
    assert receipt["header_capture"]["tensor_body_bytes_present"] == 0
    assert receipt["execution_boundary"]["native_fp4_decode"] == "not_implemented"
    assert receipt["source_eviction_assertion"]["status"] == "PASS"
    stages = {row["stage"] for row in receipt["floor_checks"]}
    assert {"before_range", "during_range_before_chunk", "during_range_after_chunk", "after_range"} <= stages
    assert any(path.name.startswith(".") for path in receipt_path.parent.iterdir()) is False

    with pytest.raises(stream.DeepSeekV4StreamError, match="already exists"):
        stream.execute_header_only(
            plan,
            range_id="layer4-header",
            header_capture_path=capture,
            receipt_path=receipt_path,
            workspace_root=tmp_path,
            free_bytes_provider=_plenty,
        )


def test_header_execution_requires_a_declared_empty_retention_path(tmp_path: Path) -> None:
    capture = tmp_path / "header.capture"
    size = _header_capture(capture)
    plan = _plan(tmp_path, capture_bytes=size, retention=False)

    with pytest.raises(stream.DeepSeekV4StreamError, match="without declared source_retention_paths"):
        stream.execute_header_only(
            plan,
            range_id="layer4-header",
            header_capture_path=capture,
            receipt_path=tmp_path / "receipt.json",
            workspace_root=tmp_path,
            free_bytes_provider=_plenty,
        )


def test_nonempty_transport_cache_refuses_instead_of_claiming_eviction(tmp_path: Path) -> None:
    capture = tmp_path / "header.capture"
    size = _header_capture(capture)
    plan = _plan(tmp_path, capture_bytes=size)
    cache = tmp_path / "transport-cache"
    cache.mkdir()
    (cache / "body.safetensors").write_bytes(b"not evicted")

    with pytest.raises(stream.DeepSeekV4StreamError, match="eviction is unproven"):
        stream.execute_header_only(
            plan,
            range_id="layer4-header",
            header_capture_path=capture,
            receipt_path=tmp_path / "receipt.json",
            workspace_root=tmp_path,
            free_bytes_provider=_plenty,
        )


def test_directory_only_xet_cache_scaffolding_is_not_treated_as_source_retention(tmp_path: Path) -> None:
    capture = tmp_path / "header.capture"
    size = _header_capture(capture)
    plan = _plan(tmp_path, capture_bytes=size)
    cache = tmp_path / "transport-cache"
    (cache / "xet-session" / "staging").mkdir(parents=True)

    receipt = stream.execute_header_only(
        plan,
        range_id="layer4-header",
        header_capture_path=capture,
        receipt_path=tmp_path / "receipt.json",
        workspace_root=tmp_path,
        free_bytes_provider=_plenty,
    )

    assertion = receipt["source_eviction_assertion"]["before_range"][0]
    assert assertion["state"] == "DIRECTORY_SCAFFOLDING_ONLY"
    assert assertion["directory_count"] == 3
    assert assertion["retained_file_count"] == 0


def test_symlink_in_transport_cache_refuses_instead_of_traversing_it(tmp_path: Path) -> None:
    capture = tmp_path / "header.capture"
    size = _header_capture(capture)
    plan = _plan(tmp_path, capture_bytes=size)
    cache = tmp_path / "transport-cache"
    cache.mkdir()
    (cache / "link").symlink_to(tmp_path)

    with pytest.raises(stream.DeepSeekV4StreamError, match="contains a symlink"):
        stream.execute_header_only(
            plan,
            range_id="layer4-header",
            header_capture_path=capture,
            receipt_path=tmp_path / "receipt.json",
            workspace_root=tmp_path,
            free_bytes_provider=_plenty,
        )


def test_floor_refuses_before_read_and_writes_no_receipt(tmp_path: Path) -> None:
    capture = tmp_path / "header.capture"
    size = _header_capture(capture)
    plan = _plan(tmp_path, capture_bytes=size)
    receipt_path = tmp_path / "receipt.json"
    inflight = plan["storage_policy"]["max_inflight_bytes"]

    def too_low(_root: Path) -> int:
        return stream.MIN_FREE_FLOOR_BYTES + inflight - 1

    with pytest.raises(stream.DeepSeekV4StreamError, match="floor crossed at before_range"):
        stream.execute_header_only(
            plan,
            range_id="layer4-header",
            header_capture_path=capture,
            receipt_path=receipt_path,
            workspace_root=tmp_path,
            free_bytes_provider=too_low,
        )
    assert not receipt_path.exists()


def test_capture_with_tensor_body_bytes_is_refused(tmp_path: Path) -> None:
    capture = tmp_path / "header.capture"
    header_bytes = _header_capture(capture, extra=b"body")
    plan = _plan(tmp_path, capture_bytes=header_bytes + 4)

    with pytest.raises(stream.DeepSeekV4StreamError, match="never tensor bytes"):
        stream.execute_header_only(
            plan,
            range_id="layer4-header",
            header_capture_path=capture,
            receipt_path=tmp_path / "receipt.json",
            workspace_root=tmp_path,
            free_bytes_provider=_plenty,
        )


def test_terminal_capture_symlink_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "header.capture"
    size = _header_capture(target)
    plan = _plan(tmp_path, capture_bytes=size)
    link = tmp_path / "header-link.capture"
    link.symlink_to(target)

    with pytest.raises(stream.DeepSeekV4StreamError, match="non-symlink"):
        stream.execute_header_only(
            plan,
            range_id="layer4-header",
            header_capture_path=link,
            receipt_path=tmp_path / "receipt.json",
            workspace_root=tmp_path,
            free_bytes_provider=_plenty,
        )

"""Fail-closed contracts for the complete DeepSeek-V4 source stream."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import deepseek_v4_gravity as gravity


def _paths(root: Path) -> dict[str, Path]:
    chunks = root / "chunks"
    metadata = root / "metadata"
    chunks.mkdir()
    metadata.mkdir()
    return {
        "chunks": chunks,
        "metadata": metadata,
        "journal": root / "stream-journal.json",
        "ranges": root / "stream-ranges.jsonl",
        "restart": root / "restart-receipt.json",
        "manifest": root / "manifest.json",
    }


def test_full_stream_constants_bind_the_pinned_index() -> None:
    assert len(gravity.FULL_SHARDS) == 46
    assert gravity.FULL_SHARDS[0] == "model-00001-of-00046.safetensors"
    assert gravity.FULL_SHARDS[-1] == "model-00046-of-00046.safetensors"
    assert gravity.FULL_EXPECTED_TENSOR_COUNT == 69187


def test_full_journal_is_distinct_from_diagnostic_contract(tmp_path: Path) -> None:
    journal = gravity._initial_journal(
        tmp_path,
        tmp_path,
        gravity.MIN_FREE_FLOOR_BYTES,
        gravity.DEFAULT_RANGE_BYTES,
        shard_names=gravity.FULL_SHARDS,
        contract="full_model",
    )
    assert journal["diagnostic_contract"]["kind"] == "full_43_layer_source_stream"
    assert journal["diagnostic_contract"]["not_full_model"] is False
    assert len(journal["diagnostic_contract"]["shards"]) == 46


def test_full_index_rejects_old_alias_count_before_body_stream(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    index = {"metadata": {"total_size": 159609485896}, "weight_map": {"alias": gravity.FULL_SHARDS[0]}}

    def fake_metadata(path: str):
        assert path == "model.safetensors.index.json"
        raw = json.dumps(index).encode("utf-8")
        return raw, {"path": path, "bytes": len(raw), "sha256": gravity._sha256(raw)}

    monkeypatch.setattr(gravity, "_direct_metadata_file", fake_metadata)
    with pytest.raises(gravity.DeepSeekV4GravityError, match="does not describe"):
        gravity._write_full_index(paths)

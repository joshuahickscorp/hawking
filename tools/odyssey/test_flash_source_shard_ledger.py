from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.odyssey import flash_repeated_accepted_decode as gate
from tools.odyssey import flash_source_shard_ledger as ledger


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _source(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    lake = tmp_path / "lake"
    root = lake / "specimens" / gate.MODEL_LAKE_SLUG
    manifest = lake / "manifests" / f"{gate.MODEL_LAKE_SLUG}.json"
    root.mkdir(parents=True)
    shard = root / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"exact source shard")
    _write_json(root / "model.safetensors.index.json", {"weight_map": {"weight": shard.name}})
    _write_json(manifest, {
        "repo": gate.REPO_ID,
        "revision": gate.PINNED_REVISION,
        "resolved_sha": gate.PINNED_REVISION,
        "path": str(root),
        "bytes": shard.stat().st_size,
        "n_files": 2,
    })
    metadata = root / ".cache" / "huggingface" / "download" / f"{shard.name}.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        f"{gate.PINNED_REVISION}\n{hashlib.sha256(shard.read_bytes()).hexdigest()}\n0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gate, "MODEL_LAKE_ROOT", lake)
    return root, shard


def test_builds_resumable_ledger_accepted_by_source_gate(tmp_path: Path, monkeypatch) -> None:
    root, _ = _source(tmp_path, monkeypatch)
    out = tmp_path / "ledger.json"

    receipt = ledger.build_source_shard_ledger(root, out)

    assert receipt["status"] == gate.SOURCE_SHARD_LEDGER_STATUS
    assert receipt["summary"]["indexed_shard_count"] == 1
    binding = {
        "path": str(out),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "schema": gate.SOURCE_SHARD_LEDGER_SCHEMA,
        "status": gate.SOURCE_SHARD_LEDGER_STATUS,
        "seal_sha256": receipt["seal_sha256"],
    }
    verified = gate._verify_shard_ledger(
        binding,
        contract_dir=tmp_path,
        model_root=root,
        manifest_sha256=receipt["source"]["model_lake_manifest_sha256"],
        index_path=root / "model.safetensors.index.json",
        index_sha256=receipt["source"]["safetensors_index_sha256"],
    )
    assert verified["indexed_shard_count"] == 1


def test_rejects_source_bytes_that_differ_from_retained_lfs_identity(tmp_path: Path, monkeypatch) -> None:
    root, shard = _source(tmp_path, monkeypatch)
    metadata = root / ".cache" / "huggingface" / "download" / f"{shard.name}.metadata"
    metadata.write_text(f"{gate.PINNED_REVISION}\n{'0' * 64}\n0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from its retained Hub LFS identity"):
        ledger.build_source_shard_ledger(root, tmp_path / "ledger.json")


def test_refuses_changed_source_when_resuming_progress(tmp_path: Path, monkeypatch) -> None:
    root, shard = _source(tmp_path, monkeypatch)
    first_out = tmp_path / "first.json"
    ledger.build_source_shard_ledger(root, first_out)
    progress = first_out.with_name(f"{first_out.name}.progress.json")
    shard.write_bytes(b"changed source shard")

    with pytest.raises(ValueError, match="different source inventory"):
        ledger.build_source_shard_ledger(root, tmp_path / "second.json", progress_path=progress)

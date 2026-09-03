"""Focused integrity and restart coverage for the complete binary compiler."""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from lab.operators import ascension_qwen30_complete_gravity as complete
from lab.receipts import seal, verify


def _write_small_source(tmp_path: Path) -> tuple[complete.CompleteBinaryGravity, Path, Path]:
    """Create a tiny, real safetensors-layout source plus a sealed audit."""

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard = model_dir / "model-00001-of-00001.safetensors"
    tensors = {
        "model.layers.0.first.weight": np.arange(4, dtype="<f4").reshape(2, 2),
        "model.layers.0.second.weight": np.arange(6, dtype="<f4").reshape(2, 3),
        "model.layers.0.third.weight": np.arange(3, dtype="<f4"),
    }
    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for name, values in tensors.items():
        raw = values.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(values.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        payload.extend(raw)
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    shard_hash = hashlib.sha256(shard.read_bytes()).hexdigest()
    index = {"weight_map": {name: shard.name for name in tensors}}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    audit_path = tmp_path / "source-audit.json"
    audit = seal(
        {
            "schema": "test.source.audit.v1",
            "files": [
                {
                    "path": "model.safetensors.index.json",
                    "bytes": (model_dir / "model.safetensors.index.json").stat().st_size,
                    "sha256": hashlib.sha256(
                        (model_dir / "model.safetensors.index.json").read_bytes()
                    ).hexdigest(),
                }
            ],
            "source": {
                "repository": "test/repository",
                "revision": "pinned-test-revision",
                "model_dir": str(model_dir),
                "shards": {
                    shard.name: {
                        "bytes": shard.stat().st_size,
                        "sha256": shard_hash,
                    }
                },
            },
        }
    )
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    worker = complete.CompleteBinaryGravity(
        model_dir=model_dir,
        source_audit=audit_path,
        root=tmp_path / "complete-gravity",
        repository="test/repository",
        model_id="test-model",
        artifact_prefix="TEST",
    )
    return worker, shard, audit_path


def _write_three_shard_source(tmp_path: Path) -> tuple[complete.CompleteBinaryGravity, tuple[Path, ...]]:
    """Create ordered one-tensor shards so header opens are observable per batch."""

    model_dir = tmp_path / "three-shard-model"
    model_dir.mkdir()
    weight_map: dict[str, str] = {}
    shard_evidence: dict[str, dict[str, object]] = {}
    shards: list[Path] = []
    for ordinal, label in enumerate(("first", "second", "third"), start=1):
        tensor_name = f"model.layers.0.{label}.weight"
        shard = model_dir / f"model-{ordinal:05d}-of-00003.safetensors"
        values = np.asarray([ordinal, ordinal + 0.5], dtype="<f4")
        raw = values.tobytes()
        header = {
            tensor_name: {
                "dtype": "F32",
                "shape": list(values.shape),
                "data_offsets": [0, len(raw)],
            }
        }
        header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        shard.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + raw)
        weight_map[tensor_name] = shard.name
        shard_evidence[shard.name] = {
            "bytes": shard.stat().st_size,
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
        }
        shards.append(shard)
    index_path = model_dir / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    audit_path = tmp_path / "three-shard-audit.json"
    audit = seal(
        {
            "schema": "test.source.audit.v1",
            "files": [
                {
                    "path": index_path.name,
                    "bytes": index_path.stat().st_size,
                    "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                }
            ],
            "source": {
                "repository": "test/repository",
                "revision": "pinned-test-revision",
                "shards": shard_evidence,
            },
        }
    )
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return complete.CompleteBinaryGravity(
        model_dir=model_dir,
        source_audit=audit_path,
        root=tmp_path / "three-shard-complete-gravity",
        repository="test/repository",
        model_id="test-model",
        artifact_prefix="TEST3",
    ), tuple(shards)


def _write_terminal_source(tmp_path: Path) -> complete.CompleteBinaryGravity:
    """One 128-value tensor keeps the legacy self-billed manifest fixture stable."""

    model_dir = tmp_path / "terminal-model"
    model_dir.mkdir()
    shard = model_dir / "model-00001-of-00001.safetensors"
    values = np.arange(128, dtype="<f4")
    raw = values.tobytes()
    header = {
        "model.layers.0.terminal.weight": {
            "dtype": "F32",
            "shape": list(values.shape),
            "data_offsets": [0, len(raw)],
        }
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + raw)
    index_path = model_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": {"model.layers.0.terminal.weight": shard.name}}),
        encoding="utf-8",
    )
    audit_path = tmp_path / "terminal-audit.json"
    audit_path.write_text(
        json.dumps(
            seal(
                {
                    "schema": "test.source.audit.v1",
                    "files": [
                        {
                            "path": index_path.name,
                            "bytes": index_path.stat().st_size,
                            "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                        }
                    ],
                    "source": {
                        "repository": "test/repository",
                        "revision": "pinned-test-revision",
                        "shards": {
                            shard.name: {
                                "bytes": shard.stat().st_size,
                                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                            }
                        },
                    },
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return complete.CompleteBinaryGravity(
        model_dir=model_dir,
        source_audit=audit_path,
        root=tmp_path / "terminal-complete-gravity",
        repository="test/repository",
        model_id="test-model",
        artifact_prefix="TERMINAL",
    )


def test_revalidation_receipt_hashes_all_current_shards_once_and_is_audit_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, shard, audit_path = _write_small_source(tmp_path)
    original = complete._sha256_file
    calls: list[Path] = []

    def counted(path: Path, **kwargs: object) -> str:
        calls.append(Path(path))
        return original(path, **kwargs)

    monkeypatch.setattr(complete, "_sha256_file", counted)
    assert worker.run(max_tensors=1) == 0
    receipt = verify(json.loads(worker.source_revalidation_path.read_text(encoding="utf-8")))
    audit = verify(json.loads(audit_path.read_text(encoding="utf-8")))
    assert receipt["status"] == "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED"
    assert receipt["source_audit_seal_sha256"] == audit["seal_sha256"]
    assert receipt["sealed_audit_index_sha256"] == hashlib.sha256(
        (shard.parent / "model.safetensors.index.json").read_bytes()
    ).hexdigest()
    assert receipt["shards"][shard.name]["expected_sha256"] == hashlib.sha256(shard.read_bytes()).hexdigest()
    assert calls.count(shard) == 1

    # The second bounded batch still hashes its tiny audit/index bindings, but
    # it must reuse the sealed full-shard receipt rather than reread the source.
    assert worker.run(max_tensors=1) == 0
    assert calls.count(shard) == 1


def test_changed_source_identity_forces_full_revalidation_and_refuses_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, shard, _ = _write_small_source(tmp_path)
    original = complete._sha256_file
    calls: list[Path] = []

    def counted(path: Path, **kwargs: object) -> str:
        calls.append(Path(path))
        return original(path, **kwargs)

    monkeypatch.setattr(complete, "_sha256_file", counted)
    assert worker.run(max_tensors=1) == 0
    mutated = bytearray(shard.read_bytes())
    mutated[-1] ^= 1
    shard.write_bytes(mutated)

    with pytest.raises(complete.CompleteGravityError, match="SHA-256 mismatch"):
        worker.run(max_tensors=1)
    assert calls.count(shard) == 2


def test_compact_progress_index_avoids_legacy_jsonl_reparse_on_next_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, _, _ = _write_small_source(tmp_path)
    assert worker.run(max_tensors=1) == 0
    indexed = verify(json.loads(worker.progress_index_path.read_text(encoding="utf-8")))
    assert indexed["completed_tensor_count"] == 1
    assert indexed["journal"]["indexed_bytes"] == worker.progress_path.stat().st_size

    def should_not_rebuild(**_: object) -> object:
        raise AssertionError("a valid progress index should make the next batch tail-only")

    monkeypatch.setattr(worker, "_rebuild_progress_index", should_not_rebuild)
    assert worker.run(max_tensors=1) == 0
    indexed_after = verify(json.loads(worker.progress_index_path.read_text(encoding="utf-8")))
    assert indexed_after["completed_tensor_count"] == 2
    assert indexed_after["journal"]["indexed_bytes"] == worker.progress_path.stat().st_size


def test_progress_index_recovers_a_durable_jsonl_tail_before_its_next_index_write(
    tmp_path: Path,
) -> None:
    """A crash after fsync(journal) but before atomic index replacement is safe."""

    worker, _, _ = _write_small_source(tmp_path)
    assert worker.run(max_tensors=1) == 0
    audit, weight_map, shard_evidence = worker._admit_source()
    revalidation, _ = worker._revalidate_current_source(
        audit=audit,
        weight_map=weight_map,
        shard_evidence=shard_evidence,
    )
    second_name = sorted(weight_map)[1]
    second_shard = weight_map[second_name]
    header = worker._header(worker.model_dir / second_shard)
    row = worker._write_tensor(
        tensor_name=second_name,
        shard=second_shard,
        source_hash=str(shard_evidence[second_shard]["sha256"]),
        info=header[second_name],
    )
    worker._append_progress(row)  # Simulate an exit before _write_progress_index.

    binding = worker._progress_binding(
        audit_seal=str(audit["seal_sha256"]),
        revalidation_seal=str(revalidation["seal_sha256"]),
    )
    planned_order = worker._planned_tensor_order(weight_map)
    recovered, journal, scheduler = worker._load_progress_index(
        binding=binding,
        planned_order=planned_order,
        shard_evidence=shard_evidence,
    )
    assert set(recovered) == set(sorted(weight_map)[:2])
    assert journal["indexed_bytes"] == worker.progress_path.stat().st_size
    assert scheduler["next_cursor"] == 2
    rebuilt = verify(json.loads(worker.progress_index_path.read_text(encoding="utf-8")))
    assert rebuilt["completed_tensor_count"] == 2


def test_later_batch_starts_at_durable_cursor_without_opening_prior_shard_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, shards = _write_three_shard_source(tmp_path)
    assert worker.run(max_tensors=1) == 0
    indexed = verify(json.loads(worker.progress_index_path.read_text(encoding="utf-8")))
    assert indexed["scheduler"]["next_cursor"] == 1
    # Existing live indexes predate the cursor field.  A normal upgrade must
    # derive it from the compact map, not fall back to a full JSONL reparse.
    legacy_index = {
        key: value
        for key, value in indexed.items()
        if key not in {"seal_sha256", "scheduler"}
    }
    worker.progress_index_path.write_text(
        json.dumps(seal(legacy_index), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    opened_headers: list[str] = []
    published: list[tuple[str, dict[str, object]]] = []
    original_header = worker._header
    original_publish = worker._publish

    def traced_header(path: Path) -> dict[str, object]:
        opened_headers.append(path.name)
        return original_header(path)

    def traced_publish(phase: str, **fields: object) -> None:
        published.append((phase, fields))
        original_publish(phase, **fields)

    def should_not_rebuild(**_: object) -> object:
        raise AssertionError("legacy compact index should migrate its cursor without JSONL reparsing")

    monkeypatch.setattr(worker, "_header", traced_header)
    monkeypatch.setattr(worker, "_publish", traced_publish)
    monkeypatch.setattr(worker, "_rebuild_progress_index", should_not_rebuild)
    assert worker.run(max_tensors=1) == 0

    initial_pack = next(fields for phase, fields in published if phase == "PACKING_COMPLETE_BINARY_GRAVITY")
    progress = initial_pack["progress"]
    assert isinstance(progress, dict)
    assert progress["completed_tensors"] == 1
    assert progress["next_cursor"] == 1
    assert opened_headers == [shards[1].name]
    indexed_after = verify(json.loads(worker.progress_index_path.read_text(encoding="utf-8")))
    assert indexed_after["scheduler"]["next_cursor"] == 2


def test_terminal_status_is_sealed_and_reuses_the_complete_manifest_across_keepalive_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final bounded batch must terminalize once, not reseal forever."""

    worker = _write_terminal_source(tmp_path)
    # This specifically exercises the old final-batch bug: reaching the
    # max-tensors boundary used to return while still publishing PACKING.
    assert worker.run(max_tensors=1) == 0
    manifest_bytes = worker.manifest_path.read_bytes()
    terminal_status = verify(json.loads(worker.status_path.read_text(encoding="utf-8")))
    terminal_receipt = verify(json.loads(worker.terminal_receipt_path.read_text(encoding="utf-8")))
    assert terminal_status["phase"] == complete.COMPLETE_CANDIDATE_PHASE
    assert terminal_status["terminal_status_is_sealed"] is True
    assert terminal_status["manifest_seal_sha256"] == terminal_receipt["candidate"]["manifest_seal_sha256"]
    assert terminal_receipt["schema"] == complete.TERMINAL_STATUS_SCHEMA
    assert terminal_receipt["binding"]["progress"]["next_cursor"] == 1

    # Simulate the already-complete Q30 lane produced before terminal receipts
    # existed.  Migration must validate and retain the manifest rather than
    # rebuild it just to refresh a heartbeat.
    worker.terminal_receipt_path.unlink()

    def must_not_rebuild_manifest() -> dict[str, dict[str, object]]:
        raise AssertionError("a valid completed manifest must not be rebuilt during terminal migration")

    monkeypatch.setattr(worker, "_full_progress_rows", must_not_rebuild_manifest)
    assert worker.run(max_tensors=1) == 0
    assert worker.manifest_path.read_bytes() == manifest_bytes
    migrated_receipt_bytes = worker.terminal_receipt_path.read_bytes()
    migrated_status = verify(json.loads(worker.status_path.read_text(encoding="utf-8")))
    assert migrated_status["phase"] == complete.COMPLETE_CANDIDATE_PHASE
    assert migrated_status["terminal_receipt_seal_sha256"] == verify(
        json.loads(migrated_receipt_bytes)
    )["seal_sha256"]

    # A normal launchd KeepAlive restart now uses the static terminal seal and
    # never calls the manifest admission/rebuild path or changes its bytes.
    def must_not_revalidate_manifest(**_: object) -> dict[str, object]:
        raise AssertionError("current terminal receipt should avoid manifest re-admission")

    monkeypatch.setattr(worker, "_admit_existing_complete_manifest", must_not_revalidate_manifest)
    assert worker.run(max_tensors=1) == 0
    assert worker.manifest_path.read_bytes() == manifest_bytes
    assert worker.terminal_receipt_path.read_bytes() == migrated_receipt_bytes
    assert verify(json.loads(worker.status_path.read_text(encoding="utf-8")))["phase"] == (
        complete.COMPLETE_CANDIDATE_PHASE
    )


def test_changed_terminal_manifest_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal receipt is only a cache while the exact manifest identity survives."""

    worker = _write_terminal_source(tmp_path)
    assert worker.run(max_tensors=1) == 0
    worker.manifest_path.write_text('{"tampered":true}\n', encoding="utf-8")

    def rebuild_is_required() -> dict[str, dict[str, object]]:
        raise AssertionError("changed manifest correctly fell through to a full rebuild")

    monkeypatch.setattr(worker, "_full_progress_rows", rebuild_is_required)
    with pytest.raises(AssertionError, match="changed manifest correctly fell through"):
        worker.run(max_tensors=1)

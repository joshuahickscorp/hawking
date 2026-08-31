"""External specimen seal: identity is a tree digest, not a location.

Negative controls the rest of the module is worthless without:

* an external specimen with no whole-tree verification is not sealable;
* changing one byte in a fixture tree changes the digest;
* reordering the manifest does not;
* a lake specimen still needs what it needed before — the external path
  must not open a hole in the lake rule.

Safetensors are never re-hashed in these tests. The live 55.6 GB
verification is recovered, not repeated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.future import external_specimen_seal as ess
from tools.future import odyssey_launch as ol
from tools.future import specimen_verify as sv
from tools.future._common import RECEIPTS


def _local_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(ess, "RECEIPTS", tmp_path)

    def _write(name, doc, recorded_by):
        out = tmp_path / name
        out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return out

    monkeypatch.setattr(ess, "write_receipt", _write)


def _fixture_tree(root: Path, files: dict[str, bytes]) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (root / name).write_bytes(body)
    return ess.manifest_from_directory(root)


def _verified_row(*, files: list[dict], **extra) -> dict:
    n = len(files)
    bytes_hashed = sum(int(f["bytes"]) for f in files)
    row = {
        "specimen": ess.SPECIMEN_NAME,
        "owner": "local_directory",
        "specimen_path": str(sv.EXTRA_SPECIMENS[ess.SPECIMEN_NAME]),
        "status": "WHOLE_TREE_VERIFIED",
        "whole_tree_verified": True,
        "n_files": n,
        "verified": n,
        "mismatched": 0,
        "no_remote_digest": 0,
        "unrecognized_digest": 0,
        "skipped_time_budget": 0,
        "bytes_hashed": bytes_hashed,
        "files": files,
    }
    row.update(extra)
    return row


def test_tree_digest_is_byte_identical_on_a_second_pass(tmp_path):
    files = {"a.bin": b"alpha", "b.txt": b"beta\n", "cfg.json": b"{}"}
    root = tmp_path / "t1"
    m1 = _fixture_tree(root, files)
    m2 = ess.manifest_from_directory(root)
    assert ess.tree_digest(m1) == ess.tree_digest(m2)
    assert len(ess.tree_digest(m1)) == 64


def test_negative_control_one_byte_change_changes_the_digest(tmp_path):
    root = tmp_path / "t"
    _fixture_tree(root, {"w.bin": b"weights", "c.json": b'{"n":1}'})
    before = ess.tree_digest(ess.manifest_from_directory(root))
    (root / "w.bin").write_bytes(b"weightt")
    after = ess.tree_digest(ess.manifest_from_directory(root))
    assert before != after


def test_negative_control_reordering_the_manifest_does_not_change_the_digest():
    rows = [
        {"path": "z.bin", "sha256": "ab" * 32, "bytes": 3},
        {"path": "a.bin", "sha256": "cd" * 32, "bytes": 1},
        {"path": "m.bin", "sha256": "ef" * 32, "bytes": 2},
    ]
    assert ess.tree_digest(rows) == ess.tree_digest(list(reversed(rows)))
    canonical = ess.canonicalize_manifest(rows)
    assert canonical.decode().splitlines()[0].startswith("a.bin\t")


def test_negative_control_unverified_external_is_not_sealable():
    row = _verified_row(
        files=[
            {
                "file": "only.bin",
                "bytes": 4,
                "digest_kind": "sha256",
                "expected": "aa" * 32,
                "actual": "aa" * 32,
                "verdict": "VERIFIED",
            }
        ],
        status="PARTIAL_NO_REMOTE_DIGEST",
        whole_tree_verified=False,
        no_remote_digest=1,
        verified=0,
        n_files=1,
        bytes_hashed=0,
    )
    with pytest.raises(ess.SealError, match="no whole-tree verification"):
        ess.seal_from_verification(row, spec_dir=None, hash_small=False)


def test_negative_control_status_label_alone_is_not_verification():
    """WHOLE_TREE_VERIFIED with a mismatch is a hypothesis, not a seal."""
    digest = hashlib.sha256(b"x").hexdigest()
    row = _verified_row(
        files=[
            {
                "file": "w.bin",
                "bytes": 1,
                "digest_kind": "sha256",
                "expected": digest,
                "actual": "ff" * 32,
                "verdict": "MISMATCH",
            }
        ],
        mismatched=1,
        verified=0,
        bytes_hashed=1,
    )
    with pytest.raises(ess.SealError, match="no whole-tree verification"):
        ess.seal_from_verification(row, spec_dir=None, hash_small=False)


def test_negative_control_absent_verification_row_is_not_sealable():
    with pytest.raises(ess.SealError, match="no verification row"):
        ess.seal_from_verification(None)


def test_negative_control_a_different_specimen_is_not_this_seal():
    digest = hashlib.sha256(b"x").hexdigest()
    row = _verified_row(
        files=[
            {
                "file": "w.bin",
                "bytes": 1,
                "digest_kind": "sha256",
                "expected": digest,
                "actual": digest,
                "verdict": "VERIFIED",
            }
        ],
        specimen="Qwen--Qwen3-0.6B@abc",
        owner="modellake",
    )
    with pytest.raises(ess.SealError, match="authorizes only"):
        ess.seal_from_verification(row, spec_dir=None, hash_small=False)


def test_recovered_sha256_is_used_and_safetensors_are_not_rehashed(tmp_path):
    weight = "aa" * 32
    row = _verified_row(
        files=[
            {
                "file": "model-00001-of-00001.safetensors",
                "bytes": 99,
                "digest_kind": "sha256",
                "expected": weight,
                "actual": weight,
                "verdict": "VERIFIED",
            }
        ]
    )
    manifest = ess.manifest_from_verification(row, spec_dir=tmp_path, hash_small=True)
    assert manifest[0]["sha256"] == weight
    assert manifest[0]["sha256_source"] == "recovered_from_verification_actual"
    # If the seal path had hashed the (absent) safetensor it would have refused.
    assert not (tmp_path / "model-00001-of-00001.safetensors").exists()


def test_git_blob_file_without_rehash_refuses_rather_than_dropping_the_row(tmp_path):
    row = _verified_row(
        files=[
            {
                "file": "config.json",
                "bytes": 2,
                "digest_kind": "git_blob_sha1",
                "expected": "bb" * 20,
                "actual": "bb" * 20,
                "verdict": "VERIFIED",
            }
        ]
    )
    with pytest.raises(ess.SealError, match="no sha256"):
        ess.manifest_from_verification(row, spec_dir=tmp_path, hash_small=False)


def test_small_git_blob_file_is_hashed_readonly(tmp_path):
    body = b'{"model_type":"fixture"}'
    (tmp_path / "config.json").write_bytes(body)
    row = _verified_row(
        files=[
            {
                "file": "config.json",
                "bytes": len(body),
                "digest_kind": "git_blob_sha1",
                "expected": "cc" * 20,
                "actual": "cc" * 20,
                "verdict": "VERIFIED",
            }
        ]
    )
    manifest = ess.manifest_from_verification(row, spec_dir=tmp_path, hash_small=True)
    assert manifest[0]["sha256"] == hashlib.sha256(body).hexdigest()
    assert manifest[0]["sha256_source"] == "recomputed_here"
    assert (tmp_path / "config.json").read_bytes() == body


def test_size_change_on_disk_refuses_the_seal(tmp_path):
    """A small file whose bytes no longer match the verified size is a different tree."""
    (tmp_path / "config.json").write_bytes(b"xx")
    row = _verified_row(
        files=[
            {
                "file": "config.json",
                "bytes": 99,
                "digest_kind": "git_blob_sha1",
                "expected": "dd" * 20,
                "actual": "dd" * 20,
                "verdict": "VERIFIED",
            }
        ]
    )
    with pytest.raises(ess.SealError, match="on-disk size"):
        ess.manifest_from_verification(row, spec_dir=tmp_path, hash_small=True)


def test_model_identity_refuses_an_invented_parameter_count(tmp_path):
    cfg = {
        "architectures": ["FixtureForCausalLM"],
        "model_type": "fixture",
        "text_config": {"hidden_size": 8, "num_hidden_layers": 2, "model_type": "fixture_text"},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    ident = ess.read_model_identity(tmp_path)
    assert ident["ok"] is True
    assert ident["architecture"]["model_type"] == "fixture"
    assert ident["parameter_count"] is None
    assert ident["parameter_count_refused"]
    assert "invent" in ident["parameter_count_refused"]


def test_model_identity_absent_config_fails_closed(tmp_path):
    ident = ess.read_model_identity(tmp_path)
    assert ident["ok"] is False
    assert ident["parameter_count"] is None


def test_empty_directory_is_not_an_identity(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ess.SealError, match="empty tree"):
        ess.manifest_from_directory(empty)


def test_accept_refuses_unverified_external_even_with_a_digest(tmp_path, monkeypatch):
    _local_receipts(tmp_path, monkeypatch)
    identity = {
        "specimen": ess.SPECIMEN_NAME,
        "specimen_owner": "local_directory",
        "specimen_path": str(sv.EXTRA_SPECIMENS[ess.SPECIMEN_NAME]),
        "whole_tree_verified": False,
        "tree_digest": "ab" * 32,
    }
    ok, why = ess.accept_as_sealed_identity(identity)
    assert ok is False
    assert "not whole-tree verified" in why


def test_accept_refuses_when_no_seal_exists(tmp_path, monkeypatch):
    _local_receipts(tmp_path, monkeypatch)
    identity = {
        "specimen": ess.SPECIMEN_NAME,
        "specimen_owner": "local_directory",
        "specimen_path": str(sv.EXTRA_SPECIMENS[ess.SPECIMEN_NAME]),
        "whole_tree_verified": True,
        "tree_digest": "ab" * 32,
    }
    ok, why = ess.accept_as_sealed_identity(identity)
    assert ok is False
    assert "no external specimen seal" in why


def test_accept_refuses_a_mismatched_tree_digest(tmp_path, monkeypatch):
    _local_receipts(tmp_path, monkeypatch)
    seal = {
        "status": "SEALED",
        "tree_digest": "ab" * 32,
        "specimen": ess.SPECIMEN_NAME,
        "specimen_path": str(sv.EXTRA_SPECIMENS[ess.SPECIMEN_NAME]),
    }
    identity = {
        "specimen": ess.SPECIMEN_NAME,
        "specimen_owner": "local_directory",
        "specimen_path": str(sv.EXTRA_SPECIMENS[ess.SPECIMEN_NAME]),
        "whole_tree_verified": True,
        "tree_digest": "cd" * 32,
    }
    ok, why = ess.accept_as_sealed_identity(identity, seal=seal)
    assert ok is False
    assert "does not match" in why


def test_ready_accepts_external_tree_digest_as_sealed_identity(monkeypatch, tmp_path):
    """Teaching _ready is the point. A matching seal is identity for this extra only."""
    digest = "ab" * 32
    seal = {
        "status": "SEALED",
        "tree_digest": digest,
        "specimen": ess.SPECIMEN_NAME,
        "specimen_path": str(sv.EXTRA_SPECIMENS[ess.SPECIMEN_NAME]),
    }
    _local_receipts(tmp_path, monkeypatch)
    (tmp_path / ess.RECEIPT).write_text(json.dumps(seal))
    identity = {
        "specimen": ess.SPECIMEN_NAME,
        "specimen_owner": "local_directory",
        "specimen_path": str(sv.EXTRA_SPECIMENS[ess.SPECIMEN_NAME]),
        "whole_tree_verified": True,
        "tree_digest": digest,
        "revision": None,
        "resolved_sha": None,
        "patient_seal": None,
    }
    ok, why = ess.accept_as_sealed_identity(identity)
    assert ok is True
    assert "tree digest is sealed identity" in why

    ready, ready_why = ol._ready(identity, require_lake_verified=True)
    assert ready is True, ready_why
    assert "tree digest" in ready_why

    # q27_id shape: path and owner, no specimen name, no tree_digest offered.
    q27_shape = {
        "specimen_owner": "local_directory",
        "specimen_path": str(sv.EXTRA_SPECIMENS[ess.SPECIMEN_NAME]),
        "whole_tree_verified": True,
        "revision": None,
        "resolved_sha": None,
        "patient_seal": None,
    }
    ready, ready_why = ol._ready(q27_shape, require_lake_verified=True)
    assert ready is True, ready_why


def test_negative_control_lake_specimen_still_requires_revision():
    """The external path must not open a hole in the lake rule."""
    identity = {
        "specimen_owner": "modellake",
        "in_specimens_listing": True,
        "whole_tree_verified": True,
        "tree_digest": "ab" * 32,
        "specimen_path": "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3-0.6B@abc",
        "revision": None,
        "resolved_sha": None,
        "patient_seal": None,
        "authorized_external": True,
    }
    ready, why = ol._ready(identity, require_lake_verified=True)
    assert ready is False
    assert why == "no sealed revision or patient seal"
    ok, accept_why = ess.accept_as_sealed_identity(
        identity,
        seal={
            "status": "SEALED",
            "tree_digest": "ab" * 32,
            "specimen": ess.SPECIMEN_NAME,
            "specimen_path": identity["specimen_path"],
        },
    )
    assert ok is False
    assert "lake" in accept_why


def test_negative_control_lake_partial_still_refuses_as_before():
    identity = {
        "specimen_owner": "modellake",
        "in_specimens_listing": True,
        "whole_tree_verified": False,
        "n_sha256_verified": 3,
        "n_files": 10,
        "tree_digest": "ab" * 32,
        "revision": "deadbeef",
    }
    ready, why = ol._ready(identity, require_lake_verified=True)
    assert ready is False
    assert "partial" in why


def test_retired_patient_rule_is_untouched():
    verified_and_sealed = {
        "patient_state": "RETIRED",
        "patient_seal": "sha256:abc",
        "whole_tree_verified": True,
        "revision": "r1",
        "tree_digest": "ab" * 32,
    }
    ready, why = ol._ready(verified_and_sealed, require_lake_verified=True)
    assert ready is True
    assert "RECURRENT_PATIENT" in why
    missing_seal = dict(verified_and_sealed, patient_seal=None)
    ready, why = ol._ready(missing_seal, require_lake_verified=True)
    assert ready is False


def test_live_verification_row_is_recoverable_without_rehashing_safetensors():
    """Cope either way: missing verification is a refusal, not a skip."""
    doc = ess.load_verification_doc()
    if doc is None:
        refused = ess.seal_authorized_external()
        assert refused["status"] == "REFUSED"
        return
    row = ess.verification_row(doc, ess.SPECIMEN_NAME)
    if row is None or not ess.is_whole_tree_row(row):
        with pytest.raises(ess.SealError):
            ess.seal_from_verification(row, spec_dir=None, hash_small=False)
        return
    assert row["n_files"] == 31
    assert row["verified"] == 31
    assert row["bytes_hashed"] == 55586059240
    assert row["owner"] == "local_directory"
    safetensors = [
        f for f in row["files"] if str(f.get("file", "")).endswith(".safetensors")
    ]
    assert safetensors, "the parent is the weights"
    recovered = ess.manifest_from_verification(
        {"files": safetensors, **{k: row[k] for k in row if k != "files"}},
        spec_dir=None,
        hash_small=False,
    )
    assert recovered
    assert all(r["sha256_source"] == "recovered_from_verification_actual" for r in recovered)
    assert all(len(r["sha256"]) == 64 for r in recovered)
    # Reordering recovered weights must not change that sub-tree digest.
    assert ess.tree_digest(recovered) == ess.tree_digest(list(reversed(recovered)))


def test_build_writes_a_static_only_receipt_and_does_not_claim_gpu():
    out = ess.build()
    assert out.parent == RECEIPTS
    assert out.name == ess.RECEIPT
    doc = json.loads(out.read_text())
    assert doc["schema"] == ess.SCHEMA
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["gpu_authority"] is False
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["seal_sha256"]
    assert doc["kind"] == "authorized_external_specimen"
    assert doc["not_lake_stock"] is True
    assert doc["read_only_expectation"]
    assert "location_is_not_authority" in doc
    assert doc["status"] in {"SEALED", "REFUSED"}
    if doc["status"] == "SEALED":
        assert len(doc["tree_digest"]) == 64
        assert doc["n_files"] == 31
        assert doc["total_bytes"] == 55586059240
        assert doc["sha256_sources"]["safetensors_rehashed"] is False
        assert doc["model_identity"]["ok"] is True
        assert doc["model_identity"]["architecture"]["model_type"] == "qwen3_5"
        assert doc["model_identity"]["parameter_count"] is None
        assert doc["model_identity"]["parameter_count_refused"]
        assert doc["tokenizer_identity"]["ok"] is True
        assert doc["tokenizer_identity"]["tokenizer_class"] == "Qwen2Tokenizer"
        assert doc["specimen_mutated"] is False
        # In-process reorder identity.
        assert ess.tree_digest(doc["manifest"]) == doc["tree_digest"]
        assert ess.tree_digest(list(reversed(doc["manifest"]))) == doc["tree_digest"]
    else:
        assert doc.get("why")


def test_specimen_is_not_written_by_the_seal_path():
    path = sv.EXTRA_SPECIMENS[ess.SPECIMEN_NAME]
    if not path.is_dir():
        refused = ess.seal_authorized_external()
        assert refused["status"] == "REFUSED"
        return
    crc = path / "crc32.txt"
    before = crc.read_bytes() if crc.is_file() else None
    cfg_mtime = (path / "config.json").stat().st_mtime
    ess.seal_authorized_external()
    assert (path / "config.json").stat().st_mtime == cfg_mtime
    if before is not None:
        assert crc.read_bytes() == before
    assert not (path / ess.RECEIPT).exists()


def test_module_has_no_stub_surface():
    src = Path(ess.__file__).read_text()
    compile(src, ess.__file__, "exec")
    assert "raise NotImplementedError" not in src
    assert "pytest.skip" not in src

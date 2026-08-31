"""Verification that can never be satisfied by a size check or a missing file.

This module exists because `specimen_curriculum_ready` was refused on evidence:
ModelLake reported most files "size only" verified and every seal said
MANIFEST_ONLY. The danger in fixing that is obvious — a verifier that counts a
file it did not hash, or rounds an undigested file into a pass, would turn a
correct refusal into a false readiness and Odyssey I would launch on sand.
"""
import hashlib
import json

import pytest

from tools.future import specimen_verify as sv
from tools.future._common import REPO, RECEIPTS


def test_available_copes_when_the_lake_is_absent():
    a = sv.available()
    assert isinstance(a["mounted"], bool)
    assert isinstance(a["n_specimens"], int)


def test_digest_kind_is_derived_from_the_etag_shape(tmp_path):
    spec = tmp_path / "spec"
    dl = spec / ".cache" / "huggingface" / "download"
    dl.mkdir(parents=True)
    (dl / "a.bin.metadata").write_text("commitsha\n" + "a" * 64 + "\n123.0\n")
    (dl / "b.txt.metadata").write_text("commitsha\n" + "b" * 40 + "\n123.0\n")
    (dl / "c.dat.metadata").write_text("commitsha\nnot-a-digest\n123.0\n")
    assert sv._read_metadata(spec, "a.bin")["digest_kind"] == "sha256"
    assert sv._read_metadata(spec, "b.txt")["digest_kind"] == "git_blob_sha1"
    assert sv._read_metadata(spec, "c.dat")["digest_kind"] == "unrecognized"
    assert sv._read_metadata(spec, "missing.bin") is None


def test_git_blob_sha1_matches_gits_own_rule(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello\n")
    expect = hashlib.sha1(b"blob 6\0hello\n").hexdigest()
    assert sv._git_blob_sha1(f) == expect


def test_sha256_is_recomputed_not_trusted(tmp_path):
    f = tmp_path / "y.bin"
    f.write_bytes(b"0123456789" * 1000)
    assert sv._sha256(f) == hashlib.sha256(b"0123456789" * 1000).hexdigest()


def _fixture_specimen(tmp_path, *, corrupt=False, undigested=False):
    """A specimen laid out exactly like ModelLake's, so the verdict is real."""
    spec = tmp_path / "specimens" / "fixture@abc"
    dl = spec / ".cache" / "huggingface" / "download"
    dl.mkdir(parents=True)
    body = b"weights" * 100
    (spec / "model.safetensors").write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    if corrupt:
        digest = "f" * 64
    (dl / "model.safetensors.metadata").write_text(f"commit\n{digest}\n1.0\n")
    if undigested:
        (spec / "orphan.bin").write_bytes(b"no metadata for me")
    # ModelLake writes its own seal; it has no upstream digest by construction.
    (spec / "MODEL_LAKE_SPECIMEN_SEAL.json").write_text("{}")
    return spec


def test_whole_tree_verified_only_when_every_file_matched(tmp_path, monkeypatch):
    spec = _fixture_specimen(tmp_path)
    monkeypatch.setattr(sv, "SPECIMENS", spec.parent)
    res = sv.verify_specimen("fixture@abc")
    assert res["status"] == "WHOLE_TREE_VERIFIED"
    assert res["whole_tree_verified"] is True
    assert res["verified"] == res["n_files"]
    assert res["modellake_mutated"] is False


def test_negative_control_one_undigested_file_refuses_whole_tree(tmp_path, monkeypatch):
    """A file with no published digest must NOT be rounded into a pass."""
    spec = _fixture_specimen(tmp_path, undigested=True)
    monkeypatch.setattr(sv, "SPECIMENS", spec.parent)
    res = sv.verify_specimen("fixture@abc")
    assert res["whole_tree_verified"] is False
    assert res["status"] == "PARTIAL_NO_REMOTE_DIGEST"
    assert res["no_remote_digest"] == 1
    orphan = [r for r in res["files"] if r["file"] == "orphan.bin"][0]
    assert orphan["verdict"] == "NO_REMOTE_DIGEST"


def test_negative_control_a_mismatch_is_corrupt_not_verified(tmp_path, monkeypatch):
    spec = _fixture_specimen(tmp_path, corrupt=True)
    monkeypatch.setattr(sv, "SPECIMENS", spec.parent)
    res = sv.verify_specimen("fixture@abc")
    assert res["whole_tree_verified"] is False
    assert res["status"] == "CORRUPT_MISMATCH"
    assert res["mismatched"] == 1


def test_modellake_own_seal_files_are_not_counted_as_specimen_files(tmp_path, monkeypatch):
    spec = _fixture_specimen(tmp_path)
    monkeypatch.setattr(sv, "SPECIMENS", spec.parent)
    names = {p.name for p in sv.specimen_files("fixture@abc")}
    assert "MODEL_LAKE_SPECIMEN_SEAL.json" not in names


def test_absent_specimen_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "SPECIMENS", tmp_path)
    with pytest.raises(sv.SpecimenError):
        sv.verify_specimen("not-a-specimen@000")


def test_record_refuses_a_row_that_is_not_a_specimen(tmp_path, monkeypatch):
    """The gate reads this receipt. A fabricated row is a readiness claim.

    A fixture leaked exactly one such row into the live receipt once, which is
    the same shape of failure as leaving a negative control live in source.
    """
    _local_receipts(sv, tmp_path, monkeypatch)
    spec = tmp_path / "specimens" / "real@1"
    spec.mkdir(parents=True)
    monkeypatch.setattr(sv, "SPECIMENS", spec.parent)
    with pytest.raises(sv.SpecimenError):
        sv.record({"specimen": "invented@0", "status": "WHOLE_TREE_VERIFIED",
                   "n_files": 1, "verified": 1, "mismatched": 0,
                   "no_remote_digest": 0, "bytes_hashed": 1,
                   "whole_tree_verified": True})
    sv.record({"specimen": "real@1", "status": "WHOLE_TREE_VERIFIED", "n_files": 1,
               "verified": 1, "mismatched": 0, "no_remote_digest": 0,
               "bytes_hashed": 1, "whole_tree_verified": True})


def _local_receipts(module, tmp_path, monkeypatch):
    """Redirect both the read side and the seal-writing side at a tmp dir."""
    monkeypatch.setattr(module, "RECEIPTS", tmp_path)

    def _write(name, doc, recorded_by):
        out = tmp_path / name
        out.write_text(json.dumps(doc, indent=1, default=str))
        return out

    monkeypatch.setattr(module, "write_receipt", _write)


def test_record_persists_one_specimen_and_is_idempotent_by_name(tmp_path, monkeypatch):
    """Whole-tree verification of 733GB does not fit one window.

    A --build that must finish all seven before writing loses every completed
    specimen when the window closes -- and that is not hypothetical: record()
    shipped with an unimported RECEIPTS and threw AFTER seven minutes of real
    hashing, so the first specimen the autonomy loop verified was lost. One
    specimen is complete work and is persisted as such.
    """
    _local_receipts(sv, tmp_path, monkeypatch)
    for nm in ("a@1", "b@2"):
        (tmp_path / "specimens" / nm).mkdir(parents=True)
    monkeypatch.setattr(sv, "SPECIMENS", tmp_path / "specimens")
    first = {"specimen": "a@1", "status": "WHOLE_TREE_VERIFIED", "n_files": 2,
             "verified": 2, "mismatched": 0, "no_remote_digest": 0,
             "bytes_hashed": 10, "whole_tree_verified": True}
    sv.record(first)
    sv.record({**first, "specimen": "b@2"})
    doc = json.loads((tmp_path / sv.RECEIPT).read_text())
    assert {r["specimen"] for r in doc["results"]} == {"a@1", "b@2"}

    # Re-verifying a specimen replaces its row rather than appending a second one.
    sv.record({**first, "verified": 1, "n_files": 2, "whole_tree_verified": False,
               "status": "PARTIAL_NO_REMOTE_DIGEST", "no_remote_digest": 1})
    doc = json.loads((tmp_path / sv.RECEIPT).read_text())
    rows = [r for r in doc["results"] if r["specimen"] == "a@1"]
    assert len(rows) == 1 and rows[0]["status"] == "PARTIAL_NO_REMOTE_DIGEST"
    assert doc["counts"]["whole_tree_verified"] == 1


def test_record_survives_a_corrupt_prior_receipt(tmp_path, monkeypatch):
    """A truncated write must not make every later verification unrecordable."""
    _local_receipts(sv, tmp_path, monkeypatch)
    (tmp_path / "specimens" / "c@3").mkdir(parents=True)
    monkeypatch.setattr(sv, "SPECIMENS", tmp_path / "specimens")
    (tmp_path / sv.RECEIPT).write_text("{not json")
    sv.record({"specimen": "c@3", "status": "WHOLE_TREE_VERIFIED", "n_files": 1,
               "verified": 1, "mismatched": 0, "no_remote_digest": 0,
               "bytes_hashed": 5, "whole_tree_verified": True})
    doc = json.loads((tmp_path / sv.RECEIPT).read_text())
    assert [r["specimen"] for r in doc["results"]] == ["c@3"]


def test_real_falcon_result_is_recorded_and_is_a_recomputation():
    """The live result must show bytes actually hashed, not files merely counted."""
    p = RECEIPTS / "SPECIMEN_VERIFICATION.json"
    if not p.exists():
        return  # built by --build; the fixtures above carry the guarantees
    doc = json.loads(p.read_text())
    for row in doc.get("results") or []:
        if row.get("whole_tree_verified"):
            assert row["bytes_hashed"] > 0, "claimed verified without hashing anything"
            assert row["mismatched"] == 0
            assert row["no_remote_digest"] == 0


def test_a_non_lake_specimen_is_labelled_not_disguised():
    """ModelLake still owns the lake. A local directory must say so."""
    # Provenance is not one bucket. An authorized directory outside the lake and
    # ModelLake's own partial/ staging area are different facts, and collapsing
    # them would let a staged download read as an external specimen or the
    # reverse.
    for name in sv.EXTRA_SPECIMENS:
        owner = sv.specimen_owner(name)
        assert owner in {"local_directory", "modellake_partial"}, owner
        assert owner != "modellake", "a specimen outside specimens/ must not read as lake stock"
        assert any(name.endswith(m) for m in ("@local", "#partial")), (
            "a specimen outside the specimens listing must be visibly marked in its name"
        )
        if owner == "modellake_partial":
            assert "hawking-modellake" in str(sv.EXTRA_SPECIMENS[name])
        else:
            assert "hawking-modellake" not in str(sv.EXTRA_SPECIMENS[name])
    lake_names = [n for n in sv.list_specimens() if n not in sv.EXTRA_SPECIMENS]
    for name in lake_names:
        assert sv.specimen_owner(name) == "modellake"


def test_the_local_mirrors_own_checksum_file_is_not_a_digest_source():
    """crc32.txt is written by the mirror and covers 8 small files, never the weights.

    Counting it would verify the tokenizer and call the 52GB of weights checked.
    """
    assert "crc32.txt" in sv.LOCAL_OWN_FILES
    name = "qwen3.8-27b-abliterated-bf16@local"
    if not sv.specimen_dir(name).is_dir():
        return
    assert "crc32.txt" not in {p.name for p in sv.specimen_files(name)}


def test_the_local_specimen_verifies_by_exactly_the_same_rule():
    """Same digests, same recomputation, same refusals -- only the owner differs."""
    name = "qwen3.8-27b-abliterated-bf16@local"
    if not sv.specimen_dir(name).is_dir():
        return
    files = sv.specimen_files(name)
    assert files, "the local specimen has no files"
    digested = [f for f in files if sv._read_metadata(sv.specimen_dir(name), f.name)]
    assert digested, "no HuggingFace .metadata sidecars; it cannot be verified this way"
    meta = sv._read_metadata(sv.specimen_dir(name), digested[0].name)
    assert meta["digest_kind"] in {"sha256", "git_blob_sha1"}


def test_build_is_bounded_and_names_what_it_did_not_reach(tmp_path, monkeypatch):
    """A whole-lake pass is not a bounded unit.

    ModelLake went from 7 specimens to 43 during one autonomy trial as the
    download workers promoted a batch. build() iterating all of them at 900s
    each became an eleven-hour call that ran 37 minutes past the end of a 1-hour
    trial with no way to stop it, and the trial was discarded.
    """
    _local_receipts(sv, tmp_path, monkeypatch)
    specs = tmp_path / "specimens"
    for i in range(4):
        d = specs / f"s{i}@x"
        (d / ".cache" / "huggingface" / "download").mkdir(parents=True)
        (d / "w.bin").write_bytes(b"x" * 64)
    monkeypatch.setattr(sv, "SPECIMENS", specs)

    out = sv.build(max_total_seconds=0.0)  # budget exhausted before the first
    doc = json.loads(out.read_text())
    skipped = set(doc["not_reached_this_pass"])
    assert skipped, "a bounded pass must name what it skipped"
    # list_specimens() also yields the registered non-lake specimens, so assert
    # the fixtures are covered rather than pinning a count to this environment.
    assert {f"s{i}@x" for i in range(4)} <= skipped
    assert doc["counts"]["specimens_examined"] == 0
    assert "budget_rule" in doc


def test_a_bounded_pass_carries_prior_verdicts_forward(tmp_path, monkeypatch):
    """Dropping an earlier verdict would make a bounded run look like a regression."""
    _local_receipts(sv, tmp_path, monkeypatch)
    specs = tmp_path / "specimens"
    (specs / "kept@1").mkdir(parents=True)
    monkeypatch.setattr(sv, "SPECIMENS", specs)
    sv.record({"specimen": "kept@1", "status": "WHOLE_TREE_VERIFIED", "n_files": 1,
               "verified": 1, "mismatched": 0, "no_remote_digest": 0,
               "bytes_hashed": 10, "whole_tree_verified": True})

    doc = json.loads(sv.build(max_total_seconds=0.0).read_text())
    kept = [r for r in doc["results"] if r["specimen"] == "kept@1"]
    assert len(kept) == 1, "a prior verdict was dropped by a bounded pass"
    assert kept[0]["whole_tree_verified"] is True
    assert "kept@1" in doc["not_reached_this_pass"]

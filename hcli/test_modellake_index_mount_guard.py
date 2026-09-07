"""build() rewrote the catalog as an empty lake when the volume was detached.

Measured against a copy of the live index: n_specimens 56 -> 0,
tier2_used_gb 4712.69 -> 0, over_budget True -> FALSE, and all 55 by-slug
records unlinked, at exit code 0 with no exception. The real lake is 852 GB over
budget, so the artifact destroyed was the only record of the overage. And with
no explicit --index-dir, resolve_index_dir cannot mkdir under /Volumes and falls
back to receipts/future/modellake-index -- the very directory holding it.
"""
import json, os, shutil, struct, sys, pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.odyssey.modellake_index import build, LakeNotMounted


def _specimen(d):
    d.mkdir(parents=True, exist_ok=True)
    hdr = {"model.embed_tokens.weight": {"dtype": "F32", "shape": [4, 4],
                                         "data_offsets": [0, 64]}}
    blob = json.dumps(hdr).encode()
    (d / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 64)
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))


@pytest.fixture
def lake_and_index(tmp_path):
    lake = tmp_path / "lake"
    _specimen(lake / "specimens" / "acme--demo")
    idx = tmp_path / "idx"
    out = build(lake=str(lake), index_dir=str(idx), manifest_dir=str(tmp_path / "nomf"))
    assert out["n_specimens"] == 1, out
    assert os.listdir(idx / "by-slug") == ["acme--demo.json"]
    return lake, idx, tmp_path


def test_absent_volume_refuses_instead_of_reporting_an_empty_lake(lake_and_index):
    lake, idx, tmp = lake_and_index
    with pytest.raises(LakeNotMounted):
        build(lake=str(tmp / "never_mounted"), index_dir=str(idx),
              manifest_dir=str(tmp / "nomf"))


def test_mounted_but_empty_volume_also_refuses(lake_and_index):
    """A remounted-empty or stale mountpoint passes the existence check."""
    lake, idx, tmp = lake_and_index
    empty = tmp / "empty"
    (empty / "specimens").mkdir(parents=True)
    (empty / "partial").mkdir(parents=True)
    with pytest.raises(LakeNotMounted):
        build(lake=str(empty), index_dir=str(idx), manifest_dir=str(tmp / "nomf"))


def test_the_index_survives_a_refused_rebuild(lake_and_index):
    lake, idx, tmp = lake_and_index
    before = json.loads((idx / "catalog.json").read_text())
    for bad in (tmp / "never_mounted",):
        with pytest.raises(LakeNotMounted):
            build(lake=str(bad), index_dir=str(idx), manifest_dir=str(tmp / "nomf"))
    after = json.loads((idx / "catalog.json").read_text())
    assert after["n_specimens"] == before["n_specimens"] == 1
    assert os.listdir(idx / "by-slug") == ["acme--demo.json"], "records were unlinked"


def test_over_budget_cannot_be_cleared_by_a_detached_drive(lake_and_index):
    """The destroyed catalog claimed over_budget=False. That must be unreachable."""
    lake, idx, tmp = lake_and_index
    cat = json.loads((idx / "catalog.json").read_text())
    cat["over_budget"] = True
    cat["tier2_used_bytes"] = 4_000_000_000_000
    (idx / "catalog.json").write_text(json.dumps(cat))
    with pytest.raises(LakeNotMounted):
        build(lake=str(tmp / "never_mounted"), index_dir=str(idx),
              manifest_dir=str(tmp / "nomf"))
    assert json.loads((idx / "catalog.json").read_text())["over_budget"] is True


def test_a_real_incremental_rebuild_still_works(lake_and_index):
    """The guards must not break the ordinary path."""
    lake, idx, tmp = lake_and_index
    out = build(lake=str(lake), index_dir=str(idx), manifest_dir=str(tmp / "nomf"))
    assert out["n_specimens"] == 1
    assert out["dropped_slugs"] == []


def test_a_genuine_single_deletion_is_still_allowed(lake_and_index):
    """Guarding mass drops must not freeze the index against real removals."""
    lake, idx, tmp = lake_and_index
    _specimen(lake / "specimens" / "acme--second")
    assert build(lake=str(lake), index_dir=str(idx),
                 manifest_dir=str(tmp / "nomf"))["n_specimens"] == 2
    shutil.rmtree(lake / "specimens" / "acme--second")
    out = build(lake=str(lake), index_dir=str(idx), manifest_dir=str(tmp / "nomf"))
    assert out["dropped_slugs"] == ["acme--second"]
    assert out["n_specimens"] == 1


def test_a_fresh_index_on_a_detached_drive_refuses_too(tmp_path):
    """The case the mass-drop guard cannot see: nothing catalogued yet.

    With no prior catalog there is nothing to drop, so only the mount check
    stands between a detached volume and a brand-new, well-formed catalog
    asserting the lake holds zero specimens and is under budget.
    """
    with pytest.raises(LakeNotMounted):
        build(lake=str(tmp_path / "never_mounted"),
              index_dir=str(tmp_path / "fresh_idx"),
              manifest_dir=str(tmp_path / "nomf"))
    assert not (tmp_path / "fresh_idx" / "catalog.json").exists(), \
        "an empty catalog was written for a lake that was never read"

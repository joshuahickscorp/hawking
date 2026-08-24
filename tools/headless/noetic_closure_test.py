#!/usr/bin/env python3
"""Observed-closure harness: pytest surface.

Synthetic fixtures prove the method (observe opens, both mismatch directions,
every hashed member is load-bearing, identity is content not st_dev). The live
test writes receipts/headless/NOETIC_CLOSURE.json against the sealed artifact
on a copy; it never mutates ~/models.

    python3 -m pytest tools/headless/noetic_closure_test.py -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("noetic_closure", HERE / "noetic_closure.py")
nc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(nc)

MODELS = Path.home() / "models"
LIVE_ARTIFACT = Path(os.environ.get("NOETIC_ARTIFACT", str(nc.DEFAULT_ARTIFACT)))
LIVE_TOKENIZER = Path(os.environ.get("NOETIC_TOKENIZER", str(nc.DEFAULT_TOKENIZER)))

_LIVE_DOC: dict | None = None


def plant_artifact(root: Path) -> tuple[Path, Path]:
    artifact = root / "artifact"
    (artifact / "tensors").mkdir(parents=True)
    tensors = {
        "alpha.bin": b"ALPHA-WEIGHTS-0001",
        "beta.bin": b"BETA-WEIGHTS-0002",
        "gamma.bin": b"GAMMA-WEIGHTS-0003",
    }
    rows = []
    for name, payload in tensors.items():
        (artifact / "tensors" / name).write_bytes(payload)
        rows.append({"name": name.split(".")[0], "artifact": name, "kind": "raw"})
    manifest = {
        "schema": "test.qwen38_uniform_q4.v1",
        "tensors": rows,
    }
    (artifact / "manifest.json").write_bytes(
        json.dumps(manifest, separators=(",", ":")).encode()
    )
    (artifact / "helper_not_used.txt").write_bytes(b"I AM CEREMONY")
    tokenizer = root / "tokenizer.json"
    tokenizer.write_bytes(b'{"model":{"vocab":{"hi":1}}}')
    return artifact, tokenizer


def hashed_from_run(artifact: Path, tokenizer: Path, consume: str = "hash"):
    io_run = nc.execute_load_io_watched(artifact, tokenizer, consume=consume)
    members = nc.hashed_members_from_observation(io_run, artifact, tokenizer)
    return io_run, members


def test_hashed_set_comes_from_observed_opens_not_from_a_loader_list(tmp_path: Path):
    artifact, tokenizer = plant_artifact(tmp_path)
    io_run, members = hashed_from_run(artifact, tokenizer)
    assert io_run["ok"] is True
    idents = {m["ident"] for m in members}
    assert "artifact/manifest.json" in idents
    assert "tokenizer.json" in idents
    assert "artifact/tensors/alpha.bin" in idents
    assert "artifact/tensors/beta.bin" in idents
    assert "artifact/tensors/gamma.bin" in idents
    assert "artifact/helper_not_used.txt" not in idents
    observed = set(io_run["watcher_read_paths"])
    for m in members:
        assert m["path"] in observed
        assert m["sha256"]
        assert "st_dev" not in m
        assert "st_ino" not in m
        assert "mtime" not in m
        assert "mtime_ns" not in m


def test_planting_a_manifest_row_is_observed_because_execution_opens_it(tmp_path: Path):
    artifact, tokenizer = plant_artifact(tmp_path)
    planted = artifact / "tensors" / "planted.bin"
    planted.write_bytes(b"PLANTED-BY-THE-HARNESS")
    man = json.loads((artifact / "manifest.json").read_text())
    man["tensors"].append(
        {"name": "planted", "artifact": "planted.bin", "kind": "raw"}
    )
    (artifact / "manifest.json").write_text(json.dumps(man))
    io_run, members = hashed_from_run(artifact, tokenizer)
    idents = {m["ident"] for m in members}
    assert "artifact/tensors/planted.bin" in idents
    by_ident = {m["ident"]: m for m in members}
    assert by_ident["artifact/tensors/planted.bin"]["sha256"] == nc.sha256_bytes(
        b"PLANTED-BY-THE-HARNESS"
    )


def test_read_but_not_hashed_is_reported(tmp_path: Path):
    artifact, tokenizer = plant_artifact(tmp_path)
    io_run, members = hashed_from_run(artifact, tokenizer)
    hashed_paths = [m["path"] for m in members]
    sneaky = (artifact / "route_table.bin").resolve()
    extra_observed = hashed_paths + [str(sneaky)]
    cmp_ = nc.compare_sets(extra_observed, hashed_paths)
    assert cmp_["n_read_but_not_hashed"] == 1
    assert str(sneaky) in cmp_["read_but_not_hashed"]
    assert cmp_["n_hashed_but_not_read"] == 0


def test_hashed_but_not_read_is_reported(tmp_path: Path):
    artifact, tokenizer = plant_artifact(tmp_path)
    io_run, members = hashed_from_run(artifact, tokenizer)
    hashed_paths = [m["path"] for m in members]
    pad = str((artifact / "shared_basis.bin").resolve())
    cmp_ = nc.compare_sets(hashed_paths, hashed_paths + [pad])
    assert cmp_["n_hashed_but_not_read"] == 1
    assert pad in cmp_["hashed_but_not_read"]
    assert cmp_["n_read_but_not_hashed"] == 0


def test_each_hashed_member_is_load_bearing(tmp_path: Path):
    artifact, tokenizer = plant_artifact(tmp_path)
    _, members = hashed_from_run(artifact, tokenizer)
    assert len(members) == 5
    removal = nc.removal_test_each(artifact, tokenizer, members)
    assert removal["copy_only"] is True
    assert removal["original_untouched"] is True
    assert removal["n_members"] == 5
    assert removal["n_broke"] == 5
    assert removal["n_ceremony"] == 0
    assert removal["all_load_bearing"] is True
    assert {t["ident"] for t in removal["trials"]} == {m["ident"] for m in members}
    assert all(t["broke"] for t in removal["trials"])
    for m in members:
        assert Path(m["path"]).is_file()


def test_unhashed_sibling_is_ceremony(tmp_path: Path):
    artifact, tokenizer = plant_artifact(tmp_path)
    helper = artifact / "helper_not_used.txt"
    assert helper.is_file()
    io_run = nc.execute_load_io_watched(artifact, tokenizer, consume="open")
    assert io_run["ok"] is True
    helper.unlink()
    io_run2 = nc.execute_load_io_watched(artifact, tokenizer, consume="open")
    assert io_run2["ok"] is True, io_run2.get("error")


def test_identity_is_content_sha256_not_st_dev(tmp_path: Path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    payload = b"same-bytes-different-inode"
    a.write_bytes(payload)
    b.write_bytes(payload)
    sa, na = nc.sha256_file(a)
    sb, nb = nc.sha256_file(b)
    assert sa == sb == nc.sha256_bytes(payload)
    assert na == nb == len(payload)
    sta, stb = a.stat(), b.stat()
    assert sta.st_ino != stb.st_ino
    finding = nc.identity_finding(REPO)
    assert finding["closure_identity"] == "sha256_of_file_bytes"
    assert "st_dev" in finding["not_used"]
    assert finding["admission_warm_receipt"]["match_key_is_content"] is False


def test_dyld_interpose_observes_a_native_open(tmp_path: Path):
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"native-open-target")
    traced = nc.trace_native_helper_open(secret, tmp_path)
    assert traced.get("ok") is True, traced.get("error")
    reads = [Path(p).resolve() for p in traced.get("read_paths") or []]
    assert secret.resolve() in reads, traced.get("read_paths")


def test_never_unlinks_under_models(tmp_path: Path):
    if not MODELS.is_dir():
        pytest.skip("~/models not present")
    victim = MODELS / "qwen38-gravity-uniform-q4-v1" / "manifest.json"
    if not victim.is_file():
        pytest.skip("sealed artifact not present")
    with pytest.raises(RuntimeError, match="refusing to mutate"):
        nc.assert_not_under_models(victim)


def test_shadow_unlink_does_not_touch_original(tmp_path: Path):
    artifact, tokenizer = plant_artifact(tmp_path)
    original = artifact / "tensors" / "alpha.bin"
    before = original.read_bytes()
    _, members = hashed_from_run(artifact, tokenizer)
    nc.removal_test_each(artifact, tokenizer, members)
    assert original.read_bytes() == before
    assert original.stat().st_nlink >= 1


def live_available() -> bool:
    return LIVE_ARTIFACT.is_dir() and (LIVE_ARTIFACT / "manifest.json").is_file() and LIVE_TOKENIZER.is_file()


def live_doc() -> dict:
    global _LIVE_DOC
    if _LIVE_DOC is None:
        _LIVE_DOC = nc.run(
            repo=REPO,
            artifact=LIVE_ARTIFACT,
            tokenizer=LIVE_TOKENIZER,
            write_receipt=True,
            do_removal=True,
        )
    return _LIVE_DOC


@pytest.mark.skipif(not live_available(), reason="sealed uniform-q4-v1 artifact not on disk")
def test_live_harness_writes_receipt_listing_every_model_specific_file():
    doc = live_doc()
    receipt = REPO / "receipts" / "headless" / "NOETIC_CLOSURE.json"
    assert receipt.is_file()
    on_disk = json.loads(receipt.read_text())
    assert on_disk["schema"] == nc.SCHEMA
    members = on_disk["hashed_members"]
    assert len(members) >= 3
    idents = {m["ident"] for m in members}
    assert "artifact/manifest.json" in idents
    assert "tokenizer.json" in idents
    tensor_members = [m for m in members if m["role"] == "tensor"]
    assert len(tensor_members) == on_disk["observation"]["io_executor"]["tensor_count"]
    for m in members:
        assert m["sha256"]
        assert m["bytes"] >= 0
        assert m["model_specific"] is True
        assert "st_dev" not in m
    tok = next(m for m in members if m["ident"] == "tokenizer.json")
    assert tok.get("outside_artifact") is True


@pytest.mark.skipif(not live_available(), reason="sealed uniform-q4-v1 artifact not on disk")
def test_live_removal_shows_every_hashed_member_is_load_bearing():
    doc = live_doc()
    rem = doc["removal"]
    assert rem["copy_only"] is True
    assert rem["original_untouched"] is True
    assert rem["n_members"] == doc["n_hashed_members"]
    assert rem["n_broke"] == rem["n_members"]
    assert rem["all_load_bearing"] is True
    assert rem["n_ceremony"] == 0
    trial_idents = [t["ident"] for t in rem["trials"]]
    member_idents = [m["ident"] for m in doc["hashed_members"]]
    assert trial_idents == member_idents
    assert all(t["broke"] for t in rem["trials"])
    assert (LIVE_ARTIFACT / "manifest.json").is_file()
    assert LIVE_TOKENIZER.is_file()


@pytest.mark.skipif(not live_available(), reason="sealed uniform-q4-v1 artifact not on disk")
def test_live_reports_both_mismatch_directions():
    doc = live_doc()
    cmp_ = doc["compare"]
    assert "io_executor" in cmp_
    assert "live_decode" in cmp_
    io = cmp_["io_executor"]
    live = cmp_["live_decode"]
    assert "read_but_not_hashed" in io and "hashed_but_not_read" in io
    assert "read_but_not_hashed" in live and "hashed_but_not_read" in live
    assert io["n_read_but_not_hashed"] == 0
    live_obs = doc["observation"]["live_decode"]
    if live_obs.get("metal_refused"):
        assert live["n_hashed_but_not_read"] >= 1
    if cmp_["gate"] == "FAIL":
        named = io["read_but_not_hashed"] + live["read_but_not_hashed"]
        assert named, "gate FAIL must name the unread-hash files"


@pytest.mark.skipif(not live_available(), reason="sealed uniform-q4-v1 artifact not on disk")
def test_live_does_not_touch_models_tree():
    doc = live_doc()
    assert (LIVE_ARTIFACT / "manifest.json").is_file()
    assert LIVE_TOKENIZER.is_file()
    for m in doc["hashed_members"]:
        p = Path(m["path"])
        assert p.is_file()
        if str(p).startswith(str(MODELS)):
            st = p.stat()
            assert stat.S_ISREG(st.st_mode)

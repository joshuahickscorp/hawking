"""Specimen curriculum: unpublished or unverified is not ready.

Acceptance the rest of the module is worthless without:

* a role whose specimen is unpublished is not ready;
* a role whose specimen is unverified (size-only / partial) is not ready;
* n_roles is a constant of five — the ratio is not improved by dropping roles;
* every role carries either a verified specimen identity or a S022 §55 block.
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.future import specimen_curriculum as sc
from tools.future import odyssey_launch as ol
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


def test_authority_is_this_module_not_a_fork():
    assert ol.propose_specimen_curriculum is sc.propose_specimen_curriculum
    assert ol._ready is sc._ready
    assert ol.CURRICULUM_ROLES is sc.CURRICULUM_ROLES
    assert len(sc.CURRICULUM_ROLES) == 5


def test_unpublished_specimen_is_not_ready():
    """Acceptance: the module refuses to call a role ready when unpublished."""
    identity = {
        "repo": "example/unpublished",
        "revision": "abc123",
        "resolved_sha": "abc123",
        "in_specimens_listing": True,
        "published_as_verified": False,
        "whole_tree_verified": False,
        "n_sha256_verified": 10,
        "n_files": 10,
        "specimen_path": "/Volumes/corpdrive/hawking-modellake/specimens/example--unpublished@abc123",
    }
    ready, why = sc._ready(identity, require_lake_verified=True)
    assert ready is False
    assert "not published as verified" in why
    gap = sc.classify_gap(identity)
    assert gap == sc.GAP_UNPUBLISHED
    block = sc.blocked_record({"role": "mid_size_dense_compiler", "ready_reason": why}, identity)
    for field in sc.S022_BLOCK_FIELDS:
        assert block.get(field), f"missing S022 §55 field {field}"
    assert block["gap_class"] == sc.GAP_UNPUBLISHED


def test_unverified_partial_manifest_is_not_ready():
    """Acceptance: size-only / partial hashing is not a sealed specimen."""
    identity = {
        "repo": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a0",
        "resolved_sha": "c1899de289a04d12100db370d81485cdf75e47ca",
        "in_specimens_listing": True,
        "whole_tree_verified": False,
        "n_sha256_verified": 2,
        "n_files": 10,
        "n_size_only_verified": 8,
        "manifest_path": "/Volumes/corpdrive/hawking-modellake/manifests/Qwen--Qwen3-0.6B@c1899de289a0.json",
        "specimen_path": "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3-0.6B@c1899de289a0",
    }
    ready, why = sc._ready(identity, require_lake_verified=True)
    assert ready is False
    assert "partial" in why
    assert "n_sha256_verified=2" in why
    assert "n_files=10" in why
    assert sc.classify_gap(identity) == sc.GAP_UNVERIFIED
    block = sc.blocked_record({"role": "very_small_dense_procedural_speed", "ready_reason": why}, identity)
    for field in sc.S022_BLOCK_FIELDS:
        assert block.get(field)
    assert "specimen_verify.py" in block["reevaluation_trigger"]


def test_n_roles_is_not_lowered_to_improve_the_ratio(monkeypatch):
    monkeypatch.setattr(sc, "_independently_verified", lambda: {})
    monkeypatch.setattr(sc, "_specimen_dirs_on_disk", lambda: set())
    monkeypatch.setattr(sc, "_manifests_on_disk", lambda: {})
    monkeypatch.setattr(sc, "_odyssey_i_patients", lambda: [])
    cur = sc.propose_specimen_curriculum(census_doc={})
    assert cur["n_roles"] == 5
    assert len(cur["roles"]) == 5
    assert [r["role"] for r in cur["roles"]] == [c[0] for c in sc.CURRICULUM_ROLES]
    assert cur["n_ready"] == 0
    assert cur["ready"] is False
    for role in cur["roles"]:
        assert role["ready"] is False
        assert role.get("verified_specimen") is None
        blocked = role.get("blocked") or {}
        for field in sc.S022_BLOCK_FIELDS:
            assert blocked.get(field), f"{role['role']} missing {field}"


def test_empty_lake_does_not_invent_a_ready_specimen(monkeypatch):
    monkeypatch.setattr(sc, "_independently_verified", lambda: {})
    monkeypatch.setattr(sc, "_specimen_dirs_on_disk", lambda: set())
    monkeypatch.setattr(sc, "_manifests_on_disk", lambda: {})
    monkeypatch.setattr(sc, "_odyssey_i_patients", lambda: [])
    cur = sc.propose_specimen_curriculum(census_doc={})
    for role in cur["roles"]:
        vs = role.get("verified_specimen")
        assert not vs or vs.get("resolved_sha") in (None, "")
        assert role["ready"] is False


def test_every_live_role_has_verified_identity_or_s022_block():
    cur = sc.propose_specimen_curriculum()
    assert cur["n_roles"] == 5
    assert cur["n_ready"] == sum(1 for r in cur["roles"] if r.get("ready"))
    assert cur["ready"] is (cur["n_ready"] == 5)
    for role in cur["roles"]:
        if role.get("ready"):
            vs = role.get("verified_specimen") or {}
            assert vs.get("repo")
            assert vs.get("resolved_sha")
            assert vs.get("n_files")
            assert vs.get("whole_tree_verified") is True
            assert role.get("blocked") is None
            why = role["ready_reason"]
            assert any(k in why for k in ("whole-tree", "RECURRENT_PATIENT", "tree digest is sealed"))
        else:
            assert role.get("verified_specimen") is None
            blocked = role.get("blocked") or {}
            for field in sc.S022_BLOCK_FIELDS:
                assert blocked.get(field), f"{role['role']} unready without {field}: {blocked}"
            assert blocked["gap_class"] in {
                sc.GAP_ABSENT,
                sc.GAP_UNPUBLISHED,
                sc.GAP_UNVERIFIED,
                sc.GAP_NOT_LISTED,
                sc.GAP_NO_CANDIDATE,
            }


def test_build_writes_receipt_without_hardware_claims():
    out = sc.build()
    path = Path(out["path"])
    assert path == RECEIPTS / sc.RECEIPT
    doc = json.loads(path.read_text())
    assert doc["schema"] == sc.SCHEMA
    assert doc["n_roles"] == 5
    assert doc["gpu_authority"] is False
    assert doc["measurement_class"] == "STATIC_ONLY"
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["seal_sha256"]
    _assert_no_hardware_claims(doc)
    assert doc["s022_section_55"]["fields"] == list(sc.S022_BLOCK_FIELDS)
    assert len(doc["roles"]) == 5


def test_whole_tree_verified_still_ready():
    identity = {
        "repo": "tiiuae/Falcon-H1-7B-Instruct",
        "revision": "41e72f27effbab80cd45b6e884688452253a3686",
        "resolved_sha": "41e72f27effbab80cd45b6e884688452253a3686",
        "in_specimens_listing": True,
        "published_as_verified": True,
        "whole_tree_verified": True,
        "bytes_hashed": 15182220635,
        "n_files": 13,
        "n_sha256_verified": 13,
        "manifest_path": "/Volumes/corpdrive/hawking-modellake/manifests/tiiuae--Falcon-H1-7B-Instruct@41e72f27effb.json",
    }
    ready, why = sc._ready(identity, require_lake_verified=True)
    assert ready is True
    assert "whole-tree" in why
    rec = sc.verified_specimen_record(
        {"role": "small_dense_alternate_architecture_transfer", "repo": identity["repo"],
         "revision": identity["revision"], "ready": True, "modellake": identity},
        identity,
    )
    assert rec["resolved_sha"] == identity["resolved_sha"]
    assert rec["manifest_path"] == identity["manifest_path"]
    assert rec["n_files"] == 13
    assert rec["n_sha256_verified"] == 13

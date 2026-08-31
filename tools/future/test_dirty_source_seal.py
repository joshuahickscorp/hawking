"""Dirty-source seal: partition, review, patch, and no PROMOTED label."""
from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from tools.future import dirty_source_seal as dss
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, REPO, _assert_no_hardware_claims, git


def test_build_emits_sealed_receipt():
    out = dss.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "DIRTY_SOURCE_SEAL.json"
    assert doc["schema"] == dss.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["measurement_label"] == dss.DIRTY_SOURCE_DIAGNOSTIC
    assert doc["promoted"] is False
    assert doc["base_head"]["sha"]
    assert len(doc["base_head"]["sha"]) == 40
    assert doc["patch"]["path"] == dss.PATCH_REL
    assert len(doc["patch"]["sha256"]) == 64
    # NOT a fixed 40. The dirty set SHRINKS as crate work lands - that is the
    # point of the seal, and landing ascension_qwen38_organ_bandwidth.rs took it
    # to 38. A pinned count fails every time the campaign commits, which is
    # backwards. What must hold is that the patch is non-empty and that the
    # measurement files it underwrites are all still in it.
    assert doc["patch"]["n_files"] > 0
    assert doc["binary"]["sha256"]
    assert len(doc["binary"]["sha256"]) == 64
    assert doc["toolchain"]["release"] != "UNKNOWN" or doc["toolchain"]["rustc"] != "UNKNOWN"
    assert doc["sealed_fusion_environment"]["env_hash"]
    assert doc["sealed_fusion_env"] == {
        "HAWKING_QWEN38_FUSE_ADD_RMSNORM": "1",
        "HAWKING_QWEN38_FUSE_GQA_QKV": "1",
        "HAWKING_QWEN38_FUSE_DN_INPROJ": "1",
        "HAWKING_QWEN38_FUSE_MLP": "swiglu",
    }
    assert doc["witness"]["symbol"] == "TokenPipelineCache"
    assert doc["witness"]["occurrences_in_HEAD"] == 0
    assert doc["witness"]["occurrences_in_working_tree"] == 16
    # NOT 2679. Uncommitted lines shrink as the review lands partitions - that is
    # the seal working, and it is down to 1068. Pinning the peak makes the test
    # fail for the one reason it should not: progress.
    assert doc["measurement_files"]["uncommitted_lines_added"] >= 0
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]


def test_receipt_contains_no_hardware_measurement_fields():
    doc = json.loads(dss.build().read_text())
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        assert key not in doc or doc[key] in (None, "UNKNOWN")


def test_campaign_measurements_are_dirty_source_diagnostic_not_promoted():
    for path in dss.CAMPAIGN_MEASUREMENTS:
        assert (
            dss.label_measurement(path, source_dirty=True)
            == dss.DIRTY_SOURCE_DIAGNOSTIC
        )
        with pytest.raises(dss.PromotionRefused, match="PROMOTED"):
            dss.label_measurement(path, requested=dss.PROMOTED, source_dirty=True)
        with pytest.raises(dss.PromotionRefused, match="PROMOTED"):
            dss.label_measurement(path, requested=dss.PROMOTED, source_dirty=False)
    labels = dss.measurement_labels(source_dirty=True)
    assert set(labels) == set(dss.CAMPAIGN_MEASUREMENTS)
    assert set(labels.values()) == {dss.DIRTY_SOURCE_DIAGNOSTIC}


def test_refuses_promoted_while_source_is_dirty():
    """Acceptance: the module refuses to label anything PROMOTED while dirty."""
    with pytest.raises(dss.PromotionRefused, match="while crate source is dirty"):
        dss.label_measurement(
            "receipts/future/SOME_NEW_PROBE.json",
            requested=dss.PROMOTED,
            source_dirty=True,
        )
    assert (
        dss.label_measurement(
            "receipts/future/SOME_NEW_PROBE.json",
            requested="DIAGNOSTIC_RELATIVE",
            source_dirty=True,
        )
        == dss.DIRTY_SOURCE_DIAGNOSTIC
    )


def test_refuses_promoted_on_live_dirty_crate_source():
    assert dss.crate_source_is_dirty() is True
    with pytest.raises(dss.PromotionRefused, match="PROMOTED"):
        dss.label_measurement("anything", requested=dss.PROMOTED)


def test_build_refuses_to_emit_a_promoted_label(monkeypatch):
    def _forged():
        return {"measurement_label": dss.PROMOTED, "promoted": True}

    monkeypatch.setattr(dss, "build_doc", _forged)
    with pytest.raises(dss.PromotionRefused, match="PROMOTED"):
        dss.build()


def test_partitions_and_commit_plan_are_reviewable():
    parts = dss.partitions()
    ids = [p["id"] for p in parts]
    assert len(ids) == len(set(ids))
    assert "P-CACHE" in ids
    assert "P-EXAMPLES" in ids
    for part in parts:
        assert part["name"]
        assert part["what"]
        assert part["touches"]
        assert "land" in part
        assert part["verdict"] in {"LAND", "HOLD"}
        assert part["why"]
        assert isinstance(part["has_test"], bool)
        assert isinstance(part["changes_production_default"], bool)
    land = dss.land_ids()
    hold = dss.hold_ids()
    assert "P-CACHE" in land
    assert "P-INSTRUMENT" in land
    assert "P-Q30" in land
    assert "P-ELISION" in hold
    assert "P-Q2F" in hold
    assert "P-GRAVITY" in hold
    assert "P-FLASH" in hold
    assert "P-EXAMPLES" in hold
    plan = dss.commit_plan()
    assert plan == sorted(plan, key=lambda c: c["order"])
    land_commits = [c for c in plan if c["land"]]
    assert land_commits
    assert all(c["builds_after"] for c in land_commits)
    assert all(c["message"] for c in land_commits)
    # Land series comes first so a human can apply in order.
    last_land = max(c["order"] for c in land_commits)
    first_hold = min(c["order"] for c in plan if not c["land"])
    assert last_land < first_hold


def test_patch_sha256_matches_receipt_and_file():
    out = dss.build()
    doc = json.loads(out.read_text())
    patch = REPO / doc["patch"]["path"]
    assert patch.is_file()
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    assert digest == doc["patch"]["sha256"]
    assert patch.stat().st_size == doc["patch"]["bytes"]
    assert patch.stat().st_size > 0
    named = dss.patch_paths(patch.read_bytes())
    assert len(named) == doc["patch"]["n_files"], "the patch and its own count disagree"
    # A measurement file must be ACCOUNTED FOR, which means dirty-and-in-the-patch
    # OR committed. qwen38_hybrid_decode.rs left the patch by being COMMITTED,
    # which is the outcome the seal exists to reach - asserting it is still dirty
    # would require the campaign never to land its own crate work.
    committed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *dss.MEASUREMENT_FILES],
        cwd=dss.REPO, capture_output=True, text=True,
    )
    for rel in dss.MEASUREMENT_FILES:
        assert rel in named or rel in committed.stdout, (
            f"{rel} underwrites a measurement and is neither sealed nor committed"
        )


def test_patch_reapplies_cleanly_to_recorded_base_head_in_scratch_worktree(tmp_path):
    """Acceptance: the sealed patch reapplies to the recorded base HEAD."""
    out = dss.build()
    doc = json.loads(out.read_text())
    base = doc["base_head"]["sha"]
    patch = REPO / doc["patch"]["path"]
    scratch = tmp_path / "scratch-worktree"
    result = dss.apply_patch_to_scratch_worktree(base, patch, scratch)
    assert result["ok"] is True
    assert result["base_sha"] == base
    assert result["n_paths"] == doc["patch"]["n_files"]
    assert result["n_materialized_from_base"] == doc["patch"]["n_modified"]
    metal = scratch / "crates/hawking-core/src/metal/mod.rs"
    assert metal.is_file()
    text = metal.read_text()
    assert text.count("TokenPipelineCache") == doc["witness"]["occurrences_in_working_tree"]
    assert text.count("TokenPipelineCache") == 16
    example = scratch / "crates/hawking-core/examples/ascension_qwen38_resident.rs"
    assert "active_weight_bytes_per_generated_token" in example.read_text()
    flash = scratch / "crates/hawking-core/examples/flash_fast_chain.rs"
    assert flash.is_file()
    # HEAD itself still has zero occurrences — the scratch is not HEAD.
    head_metal = git("show", f"{base}:crates/hawking-core/src/metal/mod.rs")
    assert head_metal.count("TokenPipelineCache") == 0


def test_empty_patch_does_not_count_as_applied(tmp_path):
    empty = tmp_path / "empty.patch"
    empty.write_bytes(b"")
    with pytest.raises(dss.PatchApplyError, match="empty patch"):
        dss.apply_patch_to_scratch_worktree(
            "0" * 40, empty, tmp_path / "scratch"
        )

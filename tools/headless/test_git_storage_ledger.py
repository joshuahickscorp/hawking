"""GIT_STORAGE_LEDGER: measured families, valid classes, unexecuted rewrite plan."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from git_storage_ledger import (  # noqa: E402
    CAS_LAYOUT,
    CLASSES,
    EXISTING_ARTIFACT_ROOT,
    FAMILY_META,
    GITIGNORE_MUST_HOLD,
    POLICY,
    RECEIPT,
    SCHEMA,
    build,
    family_id,
    git_ok,
    write_receipt,
)

RECEIPT_DOC = None


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        reuse = os.environ.get("GIT_STORAGE_REMEASURE", "0") != "1"
        if reuse and RECEIPT.is_file():
            doc = json.loads(RECEIPT.read_text())
            if doc.get("schema") == SCHEMA and doc.get("families") and doc.get("history_compaction_plan"):
                RECEIPT_DOC = doc
                return RECEIPT_DOC
        RECEIPT_DOC = write_receipt(build())
    return RECEIPT_DOC


def test_receipt_exists_with_schema_and_classifies_families():
    doc = receipt()
    assert RECEIPT.is_file(), f"missing {RECEIPT}"
    disk = json.loads(RECEIPT.read_text())
    assert disk["schema"] == SCHEMA
    assert doc["schema"] == SCHEMA
    fams = doc["families"]
    assert len(fams) >= 10
    ids = [f["id"] for f in fams]
    assert "RUN_LOG" in ids
    assert "HQ30G" in ids
    for f in fams:
        assert f["class"] in CLASSES, f"{f['id']} class {f['class']!r} not in {CLASSES}"
        for key in (
            "unique_logical_bytes",
            "unique_disk_bytes",
            "n_unique_blobs",
            "reachable_history",
            "why",
        ):
            assert key in f, f"{f['id']} missing {key}"


def test_every_declared_class_is_used():
    used = set(receipt()["classes_used"])
    missing = set(CLASSES) - used
    assert not missing, f"taxonomy unused: {sorted(missing)}"


def test_plan_is_prepared_not_executed():
    plan = receipt()["history_compaction_plan"]
    assert plan["executed"] is False
    assert "S020" in plan["s020_27"] or "s020_27" in plan
    assert plan["executed"] is not True
    assert "phases" in plan and len(plan["phases"]) >= 2
    phase_ids = [p["id"] for p in plan["phases"]]
    assert any("RUN_LOG" in i or i.endswith("RUN_LOG") or "RUN_LOG" in str(p.get("family_ids"))
               for i, p in zip(phase_ids, plan["phases"]))
    assert "predicted_pack_bytes" in plan["phases"][0]
    assert "rollback" in plan
    assert plan["rollback"]["before_any_rewrite"]
    assert any("bundle" in c for c in plan["rollback"]["before_any_rewrite"])
    assert any("mirror" in c for c in plan["rollback"]["before_any_rewrite"])
    rem = plan["remote_and_refs"]
    assert "force-push" in rem["implication"]
    assert rem["tags"] >= 1
    assert rem["local_heads"] >= 1


def test_current_git_is_measured_not_the_stale_32g():
    g = receipt()["current_git"]
    assert g["du_bytes"] > 0
    assert g["size_pack_bytes"] > 0
    # 32 GiB claim is stale; live pack is ~5.4 GiB. Allow a wide window so a
    # later lane adding objects does not fail this test, but 32G must not pass.
    gib = g["du_gib"]
    assert 1.0 <= gib < 20.0, f".git du_gib={gib} looks like the stale 32G or a failed measure"
    assert g["prior_steer_32g"]["measured_now_gib"] == gib


def test_live_pack_agrees_with_ledger_within_slack():
    raw = subprocess.run(
        ["git", "-C", str(REPO), "count-objects", "-v"],
        capture_output=True, text=True, check=True,
    ).stdout
    pack_kib = None
    for line in raw.splitlines():
        if line.startswith("size-pack:"):
            pack_kib = int(line.split(":")[1].strip().split()[0])
    assert pack_kib is not None
    live = pack_kib * 1024
    reported = receipt()["current_git"]["size_pack_bytes"]
    # Other worktrees share this object store; slack 25%.
    assert reported > 0
    ratio = abs(live - reported) / max(live, 1)
    assert ratio < 0.25, f"live pack {live} vs ledger {reported}"


def test_top_current_blobs_are_paths_with_bytes():
    top = receipt()["current_largest_blobs"]
    assert len(top) >= 20
    for row in top[:5]:
        assert row["bytes"] >= 1
        assert row["path"]
        assert "sha" in row
    # The current champion must be under GitHub's 50 MiB warning.
    assert top[0]["bytes"] < 50 * 1024 * 1024


def test_run_log_family_is_move_local_and_in_the_plan():
    run = next(f for f in receipt()["families"] if f["id"] == "RUN_LOG")
    assert run["class"] == "MOVE_LOCAL_FUTURE"
    assert run["in_rewrite_plan"] is True
    assert run["gitignore_rule"] == "workspace/campaign/odyssey/RUN_LOG.jsonl"
    # Live file was kept on disk; gitignore holds it out of git.
    gi = git_ok("show", "HEAD:.gitignore")
    assert "workspace/campaign/odyssey/RUN_LOG.jsonl" in gi


def test_hq80seg_is_absent_with_a_reason():
    row = next(f for f in receipt()["families"] if f["id"] == "HQ80SEG")
    assert row["reachable_history"] == "ABSENT"
    assert row["n_unique_blobs"] == 0
    assert row["absent_reason"]
    assert row["class"] == "MOVE_LOCAL_FUTURE"


def test_cas_consolidates_on_existing_artifacts_root():
    doc = receipt()
    assert doc["cas"]["root"] == EXISTING_ARTIFACT_ROOT == "artifacts/"
    assert doc["cas"]["layout"] == CAS_LAYOUT
    assert doc["cas"]["git_stores"] == "manifest"
    assert doc["cas"]["local_stores"] == "bytes"
    assert doc["cas"]["do_not_invent_a_second_root"] is True
    stores = doc["existing_stores"]["systems"]
    assert "artifacts_gitignore_root" in stores
    assert "hawking_experiments_is_not_the_cas" in str(doc["cas"]).lower() or doc["cas"]["hawking_experiments_is_not_the_cas"] is True


def test_gitignore_must_hold_is_actually_in_head():
    gi = git_ok("show", "HEAD:.gitignore")
    missing = [r for r in GITIGNORE_MUST_HOLD if r not in gi]
    assert not missing, missing
    assert receipt()["gitignore"]["missing_from_head"] == []


def test_policy_exists_and_does_not_contradict_gitignore():
    rel = "docs/ultragoals/ARTIFACT_STORAGE_POLICY.md"
    if POLICY.is_file():
        text = POLICY.read_text()
    else:
        # Sparse checkout: the file is in git, not on disk. Absence here is
        # not evidence the policy does not exist (see worktree contract).
        text = git_ok("show", f"HEAD:{rel}")
        assert text.strip(), f"missing {rel} on disk and in HEAD"
    text_l = text.lower()
    for needle in (
        "Git tracks",
        "must not",
        "artifacts/sha256",
        "RUN_LOG",
        "S020",
        "LFS",
        "executed: false",
        "/artifacts/",
        "*.hq80seg",
        "workspace/campaign/odyssey/RUN_LOG.jsonl",
        "content-addressed",
        "required-for-reproduction",
        "disposable",
        "negative science",
    ):
        assert needle.lower() in text_l, f"policy missing {needle!r}"
    # Do not invent a second CAS root.
    assert "do not invent" in text_l
    assert "second sha256" in text_l or "second cas" in text_l or "do not invent a second" in text_l
    gi = git_ok("show", "HEAD:.gitignore")
    # Every session gitignore rule the ledger tracks must be acknowledged.
    for rule in (
        "*.hq80seg",
        "*.f16",
        "workspace/campaign/odyssey/RUN_LOG.jsonl",
        "receipts/**/*.tar.xz",
        "/artifacts/",
    ):
        assert rule in gi
        assert rule in text


def test_family_id_splits_bulk_from_knowledge():
    assert family_id("workspace/campaign/odyssey/RUN_LOG.jsonl") == "RUN_LOG"
    assert family_id("foo/bar.hq30g") == "HQ30G"
    assert family_id("crates/hawking-core/src/lib.rs") == "CRATES_SOURCE"
    assert family_id("receipts/ascent-2026-08-16/X.json") == "RECEIPTS_ASCENT"
    assert family_id("receipts/headless/X.json") == "RECEIPTS_HEADLESS"
    assert FAMILY_META["RECEIPTS_ASCENT"]["class"] == "PRESERVE"
    assert FAMILY_META["CRATES_SOURCE"]["class"] == "KEEP_GIT"


def test_discipline_forbids_this_lane_from_executing_the_plan():
    d = receipt()["discipline"]
    assert d["s020_27_no_rewrite"]
    assert "executed is false" in d["s020_27_no_rewrite"] or "PREPARES" in d["s020_27_no_rewrite"]
    assert "does not remove old blobs" in d["s020_26_lfs"]
    assert receipt()["history_compaction_plan"]["executed"] is False

"""Negative controls for the autonomy-trial substrate freezer.

A freezer nobody has watched reject is a freezer that will silently drift into
fiction. These tests prove it can return INVALIDATED_BY_SUBSTRATE_MUTATION for
a graph file that moved, CLEAN for a file the graph never named, and that a
deleted graph file is reported rather than skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import autonomy_trial as at
from tools.future import frontiers as fr
from tools.future import trial_freeze as tf
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    REPO,
    HardwareClaimError,
    _assert_no_hardware_claims,
    sha256_file,
    write_receipt,
)


SENTINEL = "\n# trial_freeze_probe_do_not_keep\n"
OUTSIDE_GRAPH = "tools/future/ane_preboard.py"
IN_GRAPH = "tools/future/specimen_verify.py"
SUBPROCESS_ONLY = "tools/future/integration_attack.py"


def _rel(path: str) -> dict[str, object]:
    reach = tf.driver_reachability()
    for rec in reach["files"]:
        if rec["path"] == path:
            return rec
    raise AssertionError(f"{path} not in driver graph")


def test_graph_is_not_a_glob_and_reports_routes():
    """The load-bearing piece: reachability is an import graph, not tools/future/*.py."""
    reach = tf.driver_reachability()
    assert reach["driver"] == tf.DRIVER_REL
    assert reach["n_graph"] < reach["n_future_py_on_disk"]
    assert reach["n_graph"] >= 8
    names = {rec["path"] for rec in reach["files"]}
    assert tf.DRIVER_REL in names
    assert "tools/future/autonomy_trial.py" in names
    assert "tools/future/_common.py" in names
    assert IN_GRAPH in names
    assert SUBPROCESS_ONLY in names
    assert OUTSIDE_GRAPH not in names
    assert "tools/future/trial_freeze.py" not in names

    spec = _rel(IN_GRAPH)
    assert "import" in spec["routes"]
    assert "subprocess" in spec["routes"]
    assert spec["state"] == "HASHED"
    assert spec["sha256"]

    attack = _rel(SUBPROCESS_ONLY)
    assert attack["routes"] == ["subprocess"]
    assert "import" not in attack["routes"]

    driver = _rel(tf.DRIVER_REL)
    assert "driver" in driver["routes"]
    assert SUBPROCESS_ONLY in reach["argv_executed"]
    assert IN_GRAPH in reach["argv_executed"]


def test_glob_file_outside_the_graph_exists_on_disk():
    """The CLEAN negative control is only meaningful if the outside file is real."""
    assert (REPO / OUTSIDE_GRAPH).is_file()
    assert (REPO / IN_GRAPH).is_file()


def test_unknown_trial_id_is_refused():
    with pytest.raises(tf.FreezeRefused):
        tf.freeze("not-a-trial")
    with pytest.raises(tf.FreezeRefused):
        tf.freeze("")


def test_freeze_then_verify_is_clean():
    man = tf.freeze("15m")
    assert man["trial_id"] == "15m"
    assert man["schema"] == tf.SCHEMA
    assert man["evidence_class"] == "STATIC_ONLY"
    assert man["gpu_authority"] is False
    assert man["freeze_time"]
    assert man["code_files"]
    result = tf.verify_unchanged(man)
    assert result["verdict"] == tf.VERDICT_CLEAN
    assert result["moved_paths"] == []
    assert result["n_moved"] == 0


def test_mutate_file_in_graph_invalidates_naming_that_file():
    man = tf.freeze("1h")
    target = REPO / IN_GRAPH
    original = target.read_bytes()
    try:
        target.write_bytes(original + SENTINEL.encode())
        result = tf.verify_unchanged(man)
        assert result["verdict"] == tf.VERDICT_INVALIDATED
        assert IN_GRAPH in result["moved_paths"]
        hit = next(row for row in result["moved"] if row["path"] == IN_GRAPH)
        assert hit["state"] == "CHANGED"
        assert hit["frozen_sha256"] != hit["current_sha256"]
        assert hit["current_sha256"] == sha256_file(target)
    finally:
        target.write_bytes(original)
    restored = tf.verify_unchanged(man)
    assert restored["verdict"] == tf.VERDICT_CLEAN
    assert IN_GRAPH not in restored["moved_paths"]


def test_mutate_file_not_in_graph_stays_clean():
    man = tf.freeze("15m")
    names = {rec["path"] for rec in man["code_files"]}
    assert OUTSIDE_GRAPH not in names
    target = REPO / OUTSIDE_GRAPH
    original = target.read_bytes()
    try:
        target.write_bytes(original + SENTINEL.encode())
        result = tf.verify_unchanged(man)
        assert result["verdict"] == tf.VERDICT_CLEAN
        assert result["moved_paths"] == []
    finally:
        target.write_bytes(original)


def test_deleted_graph_file_is_reported_not_skipped(tmp_path: Path):
    man = tf.freeze("3h")
    ghost = tmp_path / "vanished_substrate.py"
    ghost.write_text("probe = 1\n", encoding="utf-8")
    digest = sha256_file(ghost)
    injected = json.loads(json.dumps(man))
    injected["code_files"].append(
        {
            "path": str(ghost),
            "sha256": digest,
            "state": "HASHED",
            "routes": ["import"],
        }
    )
    ghost.unlink()
    result = tf.verify_unchanged(injected)
    assert result["verdict"] == tf.VERDICT_INVALIDATED
    assert str(ghost) in result["moved_paths"]
    hit = next(row for row in result["moved"] if row["path"] == str(ghost))
    assert hit["state"] == "MISSING"
    assert hit["current_sha256"] is None
    assert hit["frozen_sha256"] == digest


def test_empty_or_malformed_manifest_is_not_clean():
    empty = tf.verify_unchanged({"schema": tf.SCHEMA, "code_files": []})
    assert empty["verdict"] == tf.VERDICT_INVALIDATED
    missing = tf.verify_unchanged({"schema": tf.SCHEMA})
    assert missing["verdict"] == tf.VERDICT_INVALIDATED
    garbage = tf.verify_unchanged("not-a-mapping")
    assert garbage["verdict"] == tf.VERDICT_INVALIDATED
    none = tf.verify_unchanged(None)
    assert none["verdict"] == tf.VERDICT_INVALIDATED


def test_manifest_contains_no_numeric_hardware_field():
    man = tf.freeze("15m")
    _assert_no_hardware_claims(man)
    blob = json.dumps(man)
    for key in HARDWARE_FIELDS:
        assert f'"{key}": ' not in blob or not isinstance(man.get(key), (int, float))
        assert man.get(key) not in (0, 0.0, 1, 1.0)


def test_write_receipt_still_refuses_a_numeric_hardware_field():
    """Negative control on the plumbing freeze relies on, not a restatement of it."""
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "_trial_freeze_hardware_probe.json",
            {"schema": "probe", "tps": 12.0, "gpu_authority": False},
            "tools/future/test_trial_freeze.py",
        )


def test_lanes_are_the_frontier_vocabulary_not_a_second_list():
    man = tf.freeze("15m")
    assert man["available_lanes"] == list(fr.THIS_HOST_LANES)
    assert man["blocked_lanes"] == list(fr.BLOCKED_ON_THIS_HOST)
    src = (REPO / "tools/future/trial_freeze.py").read_text()
    assert "THIS_HOST_LANES" in src
    assert "BLOCKED_ON_THIS_HOST" in src
    assert "AVAILABLE_LANES =" not in src
    assert "BLOCKED_LANES =" not in src


def test_resident_model_process_is_unavailable():
    man = tf.freeze("6h")
    assert man["resident_model_process"]["value"] == tf.UNAVAILABLE
    assert man["resident_model_process"]["why"]
    model = man["model_identity"]
    assert "value" in model
    if model["value"] == tf.UNAVAILABLE:
        assert model.get("why")


def test_startup_receipts_are_named_and_absence_is_recorded():
    man = tf.freeze("15m")
    paths = [row["path"] for row in man["startup_receipts"]]
    assert at.FRONTIER_REL in paths
    assert "receipts/future/ORCHESTRATION_BINDINGS.json" in paths
    assert "receipts/future/RESIDENT_IDENTITY.json" in paths
    for row in man["startup_receipts"]:
        assert row["state"] in {"HASHED", "ABSENT", "UNREADABLE"}
        if row["state"] == "ABSENT":
            assert row["sha256"] is None
            assert row["why"]
        if row["state"] == "HASHED":
            assert row["sha256"]


def test_frontier_digest_is_ids_and_states_or_unavailable():
    man = tf.freeze("15m")
    digest = man["frontier_digest"]
    if digest == tf.UNAVAILABLE:
        assert man["frontier_why"]
        return
    assert isinstance(digest, str) and len(digest) == 64
    items = man["frontier_items"]
    assert items
    ids = [row["id"] for row in items]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
    for row in items:
        assert row["id"].startswith("FT.")
        assert row["kind"] in {"NEXT_WORK", "BLOCKED", "OPEN_QUESTION"}
        assert row["state"] in {"NEXT_WORK", "BLOCKED", "OPEN_QUESTION", "SLEEPING"}


def test_git_slot_does_not_invent_a_clean_tree_on_failure():
    man = tf.freeze("15m")
    slot = man["git"]
    assert slot["head"]
    assert slot["branch"]
    assert slot["dirty"] in {True, False, tf.UNAVAILABLE}
    if slot["dirty"] is tf.UNAVAILABLE:
        assert slot["why"]


def test_build_writes_the_named_receipt_holding_frozen_builds():
    out = tf.build()
    assert out == RECEIPTS / tf.RECEIPT
    doc = json.loads(out.read_text())
    assert doc["schema"] == tf.SCHEMA
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["driver"] == tf.DRIVER_REL
    assert doc["trial_ids"] == list(at.TRIAL_IDS)
    assert len(doc["frozen_builds"]) == len(at.TRIAL_IDS)
    for row in doc["frozen_builds"]:
        assert row["trial_id"] in at.TRIAL_IDS
        assert row["code_files"]
        result = tf.verify_unchanged(row)
        assert result["verdict"] == tf.VERDICT_CLEAN
    assert IN_GRAPH in doc["files_found_by_both_routes"]
    assert SUBPROCESS_ONLY in doc["files_found_by_subprocess_only"]
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert doc["resident_callable"]["entry_point"]
    assert doc["resident_callable"]["frontier"] == "FT.CHILD_RESIDENT.launch"
    _assert_no_hardware_claims(doc)


def test_subprocess_only_mutation_also_invalidates():
    """Hashing only autonomy_run.py would miss this file. The freezer must not."""
    man = tf.freeze("15m")
    target = REPO / SUBPROCESS_ONLY
    original = target.read_bytes()
    try:
        target.write_bytes(original + SENTINEL.encode())
        result = tf.verify_unchanged(man)
        assert result["verdict"] == tf.VERDICT_INVALIDATED
        assert SUBPROCESS_ONLY in result["moved_paths"]
    finally:
        target.write_bytes(original)

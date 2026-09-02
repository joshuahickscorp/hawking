"""Parity gate for the JSON-map sidecar.

A fast path that silently changes a verdict is worse than a slow one.
These tests compare the sidecar to a full json.loads over EVERY module
(531) and EVERY gate (71), then measure time and bytes-read for both
paths. They do not write receipts/ and they do not assemble the inventory.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from tools.audit import artifact_index as ai
from tools.audit import reachability_triage as rt


@pytest.fixture(scope="module")
def triage_json(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("triage")
    p = d / "REACHABILITY_TRIAGE.json"
    blob = ai.git_show_bytes(ai.TRIAGE_REL)
    assert len(blob) > 1_000_000, "triage receipt missing from HEAD"
    p.write_bytes(blob)
    return p


@pytest.fixture(scope="module")
def graph_json(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("graph")
    p = d / "CAPABILITY_GRAPH.json"
    blob = ai.git_show_bytes(ai.GRAPH_REL)
    assert len(blob) > 10_000, "capability graph missing from HEAD"
    p.write_bytes(blob)
    return p


@pytest.fixture(scope="module")
def triage_index(triage_json, tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("tidx")
    return ai.ensure_index(
        triage_json, d / "triage.sqlite", maps=["modules"], prefer_rust=True
    )


@pytest.fixture(scope="module")
def graph_index(graph_json, tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("gidx")
    return ai.ensure_index(
        graph_json, d / "graph.sqlite", maps=["gates", "genes"], prefer_rust=True
    )


def test_walker_escaped_quote_and_brace_in_string():
    raw = (
        b'{"modules": {"k": {"summary": "say \\"hi\\" and a brace {",'
        b' "classification": "BUILT"}}}'
    )
    top = ai.walk_object_members(raw, 0)
    assert [k for k, _, _ in top] == ["modules"]
    _name, start, end = top[0]
    members = ai.walk_object_members(raw, start)
    assert members[0][0] == "k"
    obj = json.loads(raw[members[0][1] : members[0][2]])
    assert obj["summary"] == 'say "hi" and a brace {'
    assert obj["classification"] == "BUILT"


def test_capability_id_matches_triage():
    samples = [
        "tools/future/capacity_inference_rule.py",
        "tools/accelerator/accelerator_runner.py",
        "tools/future/__init__.py",
        "tools/headless/foo/bar.py",
    ]
    for s in samples:
        assert ai.capability_id(s) == rt.capability_id(s)


def test_all_531_module_objects_identical(triage_json, triage_index):
    report = ai.parity_map(triage_json, "modules", index_path=triage_index)
    print("PARITY_MODULES", json.dumps({k: report[k] for k in report if k != "mismatches"}))
    assert report["n"] == 531, report
    assert report["ok"] is True, report["mismatches"]
    assert report["n_mismatch"] == 0
    assert report["n_equal"] == 531


def test_all_71_gate_objects_identical(graph_json, graph_index):
    report = ai.parity_map(graph_json, "gates", index_path=graph_index)
    print("PARITY_GATES", json.dumps({k: report[k] for k in report if k != "mismatches"}))
    assert report["n"] == 71, report
    assert report["ok"] is True, report["mismatches"]


def test_unreachable_keyset_identical(triage_json, triage_index):
    measured = ai.measure_filter(
        triage_json, "modules", classification="UNREACHABLE", index_path=triage_index
    )
    print("UNREACHABLE", json.dumps(measured))
    assert measured["equal"] is True
    assert measured["n"] == 305
    assert measured["index_bytes"] < measured["full_bytes"]


def test_built_gates_keyset_identical(graph_json, graph_index):
    measured = ai.measure_filter(
        graph_json, "gates", status="BUILT", index_path=graph_index
    )
    print("BUILT_GATES", json.dumps(measured))
    assert measured["equal"] is True

    # Count is derived from the graph, not frozen. Lane f1 split BUILT into
    # wired+accepted, so BUILT legitimately went 26 -> 0; a hardcoded 26 was
    # asserting yesterday's vocabulary. Fidelity ("equal") is what this guards.
    doc = json.loads(graph_json.read_text())
    gates = doc["gates"]
    rows = gates.values() if isinstance(gates, dict) else gates
    expected = sum(1 for g in rows if g.get("status") == "BUILT")
    assert measured["n"] == expected
    assert measured["index_bytes"] < measured["full_bytes"]


def test_one_module_bytes_and_time(triage_json, triage_index):
    key = "tools/future/status_causality.py"
    measured = ai.measure_one(triage_json, "modules", key, index_path=triage_index)
    print("MEASURE_ONE", json.dumps(measured))
    assert measured["equal"] is True
    assert measured["index_bytes"] < measured["full_bytes"]
    assert measured["index_bytes"] < 50_000
    assert measured["full_bytes"] == triage_json.stat().st_size
    # Targeted get must not read the whole receipt.
    assert measured["index_bytes"] * 10 < measured["full_bytes"]


def test_query_module_matches_full_parse_for_every_module(triage_json, triage_index):
    """rt.query_module is the HCLI-facing API; pin it to the oracle too."""
    doc, full_bytes, _ = ai.full_parse(triage_json)
    modules = doc["modules"]
    # Drive the same sqlite the fixture built, via get() (what query_module wraps).
    mismatches = []
    bytes_sum = 0
    for name, row in modules.items():
        hit = ai.get("modules", name, json_path=triage_json, index_path=triage_index)
        bytes_sum += hit.bytes_read
        if hit.value != row:
            mismatches.append(name)
    assert mismatches == []
    assert len(modules) == 531
    print(
        "QUERY_ALL_BYTES",
        json.dumps({"full_bytes": full_bytes, "index_bytes_sum": bytes_sum, "n": 531}),
    )


def test_harness_detects_classification_corruption(triage_json, tmp_path):
    """Mutation check: a gutted sidecar must fail the UNREACHABLE comparison.

    If this stays green with the index lying, the parity test is vacuous.
    """
    idx = tmp_path / "mut.sqlite"
    ai.build_python(triage_json, idx, maps=["modules"])
    before = ai.measure_filter(
        triage_json, "modules", classification="UNREACHABLE", index_path=idx
    )
    assert before["equal"] is True
    conn = sqlite3.connect(str(idx))
    row = conn.execute(
        "SELECT key FROM entity WHERE map_name='modules' AND classification='UNREACHABLE' "
        "ORDER BY key LIMIT 1"
    ).fetchone()
    assert row, "need an UNREACHABLE row to mutate"
    conn.execute(
        "UPDATE entity SET classification='BUILT' WHERE map_name='modules' AND key=?",
        (row[0],),
    )
    conn.commit()
    conn.close()
    # Size+mtime of the JSON are unchanged, so ensure_index will trust this
    # corrupted sidecar. The filter comparison must then fail.
    with pytest.raises(ai.ArtifactParityError):
        ai.measure_filter(
            triage_json, "modules", classification="UNREACHABLE", index_path=idx
        )
    print("MUTATION_DETECTED", row[0])


def test_harness_detects_object_mismatch(triage_json, tmp_path):
    idx = tmp_path / "mut2.sqlite"
    ai.build_python(triage_json, idx, maps=["modules"])
    conn = sqlite3.connect(str(idx))
    key, start, end, stored = conn.execute(
        "SELECT key, start, end, json FROM entity WHERE map_name='modules' ORDER BY key LIMIT 1"
    ).fetchone()
    other = conn.execute(
        "SELECT json FROM entity WHERE map_name='modules' AND key != ? ORDER BY key LIMIT 1",
        (key,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE entity SET json=? WHERE map_name='modules' AND key=?",
        (other, key),
    )
    conn.commit()
    conn.close()
    # pread of the original file no longer matches the stored blob.
    with pytest.raises(ai.ArtifactParityError):
        ai.get("modules", key, json_path=triage_json, index_path=idx)
    print("OBJECT_MUTATION_DETECTED", key)


def test_original_receipts_unchanged():
    """The durable JSON is the evidence trail. We must not rewrite it."""
    proc = subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "receipts/future/REACHABILITY_TRIAGE.json",
            "civilization/CAPABILITY_GRAPH.json",
        ],
        cwd=str(ai.REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_inspect_fast_path_matches_full_parse(triage_json, triage_index, monkeypatch):
    monkeypatch.setattr(rt, "_triage_json_path", lambda: triage_json)
    # Point artifact_index.ensure_index at the already-built sidecar by
    # wrapping get_by_cap_id's json_path; inspect uses _lookup_module_row.
    cap = "future.status_causality"
    fast = rt.inspect({"id": cap})
    monkeypatch.setenv("HCLI_TRIAGE_FULLPARSE", "1")
    # load_triage still hits HEAD/live, not triage_json. Compare via get vs full_parse.
    doc, _, _ = ai.full_parse(triage_json)
    row = doc["modules"]["tools/future/status_causality.py"]
    assert fast["ok"] is True
    assert fast["value"]["disposition"] == row["disposition"]
    assert fast["value"]["classification"] == row["classification"]
    assert fast["value"]["module"] == row["module"]


def test_rust_builder_matches_python_when_present(triage_json, tmp_path):
    if ai._artifact_bin() is None:
        pytest.skip("hawking-artifact binary not built")
    rust_idx = tmp_path / "rust.sqlite"
    py_idx = tmp_path / "py.sqlite"
    ai.build_rust(triage_json, rust_idx, maps=["modules"])
    ai.build_python(triage_json, py_idx, maps=["modules"])
    r_keys, _, _ = ai.list_keys("modules", json_path=triage_json, index_path=rust_idx)
    p_keys, _, _ = ai.list_keys("modules", json_path=triage_json, index_path=py_idx)
    assert r_keys == p_keys
    report = ai.parity_map(triage_json, "modules", index_path=rust_idx)
    assert report["ok"] is True
    assert report["n"] == 531

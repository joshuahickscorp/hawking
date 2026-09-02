"""Acceptance tests for the VMCP lane.

These tests CALL the gate runners (an import is not a call) and check the
roadmap span, not a nearby receipt. Criteria are not weakened.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.acceptance.vmcp import common as C
from tools.acceptance.vmcp.gates import RUNNERS
from tools.acceptance.vmcp.run import run_all, write_receipts

ASSIGNED = list(C.GATES)


@pytest.fixture(scope="session")
def results() -> dict:
    out = run_all(ASSIGNED)
    write_receipts(out)
    return out


def test_every_assigned_gate_ran(results):
    assert set(results) == set(ASSIGNED)
    for gate, doc in results.items():
        assert doc["schema"] == C.SCHEMA
        assert doc["gate"] == gate
        assert doc["verdict"] in {"ACCEPTED", "BLOCKED"}
        assert doc["criterion_weakened"] is False
        assert doc["criterion"]["weakened"] is False
        assert doc["gpu_authority"] is False


def test_criterion_quotes_match_roadmap_spans(results):
    for gate, doc in results.items():
        meta = C.GATES[gate]
        quoted = C.quote_roadmap(meta["start"], meta["end"])
        assert doc["criterion"]["start_line"] == meta["start"]
        assert doc["criterion"]["end_line"] == meta["end"]
        assert doc["criterion"]["quoted"] == quoted
        assert quoted.strip(), f"{gate} quoted an empty span"


def test_no_acceptance_criterion_was_altered():
    # The catalog spans in this lane match CAPABILITY_GRAPH.json.
    import json

    graph = json.loads((C.REPO / "civilization" / "CAPABILITY_GRAPH.json").read_text())
    for gate, meta in C.GATES.items():
        span = graph["gates"][gate]["acceptance_span"]
        assert span["start_line"] == meta["start"], gate
        assert span["end_line"] == meta["end"], gate


def test_accepted_gates_have_call_sites_and_real_output(results):
    accepted = [g for g, d in results.items() if d["verdict"] == "ACCEPTED"]
    assert accepted, "expected at least one gate to be acceptable today"
    for gate in accepted:
        doc = results[gate]
        calls = doc["invoked_symbols"]
        assert calls, f"{gate} ACCEPTED without a call of any symbol"
        assert all(c["kind"] == "call" for c in calls)
        assert any(c.get("raised") is False for c in calls), f"{gate} every call raised"
        assert doc["checks"], f"{gate} has no checks"
        assert all(c["ok"] for c in doc["checks"]), f"{gate} ACCEPTED with a failed check"
        assert doc["blocker"] is None
        assert doc["measured"] not in (None, {}, [])


def test_blocked_gates_name_exact_missing_input(results):
    blocked = [g for g, d in results.items() if d["verdict"] == "BLOCKED"]
    for gate in blocked:
        doc = results[gate]
        blocker = doc["blocker"] or {}
        assert blocker.get("missing"), f"{gate} BLOCKED without missing="
        assert blocker.get("why"), f"{gate} BLOCKED without why="
        assert "not implemented yet" not in str(blocker.get("why")).lower()


def test_receipts_written_under_receipts_acceptance(results):
    for gate in results:
        path = C.RECEIPT_DIR / f"{gate}.json"
        assert path.is_file(), path
        assert path.stat().st_size > 200
    index = C.RECEIPT_DIR / "INDEX.json"
    assert index.is_file()
    import json

    doc = json.loads(index.read_text())
    assert doc["criterion_weakened"] is False
    assert doc["accepted_count"] == sum(1 for d in results.values() if d["verdict"] == "ACCEPTED")
    assert doc["blocked_count"] == sum(1 for d in results.values() if d["verdict"] == "BLOCKED")
    assert doc["assigned_count"] == len(ASSIGNED)


def test_e4_receipt_missing_canary_is_not_complete():
    row = {k: "x" for k in C.E4_RECEIPT_FIELDS}
    assert C.receipt_is_complete(row) is True
    dropped = dict(row)
    dropped.pop("canary")
    assert C.receipt_is_complete(dropped) is False
    dropped2 = dict(row)
    dropped2["canary"] = None
    assert C.receipt_is_complete(dropped2) is False


def test_receipt_law_untraced_tool_excluded(results):
    doc = results["VMCP_RECEIPT_LAW"]
    if doc["verdict"] != "ACCEPTED":
        pytest.skip("receipt law blocked")
    assert doc["measured"]["untraced_excluded"] is True
    receipt = doc["measured"]["receipt"]
    for key in C.E4_RECEIPT_FIELDS:
        assert key in receipt and receipt[key] is not None


def test_deep_digest_mutation_changes_hash(results):
    doc = results["VMCP_DEEP_DIGEST"]
    if doc["verdict"] != "ACCEPTED":
        pytest.skip("deep digest blocked")
    checks = {c["id"]: c for c in doc["checks"]}
    assert checks["canonical_key_order_stable"]["ok"] is True
    assert checks["value_mutation_changes_digest"]["ok"] is True
    assert checks["prove_red_green"]["ok"] is True


def test_truth_ledger_rejects_forged_claim(results):
    doc = results["VMCP_TRUTH_LEDGER"]
    if doc["verdict"] != "ACCEPTED":
        pytest.skip("truth ledger blocked")
    assert doc["output"]["verify_content_hash_after_claim_forge"] is False
    assert doc["output"]["from_dict_forged_claim"]["verdict"] == "DETECTED"


def test_compact_surface_parked_acts_are_not_empty_success(results):
    doc = results["VMCP_COMPACT_SURFACE"]
    if doc["verdict"] != "ACCEPTED":
        pytest.skip("compact surface blocked")
    for act in ("open", "make", "fix", "keep"):
        env = doc["output"][act]
        assert env["status"] == "PARKED"
        assert env["empty_success"] is False
        assert env["looked"] is False
        assert env["artifacts"] is None
        assert env["evidence"] is None
        assert env["residuals"]
        assert env["next_actions"]
        for key in C.E14_RESPONSE:
            assert key in env


def test_file_classifier_distinguishes_png_zip_wasm(results):
    doc = results["VMCP_FILE_CLASSIFIER"]
    if doc["verdict"] != "ACCEPTED":
        pytest.skip("file classifier blocked")
    sniffed = doc["output"]["sniffed"]
    assert sniffed["png"]["sniff"] == "image/png"
    assert sniffed["zip"]["sniff"] == "application/zip"
    wasm = doc["measured"]["WASM identification/validation"]
    assert wasm["magic_ok"] is True
    assert wasm["version"] == 1


def test_visual_diff_canary_red_then_green(results):
    doc = results["VMCP_VISUAL_DIFF"]
    if doc["verdict"] != "ACCEPTED":
        pytest.skip("visual diff blocked")
    assert doc["measured"]["pixel"]["changed_px"] > 0
    assert doc["measured"]["identical"]["identical"] is True
    assert 0.0 <= doc["measured"]["ssim"] <= 1.0


def test_spatial_remove_faces_red_restore_green(results):
    doc = results["VMCP_SPATIAL_VALIDATE"]
    if doc["verdict"] != "ACCEPTED":
        pytest.skip("spatial blocked")
    can = doc["measured"]["canaries"]
    assert can["remove_faces"]["face_count"] == 0
    assert can["restore"]["face_count"] == doc["measured"]["inventory"]["face_count"]


def test_state_lattice_does_not_invent_director_state(results):
    doc = results["VMCP_STATE_LATTICE"]
    if doc["verdict"] != "ACCEPTED":
        pytest.skip("lattice blocked")
    rows = {r["name"]: r for r in doc["output"]["rows"]}
    assert rows["DIRECTOR_STATE"]["disposition"] == "REJECT"
    assert rows["DIRECTOR_STATE"]["named_type_present"] is False
    assert rows["DEEP_DIGEST"]["disposition"] == "CONSOLIDATE"
    assert rows["TRUTH_LEDGER"]["canary_verdict"] == "DETECTED"


def test_web_and_pty_blockers_are_specific(results):
    web = results["VMCP_WEB_CAPTURE"]
    assert web["verdict"] == "BLOCKED"
    assert "Chrome" in web["blocker"]["missing"] or "chrome" in web["blocker"]["missing"].lower()
    pty = results["VMCP_PTY_CAPTURE"]
    assert pty["verdict"] == "BLOCKED"
    assert "PTY" in pty["blocker"]["missing"]
    # Must not look like an empty success.
    assert web["measured"]["local_dom"]
    assert pty["measured"]["pty_probe"]


def test_behavior_lab_and_offload_name_missing_paths(results):
    bhv = results["AGENTOS_BEHAVIOR_LAB"]
    assert bhv["verdict"] == "BLOCKED"
    assert "BHV-01" in bhv["blocker"]["missing"]
    assert bhv["measured"]["fixtures_implemented"] == 0
    off = results["AGENTOS_DETERMINISTIC_OFFLOAD"]
    assert off["verdict"] == "BLOCKED"
    assert "claude_offload_bench.py" in off["blocker"]["missing"]


def test_runners_are_calls_not_just_imports():
    assert set(RUNNERS) == set(ASSIGNED)
    src = Path(__file__).with_name("gates.py").read_text(encoding="utf-8")
    # Load-bearing: these symbols are passed to C.call (an import is not a call).
    assert "tools.headless.hcli_vmcp_integration.observe_file" in src
    assert "visionmcp.worldir.canonical.content_digest" in src
    assert "tools.future.vmcp.compact_surface" in src
    assert "visionmcp.worlds.spatial.io.obj.obj_file_counts" in src
    assert "visionmcp.capabilities.core_doctor_report" in src
    assert "C.call(" in src

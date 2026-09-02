"""Tests for tools/audit/reachability_triage.py.

Hermetic: decide() never leaves a row undispositioned, receipts/ are not
callers, own-test-only is not CONNECTED.

Live: the triage tool emits an inventory in which every module carries a
disposition, and count(undispositioned)==0. Spot-checks 5 UNREACHABLE and
5 CONNECTED rows against git grep.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.audit import reachability_triage as rt
from tools.future._common import RECEIPTS, load_json
from tools.future import capability_reachability as cr


# --------------------------------------------------------------------------
# Hermetic classification
# --------------------------------------------------------------------------


def _row(**overrides):
    base = {
        "module": "tools/future/widget.py",
        "callable_outside_tests": False,
        "hcli_reachable": False,
        "is_stub": False,
        "is_package_marker": False,
        "retired": None,
        "has_test": False,
        "test_exercises": False,
        "test_reads_receipt_only": False,
        "orchestration_bound": False,
    }
    base.update(overrides)
    rt.decide(base)
    return base


def test_decide_always_sets_a_known_disposition():
    cases = [
        {},
        {"hcli_reachable": True, "callable_outside_tests": True},
        {"callable_outside_tests": True},
        {"is_stub": True},
        {"has_test": True, "test_exercises": True},
        {"has_test": True, "test_reads_receipt_only": True},
        {"retired": "superseded by", "callable_outside_tests": False},
        {"orchestration_bound": True},
        {"is_stub": True, "is_package_marker": True, "hcli_reachable": True},
    ]
    for kwargs in cases:
        row = _row(**kwargs)
        assert row["disposition"] in rt.DISPOSITIONS, kwargs
        assert row["classification"] in rt.CLASSIFICATIONS, kwargs
        assert row["disposition_full"]


def test_hcli_reachable_is_connected():
    row = _row(hcli_reachable=True, callable_outside_tests=True)
    assert row["disposition"] == "CONNECTED"
    assert row["classification"] == "BUILT"


def test_triage_hcli_invocations_ignore_imports():
    """kind=import is never an HCLI invocation. Same rule as the auditor."""
    import_site = {"file": "hcli/agentos/foo.py", "line": 10, "kind": "import"}
    call_site = {"file": "hcli/agentos/foo.py", "line": 20, "kind": "call"}
    sidecar_call = {"file": "tools/future/bar.py", "line": 3, "kind": "call"}
    sub = {"file": "hcli/agentos_cli.py", "line": 50, "kind": "subprocess"}
    assert rt.hcli_invocations([], [import_site]) == []
    assert rt.hcli_invocations([sidecar_call], [import_site]) == []
    assert rt.hcli_invocations([call_site], [import_site]) == [call_site]
    assert rt.hcli_invocations([], [import_site, sub]) == [sub]
    assert rt.cite_hcli_path(
        "tools.future.widget", [import_site], {}, invocations=[]
    ) is None
    assert rt.cite_hcli_path(
        "tools.future.widget", [import_site], {}, invocations=[call_site]
    ) == "hcli/agentos/foo.py:20 (call)"


def test_sidecar_only_caller_is_parked_dormant():
    row = _row(callable_outside_tests=True, hcli_reachable=False)
    assert row["disposition"] == "PARKED"
    assert row["classification"] == "DORMANT"
    assert "HCLI" in row["wake_condition"]
    assert isinstance(row["wake"], dict)
    assert row["wake"]["required_kind"] == "call"
    assert row["wake"]["kind"]
    assert row["wake"]["predicate"]
    assert row["wake"]["blocker"]
    assert row["wake"]["schema"] == rt.WAKE_SCHEMA


def test_nothing_calls_it_is_unreachable_parked():
    row = _row(has_test=True, test_exercises=True)
    assert row["classification"] == "UNREACHABLE"
    assert row["disposition"] == "PARKED"
    assert "first production importer" in row["wake_condition"]
    assert row["wake"]["required_kind"] == "call"
    assert row["wake"]["kind"] == rt.WAKE_KIND_PRODUCTION_SYMBOL_CALL
    assert "AST Call" in row["wake"]["predicate"]


def test_stub_with_no_test_is_archive_candidate():
    row = _row(is_stub=True, has_test=False)
    assert row["disposition"] == "ARCHIVE_CANDIDATE"
    assert row["classification"] == "SCAFFOLDED"
    assert row["archive_reason"]
    assert row["archive"]["reason"] == row["archive_reason"]
    assert row["archive"]["deleted"] is False
    assert row["archive"]["schema"] == rt.ARCHIVE_SCHEMA


def test_retired_uncalled_is_archive_candidate():
    row = _row(retired="this module is retired")
    assert row["disposition"] == "ARCHIVE_CANDIDATE"
    assert row["classification"] == "ARCHIVE_CANDIDATE"


def test_mentioning_someone_elses_retirement_is_not_self_retired():
    """A docstring that says a campaign/driver was retired is not self-archive."""
    text = (
        '"""Retire the Ramanujan campaign by indexing it.\n\n'
        "The campaign is retired; its evidence is not. This script makes that durable.\n"
        '"""\n'
    )
    shape = rt.module_shape(text, "ramanujan_disband.py")
    assert shape["retired"] is None

    text2 = (
        '"""git commit takes the index.\n\n'
        "The legacy Odyssey driver was retired for ending each window with a bare commit.\n"
        '"""\n'
    )
    assert rt.module_shape(text2, "index_provenance.py")["retired"] is None

    text3 = (
        '"""This module is retired; use tools/future/wakeup.py instead.\n"""\n'
    )
    assert rt.module_shape(text3, "old.py")["retired"]


def test_data_path_helper():
    assert rt.is_data_path(Path("receipts/future/nowait/_bound_run.py"))
    assert rt.is_data_path(Path("hcli/fixtures/x.py"))
    assert not rt.is_data_path(Path("hcli/agentos/native_gate.py"))
    assert not rt.is_production_path(Path("tools/future/test_widget.py"))
    assert rt.is_production_path(Path("hcli/agentos/native_gate.py"))


def test_partition_drops_receipts_as_callers(tmp_path, monkeypatch):
    """A generated runner under receipts/ must not make a module callable."""
    rt.install_engine_extensions()
    monkeypatch.setattr(cr, "REPO", tmp_path)
    cr._TEXT_CACHE.clear()
    rt._INDEX_CACHE.clear()

    widget = tmp_path / "tools" / "future" / "widget.py"
    widget.parent.mkdir(parents=True)
    widget.write_text("def gadget():\n    return 1\n")
    own_test = tmp_path / "tools" / "future" / "test_widget.py"
    own_test.write_text(
        "from tools.future.widget import gadget\n\n"
        "def test_gadget():\n    assert gadget() == 1\n"
    )
    receipt_runner = tmp_path / "receipts" / "future" / "_bound_run.py"
    receipt_runner.parent.mkdir(parents=True)
    receipt_runner.write_text(
        "from tools.future.widget import gadget\n\n"
        "def run():\n    return gadget()\n"
    )
    files = [widget, own_test, receipt_runner]
    idx = cr.build_repo_index(files=files)
    sites = cr.find_module_import_sites(idx, "tools.future.widget", exclude_files=(widget,))
    cap = cr.build_capability(
        "tools.future.widget",
        "module",
        defined=True,
        registered=False,
        resident_visible=False,
        sites=sites,
    )
    assert cap["tested"] is True
    assert cap["callable"] is False, (
        "receipts/ import must not count as a production call site"
    )
    assert cap["call_sites"] == []


def test_real_outside_caller_still_counts(tmp_path, monkeypatch):
    rt.install_engine_extensions()
    monkeypatch.setattr(cr, "REPO", tmp_path)
    cr._TEXT_CACHE.clear()
    rt._INDEX_CACHE.clear()

    widget = tmp_path / "tools" / "future" / "widget.py"
    widget.parent.mkdir(parents=True)
    widget.write_text("def gadget():\n    return 1\n")
    caller = tmp_path / "hcli" / "gate.py"
    caller.parent.mkdir(parents=True)
    caller.write_text(
        "from tools.future.widget import gadget\n\n"
        "def use():\n    return gadget()\n"
    )
    idx = cr.build_repo_index(files=[widget, caller])
    sites = cr.find_module_import_sites(idx, "tools.future.widget", exclude_files=(widget,))
    cap = cr.build_capability(
        "tools.future.widget",
        "module",
        defined=True,
        registered=False,
        resident_visible=None,
        sites=sites,
    )
    assert cap["callable"] is True
    assert cap["call_sites"][0]["file"] == "hcli/gate.py"


# --------------------------------------------------------------------------
# Live inventory
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_doc():
    out = rt.selftest()
    return load_json(out)


def test_live_inventory_schema_and_zero_undisposed(live_doc):
    assert live_doc["schema"] == rt.SCHEMA
    assert live_doc["seal_sha256"]
    assert live_doc["evidence_tier"] == "STATIC"
    assert live_doc["engine"]["assemble_used"] is True
    counts = live_doc["counts"]
    assert counts["modules"] > 0
    assert counts["undispositioned"] == 0
    assert live_doc["undispositioned"] == []
    assert counts.get("parked_missing_wake") == 0
    assert counts.get("archive_missing_reason") == 0
    modules = live_doc["modules"]
    assert len(modules) == counts["modules"]
    for name, row in modules.items():
        assert row["disposition"] in rt.DISPOSITIONS, name
        assert row["classification"] in rt.CLASSIFICATIONS, name
        assert row["evidence_tier"] == "STATIC", name


def test_live_status_causality_is_connected_from_hcli_gates(live_doc):
    """The brief's worked example: tools.future is not uniformly dead.

    status_causality is imported by the HCLI gates. A blanket 'tools/future
    is dead' or 'tools/future is wired' fails this row.
    """
    row = live_doc["modules"]["tools/future/status_causality.py"]
    assert row["disposition"] == "CONNECTED"
    assert row["hcli_reachable"] is True
    files = {s["file"] for s in row["call_sites"]}
    assert any(f.startswith("hcli/agentos/") and f.endswith("_gate.py") for f in files), (
        f"expected an hcli/agentos/*_gate.py call site, got {sorted(files)}"
    )
    inv = row.get("hcli_invocations") or []
    assert inv, "CONNECTED must cite an HCLI symbol Call, not an import"
    assert all(s.get("kind") in {"call", "subprocess"} for s in inv)
    inv_files = {s["file"] for s in inv}
    assert any(f.startswith("hcli/agentos/") and f.endswith("_gate.py") for f in inv_files), (
        f"expected an hcli/agentos/*_gate.py symbol call, got {sorted(inv_files)}"
    )


def test_live_state_transitions_have_file_line_evidence(live_doc):
    findings = live_doc["state_transitions"]
    assert findings
    missing = [t for t in findings if t["status"] == "EXIT_MISSING"]
    assert missing, "the SEALED_SOURCE_READY / resident-child findings must surface"
    for t in findings:
        assert t["file"]
        assert t["enter"]
        assert t["status"] in {"EXIT_EXISTS", "EXIT_MISSING"}
        assert t["evidence_tier"] == "STATIC"


def test_live_spotcheck_five_unreachable_and_five_connected(live_doc):
    """Mandatory: 5 UNREACHABLE + 5 CONNECTED, each verified with git grep -nE."""
    spot = rt.run_spotchecks(live_doc)
    assert len(spot["unreachable"]) == 5
    assert len(spot["connected"]) == 5
    assert len(spot["checks"]) == 10
    by_verdict = {k: 0 for k in ("UNREACHABLE", "CONNECTED")}
    for check in spot["checks"]:
        by_verdict[check["verdict"]] += 1
        assert check["command"].startswith("git --no-optional-locks grep -nE")
        if check["verdict"] == "UNREACHABLE":
            assert check["production_import_hits"] == []
        else:
            assert check["hcli_path"] or check["hcli_hits"]
    assert by_verdict == {"UNREACHABLE": 5, "CONNECTED": 5}
    written = load_json(RECEIPTS / "REACHABILITY_TRIAGE_SPOTCHECKS.json")
    assert written["schema"] == "hawking.audit.reachability_triage_spotchecks.v1"
    assert len(written["checks"]) == 10

# --- capability manifest adapter tests ---
#
# A capability does not exist until something CALLS it. These tests require an
# AST Call of a named symbol in this adapter; an import of the target module
# is not enough. The live daemon is not started, signalled, or imported.
# Receipts are not written.



ADAPTER = Path(rt.__file__)
FIRE_CALL = "result = fires_on(levels, semantics_comparable=comparable)"
MUTATION_STUB = 'result = {"fired": False, "unreachable_mutation": True}'


def test_surface_is_three_verbs_not_one_per_module():
    assert rt.SURFACE_VERB_COUNT == 3
    assert len(rt.SURFACE_VERBS) == 3
    specs = rt.hcli_tool_specs()
    assert len(specs) == 3
    names = [s["name"] for s in specs]
    assert names == [
        "capability.discover",
        "capability.inspect",
        "capability.invoke",
    ]
    for spec in specs:
        assert spec["schema"] == "hcli.agentos.tool.v1"
        assert spec["mutation"] == "read_only"
        assert spec["provenance"] == rt.MANIFEST_RECORDED_BY
        assert "input_schema" in spec
        assert spec["name"] in rt.HANDLERS


def test_handle_unknown_verb_refuses():
    out = rt.handle("future.capacity_inference_rule", {})
    assert out["ok"] is False
    assert out["failure_class"] == "unknown_tool"


def test_import_without_call_is_not_a_call_site():
    import_only = (
        "from tools.future.capacity_inference_rule import fires_on\n"
        "alias = fires_on\n"
    )
    called = rt.adapter_called_symbols(import_only)
    assert ("tools.future.capacity_inference_rule", "fires_on") not in called
    with_call = import_only + "fires_on([], semantics_comparable=True)\n"
    called2 = rt.adapter_called_symbols(with_call)
    assert ("tools.future.capacity_inference_rule", "fires_on") in called2


def test_adapter_source_actually_calls_the_three_wired_symbols():
    called = rt.adapter_called_symbols()
    for cap_id, spec in rt.WIRED.items():
        assert (spec["dotted"], spec["symbol"]) in called, cap_id
        assert rt.wired_status(cap_id) == "CALLABLE"


def test_invoke_capacity_inference_rule():
    out = rt.handle(
        "capability.invoke",
        {
            "id": "future.capacity_inference_rule",
            "arguments": {
                "levels": [
                    {"concurrency": 1, "aggregate_decode_tps": 36.6},
                    {"concurrency": 2, "aggregate_decode_tps": 51.2},
                ],
                "semantics_comparable": True,
            },
        },
    )
    assert out["ok"] is True, out
    assert out["evidence_tier"] == "FUNCTIONAL_SIM"
    result = out["value"]["result"]
    assert result["fired"] is True
    assert result["inference"] == "SINGLE_WORKLOAD_UNDERUTILIZATION"
    assert out["value"]["symbol"] == "fires_on"
    print("INVOKE future.capacity_inference_rule", result)


def test_invoke_fidelity_hierarchy():
    out = rt.handle(
        "capability.invoke",
        {
            "id": "future.fidelity_hierarchy",
            "arguments": {
                "claim_level": "CAPABILITY",
                "measured_level": "LOCAL_FUNCTIONAL_FIDELITY",
            },
        },
    )
    assert out["ok"] is True, out
    result = out["value"]["result"]
    assert result["refusal_is_valid"] is False
    assert "64x error" in result["why"]
    print("INVOKE future.fidelity_hierarchy", result)


def test_invoke_ebpw_categories():
    out = rt.handle(
        "capability.invoke",
        {"id": "future.ebpw_categories", "arguments": {}},
    )
    assert out["ok"] is True, out
    result = out["value"]["result"]
    assert result["can_promote"] is False
    assert "complete_physical_ebpw" in result["reason"]
    print("INVOKE future.ebpw_categories", result)


def test_invoke_unwired_reports_unreachable_with_wake():
    out = rt.handle(
        "capability.invoke",
        {"id": "future.claim_scope", "arguments": {}},
    )
    assert out["ok"] is False
    assert out["failure_class"] == "UNREACHABLE"
    wake = (out.get("value") or {}).get("wake")
    assert isinstance(wake, dict)
    assert wake.get("required_kind") == "call"


def test_every_parked_module_has_a_machine_readable_wake():
    missing = rt.parked_wake_gaps()
    assert missing == [], missing[:20]
    undisposed = rt.undispositioned()
    assert undisposed == [], undisposed[:20]


def test_discover_is_compact_and_enumerates_the_dormant_pile():
    out = rt.handle("capability.discover", {"disposition": "PARKED"})
    assert out["ok"] is True
    value = out["value"]
    assert value["surface_verb_count"] == 3
    assert value["n_parked_missing_wake"] == 0
    assert value["n_undispositioned"] == 0
    assert value["n"] > 0
    ids = {row["id"] for row in value["capabilities"]}
    assert "future.capacity_inference_rule" in ids
    assert "future.fidelity_hierarchy" in ids
    assert "future.ebpw_categories" in ids
    # Compact rows, not full signatures.
    sample = value["capabilities"][0]
    assert "signature" not in sample
    assert "purpose" in sample and "status" in sample


def test_inspect_returns_typed_signature_and_wake():
    out = rt.handle("capability.inspect", {"id": "future.capacity_inference_rule"})
    assert out["ok"] is True
    entry = out["value"]
    assert entry["status"] == "CALLABLE"
    assert entry["signature"]["symbol"] == "fires_on"
    assert entry["signature"]["wired"] is True
    assert entry["wake"]["required_kind"] == "call"
    assert entry["evidence"]["tier"] == "STATIC"


def test_mutation_removing_call_path_reports_unreachable():
    """Remove one capability's real Call; the manifest must flip to UNREACHABLE.

    Restores the file in finally. An import of fires_on is left in place so
    this cannot be satisfied by counting imports as callers.
    """
    original = ADAPTER.read_text(encoding="utf-8")
    assert FIRE_CALL in original
    assert MUTATION_STUB not in original
    before = rt.wired_status("future.capacity_inference_rule", source=original)
    assert before == "CALLABLE"
    mutated = original.replace(FIRE_CALL, MUTATION_STUB, 1)
    assert FIRE_CALL not in mutated
    assert MUTATION_STUB in mutated
    try:
        ADAPTER.write_text(mutated, encoding="utf-8")
        after = rt.wired_status("future.capacity_inference_rule", source=mutated)
        live_after = rt.wired_status("future.capacity_inference_rule")
        print("MUTATION_BEFORE", before)
        print("MUTATION_AFTER", after)
        print("MUTATION_AFTER_LIVE_SCAN", live_after)
        assert after == "UNREACHABLE"
        assert live_after == "UNREACHABLE"
        # Import of fires_on remains; that must not resurrect CALLABLE.
        assert "from tools.future.capacity_inference_rule import fires_on" in mutated
    finally:
        ADAPTER.write_text(original, encoding="utf-8")
    restored = ADAPTER.read_text(encoding="utf-8")
    assert FIRE_CALL in restored
    assert MUTATION_STUB not in restored
    assert rt.wired_status("future.capacity_inference_rule") == "CALLABLE"
    print("MUTATION_RESTORED", rt.wired_status("future.capacity_inference_rule"))


def test_hcli_could_consume_the_surface_by_name():
    """HCLI ToolRegistry.register(ToolSpec(**spec, handler=...)) bind point.

    We do not import hcli (the live daemon owns it). The serialized spec is
    the documented ToolSpec.to_dict() shape; handle() is the handler.
    """
    specs = {s["name"]: s for s in rt.hcli_tool_specs()}
    invoked = rt.handle(
        "capability.invoke",
        {
            "id": "future.fidelity_hierarchy",
            "arguments": {
                "claim_level": "REPRESENTATION_FIDELITY",
                "measured_level": "REPRESENTATION_FIDELITY",
            },
        },
    )
    assert specs["capability.invoke"]["schema"] == "hcli.agentos.tool.v1"
    assert invoked["schema"] == "hcli.agentos.tool.result.v1"
    assert invoked["ok"] is True
    assert invoked["value"]["result"]["refusal_is_valid"] is True


def test_wake_required_kind_is_call_not_import():
    row = {
        "module": "tools/future/widget.py",
        "dotted": "tools.future.widget",
        "callable_outside_tests": False,
        "hcli_reachable": False,
        "is_stub": False,
        "is_package_marker": False,
        "retired": None,
        "has_test": True,
        "test_exercises": True,
        "test_reads_receipt_only": False,
        "orchestration_bound": False,
        "public_functions": [{"name": "gadget", "args": []}],
    }
    rt.decide(row)
    assert row["disposition"] == "PARKED"
    assert row["wake"]["required_kind"] == "call"
    assert row["wake"]["required_symbol"] == "tools.future.widget.gadget"
    hows = " ".join(ch["how"] for ch in row["wake"]["satisfy_by"])
    assert "import is not enough" in hows or "not an Import" in hows

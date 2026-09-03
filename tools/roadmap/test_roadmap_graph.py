"""Graph shape + adversarial auditor tests.

A validator nobody has watched refuse is decoration. The mutation check is the
load-bearing one: a BUILT gate must drop when its production call site is removed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.roadmap import ALLOWED_STATUSES, GRAPH_REL
from tools.roadmap import catalog
from tools.roadmap.auditor import audit, citation_bound_violations
from tools.roadmap.gitfs import REPO, SourceView
from tools.roadmap.parse import parse_roadmap
from tools.roadmap.__main__ import mutation_check

ROADMAP = Path("/Users/scammermike/Downloads/H-ROADMAP.md")


@pytest.fixture(scope="session")
def parsed():
    return parse_roadmap(ROADMAP)


@pytest.fixture(scope="session")
def graph():
    return audit(include_assemble=False)


def test_graph_contains_at_least_71_gates_and_25_genes_with_source_spans(parsed, graph):
    assert len(parsed["gates"]) >= 71
    assert len(parsed["genes"]) >= 25
    assert len(graph["gates"]) >= 71
    assert len(graph["genes"]) >= 25
    for entry in list(graph["gates"].values()) + list(graph["genes"].values()):
        span = entry.get("source_span") or {}
        assert span.get("file"), f"{entry.get('id')} missing source_span.file"
        assert isinstance(span.get("start_line"), int) and span["start_line"] > 0
        assert isinstance(span.get("end_line"), int) and span["end_line"] >= span["start_line"]


def test_auditor_emits_status_for_every_gate_and_gene(graph):
    for name, row in graph["gates"].items():
        assert row["status"] in ALLOWED_STATUSES, f"{name} status {row['status']!r}"
    for name, row in graph["genes"].items():
        assert row["status"] in ALLOWED_STATUSES, f"{name} status {row['status']!r}"
    assert len(graph["gates"]) >= 71
    assert len(graph["genes"]) == 25


def test_every_non_absent_verdict_cites_evidence(graph):
    for name, row in {**graph["gates"], **graph["genes"]}.items():
        if row["status"] == "ABSENT":
            continue
        refs = row.get("evidence_refs") or []
        assert refs, f"{name} status={row['status']} has empty evidence_refs"
        for ref in refs:
            assert ref.get("kind"), f"{name} evidence missing kind"
            assert (
                ref.get("file")
                or ref.get("command")
                or ref.get("note")
            ), f"{name} evidence has no file/command/note"


def test_all_13_blocked_hardware_gates_have_wake_condition(graph):
    blocked = [g for g in graph["gates"].values() if g["status"] == "BLOCKED_HARDWARE"]
    assert len(blocked) == 13, (
        f"expected 13 BLOCKED_HARDWARE gates, got {len(blocked)}: "
        + ",".join(sorted(g["id"] for g in blocked))
    )
    for g in blocked:
        wake = g.get("wake_condition")
        assert isinstance(wake, str) and wake.strip(), f"{g['id']} missing wake_condition"
        assert wake == wake.upper() and "_" in wake, f"{g['id']} wake {wake!r} is not a machine id"


def test_built_gates_have_a_non_test_call_site(graph):
    built = [g for g in graph["gates"].values() if g["status"] == "BUILT"]
    # Zero BUILT is a legal outcome: wired is not accepted.
    for g in built:
        callers = g.get("runtime_caller") or []
        assert callers, f"{g['id']} is BUILT with no runtime_caller"
        kinds = {site.get("kind") for site in callers}
        assert kinds & {"call", "subprocess"}, (
            f"{g['id']} is BUILT but runtime_caller has no call/subprocess: {kinds}"
        )
        assert not kinds <= {"import"}, (
            f"{g['id']} is BUILT on exclusively import-kind runtime_caller"
        )
        assert g.get("wired", {}).get("value") is True, f"{g['id']} BUILT but wired is not true"
        assert g.get("accepted", {}).get("value") is True, f"{g['id']} BUILT but accepted is not true"
        for site in callers:
            from tools.roadmap.reach import is_test_path

            assert not is_test_path(site["file"]), f"{g['id']} caller {site['file']} is a test"
            assert site.get("kind") != "import", (
                f"{g['id']} runtime_caller includes kind=import at {site['file']}:{site['line']}"
            )
            assert site.get("kind") != "weak_signal", (
                f"{g['id']} runtime_caller includes a weak_signal at {site['file']}:{site['line']}"
            )


def test_no_built_gate_is_import_only(graph):
    for g in graph["gates"].values():
        if g["status"] != "BUILT":
            continue
        callers = g.get("runtime_caller") or []
        kinds = [c.get("kind") for c in callers]
        assert kinds, f"{g['id']} BUILT with empty runtime_caller"
        assert any(k in {"call", "subprocess"} for k in kinds), (
            f"{g['id']} BUILT without a call/subprocess citation: {kinds}"
        )
        assert not all(k == "import" for k in kinds), (
            f"{g['id']} BUILT with exclusively import-kind runtime_caller"
        )


def test_runtime_caller_contains_only_invocations(graph):
    allowed = {"call", "subprocess"}
    for g in graph["gates"].values():
        for site in g.get("runtime_caller") or []:
            assert site.get("kind") in allowed, (
                f"{g['id']} runtime_caller has non-invocation kind {site}"
            )


def test_no_two_built_gates_share_identical_runtime_caller(graph):
    built = [g for g in graph["gates"].values() if g["status"] == "BUILT"]
    groups: dict[str, list[str]] = {}
    for g in built:
        key = json.dumps(g.get("runtime_caller") or [], sort_keys=True)
        groups.setdefault(key, []).append(g["id"])
    collisions = {tuple(v) for v in groups.values() if len(v) > 1}
    assert not collisions, (
        "BUILT gates share a byte-identical runtime_caller list (import-as-call "
        f"regression): {sorted(collisions)}"
    )


def test_graph_keeps_71_gates_25_genes_with_source_spans(graph):
    """Every APPENDIX O gate survives, and every extra gate declares its source.

    This asserted len == 71 while APPENDIX O of the superseded roadmap was the
    ONLY roster source. It no longer is: a capability the machine has but that
    document never listed would otherwise be invisible to every audit, and the
    superseded document is preserved lineage that must not be edited to add one.
    A hardcoded count would break again on the next legitimate addition, so the
    invariant is now PROVENANCE, which is what the count was standing in for:
    nothing from APPENDIX O may vanish, and nothing may appear without saying
    where it came from and why.
    """
    # Key on the explicit field, NOT on the note text: the supplement's note says
    # "NOT an APPENDIX O ledger row", so a substring match matched the negation.
    from_appendix = [
        g for g in graph["gates"].values() if g.get("roster_source") != "supplement"
    ]
    assert len(from_appendix) >= 71, "an APPENDIX O gate disappeared from the roster"
    supplement = [
        g for g in graph["gates"].values()
        if g.get("roster_source") == "supplement"
    ]
    assert len(from_appendix) + len(supplement) == len(graph["gates"]), (
        "a gate exists with neither an APPENDIX O row nor a supplement declaration"
    )
    for entry in supplement:
        assert entry.get("declared_because"), (
            f"{entry.get('id')} was added to the roster without a stated reason"
        )
    assert len(graph["genes"]) == 25
    for entry in list(graph["gates"].values()) + list(graph["genes"].values()):
        span = entry.get("source_span") or {}
        assert span.get("file"), f"{entry.get('id')} missing source_span.file"
        assert isinstance(span.get("start_line"), int) and span["start_line"] > 0
        assert isinstance(span.get("end_line"), int) and span["end_line"] >= span["start_line"]


def test_mutation_downgrades_a_built_gate(graph):
    result = mutation_check(before=graph)
    assert result["before_status"] in {"BUILT", "WIRED"}
    assert result["after_status"] not in {"BUILT", "WIRED"}, (
        f"{result['gate']} stayed {result['after_status']} after overlaying "
        f"{result['mutated_files']}; the auditor is not adversarial"
    )
    assert result["downgraded"] is True
    assert result["after_status"] in ALLOWED_STATUSES
    # Leave the result on stdout so the required evidence paste is a real run.
    print("MUTATION_BEFORE", json.dumps({
        "gate": result["gate"],
        "status": result["before_status"],
        "callers": result["before_callers"],
        "counts": result["before_counts"],
    }, indent=2))
    print("MUTATION_AFTER", json.dumps({
        "gate": result["gate"],
        "status": result["after_status"],
        "callers": result["after_callers"],
        "counts": result["after_counts"],
        "mutated_files": result["mutated_files"],
    }, indent=2))


def test_disk_truth_modules_are_present_in_git(graph):
    rows = {r["path"]: r for r in graph["disk_truth_modules"]}
    for path in catalog.DISK_TRUTH_MODULES:
        assert rows[path]["present_in_git"] is True, f"disk-truth module missing from git: {path}"


def test_theia_engine_is_present_and_its_model_ladder_is_blocked_not_absent(graph):
    """Theia WAS absent. The bounty engine now exists; the model ladder does not.

    Those are different facts and the graph must not collapse them. A trained
    Theia model cannot be produced in this checkout, but the blocker and the
    wake condition are both known, so the ladder is BLOCKED_EXTERNAL --
    never ABSENT, which would mean nothing is known about it at all.
    """
    claims = {c["claim"]: c for c in graph["verified_absent"]}
    assert claims["theia"]["verdict"] == "PRESENT"
    assert any(p.startswith("tools/theia/") for p in claims["theia"]["hawking_paths"])

    gates = graph["gates"]
    items = gates.items() if isinstance(gates, dict) else ((g["id"], g) for g in gates)
    ladder = {k: v for k, v in items if k.startswith("THEIA_")}
    assert ladder, "the THEIA gates disappeared from the catalog"
    for name, entry in ladder.items():
        assert entry["status"] == "BLOCKED_EXTERNAL", (name, entry["status"])
        assert entry.get("software_blocker"), f"{name} is blocked with no stated blocker"


def test_catalog_covers_every_appendix_o_gate(parsed):
    missing = [n for n in parsed["gates"] if n not in catalog.GATES]
    extra = [n for n in catalog.GATES if n not in parsed["gates"]]
    assert missing == [], f"catalog missing gates {missing}"
    assert extra == [], f"catalog has unknown gates {extra}"


def test_capability_reachability_assemble_is_importable():
    from tools.future.capability_reachability import assemble, build_repo_index, find_module_import_sites

    assert callable(assemble)
    assert callable(build_repo_index)
    assert callable(find_module_import_sites)


def test_no_status_is_hand_written_in_the_catalog():
    blob = Path(__file__).with_name("catalog.py").read_text()
    for status in ALLOWED_STATUSES:
        assert f'"{status}"' not in blob, f"catalog.py contains a hand-written status {status}"
        assert f"'{status}'" not in blob


def test_import_alone_cannot_justify_built():
    from tools.roadmap.auditor import _local_status

    look = {
        "defined": True,
        "defined_refs": [{"file": "hcli/scheduler.py", "line": 1, "kind": "definition"}],
        "missing_paths": [],
        "runtime_caller": [],
        "import_sites": [
            {"file": "hcli/agentos/__init__.py", "line": 17, "kind": "import"},
            {"file": "hcli/mission.py", "line": 29, "kind": "import"},
        ],
        "weak_signals": [],
        "tests": [],
        "receipts": [],
    }
    status, evidence, _hw, _sw = _local_status(
        era="I", look=look, hw_id=None, hw_probe=None, ext=None
    )
    assert status == "SCAFFOLDED", status
    assert all(e.get("kind") != "call" for e in evidence if e.get("kind") == "import")
    assert any(e.get("kind") == "import" for e in evidence)


def test_call_of_implementing_symbol_justifies_wired_not_built():
    from tools.roadmap.auditor import _local_status

    look = {
        "defined": True,
        "defined_refs": [
            {"file": "hcli/scheduler.py", "line": 72, "kind": "symbol", "note": "Scheduler"}
        ],
        "missing_paths": [],
        "runtime_caller": [
            {"file": "hcli/mission.py", "line": 245, "kind": "call", "symbol": "Scheduler"}
        ],
        "import_sites": [
            {"file": "hcli/mission.py", "line": 29, "kind": "import"},
        ],
        "weak_signals": [],
        "tests": [],
        "receipts": [],
    }
    wired_evidence = [
        {"file": "hcli/mission.py", "line": 245, "kind": "call", "note": "Scheduler"}
    ]
    status, evidence, _hw, _sw = _local_status(
        era="I",
        look=look,
        hw_id=None,
        hw_probe=None,
        ext=None,
        wired=True,
        accepted=False,
        wired_evidence=wired_evidence,
    )
    assert status == "WIRED", status
    assert any(e.get("kind") == "call" for e in evidence)


def test_wired_and_accepted_together_justify_built():
    from tools.roadmap.auditor import _local_status

    look = {
        "defined": True,
        "defined_refs": [
            {"file": "hcli/scheduler.py", "line": 72, "kind": "symbol", "note": "Scheduler"}
        ],
        "missing_paths": [],
        "runtime_caller": [
            {"file": "hcli/mission.py", "line": 245, "kind": "call", "symbol": "Scheduler"}
        ],
        "import_sites": [],
        "weak_signals": [],
        "tests": [],
        "receipts": [],
    }
    status, _evidence, _hw, _sw = _local_status(
        era="I",
        look=look,
        hw_id=None,
        hw_probe=None,
        ext=None,
        wired=True,
        accepted=True,
        wired_evidence=[
            {"file": "hcli/mission.py", "line": 245, "kind": "call", "note": "Scheduler"}
        ],
        accepted_evidence=[
            {
                "kind": "numeric_acceptance",
                "file": "receipts/future/COMPLETE_EBPW.json",
                "measured": 0.5,
                "op": "<=",
                "threshold": 1,
                "note": "measured 0.5 against required <= 1",
            }
        ],
    )
    assert status == "BUILT", status


def test_weak_signal_never_moves_status():
    from tools.roadmap.auditor import _local_status

    look = {
        "defined": True,
        "defined_refs": [{"file": "hcli/scheduler.py", "line": 1, "kind": "definition"}],
        "missing_paths": [],
        "runtime_caller": [],
        "import_sites": [],
        "weak_signals": [
            {
                "file": "hcli/scheduler.py",
                "line": 55,
                "kind": "weak_signal",
                "symbol": "NO_PROGRESS",
                "note": "name-only assignment",
            }
        ],
        "tests": [],
        "receipts": [],
    }
    status, _evidence, _hw, _sw = _local_status(
        era="I", look=look, hw_id=None, hw_probe=None, ext=None
    )
    assert status == "SCAFFOLDED", status


def test_classify_symbol_rejects_assignments():
    from tools.roadmap.gitfs import classify_symbol, definition_line

    text = "NO_PROGRESS = 3\n\ndef verification_passed(outcome):\n    return True\n"
    kind, line = classify_symbol(text, "NO_PROGRESS")
    assert kind == "assignment"
    assert line == 1
    assert definition_line(text, "NO_PROGRESS") is None
    kind, line = classify_symbol(text, "verification_passed")
    assert kind == "function"
    assert definition_line(text, "verification_passed") == line


def test_classify_symbol_accepts_exception_class():
    from tools.roadmap.gitfs import classify_symbol, definition_line

    text = "class NO_PROGRESS(Exception):\n    pass\n"
    kind, line = classify_symbol(text, "NO_PROGRESS")
    assert kind == "class"
    assert definition_line(text, "NO_PROGRESS") == line


def test_exact_cli_path_rejects_suffix_of_another_tree():
    from tools.roadmap.reach import is_exact_cli_path

    assert is_exact_cli_path("hcli/scheduler.py", "hcli/scheduler.py")
    assert is_exact_cli_path("hcli/scheduler.py", "./hcli/scheduler.py")
    # A DIFFERENT tree whose tail happens to match. `tools/haider/hcli/` was the
    # original example and no longer exists; `lab/hcli/` is a real separate
    # product in this repo and makes the same point.
    assert not is_exact_cli_path(
        "hcli/scheduler.py", "lab/hcli/scheduler.py"
    )
    assert not is_exact_cli_path("hcli/scheduler.py", "MAX_REPAIR_DEPTH")


def test_every_gate_carries_orthogonal_wired_and_accepted(graph):
    for name, row in {**graph["gates"], **graph["genes"]}.items():
        wired = row.get("wired")
        accepted = row.get("accepted")
        assert isinstance(wired, dict), f"{name} missing wired fact"
        assert isinstance(accepted, dict), f"{name} missing accepted fact"
        assert isinstance(wired.get("value"), bool), f"{name} wired.value is not bool"
        assert isinstance(accepted.get("value"), bool), f"{name} accepted.value is not bool"
        assert wired.get("evidence"), f"{name} wired has empty evidence"
        assert accepted.get("evidence"), f"{name} accepted has empty evidence"


def test_no_gate_is_built_on_wired_alone(graph):
    """Load-bearing: wired without accepted must not produce BUILT.

    Mutating _local_status so a caller returns BUILT must fail this test.
    """
    for name, row in graph["gates"].items():
        wired = (row.get("wired") or {}).get("value")
        accepted = (row.get("accepted") or {}).get("value")
        if row["status"] == "BUILT":
            assert wired is True, f"{name} is BUILT but wired is {wired}"
            assert accepted is True, f"{name} is BUILT on wired alone (accepted={accepted})"
        if wired and not accepted:
            assert row["status"] != "BUILT", (
                f"{name} is BUILT on wired alone; accepted evidence="
                f"{(row.get('accepted') or {}).get('evidence')}"
            )
            if row["status"] not in {
                "BLOCKED_HARDWARE",
                "UNREACHABLE",
                "BLOCKED_EXTERNAL",
            }:
                assert row["status"] == "WIRED", (name, row["status"])


def test_flash_complete_ebpw_le_1_is_not_built_and_cites_measured_value(graph):
    g = graph["gates"]["FLASH_COMPLETE_EBPW_LE_1"]
    assert g["status"] != "BUILT", g["status"]
    assert g["accepted"]["value"] is False
    blob = json.dumps(g)
    assert "3.139" in blob, blob[:2000]
    notes = " ".join(
        str(e.get("note"))
        for e in list(g.get("evidence_refs") or [])
        + list((g.get("accepted") or {}).get("evidence") or [])
    )
    assert "3.139" in notes, notes
    assert "<= 1" in notes or "<=1" in notes, notes
    measured = None
    for e in (g.get("accepted") or {}).get("evidence") or []:
        if e.get("kind") == "numeric_acceptance":
            measured = e.get("measured")
            assert e.get("op") == "<="
            assert e.get("threshold") == 1
    assert measured is not None
    assert abs(float(measured) - 3.139300850311054) < 1e-9


def test_no_citation_line_exceeds_file_length_at_emitting_commit(graph):
    commit = graph.get("generated_from_commit")
    assert commit, "graph missing generated_from_commit"
    violations = citation_bound_violations(graph)
    assert violations == [], "out-of-bounds citations:\n" + "\n".join(violations)
    # Every repo-relative file:line citation is bound to that commit.
    for name, row in {**graph["gates"], **graph["genes"]}.items():
        for ref in list(row.get("evidence_refs") or []) + list(row.get("runtime_caller") or []):
            rel = ref.get("file")
            line = ref.get("line")
            if not rel or str(rel).startswith("/") or not isinstance(line, int):
                continue
            assert ref.get("commit") == commit, (
                f"{name} citation {rel}:{line} commit {ref.get('commit')!r} != {commit}"
            )


def test_numeric_acceptance_compares_ebpw_receipt_against_the_bar():
    from tools.roadmap.auditor import _numeric_acceptance
    from tools.roadmap.gitfs import SourceView

    spec = {
        "kind": "numeric",
        "receipt": "receipts/future/COMPLETE_EBPW.json",
        "field": "incumbent.complete_ebpw",
        "op": "<=",
        "threshold": 1,
    }
    fact = _numeric_acceptance(spec, SourceView())
    assert fact["value"] is False
    ev = fact["evidence"][0]
    assert ev["measured"] == pytest.approx(3.139300850311054)
    assert "3.139" in ev["note"]
    assert "<= 1" in ev["note"]


def test_accepted_without_wired_is_not_built():
    from tools.roadmap.auditor import _local_status

    look = {
        "defined": True,
        "defined_refs": [{"file": "hcli/scheduler.py", "line": 1, "kind": "definition"}],
        "missing_paths": [],
        "runtime_caller": [],
        "import_sites": [],
        "weak_signals": [],
        "tests": [],
        "receipts": ["receipts/future/COMPLETE_EBPW.json"],
    }
    status, _evidence, _hw, _sw = _local_status(
        era="I",
        look=look,
        hw_id=None,
        hw_probe=None,
        ext=None,
        wired=False,
        accepted=True,
        accepted_evidence=[{"kind": "numeric_acceptance", "note": "bar met"}],
    )
    assert status == "SCAFFOLDED", status
    assert status != "BUILT"

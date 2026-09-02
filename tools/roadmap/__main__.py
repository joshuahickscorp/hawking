"""CLI: python3 -m tools.roadmap --build|--audit|--mutation-check"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.roadmap import ALLOWED_STATUSES, GRAPH_REL
from tools.roadmap.auditor import audit, write_graph
from tools.roadmap.gitfs import REPO, SourceView


def _print_counts(doc: dict) -> None:
    counts = doc["counts"]
    print(f"gates={counts['gates']} genes={counts['genes']}")
    print("gates_by_status", json.dumps(counts["gates_by_status"], sort_keys=True))
    print("genes_by_status", json.dumps(counts["genes_by_status"], sort_keys=True))
    blocked = [g for g in doc["gates"].values() if g["status"] == "BLOCKED_HARDWARE"]
    print(f"BLOCKED_HARDWARE={len(blocked)}")
    for g in blocked:
        print(f"  {g['id']} wake={g.get('wake_condition')}")
    built = [g for g in doc["gates"].values() if g["status"] == "BUILT"]
    print(f"BUILT={len(built)}")
    for g in built:
        callers = g.get("runtime_caller") or []
        site = f"{callers[0]['file']}:{callers[0]['line']}" if callers else "(none)"
        print(f"  {g['id']} caller={site}")
    wired = [g for g in doc["gates"].values() if g["status"] == "WIRED"]
    print(f"WIRED={len(wired)}")
    for g in wired:
        callers = g.get("runtime_caller") or []
        site = f"{callers[0]['file']}:{callers[0]['line']}" if callers else "(none)"
        print(f"  {g['id']} caller={site}")


def _strip_uses(text: str, needles: list[str]) -> str:
    """Rename/remove lines that mention any needle so they stop being import/call sites."""
    out = []
    for line in text.splitlines(keepends=True):
        if any(n and n in line for n in needles):
            # Keep the line shape so diffs are readable, but destroy the identifier.
            mutated = line
            for n in needles:
                if n:
                    mutated = mutated.replace(n, n + "_MUTATED_AWAY")
            if mutated.endswith("\n"):
                out.append("# MUTATION " + mutated)
            else:
                out.append("# MUTATION " + mutated + "\n")
        else:
            out.append(line)
    return "".join(out)


def mutation_check(
    prefer: str = "FLASH_COMPLETE_EBPW_LE_1",
    before: dict | None = None,
) -> dict:
    """Remove every production call site of a WIRED/BUILT gate in an overlay and re-audit.

    `before` is the unmutated audit. Tests share the session graph for this
    first pass; the overlay audit still runs for real.
    """
    if before is None:
        before = audit(view=SourceView(), include_assemble=False)
    gates = before["gates"]
    wired_statuses = {"BUILT", "WIRED"}
    target = None
    if prefer in gates and gates[prefer]["status"] in wired_statuses and gates[prefer]["runtime_caller"]:
        target = gates[prefer]
    if target is None:
        for g in gates.values():
            if g["status"] in wired_statuses and g.get("runtime_caller"):
                target = g
                break
    if target is None:
        raise SystemExit("no WIRED/BUILT gate with a production caller; auditor is not ready for mutation")

    needles: list[str] = []
    from tools.roadmap.catalog import GATES

    probe = GATES[target["id"]]
    # Destroy the implementing symbol at its real call site. Module-name
    # rewrites are not enough: an import is not what made this BUILT.
    for spec in probe.get("symbols") or []:
        if spec.get("symbol"):
            needles.append(spec["symbol"])
    for site in target.get("runtime_caller") or []:
        if site.get("symbol"):
            needles.append(str(site["symbol"]))
    for ref in target.get("code_refs") or []:
        if ref.get("note"):
            needles.append(str(ref["note"]))
        if ref.get("file"):
            needles.append(ref["file"])
            needles.append(Path(ref["file"]).stem)
    needles.extend(probe.get("modules") or [])
    needles.extend(probe.get("code_paths") or [])
    needles = [n for n in dict.fromkeys(needles) if n]

    overlay = SourceView()
    mutated_files = []
    for site in target["runtime_caller"]:
        rel = site["file"]
        original = overlay.read(rel)
        mutated = _strip_uses(original, needles)
        if mutated == original:
            continue
        overlay.overlay[rel] = mutated
        if rel not in mutated_files:
            mutated_files.append(rel)

    if not mutated_files:
        raise SystemExit(f"could not overlay any caller of {target['id']}")

    after = audit(view=overlay, include_assemble=False)
    after_row = after["gates"][target["id"]]
    return {
        "gate": target["id"],
        "before_status": target["status"],
        "after_status": after_row["status"],
        "before_callers": target["runtime_caller"],
        "after_callers": after_row.get("runtime_caller") or [],
        "mutated_files": mutated_files,
        "downgraded": after_row["status"] not in wired_statuses and after_row["status"] in ALLOWED_STATUSES,
        "before_counts": before["counts"]["gates_by_status"],
        "after_counts": after["counts"]["gates_by_status"],
    }


_COMPARE_KEYS = (
    "status",
    "evidence_refs",
    "runtime_caller",
    "import_sites",
    "weak_signals",
    "tests",
    "code_refs",
    "receipts_cited",
)


def _gate_slice(entry: dict) -> dict:
    return {k: entry.get(k) for k in _COMPARE_KEYS}


def parity_check() -> dict:
    """AST path vs hawking-index path. Exact equality of 71 statuses + citations."""
    import os

    from tools.roadmap.gitfs import SourceView as _View

    orig = os.environ.get("ROADMAP_REACH_BACKEND")
    try:
        os.environ["ROADMAP_REACH_BACKEND"] = "ast"
        ast_doc = audit(view=_View(), include_assemble=False)
        os.environ["ROADMAP_REACH_BACKEND"] = "index"
        idx_doc = audit(view=_View(), include_assemble=False)
    finally:
        if orig is None:
            os.environ.pop("ROADMAP_REACH_BACKEND", None)
        else:
            os.environ["ROADMAP_REACH_BACKEND"] = orig

    mismatches: list[dict] = []
    ast_counts = ast_doc["counts"]["gates_by_status"]
    idx_counts = idx_doc["counts"]["gates_by_status"]
    names = sorted(set(ast_doc["gates"]) | set(idx_doc["gates"]))
    if len(names) != 71:
        mismatches.append(
            {
                "gate": "<count>",
                "field": "n_gates",
                "ast": len(ast_doc["gates"]),
                "index": len(idx_doc["gates"]),
            }
        )
    for name in names:
        a = ast_doc["gates"].get(name)
        b = idx_doc["gates"].get(name)
        if a is None or b is None:
            mismatches.append({"gate": name, "field": "missing", "ast": a is not None, "index": b is not None})
            continue
        sa, sb = _gate_slice(a), _gate_slice(b)
        if sa == sb:
            continue
        for key in _COMPARE_KEYS:
            if sa.get(key) != sb.get(key):
                mismatches.append(
                    {
                        "gate": name,
                        "field": key,
                        "ast": sa.get(key),
                        "index": sb.get(key),
                    }
                )
    status_ok = ast_counts == idx_counts and not any(
        m.get("field") == "status" for m in mismatches
    )
    return {
        "identical": not mismatches,
        "status_counts_identical": ast_counts == idx_counts,
        "n_gates": len(names),
        "ast_counts": ast_counts,
        "index_counts": idx_counts,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:40],
        "ast_index": ast_doc.get("index"),
        "index_index": idx_doc.get("index"),
        "status_ok": status_ok,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Roadmap IR + adversarial auditor")
    ap.add_argument("--build", action="store_true", help="write civilization/CAPABILITY_GRAPH.json")
    ap.add_argument("--audit", action="store_true", help="run auditor and print counts")
    ap.add_argument("--mutation-check", action="store_true", help="downgrade a BUILT gate by overlaying its callers")
    ap.add_argument("--assemble", action="store_true", help="also reuse capability_reachability.assemble()")
    ap.add_argument(
        "--parity",
        action="store_true",
        help="compare AST path vs hawking-index path on all 71 gates (statuses + evidence)",
    )
    args = ap.parse_args(argv)

    if args.parity:
        result = parity_check()
        print(json.dumps(result, indent=2))
        return 0 if result.get("identical") else 1

    if args.mutation_check:
        result = mutation_check()
        print(json.dumps(result, indent=2))
        if not result["downgraded"]:
            print("MUTATION DID NOT DOWNGRADE", file=sys.stderr)
            return 1
        print("MUTATION DOWNGRADED", result["gate"], result["before_status"], "->", result["after_status"])
        return 0

    if not (args.build or args.audit):
        args.audit = True

    doc = audit(include_assemble=args.assemble)
    _print_counts(doc)
    from tools.lifecycle_events import on_hardware_profile_changed
    hp = on_hardware_profile_changed(doc)
    print(
        f"hardware_profile_changed blocked={hp['n_blocked_hardware']} "
        f"activable={len(hp['activable'])} wake_ids={hp['wake_ids']}"
    )
    if args.build:
        path = write_graph(doc)
        print("wrote", path.relative_to(REPO) if path.is_relative_to(REPO) else path)
        # Keep the pointer in ROADMAP_STATE.json in sync without regenerating the ledger.
        state_path = REPO / "civilization" / "ROADMAP_STATE.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            law = (
                "BUILT requires wired AND accepted. wired is a non-test call of the "
                "implementing symbol. accepted is the gate's own acceptance criterion "
                "demonstrably met by a receipt or measurement that meets the stated bar, "
                "not merely a receipt on the topic. wired alone is WIRED, never BUILT."
            )
            dirty = False
            if state.get("capability_graph") != GRAPH_REL:
                state["capability_graph"] = GRAPH_REL
                dirty = True
            if state.get("capability_graph_schema") != doc["schema"]:
                state["capability_graph_schema"] = doc["schema"]
                dirty = True
            if state.get("capability_graph_law") != law:
                state["capability_graph_law"] = law
                dirty = True
            if dirty:
                state_path.write_text(json.dumps(state, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

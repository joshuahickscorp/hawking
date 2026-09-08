"""G025 verify: the graph must reproduce the defects found the hard way.

A reachability audit that only describes the CURRENT wiring is worthless -- the
wiring is currently fine, which is why every check would pass. The test is
whether the graph would have SAID SO before Round 19 spent 810 s discovering it.
So each case reverts one historical fix in a scratch copy of the registry and
asserts the graph names the defect.
"""
import ast
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _graph_from(registry_src: str):
    """Build the graph against a modified copy of tool_registry.py."""
    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "hawking"
        (fake / "hcli").mkdir(parents=True)
        (fake / "tools" / "future").mkdir(parents=True)
        (fake / "receipts" / "future").mkdir(parents=True)
        (fake / "hcli" / "tool_registry.py").write_text(registry_src)
        shutil.copy(REPO / "hcli" / "engine.py", fake / "hcli" / "engine.py")
        for extra in ("__init__.py",):
            srcf = REPO / "hcli" / extra
            if srcf.exists():
                shutil.copy(srcf, fake / "hcli" / extra)
        mod = REPO / "tools" / "future" / "hcli_reachability.py"
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys,json,pathlib; sys.path.insert(0,%r); "
             "from hcli_reachability import build; "
             "print(json.dumps(build(pathlib.Path(%r))))"
             % (str(mod.parent), str(fake))],
            capture_output=True, text=True, check=True)
        import json
        return json.loads(out.stdout)


def _drop_toolspec(src: str, name: str) -> str:
    """Remove one registry.register(ToolSpec(...)) call by AST span."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "ToolSpec" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == name):
            # walk out to the enclosing statement
            for stmt in ast.walk(tree):
                if isinstance(stmt, ast.Expr) and node.lineno >= stmt.lineno \
                        and node.end_lineno <= stmt.end_lineno:
                    blanked = list(lines)
                    for i in range(stmt.lineno - 1, stmt.end_lineno):
                        blanked[i] = "\n"
                    return "".join(blanked)
    raise AssertionError(f"ToolSpec {name!r} not found -- test is stale")


def main() -> int:
    src = (REPO / "hcli" / "tool_registry.py").read_text()
    fails = []

    # Baseline: the wiring is currently sound. If this fires, the graph is
    # reporting a defect that does not exist and every case below is worthless.
    base = _graph_from(src)
    if base["findings"]["OUTPUT_THAT_CANNOT_BE_RECORDED"]:
        fails.append("baseline: graph reports a missing write door on sound wiring")
    if not base["ledger_doors"]:
        fails.append("baseline: no ledger door found on sound wiring")
    if not base["measurers"]:
        fails.append("baseline: no measuring tool found -- the check cannot fire either way")

    # CASE 1 -- Round 19: "the round reached the measurement and there was no
    # door to make it". Remove odyssey.record_measurement.
    g1 = _graph_from(_drop_toolspec(src, "odyssey.record_measurement"))
    hits = g1["findings"]["OUTPUT_THAT_CANNOT_BE_RECORDED"]
    if not hits:
        fails.append("CASE 1: removed the ledger door and the graph did NOT flag it")
    elif "odyssey.ledger" not in {h["tool"] for h in hits}:
        fails.append(f"CASE 1: flagged, but not on odyssey.ledger: {[h['tool'] for h in hits]}")
    # and it must not be fooled by filesystem.write still being present
    elif not hits[0].get("generic_write_doors_that_do_not_count"):
        fails.append("CASE 1: did not record that generic write doors were present and insufficient")

    # CASE 2 -- Round 15: odyssey.dense_anatomy required `snapshot`, the round
    # had no tool that produced a directory, and it passed the ledger JSON path
    # instead. The anatomy was never measured and 5 axes stayed OWED. Remove the
    # tool that supplies `snapshot` and the graph must say so.
    g2 = _graph_from(_drop_toolspec(src, "odyssey.ledger"))
    no_src = {(f["tool"], f["arg"]) for f in g2["findings"]["REQUIRED_ARGUMENT_WITH_NO_SOURCE"]}
    if ("odyssey.dense_anatomy", "snapshot") not in no_src:
        fails.append("CASE 2: removed the only producer of `snapshot` and the graph still "
                     f"thought dense_anatomy could source it (unsourced: {sorted(no_src)[:5]})")

    # CASE 3 -- the graph must not call a buried-but-reachable argument missing.
    # `slug` lives inside a ledger row; that is reachable, not absent.
    if ("odyssey.record_measurement", "slug") in {
            (f["tool"], f["arg"]) for f in base["findings"]["REQUIRED_ARGUMENT_WITH_NO_SOURCE"]}:
        fails.append("CASE 3: `slug` reported as having NO source when ledger rows carry it")

    for f in fails:
        print("FAIL:", f)
    print(f"{'FAILED' if fails else 'PASS'} — 3 historical defects, "
          f"{len(fails)} unreproduced")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

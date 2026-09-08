"""Does the actionable part survive the budget the ENGINE actually applies? (G019)

G019's verify: "for each such tool, a test that truncates its output to a small
budget and asserts the actionable part SURVIVES."

The earlier sweep tested DISCLOSURE -- whether a caller can tell a complete
answer from a cut one -- and reported that most tools cannot be budget-squeezed
because only a handful accept a `limit` argument. That was looking in the wrong
place. The squeeze is not an argument. It is
Engine._compact_closed_observations, which cuts every observation to
CLOSED_OBSERVATION_CHARS = 500 as HEAD 250 + marker + TAIL ~208, eliding the
middle. That is the budget a round's evidence actually passes through, and it
applies to every tool whether or not it takes a limit.

So: serialize each tool's real result the way the registry does, put it through
that exact compaction, and ask whether what survives still answers "what is
here, how much of it, and is this all of it".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hcli.tool_registry import default_tool_registry  # noqa: E402
from hcli.engine import Engine  # noqa: E402

LIMIT = int(Engine.CLOSED_OBSERVATION_CHARS)
SKIP_PREFIX = ("web.", "github.", "huggingface.", "grok.", "benchmark", "accelerator")

# What a caller needs to still see after the cut.
SCOPE_FIELDS = ("n", "total", "shown", "truncated", "truncation_note", "n_owed",
                "n_processes", "n_orphaned", "n_ranked", "n_specimens",
                "specimen_count", "count", "bytes", "delivered_chars")

# Read-only tools that take arguments. "for each such tool" means these too, and
# excluding them because they need an argument was how the first version of this
# sweep covered 28 of 72 and called it done. Arguments are chosen to be real and
# harmless: a file that exists, a pattern that matches, a catalog focus.
ARGUMENTED = {
    "fs.read": {"path": "receipts/future/G034_ODYSSEY_LEDGER.json"},
    "receipt.read": {"path": "receipts/future/G012_CAPABILITY_CLIFF.json"},
    "architecture.inspect": {"path": "receipts/future/modellake-index/catalog.json"},
    "fs.search": {"pattern": "def ", "root": "tools/future", "max_results": 50},
    "tools.catalog": {"focus": "odyssey"},
    "context.recall": {"focus": "odyssey"},
}


def compact(text: str, limit: int = LIMIT) -> str:
    """Engine._compact_closed_observations, verbatim in shape."""
    if len(text) <= limit:
        return text
    marker = "\n[... closed-turn observation elided ...]\n"
    room = max(0, limit - len(marker))
    head = room // 2
    tail = room - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def main() -> int:
    reg = default_tool_registry(str(REPO), repo_root=str(REPO))
    specs = {d["name"]: d for d in reg.discover()}
    targets = sorted(
        n for n, d in specs.items()
        if (d.get("mutation") or "read_only") == "read_only"
        and not any(p in n for p in SKIP_PREFIX)
        and (not (d.get("input_schema") or {}).get("required") or n in ARGUMENTED)
    )

    fails, rows = [], []
    for name in targets:
        props = ((specs[name].get("input_schema") or {}).get("properties") or {})
        args = dict(ARGUMENTED.get(name, {}))
        if "limit" in props and "limit" not in args:
            args["limit"] = 200
        try:
            res = reg.invoke(name, args)
        except Exception as exc:
            rows.append({"tool": name, "verdict": "ERROR", "detail": str(exc)[:100]})
            continue
        if not getattr(res, "ok", True):
            rows.append({"tool": name, "verdict": "REFUSED"})
            continue
        val = res.value if hasattr(res, "value") else res
        if not isinstance(val, dict):
            rows.append({"tool": name, "verdict": "NOT_A_DICT"})
            continue

        full = json.dumps(val, ensure_ascii=False)
        cut = compact(full)
        was_cut = len(full) > LIMIT

        # Which scope fields are still LEGIBLE -- present as a key in the
        # surviving text, not merely present in the object we started with.
        survived = [f for f in SCOPE_FIELDS if f in val and f'"{f}"' in cut]
        present = [f for f in SCOPE_FIELDS if f in val]

        verdict = "OK"
        if not present:
            verdict = "NO_SCOPE_FIELD"
        elif was_cut and not survived:
            verdict = "SCOPE_LOST_TO_THE_CUT"
            fails.append(f"{name}: {len(full)} chars cut to {LIMIT}, and NONE of its scope fields "
                         f"{present} survive -- the round sees rows and cannot tell how many "
                         f"there are or whether it has them all")
        rows.append({"tool": name, "verdict": verdict, "full_chars": len(full),
                     "was_cut": was_cut, "scope_present": present,
                     "scope_survived": survived})

    from collections import Counter
    tally = Counter(r["verdict"] for r in rows)
    for f in fails:
        print("FAIL:", f)
    cut_n = sum(1 for r in rows if r.get("was_cut"))
    print(f"\n{len(rows)} tools, {cut_n} exceed the {LIMIT}-char observation budget")
    for k, v in sorted(tally.items()):
        print(f"  {k:24s} {v}")
    if "--write" in sys.argv:
        out = REPO / "receipts" / "future" / "G019_BUDGET_SURVIVAL.json"
        out.write_text(json.dumps({"limit": LIMIT, "tally": dict(tally),
                                   "failures": fails, "rows": rows}, indent=1) + "\n")
        print("wrote", out.relative_to(REPO))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

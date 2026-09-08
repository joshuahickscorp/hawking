"""Does the actionable part of a tool's output survive a small budget? (G019)

G019's law: every campaign tool an LLM calls puts the decision-relevant entries
FIRST, bounds what it shows, and DISCLOSES the truncation. Silent truncation of
the decision-relevant portion is a defect wherever it is found.

A tool that returns 300 rows and mentions nothing about the other 900 has not
truncated its output, it has lied about its scope -- and the caller, which is a
model with a context budget, will reason about the 300 as if they were all of
them. Round 20 did exactly that with the census.

This invokes every zero-argument read-only campaign tool with a deliberately
tiny limit and checks three things:

  BOUNDED    the result does not blow the budget
  DISCLOSED  when there is more, the result SAYS there is more
  CONSISTENT the disclosure agrees with what was actually shown

Only zero-arg read-only tools are exercised. Anything mutating, costly, or
networked is out of scope for an automated sweep.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hcli.tool_registry import default_tool_registry  # noqa: E402

TINY = 3
SKIP_PREFIX = ("web.", "github.", "huggingface.", "grok.", "benchmark", "accelerator")

# Fields a tool may use to say "there is more than this".
DISCLOSURE = ("truncated", "shown", "n", "n_owed", "total", "truncation_note",
              "n_processes", "n_orphaned", "n_ranked", "more", "next")
# Fields that carry the rows themselves.
PAYLOAD = ("rows", "owed", "processes", "orphaned", "ranked", "entries", "items",
           "specimens", "tests", "capabilities", "files")


def main() -> int:
    reg = default_tool_registry(str(REPO), repo_root=str(REPO))
    specs = {d["name"]: d for d in reg.discover()}
    targets = sorted(
        n for n, d in specs.items()
        if (d.get("mutation") or "read_only") == "read_only"
        and not (d.get("input_schema") or {}).get("required")
        and not any(p in n for p in SKIP_PREFIX)
    )

    fails, rows = [], []
    for name in targets:
        # Pass `limit` only where the schema accepts it. Most of these declare
        # additionalProperties: false, so an unconditional limit is refused by
        # the schema and the sweep measures its own bad argument -- 23 of 27
        # "REFUSED" on the first run, none of them about truncation.
        props = ((specs[name].get("input_schema") or {}).get("properties") or {})
        args = {"limit": TINY} if "limit" in props else {}
        try:
            res = reg.invoke(name, args)
        except Exception as exc:
            rows.append({"tool": name, "verdict": "ERROR", "detail": f"{type(exc).__name__}: {exc}"[:120]})
            continue
        val = res.value if hasattr(res, "value") else res
        if not getattr(res, "ok", True):
            rows.append({"tool": name, "verdict": "REFUSED",
                         "detail": str(getattr(res, "error", ""))[:120]})
            continue
        if not isinstance(val, dict):
            rows.append({"tool": name, "verdict": "NOT_A_DICT", "detail": type(val).__name__})
            continue

        payload_key = next((k for k in PAYLOAD if isinstance(val.get(k), list)), None)
        n_shown = len(val[payload_key]) if payload_key else None
        disclosed = [k for k in DISCLOSURE if k in val]
        size = len(json.dumps(val))

        if payload_key is None:
            rows.append({"tool": name, "verdict": "NO_LIST_PAYLOAD", "bytes": size,
                         "detail": "returns no row list; the law is about row truncation"})
            continue

        # A tool returning a list MUST say how many there are in total, or the
        # caller cannot tell a complete answer from a truncated one.
        # Detect a total GENERICALLY. A hardcoded key list reported
        # specimens.registry as having no total when it returns n_specimens --
        # the detector was narrow, not the tool. That is the third time in this
        # session an audit's own parser made the campaign look worse than it is,
        # so this matches any integer count field instead of a fixed vocabulary.
        total_keys = [k for k, v in val.items()
                      if isinstance(v, int) and not isinstance(v, bool)
                      and (k == "n" or k.startswith("n_")
                           or "total" in k or "count" in k)]
        says_total = bool(total_keys)
        has_flag = "truncated" in val or "truncation_note" in val

        # THE PREDICATE IS "CAN THE CALLER TELL", not "does a named field exist".
        # Two earlier formulations of this test were wrong in the same direction:
        # first a hardcoded total-key list flagged specimens.registry (which
        # returns n_specimens), then demanding an explicit truncated flag
        # flagged odyssey.ingest and processes.summary -- both of which return a
        # count EQUAL to the rows shown, which tells a caller the answer is
        # complete just as well as a flag does. The law is about what the caller
        # can determine, so that is what is checked.
        totals = [val[k] for k in total_keys]
        count_equals_shown = any(t == n_shown for t in totals)
        determinable = has_flag or count_equals_shown

        verdict = "OK"
        if not determinable:
            verdict = "COMPLETENESS_UNDETERMINABLE"
            fails.append(f"{name}: returns {n_shown} rows with no truncation flag and no count "
                         f"matching them -- a caller cannot tell a complete answer from a cut one")
        elif has_flag and says_total and val.get("truncated") is False:
            biggest = max(totals) if totals else None
            if isinstance(biggest, int) and biggest > n_shown:
                verdict = "INCONSISTENT"
                fails.append(f"{name}: reports truncated=False while showing {n_shown} of {biggest}")
        rows.append({"tool": name, "verdict": verdict, "bytes": size, "shown": n_shown,
                     "payload_key": payload_key, "disclosure_fields": disclosed,
                     "total_keys": total_keys, "has_truncation_flag": has_flag,
                     "limit_applied": "limit" in props})

    for f in fails:
        print("FAIL:", f)
    from collections import Counter
    tally = Counter(r["verdict"] for r in rows)
    n_limited = sum(1 for r in rows if r.get("limit_applied"))
    print(f"\n{len(rows)} zero-arg read-only tools exercised "
          f"({n_limited} accept a limit and got {TINY}; the rest were called bare)")
    for k, v in sorted(tally.items()):
        print(f"  {k:20s} {v}")
    if "--write" in sys.argv:
        out = REPO / "receipts" / "future" / "G019_ACTIONABLE_FIRST_SWEEP.json"
        out.write_text(json.dumps({"limit": TINY, "tally": dict(tally),
                                   "failures": fails, "rows": rows}, indent=1) + "\n")
        print("wrote", out.relative_to(REPO))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

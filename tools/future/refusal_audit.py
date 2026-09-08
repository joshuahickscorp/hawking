"""Are the ledger's refusals actually mechanism-backed? (G009 verify)

G009's acceptance is "392 axes each MEASURED or REFUSED-with-mechanism; every
refusal names its reopen condition", and its verify says to check a sampled
refusal for a real mechanism, "not n/a". 114 of 392 cells are REFUSED -- 29% of
the ledger's authority rests on them, and a refusal that names no mechanism is
an OWED cell wearing a resolved label.

Sampling one is not enough to know that. This checks every one, on four tests a
placeholder cannot pass:

  NON_EMPTY        a reason exists at all
  NOT_A_PLACEHOLDER  not n/a, none, unknown, tbd, pending, N/A...
  HAS_A_MECHANISM  says WHY, not just WHAT -- a because-clause, a measured
                   quantity, a named artifact, or a structural fact
  HAS_REOPEN       states what would make it measurable again

The last is the one the campaign keeps needing: a refusal with no reopen
condition is permanent by accident.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "receipts" / "future" / "G034_ODYSSEY_LEDGER.json"

PLACEHOLDER = re.compile(r"^\s*(n/?a|none|unknown|tbd|todo|pending|-+|\?+)\s*\.?\s*$", re.I)

# A mechanism SAYS WHY. Either it reasons, or it points at something measured.
MECHANISM_HINT = re.compile(
    r"\bbecause\b|\bso\b|--|\bmeasures\b|\brequires\b|\bcannot\b|\bwould\b|\bis not\b"
    r"|\bno \w+\b|\bexceeds\b|\bunsupported\b|\bpre-quantized\b|\babsent\b|\bundefined\b"
    r"|\d", re.I)

# A reopen condition says what would CHANGE the answer.
REOPEN_HINT = re.compile(
    r"\breopen\b|\bwould become\b|\bif \b|\bonce \b|\buntil \b|\bwhen \b|\brerun\b"
    r"|\bafter \b|\bunless\b", re.I)


def audit() -> dict:
    led = json.loads(LEDGER.read_text())
    rows, tally = [], Counter()
    for rec in led["specimens"]:
        for axis, cell in rec["axes"].items():
            if cell.get("state") != "REFUSED":
                continue
            reason = (cell.get("reason") or "").strip()
            # A STRUCTURED reopen_when beats a regex over prose. The field did
            # not exist when this audit was written -- it was added because the
            # audit found 121 refusals with no way back -- so the text heuristic
            # stays as a fallback for the cells recorded before it.
            reopen_field = str(cell.get("reopen_when") or "").strip()
            checks = {
                "non_empty": bool(reason),
                "not_a_placeholder": bool(reason) and not PLACEHOLDER.match(reason),
                "has_a_mechanism": bool(reason) and bool(MECHANISM_HINT.search(reason)),
                "has_reopen": bool(reopen_field) or (bool(reason) and bool(REOPEN_HINT.search(reason))),
            }
            verdict = ("STRONG" if all(checks.values())
                       else "NO_REOPEN" if (checks["has_a_mechanism"] and not checks["has_reopen"])
                       else "WEAK")
            tally[verdict] += 1
            rows.append({"slug": rec["slug"], "axis": axis, "verdict": verdict,
                         "checks": checks, "reason": reason[:220],
                         "reopen_when": reopen_field or None,
                         "reopen_is_structured": bool(reopen_field),
                         "has_receipt": bool(cell.get("receipt"))})
    return {"n_refused": len(rows), "tally": dict(tally), "rows": rows,
            "with_structured_reopen": sum(1 for r in rows if r["reopen_is_structured"]),
            "distinct_reasons": len({r["reason"] for r in rows}),
            "with_receipt": sum(1 for r in rows if r["has_receipt"])}


def _selftest() -> list:
    """The classifier must reject placeholders and accept a real mechanism."""
    fails = []
    cases = [
        ("n/a", False, "placeholder"),
        ("N/A.", False, "placeholder with punctuation"),
        ("unknown", False, "placeholder"),
        ("", False, "empty"),
        ("expert organ is pre-quantized on disk: expert payload is ['U8'] -- a spectrum over "
         "quantization codes measures the codebook, not the organism", True, "real mechanism"),
    ]
    for text, want_mech, label in cases:
        got = bool(text) and not PLACEHOLDER.match(text) and bool(MECHANISM_HINT.search(text))
        if got != want_mech:
            fails.append(f"classifier: {label} -> mechanism={got}, want {want_mech}")
    # A reason that is only a restatement must NOT count as a reopen condition.
    if REOPEN_HINT.search("the body is too large"):
        fails.append("classifier: a flat statement was accepted as a reopen condition")
    if not REOPEN_HINT.search("reopen when a runtime supports this architecture"):
        fails.append("classifier: an explicit reopen condition was rejected")
    return fails


if __name__ == "__main__":
    fails = _selftest()
    for f in fails:
        print("SELFTEST FAIL:", f)
    print(f"selftest: {'FAILED' if fails else 'PASS'}")
    if fails:
        sys.exit(1)
    a = audit()
    print(f"\n{a['n_refused']} REFUSED cells, {a['distinct_reasons']} distinct reasons, "
          f"{a['with_receipt']} carry a receipt")
    for k, v in sorted(a["tally"].items()):
        print(f"  {k:10s} {v}")
    print("\nsample of each verdict:")
    seen = set()
    for r in a["rows"]:
        if r["verdict"] in seen:
            continue
        seen.add(r["verdict"])
        print(f"  [{r['verdict']}] {r['axis']} / {r['slug'][:44]}")
        print(f"      {r['reason'][:170]}")
    if "--write" in sys.argv:
        out = REPO / "receipts" / "future" / "G009_REFUSAL_AUDIT.json"
        out.write_text(json.dumps(a, indent=1) + "\n")
        print("\nwrote", out.relative_to(REPO))

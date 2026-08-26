#!/usr/bin/env python3
"""Traceability checker for a Hawking textbook.

Every quantitative claim carries an inline citation `[receipts/headless/FOO.json#a.b.c]`
immediately after the number it asserts. This walks each citation, opens the receipt,
resolves the JSON path, and compares. It exits non-zero when a claim is untraceable,
its receipt is missing, its path does not resolve, or the value disagrees.

The failure mode this exists to prevent has cost this program real work: a document
quotes a number nobody can trace, and the next campaign inherits it as authority.
"""
import argparse, json, re, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# `NUMBER [path#json.path]` -- the number is whatever token precedes the citation.
CLAIM = re.compile(r"(?P<value>[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)\s*"
                   r"(?P<unit>%|GB/s|bpw|BPW|EBPW|ms|ns|GiB|GB)?\s*"
                   r"\[(?P<receipt>[A-Za-z0-9_./-]+\.json)#(?P<path>[A-Za-z0-9_.\[\]-]+)\]")
# A citation with no number in front of it: a qualitative claim, still checked for
# resolvability so a dead pointer cannot hide behind prose.
BARE = re.compile(r"\[(?P<receipt>[A-Za-z0-9_./-]+\.json)#(?P<path>[A-Za-z0-9_.\[\]-]+)\]")

TOL = 1e-6


def resolve(receipt, path):
    f = REPO / receipt
    if not f.exists():
        return None, f"missing receipt {receipt}"
    try:
        cur = json.load(open(f))
    except Exception as e:
        return None, f"unreadable {receipt}: {e}"
    for part in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
                continue
            except Exception:
                return None, f"{receipt}#{path}: bad list index {part!r}"
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
            continue
        return None, f"{receipt}#{path}: no key {part!r}"
    return cur, None


def compare(claimed, actual):
    if isinstance(actual, dict) and "value" in actual:
        actual = actual["value"]
    try:
        c = float(str(claimed).replace(",", ""))
    except ValueError:
        return False, f"claim {claimed!r} is not numeric"
    if isinstance(actual, bool) or actual is None:
        return False, f"receipt holds {actual!r}, not a number"
    if isinstance(actual, str):
        m = re.search(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", actual)
        if not m:
            return False, f"receipt holds a string with no number: {actual[:60]!r}"
        actual = m.group(0).replace(",", "")
    try:
        aval = float(actual)
    except (TypeError, ValueError):
        return False, f"receipt value not numeric: {actual!r}"
    # a claim may be rounded for prose; accept a stated value that rounds to the receipt's
    dec = len(str(claimed).split(".")[1]) if "." in str(claimed) else 0
    if abs(c - aval) <= TOL or round(aval, dec) == c:
        return True, None
    return False, f"claimed {c} but receipt holds {aval}"


def check(md_path):
    text = Path(md_path).read_text()
    claims, problems = [], []
    seen = set()
    for m in CLAIM.finditer(text):
        seen.add(m.span("receipt")[0])
        val, rec, pth = m.group("value"), m.group("receipt"), m.group("path")
        actual, err = resolve(rec, pth)
        if err:
            problems.append({"kind": "UNRESOLVABLE", "claim": val, "cite": f"{rec}#{pth}",
                             "detail": err})
            claims.append({"value": val, "cite": f"{rec}#{pth}", "ok": False})
            continue
        ok, why = compare(val, actual)
        claims.append({"value": val, "cite": f"{rec}#{pth}", "ok": ok,
                       "receipt_value": actual if not isinstance(actual, (dict, list)) else "<structured>"})
        if not ok:
            problems.append({"kind": "VALUE_DISAGREES", "claim": val,
                             "cite": f"{rec}#{pth}", "detail": why})
    bare = 0
    for m in BARE.finditer(text):
        if m.span("receipt")[0] in seen:
            continue
        bare += 1
        _, err = resolve(m.group("receipt"), m.group("path"))
        if err:
            problems.append({"kind": "DEAD_POINTER",
                             "cite": f"{m.group('receipt')}#{m.group('path')}",
                             "detail": err})
    # a number with a unit and no citation anywhere on its line is an untraceable claim
    untraceable = []
    for i, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith(("|--", ">", "```")) or "UNSUPPORTED" in line:
            continue
        if re.search(r"\d[\d,]*\.?\d*\s*(bpw|BPW|EBPW|GB/s|ms|ns|tok/s)\b", line) \
                and not BARE.search(line):
            untraceable.append({"line": i, "text": line.strip()[:140]})
    return claims, problems, untraceable, bare


# The receipt has to carry the proof, not just the test file. An adversary reading only
# QWEN_TEXTBOOK_V1.json could not tell this checker was capable of failing at all.
BAD_DOC = """# injected control
The whole model is 9.99 EBPW [receipts/headless/WHOLE_MODEL_NATIVE.json#compile.complete_ebpw].
A dead pointer [receipts/headless/NO_SUCH_RECEIPT.json#a.b].
An uncited number: throughput was 412.5 GB/s.
"""


def self_control():
    """Run the checker on a document engineered to fail every way it can."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(BAD_DOC)
        path = f.name
    claims, problems, untraceable, _ = check(path)
    Path(path).unlink(missing_ok=True)
    kinds = sorted({p["kind"] for p in problems})
    return {
        "what": "a document with a wrong value, a dead pointer and an uncited number",
        "problem_kinds_caught": kinds,
        "n_problems": len(problems), "n_untraceable": len(untraceable),
        "checker_can_fail": bool(problems and untraceable),
        "expected_kinds": ["DEAD_POINTER", "VALUE_DISAGREES"],
        "all_expected_caught": all(k in kinds for k in ("DEAD_POINTER", "VALUE_DISAGREES")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("markdown")
    ap.add_argument("--emit")
    a = ap.parse_args()
    claims, problems, untraceable, bare = check(a.markdown)
    out = {
        "schema": "hawking.headless.textbook_trace.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/headless/textbook_trace.py",
        "obligation": "G006 — QWEN_TEXTBOOK_V1 traceability (directive §13)",
        "document": a.markdown,
        "n_claims": len(claims), "n_bare_citations": bare,
        "n_untraceable": len(untraceable), "n_problems": len(problems),
        "problems": problems, "untraceable_lines": untraceable,
        "claims": claims,
        "negative_control": self_control(),
        "pass": (not problems and not untraceable
                 and self_control()["all_expected_caught"]),
    }
    if a.emit:
        Path(a.emit).write_text(json.dumps(out, indent=1))
    print(f"claims={len(claims)} bare={bare} problems={len(problems)} "
          f"untraceable={len(untraceable)} pass={out['pass']}")
    for p in problems[:12]:
        print("  ", p["kind"], p.get("claim", ""), p["cite"], "--", p["detail"][:120])
    for u in untraceable[:12]:
        print("   UNTRACEABLE line", u["line"], ":", u["text"][:110])
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

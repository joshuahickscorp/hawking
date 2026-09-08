"""G026 verify: comparability must be EARNED, and losing a field must lose it.

tps_contract over the live receipts found 79 physical and 1 comparable. The
forbidden repair is loosening the contract; the required one is emitting the six
fields correctly. So this checks two things a "field exists" assertion would
miss: that the contract file is untouched, and that dropping any ONE contract
field makes the emitter's own output INCOMPARABLE. A test that only confirms
the happy path cannot tell a real emitter from a hardcoded dict.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "future"))

from physical_emitter import emit, ROUNDS  # noqa: E402
from tps_contract import audit, REQUIRED  # noqa: E402


def _newest_with_calls() -> Path:
    cands = []
    for f in ROUNDS.glob("*.json"):
        try:
            if json.loads(f.read_text()).get("model_calls"):
                cands.append(f)
        except Exception:
            pass
    if not cands:
        raise SystemExit("no round receipt with model_calls -- nothing to test")
    return max(cands, key=lambda p: p.stat().st_mtime)


def main() -> int:
    fails = []
    src = _newest_with_calls()
    rec = emit(src)

    with tempfile.TemporaryDirectory() as td:
        good = Path(td) / "good.json"
        good.write_text(json.dumps(rec, indent=1))
        a = audit(good)
        # audit() returns a VERDICT, not a `physical` flag. Asserting a key the
        # function never emits is a test that fails a correct implementation --
        # which is how this one first failed.
        if a.get("verdict") != "COMPARABLE":
            fails.append(f"baseline emitter output is {a.get('verdict')}, "
                         f"missing {a.get('missing')}")
        if not a.get("rates"):
            fails.append("baseline emitter output carries no rate key -- it is not physical")

        # THE MUTATION. Remove one contract field at a time; each must break it.
        # tps_contract walks for the first occurrence of any alias, so a field
        # also has to not be re-satisfied from somewhere else in the document.
        for field, aliases in REQUIRED.items():
            hurt = json.loads(json.dumps(rec))
            def strip(node):
                if isinstance(node, dict):
                    for a in aliases:
                        node.pop(a, None)
                    for v in node.values():
                        strip(v)
                elif isinstance(node, list):
                    for v in node:
                        strip(v)
            strip(hurt)
            p = Path(td) / f"no_{field}.json"
            p.write_text(json.dumps(hurt, indent=1))
            res = audit(p)
            if res.get("verdict") == "NOT_A_PHYSICAL_RECEIPT":
                fails.append(f"MUTATION: stripping `{field}` also removed every rate key -- "
                             f"the check cannot distinguish a missing field from a missing rate")
                continue
            if field not in (res.get("missing") or []):
                fails.append(f"MUTATION: stripped every alias of `{field}` and the receipt was "
                             f"still comparable -- that field is not load-bearing")

    # The contract itself must not have moved. Emitting correctly is the fix;
    # widening the ruler is the thing this obligation exists to forbid.
    diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--",
                           "tools/future/tps_contract.py"], cwd=REPO)
    if diff.returncode != 0:
        fails.append("tools/future/tps_contract.py has uncommitted changes -- comparability may "
                     "have been bought by loosening the ruler")

    for f in fails:
        print("FAIL:", f)
    print(f"{'FAILED' if fails else 'PASS'} — 6 contract fields mutation-tested, "
          f"contract file unmodified")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3.12
"""How many models are at each stage? Answered from receipts, not from prose.

EPOCH II G020. The directive is explicit: "Persist the counts automatically. At
any moment Hawking should be able to answer HOW MANY MODELS ARE AT EACH STAGE
without Claude reconstructing history from prose." And: "Do not confuse CENSUSED
with ODYSSEY'D."

So the stages are DERIVED. Each is a predicate over the receipt tree, and a
specimen is at stage N because a receipt says so, not because someone remembered
it was. If a stage's evidence is deleted the count drops, which is the point: the
number tracks the evidence rather than the narrative.

Stages are cumulative in intent but computed independently, so a specimen that
somehow reached stage 8 without stage 4 shows up as exactly that rather than
being silently promoted.
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECEIPTS = REPO / "receipts"

STAGES = [
    "1 REGISTERED",
    "2 CHEAPLY CENSUSED",
    "3 ODYSSEY I COMPLETE",
    "4 ODYSSEY II / TRANSFER TESTED",
    "5 DEEP GRAVITY ENTERED",
    "6 NR CREATED",
    "7 NOETIC EXECUTABLE",
    "8 NX EXECUTED",
    "9 PHYSICALLY OPTIMIZED",
    "10 ODYSSEY III ATTACKED",
    "11 SEALED / DISPOSITIONED",
]


def _norm(name):
    """A specimen key that survives the three spellings in use: the lake's
    `org--model@sha`, the receipt's `org--model`, and bare `model`."""
    n = str(name).split("@")[0]
    n = n.split("--")[-1]
    return n.lower().replace("_", "-")


def _texts():
    """(path, text) for every receipt, read once."""
    for p in sorted(RECEIPTS.rglob("*.json")):
        try:
            yield p, p.read_text(errors="replace")
        except Exception:
            continue


def collect():
    census = RECEIPTS / "odyssey-i" / "MODELLAKE_CENSUS_2026-09-09.json"
    registered = {}
    if census.exists():
        for s in json.loads(census.read_text())["specimens"]:
            registered[_norm(s["name"])] = {
                "name": s["name"],
                "gib": s.get("dir_size_gib"),
                "family": (s.get("classification") or {}).get("model_type"),
            }

    stage = defaultdict(set)
    stage["1 REGISTERED"] |= set(registered)
    for key, meta in registered.items():
        if meta["family"] is not None:
            stage["2 CHEAPLY CENSUSED"].add(key)

    # STAGES ARE DECLARED, NOT GUESSED.
    #
    # Two attempts at inferring them from filenames produced confident wrong
    # numbers: `\bNX\b` never matches `_NX_` (underscore is a word character, so
    # there is no boundary) and reported ZERO NX specimens while one exists; and
    # a "PHYSICAL" filename match promoted fifteen bodies because one receipt
    # DISCUSSES direct execution and names them all in its prose. Tightening the
    # patterns moved the errors around rather than removing them.
    #
    # A counter that is wrong is worse than no counter, because it gets quoted.
    # So the deep stages are now DECLARED by the receipt that earned them --
    # `"odyssey_stage": "8 NX EXECUTED"` -- and anything not declared is counted
    # as UNDECLARED rather than inferred. Backfilling those declarations is
    # work; pretending to know without them is a fabricated measurement.
    DECLARED = 0
    for path, text in _texts():
        try:
            rec = json.loads(text)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        st_name = rec.get("odyssey_stage")
        spec = rec.get("specimen") or rec.get("model") or ""
        many = rec.get("odyssey_stage_specimens")
        # A receipt covering SEVERAL specimens has no single `specimen` key, and
        # requiring one silently dropped the cross-layer transfer receipt --
        # seven bodies declared, two counted.
        if not st_name or not (spec or many):
            continue
        for one in (many if isinstance(many, list) and many else [spec]):
            key = _norm(one)
            if key in registered and st_name in STAGES:
                stage[st_name].add(key)
        DECLARED += 1

    # Stage 3 is the one exception, and only because its evidence is
    # unambiguous: a disposition receipt whose FILENAME carries the specimen.
    for path, _text in _texts():
        n = path.name.lower()
        if "disposition" not in n and "odyssey_i" not in n:
            continue
        for key in registered:
            if key in n:
                stage["3 ODYSSEY I COMPLETE"].add(key)
                stage["11 SEALED / DISPOSITIONED"].add(key)
    stage["_declared_receipts"] = DECLARED  # type: ignore[index]
    return registered, stage


def report(registered, stage, verbose=False):
    print(f"MODELLAKE POPULATION -- {len(registered)} specimens registered\n")
    prev = None
    for st in STAGES:
        n = len(stage.get(st, ()))
        bar = "#" * min(40, n)
        drop = "" if prev is None else f"  ({n - prev:+d} vs previous stage)"
        print(f"  {st:32s} {n:3d}  {bar}{drop}")
        prev = n
    declared = len(stage.get("_declared_receipts", ())) if not isinstance(
        stage.get("_declared_receipts"), int) else stage["_declared_receipts"]
    print(f"\n  stages 4-10 are DECLARED by receipts, not inferred. "
          f"{declared} receipt(s) currently declare one.")
    if not declared:
        print("  -> every deep stage reads 0 because no receipt declares "
              "`odyssey_stage` yet. That is UNDECLARED, not zero progress, and "
              "backfilling it is real work rather than a heuristic.")
    print("\nCENSUSED is not ODYSSEY'D: stage 2 minus stage 4 is the untouched population.")
    gap = len(stage.get("2 CHEAPLY CENSUSED", ())) - len(stage.get("4 ODYSSEY II / TRANSFER TESTED", ()))
    print(f"  {gap} specimens censused but never transfer-tested.")
    if verbose:
        for st in STAGES:
            names = sorted(stage.get(st, ()))
            if names:
                print(f"\n{st}:\n  " + "\n  ".join(names))
    return gap


if __name__ == "__main__":
    registered, stage = collect()
    gap = report(registered, stage, verbose="-v" in sys.argv)
    out = RECEIPTS / "future" / "POPULATION_STAGES.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "registered": len(registered),
        "counts": {st: len(stage.get(st, ())) for st in STAGES},
        "members": {st: sorted(stage.get(st, ())) for st in STAGES},
        "censused_but_not_transfer_tested": gap,
        "note": ("derived from receipts, not maintained by hand. If a stage's "
                 "evidence is deleted the count drops -- the number tracks the "
                 "evidence rather than the narrative."),
    }, indent=1))
    print(f"\nwrote {out}")

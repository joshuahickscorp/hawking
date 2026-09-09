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

    # Everything else is evidence-driven: a stage is claimed only if a receipt
    # mentions the specimen AND the receipt is of the right kind.
    RULES = [
        ("3 ODYSSEY I COMPLETE",          r"ODYSSEY_I|DISPOSITION|_ANATOMY|CENSUS"),
        ("4 ODYSSEY II / TRANSFER TESTED", r"CROSS_LAYER|TRANSFER|ORGAN_GATE"),
        ("5 DEEP GRAVITY ENTERED",         r"GRAVITY|ACTIVATION_AWARE|SPARSE_RESIDUAL|PROCEDURAL|STRUCTURAL_CLASSES"),
        ("6 NR CREATED",                   r"\bNR\b|NOETIC_REPRESENT"),
        ("7 NOETIC EXECUTABLE",            r"NOETIC"),
        ("8 NX EXECUTED",                  r"\bNX\b"),
        ("9 PHYSICALLY OPTIMIZED",         r"PHYSICAL|DIRECT_EXECUTION|FUSED"),
        ("10 ODYSSEY III ATTACKED",        r"ODYSSEY3|OIII|ODYSSEY_III"),
        ("11 SEALED / DISPOSITIONED",      r"DISPOSITION|SEAL"),
    ]
    for path, text in _texts():
        hay = f"{path.name}\n{text}"
        low = text.lower()
        hits = [st for st, pat in RULES if re.search(pat, path.name, re.I)]
        if not hits:
            continue
        for key in registered:
            if key in low or key in path.name.lower():
                for st in hits:
                    stage[st].add(key)
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

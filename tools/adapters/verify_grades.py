#!/usr/bin/env python3.12
"""Verify every adapter grade against the evidence it cites. Demote what is not backed.

A support grade is a claim about the world, and the registry is where that claim is easiest
to inflate: a family's shapes look familiar, a test file exists with a plausible name, and
the grade drifts upward without anyone deciding to lie.

This checks two things the eye does not:

  1. Does the cited path exist at all?
  2. If the evidence is a Rust test, does that test actually EXECUTE, or does it skip when
     its model file is absent?

The second is the one that matters here.  `integration_greedy_64.rs` was cited as
`full_parent_validation` for Qwen.  It finishes in 0.00s and prints "skipping ...: no model
on disk".  A test that skips when its weights are missing is a green checkmark that guards
nothing, and inheriting a grade from it is how a registry ends up asserting more than the
repository can support.

    python3.12 -m tools.adapters.verify_grades          # report
    python3.12 -m tools.adapters.verify_grades --apply  # rewrite the registry with honest grades
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "HAWKING_ADAPTER_REGISTRY.json"

# Ordered weakest to strongest. Demotion moves down this list.
GRADES = [
    "DECLARED",
    "SOURCE_HEADER_VALIDATED",
    "SYNTHETIC_PARITY",
    "REAL_TENSOR_DECODE",
    "SMALL_REAL_CHECKPOINT",
    "FULL_PARENT_VALIDATED",
    "PRODUCTION",
]

# What kind of evidence each grade REQUIRES. A description never licenses anything above
# DECLARED, because describing a thing is not running it.
REQUIRED_KIND = {
    "SOURCE_HEADER_VALIDATED": {"header_parse", "source_receipt", "description"},
    "SYNTHETIC_PARITY": {"synthetic_parity", "synthetic_run"},
    "REAL_TENSOR_DECODE": {"real_tensor_decode", "small_checkpoint_run", "sealed_receipt"},
    "SMALL_REAL_CHECKPOINT": {"small_checkpoint_run", "sealed_receipt"},
    "FULL_PARENT_VALIDATED": {"full_parent_validation", "sealed_receipt"},
    "PRODUCTION": {"production_receipt"},
}

SKIP_MARKERS = re.compile(r"skipping|no model on disk|no models/|not present|return;\s*$", re.I)


def _rust_test_executes(path: Path) -> tuple[bool, str]:
    """Does this Rust test do real work, or bail when its weights are missing?

    Static read rather than execution: running every cited test would be minutes of cargo
    under LIGHT_ONLY. A test whose source contains an early return guarded by a
    file-presence check is conditional evidence, and conditional evidence that is currently
    false is not evidence.
    """
    if not path.exists():
        return False, "cited test file does not exist"
    src = path.read_text(errors="ignore")
    guards = re.findall(r"(?:is_file\(\)|exists\(\)|env::var)[^\n]{0,120}", src)
    skips = SKIP_MARKERS.findall(src)
    if guards and skips:
        return False, f"skips when weights are absent ({len(guards)} presence guards, {len(skips)} skip markers)"
    return True, "no weight-presence guard found; runs unconditionally"


def _models_on_disk() -> bool:
    return any((ROOT / "models").glob("*.gguf")) if (ROOT / "models").is_dir() else False


def check_family(fam: dict) -> dict:
    name = str(fam.get("family", fam.get("id", "?")))
    claimed = fam.get("level", fam.get("grade", "DECLARED"))
    evidence = fam.get("evidence") or fam.get("parity_evidence") or []
    if isinstance(evidence, dict):
        evidence = [evidence]

    verdicts = []
    usable_kinds: set[str] = set()
    for e in evidence:
        kind = e.get("kind", "unknown")
        p = e.get("path")
        path = ROOT / p if p else None
        if path is None or not path.exists():
            verdicts.append({"kind": kind, "path": p, "ok": False, "why": "path does not exist"})
            continue
        if path.suffix == ".rs" and "tests/" in str(p):
            ok, why = _rust_test_executes(path)
            verdicts.append({"kind": kind, "path": p, "ok": ok, "why": why})
            if ok:
                usable_kinds.add(kind)
        else:
            # Trust the DECLARED kind, never the file extension. An earlier version of this
            # inferred "sealed_receipt" from a .json suffix, which handed gemma and phi
            # FULL_PARENT_VALIDATED off a header-parse receipt -- the verifier committing
            # the exact inflation it exists to catch.
            verdicts.append({"kind": kind, "path": p, "ok": True,
                             "why": f"path exists; kind declared as {kind!r}"})
            usable_kinds.add(kind)

    # Highest grade the surviving evidence licenses, by declared kind alone.
    supported = "DECLARED"
    for g in GRADES[1:]:
        if REQUIRED_KIND.get(g, set()) & usable_kinds:
            supported = g
    # A description alone never exceeds SOURCE_HEADER_VALIDATED: describing a thing is not
    # running it.
    if usable_kinds <= {"description"} and GRADES.index(supported) > GRADES.index("SOURCE_HEADER_VALIDATED"):
        supported = "SOURCE_HEADER_VALIDATED"

    inflated = GRADES.index(claimed) > GRADES.index(supported)
    return {
        "family": name,
        "claimed": claimed,
        "supported": supported,
        "inflated": inflated,
        "usable_evidence_kinds": sorted(usable_kinds),
        "evidence_verdicts": verdicts,
    }


def main() -> int:
    reg = json.loads(REGISTRY.read_text())
    fams = reg.get("families", [])
    as_list = isinstance(fams, list)
    items = fams if as_list else list(fams.values())

    results = [check_family(f) for f in items]
    inflated = [r for r in results if r["inflated"]]

    print(f"models/*.gguf present on disk: {_models_on_disk()}")
    for r in results:
        flag = "  INFLATED ->" if r["inflated"] else "  ok"
        print(f"{r['family']:16s} claimed={r['claimed']:22s}{flag} supported={r['supported']}")
        for v in r["evidence_verdicts"]:
            if not v["ok"]:
                print(f"      rejected {v['path']}: {v['why']}")

    if "--apply" in sys.argv:
        for f, r in zip(items, results):
            if r["inflated"]:
                key = "level" if "level" in f else "grade"
                f[key] = r["supported"]
                f["demoted_by_verifier"] = {
                    "from": r["claimed"],
                    "why": "cited evidence does not execute or does not exist",
                    "rejected": [v for v in r["evidence_verdicts"] if not v["ok"]],
                }
        reg["grade_verification"] = {
            "at": "generated by tools/adapters/verify_grades.py",
            "law": "a test that skips when its weights are absent is not evidence",
            "models_on_disk": _models_on_disk(),
            "demoted": [r["family"] for r in inflated],
        }
        REGISTRY.write_text(json.dumps(reg, indent=2) + "\n")
        print(f"\napplied: demoted {len(inflated)} families")

    print(f"\n{len(inflated)} of {len(results)} families claim a grade their evidence does not support")
    return 1 if inflated and "--apply" not in sys.argv else 0


if __name__ == "__main__":
    raise SystemExit(main())

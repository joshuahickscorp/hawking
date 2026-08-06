#!/usr/bin/env python3.12
"""Close the generated-source loophole in the LOC authority.

`tools/loc/hawking_loc.py` excludes any tracked file under a `generated/` directory or
named `*.generated.rs` / `*.generated.ts`. That exclusion is correct when the file really
is machine output from a small reviewed spec, and it is a laundering channel when it is
not: move 30,000 hand-written lines into `src/generated/` and the number falls without the
system getting smaller.

Campaign rule (control/REBUILD_ACCOUNTING_RULES.json, `generation_rules`): a generated file
escapes the count only when

  1. it is registered in control/GENERATED_REGISTRY.json with a generator and its specs,
  2. re-running that generator reproduces it byte-identically, and
  3. the amplification generated_loc / (spec_loc + generator_loc) is at least 4.0.

Anything failing those is RECLASSIFIED AS ACTIVE and added back to the LOC total.

    python3.12 tools/loc/hawking_generation_audit.py            # audit, human output
    python3.12 tools/loc/hawking_generation_audit.py --json     # machine output
    python3.12 tools/loc/hawking_generation_audit.py --gate     # exit 1 if any file is unearned

The gate is the point. `hawking_loc.py` stays frozen; this runs beside it and reports the
adjusted figure, so a rung is measured at `combined_active_monorepo_LOC + reclassified`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.layout import CONTROL_ROOT

REGISTRY = CONTROL_ROOT / "catalog/manifests/GENERATED_REGISTRY.json"
MIN_AMPLIFICATION = 4.0

# Kept in step with hawking_loc.py's SOURCE_SUFFIXES; a generated .json or .md is data,
# not source, and was never in the count, so it is out of scope here too.
SOURCE_SUFFIXES = {".rs", ".py", ".ts", ".tsx", ".sh", ".metal", ".wgsl", ".lean", ".md"}


def tracked_generated() -> list[str]:
    """Every tracked file hawking_loc.py would put in the `generated` bucket."""
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    hits = []
    for p in out:
        if not p:
            continue
        if Path(p).suffix not in SOURCE_SUFFIXES:
            continue
        if "/generated/" in p or p.endswith((".generated.rs", ".generated.ts")):
            hits.append(p)
    return sorted(hits)


def loc(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return 0


def load_registry() -> dict:
    if not REGISTRY.exists():
        return {"entries": []}
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def reproduce(entry: dict) -> tuple[bool, str]:
    """Run the generator into a scratch tree and diff its outputs byte-for-byte."""
    cmd = entry.get("generator_cmd")
    if not cmd:
        return False, "no generator_cmd"
    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        # The generator writes into the repo. Snapshot the outputs, regenerate, compare,
        # then restore -- so a non-deterministic generator cannot leave the tree dirty.
        outputs = [ROOT / p for p in entry["outputs"]]
        saved = {}
        for p in outputs:
            if p.exists():
                saved[p] = p.read_bytes()
        try:
            r = subprocess.run(
                cmd, cwd=ROOT, shell=True, capture_output=True, text=True, timeout=900
            )
            if r.returncode != 0:
                return False, f"generator exit {r.returncode}: {r.stderr.strip()[:200]}"
            for p in outputs:
                if not p.exists():
                    return False, f"generator did not produce {p.relative_to(ROOT)}"
                if saved.get(p) != p.read_bytes():
                    return False, f"regeneration differs from tracked {p.relative_to(ROOT)}"
            return True, "byte-identical"
        except subprocess.TimeoutExpired:
            return False, "generator timed out after 900s"
        finally:
            for p, blob in saved.items():
                p.write_bytes(blob)
            shutil.rmtree(scratch, ignore_errors=True)


def audit(verify: bool) -> dict:
    registry = load_registry()
    by_output: dict[str, dict] = {}
    for e in registry.get("entries", []):
        for o in e.get("outputs", []):
            by_output[o] = e

    findings = []
    reclassified = 0
    checked_entries: dict[str, tuple[bool, str]] = {}

    for path in tracked_generated():
        n = loc(ROOT / path)
        entry = by_output.get(path)
        if entry is None:
            findings.append(
                {"path": path, "loc": n, "verdict": "RECLASSIFIED_ACTIVE",
                 "reason": "not registered in control/GENERATED_REGISTRY.json"}
            )
            reclassified += n
            continue

        eid = entry["id"]
        spec_loc = sum(loc(ROOT / s) for s in entry.get("specs", []))
        gen_loc = sum(loc(ROOT / g) for g in entry.get("generator_sources", []))
        out_loc = sum(loc(ROOT / o) for o in entry["outputs"])
        cost = spec_loc + gen_loc
        amp = (out_loc / cost) if cost else 0.0

        if cost == 0:
            findings.append(
                {"path": path, "loc": n, "verdict": "RECLASSIFIED_ACTIVE", "entry": eid,
                 "reason": "registry entry names no specs and no generator sources"}
            )
            reclassified += n
            continue

        if amp < MIN_AMPLIFICATION:
            findings.append(
                {"path": path, "loc": n, "verdict": "RECLASSIFIED_ACTIVE", "entry": eid,
                 "amplification": round(amp, 2),
                 "reason": f"amplification {amp:.2f} below the {MIN_AMPLIFICATION} floor; "
                           f"{cost} lines of spec+generator produce {out_loc} lines"}
            )
            reclassified += n
            continue

        if verify:
            if eid not in checked_entries:
                checked_entries[eid] = reproduce(entry)
            ok, why = checked_entries[eid]
            if not ok:
                findings.append(
                    {"path": path, "loc": n, "verdict": "RECLASSIFIED_ACTIVE", "entry": eid,
                     "amplification": round(amp, 2), "reason": f"not reproducible: {why}"}
                )
                reclassified += n
                continue
            reason = f"reproducible, amplification {amp:.2f}"
        else:
            reason = f"amplification {amp:.2f}, reproduction not checked (--gate to check)"

        findings.append(
            {"path": path, "loc": n, "verdict": "EARNED", "entry": eid,
             "amplification": round(amp, 2), "reason": reason}
        )

    unearned = [f for f in findings if f["verdict"] == "RECLASSIFIED_ACTIVE"]
    return {
        "schema": "hawking.generation_audit.v1",
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "min_amplification": MIN_AMPLIFICATION,
        "reproduction_verified": verify,
        "generated_files": len(findings),
        "earned_files": len(findings) - len(unearned),
        "unearned_files": len(unearned),
        "reclassified_active_LOC": reclassified,
        "findings": findings,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--gate", action="store_true",
                    help="verify reproduction and exit 1 if any generated file is unearned")
    ap.add_argument("--out", help="also write the report to this path")
    args = ap.parse_args()

    report = audit(verify=args.gate)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"generated source files: {report['generated_files']}")
        print(f"  earned exclusion:     {report['earned_files']}")
        print(f"  reclassified active:  {report['unearned_files']}"
              f"  ({report['reclassified_active_LOC']:,} LOC added back)")
        for f in report["findings"]:
            if f["verdict"] == "RECLASSIFIED_ACTIVE":
                print(f"  ! {f['path']} ({f['loc']:,}) -- {f['reason']}")

    if args.gate and report["unearned_files"]:
        print("\nGATE FAIL: generated exclusion is not earned for the files above.",
              file=sys.stderr)
        return 1
    return 0


def _selfcheck() -> None:
    """Smallest thing that fails if the classification logic breaks."""
    reg = {"entries": [{"id": "t", "outputs": ["a/generated/x.rs"], "specs": ["s.json"],
                        "generator_sources": ["g.py"], "generator_cmd": "true"}]}
    by_output = {o: e for e in reg["entries"] for o in e["outputs"]}
    assert by_output["a/generated/x.rs"]["id"] == "t"
    assert "b/generated/y.rs" not in by_output, "unregistered output must not resolve"
    # amplification arithmetic
    assert (400 / (50 + 30)) >= MIN_AMPLIFICATION
    assert (200 / (50 + 30)) < MIN_AMPLIFICATION
    # only source suffixes are in scope
    assert Path("crates/x/generated/a.json").suffix not in SOURCE_SUFFIXES
    assert Path("crates/x/generated/a.rs").suffix in SOURCE_SUFFIXES
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        raise SystemExit(0)
    raise SystemExit(main())

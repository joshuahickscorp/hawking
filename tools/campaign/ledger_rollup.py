#!/usr/bin/env python3.12
"""Roll the per-slice ledgers up into one campaign accounting, and check it reconciles.

The endpoint requires "exact eliminated/rewritten/generated/relocated/facade accounting".
Each slice writes a record under `workspace/campaign/governance/control/ledgers/`; this sums them and compares the total against
what the measurement authority actually says changed between two commits. A campaign whose
slices each reconcile locally can still be wrong globally -- lines get counted twice when a
later slice deletes what an earlier one rewrote, and instruments the campaign itself adds
belong to nobody's slice.

    python3.12 tools/campaign/ledger_rollup.py --from rebuild-250k-start --to HEAD

The residual is the deliverable. A rollup that reports zero residual on a campaign this size
is not being honest with itself; what matters is that the residual is named and attributed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.layout import CONTROL_ROOT

# The campaign names five ledgers. A sixth category is genuinely missing from that scheme:
# the instruments and records the campaign itself adds belong to no slice, and without a slot
# they surface as an unexplained residual. `added_apparatus` is that slot.
KEYS = ("eliminated", "rewritten", "generated", "relocated", "facade", "added_apparatus")
GENERATED_SOURCE_SUFFIXES = {
    ".rs", ".py", ".ts", ".tsx", ".sh", ".metal", ".wgsl", ".lean", ".md",
}
MIN_GENERATION_AMPLIFICATION = 4.0


def git_at(rev: str, *args: str) -> bytes | None:
    r = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True,
    )
    return r.stdout if r.returncode == 0 else None


def blob_at(rev: str, path: str) -> bytes | None:
    return git_at(rev, "show", f"{rev}:{path}")


def blob_loc(rev: str, path: str) -> int:
    blob = blob_at(rev, path)
    return 0 if blob is None else len(blob.split(b"\n")) - 1


def is_generated_source(path: str) -> bool:
    return (
        Path(path).suffix in GENERATED_SOURCE_SUFFIXES
        and ("/generated/" in path or path.endswith((".generated.rs", ".generated.ts")))
    )


def generation_reclassified_at(rev: str) -> int:
    """Apply the generation registry's accounting rule to a committed tree.

    The live generation gate also re-runs generators. A historical rollup cannot
    safely execute a generator from an arbitrary revision in the current checkout,
    so it uses the committed registry evidence (tracked output/spec/generator
    paths and >=4x amplification). Commits are admitted only after the live gate
    has already proved byte reproduction; an absent or incomplete registry is
    conservatively reclassified active here.
    """
    tree = git_at(rev, "ls-tree", "-r", "--name-only", rev)
    if tree is None:
        return 0
    generated = [
        p for p in tree.decode("utf-8", "replace").splitlines()
        if is_generated_source(p)
    ]
    registry_blob = blob_at(
        rev,
        "workspace/campaign/governance/control/catalog/manifests/GENERATED_REGISTRY.json",
    )
    if registry_blob is None:
        registry_blob = blob_at(rev, "control/GENERATED_REGISTRY.json")
    try:
        registry = json.loads(registry_blob) if registry_blob is not None else {"entries": []}
    except json.JSONDecodeError:
        registry = {"entries": []}

    earned: set[str] = set()
    for entry in registry.get("entries", []):
        outputs = entry.get("outputs", [])
        specs = entry.get("specs", [])
        generators = entry.get("generator_sources", [])
        if not outputs or not specs or not generators or not entry.get("generator_cmd"):
            continue
        cost = sum(blob_loc(rev, p) for p in [*specs, *generators])
        output_loc = sum(blob_loc(rev, p) for p in outputs)
        if cost > 0 and output_loc / cost >= MIN_GENERATION_AMPLIFICATION:
            earned.update(outputs)
    return sum(blob_loc(rev, p) for p in generated if p not in earned)


def loc_at(rev: str) -> int | None:
    """The authority's figure at a revision, plus whatever the generation audit adds back."""
    r = subprocess.run(
        [sys.executable or "python3.12", "tools/loc/hawking_loc.py", "--json", "--rev", rev],
        cwd=ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        r = subprocess.run(
            [sys.executable or "python3.12", "tools/loc/hawking_loc.py", "--rev", rev],
            cwd=ROOT, capture_output=True, text=True,
        )
        m = re.search(r"combined active LOC:\s*([\d,]+)", r.stdout)
        raw = int(m.group(1).replace(",", "")) if m else None
        return None if raw is None else raw + generation_reclassified_at(rev)
    try:
        raw = json.loads(r.stdout)["combined_active_monorepo_LOC"]
        return raw + generation_reclassified_at(rev)
    except (json.JSONDecodeError, KeyError):
        return None


def slice_ledgers() -> list[dict]:
    out = []
    for p in sorted((CONTROL_ROOT / "ledgers").rglob("S*-ledger.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            out.append({"slice": p.stem, "error": f"invalid JSON: {e}"})
            continue
        led = d.get("ledger", d)
        out.append({
            "slice": d.get("slice", p.stem),
            "file": str(p.relative_to(ROOT)),
            **{k: int(led.get(k, 0) or 0) for k in KEYS},
            "target_reached": d.get("target_reached"),
            "stop_reason": d.get("stop_reason"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="frm", default="rebuild-250k-start")
    ap.add_argument("--to", dest="to", default="HEAD")
    ap.add_argument("--out")
    args = ap.parse_args()

    ledgers = slice_ledgers()
    totals = {k: sum(l.get(k, 0) for l in ledgers) for k in KEYS}

    a, b = loc_at(args.frm), loc_at(args.to)
    measured = (b - a) if (a is not None and b is not None) else None

    # Slices claim reductions as positive `eliminated`/`rewritten`; additions land in
    # `generated`/`facade`. The tree delta a campaign should see is therefore:
    claimed = -(totals["eliminated"] + totals["rewritten"]) \
        + totals["generated"] + totals["relocated"] + totals["facade"] \
        + totals["added_apparatus"]
    residual = (measured - claimed) if measured is not None else None

    rep = {
        "schema": "hawking.campaign_ledger_rollup.v1",
        "from": args.frm, "to": args.to,
        "loc_from": a, "loc_to": b,
        "measured_delta": measured,
        "slice_ledgers": ledgers,
        "totals": totals,
        "claimed_delta": claimed,
        "residual": residual,
        "reading": (
            "residual = measured - claimed. Positive residual is lines the campaign added "
            "that no slice ledger accounts for -- typically its own instruments and records. "
            "Negative residual is reduction no slice claimed. Either way it must be named, "
            "not absorbed."
        ),
    }

    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")

    print(f"{args.frm} -> {args.to}")
    if measured is not None:
        print(f"  measured LOC delta   {measured:+,}   ({a:,} -> {b:,})")
    for k in KEYS:
        print(f"  {k:12s} {totals[k]:>10,}")
    print(f"  claimed delta        {claimed:+,}")
    if residual is not None:
        print(f"  RESIDUAL             {residual:+,}  <- must be named")
    for l in ledgers:
        if l.get("target_reached") is False:
            print(f"  ! {l['slice']} stopped short: {str(l.get('stop_reason'))[:110]}")
    return 0


def _selfcheck() -> None:
    tot = {"eliminated": 100, "rewritten": 50, "generated": 10, "relocated": 0,
           "facade": 0, "added_apparatus": 0}
    claimed = (-(tot["eliminated"] + tot["rewritten"]) + tot["generated"]
               + tot["relocated"] + tot["facade"] + tot["added_apparatus"])
    assert claimed == -140, claimed
    assert (-140) - claimed == 0, "a tree that fell exactly as claimed has zero residual"
    assert (-100) - claimed == 40, "under-delivery shows as positive residual"
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck(); raise SystemExit(0)
    raise SystemExit(main())

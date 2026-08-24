#!/usr/bin/env python3
"""StorageGenome / reclamation policy (directive §8, §23).

    "Storage management must become a real subsystem, not ad hoc rm commands."
    "Storage management should be automatic but conservative."

This is the selection half. It never deletes anything: it RANKS candidates and
refuses protected classes, and `--execute` only prints the commands a human or a
supervised WorkUnit would run. That asymmetry is deliberate — the campaign has
already lost real artifacts to an automated reclaim.

The loss is on the record. `receipts/ascent-2026-08-18/G28_STORAGE_RECLAIM.json`
listed `workspace/campaign/records/runs/qwen38-27b/bf16 (51 GB, SOURCE patient)`
and the champion 3.3448 BPW artifacts on an explicit KEEP_LIST. Four days later
they are gone, cleared by the odyssey driver's `reclaim_safe`, and because
`runs/` is gitignored the deletion left no git trace at all. A KEEP_LIST written
in a receipt is a note; it is not a mechanism. This module is the mechanism.

Ranking, per §8: bytes reclaimed, regeneration cost, scientific uniqueness,
probability of future reuse.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REPO = Path(os.path.expanduser("~/Downloads/hawking-copy"))
LEDGER = REPO / "receipts/headless/ARTIFACT_LEDGER.json"
GIB = 1024 ** 3

# §8: "Never delete" classes. Membership here is a hard refusal, not a penalty.
PROTECTED_CLASSES = {
    "KEEP_ACTIVE_PARENT",
    "KEEP_ROLLBACK_PARENT",
    "KEEP_UNIQUE_SCIENCE",
    "KEEP_CURRENT_CANDIDATE",
    "UNKNOWN_DO_NOT_DELETE",
}
# Only these may ever be considered, and only with a reproduction route recorded.
ELIGIBLE_CLASSES = {"REDOWNLOADABLE", "REPRODUCIBLE_COLD", "DELETE_ELIGIBLE"}

# Paths that are never candidates regardless of classification, because losing
# them loses evidence rather than bytes. Matched case-insensitively, and both
# separator conventions are listed: a first version used only "NEGATIVE_SCIENCE"
# and let the real file `negative-science.jsonl` through — a guard that depends on
# someone spelling a filename the way you imagined is not a guard.
NEVER_SUBSTRINGS = (
    "/receipts/", "/.git/",
    "negative_science", "negative-science", "negativescience",
    "genome", "_ledger", "-ledger",
    "manifest", "capture-result", "capture_result",
)


class ProtectedArtifact(Exception):
    """Raised when something asks to delete an artifact that must never be deleted."""


def regeneration_cost(row: dict) -> float:
    """Hours, roughly. Cheap to redownload is cheap to lose; expensive to
    recompute is not."""
    cls = row.get("classification")
    gib = row.get("size_gib") or 0
    if cls == "REDOWNLOADABLE":
        # ~2.0 Gbit/s measured ceiling on this machine => ~0.9 GiB/min
        return gib / 54.0
    if cls == "REPRODUCIBLE_COLD":
        # a pack step was measured at ~470 s per candidate, plus the fetch
        return gib / 54.0 + 0.13
    return float("inf")


def uniqueness(row: dict) -> float:
    """0 = another copy exists, 1 = this is the only copy of this science."""
    p = (row.get("path") or "").lower()
    if row.get("hf_repo_id"):
        return 0.0                      # fetchable from a named repo
    if any(s in p for s in ("/receipts/", "negative", "manifest", "capture")):
        return 1.0
    return 0.6                          # unattributed: assume it matters


def reuse_probability(row: dict) -> float:
    """Recently touched artifacts are likely wanted again."""
    atime = row.get("atime") or row.get("mtime")
    if not atime:
        return 0.5
    try:
        t = time.mktime(time.strptime(atime, "%Y-%m-%dT%H:%M:%SZ"))
    except Exception:
        return 0.5
    days = max(0.0, (time.time() - t) / 86400.0)
    if days < 1:
        return 1.0
    if days < 7:
        return 0.7
    if days < 30:
        return 0.4
    return 0.15


def score(row: dict) -> dict:
    """Higher score = better deletion candidate. Bytes reclaimed pull up;
    regeneration cost, uniqueness and likely reuse pull down."""
    gib = row.get("size_gib") or 0.0
    regen = regeneration_cost(row)
    uniq = uniqueness(row)
    reuse = reuse_probability(row)
    if regen == float("inf"):
        value = 0.0
    else:
        value = gib / (1.0 + regen) * (1.0 - uniq) * (1.0 - reuse)
    return {"score": round(value, 4), "gib": gib,
            "regen_hours": (None if regen == float("inf") else round(regen, 2)),
            "uniqueness": uniq, "reuse_probability": reuse}


def is_protected(row: dict) -> tuple[bool, str]:
    cls = row.get("classification")
    path = row.get("path") or ""
    if cls in PROTECTED_CLASSES:
        return True, f"classification {cls} is a §8 never-delete class"
    if cls not in ELIGIBLE_CLASSES:
        return True, f"classification {cls!r} is not in the eligible set {sorted(ELIGIBLE_CLASSES)}"
    low = path.lower()
    for sub in NEVER_SUBSTRINGS:
        if sub in low:
            return True, f"path contains {sub!r} (case-insensitive) — evidence, not bytes"
    if not row.get("hf_repo_id") and cls == "REDOWNLOADABLE":
        return True, "classified REDOWNLOADABLE but carries no repo id, so the route is unproven"
    return False, ""


def select(rows: list, need_gib: float) -> dict:
    """Rank and pick just enough to satisfy `need_gib`. Never exceeds the need —
    reclaiming more than required is how a KEEP_LIST gets eaten."""
    candidates, refused = [], []
    for r in rows:
        prot, why = is_protected(r)
        if prot:
            refused.append({"path": r["path"], "classification": r.get("classification"),
                            "gib": r.get("size_gib"), "reason": why})
            continue
        s = score(r)
        if s["score"] <= 0:
            refused.append({"path": r["path"], "classification": r.get("classification"),
                            "gib": r.get("size_gib"),
                            "reason": "scores zero — regeneration cost is unbounded or it is unique"})
            continue
        candidates.append({**{k: r.get(k) for k in
                              ("path", "size_gib", "classification", "hf_repo_id", "atime")}, **s})
    candidates.sort(key=lambda c: -c["score"])
    picked, got = [], 0.0
    for c in candidates:
        if got >= need_gib:
            break
        picked.append(c)
        got += c["gib"]
    return {"need_gib": need_gib, "selected_gib": round(got, 2), "selected": picked,
            "satisfied": got >= need_gib,
            "considered": len(candidates), "refused": refused}


def assert_deletable(row: dict) -> None:
    """The mechanism a KEEP_LIST was missing. Call before any removal."""
    prot, why = is_protected(row)
    if prot:
        raise ProtectedArtifact(f"refusing to delete {row.get('path')}: {why}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--need-gib", type=float, default=0.0,
                    help="how much space is actually needed; 0 means report only")
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--execute", action="store_true",
                    help="print the removal commands (still does not run them)")
    args = ap.parse_args()

    if not os.path.isfile(args.ledger):
        print(f"FAIL: no artifact ledger at {args.ledger} "
              "(run tools/headless/artifact_census.py)")
        return 2
    led = json.loads(open(args.ledger).read())
    rows = led["artifacts"]

    free_gib = None
    try:
        st = os.statvfs("/System/Volumes/Data")
        free_gib = st.f_bavail * st.f_frsize / GIB
    except Exception:
        pass

    sel = select(rows, args.need_gib)
    doc = {
        "schema": "hawking.headless.storage_selection.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": ("select never deletes. It ranks and refuses. --execute prints commands for a "
                   "human or a supervised WorkUnit to run."),
        "why_conservative": ("G28_STORAGE_RECLAIM.json listed the 51 GB bf16 SOURCE patient and "
                             "the champion 3.3448 BPW artifacts on an explicit KEEP_LIST, and they "
                             "were reclaimed anyway four days later. runs/ is gitignored so the "
                             "deletion left no git trace. A KEEP_LIST in a receipt is a note; this "
                             "module is the mechanism."),
        "disk_free_gib": round(free_gib, 1) if free_gib else None,
        "ledger_total_gib": led.get("total_gib"),
        "by_class": led.get("by_class"),
        "selection": sel,
    }
    out = REPO / "receipts/headless/STORAGE_SELECTION.json"
    out.write_text(json.dumps(doc, indent=1))

    print(f"disk free            {doc['disk_free_gib']} GiB")
    print(f"ledger               {led['artifact_count']} artifacts, {led['total_gib']} GiB")
    for k, v in sorted((led.get("by_class") or {}).items()):
        print(f"  {k:<24} {v['count']:>4}  {v['gib']:>9.1f} GiB"
              + ("   PROTECTED" if k in PROTECTED_CLASSES else ""))
    print(f"\nneed                 {args.need_gib} GiB")
    print(f"eligible candidates  {sel['considered']}")
    print(f"refused              {len(sel['refused'])}")
    print(f"selected             {len(sel['selected'])} artifacts, {sel['selected_gib']} GiB"
          f"  ({'satisfies the need' if sel['satisfied'] else 'DOES NOT satisfy the need'})")
    for c in sel["selected"][:12]:
        print(f"  score={c['score']:<9} {c['gib']:>7.2f} GiB  regen={c['regen_hours']}h  "
              f"{c['path'][:88]}")
    if args.execute:
        print("\n# commands (NOT run by this tool):")
        for c in sel["selected"]:
            print(f"rm -rf {c['path']!r}   # {c['gib']} GiB, "
                  f"refetch: {c.get('hf_repo_id') or 'see ledger'}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

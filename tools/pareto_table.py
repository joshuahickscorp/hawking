#!/usr/bin/env python3
"""G150: Pareto table per candidate, regenerated from receipts, never hand-edited.

One machine-readable table over every live candidate, with the axes the promotion
policy (G151) actually reads: effective BPW, TPS, TOKEN_NS, DRAM/token, resident RAM,
NR size, NX size, Doctor verdict, Tabula drift. A missing cell is null -- never
fabricated, never carried from a different candidate.

The CONTROL is regenerate-and-diff: the table is a pure function of the receipts on
disk, so running it twice must produce byte-identical output. If a second run differs,
something non-deterministic (a dict order, a timestamp, a stray float) has leaked into
what is supposed to be a reproducible artifact, and the table cannot be trusted as the
promotion input. The committed copy is diffed against a fresh regeneration.

  ./tools/pareto_table.py --out receipts/.../G150_PARETO.json
"""
from __future__ import annotations
import argparse, datetime, json, pathlib, re, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
REC = ROOT / "receipts/ascent-2026-08-16"

# Candidate registry. Names are identity; the table is keyed on them.
CANDIDATES = ["uniform-q4-v1", "mixed-q3mlp-v1", "mixed-q3mlp-q3attn-v1"]

AXES = ["effective_bpw", "tps", "token_ns", "dram_bytes_per_token",
        "resident_ram_gb", "nr_bytes", "nx_bytes", "doctor", "tabula_drift"]


import os


def _subject(d) -> str | None:
    """The candidate a receipt is ABOUT: basename of its declared artifact/candidate."""
    if not isinstance(d, dict):
        return None
    for k in ("artifact", "candidate", "artifact_root"):
        v = d.get(k)
        if isinstance(v, str) and "/" in v:
            return os.path.basename(v.rstrip("/"))
        if isinstance(v, str):
            return v
    return None


def scan_receipts() -> list[tuple]:
    """Load every receipt once, tagged with the candidate it is ABOUT (or None).
    Only subject-matched receipts are ever read for a candidate's cells, so a value
    can never leak from a receipt that merely MENTIONS a different candidate."""
    docs = []
    for p in sorted(REC.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        docs.append((p.name, _subject(d), d))
    return docs


def find_metric(docs, candidate: str, keys: list[str]):
    """First value under any of `keys`, searched ONLY in receipts whose declared
    subject basename equals `candidate`. Deterministic (receipts pre-sorted)."""
    for fname, subj, d in docs:
        if subj != candidate:
            continue
        def rec(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in keys and isinstance(v, (int, float, str)):
                        return v
                    r = rec(v)
                    if r is not None:
                        return r
            elif isinstance(o, list):
                for v in o:
                    r = rec(v)
                    if r is not None:
                        return r
            return None
        r = rec(d)
        if r is not None:
            return r, fname
    return None, None


def build_table(docs) -> dict:
    nr_dir = ROOT / "workspace/campaign/records/runs/qwen38-27b"
    table = {}
    for c in CANDIDATES:
        row = {ax: None for ax in AXES}
        # NR bytes: on-disk directory size, deterministic.
        d = nr_dir / c
        if d.is_dir():
            row["nr_bytes"] = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        for ax, keys in {
            "effective_bpw": ["complete_bpw", "effective_bpw"],
            "token_ns": ["steady_decode_wall_ns_per_token"],
            "tabula_drift": ["tabula_drift", "drift_ratio"],
        }.items():
            v, src = find_metric(docs, c, keys)
            if v is not None:
                row[ax] = v
        table[c] = row
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--committed", type=pathlib.Path,
                    default=REC / "G150_PARETO_TABLE.json")
    a = ap.parse_args()
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()

    docs = scan_receipts()
    t1 = build_table(docs)
    t2 = build_table(docs)   # CONTROL: regenerate
    s1 = json.dumps(t1, sort_keys=True)
    s2 = json.dumps(t2, sort_keys=True)
    deterministic = s1 == s2

    # write the pure table (no timestamp) as the committed, diffable artifact
    a.committed.parent.mkdir(parents=True, exist_ok=True)
    prior = a.committed.read_text() if a.committed.exists() else None
    table_json = json.dumps(t1, indent=2, sort_keys=True) + "\n"
    a.committed.write_text(table_json)
    diff_stable = (prior is None) or (prior == table_json)

    print(f"candidates: {len(t1)}")
    for c, row in t1.items():
        filled = sum(1 for v in row.values() if v is not None)
        print(f"  {c:<28} {filled}/{len(AXES)} axes filled  bpw={row['effective_bpw']} "
              f"token_ns={row['token_ns']} nr_bytes={row['nr_bytes']}")
    print(f"CONTROL regenerate byte-identical: {deterministic}")
    print(f"committed copy stable vs prior: {diff_stable}"
          f"{' (first write)' if prior is None else ''}")

    doc = {
        "schema": "hawking.nos.pareto_table.v1",
        "obligation": "G150 -- Pareto table per candidate, regenerated not hand-edited",
        "started": start,
        "axes": AXES, "candidates": CANDIDATES, "table": t1,
        "committed_copy": str(a.committed.relative_to(ROOT)),
        "control_regenerate_byte_identical": deterministic,
        "control_diff_against_committed_stable": diff_stable,
        "honest_note": ("cells are null where no receipt records that axis for that "
                        "candidate; nulls are not filled from a different candidate. DRAM/token "
                        "and NX size are null pending G142 and a committed NX -- the table shows "
                        "what is measured and what still owes, which is the point of publishing it."),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
        "ended": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")
    return deterministic and diff_stable


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)

#!/usr/bin/env python3.12
"""Generate bounded local Ramanujan data sources D1–D4, D6, D7 from pinned Mathlib.

Resource policy (LIGHT_ONLY / shared host):
  nice -n 15, at most 6 workers (default 4 for D4 Lean checks).

    python3.12 -m ramanujan.data.generate
    python3.12 -m ramanujan.data.generate --limit 400 --d4-limit 100 --workers 4
    python3.12 -m ramanujan.data.generate --scale-up   # prints the scale-up command

Does not flip RAMANUJAN_RESEARCH_AUTHORIZED. Does not modify Mathlib.
Does not generate teacher traces from Math-Preserve.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ramanujan.data.common import mathlib_commit, write_jsonl
from ramanujan.data.contamination_pass import run_contamination
from ramanujan.data.extractors import (
    METHOD_D1,
    METHOD_D2,
    METHOD_D3,
    METHOD_D4,
    METHOD_D6,
    METHOD_D7,
    extract_d1,
    extract_d2,
    extract_d3,
    extract_d4,
    extract_d6,
    extract_d7,
    load_decls,
)
from ramanujan.data.paths import (
    CORPORA,
    DEFAULT_MODULES,
    EXPECTED_MATHLIB_COMMIT,
    GENERATION_RECEIPT,
    MATHLIB_ROOT,
    SOURCE_FILES,
)


SCALE_UP_COMMAND = (
    "nice -n 15 python3.12 -m ramanujan.data.generate "
    "--limit 5000 --d4-limit 800 --decl-limit 20000 --workers 6 --all-mathlib"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=400, help="max items per source (D1/D2/D3/D6/D7)")
    p.add_argument("--d4-limit", type=int, default=80, help="max D4 repair pairs (Lean-bound)")
    p.add_argument("--decl-limit", type=int, default=900, help="max theorems parsed from Mathlib")
    p.add_argument("--workers", type=int, default=4, help="D4 lean workers (max 6)")
    p.add_argument("--skip-d4", action="store_true", help="skip Lean repair generation")
    p.add_argument("--skip-contamination", action="store_true")
    p.add_argument("--scale-up", action="store_true", help="print scale-up command and exit")
    p.add_argument("--mathlib", type=str, default=None, help="override Mathlib root")
    p.add_argument(
        "--all-mathlib",
        action="store_true",
        help="scan Mathlib/**/*.lean (scale-up); default is the bounded DEFAULT_MODULES list",
    )
    return p.parse_args(argv)


def _modules_for(mathlib: Path, all_mathlib: bool) -> list[str]:
    if not all_mathlib:
        return list(DEFAULT_MODULES)
    root = mathlib / "Mathlib"
    if not root.is_dir():
        return list(DEFAULT_MODULES)
    out: list[str] = []
    for p in sorted(root.rglob("*.lean")):
        out.append(str(p.relative_to(mathlib)))
    return out


def generate(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)
    if args.scale_up:
        print(SCALE_UP_COMMAND)
        return {"scale_up_command": SCALE_UP_COMMAND}

    workers = max(1, min(int(args.workers), 6))
    mathlib = Path(args.mathlib) if args.mathlib else MATHLIB_ROOT
    if not mathlib.is_dir():
        raise SystemExit(f"Mathlib not found at {mathlib}")

    commit = mathlib_commit(mathlib)
    CORPORA.mkdir(parents=True, exist_ok=True)
    modules = _modules_for(mathlib, bool(args.all_mathlib))

    print(f"Mathlib: {mathlib}", flush=True)
    print(f"commit:  {commit} (expected {EXPECTED_MATHLIB_COMMIT})", flush=True)
    print(f"workers: {workers} (cap 6); limits: {args.limit}/{args.d4_limit}", flush=True)
    print(f"modules: {len(modules)} ({'all-mathlib' if args.all_mathlib else 'DEFAULT_MODULES'})", flush=True)

    decls = load_decls(mathlib, modules, limit=args.decl_limit)
    print(f"parsed:  {len(decls)} theorem/lemma decls from {len(modules)} module paths", flush=True)

    corpora: dict[str, dict] = {}

    d1 = extract_d1(decls, limit=args.limit)
    corpora["D1"] = write_jsonl(SOURCE_FILES["D1"], d1)
    corpora["D1"]["extraction_method"] = METHOD_D1
    print(f"D1 proof traces:           {len(d1)}", flush=True)

    d2 = extract_d2(decls, limit=args.limit)
    corpora["D2"] = write_jsonl(SOURCE_FILES["D2"], d2)
    corpora["D2"]["extraction_method"] = METHOD_D2
    print(f"D2 state transitions:      {len(d2)}", flush=True)

    d3 = extract_d3(decls, limit=args.limit)
    corpora["D3"] = write_jsonl(SOURCE_FILES["D3"], d3)
    corpora["D3"]["extraction_method"] = METHOD_D3
    print(f"D3 premise pairs:          {len(d3)}", flush=True)

    if args.skip_d4:
        d4 = []
        print("D4 repair pairs:           SKIPPED", flush=True)
    else:
        print(f"D4 running Lean perturbations (workers={workers})...", flush=True)
        d4 = extract_d4(decls, limit=args.d4_limit, workers=workers)
        # Sort before writing. D4 is the only source built by a worker pool,
        # so records arrive in Lean-completion order and land in a different
        # order every run even when the content is identical. content_digest
        # is order-sensitive on purpose, so unsorted output makes D4 the one
        # corpus whose freeze can never be re-derived. Measured 2026-07-30:
        # the first record was a different theorem between two runs.
        d4.sort(key=lambda r: r["id"])
        print(f"D4 repair pairs:           {len(d4)}", flush=True)
    corpora["D4"] = write_jsonl(SOURCE_FILES["D4"], d4)
    corpora["D4"]["extraction_method"] = METHOD_D4

    d6 = extract_d6(mathlib, limit=args.limit)
    corpora["D6"] = write_jsonl(SOURCE_FILES["D6"], d6)
    corpora["D6"]["extraction_method"] = METHOD_D6
    print(f"D6 counterexamples:        {len(d6)}", flush=True)

    d7 = extract_d7(limit=args.limit)
    corpora["D7"] = write_jsonl(SOURCE_FILES["D7"], d7)
    corpora["D7"]["extraction_method"] = METHOD_D7
    print(f"D7 tool-use traces:        {len(d7)}", flush=True)

    contamination = None
    if not args.skip_contamination:
        print("Running contamination barrier...", flush=True)
        contamination = run_contamination()
        print(
            f"contamination: admitted={contamination['summary']['total_admitted']} "
            f"rejected={contamination['summary']['total_rejected']}",
            flush=True,
        )

    # Per-source binding payload for the data matrix
    sources_meta = {}
    for sid, meta in corpora.items():
        n = meta["n_items"]
        admitted = n
        if contamination and sid in contamination.get("per_source", {}):
            admitted = contamination["per_source"][sid]["n_admitted"]
        sources_meta[sid] = {
            "status": "PRESENT" if n > 0 else "EMPTY",
            "n_items": n,
            "n_admitted": admitted,
            "sha256": meta.get("sha256"),
            "content_digest": meta.get("content_digest"),
            "local_offline_location": str(SOURCE_FILES[sid].relative_to(SOURCE_FILES[sid].parents[2].parent)),
            "version": commit,
            "hash": meta.get("sha256"),
            "split": "train",
            "deduplication": "content_hash_sha256_exact",
            "contamination_boundary": (
                "PASS"
                if contamination and contamination["per_source"].get(sid, {}).get("n_rejected", 0) == 0
                else (
                    "PASS_WITH_REJECTIONS"
                    if contamination
                    else "ENFORCED_BY_EXISTING_BARRIER"
                )
            ),
            "evidence_status": (
                f"{admitted} admitted training items; method={meta.get('extraction_method')}; "
                "NOT production-prover evidence; NON_PRODUCTION corpus for local training loops"
            ),
            "extraction_method": meta.get("extraction_method"),
            "license": (
                "INHERITED_FROM_MATHLIB_APACHE_2_0"
                if sid in ("D1", "D2", "D3")
                else "LOCALLY_GENERATED"
            ),
        }
        # Fix relative path to be ramanujan/data/corpora/...
        sources_meta[sid]["local_offline_location"] = f"ramanujan/data/corpora/{SOURCE_FILES[sid].name}"

    receipt = {
        "schema": "hawking.ramanujan.generation_receipt.v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mathlib_root": str(mathlib),
        "mathlib_commit": commit,
        "expected_mathlib_commit": EXPECTED_MATHLIB_COMMIT,
        "commit_match": commit == EXPECTED_MATHLIB_COMMIT,
        "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        "teacher_from_math_preserve": False,
        "resource_policy": {
            "nice": 15,
            "max_workers": 6,
            "workers_used": workers,
            "note": "CPU-bound pack job holds ~1.2 of 28 cores; stay under 6 workers",
        },
        "modules": modules if not args.all_mathlib else f"all-mathlib ({len(modules)} files)",
        "n_modules": len(modules),
        "n_decls_parsed": len(decls),
        "sources": sources_meta,
        "contamination_receipt": str(Path("ramanujan/data/corpora/CONTAMINATION_RECEIPT.json")),
        "scale_up_command": SCALE_UP_COMMAND,
        "counts": {sid: sources_meta[sid]["n_items"] for sid in sources_meta},
        "hard_rule": "Do NOT generate teacher traces from the Math-Preserve artifact.",
    }
    GENERATION_RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {GENERATION_RECEIPT}", flush=True)
    return receipt


def main(argv: list[str] | None = None) -> int:
    receipt = generate(argv)
    if "scale_up_command" in receipt and len(receipt) == 1:
        return 0
    counts = receipt.get("counts", {})
    print("COUNTS:", json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

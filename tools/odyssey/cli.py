#!/usr/bin/env python3
"""CLI for Odyssey data inventory, membership check, ingest, and barrier report.

No network. No downloads. No training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.odyssey._paths import (  # noqa: E402
    DATA_DIR,
    FIXTURE_DIR,
    MEMBERSHIP_DIR,
)
from tools.odyssey.contamination import (  # noqa: E402
    barrier_rules_document,
    build_barrier,
)
from tools.odyssey.ingest import ingest_corpus  # noqa: E402
from tools.odyssey.inventory import build_inventory, write_inventory  # noqa: E402
from tools.odyssey.teacher_assess import assess_teacher_traces  # noqa: E402


def cmd_inventory(args: argparse.Namespace) -> int:
    inv = write_inventory(Path(args.out) if args.out else None)
    print(json.dumps({"status": inv["status"], "summary": inv["summary"]}, indent=2))
    return 0


def cmd_membership_check(_args: argparse.Namespace) -> int:
    inv = build_inventory()
    declared = [c for c in inv["corpora"] if c.get("role") == "train" and c["id"] in {
        "math-core", "support-language", "long-horizon", "sovereignty-corpus"
    }]
    # Prefer the declared corpora section only
    from tools.odyssey.inventory import check_declared_corpora

    declared = check_declared_corpora()
    report = {
        "schema": "hawking.odyssey.membership_check.v1",
        "corpora": declared,
        "n_present": sum(1 for c in declared if c["status"] == "PRESENT"),
        "n_declared_not_present": sum(
            1 for c in declared if c["status"] == "DECLARED_NOT_PRESENT"
        ),
        "n_partial": sum(1 for c in declared if c["status"] == "PARTIAL"),
    }
    out = DATA_DIR / "ODYSSEY_MEMBERSHIP_CHECK.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_ingest_fixture(args: argparse.Namespace) -> int:
    raw = Path(args.raw) if args.raw else FIXTURE_DIR / "raw_fixture.jsonl"
    result = ingest_corpus(
        raw,
        corpus_id=args.corpus_id,
        role="fixture",
        out_dir=Path(args.out_dir) if args.out_dir else MEMBERSHIP_DIR / args.corpus_id,
        licence="synthetic-fixture-not-for-training",
        note="FIXTURE: proves pipeline only; not an Odyssey training corpus",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    # Fixture must reject the deliberate leaks.
    if result["n_rejected_contamination"] < 1:
        print("ERROR: expected contamination rejections on fixture leaks", file=sys.stderr)
        return 1
    return 0


def cmd_barrier_report(args: argparse.Namespace) -> int:
    barrier = build_barrier()
    rules = barrier_rules_document(barrier)

    # Run deliberate leak probes against sealed eval.
    leaks = [
        {
            "id": "leak_exact_tl02",
            "text": (
                'What does BPW stand for in model compression / quantization? '
                'Answer briefly and include the exact phrase "bits per weight".'
            ),
        },
        {
            "id": "leak_near_tl01",
            "text": (
                "In one or two sentences, define what an idempotent HTTP method means. "
                "Include the word \"idempotent\" and the idea that repeating the request "
                "does not change resource state further."
            ),
        },
        {
            "id": "leak_near_hidden_batball",
            "text": (
                "A bat and a ball cost $1.10 in total. The bat costs one dollar more "
                "than the ball. How much does the ball cost?"
            ),
        },
        {
            "id": "clean_control",
            "text": (
                "FIXTURE control: content-addressed membership does not overlap any "
                "evaluation prompt on this host."
            ),
        },
    ]
    leak_results = []
    all_leaks_rejected = True
    clean_admitted = True
    for case in leaks:
        hits = barrier.check(case["text"])
        rejected = bool(hits)
        if case["id"].startswith("leak_") and not rejected:
            all_leaks_rejected = False
        if case["id"] == "clean_control" and rejected:
            clean_admitted = False
        leak_results.append(
            {
                "id": case["id"],
                "rejected": rejected,
                "n_hits": len(hits),
                "hits": [
                    {
                        "reason": h.reason,
                        "eval_source": h.eval_source,
                        "eval_id": h.eval_id,
                        "jaccard": h.jaccard,
                    }
                    for h in hits
                ],
            }
        )

    # Also run full fixture ingest for end-to-end evidence.
    fixture_raw = FIXTURE_DIR / "raw_fixture.jsonl"
    ingest_result = None
    if fixture_raw.is_file():
        ingest_result = ingest_corpus(
            fixture_raw,
            corpus_id="ingestion_fixture_v0",
            role="fixture",
            licence="synthetic-fixture-not-for-training",
            note="FIXTURE end-to-end barrier proof",
        )

    report = {
        **rules,
        "leak_test": {
            "description": (
                "Deliberate training items that are exact or near-duplicates of held-out "
                "evaluation items must be rejected. A clean control must be admitted."
            ),
            "cases": leak_results,
            "all_deliberate_leaks_rejected": all_leaks_rejected,
            "clean_control_admitted": clean_admitted,
            "result": "PASS" if all_leaks_rejected and clean_admitted else "FAIL",
        },
        "fixture_ingest": ingest_result,
        "status": "PASS" if all_leaks_rejected and clean_admitted else "FAIL",
    }
    out = Path(args.out) if args.out else DATA_DIR / "ODYSSEY_CONTAMINATION_BARRIER.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "leak_test": report["leak_test"]["result"]}, indent=2))
    return 0 if report["status"] == "PASS" else 1


def cmd_teacher(_args: argparse.Namespace) -> int:
    report = assess_teacher_traces()
    out = DATA_DIR / "ODYSSEY_TEACHER_ASSESSMENT.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report.get("status"), "gap": report.get("gap", {}).get("numeric_gap")}, indent=2))
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    rc = 0
    rc |= cmd_inventory(args)
    rc |= cmd_membership_check(args)
    rc |= cmd_teacher(args)
    rc |= cmd_barrier_report(args)
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inventory", help="Write ODYSSEY_DATA_INVENTORY.json")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_inventory)

    s = sub.add_parser("membership-check", help="Mechanical PRESENT/DECLARED_NOT_PRESENT")
    s.set_defaults(func=cmd_membership_check)

    s = sub.add_parser("ingest-fixture", help="Run ingestion on the labelled fixture")
    s.add_argument("--raw", default=None)
    s.add_argument("--corpus-id", default="ingestion_fixture_v0")
    s.add_argument("--out-dir", default=None)
    s.set_defaults(func=cmd_ingest_fixture)

    s = sub.add_parser("barrier-report", help="Write ODYSSEY_CONTAMINATION_BARRIER.json")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_barrier_report)

    s = sub.add_parser("teacher", help="Assess teacher traces")
    s.set_defaults(func=cmd_teacher)

    s = sub.add_parser("all", help="Inventory + membership + teacher + barrier")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_all)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

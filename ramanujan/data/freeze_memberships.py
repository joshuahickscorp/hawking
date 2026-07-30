#!/usr/bin/env python3.12
"""Freeze train/dev/test memberships for Ramanujan corpora.

The split is sealed by hash so it cannot drift later. Assignment is a pure
function of each item's existing `content_hash` (bucketed into 80/10/10).

The generation-time `split` field on every item is currently the provisional
value "train". That field is NOT mutated here: mutating it would change
`content_hash` (which includes `split` in the stamped body). The sealed
MEMBERSHIP_MANIFEST is the sole authority for train/dev/test membership.

Also re-runs the Odyssey contamination barrier and records the negative
control: a sealed support-halo item presented as training text must come
back `exact_match`.

    nice -n 15 python3.12 -m ramanujan.data.freeze_memberships

Does not flip RAMANUJAN_RESEARCH_AUTHORIZED. Does not touch Math-Preserve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from ramanujan.data.common import file_sha256, read_jsonl
from ramanujan.data.paths import CORPORA, ROOT, SOURCE_FILES

SCHEMA = "hawking.ramanujan.membership_manifest.v1"
SEED_LABEL = "ramanujan-freeze-v1-content_hash_bucket"
# 80 / 10 / 10 by content_hash bucket (stable forever for a given hash).
TRAIN_PCT = 80
DEV_PCT = 10  # buckets [80, 90)
# TEST_PCT = 10  # buckets [90, 100)

MANIFEST_PATH = CORPORA / "MEMBERSHIP_MANIFEST.json"
FREEZE_RECEIPT = CORPORA / "FREEZE_RECEIPT.json"


def assign_split(content_hash: str) -> str:
    """Deterministic split from content_hash. Pure; no RNG, no wall clock."""
    if not content_hash or len(content_hash) < 8:
        raise ValueError(f"content_hash too short for freeze assignment: {content_hash!r}")
    bucket = int(content_hash[:8], 16) % 100
    if bucket < TRAIN_PCT:
        return "train"
    if bucket < TRAIN_PCT + DEV_PCT:
        return "dev"
    return "test"


def _canonical_assignment_blob(per_source: dict[str, Any]) -> bytes:
    """Canonical bytes over which membership_sha256 is taken.

    Only (source_id, content_hash, split) triples, sorted. Counts and paths
    are derived and must not affect the seal.
    """
    triples: list[list[str]] = []
    for sid in sorted(per_source):
        block = per_source[sid]
        for split_name in ("train", "dev", "test"):
            for h in block["by_split"][split_name]:
                triples.append([sid, h, split_name])
    triples.sort()
    return json.dumps(
        {
            "seed_label": SEED_LABEL,
            "train_pct": TRAIN_PCT,
            "dev_pct": DEV_PCT,
            "triples": triples,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def freeze_memberships(
    *,
    source_files: dict[str, Path] | None = None,
    manifest_path: Path | None = None,
    receipt_path: Path | None = None,
    run_contamination: bool = True,
) -> dict[str, Any]:
    files = source_files or SOURCE_FILES
    per_source: dict[str, Any] = {}
    global_counts = {"train": 0, "dev": 0, "test": 0, "total": 0}
    all_hashes: list[str] = []

    for sid, path in files.items():
        items = read_jsonl(path) if path.is_file() else []
        by_split: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
        id_to_split: dict[str, str] = {}
        missing_hash = 0
        for it in items:
            h = it.get("content_hash")
            if not h:
                missing_hash += 1
                continue
            split = assign_split(h)
            by_split[split].append(h)
            id_to_split[str(it.get("id"))] = split
            all_hashes.append(h)
            global_counts[split] += 1
            global_counts["total"] += 1
        for k in by_split:
            by_split[k] = sorted(by_split[k])
        try:
            rel = str(path.relative_to(ROOT))
        except ValueError:
            rel = str(path)
        per_source[sid] = {
            "path": rel,
            "file_sha256": file_sha256(path) if path.is_file() else None,
            "n_items": len(items),
            "n_missing_content_hash": missing_hash,
            "n": {k: len(v) for k, v in by_split.items()},
            "by_split": by_split,
            # id map is audit-only; seal does not cover it (ids are not the address).
            "id_to_split_sample": dict(list(id_to_split.items())[:5]),
            "n_ids_mapped": len(id_to_split),
        }

    assignment_blob = _canonical_assignment_blob(per_source)
    membership_sha256 = hashlib.sha256(assignment_blob).hexdigest()

    # Corpus-level hash of all content hashes (order-independent via sort).
    corpus_items_sha256 = hashlib.sha256(
        ("\n".join(sorted(all_hashes)) + "\n").encode("utf-8")
    ).hexdigest()

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_label": SEED_LABEL,
        "assignment_rule": (
            f"bucket = int(content_hash[:8], 16) % 100; "
            f"[0,{TRAIN_PCT})->train, [{TRAIN_PCT},{TRAIN_PCT + DEV_PCT})->dev, "
            f"else->test"
        ),
        "ratios": {"train": TRAIN_PCT / 100, "dev": DEV_PCT / 100, "test": (100 - TRAIN_PCT - DEV_PCT) / 100},
        "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        "teacher_from_math_preserve": False,
        "note": (
            "MEMBERSHIP_MANIFEST is the sole authority for train/dev/test. "
            "Item bodies keep generation-time split='train' so content_hash "
            "identity stays stable; loaders must use this manifest."
        ),
        "counts": global_counts,
        "corpus_items_sha256": corpus_items_sha256,
        "membership_sha256": membership_sha256,
        "per_source": per_source,
        "invariants": [
            "assignment is a pure function of content_hash",
            "membership_sha256 seals (source, content_hash, split) triples only",
            "no training loop may read dev/test hashes as train",
            "support-halo / hidden memberships remain outside this freeze",
        ],
    }

    out = manifest_path or MANIFEST_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_file_sha256 = file_sha256(out)

    # Contamination barrier + negative control.
    contamination_summary: dict[str, Any] = {"ran": False}
    negative_control: dict[str, Any] = {"ran": False}
    if run_contamination:
        from tools.odyssey.contamination import (
            barrier_rules_document,
            build_barrier,
            verify_support_halo_seal,
        )
        from tools.odyssey.normalize import extract_comparison_text
        from tools.odyssey._paths import SUPPORT_HALO_CORPUS

        seal = verify_support_halo_seal()
        barrier = build_barrier()
        rules = barrier_rules_document(barrier)

        total_in = 0
        total_rejected = 0
        for sid, path in files.items():
            for it in read_jsonl(path) if path.is_file() else []:
                total_in += 1
                text = it.get("text") or it.get("statement") or it.get("goal") or ""
                if barrier.check(str(text)):
                    total_rejected += 1

        # Negative control: sealed support-halo item must exact-match.
        halo_items = [
            json.loads(ln)
            for ln in SUPPORT_HALO_CORPUS.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        # Prefer the known tl02_bpw probe; fall back to first item.
        probe = next((x for x in halo_items if x.get("id") == "tl02_bpw"), halo_items[0])
        probe_text = extract_comparison_text(probe)
        hits = barrier.check(probe_text)
        negative_control = {
            "ran": True,
            "description": (
                f"sealed support-halo item ({probe.get('id')}) presented as training text"
            ),
            "probe_id": probe.get("id"),
            "rejected": bool(hits),
            "reason": hits[0].reason if hits else None,
            "eval_source": hits[0].eval_source if hits else None,
            "eval_id": hits[0].eval_id if hits else None,
            "pass": bool(hits) and any(h.reason == "exact_match" for h in hits),
        }
        contamination_summary = {
            "ran": True,
            "support_halo_seal_ok": bool(seal.get("ok")),
            "support_halo_sha256": seal.get("computed_sha256"),
            "n_eval_items_indexed": len(barrier.eval_items),
            "total_corpus_items_checked": total_in,
            "total_rejected_vs_eval": total_rejected,
            "barrier": rules,
            "negative_control": negative_control,
        }

    # Verify seal recompute.
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    recompute = hashlib.sha256(
        _canonical_assignment_blob(reloaded["per_source"])
    ).hexdigest()
    seal_ok = recompute == membership_sha256

    receipt = {
        "schema": "hawking.ramanujan.freeze_receipt.v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_path": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
        "manifest_file_sha256": manifest_file_sha256,
        "membership_sha256": membership_sha256,
        "membership_seal_recomputed_ok": seal_ok,
        "counts": global_counts,
        "contamination": contamination_summary,
        "negative_control_pass": bool(negative_control.get("pass")),
        "RAMANUJAN_RESEARCH_AUTHORIZED": False,
        "status": (
            "PASS"
            if seal_ok and (not run_contamination or negative_control.get("pass"))
            else "FAIL"
        ),
    }
    rpath = receipt_path or FREEZE_RECEIPT
    rpath.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": manifest, "receipt": receipt}


def load_membership(path: Path | None = None) -> dict[str, Any]:
    p = path or MANIFEST_PATH
    if not p.is_file():
        raise FileNotFoundError(f"membership manifest missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def verify_membership_seal(path: Path | None = None) -> dict[str, Any]:
    m = load_membership(path)
    recomputed = hashlib.sha256(_canonical_assignment_blob(m["per_source"])).hexdigest()
    ok = recomputed == m.get("membership_sha256")
    return {
        "ok": ok,
        "committed": m.get("membership_sha256"),
        "recomputed": recomputed,
        "counts": m.get("counts"),
    }


def split_of(content_hash: str, membership: dict[str, Any] | None = None) -> str:
    """Look up split for a content_hash. Falls back to pure assign_split."""
    if membership is None:
        return assign_split(content_hash)
    for sid, block in membership.get("per_source", {}).items():
        for split_name, hashes in block.get("by_split", {}).items():
            # linear ok for 16k; callers should use index_for_source for bulk
            if content_hash in hashes:
                return split_name
    return assign_split(content_hash)


def index_for_source(membership: dict[str, Any], source_id: str) -> dict[str, str]:
    """content_hash -> split for one source."""
    block = membership["per_source"][source_id]
    out: dict[str, str] = {}
    for split_name, hashes in block["by_split"].items():
        for h in hashes:
            out[h] = split_name
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-contamination", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args(argv)

    if args.verify_only:
        result = verify_membership_seal()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1

    out = freeze_memberships(run_contamination=not args.skip_contamination)
    receipt = out["receipt"]
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

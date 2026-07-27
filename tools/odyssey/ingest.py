"""Ingestion pipeline: raw corpus → normalized, deduped, content-addressed, barrier-checked set.

Proved end-to-end on a labelled synthetic fixture. Does not invent training
corpora; does not download anything.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.odyssey._paths import (
    JACCARD_WITHIN_CORPUS_NEAR_DUP,
    MEMBERSHIP_DIR,
    ROOT,
)
from tools.odyssey.contamination import Barrier, ContaminationHit, build_barrier
from tools.odyssey.dedup import exact_dedup_indices, find_near_duplicates
from tools.odyssey.membership import CorpusMembership, make_item_record
from tools.odyssey.normalize import extract_comparison_text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i}: {e}") from e
    return items


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")


def _hit_dicts(hits: list[ContaminationHit]) -> list[dict[str, Any]]:
    out = []
    for h in hits:
        out.append(
            {
                "reason": h.reason,
                "eval_source": h.eval_source,
                "eval_id": h.eval_id,
                "jaccard": h.jaccard,
                "train_hash": h.train_hash,
                "eval_hash": h.eval_hash,
            }
        )
    return out


def ingest_corpus(
    raw_path: Path,
    *,
    corpus_id: str,
    role: str = "train",
    out_dir: Path | None = None,
    barrier: Barrier | None = None,
    licence: str | None = None,
    note: str | None = None,
    within_near_dup_threshold: float = JACCARD_WITHIN_CORPUS_NEAR_DUP,
) -> dict[str, Any]:
    """Run the full pipeline. Returns a result dict; writes admitted JSONL + membership."""
    raw_path = Path(raw_path)
    items = load_jsonl(raw_path)
    if barrier is None:
        barrier = build_barrier()

    texts = [extract_comparison_text(it) for it in items]
    keep_exact, exact_groups = exact_dedup_indices(texts)
    exact_dup_indices = set(range(len(items))) - set(keep_exact)

    # Near-dup only among exact-survivors; drop later occurrences.
    survivor_idxs = sorted(keep_exact)
    survivor_texts = [texts[i] for i in survivor_idxs]
    survivor_ids = [str(items[i].get("id") or i) for i in survivor_idxs]
    near_hits = find_near_duplicates(
        survivor_texts,
        threshold=within_near_dup_threshold,
        ids=survivor_ids,
    )
    near_dup_drop: set[int] = set()
    for local_i, hits in near_hits.items():
        global_i = survivor_idxs[local_i]
        for hit in hits:
            other_global = survivor_idxs[hit.other_index]
            # Keep the lower index; drop the higher.
            if other_global > global_i:
                near_dup_drop.add(other_global)
            elif global_i > other_global:
                near_dup_drop.add(global_i)

    membership = CorpusMembership(
        corpus_id=corpus_id,
        role=role,
        source_path=str(raw_path.relative_to(ROOT)) if raw_path.is_relative_to(ROOT) else str(raw_path),
        n_input=len(items),
        licence=licence,
        note=note,
    )
    admitted: list[dict[str, Any]] = []
    leak_tests: list[dict[str, Any]] = []

    for i, item in enumerate(items):
        text = texts[i]
        rec_base = dict(item)

        if i in exact_dup_indices:
            group = exact_groups[next(h for h, idxs in exact_groups.items() if i in idxs)]
            rec = make_item_record(
                rec_base,
                status="rejected_exact_dup",
                reasons=[f"exact duplicate of earlier item index {group[0]}"],
            )
            membership.items.append(rec)
            membership.n_rejected_exact_dup += 1
            continue

        if i in near_dup_drop:
            rec = make_item_record(
                rec_base,
                status="rejected_near_dup",
                reasons=[f"near-duplicate within corpus (Jaccard ≥ {within_near_dup_threshold})"],
            )
            membership.items.append(rec)
            membership.n_rejected_near_dup += 1
            continue

        contam = barrier.check(text)
        if contam:
            rec = make_item_record(
                rec_base,
                status="rejected_contamination",
                reasons=["train/eval contamination barrier"],
                contamination=_hit_dicts(contam),
            )
            membership.items.append(rec)
            membership.n_rejected_contamination += 1
            leak_tests.append(
                {
                    "source_id": item.get("id"),
                    "rejected": True,
                    "hits": _hit_dicts(contam),
                }
            )
            continue

        rec = make_item_record(rec_base, status="admitted", reasons=[])
        membership.items.append(rec)
        membership.n_admitted += 1
        out_item = dict(item)
        out_item["content_sha256"] = rec["content_sha256"]
        out_item["exact_text_sha256"] = rec["exact_text_sha256"]
        out_item["membership_status"] = "admitted"
        admitted.append(out_item)

    out_dir = Path(out_dir) if out_dir else MEMBERSHIP_DIR / corpus_id
    out_dir.mkdir(parents=True, exist_ok=True)
    admitted_path = out_dir / "admitted.jsonl"
    membership_path = out_dir / "MEMBERSHIP.json"
    rejected_path = out_dir / "rejected_summary.json"
    write_jsonl(admitted_path, admitted)
    membership.write(membership_path)
    rejected_summary = {
        "n_input": membership.n_input,
        "n_admitted": membership.n_admitted,
        "n_rejected_exact_dup": membership.n_rejected_exact_dup,
        "n_rejected_near_dup": membership.n_rejected_near_dup,
        "n_rejected_contamination": membership.n_rejected_contamination,
        "contamination_rejections": leak_tests,
        "admitted_path": str(admitted_path.relative_to(ROOT))
        if admitted_path.is_relative_to(ROOT)
        else str(admitted_path),
        "membership_path": str(membership_path.relative_to(ROOT))
        if membership_path.is_relative_to(ROOT)
        else str(membership_path),
        "membership_sha256": membership.membership_sha256,
    }
    rejected_path.write_text(
        json.dumps(rejected_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "corpus_id": corpus_id,
        "role": role,
        "n_input": membership.n_input,
        "n_admitted": membership.n_admitted,
        "n_rejected_exact_dup": membership.n_rejected_exact_dup,
        "n_rejected_near_dup": membership.n_rejected_near_dup,
        "n_rejected_contamination": membership.n_rejected_contamination,
        "admitted_path": str(admitted_path),
        "membership_path": str(membership_path),
        "membership_sha256": membership.membership_sha256,
        "contamination_rejections": leak_tests,
        "barrier": {
            "n_eval_indexed": len(barrier.eval_items),
            "support_halo_seal_ok": barrier.corpus_sha256_ok,
            "hidden_commitment_ok": barrier.hidden_commitment_ok,
            "sources": list(barrier.sources_loaded),
        },
    }

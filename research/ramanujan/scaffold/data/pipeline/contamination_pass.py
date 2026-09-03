"""Run the Odyssey contamination barrier over Ramanujan local corpora."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ramanujan.data.common import read_jsonl
from ramanujan.data.paths import CONTAMINATION_RECEIPT, SOURCE_FILES


def _comparison_text(item: dict[str, Any]) -> str:
    if item.get("text"):
        return str(item["text"])
    if item.get("statement"):
        return str(item["statement"])
    if item.get("false_statement"):
        return str(item["false_statement"])
    if item.get("goal"):
        return str(item["goal"])
    return json.dumps(item, sort_keys=True)


def run_contamination(
    *,
    source_files: dict[str, Path] | None = None,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    from tools.odyssey.contamination import barrier_rules_document, build_barrier

    files = source_files or SOURCE_FILES
    barrier = build_barrier()
    rules = barrier_rules_document(barrier)

    per_source: dict[str, Any] = {}
    total_in = 0
    total_admitted = 0
    total_rejected = 0
    rejections: list[dict[str, Any]] = []

    for sid, path in files.items():
        items = read_jsonl(path) if path.is_file() else []
        admitted: list[dict[str, Any]] = []
        rejected_here: list[dict[str, Any]] = []
        for it in items:
            total_in += 1
            text = _comparison_text(it)
            hits = barrier.check(text)
            if hits:
                total_rejected += 1
                rec = {
                    "source_id": sid,
                    "item_id": it.get("id"),
                    "content_hash": it.get("content_hash"),
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
                rejected_here.append(rec)
                rejections.append(rec)
                it["admitted"] = False
                it["contamination_hits"] = rec["hits"]
            else:
                total_admitted += 1
                it["admitted"] = True
                admitted.append(it)
        # rewrite corpus with admitted flags; keep rejected for audit but mark them
        if path.is_file() and items:
            path.write_text(
                "".join(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n" for it in items),
                encoding="utf-8",
            )
        per_source[sid] = {
            "path": str(path),
            "n_input": len(items),
            "n_admitted": len(admitted),
            "n_rejected": len(rejected_here),
        }

    receipt = {
        "schema": "hawking.ramanujan.contamination_receipt.v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "barrier": rules,
        "summary": {
            "total_input": total_in,
            "total_admitted": total_admitted,
            "total_rejected": total_rejected,
            "all_admitted": total_rejected == 0,
        },
        "per_source": per_source,
        "rejections": rejections[:50],  # bounded audit sample
    }
    out = receipt_path or CONTAMINATION_RECEIPT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt

#!/usr/bin/env python3
"""Generate benchmark workload configs from the public pain radar ledger."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEDGER = ROOT / "docs/research/pain_radar/ledger.jsonl"
DEFAULT_OUT_DIR = ROOT / "tools/bench/workloads/generated"


PAIN_TO_WORKLOAD: dict[str, dict[str, Any]] = {
    "long_context_reprocessing": {
        "kind": "shared_agent",
        "prompt_token_targets": [8192, 32768, 65536],
        "description": "Repeated long shared prefix plus small user deltas.",
    },
    "cache_invalidation_opacity": {
        "kind": "cache_miss_taxonomy",
        "prompt_token_targets": [4096, 8192, 32768],
        "description": "Exact, near, and divergent prefix cases with miss reasons.",
    },
    "high_concurrency_collapse": {
        "kind": "shared_agent",
        "prompt_token_targets": [4096, 8192],
        "concurrency": [1, 2, 4, 8, 16],
        "description": "Same prefix across many active conversations.",
    },
    "long_prompt_starvation": {
        "kind": "mixed_latency",
        "prompt_token_targets": [1024, 65536],
        "description": "One long prefill mixed with short interactive requests.",
    },
    "metal_backend_gap": {
        "kind": "apple_backend_gate",
        "prompt_token_targets": [1024, 8192, 32768],
        "description": "Apple-local backend smoke/stress matrix.",
    },
    "spec_decode_regression": {
        "kind": "spec_decode_gate",
        "prompt_token_targets": [1024, 8192],
        "description": "Spec on/off net throughput and TTFT gate.",
    },
    "memory_cliff": {
        "kind": "memory_budget_matrix",
        "prompt_token_targets": [8192, 32768, 65536, 131072],
        "description": "KV/state resident byte budget and eviction behavior.",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "workload"


def workload_for(row: dict[str, Any]) -> dict[str, Any]:
    pain_class = str(row.get("pain_class") or "unknown")
    template = dict(PAIN_TO_WORKLOAD.get(pain_class, {
        "kind": "shared_agent",
        "prompt_token_targets": [4096, 8192],
        "description": "Manual triage workload.",
    }))
    engine = str(row.get("engine") or "unknown")
    title = str(row.get("title") or row.get("url") or pain_class)
    slug = slugify(f"{engine}_{pain_class}_{title}")[:96]
    return {
        "schema_version": 1,
        "name": slug,
        "kind": template["kind"],
        "description": template["description"],
        "source_issue": {
            "url": row.get("url"),
            "engine": engine,
            "title": title,
            "pain_class": pain_class,
            "score": row.get("score"),
        },
        "prompt_token_targets": template.get("prompt_token_targets", [8192]),
        "concurrency": template.get("concurrency", [1, 2, 4, 8]),
        "decode_tokens": 128,
        "temperature": 0.0,
        "shared_context": [
            "Repository map: src/, crates/, tools/, docs/.",
            "Recent transcript: user asks for iterative code changes and benchmark evidence.",
            "Constraint: Apple-local, deterministic, no unsupported backend assumptions.",
        ],
        "user_templates": [
            "Turn {request_id}: answer the issue-derived task while preserving the shared context.",
            "Turn {request_id}: summarize what changed since the previous turn and continue.",
        ],
        "success_metrics": [
            "ttft_p95_ms",
            "aggregate_tokens_per_second",
            "per_user_tokens_per_second",
            "prefix_cache_hit_rate",
            "resident_state_bytes",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--include-non-candidates", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.ledger))
    if not args.include_non_candidates:
        rows = [row for row in rows if row.get("benchmark_candidate")]
    out_dir = Path(args.out_dir)
    workloads = [workload_for(row) for row in rows]

    if args.dry_run:
        print(json.dumps(workloads, indent=2, sort_keys=True))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for workload in workloads:
        path = out_dir / f"{workload['name']}.json"
        path.write_text(json.dumps(workload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

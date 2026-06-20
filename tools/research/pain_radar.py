#!/usr/bin/env python3
"""Public pain radar for Dismantle/Hawking serving work.

This tool keeps the "vLLM vs llama.cpp" roadmap grounded in reproducible public
pain. It is intentionally offline-first:

  - `seed` writes a curated starter ledger.
  - `summarize` clusters existing ledger rows into Markdown.
  - `add` appends one manually curated public link.
  - `fetch-github` can refresh public GitHub metadata when explicitly invoked.

No command launches models, servers, or benchmarks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "docs/research/pain_radar/ledger.jsonl"
DEFAULT_CLUSTERS = ROOT / "docs/research/pain_radar/clusters.md"
DEFAULT_FIXES = ROOT / "docs/reports/apple_serving_pain_fixes.md"


PAIN_CLASSES: dict[str, dict[str, Any]] = {
    "long_context_reprocessing": {
        "label": "Long-context reprocessing",
        "benchmark": "shared_agent",
        "feature": "detached_prefix_state",
        "apple_weight": 5,
    },
    "cache_invalidation_opacity": {
        "label": "Cache invalidation opacity",
        "benchmark": "cache_miss_taxonomy",
        "feature": "reason_coded_prefix_cache",
        "apple_weight": 4,
    },
    "high_concurrency_collapse": {
        "label": "High-concurrency collapse",
        "benchmark": "high_concurrency_decode",
        "feature": "continuous_batching_scheduler",
        "apple_weight": 4,
    },
    "long_prompt_starvation": {
        "label": "Long prompt starvation",
        "benchmark": "mixed_latency",
        "feature": "chunked_prefill",
        "apple_weight": 5,
    },
    "metal_backend_gap": {
        "label": "Apple/Metal backend gap",
        "benchmark": "apple_backend_gate",
        "feature": "metal_first_defaults",
        "apple_weight": 5,
    },
    "spec_decode_regression": {
        "label": "Spec decode regression",
        "benchmark": "spec_decode_gate",
        "feature": "net_positive_spec_gate",
        "apple_weight": 5,
    },
    "memory_cliff": {
        "label": "Memory cliff",
        "benchmark": "memory_budget_matrix",
        "feature": "resident_memory_ledger",
        "apple_weight": 5,
    },
    "install_runtime_confusion": {
        "label": "Install/runtime confusion",
        "benchmark": "fresh_machine_setup",
        "feature": "one_command_apple_path",
        "apple_weight": 3,
    },
}


SEED_ROWS: list[dict[str, Any]] = [
    {
        "url": "https://github.com/ggml-org/llama.cpp/issues/19794",
        "source": "github",
        "engine": "llama.cpp",
        "title": "Prompt cache forces full re-processing on hybrid long-context turns",
        "status": "seeded",
        "pain_class": "long_context_reprocessing",
        "hardware": "unknown",
        "model": "Qwen hybrid",
        "reproducible": "maybe",
        "benchmark_candidate": True,
        "dismantle_feature": "detached_prefix_state",
        "notes": "Seed issue for repeated-prefix, multi-turn prompt cache rebuild pain.",
    },
    {
        "url": "https://github.com/ggml-org/llama.cpp/issues/21681",
        "source": "github",
        "engine": "llama.cpp",
        "title": "Prompt-cache state drift in multi-turn conversations",
        "status": "seeded",
        "pain_class": "cache_invalidation_opacity",
        "hardware": "unknown",
        "model": "hybrid recurrent/attention",
        "reproducible": "maybe",
        "benchmark_candidate": True,
        "dismantle_feature": "reason_coded_prefix_cache",
        "notes": "Good seed for exact state snapshot validation and miss-reason logging.",
    },
    {
        "url": "https://github.com/ggml-org/llama.cpp/issues/19838",
        "source": "github",
        "engine": "llama.cpp",
        "title": "Context truncation/rebuild pain during long chats",
        "status": "seeded",
        "pain_class": "long_context_reprocessing",
        "hardware": "unknown",
        "model": "unknown",
        "reproducible": "maybe",
        "benchmark_candidate": True,
        "dismantle_feature": "detached_prefix_state",
        "notes": "Maps to long conversation churn and cache rebuild latency.",
    },
    {
        "url": "https://github.com/ggml-org/llama.cpp/issues/23752",
        "source": "github",
        "engine": "llama.cpp",
        "title": "MTP speculative decoding degrades throughput on Metal",
        "status": "seeded",
        "pain_class": "spec_decode_regression",
        "hardware": "apple",
        "model": "unknown",
        "reproducible": "maybe",
        "benchmark_candidate": True,
        "dismantle_feature": "net_positive_spec_gate",
        "notes": "Apple-first spec decode must prove net tokens/sec and tail-latency wins.",
    },
    {
        "url": "https://github.com/vllm-project/vllm/issues/37168",
        "source": "github",
        "engine": "vLLM",
        "title": "Agent context mutability vs prefix/KV cache assumptions",
        "status": "seeded",
        "pain_class": "cache_invalidation_opacity",
        "hardware": "server_gpu",
        "model": "agent workloads",
        "reproducible": "maybe",
        "benchmark_candidate": True,
        "dismantle_feature": "reason_coded_prefix_cache",
        "notes": "Turns dynamic agent context into cache/state benchmark cases.",
    },
    {
        "url": "https://github.com/vllm-project/vllm/issues/22693",
        "source": "github",
        "engine": "vLLM",
        "title": "Context parallelism and sequence parallelism RFC",
        "status": "seeded",
        "pain_class": "high_concurrency_collapse",
        "hardware": "server_gpu",
        "model": "unknown",
        "reproducible": "no",
        "benchmark_candidate": False,
        "dismantle_feature": "continuous_batching_scheduler",
        "notes": "Architecture reference, not an immediate Apple-local reproduction.",
    },
    {
        "url": "https://github.com/vllm-project/vllm/issues/42024",
        "source": "github",
        "engine": "vLLM",
        "title": "KV connector changes visible capacity/concurrency behavior",
        "status": "seeded",
        "pain_class": "memory_cliff",
        "hardware": "server_gpu",
        "model": "unknown",
        "reproducible": "maybe",
        "benchmark_candidate": True,
        "dismantle_feature": "resident_memory_ledger",
        "notes": "Maps to visible capacity accounting and memory-budget explanations.",
    },
    {
        "url": "https://github.com/vllm-project/vllm/issues/1441",
        "source": "github",
        "engine": "vLLM",
        "title": "Mac/Metal/MPS support gap",
        "status": "seeded",
        "pain_class": "metal_backend_gap",
        "hardware": "apple",
        "model": "unknown",
        "reproducible": "yes",
        "benchmark_candidate": True,
        "dismantle_feature": "metal_first_defaults",
        "notes": "Strategic Apple-first positioning seed.",
    },
    {
        "url": "https://github.com/vllm-project/vllm/issues/19073",
        "source": "github",
        "engine": "vLLM",
        "title": "Metal support request/thread",
        "status": "seeded",
        "pain_class": "metal_backend_gap",
        "hardware": "apple",
        "model": "unknown",
        "reproducible": "yes",
        "benchmark_candidate": True,
        "dismantle_feature": "metal_first_defaults",
        "notes": "Tracks the gap between server-GPU vLLM and Apple-local serving.",
    },
    {
        "url": "https://github.com/ggml-org/llama.cpp/issues/20697",
        "source": "github",
        "engine": "llama.cpp",
        "title": "Disk-based context checkpoint offloading for long-context inference",
        "status": "seeded",
        "pain_class": "memory_cliff",
        "hardware": "apple",
        "model": "long-context chat",
        "reproducible": "maybe",
        "benchmark_candidate": True,
        "dismantle_feature": "resident_memory_ledger",
        "notes": "Directly maps to UMA cache budgeting and checkpoint residency choices.",
    },
    {
        "url": "https://github.com/ggml-org/llama.cpp/issues/20133",
        "source": "github",
        "engine": "llama.cpp",
        "title": "Second-turn performance drop with long context and multimodal state",
        "status": "seeded",
        "pain_class": "long_context_reprocessing",
        "hardware": "unknown",
        "model": "Qwen multimodal",
        "reproducible": "maybe",
        "benchmark_candidate": True,
        "dismantle_feature": "detached_prefix_state",
        "notes": "Seed for second-turn degradation and state-boundary tests.",
    },
    {
        "url": "https://github.com/vllm-project/vllm/issues/37729",
        "source": "github",
        "engine": "vLLM",
        "title": "Engine core deadlock under concurrent load",
        "status": "seeded",
        "pain_class": "high_concurrency_collapse",
        "hardware": "server_gpu",
        "model": "unknown",
        "reproducible": "maybe",
        "benchmark_candidate": True,
        "dismantle_feature": "continuous_batching_scheduler",
        "notes": "Maps to watchdog, cancellation, and forward-progress metrics under load.",
    },
    {
        "url": "https://github.com/vllm-project/vllm/issues/38591",
        "source": "github",
        "engine": "vLLM",
        "title": "Apple-local install/runtime failure on newer Qwen model path",
        "status": "seeded",
        "pain_class": "install_runtime_confusion",
        "hardware": "apple",
        "model": "Qwen",
        "reproducible": "maybe",
        "benchmark_candidate": True,
        "dismantle_feature": "one_command_apple_path",
        "notes": "Seed for fresh-machine Apple setup and compatibility testing.",
    },
]


@dataclass
class LedgerRow:
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str:
        return str(self.data.get("url") or "")

    @property
    def pain_class(self) -> str:
        return str(self.data.get("pain_class") or "unknown")

    @property
    def engine(self) -> str:
        return str(self.data.get("engine") or "unknown")

    def normalized(self) -> dict[str, Any]:
        row = dict(self.data)
        now = datetime.now(timezone.utc).isoformat()
        row.setdefault("source", detect_source(self.url))
        row.setdefault("status", "manual")
        row.setdefault("created_at", None)
        row.setdefault("updated_at", None)
        row.setdefault("pain_class", "unknown")
        row.setdefault("hardware", "unknown")
        row.setdefault("model", "unknown")
        row.setdefault("reproducible", "maybe")
        row.setdefault("benchmark_candidate", False)
        row.setdefault("dismantle_feature", PAIN_CLASSES.get(row["pain_class"], {}).get("feature", "triage"))
        row.setdefault("notes", "")
        row.setdefault("ingested_at", now)
        row["score"] = score_row(row)
        return row


def detect_source(url: str) -> str:
    if "github.com" in url:
        return "github"
    if "huggingface.co" in url:
        return "huggingface"
    return "manual"


def parse_github_issue(url: str) -> tuple[str, str, int] | None:
    match = re.search(r"github\.com/([^/]+)/([^/]+)/issues/(\d+)", url)
    if not match:
        return None
    return match.group(1), match.group(2), int(match.group(3))


def score_row(row: dict[str, Any]) -> int:
    pain = PAIN_CLASSES.get(str(row.get("pain_class")), {})
    score = int(pain.get("apple_weight", 1))
    if row.get("hardware") == "apple":
        score += 5
    elif row.get("hardware") == "unknown":
        score += 2
    if row.get("benchmark_candidate"):
        score += 5
    repro = row.get("reproducible")
    if repro == "yes":
        score += 5
    elif repro == "maybe":
        score += 2
    if row.get("status") in {"open", "seeded", "manual"}:
        score += 1
    return score


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}: invalid JSONL line: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def merge_rows(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_url = {str(row.get("url")): row for row in existing if row.get("url")}
    for row in incoming:
        url = str(row.get("url") or "")
        if not url:
            continue
        merged = dict(by_url.get(url, {}))
        merged.update(row)
        by_url[url] = LedgerRow(merged).normalized()
    return sorted(by_url.values(), key=lambda row: (-int(row.get("score", 0)), row.get("engine", ""), row.get("url", "")))


def fetch_github_metadata(row: dict[str, Any], token: str | None = None) -> dict[str, Any]:
    parsed = parse_github_issue(str(row.get("url") or ""))
    if not parsed:
        return row
    owner, repo, issue = parsed
    req = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{issue}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "dismantle-pain-radar",
        },
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        row["fetch_error"] = f"http {exc.code}"
        return row
    except Exception as exc:  # network/tooling failure should not poison ledger
        row["fetch_error"] = str(exc)
        return row

    row = dict(row)
    row.update({
        "title": payload.get("title") or row.get("title"),
        "status": payload.get("state") or row.get("status"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "comments": payload.get("comments"),
        "labels": [label.get("name") for label in payload.get("labels", []) if isinstance(label, dict)],
        "github_reactions": payload.get("reactions", {}).get("total_count"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    })
    body = payload.get("body") or ""
    if body:
        row["body_excerpt"] = textwrap.shorten(" ".join(body.split()), width=500, placeholder="...")
    row.pop("fetch_error", None)
    row["score"] = score_row(row)
    return row


def render_clusters(rows: list[dict[str, Any]]) -> str:
    rows = sorted(rows, key=lambda row: (-int(row.get("score", 0)), row.get("engine", ""), row.get("title", "")))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("pain_class") or "unknown"), []).append(row)

    lines = [
        "# Dismantle/Hawking Public Pain Radar",
        "",
        "Generated from `docs/research/pain_radar/ledger.jsonl`.",
        "",
        "This file tracks public complaints that can become Apple-local benchmark cases.",
        "",
        "## Cluster Summary",
        "",
        "| Pain class | Count | Benchmark | Feature |",
        "|---|---:|---|---|",
    ]
    for pain_class, pain_rows in sorted(grouped.items()):
        meta = PAIN_CLASSES.get(pain_class, {})
        lines.append(
            f"| {meta.get('label', pain_class)} | {len(pain_rows)} | "
            f"`{meta.get('benchmark', 'triage')}` | `{meta.get('feature', 'triage')}` |"
        )

    for pain_class, pain_rows in sorted(grouped.items()):
        meta = PAIN_CLASSES.get(pain_class, {})
        lines.extend(["", f"## {meta.get('label', pain_class)}", ""])
        lines.extend([
            "| Score | Engine | Hardware | Repro | Benchmark | Issue | Feature |",
            "|---:|---|---|---|---|---|---|",
        ])
        for row in pain_rows:
            title = str(row.get("title") or row.get("url"))
            issue = f"[{escape_md(title)}]({row.get('url')})"
            lines.append(
                f"| {row.get('score', 0)} | {row.get('engine', 'unknown')} | "
                f"{row.get('hardware', 'unknown')} | {row.get('reproducible', 'maybe')} | "
                f"{'yes' if row.get('benchmark_candidate') else 'no'} | {issue} | "
                f"`{row.get('dismantle_feature', meta.get('feature', 'triage'))}` |"
            )
    lines.append("")
    return "\n".join(lines)


def render_fixes(rows: list[dict[str, Any]]) -> str:
    candidates = [row for row in rows if row.get("benchmark_candidate")]
    candidates.sort(key=lambda row: (-int(row.get("score", 0)), row.get("engine", "")))
    lines = [
        "# Apple Serving Pain Fixes",
        "",
        "This report is the public-claim ledger for the Dismantle/Hawking serving work.",
        "Rows stay `planned` until a benchmark reproduces the pain and a measured fix lands.",
        "",
        "| Status | Pain | Source | Reproduction | Baseline | Fix | Post-fix | Claim |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in candidates:
        meta = PAIN_CLASSES.get(str(row.get("pain_class")), {})
        source = f"[{escape_md(str(row.get('engine', 'source')))}]({row.get('url')})"
        lines.append(
            f"| planned | {meta.get('label', row.get('pain_class'))} | {source} | "
            f"`{meta.get('benchmark', 'triage')}` | pending | "
            f"`{row.get('dismantle_feature', meta.get('feature', 'triage'))}` | pending | pending |"
        )
    lines.extend([
        "",
        "## Claim Rules",
        "",
        "- Do not mark a row `fixed` without a before/after JSONL result.",
        "- Prefer P95/P99 and cache-hit metrics over single-stream tokens/sec.",
        "- Separate 16GB-class and 96GB-class Apple hardware tiers.",
        "- Treat Hawking claims as release-direction until the model/runtime artifact exists.",
        "",
    ])
    return "\n".join(lines)


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def cmd_seed(args: argparse.Namespace) -> int:
    ledger = Path(args.ledger)
    incoming = [LedgerRow(row).normalized() for row in SEED_ROWS]
    if args.force:
        rows = incoming
    else:
        rows = merge_rows(read_jsonl(ledger), incoming)
    write_jsonl(ledger, rows)
    print(f"wrote {len(rows)} rows to {ledger}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    ledger = Path(args.ledger)
    row = LedgerRow({
        "url": args.url,
        "source": args.source or detect_source(args.url),
        "engine": args.engine,
        "title": args.title,
        "status": args.status,
        "pain_class": args.pain_class,
        "hardware": args.hardware,
        "model": args.model,
        "reproducible": args.reproducible,
        "benchmark_candidate": args.benchmark_candidate,
        "dismantle_feature": args.feature or PAIN_CLASSES.get(args.pain_class, {}).get("feature", "triage"),
        "notes": args.notes or "",
    }).normalized()
    rows = merge_rows(read_jsonl(ledger), [row])
    write_jsonl(ledger, rows)
    print(f"added/updated {args.url} in {ledger}")
    return 0


def cmd_fetch_github(args: argparse.Namespace) -> int:
    ledger = Path(args.ledger)
    rows = read_jsonl(ledger)
    token = args.github_token
    refreshed = []
    for row in rows:
        if parse_github_issue(str(row.get("url") or "")):
            refreshed.append(fetch_github_metadata(row, token=token))
        else:
            refreshed.append(row)
    write_jsonl(ledger, merge_rows([], refreshed))
    print(f"refreshed GitHub metadata for {ledger}")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.ledger))
    clusters = Path(args.clusters)
    fixes = Path(args.fixes)
    clusters.parent.mkdir(parents=True, exist_ok=True)
    fixes.parent.mkdir(parents=True, exist_ok=True)
    clusters.write_text(render_clusters(rows), encoding="utf-8")
    fixes.write_text(render_fixes(rows), encoding="utf-8")
    print(f"wrote {clusters}")
    print(f"wrote {fixes}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER), help="pain ledger JSONL")
    sub = parser.add_subparsers(dest="cmd", required=True)

    seed = sub.add_parser("seed", help="write/merge the curated starter ledger")
    seed.add_argument("--force", action="store_true", help="replace ledger instead of merging")
    seed.set_defaults(func=cmd_seed)

    add = sub.add_parser("add", help="add one manually curated public pain link")
    add.add_argument("--url", required=True)
    add.add_argument("--engine", required=True)
    add.add_argument("--title", required=True)
    add.add_argument("--pain-class", choices=sorted(PAIN_CLASSES), required=True)
    add.add_argument("--source", default=None)
    add.add_argument("--status", default="manual")
    add.add_argument("--hardware", default="unknown", choices=["apple", "server_gpu", "cpu", "unknown"])
    add.add_argument("--model", default="unknown")
    add.add_argument("--reproducible", default="maybe", choices=["yes", "maybe", "no"])
    add.add_argument("--benchmark-candidate", action="store_true")
    add.add_argument("--feature", default=None)
    add.add_argument("--notes", default="")
    add.set_defaults(func=cmd_add)

    fetch = sub.add_parser("fetch-github", help="refresh public GitHub issue metadata")
    fetch.add_argument("--github-token", default=None, help="optional GitHub token")
    fetch.set_defaults(func=cmd_fetch_github)

    summarize = sub.add_parser("summarize", help="write clusters and claim ledger docs")
    summarize.add_argument("--clusters", default=str(DEFAULT_CLUSTERS))
    summarize.add_argument("--fixes", default=str(DEFAULT_FIXES))
    summarize.set_defaults(func=cmd_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

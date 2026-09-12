"""Measure the current HCLI tool-surface baseline for Gravity reduction.

The benchmark is read-only apart from its explicit JSON receipt. It measures
catalog/schema volume, registry discovery, focused discovery, unknown-tool
recovery, and Python compile time so a future reduction can prove improvement
instead of reporting a smaller count alone.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FOCUSES = ("filesystem", "authorized defensive security", "campaign", "gravity")


def _timed(fn, repeats: int = 5) -> dict[str, int]:
    values: list[int] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        fn()
        values.append(time.perf_counter_ns() - started)
    return {
        "min_ns": min(values),
        "median_ns": int(median(values)),
        "max_ns": max(values),
    }


def build(root: Path, *, status: str = "BASELINE_ONLY") -> dict[str, Any]:
    from hcli.engine import Engine
    from hcli.tool_registry import default_tool_registry

    registry_timing = _timed(
        lambda: default_tool_registry(root, repo_root=root), repeats=3)
    registry = default_tool_registry(root, repo_root=root)
    canonical = registry.discover()
    all_specs = registry.discover(include_aliases=True)

    def compact(value: Any) -> int:
        return len(json.dumps(value, sort_keys=True, separators=(",", ":")))

    discovery_timing = _timed(lambda: registry.discover(), repeats=10)
    focused = {}
    focused_timing = {}
    for focus in FOCUSES:
        focused[focus] = registry.describe(focus, max_results=12)
        focused_timing[focus] = _timed(
            lambda focus=focus: registry.describe(focus, max_results=12), repeats=10)
    full_prompt_catalog = Engine._tool_catalog(registry)
    compact_prompt_catalog = {
        focus: Engine._compact_tool_catalog(registry, focus=focus)
        for focus in FOCUSES
    }
    compact_prompt_timing = {
        focus: _timed(
            lambda focus=focus: Engine._compact_tool_catalog(registry, focus=focus),
            repeats=10,
        )
        for focus in FOCUSES
    }
    unknown_timing = _timed(
        lambda: registry.invoke("gravity.unknown_operation", {}), repeats=20)

    source = ROOT / "hcli" / "tool_registry.py"
    source_bytes = source.stat().st_size if source.is_file() else None
    source_lines = len(source.read_text(errors="replace").splitlines()) if source.is_file() else None
    compile_timing = _timed(
        lambda: compile(source.read_text(errors="replace"), str(source), "exec"),
        repeats=5) if source.is_file() else None
    return {
        "schema": "hawking.gravity.tool_benchmark.v2",
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timing_unit": "ns",
        "surface": {
            "registered_callable_spellings": len(all_specs),
            "discoverable_model_facing_tools": len(canonical),
            "compatibility_aliases": sum(bool(item.get("alias_of")) for item in all_specs),
            "all_catalog_json_bytes": compact(all_specs),
            "discoverable_catalog_json_bytes": compact(canonical),
            "focused_catalog_json_bytes": {
                focus: compact(doc) for focus, doc in focused.items()
            },
            "full_prompt_catalog_chars": len(full_prompt_catalog),
            "compact_prompt_catalog_chars": {
                focus: len(value) for focus, value in compact_prompt_catalog.items()
            },
            "compact_prompt_reduction_percent": {
                focus: round(
                    (1 - len(value) / len(full_prompt_catalog)) * 100, 2
                )
                for focus, value in compact_prompt_catalog.items()
            },
        },
        "timing": {
            "registry_build": registry_timing,
            "discover": discovery_timing,
            "focused_discover": focused_timing,
            "compact_prompt_catalog": compact_prompt_timing,
            "unknown_tool_recovery": unknown_timing,
            "tool_registry_compile": compile_timing,
        },
        "source": {
            "path": "hcli/tool_registry.py",
            "bytes": source_bytes,
            "lines": source_lines,
        },
        "claim_boundary": (
            "This is a baseline for comparison. It does not claim a speedup, "
            "capability parity, or semantic equivalence for any future reduction."
        ),
        "next_measurement": (
            "Run the same receipt after one typed-family migration and compare "
            "first-tool correctness, invalid-call recovery, tokens, latency, "
            "compile cost, and retained tests."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--status", default="BASELINE_ONLY",
                        help="receipt state label, e.g. BASELINE_ONLY or CURRENT_PARITY")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "receipts" / "future" / "GRAVITY_TOOL_BENCHMARK_BASELINE.json")
    args = parser.parse_args()
    doc = build(args.root.resolve(), status=args.status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "surface": doc["surface"],
                      "timing": doc["timing"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

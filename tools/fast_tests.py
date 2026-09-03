#!/usr/bin/env python3
"""Run the HCLI suite across cores by sharding files, without a new dependency.

pytest-xdist would do this, but installing it needs `--break-system-packages` on
the interpreter the live daemon runs from, and breaking that to speed up a test
run is not a trade worth making.

Sharding by FILE, longest-first. The wall clock of a sharded run is the slowest
shard, so the schedule matters more than the shard count: greedy
longest-processing-time assignment keeps one slow file from deciding the whole
run. Timings come from a previous run when one is available and default to equal
weight otherwise.

    python3 tools/fast_tests.py [--shards N] [--durations]

Exit code is non-zero if any shard failed. Failures are printed once, merged.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / ".hcli" / "test-timings.json"

#: Red by design. The suite is judged with these excluded; see docs/HANDOFF.md.
PROTECTED_GATES = (
    "test_goal_verifier_synthesis", "test_hcli_overhead", "test_self_mutation_e2e",
    "test_context_compiler_runtime", "test_qwen38_prefill_pipeline",
    "test_long_context_runtime", "test_deltanet_state_checkpoint",
    "test_autonomous_frontier_metabolism", "test_capability_callsite_reachability",
    "test_resident_protected_performance", "test_resident_successor_handoff",
    "test_negative_science_runtime", "test_resident_watch_control_plane",
)


def test_files() -> list[str]:
    excluded = {f"{name}.py" for name in PROTECTED_GATES}
    found: list[str] = []
    for path in sorted((REPO / "hcli").rglob("test_*.py")):
        if path.name in excluded or "__pycache__" in path.parts:
            continue
        found.append(str(path.relative_to(REPO)))
    return found


def load_timings() -> dict[str, float]:
    try:
        return {k: float(v) for k, v in json.loads(CACHE.read_text()).items()}
    except (OSError, ValueError):
        return {}


def save_timings(timings: dict[str, float]) -> None:
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(timings, indent=1, sort_keys=True))
    except OSError:
        pass


def plan(files: list[str], shards: int, timings: dict[str, float]) -> list[list[str]]:
    """Greedy longest-processing-time. The slowest shard IS the wall clock."""
    ordered = sorted(files, key=lambda f: timings.get(f, 1.0), reverse=True)
    buckets: list[list[str]] = [[] for _ in range(shards)]
    loads = [0.0] * shards
    for name in ordered:
        i = loads.index(min(loads))
        buckets[i].append(name)
        loads[i] += timings.get(name, 1.0)
    return [b for b in buckets if b]


def run_shard(paths: list[str]) -> tuple[int, str, float]:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *paths],
        cwd=REPO, capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr, time.perf_counter() - started


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=max(2, min(16, (os.cpu_count() or 4) - 2)))
    args = ap.parse_args()

    files = test_files()
    timings = load_timings()
    shards = plan(files, args.shards, timings)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        results = list(pool.map(run_shard, shards))
    wall = time.perf_counter() - started

    passed = failed = skipped = 0
    failures: list[str] = []
    for code, out, _ in results:
        for line in out.splitlines():
            if line.startswith("FAILED"):
                failures.append(line[len("FAILED "):].strip())
        m = re.search(r"(?:(\d+) failed[, ]+)?(\d+) passed(?:[, ]+(\d+) skipped)?", out)
        if m:
            failed += int(m.group(1) or 0)
            passed += int(m.group(2) or 0)
            skipped += int(m.group(3) or 0)

    # Per-file wall time for the next run's schedule. A shard's time is shared by
    # its files; splitting it evenly is crude but converges over runs.
    fresh = dict(timings)
    for paths, (_c, _o, elapsed) in zip(shards, results):
        for name in paths:
            fresh[name] = elapsed / len(paths)
    save_timings(fresh)

    for name in sorted(set(failures)):
        print(f"FAILED {name}")
    slowest = max(elapsed for _c, _o, elapsed in results)
    print(
        f"\n{passed} passed, {failed} failed, {skipped} skipped "
        f"in {wall:.1f}s across {len(shards)} shards (slowest shard {slowest:.1f}s)"
    )
    return 1 if any(code != 0 for code, _o, _e in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

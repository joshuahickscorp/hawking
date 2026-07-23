#!/usr/bin/env python3
"""Measure the largest useful Xet concurrency, paying no bytes the campaign did not owe.

The naive autotune downloads the same shards several times at several settings and throws
them away.  That is 45 GiB of transfer to learn one integer, on a link this run has
already measured at 1.81 Gbit/s, and it violates the one-copy law twice over.

So the trial rides on shards the full stream needs regardless: each configuration fetches
a disjoint group of not-yet-resident shards, they are all kept, and the campaign is
further ahead when the autotune finishes than when it started.

What is measured, per configuration: sustained network throughput over the whole group,
per-shard spread, wall time, disk written, and free-space movement.  What is not claimed:
that a setting untested here would not do better.

    trial       run the concurrency trial and write GLM52_GENERATION_B_XET_AUTOTUNE.json
    status
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_contract as contract  # noqa: E402
from glm52_common import sha256_file  # noqa: E402

REPO = HERE.parent.parent
STATE = Path.home() / "Library/Application Support/Hawking/GLM52Gravity"
SOURCE = STATE / "source"
CACHE = STATE / "source_fetch/hf_home/hub"
REPORTS = REPO / "reports/condense/glm52_generation_b"

REPO_ID = "zai-org/GLM-5.2"
REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--zai-org--GLM-5.2/snapshots" / REVISION

# Concurrent whole-file downloads per trial.  Xet already parallelises chunk ranges inside
# one file, so this is the outer axis: how many dependency-window files to have in flight.
CONFIGURATIONS = (1, 4, 8)
SHARDS_PER_CONFIGURATION = 3
# Never let a trial approach the campaign's operating reserve.
DISK_FLOOR_BYTES = 60 * (1 << 30)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def official_shards() -> list[str]:
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())
    return sorted(set(index["weight_map"].values()))


def absent_shards() -> list[str]:
    return [name for name in official_shards() if not (SOURCE / name).exists()]


def _fetch_one(name: str, digests: dict[str, str]) -> dict:
    from huggingface_hub import hf_hub_download

    started = time.time()
    cached = hf_hub_download(REPO_ID, name, revision=REVISION, cache_dir=str(CACHE))
    seconds = time.time() - started
    blob = Path(cached).resolve()
    digest = sha256_file(blob)
    expected = digests.get(name)
    if expected and digest != expected:
        blob.unlink(missing_ok=True)
        return {"shard": name, "status": "DIGEST_MISMATCH", "seconds": seconds}
    target = SOURCE / name
    partial = target.with_suffix(target.suffix + ".partial")
    shutil.move(str(blob), str(partial))
    os.replace(partial, target)
    if Path(cached).is_symlink():
        Path(cached).unlink()
    size = target.stat().st_size
    return {"shard": name, "status": "VERIFIED", "bytes": size,
            "seconds": round(seconds, 2),
            "megabits_per_second": round(size * 8 / max(seconds, 1e-6) / 1e6, 1)}


def run_configuration(workers: int, shards: list[str], digests: dict[str, str]) -> dict:
    free_before = shutil.disk_usage(SOURCE).free
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda name: _fetch_one(name, digests), shards))
    wall = time.time() - started
    free_after = shutil.disk_usage(SOURCE).free

    ok = [row for row in rows if row["status"] == "VERIFIED"]
    total = sum(row["bytes"] for row in ok)
    per_shard = [row["megabits_per_second"] for row in ok]
    return {
        "concurrent_files": workers,
        "shards": shards,
        "verified": len(ok), "failed": len(rows) - len(ok),
        "bytes": total,
        "wall_seconds": round(wall, 2),
        # The number that decides the setting: bytes actually landed over wall time, not
        # the mean of per-file rates, which flatters concurrency by ignoring the tail.
        "sustained_megabits_per_second": round(total * 8 / max(wall, 1e-6) / 1e6, 1),
        "per_shard_megabits_per_second": per_shard,
        "per_shard_spread": {
            "min": min(per_shard) if per_shard else None,
            "median": round(statistics.median(per_shard), 1) if per_shard else None,
            "max": max(per_shard) if per_shard else None,
        },
        "free_disk_before": free_before, "free_disk_after": free_after,
        "disk_written": free_before - free_after,
        "rows": rows,
    }


def trial() -> int:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    pool = absent_shards()
    need = len(CONFIGURATIONS) * SHARDS_PER_CONFIGURATION
    if len(pool) < need:
        print(json.dumps({"status": "NOT_ENOUGH_UNFETCHED_SHARDS",
                          "absent": len(pool), "needed": need}))
        return 1
    free = shutil.disk_usage(SOURCE).free
    if free < DISK_FLOOR_BYTES + need * 6 * (1 << 30):
        print(json.dumps({"status": "REFUSED_ON_DISK", "free_gib": round(free / (1 << 30), 1)}))
        return 1

    digests = {row["path"]: row["lfs_sha256"]
               for row in contract._manifest_info()[1] if row.get("lfs_sha256")}

    results = []
    cursor = 0
    for workers in CONFIGURATIONS:
        group = pool[cursor:cursor + SHARDS_PER_CONFIGURATION]
        cursor += SHARDS_PER_CONFIGURATION
        print(f"concurrency {workers}: {group}", flush=True)
        row = run_configuration(workers, group, digests)
        print(f"  sustained {row['sustained_megabits_per_second']} Mbps "
              f"over {row['wall_seconds']}s", flush=True)
        results.append(row)

    best = max(results, key=lambda row: row["sustained_megabits_per_second"])
    baseline = next(row for row in results if row["concurrent_files"] == 1)
    gain = (best["sustained_megabits_per_second"]
            / max(baseline["sustained_megabits_per_second"], 1e-6))

    payload = {
        "schema": "hawking.glm52.generation_b_xet_autotune.v1",
        "generated_at": now(), "revision": REVISION,
        "method": {
            "axis": "concurrent whole-file downloads; Xet parallelises chunk ranges within a file",
            "no_wasted_bytes": ("each configuration fetches a disjoint group of shards the "
                                "full stream needs anyway, and every shard is kept"),
            "shards_per_configuration": SHARDS_PER_CONFIGURATION,
            "high_performance_mode": os.environ.get("HF_XET_HIGH_PERFORMANCE"),
            "decision_metric": "bytes landed over wall time, not the mean of per-file rates",
        },
        "configurations": results,
        "selected": {
            "concurrent_files": best["concurrent_files"],
            "sustained_megabits_per_second": best["sustained_megabits_per_second"],
            "gain_over_serial": round(gain, 3),
        },
        "interpretation": (
            "a gain near 1.0 across configurations means the link, not the client, is the "
            "ceiling, and raising concurrency only adds queue depth and disk pressure"),
        "not_claimed": ("that an untested setting would not do better, or that this holds "
                        "on a different link"),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    target = REPORTS / "GLM52_GENERATION_B_XET_AUTOTUNE.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"wrote": str(target.relative_to(REPO)),
                      "selected": payload["selected"],
                      "sustained": {row["concurrent_files"]:
                                    row["sustained_megabits_per_second"] for row in results}},
                     indent=2))
    return 0


def status() -> int:
    absent = absent_shards()
    print(json.dumps({"official_shards": len(official_shards()),
                      "resident": len(official_shards()) - len(absent),
                      "absent": len(absent),
                      "free_disk_gib": round(shutil.disk_usage(SOURCE).free / (1 << 30), 1)},
                     indent=2))
    return 0


def selftest() -> int:
    shards = official_shards()
    assert len(shards) == 282, len(shards)
    assert len(set(shards)) == len(shards)
    # Disjoint groups: a configuration must never be handed a shard another already took,
    # or its throughput would be measured against a warm cache rather than the network.
    pool = shards[:9]
    groups, cursor = [], 0
    for _ in CONFIGURATIONS:
        groups.append(pool[cursor:cursor + SHARDS_PER_CONFIGURATION])
        cursor += SHARDS_PER_CONFIGURATION
    flat = [name for group in groups for name in group]
    assert len(flat) == len(set(flat)), "configuration groups overlap"
    assert all(len(group) == SHARDS_PER_CONFIGURATION for group in groups)
    print(json.dumps({"selftest": "PASS", "configurations": list(CONFIGURATIONS),
                      "absent_now": len(absent_shards())}))
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    raise SystemExit({"trial": trial, "status": status, "selftest": selftest}[command]())

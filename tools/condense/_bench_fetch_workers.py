#!/usr/bin/env python3.12
"""Full-file concurrent fetch throughput at 1/2/4 workers via real _Prefetcher.

Scratch only. Deletes bodies as delivered. Honours the campaign disk floor.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm52_activation_aware_pack as m  # noqa: E402

RECEIPT = ROOT / "GLM52_REHYDRATION_RECEIPT.json"
REPO = "zai-org/GLM-5.2"
FLOOR = m.DISK_FLOOR_BYTES
# Disjoint shards, far from the live pack window.
SHARD_SETS = {
    1: [150, 151],
    2: [152, 153],
    4: [154, 155, 156, 157],
}


def _rev() -> str:
    r = json.loads(RECEIPT.read_text())
    return r["immutable_tree_url"].rstrip("/").split("/")[-1]


def _size(n: int) -> int:
    r = json.loads(RECEIPT.read_text())
    return int(r["per_file_sha256"][f"model-{n:05d}-of-00282.safetensors"]["size"])


def _free() -> int:
    return shutil.disk_usage(Path.home()).free


def make_ensure(scratch: Path, revision: str):
    os.environ["HF_HOME"] = str(scratch / "hf_home")
    os.environ["HF_HUB_CACHE"] = str(scratch / "hf_cache")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if os.environ.get("HF_XET_HIGH_PERFORMANCE") is None:
        try:
            import hf_xet  # noqa: F401
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        except ImportError:
            pass
    from huggingface_hub import hf_hub_download

    def ensure(n, source_dir, fetch=True, floor=0, body_bytes=0, reserve=True):
        name = f"model-{n:05d}-of-00282.safetensors"
        dest = Path(source_dir) / name
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        size = _size(n)
        if _free() - size < FLOOR:
            raise m.DiskFloorError(
                f"bench refuse shard {n}: free={_free()} size={size} floor={FLOOR}"
            )
        t0 = time.time()
        got = hf_hub_download(
            repo_id=REPO, filename=name, revision=revision, local_dir=str(source_dir),
        )
        dt = time.time() - t0
        mbps = (size * 8 / 1e6) / dt if dt > 0 else 0.0
        print(f"  shard {n}: {size/1e9:.2f} GB in {dt:.1f}s ({mbps:.0f} Mbit/s)", flush=True)
        return Path(got)

    return ensure


def run_trial(workers: int, shards: list[int], scratch: Path, revision: str) -> dict:
    peak_need = min(workers, len(shards)) * max(_size(n) for n in shards)
    free = _free()
    if free - peak_need < FLOOR:
        return {
            "fetch_workers": workers,
            "skipped": True,
            "reason": "insufficient_headroom",
            "free_gib": round(free / (1 << 30), 2),
            "peak_need_gib": round(peak_need / (1 << 30), 2),
        }

    source = scratch / f"w{workers}"
    if source.exists():
        shutil.rmtree(source)
    source.mkdir(parents=True)
    total_bytes = sum(_size(n) for n in shards)
    ensure = make_ensure(scratch, revision)
    pref = m._Prefetcher(
        shards, source, fetch=True, floor=FLOOR,
        workers=workers, body_bytes=m.DEFAULT_SHARD_BODY_BYTES, ensure=ensure,
    )
    t0 = time.time()
    delivered = []
    try:
        for n in shards:
            path = pref.get(n)
            delivered.append(n)
            if path.exists():
                path.unlink()
            pref.release(n)
    finally:
        pref.close()
        shutil.rmtree(source, ignore_errors=True)
        shutil.rmtree(scratch / "hf_cache", ignore_errors=True)
        shutil.rmtree(scratch / "hf_home", ignore_errors=True)

    wall = time.time() - t0
    agg = (total_bytes * 8 / 1e6) / wall if wall > 0 else 0.0
    return {
        "fetch_workers": workers,
        "skipped": False,
        "shards": shards,
        "total_bytes": total_bytes,
        "wall_seconds": round(wall, 2),
        "aggregate_mbit_s": round(agg, 1),
        "peak_resident_slots": pref.peak_resident,
        "delivered_order": delivered,
        "free_gib_after": round(_free() / (1 << 30), 2),
    }


def main() -> int:
    scratch = Path("/tmp/glm52_fetch_workers_bench")
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    revision = _rev()
    print(json.dumps({
        "free_gib_start": round(_free() / (1 << 30), 2),
        "floor_gib": FLOOR / (1 << 30),
        "revision": revision,
    }, indent=2), flush=True)

    rows = []
    for w in (1, 2, 4):
        print(f"\n=== fetch_workers={w} shards={SHARD_SETS[w]} ===", flush=True)
        row = run_trial(w, SHARD_SETS[w], scratch, revision)
        print(json.dumps(row, indent=2), flush=True)
        rows.append(row)
        time.sleep(1)

    by = {r["fetch_workers"]: r for r in rows if not r.get("skipped")}
    comparison = None
    if 2 in by and 4 in by:
        a2, a4 = by[2]["aggregate_mbit_s"], by[4]["aggregate_mbit_s"]
        comparison = {
            "workers_1_mbit_s": by.get(1, {}).get("aggregate_mbit_s"),
            "workers_2_mbit_s": a2,
            "workers_4_mbit_s": a4,
            "ratio_2_over_1": round(a2 / by[1]["aggregate_mbit_s"], 3) if 1 in by and by[1]["aggregate_mbit_s"] else None,
            "ratio_4_over_1": round(a4 / by[1]["aggregate_mbit_s"], 3) if 1 in by and by[1]["aggregate_mbit_s"] else None,
            "ratio_4_over_2": round(a4 / a2, 3) if a2 else None,
            "four_beats_two": a4 > a2 * 1.05,
            "verdict": (
                "4 workers beat 2" if a4 > a2 * 1.05
                else ("4 workers did NOT beat 2" if a4 < a2 * 0.95 else "4 roughly equal to 2")
            ),
        }

    out = {
        "schema": "hawking.glm52.fetch_workers_throughput.v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "full_file_prefetcher_scratch_evict_on_deliver_hf_xet",
        "disk_floor_gib": FLOOR / (1 << 30),
        "trials": rows,
        "summary_table": [
            {
                "fetch_workers": r["fetch_workers"],
                "aggregate_mbit_s": r.get("aggregate_mbit_s"),
                "wall_seconds": r.get("wall_seconds"),
                "total_gb": round(r["total_bytes"] / 1e9, 2) if r.get("total_bytes") else None,
                "peak_resident_slots": r.get("peak_resident_slots"),
                "skipped": r.get("skipped", False),
            }
            for r in rows
        ],
        "workers_4_vs_2": comparison,
    }
    report = ROOT / "GLM52_FETCH_WORKERS_THROUGHPUT.json"
    report.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(out["summary_table"], indent=2), flush=True)
    if comparison:
        print(json.dumps(comparison, indent=2), flush=True)
    print(f"wrote {report}", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

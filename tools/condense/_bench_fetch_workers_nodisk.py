#!/usr/bin/env python3.12
"""Disk-free concurrent stream probe against real GLM-5.2 shard URLs.

Used when free space is already at/under the campaign disk floor and no further
bodies may land. Streams bytes into memory and discards them. Uses huggingface
auth headers so rate/path match the hub client more closely than bare httpx.

Not a substitute for full-file HF/Xet downloads, but the only honest concurrent
measurement available without writing disk.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "GLM52_REHYDRATION_RECEIPT.json"
REPO = "zai-org/GLM-5.2"
PROBE_SECONDS = 30.0
SHARDS = {
    1: [180],
    2: [181, 182],
    4: [183, 184, 185, 186],
}
PRIOR_FULL_FILE_W1 = {
    "mode": "full_file_prefetcher",
    "fetch_workers": 1,
    "shards": [150, 151, 152, 153],
    "total_bytes": 21447448736,
    "wall_seconds": 305.48,
    "aggregate_mbit_s": 561.7,
    "per_stream_mbit_s": [861.0, 565.0, 526.0, 439.0],
    "note": "full-file serial prefetcher measured earlier this session",
}


def _rev() -> str:
    r = json.loads(RECEIPT.read_text())
    return r["immutable_tree_url"].rstrip("/").split("/")[-1]


def probe(workers: int, shards: list[int], revision: str, seconds: float) -> dict:
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    import httpx

    headers = build_hf_headers()
    urls = {
        n: hf_hub_url(
            repo_id=REPO,
            filename=f"model-{n:05d}-of-00282.safetensors",
            revision=revision,
        )
        for n in shards
    }
    lock = threading.Lock()
    got = {n: 0 for n in shards}
    errs: dict[int, str] = {}
    stop_at = time.time() + seconds

    def one(n: int) -> None:
        try:
            with httpx.Client(headers=headers, follow_redirects=True, timeout=60.0) as client:
                with client.stream("GET", urls[n]) as resp:
                    resp.raise_for_status()
                    for chunk in resp.iter_bytes(1024 * 256):
                        if time.time() >= stop_at:
                            break
                        with lock:
                            got[n] += len(chunk)
        except Exception as exc:  # noqa: BLE001
            errs[n] = f"{type(exc).__name__}: {exc}"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, n) for n in shards]
        for f in as_completed(futs):
            f.result()
    wall = time.time() - t0
    total = sum(got.values())
    agg = (total * 8 / 1e6) / wall if wall > 0 else 0.0
    per = {
        str(n): {
            "bytes": got[n],
            "mbit_s": round((got[n] * 8 / 1e6) / wall, 1) if wall > 0 else 0.0,
        }
        for n in shards
    }
    print(
        f"workers={workers}: {total/1e6:.1f} MB in {wall:.1f}s → {agg:.0f} Mbit/s",
        flush=True,
    )
    for n in shards:
        print(f"  shard {n}: {per[str(n)]['mbit_s']} Mbit/s", flush=True)
    return {
        "mode": "disk_free_http_stream_with_hf_headers",
        "fetch_workers": workers,
        "shards": shards,
        "wall_seconds": round(wall, 2),
        "total_bytes": total,
        "aggregate_mbit_s": round(agg, 1),
        "per_stream_mbit_s": per,
        "errors": errs,
        "caveat": (
            "Not the HF/Xet multi-connection path. Absolute Mbit/s may be lower "
            "than full-file downloads; use ratios across 1/2/4, and the prior "
            "full-file serial number for absolute single-stream reference."
        ),
    }


def main() -> int:
    revision = _rev()
    print(json.dumps({
        "revision": revision,
        "probe_seconds": PROBE_SECONDS,
        "prior_full_file_w1": PRIOR_FULL_FILE_W1,
    }, indent=2), flush=True)

    rows = []
    for w in (1, 2, 4):
        print(f"\n=== workers={w} ===", flush=True)
        rows.append(probe(w, SHARDS[w], revision, PROBE_SECONDS))

    by = {r["fetch_workers"]: r for r in rows}
    comparison = {
        "ratio_2_over_1": round(by[2]["aggregate_mbit_s"] / by[1]["aggregate_mbit_s"], 3)
        if by[1]["aggregate_mbit_s"] else None,
        "ratio_4_over_1": round(by[4]["aggregate_mbit_s"] / by[1]["aggregate_mbit_s"], 3)
        if by[1]["aggregate_mbit_s"] else None,
        "ratio_4_over_2": round(by[4]["aggregate_mbit_s"] / by[2]["aggregate_mbit_s"], 3)
        if by[2]["aggregate_mbit_s"] else None,
        "four_beats_two": by[4]["aggregate_mbit_s"] > by[2]["aggregate_mbit_s"] * 1.05,
        "two_beats_one": by[2]["aggregate_mbit_s"] > by[1]["aggregate_mbit_s"] * 1.05,
    }
    if comparison["four_beats_two"]:
        comparison["verdict"] = "4 workers beat 2 on aggregate throughput"
    elif by[4]["aggregate_mbit_s"] < by[2]["aggregate_mbit_s"] * 0.95:
        comparison["verdict"] = "4 workers did NOT beat 2"
    else:
        comparison["verdict"] = "4 workers roughly equal to 2"

    out = {
        "schema": "hawking.glm52.fetch_workers_throughput.v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "disk_state": "free space at/under campaign floor; no disk writes performed",
        "prior_full_file_serial": PRIOR_FULL_FILE_W1,
        "probes": rows,
        "summary_table": [
            {
                "fetch_workers": r["fetch_workers"],
                "aggregate_mbit_s": r["aggregate_mbit_s"],
                "per_stream_mbit_s": {
                    k: v["mbit_s"] for k, v in r["per_stream_mbit_s"].items()
                },
                "wall_seconds": r["wall_seconds"],
                "total_mb": round(r["total_bytes"] / 1e6, 1),
            }
            for r in rows
        ],
        "workers_comparison": comparison,
    }
    report = ROOT / "GLM52_FETCH_WORKERS_THROUGHPUT.json"
    report.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(out["summary_table"], indent=2), flush=True)
    print(json.dumps(comparison, indent=2), flush=True)
    print(f"wrote {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3.12
"""Rehydrate a NAMED subset of GLM-5.2 BF16 source shards, verified against the seal.

The full streaming fetcher (`glm52_source_fetch.py`) short-circuits once its ledger says
all 282 shards are verified -- which it does, from the completed traversal whose bodies
were then evicted.  That is correct for its own contract and useless for a pilot, which
needs a handful of real BF16 bodies back on disk.

This does the bounded thing: fetch exactly the shards named, verify each against the
per-file sha256 sealed in `GLM52_REHYDRATION_RECEIPT.json`, and refuse to publish one
whose hash does not match.  It never fetches the whole parent, never writes into MOP's
`~/.cache/huggingface`, and stops before a configured disk floor.

Why a subset is the right unit: the campaign's own ordering is pilot the representation
arms on representative windows FIRST, freeze one global program, and only then run the
1.507 TB traversal and pack.  Shards 1-5 carry every organ class the flagship has --
embedding and head, attention, dense MLP, routed experts, shared expert -- so they are
representative in the sense that matters, at roughly 27 GB instead of 1.5 TB.

    .venv/glm52/bin/python tools/condense/glm52_rehydrate_window.py --shards 1 2 3 4 5
    .venv/glm52/bin/python tools/condense/glm52_rehydrate_window.py --status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "GLM52_REHYDRATION_RECEIPT.json"
DEST = Path.home() / "Library/Application Support/Hawking/GLM52Gravity/pilot_source"
LEDGER = DEST / "REHYDRATE_LEDGER.jsonl"

# Never write into MOP's cache. Set before huggingface_hub is imported.
os.environ.setdefault("HF_HOME", str(DEST / "hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(DEST / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

DISK_FLOOR_BYTES = int(os.environ.get("GLM52_PILOT_DISK_FLOOR_BYTES", 60 * 10**9))


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _free_bytes() -> int:
    return shutil.disk_usage(DEST if DEST.exists() else Path.home()).free


def _log(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row}
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def rehydrate(shards: list[int]) -> int:
    from huggingface_hub import hf_hub_download

    r = json.loads(RECEIPT.read_text())
    per_file = r["per_file_sha256"]
    repo = "zai-org/GLM-5.2"
    revision = r["immutable_tree_url"].rstrip("/").split("/")[-1]

    DEST.mkdir(parents=True, exist_ok=True)
    ok = 0
    for n in shards:
        name = f"model-{n:05d}-of-00282.safetensors"
        if name not in per_file:
            print(f"{name}: NOT IN SEALED MANIFEST -- refusing", file=sys.stderr)
            _log({"event": "REFUSED", "shard": name, "why": "not in sealed manifest"})
            return 2
        want = per_file[name]["sha256"]
        size = per_file[name]["size"]
        final = DEST / name

        if final.exists() and _sha256(final) == want:
            print(f"{name}: already present and verified")
            ok += 1
            continue

        free = _free_bytes()
        if free - size < DISK_FLOOR_BYTES:
            print(f"STOP before {name}: free {free/1e9:.1f} GB minus {size/1e9:.1f} GB "
                  f"would cross the {DISK_FLOOR_BYTES/1e9:.0f} GB floor", file=sys.stderr)
            _log({"event": "STOP_DISK_FLOOR", "shard": name, "free_bytes": free})
            return 3

        t0 = time.time()
        print(f"{name}: fetching {size/1e9:.2f} GB ...", flush=True)
        got = hf_hub_download(repo_id=repo, filename=name, revision=revision,
                              local_dir=str(DEST))
        dt = time.time() - t0
        have = _sha256(Path(got))
        if have != want:
            # Quarantine rather than overwrite: a mismatched body must never be mistaken
            # for source, and keeping it lets the mismatch be investigated.
            q = DEST / f"QUARANTINE-{name}"
            Path(got).rename(q)
            print(f"{name}: HASH MISMATCH -- quarantined at {q}", file=sys.stderr)
            _log({"event": "QUARANTINE", "shard": name, "want": want, "have": have})
            return 4

        mbps = (size * 8 / 1e6) / dt if dt > 0 else 0.0
        print(f"{name}: VERIFIED in {dt:.0f}s ({mbps:.0f} Mbit/s)")
        _log({"event": "VERIFIED", "shard": name, "sha256": have, "bytes": size,
              "seconds": round(dt, 1), "megabits_per_second": round(mbps, 1)})
        ok += 1

    print(json.dumps({"verified": ok, "requested": len(shards),
                      "dest": str(DEST), "free_gb": round(_free_bytes() / 1e9, 1)}, indent=2))
    return 0 if ok == len(shards) else 1


def status() -> int:
    r = json.loads(RECEIPT.read_text())
    per_file = r["per_file_sha256"]
    present = sorted(p.name for p in DEST.glob("model-*.safetensors")) if DEST.exists() else []
    verified = []
    for name in present:
        if name in per_file and _sha256(DEST / name) == per_file[name]["sha256"]:
            verified.append(name)
    print(json.dumps({
        "dest": str(DEST),
        "present": len(present),
        "verified_against_seal": len(verified),
        "shards": verified,
        "bytes_on_disk": sum((DEST / n).stat().st_size for n in present),
        "free_gb": round(_free_bytes() / 1e9, 1),
        "note": "a pilot window, not the parent. The full parent is 1.507 TB across 282 shards.",
    }, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, nargs="+", help="shard numbers, e.g. 1 2 3 4 5")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status or not a.shards:
        return status()
    return rehydrate(a.shards)


if __name__ == "__main__":
    raise SystemExit(main())

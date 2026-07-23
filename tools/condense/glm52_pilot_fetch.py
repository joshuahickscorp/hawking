#!/usr/bin/env python3
"""Fetch exactly the shards the pilot windows need, verify them, and stop.

Not a campaign controller.  The pilot needs a bounded, named set of dependency windows
and nothing else, so this takes the shard list straight from the sealed pilot program,
downloads only what is absent, checks every file against the official LFS digest, and
exits.  Resumable by construction: an already-verified shard is skipped, and a partial
download is discarded rather than trusted.

    fetch     download and verify every missing pilot shard
    status    what is resident, what is missing, how many bytes
"""
from __future__ import annotations

import json
import os
import shutil
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
REPORTS = REPO / "reports/condense/glm52_generation_b"
PILOT = REPORTS / "GLM52_GENERATION_B_PILOT_PROGRAM.json"
LEDGER = REPORTS / "GLM52_PILOT_FETCH_LEDGER.jsonl"

REPO_ID = "zai-org/GLM-5.2"
REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
# Leave room for the pilot's own artifacts and the next window; the campaign floor is
# 5 GiB but a pilot that fills the disk blocks the run that follows it.
DISK_FLOOR_BYTES = 60 * (1 << 30)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pilot_shards() -> list[str]:
    program = json.loads(PILOT.read_text())
    shards: set[str] = set()
    for window in program["pilot_windows"]:
        shards.update(window["shards"])
    return sorted(shards)


def official_digests() -> dict[str, str]:
    _info, rows = contract._manifest_info()
    return {row["path"]: row["lfs_sha256"] for row in rows if row.get("lfs_sha256")}


def missing(shards: list[str]) -> list[str]:
    """Absent, and a broken symlink counts as absent rather than as a shard.

    A dangling link is worse than nothing: it satisfies a naive existence check while
    reading as an empty file, and a packer that trusted it would seal an artifact over
    weights it never saw.  Clear it here so the fetch replaces it.
    """
    absent: list[str] = []
    for name in shards:
        path = SOURCE / name
        if path.exists():
            continue
        if path.is_symlink():
            path.unlink()
        absent.append(name)
    return absent


def _append(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def status() -> int:
    shards = pilot_shards()
    absent = missing(shards)
    resident_bytes = sum((SOURCE / n).stat().st_size for n in shards if (SOURCE / n).exists())
    free = shutil.disk_usage(SOURCE).free
    print(json.dumps({
        "pilot_shards": len(shards), "resident": len(shards) - len(absent),
        "missing": len(absent), "resident_bytes": resident_bytes,
        "free_disk_bytes": free, "free_disk_gib": round(free / (1 << 30), 1),
        "missing_shards": absent[:10],
    }, indent=2))
    return 0


def fetch() -> int:
    from huggingface_hub import hf_hub_download

    SOURCE.mkdir(parents=True, exist_ok=True)
    shards = pilot_shards()
    absent = missing(shards)
    digests = official_digests()
    if not absent:
        print(json.dumps({"status": "ALREADY_COMPLETE", "pilot_shards": len(shards)}))
        return 0

    # Xet on, implicit tokens off: the same envelope the campaign runs under, so the
    # throughput measured here transfers to the autotune rather than describing a
    # different configuration.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    done, failed = 0, 0
    for index, name in enumerate(absent, 1):
        free = shutil.disk_usage(SOURCE).free
        if free < DISK_FLOOR_BYTES:
            _append({"at": now(), "event": "STOPPED_ON_DISK_FLOOR",
                     "free_disk_bytes": free, "floor": DISK_FLOOR_BYTES,
                     "remaining": len(absent) - index + 1})
            print(json.dumps({"status": "STOPPED_ON_DISK_FLOOR", "fetched": done,
                              "free_disk_gib": round(free / (1 << 30), 1)}))
            return 1

        started = time.time()
        try:
            cached = hf_hub_download(REPO_ID, name, revision=REVISION,
                                     cache_dir=str(STATE / "source_fetch/hf_home/hub"))
        except Exception as exc:  # noqa: BLE001 - recorded, then the next shard is tried
            failed += 1
            _append({"at": now(), "event": "FETCH_FAILED", "shard": name, "error": str(exc)})
            continue

        seconds = time.time() - started
        digest = sha256_file(Path(cached))
        expected = digests.get(name)
        if expected and digest != expected:
            failed += 1
            _append({"at": now(), "event": "DIGEST_MISMATCH", "shard": name,
                     "expected": expected, "observed": digest})
            continue

        # Move rather than copy: two full copies of a 5 GiB shard is exactly the
        # one-copy-law violation the storage policy forbids.  hf_hub_download returns a
        # symlink into its blob store, and moving the symlink would install a link whose
        # target is about to be garbage, so the blob itself is what moves.
        blob = Path(cached).resolve()
        target = SOURCE / name
        partial = target.with_suffix(target.suffix + ".partial")
        shutil.move(str(blob), str(partial))
        os.replace(partial, target)
        if Path(cached).is_symlink():
            Path(cached).unlink()  # the cache entry now points at nothing

        size = target.stat().st_size
        done += 1
        _append({"at": now(), "event": "VERIFIED", "shard": name, "bytes": size,
                 "seconds": round(seconds, 2),
                 "megabits_per_second": round(size * 8 / max(seconds, 1e-6) / 1e6, 1),
                 "sha256": digest, "digest_checked_against_official_manifest": bool(expected)})
        print(f"{index}/{len(absent)} {name} {size / (1 << 30):.2f} GiB "
              f"{size * 8 / max(seconds, 1e-6) / 1e6:.0f} Mbps", flush=True)

    print(json.dumps({"status": "COMPLETE" if not failed else "PARTIAL",
                      "fetched": done, "failed": failed,
                      "still_missing": len(missing(shards))}, indent=2))
    return 0 if not failed else 1


def selftest() -> int:
    shards = pilot_shards()
    assert shards and all(name.endswith(".safetensors") for name in shards), shards[:3]
    assert len(shards) == len(set(shards)), "pilot shard list has duplicates"
    # Every pilot shard must be a real member of the official index, or the pilot is
    # planning against something other than the pinned parent.
    index = json.loads((Path.home() / ".cache/huggingface/hub/models--zai-org--GLM-5.2"
                        / "snapshots" / REVISION / "model.safetensors.index.json").read_text())
    official = set(index["weight_map"].values())
    unknown = [name for name in shards if name not in official]
    assert not unknown, f"pilot names shards absent from the official index: {unknown[:3]}"
    print(json.dumps({"selftest": "PASS", "pilot_shards": len(shards),
                      "missing_now": len(missing(shards))}))
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    raise SystemExit({"fetch": fetch, "status": status, "selftest": selftest}[command]())

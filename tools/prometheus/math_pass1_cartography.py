#!/usr/bin/env python3.12
"""Math-Preserve PASS 1: streamed cartography over the GLM-5.2 BF16 source.

Per-shard loop: VERIFY -> CAPTURE (real teacher forward, math-domain calibration,
per-expert local sensitivity retained) -> EVICT. No packing here -- packing is PASS 3,
driven by PASS 2's frozen global allocation, and deciding per-shard precision during
PASS 1 would be exactly the greedy per-shard allocation this pipeline is built to avoid.

Entirely separate state from the original General-artifact traversal
(`glm52_source_fetch.py`): its LEDGER/PROBES/COMPACT directories are directory-globbed
with no integrity tag, so writing into them under a different convention would silently
corrupt evidence `rollup()` still depends on. This module owns its own source root,
ledger, capsule directory and lock -- nothing here reads or writes the original's state.

Reuses, not reimplements: `glm52_teacher_capture.capture_window` (the real sealed
NumPy reference forward, chained per layer, sealed+ledgered capsules) and
`eviction_authority` (organ/layer completeness, the same safety property the original
traversal used) do the actual work. This module is the fetch loop + window driver
around them, with the pack step removed and the capsule/source paths redirected.

Calibration is `glm52_capture_program.math_calibration_batch()`: every "mathematics"
domain record in the pinned corpus's TRAIN_PARTITIONS. That pool is 3 real records --
an honest constraint of the pinned corpus, not a bug to route around by blending in
adjacent domains (that would be this module redefining what "Math" means). With ~8
experts selected per token across ~37 total tokens, per-expert statistics over 256
experts are inherently sparse (~1.2 observations/expert on average); PASS 2 should
treat per-layer aggregates as the load-bearing signal and per-expert numbers as
lower-confidence supplementary evidence, not the reverse.

    python3.12 tools/prometheus/math_pass1_cartography.py run [--windows N]
    python3.12 tools/prometheus/math_pass1_cartography.py status
    python3.12 tools/prometheus/math_pass1_cartography.py selftest
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONDENSE = REPO / "tools/condense"
for _p in (HERE, CONDENSE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from glm52_common import resolve_artifact  # noqa: E402

STATE_DIR = Path(
    "/Users/scammermike/Library/Application Support/Hawking/GLM52MathPrometheus"
)
SOURCE_ROOT = STATE_DIR / "source"
CAPSULE_DIR = STATE_DIR / "capsules"
LEDGER = STATE_DIR / "PASS1_LEDGER.jsonl"
PROGRESS = STATE_DIR / "progress.json"
LOCK = STATE_DIR / "pass1.lock"

MANIFEST = REPO / "evidence" / "glm52" / "GLM52_OFFICIAL_MANIFEST.json"
SCHEDULE = REPO / "evidence" / "glm52" / "GLM52_STREAMING_SCHEDULE.json"
GRAPH = resolve_artifact("GLM52_SHARD_DEPENDENCY_GRAPH.json")

# hf_xet cache/scratch stays inside this state root, same reasoning as the original
# fetcher: MOP owns ~/.cache/huggingface and it is hard-protected.
os.environ.setdefault("HF_HOME", str(STATE_DIR / "hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(STATE_DIR / "hf_cache"))
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

DISK_FLOOR_BYTES = int(os.environ.get("GLM52_PASS1_DISK_FLOOR_BYTES", 75 * 10**9))
# Eviction is the only irreversible step. Same default as the original fetcher and the
# same reasoning: an operator who has not said otherwise gets a full disk (bounded by
# the floor above) as the worst case, not destroyed evidence.
EVICTION_PAUSED = os.environ.get("GLM52_PASS1_EVICTION_PAUSED", "1") == "1"
HASH_CHUNK = 16 << 20
SPLIT = "teacher_math"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _append_ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _ledger_rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    rows = []
    for line in LEDGER.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def verified_shards() -> set[str]:
    return {r["shard"] for r in _ledger_rows() if r.get("status") == "VERIFIED"}


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _free_bytes() -> int:
    import shutil

    return shutil.disk_usage(str(SOURCE_ROOT if SOURCE_ROOT.exists() else STATE_DIR)).free


def _fetch_one(row: dict, repo: str, revision: str) -> dict:
    """Same contract as `glm52_source_fetch._fetch_one`, pointed at this module's
    own `SOURCE_ROOT` rather than that module's hardcoded one."""
    from huggingface_hub import hf_hub_download

    name = row["path"]
    started = time.time()
    got_path = Path(hf_hub_download(
        repo_id=repo, filename=name, revision=revision,
        local_dir=str(SOURCE_ROOT), token=False,
    ))
    elapsed = max(time.time() - started, 1e-6)
    size = got_path.stat().st_size
    if size != row["logical_bytes"]:
        quarantine = got_path.with_suffix(got_path.suffix + ".badsize")
        os.replace(got_path, quarantine)
        return {"shard": name, "status": "SIZE_MISMATCH", "expected_bytes": row["logical_bytes"],
                "observed_bytes": size, "quarantined": str(quarantine), "at": _now()}
    observed = _sha256_file(got_path)
    if observed != row["lfs_sha256"]:
        quarantine = got_path.with_suffix(got_path.suffix + ".badhash")
        os.replace(got_path, quarantine)
        return {"shard": name, "status": "HASH_MISMATCH", "expected_sha256": row["lfs_sha256"],
                "observed_sha256": observed, "quarantined": str(quarantine), "at": _now()}
    return {
        "shard": name, "status": "VERIFIED", "bytes": size, "sha256": observed,
        "seconds": round(elapsed, 2),
        "megabits_per_second": round(size * 8 / elapsed / 1e6, 1),
        "at": _now(),
    }


def _protected_after(schedule: dict, index: int) -> set[str]:
    """Every shard any later window carries in -- never evictable at this point."""
    keep: set[str] = set()
    for window in schedule["windows"][index + 1:]:
        keep.update(window.get("carry_in_shards", []))
    return keep


def run(*, max_windows: int | None = None) -> int:
    import fcntl

    import glm52_teacher_capture as teacher

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    CAPSULE_DIR.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(LOCK), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.stderr.write("another PASS1 run holds the lock; exiting\n")
        return 0

    manifest = _read_json(MANIFEST)
    schedule = _read_json(SCHEDULE)
    graph = _read_json(GRAPH)
    config = teacher.official_config()
    repo, revision = manifest["repo"], manifest["revision"]
    by_path = {f["path"]: f for f in manifest["files"] if f.get("is_weight")}

    windows = schedule["windows"]
    if max_windows is not None:
        windows = windows[:max_windows]

    for index, window in enumerate(windows):
        if _free_bytes() < DISK_FLOOR_BYTES:
            _append_ledger({"event": "DISK_FLOOR_STOP", "window": window["window_id"], "at": _now()})
            sys.stderr.write(f"disk floor reached before {window['window_id']}; stopping\n")
            break

        verified = verified_shards()
        for name in window["new_fetch_shards"]:
            if name in verified:
                continue
            result = _fetch_one(by_path[name], repo, revision)
            _append_ledger(result)
            if result["status"] != "VERIFIED":
                sys.stderr.write(f"PASS1 fetch failed: {result}\n")

        capture_receipts = teacher.capture_window(
            window["window_id"], schedule=schedule, graph=graph, source_root=SOURCE_ROOT,
            capsule_dir=CAPSULE_DIR, split=SPLIT, config=config, retain_per_expert=True,
        )
        for receipt in capture_receipts:
            _append_ledger({
                "event": "CAPTURED", "window": window["window_id"],
                "capsule_id": receipt["capsule_id"], "layers": receipt["layers"],
                "input_provenance": receipt["input_provenance"], "at": _now(),
            })

        candidates = window.get("evict_after_seal_shards", [])
        if candidates and not EVICTION_PAUSED:
            protected = _protected_after(schedule, index)
            authority = teacher.eviction_authority(
                candidates, source_root=SOURCE_ROOT, graph=graph, capsule_dir=CAPSULE_DIR,
                ever_verified=verified_shards(),
            )
            freed = []
            for name in authority["authorized"]:
                if name in protected:
                    continue
                target = SOURCE_ROOT / name
                if not target.exists() or target.parent != SOURCE_ROOT:
                    continue
                size = target.stat().st_size
                target.unlink()
                freed.append({"shard": name, "bytes": size})
            _append_ledger({
                "event": "EVICT", "window": window["window_id"], "freed": freed,
                "refused_uncaptured_but_capturable": authority["refused_uncaptured_but_capturable"],
                "at": _now(),
            })

        _write_json(PROGRESS, {
            "windows_done": index + 1, "windows_total": len(schedule["windows"]),
            "last_window": window["window_id"],
            "captured_layers": sorted(teacher.captured_layers(capsule_dir=CAPSULE_DIR)),
            "eviction_paused": EVICTION_PAUSED,
            "at": _now(),
        })

    return 0


def status() -> dict:
    import glm52_teacher_capture as teacher

    return {
        "progress": _read_json(PROGRESS) if PROGRESS.exists() else None,
        "captured_layers": sorted(teacher.captured_layers(capsule_dir=CAPSULE_DIR))
        if CAPSULE_DIR.exists() else [],
        "verified_shard_count": len(verified_shards()),
        "resident_shard_count": len(list(SOURCE_ROOT.glob("*.safetensors")))
        if SOURCE_ROOT.exists() else 0,
        "eviction_paused": EVICTION_PAUSED,
        "state_dir": str(STATE_DIR),
        "at": _now(),
    }


def selftest() -> dict:
    """No network, no source, no writes: the schedule/graph/config plumbing only."""
    import glm52_capture_program as program
    import glm52_teacher_capture as teacher

    schedule = _read_json(SCHEDULE)
    graph = _read_json(GRAPH)
    assert schedule["windows"], "empty schedule"
    assert graph["tensors"], "empty dependency graph"

    records = program.math_calibration_records()
    assert len(records) >= 1, "no mathematics-domain calibration records"
    assert all(r.partition in program.corpus.TRAIN_PARTITIONS for r in records), (
        "a math calibration record leaked from outside TRAIN_PARTITIONS"
    )

    ids = teacher.calibration_ids(SPLIT, vocab_size=154_880)
    assert ids.shape[0] == len(records)
    m1 = teacher.membership_sha256(ids, SPLIT)
    m2 = teacher.membership_sha256(teacher.calibration_ids(SPLIT, vocab_size=154_880), SPLIT)
    assert m1 == m2, "math calibration membership is not deterministic"

    windows = schedule["windows"]
    total_evict = {name for w in windows for name in w.get("evict_after_seal_shards", [])}
    total_fetch = {name for w in windows for name in w.get("new_fetch_shards", [])}
    assert total_evict <= total_fetch, "a window evicts a shard PASS1 never fetches"

    print(json.dumps({"status": "PASS", "windows": len(windows),
                      "math_calibration_records": len(records)}, indent=2))
    return {"status": "PASS"}


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "status"
    if command == "run":
        max_windows = None
        if "--windows" in argv:
            max_windows = int(argv[argv.index("--windows") + 1])
        return run(max_windows=max_windows)
    if command == "status":
        print(json.dumps(status(), indent=2, sort_keys=True))
        return 0
    if command == "selftest":
        selftest()
        return 0
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

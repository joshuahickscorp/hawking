#!/usr/bin/env python3.12
"""GLM-5.2 BF16 source streamer: traverse every official weight shard in dependency-window
order, verify each against the sealed manifest hash, and evict per the sealed schedule so
the 1.507 TB source never has to be resident.

This is PART IV/XIX streaming applied to the acquisition half only.  The source is never
held whole: the sealed schedule's 20 windows peak at 134.3 GB / 26 shards resident, so a
complete 282-shard traversal runs comfortably on this disk.  ``preregistered_usable_raw_
window_bytes`` is treated as what it is -- a ceiling on bytes resident at any instant,
not a cap on bytes ever fetched.

Each shard is fully consumed in the single visit its bytes get: fetched, hash-verified,
probed for weight evidence, and PACKED into the compact sub-bit artifact before its body
is evicted.  The compact shards accumulate and outlive every BF16 window, which is what
lets a 1.507 TB parent end as a ~83 GB artifact this machine can actually host.  Eviction
requires all four -- verified, probed, packed, and unneeded downstream -- so no shard is
ever discarded having given up less than everything cheap it had.

Teacher capture now runs inside the loop: ``glm52_teacher_capture`` executes the sealed
reference forward over every still-resident layer a body carries and seals the capsule
before the body can be unlinked, so the pipeline is VERIFY -> TEACHER_CAPTURE ->
PROBE/PACK -> SEAL -> EVICT.  What is still missing is the rest of the capability half:
the packing evidence is F0 (exact physical accounting) plus F1 (weight-space
reconstruction error) only, and weight-space error is a PROXY.  A complete,
correctly-billed, executable sub-1-BPW artifact is NOT evidence that the model still
works -- that requires fitting against these capsules, output divergence, and evaluation,
none of which happen in this module.

Modes:

* ``stream`` (default) -- full 282-shard traversal with scheduled eviction.  Closes the
  "every official BF16 source shard fetched and verified" stop condition.
* ``resident`` -- never evict; fetch until the residency ceiling is reached and stop.
  Use when the scientific pipeline is ready to consume what is on disk.

Bounds that are load-bearing: free disk is re-read from the filesystem before every shard
rather than predicted; eviction only ever unlinks exact schedule-named files under the
source root that already hold a VERIFIED ledger receipt, and never a file the next window
carries in; the HF cache is redirected inside this state root so MOP's
~/.cache/huggingface is never written to; a shard is published only when its sha256
matches the sealed manifest, and a mismatch quarantines rather than overwrites.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = Path(
    "/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity/source_fetch"
)
SOURCE_ROOT = Path(
    "/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity/source"
)
LEDGER = STATE_DIR / "SOURCE_FETCH_LEDGER.jsonl"
PROGRESS = STATE_DIR / "progress.json"
LOCK = STATE_DIR / "fetch.lock"
DEFERRED = STATE_DIR / "deferred_evictions.json"
PROBES = STATE_DIR / "probes"
ROLLUP = STATE_DIR / "GLM52_SOURCE_WEIGHT_ATLAS.json"
GRAPH = ROOT / "GLM52_SHARD_DEPENDENCY_GRAPH.json"
# The compact artifact itself: this is the deliverable that outlives every BF16 window.
# On the Desktop by request: this is the first full quantized model worth keeping, and it
# must be somewhere the user can see and move it, not buried in Application Support.
COMPACT = Path(os.environ.get(
    "GLM52_COMPACT_ROOT",
    "/Users/scammermike/Desktop/GLM52-Gravity-SubBit",
))

MANIFEST = ROOT / "GLM52_OFFICIAL_MANIFEST.json"
SCHEDULE = ROOT / "GLM52_STREAMING_SCHEDULE.json"
POLICY = ROOT / "GLM52_RESOURCE_RESERVE_POLICY.json"

# hf_xet cache/scratch stays inside this state root.  MOP owns ~/.cache/huggingface and it
# is hard-protected, so it must never be the implicit default here.
os.environ.setdefault("HF_HOME", str(STATE_DIR / "hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(STATE_DIR / "hf_cache"))
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

DISK_FLOOR_BYTES = int(os.environ.get("GLM52_FETCH_DISK_FLOOR_BYTES", 75 * 10**9))
WORKERS = int(os.environ.get("GLM52_FETCH_WORKERS", "10"))
# Probing is CPU-bound and runs off the fetch critical path.  At ~845 MB/s a shard probes
# in ~6.3s against a ~25s download, so a handful of workers keeps probes ahead of fetches.
PROBE_WORKERS = int(os.environ.get("GLM52_PROBE_WORKERS", "4"))
PACK_WORKERS = int(os.environ.get("GLM52_PACK_WORKERS", "1"))
MODE = os.environ.get("GLM52_FETCH_MODE", "stream")
HASH_CHUNK = 16 << 20


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, obj) -> None:
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
    """Shards with a permanent verification receipt.  Survives eviction of the body."""
    return {r["shard"] for r in _ledger_rows() if r.get("status") == "VERIFIED"}


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _free_bytes() -> int:
    return shutil.disk_usage(str(SOURCE_ROOT if SOURCE_ROOT.exists() else ROOT)).free


def _resident(manifest: dict) -> set[str]:
    """Shard bodies actually on disk right now at their exact sealed logical size."""
    done = set()
    for row in manifest["files"]:
        if not row.get("is_weight"):
            continue
        local = SOURCE_ROOT / row["path"]
        if local.exists() and local.stat().st_size == row["logical_bytes"]:
            done.add(row["path"])
    return done


def probed_shards() -> set[str]:
    """Shards whose weight evidence is already captured and durable."""
    if not PROBES.exists():
        return set()
    return {p.name[: -len(".json")] for p in PROBES.glob("*.safetensors.json")}


def packed_shards() -> set[str]:
    """Shards already represented in the compact artifact."""
    if not COMPACT.exists():
        return set()
    return {p.stem + ".safetensors" for p in COMPACT.glob("*.gravity")}


def _pack_shard(name: str, rows: list[dict]) -> dict:
    """Pack one resident shard into the compact artifact before its body is evicted."""
    import glm52_pack as pack

    started = time.time()
    receipt = pack.pack_shard(SOURCE_ROOT / name, rows, COMPACT)
    return {
        "shard": name, "status": "PACKED", "compact_bytes": receipt["compact_bytes"],
        "whole_shard_bpw": round(receipt["whole_shard_bpw"], 6),
        "tensors": receipt["tensors"], "pack_seconds": round(time.time() - started, 1),
        "at": _now(),
    }


def _probe_shard(name: str, rows: list[dict]) -> dict:
    """Capture and durably write one shard's weight evidence before it can be evicted."""
    import glm52_shard_probe as probe

    started = time.time()
    result = probe.probe_shard(SOURCE_ROOT / name, rows)
    _write_json(PROBES / f"{name}.json", result)
    return {
        "shard": name, "status": "PROBED", "tensors": result["tensor_count"],
        "entropy_bits_per_weight": result["shard_zeroth_order_entropy_bits_per_weight"],
        "probe_seconds": round(time.time() - started, 2), "at": _now(),
    }


# Eviction is the only irreversible step in the loop, and a BF16 window carries teacher
# states that no downstream stage can reconstruct once the body is gone.  The default is
# therefore PAUSED: an operator who has not said otherwise gets the outcome where the
# worst case is a full disk, which the floor already bounds, rather than the one where
# the worst case is destroyed evidence, which nothing bounds.
EVICTION_PAUSED = os.environ.get("GLM52_EVICTION_PAUSED", "1") == "1"


def _evict(names: list[str], *, verified: set[str], probed: set[str], packed: set[str],
           protected: set[str], authorized: set[str]) -> list[dict]:
    """Unlink exact schedule-named bodies that are fully accounted for and unneeded.

    Six independent conditions, all required: the schedule named it for eviction, a
    VERIFIED receipt exists so the body is reproducible, its weight evidence is captured,
    its weights are already represented in the compact artifact, the teacher capture gate
    authorized it, and no later window carries it in.  The body streams past once --
    evicting a shard that has not been probed, packed and teacher-captured would throw
    away the single visit its bytes get.  Anything failing one condition is left alone
    rather than force-removed.
    """
    if EVICTION_PAUSED:
        return []
    freed = []
    for name in names:
        if (name in protected or name not in verified or name not in authorized
                or name not in probed or name not in packed):
            continue
        target = SOURCE_ROOT / name
        if not target.exists() or target.parent != SOURCE_ROOT:
            continue
        size = target.stat().st_size
        target.unlink()
        freed.append({"shard": name, "bytes": size})
    return freed


def _deferred() -> set[str]:
    """Shards a previous window offered for eviction and the gate refused."""
    if not DEFERRED.exists():
        return set()
    try:
        return {name for name in _read_json(DEFERRED)["shards"]
                if (SOURCE_ROOT / name).exists()}
    except (KeyError, TypeError, ValueError, OSError):
        return set()


def _teacher_authority(candidates: set[str], window: str) -> dict:
    """Capture the teacher evidence these bodies still owe, then rule on eviction.

    A capture failure denies authority rather than granting it: the shards stay,
    the disk grows against a bounded floor, and the evidence survives.  Refused
    shards are carried into the next window's candidate set, so a refusal defers
    an eviction instead of stranding a body on disk for the rest of the run.
    """
    try:
        import glm52_teacher_capture as teacher

        authority = teacher.capture_for_eviction(candidates)
    except Exception as exc:  # noqa: BLE001 - never let the gate kill the stream
        _append_ledger({"event": "TEACHER_GATE_ERROR", "window": window,
                        "error": f"{type(exc).__name__}: {exc}",
                        "authorized": [], "at": _now()})
        return {"authorized": []}
    still_refused = sorted(
        set(authority["refused_uncaptured_but_capturable"])
        | set(authority["refused_incomplete_organs"])
    )
    _write_json(DEFERRED, {"shards": still_refused, "window": window, "at": _now()})
    _append_ledger({
        "event": "TEACHER_GATE", "window": window,
        "authorized": sorted(authority["authorized"]),
        "refused_uncaptured_but_capturable":
            authority["refused_uncaptured_but_capturable"],
        "refused_incomplete_organs": authority["refused_incomplete_organs"],
        "authorized_with_unrecoverable_organs":
            authority["authorized_with_unrecoverable_organs"],
        "capture_outcome": authority.get("capture_outcome", {}),
        "at": _now(),
    })
    return authority


def _fetch_one(row: dict, repo: str, revision: str) -> dict:
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
        "completed_at": _now(),
    }


def _window_plan(schedule: dict) -> list[dict]:
    return schedule["windows"]


def _protected_after(schedule: dict, index: int) -> set[str]:
    """Every shard any later window carries in -- never evictable at this point."""
    keep: set[str] = set()
    for window in schedule["windows"][index + 1:]:
        keep.update(window.get("carry_in_shards", []))
    return keep


def run() -> int:
    import fcntl

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(LOCK), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.stderr.write("another fetcher holds the lock; exiting\n")
        return 0

    manifest = _read_json(MANIFEST)
    schedule = _read_json(SCHEDULE)
    policy = _read_json(POLICY)
    residency_ceiling = int(os.environ.get(
        "GLM52_FETCH_RESIDENCY_CEILING_BYTES",
        policy["derived"]["preregistered_usable_raw_window_bytes"],
    ))
    repo, revision = manifest["repo"], manifest["revision"]
    by_path = {f["path"]: f for f in manifest["files"] if f.get("is_weight")}
    total_weight = sum(f["logical_bytes"] for f in by_path.values())

    graph = _read_json(GRAPH)
    tensors_by_shard: dict[str, list[dict]] = {}
    for tensor in graph["tensors"]:
        tensors_by_shard.setdefault(tensor["shard"], []).append(tensor)

    verified = verified_shards()
    probed = probed_shards()
    packed = packed_shards()
    windows = _window_plan(schedule)
    wall0 = time.time()
    bytes_this_run = 0
    failed = 0
    evicted_bytes = 0
    probe_pool = ThreadPoolExecutor(max_workers=PROBE_WORKERS)
    # One heavy lane: packing drives Metal, and the campaign's resource rule allows exactly
    # one heavy job at a time.  This is the pipeline's rate limiter by design (~160s/shard
    # against a ~25s fetch), not a bottleneck to tune away.
    pack_pool = ThreadPoolExecutor(max_workers=PACK_WORKERS)
    probe_futures: dict = {}
    pack_futures: dict = {}

    def _submit_probe(name: str) -> None:
        if name not in probed and name not in probe_futures:
            probe_futures[name] = probe_pool.submit(_probe_shard, name, tensors_by_shard[name])
        if name not in packed and name not in pack_futures:
            pack_futures[name] = pack_pool.submit(_pack_shard, name, tensors_by_shard[name])

    def _drain_probes(names: set[str]) -> None:
        """Block until evidence AND compact representation are durable for these shards."""
        for name in list(names):
            future = probe_futures.pop(name, None)
            if future is not None:
                try:
                    _append_ledger(future.result())
                    probed.add(name)
                except Exception as exc:  # noqa: BLE001
                    _append_ledger({"shard": name, "status": "PROBE_ERROR",
                                    "error": f"{type(exc).__name__}: {exc}", "at": _now()})
            future = pack_futures.pop(name, None)
            if future is not None:
                try:
                    _append_ledger(future.result())
                    packed.add(name)
                except Exception as exc:  # noqa: BLE001
                    _append_ledger({"shard": name, "status": "PACK_ERROR",
                                    "error": f"{type(exc).__name__}: {exc}", "at": _now()})

    for index, window in enumerate(windows):
        wid = window["window_id"]
        resident = _resident(manifest)
        # The gate is evidence, not verification.  A shard that was verified but evicted
        # before its probe existed still owes its one cheap look, so it comes back down --
        # an attributed refetch, cheaper than never having the evidence at all.
        owed = [n for n in window["new_fetch_shards"]
                if n not in probed or n not in packed] if MODE == "stream" else [
                n for n in window["new_fetch_shards"] if n not in verified]
        pending = [by_path[n] for n in owed if n not in resident]
        for name in owed:
            if name in resident:
                _submit_probe(name)  # already on disk: mine it without touching the wire

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            queue = list(pending)
            running: dict = {}
            stopped_for_disk = False
            while queue or running:
                while queue and len(running) < WORKERS and not stopped_for_disk:
                    free = _free_bytes()
                    nxt = queue[0]
                    resident_bytes = sum(by_path[n]["logical_bytes"]
                                         for n in _resident(manifest))
                    if free - nxt["logical_bytes"] < DISK_FLOOR_BYTES:
                        _append_ledger({"event": "DISK_FLOOR_STOP", "window": wid,
                                        "free_bytes": free, "floor_bytes": DISK_FLOOR_BYTES,
                                        "at": _now()})
                        stopped_for_disk = True
                        break
                    if MODE == "resident" and \
                            resident_bytes + nxt["logical_bytes"] > residency_ceiling:
                        _append_ledger({"event": "RESIDENCY_CEILING_STOP", "window": wid,
                                        "resident_bytes": resident_bytes,
                                        "ceiling_bytes": residency_ceiling, "at": _now()})
                        stopped_for_disk = True
                        break
                    queue.pop(0)
                    running[pool.submit(_fetch_one, nxt, repo, revision)] = nxt
                if not running:
                    break
                for future in as_completed(list(running)):
                    row = running.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        result = {"shard": row["path"], "status": "ERROR", "window": wid,
                                  "error": f"{type(exc).__name__}: {exc}", "at": _now()}
                    else:
                        result["window"] = wid
                        if result["status"] == "VERIFIED":
                            verified.add(result["shard"])
                            bytes_this_run += result["bytes"]
                            _submit_probe(result["shard"])  # mine it while it is resident
                        else:
                            failed += 1
                    _append_ledger(result)
                    elapsed = max(time.time() - wall0, 1e-6)
                    _write_json(PROGRESS, {
                        "state": "RUNNING", "mode": MODE, "window": wid,
                        "window_index": index, "window_count": len(windows),
                        "repo": repo, "revision": revision,
                        "shards_verified_total": len(verified),
                        # counted from durable probe files, not the in-memory set, which
                        # only advances at window boundaries and would read 0 for a long
                        # time to anyone watching an unattended run
                        "shards_probed_total": len(probed_shards()),
                        "shards_packed_total": len(packed_shards()),
                        "total_source_shards": 282,
                        "source_fraction_verified": round(
                            sum(by_path[n]["logical_bytes"] for n in verified) / total_weight, 4),
                        "resident_bytes": sum(by_path[n]["logical_bytes"]
                                              for n in _resident(manifest)),
                        "evicted_bytes_this_run": evicted_bytes,
                        "failed_this_run": failed,
                        "aggregate_megabits_per_second": round(
                            bytes_this_run * 8 / elapsed / 1e6, 1),
                        "free_disk_bytes": _free_bytes(),
                        "updated_at": _now(),
                    })
                    break  # re-read disk and refill before waiting on the next completion
            if stopped_for_disk and MODE == "resident":
                break

        if MODE == "stream":
            candidates = set(window.get("evict_after_seal_shards", [])) | _deferred()
            _drain_probes(candidates)  # evidence must be durable before the body goes
            authority = _teacher_authority(candidates, wid)
            freed = _evict(sorted(candidates),
                           verified=verified, probed=probed, packed=packed,
                           protected=_protected_after(schedule, index),
                           authorized=set(authority["authorized"]))
            if freed:
                evicted_bytes += sum(f["bytes"] for f in freed)
                _append_ledger({
                    "event": "EVICT", "window": wid,
                    "basis": "VERIFIED_PROBED_PACKED_TEACHER_CAPTURED",
                    "note": "hash-verified against the sealed manifest, probed, packed, "
                            "and every still-capturable layer it carries is sealed in a "
                            "teacher capsule; layers already destroyed under the "
                            "pre-capture policy are listed as unrecoverable, not captured",
                    "unrecoverable_organs":
                        authority.get("authorized_with_unrecoverable_organs", {}),
                    "shards": [f["shard"] for f in freed],
                    "freed_bytes": sum(f["bytes"] for f in freed), "at": _now(),
                })

    _drain_probes(set(probe_futures) | set(pack_futures))  # nothing unmined at exit
    probe_pool.shutdown(wait=True)
    pack_pool.shutdown(wait=True)
    elapsed = max(time.time() - wall0, 1e-6)
    _write_json(PROGRESS, {
        "state": "ALL_SHARDS_VERIFIED" if len(verified) == 282 and not failed
                 else ("COMPLETED_WITH_FAILURES" if failed else "STOPPED_EARLY"),
        "mode": MODE, "repo": repo, "revision": revision,
        "shards_verified_total": len(verified),
        "shards_probed_total": len(probed_shards()),
        "total_source_shards": 282,
        "source_fraction_verified": round(
            sum(by_path[n]["logical_bytes"] for n in verified) / total_weight, 4),
        "resident_bytes": sum(by_path[n]["logical_bytes"] for n in _resident(manifest)),
        "evicted_bytes_this_run": evicted_bytes, "failed_this_run": failed,
        "aggregate_megabits_per_second": round(bytes_this_run * 8 / elapsed / 1e6, 1),
        "free_disk_bytes": _free_bytes(), "finished_at": _now(),
    })
    try:
        rollup()  # leave the atlas current without needing anyone to run a command
    except Exception as exc:  # noqa: BLE001
        _append_ledger({"event": "ROLLUP_ERROR", "error": f"{type(exc).__name__}: {exc}",
                        "at": _now()})
    return 0


def status() -> int:
    print(json.dumps(_read_json(PROGRESS) if PROGRESS.exists() else {"state": "NOT_STARTED"},
                     indent=2, sort_keys=True))
    return 0


def rollup() -> int:
    """Merge every per-shard probe into one weight atlas, aggregated by organ.

    Deterministic: probe files are read in sorted shard order and every aggregate is a
    weight-weighted sum, so a re-run over the same probes reproduces the same atlas.
    """
    files = sorted(PROBES.glob("*.safetensors.json")) if PROBES.exists() else []
    if not files:
        print(json.dumps({"state": "NO_PROBES_YET"}, indent=2))
        return 0

    def _bucket() -> dict:
        return {"tensors": 0, "elements": 0, "entropy_weighted": 0.0, "entropy_elements": 0,
                "absmax": 0.0, "max_exponent_span": 0}

    by_category: dict[str, dict] = {}
    by_budget_class: dict[str, dict] = {}
    by_section: dict[str, dict] = {}
    total = _bucket()
    shards = []

    for path in files:
        probe = _read_json(path)
        shards.append({"shard": probe["shard"], "tensors": probe["tensor_count"],
                       "elements": probe["elements"],
                       "entropy_bits_per_weight": probe["shard_zeroth_order_entropy_bits_per_weight"]})
        for tensor in probe["tensors"]:
            for group, key in ((by_category, tensor["category"]),
                               (by_budget_class, tensor["budget_class"]),
                               (by_section, tensor["section"])):
                bucket = group.setdefault(key, _bucket())
                for target in (bucket, total) if group is by_category else (bucket,):
                    target["tensors"] += 1
                    target["elements"] += tensor["elements"]
                    target["absmax"] = max(target["absmax"], tensor["absmax"])
                    target["max_exponent_span"] = max(target["max_exponent_span"],
                                                      tensor["exponent_span_log2"])
                    if tensor.get("zeroth_order_entropy_bits") is not None:
                        target["entropy_weighted"] += (
                            tensor["zeroth_order_entropy_bits"] * tensor["elements"])
                        target["entropy_elements"] += tensor["elements"]

    def _finish(bucket: dict) -> dict:
        entropy = (bucket["entropy_weighted"] / bucket["entropy_elements"]
                   if bucket["entropy_elements"] else None)
        return {
            "tensors": bucket["tensors"], "elements": bucket["elements"],
            "zeroth_order_entropy_bits_per_weight": round(entropy, 6) if entropy else None,
            "absmax": bucket["absmax"], "max_exponent_span_log2": bucket["max_exponent_span"],
        }

    atlas = {
        "schema": "hawking.glm52.source_weight_atlas.v1",
        "note": "Zeroth-order entropy is a context-free coder's rate on the BF16 values as "
                "stored. It is an empirical reference point, NOT an achievability bound and "
                "NOT a capability result: it models no weight structure and no error "
                "tolerance, so a real representation may legally land either side of it.",
        "shards_probed": len(files), "total_source_shards": 282,
        "generated_at": _now(),
        "total": _finish(total),
        "by_category": {k: _finish(v) for k, v in sorted(by_category.items())},
        "by_budget_class": {k: _finish(v) for k, v in sorted(by_budget_class.items())},
        "by_section": {k: _finish(v) for k, v in sorted(by_section.items())},
        "shards": shards,
    }
    _write_json(ROLLUP, atlas)
    summary = {k: v for k, v in atlas.items() if k not in ("shards", "by_section")}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


SAFE_TO_LEAVE_JSON = STATE_DIR / "GLM52_SAFE_TO_LEAVE_STATUS.json"
SAFE_TO_LEAVE_MD = STATE_DIR / "GLM52_SAFE_TO_LEAVE_STATUS.md"
# Keychain service names owned by glm52_telegram.  Presence is checked, never the value.
TELEGRAM_SERVICES = (
    "com.hawking.glm52.gravity.telegram.bot-token",
    "com.hawking.glm52.gravity.telegram.chat-id",
)


def _run(argv: list[str], timeout: float = 5.0) -> str:
    import subprocess

    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _host_health() -> dict:
    swap_text = _run(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
    thermal = _run(["/usr/bin/pmset", "-g", "therm"])
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from emergency_detached_campaign import parse_swap_used

        swap_used = parse_swap_used(swap_text)
    except Exception:  # noqa: BLE001 - a status writer never fails on a parse
        swap_used = None
    usage = shutil.disk_usage(str(SOURCE_ROOT if SOURCE_ROOT.exists() else ROOT))
    return {
        "free_disk_bytes": usage.free,
        "disk_floor_bytes": DISK_FLOOR_BYTES,
        "disk_above_floor": usage.free > DISK_FLOOR_BYTES,
        "physical_memory_bytes": int(_run(["/usr/sbin/sysctl", "-n", "hw.memsize"]) or 0),
        "swap_used_bytes": swap_used,
        "swap_raw": swap_text.strip(),
        "thermal_green": "No thermal warning level has been recorded" in thermal,
    }


def _controller() -> dict:
    pids = [int(value) for value in _run(
        ["/usr/bin/pgrep", "-f", "glm52_source_fetch.py run"]).split()]
    live = [pid for pid in pids if pid != os.getpid()]
    label = _run(["/bin/launchctl", "list", "com.hawking.glm52.source-fetch"])
    caffeinate = bool(_run(["/usr/bin/pgrep", "-f", "caffeinate.*glm52_source_fetch"]).strip())
    pgid = None
    if live:
        try:
            pgid = os.getpgid(live[0])
        except OSError:
            pgid = None
    heartbeat = None
    if PROGRESS.exists():
        heartbeat = round(time.time() - PROGRESS.stat().st_mtime, 1)
    return {
        "pids": live,
        "pgid": pgid,
        "launchd_label": "com.hawking.glm52.source-fetch",
        "launchd_loaded": bool(label.strip()),
        "caffeinate_active": caffeinate,
        "lease_path": str(LOCK),
        "lease_held": bool(live),
        "heartbeat_age_seconds": heartbeat,
        "eviction_paused": EVICTION_PAUSED,
    }


def safe_to_leave() -> int:
    """Write the local status pair that replaces Telegram as the leave-it signal."""
    import glm52_teacher_capture as teacher

    progress = _read_json(PROGRESS) if PROGRESS.exists() else {"state": "NOT_STARTED"}
    rows = _ledger_rows()
    manifest = _read_json(MANIFEST)
    by_path = {f["path"]: f for f in manifest["files"] if f.get("is_weight")}
    resident = _resident(manifest)
    evictions = [row for row in rows if row.get("event") == "EVICT"]
    faults = [row for row in rows
              if str(row.get("status", "")).endswith("ERROR")
              or str(row.get("event", "")).endswith("ERROR")
              or row.get("status") in {"HASH_MISMATCH", "SIZE_MISMATCH"}]
    packed = sorted(COMPACT.glob("*.gravity")) if COMPACT.exists() else []
    pack_rows = [row for row in rows if row.get("status") == "PACKED"]
    teacher_status = teacher.status()
    ledger_rows = teacher_status["ledger_rows"]

    health = _host_health()
    controller = _controller()
    telegram = "OPTIONAL_CONFIGURED" if all(
        _run(["/usr/bin/security", "find-generic-password", "-s", service]).strip()
        for service in TELEGRAM_SERVICES
    ) else "OPTIONAL_NOT_CONFIGURED"

    blockers = []
    if not controller["pids"]:
        blockers.append("no live controller process")
    if not controller["launchd_loaded"]:
        blockers.append("launchd job is not loaded")
    if not health["disk_above_floor"]:
        blockers.append("free disk is at or below the floor")
    if teacher_status["capsule_count"] == 0:
        blockers.append("no sealed teacher capsule exists")
    if teacher_status["capturable_now"]:
        blockers.append(
            f"{len(teacher_status['capturable_now'])} resident layers are uncaptured"
        )
    if EVICTION_PAUSED:
        blockers.append("eviction is paused, so the stream cannot complete unattended")

    status_obj = {
        "schema": "hawking.glm52.safe_to_leave.v1",
        "endpoint": "SAFE_TO_LEAVE" if not blockers else "NOT_SAFE_TO_LEAVE",
        "blockers": blockers,
        "controller": controller,
        "source": {
            "state": progress.get("state"),
            "current_window": progress.get("window"),
            "shards_verified": progress.get("shards_verified_total"),
            "shards_probed": progress.get("shards_probed_total"),
            "shards_packed": progress.get("shards_packed_total"),
            "total_source_shards": 282,
            "resident_shards": len(resident),
            "resident_bytes": sum(by_path[n]["logical_bytes"] for n in resident),
            "fraction_verified": progress.get("source_fraction_verified"),
        },
        "teacher": {
            "capsule_count": teacher_status["capsule_count"],
            "captured_layers": teacher_status["captured_layers"],
            "capsule_bytes_total": teacher_status["capsule_bytes_total"],
            "latest_capsule_seal": teacher_status["latest_capsule_seal"],
            "ledger_rows": ledger_rows,
            "ledger_path": teacher_status["ledger_path"],
            "uncaptured_resident_layers": teacher_status["capturable_now"],
            "policy_sealed": teacher_status["policy_sealed"],
        },
        "eviction": {
            "paused": EVICTION_PAUSED,
            "gate": "VERIFIED + PROBED + PACKED + TEACHER_CAPTURED + NOT_CARRIED_FORWARD",
            "events": len(evictions),
            "last_eviction_at": evictions[-1]["at"] if evictions else None,
            "last_eviction_basis": evictions[-1].get("basis") if evictions else None,
            "last_eviction_shards": evictions[-1].get("shards", []) if evictions else [],
        },
        "compact": {
            "format": ".gravity",
            "shards": len(packed),
            "bytes": sum(path.stat().st_size for path in packed),
            "mean_whole_shard_bpw": round(
                sum(row["whole_shard_bpw"] for row in pack_rows) / len(pack_rows), 6
            ) if pack_rows else None,
            "root": str(COMPACT),
        },
        "host": health,
        "telegram": telegram,
        "current_experiment": f"source traversal {progress.get('window')} with teacher "
                              f"capture armed",
        "next_experiment": "BASELINE_A block-level trajectory verdict "
                           "(REACHABLE / REPAIRABLE / DEAD / INVALID)",
        "last_fault": faults[-1] if faults else None,
        "faults_total": len(faults),
        "updated_at": _now(),
    }
    _write_json(SAFE_TO_LEAVE_JSON, status_obj)

    lines = [
        "# GLM-5.2 safe-to-leave status",
        "",
        f"- endpoint: **{status_obj['endpoint']}**",
        f"- blockers: {', '.join(blockers) if blockers else 'none'}",
        f"- controller pids: {controller['pids']} pgid {controller['pgid']} "
        f"launchd {'loaded' if controller['launchd_loaded'] else 'ABSENT'} "
        f"caffeinate {'active' if controller['caffeinate_active'] else 'ABSENT'}",
        f"- lease: {controller['lease_path']} held={controller['lease_held']} "
        f"heartbeat age {controller['heartbeat_age_seconds']}s",
        f"- source: {status_obj['source']['shards_verified']}/282 verified, "
        f"{status_obj['source']['resident_shards']} resident, "
        f"window {status_obj['source']['current_window']}, "
        f"state {status_obj['source']['state']}",
        f"- teacher: {teacher_status['capsule_count']} capsules, layers "
        f"{teacher_status['captured_layers']}, ledger {ledger_rows} rows, "
        f"latest seal {teacher_status['latest_capsule_seal']}",
        f"- eviction: paused={EVICTION_PAUSED}, {len(evictions)} events, last "
        f"{status_obj['eviction']['last_eviction_at']}",
        f"- compact: {status_obj['compact']['shards']} .gravity shards, "
        f"{status_obj['compact']['bytes']} bytes, mean BPW "
        f"{status_obj['compact']['mean_whole_shard_bpw']}",
        f"- disk: {health['free_disk_bytes']} free against a "
        f"{health['disk_floor_bytes']} floor",
        f"- swap: {health['swap_raw']} | thermal green: {health['thermal_green']}",
        f"- telegram: {telegram}",
        f"- next experiment: {status_obj['next_experiment']}",
        f"- updated: {status_obj['updated_at']}",
        "",
    ]
    tmp = SAFE_TO_LEAVE_MD.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines))
    os.replace(tmp, SAFE_TO_LEAVE_MD)
    print(json.dumps(status_obj, indent=2, sort_keys=True))
    return 0


def selftest() -> int:
    """Schedule/eviction invariants against the real sealed artifacts.  No network, no writes."""
    manifest = _read_json(MANIFEST)
    schedule = _read_json(SCHEDULE)
    by_path = {f["path"]: f for f in manifest["files"] if f.get("is_weight")}
    assert len(by_path) == 282, len(by_path)

    windows = schedule["windows"]
    fetched: list[str] = []
    for window in windows:
        fetched.extend(window["new_fetch_shards"])
    assert len(fetched) == 282, f"schedule must fetch every shard once, got {len(fetched)}"
    assert set(fetched) == set(by_path), "schedule fetch set must equal the manifest weight set"
    assert len(set(fetched)) == len(fetched), "no shard may be fetched twice"

    # Eviction must never drop something a later window carries in, and residency must
    # stay under the ceiling at every step -- that is what makes a full traversal fit.
    ceiling = _read_json(POLICY)["derived"]["preregistered_usable_raw_window_bytes"]
    resident: set[str] = set()
    peak = 0
    for index, window in enumerate(windows):
        resident.update(window["new_fetch_shards"])
        peak = max(peak, sum(by_path[n]["logical_bytes"] for n in resident))
        protected = _protected_after(schedule, index)
        evictable = [n for n in window.get("evict_after_seal_shards", []) if n not in protected]
        for name in window.get("evict_after_seal_shards", []):
            assert name not in protected, f"{name} evicted while a later window carries it in"
        resident.difference_update(evictable)
    assert peak <= ceiling, f"peak residency {peak} exceeds ceiling {ceiling}"

    print(json.dumps({
        "selftest": "PASS", "shards": len(by_path), "windows": len(windows),
        "peak_resident_gb": round(peak / 1e9, 1),
        "residency_ceiling_gb": round(ceiling / 1e9, 1),
        "full_traversal_fits": True,
        "source_tb": round(sum(by_path[n]["logical_bytes"] for n in by_path) / 1e12, 3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    raise SystemExit({"run": run, "status": status, "selftest": selftest,
                      "rollup": rollup, "safe-to-leave": safe_to_leave}[command]())

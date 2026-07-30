#!/usr/bin/env python3.12
"""Math-Preserve PASS 3: profile-conditioned packing from PASS 2's frozen manifest.

Per-shard loop: VERIFY -> PACK the exact per-tensor decision frozen by PASS 2 ->
EVICT. Re-streams the source PASS 1 already evicted -- the user's specification
names this explicitly ("Re-stream every required source shard"), so the second
source pass is structural, not accidental.

After 282/282 shards, finalization grades coverage against the independent official
weight map, verifies every payload hash and shard rate, writes the runtime model
index, installs the tokenizer tables, and constructs an itemized one-bit ledger
from actual on-disk bytes.  Prediction only controls admission; actual bytes decide
whether `<Base>-H0.98-Math-Preserve.gravity` is sealable.

    python3.12 tools/prometheus/math_pass3_pack.py status
    python3.12 tools/prometheus/math_pass3_pack.py run [--limit-shards N]
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONDENSE = REPO / "tools/condense"
for _p in (HERE, CONDENSE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from glm52_common import resolve_artifact  # noqa: E402

MANIFEST = resolve_artifact("PROMETHEUS_MATH_ALLOCATION_MANIFEST.json")
GRAPH = resolve_artifact("GLM52_SHARD_DEPENDENCY_GRAPH.json")
OFFICIAL_MANIFEST = REPO / "evidence" / "glm52" / "GLM52_OFFICIAL_MANIFEST.json"

STATE_DIR = Path(
    "/Users/scammermike/Library/Application Support/Hawking/GLM52MathPrometheus/pass3"
)
SOURCE_ROOT = STATE_DIR / "source"
LEDGER = STATE_DIR / "PASS3_LEDGER.jsonl"
PROGRESS = STATE_DIR / "progress.json"
TELEMETRY = STATE_DIR / "PASS3_TELEMETRY.jsonl"
LOCK = STATE_DIR / "pass3.lock"
COMPACT = Path(
    "/Users/scammermike/Library/Application Support/Hawking/Models/GLM-5.2/"
    "b4734de4facf877f85769a911abafc5283eab3d9/"
    "GLM-5.2-H0.98-Math-Preserve.gravity"
)
GENERAL = COMPACT.parent / "General-R0"
RECEIPT = REPO / "evidence" / "glm52" / "GLM52_H0_98_MATH_PRESERVE_RECEIPT.json"

os.environ.setdefault("HF_HOME", str(STATE_DIR / "hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(STATE_DIR / "hf_cache"))
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

DISK_FLOOR_BYTES = int(os.environ.get("GLM52_PASS3_DISK_FLOOR_BYTES", 75 * 10**9))
HASH_CHUNK = 16 << 20


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


# The fetch producer and the sealing consumer both write the ledger now. It is the
# campaign's evidence authority, so serialize the append rather than rely on small
# writes happening not to interleave.
_LEDGER_LOCK = threading.Lock()


def _append_ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with _LEDGER_LOCK, open(LEDGER, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _append_telemetry(row: dict) -> None:
    """Write nondeterministic timing only to a side channel, never a receipt."""
    TELEMETRY.parent.mkdir(parents=True, exist_ok=True)
    with open(TELEMETRY, "a") as fh:
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


def packed_shards() -> set[str]:
    if not COMPACT.exists():
        return set()
    return {p.stem + ".safetensors" for p in COMPACT.glob("*.gravity")}


def manifest_ready() -> tuple[bool, dict | None, str]:
    if not MANIFEST.exists():
        return False, None, f"{MANIFEST} does not exist yet -- run PASS 2's `freeze`"
    manifest = json.loads(MANIFEST.read_text())
    if not manifest.get("complete"):
        missing = manifest.get("sparse_layers_missing_evidence", [])
        return False, manifest, (
            f"{MANIFEST} exists but complete=false ({len(missing)} sparse layers "
            "still missing capsule evidence) -- PASS 2 should have refused to "
            "write this; do not pack against it"
        )
    auction = manifest.get("global_byte_auction") or {}
    if not auction.get("allocation_complete"):
        return False, manifest, "PASS2 manifest has no complete global byte auction"
    decisions = auction.get("tensor_decisions") or {}
    expected = int(_read_json(GRAPH)["tensor_count"])
    if len(decisions) != expected:
        return False, manifest, (
            f"PASS2 auction names {len(decisions)} tensors, dependency graph names "
            f"{expected}; do not pack an implicit allocation"
        )
    if auction.get("predicted_complete_bytes_with_reserve", 1 << 100) > \
            auction.get("max_complete_physical_bytes", 0):
        return False, manifest, "PASS2 auction exceeds its own frozen complete-byte ceiling"
    return True, manifest, ""


def tensor_rate_override(manifest: dict) -> dict[str, str]:
    """The packer receives the frozen decision for every exact tensor name."""
    return dict(manifest["global_byte_auction"]["tensor_decisions"])


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
        return {"shard": name, "status": "SIZE_MISMATCH", "at": _now()}
    observed = _sha256_file(got_path)
    if observed != row["lfs_sha256"]:
        quarantine = got_path.with_suffix(got_path.suffix + ".badhash")
        os.replace(got_path, quarantine)
        return {"shard": name, "status": "HASH_MISMATCH", "at": _now()}
    return {
        "shard": name, "status": "VERIFIED", "bytes": size, "sha256": observed,
        "seconds": round(elapsed, 2),
        "megabits_per_second": round(size * 8 / elapsed / 1e6, 1), "at": _now(),
    }


PACK_WORKERS = int(os.environ.get("GLM52_PASS3_PACK_WORKERS", "4"))
PACK_THREADS_PER_WORKER = int(
    os.environ.get("GLM52_PASS3_TORCH_THREADS_PER_WORKER", "7")
)


def _worker_init() -> None:
    """Give each spawned worker an independent, bounded CPU compute pool."""
    import glm52_pack as pack
    import torch

    torch.set_num_threads(PACK_THREADS_PER_WORKER)
    torch.set_num_interop_threads(1)
    pack.forge._device = lambda: torch.device("cpu")


def _worker_probe() -> tuple[int, str]:
    """Pickleable spawn-path probe used by selftest."""
    import glm52_pack as pack
    import torch

    return torch.get_num_threads(), str(pack.forge._device())


def _pack_one(name: str, rows: list[dict], override: dict) -> dict:
    """Runs in a spawned worker against a unique source and destination shard.

    Workers receive the same immutable manifest and keep all fit state process-local.
    """
    import glm52_pack as pack
    import torch

    # Defensive for direct invocation in tests; ProcessPool runs _worker_init first.
    pack.forge._device = lambda: torch.device("cpu")
    telemetry = {}
    receipt = pack.pack_shard(
        SOURCE_ROOT / name,
        rows,
        COMPACT,
        rate_override=override,
        telemetry=telemetry,
    )
    return {"shard": name, "compact_bytes": receipt["compact_bytes"],
            "complete_bpw": receipt["complete_bpw"], "telemetry": telemetry}


def _fetch_loop(pending: list[str], by_path: dict, repo: str, revision: str,
                ready: "queue.Queue", stop: "threading.Event") -> None:
    """Producer: keep verified source shards resident just ahead of the packers.

    Runs on its own thread so a ~22 s download is never wall-clock the 28 cores spend
    idle.  The queue is small on purpose: each shard is ~5.4 GB, so running far ahead
    buys nothing and spends the disk headroom the floor check exists to protect.
    """
    try:
        for name in pending:
            if stop.is_set():
                break
            # A past VERIFIED row proves bytes once passed the source hash; it does not
            # mean PASS1/PASS3 eviction left those bytes resident.
            if not (SOURCE_ROOT / name).is_file() or name not in verified_shards():
                if _free_bytes() < DISK_FLOOR_BYTES:
                    _append_ledger({"event": "DISK_FLOOR_STOP", "shard": name, "at": _now()})
                    sys.stderr.write(f"disk floor reached before {name}; stopping fetch\n")
                    break
                result = _fetch_one(by_path[name], repo, revision)
                _append_ledger(result)
                if result["status"] != "VERIFIED":
                    sys.stderr.write(f"PASS3 fetch failed: {result}\n")
                    continue
            ready.put(name)
    finally:
        ready.put(None)  # sentinel: no more shards will arrive


def run(*, limit_shards: int | None = None) -> int:
    import fcntl
    import multiprocessing
    import queue
    import threading
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    # gravity_forge._device() defaults to MPS whenever available. On this machine
    # that path has previously produced a command-buffer error that silently
    # poisoned a LATER, unrelated tensor's PQ indices rather than raising where the
    # corruption happened -- the failure surfaced elsewhere as "index does not fit
    # in 7 bits" on a tensor that was never near the actual fault. A crash here is
    # the safe outcome; the dangerous one is a bad pack that doesn't crash. Forced
    # CPU for the same reason this campaign forced it everywhere else unattended
    # correctness mattered: correctness over speed, verified once, not re-litigated
    # per script. (Re-applied per-worker in _pack_one; done here too so a
    # zero-worker/serial code path would still be safe.)

    ready, manifest, reason = manifest_ready()
    if not ready:
        raise SystemExit(reason)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(LOCK), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.stderr.write("another PASS3 run holds the lock; exiting\n")
        return 0

    override = tensor_rate_override(manifest)
    official = _read_json(OFFICIAL_MANIFEST)
    graph = _read_json(GRAPH)
    repo, revision = official["repo"], official["revision"]
    by_path = {f["path"]: f for f in official["files"] if f.get("is_weight")}
    tensors_by_shard: dict[str, list[dict]] = {}
    for tensor in graph["tensors"]:
        tensors_by_shard.setdefault(tensor["shard"], []).append(tensor)

    shard_names = sorted(by_path)
    if limit_shards is not None:
        shard_names = shard_names[:limit_shards]
    already_packed = packed_shards()
    pending = [n for n in shard_names if n not in already_packed]

    # Dynamic work queue, not fixed PACK_WORKERS-sized barriers.  Measured on the first
    # real R4 batch (4 shards, 2607 s wall): the slowest shard finished 94.4 s after the
    # fastest, and that batch's fetch ran serially in front of a fully idle pool, so the
    # barrier shape cost ~3.6 percent to stragglers and ~3.4 percent to fetch.  Here a
    # producer thread keeps sources resident ahead of the pool, a worker that finishes
    # takes the next ready shard immediately, and each shard seals -- receipt, telemetry,
    # eviction, progress -- on its own completion rather than at a batch edge.  Nothing
    # about the artifact changes; only when a worker is allowed to be idle does.
    #
    # A ThreadPoolExecutor does not create independent PyTorch intra-op pools: four fits
    # merely interleave on one ~20-thread process and the first real R4 batch consumed
    # essentially the sum of four shard compute times.  Spawned workers each receive seven
    # threads on this 28-core host, giving real process-level parallelism without
    # inheriting any parent torch state.
    ready: queue.Queue = queue.Queue(maxsize=2)
    stop = threading.Event()
    fetcher = threading.Thread(
        target=_fetch_loop, args=(pending, by_path, repo, revision, ready, stop),
        name="pass3-fetch", daemon=True,
    )
    fetcher.start()

    run_started = time.perf_counter()
    in_flight: dict = {}
    sealed = 0
    starved_worker_seconds = 0.0
    last_measured = run_started
    drained = False

    def seal(future) -> None:
        """Publish exactly one shard, independently of every other shard in flight."""
        nonlocal sealed
        name = in_flight.pop(future)
        try:
            receipt = future.result()
        except Exception as exc:  # noqa: BLE001 - one bad shard must not kill the run
            _append_ledger({"event": "PACK_ERROR", "shard": name,
                            "error": f"{type(exc).__name__}: {exc}", "at": _now()})
            sys.stderr.write(f"PASS3 pack failed for {name}: {exc}\n")
            return
        timing = receipt.pop("telemetry")
        timing.update({"event": "SHARD_TIMING", "scheduler": "dynamic_queue",
                       "shard": name, "completed_at": _now()})
        _append_telemetry(timing)
        _append_ledger({"event": "PACKED", **receipt, "at": _now()})
        target = SOURCE_ROOT / name
        if target.exists():
            size = target.stat().st_size
            target.unlink()
            _append_ledger({"event": "EVICT", "shard": name, "bytes": size, "at": _now()})
        sealed += 1
        _write_json(PROGRESS, {
            "shards_packed": len(packed_shards()), "shards_total": len(shard_names),
            "sealed_this_run": sealed, "scheduler": "dynamic_queue",
            "last_sealed": name, "at": _now(),
        })

    try:
        with ProcessPoolExecutor(
            max_workers=PACK_WORKERS,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
        ) as pool:
            while True:
                # Fill every free slot before blocking on a completion, so a worker is
                # never idle while a verified source is already sitting in the queue.
                # Block on the queue only when nothing is in flight: with work running,
                # an empty queue means the fetcher is behind and waiting on it would
                # reintroduce exactly the serialization this replaced.
                while not drained and len(in_flight) < PACK_WORKERS:
                    try:
                        name = ready.get(block=not in_flight)
                    except queue.Empty:
                        break
                    if name is None:
                        drained = True
                        break
                    future = pool.submit(
                        _pack_one, name, tensors_by_shard.get(name, []), override)
                    in_flight[future] = name
                if not in_flight:
                    break
                now = time.perf_counter()
                starved_worker_seconds += (PACK_WORKERS - len(in_flight)) * (now - last_measured)
                last_measured = now
                # `wait` watches futures, not the queue.  With a free slot and a fetch
                # still to land, blocking on futures alone would hold the pool at
                # whatever occupancy it happened to reach when the queue last ran dry --
                # observed live as one busy worker out of four.  Poll for the refill in
                # exactly that case; block outright when the pool is full or the
                # producer is finished, so a full pipeline costs no wakeups at all.
                done, _ = wait(
                    list(in_flight), return_when=FIRST_COMPLETED,
                    timeout=(None if drained or len(in_flight) == PACK_WORKERS else 5.0))
                for future in done:
                    seal(future)
    finally:
        stop.set()
        # Unblock a producer parked on a full queue so the thread can retire.
        try:
            ready.get_nowait()
        except queue.Empty:
            pass

    wall = time.perf_counter() - run_started
    if sealed:
        _append_telemetry({
            "schema": "hawking.glm52.pass3_scheduler_telemetry.v1",
            "event": "SCHEDULER_TIMING", "at": _now(), "scheduler": "dynamic_queue",
            "workers": PACK_WORKERS, "shards_sealed": sealed, "wall_seconds": wall,
            "seconds_per_shard": wall / sealed,
            "starvation_worker_seconds": starved_worker_seconds,
            "starvation_percentage": (
                starved_worker_seconds * 100.0 / max(PACK_WORKERS * wall, 1e-12)
            ),
        })

    if limit_shards is None and len(packed_shards()) == len(by_path):
        finalize(manifest)
    return 0


def _install_runtime_tree(source: Path, target: Path) -> dict:
    """Install immutable runtime files by hardlink when possible, copy otherwise."""
    import shutil

    if not source.is_dir():
        raise FileNotFoundError(f"required runtime tree does not exist: {source}")
    linked = copied = existing = 0
    for src in sorted(p for p in source.rglob("*") if p.is_file()):
        dst = target / src.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            existing += 1
            continue
        try:
            os.link(src, dst)
            linked += 1
        except OSError:
            shutil.copy2(src, dst)
            copied += 1
    return {"linked": linked, "copied": copied, "existing": existing}


def _tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _actual_byte_ledger(manifest: dict) -> tuple[dict, dict]:
    """Itemize every actual packaged byte into the binding one-bit ledger."""
    import glm52_pack as pack
    import artifact_client as gravity

    found: set[str] = set()
    decision_mismatches: list[dict] = []
    components = {
        "indices": 0,
        "codebooks": 0,
        "scales": 0,
        "metadata": 0,
        "alignment": 0,
        "protected_islands": 0,
        "doctor": 0,
        "pass_through_tensors": 0,
        "packaging": 0,
        "runtime_tables": 0,
    }
    decisions = manifest["global_byte_auction"]["tensor_decisions"]
    verified = []
    rung_by_name = {r["rung"]: r for r in pack.LADDER}

    for path in sorted(COMPACT.glob("model-*.gravity")):
        verdict = gravity.verify(path)
        verified.append(verdict)
        if not verdict["ok"]:
            raise RuntimeError(f"shard integrity verification failed: {verdict}")
        header = gravity.read_header(path)
        payload_bytes = 0
        for tensor in header["tensors"]:
            name = tensor["name"]
            found.add(name)
            body_bytes = int(tensor["bytes"])
            payload_bytes += body_bytes
            expected = decisions.get(name)
            codec = str(tensor.get("codec", ""))
            if codec.startswith("native."):
                if expected != "native":
                    decision_mismatches.append({
                        "tensor": name, "expected": expected, "observed": codec,
                    })
                reason = str(tensor.get("reason", ""))
                bucket = (
                    "protected_islands"
                    if reason.startswith("PROMETHEUS_")
                    else "pass_through_tensors"
                )
                components[bucket] += body_bytes
                continue

            observed = tensor.get("rung")
            if codec != "gravity-pq" or expected != observed:
                decision_mismatches.append({
                    "tensor": name, "expected": expected,
                    "observed": f"{codec}/{observed}",
                })
            rung = rung_by_name[observed]
            shape = [int(x) for x in tensor["shape"]]
            elements = math.prod(shape)
            cols = shape[-1]
            dim = int(rung["dim"])
            effective_dim = (
                dim if cols % dim == 0 and dim & (dim - 1) == 0
                else cols & -cols
            )
            count = elements // effective_dim
            index_bytes = math.ceil(
                count * pack.index_bits(int(rung["k"])) / 8
            )
            codebook_bytes = int(rung["k"]) * effective_dim * 2
            metadata_bytes = pack.HEADER_BYTES
            if index_bytes + codebook_bytes + metadata_bytes != body_bytes:
                raise AssertionError(
                    f"{name}: itemized PQ bytes "
                    f"{index_bytes + codebook_bytes + metadata_bytes} != {body_bytes}"
                )
            components["indices"] += index_bytes
            components["codebooks"] += codebook_bytes
            components["metadata"] += metadata_bytes
        components["packaging"] += path.stat().st_size - payload_bytes

    missing = sorted(set(decisions) - found)
    undeclared = sorted(found - set(decisions))
    if missing or undeclared or decision_mismatches:
        raise RuntimeError(
            "artifact differs from frozen allocation: "
            f"missing={len(missing)}, undeclared={len(undeclared)}, "
            f"decision_mismatches={len(decision_mismatches)}; "
            f"examples={(missing + undeclared + decision_mismatches)[:3]}"
        )

    tokenizer = COMPACT / "tokenizer"
    components["runtime_tables"] = _tree_bytes(tokenizer)
    sidecars = [
        p for p in COMPACT.rglob("*")
        if p.is_file()
        and p.suffix != ".gravity"
        and tokenizer not in p.parents
    ]
    components["packaging"] += sum(p.stat().st_size for p in sidecars)

    package_bytes = _tree_bytes(COMPACT)
    itemized_bytes = sum(components.values())
    if itemized_bytes != package_bytes:
        raise AssertionError(
            f"complete byte ledger itemizes {itemized_bytes}, package has {package_bytes}"
        )

    foundry = REPO / "tools/foundry"
    if str(foundry) not in sys.path:
        sys.path.insert(0, str(foundry))
    from one_bit_ceiling import CompleteByteLedger, assert_complete_bpw_le_one

    ledger = CompleteByteLedger(
        **{name: value * 8 for name, value in components.items()},
        metadata_alignment_reserve_bits=0,
        note="Actual packaged bytes for GLM-5.2-H0.98-Math-Preserve.gravity; "
             "no modeled-zero or payload-only exclusions.",
    )
    denominator = int(
        manifest["global_byte_auction"]["logical_weight_denominator"]
    )
    law_receipt = assert_complete_bpw_le_one(ledger, denominator)
    return (
        {
            **ledger.as_dict(denominator),
            "component_bytes": components,
            "actual_package_bytes": package_bytes,
            "itemization_reconciles": True,
        },
        {
            "shards_verified": len(verified),
            "all_shards_ok": all(v["ok"] for v in verified),
            "frozen_tensor_decisions_verified": len(found),
            "decision_mismatches": 0,
            "one_bit_law": law_receipt,
        },
    )


def finalize(manifest: dict | None = None) -> dict:
    """Turn the packed shard set into the verified standalone Odyssey substrate."""
    import hashlib
    import glm52_assemble as assembler
    import artifact_client as gravity

    if manifest is None:
        ready, manifest, reason = manifest_ready()
        if not ready:
            raise SystemExit(reason)
    coverage = assembler.check(COMPACT)
    if not coverage["complete"]:
        raise RuntimeError(
            f"refusing Math-Preserve finalization over incomplete coverage: {coverage}"
        )

    tokenizer_install = _install_runtime_tree(
        GENERAL / "tokenizer", COMPACT / "tokenizer"
    )
    # Copy the allocation manifest BYTE FOR BYTE rather than re-serializing it. The receipt
    # records the sha256 of the repository file, and the Odyssey-ready audit independently
    # requires that BOTH the repository copy and the artifact copy hash to that value. A
    # re-serialized copy carries identical JSON content and different bytes, so it fails an
    # integrity check that is correct to insist. A content-addressed artifact should carry
    # the exact bytes it was packed against, not a pretty-printed equivalent.
    (COMPACT / "PROMETHEUS_MATH_ALLOCATION_MANIFEST.json").write_bytes(MANIFEST.read_bytes())
    _write_json(COMPACT / "COVERAGE.json", coverage)

    packed, shards = assembler.packed_index(COMPACT)
    shard_hashes = {}
    for name in shards:
        header = gravity.read_header(COMPACT / name)
        shard_hashes[name] = header["integrity"]["body_sha256"]
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    index = {
        "schema": "hawking.gravity.model_index.v1",
        "assembled_at": _now(),
        "model": {
            "repo": "zai-org/GLM-5.2",
            "revision": _read_json(OFFICIAL_MANIFEST)["revision"],
            "representation": "QUANTIZED_TRANSFORMER",
            "profile": "Math-Preserve",
            "rate_label": "H0.98",
        },
        "architecture": assembler.synthesize_architecture(),
        "shards": sorted(shards),
        "shard_count": len(shards),
        "tensor_count": len(packed),
        "weight_map": {name: entry["shard"] for name, entry in sorted(packed.items())},
        "shard_body_sha256": shard_hashes,
        "allocation_manifest_sha256": manifest_sha,
        "coverage": {
            key: coverage[key]
            for key in ("official_tensors", "dispositions", "verdict")
        },
        "byte_provenance": "PASS3 streamed from immutable source; every shard "
                           "hash-verified before packing and evicted after atomic output.",
    }
    _write_json(COMPACT / "model.gravity.index.json", index)

    byte_ledger, verification = _actual_byte_ledger(manifest)
    actual_bytes = int(byte_ledger["actual_package_bytes"])
    auction = manifest["global_byte_auction"]
    if actual_bytes > int(auction["max_complete_physical_bytes"]):
        raise RuntimeError(
            f"actual Math-Preserve package {actual_bytes} bytes exceeds frozen "
            f"H0.98 ceiling {auction['max_complete_physical_bytes']}"
        )
    receipt = {
        "schema": "hawking.prometheus.math_preserve_receipt.v1",
        "at": _now(),
        "artifact": str(COMPACT),
        "artifact_index_sha256": hashlib.sha256(
            (COMPACT / "model.gravity.index.json").read_bytes()
        ).hexdigest(),
        "allocation_manifest_sha256": manifest_sha,
        "rate_label": "H0.98",
        "coverage": coverage,
        "verification": verification,
        "byte_ledger": byte_ledger,
        "target_complete_bpw": auction["target_complete_bpw"],
        "target_max_bytes": auction["max_complete_physical_bytes"],
        "actual_complete_bpw": actual_bytes * 8 / int(
            auction["logical_weight_denominator"]
        ),
        "headroom_bytes": int(auction["max_complete_physical_bytes"]) - actual_bytes,
        "tokenizer_install": tokenizer_install,
        "odyssey_input_ready": True,
    }
    _write_json(RECEIPT, receipt)
    _append_ledger({
        "event": "FINALIZED", "artifact": str(COMPACT),
        "actual_bytes": actual_bytes,
        "actual_complete_bpw": receipt["actual_complete_bpw"],
        "receipt": str(RECEIPT), "at": _now(),
    })
    _write_json(PROGRESS, {
        "state": "COMPLETE", "shards_packed": len(packed_shards()),
        "shards_total": 282, "artifact": str(COMPACT),
        "actual_complete_bpw": receipt["actual_complete_bpw"], "at": _now(),
    })
    return receipt


def status() -> dict:
    ready, manifest, reason = manifest_ready()
    auction = (manifest or {}).get("global_byte_auction", {})
    progress = _read_json(PROGRESS) if PROGRESS.is_file() else {}
    return {
        "schema": "hawking.prometheus.math_pass3_status.v1",
        "at": _now(),
        "ready_to_pack": ready,
        "state": progress.get("state", "IN_PROGRESS" if packed_shards() else "READY"),
        "reason": reason if not ready else (
            "artifact finalized and verified" if RECEIPT.is_file()
            else "global allocation complete; run `run` to pack"
        ),
        "manifest_layers": len(manifest["per_layer"]) if manifest else 0,
        "manifest_tensor_decisions": auction.get("tensor_decision_count", 0),
        "target_complete_bpw": auction.get("target_complete_bpw"),
        "predicted_complete_bpw_with_reserve": (
            auction.get("predicted_complete_bpw_with_reserve")
        ),
        "shards_packed": len(packed_shards()),
        "shards_total": 282,
        "artifact": str(COMPACT),
        "receipt": str(RECEIPT) if RECEIPT.is_file() else None,
    }


def selftest() -> None:
    """No live artifact required: prove exact per-tensor override extraction."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    synthetic = {
        "global_byte_auction": {
            "tensor_decisions": {
                "model.layers.3.mlp.experts.7.gate_proj.weight": "native",
                "model.layers.3.mlp.experts.8.gate_proj.weight": "R4",
                "model.embed_tokens.weight": "native",
            },
        }
    }
    override = tensor_rate_override(synthetic)
    assert override == synthetic["global_byte_auction"]["tensor_decisions"]
    assert override is not synthetic["global_byte_auction"]["tensor_decisions"], \
        "worker override must be a copy, not mutate the frozen manifest"
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
    ) as pool:
        threads, device = pool.submit(_worker_probe).result(timeout=30)
    assert threads == PACK_THREADS_PER_WORKER, (threads, PACK_THREADS_PER_WORKER)
    assert device == "cpu", device
    print("math_pass3_pack selftest PASS")


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "status"
    if command == "status":
        print(json.dumps(status(), indent=2, sort_keys=True))
        return 0
    if command == "selftest":
        selftest()
        return 0
    if command == "finalize":
        receipt = finalize()
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if command == "run":
        limit = None
        if "--limit-shards" in argv:
            limit = int(argv[argv.index("--limit-shards") + 1])
        return run(limit_shards=limit)
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

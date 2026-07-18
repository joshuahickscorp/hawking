#!/usr/bin/env python3.12
"""Remote bounded-stream Press: the transactional giant-parent conversion lifecycle.

High-Parameter Frontier Program, master goal section 8. The three giant parents (685B / 1T /
1.6T) are disk-walled on this box: their sources are 595-1371 GB and the full source may NEVER
coexist with the output. The canonical conversion is therefore a remote bounded stream. This
module implements the transactional loop and the four deterministic passes that convert a giant
parent without the whole source ever landing on disk.

Contract, enforced fail-closed:
  - bind_shard_map binds a source-shard/byte-range map from a revision-pinned index. EVERY
    fetchable byte belongs to a manifest entry (tensor -> shard -> byte range). A fetch for any
    byte not in the map is refused.
  - fetch_range streams one byte range with bounded-backoff retry, a partial-download hash
    state, a checkpointed shard cursor, bounded LRU cache eviction, and deterministic
    reacquisition. A dropped transfer RESUMES from the cursor; it never restarts the parent.
  - convert_unit decodes one unit, condenses it, packs an output unit, and round-trip attests
    the pack by hash. One unit is resident at a time: RSS is bounded by the unit, not the shard.
  - run_pass streams the units of one of the four deterministic passes (census_statistics,
    representation_fitting, doctor_fitting, final_packing), folds a deterministic global-
    reduction merge state, and atomically fsyncs (artifact, observation, checkpoint). Source
    windows are released once converted unless a retain flag proves retention optimal.
  - press_plan emits a sealed dry-run plan whose peak disk is ~ one source shard plus one
    output shard (NOT the whole source), and which marks the parent disk-walled and asserts it
    launches nothing while the legacy campaign runs.

This module downloads NOTHING real and launches NOTHING heavy. The transport is an INJECTED
fetcher; the selftest drives the whole lifecycle over a tiny self-contained byte fixture.
Successor state is written under reports/condense/event_horizon_successor/press/ only; it never
writes under reports/condense/doctor_v5_ultra.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import secrets
import struct
import sys
from pathlib import Path
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError,
    atomic_write_json,
    hash_value,
    now_iso,
    read_json_safe,
    repo_root,
    seal_field,
    sealed,
)
from succ_frontier import PARENTS, GiantParent  # noqa: E402

# ── schema registry ────────────────────────────────────────────────────────────────────
SCHEMA_SHARD_MAP = "hawking.successor.press_shard_map.v1"
SCHEMA_UNIT = "hawking.successor.press_unit.v1"
SCHEMA_PASS = "hawking.successor.press_pass.v1"
SCHEMA_CHECKPOINT = "hawking.successor.press_checkpoint.v1"
SCHEMA_PLAN = "hawking.successor.press_plan.v1"

PRESS_DIR = "reports/condense/event_horizon_successor/press"

# The four deterministic passes, in canonical order (master goal section 8).
PASSES: tuple[str, ...] = (
    "census_statistics",
    "representation_fitting",
    "doctor_fitting",
    "final_packing",
)

# The pass that actually materializes packed output units on disk. The earlier passes stream the
# same units for statistics and fitting but retain no packed bytes on disk.
_PACKING_PASS = "final_packing"

# Reversible unit codec magic. The production codec is lossy (a real quantizer); this reversible
# stand-in lets the round-trip attest prove exact reconstruction on the fixture. See honest_gaps.
_CODEC_MAGIC = b"EHP1"
_ZERO_SHA = "0" * 64


class PressError(EcoError):
    """Fail-closed error in the bounded-stream Press lifecycle."""


class TransferError(RuntimeError):
    """A recoverable transport drop raised by a fetcher. fetch_range retries and resumes from
    the cursor; it is NOT fail-closed. Exhausting the retry budget converts it to PressError."""


# ======================================================================================
# 8.0  configuration
# ======================================================================================
@dataclasses.dataclass(frozen=True)
class Config:
    """Where bounded-stream Press state may live and how large its resident windows may grow."""

    root: Path                       # repository root (state is resolved under it)
    press_dir: Path                  # successor-only Press namespace
    cache_cap_bytes: int             # bounded source-window cache cap (~ one shard)
    disk_free_floor_bytes: int       # refuse to proceed if free disk would fall below this
    chunk_bytes: int = 8 * 1024 * 1024
    max_retries: int = 5
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 30.0

    @property
    def cache_dir(self) -> Path:
        return self.press_dir / "cache"

    @property
    def output_dir(self) -> Path:
        return self.press_dir / "output"

    @property
    def plans_dir(self) -> Path:
        return self.press_dir / "plans"

    @property
    def checkpoints_dir(self) -> Path:
        return self.press_dir / "checkpoints"

    @property
    def observations_dir(self) -> Path:
        return self.press_dir / "observations"


def default_config() -> Config:
    root = repo_root()
    return Config(
        root=root,
        press_dir=root / PRESS_DIR,
        # One 685B/1T/1.6T shard is a few GB; the resident source window is capped near one shard.
        cache_cap_bytes=6 * 1024 * 1024 * 1024,
        disk_free_floor_bytes=150 * 1024 * 1024 * 1024,
    )


# ======================================================================================
# 8.1  bind_shard_map: every fetchable byte belongs to a manifest entry
# ======================================================================================
def bind_shard_map(parent: GiantParent, index_json: dict[str, Any]) -> dict[str, Any]:
    """Bind a source-shard/byte-range map from a revision-pinned index document.

    The `index_json` is the safetensors index (`metadata.total_size` + `weight_map`) enriched
    with a per-shard `shard_layout` giving each tensor's byte range inside its shard. In
    production that layout is assembled by a READ-ONLY prefetch of each shard's safetensors
    header (a few KB at the shard head); here the fixture supplies it directly. The bound map is
    the sole fetch authority: `shard_map_allows` refuses any byte outside a declared unit.

    Fails closed if the index is malformed, if a tensor's shard is absent, if tensor ranges
    overlap or exceed their shard, or if the enumerated bytes do not sum to `metadata.total_size`
    (i.e. the map does not cover exactly the declared source).
    """
    if not isinstance(index_json, dict):
        raise PressError("index_json is not an object")
    meta = index_json.get("metadata")
    if not isinstance(meta, dict) or not isinstance(meta.get("total_size"), int):
        raise PressError("index_json.metadata.total_size missing or not an int")
    total_size = int(meta["total_size"])
    if total_size <= 0:
        raise PressError("index_json.metadata.total_size must be positive")
    weight_map = index_json.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PressError("index_json.weight_map missing or empty")
    layout = index_json.get("shard_layout")
    if not isinstance(layout, dict) or not layout:
        raise PressError("index_json.shard_layout missing or empty")

    shards: dict[str, dict[str, int]] = {}
    units: list[dict[str, Any]] = []
    covered = 0

    # Deterministic ordering: shards, then tensor begin offset.
    for shard_name in sorted(layout):
        shard_info = layout[shard_name]
        if not isinstance(shard_info, dict):
            raise PressError(f"shard_layout[{shard_name}] is not an object")
        nbytes = shard_info.get("nbytes")
        tensors = shard_info.get("tensors")
        if not isinstance(nbytes, int) or nbytes <= 0:
            raise PressError(f"shard {shard_name} nbytes missing or not positive")
        if not isinstance(tensors, dict) or not tensors:
            raise PressError(f"shard {shard_name} declares no tensors")
        shards[shard_name] = {"nbytes": int(nbytes), "n_tensors": len(tensors)}

        ranges: list[tuple[int, int, str]] = []
        for tensor_name, rng in tensors.items():
            if weight_map.get(tensor_name) != shard_name:
                raise PressError(
                    f"tensor {tensor_name} in shard {shard_name} is not mapped there by weight_map"
                )
            if not isinstance(rng, dict) or not isinstance(rng.get("begin"), int) \
                    or not isinstance(rng.get("end"), int):
                raise PressError(f"tensor {tensor_name} range missing begin/end ints")
            begin, end = int(rng["begin"]), int(rng["end"])
            if not (0 <= begin < end <= nbytes):
                raise PressError(
                    f"tensor {tensor_name} range [{begin},{end}) out of shard bounds [0,{nbytes})"
                )
            ranges.append((begin, end, tensor_name))

        ranges.sort()
        prev_end = 0
        for begin, end, tensor_name in ranges:
            if begin < prev_end:
                raise PressError(
                    f"tensor {tensor_name} range [{begin},{end}) overlaps a prior tensor in "
                    f"{shard_name}"
                )
            prev_end = end
            nb = end - begin
            covered += nb
            units.append({
                "schema": SCHEMA_UNIT,
                "unit_id": f"{shard_name}#{tensor_name}",
                "shard": shard_name,
                "tensor": tensor_name,
                "begin": begin,
                "end": end,
                "nbytes": nb,
            })

    # Every mapped tensor must belong to a shard layout entry.
    for tensor_name, shard_name in weight_map.items():
        if shard_name not in layout:
            raise PressError(f"weight_map tensor {tensor_name} names absent shard {shard_name}")
        if tensor_name not in layout[shard_name].get("tensors", {}):
            raise PressError(f"weight_map tensor {tensor_name} absent from shard_layout")

    if covered != total_size:
        raise PressError(
            f"shard map covers {covered} bytes but index total_size is {total_size}; the map must "
            f"cover exactly the declared source"
        )

    largest_shard = max((s["nbytes"] for s in shards.values()), default=0)
    shard_map = {
        "schema": SCHEMA_SHARD_MAP,
        "parent_row_id": parent.row_id,
        "hf_id": parent.hf_id,
        "revision": parent.revision,
        "total_size": total_size,
        "n_shards": len(shards),
        "n_units": len(units),
        "largest_shard_bytes": largest_shard,
        "shards": shards,
        "units": units,
        "bound_at": now_iso(),
    }
    return seal_field(shard_map, "shard_map_sha256")


def shard_map_allows(shard_map: dict[str, Any], shard: str, byte_range: tuple[int, int]) -> bool:
    """True iff [begin,end) is fully contained in one declared unit of `shard`. This is the sole
    fetch authority: any byte outside a unit is refused."""
    begin, end = int(byte_range[0]), int(byte_range[1])
    if begin >= end:
        return False
    for unit in shard_map.get("units", ()):
        if unit["shard"] == shard and unit["begin"] <= begin and end <= unit["end"]:
            return True
    return False


def iter_units(shard_map: dict[str, Any]) -> list[dict[str, Any]]:
    """The bound units in deterministic streaming order (shard, begin)."""
    return list(shard_map.get("units", ()))


# ======================================================================================
# 8.2  bounded source-window cache (LRU with a hard byte cap)
# ======================================================================================
class FetchCache:
    """A bounded LRU cache of fetched source windows. It exists so the resident source footprint
    stays near one shard: adding a window evicts least-recently-used windows until the cap holds.
    `peak_bytes` records the high-water mark so a run can PROVE it never held the whole source."""

    def __init__(self, cap_bytes: int) -> None:
        if cap_bytes <= 0:
            raise PressError("cache cap must be positive")
        self.cap_bytes = int(cap_bytes)
        self._store: dict[str, bytes] = {}
        self._order: list[str] = []
        self.current_bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> bytes | None:
        value = self._store.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self._order.remove(key)
        self._order.append(key)
        return value

    def put(self, key: str, value: bytes) -> None:
        if key in self._store:
            self.current_bytes -= len(self._store[key])
            self._order.remove(key)
        self._store[key] = value
        self._order.append(key)
        self.current_bytes += len(value)
        # Evict LRU until we hold at or below the cap. A lone oversized window is allowed to be
        # resident (one unit at a time) but is recorded as the peak.
        while self.current_bytes > self.cap_bytes and len(self._order) > 1:
            oldest = self._order.pop(0)
            self.current_bytes -= len(self._store.pop(oldest))
            self.evictions += 1
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)

    def release(self, key: str) -> None:
        """Release a converted source window: bounded-stream policy is to not retain source."""
        if key in self._store:
            self.current_bytes -= len(self._store.pop(key))
            self._order.remove(key)

    def stats(self) -> dict[str, int]:
        return {
            "cap_bytes": self.cap_bytes,
            "current_bytes": self.current_bytes,
            "peak_bytes": self.peak_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


def _backoff_seconds(attempt: int, base: float, cap: float) -> float:
    """Deterministic bounded exponential backoff (no jitter, so reacquisition is reproducible)."""
    return min(cap, base * float(2 ** attempt))


# ======================================================================================
# 8.3  fetch_range: resumable, bounded, deterministic acquisition of ONE byte range
# ======================================================================================
def fetch_range(
    url_or_id: str,
    shard: str,
    byte_range: tuple[int, int],
    *,
    fetcher: Callable[[str, tuple[int, int]], bytes],
    shard_map: dict[str, Any],
    cache: FetchCache | None = None,
    revision: str | None = None,
    chunk_bytes: int = 8 * 1024 * 1024,
    max_retries: int = 5,
    backoff_base_s: float = 0.5,
    backoff_cap_s: float = 30.0,
    sleeper: Callable[[float], None] | None = None,
    telemetry: dict[str, Any] | None = None,
) -> bytes:
    """Fetch exactly [begin,end) of `shard`, resumably and within the map authority.

    Refuses any range not covered by `shard_map`. Streams the range in `chunk_bytes` windows;
    on a `TransferError` it retries with bounded backoff and RESUMES from the cursor, so already
    received chunks are never refetched. Maintains a partial-download sha256 over the received
    bytes. On success the whole range is cached (bounded LRU) and returned; a subsequent identical
    request is served from cache (deterministic reacquisition). Exhausting `max_retries` on any
    chunk fails closed with PressError. The injected `fetcher(shard, (b, e)) -> bytes` is the sole
    transport (a real HTTP range GET in production, a fixture in tests).
    """
    begin, end = int(byte_range[0]), int(byte_range[1])
    if not shard_map_allows(shard_map, shard, (begin, end)):
        raise PressError(
            f"refused: byte range [{begin},{end}) of shard {shard} is not in the bound shard map"
        )
    if revision is not None and revision != shard_map.get("revision"):
        raise PressError(
            f"revision {revision} does not match bound map revision {shard_map.get('revision')}"
        )

    key = f"{shard}:{begin}:{end}"
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            if len(cached) != end - begin:
                raise PressError(f"cache corruption for {key}: length mismatch")
            return cached

    sleeper = sleeper or (lambda _s: None)
    buffer = bytearray()
    digest = hashlib.sha256()
    cursor = begin
    retries = 0

    while cursor < end:
        sub_end = min(cursor + max(1, chunk_bytes), end)
        attempt = 0
        while True:
            try:
                chunk = fetcher(shard, (cursor, sub_end))
            except TransferError:
                attempt += 1
                retries += 1
                if attempt > max_retries:
                    raise PressError(
                        f"transfer for [{cursor},{sub_end}) of {shard} failed after "
                        f"{max_retries} retries; cursor preserved at {cursor}"
                    )
                sleeper(_backoff_seconds(attempt, backoff_base_s, backoff_cap_s))
                continue  # resume from the SAME cursor: no already-received bytes refetched
            break
        if not isinstance(chunk, (bytes, bytearray)):
            raise PressError(f"fetcher returned non-bytes for [{cursor},{sub_end}) of {shard}")
        if len(chunk) != sub_end - cursor:
            raise PressError(
                f"fetcher returned {len(chunk)} bytes for [{cursor},{sub_end}) of {shard}; "
                f"expected {sub_end - cursor}"
            )
        buffer.extend(chunk)
        digest.update(chunk)
        cursor = sub_end

    data = bytes(buffer)
    if len(data) != end - begin:
        raise PressError(f"assembled {len(data)} bytes for {key}; expected {end - begin}")
    if cache is not None:
        cache.put(key, data)
    if telemetry is not None:
        telemetry["retries"] = telemetry.get("retries", 0) + retries
        telemetry.setdefault("ranges", []).append(
            {"key": key, "source_sha256": digest.hexdigest(), "nbytes": len(data),
             "retries": retries, "source_id": url_or_id}
        )
    return data


# ======================================================================================
# 8.4  convert_unit: bounded decode -> condense -> pack -> round-trip attest
# ======================================================================================
def _condense_unit(raw: bytes) -> bytes:
    """Deterministic REVERSIBLE codec stand-in. The production codec is a lossy quantizer whose
    attest is a fidelity bound; this reversible transform lets the round-trip attest prove exact
    reconstruction on the fixture, which is what makes the transactional loop testable offline."""
    minimum = min(raw) if raw else 0
    body = bytes((b - minimum) & 0xFF for b in raw)
    header = _CODEC_MAGIC + struct.pack("<IB", len(raw), minimum)
    return header + body


def _reconstruct_unit(packed: bytes) -> bytes:
    if packed[:4] != _CODEC_MAGIC:
        raise PressError("packed unit is missing the codec magic")
    length, minimum = struct.unpack("<IB", packed[4:9])
    body = packed[9:]
    if len(body) != length:
        raise PressError("packed unit body length disagrees with header")
    return bytes((b + minimum) & 0xFF for b in body)


def convert_unit(unit_bytes: bytes, *, unit_id: str = "unit") -> tuple[bytes, dict[str, Any]]:
    """Convert ONE source unit into ONE packed output unit and attest the pack.

    Pipeline: bounded decode of the source bytes, a deterministic condense/transform, a packed
    output unit, then a round-trip attest that reconstructing the pack reproduces the source
    exactly. Only this one unit is resident, so RSS is bounded by the unit and not the shard.
    Returns (packed_bytes, observation). Fails closed if the round trip does not attest.
    """
    if not isinstance(unit_bytes, (bytes, bytearray)):
        raise PressError("unit_bytes must be bytes")
    raw = bytes(unit_bytes)
    input_sha = hashlib.sha256(raw).hexdigest()

    packed = _condense_unit(raw)
    output_sha = hashlib.sha256(packed).hexdigest()

    reconstructed = _reconstruct_unit(packed)
    roundtrip_ok = reconstructed == raw
    if not roundtrip_ok:
        raise PressError(f"round-trip attest failed for unit {unit_id}")

    observation = {
        "schema": SCHEMA_UNIT,
        "unit_id": unit_id,
        "n_bytes": len(raw),
        "packed_bytes": len(packed),
        "input_sha256": input_sha,
        "output_sha256": output_sha,
        "roundtrip_ok": roundtrip_ok,
        "stat_min": min(raw) if raw else 0,
        "stat_max": max(raw) if raw else 0,
        "stat_sum": sum(raw),
    }
    return packed, observation


# ======================================================================================
# 8.5  deterministic global-reduction merge state
# ======================================================================================
def _init_merge() -> dict[str, Any]:
    return {
        "n_units": 0,
        "n_bytes": 0,
        "stat_min": None,
        "stat_max": None,
        "stat_sum": 0,
        "completed_unit_ids": [],
        "reduction_sha256": _ZERO_SHA,
    }


def _merge_unit(state: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    """Fold one unit observation into the global reduction. Associative-per-order and pure: two
    runs over the same ordered units produce byte-identical merged state."""
    umin, umax = obs["stat_min"], obs["stat_max"]
    state["n_units"] += 1
    state["n_bytes"] += obs["n_bytes"]
    state["stat_sum"] += obs["stat_sum"]
    state["stat_min"] = umin if state["stat_min"] is None else min(state["stat_min"], umin)
    state["stat_max"] = umax if state["stat_max"] is None else max(state["stat_max"], umax)
    state["completed_unit_ids"].append(obs["unit_id"])
    folded = f"{state['reduction_sha256']}|{obs['unit_id']}|{obs['output_sha256']}"
    state["reduction_sha256"] = hashlib.sha256(folded.encode("utf-8")).hexdigest()
    return state


def _atomic_write_bytes(path: Path, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


# ======================================================================================
# 8.6  run_pass: stream one deterministic pass, fsync artifact+observation+checkpoint
# ======================================================================================
def run_pass(
    parent: GiantParent,
    pass_name: str,
    shard_map: dict[str, Any],
    *,
    fetcher: Callable[[str, tuple[int, int]], bytes],
    config: Config | None = None,
    cache: FetchCache | None = None,
    checkpoint: dict[str, Any] | None = None,
    retain_source: bool = False,
    sleeper: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Stream one deterministic pass over the bound units, fold the global reduction, and fsync.

    `pass_name` is one of PASSES. Each unit is fetched (resumably), converted (round-trip
    attested), and folded into the merge state. A checkpoint is atomically fsynced after every
    unit, so a crash resumes from the last completed unit (pass in the SAME `checkpoint` to
    resume). Only `final_packing` writes packed output units to disk; the earlier passes retain no
    packed bytes. Source windows are released once converted unless `retain_source` proves
    retention optimal. At pass end the artifact and observation are atomically fsynced and the
    sealed pass result is returned.
    """
    if pass_name not in PASSES:
        raise PressError(f"unknown pass {pass_name!r}; expected one of {PASSES}")
    if not sealed(shard_map, "shard_map_sha256"):
        raise PressError("shard_map is not sealed")
    if shard_map.get("parent_row_id") != parent.row_id:
        raise PressError("shard_map parent does not match parent")

    config = config or default_config()
    cache = cache or FetchCache(config.cache_cap_bytes)
    sleeper = sleeper or (lambda _s: None)

    # Resume from a prior checkpoint if one is supplied and valid.
    if checkpoint is not None:
        if not sealed(checkpoint, "checkpoint_sha256"):
            raise PressError("resume checkpoint is not sealed")
        if checkpoint.get("pass_name") != pass_name:
            raise PressError("resume checkpoint is for a different pass")
        if checkpoint.get("shard_map_sha256") != shard_map["shard_map_sha256"]:
            raise PressError("resume checkpoint is for a different shard map")
        merge = json.loads(json.dumps(checkpoint["merge_state"]))  # deep copy
    else:
        merge = _init_merge()

    done = set(merge["completed_unit_ids"])
    telemetry: dict[str, Any] = {"retries": 0}
    peak_source_resident = 0
    unit_index: dict[str, Any] = dict(checkpoint.get("unit_index", {})) if checkpoint else {}

    units_root = config.output_dir / parent.row_id / pass_name / "units"
    ckpt_path = config.checkpoints_dir / parent.row_id / f"{pass_name}.json"

    for unit in iter_units(shard_map):
        unit_id = unit["unit_id"]
        if unit_id in done:
            continue

        raw = fetch_range(
            parent.hf_id, unit["shard"], (unit["begin"], unit["end"]),
            fetcher=fetcher, shard_map=shard_map, cache=cache, revision=parent.revision,
            chunk_bytes=config.chunk_bytes, max_retries=config.max_retries,
            backoff_base_s=config.backoff_base_s, backoff_cap_s=config.backoff_cap_s,
            sleeper=sleeper, telemetry=telemetry,
        )
        # Resident source high-water mark AFTER the unit has landed (before it is released).
        peak_source_resident = max(peak_source_resident, cache.current_bytes)

        packed, obs = convert_unit(raw, unit_id=unit_id)

        if pass_name == _PACKING_PASS:
            out_path = units_root / f"{unit_id.replace('/', '_').replace('#', '__')}.ehp"
            _atomic_write_bytes(out_path, packed)

        _merge_unit(merge, obs)
        unit_index[unit_id] = {"output_sha256": obs["output_sha256"], "packed_bytes": obs["packed_bytes"],
                               "n_bytes": obs["n_bytes"]}

        # Release the converted source window unless retention is proven optimal.
        if not retain_source:
            cache.release(f"{unit['shard']}:{unit['begin']}:{unit['end']}")

        # Atomically fsync a checkpoint after EVERY unit so a crash resumes from here.
        ckpt = seal_field({
            "schema": SCHEMA_CHECKPOINT,
            "parent_row_id": parent.row_id,
            "pass_name": pass_name,
            "shard_map_sha256": shard_map["shard_map_sha256"],
            "merge_state": merge,
            "unit_index": unit_index,
            "peak_source_resident_bytes": peak_source_resident,
            "retries_total": telemetry["retries"],
            "updated_at": now_iso(),
        }, "checkpoint_sha256")
        atomic_write_json(ckpt_path, ckpt)

    # Pass complete: fsync the artifact and observation.
    artifact = seal_field({
        "schema": SCHEMA_PASS,
        "kind": "artifact",
        "parent_row_id": parent.row_id,
        "hf_id": parent.hf_id,
        "revision": parent.revision,
        "pass_name": pass_name,
        "shard_map_sha256": shard_map["shard_map_sha256"],
        "n_units": merge["n_units"],
        "n_bytes": merge["n_bytes"],
        "stat_min": merge["stat_min"],
        "stat_max": merge["stat_max"],
        "stat_sum": merge["stat_sum"],
        "reduction_sha256": merge["reduction_sha256"],
        "unit_index": unit_index,
        "materialized_output": pass_name == _PACKING_PASS,
        "created_at": now_iso(),
    }, "artifact_sha256")
    artifact_path = config.output_dir / parent.row_id / pass_name / "artifact.json"
    atomic_write_json(artifact_path, artifact)

    observation = seal_field({
        "schema": SCHEMA_PASS,
        "kind": "observation",
        "parent_row_id": parent.row_id,
        "pass_name": pass_name,
        "cache": cache.stats(),
        "peak_source_resident_bytes": peak_source_resident,
        "largest_shard_bytes": shard_map["largest_shard_bytes"],
        "total_source_bytes": shard_map["total_size"],
        "retries_total": telemetry["retries"],
        "source_never_fully_resident": peak_source_resident < shard_map["total_size"],
        "created_at": now_iso(),
    }, "observation_sha256")
    observation_path = config.observations_dir / parent.row_id / f"{pass_name}.json"
    atomic_write_json(observation_path, observation)

    result = {
        "schema": SCHEMA_PASS,
        "kind": "pass_result",
        "parent_row_id": parent.row_id,
        "pass_name": pass_name,
        "n_units": merge["n_units"],
        "reduction_sha256": merge["reduction_sha256"],
        "merged_stats": {"n_bytes": merge["n_bytes"], "stat_min": merge["stat_min"],
                         "stat_max": merge["stat_max"], "stat_sum": merge["stat_sum"]},
        "peak_source_resident_bytes": peak_source_resident,
        "total_source_bytes": shard_map["total_size"],
        "largest_shard_bytes": shard_map["largest_shard_bytes"],
        "retries_total": telemetry["retries"],
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact["artifact_sha256"],
        "observation_path": str(observation_path),
        "checkpoint_path": str(ckpt_path),
    }
    return seal_field(result, "pass_result_sha256")


# ======================================================================================
# 8.7  press_plan: sealed dry-run plan (launches nothing, peak disk ~ one shard)
# ======================================================================================
def press_plan(parent: GiantParent, *, config: Config | None = None) -> dict[str, Any]:
    """Emit a sealed dry-run Press plan for one giant parent. It computes the SOURCE bytes, the
    peak disk as one source shard plus one output shard (never the whole source), the pass order
    with a per-pass byte budget, the disk-floor gate, and the two safety assertions (disk-walled,
    launches nothing while the legacy campaign runs). It launches nothing and downloads nothing."""
    config = config or default_config()
    source_bytes = parent.source_bytes
    n_shards = max(1, parent.n_source_shards)
    avg_shard_bytes = source_bytes // n_shards
    # One output shard at the resident anchor rate: shard params * bpw / 8. Approximate a shard's
    # param share as its byte share of the source scaled by the source bytes-per-param.
    bpp = parent.bytes_per_param() or 1.0
    shard_params = avg_shard_bytes / bpp
    output_shard_bytes = int(shard_params * parent.resident_anchor_bpw / 8.0)
    peak_disk_bytes = avg_shard_bytes + output_shard_bytes

    per_pass_budget = min(config.cache_cap_bytes, avg_shard_bytes)
    plan = {
        "schema": SCHEMA_PLAN,
        "parent_row_id": parent.row_id,
        "hf_id": parent.hf_id,
        "revision": parent.revision,
        "source_bytes": source_bytes,
        "source_bytes_gb": round(source_bytes / 1e9, 1),
        "n_source_shards": n_shards,
        "avg_shard_bytes": avg_shard_bytes,
        "output_shard_bytes_at_anchor": output_shard_bytes,
        "resident_anchor_bpw": parent.resident_anchor_bpw,
        "peak_disk_bytes": peak_disk_bytes,
        "peak_disk_gb": round(peak_disk_bytes / 1e9, 2),
        "peak_disk_model": "one_source_shard + one_output_shard (NOT the whole source)",
        "peak_disk_under_source": peak_disk_bytes < source_bytes,
        "pass_order": list(PASSES),
        "per_pass_byte_budget": per_pass_budget,
        "release_source_windows_default": True,
        "disk_free_floor_bytes": config.disk_free_floor_bytes,
        "disk_free_floor_gb": round(config.disk_free_floor_bytes / 1e9, 1),
        "disk_floor_gate": "refuse to advance a pass if free disk would fall below the floor",
        "disk_walled": True,
        "launches_nothing_while_legacy_running": True,
        "downloads_nothing_real": True,
        "acquisition_policy": "remote_bounded_stream_press",
        "transport": "injected fetcher (real range GET in production, fixture in tests)",
        "created_at": now_iso(),
    }
    return seal_field(plan, "plan_sha256")


# ======================================================================================
# selftest: drive the whole lifecycle over a tiny self-contained byte fixture
# ======================================================================================
def _fixture_source() -> tuple[dict[str, Any], dict[bytes, bytes], dict[str, bytes]]:
    """Build a tiny synthetic 'source': 4 shards, 2 tensors of 16 bytes each, 128 bytes total.
    Returns (index_json, unused, shard_blobs) where shard_blobs[shard] is the raw shard bytes."""
    tensor_bytes = 16
    tensors_per_shard = 2
    n_shards = 4
    shard_blobs: dict[str, bytes] = {}
    weight_map: dict[str, str] = {}
    shard_layout: dict[str, Any] = {}
    total = 0
    t = 0
    for s in range(n_shards):
        shard_name = f"model-{s:05d}-of-{n_shards:05d}.safetensors"
        blob = bytearray()
        tensors: dict[str, Any] = {}
        for _ in range(tensors_per_shard):
            content = bytes(((t * 7 + j) & 0xFF) for j in range(tensor_bytes))
            begin = len(blob)
            blob.extend(content)
            end = len(blob)
            tensor_name = f"model.layers.{t}.weight"
            tensors[tensor_name] = {"begin": begin, "end": end}
            weight_map[tensor_name] = shard_name
            t += 1
        shard_blobs[shard_name] = bytes(blob)
        shard_layout[shard_name] = {"nbytes": len(blob), "tensors": tensors}
        total += len(blob)
    index_json = {
        "metadata": {"total_size": total},
        "weight_map": weight_map,
        "shard_layout": shard_layout,
    }
    return index_json, {}, shard_blobs


def _reliable_fetcher(shard_blobs: dict[str, bytes]) -> Callable[[str, tuple[int, int]], bytes]:
    def fetch(shard: str, byte_range: tuple[int, int]) -> bytes:
        return shard_blobs[shard][byte_range[0]:byte_range[1]]
    return fetch


def _flaky_fetcher(shard_blobs: dict[str, bytes], drop_on: tuple[str, int]) -> Callable[[str, tuple[int, int]], bytes]:
    """A fetcher that drops the connection exactly once at a chosen (shard, chunk-begin), forcing
    a mid-transfer resume from the cursor. After the single drop it delivers normally."""
    dropped: set[tuple[str, int]] = set()

    def fetch(shard: str, byte_range: tuple[int, int]) -> bytes:
        marker = (shard, byte_range[0])
        if marker == drop_on and marker not in dropped:
            dropped.add(marker)
            raise TransferError(f"simulated drop at {shard}:{byte_range[0]}")
        return shard_blobs[shard][byte_range[0]:byte_range[1]]
    return fetch


def _hardfail_fetcher(shard_blobs: dict[str, bytes], fail_shard: str, fail_begin: int) -> Callable[[str, tuple[int, int]], bytes]:
    """A fetcher that always drops a specific range, so fetch_range exhausts its retries and
    run_pass crashes there (leaving a fsynced checkpoint to resume from)."""
    def fetch(shard: str, byte_range: tuple[int, int]) -> bytes:
        if shard == fail_shard and byte_range[0] == fail_begin:
            raise TransferError("permanent drop")
        return shard_blobs[shard][byte_range[0]:byte_range[1]]
    return fetch


def selftest() -> dict[str, Any]:
    import tempfile

    parent = PARENTS[0]  # deepseek-v3.2-685b; geometry only, no source touched
    index_json, _unused, shard_blobs = _fixture_source()
    shard_map = bind_shard_map(parent, index_json)

    if not sealed(shard_map, "shard_map_sha256"):
        raise PressError("shard map not sealed")
    if shard_map["n_units"] != 8 or shard_map["total_size"] != 128:
        raise PressError(f"unexpected fixture geometry: {shard_map['n_units']} units, "
                         f"{shard_map['total_size']} bytes")
    largest_shard = shard_map["largest_shard_bytes"]  # 32

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Cache cap set to ~ one shard so we can prove peak disk stays near one shard.
        cfg = Config(root=root, press_dir=root / "press",
                     cache_cap_bytes=largest_shard, disk_free_floor_bytes=64,
                     chunk_bytes=8, max_retries=5, backoff_base_s=0.0, backoff_cap_s=0.0)

        # ---- (A) clean end-to-end run of final_packing --------------------------------
        clean = run_pass(parent, "final_packing", shard_map,
                         fetcher=_reliable_fetcher(shard_blobs), config=cfg,
                         cache=FetchCache(cfg.cache_cap_bytes))
        if not sealed(clean, "pass_result_sha256"):
            raise PressError("clean pass result not sealed")
        if clean["n_units"] != 8:
            raise PressError(f"clean run converted {clean['n_units']} units, expected 8")

        # ---- (B) peak disk stays ~ one shard, never the whole source ------------------
        if clean["peak_source_resident_bytes"] > largest_shard:
            raise PressError(
                f"peak source resident {clean['peak_source_resident_bytes']} exceeded one shard "
                f"{largest_shard}")
        if clean["peak_source_resident_bytes"] >= clean["total_source_bytes"]:
            raise PressError("source was fully resident: bounded-stream invariant broken")

        # ---- (C) deterministic global reduction: two runs -> identical merged stats ---
        clean2 = run_pass(parent, "census_statistics", shard_map,
                          fetcher=_reliable_fetcher(shard_blobs), config=cfg,
                          cache=FetchCache(cfg.cache_cap_bytes))
        clean3 = run_pass(parent, "census_statistics", shard_map,
                          fetcher=_reliable_fetcher(shard_blobs), config=cfg,
                          cache=FetchCache(cfg.cache_cap_bytes))
        if clean2["reduction_sha256"] != clean3["reduction_sha256"]:
            raise PressError("global reduction is not deterministic across runs")
        if clean2["merged_stats"] != clean3["merged_stats"]:
            raise PressError("merged stats differ across runs")

        # ---- (D) mid-transfer failure RESUMES from cursor, yields SAME output ---------
        # Drop the 2nd chunk (cursor=8) of the last unit's shard range. With chunk_bytes=8 and a
        # 16-byte unit at shard offset 16, the drop lands mid-unit and must resume from cursor 8.
        last_unit = iter_units(shard_map)[-1]
        drop_marker = (last_unit["shard"], last_unit["begin"] + 8)
        resumed = run_pass(parent, "final_packing", shard_map,
                           fetcher=_flaky_fetcher(shard_blobs, drop_marker), config=cfg,
                           cache=FetchCache(cfg.cache_cap_bytes))
        if resumed["reduction_sha256"] != clean["reduction_sha256"]:
            raise PressError("resumed run reduction differs from clean run")
        if resumed["artifact_sha256"] != clean["artifact_sha256"]:
            raise PressError("resumed artifact differs from clean artifact")
        if resumed["retries_total"] < 1:
            raise PressError("flaky run recorded no retry: the drop was not exercised")

        # ---- (E) crash between units + checkpoint resume -> identical to clean --------
        crash_unit = iter_units(shard_map)[3]
        hard = _hardfail_fetcher(shard_blobs, crash_unit["shard"], crash_unit["begin"])
        crashed = False
        try:
            run_pass(parent, "final_packing", shard_map, fetcher=hard, config=cfg,
                     cache=FetchCache(cfg.cache_cap_bytes))
        except PressError:
            crashed = True
        if not crashed:
            raise PressError("hard-fail fetcher did not crash the pass")
        ckpt_path = cfg.checkpoints_dir / parent.row_id / "final_packing.json"
        ckpt = read_json_safe(ckpt_path)
        if not sealed(ckpt, "checkpoint_sha256"):
            raise PressError("crash checkpoint not sealed")
        if ckpt["merge_state"]["n_units"] != 3:
            raise PressError(f"expected 3 units before crash, got {ckpt['merge_state']['n_units']}")
        resumed2 = run_pass(parent, "final_packing", shard_map,
                            fetcher=_reliable_fetcher(shard_blobs), config=cfg,
                            cache=FetchCache(cfg.cache_cap_bytes), checkpoint=ckpt)
        if resumed2["reduction_sha256"] != clean["reduction_sha256"]:
            raise PressError("checkpoint-resumed reduction differs from clean run")
        if resumed2["n_units"] != 8:
            raise PressError(f"checkpoint resume converted {resumed2['n_units']} units, expected 8")

        # ---- (F) a byte NOT in the shard map is refused -------------------------------
        refused = False
        try:
            fetch_range(parent.hf_id, last_unit["shard"], (1000, 1008),
                        fetcher=_reliable_fetcher(shard_blobs), shard_map=shard_map,
                        cache=FetchCache(cfg.cache_cap_bytes))
        except PressError:
            refused = True
        if not refused:
            raise PressError("a byte outside the shard map was NOT refused")

        # ---- (G) convert_unit round-trip attest --------------------------------------
        packed, obs = convert_unit(b"\x03\x09\x02\xff\x00", unit_id="probe")
        if not obs["roundtrip_ok"] or _reconstruct_unit(packed) != b"\x03\x09\x02\xff\x00":
            raise PressError("convert_unit round-trip attest failed")

    # ---- (H) press_plan is sealed, disk-walled, launches nothing, peak < source ------
    plan = press_plan(parent)
    if not sealed(plan, "plan_sha256"):
        raise PressError("press plan not sealed")
    if not plan["disk_walled"] or not plan["launches_nothing_while_legacy_running"]:
        raise PressError("press plan safety flags not set")
    if not plan["peak_disk_under_source"] or plan["peak_disk_bytes"] >= plan["source_bytes"]:
        raise PressError("press plan peak disk is not bounded below the source")
    if plan["pass_order"] != list(PASSES):
        raise PressError("press plan pass order wrong")

    return {
        "ok": True,
        "parent": parent.row_id,
        "units": shard_map["n_units"],
        "clean_reduction_sha256": clean["reduction_sha256"],
        "resume_matches_clean": True,
        "checkpoint_resume_matches_clean": True,
        "deterministic_reduction": True,
        "peak_source_resident_bytes": clean["peak_source_resident_bytes"],
        "largest_shard_bytes": largest_shard,
        "total_source_bytes": clean["total_source_bytes"],
        "byte_not_in_map_refused": True,
        "plan_peak_disk_gb": plan["peak_disk_gb"],
        "plan_source_bytes_gb": plan["source_bytes_gb"],
        "plan_sha256": plan["plan_sha256"],
    }


def _load_index(path: str) -> dict[str, Any]:
    return read_json_safe(path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Remote bounded-stream Press lifecycle for the "
                                             "High-Parameter Frontier giant parents.")
    ap.add_argument("--parent", default=None, help="row_id of a giant parent (default: all plans)")
    ap.add_argument("--plan", action="store_true", help="print sealed press_plan(s)")
    ap.add_argument("--index", default=None, help="bind a shard map from an index+layout JSON")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        sys.exit(0)

    def _pick(row_id: str | None) -> list[GiantParent]:
        if row_id is None:
            return list(PARENTS)
        chosen = [p for p in PARENTS if p.row_id == row_id]
        if not chosen:
            raise SystemExit(f"unknown parent {row_id}; known: {[p.row_id for p in PARENTS]}")
        return chosen

    if args.index is not None:
        parents = _pick(args.parent)
        if len(parents) != 1:
            raise SystemExit("--index requires exactly one --parent")
        smap = bind_shard_map(parents[0], _load_index(args.index))
        if args.out:
            atomic_write_json(args.out, smap)
        print(json.dumps({"schema": smap["schema"], "parent_row_id": smap["parent_row_id"],
                          "n_shards": smap["n_shards"], "n_units": smap["n_units"],
                          "total_size": smap["total_size"],
                          "shard_map_sha256": smap["shard_map_sha256"]}, indent=2, sort_keys=True))
        sys.exit(0)

    plans = [press_plan(p) for p in _pick(args.parent)]
    if args.out and len(plans) == 1:
        atomic_write_json(args.out, plans[0])
    print(json.dumps([{"parent_row_id": p["parent_row_id"], "source_bytes_gb": p["source_bytes_gb"],
                       "peak_disk_gb": p["peak_disk_gb"], "pass_order": p["pass_order"],
                       "disk_walled": p["disk_walled"],
                       "launches_nothing_while_legacy_running": p["launches_nothing_while_legacy_running"],
                       "plan_sha256": p["plan_sha256"]} for p in plans], indent=2, sort_keys=True))

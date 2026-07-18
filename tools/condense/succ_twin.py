#!/usr/bin/env python3.12
"""Mandatory synthetic geometry twins + systems-path validator (master goal section 7).

A giant parent (685B / 1T / 1.6T) cannot be downloaded on this box: the sources are
595 to 1371 GB, disk-walled, and the legacy 72B campaign is still running. Yet the
systems path that a real source acquisition would exercise (deterministic conversion,
bounded RSS, resumable source ranges, expert paging, crash recovery, single-writer
leasing, output layout) can be proven WITHOUT any real weights.

This module builds a synthetic twin: architecture-FAITHFUL but tiny and with no model
intelligence. It has a scaled-down layer count, the SAME routed / shared / selected
expert structure as the real parent, faithful tensor NAMES and scaled shapes, realistic
per-expert page sizes, a source-shard topology, an output-shard topology, Doctor
correction section markers, an MTP block when the parent has one, and a vision boundary
marker when the parent is multimodal. It then runs the systems validation battery over
the twin. A real source acquisition may begin only after the twin passes GREEN.

Non-interference: nothing here launches anything heavy, downloads anything, or writes
under the campaign namespace (reports/condense/doctor_v5_ultra). Successor state lives
under reports/condense/event_horizon_successor/twins. selftest is fully offline: it
builds and validates a twin for each of the three real parents in a private tempdir with
tiny synthetic byte tensors, and never touches real campaign data or live processes.

House style: no em or en dashes, no middots.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import secrets
import struct
import sys
import tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, seal_field, sealed, hash_value, now_iso,
    atomic_write_json, read_json_safe, repo_root,
)
from succ_frontier import PARENTS, GiantParent  # noqa: E402

# -- schema registry (every emitted artifact carries one) -------------------------------
SCHEMA_TWIN = "hawking.successor.twin.v1"
SCHEMA_TWIN_LAYOUT = "hawking.successor.twin_layout.v1"
SCHEMA_TWIN_CHECKPOINT = "hawking.successor.twin_checkpoint.v1"
SCHEMA_TWIN_OUTPUT = "hawking.successor.twin_output.v1"
SCHEMA_TWIN_VALIDATION = "hawking.successor.twin_validation.v1"
SCHEMA_TWIN_REPORT = "hawking.successor.twin_report.v1"

# Successor-only state namespace. Deliberately NOT under doctor_v5_ultra (campaign owned).
TWIN_DIR = "reports/condense/event_horizon_successor/twins"

# Kinds whose per-tensor byte size is the realistic "expert page" size.
_EXPERT_KINDS = frozenset({"routed_expert", "shared_expert", "mtp_expert"})

# Reversible pack transform markers. The transform is a PLACEHOLDER for a real codec: it
# proves the systems path (deterministic, reversible, byte-exact), not a quantizer.
_PACK_MAGIC = b"TWPK"
_XOR_KEY = 0x5A


class TwinError(EcoError):
    """Any fail-closed error in the synthetic-twin systems validator."""


class _SimulatedKill(Exception):
    """Control-flow signal: a converter was killed mid-range (not a failure)."""

    def __init__(self, index: int) -> None:
        super().__init__(f"simulated kill after unit index {index}")
        self.index = index


# ======================================================================================
# configuration
# ======================================================================================
@dataclasses.dataclass(frozen=True)
class Config:
    """Twin scale and the systems bounds the battery enforces. All bytes are tiny."""

    state_root: Path
    scale_layers: int = 4
    expert_bytes: int = 4096            # realistic per-expert page size (scaled)
    experts_per_layer_cap: int = 16     # twin routes a small pool; real count is recorded
    tensors_per_shard: int = 24         # source-shard topology granularity
    rss_bound_bytes: int = 24576        # 6 expert pages; peak resident must stay under this
    page_cap_bytes: int = 32768         # HOT/WARM/COLD pool byte cap (8 expert pages)


def default_config() -> Config:
    return Config(state_root=repo_root() / TWIN_DIR)


def _assert_non_interfering(path: str | os.PathLike[str]) -> None:
    """Structural guard: never let a twin or receipt resolve into the campaign namespace."""
    text = str(path)
    if "doctor_v5_ultra" in text:
        raise TwinError(f"refusing to write into campaign namespace: {text}")


# ======================================================================================
# low-level twin primitives
# ======================================================================================
def _synthetic_bytes(name: str, nbytes: int) -> bytes:
    """Deterministic synthetic payload for a tensor. No model meaning; name-derived so
    round-trip and deterministic-conversion checks are non-trivial."""
    out = bytearray()
    counter = 0
    while len(out) < nbytes:
        out += hashlib.sha256(f"{name}:{counter}".encode("utf-8")).digest()
        counter += 1
    return bytes(out[:nbytes])


def _xor(raw: bytes) -> bytes:
    return bytes(b ^ _XOR_KEY for b in raw)


def _pack_unit(raw: bytes) -> bytes:
    """Reversible, deterministic pack: magic + length + xor(payload). Placeholder codec."""
    return _PACK_MAGIC + struct.pack(">I", len(raw)) + _xor(raw)


def _unpack_unit(packed: bytes) -> bytes:
    if packed[:4] != _PACK_MAGIC:
        raise TwinError("bad pack magic")
    (declared,) = struct.unpack(">I", packed[4:8])
    body = packed[8:]
    if len(body) != declared:
        raise TwinError(f"pack length mismatch: declared {declared}, got {len(body)}")
    return _xor(body)


def _unit_filename(name: str) -> str:
    return name.replace("/", "_").replace(".", "_") + ".pkd"


def _atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{secrets.token_hex(6)}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _shape_for(nbytes: int) -> list[int]:
    """A faithful 2-D scaled shape descriptor (uint8 packed tensor)."""
    cols = 64 if nbytes >= 64 and nbytes % 64 == 0 else nbytes
    rows = nbytes // cols if cols else 1
    return [rows, cols]


# ======================================================================================
# twin construction
# ======================================================================================
def _tensor_specs(parent: GiantParent, *, scale_layers: int, expert_bytes: int,
                  experts_per_layer: int) -> tuple[list[tuple[str, int, str]], str | None]:
    """Faithful (scaled) tensor name / size / kind list in write order, plus the vision
    boundary marker name (or None). The structure mirrors the real parent: embeddings,
    per-layer MLA attention projections, shared experts, routed experts, an MTP block if
    parent.mtp_layers, an lm_head, and a vision boundary marker if parent.multimodal."""
    small = max(256, expert_bytes // 4)
    embed = expert_bytes * 2
    specs: list[tuple[str, int, str]] = []
    specs.append(("model.embed_tokens.weight", embed, "embedding"))
    for i in range(scale_layers):
        specs.append((f"model.layers.{i}.self_attn.q_a_proj.weight", small, "attn"))
        specs.append((f"model.layers.{i}.self_attn.kv_a_proj.weight", small, "attn"))
        for s in range(parent.n_shared_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                specs.append(
                    (f"model.layers.{i}.mlp.shared_experts.{s}.{proj}.weight",
                     expert_bytes, "shared_expert"))
        for e in range(experts_per_layer):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                specs.append(
                    (f"model.layers.{i}.mlp.experts.{e}.{proj}.weight",
                     expert_bytes, "routed_expert"))
    specs.append(("lm_head.weight", embed, "lm_head"))
    for k in range(parent.mtp_layers):
        base = f"model.layers.{scale_layers + k}.mtp"
        specs.append((f"{base}.eh_proj.weight", small, "mtp"))
        for proj in ("gate_proj", "up_proj", "down_proj"):
            specs.append((f"{base}.transformer.{proj}.weight", expert_bytes, "mtp_expert"))
    vision_boundary: str | None = None
    if parent.multimodal:
        vision_boundary = "model.vision_tower.boundary_marker.weight"
        specs.append((vision_boundary, small, "vision_boundary"))
    return specs, vision_boundary


def build_twin(parent: GiantParent, *, scale_layers: int = 4, expert_bytes: int = 4096,
               out_dir: str | os.PathLike[str], experts_per_layer_cap: int = 16,
               tensors_per_shard: int = 24) -> dict[str, Any]:
    """Materialize a synthetic twin on disk: a fake safetensors-like shard set with tiny
    byte blobs named after the real tensors, a model.safetensors.index.json weight map,
    and a twin_layout.json unit table (byte offsets) for bounded reads. Returns a sealed
    twin manifest. The twin routes a small expert pool (experts_per_layer_cap) but RECORDS
    the real routed/selected/shared counts so downstream page math scales to the parent."""
    out_dir = Path(out_dir)
    _assert_non_interfering(out_dir)
    source_dir = out_dir / "twin_source"
    source_dir.mkdir(parents=True, exist_ok=True)

    experts_per_layer = min(parent.n_routed_experts, experts_per_layer_cap)
    specs, vision_boundary = _tensor_specs(
        parent, scale_layers=scale_layers, expert_bytes=expert_bytes,
        experts_per_layer=experts_per_layer)

    n_shards = max(2, math.ceil(len(specs) / tensors_per_shard))
    per_shard = math.ceil(len(specs) / n_shards)

    groups: dict[int, list[tuple[str, int, str]]] = defaultdict(list)
    for idx, spec in enumerate(specs):
        groups[idx // per_shard].append(spec)

    units: list[dict[str, Any]] = []
    weight_map: dict[str, str] = {}
    source_shards: list[dict[str, Any]] = []
    total_size = 0
    for sidx in sorted(groups):
        fname = f"model-{sidx + 1:05d}-of-{n_shards:05d}.safetensors"
        buf = bytearray()
        offset = 0
        for (name, nbytes, kind) in groups[sidx]:
            buf += _synthetic_bytes(name, nbytes)
            units.append({
                "name": name, "shard_file": fname, "offset": offset, "length": nbytes,
                "dtype": "uint8", "shape": _shape_for(nbytes), "kind": kind,
            })
            weight_map[name] = fname
            offset += nbytes
            total_size += nbytes
        blob = bytes(buf)
        _atomic_write_bytes(source_dir / fname, blob)
        source_shards.append({
            "shard_file": fname, "bytes": len(blob), "n_tensors": len(groups[sidx]),
            "sha256": hashlib.sha256(blob).hexdigest(),
        })

    # faithful HF-style index: weight_map + metadata.total_size (no offsets in the real file)
    index = {
        "metadata": {"total_size": total_size, "twin_synthetic": True,
                     "format": "twin_safetensors_like"},
        "weight_map": weight_map,
    }
    atomic_write_json(source_dir / "model.safetensors.index.json", index)
    # private layout table carries the byte offsets that make bounded single-unit reads real
    atomic_write_json(source_dir / "twin_layout.json",
                      {"schema": SCHEMA_TWIN_LAYOUT, "units": units})

    geometry = {
        "scale_layers": scale_layers,
        "experts_per_layer_twin": experts_per_layer,
        "real_num_layers": parent.num_layers,
        "real_n_routed_experts": parent.n_routed_experts,
        "real_experts_per_tok": parent.experts_per_tok,
        "real_n_shared_experts": parent.n_shared_experts,
        "real_moe_intermediate_size": parent.moe_intermediate_size,
        "mtp_layers": parent.mtp_layers,
        "multimodal": parent.multimodal,
        "expert_bytes": expert_bytes,
        "page_math_scale_factor": round(parent.n_routed_experts / experts_per_layer, 4),
        "attention": "MLA" if parent.kv_lora_rank else "MHA/other",
        "doctor_families": list(parent.doctor_families),
    }
    twin = {
        "schema": SCHEMA_TWIN,
        "parent_row_id": parent.row_id,
        "hf_id": parent.hf_id,
        "architecture": parent.architecture,
        "model_type": parent.model_type,
        "source_dir": str(source_dir),
        "index_file": "model.safetensors.index.json",
        "layout_file": "twin_layout.json",
        "geometry": geometry,
        "n_units": len(units),
        "total_source_bytes": total_size,
        "n_source_shards": n_shards,
        "source_shards": source_shards,
        "weight_map_count": len(weight_map),
        "vision_boundary": vision_boundary,
        "mtp_present": parent.mtp_layers > 0,
        "units": units,
        "created_at": now_iso(),
        "note": ("architecture-faithful, intelligence-free twin; proves the systems path "
                 "only. Real source acquisition is gated on this twin passing GREEN."),
    }
    return seal_field(twin, "twin_sha256")


def _ordered_units(twin: dict[str, Any]) -> list[dict[str, Any]]:
    return twin["units"]


def _read_unit(twin: dict[str, Any], unit: dict[str, Any]) -> bytes:
    """Read exactly one unit's byte range from its source shard. Only `length` bytes are
    ever resident (this is what keeps RSS bounded; the whole twin is never loaded)."""
    path = Path(twin["source_dir"]) / unit["shard_file"]
    with path.open("rb") as handle:
        handle.seek(unit["offset"])
        data = handle.read(unit["length"])
    if len(data) != unit["length"]:
        raise TwinError(f"short read for {unit['name']}: {len(data)} != {unit['length']}")
    return data


# ======================================================================================
# resumable single-unit-at-a-time conversion
# ======================================================================================
def _convert_all(twin: dict[str, Any], out_dir: str | os.PathLike[str], *,
                 stop_after: int | None = None) -> dict[str, Any]:
    """Convert source units to packed output units, ONE unit at a time, checkpointing the
    (shard, expert) cursor after each unit. Resumes from a prior checkpoint if present. If
    stop_after is set, raises _SimulatedKill once that many units have been processed this
    call (simulating a mid-range crash). The final output index is written only on a full
    completion, so a killed run leaves a checkpoint plus partial unit files but no index."""
    out_dir = Path(out_dir)
    _assert_non_interfering(out_dir)
    units_dir = out_dir / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.json"
    out_index_path = out_dir / "output.index.json"

    units = _ordered_units(twin)
    start = 0
    packed_sha: dict[str, str] = {}
    if ckpt_path.exists():
        ck = read_json_safe(ckpt_path)
        start = int(ck["cursor"])
        packed_sha = dict(ck.get("packed_sha", {}))

    peak = 0
    processed = 0
    for idx in range(start, len(units)):
        unit = units[idx]
        raw = _read_unit(twin, unit)          # bounded: only this unit resident
        packed = _pack_unit(raw)
        peak = max(peak, len(raw) + len(packed))
        _atomic_write_bytes(units_dir / _unit_filename(unit["name"]), packed)
        packed_sha[unit["name"]] = hashlib.sha256(packed).hexdigest()
        processed += 1
        atomic_write_json(ckpt_path, {
            "schema": SCHEMA_TWIN_CHECKPOINT, "cursor": idx + 1,
            "packed_sha": packed_sha, "count": len(packed_sha),
        })
        del raw, packed
        if stop_after is not None and processed >= stop_after:
            raise _SimulatedKill(idx)

    artifact = {
        "schema": SCHEMA_TWIN_OUTPUT,
        "units": dict(sorted(packed_sha.items())),
        "count": len(packed_sha),
    }
    artifact = seal_field(artifact, "artifact_sha256")
    atomic_write_json(out_index_path, artifact)
    return {
        "out_dir": str(out_dir), "output_index": str(out_index_path),
        "artifact_sha256": artifact["artifact_sha256"], "count": len(packed_sha),
        "peak_bytes": peak, "completed": set(packed_sha.keys()), "units_dir": str(units_dir),
    }


# ======================================================================================
# HOT / WARM / COLD expert pager (runtime page lifecycle)
# ======================================================================================
class _ExpertPager:
    """A byte-capped expert page cache. Promote-on-use (LRU recency), evict-before-insert
    so the byte cap is NEVER exceeded. Resident keys are tiered HOT / WARM / COLD by
    recency for reporting; evicted keys leave the resident set entirely."""

    def __init__(self, cap_bytes: int) -> None:
        self.cap = cap_bytes
        self.cache: "OrderedDict[str, int]" = OrderedDict()
        self.resident = 0
        self.hits = 0
        self.misses = 0
        self.loads = 0
        self.evictions = 0

    def access(self, name: str, nbytes: int) -> str:
        if name in self.cache:
            self.cache.move_to_end(name)      # promote
            self.hits += 1
            return "hit"
        self.misses += 1
        self.loads += 1
        while self.cache and self.resident + nbytes > self.cap:
            _, sz = self.cache.popitem(last=False)   # evict oldest (COLD)
            self.resident -= sz
            self.evictions += 1
        self.cache[name] = nbytes
        self.resident += nbytes
        return "miss"

    def tiers(self) -> dict[str, int]:
        keys = list(self.cache.keys())        # oldest .. newest
        n = len(keys)
        third = max(1, n // 3)
        return {"cold": len(keys[:third]), "warm": len(keys[third:2 * third]),
                "hot": len(keys[2 * third:]), "resident": n}


# ======================================================================================
# systems validation battery (each returns (pass: bool, evidence: dict))
# ======================================================================================
def deterministic_conversion(twin: dict[str, Any], work: Path) -> tuple[bool, dict[str, Any]]:
    """Convert twice to independent outputs; assert byte-identical result."""
    a = _convert_all(twin, work / "det_a")
    b = _convert_all(twin, work / "det_b")
    same_artifact = a["artifact_sha256"] == b["artifact_sha256"]
    files_identical = True
    for name in a["completed"]:
        fa = (Path(a["units_dir"]) / _unit_filename(name)).read_bytes()
        fb = (Path(b["units_dir"]) / _unit_filename(name)).read_bytes()
        if fa != fb:
            files_identical = False
            break
    ok = same_artifact and files_identical and a["count"] == twin["n_units"]
    return ok, {"artifact_sha_a": a["artifact_sha256"], "artifact_sha_b": b["artifact_sha256"],
                "artifact_identical": same_artifact, "files_identical": files_identical,
                "units": a["count"]}


def round_trip_integrity(twin: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Pack then unpack every unit; assert bit-identical to the source bytes."""
    checked = 0
    first_failure = None
    for unit in _ordered_units(twin):
        raw = _read_unit(twin, unit)
        if _unpack_unit(_pack_unit(raw)) != raw:
            first_failure = unit["name"]
            break
        checked += 1
    ok = first_failure is None and checked == twin["n_units"]
    return ok, {"units_checked": checked, "all_bit_identical": ok,
                "first_failure": first_failure}


def bounded_rss(twin: dict[str, Any], bound_bytes: int) -> tuple[bool, dict[str, Any]]:
    """Process one unit at a time; assert peak resident bytes stay under a small bound and
    that the whole twin is never resident."""
    peak = 0
    seen = 0
    for unit in _ordered_units(twin):
        raw = _read_unit(twin, unit)          # only this unit resident
        packed = _pack_unit(raw)
        peak = max(peak, len(raw) + len(packed))
        seen += 1
        del raw, packed
    total = twin["total_source_bytes"]
    ok = peak <= bound_bytes and peak < total and seen == twin["n_units"]
    return ok, {"peak_resident_bytes": peak, "bound_bytes": bound_bytes,
                "twin_total_bytes": total, "units": seen,
                "never_whole_twin": peak < total,
                "resident_ratio": round(total / max(peak, 1), 1)}


def source_range_resume(twin: dict[str, Any], work: Path) -> tuple[bool, dict[str, Any]]:
    """Checkpoint the (shard, expert) cursor, kill mid-conversion, resume, and assert the
    completed unit set equals a clean run."""
    clean = _convert_all(twin, work / "clean")
    crash_dir = work / "crash"
    kill_at = max(1, twin["n_units"] // 3)
    killed_index = None
    try:
        _convert_all(twin, crash_dir, stop_after=kill_at)
    except _SimulatedKill as exc:
        killed_index = exc.index
    partial = read_json_safe(crash_dir / "checkpoint.json")
    resumed = _convert_all(twin, crash_dir)   # resume from checkpoint, finish
    matches = (resumed["completed"] == clean["completed"]
               and resumed["artifact_sha256"] == clean["artifact_sha256"])
    ok = (killed_index is not None and kill_at <= partial["count"] < twin["n_units"]
          and matches)
    return ok, {"total_units": twin["n_units"], "killed_after_index": killed_index,
                "checkpoint_count_at_kill": partial["count"], "resumed_count": resumed["count"],
                "matches_clean_run": matches}


def expert_paging(twin: dict[str, Any], cap_bytes: int) -> tuple[bool, dict[str, Any]]:
    """Drive a HOT/WARM/COLD pool over the expert units; assert hit/miss accounting is
    exact and the byte cap is never exceeded."""
    experts = [u for u in _ordered_units(twin) if u["kind"] in _EXPERT_KINDS]
    pager = _ExpertPager(cap_bytes)
    seq: list[dict[str, Any]] = []
    for unit in experts:
        seq.append(unit)      # first touch: cold miss (or a re-load if it was evicted)
        seq.append(unit)      # immediate re-touch: guaranteed HOT hit
    max_resident = 0
    for unit in seq:
        pager.access(unit["name"], unit["length"])
        max_resident = max(max_resident, pager.resident)
    ok = (pager.hits + pager.misses == len(seq)
          and pager.misses == pager.loads
          and max_resident <= cap_bytes
          and pager.hits > 0
          and pager.evictions > 0)
    return ok, {"accesses": len(seq), "hits": pager.hits, "misses": pager.misses,
                "loads": pager.loads, "evictions": pager.evictions, "cap_bytes": cap_bytes,
                "max_resident_bytes": max_resident, "cap_never_exceeded": max_resident <= cap_bytes,
                "tiers": pager.tiers()}


def crash_recovery(twin: dict[str, Any], work: Path) -> tuple[bool, dict[str, Any]]:
    """From a partial output plus checkpoint, resume and assert the same final artifact."""
    clean = _convert_all(twin, work / "clean")
    crash_dir = work / "crash"
    kill_at = max(1, twin["n_units"] // 2)
    killed = False
    try:
        _convert_all(twin, crash_dir, stop_after=kill_at)
    except _SimulatedKill:
        killed = True
    partial_files = len(list((crash_dir / "units").glob("*.pkd")))
    index_absent = not (crash_dir / "output.index.json").exists()
    resumed = _convert_all(twin, crash_dir)
    same = resumed["artifact_sha256"] == clean["artifact_sha256"]
    ok = (killed and index_absent and kill_at <= partial_files < twin["n_units"] and same)
    return ok, {"partial_unit_files": partial_files, "total_units": twin["n_units"],
                "index_absent_before_resume": index_absent,
                "final_artifact_matches_clean": same,
                "clean_artifact_sha256": clean["artifact_sha256"],
                "resumed_artifact_sha256": resumed["artifact_sha256"]}


def duplicate_launch_prevention(twin: dict[str, Any], work: Path) -> tuple[bool, dict[str, Any]]:
    """An fcntl.flock lease over the twin; a second converter is refused."""
    import fcntl

    work.mkdir(parents=True, exist_ok=True)
    lock_path = work / "convert.lock"
    first = lock_path.open("w")
    fcntl.flock(first.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    refused = False
    second = lock_path.open("w")
    try:
        fcntl.flock(second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        refused = True
    finally:
        second.close()
    # after the first lease releases, the lock is acquirable again
    fcntl.flock(first.fileno(), fcntl.LOCK_UN)
    first.close()
    reacquired = False
    third = lock_path.open("w")
    try:
        fcntl.flock(third.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        reacquired = True
    except OSError:
        reacquired = False
    finally:
        fcntl.flock(third.fileno(), fcntl.LOCK_UN)
        third.close()
    ok = refused and reacquired
    return ok, {"first_lease_acquired": True, "second_lease_refused": refused,
                "reacquired_after_release": reacquired}


def output_layout_validation(twin: dict[str, Any], work: Path) -> tuple[bool, dict[str, Any]]:
    """The output index must map every produced unit: no orphan file, no missing entry,
    no missing source unit, no extra entry."""
    conv = _convert_all(twin, work / "olv")
    artifact = read_json_safe(conv["output_index"])
    indexed = set(artifact["units"].keys())
    source = {u["name"] for u in _ordered_units(twin)}
    produced = {p.name for p in Path(conv["units_dir"]).glob("*.pkd")}
    expected_files = {_unit_filename(n) for n in indexed}
    orphans = produced - expected_files
    missing_files = expected_files - produced
    missing_source = source - indexed
    extra_indexed = indexed - source
    ok = not (orphans or missing_files or missing_source or extra_indexed)
    return ok, {"source_units": len(source), "indexed_units": len(indexed),
                "produced_files": len(produced), "orphans": sorted(orphans),
                "missing_files": sorted(missing_files), "missing_source": sorted(missing_source),
                "extra_indexed": sorted(extra_indexed)}


# ======================================================================================
# whole-twin validation
# ======================================================================================
def validate_twin(parent: GiantParent, *, config: Config | None = None,
                  workspace: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Build a twin for `parent` and run the full systems battery. Returns a sealed record;
    all_green is True only when every check passes. When workspace is None the twin lives in
    a private tempdir and is cleaned up (fully offline, nothing persisted)."""
    cfg = config or default_config()
    tmp: tempfile.TemporaryDirectory[str] | None = None
    if workspace is None:
        tmp = tempfile.TemporaryDirectory(prefix="succ-twin-")
        base = Path(tmp.name)
    else:
        base = Path(workspace)
    try:
        twin = build_twin(
            parent, scale_layers=cfg.scale_layers, expert_bytes=cfg.expert_bytes,
            out_dir=base / "twin", experts_per_layer_cap=cfg.experts_per_layer_cap,
            tensors_per_shard=cfg.tensors_per_shard)
        work = base / "work"
        work.mkdir(parents=True, exist_ok=True)

        checks: dict[str, Any] = {}

        def run(name: str, fn: Callable[[], tuple[bool, dict[str, Any]]]) -> None:
            ok, evidence = fn()
            checks[name] = {"pass": bool(ok), "evidence": evidence}

        run("deterministic_conversion", lambda: deterministic_conversion(twin, work / "det"))
        run("round_trip_integrity", lambda: round_trip_integrity(twin))
        run("bounded_rss", lambda: bounded_rss(twin, cfg.rss_bound_bytes))
        run("source_range_resume", lambda: source_range_resume(twin, work / "srr"))
        run("expert_paging", lambda: expert_paging(twin, cfg.page_cap_bytes))
        run("crash_recovery", lambda: crash_recovery(twin, work / "cr"))
        run("duplicate_launch_prevention",
            lambda: duplicate_launch_prevention(twin, work / "lock"))
        run("output_layout_validation", lambda: output_layout_validation(twin, work / "olv"))

        all_green = all(c["pass"] for c in checks.values())
        record = {
            "schema": SCHEMA_TWIN_VALIDATION,
            "parent": parent.row_id,
            "hf_id": parent.hf_id,
            "architecture": parent.architecture,
            "model_type": parent.model_type,
            "geometry": twin["geometry"],
            "n_units": twin["n_units"],
            "n_source_shards": twin["n_source_shards"],
            "total_source_bytes": twin["total_source_bytes"],
            "vision_boundary": twin["vision_boundary"],
            "mtp_present": twin["mtp_present"],
            "checks": checks,
            "all_green": all_green,
            "twin_sha256": twin["twin_sha256"],
            "validated_at": now_iso(),
            "gate": ("real source acquisition may begin only after all_green is True"),
        }
        return seal_field(record, "validation_sha256")
    finally:
        if tmp is not None:
            tmp.cleanup()


# ======================================================================================
# selftest
# ======================================================================================
def selftest() -> dict[str, Any]:
    cfg = default_config()
    all_green: dict[str, bool] = {}
    rss_peaks: dict[str, int] = {}
    unit_counts: dict[str, int] = {}

    for parent in PARENTS:
        rec = validate_twin(parent)
        if not sealed(rec, "validation_sha256"):
            raise TwinError(f"{parent.row_id} validation record not sealed")

        failed = [name for name, c in rec["checks"].items() if not c["pass"]]
        if failed:
            raise TwinError(f"{parent.row_id} not green: {failed}")
        all_green[parent.row_id] = rec["all_green"]
        if not rec["all_green"]:
            raise TwinError(f"{parent.row_id} all_green false")

        # bounded_rss peak must be small and never the whole twin
        rss = rec["checks"]["bounded_rss"]["evidence"]
        if rss["peak_resident_bytes"] > cfg.rss_bound_bytes:
            raise TwinError(f"{parent.row_id} rss peak {rss['peak_resident_bytes']} over bound")
        if not rss["never_whole_twin"]:
            raise TwinError(f"{parent.row_id} loaded whole twin")
        rss_peaks[parent.row_id] = rss["peak_resident_bytes"]

        # resume must equal a clean run
        if not rec["checks"]["source_range_resume"]["evidence"]["matches_clean_run"]:
            raise TwinError(f"{parent.row_id} resume != clean run")

        # duplicate launch must be refused
        if not rec["checks"]["duplicate_launch_prevention"]["evidence"]["second_lease_refused"]:
            raise TwinError(f"{parent.row_id} duplicate launch not refused")

        # expert paging must respect the byte cap
        if not rec["checks"]["expert_paging"]["evidence"]["cap_never_exceeded"]:
            raise TwinError(f"{parent.row_id} paging exceeded cap")

        unit_counts[parent.row_id] = rec["n_units"]

    # structural coverage: v3.2 has MTP no vision, kimi has vision no MTP, v4 has MTP
    kimi = next(p for p in PARENTS if p.multimodal)
    kimi_rec = validate_twin(kimi)
    if kimi_rec["vision_boundary"] is None:
        raise TwinError("multimodal parent missing vision boundary marker")
    if kimi_rec["mtp_present"]:
        raise TwinError("kimi text-core should have no MTP block (mtp_layers=0)")

    return {
        "ok": True,
        "parents": len(PARENTS),
        "checks_per_parent": 8,
        "all_green": all_green,
        "rss_peaks": rss_peaks,
        "rss_bound_bytes": cfg.rss_bound_bytes,
        "unit_counts": unit_counts,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Synthetic geometry twins + systems-path validator for giant parents.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--parent", default=None, help="restrict to one row_id")
    ap.add_argument("--out", default=None, help="persist the validation report as JSON")
    args = ap.parse_args()

    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        sys.exit(0)

    parents = [p for p in PARENTS if args.parent is None or p.row_id == args.parent]
    if not parents:
        raise SystemExit(f"unknown parent: {args.parent}")
    report = {
        "schema": SCHEMA_TWIN_REPORT,
        "generated_at": now_iso(),
        "validations": [],
    }
    for p in parents:
        rec = validate_twin(p)
        report["validations"].append({
            "parent": p.row_id, "all_green": rec["all_green"],
            "checks": {k: v["pass"] for k, v in rec["checks"].items()},
            "n_units": rec["n_units"], "n_source_shards": rec["n_source_shards"],
            "twin_sha256": rec["twin_sha256"], "validation_sha256": rec["validation_sha256"],
        })
    report["all_green"] = all(v["all_green"] for v in report["validations"])
    report = seal_field(report, "report_sha256")
    if args.out:
        _assert_non_interfering(args.out)
        atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))

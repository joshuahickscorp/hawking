"""Conservative offline storage planning for one rotating-model work slot.

This module deliberately does *not* observe device allocations, start a model,
or execute a capability check. Its output is therefore an offline candidate,
not a receipt or a promotion signal. The narrow boundary keeps a storage plan
from accidentally becoming an authority for the live TG campaign.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from lab.layout import resolve_workspace_path
from lab.operators.glm52_common import Glm52Error, canonical, read_sealed_json

SCHEMA = "hawking.gravity.range_scheduler.v1"
REQUIRES_LIVE_ALLOCATION_AND_CAPABILITY = "REQUIRES_LIVE_ALLOCATION_AND_CAPABILITY"
DEFAULT_ENVELOPE_BYTES = 90_000_000_000
RANGE_ALIGNMENT_BYTES = 64 * 1024

__all__ = [
    "DEFAULT_ENVELOPE_BYTES",
    "RANGE_ALIGNMENT_BYTES",
    "REQUIRES_LIVE_ALLOCATION_AND_CAPABILITY",
    "RangeScheduleError",
    "conservative_bytes",
    "load_glm52_ranges",
    "plan_candidate",
    "plan_glm52_organ_windows",
    "plan_windowed_candidate",
]


class RangeScheduleError(Glm52Error):
    """Raised when a range plan cannot be accounted for exactly once."""


def _integer(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RangeScheduleError(f"{label} must be an integer, got {value!r}")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise RangeScheduleError(f"{label} must be {qualifier}, got {value}")
    return value


def _multiplier(value: object) -> Fraction:
    if isinstance(value, Fraction):
        if value < 1:
            raise RangeScheduleError(f"scratch_multiplier must be a finite number >= 1, got {value!r}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RangeScheduleError(f"scratch_multiplier must be a finite number >= 1, got {value!r}")
    floating = float(value)
    if not math.isfinite(floating) or floating < 1.0:
        raise RangeScheduleError(f"scratch_multiplier must be a finite number >= 1, got {value!r}")
    # Parsing the user-facing decimal string avoids a binary-float product
    # accidentally rounding a charge down just before ``ceil``.
    return Fraction(str(value))


def _round_up(value: int, alignment_bytes: int = RANGE_ALIGNMENT_BYTES) -> int:
    value = _integer(value, "byte count", allow_zero=True)
    alignment_bytes = _integer(alignment_bytes, "alignment_bytes")
    return ((value + alignment_bytes - 1) // alignment_bytes) * alignment_bytes


def conservative_bytes(
    payload_bytes: int,
    *,
    scratch_multiplier: float = 1.0,
    alignment_bytes: int = RANGE_ALIGNMENT_BYTES,
) -> int:
    """Return a 64-KiB rounded payload charge including conservative scratch.

    Both the source range and the post-scratch value are rounded up. This is
    intentionally more conservative than multiplying a whole-model total: a
    small, independently scheduled range still consumes an allocatable unit.
    """
    payload_bytes = _integer(payload_bytes, "payload_bytes", allow_zero=True)
    multiplier = _multiplier(scratch_multiplier)
    rounded = _round_up(payload_bytes, alignment_bytes)
    scaled = (rounded * multiplier.numerator + multiplier.denominator - 1) // multiplier.denominator
    return _round_up(scaled, alignment_bytes)


def _selection(values: Sequence[str] | None, label: str) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise RangeScheduleError(f"{label} must be a sequence of non-empty strings")
    result = set(values)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise RangeScheduleError(f"{label} must be a non-empty sequence of non-empty strings")
    return result


def _normalise_ranges(ranges: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    if isinstance(ranges, (str, bytes)):
        raise RangeScheduleError("ranges must be an iterable of mapping objects")

    result: list[dict[str, object]] = []
    identities: set[tuple[str, int, int]] = set()
    range_ids: set[str] = set()
    for number, raw in enumerate(ranges):
        if not isinstance(raw, Mapping):
            raise RangeScheduleError(f"range {number} is not a mapping")
        shard = raw.get("shard")
        if not isinstance(shard, str) or not shard:
            raise RangeScheduleError(f"range {number} needs a non-empty shard")

        start = _integer(raw.get("start", raw.get("absolute_start")), f"range {number} start", allow_zero=True)
        end = _integer(raw.get("end", raw.get("absolute_end")), f"range {number} end", allow_zero=True)
        if end <= start:
            raise RangeScheduleError(f"range {number} has non-positive interval [{start}, {end})")
        payload_bytes = end - start
        supplied_payload = raw.get("payload_bytes")
        if supplied_payload is not None and _integer(
            supplied_payload, f"range {number} payload_bytes", allow_zero=True
        ) != payload_bytes:
            raise RangeScheduleError(
                f"range {number} payload_bytes does not equal end - start: "
                f"{supplied_payload!r} != {payload_bytes}"
            )

        name = raw.get("name", raw.get("tensor_name", ""))
        if name is not None and not isinstance(name, str):
            raise RangeScheduleError(f"range {number} name must be a string when provided")
        supplied_id = raw.get("range_id", raw.get("id"))
        range_id = supplied_id if supplied_id is not None else f"{shard}:{start}:{end}:{name or ''}"
        if not isinstance(range_id, str) or not range_id:
            raise RangeScheduleError(f"range {number} range_id must be a non-empty string")
        identity = (shard, start, end)
        if identity in identities:
            raise RangeScheduleError(
                f"range {number} duplicates the accounted interval {shard}[{start}, {end})"
            )
        if range_id in range_ids:
            raise RangeScheduleError(f"range {number} duplicates range_id {range_id!r}")
        identities.add(identity)
        range_ids.add(range_id)
        result.append(
            {
                "range_id": range_id,
                "shard": shard,
                "start": start,
                "end": end,
                "payload_bytes": payload_bytes,
                "name": name or None,
            }
        )

    if not result:
        raise RangeScheduleError("at least one shard or tensor range is required")

    # Counting partially overlapping intervals as independent work would charge
    # some source bytes twice. Adjacent intervals remain separate and exact.
    previous_by_shard: dict[str, tuple[int, int, str]] = {}
    for item in sorted(result, key=lambda row: (str(row["shard"]), int(row["start"]), int(row["end"]))):
        shard = str(item["shard"])
        start = int(item["start"])
        end = int(item["end"])
        previous = previous_by_shard.get(shard)
        if previous is not None and start < previous[1]:
            raise RangeScheduleError(
                f"range {item['range_id']!r} overlaps {previous[2]!r} in "
                f"{shard}: [{start}, {end}) overlaps [{previous[0]}, {previous[1]})"
            )
        previous_by_shard[shard] = (start, end, str(item["range_id"]))
    return result


def load_glm52_ranges(
    manifest_path: str | Path,
    graph_path: str | Path,
    *,
    shard_names: Sequence[str] | None = None,
    tensor_names: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Load selected tensor intervals from matching sealed GLM-5.2 artifacts.

    The returned intervals use half-open ``[start, end)`` byte coordinates.
    They remain only a source selection; use :func:`plan_candidate` to turn
    them into a bounded offline allocation candidate.
    """
    manifest = read_sealed_json(resolve_workspace_path(manifest_path))
    graph = read_sealed_json(resolve_workspace_path(graph_path))
    if manifest.get("repo") != graph.get("repo") or manifest.get("revision") != graph.get("revision"):
        raise RangeScheduleError("sealed GLM52 manifest and dependency graph do not have the same repo/revision")

    files = manifest.get("files")
    tensors = graph.get("tensors")
    if not isinstance(files, list) or not isinstance(tensors, list):
        raise RangeScheduleError("sealed GLM52 artifacts must contain files and tensors lists")

    known_shards: dict[str, Mapping[str, object]] = {}
    for file_row in files:
        if not isinstance(file_row, Mapping):
            raise RangeScheduleError("GLM52 manifest contains a non-mapping file row")
        path = file_row.get("path")
        if isinstance(path, str) and path:
            known_shards[path] = file_row

    requested_shards = _selection(shard_names, "shard_names")
    requested_tensors = _selection(tensor_names, "tensor_names")
    graph_shards = {
        row.get("shard") for row in tensors if isinstance(row, Mapping) and isinstance(row.get("shard"), str)
    }
    graph_names = {
        row.get("name") for row in tensors if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }
    if requested_shards is not None and not requested_shards <= graph_shards:
        missing = sorted(requested_shards - graph_shards)
        raise RangeScheduleError(f"requested shards are absent from the dependency graph: {missing}")
    if requested_tensors is not None and not requested_tensors <= graph_names:
        missing = sorted(requested_tensors - graph_names)
        raise RangeScheduleError(f"requested tensors are absent from the dependency graph: {missing}")

    selected: list[dict[str, object]] = []
    for row in tensors:
        if not isinstance(row, Mapping):
            raise RangeScheduleError("GLM52 dependency graph contains a non-mapping tensor row")
        shard = row.get("shard")
        name = row.get("name")
        if not isinstance(shard, str) or not shard or not isinstance(name, str) or not name:
            raise RangeScheduleError("GLM52 dependency graph tensor is missing shard or name")
        if requested_shards is not None and shard not in requested_shards:
            continue
        if requested_tensors is not None and name not in requested_tensors:
            continue
        manifest_row = known_shards.get(shard)
        if manifest_row is None or manifest_row.get("is_weight") is not True:
            raise RangeScheduleError(f"selected tensor {name!r} refers to non-weight or unknown shard {shard!r}")
        logical_bytes = _integer(manifest_row.get("logical_bytes"), f"manifest logical_bytes for {shard}")
        start = _integer(row.get("absolute_start"), f"graph start for {name}", allow_zero=True)
        end = _integer(row.get("absolute_end"), f"graph end for {name}", allow_zero=True)
        if end > logical_bytes:
            raise RangeScheduleError(
                f"selected tensor {name!r} ends after manifest shard size: {end} > {logical_bytes}"
            )
        selected.append(
            {
                "range_id": f"tensor:{shard}:{name}:{start}:{end}",
                "shard": shard,
                "name": name,
                "start": start,
                "end": end,
                "payload_bytes": row.get("payload_bytes"),
            }
        )
    return _normalise_ranges(selected)


def _active_charge(candidate: Mapping[str, object], number: int) -> tuple[str, int]:
    model_id = candidate.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RangeScheduleError(f"active candidate {number} needs a non-empty model_id")
    totals = candidate.get("totals")
    if not isinstance(totals, Mapping):
        raise RangeScheduleError(f"active candidate {number} needs totals.charged_total_bytes")
    charge = _integer(totals.get("charged_total_bytes"), f"active candidate {number} charged_total_bytes")
    return model_id, charge


def plan_candidate(
    model_id: str,
    *,
    ranges: Iterable[Mapping[str, object]] | None = None,
    manifest_path: str | Path | None = None,
    graph_path: str | Path | None = None,
    shard_names: Sequence[str] | None = None,
    tensor_names: Sequence[str] | None = None,
    artifact_bytes: int = 0,
    metadata_bytes: int = 0,
    scratch_multiplier: float = 1.0,
    active_candidates: Iterable[Mapping[str, object]] = (),
    allow_parallel: bool = False,
    envelope_bytes: int = DEFAULT_ENVELOPE_BYTES,
) -> dict[str, object]:
    """Build a non-authoritative 90-GB admission candidate.

    Supply either explicit ``ranges`` or both sealed GLM52 artifact paths. An
    existing active candidate blocks a new candidate by default. Supplying
    ``allow_parallel=True`` only changes that default after all existing and
    candidate model, artifact, metadata, and scratch charges fit the envelope.
    """
    if not isinstance(model_id, str) or not model_id:
        raise RangeScheduleError("model_id must be a non-empty string")
    if not isinstance(allow_parallel, bool):
        raise RangeScheduleError("allow_parallel must be a boolean")
    envelope_bytes = _integer(envelope_bytes, "envelope_bytes")
    artifact_bytes = _integer(artifact_bytes, "artifact_bytes", allow_zero=True)
    metadata_bytes = _integer(metadata_bytes, "metadata_bytes", allow_zero=True)
    multiplier = _multiplier(scratch_multiplier)

    has_artifact_paths = manifest_path is not None or graph_path is not None
    if ranges is not None and has_artifact_paths:
        raise RangeScheduleError("provide explicit ranges or sealed GLM52 paths, not both")
    if ranges is None:
        if manifest_path is None or graph_path is None:
            raise RangeScheduleError("sealed planning requires both manifest_path and graph_path")
        source_ranges = load_glm52_ranges(
            manifest_path,
            graph_path,
            shard_names=shard_names,
            tensor_names=tensor_names,
        )
        source = "sealed_glm52_manifest_and_graph"
    else:
        if shard_names is not None or tensor_names is not None:
            raise RangeScheduleError("shard_names and tensor_names apply only to sealed GLM52 paths")
        source_ranges = _normalise_ranges(ranges)
        source = "explicit_ranges"

    range_rows: list[dict[str, object]] = []
    range_payload_bytes = 0
    range_rounded_bytes = 0
    range_charged_bytes = 0
    for item in source_ranges:
        payload = int(item["payload_bytes"])
        rounded = _round_up(payload)
        charged = conservative_bytes(payload, scratch_multiplier=multiplier)
        range_payload_bytes += payload
        range_rounded_bytes += rounded
        range_charged_bytes += charged
        range_rows.append({**item, "rounded_bytes": rounded, "charged_bytes": charged})

    artifact_rounded_bytes = _round_up(artifact_bytes)
    metadata_rounded_bytes = _round_up(metadata_bytes)
    scratch_bytes = range_charged_bytes - range_rounded_bytes
    charged_total_bytes = range_charged_bytes + artifact_rounded_bytes + metadata_rounded_bytes

    active: list[dict[str, object]] = []
    active_total_bytes = 0
    for number, candidate in enumerate(active_candidates):
        if not isinstance(candidate, Mapping):
            raise RangeScheduleError(f"active candidate {number} is not a mapping")
        active_model_id, charge = _active_charge(candidate, number)
        active.append({"model_id": active_model_id, "charged_total_bytes": charge})
        active_total_bytes += charge
    combined_total_bytes = active_total_bytes + charged_total_bytes

    reasons: list[str] = []
    if charged_total_bytes > envelope_bytes:
        reasons.append("candidate_exceeds_90gb_envelope")
    if active and not allow_parallel:
        reasons.append("one_active_model_default")
    if allow_parallel and combined_total_bytes > envelope_bytes:
        reasons.append("combined_parallel_charge_exceeds_90gb_envelope")

    return {
        "schema": SCHEMA,
        "status": REQUIRES_LIVE_ALLOCATION_AND_CAPABILITY,
        "authoritative": False,
        "capability_claim": None,
        "live_allocation_observed": False,
        "source": source,
        "model_id": model_id,
        "one_active_model_default": True,
        "parallel_admission_requested": allow_parallel,
        "offline_admission": "ADMITTED" if not reasons else "BLOCKED",
        "offline_block_reasons": reasons,
        "envelope_bytes": envelope_bytes,
        "active_candidates": active,
        "ranges": range_rows,
        "exact_once_range_accounting": True,
        "totals": {
            "range_payload_bytes": range_payload_bytes,
            "range_rounded_bytes": range_rounded_bytes,
            "scratch_bytes": scratch_bytes,
            "artifact_bytes": artifact_bytes,
            "artifact_rounded_bytes": artifact_rounded_bytes,
            "metadata_bytes": metadata_bytes,
            "metadata_rounded_bytes": metadata_rounded_bytes,
            "charged_total_bytes": charged_total_bytes,
            "active_total_bytes": active_total_bytes,
            "combined_total_bytes": combined_total_bytes,
        },
        "live_requirement": (
            "This offline candidate is not an allocation, parity, benchmark, or capability result; "
            "live allocation and capability verification remain required."
        ),
    }


def plan_windowed_candidate(
    model_id: str,
    *,
    windows: Iterable[Mapping[str, object]],
    artifact_bytes: int = 0,
    metadata_bytes: int = 0,
    scratch_multiplier: float = 1.0,
    active_candidates: Iterable[Mapping[str, object]] = (),
    allow_parallel: bool = False,
    envelope_bytes: int = DEFAULT_ENVELOPE_BYTES,
) -> dict[str, object]:
    """Plan exact-once streamed windows and account the *peak*, not the sum.

    Each window must contain a non-empty ``window_id`` and ``ranges`` list.
    ``artifact_bytes``, ``metadata_bytes``, ``carry_bytes`` and
    ``prefetch_bytes`` may be supplied per window; omitted values use the
    function defaults (and zero for carry/prefetch).  Source intervals are
    still charged exactly once across all windows.  Emitted artifacts are
    conservatively retained through later windows: a plan must not turn a
    multi-window artifact family into a false 90-GB admission by taking only
    the largest individual artifact.  ``carry_bytes`` is therefore for
    *additional* caller-declared residency (for example KV state), not for
    prior window artifacts.  A caller which seals and evicts an artifact
    between windows must model that as a separate candidate; this offline
    planner cannot prove an eviction.

    The result is an offline candidate only.  ``totals.charged_total_bytes``
    is the largest window residency, so it can be passed as an active
    candidate charge to a subsequent parallel admission check.
    """
    if not isinstance(model_id, str) or not model_id:
        raise RangeScheduleError("model_id must be a non-empty string")
    if not isinstance(allow_parallel, bool):
        raise RangeScheduleError("allow_parallel must be a boolean")
    envelope_bytes = _integer(envelope_bytes, "envelope_bytes")
    artifact_bytes = _integer(artifact_bytes, "artifact_bytes", allow_zero=True)
    metadata_bytes = _integer(metadata_bytes, "metadata_bytes", allow_zero=True)
    default_multiplier = _multiplier(scratch_multiplier)
    if isinstance(windows, (str, bytes)):
        raise RangeScheduleError("windows must be an iterable of mapping objects")

    normalised_windows: list[dict[str, object]] = []
    window_ids: set[str] = set()
    all_ranges: list[dict[str, object]] = []
    for number, raw in enumerate(windows):
        if not isinstance(raw, Mapping):
            raise RangeScheduleError(f"window {number} is not a mapping")
        window_id = raw.get("window_id", raw.get("id"))
        if not isinstance(window_id, str) or not window_id:
            raise RangeScheduleError(f"window {number} needs a non-empty window_id")
        if window_id in window_ids:
            raise RangeScheduleError(f"window {number} duplicates window_id {window_id!r}")
        window_ids.add(window_id)
        raw_window_ranges = raw.get("ranges")
        if raw_window_ranges is None:
            raise RangeScheduleError(f"window {window_id!r} needs ranges")
        window_ranges = _normalise_ranges(raw_window_ranges)
        for item in window_ranges:
            item = dict(item)
            item["window_id"] = window_id
            all_ranges.append(item)
        multiplier = _multiplier(raw.get("scratch_multiplier", default_multiplier))
        window_artifact = _integer(raw.get("artifact_bytes", artifact_bytes), f"window {window_id} artifact_bytes", allow_zero=True)
        window_metadata = _integer(raw.get("metadata_bytes", metadata_bytes), f"window {window_id} metadata_bytes", allow_zero=True)
        carry = _integer(raw.get("carry_bytes", 0), f"window {window_id} carry_bytes", allow_zero=True)
        prefetch = _integer(raw.get("prefetch_bytes", 0), f"window {window_id} prefetch_bytes", allow_zero=True)
        normalised_windows.append(
            {
                "window_id": window_id,
                "ranges": window_ranges,
                "scratch_multiplier": multiplier,
                "artifact_bytes": window_artifact,
                "metadata_bytes": window_metadata,
                "carry_bytes": carry,
                "prefetch_bytes": prefetch,
            }
        )
    if not normalised_windows:
        raise RangeScheduleError("at least one streamed window is required")

    # Re-run the global overlap/duplicate check after per-window validation.
    # The per-window copy retains the owning window for the result rows.
    global_ranges = _normalise_ranges(all_ranges)
    global_by_id = {str(item["range_id"]): item for item in global_ranges}
    range_rows: list[dict[str, object]] = []
    total_payload = total_rounded = total_charged = 0
    total_artifact = total_artifact_rounded = 0
    total_metadata = total_metadata_rounded = 0
    total_carry = total_carry_rounded = 0
    total_prefetch = total_prefetch_rounded = 0
    retained_artifact_rounded = 0
    window_rows: list[dict[str, object]] = []
    for window in normalised_windows:
        window_id = str(window["window_id"])
        multiplier = window["scratch_multiplier"]
        window_payload = window_rounded = window_charged = 0
        for item in window["ranges"]:
            source = global_by_id[str(item["range_id"])]
            payload = int(source["payload_bytes"])
            rounded = _round_up(payload)
            charged = conservative_bytes(payload, scratch_multiplier=multiplier)
            row = {**source, "window_id": window_id, "rounded_bytes": rounded, "charged_bytes": charged}
            range_rows.append(row)
            window_payload += payload
            window_rounded += rounded
            window_charged += charged
        window_artifact = int(window["artifact_bytes"])
        window_metadata = int(window["metadata_bytes"])
        carry = int(window["carry_bytes"])
        prefetch = int(window["prefetch_bytes"])
        artifact_rounded = _round_up(window_artifact)
        metadata_rounded = _round_up(window_metadata)
        carry_rounded = _round_up(carry)
        prefetch_rounded = _round_up(prefetch)
        scratch = window_charged - window_rounded
        resident = (
            window_charged
            + retained_artifact_rounded
            + artifact_rounded
            + metadata_rounded
            + carry_rounded
            + prefetch_rounded
        )
        window_rows.append(
            {
                "window_id": window_id,
                "range_count": len(window["ranges"]),
                "range_payload_bytes": window_payload,
                "range_rounded_bytes": window_rounded,
                "scratch_bytes": scratch,
                "artifact_bytes": window_artifact,
                "artifact_rounded_bytes": artifact_rounded,
                "prior_retained_artifact_rounded_bytes": retained_artifact_rounded,
                "retained_artifact_rounded_bytes": retained_artifact_rounded + artifact_rounded,
                "metadata_bytes": window_metadata,
                "metadata_rounded_bytes": metadata_rounded,
                "carry_bytes": carry,
                "carry_rounded_bytes": carry_rounded,
                "prefetch_bytes": prefetch,
                "prefetch_rounded_bytes": prefetch_rounded,
                "resident_bytes": resident,
            }
        )
        total_payload += window_payload
        total_rounded += window_rounded
        total_charged += window_charged
        total_artifact += window_artifact
        total_artifact_rounded += artifact_rounded
        total_metadata += window_metadata
        total_metadata_rounded += metadata_rounded
        total_carry += carry
        total_carry_rounded += carry_rounded
        total_prefetch += prefetch
        total_prefetch_rounded += prefetch_rounded
        retained_artifact_rounded += artifact_rounded

    active: list[dict[str, object]] = []
    active_total_bytes = 0
    for number, candidate in enumerate(active_candidates):
        if not isinstance(candidate, Mapping):
            raise RangeScheduleError(f"active candidate {number} is not a mapping")
        active_model_id, charge = _active_charge(candidate, number)
        active.append({"model_id": active_model_id, "charged_total_bytes": charge})
        active_total_bytes += charge
    peak_resident = max(int(row["resident_bytes"]) for row in window_rows)
    combined_peak = active_total_bytes + peak_resident
    reasons: list[str] = []
    if peak_resident > envelope_bytes:
        reasons.append("candidate_peak_exceeds_90gb_envelope")
    if active and not allow_parallel:
        reasons.append("one_active_model_default")
    if allow_parallel and combined_peak > envelope_bytes:
        reasons.append("combined_parallel_peak_exceeds_90gb_envelope")

    return {
        "schema": SCHEMA,
        "status": REQUIRES_LIVE_ALLOCATION_AND_CAPABILITY,
        "authoritative": False,
        "capability_claim": None,
        "live_allocation_observed": False,
        "source": "explicit_windowed_ranges",
        "model_id": model_id,
        "streamed": True,
        "one_active_model_default": True,
        "parallel_admission_requested": allow_parallel,
        "offline_admission": "ADMITTED" if not reasons else "BLOCKED",
        "offline_block_reasons": reasons,
        "envelope_bytes": envelope_bytes,
        "active_candidates": active,
        "windows": window_rows,
        "ranges": range_rows,
        "exact_once_range_accounting": True,
        "totals": {
            "range_payload_bytes": total_payload,
            "range_rounded_bytes": total_rounded,
            "scratch_bytes": total_charged - total_rounded,
            "artifact_bytes": total_artifact,
            "artifact_rounded_bytes": total_artifact_rounded,
            "retained_artifact_rounded_bytes": retained_artifact_rounded,
            "metadata_bytes": total_metadata,
            "metadata_rounded_bytes": total_metadata_rounded,
            "carry_bytes": total_carry,
            "carry_rounded_bytes": total_carry_rounded,
            "prefetch_bytes": total_prefetch,
            "prefetch_rounded_bytes": total_prefetch_rounded,
            "charged_total_bytes": peak_resident,
            "peak_resident_bytes": peak_resident,
            "active_total_bytes": active_total_bytes,
            "combined_total_bytes": combined_peak,
        },
        "live_requirement": (
            "This offline streamed candidate is not an allocation, parity, benchmark, or capability result; "
            "live allocation and capability verification remain required."
        ),
    }


def plan_glm52_organ_windows(
    model_id: str,
    *,
    manifest_path: str | Path,
    graph_path: str | Path,
    artifact_bytes_per_window: int,
    metadata_bytes_per_window: int,
    scratch_multiplier: float = 1.0,
    active_candidates: Iterable[Mapping[str, object]] = (),
    allow_parallel: bool = False,
    envelope_bytes: int = DEFAULT_ENVELOPE_BYTES,
) -> dict[str, object]:
    """Derive a deterministic one-organ-per-window candidate from sealed GLM52 inputs.

    This is a replayable *offline* partition. It assigns every dependency-graph
    tensor to the graph organ that owns it, orders windows by the declared
    execution order, and models one next-organ payload as explicit prefetch.
    It intentionally does not turn a logical safetensors interval into a live
    range request or an allocation observation.
    """
    manifest = read_sealed_json(resolve_workspace_path(manifest_path))
    graph = read_sealed_json(resolve_workspace_path(graph_path))
    if manifest.get("repo") != graph.get("repo") or manifest.get("revision") != graph.get("revision"):
        raise RangeScheduleError("sealed GLM52 manifest and dependency graph do not have the same repo/revision")
    organs = graph.get("organs")
    tensors = graph.get("tensors")
    if not isinstance(organs, list) or not isinstance(tensors, list):
        raise RangeScheduleError("sealed GLM52 dependency graph must contain organs and tensors lists")

    ranges_by_name = {str(row["name"]): row for row in load_glm52_ranges(manifest_path, graph_path)}
    graph_names = {row.get("name") for row in tensors if isinstance(row, Mapping)}
    if len(ranges_by_name) != len(graph_names) or None in graph_names:
        raise RangeScheduleError("GLM52 dependency graph does not declare unique non-empty tensor names")

    ordered: list[tuple[int, Mapping[str, object]]] = []
    for number, organ in enumerate(organs):
        if not isinstance(organ, Mapping):
            raise RangeScheduleError(f"GLM52 organ {number} is not a mapping")
        execution_order = _integer(organ.get("execution_order"), f"GLM52 organ {number} execution_order", allow_zero=True)
        organ_id = organ.get("organ_id")
        names = organ.get("tensor_names")
        if not isinstance(organ_id, str) or not organ_id or not isinstance(names, list) or not names:
            raise RangeScheduleError(f"GLM52 organ {number} is missing organ_id or tensor_names")
        ordered.append((execution_order, organ))
    ordered.sort(key=lambda item: item[0])
    if [order for order, _ in ordered] != list(range(len(ordered))):
        raise RangeScheduleError("GLM52 organ execution_order must be a contiguous zero-based sequence")

    assigned: list[str] = []
    windows: list[dict[str, object]] = []
    partition_windows: list[dict[str, object]] = []
    multiplier = _multiplier(scratch_multiplier)
    artifact_bytes_per_window = _integer(artifact_bytes_per_window, "artifact_bytes_per_window", allow_zero=True)
    metadata_bytes_per_window = _integer(metadata_bytes_per_window, "metadata_bytes_per_window", allow_zero=True)
    for index, (execution_order, organ) in enumerate(ordered):
        organ_id = str(organ["organ_id"])
        names = organ["tensor_names"]
        assert isinstance(names, list)  # validated above; narrows for type checkers.
        if any(not isinstance(name, str) or name not in ranges_by_name for name in names):
            raise RangeScheduleError(f"GLM52 organ {organ_id!r} names an unknown tensor")
        assigned.extend(names)
        prefetch = 0
        if index + 1 < len(ordered):
            prefetch = _integer(
                ordered[index + 1][1].get("source_payload_bytes"),
                f"GLM52 prefetch payload after organ {organ_id}",
                allow_zero=True,
            )
        ranges = [dict(ranges_by_name[name]) for name in names]
        windows.append(
            {
                "window_id": organ_id,
                "ranges": ranges,
                "artifact_bytes": artifact_bytes_per_window,
                "metadata_bytes": metadata_bytes_per_window,
                "prefetch_bytes": prefetch,
                "scratch_multiplier": multiplier,
            }
        )
        partition_windows.append(
            {
                "window_id": organ_id,
                "execution_order": execution_order,
                "range_ids": [str(row["range_id"]) for row in ranges],
                "prefetch_bytes": prefetch,
            }
        )
    if len(assigned) != len(ranges_by_name) or set(assigned) != set(ranges_by_name) or len(set(assigned)) != len(assigned):
        raise RangeScheduleError("GLM52 organs do not partition every tensor exactly once")

    partition_material = {
        "schema": "hawking.glm52.organ_window_partition.v1",
        "repo": manifest["repo"],
        "revision": manifest["revision"],
        "manifest_seal_sha256": manifest["seal_sha256"],
        "dependency_graph_seal_sha256": graph["seal_sha256"],
        "artifact_bytes_per_window": artifact_bytes_per_window,
        "metadata_bytes_per_window": metadata_bytes_per_window,
        "scratch_multiplier": str(multiplier),
        "prefetch_contract": "next_organ_source_payload_bytes",
        "windows": partition_windows,
    }
    candidate = plan_windowed_candidate(
        model_id,
        windows=windows,
        active_candidates=active_candidates,
        allow_parallel=allow_parallel,
        envelope_bytes=envelope_bytes,
    )
    candidate["source"] = "sealed_glm52_organ_window_partition"
    candidate["partition"] = {
        "algorithm": "organ_execution_order.v1",
        "window_partition_sha256": hashlib.sha256(canonical(partition_material)).hexdigest(),
        "window_count": len(partition_windows),
        "range_count": len(ranges_by_name),
        "prefetch_contract": "next_organ_source_payload_bytes",
        "physical_range_fetch_implemented": False,
    }
    return candidate

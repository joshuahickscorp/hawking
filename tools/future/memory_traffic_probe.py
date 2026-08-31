"""MEMORY TRAFFIC PROBE — count DRAM bytes, or refuse.

The resident reports actual_read_bytes_per_token: null with status
NOT_MEASURED_NO_METAL_MEMORY_COUNTER. Every roof in this campaign divides by
the HQ38M20 catalog sum (9,878,901,136). That figure is an information /
accounting floor, not measured DRAM traffic. KV cache, DeltaNet recurrent
state, activations, metadata re-reads and cache misses may add traffic the
roof does not count.

This module enumerates every counter surface reachable from an ordinary
Metal process on this machine (Apple M3 Ultra, macOS Darwin 27, Metal) and
records what each one actually returned. It does not assume MTLCounterSet
membership. It does not convert GPU time, catalog size, or allocated-footprint
deltas into a byte-traffic number.

If a surface reports bytes read or written, HAWKING_MEMORY_TRAFFIC=1 is the
opt-in to sample it. If none does, actual_read_bytes_per_token stays
UNKNOWN and any attempt to emit a number raises.

    python3 tools/future/memory_traffic_probe.py --probe
    python3 tools/future/memory_traffic_probe.py --build
    python3 -m pytest tools/future/test_memory_traffic_probe.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools.future._common import REPO, write_receipt

RECEIPT = "MEMORY_TRAFFIC_PROBE.json"
SCHEMA = "hawking.future.memory_traffic_probe.v1"
VERSION = 1
RECORDED_BY = "tools/future/memory_traffic_probe.py"

UNKNOWN = "UNKNOWN"
STATUS_NO_COUNTER = "NOT_MEASURED_NO_METAL_MEMORY_COUNTER"
STATUS_UNTESTED = "UNTESTED_PROBE_UNAVAILABLE"
ENV_FLAG = "HAWKING_MEMORY_TRAFFIC"

# Catalog accounting floor. Cited so it cannot be mistaken for a measurement.
# Source: receipts/future/MLP_BYTE_CENSUS.json census.active_weight_bytes_per_token
CATALOG_WEIGHT_BYTES_PER_TOKEN = 9_878_901_136
CATALOG_SOURCE = "receipts/future/MLP_BYTE_CENSUS.json#census.active_weight_bytes_per_token"

PROBE_SOURCE = Path(__file__).resolve().parent / "_memory_traffic_probe.m"

# Names that would count as memory *traffic* (bytes moved), not footprint.
# Matched against enumerated counter / statistic names. Conservative on
# purpose: "bytes" alone is not enough (TiledSceneBytes is a buffer size).
_TRAFFIC_MARKERS = (
    "bytes read",
    "bytes written",
    "bytes_read",
    "bytes_written",
    "read_bytes",
    "write_bytes",
    "memory read",
    "memory write",
    "dram read",
    "dram write",
    "read bandwidth",
    "write bandwidth",
    "memory bandwidth",
    "dram bandwidth",
    "bytes/s",
    "byte/s",
    "llc miss",
    "l2 miss",
    "cache miss",
    "memory transaction",
    "mem transaction",
    "dram traffic",
    "memory traffic",
)

# Common-counter-set constants from MTLCounters.h. Presence on a device is
# queried, never assumed; Apple silicon is not guaranteed to expose all three.
COMMON_COUNTER_SETS = ("timestamp", "stageutilization", "statistic")

CLAIM_BOUNDARY = (
    "Enumeration of Metal/IOKit/GPURawCounter surfaces from an ordinary process "
    "on this host. actual_read_bytes_per_token is UNKNOWN unless a named counter "
    "counted bytes transferred. The catalog figure "
    f"{CATALOG_WEIGHT_BYTES_PER_TOKEN} is an accounting floor, not DRAM traffic. "
    "Timestamp deltas are time. currentAllocatedSize, residency allocatedSize and "
    "IOKit 'in use'/'alloc' figures are footprint. None of those is substituted "
    "for a read-byte count."
)


class ProbeUnavailable(Exception):
    """The enumerator could not be built or run. Not the same as 'no counter'."""


class UnmeasuredMemoryTraffic(ValueError):
    """Raised instead of emitting a byte count that no counter counted."""


def _norm_name(name: str) -> str:
    return re.sub(r"[_\-/]+", " ", name).strip().lower()


def name_looks_like_traffic(name: str) -> bool:
    """True only for names that claim bytes moved, not bytes held."""
    n = _norm_name(name)
    if n.startswith("is "):
        return False
    return any(marker in n for marker in _TRAFFIC_MARKERS)


def env_flag_enabled(env: Mapping[str, str] | None = None) -> bool:
    raw = (env or os.environ).get(ENV_FLAG, "")
    return raw.strip() in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


def probe(*, force: bool = False, timeout_s: int = 180) -> dict[str, Any]:
    """Compile and run the Objective-C enumerator. Live device query."""
    cached = getattr(probe, "_cache", None)
    if cached is not None and not force:
        return cached
    clang = shutil.which("clang")
    if not clang:
        raise ProbeUnavailable("clang is not on PATH; Metal counters cannot be enumerated")
    if not PROBE_SOURCE.is_file():
        raise ProbeUnavailable(f"enumerator source missing: {PROBE_SOURCE}")
    with tempfile.TemporaryDirectory(prefix="memory_traffic_probe_") as tmp:
        binary = Path(tmp) / "mtl_traffic_probe"
        build = subprocess.run(
            [
                clang,
                "-fobjc-arc",
                "-O2",
                "-framework", "Metal",
                "-framework", "Foundation",
                "-framework", "IOKit",
                str(PROBE_SOURCE),
                "-o",
                str(binary),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if build.returncode != 0:
            raise ProbeUnavailable(
                f"enumerator did not build: {build.stderr.strip()[-400:]}"
            )
        run = subprocess.run(
            [str(binary)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if run.returncode != 0:
            raise ProbeUnavailable(
                f"enumerator did not run (rc={run.returncode}): "
                f"{(run.stderr or run.stdout).strip()[-400:]}"
            )
        try:
            observed = json.loads(run.stdout)
        except json.JSONDecodeError as exc:
            raise ProbeUnavailable(f"enumerator emitted non-JSON: {exc}") from exc
        observed["_enumerator_stderr"] = (run.stderr or "").strip()
        probe._cache = observed  # type: ignore[attr-defined]
        return observed


def _collect_enumerated_names(observed: Mapping[str, Any]) -> list[dict[str, str]]:
    """Every counter / statistic name the enumerator actually returned."""
    rows: list[dict[str, str]] = []

    def add(source: str, name: str) -> None:
        if not name:
            return
        rows.append({"source": source, "name": name})

    for cset in observed.get("counter_sets") or []:
        set_name = str(cset.get("name") or "")
        add("MTLCounterSet", set_name)
        for cname in cset.get("counter_names") or []:
            add(f"MTLCounterSet.{set_name}", str(cname))
    for name in observed.get("counter_set_names") or []:
        add("MTLCounterSet", str(name))

    grc = observed.get("gpu_raw_counter") or {}
    for group in grc.get("groups") or []:
        add("GPURawCounter.group", str(group.get("name") or ""))
        for src in group.get("sources") or []:
            add("GPURawCounter.source", str(src.get("name") or ""))
            names = src.get("available_counter_names") or []
            if isinstance(names, list):
                for n in names:
                    add("GPURawCounter.counter", str(n))

    iokit = observed.get("iokit") or {}
    for entry in (iokit.get("agxaccelerator_g15x_entries") or []) + (
        iokit.get("ioaccelerator_entries") or []
    ):
        for k in entry.get("PerformanceStatistics_keys") or []:
            add("IOKit.PerformanceStatistics", str(k))
        for n in entry.get("IOReport_channel_names") or []:
            add("IOKit.IOReport", str(n))
    return rows


def classify(observed: Mapping[str, Any] | None, why_unavailable: str = "") -> dict[str, Any]:
    """Decide, from an enumeration, whether any surface reports traffic."""
    if observed is None:
        return {
            "byte_counter_available": False,
            "actual_read_bytes_per_token": UNKNOWN,
            "status": STATUS_UNTESTED,
            "why": why_unavailable or "the enumerator could not run on this host",
            "traffic_names": [],
            "counter_set_names_present": [],
            "common_counter_sets_absent_on_device": list(COMMON_COUNTER_SETS),
        }

    names = _collect_enumerated_names(observed)
    traffic = [row for row in names if name_looks_like_traffic(row["name"])]
    present_sets = [str(n) for n in (observed.get("counter_set_names") or [])]
    present_norm = {_norm_name(n) for n in present_sets}
    absent = [n for n in COMMON_COUNTER_SETS if n not in present_norm]

    sampling = observed.get("supports_counter_sampling") or {}
    device = observed.get("device") or {}
    grc = observed.get("gpu_raw_counter") or {}
    grc_err = (grc.get("copy_error") or {}).get("description") or grc.get("note")
    iokit = observed.get("iokit") or {}
    agx = (iokit.get("agxaccelerator_g15x_entries") or [{}])[0]
    perf_keys = list(agx.get("PerformanceStatistics_keys") or [])

    allocated = observed.get("current_allocated_size") or {}
    residency = observed.get("residency_set") or {}
    mtl4 = observed.get("mtl4_counter_heap") or {}

    stage_samples = observed.get("stage_boundary_counter_samples") or []
    dispatch_samples = observed.get("compute_copy_counter_samples") or []
    timestamp_worked = any(
        isinstance(s, dict)
        and isinstance(s.get("timestamp_delta_ns"), (int, float))
        and s.get("timestamp_delta_ns", 0) > 0
        for s in stage_samples
    )

    surfaces = {
        "MTLCounterSet": {
            "present": present_sets,
            "n": observed.get("n_counter_sets"),
            "detail": [
                {
                    "name": cset.get("name"),
                    "counters": cset.get("counter_names"),
                    "sample_buffer_created": bool(cset.get("sample_buffer_created")),
                    "sample_buffer_error": cset.get("sample_buffer_error"),
                }
                for cset in (observed.get("counter_sets") or [])
            ],
            "reports_memory_traffic": False,
            "what_it_returned": (
                f"{observed.get('n_counter_sets')} set(s) named {present_sets}. "
                "Common-set constants stageutilization and statistic were queried "
                "and are recorded absent when missing."
            ),
        },
        "MTLCommonCounterSet_constants": observed.get("mtl_common_counter_set_constants"),
        "MTLCommonCounter_constants": observed.get("mtl_common_counter_constants"),
        "supports_counter_sampling": sampling,
        "timestamp_counters": {
            "set_present": "timestamp" in present_norm,
            "stage_boundary_sampling_supported": bool(sampling.get("AtStageBoundary")),
            "dispatch_boundary_sampling_supported": bool(sampling.get("AtDispatchBoundary")),
            "stage_boundary_samples_nonzero": timestamp_worked,
            "reports_memory_traffic": False,
            "what_it_returned": (
                "GPUTimestamp at compute-pass stage boundary produces a time "
                "delta in nanoseconds. encoder sampleCountersInBuffer (dispatch "
                "boundary) is unsupported here and resolves to 0. Time is not bytes."
            ),
        },
        "stage_utilization": {
            "set_present": "stageutilization" in present_norm,
            "reports_memory_traffic": False,
            "what_it_returned": (
                "MTLCommonCounterSetStageUtilization constant exists in the SDK "
                f"as {(observed.get('mtl_common_counter_set_constants') or {}).get('MTLCommonCounterSetStageUtilization')!r}; "
                "device.counterSets does not include it on this GPU."
            ),
        },
        "statistic": {
            "set_present": "statistic" in present_norm,
            "reports_memory_traffic": False,
            "what_it_returned": (
                "MTLCommonCounterSetStatistic constant exists in the SDK "
                f"as {(observed.get('mtl_common_counter_set_constants') or {}).get('MTLCommonCounterSetStatistic')!r}; "
                "device.counterSets does not include it on this GPU. The "
                "statistic struct (invocations, not bytes) is therefore unreachable."
            ),
        },
        "MTL4CounterHeap": {
            "public_types": mtl4.get("header_types"),
            "memory_traffic_heap_type_in_public_api": bool(
                mtl4.get("memory_traffic_heap_type_in_public_api")
            ),
            "reports_memory_traffic": False,
            "what_it_returned": (
                "Public MTL4CounterHeapType is Invalid | Timestamp only. No byte-traffic heap type."
            ),
        },
        "MTLDevice.currentAllocatedSize": {
            "before_16mib_buffer": allocated.get("before_16mib_buffer"),
            "after_16mib_buffer": allocated.get("after_16mib_buffer"),
            "delta": allocated.get("delta"),
            "reports_memory_traffic": False,
            "what_it_returned": allocated.get("meaning"),
        },
        "MTLResidencySet": {
            "api_available": residency.get("api_available"),
            "allocatedSize_after_commit": residency.get("allocatedSize_after_commit"),
            "reports_memory_traffic": False,
            "what_it_returned": residency.get("meaning"),
        },
        "GPURawCounter": {
            "dlopen": grc.get("dlopen"),
            "n_groups": grc.get("n_groups"),
            "copy_error": grc.get("copy_error"),
            "stderr": observed.get("_enumerator_stderr"),
            "reports_memory_traffic": False,
            "reachable_without_elevated_privileges": bool(grc.get("n_groups")),
            "what_it_returned": (
                "dlopen of GPURawCounter.framework succeeded. "
                "GRCCopyAllCounterSourceGroupWithError returned "
                f"{grc_err!r} with n_groups={grc.get('n_groups')}. "
                "Instruments-style raw GPU counters are not instantiable from "
                "an ordinary Metal process here."
            ),
        },
        "IOKit.IOAccelerator": {
            "IOClass": agx.get("IOClass"),
            "gpu_core_count": agx.get("gpu-core-count"),
            "GPURawCounterBundleName": agx.get("GPURawCounterBundleName"),
            "PerformanceStatistics_keys": perf_keys,
            "PerformanceStatistics": agx.get("PerformanceStatistics"),
            "n_IOReport_channels": len(agx.get("IOReport_channel_names") or []),
            "IOReport_channel_names": agx.get("IOReport_channel_names") or [],
            "reports_memory_traffic": False,
            "what_it_returned": (
                "Unprivileged IORegistry snapshot of AGXAcceleratorG15X. Keys are "
                "allocation ('Alloc system memory', 'In use system memory'), "
                "utilization percentages, TiledSceneBytes (tiler scene buffer), "
                "and recovery counters. No bytes-read or bytes-written key."
            ),
        },
        "IOAccelMemoryInfo": {
            **(observed.get("ioaccel_memory_info") or {}),
            "reports_memory_traffic": False,
        },
        "MetalMetrics": {
            **(observed.get("metal_metrics") or {}),
            "reports_memory_traffic": False,
        },
        "MTLDevice.sampleTimestamps": {
            **(observed.get("sample_timestamps") or {}),
            "reports_memory_traffic": False,
        },
    }

    return {
        "byte_counter_available": bool(traffic),
        "actual_read_bytes_per_token": UNKNOWN if not traffic else None,
        "status": STATUS_NO_COUNTER if not traffic else "COUNTER_PRESENT_NOT_YET_SAMPLED",
        "why": (
            "No enumerated MTLCounterSet, GPURawCounter, IOKit PerformanceStatistics "
            "or IOReport channel on this device reports bytes read, bytes written, "
            "DRAM bandwidth, cache misses or memory transactions. Timestamp, "
            "allocation and utilization surfaces are present and are not traffic."
        ),
        "traffic_names": traffic,
        "enumerated_names": names,
        "counter_set_names_present": present_sets,
        "common_counter_sets_absent_on_device": absent,
        "device": {
            "name": device.get("name"),
            "hasUnifiedMemory": device.get("hasUnifiedMemory"),
            "recommendedMaxWorkingSetSize": device.get("recommendedMaxWorkingSetSize"),
            "maxBufferLength": device.get("maxBufferLength"),
            "registryID": device.get("registryID"),
        },
        "surfaces": surfaces,
        "dispatch_boundary_samples": dispatch_samples,
        "stage_boundary_samples": stage_samples,
        "env_flag": ENV_FLAG,
        "env_flag_enabled": env_flag_enabled(),
    }


def emit_actual_read_bytes(
    value: Any,
    *,
    counted_by: str | None = None,
    observed: Mapping[str, Any] | None = None,
) -> int:
    """The only door that would return a measured byte count.

    Raises unless a named counter that looks like traffic both exists on the
    device and is credited as having counted `value`. Catalog size is never
    an acceptable counted_by.
    """
    if value == CATALOG_WEIGHT_BYTES_PER_TOKEN:
        raise UnmeasuredMemoryTraffic(
            f"REFUSED: {CATALOG_WEIGHT_BYTES_PER_TOKEN} is the catalog accounting "
            "floor (MLP_BYTE_CENSUS active_weight_bytes_per_token), not measured "
            "DRAM traffic. actual_read_bytes_per_token stays UNKNOWN."
        )
    if counted_by is None or counted_by == "":
        raise UnmeasuredMemoryTraffic(
            "REFUSED: actual_read_bytes_per_token cannot be emitted without a "
            "named counter that counted it."
        )
    if counted_by in {CATALOG_SOURCE, "catalog", "active_bytes_per_token", "catalog_weight_bytes_per_token"}:
        raise UnmeasuredMemoryTraffic(
            f"REFUSED: {counted_by!r} is an accounting source, not a memory-traffic counter."
        )
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise UnmeasuredMemoryTraffic(
            f"REFUSED: {value!r} is not a counted non-negative int."
        )
    verdict = classify(observed)
    if not verdict["byte_counter_available"]:
        raise UnmeasuredMemoryTraffic(
            "REFUSED: no Metal/IOKit/GPURawCounter surface on this device reports "
            f"bytes transferred; cannot credit {counted_by!r} for {value}."
        )
    legal = {row["name"] for row in verdict["traffic_names"]}
    if counted_by not in legal:
        raise UnmeasuredMemoryTraffic(
            f"REFUSED: {counted_by!r} is not among the traffic names enumerated "
            f"on this device ({sorted(legal) or 'none'})."
        )
    return value


def measured_read_bytes_per_token(
    observed: Mapping[str, Any] | None = None,
    *,
    catalog: int | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Return counted DRAM-read bytes per token, or raise.

    The env flag HAWKING_MEMORY_TRAFFIC=1 is the opt-in to attempt a sample.
    It does not license an estimate. Catalog size is refused as an input.
    """
    if catalog is not None:
        raise UnmeasuredMemoryTraffic(
            f"REFUSED: catalog figure {catalog} is an accounting floor, not "
            "measured DRAM traffic."
        )
    if observed is None:
        try:
            observed = probe()
        except ProbeUnavailable as exc:
            raise UnmeasuredMemoryTraffic(
                f"REFUSED: enumerator unavailable ({exc}); cannot invent a byte count."
            ) from exc
    verdict = classify(observed)
    if not env_flag_enabled(env) and env is not None:
        raise UnmeasuredMemoryTraffic(
            f"REFUSED: {ENV_FLAG} is unset; measurement is opt-in and no estimate "
            "is substituted."
        )
    if not verdict["byte_counter_available"]:
        raise UnmeasuredMemoryTraffic(
            "REFUSED: no counter on this device reports bytes read or written; "
            "actual_read_bytes_per_token stays UNKNOWN. "
            f"Present MTLCounterSet(s): {verdict['counter_set_names_present'] or 'none'}. "
            f"Absent common sets: {verdict['common_counter_sets_absent_on_device']}."
        )
    raise UnmeasuredMemoryTraffic(
        "REFUSED: a traffic-named counter was listed but this module has not "
        "sampled a per-token read-byte value from it."
    )


def validate_receipt(doc: Mapping[str, Any]) -> None:
    """Fail closed if a receipt carries a number no counter counted."""
    value = doc.get("actual_read_bytes_per_token")
    available = bool(doc.get("byte_counter_available"))
    if value == CATALOG_WEIGHT_BYTES_PER_TOKEN:
        raise UnmeasuredMemoryTraffic(
            "REFUSED: receipt used the catalog figure as actual_read_bytes_per_token."
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool) and not available:
        raise UnmeasuredMemoryTraffic(
            f"REFUSED: receipt has actual_read_bytes_per_token={value!r} but "
            "byte_counter_available is false."
        )
    if value not in {UNKNOWN, None} and not available:
        if not (isinstance(value, dict) and value.get("value") == UNKNOWN):
            raise UnmeasuredMemoryTraffic(
                f"REFUSED: actual_read_bytes_per_token={value!r} is not UNKNOWN "
                "and no counter is available."
            )


def _surface_table(verdict: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, body in (verdict.get("surfaces") or {}).items():
        if not isinstance(body, dict):
            rows.append({"surface": name, "reports_memory_traffic": False, "returned": body})
            continue
        rows.append({
            "surface": name,
            "reports_memory_traffic": bool(body.get("reports_memory_traffic")),
            "what_it_returned": body.get("what_it_returned") or body.get("meaning"),
            "present": body.get("present", body.get("set_present", body.get("dlopen"))),
        })
    return rows


def build(observed: Mapping[str, Any] | None = None) -> Path:
    why = ""
    if observed is None:
        try:
            observed = probe()
        except (ProbeUnavailable, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            why = f"{type(exc).__name__}: {exc}"
            observed = None

    verdict = classify(observed, why)
    try:
        measured_read_bytes_per_token(observed, env={ENV_FLAG: "1"})
    except UnmeasuredMemoryTraffic as exc:
        refusal = str(exc)
    else:
        refusal = ""

    actual: Any = UNKNOWN
    if verdict["byte_counter_available"] and verdict.get("actual_read_bytes_per_token") not in {None, UNKNOWN}:
        actual = verdict["actual_read_bytes_per_token"]

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Establish what this machine can report about memory traffic from "
            "inside a Metal process, and refuse to fill actual_read_bytes_per_token "
            "with the catalog floor or any other uncounted figure."
        ),
        "evidence_class": "STATIC_ONLY",
        "is_a_measurement": False,
        "why_not_a_measurement": (
            "No enumerated surface reports bytes transferred. Timestamp samples "
            "are time. currentAllocatedSize / residency / IOKit in-use figures "
            "are footprint. The enumerator issued a 1 MiB and 4 MiB compute copy "
            "only to see whether any counter scaled as traffic; none did. That "
            "copy is not a token and is not a bandwidth measurement."
        ),
        "gpu_authority": False,
        "actual_read_bytes_per_token": actual,
        "status": verdict["status"],
        "byte_counter_available": verdict["byte_counter_available"],
        "catalog_weight_bytes_per_token": CATALOG_WEIGHT_BYTES_PER_TOKEN,
        "catalog_source": CATALOG_SOURCE,
        "catalog_is_not_traffic": True,
        "compared_against_catalog": (
            "cannot compare: actual_read_bytes_per_token is UNKNOWN, catalog "
            f"is {CATALOG_WEIGHT_BYTES_PER_TOKEN}"
        ),
        "env_flag": ENV_FLAG,
        "env_flag_meaning": (
            f"{ENV_FLAG}=1 opts into sampling a real byte-traffic counter if "
            "one exists. It does not license an estimate. On this device the "
            "flag still yields UNKNOWN / UnmeasuredMemoryTraffic."
        ),
        "device": verdict.get("device"),
        "counter_set_names_present": verdict.get("counter_set_names_present"),
        "common_counter_sets_absent_on_device": verdict.get("common_counter_sets_absent_on_device"),
        "traffic_names": verdict.get("traffic_names"),
        "why": verdict.get("why"),
        "refusal": refusal,
        "surfaces": verdict.get("surfaces"),
        "surface_table": _surface_table(verdict),
        "stage_boundary_samples": verdict.get("stage_boundary_samples"),
        "dispatch_boundary_samples": verdict.get("dispatch_boundary_samples"),
        "enumerator_stderr": None if observed is None else observed.get("_enumerator_stderr"),
        "negative_findings": [
            "device.counterSets on Apple M3 Ultra is ['timestamp'] only; "
            "stageutilization and statistic are SDK constants, not present sets",
            "supportsCounterSampling is true only for AtStageBoundary; "
            "AtDispatchBoundary/AtDrawBoundary/AtBlitBoundary/AtTileDispatchBoundary are false",
            "hawking ProdCbGpu samples via encoder sampleCountersInBuffer "
            "(dispatch boundary), which this device reports as unsupported; "
            "stage-boundary compute-pass attachments do produce GPUTimestamp",
            "GPURawCounter.framework dlopens but GRCCopyAllCounterSourceGroupWithError "
            "fails with 'Fail to instantiate AGXGPURawCounterSourceGroup' from an "
            "ordinary process (no entitlement, no Instruments session)",
            "powermetrics --samplers gpu_power requires superuser",
            "xctrace requires a full Xcode install; CommandLineTools is not enough",
            "IOKit PerformanceStatistics and IOReport memory channels are "
            "allocation/utilization, not bytes transferred",
            "KV cache, DeltaNet recurrent state and activations remain uncounted; "
            "this probe does not invent them",
        ],
        "what_this_does_not_prove": [
            "that DRAM traffic equals the catalog",
            "that DRAM traffic exceeds the catalog",
            "a GB/s figure, a roof, or a TPS implication",
            "that Instruments GPU counters do not exist — only that they are "
            "not instantiable from this process",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
        "resident_callable": {
            "entry_point": "tools.future.memory_traffic_probe.probe()",
            "measurement_entry_point": (
                "tools.future.memory_traffic_probe.measured_read_bytes_per_token()"
            ),
            "raises": "UnmeasuredMemoryTraffic when asked for a number no counter counted",
            "receipt": f"receipts/future/{RECEIPT}",
            "env_flag": ENV_FLAG,
        },
    }
    validate_receipt(doc)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


selftest = build


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--measure", action="store_true", help="opt-in sample; raises if uncounted")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.probe:
        try:
            print(json.dumps(probe(force=True), indent=1, sort_keys=True, default=str))
        except ProbeUnavailable as exc:
            print(json.dumps({"untested": str(exc)}, indent=1))
            return 1
        return 0
    if a.measure:
        try:
            value = measured_read_bytes_per_token(env=os.environ)
        except UnmeasuredMemoryTraffic as exc:
            print(json.dumps({"refused": str(exc), "actual_read_bytes_per_token": UNKNOWN}, indent=1))
            return 2
        print(json.dumps({"actual_read_bytes_per_token": value}, indent=1))
        return 0
    out = build()
    print(out)
    doc = json.loads(out.read_text())
    print(json.dumps({
        "actual_read_bytes_per_token": doc["actual_read_bytes_per_token"],
        "status": doc["status"],
        "counter_set_names_present": doc["counter_set_names_present"],
        "byte_counter_available": doc["byte_counter_available"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

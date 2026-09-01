#!/usr/bin/env python3
"""G103: memory and disk are scheduled resources, and the peak is predicted first.

S027 §5-§7, §26-§28, §43-§44. Live UMA accounting from vm_stat and sysctl, a
peak prediction made BEFORE a load rather than an OOM discovered after one, a
residency score, and the resource lanes a scheduler must keep separate.

THE MACHINE IS 96 GB AND THE LIBRARY IS NOT. Of the 8 sealed specimens, several
exceed free memory outright and one exceeds total memory. Admission is not a
formality here; it is the difference between a load and a swap storm.

METAL WORKING SET IS THE ADMISSION GATE, NOT FREE RAM. mmap shares pages, so a
specimen's source bytes are an upper bound on what a load costs and a lower
bound on what an executor needs resident. Both are reported, and neither is
called the answer.

    python3 tools/future/uma_resource_ledger.py --build
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402
import specimen_registry as sr  # noqa: E402

RECORDED_BY = "tools/future/uma_resource_ledger.py"
RECEIPT_NAME = "UMA_RESOURCE_LEDGER.json"

LAKE_VOLUME = "/Volumes/corpdrive"

# S027 §26. Separate lanes, because a load and a compute may overlap and a load
# and a download contend only weakly (measured 1.23x in G102).
RESOURCE_LANES = ("DISK_READ", "DISK_WRITE", "MODELLAKE_NETWORK", "GPU", "CPU", "UMA")

# S027 §43. Free memory has option value; filling it because it is empty is how
# a scheduler loses the ability to accept a sudden high-value specimen.
HEADROOM_FRACTION = 0.15


class ResourceRefused(RuntimeError):
    """A live resource reading is unavailable, so nothing may be admitted."""


def _vm_stat() -> dict[str, int]:
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise ResourceRefused(f"vm_stat is unavailable: {e}") from e
    m = re.search(r"page size of (\d+) bytes", out)
    page = int(m.group(1)) if m else 16384
    rows: dict[str, int] = {}
    for line in out.splitlines()[1:]:
        k, _, v = line.partition(":")
        v = v.strip().rstrip(".")
        if v.isdigit():
            rows[k.strip()] = int(v) * page
    if not rows:
        raise ResourceRefused("vm_stat returned no page counts")
    return {"page_size": page, **rows}


def _total_memory() -> int:
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                  capture_output=True, text=True,
                                  check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError, ValueError) as e:
        raise ResourceRefused(f"hw.memsize is unavailable: {e}") from e


def memory() -> dict[str, Any]:
    vm = _vm_stat()
    total = _total_memory()
    free = vm.get("Pages free", 0)
    spec = vm.get("Pages speculative", 0)
    inactive = vm.get("Pages inactive", 0)
    active = vm.get("Pages active", 0)
    wired = vm.get("Pages wired down", 0)
    compressed = vm.get("Pages occupied by compressor", 0)
    # Reclaimable is free plus what the kernel can take back without swapping.
    reclaimable = free + spec + inactive
    return {
        "total_bytes": total,
        "total_gb": round(total / 1e9, 1),
        "free_bytes": free,
        "free_gb": round(free / 1e9, 1),
        "reclaimable_bytes": reclaimable,
        "reclaimable_gb": round(reclaimable / 1e9, 1),
        "active_gb": round(active / 1e9, 1),
        "wired_gb": round(wired / 1e9, 1),
        "compressed_gb": round(compressed / 1e9, 1),
        "headroom_reserved_gb": round(total * HEADROOM_FRACTION / 1e9, 1),
        "admissible_bytes": max(0, int(reclaimable - total * HEADROOM_FRACTION)),
        "admissible_gb": round(
            max(0, reclaimable - total * HEADROOM_FRACTION) / 1e9, 1),
        "why_reclaimable_and_not_free": (
            "inactive and speculative pages are returned without swapping, so "
            "free alone understates what a load may use. Neither is Metal's "
            "working set, which is the real admission gate."
        ),
        "headroom_is_reserved_because": (
            "S027 §43: free memory has option value. Filling it because it is "
            "empty is how a scheduler loses the ability to accept a sudden "
            "high-value specimen or a child resident."
        ),
    }


def disk() -> dict[str, Any]:
    try:
        u = shutil.disk_usage(LAKE_VOLUME)
    except OSError as e:
        raise ResourceRefused(f"{LAKE_VOLUME} is not readable: {e}") from e
    return {
        "volume": LAKE_VOLUME,
        "total_gb": round(u.total / 1e9, 1),
        "free_gb": round(u.free / 1e9, 1),
        "used_fraction": round(u.used / u.total, 4),
    }


def predict_peak(source_bytes: int) -> dict[str, Any]:
    """S027 §5: predict the peak BEFORE loading, not discover OOM after."""
    m = memory()
    admissible = m["admissible_bytes"]
    return {
        "source_bytes": source_bytes,
        "source_gb": round(source_bytes / 1e9, 2),
        "admissible_gb": m["admissible_gb"],
        "fits": source_bytes <= admissible,
        "exceeds_total_memory": source_bytes > m["total_bytes"],
        "margin_gb": round((admissible - source_bytes) / 1e9, 2),
        "source_bytes_is_an_upper_bound_on_the_load": (
            "an mmap'd load shares pages with the page cache and does not "
            "necessarily resident-fault the whole file, so source bytes "
            "OVERSTATE what a read costs in memory"
        ),
        "and_a_lower_bound_on_the_executor": (
            "an executor that needs the weights in a Metal working set needs at "
            "least this much, and typically more for activations and state. "
            "Neither bound is the answer; both are reported so a scheduler can "
            "refuse the obvious cases without pretending to know the rest."
        ),
    }


def admission_table() -> list[dict[str, Any]]:
    rows = []
    for s in sorted(sr.schedulable(), key=lambda x: x["source_bytes"] or 0):
        p = predict_peak(s["source_bytes"] or 0)
        rows.append({
            "id": s["id"],
            "source_gb": p["source_gb"],
            "fits_in_admissible": p["fits"],
            "exceeds_total_memory": p["exceeds_total_memory"],
            "margin_gb": p["margin_gb"],
        })
    return rows


def residency_score(*, source_bytes: int, expected_reuses: int,
                    scientific_priority: float) -> dict[str, Any]:
    """S027 §7. A policy, not a frozen formula - and it prices the RELOAD."""
    import specimen_load_cost as lc
    cold = lc.rates()["quiet_cold_MB_per_s"] * 1e6
    reload_s = source_bytes / cold
    memory_cost_gb = source_bytes / 1e9
    value = (expected_reuses * reload_s * scientific_priority) / max(memory_cost_gb, 1e-9)
    return {
        "expected_reuses": expected_reuses,
        "reload_seconds": round(reload_s, 1),
        "scientific_priority": scientific_priority,
        "memory_cost_gb": round(memory_cost_gb, 2),
        "residency_value": round(value, 3),
        "formula": (
            "expected_reuses x reload_seconds x scientific_priority / memory_gb"
        ),
        "the_reload_cost_is_measured_not_assumed": (
            f"reload seconds come from the measured {round(cold / 1e6, 1)} MB/s "
            "cold rate on this volume, not from a nominal bandwidth"
        ),
        "not_frozen": (
            "S027 §7 explicitly declines to freeze this formula. It exists so "
            "KEEP-WARM versus EVICT is a rational decision rather than a reflex, "
            "and it should be replaced when eviction data exists to fit it."
        ),
    }


def build() -> dict[str, Any]:
    m = memory()
    tbl = admission_table()
    return {
        "obligation": "G103",
        "authority": "S027 §5-§7, §26-§28, §43-§44",
        "resource_lanes": list(RESOURCE_LANES),
        "memory": m,
        "disk": disk(),
        "admission_table": tbl,
        "n_sealed_that_do_not_fit": sum(1 for r in tbl if not r["fits_in_admissible"]),
        "n_sealed_exceeding_total_memory": sum(
            1 for r in tbl if r["exceeds_total_memory"]),
        "residency_score_example": residency_score(
            source_bytes=61_090_000_000, expected_reuses=3,
            scientific_priority=1.0),
        "lane_overlap_rule": {
            "measured_in": "receipts/future/SPECIMEN_LOAD_COST.json",
            "rule": (
                "DISK_READ and MODELLAKE_NETWORK may overlap: contention was "
                "measured at 1.23x while residency is worth 142x. Do not "
                "serialize a load behind a download."
            ),
        },
        "protected_lease_overrides": {
            "authority": "S027 §28",
            "rule": (
                "a protected absolute measurement suspends contaminating "
                "downloads and large loads, then resumes them automatically. "
                "This is an EXCEPTION window, not normal operation - and G095 "
                "showed the supervisor must be suspended first or it respawns "
                "the workers mid-window."
            ),
        },
        "what_this_does_not_know": (
            "Metal's working set. Apple Silicon admission is gated by the "
            "working set rather than by free RAM, and nothing here reads it. "
            "The admission table refuses the obvious cases and does not pretend "
            "to decide the marginal ones."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_measured_receipt(
            REPO / "receipts" / "future" / RECEIPT_NAME, doc, RECORDED_BY,
            provenance=measurement_provenance(lock_held=False, lane="g103-uma"),
        ))
        return 0
    print(json.dumps({k: doc[k] for k in
                      ("memory", "disk", "admission_table",
                       "n_sealed_that_do_not_fit", "lane_overlap_rule")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

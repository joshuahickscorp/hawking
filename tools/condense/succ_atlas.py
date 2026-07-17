#!/usr/bin/env python3.12
"""M3 Ultra resource atlas (master goal section 9), non-competing edition.

A FULL CPU/GPU/storage benchmark would materially compete with the live legacy worker
(20 threads at ~1990% CPU), which non-interference forbids. So this atlas does what is safe
NOW: a read-only machine inventory, and it DERIVES the empirical thread/runtime profile from
the already-completed campaign cells (which ran at their own thread count) via the sealed
harvest, rather than launching a competing benchmark. The full active benchmark (thread
sweep, GPU break-even, storage queue depths) is emitted as a post-release plan, not run.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import EcoError, seal_field, sealed, now_iso  # noqa: E402

ATLAS_SCHEMA = "hawking.successor.resource_atlas.v1"


def _sysctl(key: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", key], text=True, capture_output=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def inventory() -> dict[str, Any]:
    mem = _sysctl("hw.memsize")
    return {
        "cpu_logical": os.cpu_count(),
        "cpu_physical": _sysctl("hw.physicalcpu"),
        "perf_cores": _sysctl("hw.perflevel0.physicalcpu"),
        "eff_cores": _sysctl("hw.perflevel1.physicalcpu"),
        "ram_bytes": int(mem) if mem.isdigit() else None,
        "ram_gib": round(int(mem) / 1073741824, 1) if mem.isdigit() else None,
        "free_disk_gb": round(shutil.disk_usage(os.path.expanduser("~")).free / 1e9, 1),
        "read_only": True,
    }


def harvest_derived_profile(harvest: dict[str, Any] | None) -> dict[str, Any]:
    """Derive per-branch seconds-per-billion from the sealed harvest wall times (no new run)."""
    if not harvest:
        return {"available": False, "note": "no harvest supplied"}
    import succ_harvest
    obs = succ_harvest.eta_observations(harvest)
    per_branch: dict[str, list[float]] = {}
    for o in obs:
        pb = o["wall_seconds"] / max(o["stored_params_b"], 1e-9)
        per_branch.setdefault(o["branch"], []).append(pb)
    summary = {}
    for branch, vals in per_branch.items():
        vals.sort()
        n = len(vals)
        summary[branch] = {"n": n, "median_sec_per_billion": round(vals[n // 2], 1),
                           "min": round(vals[0], 1), "max": round(vals[-1], 1)}
    return {"available": bool(summary), "observed_thread_context": "campaign ran at its own thread count",
            "per_branch_sec_per_billion": summary,
            "note": "derived from completed cells, not a competing benchmark"}


def deferred_benchmark_plan() -> dict[str, Any]:
    return {
        "runs_only_post_release": True,
        "reason": "an active thread sweep / GPU break-even / storage queue benchmark would compete "
                  "with the live legacy worker; non-interference defers it to after release",
        "cpu_thread_sweep": [1, 2, 4, 8, 12, 16, 20],
        "cpu_phases": ["source_decode", "census", "transforms", "spectral_sketch", "codebook_fit",
                       "factor_fit", "quantize", "entropy_code", "hash", "pack", "attest",
                       "eval_preprocess", "doctor_optimize"],
        "gpu_metal_paths": ["rht_rotation", "tensor_stats", "reconstruction", "codebook_assign",
                            "factor_fit", "residual_fit", "expert_page_decode", "low_bit_gemv",
                            "doctor_corrections", "tile_synthesis"],
        "gpu_measure": "cpu_vs_gpu break-even tensor size",
        "storage_bench": ["ssd_seq_read", "ssd_random_read", "mmap_faults", "concurrent_read_write",
                          "fsync", "apfs_cow", "external_tb5_if_present", "queue_depths",
                          "page_sizes_small_expert_to_multi_mb_shard"],
        "pipeline_overlap_gate": ["deterministic_output", "rss_below_ceiling", "swap_within_policy",
                                  "memory_pressure_normal", "thermal_acceptable", "wall_time_improves"],
        "never": "advertised bandwidth used as measured ETA input",
    }


def build_atlas(harvest: dict[str, Any] | None = None) -> dict[str, Any]:
    atlas = {
        "schema": ATLAS_SCHEMA,
        "generated_at": now_iso(),
        "inventory": inventory(),
        "harvest_derived_profile": harvest_derived_profile(harvest),
        "deferred_active_benchmark": deferred_benchmark_plan(),
        "non_interference": "read-only inventory + harvest-derived profile only; no competing benchmark run",
    }
    return seal_field(atlas, "atlas_sha256")


def selftest() -> dict[str, Any]:
    atlas = build_atlas()
    if not sealed(atlas, "atlas_sha256"):
        raise EcoError("atlas not sealed")
    inv = atlas["inventory"]
    if not inv.get("read_only") or inv.get("cpu_logical") is None:
        raise EcoError("inventory incomplete")
    if not atlas["deferred_active_benchmark"]["runs_only_post_release"]:
        raise EcoError("active benchmark must be deferred (non-interference)")
    # harvest-derived profile with a tiny synthetic harvest
    hv = {"schema": "hawking.successor.empirical_harvest.v1", "rows": [
        {"status": "complete", "branch": "codec_control", "nominal_target_bpw": 4.0,
         "wall_seconds": 7600, "geometry": {"stored_parameters": 7_600_000_000}}]}
    prof = harvest_derived_profile(hv)
    if not prof["available"] or "codec_control" not in prof["per_branch_sec_per_billion"]:
        raise EcoError("harvest-derived profile failed")
    return {"ok": True, "read_only_inventory": True, "cpu_logical": inv["cpu_logical"],
            "ram_gib": inv["ram_gib"], "free_disk_gb": inv["free_disk_gb"],
            "active_benchmark_deferred": True, "atlas_sha256": atlas["atlas_sha256"]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="M3 Ultra resource atlas (non-competing).")
    ap.add_argument("--campaign-root", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True)); sys.exit(0)
    harvest = None
    if args.campaign_root:
        import succ_harvest
        harvest = succ_harvest.harvest(args.campaign_root)
    print(json.dumps(build_atlas(harvest), indent=2, sort_keys=True))

#!/usr/bin/env python3
"""S004 §4 rung instrument.

Place any speed claim on the measured-roof ladder from (bytes/token, ns/token)
plus optional measured facets. Does not run GPU work. The 411.51 GB/s control
is historical and Q80-specific. Qwen3.8 uses the landed, provisional grouped-Q4
GEMV measurements in ``HONEST_ROOF_WEIGHT_ADDRESSING.json``; that receipt did
not remeasure a current complete-token wall.

    # with repo root or tools/ on sys.path:
    from ascent.roof_rungs import place, today_table, judge_physical_limit_claim

    place(2_217_278_160, 1_170_679_064)

Rungs (complete-token wall):
    A  <=20 ms / >=50 TPS
    B  <=10 ms / >=100 TPS
    C  <=5 ms  / >=200 TPS
    D  continue toward the measured honest decode roof

``fs_per_weight_served`` is an AMORTIZED THROUGHPUT metric under concurrency.
It is never physical femtosecond latency.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "hawking.ascension.roof_and_rungs.v1"
DATE = "2026-08-16"

REPO_ROOT = Path(__file__).resolve().parents[2]
BANDWIDTH_RECEIPT = (
    REPO_ROOT / "receipts" / "ascent-2026-08-16" / "Q80_DECODE_SHAPE_BANDWIDTH.json"
)
QWEN38_ROOF_RECEIPT = (
    REPO_ROOT
    / "receipts"
    / "ascent-2026-08-16"
    / "HONEST_ROOF_WEIGHT_ADDRESSING.json"
)

# Historical Q80 control from Q80_DECODE_SHAPE_BANDWIDTH.json.
# Keep the old export as a compatibility alias, but never use it as Qwen3.8's roof.
Q80_HISTORICAL_UNIQUE_ONCE_GB_S = 411.51358589633037
HONEST_DECODE_CEILING_GB_S = Q80_HISTORICAL_UNIQUE_ONCE_GB_S
UNIQUE_ONCE_1024MIB_GB_S = 301.63405407683126
UNIQUE_ONCE_256MIB_GB_S = 410.5302328426687
REUSE_BAND_GB_S = (535.8823163028844, 637.4964044632153)
PUBLISHED_PEAK_GB_S = 819.0
FULL_OCCUPANCY_THREADS = 256 * 60  # 15360

# Landed Qwen3.8 grouped-Q4 measurements. These are provisional: the GPU lane
# was protected, but the host was CPU-contended, and current TOKEN_NS was not
# rerun. The sealed 639.252 GB/s attribution is cited from the older ledger.
QWEN38_DEFENDED_GEMV_BYTES = 13_611_663_360
QWEN38_SINGLE_GEMV_ADDR_GB_S = 699.5736545106142
QWEN38_SEALED_WEIGHT_ADDRESSING_GB_S = 639.2522348492898
QWEN38_CATALOG_ADDR_GB_S = 530.6544688491846
QWEN38_CATALOG_FULL_GB_S = 505.8100047843556
QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR = (
    QWEN38_SEALED_WEIGHT_ADDRESSING_GB_S / QWEN38_SINGLE_GEMV_ADDR_GB_S
)
QWEN38_PROVISIONAL_CAVEAT = (
    "PROVISIONAL GPU_PROTECTED_CPU_CONTENDED: 699.574 GB/s is an isolated "
    "single-GEMV address roof; 639.252 GB/s cites the sealed historical "
    "weight-addressing attribution; the 401-GEMV catalog measured 530.654 GB/s "
    "address and 505.810 GB/s full. Current complete-token TOKEN_NS/TPS was not "
    "rerun. The catalog/single-GEMV gap leaves dispatch and kernel topology "
    "headroom open."
)

# Dispatch / CB control, same receipt.
ONE_NOP_CB_GPU_NS = 3334
ONE_NOP_CB_HOST_NS = 192083
NOPS_1155_ONE_CB_GPU_NS = 1_483_875
NOPS_1155_COUNT = 1155
CBS_98_X_12_NOPS_HOST_MINUS_GPU_NS = 20_079_956
CBS_98_COUNT = 98

# 1 / 819e9 s/byte * 1e15 fs/s * (1/8 byte at 1 BPW)
FS_PER_WEIGHT_AT_1BPW_PUBLISHED_PEAK = 152.62515262515264
AMORTIZED_CAVEAT = (
    "fs_per_weight_served is an amortized throughput metric under concurrency, "
    "NOT physical femtosecond latency. A single weight's DRAM round trip is "
    "~100 ns. Femtoseconds appear only because thousands of weights are in "
    "flight at once."
)
FS_FIELD = "fs_per_weight_served (amortized throughput-derived, NOT latency)"

RUNG_A_NS = 20_000_000
RUNG_B_NS = 10_000_000
RUNG_C_NS = 5_000_000
RUNG_A_TPS = 50.0
RUNG_B_TPS = 100.0
RUNG_C_TPS = 200.0

# A physical-limit claim needs the named resource actually near saturation.
SATURATION_FRACTION = 0.90
ROOF_ATTAINED_FRACTION = 0.95

_NO_FURTHER_OPT = (
    "no further optimization is obvious",
    "no further optimisation is obvious",
    "nothing left to optimize",
    "nothing left to optimise",
    "nothing left to target here",
)

Q80_CLAIM_BOUNDARY = {
    "scope": "Q80 historical control only",
    "q80_historical_unique_once_gb_s": Q80_HISTORICAL_UNIQUE_ONCE_GB_S,
    "unique_once_1024mib_gb_s": UNIQUE_ONCE_1024MIB_GB_S,
    "reuse_band_gb_s": list(REUSE_BAND_GB_S),
    "reuse_band_is_cache_resident_not_decode_ceiling": True,
    "published_peak_819_was_not_achieved": True,
    "what_governs_decode": "REUSE vs NO-REUSE, not gather vs sequential. Each weight is read once per token.",
    "gather_vs_sequential_is_the_wrong_axis": True,
    "fs_per_weight_served_is_not_latency": True,
    "physical_limit_requires_named_saturated_resource": True,
    "no_further_optimization_is_obvious_is_not_evidence": True,
}

QWEN38_CLAIM_BOUNDARY = {
    "scope": "Qwen3.8 provisional grouped-Q4 roof only",
    "receipt": str(QWEN38_ROOF_RECEIPT.relative_to(REPO_ROOT)),
    "defended_gemv_bytes": QWEN38_DEFENDED_GEMV_BYTES,
    "single_gemv_addr_gb_s": QWEN38_SINGLE_GEMV_ADDR_GB_S,
    "sealed_weight_addressing_gb_s": QWEN38_SEALED_WEIGHT_ADDRESSING_GB_S,
    "sealed_over_single_gemv_addr": QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR,
    "catalog_addr_gb_s": QWEN38_CATALOG_ADDR_GB_S,
    "catalog_full_gb_s": QWEN38_CATALOG_FULL_GB_S,
    "timing_label": "GPU_PROTECTED_CPU_CONTENDED",
    "provisional": True,
    "current_token_ns": None,
    "current_tps": None,
    "kernel_and_dispatch_topology_headroom_closed": False,
    "caveat": QWEN38_PROVISIONAL_CAVEAT,
}

# Compatibility export for callers that use the generic/Q80 ladder.
CLAIM_BOUNDARY = Q80_CLAIM_BOUNDARY


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def _optional_finite(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    return _finite(name, value)


def _optional_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def gb_s(bytes_moved: float, ns: float) -> float | None:
    """GB/s using the sealed control's units (bytes / ns)."""
    if ns <= 0.0:
        return None
    return bytes_moved / ns


def _is_qwen38(model: str | None) -> bool:
    return str(model or "").strip().lower() in {"qwen38", "qwen3.8", "qwen3_8"}


def _roof_gb_s(model: str | None) -> float:
    if _is_qwen38(model):
        return QWEN38_SINGLE_GEMV_ADDR_GB_S
    return Q80_HISTORICAL_UNIQUE_ONCE_GB_S


def load_only_floor_ns(
    bytes_per_token: float,
    ceiling_gb_s: float = Q80_HISTORICAL_UNIQUE_ONCE_GB_S,
) -> float:
    return bytes_per_token / ceiling_gb_s


def roof_tok_s(
    bytes_per_token: float,
    ceiling_gb_s: float = Q80_HISTORICAL_UNIQUE_ONCE_GB_S,
) -> float:
    if bytes_per_token <= 0.0:
        return 0.0
    return 1.0e9 / load_only_floor_ns(bytes_per_token, ceiling_gb_s)


def rung_byte_budget(
    max_ns: float,
    ceiling_gb_s: float = Q80_HISTORICAL_UNIQUE_ONCE_GB_S,
) -> float:
    """Largest bytes/token that can hit ``max_ns`` at the selected roof."""
    return ceiling_gb_s * max_ns


def classify_rung(ns_per_token: float, fraction_of_roof: float | None) -> dict[str, Any]:
    tps = 1.0e9 / ns_per_token if ns_per_token > 0.0 else 0.0
    cleared: list[str] = []
    if ns_per_token <= RUNG_A_NS and tps >= RUNG_A_TPS:
        cleared.append("A")
    if ns_per_token <= RUNG_B_NS and tps >= RUNG_B_TPS:
        cleared.append("B")
    if ns_per_token <= RUNG_C_NS and tps >= RUNG_C_TPS:
        cleared.append("C")

    if "C" in cleared:
        at_roof = fraction_of_roof is not None and fraction_of_roof >= ROOF_ATTAINED_FRACTION
        current = "D_at_roof" if at_roof else "D"
        next_rung = None if at_roof else "D"
    elif "B" in cleared:
        current, next_rung = "B", "C"
    elif "A" in cleared:
        current, next_rung = "A", "B"
    else:
        current, next_rung = "below_A", "A"

    return {
        "ms_per_token": ns_per_token / 1.0e6,
        "tps": tps,
        "cleared": cleared,
        "current_rung": current,
        "next_rung": next_rung,
        "definitions": {
            "A": {"max_ms": 20.0, "min_tps": RUNG_A_TPS},
            "B": {"max_ms": 10.0, "min_tps": RUNG_B_TPS},
            "C": {"max_ms": 5.0, "min_tps": RUNG_C_TPS},
            "D": "continue toward the measured honest decode roof",
        },
    }


def highest_rung_reachable(
    bytes_per_token: float,
    ceiling_gb_s: float = Q80_HISTORICAL_UNIQUE_ONCE_GB_S,
) -> dict[str, Any]:
    roof = roof_tok_s(bytes_per_token, ceiling_gb_s)
    reachable: list[str] = []
    if roof >= RUNG_A_TPS and bytes_per_token <= rung_byte_budget(RUNG_A_NS, ceiling_gb_s):
        reachable.append("A")
    if roof >= RUNG_B_TPS and bytes_per_token <= rung_byte_budget(RUNG_B_NS, ceiling_gb_s):
        reachable.append("B")
    if roof >= RUNG_C_TPS and bytes_per_token <= rung_byte_budget(RUNG_C_NS, ceiling_gb_s):
        reachable.append("C")
    if reachable:
        highest = reachable[-1]
        blocked_next = {"A": "B", "B": "C", "C": "D"}.get(highest)
    else:
        highest = None
        blocked_next = "A"
    return {
        "roof_tok_s": roof,
        "reachable_at_current_bytes": reachable,
        "highest_rung_reachable_at_current_bytes": highest,
        "next_rung_blocked_by_bytes": blocked_next if highest != "C" else None,
        "byte_budgets": {
            "A_max_bytes": rung_byte_budget(RUNG_A_NS, ceiling_gb_s),
            "B_max_bytes": rung_byte_budget(RUNG_B_NS, ceiling_gb_s),
            "C_max_bytes": rung_byte_budget(RUNG_C_NS, ceiling_gb_s),
        },
        "note": (
            "These budgets assume 100% of the selected roof "
            f"({ceiling_gb_s:.2f} GB/s) and zero host tax. "
            "Host work can only make them stricter."
        ),
    }


def judge_physical_limit_claim(
    *,
    saturated_resource: str | None,
    evidence: str | None,
    achieved: float | None = None,
    ceiling: float | None = None,
    claim_text: str | None = None,
    cites_reuse_band_as_decode_ceiling: bool = False,
    cites_published_819_as_decode_ceiling: bool = False,
) -> dict[str, Any]:
    """A physical-limit claim requires a named resource actually at saturation."""
    failures: list[str] = []
    text = (claim_text or "").lower()
    if any(phrase in text for phrase in _NO_FURTHER_OPT):
        failures.append(
            "'No further optimization is obvious' is not evidence of a physical limit."
        )
    if cites_reuse_band_as_decode_ceiling:
        failures.append(
            "The 535.9-637.5 GB/s reuse band is cache-resident and is not a decode ceiling."
        )
    if cites_published_819_as_decode_ceiling:
        failures.append(
            "Published 819 GB/s was not achieved in the decode-shape control. "
            f"The Q80 historical unique-once control is "
            f"{Q80_HISTORICAL_UNIQUE_ONCE_GB_S:.2f} GB/s; a model/access-specific "
            "roof is still required."
        )
    if not saturated_resource or not str(saturated_resource).strip():
        failures.append("No hardware resource named as saturated.")
    if not evidence or not str(evidence).strip():
        failures.append("No evidence that the named resource is at saturation.")
    fraction = None
    if achieved is not None and ceiling is not None and ceiling > 0.0:
        fraction = achieved / ceiling
        if fraction < SATURATION_FRACTION:
            failures.append(
                f"Named resource is at {fraction:.4f} of its ceiling; "
                f"saturation requires >= {SATURATION_FRACTION:.2f}."
            )
    verdict = "FAIL" if failures else "PASS"
    return {
        "verdict": verdict,
        "saturated_resource": saturated_resource,
        "evidence": evidence,
        "achieved_over_ceiling": fraction,
        "failures": failures,
        "bar": (
            "A physical-limit claim REQUIRES naming the hardware resource "
            "actually at saturation with evidence. 'No further optimization "
            "is obvious' is not evidence."
        ),
    }


def flag_fs_latency_language(text: str) -> dict[str, Any]:
    """Reject copy that treats fs_per_weight_served as physical latency."""
    lowered = text.lower()
    hits: list[str] = []
    if "physical femtosecond latency" in lowered:
        hits.append("physical femtosecond latency")
    if "femtosecond latency" in lowered and "not" not in lowered:
        hits.append("femtosecond latency without a negation")
    if "per-flop time is femtoseconds" in lowered:
        hits.append("per-FLOP time is femtoseconds (amortization presented as latency)")
    if "sub-nanosecond latency" in lowered and "already true" in lowered:
        hits.append("sub-nanosecond latency claimed true via amortized concurrency")
    return {
        "flagged": bool(hits),
        "hits": hits,
        "required_label": FS_FIELD,
        "caveat": AMORTIZED_CAVEAT,
    }


def place(
    bytes_per_token: float,
    ns_per_token: float,
    *,
    active_weights_per_token: float | None = None,
    gpu_ns_per_token: float | None = None,
    wait_minus_gpu_ns: float | None = None,
    command_buffers: int | None = None,
    dispatches: int | None = None,
    occupancy_threads: float | None = None,
    flops_per_weight: float = 2.0,
    reconstruction_ns: float | None = None,
    model: str | None = None,
    artifact: str | None = None,
    sources: Mapping[str, Any] | None = None,
    measurement_label: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute the S004 §4 ladder fields from (bytes/token, ns/token)."""
    bytes_tok = _finite("bytes_per_token", bytes_per_token)
    ns_tok = _finite("ns_per_token", ns_per_token)
    weights = _optional_finite("active_weights_per_token", active_weights_per_token)
    gpu_ns = _optional_finite("gpu_ns_per_token", gpu_ns_per_token)
    wmg_ns = _optional_finite("wait_minus_gpu_ns", wait_minus_gpu_ns)
    n_cb = _optional_int("command_buffers", command_buffers)
    n_disp = _optional_int("dispatches", dispatches)
    occ_threads = _optional_finite("occupancy_threads", occupancy_threads)
    recon_ns = _optional_finite("reconstruction_ns", reconstruction_ns)
    flops = _finite("flops_per_weight", flops_per_weight)
    selected_roof_gb_s = _roof_gb_s(model)
    qwen38_profile = _is_qwen38(model)

    wall_bw = gb_s(bytes_tok, ns_tok)
    gpu_bw = gb_s(bytes_tok, gpu_ns) if gpu_ns is not None else None
    kernel_bw = gpu_bw if gpu_bw is not None else wall_bw
    floor_ns = load_only_floor_ns(bytes_tok, selected_roof_gb_s)
    roof = roof_tok_s(bytes_tok, selected_roof_gb_s)
    frac_wall = None if wall_bw is None else wall_bw / selected_roof_gb_s
    frac_gpu = None if gpu_bw is None else gpu_bw / selected_roof_gb_s
    frac_kernel = frac_gpu if frac_gpu is not None else frac_wall

    if weights is not None and weights > 0.0:
        bpw = bytes_tok * 8.0 / weights
        intensity = flops * weights / bytes_tok if bytes_tok > 0.0 else None
        fs_wall = ns_tok * 1.0e6 / weights
        fs_gpu = None if gpu_ns is None else gpu_ns * 1.0e6 / weights
        fs_roof = floor_ns * 1.0e6 / weights
    else:
        bpw = None
        intensity = None
        fs_wall = None
        fs_gpu = None
        fs_roof = None

    launch_occ = None
    if occ_threads is not None:
        launch_occ = occ_threads / FULL_OCCUPANCY_THREADS

    dispatch = {
        "one_nop_cb_gpu_ns": ONE_NOP_CB_GPU_NS,
        "one_nop_cb_host_ns": ONE_NOP_CB_HOST_NS,
        "fused_ns_per_dispatch": NOPS_1155_ONE_CB_GPU_NS / NOPS_1155_COUNT,
        "host_tax_per_serial_cb_ns": CBS_98_X_12_NOPS_HOST_MINUS_GPU_NS / CBS_98_COUNT,
        "source": str(BANDWIDTH_RECEIPT.relative_to(REPO_ROOT)),
        "dispatches": n_disp,
        "command_buffers": n_cb,
        "serial_dispatch_gpu_floor_ns": None
        if n_disp is None
        else ONE_NOP_CB_GPU_NS * n_disp,
        "fused_dispatch_gpu_floor_ns": None
        if n_disp is None
        else (NOPS_1155_ONE_CB_GPU_NS / NOPS_1155_COUNT) * n_disp,
        "serial_cb_host_floor_ns": None
        if n_cb is None
        else (CBS_98_X_12_NOPS_HOST_MINUS_GPU_NS / CBS_98_COUNT) * n_cb,
        "note": (
            "Serial CB host tax dominates fused GPU dispatch tax. "
            "1155 nops in one CB cost 1.48 ms GPU; 1155 serial CBs project to 222 ms host."
        ),
    }

    gpu_for_recon = gpu_ns if gpu_ns is not None else ns_tok
    excess_ns = gpu_for_recon - floor_ns
    reconstruction = {
        "load_only_floor_ns": floor_ns,
        "measured_gpu_or_wall_ns": gpu_for_recon,
        "excess_over_load_only_ns": excess_ns,
        "excess_is_occupancy_plus_alu_plus_reconstruction": True,
        "isolated_reconstruction_ns": recon_ns,
        "isolation_note": (
            "excess_over_load_only mixes occupancy, issue, ALU and reconstruction. "
            "A same-launch load-only probe, or a same-shape cheaper codec, is required "
            "to isolate reconstruction. Q80 mixed vs Q4 (15.2 vs 2.57 GB/s) is the "
            "existing isolation, not this difference alone."
        ),
    }

    if wmg_ns is not None:
        sync_measured = wmg_ns
        sync_note = "measured wait-minus-gpu"
    elif n_cb is not None:
        sync_measured = dispatch["serial_cb_host_floor_ns"]
        sync_note = "projected from 98-CB nop host-minus-gpu; not a live wait"
    else:
        sync_measured = None
        sync_note = "unmeasured: pass wait_minus_gpu_ns or command_buffers"
    synchronization = {
        "wait_minus_gpu_ns": wmg_ns,
        "synchronization_floor_ns": sync_measured,
        "how": sync_note,
    }

    rung = classify_rung(ns_tok, frac_wall)
    reachable = highest_rung_reachable(bytes_tok, selected_roof_gb_s)
    roof_limited = reachable["highest_rung_reachable_at_current_bytes"] is None

    if qwen38_profile:
        physical = {
            "verdict": "PROVISIONAL_NOT_PHYSICAL_LIMIT",
            "saturated_resource": "grouped-Q4 single-GEMV address stream",
            "evidence": (
                f"measurement compared with provisional {selected_roof_gb_s:.3f} GB/s "
                "single-GEMV address roof"
            ),
            "achieved_over_ceiling": frac_kernel,
            "failures": [QWEN38_PROVISIONAL_CAVEAT],
            "bar": (
                "Rerun current complete-token TOKEN_NS under clean conditions and retain "
                "the catalog/single-GEMV distinction before a physical-limit claim."
            ),
        }
    else:
        physical = judge_physical_limit_claim(
            saturated_resource=(
                "Q80 historical unique-once decode memory bandwidth"
                if frac_kernel is not None and frac_kernel >= SATURATION_FRACTION
                else None
            ),
            evidence=(
                f"kernel {frac_kernel:.4f} of Q80 historical control "
                f"{selected_roof_gb_s:.2f} GB/s"
                if frac_kernel is not None and frac_kernel >= SATURATION_FRACTION
                else None
            ),
            achieved=kernel_bw,
            ceiling=selected_roof_gb_s,
        )

    row = {
        "schema": SCHEMA,
        "model": model,
        "artifact": artifact,
        "measurement_label": measurement_label,
        "sources": dict(sources or {}),
        "bytes_per_token": bytes_tok,
        "ns_per_token": ns_tok,
        "gpu_ns_per_token": gpu_ns,
        "active_weights_per_token": weights,
        "active_decode_bpw": bpw,
        "measured_memory_bandwidth_gb_s": wall_bw,
        "measured_gpu_bandwidth_gb_s": gpu_bw,
        "arithmetic_intensity_flop_per_byte": intensity,
        "arithmetic_intensity_note": (
            f"{flops:g} flop/weight (FMA=2) / bytes_per_weight. "
            "Batch=1 decode reads each weight once; there is no reuse to amortize the load."
        ),
        "gpu_occupancy": (
            {
                "fraction_of_selected_roof": frac_kernel,
                "fraction_of_single_gemv_addr_roof": frac_kernel,
                "selected_roof_gb_s": selected_roof_gb_s,
                "launch_occupancy_vs_15360": launch_occ,
                "full_occupancy_threads": FULL_OCCUPANCY_THREADS,
                "note": (
                    "For Qwen3.8 this ratio uses the provisional isolated single-GEMV "
                    "address roof, not the Q80 control. It cannot close catalog/topology "
                    "headroom or substitute for a current complete-token measurement."
                ),
            }
            if qwen38_profile
            else {
                "fraction_of_honest_decode_ceiling": frac_kernel,
                "fraction_of_q80_historical_control": frac_kernel,
                "selected_roof_gb_s": selected_roof_gb_s,
                "launch_occupancy_vs_15360": launch_occ,
                "full_occupancy_threads": FULL_OCCUPANCY_THREADS,
                "note": (
                    "Memory-system occupancy is achieved_gb_s / Q80 historical "
                    "unique-once control. Launch occupancy is threads / 15360 from "
                    "that Q80 decode-shape control."
                ),
            }
        ),
        "dispatch_floor": dispatch,
        "reconstruction_cost": reconstruction,
        "synchronization_floor": synchronization,
        "roof_tok_s": roof,
        "fraction_of_roof": frac_wall,
        "fraction_of_roof_gpu": frac_gpu,
        FS_FIELD: fs_wall,
        "fs_per_weight_served_gpu": fs_gpu,
        "fs_per_weight_at_honest_roof": fs_roof,
        "fs_honesty": {
            "not_latency": True,
            "field_label": FS_FIELD,
            "caveat": AMORTIZED_CAVEAT,
        },
        "rung": rung,
        "reachable_at_current_bytes": reachable,
        "roof_limited_below_next_rung": roof_limited,
        "may_claim_physical_limit": physical["verdict"] == "PASS" and not qwen38_profile,
        "physical_limit": physical,
        "hardware": (
            {
                "roof_profile": "qwen38_provisional_single_gemv_addr",
                "single_gemv_addr_gb_s": QWEN38_SINGLE_GEMV_ADDR_GB_S,
                "sealed_weight_addressing_gb_s": QWEN38_SEALED_WEIGHT_ADDRESSING_GB_S,
                "sealed_over_single_gemv_addr": QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR,
                "catalog_addr_gb_s": QWEN38_CATALOG_ADDR_GB_S,
                "catalog_full_gb_s": QWEN38_CATALOG_FULL_GB_S,
                "provisional": True,
                "cpu_contended": True,
                "current_token_ns": None,
                "control_receipt": str(QWEN38_ROOF_RECEIPT.relative_to(REPO_ROOT)),
                "caveat": QWEN38_PROVISIONAL_CAVEAT,
            }
            if qwen38_profile
            else {
                "roof_profile": "q80_historical_unique_once_control",
                "q80_historical_unique_once_gb_s": Q80_HISTORICAL_UNIQUE_ONCE_GB_S,
                "unique_once_1024mib_gb_s": UNIQUE_ONCE_1024MIB_GB_S,
                "reuse_band_gb_s": list(REUSE_BAND_GB_S),
                "reuse_band_is_not_the_decode_ceiling": True,
                "published_peak_gb_s": PUBLISHED_PEAK_GB_S,
                "control_receipt": str(BANDWIDTH_RECEIPT.relative_to(REPO_ROOT)),
            }
        ),
        "claim_boundary": QWEN38_CLAIM_BOUNDARY if qwen38_profile else Q80_CLAIM_BOUNDARY,
    }
    if extra:
        row["extra"] = dict(extra)
    return row


# ---------------------------------------------------------------------------
# Today's sealed inputs. Only the inputs live here; every derived field goes
# through place() so the table cannot drift from the function.
# ---------------------------------------------------------------------------

TODAY: dict[str, dict[str, Any]] = {
    "q80_mixed": {
        "bytes_per_token": 2_217_278_160,
        "ns_per_token": 1_170_679_064,
        "active_weights_per_token": 3_562_274_816,
        "gpu_ns_per_token": 863_388_504,
        "wait_minus_gpu_ns": 123_530_935,
        "command_buffers": 337,
        "dispatches": 2893,
        "model": "q80_mixed",
        "artifact": "mixed-1p5-v1 (complete physical BPW 1.4444457; active decode BPW ~4.980)",
        "measurement_label": "DIRTY_ENGINEERING",
        "sources": {
            "bytes": "Q80_DECODE_SHAPE_BANDWIDTH.json byte_budget.mixed_1p5_v1.weight_bytes_per_token",
            "wall_gpu_wmg_cb_disp": (
                "q80-host-facets/SUMMARY.json both-arm pair medians; "
                "Q80_MIXED_RECONSTRUCTION_WALL.json (1170.7 ms / 863.4 ms / 2.57 GB/s)"
            ),
            "weights": "PHYSICAL_FLOOR.json active_params_per_token / served_weight.rs",
        },
        "extra": {
            "storage_complete_bpw": 1.4444457,
            "gpu_matvec_is_74pct_of_token": True,
            "established_gpu_matvec_gb_s": 2.57,
            "established_pct_of_honest_ceiling": 0.62,
            "established_factor_off": 160.0,
            "q4_same_shape_gb_s": 15.2,
            "mixed_vs_q4_per_byte_slowdown": 5.9,
            "dominant_term": "reconstruction cost, not bytes moved",
        },
    },
    "qwen38": {
        "model": "qwen38",
        "artifact": "qwen38-27b grouped-Q4 GEMV genome",
        "measurement_label": "PROVISIONAL_ROOF_ONLY_NO_CURRENT_TOKEN_NS",
        "defended_gemv_payload_bytes": QWEN38_DEFENDED_GEMV_BYTES,
        "current_token_ns": None,
        "current_tps": None,
        "sources": {
            "roof": str(QWEN38_ROOF_RECEIPT.relative_to(REPO_ROOT)),
            "sealed_attribution": (
                "HONEST_ROOF_WEIGHT_ADDRESSING.json "
                "sealed_ledger_cited_not_rerun / denominator_correction"
            ),
        },
        "extra": {
            "single_gemv_addr_gb_s": QWEN38_SINGLE_GEMV_ADDR_GB_S,
            "sealed_weight_addressing_gb_s": QWEN38_SEALED_WEIGHT_ADDRESSING_GB_S,
            "sealed_over_single_gemv_addr": QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR,
            "catalog_addr_gb_s": QWEN38_CATALOG_ADDR_GB_S,
            "catalog_full_gb_s": QWEN38_CATALOG_FULL_GB_S,
            "timing_label": "GPU_PROTECTED_CPU_CONTENDED",
            "provisional": True,
            "caveat": QWEN38_PROVISIONAL_CAVEAT,
        },
    },
    "dsv4f": {
        "bytes_per_token": 5_857_237_264,
        "ns_per_token": 1_037_764_208,
        "active_weights_per_token": 12_748_587_008,
        "gpu_ns_per_token": 399_023_000,
        "wait_minus_gpu_ns": None,
        "command_buffers": 137,
        "dispatches": 1857,
        "model": "dsv4f",
        "artifact": "full-43-layer-stream.gravity (TOKEN_NS billed 3.676 active BPW)",
        "measurement_label": "DIRTY_ENGINEERING",
        "sources": {
            "bytes_weights": "TOKEN_NS_DSV4F.json served_weight",
            "body_gpu_cb_disp": "DSV4F_HOST_WALL_BASELINE.json authority (warm R2-R6 median)",
            "wait_minus_gpu": (
                "NOT wall-minus-gpu: metal.wait 244.4 ms < metal.gpu 399.0 ms "
                "because GPU overlaps host I/O. Sync floor is the CB-nop projection."
            ),
        },
        "extra": {
            "host_exclusive_ns": 564_853_792,
            "metal_wait_ns": 244_428_000,
            "regime": "IO_BOUND on the token wall; GPU is 38.7% of body",
            "hc_sha": "c94da765c4bbf795b598d96209cd80821e5a81ab97a8712586f54b8c8b612597",
        },
    },
}


def _qwen38_roof_only_today() -> dict[str, Any]:
    """Report the landed roof without manufacturing a current token wall."""
    spec = TODAY["qwen38"]
    hardware = {
        "roof_profile": "qwen38_provisional_single_gemv_addr",
        "defended_gemv_payload_bytes": QWEN38_DEFENDED_GEMV_BYTES,
        "single_gemv_addr_gb_s": QWEN38_SINGLE_GEMV_ADDR_GB_S,
        "sealed_weight_addressing_gb_s": QWEN38_SEALED_WEIGHT_ADDRESSING_GB_S,
        "sealed_over_single_gemv_addr": QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR,
        "catalog_addr_gb_s": QWEN38_CATALOG_ADDR_GB_S,
        "catalog_full_gb_s": QWEN38_CATALOG_FULL_GB_S,
        "timing_label": "GPU_PROTECTED_CPU_CONTENDED",
        "provisional": True,
        "cpu_contended": True,
        "clean_box": False,
        "current_token_ns": None,
        "current_tps": None,
        "kernel_and_dispatch_topology_headroom_closed": False,
        "control_receipt": str(QWEN38_ROOF_RECEIPT.relative_to(REPO_ROOT)),
        "caveat": QWEN38_PROVISIONAL_CAVEAT,
    }
    return {
        "schema": SCHEMA,
        "model": spec["model"],
        "artifact": spec["artifact"],
        "measurement_label": spec["measurement_label"],
        "sources": dict(spec["sources"]),
        "bytes_per_token": None,
        "defended_gemv_payload_bytes": QWEN38_DEFENDED_GEMV_BYTES,
        "ns_per_token": None,
        "gpu_ns_per_token": None,
        "current_token_ns": None,
        "current_tps": None,
        "active_weights_per_token": None,
        "active_decode_bpw": None,
        "measured_memory_bandwidth_gb_s": None,
        "measured_gpu_bandwidth_gb_s": None,
        "arithmetic_intensity_flop_per_byte": None,
        "arithmetic_intensity_note": (
            "Not derived: the landed receipt measures grouped-Q4 GEMV roof facets, "
            "not a current complete-token run."
        ),
        "gpu_occupancy": {
            "fraction_of_selected_roof": None,
            "sealed_over_single_gemv_addr": QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR,
            "selected_roof_gb_s": QWEN38_SINGLE_GEMV_ADDR_GB_S,
            "launch_occupancy_vs_15360": None,
            "full_occupancy_threads": None,
            "note": (
                "91.38% is the sealed historical weight-addressing attribution divided "
                "by the provisional isolated single-GEMV address roof. It is not a "
                "current complete-token occupancy result."
            ),
        },
        "dispatch_floor": {
            "dispatches": None,
            "command_buffers": None,
            "serial_dispatch_gpu_floor_ns": None,
            "fused_dispatch_gpu_floor_ns": None,
            "serial_cb_host_floor_ns": None,
            "catalog_dispatches": 401,
            "note": (
                "No current token dispatch census is asserted. The roof receipt's "
                "production-shape catalog contains 401 GEMVs."
            ),
        },
        "reconstruction_cost": {
            "load_only_floor_ns": None,
            "measured_gpu_or_wall_ns": None,
            "excess_over_load_only_ns": None,
            "isolated_reconstruction_ns": None,
            "isolation_note": "Unmeasured for a current complete token.",
        },
        "synchronization_floor": {
            "wait_minus_gpu_ns": None,
            "synchronization_floor_ns": None,
            "how": "unmeasured for a current complete token",
        },
        "roof_tok_s": None,
        "fraction_of_roof": None,
        "fraction_of_roof_gpu": None,
        FS_FIELD: None,
        "fs_per_weight_served_gpu": None,
        "fs_per_weight_at_honest_roof": None,
        "fs_honesty": {
            "not_latency": True,
            "field_label": FS_FIELD,
            "caveat": AMORTIZED_CAVEAT,
        },
        "rung": {
            "current_rung": "UNMEASURED",
            "cleared_rungs": [],
            "next_rung": None,
            "ms_per_token": None,
            "tps": None,
            "fraction_of_roof": None,
            "note": "Current complete-token TOKEN_NS/TPS was not rerun.",
        },
        "reachable_at_current_bytes": {
            "roof_tok_s": None,
            "reachable_at_current_bytes": [],
            "highest_rung_reachable_at_current_bytes": None,
            "next_rung_blocked_by_bytes": None,
            "byte_budgets": None,
            "note": (
                "Unknown: a GEMV payload roof does not establish a current complete-token "
                "wall or all bytes moved by the token."
            ),
        },
        "roof_limited_below_next_rung": None,
        "may_claim_physical_limit": False,
        "physical_limit": {
            "verdict": "PROVISIONAL_NOT_PHYSICAL_LIMIT",
            "saturated_resource": (
                "grouped-Q4 single-GEMV unique codes+scales address stream"
            ),
            "evidence": (
                f"sealed attribution {QWEN38_SEALED_WEIGHT_ADDRESSING_GB_S:.3f} / "
                f"single-GEMV addr {QWEN38_SINGLE_GEMV_ADDR_GB_S:.3f} = "
                f"{100.0 * QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR:.2f}%; catalog addr/full "
                f"{QWEN38_CATALOG_ADDR_GB_S:.3f}/{QWEN38_CATALOG_FULL_GB_S:.3f} GB/s"
            ),
            "achieved_over_ceiling": QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR,
            "failures": [QWEN38_PROVISIONAL_CAVEAT],
            "bar": (
                "Rerun a current complete-token TOKEN_NS under clean conditions and "
                "retain the catalog/single-GEMV topology distinction."
            ),
        },
        "hardware": hardware,
        "claim_boundary": QWEN38_CLAIM_BOUNDARY,
        "extra": dict(spec["extra"]),
    }


def place_today(model_key: str) -> dict[str, Any]:
    if model_key == "qwen38":
        return _qwen38_roof_only_today()
    spec = TODAY[model_key]
    kwargs = {k: v for k, v in spec.items()}
    return place(**kwargs)


def today_table() -> dict[str, Any]:
    models = {key: place_today(key) for key in ("q80_mixed", "qwen38", "dsv4f")}
    return {
        "schema": SCHEMA,
        "date": DATE,
        "steer": "archived S004 section 4 (hawking-femtosecond-ascent/STEERS_ARCHIVE.md)",
        "hardware_roof": {
            "q80_historical_control": {
                "scope": "Q80 only; not a Qwen3.8 roof",
                "unique_once_512mib_gb_s": Q80_HISTORICAL_UNIQUE_ONCE_GB_S,
                "unique_once_1024mib_gb_s": UNIQUE_ONCE_1024MIB_GB_S,
                "reuse_band_gb_s": list(REUSE_BAND_GB_S),
                "reuse_band_is_not_the_decode_ceiling": True,
                "published_peak_gb_s": PUBLISHED_PEAK_GB_S,
                "control": {
                    "example": "crates/hawking-core/examples/q80_decode_shape_bandwidth.rs",
                    "branch": "grok/fs-occupancy-20260816-143029",
                    "receipt": str(BANDWIDTH_RECEIPT.relative_to(REPO_ROOT)),
                },
            },
            "qwen38_provisional": {
                "scope": "Qwen3.8 grouped-Q4 GEMV roof facets; not current TOKEN_NS",
                "defended_gemv_payload_bytes": QWEN38_DEFENDED_GEMV_BYTES,
                "single_gemv_addr_gb_s": QWEN38_SINGLE_GEMV_ADDR_GB_S,
                "sealed_weight_addressing_gb_s": QWEN38_SEALED_WEIGHT_ADDRESSING_GB_S,
                "sealed_over_single_gemv_addr": QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR,
                "catalog_addr_gb_s": QWEN38_CATALOG_ADDR_GB_S,
                "catalog_full_gb_s": QWEN38_CATALOG_FULL_GB_S,
                "timing_label": "GPU_PROTECTED_CPU_CONTENDED",
                "provisional": True,
                "current_token_ns": None,
                "current_tps": None,
                "kernel_and_dispatch_topology_headroom_closed": False,
                "receipt": str(QWEN38_ROOF_RECEIPT.relative_to(REPO_ROOT)),
                "caveat": QWEN38_PROVISIONAL_CAVEAT,
            },
            "other_established_measurements": {
                "q80_mixed_gpu_matvec_gb_s": 2.57,
                "q80_mixed_pct_of_q80_historical_control": 0.62,
                "q80_mixed_factor_off": 160.0,
            },
        },
        "claim_boundary": {
            "q80_historical": Q80_CLAIM_BOUNDARY,
            "qwen38_provisional": QWEN38_CLAIM_BOUNDARY,
        },
        "roof_receipts": {
            "q80_historical": verify_bandwidth_receipt(),
            "qwen38_provisional": verify_qwen38_roof_receipt(),
        },
        "models": models,
        "physical_limit_audit": physical_limit_audit(),
        "fs_latency_flags": fs_latency_flags(),
    }


def physical_limit_audit() -> list[dict[str, Any]]:
    """Existing claims run through the S004 saturation test."""
    rows = [
        {
            "receipt": str(QWEN38_ROOF_RECEIPT.relative_to(REPO_ROOT)),
            "claim": (
                "Qwen3.8 landed a provisional grouped-Q4 roof: sealed historical "
                f"weight addressing {QWEN38_SEALED_WEIGHT_ADDRESSING_GB_S:.3f} GB/s "
                f"is {100.0 * QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR:.2f}% of the "
                f"{QWEN38_SINGLE_GEMV_ADDR_GB_S:.3f} GB/s isolated address roof; "
                f"the 401-GEMV catalog is {QWEN38_CATALOG_ADDR_GB_S:.3f}/"
                f"{QWEN38_CATALOG_FULL_GB_S:.3f} GB/s addr/full."
            ),
            "result": {
                "verdict": "PROVISIONAL_NOT_PHYSICAL_LIMIT",
                "saturated_resource": (
                    "grouped-Q4 single-GEMV unique codes+scales address stream"
                ),
                "evidence": (
                    f"{QWEN38_SEALED_WEIGHT_ADDRESSING_GB_S:.3f} / "
                    f"{QWEN38_SINGLE_GEMV_ADDR_GB_S:.3f} = "
                    f"{100.0 * QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR:.2f}%; "
                    f"catalog addr/full {QWEN38_CATALOG_ADDR_GB_S:.3f}/"
                    f"{QWEN38_CATALOG_FULL_GB_S:.3f} GB/s"
                ),
                "achieved_over_ceiling": QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR,
                "failures": [QWEN38_PROVISIONAL_CAVEAT],
                "bar": (
                    "The isolated roof supports a genome-specific addressing saturation "
                    "claim only. It does not close catalog/dispatch topology headroom or "
                    "establish a current complete-token wall."
                ),
            },
        },
        {
            "receipt": "receipts/ascent-2026-08-16/Q80_MIXED_RECONSTRUCTION_WALL.json",
            "claim": "Does not claim a physical limit. Names reconstruction; 2.57 GB/s = 0.62% of 411.51.",
            "claims_physical_limit": False,
            "result": {
                "verdict": "PASS_NO_LIMIT_CLAIMED",
                "failures": [],
                "note": "Correctly refuses a bandwidth-limit story. 0.62% of the honest ceiling is the opposite of saturation.",
                "if_it_had_claimed_a_limit": judge_physical_limit_claim(
                    saturated_resource="decode bandwidth",
                    evidence="2.57 GB/s gpu_matvec",
                    achieved=2.57,
                    ceiling=HONEST_DECODE_CEILING_GB_S,
                ),
            },
        },
        {
            "receipt": "receipts/ascent-2026-08-16/G001_KERNEL_GAP.json",
            "claim": "Q80 is 21-27x off a no-model control; not a bandwidth-ceiling problem.",
            "claims_physical_limit": False,
            "result": {
                "verdict": "PASS_NO_LIMIT_CLAIMED",
                "failures": [],
                "note": "Correctly names the gap as kernel-shaped, not ceiling-shaped.",
                "if_it_had_claimed_a_limit": judge_physical_limit_claim(
                    saturated_resource="decode bandwidth",
                    evidence="15.2 GB/s on the Q4 vehicle",
                    achieved=15.2,
                    ceiling=HONEST_DECODE_CEILING_GB_S,
                ),
            },
        },
        {
            "receipt": "receipts/ascent-2026-08-16/TERMINAL_TARGET.json THE_SINGLE_SHARED_BLOCKER",
            "claim": "All three models share one blocker: ~0.4% of the 560-647 GB/s control.",
            "result": judge_physical_limit_claim(
                saturated_resource="560-647 GB/s reuse-band control",
                evidence="packed matvecs ~2.5 GB/s",
                achieved=2.5,
                ceiling=560.0,
                cites_reuse_band_as_decode_ceiling=True,
                claim_text="until packed matvec occupancy is solved NONE of these ceilings are reachable",
            ),
        },
        {
            "receipt": "receipts/ascent-2026-08-16/TERMINAL_TARGET.json machine_reference",
            "claim": "Use measured control 560-647 GB/s, not 819, when judging efficiency.",
            "result": judge_physical_limit_claim(
                saturated_resource="reuse-band DRAM control",
                evidence="sequential ~560, conflict ~647",
                achieved=560.0,
                ceiling=647.0,
                cites_reuse_band_as_decode_ceiling=True,
            ),
        },
        {
            "receipt": "receipts/ascent-2026-08-16/PHYSICAL_FLOOR.json floors.q80_mixed",
            "claim": "Q80 mixed DRAM floor 757 us / 1321 tok/s is the physical limit.",
            "result": judge_physical_limit_claim(
                saturated_resource="published 819 GB/s peak",
                evidence="bytes/token / 819e9",
                cites_published_819_as_decode_ceiling=True,
            ),
        },
        {
            "receipt": "receipts/ascent-2026-08-16/PHYSICAL_FLOOR.json answer_per_operation",
            "claim": "Sub-nanosecond latency is already true; per-FLOP time is femtoseconds.",
            "result": judge_physical_limit_claim(
                saturated_resource="GPU clock (714 ps)",
                evidence="clock period; then amortized across 60 cores",
                claim_text="across 60 cores the effective per-FLOP time is femtoseconds. There is nothing left to target here.",
            ),
        },
        {
            "receipt": "receipts/ascent-2026-08-16/PHYSICAL_FLOOR.json answer_per_token",
            "claim": "A 1 ns token is physically impossible by ~6 orders of magnitude.",
            "result": judge_physical_limit_claim(
                saturated_resource="unique-once decode memory bandwidth",
                evidence=(
                    "even at 411.51 GB/s, mixed-1p5 2.218 GB is 5.39 ms; "
                    "a 1 ns token would need petabytes/s"
                ),
                achieved=HONEST_DECODE_CEILING_GB_S,
                ceiling=HONEST_DECODE_CEILING_GB_S,
            ),
        },
        {
            "receipt": "receipts/ascent-2026-08-16/QWEN38_ARCH_CENSUS.json HARD_CONSEQUENCE_1",
            "claim": "100 TPS is physically impossible at 3.0 BPW; ceiling 92.5 tok/s.",
            "result": judge_physical_limit_claim(
                saturated_resource="published 819 GB/s peak",
                evidence="23.611e9 weights * 3 BPW / 8 / 819e9 = 10.81 ms",
                cites_published_819_as_decode_ceiling=True,
            ),
        },
        {
            "receipt": "GOAL.md G012 evidence (hawking-femtosecond-ascent)",
            "claim": "ALL THREE share one blocker: measured ~0.4% of the 560-647 GB/s control.",
            "result": judge_physical_limit_claim(
                saturated_resource="560-647 GB/s reuse band",
                evidence="~0.4% efficiency quoted for all three models",
                achieved=0.004 * 600.0,
                ceiling=600.0,
                cites_reuse_band_as_decode_ceiling=True,
            ),
        },
        {
            "receipt": "SUPERWAVE_STATE.md header (later corrected in-file)",
            "claim": "Measured bandwidth control band: 560-647 GB/s (68-79% of peak).",
            "result": judge_physical_limit_claim(
                saturated_resource="reuse-band DRAM control",
                evidence="earlier occupancy probes on a 64 MiB reused buffer",
                cites_reuse_band_as_decode_ceiling=True,
            ),
        },
        {
            "receipt": "QWEN38_ACTIVE_BUDGET_MEASURED.json CORRECTION_TO_MY_OWN_CLAIM (superseded)",
            "claim": (
                "The older Qwen3.8 current-token ceiling claim was unmeasured against "
                "its actual grouped-Q4 access pattern."
            ),
            "result": {
                "verdict": "REFUTED_BY_LANDED_PROVISIONAL_ROOF",
                "note": (
                    "HONEST_ROOF_WEIGHT_ADDRESSING.json measured the actual grouped-Q4 "
                    "address genome and refuted use of the Q80 historical control for "
                    "Qwen3.8. Its CPU-contended roof remains provisional, its token wall "
                    "was cited rather than rerun, and the catalog gap leaves topology "
                    "headroom open."
                ),
            },
        },
    ]
    for row in rows:
        if "result" in row and isinstance(row["result"], dict) and "verdict" in row["result"]:
            row["verdict"] = row["result"]["verdict"]
    return rows


def fs_latency_flags() -> list[dict[str, Any]]:
    return [
        {
            "receipt": "receipts/ascent-2026-08-16/PHYSICAL_FLOOR.json answer_per_operation",
            "flag": flag_fs_latency_language(
                "Is sub-nanosecond latency a physically targetable goal? "
                "ALREADY TRUE. across 60 cores the effective per-FLOP time is femtoseconds."
            ),
            "note": (
                "Conflates instruction-issue time (714 ps, real) with amortized "
                "per-FLOP / per-weight femtoseconds (concurrency, not latency)."
            ),
        },
        {
            "receipt": "crates/hawking-core/src/token_ns/served_weight.rs",
            "flag": flag_fs_latency_language(AMORTIZED_CAVEAT),
            "note": "Correctly labeled. Do not regress this.",
        },
        {
            "receipt": "TOKEN_NS_Q80.json / TOKEN_NS_DSV4F.json / frontier-fs-per-weight.json",
            "flag": {"flagged": False, "hits": [], "required_label": FS_FIELD},
            "note": "Carry the amortized caveat. No latency claim.",
        },
    ]


def markdown_table(doc: Mapping[str, Any]) -> str:
    models: Mapping[str, Any] = doc["models"]
    headers = [
        "model",
        "bytes/token",
        "ms/token",
        "TPS",
        "wall GB/s",
        "GPU GB/s",
        "AI flop/B",
        "occ vs roof",
        "dispatch floor (serial CB host, ms)",
        "recon excess (ms)",
        "sync floor (ms)",
        "roof tok/s",
        "frac roof (wall)",
        "fs/weight served",
        "current rung",
        "highest rung at these bytes",
    ]
    lines = [
        "# Roof and rungs — S004 §4",
        "",
        "Instrument: `python3 tools/ascent/roof_rungs.py --bytes <B> --ns <ns>`.",
        "Today's table: `python3 tools/ascent/roof_rungs.py --table-today`.",
        "",
        "Rungs (complete-token wall): **A** ≤20 ms / ≥50 TPS · **B** ≤10 ms / ≥100 TPS · **C** ≤5 ms / ≥200 TPS · **D** continue toward the measured roof.",
        "",
        f"Q80 historical unique-once control = **{Q80_HISTORICAL_UNIQUE_ONCE_GB_S:.2f} GB/s** "
        f"at 512 MiB (1024 MiB = {UNIQUE_ONCE_1024MIB_GB_S:.1f} GB/s); it is Q80-specific. "
        "The Q80 reuse band 535.9–637.5 GB/s is cache-resident, and published 819 GB/s "
        "was not achieved.",
        "",
        f"Qwen3.8 provisional grouped-Q4 roof: isolated single-GEMV address "
        f"**{QWEN38_SINGLE_GEMV_ADDR_GB_S:.3f} GB/s**; sealed historical weight-addressing "
        f"attribution **{QWEN38_SEALED_WEIGHT_ADDRESSING_GB_S:.3f} GB/s** "
        f"(**{100.0 * QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR:.2f}%**); 401-GEMV catalog "
        f"address/full **{QWEN38_CATALOG_ADDR_GB_S:.3f}/{QWEN38_CATALOG_FULL_GB_S:.3f} GB/s**. "
        "The run was GPU-protected but CPU-contended. Current complete-token TOKEN_NS/TPS "
        "was not rerun, so catalog/dispatch topology headroom remains open.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    order = ("q80_mixed", "qwen38", "dsv4f")
    for key in order:
        row = models[key]
        if key == "qwen38":
            cells = [
                key,
                f"{row['defended_gemv_payload_bytes'] / 1e9:.3f} GB GEMV*",
                "—",
                "—",
                "—",
                f"{row['hardware']['sealed_weight_addressing_gb_s']:.3f} sealed*",
                "—",
                f"{100.0 * row['hardware']['sealed_over_single_gemv_addr']:.2f}%*",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                row["rung"]["current_rung"],
                "UNMEASURED",
            ]
            lines.append("| " + " | ".join(cells) + " |")
            continue
        disp = row["dispatch_floor"]["serial_cb_host_floor_ns"]
        recon = row["reconstruction_cost"]["excess_over_load_only_ns"]
        sync = row["synchronization_floor"]["synchronization_floor_ns"]
        fs = row[FS_FIELD]
        cells = [
            key,
            f"{row['bytes_per_token'] / 1e9:.3f} GB",
            f"{row['rung']['ms_per_token']:.2f}",
            f"{row['rung']['tps']:.2f}",
            f"{row['measured_memory_bandwidth_gb_s']:.2f}",
            "—" if row["measured_gpu_bandwidth_gb_s"] is None else f"{row['measured_gpu_bandwidth_gb_s']:.2f}",
            "—" if row["arithmetic_intensity_flop_per_byte"] is None else f"{row['arithmetic_intensity_flop_per_byte']:.2f}",
            "—" if row["gpu_occupancy"]["fraction_of_honest_decode_ceiling"] is None else f"{100.0 * row['gpu_occupancy']['fraction_of_honest_decode_ceiling']:.2f}%",
            "—" if disp is None else f"{disp / 1e6:.1f}",
            f"{recon / 1e6:.2f}",
            "—" if sync is None else f"{sync / 1e6:.1f}",
            f"{row['roof_tok_s']:.2f}",
            f"{100.0 * row['fraction_of_roof']:.2f}%",
            "—" if fs is None else f"{fs:.1f}",
            row["rung"]["current_rung"],
            row["reachable_at_current_bytes"]["highest_rung_reachable_at_current_bytes"] or "none (roof < 50 TPS)",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            f"{FS_FIELD}. {AMORTIZED_CAVEAT}",
            "",
            "## Physical-limit audit",
            "",
            "| verdict | receipt |",
            "| --- | --- |",
        ]
    )
    for row in doc.get("physical_limit_audit", []):
        lines.append(f"| {row['verdict']} | {row['receipt']} |")
    lines.extend(
        [
            "",
            "A physical-limit claim requires naming the hardware resource actually at saturation with evidence. "
            "'No further optimization is obvious' is not evidence.",
            "",
            "## How to read the rungs",
            "",
            "- **qwen38**: roof-only and provisional. The 91.38% ratio compares a sealed "
            "historical weight-addressing attribution with the isolated single-GEMV address "
            "roof. The CPU-contended receipt has no current complete-token TOKEN_NS/TPS; "
            "the 401-GEMV catalog gap leaves dispatch/kernel topology headroom open.",
            "- **q80_mixed**: 0.62% of the Q80 historical control on GPU matvec. "
            "Reconstruction, not bandwidth, is the wall. "
            "Current bytes still physically allow rungs A and B (roof 185.6 TPS).",
            "- **dsv4f**: 3.57% of the Q80 historical control on GPU; token wall is host I/O. "
            "Current bytes allow A only (roof 70.3 TPS).",
        ]
    )
    return "\n".join(lines)


def verify_bandwidth_receipt() -> dict[str, Any]:
    if not BANDWIDTH_RECEIPT.is_file():
        return {"present": False, "path": str(BANDWIDTH_RECEIPT)}
    data = json.loads(BANDWIDTH_RECEIPT.read_text())
    ceiling = data["controls"]["honest_decode_ceiling_gbps"]
    once_1024 = data["controls"]["unique_once_sweep"]["unique_once_1024mib"]["median_gbps"]
    reuse = data["controls"]["reuse_band_gbps"]
    ok = (
        abs(ceiling - HONEST_DECODE_CEILING_GB_S) < 1e-9
        and abs(once_1024 - UNIQUE_ONCE_1024MIB_GB_S) < 1e-9
        and abs(reuse[0] - REUSE_BAND_GB_S[0]) < 1e-9
        and abs(reuse[1] - REUSE_BAND_GB_S[1]) < 1e-9
    )
    return {
        "present": True,
        "matches_sealed_constants": ok,
        "honest_decode_ceiling_gbps": ceiling,
        "unique_once_1024mib_gbps": once_1024,
        "reuse_band_gbps": reuse,
        "reuse_not_decode_ceiling": data["honest_control"]["reuse_64mib_x_4096_gbps"][
            "not_the_decode_ceiling"
        ],
    }


def verify_qwen38_roof_receipt() -> dict[str, Any]:
    """Verify the landed Qwen3.8 roof constants without treating it as TOKEN_NS."""
    if not QWEN38_ROOF_RECEIPT.is_file():
        return {"present": False, "path": str(QWEN38_ROOF_RECEIPT)}
    data = json.loads(QWEN38_ROOF_RECEIPT.read_text())
    observed = {
        "defended_gemv_payload_bytes": data["denominator_correction"][
            "correct_attribution"
        ]["bytes"],
        "single_gemv_addr_gb_s": data["verdict"][
            "measured_q4_addr_kernel_roof_gb_s"
        ],
        "sealed_weight_addressing_gb_s": data["denominator_correction"][
            "correct_attribution"
        ]["achieved_gb_s"],
        "sealed_over_single_gemv_addr": data["verdict"][
            "sealed_over_kernel_roof"
        ],
        "catalog_addr_gb_s": data["verdict"]["measured_q4_addr_catalog_gb_s"],
        "catalog_full_gb_s": data["verdict"]["production_catalog_at_13p6gb"][
            "full_gb_s"
        ],
        "timing_label": data["timing_label"],
    }
    expected = {
        "defended_gemv_payload_bytes": QWEN38_DEFENDED_GEMV_BYTES,
        "single_gemv_addr_gb_s": QWEN38_SINGLE_GEMV_ADDR_GB_S,
        "sealed_weight_addressing_gb_s": QWEN38_SEALED_WEIGHT_ADDRESSING_GB_S,
        "sealed_over_single_gemv_addr": QWEN38_SEALED_OVER_SINGLE_GEMV_ADDR,
        "catalog_addr_gb_s": QWEN38_CATALOG_ADDR_GB_S,
        "catalog_full_gb_s": QWEN38_CATALOG_FULL_GB_S,
        "timing_label": "GPU_PROTECTED_CPU_CONTENDED",
    }
    field_matches = {
        key: (
            observed[key] == value
            if isinstance(value, (str, int))
            else abs(observed[key] - value) < 1e-9
        )
        for key, value in expected.items()
    }
    cited_not_rerun = "sealed_ledger_cited_not_rerun" in data
    return {
        "present": True,
        "path": str(QWEN38_ROOF_RECEIPT),
        "matches_landed_constants": all(field_matches.values()),
        "field_matches": field_matches,
        "observed": observed,
        "provisional": True,
        "clean_box": bool(data.get("clean_box", False)),
        "cpu_contended": data.get("timing_label") == "GPU_PROTECTED_CPU_CONTENDED",
        "sealed_ledger_cited_not_rerun": cited_not_rerun,
        "current_token_ns_remeasured": False,
        "kernel_and_dispatch_topology_headroom_closed": False,
        "caveat": QWEN38_PROVISIONAL_CAVEAT,
    }


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=False)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--bytes", type=float, help="bytes/token")
    parser.add_argument("--ns", type=float, help="measured complete-token ns/token")
    parser.add_argument("--gpu-ns", type=float, default=None)
    parser.add_argument("--weights", type=float, default=None)
    parser.add_argument("--wait-minus-gpu-ns", type=float, default=None)
    parser.add_argument("--cbs", type=int, default=None)
    parser.add_argument("--dispatches", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--table-today", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.table_today:
        doc = today_table()
        doc["bandwidth_receipt"] = verify_bandwidth_receipt()
        text = _dump(doc)
        md = markdown_table(doc)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")
        if args.md_out:
            args.md_out.parent.mkdir(parents=True, exist_ok=True)
            args.md_out.write_text(md + "\n")
        if not args.out:
            sys.stdout.write(text + "\n")
        if not args.md_out:
            sys.stdout.write("\n" + md + "\n")
        return 0

    if args.bytes is None or args.ns is None:
        parser.error("provide --bytes and --ns, or --table-today")

    row = place(
        args.bytes,
        args.ns,
        active_weights_per_token=args.weights,
        gpu_ns_per_token=args.gpu_ns,
        wait_minus_gpu_ns=args.wait_minus_gpu_ns,
        command_buffers=args.cbs,
        dispatches=args.dispatches,
        model=args.model,
    )
    text = _dump(row)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

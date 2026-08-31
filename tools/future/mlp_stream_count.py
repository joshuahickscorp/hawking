#!/usr/bin/env python3
"""WHY DOES MLP'S ARM A CAP AT 497 WHILE DELTANET'S HITS 943?

receipts/future/MLP_ALU_ROOFLINE.json, same box, same method, arithmetic
stripped on both organs:

    MLP      ARM A   497.4 GB/s   38 B/iter = 2+2+2+32   (codes, scale, bias, x)
    DeltaNet ARM A   943.2 GB/s   38 B/iter = 4+2+32     (codes, scale, x)

Identical bytes per thread-iteration. Nearly 2x apart in achieved bandwidth.
That is not an arithmetic result and it is not a byte-count result. The visible
structural difference is the NUMBER OF SEPARATE MEMORY STREAMS each thread
reads. This sidecar holds 38 B/iter fixed, strips arithmetic, and varies only
how those bytes are addressed.

Pre-registered interpretation (frozen before any GPU timestamp):

    GB/s rises monotonically as streams are merged
        -> STREAM_COUNT_BOUND. The MLP's 497 is a packing property.
    the 2+2+2 and 4+2 shapes measure the same
        -> stream count is NOT the mechanism. Alignment/stride, which ran in
           the same process, then decide ALIGNMENT_BOUND vs NOT_STREAM_COUNT.
    2+2+2 differs from 4+2 but 2+4+32 tracks 2+2+2
        -> ALIGNMENT_BOUND (the 4-byte operand is the lift, not the count).
    neither
        -> MIXED. Do not force a verdict.

    python3 tools/future/mlp_stream_count.py --record
    python3 tools/future/mlp_stream_count.py --from receipts/future/_MLP_STREAM_COUNT_raw.json --record
    python3 tools/future/mlp_stream_count.py --measure --record
    python3 -m pytest tools/future/test_mlp_stream_count.py -q

evidence_class SELF_MEASURED_DIRTY. Absolute GB/s is measured-under-load.
The verdict is the back-to-back ratio. Does not change the production decode path.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402


RECEIPT = REPO / "receipts" / "future" / "MLP_STREAM_COUNT.json"
RAW_DEFAULT = REPO / "receipts" / "future" / "_MLP_STREAM_COUNT_raw.json"
SCHEMA = "hawking.future.mlp_stream_count.v1"
VERSION = 1
RECORDED_BY = "tools/future/mlp_stream_count.py"

BYTES_PER_ITER = 38
STREAM_RUNG_IDS = (
    "mlp_2_2_2_32",
    "dn_4_2_32",
    "mid_2_4_32",
    "pack_6_32",
    "pack_38",
)
# Merge ladder, decreasing stream count. mid_2_4_32 is a same-count size-swap
# control, not a merge step.
MERGE_ORDER = ("mlp_2_2_2_32", "dn_4_2_32", "pack_6_32", "pack_38")
ALIGN_IDS = ("align_2", "align_4", "align_16")
STRIDE_IDS = ("stride_contig",)

# Ratio bars, not absolute GB/s: other lanes may share the GPU.
SAME_RATIO = 1.08          # max/min <= this => "measure the same"
MONOTONE_SLACK = 0.97      # allow 3% noise against a merge-step increase
ALIGN_LIFT = 1.12          # 4- or 16-align (or contig stride) vs 2-align

VERDICT_STREAM = "STREAM_COUNT_BOUND"
VERDICT_ALIGN = "ALIGNMENT_BOUND"
VERDICT_NOT = "NOT_STREAM_COUNT"
VERDICT_MIXED = "MIXED"
VERDICTS = (VERDICT_STREAM, VERDICT_ALIGN, VERDICT_NOT, VERDICT_MIXED)

# Cited from the ALU-roofline receipt / ORGAN_BANDWIDTH / PATH_TO_71.
# This-run GB/s is measured; these are named targets, not this run.
MLP_ARM_A_GB_S = 497.4
DN_ARM_A_GB_S = 943.2
U1ALU_PRODUCTION_GB_S = 329.6
MLP_MS = 15.541
TOKEN_MS = 28.722
TOKEN_TPS = 34.82
CLEAN_GEMV_GB_S = 703.5

# Interleaved affine2 layout cost (static, not a measurement).
# Production group is 16 B codes + 2 B scale + 2 B bias = 20 B.
# 8 tiles/group × 6 B payload = 48 B; 8 B records (2 B pad) = 64 B storage.
PACKING_COST = {
    "production_bytes_per_group": 20,
    "interleaved_payload_bytes_per_group": 48,
    "interleaved_storage_bytes_per_group": 64,
    "expansion_payload": 2.4,
    "expansion_storage": 3.2,
    "when": "catalog bake, once per artifact",
    "does_not_change_decode_path": True,
    "note": (
        "Attaching scale+bias to each 2 B code tile replicates the per-group "
        "aux 8× (8 tiles/group). Tight 6 B records are 48/20 = 2.4× operand "
        "storage; 8 B aligned records are 64/20 = 3.2×. The probe loads 6 B "
        "of payload per iteration, not the pad."
    ),
}

PRE_REGISTERED = {
    "registered_before_measurement": True,
    "bytes_per_thread_iteration_held": BYTES_PER_ITER,
    "arithmetic": "stripped (XOR/add sink); this is streaming, not ALU",
    "same_ratio_bar": SAME_RATIO,
    "monotone_slack": MONOTONE_SLACK,
    "align_lift_bar": ALIGN_LIFT,
    "rules": [
        (
            "GB/s rises monotonically as streams are merged (4 -> 3 -> 2 -> 1) "
            "and 2+2+2 differs from 4+2 -> STREAM_COUNT_BOUND. The MLP's 497 "
            "is a packing property. An interleaved affine2 layout (6 B operand "
            "record) is the fix; cost is a one-time catalog bake that expands "
            "20 B/group to 48 B tight or 64 B with 8 B records."
        ),
        (
            "2+2+2 and 4+2 measure the same -> stream count is NOT the "
            "mechanism. Alignment/stride, which ran in the same process, then "
            "decide ALIGNMENT_BOUND vs NOT_STREAM_COUNT."
        ),
        (
            "2+2+2 differs from 4+2 but 2+4+32 tracks 2+2+2 (the 4-byte "
            "operand is the lift, not the count) -> ALIGNMENT_BOUND."
        ),
        "neither curve -> MIXED. Do not force a verdict.",
    ],
}

CLAIM_BOUNDARY = (
    "One representative MLP layer (gate+up+down) on sealed-3.14, "
    "SELF_MEASURED_DIRTY. GPU time is MTLCommandBuffer GPUStartTime/GPUEndTime "
    "for an isolated command buffer of three geo_tpr64_tg128 dispatches. Unique "
    "payload bytes (codes+scales+biases of the launched tensors) are the GB/s "
    "numerator on every rung, so the numerator does not move with packing; "
    "counted traffic per thread-iteration is 38 B on every rung. Arithmetic is "
    "stripped (XOR/add sink). Absolute GB/s is measured-under-load; the verdict "
    "is the back-to-back ratio of rungs in the same process. Alignment 2/4/16 "
    "and a contiguous-code stride arm ran in that same process. Token-ms "
    "numbers tagged projection are arithmetic over the measured probe "
    "(MLP 15.541 ms of a 28.722 ms token) and are not a resident measurement. "
    "Does not change the production decode path. Region granularity, catalog "
    "addressing, dispatch count and decode fusion are REFUTED elsewhere."
)

RUNG_SPEC = {
    "mlp_2_2_2_32": {
        "stream_count": 4,
        "bytes_per_stream": [2, 2, 2, 32],
        "shape": "MLP ARM A today",
    },
    "dn_4_2_32": {
        "stream_count": 3,
        "bytes_per_stream": [4, 2, 32],
        "shape": "DeltaNet ARM A",
    },
    "mid_2_4_32": {
        "stream_count": 3,
        "bytes_per_stream": [2, 4, 32],
        "shape": "size-swap of DeltaNet (between-rung / width control)",
    },
    "pack_6_32": {
        "stream_count": 2,
        "bytes_per_stream": [6, 32],
        "shape": "codes+scale+bias interleaved into one operand stream",
    },
    "pack_38": {
        "stream_count": 1,
        "bytes_per_stream": [38],
        "shape": "everything in one stream (x not reused across rows)",
    },
}


class MissingRung(Exception):
    """Raised rather than emit a verdict over an incomplete ladder."""


class ByteMismatch(Exception):
    """Raised when a rung's counted bytes per thread-iteration is not 38."""


class EmptyGpuSample(Exception):
    """Raised rather than divide by a missing GPU timestamp."""


def effective_gb_s(weight_bytes: int, gpu_ns: int) -> float:
    if gpu_ns <= 0:
        raise EmptyGpuSample("gpu_ns must be positive to form a bandwidth")
    if weight_bytes <= 0:
        raise ValueError("weight_bytes must be positive to form a bandwidth")
    return weight_bytes / gpu_ns


def _ratio(a: float, b: float) -> float:
    lo = min(a, b)
    hi = max(a, b)
    if lo <= 0:
        return float("inf")
    return hi / lo


def same_gb_s(a: float, b: float, bar: float = SAME_RATIO) -> bool:
    return _ratio(a, b) <= bar


def monotone_increasing(values: list[float], slack: float = MONOTONE_SLACK) -> bool:
    if len(values) < 2:
        return False
    for prev, cur in zip(values, values[1:]):
        if cur < prev * slack:
            return False
    return True


def _index_rungs(rows: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        ident = str(row.get("id") or "")
        if ident:
            out[ident] = row
    return out


def _require_rung(index: Mapping[str, Mapping[str, Any]], ident: str) -> Mapping[str, Any]:
    row = index.get(ident)
    if not isinstance(row, Mapping):
        raise MissingRung(f"refusing a verdict: missing rung {ident}")
    if "gpu_ns_median" not in row or "weight_bytes" not in row:
        raise MissingRung(f"refusing a verdict: {ident} is missing gpu_ns_median/weight_bytes")
    return row


def assert_bytes_held(row: Mapping[str, Any], ident: str) -> None:
    got = int(row.get("bytes_per_thread_iteration") or 0)
    if got != BYTES_PER_ITER:
        raise ByteMismatch(
            f"{ident}: bytes_per_thread_iteration {got} != {BYTES_PER_ITER}"
        )
    streams = row.get("bytes_per_stream")
    if isinstance(streams, list) and streams:
        summed = sum(int(x) for x in streams)
        if summed != BYTES_PER_ITER:
            raise ByteMismatch(
                f"{ident}: bytes_per_stream {streams} sum to {summed} != {BYTES_PER_ITER}"
            )


def _arm_view(raw: Mapping[str, Any]) -> dict[str, Any]:
    weight_bytes = int(raw["weight_bytes"])
    gpu_ns = int(raw["gpu_ns_median"])
    gb_s = effective_gb_s(weight_bytes, gpu_ns)
    streams = [int(x) for x in (raw.get("bytes_per_stream") or [])]
    out = {
        "id": str(raw.get("id", "")),
        "kernel": raw.get("kernel"),
        "family": raw.get("family"),
        "stream_count": int(raw.get("stream_count") or 0),
        "bytes_per_stream": streams,
        "bytes_per_thread_iteration": int(
            raw.get("bytes_per_thread_iteration") or sum(streams) or 0
        ),
        "weight_bytes": weight_bytes,
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [int(x) for x in raw.get("gpu_ns_reps", [])],
        "gpu_us_median": round(gpu_ns / 1e3, 3),
        "effective_gb_s": round(gb_s, 1),
        "dispatches": int(raw.get("dispatches", 3)),
        "encoders": int(raw.get("encoders", 1)),
        "command_buffers": int(raw.get("command_buffers", 1)),
        "threads_per_threadgroup": int(raw.get("threads_per_threadgroup", 128)),
        "access_pattern": raw.get("access_pattern") or {},
    }
    if "occupancy" in raw:
        out["occupancy"] = raw["occupancy"]
    spec = RUNG_SPEC.get(out["id"])
    if spec:
        out["shape"] = spec["shape"]
        if out["stream_count"] == 0:
            out["stream_count"] = spec["stream_count"]
        if not out["bytes_per_stream"]:
            out["bytes_per_stream"] = list(spec["bytes_per_stream"])
            out["bytes_per_thread_iteration"] = BYTES_PER_ITER
    return out


def loads_survived(mlp: Mapping[str, Any], zero: Mapping[str, Any], half: Mapping[str, Any]) -> dict[str, Any]:
    a_ns = int(mlp["gpu_ns_median"])
    zero_ns = int(zero.get("gpu_ns_median") or 0)
    half_ns = int(half.get("gpu_ns_median") or 0)
    finite = float(mlp.get("effective_gb_s") or 0.0) > 0.0 and a_ns > 0
    above_floor = zero_ns > 0 and a_ns > 1.3 * zero_ns
    scales = half_ns > 0 and half_ns < 0.85 * a_ns
    survived = bool(finite and (above_floor or scales))
    return {
        "survived": survived,
        "finite_gb_s": finite,
        "above_zero_load_floor": above_floor,
        "time_scales_with_bytes": scales,
        "stripped_gpu_ns": a_ns,
        "zero_load_gpu_ns": zero_ns or None,
        "stripped_half_gpu_ns": half_ns or None,
        "proof": (
            "stripped time exceeds the no-load floor and/or drops when bytes are halved"
            if survived
            else "cannot prove the stripped loads survived; curve would be MIXED"
        ),
    }


def project_organ_ms(achieved_gb_s: float) -> dict[str, Any]:
    if achieved_gb_s <= 0:
        raise EmptyGpuSample("projection requires positive GB/s")
    speedup = achieved_gb_s / U1ALU_PRODUCTION_GB_S
    mlp_ms = MLP_MS / speedup
    saved = MLP_MS - mlp_ms
    token_ms = TOKEN_MS - saved
    tps = 1000.0 / token_ms if token_ms > 0 else None
    return {
        "kind": "projection",
        "label": "projection from a single-layer stripped probe, not a resident measurement",
        "from": (
            f"arithmetic over measured GB/s applied to MLP {MLP_MS} ms of a "
            f"{TOKEN_MS} ms token (PATH_TO_71 / ORGAN_BANDWIDTH mlp.gpu_ms), "
            f"scaled from cited production {U1ALU_PRODUCTION_GB_S} GB/s"
        ),
        "achieved_gb_s": round(achieved_gb_s, 1),
        "cited_production_gb_s": U1ALU_PRODUCTION_GB_S,
        "probe_speedup_vs_production": round(speedup, 4),
        "baseline_mlp_ms": MLP_MS,
        "baseline_token_ms": TOKEN_MS,
        "baseline_tps": TOKEN_TPS,
        "mlp_ms_projected": round(mlp_ms, 3),
        "token_ms_projected": round(token_ms, 3),
        "tps_projected": None if tps is None else round(tps, 2),
        "delta_mlp_ms": round(-saved, 3),
        "delta_token_ms": round(-saved, 3),
        "note": (
            "PROJECTION. The rungs are arithmetic-stripped, so this is a "
            "streaming-ceiling projection, not a promise that production "
            "decode sits at this rate."
        ),
    }


def judge(measurement: Mapping[str, Any]) -> dict[str, Any]:
    rungs = measurement.get("rungs") or {}
    alignment = measurement.get("alignment") or {}
    for ident in STREAM_RUNG_IDS:
        if ident not in rungs:
            raise MissingRung(f"refusing a verdict: missing rung {ident}")
    for ident in ALIGN_IDS:
        if ident not in alignment:
            raise MissingRung(f"refusing a verdict: alignment discriminator missing {ident}")

    gb = {ident: float(rungs[ident]["effective_gb_s"]) for ident in STREAM_RUNG_IDS}
    agb = {ident: float(alignment[ident]["effective_gb_s"]) for ident in ALIGN_IDS}
    stride = measurement.get("stride") or {}
    stride_gb = None
    if "stride_contig" in stride:
        stride_gb = float(stride["stride_contig"]["effective_gb_s"])
    elif "stride_contig" in alignment:
        stride_gb = float(alignment["stride_contig"]["effective_gb_s"])

    mlp = gb["mlp_2_2_2_32"]
    dn = gb["dn_4_2_32"]
    mid = gb["mid_2_4_32"]
    p6 = gb["pack_6_32"]
    p38 = gb["pack_38"]
    a2 = agb["align_2"]
    a4 = agb["align_4"]
    a16 = agb["align_16"]
    align_best = max(a4, a16)

    same_mlp_dn = same_gb_s(mlp, dn)
    same_mlp_mid = same_gb_s(mlp, mid)
    same_dn_mid = same_gb_s(dn, mid)
    merge_vals = [gb[i] for i in MERGE_ORDER]
    monotone = monotone_increasing(merge_vals)
    merge_lifts = (dn > mlp * SAME_RATIO) or (p6 > mlp * SAME_RATIO) or (p38 > mlp * SAME_RATIO)
    width_not_count = (not same_mlp_dn) and same_mlp_mid and (dn > mlp)
    align_lifts = align_best >= a2 * ALIGN_LIFT
    stride_lifts = stride_gb is not None and stride_gb >= mlp * ALIGN_LIFT
    survived = (measurement.get("loads_survived") or {}).get("survived", True)

    if not survived:
        verdict = VERDICT_MIXED
        why = "stripped loads were not shown to survive; refusing STREAM_COUNT_BOUND / ALIGNMENT_BOUND"
    elif width_not_count:
        verdict = VERDICT_ALIGN
        why = (
            "2+2+2 differs from 4+2 but 2+4+32 tracks 2+2+2: the 4-byte "
            "operand is the lift, not the stream count"
        )
    elif same_mlp_dn and align_lifts:
        verdict = VERDICT_ALIGN
        why = (
            "2+2+2 and 4+2 measure the same, so stream count is not the "
            "mechanism; 4- or 16-byte alignment of the 2-byte payloads lifts GB/s"
        )
    elif (not same_mlp_dn) and monotone and merge_lifts:
        verdict = VERDICT_STREAM
        why = (
            "GB/s rises monotonically as streams are merged and 2+2+2 differs "
            "from 4+2; the MLP's 497 is a packing property"
        )
    elif same_mlp_dn:
        verdict = VERDICT_NOT
        why = (
            "2+2+2 and 4+2 measure the same; stream count is not the mechanism "
            "and alignment did not lift past the bar"
        )
    else:
        verdict = VERDICT_MIXED
        why = (
            "The merge curve is not monotonic and the 2+2+2 vs 4+2 pair is not "
            "the same; not forcing STREAM_COUNT_BOUND or NOT_STREAM_COUNT"
        )

    next_candidate = None
    if verdict == VERDICT_NOT:
        if stride_lifts:
            next_candidate = "per-thread code stride"
        elif not same_gb_s(p38, mlp):
            next_candidate = "activation reuse pattern (pack_38 drops or jumps vs the 4-stream shape)"
        else:
            next_candidate = "cache set conflicts (not a dedicated arm in this run)"

    return {
        "verdict": verdict,
        "why": why,
        "same_mlp_dn": same_mlp_dn,
        "same_mlp_mid": same_mlp_mid,
        "same_dn_mid": same_dn_mid,
        "width_not_count": width_not_count,
        "monotone_merge": monotone,
        "merge_lifts": merge_lifts,
        "align_lifts": align_lifts,
        "stride_lifts": stride_lifts,
        "gb_s": {**{k: round(v, 1) for k, v in gb.items()}, **{k: round(v, 1) for k, v in agb.items()}},
        "merge_gb_s": [round(x, 1) for x in merge_vals],
        "align_best_over_align_2": round(align_best / a2, 4) if a2 > 0 else None,
        "dn_over_mlp": round(dn / mlp, 4) if mlp > 0 else None,
        "pack6_over_mlp": round(p6 / mlp, 4) if mlp > 0 else None,
        "same_ratio_bar": SAME_RATIO,
        "monotone_slack": MONOTONE_SLACK,
        "align_lift_bar": ALIGN_LIFT,
        "next_candidate": next_candidate,
        "loads_survived": survived,
    }


def measurement_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    rows = raw.get("rungs")
    if not isinstance(rows, list):
        raise MissingRung("refusing a verdict: raw.rungs is not a list")
    index = _index_rungs(rows)
    rungs: dict[str, dict[str, Any]] = {}
    for ident in STREAM_RUNG_IDS:
        row = _require_rung(index, ident)
        view = _arm_view(row)
        assert_bytes_held(view, ident)
        if view["bytes_per_stream"] != list(RUNG_SPEC[ident]["bytes_per_stream"]):
            raise ByteMismatch(
                f"{ident}: bytes_per_stream {view['bytes_per_stream']} != "
                f"{RUNG_SPEC[ident]['bytes_per_stream']}"
            )
        rungs[ident] = view

    align_rows = raw.get("alignment")
    if not isinstance(align_rows, list):
        raise MissingRung("refusing a verdict: raw.alignment is not a list")
    aindex = _index_rungs(align_rows)
    alignment: dict[str, dict[str, Any]] = {}
    for ident in ALIGN_IDS:
        row = _require_rung(aindex, ident)
        view = _arm_view(row)
        assert_bytes_held(view, ident)
        alignment[ident] = view

    stride: dict[str, dict[str, Any]] = {}
    for ident in STRIDE_IDS:
        row = aindex.get(ident) or index.get(ident)
        if isinstance(row, Mapping):
            view = _arm_view(row)
            assert_bytes_held(view, ident)
            stride[ident] = view
            alignment.setdefault(ident, view)

    zero_raw = raw.get("zero_load") or {}
    half_raw = raw.get("halfk") or {}
    zero = _arm_view(zero_raw) if zero_raw else {}
    half = _arm_view(half_raw) if half_raw else {}
    survived = loads_survived(rungs["mlp_2_2_2_32"], zero or {"gpu_ns_median": 0}, half or {"gpu_ns_median": 0})

    measured = {
        "layer": int(raw.get("layer", 0)),
        "warmup": int(raw.get("warmup", 0)),
        "reps": int(raw.get("reps", 0)),
        "git_head": raw.get("git_head", ""),
        "artifact_root": raw.get("artifact_root", ""),
        "timing": raw.get("timing", "MTLCommandBuffer GPUStartTime/GPUEndTime"),
        "concurrent_load": raw.get("concurrent_load") or {},
        "concurrent_load_start": raw.get("concurrent_load_start") or {},
        "absolute_gb_s_are_measured_under_load": True,
        "bytes_per_thread_iteration_held": BYTES_PER_ITER,
        "pre_registered_interpretation": raw.get("pre_registered_interpretation") or PRE_REGISTERED,
        "projections": raw.get("projections") or [],
        "weight_bytes": int(raw.get("weight_bytes") or rungs["mlp_2_2_2_32"]["weight_bytes"]),
        "dispatches": int(raw.get("dispatches") or 3),
        "rungs": rungs,
        "alignment": alignment,
        "stride": stride,
        "zero_load": zero,
        "halfk": half,
        "loads_survived": survived,
    }
    measured["judgement"] = judge(measured)
    measured["verdict"] = measured["judgement"]["verdict"]
    return measured


def _finding(measurement: Mapping[str, Any]) -> str:
    j = measurement["judgement"]
    rungs = measurement["rungs"]
    parts = [
        f"Verdict {j['verdict']}.",
        (
            "Ladder GB/s (unique payload / GPU ns, 38 B/iter held): "
            + ", ".join(
                f"{ident} {rungs[ident]['effective_gb_s']} "
                f"({RUNG_SPEC[ident]['stream_count']} "
                f"{'stream' if RUNG_SPEC[ident]['stream_count']==1 else 'streams'} "
                f"{RUNG_SPEC[ident]['bytes_per_stream']})"
                for ident in STREAM_RUNG_IDS
            )
            + "."
        ),
        j["why"] + ".",
    ]
    if j["verdict"] == VERDICT_STREAM:
        p6 = rungs["pack_6_32"]["effective_gb_s"]
        proj = project_organ_ms(p6)
        parts.append(
            f"An interleaved affine2 layout (6+32) measured {p6} GB/s; "
            f"packing costs a {PACKING_COST['expansion_payload']}× payload "
            f"expansion at catalog bake "
            f"({PACKING_COST['production_bytes_per_group']} B/group -> "
            f"{PACKING_COST['interleaved_payload_bytes_per_group']} B tight / "
            f"{PACKING_COST['interleaved_storage_bytes_per_group']} B stored). "
            f"Projected organ {proj['mlp_ms_projected']} ms of a "
            f"{proj['token_ms_projected']} ms token (PROJECTION, stripped ceiling)."
        )
    elif j["verdict"] == VERDICT_ALIGN:
        a = measurement["alignment"]
        parts.append(
            f"align_2 {a['align_2']['effective_gb_s']} GB/s, "
            f"align_4 {a['align_4']['effective_gb_s']} GB/s, "
            f"align_16 {a['align_16']['effective_gb_s']} GB/s "
            f"(lift bar {ALIGN_LIFT}). 4-byte-aligning the 2-byte payloads is "
            "the cheaper fix if that is the lift; it does not require merging "
            "scale and bias into the code stream."
        )
    elif j["verdict"] == VERDICT_NOT:
        nxt = j.get("next_candidate") or "untested"
        parts.append(f"Next candidate: {nxt}.")
    else:
        fastest = max(rungs, key=lambda k: rungs[k]["effective_gb_s"])
        slowest = min(rungs, key=lambda k: rungs[k]["effective_gb_s"])
        sc = (measurement.get("stride") or {}).get("stride_contig") or (
            measurement.get("alignment") or {}
        ).get("stride_contig")
        extra = ""
        if sc:
            extra = (
                f" Contiguous-code stride measured {sc['effective_gb_s']} GB/s "
                f"against mlp_2_2_2_32 {rungs['mlp_2_2_2_32']['effective_gb_s']} GB/s."
            )
        parts.append(
            f"Full curve is in rungs[]: fastest {fastest} "
            f"{rungs[fastest]['effective_gb_s']} GB/s, slowest {slowest} "
            f"{rungs[slowest]['effective_gb_s']} GB/s.{extra} "
            "Do not rescue a refuted layout lever and do not promote packing "
            "from a mixed curve."
        )
    return " ".join(parts)


def build(measurement: Mapping[str, Any]) -> dict[str, Any]:
    j = judge(measurement)
    pack_rate = float(measurement["rungs"]["pack_6_32"]["effective_gb_s"])
    mlp_rate = float(measurement["rungs"]["mlp_2_2_2_32"]["effective_gb_s"])
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "SELF_MEASURED_DIRTY",
        "gpu_authority": False,
        "took_gpu_lease": True,
        "source": (
            "crates/hawking-core/examples/stream_count_probe.rs; "
            "region GPU timestamps (MTLCommandBuffer GPUStartTime/GPUEndTime); "
            "one representative MLP layer of sealed-3.14, arithmetic stripped"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "pre_registered_interpretation": measurement.get("pre_registered_interpretation")
        or PRE_REGISTERED,
        "cited": {
            "mlp_arm_a_gb_s": MLP_ARM_A_GB_S,
            "deltanet_arm_a_gb_s": DN_ARM_A_GB_S,
            "u1alu_production_gb_s": U1ALU_PRODUCTION_GB_S,
            "mlp_ms": MLP_MS,
            "token_ms": TOKEN_MS,
            "from": "receipts/future/MLP_ALU_ROOFLINE.json, ORGAN_BANDWIDTH.json, PATH_TO_71.json",
        },
        "clean_gemv_gb_s": CLEAN_GEMV_GB_S,
        "bytes_per_thread_iteration_held": BYTES_PER_ITER,
        "same_ratio": SAME_RATIO,
        "monotone_slack": MONOTONE_SLACK,
        "align_lift": ALIGN_LIFT,
        "absolute_gb_s_are_measured_under_load": True,
        "layer": measurement.get("layer"),
        "warmup": measurement.get("warmup"),
        "reps": measurement.get("reps"),
        "weight_bytes": measurement.get("weight_bytes"),
        "dispatches": measurement.get("dispatches"),
        "rungs": measurement["rungs"],
        "alignment": measurement["alignment"],
        "stride": measurement.get("stride") or {},
        "zero_load": measurement.get("zero_load") or {},
        "halfk": measurement.get("halfk") or {},
        "loads_survived": measurement.get("loads_survived") or {},
        "judgement": j,
        "verdict": j["verdict"],
        "finding": _finding({**measurement, "judgement": j, "verdict": j["verdict"]}),
        "packing_cost": PACKING_COST,
        "projection_pack_6_32": project_organ_ms(pack_rate),
        "projection_mlp_2_2_2_32": project_organ_ms(mlp_rate),
        "timing": measurement.get("timing"),
        "concurrent_load": measurement.get("concurrent_load"),
        "concurrent_load_start": measurement.get("concurrent_load_start"),
        "git_head": measurement.get("git_head", ""),
        "artifact_root": measurement.get("artifact_root", ""),
        "projections": measurement.get("projections") or [],
        "refuted_elsewhere": [
            "region_granularity",
            "catalog_addressing",
            "raw_dispatch_count",
            "fuse_representation_decode",
        ],
    }


def record(measurement: Mapping[str, Any] | None = None, *, path: Path | None = None) -> Path:
    if measurement is None:
        raise MissingRung("refusing to record a receipt without a measurement")
    doc = build(measurement)
    out = path or RECEIPT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return out


def example_binaries() -> list[Path]:
    names = ("stream_count_probe",)
    roots: list[Path] = []
    env = os.environ.get("CARGO_TARGET_DIR")
    if env:
        roots.append(Path(env))
    roots.extend(
        [
            REPO / "target",
            REPO / "workspace" / "ops" / "build" / "rust",
        ]
    )
    out: list[Path] = []
    for root in roots:
        for profile in ("release-fast", "release"):
            for name in names:
                p = root / profile / "examples" / name
                if p.is_file():
                    out.append(p)
    return out


def run_example(
    artifact_root: Path,
    *,
    layer: int = 0,
    warmup: int = 5,
    reps: int = 11,
    out: Path | None = None,
    binary: Path | None = None,
) -> dict[str, Any]:
    bins = [binary] if binary is not None else example_binaries()
    if not bins:
        raise FileNotFoundError(
            "stream_count_probe binary not found; build with "
            "`CARGO_TARGET_DIR=workspace/ops/build/rust cargo build "
            "--profile release-fast -p hawking-core --example stream_count_probe`"
        )
    exe = bins[0]
    out = out or RAW_DEFAULT
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(exe),
        "--artifact-root",
        str(artifact_root),
        "--layer",
        str(layer),
        "--warmup",
        str(warmup),
        "--reps",
        str(reps),
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{exe} exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return json.loads(out.read_text())


def load_raw(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="write the sealed receipt")
    parser.add_argument("--from", dest="raw_path", default=None, help="raw example JSON")
    parser.add_argument("--measure", action="store_true", help="run the Metal example")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/Users/scammermike/noetic/NOETIC_PARENT_A"),
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=11)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    raw: dict[str, Any] | None = None
    if args.measure:
        raw = run_example(
            args.artifact_root,
            layer=args.layer,
            warmup=args.warmup,
            reps=args.reps,
            out=RAW_DEFAULT,
        )
    elif args.raw_path:
        raw = load_raw(Path(args.raw_path))
    elif RAW_DEFAULT.is_file():
        raw = load_raw(RAW_DEFAULT)

    if raw is None:
        print(
            "no measurement: pass --from RAW.json, --measure, or write "
            f"{RAW_DEFAULT}",
            file=sys.stderr,
        )
        return 2

    measured = measurement_from_raw(raw)
    if args.record:
        path = record(measured, path=args.out)
        print(f"wrote {path} verdict={measured['verdict']}")
    else:
        print(f"verdict={measured['verdict']}")
        j = measured["judgement"]
        for ident in STREAM_RUNG_IDS:
            r = measured["rungs"][ident]
            print(
                f"  {ident}: {r['stream_count']} streams {r['bytes_per_stream']}  "
                f"{r['effective_gb_s']} GB/s  {r['gpu_us_median']} us"
            )
        print(f"  why: {j['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

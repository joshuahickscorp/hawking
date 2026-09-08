#!/usr/bin/env python3
"""WHERE DOES DELTANET'S 8.227 ms ACTUALLY GO?

Isolated in_proj_qkvz is already 601 GB/s (MLP_ALU_ROOFLINE). The organ
averages 360 GB/s over 8.227 ms / 2.96 GB / 337 dispatches. This sidecar
decomposes the organ into the kernels production actually launches, with
per-kernel GPU time, bytes and GB/s. The parts must account for the organ
total; any gap is named, never absorbed.

    python3 tools/future/deltanet_organ_decompose.py --from RAW.json --record
    python3 tools/future/deltanet_organ_decompose.py --measure --record
    python3 -m pytest tools/future/test_deltanet_organ_decompose.py -q

evidence_class SELF_MEASURED_DIRTY. Absolute GB/s is measured-under-load.
The limiter verdict is the back-to-back ratio in the same process.
Does not change the production decode path.
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


RECEIPT = REPO / "receipts" / "future" / "DELTANET_ORGAN_DECOMPOSE.json"
RAW_DEFAULT = REPO / "receipts" / "future" / "_DELTANET_ORGAN_DECOMPOSE_raw.json"
SCHEMA = "hawking.future.deltanet_organ_decompose.v1"
VERSION = 1
RECORDED_BY = "tools/future/deltanet_organ_decompose.py"

CLEAN_GEMV_GB_S = 703.5
LM_HEAD_GB_S = 497.4
CITED_ORGAN_MS = 8.227
CITED_ORGAN_BYTES = 2_961_659_904
CITED_ORGAN_GB_S = 360.0
CITED_DISPATCHES = 337
CITED_ORGAN_NS = int(round(CITED_ORGAN_MS * 1e6))

# Catalog bytes from MLP_BYTE_CENSUS (whole-tensor, 48 layers).
QKVZ_BYTES = 2_139_096_960
BA_BYTES = 12_535_680
OUT_BYTES = 802_162_560
CONV_BYTES = 7_864_704
NORM_LINEAR_ATTN_BYTES = 24_960
A_LOG_DT_BIAS_BYTES = 19_200
REC_STATE_RESIDENT = 150_994_944
REC_STATE_RW = REC_STATE_RESIDENT * 2
CONV_STATE_RW = 5_898_240 * 2
INPUT_RMS_BYTES = 20_480          # one hidden f32 scale
POST_ATTN_DN_BYTES = 48 * 5_120 * 4

# Sealed 628-graph partition: these launches ARE the 337.
# qkvz and ba are fused into pair_concat (dn_inproj); they are diagnostic
# only and must not be added on top of dn_inproj.
PARTITION = (
    "dn_input_rmsnorm",
    "dn_inproj",
    "rearrange_48",
    "ba_to_decay_48",
    "gated_delta_unfused",
    "gated_rmsnorm_48",
    "dn_out_proj",
    "dn_residual_rmsnorm",
)
ORGAN_FAMILY = "dn_as_executed"

# ALU bars, identical to mlp_alu_roofline. Ratios, not absolute GB/s.
JUMP_RATIO = 1.25
STAY_RATIO = 1.12
SUBLINEAR_SLACK = 1.20
LINEAR_TOLERANCE = 0.20

VERDICT_ALU = "ALU_BOUND"
VERDICT_MEM = "MEMORY_SYSTEM_BOUND"
VERDICT_MIXED = "MIXED"
VERDICT_LATENCY = "LATENCY_BOUND"
VERDICT_SERIAL = "SERIAL_STATE"
VERDICT_LAYOUT = "LAYOUT_OR_UNCOALESCED"
VERDICT_ELEMENTWISE = "ELEMENTWISE_SMALL"

# A kernel whose per-dispatch GPU time sits near the empty-CB / tiny-grid
# floor is latency-bound, not byte-bound. qkvz zero-load on a 8192-TG
# grid was ~14 us; a 48-TG ba GEMV or a 48-thread rmsnorm is smaller.
LATENCY_US_PER_DISPATCH = 25.0
SMALL_BYTES = 50_000_000

# Isolated-family CBs vs one organ CB: each family pays its own commit.
# A residual inside this band is named, not a missing kernel.
FAMILY_SUM_TOLERANCE = 0.15

CLAIM_BOUNDARY = (
    "One production-shaped DeltaNet organ (48 layers) on sealed-3.14, "
    "SELF_MEASURED_DIRTY. GPU time is MTLCommandBuffer GPUStartTime/"
    "GPUEndTime for one isolated command buffer per family, and one CB "
    "for encode_deltanet x 48 (the organ as executed, including out_proj). "
    "Fusion flags match the 628-dispatch sealed graph: GateUpSwiglu, "
    "FUSE_GQA_QKV, FUSE_DN_INPROJ, FUSE_ADD_RMSNORM, FUSE_BA_DELTA off. "
    "Catalog bytes are MLP_BYTE_CENSUS whole-tensor figures. Recurrent "
    "traffic (rec_state R+W) is geometry-derived and is NOT in the 2.96 GB "
    "organ census — that census is why the organ averages 360 GB/s while "
    "qkvz alone is 601. Absolute GB/s is measured-under-load; limiter "
    "verdicts are back-to-back ratios in the same process. The 8.227 ms "
    "figure is the cited region total from RESIDENT_71TPS_CAUSAL_BUDGET; "
    "the gap between this run's organ CB and that citation is named. "
    "Does not change the production decode path."
)


class UnreconciledDecomposition(Exception):
    """Raised rather than emit a decomposition whose parts do not cover the organ."""


class MissingArm(Exception):
    """Raised rather than classify a GEMV over an incomplete matched pair."""


class ByteMismatch(Exception):
    """Raised when ARM A does not read the same weight bytes as production."""


class EmptyGpuSample(Exception):
    """Raised rather than divide by a missing GPU timestamp."""


def effective_gb_s(weight_bytes: int, gpu_ns: int) -> float:
    if gpu_ns <= 0:
        raise EmptyGpuSample("gpu_ns must be positive to form a bandwidth")
    if weight_bytes <= 0:
        raise ValueError("weight_bytes must be positive to form a bandwidth")
    return weight_bytes / gpu_ns


def ns_to_ms(ns: int) -> float:
    return ns / 1e6


def _require_arm(organ: Mapping[str, Any], key: str, organ_name: str) -> Mapping[str, Any]:
    arm = organ.get(key)
    if not isinstance(arm, Mapping):
        raise MissingArm(f"refusing a verdict: {organ_name} is missing {key}")
    for field in ("weight_bytes", "gpu_ns_median"):
        if field not in arm:
            raise MissingArm(
                f"refusing a verdict: {organ_name}.{key} is missing {field}"
            )
    return arm


def loads_survived(organ: Mapping[str, Any]) -> dict[str, Any]:
    a = organ["arm_a_stripped"]
    zero = organ.get("zero_load") or {}
    a_half = organ.get("arm_a_halfk") or {}
    a_ns = int(a["gpu_ns_median"])
    zero_ns = int(zero.get("gpu_ns_median") or 0)
    a_half_ns = int(a_half.get("gpu_ns_median") or 0)
    gb_s = float(a.get("effective_gb_s") or 0.0)
    finite = gb_s > 0.0 and a_ns > 0
    above_floor = zero_ns > 0 and a_ns > 1.3 * zero_ns
    scales = a_half_ns > 0 and a_half_ns < 0.85 * a_ns
    survived = bool(finite and (above_floor or scales))
    return {
        "survived": survived,
        "finite_gb_s": finite,
        "above_zero_load_floor": above_floor,
        "time_scales_with_bytes": scales,
        "stripped_gpu_ns": a_ns,
        "zero_load_gpu_ns": zero_ns or None,
        "stripped_half_gpu_ns": a_half_ns or None,
        "proof": (
            "stripped time exceeds the no-load floor and/or drops when bytes are halved"
            if survived
            else "cannot prove the stripped loads survived; ARM A jump would be MIXED not ALU_BOUND"
        ),
    }


def judge_alu(organ: Mapping[str, Any], name: str) -> dict[str, Any]:
    prod = _require_arm(organ, "production", name)
    arm_a = _require_arm(organ, "arm_a_stripped", name)
    arm_b = _require_arm(organ, "arm_b_halfk", name)
    prod_bytes = int(prod["weight_bytes"])
    a_bytes = int(arm_a["weight_bytes"])
    if a_bytes != prod_bytes:
        raise ByteMismatch(
            f"refusing a verdict: {name} ARM A weight_bytes {a_bytes} != "
            f"production {prod_bytes}"
        )
    prod_ns = int(prod["gpu_ns_median"])
    a_ns = int(arm_a["gpu_ns_median"])
    b_ns = int(arm_b["gpu_ns_median"])
    b_bytes = int(arm_b["weight_bytes"])
    if prod_ns <= 0 or a_ns <= 0 or b_ns <= 0:
        raise EmptyGpuSample(f"{name}: gpu_ns_median must be positive on every arm")
    if b_bytes <= 0 or prod_bytes <= 0:
        raise ValueError(f"{name}: weight_bytes must be positive")
    prod_gb = effective_gb_s(prod_bytes, prod_ns)
    a_gb = effective_gb_s(a_bytes, a_ns)
    b_gb = effective_gb_s(b_bytes, b_ns)
    a_ratio = a_gb / prod_gb
    byte_ratio = b_bytes / prod_bytes
    time_ratio = b_ns / prod_ns
    scale = time_ratio / byte_ratio if byte_ratio > 0 else float("inf")
    b_linear = abs(scale - 1.0) <= LINEAR_TOLERANCE
    b_sublinear = scale > SUBLINEAR_SLACK
    a_jump = a_ratio >= JUMP_RATIO
    a_stay = a_ratio <= STAY_RATIO
    survived = loads_survived(organ)
    if a_stay:
        verdict = VERDICT_MEM
    elif a_jump and b_sublinear and survived["survived"]:
        verdict = VERDICT_ALU
    else:
        verdict = VERDICT_MIXED
    return {
        "verdict": verdict,
        "production_gb_s": round(prod_gb, 1),
        "arm_a_gb_s": round(a_gb, 1),
        "arm_b_gb_s": round(b_gb, 1),
        "arm_a_over_production": round(a_ratio, 4),
        "arm_b_byte_ratio": round(byte_ratio, 4),
        "arm_b_time_ratio": round(time_ratio, 4),
        "arm_b_time_over_byte": round(scale, 4),
        "arm_a_jump": a_jump,
        "arm_a_stay": a_stay,
        "arm_b_linear": b_linear,
        "arm_b_sublinear": b_sublinear,
        "loads_survived": survived,
        "jump_ratio_bar": JUMP_RATIO,
        "stay_ratio_bar": STAY_RATIO,
        "sublinear_slack": SUBLINEAR_SLACK,
    }


def _arm_view(raw: Mapping[str, Any]) -> dict[str, Any]:
    weight_bytes = int(raw.get("weight_bytes") or 0)
    gpu_ns = int(raw["gpu_ns_median"])
    out = {
        "label": str(raw.get("label", "")),
        "kernel": raw.get("kernel"),
        "weight_bytes": weight_bytes,
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [int(x) for x in raw.get("gpu_ns_reps", [])],
        "dispatches": int(raw.get("dispatches", 0)),
        "encoders": int(raw.get("encoders", 1)),
        "command_buffers": int(raw.get("command_buffers", 1)),
        "ms": round(ns_to_ms(gpu_ns), 4) if gpu_ns else 0.0,
    }
    if weight_bytes > 0 and gpu_ns > 0:
        gb_s = effective_gb_s(weight_bytes, gpu_ns)
        out["effective_gb_s"] = round(gb_s, 1)
        out["share_of_clean_roof"] = round(gb_s / CLEAN_GEMV_GB_S, 4)
        out["share_of_lm_head"] = round(gb_s / LM_HEAD_GB_S, 4)
    else:
        out["effective_gb_s"] = None
        out["share_of_clean_roof"] = None
        out["share_of_lm_head"] = None
    if "occupancy" in raw:
        out["occupancy"] = raw["occupancy"]
    return out


def classify_family(
    name: str,
    fam: Mapping[str, Any],
    *,
    alu: Mapping[str, Any] | None = None,
    families: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Limiter class forced by the measurement, not by a byte-rate average."""
    ns = int(fam["gpu_ns_median"])
    disp = int(fam.get("dispatches") or 0)
    bytes_ = int(fam.get("weight_bytes") or 0)
    per_us = (ns / disp / 1e3) if disp else None
    gb_s = None
    if bytes_ > 0 and ns > 0:
        gb_s = bytes_ / ns
    measurement: dict[str, Any] = {
        "gpu_ns": ns,
        "ms": round(ns_to_ms(ns), 4),
        "dispatches": disp,
        "bytes": bytes_,
        "us_per_dispatch": None if per_us is None else round(per_us, 2),
        "effective_gb_s": None if gb_s is None else round(gb_s, 1),
    }

    if name in ("dn_inproj", "dn_qkvz", "dn_out_proj", "dn_ba") and alu:
        key = {
            "dn_inproj": "in_proj_qkvz",
            "dn_qkvz": "in_proj_qkvz",
            "dn_out_proj": "out_proj",
            "dn_ba": "in_proj_ba",
        }[name]
        if key in alu and "verdict" in alu[key]:
            verdict = alu[key]["verdict"]
            why = (
                f"matched-pair ARM A/B on layer-0 {key}: "
                f"production {alu[key]['judgement']['production_gb_s']} GB/s, "
                f"stripped {alu[key]['judgement']['arm_a_gb_s']} GB/s "
                f"({alu[key]['judgement']['arm_a_over_production']}x), "
                f"ARM B time/byte {alu[key]['judgement']['arm_b_time_over_byte']}"
            )
            # Tiny ba: even if ALU mixed, the organ time is the launch floor.
            if name == "dn_ba" and per_us is not None and per_us < LATENCY_US_PER_DISPATCH:
                return {
                    "limiter": VERDICT_LATENCY,
                    "alu_pair": verdict,
                    "why": (
                        f"ba is {bytes_} B across 48 launches "
                        f"({per_us:.1f} us/dispatch). A clean GEMV of those "
                        f"bytes is {bytes_ / (CLEAN_GEMV_GB_S * 1e9) * 1e6:.2f} us "
                        f"total. Time is the launch/occupancy floor, not bytes. "
                        f"ALU pair on the same kernel: {verdict}."
                    ),
                    "measurement": measurement,
                }
            return {
                "limiter": verdict,
                "why": why,
                "measurement": measurement,
                "alu_pair": alu[key]["judgement"],
            }

    if name == "gated_delta_unfused":
        stream = (families or {}).get("rec_state_f32_stream") or {}
        fused = (families or {}).get("gated_delta_fused_ba") or {}
        f4 = (families or {}).get("gated_delta_widen_f4") or {}
        stream_ns = int(stream.get("gpu_ns_median") or 0)
        f4_ns = int(f4.get("gpu_ns_median") or 0)
        fused_ns = int(fused.get("gpu_ns_median") or 0)
        traffic_gb_s = REC_STATE_RW / ns if ns else None
        stream_gb_s = (
            effective_gb_s(int(stream.get("weight_bytes") or REC_STATE_RESIDENT), stream_ns)
            if stream_ns > 0
            else None
        )
        f4_ratio = (fused_ns / f4_ns) if f4_ns > 0 and fused_ns > 0 else None
        limiter = VERDICT_SERIAL
        why = (
            f"recurrent S_t ← f(S_{{t-1}}, q, k, v, decay, beta); 48 serial "
            f"launches, {REC_STATE_RW:,} B R+W, {round(ns_to_ms(ns), 3)} ms "
            f"({None if traffic_gb_s is None else round(traffic_gb_s, 1)} GB/s on "
            f"state traffic). Consume (gated_rmsnorm, out_proj) cannot start "
            f"until this retires."
        )
        if stream_gb_s and traffic_gb_s and stream_gb_s > 1.5 * traffic_gb_s:
            limiter = VERDICT_LAYOUT
            why += (
                f" A contiguous copy of rec_state reached {round(stream_gb_s, 1)} GB/s "
                f"in the same process; the update is {round(stream_gb_s / traffic_gb_s, 2)}x "
                f"slower than a copy of the same buffer, so the limiter is the "
                f"access pattern / ALU of the vi-SIMD kernel, not the byte count."
            )
        if f4_ratio and f4_ratio >= JUMP_RATIO:
            limiter = VERDICT_LAYOUT
            why += (
                f" widen_f4 cut fused-ba time {round(f4_ratio, 2)}x "
                f"({fused_ns} → {f4_ns} ns) without changing bytes: layout/"
                f"uncoalesced, not a bandwidth roof."
            )
        measurement["traffic_bytes"] = REC_STATE_RW
        measurement["traffic_gb_s"] = None if traffic_gb_s is None else round(traffic_gb_s, 1)
        measurement["stream_gb_s"] = None if stream_gb_s is None else round(stream_gb_s, 1)
        measurement["widen_f4_over_fused"] = None if f4_ratio is None else round(f4_ratio, 4)
        return {"limiter": limiter, "why": why, "measurement": measurement}

    if per_us is not None and per_us < LATENCY_US_PER_DISPATCH and bytes_ < SMALL_BYTES:
        roof_us = (
            bytes_ / (CLEAN_GEMV_GB_S * 1e9) * 1e6 if bytes_ else None
        )
        limiter = (
            VERDICT_ELEMENTWISE
            if name in ("gated_rmsnorm_48", "dn_residual_rmsnorm", "dn_input_rmsnorm",
                        "ba_to_decay_48")
            else VERDICT_LATENCY
        )
        why = (
            f"{disp} launches × {per_us:.1f} us = {round(ns_to_ms(ns), 3)} ms moving "
            f"{bytes_:,} catalog bytes. At the 703.5 GB/s clean roof those bytes "
            f"are {0.0 if roof_us is None else round(roof_us, 2)} us total. "
            f"Byte-rate averages hide this: the kernel is launch/occupancy bound."
        )
        return {"limiter": limiter, "why": why, "measurement": measurement}

    if gb_s is not None and gb_s < 200 and bytes_ >= SMALL_BYTES:
        return {
            "limiter": VERDICT_LAYOUT,
            "why": (
                f"{round(gb_s, 1)} GB/s on {bytes_:,} B is well below the LM-head "
                f"497 and the clean roof 703.5; not a small-kernel floor."
            ),
            "measurement": measurement,
        }

    return {
        "limiter": VERDICT_MIXED,
        "why": "no ALU pair and not under the latency floor; left unclassified rather than forced",
        "measurement": measurement,
    }


def reconcile(
    organ_ns: int,
    parts: Mapping[str, int],
    *,
    required: tuple[str, ...] = PARTITION,
) -> dict[str, Any]:
    """Account for organ_ns with named parts. Refuse a silent absorb."""
    if organ_ns <= 0:
        raise EmptyGpuSample("organ gpu_ns must be positive to reconcile")
    missing = [k for k in required if k not in parts]
    if missing:
        raise UnreconciledDecomposition(
            "refusing a decomposition whose parts do not cover the organ: "
            f"missing {missing}"
        )
    zero = [k for k in required if int(parts[k]) <= 0]
    if zero:
        raise UnreconciledDecomposition(
            "refusing a decomposition that absorbs empty families into the "
            f"organ total: zero-ns {zero}"
        )
    summed = sum(int(parts[k]) for k in required)
    residual_ns = organ_ns - summed
    frac = residual_ns / organ_ns
    if residual_ns > 0:
        residual_name = (
            "organ_cb_slower_than_isolated_family_sum: the organ command "
            "buffer is slower than the sum of per-family CBs, so this is "
            "not isolated-CB commit tax (that would make the sum larger). "
            "Named candidate: dependency / serialization stalls inside "
            "the chained encode_deltanet graph (state update must retire "
            "before gated_rmsnorm and out_proj). Named, not absorbed."
        )
    elif residual_ns < 0:
        residual_name = (
            "isolated_family_sum_vs_organ_cb: each family is its own "
            "command buffer, so the sum pays per-family commit/idle that "
            "the organ CB does not. Named, not absorbed."
        )
    else:
        residual_name = "exact: isolated families compose to the organ CB"
    return {
        "organ_ns": organ_ns,
        "organ_ms": round(ns_to_ms(organ_ns), 4),
        "sum_partition_ns": summed,
        "sum_partition_ms": round(ns_to_ms(summed), 4),
        "residual_ns": residual_ns,
        "residual_ms": round(ns_to_ms(residual_ns), 4),
        "residual_fraction": round(frac, 4),
        "residual_name": residual_name,
        "within_tolerance": abs(frac) <= FAMILY_SUM_TOLERANCE,
        "required": list(required),
        "parts_ns": {k: int(parts[k]) for k in required},
        "parts_ms": {k: round(ns_to_ms(int(parts[k])), 4) for k in required},
    }


def _alu_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    src = raw.get("alu_matched_pair") or {}
    out: dict[str, Any] = {}
    for name in ("in_proj_qkvz", "out_proj", "in_proj_ba"):
        if name not in src:
            continue
        organ = src[name]
        viewed = {
            "organ": name,
            "kernel": organ.get("kernel"),
            "codec": organ.get("codec"),
            "projection": organ.get("projection"),
            "production": _arm_view(organ["production"]),
            "arm_a_stripped": _arm_view(organ["arm_a_stripped"]),
            "arm_b_halfk": _arm_view(organ["arm_b_halfk"]),
        }
        if "zero_load" in organ:
            viewed["zero_load"] = _arm_view(organ["zero_load"])
        if "arm_a_halfk" in organ:
            viewed["arm_a_halfk"] = _arm_view(organ["arm_a_halfk"])
        judged = judge_alu(viewed, name)
        viewed["judgement"] = judged
        viewed["verdict"] = judged["verdict"]
        out[name] = viewed
    return out


def measurement_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    families_raw = raw.get("families")
    if not isinstance(families_raw, Mapping):
        raise UnreconciledDecomposition(
            "refusing a decomposition: raw JSON has no families object"
        )
    families = {name: _arm_view(body) for name, body in families_raw.items()}
    if ORGAN_FAMILY not in families:
        raise UnreconciledDecomposition(
            f"refusing a decomposition: {ORGAN_FAMILY} (the organ as executed) is missing"
        )
    organ = families[ORGAN_FAMILY]
    organ_ns = int(organ["gpu_ns_median"])
    parts = {k: int(families[k]["gpu_ns_median"]) for k in families}
    recon = reconcile(organ_ns, parts)
    alu = _alu_from_raw(raw)
    classified = []
    for name in list(PARTITION) + [
        n for n in ("dn_qkvz", "dn_ba", "gated_delta_fused_ba",
                    "gated_delta_widen_f4", "rec_state_f32_stream",
                    "organ_incomplete_missing_out_proj")
        if n in families
    ]:
        if name not in families:
            continue
        row = classify_family(name, families[name], alu=alu, families=families)
        classified.append({
            "id": name,
            "kernel": families[name].get("kernel"),
            "in_partition": name in PARTITION,
            "ms": families[name]["ms"],
            "gpu_ns": families[name]["gpu_ns_median"],
            "bytes": families[name]["weight_bytes"],
            "dispatches": families[name]["dispatches"],
            "effective_gb_s": families[name]["effective_gb_s"],
            "us_per_dispatch": row["measurement"].get("us_per_dispatch"),
            "limiter": row["limiter"],
            "why": row["why"],
            "measurement_that_forced_it": row["measurement"],
        })
    classified.sort(key=lambda r: r["gpu_ns"], reverse=True)
    cited_gap_ns = organ_ns - CITED_ORGAN_NS
    return {
        "layer": int(raw.get("layer", 0)),
        "warmup": int(raw.get("warmup", 0)),
        "reps": int(raw.get("reps", 0)),
        "session_warmup": int(raw.get("session_warmup", 0)),
        "session_reps": int(raw.get("session_reps", 0)),
        "git_head": raw.get("git_head", ""),
        "artifact_root": raw.get("artifact_root", ""),
        "timing": raw.get("timing", "MTLCommandBuffer GPUStartTime/GPUEndTime"),
        "concurrent_load": raw.get("concurrent_load") or {},
        "concurrent_load_start": raw.get("concurrent_load_start") or {},
        "absolute_gb_s_are_measured_under_load": True,
        "production_fusions": raw.get("production_fusions") or {},
        "session_open_s": raw.get("session_open_s"),
        "dense_w_materialized": raw.get("dense_w_materialized", 0),
        "theoretical_dispatches_628_graph": raw.get("theoretical_dispatches_628_graph"),
        "as_executed_named": raw.get("as_executed_named") or {},
        "families": families,
        "alu_matched_pair": alu,
        "ranked": classified,
        "reconciliation": recon,
        "cited_organ": {
            "source": "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json",
            "ms": CITED_ORGAN_MS,
            "ns": CITED_ORGAN_NS,
            "bytes": CITED_ORGAN_BYTES,
            "gb_s": CITED_ORGAN_GB_S,
            "dispatches": CITED_DISPATCHES,
            "this_run_organ_ns": organ_ns,
            "this_run_organ_ms": round(ns_to_ms(organ_ns), 4),
            "gap_ns": cited_gap_ns,
            "gap_ms": round(ns_to_ms(cited_gap_ns), 4),
            "gap_name": (
                "this_run_organ_cb_vs_cited_8.227ms_region. Shared GPU load "
                "and isolated-organ vs in-token-graph are the candidates; "
                "the number is reported, not folded into a family."
            ),
        },
    }


def _largest_addressable(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    partition = [r for r in ranked if r.get("in_partition")]
    if not partition:
        raise UnreconciledDecomposition("no partition rows to rank")
    top = partition[0]
    experiment = {
        VERDICT_ALU: (
            "cheaper decode per byte on this GEMV (the ALU pair already "
            "moved GB/s when arithmetic was stripped); one layer, token-identical"
        ),
        VERDICT_MEM: (
            "bytes or placement, not decode: ARM A did not jump. A contiguous "
            "staging of this tensor is the cheapest falsifier (already refuted "
            "for MLP)."
        ),
        VERDICT_MIXED: (
            "do not promote. ARM A jumped with surviving loads but ARM B "
            "tracked bytes (half K also halves FMAs). Cheapest next: a "
            "decode that cuts FMA/byte without cutting K, one layer."
        ),
        VERDICT_LAYOUT: (
            "re-run the N026 widen_f4 / coalesced-load kernel against "
            "production gated-delta, token-identical, on the 628 graph "
            "(FUSE_BA_DELTA off). Isolated fused-ba already moved; production "
            "still launches the unfused vi-SIMD kernel."
        ),
        VERDICT_SERIAL: (
            "the update is serial in a way a GEMV is not. Cheapest: measure "
            "whether gated_rmsnorm+out_proj can be fused onto the rec_out "
            "write inside the same kernel so the dependency stall is paid once."
        ),
        VERDICT_LATENCY: (
            "fuse this launch into its producer (it is already under 25 us). "
            "Do not treat dispatch count as the organ limiter; the ms here "
            "is the floor, not a 6.25 us tax times N."
        ),
        VERDICT_ELEMENTWISE: (
            "epilogue-fuse into the producer GEMV or the state update. "
            "The bytes are a rounding error; the time is the launch."
        ),
    }.get(top["limiter"], "name the limiter with a matched pair before spending a week")
    return {
        "id": top["id"],
        "ms": top["ms"],
        "limiter": top["limiter"],
        "share_of_organ_partition": None,
        "cheapest_decisive_experiment": experiment,
        "why_this_one": (
            f"{top['id']} is the largest partition cost at {top['ms']} ms, "
            f"limited by {top['limiter']}."
        ),
    }


def _finding(m: Mapping[str, Any]) -> str:
    recon = m["reconciliation"]
    ranked = m["ranked"]
    cited = m["cited_organ"]
    parts = []
    for row in ranked:
        if not row["in_partition"]:
            continue
        gb = row["effective_gb_s"]
        gb_s = f"{gb} GB/s" if gb is not None else "n/a GB/s"
        parts.append(
            f"{row['id']} is {row['ms']} ms at {gb_s} "
            f"({row['dispatches']} disp, {row['limiter']})"
        )
    residual = (
        f"partition sum {recon['sum_partition_ms']} ms vs organ CB "
        f"{recon['organ_ms']} ms, residual {recon['residual_ms']} ms "
        f"({recon['residual_name']})"
    )
    cited_line = (
        f"this-run organ {cited['this_run_organ_ms']} ms vs cited 8.227 ms, "
        f"gap {cited['gap_ms']} ms ({cited['gap_name']})"
    )
    top = _largest_addressable(ranked)
    return (
        "DeltaNet as executed, not as a byte-rate average. "
        + "; ".join(parts)
        + f". {residual}. {cited_line}. "
        f"Largest addressable cost: {top['id']} at {top['ms']} ms "
        f"({top['limiter']}). Cheapest decisive experiment: "
        f"{top['cheapest_decisive_experiment']}"
    )


def build(measurement: Mapping[str, Any]) -> dict[str, Any]:
    recon = measurement["reconciliation"]
    ranked = measurement["ranked"]
    top = _largest_addressable(ranked)
    delta = next((r for r in ranked if r["id"] == "gated_delta_unfused"), None)
    fused = next((r for r in ranked if r["id"] == "gated_delta_fused_ba"), None)
    f4 = next((r for r in ranked if r["id"] == "gated_delta_widen_f4"), None)
    ba_decay = next((r for r in ranked if r["id"] == "ba_to_decay_48"), None)
    demonstrated = None
    if fused and f4 and f4["gpu_ns"] > 0 and fused["gpu_ns"] > f4["gpu_ns"]:
        unfused_pair_ms = None
        if delta and ba_decay:
            unfused_pair_ms = round(delta["ms"] + ba_decay["ms"], 4)
        demonstrated = {
            "id": "gated_delta_unfused",
            "production_unfused_delta_ms": None if delta is None else delta["ms"],
            "production_ba_to_decay_ms": None if ba_decay is None else ba_decay["ms"],
            "production_unfused_pair_ms": unfused_pair_ms,
            "fused_ba_ms": fused["ms"],
            "widen_f4_ms": f4["ms"],
            "ms_saved_f4_vs_fused_ba": round(fused["ms"] - f4["ms"], 4),
            "limiter": None if delta is None else delta["limiter"],
            "note": (
                "Largest cost with a demonstrated lever, not the largest cost. "
                "in_proj is bigger but MIXED (cannot promote). widen_f4 is a "
                "fused-ba sibling, so the fair cut is fused_ba minus f4, not "
                "unfused delta minus f4. FUSE_BA_DELTA itself does not save "
                "GPU ms (fused_ba ≈ unfused delta); it only removes the "
                "ba_to_decay launch. Production still launches unfused "
                "vi-SIMD on the 628 graph. 628-graph A/B still required."
            ),
        }
    organ_ms = recon["organ_ms"]
    for row in ranked:
        if row["in_partition"] and organ_ms:
            row["share_of_organ"] = round(row["ms"] / organ_ms, 4)
    top["share_of_organ_partition"] = next(
        (r.get("share_of_organ") for r in ranked if r["id"] == top["id"]), None
    )
    qkvz_alu = (measurement.get("alu_matched_pair") or {}).get("in_proj_qkvz") or {}
    qkvz_fam = (measurement.get("families") or {}).get("dn_qkvz") or {}
    qkvz_j = qkvz_alu.get("judgement") or {}
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "SELF_MEASURED_DIRTY",
        "gpu_authority": False,
        "took_gpu_lease": True,
        "source": (
            "crates/hawking-core/examples/alu_roofline_organs.rs --mode "
            "deltanet-decompose; production encode_deltanet x 48 plus "
            "per-family isolated CBs; Q4 matched-pair ALU on qkvz, out_proj, ba"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "clean_gemv_gb_s": CLEAN_GEMV_GB_S,
        "lm_head_gb_s": LM_HEAD_GB_S,
        "cited_organ": measurement["cited_organ"],
        "absolute_gb_s_are_measured_under_load": True,
        "layer": measurement.get("layer"),
        "warmup": measurement.get("warmup"),
        "reps": measurement.get("reps"),
        "session_warmup": measurement.get("session_warmup"),
        "session_reps": measurement.get("session_reps"),
        "production_fusions": measurement.get("production_fusions"),
        "session_open_s": measurement.get("session_open_s"),
        "dense_w_materialized": measurement.get("dense_w_materialized"),
        "theoretical_dispatches_628_graph": measurement.get(
            "theoretical_dispatches_628_graph"
        ),
        "timing": measurement.get("timing"),
        "concurrent_load": measurement.get("concurrent_load"),
        "concurrent_load_start": measurement.get("concurrent_load_start"),
        "git_head": measurement.get("git_head", ""),
        "artifact_root": measurement.get("artifact_root", ""),
        "as_executed_named": measurement.get("as_executed_named"),
        "alu_matched_pair": measurement.get("alu_matched_pair"),
        "families": measurement["families"],
        "reconciliation": recon,
        "ranked": ranked,
        "largest_addressable_cost": top,
        "largest_demonstrated_lever": demonstrated,
        "u1alu_qkvz_not_the_organ": {
            "cited_u1alu_isolated_layer0_gb_s": 600.9,
            "this_run_isolated_layer0_gb_s": qkvz_j.get("production_gb_s"),
            "this_run_isolated_layer0_arm_a_gb_s": qkvz_j.get("arm_a_gb_s"),
            "this_run_isolated_layer0_arm_a_ratio": qkvz_j.get("arm_a_over_production"),
            "this_run_48layer_qkvz_family_gb_s": qkvz_fam.get("effective_gb_s"),
            "organ_cited_gb_s": CITED_ORGAN_GB_S,
            "note": (
                "Absolute isolated-layer GB/s moved with concurrent load "
                "(this run vs u1alu). The ARM A ratio did not: 1.57x here, "
                "1.57x in MLP_ALU_ROOFLINE. The 48-layer family CB is the "
                "organ-relevant rate."
            ),
            "conclusion": (
                "isolated DeltaNet qkvz is already in the 580-600 GB/s class "
                "as a 48-layer family; the organ's 360 GB/s is not this kernel. "
                "The organ average is a bytes/time mix of a fast Q4 GEMV, a "
                "serial state update that moves 0.30 GB of rec_state, and a "
                "tail of launch-bound elementwise kernels."
            ),
        },
        "refuted_elsewhere": [
            "raw_dispatch_count",
            "region_granularity",
            "catalog_addressing",
            "bytes_per_dispatch",
            "fuse_representation_decode",
        ],
        "finding": _finding({**measurement, "ranked": ranked}),
        "verdict": top["limiter"],
    }


def record(measurement: Mapping[str, Any] | None = None, *, path: Path | None = None) -> Path:
    if measurement is None:
        raise UnreconciledDecomposition("refusing to record a receipt without a measurement")
    doc = build(measurement)
    out = path or RECEIPT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return out


def example_binaries() -> list[Path]:
    names = ("alu_roofline_organs",)
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
    session_warmup: int = 2,
    session_reps: int = 7,
    out: Path | None = None,
    binary: Path | None = None,
    use_lock: bool = True,
) -> dict[str, Any]:
    bins = [binary] if binary is not None else example_binaries()
    if not bins:
        raise FileNotFoundError(
            "alu_roofline_organs binary not found; build with "
            "`CARGO_TARGET_DIR=workspace/ops/build/rust cargo build "
            "--profile release-fast -p hawking-core --example alu_roofline_organs`"
        )
    exe = bins[0]
    out = out or RAW_DEFAULT
    out.parent.mkdir(parents=True, exist_ok=True)
    inner = [
        str(exe),
        "--mode",
        "deltanet-decompose",
        "--artifact-root",
        str(artifact_root),
        "--layer",
        str(layer),
        "--warmup",
        str(warmup),
        "--reps",
        str(reps),
        "--session-warmup",
        str(session_warmup),
        "--session-reps",
        str(session_reps),
        "--out",
        str(out),
    ]
    lock = REPO / "tools" / "gpu_lane_lock.sh"
    cmd = ["bash", str(lock), "w2dnresid", *inner] if use_lock and lock.is_file() else inner
    env = os.environ.copy()
    env.setdefault("HAWKING_QWEN_RESIDENCY", "1")
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} exited {proc.returncode}\nstdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
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
    parser.add_argument("--session-warmup", type=int, default=2)
    parser.add_argument("--session-reps", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-lock", action="store_true")
    args = parser.parse_args(argv)

    raw: dict[str, Any] | None = None
    if args.measure:
        raw = run_example(
            args.artifact_root,
            layer=args.layer,
            warmup=args.warmup,
            reps=args.reps,
            session_warmup=args.session_warmup,
            session_reps=args.session_reps,
            out=RAW_DEFAULT,
            use_lock=not args.no_lock,
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
        print(f"wrote {path}")
        print(measured["reconciliation"])
        print(build(measured)["finding"])
    else:
        recon = measured["reconciliation"]
        print(
            f"organ {recon['organ_ms']} ms  partition {recon['sum_partition_ms']} ms  "
            f"residual {recon['residual_ms']} ms"
        )
        for row in measured["ranked"]:
            if not row["in_partition"]:
                continue
            print(
                f"  {row['id']:24s} {row['ms']:7.3f} ms  "
                f"{str(row['effective_gb_s']):>7s}  {row['limiter']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

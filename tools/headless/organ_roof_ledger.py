#!/usr/bin/env python3
"""N027 — three roofs + ORGAN_ROOF_LEDGER, ranked by recoverable token_ns.

S022 §1: DEVICE_THEORETICAL, DEVICE_MEASURED_SUSTAINED, and MODEL_REACHABLE
stay three numbers. Never collapse them. §11: rank organs by recoverable
token_ns, not %BW. §12: the goal is COMPLETE_TOKEN_NS, not GB/s.

CPU analysis over sealed receipts. Does not re-measure the DRAM roof, does
not load a second 27B, does not write under ~/models, does not mutate
NOETIC_PARENT_A.

    python3 tools/headless/organ_roof_ledger.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dispatch_ledger import occupancy_snapshot  # noqa: E402
from first_noetic_executable import git_head, now_iso  # noqa: E402
from noetic_operation_census import G143_COMPUTE_PEAK_GFLOPS  # noqa: E402
from organ_bandwidth import ORGANS, map_ledger_row  # noqa: E402

SCHEMA = "hawking.headless.organ_roof_ledger.v1"
RECEIPT = REPO / "receipts" / "headless" / "ORGAN_ROOF_LEDGER.json"
ORGAN_BW = REPO / "receipts" / "headless" / "ORGAN_BANDWIDTH.json"
LEDGER = REPO / "receipts" / "headless" / "DISPATCH_LEDGER.json"
GPU_LEDGER = REPO / "receipts" / "headless" / "GPU_LEDGER.json"
ROOF = REPO / "receipts" / "headless" / "BANDWIDTH_ROOF.json"
TRAFFIC = REPO / "receipts" / "headless" / "NOETIC_TRAFFIC_MODEL.json"

MEASURED = "MEASURED"
DERIVED = "DERIVED"
ABSENT = "ABSENT"
CITED = "CITED"

GPU_CORES = 60
# Below this many in-flight threadgroups the DRAM roof is not reachable.
# GPU_LEDGER's gate_proj launch is 8704 TGs (145/core) and is labelled
# bandwidth-saturated. 4 TGs/core = 240 TGs is the saturating bar used here.
SATURATING_TG_PER_CORE = 4
SATURATING_TGS = GPU_CORES * SATURATING_TG_PER_CORE  # 240

# geo_tpr64_tg128: ceil(rows/2) threadgroups (qwen38_hybrid_decode.rs GeoTpr64Tg128.launch).
HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
QKVZ_ROWS = 16384
BA_ROWS = 96
Q_PROJ_ROWS = 12288
KV_PROJ_ROWS = 1024
O_PROJ_ROWS = 5120
LIN_KEY_HEADS = 16
LIN_VALUE_HEADS = 48
LIN_KEY_DIM = 128
LIN_VALUE_DIM = 128
GQA_HEADS = 24
GQA_HEAD_DIM = 256
RMSNORM_TG = 1024  # HAWKING_RMSNORM_TG default
DN_RMSNORM_TG = 256  # HAWKING_DN_RMSNORM_TG default
ROPE_TG = 256  # HAWKING_ROPE_TG default
MHA_TG = 512  # HAWKING_MHA_TG default
RESIDUAL_TG = 256  # kernels/mod.rs TG_SIZE for qwen_next_add_residual
EMBED_TG = 256
ARGMAX_TG = 256  # sample_argmax_f32_tcb; two-pass is default off


def qty(
    value,
    *,
    kind: str,
    unit: str,
    command: str,
    note: str | None = None,
    absent_reason: str | None = None,
    source: str | None = None,
):
    if kind == ABSENT:
        if value is not None:
            raise ValueError(f"ABSENT quantity must not carry a value ({command})")
        if not absent_reason:
            raise ValueError(f"ABSENT quantity needs a physical reason ({command})")
        out = {
            "value": None,
            "kind": ABSENT,
            "unit": unit,
            "command": command,
            "absent_reason": absent_reason,
        }
        if note:
            out["note"] = note
        if source:
            out["source"] = source
        return out
    if value is None:
        raise ValueError(f"{kind} quantity has no value ({command})")
    out = {
        "value": value,
        "kind": kind,
        "unit": unit,
        "command": command,
        "absent_reason": None,
    }
    if note:
        out["note"] = note
    if source:
        out["source"] = source
    return out


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"FAIL: required input missing: {path}")
    return json.loads(path.read_text())


def geo_tpr64_tgs(rows: int) -> int:
    return (rows + 1) // 2


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def threadgroups_for(operator: str, kernel: str) -> tuple[int, str]:
    """Launch geometry in threadgroups. Cited from decode/kernels, not a counter."""
    k = kernel or ""
    if operator == "embed_lookup":
        return ceil_div(HIDDEN, EMBED_TG), (
            "qwen_uniform_q4_embedding_lookup grid=(hidden,1,1) tg=256 → 20 TGs "
            "(qwen38_hybrid_decode.rs::encode_embed)"
        )
    if operator == "argmax" or k == "sample_argmax_f32":
        return 1, (
            "sample_argmax_f32_tcb grid=(256,1,1) tg=256 → 1 TG. Two-pass "
            "(ARGMAX_GROUPS=240) is default off and did not transfer to the token."
        )
    if "gate_up_swiglu" in operator or "gate_up" in k:
        return geo_tpr64_tgs(INTERMEDIATE), (
            "geo_tpr64_tg128 ceil(17408/2)=8704 TGs (GPU_LEDGER occupancy_launch_geometry.gate_proj)"
        )
    if operator == "down_proj":
        return geo_tpr64_tgs(HIDDEN), "geo_tpr64_tg128 ceil(5120/2)=2560 TGs"
    if operator in ("out_proj", "o_proj"):
        return geo_tpr64_tgs(O_PROJ_ROWS), "geo_tpr64_tg128 ceil(5120/2)=2560 TGs"
    if operator == "dn_qkvz_ba_concat":
        rows = QKVZ_ROWS + BA_ROWS
        return geo_tpr64_tgs(rows), f"geo_tpr64_tg128 pair-concat ceil({rows}/2) TGs"
    if operator == "gqa_qkv_concat":
        rows = Q_PROJ_ROWS + KV_PROJ_ROWS + KV_PROJ_ROWS
        return geo_tpr64_tgs(rows), f"geo_tpr64_tg128 fused QKV ceil({rows}/2) TGs"
    if operator == "lm_head":
        return geo_tpr64_tgs(VOCAB), (
            "geo_tpr64_tg128 ceil(248320/2)=124160 TGs "
            "(GPU_LEDGER occupancy_launch_geometry.lm_head)"
        )
    if operator == "gated_delta" or "gated_delta_decode_vi" in k:
        tgs = LIN_KEY_DIM * LIN_VALUE_HEADS  # (kd, heads, vd)/(kd,1,1) → heads*vd
        return tgs, (
            "qwen38_gated_delta_decode_vi_simd grid=(kd,heads,vd)=(128,48,128) "
            "tg=(128,1,1) → 6144 TGs (encode_gated_delta)"
        )
    if operator in ("input_rmsnorm", "post_attention_rmsnorm", "final_rmsnorm") or k == (
        "qwen80_residual_rmsnorm_tg"
    ):
        return 1, (
            f"qwen80_residual_rmsnorm_tg grid=tg={RMSNORM_TG} → 1 TG "
            "(HAWKING_RMSNORM_TG default 1024)"
        )
    if operator == "gated_rmsnorm" or "deltanet_gated_rmsnorm" in k:
        return LIN_VALUE_HEADS, (
            f"qwen80_deltanet_gated_rmsnorm_tg grid=(48*{DN_RMSNORM_TG}) tg={DN_RMSNORM_TG} "
            "→ 48 TGs"
        )
    if operator in ("mixer_residual", "mlp_residual") or k == "qwen_next_add_residual":
        return ceil_div(HIDDEN, RESIDUAL_TG), (
            "qwen_next_add_residual grid=(hidden,) tg=256 → 20 TGs"
        )
    if operator == "qkvz_rearrange_conv_l2" or "qkvz_rearrange" in k:
        return LIN_KEY_HEADS, (
            "qwen38_qkvz_rearrange_conv_l2_f32 grid=(256, key_heads=16, 1) tg=256 → 16 TGs"
        )
    if operator == "ba_to_decay_beta" or "ba_to_decay" in k:
        return ceil_div(LIN_VALUE_HEADS, 16), (
            "qwen80_ba_to_decay_beta_f32 grid=(48,) tg=16 → 3 TGs"
        )
    if operator == "qk_norm_rope_cache" or "qk_norm_rope" in k:
        return GQA_HEADS, (
            f"qwen38_gqa_qk_norm_rope_cache_tg grid=(24*{ROPE_TG}) tg={ROPE_TG} → 24 TGs"
        )
    if operator == "mha_decode" or k == "mha_decode_f32":
        return GQA_HEADS, (
            f"mha_decode_f32 grid=(n_heads*{MHA_TG}) tg={MHA_TG} → 24 TGs "
            "(HAWKING_MHA_TG default 512; kernels/mod.rs)"
        )
    if operator == "sigmoid_gate" or "apply_sigmoid_gate" in k:
        return ceil_div(GQA_HEADS * GQA_HEAD_DIM, 256), (
            "qwen38_attention_apply_sigmoid_gate grid=(24*256,) tg=256 → 24 TGs"
        )
    return 1, f"unlisted kernel {k!r}; 1 TG assumed (occupancy not a hardware counter)"


def occupancy_factor(threadgroups: int) -> float:
    if threadgroups <= 0:
        return 1.0 / SATURATING_TGS
    return min(1.0, threadgroups / SATURATING_TGS)


def sealed_three_roofs(roof: dict[str, Any]) -> tuple[float, float, float]:
    """Copy 819 and 778.8. Do not re-derive the DRAM roof from the sweep."""
    peak = (roof.get("hardware") or {}).get("published_peak_gb_s")
    if not isinstance(peak, (int, float)) or peak <= 0:
        peak = (roof.get("answer") or {}).get("published_peak_gb_s")
    if not isinstance(peak, (int, float)) or peak <= 0:
        raise SystemExit("FAIL: BANDWIDTH_ROOF.json missing published_peak_gb_s")
    corr = ((roof.get("anchor_roof") or {}).get("correction") or {})
    sustained = corr.get("new_roof_gb_s")
    if not isinstance(sustained, (int, float)) or sustained <= 0:
        raise SystemExit(
            "FAIL: BANDWIDTH_ROOF.json missing anchor_roof.correction.new_roof_gb_s "
            "(sealed 778.8). N027 does not re-derive this from highest_dram_read_gb_s."
        )
    compute = float(G143_COMPUTE_PEAK_GFLOPS)
    if TRAFFIC.is_file():
        try:
            tdoc = json.loads(TRAFFIC.read_text())
            cited = (tdoc.get("machine") or {}).get("compute_peak_gflops")
            if isinstance(cited, (int, float)) and cited > 0:
                compute = float(cited)
        except json.JSONDecodeError:
            pass
    return float(peak), float(sustained), compute


def aggregate_ledger(dispatches: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {
        name: {
            "n": 0,
            "weight_read": 0,
            "activation_read": 0,
            "activation_write": 0,
            "traffic_bytes": 0,
            "flops": 0.0,
            "launch_overhead_ns": 0.0,
            "memory_traffic_ns_at_roof": 0.0,
            "synchronization_ns": 0.0,
            "operators": [],
            "kernels": [],
            "fusion": [],
            "dependencies": [],
            "byte_weighted_occ_num": 0.0,
            "byte_weighted_occ_den": 0.0,
            "min_threadgroups": None,
            "max_threadgroups": 0,
            "dominant_kernel_tgs": None,
            "dominant_kernel_bytes": -1,
            "launch_geometry": [],
        }
        for name in ORGANS
    }
    for row in dispatches:
        name = map_ledger_row(row)
        b = row.get("bytes") or {}
        traffic = int(b.get("total") or 0)
        weight = int(b.get("weight_read") or 0)
        act_r = int(b.get("activation_read") or 0)
        act_w = int(b.get("activation_write") or 0)
        flops = float(row.get("flops") or 0.0)
        launch = row.get("launch_overhead_ns") or {}
        launch_v = float(launch.get("value") or 0.0) if launch.get("kind") != ABSENT else 0.0
        mem_ns = float(row.get("memory_traffic_ns_at_roof") or 0.0)
        sync_ns = float(row.get("synchronization_ns") or 0.0)
        op = row.get("operator") or ""
        kernel = row.get("kernel") or ""
        tgs, why = threadgroups_for(op, kernel)
        occ = occupancy_factor(tgs)
        slot = buckets[name]
        slot["n"] += 1
        slot["weight_read"] += weight
        slot["activation_read"] += act_r
        slot["activation_write"] += act_w
        slot["traffic_bytes"] += traffic
        slot["flops"] += flops
        slot["launch_overhead_ns"] += launch_v
        slot["memory_traffic_ns_at_roof"] += mem_ns
        slot["synchronization_ns"] += sync_ns
        if op and op not in slot["operators"]:
            slot["operators"].append(op)
        if kernel and kernel not in slot["kernels"]:
            slot["kernels"].append(kernel)
        cand = row.get("fusion_candidacy")
        if cand:
            slot["fusion"].append(cand)
        for dep in row.get("dependencies") or []:
            if dep not in slot["dependencies"]:
                slot["dependencies"].append(dep)
        slot["byte_weighted_occ_num"] += traffic * occ
        slot["byte_weighted_occ_den"] += traffic
        if slot["min_threadgroups"] is None:
            slot["min_threadgroups"] = tgs
        else:
            slot["min_threadgroups"] = min(slot["min_threadgroups"], tgs)
        slot["max_threadgroups"] = max(slot["max_threadgroups"], tgs)
        if traffic >= slot["dominant_kernel_bytes"]:
            slot["dominant_kernel_bytes"] = traffic
            slot["dominant_kernel_tgs"] = tgs
        slot["launch_geometry"].append(
            {
                "operator": op,
                "kernel": kernel,
                "threadgroups": tgs,
                "occupancy_factor": occ,
                "traffic_bytes": traffic,
                "why": why,
            }
        )
    return buckets


def bound_class(ai: float | None, ridge: float) -> str:
    if ai is None:
        return "launch"
    if ai > ridge:
        return "compute"
    return "memory"


def roofs_for_organ(
    *,
    traffic_bytes: int,
    flops: float,
    occupancy: float,
    peak_gb_s: float,
    sustained_gb_s: float,
    compute_gflops: float,
) -> dict[str, Any]:
    """AI + occupancy roofline. Dependency: 1 serial CB, no claimed overlap."""
    ai = (flops / traffic_bytes) if traffic_bytes > 0 and flops > 0 else None
    ridge_theo = compute_gflops / peak_gb_s
    ridge_meas = compute_gflops / sustained_gb_s
    cls = bound_class(ai, ridge_meas)
    if ai is None or ai == 0:
        compute_limited_gb_s = None
    else:
        compute_limited_gb_s = compute_gflops / ai
    occ = occupancy if occupancy > 0 else (1.0 / SATURATING_TGS)
    mem_theo = peak_gb_s
    mem_meas = sustained_gb_s
    mem_reach = sustained_gb_s * occ
    if compute_limited_gb_s is None:
        theo_gb = mem_theo
        meas_gb = mem_meas
        reach_gb = mem_reach
    else:
        theo_gb = min(mem_theo, compute_limited_gb_s)
        meas_gb = min(mem_meas, compute_limited_gb_s)
        reach_gb = min(mem_reach, compute_limited_gb_s)
    # ns at each roof. Compute occupancy-scales the GFLOP peak the same way.
    def t_at(bw: float, flop_scale: float) -> float:
        t_m = (traffic_bytes / (bw * 1e9)) * 1e9 if bw > 0 and traffic_bytes else 0.0
        t_c = (
            (flops / (compute_gflops * flop_scale * 1e9)) * 1e9
            if flops > 0 and flop_scale > 0
            else 0.0
        )
        return max(t_m, t_c)

    theo_ns = t_at(theo_gb, 1.0)
    meas_ns = t_at(meas_gb, 1.0)
    reach_ns = t_at(reach_gb, occ)
    return {
        "arithmetic_intensity_flop_per_byte": ai,
        "ridge_theoretical_flop_per_byte": ridge_theo,
        "ridge_measured_flop_per_byte": ridge_meas,
        "bound_class": cls,
        "compute_limited_gb_s": compute_limited_gb_s,
        "occupancy_factor": occ,
        "organ_theoretical_gb_s": theo_gb,
        "organ_measured_device_gb_s": meas_gb,
        "organ_reachable_gb_s": reach_gb,
        "organ_theoretical_ns": theo_ns,
        "organ_measured_device_ns": meas_ns,
        "organ_reachable_ns": reach_ns,
    }


def fusion_summary(labels: list[str]) -> dict[str, Any]:
    n_cand = sum(1 for x in labels if isinstance(x, str) and x.startswith("candidate:"))
    n_fused = sum(1 for x in labels if isinstance(x, str) and x.startswith("already_fused:"))
    n_not = sum(1 for x in labels if isinstance(x, str) and x.startswith("not_a_candidate"))
    return {
        "n": len(labels),
        "n_candidate": n_cand,
        "n_already_fused": n_fused,
        "n_not_a_candidate": n_not,
        "labels_unique": sorted(set(labels)),
    }


def method_text(peak: float, sustained: float, compute: float) -> dict[str, Any]:
    ridge_m = compute / sustained
    ridge_t = compute / peak
    return {
        "statement": (
            "MODEL_REACHABLE is derived per organ from arithmetic intensity, "
            "launch-geometry occupancy, and the 1-CB serial dependency graph. "
            "A memory-bound organ can in principle reach DEVICE_MEASURED_SUSTAINED "
            f"({sustained} GB/s) only if it issues enough threadgroups to saturate "
            "DRAM; a compute-bound organ is capped at COMPUTE_PEAK/AI, which is "
            "below the DRAM roof. Occupancy-starved organs (1–24 TGs) cannot reach "
            "either DRAM roof. Organs are encoded serially in one command buffer "
            "(concurrent_independent=false); sibling overlap is not claimed."
        ),
        "formula": (
            "AI = FLOPs / traffic_bytes. "
            f"ridge_measured = {compute:g} GFLOP/s / {sustained} GB/s "
            f"= {ridge_m:.3f} FLOP/byte; "
            f"ridge_theoretical = {compute:g} / {peak} = {ridge_t:.3f} FLOP/byte. "
            "bound = compute if AI > ridge_measured else memory (launch if FLOPs=0). "
            "occupancy_factor = min(1, threadgroups / (60 cores × 4 TG/core)). "
            "Byte-weighted across the organ's operators. "
            "organ_reachable_gb_s = min(DEVICE_MEASURED_SUSTAINED × occupancy_factor, "
            "COMPUTE_PEAK/AI). "
            "organ_reachable_ns = max(traffic_bytes / reachable_bw, "
            "FLOPs / (COMPUTE_PEAK × occupancy_factor)). "
            "recoverable_token_ns = max(0, measured_gpu_ns − organ_reachable_ns). "
            "Device MODEL_REACHABLE_GB_S = parent_active_bytes / sum(organ_reachable_ns)."
        ),
        "why_not_pct_bw": (
            "Sampling and embedding sit at ~0% of 778.8 GB/s because they are "
            "occupancy-starved (1–20 TGs), not because they hold recoverable "
            "token_ns. mlp_gate_up at 361 GB/s (46% of 778.8) holds ~5.3 ms. "
            "Ranking by %BW would invert the doctor. S022 §11, §12."
        ),
        "why_three_roofs": (
            "819 is datasheet peak (DEVICE_THEORETICAL). 778.8 is the N017 "
            "sequential unique-once DRAM measurement (DEVICE_MEASURED_SUSTAINED), "
            "copied from BANDWIDTH_ROOF.json and not re-derived. MODEL_REACHABLE "
            "is the roof THIS graph can hit given AI + occupancy + 1-CB serial "
            "structure, and is a third number. Collapsing them Goodharts GB/s "
            "(S022 §1)."
        ),
        "dependency_structure": (
            "Production TokenCommandBuffer is one command buffer, one host wait, "
            "concurrent_independent=false. Intra-CB dispatches are encoded in "
            "layer order; ba_to_decay is a sibling of rearrange on paper and a "
            "serial launch in the graph. Overlap is unmeasured "
            "(atDispatchBoundary=false) and is not subtracted from reachable ns."
        ),
        "occupancy_source": (
            "Hardware occupancy / SIMD utilization ABSENT (GPU_LEDGER "
            "MTLDevice.counterSets = {timestamp}). Launch geometry is DERIVED "
            "from qwen38_hybrid_decode.rs / kernels/mod.rs dispatch_threads."
        ),
        "saturating_threadgroups": SATURATING_TGS,
        "saturating_tg_per_core": SATURATING_TG_PER_CORE,
        "gpu_cores": GPU_CORES,
        "compute_peak_gflops": compute,
        "compute_peak_source": (
            "NOETIC_TRAFFIC_MODEL.machine.compute_peak_gflops / "
            "G143_FLOPS_PER_TOKEN.json; not re-derived"
        ),
    }


def one_line(
    peak: float,
    sustained: float,
    model_gb: float | None,
    ranked: list[dict[str, Any]],
) -> str:
    top = ", ".join(
        f"{r['organ']} {r['recoverable_token_ns']/1e6:.2f}ms"
        for r in ranked[:3]
    )
    mg = f"{model_gb:.1f}" if model_gb is not None else "ABSENT"
    return (
        f"Three roofs stay separate: DEVICE_THEORETICAL {peak:g} GB/s, "
        f"DEVICE_MEASURED_SUSTAINED {sustained:g} GB/s (BANDWIDTH_ROOF, not "
        f"re-derived), MODEL_REACHABLE {mg} GB/s (AI+occupancy+1-CB). "
        f"Ranked by recoverable token_ns: {top}. Not ranked by %BW."
    )


def build() -> dict[str, Any]:
    organ_doc = load_json(ORGAN_BW)
    ledger_doc = load_json(LEDGER)
    gpu_doc = load_json(GPU_LEDGER)
    roof_doc = load_json(ROOF)
    peak, sustained, compute = sealed_three_roofs(roof_doc)
    raw_dram = (roof_doc.get("answer") or {}).get("highest_dram_read_gb_s")
    att = organ_doc.get("organ_attribution") or {}
    if att.get("kind") != MEASURED:
        raise SystemExit(
            "FAIL: ORGAN_BANDWIDTH.json organ_attribution is not MEASURED; "
            "N027 reuses N025 and does not re-measure organs"
        )
    organs_att = att.get("organs") or {}
    missing = [n for n in ORGANS if n not in organs_att]
    if missing:
        raise SystemExit(f"FAIL: ORGAN_BANDWIDTH missing organs {missing}")
    agg = aggregate_ledger(ledger_doc.get("dispatches") or [])
    parent_active = att.get("production_active_bytes")
    if not isinstance(parent_active, (int, float)) or parent_active <= 0:
        parent_active = organ_doc.get("prior_not_rederived", {}).get(
            "parent_active_bytes_per_token"
        )
    production_gpu_ns = att.get("production_gpu_ns_median")
    production_gb_s = att.get("production_achieved_gb_s")

    gpu_fields = gpu_doc.get("fields") or {}
    hw_occ = gpu_fields.get("hardware_occupancy_counter") or gpu_fields.get(
        "SIMD_utilization"
    )
    per_disp = gpu_fields.get("per_dispatch_gpu_ns")
    queue_wait = gpu_fields.get("GPU_QUEUE_WAIT_NS") or {}
    host_encode = gpu_fields.get("host_encode_ns") or {}
    complete_q4 = gpu_fields.get("COMPLETE_TOKEN_WALL_NS") or {}
    sync_count = gpu_fields.get("synchronization_count") or {}

    occ_snap = occupancy_snapshot()

    organ_rows: dict[str, Any] = {}
    ranked_src: list[dict[str, Any]] = []
    for name in ORGANS:
        a = agg[name]
        meas = organs_att[name]
        measured_ns = float(meas["scaled_gpu_ns"])
        occ = (
            a["byte_weighted_occ_num"] / a["byte_weighted_occ_den"]
            if a["byte_weighted_occ_den"] > 0
            else occupancy_factor(a["dominant_kernel_tgs"] or 1)
        )
        roofs = roofs_for_organ(
            traffic_bytes=a["traffic_bytes"],
            flops=a["flops"],
            occupancy=occ,
            peak_gb_s=peak,
            sustained_gb_s=sustained,
            compute_gflops=compute,
        )
        recov = max(0.0, measured_ns - roofs["organ_reachable_ns"])
        achieved_gb_s = meas.get("achieved_gb_s")
        traffic_gb_s = (
            a["traffic_bytes"] / measured_ns if measured_ns > 0 else None
        )
        measured_gflops = (
            a["flops"] / measured_ns if measured_ns > 0 and a["flops"] > 0 else None
        )
        pct_bw = (
            None
            if not isinstance(achieved_gb_s, (int, float)) or sustained <= 0
            else achieved_gb_s / sustained
        )
        flops_qty = (
            qty(
                a["flops"],
                kind=DERIVED,
                unit="FLOP/token",
                command="sum(DISPATCH_LEDGER.dispatches.flops) grouped by organ",
                source="receipts/headless/DISPATCH_LEDGER.json",
            )
            if a["flops"] > 0
            else qty(
                None,
                kind=ABSENT,
                unit="FLOP/token",
                command="DISPATCH_LEDGER.dispatches.flops",
                absent_reason=(
                    f"{name} records 0 FLOPs in DISPATCH_LEDGER (argmax comparison "
                    "count was not inventoried). AI and compute-limited BW are "
                    "therefore launch-class, not a fake zero GFLOP/s."
                    if name == "sampling"
                    else f"{name} records 0 FLOPs in DISPATCH_LEDGER"
                ),
                source="receipts/headless/DISPATCH_LEDGER.json",
            )
        )
        ai_val = roofs["arithmetic_intensity_flop_per_byte"]
        ai_qty = (
            qty(
                ai_val,
                kind=DERIVED,
                unit="FLOP/byte",
                command="FLOPs / traffic_bytes",
            )
            if ai_val is not None
            else qty(
                None,
                kind=ABSENT,
                unit="FLOP/byte",
                command="FLOPs / traffic_bytes",
                absent_reason="FLOPs ABSENT or zero; arithmetic intensity not a number",
            )
        )
        organ_rows[name] = {
            "organ": name,
            "bytes": {
                "weight_read": qty(
                    a["weight_read"],
                    kind=CITED,
                    unit="bytes/token",
                    command="DISPATCH_LEDGER + ORGAN_BANDWIDTH.organ_bytes_from_dispatch_ledger",
                    source="receipts/headless/DISPATCH_LEDGER.json",
                ),
                "activation_read": a["activation_read"],
                "activation_write": a["activation_write"],
                "traffic_bytes": qty(
                    a["traffic_bytes"],
                    kind=CITED,
                    unit="bytes/token",
                    command="sum(weight_read+activation_read+activation_write)",
                    source="receipts/headless/DISPATCH_LEDGER.json",
                    note="Roofline ns uses traffic, not weight-only.",
                ),
            },
            "flops": flops_qty,
            "arithmetic_intensity": ai_qty,
            "bound_class": roofs["bound_class"],
            "measured_bw": {
                "weight_gb_s": qty(
                    achieved_gb_s,
                    kind=CITED,
                    unit="GB/s",
                    command="ORGAN_BANDWIDTH.organ_attribution.organs.*.achieved_gb_s",
                    source="receipts/headless/ORGAN_BANDWIDTH.json",
                    note="weight_read / scaled_gpu_ns. N025, not re-measured.",
                ),
                "traffic_gb_s": qty(
                    traffic_gb_s,
                    kind=DERIVED,
                    unit="GB/s",
                    command="traffic_bytes / scaled_gpu_ns",
                ),
                "pct_of_device_measured_sustained": qty(
                    pct_bw,
                    kind=DERIVED,
                    unit="fraction",
                    command="weight_gb_s / DEVICE_MEASURED_SUSTAINED",
                    note="Diagnostic only. NOT the ranking quantity (S022 §11).",
                )
                if pct_bw is not None
                else qty(
                    None,
                    kind=ABSENT,
                    unit="fraction",
                    command="weight_gb_s / 778.8",
                    absent_reason="achieved_gb_s missing",
                ),
            },
            "measured_compute": qty(
                measured_gflops,
                kind=DERIVED,
                unit="GFLOP/s",
                command="FLOPs / scaled_gpu_ns",
            )
            if measured_gflops is not None
            else qty(
                None,
                kind=ABSENT,
                unit="GFLOP/s",
                command="FLOPs / scaled_gpu_ns",
                absent_reason="FLOPs ABSENT; compute throughput not a number",
            ),
            "dispatch_sync": {
                "n_dispatches": qty(
                    a["n"],
                    kind=CITED,
                    unit="dispatches/token",
                    command="count(DISPATCH_LEDGER.dispatches) for organ",
                    source="receipts/headless/DISPATCH_LEDGER.json",
                ),
                "launch_overhead_ns": qty(
                    a["launch_overhead_ns"],
                    kind=DERIVED,
                    unit="ns/token",
                    command=(
                        "sum(isolated_family/n − traffic/778.8). Diagnostic. "
                        "Production per_dispatch_gpu_ns is ABSENT."
                    ),
                    source="receipts/headless/DISPATCH_LEDGER.json",
                    note=(
                        "Isolated TOKEN_NS families, not a production-CB split. "
                        "Not added on top of organ_reachable_ns (that would double-count)."
                    ),
                ),
                "intra_cb_synchronization_ns": qty(
                    a["synchronization_ns"],
                    kind=CITED,
                    unit="ns/token",
                    command="DISPATCH_LEDGER.dispatches.synchronization_ns",
                    note="0 for every intra-CB dispatch. One wait_until_completed per token.",
                    source="receipts/headless/DISPATCH_LEDGER.json",
                ),
                "per_dispatch_gpu_ns": qty(
                    None,
                    kind=ABSENT,
                    unit="ns/dispatch",
                    command="MTLCounterSampleBuffer atDispatchBoundary",
                    absent_reason=(
                        (per_disp or {}).get("absent_reason")
                        or (
                            "atDispatchBoundary sampling is unsupported. Production is "
                            "one mixed CB; intra-CB dispatch GPU timestamps cannot be split."
                        )
                    ),
                    source="receipts/headless/GPU_LEDGER.json",
                ),
            },
            "occupancy": {
                "hardware_counter": qty(
                    None,
                    kind=ABSENT,
                    unit="fraction",
                    command="MTLDevice.counterSets",
                    absent_reason=(
                        (hw_occ or {}).get("absent_reason")
                        or (
                            "No SIMD-utilization or occupancy counter in "
                            "MTLDevice.counterSets (sets=['timestamp'])."
                        )
                    ),
                    source="receipts/headless/GPU_LEDGER.json",
                ),
                "launch_geometry_factor": qty(
                    occ,
                    kind=DERIVED,
                    unit="fraction",
                    command="byte-weighted min(1, TGs / (60*4))",
                    note=(
                        f"min_tgs={a['min_threadgroups']} max_tgs={a['max_threadgroups']} "
                        f"dominant_tgs={a['dominant_kernel_tgs']}. Not a hardware counter. "
                        f"Saturating bar = {SATURATING_TGS} TGs."
                    ),
                ),
                "class": (
                    "bandwidth_saturated"
                    if occ >= 0.9
                    else "occupancy_starved"
                    if occ < 0.25
                    else "partial"
                ),
                "threadgroups_min": a["min_threadgroups"],
                "threadgroups_max": a["max_threadgroups"],
                "threadgroups_dominant": a["dominant_kernel_tgs"],
            },
            "complete_ns": {
                "gpu_ns": qty(
                    measured_ns,
                    kind=CITED,
                    unit="ns/token",
                    command="ORGAN_BANDWIDTH.organ_attribution.organs.*.scaled_gpu_ns",
                    source="receipts/headless/ORGAN_BANDWIDTH.json",
                    note=(
                        "Isolated organ CB GPUEnd−GPUStart, scaled onto production GPU ns. "
                        "N025, not re-measured."
                    ),
                ),
                "complete_wall_ns": qty(
                    None,
                    kind=ABSENT,
                    unit="ns/token",
                    command="per-organ complete-wall",
                    absent_reason=(
                        "Production is one mixed command buffer. Complete-wall "
                        "(encode+submit+wait+epilogue) cannot be split onto organs; "
                        "atDispatchBoundary=false. Token-level complete-wall lives "
                        "under token.complete_ns, not here."
                    ),
                ),
            },
            "limits": {
                "theoretical_gb_s": qty(
                    roofs["organ_theoretical_gb_s"],
                    kind=DERIVED,
                    unit="GB/s",
                    command="min(DEVICE_THEORETICAL, COMPUTE_PEAK/AI)",
                    note="Device theoretical applied to this organ's AI. Occupancy not applied.",
                ),
                "measured_device_gb_s": qty(
                    roofs["organ_measured_device_gb_s"],
                    kind=DERIVED,
                    unit="GB/s",
                    command="min(DEVICE_MEASURED_SUSTAINED, COMPUTE_PEAK/AI)",
                    note="778.8 copied from BANDWIDTH_ROOF; not re-derived.",
                ),
                "reachable_gb_s": qty(
                    roofs["organ_reachable_gb_s"],
                    kind=DERIVED,
                    unit="GB/s",
                    command="min(DEVICE_MEASURED_SUSTAINED×occupancy_factor, COMPUTE_PEAK/AI)",
                ),
                "theoretical_ns": qty(
                    roofs["organ_theoretical_ns"],
                    kind=DERIVED,
                    unit="ns",
                    command="max(traffic/819e9, FLOPs/8979e9)×1e9",
                ),
                "measured_device_ns": qty(
                    roofs["organ_measured_device_ns"],
                    kind=DERIVED,
                    unit="ns",
                    command="max(traffic/778.8e9, FLOPs/8979e9)×1e9",
                ),
                "reachable_ns": qty(
                    roofs["organ_reachable_ns"],
                    kind=DERIVED,
                    unit="ns",
                    command="max(traffic/reachable_bw, FLOPs/(8979e9×occupancy))",
                ),
            },
            "recoverable_token_ns": qty(
                recov,
                kind=DERIVED,
                unit="ns/token",
                command="max(0, scaled_gpu_ns − organ_reachable_ns)",
                note="Ranking quantity (S022 §11). Not %BW, not fraction of 778.8.",
            ),
            "operators": a["operators"],
            "kernels": a["kernels"],
            "fusion": fusion_summary(a["fusion"]),
            "dependencies": a["dependencies"],
            "dense_w_materialized": 0,
            "ridge_measured_flop_per_byte": roofs["ridge_measured_flop_per_byte"],
            "ridge_theoretical_flop_per_byte": roofs["ridge_theoretical_flop_per_byte"],
        }
        ranked_src.append(
            {
                "organ": name,
                "recoverable_token_ns": recov,
                "measured_gpu_ns": measured_ns,
                "reachable_ns": roofs["organ_reachable_ns"],
                "achieved_gb_s": achieved_gb_s,
                "reachable_gb_s": roofs["organ_reachable_gb_s"],
                "bound_class": roofs["bound_class"],
                "occupancy_factor": occ,
                "pct_of_device_measured_sustained": pct_bw,
            }
        )

    ranked_recov = sorted(
        ranked_src, key=lambda r: (-r["recoverable_token_ns"], r["organ"])
    )
    for i, r in enumerate(ranked_recov, 1):
        r["rank"] = i
    ranked_pct = sorted(
        ranked_src,
        key=lambda r: (
            r["pct_of_device_measured_sustained"]
            if r["pct_of_device_measured_sustained"] is not None
            else 1e9,
            r["organ"],
        ),
    )
    for i, r in enumerate(ranked_pct, 1):
        r["rank_if_pct_bw"] = i

    sum_reach = sum(r["reachable_ns"] for r in ranked_src)
    sum_recov = sum(r["recoverable_token_ns"] for r in ranked_src)
    model_gb = (
        float(parent_active) / sum_reach if sum_reach > 0 and parent_active else None
    )
    model_traffic_gb = (
        sum(agg[n]["traffic_bytes"] for n in ORGANS) / sum_reach if sum_reach > 0 else None
    )

    parent_tok_s = (ledger_doc.get("parent") or {}).get("tok_s_sealed")
    parent_wall = (1e9 / parent_tok_s) if isinstance(parent_tok_s, (int, float)) and parent_tok_s else None

    line = one_line(peak, sustained, model_gb, ranked_recov)
    method = method_text(peak, sustained, compute)

    three = {
        "DEVICE_THEORETICAL": qty(
            peak,
            kind=CITED,
            unit="GB/s",
            command="BANDWIDTH_ROOF.hardware.published_peak_gb_s",
            source="receipts/headless/BANDWIDTH_ROOF.json",
            note="Datasheet peak. Not a measured decode roof. S022 §1.",
        ),
        "DEVICE_MEASURED_SUSTAINED": qty(
            sustained,
            kind=CITED,
            unit="GB/s",
            command="BANDWIDTH_ROOF.anchor_roof.correction.new_roof_gb_s",
            source="receipts/headless/BANDWIDTH_ROOF.json",
            note=(
                "N017 sequential unique-once DRAM roof, sealed as 778.8. "
                "Copied, not re-derived. Raw highest_dram_read_gb_s="
                f"{raw_dram} is the measurement that sealed this figure."
            ),
        ),
        "MODEL_REACHABLE": qty(
            model_gb,
            kind=DERIVED,
            unit="GB/s",
            command="parent_active_bytes / sum(organ_reachable_ns)",
            note=(
                "Roof THIS token graph can hit given AI + occupancy + 1-CB serial "
                "structure, expressed as active-weight GB/s so it sits next to "
                "production 358.2 without collapsing onto 778.8 or 819."
            ),
        )
        if model_gb is not None
        else qty(
            None,
            kind=ABSENT,
            unit="GB/s",
            command="parent_active_bytes / sum(organ_reachable_ns)",
            absent_reason="organ reachable ns or parent active bytes missing",
        ),
        "never_collapsed": True,
        "not_a_single_number": True,
        "s022_section": 1,
        "distinct": True,
    }

    return {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "obligation": (
            "N027 — THREE_ROOFS + ORGAN_ROOF_LEDGER (S022 §1, §11, §12): "
            "rank organs by recoverable token_ns, not %BW"
        ),
        "one_line": line,
        "question": (
            "What are the three roofs, what MODEL_REACHABLE can each organ hit, "
            "and which organs hold recoverable token_ns?"
        ),
        "answer": line,
        "ranking_quantity": "recoverable_token_ns",
        "not_the_ranking_quantity": [
            "pct_of_778p8",
            "pct_BW",
            "achieved_gb_s",
            "fraction_of_roof_gap",
        ],
        "s022": {
            "section_1": "three roofs never collapsed",
            "section_11": "rank by recoverable token_ns (ORGAN_ROOF_LEDGER)",
            "section_12": "the goal is COMPLETE_TOKEN_NS, not GB/s",
        },
        "three_roofs": three,
        "model_reachable_method": method,
        "compute_peak_gflops": qty(
            compute,
            kind=CITED,
            unit="GFLOP/s",
            command="NOETIC_TRAFFIC_MODEL.machine.compute_peak_gflops",
            source="receipts/headless/NOETIC_TRAFFIC_MODEL.json",
            note="G143; not re-derived.",
        ),
        "ridge": {
            "theoretical_flop_per_byte": compute / peak,
            "measured_flop_per_byte": compute / sustained,
            "formula": "COMPUTE_PEAK_GFLOPS / roof_GB_s",
        },
        "token": {
            "vehicle": "NOETIC_PARENT_A fused 756 (N025 organ CBs)",
            "production_gpu_ns": qty(
                production_gpu_ns,
                kind=CITED,
                unit="ns/token",
                command="ORGAN_BANDWIDTH.organ_attribution.production_gpu_ns_median",
                source="receipts/headless/ORGAN_BANDWIDTH.json",
            ),
            "production_achieved_gb_s": qty(
                production_gb_s,
                kind=CITED,
                unit="GB/s",
                command="ORGAN_BANDWIDTH.organ_attribution.production_achieved_gb_s",
                source="receipts/headless/ORGAN_BANDWIDTH.json",
            ),
            "parent_active_bytes": qty(
                parent_active,
                kind=CITED,
                unit="bytes/token",
                command="ORGAN_BANDWIDTH.prior_not_rederived.parent_active_bytes_per_token",
                source="receipts/headless/ORGAN_BANDWIDTH.json",
            ),
            "complete_ns": {
                "parent_tok_s_sealed": qty(
                    parent_tok_s,
                    kind=CITED,
                    unit="tok/s",
                    command="DISPATCH_LEDGER.parent.tok_s_sealed",
                    source="receipts/headless/DISPATCH_LEDGER.json",
                )
                if parent_tok_s is not None
                else qty(
                    None,
                    kind=ABSENT,
                    unit="tok/s",
                    command="DISPATCH_LEDGER.parent.tok_s_sealed",
                    absent_reason="parent tok_s_sealed missing",
                ),
                "parent_complete_wall_ns_from_tok_s": qty(
                    parent_wall,
                    kind=DERIVED,
                    unit="ns/token",
                    command="1e9 / DISPATCH_LEDGER.parent.tok_s_sealed",
                )
                if parent_wall is not None
                else qty(
                    None,
                    kind=ABSENT,
                    unit="ns/token",
                    command="1e9 / tok_s_sealed",
                    absent_reason="parent tok_s_sealed missing",
                ),
                "q4_incumbent_complete_wall_ns": qty(
                    complete_q4.get("value"),
                    kind=CITED,
                    unit="ns/token",
                    command="GPU_LEDGER.fields.COMPLETE_TOKEN_WALL_NS",
                    source="receipts/headless/GPU_LEDGER.json",
                    note=(
                        "Different vehicle (q4 incumbent, 964 dispatches). "
                        "Cited, not mixed into parent organ recoverable ns."
                    ),
                )
                if complete_q4.get("value") is not None
                else qty(
                    None,
                    kind=ABSENT,
                    unit="ns/token",
                    command="GPU_LEDGER.fields.COMPLETE_TOKEN_WALL_NS",
                    absent_reason="GPU_LEDGER complete wall missing",
                ),
            },
            "token_sync": {
                "synchronization_count": qty(
                    sync_count.get("value"),
                    kind=CITED,
                    unit="waits/token",
                    command="GPU_LEDGER.fields.synchronization_count",
                    source="receipts/headless/GPU_LEDGER.json",
                    note="One host wait per token. Not split onto organs.",
                )
                if sync_count.get("value") is not None
                else qty(
                    None,
                    kind=ABSENT,
                    unit="waits/token",
                    command="GPU_LEDGER.fields.synchronization_count",
                    absent_reason="GPU_LEDGER synchronization_count missing",
                ),
                "queue_wait_ns": qty(
                    queue_wait.get("value"),
                    kind=CITED,
                    unit="ns/token",
                    command="GPU_LEDGER.fields.GPU_QUEUE_WAIT_NS",
                    source="receipts/headless/GPU_LEDGER.json",
                    note="q4 vehicle. Host wait − GPU interval. Not intra-CB idle.",
                )
                if queue_wait.get("value") is not None
                else qty(
                    None,
                    kind=ABSENT,
                    unit="ns/token",
                    command="GPU_LEDGER.fields.GPU_QUEUE_WAIT_NS",
                    absent_reason="GPU_LEDGER GPU_QUEUE_WAIT_NS missing",
                ),
                "host_encode_ns": qty(
                    host_encode.get("value"),
                    kind=CITED,
                    unit="ns/token",
                    command="GPU_LEDGER.fields.host_encode_ns",
                    source="receipts/headless/GPU_LEDGER.json",
                    note="q4 vehicle. Token ceremony, not an organ.",
                )
                if host_encode.get("value") is not None
                else qty(
                    None,
                    kind=ABSENT,
                    unit="ns/token",
                    command="GPU_LEDGER.fields.host_encode_ns",
                    absent_reason="GPU_LEDGER host_encode_ns missing",
                ),
            },
            "model_reachable_ns": qty(
                sum_reach,
                kind=DERIVED,
                unit="ns/token",
                command="sum(organ_reachable_ns)  # 1 serial CB",
            ),
            "recoverable_token_ns_sum": qty(
                sum_recov,
                kind=DERIVED,
                unit="ns/token",
                command="sum(organ recoverable_token_ns)",
            ),
            "model_reachable_gb_s_on_traffic": qty(
                model_traffic_gb,
                kind=DERIVED,
                unit="GB/s",
                command="sum(traffic_bytes) / sum(organ_reachable_ns)",
                note="Traffic-normalized sibling of three_roofs.MODEL_REACHABLE. Still not 778.8 or 819.",
            )
            if model_traffic_gb is not None
            else qty(
                None,
                kind=ABSENT,
                unit="GB/s",
                command="sum(traffic_bytes) / sum(organ_reachable_ns)",
                absent_reason="reachable ns missing",
            ),
        },
        "organs_named": list(ORGANS),
        "organs": organ_rows,
        "ranked_by_recoverable_token_ns": ranked_recov,
        "ranked_by_pct_of_measured_bw_is_not_the_ranking": {
            "note": (
                "If we ranked by % of 778.8, occupancy-starved sampling/embedding "
                "would lead. That ranking is recorded so a reader can see it was "
                "considered and rejected (S022 §11)."
            ),
            "rows": ranked_pct,
        },
        "largest_recoverable_organ": ranked_recov[0]["organ"] if ranked_recov else None,
        "inputs_reused_not_remeasured": {
            "ORGAN_BANDWIDTH.json": "N025 per-organ GPU ns, achieved GB/s, production GPU ns",
            "DISPATCH_LEDGER.json": "bytes, FLOPs, launch overhead, sync=0, fusion, deps",
            "GPU_LEDGER.json": "hardware occupancy ABSENT, per-dispatch ABSENT, token ceremony",
            "BANDWIDTH_ROOF.json": "DEVICE_THEORETICAL 819, DEVICE_MEASURED_SUSTAINED 778.8",
        },
        "did_not_rederive_dram_roof": True,
        "did_not_load_second_27b": True,
        "did_not_mutate_parent": True,
        "did_not_write_under_models": True,
        "gpu_confirm": {
            "ran": False,
            "optional": True,
            "reason": (
                "N027 is CPU analysis over sealed receipts. A short GPU confirm "
                "is optional and was not required to derive the three roofs or "
                "the recoverable-ns ranking."
            ),
        },
        "occupancy": occ_snap,
        "dense_w_materialized": 0,
        "causal_benchmark_law": {
            "kernel_identity": "none new — N027 does not ship a kernel",
            "sentinel": "three_roofs keys DEVICE_THEORETICAL / DEVICE_MEASURED_SUSTAINED / MODEL_REACHABLE are three distinct numbers",
            "noop_would_not_pass": (
                "A receipt that ranked by %BW, collapsed the three roofs, or "
                "re-derived 778.8 from the sweep would fail."
            ),
            "bad_control": "rank-by-%BW list is recorded and must not equal rank-by-recoverable-ns",
        },
    }


def write_receipt(doc: dict[str, Any]) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=1) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="build the ledger in memory only",
    )
    args = parser.parse_args()
    doc = build()
    if not args.no_write:
        write_receipt(doc)
        print(doc["one_line"])
        print(f"wrote {RECEIPT}")
    else:
        print(doc["one_line"])
    if math.isclose(doc["three_roofs"]["DEVICE_THEORETICAL"]["value"],
                    doc["three_roofs"]["DEVICE_MEASURED_SUSTAINED"]["value"]):
        raise SystemExit("FAIL: theoretical collapsed onto measured")
    mr = doc["three_roofs"]["MODEL_REACHABLE"]["value"]
    if mr is not None and (
        math.isclose(mr, doc["three_roofs"]["DEVICE_THEORETICAL"]["value"])
        or math.isclose(mr, doc["three_roofs"]["DEVICE_MEASURED_SUSTAINED"]["value"])
    ):
        raise SystemExit("FAIL: MODEL_REACHABLE collapsed onto another roof")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

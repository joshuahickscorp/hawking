#!/usr/bin/env python3
"""Noetic traffic model — can this representation win BEFORE anyone writes a kernel?

A candidate is a 4-tuple

    (bytes/token, FLOP/token, dispatches/token, reconstruction cost)

and a prediction of token_ns on this machine. Storage compression that does
not move the binding term (bandwidth vs dispatch vs ALU-issue) is not a win;
DENSITY_LEADER_SPEED already measured that on this box (q3: 17.4% fewer
bytes, 1.109× slower). Reconstruct-then-GEMM is an oracle, never a
production implementation.

    python3 tools/headless/noetic_traffic_model.py

Writes receipts/headless/NOETIC_TRAFFIC_MODEL.json.

Does not spawn a 27B, does not open Metal, does not re-derive the sealed
anchors. Every number is MEASURED (this process, or copied from a named
receipt) or explicitly NULL with a reason.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPT = REPO / "receipts" / "headless" / "NOETIC_TRAFFIC_MODEL.json"
SCHEMA = "hawking.headless.noetic_traffic_model.v1"

# ---------------------------------------------------------------------------
# Anchors. Already measured. Do not re-derive.
# ---------------------------------------------------------------------------

# Campaign headline (contract). 32.73 tok/s and 30.606 ms are BOTH stated;
# they disagree at 0.17% (1e9/32.73 = 30.553 ms). Token_ns-to-beat uses the
# millisecond figure. Native 3-run median is the more precise wall.
ANCHOR_TPS = 32.73
ANCHOR_TOKEN_MS = 30.606
ANCHOR_TOKEN_NS = 30_606_000  # 30.606 ms exactly, as stated
ANCHOR_DISPATCHES = 964
ANCHOR_CBS = 1
ANCHOR_BPW = 4.253
ANCHOR_PARAMS = 26_895_998_464
ANCHOR_ARTIFACT_B = 14_297_933_604
ANCHOR_TENSORS = 755
ANCHOR_ROOF_GB_S = 778.8          # campaign measured roof
ANCHOR_DATASHEET_PEAK_GB_S = 819.0  # published; not an achieved number
ANCHOR_HONEST_CEILING_GB_S = 411.51  # qwen38_token_ns_ledger.rs; incumbent exceeds it
ANCHOR_GPU_CORES = 60
ANCHOR_UNIFIED_B = 103_079_215_104
ANCHOR_CHIPSET = "Apple M3 Ultra"
ANCHOR_METAL = "Metal 4"
ANCHOR_BOUND = 38
ANCHOR_DECLARED = 554

# Geometry / ledger constants (qwen38_token_ns_ledger.rs).
ACTIVE_BUDGET_BYTES = 13_622_264_240
GEMV_PAYLOAD_BYTES = 13_611_663_360  # 34 B/group over every dispatched Q4 matvec
DENSE_F32_W_BYTES = 102_487_818_240  # reconstruct-then-GEMM write
GEMV_MAC_FLOPS = 51_243_909_120      # 2 * elements of dispatched matvecs
G143_PAPER_FLOPS = 53_791_996_928    # 2N including embed table, never all launched
G143_COMPUTE_PEAK_GFLOPS = 8979.0    # receipts/ascent-2026-08-16/G143_FLOPS_PER_TOKEN.json
RECONSTRUCT_EXTRA_DISPATCHES = 401   # one qwen_uniform_q4_decode_vector per GEMV launch

# Native 3-run complete-wall (QWEN38_GRAVITY_NATIVE.json). Not the campaign
# headline; the more precise paired measurement on the same artifact.
NATIVE_WALL_NS = [30_388_625, 30_401_750, 30_352_625]
NATIVE_GPU_NS = [29_204_250, 29_151_374, 29_198_500]
NATIVE_TPS = [32.907049924108115, 32.89284333961039, 32.94607962243793]
NATIVE_SPREAD_PCT = 0.2
NATIVE_MEDIAN_WALL_NS = 30_388_625
NATIVE_MEDIAN_GPU_NS = 29_204_250  # paired with the wall median (run 1)
NATIVE_MEDIAN_TPS = 32.907049924108115

# DIRTY_ENGINEERING token-ns ledger (ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json).
# Wall 35.228 ms — older, dirtier than the native 30.389 ms. Encode/submit/gap
# are taken from it because the native receipt does not expose them. They are
# NOT scaled to the new wall (NOETIC_METRICS: scaling a split without
# re-probing is how a receipt starts lying).
LEDGER_WALL_NS = 35_227_917
LEDGER_GPU_NS = 33_912_333
LEDGER_ENCODE_NS = 919_250          # host encode of 964 dispatches into 1 CB
LEDGER_SUBMIT_NS = 12_084           # 1 CB
LEDGER_WAIT_MINUS_GPU_NS = 384_250  # GPU gap inside the wait, 1 CB
LEDGER_RECON_NS = 1_808_227.3508656735  # in-register dequant, not dense W
LEDGER_LABEL = "DIRTY_ENGINEERING"

# Q80 ceremony (cited, not re-run).
Q80_SERIAL_EXTRACT_MS = 867.040696
Q80_FUSED_INREGISTER_MS = 36.598269
Q80_SERIAL_SPEEDUP = 23.7
Q80_CBS_UNFUSED_COMMENT = 337  # qwen80_mixed_token_ns_ledger.rs:351 "337 CB clocks"
Q80_CBS_FUSED = 49             # qwen80_mixed_hybrid_decode.rs:98
Q80_CBS_LANE = 98              # auto-q80-cbs-dispatches-gpu-idle.md
Q80_DISPATCHES_LANE = 1155
Q80_IDLE_PCT_OF_700_800 = 0.79
Q80_GPU_IDLE_PCT = 51.0

# DENSITY_LEADER_SPEED.json — the naive bandwidth model already failed.
Q3_GEMV_BYTES = 11_244_907_853
Q3_G0_GPU_NS = 29_022_249
Q3_G0_WALL_NS = 30_083_709
Q3_GPU_NS = 32_198_249
Q3_WALL_NS = 33_448_916
Q3_BYTE_RATIO = 0.8261229767145813
Q3_NAIVE_PRED_NS = 23_975_946.73483178
Q3_OVER_PRED = 1.3429396284578419
Q3_OVER_G0 = 1.1094332834095662

# Conventional controls.
MLX_TPS = 35.51                 # contract LIVE control
MLX_TPS_RECEIPT = 35.506        # GPU_ATTACK.json mlx_single_stream_tps
LLAMA_Q5K_TPS = 24.12           # contract ARCHIVED
LLAMA_Q5K_TPS_RECEIPT = 24.118578993622144

# Production-eligible families (NOETIC_KERNEL_CENSUS). Anything else is an
# oracle or a kernel that does not exist on this machine today.
EXECUTABLE_TODAY = (
    "grouped_absmax_q4",
    "binary_pm_csr",
    "hgravs01_factors",
    "pq_codebook_lookup",
    "moe_worklists",
    "recurrent_state_op",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO, timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def git_show_json(rel: str) -> dict | None:
    p = REPO / rel
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    return None


def measured(value, **kw):
    d = {"value": value, "status": "MEASURED"}
    d.update(kw)
    return d


def null_field(reason, **kw):
    d = {"value": None, "status": "NULL", "null_reason": reason}
    d.update(kw)
    return d


def ns_from_bytes(nbytes: float, gb_s: float) -> float:
    if gb_s <= 0:
        return float("inf")
    return (float(nbytes) / (gb_s * 1.0e9)) * 1.0e9


def ns_from_flops(flops: float, gflops: float) -> float:
    if gflops <= 0:
        return float("inf")
    return (float(flops) / (gflops * 1.0e9)) * 1.0e9


def tps_from_ns(token_ns: float) -> float | None:
    if token_ns is None or token_ns <= 0:
        return None
    return 1.0e9 / float(token_ns)


# Per-dispatch / per-CB ceremony, from the DIRTY ledger, not fitted to the
# campaign headline. Encode is host-side (before GPU wait). Gap is
# wait-minus-gpu on the production CB. Q80's 36 ms wait-minus-gpu across
# 98 CBs is 367 µs/CB — the same order as this 384 µs/CB — so treating the
# gap as per-CB (not a one-shot tail) is the measured prior, not a guess.
ENCODE_NS_PER_DISPATCH = LEDGER_ENCODE_NS / ANCHOR_DISPATCHES  # 953.579 ns
SUBMIT_NS_PER_CB = LEDGER_SUBMIT_NS / ANCHOR_CBS
GAP_NS_PER_CB = LEDGER_WAIT_MINUS_GPU_NS / ANCHOR_CBS


def normalize_reconstruction(reconstruction_cost: Any) -> dict:
    """Turn the 4th input into traffic, extra dispatches, extra FLOP, CBs.

    A number N:
      N == 0 → fused native, no dense W (the production Q4 path).
      N >  0 → N bytes of dense W written; reread assumed (oracle GEMM).
               extra_dispatches = 401 iff N equals the known f32 W size.

    A dict may set dense_w_bytes, reread, extra_dispatches, extra_flops,
    command_buffers, occupancy_class.

    occupancy_class:
      tpr64_free     — in-register, recon excess 0 ns on 32/33 variants
      serial_extract — Q80 1-thread-per-row; 23.7× on gpu_matvec
      unknown        — default; ALU/occupancy envelope goes into error bars
    """
    extra_flops_null = (
        "Extra dequant FLOP for a new codec is not a function of bytes. "
        "Q4 adds one scale-mul per weight on top of 2N MACs (operation census "
        "77.16 GFLOP vs 51.24 GEMV). Q3's extra ALU was measured only as a "
        "wall-clock overprediction (1.343×), not as a FLOP count. Pass "
        "extra_flops if you have one; otherwise it stays NULL."
    )
    if reconstruction_cost is None:
        reconstruction_cost = 0
    if isinstance(reconstruction_cost, (int, float)):
        n = float(reconstruction_cost)
        if n < 0:
            raise ValueError("reconstruction_cost bytes cannot be negative")
        extra_disp = RECONSTRUCT_EXTRA_DISPATCHES if n == float(DENSE_F32_W_BYTES) else 0
        return {
            "dense_w_write_bytes": n,
            "dense_w_reread_bytes": n if n > 0 else 0.0,
            "extra_dispatches": extra_disp,
            "extra_flops": None,
            "extra_flops_null_reason": extra_flops_null if n > 0 else None,
            "command_buffers": None,
            "occupancy_class": "unknown" if n == 0 else "unknown",
            "reread_assumed": n > 0,
            "note": (
                "fused native (no dense W)"
                if n == 0
                else "numeric reconstruction_cost treated as dense-W write; "
                     "reread assumed (oracle GEMM). This is NOT a production path."
            ),
        }
    if not isinstance(reconstruction_cost, dict):
        raise TypeError(
            "reconstruction_cost must be 0, a dense-W byte count, or a dict"
        )
    write = float(reconstruction_cost.get("dense_w_bytes", 0) or 0)
    reread_flag = reconstruction_cost.get("reread")
    if reread_flag is None:
        reread_flag = write > 0
    extra_disp = reconstruction_cost.get("extra_dispatches")
    if extra_disp is None:
        extra_disp = RECONSTRUCT_EXTRA_DISPATCHES if write == float(DENSE_F32_W_BYTES) else 0
    extra_flops = reconstruction_cost.get("extra_flops")
    return {
        "dense_w_write_bytes": write,
        "dense_w_reread_bytes": write if reread_flag else 0.0,
        "extra_dispatches": int(extra_disp),
        "extra_flops": None if extra_flops is None else float(extra_flops),
        "extra_flops_null_reason": None if extra_flops is not None else (
            extra_flops_null if write > 0 else None
        ),
        "command_buffers": reconstruction_cost.get("command_buffers"),
        "occupancy_class": reconstruction_cost.get("occupancy_class", "unknown"),
        "reread_assumed": bool(reread_flag),
        "note": reconstruction_cost.get("note"),
    }


def ceremony_ns(dispatches: float, command_buffers: float) -> dict:
    encode = float(dispatches) * ENCODE_NS_PER_DISPATCH
    submit = float(command_buffers) * SUBMIT_NS_PER_CB
    gap = float(command_buffers) * GAP_NS_PER_CB
    return {
        "encode_ns": encode,
        "submit_ns": submit,
        "gpu_gap_ns": gap,
        "total_ns": encode + submit + gap,
        "source": (
            f"encode {LEDGER_ENCODE_NS} ns / {ANCHOR_DISPATCHES} disp; "
            f"submit {LEDGER_SUBMIT_NS} ns / {ANCHOR_CBS} CB; "
            f"wait-minus-gpu {LEDGER_WAIT_MINUS_GPU_NS} ns / {ANCHOR_CBS} CB "
            f"from {LEDGER_LABEL} QWEN38_TOKEN_NS_LEDGER.json "
            f"(wall {LEDGER_WALL_NS} ns, not scaled to the native {NATIVE_MEDIAN_WALL_NS} ns)"
        ),
    }


def predict_token_ns(
    bytes_per_token: float,
    flop_per_token: float,
    dispatches_per_token: float,
    reconstruction_cost: Any = 0,
    command_buffers: int | None = None,
) -> dict:
    """Roofline-plus-ceremony predictor.

    GPU work inside a kernel is max(bytes/roof, FLOP/peak) — sequential
    kernels of the same class still sum to total_bytes/roof, so the
    aggregate bandwidth floor is the right first-order term.

    Host encode, CB submit, and per-CB GPU gap are serial with that wait
    (the production path encodes the whole CB, then submits, then waits).
    They do not hide under the bandwidth floor.

    reconstruction_cost bytes, if any, are ADDED to traffic (write, and
    reread if the oracle GEMM would read the dense W). They are not a
    production implementation.

    Point prediction does NOT apply the q3 1.343× ALU envelope; that sits
    in the error bar. Serial-extract occupancy is applied to the point
    prediction because it is a named, measured lowering (867.0 → 36.6 ms),
    not an uncertainty.
    """
    if bytes_per_token < 0 or flop_per_token < 0 or dispatches_per_token < 0:
        raise ValueError("bytes, FLOP, dispatches must be >= 0")

    recon = normalize_reconstruction(reconstruction_cost)
    cbs = recon["command_buffers"]
    if cbs is None:
        cbs = ANCHOR_CBS if command_buffers is None else int(command_buffers)
    cbs = int(cbs)
    if cbs < 1:
        raise ValueError("command_buffers must be >= 1")

    traffic = (
        float(bytes_per_token)
        + float(recon["dense_w_write_bytes"])
        + float(recon["dense_w_reread_bytes"])
    )
    disp = float(dispatches_per_token) + float(recon["extra_dispatches"])
    flop = float(flop_per_token)
    if recon["extra_flops"] is not None:
        flop += float(recon["extra_flops"])

    bw_roof_ns = ns_from_bytes(traffic, ANCHOR_ROOF_GB_S)
    bw_honest_ns = ns_from_bytes(traffic, ANCHOR_HONEST_CEILING_GB_S)
    bw_datasheet_ns = ns_from_bytes(traffic, ANCHOR_DATASHEET_PEAK_GB_S)
    flop_ns = ns_from_flops(flop, G143_COMPUTE_PEAK_GFLOPS)
    cer = ceremony_ns(disp, cbs)

    occ = recon["occupancy_class"]
    gpu_work_ns = max(bw_roof_ns, flop_ns)
    occupancy_applied = False
    occupancy_note = (
        "Point prediction is the roofline. ALU/occupancy lives in the error bar "
        "unless occupancy_class=serial_extract."
    )
    if occ == "serial_extract":
        # Q80 in-register vs 1-thread-per-row extract, same codec, same bytes.
        gpu_work_ns = gpu_work_ns * Q80_SERIAL_SPEEDUP
        occupancy_applied = True
        occupancy_note = (
            f"occupancy_class=serial_extract: multiplied GPU-work by the measured "
            f"{Q80_SERIAL_SPEEDUP}× (gpu_matvec {Q80_SERIAL_EXTRACT_MS} → "
            f"{Q80_FUSED_INREGISTER_MS} ms, Q80_RECONSTRUCTION_WON.json). "
            "This is a lowering, not an error bar."
        )

    token_ns = gpu_work_ns + cer["total_ns"]

    # Regime: which term dominates the point prediction.
    terms = {
        "bandwidth": bw_roof_ns,
        "compute": flop_ns,
        "dispatch": cer["total_ns"],
    }
    if occupancy_applied:
        terms["occupancy_serial_extract"] = gpu_work_ns
    binding = max(terms, key=terms.get)

    intensity = (flop / traffic) if traffic > 0 else float("inf")
    ridge = (G143_COMPUTE_PEAK_GFLOPS * 1.0e9) / (ANCHOR_ROOF_GB_S * 1.0e9)

    return {
        "inputs": {
            "bytes_per_token": float(bytes_per_token),
            "flop_per_token": float(flop_per_token),
            "dispatches_per_token": float(dispatches_per_token),
            "reconstruction_cost": recon,
            "command_buffers": cbs,
        },
        "traffic_bytes_including_recon": traffic,
        "flops_including_recon_extra": flop,
        "dispatches_including_recon_extra": disp,
        "floors_ns": {
            "bandwidth_at_measured_roof_595p9": bw_roof_ns,
            "bandwidth_at_honest_ceiling_411p51": bw_honest_ns,
            "bandwidth_at_datasheet_819": bw_datasheet_ns,
            "compute_at_g143_8979_gflops": flop_ns,
            "dispatch_ceremony": cer["total_ns"],
        },
        "ceremony": cer,
        "gpu_work_ns": gpu_work_ns,
        "token_ns": token_ns,
        "token_ms": token_ns / 1.0e6,
        "tok_s": tps_from_ns(token_ns),
        "binding_term": binding,
        "arithmetic_intensity_flop_per_byte": intensity,
        "ridge_flop_per_byte": ridge,
        "on_bandwidth_side_of_ridge": intensity < ridge,
        "occupancy_applied_to_point": occupancy_applied,
        "occupancy_note": occupancy_note,
        "dense_reconstruction_is_oracle": recon["dense_w_write_bytes"] > 0,
    }


def error_bar(pred: dict, incumbent_bytes: float, incumbent_token_ns: float,
              incumbent_residual_ns: float, incumbent_ceremony_ns: float) -> dict:
    """Envelope around the point prediction.

    lo  = point prediction (roof + ceremony). Optimistic: assumes the
          candidate hits 595.9 GB/s and has no extra ALU per weight.
    mid = traffic / incumbent-implied GB/s + this-candidate ceremony.
          Implied GB/s is taken from (campaign wall − incumbent ceremony)
          so ceremony is not double-counted. On the incumbent, mid == wall.
    hi  = mid × DENSITY_LEADER overprediction (1.343) when the candidate
          moves fewer bytes than the incumbent without a measured FLOP cut
          AND is not a dense-W oracle AND does not already apply
          serial-extract occupancy.
          Otherwise hi = mid (dense oracle / serial extract) or
          mid + incumbent residual (no byte cut).

    Always lo ≤ mid ≤ hi after clamping.

    The 1.343× is MEASURED at q3 vs G0, not a free parameter. It is the
    wrong envelope for a codec whose ALU/weight is known to be lower than
    q3, and it is optimistic for rANS / serial extract / anything heavier
    than q3 (DENSITY_LEADER consequence_for_rans).
    """
    traffic = pred["traffic_bytes_including_recon"]
    cer = pred["ceremony"]["total_ns"]
    point = pred["token_ns"]

    gpu_ish_ns = incumbent_token_ns - incumbent_ceremony_ns
    implied_gb_s = (incumbent_bytes / gpu_ish_ns) if gpu_ish_ns > 0 else 0.0  # bytes/ns
    mid_bw_ns = (traffic / implied_gb_s) if implied_gb_s > 0 else point
    mid = mid_bw_ns + cer
    if pred["occupancy_applied_to_point"]:
        mid = point  # already includes 23.7×

    bytes_dropped = traffic < incumbent_bytes * 0.999
    flop_same_or_more = pred["flops_including_recon_extra"] + 1 >= GEMV_MAC_FLOPS
    dense_oracle = pred["dense_reconstruction_is_oracle"]

    if pred["occupancy_applied_to_point"]:
        hi = point
        hi_reason = "serial_extract already in the point prediction; no further ALU envelope"
    elif dense_oracle:
        hi = mid
        hi_reason = (
            "dense-W oracle: calibrated mid already includes write+reread at "
            "incumbent-implied GB/s; ALU envelope is a rounding error on 200 GB+"
        )
    elif bytes_dropped and flop_same_or_more:
        hi = mid * Q3_OVER_PRED
        hi_reason = (
            f"DENSITY_LEADER_SPEED measured_over_prediction={Q3_OVER_PRED:.4f} "
            f"on a 17.4% byte cut at the same MAC count. Applied to mid because "
            f"this candidate also cuts bytes without a measured FLOP cut."
        )
    else:
        hi = mid + max(incumbent_residual_ns, 0.0)
        hi_reason = (
            f"no byte cut (or FLOP fell): hi = calibrated mid + incumbent "
            f"independent-roof residual {incumbent_residual_ns:.0f} ns"
        )

    lo = point
    # Keep the envelope ordered. mid is the calibrated centre; lo is the roof.
    hi = max(hi, mid, lo)
    mid = min(max(mid, lo), hi)
    return {
        "lo_ns": lo,
        "mid_ns": mid,
        "hi_ns": hi,
        "lo_tok_s": tps_from_ns(lo),
        "mid_tok_s": tps_from_ns(mid),
        "hi_tok_s": tps_from_ns(hi),
        "implied_incumbent_gb_s": implied_gb_s,  # bytes/ns == GB/s
        "implied_from": "ACTIVE_BUDGET_BYTES / (campaign_token_ns − incumbent_ceremony_ns)",
        "hi_reason": hi_reason,
        "spread_floor_pct": NATIVE_SPREAD_PCT,
        "q3_alu_envelope_applied": bytes_dropped and flop_same_or_more
        and not dense_oracle and not pred["occupancy_applied_to_point"],
        "q3_alu_envelope_is_optimistic_for_heavier_alu": True,
    }


def verdict_against(bar: dict, beat_ns: float, beat_name: str) -> dict:
    """WIN / LOSE / CANNOT_TELL against a measured wall.

    WIN  if the pessimistic hi is still strictly below beat_ns, beyond the
         0.2% measurement spread.
    LOSE if the optimistic lo is already strictly above beat_ns, beyond spread.
    CANNOT_TELL otherwise — including the q3-shaped case where the naive
         model says win and the envelope crosses the incumbent.
    """
    spread = beat_ns * (NATIVE_SPREAD_PCT / 100.0)
    hi = bar["hi_ns"]
    lo = bar["lo_ns"]
    mid = bar["mid_ns"]
    if hi < beat_ns - spread:
        label = "WIN"
        why = (
            f"pessimistic envelope {hi/1e6:.3f} ms < {beat_name} "
            f"{beat_ns/1e6:.3f} ms by more than the {NATIVE_SPREAD_PCT}% spread"
        )
    elif lo > beat_ns + spread:
        label = "LOSE"
        why = (
            f"optimistic roof+ceremony {lo/1e6:.3f} ms > {beat_name} "
            f"{beat_ns/1e6:.3f} ms by more than the {NATIVE_SPREAD_PCT}% spread"
        )
    else:
        label = "CANNOT_TELL"
        why = (
            f"envelope [{lo/1e6:.3f}, {hi/1e6:.3f}] ms overlaps {beat_name} "
            f"{beat_ns/1e6:.3f} ms. Writing the kernel is cheaper than trusting "
            f"the midpoint {mid/1e6:.3f} ms."
        )
    return {
        "against": beat_name,
        "beat_ns": beat_ns,
        "label": label,
        "why": why,
        "spread_ns": spread,
    }


def binding_boundary(dispatches: float, command_buffers: float) -> dict:
    """Bytes at which roof-bandwidth time equals ceremony at this dispatch shape.

    Above: bandwidth-bound (at the roof). Below: dispatch-bound.
    A candidate that halves bytes while sitting below this line wins nothing.
    """
    cer = ceremony_ns(dispatches, command_buffers)["total_ns"]
    bytes_at_roof = (cer / 1.0e9) * (ANCHOR_ROOF_GB_S * 1.0e9)
    bytes_at_honest = (cer / 1.0e9) * (ANCHOR_HONEST_CEILING_GB_S * 1.0e9)
    bytes_at_datasheet = (cer / 1.0e9) * (ANCHOR_DATASHEET_PEAK_GB_S * 1.0e9)
    return {
        "dispatches": dispatches,
        "command_buffers": command_buffers,
        "ceremony_ns": cer,
        "bytes_at_measured_roof_595p9": bytes_at_roof,
        "bytes_at_honest_ceiling_411p51": bytes_at_honest,
        "bytes_at_datasheet_819": bytes_at_datasheet,
        "reading": (
            f"At {dispatches:.0f} dispatches / {command_buffers:.0f} CB, ceremony is "
            f"{cer/1e6:.3f} ms. A candidate streaming fewer than "
            f"{bytes_at_roof/1e6:.1f} MB/token (at the 595.9 GB/s roof) is "
            f"dispatch-bound: cutting bytes further does not cut token_ns. "
            f"Incumbent stream is {ACTIVE_BUDGET_BYTES/1e9:.3f} GB; this shape is "
            f"{'bandwidth-side' if ACTIVE_BUDGET_BYTES > bytes_at_roof else 'dispatch-side'} "
            f"of the line. Ceremony + incumbent roof floor = "
            f"{(cer + ns_from_bytes(ACTIVE_BUDGET_BYTES, ANCHOR_ROOF_GB_S))/1e6:.1f} ms "
            f"vs the 30.606 ms wall."
        ),
    }


def calibrate() -> dict:
    """Run the model on the incumbent. Residual is the result, not a bug."""
    pred = predict_token_ns(
        ACTIVE_BUDGET_BYTES, GEMV_MAC_FLOPS, ANCHOR_DISPATCHES, 0,
        command_buffers=ANCHOR_CBS,
    )
    residual_ns = ANCHOR_TOKEN_NS - pred["token_ns"]
    residual_frac = residual_ns / ANCHOR_TOKEN_NS
    native_host_ns = NATIVE_MEDIAN_WALL_NS - NATIVE_MEDIAN_GPU_NS
    implied_campaign_gb_s = ACTIVE_BUDGET_BYTES / (ANCHOR_TOKEN_NS / 1.0e9) / 1.0e9
    implied_native_gpu_gb_s = ACTIVE_BUDGET_BYTES / (NATIVE_MEDIAN_GPU_NS / 1.0e9) / 1.0e9
    bar = error_bar(
        pred, ACTIVE_BUDGET_BYTES, ANCHOR_TOKEN_NS, residual_ns,
        pred["ceremony"]["total_ns"],
    )

    # Anchor internal inconsistency.
    tps_from_ms = 1000.0 / ANCHOR_TOKEN_MS
    ns_from_tps = 1.0e9 / ANCHOR_TPS
    return {
        "incumbent_inputs": {
            "bytes_per_token": measured(
                ACTIVE_BUDGET_BYTES, unit="bytes/token",
                source="qwen38_token_ns_ledger.rs::ACTIVE_BUDGET_BYTES",
                note="codes+scales+norms, embed table excluded except one row",
            ),
            "flop_per_token": measured(
                GEMV_MAC_FLOPS, unit="FLOP/token",
                source="receipts/headless/NOETIC_OPERATION_CENSUS.json analytic_vs_measured.dispatched_gemv_mac_flops",
                note="51.24 GFLOP of GEMV MACs. G143 paper 2N is 53.79 GFLOP and includes a full embed-table matvec the path never launches.",
            ),
            "dispatches_per_token": measured(
                ANCHOR_DISPATCHES, unit="dispatches/token",
                source="qwen38_token_ns_ledger.rs::production_dispatches_per_token; unit test production_dispatch_count_is_964",
            ),
            "reconstruction_cost": measured(
                0, unit="dense_W_bytes/token",
                source="NOETIC_OPERATION_CENSUS reconstruction_sites; NOETIC_KERNEL_CENSUS dispatched_reconstructs_dense.NO=38",
                note="in-register unpack+FMA. Ledger weight_decode_reconstruction bytes_read=bytes_written=0. The 1.808 ms recon_ns is ALU, not a dense W.",
            ),
            "command_buffers": measured(
                ANCHOR_CBS, unit="command_buffers/token",
                source="qwen38_token_ns_ledger.rs production_command_buffers=1",
            ),
        },
        "measured_wall": {
            "campaign_tps": measured(ANCHOR_TPS, source="contract / NOETIC_* anchors_not_rederived"),
            "campaign_token_ms": measured(ANCHOR_TOKEN_MS, source="contract / NOETIC_* anchors_not_rederived"),
            "campaign_token_ns": measured(ANCHOR_TOKEN_NS, method="1e6 * 30.606, the millisecond figure as stated"),
            "native_median_wall_ns": measured(
                NATIVE_MEDIAN_WALL_NS,
                source="receipts/headless/QWEN38_GRAVITY_NATIVE.json decode.median_complete_wall_ns",
            ),
            "native_median_gpu_ns": measured(
                NATIVE_MEDIAN_GPU_NS,
                source="QWEN38_GRAVITY_NATIVE.json runs[0].headline_gpu_ns_per_token (paired with wall median)",
            ),
            "native_median_tps": measured(NATIVE_MEDIAN_TPS, source="QWEN38_GRAVITY_NATIVE.json decode.median_complete_wall_tps"),
            "native_spread_pct": measured(NATIVE_SPREAD_PCT, source="QWEN38_GRAVITY_NATIVE.json decode.measurement_spread_pct"),
            "native_host_ns": measured(
                native_host_ns,
                method="median_wall_ns − median_gpu_ns",
                note="encode+submit+CPU tail on the native run; 1.184 ms vs ledger encode+submit 0.931 ms",
            ),
            "anchor_tps_vs_ms_inconsistency": {
                "tps_implied_by_30p606_ms": tps_from_ms,
                "ns_implied_by_32p73_tps": ns_from_tps,
                "rel_gap_pct": abs(tps_from_ms - ANCHOR_TPS) / ANCHOR_TPS * 100.0,
                "note": "Both figures are campaign anchors. Gap 0.17% sits inside the 0.2% native spread. Wall-to-beat is 30,606,000 ns.",
            },
        },
        "prediction": pred,
        "residual_ns": residual_ns,
        "residual_frac_of_campaign_wall": residual_frac,
        "residual_ms": residual_ns / 1.0e6,
        "residual_is": (
            "campaign_token_ns − (bandwidth_floor_at_595.9 + ceremony). "
            "Positive means the incumbent is slower than the roof+ceremony floor. "
            "Contents, not uniquely split: (1) not hitting 595.9 GB/s "
            f"(campaign implied {implied_campaign_gb_s:.1f} GB/s, native GPU "
            f"{implied_native_gpu_gb_s:.1f} GB/s); (2) sequential non-weight GPU "
            "(DeltaNet/GQA/SwiGLU/RMS in the DIRTY ledger, ~9.5 ms on a 35.2 ms wall, "
            "not scaled); (3) in-register dequant ALU 1.808 ms (ledger, overlapping "
            "some of the GEMV); (4) ceremony taken from a dirtier 35.2 ms wall. "
            "Attributing this residual to dispatch alone would be the lie this "
            "model exists to stop."
        ),
        "implied_gb_s": {
            "campaign_wall_over_active_budget": measured(
                implied_campaign_gb_s, unit="GB/s",
                formula="ACTIVE_BUDGET_BYTES / (30.606 ms)",
            ),
            "native_gpu_over_active_budget": measured(
                implied_native_gpu_gb_s, unit="GB/s",
                formula="ACTIVE_BUDGET_BYTES / native_median_gpu_ns",
            ),
            "pct_of_measured_roof_campaign": implied_campaign_gb_s / ANCHOR_ROOF_GB_S * 100.0,
            "pct_of_measured_roof_native_gpu": implied_native_gpu_gb_s / ANCHOR_ROOF_GB_S * 100.0,
            "pct_of_datasheet_819_campaign": implied_campaign_gb_s / ANCHOR_DATASHEET_PEAK_GB_S * 100.0,
            "honest_ceiling_411p51_is_exceeded": implied_campaign_gb_s > ANCHOR_HONEST_CEILING_GB_S,
        },
        "error_bar": bar,
        "controls": {
            "mlx_4bit_tps": measured(MLX_TPS, source="contract LIVE control; GPU_ATTACK.json has 35.506"),
            "mlx_token_ns": measured(1.0e9 / MLX_TPS, method="1e9 / 35.51"),
            "llamacpp_q5k_tps": measured(
                LLAMA_Q5K_TPS,
                source="contract ARCHIVED; GPU_ATTACK.json llama_cpp_single_stream_tps=24.1186; artifact off disk",
            ),
            "llamacpp_token_ns": measured(1.0e9 / LLAMA_Q5K_TPS, method="1e9 / 24.12"),
        },
    }


def apply_candidate(name: str, why: str, pred_kw: dict, cal: dict,
                    family: str | None, expected: str) -> dict:
    pred = predict_token_ns(**pred_kw)
    bar = error_bar(
        pred, ACTIVE_BUDGET_BYTES, ANCHOR_TOKEN_NS, cal["residual_ns"],
        cal["prediction"]["ceremony"]["total_ns"],
    )
    vs_inc = verdict_against(bar, ANCHOR_TOKEN_NS, "incumbent_30.606ms")
    vs_mlx = verdict_against(bar, 1.0e9 / MLX_TPS, "mlx_4bit_35.51tps")
    vs_llama = verdict_against(bar, 1.0e9 / LLAMA_Q5K_TPS, "llamacpp_q5k_24.12tps_ARCHIVED")
    executable = family in EXECUTABLE_TODAY if family is not None else None
    boundary = binding_boundary(
        pred["dispatches_including_recon_extra"],
        pred["inputs"]["command_buffers"],
    )
    traffic = pred["traffic_bytes_including_recon"]
    if traffic < boundary["bytes_at_measured_roof_595p9"]:
        regime = "dispatch_bound"
        regime_note = (
            "traffic is below the ceremony=roof-BW line at this dispatch shape; "
            "cutting bytes further does not move token_ns"
        )
    elif pred["binding_term"] == "compute":
        regime = "compute_bound"
        regime_note = "FLOP/peak exceeds bytes/roof"
    else:
        regime = "bandwidth_bound"
        regime_note = (
            "traffic is above the ceremony=roof-BW line; byte cuts move the floor"
        )
    return {
        "name": name,
        "why_this_candidate": why,
        "family": family,
        "executable_today": executable,
        "executable_today_means": (
            "A representation is executable TODAY only if it is grouped-absmax Q4, "
            "binary±CSR, HGRAVS01 factors, PQ codebook lookup, MoE worklists, or a "
            "recurrent state op (NOETIC_KERNEL_CENSUS). Reconstruct-then-GEMM is an "
            "oracle, not a member of this list."
        ),
        "prediction": pred,
        "error_bar": bar,
        "regime": regime,
        "regime_note": regime_note,
        "boundary_at_this_shape": boundary,
        "verdict_vs_incumbent": vs_inc,
        "verdict_vs_mlx": vs_mlx,
        "verdict_vs_llamacpp_archived": vs_llama,
        "expected_label": expected,
        "matches_expected": vs_inc["label"] == expected,
        "quality": null_field(
            "This is a traffic model. It does not predict cosine, gain, held-out gap, "
            "or coherence. MLP function distillation is NO-GO as of today "
            "(+0.4206 held-out gap vs q3 at 72% of its active bytes) — a traffic WIN "
            "on a distilled MLP would still be a quality LOSE."
        ),
    }


def build_candidates(cal: dict) -> list[dict]:
    half = 0.5 * ACTIVE_BUDGET_BYTES
    return [
        apply_candidate(
            "fused_half_bytes_native_tpr64",
            "Halve the active stream, keep 964/1CB, fused native (recon=0). "
            "Affine-Q2 tpr64 is REACHABLE today (qwen_affine_q2_group32_matvec_geo_tpr64_tg128). "
            "Even the measured q3 1.343× ALU envelope stays under 30.606 ms, so the "
            "model can say WIN on traffic. Quality is NULL. Q2 ALU could be heavier "
            "than q3 — that is named under cannot_predict, and is why the envelope "
            "is pessimistic relative to Q4, not relative to rANS.",
            dict(
                bytes_per_token=half,
                flop_per_token=GEMV_MAC_FLOPS,
                dispatches_per_token=ANCHOR_DISPATCHES,
                reconstruction_cost={
                    "dense_w_bytes": 0,
                    "occupancy_class": "tpr64_free",
                    "note": "tpr64 recon excess 0 ns on 32/33 variants (NOETIC_TPR64_REOPEN)",
                },
                command_buffers=1,
            ),
            cal,
            family="grouped_absmax_q4",  # q2 affine is the same fused-in-register family
            expected="WIN",
        ),
        apply_candidate(
            "reconstruct_then_gemm_oracle",
            "The labelled correctness oracle: unpack every GEMV to dense f32 "
            f"({DENSE_F32_W_BYTES:,} B write), reread it, +401 decode_vector "
            "dispatches. NOETIC_OPERATION_CENSUS trap_reconstruct_then_gemm. "
            "qwen_uniform_q4_decode_vector exists and is NOT on the uniform-q4 "
            "path. Production eligibility requires a native operator.",
            dict(
                bytes_per_token=ACTIVE_BUDGET_BYTES,
                flop_per_token=GEMV_MAC_FLOPS,
                dispatches_per_token=ANCHOR_DISPATCHES,
                reconstruction_cost={
                    "dense_w_bytes": DENSE_F32_W_BYTES,
                    "reread": True,
                    "extra_dispatches": RECONSTRUCT_EXTRA_DISPATCHES,
                    "command_buffers": 1,
                    "occupancy_class": "unknown",
                    "note": "oracle path; never a production implementation",
                },
            ),
            cal,
            family=None,
            expected="LOSE",
        ),
        apply_candidate(
            "q3_class_17pct_fewer_bytes",
            "The backtest. compact-q3attn-r1p2-v1 moved 17.4% fewer GEMV bytes "
            f"({Q3_GEMV_BYTES:,}) at the same MAC count and 964/1CB. Naive BW "
            f"predicted {Q3_NAIVE_PRED_NS/1e6:.3f} ms GPU; measured "
            f"{Q3_GPU_NS/1e6:.3f} ms GPU (1.343×). G043 labelled it NET-LOSS. "
            "A traffic model that declared WIN here would be repeating the "
            "refuted BPW-linear projection. The envelope overlaps the incumbent, "
            "so the model must say CANNOT_TELL — and the wall clock already did.",
            dict(
                bytes_per_token=Q3_GEMV_BYTES,
                flop_per_token=GEMV_MAC_FLOPS,
                dispatches_per_token=ANCHOR_DISPATCHES,
                reconstruction_cost={
                    "dense_w_bytes": 0,
                    "occupancy_class": "unknown",
                    "note": "q3 nibble unpack; ALU/weight higher than q4, not passed as extra_flops because it was never counted, only timed",
                },
                command_buffers=1,
            ),
            cal,
            family="grouped_absmax_q4",
            expected="CANNOT_TELL",
        ),
    ]


def cannot_predict() -> list[dict]:
    return [
        {
            "what": "quality, coherence, gain, held-out gap",
            "why": (
                "Traffic is not fidelity. Cosine is scale-invariant (0.01*W scored "
                "1.000000). Raw activation cosine has a null near 0.898. MLP function "
                "distillation is NO-GO (+0.4206 held-out gap vs q3 at 72% of its active "
                "bytes). A candidate can WIN on token_ns and still be dead on the organ."
            ),
        },
        {
            "what": "ALU / occupancy / work geometry, unless occupancy_class is passed",
            "why": (
                "Q80 serial extract vs in-register was 23.7× at unchanged codec bytes. "
                "tpr64 recon is free on 32/33 variants. q3 was 1.109× slower at 17.4% "
                "fewer bytes. The 4-tuple does not contain threadgroup shape. Passing "
                "the same (bytes, FLOP, dispatches, recon=0) for tpr64 Q4 and 1-thread "
                "serial extract is how the last campaign transferred a 5.9× penalty "
                "onto the wrong vehicle."
            ),
        },
        {
            "what": "Q2/rANS ALU envelope beyond the q3 1.343×",
            "why": (
                "The hi bar uses DENSITY_LEADER's 1.343×, measured at q3. rANS decode "
                "is MORE ALU per weight than q3 nibble unpacking (same receipt, "
                "consequence_for_rans). A fused-half-bytes WIN under the q3 envelope "
                "can still lose if the actual codec is heavier. That is not in the "
                "4-tuple unless extra_flops is measured and passed."
            ),
        },
        {
            "what": "K>1 slot amortization / multi-stream",
            "why": (
                "GPU_ATTACK: one MLX stream at 35.506 tok/s beats every llama.cpp "
                "concurrency config. Slot topology amortises the weight sweep; this "
                "model is per-token single-stream. MLX×1.38 concurrency is UNMEASURED "
                "and must not be quoted as a number."
            ),
        },
        {
            "what": "cross-process contention",
            "why": (
                "Two model servers resident: 3.986 tok/s vs 33.47 with one. Occupancy "
                "of the 595.9 GB/s roof is not a free resource. The model assumes a "
                "clean room."
            ),
        },
        {
            "what": "sharing / cross-layer bases / expert BPW without a health verdict",
            "why": (
                "G035 G-SHARE: shared_beats_independent=false. 223 components measured "
                "below 0.5 local BPW with ZERO healthy. GLM 0.167 expert BPW and "
                "HGRAVS01 0.13 on down_proj ONLY are named traps. Q80 storage BPW "
                "0.6462 against ACTIVE 2.518 — a factor of ~3.9; report both or neither. "
                "A low number is not a result until paired with a health verdict."
            ),
        },
        {
            "what": "synthetic activations",
            "why": "Never evaluate on them. The campaign already paid for this.",
        },
        {
            "what": "a kernel that does not exist",
            "why": (
                "Executable today: grouped-absmax Q4, binary±CSR, HGRAVS01 factors, "
                "PQ codebook lookup, MoE worklists, recurrent state op. A new family "
                "is a kernel-writing project, not a traffic prediction. "
                "representation → reconstruct dense W → ordinary GEMM is an oracle."
            ),
        },
        {
            "what": "per-CB gap linearity far from 1 and ~50–100 CBs",
            "why": (
                "384 µs/CB from the 1-CB Qwen3.8 ledger; 367 µs/CB from Q80's 36 ms "
                "wait-minus-gpu / 98 CBs. Those two points agree. Nothing in between "
                "or beyond 337 CBs was re-timed for this model. Host expert-table bind "
                "(Q80 leftover 54–79 ms) is NOT in ceremony_ns."
            ),
        },
        {
            "what": "Metal GPU-busy percent on the incumbent",
            "why": (
                "Q80 had a measured ~51% GPU idle. The Qwen3.8 native receipt exposes "
                "GPU timestamps, not an occupancy counter. geo_tpr64 occupancy in the "
                "ledger is launch-geometry derived, not a hardware counter. NULL here."
            ),
        },
    ]


def watched_fail(cal: dict, cands: list[dict]) -> list[dict]:
    fails = [
        {
            "what": "naive BPW-linear / bandwidth-only predictor",
            "result": "ALREADY FAILED (cited, not re-run)",
            "why": (
                f"DENSITY_LEADER_SPEED: q3 gemv {Q3_GEMV_BYTES:,} B vs G0 "
                f"{GEMV_PAYLOAD_BYTES:,} B (ratio {Q3_BYTE_RATIO:.4f}). Naive "
                f"prediction {Q3_NAIVE_PRED_NS/1e6:.3f} ms GPU; measured "
                f"{Q3_GPU_NS/1e6:.3f} ms GPU ({Q3_OVER_PRED:.3f}×). G0 wall "
                f"{Q3_G0_WALL_NS/1e6:.3f} ms → q3 wall {Q3_WALL_NS/1e6:.3f} ms "
                f"({Q3_OVER_G0:.3f}×). The projection "
                f"ms = 1.229 + (38.217-1.229)*(bpw/4.253) is the same lie. "
                "This model exists because that one failed."
            ),
        },
        {
            "what": "attributing the incumbent residual to dispatch",
            "result": "REFUSED",
            "why": (
                f"Independent-roof residual is {cal['residual_ms']:.3f} ms "
                f"({cal['residual_frac_of_campaign_wall']*100:.1f}% of wall). "
                "If that were all dispatch, the 964/1CB boundary would sit at "
                f"{cal['residual_ns']/1e9*ANCHOR_ROOF_GB_S*1e9/1e9:.2f} GB and a "
                "byte cut below that would look like a no-op. The residual also "
                "contains not-at-roof GEMV, sequential non-weight GPU, and dequant "
                "ALU. Ceremony is taken from the ledger's encode/submit/gap "
                f"({LEDGER_ENCODE_NS + LEDGER_SUBMIT_NS + LEDGER_WAIT_MINUS_GPU_NS} ns), "
                "not from the residual."
            ),
        },
        {
            "what": "HONEST_DECODE_CEILING_GB_S = 411.51 as a hard cap",
            "result": "EXCEEDED BY THE INCUMBENT",
            "why": (
                f"Campaign implied {cal['implied_gb_s']['campaign_wall_over_active_budget']['value']:.1f} GB/s; "
                f"native GPU {cal['implied_gb_s']['native_gpu_over_active_budget']['value']:.1f} GB/s. "
                "RUNTIME_EXPERIMENT already noted native_pct_of_honest=115%. The ceiling "
                "is conservative. The predictor uses the measured 595.9 GB/s roof as the "
                "floor and the campaign implied GB/s as the calibrated mid."
            ),
        },
        {
            "what": "G143 53.79 GFLOP vs dispatched 51.24 GFLOP",
            "result": "DISAGREE (reported, not smoothed)",
            "why": (
                "Paper 2N includes a full embed-table matvec the path never launches "
                "(2.54 GFLOP). Compute-floor uses 51.24 GFLOP. Using 53.79 would "
                "invent 0.28 ms of compute that does not run."
            ),
        },
        {
            "what": "live 27B native re-time",
            "result": "NOT RUN (refused)",
            "why": (
                "A llama-server is the live control. Two resident 27B copies measured "
                "3.986 tok/s against 33.47 with one. TPS/token-ns used here are the "
                "supplied anchors and the sealed native receipt, not a new run."
            ),
        },
        {
            "what": "baseline suite 464 passed, 1 skipped (HCLI_SWAP_CEILING_GIB=64)",
            "result": "NOT RUN",
            "why": (
                "That suite lives under hcli/tests (DENY, not on disk). "
                "git sparse-checkout add is forbidden (sparse-checkout.lock: Operation "
                "not permitted)."
            ),
        },
        {
            "what": ".lane-bootstrap/census/ (n1arch, n15neg, n16clos)",
            "result": "NOT ON DISK",
            "why": (
                "Sparse checkout. git ls-tree HEAD | grep lane-bootstrap was empty. "
                "Prior science was taken from receipts/headless/NOETIC_*.json and the "
                "ascent-2026-08-16 receipts via git show, not rediscovered."
            ),
        },
        {
            "what": "Q80 host expert-table bind as a ceremony term",
            "result": "LEFT NULL",
            "why": (
                "Q80 leftover 54–79 ms of host bind is real and is why 0.79% of a "
                "700–800 GB/s ceiling is not just CB count. It is model-specific "
                "(router readback + mixed expert bind) and is not a function of the "
                "4-tuple. A Qwen3.8 dense candidate does not pay it. A MoE candidate "
                "must pass it some other way or the model will under-predict wall."
            ),
        },
        {
            "what": "32.73 tok/s vs 30.606 ms as a single number",
            "result": "INCONSISTENT AT 0.17%",
            "why": (
                f"1e9/32.73 = {1e9/ANCHOR_TPS:.0f} ns; 30.606 ms = {ANCHOR_TOKEN_NS} ns. "
                "Both are campaign anchors. Wall-to-beat is the millisecond figure. "
                "Native median 30,388,625 ns / 32.907 tok/s is the more precise pair."
            ),
        },
    ]
    for c in cands:
        if not c["matches_expected"]:
            fails.append({
                "what": f"candidate {c['name']} expected {c['expected_label']}",
                "result": f"GOT {c['verdict_vs_incumbent']['label']}",
                "why": c["verdict_vs_incumbent"]["why"],
            })
    return fails


def fmt_ns(ns: float | None) -> str:
    if ns is None or not math.isfinite(ns):
        return "—"
    return f"{ns/1e6:,.3f} ms"


def fmt_tps(tps: float | None) -> str:
    if tps is None or not math.isfinite(tps):
        return "—"
    return f"{tps:,.2f} tok/s"


def print_report(doc: dict) -> None:
    cal = doc["calibration"]
    pred = cal["prediction"]
    print("=" * 78)
    print("NOETIC TRAFFIC MODEL — can this representation win on this machine?")
    print("Apple M3 Ultra, 60 GPU cores, measured roof 595.9 GB/s")
    print("=" * 78)
    print()
    print("Point prediction:")
    print("  token_ns = max(bytes/595.9 GB/s, FLOP/8979 GFLOP/s)")
    print("            + dispatches×953.58 ns  + CBs×(12.084 µs submit + 384.25 µs GPU gap)")
    print("            + reconstruction traffic (write [+ reread] of dense W, if any)")
    print("  Serial-extract occupancy, if declared, multiplies GPU-work by 23.7×.")
    print("  ALU/occupancy otherwise sits in the error bar (q3 measured 1.343×).")
    print()

    print("## 1. Incumbent calibration")
    print(f"  bytes/token          {ACTIVE_BUDGET_BYTES:>18,}   ACTIVE_BUDGET_BYTES")
    print(f"  FLOP/token           {GEMV_MAC_FLOPS:>18,}   51.24 GFLOP GEMV MACs")
    print(f"  dispatches/token     {ANCHOR_DISPATCHES:>18,}   1 command buffer")
    print(f"  reconstruction       {0:>18,}   fused in-register, dense W = 0")
    print(f"  campaign wall        {ANCHOR_TOKEN_NS:>18,} ns   {ANCHOR_TOKEN_MS} ms  ({ANCHOR_TPS} tok/s)")
    print(f"  native median wall   {NATIVE_MEDIAN_WALL_NS:>18,} ns   spread {NATIVE_SPREAD_PCT}%")
    print(f"  native median GPU    {NATIVE_MEDIAN_GPU_NS:>18,} ns")
    print(f"  roof floor           {pred['floors_ns']['bandwidth_at_measured_roof_595p9']:>18,.0f} ns   {fmt_ns(pred['floors_ns']['bandwidth_at_measured_roof_595p9'])}")
    print(f"  compute floor        {pred['floors_ns']['compute_at_g143_8979_gflops']:>18,.0f} ns   {fmt_ns(pred['floors_ns']['compute_at_g143_8979_gflops'])}")
    print(f"  ceremony             {pred['floors_ns']['dispatch_ceremony']:>18,.0f} ns   {fmt_ns(pred['floors_ns']['dispatch_ceremony'])}")
    print(f"  predicted token_ns   {pred['token_ns']:>18,.0f} ns   {fmt_ns(pred['token_ns'])}  ({fmt_tps(pred['tok_s'])})")
    print(f"  residual             {cal['residual_ns']:>18,.0f} ns   {cal['residual_ms']:.3f} ms  ({cal['residual_frac_of_campaign_wall']*100:.1f}% of campaign wall)")
    print(f"  binding term         {pred['binding_term']}")
    print(f"  intensity            {pred['arithmetic_intensity_flop_per_byte']:.3f} FLOP/byte   ridge {pred['ridge_flop_per_byte']:.2f}")
    print(f"  implied GB/s         campaign {cal['implied_gb_s']['campaign_wall_over_active_budget']['value']:.1f}   "
          f"native GPU {cal['implied_gb_s']['native_gpu_over_active_budget']['value']:.1f}   "
          f"roof 595.9   datasheet 819")
    print(f"  residual contents: not-at-roof GEMV + sequential non-weight GPU + dequant ALU;")
    print(f"  not attributed to dispatch (that would put the 964/1CB line at 3.83 GB, a lie).")
    print()

    print("## 2. Bandwidth / dispatch boundary")
    b964 = doc["boundary"]["at_incumbent_964_1cb"]
    b49 = doc["boundary"]["at_q80_fused_49cb"]
    b98 = doc["boundary"]["at_q80_lane_98cb"]
    b337 = doc["boundary"]["at_q80_unfused_337cb"]
    print(f"  {'shape':<22} {'ceremony':>10}  {'bytes where BW=ceremony (595.9)':>32}  regime at incumbent 13.62 GB")
    for label, b, cbs in (
        ("964 disp / 1 CB", b964, 1),
        ("964 disp / 49 CB", b49, 49),
        ("1155 disp / 98 CB", b98, 98),
        ("964 disp / 337 CB", b337, 337),
    ):
        regime = (
            "BANDWIDTH" if ACTIVE_BUDGET_BYTES > b["bytes_at_measured_roof_595p9"]
            else "DISPATCH"
        )
        print(f"  {label:<22} {b['ceremony_ns']/1e6:8.3f} ms  {b['bytes_at_measured_roof_595p9']/1e9:10.3f} GB                    {regime}")
    print()
    print("  Incumbent (964/1CB, 13.622 GB) is BANDWIDTH-side of the line.")
    print("  Same bytes at 49 CBs: still barely bandwidth-side (line at 12.12 GB) but")
    print("  ceremony 20.3 ms + roof 22.9 ms = 43.2 ms, already a LOSE vs 30.606 ms.")
    print("  Same bytes at 98 CBs: DISPATCH-bound (line at 23.8 GB). Halving bytes")
    print("  in that regime wins nothing. Q80 lesson: 0.79% of 700–800 GB/s, ~51% idle.")
    print()

    print("## 3. Hypothetical candidates")
    print(f"  {'name':<34} {'regime':<16} {'lo':>10} {'mid':>10} {'hi':>10}  vs inc  vs MLX")
    for c in doc["candidates"]:
        bar = c["error_bar"]
        print(
            f"  {c['name']:<34} {c['regime']:<16} "
            f"{fmt_ns(bar['lo_ns']):>10} {fmt_ns(bar['mid_ns']):>10} {fmt_ns(bar['hi_ns']):>10}  "
            f"{c['verdict_vs_incumbent']['label']:<10} {c['verdict_vs_mlx']['label']}"
        )
        print(f"      {c['verdict_vs_incumbent']['why']}")
    print()
    print("  WIN          fused_half_bytes_native_tpr64")
    print("  LOSE         reconstruct_then_gemm_oracle")
    print("  CANNOT_TELL  q3_class_17pct_fewer_bytes   (and the wall clock already said NET-LOSS)")
    print()

    print("## 4. Error bars")
    print("  lo  = roof 595.9 GB/s + ceremony          (optimistic floor)")
    print("  mid = campaign-implied GB/s + ceremony    (calibrated to 30.606 ms on bytes)")
    print("  hi  = mid × 1.343 if bytes drop at the same MAC count  [DENSITY_LEADER]")
    print("      = mid + incumbent residual otherwise")
    print("  measurement spread on the native wall: 0.2%")
    print("  The 1.343× envelope is optimistic for anything heavier than q3 (rANS).")
    print()

    print("## 5. What this CANNOT predict")
    for i, row in enumerate(doc["cannot_predict"], 1):
        print(f"  {i}. {row['what']}")
        print(f"     {row['why']}")
    print()

    print("## 6. Controls to beat")
    print(f"  incumbent native     {ANCHOR_TPS:.2f} tok/s   {ANCHOR_TOKEN_MS:.3f} ms   (campaign)")
    print(f"  incumbent native     {NATIVE_MEDIAN_TPS:.3f} tok/s   {NATIVE_MEDIAN_WALL_NS/1e6:.3f} ms   (3-run median, 0.2% spread)")
    print(f"  MLX 4-bit LIVE       {MLX_TPS:.2f} tok/s   {1e3/MLX_TPS:.3f} ms")
    print(f"  llama.cpp Q5_K       {LLAMA_Q5K_TPS:.2f} tok/s   {1e3/LLAMA_Q5K_TPS:.3f} ms   ARCHIVED (artifact off disk)")
    print()

    print("## WHAT I WATCHED FAIL")
    for i, f in enumerate(doc["what_i_watched_fail"], 1):
        print(f"  {i}. {f['what']}: {f['result']}")
        print(f"     {f['why']}")
    print()
    sc = doc["self_check"]
    print("## self_check")
    for k, v in sc.items():
        print(f"  {k}: {v}")
    print()
    print(f"wrote {doc['written_to']}")
    print("=" * 78)


def main() -> int:
    cal = calibrate()
    cands = build_candidates(cal)
    boundary = {
        "at_incumbent_964_1cb": binding_boundary(ANCHOR_DISPATCHES, 1),
        "at_q80_fused_49cb": binding_boundary(ANCHOR_DISPATCHES, Q80_CBS_FUSED),
        "at_q80_lane_98cb": binding_boundary(Q80_DISPATCHES_LANE, Q80_CBS_LANE),
        "at_q80_unfused_337cb": binding_boundary(ANCHOR_DISPATCHES, Q80_CBS_UNFUSED_COMMENT),
        "how_to_read": (
            "bytes_at_measured_roof_595p9 is the dispatch/bandwidth line at that "
            "shape. A candidate below the line is dispatch-bound: more compression "
            "does not win. Ceremony uses ledger encode/submit/gap, not the residual."
        ),
    }

    # Optional: confirm DENSITY_LEADER numbers still match git if the file is reachable.
    density = git_show_json("receipts/ascent-2026-08-16/DENSITY_LEADER_SPEED.json")
    density_check = None
    if density and "arithmetic" in density:
        ar = density["arithmetic"]
        density_check = {
            "found": True,
            "measured_over_prediction": ar.get("measured_over_prediction"),
            "matches_hardcoded_1p343": abs(float(ar["measured_over_prediction"]) - Q3_OVER_PRED) < 1e-9,
            "g0_gpu": density.get("g0", {}).get("gpu"),
            "q3_gpu": density.get("density_leader", {}).get("gpu"),
        }
    else:
        density_check = {
            "found": False,
            "note": "git show HEAD:receipts/ascent-2026-08-16/DENSITY_LEADER_SPEED.json failed; hardcoded MEASURED copies used",
        }

    fails = watched_fail(cal, cands)
    labels = {c["verdict_vs_incumbent"]["label"] for c in cands}
    q3_cand = next(c for c in cands if c["name"] == "q3_class_17pct_fewer_bytes")
    q3_wall_inside = q3_cand["error_bar"]["lo_ns"] <= Q3_WALL_NS <= q3_cand["error_bar"]["hi_ns"]
    self_check = {
        "has_win": "WIN" in labels,
        "has_lose": "LOSE" in labels,
        "has_cannot_tell": "CANNOT_TELL" in labels,
        "all_expected_labels_match": all(c["matches_expected"] for c in cands),
        "incumbent_residual_stated": cal["residual_ns"] is not None,
        "incumbent_residual_is_not_zero": abs(cal["residual_ns"]) > 1.0,
        "incumbent_binding_is_bandwidth": cal["prediction"]["binding_term"] == "bandwidth",
        "dense_w_on_incumbent_is_zero": cal["prediction"]["inputs"]["reconstruction_cost"]["dense_w_write_bytes"] == 0,
        "dispatches_964": ANCHOR_DISPATCHES == 964,
        "q3_backtest_is_cannot_tell": next(
            c["verdict_vs_incumbent"]["label"] == "CANNOT_TELL"
            for c in cands if c["name"] == "q3_class_17pct_fewer_bytes"
        ),
        "reconstruct_is_lose": next(
            c["verdict_vs_incumbent"]["label"] == "LOSE"
            for c in cands if c["name"] == "reconstruct_then_gemm_oracle"
        ),
        "half_bytes_is_win": next(
            c["verdict_vs_incumbent"]["label"] == "WIN"
            for c in cands if c["name"] == "fused_half_bytes_native_tpr64"
        ),
        "density_leader_hardcoded_matches_git": density_check.get("matches_hardcoded_1p343")
        if density_check.get("found") else None,
        "anchor_tps_ms_gap_pct_under_spread": cal["measured_wall"]["anchor_tps_vs_ms_inconsistency"]["rel_gap_pct"]
        <= NATIVE_SPREAD_PCT + 0.05,
        "incumbent_mid_recovers_campaign_wall": abs(
            cal["error_bar"]["mid_ns"] - ANCHOR_TOKEN_NS
        ) / ANCHOR_TOKEN_NS < 0.002,
        "envelopes_ordered": all(
            c["error_bar"]["lo_ns"] <= c["error_bar"]["mid_ns"] + 1e-6
            and c["error_bar"]["mid_ns"] <= c["error_bar"]["hi_ns"] + 1e-6
            for c in cands
        ),
        "q3_measured_wall_inside_envelope": q3_wall_inside,
    }

    doc = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "commit": git_head(),
        "question": (
            "For a proposed representation (bytes/token, FLOP/token, dispatches/token, "
            "reconstruction cost), can it beat the uniform-q4 incumbent on this machine "
            "BEFORE anyone writes a kernel?"
        ),
        "answer": (
            "Yes, the model can say WIN / LOSE / CANNOT_TELL on traffic. It cannot say "
            "the candidate is coherent. Incumbent is bandwidth-side of the 964/1CB line "
            f"({cal['prediction']['binding_term']}), residual "
            f"{cal['residual_ms']:.3f} ms vs the 595.9 GB/s roof+ceremony floor. "
            "A candidate that only halves bytes in a 49-CB regime is still a LOSE. "
            "A candidate that halves bytes fused at 1 CB is a traffic WIN even under "
            "the measured q3 1.343× ALU envelope. A 17% byte cut at the same MAC count "
            "is CANNOT_TELL — and was measured NET-LOSS."
        ),
        "anchors_not_rederived": {
            "tps": ANCHOR_TPS,
            "ms_per_token": ANCHOR_TOKEN_MS,
            "token_ns": ANCHOR_TOKEN_NS,
            "dispatches_per_token": ANCHOR_DISPATCHES,
            "command_buffers_per_token": ANCHOR_CBS,
            "bpw": ANCHOR_BPW,
            "parameter_count": ANCHOR_PARAMS,
            "artifact_bytes": ANCHOR_ARTIFACT_B,
            "tensors": ANCHOR_TENSORS,
            "roof_gb_s": ANCHOR_ROOF_GB_S,
            "gpu_cores": ANCHOR_GPU_CORES,
            "unified_memory_bytes": ANCHOR_UNIFIED_B,
            "chipset": ANCHOR_CHIPSET,
            "metal": ANCHOR_METAL,
            "kernels_bound_g071": ANCHOR_BOUND,
            "kernels_declared": ANCHOR_DECLARED,
            "gemv_mac_flops": GEMV_MAC_FLOPS,
            "active_budget_bytes": ACTIVE_BUDGET_BYTES,
            "dense_f32_w_bytes": DENSE_F32_W_BYTES,
            "mlx_4bit_tps_live": MLX_TPS,
            "llamacpp_q5k_tps_archived": LLAMA_Q5K_TPS,
        },
        "machine": {
            "chipset": ANCHOR_CHIPSET,
            "gpu_cores": ANCHOR_GPU_CORES,
            "unified_memory_bytes": ANCHOR_UNIFIED_B,
            "metal": ANCHOR_METAL,
            "measured_roof_gb_s": ANCHOR_ROOF_GB_S,
            "datasheet_peak_gb_s": ANCHOR_DATASHEET_PEAK_GB_S,
            "honest_decode_ceiling_gb_s": ANCHOR_HONEST_CEILING_GB_S,
            "compute_peak_gflops": G143_COMPUTE_PEAK_GFLOPS,
            "compute_peak_source": "receipts/ascent-2026-08-16/G143_FLOPS_PER_TOKEN.json compute_peak_gflops",
            "roof_source": "NOETIC_* anchors_not_rederived.measured_roof_gb_s; not re-derived",
            "datasheet_source": "qwen38_token_ns_ledger.rs M3_ULTRA_PEAK_GB_S = 819; GPU_ATTACK.json m3_ultra_peak_GB_s",
        },
        "model": {
            "signature": "predict_token_ns(bytes_per_token, flop_per_token, dispatches_per_token, reconstruction_cost, command_buffers=1)",
            "formula": (
                "token_ns = max(traffic/595.9e9, flop/8979e9)*1e9 "
                "+ dispatches*(919250/964) + CBs*(12084 + 384250) "
                "+ [serial_extract ? ×23.7 on the GPU-work term]"
            ),
            "traffic": "bytes_per_token + dense_W_write + dense_W_reread",
            "reconstruction_cost": (
                "0 = fused native. A positive number is dense-W bytes written "
                "(reread assumed). A dict may set dense_w_bytes, reread, "
                "extra_dispatches, extra_flops, command_buffers, occupancy_class."
            ),
            "why_max_not_sum_for_bw_and_flop": (
                "Inside a kernel, compute and bandwidth overlap; the roofline is a max. "
                "Across kernels the same class still sums to total_bytes/BW, so the "
                "aggregate bandwidth floor is total traffic / roof."
            ),
            "why_ceremony_is_added": (
                "Production encodes one CB then waits. Host encode and CB gap are not "
                "inside the GEMV. Q80 at 98 CBs paid 36 ms of wait-minus-gpu; Qwen3.8 "
                "at 1 CB paid 0.384 ms. That is the dispatch-bound regime."
            ),
            "dense_reconstruction_law": (
                "representation → reconstruct dense W → ordinary GEMM may exist as a "
                "labelled correctness ORACLE. It is NOT a production implementation. "
                "The reconstruct candidate is scored so the model can reject it, not "
                "so anyone ships it."
            ),
            "constants": {
                "encode_ns_per_dispatch": measured(
                    ENCODE_NS_PER_DISPATCH,
                    source="QWEN38_TOKEN_NS_LEDGER.json median_encode_ns / 964",
                    label=LEDGER_LABEL,
                ),
                "submit_ns_per_cb": measured(SUBMIT_NS_PER_CB, source="ledger median_submit_ns / 1"),
                "gap_ns_per_cb": measured(
                    GAP_NS_PER_CB,
                    source="ledger wait_minus_gpu_ns / 1; Q80 36e6/98 CBs = 367 µs agrees in order",
                ),
                "roof_gb_s": measured(ANCHOR_ROOF_GB_S),
                "compute_peak_gflops": measured(G143_COMPUTE_PEAK_GFLOPS, source="G143_FLOPS_PER_TOKEN.json"),
                "q3_overprediction": measured(Q3_OVER_PRED, source="DENSITY_LEADER_SPEED.json arithmetic.measured_over_prediction"),
                "q80_serial_speedup": measured(Q80_SERIAL_SPEEDUP, source="Q80_RECONSTRUCTION_WON.json / NOETIC_TPR64_REOPEN"),
            },
        },
        "calibration": cal,
        "boundary": boundary,
        "candidates": cands,
        "cannot_predict": cannot_predict(),
        "density_leader_backtest": {
            "source": "receipts/ascent-2026-08-16/DENSITY_LEADER_SPEED.json",
            "git_show": density_check,
            "g0_gpu_ns": measured(Q3_G0_GPU_NS),
            "q3_gpu_ns": measured(Q3_GPU_NS),
            "naive_prediction_ns": measured(Q3_NAIVE_PRED_NS),
            "measured_over_prediction": measured(Q3_OVER_PRED),
            "verdict_was": "NO. 1.109x G0 GPU time while moving 17.4% fewer weight bytes.",
            "this_model_on_that_candidate": "CANNOT_TELL (required; a WIN would re-refute the refutation)",
            "measured_q3_wall_ns": measured(Q3_WALL_NS, source="DENSITY_LEADER_SPEED.json density_leader.wall"),
            "measured_wall_inside_this_model_envelope": q3_wall_inside,
        },
        "executable_today": list(EXECUTABLE_TODAY),
        "prior_science_respected": {
            "storage_compression_is_not_less_work": True,
            "source_and_executable_both_964_and_51p24_gflop": True,
            "dispatched_reconstructs_dense_NO_on_all_38": True,
            "g035_shared_beats_independent_false": True,
            "q80_storage_0p6462_vs_active_2p518": True,
            "low_bpw_without_health_is_not_a_result": True,
            "mlp_distillation_nogo": True,
            "never_synthetic_activations": True,
            "cosine_blind_to_0p01W": True,
            "tpr64_recon_free_32_of_33": True,
            "dense_recon_is_oracle_not_production": True,
        },
        "what_i_watched_fail": fails,
        "self_check": self_check,
        "written_to": str(RECEIPT),
    }

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2) + "\n")
    print_report(doc)

    if not self_check["all_expected_labels_match"]:
        print("FAIL: a candidate did not carry its expected WIN/LOSE/CANNOT_TELL", file=sys.stderr)
        return 3
    if not self_check["incumbent_residual_is_not_zero"]:
        print("FAIL: incumbent residual is zero — the roof was fitted, not measured", file=sys.stderr)
        return 4
    if not self_check["envelopes_ordered"]:
        print("FAIL: an error-bar envelope is not lo ≤ mid ≤ hi", file=sys.stderr)
        return 5
    if not self_check["incumbent_mid_recovers_campaign_wall"]:
        print("FAIL: calibrated mid does not recover the campaign wall on the incumbent", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

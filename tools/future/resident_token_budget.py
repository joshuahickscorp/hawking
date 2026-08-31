#!/usr/bin/env python3
"""What one decoded token costs, from a controlled fusion A/B on the resident.

Three runs of the same `--profile release` build, same box, minutes apart. The
first probe in this campaign ran WITHOUT the sealed fusion env and was therefore
measuring the unfused default graph rather than sealed-3.14. Keeping both is
what makes this useful: the pair yields a MEASURED marginal dispatch cost
instead of an inherited one.

The static walk in tools/future/tps_budget.py predicted 964 unfused and 628
fused from the encode-path call sites alone. The live probe returned exactly
964.00 and 628.00. Two independent methods agreeing to the digit is the
strongest evidence in this receipt.

Timing was taken while Grok lanes ran at maximum, so ns figures are
DIAGNOSTIC_RELATIVE. Dispatch counts and byte ledgers are exact regardless of
contention, and the A/B delta is a paired comparison in which contention
largely cancels.

    python3 tools/future/resident_token_budget.py --record
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402

RECEIPT = REPO / "receipts" / "future" / "RESIDENT_TOKEN_BUDGET.json"

BINARY = "workspace/ops/build/rust/release/examples/ascension_qwen38_resident"
ARTIFACT_ROOT = "/Users/scammermike/noetic/NOETIC_PARENT_A"
FUSION_ENV = {
    "HAWKING_QWEN38_FUSE_ADD_RMSNORM": "1",
    "HAWKING_QWEN38_FUSE_GQA_QKV": "1",
    "HAWKING_QWEN38_FUSE_DN_INPROJ": "1",
    "HAWKING_QWEN38_FUSE_MLP": "swiglu",
}

# The resident's own active_bytes_per_token is 10,727,793,881.75, but that field
# divides prefill+decode totals by generated tokens and is inflated by
# (P+N)/G = 139/128. See receipts/future/PER_GENERATED_TOKEN_INFLATION.json.
# The true per-forward-pass figure is the catalog census, confirmed to 7 ppm.
ACTIVE_BYTES_PER_TOKEN = 9_878_901_136
RESIDENT_REPORTED_ACTIVE_BYTES = 10_727_793_881.75
PER_GENERATED_TOKEN_INFLATION = 139 / 128
CLEAN_GEMV_GB_S = 703.5
PUBLISHED_PEAK_GB_S = 819.0
L40_HOST_PROBE_MARGINAL_US = 15.0

AB: dict[str, dict[str, Any]] = {
    "unfused": {
        "env": {},
        "is_production": False,
        "dispatches_per_decode_step": 964.0,
        "decode_tps": 32.745,
        "decode_wall_ms_per_token": 30.54,
        "decode_gpu_ms_per_token": 29.57,
        "active_bytes_per_token": ACTIVE_BYTES_PER_TOKEN,
        "fallbacks": 0,
    },
    "sealed_fusion": {
        "env": FUSION_ENV,
        "is_production": True,
        "dispatches_per_decode_step": 628.0,
        "decode_tps": 35.158,
        "decode_wall_ms_per_token": 28.44,
        "decode_gpu_ms_per_token": 27.64,
        "active_bytes_per_token": ACTIVE_BYTES_PER_TOKEN,
        "fallbacks": 0,
    },
    "fast_switch": {
        "env": {"HAWKING_QWEN38_FAST": "1"},
        "is_production": False,
        "result": "NO_VALID_RESPONSE",
        "note": "produced no parseable response in this probe; recorded so it "
                "is not assumed to be an equivalent way to enable fusion",
    },
}

PROD = AB["sealed_fusion"]


def marginal_dispatch_us() -> float:
    """From the paired A/B, not from a host-side ceremony probe."""
    d = AB["unfused"]["dispatches_per_decode_step"] - PROD["dispatches_per_decode_step"]
    ms = AB["unfused"]["decode_wall_ms_per_token"] - PROD["decode_wall_ms_per_token"]
    return ms * 1000.0 / d


def derived() -> dict[str, Any]:
    us = marginal_dispatch_us()
    wall = PROD["decode_wall_ms_per_token"]
    gpu = PROD["decode_gpu_ms_per_token"]
    disp = PROD["dispatches_per_decode_step"]
    gb = ACTIVE_BYTES_PER_TOKEN / 1e9
    dispatch_ms = disp * us / 1000.0
    byte_ms_at_clean = gb / CLEAN_GEMV_GB_S * 1000.0
    return {
        "production_decode_tps": PROD["decode_tps"],
        "production_ms_per_token": wall,
        "production_gpu_ms_per_token": gpu,
        "host_gap_ms_per_token": round(wall - gpu, 3),
        "host_gap_share": round((wall - gpu) / wall, 4),
        "production_dispatches_per_token": disp,
        "dispatches_removed_by_fusion": AB["unfused"]["dispatches_per_decode_step"] - disp,
        "ms_saved_by_fusion": round(
            AB["unfused"]["decode_wall_ms_per_token"] - wall, 3),
        "tps_gained_by_fusion": round(PROD["decode_tps"] - AB["unfused"]["decode_tps"], 3),
        "measured_marginal_dispatch_us": round(us, 3),
        "remaining_dispatch_ms_per_token": round(dispatch_ms, 3),
        "remaining_dispatch_share": round(dispatch_ms / wall, 4),
        "ms_if_all_dispatch_overhead_vanished": round(wall - dispatch_ms, 3),
        "tps_if_all_dispatch_overhead_vanished": round(1000.0 / (wall - dispatch_ms), 2),
        "active_gb_per_token": round(gb, 4),
        "byte_ms_at_clean_gemv": round(byte_ms_at_clean, 3),
        "byte_share_at_clean_gemv": round(byte_ms_at_clean / wall, 4),
        "effective_bandwidth_during_gpu_time_gb_s": round(gb / (gpu / 1000.0), 1),
        "roof_tps_at_clean_gemv": round(CLEAN_GEMV_GB_S / gb, 2),
        "roof_tps_at_published_peak": round(PUBLISHED_PEAK_GB_S / gb, 2),
    }


def findings() -> list[dict[str, Any]]:
    d = derived()
    return [
        {
            "id": "STATIC_WALK_AND_LIVE_PROBE_AGREE_EXACTLY",
            "what": (
                "tools/future/tps_budget.py walked the encode path and predicted "
                "964 dispatches unfused, 628 fused. The live probe returned "
                "964.00 and 628.00."
            ),
            "why_it_matters": (
                "Two independent methods, one static and one dynamic, landing on "
                "the same integers. The dispatch count is now known rather than "
                "estimated, and S017 §5's demand for a re-derived count is met "
                "twice over."
            ),
        },
        {
            "id": "MARGINAL_DISPATCH_COST_IS_6.25_US_NOT_15",
            "what": (
                f"Fusion removes 336 dispatches and {d['ms_saved_by_fusion']} ms, "
                f"which is {d['measured_marginal_dispatch_us']} us per dispatch."
            ),
            "supersedes": (
                f"the L40 host/catalog ceremony figure of "
                f"{L40_HOST_PROBE_MARGINAL_US} us, which was measured on a "
                "different class and is 2.4x too high for this one"
            ),
            "why_this_one_is_better": (
                "It is a paired A/B on the same binary and box, differing only "
                "in the fusion env. Contention affects both arms equally, so the "
                "delta survives a contended box in a way an absolute figure does "
                "not."
            ),
        },
        {
            "id": "DISPATCH_IS_WORTH_AT_MOST_5.6_MORE_TPS",
            "what": (
                f"{d['production_dispatches_per_token']} remaining dispatches at "
                f"{d['measured_marginal_dispatch_us']} us is "
                f"{d['remaining_dispatch_ms_per_token']} ms, "
                f"{d['remaining_dispatch_share'] * 100:.1f}% of the "
                f"{d['production_ms_per_token']} ms token. Removing ALL of it — "
                "not achievable, taken as a bound — gives "
                f"{d['tps_if_all_dispatch_overhead_vanished']} TPS."
            ),
            "conclusion": (
                "Dispatch elimination is real and worth pursuing, but it cannot "
                "reach even M1 (50 TPS) on its own. It is a contributor to a "
                "composed route, not the route."
            ),
        },
        {
            "id": "BYTES_ARE_THE_MAJORITY_OF_THE_TOKEN",
            "what": (
                f"{d['active_gb_per_token']} GB at the clean-GEMV "
                f"{CLEAN_GEMV_GB_S} GB/s would take {d['byte_ms_at_clean_gemv']} ms, "
                f"{d['byte_share_at_clean_gemv'] * 100:.1f}% of the token. Actual "
                f"effective bandwidth during GPU time is "
                f"{d['effective_bandwidth_during_gpu_time_gb_s']} GB/s."
            ),
            "conclusion": (
                "Byte traffic is the largest single identified term and the roof "
                f"it implies is {d['roof_tps_at_clean_gemv']} TPS. This is why "
                "the campaign moves to representation."
            ),
            "caveat": (
                "This charges every active byte as one read. It is an upper "
                "bound on byte time under a perfect-locality assumption, not a "
                "measurement."
            ),
            "and_it_cannot_be_measured_here": (
                "receipts/future/MEMORY_TRAFFIC_PROBE.json enumerated every "
                "counter surface on this M3 Ultra: MTLDevice.counterSets is "
                "['timestamp'] alone, stageutilization and statistic are absent "
                "from the device, dispatch-boundary sampling resolves to [0,0], "
                "and GPURawCounter fails to instantiate without the Instruments "
                "entitlement. No surface reports DRAM bytes to an ordinary Metal "
                "process. So this assumption cannot be confirmed OR refuted from "
                "inside the process, and the byte term stays a bound rather than "
                "becoming a measurement."
            ),
        },
        {
            "id": "THE_TOKEN_IS_GPU_BOUND",
            "what": (
                f"{d['production_gpu_ms_per_token']} ms of the "
                f"{d['production_ms_per_token']} ms token is GPU time; host gap "
                f"is {d['host_gap_ms_per_token']} ms "
                f"({d['host_gap_share'] * 100:.1f}%)."
            ),
            "conclusion": (
                "CPU submission, command buffer construction and host-side "
                "ceremony together cannot exceed ~3% of the token. The 6.25 us "
                "marginal dispatch cost is therefore mostly ON-DEVICE launch and "
                "teardown, not host submission — which is why fusing kernels "
                "helps and why faster host codegen did not."
            ),
        },
        {
            "id": "BUILD_PROFILE_IS_NOT_THE_TPS_LEVER",
            "what": (
                "release (lto=fat, cgu=1) and release-fast (lto=false, cgu=16) "
                "decoded within noise of each other on the same box. The sealed "
                "config pins release-fast while Cargo.toml says never to "
                "benchmark on it."
            ),
            "conclusion": (
                "The conflict is real but harmless: at 97% GPU time there is "
                "almost nothing for host codegen quality to work on. Benchmark "
                "on release as policy; do not expect TPS from it."
            ),
        },
    ]


def budget() -> dict[str, Any]:
    """S017 §2's categories. UNKNOWN stays UNKNOWN."""
    d = derived()
    return {
        "configuration": "sealed-3.14, fusion env set",
        "total_decode_ms_per_token": d["production_ms_per_token"],
        "categories": {
            "gpu_time_total": {
                "ms": d["production_gpu_ms_per_token"],
                "share": round(1 - d["host_gap_share"], 4),
                "status": "MEASURED",
            },
            "host_gap_total": {
                "ms": d["host_gap_ms_per_token"],
                "share": d["host_gap_share"],
                "status": "MEASURED",
                "bounds": ["cpu_submission", "command_encoder_cost",
                           "synchronization not covered by GPU time"],
            },
            "dispatch_cost": {
                "ms": d["remaining_dispatch_ms_per_token"],
                "share": d["remaining_dispatch_share"],
                "status": "MEASURED_BY_PAIRED_AB",
                "count_per_token": d["production_dispatches_per_token"],
                "marginal_us": d["measured_marginal_dispatch_us"],
                "caveat": "the marginal cost is measured over the 336 dispatches "
                          "fusion removed; extrapolating it to all 628 assumes "
                          "the remaining ones cost the same, which is not proven",
                "overlaps": "sits inside gpu_time_total, not beside it",
            },
            "weight_bytes_time": {
                "status": "BOUNDED_NOT_MEASURED",
                "upper_bound_ms": d["byte_ms_at_clean_gemv"],
                "upper_bound_share": d["byte_share_at_clean_gemv"],
                "why_not_measured": "actual_read_bytes_per_token is "
                                    "NOT_MEASURED_NO_METAL_MEMORY_COUNTER; "
                                    "active_bytes_per_token is the catalog sum, "
                                    "not what the GPU read",
                "overlaps": "sits inside gpu_time_total",
            },
            "mlp_bytes_time": {"status": "UNKNOWN",
                               "why": "needs the per-organ census re-derived "
                                      "against 10.7278 GB"},
            "attention_bytes_time": {"status": "UNKNOWN"},
            "state_bytes_time": {"status": "UNKNOWN"},
            "other_model_bytes_time": {"status": "UNKNOWN"},
            "low_bit_decode_cost": {"status": "UNKNOWN",
                                    "why": "inside gpu_time_total, not separable "
                                           "without per-kernel timing"},
            "useful_arithmetic": {"status": "UNKNOWN", "same_reason": True},
            "state_transition": {"status": "UNKNOWN"},
            "final_head_and_sampling": {"status": "UNKNOWN",
                                        "note": "lm_head + argmax run every "
                                                "decode token; skip-terminal is "
                                                "not in this tree"},
        },
        "additivity_warning": (
            "dispatch_cost and weight_bytes_time both sit INSIDE gpu_time_total. "
            "They must not be summed with it. Only gpu_time_total and "
            "host_gap_total partition the token."
        ),
        "honest_remainder": (
            "Two categories partition the token: GPU time (97%) and host gap "
            "(3%). Inside GPU time, dispatch is measured at "
            f"{d['remaining_dispatch_share'] * 100:.1f}% and bytes are bounded "
            f"above at {d['byte_share_at_clean_gemv'] * 100:.1f}%. Those two do "
            "not partition GPU time either — they overlap and neither is exact. "
            "Everything else stays UNKNOWN. Padding the gap into invented "
            "categories would make the receipt worse."
        ),
        "what_would_close_it": [
            "per-kernel GPU timestamps, which separate byte time from arithmetic",
            "a counted-read instrumented kernel, which turns "
            "actual_read_bytes_per_token from null into a number",
            "the per-organ byte census, which splits byte time by organ",
        ],
    }


def unexplained_residual() -> dict[str, Any]:
    """What is left after bytes, dispatch and host gap. This is the target.

    703 GB/s -> 337 GB/s was never a location, only a pair of endpoints. With a
    measured dispatch cost and a corrected byte count the loss chain finally has
    a bounded unexplained term, and it is large: a third of the token.
    """
    d = derived()
    total = d["production_ms_per_token"]
    byte_ms = d["byte_ms_at_clean_gemv"]
    disp_ms = d["remaining_dispatch_ms_per_token"]
    host_ms = d["host_gap_ms_per_token"]
    resid = total - byte_ms - disp_ms - host_ms
    return {
        "chain_ms": {
            "bytes_at_clean_gemv_roof": byte_ms,
            "dispatch_628_at_measured_6_25us": disp_ms,
            "host_gap": host_ms,
            "UNEXPLAINED": round(resid, 3),
            "total": total,
        },
        "unexplained_share": round(resid / total, 4),
        "tps_if_only_unexplained_were_removed": round(1000.0 / (total - resid), 2),
        "why_this_is_the_executor_recovery_target": (
            "Removing dispatch and host gap entirely reaches only 42.2 TPS. The "
            "roof is 71.21. The difference lives in this residual, so it is what "
            "'recover the addressing bandwidth' actually means now."
        ),
        "candidates_not_yet_falsified": [
            {"what": "low-bit decode ALU cost, affine2 vs q4",
             "named_by": "L41's own handoff, which falsified geometry and "
                         "pointed here instead"},
            {"what": "mixed organ sizes failing to amortise bandwidth equally",
             "named_by": "L40's redirect after catalog addressing was falsified"},
            {"what": "attention and DeltaNet arithmetic, which is not GEMV-shaped "
                     "and should not be expected to hit a GEMV roof",
             "note": "attention is 39.0% of catalogued active bytes"},
            {"what": "LM head, 675,430,440 bytes (6.8%), one large GEMV that may "
                     "already run near the clean rate and therefore is NOT here"},
            {"what": "state and KV traffic absent from the catalog census",
             "note": "MLP_BYTE_CENSUS marks KV cache, DeltaNet recurrent state "
                     "and activations UNKNOWN — they are real bytes the roof "
                     "denominator does not include"},
            {"what": "per-dispatch launch cost above the 6.25 us marginal figure "
                     "for the 628 that remain",
             "note": "the marginal cost was measured over the 336 fusion "
                     "removed; the survivors may not be identical"},
        ],
        "cheapest_falsifier": (
            "per-kernel GPU timestamps for one decode step. That splits the "
            "residual by kernel in a single run and settles which candidate "
            "owns it, without any representation work."
        ),
        "caveat": (
            "The byte term assumes every catalogued active byte is read exactly "
            "once at the clean rate. If real traffic exceeds the catalog — KV, "
            "state, activations, re-reads — then part of this residual is byte "
            "time that the roof denominator simply does not count, and the "
            "executor is closer to the roof than this makes it look."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "schema": "hawking.future.resident_token_budget.v3",
        "version": 3,
        "unexplained_residual": unexplained_residual(),
        "recorded_by": "tools/future/resident_token_budget.py",
        "evidence_class": "DIAGNOSTIC_RELATIVE",
        "gpu_authority": False,
        "took_gpu_lease": False,
        "binary": BINARY,
        "artifact_root": ARTIFACT_ROOT,
        "ab": AB,
        "derived": derived(),
        "findings": findings(),
        "budget": budget(),
        "corrects": {
            "claim": "an earlier commit in this campaign reported 964 dispatches "
                     "per token as the production figure",
            "correction": "964 is the UNFUSED default graph. Production "
                          "sealed-3.14 runs the fusion env and issues 628. The "
                          "first probe omitted the fusion env.",
            "also": "the 35.5 TPS anchor was right; it was measured with fusion. "
                    "The 32.7 figure from the first probe was the unfused arm.",
        },
        "corrects_again": {
            "claim": "v2 of this receipt reported active bytes per token as "
                     "10,727,793,881.75 and therefore a clean-GEMV roof of "
                     "65.58 TPS, with 71 TPS moved above the roof",
            "correction": (
                "That figure is the resident's active_bytes_per_token, which "
                "divides prefill+decode totals by generated tokens and is "
                "inflated by (P+N)/G = 139/128. The true per-forward-pass value "
                f"is {ACTIVE_BYTES_PER_TOKEN:,}, the roof is 71.21 TPS, and 71 "
                "is back below it."
            ),
            "caught_by": (
                "receipts/future/MLP_BYTE_CENSUS.json — an independent "
                "per-tensor sum over the HQ38M20 catalog that reconciled to "
                "9,878,901,136 and disagreed with the resident"
            ),
            "class": "receipts/future/PER_GENERATED_TOKEN_INFLATION.json",
            "lesson": (
                "The same prefill-over-generated-tokens arithmetic L42 found in "
                "TPS is present in the byte and dispatch ledgers. Finding it "
                "once should have triggered a search for the same denominator "
                "elsewhere; it did not, and an inflated anchor moved the whole "
                "milestone ladder for two commits."
            ),
        },
        "claim_boundary": (
            "Three runs of one build on a CPU-contended box. Dispatch counts and "
            "byte ledgers are exact and contention-independent. Absolute ns "
            "figures are DIAGNOSTIC_RELATIVE and must not be quoted as protected "
            "measurements; the A/B delta is a paired comparison and is the "
            "stronger of the two. The budget attributes only what these runs "
            "separated and marks overlapping terms as overlapping."
        ),
    }


def record() -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(build(), indent=1, sort_keys=True, default=str) + "\n")
    return RECEIPT


if __name__ == "__main__":
    if "--record" in sys.argv:
        print(f"wrote {record()}")
    else:
        print(json.dumps(build(), indent=1, default=str))

#!/usr/bin/env python3
"""The 9.67 ms is not in one place. Every organ but one runs at half the roof.

Region GPU timestamps for one representative decode token, joined against the
per-organ byte census, give an effective bandwidth per organ. The result is not
a hot spot:

    MLP        54.13% of bytes   55.8% of GPU time   344.1 GB/s
    DeltaNet   29.98%            29.6%               360.0 GB/s
    GQA         9.02%             9.4%               341.9 GB/s
    LM head     6.84%             4.9%               497.4 GB/s

Three organs cluster inside 5% of each other at ~350 GB/s, half the 703.5 GB/s
clean-GEMV roof, despite a 6x spread in bytes and a 3x spread in dispatch
density. The loss is uniform and structural, not localized.

The LM head is the exception and therefore the evidence: 675 MB in TWO dispatches
reaches 70.7% of the clean roof on the same box, the same catalog and the same
representation. Whatever holds the other three at 350 does not hold it.

A linear bytes-plus-dispatch model does not explain the difference. Fitted across
all four organs it returns a NEGATIVE per-dispatch cost of -1.0 us, which is
physically meaningless and is the model telling us it is wrong. DeltaNet has the
highest dispatch density of the three big organs and is the fastest of them.

    python3 tools/future/organ_bandwidth.py --record
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402

RECEIPT = REPO / "receipts" / "future" / "ORGAN_BANDWIDTH.json"
CENSUS = REPO / "receipts" / "future" / "MLP_BYTE_CENSUS.json"

CLEAN_GEMV_GB_S = 703.5
TOKEN_GPU_MS = 27.828          # measured GPU span for the traced token
ACTIVE_BYTES_TOTAL = 9_878_901_136

# Region GPU timestamps, MTLCommandBuffer GPUStartTime/GPUEndTime per region.
# Blit-boundary counter sampling is unsupported on AGXG15X and asserts if called.
REGION_MS: dict[str, float] = {
    "mlp": 15.541, "deltanet": 8.227, "gqa": 2.607, "lm_head": 1.358, "misc": 0.005,
}
REGION_DISPATCHES: dict[str, int] = {
    "mlp": 192, "deltanet": 337, "gqa": 96, "lm_head": 2, "misc": 1,
}
ORGAN_KEYS: dict[str, tuple[str, ...]] = {
    "mlp": ("mlp.gate", "mlp.up", "mlp.down"),
    "deltanet": ("attention.linear_qkvz", "attention.linear_ba",
                 "attention.linear_out", "attention.linear_conv1d"),
    "gqa": ("attention.q", "attention.k", "attention.v", "attention.o"),
    "lm_head": ("lm_head",),
}
TRACE_OVERHEAD = {
    "flag": "HAWKING_QWEN38_REGION_TIMING=1, default OFF",
    "wall_tps_off": 35.319, "wall_tps_on": 35.717,
    "gpu_ms_off": 27.200, "gpu_ms_on": 27.696,
    "gpu_overhead_pct": 1.8,
    "verdict": "under the 5% bar; the trace is production-shaped but not free",
    "dispatches_identical": 628,
    "greedy_text_identical": True,
}


class UnreconciledOrgans(Exception):
    """Raised rather than emit a census that does not cover the token."""


def _bytes() -> dict[str, int]:
    rows = json.loads(CENSUS.read_text())["census"]["by_organ"]
    by = {r["organ"]: int(r["active_bytes"]) for r in rows}
    return {g: sum(by[k] for k in ks) for g, ks in ORGAN_KEYS.items()}


def table() -> list[dict[str, Any]]:
    b = _bytes()
    out = []
    for g in ORGAN_KEYS:
        gb = b[g] / 1e9
        ms = REGION_MS[g]
        bw = gb / (ms / 1000.0)
        out.append({
            "organ": g,
            "active_bytes": b[g],
            "byte_share": round(b[g] / ACTIVE_BYTES_TOTAL, 5),
            "gpu_ms": ms,
            "time_share": round(ms / TOKEN_GPU_MS, 5),
            "effective_gb_s": round(bw, 1),
            "share_of_clean_roof": round(bw / CLEAN_GEMV_GB_S, 4),
            "dispatches": REGION_DISPATCHES[g],
            "mb_per_dispatch": round(b[g] / 1e6 / REGION_DISPATCHES[g], 1),
            "dispatches_per_gb": round(REGION_DISPATCHES[g] / gb, 1),
        })
    return out


def build() -> dict[str, Any]:
    rows = table()
    covered_b = sum(r["active_bytes"] for r in rows)
    covered_ms = sum(r["gpu_ms"] for r in rows)
    if covered_b / ACTIVE_BYTES_TOTAL < 0.99:
        raise UnreconciledOrgans(
            f"organ bytes cover only {covered_b / ACTIVE_BYTES_TOTAL:.4f} of the "
            "token; refusing to report bandwidth over an incomplete census"
        )
    slow = [r for r in rows if r["organ"] != "lm_head"]
    lo = min(r["effective_gb_s"] for r in slow)
    hi = max(r["effective_gb_s"] for r in slow)
    head = next(r for r in rows if r["organ"] == "lm_head")
    return {
        "schema": "hawking.future.organ_bandwidth.v1",
        "version": 1,
        "recorded_by": "tools/future/organ_bandwidth.py",
        "evidence_class": "DIAGNOSTIC_RELATIVE",
        "gpu_authority": False,
        "took_gpu_lease": False,
        "source": "region GPU timestamps (MTLCommandBuffer GPUStartTime/GPUEndTime), "
                  "130 samples, one representative decode token, joined to "
                  "receipts/future/MLP_BYTE_CENSUS.json",
        "trace_overhead": TRACE_OVERHEAD,
        "token_gpu_ms": TOKEN_GPU_MS,
        "organs": rows,
        "coverage": {
            "bytes_covered": covered_b,
            "byte_coverage": round(covered_b / ACTIVE_BYTES_TOTAL, 5),
            "gpu_ms_covered": round(covered_ms, 3),
            "gpu_ms_unattributed": round(TOKEN_GPU_MS - covered_ms, 3),
            "uncovered_bytes": ACTIVE_BYTES_TOTAL - covered_b,
            "uncovered_are": "norms, embedding row, A_log and dt_bias — 0.028% of "
                             "the token",
        },
        "findings": [
            {
                "id": "THE_LOSS_IS_UNIFORM_NOT_LOCALIZED",
                "what": f"MLP, DeltaNet and GQA sit between {lo} and {hi} GB/s — "
                        "inside 5% of each other — against a 703.5 GB/s clean "
                        "roof",
                "consequence": (
                    "There is no hot organ. The 9.67 ms unexplained term in the "
                    "token budget is not a gap between organs and not one slow "
                    "organ; it is distributed across all of them in proportion "
                    "to their bytes."
                ),
            },
            {
                "id": "THE_LM_HEAD_PROVES_THE_ROOF_IS_REACHABLE",
                "what": f"{head['active_bytes']:,} bytes in "
                        f"{head['dispatches']} dispatches reach "
                        f"{head['effective_gb_s']} GB/s, "
                        f"{head['share_of_clean_roof'] * 100:.1f}% of the clean roof "
                        f"and {head['effective_gb_s'] / hi:.2f}x the fastest of the "
                        "other three",
                "why_it_matters": (
                    "Same box, same catalog, same low-bit representation, same "
                    "build. Whatever holds the other three organs at ~350 GB/s "
                    "is not a property of the machine or of the representation. "
                    "This is an existence proof with 45% headroom in it."
                ),
            },
            {
                "id": "DISPATCH_COUNT_DOES_NOT_EXPLAIN_THE_DIFFERENCE",
                "what": (
                    "A least-squares fit of t = a*bytes + b*dispatches over the "
                    "four organs returns b = -1.008 us per dispatch. A negative "
                    "per-dispatch cost is physically meaningless; the model is "
                    "refuted, not fitted."
                ),
                "corroboration": (
                    "DeltaNet has the highest dispatch density of the three big "
                    "organs (113.8 per GB against MLP's 35.9) and is the FASTEST "
                    "of the three at 360 GB/s. If dispatch count set the rate, "
                    "that ordering would be inverted."
                ),
                "what_survives": (
                    "bytes per dispatch, not dispatch count: the LM head moves "
                    f"{head['mb_per_dispatch']} MB per launch against 8.8 to 27.9 "
                    "MB for the others. That is a hypothesis this measurement "
                    "supports but does not establish."
                ),
            },
        ],
        "open_question": {
            "question": "why do three organs stop at half the clean roof while a "
                        "single large contiguous GEMV reaches 70%?",
            "candidates_not_yet_tested": [
                "contiguity: the LM head streams one tensor; the others walk "
                "hundreds of per-layer tensors",
                "dependency chain: layer N+1 cannot start until layer N retires, "
                "so the memory system never sees a deep queue",
                "per-launch working set: 337 MB gives the prefetcher something to "
                "work with that 9 MB does not",
                "low-bit decode ALU cost inside the organ kernels, which the "
                "region trace cannot separate from byte time",
            ],
            "cheapest_falsifier": (
                "run the MLP of one layer as a single fused region over a "
                "contiguous staging buffer and measure its GB/s. If it stays at "
                "~350 the cause is not contiguity."
            ),
        },
        "claim_boundary": (
            "One traced token on a CPU-contended box, with the trace itself "
            "costing 1.8% of GPU time. Byte shares are exact from the catalog "
            "census; the ms figures are DIAGNOSTIC_RELATIVE. Bandwidth here is "
            "bytes-the-catalog-says divided by time-the-GPU-took, so it inherits "
            "the perfect-locality assumption: if real traffic exceeds the "
            "catalog, these are underestimates. normalization, residual, "
            "low_bit_decode and routing are folded INTO the organ kernels by "
            "production fusion and were not independently sampled — the region "
            "trace cannot say how much of any organ's time is decode versus "
            "arithmetic versus streaming."
        ),
    }


def record() -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n")
    return RECEIPT


if __name__ == "__main__":
    if "--record" in sys.argv:
        print(f"wrote {record()}")
    else:
        d = build()
        print(f"{'organ':10s} {'byte%':>7s} {'time%':>7s} {'GB/s':>7s} {'%roof':>7s} {'disp':>5s} {'MB/disp':>8s}")
        for r in d["organs"]:
            print(f"{r['organ']:10s} {r['byte_share']*100:6.2f}% {r['time_share']*100:6.1f}% "
                  f"{r['effective_gb_s']:7.1f} {r['share_of_clean_roof']*100:6.1f}% "
                  f"{r['dispatches']:5d} {r['mb_per_dispatch']:8.1f}")

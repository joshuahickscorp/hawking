#!/usr/bin/env python3
"""The live 71-TPS budget, ranked by gain per unit of experiment cost.

The important number this produces is not 71. It is 47.97: what the resident
would reach if every organ merely matched the bandwidth the LM head ALREADY
achieves on this box, with no byte reduction at all. That is the granularity
hypothesis's entire payoff, and it is a demonstrated regime rather than a
theoretical one.

The second important number is 66.54: the ceiling if every organ hit the clean
single-GEMV roof of 703.5 GB/s with today's bytes. It is not 71.21, because the
earlier 71.21 divided bytes by the roof and forgot the 0.99 ms host gap. So 71
TPS is NOT reachable at the clean roof on today's bytes. It needs the roof AND
about 7% fewer bytes, or the roof AND the host gap gone.

    python3 tools/future/causal_budget_71.py --record
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402

RECEIPT = REPO / "receipts" / "future" / "RESIDENT_71TPS_CAUSAL_BUDGET.json"

CLEAN_GEMV_GB_S = 703.5          # single clean GEMV, measured
DEMONSTRATED_GB_S = 497.4        # the LM head, measured on this box TODAY
HOST_GAP_MS = 0.989              # measured, 3 runs, stable to 5 decimals
# GPU time inside the decode step that belongs to no organ, taken from the run
# that measured BOTH the parts and the total: ORGAN_BANDWIDTH.coverage
# gpu_ms_unattributed. The uncovered work is NAMED there - norms, the embedding
# row, A_log and dt_bias, 0.028% of the token - so this is a small measured
# remainder, not a mystery.
#
# I first computed this as 0.321 ms by subtracting ORGAN_BANDWIDTH's covered
# 27.733 from WALL_GPU_RECONCILIATION's decode_gpu_ms 28.054. Those are DIFFERENT
# RUNS, and one of them has the region trace ON, which that receipt measures at
# 1.8% of GPU time (27.2 -> 27.696 with dispatches and greedy text identical). So
# 0.321 was one real 0.095 ms remainder plus 0.226 ms of trace overhead and
# run-to-run variation, presented as if it were unexplained physics. Mixing a
# total from one receipt with parts from another is precisely the cross-receipt
# error the citation machinery in this module exists to prevent, and I made it
# while building that machinery.
#
# Leaving it out entirely is still wrong: a reconstruction that sums only the
# organs it can name reports a token faster than the one measured. The honest
# term is the same-run remainder.
UNATTRIBUTED_GPU_MS = 0.095
ACTIVE_BYTES = 9_878_901_136


# ---------------------------------------------------------------------------
# CITATIONS ARE LOADED, NOT COPIED.
#
# G072 asks for a ledger that recomputes from landed receipts "rather than from
# hand-entered numbers". Every constant above and every lever below used to be a
# literal with a receipt PATH beside it, which is a citation, not a recomputation:
# if the cited receipt changed, the literal kept its old value and nothing said so.
# That is the stale-baseline failure this campaign has already paid for twice.
#
# Each row below names the receipt AND the exact path to the field. resolve()
# reads it and RAISES on drift. The literal survives only as `expect`, which makes
# a silent change impossible in both directions: a receipt that moves fails the
# expectation, and an expectation edited without the receipt fails too.
# ---------------------------------------------------------------------------


class CitationDrift(RuntimeError):
    """A ledger number no longer matches the receipt it cites."""


class CitationMissing(RuntimeError):
    """A cited receipt or field is not on disk. Never fall back to the literal."""


def _walk(obj: Any, path: Sequence[Any], where: str) -> Any:
    cur = obj
    for step in path:
        if isinstance(step, Mapping):
            key, want = next(iter(step.items()))
            if not isinstance(cur, list):
                raise CitationMissing(f"{where}: selector {step} needs a list, got {type(cur).__name__}")
            hits = [r for r in cur if isinstance(r, Mapping) and r.get(key) == want]
            if len(hits) != 1:
                raise CitationMissing(f"{where}: selector {step} matched {len(hits)} rows, need exactly 1")
            cur = hits[0]
            continue
        try:
            cur = cur[step]
        except (KeyError, IndexError, TypeError) as exc:
            raise CitationMissing(f"{where}: cannot reach {step!r} ({type(exc).__name__})") from exc
    return cur


def resolve(source: str, path: Sequence[Any], expect: float, *, rel_tol: float = 1e-6) -> float:
    """Read the cited number from disk. Refuse to guess, refuse to drift."""
    rp = REPO / source
    if not rp.exists():
        raise CitationMissing(f"{source} is not on disk; the ledger cannot cite what does not exist")
    got = _walk(json.loads(rp.read_text()), path, f"{source}:{'.'.join(map(str, path))}")
    if isinstance(got, bool) or not isinstance(got, (int, float)):
        raise CitationMissing(f"{source}:{path} is {type(got).__name__}, not a number")
    got = float(got)
    if abs(got - expect) > rel_tol * max(abs(expect), 1.0):
        raise CitationDrift(
            f"{source}:{'.'.join(map(str, path))} is {got!r}, ledger carries {expect!r}. "
            "Update the ledger from the receipt - never the other way round."
        )
    return got


CITATIONS: tuple[dict[str, Any], ...] = (
    {"id": "host_gap_ms", "expect": HOST_GAP_MS, "rel_tol": 6e-4,
     "source": "receipts/future/WALL_GPU_RECONCILIATION.json",
     "path": ["derived", "host_gap_ms_per_token"]},
    {"id": "demonstrated_gb_s", "expect": DEMONSTRATED_GB_S,
     "source": "receipts/future/MLP_REGION_FALSIFIER.json",
     "path": ["lm_head_gb_s"]},
    {"id": "clean_gemv_gb_s", "expect": CLEAN_GEMV_GB_S,
     "source": "receipts/future/MLP_REGION_FALSIFIER.json",
     "path": ["clean_gemv_gb_s"]},
    {"id": "entropy_floor_bytes", "expect": 277_697_891.2457967,
     "source": "receipts/future/MLP_CODE_INFORMATION.json",
     "path": ["measurements", "iid_redundant_bytes"]},
    {"id": "quantize_aux_u8_bytes", "expect": 534_773_760,
     "source": "receipts/future/MLP_AUXILIARY_INFORMATION.json",
     "path": ["open_byte_levers", {"id": "quantize_aux_u8"}, "bytes_eliminated_if_true"]},
    {"id": "group_size_256_bytes", "expect": 802_160_640,
     "source": "receipts/future/MLP_AUXILIARY_INFORMATION.json",
     "path": ["group_size_curve", {"group_size": 256}, "bytes_eliminated_vs_incumbent"]},
    {"id": "group_size_1024_bytes", "expect": 1_002_700_800,
     "source": "receipts/future/MLP_AUXILIARY_INFORMATION.json",
     "path": ["group_size_curve", {"group_size": 1024}, "bytes_eliminated_vs_incumbent"]},
    {"id": "unattributed_gpu_ms", "expect": 0.095,
     "source": "receipts/future/ORGAN_BANDWIDTH.json",
     "path": ["coverage", "gpu_ms_unattributed"]},
    {"id": "organ_gpu_ms_covered", "expect": 27.733,
     "source": "receipts/future/ORGAN_BANDWIDTH.json",
     "path": ["coverage", "gpu_ms_covered"]},
    {"id": "traced_token_gpu_ms", "expect": 27.828,
     "source": "receipts/future/ORGAN_BANDWIDTH.json",
     "path": ["token_gpu_ms"]},
    {"id": "region_trace_overhead_pct", "expect": 1.8,
     "source": "receipts/future/ORGAN_BANDWIDTH.json",
     "path": ["trace_overhead", "gpu_overhead_pct"]},
    # G131 promoted three levers to sealed defaults and measured the result in
    # a protected window. This module was not DRIFTED - its expect matched its
    # source exactly - it was citing a SUPERSEDED source, which no drift check
    # can catch. The citation moves; the mechanism that pins expect to source is
    # unchanged and now pins it to the current body.
    {"id": "current_body_wall_ms", "expect": 22.9024,
     "source": "receipts/future/SEALED_DEFAULT_ABSOLUTE.json",
     "path": ["measured", "wall_ms_per_token"]},
    {"id": "current_body_gpu_ms", "expect": 21.9464,
     "source": "receipts/future/SEALED_DEFAULT_ABSOLUTE.json",
     "path": ["measured", "gpu_ms_per_token"]},
    {"id": "host_gap_worth_tps", "expect": 1.214,
     "source": "receipts/future/WALL_GPU_RECONCILIATION.json",
     "path": ["derived", "tps_gain_from_deleting_all_host_work"]},
)


def resolve_all() -> dict[str, float]:
    """Every cited number, read from its receipt. Raises rather than degrading."""
    return {
        c["id"]: resolve(c["source"], c["path"], float(c["expect"]),
                         rel_tol=float(c.get("rel_tol", 1e-6)))
        for c in CITATIONS
    }

ORGANS: tuple[dict[str, Any], ...] = (
    {"organ": "mlp", "gb": 5.347795776, "ms": 15.541, "gb_s": 344.1, "dispatches": 192},
    {"organ": "deltanet", "gb": 2.961659904, "ms": 8.227, "gb_s": 360.0, "dispatches": 337},
    {"organ": "gqa", "gb": 0.891292160, "ms": 2.607, "gb_s": 341.9, "dispatches": 96},
    {"organ": "lm_head", "gb": 0.675430440, "ms": 1.358, "gb_s": 497.4, "dispatches": 2},
)

# Byte levers with a measured byte model. Capability is UNMEASURED for all.
# What measurement has KILLED, kept in the budget so it is not re-proposed.
REFUTED_LEVERS: tuple[dict[str, Any], ...] = (
    {"id": "region_granularity",
     "was_worth_tps": 6.97,
     "verdict": "GRANULARITY_REFUTED",
     "evidence": "one MLP layer in one staging buffer under one serial encoder "
                 "measured 332.2 GB/s against production 331.6, bit-identical",
     "source": "receipts/future/MLP_REGION_FALSIFIER.json"},
    {"id": "entropy_code_the_mlp_codes",
     "gb_saved_if_perfect": 0.277697891,
     "verdict": "AT_THE_FLOOR",
     "evidence": "i.i.d. Shannon 1.87018 bits of 2 stored; Markov-1 adds 0.00195; "
                 "93.5% of the code body is independent information",
     "source": "receipts/future/MLP_CODE_INFORMATION.json"},
    {"id": "fuse_representation_decode",
     "verdict": "ALREADY_FUSED",
     "evidence": "every decode+consume candidate eliminates ZERO intermediate "
                 "bytes; the kernels decode in-register and dense_w_materialized "
                 "stays 0",
     "source": "receipts/future/REPRESENTATION_DECODE_FUSION.json"},
    {"id": "eliminate_all_host_gap",
     "was_worth_tps": 1.214,
     "verdict": "BOUNDED_TOO_SMALL",
     "evidence": "host gap is 0.989 ms of a 29.043 ms token, measured over three "
                 "runs and stable to five decimals",
     "source": "receipts/future/WALL_GPU_RECONCILIATION.json"},
)

# EVERY LEVER CARRIES ITS STREAM, AND THE STREAM CARRIES THE RATE.
#
# These rungs used to bill removed bytes at the MLP organ average, 344.1 GB/s,
# which credited the auxiliary levers with 1.99 and 3.08 TPS. ECONOMICS_
# CALIBRATION measured the streams separately by dropping fractions of each and
# timing it: codes_keep_50 is faster than 2*MAD, aux_keep_50 is NOT. Removing
# half the auxiliary is inside measurement noise.
#
#   weight_codes    0.547 ms/GB   (bills at 1827 GB/s, not 344)
#   broadcast_aux   0.000 ms/GB   (measured within noise)
#   activation      2.906 ms/GB
#
# So the organ average was never a byte-class rate. Billing an aux byte at it is
# the overcredit the stream_class guard refuses, and this ladder was doing it.
# group_size_256 and group_size_1024 are additionally capability-REFUTED on
# held-out fit by AUX_CAPABILITY_SCREEN; they are kept here priced at zero rather
# than deleted, so the record shows what was once claimed for them.
BYTE_LEVERS: tuple[dict[str, Any], ...] = (
    {"id": "entropy_floor_of_mlp_codes", "gb_saved": 0.277697891,
     "stream_class": "weight_codes",
     "status": "AT_THE_FLOOR_NOT_A_LEVER",
     "source": "receipts/future/MLP_CODE_INFORMATION.json"},
    {"id": "quantize_aux_u8", "gb_saved": 0.534773760,
     "stream_class": "broadcast_aux",
     "status": "OPEN_ON_BYTES_IMMATERIAL_ON_TIME",
     "source": "receipts/future/MLP_AUXILIARY_INFORMATION.json"},
    {"id": "group_size_256", "gb_saved": 0.802160640,
     "stream_class": "broadcast_aux",
     "status": "CAPABILITY_REFUTED_AND_IMMATERIAL_ON_TIME",
     "source": "receipts/future/MLP_AUXILIARY_INFORMATION.json"},
    {"id": "group_size_1024", "gb_saved": 1.002700800,
     "stream_class": "broadcast_aux",
     "status": "CAPABILITY_REFUTED_AND_IMMATERIAL_ON_TIME",
     "source": "receipts/future/MLP_AUXILIARY_INFORMATION.json"},
)

STREAM_MS_PER_GB: dict[str, float] = {
    "weight_codes": 0.547282,
    "broadcast_aux": 0.0,
    "activation": 2.906132,
}


def lever_ms_saved(lever: Mapping[str, Any]) -> float:
    """Bill a lever at its OWN stream's measured rate, never the organ average."""
    return float(lever["gb_saved"]) * STREAM_MS_PER_GB[str(lever["stream_class"])]


def token_ms(gb_s: float | None = None, gb_saved: float = 0.0,
             host_ms: float = HOST_GAP_MS,
             unattributed_ms: float = UNATTRIBUTED_GPU_MS) -> float:
    """Token time if every organ ran at gb_s. None means keep measured rates.

    The unattributed GPU term is included by DEFAULT. A reconstruction that sums
    only the organs it can name reports a faster token than the one measured, and
    every rung above it inherits that optimism silently.
    """
    total = float(unattributed_ms)
    saved_left = gb_saved
    for o in ORGANS:
        gb = o["gb"]
        take = 0.0
        # Every byte lever in this receipt is an MLP auxiliary-array change.
        if saved_left > 0 and o["organ"] == "mlp":
            take = min(saved_left, gb)
            saved_left -= take
        if gb_s:
            total += (gb - take) / gb_s * 1000.0
        else:
            # Keep the organ's OWN measured rate. Removing bytes at that rate is
            # the honest counterfactual: it does not assume the organ also gets
            # faster per byte, which is a separate hypothesis.
            total += o["ms"] * (gb - take) / gb
    return total + host_ms


def tps(ms: float) -> float:
    return 1000.0 / ms


def ladder() -> list[dict[str, Any]]:
    now = token_ms()
    rows = [
        {"rung": "measured now", "ms": round(now, 3), "tps": round(tps(now), 2),
         "requires": "nothing", "class": "MEASURED"},
        {"rung": "every organ at the LM head's demonstrated 497.4 GB/s",
         "ms": round(token_ms(DEMONSTRATED_GB_S), 3),
         "tps": round(tps(token_ms(DEMONSTRATED_GB_S)), 2),
         "requires": "executor work only; zero byte reduction",
         "class": "DEMONSTRATED_REGIME",
         "note": "this is the granularity hypothesis's entire payoff, and the "
                 "rate is already achieved on this box by one organ"},
    ]
    for lever in BYTE_LEVERS:
        # Bill at the LEVER'S OWN stream rate. token_ms(gb_s, gb_saved) removes
        # bytes at the organ's rate, which for an auxiliary lever is the
        # overcredit ECONOMICS_CALIBRATION refuted.
        ms = token_ms(DEMONSTRATED_GB_S) - lever_ms_saved(lever)
        rows.append({
            "rung": f"demonstrated regime + {lever['id']}",
            "ms": round(ms, 3), "tps": round(tps(ms), 2),
            "requires": f"the above, plus {lever['gb_saved']:.3f} GB removed",
            "class": "DEMONSTRATED_PLUS_OPEN_BYTE_LEVER",
            "capability": "UNMEASURED",
            "stream_class": lever["stream_class"],
            "ms_saved_at_measured_stream_rate": round(lever_ms_saved(lever), 4),
            "lever_status": lever["status"],
        })
    roof = token_ms(CLEAN_GEMV_GB_S)
    rows.append({
        "rung": "every organ at the clean GEMV roof 703.5 GB/s",
        "ms": round(roof, 3), "tps": round(tps(roof), 2),
        "requires": "beating the LM head as well, on every organ",
        "class": "ROOF_ON_TODAYS_BYTES",
    })
    need_ms = 1000.0 / 71.0
    byte_ms_at_roof = need_ms - HOST_GAP_MS
    gb_for_71 = byte_ms_at_roof / 1000.0 * CLEAN_GEMV_GB_S
    rows.append({
        "rung": "71 TPS",
        "ms": round(need_ms, 3), "tps": 71.0,
        "requires": (
            f"the clean roof AND bytes down to {gb_for_71:.4f} GB "
            f"({(1 - gb_for_71 / (ACTIVE_BYTES / 1e9)) * 100:.1f}% fewer), "
            "or the clean roof AND the 0.99 ms host gap eliminated"
        ),
        "class": "NOT_REACHABLE_AT_THE_ROOF_ON_TODAYS_BYTES",
    })
    return rows


def experiments() -> list[dict[str, Any]]:
    """Ranked by TPS gain per unit of experiment cost."""
    now = token_ms()
    rows = []
    for o in ORGANS:
        if o["organ"] == "lm_head":
            continue
        demo_ms = o["gb"] / DEMONSTRATED_GB_S * 1000.0
        saved = o["ms"] - demo_ms
        rows.append({
            "id": f"reach_demonstrated_bandwidth_{o['organ']}",
            "organ": o["organ"],
            "current_ms": o["ms"],
            "current_gb_s": o["gb_s"],
            "target_ms_at_demonstrated": round(demo_ms, 3),
            "ms_saved": round(saved, 3),
            "tps_gain": round(tps(now - saved) - tps(now), 2),
            "falsifier": "one representative layer, contiguous, one/few fused "
                         "regions, identical arithmetic",
            "cost": "ONE_EXPERIMENT",
            "status": "RUNNING" if o["organ"] == "mlp" else "QUEUED",
        })
    for lever in BYTE_LEVERS:
        saved = lever_ms_saved(lever)
        ms = now - saved
        rows.append({
            "id": lever["id"],
            "gb_saved": lever["gb_saved"],
            "stream_class": lever["stream_class"],
            "lever_status": lever["status"],
            "ms_saved": round(saved, 4),
            "tps_gain": round(tps(ms) - tps(now), 2),
            "falsifier": "held-out reconstruction plus organ error on a real layer",
            "cost": "ONE_FIT_PLUS_A_CAPABILITY_SCREEN",
            "capability": "UNMEASURED",
            "status": "OPEN",
        })
    rows.append({
        "id": "eliminate_all_host_gap",
        "ms_saved": HOST_GAP_MS,
        "tps_gain": round(tps(now - HOST_GAP_MS) - tps(now), 2),
        "falsifier": "already measured; this is the CEILING of the whole host class",
        "cost": "NOT_WORTH_RUNNING",
        "status": "CLOSED",
        "why": "receipts/future/WALL_GPU_RECONCILIATION.json bounds every host "
               "term at 0.99 ms combined",
    })
    rows.sort(key=lambda r: -r["tps_gain"])
    return rows



def causal_residual() -> dict[str, Any]:
    """What measurement explains, what it does not, and which total to compare to.

    G072 asks for this and adds the clause that makes it worth having: the
    residual "must not stay a miscellaneous bucket".

    THREE numbers, and only one of them is a residual.

    HOST GAP is not one. It is wall minus GPU, a SUBTRACTION, so it absorbs
    everything unmeasured by construction and can never be nonzero-surprising.
    Reporting wall-minus-parts as "the unexplained fraction" produces a
    reassuring zero that means nothing.

    THE REAL REMAINDER IS 0.095 ms, from the run that measured both the parts
    and the total. ORGAN_BANDWIDTH covers 99.972% of the bytes and 27.733 of
    27.828 GPU ms, and it NAMES what is left: norms, the embedding row, A_log
    and dt_bias. Small, measured, named.

    THE 0.226 ms GAP TO THE UNTRACED WALL IS NOT A RESIDUAL EITHER, and this is
    where I went wrong first: I subtracted ORGAN_BANDWIDTH's covered 27.733 from
    WALL_GPU_RECONCILIATION's 28.054 and called the 0.321 difference unattributed
    GPU time. Those are different runs, and the organ run has the region trace ON
    - measured at 1.8% of GPU time, dispatches and greedy text identical. So
    0.321 was one real 0.095 remainder plus 0.226 of trace overhead and
    run-to-run variation, presented as unexplained physics. A total from one
    receipt against parts from another is the cross-receipt error this module's
    citation machinery exists to prevent.
    """
    cited = resolve_all()
    organ_ms = sum(float(o["ms"]) for o in ORGANS)
    traced_total = float(cited["traced_token_gpu_ms"])
    covered = float(cited["organ_gpu_ms_covered"])
    unattributed = float(cited["unattributed_gpu_ms"])
    rp = REPO / "receipts" / "future" / "WALL_GPU_RECONCILIATION.json"
    derived = json.loads(rp.read_text())["derived"]
    untraced_gpu_ms = float(derived["decode_gpu_ms_per_token"])
    wall_ms = float(derived["decode_wall_ms_per_token"])
    return {
        "wall_ms": wall_ms,
        "untraced_gpu_ms": untraced_gpu_ms,
        "traced_gpu_ms": traced_total,
        "organ_ms_sum": round(organ_ms, 4),
        "organ_ms_covered_in_receipt": covered,
        "organs_explain_of_traced_gpu": round(covered / traced_total, 6),
        "gpu_residual_ms": unattributed,
        "gpu_residual_is_named_not_mysterious": (
            "norms, the embedding row, A_log and dt_bias - 0.028% of the token"
        ),
        "byte_coverage": 0.99972,
        "trace_overhead_pct": float(cited["region_trace_overhead_pct"]),
        "traced_vs_untraced_ms": round(untraced_gpu_ms - traced_total, 4),
        "why_that_gap_is_not_a_residual": (
            "the organ parts come from a run with the region trace ON, which that "
            "receipt measures at 1.8% of GPU time with dispatches and greedy text "
            "identical. Comparing traced parts against an untraced total mixes "
            "runs; the difference is overhead and variation, not physics."
        ),
        "host_gap_ms": cited["host_gap_ms"],
        "host_gap_is_a_subtraction_not_a_residual": (
            "host_gap = wall - gpu by construction, so it absorbs every unmeasured "
            "host cost and cannot be a surprise. Do not report wall minus parts as "
            "the unexplained fraction; it is definitionally zero."
        ),
        "next_prey": (
            "not the remainder. ORGAN_BANDWIDTH's own finding is "
            "THE_LOSS_IS_UNIFORM_NOT_LOCALIZED: MLP, DeltaNet and GQA sit between "
            "341.9 and 360.0 GB/s, inside 5% of each other, against a 703.5 clean "
            "roof. There is no hot organ to attack. The loss is distributed in "
            "proportion to bytes, which is why byte elimination outranks execution "
            "tuning from here."
        ),
        "baseline_moved": {
            "computed_against_ms": wall_ms,
            "current_body_ms": float(cited["current_body_wall_ms"]),
            "current_body_gpu_ms": float(cited["current_body_gpu_ms"]),
            "current_body_tps": round(1000.0 / float(cited["current_body_wall_ms"]), 3),
            "ms_removed_since": round(wall_ms - float(cited["current_body_wall_ms"]), 4),
            # Read from the citation, not typed. A hard-coded source string
            # outlives the citation it describes, and this one did: the numbers
            # above moved to SEALED_DEFAULT_ABSOLUTE while this line still named
            # the receipt they had left.
            "measured_by": next(
                c["source"] for c in CITATIONS
                if c["id"] == "current_body_wall_ms"),
            "what_is_still_from_the_old_census": (
                "the PER-ORGAN ms in ORGANS are from the pre-widen_f4 region trace. "
                "The new census has eight rows (it separates mlp_gate_up from "
                "mlp_down and carries a q4_remainder and a sampling row) and does "
                "not map one-to-one onto the old four, so remapping it is a "
                "separate piece of work rather than a substitution. The organ "
                "SHARES remain the best available reading; the token TOTAL is now "
                "measured on the body that runs."
            ),
        },
    }


def build() -> dict[str, Any]:
    # Read every cited number off disk FIRST. A receipt that cannot be emitted
    # without its citations resolving cannot be emitted stale.
    cited = resolve_all()
    now = token_ms()
    return {
        "citations_resolved": cited,
        "causal_residual": causal_residual(),
        "citations": [
            {"id": c["id"], "source": c["source"], "path": [str(x) for x in c["path"]]}
            for c in CITATIONS
        ],
        "schema": "hawking.future.causal_budget_71.v1",
        "version": 1,
        "recorded_by": "tools/future/causal_budget_71.py",
        "evidence_class": "DIAGNOSTIC_RELATIVE",
        "gpu_authority": False,
        "measured_now": {
            "token_ms": round(now, 3), "tps": round(tps(now), 2),
            "active_bytes": ACTIVE_BYTES,
            "host_gap_ms": HOST_GAP_MS,
            "organs": list(ORGANS),
        },
        "the_two_numbers_that_matter": {
            "demonstrated_regime_tps": round(tps(token_ms(DEMONSTRATED_GB_S)), 2),
            "why": "what the resident reaches if every organ merely matches the "
                   "bandwidth the LM head ALREADY achieves here, with zero byte "
                   "reduction. A demonstrated regime, not a theoretical one.",
            "roof_on_todays_bytes_tps": round(tps(token_ms(CLEAN_GEMV_GB_S)), 2),
            "and_why_it_is_not_71": (
                "the earlier 71.21 divided bytes by the clean roof and omitted "
                "the 0.99 ms host gap. With it, the roof on today's bytes is "
                "66.54 TPS. 71 needs the roof AND about 7% fewer bytes, or the "
                "roof AND no host work at all."
            ),
        },
        "ladder": ladder(),
        "refuted_levers": list(REFUTED_LEVERS),
        "what_measurement_has_closed": (
            "Region granularity is refuted by measurement, not argued away: "
            "buffer contiguity and encoder count moved 331.6 to 332.2 GB/s. "
            "Representation decode has nothing to fuse; it is already "
            "in-register. The MLP code body is at its entropy floor, so perfect "
            "coding of what is stored recovers 2.8% of the token. And the entire "
            "host class is 0.99 ms. The demonstrated-497 rung stays in the ladder "
            "as an upper bound on executor work, but the one mechanism proposed "
            "for reaching it is dead."
        ),
        "experiments_ranked_by_gain": experiments(),
        "claim_boundary": (
            "Arithmetic over measured organ times, measured byte shares and a "
            "measured host gap. Every rung above 'measured now' is a TARGET, not "
            "an achievement: no organ other than the LM head has been shown to "
            "reach 497 GB/s, and no byte lever has passed a capability screen. "
            "The ms figures are DIAGNOSTIC_RELATIVE; the byte shares are exact."
        ),
    }


def record() -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n")
    return RECEIPT


if __name__ == "__main__":
    # AN UNKNOWN FLAG IS A REFUSAL, NOT A NO-OP. This is the third module found
    # with the same trap: --build printed a fresh table, exited 0 and wrote
    # NOTHING, so the terminal showed current numbers while the receipt stayed
    # stale. A tool that reports success without doing the work is the failure
    # this campaign keeps finding in its own checks.
    from _common import require_known_flags
    require_known_flags(["--build", "--record"])
    d = build()
    if "--record" in sys.argv or "--build" in sys.argv:
        print(f"wrote {record()}")
    for r in d["ladder"]:
        print(f"  {r['tps']:6.2f} TPS  {r['ms']:7.3f} ms   {r['rung']}")
    print()
    for e in d["experiments_ranked_by_gain"][:6]:
        print(f"  +{e['tps_gain']:5.2f} TPS  {e['cost']:32s} {e['id']}")

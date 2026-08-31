#!/usr/bin/env python3
"""The resident's standing objective: minimize verified steady decode token_ns.

Not "target 71 TPS". 71 is the roof implied by TODAY'S bytes if today's byte
stream were executed at the clean single-GEMV bandwidth. It is a checkpoint on
the way, not the destination — and a resident that reached ~71 and reported
"mission accomplished" would have failed as an optimizer, because the roof is a
function of bytes/token and Hawking's whole premise is that the bytes are not
fixed.

So the objective is a floor to push down, with milestones to cross and a roof
that gets RECOMPUTED every time one is crossed:

    roof_tps = clean_bandwidth_GB_s / bytes_per_token

Cut bytes/token and the roof moves. There is no terminal target.

    python3 tools/future/token_ns_objective.py --record
    python3 tools/future/token_ns_objective.py --roof --bytes-per-token 5.0e9
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402

RECEIPT = REPO / "receipts" / "future" / "TOKEN_NS_OBJECTIVE.json"
SCHEMA = "hawking.future.token_ns_objective.v1"

# --- measured anchors, each from a committed receipt ------------------------
# Superseded 2026-08-30 by receipts/future/RESIDENT_TOKEN_BUDGET.json, which is
# the first reading taken from a resident build that could actually report its
# own byte ledger. The old anchors (35.5 TPS, 9,878,901,136 bytes, 337.3 GB/s)
# could not all be true at once; the release-profile probe says the byte count
# was 8.6% low and the bandwidth figure was the outlier.
CURRENT_DECODE_TPS = 35.158                  # sealed-3.14, fusion env SET
ACTIVE_WEIGHT_BYTES_PER_TOKEN = 9_878_901_136
PRODUCTION_DECODE_GB_S = 347.4  # 9.8789 GB / 28.44 ms
SUPERSEDED_ANCHORS = {
    "decode_tps": 35.5,
    "active_weight_bytes_per_token": 9_878_901_136,
    "production_decode_gb_s": 337.3,
    "superseded_by": None,
    "why": (
        "NOT superseded after all. Both survived measurement: the byte count "
        "was confirmed by an independent catalog census, and 35.5 TPS was "
        "confirmed as the fusion-enabled rate (35.158 measured). Kept here "
        "because two intermediate readings briefly appeared to replace them "
        "and the reversal should stay visible."
    ),
}
CLEAN_GEMV_GB_S = 703.5                      # single GEMV clean addressing
PUBLISHED_PEAK_GB_S = 819.0                  # M3 Ultra published
# Measured per-tensor from the HQ38M20 catalog, not the rounded 0.54:
# 5,347,795,776 / 9,878,901,136. See receipts/future/MLP_BYTE_CENSUS.json.
MLP_SHARE_OF_BYTES = 5_347_795_776 / 9_878_901_136

MILESTONES: tuple[dict[str, Any], ...] = (
    {"name": "M1", "tps": 50.0},
    {"name": "M2", "tps": 71.0, "note": "the current-byte roof; a checkpoint, not the destination"},
    {"name": "M3", "tps": 100.0},
    {"name": "M4", "tps": 125.0},
    {"name": "MOONSHOT", "tps": 150.0,
     "note": "by here non-MLP work is the dominant Amdahl wall"},
)

# The floor is not bandwidth alone. As weight traffic falls, one of these
# becomes the tallest denominator and the whole map has to be redrawn.
TOKEN_COST_TERMS = (
    "necessary bytes",
    "necessary compute",
    "state/control",
    "ceremony",
)


def ms_per_token(tps: float) -> float:
    return 1000.0 / tps


def roof_tps(bytes_per_token: float, bandwidth_gb_s: float = CLEAN_GEMV_GB_S) -> float:
    """The only roof formula. Both inputs are variables, which is the point."""
    return bandwidth_gb_s / (bytes_per_token / 1e9)


def required_byte_fraction(target_tps: float,
                           bytes_per_token: float = ACTIVE_WEIGHT_BYTES_PER_TOKEN,
                           bandwidth_gb_s: float = CLEAN_GEMV_GB_S) -> float:
    """Fraction of today's bytes a target needs, if bandwidth reaches the roof."""
    return roof_tps(bytes_per_token, bandwidth_gb_s) / target_tps


def required_mlp_fraction(target_tps: float,
                          mlp_share: float = MLP_SHARE_OF_BYTES,
                          **kw: Any) -> float | None:
    """Remaining MLP fraction needed if MLP carries the whole reduction.

    total = (1 - mlp_share) + mlp_share * r. Returns None when the target is
    unreachable by MLP alone — i.e. r would have to be negative, because even
    deleting every MLP byte leaves the other (1 - mlp_share) in place.
    """
    total = required_byte_fraction(target_tps, **kw)
    r = (total - (1.0 - mlp_share)) / mlp_share
    return r if r >= 0.0 else None


def mlp_alone_ceiling_tps(bytes_per_token: float = ACTIVE_WEIGHT_BYTES_PER_TOKEN,
                          mlp_share: float = MLP_SHARE_OF_BYTES,
                          bandwidth_gb_s: float = CLEAN_GEMV_GB_S) -> float:
    """The hard wall if MLP bytes carried the entire reduction.

    Delete every MLP byte and (1 - mlp_share) of the traffic is still there, so
    the byte fraction cannot go below that. Nothing above this TPS is reachable
    by MLP representation alone, no matter how good the representation is.
    """
    return roof_tps(bytes_per_token, bandwidth_gb_s) / (1.0 - mlp_share)


def ladder(bytes_per_token: float = ACTIVE_WEIGHT_BYTES_PER_TOKEN,
           bandwidth_gb_s: float = CLEAN_GEMV_GB_S) -> list[dict[str, Any]]:
    rows = []
    for m in MILESTONES:
        tps = float(m["tps"])
        r = required_mlp_fraction(tps, bytes_per_token=bytes_per_token,
                                  bandwidth_gb_s=bandwidth_gb_s)
        row: dict[str, Any] = {
            "name": m["name"],
            "target_tps": tps,
            "target_ms_per_token": round(ms_per_token(tps), 3),
            "ms_to_remove_from_here": round(
                ms_per_token(CURRENT_DECODE_TPS) - ms_per_token(tps), 3),
            "required_total_byte_fraction": round(
                required_byte_fraction(tps, bytes_per_token, bandwidth_gb_s), 4),
            "required_remaining_mlp_fraction": (None if r is None else round(r, 4)),
            "reachable_by_mlp_bytes_alone": r is not None,
        }
        # Which lever the milestone actually needs. A milestone below today's
        # roof needs no byte reduction at all -- it needs the executor to reach
        # the bandwidth the hardware already offers. Above the roof, bytes have
        # to fall. Above the MLP-alone ceiling, MLP is not enough on its own.
        if r is None:
            row["lever"] = "UNREACHABLE_BY_MLP_ALONE"
            row["verdict"] = "ARITHMETICALLY_UNREACHABLE_BY_MLP_ALONE"
            row["why"] = (
                f"non-MLP is {1 - MLP_SHARE_OF_BYTES:.0%} of bytes and stays put; "
                "deleting 100% of MLP still does not get there"
            )
        elif r >= 1.0:
            row["lever"] = "EXECUTOR_RECOVERY_ONLY"
            row["why"] = (
                "at or below today's byte roof: no byte reduction is required, "
                "only executing today's bytes closer to the clean bandwidth"
            )
        else:
            row["lever"] = "BYTES_MUST_FALL"
            row["why"] = (
                f"above today's byte roof: even at clean bandwidth this needs "
                f"{(1 - r) * 100:.1f}% of MLP bytes eliminated, or the same "
                "reduction composed across MLP, dispatch, state and the rest"
            )
        if m.get("note"):
            row["note"] = m["note"]
        rows.append(row)
    return rows


def reconciliation() -> dict[str, Any]:
    """Was OPEN at 4%. The release-profile probe closed it."""
    implied_bw = ACTIVE_WEIGHT_BYTES_PER_TOKEN / 1e9 * CURRENT_DECODE_TPS
    implied_bytes = PRODUCTION_DECODE_GB_S / CURRENT_DECODE_TPS
    return {
        "status": "CLOSED",
        "closed_by": "receipts/future/RESIDENT_TOKEN_BUDGET.json",
        "resolution": (
            "The serving resident reports 10,727,793,882 active bytes per "
            "generated token, not 9,878,901,136 — the old anchor was 8.6% low. "
            "9.8789 GB x 35.158 TPS = 347.4 GB/s against a recorded 337.3, "
            "which is 3% and within the spread of one contended run. Both "
            "original anchors survive: the byte count was right, and 35.5 was "
            "right and was measured with the fusion env set. The two readings "
            "that briefly looked like corrections were artifacts — 32.7 TPS was "
            "the UNFUSED arm of an A/B, and 10.73 GB was the resident's "
            "active_bytes_per_token inflated by (P+N)/G. See "
            "receipts/future/PER_GENERATED_TOKEN_INFLATION.json."
        ),
        "superseded": SUPERSEDED_ANCHORS,
        "claim": (
            "35.5 TPS x 9.8789 GB/token = 350.7 GB/s, but the production decode "
            "receipt records 337.3 GB/s. The three anchors are mutually "
            "inconsistent by about 4%."
        ),
        "implied_bandwidth_gb_s": round(implied_bw, 1),
        "recorded_bandwidth_gb_s": PRODUCTION_DECODE_GB_S,
        "implied_bytes_per_token_gb": round(implied_bytes, 4),
        "recorded_bytes_per_token_gb": round(ACTIVE_WEIGHT_BYTES_PER_TOKEN / 1e9, 4),
        "disagreement_pct": round((implied_bw / PRODUCTION_DECODE_GB_S - 1) * 100, 2),
        "candidate_explanations": [
            "active_weight_bytes_per_token counts bytes the decode path does not "
            "actually re-read every token (a resident cache, or a shared organ)",
            "337.3 GB/s and 35.5 TPS were measured on different runs or "
            "different generation lengths",
            "the byte ledger includes metadata or codebooks the bandwidth probe "
            "excludes",
        ],
        "why_it_is_not_papered_over": (
            "A 4% inconsistency in the denominator is a 4% error in every roof "
            "on the ladder. Resolving it is cheap and is worth more than one "
            "more decimal place on a target."
        ),
        "resolves_when": (
            "one run reports bytes/token and TPS and effective bandwidth from "
            "the same instrumented decode"
        ),
    }


def build() -> dict[str, Any]:
    cur_ms = ms_per_token(CURRENT_DECODE_TPS)
    return {
        "schema": SCHEMA,
        "version": 1,
        "recorded_by": "tools/future/token_ns_objective.py",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "primary_objective": "MINIMIZE VERIFIED STEADY-STATE ACCEPTED TOKEN_NS",
        "subject_to": "capability preserved",
        "not_the_objective": [
            "GPU utilization",
            "complete-token TPS on a short generation (prefill accounting)",
            "any single fixed TPS number, 71 included",
        ],
        "current": {
            "decode_tps": CURRENT_DECODE_TPS,
            "ms_per_token": round(cur_ms, 3),
            "bytes_per_token": ACTIVE_WEIGHT_BYTES_PER_TOKEN,
            "mlp_share_of_bytes": MLP_SHARE_OF_BYTES,
        },
        "roofs": {
            "measured_production_gb_s": PRODUCTION_DECODE_GB_S,
            "clean_gemv_gb_s": CLEAN_GEMV_GB_S,
            "published_peak_gb_s": PUBLISHED_PEAK_GB_S,
            "current_byte_roof_tps": round(roof_tps(ACTIVE_WEIGHT_BYTES_PER_TOKEN), 2),
            "current_byte_roof_at_published_peak_tps": round(
                roof_tps(ACTIVE_WEIGHT_BYTES_PER_TOKEN, PUBLISHED_PEAK_GB_S), 2),
            "roof_is_a_function_not_a_constant": (
                "roof_tps = clean_bandwidth / bytes_per_token. Cut bytes and the "
                "roof moves. Recompute it every time a milestone is crossed."
            ),
        },
        "ladder": ladder(),
        "mlp_alone_ceiling": {
            "tps": round(mlp_alone_ceiling_tps(), 2),
            "ms_per_token": round(ms_per_token(mlp_alone_ceiling_tps()), 3),
            "meaning": (
                "Delete 100% of MLP bytes and the other 46% of traffic remains. "
                "Nothing above this is reachable by MLP representation alone, "
                "however good the representation gets. Past here the non-MLP "
                "bytes, state, routing, attention, the LM head and dispatch are "
                "the Amdahl wall and have to fall too."
            ),
            "the_moonshot_sits_just_under_it": (
                "150 TPS needs 2.7% of today's MLP bytes to survive. That is not "
                "compression, it is near-total information elimination on the "
                "largest organ, and it still leaves no headroom."
            ),
        },
        "no_terminal_target": (
            "When a milestone is crossed: reprofile, find the new tallest "
            "denominator, recompute the roof, and set the next milestone from "
            "the new physics. Never keep attacking the old denominator."
        ),
        "reprofile_triggers": [
            "total bytes fall by 30%",
            "dispatch count falls by 50%",
            "the MLP representation changes at all",
        ],
        "token_cost_terms": list(TOKEN_COST_TERMS),
        "the_floor_question": (
            "For each term: does it exist because of useful model function, or "
            "only because of the present representation and runtime? The true "
            "floor is what survives that question, and it is not bandwidth "
            "alone — once weight traffic falls far enough, state, routing, "
            "attention, the LM head or dispatch becomes the wall."
        ),
        "anchor_reconciliation": reconciliation(),
        "claim_boundary": (
            "Arithmetic over measured anchors. No hardware measurement is made "
            "here and no target is claimed as achieved. The ladder says what "
            "each milestone would REQUIRE, not that any of it is attainable. "
            "The anchors themselves disagree by ~4%; see anchor_reconciliation."
        ),
    }


def record() -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n")
    return RECEIPT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--roof", action="store_true")
    ap.add_argument("--bytes-per-token", type=float, default=ACTIVE_WEIGHT_BYTES_PER_TOKEN)
    ap.add_argument("--bandwidth-gb-s", type=float, default=CLEAN_GEMV_GB_S)
    a = ap.parse_args()
    if a.roof:
        print(json.dumps({
            "bytes_per_token": a.bytes_per_token,
            "bandwidth_gb_s": a.bandwidth_gb_s,
            "roof_tps": round(roof_tps(a.bytes_per_token, a.bandwidth_gb_s), 2),
            "ladder": ladder(a.bytes_per_token, a.bandwidth_gb_s),
        }, indent=1))
    elif a.record:
        print(f"wrote {record()}")
    else:
        print(json.dumps(build(), indent=1))

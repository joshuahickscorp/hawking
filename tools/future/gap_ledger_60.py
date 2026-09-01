#!/usr/bin/env python3
"""G095: the 60-TPS gap, per-organ win table, and the escalation clock.

S026 makes 60 TPS the intermediate attack target and 71 the convergence target.
This module computes both gaps from disk authority rather than from a typed
number, prices every dominant organ in COMPLETE-TOKEN TPS at 10/20/30% and at
perfect removal, ranks the live experiment set by MAX MS REMOVABLE, and runs the
escalation clock S026 §48 requires.

THE CLOCK DOES NOT FORCE ABANDONMENT. It states what each phase licenses, so
that continuing to work the incumbent past +18h is a decision someone made
rather than a default nobody noticed.

    python3 tools/future/gap_ledger_60.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402
import causal_budget_71 as cb  # noqa: E402

RECORDED_BY = "tools/future/gap_ledger_60.py"
RECEIPT_NAME = "GAP_LEDGER_60.json"
BUDGET_REL = "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"
# G131 rebased the baseline. The three levers are the SEALED DEFAULT now, so the
# gap must be measured from what the resident actually costs with nothing set,
# not from the pre-promotion budget - every rung below was arithmetic over a
# number that is 5.3 ms stale.
ABSOLUTE_REL = "receipts/future/SEALED_DEFAULT_ABSOLUTE.json"
CLOCK_REL = "receipts/future/TPS_ESCALATION_CLOCK_START.json"

CHECKPOINTS = (40.0, 50.0, 60.0, 71.0)

# S025's materiality threshold. Below this a perfect win is not worth an hour.
MATERIAL_MS = 1.0

# S026 §48. Each phase names what it LICENSES, not what it forbids.
PHASES = (
    (0.0, "DISCOVERY", "incumbent-kernel work is the privileged route"),
    (6.0, "SURVIVOR_OR_COLLAPSE",
     "require a credible multi-ms survivor OR a major search-space collapse; "
     "if neither, the absence is itself the result and must be recorded"),
    (12.0, "WIDEN",
     "activate the alternate-resident and decoding schools strongly, in "
     "parallel with incumbent work rather than instead of it"),
    (18.0, "DEPRIVILEGE",
     "if the incumbent is still under 50 TPS with no credible path, stop "
     "treating incumbent-only optimization as privileged"),
)


class GapRefused(RuntimeError):
    """Disk authority is missing or self-inconsistent."""


def _budget() -> dict[str, Any]:
    p = REPO / BUDGET_REL
    if not p.is_file():
        raise GapRefused(f"{BUDGET_REL} is not on disk; there is no live token budget")
    d = json.loads(p.read_text())
    for k in ("decode_wall_ms_per_token", "decode_wall_tps", "organs"):
        if k not in d:
            raise GapRefused(f"{BUDGET_REL} is missing {k}")
    return d


def live() -> dict[str, Any]:
    """The baseline every rung is measured from.

    ON GPU TIME, NOT WALL. G125 showed the two harnesses agree about GPU to
    0.20% and disagree about host gap by 3.4x, and G131 saw that gap span 4.7x
    across three windows while GPU moved under 0.3%. Wall is a joint claim about
    the resident AND the machine's other tenants. Every lever in this ledger is
    measured as GPU ms, so a gap denominated in wall was comparing two different
    quantities. The wall figure is still reported, beside its host gap.
    """
    p = REPO / ABSOLUTE_REL
    if p.is_file():
        a = json.loads(p.read_text())
        m = a["measured"]
        ms = float(m["gpu_ms_per_token"])
        return {
            "ms_per_token": ms,
            "tps": float(m["gpu_tps"]),
            "basis": "GPU",
            "gpu_ms_per_token": ms,
            "wall_ms_per_token": float(m["wall_ms_per_token"]),
            "wall_tps": float(m["wall_tps"]),
            "host_gap_ms": float(m["host_gap_ms"]),
            "wall_is_not_promoted": (
                "wall = gpu + host and the host gap spans 4.7x across protected "
                "windows; see receipts/future/HARNESS_RECONCILIATION.json"
            ),
            "levers_are_the_sealed_default": True,
            "source": ABSOLUTE_REL,
            "git_head": a.get("git_head"),
        }
    d = _budget()
    ms = float(d["decode_wall_ms_per_token"])
    tps = float(d["decode_wall_tps"])
    # The receipt carries both; if they disagree the receipt is not authority.
    if abs(1000.0 / ms - tps) > 0.05:
        raise GapRefused(
            f"{BUDGET_REL} is self-inconsistent: {ms} ms/token implies "
            f"{1000.0/ms:.3f} TPS but the receipt says {tps}"
        )
    return {
        "ms_per_token": ms,
        "tps": tps,
        "basis": "WALL_PRE_PROMOTION_FALLBACK",
        "gpu_ms_per_token": float(d["decode_gpu_ms_per_token"]),
        "source": BUDGET_REL,
        "git_head": d["organs"].get("git_head"),
    }


def checkpoints() -> list[dict[str, Any]]:
    cur = live()["ms_per_token"]
    out = []
    for t in CHECKPOINTS:
        need = 1000.0 / t
        out.append({
            "tps": t,
            "ms_per_token_required": round(need, 4),
            "ms_to_remove_from_live": round(cur - need, 4),
            "fraction_of_token_to_remove": round((cur - need) / cur, 4),
            "reached": cur <= need,
        })
    return out


def organ_decomposition_is_stale() -> dict[str, Any]:
    """The organ table is measured on a body that no longer runs.

    G131 promoted three levers to sealed defaults and the resident now costs
    21.9464 ms GPU. The organ rows still sum to 26.7013 - they were measured
    before the promotion. Nothing in this ledger silently rescales them, and the
    difference is named here as the next measurement rather than absorbed.
    """
    d = _budget()
    rows_total = sum(float(r["gpu_ms"]) for r in d["organs"]["rows"])
    lv = live()
    delta = rows_total - lv["ms_per_token"]
    return {
        "organ_rows_sum_ms": round(rows_total, 4),
        "live_baseline_ms": lv["ms_per_token"],
        "stale_by_ms": round(delta, 4),
        "stale_by_share": round(delta / rows_total, 4),
        "source": BUDGET_REL,
        "what_it_means": (
            "the organ table prices wins against its OWN total, not against the "
            "live baseline. Its ranking is still usable - the levers did not "
            "reorder the organs - but no absolute TPS-at-a-win figure in that "
            "table is current."
        ),
        "next_measurement": (
            "re-run the organ decomposition under the sealed default so the "
            "table and the baseline describe the same body"
        ),
    }


def organ_win_table() -> list[dict[str, Any]]:
    """S026 §12: every organ priced in COMPLETE-TOKEN TPS, not in organ percent."""
    d = _budget()
    # PRICED AGAINST ITS OWN BODY, NOT THE LIVE BASELINE. These organ times were
    # measured before the three levers became the sealed default: they sum to
    # 26.7013 ms against a live GPU baseline of 21.9464. Dividing a stale
    # numerator by a live denominator would inflate every organ's share and
    # every TPS-at-a-win figure. The decomposition is self-consistent within the
    # body that produced it, so it is priced there and flagged as stale.
    cur = sum(float(r["gpu_ms"]) for r in d["organs"]["rows"])
    rows = []
    for r in d["organs"]["rows"]:
        ms = float(r["gpu_ms"])
        entry = {
            "organ": r["organ"],
            "current_ms": ms,
            "share_of_token": round(ms / cur, 4),
            "dispatches": r["dispatches"],
        }
        for pct in (10, 20, 30, 100):
            saved = ms * pct / 100.0
            entry[f"tps_at_{pct}pct_win"] = round(1000.0 / (cur - saved), 3)
            entry[f"ms_at_{pct}pct_win"] = round(saved, 4)
        entry["perfect_removal_max_ms"] = round(ms, 4)
        entry["priced_against"] = {
            "source": BUDGET_REL,
            "token_ms": round(cur, 4),
            "basis": "the organ decomposition's own total, GPU ms",
            "not_the_live_baseline_because": (
                "these organs predate the sealed-default promotion; see "
                "organ_decomposition_is_stale"
            ),
        }
        entry["material"] = ms >= MATERIAL_MS
        entry["why"] = (
            "perfect elimination is below the materiality threshold; do not "
            f"spend an hour here" if ms < MATERIAL_MS else
            f"a 20% win here is {round(ms*0.2,3)} ms of the {round(cur,3)} ms token"
        )
        rows.append(entry)
    rows.sort(key=lambda r: -r["current_ms"])
    return rows


def three_dominant() -> dict[str, Any]:
    """S026 §11: mlp_gate_up, mlp_down and DeltaNet dominate. Price them jointly."""
    d = _budget()
    cur = float(d["decode_wall_ms_per_token"])
    want = {"mlp_gate_up", "mlp_down", "deltanet"}
    rows = [r for r in d["organs"]["rows"] if r["organ"] in want]
    if len(rows) != 3:
        raise GapRefused(
            f"expected the three dominant organs {sorted(want)}, found "
            f"{sorted(r['organ'] for r in rows)}"
        )
    total = sum(float(r["gpu_ms"]) for r in rows)
    out = {
        "organs": sorted(want),
        "combined_ms": round(total, 4),
        "share_of_token": round(total / cur, 4),
    }
    for pct in (10, 20, 30):
        saved = total * pct / 100.0
        out[f"tps_at_{pct}pct_across_all_three"] = round(1000.0 / (cur - saved), 3)
        out[f"ms_at_{pct}pct_across_all_three"] = round(saved, 4)
    out["reading"] = (
        f"these three are {out['share_of_token']:.1%} of the token. A 20% win "
        f"across all three is {out['ms_at_20pct_across_all_three']} ms, which "
        f"takes the LIVE resident to {out['tps_at_20pct_across_all_three']} TPS - "
        "not to 60 on its own. S026's arithmetic reaching 60 assumed the ~49.8 "
        "PROSPECTIVE composed path as the base, and that path is not qualified."
    )
    return out


def ranked_experiments() -> list[dict[str, Any]]:
    """S026 §H: rank every live experiment by MAX MS REMOVABLE."""
    out = []
    for e in cb.experiments():
        ms = float(e.get("ms_saved", 0.0))
        out.append({
            "id": e["id"],
            "max_ms_removable": ms,
            "material": ms >= MATERIAL_MS,
            "status": e.get("status") or e.get("lever_status") or "OPEN",
            "organ": e.get("organ"),
        })
    out.sort(key=lambda r: -r["max_ms_removable"])
    return out


def _clock_start() -> dict[str, Any]:
    """Stamped once. A clock that restarts every build measures nothing."""
    p = REPO / CLOCK_REL
    if p.is_file():
        return json.loads(p.read_text())
    doc = {
        "started_unix": time.time(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "started_at_tps": live()["tps"],
        "authority": "S026 §48",
        "note": (
            "stamped once and never rewritten. Restarting this clock on every "
            "build would make every phase permanently DISCOVERY."
        ),
    }
    p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def escalation_clock() -> dict[str, Any]:
    start = _clock_start()
    elapsed_h = (time.time() - float(start["started_unix"])) / 3600.0
    phase = PHASES[0]
    for p in PHASES:
        if elapsed_h >= p[0]:
            phase = p
    nxt = next((p for p in PHASES if p[0] > elapsed_h), None)
    cur_tps = live()["tps"]
    return {
        "started_utc": start["started_utc"],
        "started_at_tps": start["started_at_tps"],
        "elapsed_hours": round(elapsed_h, 3),
        "phase": phase[1],
        "phase_licenses": phase[2],
        "hours_to_next_phase": round(nxt[0] - elapsed_h, 3) if nxt else None,
        "next_phase": nxt[1] if nxt else None,
        "tps_moved_since_start": round(cur_tps - float(start["started_at_tps"]), 3),
        "deprivilege_condition_would_fire": cur_tps < 50.0,
        "not_an_abandonment": (
            "the clock forbids complacency, not the incumbent. Passing a phase "
            "boundary licenses widening the search; it never requires dropping "
            "work that is producing evidence."
        ),
    }


# arm_a / production, MEASURED at warmup 60 on real captured activations. These
# are the rate each matvec reaches with ALL of its decode arithmetic removed and
# every byte still loaded, so they bound the entire arithmetic school.
ARM_A_RATIO = {
    "q2": (1.5128, ("mlp_gate_up", "mlp_down"),
           "receipts/future/OP_CLASS_ABLATION.json"),
    "q4": (1.5599, ("q4_remainder", "gqa_attention"),
           "receipts/future/Q4_BITCAST_AB.json"),
}

# The DeltaNet ORGAN is 5.5971 ms but only its in-projection is a q4 matvec.
# The state update, rearrange and gated norm are different kernels with
# different arithmetic, so applying the q4 arm_a ratio to the whole organ
# OVERSTATES the ceiling. Measured isolated component, same receipt family.
DN_INPROJ_MS = 3.5885
DN_INPROJ_SOURCE = "receipts/future/_G094_RESIDENT_CTRL_raw.json isolated_components.dn_inproj"


def arithmetic_ceiling() -> dict[str, Any]:
    """What the WHOLE decode-arithmetic school is worth if it perfectly succeeds.

    Not a projection from a candidate. arm_a removes every arithmetic operation
    from the inner loop while loading every byte production loads, so its rate
    is the measured floor on time for these kernels at these bytes.
    """
    d = _budget()
    cur = float(d["decode_wall_ms_per_token"])
    rows = {r["organ"]: float(r["gpu_ms"]) for r in d["organs"]["rows"]}
    parts = []
    total = 0.0
    for codec, (ratio, organs, source) in ARM_A_RATIO.items():
        missing = [o for o in organs if o not in rows]
        if missing:
            raise GapRefused(f"{codec}: budget has no rows for {missing}")
        ms = sum(rows[o] for o in organs)
        if codec == "q4":
            # DeltaNet contributes only its in-projection, not the whole organ.
            ms += DN_INPROJ_MS
        saved = ms - ms / ratio
        total += saved
        parts.append({
            "codec": codec,
            "organs": list(organs),
            "organ_ms": round(ms, 4),
            "arm_a_over_production": ratio,
            "ms_saved_at_perfect_removal": round(saved, 4),
            "source": source,
            "deltanet_in_projection_only_ms": DN_INPROJ_MS if codec == "q4" else None,
            "deltanet_note": (
                "the DeltaNet ORGAN is 5.5971 ms; only its 3.5885 ms "
                "in-projection is a q4 matvec. The state update, rearrange and "
                "gated norm are different kernels and are NOT counted here. "
                "Counting the whole organ would overstate this ceiling by "
                "about 0.72 ms."
            ) if codec == "q4" else None,
        })
    after = cur - total
    return {
        "parts": parts,
        "total_ms_at_perfect_removal": round(total, 4),
        "token_ms_after": round(after, 4),
        "tps_after": round(1000.0 / after, 3),
        "reaches_60": after <= 1000.0 / 60.0,
        "still_short_of_60_by_ms": round(after - 1000.0 / 60.0, 4),
        "verdict": (
            "THE DECODE-ARITHMETIC SCHOOL CANNOT REACH 60 EVEN IF IT PERFECTLY "
            f"SUCCEEDS. Removing every arithmetic operation from every q2 and q4 "
            f"matvec reaches {round(1000.0/after, 1)} TPS. 60 needs "
            f"{round(after - 1000.0/60.0, 2)} ms from a DIFFERENT class - fewer "
            "bytes, less host time, or work removed rather than made cheaper."
        ),
        "what_this_does_not_say": (
            "it does not say the arithmetic school is not worth working. It is "
            "the largest single block on the board and the bitcast candidates "
            "have already taken a measured 3.854 ms of it. It says the school "
            "must not be the ONLY thing running if 60 is the target."
        ),
        "assumption": (
            "arm_a's ratio measured on one representative projection per codec "
            "is applied to every organ that runs that codec. Organs differ in "
            "shape, so this is an estimate at the organ level even though each "
            "ratio is measured. It would take a per-organ arm_a to tighten."
        ),
    }


def bytes_required_after_arithmetic() -> dict[str, Any]:
    """If arithmetic perfectly succeeds, what does 60 then demand of the BYTES?

    At arm_a the matvecs are pure streaming: every arithmetic operation is gone
    and only the loads remain. Any further saving must therefore come from
    reading fewer bytes, and the fraction is computable rather than guessed.
    """
    a = arithmetic_ceiling()
    d = _budget()
    rows = {r["organ"]: float(r["gpu_ms"]) for r in d["organs"]["rows"]}
    streaming_ms = 0.0
    for part in a["parts"]:
        ms = sum(rows[o] for o in part["organs"])
        if part["codec"] == "q4":
            ms += DN_INPROJ_MS
        streaming_ms += ms / part["arm_a_over_production"]
    need = a["token_ms_after"] - 1000.0 / 60.0
    if need <= 0:
        return {"already_reached": True}
    frac = need / streaming_ms
    return {
        "streaming_ms_at_arm_a": round(streaming_ms, 4),
        "further_ms_needed_for_60": round(need, 4),
        "fraction_of_matvec_bytes_to_remove": round(frac, 4),
        "statement": (
            f"after perfect arithmetic removal the q2 and q4 matvecs are "
            f"{round(streaming_ms, 2)} ms of pure streaming. Reaching 60 needs "
            f"{round(need, 2)} ms more, which at that point can only come from "
            f"reading fewer bytes: {round(frac * 100, 1)}% of the matvec weight "
            "bytes must stop existing."
        ),
        "why_entropy_coding_does_not_supply_it": (
            "the MLP 2-bit code body measures ~1.87 bits of entropy per 2 stored "
            "bits and the codes are 93.5% independent information, so ordinary "
            "coding has single-digit percent to give, not 17%. A cut this size "
            "is INFORMATION ELIMINATION - a smaller executable program for the "
            "same function - which is the ULTRAGOAL's stated primary weapon."
        ),
        "the_other_place_to_look": (
            "this accounting covers only the q2 and q4 matvecs. The DeltaNet "
            "state update, rearrange and gated norm are about 2.0 ms and sit "
            "OUTSIDE it entirely, as do lm_head at 1.0204 ms and sampling at "
            "0.334 ms. G053 - DeltaNet as a state machine rather than four "
            "matrices - is aimed at exactly that 2.0 ms and is still open."
        ),
        "caveat": (
            "this assumes arithmetic removal and byte removal compose, which "
            "they do not automatically: a representation that removes bytes "
            "usually changes the arithmetic too, and the accelerator has to be "
            "reopened after every representation change."
        ),
    }


# Levers that are BUILT and MEASURED on the real graph but DEFAULT-OFF. They are
# not part of `live` - the resident still runs production unless the env says
# otherwise - and pretending otherwise would be the fake-completion this campaign
# forbids. Each cites the receipt that measured it on the complete token.
BUILT_NOT_PROMOTED = (
    {
        "lever": "HAWKING_AFFINE2_GEO=bitcast",
        "gpu_ms_saved": 3.8541,
        "token_identical": True,
        "receipt": "receipts/future/BITCAST_DEQUANT_AB.json",
    },
    {
        "lever": "HAWKING_Q4_UNPACK=bitcast",
        "gpu_ms_saved": 0.6836,
        "token_identical": True,
        "receipt": "receipts/future/Q4_BITCAST_AB.json",
    },
)


def available_now() -> dict[str, Any]:
    """What the resident would be if the built levers were switched on.

    Kept SEPARATE from live() on purpose. Both levers default to off, so the
    resident is still 36.644 TPS; folding them into the live number would claim
    a promotion that has not happened.
    """
    lv = live()
    total = sum(x["gpu_ms_saved"] for x in BUILT_NOT_PROMOTED)
    after = lv["ms_per_token"] - total
    return {
        "levers": list(BUILT_NOT_PROMOTED),
        "combined_ms_saved": round(total, 4),
        "all_token_identical": all(x["token_identical"] for x in BUILT_NOT_PROMOTED),
        "live_tps": lv["tps"],
        "tps_if_promoted": round(1000.0 / after, 3),
        "ms_if_promoted": round(after, 4),
        "still_short_of_60_by_ms": round(after - 1000.0 / 60.0, 4),
        "why_not_promoted": (
            "both are opt-in env levers and production default is unchanged. "
            "Promotion needs a protected absolute measurement to replace "
            f"{BUDGET_REL}, and every window so far has had ModelLake downloads "
            "running - twice after they were SIGSTOPped and the supervisor "
            "respawned them."
        ),
        "what_promotion_requires": (
            "a protected reprofile with both levers set, replacing the resident "
            "token budget; then the default flips and this section empties"
        ),
    }


def build() -> dict[str, Any]:
    lv = live()
    cps = checkpoints()
    sixty = next(c for c in cps if c["tps"] == 60.0)
    seventyone = next(c for c in cps if c["tps"] == 71.0)
    ranked = ranked_experiments()
    avail = available_now()
    material = [r for r in ranked if r["material"] and r["status"] not in ("CLOSED",)]
    return {
        "obligation": "G095",
        "authority": "S026 §0, §1, §11, §12, §48",
        "live": lv,
        "checkpoints": cps,
        "gap_to_60_ms": sixty["ms_to_remove_from_live"],
        "gap_to_71_ms": seventyone["ms_to_remove_from_live"],
        "organ_win_table": organ_win_table(),
        "three_dominant_kernels": three_dominant(),
        "ranked_experiments": ranked,
        "built_not_promoted": avail,
        "ranked_experiments_are_gross_not_net": (
            "the bandwidth-recovery ceilings above are computed against the "
            "PRODUCTION kernels. The built-not-promoted levers have already "
            f"taken {avail['combined_ms_saved']} ms of that headroom, so the "
            "ranking overstates what remains. It is kept gross because the "
            "levers are not promoted and the resident still runs production."
        ),
        "material_open_experiments": material,
        "organ_decomposition_is_stale": organ_decomposition_is_stale(),
        "sum_of_material_open_ms": round(
            sum(r["max_ms_removable"] for r in material), 4),
        "does_the_open_set_reach_60": (
            sum(r["max_ms_removable"] for r in material) >= sixty["ms_to_remove_from_live"]
        ),
        "arithmetic_ceiling": arithmetic_ceiling(),
        "bytes_required_after_arithmetic": bytes_required_after_arithmetic(),
        "escalation_clock": escalation_clock(),
        "honest_reading": (
            "the material open experiments are bandwidth-recovery bounds - what "
            "each organ would cost if it ran at a rate already demonstrated "
            "elsewhere on this machine. They are CEILINGS, not candidates: no "
            "mechanism that reaches them is on record. Summing them and "
            "comparing to the gap says whether the gap is closable IN "
            "PRINCIPLE by the open set, not whether it is closable today."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        # This ledger MEASURES NOTHING. Every hardware number in it is read out
        # of RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json, which carries its own
        # provenance. Stamping this build time as a measurement time would be a
        # fabricated provenance, so it is marked a retrofit of that receipt.
        print(write_measured_receipt(
            REPO / "receipts" / "future" / RECEIPT_NAME, doc, RECORDED_BY,
            provenance=measurement_provenance(
                lock_held=False, lane="derived", retrofit=True),
        ))
        return 0
    print(json.dumps({k: doc[k] for k in (
        "live", "gap_to_60_ms", "gap_to_71_ms", "three_dominant_kernels",
        "material_open_experiments", "sum_of_material_open_ms",
        "does_the_open_set_reach_60", "escalation_clock")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

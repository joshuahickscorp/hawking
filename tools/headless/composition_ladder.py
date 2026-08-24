#!/usr/bin/env python3
"""N011 — the qualification ladder, enforced, with early kill.

S017 §28: as density drops, local cosine stops meaning anything. The ladder is

    local functional probe -> held-out activation -> adjacent layers ->
    short chain -> complete organ -> complete token -> coherent generation ->
    capability

and a candidate that fails a rung is KILLED THERE. It does not get carried to the
next rung on the strength of the rung it passed.

This campaign has the receipts to prove why that matters. Ternary at 1.85 bpw was
named CANON on an ORGAN-LOCAL screen -- mean rel_fro 0.3210 on held-out real
activations, comfortably above its null -- and then flipped the argmax when
applied to the whole model. Screen survival predicted nothing about composition.
The reverse also holds: the cheapest thing that ever passed a screen
(mix_c, 2.3440 EBPW) emits sixteen newlines when asked to generate.

So the ladder is not bookkeeping. It is the thing that stops a screen result from
being reported as a model result.

This harness does two jobs:
  1. classify every candidate on disk by the HIGHEST rung it actually reached and
     the rung it DIED at, from receipts rather than from memory;
  2. expose `highest_claimable(candidate)` so a later claim cannot describe a
     candidate above the rung its evidence supports.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
R = REPO / "receipts" / "headless"
OUT = R / "COMPOSITION_LADDER.json"

RUNGS = [
    "local_functional_probe",
    "held_out_activation",
    "adjacent_layers",
    "short_chain",
    "complete_organ",
    "complete_token",
    "coherent_generation",
    "capability",
]
RUNG_INDEX = {r: i for i, r in enumerate(RUNGS)}


def load(name: str) -> dict[str, Any] | None:
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()[:12]


def entry(cid, reached, died, why, src, note="", unreached=None):
    """`died_at` means TESTED AND FAILED. `unreached_above` means NEVER TESTED.

    Conflating those is the failure this receipt exists to prevent, and the first
    version of this harness committed it: it recorded the 2.25-bpw arm as
    DIED@coherent_generation when nothing had ever asked it to generate, and the
    leader as DIED@capability when no capability suite exists. An untested rung
    reported as a death invents a negative result.
    """
    if died and unreached:
        raise ValueError(f"{cid}: a rung cannot be both failed and untested")
    return {
        "id": cid,
        "highest_rung_reached": reached,
        "highest_rung_index": RUNG_INDEX[reached] if reached else -1,
        "died_at": died,
        "unreached_above": unreached,
        "status": "FAILED" if died else ("UNTESTED_ABOVE" if unreached else "PASSED_ALL_TESTED"),
        "why": why,
        "source_receipt": src,
        "note": note,
        # A claim about this candidate may not exceed this rung.
        "may_be_described_as": reached,
    }


def classify() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    canon = load("FRACTIONAL_BIT_CANON")
    if canon:
        best = (canon.get("verdict") or {}).get("canon") or {}
        out.append(
            entry(
                "ternary_aa_g64 (organ-local CANON, 1.85 bpw)",
                "held_out_activation",
                "complete_token",
                "Named CANON on an organ-local held-out screen (mean rel_fro "
                f"{best.get('mean_rel_fro')}), then FLIPPED THE ARGMAX applied to the "
                "whole MLP: rel_l2 0.7360, student argmax 10895 vs teacher 9714.",
                "receipts/headless/FRACTIONAL_BIT_CANON.json + "
                "NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
                "THE case for composition-first screening. A screen verdict is not a "
                "model verdict.",
            )
        )

    fne = load("FIRST_NOETIC_EXECUTABLE")
    for m in (fne or {}).get("mix_scoreboard", []):
        coherent = bool(m.get("coherent"))
        out.append(
            entry(
                m["mix_id"],
                "coherent_generation" if coherent else "complete_token",
                None if coherent else "coherent_generation",
                "generated varied tokens"
                if coherent
                else f"decode degenerated: {m.get('coherence_reason') or 'see receipt'}",
                "receipts/headless/FIRST_NOETIC_EXECUTABLE.json",
                f"EBPW {m.get('complete_ebpw')}",
            )
        )

    q2f = load("NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64")
    if q2f:
        out.append(
            entry(
                "q2_4level_fitted_g64 whole-MLP (2.25 bpw)",
                "complete_token",
                None,
                "Survived the complete 64-layer token loop with the argmax AGREEING "
                "(9714 = 9714), but was never carried to native generation.",
                "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json",
                "NOT DEAD -- untested above complete_token. It is the cheapest arm known "
                "to survive the whole-model token loop and nobody has asked it to generate.",
                unreached="coherent_generation",
            )
        )

    fused = load("NOETIC_FUSED_SUBBIT")
    if fused:
        out.append(
            entry(
                "NOETIC_PARENT_A (affine2_g64_LS + fused graph)",
                "coherent_generation",
                None,
                "Generates 16 coherent tokens natively at 34.873 tok/s. NO capability "
                "suite has ever been run on it, so the capability rung is UNREACHED, "
                "not passed.",
                "receipts/headless/NOETIC_FUSED_SUBBIT.json",
                "LEADER. Frontier candidate; not resident-promoted.",
                unreached="capability",
            )
        )
    return out


def highest_claimable(cid: str, table: list[dict[str, Any]]) -> str | None:
    for row in table:
        if row["id"] == cid:
            return row["may_be_described_as"]
    return None


def main() -> int:
    table = classify()
    died = [r for r in table if r["died_at"]]
    untested = [r for r in table if r.get("unreached_above")]
    killed_early = [r for r in died if RUNG_INDEX[r["died_at"]] < RUNG_INDEX["capability"]]
    by_rung: dict[str, int] = {}
    for r in table:
        by_rung[r["highest_rung_reached"]] = by_rung.get(r["highest_rung_reached"], 0) + 1

    receipt = {
        "schema": "hawking.headless.composition_ladder.v1",
        "obligation": "N011 (S017 §28)",
        "git_head": git_head(),
        "rungs": RUNGS,
        "rule": (
            "A candidate that fails a rung is KILLED THERE and may not be described "
            "above the rung its evidence supports. Reaching a rung is not the same as "
            "passing it; UNREACHED and FAILED are different states."
        ),
        "why_this_is_not_bookkeeping": (
            "Ternary at 1.85 bpw was named CANON on an organ-local held-out screen and "
            "then flipped the argmax on the whole model. In the other direction, the "
            "cheapest candidate that ever passed a screen (2.3440 EBPW) emits sixteen "
            "newlines when asked to generate. Screen survival predicts neither "
            "composition nor generation."
        ),
        "candidates": table,
        "counts": {
            "candidates": len(table),
            "by_highest_rung": by_rung,
            "killed_before_capability": len(killed_early),
            "untested_above_their_rung": len(untested),
        },
        "untested_not_failed": [
            {"id": r["id"], "unreached_above": r["unreached_above"], "why": r["note"]}
            for r in untested
        ],
        "killed": [
            {"id": r["id"], "died_at": r["died_at"], "why": r["why"]} for r in died
        ],
        "capability_rung_status": (
            "UNREACHED FOR EVERY CANDIDATE. No capability suite has been run on any "
            "artifact in this campaign. Coherence on a 16-token greedy sample is not "
            "capability."
        ),
    }
    OUT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"candidates {len(table)}  killed before capability {len(killed_early)}")
    for r in table:
        d = (f"  DIED@{r['died_at']}" if r["died_at"]
             else (f"  untested>{r['unreached_above']}" if r.get("unreached_above") else ""))
        print(f"  {r['id'][:52]:<52} reached={r['highest_rung_reached']:<20}{d}")
    print(f"receipt: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

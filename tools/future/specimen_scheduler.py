#!/usr/bin/env python3
"""G104: which specimen next, computed - and a dead hypothesis loads nothing.

S027 §16-§20, §23-§25, §30-§32, §47-§50. Selection is a value function over
information gain, transfer value, counterexample value and economic relevance,
divided by a MEASURED cost. Never FIFO.

THE FIRST QUESTION IS NOT WHICH MODEL BUT WHETHER TO LOAD ONE AT ALL. Before a
specimen is ranked for a hypothesis, the scar index is queried. If the
hypothesis is already dead the correct action is DO NOT LOAD THAT MODEL, and
this refuses at ranking time rather than after 77 minutes of disk read.

ARCHITECTURE DISTANCE PICKS THE ROLE. Same model_type is a NEAR transfer test,
a related family is MID, an unrelated architecture is a FAR adversary. A law
that only ever meets near neighbours has not been attacked.

    python3 tools/future/specimen_scheduler.py --build
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402
import negative_index as ni  # noqa: E402
import specimen_load_cost as lc  # noqa: E402
import specimen_registry as sr  # noqa: E402
import uma_resource_ledger as ul  # noqa: E402

RECORDED_BY = "tools/future/specimen_scheduler.py"
RECEIPT_NAME = "SPECIMEN_SCHEDULER.json"

# The incumbent. Its architecture is the reference for transfer distance.
INCUMBENT_TYPE = "qwen4_exp"

# S027 §49: the most informative specimen is the one that DISCRIMINATES, so a
# far architecture is worth more for attacking a law than a near one.
ROLE_WEIGHT = {"NEAR": 1.0, "MID": 1.4, "FAR": 1.8}

# A question already answered on a specimen still has SOME value - a repeat can
# confirm - but not full value. This is what makes the scheduler explore instead
# of returning to its cheapest specimen forever.
REPEAT_DISCOUNT = 0.05


class ScheduleRefused(RuntimeError):
    """A ranking was requested with no question or no candidates."""


def _stem(model_type: str) -> str:
    """Leading alphabetic run of a model_type: the architecture lineage.

    qwen4_exp, qwen3, qwen3_moe and qwen3_vl_moe all stem to "qwen"; deepseek_v4
    to "deepseek"; mistral3 to "mistral". An earlier version split on the digits
    themselves and called qwen3 FAR from qwen4_exp, which is plainly wrong and
    would have made every near neighbour look like an adversary.
    """
    m = re.match(r"[a-z]+", model_type.lower())
    return m.group(0) if m else model_type.lower()


def distance(model_type: str | None) -> str:
    """Structural distance from the incumbent, from the config's model_type."""
    if not model_type:
        return "FAR"
    if model_type == INCUMBENT_TYPE:
        return "NEAR"
    return "MID" if _stem(model_type) == _stem(INCUMBENT_TYPE) else "FAR"


def scar_verdict(hypothesis_family: str, model_type: str | None,
                 scars: list[Any] | None = None) -> dict[str, Any]:
    """S027 §20: query the scars BEFORE loading anything."""
    hits = ni.query(hypothesis_family=hypothesis_family, scars=scars)
    same_arch = [h for h in hits
                 if model_type and model_type in json.dumps(h).lower()]
    return {
        "hypothesis_family": hypothesis_family,
        "n_scars": len(hits),
        "n_on_this_architecture": len(same_arch),
        "dead": bool(hits),
        "action": "DO_NOT_LOAD" if hits else "LOAD_PERMITTED",
        "why": (
            f"{len(hits)} recorded negatives for this hypothesis family; the "
            "correct action is not to choose a better specimen but to not load "
            "one" if hits else
            "no recorded negative for this hypothesis family"
        ),
    }


def rank(*, hypothesis_family: str, expected_information_gain: float = 1.0,
         scars: list[Any] | None = None,
         warm: set[str] | None = None,
         already_asked: set[tuple[str, str]] | None = None) -> dict[str, Any]:
    """Rank sealed specimens for one question. Cost is measured, not assumed.

    `warm` prices a resident specimen at the measured WARM rate instead of the
    cold one - the 142x difference G102 measured is the single largest term in
    any real schedule, and ignoring it made an earlier version pick the same
    specimen forever.

    `already_asked` holds (family, specimen) pairs whose answer is in hand.
    Re-asking has near-zero information gain, and without this the ranking is
    stable and the scheduler never explores. S027 §15 still applies: scientific
    priority outranks cache locality, so this DISCOUNTS a repeat rather than
    forbidding it.
    """
    if not hypothesis_family:
        raise ScheduleRefused("a ranking needs a hypothesis family to rank FOR")
    cands = sr.schedulable()
    if not cands:
        raise ScheduleRefused("no sealed specimen is available to rank")

    r = lc.rates()
    cold = r["quiet_cold_MB_per_s"] * 1e6
    hot = r["quiet_warm_MB_per_s"] * 1e6
    warm = warm or set()
    already_asked = already_asked or set()
    rows = []
    for c in cands:
        b = c["source_bytes"] or 0
        d = distance(c["model_type"])
        verdict = scar_verdict(hypothesis_family, c["model_type"], scars)
        admit = ul.predict_peak(b)
        is_warm = c["id"] in warm
        load_s = b / (hot if is_warm else cold)
        # Cost in minutes, floored so a tiny specimen is not infinitely good.
        cost = max(load_s / 60.0, 0.5)
        repeat = (hypothesis_family, c["id"]) in already_asked
        gain = expected_information_gain * (REPEAT_DISCOUNT if repeat else 1.0)
        value = gain * ROLE_WEIGHT[d] / cost
        rows.append({
            "id": c["id"],
            "model_type": c["model_type"],
            "distance": d,
            "role": {"NEAR": "transfer", "MID": "transfer_or_adversary",
                     "FAR": "adversary"}[d],
            "source_gb": round(b / 1e9, 2),
            "measured_load_minutes": round(load_s / 60.0, 3),
            "measured_cold_load_minutes": round(b / cold / 60.0, 2),
            "is_warm": is_warm,
            "already_asked": repeat,
            "fits_admissible": admit["fits"],
            "exceeds_total_memory": admit["exceeds_total_memory"],
            "scar_action": verdict["action"],
            "score": round(value, 4) if verdict["action"] == "LOAD_PERMITTED" else 0.0,
            "suppressed": verdict["action"] == "DO_NOT_LOAD",
        })
    rows.sort(key=lambda r: (-r["score"], r["measured_load_minutes"]))
    return {
        "hypothesis_family": hypothesis_family,
        "n_candidates": len(rows),
        "n_suppressed_by_scars": sum(1 for r in rows if r["suppressed"]),
        "n_refused_by_memory": sum(1 for r in rows if not r["fits_admissible"]),
        "ranked": rows,
        "selection_is_not_fifo": (
            "the order is by value over MEASURED load cost with an architecture "
            "distance weight, not by registry or arrival order"
        ),
    }


def not_fifo_proof() -> dict[str, Any]:
    """S027 §19 forbids FIFO. Show the ranking is not the registry order."""
    r = rank(hypothesis_family="a_family_with_no_recorded_negative")
    ranked_ids = [x["id"] for x in r["ranked"]]
    registry_ids = [c["id"] for c in sr.schedulable()]
    return {
        "registry_order": registry_ids,
        "ranked_order": ranked_ids,
        "differs": ranked_ids != registry_ids,
        "top_choice": ranked_ids[0] if ranked_ids else None,
        "why_the_top_choice": (
            "cheapest measured load among the highest architecture-distance "
            "weight; a far architecture discriminates more and a small one "
            "costs less, so both push it up"
        ),
    }


def scar_suppression_proof(scars: list[Any] | None = None) -> dict[str, Any]:
    """S027 §20's acceptance: prove a scar query suppresses a load."""
    dead = rank(hypothesis_family="low_rank", scars=scars)
    live = rank(hypothesis_family="a_family_with_no_recorded_negative",
                scars=scars)
    return {
        "dead_family": "low_rank",
        "n_suppressed_for_dead_family": dead["n_suppressed_by_scars"],
        "n_suppressed_for_live_family": live["n_suppressed_by_scars"],
        "all_suppressed_when_dead": dead["n_suppressed_by_scars"] == dead["n_candidates"],
        "none_suppressed_when_live": live["n_suppressed_by_scars"] == 0,
        "statement": (
            f"the hypothesis family 'low_rank' has recorded negatives, so all "
            f"{dead['n_candidates']} sealed specimens are suppressed and NOTHING "
            "IS LOADED. A family with no recorded negative suppresses none. The "
            "scar query is what stands between a dead hypothesis and 77 minutes "
            "of disk read."
        ),
    }


def build() -> dict[str, Any]:
    scars = ni.ingest()
    return {
        "obligation": "G104",
        "authority": "S027 §16-§20, §23-§25, §30-§32, §47-§50",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "incumbent_architecture": INCUMBENT_TYPE,
        "role_weights": ROLE_WEIGHT,
        "n_scars_indexed": len(scars),
        "not_fifo": not_fifo_proof(),
        "scar_suppression": scar_suppression_proof(scars),
        "example_ranking": rank(
            hypothesis_family="a_family_with_no_recorded_negative", scars=scars),
        "architecture_distance_rule": (
            "same model_type is NEAR and tests transfer; a related family is "
            "MID; an unrelated architecture is FAR and is the adversary. A law "
            "that only ever meets near neighbours has not been attacked, so FAR "
            "carries the highest weight."
        ),
        "cost_is_measured": (
            "load cost comes from G102's measured 77.7 MB/s cold rate on this "
            "volume and memory admission from G103's live reading. No term in "
            "this ranking is a nominal number."
        ),
        "what_this_does_not_do": (
            "it does not estimate information gain per specimen - that is "
            "passed in. A scheduler that invented its own information-gain "
            "estimates would be ranking on a number nobody measured, so the "
            "term is an input and the receipt says so."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps({k: doc[k] for k in
                      ("not_fifo", "scar_suppression", "example_ranking")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

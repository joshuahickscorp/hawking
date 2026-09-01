"""G135: the eight defect fixes took the sovereign loop from 16% to 78%.

The adversarial lane found nine live defects in the loop and eight were fixed in
one landing. The question this answers is whether any of that mattered to the
resident's actual behaviour, measured on the loop's own log rather than asserted
from the diff.

    run 4, before the fixes    70 iterations   11 parsed and executed    16%
    run 5, after them          27 iterations   21 parsed and executed    78%

THE DOMINANT FIX IS NAMED AND IT IS NOT A GUESS. IDENTICAL_REPLY_LOOP: after one
failed-parse turn, LAST TURN froze to the constant "no work was accepted from
that turn" and tried_params stopped changing, so the context pack was
BYTE-IDENTICAL and greedy decoding returned the same reply forever. Run 4's tail
is that failure in the record - 55 consecutive unparsed turns over 8 distinct
result summaries. The pack now carries a turn number and the resident's own live
hypotheses, so two consecutive packs cannot be identical by construction.

WHAT THIS IS NOT. It is not a claim that the resident got smarter. Nothing about
the body changed - same weights, same decoding. What changed is that the HARNESS
stopped handing it the same prompt and stopped dropping what it produced. The
16% was a measurement of the harness, not of the model, and so is the 78%.

SECOND EFFECT, smaller and worth naming. 16 of 26 recorded hypothesis ids were
literally "x" - the schema's own placeholder, copied verbatim - and 2 more were
"y". An id that is not distinct makes the hypothesis register unjoinable across
turns. The placeholder is now NAME_THIS_CLAIM.

    python3 tools/future/sovereign_trial.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/sovereign_trial.py"
RECEIPT_NAME = "SOVEREIGN_TRIAL.json"

LOG_REL = "receipts/future/_G135_SOVEREIGN_TRIAL_log.jsonl"
ATTACKS_REL = "receipts/future/SOVEREIGN_ATTACKS.json"
# Run 5 begins here. Runs append to one log and carry no run marker, so the
# boundary is the count of entries preserved before run 5 started.
RUN5_START = 70
MIN_SAMPLE = 15


class TrialRefused(RuntimeError):
    """The log is too small, or the boundary would be guessed."""


def _iterations() -> list[dict[str, Any]]:
    p = REPO / LOG_REL
    if not p.is_file():
        raise TrialRefused(f"{LOG_REL} is not on disk")
    return [r for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())
            if "n" in r]


def _rate(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if len(rows) < MIN_SAMPLE:
        raise TrialRefused(
            f"{label} has {len(rows)} iterations, under {MIN_SAMPLE}. A rate off "
            "a handful of turns is noise wearing a percent sign."
        )
    parsed = sum(1 for r in rows if r.get("parsed"))
    executed = sum(1 for r in rows
                   if any(x.get("ran") for x in (r.get("results") or [])))
    return {
        "label": label,
        "n_iterations": len(rows),
        "n_parsed": parsed,
        "n_executed": executed,
        "parse_rate": round(parsed / len(rows), 4),
        "execute_rate": round(executed / len(rows), 4),
        "distinct_result_summaries": len(
            {str(r.get("results_summary")) for r in rows}),
        "distinct_belief_updates": len({str(r.get("belief_update")) for r in rows}),
    }


def arms() -> dict[str, Any]:
    its = _iterations()
    before, after = its[:RUN5_START], its[RUN5_START:]
    if not after:
        raise TrialRefused(
            "there are no post-fix iterations in this log. The boundary is a "
            "COUNT of preserved pre-fix entries, so an empty second arm means "
            "the wrong log was read - not that the fixes did nothing."
        )
    return {"before": _rate(before, "run4_pre_fix"),
            "after": _rate(after, "run5_post_fix")}


def improvement() -> dict[str, Any]:
    a = arms()
    b, f = a["before"], a["after"]
    return {
        "parse_rate_before": b["parse_rate"],
        "parse_rate_after": f["parse_rate"],
        "parse_rate_ratio": round(f["parse_rate"] / b["parse_rate"], 3)
        if b["parse_rate"] else None,
        "executed_before": b["n_executed"],
        "executed_after": f["n_executed"],
        "executed_per_iteration_before": round(
            b["n_executed"] / b["n_iterations"], 4),
        "executed_per_iteration_after": round(
            f["n_executed"] / f["n_iterations"], 4),
        "reading": (
            f"{b['n_executed']} experiments in {b['n_iterations']} turns became "
            f"{f['n_executed']} in {f['n_iterations']}. The loop was not slow; "
            "it was repeating itself."
        ),
    }


def what_changed() -> dict[str, Any]:
    held = json.loads((REPO / ATTACKS_REL).read_text())["held_ids"]
    return {
        "defects_fixed_in_the_landing": [i for i in held if i not in (
            "FAKE_SOVEREIGN", "SILENT_DROP_UNSUPPORTED", "CONTEXT_ACCUMULATION")],
        "dominant": "IDENTICAL_REPLY_LOOP",
        "mechanism": (
            "after one failed-parse turn LAST TURN froze to the constant 'no "
            "work was accepted from that turn' and tried_params stopped "
            "changing, so the pack was BYTE-IDENTICAL and greedy decoding "
            "returned the same reply. The pack now carries a turn number and "
            "the resident's own live hypotheses."
        ),
        "the_signature_is_in_run4": (
            "its tail is 55 consecutive unparsed turns across only 8 distinct "
            "result summaries"
        ),
    }


def what_this_does_not_claim() -> dict[str, Any]:
    return {
        "not_a_smarter_model": (
            "nothing about the body changed - same weights, same decoding. The "
            "HARNESS stopped handing it the same prompt and stopped dropping "
            "what it produced. The 16% measured the harness and so does the 78%."
        ),
        "not_a_controlled_experiment": (
            "the two runs are sequential, not randomised, and the mission kernel "
            "carried more evidence by run 5. The confound is named rather than "
            "argued away: a richer kernel could raise the rate on its own."
        ),
        "not_a_capability_result": (
            "parse rate is whether a reply had the right SHAPE and its work ran. "
            "It says nothing about whether the science was any good."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "obligation": "G135",
        "question": "did the eight sovereign-loop fixes change the resident's behaviour?",
        "verdict": "PARSE_AND_EXECUTE_RATE_16_TO_78_PERCENT",
        "arms": arms(),
        "improvement": improvement(),
        "what_changed": what_changed(),
        "what_this_does_not_claim": what_this_does_not_claim(),
        "evidence_class": "DERIVED_FROM_THE_LOOPS_OWN_LOG",
        "inputs": [LOG_REL, ATTACKS_REL],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(REPO / "receipts" / "future" / RECEIPT_NAME,
                            doc, RECORDED_BY))
        return 0
    print(json.dumps({k: doc[k] for k in
                      ("verdict", "arms", "improvement")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

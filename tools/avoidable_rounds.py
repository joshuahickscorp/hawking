#!/usr/bin/env python3
"""What each harness defect cost the resident, in rounds and in wall clock.

Counting bug fixes rewards finding many small ones. What matters here is
physical: a resident round on this machine is 60 to 190 seconds, almost all of
it GPU-bound prefill, so a deterministic correction that removes one avoidable
round is worth more than most kernel tuning.

Every row is a defect that was actually observed in a receipt or an event log,
not one imagined in advance. `rounds` is what the defect cost per occurrence.
`scope` says whether the fix removes the whole class or only the case seen:
CLASS fixes keep paying, CASE fixes do not.

Wall figures are marked ESTIMATE where they are derived from the measured
per-round range rather than timed directly. They are not presented as measured.
"""
from __future__ import annotations

import sys

ROUND_S_LOW, ROUND_S_HIGH = 60, 190

# (id, defect, rounds_per_occurrence, occurrences_seen, scope, evidence)
LEDGER = [
    ("D1", "root goal deleted from the prompt ([ROOT_GOAL_OMITTED])",
     None, None, "CLASS",
     "every goal unwinnable by construction; no round could have succeeded"),
    ("D2", "deep code unreachable: fs.read had no line window",
     None, None, "CLASS",
     "model located _record_model_call at line 3514 and could not read it: "
     "'Need to see the actual _record_model_call function'. Unbounded: the "
     "goal could not be completed at any number of rounds"),
    ("D3", "syntax preflight compiled replace-fragments standalone",
     None, 3, "CLASS",
     "three goals died being told to fix code that was not broken; "
     "'unexpected indent at line 1' on a correct patch"),
    ("D4", "anchor mismatch was terminal, not a retry",
     5, 1, "CLASS",
     "receipt 8a058995: 5 calls, 1135s, killed holding a patch correct on "
     "every line except one literal backslash-n"),
    ("D5", "prefix checkpoint could never advance after a restore",
     0, 5, "CLASS",
     "reuse pinned at 1398 while prompts grew to 4784; calls 4 and 5 stepped "
     "2266 and 3386 tokens that were already computed. Costs no rounds and "
     "much wall: the waste is inside each round, forever"),
    ("D6", "observation cut slid every turn, invalidating the KV prefix",
     0, None, "CLASS",
     "sliding window re-cuts on 21 of 23 turns, monotone floor on 15; each "
     "re-cut re-prefills everything past the cut"),
    ("D7", "fs.read on a missing path raised a bare traceback",
     1, 3, "CLASS",
     "three identical FileNotFoundError on hcli/tests/test_tool_registry.py "
     "in one goal; the model had no way to learn it should create the file"),
    ("D8", "nothing supervised hawkingd",
     None, 1, "CLASS",
     "the daemon that owns the model worker had no supervisor; the driver "
     "would keep cycling against a control plane that was gone"),
    ("D10", "retry grew the prompt without shrinking the completion budget",
     3, 1, "CLASS",
     "max_tokens 3243 held across prompts of 4776, 4876, 5090; the third asked "
     "for 8333 against an 8192 window, the runtime truncated the reply mid-"
     "object and it was rejected as malformed. The model's patch was correct"),
    ("D11", "a tool_use reply was judged on its placeholder operations",
     3, 1, "CLASS",
     "the model asked to READ the file to obtain an exact anchor and filled "
     "old_text with 'x' to satisfy the reply shape; the anchor preflight "
     "refused it 3x with 'matches 497 places', so the request for the bytes "
     "was the thing being refused. Self-inflicted, same day as the preflight"),
    ("D12", "a truncated tool result did not say how to reach past the cut",
     None, None, "CLASS",
     "tool_registry.py is 2341 lines; the clamp shows 169; the target is at "
     "582. Unbounded: no number of reads of the head ever reaches line 582"),
    ("D9", "one tool observation could occupy the entire input window",
     None, None, "CLASS",
     "24,000 chars against a 5,632-token usable input: one fs.read was 1.4x "
     "the whole context, so shedding everything else still did not fit"),
]


def main() -> int:
    rounds_seen = 0
    unbounded = 0
    print(f"{'id':<4} {'scope':<6} {'rounds':>7} {'seen':>5}  defect")
    for did, defect, rounds, seen, scope, _ in LEDGER:
        if rounds is None:
            unbounded += 1
            shown = "unbnd"
        else:
            shown = str(rounds)
            rounds_seen += rounds * (seen or 1)
        print(f"{did:<4} {scope:<6} {shown:>7} {str(seen or '-'):>5}  {defect}")

    print()
    print(f"defects recorded            {len(LEDGER)}")
    print(f"CLASS fixes                 {sum(1 for r in LEDGER if r[4] == 'CLASS')}"
          f" of {len(LEDGER)}")
    print(f"avoidable rounds counted    {rounds_seen}")
    print(f"  approx wall               {rounds_seen * ROUND_S_LOW / 60:.0f}"
          f" to {rounds_seen * ROUND_S_HIGH / 60:.0f} min   [ESTIMATE: "
          f"{ROUND_S_LOW}-{ROUND_S_HIGH}s per measured round]")
    print(f"defects with UNBOUNDED cost {unbounded}")
    print("  these did not waste rounds, they made the goal unreachable at any")
    print("  number of rounds, which is a different and worse failure")
    print()
    print("Not counted above: D5 and D6 cost no rounds at all. They waste time")
    print("INSIDE every round, permanently, by re-prefilling tokens the resident")
    print("had already computed. On the measured receipt that was roughly 5,600")
    print("re-stepped tokens across two calls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

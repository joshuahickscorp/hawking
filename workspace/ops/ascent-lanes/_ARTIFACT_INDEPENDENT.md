
---

# WHY THIS LANE SURVIVES THE ARTIFACT CHANGE — read this first

Q80's uniform-Q4 vehicle (4.259241 BPW) has been **abandoned as a target**. Q80 is
being re-gravitied to a <=1.5 complete-physical-BPW artifact in parallel lanes.

**Your lane is artifact-INDEPENDENT.** The cost you are attacking is a property of
the token graph, the dispatch topology, the state handling or the host/device
split — not of how expert weights are encoded. It will still be there, essentially
unchanged, on the <=1.5 artifact.

So: use the Q4 catalog purely as a **test harness and correctness reference**. It
is a convenient runnable vehicle, nothing more.

**The design rule that follows from this:** do not hard-code anything to the
uniform-Q4 representation. Where you touch a path that knows about weight
encoding, keep the mechanism generic so the <=1.5 artifact inherits it for free.
If you cannot avoid a Q4-specific assumption, isolate it behind a seam and say so
in your report. A win that has to be rebuilt for the real artifact is a half win.

Q80 measured baseline on the Q4 harness (2026-08-16, DIRTY, 12 new tokens):
    steady_state 2.479023 tok/s = 403 ms/token; prefill 11.16 s
    stage_secs over the 15.6 s run:
        moe_table_build 9.0777 (58%)   deltanet 3.3269 (21%)
        moe_combine 1.6958 (11%)       gqa 1.1335 (7%)
    fallback_count=1637 (vec=265 act=1344 sample=28)
    table_builds=1344  table_dispatches=6720
    generated ids: [8420, 594, 264, 4285, 729, 304, 13027, 429, 17431, 288, 264, 914]

Target is a 20 ms complete token (50 TPS). Everything above is ~20x too slow.

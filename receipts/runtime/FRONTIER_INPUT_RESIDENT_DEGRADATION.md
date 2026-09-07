# Measured frontier input — the resident degrades ~4.5x across requests — 2026-09-05

Measured in this worktree tonight. Fresh-process numbers are repeatable to 0.04%.

## THE PHENOMENON

Identical prompt, identical length, identical depth. Only the number of requests the
resident process had already served differs.

    16K @90%   FRESH process, 1st request        444.3 s
    16K @90%   SHARED process, 4th request      >870 s and still running when stopped

    16K @25%   FRESH process, 1st request        446.0 s
    16K @50%   SHARED process, 3rd request     1,998.0 s      <- 4.5x

Throughout the slow calls the GPU held 99% utilization, allocation was flat near 20.5 GB,
resident RSS was flat at 15.2 GB and host CPU was 0.2%. It was doing real work, slowly.

## THE INSTRUMENT IS OTHERWISE EXACT

Fresh-process repeatability at constant length:

    32K   1,437.0 s   vs  1,437.6 s      0.04%
     8K     159.9 s   vs    159.8 s      0.06%
    16K     446.0 s   vs    444.3 s      0.38%

So the 4.5x is not jitter. Process history is the largest single variable measured tonight.

## WHY IT MATTERS MORE THAN IT LOOKS

HCLI runs long missions against ONE long-lived resident. Mission c4afe4c6 spent 21,527 s
across 41 WorkUnits. Every WorkUnit late in that mission was measured on a degraded process,
and the campaign's 27.9 fresh-prefill tok/s figure is a degraded-process number: a
fresh process does 48.4 tok/s at MORE THAN TWICE the prompt length.

Any A/B that ran arm A before arm B in one resident is biased toward B looking slower.

## THE FIXED COST OF A RESTART APPEARS TO BE SMALL

    setup_s (connector construction)          0.30 s
    fitted fixed term across 8K/16K/32K       not distinguishable from zero; a pure power
                                              law through the measured points needs no
                                              positive intercept

If a restart really is near-free, restarting the resident recovers up to 4.5x.

## BUT THERE IS A REAL TRADE-OFF, AND IT IS NOT OBVIOUS

A restart destroys the prefix checkpoint. Prefix reuse is separately proven and large:
one call reused 3,222 of 3,295 prompt tokens and cut its prefill from ~116 s to 3.4 s.

    KEEP THE PROCESS   prefix reuse survives          degradation accumulates
    RESTART IT         degradation resets             every call is cold

Which wins depends on how much prefix a mission's WorkUnits actually share. That is
measurable, not arguable.

## THE QUESTION FOR THIS MISSION

1. WHAT degrades? Candidates, in order of cheapness to test:
   M1 restore-then-sequential: a restore sets reuse != 0 which forces the 916-dispatch
      route (see FRONTIER_INPUT_PREFILL_CHECKPOINT.md). Note this does NOT by itself
      explain 4.5x, because the cold first call also takes a snapshot and is therefore
      also sequential.
   M2 the KV or workspace is sized or scanned to a HIGH-WATER position rather than the
      current one, so a later request pays for the longest earlier request.
   M3 an allocation or fragmentation effect inside the live Metal context.

   DISCRIMINATOR, one process, two calls, no shared prefix, long prompt then short:
   M2 predicts the short second call is slow. M1 predicts it is fast. M3 predicts slow and
   worsening with every further call.

2. If it cannot be fixed in the resident, what is the right HCLI POLICY? Measure complete
   WorkUnit wall under (a) one persistent resident with prefix reuse, and (b) a restart per
   WorkUnit with no reuse, on the SAME mission. Report both, do not assume.

## CONSTRAINTS

- Report fresh prefill and reused/effective prefill SEPARATELY. An effective rate that hides
  fresh work is not a prefill number.
- Every file in receipts/sovereign/VERIFIER_MANIFEST.json is PROTECTED.
- Smallest executable change. Name a test that fails before it and passes after.

# ROADMAP COMPRESSION — measured

## Census, then and now

    status                baseline    now
    BUILT                      18     23
    WIRED                       0     12
    SCAFFOLDED                 29     11
    BLOCKED_HARDWARE           13     13
    BLOCKED_EXTERNAL            0     10
    ABSENT                     11      1
    UNREACHABLE                 0      1

## Where the old active future went

    MOVE_TO_COMPLETED        16
    KEEP_ACTIVE              19
    EXPERIMENT_CONTINGENT    12
    DEFERRED_PROGRAM         7
    EXTERNAL_ENVIRONMENT     3
    HARDWARE_CONTINGENT      14
    UNKNOWN_RESEARCH         0

    SUBSUMED / DERIVED_AUTOMATICALLY / RECURRING_OPERATION / REMOVE_OBSOLETE  0

Those four are reported as ZERO deliberately. No item was retired into them
during this campaign, and inventing a subsumption to make the compression
number look better would be the exact dishonesty the directive forbids. The
compression that DID happen is real and of one kind: work that was miscounted
as remaining software turned out to be blocked on absent external packages, or
already wired but never declared.

## Net future burden

    old active future (baseline, non-BUILT)   53
    active now (software connections)         19
    plus experiment/long-run contingent       12
    NET FUTURE BURDEN                         41

Progress is capability gained AND future bespoke work eliminated. The honest
reading: the gate count did not shrink -- 71 gates before and after -- but the
share of it that is 'I have not connected these two components yet' fell, and
the share correctly attributed to absent hardware, absent external packages and
runs that must happen rose. That is a denominator being told the truth, not a
roadmap getting smaller by fiat.


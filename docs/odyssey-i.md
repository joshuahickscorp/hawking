# Odyssey I

Odyssey I is the first discovery campaign: find cheaper physical
representations of real models, and find out what transfers between them.

Readiness is measured, not asserted — the current capability reachability audit reports
each required capability as READY only when a module exists **and** has a live
caller. Existence alone is not readiness; this codebase has carried
capabilities that were built, declared, and structurally unreachable.

## The funnel

Not fifty exhaustive campaigns. A funnel that spends compute where the
uncertainty is.

```
SPECIMEN CENSUS            what exists, what it costs to open
        |
ARCHITECTURE CLUSTERING    group by organ topology, not by vendor
        |
CHEAP STRUCTURAL PROBES    seconds per specimen; kill the hopeless early
        |
REPRESENTATIVE FAMILY      one member per cluster carries the experiment
        |
SURPRISE / PROMISE FILTER  only anomalies and winners continue
        |
DEEP GRAVITY SEARCH        expensive, and only here
        |
PHYSICAL BENCHMARK         measured tok/s and bytes on this machine
        |
PARETO RESIDENT CANDIDATES
```

Two rules give the funnel its leverage:

**Early stop.** A specimen that fails a cheap probe does not get a deep search.
The probe must be cheap enough that being wrong about it is affordable.

**Law transfer.** A Law derived on one cluster member is applied as a prior to
its siblings, and the prior is *tested*, not assumed. A transferred Law that
fails on a sibling is the most informative result the funnel can produce: it
bounds the Law.

## What Odyssey I is trying to learn

- what transfers between architectures, and what does not
- which organs compress and which resist
- which representations survive real capability tests
- which kernels actually matter to wall time
- which models are unusually good residents

The negative answers count. A representation that fails everywhere is a Law.

## Resident ascension bounty

Odyssey I should answer whether anything dominates the current resident
(`sealed-3.14`). A better resident multiplies every later experiment, so this
is a campaign goal rather than a side effect.

Scoring is nine-axis, and no axis is sufficient alone:

| axis | why it matters |
| --- | --- |
| coding / agent capability | the resident writes HCLI's own changes |
| structured-output reliability | a reply that will not parse is not work |
| tool competence | rounds per goal is wall time |
| resident bytes | what fits in the Metal working set |
| prefill throughput | currently the binding constraint |
| decode throughput | bandwidth-bound; sets the floor |
| reusable state | prefix reuse is prefill avoided |
| runtime compatibility | must run on the existing execution path |
| Gravity potential | headroom under representation search |

**Benchmark intelligence is not qualification.** A candidate replaces the
incumbent only after passing HCLI qualification on real goals — structured
output, tool use, and an accepted mutation — because the job is being HCLI's
worker, not scoring on a leaderboard.

## Current state

Required capabilities: 13 READY in the historical Odyssey-I readiness snapshot; current
reachability is owned by `tools/roadmap/capability_reachability.py`.

Known constraints going in, measured:

- Prompt throughput is ~25 tok/s. Prefill steps one token at a time, so it
  costs what decode costs: 580 kernel dispatches per token, 1.15M for a
  2,000-token prompt.
- No Odyssey wall time has ever been recorded. `COMPILE_ECONOMICS.jsonl` holds
  9,573 events and every one has `wall_s` exactly 0.0, so any campaign-duration
  figure in the ledgers is a budget, not a measurement. Instrument before
  claiming a wall.

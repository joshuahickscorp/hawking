# Odyssey promotion gate

**Status: CANONICAL.** The conditions under which `ODYSSEY_LAUNCH_AUTHORIZED` may be flipped
to true. Written before the decision is live, deliberately, so it is not made under pressure
from whatever happens to be green that day.

Flipping this fence starts T0-T7: real training against the Math-Preserve substrate. It is
the most expensive and least reversible action in the Hawking half of the Continuum. A
checkpoint trained on a substrate whose runtime was never honestly measured cannot be
un-trained, and the campaign's whole value is that it refuses to believe itself.

**Authority: the primary controller only.** Not a passing test suite, not a delegated lane,
not a green dashboard. Every condition below must be independently verified — a report is a
claim.

## G1 — Substrate sealed (MET, 2026-07-26)

- 282/282 shards, 59,585/59,585 frozen decisions, 0 mismatches
- complete BPW 0.9774 under the H0.98 target, 244,667,548 bytes headroom
- one-bit law legal as an exact rational
- allocation manifest byte-identical between repository and artifact
- `HAWKING_ODYSSEY_READY` audit verdict sealed, six gates green

## G2 — Runtime truth, stated honestly

Not "fast enough". **Honest.**

- `BASE_TRUE_TPS` measured, with **cold and warm reported as separate windows** on a
  per-token curve, never averaged into one number
- residency and eviction reported *alongside* the token rate — on a routed MoE a throughput
  figure without a residency figure is uninterpretable
- prefill curve per context
- hashing/verification overhead isolated, and its scope stated (cold-phase or sustained)
- the dominant decode cost **identified**, not assumed. If decode is dispatch- or
  compute-bound rather than residency-bound, that must be established by measurement,
  because it changes what acceleration is worth.

Failing this gate does not mean slow. It means *unknown*, and training on an unknown
substrate is the failure this gate exists to prevent.

## G3 — Acceleration: correctness before speed

- exact-token bit-identity against same-target greedy **re-receipted on the artifact that
  will actually be served** — not inherited from a prior commit's claim
- any promoted provider measured on **accepted tokens/second with full draft, verify and
  rollback cost**, at a 1.10 lower confidence bound per workload class
- **acceptance rate promotes nothing.** 87 percent acceptance measured 0.91x, a slowdown.
- sealed negatives still binding unless materially new evidence is produced

Acceleration may be *absent* at promotion. It may not be *wrong*. A provider that is not
bit-identical is a different model, and Odyssey would train against the wrong one.

## G4 — HIDE contracts real, not merely merged

- the live model turn loop produces **real generated tokens** through a live entry point.
  A stub provider returning canned text fails this gate outright — the entire archaeology
  exists because things looked wired and were not.
- Context OS declares capability honestly: native distinguished from effective, and no
  retrieval or compaction figure reported as native window
- tools, effects and permission adjudication reachable from the live path
- `HIDE_KERNEL_TURN` **either** promoted through its own gate **or** explicitly declared
  off-by-design at promotion, with the consequence stated. Merging the code does not pass
  its gate.

## G5 — Evaluation exists before training, not after

- the evaluation harness that will judge T7's winner is **built and exercised** before T0
  starts. Building the judge after seeing the candidates is how a tournament gets rigged by
  accident.
- support-halo measurements (technical language, general reasoning, coding, retrieval,
  tools, long context, self-correction) have a **pre-Odyssey baseline** on the substrate, or
  regression is unmeasurable
- the checkpoint tournament's scoring rule is written down before any checkpoint exists

## G6 — Launch receipts and reversibility

- rollback path proven: the substrate and its receipts survive a failed or abandoned Odyssey
- Lean/Mathlib and environment locks pinned and recorded
- the runner still **refuses T0 while the fence is false** — re-proven at promotion time,
  not assumed from the last audit

## The decision

All six green, each independently verified, and then a written statement of what is being
risked and why it is worth it. That statement is part of the gate, not a formality: if it
cannot be written honestly, the gate has not passed.

**Until then `ODYSSEY_LAUNCH_AUTHORIZED` stays false, and no lane may flip it.**

## Current standing

```
G1 substrate sealed      MET
G2 runtime truth         IN PROGRESS -- cold measured; warm run live; dominant decode cost being identified
G3 acceleration          IN PROGRESS -- re-receipt lane running; ledger stale-OPEN pending its result
G4 HIDE contracts        IN PROGRESS -- merged and green, but nothing has produced a real token yet
G5 evaluation            NOT STARTED -- the gap most likely to be skipped, named here so it cannot be
G6 launch receipts       PARTIAL -- fence false and runner refusal proven at seal; rollback unproven
```

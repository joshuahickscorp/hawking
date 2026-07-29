# TG K11 — active-byte collapse for TG2/TG1 and below

## Purpose

K11 is the required physics lane for TG2/TG1 and below. Kernel/graph cleanup
alone cannot cross the frozen routed-expert geometry:

- routed weight floor: `2,580,304,896` bytes/token;
- TG2 bandwidth at that floor: about `1.29 TB/s`;
- TG1 bandwidth at that floor: about `2.58 TB/s`.

Those rates exceed the target device's roughly 800 GB/s-class peak before
attention, head, dispatch, cache, and synchronization costs. Therefore no
unchanged-geometry receipt may claim TG2/TG1.

This contract is planning/authority only until the capable-provider,
Math-Preserve, Numeric Parity V2.1, profiler, and full artifact-binding gates
are frozen. It does not authorize model-body reads, experiments, MOP, or fence
transitions.

## Entry prerequisites

All must be independently accepted:

- bound capable GLM artifact/provider;
- exact executable, source, artifact, device, and physical-profiler identity;
- Numeric Parity V2.1 including real near-tie authority;
- accepted nonregressing K3 device MLP and K4 descriptor executor;
- Math-Preserve contract with frozen capability/task/quality gates;
- explicit source-access and disk admission;
- `BASE_TRUE_TPS` protocol and TG thresholds.

Until then, K11 may create contracts, source-body-free simulators, and byte
accounting only.

## Required budgets

For each candidate, derive a conservative active-byte budget from measured
sustained device bandwidth and leave explicit head/attention/dispatch margin.
At the absolute 800 GB/s ceiling:

- TG2 total active bytes must be below `1.6 GB/token`;
- TG1 total active bytes must be below `0.8 GB/token`.

Qualification uses measured sustained bandwidth under the bound thermal/power
window, not the peak label. The actual admitted budgets will therefore be
lower.

Report routed, shared, dense, attention, indexer, router, head, KV/cache,
transfer, and other bytes separately. Geometry estimates are diagnostics;
promotion requires physical events from the live weight-read APIs.

## Candidate families

Each family is independently versioned, default-off, and killable:

1. **Representation collapse** — reduce physical expert/head/attention bytes
   through a frozen decoder/format while preserving the required mathematical
   outputs and decisions.
2. **Active-expert collapse** — execute fewer routed experts only when an
   accepted mathematical equivalence or conservative bound proves omitted
   contributions cannot affect required outputs/decisions.
3. **Depth/work collapse** — skip or reuse layer work only under an accepted
   exact-equivalence/state-reuse proof.
4. **Resident reuse/cache collapse** — eliminate repeated physical reads only
   when real cache residency/reuse is measured and generation-safe.
5. **Head/attention byte collapse** — reduce non-routed bytes independently so
   routed savings are not hidden by another floor.

Heuristic pruning, quality-only approximation, router confidence without a
proof, widened-f32 authority, benchmark memorization, and unmeasured OS/GPU
cache assumptions cannot qualify Math-Preserve.

## Mathematical and decision authority

For every token/layer affected, bind original inputs and compute the complete
authority result. Require:

- exact greedy/top-k token, router IDs/weights/order, DSA ranks, masks, and
  stop decisions;
- complete Numeric Parity V2.1 continuous pass;
- real FP64 near-tie replacement at every load-bearing boundary;
- no partial residual/KV/sequence/trace mutation on refusal;
- path-independent deterministic replay;
- capability/task/quality gates frozen by Math-Preserve.

If a candidate intentionally changes model mathematics or quality, it belongs
to the separately authorized research lane, not K11.

## Physical evidence

Instrument actual:

- weight/cache/descriptor reads;
- D2H/H2D and mapped shared accesses;
- command buffers, waits, dispatches, encoders;
- allocations and address generations;
- GPU start/end/queue timestamps;
- cache hit/miss/eviction generations;
- memory/RSS/swap/thermal/power state.

Receipts reconcile physical bytes with static tensor identities and the
exclusive cost ledger. Caller-declared byte counts, static formulas, unified
memory labels, or command-count reductions are not physical proof.

## Source-body-free prework

Before capable access:

- build exact geometry and budget calculators;
- model TG20/TG10/TG5/TG2/TG1 byte ceilings under measured-bandwidth inputs;
- create deterministic synthetic equivalence/refusal cases;
- implement receipt schemas and public-reseal counterexamples;
- verify generation/poison/reset and cache-accounting logic;
- prepare randomized/interleaved benchmark orchestration.

No synthetic result is `BASE_TRUE_TPS`, a TG milestone, or capability evidence.

## Real evaluation protocol

After explicit authorization, use the same bound artifact/provider/device and:

- sustained contexts 2K, 8K, and 32K;
- at least 80 warm measured tokens per context/mode;
- randomized paired/interleaved baseline and candidate order;
- separate ledger-off timing and ledger-on topology;
- p50/p95 plus full sample vectors;
- cold/warm cache phases and thermal steady state;
- exact output/decision/capability evidence;
- complete active-byte category reconciliation.

TG2 requires the frozen median/p95 rule at `≤2 ms/token`; TG1 at
`≤1 ms/token`. “Below” uses a separately frozen next threshold before
measurement. A diagnostic `--target-ms` never changes the milestone ladder.

## Kill criteria

Reject a candidate if any:

- active bytes remain above the admitted physics budget;
- measured bandwidth demand exceeds sustained device capability;
- p50 or p95 regresses/fails its fixed threshold;
- V2.1, near-tie, discrete, capability, or Math-Preserve gate fails;
- claimed reuse is not tied to a live cache/address generation;
- physical and static byte ledgers do not reconcile;
- default-off isolation fails;
- memory/swap/thermal safety fails;
- evidence depends on public resealing or mutable paths;
- a real-source, MOP, or authorization boundary was crossed without authority.

## Exit and non-claims

Every candidate returns exact code/artifact/provider/device identities,
category bytes, timings, capability decisions, disposition, and negative
results. Until all entry and real-evaluation gates pass, keep:

- `RAMANUJAN_RESEARCH_AUTHORIZED=false`;
- `HIDE_KERNEL_TURN=false`;
- `ODYSSEY_LAUNCH_AUTHORIZED=false`;
- full traversal false;
- capable-artifact claim false;
- TG2/TG1/below claims false;
- MOP untouched.

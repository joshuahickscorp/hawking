# Hawking research map

This document keeps the active research questions compact. Detailed experiment
state belongs in machine receipts; completed proposals and session handoffs
remain available through Git history.

## Evidence contract

Research is admitted only when it preserves the distinction between:

- mathematical possibility;
- implementation correctness;
- component performance;
- whole-artifact density and quality;
- end-to-end physical throughput;
- production promotion.

Exact-output paths require byte or bit parity. Approximate paths require frozen
multi-window quality, worst-window disclosure, capability probes, parent-to-
condensed divergence, and whole-artifact physical bpw. Every speed result needs
a matched baseline on the same machine and resource envelope. A result cannot
be generalized across model tier, rate, branch, architecture, or execution mode
without evidence for that scope.

## Primary lanes

### Capability-first low-bpw condensation

The Doctor treats sub-bit and low-bit work as capability rate-distortion rather
than weight reconstruction alone. Representation, structural reconstruction,
repair, training, routing, and adversarial failure generation are searched as a
typed program. The canonical scientific contract is
[`plans/DOCTOR_V5.md`](plans/DOCTOR_V5.md); the expansion and proof plan is
[`plans/DOCTOR_V5_RESEARCH_PASSES.md`](plans/DOCTOR_V5_RESEARCH_PASSES.md).

### Compute for memory

TQ and STRAND move work from resident representation into deterministic decode,
bitslice, RHT, sparse/outlier, and side-information paths. The important
measurement is not nominal weight bits but total serving bytes, decode cost,
quality, and physical resource behavior. Current gates are in
[`plans/tq_compute_for_memory_appendix_2026_07_14.md`](plans/tq_compute_for_memory_appendix_2026_07_14.md)
and [`STRAND.md`](STRAND.md).

### Speculative decoding and Event Horizon

Speculation remains default-off until target/draft token semantics, cache
positioning, verification, and TQ parity are proven. Acceptance rate alone is
not throughput. Re-entry requires a matched end-to-end win after target
verification overhead, memory pressure, and batching effects:

- [`plans/hawking_event_horizon_status.md`](plans/hawking_event_horizon_status.md)
- [`plans/spec_decode_reentry_appendix_2026_07_14.md`](plans/spec_decode_reentry_appendix_2026_07_14.md)
- [`plans/spec_decode_studio_readiness_2026_07_12.md`](plans/spec_decode_studio_readiness_2026_07_12.md)

### Serving and continuous batching

The architectural direction is per-request KV slots, mixed-position batched
attention, bounded request lifecycle state, cancellation-safe cleanup, and
fair scheduling. Sequence the work:

1. decode-only batching with exact single-request parity;
2. interleaved prefill/decode with explicit memory admission;
3. speculative continuous batching only after both foundations pass.

Measure time to first token, decode throughput, tail latency, memory per active
sequence, cancellation behavior, and output identity. The current API remains
summarized in the [handbook](README.md#http-api).

### Apple-fit execution

Optimize for unified memory, memory bandwidth, shared package power, and
architecture-specific core topology rather than generic thread saturation.
Candidate work includes exact thread profiles, bounded phase overlap, native
CPU/PGO, mmap/preallocated I/O, Metal preprocessing, and host-sprint controls.
Each facet stays default-off until whole-path evidence exists.

### SSM and recurrent models

RWKV-7 and related state-space models are valuable for compact state, long
context, and local post-training. Their quality must be evaluated through the
real chat template and their throughput under the same physical accounting as
transformers. Training corpora, checkpoints, and exports remain ignored
artifacts; only manifests, hashes, commands, and validated receipts are source
controlled.

### Agentic local systems

HIDE explores stateful planner-executor-verifier loops, grammar-constrained tool
calls, persistent context manifests, session replay/fork, local fleets, and
transparent event projections. The headless backend remains authoritative; the
UI is a projection of events and emits typed intents. Product and contract
details are consolidated in the [roadmap](plans/ROADMAP.md#hide-product-contract).

## Experiment lifecycle

1. State the hypothesis and exact scope.
2. Freeze source, binary, model, workload, environment, and baseline.
3. Build the smallest correctness oracle.
4. Run cheap synthetic or component gates.
5. Run a real-artifact matched A/B at an owner-free checkpoint.
6. Record quality, bytes, time, RSS, swap, disk, power/thermal state, and
   failure behavior.
7. Classify the result as promote, retain-default-off, redesign, or kill.
8. Fold durable conclusions into a canonical document and leave bulky evidence
   in receipts.

## Kill and revival discipline

The ledger below is authoritative. Do not respawn a killed idea under a new
name. Revival requires a changed premise, a mechanism addressing the recorded
failure, a predeclared gate, and new evidence. Historical enthusiasm is not a
changed premise.

## Current priorities

1. Finish the active Doctor ladder without source or evidence drift.
2. Convert observed rung outcomes into better rate/tier/branch priors.
3. Qualify exact thread and phase profiles on real artifacts.
4. Establish trusted physical A/B authority before applying speedups to ETA.
5. Cross the signed release boundary before activating new runtime generations.
6. Treat models beyond 120B as new admission mountains with independent
   architecture, storage, lifecycle, and evidence gates.

## Baseline neutrality

Run every baseline on the same named machine, frozen suite, output length,
sampling configuration, and cold/warm policy. Record:

- effective physical bpw and all side information;
- multi-window quality and worst-window behavior;
- TTFT, inter-token latency, throughput, p50/p95 wall, and useful goodput;
- capability per joule, byte, resident byte, active parameter, and second;
- peak unified memory, pressure state, swap delta, disk, and thermals;
- exact model, source, binary, command, environment, and receipt identity.

A best-effort baseline may support a negative or contingent conclusion, never a
public win. If a competitor baseline wins, its receipt remains unchanged.
Compare against parent precision, standard GGUF quants, the strongest locally
available Apple framework, and uncompressed/reference paths as appropriate.
Doctor controls must include codec-only, runtime-only, treatment-only, and
combined paths so recovery gains are not assigned to the representation.

Headline results require at least R3 same-machine-class reproduction. MPS
quality needs CPU confirmation. PPL alone is insufficient; include divergence
and capability probes. No result may hide model downloads, caches, codebooks,
scales, residuals, adapters, or runtime-resident side state from physical bpw.

## Killed and parked levers

These conclusions remain dead until their stated premise changes:

| Lever | Conclusion |
|---|---|
| CPU/GPU sampler-tokenizer overlap | forward path dominates; no useful pipeline win |
| cross-layer weight delta coding | residual entropy and decode overhead erase benefit |
| EAGLE-3 trained head as assumed default | acceptance/quality/occupancy not generally proven |
| Eagle routing-mask predictor | insufficient useful prediction |
| f16 residual stream | memory cost without compensating quality/speed |
| block-256 FFN contextual sparsity | quality or overhead gate failed |
| host per-dispatch micro-optimization | not the decode bottleneck |
| trivial-op dispatch fusion | dead for end-to-end tokens/s |
| indirect command buffers | setup/complexity without measured decode win |
| KV working-set eviction | harms exact long-context behavior; no default |
| low-rank residual codec | side information and reconstruction miss the density gate |
| learned per-model codebook | physical bytes and lookup cost erase nominal gain |
| W4A8 as default decode | Apple batch-1 path does not justify it |
| MLA simdgroup rewrite | no end-to-end win under tested shapes |
| MoE megakernel | redundant work/occupancy loses |
| serial expert dispatch | command and bandwidth overhead lose |
| sumy-trick Q4_K v3 | register pressure loses |
| four-row predecode ILP default | unvalidated or shape-regressive |
| high-B v4r multisequence route | v3w remains superior on tested shapes |
| Q3 sub-Q4 byte cuts | quality floor failed |
| QTIP Metal trellis decode | decode-cost gate failed |
| f16 activation into predecode GEMV | no useful end-to-end advantage |
| Q4_K batched MMA for rows <= columns | shape regime loses |
| Q5_0 shuffle broadcast | micro-change did not improve whole path |
| vectorized unpack/occupancy tuning | no durable end-to-end gain |
| access-order weight repack | preprocessing/storage complexity not repaid |
| Q8-KV layer-differential precision | memory/quality trade did not clear |
| semantic cache | parked; correctness/product semantics unresolved |
| ExactShared speculation as-is | net-negative under present verification cost |
| LM-head simdmat as TPS lever | not a material decode bottleneck |
| usage-frequency vocabulary screen | certificate/quality gate not met |

The silicon audit did ship two durable patterns: shape-gated Q4_K
simdgroup-matrix work where it actually wins, and zero-copy mmap-backed Metal
weight loading. Everything else remains subject to the ledger above.

Before spawning a new lever:

1. identify the measured bottleneck;
2. search this ledger for the same mechanism under another name;
3. state the changed premise;
4. freeze correctness and performance gates;
5. budget implementation and rollback;
6. kill it promptly if the whole-path gate fails.

## Structured failures

Failures are first-class receipt-bearing artifacts. Each failure records model,
recipe, expectation, observation, receipt, reproduce command, category,
severity, roadmap effect, and pivot trigger. Severity is `warn`, `fail`, or
`wedge-threat`.

The first recorded failure was the 7B LoRA/blockwise Doctor recovery on an
18 GB machine. Every v3 configuration exceeded the 6 GB swap ceiling, timed out
at 120 minutes, or leaked a worker semaphore; observed swap reached roughly
25 GB and no candidate artifact was emitted. This is a measured hardware floor,
not a capability wedge threat. It confirms that the full-rank/blockwise 7B
recovery lane belongs on the higher-memory Studio. The receipt remains
`receipts/FAIL-001.json`.

# Speculative decoding on the 96 GB Studio: evidence and readiness

**Date:** 2026-07-12  
**Decision:** **BLOCKED / do not enqueue yet.** The 96 GB M3 Ultra removes the
old dual-residency constraint for small target/draft pairs, but it does not
remove the correctness, proposal-quality, verifier-cost, or runner blockers.
No speculative job should share the GPU with the active quantization/Doctor
ladder.

`tools/condense/spec_revive.py` is now a fail-closed readiness checker, not an
experiment launcher. It does not modify
`reports/condense/spec_oracle_gate.json`, does not enable P3, and cannot execute
the old dormant capture/train/bench sequence.

## What the current evidence actually says

| Question | Evidence | Conclusion |
|---|---|---|
| Does a trained EAGLE head pay? | `docs/dead_levers.md`: held-out tau **0.877**; on-device baseline **36.9 tok/s** versus K=2/4/8 **14.9/11.1/7.6 tok/s**. | NO-GO on the measured Qwen-3B/code regime. Do not retrain without a cheap oracle first showing tau >= 2.5. |
| Do generic free proposers pay? | `reports/free_market_tau.md`: user n-gram **1.0408**, suffix **1.0665**, retrieval **1.0874**. `reports/oracle/spec_accept.json`: best generic code n-gram tau about **1.417**, below the 2.5 gate. | NO-GO as a generic default. |
| Does high acceptance alone imply speedup? | `reports/eh_market_bench_smoke.md`: `reproduce_edit` accepted **87.4%** of drafts but delivered **34.55** versus **37.99 tok/s** (**0.91x**). | No. Draft, verify, synchronization, and rejected work must be charged. |
| Is the live free market robust? | `reports/eh_market_bench_full.md`: chat was **1.04-1.05x**, but repetitive code was **0.76-0.82x** and prose **0.67-0.78x**. | A small prompt-specific win is not a deployable policy. |
| Is there any proposal worth a fresh oracle? | `reports/oracle/spec_accept_warmstart.json` is explicitly an **ESTIMATE**; pooled warm suffix tau is **3.402** versus cold **2.514**, with wide session spread. | User-history warm-start is the only near-term proposal worth a real tokenizer/exact-target oracle. It is not live evidence. |
| Is batched verification one target forward? | `crates/hawking-core/src/speculate/router.rs` records B=1..8 verifier costs of **1.0, 2.20, 2.70, 3.25, 3.62, 3.77, 4.00, 4.15** greedy-forward equivalents. | One command buffer is not one forward's wall cost. Any tau-only gate is incomplete. |

## The correctness gate: implementation exists; TQ parity does not yet exist

The single-token Qwen greedy path consults `tq_ffn` and dispatches the Hawking
bitslice tensors in `crates/hawking-core/src/model/qwen_dense.rs`. The
`forward_tokens_batch_tcb` implementation used by `forward_tokens_verify` now
consults that same ownership map and routes all TQ-owned projections through a
batch-major B=1..8 bitslice path. It carries stored/compact/hashed/computed
runtime modes, RHT-cols, OUTL, and residual passes, and bypasses the incompatible
Q4 fused-down kernel when TQ owns `ffn_down`.

That is implementation evidence, not parity evidence. No device run has occurred
while the active ladder owns the machine, so a condensed/Doctor `.tq` target and
the new batched verifier are still not proven to represent the same distribution.

The existing `crates/hawking-core/tests/event_horizon_parity_prop.rs` is not
that proof. It is hard-coded to a lowercase 3B GGUF path, returns success when
the file is absent, and caps every logical 16/64/256-token case at 16 tokens.
Historical Q4 parity remains useful, but it cannot certify TQ verification.

Mandatory invariant:

> Every accepted or rejected draft token must be verified by the exact same
> TQ artifact, kernels, numerics, tokenizer, and greedy tie-break used by the
> single-token target path.

A `one_pass_verifier: true` Boolean is not evidence for that invariant.

## Other implementation blockers

- `crates/hawking-core/src/speculate/parallel_draft.rs` is a scaffold that emits
  zero-valued placeholder token IDs. It is not a neural draft runtime.
- `crates/hawking-core/src/speculate/tree.rs` documents the Metal tree verifier
  as TODO and supports only the CPU fallback.
- Production Event Horizon currently wires the n-gram and suffix proposers;
  the retrieval implementation is not registered in the production Qwen loop.
- Appendix C upgraded the pure router core to accept a validated measured
  verifier curve, consume per-B verifier/proposal/retokenize/sync timing, and
  advance disabled dwell without inventing failures. The production Qwen loop
  still supplies a placeholder target cost and zero timing observations, so
  post-ladder wiring and receipts remain blockers; the scaffold is not a speed
  claim or an enabled re-entry path.
- The old `spec_revive.py` invoked incompatible command-line interfaces and
  could report `spec lane complete` after late-phase failures. There is no
  validated condensed capture -> head training -> dual-runtime runner today.
  A narrower P0/P1 runner now exists for TQ parity and verifier-cost evidence;
  it does not launch a proposer, train a head, or enable re-entry.

## Encoded admission contract

`tools/condense/spec_revive.py` requires all of the following before it can
even declare the evidence ready:

1. A durable, non-empty `.tq` file, bound by SHA-256. A model directory or a
   temporary audit candidate is inadmissible.
2. A `hawking.spec_tq_batched_parity.v1` receipt bound to that artifact and the
   `Studio-M3Ultra-96` profile:
   - TQ single-token greedy is the reference and TQ batched is the verifier;
   - exact token match;
   - at least 20 prompts to at least 256 generated tokens;
   - zero skipped cases;
   - measured B=1..8 median and upper-confidence verifier costs, at least five
     trials per B;
   - source commit recorded.
3. A `hawking.spec_cost_oracle.v1` receipt bound to both the artifact and parity
   receipt:
   - exact-match acceptance, at least 50 prompts and 8,192 scored tokens;
   - code, prose, and tool-JSON each have at least 10 prompts and 1,024 tokens;
   - useful-token lower bounds and draft/verify/fixed-cost upper bounds;
   - recomputed speedup LCB >= **1.10 separately in every workload class**;
   - dual-residency peak <= **78 GiB**, zero swap, and normal memory pressure;
   - source commit recorded.
4. A real checkpointed full speculative experiment runner. This remains encoded
   as `RUNNER_IMPLEMENTED = False`, so even perfect synthetic receipts cannot
   launch dormant proposal/training code. The dedicated `spec_tq_runner.py`
   covers only prerequisite P0/P1 target parity and verifier-cost measurement.

The checker rejects missing data, skipped tests, optimistic cost arithmetic,
malformed fields, and non-finite values. Its synthetic self-test performs no
model or GPU work:

```sh
python3.12 tools/condense/spec_revive.py --selftest
python3.12 tools/condense/spec_revive.py --plan 7B
```

`--status MODEL.tq LABEL` writes an atomic readiness receipt but hashes the
entire artifact, so run it only outside an active bandwidth-heavy ladder.

## Research sequence after the current ladders

### Immediate post-run execution

1. **Run the implemented TQ-native proof path.** Build
   `hawking-tq-spec-probe`, derive a hash-bound corpus token-prompt set, and let
   `spec_tq_runner.py` acquire the lease. Proof mode requires every q/k/v/o/
   gate/up/down projection to be TQ-owned and GPU-resident; absent assets or
   coverage fail rather than skip.
2. **Execute the non-skipping parity matrix.** Run 20 x at least 256 cases for
   B=1..8 and each runtime interpretation, with single-token TQ greedy as the
   oracle. Zero mismatch and zero skip are mandatory.
3. **Measure the M3 Ultra verifier curve without a proposer.** Normalize each B
   against the same target's single-token greedy latency and record median,
   confidence bound, p95, energy, GPU time, bytes, peak memory, pressure, swap,
   commit, and thermal state. The strict finalizer requires separately bound
   physical counter sources. This is the cheapest honest oracle.
4. **Run a real-token warm-history acceptance trace.** Use the target tokenizer
   and held-out sessions; charge only suffix-recomputed tokens, prefix-cache
   overlap, misses, and lookup time. Stop if any workload cannot clear the
   cost-aware 1.10 LCB gate.

The available Qwen 7B Q4 target plus Qwen 0.5B Q4 control occupy about **5.08
GB of weight files combined**, so 96 GB is ample for a later dual-residency
control. Before using them, prove tokenizer/vocabulary identity. This is only a
Q4 control; it does not clear the TQ correctness gate. No durable `.tq` target
or EAGLE head was present in the audited model/report paths on 2026-07-12.

### Medium-term research

1. Build a real non-autoregressive parallel draft head only after an offline
   oracle clears the cost-aware gate. Train it on the served TQ/Doctor target
   distribution, not an FP16 or parent-GGUF proxy.
2. Sweep draft precision independently of target precision: Q4, 3, 2, 1, and
   sub-bit draft artifacts, with Doctor recovery. Lower draft quality is useful
   only if saved draft bytes outweigh the resulting acceptance loss.
3. Replace static routing with measured wall-clock accounting and shadow
   exploration. A disabled proposal arm needs cheap counterfactual outcomes to
   re-arm; repeated synthetic failures are not learning.
4. Add exact-match live A/B gates for accepted tok/s, p50/p95/p99 latency,
   joules per accepted token, bytes moved per accepted token, and peak unified
   memory. Report rejected work explicitly.

### Long-term paradigm shift

Co-design the target format, draft head, KV state, and verifier as one
persistent Metal execution graph. The useful primitive is not "one batched
matrix multiply" but an **exact token-commit transaction**: propose from local
persistent state, verify multiple conditional continuations against the same
compressed target representation, commit the longest exact prefix, and move
only the state needed by the committed branch. That is the path to capability
per byte and per joule; dual-loading conventional dense models is only a
control experiment.

## Detached queue integration

Do not put speculation in the 14B/72B/120B condensation critical path. Add a
separate opportunistic `P3_SPEC_RESEARCH` lane only after the above runner
exists:

- acquire the same exclusive heavy/GPU lease as Doctor and quantization;
- start only with normal pressure, zero swap, sufficient disk reserve, and no
  active heavy owner;
- fingerprint every cell by target hash, draft hash, parity-receipt hash,
  source commit, prompt-set hash, and knobs;
- checkpoint each oracle/bench cell atomically and resume from the first
  missing cell after unplug/reboot;
- trip immediately on pressure above normal, any swap growth, or the 78 GiB
  process budget; preserve artifacts and leave the cell retryable;
- treat a gated skip as **deferred**, not completed, so future valid evidence
  can schedule it;
- never delete larger-model assets automatically to make room for a spec run.

The current operator P3 gate and active Studio ladder were deliberately left
unchanged.

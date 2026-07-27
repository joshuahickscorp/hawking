# BASE_RUNTIME_MAXIMIZED — hard promotion gate

**Status: CANONICAL.** Supersedes the runtime portion of the Continuum's step ordering.

> A fast speculative layer on a slow base is not runtime completion.
> Maximize the verifier first, then multiply it.

## Why this gate exists, in one number

Active routed-expert bytes per token on the sealed artifact: **2.58 GB**
(8 experts x 3 projections x 1,378,368 bytes x 78 layers).

| target | required bandwidth | share of ~800 GB/s M3 Ultra peak |
|---|---|---|
| 0.1547 tok/s (measured today, hashed warm) | 0.4 GB/s | **0.05%** |
| 0.4074 tok/s (measured, no-hash warm) | 1.1 GB/s | 0.13% |
| 50 tok/s (product pressure target) | 129 GB/s | **16.1%** |
| 60 tok/s | 155 GB/s | 19.4% |
| bandwidth-only ceiling | 800 GB/s | **310 tok/s** |

**The runtime is running at 0.05 percent of its own memory-bandwidth roofline —
roughly 2,000x off.** 50 tok/s asks for 16 percent of peak, which a competent GEMV path
reaches routinely. The target is not ambitious; the current number is anomalous. Whatever is
consuming the difference is overhead, not physics, and it is findable.

That is the whole argument for this gate. Multiplying 0.155 by a speculative factor produces
nothing worth having.

## Blocking rule

Until `BASE_RUNTIME_MAXIMIZED` is sealed, **no acceleration provider may be**:

- trained,
- promoted,
- made default anywhere in HIDE,
- or used to satisfy any performance gate.

Speculative decoding may continue **correctness archaeology and receipt repair in light
lanes only**. Bit-identity work, ledger repair and re-receipting are permitted and useful.
Benchmarking a provider for promotion is not.

`BASE_TRUE_TPS` counts **true batch-1, non-speculative** decode only. Batching, speculative
tokens, prompt caching and partial-stack throughput do not count toward it.

## Required: the per-token cost ledger

Every token's time attributed, summing to the measured wall time, across:

artifact verification and SHA · container lookup · packed-index decode · CPU orchestration ·
host/device transfer · Metal encode/submit/synchronize · attention and IndexShare · routing ·
shared experts · routed experts · KV update · final head and sampling

An unattributed remainder is itself a finding and must be reported as its own line, not
absorbed into a neighbour.

## Required: integrity moves out of the hot loop, and is not weakened

Verify immutable artifact sections **once**, bind them to the verified manifest, and retain a
tamper-safe verified residency/index state.

**Never weaken startup or artifact integrity to improve TPS.** The measured 2.6x from
re-verification is a caching defect — the same bytes re-hashed thousands of times per run —
and the fix is to verify each tensor once per process, not to stop verifying. A change that
simply disables checking is a regression wearing a speedup's clothes and fails this gate.

## Required: prove the production path is clean

No hidden reference fallback · no NumPy hot loop · no dense reconstruction · no repeated
tensor parsing · no per-token allocation · no unnecessary CPU/GPU round trip.

Absence must be *shown*, not assumed. This campaign's repeated finding is that things are
built and unreachable, or reachable and doing more than anyone believed.

## Optimization order

1. persistent verified tensor registry and buffers
2. native packed-width execution
3. lookup-linear and 2D split kernel selection
4. routed-expert batch/fusion
5. attention/IndexShare fusion
6. GPU routing, KV, head and sampling
7. asynchronous encode/execute overlap
8. command-buffer collapse
9. one replayable token graph
10. artifact ordering, prefetch and page-fault control

## Benchmark contract for every promoted change

Same complete-token workload every time: exact output correctness · cold and warm TPS ·
2K/8K/32K context · TTFT · active bytes/token · operations/token · command buffers/token ·
memory and swap · a sustained 80+ token run.

Correctness first. A change that alters output is not an optimization.

## Closing conditions

The base lane does not close below 50 TPS unless **all four** hold:

1. every admitted native execution grammar was tested;
2. two consecutive optimizations each improved end-to-end TPS by less than 5 percent;
3. the measured memory/compute/dispatch roofline identifies the binding physical limit;
4. the exact remaining architectural change required is sealed in writing.

## After sealing, and only then

Rebenchmark the corrected speculative subsystem · compare `BASE_TRUE_TPS` against
`ACCELERATED_ACCEPTED_TPS` · integrate the winning provider into HIDE · tune streaming, tool
boundaries, rollback, KV and interaction.

## Standing

```
per-token cost ledger        INSTRUMENTED  (default-off exclusive ledger + example;
                             unit-tested; device measurement pending on controller —
                             see reports/base_runtime/PER_TOKEN_COST_LEDGER.md)
integrity out of hot loop    IN PROGRESS  (dense/row memoization delegated; ~2.6x measured available)
production path proven clean NOT STARTED  (static path audit in cost-ledger report; needs device counters)
optimization order           NOT STARTED
BASE_TRUE_TPS                0.1547 tok/s warm hashed | 0.4074 warm no-hash
gap to target                ~323x to 50 tok/s
gap to roofline              ~2000x
```

# Temporal Gravity — minimize causal execution

**Status: CANONICAL.** Extends and supersedes the ceiling language of
`BASE_RUNTIME_MAXIMIZED_GATE.md`.

> Sub-bit minimizes representation.
> Temporal Gravity minimizes causal execution.
> Speculative decoding multiplies the maximized verifier.

50-60 TPS is the **first checkpoint, never the ceiling**.

## Milestones

| | ms/token | TPS |
|---|---|---|
| TG20 | 20 | 50 |
| TG10 | 10 | 100 |
| TG5 | 5 | 200 |
| TG2 | 2 | 500 |
| TG1 | 1 | 1,000 |

TG0.5 and below continue when earned. Descend until the measured physical or
architectural frontier, not until a number looks respectable.

## Verified starting truth

- `BASE_TRUE_TPS` = **0.3961** warm, after one-time verification memoization (2.56x, integrity intact)
- ~**1,171** CPU-visible GPU waits per token
- routed-expert traffic ~**2.58 GB/token**
- 50 TPS = ~16% of cited bandwidth peak; bandwidth-only ceiling ~310 TPS
- the path is dominated by orchestration and synchronization, **not** a proven hardware limit

## THE ORDERING CONSTRAINT — read before planning any collapse work

The 1,171 waits are **not** a command-buffer batching oversight. They are **forced by where
the data lives**:

- the KV cache is a host `Vec<f32>` (`gravity_glm.rs` `cache.keys.extend_from_slice`)
- attention scoring and DSA top-k are **host Rust loops** (`for head in 0..a.n_heads`)
- therefore every projection's output must be read back to the host
- therefore every matvec must `commit_and_wait`

**Command-buffer collapse is impossible while the CPU owns the KV cache and computes
attention.** Attempting the collapse first would fail, and would likely be misread as
"per-dispatch overhead is irreducible" — the same category error that nearly buried
spec-decode behind a bug that was already fixed.

So the true dependency order is:

```
1. GPU-resident state          (activations, KV, router logits/top-k, expert offsets, head/sampling)
2. attention + DSA on device   (removes the readback that forces the waits)
3. THEN command-buffer collapse  1,171 -> <=78 -> <=8 -> <=3 -> 1 replayable token graph
4. THEN persistent GPU causal loop
```

One command buffer per layer (<=78) is an intermediate checkpoint, not a destination.

## Forbidden in the hot loop

per-projection `commit_and_wait` · CPU readback of intermediates · repeated tensor parsing ·
per-token allocation · dense reconstruction · NumPy or reference fallback · repeated
verification.

Absence must be **shown**, not assumed.

## The profiler

Per complete token, with GPU timestamps/counters where available:

CPU encode/orchestration · GPU execution and queue wait · command buffers and
synchronizations · artifact lookup and verification · residency and page faults · packed
decode · attention and IndexShare · routing and experts · KV/state · norm, head and sampling
· bytes · operations · p50/p95/p99.

**No unexplained "orchestration" bucket.** An unattributed remainder is reported as its own
line with its own magnitude, never absorbed into a neighbour.

## Optimization lanes, selected by measured geometry

shared on-chip lookup-linear · 2D split decode-FMA · native packed-width execution · fused
multi-projection MoE waves · attention/IndexShare fusion · persistent argument and buffer
registries · indirect/replayable command encoding · asynchronous CPU/GPU overlap ·
latency-aware `.gravity` ordering · active-expert, KV/state and depth reduction once software
overhead is gone.

Every promotion requires **exact output parity** or the frozen quality contract.

## Sealing conditions — no premature stop

Do not seal `BASE_RUNTIME_MAXIMIZED` at 50, 60 or 100 TPS. Seal only when **all** hold:

1. complete-token traffic and operation ledgers exist;
2. every admitted native execution grammar is measured;
3. synchronization is at the minimum proven architecture;
4. no hidden fallback remains;
5. sustained 2K/8K/32K runs are green;
6. two consecutive high-value waves each improve latency by <5%;
7. the remaining limit is **causally attributed** to bandwidth, compute, KV/state, active
   experts, or sequential depth.

If the current architecture reaches its roof, continue through active-byte, expert, state and
depth collapse rather than imposing a software TPS cap.

## Standing

```
BASE_TRUE_TPS          0.3961  (TG target for 50 TPS is 20 ms/token; currently ~2,525 ms/token)
waits/token            ~1,171  ->  target <=78 -> <=8 -> <=3 -> 1
GPU-resident state     NOT STARTED  (prerequisite for all collapse work)
attention on device    NOT STARTED  (currently host Rust loops)
KV on device           NOT STARTED  (currently host Vec<f32>)
TG profiler            IN PROGRESS  (cost-ledger instrumentation merged; GPU timestamps pending)
ACCELERATED_ACCEPTED_TPS  must never be mixed into BASE_TRUE_TPS
```

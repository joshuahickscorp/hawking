# Phase L5 — Eagle4 chain-decode pipeline restructure

**Goal:** +3-8 dec_tps on Eagle4 chain decode.
**Engineering est:** 3-6 hours focused work.
**Confidence:** MEDIUM — infrastructure shipped, design known.

**Prereq reading:** `acceleration_patterns.md` (patterns 1–10) +
`methodology_distilled_post_f2.md` (patterns 11–20) +
`path_to_100_retool.md` (this phase contributes +5 realistic to the
100-tps target; runs LAST in the sequence since it's smallest and
inherits everything else). Apply the 20/20 pre-launch checklist
before starting. Most-relevant patterns for L5: **16** (loop
restructure shipped as one infra commit with all sync points
defined), **2** (bit-identical parity gate is decisive, bench median
≥3% is the ship bar), **8** (strip-restore diagnostic edits across
the multiple commits this phase will land).

## What code is missing (Pattern 9)

| File | Lines (est) | Purpose |
|---|---|---|
| `crates/dismantle-core/src/model/deepseek_v2.rs` | ~40-60 | Chain-decode loop restructure to dispatch propose on secondary queue with SharedEvent sync |
| `crates/dismantle-core/tests/eagle4_chain_pipeline_smoke.rs` | ~60 | New smoke + parity gate |
| (no new shaders) | — | All Metal already exists |

**Net:** ~100-120 lines of Rust. No new shaders. No new MLX work.

## Why this lever

Currently Eagle4 chain decode at K=4 runs:
```
verifier(step N)  →  argmax_N
chain_propose(seed=argmax_N, k=0..K)  → drafts_(N+1)
verifier(step N+1, drafts_(N+1))  → argmax_(N+1)
```

The chain itself is serial (each k+1 propose depends on k's argmax),
but **the chain's later propose steps don't depend on the verifier's
remaining work after argmax handoff.** Currently propose step k+1
sits idle on CPU/secondary queue while the verifier finishes
post-handoff work (output norm, lm_head, sampling).

Pipeline target:
```
T:    [   verifier step N    ][ post-handoff work ]
T+s:                          [ chain step 0    ][ chain step 1 ][...]
T+2s:                                                              [ verifier step N+1 ]
```

The propose steps 1..K can overlap with verifier's post-handoff
finalization on the primary queue. Estimated savings: 2-5 ms per
verifier step × 25-40 steps/sec = 50-200 ms/sec gain = ~5-15% dec_tps.

## What's already shipped (commit e482463)

- `crate::metal::SharedEventBarrier` — Rust wrapper over MTLSharedEvent
- `TokenCommandBuffer::new_on_secondary(ctx)` — TCB constructor on secondary queue
- `MetalContext::secondary_queue()` accessor
- Profile flag `multi_queue: bool` (default false)
- Smoke test `multi_queue_smoke.rs` proving distinct queues

## What's missing

The actual chain-decode loop restructure. Currently in
`crates/dismantle-core/src/model/deepseek_v2.rs` around line 1611
(the `for k_idx in 0..=chain_k` loop). It calls head propose
sequentially with no overlap opportunity.

## Concrete plan

### Step 1 — identify split points (1 hr)

Read the Eagle4 chain decode loop start to end. Identify three points:

- A: After verifier emits argmax_N (call this `seed_token`)
- B: While verifier's post-handoff work runs (output_norm, lm_head,
  optional sampling for non-greedy)
- C: Just before verifier starts step N+1 (consumes drafts)

Confirm via stderr trace: how many ms separate A and C? That's the
overlap budget.

### Step 2 — encode head propose on secondary TCB (1-2 hr)

Replace the per-step head.forward_full_metal_no_lm_head call (already
takes `use_secondary` param from e482463) so when multi_queue=true:

- Allocate one SharedEventBarrier at decode start
- For each chain step k:
  - Open `tcb_sec = TokenCommandBuffer::new_on_secondary(ctx)`
  - Encode head propose kernels into tcb_sec
  - Commit tcb_sec (returns; verifier work on primary continues)
  - Don't wait; the next chain step depends on this draft's argmax
    which we read on completion

This requires reordering the loop so chain step k+1's input wait is
the only synchronization point.

### Step 3 — synchronization (1 hr)

Two cases:
1. K=1: no overlap inside the chain (single propose). Overlap is
   only between chain_step_0 and verifier_step_N's post-handoff.
2. K>1: chain steps are serial, but step 0 can overlap with verifier
   post-handoff.

Implementation:
```rust
// Before chain step 0:
let prep_value = barrier.encode_signal(verifier_post_handoff_cb);
// (verifier_post_handoff_cb is the primary CB doing output_norm + lm_head)

// chain step 0 on secondary:
let mut tcb_sec = TokenCommandBuffer::new_on_secondary(ctx);
// (head propose doesn't actually depend on verifier post-handoff
//  if we already have argmax_N; the barrier is unused at K=1)
```

For K>1, the secondary TCB serializes chain steps naturally; the
overlap is purely between (chain step 0+) and (verifier post-handoff).

### Step 4 — parity gate (1 hr)

`crates/dismantle-core/tests/eagle4_chain_pipeline_smoke.rs`:
- Run Eagle4 chain decode 16 tokens with `multi_queue=false`
- Run same with `multi_queue=true`
- Assert exact same token output (bit-identical)

The change is dispatch-ordering only; math must not change.

### Step 5 — bench (clean window, 30 min)

`tools/bench/path_to_125_bench.sh` with the kernel profile JSON
toggling `multi_queue=true` and `false`. 10 trials each. Median delta
≥ 3% to ship as default.

## Tests required

- New: `eagle4_chain_pipeline_smoke.rs` (bit-identical parity)
- Pre-existing must still pass: `eagle4_decode_parity` (greedy
  Off-vs-Eagle4 16 tokens), `path_b_parity` (parallel-k correctness)

## Risks + mitigations

1. **Wrong sync** → race condition → non-deterministic output.
   *Mitigation:* parity gate catches this; if it fails, ship as
   `multi_queue=false` default until fixed.

2. **No overlap actually realized** (post-handoff finishes too fast).
   *Mitigation:* if bench shows <3% gain, document and keep flag
   default-off. Code lives behind opt-in flag, no regression risk.

3. **Apple GPU serializes implicitly** despite two queues.
   *Mitigation:* Apple's docs and the M-series scheduler do support
   true parallelism, but it's GPU-time-limited. Multi-queue lets the
   GPU pick the best schedule. Worst case: same as single-queue.

## Acceleration patterns applied

- Pattern 1: chain_decode_smoke is the mid-flight signal (already exists)
- Pattern 2: smoke + parity test + clean bench levels defined above
- Pattern 6: bench runs 10 trials per config
- Pattern 9: "code missing" section is concrete (~100-120 lines)

## Acceptance criteria

- Bit-identical parity with `multi_queue=true` on at least 16 tokens
- Clean-window bench: ≥3% median dec_tps gain at K=4 chain decode
- No regression on K=1 path, ngram-spec path, or Off path

## Next-session quickstart

```
# 1. Read this plan + acceleration_patterns.md
# 2. Open crates/dismantle-core/src/model/deepseek_v2.rs:1611
# 3. Implement the loop restructure
# 4. Verify build + lib tests
# 5. Run parity smoke test
# 6. Cmd-Q Claude + run bench
# 7. Ship as multi_queue=true default if ≥3% gain
```

# Path B — Parallel-K verify-kernel rewrite (design)

**Status:** design only; no Rust changes in this commit.
**Goal:** drop the speculative-decode verify cost from `K × single-forward`
to `~1.5 × single-forward`, unlocking the spec-decode win regime per
`reports/path_to_90/stage3_spec/audit.md`.
**Estimated implementation effort:** 2-4 weeks elapsed (this is the longest
single piece of pure engineering on the path-to-90 plan).
**Estimated win:** brings spec-decode at 70% acceptance from 0.63× (regression)
to 1.6× speedup → ~38-45 dec_tps from current ~24-28.

## The arithmetic this fixes

Today's `forward_tokens_batched_for_test` runs K independent forwards,
one per draft token, in a single Metal command buffer. Wall-clock cost
≈ K × single-forward — the per-token weight-read dominates and isn't
shared across the K calls.

The fix: rewrite the three heaviest kernels in the decode path so a
single dispatch processes K queries simultaneously, sharing the weight
read. Specifically the three kernels that account for >85% of decode time:

| Kernel | Current shape | New (parallel-K) shape | Why this saves work |
|---|---|---|---|
| `mla_decode_kernel_fc` | `q (n_heads, head_dim)`, KV (seq_len, kv_lora_rank) | `q (K, n_heads, head_dim)`, same KV | KV-cache read is the dominant cost; K queries share one read |
| `lm_head_*` GEMV (Q6_K) | `x (hidden,)`, W (vocab, hidden) | `x (K, hidden)`, same W | lm_head weights are 800 MB; K queries share one read |
| `moe_block_batched_indexed` | `x (hidden,)`, route_ids (top_k,) | `x (K, hidden)`, route_ids (K, top_k) | Expert weights are the heaviest read; K queries share each expert's weights |

Pseudo-math, K=4:
- Current verify: `4 * (mla + lm_head + moe + everything_else)` ≈ 4 × 1.0 = 4.0
- Path B verify: `1 * (mla_K + lm_head_K + moe_K + everything_else)` ≈ 1.5 (math + per-K bookkeeping adds ~0.5)
- Headroom: 4.0 / 1.5 = **2.67× faster verify**

Combined with the spec-decode win formula:
- Without Path B at 70% accept: `2.5 tokens / 4.0 cost = 0.63×` (regression)
- With Path B at 70% accept: `2.5 tokens / 1.5 cost = 1.67×` (win)

## Architecture

### Kernel surfaces to modify

**1. MLA decode (attn.metal):**

Current signature (single-token):
```c
kernel void mla_decode_kernel_fc(
    device const half*  c_kv,      // (seq_len, kv_lora_rank)
    device const half*  k_pe,      // (seq_len, qk_rope_head_dim)
    device const half*  q_nope,    // (n_heads, qk_nope_head_dim)
    device const half*  q_rope,    // (n_heads, qk_rope_head_dim)
    device const half*  kv_b_proj, // (kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim))
    device float*       output,    // (n_heads * v_head_dim,)
    ...
);
```

Parallel-K signature:
```c
kernel void mla_decode_kernel_fc_kbatch(
    device const half*  c_kv,            // unchanged (shared KV)
    device const half*  k_pe,            // unchanged
    device const half*  q_nope_batched,  // (K, n_heads, qk_nope_head_dim)
    device const half*  q_rope_batched,  // (K, n_heads, qk_rope_head_dim)
    device const half*  kv_b_proj,       // unchanged (shared)
    device float*       output_batched,  // (K, n_heads * v_head_dim)
    constant uint&      K,
    ...
);
```

Grid: `(n_heads, K, threadgroups_for_seqlen)`. Each threadgroup handles
one (head, k-position) pair against the entire KV cache. The KV read
amortizes across the K thread-group columns via threadgroup memory.

Memory savings per dispatch:
- KV cache: read once into TG memory (per head), reused for all K queries
- kv_b_proj: same — small enough to cache in TG memory

**2. lm_head GEMV (Q6_K):**

Current is a `gemv_f16_dispatch` over `(vocab, hidden) @ (hidden,) -> (vocab,)`.
Each output row reads ALL of `lm_head[r, :]` once.

Parallel-K version: `(vocab, hidden) @ (K, hidden).T -> (K, vocab)`.
The lm_head row is read once and dot-producted with K queries.

For K=4, vocab=102400, hidden=2048:
- Current: 4 * 102400 * 2048 * 2 = **1.68 GB** of weight reads
- Parallel-K: 102400 * 2048 * 2 = **420 MB** of weight reads (4× saving)

Kernel pattern (mirror of `gemv_q6_k_v3` already in repo, adapted for K queries):
```c
kernel void gemv_q6_k_v3_kbatch(
    device const block_q6_k* w,           // unchanged
    device const half*       x_batched,   // (K, hidden)
    device float*            y_batched,   // (K, vocab)
    constant uint&           K,
    ...
);
```

Grid: `(vocab_rows / TG_ROWS, K)`. Each threadgroup processes ROWS_PER_TG
output rows × K queries against the same weight rows.

**3. MoE block (parallel-K routing + indexed GEMM):**

Two sub-kernels affected:
- `gemv_f32_moe_pinned_buf` (gate projection — produces routing logits)
- `moe_block_batched_indexed_metal` (the routed-expert FFN)

For K queries, each query has its own top-k=6 route. Different K queries
may select OVERLAPPING expert sets. The kernel should:
- Compute K independent gate dispatches → K independent route_ids tensors
- Build a SHARED expert-batch: all distinct experts across the K queries
- For each shared expert, GEMM against all K queries that selected it,
  weighted by each query's per-route weight
- Sum weighted contributions back to per-query outputs

In the common case where K queries' top experts overlap (~50-70% of
the time per published MoE-spec papers), this saves expert-weight reads
roughly proportionally to the overlap fraction.

Kernel:
```c
kernel void moe_block_batched_indexed_kbatch(
    device const half*       gate_w,
    device const half*       up_w,
    device const half*       down_w,
    device const half*       x_batched,           // (K, hidden)
    device const uint*       distinct_experts,    // (n_distinct,)
    device const uint*       per_k_route_idx,     // (K, top_k) — index into distinct_experts
    device const half*       per_k_route_weight,  // (K, top_k)
    device float*            y_batched,           // (K, hidden)
    constant uint&           K,
    ...
);
```

### Engine-side dispatcher changes

The engine wraps these kernels via `crates/dismantle-core/src/kernels/`.
Path B adds:

```rust
// In kernels/mod.rs (new file: kernels/parallel_k.rs)
pub fn mla_decode_kernel_fc_kbatch(...) -> Result<()>;
pub fn gemv_q6_k_v3_kbatch(...) -> Result<()>;
pub fn moe_block_batched_indexed_kbatch(...) -> Result<()>;

// In model/deepseek_v2.rs — new method that uses these
fn forward_tokens_batched_parallel_k(&mut self, tokens: &[u32], positions: &[usize])
    -> Result<Vec<Vec<f32>>>;
```

The new method is opt-in via a profile flag:
```json
{ "verify_kernels": "parallel-k" }  // default "sequential"
```

Spec-decode in `engine/spec_decode.rs` checks this flag at startup and
routes verify through the parallel-K path when enabled.

## Correctness gates (per CLAUDE.md kernel-parity gate)

Every kernel needs:

1. **Synthetic-input parity test:** in `tests/correctness/path_b_parity.rs`,
   compare parallel-K output against K sequential runs of the unbatched
   kernel on the SAME inputs. Bar: atol=1e-3 fp16 (matches existing parity
   gates).

2. **Token-identical regression:** `dismantle batch-hash --tokens 64`
   against the existing `tests/golden/_phase1_token_baseline_expanded.hashes`
   must produce bit-identical hashes when spec-decode is OFF (default). When
   spec-decode is ON with the trained head, byte-identical greedy at 64
   tokens vs no-spec baseline.

3. **PPL on WikiText-2 slice (256 samples):** ΔPPL within ±0.5% of FP16-KV
   baseline. Uses the harness shipped in B1 (`tools/bench/ppl_eval.py`).

4. **Profile parity per context-length bucket:** parallel-K MLA may have
   different optimal threads-per-TG depending on context length. Sweep
   {128, 512, 1K, 4K, 16K, 32K} and verify the chosen tile size produces
   parity at each context length.

## Per-kernel implementation order

1. **gemv_q6_k_v3_kbatch** (~3-5 days) — simplest; lm_head is the largest
   single read but the GEMV pattern is well-understood. Implement first to
   validate the K-batch dispatch graph end-to-end with the smallest kernel.

2. **mla_decode_kernel_fc_kbatch** (~5-7 days) — heaviest; KV-cache sharing
   is the real win. Requires careful TG memory budgeting (current MLA uses
   most of available TG SRAM; K-batching may force tile-size reduction).

3. **moe_block_batched_indexed_kbatch** (~5-7 days) — most algorithmically
   novel because of the expert-overlap optimization. Can land as a
   "non-overlap" K-batch first (just K sequential expert calls in one CB)
   and add overlap later as an A/B test.

4. **Engine wire-up + spec-decode integration** (~3-5 days) — once all
   three kernels pass parity, wire into `forward_tokens_batched_parallel_k`
   and gate behind `verify_kernels = "parallel-k"`. Spec-decode then routes
   through this verify automatically.

5. **Autotune sweep + per-context tuning** (~2-3 days) — re-run autotune
   for the new kernels on M3 Pro 18 GB at typical context lengths.

Total: ~3-4 weeks elapsed.

## Risk + mitigation

| Risk | Mitigation |
|---|---|
| Parallel-K MLA exceeds TG SRAM budget | Reduce per-TG tile size; accept lower per-TG occupancy in exchange for K-sharing (expected to still net positive) |
| MoE expert overlap is lower than expected | Non-overlap K-batch still saves ~20-30%; overlap optimization is bonus |
| Cross-K register pressure regresses single-K perf | Gate behind profile flag; default to sequential when spec-decode is off |
| First trained head has low acceptance, hiding Path B wins | Pre-validate Path B on synthetic 100% acceptance to prove the K-cost ratio independently of head quality |

## What this design does NOT change

- Per-token KV-cache write logic (`kv_append`) — already batches naturally
- Embedding lookup, RMSNorm — too cheap to bother
- Sampling — argmax + sample_argmax already batch fine
- The existing `forward_token` (single-token, non-spec) path — untouched

## Connections to other work

- **Tree decoding** (sibling effort, see `reports/path_to_90/tree_decode/design.md`)
  benefits ~directly from Path B: tree verify is a generalization of
  K-parallel verify with different attention masks. The Path B kernels
  written here are the substrate that tree decoding extends.
- **EAGLE-3 head training** (sibling effort, in flight) provides the
  acceptance rates that make Path B worthwhile. Without a trained head,
  Path B has nothing to verify against.
- **Continuous batching** (existing dismantle-serve infra) compounds with
  parallel-K verify in the multi-user-request regime. Each request's
  spec-decode K-batch can pack alongside other requests' K-batches into
  a (sum-of-Ks, hidden) input. Out of scope here; future work.

## Files (to be created)

```
crates/dismantle-core/shaders/parallel_k_attn.metal        (new)
crates/dismantle-core/shaders/parallel_k_lmhead.metal      (new)
crates/dismantle-core/shaders/parallel_k_moe.metal         (new)
crates/dismantle-core/src/kernels/parallel_k.rs            (new)
crates/dismantle-core/src/model/deepseek_v2.rs             (modified — forward_tokens_batched_parallel_k)
crates/dismantle-core/tests/path_b_parity.rs               (new)
profiles/deepseek-v2-lite-q4.m3pro18.json                  (modified — verify_kernels field)
reports/path_to_90/path_b/close.md                         (new, after impl)
```

## Concrete success criterion for the FIRST kernel (gemv_q6_k_v3_kbatch)

At K=4 with synthetic-100%-acceptance verify (forces all 4 draft tokens to
match), measured wall-clock per spec step should be **≤ 1.8× single-token
decode wall-clock**, NOT 4×. If we hit ≤ 1.5×, we're on track for the
full Path B target. If > 2.5×, something is wrong with the K-batch
dispatch and we re-check.

---

# Masked verify integration (path-to-90 step 12)

This section extends the Path B design to cover the **routing-aware
masked-verify variant** of `moe_block_batched_indexed_kbatch`. The
extension is load-bearing for Stage 3 (mask-driven async expert
prefetch, +5–10% tok/s in the expected regime) and converges the
Path B kernel design with EAGLE-4's `mask_logits` output.

## Why this lives in the SAME kernel as Path B

EAGLE-4's `propose()` returns a `26×64` predicted routing mask
(`DraftOutputs::routing_mask` — see
`crates/dismantle-core/src/speculate/draft_head.rs` and
`eagle4_head.rs::DraftOutputs`). Production decode needs to:

1. Read the predicted-active expert set for layers 1..26 from the
   mask.
2. Prefetch those experts' Q4 weight tiles into Apple Silicon's GPU
   L2 / TG residency hint **before** the verify dispatch needs them.
3. Run the K-batched MoE verify kernel with the prefetched-vs-on-
   demand split visible to its dispatch order.

Implementing prefetch + masked dispatch as a separate kernel from the
plain `moe_block_batched_indexed_kbatch` would mean two kernels with
~95% shared body, two parity tests, two threadgroup-memory budgets to
audit. Designing them as ONE kernel with the mask consumed as an
extra input buffer (zeros = "no prediction, on-demand load all")
keeps the substrate single and lets the mask be empty when EAGLE-4
isn't loaded.

## Kernel signature

```metal
// crates/dismantle-core/shaders/parallel_k_moe_masked.metal
kernel void moe_block_batched_indexed_kbatch_masked(
    device const float*   x_kbatch                [[buffer(0)]],  // (K, hidden)
    device const uint*    routed_indices_kbatch   [[buffer(1)]],  // (K, TOP_K_ROUTED=6)
    device const float*   routed_weights_kbatch   [[buffer(2)]],  // (K, TOP_K_ROUTED=6)
    device const uchar*   predicted_mask          [[buffer(3)]],  // (N_ROUTED=64)
                                                                  //   0 = not predicted, on-demand
                                                                  //   1 = predicted, prefetched
    // Layer-resident weight tiles (pinned from prior dispatch OR
    // just-prefetched via predicted_mask hint).
    device const uchar*   routed_gate_blocks      [[buffer(4)]],
    device const uchar*   routed_up_blocks        [[buffer(5)]],
    device const uchar*   routed_down_blocks      [[buffer(6)]],
    // Shared-expert path identical to plain k-batch — fused at dispatch
    // level since shared is always evaluated.
    device const uchar*   shared_gate_blocks      [[buffer(7)]],
    device const uchar*   shared_up_blocks        [[buffer(8)]],
    device const uchar*   shared_down_blocks      [[buffer(9)]],
    // Output: (K, hidden) — accumulated routed + shared contribution.
    device float*         out_kbatch              [[buffer(10)]],
    // Sizes via function constants (compiler unrolls).
    constant uint&        K                       [[function_constant(0)]],
    constant uint&        hidden                  [[function_constant(1)]],
    constant uint&        moe_intermediate        [[function_constant(2)]],
    constant uint&        n_shared_experts        [[function_constant(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
);
```

Rust-side dispatcher (mirrors existing `parallel_k.rs` shape):

```rust
// crates/dismantle-core/src/kernels/parallel_k.rs
pub fn moe_block_batched_indexed_kbatch_masked(
    ctx: &MetalContext,
    cb: &CommandBuffer,
    x_kbatch: &Buffer,                  // (K, hidden) device
    routed_indices_kbatch: &Buffer,     // (K, TOP_K_ROUTED) uint
    routed_weights_kbatch: &Buffer,     // (K, TOP_K_ROUTED) f32
    predicted_mask: Option<&Buffer>,    // (N_ROUTED) u8; None ⇒ all-on-demand
    routed_gate_blocks: &Buffer,
    routed_up_blocks: &Buffer,
    routed_down_blocks: &Buffer,
    shared_gate_blocks: &Buffer,
    shared_up_blocks: &Buffer,
    shared_down_blocks: &Buffer,
    out_kbatch: &mut Buffer,            // (K, hidden) device
    k: usize,
    hidden: usize,
    moe_intermediate: usize,
    n_shared_experts: usize,
) -> Result<()>;
```

When `predicted_mask == None`, the kernel runs identically to the
plain `moe_block_batched_indexed_kbatch` — single code path, no
runtime branch on a Bool function constant.

## Prefetch dispatch flow

```
Step N (decode):
  ┌──────────────────────────────────────────────────────────┐
  │ 1. CPU walk: capture (h_low, h_mid, h_high, h_shared)    │
  │ 2. eagle4 head.propose → (top_K_tokens, routing_mask,    │
  │                            calib)                        │
  │                                                          │
  │ Per MoE verify layer (1..26):                            │
  │ 3a. predicted_mask for THIS layer = routing_mask[layer]  │
  │ 3b. ASYNC: MTLResidencySet.add(expert_tiles where        │
  │       predicted_mask bit == 1)                           │
  │ 3c. ENCODE: moe_block_batched_indexed_kbatch_masked      │
  │       with predicted_mask buffer ← layer's mask row      │
  │                                                          │
  │ 4. Commit + wait.                                        │
  └──────────────────────────────────────────────────────────┘
```

The residency-set add at 3b is the Apple-Silicon-native prefetch
primitive (`MTLResidencySet.addAllocation` via `metal-rs`'s
`MTLResidencySet` binding). It hints the GPU memory controller to
keep those buffer regions in residency / cache; on UMA M-series this
is largely an L2-promotion hint rather than a copy. Cost is ~µs,
amortized across the verify kernel's runtime.

## Threadgroup memory budget (with mask buffer)

The plain `moe_block_batched_indexed_kbatch` design already accounts
for ~22 KB / 32 KB threadgroup memory (Path B § "Threadgroup memory
audit"). The masked variant adds:

- `predicted_mask` is read into a shared `threadgroup uchar[64]` once
  per dispatch (1 byte per routed expert × 64 = 64 B). Negligible.
- A routed-indices fast-path branch (skip expert evaluation when
  the mask says "predicted-inactive AND not in top-K") adds two
  comparison instructions per inner-loop iteration. Compute cost
  ≈ 0; no extra threadgroup memory.

Net budget for masked variant: ~22 KB + 64 B ≈ 22.1 KB. Fits the
32 KB / core budget with margin.

## Acceptance / parity test

`crates/dismantle-core/tests/path_b_eagle4_parity.rs` (new — to be
landed alongside step 15):

1. K=4 masked-verify vs K=4 plain (unmasked) on the SAME inputs —
   must be bit-identical at atol=1e-3 fp16. The masked path differs
   only in dispatch ORDER (prefetched experts dispatched first); the
   mathematical output is identical.
2. K=4 masked-verify vs K=1 sequential single-token MoE — must be
   bit-identical at atol=1e-3 fp16.
3. With `best_recall.npz` (eagle4 v2-routing checkpoint, 26 %
   recall): wall-clock should be **≥ 5 % faster** than unmasked
   thanks to prefetch hits.

## Dependencies + ordering against Stage 0.5

Path B kernel work (this step + 13–16) is gated on:

- **CPU `attention()` divergence fix** (chip spawned 2026-05-18) —
  the bit-identical regression at step 9 needs to land before any
  Path B parity test can validate.
- **Routing recall fine-tune** (step 11, target ≥ 60 % recall) —
  masked verify's ≥ 5 % speedup hypothesis assumes a meaningful
  fraction of predicted experts are actually fired. At eagle4 v3's
  17.78 % top-8 recall, masked prefetch is wasted bandwidth more
  often than it hits. Land step 11 OR ship masked verify behind a
  recall-gated env var until step 11 closes.

## File deltas after step 12 (this commit) lands

- `reports/path_to_90/path_b/design.md` — extended with this section.
- No code changes; design-only commit. Kernel implementation lands in
  steps 13–15 of the execution plan, in that order (easiest first to
  validate the dispatch graph: `gemv_q6_k_v3_kbatch` → `mla_decode_fc`
  → masked-MoE).

## Effort estimate (revised post-localization-halt)

The original plan estimates 5–7 days per kernel × 3 kernels = 15–21
days for Stage 2. Recommend deferring kernel implementation until
after the CPU `attention()` fix lands and step 9's bit-identical
regression passes — otherwise we'd be building K-batched verify on
top of a numerically wrong V2-Lite forward and have no way to
validate parity at K>1.

Realistic landing sequence (gated on attention() fix):

```
[attended] fix CPU attention()         — chip queued 2026-05-18
[compute]  step 9 regression passes    — clean window
[attended] step 11 routing-recall fine-tune (1 day, Python-side)
[arch]     step 13: gemv_q6_k_v3_kbatch
[compute]  step 13 parity test passes
[arch]     step 14: mla_decode_kernel_fc_kbatch
[compute]  step 14 parity test passes
[arch]     step 15: moe_block_batched_indexed_kbatch_masked
[compute]  step 15 parity test (3-way: masked vs unmasked vs K=1)
[arch]     step 16: forward_tokens_batched_parallel_k wire-up
[compute]  step 17: Stage 2 measurement (38–50 tok/s target)
```


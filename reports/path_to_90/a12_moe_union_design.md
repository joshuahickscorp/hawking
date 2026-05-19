# A1.2 — MoE expert-union K-batched kernel (design)

**Status:** design only — implementation deferred to a fresh session
after Branch 3 training lands and we have post-training bench
numbers. The decision to commit ~10-15 h to A1.2 should rest on
whether the post-training Eagle4-chain dec_tps shows that MoE
verify-time is actually the next bottleneck (vs Phase F / Stage 0.5).

## Problem

Path B Stage 2.5 `moe_block_batched_indexed_kbatch_tcb` is a
"no-overlap baseline" that re-issues K independent MoE forwards
inside one TCB. Per the Branch 1 wire-up (commit `5aec2bf`), the
parallel-k path now amortizes MLA + lm_head across K verify queries
— but each layer's MoE block is still K separate dispatches reading
the full expert weight set per query. At V2-Lite:

  Per-query MoE per layer:
    6 routed × 3 projs × ~5 MB Q4_K_M  +  shared 3 projs × ~3 MB
                                                            ≈ 99 MB
  Per token (27 layers):                                   ≈ 2.7 GB
  K=4 verify step (no overlap):                            ≈ 10.8 GB
  K=4 with ~50-70% routing overlap (union):                ≈ 3.0 GB
  K=4 with ~70% overlap saving (union):                    ≈ 7.8 GB / step

At M3 Pro 150 GB/s memory bandwidth, the union save is ~50 ms per
verify step (current chain step is ~277 ms → ~225 ms). Stacks with
Branch 3 head retrain: at ~60% accept post-training, chain emits ~5
tokens / 225 ms ≈ **22 dec_tps** vs current 7.23. Real headroom.

## Why not just prefetch?

The "MTLResidencySet.addAllocation prefetch hint" approach
(AUTONOMOUS_PLAN.md §6 D2) gives the OS/driver advance notice but
doesn't change the kernel-side weight read pattern. Cache benefit is
bounded: M3 Pro L2 ≈ 24 MB, total MoE weights per layer ≈ 99 MB.
The expert weights don't fit in L2; prefetch only helps if K=4
queries SHARE the same expert read (i.e., the union pattern). So
prefetch ≈ a weaker version of true union routing.

## Architecture — sort-based union (recommended)

The cleanest dispatch graph that matches V2-Lite's per-token MoE
shape:

```
Existing per-token MoE chain (preserved):
  gemv_f32_moe → moe_logits_buf (n_experts,)            [1 dispatch per K query]
  moe_topk_gate → (route_ids, route_weights) (top_k,)   [1 dispatch per K query]

NEW union-routing kernels (this commit):
  union_routes_sort → (sorted_triples) ((K·top_k),)     [1 dispatch total]
    triples: (expert_id, k_idx, slot, weight) sorted by expert_id
  union_routes_segment → (segment_starts) (n_experts+1,) [1 dispatch total]
    inclusive prefix-scan over sorted triples → start offsets per expert

NEW union expert kernels:
  moe_gate_up_union → routed_act packed                  [1 dispatch total]
    grid: (rows/8, n_distinct_experts, 1)
    each TG reads one expert's gate+up Q4 weight ONCE, loops
    over the (k_idx, slot, weight) range for its expert, writes
    to routed_act[k_idx, slot, mid_row]
  silu_mul_union → routed_act (in-place)                 [1 dispatch total]
  moe_down_union → routed_out packed                     [1 dispatch total]
    grid: (hidden/8, n_distinct_experts, 1)
    same pattern, output to routed_out[k_idx, slot, hidden_row]

Existing shared expert + accumulate (per K query):
  shared gate+up+silu_mul+down → shared_out[k]           [4 dispatches per K]
  route_accumulate → out_buf[k]                           [1 dispatch per K]

Total per layer per K=4: ~4 union + (6 shared + 4 acc) ≈ 14 dispatches
  vs current per layer per K=4:  K × 4 dispatches = 16-24 dispatches
  Roughly comparable dispatch count; bandwidth wins from union.
```

### Sort-based union — why sort?

The grouping problem ("which K queries selected each expert") needs a
mapping that's GPU-friendly. Options:

1. **Hash table on GPU** — race-y, requires atomics on uint32 buckets.
   Fast for fixed-size hashes but adds complexity.
2. **Bitmap then expand** — one bit per (K, expert), reduce, expand to
   triples. Two passes; memory-efficient at small n_experts (64).
3. **Sort + segment scan** — sort all K·top_k triples by expert_id; the
   sort itself is small (24 triples at K=4 top_k=6). Then one pass to
   compute segment starts (where each expert's range begins). Each
   union expert kernel TG reads its segment's [start, end) range and
   iterates.

Sort wins on simplicity. 24 triples is tiny; even a single-thread
bitonic sort or insertion sort is sub-microsecond. The segment scan is
a 1-thread reduction. Implementation budget: ~100 lines of MSL for
both.

## Detailed kernel pseudocode

### `union_routes_sort`

```metal
kernel void union_routes_sort(
    device const uint*   route_ids       [[buffer(0)]],  // (K, top_k) u32
    device const float*  route_weights   [[buffer(1)]],  // (K, top_k) f32
    device       uint*   sorted_expert   [[buffer(2)]],  // (K*top_k,) u32
    device       uint*   sorted_kidx     [[buffer(3)]],  // (K*top_k,) u32 → which K query
    device       uint*   sorted_slot     [[buffer(4)]],  // (K*top_k,) u32 → which top_k slot
    device       float*  sorted_weight   [[buffer(5)]],  // (K*top_k,) f32
    constant     uint&   k_batch         [[buffer(6)]],
    constant     uint&   top_k           [[buffer(7)]],
    threadgroup  uint*   shmem_expert    [[threadgroup(0)]],   // K*top_k
    threadgroup  uint*   shmem_kidx      [[threadgroup(1)]],
    threadgroup  uint*   shmem_slot      [[threadgroup(2)]],
    threadgroup  float*  shmem_weight    [[threadgroup(3)]],
    uint                 tid             [[thread_position_in_threadgroup]])
{
    const uint N = k_batch * top_k;
    // 1. Load + tag each (kk, slot) entry into shmem.
    if (tid < N) {
        uint kk = tid / top_k;
        uint slot = tid % top_k;
        shmem_expert[tid] = route_ids[kk * top_k + slot];
        shmem_kidx[tid]   = kk;
        shmem_slot[tid]   = slot;
        shmem_weight[tid] = route_weights[kk * top_k + slot];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Single-thread insertion sort on expert_id (N ≤ 32 typically).
    if (tid == 0) {
        for (uint i = 1; i < N; ++i) {
            uint e = shmem_expert[i];
            uint k = shmem_kidx[i];
            uint s = shmem_slot[i];
            float w = shmem_weight[i];
            int j = int(i) - 1;
            while (j >= 0 && shmem_expert[j] > e) {
                shmem_expert[j+1] = shmem_expert[j];
                shmem_kidx[j+1]   = shmem_kidx[j];
                shmem_slot[j+1]   = shmem_slot[j];
                shmem_weight[j+1] = shmem_weight[j];
                --j;
            }
            shmem_expert[j+1] = e;
            shmem_kidx[j+1]   = k;
            shmem_slot[j+1]   = s;
            shmem_weight[j+1] = w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Write out.
    if (tid < N) {
        sorted_expert[tid] = shmem_expert[tid];
        sorted_kidx[tid]   = shmem_kidx[tid];
        sorted_slot[tid]   = shmem_slot[tid];
        sorted_weight[tid] = shmem_weight[tid];
    }
}
```

Dispatch: 1 TG with 32 threads. ~1 μs.

### `union_routes_segment`

```metal
kernel void union_routes_segment(
    device const uint*  sorted_expert    [[buffer(0)]],   // (N,)
    device       uint*  segment_starts   [[buffer(1)]],   // (N_EXPERTS+1,)
    device       uint*  segment_experts  [[buffer(2)]],   // (N_DISTINCT,) — packed list of distinct expert IDs
    device       uint*  n_distinct       [[buffer(3)]],   // scalar
    constant     uint&  N                [[buffer(4)]],
    constant     uint&  n_experts        [[buffer(5)]],
    uint                tid              [[thread_position_in_threadgroup]])
{
    // Single-threaded pass — N is tiny (24 at K=4).
    if (tid != 0) return;

    // Initialize segment_starts to "no entries" (N indicates empty).
    for (uint e = 0; e <= n_experts; ++e) segment_starts[e] = N;

    // Walk sorted_expert; record start of each unique expert's run.
    uint distinct = 0;
    uint prev_e = (uint)-1;
    for (uint i = 0; i < N; ++i) {
        uint e = sorted_expert[i];
        if (e != prev_e) {
            segment_starts[e] = i;
            segment_experts[distinct++] = e;
            prev_e = e;
        }
    }
    segment_starts[n_experts] = N;  // sentinel
    n_distinct[0] = distinct;
}
```

Dispatch: 1 TG with 1 thread. ~0.5 μs.

### `moe_gate_up_union` (and `moe_down_union` analog)

Reuses the existing v2t fused-gu math but with the union dispatch
shape. Grid: `(rows/8, n_distinct_experts, 1)`. Each TG handles 8
output rows × ALL K queries that selected its expert.

```metal
kernel void moe_gate_up_union_v2t(
    device const uchar* w_all          [[buffer(0)]],   // model_buf
    device const uint*  segment_experts[[buffer(1)]],   // (n_distinct,) expert IDs
    device const uint*  segment_starts [[buffer(2)]],   // (n_experts+1,)
    device const uint*  sorted_kidx    [[buffer(3)]],   // (N,)
    device const uint*  sorted_slot    [[buffer(4)]],   // (N,)
    device const float* per_k_x        [[buffer(5)]],   // (K, hidden) packed
    device       float* routed_act     [[buffer(6)]],   // (K, top_k, routed_mid)
    constant     ulong& gate_base_off  [[buffer(7)]],
    constant     ulong& up_base_off    [[buffer(8)]],
    constant     uint&  rows           [[buffer(9)]],   // routed_mid
    constant     uint&  cols           [[buffer(10)]],  // hidden
    constant     uint&  top_k          [[buffer(11)]],
    uint2               tid2           [[thread_position_in_threadgroup]],
    uint2               tgp            [[threadgroup_position_in_grid]],
    uint                simd_lane      [[thread_index_in_simdgroup]],
    uint                simd_id        [[simdgroup_index_in_threadgroup]])
{
    uint base_row    = tgp.x * 8u + simd_id;
    uint expert_idx  = tgp.y;           // index into segment_experts
    if (base_row >= rows) return;

    uint expert = segment_experts[expert_idx];
    uint seg_start = segment_starts[expert];
    // Find this expert's segment end by scanning forward until next expert
    // (or rely on a precomputed segment_ends — slightly cleaner; in pseudocode
    // here we walk until expert changes).
    uint seg_end = seg_start;
    while (seg_end < (top_k * /*k_batch*/4u) /* TODO: pass k_batch */
           && /* sorted_expert[seg_end] == expert — needs buffer */ true) {
        ++seg_end;
    }
    // (Implementation note: pass sorted_expert + N as kernel args so we
    // can detect the boundary cleanly.)

    uint blocks_per_row = cols / 256u;
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 144ul;
    uint64_t gate_row_off = gate_base_off
        + (uint64_t)expert * per_matrix_bytes
        + (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    uint64_t up_row_off = up_base_off
        + (uint64_t)expert * per_matrix_bytes
        + (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;

    // For each query in this expert's segment, compute gate@x_k + up@x_k +
    // silu*mul. Weight stays in L1/L2 across the seg_end - seg_start passes.
    for (uint i = seg_start; i < seg_end; ++i) {
        uint kk   = sorted_kidx[i];
        uint slot = sorted_slot[i];
        device const float* x_k = per_k_x + (uint64_t)kk * (uint64_t)cols;

        float gate_partial = 0.0f;
        float up_partial   = 0.0f;
        for (uint b = 0; b < blocks_per_row; ++b) {
            uint64_t gbo = gate_row_off + (uint64_t)b * 144ul;
            uint64_t ubo = up_row_off   + (uint64_t)b * 144ul;
            for (uint k = 0; k < 8u; ++k) {
                uint elem = k * 32u + simd_lane;
                float x_e = x_k[(uint64_t)b * 256ul + (uint64_t)elem];
                gate_partial += q4_k_value(w_all, gbo, elem) * x_e;
                up_partial   += q4_k_value(w_all, ubo, elem) * x_e;
            }
        }
        gate_partial = simd_sum(gate_partial);
        up_partial   = simd_sum(up_partial);
        if (simd_lane == 0u) {
            float silu_mul = (gate_partial / (1.0f + exp(-gate_partial))) * up_partial;
            routed_act[
                ((uint64_t)kk * (uint64_t)top_k + (uint64_t)slot) * (uint64_t)rows
                + (uint64_t)base_row
            ] = silu_mul;
        }
    }
}
```

Key win: the inner loop over `blocks_per_row` (which dominates wall
time and bandwidth) executes ONCE per (expert, row). The outer loop
over `i ∈ [seg_start, seg_end)` reuses the weight read in L1/L2 across
all K queries that selected this expert. At 50% overlap, each expert
serves ~2 queries instead of being read 2× separately → ~2× bandwidth
saving.

### `moe_down_union_v2t`

Same pattern but with rows=hidden=2048, cols=routed_mid=1408. Input
is `routed_act[k, slot, :]`, output is `routed_out[k, slot, :]`.

### Per-query accumulate

The existing `encode_route_accumulate_tcb` already takes `routed_out`
(routes × hidden), `route_weights`, and per-K input/output. We need
to invoke it per K query with the SAME route_weights as before (the
union didn't change the route weights — just the dispatch order).

Actually — we need to reshape `routed_out (K, top_k, hidden)` back
into per-K `routed_out_k (top_k, hidden)` slices, then call
`encode_route_accumulate_tcb` K times. OR have a new
`route_accumulate_kbatch` kernel that handles all K in one dispatch.

K dispatches is cheap (each is K × top_k × hidden ≈ 16K f32 reads).
Single kbatch dispatch is cleaner. Implement both, pick by A/B.

## Rust dispatcher

```rust
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn moe_block_kbatch_union_tcb(
    tcb: &mut TokenCommandBuffer<'_>,
    model_buf: &PinnedBuffer,
    routed_gate_offset: usize,
    routed_up_offset: usize,
    routed_down_offset: usize,
    per_k_route_ids: &PinnedBuffer,       // packed (K, top_k) u32
    per_k_route_weights: &PinnedBuffer,   // packed (K, top_k) f32
    per_k_x: &PinnedBuffer,               // packed (K, hidden) f32
    per_k_out: &PinnedBuffer,             // packed (K, hidden) f32 — output
    shared_route_ids_buf: &PinnedBuffer,
    shared_gate_offset: Option<usize>,
    shared_up_offset: Option<usize>,
    shared_down_offset: Option<usize>,
    hidden: usize,
    routed_mid: usize,
    shared_mid: usize,
    top_k: usize,
    k_batch: usize,
    n_experts: usize,
    // Scratch buffers (allocated by caller, sized for max K × top_k):
    sorted_expert: &PinnedBuffer,
    sorted_kidx: &PinnedBuffer,
    sorted_slot: &PinnedBuffer,
    sorted_weight: &PinnedBuffer,
    segment_starts: &PinnedBuffer,
    segment_experts: &PinnedBuffer,
    n_distinct: &PinnedBuffer,
    routed_act_packed: &PinnedBuffer,     // (K, top_k, routed_mid)
    routed_out_packed: &PinnedBuffer,     // (K, top_k, hidden)
    shared_act: &PinnedBuffer,
    shared_out: &PinnedBuffer,
) -> Result<()> { ... }
```

DecodeArena additions (per-K MoE scratch):
- `batch_route_ids_packed` (K × top_k × u32)
- `batch_route_weights_packed` (K × top_k × f32)
- `batch_routed_act_packed` (K × top_k × routed_mid × f32) — actually quite large: 4 × 6 × 1408 × 4 = 135 KB
- `batch_routed_out_packed` (K × top_k × hidden × f32) — 4 × 6 × 2048 × 4 = 196 KB
- union sort scratch buffers — tiny

Total per-K scratch addition: ~350 KB. Acceptable on M3 Pro 18 GB.

## Parity test

Two parity gates required:

1. **Synthetic K=1 → bit-identical to existing `encode_moe_block_batched_indexed_tcb_with_scratch`.**
   Same inputs, same outputs. Validates the union dispatch graph
   produces correct results when there's only one query (no overlap
   possible).

2. **Synthetic K=4 random routes → max_abs_diff < 1e-3 vs K independent
   `encode_moe_block_batched_indexed_tcb_with_scratch` calls.** The
   union path's math is identical to per-K independent (same gemv
   reductions, same SiLU, same weights); only dispatch order differs.
   Output should be bit-identical modulo simd_sum non-determinism. The
   1e-3 atol accounts for that.

## A/B gate

If wall-clock improvement is <10% at K=4 on V2-Lite MoE shape (clean
window), ship the kernel as available-but-not-default in the profile
flag. Otherwise enable by default at K≥2 under `verify_kernels=parallel-k`.

The Stage 3.2 masked-prefetch variant (predicted_mask hint) becomes
a second-tier dispatcher that wraps union and adds residency-set
prefetch when `predicted_mask` is provided.

## Effort estimate

- 3 new MSL kernels (sort, segment_scan, union_gate_up + union_down): ~400-600 lines MSL
- 2 new Rust dispatchers (per-kernel TCB wrappers): ~200 lines
- 1 high-level wrapper that ties them together: ~150 lines
- Arena schema extension: ~30 lines
- Parity tests (K=1 + K=4 against existing baseline): ~150 lines
- Wire into `forward_tokens_batched_parallel_k` Phase C: ~80 lines
- shader_hash regen + profile bump

Total: ~1000-1200 lines of new code + ~10-15 hours of careful
implementation and parity validation. Single contiguous session
recommended.

## Decision matrix (post-Branch-3 bench)

| Post-training bench result | Decide A1.2 |
|---|---|
| Eagle4 chain K=4 ≥ 20 dec_tps AND ngram dec_tps ≈ Off | ship A1.2 (MoE is next bottleneck — verify time still dominates) |
| Eagle4 chain K=4 < 15 dec_tps | head still under-accepting — fix Branch 3 first (more epochs, larger K, different multi-step pattern) before A1.2 |
| Eagle4 chain K=4 in 15-20 band | borderline — start Phase F levers (lower risk, faster ship) first, return to A1.2 if F doesn't close the gap |

The "best path" answer depends on the bench. This design doc is
ready to drop into implementation when the data says so.

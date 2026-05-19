// path-to-125 Branch 2 (A1.2) — MoE expert-union K-batched routing kernels.
//
// These two kernels build the union-routing data structures that
// the K-batched expert-union GEMM kernels (in moe_union_expert.metal,
// follow-up commit) consume.
//
// Input:
//   route_ids        (K, top_k) u32  — per-K-query top_k routed experts
//   route_weights    (K, top_k) f32  — per-K-query top_k routing weights
//
// Output (consumed by moe_*_union kernels):
//   sorted_expert    (K * top_k,) u32  — flat list, sorted by expert_id
//   sorted_kidx      (K * top_k,) u32  — for each sorted entry, which K query
//   sorted_slot      (K * top_k,) u32  — for each sorted entry, which top_k slot
//                                         within that query (0..top_k-1)
//   sorted_weight    (K * top_k,) f32  — for each sorted entry, the route_weight
//   segment_starts   (n_experts+1,) u32  — sorted_expert[segment_starts[e]] = e
//                                           (or N=K*top_k if expert e has no entries)
//   segment_experts  (n_experts,) u32  — packed list of distinct experts (length n_distinct)
//   n_distinct       (1,) u32  — number of distinct experts in the union
//
// At K=1 the "union" degenerates to top_k entries already sorted by
// the topk_gate kernel (which writes in descending weight order, NOT
// expert order — so sort is still useful at K=1 for consistency).
//
// Dispatch shapes:
//   union_routes_sort:    1 TG with 32 threads (N = K * top_k ≤ 32 typical;
//                          single-thread insertion sort suffices for N ≤ 32).
//   union_routes_segment: 1 TG with 1 thread (linear scan over sorted_expert).
//
// Bandwidth: both kernels are tiny (24-element working set at K=4 top_k=6).
// Together they cost ~2 μs at K=4 — negligible vs the MoE expert GEMMs they
// enable to amortize (~50 ms saved per K=4 verify step with 70% routing overlap).

#include <metal_stdlib>
using namespace metal;

// ── union_routes_sort ───────────────────────────────────────────────────────
//
// Builds sorted triples (expert, kidx, slot, weight) ordered by expert.
//
// N = k_batch * top_k. Caller dispatches 1 TG with at least N threads
// (32 covers K≤8 top_k=4 OR K≤4 top_k=8 etc., so 32 threads is safe for
// all K*top_k ≤ 32 cases).
kernel void union_routes_sort(
    device const uint*  route_ids       [[buffer(0)]],   // (K, top_k) u32
    device const float* route_weights   [[buffer(1)]],   // (K, top_k) f32
    device       uint*  sorted_expert   [[buffer(2)]],   // (K*top_k,) u32
    device       uint*  sorted_kidx     [[buffer(3)]],   // (K*top_k,) u32
    device       uint*  sorted_slot     [[buffer(4)]],   // (K*top_k,) u32
    device       float* sorted_weight   [[buffer(5)]],   // (K*top_k,) f32
    constant     uint&  k_batch         [[buffer(6)]],
    constant     uint&  top_k           [[buffer(7)]],
    threadgroup  uint*  shmem_expert    [[threadgroup(0)]],   // 32 floats max
    threadgroup  uint*  shmem_kidx      [[threadgroup(1)]],
    threadgroup  uint*  shmem_slot      [[threadgroup(2)]],
    threadgroup  float* shmem_weight    [[threadgroup(3)]],
    uint                tid             [[thread_position_in_threadgroup]])
{
    const uint N = k_batch * top_k;

    // 1. Cooperative load + tag.
    if (tid < N) {
        uint kk = tid / top_k;
        uint slot = tid - kk * top_k;
        shmem_expert[tid] = route_ids[(uint64_t)kk * top_k + slot];
        shmem_kidx[tid]   = kk;
        shmem_slot[tid]   = slot;
        shmem_weight[tid] = route_weights[(uint64_t)kk * top_k + slot];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Single-thread insertion sort on expert_id (N ≤ 32 in practice).
    //    Sort is stable, so equal-expert entries preserve K-query order.
    if (tid == 0) {
        for (uint i = 1; i < N; ++i) {
            uint e = shmem_expert[i];
            uint k = shmem_kidx[i];
            uint s = shmem_slot[i];
            float w = shmem_weight[i];
            int j = int(i) - 1;
            while (j >= 0 && shmem_expert[j] > e) {
                shmem_expert[j + 1] = shmem_expert[j];
                shmem_kidx[j + 1]   = shmem_kidx[j];
                shmem_slot[j + 1]   = shmem_slot[j];
                shmem_weight[j + 1] = shmem_weight[j];
                --j;
            }
            shmem_expert[j + 1] = e;
            shmem_kidx[j + 1]   = k;
            shmem_slot[j + 1]   = s;
            shmem_weight[j + 1] = w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Cooperative writeback.
    if (tid < N) {
        sorted_expert[tid] = shmem_expert[tid];
        sorted_kidx[tid]   = shmem_kidx[tid];
        sorted_slot[tid]   = shmem_slot[tid];
        sorted_weight[tid] = shmem_weight[tid];
    }
}

// ── union_routes_segment ────────────────────────────────────────────────────
//
// Walks sorted_expert and records:
//   segment_starts[e]  = position of first occurrence of expert e in
//                        sorted_expert (or N if e is absent)
//   segment_starts[n_experts] = N (sentinel, used to compute segment ends)
//   segment_experts[0..n_distinct) = packed list of distinct experts
//   n_distinct[0]      = number of distinct experts
//
// Caller dispatches 1 TG with 1 thread.
kernel void union_routes_segment(
    device const uint*  sorted_expert    [[buffer(0)]],  // (N,)
    device       uint*  segment_starts   [[buffer(1)]],  // (n_experts+1,)
    device       uint*  segment_experts  [[buffer(2)]],  // (n_experts,) — packed
    device       uint*  n_distinct       [[buffer(3)]],  // (1,)
    constant     uint&  N                [[buffer(4)]],
    constant     uint&  n_experts        [[buffer(5)]],
    uint                tid              [[thread_position_in_threadgroup]])
{
    if (tid != 0) return;

    // Initialize segment_starts to N (= "no entries").
    for (uint e = 0; e <= n_experts; ++e) segment_starts[e] = N;

    // Linear pass over sorted_expert (N ≤ 32 typical).
    uint distinct = 0;
    uint prev_e = 0xFFFFFFFFu;
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

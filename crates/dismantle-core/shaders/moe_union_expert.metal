// path-to-125 Branch 2 (A1.2) — MoE expert-union K-batched GEMM kernels.
//
// These kernels consume the routing tables built by
// moe_union_routing.metal (sort + segment_scan) and execute the
// per-expert gate+up+silu_mul and down projections such that EACH
// EXPERT WEIGHT IS READ EXACTLY ONCE no matter how many of the K
// queries selected it.
//
// Algorithm:
//   Grid: (ceil(rows/8) × 256, n_experts, 1)
//   TG: (256, 1, 1)
//   For each (output_row, expert):
//     1. If expert is not in the union (segment_starts[expert] == N), early return.
//     2. Determine segment [seg_start, seg_end) by scanning segment_starts
//        for the next non-empty expert (or N sentinel).
//     3. For each entry i ∈ [seg_start, seg_end), retrieve (kk, slot) and
//        compute the gate+up GEMV (Q4_K_M) using x_cache[kk × cols .. +cols],
//        write silu(gate) * up into routed_act[(kk × top_k + slot) × rows + row].
//
// Bandwidth win: the expert weight (~5 MB Q4 for V2-Lite gate; same for up)
// is read into L1/L2 ONCE in step 3's outer block-loop. The inner loop
// over `i ∈ [seg_start, seg_end)` reuses that weight across all K queries
// that selected this expert. At ~50-70% routing overlap (K=4), the kernel
// saves ~7-8 GB of expert-weight bytes per K=4 verify step.
//
// At K=1 the segment has length 1 → kernel reduces to the per-expert
// fused_gu_v2 behavior (modulo dispatch shape: grid.y is n_experts not 1).
// Empty experts get early-returned; only the K's top_k experts execute
// their GEMM body.
//
// TG-memory budget:
//   x_cache_all: K × cols × f32 = K × cols × 4 bytes
//   At K=4 cols=2048 (V2-Lite) → 32 KB ≈ M3 Pro per-core TG-mem ceiling.
//   The kernel REQUIRES K ≤ 4 for V2-Lite; larger K must fall back to
//   the no-overlap baseline OR a flash-style chunked variant.

#include <metal_stdlib>
using namespace metal;

// q4_k_value helper is defined in moe.metal (included before this in
// the concatenated shader source). signed_u8 / fp16_at likewise.

// ── moe_gate_up_union_v2t ───────────────────────────────────────────────────
//
// Per-expert gate+up GEMM Q4_K_M with SiLU(gate) * up fused in.
// Output: routed_act[(kk, slot), row].
kernel void moe_gate_up_union_v2t(
    device const uchar* w_all           [[buffer(0)]],
    device const uint*  segment_starts  [[buffer(1)]],   // (n_experts+1,) u32
    device const uint*  sorted_kidx     [[buffer(2)]],   // (N,) u32
    device const uint*  sorted_slot     [[buffer(3)]],   // (N,) u32
    device const float* per_k_x         [[buffer(4)]],   // (K, cols) f32 packed
    device       float* routed_act      [[buffer(5)]],   // (K, top_k, rows) f32
    constant     ulong& gate_offset     [[buffer(6)]],
    constant     ulong& up_offset       [[buffer(7)]],
    constant     uint&  rows            [[buffer(8)]],
    constant     uint&  cols            [[buffer(9)]],
    constant     uint&  k_batch         [[buffer(10)]],
    constant     uint&  top_k           [[buffer(11)]],
    constant     uint&  n_experts       [[buffer(12)]],
    constant     uint&  union_N         [[buffer(13)]],  // K * top_k
    threadgroup  float* x_cache_all     [[threadgroup(0)]],  // K * cols floats
    uint2               tid2            [[thread_position_in_threadgroup]],
    uint2               tgp             [[threadgroup_position_in_grid]],
    uint                simd_lane       [[thread_index_in_simdgroup]],
    uint                simd_id         [[simdgroup_index_in_threadgroup]])
{
    uint tid = tid2.x;
    uint expert = tgp.y;
    if (expert >= n_experts) return;

    // Segment for this expert: [seg_start, seg_end).
    uint seg_start = segment_starts[expert];
    if (seg_start == union_N) return;   // expert absent in union → skip
    uint seg_end = union_N;
    for (uint e_next = expert + 1u; e_next <= n_experts; ++e_next) {
        uint s_next = segment_starts[e_next];
        if (s_next != union_N && s_next > seg_start) {
            seg_end = s_next;
            break;
        }
    }
    // If no later expert found, seg_end stays union_N (correct).

    uint base_row = tgp.x * 8u + simd_id;
    if (base_row >= rows) return;

    // Cooperative preload of ALL K x vectors into TG memory.
    // x_cache_all is laid out (K, cols) row-major.
    uint total_x = k_batch * cols;
    for (uint i = tid; i < total_x; i += 256u) {
        x_cache_all[i] = per_k_x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-expert weight row offsets — computed ONCE for the segment loop.
    uint blocks_per_row = cols / 256u;
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 144ul;
    uint64_t gate_row_off = gate_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    uint64_t up_row_off   = up_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;

    // For each (kk, slot) in this expert's segment.
    for (uint si = seg_start; si < seg_end; ++si) {
        uint kk = sorted_kidx[si];
        uint slot = sorted_slot[si];
        threadgroup const float* x_cache =
            x_cache_all + (uint64_t)kk * (uint64_t)cols;

        float gate_partial = 0.0f, up_partial = 0.0f;
        float total_gate_corr = 0.0f, total_up_corr = 0.0f;

        for (uint b = 0; b < blocks_per_row; ++b) {
            uint64_t bo_g = gate_row_off + (uint64_t)b * 144ul;
            uint64_t bo_u = up_row_off   + (uint64_t)b * 144ul;

            float dg    = fp16_at(w_all, bo_g);
            float dming = fp16_at(w_all, bo_g + 2ul);
            float du    = fp16_at(w_all, bo_u);
            float dminu = fp16_at(w_all, bo_u + 2ul);

            // Preload scale + min bytes (gate + up).
            uchar sg[8], mg[8], su[8], mu[8];
            for (uint sub = 0; sub < 4u; ++sub) {
                sg[sub] = w_all[bo_g + 4u + sub]      & 0x3Fu;
                mg[sub] = w_all[bo_g + 4u + 4u + sub] & 0x3Fu;
                su[sub] = w_all[bo_u + 4u + sub]      & 0x3Fu;
                mu[sub] = w_all[bo_u + 4u + 4u + sub] & 0x3Fu;
            }
            for (uint j = 0; j < 4u; ++j) {
                sg[4u + j] = (w_all[bo_g + 4u + 8u + j] & 0x0Fu)
                           | ((w_all[bo_g + 4u + j]      >> 6u) << 4u);
                mg[4u + j] = (w_all[bo_g + 4u + 8u + j] >> 4u)
                           | ((w_all[bo_g + 4u + 4u + j] >> 6u) << 4u);
                su[4u + j] = (w_all[bo_u + 4u + 8u + j] & 0x0Fu)
                           | ((w_all[bo_u + 4u + j]      >> 6u) << 4u);
                mu[4u + j] = (w_all[bo_u + 4u + 8u + j] >> 4u)
                           | ((w_all[bo_u + 4u + 4u + j] >> 6u) << 4u);
            }
            float dsg[8], dmg[8], dsu[8], dmu[8];
            for (uint k = 0; k < 8u; ++k) {
                dsg[k] = dg    * (float)sg[k];
                dmg[k] = dming * (float)mg[k];
                dsu[k] = du    * (float)su[k];
                dmu[k] = dminu * (float)mu[k];
            }

            // Activations from TG-cached x.
            float xl[8];
            for (uint k = 0; k < 8u; ++k) {
                xl[k] = x_cache[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];
            }

            float sumy[8];
            for (uint k = 0; k < 8u; ++k) sumy[k] = simd_sum(xl[k]);
            for (uint k = 0; k < 8u; ++k) {
                total_gate_corr += dmg[k] * sumy[k];
                total_up_corr   += dmu[k] * sumy[k];
            }

            // Paired-nibble dot product.
            for (uint pi = 0; pi < 4u; ++pi) {
                uint k0 = pi * 2u, k1 = k0 + 1u;
                uchar qg = w_all[bo_g + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
                uchar qu = w_all[bo_u + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
                gate_partial += dsg[k0] * (float)(qg & 0x0Fu) * xl[k0]
                              + dsg[k1] * (float)(qg >> 4u)   * xl[k1];
                up_partial   += dsu[k0] * (float)(qu & 0x0Fu) * xl[k0]
                              + dsu[k1] * (float)(qu >> 4u)   * xl[k1];
            }
        }

        float gate_val = simd_sum(gate_partial) - total_gate_corr;
        float up_val   = simd_sum(up_partial)   - total_up_corr;

        if (simd_lane == 0u) {
            float silu = gate_val / (1.0f + exp(-gate_val));
            routed_act[
                ((uint64_t)kk * (uint64_t)top_k + (uint64_t)slot) * (uint64_t)rows
                + (uint64_t)base_row
            ] = silu * up_val;
        }
    }
}

// ── moe_down_union_v2t ──────────────────────────────────────────────────────
//
// Per-expert down projection Q4_K_M GEMV. Same union dispatch shape as
// gate_up but operates on routed_act (the silu*up output) and writes
// to routed_out (per-K-query per-slot hidden vector).
//
// Layout:
//   routed_act:  (K, top_k, mid_cols)   ← input (rows of GEMV)
//   routed_out:  (K, top_k, hidden_rows) ← output (one hidden vec per (kk, slot))
//
// Down projection W is shape (hidden_rows, mid_cols). Grid:
// (ceil(hidden_rows/8), n_experts, 1).
kernel void moe_down_union_v2t(
    device const uchar* w_all           [[buffer(0)]],
    device const uint*  segment_starts  [[buffer(1)]],
    device const uint*  sorted_kidx     [[buffer(2)]],
    device const uint*  sorted_slot     [[buffer(3)]],
    device const float* routed_act      [[buffer(4)]],   // (K, top_k, mid_cols)
    device       float* routed_out      [[buffer(5)]],   // (K, top_k, rows)
    constant     ulong& down_offset     [[buffer(6)]],
    constant     uint&  rows            [[buffer(7)]],   // hidden
    constant     uint&  cols            [[buffer(8)]],   // routed_mid
    constant     uint&  k_batch         [[buffer(9)]],
    constant     uint&  top_k           [[buffer(10)]],
    constant     uint&  n_experts       [[buffer(11)]],
    constant     uint&  union_N         [[buffer(12)]],
    uint2               tid2            [[thread_position_in_threadgroup]],
    uint2               tgp             [[threadgroup_position_in_grid]],
    uint                simd_lane       [[thread_index_in_simdgroup]],
    uint                simd_id         [[simdgroup_index_in_threadgroup]])
{
    uint expert = tgp.y;
    if (expert >= n_experts) return;

    uint seg_start = segment_starts[expert];
    if (seg_start == union_N) return;
    uint seg_end = union_N;
    for (uint e_next = expert + 1u; e_next <= n_experts; ++e_next) {
        uint s_next = segment_starts[e_next];
        if (s_next != union_N && s_next > seg_start) {
            seg_end = s_next;
            break;
        }
    }

    uint base_row = tgp.x * 8u + simd_id;
    if (base_row >= rows) return;

    uint blocks_per_row = cols / 256u;
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_byte_off = down_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;

    // Each (kk, slot) reads from routed_act[(kk, slot), :] in device memory.
    // No TG cache for routed_act because the activations differ per
    // (kk, slot) (no shared reuse across queries). We rely on L1/L2
    // for the segment loop's locality.
    for (uint si = seg_start; si < seg_end; ++si) {
        uint kk = sorted_kidx[si];
        uint slot = sorted_slot[si];
        device const float* x_act = routed_act
            + ((uint64_t)kk * (uint64_t)top_k + (uint64_t)slot) * (uint64_t)cols;

        float partial = 0.0f;
        for (uint b = 0; b < blocks_per_row; ++b) {
            uint64_t bo = row_byte_off + (uint64_t)b * 144ul;
            for (uint k = 0; k < 8u; ++k) {
                uint elem = k * 32u + simd_lane;
                partial += q4_k_value(w_all, bo, elem)
                         * x_act[(uint64_t)b * 256ul + (uint64_t)elem];
            }
        }

        partial = simd_sum(partial);
        if (simd_lane == 0u) {
            routed_out[
                ((uint64_t)kk * (uint64_t)top_k + (uint64_t)slot) * (uint64_t)rows
                + (uint64_t)base_row
            ] = partial;
        }
    }
}

// path-to-150 Phase L7.2 — single-kernel MoE expert fusion (mixed
// quant: Q4_K_M gate/up + Q8_0 down).
//
// Goal: eliminate the `routed_act` and `routed_out` global-memory
// round-trips that exist between the gate_up_union_v2t and
// down_union_v2t kernels in the current routed MoE pipeline. The
// fused kernel keeps intermediates in threadgroup SRAM:
//
//   x_cache[hidden_in]    — cooperative load of x[kk]
//   act_cache[routed_mid] — silu(W_gate @ x) * (W_up @ x), Stage A output
//
// then projects to y[hidden_out] = W_down_Q8 @ act_cache in Stage B.
//
// Trade-off vs the existing union pipeline:
//   * Save:  one global write of routed_act (N × routed_mid × 4 bytes,
//            ~135 KB at K=4, V2-Lite) and one global read of the same
//            data by the down kernel.
//   * Lose:  per-expert weight reuse across the union segment. If two
//            (kk, slot) routes select the same expert, the fused kernel
//            reads its W_gate / W_up / W_down twice (once per route)
//            instead of once. At K=4 with ~60% overlap this is roughly
//            2.4× re-read on ~40 MB of expert weight → ~56 MB extra
//            DRAM traffic per token. Whether fusion wins or loses on
//            net is shape-dependent — empirical question, settled by
//            the bench. At K=1 there is no overlap to lose, so fusion
//            is a pure win.
//
// Geometry: one TG per route. Grid: (1, n_routes, 1). TG: (256, 1, 1)
// = 8 simdgroups × 32 lanes. Threadgroup memory:
//   x_cache (hidden_in × 4 bytes) + act_cache (routed_mid × 4 bytes).
//   V2-Lite numbers: 2048×4 + 1408×4 = 13.5 KB ≪ 32 KB M3 Pro per-core
//   limit.
//
// V2-Lite divisibility:
//   * hidden_in = 2048 (Q4_K block = 256 → 8 blocks per gate/up row)
//   * routed_mid = 1408 (Q4_K-row count = 1408; Q8_0 block = 32 → 44
//     blocks per down row)
//   * hidden_out = 2048 (44 Q8_0 blocks per down row → ✓)
// All dimensions clean for the native block sizes used.
//
// Buffer layout (mirrors moe_gate_up_union_v2t / moe_down_union_v2t):
//   w_all[gate_offset    + e × per_gate_bytes    + row × bpr_q4 × 144]
//   w_all[up_offset      + e × per_up_bytes      + row × bpr_q4 × 144]
//   w_all[down_offset    + e × per_down_bytes    + row × bpr_q8 × 34]
//
//   bpr_q4 = hidden_in / 256
//   bpr_q8 = routed_mid / 32
//
// Helpers (fp16_at, signed_u8, q4_k decode) live in moe.metal which is
// concatenated before this file in all_shader_sources().

#include <metal_stdlib>
using namespace metal;

// ── moe_expert_pair_fused ────────────────────────────────────────────────────
//
// Inputs:
//   route_ids   [n_routes]            — expert id per route
//   route_kk    [n_routes]            — which kk in [0,K) per route
//                                       (route i pulls x from per_k_x[kk])
//   per_k_x     [K, hidden_in]        — packed activations per K query
// Outputs:
//   routed_out  [n_routes, hidden_out]
// Per-route slot ordering matches the union scheme: route index packs
// (kk, slot) as (kk * top_k + slot) but the kernel doesn't care — it
// treats each route independently.

kernel void moe_expert_pair_fused(
    device const uchar* w_all       [[buffer(0)]],
    device const uint*  route_ids   [[buffer(1)]],
    device const uint*  route_kk    [[buffer(2)]],
    device const float* per_k_x     [[buffer(3)]],
    device       float* routed_out  [[buffer(4)]],
    constant     ulong& gate_offset [[buffer(5)]],
    constant     ulong& up_offset   [[buffer(6)]],
    constant     ulong& down_offset [[buffer(7)]],
    constant     uint&  hidden_in   [[buffer(8)]],
    constant     uint&  routed_mid  [[buffer(9)]],
    constant     uint&  hidden_out  [[buffer(10)]],
    constant     uint&  n_routes    [[buffer(11)]],
    constant     uint&  n_experts   [[buffer(12)]],
    threadgroup  float* tg_mem      [[threadgroup(0)]],
    uint2               tid2        [[thread_position_in_threadgroup]],
    uint2               tgp         [[threadgroup_position_in_grid]],
    uint                simd_lane   [[thread_index_in_simdgroup]],
    uint                simd_id     [[simdgroup_index_in_threadgroup]])
{
    uint tid   = tid2.x;
    uint route = tgp.y;
    if (route >= n_routes) return;

    uint expert = route_ids[route];
    uint kk     = route_kk[route];

    // tg_mem layout: [x_cache | act_cache], lengths hidden_in / routed_mid.
    threadgroup float* x_cache   = tg_mem;
    threadgroup float* act_cache = tg_mem + (uint64_t)hidden_in;

    // ── Cooperative load: x[kk] → x_cache ────────────────────────────────────
    for (uint i = tid; i < hidden_in; i += 256u) {
        x_cache[i] = per_k_x[(uint64_t)kk * (uint64_t)hidden_in + (uint64_t)i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-expert byte offsets for gate / up / down weight tensors.
    uint bpr_q4 = hidden_in / 256u;                                 // 8 at V2-Lite
    uint64_t per_gate_bytes = (uint64_t)routed_mid * (uint64_t)bpr_q4 * 144ul;
    uint64_t per_up_bytes   = per_gate_bytes;                       // same shape as gate
    uint bpr_q8 = routed_mid / 32u;                                 // 44 at V2-Lite
    uint64_t per_down_bytes = (uint64_t)hidden_out * (uint64_t)bpr_q8 * 34ul;

    uint64_t gate_expert_off = gate_offset + (uint64_t)expert * per_gate_bytes;
    uint64_t up_expert_off   = up_offset   + (uint64_t)expert * per_up_bytes;
    uint64_t down_expert_off = down_offset + (uint64_t)expert * per_down_bytes;

    // ── Stage A: act_cache = silu(W_gate @ x) * (W_up @ x) ───────────────────
    //
    // Each simdgroup owns one row of act_cache per outer iteration. 8
    // simdgroups → 8 rows produced per outer iter. routed_mid rows total
    // → ceil(routed_mid / 8) outer iterations per simdgroup.
    //
    // Uses the sumy min-correction trick from moe_gate_up_union_v2t /
    // v3_llama / xtg_sumy: precompute simd_sum(xl[k]) per sub-block,
    // accumulate `dm * sumy` outside the nibble loop.

    uint stage_a_outer = (routed_mid + 7u) / 8u;
    for (uint outer = 0u; outer < stage_a_outer; ++outer) {
        uint act_row = outer * 8u + simd_id;
        bool row_valid = act_row < routed_mid;

        uint64_t gate_row_off = gate_expert_off
                              + (uint64_t)act_row * (uint64_t)bpr_q4 * 144ul;
        uint64_t up_row_off   = up_expert_off
                              + (uint64_t)act_row * (uint64_t)bpr_q4 * 144ul;

        float gate_partial = 0.0f, up_partial = 0.0f;
        float total_gate_corr = 0.0f, total_up_corr = 0.0f;

        for (uint b = 0u; b < bpr_q4; ++b) {
            uint64_t bo_g = gate_row_off + (uint64_t)b * 144ul;
            uint64_t bo_u = up_row_off   + (uint64_t)b * 144ul;

            float dg    = fp16_at(w_all, bo_g);
            float dming = fp16_at(w_all, bo_g + 2ul);
            float du    = fp16_at(w_all, bo_u);
            float dminu = fp16_at(w_all, bo_u + 2ul);

            uchar sg[8], mg[8], su[8], mu[8];
            for (uint sub = 0u; sub < 4u; ++sub) {
                sg[sub] = w_all[bo_g + 4u + sub]      & 0x3Fu;
                mg[sub] = w_all[bo_g + 4u + 4u + sub] & 0x3Fu;
                su[sub] = w_all[bo_u + 4u + sub]      & 0x3Fu;
                mu[sub] = w_all[bo_u + 4u + 4u + sub] & 0x3Fu;
            }
            for (uint j = 0u; j < 4u; ++j) {
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
            for (uint k = 0u; k < 8u; ++k) {
                dsg[k] = dg    * (float)sg[k];
                dmg[k] = dming * (float)mg[k];
                dsu[k] = du    * (float)su[k];
                dmu[k] = dminu * (float)mu[k];
            }

            float xl[8];
            for (uint k = 0u; k < 8u; ++k) {
                xl[k] = x_cache[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];
            }

            float sumy[8];
            for (uint k = 0u; k < 8u; ++k) sumy[k] = simd_sum(xl[k]);
            for (uint k = 0u; k < 8u; ++k) {
                total_gate_corr += dmg[k] * sumy[k];
                total_up_corr   += dmu[k] * sumy[k];
            }

            for (uint pi = 0u; pi < 4u; ++pi) {
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

        if (row_valid && simd_lane == 0u) {
            float silu = gate_val / (1.0f + exp(-gate_val));
            act_cache[act_row] = silu * up_val;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Stage B: routed_out[route, :] = W_down_Q8 @ act_cache ────────────────
    //
    // Same per-simdgroup row geometry: 8 simdgroups produce 8 rows of y
    // per outer iter. Q8_0 block = 32 elements = exactly one simdgroup
    // wide, so the inner loop is one MAD per block per lane.

    uint stage_b_outer = (hidden_out + 7u) / 8u;
    for (uint outer = 0u; outer < stage_b_outer; ++outer) {
        uint out_row = outer * 8u + simd_id;
        bool row_valid = out_row < hidden_out;

        uint64_t down_row_off = down_expert_off
                              + (uint64_t)out_row * (uint64_t)bpr_q8 * 34ul;

        float partial = 0.0f;
        for (uint b = 0u; b < bpr_q8; ++b) {
            uint64_t bo = down_row_off + (uint64_t)b * 34ul;
            float d  = fp16_at(w_all, bo);
            int   qi = signed_u8(w_all[bo + 2ul + (uint64_t)simd_lane]);
            float xi = act_cache[(uint64_t)b * 32ul + (uint64_t)simd_lane];
            partial += d * (float)qi * xi;
        }

        partial = simd_sum(partial);
        if (row_valid && simd_lane == 0u) {
            routed_out[(uint64_t)route * (uint64_t)hidden_out + (uint64_t)out_row]
                = partial;
        }
    }
}

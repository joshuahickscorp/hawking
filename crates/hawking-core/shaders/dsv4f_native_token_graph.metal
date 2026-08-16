// Minimum complete DeepSeek-V4-Flash BOS token graph: compact top-6 worklist.
//
// These kernels never walk the dense 256-expert table. A device-resident
// worklist of six (expert_id, slab_offset, route_weight) entries is the only
// expert control the matvec/SwiGLU/combine path reads. Helpers are file-local
// so concatenation with moe.metal / matmul.metal cannot collide.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint DSV4F_WORKLIST_K = 6u;
constant constexpr uint DSV4F_FP4_BLOCK = 32u;
constant constexpr uint DSV4F_ACT_BLOCK = 128u;

struct Dsv4fWorklistEntry {
    uint expert_id;
    uint slab_slot;
    float route_weight;
    uint ready;
};

static_assert(sizeof(Dsv4fWorklistEntry) == 16, "Dsv4fWorklistEntry ABI drift");

// One selected expert's packed-FP4 projection. Host writes Metal gpuAddress
// values (Q80 device-expert-table transfer). The kernel never walks 256
// experts and never assumes a compact host-packed slab.
struct Dsv4fExpertRef {
    device const uchar* packed_weights;
    device const uchar* weight_scales;
};

static_assert(sizeof(Dsv4fExpertRef) == 16, "Dsv4fExpertRef ABI drift");

static inline float dsv4f_tg_bf16_value(ushort bits)
{
    return gk_bf16_value(bits);
}

static inline ushort dsv4f_tg_bf16_encode_rne(float value)
{
    return gk_bf16_encode_rne(value);
}

static inline float dsv4f_tg_e4m3fn_value(uchar bits)
{
    return gk_e4m3fn_value(bits);
}

static inline float dsv4f_tg_e8m0fnu_value(uchar bits)
{
    return gk_e8m0fnu_value(bits);
}

static inline float dsv4f_tg_e2m1fn_value(uchar packed, bool high_nibble)
{
    return gk_e2m1fn_value(packed, high_nibble);
}

static inline float dsv4f_tg_silu(float value)
{
    return gk_silu_dsv4f(value);
}

// Sort the six device route IDs into execution order (ascending expert id,
// then original top-slot) and emit the compact worklist. One thread.
#pragma clang fp contract(off)
kernel void dsv4f_pack_worklist(
    device const uint* selected_ids      [[buffer(0)]],
    device const float* selected_weights [[buffer(1)]],
    device       Dsv4fWorklistEntry* worklist [[buffer(2)]],
    device       uint* valid             [[buffer(3)]],
    constant uint& top_k                 [[buffer(4)]],
    constant uint& expert_count          [[buffer(5)]],
    uint index                           [[thread_position_in_grid]])
{
    if (index != 0u) return;
    valid[0] = 0u;
    if (top_k != DSV4F_WORKLIST_K || expert_count == 0u) {
        valid[0] = 2u;
        return;
    }
    uint ids[DSV4F_WORKLIST_K];
    float weights[DSV4F_WORKLIST_K];
    uint slots[DSV4F_WORKLIST_K];
    for (uint slot = 0u; slot < DSV4F_WORKLIST_K; ++slot) {
        const uint id = selected_ids[slot];
        if (id >= expert_count || !isfinite(selected_weights[slot])) {
            valid[0] = 16u + slot;
            return;
        }
        ids[slot] = id;
        weights[slot] = selected_weights[slot];
        slots[slot] = slot;
    }
    for (uint a = 0u; a < DSV4F_WORKLIST_K; ++a) {
        uint best = a;
        for (uint b = a + 1u; b < DSV4F_WORKLIST_K; ++b) {
            if (ids[b] < ids[best] || (ids[b] == ids[best] && slots[b] < slots[best])) {
                best = b;
            }
        }
        if (best != a) {
            const uint tid = ids[a];
            ids[a] = ids[best];
            ids[best] = tid;
            const float tw = weights[a];
            weights[a] = weights[best];
            weights[best] = tw;
            const uint ts = slots[a];
            slots[a] = slots[best];
            slots[best] = ts;
        }
    }
    for (uint i = 1u; i < DSV4F_WORKLIST_K; ++i) {
        if (ids[i] == ids[i - 1u]) {
            valid[0] = 32u + i;
            return;
        }
    }
    for (uint slot = 0u; slot < DSV4F_WORKLIST_K; ++slot) {
        worklist[slot].expert_id = ids[slot];
        worklist[slot].slab_slot = slot;
        worklist[slot].route_weight = weights[slot];
        worklist[slot].ready = 1u;
    }
    valid[0] = 1u;
}
#pragma clang fp contract(on)

// One dispatch covers all six selected experts. Thread owns (slot, row).
// Device worklist supplies slab_slot; refs[slot] is that expert's gpuAddress
// pair. Indirect buffers must also be declared via useResources.
#pragma clang fp contract(off)
kernel void dsv4f_worklist_fp4_matvec(
    device const Dsv4fWorklistEntry* worklist [[buffer(0)]],
    device const Dsv4fExpertRef* refs         [[buffer(1)]],
    device const uchar* quantized             [[buffer(2)]],
    device const uchar* act_scales            [[buffer(3)]],
    device       float* output                [[buffer(4)]],
    constant uint& rows                        [[buffer(5)]],
    constant uint& packed_cols                 [[buffer(6)]],
    constant uint& scale_cols                  [[buffer(7)]],
    constant uint& top_k                       [[buffer(8)]],
    constant uint& act_is_per_slot             [[buffer(9)]],
    uint tid                                   [[thread_position_in_grid]])
{
    if (top_k != DSV4F_WORKLIST_K || rows == 0u || packed_cols == 0u
        || packed_cols * 2u != scale_cols * DSV4F_FP4_BLOCK) {
        return;
    }
    const uint slot = tid / rows;
    const uint row = tid % rows;
    if (slot >= DSV4F_WORKLIST_K) return;
    const Dsv4fWorklistEntry entry = worklist[slot];
    if (entry.ready != 1u || entry.slab_slot >= DSV4F_WORKLIST_K) return;
    const Dsv4fExpertRef ref = refs[entry.slab_slot];
    if (ref.packed_weights == nullptr || ref.weight_scales == nullptr) return;

    const uint logical_k = packed_cols * 2u;
    const ulong act_base = act_is_per_slot != 0u
        ? (ulong)entry.slab_slot * (ulong)logical_k
        : 0ul;
    const ulong act_scale_base = act_is_per_slot != 0u
        ? (ulong)entry.slab_slot * (ulong)(logical_k / DSV4F_ACT_BLOCK)
        : 0ul;
    const ulong weight_base = (ulong)row * (ulong)packed_cols;
    const ulong scale_base = (ulong)row * (ulong)scale_cols;
    float row_accumulator = 0.0f;
    for (uint block = 0u; block < scale_cols; ++block) {
        float block_accumulator = 0.0f;
        const uint start = block * DSV4F_FP4_BLOCK;
        for (uint offset = 0u; offset < DSV4F_FP4_BLOCK; ++offset) {
            const uint col = start + offset;
            const uchar packed = ref.packed_weights[weight_base + (ulong)(col >> 1u)];
            const float activation = dsv4f_tg_e4m3fn_value(quantized[act_base + (ulong)col]);
            const float weight = dsv4f_tg_e2m1fn_value(packed, (col & 1u) != 0u);
            block_accumulator = block_accumulator + activation * weight;
        }
        const float activation_scale = dsv4f_tg_e8m0fnu_value(
            act_scales[act_scale_base + (ulong)(block / (DSV4F_ACT_BLOCK / DSV4F_FP4_BLOCK))]);
        const float weight_scale = dsv4f_tg_e8m0fnu_value(
            ref.weight_scales[scale_base + (ulong)block]);
        row_accumulator = row_accumulator
            + block_accumulator * (activation_scale * weight_scale);
    }
    output[(ulong)slot * (ulong)rows + (ulong)row] = row_accumulator;
}
#pragma clang fp contract(on)

// One simdgroup per (slot, row). Same association as one worklist slot of
// dsv4f_worklist_fp4_matvec after simd_sum. Grid: top_k * ceil(rows/8) * 256.
#pragma clang fp contract(off)
kernel void dsv4f_worklist_fp4_matvec_simd(
    device const Dsv4fWorklistEntry* worklist [[buffer(0)]],
    device const Dsv4fExpertRef* refs         [[buffer(1)]],
    device const uchar* quantized             [[buffer(2)]],
    device const uchar* act_scales            [[buffer(3)]],
    device       float* output                [[buffer(4)]],
    constant uint& rows                        [[buffer(5)]],
    constant uint& packed_cols                 [[buffer(6)]],
    constant uint& scale_cols                  [[buffer(7)]],
    constant uint& top_k                       [[buffer(8)]],
    constant uint& act_is_per_slot             [[buffer(9)]],
    uint group_id                              [[threadgroup_position_in_grid]],
    uint simd_lane                             [[thread_index_in_simdgroup]],
    uint simd_id                               [[simdgroup_index_in_threadgroup]])
{
    if (top_k != DSV4F_WORKLIST_K || rows == 0u || packed_cols == 0u
        || packed_cols * 2u != scale_cols * DSV4F_FP4_BLOCK) {
        return;
    }
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint groups_per_slot = (rows + kSimdgroupsPerThreadgroup - 1u)
        / kSimdgroupsPerThreadgroup;
    if (groups_per_slot == 0u) return;
    const uint slot = group_id / groups_per_slot;
    const uint row = (group_id % groups_per_slot) * kSimdgroupsPerThreadgroup + simd_id;
    if (slot >= DSV4F_WORKLIST_K || row >= rows) return;
    const Dsv4fWorklistEntry entry = worklist[slot];
    if (entry.ready != 1u || entry.slab_slot >= DSV4F_WORKLIST_K) return;
    const Dsv4fExpertRef ref = refs[entry.slab_slot];
    if (ref.packed_weights == nullptr || ref.weight_scales == nullptr) return;

    const uint logical_k = packed_cols * 2u;
    const ulong act_base = act_is_per_slot != 0u
        ? (ulong)entry.slab_slot * (ulong)logical_k
        : 0ul;
    const ulong act_scale_base = act_is_per_slot != 0u
        ? (ulong)entry.slab_slot * (ulong)(logical_k / DSV4F_ACT_BLOCK)
        : 0ul;
    const ulong weight_base = (ulong)row * (ulong)packed_cols;
    const ulong scale_base = (ulong)row * (ulong)scale_cols;
    float row_accumulator = 0.0f;
    for (uint block = 0u; block < scale_cols; ++block) {
        const uint col = block * DSV4F_FP4_BLOCK + simd_lane;
        const uchar packed = ref.packed_weights[weight_base + (ulong)(col >> 1u)];
        const float activation = dsv4f_tg_e4m3fn_value(quantized[act_base + (ulong)col]);
        const float weight = dsv4f_tg_e2m1fn_value(packed, (col & 1u) != 0u);
        float block_accumulator = activation * weight;
        block_accumulator = simd_sum(block_accumulator);
        if (simd_lane == 0u) {
            const float activation_scale = dsv4f_tg_e8m0fnu_value(
                act_scales[act_scale_base + (ulong)(block / (DSV4F_ACT_BLOCK / DSV4F_FP4_BLOCK))]);
            const float weight_scale = dsv4f_tg_e8m0fnu_value(
                ref.weight_scales[scale_base + (ulong)block]);
            row_accumulator = row_accumulator
                + block_accumulator * (activation_scale * weight_scale);
        }
    }
    if (simd_lane == 0u) {
        output[(ulong)slot * (ulong)rows + (ulong)row] = row_accumulator;
    }
}
#pragma clang fp contract(on)

kernel void dsv4f_worklist_swiglu(
    device const Dsv4fWorklistEntry* worklist [[buffer(0)]],
    device const ushort* gate_bf16            [[buffer(1)]],
    device const ushort* up_bf16              [[buffer(2)]],
    device       ushort* output_bf16          [[buffer(3)]],
    constant uint& width                       [[buffer(4)]],
    constant uint& top_k                       [[buffer(5)]],
    uint tid                                   [[thread_position_in_grid]])
{
    if (top_k != DSV4F_WORKLIST_K || width == 0u) return;
    const uint slot = tid / width;
    const uint index = tid % width;
    if (slot >= DSV4F_WORKLIST_K) return;
    const Dsv4fWorklistEntry entry = worklist[slot];
    if (entry.ready != 1u) return;
    const ulong off = (ulong)slot * (ulong)width + (ulong)index;
    const float gate = min(dsv4f_tg_bf16_value(gate_bf16[off]), 10.0f);
    const float up = clamp(dsv4f_tg_bf16_value(up_bf16[off]), -10.0f, 10.0f);
    output_bf16[off] = dsv4f_tg_bf16_encode_rne(
        dsv4f_tg_silu(gate) * up * entry.route_weight);
}

// Source-order combine: y = 0; y += expert_i (exec order); y += shared.
#pragma clang fp contract(off)
kernel void dsv4f_worklist_combine(
    device const ushort* routed_bf16 [[buffer(0)]],
    device const ushort* shared_bf16 [[buffer(1)]],
    device       ushort* output_bf16 [[buffer(2)]],
    constant uint& hidden             [[buffer(3)]],
    constant uint& top_k              [[buffer(4)]],
    uint index                        [[thread_position_in_grid]])
{
    if (index >= hidden || top_k != DSV4F_WORKLIST_K) return;
    float value = 0.0f;
    for (uint slot = 0u; slot < DSV4F_WORKLIST_K; ++slot) {
        value = value + dsv4f_tg_bf16_value(
            routed_bf16[(ulong)slot * (ulong)hidden + (ulong)index]);
    }
    value = value + dsv4f_tg_bf16_value(shared_bf16[index]);
    output_bf16[index] = dsv4f_tg_bf16_encode_rne(value);
}
#pragma clang fp contract(on)

// Isolated FP4 matvec against the current split packed+scale pair.
// Same association as one slot of dsv4f_worklist_fp4_matvec.
// Grid: (rows, 1, 1), TG 256.
#pragma clang fp contract(off)
kernel void dsv4f_fp4_matvec_split(
    device const uchar* packed_weights [[buffer(0)]],
    device const uchar* weight_scales  [[buffer(1)]],
    device const uchar* quantized      [[buffer(2)]],
    device const uchar* act_scales     [[buffer(3)]],
    device       float* output         [[buffer(4)]],
    constant uint& rows                 [[buffer(5)]],
    constant uint& packed_cols          [[buffer(6)]],
    constant uint& scale_cols           [[buffer(7)]],
    uint row                            [[thread_position_in_grid]])
{
    if (rows == 0u || packed_cols == 0u
        || packed_cols * 2u != scale_cols * DSV4F_FP4_BLOCK) {
        return;
    }
    if (row >= rows) return;
    const ulong weight_base = (ulong)row * (ulong)packed_cols;
    const ulong scale_base = (ulong)row * (ulong)scale_cols;
    float row_accumulator = 0.0f;
    for (uint block = 0u; block < scale_cols; ++block) {
        float block_accumulator = 0.0f;
        const uint start = block * DSV4F_FP4_BLOCK;
        for (uint offset = 0u; offset < DSV4F_FP4_BLOCK; ++offset) {
            const uint col = start + offset;
            const uchar packed = packed_weights[weight_base + (ulong)(col >> 1u)];
            const float activation = dsv4f_tg_e4m3fn_value(quantized[col]);
            const float weight = dsv4f_tg_e2m1fn_value(packed, (col & 1u) != 0u);
            block_accumulator = block_accumulator + activation * weight;
        }
        const float activation_scale = dsv4f_tg_e8m0fnu_value(
            act_scales[block / (DSV4F_ACT_BLOCK / DSV4F_FP4_BLOCK)]);
        const float weight_scale = dsv4f_tg_e8m0fnu_value(
            weight_scales[scale_base + (ulong)block]);
        row_accumulator = row_accumulator
            + block_accumulator * (activation_scale * weight_scale);
    }
    output[row] = row_accumulator;
}
#pragma clang fp contract(on)

// One simdgroup per row. Each lane owns one of the 32 weights in an FP4
// block. Grid: (ceil(rows / 8) * 256, 1, 1), TG (256, 1, 1).
#pragma clang fp contract(off)
kernel void dsv4f_fp4_matvec_split_simd(
    device const uchar* packed_weights [[buffer(0)]],
    device const uchar* weight_scales  [[buffer(1)]],
    device const uchar* quantized      [[buffer(2)]],
    device const uchar* act_scales     [[buffer(3)]],
    device       float* output         [[buffer(4)]],
    constant uint& rows                 [[buffer(5)]],
    constant uint& packed_cols          [[buffer(6)]],
    constant uint& scale_cols           [[buffer(7)]],
    uint group_id                       [[threadgroup_position_in_grid]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]])
{
    if (rows == 0u || packed_cols == 0u
        || packed_cols * 2u != scale_cols * DSV4F_FP4_BLOCK) {
        return;
    }
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows) return;
    const ulong weight_base = (ulong)row * (ulong)packed_cols;
    const ulong scale_base = (ulong)row * (ulong)scale_cols;
    float row_accumulator = 0.0f;
    for (uint block = 0u; block < scale_cols; ++block) {
        const uint col = block * DSV4F_FP4_BLOCK + simd_lane;
        const uchar packed = packed_weights[weight_base + (ulong)(col >> 1u)];
        const float activation = dsv4f_tg_e4m3fn_value(quantized[col]);
        const float weight = dsv4f_tg_e2m1fn_value(packed, (col & 1u) != 0u);
        float block_accumulator = activation * weight;
        block_accumulator = simd_sum(block_accumulator);
        if (simd_lane == 0u) {
            const float activation_scale = dsv4f_tg_e8m0fnu_value(
                act_scales[block / (DSV4F_ACT_BLOCK / DSV4F_FP4_BLOCK)]);
            const float weight_scale = dsv4f_tg_e8m0fnu_value(
                weight_scales[scale_base + (ulong)block]);
            row_accumulator = row_accumulator
                + block_accumulator * (activation_scale * weight_scale);
        }
    }
    if (simd_lane == 0u) output[row] = row_accumulator;
}
#pragma clang fp contract(on)

// 4 rows / simdgroup, 32 rows / TG. Activations are shared across the four
// rows. Grid: (ceil(rows / 32) * 256, 1, 1), TG (256, 1, 1).
#pragma clang fp contract(off)
kernel void dsv4f_fp4_matvec_split_simd_r4(
    device const uchar* packed_weights [[buffer(0)]],
    device const uchar* weight_scales  [[buffer(1)]],
    device const uchar* quantized      [[buffer(2)]],
    device const uchar* act_scales     [[buffer(3)]],
    device       float* output         [[buffer(4)]],
    constant uint& rows                 [[buffer(5)]],
    constant uint& packed_cols          [[buffer(6)]],
    constant uint& scale_cols           [[buffer(7)]],
    uint group_id                       [[threadgroup_position_in_grid]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]])
{
    if (rows == 0u || packed_cols == 0u
        || packed_cols * 2u != scale_cols * DSV4F_FP4_BLOCK) {
        return;
    }
    constexpr uint R = 4u;
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kRowsPerTg = kSimdgroupsPerThreadgroup * R;
    const uint row0 = group_id * kRowsPerTg + simd_id * R;
    if (row0 >= rows) return;
    const uint row1 = row0 + 1u;
    const uint row2 = row0 + 2u;
    const uint row3 = row0 + 3u;
    const bool has1 = row1 < rows;
    const bool has2 = row2 < rows;
    const bool has3 = row3 < rows;
    const uint r1 = has1 ? row1 : row0;
    const uint r2 = has2 ? row2 : row0;
    const uint r3 = has3 ? row3 : row0;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint block = 0u; block < scale_cols; ++block) {
        const uint col = block * DSV4F_FP4_BLOCK + simd_lane;
        const float activation = dsv4f_tg_e4m3fn_value(quantized[col]);
        const float activation_scale = dsv4f_tg_e8m0fnu_value(
            act_scales[block / (DSV4F_ACT_BLOCK / DSV4F_FP4_BLOCK)]);
        const uchar p0 = packed_weights[(ulong)row0 * (ulong)packed_cols + (ulong)(col >> 1u)];
        const uchar p1 = packed_weights[(ulong)r1 * (ulong)packed_cols + (ulong)(col >> 1u)];
        const uchar p2 = packed_weights[(ulong)r2 * (ulong)packed_cols + (ulong)(col >> 1u)];
        const uchar p3 = packed_weights[(ulong)r3 * (ulong)packed_cols + (ulong)(col >> 1u)];
        const bool hi = (col & 1u) != 0u;
        float b0 = activation * dsv4f_tg_e2m1fn_value(p0, hi);
        float b1 = activation * dsv4f_tg_e2m1fn_value(p1, hi);
        float b2 = activation * dsv4f_tg_e2m1fn_value(p2, hi);
        float b3 = activation * dsv4f_tg_e2m1fn_value(p3, hi);
        b0 = simd_sum(b0);
        b1 = simd_sum(b1);
        b2 = simd_sum(b2);
        b3 = simd_sum(b3);
        if (simd_lane == 0u) {
            const float s0 = dsv4f_tg_e8m0fnu_value(
                weight_scales[(ulong)row0 * (ulong)scale_cols + (ulong)block]);
            const float s1 = dsv4f_tg_e8m0fnu_value(
                weight_scales[(ulong)r1 * (ulong)scale_cols + (ulong)block]);
            const float s2 = dsv4f_tg_e8m0fnu_value(
                weight_scales[(ulong)r2 * (ulong)scale_cols + (ulong)block]);
            const float s3 = dsv4f_tg_e8m0fnu_value(
                weight_scales[(ulong)r3 * (ulong)scale_cols + (ulong)block]);
            a0 = a0 + b0 * (activation_scale * s0);
            a1 = a1 + b1 * (activation_scale * s1);
            a2 = a2 + b2 * (activation_scale * s2);
            a3 = a3 + b3 * (activation_scale * s3);
        }
    }
    if (simd_lane == 0u) {
        output[row0] = a0;
        if (has1) output[row1] = a1;
        if (has2) output[row2] = a2;
        if (has3) output[row3] = a3;
    }
}
#pragma clang fp contract(on)

// Isolated FP4 matvec against an execution-order interleaved organ:
// per 32-logical-weight block, [e8m0 scale | 16 packed bytes].
// Same association as dsv4f_worklist_fp4_matvec for one expert.
// Grid: (rows, 1, 1), TG 256. Not the default no-copy worklist path.
#pragma clang fp contract(off)
kernel void dsv4f_fp4_matvec_interleaved(
    device const uchar* records     [[buffer(0)]],
    device const uchar* quantized   [[buffer(1)]],
    device const uchar* act_scales  [[buffer(2)]],
    device       float* output      [[buffer(3)]],
    constant uint& rows              [[buffer(4)]],
    constant uint& packed_cols       [[buffer(5)]],
    constant uint& scale_cols        [[buffer(6)]],
    uint row                         [[thread_position_in_grid]])
{
    if (rows == 0u || packed_cols == 0u
        || packed_cols * 2u != scale_cols * DSV4F_FP4_BLOCK) {
        return;
    }
    if (row >= rows) return;
    const uint packed_per_block = DSV4F_FP4_BLOCK / 2u;
    const uint stride = 1u + packed_per_block;
    const ulong row_base = (ulong)row * (ulong)scale_cols * (ulong)stride;
    float row_accumulator = 0.0f;
    for (uint block = 0u; block < scale_cols; ++block) {
        const ulong rec = row_base + (ulong)block * (ulong)stride;
        const float weight_scale = dsv4f_tg_e8m0fnu_value(records[rec]);
        float block_accumulator = 0.0f;
        const uint start = block * DSV4F_FP4_BLOCK;
        for (uint offset = 0u; offset < DSV4F_FP4_BLOCK; ++offset) {
            const uint col = start + offset;
            const uchar packed = records[rec + 1u + (ulong)(offset >> 1u)];
            const float activation = dsv4f_tg_e4m3fn_value(quantized[col]);
            const float weight = dsv4f_tg_e2m1fn_value(packed, (offset & 1u) != 0u);
            block_accumulator = block_accumulator + activation * weight;
        }
        const float activation_scale = dsv4f_tg_e8m0fnu_value(
            act_scales[block / (DSV4F_ACT_BLOCK / DSV4F_FP4_BLOCK)]);
        row_accumulator = row_accumulator
            + block_accumulator * (activation_scale * weight_scale);
    }
    output[row] = row_accumulator;
}
#pragma clang fp contract(on)

// Generated-geometry packed matvec family. Isolated from the shipping
// library: the sweep example compiles this source itself.
//
// Packed bytes are decoded into registers and consumed in the same FMA.
// Nothing writes a dense (rows × cols) reconstruction.
//
// Function constants specialize launch geometry. The host generator
// enumerates the cartesian product and rejects invalid tuples before
// pipeline creation.

#include <metal_stdlib>
using namespace metal;

constant uint kTptg       [[function_constant(0)]];
constant uint kSgPerTg    [[function_constant(1)]];
constant uint kSplitSgs   [[function_constant(2)]];
constant uint kRowsPerSg  [[function_constant(3)]];
constant uint kVec        [[function_constant(4)]];
constant uint kUnroll     [[function_constant(5)]];
constant uint kReduce     [[function_constant(6)]];
constant uint kAccFp16    [[function_constant(7)]];
constant uint kStageX     [[function_constant(8)]];

static inline float geo_acc_add(float acc, float term)
{
    if (kAccFp16) {
        return float(half(acc) + half(term));
    }
    return acc + term;
}

static inline float geo_load_x(
    device const float* input,
    threadgroup float* x_tg,
    uint col)
{
    if (kStageX) {
        return x_tg[col];
    }
    return input[col];
}

static inline float geo_binary_dot_bits(
    uint bits,
    uint nbits,
    float scale,
    device const float* input,
    threadgroup float* x_tg,
    uint col)
{
    float sum = 0.0f;
    for (uint b = 0u; b < nbits; ++b) {
        const bool pos = ((bits >> b) & 1u) != 0u;
        sum += (pos ? scale : -scale) * geo_load_x(input, x_tg, col + b);
    }
    return sum;
}

static inline float geo_q4_dot_bytes(
    uint packed,
    uint nbytes,
    float scale,
    device const float* input,
    threadgroup float* x_tg,
    uint col)
{
    float sum = 0.0f;
    for (uint i = 0u; i < nbytes; ++i) {
        const uint byte = (packed >> (8u * i)) & 0xffu;
        const int q0 = int(byte & 0x0fu) - 8;
        const int q1 = int(byte >> 4u) - 8;
        sum += float(q0) * scale * geo_load_x(input, x_tg, col + 2u * i);
        sum += float(q1) * scale * geo_load_x(input, x_tg, col + 2u * i + 1u);
    }
    return sum;
}

// Streaming control: each thread issues `iters` float4 loads with a
// sequential stride. Used only to measure the same-box DRAM ceiling.
kernel void q80_geo_stream_control(
    device const uchar* data [[buffer(0)]],
    device float* out        [[buffer(1)]],
    constant uint& nbytes    [[buffer(2)]],
    constant uint& iters     [[buffer(3)]],
    uint tid                 [[thread_position_in_grid]],
    uint nthreads            [[threads_per_grid]])
{
    if (tid >= nthreads || nbytes < 16u || iters == 0u) {
        return;
    }
    const uint span = nbytes - 15u;
    const uint stride = nthreads * 16u;
    float acc = 0.0f;
    uint off = (tid * 16u) % span;
    for (uint i = 0u; i < iters; ++i) {
        const float4 v = *((device const float4*)(data + off));
        acc += v.x + v.y + v.z + v.w;
        off += stride;
        if (off + 16u > nbytes) {
            off = off % span;
        }
    }
    out[tid] = acc;
}

// Binary-group (Q80 gate_proj): 1-bit signs + fp16 scale / group.
// One threadgroup produces (kSgPerTg / kSplitSgs) * kRowsPerSg rows.
// kSplitSgs simdgroups cooperate on the same rows (split-K).
kernel void q80_geo_binary_matvec(
    device const uchar* signs     [[buffer(0)]],
    device const half* scales     [[buffer(1)]],
    device const float* input     [[buffer(2)]],
    device float* output          [[buffer(3)]],
    constant uint& rows           [[buffer(4)]],
    constant uint& cols           [[buffer(5)]],
    constant uint& group_size     [[buffer(6)]],
    constant uint& groups_per_row [[buffer(7)]],
    uint group_id                 [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[32];
    threadgroup float x_tg[2048];

    const bool serial_tpr = kSplitSgs == 0u;
    const uint teams = serial_tpr ? kTptg : (kSgPerTg / kSplitSgs);
    const uint rows_per_tg = teams * kRowsPerSg;
    const uint tpr = serial_tpr ? 1u : (32u * kSplitSgs);
    const uint team = serial_tpr ? lid : (simd_id / kSplitSgs);
    const uint split = serial_tpr ? 0u : (simd_id % kSplitSgs);
    const uint lane_in_row = serial_tpr ? 0u : (split * 32u + simd_lane);
    const uint row0 = group_id * rows_per_tg + team * kRowsPerSg;
    const uint weights_per_step = 8u * kVec;

    if (kStageX) {
        for (uint i = lid; i < cols && i < 2048u; i += kTptg) {
            x_tg[i] = input[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    const bool h0 = row0 < rows;
    const bool h1 = kRowsPerSg > 1u && row0 + 1u < rows;
    const bool h2 = kRowsPerSg > 2u && row0 + 2u < rows;
    const bool h3 = kRowsPerSg > 3u && row0 + 3u < rows;
    const bool h4 = kRowsPerSg > 4u && row0 + 4u < rows;
    const bool h5 = kRowsPerSg > 5u && row0 + 5u < rows;
    const bool h6 = kRowsPerSg > 6u && row0 + 6u < rows;
    const bool h7 = kRowsPerSg > 7u && row0 + 7u < rows;
    const bool live = h0;

    const uint step = tpr * weights_per_step;
    const uint tiled = (cols / step) * step;
    uint base = lane_in_row * weights_per_step;
    while (live && base + kUnroll * step <= tiled) {
        for (uint u = 0u; u < kUnroll; ++u) {
            const uint col = base + u * step;
            for (uint r = 0u; r < kRowsPerSg; ++r) {
                const uint row = row0 + r;
                if (row >= rows) {
                    continue;
                }
                const uint row_base = row * cols;
                const uint scale_base = row * groups_per_row;
                const float scale = float(scales[scale_base + col / group_size]);
                device const uchar* p = signs + ((row_base + col) >> 3u);
                uint bits = 0u;
                if (kVec == 1u) {
                    bits = uint(*p);
                } else if (kVec == 2u) {
                    bits = uint(*((device const ushort*)p));
                } else {
                    bits = *((device const uint*)p);
                }
                const float term = geo_binary_dot_bits(bits, weights_per_step, scale, input, x_tg, col);
                if (r == 0u) acc0 = geo_acc_add(acc0, term);
                else if (r == 1u) acc1 = geo_acc_add(acc1, term);
                else if (r == 2u) acc2 = geo_acc_add(acc2, term);
                else if (r == 3u) acc3 = geo_acc_add(acc3, term);
                else if (r == 4u) acc4 = geo_acc_add(acc4, term);
                else if (r == 5u) acc5 = geo_acc_add(acc5, term);
                else if (r == 6u) acc6 = geo_acc_add(acc6, term);
                else acc7 = geo_acc_add(acc7, term);
            }
        }
        base += kUnroll * step;
    }
    while (live && base < cols) {
        const uint col = base;
        const uint nbits = min(weights_per_step, cols - col);
        for (uint r = 0u; r < kRowsPerSg; ++r) {
            const uint row = row0 + r;
            if (row >= rows) {
                continue;
            }
            const uint row_base = row * cols;
            const uint scale_base = row * groups_per_row;
            float term = 0.0f;
            for (uint b = 0u; b < nbits; ++b) {
                const uint c = col + b;
                const float scale = float(scales[scale_base + c / group_size]);
                const uint flat = row_base + c;
                const uchar byte = signs[flat >> 3u];
                const bool pos = ((byte >> (flat & 7u)) & 1u) != 0u;
                term += (pos ? scale : -scale) * geo_load_x(input, x_tg, c);
            }
            if (r == 0u) acc0 = geo_acc_add(acc0, term);
            else if (r == 1u) acc1 = geo_acc_add(acc1, term);
            else if (r == 2u) acc2 = geo_acc_add(acc2, term);
            else if (r == 3u) acc3 = geo_acc_add(acc3, term);
            else if (r == 4u) acc4 = geo_acc_add(acc4, term);
            else if (r == 5u) acc5 = geo_acc_add(acc5, term);
            else if (r == 6u) acc6 = geo_acc_add(acc6, term);
            else acc7 = geo_acc_add(acc7, term);
        }
        base += step;
    }

    if (kReduce == 0u || tpr == 1u) {
        if (live && lane_in_row == 0u) {
            if (h0) output[row0] = acc0;
            if (h1) output[row0 + 1u] = acc1;
            if (h2) output[row0 + 2u] = acc2;
            if (h3) output[row0 + 3u] = acc3;
            if (h4) output[row0 + 4u] = acc4;
            if (h5) output[row0 + 5u] = acc5;
            if (h6) output[row0 + 6u] = acc6;
            if (h7) output[row0 + 7u] = acc7;
        }
        return;
    }

    if (kReduce == 1u) {
        acc0 = simd_sum(acc0);
        if (kRowsPerSg > 1u) acc1 = simd_sum(acc1);
        if (kRowsPerSg > 2u) acc2 = simd_sum(acc2);
        if (kRowsPerSg > 3u) acc3 = simd_sum(acc3);
        if (kRowsPerSg > 4u) acc4 = simd_sum(acc4);
        if (kRowsPerSg > 5u) acc5 = simd_sum(acc5);
        if (kRowsPerSg > 6u) acc6 = simd_sum(acc6);
        if (kRowsPerSg > 7u) acc7 = simd_sum(acc7);
    } else {
        for (uint off = 16u; off > 0u; off >>= 1u) {
            acc0 += simd_shuffle_down(acc0, off);
            if (kRowsPerSg > 1u) acc1 += simd_shuffle_down(acc1, off);
            if (kRowsPerSg > 2u) acc2 += simd_shuffle_down(acc2, off);
            if (kRowsPerSg > 3u) acc3 += simd_shuffle_down(acc3, off);
            if (kRowsPerSg > 4u) acc4 += simd_shuffle_down(acc4, off);
            if (kRowsPerSg > 5u) acc5 += simd_shuffle_down(acc5, off);
            if (kRowsPerSg > 6u) acc6 += simd_shuffle_down(acc6, off);
            if (kRowsPerSg > 7u) acc7 += simd_shuffle_down(acc7, off);
        }
    }

    if (kSplitSgs == 1u) {
        if (simd_lane == 0u) {
            if (h0) output[row0] = acc0;
            if (h1) output[row0 + 1u] = acc1;
            if (h2) output[row0 + 2u] = acc2;
            if (h3) output[row0 + 3u] = acc3;
            if (h4) output[row0 + 4u] = acc4;
            if (h5) output[row0 + 5u] = acc5;
            if (h6) output[row0 + 6u] = acc6;
            if (h7) output[row0 + 7u] = acc7;
        }
        return;
    }

    const uint red_base = team * kSplitSgs;
    if (simd_lane == 0u) {
        red[red_base + split] = acc0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && h0) {
        float s = 0.0f;
        for (uint i = 0u; i < kSplitSgs; ++i) {
            s += red[red_base + i];
        }
        output[row0] = s;
    }
    if (kRowsPerSg > 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0u) {
            red[red_base + split] = acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && h1) {
            float s = 0.0f;
            for (uint i = 0u; i < kSplitSgs; ++i) s += red[red_base + i];
            output[row0 + 1u] = s;
        }
    }
    if (kRowsPerSg > 2u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0u) red[red_base + split] = acc2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && h2) {
            float s = 0.0f;
            for (uint i = 0u; i < kSplitSgs; ++i) s += red[red_base + i];
            output[row0 + 2u] = s;
        }
    }
    if (kRowsPerSg > 3u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0u) red[red_base + split] = acc3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && h3) {
            float s = 0.0f;
            for (uint i = 0u; i < kSplitSgs; ++i) s += red[red_base + i];
            output[row0 + 3u] = s;
        }
    }
    if (kRowsPerSg > 4u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0u) red[red_base + split] = acc4;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && h4) {
            float s = 0.0f;
            for (uint i = 0u; i < kSplitSgs; ++i) s += red[red_base + i];
            output[row0 + 4u] = s;
        }
    }
    if (kRowsPerSg > 5u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0u) red[red_base + split] = acc5;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && h5) {
            float s = 0.0f;
            for (uint i = 0u; i < kSplitSgs; ++i) s += red[red_base + i];
            output[row0 + 5u] = s;
        }
    }
    if (kRowsPerSg > 6u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0u) red[red_base + split] = acc6;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && h6) {
            float s = 0.0f;
            for (uint i = 0u; i < kSplitSgs; ++i) s += red[red_base + i];
            output[row0 + 6u] = s;
        }
    }
    if (kRowsPerSg > 7u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0u) red[red_base + split] = acc7;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && h7) {
            float s = 0.0f;
            for (uint i = 0u; i < kSplitSgs; ++i) s += red[red_base + i];
            output[row0 + 7u] = s;
        }
    }
}

// Uniform Q4 group-64. Same launch geometry as the binary kernel.
kernel void q80_geo_q4_matvec(
    device const uchar* codes     [[buffer(0)]],
    device const half* scales     [[buffer(1)]],
    device const float* input     [[buffer(2)]],
    device float* output          [[buffer(3)]],
    constant uint& rows           [[buffer(4)]],
    constant uint& cols           [[buffer(5)]],
    constant uint& groups_per_row [[buffer(6)]],
    uint group_id                 [[threadgroup_position_in_grid]],
    uint lid                      [[thread_index_in_threadgroup]],
    uint simd_lane                [[thread_index_in_simdgroup]],
    uint simd_id                  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[32];
    threadgroup float x_tg[2048];

    constexpr uint kGroup = 64u;
    constexpr uint kCodeBytes = 32u;
    const bool serial_tpr = kSplitSgs == 0u;
    const uint teams = serial_tpr ? kTptg : (kSgPerTg / kSplitSgs);
    const uint rows_per_tg = teams * kRowsPerSg;
    const uint tpr = serial_tpr ? 1u : (32u * kSplitSgs);
    const uint team = serial_tpr ? lid : (simd_id / kSplitSgs);
    const uint split = serial_tpr ? 0u : (simd_id % kSplitSgs);
    const uint lane_in_row = serial_tpr ? 0u : (split * 32u + simd_lane);
    const uint row0 = group_id * rows_per_tg + team * kRowsPerSg;
    const uint weights_per_step = 8u * kVec;

    if (kStageX) {
        for (uint i = lid; i < cols && i < 2048u; i += kTptg) {
            x_tg[i] = input[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    const bool h0 = row0 < rows;
    const bool h1 = kRowsPerSg > 1u && row0 + 1u < rows;
    const bool h2 = kRowsPerSg > 2u && row0 + 2u < rows;
    const bool h3 = kRowsPerSg > 3u && row0 + 3u < rows;
    const bool h4 = kRowsPerSg > 4u && row0 + 4u < rows;
    const bool h5 = kRowsPerSg > 5u && row0 + 5u < rows;
    const bool h6 = kRowsPerSg > 6u && row0 + 6u < rows;
    const bool h7 = kRowsPerSg > 7u && row0 + 7u < rows;
    const bool live = h0;

    const uint step = tpr * weights_per_step;
    const uint tiled = (cols / step) * step;
    uint base = lane_in_row * weights_per_step;
    while (live && base + kUnroll * step <= tiled) {
        for (uint u = 0u; u < kUnroll; ++u) {
            const uint col = base + u * step;
            const uint group = col / kGroup;
            const uint local = col - group * kGroup;
            for (uint r = 0u; r < kRowsPerSg; ++r) {
                const uint row = row0 + r;
                if (row >= rows) {
                    continue;
                }
                const uint rgb = row * groups_per_row + group;
                const float scale = float(scales[rgb]);
                device const uchar* p =
                    codes + rgb * kCodeBytes + (local >> 1u);
                uint packed = 0u;
                if (kVec == 1u) {
                    packed = *((device const uint*)p);
                } else if (kVec == 2u) {
                    const uint2 v = *((device const uint2*)p);
                    packed = v.x;
                    const float term0 = geo_q4_dot_bytes(v.x, 4u, scale, input, x_tg, col);
                    const float term1 = geo_q4_dot_bytes(v.y, 4u, scale, input, x_tg, col + 8u);
                    const float term = term0 + term1;
                    if (r == 0u) acc0 = geo_acc_add(acc0, term);
                    else if (r == 1u) acc1 = geo_acc_add(acc1, term);
                    else if (r == 2u) acc2 = geo_acc_add(acc2, term);
                    else if (r == 3u) acc3 = geo_acc_add(acc3, term);
                    else if (r == 4u) acc4 = geo_acc_add(acc4, term);
                    else if (r == 5u) acc5 = geo_acc_add(acc5, term);
                    else if (r == 6u) acc6 = geo_acc_add(acc6, term);
                    else acc7 = geo_acc_add(acc7, term);
                    continue;
                } else {
                    const uint4 v = *((device const uint4*)p);
                    float term = geo_q4_dot_bytes(v.x, 4u, scale, input, x_tg, col);
                    term += geo_q4_dot_bytes(v.y, 4u, scale, input, x_tg, col + 8u);
                    term += geo_q4_dot_bytes(v.z, 4u, scale, input, x_tg, col + 16u);
                    term += geo_q4_dot_bytes(v.w, 4u, scale, input, x_tg, col + 24u);
                    if (r == 0u) acc0 = geo_acc_add(acc0, term);
                    else if (r == 1u) acc1 = geo_acc_add(acc1, term);
                    else if (r == 2u) acc2 = geo_acc_add(acc2, term);
                    else if (r == 3u) acc3 = geo_acc_add(acc3, term);
                    else if (r == 4u) acc4 = geo_acc_add(acc4, term);
                    else if (r == 5u) acc5 = geo_acc_add(acc5, term);
                    else if (r == 6u) acc6 = geo_acc_add(acc6, term);
                    else acc7 = geo_acc_add(acc7, term);
                    continue;
                }
                const float term = geo_q4_dot_bytes(packed, 4u, scale, input, x_tg, col);
                if (r == 0u) acc0 = geo_acc_add(acc0, term);
                else if (r == 1u) acc1 = geo_acc_add(acc1, term);
                else if (r == 2u) acc2 = geo_acc_add(acc2, term);
                else if (r == 3u) acc3 = geo_acc_add(acc3, term);
                else if (r == 4u) acc4 = geo_acc_add(acc4, term);
                else if (r == 5u) acc5 = geo_acc_add(acc5, term);
                else if (r == 6u) acc6 = geo_acc_add(acc6, term);
                else acc7 = geo_acc_add(acc7, term);
            }
        }
        base += kUnroll * step;
    }
    while (live && base < cols) {
        const uint col = base;
        for (uint r = 0u; r < kRowsPerSg; ++r) {
            const uint row = row0 + r;
            if (row >= rows) {
                continue;
            }
            const uint rgb0 = row * groups_per_row;
            float term = 0.0f;
            const uint n = min(weights_per_step, cols - col);
            for (uint c = 0u; c < n; ++c) {
                const uint cc = col + c;
                const uint group = cc / kGroup;
                const uint local = cc - group * kGroup;
                const uint rgb = rgb0 + group;
                const uchar byte = codes[rgb * kCodeBytes + (local >> 1u)];
                const uchar nibble = ((local & 1u) == 0u) ? (byte & 0x0fu) : (byte >> 4u);
                term += float(int(nibble) - 8) * float(scales[rgb]) * geo_load_x(input, x_tg, cc);
            }
            if (r == 0u) acc0 = geo_acc_add(acc0, term);
            else if (r == 1u) acc1 = geo_acc_add(acc1, term);
            else if (r == 2u) acc2 = geo_acc_add(acc2, term);
            else if (r == 3u) acc3 = geo_acc_add(acc3, term);
            else if (r == 4u) acc4 = geo_acc_add(acc4, term);
            else if (r == 5u) acc5 = geo_acc_add(acc5, term);
            else if (r == 6u) acc6 = geo_acc_add(acc6, term);
            else acc7 = geo_acc_add(acc7, term);
        }
        base += step;
    }

    if (kReduce == 0u || tpr == 1u) {
        if (lane_in_row == 0u) {
            if (h0) output[row0] = acc0;
            if (h1) output[row0 + 1u] = acc1;
            if (h2) output[row0 + 2u] = acc2;
            if (h3) output[row0 + 3u] = acc3;
            if (h4) output[row0 + 4u] = acc4;
            if (h5) output[row0 + 5u] = acc5;
            if (h6) output[row0 + 6u] = acc6;
            if (h7) output[row0 + 7u] = acc7;
        }
        return;
    }

    if (kReduce == 1u) {
        acc0 = simd_sum(acc0);
        if (kRowsPerSg > 1u) acc1 = simd_sum(acc1);
        if (kRowsPerSg > 2u) acc2 = simd_sum(acc2);
        if (kRowsPerSg > 3u) acc3 = simd_sum(acc3);
        if (kRowsPerSg > 4u) acc4 = simd_sum(acc4);
        if (kRowsPerSg > 5u) acc5 = simd_sum(acc5);
        if (kRowsPerSg > 6u) acc6 = simd_sum(acc6);
        if (kRowsPerSg > 7u) acc7 = simd_sum(acc7);
    } else {
        for (uint off = 16u; off > 0u; off >>= 1u) {
            acc0 += simd_shuffle_down(acc0, off);
            if (kRowsPerSg > 1u) acc1 += simd_shuffle_down(acc1, off);
            if (kRowsPerSg > 2u) acc2 += simd_shuffle_down(acc2, off);
            if (kRowsPerSg > 3u) acc3 += simd_shuffle_down(acc3, off);
            if (kRowsPerSg > 4u) acc4 += simd_shuffle_down(acc4, off);
            if (kRowsPerSg > 5u) acc5 += simd_shuffle_down(acc5, off);
            if (kRowsPerSg > 6u) acc6 += simd_shuffle_down(acc6, off);
            if (kRowsPerSg > 7u) acc7 += simd_shuffle_down(acc7, off);
        }
    }

    if (kSplitSgs == 1u) {
        if (simd_lane == 0u) {
            if (h0) output[row0] = acc0;
            if (h1) output[row0 + 1u] = acc1;
            if (h2) output[row0 + 2u] = acc2;
            if (h3) output[row0 + 3u] = acc3;
            if (h4) output[row0 + 4u] = acc4;
            if (h5) output[row0 + 5u] = acc5;
            if (h6) output[row0 + 6u] = acc6;
            if (h7) output[row0 + 7u] = acc7;
        }
        return;
    }

    const uint red_base = team * kSplitSgs;
    if (simd_lane == 0u) red[red_base + split] = acc0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && h0) {
        float s = 0.0f;
        for (uint i = 0u; i < kSplitSgs; ++i) s += red[red_base + i];
        output[row0] = s;
    }
    if (kRowsPerSg > 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0u) red[red_base + split] = acc1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && h1) {
            float s = 0.0f;
            for (uint i = 0u; i < kSplitSgs; ++i) s += red[red_base + i];
            output[row0 + 1u] = s;
        }
    }
    if (kRowsPerSg > 2u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0u) red[red_base + split] = acc2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && h2) {
            float s = 0.0f;
            for (uint i = 0u; i < kSplitSgs; ++i) s += red[red_base + i];
            output[row0 + 2u] = s;
        }
    }
    if (kRowsPerSg > 3u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0u) red[red_base + split] = acc3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && h3) {
            float s = 0.0f;
            for (uint i = 0u; i < kSplitSgs; ++i) s += red[red_base + i];
            output[row0 + 3u] = s;
        }
    }
    if (kRowsPerSg > 4u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0u) red[red_base + split] = acc4;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && h4) {
            float s = 0.0f;
            for (uint i = 0u; i < kSplitSgs; ++i) s += red[red_base + i];
            output[row0 + 4u] = s;
        }
    }
    if (kRowsPerSg > 5u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0u) red[red_base + split] = acc5;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && h5) {
            float s = 0.0f;
            for (uint i = 0u; i < kSplitSgs; ++i) s += red[red_base + i];
            output[row0 + 5u] = s;
        }
    }
    if (kRowsPerSg > 6u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0u) red[red_base + split] = acc6;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && h6) {
            float s = 0.0f;
            for (uint i = 0u; i < kSplitSgs; ++i) s += red[red_base + i];
            output[row0 + 6u] = s;
        }
    }
    if (kRowsPerSg > 7u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0u) red[red_base + split] = acc7;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && h7) {
            float s = 0.0f;
            for (uint i = 0u; i < kSplitSgs; ++i) s += red[red_base + i];
            output[row0 + 7u] = s;
        }
    }
}

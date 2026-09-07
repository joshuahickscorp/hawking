// Batched prefill for the Qwen3.8 hybrid resident.
//
// Decode stays on the existing per-token matvec graph. These kernels consume
// a token chunk: GEMM (simdgroup_matrix) for the sealed packings, then a
// chunkwise mixer that updates conv / recurrent / KV state in the same
// order as N calls to the decode kernels.
//
// Layout (token-major):
//   X[n * cols + col], Y[n * rows + row]
// Affine HGRAVF01 group-64: w = q * scale + bias, q in {0,1,2,3}, 16 code
// bytes/group, fp16 scale and bias.
// HQ30UQ4 group-64: w = (nibble - 8) * scale, 32 code bytes/group, fp16 scale.
//
// GEMM geometry follows gemm_q4_k_m_batched_v3w_mma_n32: one simdgroup per
// 8 output rows, N-tiled MMA so one weight dequant serves the whole chunk.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// One threadgroup = 4 simdgroups × 8 rows = 32 output rows, N-tiled to 64.
// Sharing the BK×BN activation tile across 32 rows (not 8) is the lever the
// earlier RxK microbench missed: X is loaded once per 32 rows.
constant uint QWEN38_PREFILL_SG = 4u;
constant uint QWEN38_PREFILL_BM_SG = 8u;
constant uint QWEN38_PREFILL_BM = QWEN38_PREFILL_SG * QWEN38_PREFILL_BM_SG; // 32
constant uint QWEN38_PREFILL_BK = 32u;
constant uint QWEN38_PREFILL_BN = 64u;
constant uint QWEN38_PREFILL_N_TILES = 8u; // 8 * 8 = 64
constant uint QWEN38_PREFILL_SHMEM_FLOATS =
    QWEN38_PREFILL_BM * QWEN38_PREFILL_BK          // Ws
    + QWEN38_PREFILL_BK * QWEN38_PREFILL_BN        // Xs
    + QWEN38_PREFILL_BM * QWEN38_PREFILL_BN;       // Os
// 32*32 + 32*64 + 32*64 = 5120 floats = 20 KiB.

static inline float qwen38_prefill_affine_q2_w(
    device const uchar* codes,
    device const half* scales,
    device const half* biases,
    uint row,
    uint col,
    uint cols)
{
    const uint groups_per_row = cols >> 6u;
    const uint group = col >> 6u;
    const uint local = col & 63u;
    const uint rgb = row * groups_per_row + group;
    const uchar packed = codes[rgb * 16u + (local >> 2u)];
    const uint q = (uint(packed) >> (2u * (local & 3u))) & 3u;
    return float(q) * float(scales[rgb]) + float(biases[rgb]);
}

static inline float qwen38_prefill_q4_w(
    device const uchar* codes,
    device const half* scales,
    uint row,
    uint col,
    uint cols)
{
    const uint groups_per_row = cols >> 6u;
    const uint group = col >> 6u;
    const uint local = col & 63u;
    const uint rgb = row * groups_per_row + group;
    const uchar packed = codes[rgb * 32u + (local >> 1u)];
    const uchar nibble = ((local & 1u) == 0u) ? (packed & 0x0fu) : (packed >> 4u);
    return float(int(nibble) - 8) * float(scales[rgb]);
}

// ── Affine-q2 group-64 GEMM, N <= 64 ─────────────────────────────────────
// Grid threads: ceil(rows/32)*128, TG 128. Four simdgroups, 8 rows each.
kernel void qwen38_prefill_affine_q2_g64_gemm_mma_n64(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* x_batch     [[buffer(3)]],
    device float*       y_batch     [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& batch            [[buffer(7)]],
    threadgroup float* shmem        [[threadgroup(0)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
    uint gid                        [[threadgroup_position_in_grid]])
{
    const uint row0 = gid * QWEN38_PREFILL_BM;
    if (row0 >= rows) return;
    const uint B = min(batch, QWEN38_PREFILL_BN);
    const uint NW = QWEN38_PREFILL_BN;
    threadgroup float* Ws = shmem;
    threadgroup float* Xs = shmem + QWEN38_PREFILL_BM * QWEN38_PREFILL_BK;
    threadgroup float* Os = Xs + QWEN38_PREFILL_BK * NW;

    for (uint e = tid; e < QWEN38_PREFILL_BM * NW; e += 128u) {
        Os[e] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> acc[QWEN38_PREFILL_N_TILES];
    for (uint t = 0u; t < QWEN38_PREFILL_N_TILES; ++t) {
        simdgroup_load(
            acc[t],
            Os + simd_id * QWEN38_PREFILL_BM_SG * NW + t * 8u,
            NW,
            ulong2(0, 0));
    }

    for (uint k0 = 0u; k0 < cols; k0 += QWEN38_PREFILL_BK) {
        for (uint i = tid; i < QWEN38_PREFILL_BM * QWEN38_PREFILL_BK; i += 128u) {
            const uint m = i / QWEN38_PREFILL_BK;
            const uint kk = i - m * QWEN38_PREFILL_BK;
            const uint row = row0 + m;
            const uint col = k0 + kk;
            float w = 0.0f;
            if (row < rows && col < cols) {
                w = qwen38_prefill_affine_q2_w(codes, scales, biases, row, col, cols);
            }
            Ws[i] = w;
        }
        for (uint i = tid; i < QWEN38_PREFILL_BK * NW; i += 128u) {
            const uint kk = i / NW;
            const uint n = i - kk * NW;
            const uint col = k0 + kk;
            float xv = 0.0f;
            if (n < B && col < cols) {
                xv = x_batch[(ulong)n * (ulong)cols + (ulong)col];
            }
            Xs[i] = xv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint w_row = simd_id * QWEN38_PREFILL_BM_SG * QWEN38_PREFILL_BK;
        for (uint d8 = 0u; d8 < QWEN38_PREFILL_BK; d8 += 8u) {
            simdgroup_matrix<float, 8, 8> wm;
            simdgroup_load(wm, Ws + w_row + d8, QWEN38_PREFILL_BK, ulong2(0, 0));
            for (uint t = 0u; t < QWEN38_PREFILL_N_TILES; ++t) {
                simdgroup_matrix<float, 8, 8> xm;
                simdgroup_load(xm, Xs + d8 * NW + t * 8u, NW, ulong2(0, 0));
                simdgroup_multiply_accumulate(acc[t], wm, xm, acc[t]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint t = 0u; t < QWEN38_PREFILL_N_TILES; ++t) {
        simdgroup_store(
            acc[t],
            Os + simd_id * QWEN38_PREFILL_BM_SG * NW + t * 8u,
            NW,
            ulong2(0, 0));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint slot = tid; slot < QWEN38_PREFILL_BM * NW; slot += 128u) {
        const uint m = slot / NW;
        const uint n = slot - m * NW;
        const uint row = row0 + m;
        if (n < B && row < rows) {
            y_batch[(ulong)n * (ulong)rows + (ulong)row] = Os[slot];
        }
    }
}

// ── Uniform-Q4 group-64 GEMM, N <= 64 ────────────────────────────────────
kernel void qwen38_prefill_q4_g64_gemm_mma_n64(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const float* x_batch     [[buffer(2)]],
    device float*       y_batch     [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& batch            [[buffer(6)]],
    threadgroup float* shmem        [[threadgroup(0)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
    uint gid                        [[threadgroup_position_in_grid]])
{
    const uint row0 = gid * QWEN38_PREFILL_BM;
    if (row0 >= rows) return;
    const uint B = min(batch, QWEN38_PREFILL_BN);
    const uint NW = QWEN38_PREFILL_BN;
    threadgroup float* Ws = shmem;
    threadgroup float* Xs = shmem + QWEN38_PREFILL_BM * QWEN38_PREFILL_BK;
    threadgroup float* Os = Xs + QWEN38_PREFILL_BK * NW;

    for (uint e = tid; e < QWEN38_PREFILL_BM * NW; e += 128u) {
        Os[e] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> acc[QWEN38_PREFILL_N_TILES];
    for (uint t = 0u; t < QWEN38_PREFILL_N_TILES; ++t) {
        simdgroup_load(
            acc[t],
            Os + simd_id * QWEN38_PREFILL_BM_SG * NW + t * 8u,
            NW,
            ulong2(0, 0));
    }

    for (uint k0 = 0u; k0 < cols; k0 += QWEN38_PREFILL_BK) {
        for (uint i = tid; i < QWEN38_PREFILL_BM * QWEN38_PREFILL_BK; i += 128u) {
            const uint m = i / QWEN38_PREFILL_BK;
            const uint kk = i - m * QWEN38_PREFILL_BK;
            const uint row = row0 + m;
            const uint col = k0 + kk;
            float w = 0.0f;
            if (row < rows && col < cols) {
                w = qwen38_prefill_q4_w(codes, scales, row, col, cols);
            }
            Ws[i] = w;
        }
        for (uint i = tid; i < QWEN38_PREFILL_BK * NW; i += 128u) {
            const uint kk = i / NW;
            const uint n = i - kk * NW;
            const uint col = k0 + kk;
            float xv = 0.0f;
            if (n < B && col < cols) {
                xv = x_batch[(ulong)n * (ulong)cols + (ulong)col];
            }
            Xs[i] = xv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint w_row = simd_id * QWEN38_PREFILL_BM_SG * QWEN38_PREFILL_BK;
        for (uint d8 = 0u; d8 < QWEN38_PREFILL_BK; d8 += 8u) {
            simdgroup_matrix<float, 8, 8> wm;
            simdgroup_load(wm, Ws + w_row + d8, QWEN38_PREFILL_BK, ulong2(0, 0));
            for (uint t = 0u; t < QWEN38_PREFILL_N_TILES; ++t) {
                simdgroup_matrix<float, 8, 8> xm;
                simdgroup_load(xm, Xs + d8 * NW + t * 8u, NW, ulong2(0, 0));
                simdgroup_multiply_accumulate(acc[t], wm, xm, acc[t]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint t = 0u; t < QWEN38_PREFILL_N_TILES; ++t) {
        simdgroup_store(
            acc[t],
            Os + simd_id * QWEN38_PREFILL_BM_SG * NW + t * 8u,
            NW,
            ulong2(0, 0));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint slot = tid; slot < QWEN38_PREFILL_BM * NW; slot += 128u) {
        const uint m = slot / NW;
        const uint n = slot - m * NW;
        const uint row = row0 + m;
        if (n < B && row < rows) {
            y_batch[(ulong)n * (ulong)rows + (ulong)row] = Os[slot];
        }
    }
}

// ── Batched RMSNorm: one threadgroup per token, (1+w) math ───────────────
kernel void qwen38_prefill_rmsnorm_f32(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant uint& hidden       [[buffer(3)]],
    constant float& eps         [[buffer(4)]],
    constant uint& n_tokens     [[buffer(5)]],
    threadgroup float* scratch  [[threadgroup(0)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]],
    uint token                  [[threadgroup_position_in_grid]])
{
    if (token >= n_tokens) return;
    const uint base = token * hidden;
    float sum = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        const float v = input[base + i];
        sum += v * v;
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse_rms = 1.0f / sqrt(scratch[0] / float(hidden) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < hidden; i += tg_size) {
        output[base + i] = input[base + i] * inverse_rms * (1.0f + weight[i]);
    }
}

kernel void qwen38_prefill_add_residual_f32(
    device const float* residual [[buffer(0)]],
    device const float* delta    [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant uint& n             [[buffer(3)]],
    uint id                      [[thread_position_in_grid]])
{
    if (id >= n) return;
    output[id] = residual[id] + delta[id];
}

kernel void qwen38_prefill_add_residual_rmsnorm_f32(
    device const float* residual_in [[buffer(0)]],
    device const float* delta       [[buffer(1)]],
    device float* residual_out      [[buffer(2)]],
    device const float* weight      [[buffer(3)]],
    device float* x_norm            [[buffer(4)]],
    constant uint& hidden           [[buffer(5)]],
    constant float& eps             [[buffer(6)]],
    constant uint& n_tokens         [[buffer(7)]],
    threadgroup float* scratch      [[threadgroup(0)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint token                      [[threadgroup_position_in_grid]])
{
    if (token >= n_tokens) return;
    const uint base = token * hidden;
    float sum = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        const float v = residual_in[base + i] + delta[base + i];
        residual_out[base + i] = v;
        sum += v * v;
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse_rms = 1.0f / sqrt(scratch[0] / float(hidden) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < hidden; i += tg_size) {
        x_norm[base + i] = residual_out[base + i] * inverse_rms * (1.0f + weight[i]);
    }
}

kernel void qwen38_prefill_swiglu_f32(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float* output     [[buffer(2)]],
    constant uint& n         [[buffer(3)]],
    uint id                  [[thread_position_in_grid]])
{
    if (id >= n) return;
    const float g = gate[id];
    output[id] = (g / (1.0f + exp(-g))) * up[id];
}

kernel void qwen38_prefill_q4_embed(
    device const uchar* codes   [[buffer(0)]],
    device const half* scales   [[buffer(1)]],
    device const uint* tokens   [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& n_tokens     [[buffer(4)]],
    constant uint& hidden       [[buffer(5)]],
    constant uint& vocab        [[buffer(6)]],
    constant uint& group_size   [[buffer(7)]],
    uint2 gid                   [[thread_position_in_grid]])
{
    const uint h = gid.x;
    const uint t = gid.y;
    if (h >= hidden || t >= n_tokens) return;
    const uint token = tokens[t];
    if (token >= vocab) {
        output[t * hidden + h] = 0.0f;
        return;
    }
    const uint element = token * hidden + h;
    const uint group = element / group_size;
    const uint local = element % group_size;
    const uint code_base = group * (group_size >> 1u);
    const uchar packed = codes[code_base + (local >> 1u)];
    const uchar nibble = ((local & 1u) == 0u) ? (packed & 0x0fu) : (packed >> 4u);
    output[t * hidden + h] = float(int(nibble) - 8) * float(scales[group]);
}

kernel void qwen38_prefill_copy_row(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& row      [[buffer(2)]],
    constant uint& cols     [[buffer(3)]],
    uint id                 [[thread_position_in_grid]])
{
    if (id >= cols) return;
    dst[id] = src[row * cols + id];
}

// ── Chunkwise QKVZ rearrange + causal conv + L2 ──────────────────────────
// One threadgroup per key head. Loops the chunk so conv_state matches N
// sequential decode updates.
kernel void qwen38_prefill_qkvz_rearrange_conv(
    device const float* projected_qkvz [[buffer(0)]],
    device const float* conv_weight    [[buffer(1)]],
    device float* conv_state           [[buffer(2)]],
    device float* repeated_query       [[buffer(3)]],
    device float* repeated_key         [[buffer(4)]],
    device float* convolved_value      [[buffer(5)]],
    device float* z                    [[buffer(6)]],
    constant uint& n_tokens            [[buffer(7)]],
    constant uint& qkvz_stride         [[buffer(8)]],
    constant uint& value_stride        [[buffer(9)]],
    constant float& eps                [[buffer(10)]],
    threadgroup float* scratch         [[threadgroup(0)]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint3 group                        [[threadgroup_position_in_grid]])
{
    const uint key_head = group.y;
    constexpr uint key_heads = 16u;
    constexpr uint values_per_key_head = 3u;
    constexpr uint key_head_dim = 128u;
    constexpr uint value_head_dim = 128u;
    constexpr uint conv_kernel = 4u;
    if (key_head >= key_heads) return;

    const uint value_rows_per_key_head = values_per_key_head * value_head_dim;
    const uint qkvz_rows_per_key_head = key_head_dim * 2u + value_rows_per_key_head * 2u;
    const uint key_elements = key_heads * key_head_dim;
    const uint value_base = key_head * value_rows_per_key_head;

    for (uint t = 0u; t < n_tokens; ++t) {
        const uint qkvz_token = t * qkvz_stride + key_head * qkvz_rows_per_key_head;
        threadgroup float* query_local = scratch;
        threadgroup float* key_local = scratch + 128u;
        threadgroup float* query_sums = scratch + 256u;
        threadgroup float* key_sums = scratch + 512u;

        if (tid < key_head_dim) {
            const uint query_channel = key_head * key_head_dim + tid;
            const uint key_channel = key_elements + query_channel;
            query_local[tid] = qwen38_causal_conv_update_f32(
                conv_state, conv_weight, query_channel,
                projected_qkvz[qkvz_token + tid], conv_kernel);
            key_local[tid] = qwen38_causal_conv_update_f32(
                conv_state, conv_weight, key_channel,
                projected_qkvz[qkvz_token + key_head_dim + tid], conv_kernel);
        }
        for (uint row = tid; row < value_rows_per_key_head; row += 256u) {
            const uint value_channel = key_elements * 2u + value_base + row;
            convolved_value[t * value_stride + value_base + row] =
                qwen38_causal_conv_update_f32(
                    conv_state, conv_weight, value_channel,
                    projected_qkvz[qkvz_token + key_head_dim * 2u + row],
                    conv_kernel);
            z[t * value_stride + value_base + row] = projected_qkvz[
                qkvz_token + key_head_dim * 2u + value_rows_per_key_head + row];
        }
        query_sums[tid] = tid < key_head_dim ? query_local[tid] * query_local[tid] : 0.0f;
        key_sums[tid] = tid < key_head_dim ? key_local[tid] * key_local[tid] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                query_sums[tid] += query_sums[tid + stride];
                key_sums[tid] += key_sums[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid < key_head_dim) {
            const float query_scale = rsqrt(query_sums[0] + eps) * rsqrt(float(key_head_dim));
            const float key_scale = rsqrt(key_sums[0] + eps);
            const uint value_head_base = key_head * values_per_key_head;
            for (uint repeat = 0u; repeat < values_per_key_head; ++repeat) {
                const uint destination = (value_head_base + repeat) * key_head_dim + tid;
                repeated_query[t * value_stride + destination] = query_local[tid] * query_scale;
                repeated_key[t * value_stride + destination] = key_local[tid] * key_scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ── Chunkwise gated-delta, f4 vi-widened (sealed default) ────────────────
// Same arithmetic as qwen38_gated_delta_decode_vi_simd_ba_f4, looped over
// the chunk so rec_state matches N sequential decode updates.
kernel void qwen38_prefill_gated_delta_ba_f4(
    device float* state                 [[buffer(0)]],
    device const float* query           [[buffer(1)]],
    device const float* key             [[buffer(2)]],
    device const float* value           [[buffer(3)]],
    device const float* projected_ba    [[buffer(4)]],
    device const float* a_log           [[buffer(5)]],
    device const float* dt_bias         [[buffer(6)]],
    device float* output                [[buffer(7)]],
    constant uint& n_tokens             [[buffer(8)]],
    constant uint& value_stride         [[buffer(9)]],
    constant uint& ba_stride            [[buffer(10)]],
    threadgroup float* scratch          [[threadgroup(0)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]],
    uint3 group                         [[threadgroup_position_in_grid]])
{
    constexpr uint heads = 48u;
    constexpr uint key_dim = 128u;
    constexpr uint value_dim = 128u;
    const uint head = group.y;
    const uint vi_base = group.z * 4u;
    if (head >= heads || vi_base >= value_dim) return;

    const uint state_base = head * key_dim * value_dim;
    const uint ki = tid;
    const uint index = state_base + ki * value_dim + vi_base;

    for (uint t = 0u; t < n_tokens; ++t) {
        const uint qk_base = t * value_stride + head * key_dim;
        const uint value_base = t * value_stride + head * value_dim;
        float d;
        float b;
        qwen38_ba_decay_beta_f32(
            projected_ba + t * ba_stride, a_log, dt_bias, head, d, b);
        const float kk = key[qk_base + ki];
        const float qq = query[qk_base + ki];

        float4 s = *((device const float4*)(state + index));
        float4 decayed = s * d;
        float4 part = float4(
            simd_sum(decayed.x * kk),
            simd_sum(decayed.y * kk),
            simd_sum(decayed.z * kk),
            simd_sum(decayed.w * kk));
        if (simd_lane == 0u) {
            scratch[simd_id * 4u + 0u] = part.x;
            scratch[simd_id * 4u + 1u] = part.y;
            scratch[simd_id * 4u + 2u] = part.z;
            scratch[simd_id * 4u + 3u] = part.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float4 kv = float4(
            scratch[0] + scratch[4] + scratch[8] + scratch[12],
            scratch[1] + scratch[5] + scratch[9] + scratch[13],
            scratch[2] + scratch[6] + scratch[10] + scratch[14],
            scratch[3] + scratch[7] + scratch[11] + scratch[15]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float4 vv = *((device const float4*)(value + value_base + vi_base));
        const float4 delta = (vv - kv) * b;
        s = decayed + kk * delta;

        float4 outp = float4(
            simd_sum(s.x * qq),
            simd_sum(s.y * qq),
            simd_sum(s.z * qq),
            simd_sum(s.w * qq));
        if (simd_lane == 0u) {
            scratch[simd_id * 4u + 0u] = outp.x;
            scratch[simd_id * 4u + 1u] = outp.y;
            scratch[simd_id * 4u + 2u] = outp.z;
            scratch[simd_id * 4u + 3u] = outp.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            *((device float4*)(output + value_base + vi_base)) = float4(
                scratch[0] + scratch[4] + scratch[8] + scratch[12],
                scratch[1] + scratch[5] + scratch[9] + scratch[13],
                scratch[2] + scratch[6] + scratch[10] + scratch[14],
                scratch[3] + scratch[7] + scratch[11] + scratch[15]);
        }
        *((device float4*)(state + index)) = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void qwen38_prefill_gated_rmsnorm(
    device const float* input     [[buffer(0)]],
    device const float* gate      [[buffer(1)]],
    device const float* weight    [[buffer(2)]],
    device float* output          [[buffer(3)]],
    constant uint& n_tokens       [[buffer(4)]],
    constant uint& value_stride   [[buffer(5)]],
    constant uint& heads          [[buffer(6)]],
    constant uint& value_head_dim [[buffer(7)]],
    constant float& eps           [[buffer(8)]],
    threadgroup float* red        [[threadgroup(0)]],
    uint head                     [[threadgroup_position_in_grid]],
    uint tid                      [[thread_position_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]])
{
    if (head >= heads) return;
    for (uint t = 0u; t < n_tokens; ++t) {
        const uint base = t * value_stride + head * value_head_dim;
        float local = 0.0f;
        for (uint i = tid; i < value_head_dim; i += tg_size) {
            const float v = input[base + i];
            local += v * v;
        }
        red[tid] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) red[tid] += red[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float inverse_rms = 1.0f / sqrt(red[0] / float(value_head_dim) + eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < value_head_dim; i += tg_size) {
            const float z = gate[base + i];
            const float silu = z / (1.0f + exp(-z));
            output[base + i] = input[base + i] * inverse_rms * weight[i] * silu;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ── Batched GQA RoPE + KV cache write ────────────────────────────────────
// One threadgroup per (token, head). Same (1+w) RMS + rotate_half as
// qwen38_gqa_qk_norm_rope_cache_tg.
kernel void qwen38_prefill_gqa_rope_cache(
    device const float* q_proj     [[buffer(0)]],
    device const float* k_proj     [[buffer(1)]],
    device const float* v_proj     [[buffer(2)]],
    device const float* q_norm     [[buffer(3)]],
    device const float* k_norm     [[buffer(4)]],
    device float* query            [[buffer(5)]],
    device float* key_cache        [[buffer(6)]],
    device float* value_cache      [[buffer(7)]],
    constant uint& pos0            [[buffer(8)]],
    constant uint& n_tokens        [[buffer(9)]],
    constant uint& q_stride        [[buffer(10)]],
    constant uint& kv_stride       [[buffer(11)]],
    constant uint& query_stride    [[buffer(12)]],
    constant uint& cache_seq_stride[[buffer(13)]],
    threadgroup float* red         [[threadgroup(0)]],
    uint3 tgp                      [[threadgroup_position_in_grid]],
    // Metal requires every position-class input on a kernel to be all-scalar or
    // all the same vector width. This kernel genuinely needs a 2-D
    // threadgroup_position_in_grid (tgp.x is the token, tgp.y the head), so the
    // other two widen to match and are narrowed back here. Declared scalar
    // alongside a uint3, this kernel did not compile at all -- which is why the
    // committed shader was unbuildable and the recorded 78.3 tok/s came from a
    // working tree that was never committed.
    uint3 tid3                     [[thread_position_in_threadgroup]],
    uint3 tg_size3                 [[threads_per_threadgroup]])
{
    const uint tid = tid3.x;
    const uint tg_size = tg_size3.x;
    constexpr uint n_heads = 24u;
    constexpr uint n_kv_heads = 4u;
    constexpr uint head_dim = 256u;
    constexpr uint rotary_dim = 64u;
    constexpr float rope_theta = 10000000.0f;
    constexpr float rms_epsilon = 1.0e-6f;
    const uint token = tgp.x;
    const uint head = tgp.y;
    if (token >= n_tokens || head >= n_heads) return;
    const uint sequence_slot = pos0 + token;
    const uint half_dim = rotary_dim / 2u;
    const uint q_projection_base = token * q_stride + head * (2u * head_dim);
    const uint q_base = token * query_stride + head * head_dim;

    float local = 0.0f;
    for (uint d = tid; d < head_dim; d += tg_size) {
        const float v = q_proj[q_projection_base + d];
        local += v * v;
    }
    red[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float q_inverse_rms = 1.0f / sqrt(red[0] / float(head_dim) + rms_epsilon);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = tid; dim < head_dim; dim += tg_size) {
        const float raw = q_proj[q_projection_base + dim];
        const float normed = raw * q_inverse_rms * (1.0f + q_norm[dim]);
        if (dim < rotary_dim) {
            const uint fi = dim < half_dim ? dim : dim - half_dim;
            const float inv_f = pow(rope_theta, -2.0f * float(fi) / float(rotary_dim));
            const float angle = float(sequence_slot) * inv_f;
            const float c = cos(angle);
            const float sn = sin(angle);
            const uint peer = dim < half_dim ? dim + half_dim : dim - half_dim;
            const float peer_raw = q_proj[q_projection_base + peer] * q_inverse_rms
                * (1.0f + q_norm[peer]);
            query[q_base + dim] = dim < half_dim
                ? normed * c - peer_raw * sn
                : normed * c + peer_raw * sn;
        } else {
            query[q_base + dim] = normed;
        }
    }

    if (head < n_kv_heads) {
        const uint kv_base = token * kv_stride + head * head_dim;
        float klocal = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            const float v = k_proj[kv_base + d];
            klocal += v * v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[tid] = klocal;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) red[tid] += red[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float k_inverse_rms = 1.0f / sqrt(red[0] / float(head_dim) + rms_epsilon);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint cache_base = sequence_slot * cache_seq_stride + head * head_dim;
        for (uint dim = tid; dim < head_dim; dim += tg_size) {
            const float raw = k_proj[kv_base + dim];
            const float normed = raw * k_inverse_rms * (1.0f + k_norm[dim]);
            if (dim < rotary_dim) {
                const uint fi = dim < half_dim ? dim : dim - half_dim;
                const float inv_f = pow(rope_theta, -2.0f * float(fi) / float(rotary_dim));
                const float angle = float(sequence_slot) * inv_f;
                const float c = cos(angle);
                const float sn = sin(angle);
                const uint peer = dim < half_dim ? dim + half_dim : dim - half_dim;
                const float peer_raw = k_proj[kv_base + peer] * k_inverse_rms
                    * (1.0f + k_norm[peer]);
                key_cache[cache_base + dim] = dim < half_dim
                    ? normed * c - peer_raw * sn
                    : normed * c + peer_raw * sn;
            } else {
                key_cache[cache_base + dim] = normed;
            }
            value_cache[cache_base + dim] = v_proj[kv_base + dim];
        }
    }
}

// ── Causal GQA attention, online softmax ─────────────────────────────────
// One threadgroup per (token, head). 256 threads = one per head_dim.
// Does not store the score vector, so it stays inside 32 KB at 32K context.
kernel void qwen38_prefill_gqa_attn(
    device const float* query      [[buffer(0)]],
    device const float* key_cache  [[buffer(1)]],
    device const float* value_cache[[buffer(2)]],
    device float* attn             [[buffer(3)]],
    constant uint& pos0            [[buffer(4)]],
    constant uint& n_tokens        [[buffer(5)]],
    constant uint& query_stride    [[buffer(6)]],
    constant uint& cache_seq_stride[[buffer(7)]],
    threadgroup float* scratch     [[threadgroup(0)]],
    uint3 tgp                      [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]])
{
    constexpr uint n_heads = 24u;
    constexpr uint n_kv_heads = 4u;
    constexpr uint head_dim = 256u;
    constexpr uint group_size = n_heads / n_kv_heads;
    constexpr uint kTile = 16u;
    constexpr float scale = 0.0625f; // 1/sqrt(256)
    const uint token = tgp.x;
    const uint head = tgp.y;
    // TG is 256 = head_dim, so every lane participates in the barriers.
    if (token >= n_tokens || head >= n_heads) return;

    const uint kv_h = head / group_size;
    const uint q_base = token * query_stride + head * head_dim;
    const uint query_pos = pos0 + token;
    const float qv = query[q_base + tid];

    float acc = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    const uint n_ctx = query_pos + 1u;
    for (uint t0 = 0u; t0 < n_ctx; t0 += kTile) {
        const uint tile = min(kTile, n_ctx - t0);
        float part[kTile];
        for (uint i = 0u; i < kTile; ++i) {
            part[i] = 0.0f;
        }
        for (uint i = 0u; i < tile; ++i) {
            const uint kv_base = (t0 + i) * cache_seq_stride + kv_h * head_dim;
            part[i] = simd_sum(qv * key_cache[kv_base + tid]);
        }
        if (simd_lane == 0u) {
            for (uint i = 0u; i < tile; ++i) {
                scratch[i * 8u + simd_id] = part[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score[kTile];
        for (uint i = 0u; i < tile; ++i) {
            const uint s = i * 8u;
            score[i] = (scratch[s] + scratch[s + 1u] + scratch[s + 2u] + scratch[s + 3u]
                + scratch[s + 4u] + scratch[s + 5u] + scratch[s + 6u] + scratch[s + 7u])
                * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0u; i < tile; ++i) {
            const uint kv_base = (t0 + i) * cache_seq_stride + kv_h * head_dim;
            const float m_new = max(m, score[i]);
            const float alpha = exp(m - m_new);
            const float p = exp(score[i] - m_new);
            acc = acc * alpha + p * value_cache[kv_base + tid];
            l = l * alpha + p;
            m = m_new;
        }
    }
    attn[q_base + tid] = acc / l;
}

kernel void qwen38_prefill_sigmoid_gate(
    device const float* attn     [[buffer(0)]],
    device const float* q_proj   [[buffer(1)]],
    device float* gated          [[buffer(2)]],
    constant uint& n_tokens      [[buffer(3)]],
    constant uint& query_stride  [[buffer(4)]],
    constant uint& q_stride      [[buffer(5)]],
    uint2 gid                    [[thread_position_in_grid]])
{
    constexpr uint n_heads = 24u;
    constexpr uint head_dim = 256u;
    const uint dim = gid.x;
    const uint token = gid.y;
    if (dim >= n_heads * head_dim || token >= n_tokens) return;
    const uint head = dim / head_dim;
    const uint dimension = dim - head * head_dim;
    const uint gate_offset = token * q_stride + head * (2u * head_dim) + head_dim + dimension;
    const float gate = q_proj[gate_offset];
    const float sigmoid = 1.0f / (1.0f + exp(-gate));
    const uint out = token * query_stride + dim;
    gated[out] = attn[out] * sigmoid;
}

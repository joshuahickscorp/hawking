// Qwen3.8 reconstruction discriminator.
// Occupancy-tiled in-register consume. Never writes a dense W.
// Launch family matches the production Qwen3.8 q4 winner:
//   64 threads/row, TG 128, 2 rows/TG. Grid = ceil(rows/2)*128.

#include <metal_stdlib>
using namespace metal;

static inline uint wide_extract(device const uchar* codes, uint element, uint bits)
{
    const uint bit0 = element * bits;
    const uint byte0 = bit0 >> 3u;
    const uint shift = bit0 & 7u;
    uint packed = uint(codes[byte0]);
    if (shift + bits > 8u) {
        packed |= uint(codes[byte0 + 1u]) << 8u;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}

static inline float uniform_value(
    device const uchar* codes,
    device const half* scales,
    uint element,
    uint group_size,
    uint bits,
    int bound)
{
    const uint group = element / group_size;
    const uint code = wide_extract(codes, element, bits);
    return float(int(code) - bound) * float(scales[group]);
}

// ── stream control (same recipe as Q80 discriminator) ─────────────────────
kernel void disc_stream_control(
    device const uchar* data [[buffer(0)]],
    device float* out        [[buffer(1)]],
    constant uint& nbytes    [[buffer(2)]],
    constant uint& iters     [[buffer(3)]],
    uint tid                 [[thread_position_in_grid]],
    uint nthreads            [[threads_per_grid]])
{
    if (tid >= nthreads || nbytes < 16u || iters == 0u) return;
    const uint span = nbytes - 15u;
    const uint stride = nthreads * 16u;
    float acc = 0.0f;
    uint off = (tid * 16u) % span;
    for (uint i = 0u; i < iters; ++i) {
        const float4 v = *((device const float4*)(data + off));
        acc += v.x + v.y + v.z + v.w;
        off += stride;
        if (off + 16u > nbytes) off = off % span;
    }
    out[tid] = acc;
}

// Shared tpr64 geometry: 64 threads/row, 2 rows/TG.
#define TPR64_PREAMBLE \
    threadgroup float red[4]; \
    constexpr uint kSplit = 2u; \
    const uint team = simd_id / kSplit; \
    const uint split = simd_id % kSplit; \
    const uint lane_in_row = split * 32u + simd_lane; \
    const uint row = group_id * 2u + team;

#define TPR64_REDUCE_STORE \
    acc = simd_sum(acc); \
    if (simd_lane == 0u) red[simd_id] = acc; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (split == 0u && simd_lane == 0u && row < rows) { \
        output[row] = red[team * kSplit] + red[team * kSplit + 1u]; \
    }

kernel void disc_f32_tpr64(
    device const float* weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant uint& rows         [[buffer(3)]],
    constant uint& cols         [[buffer(4)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    TPR64_PREAMBLE
    float acc = 0.0f;
    if (row < rows) {
        const uint base = row * cols;
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            const uint n = min(8u, cols - col);
            for (uint k = 0u; k < n; ++k) {
                acc += weights[base + col + k] * input[col + k];
            }
        }
    }
    TPR64_REDUCE_STORE
}

kernel void disc_q4_nibble_tpr64(
    device const uchar* codes   [[buffer(0)]],
    device const half* scales   [[buffer(1)]],
    device const float* input   [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& rows         [[buffer(4)]],
    constant uint& cols         [[buffer(5)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    TPR64_PREAMBLE
    float acc = 0.0f;
    if (row < rows) {
        const uint gpr = cols / 64u;
        const uint rgb0 = row * gpr;
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            const uint group = col / 64u;
            const uint local = col - group * 64u;
            const uint rgb = rgb0 + group;
            const float scale = float(scales[rgb]);
            const uint packed = *((device const uint*)(codes + rgb * 32u + (local >> 1u)));
            for (uint i = 0u; i < 4u; ++i) {
                const uint byte = (packed >> (8u * i)) & 0xffu;
                acc += float(int(byte & 0x0fu) - 8) * scale * input[col + 2u * i];
                acc += float(int(byte >> 4u) - 8) * scale * input[col + 2u * i + 1u];
            }
        }
    }
    TPR64_REDUCE_STORE
}

kernel void disc_uniform_bits_tpr64(
    device const uchar* codes   [[buffer(0)]],
    device const half* scales   [[buffer(1)]],
    device const float* input   [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& rows         [[buffer(4)]],
    constant uint& cols         [[buffer(5)]],
    constant uint& group_size   [[buffer(6)]],
    constant uint& bits         [[buffer(7)]],
    constant uint& bound        [[buffer(8)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    TPR64_PREAMBLE
    float acc = 0.0f;
    if (row < rows) {
        const uint row_base = row * cols;
        const int ib = int(bound);
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            const uint n = min(8u, cols - col);
            for (uint k = 0u; k < n; ++k) {
                acc += uniform_value(codes, scales, row_base + col + k, group_size, bits, ib)
                    * input[col + k];
            }
        }
    }
    TPR64_REDUCE_STORE
}

kernel void disc_binary_tpr64(
    device const uchar* signs   [[buffer(0)]],
    device const half* scales   [[buffer(1)]],
    device const float* input   [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& rows         [[buffer(4)]],
    constant uint& cols         [[buffer(5)]],
    constant uint& group_size   [[buffer(6)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    TPR64_PREAMBLE
    float acc = 0.0f;
    if (row < rows) {
        const uint row_base = row * cols;
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            const float scale = float(scales[(row_base + col) / group_size]);
            const uchar byte = signs[(row_base + col) >> 3u];
            acc += ((byte & 0x01u) ? scale : -scale) * input[col];
            acc += ((byte & 0x02u) ? scale : -scale) * input[col + 1u];
            acc += ((byte & 0x04u) ? scale : -scale) * input[col + 2u];
            acc += ((byte & 0x08u) ? scale : -scale) * input[col + 3u];
            acc += ((byte & 0x10u) ? scale : -scale) * input[col + 4u];
            acc += ((byte & 0x20u) ? scale : -scale) * input[col + 5u];
            acc += ((byte & 0x40u) ? scale : -scale) * input[col + 6u];
            acc += ((byte & 0x80u) ? scale : -scale) * input[col + 7u];
        }
    }
    TPR64_REDUCE_STORE
}

// 2-bit ternary: 0 -> 0, 1 -> +s, 2 -> -s.
kernel void disc_ternary_tpr64(
    device const uchar* codes   [[buffer(0)]],
    device const half* scales   [[buffer(1)]],
    device const float* input   [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& rows         [[buffer(4)]],
    constant uint& cols         [[buffer(5)]],
    constant uint& group_size   [[buffer(6)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    TPR64_PREAMBLE
    float acc = 0.0f;
    if (row < rows) {
        const uint row_base = row * cols;
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            const uint n = min(8u, cols - col);
            for (uint k = 0u; k < n; ++k) {
                const uint element = row_base + col + k;
                const uint code = wide_extract(codes, element, 2u);
                if (code == 0u) continue;
                const float scale = float(scales[element / group_size]);
                const float w = (code == 1u) ? scale : -scale;
                acc += w * input[col + k];
            }
        }
    }
    TPR64_REDUCE_STORE
}

// Two additive q2 lattices: levels (-1.5,-0.5,0.5,1.5) * scale.
kernel void disc_additive_tpr64(
    device const uchar* base_codes  [[buffer(0)]],
    device const half* base_scales  [[buffer(1)]],
    device const uchar* res_codes   [[buffer(2)]],
    device const half* res_scales   [[buffer(3)]],
    device const float* input       [[buffer(4)]],
    device float* output            [[buffer(5)]],
    constant uint& rows             [[buffer(6)]],
    constant uint& cols             [[buffer(7)]],
    constant uint& group_size       [[buffer(8)]],
    uint group_id                   [[threadgroup_position_in_grid]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]])
{
    TPR64_PREAMBLE
    float acc = 0.0f;
    if (row < rows) {
        const uint row_base = row * cols;
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            const uint n = min(8u, cols - col);
            for (uint k = 0u; k < n; ++k) {
                const uint element = row_base + col + k;
                const uint group = element / group_size;
                const uint bc = wide_extract(base_codes, element, 2u);
                const uint rc = wide_extract(res_codes, element, 2u);
                const float bv = (float(bc) - 1.5f) * float(base_scales[group]);
                const float rv = (float(rc) - 1.5f) * float(res_scales[group]);
                acc += (bv + rv) * input[col + k];
            }
        }
    }
    TPR64_REDUCE_STORE
}

// Binary + CSR residual, in-register. 64 lanes split the row's outliers.
kernel void disc_binary_csr_tpr64(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const uint* csr_cols     [[buffer(2)]],
    device const uint* csr_row_ptr  [[buffer(3)]],
    device const uchar* csr_signs   [[buffer(4)]],
    device const half* csr_scale    [[buffer(5)]],
    device const float* input       [[buffer(6)]],
    device float* output            [[buffer(7)]],
    constant uint& rows             [[buffer(8)]],
    constant uint& cols             [[buffer(9)]],
    constant uint& group_size       [[buffer(10)]],
    uint group_id                   [[threadgroup_position_in_grid]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]])
{
    TPR64_PREAMBLE
    float acc = 0.0f;
    if (row < rows) {
        const uint row_base = row * cols;
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            const float scale = float(scales[(row_base + col) / group_size]);
            const uchar byte = signs[(row_base + col) >> 3u];
            acc += ((byte & 0x01u) ? scale : -scale) * input[col];
            acc += ((byte & 0x02u) ? scale : -scale) * input[col + 1u];
            acc += ((byte & 0x04u) ? scale : -scale) * input[col + 2u];
            acc += ((byte & 0x08u) ? scale : -scale) * input[col + 3u];
            acc += ((byte & 0x10u) ? scale : -scale) * input[col + 4u];
            acc += ((byte & 0x20u) ? scale : -scale) * input[col + 5u];
            acc += ((byte & 0x40u) ? scale : -scale) * input[col + 6u];
            acc += ((byte & 0x80u) ? scale : -scale) * input[col + 7u];
        }
        const float rscale = float(csr_scale[0]);
        const uint begin = csr_row_ptr[row];
        const uint end = csr_row_ptr[row + 1u];
        for (uint n = begin + lane_in_row; n < end; n += 64u) {
            const uint bit = (csr_signs[n >> 3u] >> (n & 7u)) & 1u;
            const float v = bit ? rscale : -rscale;
            acc += v * input[csr_cols[n]];
        }
    }
    TPR64_REDUCE_STORE
}

// Serial rice-style CSR apply: ONE thread walks every outlier. The artifact.
kernel void disc_csr_serial_one_thread(
    device const uint* csr_cols     [[buffer(0)]],
    device const uchar* csr_signs   [[buffer(1)]],
    device const half* csr_scale    [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float* output            [[buffer(4)]],
    constant uint& nnz              [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    uint tid                        [[thread_position_in_grid]])
{
    if (tid != 0u) return;
    const float rscale = float(csr_scale[0]);
    for (uint n = 0u; n < nnz; ++n) {
        const uint bit = (csr_signs[n >> 3u] >> (n & 7u)) & 1u;
        const float v = bit ? rscale : -rscale;
        const uint col = csr_cols[n];
        // scatter-add into a single scalar sink so this times the extract
        output[0] += v * input[col];
    }
    (void)cols;
}

// Serial 1-thread-per-row uniform (the Q80 artifact, for contrast).
kernel void disc_uniform_bits_serial(
    device const uchar* codes   [[buffer(0)]],
    device const half* scales   [[buffer(1)]],
    device const float* input   [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& rows         [[buffer(4)]],
    constant uint& cols         [[buffer(5)]],
    constant uint& group_size   [[buffer(6)]],
    constant uint& bits         [[buffer(7)]],
    constant uint& bound        [[buffer(8)]],
    uint row                    [[thread_position_in_grid]])
{
    if (row >= rows) return;
    float acc = 0.0f;
    const uint row_base = row * cols;
    const int ib = int(bound);
    for (uint col = 0u; col < cols; ++col) {
        acc += uniform_value(codes, scales, row_base + col, group_size, bits, ib) * input[col];
    }
    output[row] = acc;
}

// Normalized Walsh-Hadamard of x in groups of 128. One thread per group.
kernel void disc_walsh_hadamard_x(
    device const float* x   [[buffer(0)]],
    device float* y         [[buffer(1)]],
    constant uint& n        [[buffer(2)]],
    uint gid                [[thread_position_in_grid]])
{
    constexpr uint G = 128u;
    const uint ng = n / G;
    if (gid >= ng) return;
    float v[128];
    const uint base = gid * G;
    for (uint i = 0u; i < G; ++i) v[i] = x[base + i];
    for (uint stride = 1u; stride < G; stride <<= 1u) {
        for (uint i = 0u; i < G; i += (stride << 1u)) {
            for (uint j = 0u; j < stride; ++j) {
                const float a = v[i + j];
                const float b = v[i + j + stride];
                v[i + j] = a + b;
                v[i + j + stride] = a - b;
            }
        }
    }
    const float s = 0.08838834764831845f; // 1/sqrt(128)
    for (uint i = 0u; i < G; ++i) y[base + i] = v[i] * s;
}

// Two-stage y = L @ (R @ x). Two dispatches timed in one CB by the host.
// This kernel is just uniform_bits on whatever (rows, cols) is bound.

// tg256 variant (Q80-won occupancy): 256 threads/row, each lane 8 cols.
kernel void disc_uniform_bits_tg256(
    device const uchar* codes   [[buffer(0)]],
    device const half* scales   [[buffer(1)]],
    device const float* input   [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& rows         [[buffer(4)]],
    constant uint& cols         [[buffer(5)]],
    constant uint& group_size   [[buffer(6)]],
    constant uint& bits         [[buffer(7)]],
    constant uint& bound        [[buffer(8)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint lid                    [[thread_index_in_threadgroup]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    const uint row = group_id;
    if (row >= rows) return;
    float acc = 0.0f;
    const uint row_base = row * cols;
    const int ib = int(bound);
    for (uint col = lid * 8u; col < cols; col += 2048u) {
        const uint n = min(8u, cols - col);
        for (uint k = 0u; k < n; ++k) {
            acc += uniform_value(codes, scales, row_base + col + k, group_size, bits, ib)
                * input[col + k];
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float s = 0.0f;
        for (uint i = 0u; i < 8u; ++i) s += red[i];
        output[row] = s;
    }
}

// 4 x 2-bit codes per byte, little-endian. col must be 4-aligned.
static inline float q2_byte4(
    device const uchar* codes,
    device const half* scales,
    device const float* input,
    uint element0,
    uint col,
    uint group_size,
    int bound)
{
    const uchar b = codes[(element0 * 2u) >> 3u];
    float acc = 0.0f;
    for (uint k = 0u; k < 4u; ++k) {
        const int q = int((b >> (2u * k)) & 3u) - bound;
        const float scale = float(scales[(element0 + k) / group_size]);
        acc += float(q) * scale * input[col + k];
    }
    return acc;
}

// 8 x 3-bit codes in 3 bytes (24 bits). element0*3 must be byte-aligned.
static inline float q3_pack8(
    device const uchar* codes,
    device const half* scales,
    device const float* input,
    uint element0,
    uint col,
    uint group_size,
    int bound)
{
    const uint bit0 = element0 * 3u;
    const uint byte0 = bit0 >> 3u;
    const uint packed = uint(codes[byte0])
        | (uint(codes[byte0 + 1u]) << 8u)
        | (uint(codes[byte0 + 2u]) << 16u);
    float acc = 0.0f;
    for (uint k = 0u; k < 8u; ++k) {
        const int q = int((packed >> (3u * k)) & 7u) - bound;
        const float scale = float(scales[(element0 + k) / group_size]);
        acc += float(q) * scale * input[col + k];
    }
    return acc;
}

kernel void disc_uniform_q2_bytes_tpr64(
    device const uchar* codes   [[buffer(0)]],
    device const half* scales   [[buffer(1)]],
    device const float* input   [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& rows         [[buffer(4)]],
    constant uint& cols         [[buffer(5)]],
    constant uint& group_size   [[buffer(6)]],
    constant uint& bits         [[buffer(7)]],
    constant uint& bound        [[buffer(8)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    TPR64_PREAMBLE
    float acc = 0.0f;
    if (row < rows && bits == 2u) {
        const uint row_base = row * cols;
        const int ib = int(bound);
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            acc += q2_byte4(codes, scales, input, row_base + col, col, group_size, ib);
            acc += q2_byte4(codes, scales, input, row_base + col + 4u, col + 4u, group_size, ib);
        }
    }
    TPR64_REDUCE_STORE
}

kernel void disc_uniform_q3_bytes_tpr64(
    device const uchar* codes   [[buffer(0)]],
    device const half* scales   [[buffer(1)]],
    device const float* input   [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& rows         [[buffer(4)]],
    constant uint& cols         [[buffer(5)]],
    constant uint& group_size   [[buffer(6)]],
    constant uint& bits         [[buffer(7)]],
    constant uint& bound        [[buffer(8)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    TPR64_PREAMBLE
    float acc = 0.0f;
    if (row < rows && bits == 3u) {
        const uint row_base = row * cols;
        const int ib = int(bound);
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            acc += q3_pack8(codes, scales, input, row_base + col, col, group_size, ib);
        }
    }
    TPR64_REDUCE_STORE
}

// Ternary: 2-bit codes, 0→0, 1→+s, 2→-s. 4 codes/byte.
kernel void disc_ternary_bytes_tpr64(
    device const uchar* codes   [[buffer(0)]],
    device const half* scales   [[buffer(1)]],
    device const float* input   [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& rows         [[buffer(4)]],
    constant uint& cols         [[buffer(5)]],
    constant uint& group_size   [[buffer(6)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    TPR64_PREAMBLE
    float acc = 0.0f;
    if (row < rows) {
        const uint row_base = row * cols;
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            for (uint off = 0u; off < 8u; off += 4u) {
                const uint element0 = row_base + col + off;
                const uchar b = codes[(element0 * 2u) >> 3u];
                for (uint k = 0u; k < 4u; ++k) {
                    const uint code = (b >> (2u * k)) & 3u;
                    if (code == 0u) continue;
                    const float scale = float(scales[(element0 + k) / group_size]);
                    acc += ((code == 1u) ? scale : -scale) * input[col + off + k];
                }
            }
        }
    }
    TPR64_REDUCE_STORE
}

kernel void disc_additive_bytes_tpr64(
    device const uchar* base_codes  [[buffer(0)]],
    device const half* base_scales  [[buffer(1)]],
    device const uchar* res_codes   [[buffer(2)]],
    device const half* res_scales   [[buffer(3)]],
    device const float* input       [[buffer(4)]],
    device float* output            [[buffer(5)]],
    constant uint& rows             [[buffer(6)]],
    constant uint& cols             [[buffer(7)]],
    constant uint& group_size       [[buffer(8)]],
    uint group_id                   [[threadgroup_position_in_grid]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]])
{
    TPR64_PREAMBLE
    float acc = 0.0f;
    if (row < rows) {
        const uint row_base = row * cols;
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            for (uint off = 0u; off < 8u; off += 4u) {
                const uint element0 = row_base + col + off;
                const uchar bb = base_codes[(element0 * 2u) >> 3u];
                const uchar rb = res_codes[(element0 * 2u) >> 3u];
                for (uint k = 0u; k < 4u; ++k) {
                    const uint bc = (bb >> (2u * k)) & 3u;
                    const uint rc = (rb >> (2u * k)) & 3u;
                    const uint group = (element0 + k) / group_size;
                    const float bv = (float(bc) - 1.5f) * float(base_scales[group]);
                    const float rv = (float(rc) - 1.5f) * float(res_scales[group]);
                    acc += (bv + rv) * input[col + off + k];
                }
            }
        }
    }
    TPR64_REDUCE_STORE
}

kernel void disc_binary_tg256(
    device const uchar* signs   [[buffer(0)]],
    device const half* scales   [[buffer(1)]],
    device const float* input   [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& rows         [[buffer(4)]],
    constant uint& cols         [[buffer(5)]],
    constant uint& group_size   [[buffer(6)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint lid                    [[thread_index_in_threadgroup]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    const uint row = group_id;
    if (row >= rows) return;
    float acc = 0.0f;
    const uint row_base = row * cols;
    for (uint col = lid * 8u; col < cols; col += 2048u) {
        const float scale = float(scales[(row_base + col) / group_size]);
        const uchar byte = signs[(row_base + col) >> 3u];
        acc += ((byte & 0x01u) ? scale : -scale) * input[col];
        acc += ((byte & 0x02u) ? scale : -scale) * input[col + 1u];
        acc += ((byte & 0x04u) ? scale : -scale) * input[col + 2u];
        acc += ((byte & 0x08u) ? scale : -scale) * input[col + 3u];
        acc += ((byte & 0x10u) ? scale : -scale) * input[col + 4u];
        acc += ((byte & 0x20u) ? scale : -scale) * input[col + 5u];
        acc += ((byte & 0x40u) ? scale : -scale) * input[col + 6u];
        acc += ((byte & 0x80u) ? scale : -scale) * input[col + 7u];
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float s = 0.0f;
        for (uint i = 0u; i < 8u; ++i) s += red[i];
        output[row] = s;
    }
}

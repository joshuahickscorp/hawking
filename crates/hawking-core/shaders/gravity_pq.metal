// Product-quantized matvec for `.gravity` `gravity-pq` tensors.
//
// The artifact never materializes a dense weight: each output row is a sum over `nchunk`
// column-chunks, and each chunk contributes one codebook entry (`sub` values) dotted
// against its slice of x. Reading the codebook entry is the whole dequantization, so the
// bytes this kernel touches are exactly the bytes the BPW ledger bills -- which is the
// property that makes a sub-bit artifact cheaper to RUN, not merely cheaper to store.
//
// Authority for the arithmetic is `gravity_forge.pq_execute`; the frozen fixtures under
// tests/fixtures/gravity_pq are what a change here has to keep passing.

#include <metal_stdlib>
using namespace metal;

struct GravityPQParams {
    uint dim;      // D, columns per chunk; D == subspaces * sub
    uint subspaces;// S
    uint sub;      // values per codebook entry
    uint card;     // codebook cardinality
    uint rows;
    uint cols;     // == nchunk * D
    uint nchunk;
    uint bits;     // index width, MSB-first in one contiguous stream
};

// Additive residual product quantization used by `llama.residual-pq.v1`.
// Every (row, chunk) has one index per stage; codebook values add directly,
// so execution never expands a dense row or a temporary residual tensor.
struct GravityResidualPQParams {
    uint dim;
    uint stages;
    uint card;
    uint rows;
    uint cols;
    uint nchunk;
    uint bits;
    uint reserved;
};

// Exact source-layout legacy quant grammar for Qwen-family tensors which
// retain Q5_0/Q8_0 blocks.  One SIMD group owns an output row, keeping the
// source packed bytes on-device and reducing the row with simd_sum.  The
// caller records this in the same token command buffer as the K-quant paths;
// there is no decode-to-dense staging buffer.
struct GravityRaw32Params {
    uint rows;
    uint cols;
};

static inline float gravity_fp16_at(const device uchar *p, uint64_t off) {
    ushort bits = ushort(p[off]) | (ushort(p[off + 1u]) << 8u);
    return float(as_type<half>(bits));
}

kernel void gravity_raw_q8_0_matvec(
    const device uchar *weights [[buffer(0)]],
    const device float *x        [[buffer(1)]],
    device float *y              [[buffer(2)]],
    constant GravityRaw32Params &p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg_in_tg [[simdgroup_index_in_threadgroup]],
    uint sgs_per_tg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint row = tgid * sgs_per_tg + sg_in_tg;
    if (row >= p.rows) return;
    uint blocks = p.cols / 32u;
    uint64_t row_off = uint64_t(row) * uint64_t(blocks) * 34ul;
    float sum = 0.0f;
    for (uint c = lane; c < p.cols; c += 32u) {
        uint64_t bo = row_off + uint64_t(c >> 5u) * 34ul;
        int q = int(weights[bo + 2ul + uint64_t(c & 31u)]);
        if (q >= 128) q -= 256;
        sum = fma(gravity_fp16_at(weights, bo) * float(q), x[c], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) y[row] = sum;
}

kernel void gravity_raw_q5_0_matvec(
    const device uchar *weights [[buffer(0)]],
    const device float *x        [[buffer(1)]],
    device float *y              [[buffer(2)]],
    constant GravityRaw32Params &p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg_in_tg [[simdgroup_index_in_threadgroup]],
    uint sgs_per_tg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint row = tgid * sgs_per_tg + sg_in_tg;
    if (row >= p.rows) return;
    uint blocks = p.cols / 32u;
    uint64_t row_off = uint64_t(row) * uint64_t(blocks) * 22ul;
    float sum = 0.0f;
    for (uint c = lane; c < p.cols; c += 32u) {
        uint64_t bo = row_off + uint64_t(c >> 5u) * 22ul;
        uint qh = uint(weights[bo + 2ul])
                | (uint(weights[bo + 3ul]) << 8u)
                | (uint(weights[bo + 4ul]) << 16u)
                | (uint(weights[bo + 5ul]) << 24u);
        uchar packed = weights[bo + 6ul + uint64_t(c & 15u)];
        // `c` is absolute across the row.  The nibble selection is local to
        // each 32-element GGML block; using `c < 16` accidentally selected
        // the upper nibble for every lane after block zero.
        uint in_block = c & 31u;
        uint low = in_block < 16u ? (uint(packed) & 0x0fu) : (uint(packed) >> 4u);
        int q = int(low | (((qh >> (c & 31u)) & 1u) << 4u)) - 16;
        sum = fma(gravity_fp16_at(weights, bo) * float(q), x[c], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) y[row] = sum;
}

// Pair two exact source Q5_0 projections without materializing either tensor.
// The two output waves share the activation and one command topology, while
// each SIMD group retains the source row grammar and reduction order.
static inline float gravity_raw_q5_0_dot_row(
    const device uchar *weights,
    const device float *x,
    uint row,
    uint cols,
    uint lane);

struct GravityRawQ5PairParams {
    uint rows;
    uint cols;
};

kernel void gravity_raw_q5_0_pair_matvec(
    const device uchar *gate_weights [[buffer(0)]],
    const device uchar *up_weights [[buffer(1)]],
    const device float *x [[buffer(2)]],
    device float *gate_out [[buffer(3)]],
    device float *up_out [[buffer(4)]],
    constant GravityRawQ5PairParams &p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint groups_per_wave = (p.rows + 7u) / 8u;
    bool is_gate = tgid < groups_per_wave;
    uint row = (is_gate ? tgid : tgid - groups_per_wave) * 8u + sg;
    if (row >= p.rows) return;
    float sum = gravity_raw_q5_0_dot_row(is_gate ? gate_weights : up_weights, x, row, p.cols, lane);
    if (lane == 0u) {
        (is_gate ? gate_out : up_out)[row] = sum;
    }
}

// Source-preserving Qwen2 projection wave. Q and K use Q5_0; source V may
// be either Q5_0 or Q8_0. One command encodes all three projections, applies
// their native biases, rotates Q/K from the artifact's already-authoritative
// table, and writes K/V directly into the current cache slot. This avoids a
// dense staging buffer and replaces the decomposed Q/K/V, bias, RoPE, and
// append wave only when the source grammar exactly admits it.
struct GravityRawQ5Q5QvRopeAppendParams {
    uint q_rows;
    uint kv_rows;
    uint cols;
    uint kv_off;
    uint head_dim;
    uint has_q_bias;
    uint has_k_bias;
    uint has_v_bias;
    uint v_is_q8;
};

static inline float gravity_raw_q5_0_dot_row(
    const device uchar *weights,
    const device float *x,
    uint row,
    uint cols,
    uint lane)
{
    uint blocks = cols / 32u;
    uint64_t row_off = uint64_t(row) * uint64_t(blocks) * 22ul;
    float sum = 0.0f;
    for (uint c = lane; c < cols; c += 32u) {
        uint64_t bo = row_off + uint64_t(c >> 5u) * 22ul;
        uint qh = uint(weights[bo + 2ul])
                | (uint(weights[bo + 3ul]) << 8u)
                | (uint(weights[bo + 4ul]) << 16u)
                | (uint(weights[bo + 5ul]) << 24u);
        uchar packed = weights[bo + 6ul + uint64_t(c & 15u)];
        uint in_block = c & 31u;
        uint low = in_block < 16u ? (uint(packed) & 0x0fu) : (uint(packed) >> 4u);
        int q = int(low | (((qh >> in_block) & 1u) << 4u)) - 16;
        sum = fma(gravity_fp16_at(weights, bo) * float(q), x[c], sum);
    }
    return simd_sum(sum);
}

static inline float gravity_raw_q8_0_dot_row(
    const device uchar *weights,
    const device float *x,
    uint row,
    uint cols,
    uint lane)
{
    uint blocks = cols / 32u;
    uint64_t row_off = uint64_t(row) * uint64_t(blocks) * 34ul;
    float sum = 0.0f;
    for (uint c = lane; c < cols; c += 32u) {
        uint64_t bo = row_off + uint64_t(c >> 5u) * 34ul;
        int q = int(weights[bo + 2ul + uint64_t(c & 31u)]);
        if (q >= 128) q -= 256;
        sum = fma(gravity_fp16_at(weights, bo) * float(q), x[c], sum);
    }
    return simd_sum(sum);
}

// Source-compatible llama.cpp b9430 geometry candidates.  These are opt-in
// kernels: the existing raw kernels remain the Hawking default until a
// same-model wall/p99 receipt promotes this topology.  Q5_0 uses two SIMD
// groups, four rows per group; Q8_0 uses four SIMD groups, two rows per group
// and a small cross-SIMD reduction, matching ggml-metal's N_R0/N_SG choices.
// The arithmetic still reads the exact GGML bytes directly from the mmap.
kernel void gravity_raw_q5_0_llama_matvec(
    const device uchar *weights [[buffer(0)]],
    const device float *x        [[buffer(1)]],
    device float *y              [[buffer(2)]],
    constant GravityRaw32Params &p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg_in_tg [[simdgroup_index_in_threadgroup]],
    uint sgs_per_tg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint rows_per_sg = 4u;
    const uint row0 = (tgid * sgs_per_tg + sg_in_tg) * rows_per_sg;
    const uint blocks = p.cols / 32u;
    // llama.cpp's Q5_0 path assigns two lanes to each 32-value block and
    // lets each lane consume one half-block (16 values).  The loop below is
    // the same assignment expressed in eight-value chunks.
    const uint block0 = (lane >> 1u);
    const uint in0 = (lane & 1u) * 8u;
    for (uint r = 0u; r < rows_per_sg; ++r) {
        uint row = row0 + r;
        if (row >= p.rows) { continue; }
        uint64_t row_off = uint64_t(row) * uint64_t(blocks) * 22ul;
        float sum = 0.0f;
        for (uint block = block0; block < blocks; block += 16u) {
            uint64_t bo = row_off + uint64_t(block) * 22ul;
            uint qh = uint(weights[bo + 2ul])
                    | (uint(weights[bo + 3ul]) << 8u)
                    | (uint(weights[bo + 4ul]) << 16u)
                    | (uint(weights[bo + 5ul]) << 24u);
            float d = gravity_fp16_at(weights, bo);
            for (uint j = 0u; j < 8u; ++j) {
                uint in_block = in0 + j;
                uchar packed = weights[bo + 6ul + uint64_t(in_block & 15u)];
                uint low = in_block < 16u
                    ? (uint(packed) & 0x0fu)
                    : (uint(packed) >> 4u);
                int q = int(low | (((qh >> in_block) & 1u) << 4u)) - 16;
                uint c = block * 32u + in_block;
                sum = fma(d * float(q), x[c], sum);
            }
        }
        sum = simd_sum(sum);
        if (lane == 0u) { y[row] = sum; }
    }
}

kernel void gravity_raw_q8_0_llama_matvec(
    const device uchar *weights [[buffer(0)]],
    const device float *x        [[buffer(1)]],
    device float *y              [[buffer(2)]],
    constant GravityRaw32Params &p [[buffer(3)]],
    threadgroup float *shmem [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg_in_tg [[simdgroup_index_in_threadgroup]],
    uint sgs_per_tg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint rows_per_tg = 2u;
    const uint row0 = tgid * rows_per_tg;
    const uint blocks = p.cols / 32u;
    const uint ix = lane >> 2u;       // four lanes per block
    const uint il = (lane & 3u) * 8u; // eight Q8 values per lane
    float partial[rows_per_tg] = { 0.0f, 0.0f };
    for (uint block = sg_in_tg * 8u + ix; block < blocks; block += sgs_per_tg * 8u) {
        uint64_t block_off = uint64_t(block) * 34ul;
        uint c0 = block * 32u + il;
        for (uint r = 0u; r < rows_per_tg; ++r) {
            uint row = row0 + r;
            if (row >= p.rows) { continue; }
            uint64_t bo = uint64_t(row) * uint64_t(blocks) * 34ul + block_off;
            float d = gravity_fp16_at(weights, bo);
            for (uint j = 0u; j < 8u; ++j) {
                int q = int(weights[bo + 2ul + uint64_t(il + j)]);
                if (q >= 128) { q -= 256; }
                partial[r] = fma(d * float(q), x[c0 + j], partial[r]);
            }
        }
    }

    // This is helper_mv_reduce_and_write from ggml-metal, specialized to two
    // rows and four SIMD groups.  SG0 clears all reduction slots before the
    // cross-SG barrier; every SG then contributes one partial value per row.
    for (uint r = 0u; r < rows_per_tg; ++r) {
        if (sg_in_tg == 0u) { shmem[r * 32u + lane] = 0.0f; }
        partial[r] = simd_sum(partial[r]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0u; r < rows_per_tg; ++r) {
        if (lane == 0u) { shmem[r * 32u + sg_in_tg] = partial[r]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0u; r < rows_per_tg; ++r) {
        uint row = row0 + r;
        if (row >= p.rows) { continue; }
        float total = simd_sum(shmem[r * 32u + lane]);
        if (lane == 0u && sg_in_tg == 0u) { y[row] = total; }
    }
}

kernel void gravity_raw_q5q5qv_rope_append(
    const device uchar *wq [[buffer(0)]],
    const device uchar *wk [[buffer(1)]],
    const device uchar *wv [[buffer(2)]],
    const device float *x [[buffer(3)]],
    device float *q_out [[buffer(4)]],
    device float *k_cache [[buffer(5)]],
    device float *v_cache [[buffer(6)]],
    const device float *rope [[buffer(7)]],
    const device float *q_bias [[buffer(8)]],
    const device float *k_bias [[buffer(9)]],
    const device float *v_bias [[buffer(10)]],
    constant GravityRawQ5Q5QvRopeAppendParams &p [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint half_dim = p.head_dim / 2u;
    uint q_pairs = p.q_rows / 2u;
    uint k_pairs = p.kv_rows / 2u;
    uint q_tg = (q_pairs + 7u) / 8u;
    uint k_tg = (k_pairs + 7u) / 8u;
    if (tgid < q_tg) {
        uint pair = tgid * 8u + sg;
        if (pair >= q_pairs) return;
        uint head = pair / half_dim;
        uint within = pair - head * half_dim;
        uint row0 = head * p.head_dim + within;
        uint row1 = row0 + half_dim;
        float a = gravity_raw_q5_0_dot_row(wq, x, row0, p.cols, lane);
        float b = gravity_raw_q5_0_dot_row(wq, x, row1, p.cols, lane);
        if (lane == 0u) {
            if (p.has_q_bias != 0u) { a += q_bias[row0]; b += q_bias[row1]; }
            float c = rope[within];
            float s = rope[half_dim + within];
            q_out[row0] = a * c - b * s;
            q_out[row1] = a * s + b * c;
        }
    } else if (tgid < q_tg + k_tg) {
        uint pair = (tgid - q_tg) * 8u + sg;
        if (pair >= k_pairs) return;
        uint head = pair / half_dim;
        uint within = pair - head * half_dim;
        uint row0 = head * p.head_dim + within;
        uint row1 = row0 + half_dim;
        float a = gravity_raw_q5_0_dot_row(wk, x, row0, p.cols, lane);
        float b = gravity_raw_q5_0_dot_row(wk, x, row1, p.cols, lane);
        if (lane == 0u) {
            if (p.has_k_bias != 0u) { a += k_bias[row0]; b += k_bias[row1]; }
            float c = rope[within];
            float s = rope[half_dim + within];
            k_cache[p.kv_off + row0] = a * c - b * s;
            k_cache[p.kv_off + row1] = a * s + b * c;
        }
    } else {
        uint row = (tgid - q_tg - k_tg) * 8u + sg;
        if (row >= p.kv_rows) return;
        float value = p.v_is_q8 != 0u
            ? gravity_raw_q8_0_dot_row(wv, x, row, p.cols, lane)
            : gravity_raw_q5_0_dot_row(wv, x, row, p.cols, lane);
        if (lane == 0u) {
            if (p.has_v_bias != 0u) value += v_bias[row];
            v_cache[p.kv_off + row] = value;
        }
    }
}

// One index out of the packed stream. The stream is MSB-first, so value i occupies bit
// range [i*bits, (i+1)*bits) counting from the high bit of byte 0. `codes` is uploaded
// with four bytes of tail padding so this always has a whole word to read.
static inline uint pq_index(const device uchar *codes, uint i, uint bits) {
    uint bitoff = i * bits;
    uint byte = bitoff >> 3u;
    uint shift = bitoff & 7u;
    uint word = (uint(codes[byte]) << 24) | (uint(codes[byte + 1u]) << 16)
              | (uint(codes[byte + 2u]) << 8) | uint(codes[byte + 3u]);
    return (word >> (32u - shift - bits)) & ((1u << bits) - 1u);
}

// One SIMD group per output row: the 32 lanes stride over chunks, so consecutive lanes
// read consecutive index words, and the per-row reduction is a single simd_sum rather
// than a threadgroup barrier.
kernel void gravity_pq_matvec(
    const device half         *codebooks [[buffer(0)]],
    const device uchar        *codes     [[buffer(1)]],
    const device float        *x         [[buffer(2)]],
    device float              *y         [[buffer(3)]],
    constant GravityPQParams  &p         [[buffer(4)]],
    uint  tgid                           [[threadgroup_position_in_grid]],
    uint  sg_in_tg                       [[simdgroup_index_in_threadgroup]],
    uint  sgs_per_tg                     [[simdgroups_per_threadgroup]],
    uint  lane                           [[thread_index_in_simdgroup]])
{
    uint row = tgid * sgs_per_tg + sg_in_tg;
    if (row >= p.rows) { return; }

    float acc = 0.0f;
    for (uint s = 0; s < p.subspaces; ++s) {
        const device half *cb = codebooks + s * p.card * p.sub;
        const uint xbase = s * p.sub;
        for (uint c = lane; c < p.nchunk; c += 32u) {
            uint flat = (row * p.nchunk + c) * p.subspaces + s;
            const device half *entry = cb + pq_index(codes, flat, p.bits) * p.sub;
            const device float *xs = x + c * p.dim + xbase;
            for (uint j = 0; j < p.sub; ++j) {
                acc = fma(float(entry[j]), xs[j], acc);
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) { y[row] = acc; }
}

kernel void gravity_residual_pq_matvec(
    const device half                 *codebooks [[buffer(0)]],
    const device uchar                *codes     [[buffer(1)]],
    const device float                *x         [[buffer(2)]],
    device float                      *y         [[buffer(3)]],
    constant GravityResidualPQParams  &p         [[buffer(4)]],
    uint  tgid                                   [[threadgroup_position_in_grid]],
    uint  sg_in_tg                               [[simdgroup_index_in_threadgroup]],
    uint  sgs_per_tg                             [[simdgroups_per_threadgroup]],
    uint  lane                                   [[thread_index_in_simdgroup]])
{
    uint row = tgid * sgs_per_tg + sg_in_tg;
    if (row >= p.rows) { return; }

    float acc = 0.0f;
    for (uint c = lane; c < p.nchunk; c += 32u) {
        const device float *xs = x + c * p.dim;
        for (uint stage = 0; stage < p.stages; ++stage) {
            uint flat = (row * p.nchunk + c) * p.stages + stage;
            const device half *entry = codebooks
                + (stage * p.card + pq_index(codes, flat, p.bits)) * p.dim;
            for (uint j = 0; j < p.dim; ++j) {
                acc = fma(float(entry[j]), xs[j], acc);
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) { y[row] = acc; }
}

// ---------------------------------------------------------------------------
// Additive bits=8/autotune lane.
//
// `gravity_pq_matvec` above is deliberately unchanged and remains the runtime
// default.  The kernels below are selected only through the explicit
// `PqMetalKernelVariant` API.  Production GLM gravity-pq tensors overwhelmingly
// use D=32, S=1, sub=32, card=256, bits=8, so their indices are already bytes:
// the generic four-byte MSB window is pure overhead for that geometry.
// ---------------------------------------------------------------------------

// Four independent vector accumulators shorten the dependency chain from 32
// scalar FMAs per chunk to two vector FMAs per accumulator for sub=32.  The
// host registry admits this helper only when `sub` and `dim` are multiples of
// four, which guarantees aligned half4/float4 entry points.
static inline float pq_bits8_vec4_lane(
    const device half  *codebooks,
    const device uchar *codes,
    const device float *x,
    constant GravityPQParams &p,
    uint row,
    uint first_chunk,
    uint chunk_stride)
{
    float4 acc0 = 0.0f;
    float4 acc1 = 0.0f;
    float4 acc2 = 0.0f;
    float4 acc3 = 0.0f;
    for (uint s = 0; s < p.subspaces; ++s) {
        const device half *cb = codebooks + s * p.card * p.sub;
        const uint xbase = s * p.sub;
        for (uint c = first_chunk; c < p.nchunk; c += chunk_stride) {
            uint flat = (row * p.nchunk + c) * p.subspaces + s;
            const device half *entry = cb + uint(codes[flat]) * p.sub;
            const device float *xs = x + c * p.dim + xbase;
            const device half4 *entry4 =
                reinterpret_cast<const device half4 *>(entry);
            const device float4 *xs4 =
                reinterpret_cast<const device float4 *>(xs);
            uint nvec = p.sub >> 2u;
            for (uint q = 0u; q < nvec; q += 4u) {
                if (q < nvec) {
                    acc0 = fma(float4(entry4[q]), xs4[q], acc0);
                }
                if (q + 1u < nvec) {
                    acc1 = fma(float4(entry4[q + 1u]), xs4[q + 1u], acc1);
                }
                if (q + 2u < nvec) {
                    acc2 = fma(float4(entry4[q + 2u]), xs4[q + 2u], acc2);
                }
                if (q + 3u < nvec) {
                    acc3 = fma(float4(entry4[q + 3u]), xs4[q + 3u], acc3);
                }
            }
        }
    }
    float4 v = (acc0 + acc1) + (acc2 + acc3);
    return (v.x + v.y) + (v.z + v.w);
}

// Two-float expansion used only by the unpromoted bits8-double-single
// candidate. `hi` holds the rounded leading value and `lo` its residual.
// This intentionally spends substantially more arithmetic/registers than the
// ordinary FMA path; it carries no throughput claim until a manual bounded
// exact-geometry sweep measures it.
struct PqDoubleSingle {
    float hi;
    float lo;
};

static inline PqDoubleSingle pq_ds_product(float a, float b)
{
    PqDoubleSingle out;
    volatile float hi = a * b;
    out.hi = hi;
    out.lo = metal::precise::fma(a, b, -hi);
    return out;
}

// Error-free TwoSum on the leading terms followed by a hi/lo renormalization.
// The operation order matches the CPU preflight model exactly.
static inline PqDoubleSingle pq_ds_add(PqDoubleSingle lhs, PqDoubleSingle rhs)
{
    volatile float sum = lhs.hi + rhs.hi;
    volatile float rhs_virtual = sum - lhs.hi;
    volatile float sum_error =
        (lhs.hi - (sum - rhs_virtual)) + (rhs.hi - rhs_virtual);
    volatile float tail = (lhs.lo + rhs.lo) + sum_error;
    PqDoubleSingle out;
    out.hi = sum + tail;
    volatile float hi_delta = out.hi - sum;
    out.lo = tail - hi_delta;
    return out;
}

// Fixed 32-lane tree: 0+16, 1+17, ...; then 0+8, ... down to 0+1.
// Every lane executes each shuffle; only the lower half updates. This avoids
// implementation-defined simd_sum reassociation and matches the CPU model's
// explicit tree.
static inline PqDoubleSingle pq_ds_simd_tree(
    PqDoubleSingle acc,
    uint lane)
{
    PqDoubleSingle rhs;
    rhs.hi = simd_shuffle_down(acc.hi, ushort(16));
    rhs.lo = simd_shuffle_down(acc.lo, ushort(16));
    if (lane < 16u) { acc = pq_ds_add(acc, rhs); }

    rhs.hi = simd_shuffle_down(acc.hi, ushort(8));
    rhs.lo = simd_shuffle_down(acc.lo, ushort(8));
    if (lane < 8u) { acc = pq_ds_add(acc, rhs); }

    rhs.hi = simd_shuffle_down(acc.hi, ushort(4));
    rhs.lo = simd_shuffle_down(acc.lo, ushort(4));
    if (lane < 4u) { acc = pq_ds_add(acc, rhs); }

    rhs.hi = simd_shuffle_down(acc.hi, ushort(2));
    rhs.lo = simd_shuffle_down(acc.lo, ushort(2));
    if (lane < 2u) { acc = pq_ds_add(acc, rhs); }

    rhs.hi = simd_shuffle_down(acc.hi, ushort(1));
    rhs.lo = simd_shuffle_down(acc.lo, ushort(1));
    if (lane < 1u) { acc = pq_ds_add(acc, rhs); }
    return acc;
}

// Direct byte lookup while retaining the default kernel's scalar FMA shape.
// This isolates the cost of generic packed extraction from every other change.
kernel void gravity_pq_matvec_bits8_direct(
    const device half         *codebooks [[buffer(0)]],
    const device uchar        *codes     [[buffer(1)]],
    const device float        *x         [[buffer(2)]],
    device float              *y         [[buffer(3)]],
    constant GravityPQParams  &p         [[buffer(4)]],
    uint  tgid                           [[threadgroup_position_in_grid]],
    uint  sg_in_tg                       [[simdgroup_index_in_threadgroup]],
    uint  sgs_per_tg                     [[simdgroups_per_threadgroup]],
    uint  lane                           [[thread_index_in_simdgroup]])
{
    uint row = tgid * sgs_per_tg + sg_in_tg;
    if (row >= p.rows) { return; }

    float acc = 0.0f;
    for (uint s = 0; s < p.subspaces; ++s) {
        const device half *cb = codebooks + s * p.card * p.sub;
        const uint xbase = s * p.sub;
        for (uint c = lane; c < p.nchunk; c += 32u) {
            uint flat = (row * p.nchunk + c) * p.subspaces + s;
            const device half *entry = cb + uint(codes[flat]) * p.sub;
            const device float *xs = x + c * p.dim + xbase;
            for (uint j = 0; j < p.sub; ++j) {
                acc = fma(float(entry[j]), xs[j], acc);
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) { y[row] = acc; }
}

// Numerically strengthened direct-byte candidate. Each product is represented
// by its rounded value plus FMA residual, accumulated as a double-single
// expansion, then reduced through the fixed compensated lane tree above.
// This is explicit/autotune-only; the production default remains unchanged.
kernel void gravity_pq_matvec_bits8_double_single(
    const device half         *codebooks [[buffer(0)]],
    const device uchar        *codes     [[buffer(1)]],
    const device float        *x         [[buffer(2)]],
    device float              *y         [[buffer(3)]],
    constant GravityPQParams  &p         [[buffer(4)]],
    uint  tgid                           [[threadgroup_position_in_grid]],
    uint  sg_in_tg                       [[simdgroup_index_in_threadgroup]],
    uint  sgs_per_tg                     [[simdgroups_per_threadgroup]],
    uint  lane                           [[thread_index_in_simdgroup]])
{
    uint row = tgid * sgs_per_tg + sg_in_tg;
    if (row >= p.rows) { return; }

    PqDoubleSingle acc = { 0.0f, 0.0f };
    for (uint s = 0; s < p.subspaces; ++s) {
        const device half *cb = codebooks + s * p.card * p.sub;
        const uint xbase = s * p.sub;
        for (uint c = lane; c < p.nchunk; c += 32u) {
            uint flat = (row * p.nchunk + c) * p.subspaces + s;
            const device half *entry = cb + uint(codes[flat]) * p.sub;
            const device float *xs = x + c * p.dim + xbase;
            for (uint j = 0; j < p.sub; ++j) {
                acc = pq_ds_add(
                    acc, pq_ds_product(float(entry[j]), xs[j]));
            }
        }
    }
    acc = pq_ds_simd_tree(acc, lane);
    if (lane == 0u) { y[row] = acc.hi + acc.lo; }
}

// Same row mapping as the default, but with vector loads and four independent
// vector FMA chains.  This lets the sweep distinguish byte extraction from
// arithmetic dependency depth.
kernel void gravity_pq_matvec_bits8_vec4(
    const device half         *codebooks [[buffer(0)]],
    const device uchar        *codes     [[buffer(1)]],
    const device float        *x         [[buffer(2)]],
    device float              *y         [[buffer(3)]],
    constant GravityPQParams  &p         [[buffer(4)]],
    uint  tgid                           [[threadgroup_position_in_grid]],
    uint  sg_in_tg                       [[simdgroup_index_in_threadgroup]],
    uint  sgs_per_tg                     [[simdgroups_per_threadgroup]],
    uint  lane                           [[thread_index_in_simdgroup]])
{
    uint row = tgid * sgs_per_tg + sg_in_tg;
    if (row >= p.rows) { return; }
    float acc = pq_bits8_vec4_lane(
        codebooks, codes, x, p, row, lane, 32u);
    acc = simd_sum(acc);
    if (lane == 0u) { y[row] = acc; }
}

// True 2D row x chunk-slice decomposition.  One SIMD group computes one
// deterministic slice and writes exactly one partial.  A separate kernel
// reduces those partials in ascending slice order, so there is no atomic
// accumulation and repeated runs are bit-stable.
kernel void gravity_pq_matvec_bits8_2d(
    const device half         *codebooks [[buffer(0)]],
    const device uchar        *codes     [[buffer(1)]],
    const device float        *x         [[buffer(2)]],
    device float              *partials  [[buffer(3)]],
    constant GravityPQParams  &p         [[buffer(4)]],
    constant uint             &splits    [[buffer(5)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint  lane                           [[thread_index_in_simdgroup]])
{
    uint row = tgid.x;
    uint split = tgid.y;
    if (row >= p.rows || split >= splits) { return; }
    uint first_chunk = split * 32u + lane;
    uint chunk_stride = splits * 32u;
    float acc = pq_bits8_vec4_lane(
        codebooks, codes, x, p, row, first_chunk, chunk_stride);
    acc = simd_sum(acc);
    if (lane == 0u) {
        partials[row * splits + split] = acc;
    }
}

kernel void gravity_pq_reduce_2d(
    const device float        *partials [[buffer(0)]],
    device float              *y        [[buffer(1)]],
    constant GravityPQParams  &p        [[buffer(2)]],
    constant uint             &splits   [[buffer(3)]],
    uint id                              [[thread_position_in_grid]])
{
    if (id >= p.rows) { return; }
    float acc = 0.0f;
    for (uint split = 0u; split < splits; ++split) {
        acc += partials[id * splits + split];
    }
    y[id] = acc;
}

// ---------------------------------------------------------------------------
// The elementwise ops the .gravity token graph needs in f32.
//
// The shared kernels in common.metal are half-precision (silu_mul) or fold the
// frequency math into the kernel (rope_slice_f32_inplace, plain theta^(2i/d)).
// Neither fits here: the activation path is f32 end to end, and the artifact's
// declared rope_scaling may be any construction the header names -- so the
// frequencies are computed once per position on the host, in f64, and arrive
// as a table. The kernel then applies a rotation it does not have to
// understand, which is what lets llama3, longrope and plain RoPE share it.
// ---------------------------------------------------------------------------

kernel void gravity_silu_mul_f32(
    device const float *gate [[buffer(0)]],
    device const float *up   [[buffer(1)]],
    device       float *out  [[buffer(2)]],
    constant     uint  &n    [[buffer(3)]],
    uint id                  [[thread_position_in_grid]])
{
    if (id >= n) { return; }
    float g = gate[id];
    out[id] = (g / (1.0f + exp(-g))) * up[id];
}

struct GravityRopeParams {
    uint offset;    // f32 element offset of head 0 within `x`
    uint n_heads;
    uint head_dim;
    uint interleaved; // 1: adjacent pairs (Llama NORM), 0: split-half NeoX
};

// `table` is head_dim/2 cosines followed by head_dim/2 sines.  The explicit
// layout bit keeps the legacy Gravity PQ contract intact while allowing a
// source-preserving Llama shard to use GGUF's normal adjacent pairing.
kernel void gravity_rope_table_f32(
    device       float             *x     [[buffer(0)]],
    const device float             *table [[buffer(1)]],
    constant     GravityRopeParams &p     [[buffer(2)]],
    uint id                               [[thread_position_in_grid]])
{
    uint half_dim = p.head_dim / 2u;
    if (id >= p.n_heads * half_dim) { return; }
    uint h = id / half_dim;
    uint i = id - h * half_dim;
    uint b = p.offset + h * p.head_dim + (p.interleaved != 0u ? 2u * i : i);
    float c = table[i];
    float s = table[half_dim + i];
    float x0 = x[b];
    uint b1 = p.interleaved != 0u ? b + 1u : b + half_dim;
    float x1 = x[b1];
    x[b]            = x0 * c - x1 * s;
    x[b1]           = x0 * s + x1 * c;
}

// ---------------------------------------------------------------------------
// GLM-5.2 resident-decode kernels.
//
// These exist so the residual stream, MLA KV cache, DSA indexer, sparse
// attention and router state can stay on device for a whole token. They
// reproduce the host arithmetic in gravity_glm.rs (interleaved-concat RoPE,
// ReLU'd DSA scores, stable top-k, causal sparse attend) closely enough that
// token identity is preserved against the host-state path; they are not a
// licence to redesign the attention algorithm.
// ---------------------------------------------------------------------------

kernel void gravity_add_inplace_f32(
    device       float *x [[buffer(0)]],
    device const float *y [[buffer(1)]],
    constant     uint  &n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) { return; }
    x[id] += y[id];
}

kernel void gravity_axpy_f32(
    device       float *y [[buffer(0)]],
    device const float *x [[buffer(1)]],
    constant     float &a [[buffer(2)]],
    constant     uint  &n [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) { return; }
    y[id] = fma(a, x[id], y[id]);
}

// GLM interleaved RoPE with *concatenated* halves (not NeoX scatter).
// For each head, input is rotary_dim wide; output[0..half) = first components,
// output[half..rotary_dim) = second components. cos/sin are half long.
struct GravityGlmRopeParams {
    uint n_heads;
    uint rotary_dim; // qk_rope_head_dim or the rotated prefix of index_head_dim
    uint in_stride;  // elements between heads in `x` (may exceed rotary_dim)
    uint out_stride; // elements between heads in `out`
};

// Replay-safe form: both base offsets are scalar contents so an ICB can bind
// the full persistent buffers at offset zero while sequence position changes.
struct GravityGlmPositionedRopeParams {
    uint n_heads;
    uint rotary_dim;
    uint in_stride;
    uint out_stride;
    uint input_element_offset;
    uint output_element_offset;
};

kernel void gravity_rope_interleaved_f32(
    device const float *x     [[buffer(0)]],
    device       float *out   [[buffer(1)]],
    device const float *cos   [[buffer(2)]],
    device const float *sin   [[buffer(3)]],
    constant GravityGlmRopeParams &p [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint half_dim = p.rotary_dim / 2u;
    if (id >= p.n_heads * half_dim) { return; }
    uint h = id / half_dim;
    uint i = id - h * half_dim;
    uint in_base = h * p.in_stride;
    uint out_base = h * p.out_stride;
    float first = x[in_base + 2u * i];
    float second = x[in_base + 2u * i + 1u];
    float c = cos[i];
    float s = sin[i];
    out[out_base + i] = first * c - second * s;
    out[out_base + half_dim + i] = second * c + first * s;
}

// Assemble an indexer vector in one pass: rotate the leading rotary_dim
// interleaved components into concatenated halves and preserve every tail
// component. Output may begin at a position offset in the persistent key
// cache, but input and output must not alias.
kernel void gravity_rope_prefix_tail_f32(
    device const float *x     [[buffer(0)]],
    device       float *out   [[buffer(1)]],
    device const float *cos   [[buffer(2)]],
    device const float *sin   [[buffer(3)]],
    constant GravityGlmRopeParams &p [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= p.n_heads * p.out_stride) { return; }
    uint h = id / p.out_stride;
    uint col = id - h * p.out_stride;
    uint in_base = h * p.in_stride;
    uint out_base = h * p.out_stride;
    uint half_dim = p.rotary_dim / 2u;
    if (col < half_dim) {
        float first = x[in_base + 2u * col];
        float second = x[in_base + 2u * col + 1u];
        out[out_base + col] = first * cos[col] - second * sin[col];
    } else if (col < p.rotary_dim) {
        uint pair = col - half_dim;
        float first = x[in_base + 2u * pair];
        float second = x[in_base + 2u * pair + 1u];
        out[out_base + col] = second * cos[pair] + first * sin[pair];
    } else {
        out[out_base + col] = x[in_base + col];
    }
}

kernel void gravity_rope_prefix_tail_positioned_f32(
    device const float *x     [[buffer(0)]],
    device       float *out   [[buffer(1)]],
    device const float *cos   [[buffer(2)]],
    device const float *sin   [[buffer(3)]],
    constant GravityGlmPositionedRopeParams &p [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= p.n_heads * p.out_stride) { return; }
    uint h = id / p.out_stride;
    uint col = id - h * p.out_stride;
    uint in_base = p.input_element_offset + h * p.in_stride;
    uint out_base = p.output_element_offset + h * p.out_stride;
    uint half_dim = p.rotary_dim / 2u;
    if (col < half_dim) {
        float first = x[in_base + 2u * col];
        float second = x[in_base + 2u * col + 1u];
        out[out_base + col] = first * cos[col] - second * sin[col];
    } else if (col < p.rotary_dim) {
        uint pair = col - half_dim;
        float first = x[in_base + 2u * pair];
        float second = x[in_base + 2u * pair + 1u];
        out[out_base + col] = second * cos[pair] + first * sin[pair];
    } else {
        out[out_base + col] = x[in_base + col];
    }
}

// Copy unrotated tail after a rope-interleaved prefix (indexer / query assemble).
kernel void gravity_copy_tail_f32(
    device const float *src [[buffer(0)]],
    device       float *dst [[buffer(1)]],
    constant     uint  &src_off [[buffer(2)]],
    constant     uint  &dst_off [[buffer(3)]],
    constant     uint  &n [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) { return; }
    dst[dst_off + id] = src[src_off + id];
}

// Affine LayerNorm used by the DSA indexer key path.
kernel void gravity_layernorm_affine_f32(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device       float *out [[buffer(3)]],
    constant     uint  &n [[buffer(4)]],
    constant     float &eps [[buffer(5)]],
    threadgroup  float *shmem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]])
{
    float sum = 0.0f;
    for (uint i = tid; i < n; i += tg) sum += x[i];
    shmem[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg / 2u; s > 0u; s >>= 1u) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shmem[0] / (float)n;
    float var_acc = 0.0f;
    for (uint i = tid; i < n; i += tg) {
        float d = x[i] - mean;
        var_acc += d * d;
    }
    shmem[tid] = var_acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg / 2u; s > 0u; s >>= 1u) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(shmem[0] / (float)n + eps);
    for (uint i = tid; i < n; i += tg) {
        out[i] = (x[i] - mean) * inv * weight[i] + bias[i];
    }
}

// RMSNorm matching gravity_glm::rmsnorm (mean of squares, then scale).
kernel void gravity_rmsnorm_f32(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device       float *out [[buffer(2)]],
    constant     uint  &n [[buffer(3)]],
    constant     float &eps [[buffer(4)]],
    threadgroup  float *shmem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]])
{
    float partial = 0.0f;
    for (uint i = tid; i < n; i += tg) {
        float v = x[i];
        partial += v * v;
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg / 2u; s > 0u; s >>= 1u) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(shmem[0] / (float)n + eps);
    for (uint i = tid; i < n; i += tg) {
        out[i] = x[i] * inv * weight[i];
    }
}

// Append one position's MLA keys/values into the growing cache.
// kv_b layout per head: [nope (qk_nope) | value (v_dim)]
// keys layout: [pos][head][nope | k_rot]
// values layout: [pos][head][v]
struct GravityGlmMlaAppendParams {
    uint n_heads;
    uint qk_nope;
    uint qk_rope;
    uint v_dim;
    uint pos;
};

kernel void gravity_glm_mla_append_kv(
    device const float *kv [[buffer(0)]],      // n_heads * (nope+v)
    device const float *k_rot [[buffer(1)]],   // qk_rope (shared across heads)
    device       float *keys [[buffer(2)]],
    device       float *values [[buffer(3)]],
    constant GravityGlmMlaAppendParams &p [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint qk = p.qk_nope + p.qk_rope;
    uint per_kv = p.qk_nope + p.v_dim;
    // Cover both key and value writes:  n_heads * (qk + v_dim) elements.
    uint key_elems = p.n_heads * qk;
    uint val_elems = p.n_heads * p.v_dim;
    uint total = key_elems + val_elems;
    if (id >= total) { return; }
    if (id < key_elems) {
        uint head = id / qk;
        uint d = id - head * qk;
        uint dst = (p.pos * p.n_heads + head) * qk + d;
        if (d < p.qk_nope) {
            keys[dst] = kv[head * per_kv + d];
        } else {
            keys[dst] = k_rot[d - p.qk_nope];
        }
    } else {
        uint vid = id - key_elems;
        uint head = vid / p.v_dim;
        uint d = vid - head * p.v_dim;
        uint dst = (p.pos * p.n_heads + head) * p.v_dim + d;
        values[dst] = kv[head * per_kv + p.qk_nope + d];
    }
}

// Append one position's compact MLA state without expanding per-head K/V.
// latent_cache layout: [pos][kv_lora_rank]
// rope_cache layout:   [pos][qk_rope_head_dim] (shared across heads)
struct GravityGlmMlaCompactAppendParams {
    uint latent_dim;
    uint rope_dim;
    uint pos;
};

kernel void gravity_glm_mla_append_compact(
    device const float *latent [[buffer(0)]],
    device const float *k_rot [[buffer(1)]],
    device       float *latent_cache [[buffer(2)]],
    device       float *rope_cache [[buffer(3)]],
    constant GravityGlmMlaCompactAppendParams &p [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint total = p.latent_dim + p.rope_dim;
    if (id >= total) { return; }
    if (id < p.latent_dim) {
        latent_cache[p.pos * p.latent_dim + id] = latent[id];
    } else {
        uint rope = id - p.latent_dim;
        rope_cache[p.pos * p.rope_dim + rope] = k_rot[rope];
    }
}

// Absorb the content-key projection into each head's query directly from a
// single-subspace, byte-indexed gravity-pq matrix. The logical source matrix
// is kv_b_proj [head * row_stride + key_row, latent_col]. One thread owns one
// output and visits key_row in ascending order, so there is no atomic or
// cross-thread reduction.
struct GravityPqKTransposeHeadsParams {
    uint n_heads;
    uint key_rows;
    uint row_stride;
    uint latent_dim;
    uint pq_dim;
    uint pq_sub;
    uint pq_nchunk;
};

static inline void gravity_compensated_add(
    float value,
    thread float &sum,
    thread float &compensation)
{
    float corrected = value - compensation;
    float next = sum + corrected;
    compensation = (next - sum) - corrected;
    sum = next;
}

kernel void gravity_pq_k_transpose_heads(
    device const half  *codebooks [[buffer(0)]],
    device const uchar *codes [[buffer(1)]],
    device const float *query_nope [[buffer(2)]],
    device       float *query_latent [[buffer(3)]],
    constant GravityPqKTransposeHeadsParams &p [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint total = p.n_heads * p.latent_dim;
    if (id >= total) { return; }
    uint head = id / p.latent_dim;
    uint col = id - head * p.latent_dim;
    uint chunk = col / p.pq_dim;
    uint within = col - chunk * p.pq_dim;
    float acc = 0.0f;
    float compensation = 0.0f;
    for (uint key_row = 0u; key_row < p.key_rows; ++key_row) {
        uint row = head * p.row_stride + key_row;
        uint code = uint(codes[row * p.pq_nchunk + chunk]);
        float weight = float(codebooks[code * p.pq_sub + within]);
        float product = fma(weight, query_nope[head * p.key_rows + key_row], 0.0f);
        gravity_compensated_add(product, acc, compensation);
    }
    query_latent[id] = acc;
}

// Compact absorbed MLA attention over the stable DSA score-ranked positions.
// One threadgroup owns one head. Scores, softmax normalization, and the final
// weighted-latent reduction all preserve the supplied rank order. The query
// latent and weighted-latent buffers may alias: every query read completes
// before the post-score threadgroup barrier permits any output write.
struct GravityGlmCompactRankedAttnParams {
    uint n_heads;
    uint latent_dim;
    uint rope_dim;
    uint n_keys;
    uint n_allow;
    float scale;
};

kernel void gravity_glm_compact_ranked_attn(
    device const float *query_latent [[buffer(0)]],   // n_heads * latent_dim
    device const float *query_rope [[buffer(1)]],     // n_heads * rope_dim
    device const float *latent_cache [[buffer(2)]],   // n_keys * latent_dim
    device const float *rope_cache [[buffer(3)]],     // n_keys * rope_dim
    device const uint  *ranked_idx [[buffer(4)]],     // n_allow, DSA rank order
    device       float *weighted_latent [[buffer(5)]],// n_heads * latent_dim
    constant GravityGlmCompactRankedAttnParams &p [[buffer(6)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float *scores [[threadgroup(0)]])
{
    if (head >= p.n_heads) { return; }
    device const float *qh = query_latent + head * p.latent_dim;
    device const float *qr = query_rope + head * p.rope_dim;

    // Each score has one owner and visits latent dimensions first, then the
    // shared RoPE dimensions, both in strictly ascending dimension order.
    for (uint a = tid; a < p.n_allow; a += tg) {
        uint token = ranked_idx[a];
        float score = -INFINITY;
        if (token < p.n_keys) {
            device const float *latent = latent_cache + token * p.latent_dim;
            device const float *rope = rope_cache + token * p.rope_dim;
            float dot = 0.0f;
            float compensation = 0.0f;
            for (uint d = 0u; d < p.latent_dim; ++d) {
                float product = fma(qh[d], latent[d], 0.0f);
                gravity_compensated_add(product, dot, compensation);
            }
            for (uint d = 0u; d < p.rope_dim; ++d) {
                float product = fma(qr[d], rope[d], 0.0f);
                gravity_compensated_add(product, dot, compensation);
            }
            score = dot * p.scale;
        }
        scores[a] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Serial stable softmax in the supplied DSA rank order.
    if (tid == 0u) {
        float best = -INFINITY;
        for (uint a = 0u; a < p.n_allow; ++a) {
            best = max(best, scores[a]);
        }
        float total = 0.0f;
        float total_compensation = 0.0f;
        for (uint a = 0u; a < p.n_allow; ++a) {
            float score = scores[a];
            float probability =
                (score > -INFINITY / 2.0f)
                    ? metal::precise::exp(score - best)
                    : 0.0f;
            scores[a] = probability;
            gravity_compensated_add(probability, total, total_compensation);
        }
        if (total > 0.0f) {
            for (uint a = 0u; a < p.n_allow; ++a) {
                scores[a] = metal::precise::divide(scores[a], total);
            }
        } else {
            for (uint a = 0u; a < p.n_allow; ++a) {
                scores[a] = 0.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One owner per latent output, with probability-weighted accumulation in
    // the same DSA rank order as the softmax normalization.
    device float *out = weighted_latent + head * p.latent_dim;
    for (uint d = tid; d < p.latent_dim; d += tg) {
        float acc = 0.0f;
        float compensation = 0.0f;
        for (uint a = 0u; a < p.n_allow; ++a) {
            uint token = ranked_idx[a];
            if (token < p.n_keys) {
                float product =
                    fma(scores[a], latent_cache[token * p.latent_dim + d], 0.0f);
                gravity_compensated_add(product, acc, compensation);
            }
        }
        out[d] = acc;
    }
}

// Apply the value-row window of an interleaved per-head K/V matrix directly
// from a single-subspace, byte-indexed gravity-pq tensor. One SIMD group owns
// one output row and uses the generic gravity_pq_matvec lane/chunk order and
// simd_sum so value reconstruction is arithmetically aligned with expansion.
struct GravityPqVRowsHeadsParams {
    uint n_heads;
    uint row_stride;
    uint value_row_offset;
    uint value_rows;
    uint latent_dim;
    uint pq_dim;
    uint pq_sub;
    uint pq_nchunk;
};

kernel void gravity_pq_v_rows_heads(
    device const half  *codebooks [[buffer(0)]],
    device const uchar *codes [[buffer(1)]],
    device const float *weighted_latent [[buffer(2)]],
    device       float *context [[buffer(3)]],
    constant GravityPqVRowsHeadsParams &p [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg_in_tg [[simdgroup_index_in_threadgroup]],
    uint sgs_per_tg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint total = p.n_heads * p.value_rows;
    uint id = tgid * sgs_per_tg + sg_in_tg;
    if (id >= total) { return; }
    uint head = id / p.value_rows;
    uint value_row = id - head * p.value_rows;
    uint source_row = head * p.row_stride + p.value_row_offset + value_row;
    device const float *x = weighted_latent + head * p.latent_dim;
    float acc = 0.0f;
    for (uint chunk = lane; chunk < p.pq_nchunk; chunk += 32u) {
        uint code = uint(codes[source_row * p.pq_nchunk + chunk]);
        device const half *entry = codebooks + code * p.pq_sub;
        device const float *xs = x + chunk * p.pq_dim;
        for (uint within = 0u; within < p.pq_sub; ++within) {
            acc = fma(float(entry[within]), xs[within], acc);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        context[id] = acc;
    }
}

// Build queries: per head, copy nope half from q, rope-interleaved rope half.
// `q_rope_rot` is already rope-interleaved per head (n_heads * qk_rope).
struct GravityGlmBuildQParams {
    uint n_heads;
    uint qk_nope;
    uint qk_rope;
};

kernel void gravity_glm_build_queries(
    device const float *q [[buffer(0)]],           // n_heads * (nope+rope) raw
    device const float *q_rope_rot [[buffer(1)]],  // n_heads * rope rotated
    device       float *queries [[buffer(2)]],
    constant GravityGlmBuildQParams &p [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint qk = p.qk_nope + p.qk_rope;
    if (id >= p.n_heads * qk) { return; }
    uint head = id / qk;
    uint d = id - head * qk;
    if (d < p.qk_nope) {
        queries[id] = q[head * qk + d];
    } else {
        queries[id] = q_rope_rot[head * p.qk_rope + (d - p.qk_nope)];
    }
}

// Copy the per-head non-RoPE prefix from raw q into the compact MLA layout.
kernel void gravity_copy_head_prefix_f32(
    device const float *q [[buffer(0)]],
    device       float *prefix [[buffer(1)]],
    constant GravityGlmBuildQParams &p [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint total = p.n_heads * p.qk_nope;
    if (id >= total) { return; }
    uint head = id / p.qk_nope;
    uint d = id - head * p.qk_nope;
    prefix[id] = q[head * (p.qk_nope + p.qk_rope) + d];
}

// DSA index scores: for each cached index key, sum_h w_h * relu(dot(q_h, k) * dim_scale).
struct GravityGlmDsaParams {
    uint n_keys;
    uint n_heads;
    uint head_dim;
    uint pos;         // causal: mask t > pos
    float dim_scale;
    float head_scale;
};

kernel void gravity_glm_dsa_scores(
    device const float *q_full [[buffer(0)]],       // n_heads * head_dim
    device const float *index_keys [[buffer(1)]],   // n_keys * head_dim
    device const float *head_weights [[buffer(2)]], // n_heads
    device       float *scores [[buffer(3)]],
    constant GravityGlmDsaParams &p [[buffer(4)]],
    uint t [[thread_position_in_grid]])
{
    if (t >= p.n_keys) { return; }
    if (t > p.pos) {
        scores[t] = -INFINITY;
        return;
    }
    device const float *key = index_keys + t * p.head_dim;
    float acc = 0.0f;
    for (uint h = 0; h < p.n_heads; ++h) {
        device const float *qh = q_full + h * p.head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < p.head_dim; ++d) {
            dot = fma(qh[d], key[d], dot);
        }
        float relu = max(dot * p.dim_scale, 0.0f);
        float weight = head_weights[h] * p.head_scale;
        acc = fma(weight, relu, acc);
    }
    scores[t] = acc;
}

// Stable descending top-k (np.argsort stable, lower index first on ties).
// Single thread, serial selection — exact over the host topk_desc.
// `selected` is an n-byte scratch (0/1) supplied by the caller.
struct GravityGlmTopkParams {
    uint n;
    uint k;
};

kernel void gravity_glm_stable_topk_f32(
    device const float *values [[buffer(0)]],
    device       uint  *indices [[buffer(1)]],
    device       uchar *selected [[buffer(2)]],
    constant GravityGlmTopkParams &p [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid != 0u) { return; }
    uint k = p.k < p.n ? p.k : p.n;
    for (uint i = 0; i < p.n; ++i) selected[i] = 0;
    for (uint slot = 0; slot < k; ++slot) {
        uint best_i = 0xFFFFFFFFu;
        float best_v = -INFINITY;
        for (uint i = 0; i < p.n; ++i) {
            if (selected[i]) continue;
            float v = values[i];
            if (best_i == 0xFFFFFFFFu
                || v > best_v
                || (v == best_v && i < best_i)) {
                best_v = v;
                best_i = i;
            }
        }
        indices[slot] = best_i;
        if (best_i != 0xFFFFFFFFu) selected[best_i] = 1;
    }
}

// Parallel exact stable top-k for the admitted n<=32K, k<=2048 DSA domain.
//
// A monotone IEEE-f32 key occupies the high 32 bits; inverted position
// occupies the low 32 bits, so unsigned descending order is precisely
// (score descending, lower position first). Sixteen 4-bit histogram passes
// identify the unique kth composite key. Exactly k qualifying keys then fit
// in 16 KiB of threadgroup memory and are bitonic-ranked in place.
inline ulong gravity_glm_score_position_key(float value, uint position)
{
    // DSA scores are required finite. Mapping NaN to -inf keeps malformed
    // arithmetic from outranking a valid score; complete-token parity gates
    // separately reject any resulting decision drift.
    if (isnan(value)) value = -INFINITY;
    if (value == 0.0f) value = 0.0f; // canonicalize -0/+0 host equality
    uint bits = as_type<uint>(value);
    uint ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    return ((ulong)ordered << 32) | (ulong)(0xFFFFFFFFu - position);
}

kernel void gravity_glm_radix_topk_f32(
    device const float *values [[buffer(0)]],
    device       uint  *indices [[buffer(1)]],
    constant GravityGlmTopkParams &p [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]])
{
    threadgroup atomic_uint histogram[16];
    threadgroup atomic_uint selected_count;
    threadgroup ulong ranked[2048];
    threadgroup ulong prefix;
    threadgroup uint prefix_nibbles;
    threadgroup uint target_rank;
    threadgroup uint invalid;

    uint out_k = min(p.k, p.n);
    if (out_k == 0u) return;
    if (tid == 0u) {
        prefix = 0ul;
        prefix_nibbles = 0u;
        target_rank = out_k - 1u;
        invalid = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // MSD radix-select the kth-largest unique (score, inverted-position) key.
    for (uint pass = 0u; pass < 16u; ++pass) {
        if (tid < 16u) {
            atomic_store_explicit(&histogram[tid], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint known = prefix_nibbles;
        ulong mask = known == 0u ? 0ul : (~0ul << (64u - 4u * known));
        ulong wanted = prefix;
        uint shift = 60u - 4u * pass;
        for (uint i = tid; i < p.n; i += tg) {
            ulong key = gravity_glm_score_position_key(values[i], i);
            if ((key & mask) == wanted) {
                uint digit = (uint)((key >> shift) & 0xFul);
                atomic_fetch_add_explicit(&histogram[digit], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            uint rank = target_rank;
            bool found = false;
            for (int digit = 15; digit >= 0; --digit) {
                uint count = atomic_load_explicit(
                    &histogram[(uint)digit], memory_order_relaxed);
                if (rank < count) {
                    prefix |= ((ulong)(uint)digit << shift);
                    prefix_nibbles = pass + 1u;
                    target_rank = rank;
                    found = true;
                    break;
                }
                rank -= count;
            }
            if (!found) invalid = 1u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    atomic_store_explicit(&selected_count, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong threshold = prefix;
    for (uint i = tid; i < p.n; i += tg) {
        ulong key = gravity_glm_score_position_key(values[i], i);
        if (key >= threshold) {
            uint slot = atomic_fetch_add_explicit(
                &selected_count, 1u, memory_order_relaxed);
            if (slot < out_k) ranked[slot] = key;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint selected = atomic_load_explicit(&selected_count, memory_order_relaxed);
    if (tid == 0u && selected != out_k) invalid = 1u;
    uint width = 1u;
    while (width < out_k) width <<= 1u;
    for (uint i = out_k + tid; i < width; i += tg) ranked[i] = 0ul;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Ascending bitonic sort, then emit in reverse for descending score rank.
    for (uint size = 2u; size <= width; size <<= 1u) {
        for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {
            for (uint i = tid; i < width; i += tg) {
                uint peer = i ^ stride;
                if (peer > i) {
                    ulong a = ranked[i];
                    ulong b = ranked[peer];
                    bool ascending = (i & size) == 0u;
                    if ((ascending && a > b) || (!ascending && a < b)) {
                        ranked[i] = b;
                        ranked[peer] = a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint slot = tid; slot < out_k; slot += tg) {
        if (invalid) {
            indices[slot] = 0xFFFFFFFFu;
        } else {
            uint inverted_position = (uint)ranked[width - 1u - slot];
            indices[slot] = 0xFFFFFFFFu - inverted_position;
        }
    }
}

// Reorder the unique score-ordered top-k IDs into ascending position order,
// matching the host sparse-attention accumulation order. One 256-thread group
// sorts at most 2048 u32 IDs in <=8 KiB of dynamic threadgroup memory.
//
// Bitonic padding uses UINT_MAX, which is outside the admitted context-position
// domain. Input and output may alias: every live element is loaded into shared
// memory before the first output write.
struct GravityGlmSortU32Params {
    uint n;
};

kernel void gravity_glm_sort_u32_ascending(
    device const uint *input [[buffer(0)]],
    device       uint *output [[buffer(1)]],
    constant GravityGlmSortU32Params &p [[buffer(2)]],
    threadgroup uint *items [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]])
{
    uint width = 1u;
    while (width < p.n) { width <<= 1u; }

    for (uint i = tid; i < width; i += tg) {
        items[i] = i < p.n ? input[i] : 0xFFFFFFFFu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint span = 2u; span <= width; span <<= 1u) {
        for (uint stride = span >> 1u; stride > 0u; stride >>= 1u) {
            for (uint i = tid; i < width; i += tg) {
                uint peer = i ^ stride;
                if (peer > i) {
                    uint a = items[i];
                    uint b = items[peer];
                    bool ascending = (i & span) == 0u;
                    if ((a > b) == ascending) {
                        items[i] = b;
                        items[peer] = a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint i = tid; i < p.n; i += tg) {
        output[i] = items[i];
    }
}

// Sparse multi-head attention over an allow-list of key positions (DSA top-k).
// One threadgroup per head; threads stride over allow entries for the dot,
// then a serial softmax+accumulate on thread 0 for host-matching order.
struct GravityGlmSparseAttnParams {
    uint n_heads;
    uint qk_dim;
    uint v_dim;
    uint n_keys;      // total cached positions
    uint n_allow;     // length of allow_idx
    float scale;
};

kernel void gravity_glm_sparse_attn(
    device const float *queries [[buffer(0)]],    // n_heads * qk_dim
    device const float *keys [[buffer(1)]],       // n_keys * n_heads * qk_dim
    device const float *values [[buffer(2)]],     // n_keys * n_heads * v_dim
    device const uint  *allow_idx [[buffer(3)]],  // n_allow
    device       float *context [[buffer(4)]],    // n_heads * v_dim
    constant GravityGlmSparseAttnParams &p [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float *shmem [[threadgroup(0)]])
{
    if (head >= p.n_heads) { return; }
    device const float *qh = queries + head * p.qk_dim;

    // Phase 1: each thread scores a strided subset of allow entries into shmem.
    // shmem[0..n_allow) = scores (we require n_allow * sizeof(float) shmem).
    for (uint a = tid; a < p.n_allow; a += tg) {
        uint t = allow_idx[a];
        float s = -INFINITY;
        if (t < p.n_keys) {
            device const float *kh = keys + (t * p.n_heads + head) * p.qk_dim;
            float dot = 0.0f;
            for (uint d = 0; d < p.qk_dim; ++d) {
                dot = fma(qh[d], kh[d], dot);
            }
            s = dot * p.scale;
        }
        shmem[a] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float best = -INFINITY;
        for (uint a = 0; a < p.n_allow; ++a) {
            best = max(best, shmem[a]);
        }
        float total = 0.0f;
        for (uint a = 0; a < p.n_allow; ++a) {
            float s = shmem[a];
            float e = (s > -INFINITY / 2.0f) ? exp(s - best) : 0.0f;
            shmem[a] = e;
            total += e;
        }
        device float *out = context + head * p.v_dim;
        for (uint d = 0; d < p.v_dim; ++d) out[d] = 0.0f;
        if (total > 0.0f) {
            for (uint a = 0; a < p.n_allow; ++a) {
                float w = shmem[a] / total;
                if (w == 0.0f) continue;
                uint t = allow_idx[a];
                if (t >= p.n_keys) continue;
                device const float *vh = values + (t * p.n_heads + head) * p.v_dim;
                for (uint d = 0; d < p.v_dim; ++d) {
                    out[d] = fma(w, vh[d], out[d]);
                }
            }
        }
    }
}

// Router: corrected = sigmoid(logits) + bias. Written for the group-score path
// on the host side of a small read of indices only; scores stay resident.
kernel void gravity_glm_router_correct(
    device const float *logits [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device       float *scores [[buffer(2)]],     // sigmoid
    device       float *corrected [[buffer(3)]],
    constant     uint  &n [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) { return; }
    float s = 1.0f / (1.0f + exp(-logits[id]));
    scores[id] = s;
    corrected[id] = s + bias[id];
}

struct GravityRouterSelectParams {
    uint n_experts;
    uint n_group;
    uint topk_group;
    uint experts_per_token;
    uint norm_topk_prob;
    float routed_scaling_factor;
};

struct GravityExpertTraceCopyParams {
    uint count;
    uint destination_offset;
};

kernel void gravity_glm_expert_trace_copy(
    const device uint *expert_indices [[buffer(0)]],
    device uint *expert_trace [[buffer(1)]],
    constant GravityExpertTraceCopyParams &p [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= p.count) { return; }
    expert_trace[p.destination_offset + id] = expert_indices[id];
}

// Exact noaux_tc router selection with stable lower-index ties. One thread is
// intentional: the flagship router has only 256 experts, while preserving the
// host reduction/selection order is part of the model's discrete contract.
kernel void gravity_glm_router_select_noaux_f32(
    device const float *logits [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device       float *scores [[buffer(2)]],
    device       float *corrected [[buffer(3)]],
    device        uint *expert_indices [[buffer(4)]],
    device       float *expert_weights [[buffer(5)]],
    device        uint *expert_exec_slots [[buffer(6)]],
    constant GravityRouterSelectParams &p [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    if (id != 0u) { return; }

    float group_scores[64];
    bool group_chosen[64];
    uint per_group = p.n_experts / p.n_group;

    for (uint expert = 0u; expert < p.n_experts; ++expert) {
        float s = 1.0f / (1.0f + exp(-logits[expert]));
        scores[expert] = s;
        corrected[expert] = s + bias[expert];
    }

    for (uint group = 0u; group < p.n_group; ++group) {
        float first = -INFINITY;
        float second = -INFINITY;
        uint begin = group * per_group;
        for (uint local = 0u; local < per_group; ++local) {
            float value = corrected[begin + local];
            if (value > first) {
                second = first;
                first = value;
            } else if (value > second) {
                second = value;
            }
        }
        group_scores[group] = first + ((per_group > 1u) ? second : 0.0f);
        group_chosen[group] = false;
    }

    for (uint slot = 0u; slot < p.topk_group; ++slot) {
        float best = -INFINITY;
        uint best_group = 0xFFFFFFFFu;
        for (uint group = 0u; group < p.n_group; ++group) {
            if (!group_chosen[group]
                && (best_group == 0xFFFFFFFFu || group_scores[group] > best)) {
                best = group_scores[group];
                best_group = group;
            }
        }
        group_chosen[best_group] = true;
    }

    for (uint slot = 0u; slot < p.experts_per_token; ++slot) {
        float best = -INFINITY;
        uint best_expert = 0xFFFFFFFFu;
        for (uint expert = 0u; expert < p.n_experts; ++expert) {
            if (!group_chosen[expert / per_group]) { continue; }
            bool already_chosen = false;
            for (uint prior = 0u; prior < slot; ++prior) {
                already_chosen = already_chosen || expert_indices[prior] == expert;
            }
            if (!already_chosen
                && (best_expert == 0xFFFFFFFFu || corrected[expert] > best)) {
                best = corrected[expert];
                best_expert = expert;
            }
        }
        expert_indices[slot] = best_expert;
        expert_weights[slot] = scores[best_expert];
    }

    float total = 0.0f;
    if (p.norm_topk_prob != 0u) {
        for (uint slot = 0u; slot < p.experts_per_token; ++slot) {
            total += expert_weights[slot];
        }
        total += 1.0e-20f;
    } else {
        total = 1.0f;
    }
    for (uint slot = 0u; slot < p.experts_per_token; ++slot) {
        expert_weights[slot] =
            (expert_weights[slot] / total) * p.routed_scaling_factor;
        expert_exec_slots[slot] = slot;
    }

    // A second device-owned view gives execution order without disturbing
    // the score-ranked diagnostic IDs or their aligned weights. Insertion
    // sort is exact and bounded (flagship k=8); lower expert ID wins.
    for (uint slot = 1u; slot < p.experts_per_token; ++slot) {
        uint selected_slot = expert_exec_slots[slot];
        uint selected_expert = expert_indices[selected_slot];
        uint pos = slot;
        while (pos > 0u) {
            uint prior_slot = expert_exec_slots[pos - 1u];
            if (expert_indices[prior_slot] <= selected_expert) { break; }
            expert_exec_slots[pos] = prior_slot;
            --pos;
        }
        expert_exec_slots[pos] = selected_slot;
    }
}

// ---------------------------------------------------------------------------
// Cache-indexed routed-expert address-table proof.
//
// These layouts are frozen against DeviceExpertTensorRef (56 B) and
// DeviceExpertTriplet (176 B) in gravity_glm_resident.rs. Pointer fields are
// host-populated Metal gpuAddress values. Every indirectly referenced resource
// must also be declared through useResources before the dispatch.
// ---------------------------------------------------------------------------

struct GravityDeviceExpertTensorRef {
    const device uchar *primary;
    const device uchar *secondary;
    uint dim;
    uint subspaces;
    uint sub;
    uint card;
    uint rows;
    uint cols;
    uint nchunk;
    uint bits;
    uint kind;
    uint generation;
};

struct GravityDeviceExpertTriplet {
    GravityDeviceExpertTensorRef gate;
    GravityDeviceExpertTensorRef up;
    GravityDeviceExpertTensorRef down;
    uint ready_mask;
    uint generation;
};

static_assert(sizeof(GravityDeviceExpertTensorRef) == 56,
              "GravityDeviceExpertTensorRef ABI drift");
static_assert(sizeof(GravityDeviceExpertTriplet) == 176,
              "GravityDeviceExpertTriplet ABI drift");

constant constexpr uint GRAVITY_EXPERT_KIND_PQ = 1u;
constant constexpr uint GRAVITY_EXPERT_KIND_NATIVE_BF16 = 2u;
constant constexpr uint GRAVITY_EXPERT_KIND_ANY_SUPPORTED = 0u;
constant constexpr uint GRAVITY_EXPERT_TRIPLET_READY = 7u;

struct GravityDeviceExpertValidateParams {
    uint n_experts;
    uint experts_per_token;
    uint generation;
    uint required_kind;
    uint hidden;
    uint intermediate;
};

struct GravityDeviceExpertMatvecParams {
    uint n_experts;
    uint experts_per_token;
    uint generation;
    uint execution_position;
    uint projection;
    uint rows;
    uint cols;
    uint allow_other_kind;
};

static_assert(sizeof(GravityDeviceExpertValidateParams) == 24,
              "GravityDeviceExpertValidateParams ABI drift");
static_assert(sizeof(GravityDeviceExpertMatvecParams) == 32,
              "GravityDeviceExpertMatvecParams ABI drift");

static inline bool gravity_device_expert_tensor_valid(
    const device GravityDeviceExpertTensorRef &tensor,
    uint generation,
    uint required_kind)
{
    uint admitted_kind =
        required_kind == GRAVITY_EXPERT_KIND_ANY_SUPPORTED
        ? tensor.kind
        : required_kind;
    if (tensor.generation != generation ||
        tensor.kind != admitted_kind ||
        tensor.primary == nullptr ||
        tensor.rows == 0u ||
        tensor.cols == 0u) {
        return false;
    }
    if (admitted_kind == GRAVITY_EXPERT_KIND_PQ) {
        return tensor.secondary != nullptr &&
               tensor.bits > 0u &&
               tensor.bits <= 8u &&
               tensor.subspaces > 0u &&
               tensor.sub > 0u &&
               tensor.dim == tensor.subspaces * tensor.sub &&
               tensor.card == (1u << tensor.bits) &&
               tensor.nchunk > 0u &&
               tensor.cols == tensor.nchunk * tensor.dim;
    }
    if (admitted_kind == GRAVITY_EXPERT_KIND_NATIVE_BF16) {
        return tensor.secondary == nullptr;
    }
    return false;
}

kernel void gravity_glm_expert_table_validate(
    const device uint *expert_indices [[buffer(0)]],
    const device uint *expert_exec_slots [[buffer(1)]],
    const device GravityDeviceExpertTriplet *table [[buffer(2)]],
    device atomic_uint *miss_mask [[buffer(3)]],
    constant GravityDeviceExpertValidateParams &p [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id != 0u) { return; }
    uint missing = 0u;
    for (uint execution_position = 0u;
         execution_position < p.experts_per_token;
         ++execution_position) {
        uint bit = 1u << execution_position;
        uint slot = expert_exec_slots[execution_position];
        if (slot >= p.experts_per_token) {
            missing |= bit;
            continue;
        }
        uint expert = expert_indices[slot];
        if (expert >= p.n_experts) {
            missing |= bit;
            continue;
        }
        const device GravityDeviceExpertTriplet &entry = table[expert];
        bool ready =
            entry.ready_mask == GRAVITY_EXPERT_TRIPLET_READY &&
            entry.generation == p.generation &&
            gravity_device_expert_tensor_valid(
                entry.gate, p.generation, p.required_kind) &&
            gravity_device_expert_tensor_valid(
                entry.up, p.generation, p.required_kind) &&
            gravity_device_expert_tensor_valid(
                entry.down, p.generation, p.required_kind) &&
            entry.gate.rows == p.intermediate &&
            entry.gate.cols == p.hidden &&
            entry.up.rows == p.intermediate &&
            entry.up.cols == p.hidden &&
            entry.down.rows == p.hidden &&
            entry.down.cols == p.intermediate;
        if (!ready) {
            missing |= bit;
        }
    }
    atomic_store_explicit(miss_mask, missing, memory_order_relaxed);
}

kernel void gravity_glm_expert_table_pq_matvec(
    const device uint *expert_indices [[buffer(0)]],
    const device uint *expert_exec_slots [[buffer(1)]],
    const device GravityDeviceExpertTriplet *table [[buffer(2)]],
    device atomic_uint *miss_mask [[buffer(3)]],
    const device float *x [[buffer(4)]],
    device float *y [[buffer(5)]],
    constant GravityDeviceExpertMatvecParams &p [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg_in_tg [[simdgroup_index_in_threadgroup]],
    uint sgs_per_tg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (atomic_load_explicit(miss_mask, memory_order_relaxed) != 0u) {
        return;
    }
    if (p.execution_position >= p.experts_per_token) {
        return;
    }
    uint slot = expert_exec_slots[p.execution_position];
    if (slot >= p.experts_per_token) {
        return;
    }
    uint expert = expert_indices[slot];
    if (expert >= p.n_experts) {
        return;
    }
    const device GravityDeviceExpertTriplet &entry = table[expert];
    const device GravityDeviceExpertTensorRef *tensor =
        p.projection == 0u ? &entry.gate :
        (p.projection == 1u ? &entry.up : &entry.down);
    if (p.allow_other_kind != 0u &&
        tensor->kind == GRAVITY_EXPERT_KIND_NATIVE_BF16) {
        return;
    }
    bool valid =
        p.projection <= 2u &&
        entry.ready_mask == GRAVITY_EXPERT_TRIPLET_READY &&
        entry.generation == p.generation &&
        gravity_device_expert_tensor_valid(
            *tensor, p.generation, GRAVITY_EXPERT_KIND_PQ) &&
        tensor->rows == p.rows &&
        tensor->cols == p.cols;
    if (!valid) {
        if (tgid == 0u && sg_in_tg == 0u && lane == 0u) {
            atomic_fetch_or_explicit(
                miss_mask, 1u << p.execution_position, memory_order_relaxed);
        }
        return;
    }

    uint row = tgid * sgs_per_tg + sg_in_tg;
    if (row >= tensor->rows) { return; }
    const device half *codebooks =
        reinterpret_cast<const device half *>(tensor->primary);
    const device uchar *codes = tensor->secondary;
    float acc = 0.0f;
    if (tensor->bits == 8u && tensor->subspaces == 1u) {
        // Preserve the qualified R4 direct-byte path exactly.
        for (uint chunk = lane; chunk < tensor->nchunk; chunk += 32u) {
            uint flat = row * tensor->nchunk + chunk;
            const device half *entry_values =
                codebooks + uint(codes[flat]) * tensor->sub;
            const device float *xs = x + chunk * tensor->dim;
            for (uint j = 0u; j < tensor->sub; ++j) {
                acc = fma(float(entry_values[j]), xs[j], acc);
            }
        }
    } else {
        // Packed-PQ path used by R0 (D8/S1/sub8/card128/bits7) and any
        // descriptor satisfying the same immutable tensor invariants.
        for (uint s = 0u; s < tensor->subspaces; ++s) {
            const device half *codebook =
                codebooks + s * tensor->card * tensor->sub;
            const uint xbase = s * tensor->sub;
            for (uint chunk = lane; chunk < tensor->nchunk; chunk += 32u) {
                uint flat =
                    (row * tensor->nchunk + chunk) * tensor->subspaces + s;
                const device half *entry_values =
                    codebook + pq_index(codes, flat, tensor->bits) * tensor->sub;
                const device float *xs =
                    x + chunk * tensor->dim + xbase;
                for (uint j = 0u; j < tensor->sub; ++j) {
                    acc = fma(float(entry_values[j]), xs[j], acc);
                }
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        y[row] = acc;
    }
}

// Native-BF16 indirect counterpart. `matmul.metal` precedes this source in the
// single Metal translation unit and establishes contract(off) for the
// qualified sequential path. Reassert it here so the multiply and add remain
// separate even if source ordering changes.
#pragma clang fp contract(off)
kernel void gravity_glm_expert_table_native_bf16_matvec(
    const device uint *expert_indices [[buffer(0)]],
    const device uint *expert_exec_slots [[buffer(1)]],
    const device GravityDeviceExpertTriplet *table [[buffer(2)]],
    device atomic_uint *miss_mask [[buffer(3)]],
    const device float *x [[buffer(4)]],
    device float *y [[buffer(5)]],
    constant GravityDeviceExpertMatvecParams &p [[buffer(6)]],
    uint row [[thread_position_in_grid]])
{
    if (atomic_load_explicit(miss_mask, memory_order_relaxed) != 0u) {
        return;
    }
    if (p.execution_position >= p.experts_per_token) {
        return;
    }
    uint slot = expert_exec_slots[p.execution_position];
    if (slot >= p.experts_per_token) {
        return;
    }
    uint expert = expert_indices[slot];
    if (expert >= p.n_experts) {
        return;
    }
    const device GravityDeviceExpertTriplet &entry = table[expert];
    const device GravityDeviceExpertTensorRef *tensor =
        p.projection == 0u ? &entry.gate :
        (p.projection == 1u ? &entry.up : &entry.down);
    if (p.allow_other_kind != 0u &&
        tensor->kind == GRAVITY_EXPERT_KIND_PQ) {
        return;
    }
    bool valid =
        p.projection <= 2u &&
        entry.ready_mask == GRAVITY_EXPERT_TRIPLET_READY &&
        entry.generation == p.generation &&
        gravity_device_expert_tensor_valid(
            *tensor, p.generation, GRAVITY_EXPERT_KIND_NATIVE_BF16) &&
        tensor->rows == p.rows &&
        tensor->cols == p.cols;
    if (!valid) {
        if (row == 0u) {
            atomic_fetch_or_explicit(
                miss_mask, 1u << p.execution_position, memory_order_relaxed);
        }
        return;
    }
    if (row >= tensor->rows) {
        return;
    }

    const device ushort *weight_bits =
        reinterpret_cast<const device ushort *>(tensor->primary);
    const device ushort *row_bits =
        weight_bits + ulong(row) * ulong(tensor->cols);
    float acc = 0.0f;
    for (uint col = 0u; col < tensor->cols; ++col) {
        uint wide_bits = uint(row_bits[col]) << 16;
        float weight = as_type<float>(wide_bits);
        float product = weight * x[col];
        acc = acc + product;
    }
    y[row] = acc;
}

struct GravityDeviceExpertAxpyParams {
    uint n;
    uint experts_per_token;
    uint execution_position;
    uint use_router_weight;
};

kernel void gravity_glm_expert_table_zero_f32(
    device float *x [[buffer(0)]],
    device atomic_uint *miss_mask [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n ||
        atomic_load_explicit(miss_mask, memory_order_relaxed) != 0u) {
        return;
    }
    x[id] = 0.0f;
}

kernel void gravity_glm_expert_table_silu_mul_f32(
    const device float *gate [[buffer(0)]],
    const device float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    device atomic_uint *miss_mask [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n ||
        atomic_load_explicit(miss_mask, memory_order_relaxed) != 0u) {
        return;
    }
    float g = gate[id];
    out[id] = (g / (1.0f + exp(-g))) * up[id];
}

kernel void gravity_glm_expert_table_axpy_f32(
    device float *y [[buffer(0)]],
    const device float *x [[buffer(1)]],
    const device float *expert_weights [[buffer(2)]],
    const device uint *expert_exec_slots [[buffer(3)]],
    device atomic_uint *miss_mask [[buffer(4)]],
    constant GravityDeviceExpertAxpyParams &p [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= p.n ||
        atomic_load_explicit(miss_mask, memory_order_relaxed) != 0u) {
        return;
    }
    float scale = 1.0f;
    if (p.use_router_weight != 0u) {
        if (p.execution_position >= p.experts_per_token) { return; }
        uint slot = expert_exec_slots[p.execution_position];
        if (slot >= p.experts_per_token) { return; }
        scale = expert_weights[slot];
    }
    y[id] += x[id] * scale;
}

kernel void gravity_glm_expert_table_residual_add_f32(
    device float *residual [[buffer(0)]],
    const device float *expert_output [[buffer(1)]],
    device atomic_uint *miss_mask [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n ||
        atomic_load_explicit(miss_mask, memory_order_relaxed) != 0u) {
        return;
    }
    residual[id] += expert_output[id];
}

// Zero a buffer (used when starting a residual accumulate).
kernel void gravity_zero_f32(
    device float *x [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) { return; }
    x[id] = 0.0f;
}

// Append one index key (idim floats) at position `pos`.
kernel void gravity_glm_append_index_key(
    device const float *k_full [[buffer(0)]],
    device       float *index_keys [[buffer(1)]],
    constant     uint  &pos [[buffer(2)]],
    constant     uint  &idim [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= idim) { return; }
    index_keys[pos * idim + id] = k_full[id];
}

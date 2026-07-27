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
};

// NeoX pairing: element i pairs with i + head_dim/2, not with i + 1.
// `table` is head_dim/2 cosines followed by head_dim/2 sines.
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
    uint b = p.offset + h * p.head_dim + i;
    float c = table[i];
    float s = table[half_dim + i];
    float x0 = x[b];
    float x1 = x[b + half_dim];
    x[b]            = x0 * c - x1 * s;
    x[b + half_dim] = x0 * s + x1 * c;
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

kernel void gravity_scale_f32(
    device       float *x [[buffer(0)]],
    constant     float &s [[buffer(1)]],
    constant     uint  &n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) { return; }
    x[id] *= s;
}

kernel void gravity_sigmoid_f32(
    device const float *x [[buffer(0)]],
    device       float *y [[buffer(1)]],
    constant     uint  &n [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) { return; }
    y[id] = 1.0f / (1.0f + exp(-x[id]));
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

// DSA index scores: for each cached index key, sum_h w_h * relu(dot(q_h, k) * dim_scale).
struct GravityGlmDsaParams {
    uint n_keys;
    uint n_heads;
    uint head_dim;
    uint pos;         // causal: mask t > pos
    float dim_scale;
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
        acc = fma(head_weights[h], relu, acc);
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

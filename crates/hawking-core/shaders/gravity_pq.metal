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

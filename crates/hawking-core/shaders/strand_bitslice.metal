
#include <metal_stdlib>
using namespace metal;

struct BitsliceEntry {
    uint bit_offset;
    uint init_state;
    uint out_off;
    uint n;
    int  eff[8];
    int  off[8];
    uint d;            // vector dim (1 = scalar). last field => scalar kernels read the same first 80 B.
};

static inline uint bs_load_u32_le(device const uchar* p, uint widx) {
    uint b = widx << 2;
    return (uint)p[b]
         | ((uint)p[b + 1] << 8)
         | ((uint)p[b + 2] << 16)
         | ((uint)p[b + 3] << 24);
}

// ─── Computed codebook (Variant A) — inline integer Acklam, no LUT gather ─────
//
// Reproduces strand_quant::codebook::qcb(state, L) **byte-for-byte** in pure
// integer arithmetic. The CPU path evaluates Acklam's central rational in i128
// Q40 fixed point; Metal has no 128-bit integer type, so the two products that
// overflow 64 bits (q*q ≈ 2^78 and coeff*r2 ≈ 2^87) are done with a 64×64→128
// limb multiply (`bs_umul64_128`) followed by a 128-bit arithmetic shift-right
// by 40 that lands back in i64 (`bs_smul_shift40`). Every other operand fits a
// signed 64-bit `long` (|q|<2^39, |num|<2^37, |num<<12|<2^49). This was proven
// byte-identical to the i128 path for all 31_164 central ranks (L 4..=14); the
// gate-bitslice identity matrix asserts it end-to-end vs the CPU decoder.
//
// no float anywhere => deterministic across CPU/Metal/NEON (the STRAND contract).

#define BS_ACKLAM_FB 40
#define BS_QUANTILE_SHIFT 12
#define BS_QUANTILE_CLAMP_Q12 (6 * (1 << BS_QUANTILE_SHIFT))

// Acklam central numerator/denominator coefficients, pre-scaled to Q40 exactly
// (== round(coef * 2^40)); identical integers to strand_quant ACKLAM_{A,B}_Q.
constant long BS_ACKLAM_A_Q[6] = {
    -43647126486026L, 242932804329501L, -303386605671354L,
    152125956971009L, -33716302037132L, 2756066937579L,
};
constant long BS_ACKLAM_B_Q[5] = {
    -59897104064522L, 177665506509332L, -171192838788807L,
    73448819171239L, -14602263792188L,
};
// Per-L left-tail length (p < P_LOW), L=4..=14 at index L-4; matches the CPU
// TAIL_LEFT_LEN. Ranks in [0,t) and [n-t,n) use the stored tail prefix; the rest
// use the integer central rational.
constant uint BS_TAIL_LEFT_LEN[11] = {0u,1u,2u,3u,6u,12u,25u,50u,99u,199u,397u};

// 64×64 → (hi,lo) unsigned, via 32-bit limbs (no 128-bit type on Metal).
static inline void bs_umul64_128(ulong a, ulong b, thread ulong& hi, thread ulong& lo) {
    ulong al = a & 0xffffffffUL, ah = a >> 32;
    ulong bl = b & 0xffffffffUL, bh = b >> 32;
    ulong ll = al * bl;
    ulong lh = al * bh;
    ulong hl = ah * bl;
    ulong hh = ah * bh;
    ulong mid = (ll >> 32) + (lh & 0xffffffffUL) + (hl & 0xffffffffUL);
    lo = (ll & 0xffffffffUL) | (mid << 32);
    hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
}

// (a*b) >> 40 with i128-arithmetic-shift (floor) semantics, result fits long.
static inline long bs_smul_shift40(long a, long b) {
    bool neg = (a < 0) ^ (b < 0);
    ulong ua = (ulong)(a < 0 ? -a : a);
    ulong ub = (ulong)(b < 0 ? -b : b);
    ulong hi, lo;
    bs_umul64_128(ua, ub, hi, lo);
    // logical >>40 of the 128-bit magnitude (40 < 64).
    ulong shifted = (lo >> BS_ACKLAM_FB) | (hi << (64 - BS_ACKLAM_FB));
    if (!neg) {
        return (long)shifted;
    }
    // floor: a negative value with any dropped low bits rounds down by one.
    ulong dropped = lo & ((1UL << BS_ACKLAM_FB) - 1UL);
    ulong mag = (dropped != 0UL) ? shifted + 1UL : shifted;
    return -(long)mag;
}

// State→rank hash, byte-identical to strand_quant::codebook::hash_state (usize is
// 64-bit on the host; ulong wraps mod 2^64 exactly as wrapping_mul does).
static inline ulong bs_hash_state(ulong s, uint l_bits) {
    ulong mask = (1UL << l_bits) - 1UL;
    uint r = max(l_bits / 2u, 1u);
    ulong h = s & mask;
    h = (h ^ (h >> r)) & mask;
    h = (h * 0x2545F4914F6CDD1DUL) & mask;
    h = (h ^ (h >> r)) & mask;
    h = (h * 0x9E3779B97F4A7C15UL) & mask;
    return h & mask;
}

// Acklam central rational for rank r (Q12), pure integer. Mirrors the CPU
// acklam_central_q12 exactly.
static inline int bs_acklam_central_q12(uint r, uint l_bits) {
    uint l_plus_1 = l_bits + 1u;
    long q_num = (long)(2u * r + 1u) - ((long)1 << l_bits); // (2r+1) - 2^L
    long q_q = (q_num << BS_ACKLAM_FB) >> l_plus_1;          // q in Q40, fits long
    long r2_q = bs_smul_shift40(q_q, q_q);                   // q^2 in Q40
    long one_q = (long)1 << BS_ACKLAM_FB;

    long num = BS_ACKLAM_A_Q[0];
    for (uint i = 1; i < 6u; ++i) num = bs_smul_shift40(num, r2_q) + BS_ACKLAM_A_Q[i];
    num = bs_smul_shift40(num, q_q);

    long den = BS_ACKLAM_B_Q[0];
    for (uint i = 1; i < 5u; ++i) den = bs_smul_shift40(den, r2_q) + BS_ACKLAM_B_Q[i];
    den = bs_smul_shift40(den, r2_q);
    den += one_q;

    // round((num/den) * 2^12), half away from zero.
    long nn = num << BS_QUANTILE_SHIFT;
    long q12 = ((nn >= 0) == (den > 0)) ? (nn + den / 2) / den : (nn - den / 2) / den;
    return clamp((int)q12, -BS_QUANTILE_CLAMP_Q12, BS_QUANTILE_CLAMP_Q12);
}

// Quantile (Q12) at rank r: central via integer Acklam, tail via the stored
// antisymmetric prefix. `tail_q12` holds the t left-tail values [q(0)..q(t)).
static inline int bs_quantile_q12_computed(uint r, uint l_bits,
                                           device const int* tail_q12, uint t) {
    uint n = 1u << l_bits;
    if (r < t) {
        return tail_q12[r];          // left tail, verbatim
    } else if (r >= n - t) {
        return -tail_q12[n - 1u - r]; // right tail = -left prefix
    }
    return bs_acklam_central_q12(r, l_bits);
}

// State-indexed codebook value (Q12) — the drop-in for sh_lut[state].
static inline int bs_qcb(uint state, uint l_bits, device const int* tail_q12, uint t) {
    uint rank = (uint)bs_hash_state((ulong)state, l_bits);
    return bs_quantile_q12_computed(rank, l_bits, tail_q12, t);
}

kernel void strand_bitslice_decode(
    device   const uchar*          w_bits   [[buffer(0)]],
    device         int*            out_q12  [[buffer(1)]],
    device   const BitsliceEntry*  tbl      [[buffer(2)]],
    constant       uint&           n_blocks [[buffer(3)]],
    constant       uint&           k_bits   [[buffer(4)]],
    constant       uint&           l_bits   [[buffer(5)]],
    device   const int*            lut_q12  [[buffer(6)]],
    threadgroup    int*            sh_lut   [[threadgroup(0)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint gidx [[thread_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]])
{
    
    uint lut_n = 1u << l_bits;
    for (uint s = tid; s < lut_n; s += tgs) sh_lut[s] = lut_q12[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gidx >= n_blocks) return;

    uint state_mask = lut_n - 1u;              
    uint input_mask = (1u << k_bits) - 1u;     

    device const BitsliceEntry* e = &tbl[gidx];

    uint  state    = e->init_state & state_mask;
    uint  bitpos   = e->bit_offset;
    uint  n        = e->n;
    uint  obase    = e->out_off;
    uint  word_idx = bitpos >> 5;
    uint  bit_in_w = bitpos & 31u;
    
    ulong acc      = (ulong)(bs_load_u32_le(w_bits, word_idx) >> bit_in_w);
    uint  have     = 32u - bit_in_w;

    for (uint j = 0; j < n; ++j) {
        if (have < k_bits) {                       
            ulong nxt = (ulong)bs_load_u32_le(w_bits, ++word_idx);
            acc |= nxt << have;
            have += 32u;
        }
        uint sym = (uint)acc & input_mask;         
        acc >>= k_bits;
        have -= k_bits;

        state = ((state << k_bits) | sym) & state_mask;
        int q  = sh_lut[state];
        uint sb = j >> 5;                          
        int es = e->eff[sb];

        int w  = (int)(((long)es * (long)q) >> 16) + e->off[sb];

        out_q12[obase + j] = w;                    
    }
}

// Computed-codebook scalar decode: identical recurrence to strand_bitslice_decode
// but the per-state codebook value is COMPUTED inline (bs_qcb) instead of gathered
// from a staged 2^L LUT. No lut_q12 buffer, no sh_lut threadgroup, no staging
// barrier — only a tiny tail prefix (tail_q12, ≤ tail_len entries) is carried.
// Output is byte-for-byte identical to strand_bitslice_decode on the frozen
// Gaussian codebook (Variant A; gated vs decode_tensor_fixed and the NEON twin).
kernel void strand_bitslice_decode_computed(
    device   const uchar*          w_bits   [[buffer(0)]],
    device         int*            out_q12  [[buffer(1)]],
    device   const BitsliceEntry*  tbl      [[buffer(2)]],
    constant       uint&           n_blocks [[buffer(3)]],
    constant       uint&           k_bits   [[buffer(4)]],
    constant       uint&           l_bits   [[buffer(5)]],
    device   const int*            tail_q12 [[buffer(6)]],
    constant       uint&           tail_len [[buffer(7)]],
    uint gidx [[thread_position_in_grid]])
{
    if (gidx >= n_blocks) return;

    uint state_mask = (1u << l_bits) - 1u;
    uint input_mask = (1u << k_bits) - 1u;

    device const BitsliceEntry* e = &tbl[gidx];

    uint  state    = e->init_state & state_mask;
    uint  bitpos   = e->bit_offset;
    uint  n        = e->n;
    uint  obase    = e->out_off;
    uint  word_idx = bitpos >> 5;
    uint  bit_in_w = bitpos & 31u;

    ulong acc      = (ulong)(bs_load_u32_le(w_bits, word_idx) >> bit_in_w);
    uint  have     = 32u - bit_in_w;

    for (uint j = 0; j < n; ++j) {
        if (have < k_bits) {
            ulong nxt = (ulong)bs_load_u32_le(w_bits, ++word_idx);
            acc |= nxt << have;
            have += 32u;
        }
        uint sym = (uint)acc & input_mask;
        acc >>= k_bits;
        have -= k_bits;

        state = ((state << k_bits) | sym) & state_mask;
        int q  = bs_qcb(state, l_bits, tail_q12, tail_len);
        uint sb = j >> 5;
        int es = e->eff[sb];

        int w  = (int)(((long)es * (long)q) >> 16) + e->off[sb];

        out_q12[obase + j] = w;
    }
}

kernel void strand_bitslice_entry_sizeof(device uint* out [[buffer(0)]]) {
    out[0] = (uint)sizeof(BitsliceEntry);
}

// ---- B.7 vector trellis (d=2) -------------------------------------------------
// Identical recurrence to strand_bitslice_decode, except: one symbol advances the
// state and emits D outputs from sh_lut[state*D + j]; n_steps = ceil(n/D); the LUT
// staged in threadgroup memory is 2^L * D ints. D is hardcoded to 2 here.
#define STRAND_VEC_D 2u

kernel void strand_bitslice_decode_vec(
    device   const uchar*          w_bits   [[buffer(0)]],
    device         int*            out_q12  [[buffer(1)]],
    device   const BitsliceEntry*  tbl      [[buffer(2)]],
    constant       uint&           n_blocks [[buffer(3)]],
    constant       uint&           k_bits   [[buffer(4)]],
    constant       uint&           l_bits   [[buffer(5)]],
    device   const int*            lut_q12  [[buffer(6)]],
    threadgroup    int*            sh_lut   [[threadgroup(0)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint gidx [[thread_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]])
{
    const uint D = STRAND_VEC_D;
    uint lut_n = (1u << l_bits) * D;                 // 2^L * D entries
    for (uint s = tid; s < lut_n; s += tgs) sh_lut[s] = lut_q12[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gidx >= n_blocks) return;

    uint state_mask = (1u << l_bits) - 1u;
    uint input_mask = (1u << k_bits) - 1u;

    device const BitsliceEntry* e = &tbl[gidx];

    uint  state    = e->init_state & state_mask;
    uint  bitpos   = e->bit_offset;
    uint  n        = e->n;
    uint  obase    = e->out_off;
    uint  word_idx = bitpos >> 5;
    uint  bit_in_w = bitpos & 31u;

    ulong acc      = (ulong)(bs_load_u32_le(w_bits, word_idx) >> bit_in_w);
    uint  have     = 32u - bit_in_w;

    uint  n_steps  = (n + D - 1u) / D;
    uint  produced = 0u;
    for (uint t = 0; t < n_steps; ++t) {
        if (have < k_bits) {
            ulong nxt = (ulong)bs_load_u32_le(w_bits, ++word_idx);
            acc |= nxt << have;
            have += 32u;
        }
        uint sym = (uint)acc & input_mask;
        acc >>= k_bits;
        have -= k_bits;

        state = ((state << k_bits) | sym) & state_mask;
        uint base = state * D;

        uint emit = min(D, n - produced);
        for (uint j = 0; j < emit; ++j) {
            uint i  = produced + j;
            int  q  = sh_lut[base + j];
            uint sb = i >> 5;
            int  es = e->eff[sb];
            int  w  = (int)(((long)es * (long)q) >> 16) + e->off[sb];
            out_q12[obase + i] = w;
        }
        produced += emit;
    }
}

kernel void strand_bitslice_gemv_partials(
    device   const uchar*          w_bits   [[buffer(0)]],
    device   const float*          x        [[buffer(1)]],   
    device         float*          partials [[buffer(2)]],   
    device   const BitsliceEntry*  tbl      [[buffer(3)]],
    constant       uint&           n_blocks [[buffer(4)]],
    constant       uint&           cols     [[buffer(5)]],
    constant       uint&           k_bits   [[buffer(6)]],
    constant       uint&           l_bits   [[buffer(7)]],
    device   const int*            lut_q12  [[buffer(8)]],
    threadgroup    int*            sh_lut   [[threadgroup(0)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint gidx [[thread_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]])
{
    uint lut_n = 1u << l_bits;
    for (uint s = tid; s < lut_n; s += tgs) sh_lut[s] = lut_q12[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gidx >= n_blocks) return;

    uint state_mask = lut_n - 1u;
    uint input_mask = (1u << k_bits) - 1u;
    const float Q12_TO_F32 = 1.0f / 4096.0f;

    device const BitsliceEntry* e = &tbl[gidx];
    uint  state    = e->init_state & state_mask;
    uint  n        = e->n;
    uint  col0     = e->out_off % cols;          
    uint  word_idx = e->bit_offset >> 5;
    uint  bit_in_w = e->bit_offset & 31u;
    ulong acc      = (ulong)(bs_load_u32_le(w_bits, word_idx) >> bit_in_w);
    uint  have     = 32u - bit_in_w;

    float partial = 0.0f;
    for (uint j = 0; j < n; ++j) {
        if (have < k_bits) {
            ulong nxt = (ulong)bs_load_u32_le(w_bits, ++word_idx);
            acc |= nxt << have;
            have += 32u;
        }
        uint sym = (uint)acc & input_mask;
        acc >>= k_bits;
        have -= k_bits;
        state = ((state << k_bits) | sym) & state_mask;
        int q  = sh_lut[state];
        uint sb = j >> 5;
        int w  = (int)(((long)e->eff[sb] * (long)q) >> 16) + e->off[sb];
        partial += (float)w * Q12_TO_F32 * x[col0 + j];
    }
    partials[gidx] = partial;
}

// B.7 vector (d=2) fused B=1 partials: same ALU-bound inner loop as the scalar
// gemv_partials, but one symbol emits D weights, each MAC'd against the matching x.
kernel void strand_bitslice_gemv_partials_vec(
    device   const uchar*          w_bits   [[buffer(0)]],
    device   const float*          x        [[buffer(1)]],
    device         float*          partials [[buffer(2)]],
    device   const BitsliceEntry*  tbl      [[buffer(3)]],
    constant       uint&           n_blocks [[buffer(4)]],
    constant       uint&           cols     [[buffer(5)]],
    constant       uint&           k_bits   [[buffer(6)]],
    constant       uint&           l_bits   [[buffer(7)]],
    device   const int*            lut_q12  [[buffer(8)]],
    threadgroup    int*            sh_lut   [[threadgroup(0)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint gidx [[thread_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]])
{
    const uint D = STRAND_VEC_D;
    uint lut_n = (1u << l_bits) * D;
    for (uint s = tid; s < lut_n; s += tgs) sh_lut[s] = lut_q12[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gidx >= n_blocks) return;

    uint state_mask = (1u << l_bits) - 1u;
    uint input_mask = (1u << k_bits) - 1u;
    const float Q12_TO_F32 = 1.0f / 4096.0f;

    device const BitsliceEntry* e = &tbl[gidx];
    uint  state    = e->init_state & state_mask;
    uint  n        = e->n;
    uint  col0     = e->out_off % cols;
    uint  word_idx = e->bit_offset >> 5;
    uint  bit_in_w = e->bit_offset & 31u;
    ulong acc      = (ulong)(bs_load_u32_le(w_bits, word_idx) >> bit_in_w);
    uint  have     = 32u - bit_in_w;

    float partial  = 0.0f;
    uint  n_steps  = (n + D - 1u) / D;
    uint  produced = 0u;
    for (uint t = 0; t < n_steps; ++t) {
        if (have < k_bits) {
            ulong nxt = (ulong)bs_load_u32_le(w_bits, ++word_idx);
            acc |= nxt << have;
            have += 32u;
        }
        uint sym = (uint)acc & input_mask;
        acc >>= k_bits;
        have -= k_bits;
        state = ((state << k_bits) | sym) & state_mask;
        uint base = state * D;

        uint emit = min(D, n - produced);
        for (uint j = 0; j < emit; ++j) {
            uint i  = produced + j;
            int  q  = sh_lut[base + j];
            uint sb = i >> 5;
            int  w  = (int)(((long)e->eff[sb] * (long)q) >> 16) + e->off[sb];
            partial += (float)w * Q12_TO_F32 * x[col0 + i];
        }
        produced += emit;
    }
    partials[gidx] = partial;
}

kernel void strand_bitslice_reduce_rows(
    device const float* partials [[buffer(0)]],
    device       float* y        [[buffer(1)]],
    constant     uint&  rows     [[buffer(2)]],
    constant     uint&  bpr      [[buffer(3)]],
    uint gidx [[thread_position_in_grid]])
{
    if (gidx >= rows) return;
    float acc = 0.0f;
    uint base = gidx * bpr;
    for (uint b = 0; b < bpr; ++b) acc += partials[base + b];
    y[gidx] = acc;
}

// Residual two-part serving (HAWKING_TQ_RESIDUAL): identical to
// strand_bitslice_reduce_rows but ACCUMULATES into y instead of overwriting it
// (`y[gidx] += acc`). The serving recipe runs the base STRAND pass through the
// plain reduce (which seeds/overwrites y) and the residual STRAND pass through
// this accumulate reduce, so the final output is
//   y = bitslice_gemv(base, x) + bitslice_gemv(residual, x)
// = (decode(base) + decode(residual)) · x — the decoded-sum the residual bake
// (W ≈ STRAND_b1(W) + STRAND_b2(W − STRAND_b1(W))) targets, with the two passes
// kept COMPRESSED in RAM and summed at GEMV time. y MUST be pre-seeded by the
// base pass (or zeroed) before this kernel runs.
kernel void strand_bitslice_reduce_rows_accum(
    device const float* partials [[buffer(0)]],
    device       float* y        [[buffer(1)]],
    constant     uint&  rows     [[buffer(2)]],
    constant     uint&  bpr      [[buffer(3)]],
    uint gidx [[thread_position_in_grid]])
{
    if (gidx >= rows) return;
    float acc = 0.0f;
    uint base = gidx * bpr;
    for (uint b = 0; b < bpr; ++b) acc += partials[base + b];
    y[gidx] += acc;
}

template <uint B>
static inline void bs_gemm_partials_impl(
    device   const uchar*          w_bits,
    device   const float*          xt,        
    device         float*          partials,  
    device   const BitsliceEntry*  tbl,
    uint n_blocks, uint cols, uint k_bits, uint l_bits,
    device   const int*            lut_q12,
    threadgroup    int*            sh_lut,
    uint tid, uint gidx, uint tgs)
{
    uint lut_n = 1u << l_bits;
    for (uint s = tid; s < lut_n; s += tgs) sh_lut[s] = lut_q12[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gidx >= n_blocks) return;

    uint state_mask = lut_n - 1u;
    uint input_mask = (1u << k_bits) - 1u;
    const float Q12_TO_F32 = 1.0f / 4096.0f;

    device const BitsliceEntry* e = &tbl[gidx];
    uint  state    = e->init_state & state_mask;
    uint  n        = e->n;
    uint  col0     = e->out_off % cols;          
    uint  word_idx = e->bit_offset >> 5;
    uint  bit_in_w = e->bit_offset & 31u;
    ulong acc      = (ulong)(bs_load_u32_le(w_bits, word_idx) >> bit_in_w);
    uint  have     = 32u - bit_in_w;

    float pacc[B];
    for (uint b = 0; b < B; ++b) pacc[b] = 0.0f;

    for (uint j = 0; j < n; ++j) {
        if (have < k_bits) {
            ulong nxt = (ulong)bs_load_u32_le(w_bits, ++word_idx);
            acc |= nxt << have;
            have += 32u;
        }
        uint sym = (uint)acc & input_mask;
        acc >>= k_bits;
        have -= k_bits;
        state = ((state << k_bits) | sym) & state_mask;
        int q  = sh_lut[state];
        uint sb = j >> 5;
        int w  = (int)(((long)e->eff[sb] * (long)q) >> 16) + e->off[sb];
        float wf = (float)w * Q12_TO_F32;
        device const float* xp = xt + (ulong)(col0 + j) * B;
        for (uint b = 0; b < B; ++b) pacc[b] += wf * xp[b];
    }
    device float* pp = partials + (ulong)gidx * B;
    for (uint b = 0; b < B; ++b) pp[b] = pacc[b];
}

#define BS_GEMM_KERNEL(NAME, B)                                                          \
kernel void NAME(                                                                        \
    device   const uchar*          w_bits   [[buffer(0)]],                               \
    device   const float*          xt       [[buffer(1)]],                               \
    device         float*          partials [[buffer(2)]],                               \
    device   const BitsliceEntry*  tbl      [[buffer(3)]],                               \
    constant       uint&           n_blocks [[buffer(4)]],                               \
    constant       uint&           cols     [[buffer(5)]],                               \
    constant       uint&           k_bits   [[buffer(6)]],                               \
    constant       uint&           l_bits   [[buffer(7)]],                               \
    device   const int*            lut_q12  [[buffer(8)]],                               \
    threadgroup    int*            sh_lut   [[threadgroup(0)]],                          \
    uint tid  [[thread_position_in_threadgroup]],                                       \
    uint gidx [[thread_position_in_grid]],                                              \
    uint tgs  [[threads_per_threadgroup]])                                              \
{                                                                                        \
    bs_gemm_partials_impl<B>(w_bits, xt, partials, tbl, n_blocks, cols, k_bits, l_bits, \
                             lut_q12, sh_lut, tid, gidx, tgs);                          \
}

BS_GEMM_KERNEL(strand_bitslice_gemm_partials_b4,  4)
BS_GEMM_KERNEL(strand_bitslice_gemm_partials_b16, 16)
BS_GEMM_KERNEL(strand_bitslice_gemm_partials_b64, 64)

kernel void strand_bitslice_reduce_rows_gemm(
    device const float* partials [[buffer(0)]],
    device       float* y        [[buffer(1)]],
    constant     uint&  rows     [[buffer(2)]],
    constant     uint&  bpr      [[buffer(3)]],
    constant     uint&  batch    [[buffer(4)]],
    uint gidx [[thread_position_in_grid]])
{
    if (gidx >= rows * batch) return;
    uint r = gidx / batch;
    uint b = gidx % batch;
    float acc = 0.0f;
    ulong base = (ulong)r * bpr * batch + b;
    for (uint blk = 0; blk < bpr; ++blk) acc += partials[base + (ulong)blk * batch];
    y[(ulong)r * batch + b] = acc;
}

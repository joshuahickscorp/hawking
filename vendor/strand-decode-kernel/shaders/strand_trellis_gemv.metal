
#include <metal_stdlib>
using namespace metal;

struct BlockEntry {
    uint  bit_offset;   
    uint  init_state;   
    int   scale_q;      
    int   eff[8];       
    ushort n;           
    ushort d;           
    uint  _pad;         
};

static inline uint load_u32_le(device const uchar* p, uint widx) {
    uint b = widx << 2;
    return (uint)p[b]
         | ((uint)p[b + 1] << 8)
         | ((uint)p[b + 2] << 16)
         | ((uint)p[b + 3] << 24);
}

// ─── Computed codebook (Variant A) — inline integer Acklam, no LUT gather ─────
// Byte-for-byte twin of strand_quant::codebook::qcb in pure integer arithmetic.
// See strand_bitslice.metal for the full derivation; this is the same code copied
// into this independently-compiled shader library.
#define BS_ACKLAM_FB 40
#define BS_QUANTILE_SHIFT 12
#define BS_QUANTILE_CLAMP_Q12 (6 * (1 << BS_QUANTILE_SHIFT))
constant long BS_ACKLAM_A_Q[6] = {
    -43647126486026L, 242932804329501L, -303386605671354L,
    152125956971009L, -33716302037132L, 2756066937579L,
};
constant long BS_ACKLAM_B_Q[5] = {
    -59897104064522L, 177665506509332L, -171192838788807L,
    73448819171239L, -14602263792188L,
};
static inline void gv_umul64_128(ulong a, ulong b, thread ulong& hi, thread ulong& lo) {
    ulong al = a & 0xffffffffUL, ah = a >> 32;
    ulong bl = b & 0xffffffffUL, bh = b >> 32;
    ulong ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    ulong mid = (ll >> 32) + (lh & 0xffffffffUL) + (hl & 0xffffffffUL);
    lo = (ll & 0xffffffffUL) | (mid << 32);
    hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
}
static inline long gv_smul_shift40(long a, long b) {
    bool neg = (a < 0) ^ (b < 0);
    ulong ua = (ulong)(a < 0 ? -a : a), ub = (ulong)(b < 0 ? -b : b);
    ulong hi, lo; gv_umul64_128(ua, ub, hi, lo);
    ulong shifted = (lo >> BS_ACKLAM_FB) | (hi << (64 - BS_ACKLAM_FB));
    if (!neg) return (long)shifted;
    ulong dropped = lo & ((1UL << BS_ACKLAM_FB) - 1UL);
    return -(long)((dropped != 0UL) ? shifted + 1UL : shifted);
}
static inline ulong gv_hash_state(ulong s, uint l_bits) {
    ulong mask = (1UL << l_bits) - 1UL;
    uint r = max(l_bits / 2u, 1u);
    ulong h = s & mask;
    h = (h ^ (h >> r)) & mask;
    h = (h * 0x2545F4914F6CDD1DUL) & mask;
    h = (h ^ (h >> r)) & mask;
    h = (h * 0x9E3779B97F4A7C15UL) & mask;
    return h & mask;
}
static inline int gv_acklam_central_q12(uint rk, uint l_bits) {
    uint l_plus_1 = l_bits + 1u;
    long q_num = (long)(2u * rk + 1u) - ((long)1 << l_bits);
    long q_q = (q_num << BS_ACKLAM_FB) >> l_plus_1;
    long r2_q = gv_smul_shift40(q_q, q_q);
    long one_q = (long)1 << BS_ACKLAM_FB;
    long num = BS_ACKLAM_A_Q[0];
    for (uint i = 1; i < 6u; ++i) num = gv_smul_shift40(num, r2_q) + BS_ACKLAM_A_Q[i];
    num = gv_smul_shift40(num, q_q);
    long den = BS_ACKLAM_B_Q[0];
    for (uint i = 1; i < 5u; ++i) den = gv_smul_shift40(den, r2_q) + BS_ACKLAM_B_Q[i];
    den = gv_smul_shift40(den, r2_q);
    den += one_q;
    long nn = num << BS_QUANTILE_SHIFT;
    long q12 = ((nn >= 0) == (den > 0)) ? (nn + den / 2) / den : (nn - den / 2) / den;
    return clamp((int)q12, -BS_QUANTILE_CLAMP_Q12, BS_QUANTILE_CLAMP_Q12);
}
static inline int gv_qcb(uint state, uint l_bits, device const int* tail_q12, uint t) {
    uint rk = (uint)gv_hash_state((ulong)state, l_bits);
    uint n = 1u << l_bits;
    if (rk < t) return tail_q12[rk];
    if (rk >= n - t) return -tail_q12[n - 1u - rk];
    return gv_acklam_central_q12(rk, l_bits);
}

kernel void strand_trellis_gemv(
    device   const uchar*       w_bits  [[buffer(0)]],
    device   const float*       x_rht   [[buffer(1)]],
    device         float*       y       [[buffer(2)]],
    constant       uint&        rows    [[buffer(3)]],
    constant       uint&        cols    [[buffer(4)]],
    device   const BlockEntry*  tbl     [[buffer(5)]],
    constant       uint&        k_bits  [[buffer(6)]],
    constant       uint&        l_bits  [[buffer(7)]],
    device   const int*         lut_q12 [[buffer(8)]],
    threadgroup    int*         sh_lut  [[threadgroup(0)]],
    threadgroup    float*       sh_red  [[threadgroup(1)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]],   
    uint tgs  [[threads_per_threadgroup]])         
{
    if (gid >= rows) return;

    uint bpr = cols / 256u;                         

    uint lut_n = 1u << l_bits;                       
    for (uint s = tid; s < lut_n; s += tgs) sh_lut[s] = lut_q12[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint state_mask = lut_n - 1u;                    
    uint input_mask = (1u << k_bits) - 1u;           
    const float Q12_TO_F32 = 1.0f / 4096.0f;         

    float partial = 0.0f;

    for (uint b = tid; b < bpr; b += tgs) {
        device const BlockEntry* e = &tbl[(uint64_t)gid * bpr + b];

        uint state   = e->init_state & state_mask;   
        uint bitpos  = e->bit_offset;                
        uint col0    = b * 256u;                      
        uint n       = (uint)e->n;

        uint  word_idx = bitpos >> 5;
        uint  bit_in_w = bitpos & 31u;
        ulong acc      = (ulong)(load_u32_le(w_bits, word_idx) >> bit_in_w);
        uint  have     = 32u - bit_in_w;

        for (uint j = 0; j < n; ++j) {
            if (have < k_bits) {                      
                ulong nxt = (ulong)load_u32_le(w_bits, ++word_idx);
                acc |= nxt << have;                   
                have += 32u;
            }
            uint sym = (uint)acc & input_mask;        
            acc >>= k_bits;
            have -= k_bits;

            state = ((state << k_bits) | sym) & state_mask;   
            int q  = sh_lut[state];                            
            int es = e->eff[j >> 5];                           

            int w  = (int)(((long)es * (long)q) >> 16);

            partial += (float)w * Q12_TO_F32 * x_rht[col0 + j];
        }
    }

    sh_red[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) sh_red[tid] += sh_red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) y[gid] = sh_red[0];
}

// Computed-codebook fused GEMV: same recurrence as strand_trellis_gemv but the
// per-state codebook value is computed inline (gv_qcb) — no lut_q12 buffer, no
// sh_lut staging. Carries only the tiny tail prefix. The integer w (Q12) is
// byte-identical to the gather kernel; the float MAC/reduction is unchanged.
kernel void strand_trellis_gemv_computed(
    device   const uchar*       w_bits   [[buffer(0)]],
    device   const float*       x_rht    [[buffer(1)]],
    device         float*       y        [[buffer(2)]],
    constant       uint&        rows     [[buffer(3)]],
    constant       uint&        cols     [[buffer(4)]],
    device   const BlockEntry*  tbl      [[buffer(5)]],
    constant       uint&        k_bits   [[buffer(6)]],
    constant       uint&        l_bits   [[buffer(7)]],
    device   const int*         tail_q12 [[buffer(8)]],
    constant       uint&        tail_len [[buffer(9)]],
    threadgroup    float*       sh_red   [[threadgroup(0)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]])
{
    if (gid >= rows) return;

    uint bpr = cols / 256u;
    uint state_mask = (1u << l_bits) - 1u;
    uint input_mask = (1u << k_bits) - 1u;
    const float Q12_TO_F32 = 1.0f / 4096.0f;

    float partial = 0.0f;
    for (uint b = tid; b < bpr; b += tgs) {
        device const BlockEntry* e = &tbl[(uint64_t)gid * bpr + b];
        uint state   = e->init_state & state_mask;
        uint bitpos  = e->bit_offset;
        uint col0    = b * 256u;
        uint n       = (uint)e->n;
        uint  word_idx = bitpos >> 5;
        uint  bit_in_w = bitpos & 31u;
        ulong acc      = (ulong)(load_u32_le(w_bits, word_idx) >> bit_in_w);
        uint  have     = 32u - bit_in_w;
        for (uint j = 0; j < n; ++j) {
            if (have < k_bits) {
                ulong nxt = (ulong)load_u32_le(w_bits, ++word_idx);
                acc |= nxt << have;
                have += 32u;
            }
            uint sym = (uint)acc & input_mask;
            acc >>= k_bits; have -= k_bits;
            state = ((state << k_bits) | sym) & state_mask;
            int q  = gv_qcb(state, l_bits, tail_q12, tail_len);
            int es = e->eff[j >> 5];
            int w  = (int)(((long)es * (long)q) >> 16);
            partial += (float)w * Q12_TO_F32 * x_rht[col0 + j];
        }
    }

    sh_red[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) sh_red[tid] += sh_red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) y[gid] = sh_red[0];
}

kernel void strand_trellis_gemv_predec(
    device   const uchar*       w_bits  [[buffer(0)]],
    device   const float*       x_rht   [[buffer(1)]],
    device         float*       y       [[buffer(2)]],
    constant       uint&        rows    [[buffer(3)]],
    constant       uint&        cols    [[buffer(4)]],
    device   const BlockEntry*  tbl     [[buffer(5)]],
    constant       uint&        k_bits  [[buffer(6)]],
    constant       uint&        l_bits  [[buffer(7)]],
    device   const int*         lut_q12 [[buffer(8)]],
    threadgroup    int*         sh_lut  [[threadgroup(0)]],  
    threadgroup    float*       sh_red  [[threadgroup(1)]],  
    threadgroup    int*         sh_wq12 [[threadgroup(2)]],  
    uint tid  [[thread_position_in_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]])
{
    if (gid >= rows) return;

    uint bpr = cols / 256u;

    uint lut_n = 1u << l_bits;
    for (uint s = tid; s < lut_n; s += tgs) sh_lut[s] = lut_q12[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint state_mask = lut_n - 1u;
    uint input_mask = (1u << k_bits) - 1u;
    const float Q12_TO_F32 = 1.0f / 4096.0f;

    for (uint b = tid; b < bpr; b += tgs) {
        device const BlockEntry* e = &tbl[(uint64_t)gid * bpr + b];
        uint state   = e->init_state & state_mask;
        uint col0    = b * 256u;
        uint n       = (uint)e->n;
        uint bitpos  = e->bit_offset;
        uint  word_idx = bitpos >> 5;
        uint  bit_in_w = bitpos & 31u;
        ulong acc      = (ulong)(load_u32_le(w_bits, word_idx) >> bit_in_w);
        uint  have     = 32u - bit_in_w;
        for (uint j = 0; j < n; ++j) {
            if (have < k_bits) {
                ulong nxt = (ulong)load_u32_le(w_bits, ++word_idx);
                acc |= nxt << have;
                have += 32u;
            }
            uint sym = (uint)acc & input_mask;
            acc >>= k_bits; have -= k_bits;
            state = ((state << k_bits) | sym) & state_mask;
            int q  = sh_lut[state];
            int es = e->eff[j >> 5];
            sh_wq12[col0 + j] = (int)(((long)es * (long)q) >> 16);   
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float partial = 0.0f;
    for (uint i = tid; i < cols; i += tgs) {
        partial += (float)sh_wq12[i] * Q12_TO_F32 * x_rht[i];
    }

    sh_red[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) sh_red[tid] += sh_red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) y[gid] = sh_red[0];
}

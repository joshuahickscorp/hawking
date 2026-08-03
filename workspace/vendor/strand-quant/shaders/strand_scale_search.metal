
#include <metal_stdlib>
using namespace metal;

constant float Q12_INV = 0x1p-12f;
constant float S16_INV = 0x1p-16f;

struct SearchParams {
    uint block_len;     
    uint num_states;    
    uint k_bits;        
    uint n_sub;         
    uint sub_size;      
    uint max_block_len; 
    uint n_sub_max;     
    uint adaptive;      
    uint affine_min;    
    uint _p0;
    uint _p1;
    uint _p2;
};

static inline int eff_scale_q_m(int scale_q, uint code) {
    const long mult = (long)(code & 0x3Fu) + 1;   
    return (int)(((long)scale_q * mult) >> 6);
}

static inline int eff_min_q_m(int min_base_q, uint code) {
    const long mag = (long)(code & 0x1Fu);        
    if (mag == 0) { return 0; }
    const long base = (long)abs(min_base_q);
    const long s = (code & 0x20u) ? (base * mag) : -(base * mag);
    return (int)(s / 31);
}

static inline int round_pos_to_i32(float x) {
    if (!isfinite(x) || x >= 2147483648.0f) { return 2147483647; }
    const float xf = floor(x);
    const long r = (long)xf + ((x - xf >= 0.5f) ? 1 : 0);
    return (int)min(r, (long)2147483647);
}

static float greedy_mse_off(threadgroup const float* w, uint n, float scale,
                            float off, device const int* lut, uint k, uint mask) {
    uint state = 0u;
    float acc = 0.0f;
    const uint n_in = 1u << k;
    for (uint i = 0u; i < n; i++) {
        const float target = w[i];
        float best = INFINITY;
        uint best_in = 0u;
        for (uint inp = 0u; inp < n_in; inp++) {
            const uint ns = ((state << k) | inp) & mask;
            const float q  = (float)lut[ns] * Q12_INV;
            const float t1 = scale * q;     
            const float lv = t1 + off;      
            const float d  = target - lv;   
            const float e  = d * d;
            if (e < best) { best = e; best_in = inp; }
        }
        state = ((state << k) | best_in) & mask;
        acc += best;
    }
    return acc;
}

kernel void scale_search_block(
    device const float*      weights   [[ buffer(0) ]],
    device const int*        lut       [[ buffer(1) ]],
    device float*            levels    [[ buffer(2) ]],
    constant SearchParams*   params    [[ buffer(3) ]],
    device int*              scale_out [[ buffer(4) ]],
    device uchar*            mult_out  [[ buffer(5) ]],
    device int*              minb_out  [[ buffer(6) ]],
    device uchar*            minc_out  [[ buffer(7) ]],
    threadgroup float*       shf       [[ threadgroup(0) ]],
    threadgroup int*         shi       [[ threadgroup(1) ]],
    threadgroup float*       shw       [[ threadgroup(2) ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tgsz [[ threads_per_threadgroup ]],
    uint bid  [[ threadgroup_position_in_grid ]]
) {
    constant SearchParams& p = params[bid];
    const uint blen  = p.block_len;
    const uint k     = p.k_bits;
    const uint mask  = p.num_states - 1u;
    const uint n_sub = p.n_sub;
    const uint nsm   = p.n_sub_max;

    const uint MEANS = nsm * 64u;       
    const uint ABSM  = MEANS + nsm;     
    const uint SEED  = ABSM + 1u;       
    const uint MULT0 = 1u;              
    const uint CODE0 = 1u + nsm;        
    const uint MINB  = 1u + 2u * nsm;   
    const uint DEGEN = MINB + 1u;       

    device const float* wsrc = weights + (ulong)bid * p.max_block_len;
    for (uint i = tid; i < blen; i += tgsz) { shw[i] = wsrc[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float am = 0.0f;
        for (uint i = 0u; i < blen; i++) {
            const float a = fabs(shw[i]);
            if (a > am) { am = a; }
        }
        shf[ABSM] = am;
        float qmax = (float)lut[mask] * Q12_INV; 
        if (!(qmax > 0.0f)) { qmax = 1.0f; }
        shf[SEED] = am / qmax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float absmax = shf[ABSM];
    const float seed   = shf[SEED];

    const float MULTS[11] = {0.55f, 0.65f, 0.75f, 0.85f, 0.92f, 1.0f,
                             1.08f, 1.18f, 1.30f, 1.45f, 1.65f};
    if (absmax > 0.0f && tid < 11u) {
        const float s = seed * MULTS[tid];
        shf[tid] = greedy_mse_off(shw, blen, s, 0.0f, lut, k, mask);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        int sq = 0;
        if (absmax > 0.0f) {
            float best = INFINITY;
            float bs = seed;            
            for (uint t = 0u; t < 11u; t++) {
                if (shf[t] < best) { best = shf[t]; bs = seed * MULTS[t]; }
            }
            sq = round_pos_to_i32(bs * 65536.0f); 
        }
        shi[0] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int scale_q = shi[0];

    if (p.adaptive != 0u) {
        for (uint lane = tid; lane < n_sub * 64u; lane += tgsz) {
            const uint sb = lane >> 6;
            const uint c  = lane & 63u;
            const int es  = eff_scale_q_m(scale_q, c);
            float mse = INFINITY;       
            if (es != 0) {
                const float es_real = (float)es * S16_INV;
                const uint lo = sb * p.sub_size;
                const uint hi = min(lo + p.sub_size, blen);
                mse = greedy_mse_off(shw + lo, hi - lo, es_real, 0.0f, lut, k, mask);
            }
            shf[lane] = mse;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < n_sub) {
        uint best_c = 63u;              
        if (p.adaptive != 0u) {
            const uint lo = tid * p.sub_size;
            const uint hi = min(lo + p.sub_size, blen);
            bool all_zero = true;
            for (uint i = lo; i < hi; i++) {
                if (shw[i] != 0.0f) { all_zero = false; break; }
            }
            if (!all_zero) {
                float best = INFINITY;
                for (uint c = 0u; c < 64u; c++) {  
                    const float v = shf[(tid << 6) + c];
                    if (v < best) { best = v; best_c = c; }
                }
            }
        }
        shi[MULT0 + tid] = (int)best_c;
        mult_out[(ulong)bid * nsm + tid] = (uchar)best_c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int min_base_q = 0;
    if (p.affine_min != 0u) {
        if (tid < n_sub) {
            const uint lo = tid * p.sub_size;
            const uint hi = min(lo + p.sub_size, blen);
            float sum = 0.0f;           
            for (uint i = lo; i < hi; i++) { sum += shw[i]; }
            shf[MEANS + tid] = (hi > lo) ? (sum / (float)(hi - lo)) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float ba = 0.0f;
            for (uint sb = 0u; sb < n_sub; sb++) {
                const float a = fabs(shf[MEANS + sb]);
                if (a > ba) { ba = a; }
            }
            const bool degen = ba < 1e-12f;
            shi[MINB]  = degen ? 0 : round_pos_to_i32(ba * 4096.0f);
            shi[DEGEN] = degen ? 1 : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        min_base_q = shi[MINB];
        const bool degen = shi[DEGEN] != 0;
        if (!degen) {
            for (uint lane = tid; lane < n_sub * 32u; lane += tgsz) {
                const uint sb = lane >> 5;
                const uint j  = lane & 31u;
                const bool pos = shf[MEANS + sb] >= 0.0f;
                const uint code = pos ? (32u + j) : j;
                const int es = eff_scale_q_m(scale_q, (uint)shi[MULT0 + sb]);
                const float es_real = (float)es * S16_INV;  
                const float off = (float)eff_min_q_m(min_base_q, code) * Q12_INV;
                const uint lo = sb * p.sub_size;
                const uint hi = min(lo + p.sub_size, blen);
                shf[lane] = greedy_mse_off(shw + lo, hi - lo, es_real, off, lut, k, mask);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < n_sub) {
            uint best_code;
            if (degen) {
                best_code = 0u;         
            } else {
                const bool pos = shf[MEANS + tid] >= 0.0f;
                best_code = pos ? 32u : 0u;
                float best = INFINITY;
                for (uint j = 0u; j < 32u; j++) {  
                    const float v = shf[(tid << 5) + j];
                    if (v < best) { best = v; best_code = pos ? (32u + j) : j; }
                }
            }
            shi[CODE0 + tid] = (int)best_code;
            minc_out[(ulong)bid * nsm + tid] = (uchar)best_code;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint ns_total = n_sub * p.num_states;
    for (uint idx = tid; idx < ns_total; idx += tgsz) {
        const uint sb = idx / p.num_states;
        const uint s  = idx - sb * p.num_states;
        const int es  = eff_scale_q_m(scale_q, (uint)shi[MULT0 + sb]);
        const int rq  = (int)(((long)es * (long)lut[s]) >> 16);  
        const int off = (p.affine_min != 0u)
            ? eff_min_q_m(min_base_q, (uint)shi[CODE0 + sb]) : 0;
        levels[(ulong)bid * nsm * p.num_states + (ulong)sb * p.num_states + s]
            = (float)(rq + off) * Q12_INV;
    }

    if (tid == 0u) {
        scale_out[bid] = scale_q;
        minb_out[bid]  = min_base_q;
    }
}
